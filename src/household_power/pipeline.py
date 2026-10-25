"""End-to-end training, validation, refit, testing, and artifact generation pipeline."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from .data import RawDataStats, prepare_and_cache_hourly_data
from .evaluation import (
    ModelMetrics,
    calculate_metrics,
    compute_common_evaluation_mask,
    evaluate_high_load,
    metrics_to_dataframe,
)
from .features import FEATURE_SETS, build_features
from .models import (
    PersistenceBaseline,
    SeasonalNaiveBaseline,
    get_hyperparameter_candidates,
    get_model_pipeline,
)
from .plots import (
    plot_ablation_comparison,
    plot_mae_by_hour,
    plot_model_comparison,
    plot_one_week_forecast,
    plot_residuals_over_time,
)
from .splits import (
    DatasetSplits,
    compute_chronological_splits,
    get_expanding_cv_folds,
)

logger = logging.getLogger(__name__)


class PipelineRunner:
    """Orchestrates end-to-end pipeline execution with strict causal integrity."""

    def __init__(
        self,
        raw_dir: str | Path = "data/raw",
        processed_dir: str | Path = "data/processed",
        artifacts_dir: str | Path = "artifacts",
        force_download: bool = False,
        force_process: bool = False,
    ):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.artifacts_dir = Path(artifacts_dir)
        self.models_dir = self.artifacts_dir / "models"
        self.metrics_dir = self.artifacts_dir / "metrics"
        self.metadata_dir = self.artifacts_dir / "metadata"
        self.predictions_dir = self.artifacts_dir / "predictions"
        self.figures_dir = self.artifacts_dir / "figures"

        for p in [
            self.models_dir,
            self.metrics_dir,
            self.metadata_dir,
            self.predictions_dir,
            self.figures_dir,
        ]:
            p.mkdir(parents=True, exist_ok=True)

        self.force_download = force_download
        self.force_process = force_process

    def run(self) -> Dict[str, Any]:
        """Execute the complete training, validation, refit, test and evaluation cycle."""
        logger.info("================ STARTING ML-01 PIPELINE ================")

        # 1. Load and aggregate data
        hourly_df, data_stats = prepare_and_cache_hourly_data(
            raw_dir=self.raw_dir,
            processed_dir=self.processed_dir,
            force_download=self.force_download,
            force_process=self.force_process,
        )

        # 2. Build features
        X_all, y_all, valid_mask = build_features(hourly_df)

        # 3. Chronological splits
        splits_meta, train_mask, val_mask, test_mask, first85_mask = compute_chronological_splits(
            hourly_df, valid_mask
        )

        # 4. Expanding CV on initial 70% Train
        cv_folds = get_expanding_cv_folds(hourly_df, valid_mask, splits_meta.train_boundary, n_folds=3)
        candidate_configs = get_hyperparameter_candidates()

        logger.info("Executing Expanding Temporal Cross-Validation on Train Set...")
        cv_records = []
        best_configs: Dict[str, Dict[str, Any]] = {}

        for model_name, configs in candidate_configs.items():
            best_cfg = None
            best_cv_mae = float("inf")

            for cfg_idx, cfg in enumerate(configs):
                fold_maes = []
                fold_rmses = []
                fold_r2s = []
                fold_times = []
                fold_details = []

                for f_train_m, f_val_m, f_meta in cv_folds:
                    X_f_tr, y_f_tr = X_all.loc[f_train_m], y_all.loc[f_train_m]
                    X_f_va, y_f_va = X_all.loc[f_val_m], y_all.loc[f_val_m]

                    pipe = get_model_pipeline(model_name, **cfg)
                    t0 = time.time()
                    pipe.fit(X_f_tr, y_f_tr)
                    t_fit = time.time() - t0

                    preds = pipe.predict(X_f_va)
                    m = calculate_metrics(y_f_va, preds, fit_time_seconds=t_fit, model_name=model_name)

                    fold_maes.append(m.mae)
                    fold_rmses.append(m.rmse)
                    fold_r2s.append(m.r2)
                    fold_times.append(t_fit)
                    fold_details.append({
                        "fold": f_meta.fold,
                        "mae": m.mae,
                        "rmse": m.rmse,
                        "r2": m.r2,
                    })

                mean_mae = float(np.mean(fold_maes))
                std_mae = float(np.std(fold_maes, ddof=1)) if len(fold_maes) > 1 else 0.0
                mean_rmse = float(np.mean(fold_rmses))
                std_rmse = float(np.std(fold_rmses, ddof=1)) if len(fold_rmses) > 1 else 0.0
                mean_r2 = float(np.mean(fold_r2s))
                std_r2 = float(np.std(fold_r2s, ddof=1)) if len(fold_r2s) > 1 else 0.0
                mean_time = float(np.mean(fold_times))

                rec = {
                    "model_name": model_name,
                    "config_idx": cfg_idx,
                    "config": str(cfg),
                    "mean_cv_mae": mean_mae,
                    "std_cv_mae": std_mae,
                    "mean_cv_rmse": mean_rmse,
                    "std_cv_rmse": std_rmse,
                    "mean_cv_r2": mean_r2,
                    "std_cv_r2": std_r2,
                    "mean_fit_time_seconds": mean_time,
                }
                for fd in fold_details:
                    f_num = fd["fold"]
                    rec[f"fold_{f_num}_mae"] = fd["mae"]
                    rec[f"fold_{f_num}_rmse"] = fd["rmse"]
                    rec[f"fold_{f_num}_r2"] = fd["r2"]

                cv_records.append(rec)

                if mean_mae < best_cv_mae:
                    best_cv_mae = mean_mae
                    best_cfg = cfg

            best_configs[model_name] = best_cfg
            logger.info(f"Selected best config for {model_name} (CV MAE: {best_cv_mae:.4f}): {best_cfg}")

        cv_df = pd.DataFrame(cv_records)
        cv_df.to_csv(self.metrics_dir / "cv_metrics.csv", index=False)

        # 5. Validation Evaluation on Common Mask (15% chronological slice)
        logger.info("Evaluating all 7 models on chronological Validation set (15%)...")
        val_common_mask, val_mask_stats = compute_common_evaluation_mask(
            X_all, val_mask, total_calendar_hours=splits_meta.val_boundary.total_hours
        )

        X_train_full, y_train_full = X_all.loc[train_mask], y_all.loc[train_mask]
        X_val_common, y_val_common = X_all.loc[val_common_mask], y_all.loc[val_common_mask]

        # Calculate baselines on val
        persistence_val_pred = X_val_common["lag_1"].to_numpy()
        seasonal_val_pred = X_val_common["lag_24"].to_numpy()

        p_val_raw_mae = float(np.mean(np.abs(y_val_common.to_numpy() - persistence_val_pred)))
        s_val_raw_mae = float(np.mean(np.abs(y_val_common.to_numpy() - seasonal_val_pred)))

        p_val_m = calculate_metrics(
            y_val_common,
            persistence_val_pred,
            persistence_mae=p_val_raw_mae,
            seasonal_naive_mae=s_val_raw_mae,
            model_name="Persistence",
        )
        s_val_m = calculate_metrics(
            y_val_common,
            seasonal_val_pred,
            persistence_mae=p_val_raw_mae,
            seasonal_naive_mae=s_val_raw_mae,
            model_name="Seasonal Naive",
        )

        val_metrics_list = [p_val_m, s_val_m]
        val_predictions: Dict[str, np.ndarray] = {
            "Persistence": persistence_val_pred,
            "Seasonal Naive": seasonal_val_pred,
        }

        # Fit configured ML models on Train (70%) and evaluate on Val (15%)
        val_fitted_models = {}
        for model_name, cfg in best_configs.items():
            pipe = get_model_pipeline(model_name, **cfg)
            t0 = time.time()
            pipe.fit(X_train_full, y_train_full)
            t_fit = time.time() - t0

            preds = pipe.predict(X_val_common)
            m = calculate_metrics(
                y_val_common,
                preds,
                persistence_mae=p_val_raw_mae,
                seasonal_naive_mae=s_val_raw_mae,
                fit_time_seconds=t_fit,
                model_name=model_name,
            )
            val_metrics_list.append(m)
            val_predictions[model_name] = preds
            val_fitted_models[model_name] = pipe

        val_metrics_df = metrics_to_dataframe(val_metrics_list)
        val_metrics_df.to_csv(self.metrics_dir / "validation_metrics.csv", index=False)

        # SELECT WINNER BASED STRICTLY ON VALIDATION MAE
        # Baselines included or ML models only? Winner amongst ML models
        ml_val_df = val_metrics_df[~val_metrics_df["model_name"].isin(["Persistence", "Seasonal Naive"])]
        winner_row = ml_val_df.sort_values("mae").iloc[0]
        winner_name = str(winner_row["model_name"])
        winner_config = best_configs[winner_name]
        logger.info(
            f"*** VALIDATION WINNER SELECTED: {winner_name} (Validation MAE: {winner_row['mae']:.4f} kW) ***"
        )

        # 6. Ablation Study on Validation Set using Validation Winner Model
        logger.info(f"Running Ablation Study on Validation set with winner model: {winner_name}...")
        ablation_records = []
        for ab_name, ab_features in FEATURE_SETS.items():
            ordered_features = [c for c in X_all.columns if c in ab_features]
            if ab_name == "full":
                ab_preds = val_predictions[winner_name]
                winner_val_row = val_metrics_df[val_metrics_df["model_name"] == winner_name].iloc[0]
                t_fit = float(winner_val_row["fit_time_seconds"])
            else:
                ab_pipe = get_model_pipeline(winner_name, **winner_config)
                X_tr_ab = X_train_full[ordered_features]
                X_va_ab = X_val_common[ordered_features]

                t0 = time.time()
                ab_pipe.fit(X_tr_ab, y_train_full)
                t_fit = time.time() - t0

                ab_preds = ab_pipe.predict(X_va_ab)

            ab_m = calculate_metrics(
                y_val_common,
                ab_preds,
                persistence_mae=p_val_raw_mae,
                seasonal_naive_mae=s_val_raw_mae,
                fit_time_seconds=t_fit,
                model_name=ab_name,
            )
            ablation_records.append({
                "config_name": ab_name,
                "features": ordered_features,
                "num_features": len(ordered_features),
                "mae": ab_m.mae,
                "rmse": ab_m.rmse,
                "r2": ab_m.r2,
                "fit_time_seconds": t_fit,
            })

        ablation_df = pd.DataFrame(ablation_records)
        ablation_df.to_csv(self.metrics_dir / "ablation_metrics.csv", index=False)
        plot_ablation_comparison(ablation_df, self.figures_dir / "ablation_comparison.png")

        # 7. Refit ML Models on First 85% (Train + Val) and Single Test Evaluation (Last 15%)
        logger.info("Refitting all models on First 85% and evaluating once on held-out Test (15%)...")
        test_common_mask, test_mask_stats = compute_common_evaluation_mask(
            X_all, test_mask, total_calendar_hours=splits_meta.test_boundary.total_hours
        )

        X_first85, y_first85 = X_all.loc[first85_mask], y_all.loc[first85_mask]
        X_test_common, y_test_common = X_all.loc[test_common_mask], y_all.loc[test_common_mask]

        # Calculate high-load threshold (90th percentile of active power kW on Train set only!)
        high_load_p90 = float(np.percentile(y_train_full, 90))
        logger.info(f"High-load threshold (90th percentile of Train set): {high_load_p90:.4f} kW")

        # Baselines on test
        persistence_test_pred = X_test_common["lag_1"].to_numpy()
        seasonal_test_pred = X_test_common["lag_24"].to_numpy()

        p_test_raw_mae = float(np.mean(np.abs(y_test_common.to_numpy() - persistence_test_pred)))
        s_test_raw_mae = float(np.mean(np.abs(y_test_common.to_numpy() - seasonal_test_pred)))

        p_test_m = calculate_metrics(
            y_test_common,
            persistence_test_pred,
            persistence_mae=p_test_raw_mae,
            seasonal_naive_mae=s_test_raw_mae,
            model_name="Persistence",
        )
        s_test_m = calculate_metrics(
            y_test_common,
            seasonal_test_pred,
            persistence_mae=p_test_raw_mae,
            seasonal_naive_mae=s_test_raw_mae,
            model_name="Seasonal Naive",
        )

        test_metrics_list = [p_test_m, s_test_m]
        test_pred_df = pd.DataFrame(
            {
                "timestamp": y_test_common.index,
                "actual": y_test_common.to_numpy(),
                "Persistence": persistence_test_pred,
                "Seasonal Naive": seasonal_test_pred,
            },
            index=y_test_common.index,
        )

        # Save baseline estimators
        joblib.dump(PersistenceBaseline(), self.models_dir / "persistence.joblib")
        joblib.dump(SeasonalNaiveBaseline(), self.models_dir / "seasonal_naive.joblib")

        refitted_models = {}
        for model_name, cfg in best_configs.items():
            pipe = get_model_pipeline(model_name, **cfg)
            t0 = time.time()
            pipe.fit(X_first85, y_first85)
            t_fit = time.time() - t0

            preds = pipe.predict(X_test_common)
            m = calculate_metrics(
                y_test_common,
                preds,
                persistence_mae=p_test_raw_mae,
                seasonal_naive_mae=s_test_raw_mae,
                fit_time_seconds=t_fit,
                model_name=model_name,
            )
            test_metrics_list.append(m)
            test_pred_df[model_name] = preds
            refitted_models[model_name] = pipe

            # Save model joblib
            slug = model_name.lower().replace(" ", "_")
            joblib.dump(pipe, self.models_dir / f"{slug}.joblib")

        # Save winner model as winner_model.joblib
        joblib.dump(refitted_models[winner_name], self.models_dir / "winner_model.joblib")

        test_metrics_df = metrics_to_dataframe(test_metrics_list)
        test_metrics_df.to_csv(self.metrics_dir / "test_metrics.csv", index=False)
        test_pred_df.to_csv(self.predictions_dir / "test_predictions.csv", index=False)

        # 8. High-Load Evaluation on Test Set
        logger.info("Evaluating high-load breakdown on test set...")
        high_load_results = {}
        for model_col in ["Persistence", "Seasonal Naive"] + list(best_configs.keys()):
            high_load_results[model_col] = evaluate_high_load(
                y_test_common,
                test_pred_df[model_col],
                threshold_p90=high_load_p90,
            )

        # Save high load summary
        high_load_rows = []
        for model_col, res in high_load_results.items():
            high_load_rows.append({
                "model_name": model_col,
                "threshold_p90_kW": high_load_p90,
                "high_load_count": res["high_load"]["count"],
                "high_load_mae": res["high_load"]["mae"],
                "high_load_rmse": res["high_load"]["rmse"],
                "normal_load_count": res["normal_load"]["count"],
                "normal_load_mae": res["normal_load"]["mae"],
                "normal_load_rmse": res["normal_load"]["rmse"],
                "overall_mae": res["overall"]["mae"],
            })
        high_load_df = pd.DataFrame(high_load_rows)
        high_load_df.to_csv(self.metrics_dir / "high_load_metrics.csv", index=False)

        # Predeclared ablation on test set (for reporting transparency)
        test_ablation_records = []
        for ab_name, ab_features in FEATURE_SETS.items():
            ordered_features = [c for c in X_all.columns if c in ab_features]
            if ab_name == "full":
                ab_preds = test_pred_df[winner_name].to_numpy()
            else:
                ab_pipe = get_model_pipeline(winner_name, **winner_config)
                ab_pipe.fit(X_first85[ordered_features], y_first85)
                ab_preds = ab_pipe.predict(X_test_common[ordered_features])

            ab_m = calculate_metrics(
                y_test_common,
                ab_preds,
                persistence_mae=p_test_raw_mae,
                seasonal_naive_mae=s_test_raw_mae,
                model_name=ab_name,
            )
            test_ablation_records.append({
                "config_name": ab_name,
                "num_features": len(ordered_features),
                "mae": ab_m.mae,
                "rmse": ab_m.rmse,
                "r2": ab_m.r2,
            })
        test_ab_df = pd.DataFrame(test_ablation_records)
        test_ab_df.to_csv(self.metrics_dir / "test_ablation_metrics.csv", index=False)

        # 9. Split and Execution Metadata JSON
        cv_fold_meta_list = [asdict(f_meta) for _, _, f_meta in cv_folds]
        metadata = {
            "dataset_stats": asdict(data_stats),
            "splits": asdict(splits_meta),
            "cv_folds": cv_fold_meta_list,
            "cv_folds_count": len(cv_folds),
            "val_common_mask_stats": val_mask_stats,
            "test_common_mask_stats": test_mask_stats,
            "best_hyperparameters": best_configs,
            "validation_selected_winner": winner_name,
            "winner_config": winner_config,
            "high_load_threshold_train_p90_kW": high_load_p90,
            "timestamp_created": pd.Timestamp.now().isoformat(),
        }
        with open(self.metadata_dir / "split_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # 10. Generate Publication Figures
        logger.info("Generating publication-quality figures...")
        plot_one_week_forecast(
            test_pred_df,
            winner_col=winner_name,
            output_path=self.figures_dir / "one_week_actual_vs_predicted.png",
        )
        plot_residuals_over_time(
            test_pred_df,
            winner_col=winner_name,
            output_path=self.figures_dir / "residuals_over_time.png",
        )
        plot_mae_by_hour(
            test_pred_df,
            model_cols=["Persistence", "Seasonal Naive", winner_name],
            output_path=self.figures_dir / "mae_by_hour.png",
        )
        plot_model_comparison(
            test_metrics_df,
            output_path=self.figures_dir / "model_comparison.png",
            winner_name=winner_name,
        )

        logger.info("================ ML-01 PIPELINE COMPLETED SUCCESSFULLY ================")
        return {
            "validation_winner": winner_name,
            "validation_metrics": val_metrics_df,
            "test_metrics": test_metrics_df,
            "metadata": metadata,
        }
