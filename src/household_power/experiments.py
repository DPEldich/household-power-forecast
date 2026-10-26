"""Load dynamics feature experiments, expanding CV selection, and quantile uncertainty evaluation (ML-03)."""

from __future__ import annotations

from dataclasses import asdict
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import xgboost as xgb

from .data import prepare_and_cache_hourly_data
from .evaluation import (
    calculate_metrics,
    compute_common_evaluation_mask,
    evaluate_high_load,
    metrics_to_dataframe,
)
from .features import (
    BASE_FEATURE_NAMES,
    EXTENDED_FEATURES,
    build_extended_features,
    build_features,
)
from .splits import (
    DatasetSplits,
    compute_chronological_splits,
    get_expanding_cv_folds,
)
from .uncertainty import (
    evaluate_uncertainty,
    pinball_loss,
    repair_and_clip_quantiles,
)

logger = logging.getLogger(__name__)


def compute_high_load_threshold(y_train: pd.Series | np.ndarray, percentile: float = 90.0) -> float:
    """Compute high-load cutoff strictly from training split values."""
    arr = np.asarray(y_train, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        raise ValueError("Cannot compute high-load threshold from empty or non-finite training array")
    return float(np.percentile(arr, percentile))


def evaluate_candidate_cv(
    cand_fold_metrics: List[Dict[str, float]],
    baseline_fold_metrics: List[Dict[str, float]],
    cand_id: str,
) -> Dict[str, Any]:
    """Evaluate candidate CV performance against baseline using strict protocol rules.

    Rules:
    1. Overall MAE ratio <= 1.01 (no severe overall regression)
    2. Candidate high-load MAE <= baseline high-load MAE in at least 2 of 3 folds
    """
    if len(cand_fold_metrics) != len(baseline_fold_metrics) or len(cand_fold_metrics) == 0:
        raise ValueError("Fold metrics lists must be non-empty and of identical length")

    mean_ov_mae = float(np.mean([r["overall_mae"] for r in cand_fold_metrics]))
    mean_hl_mae = float(np.mean([r["high_load_mae"] for r in cand_fold_metrics]))
    mean_norm_mae = float(np.mean([r["normal_load_mae"] for r in cand_fold_metrics]))

    base_ov_mae = float(np.mean([r["overall_mae"] for r in baseline_fold_metrics]))
    base_hl_mae = float(np.mean([r["high_load_mae"] for r in baseline_fold_metrics]))

    non_reg_folds = 0
    for c_rec, b_rec in zip(cand_fold_metrics, baseline_fold_metrics):
        if c_rec["high_load_mae"] <= b_rec["high_load_mae"] + 1e-9:
            non_reg_folds += 1

    overall_ratio = mean_ov_mae / base_ov_mae if base_ov_mae > 0 else float("inf")
    overall_pass = overall_ratio <= 1.01
    folds_pass = non_reg_folds >= 2
    hl_improved_on_average = mean_hl_mae < base_hl_mae

    eligible = bool(overall_pass and folds_pass and hl_improved_on_average and (cand_id != "baseline"))

    return {
        "candidate_id": cand_id,
        "mean_overall_mae": mean_ov_mae,
        "mean_high_load_mae": mean_hl_mae,
        "mean_normal_load_mae": mean_norm_mae,
        "overall_mae_ratio_vs_baseline": overall_ratio,
        "overall_constraint_pass": overall_pass,
        "high_load_non_regression_folds": non_reg_folds,
        "high_load_folds_pass": folds_pass,
        "high_load_improved_vs_baseline": hl_improved_on_average,
        "eligible_for_promotion": eligible,
    }


def confirm_validation_candidate(
    cand_val_metrics: Dict[str, float],
    baseline_val_metrics: Dict[str, float],
    cand_id: str,
) -> Tuple[bool, str]:
    """Validate candidate on chronological confirmation split.

    Must satisfy:
    1. Overall MAE <= baseline * 1.01
    2. High-load MAE <= baseline high-load MAE
    If either fails, strictly fallback to 'baseline'.
    """
    if cand_id == "baseline":
        return False, "baseline"

    overall_pass = cand_val_metrics["overall_mae"] <= baseline_val_metrics["overall_mae"] * 1.01
    hl_pass = cand_val_metrics["high_load_mae"] <= baseline_val_metrics["high_load_mae"]

    confirmed = bool(overall_pass and hl_pass)
    promoted = cand_id if confirmed else "baseline"
    return confirmed, promoted


def align_baseline_test_predictions(
    baseline_csv_path: Path,
    test_timestamps: pd.DatetimeIndex,
    expected_actual: np.ndarray,
) -> np.ndarray:
    """Load baseline test predictions, validate integrity and timestamps, and align strictly to index."""
    if not baseline_csv_path.exists():
        raise FileNotFoundError(f"Baseline test predictions not found at {baseline_csv_path}")

    base_df = pd.read_csv(baseline_csv_path, parse_dates=["timestamp"])
    if base_df["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps found in baseline test predictions CSV")
    if "XGBoost" not in base_df.columns:
        raise KeyError("Baseline model 'XGBoost' not found in baseline test predictions CSV")

    base_df = base_df.set_index("timestamp")
    missing_ts = test_timestamps.difference(base_df.index)
    if len(missing_ts) > 0:
        raise ValueError(f"Baseline test predictions CSV missing {len(missing_ts)} test timestamps")

    aligned_actual = base_df.loc[test_timestamps, "actual"].to_numpy()
    if not np.allclose(aligned_actual, expected_actual, atol=1e-5):
        raise ValueError("Ground truth active power in baseline test predictions differs from evaluation slice")

    return base_df.loc[test_timestamps, "XGBoost"].to_numpy()


class ExperimentRunner:
    """Encapsulates the ML-03 causal load-dynamics protocol and uncertainty evaluation."""

    def __init__(
        self,
        raw_dir: str | Path = "data/raw",
        processed_dir: str | Path = "data/processed",
        baseline_artifacts_dir: str | Path = "artifacts",
        experimental_dir: str | Path = "artifacts/experimental/ml03",
        fixed_xgb_config: Optional[Dict[str, Any]] = None,
    ):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.baseline_artifacts_dir = Path(baseline_artifacts_dir)
        self.experimental_dir = Path(experimental_dir)

        self.models_dir = self.experimental_dir / "models"
        self.metrics_dir = self.experimental_dir / "metrics"
        self.metadata_dir = self.experimental_dir / "metadata"
        self.predictions_dir = self.experimental_dir / "predictions"
        self.figures_dir = self.experimental_dir / "figures"

        for p in [
            self.models_dir,
            self.metrics_dir,
            self.metadata_dir,
            self.predictions_dir,
            self.figures_dir,
        ]:
            p.mkdir(parents=True, exist_ok=True)

        if fixed_xgb_config is not None:
            self.fixed_xgb_config = dict(fixed_xgb_config)
            logger.info(f"Using explicitly supplied XGBoost configuration: {self.fixed_xgb_config}")
        else:
            # Bind baseline XGBoost hyperparameter config strictly from baseline metadata
            baseline_meta_path = self.baseline_artifacts_dir / "metadata/split_metadata.json"
            if not baseline_meta_path.exists():
                raise FileNotFoundError(f"Baseline metadata not found at {baseline_meta_path}")

            meta = json.loads(baseline_meta_path.read_text(encoding="utf-8"))
            best_xgb = meta.get("best_hyperparameters", {}).get("XGBoost")
            if not best_xgb:
                raise KeyError("best_hyperparameters.XGBoost missing from baseline split_metadata.json")

            self.fixed_xgb_config = {
                "n_estimators": int(best_xgb["n_estimators"]),
                "learning_rate": float(best_xgb["learning_rate"]),
                "max_depth": int(best_xgb["max_depth"]),
                "subsample": float(best_xgb["subsample"]),
                "random_state": 42,
                "n_jobs": 4,
            }
            logger.info(f"Bound fixed XGBoost configuration from baseline metadata: {self.fixed_xgb_config}")

    def define_candidates(self) -> Dict[str, Dict[str, Any]]:
        """Declare bounded candidates before execution."""
        return {
            "baseline": {
                "name": "baseline",
                "description": "Baseline XGBoost (20 features, unweighted)",
                "feature_type": "baseline",
                "features": BASE_FEATURE_NAMES,
                "weighted": False,
                "weight_multiplier": 1.0,
            },
            "enhanced_features": {
                "name": "enhanced_features",
                "description": "Enhanced load-dynamics features (26 features, unweighted)",
                "feature_type": "extended",
                "features": EXTENDED_FEATURES,
                "weighted": False,
                "weight_multiplier": 1.0,
            },
            "enhanced_weighted": {
                "name": "enhanced_weighted",
                "description": "Enhanced load-dynamics with mild high-load training weight (w=1.5)",
                "feature_type": "extended",
                "features": EXTENDED_FEATURES,
                "weighted": True,
                "weight_multiplier": 1.5,
            },
        }

    def _get_point_pipeline(self) -> Pipeline:
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            (
                "regressor",
                xgb.XGBRegressor(
                    objective="reg:squarederror",
                    eval_metric="mae",
                    **self.fixed_xgb_config,
                ),
            ),
        ])

    def _get_quantile_pipeline(self, alpha: float) -> Pipeline:
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            (
                "regressor",
                xgb.XGBRegressor(
                    objective="reg:quantileerror",
                    quantile_alpha=float(alpha),
                    **self.fixed_xgb_config,
                ),
            ),
        ])

    def run(self) -> Dict[str, Any]:
        """Execute the declared ML-03 protocol."""
        logger.info("================ STARTING ML-03 EXPERIMENT PROTOCOL ================")

        # 1. Load cached data
        hourly_df, data_stats = prepare_and_cache_hourly_data(
            raw_dir=self.raw_dir,
            processed_dir=self.processed_dir,
            force_download=False,
            force_process=False,
        )

        # 2. Build features
        X_base, y_all, valid_mask = build_features(hourly_df)
        X_ext, _, _ = build_extended_features(hourly_df)

        # 3. Splits
        splits_meta, train_mask, val_mask, test_mask, first85_mask = compute_chronological_splits(
            hourly_df, valid_mask
        )
        cv_folds = get_expanding_cv_folds(hourly_df, valid_mask, splits_meta.train_boundary, n_folds=3)

        candidates = self.define_candidates()

        # Step 4: Write protocol.json BEFORE CV execution (audit trail)
        protocol_meta = {
            "protocol_name": "ML-03: Load Dynamics & Quantile Uncertainty",
            "candidate_definitions": candidates,
            "fixed_model_config": self.fixed_xgb_config,
            "raw_objectives": {
                "point_model": "reg:squarederror",
                "quantile_models": "reg:quantileerror",
            },
            "preprocessing_policy": {
                "imputer": "SimpleImputer(strategy='median')",
                "description": "Fitted strictly on training split feature matrix X_train; transforms val/test using learned training medians",
            },
            "quantile_alpha_levels": [0.1, 0.5, 0.9],
            "quantile_raw_to_repaired_policy": "clip >= 0.0 kW, deterministic row-wise sort enforcing 0 <= P10 <= P50 <= P90",
            "splits": {
                "train_boundary": splits_meta.train_boundary.__dict__,
                "val_boundary": splits_meta.val_boundary.__dict__,
                "test_boundary": splits_meta.test_boundary.__dict__,
                "first85_boundary": splits_meta.first85_boundary.__dict__,
            },
            "cv_folds_count": len(cv_folds),
            "cv_folds": [asdict(f_meta) for _, _, f_meta in cv_folds],
            "selection_rules": {
                "cv_overall_mae_ratio_max": 1.01,
                "cv_high_load_non_regression_folds_min": 2,
                "val_confirmation_overall_mae_ratio_max": 1.01,
                "val_confirmation_high_load_mae_max_ratio": 1.00,
                "fallback_policy": "Revert to baseline if candidate fails confirmation",
            },
            "disclaimer": "Test set is a historical reused benchmark; not fresh unseen independent data.",
            "protocol_declared_timestamp": pd.Timestamp.now().isoformat(),
        }
        with open(self.metadata_dir / "protocol.json", "w", encoding="utf-8") as f:
            json.dump(protocol_meta, f, indent=2)
        logger.info(f"Persisted pre-execution protocol to {self.metadata_dir / 'protocol.json'}")

        # 5. Expanding CV on initial 70% Train
        logger.info("Running Expanding CV for candidate selection on Train set...")
        cv_records = []
        fold_details_by_cand: Dict[str, List[Dict[str, Any]]] = {k: [] for k in candidates}

        for f_idx, (f_train_m, f_val_m, f_meta) in enumerate(cv_folds):
            y_f_tr = y_all.loc[f_train_m]
            fold_threshold = compute_high_load_threshold(y_f_tr, 90.0)

            # Common mask on fold val
            f_val_common_m, _ = compute_common_evaluation_mask(
                X_base, f_val_m, total_calendar_hours=f_meta.val_total_hours
            )
            y_f_va = y_all.loc[f_val_common_m]

            for cand_id, cand_meta in candidates.items():
                X_feat = X_ext if cand_meta["feature_type"] == "extended" else X_base
                cols = cand_meta["features"]

                X_tr = X_feat.loc[f_train_m, cols]
                y_tr = y_f_tr
                X_va = X_feat.loc[f_val_common_m, cols]

                pipe = self._get_point_pipeline()
                t0 = time.time()
                if cand_meta["weighted"]:
                    weights = np.where(y_tr.to_numpy() >= fold_threshold, cand_meta["weight_multiplier"], 1.0)
                    pipe.fit(X_tr, y_tr, regressor__sample_weight=weights)
                else:
                    pipe.fit(X_tr, y_tr)
                t_fit = time.time() - t0

                preds = pipe.predict(X_va)
                m_overall = calculate_metrics(y_f_va, preds, fit_time_seconds=t_fit, model_name=cand_id)
                m_hl = evaluate_high_load(y_f_va, pd.Series(preds, index=y_f_va.index), threshold_p90=fold_threshold)

                rec = {
                    "candidate_id": cand_id,
                    "fold": f_meta.fold,
                    "fold_threshold_p90": fold_threshold,
                    "overall_mae": m_overall.mae,
                    "overall_rmse": m_overall.rmse,
                    "high_load_mae": m_hl["high_load"]["mae"],
                    "high_load_rmse": m_hl["high_load"]["rmse"],
                    "normal_load_mae": m_hl["normal_load"]["mae"],
                    "fit_time_seconds": t_fit,
                }
                fold_details_by_cand[cand_id].append(rec)
                cv_records.append(rec)

        cv_df = pd.DataFrame(cv_records)
        cv_df.to_csv(self.metrics_dir / "cv_candidate_metrics.csv", index=False)

        # Aggregate CV statistics using production helper
        cv_summary = {}
        for cand_id in candidates:
            cv_summary[cand_id] = evaluate_candidate_cv(
                fold_details_by_cand[cand_id],
                fold_details_by_cand["baseline"],
                cand_id,
            )

        logger.info(f"CV candidate evaluation summary: {cv_summary}")

        eligible_cands = [
            cid for cid, s in cv_summary.items() if s["eligible_for_promotion"]
        ]
        if eligible_cands:
            cv_selected_cand_id = min(eligible_cands, key=lambda c: cv_summary[c]["mean_high_load_mae"])
            logger.info(f"CV selected candidate: {cv_selected_cand_id}")
        else:
            cv_selected_cand_id = "baseline"
            logger.info("No candidate passed CV constraints. CV selected: baseline")

        # 6. Validation Confirmation
        logger.info("Executing Validation confirmation...")
        train_full_m = train_mask
        val_common_m, _ = compute_common_evaluation_mask(
            X_base, val_mask, total_calendar_hours=splits_meta.val_boundary.total_hours
        )

        y_train_full = y_all.loc[train_full_m]
        y_val_common = y_all.loc[val_common_m]

        # High-load threshold computed strictly from full train (70%)
        train_hl_threshold = compute_high_load_threshold(y_train_full, 90.0)

        # Evaluate baseline on val
        base_pipe = self._get_point_pipeline()
        base_pipe.fit(X_base.loc[train_full_m, BASE_FEATURE_NAMES], y_train_full)
        base_val_preds = base_pipe.predict(X_base.loc[val_common_m, BASE_FEATURE_NAMES])
        base_val_m = calculate_metrics(y_val_common, base_val_preds, model_name="baseline")
        base_val_hl = evaluate_high_load(y_val_common, pd.Series(base_val_preds, index=y_val_common.index), threshold_p90=train_hl_threshold)

        val_records = [{
            "candidate_id": "baseline",
            "overall_mae": base_val_m.mae,
            "overall_rmse": base_val_m.rmse,
            "high_load_mae": base_val_hl["high_load"]["mae"],
            "normal_load_mae": base_val_hl["normal_load"]["mae"],
        }]

        val_cand_metrics = val_records[0]

        if cv_selected_cand_id != "baseline":
            cand_meta = candidates[cv_selected_cand_id]
            X_feat = X_ext if cand_meta["feature_type"] == "extended" else X_base
            cols = cand_meta["features"]

            cand_pipe = self._get_point_pipeline()
            if cand_meta["weighted"]:
                w = np.where(y_train_full.to_numpy() >= train_hl_threshold, cand_meta["weight_multiplier"], 1.0)
                cand_pipe.fit(X_feat.loc[train_full_m, cols], y_train_full, regressor__sample_weight=w)
            else:
                cand_pipe.fit(X_feat.loc[train_full_m, cols], y_train_full)

            cand_val_preds = cand_pipe.predict(X_feat.loc[val_common_m, cols])
            cand_val_m = calculate_metrics(y_val_common, cand_val_preds, model_name=cv_selected_cand_id)
            cand_val_hl = evaluate_high_load(y_val_common, pd.Series(cand_val_preds, index=y_val_common.index), threshold_p90=train_hl_threshold)

            val_cand_metrics = {
                "candidate_id": cv_selected_cand_id,
                "overall_mae": cand_val_m.mae,
                "overall_rmse": cand_val_m.rmse,
                "high_load_mae": cand_val_hl["high_load"]["mae"],
                "normal_load_mae": cand_val_hl["normal_load"]["mae"],
            }
            val_records.append(val_cand_metrics)

        val_confirmed, final_selected_point_cand = confirm_validation_candidate(
            val_cand_metrics, val_records[0], cv_selected_cand_id
        )
        logger.info(
            f"Validation confirmation decision: confirmed={val_confirmed}, promoted_model={final_selected_point_cand}"
        )

        val_cand_df = pd.DataFrame(val_records)
        val_cand_df.to_csv(self.metrics_dir / "validation_candidate_metrics.csv", index=False)

        # Step 7: Persist selection_frozen.json BEFORE any test evaluation
        selected_meta = candidates[final_selected_point_cand]
        quant_X_feat = X_ext if selected_meta["feature_type"] == "extended" else X_base
        quant_cols = selected_meta["features"]

        frozen_selection = {
            "cv_selected_candidate": cv_selected_cand_id,
            "validation_confirmation": {
                "candidate_evaluated": cv_selected_cand_id,
                "confirmed": val_confirmed,
                "promoted_point_model": final_selected_point_cand,
                "validation_metrics": val_records,
            },
            "selected_feature_set": selected_meta["feature_type"],
            "selected_feature_names": quant_cols,
            "high_load_threshold_train_p90_kW": train_hl_threshold,
            "frozen_timestamp": pd.Timestamp.now().isoformat(),
        }
        with open(self.metadata_dir / "selection_frozen.json", "w", encoding="utf-8") as f:
            json.dump(frozen_selection, f, indent=2)
        logger.info(f"Persisted frozen selection to {self.metadata_dir / 'selection_frozen.json'}")

        # 8. Quantile validation evaluation (Fit on Train 70%, evaluate on Val 15%)
        val_q_pipes = {}
        val_q_raw = {}
        for alpha in [0.1, 0.5, 0.9]:
            q_pipe = self._get_quantile_pipeline(alpha)
            q_pipe.fit(quant_X_feat.loc[train_full_m, quant_cols], y_train_full)
            val_q_pipes[alpha] = q_pipe
            val_q_raw[alpha] = q_pipe.predict(quant_X_feat.loc[val_common_m, quant_cols])

        val_p10, val_p50, val_p90 = repair_and_clip_quantiles(
            val_q_raw[0.1], val_q_raw[0.5], val_q_raw[0.9]
        )
        val_unc_metrics = evaluate_uncertainty(
            y_val_common.to_numpy(), val_p10, val_p50, val_p90, threshold_p90=train_hl_threshold
        )

        # Always obtain enhanced_features predictions on validation for audit
        if cv_selected_cand_id == "enhanced_features":
            enh_val_preds = cand_val_preds
        else:
            enh_pipe = self._get_point_pipeline()
            enh_pipe.fit(X_ext.loc[train_full_m, EXTENDED_FEATURES], y_train_full)
            enh_val_preds = enh_pipe.predict(X_ext.loc[val_common_m, EXTENDED_FEATURES])

        # Save validation predictions CSV for independent audit
        val_point_pred_chosen = base_val_preds if final_selected_point_cand == "baseline" else cand_val_preds
        val_pred_df = pd.DataFrame(
            {
                "timestamp": y_val_common.index,
                "actual": y_val_common.to_numpy(),
                "baseline_pred": base_val_preds,
                "enhanced_features_pred": enh_val_preds,
                "point_pred": val_point_pred_chosen,
                "p10": val_p10,
                "p50": val_p50,
                "p90": val_p90,
            },
            index=y_val_common.index,
        )
        val_pred_df.to_csv(self.predictions_dir / "experimental_val_predictions.csv", index=False)

        # 9. Refit on First 85% for Descriptive Test Evaluation
        logger.info("Refitting selected point model and quantile models on First 85% for descriptive test...")
        test_common_m, test_mask_stats = compute_common_evaluation_mask(
            X_base, test_mask, total_calendar_hours=splits_meta.test_boundary.total_hours
        )
        X_first85 = quant_X_feat.loc[first85_mask, quant_cols]
        y_first85 = y_all.loc[first85_mask]

        X_test_eval = quant_X_feat.loc[test_common_m, quant_cols]
        y_test_eval = y_all.loc[test_common_m]

        final_point_pipe = self._get_point_pipeline()
        if selected_meta["weighted"]:
            w_85 = np.where(y_first85.to_numpy() >= train_hl_threshold, selected_meta["weight_multiplier"], 1.0)
            final_point_pipe.fit(X_first85, y_first85, regressor__sample_weight=w_85)
        else:
            final_point_pipe.fit(X_first85, y_first85)

        point_test_preds = final_point_pipe.predict(X_test_eval)

        # Load baseline test predictions using alignment helper
        baseline_test_csv = self.baseline_artifacts_dir / "predictions/test_predictions.csv"
        base_test_preds = align_baseline_test_predictions(
            baseline_test_csv, test_timestamps=y_test_eval.index, expected_actual=y_test_eval.to_numpy()
        )

        m_point_test = calculate_metrics(y_test_eval, point_test_preds, model_name=final_selected_point_cand)
        m_point_hl = evaluate_high_load(y_test_eval, pd.Series(point_test_preds, index=y_test_eval.index), threshold_p90=train_hl_threshold)

        m_base_test = calculate_metrics(y_test_eval, base_test_preds, model_name="baseline_xgboost")
        m_base_hl = evaluate_high_load(y_test_eval, pd.Series(base_test_preds, index=y_test_eval.index), threshold_p90=train_hl_threshold)

        test_point_df = pd.DataFrame([
            {
                "model_name": "baseline_xgboost",
                "overall_mae": m_base_test.mae,
                "overall_rmse": m_base_test.rmse,
                "high_load_mae": m_base_hl["high_load"]["mae"],
                "high_load_rmse": m_base_hl["high_load"]["rmse"],
                "normal_load_mae": m_base_hl["normal_load"]["mae"],
            },
            {
                "model_name": final_selected_point_cand,
                "overall_mae": m_point_test.mae,
                "overall_rmse": m_point_test.rmse,
                "high_load_mae": m_point_hl["high_load"]["mae"],
                "high_load_rmse": m_point_hl["high_load"]["rmse"],
                "normal_load_mae": m_point_hl["normal_load"]["mae"],
            },
        ])
        test_point_df.to_csv(self.metrics_dir / "test_point_comparison.csv", index=False)

        # Quantiles refit on First 85%
        test_q_pipes = {}
        test_q_raw = {}
        for alpha in [0.1, 0.5, 0.9]:
            q_pipe = self._get_quantile_pipeline(alpha)
            q_pipe.fit(X_first85, y_first85)
            test_q_pipes[alpha] = q_pipe
            test_q_raw[alpha] = q_pipe.predict(X_test_eval)

        test_p10, test_p50, test_p90 = repair_and_clip_quantiles(
            test_q_raw[0.1], test_q_raw[0.5], test_q_raw[0.9]
        )
        test_unc_metrics = evaluate_uncertainty(
            y_test_eval.to_numpy(), test_p10, test_p50, test_p90, threshold_p90=train_hl_threshold
        )

        unc_rows = []
        for split_name, res in [("validation", val_unc_metrics), ("test", test_unc_metrics)]:
            for subset_name, vals in res.items():
                r = {"split": split_name, "subset": subset_name}
                r.update(vals)
                unc_rows.append(r)
        unc_df = pd.DataFrame(unc_rows)
        unc_df.to_csv(self.metrics_dir / "uncertainty_metrics.csv", index=False)

        # Save test predictions table
        test_pred_df = pd.DataFrame(
            {
                "timestamp": y_test_eval.index,
                "actual": y_test_eval.to_numpy(),
                "baseline_xgboost": base_test_preds,
                "point_pred": point_test_preds,
                "p10": test_p10,
                "p50": test_p50,
                "p90": test_p90,
            },
            index=y_test_eval.index,
        )
        test_pred_df.to_csv(self.predictions_dir / "experimental_test_predictions.csv", index=False)

        # 10. Save Model Estimators
        joblib.dump(final_point_pipe, self.models_dir / "selected_point_model.joblib")
        joblib.dump(test_q_pipes[0.1], self.models_dir / "quantile_p10.joblib")
        joblib.dump(test_q_pipes[0.5], self.models_dir / "quantile_p50.joblib")
        joblib.dump(test_q_pipes[0.9], self.models_dir / "quantile_p90.joblib")

        # 11. Diagnostics Breakdown
        diag_by_hour, diag_by_month, diag_by_change = self._compute_diagnostics(
            test_pred_df, X_ext.loc[test_common_m], train_hl_threshold
        )
        diag_by_hour.to_csv(self.metrics_dir / "diagnostics_by_hour.csv", index=False)
        diag_by_month.to_csv(self.metrics_dir / "diagnostics_by_month.csv", index=False)
        diag_by_change.to_csv(self.metrics_dir / "diagnostics_by_load_change.csv", index=False)

        # 12. Figures
        self._generate_figures(test_pred_df, diag_by_hour, diag_by_change)

        # 13. Complete Metadata JSON referencing frozen protocol and selection
        meta = {
            "protocol_name": "ML-03: Load Dynamics & Quantile Uncertainty",
            "protocol_file": "metadata/protocol.json",
            "selection_frozen_file": "metadata/selection_frozen.json",
            "candidate_definitions": candidates,
            "expanding_cv_summary": cv_summary,
            "cv_selected_candidate": cv_selected_cand_id,
            "validation_confirmation": frozen_selection["validation_confirmation"],
            "selected_feature_set": selected_meta["feature_type"],
            "selected_feature_names": quant_cols,
            "high_load_threshold_train_p90_kW": train_hl_threshold,
            "test_point_metrics": test_point_df.to_dict(orient="records"),
            "uncertainty_metrics": unc_df.to_dict(orient="records"),
            "nominal_interval_target_pct": 80.0,
            "quantile_alpha_levels": [0.1, 0.5, 0.9],
            "limitations": [
                "Test set is a historical reused benchmark; not an independent untouched test.",
                "Data represents a single residential household in Sceaux, France (2006-2010).",
                "80% nominal interval coverage is a target, not a mathematical guarantee.",
            ],
            "timestamp_created": pd.Timestamp.now().isoformat(),
        }
        with open(self.metadata_dir / "experiment_metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        logger.info("================ ML-03 EXPERIMENT COMPLETED SUCCESSFULLY ================")
        return {
            "cv_selected_candidate": cv_selected_cand_id,
            "promoted_model": final_selected_point_cand,
            "validation_confirmed": val_confirmed,
            "test_point_df": test_point_df,
            "uncertainty_df": unc_df,
            "metadata": meta,
        }

    def _compute_diagnostics(
        self,
        pred_df: pd.DataFrame,
        X_feat_df: pd.DataFrame,
        threshold: float,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        ts = pd.DatetimeIndex(pred_df["timestamp"])
        y = pred_df["actual"].to_numpy()
        y_point = pred_df["point_pred"].to_numpy()
        p10 = pred_df["p10"].to_numpy()
        p50 = pred_df["p50"].to_numpy()
        p90 = pred_df["p90"].to_numpy()

        err_abs = np.abs(y - y_point)
        covered = (y >= p10) & (y <= p90)
        width = p90 - p10

        hour_recs = []
        for h in range(24):
            idx = ts.hour == h
            if np.sum(idx) > 0:
                hour_recs.append({
                    "hour": h,
                    "count": int(np.sum(idx)),
                    "point_mae": float(np.mean(err_abs[idx])),
                    "p50_mae": float(np.mean(np.abs(y[idx] - p50[idx]))),
                    "coverage_pct": float(np.mean(covered[idx]) * 100.0),
                    "mean_width_kw": float(np.mean(width[idx])),
                    "mean_actual": float(np.mean(y[idx])),
                })
        df_hour = pd.DataFrame(hour_recs)

        month_recs = []
        for m in range(1, 13):
            idx = ts.month == m
            if np.sum(idx) > 0:
                month_recs.append({
                    "month": m,
                    "count": int(np.sum(idx)),
                    "point_mae": float(np.mean(err_abs[idx])),
                    "coverage_pct": float(np.mean(covered[idx]) * 100.0),
                    "mean_width_kw": float(np.mean(width[idx])),
                    "mean_actual": float(np.mean(y[idx])),
                })
        df_month = pd.DataFrame(month_recs)

        diff_1h = np.abs(X_feat_df["diff_1h"].fillna(0.0).to_numpy())
        bins = [0.0, 0.2, 0.5, 1.0, float("inf")]
        bin_labels = ["< 0.2 kW (Stable)", "0.2 - 0.5 kW (Mild)", "0.5 - 1.0 kW (Moderate)", ">= 1.0 kW (Sharp)"]
        change_categories = pd.cut(diff_1h, bins=bins, labels=bin_labels, right=False)

        change_recs = []
        for cat in bin_labels:
            idx = np.asarray(change_categories == cat)
            if np.sum(idx) > 0:
                change_recs.append({
                    "load_change_category": cat,
                    "count": int(np.sum(idx)),
                    "ratio_pct": float(np.sum(idx) / len(y) * 100.0),
                    "point_mae": float(np.mean(err_abs[idx])),
                    "coverage_pct": float(np.mean(covered[idx]) * 100.0),
                    "mean_width_kw": float(np.mean(width[idx])),
                })
        df_change = pd.DataFrame(change_recs)

        return df_hour, df_month, df_change

    def _generate_figures(
        self,
        test_pred_df: pd.DataFrame,
        diag_hour: pd.DataFrame,
        diag_change: pd.DataFrame,
    ):
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        fig, ax = plt.subplots(figsize=(14, 5), dpi=150)

        sample_slice = test_pred_df.iloc[100:100 + 168].copy()
        ts = pd.to_datetime(sample_slice["timestamp"])

        ax.fill_between(
            ts,
            sample_slice["p10"],
            sample_slice["p90"],
            color="#aec7e8",
            alpha=0.6,
            label="P10 – P90 Interval (Nominal 80%)",
        )
        ax.plot(ts, sample_slice["p50"], color="#1f77b4", linestyle="--", linewidth=1.5, label="P50 Forecast (Median)")
        ax.plot(ts, sample_slice["point_pred"], color="#2ca02c", linestyle="-", linewidth=1.5, label="Point Model Forecast")
        ax.plot(ts, sample_slice["actual"], color="#d62728", linewidth=1.8, label="Actual Power (Ground Truth)")

        ax.set_title("ML-03: Next-Hour Forecast with P10–P90 Uncertainty Interval (1-Week Test Slice)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Timestamp", fontsize=11)
        ax.set_ylabel("Active Power (kW)", fontsize=11)
        ax.legend(loc="upper right", frameon=True)
        plt.tight_layout()
        fig.savefig(self.figures_dir / "one_week_interval_forecast.png")
        plt.close(fig)

        fig, ax1 = plt.subplots(figsize=(10, 4.5), dpi=150)
        ax2 = ax1.twinx()

        hours = diag_hour["hour"]
        ax1.plot(hours, diag_hour["coverage_pct"], color="#1f77b4", marker="o", linewidth=2, label="Observed Coverage (%)")
        ax1.axhline(80.0, color="#d62728", linestyle="--", linewidth=1.5, label="Nominal Target (80%)")
        ax2.bar(hours, diag_hour["mean_width_kw"], color="#ff7f0e", alpha=0.3, width=0.5, label="Mean Interval Width (kW)")

        ax1.set_xlabel("Hour of Day (0 – 23)", fontsize=11)
        ax1.set_ylabel("Empirical Coverage (%)", color="#1f77b4", fontsize=11)
        ax2.set_ylabel("Interval Width (kW)", color="#ff7f0e", fontsize=11)
        ax1.set_xticks(range(0, 24, 2))
        ax1.set_ylim(40, 100)
        ax1.set_title("ML-03: Diurnal Coverage & Interval Width across 24 Hours", fontsize=13, fontweight="bold")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", frameon=True)
        plt.tight_layout()
        fig.savefig(self.figures_dir / "interval_coverage_by_hour.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
        bars = ax.bar(diag_change["load_change_category"], diag_change["point_mae"], color="#9467bd", width=0.5)
        ax.set_title("ML-03: Point Forecast MAE by 1-Hour Load Change Magnitude", fontsize=13, fontweight="bold")
        ax.set_xlabel("Load Dynamics (|diff_1h| Category)", fontsize=11)
        ax.set_ylabel("MAE (kW)", fontsize=11)
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.01, f"{yval:.4f}", ha="center", va="bottom", fontsize=10)
        plt.tight_layout()
        fig.savefig(self.figures_dir / "load_dynamics_diagnostics.png")
        plt.close(fig)
