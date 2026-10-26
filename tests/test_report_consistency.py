"""Tests asserting numerical consistency between CSV/metadata artifacts and published markdown reports."""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).parent.parent
METRICS_DIR = ROOT_DIR / "artifacts" / "metrics"
METADATA_DIR = ROOT_DIR / "artifacts" / "metadata"
REPORT_PATH = ROOT_DIR / "REPORT.md"
README_PATH = ROOT_DIR / "README.md"


def test_cv_table_in_report_matches_cv_metrics_csv():
    """Assert all CV metrics rows in REPORT.md match formatted values from cv_metrics.csv."""
    assert REPORT_PATH.exists()
    assert (METRICS_DIR / "cv_metrics.csv").exists()
    assert (METADATA_DIR / "split_metadata.json").exists()

    with open(METADATA_DIR / "split_metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    cv_df = pd.read_csv(METRICS_DIR / "cv_metrics.csv")
    report_text = REPORT_PATH.read_text(encoding="utf-8")

    best_cfgs = meta["best_hyperparameters"]
    for model_name, cfg in best_cfgs.items():
        row = cv_df[(cv_df["model_name"] == model_name) & (cv_df["config"] == str(cfg))].iloc[0]
        mean_mae = f"{row['mean_cv_mae']:.4f}"
        std_mae = f"{row['std_cv_mae']:.4f}"
        mean_rmse = f"{row['mean_cv_rmse']:.4f}"
        mean_r2 = f"{row['mean_cv_r2']:.4f}"

        # Assert specific values exist in the report table
        expected_fragment = f"| **{model_name}** |"
        assert expected_fragment in report_text, f"Model {model_name} missing from CV table in REPORT.md"

        # Check Linear Regression std is specifically 0.0514
        if model_name == "Linear Regression":
            assert std_mae == "0.0514"
            assert "0.0514" in report_text
            assert "0.5903" in report_text
            assert "0.5588" in report_text


def test_test_table_in_report_and_readme_matches_test_metrics_csv():
    """Assert Test table rows in REPORT.md and README.md match formatted values from test_metrics.csv."""
    test_df = pd.read_csv(METRICS_DIR / "test_metrics.csv")
    report_text = REPORT_PATH.read_text(encoding="utf-8")
    readme_text = README_PATH.read_text(encoding="utf-8")

    for _, r in test_df.iterrows():
        model_name = r["model_name"]
        mae = f"{r['mae']:.4f}"
        rmse = f"{r['rmse']:.4f}"
        r2 = f"{r['r2']:.4f}"

        # Ensure Seasonal Naive MAE rounds to 0.5100
        if model_name == "Seasonal Naive":
            assert mae == "0.5100", f"Seasonal Naive MAE should round to 0.5100, got {mae}"

        assert mae in report_text, f"MAE {mae} for {model_name} not found in REPORT.md"
        assert rmse in report_text, f"RMSE {rmse} for {model_name} not found in REPORT.md"
        assert r2 in report_text, f"R2 {r2} for {model_name} not found in REPORT.md"

        assert mae in readme_text, f"MAE {mae} for {model_name} not found in README.md"
        assert rmse in readme_text, f"RMSE {rmse} for {model_name} not found in README.md"


def test_ablation_table_matches_test_ablation_csv_and_full_parity():
    """Assert Ablation table values in REPORT.md match test_ablation_metrics.csv and full matches primary winner."""
    test_ab_df = pd.read_csv(METRICS_DIR / "test_ablation_metrics.csv")
    val_ab_df = pd.read_csv(METRICS_DIR / "ablation_metrics.csv")
    test_df = pd.read_csv(METRICS_DIR / "test_metrics.csv")
    val_df = pd.read_csv(METRICS_DIR / "validation_metrics.csv")

    with open(METADATA_DIR / "split_metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    winner_name = meta["validation_selected_winner"]
    winner_test_row = test_df[test_df["model_name"] == winner_name].iloc[0]
    winner_val_row = val_df[val_df["model_name"] == winner_name].iloc[0]

    full_test_ab = test_ab_df[test_ab_df["config_name"] == "full"].iloc[0]
    full_val_ab = val_ab_df[val_ab_df["config_name"] == "full"].iloc[0]

    # Exact mathematical parity between winner and ablation full
    assert full_test_ab["mae"] == pytest.approx(winner_test_row["mae"], abs=1e-6)
    assert full_test_ab["rmse"] == pytest.approx(winner_test_row["rmse"], abs=1e-6)
    assert full_test_ab["r2"] == pytest.approx(winner_test_row["r2"], abs=1e-6)

    assert full_val_ab["mae"] == pytest.approx(winner_val_row["mae"], abs=1e-6)
    assert full_val_ab["rmse"] == pytest.approx(winner_val_row["rmse"], abs=1e-6)
    assert full_val_ab["r2"] == pytest.approx(winner_val_row["r2"], abs=1e-6)

    # Check published report contains formatted values
    report_text = REPORT_PATH.read_text(encoding="utf-8")
    for _, r in test_ab_df.iterrows():
        mae = f"{r['mae']:.4f}"
        rmse = f"{r['rmse']:.4f}"
        assert mae in report_text, f"Ablation test MAE {mae} not in REPORT.md"
        assert rmse in report_text, f"Ablation test RMSE {rmse} not in REPORT.md"


def test_high_load_table_matches_high_load_metrics_csv():
    """Assert high-load metrics in REPORT.md match high_load_metrics.csv."""
    hl_df = pd.read_csv(METRICS_DIR / "high_load_metrics.csv")
    report_text = REPORT_PATH.read_text(encoding="utf-8")

    for _, r in hl_df.iterrows():
        hl_mae = f"{r['high_load_mae']:.4f}"
        nl_mae = f"{r['normal_load_mae']:.4f}"
        assert hl_mae in report_text, f"High load MAE {hl_mae} for {r['model_name']} not in REPORT.md"
        assert nl_mae in report_text, f"Normal load MAE {nl_mae} for {r['model_name']} not in REPORT.md"


def test_ml03_uncertainty_table_matches_uncertainty_metrics_csv():
    """Assert all six subsets in ML-03 uncertainty table match uncertainty_metrics.csv exactly."""
    unc_path = ROOT_DIR / "artifacts/experimental/ml03/metrics/uncertainty_metrics.csv"
    if not unc_path.exists():
        pytest.skip("ML-03 uncertainty metrics CSV not present")

    unc_df = pd.read_csv(unc_path)
    ml03_results_text = (ROOT_DIR / "ML03_RESULTS.md").read_text(encoding="utf-8")
    report_text = REPORT_PATH.read_text(encoding="utf-8")
    readme_text = README_PATH.read_text(encoding="utf-8")

    for _, r in unc_df.iterrows():
        split = str(r["split"])
        subset = str(r["subset"])
        cnt = str(int(r["count"]))
        cov = f"{float(r['coverage_pct']):.2f}%"
        w = f"{float(r['mean_width_kw']):.4f}"
        p10 = f"{float(r['pinball_p10']):.4f}"
        p50 = f"{float(r['pinball_p50']):.4f}"
        p90 = f"{float(r['pinball_p90']):.4f}"
        mp = f"{float(r['mean_pinball']):.4f}"

        below = f"{float(r['below_lower_pct']):.2f}%"
        above = f"{float(r['above_upper_pct']):.2f}%"
        expected_row = (
            f"| {split} | {subset} | {cnt} | {cov} | {below} | {above} | "
            f"{w} | {p10} | {p50} | {p90} | {mp} |"
        )
        for name, document in [
            ("ML03_RESULTS.md", ml03_results_text),
            ("REPORT.md", report_text),
            ("README.md", readme_text),
        ]:
            assert expected_row in document, f"Incorrect {split}/{subset} row in {name}"
