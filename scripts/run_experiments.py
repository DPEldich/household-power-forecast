"""CLI script to execute the ML-03 causal load-dynamics experiment and uncertainty evaluation."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from household_power.experiments import ExperimentRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_experiments")


def main():
    parser = argparse.ArgumentParser(
        description="Run ML-03 load dynamics and quantile uncertainty experiments."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=ROOT / "data/raw",
        help="Path to raw UCI data directory",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=ROOT / "data/processed",
        help="Path to processed hourly data directory",
    )
    parser.add_argument(
        "--baseline-artifacts-dir",
        type=Path,
        default=ROOT / "artifacts",
        help="Path to baseline ML-01 artifacts",
    )
    parser.add_argument(
        "--experimental-dir",
        type=Path,
        default=ROOT / "artifacts/experimental/ml03",
        help="Path to ML-03 experimental artifacts destination",
    )

    args = parser.parse_args()

    logger.info(f"Root directory: {ROOT}")
    logger.info(f"Target experimental directory: {args.experimental_dir}")

    runner = ExperimentRunner(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        baseline_artifacts_dir=args.baseline_artifacts_dir,
        experimental_dir=args.experimental_dir,
    )

    results = runner.run()

    print("\n" + "=" * 60)
    print("           ML-03 EXPERIMENT RESULTS SUMMARY")
    print("=" * 60)
    print(f"CV Selected Candidate:    {results['cv_selected_candidate']}")
    print(f"Validation Confirmation:  {'PASSED' if results['validation_confirmed'] else 'REVERTED TO BASELINE'}")
    print(f"Final Promoted Model:     {results['promoted_model']}")
    print("-" * 60)
    print("Test Point Performance:")
    print(results["test_point_df"].to_string(index=False))
    print("-" * 60)
    print("Test Uncertainty Performance (Nominal 80% [P10, P90]):")
    unc_test = results["uncertainty_df"][results["uncertainty_df"]["split"] == "test"]
    print(unc_test[["subset", "count", "coverage_pct", "undercoverage_pct", "overcoverage_pct", "mean_width_kw", "mean_pinball"]].to_string(index=False))
    print("=" * 60)
    print(f"Artifacts persisted under: {args.experimental_dir}\n")


if __name__ == "__main__":
    main()
