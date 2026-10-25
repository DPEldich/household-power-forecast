"""Command Line Interface for running the ML-01 Electricity Forecasting Pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure src is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from household_power.pipeline import PipelineRunner


def main():
    parser = argparse.ArgumentParser(description="Run ML-01 Household Power Forecasting Pipeline")
    parser.add_argument("--raw-dir", default="data/raw", help="Directory for raw data zip and txt")
    parser.add_argument("--processed-dir", default="data/processed", help="Directory for processed hourly data")
    parser.add_argument("--artifacts-dir", default="artifacts", help="Directory for model artifacts and metrics")
    parser.add_argument("--force-download", action="store_true", help="Force re-downloading dataset from UCI")
    parser.add_argument("--force-process", action="store_true", help="Force re-aggregating minute dataset")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    runner = PipelineRunner(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        artifacts_dir=args.artifacts_dir,
        force_download=args.force_download,
        force_process=args.force_process,
    )

    results = runner.run()
    winner = results["validation_winner"]
    print(f"\nPipeline finished successfully! Validation winner: {winner}")
    print("\nTest Set Evaluation Results:")
    print(results["test_metrics"].to_string(index=False))


if __name__ == "__main__":
    main()
