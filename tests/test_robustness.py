"""Regression and robustness tests for index validation and safe caching."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from household_power.data import download_uci_dataset, prepare_and_cache_hourly_data
from household_power.features import build_features


def test_build_features_rejects_discontinuous_index():
    """Verify build_features fails immediately if timestamps skip an hour."""
    # 00:00, 01:00, then jumps to 03:00 (missing 02:00)
    timestamps = pd.to_datetime(["2020-01-01 00:00:00", "2020-01-01 01:00:00", "2020-01-01 03:00:00"])
    df = pd.DataFrame({
        "active_power_ffill3": [1.0, 2.0, 3.0],
        "target": [1.0, 2.0, 3.0],
    }, index=timestamps)

    with pytest.raises(ValueError, match="exact 1-hour spacing"):
        build_features(df)


def test_build_features_rejects_non_monotonic_index():
    """Verify build_features fails immediately if timestamps are not monotonically increasing."""
    timestamps = pd.to_datetime(["2020-01-01 01:00:00", "2020-01-01 00:00:00"])
    df = pd.DataFrame({
        "active_power_ffill3": [1.0, 2.0],
        "target": [1.0, 2.0],
    }, index=timestamps)

    with pytest.raises(ValueError, match="strictly monotonic increasing"):
        build_features(df)


def test_build_features_rejects_duplicate_index():
    """Verify build_features fails immediately if timestamps contain duplicates."""
    timestamps = pd.to_datetime(["2020-01-01 00:00:00", "2020-01-01 00:00:00"])
    df = pd.DataFrame({
        "active_power_ffill3": [1.0, 2.0],
        "target": [1.0, 2.0],
    }, index=timestamps)

    with pytest.raises(ValueError, match="unique timestamps"):
        build_features(df)


def test_force_download_invalidates_cache(tmp_path: Path, monkeypatch):
    """Verify that calling prepare_and_cache_hourly_data with force_download=True does not reuse old cache."""
    processed_dir = tmp_path / "processed"
    raw_dir = tmp_path / "raw"
    processed_dir.mkdir(parents=True)
    raw_dir.mkdir(parents=True)

    dummy_parquet = processed_dir / "hourly_power.parquet"
    dummy_stats = processed_dir / "data_stats.json"

    # Pre-create stale dummy cache
    dummy_parquet.write_text("dummy")
    dummy_stats.write_text("{}")

    download_called = False

    def mock_download(dest_dir, force=False):
        nonlocal download_called
        download_called = True
        txt_path = Path(dest_dir) / "household_power_consumption.txt"
        txt_path.write_text("Date;Time;Global_active_power\n16/12/2006;17:24:00;4.216\n")
        return txt_path

    monkeypatch.setattr("household_power.data.download_uci_dataset", mock_download)

    # When force_download is True, stale cache must NOT be used
    try:
        prepare_and_cache_hourly_data(
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            force_download=True,
            force_process=False,
        )
    except Exception:
        pass  # Just testing that download was invoked and cache was bypassed

    assert download_called is True
