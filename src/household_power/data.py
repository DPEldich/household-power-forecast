"""Data loading, download, inspection and hourly aggregation module."""

from __future__ import annotations

import logging
import os
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

UCI_DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/235/"
    "individual+household+electric+power+consumption.zip"
)
ZIP_FILENAME = "individual+household+electric+power+consumption.zip"
TXT_FILENAME = "household_power_consumption.txt"


@dataclass
class RawDataStats:
    total_minute_rows: int
    missing_minute_readings: int
    duplicate_minute_timestamps: int
    first_minute_timestamp: str
    last_minute_timestamp: str
    total_hourly_intervals: int
    complete_target_hours: int
    incomplete_target_hours: int
    valid_target_ratio: float


def download_uci_dataset(dest_dir: str | Path = "data/raw", force: bool = False) -> Path:
    """Download the official UCI dataset zip file atomically and extract the raw txt file safely.

    Parameters
    ----------
    dest_dir : str | Path
        Directory where raw zip and txt files will be stored.
    force : bool
        If True, re-download even if already present.

    Returns
    -------
    Path
        Path to the extracted text file.
    """
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    zip_path = dest_path / ZIP_FILENAME
    tmp_zip_path = dest_path / (ZIP_FILENAME + ".tmp")
    txt_path = dest_path / TXT_FILENAME

    if not txt_path.exists() or force:
        if not zip_path.exists() or force:
            logger.info(f"Downloading UCI dataset from {UCI_DATASET_URL} to {zip_path} (atomic download)...")
            try:
                response = requests.get(UCI_DATASET_URL, stream=True, timeout=60)
                response.raise_for_status()
                with open(tmp_zip_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                # Atomically replace final zip once completely downloaded
                tmp_zip_path.replace(zip_path)
                logger.info("Download completed successfully.")
            except Exception as e:
                if tmp_zip_path.exists():
                    tmp_zip_path.unlink()
                raise RuntimeError(f"Failed to download UCI dataset: {e}") from e

        logger.info(f"Extracting expected file '{TXT_FILENAME}' safely from {zip_path}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            # Find and extract only the target file without path traversal
            target_member = None
            for member in zip_ref.infolist():
                if member.filename.startswith("/") or ".." in member.filename:
                    continue
                if Path(member.filename).name == TXT_FILENAME:
                    target_member = member
                    break

            if target_member is not None:
                zip_ref.extract(target_member, dest_path)
            else:
                raise FileNotFoundError(f"'{TXT_FILENAME}' not found inside archive {zip_path}")

    if not txt_path.exists():
        candidates = list(dest_path.glob("*.txt"))
        if candidates:
            txt_path = candidates[0]
        else:
            raise FileNotFoundError(f"Could not locate extracted raw text file in {dest_path}")

    return txt_path


def load_raw_minute_data(
    txt_path: str | Path,
) -> Tuple[pd.DataFrame, RawDataStats]:
    """Read raw UCI minute-level data memory-efficiently.

    Extracts Date, Time, and Global_active_power (kW). Handles '?' as NaN,
    parses timestamps dayfirst, detects duplicates, and compiles quality metrics.
    """
    path = Path(txt_path)
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path}")

    logger.info(f"Loading raw minute data from {path}...")
    # Use specific usecols and dtypes for memory efficiency
    df = pd.read_csv(
        path,
        sep=";",
        usecols=["Date", "Time", "Global_active_power"],
        na_values=["?", ""],
        dtype={"Global_active_power": "float32"},
        low_memory=False,
    )

    total_minute_rows = len(df)
    missing_minute_readings = int(df["Global_active_power"].isna().sum())

    # Build datetime timestamp
    logger.info("Parsing minute timestamps (format: DD/MM/YYYY HH:MM:SS)...")
    # String concatenation and pd.to_datetime
    datetime_str = df["Date"] + " " + df["Time"]
    df["timestamp"] = pd.to_datetime(
        datetime_str,
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )
    df = df.drop(columns=["Date", "Time"])

    # Check for invalid datetime strings
    null_timestamps = int(df["timestamp"].isna().sum())
    if null_timestamps > 0:
        logger.warning(f"Found {null_timestamps} invalid datetime rows. Dropping them.")
        df = df.dropna(subset=["timestamp"])

    # Check and handle duplicates
    duplicate_count = int(df["timestamp"].duplicated().sum())
    if duplicate_count > 0:
        logger.warning(f"Found {duplicate_count} duplicate timestamps. Dropping duplicates (keeping first).")
        df = df.drop_duplicates(subset=["timestamp"], keep="first")

    df = df.sort_values("timestamp").reset_index(drop=True)
    first_ts = str(df["timestamp"].iloc[0])
    last_ts = str(df["timestamp"].iloc[-1])

    logger.info(
        f"Loaded {len(df)} minute records from {first_ts} to {last_ts}. "
        f"Missing power readings: {missing_minute_readings} ({missing_minute_readings / total_minute_rows:.2%})"
    )

    # Placeholder stats; complete_target_hours calculated in resample_hourly
    stats = RawDataStats(
        total_minute_rows=total_minute_rows,
        missing_minute_readings=missing_minute_readings,
        duplicate_minute_timestamps=duplicate_count,
        first_minute_timestamp=first_ts,
        last_minute_timestamp=last_ts,
        total_hourly_intervals=0,
        complete_target_hours=0,
        incomplete_target_hours=0,
        valid_target_ratio=0.0,
    )
    return df, stats


def resample_hourly(
    df_minute: pd.DataFrame,
    stats: Optional[RawDataStats] = None,
) -> Tuple[pd.DataFrame, RawDataStats]:
    """Aggregate minute records into hourly intervals [s, s+1h) left-labeled.

    Strict rules applied:
    - Continuous hourly index preserved from start to end without missing intervals.
    - An hour s is a valid target y_s iff it contains exactly 60 valid (non-missing) minute observations.
    - Never fill/synthesize truth labels for scoring.
    - Historical inputs causally bounded: raw hourly power is stored, and missing inputs
      can be causally bounded ffilled (max 3h).
    """
    logger.info("Resampling minute data to left-labeled continuous hourly series...")
    df_indexed = df_minute.set_index("timestamp")

    # Resample left-labeled: label='left', closed='left'
    resampled = df_indexed["Global_active_power"].resample(
        "1h",
        closed="left",
        label="left",
    ).agg(["count", "mean"])

    resampled = resampled.rename(columns={"count": "valid_minutes", "mean": "hourly_mean"})

    # Ensure continuous hourly index
    full_hourly_index = pd.date_range(
        start=resampled.index.min(),
        end=resampled.index.max(),
        freq="1h",
        name="timestamp",
    )
    hourly_df = resampled.reindex(full_hourly_index)
    hourly_df["valid_minutes"] = hourly_df["valid_minutes"].fillna(0).astype(int)

    # Target definition: strictly requiring 60 valid minutes
    hourly_df["is_valid_target"] = hourly_df["valid_minutes"] == 60
    hourly_df["target"] = np.where(hourly_df["is_valid_target"], hourly_df["hourly_mean"], np.nan)

    # Raw input active power: strictly hourly mean when 60 valid minutes exist, else NaN
    hourly_df["raw_active_power"] = np.where(
        hourly_df["is_valid_target"],
        hourly_df["hourly_mean"],
        np.nan,
    )

    # Causal bounded forward fill (max 3 hours) for historical inputs only
    hourly_df["active_power_ffill3"] = hourly_df["raw_active_power"].ffill(limit=3)

    total_hourly = len(hourly_df)
    valid_targets = int(hourly_df["is_valid_target"].sum())
    incomplete_targets = total_hourly - valid_targets
    valid_ratio = valid_targets / total_hourly if total_hourly > 0 else 0.0

    if stats is None:
        stats = RawDataStats(
            total_minute_rows=len(df_minute),
            missing_minute_readings=0,
            duplicate_minute_timestamps=0,
            first_minute_timestamp=str(hourly_df.index[0]),
            last_minute_timestamp=str(hourly_df.index[-1]),
            total_hourly_intervals=total_hourly,
            complete_target_hours=valid_targets,
            incomplete_target_hours=incomplete_targets,
            valid_target_ratio=valid_ratio,
        )
    else:
        stats.total_hourly_intervals = total_hourly
        stats.complete_target_hours = valid_targets
        stats.incomplete_target_hours = incomplete_targets
        stats.valid_target_ratio = valid_ratio

    logger.info(
        f"Hourly aggregation complete: {total_hourly} total hours. "
        f"Valid targets (60 non-missing mins): {valid_targets} ({valid_ratio:.2%}). "
        f"Incomplete targets: {incomplete_targets}."
    )

    return hourly_df, stats


def prepare_and_cache_hourly_data(
    raw_dir: str | Path = "data/raw",
    processed_dir: str | Path = "data/processed",
    force_download: bool = False,
    force_process: bool = False,
) -> Tuple[pd.DataFrame, RawDataStats]:
    """Download raw data if needed, aggregate hourly, and cache to disk.

    If force_download is True, processed cache is automatically invalidated.
    """
    proc_path = Path(processed_dir)
    proc_path.mkdir(parents=True, exist_ok=True)
    hourly_parquet = proc_path / "hourly_power.parquet"
    stats_json = proc_path / "data_stats.json"

    # Force download must invalidate downstream processed cache
    if force_download:
        force_process = True

    import json

    if hourly_parquet.exists() and stats_json.exists() and not force_process:
        logger.info(f"Loading cached hourly data from {hourly_parquet}...")
        hourly_df = pd.read_parquet(hourly_parquet)
        with open(stats_json, "r", encoding="utf-8") as f:
            stats_dict = json.load(f)
        stats = RawDataStats(**stats_dict)
        return hourly_df, stats

    txt_path = download_uci_dataset(raw_dir, force=force_download)
    df_minute, stats = load_raw_minute_data(txt_path)
    hourly_df, stats = resample_hourly(df_minute, stats)

    # Cache processed dataset
    logger.info(f"Caching hourly dataset to {hourly_parquet}...")
    hourly_df.to_parquet(hourly_parquet)
    with open(stats_json, "w", encoding="utf-8") as f:
        json.dump(asdict(stats), f, indent=2)

    return hourly_df, stats
