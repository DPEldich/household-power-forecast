"""Chronological split and expanding temporal cross-validation module."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SplitBoundary:
    name: str
    start_timestamp: str
    end_timestamp: str
    total_hours: int
    valid_target_hours: int


@dataclass
class DatasetSplits:
    train_boundary: SplitBoundary
    val_boundary: SplitBoundary
    test_boundary: SplitBoundary
    first85_boundary: SplitBoundary
    total_eval_hours: int
    warmup_hours: int


@dataclass
class CVFoldBoundary:
    fold: int
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    train_total_hours: int
    train_valid_hours: int
    val_total_hours: int
    val_valid_hours: int


def compute_chronological_splits(
    df: pd.DataFrame,
    valid_mask: pd.Series,
    warmup_hours: int = 168,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[DatasetSplits, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Compute strictly chronological 70% train, 15% val, 15% test boundaries.

    Folds/boundaries are defined by calendar hourly intervals before filtering
    for valid targets.

    Returns
    -------
    splits_meta : DatasetSplits
    train_mask : pd.Series (bool)
    val_mask : pd.Series (bool)
    test_mask : pd.Series (bool)
    first85_mask : pd.Series (bool)
        Union of train and val (first 85% used for refitting before test).
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Index must be DatetimeIndex.")

    # Exclude initial warmup
    usable_index = df.index[df.index >= df.index[0] + pd.Timedelta(hours=warmup_hours)]
    total_usable = len(usable_index)

    n_train = int(np.floor(train_ratio * total_usable))
    n_val = int(np.floor(val_ratio * total_usable))
    n_test = total_usable - n_train - n_val

    # Exact calendar interval slices
    train_idx = usable_index[:n_train]
    val_idx = usable_index[n_train : n_train + n_val]
    test_idx = usable_index[n_train + n_val :]
    first85_idx = usable_index[: n_train + n_val]

    # Timestamp boundaries
    train_start, train_end = str(train_idx[0]), str(train_idx[-1])
    val_start, val_end = str(val_idx[0]), str(val_idx[-1])
    test_start, test_end = str(test_idx[0]), str(test_idx[-1])
    first85_start, first85_end = str(first85_idx[0]), str(first85_idx[-1])

    # Boolean interval masks across the whole dataframe
    train_interval = (df.index >= train_idx[0]) & (df.index <= train_idx[-1])
    val_interval = (df.index >= val_idx[0]) & (df.index <= val_idx[-1])
    test_interval = (df.index >= test_idx[0]) & (df.index <= test_idx[-1])
    first85_interval = (df.index >= first85_idx[0]) & (df.index <= first85_idx[-1])

    # Masks requiring both belonging to interval AND having valid target
    train_mask = train_interval & valid_mask
    val_mask = val_interval & valid_mask
    test_mask = test_interval & valid_mask
    first85_mask = first85_interval & valid_mask

    splits_meta = DatasetSplits(
        train_boundary=SplitBoundary(
            name="train_70",
            start_timestamp=train_start,
            end_timestamp=train_end,
            total_hours=len(train_idx),
            valid_target_hours=int(train_mask.sum()),
        ),
        val_boundary=SplitBoundary(
            name="val_15",
            start_timestamp=val_start,
            end_timestamp=val_end,
            total_hours=len(val_idx),
            valid_target_hours=int(val_mask.sum()),
        ),
        test_boundary=SplitBoundary(
            name="test_15",
            start_timestamp=test_start,
            end_timestamp=test_end,
            total_hours=len(test_idx),
            valid_target_hours=int(test_mask.sum()),
        ),
        first85_boundary=SplitBoundary(
            name="first_85_refit",
            start_timestamp=first85_start,
            end_timestamp=first85_end,
            total_hours=len(first85_idx),
            valid_target_hours=int(first85_mask.sum()),
        ),
        total_eval_hours=total_usable,
        warmup_hours=warmup_hours,
    )

    logger.info(
        f"Chronological split boundaries:\n"
        f"  Train (70%): {train_start} to {train_end} ({splits_meta.train_boundary.total_hours}h total, {splits_meta.train_boundary.valid_target_hours} valid)\n"
        f"  Val   (15%): {val_start} to {val_end} ({splits_meta.val_boundary.total_hours}h total, {splits_meta.val_boundary.valid_target_hours} valid)\n"
        f"  Test  (15%): {test_start} to {test_end} ({splits_meta.test_boundary.total_hours}h total, {splits_meta.test_boundary.valid_target_hours} valid)"
    )

    return splits_meta, train_mask, val_mask, test_mask, first85_mask


def get_expanding_cv_folds(
    df: pd.DataFrame,
    valid_mask: pd.Series,
    train_boundary: SplitBoundary,
    n_folds: int = 3,
) -> List[Tuple[pd.Series, pd.Series, CVFoldBoundary]]:
    """Generate n_folds expanding temporal CV splits within the initial train set.

    Dividing train into (n_folds + 1) chronological blocks:
      Fold 1: train Block 1, validate Block 2
      Fold 2: train Blocks 1..2, validate Block 3
      Fold 3: train Blocks 1..3, validate Block 4
    """
    train_start_ts = pd.Timestamp(train_boundary.start_timestamp)
    train_end_ts = pd.Timestamp(train_boundary.end_timestamp)

    train_usable_idx = df.index[(df.index >= train_start_ts) & (df.index <= train_end_ts)]
    M = len(train_usable_idx)
    block_size = M // (n_folds + 1)

    folds = []
    for f in range(n_folds):
        train_end_idx = (f + 1) * block_size
        val_start_idx = train_end_idx
        val_end_idx = (f + 2) * block_size if f < n_folds - 1 else M

        fold_train_idx = train_usable_idx[:train_end_idx]
        fold_val_idx = train_usable_idx[val_start_idx:val_end_idx]

        f_train_interval = (df.index >= fold_train_idx[0]) & (df.index <= fold_train_idx[-1])
        f_val_interval = (df.index >= fold_val_idx[0]) & (df.index <= fold_val_idx[-1])

        f_train_mask = f_train_interval & valid_mask
        f_val_mask = f_val_interval & valid_mask

        meta = CVFoldBoundary(
            fold=f + 1,
            train_start=str(fold_train_idx[0]),
            train_end=str(fold_train_idx[-1]),
            val_start=str(fold_val_idx[0]),
            val_end=str(fold_val_idx[-1]),
            train_total_hours=len(fold_train_idx),
            train_valid_hours=int(f_train_mask.sum()),
            val_total_hours=len(fold_val_idx),
            val_valid_hours=int(f_val_mask.sum()),
        )
        folds.append((f_train_mask, f_val_mask, meta))

        logger.info(
            f"CV Fold {f+1}: Train [{meta.train_start} -> {meta.train_end}] ({meta.train_valid_hours} valid) | "
            f"Val [{meta.val_start} -> {meta.val_end}] ({meta.val_valid_hours} valid)"
        )

    return folds
