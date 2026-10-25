"""Tests for strict chronological splitting and expanding cross-validation."""

from __future__ import annotations

import pandas as pd
import pytest

from household_power.splits import compute_chronological_splits, get_expanding_cv_folds


def test_chronological_split_strict_ordering():
    """Verify train is strictly before val, and val strictly before test."""
    # Create 1000 hours
    timestamps = pd.date_range("2020-01-01 00:00:00", periods=1000, freq="1h")
    # Simulate some invalid targets interspersed
    valid_mask = pd.Series(True, index=timestamps)
    valid_mask.iloc[10:20] = False
    valid_mask.iloc[720:730] = False

    df = pd.DataFrame(index=timestamps)

    splits_meta, train_mask, val_mask, test_mask, first85_mask = compute_chronological_splits(
        df, valid_mask, warmup_hours=168, train_ratio=0.70, val_ratio=0.15
    )

    t_tr = timestamps[train_mask]
    t_va = timestamps[val_mask]
    t_te = timestamps[test_mask]

    # Non-empty
    assert len(t_tr) > 0
    assert len(t_va) > 0
    assert len(t_te) > 0

    # Strict temporal ordering: max(train) < min(val)
    assert t_tr.max() < t_va.min()
    # Strict temporal ordering: max(val) < min(test)
    assert t_va.max() < t_te.min()

    # First 85% must equal union of train and val
    assert (train_mask | val_mask).equals(first85_mask)


def test_expanding_cv_folds_within_train():
    """Verify that expanding CV folds are strictly monotonic and contained within train."""
    timestamps = pd.date_range("2020-01-01 00:00:00", periods=1000, freq="1h")
    valid_mask = pd.Series(True, index=timestamps)
    df = pd.DataFrame(index=timestamps)

    splits_meta, train_mask, _, _, _ = compute_chronological_splits(
        df, valid_mask, warmup_hours=168, train_ratio=0.70, val_ratio=0.15
    )

    folds = get_expanding_cv_folds(df, valid_mask, splits_meta.train_boundary, n_folds=3)
    assert len(folds) == 3

    prev_train_count = 0
    for f_idx, (f_tr_m, f_va_m, f_meta) in enumerate(folds):
        f_tr_ts = timestamps[f_tr_m]
        f_va_ts = timestamps[f_va_m]

        # Each fold has expanding training set
        assert len(f_tr_ts) > prev_train_count
        prev_train_count = len(f_tr_ts)

        # Within fold: train strictly precedes validation
        assert f_tr_ts.max() < f_va_ts.min()

        # Both train and val are strictly inside global train boundary
        train_start = pd.Timestamp(splits_meta.train_boundary.start_timestamp)
        train_end = pd.Timestamp(splits_meta.train_boundary.end_timestamp)
        assert f_tr_ts.min() >= train_start
        assert f_va_ts.max() <= train_end
