"""
Tests for data loading and splitting -- the part of the pipeline most prone
to silent, hard-to-notice bugs (a leak doesn't crash anything, it just
quietly makes your numbers wrong).
"""
import pandas as pd
import pytest

from src.data import load_raw, make_splits, TARGET_COL


@pytest.fixture(scope="module")
def raw_data():
    df, feature_names = load_raw()
    return df, feature_names


def test_raw_data_shape(raw_data):
    df, feature_names = raw_data
    assert df.shape[0] == 569
    assert len(feature_names) == 30
    assert TARGET_COL in df.columns


def test_no_missing_values(raw_data):
    df, _ = raw_data
    assert df.isnull().sum().sum() == 0


def test_target_is_binary(raw_data):
    df, _ = raw_data
    assert set(df[TARGET_COL].unique()) == {0, 1}


def test_target_convention_matches_known_class_balance(raw_data):
    """Sanity check the 0/1 remap didn't get flipped back by accident.

    This dataset is well known to be ~62.7% benign / ~37.3% malignant. If
    someone edits data.py and the mapping silently inverts, every downstream
    recall/precision number would mean the opposite of what it claims to --
    this test exists specifically to catch that class of bug.
    """
    df, _ = raw_data
    malignant_fraction = df[TARGET_COL].mean()
    assert 0.30 < malignant_fraction < 0.45


def test_splits_do_not_overlap(raw_data):
    df, _ = raw_data
    train_df, val_df, test_df = make_splits(df)
    combined = pd.concat([train_df, val_df, test_df])
    assert combined.drop_duplicates().shape[0] == len(combined)


def test_splits_cover_full_dataset(raw_data):
    df, _ = raw_data
    train_df, val_df, test_df = make_splits(df)
    assert len(train_df) + len(val_df) + len(test_df) == len(df)


def test_splits_are_stratified(raw_data):
    df, _ = raw_data
    train_df, val_df, test_df = make_splits(df)
    overall = df[TARGET_COL].mean()
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        split_rate = split_df[TARGET_COL].mean()
        assert abs(split_rate - overall) < 0.05, f"{name} split class balance drifted too far from overall"


def test_splits_are_reproducible(raw_data):
    df, _ = raw_data
    a1, a2, a3 = make_splits(df, random_state=42)
    b1, b2, b3 = make_splits(df, random_state=42)
    pd.testing.assert_frame_equal(a1, b1)
    pd.testing.assert_frame_equal(a2, b2)
    pd.testing.assert_frame_equal(a3, b3)


def test_different_seeds_give_different_splits(raw_data):
    df, _ = raw_data
    a1, _, _ = make_splits(df, random_state=42)
    b1, _, _ = make_splits(df, random_state=7)
    assert not a1.equals(b1)
