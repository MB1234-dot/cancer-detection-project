"""
Single source of truth for loading and splitting the data. Every other
script (feature analysis, training, evaluation, SHAP, the app) imports from
here instead of re-implementing its own train_test_split call -- that
duplication is exactly how a project ends up with two slightly different
splits floating around and nobody noticing.
"""
from typing import List, Tuple

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from src import config

TARGET_COL = "diagnosis"


def load_raw() -> Tuple[pd.DataFrame, List[str]]:
    """Load the Wisconsin Breast Cancer dataset with a corrected target.

    sklearn's built-in encoding is 0=malignant, 1=benign -- the opposite of
    the usual "positive class = disease present" medical ML convention. We
    flip it here, once, so every downstream script inherits the correct
    convention automatically instead of everyone needing to remember to flip
    it themselves.
    """
    raw = load_breast_cancer(as_frame=True)
    df = raw.frame.copy()
    df[TARGET_COL] = df["target"].map({0: 1, 1: 0})  # 1 = malignant, 0 = benign
    df = df.drop(columns=["target"])
    feature_names = list(raw.feature_names)
    return df, feature_names


def make_splits(
    df: pd.DataFrame,
    test_size: float = config.TEST_SIZE,
    val_size: float = config.VAL_SIZE,
    random_state: int = config.RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train / validation / test split.

    `val_size` is the fraction of the *non-test* remainder, not of the full
    dataset -- e.g. test_size=0.2, val_size=0.2 yields roughly a 64/16/20
    train/val/test split of the original data, not 60/20/20.

    Returns three DataFrames, each still containing the target column.
    """
    y = df[TARGET_COL]
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, stratify=y, random_state=random_state
    )
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size,
        stratify=train_val_df[TARGET_COL],
        random_state=random_state,
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def load_splits() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load previously-saved train/val/test CSVs (see split_data.py)."""
    train_df = pd.read_csv(config.TRAIN_PATH)
    val_df = pd.read_csv(config.VAL_PATH)
    test_df = pd.read_csv(config.TEST_PATH)
    return train_df, val_df, test_df


def load_selected_features() -> List[str]:
    """Load the feature list chosen by the VIF multicollinearity pass."""
    import json

    with open(config.SELECTED_FEATURES_PATH) as f:
        return json.load(f)["selected_features"]
