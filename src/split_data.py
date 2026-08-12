"""
Create the train / validation / test split and persist it to disk.

This is its own pipeline stage, deliberately separate from training, so that
every later script (feature analysis, training, evaluation, SHAP, the app)
reads the *same* three files instead of each re-deriving its own split --
which is exactly how the earlier version of this project ended up reusing
the test set for both threshold selection and final reporting: two scripts
each called train_test_split independently with the same random_state, so
they silently produced the same test set without that being an explicit,
visible contract anywhere. Now there's exactly one split, computed once,
saved once, and everything downstream just loads it.

Run: python3 -m src.split_data   (from project root)
"""
from src import config
from src.data import load_raw, make_splits

logger = config.get_logger(__name__)


def main() -> None:
    config.ensure_dirs()
    df, _ = load_raw()
    train_df, val_df, test_df = make_splits(df)

    train_df.to_csv(config.TRAIN_PATH, index=False)
    val_df.to_csv(config.VAL_PATH, index=False)
    test_df.to_csv(config.TEST_PATH, index=False)

    total = len(df)
    for name, split_df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        malignant_pct = split_df["diagnosis"].mean() * 100
        logger.info(
            "%-10s: %4d rows (%.1f%% of total) | %.1f%% malignant",
            name, len(split_df), 100 * len(split_df) / total, malignant_pct,
        )

    # sanity check: the three splits must not share any rows. We can't compare
    # original dataframe indices (they were reset on save), so we compare on
    # content instead -- concatenate all three, and if row-level duplicates
    # show up across splits, something is badly wrong with the split logic.
    import pandas as pd
    combined = pd.concat([train_df, val_df, test_df])
    n_unique = combined.drop_duplicates().shape[0]
    assert n_unique == len(combined), (
        "Overlapping rows detected across train/val/test splits -- this "
        "would silently leak information between stages."
    )
    logger.info("Verified: no overlapping rows across train/val/test (%d total).", len(combined))


if __name__ == "__main__":
    main()
