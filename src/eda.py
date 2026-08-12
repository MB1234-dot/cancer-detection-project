"""
Exploratory Data Analysis - Breast Cancer Wisconsin (Diagnostic) Dataset.

Run: python3 -m src.eda   (from project root)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src import config
from src.data import load_raw

logger = config.get_logger(__name__)


def main() -> None:
    config.ensure_dirs()
    df, feature_names = load_raw()

    logger.info("Shape: %s", df.shape)
    counts = df["diagnosis"].value_counts()
    pct = df["diagnosis"].value_counts(normalize=True) * 100
    for cls in counts.index:
        label = "malignant" if cls == 1 else "benign"
        logger.info("Class %s (%d): %d rows (%.1f%%)", label, cls, counts[cls], pct[cls])
    logger.info("Missing values: %d", df.isnull().sum().sum())

    plt.figure(figsize=(5, 4))
    sns.countplot(x="diagnosis", hue="diagnosis", data=df, palette=["#4C72B0", "#C44E52"], legend=False)
    plt.xticks([0, 1], ["Benign (0)", "Malignant (1)"])
    plt.title("Class Balance")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "class_balance.png", dpi=150)
    plt.close()

    mean_features = [f for f in feature_names if "mean" in f]
    plt.figure(figsize=(12, 10))
    corr = df[mean_features].corr()
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True, cbar_kws={"shrink": 0.7})
    plt.title("Correlation Between 'Mean' Features")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    sns.kdeplot(data=df, x="mean concave points", hue="diagnosis", fill=True,
                common_norm=False, palette={0: "#4C72B0", 1: "#C44E52"})
    plt.title("Mean Concave Points: Malignant vs Benign")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "concave_points_distribution.png", dpi=150)
    plt.close()

    df.to_csv(config.RAW_DATA_PATH, index=False)
    logger.info("Saved raw data to %s and figures to %s", config.RAW_DATA_PATH, config.FIGURES_DIR)
    logger.info(
        "Class balance is ~63/37 -- mild imbalance. Not extreme enough for "
        "resampling, but enough that accuracy alone would be misleading."
    )


if __name__ == "__main__":
    main()
