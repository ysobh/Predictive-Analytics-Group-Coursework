import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RANDOM_STATE = 42
CSV_PATH = Path("titanic_clean.csv")


def format_percentage(series: pd.Series) -> pd.Series:
    return (series * 100).round(1)


def build_sex_labels(sex_series: pd.Series) -> pd.Series:
    sex_map = {0: "male", 1: "female", "0": "male", "1": "female"}
    return sex_series.map(sex_map).fillna(sex_series.astype(str))


def print_dataset_summary(df: pd.DataFrame) -> None:
    print("Dataset shape:")
    print(df.shape)
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nData types:")
    print(df.dtypes)
    print("\nDescribe:")
    print(df.describe(include="all"))


def print_survival_rates(df: pd.DataFrame) -> None:
    sex_labels = build_sex_labels(df["Sex"])
    sex_rates = format_percentage(df.groupby(sex_labels)["Survived"].mean())
    pclass_rates = format_percentage(df.groupby("Pclass")["Survived"].mean())

    print("\nSurvival rate by Sex:")
    for label in ["male", "female"]:
        if label in sex_rates.index:
            print(f"{label}: {sex_rates[label]:.1f}%")

    print("\nSurvival rate by Pclass:")
    class_labels = {1: "1st", 2: "2nd", 3: "3rd"}
    for pclass in [1, 2, 3]:
        if pclass in pclass_rates.index:
            print(f"{class_labels[pclass]}: {pclass_rates[pclass]:.1f}%")


def save_survival_count_plot(df: pd.DataFrame) -> None:
    counts = df["Survived"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.index.astype(str), counts.values, color=["#d95f02", "#1b9e77"])
    ax.set_title("Survival Counts")
    ax.set_xlabel("Survived")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig("eda_survival_count.png", dpi=150)
    plt.close(fig)
    print("Saved eda_survival_count.png")


def save_age_distribution_plot(df: pd.DataFrame) -> None:
    survived_groups = {
        0: ("Did not survive", "#d95f02"),
        1: ("Survived", "#1b9e77"),
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    for survived, (label, color) in survived_groups.items():
        ages = df.loc[df["Survived"] == survived, "Age"].dropna()
        ax.hist(ages, bins=20, alpha=0.6, label=label, color=color, edgecolor="black")

    ax.set_title("Age Distribution by Survival")
    ax.set_xlabel("Age")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig("eda_age_distribution.png", dpi=150)
    plt.close(fig)
    print("Saved eda_age_distribution.png")


def save_survival_by_sex_plot(df: pd.DataFrame) -> None:
    sex_labels = build_sex_labels(df["Sex"])
    survival_rates = df.assign(SexLabel=sex_labels).groupby("SexLabel")["Survived"].mean()
    ordered_index = [label for label in ["male", "female"] if label in survival_rates.index]
    survival_rates = survival_rates.reindex(ordered_index)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(survival_rates.index, survival_rates.values * 100, color=["#7570b3", "#e7298a"][: len(survival_rates)])
    ax.set_title("Survival Rate by Sex")
    ax.set_xlabel("Sex")
    ax.set_ylabel("Survival Rate (%)")
    ax.set_ylim(0, 100)

    for idx, value in enumerate(survival_rates.values * 100):
        ax.text(idx, value + 1, f"{value:.1f}%", ha="center")

    fig.tight_layout()
    fig.savefig("eda_survival_by_sex.png", dpi=150)
    plt.close(fig)
    print("Saved eda_survival_by_sex.png")


def save_correlation_heatmap(df: pd.DataFrame) -> None:
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    heatmap = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title("Correlation Heatmap")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)

    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", color="black", fontsize=8)

    fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig("eda_correlation_heatmap.png", dpi=150)
    plt.close(fig)
    print("Saved eda_correlation_heatmap.png")


def main() -> None:
    np.random.seed(RANDOM_STATE)
    df = pd.read_csv(CSV_PATH)

    print_dataset_summary(df)
    print_survival_rates(df)
    save_survival_count_plot(df)
    save_age_distribution_plot(df)
    save_survival_by_sex_plot(df)
    save_correlation_heatmap(df)


if __name__ == "__main__":
    main()
