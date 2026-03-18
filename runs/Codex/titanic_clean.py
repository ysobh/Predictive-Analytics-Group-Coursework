from __future__ import annotations

import pandas as pd


RANDOM_STATE = 42
INPUT_PATH = "titanic.csv"
OUTPUT_PATH = "titanic_clean.csv"


def print_schema_summary(df: pd.DataFrame) -> None:
    print("Schema summary:")
    for column in df.columns:
        dtype = df[column].dtype
        missing = df[column].isna().sum()
        print(f"- {column}: dtype={dtype}, missing_values={missing}")


def main() -> None:
    df = pd.read_csv(INPUT_PATH)
    transformations: list[str] = []

    print_schema_summary(df)
    print()

    columns_to_drop = ["PassengerId", "Name", "Ticket", "Cabin"]
    existing_columns_to_drop = [column for column in columns_to_drop if column in df.columns]
    df = df.drop(columns=existing_columns_to_drop)
    transformations.append(
        "Dropped columns: " + ", ".join(existing_columns_to_drop)
    )

    missing_age_count = int(df["Age"].isna().sum())
    age_median = float(df["Age"].median())
    df["Age"] = df["Age"].fillna(age_median)
    transformations.append(
        f"Imputed {missing_age_count} missing Age values with median: {age_median}"
    )

    missing_embarked_count = int(df["Embarked"].isna().sum())
    embarked_mode = df["Embarked"].mode(dropna=True).iloc[0]
    df["Embarked"] = df["Embarked"].fillna(embarked_mode)
    transformations.append(
        f"Imputed {missing_embarked_count} missing Embarked values with mode: {embarked_mode}"
    )

    sex_mapping = {"male": 0, "female": 1}
    df["Sex"] = df["Sex"].map(sex_mapping)
    transformations.append("Encoded Sex column as binary: male=0, female=1")

    df.to_csv(OUTPUT_PATH, index=False)

    print("Transformations applied:")
    for transformation in transformations:
        print(f"- {transformation}")

    print()
    print(f"Final clean dataframe shape: {df.shape}")
    print(f"Saved cleaned dataframe to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
