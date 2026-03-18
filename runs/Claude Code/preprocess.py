import pandas as pd

RANDOM_STATE = 42

# ─────────────────────────────────────────────
# 1. Load
# ─────────────────────────────────────────────
DATA_PATH = "../../data/Titanic-Dataset.csv"
OUTPUT_PATH = "titanic_clean.csv"

df = pd.read_csv(DATA_PATH)

# ─────────────────────────────────────────────
# 2. Schema summary
# ─────────────────────────────────────────────
print("=" * 55)
print("SCHEMA SUMMARY")
print("=" * 55)
summary = pd.DataFrame({
    "dtype": df.dtypes,
    "missing": df.isnull().sum(),
})
print(summary.to_string())
print(f"\nShape before cleaning: {df.shape}")

# ─────────────────────────────────────────────
# 3. Cleaning
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("TRANSFORMATIONS")
print("=" * 55)

# 3a. Drop identifier / high-cardinality columns
drop_cols = ["PassengerId", "Name", "Ticket", "Cabin"]
df.drop(columns=drop_cols, inplace=True)
print(f"Dropped columns: {drop_cols}")

# 3b. Impute Age with median
age_missing = df["Age"].isnull().sum()
age_median = df["Age"].median()
df["Age"] = df["Age"].fillna(age_median)
print(f"Imputed {age_missing} missing Age values with median: {age_median}")

# 3c. Impute Embarked with mode
embarked_missing = df["Embarked"].isnull().sum()
embarked_mode = df["Embarked"].mode()[0]
df["Embarked"] = df["Embarked"].fillna(embarked_mode)
print(f"Imputed {embarked_missing} missing Embarked values with mode: '{embarked_mode}'")

# 3d. Encode Sex as binary (male=0, female=1)
sex_map = {"male": 0, "female": 1}
df["Sex"] = df["Sex"].map(sex_map)
print(f"Encoded Sex column as binary (male=0, female=1)")

# ─────────────────────────────────────────────
# 4. Final shape
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("FINAL DATAFRAME")
print("=" * 55)
print(f"Shape after cleaning: {df.shape}")
print(f"\nRemaining missing values per column:")
print(df.isnull().sum().to_string())

# ─────────────────────────────────────────────
# 5. Save
# ─────────────────────────────────────────────
df.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved cleaned dataframe to: {OUTPUT_PATH}")
