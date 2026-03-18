"""
Titanic Survival Prediction - Logistic Regression Classifier
Trained on titanic_clean.csv with stratified 80/20 train/test split.
RANDOM_STATE = 42 throughout.
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
data_path = Path(__file__).parent.parent / "Claude Code" / "titanic_clean.csv"
df = pd.read_csv(data_path)

# ---------------------------------------------------------------------------
# 2. Split into features and target
# ---------------------------------------------------------------------------
X = df.drop(columns=["Survived"])
y = df["Survived"]

# ---------------------------------------------------------------------------
# 3. Stratified 80/20 train/test split
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# ---------------------------------------------------------------------------
# 4. Build preprocessing + classifier Pipeline
#    Preprocessing happens inside the pipeline, fitted on training data only.
# ---------------------------------------------------------------------------
numeric_features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
categorical_features = ["Embarked"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ]
)

# Fit on training data only
pipeline.fit(X_train, y_train)

# ---------------------------------------------------------------------------
# 5. Evaluate on test set
# ---------------------------------------------------------------------------
y_pred = pipeline.predict(X_test)

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall    = recall_score(y_test, y_pred, average="weighted")
f1        = f1_score(y_test, y_pred, average="weighted")

print("=" * 40)
print("  Logistic Regression — Test Results")
print("=" * 40)
print(f"  Accuracy  : {accuracy:.4f}")
print(f"  Precision : {precision:.4f}  (weighted)")
print(f"  Recall    : {recall:.4f}  (weighted)")
print(f"  F1 Score  : {f1:.4f}  (weighted)")

# ---------------------------------------------------------------------------
# 6. Confusion matrix
# ---------------------------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(f"  Labels: [Not Survived (0), Survived (1)]")
print(f"\n  {'':>15} Predicted 0  Predicted 1")
print(f"  {'Actual 0':>15}  {cm[0, 0]:^11}  {cm[0, 1]:^11}")
print(f"  {'Actual 1':>15}  {cm[1, 0]:^11}  {cm[1, 1]:^11}")
print("=" * 40)
