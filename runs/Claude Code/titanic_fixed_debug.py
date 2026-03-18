import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

RANDOM_STATE = 42

df = pd.read_csv('titanic_clean.csv')
# Encode any remaining categorical columns so LogisticRegression can accept them
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
X = df.drop('Survived', axis=1)
y = df['Survived']

# Bug 1 fixed: shuffle=False removed (default shuffle=True ensures a
# representative, unbiased split instead of preserving the original row order,
# which in the Titanic dataset is correlated with passenger class / survival).
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
model.fit(X_train, y_train)

# Bug 2 fixed: predict on X_test (not X_train).
# The original code called model.predict(X_train), producing predictions whose
# length matches the training set (~800) but compared them against y_test
# (~200), causing a ValueError and — even conceptually — measuring training
# predictions against test labels, which is invalid.
y_pred = model.predict(X_test)

print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred, average='weighted'):.4f}")
print(f"F1:        {f1_score(y_test, y_pred, average='weighted'):.4f}")
