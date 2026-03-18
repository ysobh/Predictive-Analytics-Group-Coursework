import matplotlib
matplotlib.use('Agg')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_STATE = 42

# ── 1. Load and describe ──────────────────────────────────────────────────────
df = pd.read_csv('titanic_clean.csv')

print("=== Shape ===")
print(df.shape)

print("\n=== Column names and dtypes ===")
print(df.dtypes)

print("\n=== describe() ===")
print(df.describe())

# ── 2. Survival rates ─────────────────────────────────────────────────────────
print("\n=== Survival rate by Sex ===")
sex_rate = df.groupby('Sex')['Survived'].mean() * 100
for sex, rate in sex_rate.items():
    print(f"  {sex}: {rate:.1f}%")

print("\n=== Survival rate by Pclass ===")
pclass_rate = df.groupby('Pclass')['Survived'].mean() * 100
for pclass, rate in pclass_rate.items():
    print(f"  Class {pclass}: {rate:.1f}%")

# ── 3a. Survival count bar chart ──────────────────────────────────────────────
fig, ax = plt.subplots()
df['Survived'].value_counts().sort_index().plot(kind='bar', ax=ax, color=['steelblue', 'salmon'])
ax.set_title('Survival Counts')
ax.set_xlabel('Survived (0 = No, 1 = Yes)')
ax.set_ylabel('Count')
ax.set_xticklabels(['0', '1'], rotation=0)
plt.tight_layout()
plt.savefig('eda_survival_count.png')
plt.close()
print("Saved eda_survival_count.png")

# ── 3b. Age distribution histogram coloured by Survived ───────────────────────
fig, ax = plt.subplots()
for survived, grp in df.groupby('Survived'):
    label = 'Survived' if survived == 1 else 'Not Survived'
    ax.hist(grp['Age'].dropna(), bins=20, alpha=0.6, label=label)
ax.set_title('Age Distribution by Survival')
ax.set_xlabel('Age')
ax.set_ylabel('Count')
ax.legend()
plt.tight_layout()
plt.savefig('eda_age_distribution.png')
plt.close()
print("Saved eda_age_distribution.png")

# ── 3c. Survival rate by Sex grouped bar chart ────────────────────────────────
fig, ax = plt.subplots()
sex_survival = df.groupby('Sex')['Survived'].mean() * 100
sex_survival.plot(kind='bar', ax=ax, color=['steelblue', 'salmon'])
ax.set_title('Survival Rate by Sex')
ax.set_xlabel('Sex')
ax.set_ylabel('Survival Rate (%)')
ax.set_xticklabels(sex_survival.index, rotation=0)
plt.tight_layout()
plt.savefig('eda_survival_by_sex.png')
plt.close()
print("Saved eda_survival_by_sex.png")

# ── 3d. Correlation heatmap ───────────────────────────────────────────────────
numeric_df = df.select_dtypes(include='number')
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
ax.set_title('Correlation Heatmap of Numeric Features')
plt.tight_layout()
plt.savefig('eda_correlation_heatmap.png')
plt.close()
print("Saved eda_correlation_heatmap.png")
