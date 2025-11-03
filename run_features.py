import pandas as pd
from autism_ml.features import build_preprocessor, feature_selection

print("🚀 Loading dataset...")
df = pd.read_csv('./data/processed/autism_cleaned.csv')

# Drop leakage or irrelevant columns
drop_cols = ['result', 'age_desc']
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# Separate features & labels
X = df.drop(columns=['Class_ASD'])
y = df['Class_ASD']

# Build preprocessing pipeline
print("⚙️ Building preprocessing pipeline...")
preprocessor, num_cols, cat_cols = build_preprocessor(df, drop_cols=[])

print("Numeric columns:", num_cols)
print("Categorical columns:", cat_cols)

# Perform feature selection
pipeline = feature_selection(X, y, preprocessor, min_features_to_select=5)

if pipeline:
    print("🎉 Feature selection pipeline created successfully!")
else:
    print("⚠️ Feature selection failed. Check errors above.")
