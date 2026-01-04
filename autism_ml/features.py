"""
Feature engineering + RFECV feature selection
Running separately for 3 datasets:
    - child
    - adolescent
    - adult
Each produces:
    - <name>_feature_selector.pkl
    - <name>_selected_features.txt
"""

import os
import joblib
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold

# --------------------------
#  Dataset paths
# --------------------------
DATASETS = {
    "child": "../data/processed/child_clean.csv",
    "adolescent": "../data/processed/adolescent_clean.csv",
    "adult": "../data/processed/adult_clean.csv"
}

ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

TARGET = "Class_ASD"


# -----------------------------------------------------
#          ORIGINAL FUNCTIONS (unchanged)
# -----------------------------------------------------
def build_preprocessor(df, drop_cols=None):
    """Builds preprocessing pipeline for numeric + categorical."""
    if drop_cols is None:
        drop_cols = []

    X = df.drop(columns=drop_cols) if drop_cols else df.copy()

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    if TARGET in numeric_cols:
        numeric_cols.remove(TARGET)
    if TARGET in cat_cols:
        cat_cols.remove(TARGET)

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, cat_cols)
    ])

    return preprocessor, numeric_cols, cat_cols


def feature_selection(X, y, preprocessor, min_features_to_select=5):
    """RFECV wrapper."""
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight='balanced'
    )

    pipeline = Pipeline([
        ('pre', preprocessor),
        ('sel', RFECV(
            estimator=rf,
            step=1,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='f1',
            min_features_to_select=min_features_to_select,
            n_jobs=-1
        ))
    ])

    print("🔍 Starting RFECV...")

    try:
        pipeline.fit(X, y)
        print("✅ RFECV completed successfully.")
    except Exception as e:
        print("❌ RFECV failed:", e)
        return None

    return pipeline


def transform_with_selector(pipeline, df_X):
    return pipeline.transform(df_X)


# -----------------------------------------------------
#           NEW CODE: Handle 3 datasets
# -----------------------------------------------------
def run_for_dataset(name, path):
    print("\n==============================")
    print(f"🔹 Running for: {name.upper()}")
    print("==============================")

    df = pd.read_csv(path)

    # fix target labelling
    if "Class/ASD" in df.columns:
        df.rename(columns={"Class/ASD": TARGET}, inplace=True)

    df[TARGET] = df[TARGET].replace({
        'YES': 1, 'NO': 0,
        'yes': 1, 'no': 0,
        'Yes': 1, 'No': 0
    }).astype(int)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    preprocessor, num_cols, cat_cols = build_preprocessor(df, drop_cols=[TARGET])

    pipeline = feature_selection(X, y, preprocessor)

    if pipeline is None:
        print(f"⚠️ Skipping {name}, selector failed.")
        return

    # Extract selected features (the mask applies BEFORE preprocessing)
    selector = pipeline.named_steps['sel']
    # Get transformed feature names BEFORE RFECV selection
    ohe = pipeline.named_steps['pre'].named_transformers_['cat']['onehot']
    cat_cols = pipeline.named_steps['pre'].transformers_[1][2]  # categorical columns
    numeric_cols = pipeline.named_steps['pre'].transformers_[0][2]  # numeric columns

    # Expanded categorical feature names from OneHotEncoder
    cat_expanded = list(ohe.get_feature_names_out(cat_cols))

    # Full feature list after preprocessing
    all_features_after_pre = numeric_cols + cat_expanded

    # Mask after RFECV selection
    mask = selector.support_
    selected_cols = [col for col, keep in zip(all_features_after_pre, mask) if keep]


    # Print
    print("\n📌 Selected Features:")
    for col in selected_cols:
        print("   -", col)

    # Save .pkl
    model_path = os.path.join(ARTIFACT_DIR, f"{name}_feature_selector.pkl")
    joblib.dump(pipeline, model_path)
    print(f"💾 Saved pipeline → {model_path}")

    # Save report
    txt_path = os.path.join(ARTIFACT_DIR, f"{name}_selected_features.txt")
    with open(txt_path, "w") as f:
        f.write("Selected Features:\n")
        for col in selected_cols:
            f.write(col + "\n")
    print(f"📝 Saved feature list → {txt_path}")


# -----------------------------------------------------
#               MAIN RUNNER
# -----------------------------------------------------
def run_all():
    print("\n🚀 FEATURE SELECTION FOR ALL DATASETS\n")

    for name, path in DATASETS.items():
        if not os.path.exists(path):
            print(f"❌ Missing: {path}")
            continue

        run_for_dataset(name, path)

    print("\n🎯 Completed feature selection for all 3 datasets!")


if __name__ == "__main__":
    run_all()
