"""
Feature engineering and selection utilities.
Includes pipelines for encoding, scaling and a wrapper for RFECV feature selection.
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

ARTIFACT_DIR = 'artifacts'
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def build_preprocessor(df, drop_cols=None):
    """
    Builds preprocessing pipeline for numeric and categorical features.
    Returns preprocessor, numeric column list, and categorical column list.
    """
    if drop_cols is None:
        drop_cols = []

    X = df.drop(columns=drop_cols) if drop_cols else df.copy()

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Remove target variable if present
    if 'Class_ASD' in numeric_cols:
        numeric_cols.remove('Class_ASD')
    if 'Class_ASD' in cat_cols:
        cat_cols.remove('Class_ASD')

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, cat_cols)
    ])

    return preprocessor, numeric_cols, cat_cols


def feature_selection(X, y, preprocessor, min_features_to_select=5):
    """
    Uses RFECV with RandomForest to select top features.
    Saves the preprocessor + selector pipeline as 'feature_selector.pkl'.
    """
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight='balanced'
    )

    pipeline = Pipeline(steps=[
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

    print("🔍 Starting feature selection using RFECV...")

    try:
        pipeline.fit(X, y)
        print("✅ RFECV completed successfully.")

        # Save the pipeline
        save_path = os.path.join(ARTIFACT_DIR, 'feature_selector.pkl')
        joblib.dump(pipeline, save_path)
        print(f"💾 Saved feature selector pipeline → {save_path}")

    except Exception as e:
        print("❌ Error during feature selection:", str(e))
        return None

    return pipeline


def transform_with_selector(pipeline, df_X):
    """
    Transforms input data using the trained feature selector pipeline.
    """
    return pipeline.transform(df_X)
