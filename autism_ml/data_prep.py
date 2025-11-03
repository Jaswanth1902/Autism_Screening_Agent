"""
Load dataset (ARFF, CSV, TXT, or extension-less), clean, encode, impute, and save as CSV.
Outputs: data/processed/autism_cleaned.csv
"""

import os
import pandas as pd
import numpy as np
from scipy.io import arff

# --- Paths ---
RAW_PATH = os.path.join('../data/raw/autism_combined_all')   # no extension
OUT_PATH = os.path.join('../data/processed/autism_cleaned.csv')


# ---------- 1. LOADING FUNCTION ----------
def load_data(path=RAW_PATH):
    """Loads dataset automatically (supports .arff, .csv, .txt, or no extension)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Dataset not found at: {path}")

    # Try ARFF first (some ARFF files have no .arff extension)
    try:
        raw, meta = arff.loadarff(path)
        df = pd.DataFrame(raw)
        for col in df.select_dtypes([object]).columns:
            try:
                df[col] = df[col].str.decode('utf-8')
            except Exception:
                pass
        print(f"✅ Loaded ARFF-like file with shape {df.shape}")
        return df
    except Exception:
        pass  # not ARFF, try as text/CSV

    # Try detecting delimiter automatically
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        sample = f.readline()

    if sample.count(',') >= sample.count('\t'):
        sep = ','
    else:
        sep = '\t'

    try:
        df = pd.read_csv(path, sep=sep)
        print(f"✅ Loaded delimited text file (sep='{sep}') with shape {df.shape}")
        return df
    except Exception as e:
        raise ValueError(f"❌ Could not load file at {path}. Error: {e}")


# ---------- 2. CLEANING FUNCTION ----------
def clean(df):
    """Clean and standardize autism dataset."""
    df.columns = [c.strip() for c in df.columns]
    df.replace({'?': np.nan, '': np.nan, 'NA': np.nan, 'na': np.nan}, inplace=True)
    df = df.drop_duplicates()

    # Rename inconsistent column
    if 'Class/ASD' in df.columns:
        df.rename(columns={'Class/ASD': 'Class_ASD'}, inplace=True)

    # Binary yes/no columns
    binary_map = {'yes': 1, 'no': 0, 'YES': 1, 'NO': 0, 'Yes': 1, 'No': 0}
    for col in ['jundice', 'austim', 'used_app_before']:
        if col in df.columns:
            df[col] = df[col].map(binary_map).astype('Int64')

    # Target variable
    if 'Class_ASD' in df.columns:
        df['Class_ASD'] = df['Class_ASD'].map(binary_map).astype('Int64')

    # Normalize gender values
    if 'gender' in df.columns:
        df['gender'] = df['gender'].replace({'m': 'M', 'f': 'F', 'male': 'M', 'female': 'F', 'self': 'self'})

    # Convert numeric-looking text to numbers
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = pd.to_numeric(df[col])
            except Exception:
                pass

    print(f"🧹 Cleaned data. Shape after cleaning: {df.shape}")
    return df


# ---------- 3. IMPUTATION FUNCTION ----------
def basic_impute(df):
    """Fill missing values with median/mode."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    for c in numeric_cols:
        df[c] = df[c].fillna(df[c].median())

    for c in cat_cols:
        mode_val = df[c].mode().iloc[0] if not df[c].mode().empty else "Unknown"
        df[c] = df[c].fillna(mode_val)

    print(f"🧩 Imputed missing values. Remaining NaNs: {df.isna().sum().sum()}")
    return df


# ---------- 4. SAVE FUNCTION ----------
def save(df, out=OUT_PATH):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"💾 Saved cleaned data to: {out}")


# ---------- 5. MAIN RUN ----------
def run():
    df = load_data()
    df = clean(df)
    df = basic_impute(df)
    save(df)


if __name__ == '__main__':
    run()
