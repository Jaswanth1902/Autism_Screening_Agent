"""
Clean and save three ARFF autism datasets: child, adolescent, adult
"""

import os
import pandas as pd
import numpy as np
from scipy.io import arff

# --- INPUT DATA PATHS (change filenames if different) ---
INPUT_FILES = {
    "child": "../data/raw/Autism-Child-Data.arff",
    "adolescent": "../data/raw/Autism-Adolescent-Data.arff",
    "adult": "../data/raw/Autism-Adult-Data.arff"
}

# --- OUTPUT FOLDER ---
OUT_DIR = "../data/processed/"


# ---------- LOAD ARFF ----------
def load_arff(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ File not found: {path}")
    
    data, meta = arff.loadarff(path)
    df = pd.DataFrame(data)
    
    # decode byte strings
    for col in df.select_dtypes(include=[object]):
        df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
    
    print(f"📥 Loaded ARFF: {os.path.basename(path)} | Shape: {df.shape}")
    return df


# ---------- CLEAN ----------
def clean(df):
    df.columns = [c.strip() for c in df.columns]
    df.replace({'?': np.nan, '': np.nan}, inplace=True)
    df.drop_duplicates(inplace=True)

    # Normalize target
    if 'Class/ASD' in df.columns:
        df.rename(columns={'Class/ASD': 'Class_ASD'}, inplace=True)

    # Map Yes/No to 1/0
    binary_map = {'yes': 1, 'y': 1, 'no': 0, 'n': 0}
    for col in df.columns:
        if df[col].dtype == 'object' and df[col].isin(binary_map.keys()).any():
            df[col] = df[col].map(binary_map)

    # Normalize gender
    if 'gender' in df.columns:
        df['gender'] = df['gender'].replace(
            {'m': 'M', 'f': 'F', 'male': 'M', 'female': 'F'}
        )

    print("🧹 Cleaning complete.")
    return df


# ---------- IMPUTE ----------
def impute(df):
    numeric = df.select_dtypes(include=[np.number]).columns
    categorical = df.select_dtypes(exclude=[np.number]).columns

    for c in numeric:
        df[c] = df[c].fillna(df[c].median())

    for c in categorical:
        mode = df[c].mode()
        df[c] = df[c].fillna(mode.iloc[0] if not mode.empty else "Unknown")

    print("🧩 Missing values filled.")
    return df


# ---------- SAVE ----------
def save(df, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{name}_clean.csv")
    df.to_csv(out_path, index=False)
    print(f"💾 Saved {name.upper()} → {out_path}\n")


# ---------- MAIN ----------
def run():
    print("🚀 Preprocessing beginning...\n")
    for name, path in INPUT_FILES.items():
        print(f"🔹 Processing {name.upper()} dataset")
        df = load_arff(path)
        df = clean(df)
        df = impute(df)
        save(df, name)
    print("🎯 All datasets processed successfully!")


if __name__ == "__main__":
    run()
