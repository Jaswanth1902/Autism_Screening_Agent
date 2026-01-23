import pandas as pd
import os

# --- PATHS ---
INPUT_FILE = os.path.join("data", "raw", "Autism_Screening_Data_Combined.csv")
OUT_DIR = os.path.join("data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

def preprocess_kaggle():
    print(f"📥 Loading: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    
    # 1. Clean column names
    df.columns = [c.strip() for c in df.columns]
    
    # 2. Map Features to match existing schema
    # Map A1-A10 (Already numeric based on analysis, but let's ensure they are float)
    score_cols = [f"A{i}" for i in range(1, 11)]
    for col in score_cols:
        df[col] = df[col].astype(float)
        # Rename to match pipeline if needed (models.py expects A{i}_Score)
        df.rename(columns={col: f"{col}_Score"}, inplace=True)
    
    # Map binary features
    binary_map = {'yes': 1, 'y': 1, 'no': 0, 'n': 0}
    df['jundice'] = df['Jauundice'].str.lower().map(binary_map)
    df['austim'] = df['Family_ASD'].str.lower().map(binary_map)
    df['Class_ASD'] = df['Class'].str.upper().map({'YES': 1, 'NO': 0})
    
    # Map gender
    df['gender'] = df['Sex'].str.upper().replace({'M': 'M', 'F': 'F'})
    
    # Normalize age column name
    df.rename(columns={'Age': 'age'}, inplace=True)
    
    # Select final columns to keep (consistent with existing _clean.csv files)
    # Note: Existing files have: A1_Score...A10_Score, age, gender, ethnicity, jundice, austim, contry_of_res, result, age_desc, relation, Class_ASD
    # Kaggle dataset lacks: ethnicity, contry_of_res, result, age_desc, relation
    # We will provide defaults or handle missing cols in models.py
    
    final_cols = [f"A{i}_Score" for i in range(1, 11)] + ['age', 'gender', 'jundice', 'austim', 'Class_ASD']
    df_clean = df[final_cols].copy()
    
    # 3. Split by Age Group
    # Toddler: 1-3
    toddler_df = df_clean[df_clean['age'] <= 3].copy()
    # Child: 4-11
    child_df = df_clean[(df_clean['age'] >= 4) & (df_clean['age'] <= 11)].copy()
    # Adolescent: 12-16
    adolescent_df = df_clean[(df_clean['age'] >= 12) & (df_clean['age'] <= 16)].copy()
    # Adult: 17+
    adult_df = df_clean[df_clean['age'] >= 17].copy()
    
    # 4. Save (Augmented or New)
    # Since we want to use the larger dataset primarily, we save them as kaggle_<name>.csv
    # and then models.py can be updated to point to these or combined ones.
    
    save_map = {
        "toddler": toddler_df,
        "child": child_df,
        "adolescent": adolescent_df,
        "adult": adult_df
    }
    
    for name, subset in save_map.items():
        out_path = os.path.join(OUT_DIR, f"{name}_kaggle.csv")
        subset.to_csv(out_path, index=False)
        print(f"💾 Saved {name.upper()} ({len(subset)} rows) → {out_path}")

    print("\n🎯 Preprocessing complete!")

if __name__ == "__main__":
    preprocess_kaggle()
