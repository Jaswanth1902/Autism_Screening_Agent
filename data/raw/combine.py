# merge_autism_datasets.py
import pandas as pd
from scipy.io import arff
import os

# 🔹 Paths to your datasets
paths = [
    "./Autism-Adolescent-Data.arff",
    "./Autism-Child-Data.arff",
    "./Autism-Adult-Data.arff"
]

dfs = []

for path in paths:
    data, meta = arff.loadarff(path)
    df = pd.DataFrame(data)

    # Decode byte strings (ARFF sometimes stores categorical fields as bytes)
    df = df.applymap(lambda x: x.decode() if isinstance(x, bytes) else x)

    # Add a column to track source dataset
    if "adolescent" in path.lower():
        df["age_group"] = "adolescent"
    elif "child" in path.lower():
        df["age_group"] = "child"
    elif "adult" in path.lower():
        df["age_group"] = "adult"
    else:
        df["age_group"] = "unknown"

    dfs.append(df)
    print(f"✅ Loaded {path} → shape {df.shape}")

# 🔹 Combine all datasets
combined_df = pd.concat(dfs, ignore_index=True)
print(f"\n📊 Combined dataset shape: {combined_df.shape}")

# 🔹 Optional cleanup
combined_df.columns = combined_df.columns.str.strip()
combined_df.drop_duplicates(inplace=True)

# 🔹 Save the combined dataset
combined_df.to_csv("../raw/autism_combined_all", index=False)
print("\n💾 Saved to: ../raw/autism_combined_all")

# 🔹 Quick summary
print("\n📋 Column summary:")
print(combined_df.info())
print("\n🔍 Class distribution:")
print(combined_df["Class/ASD"].value_counts())
print("\n🧠 Age groups distribution:")
print(combined_df["age_group"].value_counts())
