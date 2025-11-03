import pandas as pd
df = pd.read_csv("../data/processed/autism_cleaned.csv")

print(df.shape)
print(df.columns)
print(df.isnull().sum())
print(df['Class_ASD'].value_counts())
