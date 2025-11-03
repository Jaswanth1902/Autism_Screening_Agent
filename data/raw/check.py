import pandas as pd
from scipy.io import arff

paths = [
    "./Autism-Adolescent-Data.arff",
    "./Autism-Child-Data.arff",
    "./Autism-Adult-Data.arff"
]

for path in paths:
    data, meta = arff.loadarff(path)
    df = pd.DataFrame(data)
    print(f"\n🔹 {path}")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns[:10]))
