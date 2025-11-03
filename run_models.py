# run_models_pro.py
import joblib
from autism_ml.models import train_advanced

print("🚀 Loading feature selector pipeline...")
pipeline = joblib.load('./artifacts/feature_selector.pkl')

train_advanced(pipeline)


print("✅ Model training and saving complete.")
