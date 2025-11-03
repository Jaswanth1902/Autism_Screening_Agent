import pandas as pd
import joblib

# ✅ Update paths
PIPELINE_PATH = r"./autism_ml/artifacts/feature_selector.pkl"
MODEL_PATH = r"./artifacts/model_best_pro_20251029_1633.pkl"

print(f"📂 Using pipeline: {PIPELINE_PATH}")
print(f"📂 Using model: {MODEL_PATH}\n")

# 🔹 Load pipeline and model
pipeline = joblib.load(PIPELINE_PATH)
model = joblib.load(MODEL_PATH)

# ✅ Complex but valid baby dataset samples (0=No, 1=Yes)
sample_data = pd.DataFrame([
    {   # Likely non-autistic
        "A1_Score": 0, "A2_Score": 0, "A3_Score": 1, "A4_Score": 0, "A5_Score": 1,
        "A6_Score": 0, "A7_Score": 0, "A8_Score": 1, "A9_Score": 0, "A10_Score": 1,
        "age": 3, "gender": "m", "ethnicity": "White-European",
        "jundice": 0, "austim": 0, "contry_of_res": "United States",
        "used_app_before": 0, "result": 5, "age_desc": "4 and below",
        "relation": "Parent"
    },
    {   # Likely autistic
        "A1_Score": 1, "A2_Score": 1, "A3_Score": 1, "A4_Score": 1, "A5_Score": 1,
        "A6_Score": 1, "A7_Score": 0, "A8_Score": 1, "A9_Score": 1, "A10_Score": 1,
        "age": 4, "gender": "f", "ethnicity": "Latino",
        "jundice": 1, "austim": 1, "contry_of_res": "India",
        "used_app_before": 1, "result": 9, "age_desc": "4 and below",
        "relation": "Parent"
    },
    {   # Borderline
        "A1_Score": 0, "A2_Score": 1, "A3_Score": 1, "A4_Score": 0, "A5_Score": 1,
        "A6_Score": 0, "A7_Score": 1, "A8_Score": 0, "A9_Score": 1, "A10_Score": 0,
        "age": 5, "gender": "m", "ethnicity": "Asian",
        "jundice": 0, "austim": 0, "contry_of_res": "India",
        "used_app_before": 1, "result": 7, "age_desc": "4 and below",
        "relation": "Parent"
    }
])


print("🧠 Running preprocessing pipeline...")
X_trans = pipeline.transform(sample_data)

print("Expected columns:", list(pipeline.feature_names_in_))
print("Your columns:", list(sample_data.columns))

# ✅ Predict probabilities
probs = model.predict_proba(X_trans)[:, 1]
preds = (probs >= 0.5).astype(int)

# 🧾 Show results
df_pred = pd.DataFrame({
    "Predicted_Class_ASD": preds,
    "Probability_ASD": probs.round(3)
})
print("\n✅ Predictions:")
print(df_pred)

# 🎯 Friendly output
for i, (pred, prob) in enumerate(zip(preds, probs)):
    label = "AUTISM" if pred == 1 else "NON-AUTISM"
    print(f"Sample {i+1}: Predicted {label} (prob={prob:.3f})")
