import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# --- Configuration ---
MODEL_PATH = os.path.join("autism_ml", "artifacts", "child_model_20251204_1503.pkl")

# --- Load Model & Preprocessor ---
if not os.path.exists(MODEL_PATH):
    print(f"❌ Model file not found: {MODEL_PATH}")
    model_data = None
else:
    print(f"📥 Loading model: {MODEL_PATH}")
    model_data = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully.")

def add_engineered_features(df):
    """Adds engineered features expected by the model."""
    df = df.copy()
    score_cols = [f"A{i}_Score" for i in range(1, 11)]
    
    # Ensure scores are float
    for col in score_cols:
        if col in df.columns:
            df[col] = df[col].astype(float)
            
    df["score_sum"] = df[score_cols].sum(axis=1)
    df["score_mean"] = df[score_cols].mean(axis=1)
    
    # Ensure binary columns are present
    for col in ("austim", "jundice"):
        if col not in df.columns:
            df[col] = 0
        else:
            df[col] = df[col].astype(float)

    df["family_risk"] = df["austim"] + df["jundice"]
    
    # Drop meta columns if they exist
    for drop_col in ("result", "age_desc"):
        if drop_col in df.columns:
            df = df.drop(columns=[drop_col])
            
    return df

@app.route("/predict", methods=["POST"])
def predict():
    if model_data is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Create DataFrame from input
        # Map frontend field names to model field names
        input_data = {
            "A1_Score": data.get("A1_Score", "0"),
            "A2_Score": data.get("A2_Score", "0"),
            "A3_Score": data.get("A3_Score", "0"),
            "A4_Score": data.get("A4_Score", "0"),
            "A5_Score": data.get("A5_Score", "0"),
            "A6_Score": data.get("A6_Score", "0"),
            "A7_Score": data.get("A7_Score", "0"),
            "A8_Score": data.get("A8_Score", "0"),
            "A9_Score": data.get("A9_Score", "0"),
            "A10_Score": data.get("A10_Score", "0"),
            "age": data.get("age", 5),
            "gender": str(data.get("gender", "m")).lower()[0] if data.get("gender") else "m",
            "jundice": str(data.get("jaundice", "0")),
            "austim": str(data.get("autism_in_family", "0")),
            "ethnicity": data.get("ethnicity", "Others"),
            "contry_of_res": data.get("country", "Jordan"),
            "used_app_before": str(data.get("used_app_before", "0")),
            "relation": data.get("relation", "Parent")
        }

        print(f"DEBUG: Input data: {input_data}")
        df = pd.DataFrame([input_data])
        
        # Explicit type conversions
        score_cols = [f"A{i}_Score" for i in range(1, 11)]
        for col in score_cols + ["age", "jundice", "austim", "used_app_before"]:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # 🔹 Add engineered features
        df = add_engineered_features(df)
        
        # 🔹 Preprocessing
        pipeline = model_data["preprocessor"]
        model = model_data["model"]
        
        print(f"DEBUG: Processing through pipeline...")
        X_trans = pipeline.transform(df)
        
        # 🔹 Prediction
        print(f"DEBUG: Making prediction...")
        # Handle cases where model might return 1D array or different structure
        probs = model.predict_proba(X_trans)
        print(f"DEBUG: Probs structure: {probs}")
        
        prob = float(probs[0][1]) # Probability of class 1 (ASD)
        prediction = int(prob >= 0.5)
        
        print(f"DEBUG: Result: {prediction} (prob: {prob})")
        return jsonify({
            "prediction": prediction,
            "probability": round(prob, 4),
            "is_autistic": bool(prediction)
        })

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model_loaded": model_data is not None})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
