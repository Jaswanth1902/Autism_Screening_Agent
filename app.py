import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import traceback

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# --- Configuration ---
MODEL_PATH = os.path.join("autism_ml", "artifacts", "child_model_20251204_1503.pkl")

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyDIOo4mfGFzj_08z9dVlrwgXgC44TBSWBc") # Placeholder
genai.configure(api_key=GEMINI_API_KEY)

# Chatbot System Prompt
SYSTEM_PROMPT = """
You are a helpful assistant for an Autism Screening app. Keep responses SHORT and CONCISE (2-3 sentences max).

Key points:
- This app uses ISAA (Indian Scale for Assessment of Autism) with 40 questions across 6 domains
- Domains: Social Relationship, Emotional Responsiveness, Speech-Language, Behavior Patterns, Sensory Aspects, Cognitive
- Karnataka resources: ASHA, Com DEALL, AIISH (Bangalore, Mysuru, Hubli, Mangaluru)
- Always remind users: This is a screening tool, NOT a diagnosis. Consult a specialist for evaluation.

Be empathetic. Avoid long explanations unless specifically asked.
"""
generation_config = {
  "temperature": 0.7,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 256,
}
chat_model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    generation_config=generation_config,
    system_instruction=SYSTEM_PROMPT
)
chat_session = chat_model.start_chat(history=[])

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

def calculate_isaa_risk(data):
    """
    Calculates risk level based on the ISAA (Indian Scale for Assessment of Autism) 
    approximated logic for 40 aspects.
    Data format: Q1 to Q40 with values 0, 0.5, or 1.
    For standard ISAA, scores are 1-5, but we normalize here.
    """
    total_score = 0
    # Map our 0-1 values to ISAA 1-5 scale:
    # 0 -> 1, 0.25 -> 2, 0.5 -> 3, 0.75 -> 4, 1 -> 5
    score_map = {
        0: 1, 0.25: 2, 0.5: 3, 0.75: 4, 1: 5,
        "0": 1, "0.25": 2, "0.5": 3, "0.75": 4, "1": 5
    }
    
    question_count = 0
    for i in range(1, 41):
        val = data.get(f"Q{i}", 0)
        total_score += score_map.get(val, 1)
        question_count += 1
        
    # ISAA Thresholds:
    # < 70: No Autism
    # 70-106: Mild
    # 107-153: Moderate
    # > 153: Severe
    
    risk_level = "Low"
    if total_score > 153:
        risk_level = "High"
    elif total_score >= 70:
        risk_level = "Moderate"
        
    return {
        "score": total_score,
        "risk_level": risk_level,
        "is_autistic": total_score >= 70
    }

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

@app.route("/evaluate_isaa", methods=["POST"])
def evaluate_isaa():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400

        result = calculate_isaa_risk(data)

        # --- Hybrid Inference Integration ---
        # Map 40 ISAA questions to 10 AQ-10 inputs for ML model
        # Logic: Average relevant ISAA values (0.0 to 1.0)
        def get_avg(qs):
            vals = [float(data.get(q, 0)) for q in qs]
            return sum(vals) / len(vals) if vals else 0

        aq10_inputs = {
            "A1_Score": get_avg(["Q1", "Q2"]),
            "A2_Score": get_avg(["Q3"]),
            "A3_Score": get_avg(["Q4"]),
            "A4_Score": get_avg(["Q5"]),
            "A5_Score": get_avg(["Q6"]),
            "A6_Score": get_avg(["Q7", "Q16"]),
            "A7_Score": get_avg(["Q15", "Q17", "Q18"]),
            "A8_Score": get_avg(["Q8", "Q24"]),
            "A9_Score": get_avg(["Q9", "Q30"]),
            "A10_Score": get_avg(["Q10", "Q25"]),
            "age": data.get("age", 5),
            "gender": data.get("gender", "m"),
            "jaundice": data.get("jaundice", "0"),
            "autism_in_family": data.get("autism_in_family", "0")
        }

        # Run ML inference
        ml_prob = 0
        if model_data:
            try:
                # Reuse the internal predict logic but with mapped inputs
                # We need to call predict() logic or factor it out. 
                # For now, let's keep it simple: 
                # Since we don't want to duplicate the entire predict logic, 
                # we'll mock a request-like object or just call the model directly
                df_ml = pd.DataFrame([{
                    "A1_Score": aq10_inputs["A1_Score"],
                    "A2_Score": aq10_inputs["A2_Score"],
                    "A3_Score": aq10_inputs["A3_Score"],
                    "A4_Score": aq10_inputs["A4_Score"],
                    "A5_Score": aq10_inputs["A5_Score"],
                    "A6_Score": aq10_inputs["A6_Score"],
                    "A7_Score": aq10_inputs["A7_Score"],
                    "A8_Score": aq10_inputs["A8_Score"],
                    "A9_Score": aq10_inputs["A9_Score"],
                    "A10_Score": aq10_inputs["A10_Score"],
                    "age": aq10_inputs["age"],
                    "gender": str(aq10_inputs["gender"]).lower()[0] if aq10_inputs["gender"] else "m",
                    "jundice": str(aq10_inputs["jaundice"]),
                    "austim": str(aq10_inputs["autism_in_family"]),
                    "ethnicity": "Others",
                    "contry_of_res": "Jordan",
                    "used_app_before": "0",
                    "relation": "Parent"
                }])
                
                df_ml = add_engineered_features(df_ml)
                X_trans = model_data["preprocessor"].transform(df_ml)
                probs = model_data["model"].predict_proba(X_trans)
                ml_prob = float(probs[0][1])
            except Exception as ml_e:
                print(f"⚠️ ML Hybrid Inference warning: {ml_e}")

        # Combine Logical Risk and ML Probability for a Confidence Index
        # Normalized clinical score (0-1)
        clinical_prob = (result["score"] - 40) / 160
        
        # Weighted average: 60% Clinical (ISAA is the gold standard here), 40% ML
        final_prob = (clinical_prob * 0.6) + (ml_prob * 0.4) if ml_prob > 0 else clinical_prob

        return jsonify({
            "prediction": 1 if final_prob >= 0.5 else 0,
            "probability": round(final_prob, 4),
            "risk_level": result["risk_level"],
            "is_autistic": result["is_autistic"],
            "total_score": result["score"],
            "ml_confidence": round(ml_prob, 4) if ml_prob > 0 else None
        })
    except Exception as e:
        print(f"❌ ISAA Evaluation error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model_loaded": model_data is not None})

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        user_message = data.get("message", "")
        if not user_message:
            return jsonify({"error": "No message provided"}), 400

        # Send message to Gemini
        response = chat_session.send_message(user_message)
        bot_reply = response.text
        
        return jsonify({"response": bot_reply})
        
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"❌ Chat Error: {e}")
        print(f"Full traceback:\n{error_details}")
        return jsonify({"response": f"Error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
