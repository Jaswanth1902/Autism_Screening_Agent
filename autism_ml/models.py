"""
Autism Detection ML Training Pipeline
-------------------------------------
Features:
✅ Stable Optuna tuning (Stratified CV)
✅ Feature engineering with engineered columns
✅ Stacking ensemble (RF, XGB, LGBM, CatBoost)
✅ Probability calibration (isotonic)
✅ ROC / PR / Calibration plots
✅ Permutation importance
✅ Robust SHAP explainability (TreeExplainer → KernelExplainer fallback)
✅ Deterministic reproducibility
"""

import os
import re
import joblib
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Use non-GUI backend to avoid Tkinter issues
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score
)
from sklearn.metrics import (
    f1_score, roc_auc_score, classification_report, confusion_matrix,
    brier_score_loss, matthews_corrcoef, precision_recall_curve, auc,
    RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.inspection import permutation_importance

import optuna
import shap

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Constants
ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)
RANDOM_STATE = 42
OPTUNA_TRIALS = 30

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------
# Utilities
# ---------------------------
def safe_float_cell(x):
    """Convert any weird cell (like '[0.48]') to float safely."""
    if x is None:
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip()
    s = re.sub(r'^[\[\(]+|[\]\)]+$', '', s)
    try:
        return float(s)
    except Exception:
        return np.nan


def ensure_numeric_dataframe(X_df):
    """Apply safe_float_cell and replace NaN with 0."""
    X_num = X_df.applymap(safe_float_cell)
    return X_num.fillna(0)


# ---------------------------
# Feature engineering
# ---------------------------
def add_engineered_features(df):
    df = df.copy()
    score_cols = [f"A{i}_Score" for i in range(1, 11) if f"A{i}_Score" in df.columns]
    if score_cols:
        df["score_sum"] = df[score_cols].sum(axis=1)
        df["score_mean"] = df[score_cols].mean(axis=1)
    for col in ("austim", "jundice"):
        if col not in df.columns:
            df[col] = 0
    df["family_risk"] = df["austim"].astype(float) + df["jundice"].astype(float)
    for drop_col in ("result", "age_desc"):
        if drop_col in df.columns:
            df = df.drop(columns=[drop_col])
    return df


# ---------------------------
# Optuna tuning for XGBoost
# ---------------------------
def tune_xgb(X, y, n_trials=OPTUNA_TRIALS):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "random_state": RANDOM_STATE,
            "use_label_encoder": False,
            "n_jobs": -1,
        }
        model = XGBClassifier(eval_metric="logloss", **params)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(model, X, y, cv=cv, scoring="f1", n_jobs=-1)
        return float(scores.mean())

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print("✅ Optuna XGB best params:", study.best_params)
    return study.best_params


# ---------------------------
# Main Training Pipeline
# ---------------------------
def train_advanced(pipeline, data_csv="data/processed/autism_cleaned.csv"):
    print("🚀 Loading dataset...")
    df = pd.read_csv(data_csv)
    df = add_engineered_features(df)

    if "Class_ASD" not in df.columns:
        raise ValueError("Target column 'Class_ASD' not found in dataset")

    X_raw = df.drop(columns=["Class_ASD"])
    y = df["Class_ASD"].astype(int)

    print("🔧 Transforming features with pipeline...")
    X_trans = pipeline.transform(X_raw)

    try:
        feature_names = pipeline.get_feature_names_out(X_raw.columns)
        X_trans_df = pd.DataFrame(X_trans, columns=feature_names)
    except Exception:
        X_trans_df = pd.DataFrame(X_trans)

    X_trans_df = ensure_numeric_dataframe(X_trans_df)
    assert np.isfinite(X_trans_df.to_numpy()).all(), "Non-finite values remain in X_trans_df"

    X_train, X_test, y_train, y_test = train_test_split(
        X_trans_df, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"✅ Data split: {len(X_train)} train / {len(X_test)} test")

    # Tune XGBoost
    print("🔍 Running Optuna for XGBoost...")
    best_xgb_params = tune_xgb(X_train, y_train, n_trials=OPTUNA_TRIALS)
    best_xgb_params.update({
        "use_label_encoder": False, "eval_metric": "logloss",
        "random_state": RANDOM_STATE, "n_jobs": -1
    })

    # Models
    rf = RandomForestClassifier(n_estimators=300, max_depth=10, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
    xgb = XGBClassifier(**best_xgb_params)
    lgb = LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    cat = CatBoostClassifier(verbose=0, random_state=RANDOM_STATE)

    print("🧠 Building stacking ensemble...")
    estimators = [("rf", rf), ("xgb", xgb), ("lgb", lgb), ("cat", cat)]
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        stack_method="predict_proba",
        n_jobs=-1,
    )
    stack.fit(X_train, y_train)

    print("⚙️ Calibrating probabilities with isotonic (cv=5)...")
    calibrated = CalibratedClassifierCV(stack, method="isotonic", cv=5)
    calibrated.fit(X_train, y_train)

    # Evaluation
    print("📈 Evaluating on test set...")
    y_pred = calibrated.predict(X_test)
    y_prob = calibrated.predict_proba(X_test)[:, 1]

    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_prob)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    auc_pr = auc(recall, precision)
    brier = brier_score_loss(y_test, y_prob)
    mcc = matthews_corrcoef(y_test, y_pred)

    print("\n🎯 Final Model Metrics")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {auc_roc:.4f}")
    print(f"PR AUC : {auc_pr:.4f}")
    print(f"Brier Score: {brier:.4f}")
    print(f"MCC: {mcc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    model_path = os.path.join(ARTIFACT_DIR, f"model_best_pro_{timestamp}.pkl")
    joblib.dump(calibrated, model_path)
    print(f"💾 Saved model → {model_path}")

    # ROC / PR / Calibration plots
    try:
        RocCurveDisplay.from_predictions(y_test, y_prob)
        plt.title("ROC Curve")
        plt.savefig(os.path.join(ARTIFACT_DIR, f"roc_{timestamp}.png"), bbox_inches="tight")
        plt.close()

        PrecisionRecallDisplay.from_predictions(y_test, y_prob)
        plt.title("Precision-Recall Curve")
        plt.savefig(os.path.join(ARTIFACT_DIR, f"pr_{timestamp}.png"), bbox_inches="tight")
        plt.close()

        prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
        plt.plot(prob_pred, prob_true, marker="o")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Fraction of positives")
        plt.title("Calibration Curve")
        plt.savefig(os.path.join(ARTIFACT_DIR, f"calibration_{timestamp}.png"), bbox_inches="tight")
        plt.close()

        print("✅ Saved ROC/PR/Calibration plots.")
    except Exception as e:
        print("⚠️ Failed to save ROC/PR/Calibration plots:", e)

    # Permutation Importance
    try:
        print("🔁 Calculating permutation importance (may take time)...")
        perm = permutation_importance(calibrated, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
        perm_df = pd.DataFrame({
            "feature": X_test.columns,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std
        }).sort_values("importance_mean", ascending=False)
        perm_df.to_csv(os.path.join(ARTIFACT_DIR, f"permutation_importance_{timestamp}.csv"), index=False)
        print("✅ Permutation importance saved.")
    except Exception as e:
        print("⚠️ Permutation importance failed:", e)

    # ---------------------------
    # SHAP Explainability
    # ---------------------------
    print("🧩 Generating SHAP explainability plots...")

    try:
        X_sample = shap.utils.sample(X_test, min(100, len(X_test)), random_state=42)
        model_for_shap = calibrated

        # Try TreeExplainer first
        try:
            explainer = shap.TreeExplainer(model_for_shap)
            shap_values = explainer.shap_values(X_sample)
            print("✅ Using TreeExplainer for SHAP.")
        except Exception as e:
            print(f"⚠️ TreeExplainer failed ({e}), switching to KernelExplainer...")

            def predict_fn(data):
                if isinstance(data, pd.DataFrame):
                    data = data.values
                elif np.ndim(data) == 1:
                    data = data.reshape(1, -1)
                return model_for_shap.predict_proba(data)[:, 1]

            background = shap.utils.sample(X_sample, 50, random_state=42)
            explainer = shap.KernelExplainer(predict_fn, background)
            shap_values = explainer.shap_values(X_sample)
            print("✅ Using KernelExplainer for SHAP.")

        shap_summary_path = os.path.join(ARTIFACT_DIR, f"shap_summary_{timestamp}.png")
        shap_bar_path = os.path.join(ARTIFACT_DIR, f"shap_bar_{timestamp}.png")

        plt.figure()
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.tight_layout()
        plt.savefig(shap_summary_path, dpi=300, bbox_inches="tight")
        plt.close()

        plt.figure()
        shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(shap_bar_path, dpi=300, bbox_inches="tight")
        plt.close()

        np.save(os.path.join(ARTIFACT_DIR, f"shap_values_{timestamp}.npy"), shap_values)
        print(f"✅ SHAP plots saved → {shap_summary_path}, {shap_bar_path}")

    except Exception as e:
        print(f"⚠️ SHAP explainability skipped due to error: {e}")

    print("🎉 Training pipeline finished. All artifacts are in:", ARTIFACT_DIR)
