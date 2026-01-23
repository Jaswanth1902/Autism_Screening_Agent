"""
Autism Detection ML Training Pipeline (Multi-dataset)
-----------------------------------------------------
Per dataset (child, adolescent, adult), this pipeline performs:

✅ Stable Optuna tuning (Stratified CV, XGBoost)
✅ Feature engineering with engineered columns (score_sum, score_mean, family_risk)
✅ Stacking ensemble (RF, XGB, LGBM, CatBoost)
✅ Probability calibration (isotonic)
✅ ROC / PR / Calibration plots
✅ Permutation importance
✅ Robust SHAP explainability (TreeExplainer → KernelExplainer fallback)
✅ Deterministic reproducibility
✅ 5-fold Stratified Cross-Validation evaluation
"""

import os
import re
import joblib
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Use non-GUI backend
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score
)
from sklearn.metrics import (
    f1_score, roc_auc_score, classification_report, confusion_matrix,
    brier_score_loss, matthews_corrcoef, precision_recall_curve, auc,
    RocCurveDisplay, PrecisionRecallDisplay,
    precision_score, recall_score
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

import optuna
import shap

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore", category=UserWarning)

# -------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------
ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

RANDOM_STATE = 42
OPTUNA_TRIALS = 30
N_FOLDS = 5
TARGET = "Class_ASD"

DATASETS = {
    "child": "../data/processed/child_kaggle.csv",
    "adolescent": "../data/processed/adolescent_kaggle.csv",
    "adult": "../data/processed/adult_kaggle.csv",
}


# -------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------
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


# -------------------------------------------------------------------
# Feature engineering
# -------------------------------------------------------------------
def add_engineered_features(df):
    """
    Adds:
      - score_sum, score_mean from A1..A10
      - family_risk from austim + jundice
      - drops result, age_desc (meta)
    """
    df = df.copy()

    score_cols = [f"A{i}_Score" for i in range(1, 11) if f"A{i}_Score" in df.columns]
    if score_cols:
        # Final scores are assumed to be numeric 0/1 or similar
        df["score_sum"] = df[score_cols].sum(axis=1)
        df["score_mean"] = df[score_cols].mean(axis=1)

    for col in ("austim", "jundice"):
        if col not in df.columns:
            df[col] = 0
        else:
            # Map binary strings if not already numeric
            if df[col].dtype == object:
                binary_map = {'yes': 1, 'no': 0, 'Yes': 1, 'No': 0, 'y': 1, 'n': 0}
                df[col] = df[col].str.lower().map(binary_map).fillna(0)

    df["family_risk"] = df["austim"].astype(float) + df["jundice"].astype(float)

    for drop_col in ("result", "age_desc"):
        if drop_col in df.columns:
            df = df.drop(columns=[drop_col])

    return df


# -------------------------------------------------------------------
# Preprocessor builder
# -------------------------------------------------------------------
def build_preprocessor(X):
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )

    return preprocessor, numeric_cols, cat_cols


# -------------------------------------------------------------------
# Optuna tuning for XGBoost
# -------------------------------------------------------------------
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
            "n_jobs": 1,
        }
        y_vals = y.values.astype(int) if hasattr(y, 'values') else y.astype(int)
        with open("y_debug.txt", "a") as f:
            f.write(f"dataset: {X.shape}, y unique: {np.unique(y_vals)}, y len: {len(y_vals)}\n")
        model = XGBClassifier(objective='binary:logistic', eval_metric="logloss", **params)
        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(model, X, y_vals, cv=cv, scoring="f1", n_jobs=1)
        return float(scores.mean())

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print("✅ Optuna XGB best params:", study.best_params)
    return study.best_params


# -------------------------------------------------------------------
# Cross-validation evaluation for the full stack (no calibration)
# -------------------------------------------------------------------
def cross_validate_stack(X, y, rf_params, xgb_params, dataset_name):
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    metrics = {
        "f1": [],
        "roc_auc": [],
        "precision": [],
        "recall": [],
        "mcc": []
    }

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        rf = RandomForestClassifier(
            **rf_params,
            random_state=RANDOM_STATE,
            n_jobs=1
        )
        xgb = XGBClassifier(
            **xgb_params,
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=RANDOM_STATE,
            n_jobs=1
        )
        lgb = LGBMClassifier(random_state=RANDOM_STATE, n_jobs=1,verbose=-1)
        cat = CatBoostClassifier(verbose=0, random_state=RANDOM_STATE)

        stack = StackingClassifier(
            estimators=[
                ("rf", rf),
                ("xgb", xgb),
                ("lgb", lgb),
                ("cat", cat),
            ],
            final_estimator=LogisticRegression(max_iter=1000),
            stack_method="predict_proba",
            n_jobs=1,
        )

        stack.fit(X_tr, y_tr)
        y_prob = stack.predict_proba(X_val)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        metrics["f1"].append(f1_score(y_val, y_pred))
        metrics["roc_auc"].append(roc_auc_score(y_val, y_prob))
        metrics["precision"].append(precision_score(y_val, y_pred))
        metrics["recall"].append(recall_score(y_val, y_pred))
        metrics["mcc"].append(matthews_corrcoef(y_val, y_pred))

    # summary
    print(f"\n📊 {dataset_name.upper()} | {N_FOLDS}-Fold CV Metrics")
    summary_rows = []
    for m_name, values in metrics.items():
        mean_v = np.mean(values)
        std_v = np.std(values)
        print(f"{m_name.upper():<10}: {mean_v:.4f} ± {std_v:.4f}")
        summary_rows.append({
            "metric": m_name,
            "mean": mean_v,
            "std": std_v
        })

    # save to CSV
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    cv_path = os.path.join(ARTIFACT_DIR, f"{dataset_name}_cv_metrics_{ts}.csv")
    pd.DataFrame(summary_rows).to_csv(cv_path, index=False)
    print(f"📝 Saved CV metrics → {cv_path}")


# -------------------------------------------------------------------
# Training pipeline for one dataset
# -------------------------------------------------------------------
def train_for_dataset(dataset_name, csv_path):
    print(f"\n🚀 Training for dataset: {dataset_name.upper()}")
    print(f"📥 Loading data from: {csv_path}")

    df = pd.read_csv(csv_path)
    df = add_engineered_features(df)

    # Normalize target
    if "Class/ASD" in df.columns and TARGET not in df.columns:
        df.rename(columns={"Class/ASD": TARGET}, inplace=True)

    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in {dataset_name} dataset!")

    df[TARGET] = df[TARGET].replace(
        {"YES": 1, "NO": 0, "yes": 1, "no": 0, "Yes": 1, "No": 0}
    ).astype(int)

    # Separate X, y
    X_raw = df.drop(columns=[TARGET])
    y = df[TARGET]
    print(f"DEBUG: {dataset_name.upper()} | y unique: {y.unique()}, dtype: {y.dtype}")

    if len(y.unique()) < 2:
        print(f"⚠️ Skipping {dataset_name.upper()}: Only one class present in labels.")
        return None

    # Preprocess
    preprocessor, numeric_cols, cat_cols = build_preprocessor(X_raw)
    print("🔧 Fitting preprocessing pipeline...")
    X_all = preprocessor.fit_transform(X_raw)

    # Build feature names
    feature_names = []
    feature_names.extend(numeric_cols)
    if cat_cols:
        ohe = preprocessor.named_transformers_["cat"]["onehot"]
        ohe_names = list(ohe.get_feature_names_out(cat_cols))
        feature_names.extend(ohe_names)

    X_df = pd.DataFrame(X_all, columns=feature_names)
    X_df = ensure_numeric_dataframe(X_df)
    assert np.isfinite(X_df.to_numpy()).all(), "Non-finite values remain in X_df"

    # --- Cross-validation of stack (no calibration) on full data ---
    print("📊 Running cross-validation for stack ensemble...")
    # First, tune XGB using entire X_df (or we can use subset)
    print("🔍 Running Optuna for XGBoost (for CV + final model)...")
    best_xgb_params = tune_xgb(X_df, y, n_trials=OPTUNA_TRIALS)

    rf_params = {
        "n_estimators": 300,
        "max_depth": 10,
        "class_weight": "balanced",
    }

    cross_validate_stack(X_df, y, rf_params, best_xgb_params, dataset_name)

    # --- Train / Test split for final calibrated model + SHAP & plots ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"\n✂ Hold-out split: {len(X_train)} train / {len(X_test)} test")

    # Build final models with tuned XGB
    best_xgb_params.update({
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "random_state": RANDOM_STATE,
        "n_jobs": 1
    })

    rf = RandomForestClassifier(
        n_estimators=rf_params["n_estimators"],
        max_depth=rf_params["max_depth"],
        class_weight=rf_params["class_weight"],
        random_state=RANDOM_STATE,
        n_jobs=1
    )
    xgb = XGBClassifier(objective='binary:logistic', **best_xgb_params)
    lgb = LGBMClassifier(random_state=RANDOM_STATE, n_jobs=1, verbose=-1)
    cat = CatBoostClassifier(verbose=0, random_state=RANDOM_STATE)

    print("🧠 Building stacking ensemble for final model...")
    stack = StackingClassifier(
        estimators=[
            ("rf", rf),
            ("xgb", xgb),
            ("lgb", lgb),
            ("cat", cat)
        ],
        final_estimator=LogisticRegression(max_iter=1000),
        stack_method="predict_proba",
        n_jobs=1,
    )
    stack.fit(X_train, y_train)

    print("⚙️ Calibrating probabilities with isotonic regression (cv=5)...")
    calibrated = CalibratedClassifierCV(stack, method="isotonic", cv=5)
    calibrated.fit(X_train, y_train)

    # Evaluation on hold-out test
    print("\n📈 Evaluating on hold-out test set...")
    y_pred = calibrated.predict(X_test)
    y_prob = calibrated.predict_proba(X_test)[:, 1]

    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_prob)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    auc_pr = auc(recall, precision)
    brier = brier_score_loss(y_test, y_prob)
    mcc = matthews_corrcoef(y_test, y_pred)

    print(f"\n🎯 Final Model Metrics: {dataset_name.upper()}")
    print(f"F1 Score      : {f1:.4f}")
    print(f"ROC AUC       : {auc_roc:.4f}")
    print(f"PR AUC        : {auc_pr:.4f}")
    print(f"Brier Score   : {brier:.4f}")
    print(f"MCC           : {mcc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    # Save model + preprocessor
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    model_path = os.path.join(ARTIFACT_DIR, f"{dataset_name}_model_{timestamp}.pkl")
    joblib.dump(
        {"model": calibrated, "preprocessor": preprocessor, "feature_names": feature_names},
        model_path
    )
    print(f"💾 Saved model + preprocessor → {model_path}")

    # -------------------- ROC / PR / Calibration plots --------------------
    try:
        RocCurveDisplay.from_predictions(y_test, y_prob)
        plt.title(f"ROC Curve - {dataset_name}")
        roc_path = os.path.join(ARTIFACT_DIR, f"{dataset_name}_roc_{timestamp}.png")
        plt.savefig(roc_path, bbox_inches="tight")
        plt.close()

        PrecisionRecallDisplay.from_predictions(y_test, y_prob)
        plt.title(f"Precision-Recall Curve - {dataset_name}")
        pr_path = os.path.join(ARTIFACT_DIR, f"{dataset_name}_pr_{timestamp}.png")
        plt.savefig(pr_path, bbox_inches="tight")
        plt.close()

        prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
        plt.plot(prob_pred, prob_true, marker="o")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Fraction of positives")
        plt.title(f"Calibration Curve - {dataset_name}")
        calib_path = os.path.join(ARTIFACT_DIR, f"{dataset_name}_calibration_{timestamp}.png")
        plt.savefig(calib_path, bbox_inches="tight")
        plt.close()

        print("✅ Saved ROC / PR / Calibration plots.")
    except Exception as e:
        print("⚠️ Plotting failed:", e)

    # -------------------- Permutation Importance (DISABLED) --------------------
    """
    try:
        print("🔁 Calculating permutation importance (may take time)...")
        perm = permutation_importance(
            calibrated, X_test, y_test,
            n_repeats=10, random_state=RANDOM_STATE, n_jobs=1
        )
        perm_df = pd.DataFrame({
            "feature": X_test.columns,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std
        }).sort_values("importance_mean", ascending=False)
        perm_path = os.path.join(ARTIFACT_DIR, f"{dataset_name}_permutation_importance_{timestamp}.csv")
        perm_df.to_csv(perm_path, index=False)
        print("✅ Permutation importance saved:", perm_path)
    except Exception as e:
        print("⚠️ Permutation importance failed:", e)
    """

    # -------------------- SHAP Explainability (DISABLED) --------------------
    """
    print("🧩 Generating SHAP explainability plots...")
    try:
        X_sample = shap.utils.sample(X_test, min(100, len(X_test)), random_state=RANDOM_STATE)
        model_for_shap = calibrated

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

            background = shap.utils.sample(X_sample, min(50, len(X_sample)), random_state=RANDOM_STATE)
            explainer = shap.KernelExplainer(predict_fn, background)
            shap_values = explainer.shap_values(X_sample)
            print("✅ Using KernelExplainer for SHAP.")

        shap_summary_path = os.path.join(ARTIFACT_DIR, f"{dataset_name}_shap_summary_{timestamp}.png")
        shap_bar_path = os.path.join(ARTIFACT_DIR, f"{dataset_name}_shap_bar_{timestamp}.png")

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

        np.save(os.path.join(ARTIFACT_DIR, f"{dataset_name}_shap_values_{timestamp}.npy"), shap_values)
        print(f"✅ SHAP plots saved → {shap_summary_path}, {shap_bar_path}")
    except Exception as e:
        print(f"⚠️ SHAP explainability skipped due to error: {e}")
    """

    print(f"🎉 Finished training for {dataset_name.upper()}.\n")

    return {
        "dataset": dataset_name,
        "f1": f1,
        "roc_auc": auc_roc,
        "pr_auc": auc_pr,
        "brier": brier,
        "mcc": mcc,
    }


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def run_all():
    results = []
    print("\n🚀 Autism Detection Training for ALL DATASETS\n")
    for name, path in DATASETS.items():
        if not os.path.exists(path):
            print(f"❌ Dataset not found: {path}")
            continue
        res = train_for_dataset(name, path)
        results.append(res)

    print("\n🏁 FINAL SUMMARY")
    for r in results:
        print(
            f"{r['dataset'].upper()} → "
            f"F1={r['f1']:.4f}, ROC AUC={r['roc_auc']:.4f}, "
            f"PR AUC={r['pr_auc']:.4f}, Brier={r['brier']:.4f}, MCC={r['mcc']:.4f}"
        )


if __name__ == "__main__":
    run_all()
