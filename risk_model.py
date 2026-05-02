# ============================================================
# risk_model.py (FINAL UPDATED – BLUE THEME + HIGH ACCURACY)
# ============================================================

from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import shap
import plotly.graph_objects as go
import plotly.io as pio

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Optional libs
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except:
    XGB_AVAILABLE = False

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

RISK_MODEL_FILE = MODEL_DIR / "risk_model.pkl"

# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "fitness_days_remaining",
    "days_since_last_induction",
    "historical_minor_fault_count",
    "vibration_trend",
    "mileage_last_30_days",
    "risk_interaction",
]

TARGET = "breakdown_within_7_days"

# ============================================================
# VALIDATION
# ============================================================

def validate_features(df):
    missing = [col for col in FEATURES if col not in df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")

# ============================================================
# TRAIN MODEL
# ============================================================

def train_risk_model(df):

    validate_features(df)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    # 🔥 MODEL SELECTION
    if XGB_AVAILABLE:
        model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric="logloss"
        )
        print("Using XGBoost")
    else:
        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=10,
            class_weight="balanced",
            random_state=42
        )
        print("Using RandomForest")

    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    print("\n=== MODEL REPORT ===")
    print(classification_report(y_test, y_pred))

    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_proba))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(model, RISK_MODEL_FILE)

    return model

# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    if not RISK_MODEL_FILE.exists():
        print("Training model...")
        from synthetic_data import generate_synthetic_fleet
        df = generate_synthetic_fleet(num_trains=500)
        return train_risk_model(df)

    return joblib.load(RISK_MODEL_FILE)

# ============================================================
# PREDICTION
# ============================================================

def predict_failure_probability(model, df):
    validate_features(df)
    return model.predict_proba(df[FEATURES])[:, 1]

# ============================================================
# SHAP CORE
# ============================================================

def _get_shap_values(model, df):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df[FEATURES])

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    return shap_values

# ============================================================
# SHAP EXPLANATION TEXT
# ============================================================

def get_risk_shap_explanation(model, df):

    try:
        shap_values = _get_shap_values(model, df)

        explanations = []

        for i in range(len(df)):
            values = shap_values[i]

            pairs = list(zip(FEATURES, values))

            pairs = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)

            text = " | ".join(
                [
                    f"{name}: {'↑' if val > 0 else '↓'}{round(float(val), 4)}"
                    for name, val in pairs
                ]
            )

            explanations.append(text)

        return explanations

    except Exception as e:
        print("SHAP error:", e)
        return ["SHAP unavailable"] * len(df)

# ============================================================
# 🔵 BLUE SHAP PLOT (UPDATED)
# ============================================================

def generate_shap_plot_html(model, df):

    try:
        shap_values = _get_shap_values(model, df)

        mean_abs = np.abs(shap_values).mean(axis=0)

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=mean_abs,
                y=FEATURES,
                orientation="h",
                marker=dict(
                    color="#3B82F6",   # 🔵 BLUE
                    line=dict(color="#1E3A8A", width=2)
                )
            )
        )

        fig.update_layout(
            title="Global Feature Importance (SHAP)",
            xaxis_title="Mean |SHAP Value|",
            template="plotly_dark",
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            font=dict(color="white"),
            height=450
        )

        fig.update_yaxes(autorange="reversed")

        return pio.to_html(fig, full_html=False)

    except Exception as e:
        print("SHAP plot error:", e)
        return "<p>SHAP unavailable</p>"

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def get_feature_importance(model):
    return dict(zip(FEATURES, model.feature_importances_))

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    from synthetic_data import generate_synthetic_fleet

    df = generate_synthetic_fleet(500)

    model = train_risk_model(df)

    print("\nFeature Importance:")
    print(get_feature_importance(model))