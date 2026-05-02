from pathlib import Path
import joblib
import pandas as pd
import shap

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# ============================================================
# Paths
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

MAINT_MODEL_FILE = MODEL_DIR / "maintenance_model.pkl"


# ============================================================
# Feature Definition
# ============================================================
FEATURES = [
    "days_since_last_induction",
    "historical_minor_fault_count",
    "vibration_trend",
    "mileage_last_30_days",
    "current_mileage_km",
]

TARGET = "maintenance_required_3_7_days"


# ============================================================
# Train Model
# ============================================================
def train_maintenance_model(df: pd.DataFrame):

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.30,
        random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=6,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n=== Maintenance Model Evaluation ===")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, MAINT_MODEL_FILE)
    print(f"\nModel saved to {MAINT_MODEL_FILE}")

    return model


# ============================================================
# LOAD MODEL  ✅ (Matches hybrid engine)
# ============================================================
def load_maintenance_model():
    if not MAINT_MODEL_FILE.exists():
        raise FileNotFoundError(
            "Maintenance model not trained yet. Run maintenance_model.py first."
        )
    return joblib.load(MAINT_MODEL_FILE)


# ============================================================
# Predict Maintenance Probability
# ============================================================
def predict_maintenance_probability(model, df: pd.DataFrame):

    X = df[FEATURES]

    probabilities = model.predict_proba(X)[:, 1]

    return probabilities


# ============================================================
# SHAP Explainability (Optional but Ready)
# ============================================================
def explain_maintenance_prediction(model, df: pd.DataFrame):

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df[FEATURES])

    return shap_values


# ============================================================
# Standalone Test
# ============================================================
if __name__ == "__main__":

    from synthetic_data import generate_synthetic_fleet

    print("Generating synthetic data...")
    df = generate_synthetic_fleet(num_trains=500)

    print("Training maintenance model...")
    model = train_maintenance_model(df)

    sample_probs = predict_maintenance_probability(model, df.head())

    print("\nSample Maintenance Probabilities:")
    print(sample_probs)
