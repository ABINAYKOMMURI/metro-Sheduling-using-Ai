# ============================================================
# sandbox_ml_engine.py (FINAL WORKING VERSION)
# ============================================================

import pandas as pd
from risk_model import load_model


# ============================================================
# Feature Engineering Layer
# ============================================================

def build_ml_features_from_sandbox_input(
    fitness_days,
    job_card_status,
    current_mileage,
    branding_status,
    cleaning_status,
    depot_track,
):
    """
    Converts sandbox inputs into ML features.
    MUST match training features EXACTLY (order + names).
    """

    days_since_last_induction = max(1, 90 - fitness_days)

    if job_card_status == "CRITICAL":
        fault = 6
    elif job_card_status == "MINOR":
        fault = 3
    else:
        fault = 1

    vibration = min(1.0, current_mileage / 200000)

    mileage_last_30_days = max(2000, min(5000, current_mileage * 0.03))

    risk_interaction = (vibration * fault) / 5

    # 🔥 STRICT FEATURE ORDER (VERY IMPORTANT)
    df = pd.DataFrame([[
        fitness_days,
        days_since_last_induction,
        fault,
        vibration,
        mileage_last_30_days,
        risk_interaction
    ]], columns=[
        "fitness_days_remaining",
        "days_since_last_induction",
        "historical_minor_fault_count",
        "vibration_trend",
        "mileage_last_30_days",
        "risk_interaction"
    ])

    return df


# ============================================================
# Risk Band Logic
# ============================================================

def risk_band(p):
    if p < 0.2:
        return "LOW"
    elif p < 0.5:
        return "MEDIUM"
    else:
        return "HIGH"


# ============================================================
# Main Sandbox ML Execution
# ============================================================

def run_sandbox_ml(
    fitness_days,
    job_card_status,
    current_mileage,
    branding_status,
    cleaning_status,
    depot_track,
):

    try:
        # 🔥 ALWAYS LOAD FRESH MODEL (CRITICAL FIX)
        model = load_model()

        if model is None:
            raise ValueError("Model not loaded")

        # Build features
        df = build_ml_features_from_sandbox_input(
            fitness_days,
            job_card_status,
            current_mileage,
            branding_status,
            cleaning_status,
            depot_track,
        )

        print("🔍 INPUT FEATURES:")
        print(df)

        print("🔍 MODEL EXPECTS:")
        print(model.feature_names_in_)

        # Predict
        prob = model.predict_proba(df)[0][1]

        maintenance = min(1.0, prob + 0.12)

        return {
            "breakdown_probability": round(float(prob), 3),
            "maintenance_probability": round(float(maintenance), 3),
            "breakdown_risk_band": risk_band(prob),
            "maintenance_risk_band": risk_band(maintenance),
        }

    except Exception as e:
        print("❌ REAL ERROR:", e)

        return {
            "breakdown_probability": 0,
            "maintenance_probability": 0,
            "breakdown_risk_band": "ERROR",
            "maintenance_risk_band": "ERROR",
        }