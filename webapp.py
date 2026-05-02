# ============================================================
# webapp.py (FINAL STABLE VERSION – NO ERRORS)
# ============================================================

from pathlib import Path
import pandas as pd
from flask import Flask, render_template, request

from sklearn.metrics import confusion_matrix

from synthetic_data import generate_synthetic_fleet
from hybrid_engine_v2 import run_full_hybrid_engine, select_top_hybrid

from risk_model import load_model, get_risk_shap_explanation, generate_shap_plot_html
from sandbox_ml_engine import run_sandbox_ml


# ============================================================
# APP SETUP
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/")
def dashboard():

    df = generate_synthetic_fleet(num_trains=25, seed=None)

    hybrid_df = run_full_hybrid_engine(df)
    top10, rejected = select_top_hybrid(hybrid_df, top_n=10)

    model = load_model()

    # -----------------------------
    # SHAP
    # -----------------------------
    try:
        shap_explanations = get_risk_shap_explanation(model, top10)
        top10["shap_explanation"] = shap_explanations
    except:
        top10["shap_explanation"] = ["Unavailable"] * len(top10)

    try:
        shap_plot_html = generate_shap_plot_html(model, df)
    except:
        shap_plot_html = "<p>SHAP unavailable</p>"

    # ============================================================
    # CONFUSION MATRIX
    # ============================================================

    df["breakdown_within_7_days"] = (
        (df["fitness_days_remaining"] < 20) |
        (df["historical_minor_fault_count"] > 3)
    ).astype(int)

    features = [
        "fitness_days_remaining",
        "days_since_last_induction",
        "historical_minor_fault_count",
        "vibration_trend",
        "mileage_last_30_days",
        "risk_interaction"
    ]

    X = df[features]
    y_true = df["breakdown_within_7_days"]

    probs = model.predict_proba(X)[:, 1]
    y_pred = (probs > 0.5).astype(int)

    cm = confusion_matrix(y_true, y_pred)

    confusion_data = {
        "tn": int(cm[0][0]),
        "fp": int(cm[0][1]),
        "fn": int(cm[1][0]),
        "tp": int(cm[1][1]),
    }

    # -----------------------------
    # METRICS
    # -----------------------------
    accuracy = (cm[0][0] + cm[1][1]) / cm.sum()
    precision = cm[1][1] / (cm[1][1] + cm[0][1] + 1e-6)
    recall = cm[1][1] / (cm[1][1] + cm[1][0] + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    metrics = {
        "accuracy": round(float(accuracy), 2),
        "precision": round(float(precision), 2),
        "recall": round(float(recall), 2),
        "f1": round(float(f1), 2),
    }

    # ============================================================
    # 📊 CHART DATA (FIXED)
    # ============================================================

    try:
        bar_labels = top10["train_id"].astype(str).tolist()
        bar_values = top10["final_hybrid_score"].astype(float).round(2).tolist()
    except:
        bar_labels = []
        bar_values = []

    try:
        status_counts = hybrid_df["hybrid_status"].value_counts().to_dict()
        pie_labels = [str(k) for k in status_counts.keys()]
        pie_values = [int(v) for v in status_counts.values()]
    except:
        pie_labels = []
        pie_values = []

    # -----------------------------
    # METADATA
    # -----------------------------
    metadata = {
        "fleet_size": int(len(hybrid_df)),
        "eligible_count": int((hybrid_df["hybrid_status"] != "DEFER").sum()),
        "rejected_count": int((hybrid_df["status"] == "REJECTED").sum()),
        "fleet_avg_mileage": float(round(hybrid_df["current_mileage_km"].mean(), 0)),
    }

    return render_template(
        "dashboard.html",
        top_trains=top10.to_dict(orient="records"),
        rejected_trains=rejected.to_dict(orient="records"),
        metadata=metadata,
        shap_plot_html=shap_plot_html,
        confusion_data=confusion_data,
        metrics=metrics,
        bar_labels=bar_labels,
        bar_values=bar_values,
        pie_labels=pie_labels,
        pie_values=pie_values
    )


# ============================================================
# SANDBOX
# ============================================================

@app.route("/sandbox", methods=["GET", "POST"])
def sandbox():

    result = None

    default_inputs = {
        "fitness_days_remaining": 30,
        "job_card_status": "MINOR",
        "current_mileage_km": 126000,
        "branding_status": "LOW",
        "cleaning_status": "DONE",
        "depot_track": "A3",
    }

    if request.method == "POST":

        fitness_days = int(request.form.get("fitness_days_remaining"))
        job_card_status = request.form.get("job_card_status")
        current_mileage = float(request.form.get("current_mileage_km"))
        branding_status = request.form.get("branding_status")
        cleaning_status = request.form.get("cleaning_status")
        depot_track = request.form.get("depot_track")

        default_inputs.update({
            "fitness_days_remaining": fitness_days,
            "job_card_status": job_card_status,
            "current_mileage_km": current_mileage,
            "branding_status": branding_status,
            "cleaning_status": cleaning_status,
            "depot_track": depot_track,
        })

        fault = 6 if job_card_status == "CRITICAL" else 3 if job_card_status == "MINOR" else 1
        vibration = min(1.0, current_mileage / 200000)

        df = pd.DataFrame([{
            "train_id": "TXX",
            "fitness_days_remaining": fitness_days,
            "job_card_status": job_card_status,
            "current_mileage_km": current_mileage,
            "branding_contract_status": branding_status,
            "cleaning_status": cleaning_status,
            "depot_track": depot_track,
            "days_since_last_induction": max(1, 90 - fitness_days),
            "historical_minor_fault_count": fault,
            "vibration_trend": vibration,
            "mileage_last_30_days": 3500,
            "risk_interaction": (vibration * fault) / 5,
        }])

        hybrid_df = run_full_hybrid_engine(df)
        row = hybrid_df.iloc[0]

        ml = run_sandbox_ml(
            fitness_days,
            job_card_status,
            current_mileage,
            branding_status,
            cleaning_status,                                                         
            depot_track
        )

        breakdown_prob = ml["breakdown_probability"]
        maintenance_prob = ml["maintenance_probability"]

        certainty = abs(breakdown_prob - 0.5) * 2
        agreement = 1 - abs(breakdown_prob - maintenance_prob)
        stability = 1 if job_card_status == "CLEAR" else 0.7 if job_card_status == "MINOR" else 0.3

        confidence_score = round((0.5 * certainty + 0.3 * agreement + 0.2 * stability) * 100, 1)
        confidence_label = "HIGH" if confidence_score >= 80 else "MEDIUM" if confidence_score >= 50 else "LOW"

        result = {
            "hybrid_status": row["hybrid_status"],
            "final_hybrid_score": round(float(row["final_hybrid_score"]), 2),
            "risk_probability": breakdown_prob,
            "maintenance_probability": maintenance_prob,
            "breakdown_risk_band": ml["breakdown_risk_band"],
            "maintenance_risk_band": ml["maintenance_risk_band"],
            "confidence_score": confidence_score,
            "confidence_label": confidence_label,
        }

    return render_template("sandbox.html", result=result, default_inputs=default_inputs)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    app.run(debug=True)
