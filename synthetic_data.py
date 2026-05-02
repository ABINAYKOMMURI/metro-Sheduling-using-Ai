import datetime as dt
from typing import List

import numpy as np
import pandas as pd


def generate_synthetic_fleet(
    num_trains: int = 200, seed: int | None = 42
) -> pd.DataFrame:
    """
    Generate synthetic fleet snapshot for Hybrid MCDA + ML system.

    Includes:
        - Operational fields (for MCDA)
        - ML features
        - ML target labels
    """

    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    # -------------------------------
    # Train IDs
    # -------------------------------
    train_ids: List[str] = [
        f"T{str(i).zfill(3)}" for i in range(1, num_trains + 1)
    ]

    # -------------------------------
    # Core Operational Fields (Phase-1)
    # -------------------------------
    fitness_days = rng.integers(0, 96, num_trains)

    job_status_choices = ["CLEAR", "MINOR", "CRITICAL"]
    job_status_probs = [0.6, 0.3, 0.1]
    job_cards = rng.choice(job_status_choices, size=num_trains, p=job_status_probs)

    mileage = rng.normal(loc=126_000, scale=8_000, size=num_trains)
    mileage = np.clip(mileage, 100_000, 150_000)

    branding_categories = ["NONE", "LOW", "MEDIUM", "HIGH"]
    branding_probs = [0.5, 0.15, 0.2, 0.15]
    branding_status = rng.choice(
        branding_categories, size=num_trains, p=branding_probs
    )

    penalty_risks = []
    for status in branding_status:
        if status == "HIGH":
            penalty_risks.append(float(rng.uniform(5.0, 15.0)))
        elif status == "MEDIUM":
            penalty_risks.append(float(rng.uniform(2.0, 5.0)))
        elif status == "LOW":
            penalty_risks.append(float(rng.uniform(0.0, 3.0)))
        else:
            penalty_risks.append(0.0)

    cleaning_status = rng.choice(
        ["DONE", "PENDING"], size=num_trains, p=[0.7, 0.3]
    )

    depot_track = rng.choice(
        ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4"],
        size=num_trains
    )

    # -------------------------------
    # Last Induction Date
    # -------------------------------
    today = dt.date.today()
    last_induction_dates = []
    days_since_last_induction = []

    for _ in range(num_trains):
        days_ago = int(rng.integers(0, 30))
        last_induction_dates.append(
            (today - dt.timedelta(days=days_ago)).isoformat()
        )
        days_since_last_induction.append(days_ago)

    # -------------------------------
    # ML FEATURE ENGINEERING
    # -------------------------------
    historical_minor_fault_count = rng.integers(0, 6, num_trains)

    vibration_trend = rng.uniform(0.0, 1.0, num_trains)

    mileage_last_30_days = rng.integers(2000, 6000, num_trains)

    # -------------------------------
    # SYNTHETIC TARGET GENERATION
    # -------------------------------

    # Breakdown Risk
    fitness_risk = 1 / (fitness_days + 1)

    breakdown_risk_score = (
        0.6 * vibration_trend
        + 0.5 * (historical_minor_fault_count / 5)
        + 0.3 * fitness_risk
        + 0.2 * (mileage_last_30_days / 6000)
        + 0.1 * (mileage / 200000)  # age_factor
        + 0.1 * np.array([1 if track.startswith('A') else 0 for track in depot_track])  # depot_efficiency
    )

    breakdown_probability = 1 / (
        1 + np.exp(-20 * (breakdown_risk_score - 0.5))
    )

    breakdown_within_7_days = rng.binomial(
        1, breakdown_probability
    )

    # Maintenance Need Risk
    maintenance_risk_score = (
        0.4 * (np.array(days_since_last_induction) / 30)
        + 0.3 * (historical_minor_fault_count / 5)
        + 0.3 * (mileage_last_30_days / 6000)
    )

    maintenance_probability = 1 / (
        1 + np.exp(-20 * (maintenance_risk_score - 0.5))
    )

    maintenance_required_3_7_days = rng.binomial(
        1, maintenance_probability
    )

    # -------------------------------
    # FINAL DATAFRAME
    # -------------------------------
    df = pd.DataFrame(
        {
            # Core Operational
            "train_id": train_ids,
            "fitness_days_remaining": fitness_days,
            "job_card_status": job_cards,
            "current_mileage_km": mileage.round(0).astype(int),
            "branding_contract_status": branding_status,
            "branding_penalty_risk_lakh": np.round(penalty_risks, 2),
            "cleaning_status": cleaning_status,
            "depot_track": depot_track,
            "last_induction_date": last_induction_dates,

            # ML Features
            "days_since_last_induction": days_since_last_induction,
            "historical_minor_fault_count": historical_minor_fault_count,
            "vibration_trend": vibration_trend,
            "mileage_last_30_days": mileage_last_30_days,

            # Derived Features
            "risk_interaction": (vibration_trend * historical_minor_fault_count) / 5,  # Normalized interaction
            "age_factor": mileage / 200000,  # Normalized age
            "depot_efficiency": [1.0 if track.startswith('A') else 0.5 for track in depot_track],  # Depot efficiency

            # ML Targets
            "breakdown_within_7_days": breakdown_within_7_days,
            "maintenance_required_3_7_days": maintenance_required_3_7_days,
        }
    )

    return df


if __name__ == "__main__":
    df = generate_synthetic_fleet(500)
    print(df.head())
