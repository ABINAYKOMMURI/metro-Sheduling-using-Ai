from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List, Dict, Any

import numpy as np
import pandas as pd


# Weights as per specification
WEIGHTS = {
    "fitness": 0.30,
    "maintenance": 0.25,
    "mileage": 0.15,
    "branding": 0.15,
    "cleaning": 0.10,
    "depot": 0.05,
}


def fitness_score(days_remaining: int) -> int:
    """Score fitness certificate days remaining using bucketed logic.

    60-90 days:   95-100
    30-59 days:   80-94
    15-29 days:   60-79
    7-14 days:    40-59
    4-6 days:     20-39
    1-3 days:     AUTO-REJECT (handled upstream)
    <=0 days:     AUTO-REJECT
    """
    if days_remaining >= 90:
        return 100
    if 60 <= days_remaining < 90:
        # linear 95-100
        return int(95 + 5 * (days_remaining - 60) / 30)
    if 30 <= days_remaining < 60:
        return int(80 + 14 * (days_remaining - 30) / 30)
    if 15 <= days_remaining < 30:
        return int(60 + 19 * (days_remaining - 15) / 15)
    if 7 <= days_remaining < 15:
        return int(40 + 19 * (days_remaining - 7) / 8)
    if 4 <= days_remaining < 7:
        return int(20 + 19 * (days_remaining - 4) / 3)
    # 1-3 or negative handled as reject; we won't call for those
    return 0


def maintenance_score(job_card_status: str) -> int:
    status = (job_card_status or "").upper()
    if status == "CLEAR":
        return 100
    if status == "MINOR":
        return 70
    # CRITICAL should already be auto-rejected
    return 0


def mileage_score(current_mileage: float, fleet_avg: float) -> int:
    """Score based on deviation from fleet average mileage.

    Underutilized (< -5%): bonus up to +20.
    Overutilized (> +5%): penalty up to -20.
    Within ±5%: neutral-ish.
    """
    if fleet_avg <= 0:
        return 80

    deviation_pct = (current_mileage - fleet_avg) / fleet_avg * 100.0

    # Within ±5%: neutral 80
    if -5 <= deviation_pct <= 5:
        return 80

    # Underutilized: lower mileage than average by more than 5%
    if deviation_pct < -5:
        # Max bonus at -20% or lower
        bonus = min(20, (-deviation_pct - 5) * 1.0)
        return int(80 + bonus)

    # Overutilized: higher mileage than average by more than 5%
    penalty = min(20, (deviation_pct - 5) * 1.0)
    return int(80 - penalty)


def branding_score(branding_status: str) -> int:
    status = (branding_status or "").upper()
    if status == "HIGH":
        return 100
    if status == "MEDIUM":
        return 70
    if status == "LOW":
        return 40
    # No contract = neutral baseline
    return 50


def cleaning_score(cleaning_status: str) -> int:
    status = (cleaning_status or "").upper()
    if status == "DONE":
        return 100
    if status == "PENDING":
        return 30
    return 0


def depot_position_score(depot_track: str) -> int:
    """A-track = easy access (100), B-track = shunting (0)."""
    if not depot_track:
        return 0
    return 100 if str(depot_track).upper().startswith("A") else 0


@dataclass
class TrainDecision:
    train_id: str
    total_score: float
    status: str  # "ELIGIBLE" or "REJECTED"
    rejection_reason: str
    explanation: str


def evaluate_train(row: pd.Series, fleet_avg_mileage: float) -> TrainDecision:
    """Apply hard constraints and weighted scoring for a single train."""
    train_id = row["train_id"]

    days_remaining = int(row["fitness_days_remaining"])
    job_card_status = str(row["job_card_status"])
    mileage = float(row["current_mileage_km"])
    branding_status = str(row["branding_contract_status"])
    cleaning_status = str(row["cleaning_status"])
    depot_track = str(row["depot_track"])

    # Hard constraints
    hard_reasons: List[str] = []

    if days_remaining <= 3:
        hard_reasons.append("Fitness certificate expires in <=3 days or has expired")

    if job_card_status.upper() == "CRITICAL":
        hard_reasons.append("Critical open job card (safety-critical defect)")

    # More hard constraints could be added here (e.g., mandatory systems)

    if hard_reasons:
        return TrainDecision(
            train_id=train_id,
            total_score=0.0,
            status="REJECTED",
            rejection_reason="; ".join(hard_reasons),
            explanation="Auto-rejected due to safety/legal constraints.",
        )

    # Soft scores
    f_score = fitness_score(days_remaining)
    m_score = maintenance_score(job_card_status)
    mile_score = mileage_score(mileage, fleet_avg_mileage)
    b_score = branding_score(branding_status)
    c_score = cleaning_score(cleaning_status)
    d_score = depot_position_score(depot_track)

    total = (
        f_score * WEIGHTS["fitness"]
        + m_score * WEIGHTS["maintenance"]
        + mile_score * WEIGHTS["mileage"]
        + b_score * WEIGHTS["branding"]
        + c_score * WEIGHTS["cleaning"]
        + d_score * WEIGHTS["depot"]
    )

    # Build explanation string
    drivers: List[str] = []

    if branding_status.upper() == "HIGH":
        drivers.append("branding contract is in HIGH urgency band")
    elif branding_status.upper() == "MEDIUM":
        drivers.append("branding contract has MEDIUM urgency")
    elif branding_status.upper() == "LOW":
        drivers.append("branding contract has LOW urgency")
    else:
        drivers.append("no active branding contract (neutral)")

    if mileage < fleet_avg_mileage * 0.95:
        drivers.append("underutilized mileage (boosted to balance fleet wear)")
    elif mileage > fleet_avg_mileage * 1.05:
        drivers.append("overutilized mileage (slightly deprioritized)")
    else:
        drivers.append("mileage close to fleet average")

    if depot_track.upper().startswith("A"):
        drivers.append("located on A-track (low shunting cost)")
    else:
        drivers.append("located on B-track (requires shunting)")

    if cleaning_status.upper() == "DONE":
        drivers.append("cleaning completed within last cycle")
    else:
        drivers.append("cleaning pending")

    if job_card_status.upper() == "MINOR":
        drivers.append("only MINOR job card issues open")
    else:
        drivers.append("no open job cards")

    explanation = (
        f"Overall score {total:.1f}/100. Key factors: "
        + "; ".join(drivers)
        + "."
    )

    return TrainDecision(
        train_id=train_id,
        total_score=float(total),
        status="ELIGIBLE",
        rejection_reason="",
        explanation=explanation,
    )


def score_fleet(df: pd.DataFrame) -> pd.DataFrame:
    """Score an entire fleet DataFrame and return with additional columns."""
    if df.empty:
        return df.copy()

    fleet_avg = float(df["current_mileage_km"].mean())

    decisions: List[TrainDecision] = []
    for _, row in df.iterrows():
        decisions.append(evaluate_train(row, fleet_avg))

    # Convert decisions to DataFrame
    dec_df = pd.DataFrame(
        {
            "train_id": [d.train_id for d in decisions],
            "total_score": [d.total_score for d in decisions],
            "status": [d.status for d in decisions],
            "rejection_reason": [d.rejection_reason for d in decisions],
            "explanation": [d.explanation for d in decisions],
        }
    )

    # Merge back onto original DF
    out = df.merge(dec_df, on="train_id", how="left")

    # Also keep component scores for transparency
    fleet_avg = float(out["current_mileage_km"].mean())
    out["fitness_score"] = out["fitness_days_remaining"].apply(fitness_score)
    out["maintenance_score"] = out["job_card_status"].apply(maintenance_score)
    out["mileage_score"] = out["current_mileage_km"].apply(
        lambda x: mileage_score(float(x), fleet_avg)
    )
    out["branding_score"] = out["branding_contract_status"].apply(branding_score)
    out["cleaning_score"] = out["cleaning_status"].apply(cleaning_score)
    out["depot_score"] = out["depot_track"].apply(depot_position_score)

    return out


def select_top_trains(
    scored_df: pd.DataFrame, top_n: int = 10
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (top_n_eligible, rejected) dataframes."""
    eligible = scored_df[scored_df["status"] == "ELIGIBLE"].copy()
    eligible = eligible.sort_values("total_score", ascending=False)

    top_eligible = eligible.head(top_n).reset_index(drop=True)

    rejected = scored_df[scored_df["status"] == "REJECTED"].copy()
    rejected = rejected.sort_values("train_id").reset_index(drop=True)

    return top_eligible, rejected


if __name__ == "__main__":
    # Quick self-test when run directly
    from synthetic_data import generate_synthetic_fleet

    df = generate_synthetic_fleet()
    scored = score_fleet(df)
    top10, rejected = select_top_trains(scored, top_n=10)

    print("Top 10 eligible trains:")
    print(top10[["train_id", "total_score", "status"]])

    print("\nRejected trains:")
    print(rejected[["train_id", "status", "rejection_reason"]])
