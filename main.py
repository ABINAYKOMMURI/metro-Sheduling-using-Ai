from pathlib import Path

from synthetic_data import generate_synthetic_fleet
from scoring_engine import score_fleet, select_top_trains


def main():
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    data_dir.mkdir(exist_ok=True)

    print("Generating synthetic fleet data...")
    df = generate_synthetic_fleet(num_trains=25, seed=42)

    print("Scoring fleet...")
    scored = score_fleet(df)

    top10, rejected = select_top_trains(scored, top_n=10)

    # Save CSVs
    fleet_path = data_dir / "synthetic_trains.csv"
    scored_path = data_dir / "scored_trains.csv"
    top10_path = data_dir / "top10_trains.csv"
    rejected_path = data_dir / "rejected_trains.csv"

    df.to_csv(fleet_path, index=False)
    scored.to_csv(scored_path, index=False)
    top10.to_csv(top10_path, index=False)
    rejected.to_csv(rejected_path, index=False)

    print(f"Saved raw fleet data to {fleet_path}")
    print(f"Saved scored fleet data to {scored_path}")
    print(f"Saved top 10 list to {top10_path}")
    print(f"Saved rejected trains list to {rejected_path}")

    print("\n=== TOP 10 TRAINS FOR INDUCTION ===")
    print(
        top10[
            [
                "train_id",
                "total_score",
                "fitness_days_remaining",
                "job_card_status",
                "branding_contract_status",
                "cleaning_status",
                "depot_track",
            ]
        ]
    )

    print("\n=== REJECTED TRAINS (HARD CONSTRAINTS) ===")
    if rejected.empty:
        print("No trains were auto-rejected.")
    else:
        print(rejected[["train_id", "rejection_reason"]])


if __name__ == "__main__":
    main()