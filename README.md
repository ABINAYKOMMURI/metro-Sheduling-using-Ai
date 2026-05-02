# KMRL Train Induction System — Phase 1 Prototype

This is a self-contained Phase 1 prototype of the **KMRL Train Induction System**.

## What this contains

- Synthetic data generator for a 25-train fleet
- Rule-based scoring engine implementing:
  - Fitness certificate constraints and scoring
  - Maintenance / job card constraints and scoring
  - Mileage balancing
  - Branding contract urgency
  - Cleaning status
  - Depot position (A vs B tracks)
- Hard constraint auto-rejection (safety/legal)
- Ranked induction list (top 10 trains)
- Explainable reasons per train
- Simple Flask web dashboard to visualize nightly decisions
- CLI script to generate CSV outputs

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Run CLI scoring once

```bash
python main.py
```

This will:

- Generate synthetic fleet data for 25 trains
- Score them
- Save:
  - `data/synthetic_trains.csv`
  - `data/scored_trains.csv`
- Print the **top 10 induction list** and rejected trains summary.

### Run the web dashboard

```bash
export FLASK_APP=webapp.py        # On Windows: set FLASK_APP=webapp.py
flask run
```

Then open the URL shown in the terminal (usually http://127.0.0.1:5000).

The dashboard will:

- Generate a fresh synthetic dataset on each page load
- Score all trains
- Show:
  - Top 10 recommended trains with breakdown
  - Rejected trains with reasons
  - Fleet summary stats

## Project Layout

- `main.py` — CLI entry point
- `synthetic_data.py` — Synthetic dataset generator
- `scoring_engine.py` — Constraint checks and multi-criteria scoring
- `webapp.py` — Flask-based supervisor dashboard
- `data/` — CSV exports (created when you run `main.py`)
- `templates/` — HTML templates for the dashboard
- `static/` — CSS styles

## Notes

- This is **Phase 1 only**:
  - No live IBM Maximo / SCADA / CMRS integration
  - All data is **synthetic** but shaped to match the real-world logic
- Phase 2 can replace the synthetic generator with real API/data connectors
  while reusing the same scoring engine and dashboard.

