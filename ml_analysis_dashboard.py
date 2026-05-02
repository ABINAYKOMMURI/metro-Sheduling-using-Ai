# ============================================================
# ML ANALYSIS DASHBOARD (STANDALONE - NO WEB)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# ============================================================
# GENERATE DATASETS
# ============================================================

def generate_dataset(size=200):

    df = pd.DataFrame({
        "fitness_days_remaining": np.random.randint(10, 120, size),
        "historical_minor_fault_count": np.random.randint(0, 6, size),
        "current_mileage_km": np.random.randint(50000, 200000, size)
    })

    df["days_since_last_induction"] = 90 - df["fitness_days_remaining"]
    df["vibration_trend"] = df["current_mileage_km"] / 200000
    df["mileage_last_30_days"] = df["current_mileage_km"] * 0.03
    df["risk_interaction"] = (
        df["vibration_trend"] * df["historical_minor_fault_count"]
    ) / 5

    # 🔥 Improved realistic label
    df["breakdown"] = (
        (df["fitness_days_remaining"] < 25) &
        (df["historical_minor_fault_count"] > 2)
    ) | (
        (df["vibration_trend"] > 0.6)
    )

    df["breakdown"] = df["breakdown"].astype(int)

    return df


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(df):

    features = [
        "fitness_days_remaining",
        "days_since_last_induction",
        "historical_minor_fault_count",
        "vibration_trend",
        "mileage_last_30_days",
        "risk_interaction"
    ]

    X = df[features]
    y = df["breakdown"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return model, acc, cm, X_test, y_test


# ============================================================
# PLOT FUNCTIONS
# ============================================================

def plot_accuracy_comparison(acc_dict):

    plt.figure()
    plt.bar(acc_dict.keys(), acc_dict.values())
    plt.title("Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.show()


def plot_confusion_matrix(cm):

    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i][j], ha="center", va="center")

    plt.show()


def plot_trend(df):

    plt.figure()
    plt.plot(df["vibration_trend"].values)
    plt.title("Vibration Trend Over Time")
    plt.xlabel("Time")
    plt.ylabel("Vibration")
    plt.show()


def plot_risk_distribution(df):

    counts = df["breakdown"].value_counts()

    plt.figure()
    plt.pie(counts, labels=["Safe", "Risk"], autopct="%1.1f%%")
    plt.title("Risk Distribution")
    plt.show()


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":

    print("Running ML Analysis...\n")

    # Multiple datasets
    datasets = {
        "Small": generate_dataset(200),
        "Medium": generate_dataset(500),
        "Large": generate_dataset(1000)
    }

    accuracy_results = {}

    for name, df in datasets.items():

        print(f"Processing {name} dataset...")

        model, acc, cm, X_test, y_test = train_model(df)

        print(f"Accuracy ({name}): {round(acc, 3)}")
        print("Confusion Matrix:")
        print(cm)
        print("-" * 40)

        accuracy_results[name] = acc

        # Show graphs
        plot_confusion_matrix(cm)
        plot_trend(df)
        plot_risk_distribution(df)

    # Compare all datasets
    plot_accuracy_comparison(accuracy_results)

    print("\n✅ Analysis Complete")