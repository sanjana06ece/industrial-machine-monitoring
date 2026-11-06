
import os
import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ----------------------------------
# 1. PROJECT PATHS
# ----------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR, "data", "ai4i2020.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR, "outputs"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------
# 2. LOAD DATA
# ----------------------------------

df = pd.read_csv(DATA_PATH)

features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

X = df[features]

# Failure labels are used only for
# evaluation, never for model training.

y_true = df["Machine failure"].astype(int)

# ----------------------------------
# 3. SCALE FEATURES
# ----------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ----------------------------------
# 4. TEST DIFFERENT CONTAMINATION
# ----------------------------------

contamination_values = [
    0.01, 0.02, 0.03, 0.05, 0.08
]

results = []

print("\nISOLATION FOREST OPTIMISATION")
print("=" * 55)

for contamination in contamination_values:

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42
    )

    model.fit(X_scaled)

    predictions = model.predict(X_scaled)

    # Convert -1 anomaly and 1 normal
    # into binary evaluation labels.

    y_pred = (predictions == -1).astype(int)

    precision = precision_score(
        y_true, y_pred, zero_division=0
    )

    recall = recall_score(
        y_true, y_pred, zero_division=0
    )

    f1 = f1_score(
        y_true, y_pred, zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    results.append({
        "Contamination": contamination,
        "Detected_Anomalies": int(y_pred.sum()),
        "True_Positive": int(tp),
        "False_Positive": int(fp),
        "False_Negative": int(fn),
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1
    })

    print(f"\nContamination: {contamination}")
    print("Detected anomalies:", int(y_pred.sum()))
    print("True positives:", tp)
    print("False positives:", fp)
    print("False negatives:", fn)
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-score:", round(f1, 4))

# ----------------------------------
# 5. SAVE COMPARISON RESULTS
# ----------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "isolation_forest_optimization.csv"
    ),
    index=False
)

print("\nOPTIMISATION SUMMARY")
print("=" * 55)

print(
    results_df[
        [
            "Contamination",
            "Detected_Anomalies",
            "Precision",
            "Recall",
            "F1_Score"
        ]
    ].to_string(index=False)
)

best_row = results_df.loc[
    results_df["F1_Score"].idxmax()
]

print("\nBest F1-score contamination:")
print(best_row["Contamination"])

print("Best F1-score:", round(best_row["F1_Score"], 4))

print("\nResults saved successfully!")
