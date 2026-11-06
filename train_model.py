
import os
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

# -----------------------------------------
# 1. PROJECT PATHS
# -----------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR, "outputs", "cleaned_sensor_data.csv"
)

ORIGINAL_DATA_PATH = os.path.join(
    BASE_DIR, "data", "ai4i2020.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------------
# 2. LOAD SENSOR DATA
# -----------------------------------------

df = pd.read_csv(DATA_PATH)
original_df = pd.read_csv(ORIGINAL_DATA_PATH)

features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

X = df[features]

print("\nISOLATION FOREST ANOMALY DETECTION")
print("=" * 50)

print("Total observations:", len(X))
print("Sensor features:", len(features))

# -----------------------------------------
# 3. FEATURE SCALING
# -----------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# -----------------------------------------
# 4. TRAIN ISOLATION FOREST
# -----------------------------------------

model = IsolationForest(
    n_estimators=200,
    contamination=0.08,
    random_state=42
)

model.fit(X_scaled)

print("\nIsolation Forest model trained successfully!")

# -----------------------------------------
# 5. PREDICT ANOMALIES
# -----------------------------------------

predictions = model.predict(X_scaled)

# Higher score means more anomalous
anomaly_scores = -model.decision_function(X_scaled)

results = df.copy()

results["Anomaly_Prediction"] = predictions
results["Anomaly_Score"] = anomaly_scores

results["Status"] = results[
    "Anomaly_Prediction"
].map({
    1: "Normal",
    -1: "Anomaly"
})

# Save results
results.to_csv(
    os.path.join(OUTPUT_DIR, "anomaly_results.csv"),
    index=False
)

# -----------------------------------------
# 6. DISPLAY DETECTION RESULTS
# -----------------------------------------

normal_count = (predictions == 1).sum()
anomaly_count = (predictions == -1).sum()

print("\nANOMALY DETECTION RESULTS")
print("-" * 50)

print("Normal observations:", normal_count)
print("Anomalous observations:", anomaly_count)

print(
    "Anomaly percentage:",
    round(anomaly_count / len(predictions) * 100, 2),
    "%"
)

# -----------------------------------------
# 7. EVALUATION USING FAILURE LABELS
# -----------------------------------------

# Failure labels are used ONLY for evaluation.
# They are NOT used to train the model.

if "Machine failure" in original_df.columns:

    actual_labels = original_df[
        "Machine failure"
    ].astype(int)

    predicted_labels = (
        predictions == -1
    ).astype(int)

    print("\nPOST-TRAINING VALIDATION")
    print("-" * 50)

    print("\nConfusion Matrix:")
    print(confusion_matrix(actual_labels, predicted_labels))

    print("\nClassification Report:")
    print(
        classification_report(
            actual_labels,
            predicted_labels,
            zero_division=0
        )
    )

# -----------------------------------------
# 8. ANOMALY VISUALIZATION
# -----------------------------------------

plt.figure(figsize=(12, 6))

normal_mask = predictions == 1
anomaly_mask = predictions == -1

plt.scatter(
    range(len(predictions)),
    anomaly_scores,
    c=normal_mask,
    cmap="coolwarm",
    s=12,
    label="Observations"
)

plt.scatter(
    range(len(predictions)),
    anomaly_scores,
    c=anomaly_mask,
    cmap="coolwarm",
    s=12
)

plt.title("Isolation Forest - Anomaly Detection")
plt.xlabel("Observation Number")
plt.ylabel("Anomaly Score")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "isolation_forest_results.png")
)

plt.close()

# -----------------------------------------
# 9. SAVE TRAINED MODEL AND SCALER
# -----------------------------------------

joblib.dump(
    model,
    os.path.join(MODEL_DIR, "isolation_forest.pkl")
)

joblib.dump(
    scaler,
    os.path.join(MODEL_DIR, "scaler.pkl")
)

print("\nMODEL SAVED SUCCESSFULLY!")
print("Model: models/isolation_forest.pkl")
print("Scaler: models/scaler.pkl")
print("Results: outputs/anomaly_results.csv")
print("\nISOLATION FOREST COMPLETED!")
