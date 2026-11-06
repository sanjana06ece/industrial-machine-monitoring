
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Load dataset
df = pd.read_csv("data/ai4i2020.csv")

features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

X = df[features]
y = df["Machine failure"]

# Same train-test split for both models
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Scale features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

results = []

# Compare two contamination values
for contamination in [0.05, 0.08]:

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42
    )

    # Train without using failure labels
    model.fit(X_train_scaled)

    # Predict unseen test records
    predictions = model.predict(X_test_scaled)

    # Convert -1 (anomaly) to 1 and 1 (normal) to 0
    predictions = [1 if p == -1 else 0 for p in predictions]

    precision = precision_score(
        y_test, predictions, zero_division=0
    )

    recall = recall_score(
        y_test, predictions, zero_division=0
    )

    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    cm = confusion_matrix(y_test, predictions)

    results.append({
        "Contamination": contamination,
        "Detected Anomalies": sum(predictions),
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    print("\nContamination:", contamination)
    print("Confusion Matrix:\n", cm)
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))

# Display comparison
comparison = pd.DataFrame(results)

print("\nFINAL CONTAMINATION COMPARISON")
print("=" * 50)
print(comparison.to_string(index=False))

# Save results
comparison.to_csv(
    "outputs/contamination_test_comparison.csv",
    index=False
)

print("\nComparison saved successfully!")
