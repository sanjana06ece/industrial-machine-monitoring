
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

# Load dataset
df = pd.read_csv("data/ai4i2020.csv")

# Select sensor features
features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

X = df[features]
y = df["Machine failure"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Scale using training data only
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Isolation Forest without failure labels
model = IsolationForest(
    n_estimators=200,
    contamination=0.08,
    random_state=42
)

model.fit(X_train_scaled)

# Predict anomalies on unseen test data
predictions = model.predict(X_test_scaled)

# Convert predictions:
# -1 = anomaly (1)
#  1 = normal  (0)
predictions = [1 if value == -1 else 0 for value in predictions]

# Evaluate using test labels only
print("\nISOLATION FOREST - TEST SET EVALUATION")
print("=" * 50)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

print("Precision:", round(
    precision_score(y_test, predictions, zero_division=0), 4
))

print("Recall:", round(
    recall_score(y_test, predictions, zero_division=0), 4
))

print("F1-score:", round(
    f1_score(y_test, predictions, zero_division=0), 4
))

print("\nTest evaluation completed successfully!")
