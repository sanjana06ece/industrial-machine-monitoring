
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Define project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data", "ai4i2020.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. Load dataset
df = pd.read_csv(DATA_PATH)

print("\n===== INDUSTRIAL MACHINE DATA ANALYSIS =====")

print("\nDataset Shape:", df.shape)

print("\nFirst 5 Records:")
print(df.head())

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())

# 3. Check missing and duplicate values
print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Records:", df.duplicated().sum())

# 4. Select sensor parameters
features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

sensor_data = df[features].copy()

# Convert values to numeric and handle invalid values
sensor_data = sensor_data.apply(
    pd.to_numeric, errors="coerce"
)

sensor_data = sensor_data.replace(
    [np.inf, -np.inf], np.nan
)

sensor_data = sensor_data.fillna(
    sensor_data.median()
)

# Save cleaned sensor data
sensor_data.to_csv(
    os.path.join(OUTPUT_DIR, "cleaned_sensor_data.csv"),
    index=False
)

# 5. Generate distribution graphs
for column in features:

    plt.figure(figsize=(8, 5))

    sns.histplot(sensor_data[column], kde=True)

    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")

    plt.tight_layout()

    filename = column.replace(" ", "_").replace("/", "_")

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            filename + "_distribution.png"
        )
    )

    plt.close()

# 6. Generate boxplot
plt.figure(figsize=(11, 6))

sns.boxplot(data=sensor_data)

plt.title("Industrial Sensor Parameters - Boxplot")
plt.xticks(rotation=20)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "sensor_boxplot.png")
)

plt.close()

# 7. Generate correlation heatmap
plt.figure(figsize=(9, 6))

correlation = sensor_data.corr()

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Sensor Parameter Correlation")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "correlation_heatmap.png")
)

plt.close()

# 8. Generate scatter plot
plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=sensor_data,
    x="Rotational speed [rpm]",
    y="Torque [Nm]"
)

plt.title("Rotational Speed vs Torque")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "speed_vs_torque.png")
)

plt.close()

# 9. Final results
print("\n===== ANALYSIS COMPLETED =====")

print("Total Records:", len(sensor_data))
print("Total Sensor Features:", len(features))

print("\nCleaned dataset and graphs saved in outputs folder.")
