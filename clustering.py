
import os
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# -----------------------------------------
# 1. PROJECT PATHS
# -----------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR, "outputs", "cleaned_sensor_data.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------------
# 2. LOAD DATA
# -----------------------------------------

df = pd.read_csv(DATA_PATH)

features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

X = df[features]

print("\nK-MEANS MACHINE OPERATING PATTERN ANALYSIS")
print("=" * 50)

print("Total observations:", len(X))

# -----------------------------------------
# 3. STANDARDIZE DATA
# -----------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# -----------------------------------------
# 4. TRAIN K-MEANS
# -----------------------------------------

model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

clusters = model.fit_predict(X_scaled)

df["Cluster"] = clusters

print("\nK-Means clustering completed!")

# -----------------------------------------
# 5. EVALUATE CLUSTERS
# -----------------------------------------

score = silhouette_score(
    X_scaled,
    clusters,
    sample_size=3000,
    random_state=42
)

print("\nSILHOUETTE SCORE:", round(score, 4))

print("\nCLUSTER DISTRIBUTION:")
print(df["Cluster"].value_counts().sort_index())

# -----------------------------------------
# 6. PCA VISUALIZATION
# -----------------------------------------

pca = PCA(n_components=2)

reduced_data = pca.fit_transform(X_scaled)

df["PCA1"] = reduced_data[:, 0]
df["PCA2"] = reduced_data[:, 1]

plt.figure(figsize=(10, 6))

plt.scatter(
    df["PCA1"],
    df["PCA2"],
    c=clusters,
    cmap="viridis",
    s=12
)

plt.title("K-Means Machine Operating Clusters")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "kmeans_clusters.png")
)

plt.close()

# -----------------------------------------
# 7. CLUSTER PROFILE
# -----------------------------------------

profile = df.groupby("Cluster")[features].mean()

print("\nCLUSTER SENSOR PROFILES:")
print(profile.round(2))

profile.to_csv(
    os.path.join(OUTPUT_DIR, "cluster_profiles.csv")
)

# -----------------------------------------
# 8. SAVE RESULTS
# -----------------------------------------

df.to_csv(
    os.path.join(OUTPUT_DIR, "cluster_results.csv"),
    index=False
)

joblib.dump(
    model,
    os.path.join(MODEL_DIR, "kmeans_model.pkl")
)

joblib.dump(
    scaler,
    os.path.join(MODEL_DIR, "kmeans_scaler.pkl")
)

joblib.dump(
    pca,
    os.path.join(MODEL_DIR, "pca_model.pkl")
)

print("\nK-MEANS MODEL SAVED SUCCESSFULLY!")
print("Results: outputs/cluster_results.csv")
print("Profiles: outputs/cluster_profiles.csv")
print("Graph: outputs/kmeans_clusters.png")
print("\nCLUSTERING COMPLETED!")
