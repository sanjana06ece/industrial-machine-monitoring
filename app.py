
import os
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px

# -----------------------------------------
# 1. PROJECT CONFIGURATION
# -----------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR, "outputs", "cleaned_sensor_data.csv"
)

ANOMALY_PATH = os.path.join(
    BASE_DIR, "outputs", "anomaly_results.csv"
)

CLUSTER_PATH = os.path.join(
    BASE_DIR, "outputs", "cluster_results.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")

FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

st.set_page_config(
    page_title="Industrial Machine Health Monitoring",
    page_icon="🏭",
    layout="wide"
)

# -----------------------------------------
# 2. LOAD DATA AND MODELS
# -----------------------------------------

@st.cache_data
def load_data():
    sensor = pd.read_csv(DATA_PATH)
    anomaly = pd.read_csv(ANOMALY_PATH)
    clusters = pd.read_csv(CLUSTER_PATH)

    return sensor, anomaly, clusters


@st.cache_resource
def load_models():
    isolation_model = joblib.load(
        os.path.join(MODEL_DIR, "isolation_forest.pkl")
    )

    isolation_scaler = joblib.load(
        os.path.join(MODEL_DIR, "scaler.pkl")
    )

    kmeans_model = joblib.load(
        os.path.join(MODEL_DIR, "kmeans_model.pkl")
    )

    kmeans_scaler = joblib.load(
        os.path.join(MODEL_DIR, "kmeans_scaler.pkl")
    )

    pca_model = joblib.load(
        os.path.join(MODEL_DIR, "pca_model.pkl")
    )

    return (
        isolation_model,
        isolation_scaler,
        kmeans_model,
        kmeans_scaler,
        pca_model
    )


# -----------------------------------------
# 3. CHECK REQUIRED FILES
# -----------------------------------------

required_files = [
    DATA_PATH,
    ANOMALY_PATH,
    CLUSTER_PATH,
    os.path.join(MODEL_DIR, "isolation_forest.pkl"),
    os.path.join(MODEL_DIR, "scaler.pkl"),
    os.path.join(MODEL_DIR, "kmeans_model.pkl"),
    os.path.join(MODEL_DIR, "kmeans_scaler.pkl"),
    os.path.join(MODEL_DIR, "pca_model.pkl")
]

missing_files = [
    file for file in required_files
    if not os.path.exists(file)
]

if missing_files:
    st.error("Required project files are missing.")

    for file in missing_files:
        st.write(file)

    st.stop()


sensor_data, anomaly_data, cluster_data = load_data()

(
    isolation_model,
    isolation_scaler,
    kmeans_model,
    kmeans_scaler,
    pca_model
) = load_models()


# -----------------------------------------
# 4. DASHBOARD HEADER
# -----------------------------------------

st.title("🏭 Smart Industrial Machine Health Monitoring")

st.subheader(
    "Unsupervised Learning-Based Anomaly Detection System"
)

st.write(
    "An intelligent monitoring prototype using "
    "Isolation Forest and K-Means clustering."
)

st.divider()


# -----------------------------------------
# 5. SIDEBAR NAVIGATION
# -----------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Dashboard Section",
    [
        "Overview",
        "Anomaly Detection",
        "K-Means Clustering",
        "Machine Assessment"
    ]
)


# -----------------------------------------
# 6. OVERVIEW
# -----------------------------------------

if page == "Overview":

    st.header("Machine Monitoring Overview")

    total = len(anomaly_data)

    normal_count = (
        anomaly_data["Status"] == "Normal"
    ).sum()

    anomaly_count = (
        anomaly_data["Status"] == "Anomaly"
    ).sum()

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Observations", f"{total:,}")
    col2.metric("Normal Observations", f"{normal_count:,}")
    col3.metric("Detected Anomalies", f"{anomaly_count:,}")

    st.divider()

    st.subheader("Sensor Parameter Distribution")

    selected_feature = st.selectbox(
        "Select Sensor Parameter",
        FEATURES
    )

    fig = px.histogram(
        sensor_data,
        x=selected_feature,
        nbins=40,
        title=f"Distribution of {selected_feature}"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sensor Data Preview")

    st.dataframe(
        sensor_data.head(20),
        use_container_width=True
    )

    st.download_button(
        "Download Cleaned Sensor Dataset",
        sensor_data.to_csv(index=False),
        file_name="cleaned_sensor_data.csv",
        mime="text/csv"
    )


# -----------------------------------------
# 7. ANOMALY DETECTION
# -----------------------------------------

elif page == "Anomaly Detection":

    st.header("Isolation Forest Anomaly Detection")

    st.write(
        "This model identifies unusual sensor patterns "
        "without using machine failure labels during training."
    )

    total = len(anomaly_data)

    normal_count = (
        anomaly_data["Status"] == "Normal"
    ).sum()

    anomaly_count = (
        anomaly_data["Status"] == "Anomaly"
    ).sum()

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Records", f"{total:,}")
    col2.metric("Normal", f"{normal_count:,}")
    col3.metric("Anomalies", f"{anomaly_count:,}")

    st.divider()

    fig = px.pie(
        anomaly_data,
        names="Status",
        title="Normal vs Anomalous Observations",
        hole=0.4
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Anomaly Score Distribution")

    fig = px.histogram(
        anomaly_data,
        x="Anomaly_Score",
        color="Status",
        nbins=40,
        title="Distribution of Anomaly Scores"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Detected Anomalous Records")

    anomalies = anomaly_data[
        anomaly_data["Status"] == "Anomaly"
    ].sort_values(
        "Anomaly_Score",
        ascending=False
    )

    st.dataframe(
        anomalies.head(100),
        use_container_width=True
    )

    st.download_button(
        "Download Anomaly Results",
        anomaly_data.to_csv(index=False),
        file_name="anomaly_results.csv",
        mime="text/csv"
    )


# -----------------------------------------
# 8. K-MEANS CLUSTERING
# -----------------------------------------

elif page == "K-Means Clustering":

    st.header("Machine Operating Pattern Analysis")

    st.write(
        "K-Means groups similar machine operating "
        "conditions into three clusters."
    )

    cluster_counts = (
        cluster_data["Cluster"]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    cluster_counts.columns = [
        "Cluster", "Observations"
    ]

    fig = px.bar(
        cluster_counts,
        x="Cluster",
        y="Observations",
        color="Cluster",
        title="Machine Operating Cluster Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("PCA Cluster Visualization")

    fig = px.scatter(
        cluster_data,
        x="PCA1",
        y="PCA2",
        color="Cluster",
        title="K-Means Clusters Using PCA",
        opacity=0.65
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cluster Sensor Profiles")

    profile = cluster_data.groupby(
        "Cluster"
    )[FEATURES].mean().round(2)

    st.dataframe(
        profile,
        use_container_width=True
    )

    st.download_button(
        "Download Cluster Results",
        cluster_data.to_csv(index=False),
        file_name="cluster_results.csv",
        mime="text/csv"
    )


# -----------------------------------------
# 9. INDIVIDUAL MACHINE ASSESSMENT
# -----------------------------------------

elif page == "Machine Assessment":

    st.header("Individual Machine Sensor Assessment")

    st.write(
        "Enter machine sensor readings to identify "
        "unusual operating patterns."
    )

    with st.form("machine_sensor_form"):

        col1, col2 = st.columns(2)

        with col1:

            air_temp = st.number_input(
                "Air Temperature (K)",
                min_value=250.0,
                max_value=350.0,
                value=300.0
            )

            process_temp = st.number_input(
                "Process Temperature (K)",
                min_value=250.0,
                max_value=400.0,
                value=310.0
            )

            speed = st.number_input(
                "Rotational Speed (rpm)",
                min_value=500,
                max_value=4000,
                value=1500
            )

        with col2:

            torque = st.number_input(
                "Torque (Nm)",
                min_value=0.0,
                max_value=100.0,
                value=40.0
            )

            tool_wear = st.number_input(
                "Tool Wear (minutes)",
                min_value=0,
                max_value=300,
                value=100
            )

        submitted = st.form_submit_button(
            "Analyse Machine",
            use_container_width=True
        )

    if submitted:

        input_data = pd.DataFrame(
            [[
                air_temp,
                process_temp,
                speed,
                torque,
                tool_wear
            ]],
            columns=FEATURES
        )

        # Isolation Forest prediction
        scaled_input = isolation_scaler.transform(
            input_data
        )

        prediction = isolation_model.predict(
            scaled_input
        )[0]

        anomaly_score = -isolation_model.decision_function(
            scaled_input
        )[0]

        # K-Means prediction
        scaled_cluster_input = kmeans_scaler.transform(
            input_data
        )

        cluster_prediction = kmeans_model.predict(
            scaled_cluster_input
        )[0]

        st.divider()

        st.subheader("Machine Assessment Results")

        if prediction == -1:

            st.error(
                "UNUSUAL SENSOR PATTERN DETECTED"
            )

            st.write(
                "The Isolation Forest model classified "
                "this reading as an anomaly."
            )

        else:

            st.success(
                "NO ANOMALY DETECTED"
            )

            st.write(
                "The sensor reading was classified "
                "as normal by the Isolation Forest model."
            )

        col1, col2 = st.columns(2)

        col1.metric(
            "Anomaly Score",
            f"{anomaly_score:.4f}"
        )

        col2.metric(
            "Operating Cluster",
            f"Cluster {cluster_prediction}"
        )

        st.info(
            "This is an algorithmic anomaly assessment, "
            "not a confirmed machine-failure diagnosis."
        )


# -----------------------------------------
# 10. FOOTER
# -----------------------------------------

st.divider()

st.caption(
    "Major Project | Unsupervised Machine Learning | "
    "Isolation Forest and K-Means"
)

st.caption(
    "Educational prototype using the AI4I 2020 "
    "synthetic predictive maintenance dataset. "
    "Not validated for real industrial safety decisions."
)
