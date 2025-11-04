import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from prophet import Prophet
from sklearn.ensemble import IsolationForest

# ----------------- UI Styling -----------------
st.markdown(
    """
    <style>
    .title {
        font-size: 36px;
        background: linear-gradient(90deg, #00bcd4, #007acc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
        text-align: center;
    }
    </style>
    <h1 class="title">Cyber Threat Intelligence Dashboard</h1>
    """,
    unsafe_allow_html=True
)

# ----------------- Demo Data Generator -----------------
def generate_mock_threat_data(num_entries=100):
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=num_entries, freq='D')
    descriptions = [f"Threat {i}: Description of threat." for i in range(1, num_entries + 1)]
    severities = np.random.choice(['Low', 'Medium', 'High', 'Critical'], num_entries)
    latitudes = np.random.uniform(low=-90.0, high=90.0, size=num_entries)
    longitudes = np.random.uniform(low=-180.0, high=180.0, size=num_entries)
    types = np.random.choice(['Malware', 'Phishing', 'Ransomware', 'DDoS'], num_entries)

    return pd.DataFrame({
        'publishedDate': dates,
        'description': descriptions,
        'severity': severities,
        'latitude': latitudes,
        'longitude': longitudes,
        'type': types
    })

# ----------------- Upload or Demo Data -----------------
uploaded_file = st.file_uploader("📂 Upload your threat data (CSV format)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Data loaded successfully from your file!")
else:
    st.info("ℹ️ No file uploaded. Using demo threat data instead.")
    df = generate_mock_threat_data()

# Normalize columns
df.columns = df.columns.str.strip().str.lower()
rename_map = {
    'date': 'publisheddate',
    'desc': 'description',
    'severity_level': 'severity',
    'lat': 'latitude',
    'lon': 'longitude',
    'threat_type': 'type'
}
df = df.rename(columns=rename_map)

# ----------------- Display Threat Data -----------------
st.subheader("Recent Threats")
st.dataframe(df)

# ----------------- Threats Over Time -----------------
if not df.empty:
    df['date'] = pd.to_datetime(df['publisheddate'])
    threats_over_time = df.groupby(df['date'].dt.to_period('M')).size().reset_index(name='count')
    threats_over_time['date'] = threats_over_time['date'].dt.strftime('%Y-%m')

    fig = px.line(threats_over_time, x='date', y='count',
                  title='Threats Over Time',
                  color_discrete_sequence=['#00bcd4'])
    st.plotly_chart(fig)

# ----------------- Search Function -----------------
search_term = st.text_input("🔍 Search for a specific threat:")
if search_term:
    filtered_data = df[df['description'].str.contains(search_term, case=False, na=False)]
    st.dataframe(filtered_data)

# ----------------- Geolocation Mapping -----------------
if 'latitude' in df.columns and 'longitude' in df.columns:
    st.subheader("🌍 Threats by Location")
    map_fig = px.scatter_geo(
        df,
        lat='latitude',
        lon='longitude',
        text='description',
        title='Threats by Geolocation',
        hover_name='description',
        color='severity',
        size_max=15
    )
    st.plotly_chart(map_fig)
else:
    st.warning("Geolocation data is not available.")

# ----------------- Alerts Section -----------------
def generate_mock_alerts(num_alerts=5):
    alerts = [
        {"date": f"2024-11-0{i+1}", "description": f"Critical vulnerability alert for Software {i+1}"}
        for i in range(num_alerts)
    ]
    return pd.DataFrame(alerts)

alerts_df = generate_mock_alerts()
if not alerts_df.empty:
    st.subheader("🚨 Recent Alerts")
    st.dataframe(alerts_df)

# ----------------- Threat Classification -----------------
if 'severity' in df.columns:
    severity_counts = df['severity'].value_counts()
    st.subheader("📊 Threat Classification by Severity")
    st.bar_chart(severity_counts)
else:
    st.warning("Severity data is not available.")

# =========================================================
# 🔮 PREDICTION MODULE (PROPHET)
# =========================================================
st.header("📈 Threat Trend Prediction")

try:
    daily_data = df.groupby(pd.to_datetime(df['publisheddate'])).size().reset_index(name='Count')
    forecast_df = daily_data.rename(columns={'publisheddate': 'ds', 'Count': 'y'})

    model = Prophet()
    model.fit(forecast_df)
    future = model.make_future_dataframe(periods=15)
    forecast = model.predict(future)

    fig_pred = px.line(forecast, x='ds', y='yhat', title='Predicted Threat Trends (Next 15 Days)')
    fig_pred.add_scatter(x=forecast_df['ds'], y=forecast_df['y'], mode='lines', name='Actual')
    st.plotly_chart(fig_pred)

    st.info("📊 This prediction helps identify potential surge periods in cyber threats.")
except Exception as e:
    st.error(f"Prediction module error: {e}")

# =========================================================
# ⚠️ ANOMALY DETECTION MODULE (ISOLATION FOREST)
# =========================================================
st.header("🚨 Anomaly Detection in Threat Data")

try:
    anomaly_data = daily_data.copy()
    anomaly_data['Date'] = pd.to_datetime(anomaly_data['publisheddate'], errors='coerce')

    clf = IsolationForest(contamination=0.1, random_state=42)
    anomaly_data['Anomaly'] = clf.fit_predict(anomaly_data[['Count']])

    anomalies = anomaly_data[anomaly_data['Anomaly'] == -1]

    fig2 = px.scatter(
        anomaly_data, x='publisheddate', y='Count',
        color=anomaly_data['Anomaly'].map({1: 'Normal', -1: 'Anomaly'}),
        title='Anomaly Detection in Threat Trends'
    )
    fig2.add_scatter(x=anomalies['publisheddate'], y=anomalies['Count'],
                     mode='markers', marker=dict(size=10, color='red'), name='Anomalies')
    st.plotly_chart(fig2)

    st.info("🔴 Red points represent unusual spikes that could indicate targeted or zero-day attacks.")
except Exception as e:
    st.error(f"Anomaly detection module error: {e}")

# ----------------- Threat Type Filter -----------------
threat_types = df['type'].unique().tolist() if 'type' in df.columns else []
selected_type = st.selectbox("Select Threat Type", options=['All'] + threat_types)

if selected_type != 'All':
    filtered_df = df[df['type'] == selected_type]
else:
    filtered_df = df

st.dataframe(filtered_df)

# ----------------- Export Data -----------------
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df_to_csv(filtered_df)
st.download_button(
    label="💾 Download filtered data as CSV",
    data=csv,
    file_name='threat_data.csv',
    mime='text/csv',
)
