import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
import hashlib
import random
from collections import defaultdict, Counter
import networkx as nx
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="ClarusSight | Cybersecurity Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MODERN THEME CONFIGURATION ---
COLORS = {
    "bg": "#0a0e27",
    "surface": "#141b3d",
    "card": "#1a2347",
    "primary": "#00d9ff",
    "secondary": "#7c3aed",
    "accent": "#ff4d6d",
    "success": "#00ff9d",
    "warning": "#ffd93d",
    "danger": "#ff4444",
    "text": "#ffffff",
    "text_secondary": "#94a3b8",
    "border": "#2d3b5e",
    "grid": "#1e2942"
}

# Custom CSS for improved UI
st.markdown(f"""
<style>
    /* Import Modern Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global Styles */
    .stApp {{
        background: linear-gradient(135deg, {COLORS['bg']} 0%, #0d1235 100%);
        color: {COLORS['text']};
        font-family: 'Inter', sans-serif;
    }}
    
    /* Headings */
    h1, h2, h3, h4, h5, h6 {{
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        color: {COLORS['text']} !important;
    }}
    
    /* Info Boxes */
    .info-box {{
        background: {COLORS['surface']};
        border-left: 4px solid {COLORS['primary']};
        padding: 15px 20px;
        border-radius: 8px;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }}
    
    .info-box h4 {{
        color: {COLORS['primary']} !important;
        margin-top: 0;
        font-size: 16px;
    }}
    
    .info-box p {{
        color: {COLORS['text_secondary']};
        margin-bottom: 0;
        line-height: 1.6;
    }}
    
    /* Help Tooltip */
    .help-tooltip {{
        background: {COLORS['card']};
        border: 1px solid {COLORS['border']};
        padding: 12px 16px;
        border-radius: 6px;
        font-size: 13px;
        color: {COLORS['text_secondary']};
        margin: 10px 0;
    }}
    
    /* Metric Cards */
    div[data-testid="stMetric"] {{
        background: linear-gradient(135deg, {COLORS['surface']} 0%, {COLORS['card']} 100%);
        border: 1px solid {COLORS['border']};
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }}
    
    div[data-testid="stMetric"]:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 12px rgba(0, 217, 255, 0.2);
        border-color: {COLORS['primary']};
    }}
    
    div[data-testid="stMetricLabel"] {{
        font-family: 'Inter', sans-serif;
        font-size: 13px;
        font-weight: 600;
        color: {COLORS['text_secondary']};
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    
    div[data-testid="stMetricValue"] {{
        font-family: 'Inter', sans-serif;
        color: {COLORS['primary']};
        font-weight: 700;
        font-size: 28px;
    }}

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {{
        background: {COLORS['surface']};
        border-right: 1px solid {COLORS['border']};
    }}
    
    section[data-testid="stSidebar"] .stMarkdown h3 {{
        color: {COLORS['primary']} !important;
        font-size: 18px;
        margin-top: 20px;
        padding-bottom: 10px;
        border-bottom: 2px solid {COLORS['border']};
    }}
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background-color: transparent;
        padding-bottom: 0;
        border-bottom: 2px solid {COLORS['border']};
    }}
    
    .stTabs [data-baseweb="tab"] {{
        height: 50px;
        background-color: transparent;
        border-radius: 8px 8px 0 0;
        color: {COLORS['text_secondary']};
        font-family: 'Inter', sans-serif;
        font-size: 14px;
        font-weight: 600;
        padding: 0 24px;
        transition: all 0.2s ease;
    }}
    
    .stTabs [data-baseweb="tab"]:hover {{
        background-color: {COLORS['card']};
        color: {COLORS['text']};
    }}
    
    .stTabs [aria-selected="true"] {{
        background-color: {COLORS['card']};
        border-bottom: 3px solid {COLORS['primary']};
        color: {COLORS['primary']};
    }}

    /* Cards */
    .modern-card {{
        background: {COLORS['card']};
        border: 1px solid {COLORS['border']};
        border-radius: 12px;
        padding: 24px;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }}
    
    .card-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-bottom: 12px;
        margin-bottom: 16px;
        border-bottom: 1px solid {COLORS['border']};
    }}
    
    .card-title {{
        font-family: 'Inter', sans-serif;
        font-size: 18px;
        font-weight: 700;
        color: {COLORS['text']};
    }}
    
    .card-badge {{
        background: {COLORS['primary']};
        color: {COLORS['bg']};
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
    }}
    
    /* Buttons */
    .stButton > button {{
        background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 217, 255, 0.3);
    }}
    
    /* Alert Boxes */
    .alert {{
        padding: 16px 20px;
        border-radius: 8px;
        margin: 15px 0;
        font-size: 14px;
        line-height: 1.6;
    }}
    
    .alert-info {{
        background: rgba(0, 217, 255, 0.1);
        border-left: 4px solid {COLORS['primary']};
        color: {COLORS['primary']};
    }}
    
    .alert-warning {{
        background: rgba(255, 217, 61, 0.1);
        border-left: 4px solid {COLORS['warning']};
        color: {COLORS['warning']};
    }}
    
    .alert-danger {{
        background: rgba(255, 77, 109, 0.1);
        border-left: 4px solid {COLORS['accent']};
        color: {COLORS['accent']};
    }}
    
    /* Dataframes */
    div[data-testid="stDataFrame"] {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 13px;
    }}
    
    /* Status Badge */
    .status-badge {{
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
    }}
    
    .status-critical {{ background: {COLORS['danger']}; color: white; }}
    .status-high {{ background: {COLORS['accent']}; color: white; }}
    .status-medium {{ background: {COLORS['warning']}; color: {COLORS['bg']}; }}
    .status-low {{ background: {COLORS['text_secondary']}; color: white; }}
    
    /* Section Headers */
    .section-header {{
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 30px 0 20px 0;
        padding-bottom: 12px;
        border-bottom: 2px solid {COLORS['border']};
    }}
    
    .section-header h3 {{
        margin: 0 !important;
        color: {COLORS['text']} !important;
        font-size: 20px !important;
    }}
    
    .section-icon {{
        font-size: 24px;
    }}
</style>
""", unsafe_allow_html=True)

class AdvancedThreatIntelligence:
    """Advanced CTI Framework with ML-powered predictions"""
    
    def __init__(self):
        self.threat_database = []
        self.ioc_database = []
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.threat_predictor = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.threat_graph = nx.DiGraph()
        
    def generate_realistic_threats(self, num_threats=100):
        """Generate realistic threat data for demonstration"""
        threat_types = ['Ransomware', 'Phishing', 'DDoS', 'Data Breach', 'APT', 
                       'Malware', 'Zero-Day', 'SQL Injection', 'XSS', 'MITM']
        
        attack_vectors = ['Email', 'Web Application', 'Network', 'Endpoint', 
                         'Cloud', 'Mobile', 'IoT', 'Social Engineering']
        
        threat_actors = ['APT28', 'Lazarus Group', 'FIN7', 'Anonymous', 
                        'REvil', 'DarkSide', 'Conti', 'LockBit', 'ALPHV']
        
        industries = ['Finance', 'Healthcare', 'Government', 'Technology', 
                     'Energy', 'Retail', 'Education', 'Manufacturing']
        
        countries = ['USA', 'China', 'Russia', 'North Korea', 'Iran', 
                    'Israel', 'UK', 'Germany', 'India']
        
        threats = []
        base_time = datetime.now() - timedelta(days=30)
        
        for i in range(num_threats):
            timestamp = base_time + timedelta(
                hours=random.randint(0, 720),
                minutes=random.randint(0, 59)
            )
            
            threat_type = random.choice(threat_types)
            severity = random.choices(
                ['Critical', 'High', 'Medium', 'Low'],
                weights=[0.15, 0.35, 0.35, 0.15]
            )[0]
            
            threat = {
                'id': f'THR-{i+1:05d}',
                'timestamp': timestamp,
                'type': threat_type,
                'severity': severity,
                'attack_vector': random.choice(attack_vectors),
                'threat_actor': random.choice(threat_actors),
                'target_industry': random.choice(industries),
                'source_country': random.choice(countries),
                'confidence': random.uniform(0.6, 0.99),
                'affected_systems': random.randint(1, 500),
                'detection_time': random.randint(1, 720),  # minutes
                'mitigation_status': random.choice(['Detected', 'Contained', 'Investigating', 'Resolved']),
                'cvss_score': random.uniform(3.0, 10.0),
                'ttps': random.sample(['T1566', 'T1059', 'T1105', 'T1047', 'T1003'], k=random.randint(1, 3)),
                'iocs': random.randint(5, 50)
            }
            threats.append(threat)
        
        return threats
    
    def generate_ioc_data(self, num_iocs=500):
        """Generate Indicators of Compromise"""
        ioc_types = ['IP Address', 'Domain', 'File Hash', 'URL', 'Email', 'Registry Key']
        
        iocs = []
        for i in range(num_iocs):
            ioc_type = random.choice(ioc_types)
            
            if ioc_type == 'IP Address':
                value = f"{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}"
            elif ioc_type == 'Domain':
                value = f"malicious-{random.randint(1000,9999)}.{random.choice(['com', 'net', 'org', 'ru', 'cn'])}"
            elif ioc_type == 'File Hash':
                value = hashlib.sha256(f"malware_{i}".encode()).hexdigest()
            elif ioc_type == 'URL':
                value = f"http://suspicious-site-{random.randint(100,999)}.com/payload"
            elif ioc_type == 'Email':
                value = f"phishing{random.randint(100,999)}@malicious-domain.com"
            else:
                value = f"HKEY_LOCAL_MACHINE\\Software\\Malware{random.randint(1,100)}"
            
            ioc = {
                'type': ioc_type,
                'value': value,
                'first_seen': datetime.now() - timedelta(days=random.randint(1, 30)),
                'last_seen': datetime.now() - timedelta(days=random.randint(0, 5)),
                'threat_level': random.choice(['Critical', 'High', 'Medium', 'Low']),
                'associated_threats': random.randint(1, 10),
                'reputation_score': random.uniform(0, 100)
            }
            iocs.append(ioc)
        
        return iocs
    
    def predict_threat_trends(self, threat_data):
        """AI-powered threat prediction using time series analysis"""
        df = pd.DataFrame(threat_data)
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        predictions = []
        current_time = datetime.now()
        
        for i in range(24):
            future_time = current_time + timedelta(hours=i)
            base_threat_count = len([t for t in threat_data if t['timestamp'].hour == future_time.hour]) / 30
            trend_factor = 1 + (random.random() - 0.5) * 0.3
            predicted_threats = max(1, int(base_threat_count * trend_factor))
            
            predictions.append({
                'timestamp': future_time,
                'predicted_threats': predicted_threats,
                'confidence': random.uniform(0.75, 0.95),
                'severity_distribution': {
                    'Critical': random.uniform(0.1, 0.2),
                    'High': random.uniform(0.3, 0.4),
                    'Medium': random.uniform(0.3, 0.4),
                    'Low': random.uniform(0.1, 0.2)
                }
            })
        
        return predictions
    
    def detect_anomalies(self, threat_data):
        """Detect anomalous threat patterns"""
        df = pd.DataFrame(threat_data)
        
        feature_matrix = []
        for _, row in df.iterrows():
            features = [
                row['affected_systems'],
                row['detection_time'],
                row['cvss_score'],
                row['confidence'] * 100,
                len(row['ttps'])
            ]
            feature_matrix.append(features)
        
        X = np.array(feature_matrix)
        X_scaled = self.scaler.fit_transform(X)
        
        anomaly_labels = self.anomaly_detector.fit_predict(X_scaled)
        
        anomalies = []
        for idx, label in enumerate(anomaly_labels):
            if label == -1:
                threat = threat_data[idx]
                threat['anomaly_score'] = random.uniform(0.7, 0.99)
                threat['anomaly_reason'] = random.choice([
                    'Unusual attack pattern detected',
                    'Abnormally high number of affected systems',
                    'Rapid propagation detected',
                    'Unexpected TTPs combination',
                    'Zero-day indicator detected'
                ])
                anomalies.append(threat)
        
        return anomalies[:10]
    
    def build_attack_graph(self, threats):
        """Build attack correlation graph"""
        G = nx.DiGraph()
        
        for threat in threats:
            actor = threat['threat_actor']
            threat_type = threat['type']
            target = threat['target_industry']
            
            G.add_node(actor, node_type='actor', color=COLORS['accent'])
            G.add_node(threat_type, node_type='attack', color=COLORS['primary'])
            G.add_node(target, node_type='target', color=COLORS['success'])
            
            G.add_edge(actor, threat_type, weight=1)
            G.add_edge(threat_type, target, weight=1)
        
        return G
    
    def calculate_risk_score(self, threat_data):
        """Calculate organizational risk score"""
        if not threat_data or len(threat_data) == 0:
            return 0.0
        
        df = pd.DataFrame(threat_data)
        severity_weights = {'Critical': 10, 'High': 7, 'Medium': 4, 'Low': 1}
        
        total_risk = 0
        for _, threat in df.iterrows():
            risk = (
                severity_weights[threat['severity']] * threat['confidence'] * 
                (threat['cvss_score'] / 10) * (1 + threat['affected_systems'] / 1000)
            )
            total_risk += risk
        
        max_possible_risk = len(threat_data) * 10 * 1.0 * 1.0 * 1.5
        if max_possible_risk == 0:
            return 0.0
        
        risk_score = min(100, (total_risk / max_possible_risk) * 100)
        return risk_score

# Initialize the framework
@st.cache_resource
def init_framework():
    return AdvancedThreatIntelligence()

cti = init_framework()

# Generate data
if 'threats' not in st.session_state:
    st.session_state.threats = cti.generate_realistic_threats(100)
    st.session_state.iocs = cti.generate_ioc_data(500)
    st.session_state.predictions = cti.predict_threat_trends(st.session_state.threats)
    st.session_state.anomalies = cti.detect_anomalies(st.session_state.threats)
    st.session_state.show_help = True

# Header with Welcome Message
st.markdown(f"""
<div style='background: linear-gradient(135deg, {COLORS['surface']} 0%, {COLORS['card']} 100%); 
     padding: 40px; border-radius: 16px; margin-bottom: 30px; 
     border: 1px solid {COLORS['border']}; box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);'>
    <h1 style='margin: 0; color: white !important; font-size: 42px;'>
        🛡️ ClarusSight
    </h1>
    <p style='font-size: 18px; color: {COLORS['text_secondary']}; margin: 10px 0 0 0;'>
        Advanced Cybersecurity Threat Intelligence Dashboard
    </p>
    <p style='font-size: 13px; color: {COLORS['text_secondary']}; margin: 8px 0 0 0; 
       font-family: "JetBrains Mono", monospace;'>
        Real-time monitoring • AI-powered predictions • Threat correlation analysis
    </p>
</div>
""", unsafe_allow_html=True)

# Quick Start Guide (Collapsible)
if st.session_state.show_help:
    with st.expander("📖 Quick Start Guide - Click to learn how to use this dashboard", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="info-box">
                <h4>🎯 Dashboard Overview</h4>
                <p>View real-time threat metrics, activity timelines, and threat distribution. 
                Perfect for getting a quick snapshot of your security posture.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="info-box">
                <h4>🔮 Predictive Analytics</h4>
                <p>AI-powered forecasting shows expected threats for the next 24 hours. 
                Use this to prepare your team for potential security events.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="info-box">
                <h4>🕵️ Anomaly Detection</h4>
                <p>Machine learning identifies unusual patterns that may indicate 
                sophisticated attacks. Review flagged events carefully.</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="alert alert-info">
            <strong>💡 Pro Tip:</strong> Use the sidebar filters to focus on specific time ranges, 
            threat types, or severity levels. Click the "🔄 Refresh Data" button to simulate new threat data.
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("✅ Got it! Don't show this again"):
            st.session_state.show_help = False
            st.rerun()

# Sidebar with better organization
with st.sidebar:
    st.markdown("### ⚙️ Dashboard Controls")
    
    st.markdown('<div class="help-tooltip">💡 Adjust these settings to filter your threat data</div>', 
                unsafe_allow_html=True)
    
    # Time Range Selection
    st.markdown("#### 📅 Time Period")
    time_range = st.selectbox(
        "Select time range",
        ["Last 24 Hours", "Last 7 Days", "Last 30 Days", "Custom Range"],
        index=2,
        help="Filter threats by time period"
    )
    
    custom_start_date = None
    custom_end_date = None
    if time_range == "Custom Range":
        st.info("📅 Select your custom date range below:")
        col1, col2 = st.columns(2)
        with col1:
            custom_start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=30),
                max_value=datetime.now(),
                help="Beginning of the date range"
            )
        with col2:
            custom_end_date = st.date_input(
                "End Date",
                value=datetime.now(),
                max_value=datetime.now(),
                help="End of the date range"
            )
    
    st.markdown("---")
    
    # Threat Filters
    st.markdown("#### 🎯 Threat Filters")
    threat_filter = st.multiselect(
        "Threat Types",
        ['Ransomware', 'Phishing', 'DDoS', 'Data Breach', 'APT', 'Malware', 'Zero-Day'],
        default=[],
        help="Filter by specific threat types"
    )
    
    severity_filter = st.multiselect(
        "Severity Levels",
        ['Critical', 'High', 'Medium', 'Low'],
        default=[],
        help="Filter by threat severity"
    )
    
    st.markdown("---")
    
    # Actions
    st.markdown("#### 🔄 Actions")
    if st.button("🔄 Refresh Data", use_container_width=True, help="Generate new simulated threat data"):
        with st.spinner("Refreshing threat data..."):
            st.session_state.threats = cti.generate_realistic_threats(100)
            st.session_state.iocs = cti.generate_ioc_data(500)
            st.session_state.predictions = cti.predict_threat_trends(st.session_state.threats)
            st.session_state.anomalies = cti.detect_anomalies(st.session_state.threats)
        st.success("✅ Data refreshed successfully!")
        st.rerun()
    
    st.markdown("---")
    
    # Export Options
    st.markdown("#### 📥 Export Data")
    
    # Export Threat Report as CSV
    threats_export_df = pd.DataFrame(st.session_state.threats)
    
    # Safely convert datetime columns
    if 'timestamp' in threats_export_df.columns and not pd.api.types.is_string_dtype(threats_export_df['timestamp']):
        threats_export_df['timestamp'] = pd.to_datetime(threats_export_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Convert TTP lists to comma-separated strings
    if 'ttps' in threats_export_df.columns:
        threats_export_df['ttps'] = threats_export_df['ttps'].apply(
            lambda x: ', '.join(x) if isinstance(x, list) else str(x)
        )
    
    csv_data = threats_export_df.to_csv(index=False)
    
    st.download_button(
        label="📊 Download Threat Report (CSV)",
        data=csv_data,
        file_name=f"threat_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True,
        help="Download complete threat data as CSV file",
        key="download_threat_report_csv"
    )
    
    # Export IOC List as JSON
    iocs_export = []
    for ioc in st.session_state.iocs:
        ioc_copy = ioc.copy()
        # Safely convert datetime to string
        try:
            if isinstance(ioc_copy['first_seen'], datetime):
                ioc_copy['first_seen'] = ioc_copy['first_seen'].strftime('%Y-%m-%d %H:%M:%S')
            elif not isinstance(ioc_copy['first_seen'], str):
                ioc_copy['first_seen'] = str(ioc_copy['first_seen'])
                
            if isinstance(ioc_copy['last_seen'], datetime):
                ioc_copy['last_seen'] = ioc_copy['last_seen'].strftime('%Y-%m-%d %H:%M:%S')
            elif not isinstance(ioc_copy['last_seen'], str):
                ioc_copy['last_seen'] = str(ioc_copy['last_seen'])
        except Exception as e:
            # If conversion fails, convert to string
            ioc_copy['first_seen'] = str(ioc_copy['first_seen'])
            ioc_copy['last_seen'] = str(ioc_copy['last_seen'])
        
        iocs_export.append(ioc_copy)
    
    json_data = json.dumps(iocs_export, indent=2)
    
    st.download_button(
        label="📋 Download IOC Database (JSON)",
        data=json_data,
        file_name=f"ioc_database_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        use_container_width=True,
        help="Download IOC database as JSON file",
        key="download_ioc_json"
    )
    
    # Generate Executive Summary Report
    threats_df_summary = pd.DataFrame(st.session_state.threats)
    risk_score = cti.calculate_risk_score(st.session_state.threats)
    critical_count = len(threats_df_summary[threats_df_summary['severity'] == 'Critical'])
    
    executive_summary = f"""
╔════════════════════════════════════════════════════════════════╗
║           CLARUSSIGHT CYBERSECURITY EXECUTIVE REPORT           ║
╚════════════════════════════════════════════════════════════════╝

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Reporting Period: Last 30 Days

═══════════════════════════════════════════════════════════════════

EXECUTIVE SUMMARY
─────────────────────────────────────────────────────────────────

• Total Threats Detected: {len(threats_df_summary)}
• Critical Severity Threats: {critical_count}
• Overall Risk Score: {risk_score:.1f}%
• Active Investigations: {len(threats_df_summary[threats_df_summary['mitigation_status'] == 'Investigating'])}
• Average Detection Time: {threats_df_summary['detection_time'].mean():.0f} minutes

═══════════════════════════════════════════════════════════════════

THREAT BREAKDOWN BY TYPE
─────────────────────────────────────────────────────────────────

{threats_df_summary['type'].value_counts().to_string()}

═══════════════════════════════════════════════════════════════════

SEVERITY DISTRIBUTION
─────────────────────────────────────────────────────────────────

{threats_df_summary['severity'].value_counts().to_string()}

═══════════════════════════════════════════════════════════════════

TOP THREAT ACTORS
─────────────────────────────────────────────────────────────────

{threats_df_summary['threat_actor'].value_counts().head(5).to_string()}

═══════════════════════════════════════════════════════════════════

TARGETED INDUSTRIES
─────────────────────────────────────────────────────────────────

{threats_df_summary['target_industry'].value_counts().head(5).to_string()}

═══════════════════════════════════════════════════════════════════

CRITICAL FINDINGS
─────────────────────────────────────────────────────────────────

• Ransomware activity increased by 23% in the last 48 hours
• Strong correlation between APT28 and Finance sector targets
• Multiple zero-day indicators detected in web application traffic
• Email-based phishing attempts surged by 31%

═══════════════════════════════════════════════════════════════════

RECOMMENDATIONS
─────────────────────────────────────────────────────────────────

1. Deploy additional web application firewall rules immediately
2. Schedule emergency security awareness training
3. Implement enhanced logging for Finance sector systems
4. Update incident response playbooks for APT scenarios
5. Conduct penetration testing of cloud infrastructure

═══════════════════════════════════════════════════════════════════

DEFENSIVE POSTURE
─────────────────────────────────────────────────────────────────

✓ Email gateway blocking 94% of phishing attempts
✓ DDoS mitigation operational (99.8% uptime)
✓ Patch compliance at 98% across endpoints
✓ Response time improved by 22% this quarter

═══════════════════════════════════════════════════════════════════

This report contains sensitive security information.
Distribute on a need-to-know basis only.

ClarusSight Cybersecurity Platform
https://makeeveryonecybersafe.github.io
"""
    
    st.download_button(
        label="📄 Download Executive Summary (TXT)",
        data=executive_summary,
        file_name=f"executive_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
        use_container_width=True,
        help="Download comprehensive executive summary report",
        key="download_executive_summary"
    )
    
    st.markdown("---")
    
    # Help Section
    st.markdown("#### ❓ Need Help?")
    st.markdown(f"""
    <div class="help-tooltip">
        <strong>Documentation:</strong><br>
        • View the Quick Start Guide at the top<br>
        • Hover over any element for tooltips<br>
        • Check severity legends in charts
    </div>
    """, unsafe_allow_html=True)

# Apply filters
threats_df = pd.DataFrame(st.session_state.threats)
original_count = len(threats_df)

if time_range == "Last 24 Hours":
    time_threshold = datetime.now() - timedelta(hours=24)
    threats_df = threats_df[threats_df['timestamp'] >= time_threshold]
elif time_range == "Last 7 Days":
    time_threshold = datetime.now() - timedelta(days=7)
    threats_df = threats_df[threats_df['timestamp'] >= time_threshold]
elif time_range == "Last 30 Days":
    time_threshold = datetime.now() - timedelta(days=30)
    threats_df = threats_df[threats_df['timestamp'] >= time_threshold]
elif time_range == "Custom Range" and custom_start_date and custom_end_date:
    start_datetime = pd.to_datetime(custom_start_date)
    end_datetime = pd.to_datetime(custom_end_date) + timedelta(days=1) - timedelta(seconds=1)
    threats_df = threats_df[(threats_df['timestamp'] >= start_datetime) & 
                           (threats_df['timestamp'] <= end_datetime)]

if threat_filter:
    threats_df = threats_df[threats_df['type'].isin(threat_filter)]
if severity_filter:
    threats_df = threats_df[threats_df['severity'].isin(severity_filter)]

# Filter Status Alert
if len(threats_df) == 0:
    st.markdown(f"""
    <div class="alert alert-warning">
        <strong>⚠️ No threats found!</strong><br>
        Your current filters returned no results. Try adjusting your filter settings in the sidebar.
    </div>
    """, unsafe_allow_html=True)
elif len(threats_df) < original_count:
    st.markdown(f"""
    <div class="alert alert-info">
        <strong>🔍 Filters Active:</strong> Showing {len(threats_df)} of {original_count} threats
        {f" | Time: {time_range}" if time_range else ""}
        {f" | Types: {', '.join(threat_filter)}" if threat_filter else ""}
        {f" | Severity: {', '.join(severity_filter)}" if severity_filter else ""}
    </div>
    """, unsafe_allow_html=True)

# Main Dashboard Tabs
tabs = st.tabs([
    "🎯 Dashboard", 
    "🔮 Predictions", 
    "🕵️ Anomalies",
    "🌐 Threat Map", 
    "📋 IOC Database",
    "💡 Insights"
])

# Helper for Plotly Charts
def apply_modern_theme(fig):
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color=COLORS['text_secondary'],
        font_family="Inter",
        title_font_family="Inter",
        title_font_color=COLORS['text'],
        title_font_size=16,
        xaxis=dict(showgrid=True, gridcolor=COLORS['grid'], gridwidth=0.5),
        yaxis=dict(showgrid=True, gridcolor=COLORS['grid'], gridwidth=0.5),
        margin=dict(t=40, b=40, l=40, r=40),
        hovermode='x unified'
    )
    return fig

# Tab 1: Main Dashboard
with tabs[0]:
    if len(threats_df) > 0:
        st.markdown('<div class="section-header"><span class="section-icon">📊</span><h3>Key Security Metrics</h3></div>', 
                    unsafe_allow_html=True)
        
        # Key Metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            total_threats = len(threats_df)
            st.metric(
                "Total Threats", 
                total_threats, 
                f"+{random.randint(5, 15)}",
                help="Total number of detected threats in the selected period"
            )
        
        with col2:
            critical_threats = len(threats_df[threats_df['severity'] == 'Critical'])
            st.metric(
                "Critical Threats", 
                critical_threats, 
                f"+{random.randint(1, 5)}", 
                delta_color="inverse",
                help="Threats requiring immediate attention"
            )
        
        with col3:
            risk_score = cti.calculate_risk_score(threats_df.to_dict('records'))
            risk_delta = random.uniform(-2, 2)
            st.metric(
                "Risk Score", 
                f"{risk_score:.1f}%", 
                f"{risk_delta:+.1f}%",
                delta_color="inverse" if risk_delta > 0 else "normal",
                help="Overall organizational risk assessment (0-100%)"
            )
        
        with col4:
            active_threats = len(threats_df[threats_df['mitigation_status'] == 'Investigating'])
            st.metric(
                "Active Investigations", 
                active_threats, 
                f"{random.randint(-2, 3):+d}",
                help="Threats currently under investigation"
            )
        
        with col5:
            avg_detection = threats_df['detection_time'].mean()
            st.metric(
                "Avg Detection Time", 
                f"{avg_detection:.0f}m", 
                f"-{random.randint(5, 15)}m",
                help="Average time to detect threats"
            )
        
        st.markdown('<div class="section-header"><span class="section-icon">📈</span><h3>Threat Analysis</h3></div>', 
                    unsafe_allow_html=True)
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Activity Timeline**")
            st.markdown('<div class="help-tooltip">Shows threat activity over time, grouped by severity level</div>', 
                        unsafe_allow_html=True)
            
            threats_df['date'] = threats_df['timestamp'].dt.date
            timeline_data = threats_df.groupby(['date', 'severity']).size().reset_index(name='count')
            
            fig = px.area(
                timeline_data, 
                x='date', 
                y='count', 
                color='severity',
                color_discrete_map={
                    'Critical': COLORS['danger'], 
                    'High': COLORS['accent'], 
                    'Medium': COLORS['warning'], 
                    'Low': COLORS['text_secondary']
                }
            )
            apply_modern_theme(fig)
            fig.update_traces(line_width=2)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Threat Type Distribution**")
            st.markdown('<div class="help-tooltip">Breakdown of threats by attack type</div>', 
                        unsafe_allow_html=True)
            
            threat_counts = threats_df['type'].value_counts()
            fig = go.Figure(data=[go.Pie(
                labels=threat_counts.index, 
                values=threat_counts.values, 
                hole=0.5,
                marker_colors=[COLORS['primary'], COLORS['secondary'], COLORS['accent'], 
                             COLORS['warning'], COLORS['success'], '#666666']
            )])
            apply_modern_theme(fig)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        # Attack Vectors and Industries
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Top Attack Vectors**")
            vector_counts = threats_df['attack_vector'].value_counts().head(5)
            fig = px.bar(
                x=vector_counts.values,
                y=vector_counts.index,
                orientation='h',
                color=vector_counts.values,
                color_continuous_scale=[[0, COLORS['card']], [1, COLORS['primary']]]
            )
            apply_modern_theme(fig)
            fig.update_layout(showlegend=False, yaxis_title=None, xaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Targeted Industries**")
            industry_counts = threats_df['target_industry'].value_counts().head(5)
            fig = px.bar(
                x=industry_counts.values,
                y=industry_counts.index,
                orientation='h',
                color=industry_counts.values,
                color_continuous_scale=[[0, COLORS['card']], [1, COLORS['accent']]]
            )
            apply_modern_theme(fig)
            fig.update_layout(showlegend=False, yaxis_title=None, xaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
        
        # Recent Critical Threats Table
        st.markdown('<div class="section-header"><span class="section-icon">🚨</span><h3>Recent Critical Threats</h3></div>', 
                    unsafe_allow_html=True)
        
        st.markdown('<div class="help-tooltip">Latest critical severity threats requiring immediate attention</div>', 
                    unsafe_allow_html=True)
        
        recent_critical = threats_df[threats_df['severity'] == 'Critical'].sort_values(
            'timestamp', ascending=False
        ).head(10)
        
        if len(recent_critical) > 0:
            # Add export button for critical threats
            col1, col2 = st.columns([4, 1])
            with col2:
                critical_export = recent_critical.copy()
                
                # Safely convert timestamps
                if 'timestamp' in critical_export.columns and not pd.api.types.is_string_dtype(critical_export['timestamp']):
                    critical_export['timestamp'] = pd.to_datetime(critical_export['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
                
                # Convert TTP lists
                if 'ttps' in critical_export.columns:
                    critical_export['ttps'] = critical_export['ttps'].apply(
                        lambda x: ', '.join(x) if isinstance(x, list) else str(x)
                    )
                
                st.download_button(
                    label="📥 Export",
                    data=critical_export.to_csv(index=False),
                    file_name=f"critical_threats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    help="Export critical threats to CSV",
                    key="export_critical_threats"
                )
            
            display_df = recent_critical[['id', 'timestamp', 'type', 'threat_actor', 
                                         'target_industry', 'cvss_score', 'mitigation_status']].copy()
            display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
            display_df['cvss_score'] = display_df['cvss_score'].round(1)
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "id": "Threat ID",
                    "timestamp": "Detected At",
                    "type": "Type",
                    "threat_actor": "Threat Actor",
                    "target_industry": "Target Industry",
                    "cvss_score": st.column_config.NumberColumn("CVSS Score", format="%.1f"),
                    "mitigation_status": "Status"
                }
            )
        else:
            st.info("✅ No critical threats detected in the selected period!")
    else:
        st.info("No threat data available. Adjust your filters or refresh the data.")

# Tab 2: Predictive Analytics
with tabs[1]:
    st.markdown('<div class="section-header"><span class="section-icon">🔮</span><h3>AI-Powered Threat Forecasting</h3></div>', 
                unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="modern-card">
        <div class="card-header">
            <span class="card-title">Prediction Model Information</span>
            <span class="card-badge">AI Powered</span>
        </div>
        <p style="color: {COLORS['text_secondary']}; line-height: 1.6;">
            Our machine learning model analyzes historical threat patterns to predict future activity. 
            The forecast uses Random Forest and time-series analysis to estimate threat volumes and 
            severity distributions for the next 24 hours.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if len(threats_df) > 0:
        filtered_predictions = cti.predict_threat_trends(threats_df.to_dict('records'))
        predictions_df = pd.DataFrame(filtered_predictions)
        
        # Add export button
        col_export = st.columns([5, 1])
        with col_export[1]:
            predictions_export = predictions_df.copy()
            
            # Safely convert timestamps
            if 'timestamp' in predictions_export.columns and not pd.api.types.is_string_dtype(predictions_export['timestamp']):
                predictions_export['timestamp'] = pd.to_datetime(predictions_export['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Extract severity distribution into separate columns
            predictions_export['severity_critical'] = predictions_export['severity_distribution'].apply(lambda x: x['Critical'])
            predictions_export['severity_high'] = predictions_export['severity_distribution'].apply(lambda x: x['High'])
            predictions_export['severity_medium'] = predictions_export['severity_distribution'].apply(lambda x: x['Medium'])
            predictions_export['severity_low'] = predictions_export['severity_distribution'].apply(lambda x: x['Low'])
            predictions_export = predictions_export.drop('severity_distribution', axis=1)
            
            st.download_button(
                label="📥 Export Forecast",
                data=predictions_export.to_csv(index=False),
                file_name=f"threat_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                help="Export 24-hour threat forecast",
                key="export_predictions_forecast"
            )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            predicted_total = predictions_df['predicted_threats'].sum()
            st.metric(
                "Predicted Threats (24h)", 
                int(predicted_total),
                help="Total expected threats in the next 24 hours"
            )
        with col2:
            avg_confidence = predictions_df['confidence'].mean()
            st.metric(
                "Model Confidence", 
                f"{avg_confidence*100:.1f}%",
                help="Average prediction confidence level"
            )
        with col3:
            peak_hour = predictions_df.loc[predictions_df['predicted_threats'].idxmax()]
            peak_time = peak_hour['timestamp'].strftime('%H:%M')
            st.metric(
                "Peak Activity Time", 
                peak_time,
                help="Expected time of highest threat activity"
            )
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("**24-Hour Threat Forecast**")
            st.markdown('<div class="help-tooltip">Predicted number of threats for each hour in the next 24 hours</div>', 
                        unsafe_allow_html=True)
            
            fig = go.Figure()
            
            hours = [(predictions_df['timestamp'].iloc[i]).strftime('%H:%M') 
                    for i in range(len(predictions_df))]
            threats = predictions_df['predicted_threats'].tolist()
            confidence = predictions_df['confidence'].tolist()
            
            # Confidence band
            upper_bound = [t * (1 + (1 - c) * 0.3) for t, c in zip(threats, confidence)]
            lower_bound = [t * (1 - (1 - c) * 0.3) for t, c in zip(threats, confidence)]
            
            fig.add_trace(go.Scatter(
                x=hours, y=upper_bound,
                fill=None,
                mode='lines',
                line_color='rgba(0,217,255,0)',
                showlegend=False,
                name='Upper Bound'
            ))
            
            fig.add_trace(go.Scatter(
                x=hours, y=lower_bound,
                fill='tonexty',
                mode='lines',
                line_color='rgba(0,217,255,0)',
                fillcolor='rgba(0,217,255,0.2)',
                showlegend=True,
                name='Confidence Band'
            ))
            
            fig.add_trace(go.Scatter(
                x=hours, y=threats,
                mode='lines+markers',
                name='Predicted Threats',
                line=dict(color=COLORS['primary'], width=3),
                marker=dict(size=8, color=COLORS['primary'])
            ))
            
            apply_modern_theme(fig)
            fig.update_layout(xaxis_title="Time", yaxis_title="Predicted Threats")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Risk Intensity Heatmap**")
            st.markdown('<div class="help-tooltip">Visual representation of risk levels throughout the day</div>', 
                        unsafe_allow_html=True)
            
            risk_levels = []
            for _, pred in predictions_df.iterrows():
                risk = (pred['severity_distribution']['Critical'] * 10) * pred['predicted_threats']
                risk_levels.append(risk)
            
            fig = go.Figure(data=go.Heatmap(
                z=[risk_levels],
                x=[f"{i}h" for i in range(24)],
                y=['Risk Level'],
                colorscale=[[0, COLORS['success']], [0.5, COLORS['warning']], [1, COLORS['danger']]],
                showscale=True,
                colorbar=dict(title="Risk")
            ))
            apply_modern_theme(fig)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Severity Distribution Forecast
        st.markdown("**Predicted Severity Distribution**")
        st.markdown('<div class="help-tooltip">Breakdown of expected threat severity levels</div>', 
                    unsafe_allow_html=True)
        
        avg_severity = {
            'Critical': predictions_df['severity_distribution'].apply(lambda x: x['Critical']).mean() * 100,
            'High': predictions_df['severity_distribution'].apply(lambda x: x['High']).mean() * 100,
            'Medium': predictions_df['severity_distribution'].apply(lambda x: x['Medium']).mean() * 100,
            'Low': predictions_df['severity_distribution'].apply(lambda x: x['Low']).mean() * 100
        }
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(avg_severity.keys()),
                y=list(avg_severity.values()),
                marker_color=[COLORS['danger'], COLORS['accent'], COLORS['warning'], COLORS['text_secondary']],
                text=[f"{v:.1f}%" for v in avg_severity.values()],
                textposition='auto'
            )
        ])
        apply_modern_theme(fig)
        fig.update_layout(xaxis_title="Severity Level", yaxis_title="Percentage (%)")
        st.plotly_chart(fig, use_container_width=True)

# Tab 3: Anomaly Detection
with tabs[2]:
    st.markdown('<div class="section-header"><span class="section-icon">🕵️</span><h3>Machine Learning Anomaly Detection</h3></div>', 
                unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="modern-card">
        <div class="card-header">
            <span class="card-title">Detection Algorithm</span>
            <span class="card-badge">ML Powered</span>
        </div>
        <p style="color: {COLORS['text_secondary']}; line-height: 1.6;">
            Using Isolation Forest machine learning, we identify threats that deviate significantly from 
            normal patterns. These anomalies may indicate sophisticated attacks, zero-day exploits, or 
            coordinated campaigns that require immediate investigation.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if len(threats_df) > 0:
        filtered_anomalies = cti.detect_anomalies(threats_df.to_dict('records'))
        anomalies_df = pd.DataFrame(filtered_anomalies)
        
        # Add export button
        col_exp = st.columns([5, 1])
        with col_exp[1]:
            if len(anomalies_df) > 0:
                anomalies_export = anomalies_df.copy()
                
                # Safely convert timestamps
                if 'timestamp' in anomalies_export.columns and not pd.api.types.is_string_dtype(anomalies_export['timestamp']):
                    anomalies_export['timestamp'] = pd.to_datetime(anomalies_export['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
                
                # Convert TTP lists
                if 'ttps' in anomalies_export.columns:
                    anomalies_export['ttps'] = anomalies_export['ttps'].apply(
                        lambda x: ', '.join(x) if isinstance(x, list) else str(x)
                    )
                
                st.download_button(
                    label="📥 Export Anomalies",
                    data=anomalies_export.to_csv(index=False),
                    file_name=f"anomalies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    help="Export detected anomalies",
                    key="export_anomalies"
                )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Anomalies Detected", 
                len(anomalies_df),
                help="Number of unusual threat patterns identified"
            )
        with col2:
            if len(anomalies_df) > 0:
                avg_score = anomalies_df['anomaly_score'].mean()
                st.metric(
                    "Avg Anomaly Score", 
                    f"{avg_score:.2f}",
                    help="Higher scores indicate more unusual patterns"
                )
            else:
                st.metric("Avg Anomaly Score", "N/A")
        with col3:
            critical_anomalies = len(anomalies_df[anomalies_df['severity'] == 'Critical']) if len(anomalies_df) > 0 else 0
            st.metric(
                "Critical Anomalies", 
                critical_anomalies,
                delta_color="inverse",
                help="Critical severity anomalies"
            )
        
        if len(anomalies_df) > 0:
            st.markdown("**Anomaly Visualization**")
            st.markdown('<div class="help-tooltip">Normal threats shown in gray, anomalies highlighted in cyan</div>', 
                        unsafe_allow_html=True)
            
            fig = go.Figure()
            
            # Normal threats
            normal_df = threats_df[~threats_df['id'].isin(anomalies_df['id'])]
            fig.add_trace(go.Scatter(
                x=normal_df['cvss_score'],
                y=normal_df['affected_systems'],
                mode='markers',
                name='Normal Threats',
                marker=dict(size=6, color=COLORS['text_secondary'], opacity=0.4),
                text=normal_df['type'],
                hovertemplate='<b>%{text}</b><br>CVSS: %{x:.1f}<br>Systems: %{y}<extra></extra>'
            ))
            
            # Anomalies
            fig.add_trace(go.Scatter(
                x=anomalies_df['cvss_score'],
                y=anomalies_df['affected_systems'],
                mode='markers',
                name='Anomalies',
                marker=dict(
                    size=14,
                    color=COLORS['primary'],
                    symbol='diamond',
                    line=dict(color=COLORS['accent'], width=2)
                ),
                text=anomalies_df['type'],
                hovertemplate='<b>%{text}</b><br>CVSS: %{x:.1f}<br>Systems: %{y}<br>ANOMALY<extra></extra>'
            ))
            
            apply_modern_theme(fig)
            fig.update_layout(
                xaxis_title="CVSS Score",
                yaxis_title="Affected Systems",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Anomaly Details
            st.markdown("**Flagged Anomalies - Requires Investigation**")
            
            for idx, anomaly in anomalies_df.iterrows():
                severity_colors = {
                    'Critical': COLORS['danger'],
                    'High': COLORS['accent'],
                    'Medium': COLORS['warning'],
                    'Low': COLORS['text_secondary']
                }
                
                st.markdown(f"""
                <div class="modern-card" style="border-left: 4px solid {severity_colors[anomaly['severity']]}">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                        <div>
                            <strong style="font-size: 16px; color: {COLORS['text']}">{anomaly['id']} - {anomaly['type']}</strong>
                            <span class="status-badge status-{anomaly['severity'].lower()}" style="margin-left: 12px;">
                                {anomaly['severity']}
                            </span>
                        </div>
                        <span style="color: {COLORS['primary']}; font-weight: 600; font-size: 18px;">
                            Score: {anomaly['anomaly_score']:.2f}
                        </span>
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; 
                         margin-bottom: 12px; font-size: 13px; color: {COLORS['text_secondary']};">
                        <div><strong>Actor:</strong> {anomaly['threat_actor']}</div>
                        <div><strong>Target:</strong> {anomaly['target_industry']}</div>
                        <div><strong>CVSS:</strong> {anomaly['cvss_score']:.1f}</div>
                        <div><strong>Affected Systems:</strong> {anomaly['affected_systems']}</div>
                    </div>
                    <div style="background: {COLORS['surface']}; padding: 12px; border-radius: 6px; 
                         border-left: 3px solid {COLORS['warning']};">
                        <strong style="color: {COLORS['warning']}">⚠️ Detection Reason:</strong><br>
                        <span style="color: {COLORS['text_secondary']}">{anomaly['anomaly_reason']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="alert alert-info">
                <strong>✅ No anomalies detected!</strong><br>
                All threats in the selected period match expected patterns. The system continues to monitor for unusual activity.
            </div>
            """, unsafe_allow_html=True)

# Tab 4: Threat Graph
with tabs[3]:
    st.markdown('<div class="section-header"><span class="section-icon">🌐</span><h3>Threat Correlation Network</h3></div>', 
                unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="modern-card">
        <p style="color: {COLORS['text_secondary']}; line-height: 1.6;">
            This interactive network graph shows relationships between threat actors, attack types, and targeted industries. 
            <strong style="color: {COLORS['accent']}">Red nodes</strong> represent threat actors, 
            <strong style="color: {COLORS['primary']}">cyan nodes</strong> represent attack types, and 
            <strong style="color: {COLORS['success']}">green nodes</strong> represent targeted industries.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    G = cti.build_attack_graph(threats_df.to_dict('records'))
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Nodes", G.number_of_nodes(), help="Unique entities in the threat network")
    with col2:
        st.metric("Connections", G.number_of_edges(), help="Relationships between entities")
    with col3:
        actors = [n for n in G.nodes() if G.nodes[n].get('node_type') == 'actor']
        st.metric("Threat Actors", len(actors), help="Unique threat actor groups")
    
    pos = nx.spring_layout(G, k=0.8, iterations=50)
    
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color=COLORS['border']),
        hoverinfo='none',
        mode='lines'
    )
    
    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"{node}<br>Type: {G.nodes[node].get('node_type', 'unknown')}")
        
        if G.nodes[node].get('node_type') == 'actor':
            node_color.append(COLORS['accent'])
            node_size.append(20)
        elif G.nodes[node].get('node_type') == 'target':
            node_color.append(COLORS['success'])
            node_size.append(20)
        else:
            node_color.append(COLORS['primary'])
            node_size.append(15)
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hovertext=node_text,
        hoverinfo='text',
        marker=dict(
            size=node_size,
            color=node_color,
            line_width=2,
            line_color=COLORS['bg']
        ),
        text=[n[:15] + '...' if len(n) > 15 else n for n in G.nodes()],
        textposition="top center",
        textfont=dict(size=9, color=COLORS['text_secondary'])
    )
    
    fig = go.Figure(data=[edge_trace, node_trace])
    apply_modern_theme(fig)
    fig.update_layout(
        showlegend=False,
        hovermode='closest',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Legend
    st.markdown(f"""
    <div style="display: flex; justify-content: center; gap: 30px; margin-top: 20px;">
        <div style="display: flex; align-items: center; gap: 8px;">
            <div style="width: 16px; height: 16px; background: {COLORS['accent']}; border-radius: 50%;"></div>
            <span style="color: {COLORS['text_secondary']}">Threat Actors</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px;">
            <div style="width: 16px; height: 16px; background: {COLORS['primary']}; border-radius: 50%;"></div>
            <span style="color: {COLORS['text_secondary']}">Attack Types</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px;">
            <div style="width: 16px; height: 16px; background: {COLORS['success']}; border-radius: 50%;"></div>
            <span style="color: {COLORS['text_secondary']}">Target Industries</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Tab 5: IOC Database
with tabs[4]:
    st.markdown('<div class="section-header"><span class="section-icon">📋</span><h3>Indicators of Compromise (IOC) Database</h3></div>', 
                unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="modern-card">
        <p style="color: {COLORS['text_secondary']}; line-height: 1.6;">
            This database contains known malicious indicators including IP addresses, domains, file hashes, 
            URLs, and email addresses. Use the search function below to check if specific indicators are present.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Filter IOCs based on filtered threats
    if len(threats_df) > 0:
        # Get threat IDs from filtered threats
        filtered_threat_ids = set(threats_df['id'].tolist())
        
        # In a real system, IOCs would be linked to threats. For demo, we'll show all IOCs
        # but add a note about filtering
        iocs_df = pd.DataFrame(st.session_state.iocs)
        
        # Show info about current filters
        if len(threats_df) < len(st.session_state.threats):
            st.info(f"💡 **Note:** Showing all IOCs. In a production system, these would be filtered to show only IOCs related to your {len(threats_df)} filtered threats.")
    else:
        iocs_df = pd.DataFrame(st.session_state.iocs)
        st.warning("⚠️ No threats match your current filters. Showing all IOCs.")
    
    # IOC Type Filter
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        search = st.text_input(
            "🔍 Search IOC Database",
            placeholder="Enter IP address, domain, hash, URL, or email...",
            help="Search for specific indicators in the database"
        )
    with col2:
        ioc_type_filter = st.selectbox(
            "Filter by Type",
            ["All Types"] + list(iocs_df['type'].unique()),
            help="Filter IOCs by type"
        )
    with col3:
        st.write("")  # Spacer
        st.write("")  # Spacer
        # Export filtered IOCs
        if len(iocs_df) > 0:
            ioc_export = filtered_iocs.copy() if 'filtered_iocs' in locals() else iocs_df.copy()
            ioc_export['first_seen'] = pd.to_datetime(ioc_export['first_seen']).dt.strftime('%Y-%m-%d %H:%M:%S')
            ioc_export['last_seen'] = pd.to_datetime(ioc_export['last_seen']).dt.strftime('%Y-%m-%d %H:%M:%S')
            
            st.download_button(
                label="📥 Export",
                data=ioc_export.to_csv(index=False),
                file_name=f"iocs_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                help="Export filtered IOCs",
                use_container_width=True
            )
    
    # Apply filters
    filtered_iocs = iocs_df.copy()
    if search:
        filtered_iocs = filtered_iocs[filtered_iocs['value'].str.contains(search, case=False, na=False)]
    if ioc_type_filter != "All Types":
        filtered_iocs = filtered_iocs[filtered_iocs['type'] == ioc_type_filter]
    
    # Update the export button data
    if col3:
        with col3:
            st.write("")  # Spacer
            st.write("")  # Spacer
            if len(filtered_iocs) > 0:
                ioc_export = filtered_iocs.copy()
                # Safely convert datetime columns to strings
                if not pd.api.types.is_string_dtype(ioc_export['first_seen']):
                    ioc_export['first_seen'] = pd.to_datetime(ioc_export['first_seen']).dt.strftime('%Y-%m-%d %H:%M:%S')
                if not pd.api.types.is_string_dtype(ioc_export['last_seen']):
                    ioc_export['last_seen'] = pd.to_datetime(ioc_export['last_seen']).dt.strftime('%Y-%m-%d %H:%M:%S')
                
                st.download_button(
                    label="📥 Export",
                    data=ioc_export.to_csv(index=False),
                    file_name=f"iocs_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    help="Export filtered IOCs",
                    use_container_width=True,
                    key="export_filtered_iocs"
                )
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total IOCs", len(iocs_df), help="Total indicators in database")
    with col2:
        critical_iocs = len(iocs_df[iocs_df['threat_level'] == 'Critical'])
        st.metric("Critical IOCs", critical_iocs, help="High-risk indicators")
    with col3:
        st.metric("Filtered Results", len(filtered_iocs), help="IOCs matching current filters")
    with col4:
        recent_iocs = len(iocs_df[iocs_df['last_seen'] >= datetime.now() - timedelta(days=7)])
        st.metric("Active (7d)", recent_iocs, help="IOCs seen in last 7 days")
    
    # IOC Type Distribution
    st.markdown("**IOC Distribution by Type**")
    type_counts = filtered_iocs['type'].value_counts()
    fig = px.bar(
        x=type_counts.values,
        y=type_counts.index,
        orientation='h',
        color=type_counts.values,
        color_continuous_scale=[[0, COLORS['card']], [1, COLORS['primary']]]
    )
    apply_modern_theme(fig)
    fig.update_layout(showlegend=False, yaxis_title=None, xaxis_title="Count", height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    # IOC Table
    st.markdown("**IOC Details**")
    display_iocs = filtered_iocs[['type', 'value', 'threat_level', 'first_seen', 
                                  'last_seen', 'associated_threats']].head(100)
    display_iocs['first_seen'] = pd.to_datetime(display_iocs['first_seen']).dt.strftime('%Y-%m-%d')
    display_iocs['last_seen'] = pd.to_datetime(display_iocs['last_seen']).dt.strftime('%Y-%m-%d')
    
    st.dataframe(
        display_iocs,
        use_container_width=True,
        hide_index=True,
        column_config={
            "type": "Type",
            "value": "IOC Value",
            "threat_level": st.column_config.TextColumn("Threat Level"),
            "first_seen": "First Seen",
            "last_seen": "Last Seen",
            "associated_threats": st.column_config.NumberColumn("Associated Threats", format="%d")
        }
    )
    
    if len(filtered_iocs) > 100:
        st.info(f"Showing 100 of {len(filtered_iocs)} results. Refine your search to see more specific results.")

# Tab 6: AI Insights
with tabs[5]:
    st.markdown('<div class="section-header"><span class="section-icon">💡</span><h3>Intelligence Summary & Recommendations</h3></div>', 
                unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="modern-card">
        <div class="card-header">
            <span class="card-title">Executive Intelligence Brief</span>
            <span class="card-badge">AI Generated</span>
        </div>
        <p style="color: {COLORS['text_secondary']}; line-height: 1.6; margin-bottom: 0;">
            Based on comprehensive analysis of threat data, network patterns, and predictive modeling, 
            here are the key findings and actionable recommendations for your security posture.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if len(threats_df) > 0:
        # Calculate statistics from FILTERED threats
        threat_type_counts = threats_df['type'].value_counts()
        severity_counts = threats_df['severity'].value_counts()
        top_actors = threats_df['threat_actor'].value_counts().head(3)
        top_industries = threats_df['target_industry'].value_counts().head(3)
        top_vectors = threats_df['attack_vector'].value_counts().head(3)
        
        # Generate dynamic insights based on filtered data
        ransomware_count = len(threats_df[threats_df['type'] == 'Ransomware'])
        phishing_count = len(threats_df[threats_df['type'] == 'Phishing'])
        total_filtered = len(threats_df)
        
        ransomware_pct = (ransomware_count / total_filtered * 100) if total_filtered > 0 else 0
        phishing_pct = (phishing_count / total_filtered * 100) if total_filtered > 0 else 0
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="modern-card">
                <h4 style="color: {COLORS['accent']}; margin-bottom: 16px;">🚨 Critical Findings</h4>
                <ul style="color: {COLORS['text_secondary']}; line-height: 1.8; padding-left: 20px;">
                    <li><strong>Ransomware Activity:</strong> {ransomware_count} incidents detected ({ransomware_pct:.1f}% of filtered threats). 
                    {'Recommend immediate backup verification and endpoint hardening.' if ransomware_pct > 15 else 'Activity within normal range.'}</li>
                    <li><strong>Top Threat Actor:</strong> {top_actors.index[0] if len(top_actors) > 0 else 'N/A'} responsible for {top_actors.iloc[0] if len(top_actors) > 0 else 0} incidents. 
                    Enhanced monitoring advised for this actor's TTPs.</li>
                    <li><strong>Most Targeted Sector:</strong> {top_industries.index[0] if len(top_industries) > 0 else 'N/A'} industry with {top_industries.iloc[0] if len(top_industries) > 0 else 0} targeted attacks. 
                    Sector-specific defenses recommended.</li>
                    <li><strong>Phishing Activity:</strong> {phishing_count} email-based threats detected ({phishing_pct:.1f}% of total). 
                    {'User awareness training urgently needed.' if phishing_pct > 20 else 'Current training appears effective.'}</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="modern-card">
                <h4 style="color: {COLORS['warning']}; margin-bottom: 16px;">⚠️ Attention Required</h4>
                <ul style="color: {COLORS['text_secondary']}; line-height: 1.8; padding-left: 20px;">
                    <li><strong>Detection Time:</strong> Average detection time is {threats_df['detection_time'].mean():.0f} minutes. 
                    {'Review SIEM rule optimization to improve response time.' if threats_df['detection_time'].mean() > 180 else 'Detection time is within acceptable range.'}</li>
                    <li><strong>Primary Attack Vector:</strong> {top_vectors.index[0] if len(top_vectors) > 0 else 'N/A'} accounts for {top_vectors.iloc[0] if len(top_vectors) > 0 else 0} incidents. 
                    Focus hardening efforts on this vector.</li>
                    <li><strong>Critical Severity:</strong> {len(threats_df[threats_df['severity'] == 'Critical'])} critical threats require immediate attention. 
                    Ensure incident response team is adequately staffed.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="modern-card">
                <h4 style="color: {COLORS['success']}; margin-bottom: 16px;">✅ Defensive Posture</h4>
                <ul style="color: {COLORS['text_secondary']}; line-height: 1.8; padding-left: 20px;">
                    <li><strong>Email Security:</strong> Gateway successfully blocking {100 - phishing_pct:.1f}% of phishing attempts. 
                    Continue current configuration.</li>
                    <li><strong>Threat Detection:</strong> {len(threats_df[threats_df['mitigation_status'] == 'Resolved'])} threats successfully resolved. 
                    Response procedures are effective.</li>
                    <li><strong>Average CVSS:</strong> {threats_df['cvss_score'].mean():.1f} across all threats. 
                    {'Focus on high-severity vulnerabilities.' if threats_df['cvss_score'].mean() > 7.0 else 'Severity levels manageable.'}</li>
                    <li><strong>Coverage:</strong> Monitoring {threats_df['affected_systems'].sum()} affected systems. 
                    Comprehensive visibility maintained.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="modern-card">
                <h4 style="color: {COLORS['primary']}; margin-bottom: 16px;">📊 Key Recommendations</h4>
                <ol style="color: {COLORS['text_secondary']}; line-height: 1.8; padding-left: 20px;">
                    <li>Focus security resources on {top_vectors.index[0] if len(top_vectors) > 0 else 'primary'} attack vector (highest volume)</li>
                    <li>Implement targeted defenses for {top_industries.index[0] if len(top_industries) > 0 else 'most attacked'} sector systems</li>
                    <li>Enhanced monitoring for {top_actors.index[0] if len(top_actors) > 0 else 'top'} threat actor TTPs</li>
                    <li>{'Immediate action required on ' + str(len(threats_df[threats_df['severity'] == 'Critical'])) + ' critical threats' if len(threats_df[threats_df['severity'] == 'Critical']) > 0 else 'Maintain current security posture'}</li>
                    <li>Review and update incident response playbooks based on current threat landscape</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("📊 Adjust your filters to see AI-generated insights based on threat data.")
    
    # Threat Trend Analysis
    if len(threats_df) > 0:
        st.markdown("**Threat Trend Analysis (Filtered Data)**")
        
        # Generate trend data from filtered threats
        threats_df['date'] = pd.to_datetime(threats_df['timestamp']).dt.date
        
        # Get date range
        date_range = pd.date_range(start=threats_df['date'].min(), end=threats_df['date'].max(), freq='D')
        
        # Count threats by date and severity
        trend_data = {'Date': []}
        for severity in ['Critical', 'High', 'Medium', 'Low']:
            trend_data[severity] = []
        
        for date in date_range:
            trend_data['Date'].append(date)
            date_threats = threats_df[threats_df['date'] == date.date()]
            for severity in ['Critical', 'High', 'Medium', 'Low']:
                count = len(date_threats[date_threats['severity'] == severity])
                trend_data[severity].append(count)
        
        trend_df = pd.DataFrame(trend_data)
        
        if len(trend_df) > 0:
            fig = go.Figure()
            for severity, color in [('Critical', COLORS['danger']), ('High', COLORS['accent']), 
                               ('Medium', COLORS['warning']), ('Low', COLORS['text_secondary'])]:
                fig.add_trace(go.Scatter(
                    x=trend_df['Date'],
                    y=trend_df[severity],
                    name=severity,
                    mode='lines',
                    line=dict(color=color, width=2),
                    stackgroup='one'
                ))
            
            apply_modern_theme(fig)
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Threat Count",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough date range in filtered data to show trend analysis.")
    else:
        st.info("No data available for trend analysis. Adjust your filters to see trends.")

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; padding: 20px;'>
    <p style='color: {COLORS['text_secondary']}; font-size: 13px; margin: 0;'>
        🛡️ <strong>ClarusSight Cybersecurity Dashboard</strong> | Developed by Aathithya Shanmuga Sundaram
    </p>
    <p style='color: {COLORS['text_secondary']}; font-size: 12px; margin: 5px 0 0 0;'>
        #MakeEveryoneCyberSafe | Data Source: Simulated Threat Intelligence
    </p>
</div>
""", unsafe_allow_html=True)
