import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(
    page_title="Academic Behaviour Intelligence Engine",
    page_icon="🎓",
    layout="wide"
)

# ==========================================================
# CUSTOM PREMIUM UI
# ==========================================================
st.markdown("""
<style>
.title {
    font-size: 44px;
    font-weight: 800;
    color: #0F172A;
}
.section {
    background: linear-gradient(90deg, #7C3AED, #06B6D4);
    padding: 12px;
    border-radius: 10px;
    color: white;
    font-weight: 600;
    margin-top: 20px;
}
.card {
    background-color: #F8FAFC;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 6px 15px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# LOAD DATA
# ==========================================================
@st.cache_data
def load_data():
    return pd.read_csv("student_burnout_dataset.csv")

df = load_data()

# ==========================================================
# FEATURE ENGINEERING
# ==========================================================
def engineer(df):
    df["stress_index"] = 0.6*(df["avg_assignment_delay_days"]/10) + 0.4*((100-df["attendance_percent"])/100)
    max_trend = df["login_trend_slope"].abs().max() or 1
    df["engagement_shift"] = df["login_trend_slope"].abs()/max_trend
    df["emotion_index"] = np.maximum(0, -df["sentiment_score"])
    df["risk_engine"] = (
        0.25*(25-df["lms_logins_weekly"])/25 +
        0.20*df["stress_index"] +
        0.20*df["engagement_shift"] +
        0.20*df["emotion_index"] +
        0.15*df["study_session_variability"]
    )
    df["risk_engine"] = 100*df["risk_engine"]
    return df

df = engineer(df)

low_cut = df["risk_engine"].quantile(0.4)
high_cut = df["risk_engine"].quantile(0.75)

# ==========================================================
# HEADER
# ==========================================================
st.markdown('<div class="title">🎓 Behavioural Early Warning Intelligence System</div>', unsafe_allow_html=True)
st.write("Advanced behavioural modelling with dynamic risk simulation and institutional analytics.")

# ==========================================================
# INPUT SECTION
# ==========================================================
st.markdown('<div class="section">📥 Student Behaviour Input Panel</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    attendance = st.slider("Attendance (%)", 50, 100, 80)
    login_freq = st.slider("LMS Activity Frequency", 1, 25, 12)
    delay = st.slider("Assignment Delay (Days)", 0.0, 10.0, 2.0)

with col2:
    login_trend = st.slider("Engagement Trend", -2.0, 2.0, 0.0)
    sentiment = st.slider("Sentiment Score", -1.0, 1.0, 0.2)
    variability = st.slider("Study Pattern Variability", 0.0, 1.0, 0.3)

# ==========================================================
# LIVE RISK CALCULATION
# ==========================================================
stress = 0.6*(delay/10) + 0.4*((100-attendance)/100)
max_trend = df["login_trend_slope"].abs().max() or 1
engagement = abs(login_trend)/max_trend
emotion = max(0, -sentiment)

live_score = (
    0.25*(25-login_freq)/25 +
    0.20*stress +
    0.20*engagement +
    0.20*emotion +
    0.15*variability
)

live_score = round(100*live_score,2)

if live_score < low_cut:
    level = "Low Risk"
    color = "#16A34A"
elif live_score < high_cut:
    level = "Moderate Risk"
    color = "#F97316"
else:
    level = "High Risk"
    color = "#DC2626"

# ==========================================================
# GAUGE + SCORE
# ==========================================================
st.markdown('<div class="section">📊 Risk Assessment</div>', unsafe_allow_html=True)

col3, col4 = st.columns([1,1])

with col3:
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=live_score,
        gauge={
            'axis': {'range': [0,100]},
            'bar': {'color': color},
            'steps': [
                {'range':[0,low_cut],'color':'#BBF7D0'},
                {'range':[low_cut,high_cut],'color':'#FED7AA'},
                {'range':[high_cut,100],'color':'#FECACA'}
            ]
        }
    ))
    st.plotly_chart(gauge, use_container_width=True)

with col4:
    st.markdown(f'<div class="card"><h2 style="color:{color}">{level}</h2><h3>Score: {live_score}</h3></div>', unsafe_allow_html=True)

# ==========================================================
# TRIGGERS (3 CLEAR POINTS)
# ==========================================================
st.markdown('<div class="section">🔍 Primary Behavioural Triggers</div>', unsafe_allow_html=True)

trigger_values = {
    "Attendance Gap": 100-attendance,
    "Assignment Delay": delay,
    "Negative Emotion": emotion,
    "Low Engagement": 25-login_freq,
    "Study Instability": variability*10
}

sorted_triggers = sorted(trigger_values.items(), key=lambda x:x[1], reverse=True)
top3 = sorted_triggers[:3]

for t in top3:
    st.write(f"• {t[0]} contributing significantly to elevated risk.")

# Trigger chart
trigger_df = pd.DataFrame(top3, columns=["Factor","Intensity"])
fig_trig = px.bar(trigger_df, x="Intensity", y="Factor", orientation="h",
                  color="Intensity", color_continuous_scale="magma")
st.plotly_chart(fig_trig, use_container_width=True)

# ==========================================================
# INTERVENTION (3 STRUCTURED POINTS)
# ==========================================================
st.markdown('<div class="section">🛠 Recommended Intervention Strategy</div>', unsafe_allow_html=True)

if level == "High Risk":
    st.error("""
    1️⃣ Immediate academic advisor consultation  
    2️⃣ Mental health & emotional support referral  
    3️⃣ Academic workload restructuring & monitoring  
    """)
elif level == "Moderate Risk":
    st.warning("""
    1️⃣ Weekly behavioural tracking & mentoring  
    2️⃣ Structured academic planning support  
    3️⃣ Attendance & submission monitoring  
    """)
else:
    st.success("""
    1️⃣ Encourage leadership & engagement programs  
    2️⃣ Skill enhancement workshops  
    3️⃣ Positive reinforcement feedback  
    """)

# ==========================================================
# ANIMATED TREND SIMULATION
# ==========================================================
st.markdown('<div class="section">📈 Simulated Risk Trend Projection</div>', unsafe_allow_html=True)

trend_months = pd.DataFrame({
    "Month": ["Jan","Feb","Mar","Apr","May","Jun"],
    "Projected Risk": np.linspace(live_score*0.8, live_score, 6)
})

fig_trend = px.line(trend_months, x="Month", y="Projected Risk",
                    markers=True, line_shape="spline",
                    color_discrete_sequence=["#3B82F6"])
st.plotly_chart(fig_trend, use_container_width=True)

# ==========================================================
# INSTITUTIONAL ANALYTICS (UPGRADED)
# ==========================================================
st.markdown('<div class="section">🏫 Institutional Behavioural Analytics</div>', unsafe_allow_html=True)

col5, col6 = st.columns(2)

with col5:
    donut = px.pie(df, names="burnout_risk_level", hole=0.5,
                   color_discrete_sequence=["#16A34A","#F97316","#DC2626"])
    st.plotly_chart(donut, use_container_width=True)

with col6:
    scatter = px.scatter(df,
                         x="attendance_percent",
                         y="dropout_probability",
                         color="burnout_risk_level",
                         size="risk_engine",
                         hover_data=["avg_assignment_delay_days"],
                         color_discrete_sequence=["#16A34A","#F97316","#DC2626"])
    st.plotly_chart(scatter, use_container_width=True)

st.markdown("---")
st.write("Behavioural Early Warning Intelligence | Advanced Hackathon Demonstration")