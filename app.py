import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import shap
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="🌍 AI for Education Equity",
    page_icon="🎓",
    layout="wide"
)

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    file_path = "education.csv"
    if not os.path.exists(file_path):
        st.error(f"❌ File not found: {file_path}")
        st.write("📁 Current directory files:", os.listdir())
        st.stop()

    return pd.read_csv(file_path)



    # Ensure consistent column names
    df = df.rename(columns={
        "female_enrollment": "Female_Enrollment",
        "male_enrollment": "Male_Enrollment",
        "country": "Country",
        "year": "Year"
    })
    return df

df = load_data()

# ================= FEATURE ENGINEERING =================
ml_df = df.copy()

ml_df["gender_gap"] = ml_df["Male_Enrollment"] - ml_df["Female_Enrollment"]
ml_df["egii"] = np.where(
    ml_df["Male_Enrollment"] > 0,
    ml_df["gender_gap"] / ml_df["Male_Enrollment"],
    0
)
ml_df["high_risk"] = (ml_df["egii"] > 0.20).astype(int)

features = ["Female_Enrollment", "Male_Enrollment", "gender_gap", "egii"]
ml_df = ml_df.dropna(subset=features)

X = ml_df[features]
y = ml_df["high_risk"]

# ================= TRAIN MODEL =================
@st.cache_resource
def train_model(X, y):
    stratify_arg = y if y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=stratify_arg
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X_train_s, y_train)

    acc = accuracy_score(y_test, model.predict(X_test_s)) if len(y_test) > 0 else 0
    return model, scaler, acc

model, scaler, model_accuracy = train_model(X, y)

# ================= SIDEBAR =================
st.sidebar.header("🌍 Country Selection")
country = st.sidebar.selectbox(
    "Select Country",
    sorted(df["Country"].dropna().unique())
)

filtered = df[df["Country"] == country].sort_values("Year")
if filtered.empty:
    st.stop()

# ================= COUNTRY METRICS =================
avg_female = filtered["Female_Enrollment"].mean()
avg_male = filtered["Male_Enrollment"].mean()
gender_gap = max(avg_male - avg_female, 0)
egii = gender_gap / avg_male if avg_male > 0 else 0

country_features = pd.DataFrame([[avg_female, avg_male, gender_gap, egii]], columns=features)
scaled_country = scaler.transform(country_features)

prob = model.predict_proba(scaled_country)[0][1] if len(model.classes_) == 2 else model.predict_proba(scaled_country)[0][0]

# ================= HEADER =================
st.markdown("""
<h1 style="text-align:center;">🎓 AI for Education Equity</h1>
<p style="text-align:center;">Fairness-Aware AI for Gender Inequality</p>
""", unsafe_allow_html=True)

# ================= KPI =================
c1, c2, c3, c4 = st.columns(4)
c1.metric("Female Enrollment", f"{avg_female:.1f}%")
c2.metric("Male Enrollment", f"{avg_male:.1f}%")
c3.metric("Gender Gap", f"{gender_gap:.1f}%")
c4.metric("EGII", f"{egii:.2f}")

# ================= TRENDS =================
st.subheader("📈 Enrollment Trends")

chart_data = filtered.melt(
    id_vars="Year",
    value_vars=["Female_Enrollment", "Male_Enrollment"],
    var_name="Gender",
    value_name="Enrollment"
)

chart = alt.Chart(chart_data).mark_line(point=True).encode(
    x="Year:O",
    y="Enrollment",
    color="Gender",
    tooltip=["Year", "Gender", "Enrollment"]
).interactive()

st.altair_chart(chart, use_container_width=True)

# ================= RISK GAUGE =================
st.subheader("🎯 Education Inequality Risk")

gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=prob * 100,
    number={"suffix": "%"},
    gauge={
        "axis": {"range": [0, 100]},
        "steps": [
            {"range": [0, 30], "color": "#2ECC71"},
            {"range": [30, 60], "color": "#F1C40F"},
            {"range": [60, 100], "color": "#E74C3C"}
        ],
        "bar": {"color": "darkred"}
    }
))

gauge.update_layout(height=350)
st.plotly_chart(gauge, use_container_width=True)

# ================= SHAP GLOBAL =================
st.subheader("🧠 Global AI Explainability")

X_scaled = scaler.transform(X)
explainer = shap.TreeExplainer(model)
shap_exp = explainer(X_scaled)

shap_vals = shap_exp.values[:, :, 1] if shap_exp.values.ndim == 3 else shap_exp.values

importance_df = (
    pd.DataFrame(shap_vals, columns=features)
    .abs()
    .mean()
    .sort_values()
    .reset_index()
    .rename(columns={"index": "Feature", 0: "Importance"})
)

fig = px.bar(
    importance_df,
    x="Importance",
    y="Feature",
    orientation="h",
    title="Global Feature Importance (SHAP)",
    color="Importance"
)
st.plotly_chart(fig, use_container_width=True)

# ================= SHAP LOCAL =================
st.subheader("🔍 Why this Country? (Local SHAP)")

local_exp = explainer(scaled_country)
local_vals = local_exp.values[0, :, 1] if local_exp.values.ndim == 3 else local_exp.values[0]

local_df = pd.DataFrame({
    "Feature": features,
    "SHAP Value": local_vals
}).sort_values("SHAP Value")

fig_local = px.bar(
    local_df,
    x="SHAP Value",
    y="Feature",
    orientation="h",
    title="Country-Specific Feature Impact",
    color="SHAP Value",
    color_continuous_scale="RdBu"
)
st.plotly_chart(fig_local, use_container_width=True)

# ================= AFGHANISTAN ALERT =================
if country.lower() == "afghanistan":
    st.error("""
🇦🇫 **Critical Alert**  
Severe gender-based educational exclusion detected.  
Immediate global intervention required.
""")

# ================= FOOTER =================
st.markdown("""
<hr>
<p style="text-align:center;">
<b>Niamatullah Samadi</b><br>
AI • Data Science • Social Good
</p>
""", unsafe_allow_html=True)



