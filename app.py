"""
Author: Mohammed Adhil Ali
Final AI Lead Intelligence Dashboard
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from sentence_transformers import SentenceTransformer

from lead_scoring import process_leads
from column_mapper import auto_map_columns, load_memory, save_memory

st.set_page_config(page_title="Lead Intelligence", layout="wide")

# ---------------------------------------------------
# 🎨 UI STYLE
# ---------------------------------------------------

st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
}
h1, h2, h3 {
    color: white !important;
}
.kpi-card {
    padding: 18px;
    border-radius: 12px;
    text-align: center;
    font-weight: bold;
    font-size: 18px;
    color: white;
}
.hot { background: linear-gradient(135deg, #ff416c, #ff4b2b); }
.warm { background: linear-gradient(135deg, #f7971e, #ffd200); color: black; }
.cold { background: linear-gradient(135deg, #2193b0, #6dd5ed); }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>🚀 AI Lead Intelligence Dashboard</h1>", unsafe_allow_html=True)

COLOR_MAP = {"HOT": "#FF4B4B", "WARM": "#FFA500", "COLD": "#1E90FF"}

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.markdown("## 🧭 Navigation")
page = st.sidebar.radio("", ["Upload Data", "Dashboard"])

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

if "processed_df" not in st.session_state:
    st.session_state.processed_df = None

# ===================================================
# 📂 UPLOAD PAGE
# ===================================================

if page == "Upload Data":

    st.subheader("📂 Upload Dataset")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        columns = df.columns.tolist()

        mapping, _, suggestions = auto_map_columns(columns)
        memory = load_memory()

        st.subheader("🧠 Column Mapping")

        expected_cols = ["Full_Name", "Interest_Level", "Purchase_Timeline", "Budget_Range"]

        user_mapping = {}

        for col in columns:
            default = memory.get(col, suggestions.get(col, "Ignore"))

            selected = st.selectbox(
                f"{col}",
                ["Ignore"] + expected_cols,
                index=(["Ignore"] + expected_cols).index(default)
                if default in expected_cols else 0
            )

            if selected != "Ignore":
                user_mapping[col] = selected

        if st.button("🚀 Run Intelligence"):

            for k, v in user_mapping.items():
                memory[k] = v
            save_memory(memory)

            df = df.rename(columns=user_mapping)

            model = load_model()
            df = process_leads(df, model)

            # ✅ Conversion Probability
            df["Conversion_Probability"] = (df["Lead_Score"] * df["Confidence"]).round(2)

            st.session_state.processed_df = df
            st.success("✅ Done! Go to Dashboard")

# ===================================================
# 📊 DASHBOARD
# ===================================================

if page == "Dashboard":

    if st.session_state.processed_df is None:
        st.warning("Upload data first")
        st.stop()

    df = st.session_state.processed_df.copy()

    # KPI
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total", len(df))
    col2.metric("🔥 HOT", (df["Lead_Category"] == "HOT").sum())
    col3.metric("🌤️ WARM", (df["Lead_Category"] == "WARM").sum())
    col4.metric("❄️ COLD", (df["Lead_Category"] == "COLD").sum())

    # -------------------------------
    # 🔥 PRIORITY LEADS (NEW)
    # -------------------------------
    st.subheader("🚀 Priority Leads (High Confidence + High Score)")

    priority = df[
        (df["Lead_Score"] > 70) &
        (df["Confidence"] > 0.6)
    ].sort_values(by="Conversion_Probability", ascending=False)

    st.dataframe(
        priority[[
            "Full_Name",
            "Lead_Score",
            "Confidence",
            "Conversion_Probability",
            "Lead_Category"
        ]],
        use_container_width=True
    )

    # -------------------------------
    # 🔥 TOP 5 LEADS
    # -------------------------------
    st.subheader("🔥 Top 5 Leads")

    top = df.sort_values(by="Lead_Score", ascending=False).head(5)

    st.dataframe(
        top[[
            "Full_Name",
            "Lead_Score",
            "Confidence",
            "Conversion_Probability",
            "Lead_Category"
        ]],
        use_container_width=True
    )

    # -------------------------------
    # 📊 CHARTS
    # -------------------------------
    colA, colB = st.columns(2)

    pie = px.pie(df, names="Lead_Category", color="Lead_Category", color_discrete_map=COLOR_MAP)
    colA.plotly_chart(pie, use_container_width=True)

    hist = px.histogram(df, x="Lead_Score", nbins=10)
    colB.plotly_chart(hist, use_container_width=True)

    # -------------------------------
    # 🔍 FILTER
    # -------------------------------
    st.subheader("🔍 Filter Leads")

    f1, f2 = st.columns(2)
    category = f1.selectbox("Category", ["All", "HOT", "WARM", "COLD"])
    name = f2.text_input("Search Name")

    filtered = df.copy()

    if category != "All":
        filtered = filtered[filtered["Lead_Category"] == category]

    if name:
        filtered = filtered[filtered["Full_Name"].str.contains(name, case=False)]

    def highlight(row):
        if row["Lead_Category"] == "HOT":
            return ["background-color: #5c1a1a"] * len(row)
        elif row["Lead_Category"] == "WARM":
            return ["background-color: #5c4a1a"] * len(row)
        else:
            return ["background-color: #1a3a5c"] * len(row)

    styled = filtered.style.apply(highlight, axis=1).set_properties(**{
        "white-space": "normal",
        "word-wrap": "break-word"
    })

    st.dataframe(
        styled[[
            "Full_Name",
            "Interest_Level",
            "Purchase_Timeline",
            "Budget_Range",
            "Lead_Score",
            "Confidence",
            "Conversion_Probability",
            "Lead_Category",
            "Explanation"
        ]],
        use_container_width=True
    )

    # DOWNLOAD
    st.download_button("⬇️ Download", filtered.to_csv(index=False), "leads.csv")