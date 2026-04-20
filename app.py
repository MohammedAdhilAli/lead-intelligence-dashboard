"""
Author: Mohammed Adhil Ali
Project: AI Lead Intelligence Dashboard
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

h1, h2, h3, h4, h5, h6, p {
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

.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------

st.markdown("""
<h1 style='text-align: center; margin-bottom: 30px;'>
🚀 AI Lead Intelligence Dashboard
</h1>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# COLORS
# ---------------------------------------------------

COLOR_MAP = {
    "HOT": "#FF4B4B",
    "WARM": "#FFA500",
    "COLD": "#1E90FF"
}

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.markdown("<h2>🧭 Navigation</h2>", unsafe_allow_html=True)
page = st.sidebar.radio("", ["Upload Data", "Dashboard"])

# ---------------------------------------------------
# MODEL
# ---------------------------------------------------

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

if "processed_df" not in st.session_state:
    st.session_state.processed_df = None

# ===================================================
# 📂 UPLOAD PAGE
# ===================================================

if page == "Upload Data":

    st.markdown("<h2>📂 Upload Dataset</h2>", unsafe_allow_html=True)

    # -------------------------------
    # 🧪 SAMPLE DATA
    # -------------------------------
    st.markdown("### 🧪 Try Sample Dataset")

    try:
        with open("Sample_Dataset.csv", "rb") as file:
            st.download_button(
                label="⬇️ Download Sample Dataset",
                data=file,
                file_name="Sample_Dataset.csv",
                mime="text/csv"
            )

        st.markdown("### 👀 Sample Preview")
        sample_df = pd.read_csv("Sample_Dataset.csv")
        st.dataframe(sample_df.head())

    except:
        st.warning("Sample dataset not found.")

    # -------------------------------
    # 📂 FILE UPLOAD
    # -------------------------------
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")

        columns = df.columns.tolist()
        mapping, _, suggestions = auto_map_columns(columns)

        # 🔥 LOAD MEMORY
        memory = load_memory()

        st.markdown("<h3>🧠 Column Mapping (Editable)</h3>", unsafe_allow_html=True)

        expected_cols = [
            "Full_Name",
            "Interest_Level",
            "Purchase_Timeline",
            "Budget_Range"
        ]

        user_mapping = {}

        col1, col2 = st.columns(2)

        for i, col in enumerate(columns):

            # Priority: Memory → Suggestion → Ignore
            default_value = memory.get(col, suggestions.get(col, "Ignore"))

            with (col1 if i % 2 == 0 else col2):
                selected = st.selectbox(
                    f"Map: {col}",
                    ["Ignore"] + expected_cols,
                    index=(["Ignore"] + expected_cols).index(default_value)
                    if default_value in expected_cols else 0,
                    key=f"map_{col}"
                )

                if selected != "Ignore":
                    user_mapping[col] = selected

        # -------------------------------
        # 🚀 RUN ENGINE
        # -------------------------------
        if st.button("🚀 Run Intelligence"):

            if not user_mapping:
                st.error("⚠️ Please map at least one column.")
                st.stop()

            # 🔥 SAVE MAPPING MEMORY
            for original, mapped in user_mapping.items():
                memory[original] = mapped
            save_memory(memory)

            df = df.rename(columns=user_mapping)

            model = load_model()
            df = process_leads(df, model)

            df["Lead_Score"] = df["Lead_Score"].clip(0, 100)

            def recommend(row):
                if row["Lead_Category"] == "HOT":
                    return "Call Immediately"
                elif row["Lead_Category"] == "WARM":
                    return "Follow-up Email"
                else:
                    return "Add to Nurture Campaign"

            df["Recommended_Action"] = df.apply(recommend, axis=1)

            st.session_state.processed_df = df
            st.success("✅ Processing complete! Go to Dashboard")

# ===================================================
# 📊 DASHBOARD PAGE
# ===================================================

if page == "Dashboard":

    st.markdown("<h2>📊 Dashboard Overview</h2>", unsafe_allow_html=True)

    if st.session_state.processed_df is None:
        st.warning("⚠️ Please upload data first.")
        st.stop()

    df = st.session_state.processed_df

    # KPI
    total = len(df)
    hot = (df["Lead_Category"] == "HOT").sum()
    warm = (df["Lead_Category"] == "WARM").sum()
    cold = (df["Lead_Category"] == "COLD").sum()
    avg = round(df["Lead_Score"].mean(), 2)

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.markdown(f"<div class='kpi-card'>Total<br>{total}</div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='kpi-card hot'>🔥 HOT<br>{hot}</div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='kpi-card warm'>🌤️ WARM<br>{warm}</div>", unsafe_allow_html=True)
    col4.markdown(f"<div class='kpi-card cold'>❄️ COLD<br>{cold}</div>", unsafe_allow_html=True)
    col5.markdown(f"<div class='kpi-card'>Avg<br>{avg}</div>", unsafe_allow_html=True)

    # Charts
    colA, colB = st.columns(2)

    pie = px.pie(
        df,
        names="Lead_Category",
        color="Lead_Category",
        color_discrete_map=COLOR_MAP,
        hole=0.4
    )
    pie.update_layout(plot_bgcolor="rgba(0,0,0,0)", font_color="white")
    colA.plotly_chart(pie, use_container_width=True)

    bins = list(range(0, 110, 10))
    labels = [f"{i}-{i+10}" for i in bins[:-1]]

    df["Score_Range"] = pd.cut(df["Lead_Score"], bins=bins, labels=labels)

    hist = df["Score_Range"].value_counts().sort_index().reset_index()
    hist.columns = ["Range", "Count"]

    bar = px.bar(
        hist,
        x="Range",
        y="Count",
        text="Count",
        color="Range",
        color_discrete_sequence=px.colors.sequential.Tealgrn
    )
    bar.update_layout(plot_bgcolor="rgba(0,0,0,0)", font_color="white")
    colB.plotly_chart(bar, use_container_width=True)

    # Top Leads
    st.markdown("<h3>🔥 Top Opportunities</h3>", unsafe_allow_html=True)
    top = df.sort_values(by="Lead_Score", ascending=False).head(5)
    st.dataframe(top, use_container_width=True)

    # Filter
    st.markdown("<h3>🔍 Filter Leads</h3>", unsafe_allow_html=True)

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
            return ["background-color: #5c1a1a; color: white"] * len(row)
        elif row["Lead_Category"] == "WARM":
            return ["background-color: #5c4a1a; color: white"] * len(row)
        else:
            return ["background-color: #1a3a5c; color: white"] * len(row)

    st.dataframe(filtered.style.apply(highlight, axis=1), use_container_width=True)

    st.download_button(
        "⬇️ Download Data",
        filtered.to_csv(index=False),
        "leads.csv"
    )