"""
Author: Mohammed Adhil Ali
Final Clean AI Lead Intelligence Dashboard
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from sentence_transformers import SentenceTransformer

from lead_scoring import process_leads
from column_mapper import auto_map_columns, load_memory, save_memory

st.set_page_config(page_title="Lead Intelligence", layout="wide")

# ---------------- STYLE ----------------
st.markdown("""
<style>
.stApp { background: linear-gradient(to right, #0f2027, #203a43, #2c5364); }
h1, h2, h3, h4 { color: white !important; }
.kpi-card {
    padding: 18px; border-radius: 12px; text-align: center;
    font-weight: bold; font-size: 18px; color: white;
}
.hot { background: linear-gradient(135deg, #ff416c, #ff4b2b); }
.warm { background: linear-gradient(135deg, #f7971e, #ffd200); color: black; }
.cold { background: linear-gradient(135deg, #2193b0, #6dd5ed); }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>🚀 AI Lead Intelligence Dashboard</h1>", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
page = st.sidebar.radio("Navigation", ["Upload Data", "Dashboard"])

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

if "processed_df" not in st.session_state:
    st.session_state.processed_df = None

# ===================================================
# 📂 UPLOAD
# ===================================================

if page == "Upload Data":

    st.subheader("📂 Upload Dataset")

    # Sample dataset
    sample = pd.DataFrame({
        "Full_Name": ["Rahul", "Anita", "Vikram"],
        "Interest_Level": ["very interested", "interested", "just browsing"],
        "Purchase_Timeline": ["immediate", "1-3 months", "later"],
        "Budget_Range": ["high", "medium", "low"]
    })

    st.download_button("⬇️ Sample Dataset", sample.to_csv(index=False), "sample.csv")
    st.dataframe(sample)

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)

        mapping, _, suggestions = auto_map_columns(df.columns)
        memory = load_memory()

        user_mapping = {}

        for col in df.columns:
            default = memory.get(col, suggestions.get(col, "Ignore"))

            selected = st.selectbox(
                col,
                ["Ignore", "Full_Name", "Interest_Level", "Purchase_Timeline", "Budget_Range"]
            )

            if selected != "Ignore":
                user_mapping[col] = selected

        if st.button("🚀 Run"):

            for k, v in user_mapping.items():
                memory[k] = v
            save_memory(memory)

            df = df.rename(columns=user_mapping)

            model = load_model()
            df = process_leads(df, model)

            df["Lead_Score"] = df["Lead_Score"].clip(0, 100)

            # ✅ Probability
            df["Conversion_Probability"] = (df["Lead_Score"] * df["Confidence"]).round(2)

            # ✅ Recommendation
            def recommend(row):
                if row["Lead_Category"] == "HOT":
                    return "Call Immediately"
                elif row["Lead_Category"] == "WARM":
                    return "Follow-up Email"
                else:
                    return "Nurture Campaign"

            df["Recommended_Action"] = df.apply(recommend, axis=1)

            # ✅ SHORT EXPLANATION
            def short_exp(text):
                try:
                    parts = text.split("|")
                    return " | ".join([p.split(":")[1].strip().split("(")[0] for p in parts])
                except:
                    return text

            df["Explanation"] = df["Explanation"].apply(short_exp)

            st.session_state.processed_df = df
            st.success("Done!")

# ===================================================
# 📊 DASHBOARD
# ===================================================

if page == "Dashboard":

    if st.session_state.processed_df is None:
        st.warning("Upload data first")
        st.stop()

    df = st.session_state.processed_df.copy()

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total", len(df))
    c2.metric("HOT", (df["Lead_Category"] == "HOT").sum())
    c3.metric("WARM", (df["Lead_Category"] == "WARM").sum())
    c4.metric("COLD", (df["Lead_Category"] == "COLD").sum())

    # Charts
    colA, colB = st.columns(2)

    pie = px.pie(df, names="Lead_Category")
    colA.plotly_chart(pie, use_container_width=True)

    df["Score_Range"] = pd.cut(df["Lead_Score"], bins=list(range(0,110,10)))
    bar = px.histogram(df, x="Score_Range")
    colB.plotly_chart(bar, use_container_width=True)

    # 🔥 TOP 5 (NO EXPLANATION)
    st.subheader("🔥 Top 5 Leads")

    top = df.sort_values(by="Conversion_Probability", ascending=False).head(5)

    st.dataframe(
        top[[
            "Full_Name",
            "Lead_Score",
            "Confidence",
            "Conversion_Probability",
            "Lead_Category",
            "Recommended_Action"
        ]],
        use_container_width=True
    )

    # 🔍 FILTER TABLE (WITH EXPLANATION)
    st.subheader("🔍 Filter Leads")

    cat = st.selectbox("Category", ["All","HOT","WARM","COLD"])
    name = st.text_input("Search")

    filtered = df.copy()

    if cat != "All":
        filtered = filtered[filtered["Lead_Category"] == cat]

    if name:
        filtered = filtered[filtered["Full_Name"].str.contains(name, case=False)]

    def highlight(row):
        if row["Lead_Category"] == "HOT":
            return ["background-color:#5c1a1a"]*len(row)
        elif row["Lead_Category"] == "WARM":
            return ["background-color:#5c4a1a"]*len(row)
        else:
            return ["background-color:#1a3a5c"]*len(row)

    styled = filtered.style.apply(highlight, axis=1)

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
            "Recommended_Action",
            "Explanation"
        ]],
        use_container_width=True
    )

    st.download_button("⬇️ Download", filtered.to_csv(index=False), "leads.csv")