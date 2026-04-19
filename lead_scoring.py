import pandas as pd
from sentence_transformers import util

# ---------------------------------------------------
# REFERENCE PHRASES
# ---------------------------------------------------

INTEREST_REF = {
    "high": ["ready to buy", "very interested", "serious buyer", "high intent", "eager buyer"],
    "medium": ["considering", "moderate interest", "thinking"],
    "low": ["just browsing", "curious", "window shopping"]
}

TIMELINE_REF = {
    "short": ["immediate", "asap", "urgent", "today", "right away"],
    "medium": ["soon", "1-3 months", "this week"],
    "long": ["later", "future", "not sure", "exploring"]
}

BUDGET_REF = {
    "high": ["premium", "high budget", "large budget"],
    "medium": ["average", "medium"],
    "low": ["low", "cheap"]
}

# ---------------------------------------------------
# AI MATCHING
# ---------------------------------------------------

def get_ai_score(text, reference_dict, score_map, model):

    if pd.isna(text) or str(text).strip() == "":
        return 0.3

    text_embedding = model.encode(str(text), convert_to_tensor=True)

    best_score = 0

    for category, phrases in reference_dict.items():
        phrase_embeddings = model.encode(phrases, convert_to_tensor=True)
        similarity = util.cos_sim(text_embedding, phrase_embeddings).max().item()

        weighted_score = similarity * score_map[category]
        best_score = max(best_score, weighted_score)

    return best_score


# ---------------------------------------------------
# SCORING + EXPLANATION
# ---------------------------------------------------

def calculate_lead_score(row, model):

    interest = row.get("Interest_Level", "")
    timeline = row.get("Purchase_Timeline", "")
    budget = row.get("Budget_Range", "")

    interest_score = get_ai_score(
        interest, INTEREST_REF,
        {"high": 1.0, "medium": 0.6, "low": 0.2},
        model
    )

    timeline_score = get_ai_score(
        timeline, TIMELINE_REF,
        {"short": 1.0, "medium": 0.6, "long": 0.2},
        model
    )

    budget_score = get_ai_score(
        budget, BUDGET_REF,
        {"high": 1.0, "medium": 0.6, "low": 0.2},
        model
    )

    score = (interest_score * 0.4 + timeline_score * 0.3 + budget_score * 0.3) * 100

    explanation = f"Interest: {interest} | Timeline: {timeline} | Budget: {budget}"

    return round(score), explanation


# ---------------------------------------------------
# CATEGORY
# ---------------------------------------------------

def classify_lead(score):
    if score >= 75:
        return "HOT"
    elif score >= 45:
        return "WARM"
    else:
        return "COLD"


# ---------------------------------------------------
# MAIN PROCESS
# ---------------------------------------------------

def process_leads(df, model):

    for col in ["Interest_Level", "Purchase_Timeline", "Budget_Range"]:
        if col not in df.columns:
            df[col] = ""

    results = df.apply(lambda row: calculate_lead_score(row, model), axis=1)

    df["Lead_Score"] = results.apply(lambda x: x[0]).clip(0, 100)
    df["Explanation"] = results.apply(lambda x: x[1])

    df["Lead_Category"] = df["Lead_Score"].apply(classify_lead)
    df["Conversion_Probability"] = df["Lead_Score"]

    return df