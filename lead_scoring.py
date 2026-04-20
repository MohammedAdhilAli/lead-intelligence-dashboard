"""
Author: Mohammed Adhil Ali
Optimized AI Lead Scoring with Confidence
"""

import numpy as np

# ---------------------------------------------------
# 🎯 REFERENCE TEXTS
# ---------------------------------------------------

REFERENCE = {
    "HIGH": [
        "ready to buy immediately",
        "urgent purchase",
        "highly interested buyer",
        "needs product now",
        "priority client"
    ],
    "MEDIUM": [
        "interested but exploring",
        "considering options",
        "may buy soon",
        "evaluating product"
    ],
    "LOW": [
        "just browsing",
        "not interested",
        "no urgency",
        "casual inquiry"
    ]
}

# ---------------------------------------------------
# ⚡ PRECOMPUTE EMBEDDINGS (IMPORTANT OPTIMIZATION)
# ---------------------------------------------------

REFERENCE_EMBEDDINGS = {}

def initialize_reference_embeddings(model):
    global REFERENCE_EMBEDDINGS

    if REFERENCE_EMBEDDINGS:
        return  # already computed

    for category, phrases in REFERENCE.items():
        REFERENCE_EMBEDDINGS[category] = model.encode(phrases)

# ---------------------------------------------------
# 🔧 UTILITIES
# ---------------------------------------------------

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ---------------------------------------------------
# 🧠 AI SCORING FUNCTION
# ---------------------------------------------------

def get_ai_score(text, model):
    text = str(text)

    text_vec = model.encode([text])[0]

    scores = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}

    for category, vectors in REFERENCE_EMBEDDINGS.items():
        sims = []

        for vec in vectors:
            sim = cosine_similarity(text_vec, vec)
            sims.append(sim)

        scores[category] = max(sims)

    # Best category
    best_category = max(scores, key=scores.get)
    confidence = scores[best_category]

    if best_category == "HIGH":
        score = 1.0
    elif best_category == "MEDIUM":
        score = 0.6
    else:
        score = 0.2

    return score, confidence

# ---------------------------------------------------
# 🧮 FINAL LEAD SCORE
# ---------------------------------------------------

def calculate_lead_score(row, model):

    interest = row.get("Interest_Level", "")
    timeline = row.get("Purchase_Timeline", "")
    budget = row.get("Budget_Range", "")

    i_score, i_conf = get_ai_score(interest, model)
    t_score, t_conf = get_ai_score(timeline, model)
    b_score, b_conf = get_ai_score(budget, model)

    final_score = (
        i_score * 0.4 +
        t_score * 0.3 +
        b_score * 0.3
    ) * 100

    # Average confidence
    confidence = (i_conf + t_conf + b_conf) / 3

    return round(final_score, 2), round(confidence, 3)

# ---------------------------------------------------
# 🏷️ CATEGORY
# ---------------------------------------------------

def categorize(score):
    if score >= 70:
        return "HOT"
    elif score >= 40:
        return "WARM"
    else:
        return "COLD"

# ---------------------------------------------------
# 🔄 PROCESS DATAFRAME
# ---------------------------------------------------

def process_leads(df, model):

    # 🔥 Initialize reference embeddings ONCE
    initialize_reference_embeddings(model)

    scores = []
    confidences = []

    for _, row in df.iterrows():
        score, conf = calculate_lead_score(row, model)
        scores.append(score)
        confidences.append(conf)

    df["Lead_Score"] = scores
    df["Confidence"] = confidences
    df["Lead_Category"] = df["Lead_Score"].apply(categorize)

    return df