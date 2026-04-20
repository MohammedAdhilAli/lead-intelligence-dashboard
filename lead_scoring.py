"""
Author: Mohammed Adhil Ali
Explainable AI Lead Scoring
"""

import numpy as np

REFERENCE = {
    "HIGH": [
        "ready to buy immediately",
        "urgent purchase",
        "highly interested buyer",
        "needs product now"
    ],
    "MEDIUM": [
        "considering options",
        "interested but exploring",
        "may buy soon"
    ],
    "LOW": [
        "just browsing",
        "not interested",
        "no urgency"
    ]
}

REFERENCE_EMBEDDINGS = {}

def initialize_reference_embeddings(model):
    global REFERENCE_EMBEDDINGS
    if REFERENCE_EMBEDDINGS:
        return
    for category, phrases in REFERENCE.items():
        REFERENCE_EMBEDDINGS[category] = model.encode(phrases)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 🔥 NEW: returns category + confidence + matched phrase
def analyze_text(text, model):
    text_vec = model.encode([str(text)])[0]

    best_category = None
    best_score = -1
    best_phrase = ""

    for category, vectors in REFERENCE_EMBEDDINGS.items():
        for i, vec in enumerate(vectors):
            sim = cosine_similarity(text_vec, vec)

            if sim > best_score:
                best_score = sim
                best_category = category
                best_phrase = REFERENCE[category][i]

    if best_category == "HIGH":
        score = 1.0
    elif best_category == "MEDIUM":
        score = 0.6
    else:
        score = 0.2

    return score, best_score, best_category, best_phrase

def calculate_lead_score(row, model):

    interest = row.get("Interest_Level", "")
    timeline = row.get("Purchase_Timeline", "")
    budget = row.get("Budget_Range", "")

    i_s, i_c, i_cat, i_phrase = analyze_text(interest, model)
    t_s, t_c, t_cat, t_phrase = analyze_text(timeline, model)
    b_s, b_c, b_cat, b_phrase = analyze_text(budget, model)

    final_score = (i_s * 0.4 + t_s * 0.3 + b_s * 0.3) * 100
    confidence = (i_c + t_c + b_c) / 3

    # 🔥 EXPLANATION LOGIC
    explanation = []

    explanation.append(f"Interest: {i_cat} ({i_phrase})")
    explanation.append(f"Timeline: {t_cat} ({t_phrase})")
    explanation.append(f"Budget: {b_cat} ({b_phrase})")

    explanation_text = " | ".join(explanation)

    return round(final_score, 2), round(confidence, 3), explanation_text

def categorize(score):
    if score >= 70:
        return "HOT"
    elif score >= 40:
        return "WARM"
    else:
        return "COLD"

def process_leads(df, model):

    initialize_reference_embeddings(model)

    scores = []
    confidences = []
    explanations = []

    for _, row in df.iterrows():
        score, conf, exp = calculate_lead_score(row, model)
        scores.append(score)
        confidences.append(conf)
        explanations.append(exp)

    df["Lead_Score"] = scores
    df["Confidence"] = confidences
    df["Explanation"] = explanations
    df["Lead_Category"] = df["Lead_Score"].apply(categorize)

    return df