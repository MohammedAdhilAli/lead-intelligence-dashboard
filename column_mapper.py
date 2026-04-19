"""
Author: Mohammed Adhil Ali
Project: AI Lead Intelligence Dashboard
"""

import json
import os
from rapidfuzz import process

# SAFE IMPORT
try:
    from sentence_transformers import SentenceTransformer, util
except:
    SentenceTransformer = None
    util = None

from functools import lru_cache

MEMORY_FILE = "column_memory.json"

# ---------------------------------------------------
# MODEL LOADING (LAZY + CACHED)
# ---------------------------------------------------

@lru_cache(maxsize=1)
def get_model():
    try:
        if SentenceTransformer:
            return SentenceTransformer("all-MiniLM-L6-v2")
    except:
        return None
    return None

# ---------------------------------------------------
# LOAD & SAVE MEMORY
# ---------------------------------------------------

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return {}

def save_memory(memory):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=4)

# ---------------------------------------------------
# SEMANTIC DEFINITIONS
# ---------------------------------------------------

schema_meanings = {
    "Full_Name": ["customer name", "client name", "full name", "person name"],
    "Interest_Level": ["interest level", "buying intent", "customer interest", "engagement"],
    "Purchase_Timeline": ["purchase time", "decision time", "urgency", "timeline"],
    "Budget_Range": ["budget", "price range", "cost", "deal value", "amount"]
}

# ---------------------------------------------------
# EMBEDDINGS (LAZY LOAD)
# ---------------------------------------------------

schema_embeddings = {}

def get_schema_embeddings():
    global schema_embeddings

    if schema_embeddings:
        return schema_embeddings

    model = get_model()
    if not model:
        return {}

    schema_embeddings = {
        key: model.encode(values, convert_to_tensor=True)
        for key, values in schema_meanings.items()
    }

    return schema_embeddings

# ---------------------------------------------------
# FUZZY MATCH
# ---------------------------------------------------

def score_match(column, keywords):
    match = process.extractOne(column, keywords)
    return match[1] if match else 0

# ---------------------------------------------------
# SEMANTIC MATCH (SAFE)
# ---------------------------------------------------

def semantic_match(column):

    model = get_model()
    if not model or not util:
        return None, 0

    try:
        col_embedding = model.encode(column, convert_to_tensor=True)
        embeddings_dict = get_schema_embeddings()

        best_label = None
        best_score = 0

        for label, embeddings in embeddings_dict.items():
            score = util.cos_sim(col_embedding, embeddings).max().item() * 100

            if score > best_score:
                best_score = score
                best_label = label

        return best_label, best_score

    except:
        return None, 0

# ---------------------------------------------------
# MAIN FUNCTION
# ---------------------------------------------------

def auto_map_columns(columns):

    memory = load_memory()
    mapping = {}
    confidence = {}
    suggestions = {}

    standard_columns = [
        "Full_Name",
        "Interest_Level",
        "Purchase_Timeline",
        "Budget_Range"
    ]

    for col in columns:

        col_lower = col.lower()

        # EXACT MATCH
        if col in standard_columns:
            mapping[col] = col
            confidence[col] = 100
            suggestions[col] = col
            continue

        # MEMORY
        if col_lower in memory:
            mapping[col] = memory[col_lower]
            confidence[col] = 100
            suggestions[col] = memory[col_lower]
            continue

        # FUZZY
        fuzzy_scores = {
            "Full_Name": score_match(col_lower, ["name", "customer", "client"]),
            "Interest_Level": score_match(col_lower, ["interest", "intent"]),
            "Purchase_Timeline": score_match(col_lower, ["timeline", "time", "period"]),
            "Budget_Range": score_match(col_lower, ["budget", "price", "value"])
        }

        fuzzy_best = max(fuzzy_scores, key=fuzzy_scores.get)
        fuzzy_score = fuzzy_scores[fuzzy_best]

        # SEMANTIC
        semantic_best, semantic_score = semantic_match(col_lower)

        # HYBRID DECISION
        if semantic_best and semantic_score > fuzzy_score:
            best_match = semantic_best
            best_score = semantic_score
        else:
            best_match = fuzzy_best
            best_score = fuzzy_score

        confidence[col] = best_score
        suggestions[col] = best_match

        # AUTO MAP
        if best_score > 85:
            mapping[col] = best_match

    return mapping, confidence, suggestions

# ---------------------------------------------------
# UPDATE MEMORY
# ---------------------------------------------------

def update_memory(user_mapping):

    memory = load_memory()

    for original, mapped in user_mapping.items():
        memory[original.lower()] = mapped

    save_memory(memory)