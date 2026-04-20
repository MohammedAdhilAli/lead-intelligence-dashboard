"""
Author: Mohammed Adhil Ali
Column Mapping + Memory System
"""

import json
import os

# ---------------------------------------------------
# 📁 MEMORY FILE
# ---------------------------------------------------

MEMORY_FILE = "column_memory.json"

# ---------------------------------------------------
# 🔄 LOAD MEMORY
# ---------------------------------------------------

def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return {}
    with open(MEMORY_FILE, "r") as f:
        return json.load(f)

# ---------------------------------------------------
# 💾 SAVE MEMORY
# ---------------------------------------------------

def save_memory(memory):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=4)

# ---------------------------------------------------
# 🧠 AUTO COLUMN MAPPING
# ---------------------------------------------------

def auto_map_columns(columns):
    """
    Returns:
    mapping → confident mappings
    unmapped → columns not mapped
    suggestions → best guess for UI
    """

    mapping = {}
    suggestions = {}
    unmapped = []

    for col in columns:
        col_lower = col.lower()

        # -------------------------------
        # NAME
        # -------------------------------
        if any(k in col_lower for k in ["name", "customer", "client"]):
            mapping[col] = "Full_Name"
            suggestions[col] = "Full_Name"

        # -------------------------------
        # INTEREST
        # -------------------------------
        elif any(k in col_lower for k in ["interest", "intent", "engagement"]):
            mapping[col] = "Interest_Level"
            suggestions[col] = "Interest_Level"

        # -------------------------------
        # TIMELINE
        # -------------------------------
        elif any(k in col_lower for k in ["timeline", "time", "decision", "purchase"]):
            mapping[col] = "Purchase_Timeline"
            suggestions[col] = "Purchase_Timeline"

        # -------------------------------
        # BUDGET
        # -------------------------------
        elif any(k in col_lower for k in ["budget", "price", "value", "amount"]):
            mapping[col] = "Budget_Range"
            suggestions[col] = "Budget_Range"

        # -------------------------------
        # UNKNOWN
        # -------------------------------
        else:
            unmapped.append(col)
            suggestions[col] = "Ignore"

    return mapping, unmapped, suggestions