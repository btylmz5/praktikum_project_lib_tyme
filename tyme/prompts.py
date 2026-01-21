from __future__ import annotations
import json
from typing import Any, Optional


def build_suggest_prompt(
    profile: dict[str, Any],
    task: str,
    target: Optional[str],
    exclude_columns: Optional[list[str]] = None,
) -> str:
    target_line = f"Target column: {target}" if target else "Target column: (not provided)"
    task_line = f"Task type: {task} (classification/regression/unspecified)"
    
    exclude_text = ""
    if exclude_columns:
        exclude_text = f"- Do NOT use the following columns in any suggestions: {', '.join(exclude_columns)}\n"

    # Separate columns by type for clearer prompt
    numeric_cols = [c["name"] for c in profile["columns"] if c["inferred_type"] == "numeric"]
    categorical_cols = [c["name"] for c in profile["columns"] if c["inferred_type"] == "categorical"]
    other_cols = [c["name"] for c in profile["columns"] if c["inferred_type"] not in ("numeric", "categorical")]

    schema = [
        {
            "name": "string (e.g., 'Log_FeatureX' or 'Ratio_ColA_ColB')",
            "depends_on": ["colA", "colB"],
            "how": "string (precise step-by-step transformation description)",
            "why": "string (statistical justification)",
            "feature_type": "numeric|categorical|datetime|text|interaction",
            "risk": "none|leakage|overfit|data_quality|unknown",
        }
    ]

    return (
        "You are a Kaggle Grandmaster and Senior Feature Engineer.\n"
        "Your goal is to win a competition by creating NEW, high-value information from an existing dataset.\n\n"
        f"{task_line}\n"
        f"{target_line}\n\n"
        "AVAILABLE COLUMNS (By Type):\n"
        f"NUMERIC: {', '.join(numeric_cols)}\n"
        f"CATEGORICAL: {', '.join(categorical_cols)}\n"
        f"OTHER: {', '.join(other_cols)}\n\n"
        "GUIDELINES:\n"
        "1. **Strict Column Usage**: You MUST ONLY use the columns listed above. Do NOT invent columns.\n"
        "2. **Constraint**: You MUST propose exactly 10 suggestions. Fill the list with simple features if needed to reach 10.\n"
        "3. **Respect Data Types**: ONLY apply math (Log, Ratio, Diff) to NUMERIC columns. Do NOT divide by Categorical columns.\n"
        "4. **Focus**: Look for Interactions (Ratio between two numerics) and Aggregations (Group by Categorical, Mean of Numeric).\n"
        "5. **Why**: Explain the *statistical mechanism*.\n"
        "6. **Leakage**: If a feature uses future info, set risk='leakage'.\n\n"
        "TEMPLATE EXAMPLES (Replace placeholders with ACTUAL columns):\n"
        "- Suggestion: 'Ratio_NumA_NumB'. How: 'NumA / NumB'. Why: 'Captures efficiency'.\n"
        "- Suggestion: 'Log_NumA'. How: 'log(NumA)'. Why: 'Stabilizes variance'.\n"
        f"{exclude_text}"
        "Your Output MUST be a valid JSON array of 10 suggestions obeying this exact schema:\n"
        f"{json.dumps(schema, indent=2)}\n\n"
        "FULL DATASET PROFILE (JSON):\n"
        f"{json.dumps(profile, ensure_ascii=False)}\n"
    )


def build_chat_prompt(
    profile: dict[str, Any],
    suggestions_jsonable: list[dict[str, Any]],
    history: list[dict[str, str]],
    user_message: str,
) -> str:
    """
    We use a single prompt string for Ollama generate. We embed structured context
    and a short conversation history.
    """
    # keep history short to avoid context bloat
    last_history = history[-8:]

    return (
        "You are a helpful feature-engineering assistant.\n"
        "The user already generated feature suggestions for a CSV dataset.\n"
        "Your job is to discuss and refine these suggestions, answer questions, warn about leakage,\n"
        "and provide implementation guidance (pandas/sklearn style) when asked.\n"
        "If the user references a number, interpret it as the corresponding suggestion index (1-based).\n"
        "Be concrete and actionable.\n\n"
        "DATASET PROFILE (JSON):\n"
        f"{json.dumps(profile, ensure_ascii=False)}\n\n"
        "SUGGESTIONS (JSON):\n"
        f"{json.dumps(suggestions_jsonable, ensure_ascii=False)}\n\n"
        "RECENT CHAT:\n"
        + "\n".join([f"{m['role'].upper()}: {m['content']}" for m in last_history])
        + "\n\n"
        f"USER: {user_message}\n"
        "ASSISTANT:"
    )
