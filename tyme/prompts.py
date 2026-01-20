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

    schema = [
        {
            "name": "string",
            "depends_on": ["colA", "colB"],
            "how": "string (step-by-step transformation description)",
            "why": "string (why it may help predictive performance)",
            "feature_type": "numeric|categorical|datetime|text|unknown",
            "risk": "none|leakage|overfit|data_quality|unknown",
        }
    ]

    return (
        "You are a senior data scientist specialized in feature engineering.\n"
        "You will be given a dataset profile extracted from a CSV.\n"
        f"{task_line}\n"
        f"{target_line}\n\n"
        "Your job: propose 8-12 feature engineering ideas that could improve a ML model.\n"
        "Important rules:\n"
        "- Do NOT use target leakage. If a suggestion risks leakage, set risk='leakage' and explain why.\n"
        f"{exclude_text}"
        "- Suggestions must be generally applicable and based only on columns that exist.\n"
        "- Output MUST be ONLY a valid JSON array. No markdown, no commentary.\n"
        f"- Each element MUST follow this schema keys exactly:\n{json.dumps(schema, indent=2)}\n\n"
        "DATASET PROFILE (JSON):\n"
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
