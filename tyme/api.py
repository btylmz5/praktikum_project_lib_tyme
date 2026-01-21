from __future__ import annotations
import pandas as pd
from typing import Any, Optional

from .profile import profile_df
from .prompts import build_suggest_prompt, build_chat_prompt
from .ollama_client import generate_text
from .parsing import parse_suggestions, Suggestion

def get_profile(df: pd.DataFrame) -> dict[str, Any]:
    """
    Generate a statistical profile of the DataFrame.
    
    Args:
        df: Input pandas DataFrame.
        
    Returns:
        Dictionary containing profile metadata (shapes, columns, types, stats).
    """
    return profile_df(df)

def ask_question(
    profile: dict[str, Any],
    suggestions: list[Suggestion],
    history: list[dict[str, str]],
    question: str,
    model: str = "llama3.2"
) -> str:
    """
    Ask a question about the dataset/suggestions in a chat context.

    Args:
        profile: Dataset profile (from get_profile).
        suggestions: List of Suggestion objects (from get_suggestions).
        history: List of chat messages (role/content dicts).
        question: The user's question.
        model: Ollama model name.

    Returns:
        The LLM's answer as a string.
    """
    suggestions_jsonable = [s.model_dump() for s in suggestions]
    
    chat_prompt = build_chat_prompt(
        profile=profile,
        suggestions_jsonable=suggestions_jsonable,
        history=history,
        user_message=question
    )

    ans = generate_text(
        model=model, 
        prompt=chat_prompt, 
        temperature=0.4, 
        num_predict=900
    )
    return ans.strip()

def get_suggestions(
    df: pd.DataFrame,
    model: str = "llama3.2",
    task: str = "unspecified",
    target: Optional[str] = None,
    exclude_columns: Optional[list[str]] = None,
) -> list[Suggestion]:
    """
    Analyze a DataFrame and generate feature engineering suggestions using an LLM.

    Args:
        df: Input pandas DataFrame.
        model: Ollama model name (default: "llama3.2").
        task: ML task type ("classification", "regression", "unspecified").
        target: Target column name (optional).
        exclude_columns: List of columns to exclude from suggestions.

    Returns:
        List of Suggestion objects.
    """
    # 1. Profile the DataFrame
    prof = profile_df(df)

    # 2. Build the prompt
    suggest_prompt = build_suggest_prompt(
        prof, 
        task=task, 
        target=target, 
        exclude_columns=exclude_columns
    )

    # 3. Call LLM
    raw = generate_text(
        model=model, 
        prompt=suggest_prompt, 
        temperature=0.3, 
        num_predict=2500
    )

    # 4. Parse response
    suggestions = parse_suggestions(raw)
    return suggestions
