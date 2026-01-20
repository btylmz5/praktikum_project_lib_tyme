from __future__ import annotations
import pandas as pd
from typing import Optional

from .profile import profile_df
from .prompts import build_suggest_prompt
from .ollama_client import generate_text
from .parsing import parse_suggestions, Suggestion

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
        num_predict=1100
    )

    # 4. Parse response
    suggestions = parse_suggestions(raw)
    return suggestions
