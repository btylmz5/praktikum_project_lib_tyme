from __future__ import annotations
import json
from typing import Literal, List

from pydantic import BaseModel, Field, ValidationError


Risk = Literal["none", "leakage", "overfit", "data_quality", "unknown"]
FType = Literal["numeric", "categorical", "datetime", "text", "unknown"]


class Suggestion(BaseModel):
    name: str = Field(..., min_length=1)
    depends_on: List[str] = Field(default_factory=list)
    how: str = Field(..., min_length=1)
    why: str = Field(..., min_length=1)
    feature_type: FType = "unknown"
    risk: Risk = "unknown"


def _extract_json_array(text: str) -> str:
    """
    Robustly extracts the first valid JSON array from text by counting brackets.
    """
    start_idx = text.find("[")
    if start_idx == -1:
         raise ValueError("Could not find a JSON array start '[' in the model output.")
    
    # Bracket counting to find the matching closing bracket
    count = 0
    for i in range(start_idx, len(text)):
        char = text[i]
        if char == "[":
            count += 1
        elif char == "]":
            count -= 1
            if count == 0:
                # Found the matching closing bracket
                return text[start_idx : i + 1]
    
    raise ValueError("Found start '[' but no matching closing ']' for JSON array.")


def parse_suggestions(raw: str) -> list[Suggestion]:
    candidate = _extract_json_array(raw)
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON decode failed: {e}") from e

    if not isinstance(data, list):
        raise ValueError("Expected a JSON array (list) at top level.")

    out: list[Suggestion] = []
    errors = []
    for i, item in enumerate(data):
        try:
            out.append(Suggestion.model_validate(item))
        except ValidationError as e:
            errors.append((i, str(e)))

    if not out:
        raise ValueError(f"All suggestions failed validation. Example errors: {errors[:2]}")
    return out
