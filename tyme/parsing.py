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
    Tries to pull the first JSON array from the model output.
    Works when the model wraps output in ```json ... ``` or adds commentary.
    """
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Could not find a JSON array in the model output.")
    return text[start : end + 1]


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
