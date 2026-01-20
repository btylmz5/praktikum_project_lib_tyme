from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional

from .parsing import Suggestion


@dataclass
class SessionState:
    csv_path: str
    model: str
    task: str  # "classification" | "regression" | "unspecified"
    target: Optional[str]

    profile: dict[str, Any]
    suggestions: list[Suggestion]

    history: list[dict[str, str]] = field(default_factory=list)  # [{"role":"user","content":"..."}, ...]
