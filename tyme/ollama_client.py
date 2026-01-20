from __future__ import annotations
from typing import Optional
import ollama


def generate_text(
    model: str,
    prompt: str,
    temperature: float = 0.3,
    num_predict: int = 900,
) -> str:
    resp = ollama.generate(
        model=model,
        prompt=prompt,
        options={
            "temperature": temperature,
            "num_predict": num_predict,
        },
    )
    # ollama python lib typically returns {'response': '...'}
    return resp.get("response", "")
