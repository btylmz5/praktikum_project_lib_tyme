from __future__ import annotations
import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
    # Try common robust settings
    read_attempts = [
        dict(sep=None, engine="python"),
        dict(sep=";"),
        dict(sep=","),
    ]

    last_err = None
    for opts in read_attempts:
        try:
            return pd.read_csv(
                path,
                **opts,
                encoding="utf-8",
                on_bad_lines="skip",
                low_memory=False,
            )
        except Exception as e:
            last_err = e

        # encoding fallback
        try:
            return pd.read_csv(
                path,
                **opts,
                encoding="latin1",
                on_bad_lines="skip",
                low_memory=False,
            )
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Failed to read CSV: {path}. Last error: {last_err}")
