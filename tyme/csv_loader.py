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
    best_df = None

    for opts in read_attempts:
        try:
            df = pd.read_csv(
                path,
                **opts,
                encoding="utf-8",
                on_bad_lines="skip",
                low_memory=False,
            )
            # If we found multiple columns, it's likely the correct parsing
            if df.shape[1] > 1:
                return df
            # Otherwise keep it as a fallback
            if best_df is None:
                best_df = df
        except Exception as e:
            last_err = e

    # Check fallback
    if best_df is not None:
        return best_df

    # encoding fallback (only if we didn't return above)
    try:
        df = pd.read_csv(
            path,
            sep=None,
            engine="python",
            encoding="latin1",
            on_bad_lines="skip",
            low_memory=False,
        )
        return df
    except Exception as e:
        last_err = e

    raise RuntimeError(f"Failed to read CSV: {path}. Last error: {last_err}")
