from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
import warnings
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
)


def _infer_col_type(s: pd.Series) -> str:
    if is_bool_dtype(s):
        return "categorical"
    if is_datetime64_any_dtype(s):
        return "datetime"
    if is_numeric_dtype(s):
        return "numeric"

    # object/string: decide categorical vs text vs datetime-ish
    s2 = s.dropna().astype(str)
    if len(s2) == 0:
        return "unknown"

    # try datetime parse on a small sample
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        sample = s2.sample(min(50, len(s2)), random_state=0)
        parsed = pd.to_datetime(sample, errors="coerce", utc=False)
    if parsed.notna().mean() > 0.9:
        return "datetime"

    nunique = s2.nunique(dropna=True)
    unique_ratio = nunique / max(len(s2), 1)
    avg_len = s2.str.len().mean()

    # heuristics
    if unique_ratio < 0.2:
        return "categorical"
    if avg_len >= 30 and unique_ratio > 0.5:
        return "text"
    return "categorical"


def profile_df(df: pd.DataFrame, max_top_values: int = 3) -> dict[str, Any]:
    n_rows, n_cols = df.shape
    cols = []

    for col in df.columns:
        s = df[col]
        missing = float(s.isna().mean())
        inferred = _infer_col_type(s)

        entry: dict[str, Any] = {
            "name": str(col),
            "inferred_type": inferred,
            "dtype": str(s.dtype),
            "missing_ratio": round(missing, 4),
        }

        non_null = s.dropna()
        entry["n_unique"] = int(non_null.nunique()) if len(non_null) else 0

        # sample values (safe string)
        if len(non_null):
            sample_vals = non_null.sample(min(3, len(non_null)), random_state=0).astype(str).tolist()
        else:
            sample_vals = []
        entry["sample_values"] = sample_vals

        if inferred == "numeric" and len(non_null):
            nn = pd.to_numeric(non_null, errors="coerce").dropna()
            if len(nn):
                entry["stats"] = {
                    "min": float(np.nanmin(nn)),
                    "max": float(np.nanmax(nn)),
                    "mean": float(np.nanmean(nn)),
                    "std": float(np.nanstd(nn)),
                }

        if inferred == "categorical" and len(non_null):
            vc = non_null.astype(str).value_counts(dropna=True).head(max_top_values)
            entry["top_values"] = [{"value": k, "count": int(v)} for k, v in vc.items()]

        cols.append(entry)

    return {
        "shape": {"rows": int(n_rows), "cols": int(n_cols)},
        "columns": cols,
    }
