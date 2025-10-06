from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import pandas as pd


def clean_names(values: Iterable[str]) -> List[str]:
    cleaned: List[str] = []
    for value in values:
        text = str(value).strip()
        if text:
            cleaned.append(text)
    return cleaned


def _load_dataframe_from_path(path: str | Path, sheet: Optional[str] = None) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file_path)
    if suffix in {".xlsx", ".xls"}:
        data = pd.read_excel(file_path, sheet_name=sheet)
        if isinstance(data, dict):
            # Prefer the first non-empty sheet
            for df in data.values():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    return df
            # Fallback to the first sheet even if empty
            return next(iter(data.values()))
        return data
    raise ValueError(f"Unsupported file type: {suffix}")


def load_names_from_file(path: str | Path, column: str = "name", sheet: Optional[str] = None) -> List[str]:
    """Backward-compatible: returns a list of deduplicated names."""
    frame = _load_dataframe_from_path(path, sheet=sheet)
    if column not in frame.columns:
        available = ", ".join(frame.columns.astype(str))
        raise ValueError(f"Column '{column}' not found. Available columns: {available}")
    cleaned_series = frame[column].astype(str).str.strip()
    # Deduplicate while preserving order
    unique_names = list(dict.fromkeys(clean_names(cleaned_series.dropna())))
    return unique_names


@dataclass
class FileSourceContext:
    frame: Optional[pd.DataFrame] = None
    src_column: Optional[str] = None

    @classmethod
    def from_path(cls, path: str | Path, column: str = "name", sheet: Optional[str] = None) -> "FileSourceContext":
        frame = _load_dataframe_from_path(path, sheet=sheet).copy()
        if column not in frame.columns:
            available = ", ".join(frame.columns.astype(str))
            raise ValueError(f"Column '{column}' not found. Available columns: {available}")
        frame["__clean_name__"] = frame[column].astype(str).str.strip()
        return cls(frame=frame, src_column=column)

    def unique_names(self) -> List[str]:
        if self.frame is None:
            return []
        return list(dict.fromkeys(clean_names(self.frame["__clean_name__"].dropna())))


def load_file_with_dedup(path: str | Path, column: str = "name", sheet: Optional[str] = None) -> List[str]:
    """Convenience wrapper that returns unique names without exposing context."""
    ctx = FileSourceContext.from_path(path, column=column, sheet=sheet)
    return ctx.unique_names()


def expand_results_to_source(results: Sequence[dict], ctx: Optional[FileSourceContext] = None) -> pd.DataFrame:
    """Expand per-unique-name results back to the original file rows using cleaned names.

    If no context is provided, returns a DataFrame built from results as-is.
    Expected that each result item contains a 'name' field.
    """
    if ctx is None or ctx.frame is None or ctx.src_column is None:
        return pd.DataFrame(results)
    base = ctx.frame.copy()
    if "__clean_name__" not in base.columns:
        base["__clean_name__"] = base[ctx.src_column].astype(str).str.strip()
    res_df = pd.DataFrame(results).copy()
    if res_df.empty:
        return base.drop(columns=["__clean_name__"]) if "__clean_name__" in base.columns else base
    if "name" not in res_df.columns:
        # nothing to merge on; just attach results loosely
        return base
    res_df["__clean_name__"] = res_df["name"].astype(str).str.strip()
    # Keep first in case of duplicates
    res_df = res_df.drop_duplicates(subset=["__clean_name__"], keep="first")
    merged = base.merge(res_df, on="__clean_name__", how="left")
    return merged.drop(columns=["__clean_name__"]) if "__clean_name__" in merged.columns else merged


def resolve_names(
    input_mode: str,
    single_name: str = "",
    name_list: Optional[Sequence[str]] = None,
    *,
    input_file: Optional[str | Path] = None,
    file_column: str = "name",
    file_sheet: Optional[str] = None,
) -> List[str]:
    mode = str(input_mode).lower().strip()
    if mode == "single":
        return clean_names([single_name])
    if mode == "list":
        return clean_names(name_list or [])
    if mode == "file":
        if not input_file:
            raise ValueError("input_file path is required for input_mode='file'")
        return load_file_with_dedup(input_file, column=file_column, sheet=file_sheet)
    raise ValueError("Unknown input_mode. Expected 'single', 'list', or 'file'.")
