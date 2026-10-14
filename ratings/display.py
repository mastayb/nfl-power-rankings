"""Display helpers shared by the Reddit table and the graphics."""

from __future__ import annotations

import pandas as pd


def display_ranks(ratings: pd.Series, decimals: int = 1) -> pd.Series:
    """Rank labels with ties: teams whose ratings round to the same displayed value
    share a rank ("T-1", "T-1", "3", ...), standard competition ranking."""
    shown = ratings.round(decimals)
    rank = shown.rank(ascending=False, method="min").astype(int)
    tied = shown.duplicated(keep=False)
    return pd.Series([f"T-{r}" if t else str(r) for r, t in zip(rank, tied)], index=ratings.index)
