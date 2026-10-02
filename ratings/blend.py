"""Measured market + résumé blend (coefficients from scripts/calibrate_blend.py).

    blended rating = b_market * market rating + b_resume * résumé rating
    blended HFA    = b_market * market HFA + b_resume * résumé HFA + a

b_market < 1 is regression to the mean over the rest of the season; it
scales every team equally, so the order changes only through b_resume.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from ratings.backtest import resume_ratings

BLEND_PATH = Path(__file__).resolve().parent.parent / "config" / "resume_blend.yaml"


@dataclass
class Blend:
    cutoff_week: int
    a: float
    b_market: float
    b_resume: float
    half_life: float
    cap: float


def completed_week(schedule: pd.DataFrame) -> int:
    """Last week in which every game has a result (0 before week 1 ends)."""
    done = schedule.groupby("week").result.apply(lambda r: r.notna().all())
    weeks = done[done].index
    return int(weeks.max()) if len(weeks) else 0


def load_blend(cutoff_week: int, path: Path = BLEND_PATH) -> Blend:
    """Coefficients for `cutoff_week`, clamped to the calibrated range."""
    cfg = yaml.safe_load(open(path))
    rows = {r["cutoff_week"]: r for r in cfg["by_cutoff"]}
    k = min(max(cutoff_week, min(rows)), max(rows))
    r = rows[k]
    return Blend(k, r["a"], r["b_market"], r["b_resume"], cfg["half_life_weeks"], cfg["cap"])


def blend_ratings(market: pd.Series, market_hfa: float, schedule: pd.DataFrame,
                  blend: Blend) -> tuple[pd.DataFrame, float]:
    """Per-team market, résumé and blended ratings, plus the blended HFA."""
    res, res_hfa = resume_ratings(schedule, blend.cutoff_week, blend.half_life, blend.cap)
    out = pd.DataFrame({"market": market, "resume": res.reindex(market.index).fillna(0.0)})
    out["blend"] = blend.b_market * out.market + blend.b_resume * out.resume
    out["blend"] -= out.blend.mean()
    hfa = blend.b_market * market_hfa + blend.b_resume * res_hfa + blend.a
    return out, hfa
