"""Season-long market + résumé blend (weights from scripts/calibrate_blend.py).

    blended rating = b_market * market + b_margin * résumé + b_neutral * luck-neutral résumé
    blended HFA    = the same weights on each part's HFA, plus a

All weights follow smooth curves in the cutoff week: early on the market
dominates and luck counts as real; the résumé's share grows through the
season and shifts to the luck-neutral version. b_market < 1 is regression to
the mean over the rest of the season and scales every team equally.
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
    cutoff_week: int       # completed week the résumé runs through
    curve_week: int        # week the weights were read at (clamped to the fitted range)
    a: float
    b_market: float
    b_margin: float
    b_neutral: float
    half_life: float
    cap: float

    @property
    def resume_weight(self) -> float:
        return self.b_margin + self.b_neutral


def completed_week(schedule: pd.DataFrame) -> int:
    """Last week in which every game has a result (0 before week 1 ends)."""
    done = schedule.groupby("week").result.apply(lambda r: r.notna().all())
    weeks = done[done].index
    return int(weeks.max()) if len(weeks) else 0


def load_blend(cutoff_week: int, path: Path = BLEND_PATH) -> Blend:
    """Weights for `cutoff_week`, clamped to the calibrated range."""
    cfg = yaml.safe_load(open(path))
    rows = {r["cutoff_week"]: r for r in cfg["by_cutoff"]}
    k = min(max(cutoff_week, min(rows)), max(rows))
    r = rows[k]
    return Blend(cutoff_week, k, r["a"], r["b_market"], r["b_margin"], r["b_neutral"],
                 cfg["half_life_weeks"], cfg["cap"])


def blend_ratings(market: pd.Series, market_hfa: float, schedule: pd.DataFrame,
                  blend: Blend) -> tuple[pd.DataFrame, float]:
    """Per-team market, résumé, luck-neutral résumé and blended ratings + blended HFA.

    `schedule` needs `result` and `margin_neutral` (result minus luck points)
    for played games.
    """
    k = blend.cutoff_week
    res, res_hfa = resume_ratings(schedule, k, blend.half_life, blend.cap, col="result")
    neu, neu_hfa = resume_ratings(schedule, k, blend.half_life, blend.cap, col="margin_neutral")
    out = pd.DataFrame({"market": market,
                        "resume": res.reindex(market.index).fillna(0.0),
                        "resume_neutral": neu.reindex(market.index).fillna(0.0)})
    out["blend"] = (blend.b_market * out.market + blend.b_margin * out.resume
                    + blend.b_neutral * out.resume_neutral)
    out["blend"] -= out.blend.mean()
    hfa = (blend.a + blend.b_market * market_hfa + blend.b_margin * res_hfa
           + blend.b_neutral * neu_hfa)
    return out, hfa
