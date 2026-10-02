"""Fit the season-long market + résumé blend curve and pin it in config.

Run: python scripts/calibrate_blend.py
Writes config/resume_blend.yaml, read by ratings/blend.py.

Rest-of-season margin = a(k) + b_market(k) * market + b_margin(k) * résumé
                        + b_neutral(k) * luck-neutral résumé
with every weight quadratic in the cutoff week k, fitted jointly over weeks
2-12 of 1999-2025 (see scripts/resume_season_curve.py for the out-of-sample
comparison that chose this model).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ratings.backtest import load_history  # noqa: E402
from ratings.evaluate import CURVE_WEEKS, build_panel, curve_design, curve_weights  # noqa: E402
from ratings.features import season_features  # noqa: E402
from ratings.pbp_cache import load_pbp_cached  # noqa: E402

SEASONS = range(1999, 2026)
WEEKS = list(range(CURVE_WEEKS[0], CURVE_WEEKS[1] + 1))
HALF_LIFE, CAP = 3.0, 21.0
VARIANTS = {"margin": ("margin", CAP), "neutral": ("margin_neutral", CAP)}
PREDS = ["market_pred", "pred_margin", "pred_neutral"]
NAMES = ["a", "b_market", "b_margin", "b_neutral"]
N_BOOT = 300
OUT = ROOT / "config" / "resume_blend.yaml"


def main() -> None:
    hist = load_history(SEASONS)
    feats = season_features(SEASONS, load_pbp_cached)
    hist = hist.merge(feats[["game_id", "margin_neutral"]], on="game_id", how="left")
    hist["margin"] = hist.result
    stack = pd.concat([build_panel(hist, k, VARIANTS, half_life=HALF_LIFE).assign(k=k)
                       for k in WEEKS], ignore_index=True)
    X, y = curve_design(stack, PREDS), stack.result.to_numpy(float)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    w = curve_weights(beta, WEEKS, NAMES)

    rng = np.random.default_rng(0)
    seasons = stack.season.to_numpy()
    uniq = np.unique(seasons)
    rows_by = {s: np.flatnonzero(seasons == s) for s in uniq}
    draws = []
    for _ in range(N_BOOT):
        idx = np.concatenate([rows_by[s] for s in rng.choice(uniq, len(uniq))])
        b = np.linalg.lstsq(X[idx], y[idx], rcond=None)[0]
        draws.append(curve_weights(b, WEEKS, NAMES).to_numpy())
    lo, hi = np.percentile(np.array(draws), [5, 95], axis=0)
    print(w.round(3).to_string())

    rows = []
    for i, k in enumerate(WEEKS):
        r = {"cutoff_week": k}
        for j, n in enumerate(NAMES):
            r[n] = round(float(w.iloc[i, j]), 4)
            r[f"{n}_90"] = [round(float(lo[i, j]), 4), round(float(hi[i, j]), 4)]
        rows.append(r)
    doc = {
        "description": (
            "Season-long blend: rest-of-season margin = a + b_market * market + b_margin * résumé"
            " + b_neutral * luck-neutral résumé, each weight quadratic in the cutoff week k and"
            " fitted jointly over weeks 2-12. market: closing lines through week k+1; résumé:"
            " results through week k; luck-neutral résumé: results minus fumble + INT luck"
            " points; margins capped +-21, 3-week half-life. Intervals: 90% season bootstrap."
            " Weeks outside 2-12 use the nearest end."),
        "seasons": f"{SEASONS.start}-{SEASONS.stop - 1}",
        "half_life_weeks": HALF_LIFE,
        "cap": CAP,
        "by_cutoff": rows,
    }
    with open(OUT, "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False, width=100, allow_unicode=True)
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
