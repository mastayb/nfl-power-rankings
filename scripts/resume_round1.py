"""Round 1: which single résumé feature adds the most to the market rating?

Run: python scripts/resume_round1.py

For each cutoff week and variant, out-of-sample RMSE of rest-of-season
margins (points) for:
  alone      result ~ variant prediction
  + market   result ~ market rating prediction + variant prediction (vs market alone)
  + close    result ~ each game's closing spread + variant prediction (vs spread alone)
Deltas are vs the base model; negative = the variant helps.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ratings.backtest import load_history  # noqa: E402
from ratings.evaluate import build_panel, compare  # noqa: E402
from ratings.features import season_features  # noqa: E402
from ratings.pbp_cache import load_pbp_cached  # noqa: E402

SEASONS = range(1999, 2026)
CUTOFFS = [3, 4, 6, 8]
VARIANTS = {                      # name -> (feature column, cap)
    "margin": ("margin", 21.0),   # current résumé (baseline)
    "margin_neutral": ("margin_neutral", 21.0),
    "net_epa": ("net_epa", None),
    "net_epa_live": ("net_epa_live", None),
    "net_sr": ("net_sr", None),
    "avg_wp": ("avg_wp", None),
}


def main() -> None:
    hist = load_history(SEASONS)
    feats = season_features(SEASONS, load_pbp_cached)
    cols = ["game_id"] + [c for c, _ in VARIANTS.values() if c != "margin"]
    hist = hist.merge(feats[cols], on="game_id", how="left")
    missing = hist[cols[1:]].isna().any(axis=1).sum()
    print(f"{len(hist)} games, {SEASONS.start}-{SEASONS.stop - 1}; games missing pbp features: {missing}")
    hist["margin"] = hist.result

    rows = []
    for k in CUTOFFS:
        panel = build_panel(hist, k, VARIANTS)
        for name in VARIANTS:
            p = f"pred_{name}"
            alone = compare(panel, [], [p])
            mkt = compare(panel, ["market_pred"], [p])
            cls = compare(panel, ["spread_line"], [p])
            rows.append({
                "cutoff": k, "variant": name, "n": len(panel),
                "alone_rmse": alone["loso_rmse"],
                "mkt_base": mkt["loso_rmse_base"], "mkt_delta": mkt["loso_delta"],
                "mkt_lo": mkt["loso_delta_90"][0], "mkt_hi": mkt["loso_delta_90"][1],
                "mkt_rw_delta": mkt["rw_delta"],
                "weight_pts": mkt["coef"][1] * panel[p].std(),  # pts per 1 SD of the variant
                "close_delta": cls["loso_delta"],
                "close_lo": cls["loso_delta_90"][0], "close_hi": cls["loso_delta_90"][1],
            })
        print(f"  cutoff {k} done")
    res = pd.DataFrame(rows)

    pd.set_option("display.width", 200)
    for k, g in res.groupby("cutoff"):
        print(f"\n=== cutoff week {k}: {g.n.iloc[0]} later games; market alone RMSE "
              f"{g.mkt_base.iloc[0]:.3f} ===")
        print(f"{'variant':16} {'alone':>7}   {'+market: delta [90% LOSO]':>34} {'rand-wk':>8}"
              f" {'pts/SD':>7}   {'+closing: delta [90% LOSO]':>34}")
        for r in g.sort_values("mkt_delta").itertuples():
            print(f"{r.variant:16} {r.alone_rmse:7.3f}   {r.mkt_delta:+.4f} [{r.mkt_lo:+.4f}, {r.mkt_hi:+.4f}]"
                  f"     {r.mkt_rw_delta:+.4f} {r.weight_pts:7.2f}   "
                  f"{r.close_delta:+.4f} [{r.close_lo:+.4f}, {r.close_hi:+.4f}]")
    out = ROOT / "data" / "resume_round1.csv"
    res.to_csv(out, index=False)
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
