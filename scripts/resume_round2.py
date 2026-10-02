"""Round 2: does any feature add to market + margin? Best combination? Half-life?

Run: python scripts/resume_round2.py

All scores are out-of-sample RMSE of rest-of-season margins (points), leave-
one-season-out over 1999-2025, with 90% intervals from resampling held-out
seasons, plus the random-weeks check. Negative delta = better.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ratings.backtest import load_history  # noqa: E402
from ratings.evaluate import build_panel, compare  # noqa: E402
from ratings.features import add_deserved, season_features  # noqa: E402
from ratings.pbp_cache import load_pbp_cached  # noqa: E402

SEASONS = range(1999, 2026)
CUTOFFS = [3, 4, 6, 8]
VARIANTS = {
    "margin": ("margin", 21.0),
    "margin_neutral": ("margin_neutral", 21.0),
    "deserved": ("deserved", None),
    "pgwe": ("pgwe", None),
    "net_epa": ("net_epa", None),
    "net_epa_pass": ("net_epa_pass", None),
    "net_epa_rush": ("net_epa_rush", None),
    "net_sr": ("net_sr", None),
    "avg_wp": ("avg_wp", None),
}
COMBOS = {
    "margin (current)": ["margin"],
    "margin_neutral": ["margin_neutral"],
    "deserved": ["deserved"],
    "margin + deserved": ["margin", "deserved"],
    "margin_neutral + deserved": ["margin_neutral", "deserved"],
    "margin + pass + rush EPA": ["margin", "net_epa_pass", "net_epa_rush"],
    "margin + deserved + avg_wp": ["margin", "deserved", "avg_wp"],
    "all features": list(VARIANTS),
}
HALF_LIVES = [1.5, 3.0, 5.0, 8.0, 1000.0]
HL_VARIANTS = {k: VARIANTS[k] for k in ("margin", "deserved")}


def fmt(r: dict) -> str:
    lo, hi = r["loso_delta_90"]
    return f"{r['loso_delta']:+.4f} [{lo:+.4f}, {hi:+.4f}]  rw {r['rw_delta']:+.4f}"


def p(names):
    return [f"pred_{n}" for n in names]


def main() -> None:
    hist = load_history(SEASONS)
    feats, info = add_deserved(season_features(SEASONS, load_pbp_cached))
    print(f"deserved-margin model: {info}")
    cols = ["game_id"] + [c for c, _ in VARIANTS.values() if c != "margin"]
    hist = hist.merge(feats[cols], on="game_id", how="left")
    hist["margin"] = hist.result

    rows = []
    for k in CUTOFFS:
        panel = build_panel(hist, k, VARIANTS)
        print(f"\n=== cutoff week {k} ({len(panel)} later games) ===")
        print("A. Each feature added on top of market + margin:")
        for name in VARIANTS:
            if name == "margin":
                continue
            r = compare(panel, ["market_pred", "pred_margin"], [f"pred_{name}"])
            rows.append({"part": "A", "cutoff": k, "name": name, **r})
            print(f"   + {name:16} {fmt(r)}")

        print("B. Combinations vs market alone (and vs market + margin):")
        base = compare(panel, ["market_pred"], p(["margin"]))
        for label, names in COMBOS.items():
            r = compare(panel, ["market_pred"], p(names))
            rel = r["loso_rmse"] - base["loso_rmse"]
            rows.append({"part": "B", "cutoff": k, "name": label, **r})
            print(f"   {label:28} {fmt(r)}   vs current {rel:+.4f}")

        print("C. Résumé half-life (weeks), market + résumé vs market alone:")
        for hl in HALF_LIVES:
            hp = build_panel(hist, k, HL_VARIANTS, resume_half_life=hl)
            for name in HL_VARIANTS:
                r = compare(hp, ["market_pred"], [f"pred_{name}"])
                rows.append({"part": "C", "cutoff": k, "name": f"{name} hl={hl:g}", **r})
            line = "  ".join(f"{n} {r2['loso_delta']:+.4f}" for n, r2 in
                             [(n, compare(hp, ['market_pred'], [f'pred_{n}'])) for n in HL_VARIANTS])
            print(f"   half-life {'none' if hl >= 1000 else f'{hl:g}':>4}: {line}")

    out = ROOT / "data" / "resume_round2.csv"
    pd.DataFrame(rows).drop(columns=["coef"]).to_csv(out, index=False)
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
