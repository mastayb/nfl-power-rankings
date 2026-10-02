"""Season-long blend: coefficients as smooth functions of the cutoff week.

Run: python scripts/resume_season_curve.py

Stacks the cutoff panels for weeks 2..12 (each row = a later game, predicted
from ratings as of cutoff k) and compares, out of sample (leave one season out):

  per-week     separate market + margin fit at each k (current approach)
  switch@6     margin before week 6, luck-neutral margin from week 6
  curve        market, margin, luck-neutral margin, each with a weight that is
               quadratic in k, fitted jointly across all weeks
  curve+des    curve plus deserved margin (luck-neutral EPA)
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
from ratings.evaluate import build_panel  # noqa: E402
from ratings.features import add_deserved, season_features  # noqa: E402
from ratings.pbp_cache import load_pbp_cached  # noqa: E402

SEASONS = range(1999, 2026)
CUTOFFS = list(range(2, 13))
VARIANTS = {"margin": ("margin", 21.0), "neutral": ("margin_neutral", 21.0),
            "deserved": ("deserved", None)}
N_BOOT = 1000
OUT = ROOT / "data" / "resume_season_curve.csv"


def basis(k: np.ndarray) -> np.ndarray:
    """[1, t, t^2] with t = (k - 2) / 10, so t runs 0..1 over weeks 2..12."""
    t = (np.asarray(k, float) - 2) / 10
    return np.column_stack([np.ones_like(t), t, t**2])


def curve_design(df: pd.DataFrame, preds: list[str]) -> np.ndarray:
    B = basis(df.k.to_numpy())
    cols = [B]  # intercept curve
    for p in preds:
        cols.append(B * df[p].to_numpy(float)[:, None])
    return np.hstack(cols)


def per_week_design(df: pd.DataFrame, preds_by_k) -> np.ndarray:
    """Block design: a separate intercept + coefficients for every k."""
    blocks = []
    for k in CUTOFFS:
        on = (df.k.to_numpy() == k).astype(float)[:, None]
        preds = preds_by_k(k)
        X = np.column_stack([np.ones(len(df))] + [df[p].to_numpy(float) for p in preds])
        blocks.append(X * on)
    return np.hstack(blocks)


MODELS = {
    "market only (per-week)": lambda d: per_week_design(d, lambda k: ["market_pred"]),
    "per-week market+margin (current)": lambda d: per_week_design(d, lambda k: ["market_pred", "pred_margin"]),
    "switch@6 (per-week)": lambda d: per_week_design(
        d, lambda k: ["market_pred", "pred_margin" if k < 6 else "pred_neutral"]),
    "curve: market+margin": lambda d: curve_design(d, ["market_pred", "pred_margin"]),
    "curve: market+margin+neutral": lambda d: curve_design(d, ["market_pred", "pred_margin", "pred_neutral"]),
    "curve: +deserved": lambda d: curve_design(d, ["market_pred", "pred_margin", "pred_neutral", "pred_deserved"]),
}
BASE = "per-week market+margin (current)"


def loso_errors(stack: pd.DataFrame, design) -> np.ndarray:
    X = design(stack)
    y = stack.result.to_numpy(float)
    err = np.empty(len(stack))
    seasons = stack.season.to_numpy()
    for s in np.unique(seasons):
        te = seasons == s
        beta = np.linalg.lstsq(X[~te], y[~te], rcond=None)[0]
        err[te] = y[te] - X[te] @ beta
    return err


def main() -> None:
    hist = load_history(SEASONS)
    feats, _ = add_deserved(season_features(SEASONS, load_pbp_cached))
    hist = hist.merge(feats[["game_id", "margin_neutral", "deserved"]], on="game_id", how="left")
    hist["margin"] = hist.result
    stack = pd.concat([build_panel(hist, k, VARIANTS).assign(k=k) for k in CUTOFFS],
                      ignore_index=True)
    print(f"stacked {len(stack)} (game, cutoff) rows over weeks {CUTOFFS[0]}-{CUTOFFS[-1]}")

    errs = {name: loso_errors(stack, fn) for name, fn in MODELS.items()}
    sq = pd.DataFrame({n: e**2 for n, e in errs.items()}).assign(season=stack.season, k=stack.k)

    # overall + bootstrap over held-out seasons, paired vs the current approach
    rng = np.random.default_rng(0)
    by_s = sq.groupby("season")
    sums, counts = by_s.sum(), by_s.size()
    seasons = sums.index.to_numpy()
    print(f"\nOut-of-sample RMSE over all cutoffs (delta vs '{BASE}', 90% season bootstrap):")
    for n in MODELS:
        rm = np.sqrt(sums[n].sum() / counts.sum())
        boots = []
        for _ in range(N_BOOT):
            pk = rng.choice(seasons, len(seasons))
            boots.append(np.sqrt(sums[n].loc[pk].sum() / counts.loc[pk].sum())
                         - np.sqrt(sums[BASE].loc[pk].sum() / counts.loc[pk].sum()))
        lo, hi = np.percentile(boots, [5, 95])
        base = np.sqrt(sums[BASE].sum() / counts.sum())
        print(f"  {n:34} {rm:.4f}   {rm - base:+.4f} [{lo:+.4f}, {hi:+.4f}]")

    print("\nBy cutoff week (RMSE minus current per-week fit):")
    per_k = sq.groupby("k")[list(MODELS)].mean().pow(0.5)
    rel = per_k.sub(per_k[BASE], axis=0).drop(columns=BASE)
    print(rel.round(4).to_string())

    # fitted curves on all seasons, for the chosen curve model
    preds = ["market_pred", "pred_margin", "pred_neutral"]
    beta = np.linalg.lstsq(curve_design(stack, preds), stack.result.to_numpy(float), rcond=None)[0]
    B = basis(np.array(CUTOFFS))
    names = ["a"] + ["b_market", "b_margin", "b_neutral"]
    curves = pd.DataFrame({n: B @ beta[3 * i:3 * i + 3] for i, n in enumerate(names)},
                          index=pd.Index(CUTOFFS, name="week"))
    print("\nFitted weights, curve: market + margin + neutral (all seasons):")
    print(curves.round(3).to_string())
    curves.to_csv(OUT)
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
