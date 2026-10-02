"""Cross-validated comparison of résumé variants (see scripts/resume_round1.py).

For each cutoff week k and each season, every résumé variant is turned into
team ratings from weeks 1..k (ridge, 3-week half-life, same as the baseline
résumé); later games then get a prediction from each variant. Linear models
on top of those predictions are scored out of sample two ways:

  LOSO          leave one season out: fit on 26 seasons, score the 27th.
  random weeks  hold out a random 30% of (season, week) target units; repeat.

LOSO is the primary score. Random weeks shares team-seasons between train and
test, so it should look the same or slightly rosier; a big gap means leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ratings.backtest import ridge_ratings
from ratings.linear import decay_weight


def variant_ratings(season_games: pd.DataFrame, k: int, col: str, cap: float | None,
                    half_life: float = 3.0) -> tuple[pd.Series, float]:
    g = season_games[(season_games.week <= k) & season_games[col].notna()]
    y = g[col].to_numpy(float)
    if cap is not None:
        y = np.clip(y, -cap, cap)
    return ridge_ratings(g, y, decay_weight((k + 1 - g.week) * 7, half_life))


def build_panel(hist: pd.DataFrame, k: int, variants: dict[str, tuple[str, float | None]],
                half_life: float = 3.0) -> pd.DataFrame:
    """Later games (week > k) with market_pred and pred_<variant> columns."""
    out = []
    for season, g in hist.groupby("season"):
        lines = g[g.week <= k + 1]
        m, m_hfa = ridge_ratings(lines, lines.spread_line.to_numpy(float),
                                 decay_weight((k + 1 - lines.week) * 7, half_life))
        later = g[g.week > k].copy()
        nn = ~later.neutral.to_numpy(bool)

        def pred(r, hfa):
            return (later.home_team.map(r).fillna(0).to_numpy()
                    - later.away_team.map(r).fillna(0).to_numpy() + hfa * nn)

        later["market_pred"] = pred(m, m_hfa)
        for name, (col, cap) in variants.items():
            r, hfa = variant_ratings(g, k, col, cap, half_life)
            later[f"pred_{name}"] = pred(r, hfa)
        out.append(later)
    return pd.concat(out, ignore_index=True)


def _design(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return np.column_stack([np.ones(len(df))] + [df[c].to_numpy(float) for c in cols])


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, cols: list[str],
                 target: str = "result") -> np.ndarray:
    beta = np.linalg.lstsq(_design(train, cols), train[target].to_numpy(float), rcond=None)[0]
    return _design(test, cols) @ beta


def loso_sse(panel: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Out-of-sample sum of squared errors per held-out season."""
    sse = {}
    for s in panel.season.unique():
        tr, te = panel[panel.season != s], panel[panel.season == s]
        e = te.result.to_numpy(float) - _fit_predict(tr, te, cols)
        sse[s] = float(e @ e)
    return pd.Series(sse)


def random_weeks_rmse(panel: pd.DataFrame, cols: list[str], reps: int = 20,
                      frac: float = 0.3, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    units = panel[["season", "week"]].drop_duplicates().to_numpy()
    key = panel.season.to_numpy() * 100 + panel.week.to_numpy()
    out = []
    for _ in range(reps):
        pick = units[rng.random(len(units)) < frac]
        test_mask = np.isin(key, pick[:, 0] * 100 + pick[:, 1])
        tr, te = panel[~test_mask], panel[test_mask]
        e = te.result.to_numpy(float) - _fit_predict(tr, te, cols)
        out.append(np.sqrt(np.mean(e**2)))
    return np.array(out)


def compare(panel: pd.DataFrame, base: list[str], extra: list[str], n_boot: int = 1000,
            seed: int = 0) -> dict:
    """Out-of-sample RMSE of base vs base + extra, LOSO and random weeks.

    The LOSO interval resamples held-out seasons (90%); negative delta = better.
    """
    n = panel.groupby("season").size()
    s0, s1 = loso_sse(panel, base), loso_sse(panel, base + extra)
    rmse0 = np.sqrt(s0.sum() / n.sum())
    rmse1 = np.sqrt(s1.sum() / n.sum())
    rng = np.random.default_rng(seed)
    seasons = s0.index.to_numpy()
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(seasons, len(seasons))
        boots.append(np.sqrt(s1[pick].sum() / n[pick].sum()) - np.sqrt(s0[pick].sum() / n[pick].sum()))
    rw0, rw1 = random_weeks_rmse(panel, base), random_weeks_rmse(panel, base + extra)
    full = np.linalg.lstsq(_design(panel, base + extra), panel.result.to_numpy(float), rcond=None)[0]
    return {
        "loso_rmse_base": rmse0, "loso_rmse": rmse1, "loso_delta": rmse1 - rmse0,
        "loso_delta_90": tuple(np.percentile(boots, [5, 95])),
        "rw_delta": float(np.mean(rw1 - rw0)), "rw_delta_sd": float(np.std(rw1 - rw0)),
        "coef": full[1:],
    }
