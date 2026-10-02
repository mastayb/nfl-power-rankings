"""Historical tests of market vs résumé, ATS momentum, and playing down.

All inputs are nflverse schedules (closing spread_line, final result) for
regular-season games. "ATS residual" is result - spread_line: how much the
home team beat the closing spread by. From a team's perspective it is
signed so positive means the team beat the spread.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ratings.linear import decay_weight

BREAKEVEN = 110 / 210  # ATS win rate needed at -110


def load_history(seasons) -> pd.DataFrame:
    import nflreadpy as nfl

    s = nfl.load_schedules(seasons=list(seasons)).to_pandas()
    s = s[(s.game_type == "REG") & s.result.notna() & s.spread_line.notna()].copy()
    s["neutral"] = s.location.eq("Neutral")
    s["ats"] = s.result - s.spread_line
    return s.reset_index(drop=True)


def ridge_ratings(games: pd.DataFrame, y: np.ndarray, weights: np.ndarray,
                  ridge: float = 1e-3) -> tuple[pd.Series, float]:
    """Ratings + HFA by weighted ridge LS; the tiny ridge keeps early-season
    disconnected schedules solvable and centres ratings at ~0."""
    teams = sorted(set(games.home_team) | set(games.away_team))
    idx = {t: i for i, t in enumerate(teams)}
    k = len(teams)
    X = np.zeros((len(games), k + 1))
    X[np.arange(len(games)), games.home_team.map(idx)] = 1
    X[np.arange(len(games)), games.away_team.map(idx)] = -1
    X[:, k] = ~games.neutral.to_numpy(bool)
    sw = np.sqrt(weights)
    A = np.vstack([X * sw[:, None], np.sqrt(ridge) * np.eye(k + 1)[:k]])
    b = np.concatenate([y * sw, np.zeros(k)])
    beta, *_ = np.linalg.lstsq(A, b, rcond=None)
    r = pd.Series(beta[:k], index=teams)
    return r - r.mean(), float(beta[k])


def resume_ratings(season_games: pd.DataFrame, k: int, half_life: float = 3.0,
                   cap: float = 21.0, col: str = "result") -> tuple[pd.Series, float]:
    """Schedule-adjusted résumé through week k: per-game `col` (home minus
    away margin) capped at +-cap, decayed by age in weeks as of week k+1."""
    res = season_games[(season_games.week <= k) & season_games[col].notna()]
    return ridge_ratings(res, res[col].clip(-cap, cap).to_numpy(float),
                         decay_weight((k + 1 - res.week) * 7, half_life))


def cutoff_ratings(season_games: pd.DataFrame, k: int, half_life: float = 3.0,
                   cap: float = 21.0) -> pd.DataFrame:
    """Market and résumé ratings as of just before week k+1 kicks off.

    market: closing lines of weeks 1..k+1 (week k+1 lines are pre-game info).
    résumé: results of weeks 1..k, margins capped at +-cap.
    Both decayed by age in weeks with the same half-life (recency bias).
    """
    lines = season_games[season_games.week <= k + 1]
    m, m_hfa = ridge_ratings(lines, lines.spread_line.to_numpy(float),
                             decay_weight((k + 1 - lines.week) * 7, half_life))
    s, s_hfa = resume_ratings(season_games, k, half_life, cap)
    out = pd.DataFrame({"market": m, "resume": s}).fillna(0.0)
    out["gap"] = out.resume - out.market
    out.attrs["market_hfa"], out.attrs["resume_hfa"] = m_hfa, s_hfa
    return out


def cutoff_panel(hist: pd.DataFrame, k: int, **kw) -> pd.DataFrame:
    """Every game after week k with both teams' cutoff ratings attached."""
    rows = []
    for season, g in hist.groupby("season"):
        rt = cutoff_ratings(g, k, **kw)
        later = g[g.week > k].copy()
        for side in ("home", "away"):
            t = later[f"{side}_team"]
            for col in ("market", "resume", "gap"):
                later[f"{col}_{side}"] = t.map(rt[col]).to_numpy()
        nn = ~later.neutral.to_numpy(bool)
        later["market_pred"] = later.market_home - later.market_away + rt.attrs["market_hfa"] * nn
        later["resume_pred"] = later.resume_home - later.resume_away + rt.attrs["resume_hfa"] * nn
        later["gap_diff"] = later.gap_home - later.gap_away
        later["weeks_ahead"] = later.week - k
        rows.append(later)
    return pd.concat(rows, ignore_index=True)


def team_games(hist: pd.DataFrame) -> pd.DataFrame:
    """One row per team per game, from that team's perspective."""
    parts = []
    for side, sign in (("home", 1), ("away", -1)):
        opp = "away" if side == "home" else "home"
        parts.append(pd.DataFrame({
            "season": hist.season, "week": hist.week, "game_id": hist.game_id,
            "team": hist[f"{side}_team"], "opp": hist[f"{opp}_team"], "is_home": side == "home",
            "spread": sign * hist.spread_line,   # >0: team favoured by that many
            "margin": sign * hist.result,
            "ats": sign * hist.ats,              # >0: team beat the spread
        }))
    return pd.concat(parts, ignore_index=True).sort_values(["season", "team", "week"])


# --- statistics ------------------------------------------------------------------------

def ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(X, y, rcond=None)[0]


def season_bootstrap(df: pd.DataFrame, fn, n: int = 1000, seed: int = 0) -> np.ndarray:
    """Resample whole seasons with replacement (games within a season share
    ratings, so they are not independent); return fn() per replicate."""
    rng = np.random.default_rng(seed)
    groups = [g for _, g in df.groupby("season")]
    out = []
    for _ in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        out.append(fn(pd.concat([groups[i] for i in pick], ignore_index=True)))
    return np.array(out)


def cover_rate(ats: pd.Series) -> tuple[float, int]:
    """ATS win rate excluding pushes, and the number of non-push games."""
    decided = ats[ats != 0]
    return float((decided > 0).mean()) if len(decided) else np.nan, len(decided)


def blend_coefficients(hist: pd.DataFrame, cutoffs, n_boot: int = 300) -> pd.DataFrame:
    """Regress later results on market and résumé predictions at each cutoff:

        result = a + b_market * market_pred + b_resume * resume_pred

    with 90% season-bootstrap intervals."""
    def fit(d):
        X = np.column_stack([np.ones(len(d)), d.market_pred, d.resume_pred])
        return ols(X, d.result.to_numpy(float))

    rows = []
    for k in cutoffs:
        p = cutoff_panel(hist, k)
        b = fit(p)
        lo, hi = np.percentile(season_bootstrap(p, fit, n_boot), [5, 95], axis=0)
        rows.append({"cutoff_week": k, "n_games": len(p), "a": b[0],
                     "b_market": b[1], "b_market_90": [lo[1], hi[1]],
                     "b_resume": b[2], "b_resume_90": [lo[2], hi[2]]})
    return pd.DataFrame(rows)
