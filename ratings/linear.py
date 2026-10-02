"""Stage B (linear part): weighted least squares for team ratings.

    y_g = r_home - r_away + H * (1 - neutral_g) + e_g,   sum_t r_t = 0
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MarginFit:
    ratings: pd.Series      # r_t, sums to 0
    se: pd.Series           # standard error of each r_t
    hfa: float
    hfa_se: float
    resid_sd: float         # weighted residual SD (points)
    n_games: int


def fit_margins(games: pd.DataFrame, y: str, teams: list[str],
                weights: np.ndarray | None = None) -> MarginFit:
    """WLS fit of home margins `games[y]` on home/away indicators plus HFA.

    `games` needs home_team, away_team, neutral. The sum-to-zero constraint is
    imposed by dropping the last team and setting it to -sum(others).
    """
    idx = {t: i for i, t in enumerate(teams)}
    n, k = len(games), len(teams)
    X = np.zeros((n, k + 1))
    for row, (h, a) in enumerate(zip(games.home_team, games.away_team)):
        X[row, idx[h]] += 1
        X[row, idx[a]] -= 1
    X[:, k] = ~games.neutral.to_numpy(bool)

    # r_last = -sum(r_others): reparametrise to k-1 free ratings + HFA.
    C = np.zeros((k + 1, k))
    C[: k - 1, : k - 1] = np.eye(k - 1)
    C[k - 1, : k - 1] = -1
    C[k, k - 1] = 1
    Z = X @ C

    w = np.ones(n) if weights is None else np.asarray(weights, float)
    sw = np.sqrt(w)
    yv = games[y].to_numpy(float)
    beta, *_ = np.linalg.lstsq(Z * sw[:, None], yv * sw, rcond=None)
    if np.linalg.matrix_rank(Z * sw[:, None]) < k:
        raise ValueError("ratings not identified: game graph is disconnected")

    resid = yv - Z @ beta
    dof = max(n - k, 1)
    s2 = float(np.sum(w * resid**2) / dof)
    cov_z = s2 * np.linalg.inv((Z * w[:, None]).T @ Z)
    cov = C @ cov_z @ C.T
    full = C @ beta
    return MarginFit(
        ratings=pd.Series(full[:k], index=teams),
        se=pd.Series(np.sqrt(np.diag(cov)[:k]), index=teams),
        hfa=float(full[k]),
        hfa_se=float(np.sqrt(cov[k, k])),
        resid_sd=float(np.sqrt(s2)),
        n_games=n,
    )


def predict_margin(fit: MarginFit, games: pd.DataFrame) -> np.ndarray:
    r = fit.ratings
    return (r.loc[games.home_team].to_numpy() - r.loc[games.away_team].to_numpy()
            + fit.hfa * ~games.neutral.to_numpy(bool))


# --- Stage B inputs: current Kalshi lines + decayed past closing lines --------

def decay_weight(age_days, half_life_weeks: float | None) -> np.ndarray:
    """0.5 ** (age / half-life); None means no decay."""
    age = np.maximum(np.asarray(age_days, float), 0.0)
    if half_life_weeks is None:
        return np.ones_like(age)
    return 0.5 ** (age / (7.0 * half_life_weeks))


def line_observations(stage_a: pd.DataFrame, games: pd.DataFrame, schedule: pd.DataFrame,
                      model, sigma: float, asof, half_life_weeks: float | None = 3.0,
                      results_weight: float = 0.0) -> pd.DataFrame:
    """Observations of expected home margin for the Stage B fit.

    - kalshi: Stage A `mean` for every game with open Kalshi markets, weight 1.
    - close:  played games' closing sportsbook spread (nflverse spread_line, a
              median) converted to an expected margin with the same margin
              model, decayed by age.
    - result: played games' final margin, weight `results_weight` x decay
              (0 = off). Lines are ~2 pts noisy and results ~sigma, so the
              likelihood-consistent value is about (2 / sigma)**2.
    """
    from ratings.ladder import mean_from_line

    asof = pd.Timestamp(asof).tz_localize(None).normalize()
    k = stage_a.merge(games[["game_id", "home_team", "away_team", "gameday"]], on="game_id")
    obs = [pd.DataFrame({
        "game_id": k.game_id, "home_team": k.home_team, "away_team": k.away_team,
        "neutral": k.neutral.astype(bool), "y": k["mean"], "source": "kalshi",
        "weight": 1.0, "gameday": pd.to_datetime(k.gameday),
    })]

    played = schedule[schedule.result.notna() & schedule.spread_line.notna()
                      & ~schedule.game_id.isin(k.game_id)].copy()
    played["gameday"] = pd.to_datetime(played.gameday)
    decay = decay_weight((asof - played.gameday).dt.days, half_life_weeks)
    base = {"game_id": played.game_id, "home_team": played.home_team,
            "away_team": played.away_team, "neutral": played.neutral.astype(bool),
            "gameday": played.gameday}
    lines = {l: mean_from_line(model, l, sigma) for l in played.spread_line.unique()}
    obs.append(pd.DataFrame(base | {"y": played.spread_line.map(lines), "source": "close",
                                    "weight": decay}))
    if results_weight > 0:
        obs.append(pd.DataFrame(base | {"y": played.result.astype(float), "source": "result",
                                        "weight": results_weight * decay}))
    return pd.concat(obs, ignore_index=True)
