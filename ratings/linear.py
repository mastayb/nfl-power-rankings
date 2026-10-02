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
