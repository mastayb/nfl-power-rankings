"""Stage A: spread ladder + moneyline -> implied home margin mu_g, sigma_g.

Every market is turned into an observation q = G(x): the probability the home
margin M beats x, counting a push at x as half (Kalshi settles a tied
moneyline at $0.50 per side):

    G(x) = P(M > x) + 0.5 * P(M = x)

  home spread rung "home wins by over t"   -> x = t,  q = mid
  away spread rung "away wins by over t"   -> x = -t, q = 1 - mid
  home moneyline                           -> x = 0,  q = mid
  away moneyline                           -> x = 0,  q = 1 - mid

Two margin models share that interface:
  NormalModel     M ~ N(mu, sigma), continuous.
  KeyNumberModel  integer M with P(m) proportional to w(|m|) * N-bucket(m; mu,
                  sigma), where w is estimated from historical results vs
                  closing spreads, so 3, 7, 10, 14 carry their real mass and
                  ties are rare.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq, least_squares
from scipy.stats import norm

WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "data" / "key_number_weights.csv"
WEIGHT_SEASONS = range(2015, 2026)  # PAT moved back in 2015, shifting key numbers
MAX_M = 21  # |m| above this shares one tail weight (beyond 21 is sampling noise)
GRID = np.arange(-80, 81)

EDGE = 0.02        # drop quotes with mid <= 2c or >= 98c (tick-size distortion)
MIN_WIDTH = 0.01   # one tick
SIGMA_BOUNDS = (6.0, 25.0)


# --- margin models -------------------------------------------------------------

class NormalModel:
    name = "normal"

    def G(self, x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
        return norm.sf((np.asarray(x, float) - mu) / sigma)

    def mean(self, mu: float, sigma: float) -> float:
        return float(mu)

    def median_line(self, mu: float, sigma: float) -> float:
        return float(mu)


class KeyNumberModel:
    name = "key_number"

    def __init__(self, weights: np.ndarray):
        """weights[k] for |m| = k, k = 0..MAX_M; weights[MAX_M + 1] for the tail."""
        self.weights = np.asarray(weights, float)
        a = np.minimum(np.abs(GRID), MAX_M + 1)
        self._w_grid = self.weights[a]

    def pmf(self, mu, sigma: float) -> np.ndarray:
        """P(M = m) over GRID; a row per mu if mu is an array."""
        mu = np.asarray(mu, float)[..., None]
        buckets = norm.cdf((GRID + 0.5 - mu) / sigma) - norm.cdf((GRID - 0.5 - mu) / sigma)
        p = self._w_grid * buckets
        return p / p.sum(axis=-1, keepdims=True)

    def G(self, x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
        p = self.pmf(mu, sigma)
        x = np.atleast_1d(np.asarray(x, float))
        gt = (GRID[None, :] > x[:, None]) @ p
        eq = (GRID[None, :] == x[:, None]) @ p
        return gt + 0.5 * eq

    def mean(self, mu: float, sigma: float) -> float:
        """Expected margin. Differs from mu because w is not flat near mu."""
        return float(self.pmf(mu, sigma) @ GRID)

    def median_line(self, mu: float, sigma: float) -> float:
        """Fair spread: the line s with G(s) = 0.5, pushes counted half.

        G is evaluated on the half-point grid and linearly interpolated, so
        this is directly comparable to a sportsbook spread (a median, not a
        mean: books shade toward key numbers).
        """
        xs = np.arange(-40.0, 40.01, 0.5)
        g = self.G(xs, mu, sigma)
        i = int(np.argmax(g < 0.5))
        return float(xs[i - 1] + (g[i - 1] - 0.5) / (g[i - 1] - g[i]) * 0.5)


def estimate_key_weights(results: np.ndarray, spreads: np.ndarray,
                         iters: int = 50) -> tuple[np.ndarray, float]:
    """Fit w(|m|) and sigma so the model reproduces historical margins.

    Each game's margin is modelled around its closing spread. Iteratively
    rescale w so expected |m| frequencies match observed ones, and pick sigma
    so the model's mean squared error around the spread matches the data's.
    """
    results = np.asarray(results, float)
    spreads = np.asarray(spreads, float)
    obs_abs = np.minimum(np.abs(results), MAX_M + 1).astype(int)
    observed = np.bincount(obs_abs, minlength=MAX_M + 2).astype(float)
    target_mse = float(np.mean((results - spreads) ** 2))
    a_grid = np.minimum(np.abs(GRID), MAX_M + 1)

    mus, counts = np.unique(spreads, return_counts=True)
    sq_dev = (GRID[None, :] - mus[:, None]) ** 2

    w = np.ones(MAX_M + 2)
    sigma = float(np.sqrt(target_mse))
    for _ in range(iters):
        model = KeyNumberModel(w)

        def mse_gap(s):
            return counts @ (model.pmf(mus, s) * sq_dev).sum(axis=1) / counts.sum() - target_mse

        sigma = brentq(mse_gap, 8.0, 20.0, xtol=1e-3)
        pm = counts @ model.pmf(mus, sigma)
        expected = np.bincount(a_grid, weights=pm, minlength=MAX_M + 2)
        w = w * (observed + 0.5) / (expected + 0.5)
        w = w / np.average(w, weights=expected)  # pin overall scale
    return w, sigma


def load_key_weights(path: Path = WEIGHTS_PATH, rebuild: bool = False) -> KeyNumberModel:
    if rebuild or not path.exists():
        import nflreadpy as nfl

        s = nfl.load_schedules(seasons=list(WEIGHT_SEASONS)).to_pandas()
        s = s[s.result.notna() & s.spread_line.notna()]
        w, sigma = estimate_key_weights(s.result.to_numpy(), s.spread_line.to_numpy())
        df = pd.DataFrame({"abs_margin": [str(k) for k in range(MAX_M + 1)] + [f">{MAX_M}"],
                           "weight": w})
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(f"# seasons {WEIGHT_SEASONS.start}-{WEIGHT_SEASONS.stop - 1}, "
                    f"{len(s)} games, fitted sigma {sigma:.3f}\n")
            df.to_csv(f, index=False)
    df = pd.read_csv(path, comment="#")
    return KeyNumberModel(df.weight.to_numpy())


# --- observations ----------------------------------------------------------------

def game_observations(markets: pd.DataFrame, home: str, away: str) -> pd.DataFrame:
    """Tidy snapshot rows for one game -> (x, q, weight, source) observations.

    `markets` are the game's rows from the tidy parquet (kinds game_winner and
    spread), with `team` in nflverse codes.
    """
    rows = []
    for r in markets.itertuples():
        if r.mid is None or np.isnan(r.mid) or r.mid <= EDGE or r.mid >= 1 - EDGE:
            continue
        if r.team == home:
            sign, q = 1, r.mid
        elif r.team == away:
            sign, q = -1, 1 - r.mid
        else:
            raise ValueError(f"{r.market}: team {r.team} not in {away}@{home}")
        if r.kind == "spread":
            x, src = sign * r.threshold, "spread"
        elif r.kind == "game_winner":
            x, src = 0.0, "moneyline"
        else:
            continue
        rows.append({"market": r.market, "x": x, "q": q, "source": src,
                     "weight": (MIN_WIDTH / max(r.width, MIN_WIDTH)) ** 2})
    return pd.DataFrame(rows, columns=["market", "x", "q", "source", "weight"])


# --- fitting ----------------------------------------------------------------------

@dataclass
class GameFit:
    mu: float             # location parameter of the underlying normal
    sigma: float
    mean: float           # expected home margin (what Stage B consumes)
    median: float         # fair spread, comparable to a sportsbook line
    rmse_cents: float     # weighted RMS of (market - model), in cents
    max_abs_cents: float
    n_obs: int


def _residuals(model, obs: pd.DataFrame, mu: float, sigma: float) -> np.ndarray:
    return (obs.q.to_numpy() - model.G(obs.x.to_numpy(), mu, sigma)) * np.sqrt(obs.weight.to_numpy())


def _summarise(model, obs, mu, sigma) -> GameFit:
    raw = obs.q.to_numpy() - model.G(obs.x.to_numpy(), mu, sigma)
    w = obs.weight.to_numpy()
    return GameFit(mu=float(mu), sigma=float(sigma),
                   mean=model.mean(mu, sigma), median=model.median_line(mu, sigma),
                   rmse_cents=float(100 * np.sqrt(np.average(raw**2, weights=w))),
                   max_abs_cents=float(100 * np.abs(raw).max()), n_obs=len(obs))


def _mu_start(obs: pd.DataFrame, sigma: float) -> float:
    ml = obs[obs.x == 0]
    q = ml.q.mean() if len(ml) else obs.q.mean()
    return float(sigma * norm.ppf(np.clip(q, 0.02, 0.98)))


def fit_game(model, obs: pd.DataFrame, sigma: float | None = None) -> GameFit:
    """Fit mu (and sigma unless given) for one game by weighted least squares."""
    if len(obs) == 0:
        raise ValueError("no usable observations")
    if sigma is None:
        if obs.x.nunique() < 2:
            raise ValueError("need at least two distinct thresholds to fit sigma")
        start = [_mu_start(obs, 13.0), 13.0]
        res = least_squares(lambda p: _residuals(model, obs, p[0], p[1]), start,
                            bounds=([-40, SIGMA_BOUNDS[0]], [40, SIGMA_BOUNDS[1]]))
        mu, sigma = res.x
    else:
        res = least_squares(lambda p: _residuals(model, obs, p[0], sigma),
                            [_mu_start(obs, sigma)], bounds=([-40], [40]))
        mu = res.x[0]
    return _summarise(model, obs, mu, sigma)


def fit_pooled_sigma(model, games: dict[str, pd.DataFrame]) -> tuple[float, dict[str, GameFit]]:
    """One sigma shared by all games, one mu per game."""
    keys = list(games)
    start = [_mu_start(games[k], 13.0) for k in keys] + [13.0]

    def resid(p):
        return np.concatenate([_residuals(model, games[k], p[i], p[-1]) for i, k in enumerate(keys)])

    lo = [-40] * len(keys) + [SIGMA_BOUNDS[0]]
    hi = [40] * len(keys) + [SIGMA_BOUNDS[1]]
    res = least_squares(resid, start, bounds=(lo, hi))
    sigma = float(res.x[-1])
    return sigma, {k: _summarise(model, games[k], res.x[i], sigma) for i, k in enumerate(keys)}
