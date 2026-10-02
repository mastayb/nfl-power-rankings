"""De-vig mutually exclusive outcome sets.

Power method: find k with sum(p_i ** k) = 1. Raising to k > 1 shrinks small
prices proportionally more, which matches where the overround sits (longshots,
especially at the 1c tick). Also handles sets that sum below 1 (k < 1).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def power_devig(p) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    if abs(p.sum() - 1) < 1e-12:
        return p
    k = brentq(lambda k: np.sum(p**k) - 1.0, 0.05, 20.0)
    return p**k
