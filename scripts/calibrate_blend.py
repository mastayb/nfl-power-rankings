"""Measure the market/résumé blend by cutoff week and pin it in config.

Run: python scripts/calibrate_blend.py
Writes config/resume_blend.yaml, read by scripts/stage_b.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ratings.backtest import blend_coefficients, load_history  # noqa: E402

SEASONS = range(1999, 2026)
CUTOFFS = range(2, 11)
OUT = ROOT / "config" / "resume_blend.yaml"


def main() -> None:
    coefs = blend_coefficients(load_history(SEASONS), CUTOFFS)
    print(coefs.round(3).to_string(index=False))
    doc = {
        "description": ("Rest-of-season margin = a + b_market * market_pred + b_resume * resume_pred,"
                        " fitted on regular-season games after each cutoff week. market: closing"
                        " lines through week k+1; résumé: results through week k, capped +-21;"
                        " both decayed with a 3-week half-life. Intervals: 90% season bootstrap."),
        "seasons": f"{SEASONS.start}-{SEASONS.stop - 1}",
        "half_life_weeks": 3.0,
        "cap": 21.0,
        "by_cutoff": [{k: (round(float(v), 4) if isinstance(v, float) else
                           [round(float(x), 4) for x in v] if isinstance(v, list) else int(v))
                       for k, v in row.items()} for row in coefs.to_dict("records")],
    }
    with open(OUT, "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False, width=100, allow_unicode=True)
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
