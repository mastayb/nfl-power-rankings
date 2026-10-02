# Ideas parking lot

## Tune-your-own power rankings site (future side project)

A public site with a small, neat UI where anyone can hand-tune the weight of
every input and watch the ranking update: Does résumé matter? Do markets?
QBR, EPA, success rate, time of possession, luck, close games...

- Backend: this repo's per-team, per-game features (market rating, résumé
  variants, EPA/success rate, fumble/INT luck, game control), refreshed weekly.
- Default weights = the empirically measured blend (scripts/calibrate_blend.py,
  scripts/resume_round1.py), so users start from "what the data says" and
  can see how far their own preferences move teams.
- Could show, live, how predictive the user's weighting would have been
  historically (same LOSO backtest), as a score next to their ranking.

## Team-specific ATS inefficiencies (future research, after the rankings model)

Hunch: league-wide ATS patterns don't hold (confirmed: scripts/historical_test.py
finds no ATS momentum, no league-wide "playing down"), but individual teams may
be mispriced for discrete runs. Ben's experience: ~15-3 betting one NBA team
that played to its opponent's level when the spread was > 5.

Data points as of 2026-10-02:
- TB has failed to cover 12 straight (2025 wk 10 -> 2026 wk 3). Longest in
  1999-2025 nflverse data; no team had a streak > 10, and 10+ happened once.
- Back-of-envelope: a fair coin over ~13k team-games predicts ~6 streaks of
  10+; we saw 1. Long ATS streaks look shorter than chance -> the market seems
  to correct runs. Needs a proper test (cover rate ~49%, pushes, offseason).

How to test without fooling ourselves:
- Pre-register the pattern definition (e.g. "team's ATS residual by spread
  bucket", "plays to opponent level") and detect it on games 1..n only, then
  score it strictly forward on games n+1.. (walk-forward, no peeking).
- Correct for the search: 32 teams x many windows x many pattern types will
  always produce some 15-3 runs. Calibrate with permutation/simulation of the
  same search on shuffled ATS outcomes.
- Regime/run detection: changepoint or hidden-Markov models on each team's
  ATS residual series; ask whether a detected regime persists out of sample.
- Measure against the line available when you'd bet (opener vs close if we
  can get openers) and against the -110 breakeven (52.4%).
