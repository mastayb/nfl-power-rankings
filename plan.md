# Market-Implied NFL Power Ratings — Build Plan

Oct 2, 2026 · @Ben Mastay

## Goal and output

The output is one number per team: r, its expected point margin against a league-average team on a neutral field. The 32 values are solved jointly so they best reproduce every market price at once: game lines plus the division, #1 seed, conference, and Super Bowl futures.

The headline metric is win % vs. an average team, Φ(r / σ), with σ ≈ 13.5 points. Treat σ as a fitted parameter, not a constant.

The key property is that futures are explained through the actual remaining schedule, tiebreakers, and bracket. Schedule strength and division strength are then absorbed by the simulation instead of leaking into r.

## Data sources

Kalshi alone covers every input, through its public REST market-data API with no key needed for reads. Its game markets list months ahead, which matters: future weeks' games act as the lookahead lines that make the 32 ratings identifiable from game prices alone ([predictionmarketspicks](https://predictionmarketspicks.com/nfl/kalshi)).

| Input | Kalshi series | Shape | Use |
| --- | --- | --- | --- |
| Game winner | KXNFLGAME | One binary per team per game | P(win) → margin via σ |
| Spread | KXNFLSPREAD | Ladder: "wins by over k.5" per threshold | Full margin distribution per game; best μ estimate |
| Win totals | KXNFLWINS | Over/under per team | Schedule-dependent, so fit through the sim |
| Division / conference / Super Bowl | Discover via `/series` | Mutually exclusive outcome sets | De-vig as a set, fit through the sim |
| #1 seed | Check whether listed | Same | Optional |

Sources for the series list: [Kalshi docs](https://docs.kalshi.com/api-reference/market/get-series), [crystalff PR on live NFL series](https://github.com/jontrieu22-lang/crystalff/pull/2), [kalshi-cockpit NFL series list](https://github.com/josephsapinoso/kalshi-cockpit/issues/202). That PR notes there are roughly 350 KXNFL series, mostly futures, so discover futures tickers once and pin them in config.

The docs show `external-api.kalshi.com/trade-api/v2` as the base URL, while public examples use `api.elections.kalshi.com/trade-api/v2` ([example](https://apify.com/leorochasantos/kalshi-markets)). Confirm which one works from your network on day one.

Price each market from the bid/ask midpoint, not last trade, and record the spread width as a liquidity weight. Snapshot raw JSON to disk with a timestamp so every fit is reproducible.

For schedule, results, and tiebreaker inputs, use nflverse: `nflreadpy` in Python, or `nflreadr` plus `nflseedR` if you go R.

## Model

The fit runs in two stages. Stage A turns each game's markets into one implied margin, μ\_g. Stage B solves for ratings that explain all the μ\_g values and the futures together.

### Stage A: per-game implied margin

Each KXNFLSPREAD rung is a probability that the home margin M clears k. Fit μ\_g and σ\_g per game against the ladder plus the moneyline:

```latex
P(M > k) = 1 - F\left(\frac{k + 0.5 - \mu_g}{\sigma_g}\right)
```

Start with a normal F. Then upgrade to the empirical NFL margin distribution, because key numbers (3, 7, 10) put real mass at specific margins and a normal smears them. Record the fit residual: a ladder that a single μ can't explain points to a stale or thin market.

### Stage B: ratings

Each game's margin is a rating difference plus home field (H = 0 at neutral sites):

```latex
\mu_g = r_{home} - r_{away} + H, \quad \sum_t r_t = 0
```

Win totals need no simulation. Expected wins and the full wins distribution are exact: current wins plus a Poisson-binomial over the remaining games, with p\_g = Φ(μ\_g / σ). This makes them cheap, differentiable, and fully schedule-aware.

Division, #1 seed, conference, and Super Bowl probabilities come from a Monte Carlo season. Draw each remaining game's margin as N(μ\_g, σ), apply tiebreakers and seeding, then run the bracket.

The joint objective adds a weighted squared error on game margins to two KL penalties on the futures:

```latex
L(r) = \sum_g w_g\,(\hat\mu_g - \mu_g(r))^2 + \lambda_W\,\mathrm{KL}(q_W \| p_W(r)) + \lambda_F \sum_m \mathrm{KL}(q_m \| p_m(r))
```

Here w\_g is a weight from liquidity and time to kickoff, q is the de-vigged market, and p is the model. Start with λ small: game lines should dominate, and futures should mostly break ties among the near-term lines.

### De-vig

Within each mutually exclusive set, such as one division's four teams or the Super Bowl's 32, the YES midpoints sum above 1. The excess is concentrated in longshots, especially at the 1¢ minimum tick. Use the power method: find k so that the probabilities raised to the k sum to 1. Drop or heavily downweight any contract quoted at 1–2¢.

### Optimizing through the simulation

Use common random numbers: draw one fixed matrix Z of size N\_sims × G and set margins to μ(r) + σZ. The simulated probabilities are then deterministic in r, though piecewise-constant. Optimize the 32 ratings with CMA-ES, starting from the Stage B solution without the simulated futures (lines plus win totals). Use about 20k simulations per evaluation, fully vectorized in numpy.

### Optional: current vs. rest-of-season strength

Give each team two ratings: r\_now drives this week's game, and r\_ros drives everything else. Add a penalty α(r\_now − r\_ros)². A large gap usually means an injury or a QB change, which is a useful output in its own right.

## Repo layout and build phases

The stack is Python 3.12 with numpy, scipy, pandas, httpx, `cma`, nflreadpy, and pytest. Each phase ends with something runnable you can check by hand.

```
nfl-market-ratings/
  config/series.yaml        # pinned Kalshi series/event tickers per season
  ratings/kalshi.py         # client: paging, mids, spread width, raw snapshots
  ratings/schedule.py       # nflverse schedule + results, team code mapping
  ratings/devig.py          # power + Shin de-vig over mutually exclusive sets
  ratings/ladder.py         # Stage A: spread ladder + ML -> mu_g, sigma_g
  ratings/linear.py         # Stage B: WLS + win-total Poisson-binomial
  ratings/season_sim.py     # vectorized sim: tiebreakers, seeding, bracket
  ratings/fit.py            # joint objective, CRN, CMA-ES
  ratings/report.py         # table + chart: r, win% vs avg, residual by market
  data/snapshots/           # raw JSON, timestamped
  tests/
```

1. Build the Kalshi client and snapshot. Discover the NFL series, pin the tickers, and dump one full snapshot. Check: every remaining game has a KXNFLGAME event, and each division set has four teams.
2. Map teams and games. Join Kalshi event tickers to nflverse game IDs, and flag neutral sites (international games). Check: zero unmatched games.
3. Implement Stage A. Produce μ\_g and the fit residual per game. Check: μ\_g lands within about 0.5 points of a sportsbook spread for a few games.
4. Implement Stage B using lines only. Check: ratings sum to 0, and HFA comes out in a plausible 1–2.5 point range when fitted.
5. Add win totals. Check: expected wins vs. the KXNFLWINS lines; large misses are interesting.
6. Build the season simulator. Implement division tiebreakers through common games, then conference record, then a coin flip. Validate against nflseedR on the same inputs; target matching P(division) within about 1 point.
7. Run the joint fit, then the report.

## Validation and pitfalls

The decisive test is out-of-sample. Fit on futures and win totals only, then predict next week's closing μ\_g. Score the RMSE against three baselines:

- A ranking from Super Bowl odds alone.
- The same futures fit without schedule adjustment.
- Baldwin's published numbers.

If the schedule correction matters, the full model should beat all three.

**Pitfalls**

- **Correlated sets.** A team's division, conference, and Super Bowl probabilities are nested, so they aren't independent evidence. Don't let λ\_F grow just because you have many futures markets.
- **Stale markets.** Futures can sit with wide spreads for days. Weight by inverse spread width, and drop any set where the sum of mids exceeds 1.15.
- **Fixed σ.** With σ fixed at 13.5, win % vs. average is only as good as that constant. Fit σ in Stage A across all ladders and report it.
- **Overtime ties.** These are rare enough to ignore in the simulation. If needed, treat a tie as half a win in win totals.
- **Tiebreaker fidelity.** Simplified tiebreakers mostly affect seeding markets, so the #1 seed is where errors will surface first. That's what the nflseedR cross-check is for.
- **Settlement wording.** Read each futures series' rules for postponed games and for edge cases like "conference champion" vs. "plays in the Super Bowl" before mapping it to a simulation outcome.

## Kickoff prompt for Claude Code

Export this doc to Markdown, save it as `PLAN.md` in an empty repo, then paste the prompt below. It is scoped to phases 1–2 so you can check the data before any modeling.

```
Read PLAN.md. We're building the market-implied NFL ratings model it describes.
Do phases 1 and 2 only, then stop and show me results.

1. Hit the Kalshi public market-data API (try both base URLs in PLAN.md; keep
   whichever responds). Discover every NFL series relevant to: game winner,
   spread ladder, season win totals, division winner, conference champion,
   Super Bowl champion, #1 seed. Write them to config/series.yaml with each
   series' settlement rule summary. Don't guess tickers -- list /series and
   filter.
2. Snapshot all open markets in those series to data/snapshots/<UTC ts>.json
   (raw), plus a tidy parquet: event, market, team, threshold, yes_bid,
   yes_ask, mid, width, close_time.
3. Pull the current-season schedule/results via nflreadpy and join every Kalshi
   game event to an nflverse game_id. Flag neutral sites.
4. Print: unmatched games (expect 0); per mutually exclusive futures set,
   the sum of mids; per game, number of spread rungs.

Use pytest for the join and the mid/width math. No modeling yet.
```
