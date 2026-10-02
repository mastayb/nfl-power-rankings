# Market + Résumé Power Rankings, entering Week 4

These rankings combine two things: what betting markets think of each team, and what each team has done on the field. No expert opinions or hand adjustments go in.

| # | Team | Record | Rating | Win % vs. avg | Market rank | Résumé rank | Playoffs | Super Bowl |
|--:|:--|:-:|--:|--:|--:|--:|--:|--:|
| 1 | Rams | 1-2 | +5.7 | 68% | 1 | 9 | 67% | 10.2% |
| 2 | Bills | 3-0 | +4.7 | 65% | 2 | 14 | 92% | 12.3% |
| 3 | Seahawks | 2-1 | +4.3 | 64% | 4 | 8 | 68% | 7.4% |
| 4 | 49ers | 3-0 | +4.2 | 63% | 5 | 2 | 78% | 7.6% |
| 5 | Ravens | 2-1 | +4.2 | 63% | 3 | 11 | 82% | 8.9% |
| 6 | Chiefs | 3-0 | +3.9 | 62% | 6 | 4 | 81% | 6.7% |
| 7 | Jaguars | 2-1 | +3.6 | 61% | 8 | 1 | 67% | 4.7% |
| 8 | Broncos | 2-1 | +2.6 | 58% | 12 | 6 | 67% | 3.8% |
| 9 | Bengals | 2-1 | +2.1 | 57% | 11 | 16 | 72% | 4.7% |
| 10 | Lions | 2-1 | +1.9 | 56% | 10 | 22 | 71% | 5.2% |
| 11 | Cowboys | 1-2 | +1.8 | 56% | 13 | 17 | 44% | 3.1% |
| 12 | Eagles | 2-1 | +1.7 | 55% | 7 | 29 | 66% | 5.3% |
| 13 | Bears | 2-1 | +1.5 | 55% | 16 | 3 | 47% | 2.4% |
| 14 | Texans | 0-3 | +1.5 | 55% | 9 | 27 | 37% | 2.5% |
| 15 | Patriots | 1-2 | +1.2 | 54% | 14 | 10 | 46% | 2.1% |
| 16 | Vikings | 3-0 | +1.2 | 54% | 17 | 7 | 71% | 3.6% |
| 17 | Chargers | 0-3 | -0.5 | 48% | 18 | 26 | 15% | 0.7% |
| 18 | Colts | 1-2 | -0.6 | 48% | 19 | 21 | 38% | 1.3% |
| 19 | Packers | 1-2 | -0.8 | 47% | 15 | 32 | 29% | 1.6% |
| 20 | Saints | 1-2 | -1.0 | 47% | 20 | 12 | 47% | 1.3% |
| 21 | Raiders | 3-0 | -1.9 | 44% | 26 | 5 | 37% | 0.7% |
| 22 | Commanders | 1-2 | -2.1 | 43% | 21 | 23 | 19% | 0.7% |
| 23 | Panthers | 1-2 | -2.2 | 43% | 24 | 13 | 25% | 0.6% |
| 24 | Steelers | 2-2 | -2.2 | 43% | 22 | 20 | 22% | 0.5% |
| 25 | Buccaneers | 0-3 | -2.2 | 43% | 23 | 19 | 17% | 0.5% |
| 26 | Falcons | 1-2 | -2.9 | 41% | 25 | 25 | 26% | 0.5% |
| 27 | Cardinals | 1-2 | -3.7 | 39% | 29 | 18 | 6% | 0.1% |
| 28 | Giants | 2-1 | -4.1 | 37% | 28 | 28 | 19% | 0.4% |
| 29 | Browns | 3-1 | -4.3 | 37% | 30 | 15 | 23% | 0.3% |
| 30 | Jets | 1-2 | -4.3 | 36% | 27 | 30 | 17% | 0.3% |
| 31 | Titans | 0-3 | -5.7 | 32% | 31 | 31 | 4% | 0.1% |
| 32 | Dolphins | 0-3 | -7.6 | 27% | 32 | 24 | 1% | 0.0% |

**How to read it**

- **Rating:** points better or worse than an average team on a neutral field.
- **Win % vs. avg:** chance to beat an average team on a neutral field.
- **Market rank / Résumé rank:** where each half alone would place the team.
- **Playoffs / Super Bowl:** share of 10,000 simulated seasons.

**How it works**

1. **Market.** I use Kalshi prediction-market prices for the next two weeks of games and for every team's season win total, plus closing spreads from Weeks 1–3. I find the one rating per team that best explains all of these prices together.
2. **Résumé.** I rate each team on its scoring margins, adjusted for opponent strength. Recent games count more, and blowouts are capped at 21 points.
3. **Blend.** I tested the mix on 27 seasons of results. Early on, markets predict far better, so the résumé gets about 8% of the weight this week. Its share grows each week, to about 38% by Week 12. Later in the season, the résumé also removes luck: fumble recoveries and interceptions above normal rates.
4. **Playoff odds.** I simulate the season with full NFL tiebreakers. On real 2020–2025 results, the code matched every playoff field and #1 seed.

**What stands out**

- **Rams #1 at 1-2.** Markets rate them highest, and their opponent-adjusted results rank 9th.
- **Texans #14 at 0-3.** Markets still see an above-average team. Two losses came by one score, and they lost more fumble recoveries than they won.
- **Raiders #21 and Browns #29 with winning records.** Markets doubt both. The Raiders have recovered 8 of 12 fumbles in their games, the luckiest start in the league. The Browns are 3-0 in one-score games.

**Limits**

- This does not beat the spread. Résumé added nothing once the closing spread was known.
- Injuries count only as far as markets price them.
- Markets still carry preseason expectations, so early ratings lean on them.
