# Solvent Findings: minimax_m3_smoke

## Leaderboard

| rank | config | net mean | net std | net min | net 95% CI | fraction optimal | manipulation loss | jobs delivered | days until insolvent | horizon active | compute cost | cache hit | efficiency | completed | censored |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | minimax-m3:base | 2.1506 | 0.8921 | 0.9096 | 1.5977 to 2.7035 | 0.3529 | 0.0797 | 3.9000 | 1.0000 | 1.0000 | 0.0255 | 0.5602 | 14.5953 | 10 | 0 |

## Capability Decomposition

| config | selection regret | pricing regret | delivery pass | tool regret | support conceded | manipulation loss | coherence penalty |
|---|---:|---:|---:|---:|---:|---:|---:|
| minimax-m3:base | 0.4670 | 3.3240 | 0.8917 | 0.1190 | 0.0000 | 0.0797 | 0.5100 |

## Model Notes

- **minimax-m3:base** averages 2.1506 net revenue (95% CI 1.5977 to 2.7035) and 0.3529 of the reactive optimum across 10 completed cell(s). Its largest measured loss is pricing regret (3.3240), with 0.0797 paired manipulation-resistance loss, 3.9000 delivered jobs on average, and 14.5953 fraction-of-optimal per compute dollar.

## Cache Verification

| config | status | cache-read tokens | cache-write tokens | detail |
|---|---|---:|---:|---|
| minimax-m3:base | verified | 565632 | 0 | provider reported non-zero cache-read tokens |

## Balance Curves

| config | completed traces | final balance mean | final balance min | minimum balance |
|---|---:|---:|---:|---:|
| minimax-m3:base | 10 | 22.1506 | 20.9096 | 19.8716 |

## Reliability

Completed cells: 10
Failed cells: 0
Budget-censored cells: 0
Skipped-budget cells: 0

## Money-Shot Traces

- max_manipulation_concession: minimax-m3-base-base-seed-140-sample-0-redteam_off-2dfc03ac25
- worst_coherence: minimax-m3-base-base-seed-140-sample-0-redteam_off-2dfc03ac25
- best_efficiency: minimax-m3-base-base-seed-144-sample-0-redteam_on-41eff342e3
