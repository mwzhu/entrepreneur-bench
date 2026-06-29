# M0.5 Throughput Smoke

Status: measured

## Setup

- Results directory: `rl/artifacts/evals/m05_vllm_64/evals/entrepreneur_bench--Qwen--Qwen3-4B-Instruct-2507/0e238877`
- Node spec: Prime MassedCompute 1x A6000_48GB on-demand, vLLM 0.10.0, Qwen/Qwen3-4B-Instruct-2507, max_model_len=65536, concurrency=3, tool_call_parser=hermes
- Hourly rate: $0.54/hr
- Episodes scored: 64
- Errors: None
- Gate note: T1 satisfied.

## Throughput

- Total model turns: 1629
- Aggregate wall time: 2286.14s
- Turns/sec: 0.713
- Mean turns/episode: 25.45
- Mean wall-clock/episode: 35.72s
- P95 wall-clock/episode: 77.35s
- Cost per 1k C1 episodes: $5.36

## Budget Derivation

- Budget envelope: $300
- Affordable C1 episodes at measured rate: 55989
- With GRPO group size G=8, `max_steps x batch_size <= 6998` before other budget slices.
- Suggested split: reserve ~60% for C1-C2, ~20% for C3+sweep, ~20% for baselines+eval; update this section after choosing the actual training schedule.

## Gate Decision

- T1: pass. The run completed 64 C1 wrapper episodes with 64 scored result rows and 64 trace files.
- T2: pass. Measured throughput is 0.713 turns/sec, mean 25.45 turns/episode, and $5.36 per 1k C1 episodes on the recorded topology.
- T3: pass. At the measured rate, the $300 envelope buys about 55,989 C1 episodes, implying `max_steps x batch_size <= 6998` at GRPO group size G=8 before other budget slices.
- T4: pass. Proceed to Milestone 1 with the measured budget split; reserve about 60% for C1-C2, 20% for C3+sweep, and 20% for baselines+eval, and rerun this smoke if the model, vLLM parser, max-model-len, concurrency, or GPU topology changes.

## Scored Sample

| Trace | Turns | Wall s | Input tokens | Output tokens | Expected net | Selection regret | Pricing regret | Tool regret |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rl/artifacts/traces/m05_vllm_64/4b4e5b308b294584846713297500f2fc-seed-1000.jsonl | 45 | 50.43 | 562979 | 1502 | 572.99 | 2.96 | 354.95 | 179.70 |
| rl/artifacts/traces/m05_vllm_64/e433c97d50654abfbb75d4256a2dd96c-seed-1001.jsonl | 40 | 55.19 | 465236 | 1686 | -185.17 | 3.25 | 272.19 | 0.00 |
| rl/artifacts/traces/m05_vllm_64/24f9870c0cc740c58d97e912d3803ae7-seed-1002.jsonl | 29 | 48.76 | 249135 | 1552 | 771.42 | 357.94 | 198.08 | 9.44 |
| rl/artifacts/traces/m05_vllm_64/f5d22083a907404e8136b17b76406770-seed-1003.jsonl | 49 | 55.53 | 657253 | 1593 | 1139.26 | 265.66 | 355.16 | 28.13 |
| rl/artifacts/traces/m05_vllm_64/7078ade01d3e496c96f9f2fea40f64a3-seed-1004.jsonl | 34 | 38.06 | 327502 | 1152 | 1052.87 | 1105.82 | 378.04 | 0.00 |
| rl/artifacts/traces/m05_vllm_64/4a88f8117ae749ccb89fd71c0de038fa-seed-1005.jsonl | 28 | 42.37 | 223014 | 1382 | 396.38 | 553.81 | 102.01 | 0.00 |
| rl/artifacts/traces/m05_vllm_64/b074c482f68e4d7f9b611c178d1cacc7-seed-1006.jsonl | 48 | 65.08 | 665883 | 1966 | -207.43 | 434.97 | 308.44 | 0.00 |
| rl/artifacts/traces/m05_vllm_64/544bac5103cc48928144f846bf0574e6-seed-1007.jsonl | 9 | 10.36 | 31581 | 332 | 8.97 | 999.18 | 21.55 | 40.70 |
| rl/artifacts/traces/m05_vllm_64/281fe7429b2c4acb800e32d2deeaee52-seed-1008.jsonl | 9 | 8.36 | 31469 | 302 | 265.22 | 1157.96 | 65.70 | 3.08 |
| rl/artifacts/traces/m05_vllm_64/41ae5c6bdb03477daa33e02ab23fc7c9-seed-1009.jsonl | 28 | 45.02 | 223993 | 1461 | 578.93 | 736.81 | 220.84 | 13.04 |
| rl/artifacts/traces/m05_vllm_64/dc2a6665f1634a399f7e3ebf5e4a0896-seed-1010.jsonl | 9 | 10.49 | 32052 | 355 | 332.85 | 1100.03 | 52.05 | 1.96 |
| rl/artifacts/traces/m05_vllm_64/a96b0ff554fb4819a347bc694cc3f5b2-seed-1011.jsonl | 28 | 41.83 | 224152 | 1366 | 486.08 | 984.75 | 111.20 | 0.00 |
| rl/artifacts/traces/m05_vllm_64/6c973694619d4967a15b8bc17d33509e-seed-1012.jsonl | 10 | 10.41 | 37087 | 381 | 405.95 | 47.21 | 69.65 | 0.05 |
| rl/artifacts/traces/m05_vllm_64/06a6b7f382f04bc08567143c8c799854-seed-1013.jsonl | 16 | 22.83 | 82738 | 869 | 296.55 | 1727.97 | 68.54 | 0.00 |
| rl/artifacts/traces/m05_vllm_64/a448afd8ba79401887f768fb7bfe42b5-seed-1014.jsonl | 9 | 24.51 | 30852 | 1051 | 45.94 | 660.53 | 58.55 | 38.24 |
| rl/artifacts/traces/m05_vllm_64/9eb7a7b960c74795b67bd930fd086010-seed-1015.jsonl | 9 | 7.47 | 31619 | 285 | 334.46 | 345.16 | 110.85 | 0.76 |
| rl/artifacts/traces/m05_vllm_64/6bcc90dc0b514c14a04fff8d63ca78c1-seed-1016.jsonl | 9 | 9.47 | 32005 | 374 | 150.71 | 734.04 | 21.20 | 6.28 |
| rl/artifacts/traces/m05_vllm_64/8c98cc55b4334550828abae66f48db21-seed-1017.jsonl | 43 | 62.33 | 514501 | 1945 | -222.73 | 393.86 | 426.14 | 0.00 |
| rl/artifacts/traces/m05_vllm_64/495bbe90702b4a948f5c0a405d6a06f5-seed-1018.jsonl | 59 | 77.35 | 1005227 | 2209 | -501.38 | 4.26 | 525.47 | 0.00 |
| rl/artifacts/traces/m05_vllm_64/1e4bc8355fbb4a0aa3740c426e9411fb-seed-1019.jsonl | 28 | 50.34 | 225618 | 1539 | 736.43 | 872.19 | 216.37 | 0.00 |
