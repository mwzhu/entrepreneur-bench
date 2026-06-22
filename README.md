# Entrepreneur-Bench

[**📊 Read the full report →**](https://mwzhu.github.io/entrepreneur-bench/report.html)

Entrepreneur-Bench is a [Vending-Bench](https://andonlabs.com/evals/vending-bench)–style simulation that measures whether an AI agent can run a freelance software-services business (think Upwork): pick the right jobs, negotiate prices, deliver the work, and manage a budget over a multi-week horizon.

Instead of scoring only total profit, it decomposes performance into the capabilities that actually matter for running a business — **problem selection** (which jobs to take), **pricing** (capturing surplus without overshooting), **tool selection** (choosing the right delivery model for the job), and **resource allocation** — each measured as regret against an omniscient optimum.

The benchmark runs on **Solvent**, a deterministic evaluation environment: bounded context, prompt-cache cost accounting, provider-neutral model clients, budgeted/resumable experiment matrices, and per-capability scorecards.

## How it works

- The agent is a solo freelancer working a **job board**. New jobs (CSV-cleaning and field-extraction tasks) arrive a few times per day and expire if untouched.
- Each job shows a **starting price** and has a **hidden client ceiling**. The agent can accept at the starting price or make a **single** counter-offer; if the counter is at or below the ceiling it lands, otherwise the starting price stays open.
- Accepted jobs only pay out after a **successful delivery**. Delivery uses a public menu of "tools" (models) that trade cost against capability, and the agent must infer each job's hidden difficulty to pick well.
- Business time, tool-call overhead, and delivery charges all draw down the balance. The episode ends at the horizon or on insolvency.
- Scoring reports realized net profit plus a **regret decomposition** against omniscient, realizable, and threshold-policy reference policies.

## Latest results

A 30-day (43,200-tick) run comparing **DeepSeek-V3.2** and **MiniMax-M3** across 4 seeds each. DeepSeek finishes well ahead (≈$14.9k vs ≈$11.9k net profit from a $1,000 start) — driven almost entirely by engagement: it works nearly every profitable job, while MiniMax leaves ~20% of good jobs on the table. Full charts, metrics, and trace-level findings are in the [report](https://mwzhu.github.io/entrepreneur-bench/report.html).

---

## Run

From the repository root:

```bash
python3 -m solvent.cli.main run --agent stub:happy_path --seed 42 --scorecard
```

Live model runs use the same environment boundary:

```bash
python3 -m solvent.cli.main run \
  --agent claude-opus-4-8:+procedure \
  --seed 42 \
  --model-max-turns 10 \
  --model-max-tokens 512 \
  --scorecard
```

Set `ANTHROPIC_API_KEY` for Claude runs. If the benchmark label differs from the provider model id available in your account, set an alias, for example:

```bash
export SOLVENT_MODEL_ALIAS_CLAUDE_OPUS_4_8=<provider-model-id>
```

Additional live providers are selected by model family:

| family | credential |
|---|---|
| `gpt-*` | `OPENAI_API_KEY` |
| `gemini-*` | `GOOGLE_API_KEY` or `GEMINI_API_KEY` |
| `kimi-*`, `glm-*`, `minimax-*`, `deepseek-*` | native provider key or `OPENROUTER_API_KEY` fallback |

For OpenRouter fallback, use a standard inference API key. Management keys can introspect `/api/v1/key` but cannot run chat completions, and `doctor --probe-live` will flag them.

## Score And Replay

Score an existing trace without rerunning the harness:

```bash
python3 -m solvent.cli.main score runs/seed-42-stub.jsonl --json
```

Replay builds a trace view and scorecard from saved artifacts without model calls:

```bash
python3 -m solvent.cli.main replay runs/seed-42-stub.jsonl \
  --scorecard-output runs/seed-42-stub.scorecard.json \
  --view-output runs/seed-42-stub.view.json
```

## Compare

Compare configs over a seed split and build the static viewer:

```bash
python3 -m solvent.cli.main compare \
  --a claude-opus-4-8:base \
  --b claude-opus-4-8:+procedure \
  --seeds test \
  --samples 3 \
  --temperature 0 \
  --model-max-turns 10 \
  --model-max-tokens 512 \
  --trace-dir runs/compare-v0_4 \
  --viewer
```

`--model-max-turns` bounds ReAct loop iterations; `--model-max-tokens` bounds provider response length. They are useful for live smoke runs and are recorded in comparison summaries.

Optional v0.4b dynamics are available behind flags:

```bash
python3 -m solvent.cli.main compare \
  --a claude-opus-4-8:base \
  --b claude-opus-4-8:+procedure \
  --seeds test \
  --work-time \
  --job-ttl-ticks 2 \
  --reputation
```

`--work-time` lets delivery duration advance business time and overhead; `--job-ttl-ticks` expires jobs after arrival; `--reputation` gates future high-value board access based on delivery/support outcomes.

Stub baselines still work:

```bash
python3 -m solvent.cli.main compare \
  --a stub:naive \
  --b stub:procedure \
  --seeds dev \
  --redteam-paired
```

## Characterize

Validate the shipped hand-authored delivery menu:

```bash
python3 -m solvent.cli.main characterize --validate-menu
```

Generate a characterized menu artifact and brain profiles from dev seeds:

```bash
python3 -m solvent.cli.main characterize --generate-menu --seeds dev
```

## Doctor

Check local readiness and live-model credential setup:

```bash
python3 -m solvent.cli.main doctor --agent claude-opus-4-8:base
```

Check the canonical experiment config without spending:

```bash
python3 -m solvent.cli.main doctor --config configs/experiments/vb_style_v1.yaml --json
```

Add `--probe-live` for tiny live gateway probes. This can spend a small amount and should be run only when you want to verify inference credentials.

## Experiment Workflow

Estimate the full canonical matrix before spending:

```bash
python3 -m solvent.cli.main estimate configs/experiments/vb_style_v1.yaml
```

Run the resumable, budget-capped matrix:

```bash
python3 -m solvent.cli.main experiment run configs/experiments/vb_style_v1.yaml
```

By default this writes `runs/vb_style_v1`. Generate the report, leaderboard JSON, and static multi-model viewer from the ledger and scorecards:

```bash
python3 -m solvent.cli.main findings runs/vb_style_v1
```

Use the no-spend smoke path to verify the orchestration shape:

```bash
python3 -m solvent.cli.main experiment smoke configs/experiments/vb_style_v1.yaml \
  --model fake:base \
  --run-dir runs/smoke-v0_5
```

## Test

```bash
python3 -m pytest -q
```

The suite covers the public tool adapter, LLM harness with fake/recorded clients, delivery menu, business-time market streams, dynamic scoring, pricing, provider schema translation, cost estimates, experiment resume/budget guards, findings generation, comparison, replay, and the static viewers.
