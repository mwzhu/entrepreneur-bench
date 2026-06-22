# Solvent v0.5

Solvent is a deterministic freelance-operation eval environment. v0.5 turns the v0.4 attribution backend into a long-horizon, multi-model experiment platform: bounded context, prompt-cache accounting, provider-neutral model clients, budgeted experiment matrices, and Vending-Bench-style findings with Solvent's per-capability decomposition.

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
| `kimi-*`, `glm-*`, `minimax-*` | native provider key or `OPENROUTER_API_KEY` fallback |

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

Check the canonical v0.5 experiment config without spending:

```bash
python3 -m solvent.cli.main doctor --config configs/experiments/vb_style_v1.yaml --json
```

Add `--probe-live` for tiny live gateway probes. This can spend a small amount and should be run only when you want to verify inference credentials.

## v0.5 Experiment Workflow

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
