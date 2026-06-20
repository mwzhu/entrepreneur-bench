# Solvent v0.2

Solvent v0.2 is the attribution-machine slice of the deterministic freelance-operation eval environment. It can run a seeded freelance episode, write a JSONL trace, reconstruct hidden ground truth from seed metadata, and emit a per-stage scorecard for selection, pricing, delivery, support/manipulation, and coherence.

## Run

From the repository root:

```bash
python3 -m solvent.cli.main run --agent stub --seed 42
```

After installing the package, the equivalent console command is:

```bash
solvent run --agent stub --seed 42
```

Useful smoke-test options:

```bash
python3 -m solvent.cli.main run \
  --agent stub \
  --stub-mode happy_path \
  --seed 42 \
  --horizon 5 \
  --trace-path runs/test.jsonl
```

The run writes a deterministic JSONL trace with no wall-clock timestamps. Re-running the same seed and config to the same fresh path produces byte-identical trace content.

## Score

Run an episode and write a scorecard next to it:

```bash
python3 -m solvent.cli.main run \
  --agent stub \
  --stub-mode happy_path \
  --seed 42 \
  --scorecard \
  --trace-path runs/seed-42-stub.jsonl
```

Score an existing trace without rerunning the harness:

```bash
python3 -m solvent.cli.main score runs/seed-42-stub.jsonl --json
```

Compare two deterministic stub configs over paired seeds, including red-team-on/off manipulation resistance:

```bash
python3 -m solvent.cli.main compare \
  --a stub:naive \
  --b stub:procedure \
  --seeds 40,41,42,43,44 \
  --trace-dir runs/compare-v0_2 \
  --redteam-paired
```

## Test

```bash
python3 -m pytest -q
```

The suite covers the clock, ledger, data-clean verifier, versioned market generation, manipulation flow, scoring formulas, environment invariants, hidden-field trace guardrails, and CLI smoke paths.
# entrepreneur-bench
