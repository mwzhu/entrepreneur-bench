# M0/M0.5 Gate Status

Date: 2026-06-29

## Local M0 Gates

Status: pass for G1-G8 and G10-G14.

Evidence:

```bash
uv run --extra dev --extra rl pytest
uv run --extra dev --extra rl pytest rl/tests/test_rl_reward.py -q
uv run --extra dev --extra rl pytest rl/tests/test_throughput_summary.py -q
uv run --extra dev --extra rl python -c "import verifiers as vf; env = vf.load_environment('entrepreneur_bench', horizon_days=1); print(type(env).__name__, env.env_id, len(env.tool_defs or []))"
bash -n rl/scripts/run_m0_eval.sh
uv run --extra dev --extra rl python -m py_compile rl/scripts/summarize_throughput.py
```

Most recent local results:

- Full suite: 236 passed.
- RL gate suite: 10 passed.
- Throughput summarizer CLI suite: 1 passed.
- Verifiers loader: `EntrepreneurEnv entrepreneur_bench 17`.
- Actual Verifiers rollout loop: covered by `test_verifiers_rollout_loop_executes_tool_calls_and_scores`.

## External Gates

Status: pass for G9 and T1-T4.

Evidence:

- Prime CLI authentication works with the user-provided token; token was used only from the process environment and not written into repo artifacts.
- Prime wallet balance reported by CLI: $10.00.
- Prime Inference model list did not include the exact `Qwen/Qwen3-4B-Instruct-2507`; the smallest available hosted Qwen smoke used `qwen/qwen3-8b`.
- Direct Prime Inference sanity check with `qwen/qwen3-8b` returned `ok`.
- G9 was a wrapper/auth/tool-call contract smoke, not a baseline-quality measurement. It was served by hosted `qwen/qwen3-8b`; Milestone 1 baselines must set `brain_model`/provenance to the actually served model instead of reusing the M0 canonical 4B pricing id.
- G9 command completed successfully:
  `MODEL=qwen/qwen3-8b NUM_EXAMPLES=1 MAX_CONCURRENT=1 OUTPUT_DIR=rl/artifacts/evals/m0_qwen_g9 TRACE_DIR=rl/artifacts/traces/m0_qwen_g9 TIMEOUT_SECONDS=900 bash rl/scripts/run_m0_eval.sh`
- G9 result: reward `0.6760305000000001`, 25 model turns, 24 tool calls, 603.42s wall time, 162207 input tokens, 25513 output tokens.
- G9 trace: `rl/artifacts/traces/m0_qwen_g9/2b8e346c43fc4330b9befa39cabf7e14-seed-1000.jsonl`; final event is `terminated` with reason `horizon`.
- G9 results file: `rl/artifacts/evals/m0_qwen_g9/evals/entrepreneur_bench--qwen--qwen3-8b/d62afc0e/results.jsonl`.
- M0.5 intended-topology smoke ran on Prime MassedCompute pod `879381f8c2dd46c2b02af1cb870520f3`: 1x A6000_48GB on-demand at $0.54/hr, vLLM 0.10.0, `Qwen/Qwen3-4B-Instruct-2507`, `max_model_len=65536`, concurrency 3.
- vLLM tool-call parser note: `qwen3_coder` served text but did not convert Qwen `<tool_call>` blocks into OpenAI tool calls; restarting with `--tool-call-parser hermes` produced real tool calls and is the measured configuration.
- M0.5 command completed 64 C1 episodes in 784.86s with 64 result rows and 64 trace files.
- M0.5 results directory: `rl/artifacts/evals/m05_vllm_64/evals/entrepreneur_bench--Qwen--Qwen3-4B-Instruct-2507/0e238877`.
- M0.5 trace directory: `rl/artifacts/traces/m05_vllm_64`.
- `rl/scripts/summarize_throughput.py` wrote the gate-satisfying measurement to `rl/artifacts/throughput_smoke.md`: 0.713 turns/sec, mean 25.45 turns/episode, mean 35.72s/episode, p95 77.35s/episode, and $5.36 per 1k C1 episodes.
- Budget derivation: the measured rate implies about 55,989 C1 episodes under a $300 envelope and `max_steps x batch_size <= 6998` at GRPO group size G=8 before other budget slices.
- Go/no-go: pass. Proceed to Milestone 1 with the recorded 60/20/20 budget split, and rerun M0.5 if the model, parser, max-model-len, concurrency, or GPU topology changes.
- Reward metric note: `build_rubric()` optimizes the combined `terminal_reward` but also emits each sub-reward term as a zero-weight metric so W&B can track shaped-vs-true divergence without changing training reward.
- Breach-fee tuning note: S11 currently covers finalize timing, idempotence, trace order, insolvency relabeling, and zero-fee compatibility. If `breach_fee_frac` is tuned away from the M0 value, add a standalone delivered-EV-greater-than-breached-EV assertion for the chosen fee.

Resource cleanup:

- The vLLM endpoint and Prime pod were terminated after the measurement.
