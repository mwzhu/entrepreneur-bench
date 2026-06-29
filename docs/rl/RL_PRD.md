# Entrepreneur-Bench RL — Product Requirements Document

**Status:** Draft v1 · **Date:** 2026-06-28 · **Owner:** project author
**Companion doc:** [`RL_TECHNICAL_DESIGN.md`](./RL_TECHNICAL_DESIGN.md)

---

## 1. One-line goal

Take a small open-weights model, RL it with GRPO against the existing Entrepreneur-Bench / Solvent benchmark, and show it **beats its own zero-shot baseline on a held-out eval split** — reported on the per-capability **regret decomposition**, not just total profit.

**Stretch goal:** close the gap to — or beat — the larger hosted models already benchmarked: DeepSeek-V3.2 ≈ **$14.9k** net profit, MiniMax-M3 ≈ **$11.9k** (from a $1,000 start, 30-day horizon).

## 2. Why this project exists

This is a **portfolio piece** aimed at applied-AI startups (enterprise-automation agents are the domain interest). It demonstrates an end-to-end RL-on-agents capability and produces a credible artifact set. The framing — *an eval and an RL environment are the same artifact* — is taken directly from Zapier's AutomationBench (built on Prime Intellect's Verifiers so one environment serves both eval and RL-with-verifiable-rewards). The eval already exists in this repo; **the project is to close the loop and train against it.**

Two carried-in lessons (from AutomationBench + the "Vibe RL" post):
1. **The reward function is the whole game.** Reward hacking is the default outcome; you only catch it by inspecting actual rollouts, not reward curves. (Zapier caught a hack when a tool-call metric collapsed while reward stayed flat.)
2. **Unit-test the reward function** so you trust the rubric.

## 3. Target outcomes & deliverables

Two narrative deliverables (both in scope):

| Deliverable | Description |
|---|---|
| **D1 — Engineering-log blog** | "Vibe RL"-style messy, honest log. Centerpiece is a **reward-hacking story** caught via metric divergence. Captures: baseline number, first-run failure, any reward hack + its metric divergence, per-capability before/after, concrete hyperparameter findings. |
| **D2 — Results README / section** | Tight portfolio-facing summary. Headline = **per-capability before/after regret table** + the "beat baseline" claim, with the DeepSeek/MiniMax reference numbers for context. |
| **D3 — Published Verifiers env** | Package the Solvent-as-Verifiers wrapper and **push it to the Prime Intellect Environments Hub** (the AutomationBench analog). Strengthens the "eval == RL env" thesis and is itself a portfolio artifact. |

Supporting artifacts to capture *as we build* (inputs to D1/D2): zero-shot baseline scorecards, the first training run's failure mode, canary-metric time series (W&B), per-seed held-out eval scorecards, and the reward-unit-test suite.

## 4. Success metrics

**Primary (must-hit for the project to "work"):**
- Trained policy's **`expected_net_revenue` on held-out test seeds** > its own zero-shot baseline, with the gap larger than per-seed noise (report mean ± across seeds/samples).
- Improvement is **attributable on the regret decomposition** — at least one of `selection_regret`, `pricing_regret`, `oracle_tool_regret` drops materially baseline→trained, and we can say *which capability* improved.

**Secondary / credibility:**
- A documented reward-hack (or a clean "we instrumented for it and here's why none survived") with the **canary divergence** that surfaced it.
- A reproducible held-out eval protocol (train on dev seeds, never on test seeds).
- At least one concrete, reported **hyperparameter finding** (LoRA α, LR, batch/group size, curriculum effect).

**Stretch:** trained policy `expected_net_revenue` approaches MiniMax-M3 (~$11.9k) at the 30-day horizon.

**Non-metric success:** the env is clean enough to publish to the Hub and a third party could run "one command from eval to RL."

## 5. Scope decisions (locked)

| Decision | Choice | Rationale |
|---|---|---|
| Compute budget | **Shoestring (< ~$300)** | Portfolio piece; prove the loop + catch a hack, not win a leaderboard war. |
| Effort | **50–100 hours** | Bounds the milestone sequence; sweep is light, not exhaustive. |
| Infra | **Prime Intellect / Prime Lab (hosted)** | CLI already installed; matches AutomationBench lineage; no infra management. |
| Algorithm | **GRPO, no SFT cold-start** | Group-relative, no value net, good fit for verifiable rewards. Cleaner story. Cold-start risk is mitigated by the instruct base + short-horizon curriculum (see tech design §6). |
| Base model | **Decide from a cheap zero-shot baseline** (Qwen3-4B vs 8B) | Only DeepSeek/MiniMax baselines exist today; need a non-trivial-but-beatable open baseline. Milestone 0. |
| Horizon | **Curriculum: short → 30-day** | Fast credit assignment early; faithful 30-day final eval. |
| Dynamics | **Core mechanics + `job-ttl` (expiry pressure) ON + a lightweight commitment-breach fee**; reputation/work-time OFF | TTL punishes do-nothing/dithering; the breach fee (accept-then-never-deliver costs a fraction of contract price) closes the accept-without-deliver reward-hack loophole as a *sim consequence* rather than a reward term. Full reputation stays off. |
| Coherence | **Observed, never scored or rewarded** | The deterministic coherence checks double-count profit/regret and are a shaping-hack magnet; raw counts stay as diagnostics/canaries, long-horizon incoherence is a qualitative writeup finding. |
| Reward | **Hybrid, terminal-first:** dominant terminal true-profit + **terminal** regret shaping to start; add per-step shaping only if credit assignment proves too slow | Leverages the already-computed reference policies as a shaping head start; terminal term dominates so shaping can't override ground truth; starting terminal-only minimizes both implementation and reward-hacking surface. |
| Reward-hack approach | **Honest instrument-and-hunt** | Best reward from the start, all canaries always-on; report whatever emerges organically. |

## 6. Users & usage

- **Primary user:** the project author (training, inspecting rollouts, writing the log).
- **Secondary audience:** applied-AI hiring teams reading D1/D2; Hub users who install D3.
- **Key workflow:** `prime train run <config>` for training; Solvent's existing `score` / `replay` / `compare --redteam-paired` CLI as the rollout-inspection layer; W&B for curves + canaries.

## 7. Constraints & assumptions

- Budget is the binding constraint. Most training stays at **short horizons**; only a handful of **30-day rollouts** are run (eval needs no gradients).
- Rollouts are cheap on the **env side** — delivery outcomes are a characterized lookup table + seeded RNG, **no nested model calls** (verified, tech design §1.4). The only real cost is the policy's own generations.
- Prime Lab GPU availability may be tight; the plan keeps the infra layer abstract (an OpenAI-compatible vLLM endpoint) so it can fall back to a self-managed pod.
- The repo's seed splits are tiny (5 dev / 5 test); the plan **expands the dev training-seed pool** while keeping test seeds strictly held out (tech design §5).

## 8. Risks (product-level; technical risks in design doc §7)

| Risk | Severity | Mitigation |
|---|---|---|
| Shoestring + honest-hunt → no dramatic reward-hack appears | Med | Canaries always-on so any hack is caught first time; treat the simplest-defensible reward as a real baseline ablation so even a mild hack is a story. |
| Open base model too weak → baseline is incoherent, "beating it" is unimpressive | Med | Milestone 0 picks the model from a real zero-shot baseline; require "non-trivial but beatable." |
| Long-horizon episodes too expensive to train on at budget | High | Curriculum keeps gradient steps on short horizons; 30-day used mainly for eval. |
| Dense shaping teaches gaming the reference gap rather than profit | Med | Terminal true-profit dominates; canaries watch shaped-reward-vs-true-profit divergence. |
| Can't finish in 50–100h | Med | Milestones are ordered so a publishable result (beat baseline + one capability win) lands before the sweep/stretch work. |

## 9. Out of scope (v1)

- Full hyperparameter sweep across many models/sizes (one model, light sweep only).
- Reputation and work-time dynamics. (The lightweight commitment-breach fee — §scope above — is **in scope** and is distinct from the full reputation system: it is a single ledger debit on accepted-undelivered jobs, defaulting to 0 so existing traces score unchanged.)
- Multi-node / large-model training.
- Real delivery models (delivery stays the characterized menu).
- Any change to the headline DeepSeek/MiniMax numbers (used as-is for context).

## 10. Milestone overview (detail in design doc §7)

0. **Env contract (gate)** — `StatefulToolEnv` wrapper (incl. a one-tool toy env proving hidden-arg injection + schema stripping), breach-fee mechanic with `finalize()` ordering, trace export + `finalize()` before scoring, seed-split code change; one scripted + one Qwen rollout scored clean; reward tests S6/S10/S11 green.
0.5. **Throughput smoke (gate)** — measure turns/sec and $/1k short episodes on real env + node topology; re-derive the budget before committing batch/step counts.
1. **Baseline & model pick** — Qwen zero-shot on dev+test **under the canonical breach config**; pick a non-trivial-but-beatable model.
2. **Smoke** — 1 GRPO step end-to-end on a 1–2 day horizon; confirm reward signal + valid tool-calls.
3. **Short-horizon learning** — show learning curve beats baseline on short episodes.
4. **Reward shaping** — add (terminal) regret terms; confirm terminal still dominates (S6).
5. **Reward-hack hunt** — read rollouts, watch canary divergence, document findings.
6. **Horizon curriculum** — anneal up toward 30-day.
7. **Light sweep** — LoRA α, LR, group/batch size.
8. **Final held-out eval** — test seeds, 30-day, regret table; write D1/D2; push D3.
