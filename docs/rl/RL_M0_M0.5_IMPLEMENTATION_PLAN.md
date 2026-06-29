# Milestone 0 + 0.5 — Implementation Plan

**Scope:** Env contract (M0) + Throughput smoke (M0.5), combined.
**Status:** Implemented; M0/M0.5 gates passed · **Date:** 2026-06-29
**Parents:** [`RL_PRD.md`](./RL_PRD.md) §10 M0/0.5 · [`RL_TECHNICAL_DESIGN.md`](./RL_TECHNICAL_DESIGN.md) §2, §3, §5, §6.4, §7
**Owner:** project author

> These two milestones are the **gates before any GPU-dollar is spent**. M0 proves the
> Solvent↔Verifiers integration contract is sound and the reward is trustworthy. M0.5
> replaces the §6.4 cost *guess* with a *measured* throughput number and re-derives the
> training budget. Nothing in Milestones 1–8 is trustworthy until both pass.

---

## 0. Why combined, and what "done" means

M0 and M0.5 share one piece of infrastructure — a working, scored rollout through the
Verifiers wrapper. M0 builds and validates it on CPU/correctness terms; M0.5 runs the
*same* wrapper against a real vLLM-served policy to measure turns/sec and $/episode. Doing
them together avoids standing the integration up twice.

**Definition of done (headline):** a deterministic scripted policy *and* a zero-shot Qwen
policy both run through the wrapper → `finalize()` → `score_trace` with **no integration
errors**; reward unit tests **S6/S10/S11** are green; and we have a **measured** turns/sec +
$/1k-C1-episode number with an affordable `max_steps × batch_size × G` backed out from it.
Full scored rubric in §9.

---

## 1. Codebase facts this plan is built on (verified 2026-06-29)

All re-verified against the current tree; line numbers may drift but the structure is stable.

| Fact | Location | Consequence for M0 |
|---|---|---|
| `EnvConfig` is a `frozen=True` dataclass, 7 required fields (`seed`, `config_id`, `start_balance`, `horizon_ticks`, `overhead_per_tick`, `tool_call_cost`, `trace_path`), uses `Decimal`/`Path`. **No `breach_fee_frac` field today.** | `solvent/env/models.py:9-48` | Add `breach_fee_frac: Decimal = Decimal("0")` (defaulting 0 keeps every existing trace score-identical). |
| `finalize()` order today: cached-summary guard → set `terminated_reason` → emit `terminated` → build `EpisodeSummary` from `ledger.balance` → `trace.close()`. | `solvent/env/env.py:691-711` | Breach sweep slots **after** the cache guard and **before** the `terminated` emit (§3.2). |
| Delivered set = `submitted_jobs`; commitments = `accepted_jobs`. `AcceptedJob` carries `contract_price`, `submitted`, `delivery_model`, `paid`. | `env.py:52,156,564-566`; `models.py:159` | Breach set = `accepted_jobs.keys() − submitted_jobs`. |
| `ToolAdapter` exposes `schemas()` (mode-gated), `observe()`, `dispatch({"name","arguments"})` — the **one** mutation point. | `solvent/env/tool_api.py:101-200` | The registered Verifiers tool wraps `dispatch` + returns `observe()`. |
| `parse_seeds`/`seed_split_label` special-case only `{"dev","test"}`; everything else → `"ad_hoc"`. A `seeds_train.txt` path *resolves* but mislabels provenance. | `solvent/cli_seed.py:6-16` | Add `"train"` to both sets (§4). |
| `price_for_model` raises `KeyError` on unknown model id. | `solvent/env/pricing.py:328-332` | Add a `$0` entry for the chosen Qwen id (§5). |
| Scoring is pure trace replay; terminal target = `_expected_net_revenue` (control variate). | `solvent/scoring/scorecard.py:38,285-306` | Reward reads this from the scored trace. |
| `pricing_regret`/`selection_regret` credit **accepted**, not delivered jobs → accept-without-deliver loophole. | `scorecard.py:225,252` | Reward uses new **delivered-gated** helpers (§6). |
| `_pricing`/`_selection` are **private `ScorecardBuilder` methods** bound to instance state (`self.facts`, `self.jobs_by_id`, `self._accepted_jobs()`) — not importable/reusable as-is. | `scorecard.py:224,251` | Must **extract** into shared parameterized helpers (§6, G12), not "reuse". |
| Public `Scorecard` exposes only **aggregate** signals; job-level facts (`accepted_jobs`, delivery attempts, good IDs, menu) live in `TraceFacts`/`ScorecardBuilder` and are discarded by `score_trace`. | `scoring/models.py`; `events.py:61`; `scorecard.py:38-50` | Add `build_reward_context()` to surface them for delivered-gating (§6.0, G14). |
| Alias resolution is **canonical→wire**: `resolve_model_name(m)=os.environ.get(model_alias_env_var(m), m)`. | `harness/providers/base.py:40-46` | Price/meter by canonical id; resolve to wire only for the generation call (§5, G4). |
| `fraction_of_optimal_per_compute_dollar` is `None` when `brain_cost == 0`. | `scorecard.py:420` | At $0 pricing use a **token-normalized** compute canary, not the per-dollar field (§7.4). |
| Policy tokens enter the trace only via the `brain_metered` event (`LLMHarness._record_compute`), which `events.py` reconstructs into `brain_tokens_*`/`brain_cost`. Dropping `LLMHarness` ⇒ those fields read **zero** unless re-emitted. | `harness/llm.py:158-185`; `events.py:294`; `scorecard.py:415-421` | Wrapper emits `brain_metered` per turn (§7.4, G13). |
| `episode_started` emits a `provenance` block that does **not** list `breach_fee_frac` today. | `env.py:63` | Emit breach provenance **only when nonzero** to keep default-0 byte-identity (§3.1, G1). |
| Packaging: `find` only includes `solvent*`; pytest `testpaths=["tests"]`, `pythonpath=["."]`. | `pyproject.toml:24,38` | Extend both so `entrepreneur_bench`/`rl/tests` are discovered (§2.1, G11). |
| Tooling: no bare `python`; use `uv run pytest` (`pyproject` pins pytest≥8). `prime` CLI present at `~/.local/bin/prime`. `verifiers` **not yet installed**. | `pyproject.toml`; shell | M0 task 1 installs the RL deps in a separate env (§2). |

---

## 2. Prerequisites & environment setup (M0, task 0)

The Solvent package itself has **zero runtime deps** (`pyproject` `dependencies = []`). The RL
wrapper adds heavy deps (`verifiers`, `prime-rl`, `vllm`, `wandb`) that must **not** pollute
the Solvent eval/scoring path. Decision: keep RL code in a sibling package importing Solvent as a
library.

**Canonical module name = `entrepreneur_bench`** (top-level import, *not* `rl.entrepreneur_bench`).
The training TOML and the eventual Prime Hub publication (D3) both need one stable module name;
`rl/` is the on-disk location, not part of the import path. Layout: `rl/entrepreneur_bench/`,
imported as `import entrepreneur_bench`.

Tasks:
1. **Packaging — make `entrepreneur_bench` discoverable + tests collected** (fixes the
   `include = ["solvent*"]` / `testpaths = ["tests"]` gap in `pyproject.toml:24,38`):
   - `[tool.setuptools.packages.find]`: set `where = [".", "rl"]` and
     `include = ["solvent*", "entrepreneur_bench*"]` so both packages install.
   - `[tool.pytest.ini_options]`: `testpaths = ["tests", "rl/tests"]` and
     `pythonpath = [".", "rl"]` so `uv run pytest` collects `rl/tests` and `import entrepreneur_bench`
     resolves without an editable install.
   - **Acceptance:** `uv run python -c "import entrepreneur_bench"` succeeds and
     `uv run pytest rl/tests` collects (G11).
   - *Note (D3, later):* for Hub publication the env ships as a standalone package with its own
     `pyproject` (Verifiers `vf-install` convention). M0 keeps it in-repo via the find/path config
     above; the standalone packaging is a Milestone-8 task, not M0.
2. Add an optional extra to `pyproject.toml`:
   `rl = ["verifiers>=<pin>", "wandb", "openai"]` (vLLM/prime-rl are installed in the GPU pod, not locally).
3. Create `rl/entrepreneur_bench/` package: `__init__.py`, `environment.py` (the wrapper),
   `rewards.py` (delivered-gated helpers + Rubric builder), `seeds.py` (dataset builder).
4. Create `rl/tests/test_rl_reward.py` (S1–S11; M0 must green S6/S10/S11, others land here too).
5. Pin the **exact `verifiers` version** and record the `StatefulToolEnv` / `update_tool_args`
   / `setup_state` / `@vf.cleanup` signatures actually present in that version — the design
   flags these as the one Verifiers-API uncertainty. The toy env (§7) is what de-risks them.
6. Confirm `uv run pytest` runs the existing suite green *before* any change (baseline).

---

## 3. Workstream A — Sim-side breach-fee mechanic

Implements [design §3.4](./RL_TECHNICAL_DESIGN.md). This is a Solvent core change, gated behind
a default-0 field so existing traces are untouched.

### 3.1 `EnvConfig` field + conditional provenance (protects byte-identity)
`solvent/env/models.py`: add `breach_fee_frac: Decimal = Decimal("0")` to `EnvConfig`.

**Provenance must be conditional, or G1 byte-identity breaks.** The `episode_started` event emits
a `provenance` block (`env.py:63`) that does **not** list `breach_fee_frac` today. Adding it
*unconditionally* would change every newly generated trace and fail the default-0 byte-identity
criterion. Rule: **emit `breach_fee_frac` into provenance only when `breach_fee_frac != 0`.** At
the default (0) the provenance dict is byte-for-byte unchanged; when the fee is active, the trace
records it for reproducibility. (Same conditional applies to any breach metadata elsewhere in
`episode_started`.) This is what makes G1's "byte-identical at default" claim true rather than
aspirational.

### 3.2 `finalize()` breach sweep — exact ordering
`solvent/env/env.py:691-711`, insert the sweep with this order (the order is the contract S11 tests):
1. **Idempotence guard first** (unchanged): `if self._summary is not None: return self._summary`.
2. **Breach sweep next:** for each `job_id in accepted_jobs.keys() − submitted_jobs`:
   - `fee = (breach_fee_frac × accepted_jobs[job_id].contract_price)` (quantize to cents);
   - `self.ledger.debit_burn(fee)`;
   - `self._emit("breach", {"job_id", "contract_price", "fee", "balance_after": ledger.balance}, fee)`.
   - Skip entirely (no events, no debits) when `breach_fee_frac == 0`.
3. **Re-evaluate insolvency after debits:** if `ledger.balance < 0`, set `terminated_reason = "insolvent"` even if horizon had been reached (breach-caused insolvency is real; avoids a lying `"horizon"` reason that `r_solvency` would miss).
4. **Emit the single `terminated` event last** with the (possibly updated) reason.
5. **Build `EpisodeSummary`** from the now-post-breach `ledger.balance`, then `trace.close()`.

**Invariant:** all `breach` events precede the one `terminated` event; `terminated` is always
last ⇒ `events[-1]["balance_after"]` is the true post-breach final balance. Replay/viewer
assumptions stay intact.

> Note on classification: `events.py` classifies any positive non-overhead `burn_delta` as
> `tool_burn` (`scoring/events.py:175-179`), so the breach debit shows up as tool-burn in v1 —
> headline net revenue is correct; only the burn *attribution* is coarse. Acceptable for M0;
> add a `breach_burn` fact later only if breach diagnostics need their own chart.

### 3.3 Tests (in workstream A, but S11 is an M0 gate)
S11 sub-asserts (a)–(f) per [design §3.5](./RL_TECHNICAL_DESIGN.md): finalize-time firing,
idempotence (double `finalize()` debits once), trace order, insolvency relabel, delivered>breached
EV at the chosen fee, and `breach_fee_frac=0` ⇒ score/event-order identical to today.

---

## 4. Workstream B — Seed-split provenance code change

Implements [design §5.1](./RL_TECHNICAL_DESIGN.md). Tiny but gates eval-integrity reporting.

- `solvent/cli_seed.py`: add `"train"` to the set in `parse_seeds` (line 7) **and** in
  `seed_split_label` (line 16), so `parse_seeds("train")` reads `configs/seeds_train.txt` and
  traces are labeled `"train"` (not `"ad_hoc"`).
- Create `solvent/configs/seeds_train.txt` with a disjoint range (e.g. `1000–1127`, 128 seeds,
  one per line; comments allowed — `read_seed_file` strips `#`).
- Keep `seeds_test.txt` (`140–144`) strictly held out; leave it untouched.
- **Test (extends `tests/test_seed_split.py`):** assert `train ∩ test == ∅`, `train ∩ dev == ∅`,
  and that a trace produced with `split="train"` carries `seed_split_label == "train"`.

---

## 5. Workstream C — Canonical model id + pricing entry (the #1 footgun)

Implements [design §1.1 / Appendix](./RL_TECHNICAL_DESIGN.md). **Fixes the id-drift trap:** a
pricing entry under one id while the eval/cost path hits the model under a *different* id passes
the unit test yet still `KeyError`s live.

**One canonical internal id, alias resolves canonical → wire.** Define a single
`brain_model = "qwen3-4b-instruct"` (the **internal/priced** id, default; M1 may promote to
`qwen3-8b-instruct` via the identical mechanism). The existing alias machinery maps **canonical →
wire**, not the reverse: `resolve_model_name(model)` returns
`os.environ.get(model_alias_env_var(model), model)` (`harness/providers/base.py:40-46`), and
`model_alias_env_var("qwen3-4b-instruct")` → `SOLVENT_MODEL_ALIAS_QWEN3_4B_INSTRUCT`. So everything
*prices* by the canonical id and *resolves* to the wire id only when issuing the vLLM request.

- `solvent/env/pricing.py`: add a `$0` `BrainPrice` entry to `DEFAULT_BRAIN_PRICES` keyed on the
  **canonical** id `qwen3-4b-instruct`. Zero rates so RL/eval cost accounting doesn't attribute
  provider cost to a self-served model (tokens are still recorded — §7.4).
- Set `SOLVENT_MODEL_ALIAS_QWEN3_4B_INSTRUCT=Qwen/Qwen3-4B-Instruct-2507` (env / secrets). Wrapper
  sets `EnvConfig.brain_model = "qwen3-4b-instruct"` and meters/prices by the canonical id;
  `resolve_model_name("qwen3-4b-instruct")` yields the wire name for the actual generation call.
- **Test (extends `tests/test_pricing_multimodel.py`):** `price_for_model("qwen3-4b-instruct")`
  returns without `KeyError` and yields `brain_cost == 0`; and an alias-direction assertion: with
  `SOLVENT_MODEL_ALIAS_QWEN3_4B_INSTRUCT=Qwen/Qwen3-4B-Instruct-2507` set,
  `resolve_model_name("qwen3-4b-instruct") == "Qwen/Qwen3-4B-Instruct-2507"` (canonical→wire), so
  the live generation path and the priced/metered path can't diverge.

---

## 6. Workstream D — Reward module (delivered-gated) + Rubric

Implements [design §3.1–3.3, §3.5](./RL_TECHNICAL_DESIGN.md). Lives in `rl/entrepreneur_bench/rewards.py`.

### 6.0 Reward data source — the public `Scorecard` is not enough (required API task)
The delivered-gated helpers need **job-level facts**: delivered job IDs, the accepted-job facts
(`contract_price`, `via_counter`), the `Job` objects, and the `good_ids` set. The public
`Scorecard` (`scoring/models.py`) exposes only **aggregate signals** — none of these. Those facts
live in `TraceFacts` (`events.py:61`: `accepted_jobs`, `delivery_attempts`, legacy `submissions`,
…) and in the `ScorecardBuilder` (which also computes `jobs`, `jobs_by_id`, `reachable`, `delivery_menu`,
`_good_job_ids()`). `score_trace` builds all of this then throws the intermediate context away
(`scorecard.py:38-50`).

**Task — expose a reward context, don't reconstruct in `rewards.py`.** Add a
`build_reward_context(trace_path) -> RewardContext` (or have `ScorecardBuilder` expose
`reward_context()`) returning the job-level facts the helpers consume:
`{jobs_by_id, accepted_facts, delivered_job_ids, good_ids, delivery_menu, expected_net_revenue,
oracle_tool_regret, terminated_reason}`. **Important:** `delivered_job_ids` must come from
`ScorecardBuilder._delivery_attempts()` (or the same normalized logic): tool-mediated RL rollouts
emit `delivered`/`delivery_passed|failed` events that become `facts.delivery_attempts`, while
legacy direct-mode `submit` traces become `facts.submissions` and are normalized by
`_delivery_attempts()`. Do **not** define the delivered set as `{s.job_id for s in facts.submissions}`;
that would be empty for real tool-mediated RL rollouts and would zero out delivered-gated shaping.
This reuses
the builder's existing reconstruction (single source of truth, dovetails with G12) instead of
duplicating `load_events → facts_from_events → Market → jobs` inside the reward module. Rewards
then call `pricing_regret_over(delivered_job_ids, …)` / `selection_regret_over(delivered_job_ids, …)`
against that context. **Acceptance (G14):** `build_reward_context` returns all listed fields and
the reward functions consume it without re-implementing trace reconstruction.

### 6.1 Reward functions (read from the reward context of §6.0)
| Component | Source | Sign | Weight |
|---|---|---|---|
| `r_expected_net` | `_expected_net_revenue` (`scorecard.py:285-306`), `/1000` scale | + | 1.0 (dominant) |
| `r_pricing_neg_regret` | **new helper:** surplus-left over **delivered** jobs (formula from `_pricing`, `scorecard.py:251-283`, gated to normalized `delivered_job_ids`) | − | ~0.15 |
| `r_tool_neg_regret` | `oracle_tool_regret` (`scorecard.py:372-397`) — already per-delivery-attempt, used as-is | − | ~0.15 |
| `r_selection_neg_regret` | **new helper:** selection regret over **delivered** jobs (formula from `_selection`, `scorecard.py:224-249`, gated) | − | ~0.15 |
| `r_solvency` | `terminated_reason == "insolvent"` (`env.py:843`) | − | ~0.10 |

**Required prerequisite task — extract the scorecard math into shared, parameterized helpers.**
`_pricing` (`scorecard.py:251`) and `_selection` (`scorecard.py:224`) are **private
`ScorecardBuilder` methods bound to instance state** (`self.facts`, `self.jobs_by_id`,
`self._accepted_jobs()`, `self._good_job_ids()`) — they cannot be imported or "reused" as-is.
There is no shared helper today; one must be *created*. Concretely:

1. Refactor the pricing-regret loop body into a module-level helper
   `pricing_regret_over(job_ids, accepted_facts, jobs_by_id, good_ids) -> Decimal` (surplus-left
   over the passed `job_ids`).
2. Refactor the selection-regret computation into
   `selection_regret_over(chosen_ids, good_ids, ...) -> Decimal`.
3. **Re-point `ScorecardBuilder._pricing`/`_selection` to call the new helpers** with
   `job_ids = accepted_jobs` — so the scorecard's existing numbers are unchanged (guarded by the
   existing `tests/test_scoring_scorecard.py` + S8 determinism) and there is exactly **one** source
   of truth for the math.
4. RL rewards call the *same* helpers with `job_ids = delivered_job_ids` from `RewardContext`
   (delivered-gated; normalized across direct `submit` and tool-mediated `deliver` traces).

This is the single-source-of-truth refactor; the delivered-gating (passing normalized
`delivered_job_ids`) is
what closes the accept-without-deliver loophole on the reward side, while the breach fee (§3)
closes it on the sim side. `oracle_tool_regret` is already per-delivery-attempt, so it is used
as-is (no refactor). No coherence term — coherence is observed, never scored
([design §3.4](./RL_TECHNICAL_DESIGN.md)).

**Acceptance for the refactor (G12):** existing scorecard tests stay green (numbers unchanged) and
both `pricing_regret_over`/`selection_regret_over` are importable from a shared module and called
by both `ScorecardBuilder` and `rewards.py`.

### 6.2 M0 reward tests (the three gates + supporting)
The full S1–S11 suite lands here; **M0 gates on S6, S10, S11**:
- **S6 dominance invariant (the important one):** `max Σ|shaping| over a seed < min ΔR_terminal`
  between a do-nothing and a near-optimal policy on that seed. Run across all training seeds.
  Guarantees shaping can never outvote profit.
- **S10 delivery-gating (anti-commitment-hack):** a trace that *accepts* a good well-priced job
  but never delivers earns **zero** pricing/selection shaping; `dropped_jobs` increments; the same
  job delivered earns the credit.
- **S11 breach fee:** the §3.3 (a)–(f) assertions.
- S1–S5, S7–S9 (monotonicity, pricing/tool/selection regret, solvency, control-variate sanity,
  determinism, canary extraction) are authored here too but are not M0 blockers.

All tests are pure functions over hand-built / fixture traces (scoring is deterministic replay),
runnable via `uv run pytest rl/tests`.

---

## 7. Workstream E — The Verifiers wrapper + toy env (the contract proof)

Implements [design §2](./RL_TECHNICAL_DESIGN.md). **The single most de-risking step.**

### 7.1 One-tool toy env first
Before wiring all of Solvent, build a minimal `StatefulToolEnv` with **one** tool to prove the
four Verifiers-API uncertainties in isolation:
1. `setup_state()` puts a per-rollout mutable object in `state`.
2. `update_tool_args()` injects that object as a **hidden arg** (`_adapter`) — *stripped from the
   model-facing schema*.
3. A tool advertised via a hand-built schema (mirroring `TOOL_SCHEMAS` shape) with a
   `**arguments` signature dispatches and returns a string the runner appends as the tool message.
4. The Verifiers generation path exposes per-response token usage through a concrete hook we can
   use for metering: either a documented callback/custom client wrapper that receives
   `response.usage`, or an equivalent state-visible response object. The toy env must emit a
   synthetic `brain_metered`-shaped record from this hook and assert cumulative input/output counts
   match the generated responses.

If the toy env round-trips a hidden arg with the model never seeing it **and** proves a token-usage
capture hook before the full wrapper, the contract holds.

### 7.2 Full wrapper (`rl/entrepreneur_bench/environment.py`)
Per design §2.1 schematic, with the real `EnvConfig` (Decimal/Path, 7 required fields,
`horizon_minutes` for business-time, `breach_fee_frac`, `seed_split` provenance):
- `EntrepreneurEnv(vf.StatefulToolEnv)` with `setup_state` building a fresh seeded `Environment`
  + `ToolAdapter`; `update_tool_args` injecting `_adapter`; `is_completed` returning
  `env.terminated()`.
- One registered tool **per Solvent tool name**, each wrapping `adapter.dispatch` and returning
  `json.dumps({"result", "observation": adapter.observe()})`; advertised via `TOOL_SCHEMAS`
  filtered by `adapter.schemas()` (mode-gating: `submit` in direct, `deliver`/`list_models` in
  tool-mediated), `_adapter` stripped.
- **`finalize()` before scoring** in a `@vf.cleanup` hook (runs once/rollout) so the trace is
  closed and the terminal reason stamped before the Rubric reads it.
- System prompt = Solvent's `system_prompt(ablations)` (`harness/prompts.py:30-40`) for
  apples-to-apples with hosted baselines.
- `max_turns` from Solvent's formula `expected_jobs*10 + 200` ([design §1.2](./RL_TECHNICAL_DESIGN.md)).
- **No `LLMHarness`** ⇒ the $-budget guard never applies; episodes end only on horizon/insolvency.

### 7.3 Seed dataset builder (`seeds.py`)
One dataset row per seed for the requested `split`/`horizon_days`, each row's `info` carrying
`seed`, `config_id`, `split`. Uses the §4 `parse_seeds("train"/"test")`.

### 7.4 Meter Verifiers policy-token usage into the trace (Open Q2 — yes, gated by toy env)
Dropping `LLMHarness` removes the only thing that emits the `brain_metered` trace event
(`harness/llm.py:166`), which `events.py:294` reconstructs into `brain_tokens_in/out` and
`brain_cost` (`scorecard.py:415-421`). Without action those compute fields would silently read
**zero** for RL rollouts. Fix: the wrapper emits the same event per turn.

- Use the concrete token-usage hook proven by the toy env (§7.1). Preferred path: wrap/configure the
  Verifiers OpenAI-compatible client so every generation response passes through a small recorder
  with access to `response.usage`; fallback path: if Verifiers exposes usage only in rollout state,
  read it from there in the same per-generation callback. This must be resolved before the full
  wrapper is considered passing G7/G13.
- After each policy generation, build a `TokenUsage` (`solvent/env/pricing.py`) from that response
  usage.
- Emit via `state["env"]._emit("brain_metered", {…}, Decimal("0"))` using the **exact payload shape**
  `LLMHarness._record_compute` uses (`llm.py:158-185`): per-turn + cumulative
  `input/output/cache_read/cache_write` tokens, `cost`, `model`, `ablations`.
- `cost = brain_cost("qwen3-4b-instruct", usage)` — **$0** under our pricing entry (§5), but the
  **token counts are still recorded**, so `brain_tokens_in/out` populate.
- **Caveat (corrects an earlier claim):** `fraction_of_optimal_per_compute_dollar` is `None`
  whenever `brain_cost == 0` (`scorecard.py:420`), so at $0 pricing this scorecard field does
  **not** populate. The compute-efficiency canary must therefore be **token-normalized**, computed
  in the W&B logging layer (not read from the scorecard): e.g.
  `expected_net_revenue / max(brain_tokens_out, 1)` (profit per output token, or per-1k-tokens).
  This still surfaces the "over-spend tokens to force pass-rate" degeneracy (design §4) without a
  meaningful dollar cost. *(Alternative if a dollar-denominated metric is wanted: introduce a
  non-ledger **shadow** compute price used only for analysis, never charged — deferred unless the
  token-normalized canary proves insufficient.)*
- Keep `tool_call_cost = 0` so this metering event never charges the ledger (LLM convention).
- **Test (S12, supporting):** a rollout's scored trace has non-zero `brain_tokens_in/out` and the
  cumulative counts in the last `brain_metered` event equal the per-turn sums; and the
  token-normalized canary is computed and finite for a non-empty rollout.

### 7.5 Two rollouts scored clean (M0 gates)
- **Scripted deterministic policy** (no model): a fixed action sequence driving the adapter →
  `finalize()` → `score_trace`, end-to-end, **no integration errors**. This isolates the
  wrapper from model variance.
- **One zero-shot Qwen rollout** through the same wrapper against a vLLM endpoint (can reuse the
  M0.5 endpoint), scored end-to-end.

---

## 8. Milestone 0.5 — Throughput smoke (gate)

Implements [design §6.4, §7](./RL_TECHNICAL_DESIGN.md). Runs *after* §7 works, on the intended
node topology.

### 8.1 Method
1. Stand up the intended topology: one vLLM **inference node** serving `Qwen3-4B-Instruct-2507`
   + LoRA-capable, one **trainer** context (Prime Lab, or self-managed pod fallback — same
   OpenAI-compatible endpoint either way).
2. Run **N ≥ 64** C1-horizon (`horizon_days=2`) episodes through the wrapper (no gradient — pure
   rollout generation), `temperature=1.0`, `max_tokens=512`/turn.
3. Measure and record:
   - **turns/sec** (per-rollout and aggregate at the cap'd concurrency vLLM sustains),
   - **mean turns/episode** and the distribution (C1 expected ~30–60 turns),
   - **wall-clock per episode** and **$ per 1k C1 episodes** at the node's hourly rate,
   - GPU-hours implied for a candidate `max_steps × batch_size × G`.
4. **Back out** an affordable `max_steps × batch_size × G` (GRPO group `G≥8`, design §6.2) from
   the measured $/episode and the < ~$300 envelope — *not* from a guess.
5. Re-derive the §6.4 budget split (~60% C1–C2 / ~20% C3+sweep / ~20% baselines+eval) against
   the real number. If throughput is far worse than hoped, **cut scope before committing**
   (fewer steps, smaller batch, 2-point α check only, cap C3).

### 8.2 Artifacts
Write `rl/artifacts/throughput_smoke.md` with the raw numbers, the node spec + hourly rate, the
derived affordable config, and the re-derived budget. This is a direct input to D1 and the
go/no-go for Milestone 1+.

---

## 9. Exit-criterion rubric (scored)

Both milestones pass only when **every Gate item is ✅**. Quality items are tracked but
non-blocking. A single ❌ Gate item blocks progression to Milestone 1.

### M0 — Env contract (gate)

| # | Gate criterion | Pass test | Status |
|---|---|---|---|
| G1 | `breach_fee_frac` added to `EnvConfig`, defaults `0`; **provenance emitted only when nonzero** (§3.1) | Existing suite green; default-0 trace **byte-identical** (provenance unchanged) & score/event-order identical | ✅ local tests |
| G2 | `finalize()` breach sweep in exact §3.2 order | **S11** (a)–(f) green | ✅ local tests |
| G3 | `"train"` recognized in `parse_seeds` + `seed_split_label`; `seeds_train.txt` exists & disjoint | seed-split test green (`train∩test=∅`, label=`"train"`) | ✅ local tests |
| G4 | `$0` pricing entry on **canonical id** `qwen3-4b-instruct` + alias resolves **canonical→wire** (§5) | pricing test: no `KeyError`, `brain_cost==0`; `resolve_model_name("qwen3-4b-instruct") == "Qwen/Qwen3-4B-Instruct-2507"` with the alias env var set | ✅ local tests |
| G5 | Delivered-gated pricing/selection reward helpers | **S10** green (accept-no-deliver ⇒ 0 shaping) | ✅ local tests |
| G6 | Shaping can never outvote profit | **S6** green across all training seeds | ✅ local tests |
| G7 | Toy one-tool `StatefulToolEnv` proves hidden-arg injection, schema stripping, and token-usage capture hook | model never sees `_adapter`; tool dispatches & returns observation; synthetic metering record has cumulative usage | ✅ local tests |
| G8 | Scripted deterministic rollout: drive → `finalize()` → `score_trace` | runs end-to-end, **no integration errors**; scorecard fields present | ✅ local tests |
| G9 | One zero-shot Qwen rollout through the wrapper, scored | runs → finalize → score, no integration errors | ✅ live Prime smoke: hosted `qwen/qwen3-8b`, seed 1000, trace/result saved under `rl/artifacts/` |
| G10 | `finalize()` invoked before scoring via `@vf.cleanup`, once per rollout | trace closed + terminal reason stamped pre-score; no double-finalize | ✅ local tests |
| G11 | `entrepreneur_bench` packaged/discoverable; `rl/tests` collected (§2.1) | `uv run python -c "import entrepreneur_bench"` ok; `uv run pytest rl/tests` collects | ✅ local tests + loader |
| G12 | Scorecard pricing/selection math extracted to shared helpers, single source of truth (§6) | existing scorecard tests green (numbers unchanged); helpers imported by both scorecard & `rewards.py` | ✅ local tests |
| G13 | Verifiers policy tokens metered into the trace via `brain_metered` (§7.4); compute canary token-normalized | **S12** green: non-zero `brain_tokens_in/out`, cumulative == per-turn sums, token-normalized canary finite | ✅ local tests |
| G14 | `build_reward_context(trace_path)` exposes job-level facts for delivered-gating (§6.0) | returns normalized `delivered_job_ids`/`accepted_facts`/`good_ids`/etc.; rewards consume it without re-reconstructing the trace | ✅ local tests |

| Quality (non-blocking) | Target |
|---|---|
| S1–S5, S7–S9 reward tests authored & green | full suite passes in CI |
| Canary extraction (S9) covers every §4 metric | all canaries present & typed |
| First-failure traces captured under `rl/artifacts/` as they occur | D1 input |

### M0.5 — Throughput smoke (gate)

| # | Gate criterion | Pass test | Status |
|---|---|---|---|
| T1 | ≥64 C1 episodes run through the wrapper on intended topology | run completes, traces scored | ✅ 64 clean C1 episodes on Prime MassedCompute 1x A6000_48GB with vLLM/Qwen3-4B; 64 result rows + 64 traces |
| T2 | turns/sec, mean turns/episode, $/1k C1 episodes measured & recorded | numbers in `rl/artifacts/throughput_smoke.md` | ✅ measured: 0.713 turns/sec, 25.45 mean turns/episode, $5.36 per 1k C1 episodes |
| T3 | Affordable `max_steps × batch_size × G` (G≥8) backed out from measurement | config derived from data, not guess | ✅ measured budget supports about 55,989 C1 episodes; at G=8, `max_steps x batch_size <= 6998` before other slices |
| T4 | §6.4 budget split re-derived vs. < ~$300 envelope; scope cut if needed | explicit go/no-go decision recorded | ✅ go for M1 with 60/20/20 split; rerun smoke if topology/model/parser/concurrency changes |

**Combined sign-off:** all M0 G1–G14 ✅ **and** M0.5 T1–T4 ✅. Proceed to Milestone 1
(baseline & model pick) using the measured throughput and budget envelope in
`rl/artifacts/throughput_smoke.md`.

### Resolved review decisions (codex passes, 2026-06-29)
- **Module name:** canonical top-level `entrepreneur_bench` (on disk under `rl/`), not `rl.entrepreneur_bench` — stable for the training TOML and D3 Hub publication. (§2, G11)
- **Token metering:** **yes** — wrapper emits `brain_metered` so Solvent compute fields reflect policy-generation tokens. (§7.4, G13)
- **Packaging:** `find` where/include + pytest testpaths/pythonpath fix the discovery gap. (§2.1, G11)
- **Reward helpers:** scorecard `_pricing`/`_selection` refactored into shared parameterized helpers (single source of truth), not "reused" as private methods. (§6, G12)
- **Model-id drift:** one canonical priced id + explicit wire alias. (§5, G4)
- **Byte-identity:** breach provenance emitted only when `breach_fee_frac != 0`. (§3.1, G1)
- **Reward data source (review 2):** public `Scorecard` lacks job-level facts; add `build_reward_context(trace_path)` exposing normalized `delivered_job_ids`/`accepted_facts`/`good_ids`/`delivery_menu` from the builder. (§6.0, G14)
- **Alias direction (review 2):** resolution is **canonical→wire** (`resolve_model_name(canonical)` → wire); price/meter by canonical. (§5, G4)
- **Per-compute-dollar (review 2):** `fraction_of_optimal_per_compute_dollar` is `None` at `$0` cost; compute-efficiency canary is **token-normalized**, computed in the W&B layer. (§7.4)
- **Delivered set (review 3):** delivered-gating uses `ScorecardBuilder._delivery_attempts()` /
  normalized `delivered_job_ids`, not `facts.submissions`, so tool-mediated `deliver` rollouts are
  counted. (§6.0, G14)
- **Token usage hook (review 3):** toy env must prove the Verifiers generation usage hook before the
  full wrapper relies on it for `brain_metered`. (§7.1, §7.4, G7/G13)
- **Reward decomposition (review 4):** `build_rubric()` keeps the combined `terminal_reward` as the
  only weighted optimization target, and registers each sub-reward term as a zero-weight metric so
  W&B can monitor shaped-vs-true divergence without changing the reward.
- **G9 provenance (review 4):** the hosted Prime smoke used `qwen/qwen3-8b` because hosted 4B was
  unavailable; M1 baselines must set `brain_model`/provenance to the actually served model, not the
  M0 canonical 4B pricing id.
- **Tool schema surface (review 4):** RL wrapper schema filtering uses the public
  `schemas_for_delivery_mode()` helper, not `ToolAdapter` private attributes.
- **Breach-fee EV assertion (review 4):** add a standalone delivered-EV-greater-than-breached-EV
  assertion if the chosen `breach_fee_frac` is tuned beyond the M0 value.

---

## 10. Suggested task order (build sequence)

1. §2 env setup + **packaging (G11)** + pin `verifiers`, confirm existing suite green.
2. §7.1 **toy env** (resolve the Verifiers-API uncertainty first — cheapest de-risk).
3. §3 breach mechanic + S11; §4 seed code + test; §5 canonical-id + pricing + alias test (independent, parallelizable).
4. §6 **scorecard helper refactor (G12)** + **`build_reward_context` (G14)** → delivered-gated reward helpers + S6/S10 (depends on §3 for breach-aware scoring).
5. §7.2–7.5 full wrapper + token metering (G13/S12) + scripted rollout (G8) + Qwen rollout (G9/G10).
6. §8 throughput smoke (depends on a working wrapper + vLLM endpoint).
7. Score the §9 rubric; record artifacts; go/no-go.

## 11. Risks specific to M0/0.5

| Risk | Mitigation |
|---|---|
| `verifiers` API (`update_tool_args`/`setup_state`/`@vf.cleanup`) differs from design schematic | Toy env (§7.1) proves it before the full wrapper; pin the exact version (§2.4). |
| Breach sweep double-debits / reorders trace on second `finalize()` | Idempotence guard first; S11 (b)+(c) lock it. |
| vLLM build rejects tool-calling for Qwen3-Instruct, or `max_completion_tokens` quirk | Inject a custom client / use bare-endpoint payload keys (design §1.1); test in G9. |
| Prime Lab GPU scarcity blocks M0.5 | Self-managed pod fallback, same OpenAI-compatible endpoint (design §7 cross-cutting). |
| Throughput far worse than the <$300 hypothesis | T4 forces an explicit scope cut *before* committing to Milestone 1+. |
