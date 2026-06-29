# Entrepreneur-Bench RL — Technical Design

**Status:** Draft v1 · **Date:** 2026-06-28
**Companion doc:** [`RL_PRD.md`](./RL_PRD.md)
**Stack:** Verifiers (env wrapper) · Prime-RL / GRPO (training) · LoRA · vLLM (policy serving) · Prime Lab (hosted GPUs) · W&B (tracking) · Solvent CLI (rollout inspection)

> All `file:line` references are from an audit of the repo at the time of writing. Verify against current code before relying on a specific line number — the structure is stable, the exact lines may drift.

---

## 1. Codebase audit (answers to the open questions)

### 1.1 Policy serving — can Solvent point at a local vLLM endpoint? **Yes, via client injection (cleanest) or env vars (hack).**

The provider-neutral client is a `ModelClient` Protocol (`solvent/harness/providers/base.py:12-37`) with a router `client_for_model` (`solvent/harness/model_client.py:47-59`) that selects a provider by **hardcoded model-name prefix** (`claude-` → Anthropic, `gemini-` → Google, else an OpenAI-compatible client keyed on the first hyphen segment: `gpt`, `deepseek`, `minimax`, `kimi`, `glm`, `grok`). There's an OpenRouter fallback (`openai_compat.py:58-65`) and OpenRouter-only payload keys are correctly suppressed for bare endpoints (`openai_compat.py:78,107-110`), so they won't break vLLM.

Three ways to reach a local vLLM server at `http://localhost:8000/v1`:
- **Env-var redirect (no code):** `OPENAI_BASE_URL=http://localhost:8000/v1`, `OPENAI_API_KEY=sk-anything`, name the run model `gpt-…` and optionally `SOLVENT_MODEL_ALIAS_<MODEL>=<vllm-model-id>` to set the wire model name (`openai_compat.py:54,130`, `base.py:40-46`). Caveat: `gpt-`-prefixed models send `max_completion_tokens` instead of `max_tokens` (`openai_compat.py:138-141`), which some vLLM builds reject.
- **★ Recommended — inject a custom client:** `LLMHarness(model=<priced-id>, client=YourVLLMClient(...))` (`llm.py:52` accepts an injected `client`). This **bypasses all prefix-routing and the `max_completion_tokens` quirk**. For our RL wrapper we don't even use `LLMHarness` — Verifiers owns generation (§2) — but this is how any Solvent-side eval of the local model is wired.
- The one remaining coupling either way: every turn calls `brain_cost(self.model, usage)` (`llm.py:165`) → `price_for_model` → **`KeyError` if the model id isn't in the pricing table** (`solvent/env/pricing.py:328-332`). **Action:** add a `$0` pricing entry for our model id (e.g. `qwen3-4b-instruct`), or name the run after an existing priced id. This is the #1 footgun.

**Blocker list for RL wrapping** (from the audit): (1) pricing-table `KeyError` on unknown model — must add a $0 entry; (2) no first-class `--base-url` flag — use client injection; (3) the legacy `LLMHarness` history format replays turns as serialized-JSON **user** messages, *not* native assistant/tool messages (`openai_compat.py:126-129`) — **we do not reuse it for RL** (see §2.1); (4) the cost **budget guard** truncates episodes by $-spend (`llm.py:186-198`, `experiment/runner.py:222-231`) — the cause of the documented day-6–9 truncation on "14-day" runs — **must be disabled** (`budget_limit=None`) for RL.

### 1.2 Episode decomposition — how many model turns per episode?

The agent loop is event-driven, **not** tick-proportional. One model turn = one tool call. Tools available (`tool_api.py:31-98`): `list_jobs`, `inspect_job`, `clarify`, `bid` (single counter-offer), `accept`, `decline`, `submit` (direct mode) / `deliver` + `list_models` (tool-mediated mode), `respond` (manipulation), `check_balance`, `list_in_progress`, `end_tick`, `advance_to_next_event`, and a `mem_*` scratchpad. The observation each turn is a JSON dict: `tick`, `horizon`, `business_time`, `days_remaining`, `balance`, `available_jobs` (public briefs only — hidden `internal_difficulty`/`reservation_price` are stripped, `models.py:98-108`), `awaiting_decision`, `delivery_models`, `in_progress` (`tool_api.py:121-152`).

**Turn count drivers** (`experiment/runner.py:198-212`): `expected_jobs = round(arrival_rate_per_day * horizon_min/1440)`, `max_turns = expected_jobs*10 + 200`. For the 30-day `full_v06` (~4.5 jobs/day → ~135 jobs) the YAML pins `max_turns: 1600`. **This is a very long multi-turn episode — the core reason for the short→long curriculum.** A 1–2-day horizon yields ~1–9 jobs → tens of turns, which is what early training runs on.

Sim time is decoupled from turns: in business-time mode `end_tick` aliases to `advance_to_next_event` and jumps the clock straight to the next arrival/expiry (`env.py:625-645`); most tool calls consume no sim time, only `tool_call_cost`.

### 1.3 Scorecard & dense-regret availability — **PARTIAL, leaning AVAILABLE.**

Scoring is **pure trace replay**: an episode writes a JSONL event trace; `score_trace` (`scoring/scorecard.py:38`) reconstructs the market from the trace seed and recomputes everything. Net profit is read from the ledger: `net_revenue = end_balance − start_balance` (`scoring/events.py:321,303`). A lower-variance **control-variate** net also exists: `_expected_net_revenue` (`scorecard.py:285-306`) replaces each delivery's realized 0/1 with the menu `pass_prob` — **this is our preferred terminal RL target** (§3).

The "four-capability" regret maps to **three explicit terminal scalars** plus delivery pass-rate and a coherence penalty:
- **Pricing** — `pricing_regret = Σ_{good accepted} (reservation_price − contract_price)` (`scorecard.py:251-283`). **Per-decision decomposable** (loop body is per accepted job).
- **Tool selection** — `oracle_tool_regret = Σ_{attempts} (best_model_EV − chosen_model_EV)` (`scorecard.py:372-397`). **Per-decision decomposable** (loop body is per delivery attempt).
- **Problem selection** — `selection_regret = missed_good + chased_decoys` (`scorecard.py:224-249`). `chased_decoys` is per-job; `missed_good` is a set-level DP difference (`optimal_value − chosen_value`) — only naturally aggregate, needs marginal attribution for full density.
- **"Resource allocation"** is **not a standalone field** — scheduling/time-feasibility is folded into the selection DP. We treat it as part of selection rather than inventing a term.

**Verdict:** dense per-decision reward is directly available for **pricing** and **tool-selection**; **selection** needs a marginal-attribution shim (§3.4). Aggregate terminal regret is available for all three for free.

**Two findings that shape the reward (both verified):**
- **Acceptance, not delivery, earns regret credit.** `selection_regret` uses `chosen = accepted_jobs` (`scorecard.py:225`) and `pricing_regret` iterates `accepted` (`scorecard.py:252`). A job accepted (and well-priced) but never delivered *lowers* both regrets at zero payout — the **accept-without-deliver reward-hack loophole**. We delivery-gate the shaping and add a sim breach fee (§3.4).
- **`coherence_penalty` is observed-only.** It does **not** feed `gross_score` (`scorecard.py:364-370`), `net_revenue`, or any `fraction_of_*` optimum — it is a side column in `findings`/`compare`/CLI only (`leaderboard.py:136`, `compare.py:37`, `cli/main.py:495`). We drop it from all scored/rewarded quantities and keep the raw counts as diagnostics (§3.4, §4).

### 1.4 Delivery menu — **resolved without real model calls. YES.**

`env.deliver()` → `DeliveryMenu.resolve()` (`delivery/menu.py:92-103`): `pass_prob` is a dict lookup into a frozen profile (`menu_data/menu_v0_4.json` — 3 tools × 2 task types × 3 difficulties), outcome = `random.Random(draw_key).random() < pass_prob`, price = static public price. **No network, no nested LLM.** `characterize --generate-menu` only replays *stub* harnesses to attach profiles; it does not call real models. **Implication: RL rollouts pay only for the policy's own generations.**

### 1.5 Reference policies (4) & determinism.

All in `scoring/optimal.py`, computed post-hoc from the reconstructed market — closed-form/DP, no simulation of the LLM:
- **omniscient** (`optimal.py:23-34`) — sees hidden difficulty/reservation/full future stream; exact schedule DP (≤16 jobs) else relaxation upper bound.
- **realizable** (`optimal.py:37-58`) — same, but caps capability at the agent's observed `average_verify_score`.
- **threshold-policy** (`optimal.py:61-111`) — **online greedy, no lookahead, beatable** (the honest baseline).
- **joint-optimum** (`optimal.py:114-168`) — omniscient + per-job model downgrade.

Scorecard reports `fraction_of_{omniscient,realizable,threshold_policy,joint}_optimum` (`scorecard.py:98-124`). **Determinism:** every random draw is a string-keyed `random.Random` (arrivals `market.py:111`, market params `market.py:77`, task/difficulty `market.py:147`, delivery `menu.py:95`); **no bare `random.*`**. Given seed+config+model outputs, an episode is bit-reproducible. The only stochastic element in RL is the policy's own sampling — exactly what GRPO groups over.

### 1.6 Seed splits & canary metrics (already logged).

Splits are plain text files (`solvent/configs/seeds_dev.txt` = `40–44`, `seeds_test.txt` = `140–144`), resolved by `cli_seed.py:6-29`, tagged onto each trace as `seed_split_label` (`env.py:93`, `scorecard.py:110`). **Only seed-level splits, 5 each, no job-type holdout.** We expand the dev pool (§5).

The scorecard already computes a rich behavioral metric set we reuse as **canaries** (§4): selection `precision/recall/decoys_chosen/good_chosen` (`scorecard.py:241-249`); pricing `counter_accepts/floor_accepts/surplus_left/average_price_ratio` (`scorecard.py:275-283`); delivery `pass_rate/average_verify_score` (`scorecard.py:317-326`); coherence `dropped_jobs/invalid_actions/action_loops/undelivered_in_progress` (`scorecard.py:342-362`); compute `brain_tokens/brain_cost/fraction_of_optimal_per_compute_dollar` (`scorecard.py:415-425`); and from the leaderboard `days_until_insolvent`, `horizon_fraction_active`, insolvency reason (`findings/leaderboard.py:367-388`).

---

## 2. RL environment wrapper design

### 2.1 Shape: a Verifiers `StatefulToolEnv` driving Solvent's `Environment` directly

We do **not** reuse `LLMHarness` for rollouts (its history format replays turns as serialized-JSON user messages — wrong for GRPO, which must train on the policy's own native assistant/tool-call tokens). Instead Verifiers owns generation, and our wrapper drives Solvent's `Environment` + `ToolAdapter` as the state machine.

Solvent already exposes tool **schemas** (`TOOL_SCHEMAS`, `solvent/env/tool_api.py:31-98`) and a **dispatch** (`ToolAdapter.dispatch`, `solvent/env/tool_api.py:154-200`) that **mutates an `Environment`**. Because the tools carry per-rollout mutable state, the correct Verifiers base is **`StatefulToolEnv`, not `ToolEnv`** — Prime's docs are explicit that `ToolEnv` is for *stateless/read-only* tools, while `StatefulToolEnv` holds per-rollout resources in the `state` dict (via `setup_state()`) and injects them into tool calls through **hidden arguments** (`update_tool_args()`), kept out of the model's tool schema. Our per-rollout `Environment`/`ToolAdapter` are exactly those hidden resources.

> **The block below is schematic** — it shows the integration shape, not compile-ready code. Real `EnvConfig` construction must supply all seven required fields (`seed`, `config_id`, `start_balance`, `horizon_ticks`, `overhead_per_tick`, `tool_call_cost`, `trace_path`), use `Decimal`/`Path` types (it's `frozen=True`), set `seed_split` for provenance, and `breach_fee_frac` is a **new field we add** (§3.4). See `solvent/env/models.py:9-48` for the real signature.

```python
# entrepreneur_bench/environment.py  (the Verifiers env module) — SCHEMATIC
import verifiers as vf
from decimal import Decimal
from pathlib import Path
from solvent.env.env import Environment
from solvent.env.models import EnvConfig
from solvent.env.tool_api import ToolAdapter, TOOL_SCHEMAS   # NOTE: solvent.env, not solvent.harness

def load_environment(horizon_days=2, delivery_mode="tool_mediated",
                     job_ttl=True, split="train", breach_fee_frac=0.25, **kw) -> vf.Environment:
    dataset = build_seed_dataset(split, horizon_days)      # one row per seed (§5)
    rubric  = build_rubric()                               # §3
    return EntrepreneurEnv(dataset=dataset, rubric=rubric, parser=vf.Parser(),
                           horizon_days=horizon_days, job_ttl=job_ttl,
                           delivery_mode=delivery_mode, breach_fee_frac=breach_fee_frac, **kw)

class EntrepreneurEnv(vf.StatefulToolEnv):           # ← StatefulToolEnv (per-rollout mutable env)
    async def setup_state(self, state):
        horizon_min = self.horizon_days * 1440
        cfg = EnvConfig(
            seed=state["info"]["seed"],
            config_id=state["info"]["config_id"],            # required
            start_balance=Decimal("1000.00"),                # Decimal, not str
            horizon_ticks=horizon_min,                       # required; business-time → minutes
            horizon_minutes=horizon_min,                     # enables business-time mode
            overhead_per_tick=Decimal("0"),                  # required
            overhead_per_minute=Decimal("0.006944"),         # 30-day overhead (runner.py:163-189)
            tool_call_cost=Decimal("0"),                     # required; LLM convention
            trace_path=state["trace_path"],                  # required; per-rollout JSONL sink
            delivery_mode=self.delivery_mode,
            job_ttl_minutes=(min(1440, horizon_min) if self.job_ttl else None),
            reputation_enabled=False, work_time_enabled=False,
            breach_fee_frac=Decimal(str(self.breach_fee_frac)),  # NEW field, §3.4
            seed_split=state["info"]["split"],               # provenance label
        )
        env = Environment(cfg)
        state["env"], state["adapter"] = env, ToolAdapter(env)
        await super().setup_state(state)

    def update_tool_args(self, args, state):                 # inject the rollout's adapter (hidden arg)
        return {**args, "_adapter": state["adapter"]}

    async def is_completed(self, messages, state):
        return state["env"].terminated()                     # horizon OR insolvency (env.py:839-845)

# --- The registered tools ARE what mutate Solvent. One factory per Solvent tool name. ---
# StatefulToolEnv calls the tool fn (with the hidden `_adapter` injected by update_tool_args)
# and appends its STRING return value as the tool message — so returning the fresh observation
# is how the model sees new state. There is NO separate env_response dispatch; the tool does it.
def make_solvent_tool(name: str):
    def _tool(_adapter, **arguments):                        # `_adapter` hidden from the model schema
        result = _adapter.dispatch({"name": name, "arguments": arguments})   # the one mutation point
        return json.dumps({"result": result, "observation": _adapter.observe()})
    _tool.__name__ = name
    return _tool   # advertise with the model-facing schema from TOOL_SCHEMAS, mode-gated via adapter.schemas()
```

**Dispatch contract (the thing to get right first):** every Solvent state mutation goes through exactly one call — `ToolAdapter.dispatch({"name", "arguments"})` (`tool_api.py:154`) — and it lives **inside the registered tool function**, not in `env_response`. The model emits a native tool call → `StatefulToolEnv` injects the hidden `_adapter` → the tool fn dispatches and returns the fresh `observe()` as the tool message. Tools are advertised to the model using `TOOL_SCHEMAS` filtered by `adapter.schemas()` (mode-gating, `tool_api.py:115-119`), with `_adapter` stripped from the advertised schema.

Notes:
- **`finalize()` before scoring (required).** When `is_completed` fires, the wrapper must call `state["env"].finalize()` (`env.py:691`, returns `EpisodeSummary` and stamps the terminal reason) **before** the Rubric reads/`score_trace`s the JSONL at `trace_path`. Do this in a `@vf.cleanup` hook so it runs once per rollout.
- **Each Solvent tool is registered as a hidden-arg `StatefulToolEnv` tool** so the model emits native function calls while the per-rollout `Environment` is injected out-of-band. Mode-gating (`submit` in direct; `deliver`/`list_models` in tool-mediated) follows `ToolAdapter.schemas` (`tool_api.py:115-119`).
- **Turn cap:** Verifiers' base `max_turns` plus our stop condition. Set `max_turns` from Solvent's formula (`expected_jobs*10 + 200`, §1.2) so short-horizon episodes cap at ~tens of turns.
- **Disable the budget guard** — we never construct `LLMHarness`, so the $-budget truncation (`llm.py:186-198`) doesn't apply. Episodes end only on horizon/insolvency. (Resolves the day-6–9 truncation issue for RL.)
- **Invalid model output** is handled by `ToolAdapter.dispatch` returning `{"ok": False, "error": …}` and charging `tool_call_cost` (`tool_api.py:154-200`) — fed back as the tool message; no crash, no retry. Also a canary source (`invalid_actions`).

### 2.2 Observation / action format

- **Observation:** the `ToolAdapter.observe()` JSON (§1.2), delivered as the system/seed user message on turn 1 and as tool-result messages thereafter. Hidden fields stay hidden (already stripped by `to_public()`).
- **Action:** native OpenAI-style tool calls (vLLM serves tool-calling for Qwen3-Instruct). The Verifiers `Parser` extracts the call; malformed calls flow to the `{"ok": False}` path.
- **System prompt:** reuse Solvent's `system_prompt(ablations)` (`harness/prompts.py:30-40`) so the framing matches the hosted-model baselines (apples-to-apples).

### 2.3 Context management

Solvent's `ContextManager` (`harness/context.py`) is bypassed — Verifiers manages the message list. For short-horizon training, full history fits. As the curriculum lengthens horizons, cap context with Verifiers' own truncation; long 30-day episodes are **eval-only** (no gradients), so context cost there is bounded by a single forward pass per turn, not by training memory.

### 2.4 Episode boundaries & reward surfacing

- **Boundary:** `setup_state` builds a fresh seeded `Environment`; episode ends on `env.terminated()` or `max_turns`.
- **Reward surfacing:** at termination the wrapper writes the Solvent JSONL trace and calls `score_trace` to get the full scorecard; the **terminal reward** is read by the Rubric reward functions from `state` (§3). **Dense shaping** is attached per-turn via `add_trajectory_step` at the moment of a pricing/tool decision (§3.3).

---

## 3. Reward design

### 3.1 Hybrid: dominant terminal true-profit + small dense regret shaping

Per the locked decision: a **dominant terminal term** that is ground-truth profit, plus **small dense shaping** from the already-computed per-decision regret. The terminal term must dominate so shaping can never override ground truth.

**Terminal reward (primary):**
> `R_terminal = expected_net_revenue` (the control-variate net from `scorecard.py:285-306`), with GRPO **grouping by seed**.

Why this exact choice:
- **Control variate** removes delivery RNG luck (replaces realized 0/1 with `pass_prob`) → lower-variance advantage than raw `net_revenue`, without biasing toward profit.
- **Grouping by seed** (each GRPO group = G rollouts of the *same* episode seed) means the omniscient optimum is constant within a group, so the group-relative advantage reflects **pure policy skill**, not seed difficulty. This is the single biggest variance reduction available and it's free given the env's determinism (§1.5).
- Scale: divide by a constant (e.g. `/1000`) for numerical comfort; GRPO's within-group normalization makes absolute scale secondary.

### 3.2 Shaping terms (small, bounded, terminal-dominated)

Two equivalent ways to add the regret signal; we use **(A) terminal shaping first** (simplest, hardest to game), and only add **(B) per-step shaping** if credit assignment proves too slow:

**(A) Terminal shaping** — extra Rubric reward functions with small weights:
```python
rubric = vf.Rubric(
  funcs   = [r_expected_net, r_pricing_neg_regret, r_tool_neg_regret, r_selection_neg_regret,
             r_solvency],
  weights = [1.0,            0.15,                 0.15,             0.15,
             0.10],
)
```
where each `r_*_neg_regret = −regret/scale` from the scorecard scalars (§1.3) and `r_solvency` penalizes insolvency-termination. **No coherence term** — see §3.4 for why coherence is observed, never scored or rewarded. Weights chosen so Σ|shaping| ≪ the dynamic range of `R_terminal` on a typical seed — verified empirically before any real run (§3.5 test S6).

> **Delivery-gating (required, see §3.4):** `pricing_regret` and `selection_regret` as scored today credit **acceptance**, not delivery (`scorecard.py:225,252`), so a policy can collect the negative-regret shaping by accepting good jobs it never delivers. The shaping reward functions therefore compute regret over **delivered** jobs only, so acceptance alone earns no shaping credit. This closes the loophole on the reward side; §3.4 closes it on the sim side too.

**(B) Per-step shaping** (optional, via `add_trajectory_step`): attach the per-decision regret **at the `deliver` turn**, not the `accept` turn — `−(reservation − contract)/scale` for pricing and `−(best_EV − chosen_EV)/scale` for tool, both keyed to the job being delivered. Pricing and tool regret are cleanly per-decision (§1.3). For **selection**, attribute `missed_good` marginally — credit/charge each *delivered* job by its marginal contribution to the schedule-DP optimum (the shim noted in §1.3); until that shim lands, selection shaping stays terminal-only. Gating on delivery (not acceptance) is what makes per-step shaping safe.

> **Reward-hacking guardrail baked into the design:** dense shaping that rewards "matching the reference" can teach gaming the reference gap rather than making money. The terminal `expected_net_revenue` dominating, delivery-gating (§3.4), plus the canary that watches **shaped-reward-up while true-profit-flat** (§4), are the explicit defenses — and are themselves the instruments that would *surface* such a hack for the writeup.

### 3.3 Mapping summary (scorecard → reward)

| Reward component | Source | Sign | Weight |
|---|---|---|---|
| Terminal profit | `expected_net_revenue` (`scorecard.py:285`) — used directly | + | 1.0 (dominant) |
| Pricing | **new RL helper:** surplus-left regret over *delivered* jobs (formula from `_pricing`, `scorecard.py:251-283`, but gated to delivered) | − | ~0.15 |
| Tool selection | `oracle_tool_regret` (`scorecard.py:372-397`) — already per-delivery-attempt, used directly | − | ~0.15 |
| Problem selection | **new RL helper:** selection regret over *delivered* jobs (formula from `_selection`, `scorecard.py:224-249`, but gated to delivered) | − | ~0.15 |
| Solvency | `terminated_reason == insolvent` (`env.py:842`) | − | ~0.10 |

*Important: `pricing_regret` and `selection_regret` **as the scorecard emits them today are computed over *accepted* jobs** (`scorecard.py:225,252`), which is the accept-without-deliver loophole (§3.4). The reward therefore uses **new RL helper functions** that apply the same formulas but over *delivered* jobs — it does not read the existing scorecard fields directly. `oracle_tool_regret` is already keyed to delivery attempts, so it's used as-is. `coherence_penalty` is deliberately **absent** — observed, never scored or rewarded (§3.4).*

### 3.4 Commitment integrity & the "observed, not scored" rule for coherence

**Coherence is observed, never scored or rewarded.** The composite `coherence_penalty` (`scorecard.py:342-362`) is a hand-weighted sum of `dropped_jobs`, `duplicate_bids`, `invalid_actions`, `action_loops`. We exclude it from every scored/rewarded quantity because (a) it double-counts consequences profit + regret already price (a dropped job is forgone revenue; an invalid action already burns `tool_call_cost` and a turn toward `max_turns`); (b) its weights are arbitrary and dilute the regret-against-reference thesis; (c) in an RL reward it is a shaping-hack magnet that buys nothing the environment's own economics don't already enforce. **Blast radius (verified):** `coherence_penalty` does **not** feed `gross_score` (`scorecard.py:364-370`), `net_revenue`, or any `fraction_of_*` optimum — it only appears as a side column in `findings`/`compare`/CLI (`findings/leaderboard.py:136`, `scoring/compare.py:37`, `cli/main.py:495`). So removing it from scored output is a *reporting* change, not a change to any core scored number. The raw counts survive as **descriptive diagnostics + canaries** (§4); long-horizon incoherence is a **qualitative findings section** in the writeup, not a metric column.

**The accept-without-deliver loophole (a real reward hack to defend against).** Accepting a job is a *commitment* with no payout until delivery, but as scored today both `selection_regret` (`chosen = accepted_jobs`, `scorecard.py:225`) and `pricing_regret` (iterates `accepted`, `scorecard.py:252`) give credit at **acceptance**. So a policy can drive both shaping terms down by accepting (and well-pricing) every good job while **never delivering** — dodging delivery cost and pass-rate risk. Terminal profit stays ~0, so this is precisely a **shaped-reward-up / true-profit-flat divergence**. The direct measure is `dropped_jobs = accepted − submitted` (= `undelivered_in_progress`, `scorecard.py:356,359`). We treat commitment-breach as a **first-class failure** on three layers:

1. **Reward side (§3.2):** delivery-gate the shaping — pricing/selection/tool regret count **delivered** jobs only, so acceptance earns no shaping credit.
2. **Sim side (new mechanic — *commitment breach fee*):** make breach negative-EV in ground truth, not just unrewarded. A new `EnvConfig` field `breach_fee_frac` (default `0`, preserving existing-trace scoring) debits `breach_fee_frac × contract_price` for every job in `accepted_jobs` that was never delivered, plus a `breach` event for the trace. This flows straight into `expected_net_revenue`, so the **dominant terminal term already prices the hack** — no bespoke reward term needed. It's a lightweight, standalone consequence (distinct from the full `reputation_enabled` dynamic, which stays off).

   **Assess breach at `finalize()`, not "at expiry" — important Solvent semantics.** Once a job is accepted it is **resolved** for clock purposes (`_job_resolved` returns true for accepted jobs, `env.py:672-673`), so `next_event_time()` never emits its expiry as an event (`env.py:683-685`) — an accepted job carries *no active expiry* and simply sits until delivered or horizon. There is therefore no mid-episode "expiry" hook to attach to; the correct, deterministic hook is **`finalize()`** (`env.py:691`), which already runs once at episode end before scoring (§2.1). (If we ever want *mid-episode* breach pressure we'd have to track accepted-undelivered expiry separately and inject those times into `next_event_time` — explicitly **out of scope**; since our RL reward is terminal anyway, end-of-episode assessment is sufficient.)

   **Exact `finalize()` ordering (gets the terminal reason, trace contract, and idempotence right).** Current `finalize()` (`env.py:691-711`) does: return cached summary if present → set `terminated_reason` → emit `terminated` → build `EpisodeSummary` from `ledger.balance` → `trace.close()`. Keep the cached-summary guard first; the breach sweep must run **after** `if self._summary is not None: return self._summary` and **before** any terminal event/summary work. Otherwise a second `finalize()` call could double-debit breach fees and emit duplicate `breach` rows. Required order:
   1. **Idempotence guard first:** if `self._summary is not None`, return immediately with no new events or debits.
   2. **Breach sweep next:** for each job in `accepted_jobs − delivered`, `ledger.debit_burn(breach_fee_frac × contract_price)` and `_emit("breach", …)` — each breach event carries the updated `balance_after`. For v1 this debit is intentionally reported as ordinary burn because `events.py` classifies any positive non-overhead `burn_delta` as `tool_burn` (`events.py:175-179`); headline net revenue is unaffected. If breach diagnostics become important, add a separate `breach_burn` fact later rather than overloading tool-burn charts.
   3. **Re-evaluate insolvency after the debits:** if the ledger is now insolvent, set `terminated_reason = "insolvent"` even if horizon had been reached. **Breach-caused insolvency is real insolvency** — otherwise `terminated_reason` would say `"horizon"` while the final balance is negative (a lying reason), and `r_solvency` would miss it. *(Decision: relabel to `"insolvent"` for simplicity so `r_solvency` and existing consumers Just Work; if we later want to preserve the horizon-vs-breach distinction for analysis, use a distinct reason string that `r_solvency` also matches. Flagged, not blocking.)*
   4. **Emit the single `terminated` event last**, with the (possibly updated) reason — `terminated` is always the final trace event.
   5. **Build `EpisodeSummary`** from the now-post-breach `ledger.balance` (so `net_revenue`/`end_balance` include breach), then `trace.close()`.

   **Trace-order invariant:** any `breach` events precede the one `terminated` event, and `terminated` is always last, so `events[-1]["balance_after"]` is the true post-breach final balance. This keeps replay/viewer assumptions intact. Tested by **S11**. With `breach_fee_frac = 0` no breach events are emitted and the trace remains score-identical and event-order-identical to today; if implementation also omits zero-valued breach metadata from `episode_started`, newly generated zero-breach traces can remain byte-identical too. The fee must stay small enough that a delivered job always beats a breached one in EV (also S11).
3. **Canary side (§4):** `dropped_jobs` / `undelivered`-ratio is logged always-on as both an incoherence signal and the breach reward-hack signature.

### 3.5 Reward unit-test suite (`tests/test_rl_reward.py`)

Because scoring is deterministic trace replay (§1.3), reward tests are pure functions over hand-built / fixture traces. Minimum suite:

- **S1 monotonicity:** more good jobs delivered at fixed price ⇒ `R_terminal` strictly higher.
- **S2 pricing:** contract exactly at `reservation_price` ⇒ `pricing_regret == 0`; lowballing ⇒ positive regret equal to the surplus left.
- **S3 tool selection:** choosing the oracle-best model for a job's hidden difficulty ⇒ `oracle_tool_regret == 0`; over-spending on `tool-pro` for an easy job ⇒ positive regret.
- **S4 selection:** taking a decoy ⇒ `chased_decoys` increments; ignoring a reachable good job ⇒ `missed_good` rises.
- **S5 solvency:** an episode that goes insolvent ⇒ `r_solvency` fires; horizon-terminated ⇒ does not.
- **S6 dominance invariant (the important one):** assert `max Σ|shaping over a seed| < min ΔR_terminal between a do-nothing and a near-optimal policy on that seed`. This is the guarantee that shaping can't outvote profit. Run it across all training seeds as a gate.
- **S7 control-variate sanity:** `expected_net_revenue` equals `net_revenue` in expectation over many delivery draws on a fixed policy/seed (Monte-Carlo tolerance check).
- **S8 determinism:** same seed + same recorded action sequence ⇒ identical scorecard (guards reproducibility).
- **S9 canary extraction:** every canary in §4 is present and correctly typed in the scored output (so live-rollout monitoring can't silently lose a metric).
- **S10 delivery-gating (anti-commitment-hack):** a trace that *accepts* a good job at a great price but never delivers it earns **zero** pricing/selection **shaping** credit (regret computed over delivered jobs), and `dropped_jobs` increments. Contrast: the same job delivered earns the credit. This is the unit test that locks the loophole shut.
- **S11 breach fee (assessed at `finalize()`, with ordering invariants):** with `breach_fee_frac > 0`, a job left in `accepted_jobs` and never delivered is debited `breach_fee_frac × contract_price` **at `finalize()`** and emits a `breach` event; a delivered job is not. Assert: (a) breach fires at episode end regardless of any "expiry" timing (accepted jobs have no active expiry, §3.4); (b) **idempotence** — calling `finalize()` twice emits/debits breach exactly once; (c) **trace order** — all `breach` events precede the single `terminated` event and `terminated` is last, so `events[-1]["balance_after"]` equals the post-breach balance and `net_revenue` includes the debit; (d) **terminal reason** — if breach debits push the ledger insolvent, `terminated_reason` becomes `"insolvent"` (not a stale `"horizon"`) and `r_solvency` fires; (e) delivered-job EV > breached-job EV at the chosen fee (so the fee never discourages legitimate delivery); (f) with `breach_fee_frac = 0`, no breach events emit and traces score-identically/event-order-identically (byte-identically too if zero-valued breach metadata is omitted).

Run S1–S11 in CI **and** against a sample of live training rollouts each run (§8) — the "trust the rubric" lesson.

---

## 4. Reward-hacking detection plan & canary metrics

**Method:** log every canary below to W&B *per rollout* (mean over the batch), alongside the shaped reward and the terminal `expected_net_revenue`. The catch signal is **divergence**: shaped reward (or any sub-metric) trending up while **`fraction_of_omniscient_optimum` and `expected_net_revenue` stay flat or fall** — exactly the Zapier metric-divergence catch. Pair this with periodic `compare --redteam-paired` and human rollout reads (§8).

| Degenerate strategy | Canary signature | Metric (file) |
|---|---|---|
| Only trivial / zero-risk jobs | `recall` ↓, `good_chosen` ↓, `surplus_left` ↑ while `pass_rate` ↑ | `scorecard.py:241-326` |
| Never counter-offer / always lowball | `counter_accepts` → 0, `surplus_left` ↑, `pricing_regret` ↑ | `scorecard.py:275-283` |
| Do-nothing (avoid insolvency) | `chosen_jobs`/`jobs_delivered` → 0, `horizon_fraction_active` ↓, net ≈ −overhead | `leaderboard.py:367-381` |
| **Abandoned commitment** (accept good jobs, never deliver) | `dropped_jobs` ↑ / undelivered-ratio ↑ while pricing+selection shaping looks favorable but `expected_net_revenue` flat | `scorecard.py:356,359` |
| Over-spend tool to force pass-rate | `tool_price_charged` ↑, `oracle_tool_regret` ↑, profit flat | `scorecard.py:372-397` |
| Incoherence / looping (long-horizon, *observed not scored*) | `invalid_actions` ↑, `action_loops` ↑, `duplicate_bids` ↑, `brain_tokens` ↑, `fraction_of_optimal_per_compute_dollar` ↓ | `scorecard.py:342-425` |
| **Game the reference gap** (shaping hack) | `fraction_of_threshold_policy` → ~1 while `fraction_of_omniscient` and `expected_net_revenue` flat | `scorecard.py:98-124` |

> **Observed, not scored.** The incoherence and abandoned-commitment rows are *diagnostics and hack-canaries only* — none of them enters any score or reward (§3.4). They power detection (and the qualitative long-horizon-incoherence findings section), but the policy is never optimized against them. The only place commitment-breach affects an optimized quantity is via the **sim breach fee flowing into `expected_net_revenue`** (§3.4), not via a coherence term.

**Always-on dashboard (W&B):** `expected_net_revenue`, `fraction_of_omniscient`, `fraction_of_threshold`, `recall`, `precision`, `surplus_left`, `counter_accepts`, `pass_rate`, `dropped_jobs`, `invalid_actions`, `action_loops`, `insolvency_rate`, `horizon_fraction_active`, `brain_tokens`, plus the shaped reward and each sub-reward term. Any sustained divergence triggers a rollout read.

**Tooling reuse:** the existing `compare --redteam-paired` (`demo.py:90-188`) and the four reference policies are the divergence instruments — they already exist, so the detection plan is mostly *wiring existing metrics into W&B*, not new code.

---

## 5. Train/eval split & evaluation protocol

### 5.1 Splits

The repo ships 5 dev (`40–44`) / 5 test (`140–144`) seeds — too few for GRPO (which wants many distinct "examples"). Plan:
- **Expand the dev training pool** to ~64–128 seeds in a documented disjoint range (e.g. `1000–1127`), written to a new `seeds_train.txt`. Keep the **original test seeds `140–144` strictly held out**, and optionally add `145–164` for a larger held-out eval. The offset scheme guarantees non-overlap; assert disjointness in a test.
- **Code support is required, not just a file.** `parse_seeds` only special-cases `"dev"`/`"test"` and `seed_split_label` tags everything else `"ad_hoc"` (`cli_seed.py:6-16`). A `seeds_train.txt` path *resolves* (the `path.exists()` branch) but would be mislabeled `ad_hoc` in traces/scorecards. Extend both to recognize `"train"` (add it to the `{"dev","test"}` sets) so split provenance is correct end-to-end. Small change, but it gates eval-integrity reporting.
- **Job-type holdout (optional, stretch):** the market mixes CSV-clean and field-extract task types via config, not split. If time permits, hold out one difficulty band or task mix for an OOD eval; v1 ships seed-level holdout only (matches the AutomationBench public/private analog).

### 5.2 Eval protocol

- **Same config for both sides (apples-to-apples).** The trained policy and its zero-shot baseline are evaluated under an **identical canonical eval config — including the breach fee** (§3.4). So baselines are (re)run with `breach_fee_frac` set to the training value, not the legacy zero-breach config. The breach fee only fires on accepted-undelivered jobs, so for a *competent* policy that delivers what it accepts it is **small but not always zero** — it removes a degenerate exploit without much changing benchmark difficulty for good play. Running both head-to-head numbers under it costs little (Qwen is cheap) and keeps the headline claim clean.
- **Baseline (Milestone 0):** zero-shot Qwen3-4B and 8B through the **same** env wrapper (or Solvent's eval path with the injected vLLM client) on dev+test, at the terminal 30-day horizon, **under the canonical breach config**. Report `expected_net_revenue` and the regret decomposition.
- **Trained policy:** evaluate the **final LoRA checkpoint** on the **held-out test seeds** at 30-day horizon, ≥3 samples/seed (temperature matched to baseline), greedy and sampled both reported.
- **Headline table (D2):** baseline vs trained, per held-out seed and aggregate, columns = `expected_net_revenue`, `fraction_of_omniscient`, `selection_regret`, `pricing_regret`, `oracle_tool_regret`. The DeepSeek-V3.2 ($14.9k) and MiniMax-M3 ($11.9k) rows are **context only — not apples-to-apples** unless replayed/re-run under the breach config, and are footnoted as such (they ran under the original zero-breach config; re-running hosted models costs API money). Note breach is *not* fully inert even for these: DeepSeek's full-v06 traces show `dropped_jobs=0`, but a MiniMax full-v06 scorecard has `dropped_jobs=1`, so its breach-adjusted number would shift slightly. The apples-to-apples *claim* therefore rests **only** on the Qwen-baseline-vs-trained pair, which share the config exactly.
- **Claim discipline:** "beats baseline" requires the test-seed gap to exceed per-seed sample noise (report mean ± std; a simple paired test across seeds). Never tune on test seeds — only dev curves drive decisions.

---

## 6. Training setup

### 6.1 Model

**Decide from Milestone 0.** Default lean: **Qwen3-4B-Instruct** (non-thinking; token efficiency matters because episodes are long multi-turn). Promote to **8B** only if the 4B zero-shot baseline is incoherent (can't emit valid tool calls / never gets reward). Add a `$0` pricing entry for the chosen id (§1.1).

### 6.2 GRPO + LoRA config (Prime-RL TOML)

Prime-RL has first-class Verifiers support: the env is installed as a standalone Python module and referenced from the training TOML. Skeleton (`configs/rl/entrepreneur-bench.toml`):

```toml
model = "Qwen/Qwen3-4B-Instruct-2507"
max_steps = 200
batch_size = 64                # examples (seeds) per step  — tune
rollouts_per_example = 8       # GRPO group size G (same-seed group, §3.1)
env_file = ["../../secrets.env"]

[sampling]
max_tokens = 512               # per turn; episodes are many turns
temperature = 1.0

[lora]
r = 32
alpha = 128                    # Vibe-RL prior: α ≥ 128 mattered; start here
dropout = 0.0

[optim]
lr = 1.5e-6                    # Vibe-RL prior: 1e-6–2e-6 for stability

[[env]]
id = "entrepreneur-bench"      # the pushed Verifiers env
args = { horizon_days = 2, split = "train", delivery_mode = "tool_mediated", job_ttl = true }

[wandb]
project = "entrepreneur-bench-rl"
name = "qwen3-4b-h2-bs64-r8-a128"
```

Hyperparameter priors (from Vibe RL, treated as *starting points* — that was single-turn, ours is long multi-turn): **LoRA α ≥ 128** (lower got stuck in local minima), **LR 1e-6–2e-6**, **batch ~512** helped (128 too small) — but at shoestring budget we start smaller (`batch_size=64`, `G=8` → 512 rollouts/step) and scale only if curves justify it. Group size `G` is the GRPO knob that matters most for credit assignment; keep `G≥8`.

### 6.3 Horizon curriculum

Anneal horizon as the policy stabilizes; each stage is a separate `prime train run` resuming from the prior LoRA checkpoint:

| Stage | `horizon_days` | ~jobs | ~turns | Purpose |
|---|---|---|---|---|
| C0 smoke | 1 | ~1 | ~10–20 | wiring, valid tool-calls, reward signal exists |
| C1 | 2 | ~3–5 | ~30–60 | first real learning curve vs baseline |
| C2 | 5 | ~12–18 | ~120–200 | shaping + reward-hack hunt |
| C3 | 14 | ~50–65 | ~500–700 | scale credit assignment |
| C4 (eval-mostly) | 30 | ~135 | ~1600 | final eval; light/no training |

Most **gradient steps** live in C1–C2 (cheap turns); C3–C4 are short bursts + eval. Curriculum effect (does annealing beat training-at-30-day-from-scratch?) is itself a reported hyperparameter finding.

### 6.4 Compute estimate (shoestring, target < ~$300 — **pending a measured throughput smoke**)

> **The < $300 number is a target, not a derived figure.** Prime-RL's production topology is typically **one inference node (vLLM) + one or more trainer nodes**, with a single combined GPU mainly for debugging. The dollar envelope below is only trustworthy *after* Milestone 0.5 measures real throughput on our actual env. Treat everything here as a hypothesis to falsify, not a budget to commit to. **(Milestone 0.5 is a hard gate — see §7.)**

Rollout cost is dominated by the policy's own generations (delivery is free, §1.4). Provisional envelope (Qwen3-4B + LoRA, vLLM serving):

- **Cost driver:** turns are *sequential within an episode*, so wall-clock ∝ (turns/episode × steps). Short horizons are essential — a C1 episode (~40 turns) is ~40× cheaper in sequential generations than a 30-day one (~1600 turns).
- **Measure first (Milestone 0.5):** before committing batch/step counts, measure **turns/sec** and **$ per 1k short-horizon episodes** on the real env at C1 horizon, on the intended node topology. Back out affordable `max_steps × batch_size × G` from that, not from a guess.
- **Budget split (indicative):** ~60% on C1–C2 training (the learning + hack-hunt story), ~20% on C3 + light sweep, ~20% on Milestone-0 baselines + final 30-day eval.
- **GPU-hours (provisional):** target **~80–150 H100-hours**. At Prime on-demand/community H100 pricing (~$1.5–2.5/hr — **verify current rates + whether a separate inference node is needed**), that's ~$150–300. If the throughput smoke says otherwise, cut scope: 2-point LoRA-α check only, cap C3, fewer steps.
- **Eval is cheap:** final eval = (baseline + trained) × ~25 test rollouts × forward-pass only.

Full-horizon training at scale is **out of budget** regardless. The design keeps gradients on short horizons and uses 30-day for eval. (Mitigates the documented compute-truncation pain too — eval rollouts run to true horizon with no budget guard.)

---

## 7. Experiment milestone sequence & risks

| # | Milestone | Exit criterion | Primary risk → mitigation |
|---|---|---|---|
| **0 — Env contract (gate)** | **The single most de-risking step.** `StatefulToolEnv` wrapper; **a one-tool toy env proving hidden-arg injection + `TOOL_SCHEMAS` advertising work with a `**arguments` signature and the `_adapter` stripped from the model-facing schema** (the one Verifiers-API uncertainty); `breach_fee_frac` env field + `breach` event with the §3.4 `finalize()` ordering; delivery-gated shaping; `seeds_train` split-label code change; `$0` pricing entry; **trace export + `env.finalize()` before scoring**; **one deterministic scripted-policy rollout** (no model) scored end-to-end; **one zero-shot Qwen rollout**; reward tests **S6/S10/S11** green | The toy env confirms the dispatch/schema contract; a scripted policy and a Qwen rollout both run → finalize → score with **no integration errors**; S6/S10/S11 pass | If this is shaky, nothing downstream is trustworthy — fix here before spending a GPU-dollar |
| 0.5 — **Throughput smoke (gate)** | Measure **turns/sec** and **$ per 1k C1-horizon episodes** on the real env + intended node topology (inference node + trainer). Back out affordable `max_steps × batch_size × G` | A measured number replaces the §6.4 guess; budget plan re-derived from it | Throughput far worse than hoped → cut scope (fewer steps, smaller batch, drop sweep) *before* committing |
| 1 | **Baseline & model pick** — Qwen 4B/8B zero-shot on dev+test **under the canonical breach config** (§5.2) | A baseline scorecard exists; model chosen (non-trivial but beatable) | Base model too weak → escalate 4B→8B; can't tool-call → adjust system prompt/parser |
| 2 | **Smoke (C0)** — 1 GRPO step end-to-end, 1-day horizon | Non-zero valid-tool-call rate; reward varies across rollouts; W&B logs canaries | Cold-start: base never gets reward → shorten horizon further / add a 1-shot format example in the prompt (not SFT) |
| 3 | **Short-horizon learning (C1)** | Dev `expected_net_revenue` curve rises above zero-shot baseline | No learning → check group size G, LR, α (Vibe-RL priors); confirm advantage isn't all-zero (same-seed grouping needed) |
| 4 | **Reward shaping** | Adding shaping speeds learning *without* the dominance invariant (S6) failing | Shaping outvotes profit → lower shaping weights; re-run S6 gate |
| 5 | **Reward-hack hunt** | Documented divergence (a real hack) **or** a clean "instrumented, none survived" with the canary plots | No dramatic hack appears → the simplest-reward ablation (profit-only vs shaped) becomes the story; read rollouts regardless |
| 6 | **Horizon curriculum (C2→C3)** | Policy still improves as horizon grows; no collapse | Long-horizon credit assignment fails → keep more steps at C2; report it as a finding |
| 7 | **Light sweep** | One concrete hyperparameter finding (α, LR, or G) | Budget overrun → 2-point α check only |
| 8 | **Final held-out eval + writeups** | Test-seed gap > noise; regret table done; D1/D2 drafted; **D3 pushed to Hub** | Result marginal → report honestly with per-seed bars; the *method* is the portfolio value |

**Cross-cutting risks:** Prime Lab GPU scarcity (fallback: self-managed pod, same vLLM endpoint); reproducibility drift (S8 + fixed seeds); blog needs the failure artifacts (capture every first-failure trace as it happens, don't reconstruct later).

---

## 8. Observability & tooling

- **W&B** (primary): training curves, the always-on canary dashboard (§4), per-sub-reward decomposition, and per-stage curriculum panels. The key custom panel is **shaped-reward vs `expected_net_revenue` vs `fraction_of_omniscient`** overlaid — the divergence detector.
- **Solvent CLI as the rollout-inspection layer:** every training/eval batch exports JSONL traces; use `solvent score` for the full scorecard, `solvent replay` to eyeball a single episode turn-by-turn, and `solvent compare --redteam-paired` periodically for manipulation-resistance and metric-divergence checks. This is the "look at actual rollouts" discipline that catches hacks.
- **Rollout export:** the Verifiers env writes a Solvent trace per rollout (same JSONL the scorer already consumes); a thin exporter samples K rollouts/step into a `rollouts/` dir for human reading + the reward-test harness.
- **Reward tests against live rollouts:** run `tests/test_rl_reward.py` (S1–S11, including the S10 delivery-gating and S11 breach-fee gates) in CI on fixtures **and** as a sampled check on exported live rollouts each run, so a rubric regression can't slip in silently.
- **Artifact capture for the writeup:** baseline scorecards, the first-failure trace, the canary time series at the moment of any divergence, per-capability before/after tables, and the final hyperparameter findings — all saved under `rl/artifacts/` as they're produced (inputs to D1/D2).

---

## Appendix — key integration facts (quick reference)

- Inject vLLM client via `LLMHarness(client=…)` for Solvent-side eval; for RL, Verifiers owns generation. (`llm.py:52`)
- Add a `$0` pricing entry for the model id or hit a `KeyError`. (`pricing.py:328-332`)
- Disable the cost budget guard for RL/eval (we bypass `LLMHarness`; for Solvent eval set `budget_usd: 0`). (`llm.py:186-198`, `runner.py:223`)
- Delivery = table lookup + seeded RNG; **no nested model calls.** (`menu.py:92-103`)
- Terminal reward = `expected_net_revenue`, GRPO group by seed. (`scorecard.py:285-306`)
- Per-decision regret available now for pricing + tool; selection needs marginal-attribution shim. (`scorecard.py:251-397`)
- Canaries already computed by the scorecard — wire to W&B, don't recompute. (`scorecard.py:241-425`, `leaderboard.py:367-388`)
- Expand `seeds_train.txt`; keep `seeds_test.txt` (`140–144`) held out. (`cli_seed.py:6-29`)
