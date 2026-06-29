# Solvent v0.6 — Realism & Scoring Overhaul: Implementation Plan

Source: Design session (2026-06-21) reviewing the freelance-business simulation in
`solvent/`. The conversation worked through (a) making the market mechanics and
economy realistic (Upwork-style undisclosed-budget RFQ with a starting price,
real dollar amounts, stochastic arrivals, mixed difficulty, 60-day horizon),
(b) simplifying the experiment (drop the trivial red-team/support axis), (c)
giving the long-horizon agent external memory, and (d) fixing scoring fairness
and variance. All user design decisions referenced below are final (see each
task's Solution section). Numbers like "$50–$500" and "×100 menu" are the
agreed targets, not placeholders.

## Shared context (read once)

The simulation has four layers; every task touches one or more:

- **Environment** — [`solvent/env/env.py`](solvent/env/env.py): the stateful
  episode. Emits a `episode_started` event with all config + a `provenance`
  block, exposes tool methods (`bid`, `deliver`, …), advances a business-time
  clock, and writes a JSONL trace. Config object is `EnvConfig`
  ([`solvent/env/models.py:9`](solvent/env/models.py:9)).
- **Market** — [`solvent/env/market.py`](solvent/env/market.py): seeded,
  procedural job generator. `_generate_v0_2_jobs`
  ([`market.py:68`](solvent/env/market.py:68)) sets each job's
  `reservation_price`/`est_cost`/decoy/manipulation; `_generate_stream_jobs`
  ([`market.py:105`](solvent/env/market.py:105)) assigns arrival/expiry times.
  Job value fields live on `Job`/`JobPublic`
  ([`solvent/env/models.py:51`](solvent/env/models.py:51)).
- **Scoring** — [`solvent/scoring/scorecard.py`](solvent/scoring/scorecard.py)
  replays a trace into facts, then compares the agent's decisions to reference
  optima from [`solvent/scoring/optimal.py`](solvent/scoring/optimal.py). Metric
  dataclasses are in [`solvent/scoring/models.py`](solvent/scoring/models.py).
  Aggregation/leaderboard is in
  [`solvent/findings/leaderboard.py`](solvent/findings/leaderboard.py); the HTML
  viewer is [`solvent/viewer/static/app.js`](solvent/viewer/static/app.js).
- **Harness** — [`solvent/harness/llm.py`](solvent/harness/llm.py): a ReAct loop
  that, each turn, feeds the model a fresh observation + the tool schemas +
  windowed history. Tools are defined in
  [`solvent/env/tool_api.py`](solvent/env/tool_api.py). Context windowing is
  [`solvent/harness/context.py`](solvent/harness/context.py).
- **Experiment runner** — [`solvent/experiment/`](solvent/experiment): builds the
  cell matrix and runs episodes. `_run_cell`
  ([`solvent/experiment/runner.py:161`](solvent/experiment/runner.py:161)) is
  where the hardcoded economy lives; `ExperimentConfig`/`MarketConfig` and YAML
  parsing are in
  [`solvent/experiment/config.py`](solvent/experiment/config.py). Canonical run
  config: [`configs/experiments/vb_style_v1.yaml`](configs/experiments/vb_style_v1.yaml).

**Key mechanic today (what's changing):** acceptance is a one-shot hidden
threshold — `accepted = price <= job.reservation_price`
([`env.py:176`](solvent/env/env.py:176)) — with no visible budget, no
fallback, no competition, no reputation (reputation flag exists but is off).
Money is in cents-to-dollars; jobs are all `easy`; the omniscient optimum is a
labeled upper-bound relaxation above ~16 jobs.

---

## Parallelization & ownership

Three independent workstreams. Within a stream, tasks that touch the **same
function** are flagged as serialization points.

| Stream | Tasks | Primary files | Notes |
|---|---|---|---|
| **A — World/economy** | 1, 2, 3, 4, 5 | `env/market.py`, `env/env.py`, `env/models.py`, `env/tool_api.py`, `experiment/*` | Tasks **1 & 2 both edit `_generate_v0_2_jobs`** → one owner or coordinate. Tasks 3/4/5 are conflict-free. |
| **B — Scoring** | 6, 7, 8, 9, 10, 11, 13 | `scoring/scorecard.py`, `scoring/optimal.py`, `scoring/models.py`, `findings/leaderboard.py`, `viewer/*` | **`scorecard.py` is a hotspot** (6, 7, 8, 10, 11). Split by method (each task owns distinct methods) or sequence on a shared branch. |
| **C — Harness/memory** | 12 (+ ctx part of 5) | `env/tool_api.py`, `harness/llm.py`, `harness/context.py` | Task 12 adds tools in `tool_api.py`; coordinate with Task 2 which also edits `TOOL_SCHEMAS`. |

**Cross-stream dependencies (hard):**
- **Task 7 (pricing rework) depends on Task 2** (needs `starting_price` on `Job`
  and the new accept/counter/reject negotiation terminal states).
- **Task 8 (expected-value scoring) depends on Task 7** (uses the final
  `contract_price`).
- **Task 13 (joint optimum) is gated on Task 4** (pointless while all-easy).
- **Tasks 6, 7 share `PricingSignal`/`Scorecard` dataclasses** in
  `scoring/models.py` — agree the dataclass shape first (see Task 6/7).

**Safe-to-start-immediately, no dependencies:** Tasks 1, 3, 4, 5, 6, 9, 12.

---

## Task 1 — Rescale the economy to realistic dollars

### Context
Every dollar amount is currently tiny: episodes start at `$20.00`, good jobs are
worth ~`$1.00–1.80`, decoys are money-losers in cents, delivery-tool prices are
`$0.02–0.45`, and overhead is `$0.000035/min`. These are set in two places:
job/decoy values in `_generate_v0_2_jobs`
([`market.py:78-85`](solvent/env/market.py:78)); the start balance, overhead,
and tool-call cost in `_run_cell`
([`runner.py:163-188`](solvent/experiment/runner.py:163)); and the delivery menu
prices in [`solvent/delivery/menu_data/menu_v0_4.json:12`](solvent/delivery/menu_data/menu_v0_4.json:12).

### Problem
The user wants realistic figures. A naive change to only job prices breaks the
economics: a `$0.45` tool on a `$300` job is rounding error, so model selection
(Task 13's whole point) becomes trivial. **Job prices, est_cost, decoy values,
menu prices, and overhead must all move together.**

### Desired outcome
- Start balance `$1000`.
- Good-job `reservation_price ∈ [$50, $500]`.
- `est_cost` ≈ 10–20% of reservation (so good jobs stay profitable).
- Decoy jobs remain money-losers (reservation < cost), scaled into the same
  order of magnitude.
- Menu prices ≈ ×100 of today (`tool-mini $2`, `tool-mid $12`, `tool-pro $45`)
  so the price/quality tradeoff is material on mid-size jobs.
- Daily rent ≈ `$10/day` (small vs. a $1000 balance + job income over 60 days).
- All existing tests that assert specific dollar values updated.

### Solution
Linear-ish rescale, but choose ranges to match the agreed targets (it is a
re-parameterization, not a constant multiply — current good range is 1.8× wide,
target is 10× wide). Keep decoys structurally unprofitable so the
decline-the-decoy skill survives. Rent is modeled through the existing
`overhead_per_minute` knob (`$10/day = 10/1440 ≈ $0.006944/min`).

### Implementation detail
- `_generate_v0_2_jobs` ([`market.py:78-85`](solvent/env/market.py:78)): change
  the good branch to `reservation_price = randrange(5000, 50001)/100`
  (`$50.00–$500.00`) and `est_cost = (reservation_price * randrange(10,21)/100)`
  quantized; change the decoy branch to e.g. `reservation_price =
  randrange(2000, 5000)/100` and `est_cost = randrange(6000, 9000)/100` (still
  negative value). Keep the seeded `rng` keying unchanged for reproducibility.
- `_run_cell` ([`runner.py:166-170`](solvent/experiment/runner.py:166)): set
  `start_balance=Decimal("1000.00")`,
  `overhead_per_minute=Decimal("0.006944")`, and a consistent
  `overhead_per_tick` (legacy tick-mode; business mode uses per-minute, so just
  keep it sane, e.g. `Decimal("10")`). Leave `tool_call_cost` at `0` for LLM
  cells.
- Menu: edit the three `price` fields in
  [`menu_v0_4.json:12-14`](solvent/delivery/menu_data/menu_v0_4.json:12) to
  `"2.00"`, `"12.00"`, `"45.00"`. The menu checksum is content-derived
  ([`menu.py:60`](solvent/delivery/menu.py:60)) so it updates automatically; the
  no-dominated-tools validation ([`menu.py:133`](solvent/delivery/menu.py:133))
  still holds (prices scaled uniformly).
- The task generators also set their own placeholder prices
  ([`data_clean.py:49-51`](solvent/tasks/data_clean.py:49),
  [`extract.py:19`](solvent/tasks/extract.py:19)) — these are **overridden** by
  `_generate_v0_2_jobs` for the stream market, so they only matter for the
  legacy `data_clean_static_v0_1` path. Leave them or scale for consistency.
- Verify: run the scorecard on a regenerated trace and confirm `net_revenue`,
  `omniscient_optimal_net`, and menu-based `_best_tool_value` are all in the new
  scale; update any test asserting old values (grep `"20.00"`, `"1.80"`, `"0.45"`).

---

## Task 2 — Add starting price + one-shot counter-offer negotiation

### Context
Today a job has only a hidden `reservation_price`; the agent submits one `bid`
and is accepted iff `price <= reservation_price`
([`env.py:176`](solvent/env/env.py:176)), else declined permanently
([`env.py:165-172`](solvent/env/env.py:165)). The public job
([`JobPublic`, models.py:51](solvent/env/models.py:51)) exposes no budget.

### Problem
This is a blind name-your-price game with no anchor. We want the Upwork-style
"undisclosed-budget RFQ with a posted starting price": the client posts a
visible floor, the freelancer can accept it or counter once, and a rejected
counter leaves the original floor offer open for an explicit accept/decline
decision.

### Desired outcome
Each job has a **visible** `starting_price` and a **hidden** `reservation_price`
with `starting_price < reservation_price`. The job negotiation state machine:
- `accept(job_id)` before any counter → accepted at **`starting_price`**.
- `bid(job_id, counter)` is allowed once. If `counter <= reservation_price`,
  accepted at **`counter`**.
- If `counter > reservation_price`, emit a rejected-counter state; the agent then
  chooses either `accept(job_id)` at the original **`starting_price`** or
  `decline(job_id)` permanently.

`starting_price = reservation_price × (1 − d)`, `d` noisy in `~[0.10, 0.40]`
(**confirmed: $50–$500 is the reservation/ceiling range**, starting set below).

### Solution
Add `starting_price` to the job model and surface it publicly. Add explicit
`accept` and `decline` tools and keep `bid` as the one-shot counter tool. The
agent's skill becomes anchor-relative price discovery plus an explicit
walk-away decision: counter just below the hidden ceiling, accept the floor when
the job is still worth it, or decline decoys/low-value work.

### Implementation detail
- `Job` ([`models.py:80-94`](solvent/env/models.py:80)): add
  `starting_price: Decimal`. `JobPublic`
  ([`models.py:51-59`](solvent/env/models.py:51)): add `starting_price: Decimal`
  and include it in `to_public` ([`models.py:96-105`](solvent/env/models.py:96)).
- `_generate_v0_2_jobs` ([`market.py:93-102`](solvent/env/market.py:93)): set
  `starting_price = (reservation_price * (Decimal(100 - rng.randrange(10,41)) /
  100)).quantize(Decimal("0.01"))` using the existing per-job `rng`. **(Touches
  the same function as Task 1 — coordinate.)**
- Negotiation state: add a per-job state for `open`, `counter_rejected`,
  `accepted`, and `declined` (or equivalent sets/maps). A counter below
  `starting_price` is dominated by accepting the floor; either reject it as an
  invalid bid or clamp/normalize it explicitly in the tool contract.
- `accept` ([new env/tool method]): if the job is `open` or `counter_rejected`,
  create `AcceptedJob(contract_price=job.starting_price)` and emit an acceptance
  event with `contract_price`, `starting_price`, and whether a counter preceded
  it.
- `bid` ([`env.py:152-188`](solvent/env/env.py:152)): allow exactly one counter.
  If `price <= job.reservation_price`, create `AcceptedJob(contract_price=price)`
  and emit `bid_accepted` with `counter_price`, resolved `contract_price`,
  `starting_price`, and `counter_accepted=True`. If `price >
  job.reservation_price`, emit `counter_rejected` with `counter_price` and leave
  the floor offer open.
- `decline` ([new env/tool method]): if the job is `open` or `counter_rejected`,
  mark it declined permanently and emit `job_declined`; this restores an explicit
  way to walk away from decoys after a rejected high counter.
- Tool surface: the `bid` schema description
  ([`tool_api.py:39-43`](solvent/env/tool_api.py:39)), new `accept`/`decline`
  schemas, and the observation ([`tool_api.py:79-108`](solvent/env/tool_api.py:79))
  must expose `starting_price` per available job plus any jobs awaiting
  post-counter accept/decline. `starting_price` rides along in `JobPublic` once
  `to_public` includes it. Update the system prompt
  ([`prompts.py:4`](solvent/harness/prompts.py:4)) to explain the starting-price
  / counter mechanic.
- Provenance/replay: nothing new needed — `starting_price` is in the job, and
  the scorer reconstructs jobs from the seeded market. The trace **must** carry
  the terminal acceptance contract price for Task 8, including jobs that later
  fail delivery and never emit a `paid` event.
- Verify: a counter ≤ reservation pays the counter; a counter > reservation pays
  nothing yet and moves to `counter_rejected`; from there, `accept` pays
  `starting_price` and `decline` closes the job. Update existing env/bid tests
  (e.g. over-ask no longer returns `{"accepted": false}`) and any stub flow that
  assumes `bid` is the only way to accept a job; add new tests in `tests/`
  mirroring `test_env_episode.py`.

---

## Task 3 — Poisson (stochastic, clumpy) job arrivals

### Context
`_generate_stream_jobs` ([`market.py:105-122`](solvent/env/market.py:105))
currently spaces arrivals **evenly**: `arrival = index * horizon /
expected_jobs` (line 112), with a deterministic count `expected_jobs =
round(rate × horizon / 1440)` (line 107).

### Problem
Real job boards are clumpy — jobs arrive in bursts, not on a metronome. The user
wants "3–6 jobs/day" to be genuinely stochastic (**confirmed**), so both the
**count** and the **timing** should follow a Poisson process.

### Desired outcome
Arrival times are a seeded Poisson process at rate `λ = arrival_rate_per_day /
1440` per minute over `[0, horizon)`; the number of jobs is the (random) count
that process yields. Fully reproducible per seed. Expiry logic unchanged
(`expiry = arrival + ttl`).

### Solution
Generate inter-arrival gaps as `Exponential(λ)` cumulatively until exceeding the
horizon; that yields both a random count and clumpy times. Generate that many
job contents (decoy/difficulty/manipulation assignment in `_generate_v0_2_jobs`
is per-index and seed-keyed, so it composes unchanged).

### Implementation detail
- In `_generate_stream_jobs` ([`market.py:105-122`](solvent/env/market.py:105)):
  - `rng = random.Random(f"{self.seed}:stream-arrivals")`.
  - `lam = float(self.arrival_rate_per_day) / 1440`; walk `t += rng.expovariate(lam)`
    appending `int(t)` while `t < horizon`; this list is the arrival minutes.
  - Set `self.market_size = len(arrivals)` (replaces the line-108 deterministic
    count) **before** calling `_generate_v0_2_jobs()` so it generates the right
    number of job contents.
  - Zip the generated jobs with the sorted arrival minutes; set
    `arrival_tick=arrival_minute=arrival`, `expiry_minute=min(horizon, arrival +
    ttl)` (keep the existing `ttl` line 109).
- Edge cases: guarantee ≥1 job (if the Poisson draw yields zero on a tiny
  horizon, force one arrival at 0) to keep smoke runs meaningful.
- Note the downstream estimate `jobs_over_horizon`
  ([`estimate.py:64`](solvent/experiment/estimate.py:64)) stays an expectation
  for cost budgeting — fine, it's only an estimate.
- Verify: same seed → identical arrival sequence; mean count over many seeds ≈
  `rate × days`; gaps are exponentially distributed (not constant).

---

## Task 4 — Mix in medium/hard difficulty

### Context
`EnvConfig.difficulty_distribution`
([`models.py:30`](solvent/env/models.py:30)) defaults to `{"easy": 1.0}` and the
market samples per-job difficulty from it
([`market.py:74`](solvent/env/market.py:74)). The task generators already
support `easy/med/hard` ([`data_clean.py:26-39`](solvent/tasks/data_clean.py:26),
[`extract.py:52-73`](solvent/tasks/extract.py:52)) and the menu has full
pass/duration profiles per difficulty. **But the experiment runner never sets
`difficulty_distribution`** — `MarketConfig`
([`config.py:13-18`](solvent/experiment/config.py:13)) has no such field and
`_run_cell` ([`runner.py:163-188`](solvent/experiment/runner.py:163)) omits it —
so every run is all-easy.

### Problem
All-easy jobs make tool selection trivial (`tool-mini` dominates on expected
value for every easy job), so the entire model-selection mechanic and Task 13
are dormant. We need a real difficulty spread.

### Desired outcome
Experiments run a configurable difficulty mix (default e.g. `{easy:0.4,
med:0.4, hard:0.2}`), threaded from YAML → `MarketConfig` → `EnvConfig` → market.
Difficulty is recorded in provenance (already happens via
[`env.py:88`](solvent/env/env.py:88)) so the scorer reconstructs it.

### Implementation detail
- `MarketConfig` ([`config.py:13-18`](solvent/experiment/config.py:13)): add
  `difficulty_distribution: dict[str, float] = field(default_factory=lambda:
  {"easy": 1.0})`.
- `experiment_config_from_dict`
  ([`config.py:71-76`](solvent/experiment/config.py:71)): parse
  `market.difficulty_distribution` via `_float_dict` like `task_mix`.
- `_run_cell` ([`runner.py:163-188`](solvent/experiment/runner.py:163)): pass
  `difficulty_distribution=config.market.difficulty_distribution` into
  `EnvConfig`.
- Confirm the env forwards it to `Market`
  ([`env.py:40-49`](solvent/env/env.py:40), it reads `config.difficulty_distribution`)
  and that the scorer's market reconstruction reads it from facts
  ([`events.py:111`](solvent/scoring/events.py:111)) — both already wired, just
  unused.
- Update [`configs/experiments/vb_style_v1.yaml`](configs/experiments/vb_style_v1.yaml)
  `market:` inline mapping to include `difficulty_distribution: {easy: 0.4,
  med: 0.4, hard: 0.2}`. The homegrown parser is flat
  ([`config.py:151`](solvent/experiment/config.py:151)), so do **not** convert
  this to an indented nested YAML block unless the parser is upgraded first.
- Verify: generated jobs show a mix of `internal_difficulty`; `tool_selection`
  regret is no longer trivially zero for a model that always picks the cheapest
  tool.

---

## Task 5 — 60-day horizon, scaled `max_turns`, 30k context window

### Context
The full run target is **60 days** with **3–6 jobs/day** → **~180–360 jobs per
episode**. Horizon is set by `ExperimentConfig.horizon_minutes`
([`config.py:28`](solvent/experiment/config.py:28)). `max_turns` defaults to
**200** in the harness ([`llm.py:31`](solvent/harness/llm.py:31)) and
`_harness_for_cell` ([`runner.py:201-208`](solvent/experiment/runner.py:201))
never overrides it. The context policy is `sliding_window` with
`ctx_window_tokens=24000` ([`config.py:30-31`](solvent/experiment/config.py:30)).

### Problem
At ~270 jobs the agent needs ≫200 turns (each job is inspect + bid + deliver +
maybe resolve, plus time-advances) — with `max_turns=200` it hits the turn cap
after ~40 jobs and the run is meaningless. The 24k window is already vending-
bench-style (last-N-tokens, [`context.py:36-53`](solvent/harness/context.py:36));
it just needs to match the 30k reference and never be `none`.

### Desired outcome
- `horizon_minutes` configurable to `86400` (60 days); `vb_style_v1` updated.
- `max_turns` scales with expected jobs (≈ `expected_jobs × 10 + buffer`,
  ~2,000–3,000 for the full run), configurable.
- `ctx_window_tokens` default bumped 24000 → 30000; no experiment uses
  `context_policy: none`.

### Solution
Make `max_turns` a derived/config value passed through to the harness. Keep the
sliding window (it's the intended long-horizon behavior); the forgetting it
causes is what motivates Task 12 (memory tools).

### Implementation detail
- `ExperimentConfig` ([`config.py:21-36`](solvent/experiment/config.py:21)): add
  `max_turns: int | None = None` and parse it
  ([`config.py:83-98`](solvent/experiment/config.py:83)). Bump
  `ctx_window_tokens` default to `30000` (line 31 + parse default line 92).
- `_harness_for_cell` ([`runner.py:201-208`](solvent/experiment/runner.py:201)):
  compute `max_turns = config.max_turns or (expected_jobs * 10 + 200)` where
  `expected_jobs = round(arrival_rate_per_day * horizon_minutes / 1440)`, and
  pass it into `LLMHarness.from_config_id` (which already accepts `max_turns`,
  [`llm.py:62-87`](solvent/harness/llm.py:62)).
- Update [`configs/experiments/vb_style_v1.yaml`](configs/experiments/vb_style_v1.yaml):
  `horizon_minutes: 86400`, `ctx_window_tokens: 30000`, and the new
  `arrival_rate_per_day` (~4.5; actual count is Poisson per Task 3).
- **Cost guard:** the 60-day × 7-model × 3-sample matrix is large. Recommend a
  short pilot config (14-day, 1–2 models, 1 sample) validated before the full
  run; `smoke_experiment_config` ([`config.py:101`](solvent/experiment/config.py:101))
  is the existing hook for this.
- Verify: a long-horizon run reaches the horizon (terminated reason `horizon`,
  not `turn_cap`) for a competent agent.

---

## Task 6 — Drop the red-team condition and the support metric

### Context
Experiments run paired `redteam_off`/`redteam_on` conditions
([`matrix.py:8`](solvent/experiment/matrix.py:8),
[`vb_style_v1.yaml`](configs/experiments/vb_style_v1.yaml) `conditions:`). Red-team
injects a single canned "30% off?" manipulation
([`market.py:87-92`](solvent/env/market.py:87)); the agent's handling is scored
into the **support** bucket (`SupportSignal`,
[`models.py:37-43`](solvent/scoring/models.py:37);
`_support`, [`scorecard.py:153-165`](solvent/scoring/scorecard.py:153)).

### Problem
Per the user, the red-team effort is trivial and low-value right now. Running
both conditions doubles cell count and cost. Drop the condition and remove the
support bucket from outputs to simplify.

### Desired outcome
- Experiments run `conditions: [redteam_off]` only, `manipulation_rate: 0` →
  ~halves cell count and spend.
- The `support` metric is removed from the scorecard, leaderboard, and viewer
  (or cleanly hidden), so no downstream code references a now-meaningless bucket.

### Solution
Config-level disable plus a scoring/UI cleanup. Keep the `SupportSignal` class
and `_support` method *available* (don't delete the manipulation machinery — it
may return later), but stop computing/surfacing it in experiment outputs. **Agree
the `Scorecard` dataclass shape with Task 7's owner** (both edit
`scoring/models.py`).

### Implementation detail
- Config: set `conditions: [redteam_off]` and `market.manipulation_rate: 0` in
  [`vb_style_v1.yaml`](configs/experiments/vb_style_v1.yaml). (Matrix already
  defaults to `redteam_off`, [`config.py:27`](solvent/experiment/config.py:27).)
- Scorecard: make `support` optional on `Scorecard`
  ([`models.py:92`](solvent/scoring/models.py:92)) or stop populating it in
  `build` ([`scorecard.py:77`](solvent/scoring/scorecard.py:77)); guard the
  viewer scorecard row ([`app.js:322`](solvent/viewer/static/app.js:322)).
- Leaderboard: remove `support_conceded_value` and
  `manipulation_resistance_loss` from the metric labels and rows
  ([`leaderboard.py:61,63,80,82,103,123,125,211,217`](solvent/findings/leaderboard.py:61))
  and the viewer columns + metric-keys list
  ([`app.js:406-407`](solvent/viewer/static/app.js:406),
  [`app.js:695,697`](solvent/viewer/static/app.js:695)).
- Trace viewer: the `manipulation_*` event renderers
  ([`trace_view.py:22,222,432`](solvent/viewer/trace_view.py:22)) can stay
  (harmless when no such events exist).
- Verify: a run with `manipulation_rate:0` produces scorecards/leaderboards with
  no support columns and no errors; update tests asserting support fields.

---

## Task 7 — Rework `pricing_regret` for the counter-offer mechanic

### Context
`_pricing` ([`scorecard.py:115-140`](solvent/scoring/scorecard.py:115)) currently
computes `pricing_regret = surplus_left + lost_to_overprice`
([`PricingSignal`, models.py:19-26](solvent/scoring/models.py:19)), where
`lost_to_overprice` is the value of good jobs **declined** for over-pricing.

### Problem
Under Task 2's mechanic, over-asking no longer automatically declines a job or
automatically falls back to `starting_price`. It creates a rejected-counter state,
after which the agent explicitly accepts the floor or declines. So
`lost_to_overprice` (declined-good-jobs due to over-pricing) is no longer the
right pricing term. **This task depends on Task 2.**

### Desired outcome
`pricing_regret` reflects the new rule:
- **Under-pricing** (counter accepted, below reservation): `surplus_left =
  reservation − contract_price` (unchanged in spirit).
- **Accepting the floor without countering**: `reservation − starting_price`
  (timidity forfeits surplus versus the omniscient counter-at-reservation policy).
- **Counter rejected → accept floor**: `reservation − starting_price` (plus any
  separate efficiency/coherence cost for the wasted turn, if desired).
- **Counter rejected → decline**: no pricing regret; if the job was good, this is
  selection/opportunity regret, and if it was a decoy it is correct play.
- The old "declined good job due to overprice" term is removed.

### Solution
Read the terminal negotiation outcome the env now emits (`accept` at floor,
accepted counter, rejected counter then accept/decline) and recompute regret from
that terminal state. Bids are now negotiation attempts, not the accepted/chosen
set. The facts layer should expose an explicit accepted-jobs map/set from
terminal acceptance events, and scoring should use that accepted set anywhere it
currently infers acceptance from `BidFact.accepted`. Optimal per-job pricing play
is unchanged: counter exactly at `reservation`, capturing full surplus; the
floor-accept paths are intentionally penalized by the surplus left on the table.

### Implementation detail
- Extend the bid fact extraction (the `bids`/`BidFact` source in
  [`scoring/events.py`](solvent/scoring/events.py)) to carry `starting_price`,
  `counter_price` if present, resolved `contract_price`, and a terminal outcome
  enum from the new acceptance / counter-rejected / declined payloads (Task 2).
- Add an accepted-job fact/map built from terminal acceptance events (floor
  accepts and accepted counters). Re-point `_selection`
  ([`scorecard.py:88`](solvent/scoring/scorecard.py:88)), `_pricing`
  ([`scorecard.py:116`](solvent/scoring/scorecard.py:116)), and `_coherence`
  ([`scorecard.py:168`](solvent/scoring/scorecard.py:168)) at this accepted set
  instead of deriving "chosen/accepted" from bids. Otherwise floor accepts are
  invisible to scoring, and counter-rejected-then-declined jobs look wrongly
  chosen.
- `_pricing` ([`scorecard.py:115-140`](solvent/scoring/scorecard.py:115)):
  - Scope pricing regret to good/profitable jobs; accepting a decoy is a
    selection error, not pricing regret.
  - Accepted counter at `c <= reservation`: `reservation − c`.
  - Floor accepted, with or without a rejected counter first:
    `reservation − starting_price`.
  - Declined after rejected counter: `0` pricing regret; selection scoring handles
    whether walking away was good or bad.
  - `pricing_regret` is the sum of these terminal-state losses, with no additive
    second over-ask term.
- `PricingSignal` ([`models.py:19-26`](solvent/scoring/models.py:19)): replace
  `lost_to_overprice`/`declined_good_jobs` with fields that describe floor accepts
  and rejected counters (coordinate dataclass shape with Task 6). Update the
  leaderboard `pricing_regret` and viewer column to match.
- Verify: an agent countering exactly at reservation has `pricing_regret ≈ 0`;
  one always accepting the floor accrues `reservation − starting` per good job;
  one countering too high then declining good jobs gets selection regret rather
  than pricing regret.

---

## Task 8 — Expected-value (control-variate) scoring to kill delivery-luck variance

### Context
Delivery pass/fail is a seeded Bernoulli draw keyed by `(seed, job, model,
attempt)` ([`_delivery_draw_key`, env.py:644-645](solvent/env/env.py:644)); a
fail charges the tool price with no revenue. `net_revenue` is the **realized**
outcome, while the references use **expected** value (`reservation × pass_prob`),
so per-seed `fraction_of_omniscient_optimal` can exceed 1.0 and carries large
non-skill variance. **Depends on Task 7** (uses final `contract_price`).

### Problem
With ~270 jobs the delivery variance inflates confidence intervals (needs more
expensive samples to rank models) and produces misleading `>1.0` per-seed
numbers. The noise is luck, not skill.

### Desired outcome
A variance-reduced **decision-quality** score reported alongside realized net:
for each delivery, credit `contract_price × pass_prob(job, model, difficulty) −
model_price` instead of the realized 0/1. Equivalent control-variate form (zero
bias): `net_CV = net_revenue − Σ (1[pass_j] − pass_prob_j) × contract_price_j`.
This removes the dominant `Σ contract_price² · p(1−p)` variance and puts the
agent on the same expected basis as the references, so `>1.0` disappears.

### Solution
Mostly scorecard-layer change after Task 2/7 have added acceptance contract
prices to the trace/facts — no extra LLM calls. The scorer already has
`pass_prob` ([`DeliveryMenu.pass_prob`, menu.py:72](solvent/delivery/menu.py:72))
and the chosen model/difficulty per delivery in the trace. Keep realized
`net_revenue` as a secondary "survival" metric (a fail near insolvency is real
risk).

### Implementation detail
- In `ScorecardBuilder` ([`scorecard.py:45`](solvent/scoring/scorecard.py:45)),
  add an `expected_net_revenue` computed from each `DeliveryAttemptFact`
  ([`_delivery_attempts`, scorecard.py:252](solvent/scoring/scorecard.py:252)):
  `Σ contract_price × pass_prob(job.type, model, difficulty) − price_charged`,
  minus the same overhead the realized net pays.
- Contract price is not available from failed `paid` events. Task 2/7 must add
  resolved `contract_price` to `BidFact`/acceptance facts and Task 8 must join
  each `DeliveryAttemptFact` back to its accepted bid by `job_id`; otherwise
  failed deliveries cannot be scored on an expected-revenue basis.
- Add `expected_net_revenue` and `fraction_of_omniscient_optimal_expected` (and
  `_realizable_expected`) to `Scorecard`
  ([`models.py:75-99`](solvent/scoring/models.py:75)); compute the fractions via
  the existing `_fraction` ([`scorecard.py:276`](solvent/scoring/scorecard.py:276)).
- Make the **expected** fraction the leaderboard headline; keep realized as
  secondary. Update [`leaderboard.py`](solvent/findings/leaderboard.py) and the
  viewer columns.
- Verify: across many seeds, the expected fraction has materially lower variance
  than realized and never exceeds 1.0 on the scheduling dimension; realized mean
  ≈ expected mean.

---

## Task 9 — Common-random-numbers (CRN) paired comparison reporting

### Context
The delivery draw key contains **no agent identity and no bid price**
([`env.py:645`](solvent/env/env.py:645)), so two agents that deliver the same
job with the same model on the same seed get the **identical** pass/fail. The
leaderboard currently reports independent per-model means
([`leaderboard.py`](solvent/findings/leaderboard.py)).

### Problem
Independent means discard the shared-luck correlation, widening the variance of
model-vs-model differences and wasting the CRN structure already baked into the
draw keys.

### Desired outcome
Model/ablation comparisons reported as **paired differences on shared seeds**
(A − B per seed, then aggregated), collapsing the common delivery/market luck and
sharpening rankings at the same sample cost.

### Solution
Aggregation-only change; no env or scoring-math change. Stacks on Task 8 (pair
the expected-value metric for maximum variance reduction).

### Implementation detail
- In [`solvent/scoring/aggregate.py`](solvent/scoring/aggregate.py) /
  [`leaderboard.py`](solvent/findings/leaderboard.py), add a paired-difference
  summary: for each (seed, sample) shared across two configs, compute the metric
  delta, then summarize the deltas (mean, CI) instead of differencing two
  independent means.
- Reuse the existing distribution summarizer
  ([`summarize_distribution`, leaderboard.py:123](solvent/findings/leaderboard.py:123))
  on the per-seed deltas.
- Verify: paired-difference CIs are tighter than the implied
  difference-of-independent-means CIs on the same data.

---

## Task 10 — Tie-tolerant selection precision/regret

### Context
`_selection` ([`scorecard.py:87-113`](solvent/scoring/scorecard.py:87)) scores
the agent's chosen job set against the omniscient reference's **single** selected
subset (`reference.selected_jobs`): a profitable job the agent took that isn't in
*that exact* subset is counted as a "decoy chosen" and penalized
([`scorecard.py:96-104`](solvent/scoring/scorecard.py:96)).

### Problem
When multiple job subsets are equally optimal (common with ties / alternative
schedules), the agent is wrongly penalized for picking a different-but-equally-
good subset, and precision drops below 1.0 for optimal play.

### Desired outcome
Selection precision/regret judge "is this job in *some* optimal subset", not
membership in the one argmax subset, so alternative optima aren't penalized.

### Solution
Replace the single-subset membership test with a marginal inclusion test:
"does there exist an optimal schedule containing this job?" A simple per-job
value threshold is sound in direct/no-duration mode, but it is **not** tie-safe
for tool-mediated business-time schedules because a high-value job may conflict
with a better combination of other jobs. The per-job value helper already exists
(`_job_selection_value`, [`scorecard.py:232`](solvent/scoring/scorecard.py:232)).

### Implementation detail
- In `_selection` ([`scorecard.py:87-113`](solvent/scoring/scorecard.py:87)),
  derive a "good set" by exact optimal-subset union where tractable, or by
  recomputing the optimal value with each candidate forced in/out. Recompute
  `precision`, `good_chosen`, `decoys_chosen`, `missed_good`, `chased_decoys`
  against that set.
- At ~270 jobs the reference is already a relaxation, so this tie-tolerant set is
  approximate at scale too; keep that label/interpretation consistent with
  `ReferenceResult.relaxation`.
- Verify: an agent that selects an alternative optimal subset scores precision
  1.0 / regret ≈ 0; a genuinely worse selection still accrues regret.

---

## Task 11 — Threshold-policy baseline reference (`fraction_of_threshold_policy`)

### Context
Both references — `omniscient_reference` and `realizable_reference`
([`optimal.py:23-58`](solvent/scoring/optimal.py:23)) — are **offline/clairvoyant**
(they see all future arrivals and schedule with perfect foresight). The agent is
online and cannot. So `fraction_of_omniscient_optimal` mixes genuine mistakes
with the unavoidable cost of not seeing the future.

### Problem
There is no practical online baseline for "how well does a good no-crystal-ball
policy do?", so the headline fraction systematically mixes agent mistakes with
the unavoidable cost of not seeing future arrivals.

### Desired outcome
A third reference, `threshold_policy_reference_net`, estimating the expected net
of a good **online heuristic** (a reservation-value / threshold policy that
accepts a job only when its value exceeds the opportunity cost of the time it
occupies), with `fraction_of_threshold_policy` reported alongside the
clairvoyant ceiling. This is a beatable baseline, not an online optimum.

### Solution
Monte-Carlo a threshold policy over the seed/arrival distribution (the agent gets
no distribution params today, but the *reference* may use them). This is a
defensible fairness baseline and directly rewards the opportunity-cost reasoning
the benchmark wants, but it is **not** an upper bound: a strong agent beating it
is the intended signal. New, self-contained code; no env change.

### Implementation detail
- Add `threshold_policy_reference(jobs, facts, ...)` to
  [`optimal.py`](solvent/scoring/optimal.py): simulate a single-server threshold
  policy over the seeded arrival stream (accept iff job expected value > running
  opportunity-cost threshold derived from `arrival_rate_per_day`, remaining
  horizon, and the menu), averaged over N inner draws for the delivery RNG.
- Add `threshold_policy_reference_net` + `fraction_of_threshold_policy` to
  `Scorecard` ([`models.py:75-99`](solvent/scoring/models.py:75)) and wire into
  `build` ([`scorecard.py:57-85`](solvent/scoring/scorecard.py:57)) and the
  leaderboard.
- Note the ~270-job scale: the policy sim is O(jobs), cheap; cache per
  `(seed, market_config)` since it's agent-independent.
- Verify: the threshold baseline is stable/reproducible and meaningfully below
  the clairvoyant reference; agents may exceed it, and that is the intended
  skill signal.

---

## Task 12 — Agent memory tools (scratchpad + key-value store)

### Context
The harness feeds the model a fresh observation + the last ~30k tokens of history
each turn ([`llm.py:96-116`](solvent/harness/llm.py:96);
`_sliding_window`, [`context.py:36-53`](solvent/harness/context.py:36)). At ~270
jobs a compact per-job ledger (~150 tokens each ≈ 40k tokens) **exceeds the
window**, so the agent provably cannot hold its own job history — it forgets
which jobs it bid on, accepted, delivered, and which models worked.

### Problem
Without external memory, a 60-day run mostly measures amnesia-coping (duplicate
bids, forgotten acceptances, repeated model mistakes inflating the coherence
penalty) rather than business judgment.

### Desired outcome
Agent-controlled, persistent memory tools — **Phase 1: scratchpad + key-value
store, no embeddings** (a vector DB is explicitly deferred). Reads/writes are
deterministic, free (or normal `tool_call_cost`), and do **not** advance business
time. Keyed lookup (`mem_read(job_id)`) fits the structured state better than
semantic search.

### Solution
Add `mem_write`, `mem_read`, `mem_list`, `mem_delete` tools backed by a dict that
persists across turns within an episode. This relocates (doesn't remove) the
memory-management skill and is realistic (a CRM/notebook). The existing
`scratchpad` context policy ([`context.py:55-75`](solvent/harness/context.py:55))
can complement it as an auto-summary. **Defer the vector DB** — it pulls a
non-deterministic external embedding API into the replay/scoring path.

### Implementation detail
- Backing store: a per-episode dict (on the `Environment` or the harness). Keep
  it **out of** the business ledger; expose via new methods or directly in the
  `ToolAdapter`.
- `TOOL_SCHEMAS` ([`tool_api.py:31-67`](solvent/env/tool_api.py:31)): add
  `mem_write{key, value}`, `mem_read{key}`, `mem_list{}`, `mem_delete{key}`;
  dispatch them in `_invoke` ([`tool_api.py:158-183`](solvent/env/tool_api.py:158)).
  **(Coordinate with Task 2, which also edits `TOOL_SCHEMAS`.)**
- Determinism: pure dict → trace-replayable. If/when Phase 2 (vector DB) is
  added, pin the embedding model and cache embeddings by text-hash.
- System prompt ([`prompts.py`](solvent/harness/prompts.py)): tell the agent it
  has a persistent notebook and should record jobs/bids/outcomes there.
- Verify: across a long run, duplicate-bid and dropped-job counts (coherence
  inputs, [`scorecard.py:167-187`](solvent/scoring/scorecard.py:167)) drop when
  the agent uses memory; memory ops don't move `business_time` or the balance.

---

## Task 13 — (Gated) Approximate joint model×schedule optimum

### Context
The omniscient optimum picks each job's **value-maximizing** model independently
(`_best_tool_value`/`_best_tool_duration`,
[`optimal.py:79-104`](solvent/scoring/optimal.py:79)) and then schedules
([`_business_time_selected_jobs`, optimal.py:107](solvent/scoring/optimal.py:107)),
never trading a faster/cheaper model on one job to fit more jobs overall. The
exact DP is exponential (capped at 16 jobs → relaxation above
[`optimal.py:108-112`](solvent/scoring/optimal.py:108)).

### Problem
The reference isn't a true joint optimum over (model × schedule). **Gated on Task
4**: while all jobs are easy, `tool-mini` dominates and the joint optimum ≈
current, so this is pointless until a difficulty mix exists. At the 60-day /
~270-job scale, an **exact** joint optimum is infeasible — this must be a
**heuristic/approximate** reference, clearly labeled (like the existing
relaxation).

### Desired outcome
A better-but-approximate joint model+schedule reference for tool-mediated runs
with mixed difficulty, labeled as a relaxation/upper-or-lower bound so the
fraction is interpreted correctly. Only pursue after Task 4 lands and only if the
gap between current and joint optima proves material.

### Solution
A greedy/local-search joint assignment: schedule by value density while allowing
per-job model downgrades when they free capacity for a net-positive additional
job. Keep the `relaxation=True` labeling discipline already in
`ReferenceResult` ([`optimal.py:12-16`](solvent/scoring/optimal.py:12)).

### Implementation detail
- Extend `_business_time_selected_jobs`
  ([`optimal.py:107-135`](solvent/scoring/optimal.py:107)) (or add a sibling) to
  branch over models per job in the value/duration functions rather than the
  pre-collapsed `_best_tool_value`/`_best_tool_duration`.
- Mark results `relaxation=True` and surface the label in the scorecard/viewer
  (the viewer already shows "upper bound" via
  `omniscient_reference_relaxation`).
- Verify (after Task 4): on mixed-difficulty seeds the joint reference ≥ the
  current per-job-max reference, and the difference is reported.

---

## Suggested order

1. **Tasks 4, 5, 6 (config-only, instant):** difficulty mix, 60-day horizon +
   `max_turns` + 30k window, drop red-team/support. Cheapest, simplifies
   everything downstream, unblocks the others. Low risk.
2. **Tasks 1 + 2 (env/economy, one owner — same function):** money rescale and
   the starting-price/counter mechanic. The most behavior-changing pair; do them
   together to avoid `_generate_v0_2_jobs` conflicts.
3. **Task 3 (Poisson arrivals):** independent, can run in parallel with #2.
4. **Task 12 (memory tools):** independent (coordinate the `TOOL_SCHEMAS` edit
   with Task 2); needed before any meaningful 60-day run.
5. **Task 7 (pricing rework):** immediately after Task 2 (depends on it).
6. **Task 8 (expected-value scoring) + Task 9 (CRN pairing):** after Task 7;
   biggest payoff for trustworthy, cheap-to-rank results.
7. **Task 10 (tie-tolerant selection):** independent scoring fix, medium priority.
8. **Task 11 (threshold-policy baseline):** high interpretive value, net-new
   code; after the scoring refactors settle.
9. **Task 13 (joint optimum):** last, and only after Task 4 proves model
   selection is non-trivial; approximate only.

**Before the full 60-day × 7-model × 3-sample matrix, run a short pilot** (14-day,
1–2 models, 1 sample via `smoke_experiment_config`) to validate the new mechanic
+ memory + scoring end-to-end — that's where the real cost savings is.
