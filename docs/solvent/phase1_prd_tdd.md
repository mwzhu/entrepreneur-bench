# Solvent Phase 1 — Freelance Operation Eval Environment

## Concept / PRD + Technical Design Document

**Status:** Design complete, pre-build · **Owner:** [you] · **Companion:** `solvent_three_phase_plan.md`

---

# PART A — PRODUCT REQUIREMENTS (PRD)

## A.1 One-liner

A reproducible, deterministically-scored environment in which a swappable agent harness runs a **freelance services business over a horizon** — discovering gigs, choosing which to bid on, pricing them, delivering the work, and handling support and manipulation — scored on net revenue **and** a per-stage, ground-truth decomposition that says *which capability* succeeded or failed.

## A.2 Problem & motivation

Economic-agent results today emit a single confounded number. Vending-Bench 2 scores a year-end balance; SWE-Lancer scores dollars earned on single tasks; Automaton emits nothing reproducible at all. A scalar like net revenue is a product of every capability (`selection × pricing × delivery × support × coherence`), so a low score is uninterpretable — you cannot tell whether the agent chose the wrong work, priced it wrong, or couldn't do it.

The deployed-agent literature confirms these are *distinct* failures that occur together:

- **Selection:** Project Vend's Claudius ignored a $100 offer for a ~$15 item.
- **Pricing:** it sold high-margin items below cost; GPT-5.1 in VB2 overpaid suppliers ($2.40 cans, $6 energy drinks); Project Vend 2's summary — the models decide "like a friend who wants to be nice rather than on hard-nosed market principles."
- **Delivery:** SWE-Lancer's best model resolves only ~26% of IC SWE tasks.
- **Support/manipulation:** staff and WSJ reporters repeatedly talked the agent into discounts, free items, and ex-post price cuts.
- **Coherence:** the "blue blazer" identity crisis; the CEO/agent spiraling into "eternal transcendence" overnight.

**The gap:** every existing environment can see *that* agents fail; none attributes the failure to a capability. Phase 1 closes this gap for one vertical, deterministically and reproducibly.

## A.3 Goals

1. **G1 — Decomposed economic score.** Produce, per run, a net-revenue scalar *and* per-stage signals (selection, pricing, delivery, support+manipulation, coherence), each scored against environment-held ground truth.
2. **G2 — Reproducibility.** Seeded markets; identical conditions across configs; distributions over seeds.
3. **G3 — Gaming resistance.** Only deterministic checks move money. No LLM judges payment.
4. **G4 — Swappable harness.** Drop in different models and memory/planning/procedure configs; compare on a scoreboard.
5. **G5 — External validity.** A thin SWE-Lancer anchor confirms sim rankings track reality.
6. **G6 — Demoable.** A clickable trace viewer + a cross-config scoreboard with a dollar delta.

## A.4 Non-goals (Phase 1)

- **NG1** Oracle-substitution attribution engine (Phase 2; Phase 1 only *logs everything needed* to run it later).
- **NG2** Procurement/supplier negotiation as a full axis (Phase 2).
- **NG3** Multi-agent / competition (Phase 2).
- **NG4** RL training loop (Phase 3; Phase 1 keeps the boundary clean so it's possible).
- **NG5** Real money. All payments are simulated; the SWE-Lancer anchor uses recorded real prices, not live payouts.

## A.5 Users & use cases

| User | Use case |
|---|---|
| Eval engineer (you) | Compare harness configs; localize a model's weakest economic capability |
| Harness developer | Measure whether a scaffold change (memory, procedure) improves business outcomes |
| Researcher / reviewer | Inspect a trace; verify a result is real and gaming-resistant |
| (Phase 3) RL engineer | Consume per-stage signals as a dense reward |

## A.6 Success criteria ("done" for Phase 1)

- **S1** Two configs (e.g., memory vs. no-memory) yield a reproducible scoreboard delta on identical seeds, with mean ± std over ≥5 seeds.
- **S2** The scorecard localizes loss to a stage (worked example: "delivery 100%, but pricing leaks $2.40 and selection chases a decoy").
- **S3** A seeded manipulation event measurably lowers a naive config's score and not a hardened config's — entirely from the ledger, no LLM judge.
- **S4** *(directional sanity check, not a pass/fail criterion — and deferred past the first demo)* If/when the SWE-Lancer anchor is run, sim-rank of models is *not contradicted* by their anchor rank. With only a handful of anchor tasks and 2–3 models this is underpowered, so treat it as a smell test, not evidence of external validity. Real validity work is Phase 2.
- **S5** `solvent run`, `solvent compare`, and the trace viewer all work end-to-end from one command.

## A.7 Positioning

Don't compete on realism (Andon/Anthropic own it). The wedge is **attribution** + **RL-readiness**. Lead every external description with: *Solvent measures, per-capability and deterministically, exactly the failures Project Vend and Vending-Bench can only observe in aggregate.*

**The pitch (use ~verbatim):**

> I built a deterministic freelance-business eval where agents choose jobs, bid, deliver verifiable work, and handle adversarial customer requests. Unlike balance-only business evals, Solvent decomposes losses by capability, so you can see whether the agent failed because it selected bad work, underpriced, failed delivery, got manipulated, or lost operational coherence.

**Honest scope of the claim.** Phase 1 measures performance against a *designed market* (a `reservation_price` is an authored acceptance threshold, not a measurement of real customer demand). That is the right thing to measure for an *attribution instrument*, and fraction-of-optimal keeps scores interpretable within the designed market — but external validity (does sim rank track reality?) is **explicitly deferred** to the SWE-Lancer anchor and Phase 2. Name Phase 1 as an internal-consistency / attribution instrument, not a validity claim.

**Lead the demo with pricing-attribution, not manipulation.** The most *differentiating* trace is: the agent delivers flawlessly and prices below cost, the scalar reads "mediocre," and the decomposition shows pricing is the entire loss — the wedge made visible. Use the manipulation event ("agent got talked into giving money away") as the visceral second beat.

---

# PART B — CONCEPTUAL DESIGN

## B.1 The environment model

A **freelance operation** modeled VB-style: the harness is a one-person agency running over a horizon under solvency pressure.

- **Budget:** starts at `B0` credits (default $20.00 — enough to attempt a code gig plus several cheap gigs).
- **Burn:** every model/tool call is metered at real token cost and deducted; a small per-tick overhead fee (`OVERHEAD`, default $0.05) creates background pressure even when idle (VB's "daily fee").
- **Horizon:** `N_TICKS` (default 30) or until insolvent (`balance ≤ 0`).
- **Market:** a seeded job board; gigs arrive on a schedule.
- **Revenue:** credited only when a delivered artifact passes a deterministic verifier.
- **Score:** net revenue + fraction-of-optimal + per-stage decomposition.

Set breadth = 1 (one task type, oracle selection/pricing, long horizon) and you recover Vending-Bench's coherence-only eval exactly.

## B.2 The five stages and what each measures

| Stage | Agent decision | Ground truth used | Signal |
|---|---|---|---|
| Selection | which gigs to pursue given budget | true value of each job; decoy flags | precision, recall, selection regret |
| Pricing (**flagship**) | what to bid on a chosen gig | hidden reservation price | price/reservation price, pricing regret |
| Delivery | produce the artifact | rubric | pass rate |
| Support + manipulation | revise; resist bad-faith requests | rubric; manipulation script | recovery rate; manipulation-resistance |
| Coherence | keep books, finish jobs, don't melt down | ledger | bookkeeping error, dropped jobs, meltdown flag |

## B.3 Scoring philosophy

1. **Deterministic path.** A sale is accepted iff `bid ≤ reservation price`. A deliverable is paid iff the verifier passes. A revision succeeds iff the (possibly updated) rubric passes. LLMs may generate posting/complaint *prose*; they never decide outcomes.
2. **Ground truth → per-stage scoring.** Each decision is compared to the knowable-optimal decision at that stage.
3. **Fraction-of-optimal.** Compute the optimal reference policy from ground truth; report `net / optimal_net`.
4. **Embryonic decomposition now, full attribution later.** Phase 1 ships the per-stage signals and logs every event so Phase 2's oracle substitution is a re-run, not a rewrite.

---

# PART C — TECHNICAL DESIGN

## C.1 System architecture

```
                    ┌─────────────────────────────────────────────┐
                    │                 HARNESS                      │
                    │  (the system under test — swappable)         │
                    │  ReAct loop · memory · planner · procedure   │
                    └───────────────┬───────────────▲─────────────┘
                          tool calls │               │ observations
                                     ▼               │
   ┌─────────────────────────────────────────────────────────────┐
   │                       ENVIRONMENT                            │
   │                                                              │
   │  ┌──────────┐   ┌───────────┐   ┌────────────┐   ┌────────┐  │
   │  │  Market  │   │ Customer  │   │  Verifier  │   │ Ledger │  │
   │  │ (seeded  │   │  Model    │   │ (per type, │   │  +     │  │
   │  │  jobs +  │   │ (reservation price +    │   │determinis- │   │ Trace  │  │
   │  │  hidden  │   │ manip.    │   │   tic)     │   │ (JSONL)│  │
   │  │  truth)  │   │ events)   │   │            │   │        │  │
   │  └──────────┘   └───────────┘   └────────────┘   └────────┘  │
   │                                                              │
   │  Budget/Burn accountant · Clock · Termination                │
   └───────────────────────────┬──────────────────────────────────┘
                                ▼
                    ┌───────────────────────┐
                    │       Scorer          │
                    │ per-stage signals,    │
                    │ fraction-of-optimal,  │
                    │ optimal ref policy    │
                    └───────────────────────┘
```

**The boundary is load-bearing:** the environment sees only tool calls and emits only observations + reward. It knows nothing about what's inside the harness. This is what makes harnesses swappable (Phase 1) and the whole thing an RL environment (Phase 3, harness = policy).

## C.2 Repo layout

```
solvent/
  harness/        # swappable agent scaffold: loop, memory, planner, procedure, metering
  env/
    market.py     # seeded job generator + arrival schedule
    customer.py   # deterministic reservation price + manipulation events
    verifier/     # per-task-type deterministic checkers
    ledger.py     # balance, burn accounting, event log
    clock.py      # ticks, overhead, termination
    env.py        # orchestrator: exposes tool API, steps, emits reward
  tasks/          # task-type definitions + generators (data_clean, copywriting, extract, code_fix)
  anchor/         # SWE-Lancer integration (IC SWE delivery + Manager selection)
  scoring/
    signals.py    # per-stage signal computation
    optimal.py    # optimal reference policy (knapsack)
    scorecard.py  # assemble + serialize
  trace/          # JSONL writer + web viewer
  cli/            # solvent run | compare | replay | characterize
  configs/        # harness configs, seed sets (dev/test split)
```

## C.3 Data model

### C.3.1 Job

```python
@dataclass
class Job:
    id: str
    type: str                 # "data_clean" | "copywriting" | "extract" | "code_fix"
    brief: str                # PUBLIC: the spec the agent sees
    inputs: dict              # PUBLIC: CSV / docs / repo handle
    arrival_tick: int         # PUBLIC

    # ---- hidden ground truth (never exposed via the tool API) ----
    rubric: Rubric            # deterministic pass condition
    reservation_price: float                # hidden reservation price (USD)
    est_cost: float           # expected compute cost to deliver (USD)
    true_value: float         # reservation_price - est_cost  (negative => decoy)
    is_decoy: bool
    revision_event: Optional[RevisionEvent]   # seeded post-acceptance change
    manipulation: Optional[ManipulationEvent] # seeded bad-faith request
```

### C.3.2 Rubric / Verifier result

```python
@dataclass
class Rubric:
    checks: list[Check]       # programmatic predicates over the artifact
    # each Check: name, fn(artifact, inputs) -> bool, weight

@dataclass
class VerifyResult:
    passed: bool              # all required checks pass
    score: float              # graded [0,1] for partial credit / RL shaping
    failed_checks: list[str]
```

### C.3.3 Episode / Event / Scorecard

```python
@dataclass
class Episode:
    seed: int
    config_id: str
    start_balance: float
    horizon: int
    jobs: list[Job]
    events: list[Event]       # the full ordered trace
    end_balance: float
    terminated_reason: str    # "insolvent" | "turn_cap"

@dataclass
class Event:                  # one JSONL line
    tick: int
    kind: str                 # see C.5
    payload: dict
    balance_after: float
    burn_delta: float

@dataclass
class Scorecard:
    net_revenue: float                  # revenue − burn  (the economic bottom line)
    gross_score: float                  # task performance ignoring burn (for cross-model fairness)
    fraction_of_omniscient_optimal: float   # vs capability-blind ceiling (upper bound)
    fraction_of_realizable: float       # vs capability-conditioned reference policy (fair bar)
    selection: SelectionSignal
    pricing: PricingSignal
    delivery: DeliverySignal
    support: SupportSignal
    coherence: CoherenceSignal
    manipulation_resistance: float      # paired delta: score(redteam_off) − score(redteam_on), same seed
```

> Report **gross** (quality of work, burn-blind) and **net** (after burn) side by side: a cheap-but-worse model vs. an expensive-but-better one is real economic signal, but you must not let burn differences masquerade as capability differences. `fraction_of_optimal` elsewhere in this doc refers to `fraction_of_omniscient_optimal` unless stated.

### C.3.4 Seed config

```yaml
seed: 42
split: dev            # dev | test  (never tune on test)
start_balance: 20.00
horizon_ticks: 30
overhead_per_tick: 0.05
task_mix: {data_clean: 0.4, copywriting: 0.3, extract: 0.2, code_fix: 0.1}
decoy_rate: 0.25
manipulation_rate: 0.2
difficulty: easy      # controls delivery hardness (per-stage dial)
```

## C.4 Episode lifecycle (tick loop)

```python
def run_episode(harness, env_cfg, seed) -> Episode:
    env = Environment(seed, env_cfg)          # generates market + ground truth
    harness.reset(env.tool_api())             # harness sees only the tool API
    while not env.terminated():
        obs = env.observe()                   # board, balance, in-progress jobs (PUBLIC ONLY)
        action = harness.act(obs)             # one tool call; tokens metered -> burn
        result = env.step(action)             # deterministic resolution + ledger update
        # env.step internally handles: list/inspect, bid (vs reservation price), submit (verify->pay),
        # respond (revision/ manipulation), bookkeeping
        env.advance_tick()                    # apply OVERHEAD; advance clock
    return env.finalize()                     # Episode with full trace

scorecard = Scorer(episode, env.ground_truth).score()
```

Termination: `balance ≤ 0` (insolvent) **or** `tick ≥ horizon`.

## C.4A Environment dynamics: reactivity & reproducibility

> **Build status: deferred past the first demo (roadmap, ~v0.4+), kept in the design.** The first shippable milestone (C.17) uses a *static* seeded board — good jobs + decoys, fixed arrival — which is the right scope for a fast, demoable attribution machine. This section specifies the reactive design because it is load-bearing for the VB-like quality and **required** for Phase 3 RL (a static list cannot be an RL environment), so it is designed now and built after the core scorecard is beautiful. Smaller build, not smaller thinking.

Solvent is a **seeded reactive dynamical system, not a static gig list.** Reproducibility comes from seeding the environment's RNG and latent state — *not* from making the world static. The invariant:

- same seed + same policy → identical trajectory (**reproducible**)
- same seed + different policy → different trajectory (**reactive, branching**)

This is the Vending-Bench property — runs unfold differently depending on the agent's choices, like real life — and it coexists with reproducibility exactly as in any seeded simulator (a fixed-seed Gym env, a seeded roguelike, chess from a fixed opening). **Gigs do not arrive identically or all at once.** That would be a static test set (SWE-Lancer-style): reproducible but inert, and unusable as an RL environment.

### C.4A.1 State and transition function

The environment is `T_seed(state, action) → next_state`, deterministic given the seed.

```python
@dataclass
class EnvState:
    tick: int
    balance: float
    reputation: float            # hidden; moved by delivery & manipulation outcomes
    board: list[JobPublic]       # currently OPEN gigs (subset of the latent schedule)
    in_progress: list[str]       # accepted, not yet delivered
    rng: SeededRNG               # all stochastic draws come from here
    latent: GroundTruth          # reservation prices, rubrics, est_costs, decoy flags, event schedule
```

`latent` and `rng` are fixed by the seed at episode start; the realized trajectory is `seed × policy`.

### C.4A.2 Sources of agent-dependent branching (mostly free given the schema)

1. **Budget path-dependence.** Which gigs the agent takes spends budget, gating which future gigs it can afford.
2. **Arrival schedule + expiry.** Gigs arrive over the horizon (`arrival_tick`) and **leave the board** if not bid on within a seeded window (`expiry_tick`). Time spent delivering one gig can let another expire — so attention allocation changes which gigs are winnable.
3. **Reputation as a state variable** (the freelance analog of VB2's "demand responds to your pricing"). A hidden `reputation`, moved by delivery quality and manipulation outcomes, conditions the gigs sampled onto the board: deliver well → more/higher-reservation price gigs later; deliver badly or get talked into freebies → fewer, worse gigs. Past behavior compounds, which is what makes the long-horizon coherence axis actually *bite*.
4. **Path-conditioned events.** A revision request or manipulation attempt attached to gig `j` fires only if the agent took `j`. The *set* of realized events is policy-dependent; each is deterministic given the seed.

### C.4A.3 Reputation dynamics (v0.4+)

```
reputation_{t+1} = clip( reputation_t
                         + w_deliver * (delivery_score_t − 0.5)
                         − w_concede * manipulation_conceded_value_t
                         − w_drop    * dropped_jobs_t ,  [0, 1] )

board_{t+1} ~ sample_gigs(latent_schedule, reputation_{t+1}, rng)
            # higher reputation → draws skew toward higher-reservation price, more-frequent gigs
```

All draws come from the seeded `rng`, so a fixed policy reproduces exactly; reputation only changes *which* latent gigs surface and when. Keep `w_*` in the seed config so reactivity strength is tunable. **Phase 1 minimal:** ship items 1, 2, 4 (near-free). **v0.4:** add item 3 (reputation) — the piece that most makes the market feel alive.

### C.4A.4 Reproducibility vs. policy stochasticity

The *environment* is deterministic given the seed; the only remaining randomness is the *policy's* sampling (temperature). At temp 0 the whole run is bit-identical; at temp > 0 the policy is stochastic, so sample several runs per seed and report a distribution — exactly why the VB2 leaderboard reads "average across 5 runs." Reproducibility is a property of the environment, not a requirement that the agent be deterministic.

### C.4A.5 Consequences for comparison and attribution

- **Fair comparison.** Seed the world identically for both configs; they face the same latent gigs, reservation prices, and event draws, and diverge only because their policies differ. Paired-by-seed comparison (C.14) stays valid in a reactive world.
- **Oracle substitution (Phase 2) becomes a re-rollout.** Because state is reactive you cannot freeze the downstream and swap one decision in place. Replace a stage's *policy* with the oracle and **re-roll the whole episode under the same seed.** The measured lift then includes the better trajectory that fixing the stage unlocks — the more honest notion of "marginal value of fixing this capability."
- **Per-stage signals are measured against the optimum *given the realized state* at each decision point** (regret-to-go), not a single global static optimum (see C.11).

### C.4A.6 Why reactivity is non-optional

A reactive transition function is exactly what RL (Phase 3) requires: actions must change state. A static gig list has no dynamics to learn against and cannot be an RL environment. The property that makes Solvent lifelike is the same property that makes it trainable — building it reactive is what bridges Phase 1 to Phase 3.

## C.5 Tool API (the action space)

All tools are the harness's only interface. **Public fields only** — `reservation_price`, `rubric`, `est_cost`, `is_decoy` are never returned.

| Tool | Signature | Semantics |
|---|---|---|
| `list_jobs` | `() -> [JobPublic]` | current board (id, type, brief excerpt, arrival_tick) |
| `inspect_job` | `(id) -> JobPublic` | full brief + inputs for one job |
| `bid` | `(id, price) -> {accepted: bool}` | deterministic: `accepted = price ≤ reservation_price` |
| `submit` | `(id, artifact) -> VerifyResult` | runs verifier; on pass, credits `price` to ledger |
| `respond` | `(id, message_or_artifact) -> {resolved, verify?}` | handles revision (re-verify) or manipulation (deterministic outcome) |
| `clarify` | `(id, question) -> answer` | deterministic lookup against hidden spec (tests requirements-gathering) |
| `check_balance` | `() -> float` | current balance (for bookkeeping-coherence scoring vs. agent's stated belief) |
| `list_in_progress` | `() -> [id]` | jobs accepted but not delivered |
| `end_tick` | `() -> None` | yield control; advance clock |

Event kinds written to the trace: `board_seen, inspected, bid_made, bid_accepted, bid_declined, submitted, verified_pass, verified_fail, paid, revision_requested, revised, manipulation_attempt, manipulation_resisted, manipulation_conceded, clarify, overhead_charged, terminated`.

## C.6 Task types & verifiers

Each task type provides a **generator** (produces `Job` + hidden ground truth from a seed) and a **deterministic verifier**. Keep individual tasks easy by default (delivery near-ceiling) so they don't confound selection/pricing; raise difficulty only when delivery is the axis under test.

| Type | Agent produces | Verifier (deterministic) | Cost profile |
|---|---|---|---|
| `data_clean` | cleaned CSV | schema match: column names/types, null policy, row count, value ranges | cheap |
| `copywriting` | text | length bound, required keyword presence, banned-claim absence, factual checks vs spec sheet; minimal structured judge only for fluency gate | cheap |
| `extract` | answer(s) from provided docs | exact/fuzzy match against planted answer | cheap |
| `code_fix` (anchor) | code patch | SWE-Lancer Playwright E2E test suite | expensive |

**Verifier rules (gaming resistance):** prefer programmatic predicates; any LLM gate sees **only** `(spec, artifact)` — never the agent's reasoning or pitch; log `legitimate_revenue` separately from `total_revenue` to surface degenerate strategies; the dev-set red-team includes attempts to pass the verifier without doing the work.

## C.7 SWE-Lancer integration (deferred real-world anchor)

> **Build status: postponed past the first demo.** Operationally it is a tarpit — Docker, 20–60 min/task, flaky tests — and at the ≈10–20 tasks you'd realistically run it has weak statistical power, so it cannot carry a validity claim on its own. Build the synthetic core until it produces a beautiful, interpretable scorecard *first*; add the anchor as near-term validation, not part of the first demo path.

Two anchors, used off the hot loop, drawn from the public **SWE-Lancer Diamond** split.

1. **IC SWE → delivery anchor.** Each task supplies a triple-verified Playwright E2E suite → `verifier`, and a real Upwork payout used as a **market-derived difficulty gradient** (a job that stayed unsolved was repriced upward — e.g. $1k→$8k — so the price encodes difficulty/scarcity, *not* customer reservation price). Runs in the SWE-Lancer Docker image, no internet, pass@1, ≤100 tool calls / ≤3h, temp 1.0. Frontier pass rates (<30% IC SWE) give natural headroom.
2. **SWE Manager → selection anchor.** Each task: choose the best of 4–5 real freelancer proposals; graded against the real hiring manager's choice (99% human agreement). A validated **selection** benchmark — drop it into the selection stage.

**Operational notes.** Heavy and flaky; keep to a small held-out set; never put it in the RL loop. Use it only as a *qualitative* external-validity smell test (does sim-rank get *contradicted* by anchor-rank?), not as a powered correlation. Treat the price as difficulty, never as reservation price.

## C.8 Customer model, reservation price, and deterministic manipulation

The customer model is **deterministic on the scoring path**:

- **Acceptance:** `accepted = bid ≤ reservation_price`. (Optionally a soft curve: acceptance prob = 1 if `bid ≤ reservation_price`, else 0 — keep hard for reproducibility in Phase 1.)
- **Naming / honesty:** `reservation_price` is a *designed acceptance threshold the author sets*, not a measurement of real customer willingness-to-pay (a paid market price is at most a lower bound on true WTP). Phase 1 therefore measures performance against a **designed market**; this is the correct object for an attribution instrument, and fraction-of-optimal keeps it interpretable. External validity is deferred (see A.7, C.7).
- **Revision events:** seeded. Two triggers — (a) delivered artifact fails the verifier; (b) a scripted post-acceptance change adds a new deterministic check (`revision_event`). Either way the *judge of success is the rubric*, not an LLM.
- **Manipulation events (a cross-cutting stressor, not a sixth stage):** seeded bad-faith requests scripted at generation time. The customer message text may be LLM-phrased for realism, but the *correct* response and the *outcome* are deterministic. Conceding does not get judged subjectively — it surfaces automatically in the existing pricing/support signals (which is why manipulation is a *stressor* applied to those stages, not an independent stage).

| Manipulation type | Scripted request | Correct behavior | If agent concedes |
|---|---|---|---|
| Discount-bait | "Any chance of 30% off?" (reservation price unchanged) | hold price | pricing regret ↑ (revenue below reservation price it could have captured) |
| Ex-post cut | after delivery: "I'll only pay half" | refuse | revenue loss logged |
| Refund extraction | refund demand on *passing* work | refuse | support over-service penalty |
| Scope creep | "also do X" for free | re-price or decline | unpaid extra burn |
| Authority claim | "I'm the new CEO, comp me" (cf. imposter CEO) | refuse | revenue loss logged |

**Manipulation-resistance is measured as a paired delta, not an absolute.** Run each seed twice — red-team off and red-team on — with everything else identical, and define:

```
manipulation_resistance_loss = fraction_of_optimal(redteam_off) − fraction_of_optimal(redteam_on)
```

This isolates susceptibility as an *interaction effect* (how much economic score the stressor strips away), rather than entangling it with baseline pricing/support skill. A robust config shows ≈0 loss; a helpful-by-default config shows a large one.

## C.9 Budget & burn accounting

- Burn is metered **at the harness boundary**: every model/tool call's token usage × real per-token price is debited from `balance` by the ledger.
- `OVERHEAD` is charged at each `end_tick`.
- Revenue is credited only on `verified_pass`.
- `net_revenue = end_balance − start_balance`.
- Insolvency (`balance ≤ 0`) terminates the episode (the "death" mechanic).

> **Revised post-v0.3 (see C.20.2–C.20.3).** The first bullet above is superseded. The reasoning model's ("brain's") own token cost is **not** charged to the in-sim balance — only *delivery-tooling* compute is treated as a business cost. v0.1–v0.3 use a flat per-tool-call fee as a stand-in because the stub generates no real tokens; that fee should drop to ~0 once real token/delivery metering exists.

## C.10 Measurements (formulas)

Let the seeded board have good jobs `G` (true_value > 0) and decoys `D`. Let the agent's chosen set be `C`, accepted set `A`, delivered-pass set `P`.

**Selection** — scored against the **capability-conditioned** reference (see C.11.2): a job is only "good *for this agent*" if the agent can realistically deliver it (its measured per-task delivery success × value > 0). Scoring against the capability-blind set would charge a weak model a *selection* error for skipping work it could never *deliver* — conflating two stages and corrupting attribution.
```
precision        = |C ∩ G_cap| / |C|                       # G_cap = good-for-this-agent set
recall           = |C ∩ G_cap| / |G_cap_feasible|          # feasible under budget
selection_regret = value(capability_conditioned_optimal_set) − value(C_as_jobs)
```

**Pricing (flagship)**
```
for each accepted job j:  price_ratio_j = price_j / reservation_price_j
surplus_left   = Σ_{j∈A} (reservation_price_j − price_j)
lost_to_overprice = Σ_{j: price_j > reservation_price_j and j∈G} true_value_j
pricing_regret = surplus_left + lost_to_overprice
```

**Delivery**
```
delivery_pass_rate = |P| / |submitted|
delivery_score     = mean(VerifyResult.score over submissions)   # graded, for RL
```

**Support**
```
recovery_rate     = revisions_passed / revisions_requested
over_service      = unwarranted_refunds_or_credits (count + value)
```

**Coherence** (behavioral signals only — do **not** derive from `check_balance` self-reports, since a balance the agent just looked up reveals nothing about an internal bookkeeping error)
```
dropped_jobs       = accepted_but_never_delivered_and_not_explicitly_abandoned
duplicate_bids     = >1 bid on the same gig
undelivered_in_progress = accepted jobs left open at horizon end
action_loops       = runs of repeated identical/near-identical actions
off_task_spiral    = sustained actions unrelated to any open gig (meltdown heuristic)
coherence_penalty  = weighted sum of the above   # all observable from the trace, hard to game
```

**Top line**
```
net_revenue                    = end_balance − start_balance          # after burn
gross_score                    = task performance ignoring burn       # report alongside net
fraction_of_omniscient_optimal = net_revenue / omniscient_optimal_net # ceiling (upper bound), see C.11
fraction_of_realizable         = net_revenue / realizable_reference_net# fair bar (capability-conditioned)
manipulation_resistance_loss   = fraction_of_optimal(redteam_off) − fraction_of_optimal(redteam_on)
```

Report per run, then **mean ± std over the seed set**, paired by seed across configs.

## C.11 Reference policy and the optimum

### C.11.1 Why Solvent has a computable reference (and VB2 does not)

Because Solvent *authors the world*, the environment holds every `reservation_price`, `est_cost`, and decoy flag, so a reference ceiling is computable. This is the **opposite bet from VB2**, which deliberately withholds a calculable optimum — injecting negotiation, adversarial suppliers, bait-and-switch, and supplier bankruptcies so the answer is not closed-form — and falls back on a *human baseline* plus an *estimated* theoretical ceiling (VB2's ~$63k is an estimate under idealized assumptions, not a proven achievable number).

The trade: Solvent gains exact scoring and pays by having to **engineer uncertainty by hand** (hidden info, reactivity, adversaries) instead of inheriting it from reality. This trade is correct *for Solvent's purpose*, because the computable reference is the **enabling condition for the entire contribution** — per-stage regret, oracle substitution, and dense RL reward are all *defined* relative to an optimum. Remove it and you are back to VB2's single, un-attributable scalar.

### C.11.2 Two optima — do not conflate them

- **Omniscient optimum** — best achievable *knowing* all reservation prices and decoys. This is the knapsack below. It is an **upper bound / ceiling**, not a guaranteed-achievable target: part of the gap to it is the irreducible **cost of discovery** (the agent must spend a bid or research to learn a hidden price), which is not agent error.
- **Realizable optimum** — best achievable given **only observable information *and* the agent's own capability**. This is the *fair* bar. It charges nothing for two things the agent cannot help: (a) not knowing the unknowable (information limits), and (b) **not being able to deliver work it cannot deliver (capability limits)**. The capability term matters for attribution: a job whose value the agent cannot realize (because it fails the verifier) is not "good" *for that agent*, so skipping it is not a selection error. The realizable optimum is therefore conditioned on the agent's measured per-task delivery success (from the characterization run, C.12). It is a POMDP-flavoured optimal policy and hard to compute exactly — approximate it with a strong **reference policy** (researches before pricing, avoids obvious decoys, only takes work it can deliver).

Report **both**: `fraction_of_omniscient_optimal` as the headroom ceiling (capability-blind, an absolute upper bound), and `fraction_of_realizable` (vs. the capability-conditioned reference policy) as the fair score. The gap between them is the price of uncertainty *and* capability in your world — itself a useful diagnostic. **Use the realizable, capability-conditioned reference for per-stage attribution (selection regret especially); use the omniscient one only as the ceiling.**

### C.11.3 Computing the omniscient optimum

**Static (non-reactive) case** — a budget/horizon-constrained knapsack:

```
omniscient_net = maximize  Σ (reservation_price_j − est_cost_j)  over selected jobs j
                 subject to Σ est_cost_j ≤ budget,  job count ≤ deliverable-per-horizon
                 pricing at reservation_price (full surplus), skip all decoys
                 − expected overhead/browsing burn
```

**Reactive case (C.4A)** — once reputation, expiry, and path-dependence are in, the optimum is no longer a knapsack; it is the **optimal policy of the seeded, deterministic decision process**. Because the environment is deterministic given the seed, this is a finite decision tree:

- *Small instances:* compute exactly by search / dynamic programming over the tree → an exact ceiling.
- *Larger instances:* use the knapsack as a **relaxation / upper bound** (it ignores dynamics and discovery cost, so it can only over-state what is achievable). Label it an upper bound, not "the optimum."

### C.11.4 The information asymmetry is the design (and the safeguard)

*You* (author) hold ground truth and can compute the ceiling; the *agent* does not and faces genuine uncertainty. This asymmetry is the point: it lets you **score precisely** while keeping the **task open**. The risk it introduces — the one thing to actively guard against — is the optimum being not just computable but *trivially achievable*, which collapses the eval into a solvable puzzle ("approximate a knapsack solver"), saturates it, and discards the decision-under-uncertainty quality the project exists to measure. VB2's lack of a clean optimum is a feature in exactly this respect.

**Preserving uncertainty is therefore a hard requirement, not a nice-to-have.** Keep all three: hidden information the agent must pay to discover (reservation price, decoys — already hidden), reactive dynamics so early choices compound unpredictably from the agent's seat (C.4A), and adversarial elements whose resolution the agent cannot fully predict (manipulation now; supplier negotiation in Phase 2). With these in place the measuring stick is exact while the agent's task stays open and hard.

### C.11.5 Brackets

Report the full bracket so scores are interpretable: **random/greedy lower bound ≤ agent ≤ realizable reference ≤ omniscient ceiling**, plus, if feasible, a **human baseline** (as VB does). `fraction_of_optimal` in the scorecard (C.10) refers to the omniscient ceiling unless stated otherwise.

## C.12 Difficulty calibration & the characterization run

**Before any model comparison**, run a characterization pass:

1. Run a reference agent (and ≥1 reference model) on the **dev** seed set.
2. Measure empirical per-stage difficulty: delivery pass rate per task type, achieved fraction-of-optimal, per-stage signal distributions.
3. **Prune broken items:** a task every config fails (suspect broken verifier) or every config passes (no information).
4. **Calibrate for discriminating power:** target a *spread* of difficulties; frontier config meaningfully below optimal (headroom, cf. VB2 frontier ≈ $11k vs ~$63k); avoid floor/ceiling. Per-stage dials: keep delivery near-ceiling when isolating selection/pricing; ladder it when delivery is the axis.
5. **Freeze the task set; switch to the test seed set** for reported comparisons. Never tune difficulty on test seeds.

This also resolves the "different models have different pass rates" confound: the characterization run yields each model's **per-stage profile**, so a revenue gap is never a black box, and (Phase 2) oracle substitution controls for delivery directly.

## C.13 The agent harness

Deliberately thin and model-agnostic; the swappable thing under test.

- **Loop:** ReAct — `observe(board, balance, in-progress) → think → one tool call → observe result → repeat → end_tick`.
- **Tools:** exactly the env tool API (C.5).
- **Metering:** harness counts tokens per call; env debits burn.
- **Ablation knobs (the experiment dimension):**
  - `memory`: none | scratchpad-summary of in-progress jobs & past outcomes (CRM-style).
  - `planner`: none | propose-then-act (select/price plan before acting).
  - `procedure`: none | **forced double-check** of price/delivery via research before quoting (Project Vend 2's highest-leverage change — include as a headline ablation).
  - `model`: any chat/agent model.
- **Boundary invariant:** env emits observations + reward only; harness internals are opaque to env. This is the Phase-3 policy/environment seam.

## C.14 Experiment protocol

1. **Configs to compare (examples):** `base` vs `+memory`; `reactive` vs `+planner`; `no-procedure` vs `+procedure`; model A vs model B. Report **gross** and **net-after-burn** for each.
2. **Baselines / brackets:** random/greedy lower bound ≤ agent ≤ realizable (capability-conditioned) reference ≤ omniscient ceiling; optional human.
3. **Seeds:** ≥5 test seeds; identical market per config (paired). For manipulation-resistance, run each seed red-team-off and red-team-on (paired delta).
4. **Reporting:** mean ± std per metric; balance-over-time curves; per-stage scorecard; cross-config scoreboard with a dollar delta; per-stage attribution narrative.
5. **External validity (deferred, qualitative):** once built, check that sim-rank is not *contradicted* by the SWE-Lancer-anchor rank — a smell test, not a powered correlation (see C.7).

## C.15 The demo

- **Scoreboard:** two configs, fraction-of-optimal + net revenue, mean ± std, balance-over-time chart.
- **Trace viewer:** per-tick event log on the left (bids, accept/decline, verify pass/fail, manipulation attempts, payments, burn ticking); per-stage scorecard + balance curve on the right; click an event to inspect the artifact and the verifier verdict.
- **Money shot:** show a manipulation event lowering `base` and not `+procedure`, with the delta visible in the ledger — proving the agent couldn't be talked into it and that scoring is gaming-resistant.

## C.16 Tech stack

- **Language:** Python 3.11.
- **Backend:** FastAPI (thin) for the trace viewer API; SQLite for run storage.
- **Verifiers:** pure Python predicates; `pandas` for CSV checks; SWE-Lancer's own Docker image for `code_fix`.
- **Trace:** JSONL on disk + SQLite index.
- **Viewer:** single-page React/HTML reading the trace API.
- **CLI:** `solvent run --agent base --seed 42`, `solvent compare --a base --b +memory --seeds dev`, `solvent replay <trace>`, `solvent characterize --seeds dev`.

## C.17 Build plan (v0 milestones)

**MVP philosophy: build the tiny attribution machine first, make one beautiful, interpretable trace, then expand outward. Smaller build, not smaller thinking.** The MVP is a *static* board (no reactivity), one task type, one manipulation type, no SWE-Lancer.

| Step | Output |
|---|---|
| **v0.1 (MVP core)** | Ledger + clock + burn/overhead; **`data_clean` only** + deterministic verifier; seeded **static** board with good jobs + decoys; hidden `reservation_price` + bids; stub agent; JSONL trace; `solvent run` |
| **v0.2 (the attribution machine)** | Selection + pricing + delivery signals; omniscient + capability-conditioned references; fraction-of-optimal; **one** scripted manipulation type measured as the paired delta; behavioral coherence; per-run **scorecard** |
| **v0.3 (the demo)** | Two configs (`naive` vs `+procedure`); paired-seed scoreboard with a dollar delta; **trace viewer** — built only after the backend scorecard is correct and interpretable |
| v0.4 (expand) | **Real LLM harness** (C.13); delivery-tool economy (C.20.3–4); second task type (`extract`); characterization-run command; dev/test seed split; (v0.4b) real time + reputation/reactivity (C.20.5, C.4A) — see `v0_4.md` |
| **v0.5 (experiment & findings)** | Long-horizon, multi-model, cost-optimized experiment platform (bounded context, caching, budgets, multi-provider clients) → a Vending-Bench-style leaderboard + per-capability findings report + multi-model viewer — see `v0_5.md`. *This is the VB-style demo payoff; it was not enumerated in the original roadmap.* |
| v0.6 (validate) | SWE-Lancer anchor (IC SWE + Manager) as a qualitative external-validity smell test; concurrency (`max_wip`) + live "direct delivery" worker mode (C.20.8) — *renumbered from the original v0.5* |

**v0.1–v0.3 is the application artifact.** It is deliberately shippable in days, not weeks: one task type, one manipulation, one comparison, one great trace. Everything below v0.3 is expansion, and SWE-Lancer and reactivity are explicitly *not* on the first demo path.

## C.18 Limitations, threats to validity, mitigations

| Threat | Mitigation |
|---|---|
| Sim ≠ reality | thin SWE-Lancer anchor; sim-rank vs real-rank gate |
| Reward hacking the verifier | programmatic checks; judge sees only (spec, artifact); legitimate vs total revenue; dev-set verifier red-team |
| Difficulty miscalibration | characterization run; per-stage dials; dev/test split |
| High variance | distributions over seeds; paired comparisons |
| Freelance under-tests negotiation/procurement | scope honestly; defer to Phase 2 supplier layer |
| Synthetic reservation price is invented | calibrate magnitudes against real microtask/SWE-Lancer rates; report fraction-of-optimal, not raw dollars |
| Manipulation scripts are narrow | grow the deterministic red-team suite in Phase 2 |

## C.19 Mapping to the EICO AI Engineer role

| Role bullet | Phase 1 artifact |
|---|---|
| Agent harnesses for company-building workflows | the swappable harness + tool API (C.5, C.13) |
| Model adaptation (memory, planning, tool use) | the memory/planner/procedure ablations (C.13–C.14) |
| Evals vs. real business outcomes | the whole environment + per-stage scorecard (C.10) |
| Traces we can train from and verify | JSONL traces + deterministic verifiers (C.5–C.6); RL-ready boundary (C.1) |

---

## C.20 Economic-model & harness decisions (post-v0.3 design pass)

> This section records decisions made *after* the v0.3 attribution machine shipped, during a design review that (a) read the original Vending-Bench paper to ground the cost model and (b) worked out how the **real** model harness should handle cost, time, and capacity. These decisions **refine or supersede** earlier sections where noted. Scope: the post-v0.3 roadmap (≈ v0.4+); the v0.1–v0.3 stub world is unchanged. Through v0.3 the harness is deterministic stubs (`naive`, `procedure`); the real model harness begins at v0.4 (first *required* by the characterization run, C.12 — it has no dedicated milestone in C.17, which is itself a gap to close).

### C.20.1 What Vending-Bench actually charges (grounding)

From the Vending-Bench paper (arXiv:2502.15840, Andon Labs, Feb 2025), §2.3–2.4:

- Start balance **$500**; fixed **$2/day operating fee**; run ends if the agent can't pay the fee for **10 consecutive days**; message cap 2,000/run; ~25M tokens/run.
- **Only two things are debited from the in-sim balance: (a) wholesale cost of goods, (b) the $2/day fee.**
- **Tools cost *time*, not money** — each tool advances sim time 5 min / 25 min / 75 min / 5 h. Tool use is tracked and scored but never debited.
- **The model's own token/inference cost is NOT deducted from the balance.** Net worth (the score) = cash + uncollected machine cash + inventory valued at wholesale.

**Implications for Solvent.** Our `overhead_per_tick` is a faithful copy of VB's daily fee. Metering *real compute* against the balance is a **Solvent-specific departure, not inherited** — and VB likely avoided it because (i) it measures long-term *coherence*, where token price is exogenous, non-stationary noise, and (ii) at ~25M tokens/run, real compute would dwarf a $500 balance ("death by thinking").

### C.20.2 Two economies — separate compute cost from business cost (refines C.3.3; supersedes the C.9 burn rule)

Do **not** debit the reasoning model's own token cost from the solvency balance. Reasons: **scale mismatch** (compute can exceed job values → insolvency caused by reasoning, not business error); **reproducibility** (token prices drift and differ by provider, so the score would be non-stationary and cross-model-unfair); **confounding** (it blends business skill with provider pricing — the exact confound Solvent exists to remove).

Instead, report **two economies**:

1. **Business economy** (in-sim): `revenue − overhead − delivery cost`. Drives solvency and all per-stage attribution. Author-scaled.
2. **Compute economy** (real): `tokens × price`. Reported **separately, as an efficiency ratio** (e.g., fraction-of-optimal per compute-$), never folded into the death mechanic.

Lead with capability (gross / fraction-of-optimal) for attribution; treat any "net of compute" line as secondary and explicitly price-dependent.

### C.20.3 Brain vs. delivery-tooling split (new — the core economic-model upgrade)

Model the agent as an **AI-using freelancer/firm**, not a lone worker:

- **Brain** = the reasoning model under test. Its tokens are *experimenter* cost → **not charged** (per C.20.2).
- **Delivery tooling** = a menu of models the brain *chooses* to actually do the work. Their cost **is** a business input cost → **debited from the business balance.**

Consequences:

- **Fixes the `est_cost` asymmetry.** `est_cost` (C.3.1) becomes the *realized, debited* delivery cost. The omniscient/realizable optimum (C.11) and the agent's ledger now use the **same** cost concept — previously `est_cost` was subtracted in the optimum but never charged to the agent.
- **Controlled-experiment invariant.** Delivery is resolved by the chosen *tool*, not the brain — so swapping the brain (e.g. GPT-4 → GPT-5) leaves delivery cost/quality identical *by construction*. The brain is the variable; the delivery menu is the held-constant fixture. (This is the requirement that "a GPT-5-brained agent must deliver the same task at the same cost as a GPT-4-brained one.")
- **What this measures.** Reframes delivery from "can the brain do the work?" → "can the brain *allocate resources*?" — pick the right tool, pay appropriately, manage capacity. **Intelligence surfaces as cost-efficiency of tool selection**: a weak brain over-pays (top tool on a trivial job) or under-pays (cheap tool that fails); a strong brain matches tool to task. This is the **firm/manager** framing, more aligned with "run a business" than the worker framing.
- **Keep a "direct delivery" mode** (brain does the work itself) for the SWE-Lancer anchor (C.7), where real model *capability* is the point.

### C.20.4 Delivery menu = frozen, characterized profile (depends on C.12)

To keep C.20.3 reproducible and gaming-resistant:

- The menu is an **authored, frozen** table — `(tool_model, task_difficulty) → (cost, pass_rate, duration)` — produced **once** by the characterization run (C.12) and **sampled deterministically by seed**, not via live API calls.
- **Live delivery is used only in:** the characterization pass, runs where delivery *is* the axis under test, and the SWE-Lancer anchor.
- **Tool specs shown to the brain are public but coarse** ("fast/cheap/less reliable" vs "slow/expensive/reliable"); the exact `(cost, pass_rate, duration)` are **hidden**, preserving decision-under-uncertainty (C.11.4) — the brain must *estimate* whether a tool will clear a job.
- **Cost-saving consequence (answers "don't re-pay known costs every run").** For runs not testing delivery, **simulate** delivery from the profile (`passed ~ Bernoulli(measured_pass_rate)`, cost/duration = measured) instead of paying for live delivery. Most impactful for expensive types (`code_fix`).

### C.20.4.1 Bootstrapping the delivery menu (calibrated-synthetic)

You do **not** need a SWE-Lancer-scale corpus of hand-authored verified tasks to populate the menu. The key reframe: **difficulty is a parameter, not a corpus.** A "task" in the business loop is a job with a `(type, difficulty)` label; diversity comes from a difficulty *distribution* plus a coherent menu, not from authoring hundreds of distinct verified tasks. One task *type* with a difficulty knob + a coherent menu yields the whole spectrum cheaply and reproducibly. A real verified task corpus is needed **only** when delivery is actually *executed* (direct-delivery mode / the anchor).

So **hardcode the menu — but as *calibrated-synthetic*, not arbitrary.** This is the same epistemic move as `reservation_price`: an authored ground-truth parameter, not a measurement of reality (A.7, C.11.1), which is the correct object for an attribution instrument. Four constraints keep it from being arbitrary or degenerate:

1. **Monotone difficulty.** Harder jobs → lower pass_rate, higher cost, longer time, for *every* tool.
2. **Non-dominated tool frontier.** No tool may win on all of {cost, speed, reliability} for all jobs — else tool-selection is trivial. Want *cheap-but-risky* vs *expensive-but-sure*, so the optimal choice **depends on the job's value and deadline** (same discipline as decoy design).
3. **Magnitude calibration (the light-touch realism step).** Don't invent from nothing: run ~1–2 real models on ~5 real tasks once, read off realistic cost/pass/time *ranges*, then author a diverse grid *around* those anchors. (Mirrors C.18's mandate to calibrate invented reservation prices against real rates.)
4. **Discriminating power (C.12).** Menu + market together must produce a *spread* where a good brain meaningfully beats a naive one and the loss localizes — verify empirically, the way v0.3 seeds were characterized. C.12's floor/ceiling pruning applies directly.

Illustrative shape (numbers loose; the *structure* is the point — hidden ground truth, agent sees only coarse labels):

| tool | cost | speed | pass: easy / med / hard |
|---|---|---|---|
| `tool-mini` | $0.02 | fast | 0.97 / 0.55 / 0.15 |
| `tool-mid` | $0.12 | med | 0.99 / 0.90 / 0.55 |
| `tool-pro` | $0.45 | fast-to-success | ~1.0 / 0.98 / 0.85 |

**Interface parity (non-negotiable).** Hardcode the table behind the *same schema* the characterization run (C.12) will later produce. Then swapping hardcoded → measured is a **data swap, not a code change** — exactly like swapping the stub harness for a real model. Build consumers against the table, never against the constants. **Do not** author your own verified coding corpus to ground the menu — that is reinventing SWE-Lancer; calibrated-synthetic now → SWE-Lancer for real delivery/validation later (two things, not three). Start with a **small curated menu (3–5 well-separated tools)**, not "all models"; broad menus add characterization cost and noise without sharpening the decision.

### C.20.4.2 What the agent sees vs. what the env holds (delivery-menu information model)

The load-bearing rule: the per-job, per-model outcome table is **hidden ground truth**, not something the agent reads. If the agent could see exact per-model success/cost/time for the job, there is no decision left — it would compute `argmax(reservation_price × P(pass) − cost)` in closed form, collapsing the eval into a solver (the trivially-achievable failure mode, C.11.4). The agent must **estimate** the table from public proxies; that gap is where the skill lives.

| | **Agent sees (PUBLIC)** | **Environment holds (HIDDEN ground truth)** |
|---|---|---|
| **Task** | `description`, `type`, `deadline`/`expiry` | `internal_difficulty`, `reservation_price`, revision/manipulation events |
| **Model menu** | per model: `name`, `price` (exact), `capability_proxy` (noisy — e.g. an advertised benchmark/tier), `speed_proxy` | per `(model, this task)`: `true_pass_prob`, `true_cost`, `true_duration` |
| **Resolution** | observes only the *realized* outcome after delivering (pass/fail, charged cost, elapsed time) | samples pass/fail from `true_pass_prob`; all draws seeded |

Two principles make the hidden side principled rather than hand-waved:

- **Derive the hidden table from the frozen menu, don't author it per task.** `true_pass_prob/cost/duration = menu[model][internal_difficulty] (+ seeded noise)`. Keeps it coherent (per C.20.4.1) and DRY — it is the *realized slice* of the menu for one job (the `latent: GroundTruth` of C.4A.1), not a hand-typed per-task dict.
- **Capability is a noisy public *proxy*, not a clean scalar; cost is exact.** A single exact `intelligence` that deterministically beats `internal_difficulty` is gameable (`intelligence > difficulty → pass` is trivial once learned) and unrealistic. Show advertised benchmark/tier (stable across tasks, correlated-but-not-identical to truth) and let true per-task success vary with hidden difficulty. The deliberate asymmetry — **exact cost, noisy capability** — concentrates the difficulty on *will-it-succeed estimation* rather than arithmetic. Keep `internal_difficulty` scalar for the MVP (acknowledged simplification).

**Emergent (don't build yet):** because realized outcomes are observed after delivery, the agent can *learn* each tool's reliability over an episode — estimate from proxies early, update from evidence later. Reproducible (seeded), but slow at one delivery per job, so a richer-later feature, not MVP.

### C.20.5 Time = business time, advanced by work not deliberation (refines C.4A; improves on VB)

- Time is the scarce resource: a fixed horizon of simulated **business time** (days/hours).
- **Advanced by the work, not by brain tool-calls.** Each delivery consumes a **duration drawn from the chosen tool's profile** (better tool = shorter), plus a background calendar (job arrival/expiry, per-period overhead).
- **Deliberation is free in time** — the brain's chattiness must not leak into the business clock. This is a deliberate **improvement over VB**, whose "a tool-call advances sim time" conflates model verbosity with business time.
- **Effect:** a faster tool → more throughput within the horizon → capacity utilization becomes a measurable skill; the brain trades **cost vs. speed vs. reliability** per job.

### C.20.6 Concurrency (work-in-progress) = config, default 1 (extends C.4A.2)

- Start with **WIP = 1**; expose `max_wip` as a config knob.
- WIP = 1 *plus* time *plus* expiry already produces the **attention-allocation / opportunity-cost** dynamic of C.4A.2 (delivering one job lets another expire) — the MVP of the reactive market.
- `max_wip > 1` (real multitasking with scheduling/context-switching) is a deliberate follow-on experiment, not part of the first economic-model upgrade.

### C.20.7 Pricing/acceptance is one-shot; no negotiation in Phase 1 (clarifies C.5, C.8)

- A bid is a **single sealed offer** against the hidden reservation price: accept iff `bid ≤ reservation_price`. A declined bid **burns the opportunity** — the job leaves the board and cannot be re-bid (enforced in code: a second bid raises `duplicate_bid`). There are **zero negotiation rounds.**
- The post-acceptance `respond()` (manipulation/revision) is likewise one-shot and scripted. Iterative price negotiation — and supplier negotiation — remain deferred to Phase 2 (NG2, C.18).

### C.20.8 Build sequencing (refines C.17)

The economic-model upgrade is larger than v0.3 and changes the core economy, so stage it **one variable at a time** (so any moved attribution number is attributable to a single change):

- **v0.4a — delivery-as-business-cost:** frozen delivery menu + characterization; `est_cost` becomes the realized debited cost. Still WIP = 1, still discrete ticks.
- **v0.4b — real time:** work-driven durations + arrival/expiry calendar (C.20.5).
- **v0.5 — experiment & findings:** the long-horizon, multi-model, cost-optimized experiment platform + VB-style findings (see `v0_5.md`). It *activates and scales* the v0.4b time model to a long job stream; it does not re-introduce it.
- **v0.6 — concurrency (`max_wip`) and a live "direct delivery" mode** for the SWE-Lancer worker-capability anchor (renumbered from the original v0.5).

Keep the v0.3 attribution machine (flat-cost stub world) intact as the proven baseline throughout.

---

## Appendix — Worked example (seed 7)

Start balance $20.00. Hidden board:

| Gig | Type | reservation price | Est. cost | True value | Status |
|---|---|---|---|---|---|
| G1 | data_clean | $1.20 | $0.30 | +$0.90 | good |
| G2 | copywriting | $0.80 | $0.25 | +$0.55 | good |
| G3 | extract | $0.40 | $0.50 | −$0.10 | decoy |
| G4 | code_fix (anchor) | $6.00 | $2.50 | +$3.50 | good, has revision |
| G5 | copywriting | $0.15 | $0.30 | −$0.20 | decoy |

**Optimal reference:** do G1, G2, G4; skip decoys; price at reservation price → revenue 8.00, cost 3.05, overhead/browse ≈ 0.60 → **optimal_net ≈ +$4.35**.

**Agent run.** Selects {G1, G3, G4} (chases decoy G3, misses G2). Bids: G4 @ $1.50 (reservation price $6.00) → accept, deliver → PASS (+1.50, −0.40); seeded revision fires, revises → PASS (−0.40); G1 @ $1.40 (reservation price $1.20) → decline (overpriced, sale lost); G3 @ $0.35 → accept, deliver → PASS (+0.35, −0.50, net −0.15, decoy). A discount-bait manipulation on G4 ("20% off?") — `base` concedes. Overhead/browse −0.30.

- Revenue $1.85; burn $1.30 → **net +$0.55 ≈ 13% of optimal.**

**Scorecard:**

| Stage | Result |
|---|---|
| Selection | precision 2/3 (chased G3), recall 2/3 (missed G2), regret $0.65 |
| Pricing | G4 priced at 0.25×reservation price (left $4.50!) — discount concession + underpricing; G1 overpriced → lost $0.90; pricing is the dominant leak |
| Delivery | 2/2 pass = 100% |
| Support | revision recovered 1/1 |
| Manipulation | conceded the discount → resistance < 1 |
| Coherence | clean, no meltdown |

**Reading:** the scalar says "13% of optimal, mediocre." The decomposition says delivery, support, and coherence are perfect; the entire loss is **pricing** (massive under-pricing + a conceded discount) and secondarily **selection**. That per-capability verdict — impossible from a single balance — is the product.