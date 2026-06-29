# Solvent — Three-Phase Plan

*An economic eval-and-RL environment for company-building agents.*

**Author:** [you] · **Status:** Planning · **Audience:** EICO AI Engineer application + internal build doc

---

## 0. One-paragraph thesis

Today's strongest "agent earns money" artifacts split cleanly into two halves that never meet. **Automaton** (Conway Research) is a sophisticated agent *harness* — wallet, self-modification, replication, survival loop — deployed straight into the open internet, where it is unmeasurable and unreproducible. **Vending-Bench 2** and **SWE-Lancer** are the opposite: real, scored *environments*, but they emit a single confounded number (final balance, dollars earned) that tells you *that* an agent failed, never *which capability* failed or by how much. **Solvent builds the missing thing: a reproducible, economically-scored environment that decomposes "run a business" into per-capability, ground-truth-attributable signals — first as an evaluation, ultimately as a reinforcement-learning environment you can train a harness against.** Phase 1 instantiates it in a freelance vertical; Phase 2 generalizes it into an Automaton-class business eval; Phase 3 turns it into a training environment that measurably improves an Automaton-style harness.

---

## 1. The framing that the whole plan rests on

### 1.1 Harness vs. environment

- A **harness** is the scaffolding around the model: the control loop, tools, memory, planning, self-modification, metering.
- An **environment** is the world the harness acts in *plus the signal that scores it*: reproducible state, a reward, ground truth.

| System | Harness | Environment | Gap |
|---|---|---|---|
| Automaton | Sophisticated (wallet, replication, self-mod, survival) | None — the open internet | No reproducibility, no reward, no ground truth |
| Vending-Bench 2 | Simple (fixed ReAct loop, ~handful of tools) | Strong (seeded sim, 1-year horizon, balance score) | Single confounded scalar; ~60–100M tokens/run (un-trainable) |
| SWE-Lancer | Simple scaffold (Docker, no internet, pass@1) | Strong (real Upwork tasks, E2E-test-graded, real prices) | Single-task delivery only; no operation/horizon |
| **Solvent** | Swappable (the thing under test) | **Built here** | — |

**Solvent supplies the environment**, so that harnesses of Automaton's class become measurable and, in Phase 3, trainable.

### 1.2 Money is a confounded scalar

In an open "make money" task, net revenue is a product of every capability at once:

```
net_revenue ≈ f(demand-finding, selection, pricing, delivery, support, coherence, ...)
```

A single dollar figure is the *sum* of all of them. A low number is uninterpretable: the same bad outcome could be perfect selection + broken pricing, or perfect pricing + broken selection. **Vending-Bench's design works precisely by setting every term but coherence to ≈1 (the vending business is trivial), so balance ≈ coherence.** You cannot replicate that trick while broadening the task, because broadening *is* letting the other terms vary. The only way forward is to **un-sum the scalar**: score each stage against ground truth, and attribute failures per-capability.

### 1.3 The decomposition is not arbitrary — it is the observed failure taxonomy

The stage decomposition maps one-to-one onto how real deployed agents fail (Project Vend 1 & 2, Vending-Bench 2):

| Stage | EICO's seven questions | Observed real-world failure |
|---|---|---|
| Selection / demand-finding | Find demand; decide what to do | Project Vend: ignored a $100 offer for a $15 item |
| Pricing / offer | Make an offer; turn funding into revenue | Sold high-margin items below cost; overpaid suppliers ($2.40 cans); "decided like a friend who wants to be nice" |
| Delivery | Deliver the work | SWE-Lancer: frontier models fail the majority of IC SWE tasks |
| Support / negotiation | Handle support; improve after failure | Talked into discounts, free items, ex-post price cuts (staff + WSJ) |
| Coherence | Operate over a long horizon | Identity crisis ("blue blazer"); CEO/agent spiraling into "eternal transcendence" overnight |

This is the central selling point: **Solvent measures, per-capability and deterministically, exactly the failures that Project Vend and Vending-Bench can only observe in aggregate.**

### 1.4 Positioning (why this is defensible)

Andon Labs and Anthropic own "realistic business simulation" and "real-world deployment" — VB2 already has adversarial suppliers, refunds, a frontier leaderboard, a monthly improvement trendline, and a multi-agent Arena; Project Vend is on phase two with multiple cities and a CEO agent. **Do not compete on realism.** Solvent's wedge is the thing none of them do: **attribution** (per-stage ground-truth signals + oracle substitution) and **RL-usability** (dense reward + cheap rollouts). The literature is the evidence for the unmet need: everyone can see *that* agents fail; nobody's harness tells them *which capability* failed and *what fixing it is worth*.

---

## 2. Cross-cutting design principles (hold across all three phases)

1. **Determinism on the scoring path.** LLMs may *dress the world* (write a job posting, phrase a complaint) but **only deterministic checks move money or decide outcomes.** This is what gives the "money from outside the lab" property in reproducible form, and it is directly validated by Project Vend's #1 exploit (humans talking the agent into freebies). If an LLM judged payment, you'd be measuring persuasion susceptibility, not economic skill.
2. **The environment holds ground truth.** Because the market is generated, you know the hidden reservation price, the true value of every opportunity, the rubric, and the decoy flags. Every stage decision is scored against the *knowable-optimal* decision at that stage.
3. **Attribution over aggregation.** Never report only a scalar. Report per-stage signals and (Phase 2+) oracle-substitution lift per capability.
4. **Fraction-of-optimal, not raw dollars.** Compute the optimal reference policy from ground truth; normalize. Absolute dollars in a synthetic economy are meaningless; "% of optimal" is comparable and interpretable (cf. VB2's frontier ≈ $11k vs ~$63k optimal).
5. **Reproducibility.** Seeded markets; identical conditions per config; distributions over seeds, not point estimates (variance is itself a finding). Dev/test seed hygiene — never tune difficulty on evaluation seeds.
6. **Breadth as a knob.** VB is the breadth-1 corner of Solvent. You can always collapse to "one fixed business, long horizon, coherence-only" or widen to the full open loop. Difficulty is a per-stage dial, so a hard stage never silently confounds another.
7. **Sim + thin real anchor.** A pure sim measures performance-in-your-sim. Keep a small real-world calibration set (SWE-Lancer) and confirm sim-rank correlates with real-rank. This is the VB ↔ Project Vend pairing, in one codebase.

---

## 3. Phase 1 — Upwork freelance operation eval environment

> Full PRD + technical design in the companion document. Summary here.

**Goal.** Ship a reproducible, deterministically-scored environment in which a swappable agent harness runs a **freelance operation as a business over a horizon** — discovering gigs, selecting which to bid on, pricing them, delivering, and handling support/manipulation — scored on net revenue *and* an embryonic per-stage decomposition.

**Why freelance.** It is the only online vertical where the two hardest-to-build parts of an economic eval are **objective and largely handed to you**: a delivery verifier and a reservation price. Every other vertical (dropshipping, content, SaaS) forces you to invent and defend a demand model — the reskin/confound trap. SWE-Lancer hands you both: real Upwork prices (a market-derived *difficulty gradient*, not a measured reservation price) and triple-verified E2E tests (gaming-resistant verifier), and it has *two* usable anchors — IC SWE tasks (delivery) and SWE Manager tasks (selection-against-ground-truth).

**Shape (VB-derived).** Starting credit budget; real metered burn per model/tool call; a long horizon with a per-tick overhead fee; seeded job board; insolvency or turn-cap termination; score includes a money signal. VB is recovered exactly by setting breadth = 1.

**Composition.** Mostly *cheap synthetic verifiable gigs* (data-cleaning to schema, copy-to-constraints, extraction) — cheap, fast, RL-friendly, multi-type so it doesn't read as a coding eval — plus a *thin slice of real SWE-Lancer tasks* as the external-validity anchor (off the hot loop).

**Stages measured (Phase 1):** selection, pricing (flagship), delivery, support + manipulation-resistance, coherence.

**Key Phase-1 decisions locked from the deep read:**
- **Pricing is the flagship signal** — "helpfulness backfires economically" is the most consistent finding across VB2, Project Vend 1 & 2.
- **Manipulation-resistance is measured deterministically** — seeded adversarial requests; caving surfaces automatically in the ledger as pricing/support regret. No LLM judge.
- **A forced-procedure/reflection scaffold is a headline harness ablation** — Project Vend 2 found it the single highest-leverage change ("bureaucracy matters").

**Deliverable.** A runnable backend (`solvent run --agent base --seed 42`), a structured JSONL trace per run, a per-run scorecard, a cross-config scoreboard, and a clickable trace viewer for the demo.

**Success criteria.**
- Two harness configs produce a clear, reproducible scoreboard delta on identical seeds.
- The scorecard localizes a config's loss to a specific stage (e.g., "delivers fine, prices below cost").
- A seeded manipulation event is shown to reduce a naive config's score and not a hardened one — measured purely from the ledger.
- Sim-rank of ≥2 models matches their SWE-Lancer-anchor rank (external-validity check).

---

## 4. Phase 2 — Generalized Automaton eval environment

**Goal.** Widen Phase 1 from one vertical into an **Automaton-class business eval** with the **full attribution engine** and the axes the literature shows are central. This is where Solvent stops being "a freelance bench" and becomes "the measurement for the open company-building task."

### 4.1 What gets added

1. **The attribution engine — oracle substitution.** Re-run a seed with one stage replaced by an oracle; the lift in net revenue = the **marginal value of fixing that capability**. Converts the confounded scalar into a per-capability bottleneck ranking ("delivers and stays coherent fine, but pricing is the binding constraint; fixing it alone recovers $X"). This is the core research contribution and is impossible in VB2/Project Vend (single balance, no stages to ablate).
2. **More business models (breadth knob ≥ 2).** Add verticals chosen for the *new capability each forces*, not for cosmetic variety:
   - **Procurement/negotiation** (the axis freelance under-tests). VB2's whole upgrade was supply-side messiness — adversarial suppliers, bait-and-switch, delivery delays, suppliers going out of business, negotiation as the top differentiator. Add a deterministic supplier/subcontractor layer with seeded adversarial counterparties so negotiation and supply-robustness become measurable.
   - **Demand-finding-heavy** (e.g., a productized-offer/dropshipping-style selection problem) where the agent must *discover* which opportunities have reservation price > 0, scored as precision/recall against the planted demand set.
   - **Retention/relationship** (subscription-style) where repeat customers and churn add a relationship dynamic VB lacks.
3. **Manipulation-resistance as a first-class, expanded axis.** Generalize Phase 1's seeded adversarial requests into a deterministic red-team suite (discount-baiting, ex-post price cuts, refund extraction on passing work, scope creep, impersonation/authority claims à la the Project Vend "imposter CEO"). All outcomes resolve deterministically; susceptibility surfaces in the ledger.
4. **Multi-agent extension (optional, Arena-style).** Role-separated harnesses (cf. Project Vend's Clothius/CEO) and competition at a shared market (cf. Vending-Bench Arena: price wars, optional trade, individual scoring). Tests coordination and competitive pricing.
5. **Full reward decomposition + reference policies per stage**, generalized across verticals.

### 4.2 Deliverables

- A vertical-agnostic environment interface (`Market`, `Verifier`, `CustomerModel`, `SupplierModel`) with ≥3 verticals implemented.
- The oracle-substitution harness and a per-capability attribution report.
- The deterministic manipulation/red-team suite.
- A cross-vertical, cross-model leaderboard with attribution columns (not just a balance).

### 4.3 Success criteria

- For a fixed model, the attribution report correctly localizes its weakest capability, and oracle substitution confirms it (the predicted bottleneck yields the largest lift).
- Manipulation-resistance scores separate a helpful-by-default config from a hardened one, deterministically.
- Difficulty is calibrated per stage (no floor/ceiling) and headroom is preserved (frontier well below optimal).

---

## 5. Phase 3 — RL environment to improve the Automaton harness

**Goal.** Turn the eval into a **training environment** and demonstrate a measurable improvement to an Automaton-class harness's economic competence. Anthropic explicitly names this as the path forward (Project Vend 1: fine-tuning for business management via RL, rewarding sound decisions and discouraging selling at a loss).

### 5.1 Why the eval isn't yet an RL environment (and what must change)

An **eval** runs occasionally to measure; an **RL environment** must support a training loop. VB2 is the former and a poor candidate for the latter: each rollout is a simulated *year*, ~60–100M tokens — you cannot run RL over episodes that long or expensive. Phase 3 requires:

1. **Cheap, resettable, fast rollouts.** Synthetic gigs designed for speed; short configurable horizons; deterministic resets from seed. SWE-Lancer tasks (Docker, 20–60 min each) are explicitly excluded from the training loop and kept only as a periodic real-world anchor.
2. **Dense reward.** A single terminal balance after a long horizon is the worst possible RL signal (sparse, delayed, high-variance — which is exactly why VB shows wild run-to-run spread). **Phase 1/2's per-stage signals double as reward shaping**: a selection-regret penalty, a pricing-regret penalty, a delivery-pass reward, a manipulation-resistance penalty — signal every few steps. *The attribution layer is the reward function.* This is the load-bearing reason the earlier phases were designed the way they were.
3. **A policy/environment boundary.** The harness is the policy; Solvent is the world + reward. The clean boundary from Phase 1 (env sees only tool calls, emits observations + reward) is what makes this drop-in.

### 5.2 What "improve the harness" means here

Two complementary levers, both measurable in Solvent:
- **Scaffold/harness changes** (no training): e.g., the forced-procedure/reflection scaffold, a CRM-style memory tool, payment-before-delivery affordances. Project Vend 2 shows these move real profit; Solvent quantifies the lift per change.
- **Policy training** (RL/fine-tuning): reward-shaped training against the dense per-stage signals, targeting the specific capability the attribution report flags (e.g., pricing). Demonstrate a before/after on held-out seeds.

### 5.3 Sim-to-real calibration (the honesty mechanism)

Periodically evaluate the trained/scaffolded harness on the held-out SWE-Lancer anchor and confirm the synthetic improvement transfers (rank correlation, earn-rate lift). This is the VB ↔ Project Vend pairing closing the loop: train cheaply in sim, validate against reality.

### 5.4 Deliverables & success criteria

- A `gym`-style environment wrapper (reset/step/reward) with dense per-stage reward and sub-minute synthetic rollouts.
- At least one demonstrated improvement: a scaffold change *or* a reward-shaped fine-tune that raises fraction-of-optimal on held-out seeds, with the gain attributable to the targeted capability.
- Transfer check on the SWE-Lancer anchor.

### 5.5 Honest caveats (state these explicitly)

Phase 3 is the **north star, not a guaranteed portfolio result**. Full RL on LLM agents is hard and expensive. The defensible application claim is: (a) the environment is RL-shaped (dense reward, cheap rollouts, clean boundary), (b) a *proof-of-concept* training or scaffold improvement with a measured, attributable lift, and (c) a credible account of what full training would require. Do not claim a finished RL training win you have not produced.

---

## 6. Milestones

| Milestone | Phase | Output |
|---|---|---|
| M0 | 1 | Env skeleton: budget/burn, tick loop, one synthetic task type + verifier, stub agent, JSONL trace |
| M1 | 1 | Selection + pricing + delivery stages; hidden reservation price; per-stage scorecard; fraction-of-optimal; seeded market |
| M2 | 1 | Support + deterministic manipulation events; coherence metrics; 2-config scoreboard; trace viewer (demo-ready) |
| M3 | 1 | SWE-Lancer anchor (IC SWE delivery + Manager selection); external-validity check; characterization run protocol |
| M4 | 2 | Oracle-substitution attribution engine; per-capability report |
| M5 | 2 | Vertical-agnostic interfaces; procurement/negotiation vertical; expanded red-team suite |
| M6 | 2 | ≥3 verticals; cross-vertical leaderboard with attribution columns; (optional) multi-agent/Arena |
| M7 | 3 | gym-style wrapper; dense reward; sub-minute synthetic rollouts |
| M8 | 3 | One demonstrated, attributable harness improvement + SWE-Lancer transfer check |

**For the application:** M0–M3 (Phase 1) is the concrete deliverable. M4–M6 is the "generalized design" (specced, partially built). M7–M8 is the research direction.

---

## 7. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Reward hacking the verifier | Programmatic checks over LLM-judging; judge sees only spec + artifact, never the agent's pitch; track legitimate vs total revenue; red-team the verifier on a dev set |
| Sim measures sim, not reality | Thin SWE-Lancer real anchor; sim-rank vs real-rank correlation gate |
| Synthetic difficulty miscalibrated (floor/ceiling) | Characterization/control run before comparisons; per-stage difficulty dials; dev/test seed split |
| High variance hides signal | Distributions over seeds; paired-by-seed comparisons; report mean ± std |
| Freelance under-tests negotiation | Procurement/supplier layer in Phase 2; scope honestly in Phase 1 |
| "Just another business sim" perception | Lead with attribution + RL-usability; cite VB2/Project Vend as the unmet-need evidence |
| Phase 3 RL doesn't converge in time | Frame as north star; ship scaffold-change improvement as the achievable proof |

---

## 8. Mapping to the EICO AI Engineer role

| Role responsibility | Where it shows up |
|---|---|
| Agent backends and harnesses for company-building workflows | The swappable harness + tool API (Phase 1); multi-agent (Phase 2) |
| Model adaptation: prompting, memory, planning, post-training, RAG, tool use | Harness ablations (Phase 1); reward-shaped training (Phase 3) |
| Evals and feedback loops vs. real business outcomes | The entire environment + per-stage attribution (Phases 1–2) |
| Internal systems that generate traces we can train from and verify | JSONL traces + deterministic verifiers (Phase 1); dense-reward RL env (Phase 3) |

One-liner for the application: *Vending-Bench made a narrow business measurable; Automaton made the open task deployable but not measurable; SWE-Lancer measured single-task delivery. I built the measurement for the open task — decomposed, attributable, and RL-shaped — with freelance as the first vertical and a real-world anchor for validity.*

---

## Appendix A — Glossary

- **Reservation price** — the hidden threshold at or below which a (deterministic) customer accepts a bid; the environment's designed stand-in for willingness-to-pay (not a measurement of real demand). Ground truth held by the environment.
- **Selection regret** — value(optimal feasible job set) − value(agent's chosen set).
- **Pricing regret** — surplus left on accepted jobs + value of jobs lost to overpricing.
- **Fraction-of-optimal** — net revenue ÷ net revenue of the computed optimal reference policy.
- **Oracle substitution** — replacing one stage with a ground-truth-optimal oracle to measure that stage's marginal contribution.
- **Breadth knob** — the dial from "one fixed business (VB)" to "full open company-building loop."
- **Anchor** — a held-out real-world task slice (SWE-Lancer) used to validate that sim rankings track reality.