# Agent Training Plan: features, networks, search, and the bake-off

## Context

`Code/AGENT_INTERFACE_PLAN.md` adds the *seams*: observations, forking, determinization, option
encoding, the agent protocol, parallelism, the evaluation harness and the conformance suite. It
deliberately stops short of saying what the networks look like, how search actually works, how
training runs, or how the variants get compared.

This plan is that layer. It assumes the interface plan's milestones exist and names which ones each
step needs. It is written so that someone who has never seen this project can build it: every
component has a file, a shape, a formula and an acceptance test.

**Read first:** `Code/AGENT_INTERFACE_PLAN.md` (especially Milestones C, D, F, G, I, J and the
Options register). This document does not repeat anything decided there.

### Decisions (2026-09-23)

- **The comparison is the deliverable.** This is a master's project, and the ablation across search
  regimes, hidden-information handling and action representation is intended as the thesis
  evaluation chapter. Losing arms are results, not waste.
- **Maximum options, decide by measurement.** Same policy as the interface plan. Every arm below is
  reachable; which ones ship is decided by the diagnostics in M9. The Options register at the end
  extends the interface plan's.
- **Phase 1.1, Archon, Fignor vs Igor first.** The card vocabulary, the encoder and the harness all
  handle the full 370-card registry and all formats from day one, but every number in this plan is
  for the two-preset Archon matchup.
- **Two packages, one rule: engine workers never import `torch`.** See M0. This is what keeps the
  process-fork backend (interface plan E1) and the PyPy option available.
- **Search-free Deep Monte-Carlo is kept as a real arm**, not a fallback (M8). It is the only design
  here that needs no forking at all, and it is the cheapest per game by two orders of magnitude.
- **Every scale decision is made from measured throughput, never from this document's estimates.**
  The work is organized as budget **tiers** with numeric go/no-go gates (M10). Each tier is a
  complete, reportable experiment whose output is the next tier's input, so a run can be scaled
  down, paused or abandoned on evidence instead of hope. **Nothing below assumes the full budget is
  available.**
- **Phase 1.1 is the first pool, not the only one.** Every feature layout, vocabulary and checkpoint
  format is built so that the Phase 2 (159-card) and Phase 3 (370-card) pools, random legal decks,
  alliance decks and the Reversal/Adaptive formats are a *data* change rather than a rewrite. M11
  states the five invariants that have to hold and what breaks if they don't.

### Measured baseline (this machine, this engine)

Everything below was measured, not estimated. Commands are in `tools/bench_engine.py` (interface
plan Milestone 0).

| Quantity | Value |
|---|---|
| Hardware | 6 physical / 12 logical cores, 32 GB RAM, RTX 5060 Ti (8 GB) |
| Engine throughput, views on | ~39,000 decisions/sec |
| Raw `replay()` throughput, views off | ~49,000 decisions/sec (~20 µs per decision of prefix) |
| Replay fork at mid-game (75-decision prefix) | **~1.5 ms** |
| Process fork (interface plan E1, WSL only) | 0.69 ms flat, at any decision |
| `Game()` construction | 0.31 ms |
| `HeuristicBot` mirror game | **19.9 turns, ~152 decisions**, 7.5 decisions/turn |
| `RandomBot` mirror game | 55.3 turns, 223 decisions |
| Card vocabulary | **49 distinct cards** across Fignor+Igor; 370 in `CARD_DEFS` |
| Deck composition | 36 cards/player; Fignor 33 distinct, Igor 30 distinct; both Dis/Logos/Shadows |

**Decision mix under competent play** (200 `HeuristicBot` mirror games, 30,508 decisions). This
differs materially from the random-play profile in the interface plan and supersedes it for all
design purposes — a trained agent plays much more like this than like `RandomBot`:

| Decision kind | % of all (competent) | % of all (random) | worst choice space |
|---|---|---|---|
| `CHOOSE_ACTION` | **59.0%** | 64.1% | 20 |
| `CHOOSE_CARDS` | **13.0%** | 5.1% | **254** |
| `CHOOSE_HOUSE` | 11.1% | 22.4% | 3 |
| `CHOOSE_FLANK` | **11.0%** | 3.9% | 2 |
| `CHOOSE_HOUSE_FOR_EFFECT` | 2.3% | 1.1% | 3 |
| `TAKE_ARCHIVE` | 2.0% | 2.4% | 2 |
| `MULLIGAN` | 1.3% | 0.9% | 2 |
| `ORDER_EFFECTS` | 0.2% | 0.2% | **6** (n = 3) |
| Forced (one legal outcome) | **7.0%** | 11.7% | — |

**Three consequences, which drive this plan:**

1. **`CHOOSE_CARDS` is 2.5× more common than the random-play profile suggested** — 13% of decisions,
   not 5%. The multi-select treatment is therefore a first-class axis, not a footnote, and it is
   worth resolving carefully in M3.
2. **`ORDER_EFFECTS` reaches n = 3** under competent play, not n = 2. Still trivially enumerable
   (6 permutations), but the sequential fallback must exist before Phase 2/3 pools.
3. **Competent games are ~152 decisions**, so every throughput estimate below uses 152, not 223.

### The fixed entity set: 72 tokens, always

Every card in the game is in exactly one zone at all times, and both decklists are public
(`PlayerView.decklist`, "always visible"). There are 36 cards per player. So **an observation always
contains exactly 72 card entities** — no padding, no variable-length batching, no masking of absent
entities. This is unusual and it simplifies the encoder considerably. What varies is not *which*
entities exist but *what is known about where they are*, which is a per-entity feature.

### Scope boundary

**Out of scope:** the generator-to-explicit-stack engine refactor (interface plan scope boundary
stands), and any Phase 2/3 card work.

**In scope as options:** every arm in M5, M6 and M8.

---

## Milestone M0: layout, dependencies, environments

**Two packages, split by whether they import `torch`.**

```
Code/Non-GUI/
  agent/                    # pure stdlib. Imported by engine workers. NEVER imports torch.
    __init__.py
    spec.py                 # feature layout constants + FEATURE_VERSION
    features.py             # Observation -> flat float arrays (array('f'), no numpy)
    search/
      core.py               # shared MCTS: tree, PUCT, backup, batching, budgets
      within_turn.py        # regime A
      full_game.py          # regime B
      leaf.py               # leaf value estimators (student / belief+oracle / heuristic)
      policies.py           # default + rollout policies
    agents/
      net_agent.py          # search-free: BC net, DMC net
      search_agent.py       # search agent, parameterized by regime + leaf estimator
      registry_entries.py   # registration into bots/registry.py (interface plan I)
  ml/                       # torch only. CPython, WSL, GPU. Never imported by engine workers.
    model.py                # trunk + heads
    encode.py               # flat arrays -> batched tensors
    bc_train.py             # M3 screens
    selfplay_train.py       # M7 learner
    belief.py, oracle.py    # M6 auxiliary nets
    checkpoints.py          # content-hashed checkpoint registry
    infer_server.py         # implements InferenceClient (interface plan H)
  tools/
    run_bakeoff.py          # M9 orchestration
```

**Why the split:** `os.fork()` does not survive CUDA initialization, and PyPy cannot run PyTorch.
Keeping `agent/` torch-free means engine workers can fork freely and can later run under PyPy while
the network stays in a CPython inference server. A test asserts `agent` imports cleanly with `torch`
absent from `sys.modules`, and that no module under `agent/` imports it.

**Environments** (interface plan Milestone 0): WSL2 Ubuntu with CPython 3.12 and a CUDA 12.8+
PyTorch wheel (the 5060 Ti is Blackwell; older wheels lack the kernels). Windows keeps a CPU-only
PyTorch for GUI play. All pipeline output goes under `$KEYFORGE_DATA` on the Linux filesystem.

**Dependencies:** `torch` (CUDA build, WSL) / `torch` (CPU, Windows), plus nothing else mandatory.
No RL framework, no Lightning — the loops here are short and explicit.

### Configuration and provenance

Everything that can be varied lives in **one config file per experiment** (`configs/*.yaml`), never
in code or command-line flags scattered across scripts. A run resolves its config, hashes the
resolved form, and writes that hash into every artifact it produces — checkpoints, trajectory
shards, metric records and evaluation rows. Two runs are comparable if and only if their config
hashes differ in ways you can enumerate.

The config carries: `FEATURE_VERSION`, vocabulary version, `ENGINE_VERSION`, network sizes, search
settings, training hyperparameters, the tier (M10), and the deck/format/pool selection.

**Run journal.** Each run owns `$KEYFORGE_DATA/runs/<run_id>/journal.jsonl`: an append-only record
of every human intervention — a learning-rate change, a tier downgrade, a restart, a killed actor.
Mid-run changes are *expected* under a variable budget; what makes them survivable is that they are
recorded with the game index at which they happened, so a learning curve can always be read against
what was actually done to it.

**Artifact naming:** `<run_id>/<kind>/<content_hash>`, never a path that encodes a hyperparameter —
hyperparameters change, and a renamed artifact is an unfindable artifact.

### What invalidates what

| Change | Invalidates | Recovery |
|---|---|---|
| `ENGINE_VERSION` (any rules or card fix) | all replay records, all trajectories | regenerate data; checkpoints survive |
| `FEATURE_VERSION` (non-reserved feature layout) | all checkpoints, all encoded tensors | retrain, or re-run behaviour cloning |
| Reserved-slot feature added (M11) | nothing | old checkpoints load with the new dims zeroed |
| Vocabulary append (new cards) | nothing | new rows attribute-initialized (M11) |
| Network size change | that checkpoint lineage only | warm-start from behaviour cloning |
| Search or training hyperparameters | nothing | recorded in the config hash; runs stay comparable |

The critical line is the second row against the third: **adding a feature in a reserved slot is
free, consuming a new slot costs every checkpoint you have.** M11 explains how to stay on the free
side of it.

**Acceptance:** `torch.cuda.is_available()` is true in WSL; `python -c "import agent"` succeeds in an
environment with no torch installed; `tools/bench_engine.py` reproduces the table above; every
artifact written by a smoke run carries a resolved-config hash, and re-resolving the config
reproduces that hash byte-for-byte.

---

## Milestone M1: the tensor specification

**Depends on:** interface plan C (observation layer, leak fix), D (card-location knowledge), F
(`option_key` / `option_features`, versioned card vocabulary).

`agent/spec.py` holds every constant and a `FEATURE_VERSION` integer stamped into checkpoints and
training shards. Changing any layout bumps it; loaders refuse a mismatch.

`agent/features.py` exposes one function:

```
encode(obs, decision) -> (entities: array('f')[72 * E],
                          globals: array('f')[G],
                          options: array('f')[n_options * O],
                          pointers: array('i')[n_options])
```

No numpy, so it stays fast under PyPy. `ml/encode.py` turns batches of these into tensors.

**Everything is encoded from the deciding player's perspective** ("mine" / "theirs", never "player
1" / "player 2"), so the network is seat-agnostic and both seats contribute to the same training
signal.

### Per-entity features (72 rows, E ≈ 96 floats + 1 vocab index)

*Identity and static attributes, from `CardDef`:*
- `card_id` — index into the versioned, append-only vocabulary (370 entries today), fed to an
  embedding table rather than one-hot.
- house (7), card type (4), base power, base armor, æmber bonus, keyword multi-hot (elusive,
  skirmish, taunt, poison, assault, hazardous, and the rest of the CotA set), assault/hazardous
  magnitudes, trait multi-hot hashed into 32 buckets.

*Location, with the information set respected:*
- zone one-hot over 16 codes: `my_hand`, `my_deck_unordered`, `my_discard`, `my_archive`,
  `my_purged`, `my_creature`, `my_artifact`, `my_upgrade_attached`, and the eight mirror codes for
  the opponent, plus `opp_unseen`.
- **The rule that prevents leaks:** a card is emitted in its true zone only if the viewer is entitled
  to know it is there. Otherwise it goes in `opp_unseen` (somewhere in their hand or deck) or
  `my_deck_unordered` (in my deck, position unknown). Because both decklists are public, the *set*
  of unseen cards is known exactly — only the split is hidden. Entitlement comes from the
  card-location knowledge tracked in interface plan D, not from a local guess.
- `known_to_me`, `known_to_them`, `owner_is_me`, `controller_is_me`.

*Mutable in-play state:*
- damage, computed power, computed armor, armor used this turn, captured æmber, stunned, ready /
  exhausted, flank one-hot (left / right / both / neither), normalized battleline index, number of
  attached upgrades, number of cards underneath, damage-prevented flag, `fought_this_turn`.

*Decision-relative flags:*
- `is_legal_option` — this entity is referenced by an option in the current decision.
- `is_source_card` — this entity asked the current decision (from `Decision.source_card`, interface
  plan B).

### Global features (G ≈ 140 floats)

æmber, keys, chains, hand count, deck count, discard count, archive count and purged count for both
sides; the **current computed key cost** (dynamic — Iron Obelisk, Murmook); normalized turn number;
active house (8, including none); my deck's three houses and theirs (7 each); this-turn counters
(cards played, creatures destroyed in fight, keys forged — the counters from interface plan L.6);
an aggregate summary of active duration effects, as a count per `(variable, op)` pair; and the match
context (format one-hot, game index within the match, match score, starting chains, deck swap flag).

Plus the current decision: kind one-hot (13), `DecisionIntent` one-hot (~18, interface plan B),
`affects` one-hot (4), `optional`, `min_n`, `max_n`, `n_options`.

### Per-option features (O ≈ 48 floats + 1 pointer)

- verb one-hot (~14): `PlayCard`, `DiscardCard`, `UseAction`, `UseOmni`, `Reap`, `Fight`, `EndTurn`,
  plus the non-action option families: select-card, select-house, select-flank, boolean, number,
  mode, player-id.
- `pointer` — the index (0–71) of the entity this option refers to, or −1. This is what makes the
  policy head a pointer network: the option's meaning comes from the entity embedding it points at.
- scalar payload: number value (normalized), house id, boolean value, flank id, mode id.

### Reserved space — the thing that makes later pools cheap

Each feature block ends with explicitly **reserved, always-zero slots**: 16 per entity, 24 global,
8 per option, plus 16 spare keyword bits and 8 spare `DecisionIntent` slots. They exist so that
Phase 2/3 state — a new keyword, a new per-turn counter, a new intent, a new zone — can be added
**without shifting a single existing index**, which is the difference between "old checkpoints keep
loading" and "retrain everything" (see the table in M0).

The rule, enforced by a test: adding a feature into a reserved slot bumps a *minor* version and
leaves every checkpoint loadable with the new dimension zero-initialized. Adding one anywhere else
bumps `FEATURE_VERSION` and invalidates all of them. `spec.py` states, per block, how many reserved
slots remain; when a block reaches zero, that is a deliberate decision point, not a surprise.

**Vocabulary growth.** The card vocabulary is versioned and append-only (interface plan F). It is
sized for the whole 370-card registry from day one even though Phase 1.1 uses 49, so moving to a
larger pool adds *rows that already exist* rather than resizing anything. No index is ever reused or
reordered, including when a card is errata'd — an errata changes attributes, not identity.

**Acceptance.**
1. **No leak:** over 1,000 seeded games, at every decision, reconstructing the encoded observation
   never places a card in a zone the viewer is not entitled to know (the interface plan C acceptance
   test, extended to the encoder).
2. **Determinism:** the same `(obs, decision)` encodes byte-identically across processes and
   platforms.
3. **Throughput:** encoding costs < 25% of the engine's per-decision cost, measured by
   `tools/bench_engine.py`. If it does not, apply the incremental-encoding item from interface plan
   L.
4. **Coverage:** every `DecisionKind` in the registry encodes without an unhandled branch; an
   unknown option type is a hard error, never a zero vector. This includes the five kinds that never
   fire in the Phase 1.1 pool (`YES_NO`, `CHOOSE_NUMBER`, `CHOOSE_MODE`, `BID_CHAINS`,
   `CHOOSE_FIRST_PLAYER`) — they are encoded and tested from day one so that Phase 2/3 and the
   match formats need no encoder change.
5. **Pool-independence:** the same encoder runs unchanged over Phase 2 and Phase 3 decks and over
   random legal and alliance decks, producing the same shapes. Test it now, while those pools are
   not the target — it is the cheapest possible time to find out the layout is wrong.

---

## Milestone M2: the network

`ml/model.py`. One trunk, four heads. Starting sizes are given; they are hyperparameters, and M3
tunes them.

**Trunk.**
- Entity tower: `card_id` → `Embedding(370, 64)`, concatenated with the rest of the entity features,
  through `Linear(64 + E, 128)` + LayerNorm.
- Global token: global features through `Linear(G, 128)`, prepended as token 0.
- **73 tokens** (1 global + 72 entities) through 4 pre-LN `TransformerEncoderLayer`s, `d_model=128`,
  `nhead=4`, `dim_feedforward=256`, dropout 0.1 during behaviour cloning and 0.0 during self-play.
- No positional encoding: the entity set is unordered, and everything positional (flank, battleline
  index, zone) is already an explicit feature.
- Output: `g` = token 0's embedding; `h[i]` = entity *i*'s embedding.

**Policy head (pointer scoring).** For each legal option *o*:

```
e_o     = MLP([verb_embedding(32) ; h[pointer(o)] (or a learned null vector) ; scalar feats]) -> 128
logit_o = <W_q g, e_o> / sqrt(128)
p       = softmax over the option list only
```

There is no fixed action vocabulary and no masking step — the head only ever scores options the
engine actually offered, so illegal moves are unrepresentable rather than suppressed. A
fixed-vocabulary head remains available through `option_key` (interface plan F) as an M3 ablation.

**Value head.** `MLP(128 -> 128 -> 1)`, `tanh`, giving *v* ∈ [−1, 1] from the deciding player's
perspective. Trained against `Game.outcome_for(pid)` (+1 / 0 / −1, with the `max_turns` draw mapped
explicitly).

**Auxiliary heads** (M6; trained from the privileged label file, never from an ordinary
observation):
- **Oracle value:** `MLP([g ; hidden-state summary] -> 128 -> 1)`, `tanh`. The hidden-state summary
  is a count vector over the vocabulary for the opponent's actual hand, plus my actual next *k* = 5
  draws.
- **Belief:** for each of the opponent's unseen entities, `MLP(h[i] -> 1)` → logit that the card is
  in their **hand** rather than their deck. Sampling under the hand-count constraint is in M6.

**Size and speed.** ~0.7 M parameters. Target: ≥ 20,000 single-position evaluations/sec at batch 512
in fp16 on the 5060 Ti. The engine, not the GPU, must remain the bottleneck; if it is not, the
network is too large for this stage.

**Capacity ladder.** Size is a config value, re-decided per pool by the M3 capacity ablation rather
than guessed. Starting points:

| Pool | Cards | `d_model` / layers | Approx. params |
|---|---|---|---|
| Phase 1.1 (Fignor/Igor) | 49 | 128 / 4 | ~0.7 M |
| Phase 2 presets + random | 159 | 192 / 6 | ~2.4 M |
| Phase 3, random legal decks | 370 | 256 / 8 | ~6 M |

Growing the trunk means a new lineage, not a resumed one — but behaviour cloning makes a fresh start
cheap (hours, M3), so this is a scheduled cost rather than a setback. **Growing only the embedding
table does not**: see below.

**Embedding growth without retraining.** When the vocabulary extends past the cards a checkpoint was
trained on, the new rows are initialized from the attribute tower's output for that card, not
randomly, and existing rows and their optimizer state are preserved. A Phase 1.1 checkpoint
therefore loads against a Phase 2 vocabulary and plays legal, non-random KeyForge immediately,
before any fine-tuning. This is also why every entity carries full attribute features alongside its
identity embedding: **the attribute path is what carries a card the network has never seen**, and
M3's Screen 4 measures how much of the work it actually does.

**Acceptance.** Shape tests for every `DecisionKind`; the policy sums to 1 over exactly the offered
options; the throughput target above; a checkpoint saves and reloads across Windows and WSL with
matching digests; a checkpoint records `FEATURE_VERSION`, the vocabulary version and
`ENGINE_VERSION`, and refuses to load against a mismatch.

---

## Milestone M3: the cheap screens — behaviour cloning and value regression

**Depends on:** interface plan 0, A, B, C, F, K. **Needs no forking, no search, no RL.** This is the
first milestone that produces a usable agent, and it decides two of the four axes.

### Data

`sim/generate.py` (interface plan J) writes trajectories. For the screens, generate:

- 20,000 `HeuristicBot` mirror games — the imitation target;
- 5,000 games with an ε-greedy `HeuristicBot` (ε = 0.1) — state coverage the pure heuristic never
  visits;
- 5,000 `RandomBot` games — the near-random states early self-play will actually occupy.

At ~152 decisions/game that is ~4.5 M labelled decisions, generated in roughly three minutes of
wall-clock across 5 workers. **Split by game, never by decision** (90/5/5): consecutive decisions
within a game are heavily correlated, and a decision-level split leaks the answer across the split.

Store replay records, not tensors (interface plan J's storage trade-off): regeneration costs ~75
seconds per 10,000 games and stays correct when the encoder changes.

### Screen 1: policy imitation

Cross-entropy against the heuristic's chosen option. AdamW, lr 1e-3 with cosine decay to 1e-5,
weight decay 1e-4, batch 512, 4 epochs, gradient clipping at 1.0.

Report **top-1 accuracy per `DecisionKind`**, not just overall — the overall number is dominated by
`CHOOSE_ACTION` and hides everything else.

### Screen 2: value regression

Same trunk, same corpus, label each position with the eventual outcome. Report held-out log-loss and
**accuracy as a function of turn number** — this says how early KeyForge is decided, which is a
thesis result in its own right and also tells you whether the encoding carries the signal at all.

### Screen 3: the multi-select bake-off (decides the axis)

Train all three treatments as parallel heads on one shared trunk, on the same targets:

1. **Enumerate** — a categorical over enumerated subsets/permutations. Worst observed space under
   competent play is **254**, `ORDER_EFFECTS` reaches 6. Subset embedding = attention-pooled member
   embeddings.
2. **Sequential** — chained single picks, each restricted to option indices greater than the last
   (which collapses the *k*! orderings reaching the same set into one path), plus an explicit stop
   action covering `min_n = 0`.
3. **Independent top-k** — per-option sigmoid, take the *k* highest.

Metric: **exact-set-match accuracy** on `CHOOSE_CARDS` and exact-permutation match on
`ORDER_EFFECTS`, on held-out games. A second, sharper metric arrives in M9 (agreement with a
high-budget reference search on a fixed position suite).

This axis is settled here and then held fixed, because at 13% of decisions its effect on win rate is
on the order of a point, and resolving one point by game outcomes needs ~10,000 games per pairing.
Per-decision metrics are far more sensitive per unit of compute.

### Screen 4: encoder ablations

Same protocol, one factor at a time: pointer head vs fixed vocabulary; card-ID embedding vs
attributes only vs both; 2 / 4 / 6 trunk layers; with and without `DecisionIntent` (this one
quantifies what interface plan B bought — expect a large drop on `CHOOSE_CARDS` without it).

### Screen 5: generalization to unseen cards and decks (run once, early)

Cheap, and it decides how much of M11 is realistic. Using `--random-decks` over the full 370-card
registry, generate a modest heuristic corpus, then train on decks drawn from a restricted card set
and measure imitation accuracy on decks containing cards **held out of training entirely**.

Report accuracy on seen-card decisions vs. unseen-card decisions, for each of the three identity
encodings. If attributes-only and both-paths degrade gracefully while ID-only collapses, the
extension story in M11 holds. If all three collapse, the attribute features are too weak and need
enriching *now*, while fixing it costs a morning rather than a retrain.

This screen runs at Tier 0 and is worth its cost even if Phase 2/3 never happens, because it also
says how much the network is memorizing 49 specific cards rather than learning KeyForge.

### Acceptance

- ≥ **80%** top-1 on `CHOOSE_ACTION` and ≥ 90% on the binary kinds.
- The BC agent, played with no search, scores **≥ 40%** against `HeuristicBot` over 2,000 paired
  seeds. It is imitating, so it should approach parity, not exceed it; well below 40% means the
  encoding or the head is losing information.
- Value head beats a constant predictor by a clear margin from turn 5 onward.
- A written decision, with numbers, for the multi-select treatment and each ablation.
- The BC checkpoint is registered and becomes the **warm start** for every run in M7.

---

## Milestone M4: the shared search core

**Depends on:** interface plan D (fork / `fork_determinized` / `fork_many` / `run_until`), F (option
keys), G (agent protocol, budgets, hooks, capabilities), plus H for batching.

`agent/search/core.py`. One implementation serving both regimes.

**Tree representation.** A node is keyed by `observation_key(obs)` (interface plan C) — an
*information set*, not a state. Per the interface plan's tree-node guidance, a node stores the
**path of choice indices from the root, never a live `Game`**; state is regenerated by
`fork` + `apply`. Per edge: `N` (visits), `W` (total value), `P` (prior), and `A` (**availability
count** — the number of iterations in which this option was legal).

**Selection (PUCT, adapted for information sets).**

```
Q(s,a) = W(s,a) / N(s,a)          (0 when N = 0)
U(s,a) = c_puct * P(s,a) * sqrt(A(s,a)) / (1 + N(s,a))
select argmax over options legal in THIS iteration's determinization
```

The availability count in place of the usual `sqrt(sum of N)` is the ISMCTS correction (Cowling,
Powley & Whitehouse, 2012): different determinizations make different options legal, and without it
options that happen to be legal rarely are systematically over-explored. **This is the single
easiest thing to get wrong in the whole implementation.**

**One iteration.**
1. Determinize: `fork_determinized(viewer, rng, resample=...)` at the root. The `resample` mode is
   the regime's (M5).
2. Descend by PUCT, replaying choice indices into the fork, until an unexpanded node.
3. Expand: evaluate the leaf (M6), set priors `P` from the policy head, initialize `N = W = 0`.
4. Back up along the path, negating value at every change of acting player.
5. Increment `A` for every option legal in this iteration at every node visited.

**Root exploration.** Dirichlet noise on root priors: `P = (1 − ε)·P + ε·Dir(α)`, with ε = 0.25 and
**α = 0.8** (the AlphaZero rule of thumb `α ≈ 10 / mean branching`; mean branching here is ~13–20).

**Move selection.** Sample from visit counts with temperature τ = 1.0 for the first 15 decisions of a
game, then τ = 0.1. Evaluation always uses τ = 0.

**Batching.** Leaf evaluations go through `InferenceClient` (interface plan H). Virtual loss of 1
visit is applied on descent and removed on backup so that a worker can keep 8–16 leaves in flight
without collapsing onto one path.

**Subtree reuse.** On `observe(event)` (interface plan G hooks), advance the root to the child
matching the edge actually taken and discard siblings. Availability counts carry over; visit counts
carry over.

**Caches.** An LRU network-evaluation cache and a transposition table, both keyed by
`observation_key`. Determinized search revisits the same information sets constantly, so hit rates
should be high; report them.

**Budgets.** `decide` receives a `Budget` (simulations and/or wall-clock) from the harness, so
compute-normalized comparison is a harness setting rather than a per-agent flag.

**No-network debug mode.** The core must run against the heuristic value function plus a uniform
prior (interface plan J's no-network baseline), so that availability counts, determinization,
backup signs and subtree reuse can all be debugged before a network exists. **Build this first.**

**Acceptance.**
- On a set of constructed positions (`setup_script`, interface plan D) with a known best move,
  search with the heuristic evaluator finds it at 200 simulations.
- Backup signs verified: a forced-loss position returns *v* → −1 and a forced win → +1.
- Availability counts: in a position where an option is legal in exactly half of determinizations,
  its `A` is within sampling error of half the iterations.
- Subtree reuse is exactly equivalent to a fresh search at the same total simulation count, modulo
  RNG (asserted on seeded positions).
- The conformance suite (interface plan K) passes for the search agent at every privilege level.

---

## Milestone M5: the two search regimes

Both are `agent/search/*.py` on the M4 core, selected by a constructor argument. **Note the horizon
arithmetic:** a turn is ~7.5 decisions with branching up to 20, so at 100 simulations neither regime
sees far past the current turn except along its principal variation. The regimes therefore differ
mainly in *where the value head is asked to evaluate*, which is exactly what the bake-off measures.

### A: within-turn search (`within_turn.py`)

- The tree spans only decisions belonging to the acting player within the current turn.
- **Leaf condition:** end of turn, reached via `run_until(end_of_turn, policy=rollout_policy)`. The
  rollout policy is an option: network-greedy (accurate, costs evaluations), `HeuristicBot` (free by
  comparison — ~30,000 decisions/sec, and a turn completes in ~4 decisions), or random (baseline).
  Default: `HeuristicBot`.
- **Leaf value:** the value estimator (M6) applied to the **end-of-turn** observation — a quiet,
  consistent evaluation point.
- **Determinization:** `resample=OWN_DECK` only. Within your own turn the opponent's hand is nearly
  irrelevant; what is stochastic is your own draws. This makes regime A a *stochastic single-agent
  search*, which is a far easier object than full imperfect-information search.
- Opponent decisions arising mid-resolution are resolved by a fixed policy without branching
  (option: branch on them).

### B: full-game ISMCTS (`full_game.py`)

- Standard PUCT over all decisions regardless of owner, negating value at each change of acting
  player; the tree crosses turn boundaries.
- **Leaf value:** the estimator applied wherever the leaf falls, mid-turn included.
- **Determinization:** `resample=OPPONENT_PRIVATE` (default) or `ALL` — the search must invent a
  concrete opponent hand in order to let the opponent act at all.

**Acceptance.** Both regimes pass the conformance suite; both beat `RandomBot` ≥ 95% and
`HeuristicBot` ≥ 55% using only the heuristic evaluator at 200 simulations, before any trained
network exists. If a regime cannot beat the heuristic with the heuristic's own evaluation function
plus search, the search is broken, not the network.

---

## Milestone M6: leaf value estimators and the hidden-information axis

`agent/search/leaf.py`. **These two are not alternatives in the same slot** — one trains the value
function, the other decides which worlds get sampled — but at the leaf they are two estimators of
the same quantity, `E[V | what I can see]`, which is what makes the comparison well posed.

**S (student, default):** one forward pass of the value head on the observation. Cost: 1 evaluation.
Note that regressing a student onto oracle values over self-play data already learns the posterior
expectation under the self-play distribution, so distillation performs belief inference implicitly,
limited only by what the observation encoding contains.

**B (belief-sampled oracle):** sample *K* = 8 hands from the belief head, evaluate the oracle value
head on each, average. Cost: *K* evaluations, but **no extra engine forks** — the oracle head takes
the hand as input features, so sampling is pure network work and batches on the GPU.

**Belief sampling under constraints.** The belief head gives per-card marginals; a sample must
respect the opponent's exact hand count. Use sequential sampling without replacement: draw cards one
at a time in proportion to remaining marginals until `hand_count` is reached, renormalizing after
each draw. Validate against the constraint that every sampled world satisfies
`check_invariants` when materialized through `fork_determinized`.

**Oracle-guided value training (Suphx-style), available in both:** train the oracle head first —
its targets are far less noisy — then either anneal the oracle features out with decaying dropout or
distil into the student. This mostly buys training speed rather than final strength.

**Training data:** the privileged, separately-flagged label file from interface plan J. The true
opponent hand and deck order are free labels in self-play. They must never reach an ordinary agent's
input; the capability system (interface plan G) enforces this and the conformance suite tests it.

**Interaction with the regimes, stated explicitly** because it shapes M9's matrix:
- Under **regime A**, opponent-hand sampling is nearly inert, so S vs B is a pure leaf-estimator
  comparison.
- Under **regime B**, world sampling is *mandatory* regardless — so "B with estimator S" means
  uniform determinization plus a distilled value, and the belief net's second job there is to make
  the sampling itself non-uniform.

**Acceptance.** Belief head calibration: reliability diagram over held-out games, plus log-loss
against a uniform-over-consistent-worlds baseline — it must beat it decisively or the belief arm is
dead on arrival. Oracle head beats the student on held-out value log-loss (it sees strictly more, so
if it does not, something is wrong). Sampled worlds always satisfy the count constraints.

---

## Milestone M7: the self-play training loop

`ml/selfplay_train.py` (learner) plus the self-play actors from interface plan H.

**Topology.** 5 engine worker processes × K concurrent games each (K tuned so the CPU keeps working
while leaves wait on the GPU; rule of thumb K ≥ decision rate × batch round-trip latency), one
inference server owning the GPU, one learner. Workers write sharded append-only trajectory files;
the learner reads them; actors reload checkpoints at game boundaries only.

**Targets.** Policy target π = normalized visit counts at the root. Value target *z* = the game
outcome from that player's perspective. Auxiliary targets from the privileged file.

**Loss.**

```
L = CE(pi, p) + 1.0 * MSE(z, v) + 0.25 * BCE(belief) + 0.25 * MSE(oracle) + 1e-4 * ||theta||^2
```

**Optimizer.** AdamW, lr 2e-3 with cosine decay, batch 1024, gradient clip 1.0, roughly one gradient
step per 8–16 new positions. Replay buffer = the most recent ~1 M positions (~6,500 games), sampled
uniformly.

**Search settings during self-play.** 100 simulations, with **playout cap randomization** (KataGo):
75% of decisions get 25 simulations and are *not* recorded as policy targets, 25% get the full 100
and are recorded. This is the largest single efficiency lever in the AlphaZero literature and it
cuts mean simulations per decision from 100 to ~44.

**Warm start.** Initialize from the M3 behaviour-cloning checkpoint. This skips the near-random
phase outright: heuristic-level games are 19.9 turns / 152 decisions against 55.3 / 223 at random
level, so it is ~1.5× fewer decisions per game from the first game played.

**Mirror augmentation (free 2×).** Reverse both battlelines and swap `CHOOSE_FLANK` labels. The
interface plan verified this is exact for Fignor/Igor (no card in the pool names left or right, and
the four engine sites that force a left-flank placement are absent from both decks). Re-check per
pool.

**Checkpoint gating.** Every 1,000 gradient steps, run a 400-game SPRT against the current best
(paired seeds, both seat orders, both deck assignments). Promote only on a pass. Keep every promoted
checkpoint — they are the frozen anchors for M9 and the rating ladder.

**Other settings:** training-only `max_turns` cap; resignation stays **off** until the value head is
calibrated, then enabled with 10% of games exempt to measure the false-resignation rate.

### Running it under a variable budget

A run must survive being interrupted, resumed on a different day, and rescoped halfway through.
Three properties make that true, and each is a build requirement rather than a hope:

- **Every run is checkpoint-resumable, not restart-only.** Actor shards are append-only and
  worker-exclusive; the learner checkpoints optimizer state alongside weights; resuming re-reads the
  buffer from shards. Killing the run at any moment costs at most the in-flight games.
- **A run is defined by a game budget, not a wall-clock target.** `--games N` with a running count,
  so "how far through am I" is always answerable and a tier change is a budget edit, not a restart.
- **Hyperparameters may change mid-run, and the change is recorded** in the run journal (M0) with
  the game index. A learning curve annotated with its interventions is honest and still reportable;
  an unannotated one is neither.

**Every promoted checkpoint is a bankable result.** If the budget disappears at game 18,000, the
last promoted checkpoint, its rating against the frozen anchors, and the learning curve to that
point are a complete, reportable outcome — a worse agent than the plan intended, not a failed
experiment. Design the reporting so that this is true from the first promotion onward, rather than
assembling it at the end.

**Acceptance.** The loop is resumable after a kill at any point with no duplicated or lost shards;
a run reproduces from `(run seed, game index, seat)`; the promoted checkpoint beats the BC warm start
by a statistically significant margin within the first 20,000 games — if it does not, stop and debug
rather than spending days of GPU time.

---

## Milestone M8: search-free Deep Monte-Carlo (the third arm)

`agent/agents/net_agent.py` + `ml/selfplay_train.py --mode dmc`. Kept because it needs **no forking
at all** and costs ~1 network evaluation per decision against ~4,400 for a searching agent — about
two orders of magnitude cheaper per game.

Following DouZero: no policy head and no tree. A Q-network takes the **action as an input** (the
same option embedding from M2) and is trained by regressing Q(s, a) toward the **Monte-Carlo return**
of the game in which that action was taken. Action selection is ε-greedy over Q, ε decaying 0.3 → 0.01.
The trunk, encoder and option features are all shared with the other arms, so this arm costs one
training script, not a second stack.

**Why it is here:** DouZero reached state of the art on DouDizhu — a larger action space and more
hidden information than this — with no search whatsoever. Its weakness is exactly KeyForge's
tactical within-turn sequencing, which is what makes it an informative contrast rather than a
duplicate.

**Acceptance.** Beats `HeuristicBot` over 2,000 paired seeds within a compute budget comparable to
one M7 run, or is reported as a negative result with its learning curve.

---

## Milestone M9: the bake-off

**The structure that makes 12 cells affordable: only two of them need a training run.** The search
regime shapes the policy targets, so it must be trained separately. Everything else is either an
auxiliary head trained on the same data, or an inference-time swap.

### What is decided where

| Axis | Options | Decided by | Cost |
|---|---|---|---|
| Multi-select | enumerate / sequential / top-k | M3 screens + position-suite agreement | no training run |
| Policy head | pointer (default) / fixed vocabulary | M3 Screen 4 | no training run |
| Hidden-info leaf | student / belief+oracle | inference-time swap | no training run |
| Determinization | own-deck / opponent-private / all | inference-time swap + diagnostic | no training run |
| **Search regime** | **within-turn / full-game** | **separate training runs** | **2 arms** |

### The training arms

Two arms × **2 seeds each = 4 runs**. Two seeds per arm is the minimum that distinguishes an arm
effect from run-to-run variance; with one run per arm you cannot, and that is the single biggest
statistical weakness available to this design. If the budget only allows one seed per arm, say so
explicitly as a limitation and lean on learning curves at matched checkpoints rather than endpoints.

### The evaluation matrix

From those runs, evaluate **train regime (2) × play regime (2) × leaf estimator (2) = 8 cells**, plus
anchors: `RandomBot`, `HeuristicBot`, the BC net with no search, the DMC agent (M8), and each
promoted checkpoint.

The **mismatched cells** (trained with regime A, played with regime B) are the interesting ones: they
separate "which search plays better" from "which search trains a better network", which a plain
two-arm comparison cannot do.

### Protocol (all of it non-negotiable)

- Both seat orders × both deck assignments, **paired seeds**, duplicate evaluation (the same seed
  played twice with agents swapped). The 57.5% first-player win rate confounds seat with deck and
  must be controlled every time.
- Compute normalization: report strength against **simulations per decision and wall-clock**, not a
  single number. A 100-simulation searcher against a 1-evaluation DMC agent is not a fair single
  comparison, it is two points on a curve.
- SPRT for development comparisons; fixed N for anything reported. Standard error ≈ `0.5/√N`: ~1,000
  games for a 3-point gap, ~10,000 for 1 point.
- Bradley-Terry ratings with confidence intervals and an explicit non-transitivity report
  (`sim/rate.py`, interface plan I). A > B > C > A is common in card games and finding it is a
  result: it means strength here is not one-dimensional.

### The two diagnostics that outrank the matrix

Run these **before** committing to the full matrix; they can cancel whole arms.

- **`tools/diag_hidden_info.py`** — the same agent with and without privileged observation, and with
  each `resample` mode. This separates the cost of not knowing your own draws from the cost of not
  knowing their hand, and puts a hard **upper bound on what all belief modelling can ever buy**. If
  the gap is ~2 points, the belief arm is settled and M6's B estimator can be dropped.
- **`tools/diag_search_curve.py`** — one fixed network at 1 / 10 / 100 / 1,000 simulations. A flat
  curve rules out search-heavy designs (and interface plan Milestone E); a steep one rules out M8.
  It also locates the operating point for every comparison above.

### Decision rules, written in advance

| Finding | Consequence |
|---|---|
| Hidden-info gap < 3 points | Drop the belief arm; student estimator only |
| Search curve flat from 10 → 100 sims | DMC (M8) becomes the primary design; E1/E2 stay off |
| Search curve steep to 1,000 sims | Turn on interface plan E1 (process fork, WSL); re-budget |
| Regimes within noise at 100 sims | Report as such; pick the cheaper (within-turn) and say why |
| Multi-select variants within noise on the position suite | Keep enumerate (simplest); note the Phase 2/3 caveat |

### If the budget shrinks: what to cut, in order

The matrix degrades gracefully. Cut from the bottom of this list, and each remaining row is still a
coherent result you can write up:

1. **8 cells → 4** — drop the mismatched train/play cells. You lose the ability to separate "plays
   better" from "trains better", and keep the headline regime comparison.
2. **2 seeds → 1 per arm** — the headline comparison becomes suggestive rather than conclusive.
   Report it as such and lean on learning curves at matched checkpoints, which carry much more
   signal per run than endpoints do.
3. **2 arms → 1** — pick within-turn (cheaper, and better matched to the game's structure). The
   deliverable becomes "an AlphaZero-style agent for KeyForge plus the diagnostics", not the regime
   comparison. **This is still a complete project.**
4. **Trained arms → 0** — Tier 1 only: search with the heuristic evaluator, both diagnostics, and
   the behaviour-cloned agent. The thesis becomes an engineering-and-measurement contribution:
   what it costs to search this game, what hidden information is worth, what search is worth.

Decide which of these you are on **before** starting a run, using the measured throughput from M10,
not after discovering you ran out of time.

---

## Milestone M10: instrumentation, budget tiers and go/no-go gates

**Build this early** — the telemetry lands with M0/M3, long before the gates matter, because a
screen you cannot measure is a screen you cannot trust either.

### Telemetry

`agent/telemetry.py` (stdlib only, so actors can use it) and `tools/monitor.py`. Every actor and the
learner append JSONL records to `$KEYFORGE_DATA/runs/<run_id>/metrics.jsonl` once a minute.

- **Actors:** games/hour, decisions/sec, mean simulations/decision, mean fork cost (bucketed by
  prefix length, since replay forking is linear in it), network evaluations/sec, evaluation-cache
  hit rate, mean game length in turns and decisions, forfeits, resignations.
- **Learner:** gradient steps/sec, positions consumed/sec, buffer fill, each loss component,
  **policy entropy**, value-head calibration error, gradient norm, learning rate.
- **System:** GPU utilization and memory, per-worker CPU, inference batch-size distribution and
  round-trip latency, shard write rate, disk used.
- **Ladder:** every promotion attempt with its SPRT verdict, and current-best Elo against each
  frozen anchor.

`tools/monitor.py` prints one screen, and its job is the **derived** numbers, not the raw ones:
games/hour, hours to the next gate, projected total games by the budget's end, and the
**actor:learner ratio** (positions produced ÷ positions consumed).

**The three numbers that actually drive decisions:**

| Number | Healthy | Unhealthy means |
|---|---|---|
| Games/hour | within 60% of projection | re-plan the tier before continuing |
| Actor:learner ratio | ~0.5–2 | ≫2: learner starved. ≪0.5: actors starved |
| Promoted Elo slope per 10k games | clearly positive | converged, stalled, or broken |

Everything else in the metrics file is for diagnosis once one of those three goes wrong.

### Budget tiers

Each tier is a **complete, reportable experiment**, and each one's output is the next one's input.
Pick by what you have. Times are for this machine, from the measurements in this document.

| Tier | Machine time | What runs | What you can claim | What you give up |
|---|---|---|---|---|
| **0** | hours | M0–M3 (all five screens) | Encoder, head and multi-select decisions on per-decision metrics; a behaviour-cloned agent rated against both bots; how much generalizes to unseen cards | Everything about search and RL |
| **1** | ~2 days | + M4, M5 with the heuristic evaluator, + both diagnostics | Whether search helps at all and by how much; what hidden information costs; the search-vs-simulations curve | Any trained-network comparison |
| **2** | ~4 days | + one M7 run (one arm, one seed, 50k games) | A trained AlphaZero-style agent and its rating ladder; leaf estimator and play regime compared as inference-time swaps on it | The training-regime comparison |
| **3** | ~7–10 days | 2 arms × 2 seeds + the full 8-cell matrix | The full ablation as designed | — |
| **4** | more | + M8's DMC arm, larger networks, Phase 2 pool (M11) | Cross-paradigm comparison and generalization across pools | — |

**Tier 1 is the one to protect.** It is cheap, it produces the two diagnostics that can cancel whole
arms above it, and it is the difference between "I built an agent" and "I measured what this game
needs", which is the more defensible thesis if compute runs short.

### Go/no-go gates

Each gate has a number and a consequence. Write the verdict into the run journal.

| Gate | When | Pass | Fail → do this |
|---|---|---|---|
| **G1** | End of M3 | BC agent ≥ 40% vs `HeuristicBot` over 2,000 paired seeds | Fix the encoder or head. **Do not start search work** — everything downstream inherits the defect |
| **G2** | End of M5 | Search + heuristic evaluator ≥ 55% vs `HeuristicBot` at 200 sims | The search is broken, not the network. Re-run M4's availability-count and backup-sign tests. **Do not train** |
| **G3** | Diagnostics | — | Apply M9's decision rules; arms may be cancelled here, which is a success, not a setback |
| **G4** | First 20k games of any M7 run | Promoted checkpoint beats the BC warm start on SPRT | Stop the run. Days of GPU time on a loop that isn't learning is the most expensive failure available here |
| **G5** | 2 hours into any run | Measured games/hour ≥ 60% of projection | Re-plan the tier *now*, using measured throughput, before committing further |
| **G6** | Every 25k games | Elo slope over the last 25k exceeds noise | The run has converged or stalled. Bank the checkpoint and stop; spend the time on the matrix instead |

### Scaling knobs, ranked by damage per unit saved

Turn these down from the top when short, and up from the bottom when you have more:

1. **Evaluation games per pairing** — SPRT already does this automatically. No damage to the agent.
2. **Simulations 100 → 50** — costs policy-target quality, and the search curve tells you exactly
   how much before you commit.
3. **Buffer size / positions retained** — cheap, mild.
4. **Seeds per arm 2 → 1** — costs the statistical validity of the headline claim only.
5. **Games per run 50k → 25k** — costs final strength; report the learning curve, not an endpoint.
6. **Drop the belief arm** — free if G3's hidden-information gap came back small.
7. **Drop the full-game arm** — costs the headline comparison. Last resort.

**When you have more:** turn on interface plan E1 (flat 0.69 ms process fork, worth roughly 2× late
in games), then raise simulations, then add M8's DMC arm, then grow the pool (M11) — in that order,
because that is decreasing throughput-per-unit-of-insight.

### Failure playbook

| Symptom | Likely cause | Action |
|---|---|---|
| Policy entropy collapses in the first few thousand games | Too little root noise, or τ decaying too early | Raise ε toward 0.35; check the τ schedule actually fires |
| Value output saturates at ±1 everywhere | Outcome leaking into features, or buffer window far too short | Audit features against the M1 leak test; widen the window |
| Search plays *worse* than the raw policy | Availability counts or backup sign | Re-run M4's acceptance tests — both failures are silent |
| Promotion never passes | Gate too strict, or genuinely not learning | Check Elo slope and loss separately before loosening the gate |
| GPU utilization < 30% | Actors starved by fork cost | Raise K (concurrent games/worker); enable E1 |
| Actor:learner ratio ≫ 2 | Learner too slow | Larger batch, fewer auxiliary heads, or fewer actors |
| Games lengthen as training proceeds | Value miscalibration; agents avoiding commitment | Check calibration before enabling resignation |
| Throughput decays over a long run | Log growth, buffer growth, or a leak | Apply interface plan L.6's log counters; watch RSS |
| OOM on the GPU | Batch or K too high | Lower batch first — it costs less than lowering K |

---

## Milestone M11: scaling to larger pools, deck sources and formats

This milestone is mostly **verification that earlier choices held**, not new construction. That is
the point: extension should be a data change.

### The five invariants that make extension cheap

1. **72 entity tokens, always.** Every card is in exactly one zone and both decklists are public, at
   any pool size. This holds for Phase 2/3, random legal decks and alliance decks, all of which are
   36 cards a side. If a future format changes deck size, the constant moves into `spec.py` and
   padding plus an entity mask switch on — a contained change, and the only one anticipated.
2. **Card identity is a versioned, append-only vocabulary index**, sized for all 370 from day one.
   New cards are new rows; no existing index ever moves.
3. **Every entity also carries full attribute features**, so a card the network has never seen is
   never a cold one-hot. This is the actual generalization mechanism, and M3's Screen 5 measures it.
4. **Both decklists are already in every observation**, so generalizing across decks is a *data*
   problem, not an architecture one.
5. **Options are scored through features, never a fixed action vocabulary**, so the policy head does
   not grow with the pool at all.

### Curriculum, each stage warm-starting from the last

1. Fignor vs Igor, 49 cards — this plan.
2. Phase 2 presets, then random legal decks from the 159-card pool.
3. Phase 3, 370 cards, random legal decks (any 3 of 7 houses).
4. Alliance decks.
5. Reversal and Adaptive formats — the encoder already carries match context and the bid/first-player
   decision kinds, so this is a data and evaluation change, not an encoder change.

### The generalization experiment (the thesis payoff)

Train on a sampled set of decks; evaluate on decks **never seen in training**, using
`--random-decks`. Plot win rate against `HeuristicBot` on held-out decks as a function of the number
of training decks. A per-ID-only encoder cannot run this experiment at all, which is precisely why
both identity paths are built in M2 and compared in M3.

Shards are partitioned by deck-pair so that a held-out evaluation set is clean by construction
rather than by filtering after the fact.

### What actually breaks at scale, and the mitigation

| Breaks | When | Mitigation |
|---|---|---|
| `CHOOSE_CARDS` space exceeds the enumeration cap (254 observed today) | Larger pools, bigger boards | The sequential treatment from M3 becomes mandatory. **Keep it built and tested even if enumerate wins at Phase 1.1.** The cap is a config value that raises a hard error, never a silent truncation |
| `ORDER_EFFECTS` grows past n = 3 | Trigger-dense Phase 3 boards | Sequential permutation generation — linear nodes, never `n!` options at one node |
| The five dormant decision kinds start firing | Phase 2/3 cards, Reversal/Adaptive | Already encoded and conformance-tested (M1 acceptance 4), but they arrive with near-zero training mass — oversample them in the buffer for the first few thousand games |
| Mirror augmentation stops being exact | Phase 3 | Four engine sites force a left-flank placement (Overlord Greking, Collar of Subordination, Harland Mindlock's revert, Spangler Box). Absent from Fignor/Igor; at Phase 3, skip augmentation for games containing them |
| Buffer composition skews | Many decks | Measure the buffer window in **games**, not positions, and stratify by deck-pair |
| Network under-capacity | 159 and 370-card pools | Re-run M3's capacity ablation per pool and move up the M2 ladder; warm-start from behaviour cloning on the new pool |
| Reserved feature slots run out | Enough new state | A deliberate `FEATURE_VERSION` bump at a moment of your choosing, not an accident (M0, M1) |

### Acceptance

- A Phase 1.1 checkpoint **loads against a Phase 2 vocabulary and plays legal, non-random KeyForge
  with no retraining** — new rows attribute-initialized, old rows and optimizer state preserved.
- The held-out-deck experiment runs end to end and produces the curve above.
- The conformance suite (interface plan K) passes on random legal decks, alliance decks, and the
  Reversal and Adaptive formats.
- **No `FEATURE_VERSION` bump was required to reach the Phase 2 pool.** If one was, the reserved-slot
  design in M1 failed and is worth fixing before Phase 3 rather than after.

---

## Compute budget

Estimated from the measurements above. **These are projections to be replaced by measurement:** at
gate G5, two hours into any run, recompute every figure here from the observed games/hour and
re-pick the tier. The formula is `games/hour = 3600 ÷ (decisions/game × (1 − forced) × mean sims ×
per-sim cost ÷ workers)`, and every term on the right is in the metrics file.

**Assumptions:** 152 decisions/game, 7% forced and auto-resolved, playout cap randomization giving
~44 mean simulations/decision, ~1.5 ms replay fork plus ~0.5 ms tree and evaluation overhead per
simulation, 5 workers.

| Item | Per game | Throughput | Tier | Notes |
|---|---|---|---|---|
| Data generation (no search) | ~5 ms | ~200 games/sec/worker | 0 | M3 corpus in minutes |
| Search with the heuristic evaluator + diagnostics | — | hours | 1 | the arm-cancelling measurements |
| Self-play with search, 100 sims | ~6,200 sims ≈ 12 s/worker | **~1,500 games/hour**, ~35,000/day | 2+ | 5 workers |
| One 50,000-game training run | — | **~1.5 days** | 2 | one arm, one seed |
| Four runs (2 arms × 2 seeds) | — | **~6 days** | 3 | the bake-off's training cost |
| Evaluation, 8 cells round-robin at ~700 games/pairing | — | **~1 day** | 3 | SPRT usually finishes sooner |

**Total: roughly 7–10 days of machine time** for the full bake-off, plus the M3 screens (hours) and
the diagnostics (hours). Interface plan E1's flat 0.69 ms fork is worth roughly a 2× throughput
improvement late in games and is the first lever to pull if these numbers slip.

Every figure here is an estimate keyed to `tools/bench_engine.py`. Re-measure before committing to a
multi-day run.

---

## Risks

- **Availability counts.** The easiest thing to implement wrong in M4, and a wrong implementation
  produces an agent that merely looks mediocre rather than one that crashes. The statistical
  acceptance test in M4 is the only thing that catches it.
- **Backup sign errors across change of acting player.** Same failure mode: silent, not loud. The
  forced-win/forced-loss position tests exist for this.
- **`CHOOSE_CARDS` at 13%.** The random-play profile in the interface plan understated this by 2.5×.
  If the multi-select treatment is weak, it degrades one decision in eight.
- **Encoder drift.** Any change to `spec.py` invalidates stored tensors and every checkpoint.
  `FEATURE_VERSION` plus storing replay records rather than tensors is the mitigation.
- **Rules changes invalidate training data.** Records are only valid within one `ENGINE_VERSION`
  (interface plan A). A Phase 2/3 card fix mid-project means regenerating.
- **One seed per training arm** would make the headline comparison statistically indefensible. Budget
  for two, or report the limitation plainly.
- **Belief head miscalibration** would make the B estimator worse than the student while costing 8×
  more evaluations. The calibration acceptance test in M6 gates it.
- **The heuristic ceiling.** Past ~95% against `HeuristicBot`, it stops discriminating and the ladder
  has to run on frozen self-play checkpoints.
- **Planning from this document's estimates instead of from measurement.** Every throughput figure
  here is a projection from micro-benchmarks, and real pipelines lose factors of two to things no
  benchmark shows. G5 exists precisely for this, and the failure mode it prevents — discovering at
  day six of a ten-day budget that the run needed fifteen — is the most likely way this project runs
  out of time.
- **Reserved slots running out quietly.** If `spec.py` stops reporting remaining capacity, or a
  feature gets added outside the reserved blocks without a version bump, checkpoint compatibility
  breaks silently and every stored tensor becomes subtly wrong rather than obviously broken. The
  M1 test is the only guard.
- **Building extension features nobody uses.** The mirror of the above: the sequential multi-select
  treatment, the attribute path and the dormant decision kinds all cost real work for pools that may
  never be reached. They are kept because each is cheap *now* and expensive to retrofit, but that
  judgement should be revisited if Phase 1.1 alone ends up filling the whole project.

---

## Decisions taken (flagged rather than buried)

- **72 fixed entity tokens.** Every card is always in exactly one zone and both decklists are public,
  so observations have no variable-length entity set. Hidden information is expressed as a *zone*
  feature (`opp_unseen`, `my_deck_unordered`), never by omitting an entity.
- **Perspective-relative encoding.** Everything is "mine/theirs", never "player 1/player 2", so both
  seats train one network.
- **Pointer policy head, no fixed action vocabulary.** Illegal options are unrepresentable rather
  than masked. The fixed-vocabulary head stays available as an M3 ablation.
- **Multi-select is decided by per-decision metrics, not win rates.** At 13% of decisions, its effect
  is ~1 point, which needs ~10,000 games per pairing to resolve by outcome.
- **Only the search regime gets separate training runs.** Everything else is an auxiliary head or an
  inference-time swap — this is what turns 12 cells into 4 runs.
- **Two seeds per training arm**, budget permitting, because one makes the comparison indefensible.
- **`agent/` never imports `torch`.** This preserves the process-fork backend and the PyPy option.
- **Diagnostics run before the matrix.** The hidden-information and search-curve diagnostics can
  cancel entire arms for a fraction of their cost.
- **Tiers, not a single plan.** Each tier is a complete and reportable experiment rather than a
  partial one, so running out of budget degrades the result instead of voiding it. Tier 1 is
  protected because it carries the diagnostics.
- **Runs are budgeted in games, resumable, and journalled.** Mid-run hyperparameter changes are
  expected and recorded against the game index at which they happened, because a learning curve
  annotated with its interventions is still reportable and an unannotated one is not.
- **Reserved feature slots, from the first version.** 16 per entity, 24 global, 8 per option, plus
  spare keyword and intent bits. Adding a Phase 2/3 feature in a reserved slot keeps every
  checkpoint loadable; adding one outside them costs all of them.
- **Both card-identity paths are built** (embedding and attributes), because the attribute path is
  the only thing that carries a card the network has never seen, and it is also the only way to run
  the held-out-deck generalization experiment.

---

## Sequencing

| Stage | This plan | Tier | Needs from the interface plan | Produces |
|---|---|---|---|---|
| Foundation | M0, M1, M10 (telemetry) | 0 | 0, A, B, C, F | leak-free encoded observations; measurable everything |
| Screens | M2, M3 | 0 | K | a playable BC agent; multi-select, encoder and generalization decided |
| Search | M4, M5 | 1 | D, G | search agents that beat the heuristic with no network |
| Estimators | M6 | 1–2 | J (privileged labels) | belief + oracle heads |
| Training | M7, M8 | 2–4 | H, I | trained checkpoints, rating ladder |
| Comparison | M9, M10 (gates) | 3 | I, J | the thesis evaluation chapter |
| Extension | M11 | 4 | — | Phase 2/3 pools, random and alliance decks, all formats |

The minimum path to something that plays KeyForge at all is **M0 → M1 → M2 → M3**, which needs no
forking, no determinization, no search and no self-play — Tier 0, a matter of hours.

**M10 is split across the schedule on purpose:** its telemetry is built at the very start, because
the screens need it too, while its gates and tiers apply from M7 onward. Do not defer the
instrumentation to the point where you need it — by then you are already spending the budget you
cannot measure.

---

## Options register (extends the interface plan's)

### Networks and features

| Option | Gives | Decided by | Default |
|---|---|---|---|
| Trunk depth 2 / 4 / 6 | Capacity vs evaluation cost | M3 Screen 4 | 4 |
| Card-ID embedding / attributes / both | Sharpness vs generalization to unseen cards | M3 Screen 4 | both |
| Pointer head / fixed vocabulary | Action representation | M3 Screen 4 | pointer |
| Trait hashing buckets (32) | Trait features without a vocabulary | — | 32 |
| fp16 / bf16 inference | GPU throughput | Benchmark | fp16 |

### Search

| Option | Gives | Decided by | Default |
|---|---|---|---|
| Regime: within-turn / full-game | The headline axis | M9 | both trained |
| Rollout policy: heuristic / network / random | Turn completion in regime A | M9 ablation | heuristic |
| Leaf estimator: student / belief+oracle | Hidden-info handling | M9 + hidden-info diagnostic | student |
| Belief samples K | Estimator variance vs cost | M9 | 8 |
| `c_puct` | Exploration | Tuning sweep | 1.5 |
| Dirichlet α, ε | Root exploration | `10 / mean branching` | 0.8, 0.25 |
| Virtual loss | In-search batching | Benchmark | 1 |
| Branch on opponent mid-turn decisions (regime A) | Fidelity vs cost | Ablation | off |

### Training

| Option | Gives | Decided by | Default |
|---|---|---|---|
| BC warm start | Skips the near-random phase | M3 results | **on** |
| Playout cap randomization (75/25 at 25/100) | ~2.3× cheaper search per game | — | **on** |
| Mirror augmentation | 2× data, exact for Fignor/Igor | Pool check | **on** |
| Oracle annealing vs distillation | Two routes to the student | M6 | distillation |
| Resignation | Shorter games | False-resignation rate | off until calibrated |
| DMC arm (M8) | Search-free contrast | Search curve | built, run if curve is flat |
| Seeds per training arm | Statistical validity | Budget | 2 |

### Budget and scale

| Option | Gives | Decided by | Default |
|---|---|---|---|
| Tier 0 / 1 / 2 / 3 / 4 | How much of the plan runs | Measured games/hour at G5 | decide at G5, re-decide at G6 |
| Run budget in games | Resumable, rescopable runs | Tier | 50,000 |
| Telemetry sampling interval | Monitoring cost vs resolution | — | 60 s |
| Promotion gate strictness | Ladder quality vs promotion rate | Promotion rate | SPRT, 400 games |
| Evaluation matrix: 8 / 4 / 2 / 0 cells | Depth of the comparison | Budget, via M9's cut list | 8 |
| E1 process fork | ~2× late-game throughput | Search curve; WSL only | off until needed |

### Pool and extension

| Option | Gives | Decided by | Default |
|---|---|---|---|
| Pool: Phase 1.1 / 2 / 3 | Card vocabulary in play | Tier 4; project scope | Phase 1.1 (49 cards) |
| Deck source: presets / random legal / alliance | Generalization beyond two decks | M3 Screen 5 | presets |
| Identity encoding: ID / attributes / both | Generalization to unseen cards | M3 Screens 4 and 5 | both |
| Network capacity rung | Capacity matched to pool size | M3 capacity ablation, per pool | 128 / 4 layers |
| Multi-select: enumerate / sequential | Survives pool growth | Enumeration cap exceeded | enumerate, sequential kept tested |
| Mirror augmentation | 2× data | Exact for Fignor/Igor; per-game check at Phase 3 | on |
| Dormant decision kinds oversampling | Training mass for rare kinds | Phase 2/3 only | off |
