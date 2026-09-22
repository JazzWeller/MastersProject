# Agent Interface Plan: one seam, every agent

## Context

The engine today supports exactly two agent shapes: `RandomBot` and `HeuristicBot`, both pure
functions of the redacted `PlayerView`, both driven one decision at a time through
`Controller.decide(view, decision)`. Every other agent family we might want — behaviour-cloned
networks, within-turn search, full-game ISMCTS, belief-modelling search, oracle/cheating
baselines, search-free Deep Monte-Carlo, parallel self-play training — needs capabilities the
engine does not expose. This plan adds those capabilities **once**, so that any of them plugs in
with no further engine, harness or GUI change.

### Decisions (2026-09-21)

- **Maximum options, decide usage later.** Every capability in this plan is designed so it *can*
  be switched on. Which ones actually get used is decided later, by measurement. The Options
  register at the end lists each switch, what decides it, and its default.
- **Platform: training in WSL, GUI on Windows.** Self-play, training, data generation and the
  evaluation harness run in WSL2 Ubuntu (26.04), where the GPU is visible (RTX 5060 Ti, driver
  610.47). The GUI and interactive play stay on Windows.
- **Interpreter: CPython 3.12 on both sides.** `keyforge`, `bots` and `sim` also stay 3.11-syntax
  compatible, which keeps the PyPy option open.
- **Formats: Archon first, all formats supported.** The critical path is single Archon games. But
  Reversal and Adaptive matches, and every deck source (presets, random legal decks, alliance decks),
  go through the same interfaces, so any agent can play any of them without changes.
- **Randomness: keyed per event.** Breaking previously saved games is accepted (Milestone A).

### Measured baseline

Fignor vs Igor, 200 games, `RandomBot` both seats, single-threaded:

- **223 decisions/game**, **39,224 decisions/sec**, ~134 games/sec.
- **11.7%** of all decisions have exactly one legal outcome.
- **First player wins 57.5%** — seat and deck bias are large and confounded, and must be
  controlled in every comparison this harness ever reports.
- **Competent play is much shorter**: `HeuristicBot` mirrors average **19.9 turns / 151
  decisions** per game at 195 games/sec (29,324 decisions/sec), against 55.3 turns / 223 decisions
  for `RandomBot`. Early self-play agents sit near the random end.
- **This machine: 6 physical / 12 logical cores, 32 GB RAM, NVIDIA RTX 5060 Ti (8 GB).** WSL
  Ubuntu is installed. PyTorch is not installed yet.

| Decision kind | % of all | max options | worst choice space | p99 space |
|---|---|---|---|---|
| `CHOOSE_ACTION` | 64.1% | 19 | 19 | 13 |
| `CHOOSE_HOUSE` | 22.4% | 3 | 3 | 3 |
| `CHOOSE_CARDS` | 5.1% | 27 | 379 | 154 |
| `CHOOSE_FLANK` | 3.9% | 2 | 2 | 2 |
| `TAKE_ARCHIVE` | 2.4% | 2 | 2 | 2 |
| `CHOOSE_HOUSE_FOR_EFFECT` | 1.1% | 3 | 3 | 3 |
| `MULLIGAN` | 0.9% | 2 | 2 | 2 |
| `ORDER_EFFECTS` | 0.2% | 2 | 2 | 2 |

`YES_NO`, `CHOOSE_NUMBER`, `CHOOSE_MODE`, `BID_CHAINS` and `CHOOSE_FIRST_PLAYER` never fire in the
Phase 1.1 Fignor/Igor pool, but the interface must cover them because Phase 2/3 cards and the
Reversal/Adaptive formats use them.

### Verified engine facts this plan depends on

Each of these was checked directly, not assumed:

- **A `Game` cannot be copied.** It holds `self._driver = self._run()`, and every card effect is a
  `yield from` inside that generator; `copy.deepcopy(game)` raises
  `TypeError: cannot pickle 'generator' object`. The same is true of `Match`
  (`match.py:113`).
- **Replay is the only working fork, and it is linear in game length.** `replay(config, record,
  upto)` measured at **1.11 / 2.01 / 2.91 / 3.95 ms** at decisions 50 / 100 / 150 / 200 of a
  200-decision game — ≈20 µs per decision of prefix.
- **The data state *does* copy once the driver is detached — but slowly.** At decision 110 (157 log
  events), with `_driver` removed: plain `deepcopy` **4.44 ms**; sharing card definitions **3.61 ms**;
  also sharing log events **2.60 ms**; upper bound with choice logs shared **2.32 ms** — against
  ≈2.2 ms to replay to that same point. Generic `deepcopy` is therefore *not* a win; only a
  hand-written copy could be (Milestone E).
- **Seeded games are deterministic across processes.** 30 seeded games produce byte-identical
  choice records, winners and turn counts under `PYTHONHASHSEED` = 0, 1, 2 and 12345 — the engine
  already sorts where set iteration would otherwise vary (e.g. `player_houses`,
  `game.py:195`). A replay record is therefore a valid unit of transfer between Windows `spawn`
  worker processes. This needs a regression test so it stays true as cards land.
- **Card instance ids are process-global.** `cards/card.py:11` uses a module-level
  `itertools.count(1)`, so the same seed played twice in one process yields ids 5905… and then
  5977…. Choice records are immune (they store option indices), but anything keyed by
  `instance_id` — option keys, state hashes, information-set keys, tree reuse, the `iid=` fields in
  the log — is not stable across forks or games.
- **The engine reads its own log for rules.** `forged_key_on_turn` (Key Hammer, Tendrils of Pain),
  `creatures_destroyed_in_fight_this_turn` (The Warchest) and `creatures_played_on_turn` (Lifeweb)
  scan `self.log.events` (`game.py:516–541`). The log is game state: it cannot be disabled in
  search forks for speed, it must be in any state hash or snapshot, and redaction must happen on
  the view side only.
- **End-of-turn cleanups are closures over live objects.** Six sites append lambdas to
  `game._end_of_turn_cleanups` that capture cards or the game itself — `dis.py:367`,
  `logos.py:418`, `mars.py:396`, `mars.py:662`, `sanctum.py:505`, `untamed.py:209`. `deepcopy`
  shares function objects by reference, so a copied game's cleanups would mutate the *original*
  game. They also make the state unpicklable.
- **One RNG stream serves everything.** `game.rng` drives the setup shuffles (`game.py:270`), the
  first-player pick (`:276`), mulligan shuffle-ins (`:304`), the random card pick in
  `steps.py:156`, and every shuffle-in effect across the house modules. Consumption therefore
  depends on the players' choices, and a replay fork inherits the *true* future RNG state.
- **Hidden card identities leak through the log.** `archive`, `return_to_hand` and
  `shuffle_into_deck` events record `card=<name>, iid=<id>`, and `build_view` hands the last 20 raw
  events to both players via `log_tail`.
- **Some public decisions leave no trace.** A declined mulligan and a declined archive pickup are
  not logged (`game.py:300–306`, `:346–357`), though both are public in real play.
- **Constructed positions are not replayable.** `tests/helpers.py` builds scenarios by mutating
  state directly (`put_creature`, `put_artifact`, `hand_card`, `make_card`), so no choice record
  reproduces them, and replay-forking one is impossible.
- **`Match` has no replay.** `replay()` is `Game`-only; `Match` has its own driver and RNG
  (`match.py:95`), and match-level decisions arrive with a `MatchView`.
- **Agents are hardcoded at both entry points.** `sim/simulate.py:98` builds `RandomBot` for both
  seats (`--p1`/`--p2` accept only `"random"`), and `gui/engine_bridge.py:75` and `:214` build
  `HeuristicBot` for every AI seat.
- **Checked and harmless:** `current_queue._sequence` is also a module-level counter, but
  `QueueItem` and `new_batch_id` are unused, so it affects nothing.

### What each agent family needs

| Family | Observation | Provenance | Fork | Determinize | Batch | Hooks | Parallel | Trajectories |
|---|---|---|---|---|---|---|---|---|
| `RandomBot` / `HeuristicBot` | yes | — | — | — | — | — | — | — |
| Behaviour-cloned net | yes | **yes** | — | — | **yes** | — | **yes** | **yes** |
| Within-turn search | yes | **yes** | **many** | chance only | **yes** | reuse | **yes** | **+ policy** |
| Full-game ISMCTS | yes | **yes** | **many** | **yes** | **yes** | reuse | **root-parallel** | **+ policy** |
| Belief-modelling ISMCTS | + full history | **yes** | **many** | **weighted** | **yes** | **yes** | **root-parallel** | **+ hidden labels** |
| Oracle / cheating baseline | privileged | **yes** | exact | — | **yes** | — | **yes** | **+ hidden labels** |
| Search-free Deep MC | yes | **yes** | — | — | **yes** | — | **yes** | **yes** |

Every cell that isn't already "yes" for the two existing bots is covered by a milestone below.

### Scope boundary

**Out of scope:** converting the generator-driven card-effect layer to an explicit continuation
stack. Card effects stay generators.

**In scope, as options:** Milestone E adds two faster branching backends behind the same API:
- a **process-fork backend** for Linux/WSL, where `os.fork()` copies the whole process, suspended
  generators included;
- a **snapshot-copy backend** for any platform. At the three top-level boundary decisions (88.9% of
  all decisions), no card effect is suspended, so only the turn loop has to become resumable, not the
  card effects.
No agent is affected by which backend is active.

---

## Milestone 0: baseline, environments, benchmark

**Baseline and branch.** Both suites green, then branch `agent-api` off `master`.

**Two environments, one checkout.**
- **Windows**: a 3.12 virtual environment for the GUI, tests and interactive play. The current
  `python` is the Microsoft Store 3.12.10 build; a python.org 3.12 is an equally good base.
- **WSL Ubuntu 26.04**: its system Python is 3.14.4, so install CPython 3.12 separately
  (`uv python install 3.12` works on any Ubuntu release) and create a 3.12 virtual environment.
  - Install PyTorch there from a CUDA 12.8+ wheel index, since the RTX 5060 Ti is Blackwell.
  - Confirm `torch.cuda.is_available()` inside WSL.
- **One working tree.** WSL runs the Windows checkout through `/mnt/c`. Measured engine throughput
  is unaffected, because only imports cross that boundary. Both sides run 3.12, so they can share
  `__pycache__`.
  - *Option:* a second clone inside WSL, synced through git, if git or import latency across
    `/mnt/c` becomes annoying.
- **All pipeline output lives on the Linux filesystem**: trajectory shards, checkpoints, JSONL
  results and caches, under a data root set by one environment variable (e.g. `KEYFORGE_DATA`).
  Heavy file I/O across `/mnt/c` is slow.
- **The GUI reaches checkpoints** either through `\\wsl$\Ubuntu\…` (the Windows 10 path) or through
  an explicit export step.
  - Records and the registry refer to checkpoints by content hash plus a per-platform root, never by
    absolute path — Windows and Linux paths differ.
- **`.gitattributes`**: `*.sh text eol=lf`. Otherwise helper scripts run from WSL break on the CRLF
  line endings of a Windows checkout.

**Cross-platform determinism.** The Milestone A determinism test and the golden seed corpus
(Milestone L) run under both Windows 3.12 and WSL 3.12, and must produce identical digests. A saved
game made on either side must replay on the other.

`tools/bench_engine.py` — a repeatable command reporting decisions/sec, games/sec, the decision-kind
histogram, choice-space percentiles, replay-fork cost at 25/50/75/100% of a game, and copy cost.
Every milestone re-runs it. These seams sit on the hot path of every future self-play run, and a
quiet 2× throughput regression would cost weeks of training time that nobody would attribute to
this work.

## Milestone A: determinism and identity

Foundational; several later milestones and the contract suite cannot be written without it.

- **Per-game instance ids.** Replace the module-level `_instance_counter` with a counter owned by
  the `Game`, assigned in deck-build order. Same seed → same ids, in any process, in any fork.
  `tests/helpers.make_card` takes the game (or its counter).
- **`Game.state_hash()`** — a canonical hash over everything that affects future play: zones and
  their order, per-card state, players, active effects, pending cleanups, the log, the RNG state and
  the pending decision. Needed by fork equivalence tests, the non-mutation contract test,
  transposition tables and snapshot verification.
- **`Game.outcome_for(pid)`** → `+1` / `0` / `-1`, with the `max_turns` draw mapped explicitly.
- **Engine version stamp.** An `ENGINE_VERSION` plus a rules hash (over card data and effect
  modules) written into every replay record. `replay()` refuses a mismatched record outright rather
  than relying on "the record doesn't fit" — a record made before a card fix can still *fit* and
  silently play a different game, which would poison both training data and ratings.
- **Cross-process determinism regression test** — the `PYTHONHASHSEED` check above, run in the
  suite, so a future `for h in some_set:` in a card effect is caught the day it lands.
- **Portable across interpreters.** Python guarantees only that `random()` produces the same
  sequence across versions. `shuffle`, `choice` and `randrange` may change between releases.
  - Today the seeded-game digests match between CPython 3.12.10 and 3.14.7, but that's luck, not a
    contract.
  - This machine already runs two interpreters (`python` → 3.12, `py` → 3.14, and 3.14 under WSL),
    and PyPy would be a third implementation.
  - The fix: since keyed randomness rewrites every random call anyway, build the engine's shuffles
    and picks on `rng.random()` (its own Fisher–Yates). Records then replay identically on any
    interpreter.
  - Also record the interpreter implementation and version in the version stamp, for diagnosis.
- **Per-event keyed randomness** (decided: breaking previously saved games is acceptable).
  - **What changes.** The single `game.rng` stream is replaced by randomness derived from
    `(seed, player, event kind, per-kind counter)`. So "player 2's third reshuffle" or "player 1's
    second random discard" gets the same randomness for a given seed, whatever else happened in the
    game.
  - **Why.** Today one shared stream means any divergence in play — a mulligan, a shuffle-in
    effect — shifts every later random event for *both* players. Keyed randomness keeps two
    different agents' games dealt alike for as long as their play allows, which is what makes
    paired-seed and duplicate evaluation (Milestone I) low-variance.
  - **Why keyed rather than one stream per player.** Per-player streams still drift whenever one
    agent triggers a shuffle-in that the other doesn't.
  - **Reproducibility is unchanged:** same seed + same choices = same game, in any process.
  - **One-time cost:** every seeded outcome changes.
    - Replays of games saved before the change (the GUI history database,
      `Code/GUI/data/history.sqlite3`) are refused by the version stamp. The history list itself
      still displays, because results are stored as plain columns.
    - Tests that assert on a specific shuffle outcome need their expected values re-pinned once.
  - **Ordering:** land it together with the version stamp, and **before** the golden seed corpus in
    Milestone L is recorded.

## Milestone B: decision provenance

**The problem, with receipts.** `HeuristicBot._choose_cards` dispatches on `decision.prompt`
substrings — `"sacrifice"`, `"to fight"`, `"to ready"`, `"heal"`, `"attach"`, and a literal
`startswith("ozmo:")` — and carries `self._last_mode` across decisions because Ozmo's target prompt
is byte-identical whether the mode was "Heal 3" or "Stun". The same `DecisionKind`, with the same
option type (`List[Card]`), means *destroy these* or *heal these* depending only on which card
asked. A learned agent conditioned on state and options is blind in exactly this way, and no amount
of entity encoding fixes it.

- **`keyforge/enums.py`**: new `DecisionIntent`, with values *derived by audit* of the **141
  `choose_cards` call sites**, 19 `yes_no`, 9 `choose_mode`/`choose_number` and 7 `order_effects`
  — not invented up front. Expect roughly `DESTROY`, `DAMAGE`, `HEAL`, `STUN`, `READY`, `EXHAUST`,
  `DISCARD`, `ARCHIVE`, `PURGE`, `RETURN_TO_HAND`, `CAPTURE`, `ATTACH`, `SACRIFICE`, `FIGHT_TARGET`,
  `USE_TARGET`, `SHUFFLE_IN`, `REVEAL`, `GAIN`.
- **`keyforge/decision.py`**: `Decision` gains `source_card`, `intent`, `affects`
  (`FRIENDLY` / `ENEMY` / `ANY` / `NONE`) and `optional`, all defaulting to `None`/`False`, so the
  GUI, `replay.py` and both bots are untouched by the field addition itself.
- **Invert the prompt.** `prompt` becomes derived from `(source_card, intent)` wherever it can be;
  the GUI's `option_labels.py` keeps rendering it. Today the string is the only carrier of meaning.
- **An unrecognized intent is a hard error**, never a silent default — a silently mislabelled
  decision would corrupt training data invisibly.
- **Acceptance.** Rewrite `HeuristicBot._choose_cards` to branch on `intent`/`source_card` with zero
  prompt parsing, and delete `_last_mode`. Its win rate against `RandomBot` over 2,000 paired seeds
  must be statistically unchanged. The existing bot is the oracle for whether the tags carry enough
  meaning to play on.

## Milestone C: the observation layer (and the leak)

**First, the leak.** Hidden card identities reach both players through `log_tail`. A learned agent
would find this and train on it; a determinized search would be searching worlds inconsistent with
information the agent can actually read. Fix before anything learns.

- **`keyforge/log.py`**: `LogEvent` gains `visible_to: frozenset[int]`, defaulting to both players;
  the zone-transition helpers that log card identities set it to the entitled player only.
  Filtering happens in the *view*. The engine's own log stays complete, because the rules read it.
- **`keyforge/observation.py`** (new): an immutable, serializable per-player observation.
  - **No live `Card` references.** `PlayerView` currently hands out the engine's own objects, so an
    agent can mutate engine state through its own observation, or read a field it is not entitled
    to. Cards appear by per-game `instance_id` plus their public state.
  - **The full public history, built from the decision stream as well as the log**, because a
    declined mulligan or a declined archive pickup is public information that leaves no log entry.
    Every house choice and every card played or held back is here — this is what belief models
    condition on.
  - **Serializable by construction**, since observations cross process boundaries (Milestone H) and
    are written to disk (Milestone J).
- **`observation_key(obs)`** — a canonical hash of an observation, used as the information-set key
  for ISMCTS nodes and for transposition tables. It must be id-stable (Milestone A).
- **Match-level observations**: `BID_CHAINS` and `CHOOSE_FIRST_PLAYER` arrive with a `MatchView`
  between games; the observation layer covers it so match-format agents need no special case.
- **Match context in every in-game observation.** Include:
  - the format;
  - the game number within the match and the match score;
  - the starting chains won in an Adaptive bid;
  - which deck is whose after an Adaptive deck swap.
  In a single Archon game these are constants. One agent can then play every format and condition
  on the stakes, rather than needing a separate agent per format.
- **Deck-agnostic by construction.** Both decklists are already part of every observation, and cards
  are encoded by the versioned vocabulary plus attributes (Milestone F). So preset decks, random
  legal decks and alliance decks all look the same to an agent.
- **Generic decision handling.** Every decision — in-game, match-level, or a future kind such as a
  deck pick or ban for another format — reaches the agent the same way: a kind, an intent, and a
  list of options with keys and features. A learned agent treats the kind as one more input
  feature. So adding a format adds a feature value, not an agent rewrite.
- **`full_state_observation(game)`** — the privileged, everything-visible observation for oracle
  baselines and the hidden-information diagnostic. Only reachable through the privileged
  capability in Milestone G.
- **`PlayerView` stays as it is** for the GUI and the existing bots.
- **Acceptance.** Over 1,000 seeded games, at every decision, player P's observation contains no card
  identity P is not entitled to — including through history.

## Milestone D: forking, branching and determinization (replay backend)

All forking goes through one API. This milestone implements it by replay; Milestone E can swap the
backend.

- **`Game.fork()`** → an independent game at the same point, carrying the *true* hidden state and the
  *true* RNG state. This is an exact duplicate, so it is **privileged**: a normal agent handed it
  could search its own future draws.
- **`Game.fork_determinized(viewer, rng, resample=...)`** → fork, then resample every card whose
  location `viewer` does not know, respecting hand, archive and deck counts on both sides, and then
  **reseed the game RNG from `rng`**. Without the reseed, future random events in the fork — the
  random card in `steps.py:156`, every reshuffle, every shuffle-in — would follow the true game's
  future. `resample` selects what to resample: `OWN_DECK` (chance only — the within-turn search
  primitive), `OPPONENT_PRIVATE` (hand and archive), or `ALL`. Selective resampling also lets the
  hidden-information diagnostic separate "cost of not knowing my draws" from "cost of not knowing
  their hand".
- **`Game.fork_many(n, ..., seeds)`** → n branches from one state, each with its own determinization
  and RNG seed. Under this backend it is n replays. The API exists so that search agents are written
  against a batch primitive, which the fast backend can then serve at a fraction of the cost.
- **A determinized world has its own record**: the true prefix, the determinization seed and
  `resample` mode, then the choices made inside it. Re-dealing mutates hidden state, so without this
  a determinized world could never be forked again or shipped to another worker. Replay records
  stay the one universal currency.
- **Two ways to branch, and which one search agents should use.**
  - **Branch-and-hold**: `fork()` / `fork_determinized()` return a `Game` in the caller's process.
    Only the replay and snapshot-copy backends can provide this.
  - **Branch-and-run**: `run_branches(n, work, determinize=..., seeds=...)` runs `work(game)` on each
    of n branches and returns the results. It works with **every** backend: the replay and copy
    backends run `work` in-process, and the process-fork backend runs it in a forked child and pipes
    the result back.
  - Search agents written against branch-and-run can switch backends with no code change. That is
    the recommended form for any search code that should stay portable between Windows and WSL.
- **`apply(choice_indices)`** and **`run_until(predicate, policy)`** — advance a fork along a
  recorded path, or play it forward with a default policy until a condition holds: end of turn,
  next decision by player P, next boundary decision, game over. The turn-bounded search leaf is
  `run_until(end_of_turn)`.
- **Card location knowledge.** Per card instance, which players know where it currently is — set on
  draw, reveal, archive, put-on-top-of-deck and look-at-hand. `Player.hand_revealed_to` is today's
  coarse version and folds into it. Without it, determinization would scramble cards whose position
  the viewer legitimately knows.
- **Positions as replayable data.** A new `GameConfig.setup_script`: a deterministic list of setup
  operations (place creature, set æmber, put card in hand, …) applied after normal setup and before
  the first decision. `tests/helpers` builds scenarios through it, so a *constructed* position gets a
  replay record like any other and can be forked, shipped to a worker, and used as a position suite.
- **Match records and match replay.** Match-level choices are recorded alongside the per-game
  records, plus `replay_match()` and `Match.fork()`. Match-format agents then fork exactly like game
  agents.
- **Tree-node guidance for search agents**, written into the module docstring: store the path of
  choice indices from the root, never live `Game` objects. State is regenerated by
  `fork` + `apply`. This bounds memory and keeps trees serializable.
- **Acceptance.**
  - Fork equivalence: the original and the fork have equal `state_hash`; diverging one leaves the
    other untouched.
  - Determinization consistency: the viewer's knowledge is preserved, zone counts are preserved,
    and `check_invariants` passes.
  - Statistical: over many samples, each unseen card lands in each legal location at the expected
    rate.
  - **No future leakage:** across many seeds, a determinized fork's subsequent random events are
    uncorrelated with the true game's.

## Milestone E: fast branching backends (options)

Both backends sit behind Milestone D's API, and whether to switch them on is decided by Milestone
J's search-curve diagnostic. Measured costs, with the root at decision 110 of a game:

| Backend | Cost per branch | Valid at | Platforms | Maintenance |
|---|---|---|---|---|
| Replay (Milestone D, default) | ≈16–20 µs × prefix length: 1.1 ms at decision 50, 4.0 ms at decision 200 | any decision | all | none |
| **E1: process fork** | **0.69 ms flat**; 1.85 ms including 20 decisions of work and a pipe back, against 2.29 ms for replay doing the same work | **any decision**, mid-resolution included | Linux / WSL | none |
| E2: snapshot copy | unmeasured for a hand-written copy; generic `deepcopy` measured at 2.3–4.4 ms, which is no better than replay | boundary decisions, plus a short replay | all | high |

### E1: process-fork backend (WSL / Linux)

`os.fork()` duplicates the entire process, including the suspended generator frames that make
`deepcopy` impossible. So the child holds an exact, drivable copy of the game at *any* decision.
- **Cost stays flat** across the game while replay grows linearly: about break-even with replay
  around decision 50, and roughly 4–5× cheaper near decision 200.
- **The six closure sites are a non-issue.** The whole address space is copied, so every captured
  reference in the child points at the child's own objects. No cleanup conversion is needed for
  this backend.
- **Serves branch-and-run.** The search tree stays in the parent, each simulation is a forked child,
  and results come back over a pipe or shared memory. A child re-deals hidden state itself for
  determinized simulations.
- **Constraints:**
  - The forking process must never have initialized CUDA — CUDA does not survive `fork`. So engine
    workers never import `torch`, and all network evaluation goes through the inference server
    (Milestone H).
  - The forking process must be single-threaded, which engine workers already are.
  - Copy-on-write means cost grows with how much memory a child touches. Re-measure with realistic
    search children.
- **Build size: small.** A fork/pipe wrapper implementing `run_branches`, plus the equivalence
  fuzz below run against it.

### E2: snapshot-copy backend (any platform)

The only fast backend that works on Windows and that supports in-process branch-and-hold.

- **Boundary-resumable driver.** At `CHOOSE_ACTION`, `CHOOSE_HOUSE` and `TAKE_ARCHIVE` — exactly
  `simulate.py`'s existing `_BOUNDARY_KINDS` — the generator stack is only `_run`'s turn loop and one
  step of `_take_turn` (`game.py:310–373`). Make the current turn step explicit game state and add a
  `_resume()` generator that re-enters there. A copied game is then drivable. No card effect
  changes.
- **Cleanups as data, not closures.** Convert the six lambda sites to `(operation, instance_id)`
  records resolved against the game at cleanup time. Add a test asserting
  `_end_of_turn_cleanups` never contains a callable, so new cards can't reintroduce the problem.
- **Hand-written `Game.copy()`**: shares immutable card definitions and the append-only log events,
  and copies only mutable state.
- **Non-boundary decisions** (the other 11.1%): copy the most recent boundary snapshot, then `apply`
  the handful of choices made since.
- **When to switch it on.** Where `copy()` beats replay by **≥3× at the median decision** — and, in
  WSL, where it also beats E1. Otherwise it stays off and nothing changes.
- **Equivalence fuzz, for every backend**: at thousands of random points across seeded games,
  every active backend must produce the same `state_hash` as replay, and so must their continuations
  under identical choices. For E2 this is also the maintenance guard: a new card field that
  `copy()` forgets fails it.

## Milestone F: option encoding and the multi-select helper

- **`keyforge/encoding.py`** (new): `option_key(decision, option)` → a stable, hashable key across
  every option type the engine produces (`Card`, the seven action dataclasses, `House`, `bool`, flank
  and mode strings, numbers and player ids). `option_features(decision, option)` → a per-option
  feature vector. Fixed-vocabulary policy heads index the key space; pointer/embedding heads consume
  the features. One module serves both, so that choice stays an agent-level decision.
- **Versioned, append-only card vocabulary**: card-definition → integer id, stored as a data file
  and stamped into checkpoints. When the Phase 2/3 pools add cards, existing ids never shift, so a
  trained network stays loadable.
- **`bots/option_space.py`** (new): the three multi-select treatments, implemented once.
  1. **Enumerate** subsets and permutations under a size cap. Viable today: the worst observed space
     is 379 (a `min_n=0, max_n=2` choice over 27 options), p99 is 154, and `ORDER_EFFECTS` never
     exceeds n = 2 in this pool.
  2. **Sequential decomposition**, with each successive pick restricted to option indices greater
     than the last. That collapses the `k!` orderings that reach the same *set* into one path, so a
     search tree doesn't duplicate work. Done agent-side with virtual nodes; the engine sees one
     submitted list.
  3. **Independent top-k** — cheapest, ignores interactions, kept as a baseline and ablation.
  `min_n = 0` (choosing nothing) is handled explicitly in all three.
- **Acceptance.** Fuzz every strategy against every decision seen across 1,000 random games; every
  produced choice satisfies `Decision.validate`.

## Milestone G: agent protocol and the in-process driver

- **Lifecycle hooks** on the agent protocol, all defaulting to no-ops:
  - `on_game_start(seat, public config)`;
  - `observe(event)` — pushed for *every* public event and decision, including the opponent's;
  - `on_game_end(outcome)`.
  Today a `Controller` only hears about the game when it must decide itself. A belief filter needs
  incremental updates on every opponent move, and MCTS subtree reuse needs to be told which edge was
  actually taken. `HeuristicBot`'s `_last_mode` is already evidence that agents carry state.
- **Budget.** `decide` receives a `Budget` (simulation count and/or wall-clock) set by the harness.
  Compute-normalized comparison is then a harness setting, not a per-agent flag.
- **Capabilities, enforced by construction.** The driver hands each agent a capability object
  matching its registered privilege level:
  - **observation-only** (default) — observations only;
  - **search** — adds `fork_determinized`, `fork_many`, `apply`, `run_until`;
  - **privileged** — adds `fork` and `full_state_observation`.
  An agent cannot reach what it wasn't given, which makes the contract test enforceable instead of
  conventional.
- **`BatchController`** — `decide_many([(obs, decision), ...])`. `Controller` is unchanged; an
  adapter wraps a synchronous agent as a batch of one, so `RandomBot` and `HeuristicBot` run
  untouched.
- **`sim/driver.py`** (new): pumps K games concurrently in one process, groups pending decisions by
  agent, calls `decide_many` once per agent per round, and submits. The engine is already externally
  driven (`pending_decision`/`submit`), so this needs no engine change. It covers `Match` as well as
  `Game`, through the same protocol.
- **Forced-decision auto-resolve** as a driver option: 11.7% of decisions have one legal outcome.
  Still recorded, so replays stay exact.
- **Seeding**: each agent's per-game seed derives from `(run seed, game index, seat)`, so any single
  game out of a parallel run reproduces standalone.
- **Fault isolation**: an agent that raises, returns an illegal choice, or blows its budget forfeits
  that game. The forfeit is recorded with the full replay record for debugging, and the run
  continues — a 10,000-game evaluation must not die on game 7,431.

## Milestone H: parallelism and inference

- **The unit of transfer is the replay record** (config dict, choice indices, engine version) —
  never a `Game`, which cannot be pickled. That holds between processes, between WSL and Windows,
  and between machines. Cross-process determinism is verified above and regression-tested in
  Milestone A. Every entry point gets its `if __name__ == "__main__":` guard.
- **Start methods, per platform.**
  - **WSL, CPython 3.12**: `multiprocessing` defaults to `fork`. That gives fast pool startup and
    copy-on-write sharing of warmed state.
  - **Windows**: only `spawn` exists, so workers re-import everything and inherit nothing.
  - Code must work under both. Never rely on state inherited from the parent.
- **The CUDA rule.** Only the inference server process ever initializes CUDA. Engine workers never
  import `torch`. CUDA does not survive `fork`, so a worker pool forked from a CUDA-initialized parent
  would break — and so would E1's per-simulation forks. Create worker pools before any CUDA
  initialization, or set `forkserver` explicitly for any pool that must start after it.
- **Two levels of parallelism**: P worker processes × K concurrent games each. The GIL means threads
  add nothing to engine throughput; processes are the only lever.
- **`InferenceClient`** abstraction with two implementations:
  - in-process (CPU, small networks);
  - a remote **inference server** process that owns the model (GPU) and batches requests across all
    workers.
  The same client serves driver-level batching *and* in-search leaf batching with virtual loss, so a
  search agent needs no separate path when it moves to the server.
- **Root-parallel search**: ship the root's replay record plus distinct determinization seeds to
  several workers, and merge their root statistics. This is how "branch many games from the same
  state" scales past one core, and it is the natural parallelization of ISMCTS.
- **Checkpoint registry**: versioned weights with content hashes. Actors reload at game boundaries,
  and every record names the checkpoint that produced it.
- **Self-play actor loop**: workers play games with the current checkpoint and write sharded,
  append-only trajectory files. It is resumable after interruption, and no two workers write to the
  same file.

## Milestone I: the evaluation harness

- **`bots/registry.py`** (new): name → constructor with kwargs, plus privilege level. It is the
  *single* source of agents for `sim/simulate.py`, `main.py` and the GUI's opponent choice — today
  `simulate.py:98` hardcodes `RandomBot` and `engine_bridge.py:75` / `:214` hardcode
  `HeuristicBot`. Learned agents register as `name@checkpoint`.
- **Pairing matrix**: both seat orders × both deck assignments, with paired seeds across all
  candidates. Non-negotiable given the 57.5% first-player win rate.
- **Duplicate evaluation**: the same seed is played twice with the agents swapped across both seat
  and deck. Differences then come from play, not the deal. Keyed randomness (Milestone A) is what
  keeps the deals aligned after the agents' choices diverge.
- **Position suites**: fixed files of positions, from replay-record prefixes or `setup_script`
  scenarios. Every candidate plays every position from both sides, which is the way to test a
  specific skill such as æmber denial timing or lethal detection.
- **Every format and deck source through one harness.**
  - `--format archon|reversal|adaptive` is already in `simulate.py` and runs through the driver.
  - Archon is the default.
  - Deck sources: the presets, random legal decks (`--random-decks`) and alliance decks.
  - Ratings are kept per format, because a strong Archon agent isn't automatically a strong bidder.
- **Trained agents in the Windows GUI**, two options behind the same `InferenceClient`:
  - **In-process**: a CPU build of PyTorch in the Windows virtual environment — plenty for one
    interactive game.
  - **Remote**: the GUI's agent calls the inference server running in WSL over `localhost` (WSL2
    forwards it), so the Windows side needs no PyTorch at all.
- **Per-game JSONL records**: agents, agent config hash, checkpoint, engine version, decks, seat,
  seeds, winner, turns, decision count, per-agent wall-clock and network-evaluation counts, forfeits,
  and the full replay record. Ratings are computed from the file afterwards, never during the run.
- **`sim/rate.py`** (new): Bradley-Terry ratings with confidence intervals and an explicit
  non-transitivity report.
- **Sample-size guidance in the tool**: standard error ≈ `0.5/√N` — about 1,000 games to resolve a
  3-point gap and 10,000 for 1 point.
- **Frozen anchors**: `RandomBot`, `HeuristicBot` and saved checkpoints. Watch the heuristic
  ceiling — past about 95% it stops discriminating.

## Milestone J: data generation, baselines and diagnostics

- **Trajectory schema** (`sim/generate.py` and the self-play actors):
  - the observation;
  - the legal options' keys and features;
  - the chosen option;
  - **the agent's policy distribution** — MCTS visit counts, which AlphaZero training uses as its
    policy target and cannot do without;
  - the agent's value estimate;
  - the eventual outcome;
  - a separate, flagged **privileged file** holding the true hidden state (the opponent's actual
    hand and deck order). These are the free labels for belief networks and oracle-guided value
    training, kept apart so they cannot leak into an ordinary agent's inputs.
- **Storage trade-off, decided explicitly**: store replay records and regenerate observations on
  demand (compact, always consistent with the current encoder), or store encoded tensors (fast to
  train from, but invalidated by any encoder change). At ~30k labelled decisions/sec, regeneration is
  cheap: 10,000 games ≈ 2.2M samples in about 75 seconds.
- **A no-network baseline evaluator**: a heuristic value function (from `HeuristicBot`'s
  fight/æmber logic) plus a uniform prior. Search mechanics — ISMCTS availability counts,
  determinization, subtree reuse — can then be tested and debugged before any network exists.
- **Behaviour-cloning and value-regression screens** against `HeuristicBot`, for ranking encoders
  and policy heads with no RL loop and no search.
- **`tools/diag_hidden_info.py`**: the same agent with and without privileged observation, and with
  each `resample` mode from Milestone D. This separates the value of knowing your own draws from the
  value of knowing their hand, and puts an upper bound on what belief modelling can buy.
- **`tools/diag_search_curve.py`**: one fixed network at 1 / 10 / 100 / 1000 simulations. A flat curve
  rules out search-heavy designs and Milestone E; a steep one rules out search-free designs.

## Milestone K: the conformance suite — the definition of done

`tests/test_agent_contract.py`, parameterized over every agent in the registry:

1. **Legality** — every submission satisfies `Decision.validate`, across every `DecisionKind`
   encountered in N seeded games, for `Game` and `Match` alike.
2. **Multi-select correctness** — respects `min_n`/`max_n`, returns a true permutation for
   `ORDER_EFFECTS`, and handles `min_n = 0`.
3. **Reproducibility** — the same seed gives a byte-identical choice record, in-process *and* when
   replayed in a freshly spawned worker.
4. **Non-mutation** — `state_hash` is unchanged across every `decide`, and `check_invariants` passes
   at every boundary.
5. **Termination** — the game finishes within `max_turns`.
6. **Batch/sync agreement** — `decide_many` equals individual `decide` calls under the same seeds.
7. **Hooks** — `on_game_start`, `observe` and `on_game_end` fire in order, with no event missing or
   duplicated, and opponent decisions included.
8. **Budget** — the agent stays within its `Budget`.
9. **Privilege** — the agent never obtains a capability above its registered level. A search agent
   holding a determinized fork cannot recover the true hidden state from it.

An agent that passes this suite runs in the harness, the GUI, the data generator, the self-play
actors and both diagnostics with no further change. **That is the acceptance criterion for this
whole plan.**

## Milestone L: throughput — more games per hour

Every figure below was either measured on this engine or is labelled as an estimate. Each
optimization lands with before/after numbers from `tools/bench_engine.py`, and none may change a
single seeded game.

**Safety net first: a golden seed corpus.** Check in a fixed set of seeded games, each with its
replay record, outcome and final `state_hash`:
- `RandomBot` and `HeuristicBot` pairings;
- Fignor/Igor in both seat orders;
- the `setup_script` positions.
A pure-performance change must leave every one byte-identical. That lets the engine be optimized
aggressively without anyone having to re-verify rules by hand.

### Where the time goes

Profiled over 150 random games, plus direct wall-clock measurements:

| Hotspot | Share | Cause |
|---|---|---|
| `_legal_actions` | ~35% | Recomputed at every `CHOOSE_ACTION` (21k calls). It calls `why_not_playable` for every hand card (115k calls), each of which walks the effects through `player._apply` (391k) and then `duration_effects_for` (460k, a linear filter over all effects). |
| `build_view` | **26% of wall time** | Rebuilt from scratch for every decision: 38.9k decisions/sec with views, **52.5k without**. |
| `player_houses` | ~6% | Re-sorts `{c.house for c in all_cards}` on every call (7.7k calls), although the decklist never changes mid-game. |
| `Enum.__hash__` | ~3% | 302k Python-level hash calls. About 280k of them come from `player_houses`' set comprehension, which hashes 36 enums per call. |
| `why_not_playable` | **27% of engine time once views are gone** | Called for every card in hand at every `CHOOSE_ACTION` (115k calls). Each call recomputes checks that are identical for every card, and formats a reason string for every card it rejects. |

### Engine hot path

1. **Lazy observations.** Build an observation only when an agent asks for one. Never build one
   during `run_until` rollouts driven by option-only default policies. Update incrementally from
   events instead of rebuilding. Measured ceiling: **+35% decisions/sec**.
2. **One play gate per decision, not per card.** Today `why_not_playable` recomputes the
   player-wide checks for every card in hand: the effect scans for can-play, card limits and the
   non-Logos allowance, the first-turn limit, and a `hand.cards()` list copy plus a membership scan.
   It also builds an f-string explanation for every rejected card, which is where most of the
   `House.value` enum overhead comes from.
   - Compute the player-level gate once per `_legal_actions` call, then check each card with a
     boolean fast path.
   - Keep `why_not_playable` (the strings) for the GUI, built on the *same* gate. That preserves its
     docstring guarantee that legality and the GUI's explanation can never disagree.
3. **Index active effects by `(player, variable)`** in `ActiveEffectList`, kept up to date on add,
   remove and tick. `duration_effects_for` then returns matches directly instead of filtering the
   whole list 460k times.
   - Also cache the effect-derived player flags (`get_can_play_cards`,
     `get_non_logos_cards_playable`, …), invalidated on effect changes and house selection.
   - Target (an estimate): halve `_legal_actions`.
4. **Compute `player_houses` once at setup** (~6%). This also removes almost all of the enum
   hashing, since its set comprehension is about 280k of the 302k `Enum.__hash__` calls. So no
   riskier change to enum hashing is needed.
5. **`submit_index(i)` fast path** for the driver, replay and `apply`.
   - It only checks bounds: no `_index_of` object scans and no re-encoding.
   - It is for trusted, version-stamped sources only. Human and GUI input keep full validation.
   - `choice_log` becomes optional. It stores full choice objects and so keeps `Card` references
     alive, while `choice_record` alone is sufficient.
6. **Turn the log-scanning rules into counters.** The three queries at `game.py:516–541` get
   per-turn counters updated on the matching events: O(1) instead of O(log length). This matters
   more as games lengthen and forks multiply.
7. **`slots=True`** on the hot dataclasses (action types, `Decision`, `LogEvent`) and `__slots__`
   on `Card` and the type objects. That gives faster attribute access, less memory, and cheaper
   copies for Milestone E.

### Measured non-wins (don't spend time here)

- **Disabling the cyclic GC**: only +2% (38.9k → 39.6k decisions/sec). Every finished game also
  leaves ~600 objects in reference cycles, so turning the GC off would leak memory. Leave it on;
  `gc.freeze()` after worker startup is harmless.
- **Game setup**: `Game()` to the first decision costs 0.27 ms, so deck resolution needs no
  caching.

### Platform and interpreter (measured)

Same engine, same games, views off:

| Environment | Engine | Start a 5-process pool + import |
|---|---|---|
| Windows, CPython 3.12.10 (Microsoft Store build) | 53,083 decisions/s | 0.41 s (`spawn`) |
| Windows, CPython 3.14.7 | 51,360 decisions/s | 0.55 s (`spawn`) |
| **WSL Ubuntu, CPython 3.14.4** | **62,520 decisions/s (+18%)** | 0.49 s (`forkserver`) |

- **Decided: self-play and training run under WSL.** It is 18% faster on identical code, it is
  where Linux-first ML tooling works best, it installs PyPy most easily, and it enables E1's process
  forking. The GUI stays on Windows.
  - The measured WSL figure is under CPython 3.14.4. Re-measure under the WSL 3.12 environment once
    it exists (Milestone 0).
  - Code runs from the Windows checkout (only imports cross `/mnt/c`); everything the pipeline
    writes goes to the Linux filesystem.
- **Newer CPython is not a free win here.** 3.14 on Windows measured slightly *slower* than 3.12.
- **Worker startup is not a factor** with persistent pools: about half a second, once. Note that
  Python 3.14 made `forkserver` the Linux default, so copy-on-write sharing via `fork` has to be
  requested explicitly.
- **GPU.** The RTX 5060 Ti is a Blackwell card, so PyTorch needs a CUDA 12.8+ build (PyTorch 2.7 or
  later); older wheels don't include Blackwell kernels. CUDA works inside WSL2 through the Windows
  NVIDIA driver. 8 GB is ample for DouZero-scale networks and large inference batches.
- **PyPy for engine workers.** `keyforge`, `bots` and `sim` import only the standard library. So
  the engine workers can run under PyPy while the network stays in a CPython inference server, and
  the Milestone H process split allows that with no further change.
  - PyPy is not installed on Windows or in WSL. `apt` in WSL is the easy route.
  - Check for 3.12-only syntax first, since PyPy targets 3.10/3.11.
  - Portable randomness (Milestone A) is what lets PyPy workers and the CPython GUI replay the same
    records.
  - JIT gains on object-heavy pure Python are commonly several-fold. This engine is built on
    generators, though, so the gain has to be measured rather than assumed.
- **mypyc or Cython** on the legal-action path, `player.py` and `effect_object.py` is the fallback
  if PyPy disappoints. It needs type-clean code and a build step.
- **Free-threaded CPython (3.14t)** is a future option only: it would mean leaving the decided 3.12.
  Threads could share one search tree and evaluation cache without IPC. It depends on Milestone A's
  per-game counters, and on PyTorch's free-threading support at the time.

### Game length (self-play training only — never in evaluation)

- **A training `max_turns` cap** that ends pathological games as draws. It mostly matters early,
  while agents still play at near-random length (55.3 turns rather than 19.9).
- **Resignation**, as in AlphaZero: resign once the value estimate stays below a threshold for k
  consecutive decisions. Play about 10% of games with resignation disabled, to measure the
  false-resignation rate and tune the threshold.
- **Forced-decision auto-resolve** (11.7% of decisions, Milestone G).

### Search cost per decision

- **Playout cap randomization** (from KataGo):
  - Most decisions get a small search budget and are not recorded as policy targets.
  - A random fraction of decisions get the full budget, and only those are recorded.
  - It is the largest single self-play efficiency lever reported in the AlphaZero-family
    literature.
- **Subtree reuse** between decisions. It needs the hooks from Milestone G and the option keys from
  Milestone F.
- **A network evaluation cache** keyed by `observation_key`. Determinized search revisits the same
  information sets constantly. Keep it as an LRU in each worker, or share it in the inference
  server.
- **A transposition table** keyed by `observation_key`.
- **Shallow horizons**: turn-bounded search that calls the value network at end of turn, with
  view-free `run_until` rollouts.

### Inference

- **Pipelining**: each worker runs enough concurrent games that the CPU keeps advancing some while
  others wait on the GPU. Rule of thumb: K ≥ the worker's decision rate × the batch round-trip
  latency.
- **Small networks first** (DouZero-scale MLPs), with fp16/bf16 and TorchScript or ONNX Runtime in
  the inference server.

### Feature encoding and IPC

Once the engine fixes land, turning observations into tensors is likely to become the next
bottleneck. In Python RL pipelines, encoding routinely costs more than the simulator it serves.

- **Encode once per decision, into preallocated arrays.** Where possible, update only the entities
  that changed since the last decision rather than re-encoding the whole board.
- **Decide where encoding runs.** If engine workers run under PyPy, avoid numpy in the workers: it
  is slow under PyPy. Instead, workers emit compact plain-Python encodings (`array('f')` or bytes)
  and the CPython inference server builds tensors from them.
- **Shared-memory buffers between workers and the inference server** (`multiprocessing.shared_memory`
  holding fixed-size encoded slots), with many requests per message. Don't pickle one observation
  per request through a queue: at tens of thousands of decisions per second, per-message overhead
  alone would cap throughput.

### Orchestration

- **Size for this machine:** 5 engine workers plus the inference server/parent on 6 physical cores.
  Measure whether the extra hyperthreads add anything for this workload.
- **Persistent worker pools.** Starting a 5-process pool and importing the engine takes 0.41–0.55 s:
  negligible once, prohibitive per game. Never start one process per game.
- **Dispatch in chunks of seeds**, not one message per game.
- **Workers write shards directly** rather than pickling results back through the parent.

### Fewer games for the same answer

These are throughput-equivalent, and often the biggest win in evaluation:
- **Sequential testing (SPRT)** in `sim/rate.py`: stop an A-vs-B comparison as soon as it is
  statistically decided, instead of running a fixed N. This is standard practice in chess-engine
  testing, and clear-cut comparisons finish on a fraction of the fixed budget.
- **Paired seeds and duplicate evaluation** (Milestone I) reduce variance, so fewer games resolve
  the same gap.

### More learning per game

These are equivalent to more games per hour, because they get more training signal out of each
game played.

- **Warm-start from behaviour cloning.** Initialize the network from `HeuristicBot` imitation
  (Milestone J) instead of random weights, so self-play skips the near-random phase. Measured: games
  at heuristic-level play take 19.9 turns and 151 decisions, against 55.3 turns and 223 decisions at
  random-level play. That is ~1.5× fewer decisions and ~2.8× fewer turns per game from day one,
  before any learning.
- **Mirror augmentation: a free 2× on training data.**
  - Why it is valid: no card text in the pool mentions "left" or "right", and flank status is
    symmetric (the first or last creature, `zones.py:223`). So reversing both battlelines, and
    swapping `CHOOSE_FLANK` labels, gives a position with the same value and the correspondingly
    mirrored policy.
  - Caveat: four engine sites always place a creature on the left flank:
    - Overlord Greking and Collar of Subordination taking control (`dis.py:479`, `dis.py:564`);
    - Harland Mindlock's control revert and Spangler Box's return (`game.py:1462`,
      `game.py:1478`). The docstring on `_revert_temp_control` records this as a deliberate scope
      trim: a real flank choice would need `leave_play` to become a generator.
  - In games containing those four cards, the engine's dynamics are not exactly mirror-symmetric.
    **Fignor and Igor contain none of the four**, so mirroring is exact for Phase 1.1. For larger
    pools, either skip augmentation for games involving them, or give those sites a real flank
    choice.
  - Re-check the "left/right" assumption as each new card lands.
- **Reanalyse** (from MuZero): periodically re-run search with the latest network on stored
  positions, refreshing their policy and value targets instead of generating new games. Stored
  positions are just replay records, so it costs no new infrastructure. It is only valid within one
  engine version.

---

## Risks

- **Fork cost.** Replay forking is linear in game length, so full-game ISMCTS at high simulation
  counts is expensive on replay alone. Milestone D fixes the *interface*. Milestone E offers E1, a
  flat 0.69 ms process fork in WSL, and E2, a snapshot copy that also works on Windows. Milestone J's
  search curve decides which, if either, gets switched on.
- **Snapshot maintenance.** A hand-written `copy()` must track every mutable field as Phase 3 cards
  add state. The equivalence fuzz in Milestone E is the only thing standing between that and silent
  state corruption, so it runs in the normal suite, not occasionally.
- **The 141-call-site audit** in Milestone B is the largest mechanical chunk here, and it cannot be
  deferred — every learned agent is blind without it.
- **Keyed randomness** invalidates every existing seeded record and seed-pinned test, once. This is
  accepted. It lands with the version stamp and before the golden corpus is recorded, so the break
  happens exactly once.
- **Records are only valid within an engine version.** Any later rules change (a card fix, Phase 3
  cards) can make older records replay differently; the version stamp refuses them rather than
  replaying them wrong. Training data stored as replay records must be regenerated after rules
  changes. This is the deciding factor in Milestone J's storage trade-off.
- **Observation allocation on the hot path.** `build_view` already rebuilds every list per call;
  Milestone C must be benchmarked against `tools/bench_engine.py`, not merely written.
- **Log growth.** The three rules functions scan the whole log per call, and forks copy or replay it.
  This is fine at 157 events; Milestone L's counters remove the scan.
- **Optimizations that change rules behaviour.** Caching effect-derived flags is exactly where a
  missed invalidation silently changes which actions are legal. The golden seed corpus in Milestone
  L is the guard, and it must run on every performance change.

## Sequencing

| Stage | Milestones | Unlocks |
|---|---|---|
| Foundation | 0, **A** | ids, hashing, versioning — everything below depends on these |
| Cheap screens | **B, C, F, K** + J's screens | behaviour cloning, value regression, observation-only agents |
| Search | **D, G** | ISMCTS, within-turn search, belief sampling, budgets, hooks |
| Scale | **H, I** | parallel self-play, inference server, rated evaluation, GUI play |
| Speed (options) | **E** | E1 process fork in WSL (small build); E2 snapshot copy (larger build, any platform) — switched on when the search curve says search pays |
| Throughput | **L** | golden corpus and engine hot-path items any time after A; the platform choice (WSL vs Windows) before H; search, inference and encoding items alongside G/H; PyPy once H exists |

The minimum path to the cheap screens is **0 → A → B → C → F → K**. None of it needs forking,
determinization, parallelism or a line of search code.

---

## Options register

Every switch this plan makes available, what decides whether to use it, and where it starts. Each
one is designed so that turning it on or off needs no change to any agent. "Measure" means the
decision waits on a benchmark or diagnostic named in the plan.

### Branching

| Option | Gives | Decided by | Default |
|---|---|---|---|
| Replay fork (D) | Exact fork at any decision, on any platform | — | **on** (always available) |
| E1: process fork (WSL) | 0.69 ms flat, exact at any decision | Search curve (J); E1 vs replay with realistic search children | off |
| E2: snapshot copy | Fast in-process branch-and-hold, Windows included | ≥3× faster than replay at the median decision | off |
| Determinization mode: own deck / opponent private / all | Chance-only search vs full hidden-information search | Hidden-info diagnostic (J) | per agent |

### Engine and interpreter

| Option | Gives | Decided by | Default |
|---|---|---|---|
| Engine hot-path fixes (L 1–7) | Removes the measured hotspots | Golden corpus stays byte-identical | **build** |
| Lazy observations | Up to +35% decisions/sec | — | **on** |
| PyPy engine workers | Unmeasured; commonly several-fold | Benchmark under PyPy in WSL; 3.11-syntax check | off |
| mypyc / Cython hot modules | Unmeasured | Only if PyPy disappoints | off |
| Free-threaded CPython | Shared trees and caches without IPC | Requires leaving 3.12 | off (future) |

### Parallelism and inference

| Option | Gives | Decided by | Default |
|---|---|---|---|
| P worker processes × K concurrent games | CPU parallelism plus inference batching | Benchmark; K ≥ decision rate × batch latency | P = 5, K tuned |
| In-process inference vs inference server | GPU batching across processes | GPU utilization | in-process for BC/eval, server for self-play |
| Shared-memory IPC | No per-message pickling | Worker↔server throughput | on with the server |
| Root-parallel search | Several cores on one decision | Search curve; wall-clock per move | off |
| Evaluation cache / transposition table | Fewer network evaluations | Cache hit rate | on for search agents |
| GUI inference: in-process CPU vs remote to WSL | Trained agents playable in the GUI | Preference | either |

### Self-play and training

| Option | Gives | Decided by | Default |
|---|---|---|---|
| Forced-decision auto-resolve | 11.7% fewer agent calls | — | **on** |
| Training `max_turns` cap | Bounds long early games | Game-length distribution | on early in training |
| Resignation (10% of games exempt) | Shorter self-play games | False-resignation rate | off until the value net is calibrated |
| Playout cap randomization | Cheaper search per decision | Policy-target quality vs speed | on for search agents |
| Subtree reuse | Less search per decision | — | on for search agents |
| Behaviour-cloning warm start | Skips the near-random phase | BC screen results | on |
| Mirror augmentation | 2× training data | Exact for Fignor/Igor; re-check per pool | on for Phase 1.1 |
| Reanalyse | Fresh targets without new games | Learner vs actor balance | off |
| Storage: replay records vs encoded tensors | Compact and always current vs fast to load | Encoder stability; rules changes | both supported |
| Privileged hidden-state labels | Belief networks, oracle-guided value | Hidden-info diagnostic | always written, optionally used |

### Agents and decisions

| Option | Gives | Decided by | Default |
|---|---|---|---|
| Multi-select: enumerate / sequential / top-k | Action representation for `CHOOSE_CARDS` and `ORDER_EFFECTS` | BC screen; pool growth | enumerate (≤379 today) |
| Policy head: fixed vocabulary vs option embedding | Both are served by Milestone F | BC screen | both available |
| Privilege: observation / search / privileged | Enforced capabilities | Registry entry | observation |
| Budget: simulations vs wall-clock | Compute-normalized comparisons | Experiment design | per run |

### Evaluation, formats and decks

| Option | Gives | Decided by | Default |
|---|---|---|---|
| Paired seeds and duplicate evaluation | Lower variance | — | **on** |
| SPRT early stopping | Fewer evaluation games | — | on for development comparisons; fixed N for reported results |
| Position suites | Skill-specific tests | — | available |
| Formats: Archon / Reversal / Adaptive | Match play, bidding, deck swaps | Thesis scope | **Archon** |
| Deck sources: presets / random legal / alliance | Generalization beyond Fignor/Igor | Phase 2/3 scope | presets |

### Platform

| Option | Gives | Decided by | Default |
|---|---|---|---|
| One Windows checkout run via `/mnt/c` vs a second clone in WSL | Simplicity vs faster git and imports in WSL | Latency in practice | one checkout |
