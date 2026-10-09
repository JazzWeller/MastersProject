# Agent Observation Plan: everything a player may know, losslessly, into the network

**Status (2026-10-09): O0, Part R and O1–O9 done; O10's BC ladder half run**
(details under "Progress" below). Built before that:
- O10's actor benchmark, with its v1 baseline;
- O8's persistent inference connection and its faster server result handling;
- O9's faster training: vectorized losses, bf16, a fused optimizer, a compiled trunk and faster batch
  assembly;
- faster search forks (cached boundary snapshots, a cheaper copy) and CUDA graphs in the inference
  server, ahead of Part R.
- O7's attention layer (`ml/layers.py`, now the default) and O7's memory options for larger
  networks: activation checkpointing and gradient accumulation (`micro_batch`), 2026-10-05.

### Progress (2026-10-05)

**O0, done.**
- **The draw leak is closed.** Log events have per-field privacy; a `draw` stays public with its
  count, and its iids are the drawer's alone in `Observation.history` and `PlayerView.log_tail`.
  `tests/test_agent_observation_o0.py` checks contents against what each viewer was entitled to
  when the event happened (`tests/_entitlement.py`). It found 8,490 draw leaks in 20 games before the
  fix and finds none in 300+ games after it; nothing else leaked.
- **The resample test exists** (`test_agent_training_m7.py`).
- **v1 baselines** (`runs/obs-baseline/`, `tools/obs_baseline.py`, Tier 0 network, validation split):
  - value log-loss 0.4827 (constant 0.6931): at turn boundaries 0.4841, mid-turn 0.4781;
  - belief log-loss 0.5264, against 0.5406 for uniform over consistent worlds;
  - encode time per decision: extract 54 µs + encode 34 µs, against 21 µs for the engine's submit;
  - Screen 5 rerun (bf16): held-out CHOOSE_ACTION top-1 0.846 (attributes only) / 0.829 (both) /
    0.724 (ID only), seen ~0.90; Tier 0's 0.846 / 0.829 / 0.715 reproduced.

**Part R.**
- **R0.** `keyforge_ref/` is the engine frozen at `dbc0943` (plus the rules fixes below, applied to
  both). `tools/diff_engines.py` drives two engines in lockstep and compares, after every choice, the
  pending decision, every log event, the RNG counters, the choice record, periodically the whole
  canonical state, and (compiled) that no `NativeFrame` is on the stack. Today's engine equals the
  reference on 100,000 fuzz games (27.4M decisions; three bots; presets, random and alliance decks;
  Archon, Reversal and Adaptive). Coverage (`tools/ref_coverage.py`, 20,000 games): 404 of 413
  suspension points; the other nine get constructed positions in
  `tests/test_part_r_coverage_scenarios.py`, or (`Game._resume`, native-copy only) a written reason.
  Two of the missed points were rules bugs, fixed as their own changes: **Ozmo** looked for a "Mars"
  trait no card has, so it never had a target; **Tireless Crocag's** destroy check counted the
  creatures being destroyed, so it never fired (a bounce/archive/control-change gap is recorded in
  the rulings).
- **R1, departure.** Stored callables are not rewritten as `Ref`s. A closure is already a routine
  id (its code object) plus an environment (its cells), so it is treated as data generically: the
  copier remaps cells through one memo with the frames, the snapshot writes the code path plus the
  cells, and `state_hash` hashes qualname + captured values + defaults. Same goals, no source rewrite
  of 85 card sites, less rules risk. `state_hash` is exact (handlers and cleanups by content); the
  golden corpus's hashes were regenerated, its play unchanged. The census found closures capture
  Cards, frozensets and (Charge) a Player, which the old `_rebind_closure` didn't rebind.
- **R2–R5.** `keyforge/vm.py` is the machine; `tools/compile_engine.py` compiles all 367 generator
  functions (232 pausing, 135 stubs) into `keyforge/compiled/`. Routines are *structured*: they keep
  the source's `if`/`while`/`for`/`try`, and a resumed frame finds its way back through guards;
  calls run inline on the Python stack and a frame is made only when something suspends.
  `NativeFrame` runs anything uncompiled (tested with no routine registered). 300 random programs in
  the subset run identically to native; unsupported constructs are compile errors with their line.
- **R6.** `Game.copy()` works at every decision in compiled execution (frames remapped,
  `keyforge/copying.py`); `fork()` copies there, `fork_by_replay()` is the cross-check.
  `Game.snapshot()`/`restore()` (`keyforge/snapshot.py`): canonical JSON object graph, ~130 KB,
  ~13 ms per round trip; equal states give equal bytes. 1,231 copies and 394 round trips at
  decisions of 10 kinds each equal a replay fork and continue identically.
- **R7.** The compiler emits each suspension point's call site and its remaining operations;
  `Game.resolution_view(viewer)` lists the frames with ability kind, trigger event, source card,
  locals and remaining operations, hidden cards as ("hidden", zone, side). A re-deal now returns σ
  and relabels the cards frames hold, so worlds stay consistent with what is resolving; I1 holds
  with frames (the view is identical in determinized forks, and fails without the relabel).
- **R8.** Compiled execution is the default (`KEYFORGE_EXECUTION=native` runs the generators, kept
  as the oracle). Compiled equals native on 100,000 fuzz games (27.4M decisions, 20 min). The suite
  passes in both modes on CPython 3.12 (Windows), CPython 3.14 and PyPy 3.11 (WSL). The README has an execution section and a
  guide to writing a card effect in the compiled subset. `ENGINE_VERSION` stays 1.1.2: no record
  replays differently.
- **Throughput** (CPython 3.12, Fignor/Igor):

  | | native | compiled |
  |---|---|---|
  | Engine with HeuristicBot, decisions/s | 20.1–20.4k | 19.0–19.6k (94–96%) |
  | Bare replay, 30 games | 81–93 ms | 100–104 ms (~0.83×) |
  | Copy at a boundary / mid-resolution | ~230 µs / replay ~1.5 ms | ~260 µs / ~260 µs |
  | Search sims/s at 100 sims, within-turn / full-game | 2,103 / 1,397 | 1,942 / 1,304 |
  | The same, native before Part R | 1,806–1,842 / 1,250–1,297 | |

  Search got 16–17% faster in both modes from dropping a type object's back reference to its card
  and releasing finished worlds (`Game.release()`), which lets reference counting free them instead of
  the cyclic collector. Compiled search beats native search as it was before Part R; it is ~92% of
  native search today.

**O1, done** (2026-10-06).
- **The journal** (`keyforge/journal.py`): every zone reports its moves; entries are as specified,
  plus `reveal`, `search_reveal` and `reveal_hand`/`unreveal_hand` notes. `cause` is filled by
  `Game._caused`, which every ability call goes through (the resolution view reads the ability kind
  off it). Dealt cards come from a hidden `setup` zone. A peek needs no op of its own: a decision
  offering cards from a zone hidden from the chooser is one, and the projection shows the chooser
  those cards.
- **Decision records** are written raw (the decision and its encoded choice) and turned into data
  only when read, copied or snapshotted. Where each option's card was is taken from the
  projection's own fold of the journal.
- **Log schemas** (`log.EVENT_SCHEMAS`, 62 kinds): `draw.iids` is the drawer's; `archive`,
  `return_to_hand` and `under_card` may be narrowed at their call sites (`SITE_RESTRICTED`). A
  registry test scans the source for every `log.add`.
- **The projection** (`keyforge/projection.py`, `Game.projected(viewer)`), incremental, with codes
  public / private / revealed. `Observation.history` is built from it (`HistoryEntry.stream`:
  log, zone, decision).
- **Acceptance** (`tools/o1_acceptance.py`; 1,000 games per pool: Phase 1.1, Phase 2, Phase 3
  presets and random decks; 4,000 games): completeness at every decision, σ-invariance for both
  viewers, no over-hiding, determinism (replay, and a copy at a random decision played on). Zero
  problems. σ found two real leaks while building: the deal showed deck order, and a decision
  showed the chooser's hidden choice to the other player. Compiled equals native on 10,000 fuzz
  games with journals compared.
- **Throughput.** Bare engine (`tools/bench_engine.py`, RandomBot, no views): 27.5k → 24.4k
  decisions/s, **−11%**, a point over the gate after optimization (pending-leave merging, bulk
  deal, lazy decision records). About 3 of the 11 points are extra cyclic-GC work. Where views are
  built (HeuristicBot) the engine is **5% faster** than before O1 (12.1k → 12.7k): `build_view`
  redacted the whole log every decision to show 20 events; it now reads only the tail.

**O2, done** (2026-10-06).
- `keyforge/knowledge.py`: one `Tracker` per viewer, fed only by the projection and the public
  decklists. Each card has a mask of possible zones (exact = one zone), a provenance group, last
  seen (turn, zone), how it last left sight, and per deck the known top and bottom sequences. Its
  rules: a shown move makes a card exact; a draw from a known end moves that card (and not one known
  at the other end); the pigeonhole rule (a zone holding as many cards as may be in it holds exactly
  those; an empty zone none). Second-order knowledge (`second_order_tracker`) is the same tracker
  fed the part of the viewer's projection the opponent sees. `card_knowledge` exports all of it per
  card, with O3's priors. Trackers are kept on the game and copied with it; a determinization drops
  every viewer's but the determinizing one's.
- **Soundness** (`tools/o2_acceptance.py`, 1,000 games per pool): every true zone in its mask and
  every known deck position true, for both viewers and the second-order tracker, at every
  decision: no problems in 4,000 games. 4.6% of the opponent's hidden cards are placed exactly;
  the mean mask is 2.02 zones.
- **Against brute force.** `world_masks` is a DP over the opponent's journal entries: each hidden
  move takes any card that may be in its zone (a tracked card, or an anonymous one while the zone's
  public count allows), constrained by everything the viewer saw (moves, notes, revealed hands,
  known deck ends). At 73,630 positions no mask missed a zone some world allows; 220 (0.3%) were
  wider than the worlds -- the tracker counts per zone, the worlds per card, which O3's group counts
  capture. The plan's five scenarios (a Lights Out bounce, Snudge returning an artifact, an
  exhausted deck reshuffled, a card put on top, a revealed hand later partly archived) are found in
  fuzz games, and there the masks equal the worlds' exactly.
- Building the oracle found two journal issues, both fixed: Vespilon Theorist revealed the card
  it had just drawn in limbo (the note came after its move's entry, completed in place); it now
  reveals the top card and then draws it, so the draw is known. And Chaos Portal's reveal now
  records that it is the top card.

**O3, done** (2026-10-06).
- `keyforge/determinize.py`; `Game.fork_determinized(viewer, rng, ..., sampler=, weights=,
  history=)`; `search.determinization` (config default `chance_exact`, wired through
  `SearchSettings` and every capability). A world is sampled on the source game (where the
  knowledge caches are) and applied to the fork by instance id, unjournaled; cards that stay in a
  zone keep their places, so frames are relabelled role for role. Cards in the opponent's zones
  that they don't own (one of mine archived from play) stay put.
- **`constrained`**: a DP over mask classes (multinomial counts, the deck taking the rest; plans
  cached per position). **`chance_exact`** (`ChanceFilter`): atoms of exchangeable cards with the
  exact joint distribution of their counts in hand / archive / deck / elsewhere (under cards); a
  shuffle merges the atoms certainly in the deck (without it the state blew up to 2.7M; with it
  the largest seen is 28 states). **`belief`**: chance_exact's support, the hand drawn by weight.
  **`uniform`**: the legacy re-deal, now unjournaled too.
- **History in a world**: `drop` (default) cuts the opponent's projection to what the viewer saw
  before the world began. `relabel` relabels the entries the viewer didn't see, accepted only when
  the relabelled journal folds to the world's zones and the opponent's tracker, built from it, is
  sound in the world; it was accepted for 75% of worlds (33,000 of 44,262), the rest fall back to `drop`.
- **Acceptance** (`tools/o3_acceptance.py`): every world from constrained / chance_exact / belief
  satisfies every mask, count and known deck position, leaves the viewer's projection unchanged,
  and the opponent's tracker inside it is sound (1,000 games, 132,786 worlds). `constrained` passes a
  chi-square test on an enumerable scenario. chance_exact equals a brute-force enumeration of the
  generative process (`tests/_chance_brute.py`) on 8 scenarios to 1e-12, P(next draw) included.
- **Calibration** (400 games of epsilon-greedy HeuristicBots, as in the BC corpus, 1.7M
  card-positions): P(hand) log-loss **0.498** for chance_exact against **0.548** for the uniform
  prior on the same positions -- below the plan's 0.541, and below O0's belief head (0.526).
- **Cost**: 37–92 µs per chance_exact world, 57–176 µs constrained, against a replay fork of
  3.5–12.7 ms.

**O4, done** (2026-10-06).
- `keyforge/infoset_v2.py` (v1's `infoset.py` untouched): entities in public decklist order, each
  with `card_knowledge` (O2/O3) and, where the viewer can see it, every `Card` and type-object
  attribute (cards as entity pointers, hidden ones as ("hidden", side)), its position and host;
  every player counter and flag; effect tokens (all four kinds; values and conditions evaluated
  now); cleanups, temporary control, redirected hits; resolution tokens (R7); the turn cap and
  turns left; match tokens.
- `agent/state_registry.py` classifies every attribute of `Game`, `Player`, `Card`, the type
  objects, the effect classes and the zones (encoded / derived / hidden / bookkeeping); the test
  fails on an unclassified attribute or a bookkeeping one off its default.
- **Acceptance** (`tests/test_agent_observation_o4.py`): coverage over fuzz games of every pool;
  the extract byte-identical in every consistent world (~3,500 checked while building, no
  difference); resolution tokens identical after replay, copy and snapshot. Building it found two
  bugs: a copied frame's rebuilt function made the resolution view call a Play ability an
  "effect" (kinds are now matched by code object), and the re-deal mapped hand cards by position
  after shuffling the hand, so frames named different cards.
- The extract is the reference form (~1.1 ms per call, dicts); O5's encoder reads the same state
  directly.

**O5, done** (2026-10-06).
- **v1 frozen**: `agent/features_v1.py`; `agent/features.py` re-exports it, so every v1 import and
  checkpoint path is unchanged (M1's tests pass through it).
- **v2**: `agent/spec_v2.py` (typed token blocks: GLOBAL, ENTITY ×72, EFFECT, RESOLUTION,
  RES_POINTER, RES_OP, CLEANUP, MATCH, OPTION; int and float columns; pointers as entity indices
  with NO_POINTER / HIDDEN_MINE / HIDDEN_THEIRS) and `agent/features_v2.py` (`encode_v2`,
  `encode_v2_for`, `encode_v2_match`). An entity carries the tracker's knowledge (mask as zone-code
  bits, deck position, provenance group as rank and size, last sighting, exit op, what the
  opponent knows) and chance_exact's priors, and, where visible, every attribute the registry
  marks encoded. Nothing is clipped; counts are scaled only.
- **Vocabularies** (`agent/vocab/`, 20 append-only files, `tools/build_vocab_v2.py`): built from
  the card data, the engine's source by AST, its registries, the compiled routines (routines,
  sites, roles, reachable operations) and a fuzz harvest for values made at run time (mode names
  such as Play-resolution's "effect"/"check", trigger-option tags). An unknown entry raises;
  `VOCAB_V2_HASH` is in the v2 stamp.
- **Static meaning, three tables** (`agent/static_v2.py`): attributes (traits from the vocabulary),
  parsed text rows (trigger, verb, amount, scope, conditions; 570 rows), and engine signatures (the
  `steps`/`game` calls, the lasting effects created, the state attributes written, the hooks
  granted to a host, literal amounts; every card with an effect has a non-empty one). The text
  embedding has its loader and `tools/build_text_embedding.py`; the table itself is not built (it
  needs `sentence-transformers` and a model download).
- **A leak found and fixed in the resolution view**: during the mulligan the setup frame holds the
  opponent's deck as a list in build order, so each element's "hidden, hand" / "hidden, deck"
  said which card was in the hand. Hidden cards inside a list or set are now an order-free count
  per (zone, side). The invariance tests could not see it (a world relabels positions to match).
- **Acceptance** (`tests/test_agent_observation_o5.py`, `tools/o5_acceptance.py`): no leak (bytes
  identical in consistent worlds), determinism across processes and hash seeds, every decision
  kind, every deck source and format (match-level decisions included), vocabulary coverage of all
  370 cards, v1 bit-identical. The sweep: 4,000 games (1,000 per pool), 1.75M encodes, 356,360
  worlds, no problem. **Cost**: ~0.85 ms per encode on CPython 3.12 (v1: ~90 µs); PyPy averaged
  3.7 ms in the 10-worker sweep (to be measured on a quiet machine); O8/O10 measure it against a
  simulation.

**O6, done** (2026-10-07).
- `agent/history.py`: `HistoryEncoder(game, viewer)` folds the viewer's projection into event rows
  (kind, actor side, from/to zone, numeric fields, house, flank, visibility, absolute turn, step,
  sequence index; decision rows add kind, intent, source and the chosen/offered options) plus a
  separate (row, role, entity) pointer list, variable length. A hidden identity is "a hidden card
  from zone Z of side S". Rows take their turn from the event, not from when they were folded.
- **All three representations**: full rows, turn tokens (one per half-turn, with every field the
  plan lists) and summaries (per entity and global). Turn tokens and summaries are folded
  incrementally (`_Folds`, carried by `copy`): 217 µs per call where recomputing took ~20 ms.
- **Prefix and suffix**: `freeze()` marks the prefix with a rolling blake2b digest; a world's copy
  appends only its suffix. `history_for(game, viewer)` keeps one encoder per viewer on the game
  (a cache the copier, snapshot and re-deal know about).
- **Acceptance** (`tests/test_agent_observation_o6.py`): history bytes identical in every
  chance_exact world of every pool; prefix + suffix byte-identical to encoding the world from
  scratch, turn tokens and summaries included; the byte form round-trips; hidden identities stay
  hidden. **Cost**: a world used to rebuild the viewer's projection to extend its history (188% of
  a simulation); `fork_determinized` now carries the determinizing viewer's caches
  (`copy(caches=viewer)`), and the suffix costs ~26% of a simulation (loaded machine; O10 measures
  it quietly).
- Left to O9, where the shards are written: one stream per game and seat with a cursor per position,
  and mirroring of event rows' flank fields.

**O7, done** (2026-10-07; attention and throughput 2026-10-08).
- `ml/encode_v2.py`: `collate_v2` pads each token block, the history rows and pointers, turn tokens
  and summaries, with masks; `bucket_by_length`; `static_tables_tensor` (the three static tables).
- `ml/model_v2.py`, `KeyForgeNetV2` (v1's `ml/model.py` untouched): one input projection per token
  type (`ColumnEmbed`: ids, bit fields, pointers, numbers) plus a type embedding; pointer enrichment
  (role embedding + the card's identity and static rows) for entities, effects, resolution frames,
  history rows and options; sinusoidal time features (no table, no cap). `history_arch`: `joint`,
  `stream` (block-causal), `turn_tokens`, `summary`, `none`; `options_in_trunk` either way. Heads:
  policy, value, top-k, Q, multi-select, sequential, belief v2 (O3's prior logits plus a learned
  difference, and P(next draw)), oracle over hand/archive/deck, and the auxiliary heads.
- `ml/layers.py`, `AttnSpec`: v2 never builds a [B, T, T] mask. Padding is a key mask [B, 1, 1, T],
  and `stream`'s history is a second, causal SDPA call over the history rows alone. The first version
  built the dense mask: at batch 512 and 1,345 tokens it "allocated" 21 GiB (spilling past the 8 GiB
  card) and ran at ~170 evaluations/s. The rewrite equals the dense mask (tested). v1's path is
  unchanged.
- Launch overhead: the eager forward issued 613 kernels for ~2 ms of GPU work. `ColumnEmbed` is
  vectorized across columns (one shared id table per block, one bit unpack, all pointer columns
  gathered at once; 613 -> 346 kernels, 9.9 -> 5.6 ms at batch 64), and a card's identity (embedding
  plus static meaning, including its parsed text rows) is computed once per call for the 370-card
  vocabulary and gathered (bit-identical; batch 512: 20.0 -> 15.7 ms). `subset_size` is continuous
  (it was an embedding capped at 63).
- **Acceptance** (`tests/test_agent_observation_o7.py`, WSL): shapes for every architecture, both
  option placements and every decision kind met in fuzz games; padding never changes an output;
  the stamp carries feature version 2 and the vocabulary hash.
- **Throughput** (`tools/bench_model_v2.py`; network only, eager, bf16; 691 fuzz positions with their
  histories, mean 831 rows, max 3,357; training = a policy + value + belief step, AdamW; — = out of
  memory on the 8 GiB card):

  | `d_model`/layers, architecture | Tokens | Inference/s, batch 64 / 512 | Training/s, batch 64 / 512 |
  |---|---|---|---|
  | 128/4 (1.9M), `none` | 107 | 9.8k / 31.9k | 3.3k / 10.5k |
  | 128/4, `summary` | 107 | 9.3k / 31.6k | 2.9k / 10.3k |
  | 128/4, `turn_tokens` | 150–184 | 8.7k / 19.8k | 3.1k / 6.4k |
  | 128/4, `joint` | 909–1,345 | 2.8k / 1.7k | 788 / — |
  | 128/4, `stream` (whole sequence) | 909–1,345 | 3.6k / 2.4k | 1.0k / — |
  | 256/8 (9.2M), `none` | 107 | 7.9k / 10.1k | 2.2k / 3.2k |
  | 256/8, `summary` | 107 | 7.5k / 10.1k | 2.3k / 3.2k |
  | 256/8, `turn_tokens` | 150–184 | 7.2k / 6.2k | 2.0k / — |
  | 256/8, `joint` | 909–1,345 | 941 / — | 116 / — |
  | 256/8, `stream` (whole sequence) | 909–1,345 | 1.1k / — | 321 / — |

  The full-history architectures train only at small batches without the memory options (O7's
  checkpointing and micro-batching), and are ~10x the cost of `none` per evaluation when the
  whole history is sent. `stream` with its prefix cached is O8's.

**O8, done** (2026-10-08).
- **Request protocol v2** (`agent/agents/requests.py`): `RequestV2` carries an `EncodedV2`, the turn,
  and its history as a `HistoryRef` (the prefix's digest, which starts from both decklists, plus the
  request's own suffix rows), whole `HistoryRows` bytes (the naive path, kept as the reference), or a
  `HistoryFolded` (turn tokens and summaries as flat arrays, built by the workers). A search freezes
  its prefix on the true game before forking (`capability.history()`, `Search` calls the
  evaluator's `prepare`); every world carries it, so a leaf sends only what its world added. The
  prefix's rows go out once per search; a server that lost them answers `PREFIX_MISSING` and the
  client resends.
- **The server** (`ml/infer_v2.py`, `TorchModelV2`): an LRU of prefixes bounded by bytes; for
  `stream`, each layer's keys and values computed once per prefix (`KeyForgeNetV2.history_prefix`),
  so a leaf runs only its state tokens and suffix (`encode_state(batch, past)`). `stream` needs
  `history_time: absolute` (a row's features must not change during a search: the current turn
  goes on the global token instead); `joint` keeps the relative offset by default. The input
  embedding is compiled (`torch.compile`, dynamic shapes; every padded dimension at least 2 rows and
  duck sizing off, or each new shape mix recompiles for ~20 s).
- **The client** (`agent/search/leaf_v2.py`, leaf `student_v2`): encodes v2, sends prefix
  references, and caches answers under (encoding, head, arguments, the whole history's digest).
  The opponent's history inside a world follows O3's history option and travels whole.
- **Acceptance** (`tests/test_agent_observation_o8.py`, WSL):
  - outputs with and without the prefix cache, through real searches (within-turn and full-game,
    every answer checked against the same request sent whole): `joint` bit-identical, `stream`
    within 1.3e-7 (fp32, CPU: the cached path multiplies differently shaped matrices); the cached
    forward against the whole sequence directly: within 1e-5;
  - prefix hit rate within a search **98.3–98.6%** (the plan's bar: 95%);
  - a lost prefix is resent and the search completes;
  - pipe bytes per request: 29 KB within-turn and 38 KB full-game with the prefix cached (most of
    it the encoding, as int64 bytes; see below) against 63 KB whole, at the fuzz games' history
    lengths.
- **The actor benchmark** (`tools/bench_actor_demand.py --v2`, an untrained 128/4 network; 8 workers x
  16 within-turn games against HeuristicBot, 60 s warmup, 150 s measured; the server now reports
  its own batch timing over the measured window):

  | | Searches/s | Evaluations/s | Waiting on inference | Server: requests per batch, ms in the model | Evaluation cache hits |
  |---|---|---|---|---|---|
  | v1, Tier 0 (same day) | 74–81 | 3.1–3.4k | 8% | 58, 5.1 ms | 37% |
  | v2 `none`, first version | 37.8 | 1.7k | 24% | 78, 20 ms | 21% |
  | v2 `none` | **40.9** | 1.9k | **11%** | 66, 14 ms | 22% |
  | v2 `stream`, first version | 32.1 | 1.6k | 31% | | 19% |
  | v2 `stream`, prefix cached | **34.4** | 1.7k | **18%** | 71, 25 ms | 20% |
  | v2 `summary`, first version | 24.4 | 1.2k | 15% | | 20% |
  | v2 `joint`, first version | 29.5 | 1.4k | 36% | | 20% |

  - **Waiting falls below v1's 25–28% baseline** (the plan's bar) for `none` and `stream`.
  - What made the difference was not the network but the pipe. The live server spent half of
    each batch's model time with its connection threads holding the GIL to rebuild nine pickled
    blocks per request. An `EncodedV2` now pickles as one bytes payload: 54 requests pickle in
    2.0 ms instead of 8.8 and unpickle in 0.45 ms instead of 5.5. Narrowing the ints to int16 halved
    the bytes but converting each value back cost more than sending it, so the ints stay int64.
    Compiling the server's input embedding made a call 2.5–3.4x faster in isolation and changed
    nothing live.
  - **What remains is the workers' own work.** A v2 encode of a fresh world costs ~0.65 ms on
    CPython 3.14 (v1's extract + encode: ~0.08 ms), about twice per simulation; folding turn tokens
    and summaries adds ~0.4 ms per request. PyPy is no help here: v2 encodes are 2–6x slower than on
    CPython and the engine ~4x slower. So v2 actors run at about half v1's searches per second;
    O10 weighs that against what v2 buys.
  - The evaluation cache hits less often under v2 (20–22% against 37%): the encoding is lossless, so
    fewer leaves are identical, and the history digest is part of the key.

**O9, done** (2026-10-09).
- **Data.** Tier 0's 30,000 games regenerated with today's engine (1.1.2, compiled; the same config and
  seeds), so v1 and v2 see the same games: 7,463,969 records each, v1's value-only records from the
  other seat included. `ml/dataset_v2.py` (`DATASET_VERSION` 4): per record the v2 encoding (ints in
  the narrowest type their columns allow, the zone masks and trigger bits in a side array, floats
  float16), the event history as a cursor into each game's per-seat stream (stored once per game),
  the folded turn tokens and summaries, v1's labels, and the privileged labels -- every entity's
  true zone class, the opponent's deck order (their next draw), my next five draws. 12 KB a record,
  89 GB; encoded in 29 minutes with 8 workers.
- **A round trip through the shards equals live encoding** (`tests/test_agent_observation_o9.py`):
  every block, the history to each cursor, turn tokens and summaries (exact, or to float16 where
  stored so), and the privileged labels against the true game.
- **The shuffle had to change.** A window of whole shards puts a game's ~220 positions a few steps
  apart, and inputs that identify a game let the network memorize its outcome while the window
  lasts: with turn tokens the training value MSE fell to 0.16–0.18 while the validation log-loss was
  0.59 (summaries lost v2's policy gain the same way). The **packed corpus** (`pack_corpus_v2`,
  `PackedCorpusV2`) stores each shard as zlib chunks of 16 records (18x smaller: 12 GB), memory-maps
  them (read through once so they sit in the page cache), and trains from segments of 4,096 chunks
  drawn at random across the corpus, decoded on a background thread: a game then contributes at
  most one chunk per segment. Batches load at ~21,700 samples/s; the event-history streams are read
  per chunk only for the architectures that use them. Peak memory ~6 GB. A packed corpus yields the
  same records as the unpacked one (tested).
- **Learner** (`ml/bc_train_v2.py`): v1's losses plus belief v2 (hand / archive / deck over the
  opponent's hidden cards, and their next draw, both against O3's prior) and the oracle over the
  opponent's hand, their archive and my next draws; `bc.history_dropout` (default 0) drops event
  rows in training. `configs/tier0b_bc.json`. Checkpoints carry the v2 stamp
  (`ml.checkpoints.load_model_v2`).
- **The v2 self-play shard format** is the BC format plus v1's search targets per record: the visit
  distribution over options or multi-select candidates, whether the search was full (playout cap),
  the search's value, value-only records for the other seat. Nothing is written in it until
  self-play starts.
- **Tier 0b training speed** (batch 512, bf16, steady state): no history / summaries ~7k samples/s
  (one epoch ~16 min), turn tokens ~4k, `stream` ~550 and `joint` ~290 with activation checkpointing
  (~3.5 h and ~6.5 h an epoch; 5.7 GiB peak). Every rung gets one epoch, so they compare at equal
  data.

**O10, the BC ladder so far** (2026-10-08; one epoch each on the packed corpus, the same validation
games; single runs, so a difference of a point or so may be noise). History rungs are trained on top
of the cheap rungs rather than before them (a departure from the table's order: the expensive
history runs are then done once, on the best non-history inputs).

| Rung | Adds | CHOOSE_ACTION | CHOOSE_HOUSE | CHOOSE_CARDS (enumerate) | Value log-loss | Belief P(hand) (prior) |
|---|---|---|---|---|---|---|
| A0 | v1 (bf16) | 0.909 | 0.942 | 0.863 | **0.4935** | 0.536 (uniform 0.541) |
| A1 | v2, no knowledge, attributes only, options outside | 0.909 | 0.945 | 0.862 | 0.502 | 0.515 (0.516) |
| A2 | + knowledge and O3's priors | 0.987 | 0.936 | 0.968 | 0.504 | 0.506 (0.516) |
| A7 | + all static card meaning | **0.990** | 0.938 | **0.968** | 0.503 | 0.504 (0.516) |
| A8 | + options in the trunk | 0.941 | 0.939 | 0.935 | 0.504 | 0.504 (0.516) |
| A3 | A8 + history summaries | 0.907 | 0.945 | 0.850 | 0.498 | 0.503 (0.516) |

- **Knowledge is what moves imitation**: A1 equals v1, A2 adds 7.8 points of CHOOSE_ACTION and 10.6
  of CHOOSE_CARDS. The belief head beats O3's exact prior from A2 on (0.506 against 0.516).
- **Options in the trunk cost 5 points** (A8 against A7): under the decision rules it stays built and
  switched off, and the remaining history rungs are rebased on A7 (A3 to be rerun there).
- **Value has not improved on v1** (0.4935); the best v2 value so far is with summaries (0.498).
- Still to run: A3 on A7, A4 (turn tokens), A6 (`stream`), A5 (`joint`), Screen 5 with v2
  (`tools/screen5_v2.py`), the search rungs (`tools/eval_v2.py`), G3 again, the actor benchmark with
  trained networks, and the linear probes (`tools/probe_v2.py`).

Written 2026-09-28. Amended 2026-09-30:
- the engine refactor is scheduled (Part R);
- self-play is held until everything here is complete;
- the inventory is corrected against the dead-code cleanup (committed as `f90a9e0` and merged to
  master as `2c4f47f` on 2026-09-30).

Every number below was measured on this engine on 2026-09-28 or 2026-09-30, unless it is marked as
an estimate.

## Context

`Code/AGENT_INTERFACE_PLAN.md` built the observation seams, and `Code/AGENT_TRAINING_PLAN.md` built
the network, search and training on top of them. Today the network's input (training plan M1,
`FEATURE_VERSION` 1) is a snapshot of the current state: `keyforge/infoset.py` extracts it and
`agent/features.py` encodes it. This plan replaces that input with one that carries **everything a
player is entitled to know**:
- the full current state;
- the public history of the game;
- the player's own private history;
- every hard fact about hidden cards that follows from all of those;
- what the opponent knows about the player's cards.

Nothing is capped or compressed unless a measurement shows it has to be.

It exists because of three findings:

1. **M6: the belief head barely beats uniform** (log-loss 0.526 vs 0.541) because the input
   carries no history, and inferring a hand is mostly a matter of history.
2. **An audit compared what the engine knows with what the encoder writes.** It found every
   loss listed under "Inventory" below, including public facts that the search's determinization
   contradicts.
3. **The same audit found a leak in `Observation.history`** (Inventory A). It has to be closed
   before any history reaches a network.

The plan has two parts:
- **Part O** (O0–O11) is the network's input.
- **Part R** (R0–R8) is the engine refactor that replaces suspended Python generators with an
  explicit, serializable resolution stack. Without it, part of the game state — what is still left
  to resolve — can be neither shown to the network nor copied cheaply.

**Read first:** training plan M1, M2 and M6 and its "Findings and open items"; interface plan C
and D.

### Decisions (2026-09-28, amended 2026-09-30)

- **Entitlement is the only content limit.** The network receives everything the deciding player
  could legally know at that moment, and nothing else.
  - **In:** the current public state; the full public history, including every decision's public
    part; the player's own private information and history; what was revealed to them, remembered
    after the reveal ends; every constraint on hidden cards that follows from all of that; and what
    the opponent knows about the player's cards.
  - **Out:** hidden card identities, deck order, RNG state, the opponent's offered options, and
    anything in the future.
  - The privileged labels that train the auxiliary heads are unchanged, and they are never inputs.
- **No silent caps.** Every size limit is a config value that defaults to "none". Where the
  implementation genuinely needs a bound, exceeding it raises an error and never truncates, as
  `enumerate_cap` already does.
- **Lossless first, compressed by measurement.** Use vocabularies instead of hash buckets, tokens
  instead of counts, and full event histories instead of summaries. Compressed forms are built as
  options and adopted only under O10's rules: summaries, per-turn tokens, memory tokens, caps.
- **"Necessary" is defined before anything is measured** (O10). A restriction is adopted only when
  the unrestricted version does one of three things:
  - would starve the self-play actors: the GPU is saturated while actors wait, measured with O10's
    actor-shaped benchmark rather than a self-play run;
  - exceeds GPU memory at the learner's batch size;
  - is measurably worse (for example, it memorizes).

  Nothing is capped on a guess.
- **Knowledge is derived, never read.** Everything the encoder says about hidden cards comes from a
  per-viewer knowledge tracker. That tracker consumes only that viewer's projected event stream.
  Freedom from leaks then holds by construction, and it is tested by invariance under re-dealing, as
  M1 does today.
- **The search must agree with the network.** Every determinized world must satisfy every fact the
  network is told (O3). A world that puts a publicly bounced creature back in the deck contradicts
  the network's own input.
- **v1 stays reproducible.** `FEATURE_VERSION` 1 checkpoints, data and encoder stay runnable as the
  baseline arm, and v2 is `FEATURE_VERSION` 2. "What does seeing more buy?" is itself a thesis
  result (O10).
- **The engine refactor is scheduled, not gated** (decided 2026-09-30). Part R replaces the
  engine's suspended Python generators with an explicit resolution stack. This supersedes the scope
  boundary in both earlier plans ("card effects stay generators").
- **Option-consequence lookahead features stay gated** (O11). They are specified in full and built
  only if O10 shows a gap they would close.
- **No self-play until everything here is complete** (decided 2026-09-30). That means Part O, Part
  R and O10's decisions. After that, the training plan's smoke run comes first, then its Tier 2 and 3
  runs. There is no v1 self-play run in the meantime.
  - The aborted `wt-s0` run stays as recorded in the training plan.
  - Every measurement this plan needs is taken from behaviour cloning, search evaluation games and
    benchmarks, never from a self-play run (O10).
- **No rules change.** The rules hash is unaffected. The new engine bookkeeping is excluded from
  `state_hash`, and replays stay valid. Part R changes how the engine runs, not what it does. Every
  step is checked trace-for-trace against the engine as it is today.
- **The rulebook decides visibility.** Which zones and operations are visible to whom follows the
  official Master Rulebook, as the engine already encodes it. Any case it leaves open goes to the
  user, not to a guess.

### Measured sizes (40 HeuristicBot Fignor/Igor games)

| Quantity | min / median / max |
|---|---|
| Log events per game | 152 / 230 / 355 |
| Events visible to one seat | 148 / 228 / 352 |
| Turns | 12 / 19 / 30 |
| Decisions | 91 / 154 / 251 |

The most frequent events per game are:

| Event | Per game |
|---|---|
| `play_card` | 52.5 |
| `draw` | 21.4 |
| `turn_start` | 20.0 |
| `choose_house` | 19.0 |
| `damage` | 16.7 |
| `destroyed` | 14.9 |
| `shortfall` | 7.5 |
| `use_action` | 7.5 |
| `reap` | 6.9 |
| `fight` | 5.8 |
| `duration_effect` | 4.5 |

So at Phase 1.1 an uncapped event history is a few hundred tokens. That is large next to the 73
tokens the trunk sees today, but not out of reach; O7 estimates the cost.

---

## Inventory: what the network cannot see today

Each row says where the information is lost; the milestone that restores it is in brackets.

### A. A leak (fixed first)

| What | Where | Evidence |
|---|---|---|
| Drawn cards' instance ids are logged as visible to both players [O0, O1] | `keyforge/effects/steps.py:123`: the `draw` event has no `visible_to` | Seed 3, after 60 decisions of HeuristicBot play: `draw` events in player 1's `Observation.history` name 7 of the 8 cards in player 2's hand. |

The Milestone C fuzz test (`test_history_never_exceeds_what_the_viewer_is_entitled_to`) misses this
because it compares event *kinds*, never their contents. The leak does not affect current training,
since `InfoSet` reads no history. It does affect `Observation.history` and `PlayerView.log_tail`.
The GUI reads the raw log and is unaffected.

### B. Hard knowledge about hidden cards, dropped [O1, O2, O3]

**Cards publicly returned to a hand stay known there.**
- Examples in the current pool: Lights Out, Snudge, Bad Penny, and Help from Future Self (which
  reveals the Timetraveller it takes).
- Lost in two places:
  - `infoset.py:415` only locates an opponent's hand cards during a full-hand reveal, so the
    returned cards show as `opp_unseen`;
  - `game.py:437` (`_resample_hidden_pool`) shuffles them back into the deck in every sampled world.

**Transitions hidden more than the rules require.**
- Bear Flute logs cards returned from the public discard as visible to their owner only
  (`effects/named/untamed.py:354`). Its comment says it "errs toward not leaking", because it
  doesn't track which zone each card came from.
- `steps.shuffle_into_deck` hid cards the same way, but it had no callers, and the dead-code cleanup
  (`f90a9e0`) deleted it.

**Public transitions that aren't logged at all.**
- Help from Future Self shuffles the discard into the deck with no event (`logos.py:33-35`).
- Bear Flute does the same (`untamed.py:355-356`).

**Known deck positions.**
- A card put on top of a deck from the discard or from play is the next draw (`steps.py:182`,
  `brobnar.py:45`, `untamed.py:406`).
- `InfoSet` has no deck positions.

**Known deck composition after a reshuffle.**
- When a deck runs out, it is re-formed from the public discard.
- The uniform resample ignores that.

**Memory of reveals.**
- `reveal_hand` logs no card contents.
- `hand_revealed_to` is cleared at the end of the turn (`game.py:1073`), so the knowledge is
  forgotten.
- There are no reveal cards in Fignor or Igor, but several in the CotA pool.

**What the opponent knows about my cards.**
- `KNOWN_TO_THEM` is set only during a full-hand reveal.

### C. History, dropped entirely [O1, O6]

None of the roughly 230 events per game reaches the network. Of the decision stream, only
mulligan and archive decisions are recorded at all. So the network can't see:
- which house each player chose each turn;
- what was played, discarded or held back, and when;
- which public targets the opponent picked, and which ones they passed over;
- which optional abilities were declined.

### D. Current state, dropped or lossy [O4]

**Lasting effects** (`infoset.py:455`, `features.py:372`).
- The engine has four kinds (`ActiveEffectList`): duration, trigger, instead and modifier.
- Only duration effects reach the encoder, as +0.5 in a slot for their variable. The op, value,
  remaining duration, whether it is active now, its condition, its controller and its source are
  all dropped.
- The other three kinds don't reach it at all; they can only be inferred when their source card is
  in play.

**Player state never read** (from introspecting `Player` against `infoset.py`; each one is
checked to be both written and read by the engine):
- `hand_plays_this_turn`
- `used_this_turn` (the rule of six)
- `discarded_untamed_this_turn` (Giant Sloth)
- `next_entry_ready`
- `next_mars_creature_ready`
- `ExtraHousePlayable` (Witch of the Wilds)
- `CardsPlayed`

**Card state never read** (from introspecting `Card`, checked the same way):
- `CanBeUsed`
- `destined_zone`
- `archive_return_to_owner`
- `granted_action`
- `extra_triggers`
- `redirect_fight_damage_to`
- `purged_by`

Three fields that introspection also turned up carry no information, so they are left out of this
list:
- `archive_choice_made_this_turn` was never read, and the cleanup (`f90a9e0`) deleted it.
- `IgnoreElusive` is never set to true.
- `_ember_imp_effect` only holds the effect object so it can be unregistered; the effect itself is
  covered by effect tokens.

The coverage registry (O4) records the two that remain, with these reasons.

**Game state never read** (from introspecting `Game`):
- `_temp_control` (temporary control, and when it reverts)
- `_end_of_turn_cleanups`
- `_redirected_hits`

**The host of an upgrade or under-card** (`infoset.py:328`).
- It copies its host's flank and index, not which battleline the host is on.
- That's ambiguous when both sides have a creature at that index.

**Resolution context** [Part R, O4].
- The network gets the card that asked the decision and its intent tag.
- Pending triggers and the rest of the resolving ability are invisible. They exist only as the
  local state of suspended Python generators in `game.py` and `effects/`.

**The turn cap.**
- `max_turns` is not an input, although the value target maps a capped game to a draw.
- Training caps games at 60 turns while evaluation caps them at 200.

### E. Lossy encodings and caps [O5]

| What | Today |
|---|---|
| Traits | crc32-hashed into 32 buckets |
| Mode names, trigger events | crc32-hashed into 8 buckets |
| Effect variables | 24 named slots plus a shared "other" slot |
| `min_n`, `max_n` | clipped at 40 |
| `ORDER_EFFECTS` positions | positions beyond 8 share one embedding |
| Card text | 24 yes/no verb flags; amounts, targets and conditions lost |
| Pointers / in-play index | int8 / uint8: fine at 72 entities, a limit for anything else that points |
| Belief head | hand vs not-hand only; the archive is merged with the deck |

### F. Match context [O4]

For previous games of a Reversal or Adaptive match, only the score, game number, starting chains
and the deck-swap flag are recorded.

---

## Invariants (all tested, all permanent)

| # | Invariant | How it is tested |
|---|---|---|
| **I1** | **Entitlement.** The v2 encoding for viewer *v* is a function of the decklists, the game configuration, the public state, and *v*'s projected event stream (O1). | Encodings stay byte-identical across every consistent re-deal (O3), and across every permutation σ of hidden identities applied to the privileged journal and log (O1). |
| **I2** | **Soundness.** Every fact the knowledge tracker asserts is true in the true game, at every decision. | Fuzzing. |
| **I3** | **Consistency.** Every determinized world satisfies every fact the searcher's tracker asserts, and every public fact. | Fuzzing. |
| **I4** | **Coverage.** Every attribute of every engine state object is classified as encoded, derived, hidden (with the rule that hides it) or bookkeeping (with the reason). | An unclassified attribute fails the build. |
| **I5** | **No silent truncation.** Every limit is a config value. | Exceeding an implementation bound raises an error. |
| **I6** | **Determinism.** | Output is byte-identical across processes, hash seeds and platforms (M1's test, extended). |
| **I7** | **v1 reproducibility.** | v1 checkpoints evaluate bit-identically on the v1 path. |
| **I8** | **Execution equivalence.** The compiled engine (Part R) and native generator execution of the same source behave identically. | After every decision, both engines must show the same pending decision, log, journal, RNG counters and shared state hash. Checked by the permanent differential test (R0, R8) on every corpus, in CI, in both modes. |

---

## Milestone O0: prerequisites, the leak, baselines

1. **Close the `draw` leak.**
   - Redact drawn instance ids for the non-owner in `Observation.history` and
     `PlayerView.log_tail`. The raw `game.log` stays complete for the rules and the GUI.
   - Rewrite the Milestone C fuzz test to compare **contents**. For every history entry, every card
     identity present must be one the viewer was entitled to at the time of the event. That is
     checked against privileged state recorded during the fuzz.
   - The new test must fail on today's code and pass after the fix.
2. **The resample configuration: already done.** Commit `76b000a` (2026-09-28) set every shipped
   config and `DEFAULTS` to `all`. A test holds the configs to it, and the learner refuses any
   other value. All that's left is to verify the test exists when O0 starts.
3. **Record the v1 baselines**, on fixed seeds and under `runs/obs-baseline/`. Tier 0 already
   provides BC top-1 per decision kind. The rest are new:
   - value log-loss by turn bucket, and for mid-turn positions separately from turn boundaries;
   - belief log-loss against "uniform over consistent worlds";
   - generalization to held-out cards (a Screen 5 rerun);
   - encode time per decision.
4. **The turn cap needs no self-play measurement.** The turn-cap input (O4) is included whatever
   the numbers say. How often games reach a cap is reported from O10's evaluation games.
5. **Start from a clean working tree: done.** The dead-code cleanup that had sat uncommitted since
   2026-09-28 was verified and committed on 2026-09-30 as `f90a9e0`, and merged to master as
   `2c4f47f`. It removed `current_queue.py`, `bots/torch_model.py`, the old `sim/` actor pool, and
   unused enums and helpers. Part R's reference engine (R0) is frozen from a known commit.

**Acceptance.** The content-level leak test goes from failing to passing, and every baseline is
recorded with its config hash.

## Milestone O1: the zone journal and the per-viewer projection

**Files:** `keyforge/journal.py`, `keyforge/projection.py`, `keyforge/zones.py`,
`keyforge/game.py`. `Observation` switches to the projection.

**The journal.** Every zone learns its owner and kind: `hand`, `deck`, `discard`, `archive`,
`purged`, `battleline`, `artifacts`, `attached`, `under` or `limbo`. Every mutation appends one
entry to `game.journal`. That covers:
- `add`, `remove`, `push`, `pop` and `take_all`;
- `draw_top`, `put_on_top`, `put_on_bottom`, `shuffle_in` and `shuffle`;
- attaching and detaching upgrades;
- placing a card under another;
- entering and leaving limbo.

Each entry is:

    (seq, turn, iid, owner, controller, from_zone, to_zone, op, deck_epoch, public_position, cause)

- `deck_epoch` increments on every shuffle of that deck.
- `public_position` is set for top and bottom placements.
- `cause` is the source card of the frame on top of the resolution stack, if there is one. It is
  filled from R5 onward, once every effect runs as an explicit frame (Part R). Before that it is
  empty.

Operations that move nothing but change knowledge are journal entries too: `reveal` (a card or a
hand shown to a player), `peek` (a player privately looks at cards, such as "look at the top 3"),
and `search_reveal`.

How the journal behaves:
- It is complete and privileged, like `game.log`.
- It is excluded from `state_hash`.
- `Game.copy()` copies it (at any decision, from R6), and replay regenerates it.
- It is shared code, so native and compiled execution both produce it, and the differential test
  (I8) compares it.
- A determinization re-deal is not a game event, so it is not journaled. A world's journal is the
  real one, relabelled by the re-deal's permutation (O3).

**The visibility rule** is one function, `projection.visible(entry, viewer)`:
- The *fact* of every entry is public: its op, zones, owner and count. KeyForge zone sizes are
  public.
- The card's *identity* is visible to *v* if any of these hold:
  - it was visible to *v* in the source zone;
  - it is visible to *v* in the destination zone;
  - the op reveals it to *v*.
- Zone visibility is the engine's existing rulebook encoding:
  - hands and archives: owner only;
  - decks: nobody;
  - discard, purged and play: both players;
  - under a card: owner only.

**Log events** keep their current form for the rules and the GUI. Each gains a field-level
redaction schema: every event kind declares which of its fields are private, and to whom (for
example, `draw.iids` is visible to the owner only). A registry test fails on any event kind that has
no schema.

**Decision events.** Every submitted decision emits one. It carries:
- the player, kind, intent, source card, `affects` and `optional`;
- the chosen submission in public form:
  - each chosen option as (verb, pointer) if its card is visible to the other player at that moment;
  - otherwise as (verb, "hidden, from zone Z");
  - payloads (house, flank, bool, mode, number) exactly as they are.
- if every offered option is public (for example, targets on the board), the whole offered set, so
  that "chose X over Y" is visible.

The number of options offered from a hidden hand is not included.

**The projection.** `Game.projected(viewer)` is an append-only stream per viewer, built
incrementally behind a cursor. It contains:
- journal entries and log events, redacted by the rule and the schemas;
- decision events;
- reveal and peek contents, for the viewer entitled to them.

Every event carries a visibility code:
- *public*;
- *private to me*;
- *mine, revealed to them*.

`Observation.history` is built from this stream.

**Acceptance.**
- **σ-invariance.** Run 1,000 fuzz games per pool: Phase 1.1, Phase 2 and Phase 3 presets, plus
  random decks. Apply a random permutation of each player's hidden cards to the privileged journal
  and log. Each viewer's projection must stay byte-identical.
- **Completeness.** An instrumented fuzz diffs every zone's contents after each engine step against
  the journal. There must be zero unjournaled moves.
- **No over-hiding.** Every entry whose source or destination is public carries its identity to
  both viewers. This catches the Bear Flute and `shuffle_into_deck` cases.
- **Determinism.** Replay and copy produce identical journals, byte-identical across processes.
- **Throughput.** Engine throughput loss is at most 10% on `tools/bench_engine.py`; optimize before
  accepting if it is higher.

## Milestone O2: the knowledge tracker

**File:** `keyforge/knowledge.py`. There is one tracker per viewer. It is fed only by that viewer's
projection, costs O(1) per event, and is copied with the game.

**The viewer's knowledge of each card is one of three kinds:**
- **Exact:** its zone, side and position, where known. That covers cards in play, discard and
  purged; my own hand and archive; and a card known to be in the opponent's hand.
- **Constrained:** a set of hidden zones on one side where it might be. For example, {hand,
  archive} after "a card left their hand face down" while this card was known to be in that hand.
  It also records its **provenance group**: the cards that entered that side's hidden pool at the
  same public event (the initial deck, a reshuffled discard, a card shuffled in by an effect).
- **Deck position:** known top and bottom sequences per deck epoch. These come from public top and
  bottom placements, `reveal_top` and peeks. A shuffle clears them, and draws consume them.

**Also tracked:**
- **Memory of reveals:** the last time each opponent card was seen, and where. That includes cards
  seen in a revealed hand after the reveal has ended.
- **Second-order knowledge:** the same state machine, run over the events the opponent can see
  (public events plus my reveals to them) for my own cards. This gives exactly what they know about
  my hand, archive and deck top. It's computable because every event the opponent sees about my
  cards is one I also see.
- **Counts per hidden zone,** which are public.

**Exported per entity:**
- the kind of knowledge;
- the exact zone, or the mask of possible zones;
- the known deck position;
- the provenance group;
- turns since it was last seen in public;
- how it last left public view (an op code);
- what the opponent knows about it;
- the exact prior probabilities from O3: P(hand), P(archive), P(deck), P(next draw).

**Acceptance.**
- **Soundness fuzz.** Every exact claim is true, and every card's true zone lies inside its set of
  possible zones. This is checked at every decision, for both viewers, over 1,000 games per pool.
- **Constructed scenarios.** Compare the tracker's sets with a brute-force enumeration of the
  consistent worlds in each of these:
  - a Lights Out bounce;
  - an artifact returned by Snudge;
  - an exhausted deck being reshuffled;
  - a card put on top of a deck;
  - a hand reveal after which part of that hand is archived.

## Milestone O3: knowledge-consistent determinization and exact priors

**Files:** `keyforge/determinize.py`; `Game.fork_determinized(viewer, rng, sampler=...)`; the
`search.determinization` config.

| Sampler | Distribution | Role |
|---|---|---|
| `uniform` | today's `_resample_hidden_pool` / `_resample_own_deck` | legacy, diagnostics |
| `constrained` | uniform over assignments consistent with the tracker | the cheapest honest world |
| `chance_exact` | the posterior when draws and shuffles are random and the opponent's choices carry no information | default once built |
| `belief` | `chance_exact`'s support, weighted by the network's belief head | the hidden-information arm |

**`constrained`** keeps known cards fixed, respects each card's mask of possible zones and each
zone's count, and keeps known deck positions fixed. It samples exactly:
1. Cards fall into at most 7 mask classes per side: the non-empty subsets of {hand, archive, deck}.
2. A DP over how many cards of each class go to each zone counts the consistent assignments.
3. A split of counts is sampled from that DP, then cards are assigned uniformly within each class.
4. Unknown deck positions are shuffled around the known ones.

**`chance_exact`** tracks, per side, the joint distribution of how many cards of each provenance
group are in each hidden zone. It updates exactly on every public event:
- a draw removes one card uniformly from the deck (a multivariate hypergeometric step over the
  groups);
- a face-down move from hand to archive takes a uniformly random card from the hand;
- a shuffle merges groups in the deck.

The state is a distribution over count vectors, which is small at these sizes. It gives exact
probabilities for every hidden card, and those are exported to O2 as input features.

**Where `chance_exact` and `constrained` differ:** suppose a known discard is shuffled into a deck
from which some cards had already been drawn. The shuffled-in cards are then *less* likely to be in
the hand than cards from the original deck. `constrained` treats them all the same.

**`belief`** samples sequentially under the same constraints, with weights from the belief head.
This is what `leaf.sample_hand` does today, extended to the archive and the deck. It can
optionally apply importance weights back to `chance_exact`.

**The opponent's private history inside a world.** A world re-deals the opponent's hidden cards,
so their private history in the copied log (their own draws, their face-down archives) no longer
matches their sampled hand. Two options are built:
- **`drop`:** encodings from the opponent's perspective inside worlds leave out their private
  events. Training applies the same dropout so the input distribution matches.
- **`relabel`:** apply the re-deal's permutation σ to their private events. It is accepted only when
  a checker confirms that σ respects the public timeline: no relabelled draw may name a card that
  was public at that moment.

`drop` stays the default until `relabel` passes its checker on the fuzz.

**Worlds and the resolution stack.** Until R6, worlds are built by replay plus a re-deal. From R6
on, they are built by copy plus a re-deal, at any decision: about 4× cheaper at mid-game (0.33 ms
against 1.39 ms, measured). A world's resolution frames are part of the world. The re-deal's σ
relabels any hidden card referenced inside a frame, and the consistency checker validates frames
too (R7).

**Acceptance.**
- Every sample satisfies every constraint (I3) across the fuzz.
- `constrained` is uniform: a χ² test on scenarios small enough to enumerate.
- `chance_exact`'s probabilities equal a brute-force enumeration of the generative process on the
  constructed scenarios.
- `chance_exact` is calibrated on real games: the log-loss of its P(hand) on held-out positions
  (from the BC corpus and O10's evaluation games) is at most the uniform baseline's 0.541. That figure becomes the new baseline the
  belief head must beat.
- Sampling costs at most 10% of a replay fork (about 1.5 ms).

## Milestone O4: the complete current state

**Files:** `keyforge/infoset.py` (the v2 extract), `agent/state_registry.py`, and the
resolution view from R7.

1. **Coverage registry (I4).** A table classifies every attribute of `Game`, `Player`, `Card`,
   `CreatureType` and the artifact types, the four effect classes, and the zones. Its test
   introspects live objects across the fuzz (all pools) and fails on either of two things:
   - an unclassified attribute;
   - an attribute classified as bookkeeping whose value ever differs from its default. This catches
     per-card attributes set dynamically, such as `destined_zone` and `purged_by`.

   The initial contents are the lists under Inventory D.
2. **Effect tokens.** One token per lasting effect, of any kind. Each token holds:
   - its kind: duration, trigger, instead or modifier;
   - its name from a vocabulary: the variable, event or kind;
   - its op;
   - its value, evaluated now. Callable values are evaluated too; the invariance test checks that
     they read no hidden state;
   - its remaining duration in turns, with a flag for infinite;
   - whether it is active now (its condition, evaluated);
   - whether it has a condition at all;
   - the affected side and the controlling side;
   - a pointer to its source entity;
   - the turn it was created, from the projection.
3. **Per-player state:** every counter and flag listed in Inventory D.
4. **Per-card state:**
   - every attribute listed in Inventory D;
   - host pointers for upgrades and under-cards: the index of the host entity, not a copied
     position;
   - temporary control, with the original controller and when it expires;
   - pending end-of-turn cleanups, as tokens (an operation vocabulary plus a target pointer).
5. **Resolution tokens, read from the explicit stack** (R7). There is one token per frame, top to
   bottom. Each token carries:
   - its routine, from a vocabulary;
   - its suspension point (`pc`) and depth;
   - its source card pointer;
   - its ability kind (play, reap, fight, action, omni, destroyed, trigger, or a kernel step);
   - its trigger event;
   - its locals, as (role, value) pairs with cards as pointers;
   - the pending trigger batch, where there is one.

   Each frame also points into the compiler's static table of **operations still reachable** from
   its `pc`: each choice with its intent and `affects`, each primitive step with its literal
   arguments, and loops marked. What remains to resolve is therefore data, not code.

   The view is redacted per viewer by O1's visibility rule: a local naming a card the viewer can't
   see appears as "a hidden card from zone Z". Abilities resolve face up, so the frames themselves
   are public.
6. **The turn cap.** Add `max_turns` (or none) and the number of turns remaining. The value head can
   then learn what the cap means, instead of treating turn 55 as "nearly a draw" when evaluation
   runs to 200.
7. **Match tokens.** One per previous game of the match: winner, keys, turns, chains, whether decks
   were swapped, and who went first. As an option, those games' full projected streams can be added
   as history (O6).

**Acceptance.**
- The coverage test passes on the fuzz.
- Every new field passes I1 (invariance under determinization and under σ). Resolution tokens are
  included.
- Resolution tokens are identical under replay, copy and snapshot round-trips (R6) at random
  decisions of every kind.

## Milestone O5: a lossless encoding, `FEATURE_VERSION` 2

**Files:** `agent/spec.py` (the v2 blocks), `agent/features.py` (v2), `agent/vocab/*.json`, and
`agent/features_v1.py` (v1, frozen).

**Blocks.**
- The blocks are GLOBAL v2, ENTITY v2, EFFECT, RESOLUTION, HISTORY (O6), MATCH and OPTION v2.
  ENTITY v2 has 72 rows and carries the O2 knowledge fields and the O4 card state.
- The variable-length blocks are padded arrays with masks.
- The entity count stays 72, because that is what the game has, not a cap. The masks make a
  different count a data change if a later pool has token creatures.

**Vocabularies replace hashing.** This covers traits, mode names, trigger events, effect variables
and kinds, log event kinds, ability kinds, cleanup operations and zone codes.
- Each vocabulary is an append-only JSON file, versioned like the card vocabulary, and feeds a
  learned embedding.
- An unknown entry raises an error in training and in data generation. A new card must extend the
  vocabulary; nothing is silently mapped to "other".

**Caps removed.**
- `min_n` and `max_n` are no longer clipped.
- `ORDER_EFFECTS` positions and subset sizes use continuous encodings instead of fixed tables.
- Pointers become int16, and in-play indices uint16.

**Static card meaning: three options, all built.**
- **Parsed text.** Each card's text is parsed into ability rows: trigger, verb, amount, target scope
  and condition. Each card gets a variable-length set of rows, pooled by a small set encoder.
- **Engine signatures.** A static analysis (`ast`) of each card's effect functions in
  `effects/named/*.py` records which `steps.*` and `Game` methods they call, with which literal
  arguments. The result is a multi-hot vector plus amounts, and it is exact about what the engine
  actually does.
- **Text embedding.** A pretrained sentence encoder's embedding of the card text, precomputed
  offline into a versioned table. The dependency is optional, and the embedding is used only if the
  table exists.

**v1 is frozen** as `features_v1.py`, and v1 checkpoints load only with it (I7).

**Acceptance.**
- Every M1 test is ported to v2:
  - no leak across 1,000 games;
  - determinism;
  - every decision kind;
  - preset, random and alliance decks;
  - every format.
- The vocabularies cover the whole 370-card registry.
- The v1 path stays bit-identical.

## Milestone O6: history encoding

**Files:** `agent/history.py`, plus the shard and dataset formats.

**Event rows** are compact integers, expanded on the GPU the way entities are today. Each row
holds:
- the event kind and the actor's side;
- a variable-length list of (entity pointer, role) pairs;
- from-zone and to-zone codes;
- the numeric fields for that kind (amounts, counts);
- house and flank;
- the visibility code;
- the absolute turn, the offset from the current turn, the step within the turn, and a sequence
  index.

Decision events add their kind, intent, source pointer, and pointers to the chosen and offered
options. A redacted identity appears as "a hidden card from zone Z of side S".

**Three representations, all built** (O10 decides the default):

1. **Full events:** one token per projected event, uncapped (`history.max_events: null`).
2. **Turn tokens:** one token per half-turn, holding:
   - the actor, and the house chosen or forced;
   - the cards played, discarded and archived, as sets of pointers;
   - reaps, fights and uses;
   - Æmber gained, stolen, captured and lost;
   - keys forged and cards drawn;
   - hand size at the start and end of the turn;
   - whether the archive was taken or declined;
   - chains shed.
3. **Summaries:**
   - Per entity: times played, reaped, fought and used; turns since it was last seen in public; its
     last public zone; how it left view; whether it was ever in a revealed hand; the turn it was
     drawn (own cards only).
   - Global: each side's house choices (counts, the last three, and turns since each house was
     chosen); hand size at the end of each of their last three turns; mulligans; declined archive
     pickups.

**Incremental and shared.**
- `HistoryEncoder(viewer)` consumes the projection.
- At each real decision it freezes a **prefix**, which every simulation of that search shares.
- A sampled world only appends its **suffix**: the events it simulated.
- The prefix carries a rolling blake2b digest. Cache keys use the digest, never the full history
  bytes.

**Storage.**
- Self-play and BC shards store each game's projected streams once per seat. Each position stores a
  cursor into them, so storage grows with events per game rather than events × positions.
- BC corpora are re-encoded from their replay records, which are the source of truth, as
  `ml/dataset.py` already provides.
- Mirror augmentation also mirrors the flank and battleline fields of event rows.

**Acceptance.**
- History bytes are invariant under determinization and under σ (I1).
- Prefix plus suffix is byte-identical to encoding from scratch.
- A round trip through the shards is identical to live encoding.
- The per-leaf encoding overhead is measured against the cost of one simulation (about 1.5–2 ms).

## Milestone O7: network v2

**File:** `ml/model_v2.py`. v1's `ml/model.py` is untouched.

**One trunk over typed token sets.** The token types are:
- the global token and the 72 entities;
- effects and resolution frames;
- history, in whichever representation is chosen;
- match tokens;
- options, as a choice (`options_in_trunk`), so that options can attend to the state and the
  history before they are scored.

Each type has its own input projection plus a type embedding.

**Pointer enrichment at input.** A token that refers to cards adds, for each pointer, a role
embedding combined with that card's identity embedding and static row. An event is then "about" its
cards from the first layer.

**Time** uses continuous sinusoidal features of the relative turn offset, the sequence index and
the turn number. There is no table, so no cap.

**Attention** uses `scaled_dot_product_attention` with padding masks, and sequences are bucketed by
length within batches.
- The layer is built: `ml/layers.py`'s `FastEncoderLayer`, already v1's default trunk
  (`network.attention: sdpa`, 2026-10-05). v2 adds the padding and block-causal masks to it.
- The flash kernel takes no arbitrary mask, so a masked batch runs on cuDNN or the memory-efficient
  kernel. This is a second reason to bucket by length: a full bucket needs no mask.

**History architectures** (`model.history_arch`), all built:
- **`joint`:** history tokens are ordinary tokens in the trunk. This is the most expressive, and
  nothing can be cached.
- **`stream`:** a block-causal mask. History tokens attend only to earlier history, while state
  tokens attend to everything. The history part of every layer then depends only on the prefix, so
  its keys and values are cached per prefix (O8).
- **`turn_tokens`, `summary` and `none`:** the cheaper representations, and the v1 equivalent.

Optional memory compression (Perceiver-style latent tokens, `stream_memory`) defaults to "all",
which means uncapped.

**Heads.**
- Policy, value, multi-select and Q keep their interfaces. Options may now come from the trunk.
- **Belief v2:** for each hidden entity, a distribution over {hand, archive, deck}, plus P(next
  draw). O3's exact prior is an input, so the head only has to learn the difference from it, and
  that difference is where inference from history shows up.
- The oracle's input is extended to cover the archive.
- **Optional auxiliary heads,** weighted 0 by default: predicting the opponent's next house choice
  and next card played. These targets reward using the history.

**Capacity.** Re-run the M3 capacity ladder on v2 inputs, since more inputs may need more width or
depth.
- **The ladder is 128/4, 192/6, 256/8, 384/8 and 512/12** (`d_model`/layers), crossed with the
  history architectures.
- **Measure it on targets that can show a difference:**
  - value log-loss, split into mid-turn and boundary positions;
  - agreement with a 1,000-simulation search, since BC on HeuristicBot saturates at ~0.91 for every
    size Screen 4 tried;
  - search win rate at equal wall time.
- **The GPU has 8 GiB**, shared between the learner and the inference server during self-play.
  Activation checkpointing (`activation_checkpointing`) and gradient accumulation (`micro_batch`)
  are built (2026-10-05). They change the cost, not the model; tests hold both to the plain path's
  losses and gradients.
- **Measured on real BC steps at 73 tokens** (Tier 0 corpus, batch 512, bf16, compiled):

  | `d_model` / layers | Params | Plain | Checkpointing | Pieces of 256 |
  |---|---|---|---|---|
  | 128 / 4 | 1.0M | 17.4k samples/s, 0.82 GiB | 16.0k, 0.43 GiB | — |
  | 256 / 8 | 5.7M | 6.7k, 2.70 GiB | 5.8k, 0.97 GiB | 6.0k, 1.45 GiB |
  | 384 / 8 | 12.6M | 4.2k, 4.05 GiB | 3.6k, 1.46 GiB | 3.9k, 2.23 GiB |
  | 512 / 12 | 30.5M | out of memory | 1.7k, 2.38 GiB | 1.9k, 4.26 GiB |
  | 768 / 12 | 68M | — | 0.9k, 3.76 GiB | 1.0k, 4.06 GiB (pieces of 128) |

  Each option costs 8–15% of samples/s, so neither is on by default. Pieces of 128 cost the 1M
  network 60%: every piece repeats the heads' launches and loss preparation.
- **At 400 tokens** (trunk only, batches of 64): 256/8 trains at 1.2k samples/s in 1.7 GiB, 384/8 at
  720, and 512/12 at 320 in 4.8 GiB.
- **What actually sets the size is throughput, not memory.**
  - The learner needs ~4.6k samples/s at `positions_per_step` 12 and batch 1,024, given ~200k
    positions/hour from the actors.
  - Inference needs ~5k evaluations/s at the actors' batch sizes, on the same GPU.
  - At 73 tokens, 256/8 keeps up with both, and 384/8 about does. At 400 tokens nothing above 128/4
    does unless positions are reused less (a lower replay ratio is itself an option).
  - Beyond that: train a larger network offline and distil it into a smaller one for the actors,
    or use the large network only at the root.

**Cost estimate, to be replaced by measurement.** Assume sequences of about 400 tokens instead of 73:
- Attention costs about 30× more, and the feed-forward layers about 5.5× more.
- So `joint` might run at 4–8k evaluations/s, against 41–43k today.
- Self-play demand, measured without self-play (O10's v1 baseline, 2026-10-02), is 2.1k
  evaluations/s for the within-turn arm and 2.9k for the full-game arm. The GPU is about 7%
  utilized at that rate.
- Part R pushes demand up. Once forks are copies at every decision (R6), simulations get cheaper,
  so each actor asks the GPU for more evaluations per second.
- So `joint` may or may not keep up at Phase 1.1. O10 decides with measured numbers, taken after
  R6.

**Acceptance.**
- Shapes are correct for every mix of tokens and every decision kind.
- **Padding invariance:** appending padding never changes an output.
- Throughput is measured for each architecture, at batch 512 and at the batch sizes the actors
  actually send.
- Checkpoints record feature version 2 and every vocabulary version.

## Milestone O8: inference and search plumbing

- **Request protocol v2.**
  - A `Request` carries a reference to its prefix (a game key plus the digest) and its own suffix.
  - The server holds an LRU of prefix tensors, and for `stream` also the per-layer keys and values,
    so each search sends its prefix once.
  - The naive path, sending the full history with every request, stays as the reference
    implementation.
- **Evaluation cache.** It is keyed by (state bytes, history digest), and its hit rates are
  reported against v1's.
- **Per-call overhead.** O10's v1 baseline found workers waiting on inference 25–28% of their time
  with the GPU about 93% idle. What has been measured since (2026-10-02):
  - **Persistent connections: done.** Each worker now holds one connection, with `TCP_NODELAY` set.
    Without that option, a long-lived connection stalls about 40 ms per large message on a delayed
    ACK: 52 ms against 11 ms per round trip. With it, a call takes 9.2 ms, against 10.0 ms on a fresh
    connection. Before the option was set, actor throughput didn't change (2,049 against 2,101
    evaluations/s, within noise). With it, the gain is the ~1 ms per call above, so the connection
    was not where the time went.
  - **Server result handling was where it went: fixed.** On a 229-request batch,
    `TorchModel.predict_many` spent 16 of its 22 ms copying results to the CPU one request at a
    time. It now copies once per head, with identical answers on every head:
    - 32 requests: 7.3 → 5.5 ms;
    - 229 requests: 20.5 → 7.0 ms;
    - 512 requests: 48.8 → 14.3 ms.

    In the actor benchmark (5 × 16 games), waiting fell from 25% to 14% of worker time, and searches
    per second rose 12% (49.6 → 55.7).
  - Shared memory stays an option behind both.
- **Search.**
  - `search.determinization` is wired through `SearchSettings`.
  - Encodings at the opponent's nodes inside worlds follow O3's private-history option.
  - `fork_backend` defaults to `copy` at every decision (R6); replay and process fork remain
    available.

**Acceptance.**
- Outputs are identical with and without the prefix cache: bitwise on CPU, within fp16 tolerance on
  GPU.
- The prefix cache hit rate within a search is at least 95%.
- Pipe bytes per request and server throughput are measured.
- The actor benchmark's share of time spent waiting on inference falls from its v1 baseline (25–28%).

## Milestone O9: data and training

- **Data.**
  - Generate after R7, in compiled execution, so every v2 rung sees the same data, resolution tokens
    included.
  - Re-encode the BC corpora as v2 (`DATASET_VERSION` 4).
  - Define the v2 self-play shard format, using O6's streams and cursors. Nothing is written in it
    until self-play starts (see Sequencing).
  - Extend the privileged labels with the opponent's archive and deck order, for belief v2 and
    next-draw prediction.
- **Tier 0b.** Behaviour cloning with v2 on the same games as Tier 0, at least one run per history
  representation. This is the cheapest place to see what the new inputs buy. Tier 0 took 2 h for 4
  epochs; Tier 0b is estimated at 5–10× that.
- **Learner.**
  - Add the belief v2 loss.
  - Add history token dropout (training only, `training.history_dropout`, default 0), alongside
    weight decay, to counter memorization. The random-deck value finding shows that unique inputs
    invite memorization.

**Training throughput (measured 2026-10-02 on the Tier 0 corpus).** Behaviour cloning was
compute-bound, not data-bound: waiting for data took 1 ms of a 100 ms step. Most of the step went to
fp32 arithmetic, plus multi-select losses built from Python loops of single-element GPU writes. Two
changes:
- **The multi-select losses are vectorized.** The losses are equal and the gradients identical, and
  a test checks them against per-row references.
- **Training precision is a config value,** `bc.precision` and `selfplay.precision`. Both default to
  `bf16` autocast.

| Batch 512 | Samples / s | GPU busy |
|---|---|---|
| Before (fp32) | 5,150 | 62% |
| Vectorized losses, fp32 | 6,677 | 76% |
| Vectorized losses, bf16 | **10,855** (2.1×) | 57% |
| Vectorized losses, bf16, batch 2048 | **14,128** (2.7×) | 83% |

**Comparability.** Tier 0's v1 numbers were trained in fp32. The ladder's A0 rung is therefore
retrained at the same precision as the v2 rungs, which is cheap now, so precision never confounds
"what does seeing more buy". Batch 2048 changes the optimization, so it's an option to validate with
a retuned learning rate, not a new default.

**Where the bf16 step goes (2026-10-02).** The torch profiler puts a batch-512 step at about 47 ms
of CPU time against 25 ms of GPU kernels, over about 1,100 kernel launches. The step is
**launch-bound**: the GPU waits for Python to issue work.
- **Fused AdamW (kept):** one kernel for the whole update, +8% (10,970 against 10,170 samples/s,
  A/B in one process). Both the BC trainer and the learner use it on CUDA.
- **A sync-free step (tried, reverted):** index tensors built and pinned on the prefetch thread,
  masked means instead of boolean gathers, losses read back a step late. This was 5–6% *slower*.
  The CPU never waited on the GPU, and the prefetch thread's Python competed with the main thread
  for the GIL.
- **`torch.compile` of the trunk (kept; `bc.compile` / `selfplay.compile`, default on):** this fuses
  the layer norms, activations and autocast casts. End to end in `train()`, steady state: **13,506
  against 11,060 samples/s (+22%)**, about 20 s of one-time compile. Checkpoint keys are unchanged.
  `build-essential` and `python3.14-dev` were installed in WSL for it (2026-10-03).
- **Faster batch assembly (kept):** the shards are indexed as plain arrays, not `np.memmap` objects,
  and two per-row loops are vectorized. Each batch takes 4.6 ms instead of 5.8 ms, with identical
  output.
- **A loader process (tried, reverted):** this ran batch assembly and `MultiPrep` in a DataLoader
  worker, with pinned memory and a sync-free step. It was slower in every arrangement (10–12k
  samples/s). The page-locking thread is starved of the GIL by the launch-bound step, as the prefetch
  thread was before, and a shorter GIL switch interval didn't help. The GPU syncs in the current step
  are what give the prefetch thread its turns.
- **Batch 2048** amortizes launches too (14,128 samples/s), with the caveat above.
- **Evaluation prefetches its batches:** 58 s instead of 71 s on Tier 0's validation split, with an
  identical report.

**bf16 against fp32, measured (2026-10-02).** Screen 4's `reference` network was retrained in bf16:
one epoch, the same seed, data order and corpus (run `tier0-bf16`):
```
tools.run_screens --stages ablations --ablations reference --ablation-epochs 1 --data <Tier 0 corpus>
```

| Metric | fp32 (Tier 0) | bf16 |
|---|---|---|
| CHOOSE_ACTION top-1 | 0.9101 | 0.9091 |
| CHOOSE_HOUSE top-1 | 0.9375 | 0.9377 |
| CHOOSE_CARDS: enumerate / sequential / top-k | 0.8631 / 0.8521 / 0.8556 | 0.8575 / 0.8545 / 0.8578 |
| Value log-loss (constant: 0.6931) | 0.4928 | 0.4929 |
| Wall time (train + evaluation) | ~1,500–2,000 s | 779 s |

- **Every difference sits inside the spread of Tier 0's one-epoch fp32 variants.** CHOOSE_ACTION
  ranges 0.9049–0.9121 across them, enumerate 0.8578–0.8639, and value log-loss 0.4928–0.4961.
- **Enumerate's −0.56 points looks like noise.** The two other multi-select heads, trained on the
  same targets, moved up instead.
- **bf16 is accepted as equivalent.** This rests on one run per precision. A second seed would
  tighten the comparison if it's ever needed.

**Acceptance.** A round trip through the shards is identical to live encoding, and Tier 0b's wall
time and memory are recorded.

## Milestone O10: measurement and decisions

This is where "unless necessary" gets decided.

**The ladder.** Each rung adds to the one before it. The BC-level rungs come first, then the
search-level rungs.

| Rung | Adds |
|---|---|
| A0 | v1, the baseline |
| A1 | the complete current state and the lossless encoding (O4, O5); no history, no knowledge |
| A2 | knowledge and exact priors (O2, and the priors from O3) |
| A3 | history summaries |
| A4 | turn tokens |
| A5 | the full event history, `joint` |
| A6 | the full event history, `stream` |
| A7 | static card meaning, each option separately |
| A8 | options in the trunk |

**Metrics.**
- Belief log-loss, against the `chance_exact` prior (the new honest baseline) and against uniform
  (0.541).
- Value log-loss by turn bucket, split into mid-turn positions and turn boundaries.
- BC top-1 per decision kind.
- Generalization to held-out cards (a Screen 5 rerun).
- Search win rate at equal simulations against HeuristicBot, plus v1 against v2 head to head, on
  paired seeds with both seat orders.
- The G3 hidden-information gap, measured again: how much of the exact-fork advantage v2 recovers
  (today +0.005 within-turn and +0.061 full-game).
- Throughput: evaluations/s and games/hour.

**Measuring actor demand without self-play.** The decision rules below need to know how fast
self-play actors would ask for evaluations. That is measured with **search evaluation games shaped
like an actor**:
- the same number of processes and concurrent games per process;
- the same search settings and batched inference;
- the opponent is HeuristicBot, nothing is learned, and no shards are written.

The benchmark is `tools/bench_actor_demand.py`. The user confirmed (2026-10-02) that it is not
self-play, so it can run at any time:
- **A baseline is taken now,** with today's engine and the Tier 0 (v1) network.
- **The deciding measurement is repeated after R6** with the v2 network, because copy-anywhere
  changes the cost of a simulation.

**The v1 baseline (measured 2026-10-02).** The setup:
- each arm's config: 5 workers × 16 games, 100 simulations on 25% of decisions and 25 on the rest;
- the Tier 0 network on the RTX 5060 Ti;
- 60 s of warmup, then 480 s measured;
- results under `$KEYFORGE_DATA/bench/actor_demand/baseline-v1-*.json`.

| Quantity | Within-turn | Full-game |
|---|---|---|
| Evaluations requested per second | 2,101 | 2,949 |
| Requests per call | 49 | 75 |
| Evaluations per simulation (after the evaluation cache) | 0.97 | 1.43 |
| Searched decisions / simulations per second | 49.6 / 2,167 | 47.3 / 2,070 |
| Share of worker time spent waiting on inference | 25% | 28% |
| GPU utilization: mean / p90 | 7% / 13% | 7% / 9% |
| GPU memory, peak | 1.27 GB | 1.36 GB |
| Games per hour against HeuristicBot | 2,593 | 2,469 |
| Self-play games per hour, estimated (both seats searching) | ~1,300 | ~1,230 |

What it shows:
- **Demand matches the plan's estimate** (about 3k/s). It is roughly a 15th of what the v1 network
  can serve: 47k evaluations/s at batch 512 in the M2 test.
- **Even so, workers lose a quarter of their time waiting on inference** while the GPU is about 93%
  idle. Attribution (2026-10-02) puts the cost in how the server hands back results, not in the
  network or the connection:
  - on a 229-request batch (five workers' calls batched together), `predict_many` takes 22 ms, and
    16 ms of that is a device-to-host copy for each request;
  - connecting per call costs about 1 ms.

  O8 takes this on.
- **The estimate of about 1,300 self-play games per hour** sits below the plan's 1,500 and above the
  aborted `wt-s0` run's ~900, which shared the machine with its learner. A real run would also
  share it.
- **None of O10's cap conditions comes close** at v1 cost per evaluation. The deciding run after R6
  is the one that counts.

**Scaling the actors (2026-10-02, after O8's server fix).** These are the same within-turn
settings, 30 s warmup and 150 s measured. Searches per second is the reliable figure: a short window
finishes too few games for games per hour.

| Workers × games each | Searches / s | Simulations / s | Waiting on inference | GPU mean |
|---|---|---|---|---|
| 5 × 16 | 55.7 | 2,427 | 14% | 7% |
| 8 × 16 | 71.5 | 3,105 | 19% | 8% |
| 10 × 16 | 74.1 | 3,228 | 24% | 8% |
| 5 × 32 | 55.1 | 2,397 | 8% | 6% |
| 10 × 32 | 81.9 | 3,555 | 17% | 8% |

- **Throughput is set by the CPU,** with 6 cores and 12 threads. Beyond 8 workers, the extra
  hyperthreads add little.
- **The default is now 8 actors** (`selfplay.workers`): +28% over 5, and it leaves room for the
  learner and the server.
- **The GPU stays about 93% idle at every setting.** That headroom is what a larger network (O7)
  will use. It is free only while the server answers quickly, because actors wait on every call.

**Where an actor's time goes, and cheaper forks (2026-10-02).** A single within-turn worker (16
games, 90 s) was timed with wrappers around its main phases. cProfile overstated the copy about
3×, because it times every getattr/setattr.

| Phase | Before | After |
|---|---|---|
| Forking the root world, every simulation | 57% | 40% |
| ↳ replay forks off a boundary decision (30% of forks) | 32% (1.6 ms each) | gone |
| ↳ `Game.copy` | 23% (0.49 ms each) | 37% (0.40 ms each, now also serving the forks above) |
| Waiting on inference | 18% | 24% |
| Infoset building and encoding | 9% | 13% |
| Descent and end-of-turn rollouts | ~13% | ~20% |
| Searches in 90 s | 1,376 | 1,936 (+41%) |

The two changes, both in `keyforge/game.py`, behave identically:
- **A fork off a boundary decision copies a cached snapshot.** The true game keeps a private copy
  of itself at its latest boundary decision. A fork copies that and replays only the few choices
  since (`Game._fork_from_snapshot`), instead of replaying the whole game. Tests check `state_hash`
  against the replay fork at every non-boundary decision of 14 games, and every resample mode.
- **Cards are copied by generated straight-line code** instead of a getattr/setattr loop over 35
  slot names: 2× per card in isolation, ~17% per copy in a busy worker.

A copy is now mostly allocation, roughly 300 objects for 72 cards. Making it cheaper again is Part
R's job (copy anywhere, snapshot/restore).

At 8 workers × 16 games (30 s warmup, 150 s measured), more workers now pay, because each worker
waits longer:

| Workers × games each | Searches / s | Simulations / s | Waiting on inference |
|---|---|---|---|
| 8 × 16, before | 71.5 | 3,105 | 19% |
| 8 × 16 | **92.0** (+29%) | 4,006 | 25% |
| 8 × 32 | 94.1 | 4,084 | 17% |
| 10 × 32 | 104.7 | 4,559 | 22% |
| 12 × 16 | 108.6 | 4,715 | 34% |

The default stays at 8 actors. The benchmark runs no learner, and a real run's learner needs those
cores. The count is chosen when self-play starts, by measuring with the learner running (O10's
deciding run).

**The inference server, and a cheaper copy again (2026-10-03).** With faster actors the server
became the limit. It was busy 75% of the time under 8 workers, at 15 ms per batch: 6 ms collating
and the rest launching kernels. On a captured batch of real requests, the GPU worked 2 ms of a
7.5 ms call.
- **CUDA graphs.** The trunk, value head and policy softmax are captured once per padded shape
  (rows up to 8…512, options up to 8…128), then replayed: 2.7 ms against 5.2 ms on 50-request
  batches. The rarer heads run eagerly from the captured outputs. Answers match the eager path within
  fp16 noise, at most 1e-3 on values and 4e-3 on probabilities. A swapped-in network (a promotion)
  drops the graphs.
- **`collate` is whole-array NumPy:** 3.4× faster, with identical output.
- **Live, 8 workers:** server busy 43% instead of 75%, 6 ms per batch, actors wait 14% instead of
  25%, +14% searches per second.
- **`Game.copy` 10% cheaper:** only cards that reference other cards are relinked, and the choice
  record is copied shallowly.
- **Tried and dropped:**
  - A larger young-generation GC threshold. The cyclic collector is a third of a copy in isolation,
    but in a worker the change was within noise or worse.
  - `torch.compile` of the server trunk, alone: 15%, superseded by the graphs.
  - Engine and infoset micro-optimization: both profiles are flat, and v2 replaces the infoset and
    encoder.

At 8 × 16: **111–121 searches per second** over three runs, against 92.0 before and 71.5 at the
start of the day. Repeated runs differ by up to 10%, so a single run cannot judge a change smaller
than that.

**Probes.**
- Targeted positions where the right move depends on only one new kind of information:
  - a bounced creature in the opponent's hand;
  - Lash of Broken Dreams' +3 key cost on their next turn;
  - a known top card;
  - the remaining steps of an ability that is resolving.
- Linear probes on trunk embeddings, so that each input's use is shown directly.

**Decision rules, written now.**
- A feature group stays on by default in either of two cases:
  - it improves a primary metric (belief log-loss, value log-loss, BC top-1, search win rate) by
    more than 2 standard errors at equal compute;
  - it is neutral within noise and costs less than 10% of throughput.

  Otherwise it stays built and switched off.
- A compression or cap is adopted only if the uncapped version meets one of three conditions. This
  covers turn tokens instead of events, `stream` memory limits and `history.max_events`. The
  conditions are:
  - it starves the actors: in the actor-shaped benchmark above, the GPU is at 100% while the games
    wait on inference;
  - it exceeds GPU memory at the learner's batch size, even with checkpointing and accumulation;
  - it is measurably worse, meaning a train/validation gap in value loss beyond v1's.
- The winning v2 configuration becomes the input for the training plan's Tier 2 and 3 self-play
  arms. Those runs start only after the self-play gate in Sequencing. A v1 arm is kept for the
  comparison if the budget allows.
- If v2 knowledge closes more of the gap, the within-turn belief arm (dropped at G3) is
  re-evaluated.

## Milestone O11 (gated, costly): option-consequence features

For each legal option:
1. Apply it in K `chance_exact` worlds, never in the true game.
2. Run each world to the next decision.
3. Append the averaged changes in public state to the option's features: Æmber, keys, creatures
   destroyed and damaged, cards drawn and discarded.

It is leak-free by construction, because it only uses sampled worlds. It is also expensive: about
20 options × K worlds × about 1.5 ms per decision. It is meant for search roots and the search-free
agents (NetAgent and DMC), not for every leaf.

**Gate.** Build it if A8 (options in the trunk) still leaves a gap in the policy metrics that the
search-free agents feel. It is measured on the same ladder.

---

## Part R: the explicit resolution stack (engine refactor)

**Scheduled, not gated** (decided 2026-09-30). This supersedes the scope boundary in the interface
plan and the training plan ("card effects stay generators").

### Why

Today, everything between two decisions that is still to happen lives inside suspended Python
generators.

**Size of the generator layer** (measured, AST scan of `keyforge/`):

| Quantity | Count |
|---|---|
| Generator functions | 367 |
| ...that actually pause | 232 |
| ...that are stubs and never pause (`return; yield`) | 135 |
| Suspension points | 551 |
| Functions that pause inside a loop | 30 |
| Generator functions nested inside other functions | 40 |

**Where they live:**
- **The kernel:** 37 pausing methods in `game.py`, covering the turn loop, the action pipeline,
  `_play_card`, `_fight`, `destroy_cards`, `_fire_event`, the forge step and the choice helpers.
  There are 5 more in `match.py` and 4 in `effects/steps.py`.
- **Card effects:** 306 generator functions in `effects/named/*.py`, of which 180 pause, and 15 in
  `effects/generic.py`, of which 6 pause. These serve 363 cards with hooks: `on_play` 182,
  `register_passive` 66, `on_reap` 45, `on_action` 42, `on_fight` 24, `on_omni` 17, plus the
  destroyed and before-fight hooks.

**Callables stored in game state:**
- 85 sites create lingering-effect objects (35 duration, 31 trigger, 15 modifier, 4 instead). Their
  handlers, values and conditions are closures.
- Upgrade-granted `extra_triggers` hold closures too.
- `Game.copy()` re-points those closures by rewriting their cells (`_rebind_closure`).
- `state_hash` can only approximate them, as `"<callable>"`.

**What that costs:**

1. **The network can't see what is left to resolve,** only the asking card and its intent
   (Inventory D).
2. **Snapshot copy works only at boundary decisions:** CHOOSE_ACTION, CHOOSE_HOUSE and
   TAKE_ARCHIVE. In HeuristicBot play that is 72.4% of decisions (30 games, 4,786 decisions).
   - Everywhere else, a fork has to replay the game.
   - At mid-game a replay fork takes **1.39 ms** and a copy **0.33 ms**, a 4.2× difference (median
     of 62 forks at decisions 40, 75 and 110).
   - Every search whose root is mid-resolution pays the replay price on every simulation.
3. **A game can't be serialized mid-resolution.** There is no save and restore, and no way to send a
   state to another process except by replaying its choices.
4. **`state_hash` approximates** closures as `"<callable>"` and pending cleanups by their count.

### Target

- **State as data plus a stack.** Game state is plain data plus an explicit **resolution stack** of
  frames. A frame holds:
  - its routine id;
  - its suspension point (`pc`);
  - its locals, as plain data. Cards are held by reference, remapped on copy and serialized as
    instance ids;
  - its finalizers;
  - where its return value goes.
- **How decisions flow.** `pending_decision` is the Decision the top frame suspended on. `submit`
  delivers the choice to that frame and runs the machine until the next suspension.
- **Every callable stored in game state becomes a `Ref`:** a routine id plus its captured
  environment, as plain data. That covers lingering-effect handlers, values and conditions,
  upgrade-granted triggers, and `ORDER_EFFECTS` items. Card hooks stay static (looked up by card
  name and hook kind).
- **Copy, serialize, hash.**
  - `Game.copy()` is valid at **every** decision.
  - `Game.snapshot()` and `Game.restore()` serialize any state.
  - `state_hash` is exact.
- **The public API is unchanged:** `Game`, `submit`, `submit_index`, `pending_decision`,
  `view_for`, `log`, `choice_record`, `replay`, `fork`, `fork_determinized`, `copy`, `run_until`,
  `state_hash` and `Match`. The GUI, the bots and the agents need no changes.

### Approach: compile, don't rewrite

Card effects keep being written as generator functions. That is how all 363 hooked cards are
written today, and how new cards will be written.

A compiler turns each pausing function into an explicit state machine:
- its suspension points become numbered `pc` values;
- its locals live in the frame;
- each `yield from` call pushes a frame.

Python code runs natively between suspension points, so the overhead is paid per suspension, not
per statement. The original source stays the single source of truth.

| Route considered | Why not |
|---|---|
| Hand-write every effect as an explicit state machine | 232 pausing functions and 551 suspension points converted by hand. Card code becomes unreadable, and every future card pays the same cost. |
| A declarative effect language interpreted by a VM | It re-expresses the semantics of 363 cards, which is the largest fidelity risk available, and partial coverage still needs escape hatches. The compiler's static "remaining operations" tables (R7) give the network the same view of what is left. |
| Process fork (interface plan E1) | It already works in WSL and copies suspended generators. But the state stays opaque: it can't be encoded or serialized, and it doesn't work on Windows. It stays as a backend option. |

### Dual execution: the permanent oracle

The same source runs in two ways:
- **`native`:** Python generators, which is today's engine;
- **`compiled`:** the explicit stack.

Because both come from one source, every future rules fix applies to both. A differential test that
compares them keeps validating the compiler for as long as the project lives (I8).

A frozen copy of the pre-refactor engine (R0) is the second oracle. It checks the source changes
made in R1.

### R0: the oracle and the harness

**Precondition.** A clean working tree. The dead-code cleanup that stood in the way was committed on
2026-09-30 (`f90a9e0`, O0 item 5).

**Freeze the engine** as `keyforge_ref/`, a vendored copy at the starting commit, importable side by
side with `keyforge/`.

**Write `tools/diff_engines.py`.** It drives two engines in lockstep with identical choices and
compares them after every submit:
- the pending decision: player, kind, option keys, `min_n`/`max_n`, intent, `affects` and source;
- the log: every event's kind and data;
- the journal, once O1 exists;
- the RNG counters;
- the choice record;
- `state_hash` over the components both engines share.

**Corpora:**
1. The golden corpus: 22 recorded games with state hashes.
2. The whole test suite (888 test functions), run under both execution modes.
3. A fuzz of 100,000 games:
   - played by RandomBot, HeuristicBot and an ε-greedy HeuristicBot;
   - over Fignor/Igor, every Phase 2 and 3 preset, random legal decks and alliance decks;
   - in Archon single games and in Reversal and Adaptive matches.
4. Coverage-directed scenarios. Coverage is measured on the reference, per routine and per
   suspension point. Every point the fuzz misses gets a `setup_script` position that reaches it, or
   a written reason why it can't be reached.

**Benchmarks:**
- `tools/bench_engine.py` decisions/s;
- replay fork against copy, at boundary and at mid-resolution decisions;
- search simulations/s at 100 simulations with G3's settings.

**Acceptance.** The harness passes the reference against itself, the coverage report exists, and
the baselines are recorded.

### R1: stored callables become data, in the source

Both execution modes share this change.
- **Introduce `Ref(routine id, env)`.** Routines live in a registry. `env` holds plain data: cards,
  ints, strings, houses and tuples.
- **Rewrite every stored callable as a `Ref`:**
  - the 85 lingering-effect creation sites;
  - upgrade-granted `extra_triggers`;
  - the ad hoc `ORDER_EFFECTS` tuples (`("extra", card, handler)`);
  - the factory closures in `effects/generic.py` (for example, `gain_n(n)` becomes a `Ref` with `n`
    in its environment).
- **Hashing and copying.** `state_hash` hashes `Ref`s exactly, so the `"<callable>"` and
  cleanup-count approximations go away. `_rebind_closure` is deleted, and copy remaps `Ref`
  environments instead.
- **Loops that pause while iterating a live engine list.** The R3 linter finds these among the 30
  looping functions. Python iterates a live list by index, so a list that changes mid-loop matters.
  Each one is made explicit, either as a materialized copy or as the live list plus an index:
  whichever matches the reference trace.

**Acceptance.** Native execution after R1 is trace-identical to `keyforge_ref` on every corpus.

### R2: the machine, with an adapter

- **`keyforge/vm.py`** holds the frames and the stack. It defines the step protocol: suspend with a
  Decision, call, return, run finalizers. It also runs the driver loop behind `submit` and
  `submit_index`.
- **A `NativeFrame` adapter** wraps a live generator, so code that isn't compiled yet runs inside
  the machine unchanged. `copy()` works whenever no `NativeFrame` is live, and falls back to replay
  otherwise. The engine therefore works at every step of the migration.
- The game's driver becomes the root frame.

**Acceptance.** With everything still running native, every corpus stays trace-identical.

### R3: the compiler

**`tools/compile_engine.py`** works in four steps:
1. Parse each pausing function's AST.
2. Check it against the supported subset.
3. Split it into blocks at its suspension points.
4. Emit a state-machine routine.

**The supported subset** comes from a census of the constructs inside the pausing functions (AST
scan, 2026-09-30). It covers:
- assignment, including augmented (40 sites) and annotated (1) assignment;
- `if`/`elif`/`else`, conditional expressions (26 sites) and `while`;
- `for` over sequences, with the iterator state (sequence plus index) kept in the frame;
- `break`, `continue`, and `return` with a value;
- `yield Decision(...)`;
- `yield from` calls: to routines, to `Ref`s, and to dynamic hooks such as `card.card_def.on_reap`;
- `try`/`finally` around a pause, which occurs at exactly 2 sites: Replicator and
  `use_artifact_ability`;
- yields inside expressions, hoisted into temporaries;
- nested functions and lambdas: 5 of each inside pausing functions. A nested function is lifted into
  a routine with an explicit environment. A lambda that is only called on the spot stays plain
  Python.

The census found none of these, so they are not supported: `with`, `nonlocal`, `global`, `async`,
and a yield inside a comprehension. Anything outside the subset is a compile error with its source
line, never a silent miscompile (I5).

**Output.** Generated modules go under `keyforge/compiled/` and are checked into git. Each block is
annotated with its source line. A test fails if the generated modules are out of date.

**Compiler tests:**
- every construct;
- randomized programs within the subset, run natively and compiled, with identical traces
  (property-based).

**Acceptance.**
- The compiler's tests pass.
- Every pausing function in `keyforge/` either compiles, or is listed together with the construct
  that blocks it. That list must be empty by the end of R5.

### R4: compile the kernel

- Compile `game.py` (37 pausing methods), `match.py` (5) and `effects/steps.py` (4).
- `_resume` and the boundary-only restriction on `copy()` become dead once nothing native remains
  under the kernel.

**Acceptance.** Every corpus is trace-identical between native and compiled execution.

### R5: compile the card effects

- **Order:** `effects/generic.py` first, then one house at a time. Dis, Logos and Shadows come first,
  because they are Fignor's and Igor's houses. Brobnar, Mars, Sanctum and Untamed follow.
- **After each module:** the differential corpora, plus the coverage-directed scenarios for that
  module's cards.
- **Rules bugs found along the way** are fixed as separate changes, in the source, so both modes get
  the fix. Each fix takes its ruling from the Master Rulebook and the keyteki card database. None is
  folded into a refactor commit.

**Acceptance.** No `NativeFrame` appears in compiled execution on any corpus, and every suspension
point is covered or has a written reason why it can't be reached.

### R6: copy anywhere, serialization, faster forks

- **`Game.copy()` at every decision.** It copies the data and the frames, remapping cards inside
  frame locals, inside `Ref` environments and inside the pending Decision's options. An unknown
  object type in a frame is an error (I5).
- **`Game.snapshot() -> bytes` and `Game.restore(bytes)`.** Canonical, versioned and pickle-free.
- **`fork()` uses copy.** Replay stays as the cross-check: a copy-fork must equal a replay-fork at
  random decisions of every kind.
- **Search:** `fork_backend` defaults to `copy` everywhere. The process-fork backend stays as an
  option. A copy (0.33 ms) is already faster than a process fork (0.69 ms).

**Acceptance.**
- Copy at mid-resolution decisions is measured, with a target within 20% of today's boundary copy.
- Search simulations/s at G3 settings is at least today's.
- Snapshot round trips are identical at random decisions of every kind.
- A copy that continues is identical to the original continuing with the same choices.

### R7: the resolution stack as information

- **`game.resolution_view(viewer)`** lists, for each frame from top to bottom:
  - its routine, from a vocabulary, plus `pc` and depth;
  - its source card;
  - its ability kind and trigger event;
  - its locals, as (role, value) pairs with cards as pointers;
  - the pending trigger batch, where there is one.
- **A static table of remaining operations.** For each (routine, `pc`) the compiler emits the
  operations still reachable:
  - each `choose_*` call, with its intent and `affects`;
  - each `steps.*` or `Game` primitive, with its literal arguments;
  - loops, marked as such.
- **Per-viewer redaction** follows O1's visibility rule. I1's invariance test covers frames.
- **Determinized worlds (O3).** The re-deal's σ relabels hidden card references inside frames, and
  the consistency checker covers frames.

**Acceptance.**
- I1 holds with frames included.
- Every routine has its table of remaining operations.
- O4's resolution tokens read this view.

### R8: retire the adapter, document

- **Execution modes.** Compiled becomes the default. Native stays runnable as the oracle, and CI
  runs the differential corpora in both modes.
- **A lint test** fails if any pausing function in `keyforge/` is missing from the compiled output.
- **An authoring guide for new cards:** write a generator in the supported subset, then run the
  compiler. The up-to-date test enforces the second step.
- **Bump the engine version's minor number;** the rules hash is unchanged. The scope-boundary lines
  in the interface and training plans that ruled this refactor out were already removed on
  2026-10-02.

**Acceptance.**
- The full suite passes in both modes, on CPython 3.12 (Windows and the GUI), CPython 3.14 (WSL) and
  PyPy 3.11.
- Raw engine decisions/s is at least 90% of native, and search simulations/s at least native. If
  raw throughput misses, the measurement picks the optimization, such as per-routine slotted frame
  classes.

---

## Sequencing

| Stage | Milestones | Depends on | Produces |
|---|---|---|---|
| Foundation | O0 | a clean working tree | the leak closed, baselines recorded |
| Engine | R0 → R1 → R2 → R3 → R4 → R5 → R6 → R7 → R8 | O0 | the explicit stack, copy anywhere, serialization, the resolution view |
| Knowledge | O1 → O2 → O3 | O0 (O3 switches to copy-built worlds at R6) | the projection, the tracker, consistent worlds, exact priors |
| State | O4, O5 | O0 (O5 after O4; O4's resolution tokens need R7) | the complete, lossless v2 encoding |
| History | O6 | O1 (O2 for knowledge fields) | event, turn and summary encodings |
| Network | O7 | O5, O6 | the v2 trunk and heads |
| Plumbing | O8 | O7, O3, R6 | cached prefixes, consistent search |
| Training data | O9 | O6, O7, R7 | v2 data, Tier 0b |
| Decision | O10 | O9 (BC rungs), O8 (search rungs) | defaults for every option below |
| Gated | O11 | O10 | built only if its gate opens |
| **Self-play** | the training plan's smoke run, then Tier 2/3 | **everything above**, R8 included | trained v2 checkpoints |

**Two tracks run in parallel after O0.**
- **The engine track** (R0–R8). It has priority because it carries the most risk and other stages
  wait on it. Knowledge and encoding work proceeds alongside it.
- **The observation track** (O1–O9). O1's journal hooks live in the zone classes, which the compiler
  doesn't touch. O1, O2, O5, O6 and O7 don't need Part R. O3, O4, O8 and O9 take Part R's results
  when they land, as the table shows.

**The first answer** comes from O10's BC rungs. They need O9, and therefore R7. Behaviour cloning is
not self-play, so these rungs run as soon as their inputs exist.

**The self-play gate.** No self-play of any kind runs until every row above it is complete:
- Part O through O10;
- Part R through R8;
- O11 either built or closed by its gate.

Then the training plan's smoke run (`configs/smoke_selfplay.json`) checks the pipeline end to end,
and only after that do the Tier 2 and 3 runs start, with O10's chosen v2 configuration. Until then,
every throughput number comes from O10's actor-shaped evaluation benchmark.

**Relative engineering size:**

| Milestone | Size |
|---|---|
| O0 | S |
| O1 | L |
| O2 | M |
| O3 | M (`constrained`), L (`chance_exact`), S (`belief`) |
| O4 | M |
| O5 | M |
| O6 | M |
| O7 | M |
| O8 | M |
| O9 | M, plus compute |
| O10 | compute-bound |
| O11 | M |
| R0 | M |
| R1 | M |
| R2 | M |
| R3 | L |
| R4 | M |
| R5 | L |
| R6 | M |
| R7 | M |
| R8 | S |
| Part R overall | XL |

## Risks

| Risk | Mitigation |
|---|---|
| Memorization: with history, every position is unique | History dropout, weight decay and more games. The train/validation gap is an O10 metric and a decision rule. |
| Throughput collapses | Measured per architecture before any self-play. `stream` with prefix caching exists, and the compression options are adopted only by rule. |
| A leak regresses | The I1 invariance and σ tests stay in the suite, alongside the content-level history fuzz. |
| Worlds are biased: `chance_exact` assumes the opponent's choices carry no information | The `belief` sampler, and a measured calibration of the priors. |
| Instrumenting zones breaks the engine | The full existing suite, replay determinism, and the benchmark within 10%. |
| Complexity | Every group sits behind its own config switch, and the v1 path is frozen and runnable. |
| GPU memory at the learner | Activation checkpointing and gradient accumulation, which cost time, not capability. |
| The compiled engine diverges from today's | Dual execution with a permanent differential test (I8), coverage-directed scenarios, and the frozen R0 reference for source changes. |
| A compiler bug in a rarely used construct | The subset is closed, so anything else is a compile error. Property-based tests run randomized programs against native execution. |
| The compiled engine is slower | Overhead is paid only at suspension points. R8 has a throughput gate. Copy-anywhere makes search faster regardless. |
| A rules fix lands during the refactor | The generator source stays the single source of truth. A fix lands there, as its own change, and both modes get it. |
| The refactor delays everything else | The observation track runs in parallel. Only O3's copy-built worlds, O4's resolution tokens, O8 and O9 wait for Part R. |

## Options register

| Option | Values | Default for the v2 arm | Decided by |
|---|---|---|---|
| `observation.version` | 1 / 2 | 2 (baselines: 1) | — |
| `observation.knowledge` | none / hard / hard+priors | hard+priors | O10 A2 |
| `observation.effects` | v1 counts / tokens | tokens | O10 A1 |
| `observation.resolution` | none / frames | frames | O10 A1 |
| `observation.history` | none / summary / turn_tokens / events | events | O10 A3–A6 |
| `observation.history_max_events` | null / N | null (uncapped) | O10 rule |
| `observation.match_history` | summaries / full streams | summaries | Adaptive and Reversal runs |
| `observation.static_semantics` | flags / parsed / signatures / text_embedding (combinable) | parsed + signatures | O10 A7, Screen 5 |
| `observation.options_in_trunk` | off / on | on | O10 A8 |
| `observation.unknown_vocab` | error / unknown-id | error | — |
| `model.history_arch` | joint / stream / turn_tokens / summary / none | joint | O10 rule |
| `model.stream_memory` | all / K latents | all | O10 rule |
| `model.aux_opponent_heads` | loss weight | 0 | O10 |
| `training.history_dropout` | 0–1 | 0 | O10 (memorization) |
| `search.determinization` | uniform / constrained / chance_exact / belief | chance_exact | O10, the G3 rerun |
| `search.world_private_history` | drop / relabel | drop | O3's checker |
| `search.resample` | own_deck / opponent_private / all | all | the diagnostic's conditions |
| `search.fork_backend` | replay / copy / process | copy at every decision (from R6) | R6's benchmarks |
| `engine.execution` | native / compiled | compiled (from R8); native stays as the oracle | R8 |
| Option-consequence features (O11) | off / K worlds | off | O11's gate |

**Deliberately not inputs,** because they fall outside entitlement:
- hidden identities and deck order (available only through privileged training labels);
- RNG state;
- the opponent's offered options and their count;
- anything in the future.

**Opponent identity** (self, HeuristicBot or a human) is legitimate information. But conditioning on
it changes what the evaluation measures, so it is recorded here and not planned.
