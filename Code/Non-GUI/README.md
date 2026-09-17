# KeyForge Phase 1.1 (Archon) — Non-GUI

An implementation of Phase 1.1 of the project (`PHASE_1_1_PLAN.md`): the
Archon format using the two custom decks, Fignor and Igor. Pure Python
standard library, no dependencies.

## Layout

- `keyforge/` — the engine (no `print()`/`input()` anywhere):
  - `enums.py`, `config.py` — houses, card types, decision kinds, `GameConfig`
  - `cards/` — `Card`/`TypeObject`, all 49 card definitions, the Fignor/Igor decklists
  - `zones.py`, `player.py` — Deck/Hand/DiscardPile/Archive/PurgedZone/PlayArea, and `Player` with its `get_*` functions
  - `effects/` — `EffectObject` family (`DurationEffect`/`InsteadEffect`/`TriggerEffect`), basic steps, and generic/named card effects
  - `current_queue.py` — tier constants used to order queued effects
  - `decision.py` — `Decision` + choice validation
  - `game.py` — `Game`: setup, turn order, legal actions, play/reap/fight/destroy, `submit()`
  - `view.py` — `PlayerView`, a read-only snapshot with hidden information removed
  - `log.py` — structured event log
- `bots/` — `Controller` interface, a seeded `RandomBot` (fuzz tests) and a
  rules-aware `HeuristicBot` (the GUI's opponent)
- `keyforge/replay.py` — compact, exact game records: each decision stored as
  the index of the chosen option, replayable from the config's seed
- `text_ui/` — a playable command-line client (`python -m text_ui.main`)
- `sim/` — headless batch simulation (`python -m sim.simulate`)
- `tests/` — unit and interaction tests, `unittest`-based (also runs under `pytest`)

## Bot-first decision API

```python
game = Game(GameConfig(decks=("fignor", "igor"), seed=42))
while not game.is_over:
    d = game.pending_decision
    choice = controllers[d.player].decide(game.view_for(d.player), d)
    game.submit(choice)
result = game.result
```

Rule and effect code is written as Python generators: any step that needs a
player's input does `choice = yield Decision(...)`, so resolution can pause
mid-effect and resume exactly where it left off once `submit()` is called
again. `Decision.options` always holds only the currently legal choices.

## Running things

```bash
# Play a game (default: you vs. a random bot)
python -m text_ui.main --p1 human --p2 random --seed 1

# Batch-simulate random vs. random with invariant checks
python -m sim.simulate --games 1000 --p1-deck fignor --p2-deck igor --check-invariants

# Tests
python -m unittest discover -s tests
```

## Notable design points

- **Decision-pausing engine**: `Game._run()` is a generator driving the whole
  game; `Game.submit()` sends a validated choice into it via `.send()` and
  captures the next yielded `Decision` (or the final result on `StopIteration`).
- **Nested resolution**: rather than an explicit frame stack, a queued
  effect's own resolution (e.g. Wild Wormhole playing a card) is a nested
  generator call — Python's call stack gives the same "inner resolution
  finishes before the outer one continues" behavior the plan describes.
- **Play-trigger ties**: `Game._play_resolution` implements the tie between a
  card's own Play effect and the play-trigger check exactly as worked out for
  Wild Wormhole + Library Access in the plan (see `tests/test_wild_wormhole.py`).
- **Destroy pipeline**: `Game.destroy_cards` tags, resolves all `Destroyed`
  effects (ties ordered by the active player), then moves every tagged card
  at once — to the discard pile, unless a card set `destined_zone` (Bad Penny).
- **Card-count invariant**: `sim/simulate.py`'s `check_invariants` tallies
  every card in the game by its **owner**, not by whichever zone/side it
  currently sits on — important since upgrades and destroy targets can end
  up under the other player's control.
