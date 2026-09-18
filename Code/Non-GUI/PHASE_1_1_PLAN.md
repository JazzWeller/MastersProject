# Phase 1.1 (Archon) — Non-GUI Implementation Plan

**Scope:** KeyForge Phase 1.1, Archon format: each player plays their own deck, Fignor or Igor. Everything runs as text, with no images or animation. The main users are **bots**. Humans can also play through a text interface for testing.

**Sources:** *Master's Project specs and guide.pdf* (updated 9/14) and the card images in `Phase 1/Cards/`. Every clarification question has been answered; section 9 lists the decisions.

**Status:** Final. Ready to implement.

---

## 1. Technology choices

| Decision | Choice |
|---|---|
| Language | Python 3.12, used for every part of the project |
| Dependencies | Standard library only |
| Tests | `unittest` (also runs under pytest) |
| Location | `Code/Non-GUI/`; the future GUI in `Code/GUI/` imports the engine |
| Randomness | One seeded `random.Random` per game, so any game can be replayed exactly from its seed and choice log |

---

## 2. Architecture

```
Code/Non-GUI/
  keyforge/                    # engine package: no print()/input() anywhere
    enums.py                   # House, CardType, Trigger, Zone, Flank, DecisionKind
    config.py                  # GameConfig(decks, first_player=None, seed=None, max_turns=None)
    cards/
      card.py                  # Card + TypeObject (ActionType, ArtifactType, CreatureType, UpgradeType)
      card_data.py             # definitions for all 49 Phase 1 cards
      decks.py                 # Fignor and Igor decklists (3 pods x 12 cards)
    zones.py                   # Deck (deque), DiscardPile (stack), Hand, Archive, PurgedZone, PlayArea
    player.py                  # Player variables + get functions
    effects/
      effect_object.py         # EffectObject, DurationEffect, InsteadEffect, TriggerEffect
      steps.py                 # basic steps: gain, steal, capture, draw, archive, discard, discard_random,
                               #   purge, put, deal_damage (armor), heal, destroy, sacrifice, return_to_hand,
                               #   shuffle_into_deck, ready, exhaust, use_creature
      generic.py               # generic effects (Steal X, Archive X, DrawUpToLimit ±X, ...)
      named.py                 # named effects (Arise, Gateway to Dis, ...)
    current_queue.py           # CurrentQueue: stack of priority-queue frames (see 3.5)
    actions.py                 # PlayCard, DiscardCard, UseAction, UseOmni, Reap, Fight, EndTurn
    decision.py                # Decision (what the engine needs next) + choice validation
    game.py                    # Game: setup, turn order, legal actions, rules, submit()
    view.py                    # PlayerView: read-only state with hidden information removed
    log.py                     # structured event log
  bots/
    base.py                    # Controller interface: decide(view, decision) -> choice
    random_bot.py              # uniformly random legal choices
  text_ui/
    main.py                    # play a game: any mix of human/bot seats
    human_controller.py        # Controller that reads stdin
    render.py                  # board, hand, zone, and log text
    commands.py                # info commands (board, hand, discard, card <name>, effects, ...)
  sim/
    simulate.py                # headless batch runs: N games, win-rate summary, invariant checks
  tests/
    test_zones.py  test_player.py  test_effects_framework.py  test_queue_and_interactions.py
    test_setup_and_first_turn.py  test_turn_order.py  test_fight_and_armor.py  test_destroy.py
    test_draw_and_chains.py  test_rule_of_six.py  test_decision_api.py  test_replay.py
    test_cards_dis.py  test_cards_logos.py  test_cards_shadows.py  test_simulation.py
  README.md
```

### 2.1 Bot-first decision API
The engine runs on its own until a player must choose something. It then stops and exposes a `Decision`. Bots, the text UI, and later the GUI all drive the game with the same loop:

```python
game = Game(GameConfig(decks=("fignor", "igor"), first_player=None, seed=42, max_turns=None))
while not game.is_over:
    d = game.pending_decision                  # Decision(player, kind, prompt, options, min_n, max_n)
    choice = controllers[d.player].decide(game.view_for(d.player), d)
    game.submit(choice)                        # validates the choice, then continues to the next decision
result = game.result                           # winner, turns played, reason ("3 keys" / "turn limit")
```

- **Decision kinds:**
  - `MULLIGAN`, `CHOOSE_HOUSE`, `TAKE_ARCHIVE`
  - `CHOOSE_ACTION` (the legal-action list)
  - `CHOOSE_CARDS` (targets, with min and max counts), `CHOOSE_FLANK`, `CHOOSE_HOUSE_FOR_EFFECT`
  - `ORDER_EFFECTS` (breaks ties in the CurrentQueue), `YES_NO`
- **How pausing works:** rule and effect code is written as Python generators. Each step that needs input does `choice = yield Decision(...)`, so effects can pause and resume mid-resolution.
- **Legal options only:** `Decision.options` always contains just the legal choices, so bots never have to check rules themselves.
- **No lookahead needed:** the Phase 1 algorithm only picks from the current options, so the engine has no `clone()` or state-copy feature. That keeps the engine simpler.
- **Replay for debugging:** each game records its seed and its list of choices. Feeding the same seed and choices back in reproduces the game exactly, which makes bugs found in simulations easy to reproduce.
- **Why this design:** a GUI's event loop can't sit waiting on a blocking callback. The pause-and-submit model therefore serves bots, the text UI, and the GUI equally well.
- **`game.view_for(player)`:** hides the opponent's hand and archive contents and the deck order, but shows counts. Every bot receives this view, so no bot can see hidden information.

---

## 3. Data model (follows the spec's "Classes" and "Variables" sections)

### 3.1 Card and TypeObject
- **`Card`:** `id`, `name`, `house`, `tags`, `type`, `aember_on_play`, `aember_captured`, `image` (path only, unused here), `owner`, `controller`, `type_object`, plus the per-card variables `Elusive`, `IgnoreElusive`, `Skirmish`, `CanBeUsed`, `Exhausted`, and a `destroyed` tag.
- **Instances:** each physical copy gets a unique `instance_id`. The printed `id` is shared between copies (Doc Bookton = 311).
- **`TypeObject` subclasses:**
  - `ActionType`: discarded after it resolves.
  - `ArtifactType`: enters play exhausted.
  - `CreatureType`: `power`, `armor`, `damage`, `armor_used_this_turn`, `upgrades[]`; enters exhausted; can reap and fight.
  - `UpgradeType`: attaches to a creature; its effects count as the creature's.
- **Get functions:** values that effects can change are only read through them, e.g. `get_power`, `get_armor`, `is_elusive`, `has_skirmish`, `can_be_used`.

### 3.2 Zones
| Zone | Implementation | Visibility |
|---|---|---|
| Deck | `collections.deque` (top = left end) | Hidden. Both players can list the starting 36-card decklist, unordered. |
| DiscardPile | stack (list) | Both players |
| Hand | list | Owner only; opponent sees the count |
| Archive | list | Owner only; opponent sees the count |
| PurgedZone | list | Both players |
| PlayArea | two ordered lists: creatures and artifacts. Upgrades are stored on their creature. | Both players |

Empty deck:
- **Drawing** from an empty deck shuffles the discard pile into the deck first. If both are empty, the draw stops.
- **Top-card effects** that don't say "draw" (Wild Wormhole) do **not** shuffle the discard pile in. The effect just does nothing.

### 3.3 Player
- **Variables** use the spec's default values: `CanKeyForge`, `CanPlayActions`, `CanPlayCreatures`, `CardDrawModifier`, `CardPlayedLimit`, `CardsPlayed` (card name → count, reset each turn), `Chains` (0–24), `DrawUpToLimit`, `KeyForgeCost`, `NonLogosCardsPlayable` (default 0), `HouseSelection`. Each player also has `aember`, `keys`, `cards_played_or_discarded_this_turn` (for the first-turn rule), and all zones.
- **Get functions:** each `get_*` starts from the base value and applies the matching `DurationEffect`s in the order they were added, checking `PlayerAffected` and `Conditional`. The key cost can't go below 0.

### 3.4 Effect objects
- `DurationEffect(remaining_duration | INFINITE, player_affected, conditional, variable, op(+,-,=), value)`, `InsteadEffect(...)`, and `TriggerEffect(...)` all extend `EffectObject(source_card, controller)`.
- **ActiveEffectList** holds every lasting effect. **TriggerEffectList** and **InsteadEffectList** are filtered views of it.
- **Durations** go down by 1 at the end of every turn and are removed at 0: "1" means until the end of this turn, "2" means through the opponent's next turn. Infinite effects are removed when their source card leaves play.
- **Passive effects** (Mother, Succubus, Howling Pit, Elusive, Skirmish, Ember Imp, Titan Mechanic) are registered when the card enters play and removed when it leaves.
- **Step failure:** if a step does nothing at all, the rest of that effect stops. A partly successful step counts as success, so later steps still run.

### 3.5 CurrentQueue, resolution order, and triggers
The spec's Wild Wormhole + Library Access interaction requires the model below.

- **Frames:** the CurrentQueue is a **stack of frames**, and each frame is a priority queue (`heapq` of `(tier, sequence, item)`). When an item's resolution causes new effects (Wild Wormhole playing a card, a destroy during a fight), they go into a **new frame**. That frame resolves completely before the outer frame continues.
- **Tiers:** `BEFORE_FIGHT < FIGHTING < AFTER (fight/reap) < DESTROYED < END`. Items that are in the same tier and came from the same event are **tied**, and the active player picks the order (`ORDER_EFFECTS`). No prompt appears when only one item is present.
- **Playing a card:**
  1. Gain the æmber bonus.
  2. Queue two tied items: the card's **Play effect** and a **play-trigger check** for that card.
  3. The play-trigger check looks up "when a card is played" triggers **at the moment it resolves, not when it is queued**.
  4. A card never triggers its own Library Access. If no play triggers exist when the card is played, the Play effect resolves first and the player isn't asked.
- **The two spec examples then work out as the spec says:**
  - *Wild Wormhole plays Library Access:* Wormhole's effect runs first (there are no triggers yet). That registers Library Access's trigger. Then Wormhole's play-trigger check resolves, finds it, and draws 1 card.
  - *Library Access already in play, then Wild Wormhole:*
    1. Gain Wormhole's æmber.
    2. The player chooses whether Wormhole's effect or the Library Access draw resolves first.
    3. Inside a new frame: gain the played card's æmber.
    4. The player chooses whether that card's effect or its Library Access draw resolves first.
- **Instead effects:** checked right before each destroy, move, or forge step (Bad Penny).

---

## 4. Rules engine (`game.py`)

### 4.1 Setup (official rules)
1. Choose each player's deck (Fignor or Igor; both players may choose the same one).
2. Choose the first player: `GameConfig.first_player` if given, otherwise random from the seed. The first player is **Player 1**.
3. Build 36 cards per player, shuffle, and set æmber, keys, and chains to 0.
4. Deal **7 cards to Player 1 and 6 to Player 2**.
5. Player 1 may mulligan: shuffle the hand into the deck and draw **one fewer card** than before. Then Player 2 may do the same.

### 4.2 Turn order
1. **Forge a key** if `get_can_key_forge()` and æmber ≥ `get_key_forge_cost()`. At 3 keys the game ends.
2. **Choose a house.** If `get_house_selection()` returns a value (Control the Weak), use it. Otherwise the player chooses. Then set `CanBeUsed` on that house's controlled cards.
3. **Archive (optional):** if the archive isn't empty, ask `TAKE_ARCHIVE` (yes/no) right away. Taking it moves **all** archived cards to hand. The choice can't be changed later this turn.
4. **Main loop:** `CHOOSE_ACTION` → resolve → win check → repeat until `EndTurn`. `EndTurn` is always an option.
5. **Cleanup:**
   - Ready all controlled cards, clear `CanBeUsed`, reset `CardsPlayed` and the first-turn counter, and clear `HouseSelection`.
   - Refresh armor on all creatures and count down durations.
6. **Draw** (see 4.6).
7. **Turn limit:** if `max_turns` is set and reached, the game ends as a draw ("turn limit"). The default is no limit.

### 4.3 Legal actions
| Action | Conditions |
|---|---|
| **Play from hand** | Card is the selected house, **or** it's non-Logos and `NonLogosCardsPlayable > 0`. `CanPlayActions` / `CanPlayCreatures` allow it. The `CardPlayedLimit` isn't reached. Rule of 6 allows it. Upgrades need a creature in play. The first-turn rule allows it. |
| **Discard from hand** | Card is the selected house, and the first-turn rule allows it |
| **Use artifact Action** | `CanBeUsed`, not exhausted, rule of 6 |
| **Use Omni** | Not exhausted, rule of 6; house doesn't matter |
| **Reap / creature Action** | `CanBeUsed`, not exhausted, rule of 6 |
| **Fight** | Same as Reap, and the opponent has at least one creature |
| **End turn** | Always |

- **First-turn rule:** on the first player's first turn, only **one** card may be played or discarded **from hand**. Cards that the played card plays or discards through its effects don't count. Using cards already in play and taking the archive are still allowed.
- **Rule of 6:** counts plays and uses per card name, shared across copies. Discards don't count. Cards played by Wild Wormhole count.
- **Phase Shift allowance:** playing a non-Logos card while `NonLogosCardsPlayable > 0` and outside the selected house uses 1. A non-Logos card played by Wild Wormhole also uses 1, but the count never goes below 0.

### 4.4 Playing a card
1. Remove the card from hand and record the play (`CardsPlayed`, `CardPlayedLimit`, first-turn counter, Phase Shift allowance).
2. By type:
   - **Action:** æmber bonus → Play effect / play-trigger check (3.5) → discard, unless the effect already moved the card.
   - **Creature:** choose a flank → place exhausted → register passives → æmber bonus → Play effect / play-trigger check.
   - **Artifact:** place exhausted → register passives → æmber bonus → Play effect / play-trigger check.
   - **Upgrade:** choose any creature (either side) → attach → æmber bonus → play-trigger check.

### 4.5 Reap, fight, damage, destroy, leaving play
- **Reap:** exhaust, gain 1 æmber, queue After Reap effects.
- **Fight:** exhaust the attacker, then choose an enemy target.
  1. **Before Fight:** Assault (summed) and Hazardous (summed) resolve, then a destroy check. If either creature was destroyed, skip to After Fight.
  2. **Fight phase:** skipped if the **Target** is Elusive and hasn't been a fight Target yet this turn, unless `IgnoreElusive` is set. Otherwise damage is dealt both ways (Skirmish protects the attacker), Fighting effects resolve, and a destroy check runs.
  3. **After Fight:** runs only if the attacker survived.
- **Damage (with armor):** `deal_damage(creature, amount)` first subtracts the armor the creature has left this turn (`get_armor() - armor_used_this_turn`), then adds the rest to its damage. Armor refreshes when each turn ends. Every Phase 1 creature has 0 armor, but the function is built and tested with synthetic cards. `heal` lowers damage, never below 0, and returns how much it actually healed (used by Guardian Demon).
- **Destroyed:** damage ≥ power.
- **Destroy pipeline:**
  1. Tag the card. Cards already tagged are ignored.
  2. Resolve all Destroyed effects, including ones caused by destroys that happen during this.
  3. Check Instead effects.
  4. Move all tagged cards to their owners' discard piles at the same time.
- **Leaving play** (for any reason):
  - Captured æmber goes to the opponent of the card's controller.
  - Upgrades go to their owners' discard piles.
  - The card's passive and infinite effects are removed, and its damage, exhaustion, armor use, and flags reset.

### 4.6 Draw step
```
base = max(0, get_draw_up_to_limit() - hand_size)
if base == 0: no draw, chains unchanged
n = base + get_card_draw_modifier()
if chains > 0: n -= ceil(chains / 6); chains -= 1
draw max(0, n)
```
Draws from effects use `max(0, X + CardDrawModifier)` and ignore chains.

---

## 5. Card effects (all 49), per the updated spec

Targeting: "a creature" means any creature on either side, "friendly" means yours, and "enemy" means the opponent's.

### Dis (17)
| Card | Implementation |
|---|---|
| Arise | Choose one of your houses. Put that house's creatures from your discard pile into your hand. Gain 1 chain. |
| Control the Weak | Choose one of the enemy's houses. DurationEffect(2) forces the enemy's `HouseSelection`. |
| Creeping Oblivion | Choose **one** discard pile, then purge 0–2 cards from it |
| Dominator Bauble | Action: choose a friendly creature, then use it if possible (reap / fight / its Action, ignoring house) |
| Dust Imp (2) | Destroyed: gain 2 |
| Ember Imp (2) | Passive: enemy `CardPlayedLimit` = 2 |
| Gateway to Dis | Destroy every creature in one batch. Gain 3 chains. |
| Guardian Demon (4) | Play/After Fight/After Reap: heal up to 2 from one creature, then deal the amount **actually healed** to another creature |
| Lash of Broken Dreams | Action: DurationEffect(2) enemy `KeyForgeCost` +3 |
| Library of the Damned | Action: archive 1 |
| Lifeward | Omni: sacrifice Lifeward, then DurationEffect(2) enemy `CanPlayCreatures` = False |
| Pit Demon (5) | Action: steal 1 |
| Shooler (5) | Play: if the enemy has ≥ 4, steal 1 |
| Snudge (4) | After Fight/After Reap: return any artifact, or a flank creature (either side), to its owner's hand |
| Succubus (3) | Passive DrawUpToLimit Enemy −1 |
| The Terror (5) | Play: if the enemy has 0, gain 2 |
| Three Fates | Destroy the 3 most powerful creatures. The active player chooses among ties. |

### Logos (14)
| Card | Implementation |
|---|---|
| Doc Bookton (5) | After Reap: draw 1 |
| Help From Future Self | Search the deck, then the discard pile, for a Timetraveler. If found, reveal and take it. Shuffle the discard pile into the deck. |
| Labwork | Archive 1 |
| Library Access | Purge itself, then TriggerEffect(1): whenever you play another card, draw 1 |
| Library of Babble | Action: draw 1 |
| Mother (5) | Passive DrawUpToLimit Player +1 |
| Phase Shift | DurationEffect(1): `NonLogosCardsPlayable` +1 |
| Quixo the Adventurer (3) | Passive Skirmish. After Fight: draw 1. |
| Scrambler Storm | DurationEffect(2) enemy `CanPlayActions` = False |
| Sloppy Labwork | Archive 1, then discard 1 card of your choice |
| The Howling Pit | Passive DrawUpToLimit Player, Enemy +1 |
| Timetraveler (2) | Play: draw 2. Action: shuffle this card into your deck. |
| Titan Mechanic (6) | While on a flank: `KeyForgeCost` −1 for both players |
| Wild Wormhole | Look at the top card. If the deck is empty, do nothing (no reshuffle). Play the card, ignoring house. If it can't be played (`CanPlayCreatures` = False, an upgrade with no creatures, ...), put it back on top. |

### Shadows (18)
| Card | Implementation |
|---|---|
| Bad Penny (1) | InsteadEffect: when destroyed, goes to hand instead of discard |
| Bait and Switch | If the enemy has more æmber than you, steal 1; repeat while that is still true (printed card; the spec stole once unconditionally). |
| Booby Trap | Choose a non-flank creature: deal 4 to it and 2 to each neighbor |
| Duskrunner (upgrade) | Host creature gains After Reap: steal 1 |
| Ghostly Hand | If the enemy has exactly 1, steal 1 |
| Lights Out | Return up to 2 enemy creatures to hand |
| Miasma | DurationEffect(2) enemy `CanKeyForge` = False |
| Nerve Blast | Steal 1. If that worked, deal 2 to a creature. |
| Noddy the Thief (2) | Elusive. Action: steal 1. |
| Old Bruno (3) | Elusive. Play: capture 3. |
| One Last Job | Purge each friendly Shadows creature, then steal 1 for each purged |
| Oubliette | Purge a creature with power ≤ 3 |
| Pawn Sacrifice | Sacrifice a friendly creature. If you do, choose 2 different creatures and deal 3 to each (only 1 if just one exists). |
| Relentless Whispers | Deal 2 to a creature. If that destroys it, steal 1. |
| Silvertooth (2) | Play: ready itself |
| Subtle Maul | Action: enemy discards 1 card at random |
| Too Much To Protect | Steal max(enemy æmber − 6, 0) |
| Urchin (1) | Elusive. Play: steal 1. |

Assault, Hazardous, Fighting effects, `IgnoreElusive`, and armor aren't used by any Phase 1 card. They are still built and tested with synthetic test-only cards.

---

## 6. Bots and simulation
- **`bots/base.py`:** `Controller.decide(view, decision) -> choice`. This is the only interface the Phase 1 algorithm needs.
- **`bots/random_bot.py`:** seeded, picks uniformly from `decision.options`. Used as the baseline opponent and for fuzz testing.
- **`sim/simulate.py`:**
  - Command: `python -m sim.simulate --games 1000 --p1 random --p2 random --p1-deck fignor --p2-deck igor [--first p1|p2] [--seed N] [--max-turns N]`
  - Prints win rates, average turns, first-player win rate, and draws caused by the turn limit.
  - `--check-invariants` checks after every decision:
    - each player's 36 cards are all accounted for
    - æmber ≥ 0 and chains are within 0–24
    - no exceptions are raised
  - `--save-log` writes a game's seed and choices so it can be replayed exactly.

## 7. Text UI (human testing)
- **Command:** `python -m text_ui.main [--p1 human|random] [--p2 human|random] [--p1-deck ...] [--p2-deck ...] [--first p1|p2] [--seed N] [--max-turns N]`
- **Prompts:** each decision is a numbered menu with a status line (keys, æmber, chains, and zone counts). When both seats are human, a "pass to the other player" screen hides the hand between turns.
- **Info commands** don't use up a decision:
  - `board`, `hand`, `archive`
  - `discard [me|opp]`, `purged [me|opp]`, `decklist [me|opp]`
  - `card <name>`, `effects`, `log [n]`, `help`, `quit`
- **Game log:** events print as they happen. A bot seat's choices are printed too, so a human can watch a bot play.

## 8. Testing strategy and milestones

**Tests:**
1. **Units:** zones, get functions, durations, instead/trigger lists, queue frames and tie ordering.
2. **Rules:** setup (7/6 cards, mulligan to one fewer, choosing the first player), first-turn rule, optional archive pickup, key forging, draw and chains, rule of 6, CardPlayedLimit, fight phases, Elusive (Target only), Skirmish, Assault, Hazardous, armor, destroy pipeline, turn limit.
3. **Spec interactions:** Wild Wormhole with Library Access (both orders), Wild Wormhole with Phase Shift, Wild Wormhole with an empty deck.
4. **Every card** has at least one test built with a scripted controller. Extra tests cover edge cases: Bait and Switch looping, Gateway to Dis with Dust Imp and Bad Penny, Guardian Demon on an undamaged creature.
5. **Decision API and replay:** only legal options are offered, invalid submits are rejected, and replaying a saved seed and choice log reproduces the same game.
6. **Simulation:** 1,000 random-vs-random games across every deck pairing, with invariant checks on.

**Milestones:**
1. Scaffolding, enums, Card/TypeObject, all 49 card definitions, decklists
2. Zones, Player, get functions
3. Effect objects, ActiveEffectList, CurrentQueue frames, basic steps
4. Decision API: generator engine, `submit`, `view_for`, choice log and replay
5. Core rules: setup, turn order, legal actions, play/discard, reap, fight, armor, destroy, draw, win and turn limit
6. Card effects: generic, then Dis → Logos → Shadows, plus the interaction tests
7. Random bot and simulation harness; fix whatever the fuzzing finds
8. Text UI with human controller
9. README and a final manual playthrough

Out of scope: GUI, Reversal (1.2), Adaptive bidding (1.3), and the Phase 1 algorithm itself.

---

## 9. Decisions log (every question answered)

**Round 1**
- Python for everything, in `Code/Non-GUI`.
- Doc Bookton's ID is 311.
- Where the spec and the printed card disagreed, the card wins (the guide now matches the cards).
- Bots are the main users; humans play for testing.
- The first player can be set in setup and is random by default.
- Official setup and first-turn rules.
- Draw-step formula as in 4.6.
- A creature is destroyed when damage ≥ power.
- The armor function is built even though all Phase 1 armor is 0.
- A step that only partly works still counts as success.
- Upgrades can attach to either side and go to discard when the host leaves play.
- Taking the archive is optional (official rule).
- The deck uses a deque.
- The rule of 6 counts plays and uses (not discards), including cards played by Wild Wormhole.
- A card played by Wild Wormhole still follows play restrictions.
- No turn limit by default, but one can be set.

**Round 2**
1. The bot only picks from the current options, so there's no clone or lookahead.
2. Phase Shift covers only playing cards from hand, not using cards already in play.
3. If Wild Wormhole plays a non-Logos card when the Phase Shift allowance is 0, the card is still played and the allowance stays at 0.
4. The first-turn rule limits only plays and discards from hand. Reaping, fighting, using artifacts, and taking the archive are still allowed.
5. Effects are resolved using the model in 3.5: nested frames, play triggers checked when they resolve, and a card never triggers its own Library Access.
6. Armor refreshes at the end of every turn, for both players.
7. Taking the archive is a yes/no question right after choosing a house.
8. Creeping Oblivion purges 0–2 cards from one discard pile of the player's choice.
