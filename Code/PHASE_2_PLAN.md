# Upgrade Plan: Phase 1.2 (Reversal), 1.3 (Adaptive), 2.0 (all CotA Dis/Logos/Shadows cards, any deck), 2.1 (Alliance)

## Context

The client (engine `Code/Non-GUI/keyforge` + pygame GUI `Code/GUI`) currently implements **Phase 1.1**. That means 49 cards, two hard-coded decks (Fignor, Igor) and art from `Phase 1/Cards`. The spec (*Master's Project specs and guide.pdf*, p.2) defines the next versions:

- **Phase 1.2 Reversal**: "Swap and play"
- **Phase 1.3 Adaptive**: "Play, Swap and Play, Bid and Play". The deck that won both rounds is bid on. The deck's owner starts the bidding at 0 chains, and bids alternate until someone stops or the bid reaches 24 chains.
- **Phase 2**: "Three houses, any deck from those so all cards (Logos, Shadow, Dis)"
- **Phase 2.1 Alliance**: "Each house from each deck can be used separately to form a new deck"

**Card pool (confirmed with you):** every *Call of the Archons* (set 341) card in Dis, Logos and Shadows. I checked it: the 159 images in `Phase 2/Cards/` are exactly CotA's 54 Dis, 53 Logos and 52 Shadows cards, shown with newer reprint art. File names like `907-057.png` are the reprint's set and number. The pool is the 49 Phase 1 cards plus **110 new cards**. There is no armor anywhere in the pool. CotA has no Enhance, so the existing "Æmber on Play" bonus model is enough.

**Decks (confirmed):** a deck builder over the 159-card pool, plus bundled preset decks (Fignor, Igor and a few more) stored as JSON.

**Research done, with sources:**
- **Card text.** Card names, stats and text come from **keyteki's card database** (`github.com/keyteki/keyteki-json-data`, `packs/CotA.json`). I cross-checked them against the official Master Vault API for sets 907 and 964, and read three card images directly (Master of 1/2/3).
- **Rules and errata.** The official **KeyForge Master Rulebook v18.3** (Ghost Galaxy, Feb 2026) provided the errata section, the glossary, and the card-specific FAQ.
- **Match formats.** Reversal and Adaptive come from the spec. The rulebook's chain-bidding, chains and mulligan rules fill the gaps, cross-checked against keyteki's implementation (`server/game/gamesteps/setup/{ChainBiddingPrompt,FirstPlayerSelection,setupphase}.js`, `GameActions/DrawAction.js`).
- **Archon Arcana is not used** (your decision).

**Rules source order (your standing rule):** official errata > card text (keyteki) > rulebook glossary/FAQ > spec PDF. Where the spec contradicts official rules, flag it and ask; don't silently pick one.

### Official errata that affects the pool
| Card | Canonical text to implement |
|---|---|
| *All* "Fight:" / "Reap:" | Read as "After Fight:" / "After Reap:" (Guardian Demon, Doc Bookton, Ozmo, Master of 1/2/3, Silent Dagger, Duskrunner, Spectral Tunneler's granted ability, Rocket Boots, Selwyn, Nexus, Neutron Shark, Replicator) |
| Bait and Switch | "…steal 1. Repeat the **preceding effect** if your opponent still has more than you." (it repeats once at most, per FAQ) |
| Experimental Therapy | "This creature **may be used as if it belonged to the active house**. Play: Stun and exhaust this creature." (replaces "belongs to all houses") |
| Library Access | "…each time you play **another** card, draw a card. **Purge Library Access.**" |
| Magda the Rat | "Elusive. Play: Steal 2. If Magda the Rat leaves play, your opponent steals 2." |
| Tendrils of Pain | "Deal 1 to each creature. Deal **4 to each creature instead** if your opponent forged a key on their previous turn." |

**Safe Place (decided: CotA version):** the reprint art says "spend … as if it were in your pool", but CotA and the rulebook FAQ say "spend when forging keys". Implement the CotA wording and show the official text in the card inspector. The other reprint wording changes (Sacrifice→Destroy, "opponent's"→"enemy", "Leaves Play:" → "If … leaves play") are functionally identical. The rulebook's "as if it were yours" rule makes Sacrifice and Destroy equivalent here.

---

## Deliverable docs (written first, committed with the code)
- `Code/PHASE_2_PLAN.md`: this plan.
- `Code/PHASE_2_CARD_RULINGS.md`: one entry per pool card. Each entry has the canonical text, applicable errata, and the rulebook FAQ rulings quoted as test cases. Examples:
  - Stealer of Souls: no trigger if both fighters die.
  - Tolas with Gateway to Dis: no gain. Tolas with Bad Penny: gain.
  - Key Hammer unforges exactly one key.
  - Two Phase Shifts give two allowances.
  - Neutron Shark stops once it leaves play.
  - Masterplan plays the card before it is destroyed.
  - Magda the Rat with Gateway to Dis: captured Æmber moves first.
  - Spangler Box *puts* cards into play (ignores play restrictions).
  - Reverse Time flips the deck's order.
  - Pitlord (must) vs Restringuntus (cannot): cannot wins.
  - Faygin on an enemy Urchin sends it to its owner's hand.
  - Replicator: you choose which After Reap ability it copies.
  - Nexus + Spectral Tunneler: the granted reap ability resolves.
  - Shadow Self: redirect after armor; poison applies.
  - Gabos Longarms vs elusive creatures.
  - Bouncing Deathquark repeat condition.
  - Wild Wormhole + Library Access ordering (already implemented).

---

## v1.2 / v1.3: Match formats (Reversal, Adaptive)
These formats don't depend on the new cards. They work with Fignor/Igor today and with any Phase 2 or Alliance deck later, so this milestone can go first.

**Decisions:**
- **Starting chains shrink the opening hand (official rule, you confirmed; overrides the spec's step-5-only wording).** Opening hand = 7 or 6 minus ceil(chains / 6), then shed 1 chain. A mulligan draws one fewer card than the current hand and does not shed another chain.
- **First player.** Game 1 is random. In each later game, the loser of the previous game chooses to go first or second. The rules don't cover this, so it follows keyteki's `FirstPlayerSelection`.
- **Reversal** is a single game in which each player uses the opponent's deck. In-game, the player using a deck is its owner (owner/controller rules are unchanged).
- **Adaptive:**
  - Game 1: each player uses their own deck.
  - Game 2: decks are swapped.
  - If one player won both games, they win the match 2–0.
  - Otherwise the same deck won both games, and game 3 bids on that deck. Its owner opens at 0 chains, and the opponent must bid higher or pass; bids alternate. The player who passes loses the bid, and a bid of 24 ends bidding.
  - The bid winner plays that deck starting with the bid's chains (max 24, the rulebook cap). The loser plays the other deck with 0 chains.

**Engine** (`Code/Non-GUI/keyforge/`):
- New `match.py` with `Match(format, decks, seed)`. It is a generator driver with the same `pending_decision` / `submit()` / `view_for()` API as `Game`, so the bots, text UI and GUI drive a match exactly as they drive a game. It creates each `Game` in turn, forwards its decisions, and records results, the deck used by each seat, the first player and final chains.
- `GameConfig` gets `starting_chains={1: n, 2: n}`. `Game._setup` applies the chain penalty to the opening draw, sheds one chain, and logs `shed_chain` (source "opening hand"). `_maybe_mulligan` is unchanged (draws one fewer, no shed).
- New decision kinds:
  - `BID_CHAINS`, with options `"pass"` or integers from current+1 to 24
  - `CHOOSE_FIRST_PLAYER`, with options "first" / "second"
- Per-game seeds are derived from the match seed, so a whole match replays exactly from one seed plus its choice record.

**Bots:** `HeuristicBot` bids up to a configurable ceiling (default 4 chains, scaled by how decisively the deck won) and chooses to go first. `RandomBot` makes random legal choices, which also fuzzes the bidding.

**Text UI and sim:**
- `text_ui.main --format archon|reversal|adaptive`
- `sim.simulate --format adaptive --matches N` reports match win rates, how often the match reaches a third game, and the average winning bid

**GUI** (`Code/GUI/gui/`):
- The menu gets a **Format** selector (Archon / Reversal / Adaptive).
- A **Match interstitial** scene between games shows the score so far, which deck each seat will use next, and whether bidding happens.
- A **First-player choice** screen appears for the previous game's loser.
- The **Bidding** scene has Raise (+1 / custom), Pass, the current bid, and a preview ("with N chains your opening hand is X cards"). Hot-seat games show the pass-device curtain between bids.
- A **Match over** screen shows each game's result.
- The HUD's chain tracker and shed-chain animation are reused.

**History and replay:** a new `matches` table (format, deck lists, seed, match-level choice record) with game rows linked to it. Past Games groups games by match, and replay can step through a whole match.

**Tests:**
- Opening hand with chains (0/1/6/7/13/24) and the mulligan-with-chains case.
- Reversal swaps decks and ownership correctly.
- Adaptive: a 2–0 match ends after two games; a 1–1 split goes to bidding; bidding covers owner-opens-at-0, raise, pass, the 24 cap, and bid-winner assignment.
- The previous game's loser chooses the first player.
- A full match replays exactly from its seed and record.
- GUI: a human-vs-bot Adaptive playthrough through real click events, including the bid and first-player screens.

---

## v2.0, Milestone A: Card data (engine)
- **`Code/Non-GUI/keyforge/cards/data/cota_pool.json`** (generated, committed; no network at runtime). Built by `Code/Non-GUI/tools/build_card_data.py` from keyteki `CotA.json` + the `Phase 2/Cards` file-name mapping + a hand-kept errata table. One entry per card:
  - CotA number (becomes the card `id`)
  - name, house, type, power, armor, Æmber
  - traits, keywords
  - printed text and canonical text
  - image path (`Phase 2/Cards/<Dis|Logos|Shadow>/<file>.png`)
- **`CardDef`** (`keyforge/cards/card.py`): add `number`, `traits`, `keywords` (elusive, skirmish, taunt, poison, hazardous X, versatile) and `text`. Also add a general ability list (see B3). The stats come from the JSON, and `card_data.py` only attaches effect functions by name.
- **Split effects** into `keyforge/effects/named/{dis,logos,shadows}.py`. The current `named.py` is already 510 lines. Phase 1 effects move into these files unchanged.
- **Audit the 49 existing cards** against the canonical text (e.g., Guardian Demon: "Heal up to 2 … deal that amount"). List any spec-vs-official conflicts in the rulings doc for you to decide.
- **Registry test:** every one of the 159 names has a definition, art and canonical text.

## v2.0, Milestone B: Rules engine generalization (`keyforge/game.py`, `player.py`, `effects/`)
Each item names the cards that need it.

1. **Computed card stats (get functions, per the spec's design):**
   - `get_power`: base + `+1 power` counters (Eater of the Dead) + upgrades (Flame-Wreathed +2) + lasting effects.
   - `get_armor`: modifiers such as Red-Hot Armor's loss, with used armor lost first.
   - `get_keywords`: printed + upgrades (Ring of Invisibility) + grants (Flame-Wreathed hazardous 2).
   - `get_houses`: Sneklifter re-housing, Versatile / Experimental Therapy "use as if active house".
   - Elusive and Skirmish move from per-card passives to keyword lookups.
2. **Keywords:**
   - **Taunt** restricts fight targets (Pitlord, Truebaru).
   - **Poison** destroys when power damage is actually dealt, not when armor or other effects prevent it (Macis Asp, Mooncurser).
   - **Hazardous X** (Flame-Wreathed).
   - **Splash**: generalize Booby Trap's special case.
   - **Versatile** (Mack the Knife).
3. **Ability model.** Replace `Card.extra_triggers` with a single `abilities(card)` query covering printed, upgrade-granted and lasting-granted abilities, e.g. Spectral Tunneler's "After Reap: draw" for the turn, Silent Dagger, Duskrunner, Rocket Boots, Transposition Sandals. Replicator picks one After Reap ability to copy. Kinds covered: Play, After Reap, After Fight, Before Fight, Action, Omni, Destroyed, and "leaves play" (Magda).
4. **Event and trigger bus** (extends `TriggerEffect`), with simultaneous triggers ordered by the active player:
   - `after_destroyed` fires when the card leaves play (Soul Snatcher, Tolas).
   - `destroyed_fighting`, where the attacker must still be in play (Overlord Greking, Stealer of Souls, Brain Eater).
   - `artifact_used` (Veylan Analyst), `card_played` including artifacts (Carlo Phantom), `key_forged` (Strange Gizmo, Interdimensional Graft), `end_of_turn` (Shaffles), `before_fight` (Evasion Sigil, Gabos Longarms).
5. **Replacement ("instead") effects** through `InsteadEffect`:
   - reap gain becomes a steal (Dimension Door)
   - damage to a neighbor goes to Shadow Self, after armor
   - leaving play to discard becomes purge (Annihilation Ritual)
   - Bad Penny, Dextre's destination
   - Gabos Longarms' damage redirect
6. **Damage pipeline** (rulebook "Damage"):
   - pending damage → prevention → armor → replacement → dealt → poison
   - all damage from one ability is simultaneous, with one destroy check afterwards (Poison Wave, Tendrils, Twin Bolt Emission, Positron Bolt, Longfused Mines, Booby Trap).
7. **Stun counters** (Experimental Therapy; Ozmo). The next use of a stunned creature only exhausts it and removes the stun; effects that "use" it do the same.
8. **Control changes** via `take_control(card, pid, until=…)`:
   - Cards: Harland Mindlock (until it leaves play), Smiling Ruth, Sneklifter (artifact), Collar of Subordination (upgrade-based), Overlord Greking (put into play from the enemy discard), Spangler Box (passes control of itself).
   - The most recent effect wins, the new controller picks the flank, and cards leaving play go to the owner's zones (already partly implemented).
9. **"Use" framework:** `use_card(card, as_if_yours, ignore_house)`.
   - Covers Dominator Bauble (existing), Deipno Spymaster, Transposition Sandals, Remote Access, Poltergeist, Nexus, Novu Archaeologist and Masterplan's Omni.
   - Restrictions:
     - Skippy Timehog: can't use any cards.
     - Foggify: can't fight.
     - Tentacus: pay 1 to use an artifact.
     - Stun and exhaustion also block use.
10. **House choice:** `must` / `cannot` sets (Pitlord, Control the Weak, Restringuntus). Cannot beats must, and with no legal house the player has no active house.
11. **Play restrictions and costs:**
    - Truebaru: lose 3 in order to play.
    - Treasure Map: no more plays this turn.
    - Customs Office: pay 1 to play an artifact.
    - Speed Sigil and Silvertooth modify how a card enters play (enters ready).
    - Keep Ember Imp, Scrambler Storm, Lifeward and Phase Shift working.
12. **Keys:**
    - Per-player forge history ("forged on their previous turn": Key Hammer, Tendrils of Pain).
    - Unforge (Key Hammer, choose one key).
    - Forge at +N current cost mid-turn (Key of Darkness).
    - Skip the forge step and receive the Æmber the opponent spends (The Sting).
    - Spend Æmber from cards (Pocket Universe, Safe Place): a new decision to split the payment when more than one source can pay.
    - Æmber stored on artifacts goes back to the supply when they leave play.
13. **Zones and information:**
    - Cards under cards, facedown and visible only to the controller (Masterplan).
    - Purged-by tracking (Spangler Box returns its cards by putting them into play).
    - Deck ↔ discard swap that flips the deck's order (Reverse Time); shuffle hand and discard into the deck (Screaming Cave).
    - Reveal the top of the deck (Chaos Portal, Vespilon Theorist, Crazy Killing Machine, Neutron Shark).
    - Look at the opponent's hand (Psychic Bug, Imperial Traitor, A Fair Game). `PlayerView` gets a `revealed` set so hidden cards show only to the right viewer.
14. **New decision kinds:**
    - `CHOOSE_NUMBER` (Dance of Doom)
    - `CHOOSE_MODE` (Knowledge is Power)
    - `AEMBER_SPEND_SPLIT`
    - "may" and "repeat?" prompts reuse `YES_NO` (Master of X, Imperial Traitor, Bouncing Deathquark).
15. **Lasting-effect durations:** add "until X leaves play", per-card scoped effects, and "for the remainder of the turn". These sit next to the existing turn counters in `effect_object.py`.

## v2.0, Milestone C: Implement the 110 new cards
Reuse the generic factories (`effects/generic.py`: gain, steal, capture, archive, draw, damage, duration) wherever they fit. Put named effects in the per-house modules. Order the work by dependency, simplest first, and add each card's tests from the rulings doc in the same commit:
1. **Simple (existing steps only):** Hand of Dis, Key Hammer, Mind Barb, Fear, Hysteria, Charette, Tocsin, Dust Imp-like ones, Random Access Archives, Hidden Stash, Neuro Syphon, Dr. Escotera, Doc-style reapers, Dodger, Umbra, Routine Job, Seeker Needle, Finishing Blow, Anomaly Exploiter, and others like them.
2. **Keyword / stat cards:** Pitlord, Truebaru, Macis Asp, Mooncurser, Flame-Wreathed, Ring of Invisibility, Red-Hot Armor, Eater of the Dead, Mack the Knife, Experimental Therapy.
3. **Triggers and replacements:** Soul Snatcher, Tolas, Overlord Greking, Stealer of Souls, Brain Eater, Veylan Analyst, Carlo Phantom, Strange Gizmo, Interdimensional Graft, Shaffles, Evasion Sigil, Gabos Longarms, Dimension Door, Shadow Self, Annihilation Ritual, Magda the Rat.
4. **Control and use:** Harland Mindlock, Smiling Ruth, Sneklifter, Collar, Spangler Box, Deipno Spymaster, Transposition Sandals, Remote Access, Poltergeist, Nexus, Replicator, Skippy Timehog, Foggify, Tentacus, Customs Office.
5. **Keys and zones:** Key of Darkness, The Sting, Pocket Universe, Safe Place, Masterplan, Reverse Time, Screaming Cave, Chaos Portal, Vespilon Theorist, Crazy Killing Machine, Neutron Shark, Dysania, Psychic Bug, Imperial Traitor, A Fair Game, Mobius Scroll.
6. **Everything else** in the pool. The full pool listing from research, with house, file, stats and text for all 159 cards, is copied into the rulings doc.

Pool-specific notes:
- Imperial Traitor: no Sanctum cards exist in Phase 2 decks, so it only looks at the hand.
- Ozmo: its "Mars creature" text has no targets, so it is effectively just Elusive.
- Sneklifter: the re-housing never matters here, because every deck has the same three houses.
- All three are implemented generally anyway, so Phase 3 won't need rework.

## v2.0, Milestone D: Decks
- **Model** (`keyforge/cards/decks.py`): `Deck(name, pods={House: [12 card names]})`. Validation: exactly 3 houses (Dis, Logos, Shadows for Phase 2), 12 per house, all cards in the pool. Duplicates are allowed, as in real KeyForge.
- **Storage:** bundled presets in `Code/Non-GUI/decks/*.json` (Fignor, Igor, plus about 4 curated decks). User decks go in `Code/GUI/data/decks/` (git-ignored). `GameConfig.decks` accepts a preset name, a file path or a `Deck` object. `player_houses` uses the deck's identity houses.
- **Random legal decks** (`decks.random_deck(rng)`) for the simulator and a "Random" button in the builder.
- **History/replay** (`Code/GUI/gui/history.py`) stores the **full decklists** in each game record, not deck names, so replays survive later deck edits. Add a schema migration for old rows.
- **Text UI and sim:** `--p1-deck <name|path|random>`. Add `sim.simulate --random-decks`.

## v2.0, Milestone E: GUI client upgrade (`Code/GUI/gui/`)
- **Art:** `settings.CARD_ART_DIR` points at the repo root, and paths come from the card JSON (`Phase 2/Cards/...`; note the folder is named `Shadow`). Extend `test_assets.py` to all 159 cards.
- **Card inspector:** add an "Official text" panel whenever the canonical text differs from the art (errata cards, Safe Place, and the Fight/Reap wording).
- **Board state visuals** (`sprites/card_sprite.py`, `hud.py`):
  - stun, +1 power counters, computed power
  - taunt (creature slid forward, per the rulebook)
  - poison / hazardous / versatile chips
  - owner ≠ controller badge
  - Æmber on artifacts
  - cards under Masterplan and the Spangler Box purged set
  - key unforge animation
- **Decision UI** (`decision/panel.py`): add screens for the new decisions:
  - number picker showing the powers currently in play
  - mode picker
  - Æmber-split slider at forge time
  - "may" / "repeat?" prompts
  - revealed-hand and revealed-deck-top overlays that respect viewer visibility
- **Log sentences and animations** (`option_labels.py`, `anim/director.py`) for the new log events: take_control, unforge, stun, power_counter, pay, lose, reveal, swap, put_into_play, under_card. The existing catch-all "settle" pass keeps the board in sync regardless.
- **Why-not reasons** (`game.why_not_playable`, plus a new `why_not_usable`): Treasure Map, Skippy, Foggify, stun, Truebaru's cost, Customs Office / Tentacus payment, Pitlord / Restringuntus / Control the Weak.
- **Menu and scenes:**
  - The deck dropdown lists presets and user decks.
  - New **Deck Builder** scene: a house tab per house showing the card grid for that house (53 cards, filterable by type). Click a card to add it to the 12-card pod (you can add the same card more than once). Includes counters, validation, Random fill, Save / Load / Delete, and right-click inspect.
  - The decklist viewer (D) works with any deck.
- **Bots:** make `HeuristicBot` handle every new decision kind (safe defaults), and target taunt-legal fights.

## v2.1 Milestone F: Alliance
- **Model:** `AllianceDeck(name, pods={House: (source_deck_name, [12 cards])})`. One pod per house, taken from any saved deck. The three houses must be different. Pods are copied by value, so later edits to the source deck don't change the alliance.
- **GUI Alliance Builder** scene: for each house, pick a source deck and preview its pod. Validate and save as a playable deck (the menu marks it "Alliance"). History stores the resolved decklist.
- **Engine:** no rules change. An alliance deck is a `Deck` with a source label, and the identity houses come from its pods.
- **Tests:** build, validate, play and replay an alliance game.

## Out of scope
Phase 3 houses: the engine is built generally so it's ready, but no Phase 3 cards get implemented. The "Algorithm" phases in the spec are separate work.

## Suggested order
1. v1.2/v1.3 match formats (small; exercises the match/replay/history plumbing on the existing decks).
2. v2.0 Milestones A → B → C → D → E.
3. v2.1 Milestone F.

---

## Critical files
- **Engine:**
  - `Code/Non-GUI/keyforge/match.py` (new), `config.py` (starting chains)
  - `Code/Non-GUI/keyforge/game.py`: fight/damage/destroy pipeline, legal actions, forge, house choice
  - `keyforge/player.py`: get functions
  - `keyforge/effects/{effect_object,steps,generic}.py`, plus new `effects/named/`
  - `keyforge/cards/{card,card_data,decks}.py`, plus new `cards/data/cota_pool.json`
  - `keyforge/enums.py`: DecisionKind
  - `keyforge/view.py`: revealed info
  - `keyforge/log.py`
  - `bots/heuristic_bot.py`
  - `sim/simulate.py`
  - `text_ui/`
- **GUI:**
  - `Code/GUI/gui/{settings,assets,snapshot,option_labels,history,engine_bridge}.py`
  - `decision/panel.py`, `sprites/card_sprite.py`, `anim/director.py`
  - `scenes/menu_scene.py`, plus new `scenes/deck_builder_scene.py`, `scenes/alliance_builder_scene.py`, `scenes/match_scene.py` (interstitial, first-player and bidding screens)
- **Reuse:** `Game.choose_cards / choose_house / yes_no / order_effects`, `Game.destroy_cards` (tagging pipeline), `Game.use_creature_ability`, `generic.duration_effect`, `steps.*`, `_play_resolution` (tied play-trigger ordering), `replay.encode_choice`.

## Verification
1. **Engine unit tests** (`python -m unittest discover -s tests` in `Code/Non-GUI`): new `tests/test_cards_phase2_{dis,logos,shadows}.py`, at least one test per new card plus every FAQ ruling in the rulings doc. New `test_rules_phase2.py` covers taunt, poison, stun, control, simultaneous damage, replacement order, house must/cannot, unforge, and Æmber spend split. The existing 49-card tests keep passing, with Library Access and Guardian Demon adjusted only if the audit finds conflicts.
2. **Registry test:** all 159 pool cards have a definition, art, and canonical text, and the JSON's houses and counts match CotA exactly (54/53/52).
3. **Fuzz:** `python -m sim.simulate --games 10000 --random-decks --check-invariants`, extended with invariants for:
   - card conservation by owner, and no card in two zones
   - Æmber ≥ 0, keys 0–3
   - controller consistency after leaving play
   - no stuck pending decisions
   - every pool card played or used at least once across the run (coverage report)
4. **GUI tests** (`python -m unittest discover -s tests` in `Code/GUI`):
   - extended assets, labels and snapshot tests for all 159 cards
   - headless bot-vs-bot autoplay with random decks, checking the board never drifts from the engine
   - UI playthroughs of the deck builder and alliance builder
   - a replay of a user-deck game after that deck is edited
5. **Manual playtest** via `python main.py`: build a deck in the builder, play human vs. bot, force the key interactions using seeded presets, and check the inspector's official-text panel.
