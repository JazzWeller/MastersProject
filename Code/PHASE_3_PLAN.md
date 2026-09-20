# Upgrade Plan: Phase 3 (all 7 houses, the full first expansion, any deck)

## Context

The client (engine `Code/Non-GUI/keyforge` + pygame GUI `Code/GUI`) currently implements **Phase 1.1 → 2.1**: the 159-card Dis/Logos/Shadows pool, the deck builder, alliance decks, and the Archon/Reversal/Adaptive match formats. Baseline at the time of writing: **327 engine tests and 131 GUI tests pass** (`python -m unittest discover -s tests` in each of `Code/Non-GUI` and `Code/GUI`), 159/159 pool cards have definitions and art, and 158 of them have gameplay hooks (Macis Asp needs none — it is poison-keyword-only).

The spec (*Master's Project specs and guide.pdf*, p.2) defines the next version in one line:

> **Phrase 3: All first expansion, all 7 houses, any deck.**

**First expansion = Call of the Archons (set 341), 370 cards.** Verified against the art on disk: `Phase 3/Cards/` holds exactly 370 images in 7 house folders — Brobnar 52, Dis 54, Logos 53, Mars 52, Sanctum 55, Shadow 52, Untamed 52 — which is CotA's house breakdown exactly. The `Dis`, `Logos` and `Shadow` folders are file-for-file identical to `Phase 2/Cards/`, so the Phase 3 art is a superset and can become the single art root.

**So Phase 3 = 211 new cards** (Brobnar 52, Mars 52, Sanctum 55, Untamed 52: 89 creatures, 78 actions, 31 artifacts, 13 upgrades) plus the engine/GUI work to stop assuming three houses.

### Sources (same order as Phase 2, per your standing rule)
Official errata > card text (keyteki) > rulebook glossary/FAQ > spec PDF. Archon Arcana is not used.

- **Card text**: keyteki `packs/CotA.json` (370 cards, all 7 houses), cross-checked against the official Master Vault API.
- **Rules/errata/FAQ**: KeyForge Master Rulebook v18.3.
- **Art mapping** (the one real research risk, see Milestone A): the image file names are `<set>-<number>.png` for whichever reprint each scan came from. keyteki has packs for sets 341, 435, 452, 496, 600, 700, 722, 800, 855, 874, 886, 918, 928 and 939 — but **not for 907 and 964**, which account for 44 and 25 of the new-house files. Phase 2 resolved those two sets against the Master Vault API; the same route is planned here, with a fallback.

New-house art by set prefix:

| House | Set prefixes (count) |
|---|---|
| Brobnar | 907×15, 964×12, 341×8, 918×5, 435×4, 600×3, 700×2, 452/722/800×1 |
| Mars | 939×19, 341×8, 918×8, 435×6, 700×5, 600×4, 800×1, 928×1 |
| Sanctum | 918×18, 907×17, 341×12, 435×3, 874×2, 886×2, 496×1 |
| Untamed | 964×13, 907×12, 341×7, 918×4, 700×4, 874×4, 855×3, 452×3, 435×1, 886×1 |

---

## Milestone 0: Baseline and branch

All of Phase 2/2.1 is currently **uncommitted** on `phase-1-1-gui`. Before any Phase 3 edits: run both suites, commit Phase 2/2.1 as its own commit (it is a coherent, tested unit), then branch `phase-3` off it. Phase 3 touches the same files heavily, and an unreviewable mega-diff of Phase 2 + Phase 3 would be the worst outcome for a project graded on its history.

## Milestone A: Card data for the four new houses

- **`Code/PHASE_3_CARD_POOL.md`** — the researched table for Brobnar, Mars, Sanctum and Untamed, in exactly the Phase 2 format (`# | Card | Type | Pow | Æ | Traits | Keywords | Canonical text | Errata | Art | P1`). Phase 2's table stays where it is; the builder reads both, so the Phase 2 research and its errata annotations keep their provenance.
- **Art mapping.** Build `set+number → card name` maps from the keyteki packs listed above, then join to the file names. For **907 and 964**, pull card records from the Master Vault deck API (`/api/decks/?expansion=<set>&links=cards`, which returns `card_number`, `house` and `card_title`) until every referenced number is covered. **Fallback if that stalls:** the card name is printed on every image, so unresolved files can be read directly and mapped by name — 69 images worst case, and it is self-verifying, because each house folder must end up a bijection with that house's CotA card list.
- **`tools/build_card_data.py`** extends to both tables and regenerates `keyforge/cards/data/cota_pool.json` with all 370 entries (art paths repointed to `Phase 3/Cards/...` for every card, including the 159 existing ones — `Phase 2/Cards` stays on disk but stops being referenced).
- **Effects split**: new `keyforge/effects/named/{brobnar,mars,sanctum,untamed}.py` alongside the existing three. The Phase 2 modules are already 576–701 lines each; the new ones should not be allowed to exceed that.
- **Registry test** (`tests/test_card_registry.py`): 370 cards, per-house counts 52/54/53/52/55/52/52, every card has a definition, canonical text and art that exists on disk, no duplicate names. **Delete `test_no_pool_card_has_armor`** — 17 of the new creatures have printed armor (Sanctum 13, Mars 3, Brobnar 1) — and replace it with a test that armor is parsed correctly for those 17.
- **`Code/PHASE_3_CARD_RULINGS.md`**: one entry per new card, canonical text + errata + the MRB FAQ rulings quoted as test cases, exactly as Phase 2 did.

### Errata to confirm during research
The "Fight:"/"Reap:" → "After Fight:"/"After Reap:" reading already applied in Phase 2 covers a large share of the new houses too (Sanctum Guardian, Grabber Jammer, Yxilo Bolter, Zyzzix the Many, …). The rest of the errata list is a research *output* of this milestone, not an input — no card gets implemented from memory.

## Milestone B: Seven houses everywhere

Mechanically small, but it touches every layer, so it goes early and lands as one commit.

- **`keyforge/enums.py`**: `House` gains `BROBNAR`, `MARS`, `SANCTUM`, `UNTAMED`.
- **`keyforge/cards/decks.py`**: `IDENTITY_HOUSES` disappears. A deck is **exactly 3 distinct houses out of 7**, 12 cards per pod, duplicates allowed. `validate_deck` checks distinctness and count rather than a fixed triple; `random_deck(rng)` first samples 3 houses, then fills. `AllianceDeck` is unchanged in shape — its three pods must simply come from three different houses.
- **`keyforge/game.py`**: the two `card.house != House.LOGOS` checks (Phase Shift's non-Logos allowance) are correct as written and stay; `player_houses` already derives from the deck.
- **Bundled presets**: Fignor and Igor are Dis/Logos/Shadows and keep working untouched. Add ~4 new presets covering the new houses (at least one per house, at least one with no Phase 2 house at all) once Milestone D has enough cards to make them legal.
- **GUI**: `settings.HOUSE_COLORS` gains four colors; `assets.house_emblem` gains four emblems (it currently hard-codes Dis/Logos/Shadows shapes and silently falls back to a plain ring otherwise); `anim/particles.py` stops hard-coding the three house colors; `scenes/deck_builder_scene.py`'s `HOUSES` tuple becomes "the deck's 3 chosen houses" plus a **house-picker step** (choose 3 of 7, after which the existing per-house tabs work unchanged); `scenes/alliance_builder_scene.py` picks each pod's house as well as its source deck; `menu_scene.py`'s hard-coded "Fignor · Dis / Logos / Shadows" line reads from the deck.
- **Bot**: `heuristic_bot._house_score` is already house-agnostic; the `House.SHADOWS` reference in `_is_good_play` (One Last Job) is card-specific and stays.

## Milestone C: Rules-engine generalization

Each item names the cards that force it, and each lands with tests before the cards that need it are implemented. Phase 2 already built the hard parts — control changes, the "use as if yours" framework, stun, taunt, poison, the damage/destroy pipeline, replacement effects, lasting durations, `get_power`/`get_keywords`/`get_hazardous` — so this is extension, not rework.

1. **Armor becomes a real computed stat.** `Game.get_armor` today returns `base_armor` only. It must become base + upgrade grants + other creatures' passives, with `armor_used_this_turn` unchanged: *Shoulder Armor* (+2 armor/+2 power **while on a flank**), *Protect the Weak* (+1 armor and taunt), *Bulwark* (each neighbor +2 armor), *Grey Monk* (each friendly creature +1 armor), and *Red-Hot Armor* (already implemented; must read a computed value instead of zeroing `base_armor`).
2. **Damage-prevention layer** in `steps.deal_damage`, ahead of armor: "cannot be dealt damage for the remainder of the turn" as a scoped lasting effect — *Shield of Justice*, *Potion of Invulnerability*, *Protectrix*.
3. **Assault X**, the before-fight mirror of hazardous (*Ancient Bear*), and **printed** hazardous on a creature (*Briar Grubbling*, hazardous 5) — `get_hazardous` currently sums upgrades only.
4. **Fully heal** as a first-class step (*Duma the Martyr*, *Protectrix*, *Cleansing Wave*, *Armageddon Cloak*); `steps.heal` already covers the bounded case.
5. **Ready-and-use chains**: "Ready and fight with a friendly creature" needs the use framework to accept a **forced use type** and a **"you may"** wrapper, and to be re-entrant for "up to 3 different friendly creatures, one at a time" — *Anger*, *Relentless Assault*, *Ganger Chieftain*, *Gauntlet of Command*, *Sergeant Zakiel*, *Commpod*, *Squawker*, *Swap Widget*.
6. **Dynamic stat and keyword sources from other cards.** `get_power` gains conditional modifiers (*Mushroom Man*: +3 per unforged key; *Shoulder Armor*: flank-conditional), and `get_keywords` gains grants from other creatures' passives (*Halacor*: each friendly flank creature gains skirmish).
7. **Key cost as a get function with dynamic terms** (*Iron Obelisk*: +1 per damaged friendly Brobnar creature — recomputed at forge time, never stored; *Murmook*; *Jammer Pack* via an upgrade), plus **forge-now variants**: at current cost (*Key Charge*, *Chota Hazri*), at +9 reduced per card in hand (*Key Abduction*), at no cost (*Epic Quest*). Phase 2's *Key of Darkness* already opened this door. "After you forge a key" (*Bilgum Avalanche*) reuses the existing `key_forged` event.
8. **Capture family extensions**: capture **from its own side** (*Mindwarper*, *Hypnotic Command*), capture all but N (*Gatekeeper*), a lasting "each time a friendly creature fights, it captures 1" (*Take Hostages*), and a replacement on æmber entering a pool (*Ether Spider*: each Æ that would be added to your opponent's pool is captured instead). Plus **steal immunity** (*The Vaultkeeper*).
9. **Reveal-as-cost with a variable count** (*Commpod*, *Orbital Bombardment*, *Incubation Chamber*, *Zyzzix the Many*, *Martians Make Bad Allies*, *Deep Probe*): `CHOOSE_CARDS` with min 0, a reveal log event, and `PlayerView.revealed` (added in Phase 2) driving who sees what.
10. **Trait-, house- and keyword-matters selectors**: *Custom Virus* ("shares a trait with the purged creature"), *Curiosity* (Scientist), *Troop Call* (Niffle), *Honorable Claim* (Knight), *Begone!* (Dis), *Perilous Wild* (each **elusive** creature). Traits already live on `CardDef.tags`; this is a selector helper, not new state.
11. **Granted quoted abilities and destroyed-replacement upgrades** — *Armageddon Cloak* ("gains hazardous 2 and, 'Destroyed: Fully heal this creature and destroy Armageddon Cloak instead'"), *Biomatrix Backup*, *Phoenix Heart*, *Mantle of the Zealot*. Extends Phase 2's ability model (`abilities(card)`) plus `InsteadEffect`; the single most intricate item in the milestone.
12. **Battleline manipulation and flank conditions**: swap two friendly creatures' positions (*Sanctum Guardian*), and flank-conditional effects (*Valdr*, *Staunch Knight*, *Radiant Truth*, *Halacor*, *Shoulder Armor*) — `card.flank` and `forced_flank` already exist.
13. **Per-creature use restrictions and enters-play states**: "cannot reap" (*Tireless Crocag*), "can only fight" for the turn (*Horseman of War*), enters play **stunned** (*Yxilx Dominator*), enters play **ready** (*Soft Landing*, *Swap Widget*, and the existing Speed Sigil / Silvertooth path).
14. **Fight-damage modifiers**: "+2 damage while attacking a flank creature" (*Valdr*) and "deals no damage when fighting" (*Ether Spider*).
15. **Copy effects**: *Mimicry* — when played, treat it as a copy of an action card in the opponent's discard pile (a dynamic `CardDef` substitution for one resolution).
16. **Zone odds and ends**: discard from the top until a match (*Sound the Horns*, *Invasion Portal*), return a creature from the discard to the **top of the deck** (*World Tree*), put an enemy creature into **your** archives with a leaves-archive replacement (*Sample Collection*), shuffle each creature of a type into its owner's deck (*Mating Season*), "each player with 6Æ or more is reduced to 5Æ" (*Doorstep to Heaven*), purge if this damage destroys it (*Yxilo Bolter*).
17. **Stun** needs no new machinery (Phase 2 built it) but gains ~14 users, including stun-with-neighbors (*Tremor*) and stun-on-enter.

## Milestone D: Implement the 211 new cards

Same method as Phase 2 Milestone C: reuse the `effects/generic.py` factories wherever they fit, put named effects in the per-house module, order by dependency, and commit each card with its tests and its rulings-doc entry. Suggested waves, **one house at a time**, so each house is playable end to end before the next starts:

1. **Brobnar (52)** — the simplest house mechanically: damage, ready-and-fight, chains, key-cost pressure. Proves C.3, C.5, C.7, C.14.
2. **Sanctum (55)** — armor, taunt, healing, damage prevention, Knights. Proves C.1, C.2, C.4, C.11, C.12.
3. **Mars (52)** — stun, capture, reveal-as-cost, archives, Mars-counting effects. Proves C.8, C.9, C.16.
4. **Untamed (52)** — forge tricks, discard recursion, trait/keyword sweepers, Mimicry. Proves C.6, C.10, C.15.

Per-house exit criteria: every card in that house has a hook or a documented reason it needs none, a test per card, and a 2,000-game fuzz run with decks containing that house and no stuck decisions.

## Milestone E: GUI upgrade

- **Art**: `CardDef.image` paths point at `Phase 3/Cards/…`; `test_assets.py` extends to all 370. `assets._raw` caches every loaded image forever — at 200×280 that is ~83 MB of raw surfaces if all 370 load (the deck builder now browses 7 houses), so add LRU eviction to the raw cache and leave the scaled cache as-is.
- **Card sprite** (`sprites/card_sprite.py`): an **armor chip** (including armor spent this turn), **assault/hazardous** chips, a "cannot be dealt damage" shield state, and the enters-play-stunned/ready states. Taunt, stun, power counters, poison and versatile already render.
- **Decision UI** (`decision/panel.py`): variable-count reveal selection ("reveal any number"), the "choose one" mode picker (already present, now used by *Begone!*), and the ready-and-fight sub-prompts.
- **Log sentences and animations** (`option_labels.py`, `anim/director.py`) for the new events: reveal, ready, swap, capture-from-own-side, armor absorb, prevention, forge-at-cost.
- **Why-not reasons** (`game.why_not_playable` / `why_not_usable`): cannot reap, can only fight, stunned on entry, no legal fight target under taunt, key-cost modifiers at forge time.
- **Deck builder**: the house-picker step (3 of 7) and a browse grid over 370 cards, with the type filter plus a **name search box** — 53 cards per house was browsable; 370 across a builder session is not, without search.
- **Alliance builder**: per-pod house choice as well as source deck.

## Milestone F: Bots, simulation, performance

- **`HeuristicBot`**: armor-aware fight value (a 4-power attacker no longer kills a 4-power/2-armor defender), assault/hazardous in the same calculation, don't attack into prevention, use stun sensibly, and safe defaults for every new decision kind. Its house score should also weight artifacts and Omni abilities, which the new houses lean on and it currently undervalues.
- **`sim.simulate`**: `--random-decks` samples 3 of 7 houses; the coverage report asserts **every one of the 370 cards was played or used at least once** across a long run; invariants unchanged, plus armor never exceeding its computed value and capture totals conserved.
- **Performance budget**: the GUI suite already takes ~219 s. Phase 3 should keep it under ~5 minutes — headless autoplay tests get seeded short games rather than more of them.

## Milestone G: Docs and playtest

`Code/PHASE_3_PLAN.md` (this file), `Code/PHASE_3_CARD_POOL.md`, `Code/PHASE_3_CARD_RULINGS.md`, and a README refresh. Then a manual playtest per house via `python main.py`: build a 3-of-7 deck in the builder, play human vs. bot, and force the marquee interactions (armor + Red-Hot Armor, taunt walls, Mimicry, Key Charge, Ether Spider, Armageddon Cloak).

---

## Decisions taken (flagged rather than buried)

- **"First expansion" = Call of the Archons, set 341.** The 370 images on disk match CotA's house counts exactly, so the pool is unambiguous even though the spec never names the set.
- **One art root.** All 370 cards reference `Phase 3/Cards/…`; the identical `Phase 2/Cards` copies stay on disk but stop being referenced.
- **Two pool tables, one generated JSON.** `PHASE_2_CARD_POOL.md` is not rewritten; `PHASE_3_CARD_POOL.md` adds the four new houses and the builder merges them into `cota_pool.json` (now 370 entries, same file name — it was always "the CotA pool").
- **Deck legality = any 3 distinct houses of the 7**, 12 per pod, duplicates allowed. Alliance decks likewise: one pod per house, from up to 3 different decks.
- **Random decks sample houses too**, so the fuzzer exercises cross-house interactions no bundled preset covers.

## Out of scope

- **Mavericks and anomalies.** Real CotA decks can contain a card printed in a different house; the deck model derives a card's house from its definition, so this is not supported and should be stated in the README rather than half-built.
- **Non-CotA sets** (Age of Ascension onward) and the keywords that arrive with them (Enhance, Ward, Alpha/Omega, Deploy) — none appear in CotA.
- **Importing real decks from Master Vault by name.** Tempting once the full set exists, and cheap to add later; it is not in the spec.
- **"Algorithm Phase 2"**, which the spec schedules *after* implementation phase 3 — Phase 3 should leave the engine's `PlayerView` / decision API untouched so that work can start cleanly.

## Risks

1. **Art mapping for sets 907 and 964** (69 files) — keyteki has no pack for either. Mitigated by the Master Vault API route and the read-the-image fallback; the per-house bijection check catches any mistake.
2. **Volume.** 211 cards is ~1.3× the Phase 2 card work, and Sanctum/Mars carry the intricate items (prevention, granted quoted abilities, capture replacements). The per-house waves exist so the project always has a shippable state.
3. **Armor is genuinely new.** Every fight, damage and destroy test written so far ran with zero armor in the pool; expect the fight pipeline and the bot's fight evaluation to need real work, not a one-line change.
4. **GUI test runtime and art memory** — see Milestone F.

## Suggested order

Milestone 0 → A (research/data, the long pole) → B (7 houses) → C (rules) interleaved with D (cards, one house at a time: Brobnar → Sanctum → Mars → Untamed) → E → F → G.

## Critical files

- **Engine**: `keyforge/enums.py`, `keyforge/cards/{decks,card,card_data}.py`, `keyforge/cards/data/cota_pool.json`, `keyforge/effects/named/{brobnar,mars,sanctum,untamed}.py` (new), `keyforge/effects/{steps,generic,effect_object}.py`, `keyforge/game.py` (`get_armor`, `get_power`, `get_keywords`, `get_hazardous`, the fight/damage pipeline, forge, the use framework), `keyforge/view.py`, `tools/build_card_data.py`, `bots/heuristic_bot.py`, `sim/simulate.py`.
- **GUI**: `gui/settings.py` (house colors), `gui/assets.py` (emblems, art cache), `gui/sprites/card_sprite.py`, `gui/decision/panel.py`, `gui/option_labels.py`, `gui/anim/{director,particles}.py`, `gui/scenes/{deck_builder,alliance_builder,menu}_scene.py`.
- **Docs**: `Code/PHASE_3_CARD_POOL.md`, `Code/PHASE_3_CARD_RULINGS.md`.

## Verification

1. `python -m unittest discover -s tests` in `Code/Non-GUI`: the 327 existing tests keep passing, plus `test_cards_phase3_{brobnar,mars,sanctum,untamed}.py` (one test per new card minimum, every FAQ ruling from the rulings doc) and `test_rules_phase3.py` (armor sources, prevention, assault, ready-and-fight chains, dynamic key cost, capture-from-own-side, reveal-as-cost, granted quoted abilities, Mimicry).
2. Registry: 370 cards, correct per-house counts, art on disk, canonical text, armor parsed for the 17 armored creatures.
3. Fuzz: `python -m sim.simulate --games 10000 --random-decks --check-invariants`, with a coverage report proving all 370 cards were exercised.
4. `python -m unittest discover -s tests` in `Code/GUI`: assets/labels/snapshot for 370 cards, headless autoplay with random 3-of-7 decks, deck-builder and alliance-builder playthroughs.
5. Manual playtest per house (Milestone G).
