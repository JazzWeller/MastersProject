# Playtest fixes — findings and plan

Two passes, as requested:

1. **Usability pass** — every screen and interaction state, looking for
   anything that's missing, misleading, or can't be done.
2. **Rules pass** — the engine checked against the project spec *and* the
   printed text on the card art (which is authoritative for "regular
   KeyForge"). Card text was read directly off `Phase 1/Cards/*.png`.

Everything marked **confirmed** was reproduced by running code or reading
the relevant source path, not inferred.

---

## 0. The reported first-turn bug

> "only one action can be played during step 3 on the second player's
> first turn"

**The engine does not limit the second player.** `Game._legal_actions`
applies the one-card limit only when `turn_number == 1 and pid ==
_first_player`. I drove the second player's first turn two ways:

- straight through the engine: 1–6 cards played depending on hand;
- through **real GUI clicks** (card click → chooser → Play), human-vs-bot
  and hot-seat: the second player played 2–4 cards, the first player
  exactly 1.

Every time the second player stopped early, a real rule explained it
(300 fuzzed games): Ember Imp's 2-card limit, Lifeward's no-creatures
effect, or no cards of the chosen house left — plus one genuine bug
(R3, Duskrunner). So the rule itself is implemented correctly, but the
UI makes it look broken:

- **The menu defaults "First player" to Random and nothing on screen ever
  says who went first.** "Player 2" is frequently the *first* player,
  and is then correctly limited to one card. This is the most likely
  explanation for what you saw.
- **Nothing explains why a hand card isn't playable.** Unplayable cards
  look identical to playable ones apart from the missing glow — whether
  the reason is the first-turn rule, the wrong house, Ember Imp, Lifeward,
  or Scrambler Storm.

Both are fixed below (U1, U2). If it still happens after those fixes,
the new "why can't I play this" reasons will show which rule is blocking
it.

---

## 0.4 A game ended with a winner who never forged a key

Reported: the game declared the other player the winner, though they had
never shown more than 3 Æmber or any keys.

**Not reproduced yet.** What was checked:

- **Engine, 1,500 bot-vs-bot games:** every game ended with "3 keys", and
  in every one the winner had exactly three `forge_key` events and
  `keys == 3`. No other code path sets a winner — the only other ending
  is the turn limit, which has no winner and displays "It's a draw".
  Lowest key cost ever seen: 5 (Titan Mechanic), never below.
- **GUI at real animation speed, 12 full human-vs-bot games (~90k settled
  frames):** the HUD's displayed Æmber always matched the engine once
  animations settled, and the key icons always matched the engine's key
  count. Every winner had three forge events.
- **Occlusion:** artifact and hand cards do sit on the opponent's HUD
  strip (see 0.6), but in the frames inspected the Æmber number and
  keys stayed readable. The layout fix removes this regardless.

So the engine can't produce this result on its own. Either the GUI misled
the display in a state these runs didn't hit, or the state that caused it
wasn't recorded. **Neither can be checked after the fact**, because the
GUI starts every menu game with an unrecorded random seed and the game
over screen hides the board. The fixes make any future case verifiable:

**W1. Every game is replayable.** When the menu's seed is "Random", pick
a concrete seed and store it. On game over (and on crash or quit), write
`logs/game-<timestamp>.json` with decks, seats, first player, seed, and
the engine's `choice_log`. A small `tools/replay.py` replays a file
through the engine and prints the log, so a reported game can be
reproduced exactly.

**W2. The game-over screen shows how the game was won.** Final keys and
Æmber for both players, the turn each key was forged, and "View Board"
(U15), so a surprising result can be checked on the spot.

**W3. Key forges are impossible to miss.** Today a forge happens at the
start of the forging player's turn, right after the other player clicks
End Turn, and is shown only as a 1.2 s banner at the top of the screen
while the End Turn animations play. *Fix:* a longer banner and a key
icon that animates into that player's HUD slot, a gold log line
(already coloured), and "Forged turn N" on hover of each key icon.

**W4. Keep testing for it.** A GUI test that plays full games at real
animation speed and, on every settled frame, asserts that the displayed
Æmber/keys equal the engine's and that no card covers either player's
Æmber or key icons, using rotated hitboxes (M1). At game over it asserts
the winner has three forge events. (A throwaway version of this check is
what found the HUD overlap.)

If it happens again before W1 lands, the most useful details are:
human-vs-bot or hot-seat, which player number you were, whether either
side had artifacts in play, and a screenshot of the log panel before
clicking past the game-over screen.

## 0.45 Game history and replay viewer

Supersedes W1's flat log files.

**H1. Every game is recorded in a local SQLite database**
(`Code/GUI/data/history.sqlite3`, git-ignored). One row per game with
queryable summary columns — start/end time, decks, seat types, first
player, seed, winner, end reason, turn count, status (finished /
abandoned) — plus the full move record as a zlib-compressed JSON blob.

**H2. The move record is compact and exact.** The engine is deterministic
for a given seed, so a game is fully described by its settings plus, for
every decision, the *index* of the chosen option (or the list of indices
for multi-select and ordering). A typical game compresses to a few KB.
The row is written when the game starts and updated at every turn
boundary, so a crash or quit still leaves a replayable (abandoned)
record. Menu games with seed "Random" get a concrete generated seed so
they can be replayed.

**H3. "Past Games" in the main menu** lists recorded games newest first
— date, decks, who played, result, turns — with paging and a delete
option.

**H4. Step-by-step visual replay.** Opening a game rebuilds it on the
normal board, using the same animations: next/previous decision,
next/previous turn, play/pause with speed control, jump to start/end, and
a scrubber showing the current turn. Both hands can be revealed or shown
from either player's perspective. Stepping backwards rebuilds from the
start instantly (the engine replays thousands of decisions per second).

**H5. Tests:** record → compress → store → load → replay reproduces the
identical final state and log for fuzzed games; an abandoned record
replays up to where it stopped; the history list and replay scene
survive every decision kind.

## 0.5 Clicks select the wrong card

Measured by painting every card in the real draw order into an ID map,
then asking the click code what it would select at each canvas pixel
(8 seeds, ~370k pixels, taken at CHOOSE_ACTION decisions):

- **~19% of pixels on a glowing card** resolved to a different card or to
  nothing.
- **~18k pixels** on a *non*-glowing card (or empty felt) still selected
  a glowing card underneath.

Two independent causes, both confirmed:

**M1. Tilted cards have their hitbox tilted the wrong way.**
`CardSprite.contains_point` un-rotates by `-rot`; pygame's `rotozoom`
rotates by `+rot`. For an outer hand card (10°) only 80.5% of its drawn
pixels agreed with the hitbox; with the sign flipped, 98.7% (the rest are
rounded corners). This affects hover and the zoom panel too, not just
clicks.

**M2. Clicks don't pick the card on top.** `DecisionPanel.click_board`
tests only glowing cards, in dictionary order (its sort key is sprite
scale, which is always 1.0), so where fanned cards overlap it returns
whichever happens to come first — often the card *underneath*. Hover
uses a different rule (draw order), so the zoom panel can show the card
you're pointing at while the click picks its neighbour.

Fixing M1 alone: 2,358 → 1,672 mismatched pixels. M1 + M2 (hit-test
*every* visible card topmost-first by draw order, then act only if that
card is legal): **113**, all at rounded corners.

**M3. One shared hit-test.** Hover, click, middle/right-click inspect
and the glow all have to answer "which card is under the cursor" the same
way. *Fix:* a single `Board.card_at(pos)` that returns the topmost
visible card by draw order; everything calls it. Add a regression test
that re-runs the ID-map comparison above and requires ≥99% agreement.

**M4. Clicking a card while the chooser is open does nothing.** The
first click only closes the chooser, so clicking card B while A's
chooser is showing appears to be ignored or to "stick" on A. *Fix:*
close the chooser, then process the click normally.

**M5. Upgrades cover part of their host.** An upgrade is drawn on top of
its creature, so clicking that part of the creature hits the upgrade
(never a legal target by itself). *Fix:* route clicks on an upgrade to
its host.

**M6. Clicks during animations are silently dropped.** Input is ignored
while the animator is busy, with no feedback. *Fix:* queue the click and
apply it when the animation settles if the same card is still under the
cursor and still legal — or at minimum show a brief "busy" cursor.

## 0.6 Rows overlap

Measured from the actual layout code (card extents vs. bands, 1–10 cards):

| Overlap | px |
|---|---|
| opponent artifacts ↔ opponent HUD | **40 — the whole HUD strip** |
| your artifacts ↔ your HUD | **40 — the whole HUD strip** |
| opponent artifacts ↔ opponent creatures | 41 |
| your creatures ↔ your artifacts | 41 |
| opponent hand ↔ opponent artifacts | 23 |
| opponent hand ↔ opponent HUD | 20 |
| your artifacts ↔ your hand | 20 |
| your HUD ↔ your hand | 17 |
| your hand ↔ action bar | 3 |

Artifact cards are 147 px tall in a 56 px band, so whenever a player has
an artifact their Æmber, keys and key cost are hidden underneath it. The
previous pass accepted "bounded bleed" between bands; that was wrong.

**L1. Redesign the vertical layout so every band fully contains its
cards.** There isn't room for ten separate bands at readable card sizes,
so change the structure rather than squeezing further:

- **Artifacts move into a side lane in the same band as creatures** (right
  end of each board row, behind a divider), instead of a band of their
  own. This removes two bands.
- **The opponent's hand becomes a compact, flat row of card backs**
  (64×90). You can't read them anyway; only the count matters.
- Budget (px, top to bottom, 10 px gaps): opponent hand 90 · HUD 40 ·
  board 180 · prompt 34 · board 180 · HUD 40 · your hand 205 · action
  bar 38 ≈ 880 of 900. That fits **larger** cards than today: board cards
  120×168 (up from 105×147), hand cards 130×182 (up from 112×157).
- Creature lane ≈ 720 px wide, room for 5 creatures with visible gaps;
  beyond that they compress *horizontally within the row*. Artifact lane
  ≈ 390 px (3 artifacts, then compress).
- Upgrades render as a tab inside the host's bounds, not above it.

**L2. Visible separation.** Each board row gets a faint panel with a
label ("Creatures" / "Artifacts") and a divider between the two players'
halves, so rows read as distinct areas even when empty (this replaces the
dashed empty-row ghosts).

**L3. Layout invariant test.** For every zone and every card count 1–12,
every card's rotated bounding box must lie inside its own band, and no
two bands may intersect. This replaces the weaker "stays on canvas"
test, which the old layout passed while still overlapping.

**L4. Re-verify with screenshots** at 1600×900 and 1280×720, with 0, 3
and 8 creatures and 0 and 3 artifacts per side.

## 0.7 Automated sweep findings

A fuzz sweep (40 hot-seat games, every decision checked) confirmed that
every decision can be answered by clicking and no log event is silently
dropped. It also found:

- **U23.** Modal titles can overflow the modal
  ("Control the Weak: choose the enemy's house"). *Fix:* wrap titles, or
  size the modal to the title.
- **U24.** The Options card grid scrolls with no scroll indicator (seen
  with a 17-card discard pile for Creeping Oblivion). *Fix:* scrollbar +
  "N more" hint, sharing the pile browser's grid (U5).

---

## 1. Rules pass — engine bugs

Ordered by how much they change the game.

**R1. Scrambler Storm does nothing.** *(confirmed)*
`CanPlayActions` is set by the card but never read anywhere — neither
`_legal_actions` nor `_play_card` checks `get_can_play_actions`. Printed
text: "Your opponent cannot play action cards on their next turn."
*Fix:* check it in both places (so Wild Wormhole flipping an action onto
the table is also refused and returned to the deck top, per the spec's
Wild Wormhole rule).

**R2. Cards that enter play mid-turn can never be used that turn.**
*(confirmed)* `CanBeUsed` is set only at house selection, for cards
already in play. Silvertooth ("enters play ready") arrives ready but
with `CanBeUsed=False`, so it can't reap or fight — the card's entire
point. *Fix:* in `_play_card` (and anywhere a card enters play under your
control), set `CanBeUsed = card.house == player.selected_house`. Add a
test that Silvertooth can reap the turn it's played.

**R3. Duskrunner can't be played unless you control a creature.**
*(confirmed, 52 of 300 games)* `_legal_actions` requires
`player.play_area.creatures`, but upgrades attach to *any* creature (spec:
"Attaches to any creature"; `_play_card` already allows enemy targets).
*Fix:* require a creature in either play area.

**R4. Lights Out lets you return zero creatures.** *(confirmed)* Printed
text: "Return 2 enemy creatures to their owner's hand." Engine uses
`min_n=0`. *Fix:* `min_n = max_n = min(2, len(enemy creatures))`.

**R5. Guardian Demon can hit the creature it just healed.** *(confirmed)*
Printed text: "Deal damage to *another* creature." When no other
creature exists the code falls back to the full list. *Fix:* no other
creature → no damage. Also only offer damaged creatures as the heal
target (choosing an undamaged one silently wastes the ability).

**R6. The Howling Pit uses the wrong mechanic.** *(confirmed; decided)*
Printed text: "During their 'draw cards' step, each player refills their
hand to 1 additional card." Implemented (following the spec) as
`CardDrawModifier +1`, which draws nothing when a hand is already full.
*Fix (approved):* `DrawUpToLimit +1` for both players, like Mother. This
follows the card over the spec.

**R7. Ember Imp — keep as the spec says.** *(decided)* The printed card
limits all plays; the spec limits only cards played from hand, so Wild
Wormhole can exceed it. **No change**, by your decision. Add a comment at
`ember_imp_register` noting the deliberate spec choice, and a test pinning
the behavior so it isn't "fixed" later by mistake.

**Rules arbiter going forward:** Archon Arcana
(https://www.archonarcana.com/wiki) decides what a card, effect, or rule
does. Where the spec directly contradicts it, ask first, with Archon
Arcana as the default. Every card and rule in this plan gets checked
against its Archon Arcana page during implementation, not just the ones
already flagged.

**R8. Dominator Bauble offers creatures it can't use.** Exhausted
creatures are offered, and choosing one silently wastes the Bauble.
*Fix:* offer only ready creatures that pass the rule of six; if none, the
Action does nothing (and ideally isn't offered — the card can still be
exhausted for no effect, which is legal but a trap).

**R9. Forced house and skipped forge happen silently.** Control the Weak
skips the house decision entirely and Miasma skips the forge step, with
no log event for either. The rules are correct; the player just has no
way to know. *Fix:* log `house_forced` (with source card) and
`forge_skipped` (with reason), and surface both as banners (U9).

**Checked and correct** (no change): turn order and readying; key forging
and the 3-key win; mulligan order and hand sizes (7/6, one fewer on
mulligan); archive pickup; first-turn rule; rule of six (plays + uses
share a count); discard only from the active house; Phase Shift
allowance; Elusive (once per turn, reset correctly); Skirmish; armor
reset; captured Æmber on leaving play; upgrades discarded with their
host; destroy pipeline ordering and Bad Penny; chains in the draw step;
duration timing for Miasma, Lash, Lifeward, Control the Weak, Scrambler
Storm; Library Access trigger; Titan Mechanic applying to both players
(matches the printed "each key costs –1"); every other card's effect
against its printed text.

**R10. The bot opponent plays randomly.** Not a rules bug, but it changes
how the game feels more than any single card: `RandomBot` picks End Turn
with the same probability as any other option, so it routinely ends its
turn with plays left and mulligans at random. *Fix:* a simple heuristic
bot in `bots/` — pick the house with the most playable cards + usable
creatures, play everything playable, reap with ready creatures, fight
only favorable trades, end turn last. Keep `RandomBot` for fuzz tests.

---

## 2. Usability pass

### 2.1 Can't inspect a card where you'd expect to

**U3. Mulligan cards aren't hoverable or inspectable.** *(confirmed)*
The mulligan screen draws cards itself, but hover only tracks board
sprites — so the sidebar still says "Hover a card to inspect it" while
you hover a card. Worse, the hidden hand sprites are still hover targets
at their old fan position, so hovering empty space near the bottom
"inspects" an invisible card.
*Fix:* a general **hover-target registry**: anything that draws a card
face (mulligan row, chooser thumbnail, Options grid, pile browser,
decklist, log card names) registers `(rect, card info)` each frame, and
hover/zoom/inspect read from it instead of only the sprite pool. Skip
sprites that aren't drawn.

**U4. Middle-click is the only way to open the inspector.** Most
laptop trackpads have no middle button. *Fix:* right-click opens the
inspector (drop "pin" — the inspector supersedes it), and clicking the
sidebar zoom panel also opens it. Keep middle-click as an alias.

**U5. The Options grid is tiny and anonymous.** Creeping Oblivion,
tie-breaks, and any card pick from a pile use 96×134 thumbnails with no
names and no way to read them. *Fix:* reuse the pile-browser cell
(140×196 + name) and the U3 hover registry.

**U6. The zoom panel is covered or stale during overlays.** The mulligan
dim covers the board but not the sidebar, so a stale zoom sits next to
it. *Fix:* the zoom follows the U3 registry in every state.

### 2.2 Decisions that don't give you what you need to decide

**U1. Nobody is told who goes first, or about the first-turn rule.**
*Fix:* a "Player N goes first" banner after mulligans; a "First player"
tag on that player's HUD for turn 1; a persistent chip during that turn
— "First turn: play or discard 1 card", switching to "First-turn limit
reached" once used. Rename menu seats to show turn order after start.

**U2. No reason is shown for unplayable hand cards.** *Fix:* the engine
exposes a per-card `why_not_playable(card) -> Optional[str]` (wrong
house, first-turn limit, Ember Imp limit, can't play creatures/actions,
no creature to attach to, rule of six). The GUI dims those cards and
shows the reason on hover. When the only legal action left is End Turn,
the prompt says so ("No more plays — End Turn") and the End Turn
confirmation is skipped.

**U7. Taking the archive is decided blind.** *(confirmed)* The
"Take your archive?" Yes/No modal blocks pile clicks, so you can't see
what's in it. *Fix:* show the archived cards inline in that decision
(same layout as the mulligan screen, smaller).

**U8. House choice gives no information.** Three plain rows reading
"Dis / Logos / Shadows". *Fix:* per house, the house emblem plus "N
playable in hand · M ready in play", with those cards highlighted on
hover. (A forced house skips this — see U9.)

**U9. Silent rule effects.** Forced house (Control the Weak), skipped
forge (Miasma), a key you *can't* forge despite enough Æmber (Lash,
Miasma) — none are announced. *Fix:* banners fed by R9's new log events,
and the HUD key-cost readout turns red with the reason.

**U10. Effect ordering shows no order.** *(confirmed)* ORDER_EFFECTS
highlights picked rows but never numbers them, and the labels "Play
effect" / "Play triggers" are cryptic. *Fix:* number picks 1, 2, …, add
Undo, and describe rows by source ("Wild Wormhole: play effect",
"Library Access: draw a card").

**U11. Flank choice is abstract.** "Left flank / Right flank" rows.
*Fix:* show ghost slots at both ends of your creature row and let you
click one.

**U12. The confirm guard never disarms.** "Confirm End Turn?" stays armed
after unrelated clicks. *Fix:* disarm on any click that isn't the armed
control, or after ~3 s.

**U13. Multi-select state is unclear on the board.** For Lights Out
and Pawn Sacrifice, picked cards glow white, but there's no count and no
way to see how many picks are left. *Fix:* a prompt counter ("1 of 2
chosen") and a numbered badge on each pick.

### 2.3 Things you can't do when you should be able to

**U14. You can't browse piles on the opponent's turn.** *(confirmed)*
`_maybe_click_pile` runs only during your own decision, after the
bot/opponent early return. Piles are public information at all times.
*Fix:* handle pile clicks before the decision gating (the decklist key
already works any time).

**U15. The final board disappears at game over.** *(confirmed)* The
game-over scene replaces the board completely. *Fix:* draw the board
dimmed underneath, add "View Board" (hides the panel until clicked), and
say "You win!" / "You lose" when a human is seated.

**U16. No way to quit from the menu.** *Fix:* a Quit button, with Esc
on the menu asking to confirm.

### 2.4 Board readability

**U17. Creature power and armor aren't shown on the board.** Only
damage appears. Fights are hard to read without hovering every creature.
*Fix:* a power badge (and armor when non-zero) on every creature, with
damage shown as "power − damage" remaining.

**U18. You can't tell which creatures can still act.** Only exhausted
creatures are marked; creatures of other houses look identical to usable
ones. *Fix:* on your turn, dim creatures and artifacts that can't be
used, with the reason on hover (reuse U2).

**U19. The hand isn't sorted.** Cards keep draw order. *Fix:* sort by
house, then type, then name; the active house's cards come first.

**U20. HUD effect chips silently truncate.** When chips don't fit they
just stop drawing. *Fix:* collapse overflow into a "+N" chip with the
full list on hover; hover any chip to see the source card.

**U21. Text is too small at smaller window sizes.** Pile labels (10 px),
log (13 px), and hints (11 px) drop to ~8 px at a 1280×720 window.
*Fix:* a 12 px minimum for any label, and a UI-scale setting.

**U22. The log is text only.** *Fix:* hover a card name in the log to
zoom it (via U3); log the missing event kinds that are currently
dropped silently (e.g. `put_on_top`), and give clearer text for
`duration_effect` ("Miasma: your opponent skips forging next turn").

---

## 3. Order of work

0. **W1–W4** — replay logs and a verifiable game-over screen, so the win
   report can be diagnosed if it recurs.
   **M1–M6, L1–L4** — misclicks and overlapping rows. These make the
   game unreliable to play regardless of anything else, and the layout
   change affects where most other UI fixes are drawn, so they go first.
1. **Rules R1–R5, R9** — actual wrong gameplay. Each gets a focused
   engine test (Scrambler Storm blocks actions; Silvertooth can reap on
   entry; Duskrunner onto an enemy creature; Lights Out takes exactly 2;
   Guardian Demon never self-targets).
2. **U1, U2** — the two changes that explain the first-turn confusion.
3. **U3–U6** — one hover/inspect system everywhere, reachable without a
   middle mouse button.
4. **U7–U13** — decision screens.
5. **R6, R7 (comment + test only), R8** — Howling Pit follows the card;
   Ember Imp stays per spec. Before implementing, check every card
   against Archon Arcana and bring back any new spec conflicts.
6. **U14–U22, R10** — board, log, bot.
7. **Verification** — full test suites, plus a screenshot pass over every
   decision kind, including an explicit second-player first turn.


---

## 4. Delivered

Everything in sections 0.4–2 was implemented, in the order of section 3.
Engine: 94 tests (20 new). GUI: 5 new test files plus rewrites of the
layout and playthrough tests; see `Code/GUI/README.md` for the list.

**Rules (engine).** R1 Scrambler Storm now blocks actions, including Wild
Wormhole plays. R2 cards entering play are usable that turn (Silvertooth).
R3 Duskrunner can go on an enemy creature. R4 Lights Out returns exactly 2.
R5 Guardian Demon only heals damaged creatures and never damages the one it
healed. R6 The Howling Pit raises both hand refills (approved over the
spec). R7 Ember Imp kept per spec, with a comment and a test. R8 Dominator
Bauble offers only usable creatures. R9 forced houses, skipped forges and
turn starts are logged, and key forges record their turn and cost.
`Game.why_not_playable` explains every play restriction using the same
checks as `_legal_actions`. R10 `bots/heuristic_bot.py` beats the random
bot 300/300 and ends games in ~19 turns instead of ~55.

**First-turn report.** Confirmed the engine and the GUI never limit the
second player (engine fuzz + real GUI clicks). The confusion is addressed
by U1/U2: the first player is announced and tagged, the prompt bar tracks
the one-card limit, and every unplayable card says why.

**Win-condition report.** Still not reproduced (0.4). Every game is now
recorded (H1–H5) and the game-over screen shows each key's forge turn
(W2), so a recurrence can be replayed step by step from Past Games. W4's
real-speed test checks the HUD against the engine and for occlusion on
every settled frame.

**Clicks and layout.** M1 hitbox rotation sign fixed; M2/M3 one
`Board.card_at` used by hover, click and inspect, topmost card first
(≥99% pixel agreement, tested); M4 chooser click-through; M5 upgrade →
host; M6 clicks during animations are applied when the animation ends.
L1–L3: creatures and artifacts share a row in two lanes, the opponent's
hand is a compact row, every band contains its cards (tested for 1–12
cards per zone), cards got bigger (board 120×168, hand 130×182). Exhausted
creatures stay upright with an "Exhausted" badge instead of turning
sideways into their neighbours.

**Second pass: found by reading screenshots of the implemented UI.**
- Banners were drawn in the prompt band, on top of the prompt text and the
  house-choice modal → moved over the opponent's board.
- After the first player's one card, an Options modal holding only "End
  Turn" covered the board → CHOOSE_ACTION never auto-opens the list.
- "End Turn (nothing else to do)" overflowed its button; the first-turn
  status chip ran into the prompt → shorter labels, prompt kept clear.
- Flank ghost slots overlapped the creature already in play → placed past
  each end of the row.
- Game-over table columns collided; banner subtitles were clipped;
  effect wording contradicted itself ("next turn" vs "this turn") → fixed.
- Found by the new tests: the mulligan's hidden-card set could go stale
  and make your hand unclickable; with no creatures the two flank slots
  were the same rect → both fixed.

**Not done.** Checking every card against Archon Arcana: the wiki
serves a human-verification page to automated requests, so it couldn't be
read. Card behavior was checked against the printed text on the card art
instead (official FFG text). A pass against Archon Arcana's rulings is
still open.

## 5. Live-play screenshot session

75 complete games were played through the real GUI click path: human as P1
vs the bot, human as P2 vs the bot, and hot-seat. The "human" clicked at
random among what the UI offered (cards, modal rows, flank slots, action
bar), with random hovers and right-click inspections. Screenshots were taken
at decisions, mid-animation and at game over, and read by eye. Every idle
frame was also checked automatically:
- HUD Æmber against the engine.
- The drawn snapshot against a fresh one: players, cards, zones, damage,
  exhaustion.
- Each card sprite in its slot, visible, and face-up only when it should be.
- A click on a card's centre hits that card.
- At game over, the winner has 3 keys and the title matches the seat.

**Win condition.** Not reproduced. Every game ended by 3 keys, with the
winner holding three keys and the title matching.

**Fixed:**
- **Escape left the game at once.** One stray Esc, for example for a
  popup that had already closed, threw the game away. A human-seat game now
  asks for a second Esc within 3 s. Spectating still leaves at once.
- **Right-click could open the wrong card.** Inspect used the previous
  frame's hover. It now resolves the card under the click itself.
- **Three Fates asked pointless questions.** It asked for a choice among
  tied creatures even when all of them were going to be destroyed. It now
  asks only when a tie crosses the cut-off, as "choose N of the tied".
- **Log gaps:**
  - Chains gained (Gateway to Dis, Arise) weren't logged.
  - Chains shed during the draw step weren't logged: "draw 1 fewer card
    because of chains".
  - Captured Æmber returning when its creature left play wasn't logged.
    It now also gets a "+N Æ" floater.
- **Log wording.**
  - Lasting-effect lines said "the opponent can't forge a key", which read
    backwards when the opponent played the card. They now name who is hit:
    "Miasma: You can't forge a key next turn."
  - "X is destroyed" now comes before what leaving play causes.

**Checked and correct:**
- Hand refill chips (Mother, The Howling Pit, Succubus) and chains
  lowering the refill.
- Titan Mechanic's key cost for both players.
- Phase Shift allowing a Dis creature.
- Too Much To Protect, Old Bruno capture/release, Pawn Sacrifice targets,
  Guardian Demon excluding the healed creature, and Control the Weak.

**Noted, not changed:**
- **Crowded rows.** Rows of 4+ artifacts or 8 creatures overlap heavily.
  Hover and inspect still pick the right card.
- **Bot targeting.** The bot sometimes damages its own creatures with
  Pawn Sacrifice.
