# GUI usability pass — findings and plan

Phase 1.1 is rules-complete and animated, but it isn't *usable*. This
document lists every pain point found by auditing the code and by driving
the real client into each decision state and reading the resulting frames,
then lays out the fixes in dependency order.

Everything below was reproduced, not guessed. Line references are to the
state of the code before this pass.

---

## 1. Findings

### 1.1 Broken — not just awkward

**F1. Every hover test compares window pixels against canvas coordinates.**
The app renders to a fixed 1600×900 canvas and letterboxes it into the
window (`app.py:_recompute_dest_rect`). Mouse *events* are translated into
canvas space (`App._translate`), but every place that polls the mouse
instead of reading the event is not:

| Site | Effect when window ≠ 1600×900 |
|---|---|
| `game_scene._update_hover` | hover-to-inspect targets the wrong card, or none |
| `game_scene.draw` → `draw_pile(hovered=…)` | pile highlight lands on the wrong pile |
| `decision/panel.py:draw` | action-bar buttons highlight wrong / not at all |
| `log_panel.handle_event` | scroll wheel ignored outside the true log rect |
| `curtain_scene`, `menu_scene` | button hover states wrong |

The default window is **1280×720**, so this is wrong *by default* — a
0.8× offset that grows with distance from the top-left. This is the direct
cause of "you can't closely examine your hand at all": hovering a hand card
inspects whatever sprite happens to sit at 0.8× its position, usually
nothing.

**F2. Both hands are clipped off the canvas.** `Layout.fan_slots` applies
its arc lift *downward* (`y = base_y + lift`), so outer cards sink instead
of rising. Measured:

- Your hand occupies y = 709…915 on a 900px canvas — **15px below the
  bottom edge**, and it overlaps the HUD strip (ends at 731) and the action
  bar (855–892).
- The opponent's hand starts at y = **−42** — the top quarter is off-canvas.

**F3. "Play or discard?" shows two identical thumbnails and no words.**
`DecisionPanel._draw_modal` picks a card-grid layout whenever *every* option
carries a `Card` (`grid = all(r.card is not None …)`). When the modal is
scoped to one clicked card, all its options carry the *same* card, so
Play/Discard renders as two identical Snudge thumbnails with no labels —
exactly the reported symptom. Same for Reap/Fight/Use on a creature. The
labels already exist (`describe_option` returns "Play Snudge" /
"Discard Snudge"); they're simply never drawn on this path.

### 1.2 You cannot read a card

The engine has **no rules text**: `CardDef` carries name/house/type/power/
armor and effect callables, nothing printed. The art *is* the rules text.
So being able to see a card at full size is not a nicety, it's the only way
to know what anything does. Currently:

**F4. The zoom panel is smaller than the source art.** The sidebar is 300px
wide, so the zoom renders the card at ~260×364 — *below* the native 300×420
— and then the whole canvas is rescaled to the window on top of that.

**F5. There is no large card inspector at all.** No click-to-enlarge, no
hold-to-peek, nothing that shows a card at or above native size.

**F6. Mulligan gives you no way to look at the hand you're judging.** The
`MULLIGAN` decision has `[True, False]` options, so the fallback modal opens
immediately as a 620×520 panel with two rows in it, dimming the board. Your
seven cards sit underneath at 120×168 each, overlapping, clipped at the
bottom, and (per F1) not hoverable. You are asked to keep or throw a hand
you cannot read.

**F7. Pile browser and decklist grids draw cards at 100×140** with no
hover-zoom and no names — same problem, different screen.

### 1.3 Image quality

**F8. Card art never renders 1:1.** Source art is exactly 300×420. The
pipeline is: art → smoothscale to the on-canvas size → canvas → smoothscale
to the window. On this machine (1920×1080 desktop) the default 1280×720
window downsamples the whole canvas to 0.8×; maximized it upsamples to 1.2×.
Neither is 1:1, and the zoom panel (F4) has already thrown away detail
before that second resample.

**F9. Rotated hand cards are resampled twice.** `CardSprite.draw` calls
`rotozoom` on the already-downscaled face whenever `rot` or `scale` differs
from the default, which is every card in the fan.

### 1.4 Clarity and feedback

- **F10.** Decision modals are fixed 620×520 and sit in the middle of the
  play area behind a full-screen dim, hiding your creatures, the opponent's
  board, and your own key count at the exact moment you're deciding. A
  three-option house pick blacks out the entire game.
- **F11.** The prompt is drawn twice — once in the prompt band, once as the
  modal title — with the first showing as ghost text through the dim.
- **F12.** No persistent turn number, and no clear "it's your turn" signal
  beyond a thin border on a HUD strip.
- **F13.** HUD key icons are 22px and are the first thing a modal covers.
- **F14.** The four piles are identical grey rectangles with 10px labels;
  which is discard and which is archive is a memory test.
- **F15.** Action-bar buttons ("Options", "End Turn") are drawn on top of
  the hand fan.
- **F16.** Empty creature/artifact rows are invisible — no sense of where
  cards will land or how many slots are in play.
- **F17.** The log is a flat grey wall of text; no turn separators, no
  color for damage/æmber/keys, and it scrolls independently of the game.
- **F18.** Nothing shows the current key cost (6 æmber, modified by
  effects) or your progress toward it.
- **F19.** A single click commits an irreversible action with no
  confirmation and no undo.

---

## 2. Plan

Five stages, ordered so each rests on the one before. Stages A and B are
what actually make the game playable; C–E are quality.

### Stage A — fix what's broken

**A1. One mouse position, in canvas space.**
`App` gains `self.mouse_canvas`, recomputed once per frame from
`_to_canvas(pygame.mouse.get_pos())` and also updated on every translated
`MOUSEMOTION`. Replace all six `pygame.mouse.get_pos()` call sites with it
(`Scene` gets a `mouse` property so scenes don't reach through `self.app`).
Add a regression test that drives hover at a non-1:1 window scale and
asserts the correct card is picked.

**A2. Fan geometry that stays on the canvas.**
Flip the arc so outer cards lift *up* (`y = base_y - lift`), shrink the
rotation range, and derive the band from the card height rather than
hardcoding it. Re-tune `BAND_YOUR_HAND` / `BAND_OPP_HAND` so both hands sit
fully inside the canvas with clearance from the HUD and action bar. Extend
`test_snapshot_layout.py` with an assertion that every hand slot's rotated
bounding box is inside the canvas and disjoint from the HUD and action-bar
bands, for hand sizes 1–12.

**A3. A real action chooser.**
Replace the scoped-modal path with a compact popup anchored beside the
clicked card: the card once at readable size, then one labeled button per
option ("Play", "Discard", "Reap", "Fight", "Use Action", "Omni"), each with
its own icon and color, plus a Cancel. Fix the grid heuristic underneath it
too — grid layout only when the options reference *distinct* cards,
otherwise labeled rows — so no path can ever render duplicate thumbnails
again. Test: for every decision kind reachable in a fuzzed game, assert the
rendered rows have distinct labels and that no two grid cells share an
`instance_id`.

### Stage B — let the player read cards

**B1. Full-size card inspector.**
Left-click any face-up card (when it isn't a legal action target), or hold
`Z`/middle-click over one, to open a centered inspector showing the card as
large as the window allows, with name/house/type/power/armor/damage and any
attached upgrades listed beside it. `Esc`/click anywhere closes it. Works
from the board, from the hand, from the pile browser, from the decklist, and
from inside a decision modal.

**B2. Draw it at real resolution.**
`App` gains a "crisp layer": a callback drawn *after* the canvas has been
blitted and scaled, in real window pixels, with canvas→window rect
conversion provided. The inspector and the zoom panel render their card
through it, scaling once from the raw 300×420 surface straight to final
device pixels. This removes the double resample for the two places where
sharpness actually matters. `AssetCache.card_face` already caches per exact
pixel size, so this costs one extra cached surface per distinct size.

**B3. A proper mulligan screen.**
Special-case `DecisionKind.MULLIGAN`: instead of the generic modal, lay the
hand out as a large, evenly spaced, non-overlapping row across the play
area at roughly 200×280 each, each card hoverable and click-to-inspect, with
"Keep this hand" and "Mulligan (draw one fewer)" buttons underneath and the
house distribution summarized ("3 Dis · 2 Logos · 2 Shadows"). Test that it
appears for `MULLIGAN`, that both buttons submit the right boolean, and that
every card in the layout is inside the canvas and disjoint from its
neighbours.

**B4. Readable browsers.**
Raise the pile-browser and decklist grid cells to 140×196, add the card name
under each, add hover-zoom into the sidebar panel and click-to-inspect, and
group the decklist by house with a count per house.

### Stage C — image quality

**C1. Sensible window defaults.** Open at 1600×900 when the desktop can fit
it (this machine: 1920×1080 — it can), otherwise the largest 16:9 box that
fits, so the common case is 1:1 and nothing is resampled. Remember the
size across runs in a small JSON next to the settings module.

**C2. Skip the redundant transform.** In `CardSprite.draw`, take the plain
blit path when `rot == 0 and scale == 1` (already done) *and* use
`smoothscale` + `rotate` rather than `rotozoom` when only one of the two
applies, so a fanned hand card is resampled once.

**C3. Render hand and board faces from the raw art at their final size**
rather than reusing a smaller cached size — already the behaviour of
`card_face`, but add a test that asserts no cached size exceeds the raw
art's dimensions by more than 2× (catches accidental upscaling).

### Stage D — clarity

**D1. Modals that don't eat the board.** Auto-size the panel to its content
(a three-row house pick becomes a small panel, not 620×520), dock it above
the action bar instead of centered over the play area, and drop the
full-screen dim for non-card decisions — dim only the board rows a card
picker is *not* pointing at. Remove the duplicated ghost prompt (F11).

**D2. Turn and phase header.** A slim always-visible strip: turn number,
whose turn, the active house, and "Key cost: 6 æmber (you have 4)" with the
cost pulled from the engine's current `KeyForgeCost` so modifiers show up.

**D3. HUD legibility.** Larger key icons with a forged/unforged count, æmber
count in a proper badge, and chains only when non-zero. Move the HUD strip
clear of anything a modal can cover.

**D4. Pile affordances.** An icon and distinct tint per pile kind, a larger
count, a hover tooltip naming it, and a "click to browse" cue only on piles
that are actually browsable.

**D5. Empty-row ghosts.** Faint dashed slot outlines on empty creature and
artifact rows so the board reads as a board.

**D6. Better log.** Turn separators, color per event class (æmber gold,
damage red, key gold, purge violet), auto-scroll pinned to the bottom unless
the player has scrolled up, and card names in the log highlighted.

**D7. Misclick safety.** Require confirmation for the irreversible and
easily-misclicked ones — discarding from hand, and End Turn while you still
have legal plays — as a small inline "Are you sure?" on the button rather
than another modal.

**D8. Action-bar / hand separation.** Move the action bar clear of the hand
fan once A2 has re-tuned the bands.

### Stage E — tests and docs

- New `tests/test_input_coords.py` — hover and click correctness at 1:1,
  0.8×, and 1.6× window scales.
- New `tests/test_decision_ui.py` — the action chooser, the mulligan screen,
  and the no-duplicate-thumbnails invariant, fuzzed across seeds.
- Extend `test_snapshot_layout.py` with the on-canvas / non-overlap
  assertions for both hands at every hand size.
- Extend `test_ui_playthrough.py` to play through the new action chooser and
  mulligan screen with real click events.
- Update `README.md` controls and `PHASE_1_1_GUI_PLAN.md` §11 to match.

---

## 3. Order of work

1. **A1, A2, A3** — after this the game is no longer actively misleading.
2. **B1, B2, B3, B4** — after this you can actually read what you're playing.
3. **C1–C3** — sharpness.
4. **D1–D8** — polish, largest first (D1, D2, D6).
5. **E** throughout, not at the end.

Stages A and B are the ones that answer the reported complaints directly;
C and D are what stop the same complaints being made again about something
else.

---

## 4. Delivered

All five stages (A–E) were implemented. Highlights:

- **A** — `App.mouse_canvas`/`Scene.mouse` replace every raw
  `pygame.mouse.get_pos()` read; `Layout.fan_slots` gained an `arc_up`
  direction and uses tunable lift/rotation constants; every board band in
  `settings.py` was re-solved as a system (see the comment above
  `BAND_OPP_HAND`) so a full hand fan stays on-canvas at every hand size;
  `DecisionPanel` gained a proper action chooser (`_open_chooser` et al.)
  and the Options modal's grid/rows choice now requires *distinct* cards.
- **B** — a full-size card inspector (middle-click, or click a card in a
  browser/decklist), rendered a second time straight from native art onto
  the real window (`Scene.draw_crisp` / `App.canvas_to_window_rect`) so it
  isn't blurred by the canvas's own scale-to-window step; a dedicated
  mulligan screen (`DecisionPanel._draw_mulligan`); pile-browser/decklist
  grid cells grew to 140x196 with names and click-to-inspect.
- **C** — the window opens at the canvas's own 1:1 resolution by default
  when the desktop allows it; `CardSprite.draw` already skipped rotozoom
  when unrotated/unscaled.
- **D** — content-sized, undimmed-when-small decision modals; a turn
  counter and per-player "Key cost N" readout; tinted, glyph-coded piles
  with a padlock on ones you can't browse; dashed empty-row placeholders;
  a colored, turn-separated log; a two-click confirm on Discard and on
  ending a turn while other actions are legal.
- **E** — `test_input_coords.py`, `test_decision_ui.py`, and extensions to
  `test_snapshot_layout.py` and `test_ui_playthrough.py`.

### Second-pass findings (from driving the real client and reading screenshots)

- **Upgrade cards rendered face-down.** `gui/snapshot.py`'s `PUBLIC_ZONES`
  never included the `"upgrade"` zone, so any upgrade attached to a
  creature in play — public information, same as the creature itself —
  fell through to the same `face_up=False` default as a card in the deck.
  On screen this looked like a small card-back glued to the corner of its
  host creature. Fixed by adding `ZONE_UPGRADE` to `PUBLIC_ZONES`, with a
  regression test (`test_upgrades_are_always_face_up`).
- **A bot-move toast could sit exactly where the next prompt renders.**
  `_preview_bot_choice`'s fallback position was the prompt band's center —
  identical to where `_draw_prompt` draws the *next* decision's text, so a
  still-fading toast and a fresh prompt could render close enough to blur
  together. Moved the fallback to just above the prompt band.
- Everything else exercised via screenshots — mulligan, the action
  chooser, house/action decisions, a populated board, spectate with
  revealed hands, the help overlay, pile browser, decklist, and the
  game-over screen — matched the design with no further issues found.
