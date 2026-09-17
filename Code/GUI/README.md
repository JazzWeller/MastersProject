# KeyForge Phase 1.1 (Archon) — GUI

A graphical, animated, fully playable client for Phase 1.1 of the project
(Archon format, Fignor vs. Igor), built on top of the engine in
[`Code/Non-GUI`](../Non-GUI). See
[`PHASE_1_1_GUI_PLAN.md`](PHASE_1_1_GUI_PLAN.md) for the design.

![Screenshot](docs/screenshot.png)

The GUI never reimplements a rule: it renders `Game` state and submits
choices for `game.pending_decision`, exactly like the engine's own text UI.

## Running it

```bash
pip install -r requirements.txt
python main.py
```

That opens the menu: pick each seat's deck (Fignor/Igor) and type
(Human/Bot), who goes first, and Start. Or skip the menu and jump straight
into a match:

```bash
python main.py --p1 human --p2 bot --p1-deck fignor --p2-deck igor --seed 1
```

`--p1`/`--p2` accept `human` or `bot`; the other flags mirror the text UI's
(`--p1-deck`, `--p2-deck`, `--first p1|p2`, `--seed`, `--max-turns`).

## Playing

- **Click a glowing card** to play/discard/reap/fight/use it. If a card has
  more than one legal option (e.g. a hand card that could be played *or*
  discarded), a small chooser opens beside it — the card shown once, with a
  labeled button per option ("Play" / "Discard" / ...). Discarding, and
  ending your turn while other actions are still legal, arm on the first
  click and need a second click ("Confirm?") to actually go through.
- **Options (O)** — always available — lists every legal choice as plain
  buttons or a card grid. It's the guaranteed way to answer *any* decision,
  including target selection from a pile, choosing a house, ordering
  simultaneous effects, and so on.
- **Middle-click** any face-up card — in your hand, on the board, in a pile
  browser, or in a decklist — to open a full-size, readable view of it, with
  its name/house/type/stats alongside. Click anywhere or press Esc to close.
- **Right-click** a card to pin it in the zoom panel on the right; hover
  any face-up card to preview it there.
- **Click a pile** (deck/discard/archive/purged, bottom-left of each side,
  each tinted and outlined differently) to browse its contents, where
  that's allowed (discard and purged are always public; archive only for
  its owner — shown with a small padlock when it isn't; the deck's order is
  always hidden). Click a card in the browser to inspect it full-size.
- **D** opens either player's full 36-card decklist, unordered, as the
  spec allows at any time. **H** (or **/**) shows the control list
  in-game. **M** mutes/unmutes sound.
- **Mulligan** gets its own screen: your whole hand laid out large enough
  to actually read, with a house breakdown and "Keep This Hand" /
  "Mulligan" buttons, rather than a Yes/No prompt over a hand you can't see.
- **Space** skips the current animation; **1/2/3** set animation speed to
  0.5x/1x/2x; **F11** toggles fullscreen; **F3** shows the frame rate;
  **Esc** backs out to the previous screen.
- **Hot-seat** (human vs. human): a "Pass to Player N" screen hides the
  board between turns whenever the pending decision belongs to the other
  player.
- **Spectating** (bot vs. bot, no human seat): players are addressed as
  "Player 1"/"Player 2" throughout rather than "you", and **R** reveals
  both hands face up.

See [`UX_FIX_PLAN.md`](UX_FIX_PLAN.md) for the usability pass that produced
most of the above — what was wrong, why, and how it was fixed.

## Layout

```
gui/
  settings.py        theme colors, sizes, board layout bands, animation timings
  app.py              window, scene stack, logical-canvas scaling, hotkeys
  assets.py           card art (rounded, cached, scaled), procedural card back
                       and icons, fonts, optional sound
  engine_bridge.py     owns the Game + bot seats; before/events/after per submit()
  snapshot.py          BoardSnapshot: every card's zone/position/face_up, per viewer
  layout.py            board bands -> concrete rects and card-fan/row slots
  board.py             the CardSprite pool + HUD/particle/overlay state
  option_labels.py     turns any Decision option or log event into a sentence
  anim/
    tween.py, easing.py, animator.py   a small Tween/Sequence/Parallel/Delay/Call
                                         tree, played one queued "beat" at a time
    particles.py, director.py          particle effects; log event -> beat catalogue
  sprites/
    card_sprite.py, hud.py, piles.py, widgets.py, log_panel.py, overlays.py
  decision/
    panel.py            answers every DecisionKind: card-click + an Options modal
  scenes/
    menu_scene.py, game_scene.py, curtain_scene.py, game_over_scene.py
main.py                 entry point
tests/                  see below
```

### How a move gets animated

`EngineBridge.submit()` snapshots the board before and after the call and
returns the new slice of `game.log.events`. `Director.build()` walks those
events in order and turns each into a short animation ("beat"): a card
flies from hand to its flank, æmber gems spark off to the HUD, a fight
lunge-and-shake, a destroyed card crumbles to embers, and so on — then
always finishes with a catch-all "settle" pass that tweens *every* card to
its true position and every HUD counter to its true value. That settle
pass is what guarantees the board can never end up visually out of sync
with the engine, even for a rules interaction this module doesn't
specifically animate.

## Tests

```bash
python -m unittest discover -s tests
```

- `test_assets.py` — every one of the 49 Phase 1 cards resolves to real
  art at every size the app uses.
- `test_option_labels.py` — every option type and log-event kind produces
  a readable sentence.
- `test_snapshot_layout.py` — board rects stay on-canvas and don't
  collide; hidden zones (deck, the opponent's hand/archive) never carry a
  face-up card for the wrong viewer.
- `test_headless_autoplay.py` — runs full bot-vs-bot games through
  Board + Director + Animator and checks the board never drifts out of
  sync with the engine.
- `test_ui_playthrough.py` — plays whole games through **real pygame click
  events** (not direct engine calls) for human-vs-bot, bot-vs-human, and
  human-vs-human hot-seat, plus menu → game and game-over → rematch/menu
  navigation.
- `test_game_scene_features.py` — the decklist viewer, the help overlay,
  spectate's "reveal hands" toggle, and the you/opponent vs. "Player N"
  phrasing switch.
- `test_input_coords.py` — hover and click resolve to the right card at
  window sizes other than the canvas's own 1600x900 (the bug behind most
  of `UX_FIX_PLAN.md`'s findings).
- `test_decision_ui.py` — the action chooser and Options grid never show
  the same card's art twice; MULLIGAN always opens the dedicated review
  screen; Discard and a premature End Turn always require a second click.
- `test_app_window.py` — letterbox math and window-resize survival.
- `test_performance.py` — a populated board stays comfortably under the
  frame budget.

All run headless (`SDL_VIDEODRIVER=dummy`, set automatically by
`tests/helpers.py`) so they need no display.

## Notable simplifications vs. the design doc

- The opening deal (before the first Decision, i.e. before any mulligan)
  isn't animated — `Game.__init__` deals hands internally before the GUI
  ever sees the game, so there's no "before" state to animate from. Every
  draw after that animates normally.
- A card's face flips to its final, correct state (per the current
  viewer's visibility rules) at the start of each move's animation rather
  than mid-flight; only its *position* is what's actually tweened. Reads
  cleanly in practice and was far simpler to get right.
- The hot-seat viewer switch is instant (no cross-fade) once the "I'm
  Ready" curtain is dismissed.
