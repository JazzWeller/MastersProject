# Phase 1.1 (Archon) — GUI Implementation Plan

**Scope:** A graphical, fully playable client for KeyForge Phase 1.1 (Archon, Fignor vs. Igor), with card images and animation. It is built on the existing engine in `Code/Non-GUI/keyforge/` and does **not** reimplement any rules. The engine stays the single source of truth; the GUI only renders `Game` state and submits choices for `game.pending_decision`.

**Sources:** *Master's Project specs and guide.pdf*, `Code/Non-GUI/PHASE_1_1_PLAN.md` (engine plan, §2.1 says the GUI lives in `Code/GUI/` and imports the engine), the engine code as of 9/14 (73 tests passing), and the card art in `Phase 1/Cards/<House>/*.png` (300×420 RGBA, full card text printed on the image).

**Status:** Draft, ready for review.

---

## 1. Technology choices

| Decision | Choice | Why |
|---|---|---|
| Language | Python 3.12 | Matches the engine plan's rule of one language for the whole project. The GUI imports `keyforge` directly. |
| Graphics library | **pygame-ce 2.5.x** (`pip install pygame-ce`) | Full control of the frame loop, cheap sprite transforms (scale, rotate, alpha), and easy particles. That suits card animation better than Qt widgets or tkinter. It is actively maintained and ships Windows wheels. |
| Image processing | pygame `smoothscale` (Pillow 10.4 is already installed if needed for offline pre-processing) | |
| Fonts (online) | **Cinzel** (titles, banners) + **Inter** (UI text) from Google Fonts, OFL license, bundled in `assets/fonts/` | Falls back to pygame's default font if the files are missing |
| Sound (online, optional) | **Kenney.nl "Casino Audio"** (card slide, place, shuffle) and **"Interface Sounds"** (clicks, confirms), CC0 | Mute toggle. The game runs silently if the files are missing. |
| Icons (online, optional) | game-icons.net (CC BY 3.0) for house emblems, key, and chain. Credited in `CREDITS.md`. | Fallback: simple shapes drawn in code (gem polygon, key outline) |
| Tests | `unittest`, headless via `SDL_VIDEODRIVER=dummy` | Same test runner as the engine |

All downloaded assets are listed in `Code/GUI/CREDITS.md` with their license. Nothing is loaded from the network at runtime.

---

## 2. Visual design: "Æmber Forge"

A dark, arcane tabletop lit by glowing æmber, so the colorful card art stands out.

| Token | Color | Use |
|---|---|---|
| `bg_deep` | `#0E0B16` | Window background (subtle radial vignette) |
| `felt` | `#1A1526` → `#241C33` | Board play-mat gradient |
| `aember` | `#F2A93B` (glow `#FFD27A`) | Æmber counters, legal-move glow, primary buttons |
| `key_gold` | `#E8C15A` | Forged keys |
| `text` / `text_dim` | `#EDE6F5` / `#9C92AE` | UI text |
| `danger` | `#E5484D` | Damage numbers and tokens, destroy tint |
| `heal` | `#46C18F` | Heal numbers |
| House **Dis** | `#D0246E` | Matches the magenta banner on Dis cards |
| House **Logos** | `#2F9BD6` | |
| House **Shadows** | `#4E9C6B` | |
| `purge` | `#8B5CF6` | Purge swirl, purged pile |

- **Typography:** Cinzel for turn banners, player names, and the title. Inter for the log, buttons, and prompts.
- **Card treatment:** rounded-corner mask on the art, soft drop shadow, and a 3px outer glow when a card is a legal choice (æmber) or selected (white). Exhausted cards are rotated 90° and dimmed to 80% brightness.
- **Base resolution:** 1600×900 logical canvas, scaled to the window with letterboxing, so the layout math stays in one coordinate system. The window is resizable and has a fullscreen toggle (F11).

### 2.1 Board layout (logical 1600×900, viewer always at the bottom)

```
┌───────────────────────────────────────────────────────────────┬──────────────┐
│  [opp hand: face-down fan, half off-screen]                    │              │
│  OPP HUD: name · house badge · Æ 4 · 🔑🔑○ · chains 2 · effects │  CARD ZOOM   │
│  [deck][discard][archive][purged]      opp artifacts row       │  (hovered    │
│                                        opp creatures row       │   card at    │
│ ─────────────── prompt / turn banner line ──────────────────── │   300×420)   │
│                                        your creatures row      │              │
│  [deck][discard][archive][purged]      your artifacts row      ├──────────────┤
│  YOUR HUD: name · house badge · Æ 7 · 🔑○○ · chains 0 · effects │  GAME LOG    │
│  [your hand: face-up fan, hover lifts card]     [END TURN]     │  (scrolls)   │
└───────────────────────────────────────────────────────────────┴──────────────┘
          main board ≈ 1300px wide                                 side ≈ 300px
```

- **Card sizes:** board cards are 105×147 (147 wide when exhausted and rotated), hand cards 120×168, and zoom 300×420 (native). Rows spread cards out evenly and overlap them when more than about 8 are present.
- **Upgrades:** drawn tucked under their host creature, offset about 18px upward, and shown in the zoom on hover.
- **Tokens on creatures:** a red damage drop with a number and an amber gem with the captured-æmber count. Small badges show Elusive and Skirmish when active.
- **Piles:** decks show a stacked card back with a count. The discard and purged piles show their top card face-up. The archive shows backs plus a count, and the owner can click to browse it.
- **Card back:** drawn in code (dark purple, gold filigree border, central æmber gem), then cached.

---

## 3. Architecture

```
Code/GUI/
  PHASE_1_1_GUI_PLAN.md
  README.md
  CREDITS.md
  requirements.txt              # pygame-ce>=2.5,<2.6
  main.py                       # entry: python main.py [--p1 human|random] [--p2 ...] [--p1-deck ...] [--seed N] [--first p1|p2] [--speed 1.0]
  assets/
    fonts/  sounds/  icons/     # downloaded assets (see §1); card art is read from ../../Phase 1/Cards
  gui/
    settings.py                 # theme tokens, sizes, animation timings, paths, user prefs (speed, sound, confirm end turn)
    app.py                      # App: window, 60 FPS clock, logical-canvas scaling, scene stack, global hotkeys
    assets.py                   # AssetCache: card image by Card.image path, scaled/rotated variants cached by (path,size,rot); card back; fonts; sounds
    engine_bridge.py            # EngineBridge: owns Game + seat types; submit(); before/after snapshots; new log events; bot stepping
    snapshot.py                 # BoardSnapshot: instance_id -> CardState(zone, owner, controller, index, exhausted, damage, captured, host_iid, face_up_for)
    layout.py                   # Layout(viewer): rects for every zone/slot; slot_for(card_state); flank ghost slots; hand fan curve
    option_labels.py            # human-readable text for any option/log event (Card, action dataclasses, House, bool, int pile id, "reap"/"fight"/"action", "effect"/"check", TriggerEffect, tuple)
    sprites/
      card_sprite.py            # CardSprite: x,y,rot,scale,alpha,flip(0..1),glow,tint; face/back; tokens; hit-testing with rotation
      hud.py                    # PlayerHUD: animated æmber counter, 3 key slots, chains, house badge, status chips for active effects
      piles.py                  # PileWidget for deck/discard/archive/purged
      widgets.py                # Button, IconButton, Tooltip, Toast, Banner, Modal, ScrollList, CardGridBrowser
      log_panel.py              # scrollable game log with house-colored card names
    anim/
      easing.py                 # linear, ease_in/out cubic, ease_out_back, ease_in_out_sine
      tween.py                  # Tween(target, attr, to, dur, ease), Parallel, Sequence, Delay, Call
      animator.py               # runs the beat queue; is_busy blocks input; speed multiplier; "skip" (Space) jumps to end
      particles.py              # ParticleSystem: æmber sparks, gem arcs, destroy embers, purge swirl, key-forge burst, confetti
      director.py               # (before snapshot, events, after snapshot) -> list of beats (see §5)
    scenes/
      menu_scene.py             # title, deck pickers, seat types (Human / Random bot), first player, seed, speed, Start
      game_scene.py             # board rendering + input routing to DecisionUI + animator
      curtain_scene.py          # hot-seat "Pass to Player N" screen hiding hands
      game_over_scene.py        # overlay: winner, keys, turns; Rematch / Menu
    decision/
      base.py                   # DecisionUI interface: on_enter(decision, view), handle_event, draw, result -> choice | None
      mulligan.py  house.py  archive.py  action.py  flank.py  cards.py  order.py  yes_no.py
      fallback_panel.py         # F1: raw list of decision.options as buttons; always works for any decision
  tests/
    test_assets.py  test_option_labels.py  test_snapshot_layout.py  test_director.py  test_headless_autoplay.py
```

### 3.1 Main loop (no threads)

```python
while running:
    dt = clock.tick(60) / 1000
    for event in pygame.event.get():
        scene.handle_event(event)          # ignored by the board while animator.is_busy (except Space = skip, hover zoom, log scroll)
    animator.update(dt * speed)
    if not animator.is_busy and not game.is_over:
        d = game.pending_decision
        if bridge.seat_is_bot(d.player):
            bridge.step_bot(dt)            # think delay -> preview beat (highlight chosen card) -> submit
        elif decision_ui is None:
            decision_ui = DecisionUI.for_kind(d.kind)(d, game.view_for(d.player))
        elif (choice := decision_ui.result) is not None:
            bridge.submit(choice)          # snapshot before/after + new events -> director -> animator
            decision_ui = None
    scene.draw(screen)
```

- **EngineBridge.submit** follows the same order as the engine's own loop in `text_ui/main.py`: `before = snapshot(game)`, `start = len(game.log.events)`, `game.submit(choice)`, `events = game.log.events[start:]`, `after = snapshot(game)`, then `animator.enqueue(director.build(before, events, after))`.
- **Bots** use the existing `bots.random_bot.RandomBot` through the same `Controller.decide(view, decision)` call. Any future Phase 1 algorithm bot plugs in without GUI changes.
- **Engine import:** `main.py` adds `Code/Non-GUI` to `sys.path`. No packaging is needed.
- **Hidden information:** decisions, labels, and hand contents come from `game.view_for(viewer)`. The snapshot tracks every card by `instance_id` so animations can follow them, but a sprite only gets its face image once the viewer is allowed to see it (own hand/archive, public zones, or a revealed play).

### 3.2 Seats and viewer

| Mode | Bottom seat (viewer) | Curtain |
|---|---|---|
| Human vs. bot | The human, always | None |
| Human vs. human (hot-seat) | Whoever owns the pending decision; the board flips with a 400ms cross-fade | "Pass to Player N" screen whenever the deciding human changes (turn start, P2 mulligan). Hands stay hidden until "I'm ready" is clicked. |
| Bot vs. bot (spectate) | Player 1 | None. A "Reveal hands" toggle shows both hands face-up. Speed slider and pause (P). |

---

## 4. Small, additive engine changes (Milestone 0)

The GUI needs to know **which physical card** each event refers to. Today, log events only carry `card=card.name`, which is ambiguous for duplicates (Mother ×2, Wild Wormhole ×3). These changes are additive keyword arguments, so the text UI, simulator, and all 73 existing tests keep working.

1. **Instance ids in log events.** Add `iid=card.instance_id` (and `player`/`owner` where missing) to every card-related `log.add` in `keyforge/effects/steps.py`, `keyforge/effects/named.py`, and `keyforge/game.py`:
   - `archive`, `discard`, `discard_random`, `purge`, `put_on_top`, `put_on_bottom`, `damage`, `heal`, `return_to_hand`, `shuffle_into_deck`, `ready`, `exhaust`, `capture`
   - `play_card` (+ `type`, `flank`, `from_deck_top`), `discard_from_hand`, `use_action`, `use_omni`, `reap`, `destroyed` (+ `destination`)
   - `fight` gets `attacker_iid` and `target_iid`
   - `draw` gets `iids=[...]`, the ids of the drawn cards
   - `arise` and `help_from_future_self` get `iids` for the cards they move
2. **Card image paths.** Fill in `CardDef.image` for all 49 cards, which the spec's Card class already lists ("Image: image of the card"). The path is `"<House>/<slug>.png"`, relative to `Phase 1/Cards/`. One alias is required: `Timetraveler` → `Logos/timetraveller.png`. Every other name slugifies correctly (verified).
3. **Public effect summary (optional, nice-to-have).** Add `active_effects: List[dict]` to `PlayerView` with `{source, player_affected, variable, value, remaining}` for public lasting effects (Miasma, Scrambler Storm, Lash, Lifeward, Control the Weak, Phase Shift, Library Access, passives). The HUD shows these as status chips. Without this change, the bridge can read `game.active_effects` directly, since all of it is public information.
4. Add one engine test that checks every log event with a `card` key also has an `iid`, then re-run `python -m unittest discover -s tests`.

---

## 5. Animation system

### 5.1 Approach: beats, then a settle pass
- **The problem:** one `submit()` can run many state changes. For example, Gateway to Dis destroys every creature, Dust Imp gains æmber, and captured æmber is returned. By the time the GUI animates, the engine is already in the final state.
- **Beats:** `Director.build` walks the new log events **in order** and turns each one into a *beat*, a small group of tweens played in sequence. Beats move sprites to the event's destination zone and update the HUD's **displayed** values (æmber, keys, chains) step by step.
- **Settle pass:** a final beat always tweens **every** sprite to its slot in the `after` snapshot (position, rotation, face, tokens) and snaps the HUD to the true values. So even if an event has no animation, or an unusual interaction happens, the screen always ends in the correct state. Unknown events simply produce no beat.
- **Input:** blocked while the animator is busy. **Space** skips to the end of the queue, **1/2/3** set the animation speed (0.5×/1×/2×), and a settings toggle turns animations off entirely (instant).

### 5.2 Event → beat catalogue (1× speed)

| Event | Animation | ms |
|---|---|---|
| Setup deal / `draw` | Card backs slide from deck to hand with an 80ms stagger. Own cards flip face-up mid-flight (scaleX 1→0, swap face, 0→1). The hand fan re-spaces. | 300 each |
| `mulligan` | Hand sweeps into deck, deck jitters (shuffle), then redeal | 900 |
| `reshuffle` / `shuffle_into_deck` | Discard cards fly into the deck, then the deck jitters | 500 |
| `choose_house` | Banner slides in from the left: "Turn 7 · Igor · LOGOS" with the house emblem and color wash. Usable cards pulse once. | 900 |
| `take_archive` | Archived cards fan out of the archive box into the hand | 400 |
| `play_card` creature/artifact | Card lifts from hand (or flips from the opponent's hand or deck top), arcs to its flank slot with ease-out-back, neighbors slide over, then it rotates to exhausted. Æmber bonus gems fly to the pool. | 450 + 200 |
| `play_card` action | Card flies to a center "stage" at 1.4× scale with a house-colored glow and holds (so bot plays can be read), then resolves. It dissolves to discard after its effects' beats. | 350 + hold 700 |
| `play_card` upgrade | Card flies to the host and tucks under it with a small shimmer | 450 |
| `discard_from_hand` / `discard` / `discard_random` | Card slides to the discard pile with a slight spin. Random discards show a card back "shuffling" in the opponent's hand first. | 350 |
| `reap` | Creature pulses, exhausts, and 1 æmber gem arcs to the pool; the counter pops. | 500 |
| `fight` | Attacker lifts and lunges 60% toward the target (ease-in), both shake on impact, red damage numbers float up, then the attacker returns. If Elusive stops the fight, the target shows a "shimmer" with the text "Elusive!" and nothing takes damage. | 700 |
| `damage` | Red flash, 6px shake, floating "−N"; the damage token counts up | 400 |
| `heal` | Green shimmer, floating "+N" | 350 |
| `destroyed` | Red tint, then crack and ember particles as the card shrinks and fades. A ghost card flies to the discard pile, or to hand for Bad Penny. Upgrades peel off to their owners' discard piles. | 650 |
| `purge` | Violet swirl particles; the card spirals into the purged pile | 600 |
| `archive` | Card slides into the archive box (face-down for the opponent) | 350 |
| `return_to_hand` | Card lifts and flies to its owner's hand (flips face-down if that's the opponent) | 400 |
| `gain` / `steal` / `capture` | Æmber gems (one per æmber, max 8) arc between pools or pool ↔ creature token, with pops on arrival | 500 |
| Captured æmber released | Gems arc from the leaving creature to the opponent's pool | 400 |
| `forge_key` | Pool gems converge on the next key slot, the key flips from outline to gold, a burst plus a brief screen bloom play, and a "Key Forged!" banner appears | 1200 |
| Chains change | The chain icon rattles and the count ticks | 400 |
| `duration_effect` | A status chip (e.g. "Miasma · can't forge · 1 turn") slides into the affected player's HUD. Chips fade out when effects expire (found by the settle diff). | 400 |
| Turn cleanup (settle diff) | Exhausted cards rotate upright together, then the end-of-turn draw | 300 |
| Game over | Winner's keys glow, æmber confetti falls, results panel rises | 1500 |

### 5.3 Bot pacing
Before each bot submit, the bridge plays a short **preview beat**: a 500ms think delay, then the chosen card glows with an "Opponent: Reap with Pit Demon" toast. Fights draw an arrow to the target. This way the human can follow what the bot is doing. In spectate mode the think delay is controlled by the speed slider.

---

## 6. Interaction design (one DecisionUI per `DecisionKind`)

Every decision shows its `decision.prompt` on the center prompt line. **F1** opens the fallback panel, which lists the raw `decision.options` as labeled buttons. It is a guaranteed way to answer any decision, including combinations the richer UI below doesn't anticipate.

| Kind | UI |
|---|---|
| `MULLIGAN` | Hand shown enlarged in the center. **Keep** / **Mulligan** buttons; the note says "Mulligan draws one fewer card". |
| `CHOOSE_HOUSE` | Three large house emblems. Each shows how many hand cards and ready in-play cards belong to that house (counted from the view). Hovering one highlights those cards on the board. |
| `TAKE_ARCHIVE` | Archive contents fanned in a tray. **Take all into hand** / **Leave them**. |
| `CHOOSE_ACTION` | **Direct manipulation.** Cards with at least one legal option glow æmber; everything else is dimmed. Clicking a card opens a small radial menu of *only* its legal options ("Play", "Discard", "Reap", "Fight", "Action", "Omni"). If a card has exactly one legal option, a double-click does it. **Drag** a hand card onto your play area to Play it (the drop position picks the flank, see below) or onto your discard pile to Discard it. A large **End Turn** button is always shown; an optional confirmation appears when legal plays remain. |
| `CHOOSE_FLANK` | Ghost slots pulse at the left and right ends of your creature row; click one. If the Play started with a drag, the drop side is remembered and submitted automatically. |
| `CHOOSE_CARDS` (Card options) | Eligible cards glow, other cards dim, and a "targeting" cursor appears. Click to toggle; a counter shows "Selected 1 / up to 2". **Confirm** is enabled when `min_n ≤ k ≤ max_n`. If `min_n == max_n == 1`, a click submits immediately. If `min_n == 0`, a **Choose none** button appears. When options sit in a hidden or stacked zone (discard pile, archive, deck), a **CardGridBrowser** modal opens with only those cards. |
| `CHOOSE_CARDS` (non-Card options) | Labeled buttons via `option_labels`: pile ids `1`/`2` become "Your discard pile (5)" / "Opponent's discard pile (3)" (Creeping Oblivion), and `"reap"`/`"fight"`/`"action"` become ability buttons on the creature (Dominator Bauble). |
| `CHOOSE_HOUSE_FOR_EFFECT` | Same emblem picker as `CHOOSE_HOUSE`, limited to `options` (Arise, Control the Weak) |
| `ORDER_EFFECTS` | Modal with one chip per item. Click the chips in the order they should resolve (numbers 1..n appear), or drag to reorder, then **Confirm**. Labels: `"effect"` → "‹card› Play effect", `"check"` → "Play triggers (Library Access: draw)", `TriggerEffect` → "‹source›: ‹description›", `Card` → "‹name› Destroyed effect", `("extra", card, fn)` → "‹name› upgrade effect". The card name for `"effect"`/`"check"` comes from the latest `play_card` event. |
| `YES_NO` | Two buttons under the prompt |

**Always available (never uses a decision):**
- Hover any visible card to see it at full size in the zoom panel. Right-click pins the zoom.
- Click the discard, purged, or your own archive pile to browse it.
- **Decklist** button: grid of both players' 36-card lists, unordered (allowed by the spec).
- Log panel with sentences like "Igor steals 1 Æmber with Nerve Blast". Hovering a card name in the log highlights that card.
- Status chips show lasting effects. Hovering a chip explains the effect and its remaining duration.
- Esc opens a pause menu: Resume, Settings (speed, sound, confirm end turn, reveal hands in spectate), Concede, Main Menu.

---

## 7. Screens
1. **Title / Setup:** "KEYFORGE · Phase 1.1 Archon" in Cinzel over a slow particle drift of æmber motes. Controls:
   - Deck picker per player: Fignor or Igor, with house emblems and a "view list" button
   - Seat type: Human or Random Bot
   - First player: Random, P1, or P2
   - Optional seed, and animation speed
   - **Start**
2. **Game:** see §2.1 and §6. The opening deal and mulligans play as a short sequence.
3. **Curtain (hot-seat only):** full-screen blur of the board with "Pass to Player 2" and a ready button.
4. **Game over:** "Fignor forges the third key!" (or "Draw — turn limit"), turns played, keys, and æmber. Buttons: **Rematch** (same settings and a new seed), **Swap seats**, **Main menu**.

---

## 8. Testing
1. **`test_assets.py`:** every card in both decklists resolves to an existing image and loads at every cached size; the card back renders.
2. **`test_option_labels.py`:** every option type from §6 and every log event kind produces a non-empty label.
3. **`test_snapshot_layout.py`:** snapshot contents match `view_for` visibility (the opponent's hand is never face-up for the viewer). The layout returns non-overlapping, on-screen slots for rows of 0–12 cards, and flank ghost slots sit at the row ends.
4. **`test_director.py`:** scripted games (reusing `Non-GUI/tests/helpers.py`) produce the expected beat kinds: Gateway to Dis with Dust Imp and Bad Penny, Wild Wormhole playing Library Access, a fight with an Elusive target, Old Bruno capture then destroyed.
5. **`test_headless_autoplay.py`:** 200 bot-vs-bot games run through `EngineBridge` + `Director` + `Animator` at instant speed with `SDL_VIDEODRIVER=dummy`. Checks after each settle:
   - no exceptions
   - every `instance_id` sprite sits in the same zone and index as the snapshot
   - the HUD's displayed values equal the true values
6. **Manual playtest checklist** (in the README):
   - every `DecisionKind` answered through the rich UI *and* through F1
   - Wild Wormhole + Library Access in both orders
   - Three Fates tie-break
   - Creeping Oblivion pile choice
   - Dominator Bauble reap/fight/action
   - Lights Out choosing 0, 1, and 2
   - Duskrunner on an enemy creature
   - Titan Mechanic moved off the flank
   - Lifeward and Scrambler Storm chips expiring
   - first-turn rule (only one play or discard offered)
   - rule of 6
   - hot-seat curtain never leaks a hand
   - window resize and fullscreen
   - Space-skip in the middle of a long chain

---

## 9. Milestones
Each milestone ends in a runnable state.

0. **Engine prep** (§4): instance ids in log events, image paths, optional effect summary; the 73 engine tests plus the new log test pass.
1. **Scaffold:** `requirements.txt`, `main.py`, `App` with scene stack and logical-canvas scaling, `settings.py` theme, `AssetCache` (card images, generated card back, fonts); `test_assets.py`.
2. **Static board:** `BoardSnapshot`, `Layout`, `CardSprite` (no tweens yet), HUD, piles, zoom panel, log panel. Sprites snap straight to their slots. Human vs. bot is **playable end to end using only the F1 fallback panel.**
3. **Rich decisions:** every DecisionUI from §6: glow and dim, radial action menu, drag-to-play and drag-to-discard, flank ghosts, targeting with the grid browser, order-effects modal, house picker. `test_option_labels.py`.
4. **Animation core:** easing, tweens, animator with input blocking, skip and speed, director beats for card movement (draw, play, discard, archive, purge, return, destroy) plus the settle pass. `test_director.py` and `test_headless_autoplay.py`.
5. **Juice:** particles (gems, embers, purge swirl, key burst, confetti), fight lunge and shake, floating numbers, banners, status chips, bot preview beats, sounds.
6. **Screens and modes:** title/setup, hot-seat curtain with board flip, spectate mode, pause and settings, game over with Rematch.
7. **Polish and docs:** resize and fullscreen, performance pass (cache scaled and rotated surfaces; aim for a steady 60 FPS with about 40 sprites), README with run instructions and the playtest checklist, `CREDITS.md`, a full manual playthrough of each seat mode.

**Out of scope:** Reversal (1.2) and Adaptive bidding (1.3) screens, the Phase 1 algorithm bot, networking/online play, and replay save/load (possible stretch goal, since the engine already records seed + choices, but choices are object references and would need index-based serialization first).

---

## 10. Decisions log (formerly "open questions" — all resolved)
1. **pygame-ce** is the one third-party dependency. Approved.
2. The §4 engine additions were made directly in `Code/Non-GUI/keyforge/` (instance ids on log events, `CardDef.image`, `PlayerView.active_effects`), all additive — the 74 engine tests still pass, plus a new one (`test_log_iids.py`) guarding the addition itself.
3. Fonts (Cinzel, Inter — Google Fonts, OFL) are vendored under `assets/fonts/`. Icons are drawn procedurally in `gui/assets.py`, so nothing to fetch or credit. Sound effects (`assets/sounds/`) are procedurally synthesized by `tools/generate_sounds.py` from the standard library alone, rather than downloaded — see `CREDITS.md`.

## 11. Delivered vs. the plan
The GUI was implemented in two passes. What shipped, and where it differs
from this document, is tracked in `Code/GUI/README.md` under "Notable
simplifications" — in short: `gui/decision/panel.py` unifies §6's
per-`DecisionKind` widgets into one card-click-or-Options-modal flow rather
than nine separate files (still answers every decision, including a
guaranteed fallback); the opening deal doesn't animate; a card's face syncs
at the start of a move rather than mid-flight; the hot-seat switch is
instant rather than a cross-fade. Small features beyond the original
plan: a decklist viewer (**D**, per the spec's "view either deck's full
list at any time"), an in-game control-reference overlay (**H**/**/**), a
mute toggle (**M**), and spectate's "reveal hands" toggle (**R**).
