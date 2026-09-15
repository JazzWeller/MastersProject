# Third-party assets

All assets used by the GUI, and where they came from.

## Fonts

- **Cinzel** (Bold) — used for titles, banners, and house names.
  SIL Open Font License 1.1. https://fonts.google.com/specimen/Cinzel
  Fetched from the `google/fonts` repository (`ofl/cinzel/Cinzel[wght].ttf`).
- **Inter** (variable) — used for all UI text (HUD, buttons, log, tooltips).
  SIL Open Font License 1.1. https://fonts.google.com/specimen/Inter
  Fetched from the `google/fonts` repository (`ofl/inter/Inter[opsz,wght].ttf`).

Both are variable fonts; pygame renders them at their default (regular)
instance. `gui/assets.py` bolds UI text itself (`SysFont`-style synthetic
bold via `set_bold(True)`) rather than shipping a second static weight.

## Card art

`Phase 1/Cards/<House>/<card-slug>.png` — provided by the project owner
(sourced from the KeyForge card database / Archon Arcana community scans for
personal, non-commercial project use). Not modified by the GUI beyond
runtime scaling and rounded-corner masking.

## Icons

No external icon set is used. House emblems, the key, the chain link, and
the æmber gem are drawn procedurally in `gui/assets.py` (simple vector
shapes via `pygame.draw` / `pygame.gfxdraw`), so there is nothing to credit
or attribute and nothing that can go missing at runtime.

## Sound

The six short effects in `assets/sounds/` (`click`, `card_move`, `damage`,
`destroy`, `key_forge`, `gain`) are original, procedurally synthesized by
`tools/generate_sounds.py` from Python's standard library alone (`wave` +
`math`) -- nothing downloaded, nothing to attribute. Re-run that script to
regenerate them, or replace any file with something nicer; `gui/assets.py`
loads whatever `.ogg`/`.wav` files it finds in `assets/sounds/` by name and
silently no-ops for any that are missing, so a swap needs no code change.
Press **M** in-game to mute.
