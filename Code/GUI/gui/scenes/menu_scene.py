"""Title / setup screen: pick each seat's deck and type, who goes first,
and start a game."""

from __future__ import annotations

import os
import random

import pygame

from .. import settings as S
from ..anim.particles import Particle, ParticleSystem
from ..app import Scene
from ..engine_bridge import MatchSettings
from ..sprites.widgets import Button, draw_panel

from keyforge.cards.decks import DECKS as _BUNDLED_DECKS, deck_label, resolve_deck

# Bundled presets (Fignor, Igor, plus the curated Milestone D decks), sorted
# for a stable cycling order.
_PRESET_DECKS = sorted(_BUNDLED_DECKS.keys())
SEATS = ["human", "bot"]
FIRST_CHOICES = [None, 1, 2]
FIRST_LABELS = {None: "Random", 1: "Player 1", 2: "Player 2"}
FORMATS = ["archon", "reversal", "adaptive"]
FORMAT_LABELS = {"archon": "Archon", "reversal": "Reversal", "adaptive": "Adaptive"}


def _user_deck_files() -> list:
    try:
        files = [f for f in os.listdir(S.USER_DECKS_DIR) if f.endswith(".json")]
    except FileNotFoundError:
        return []
    return sorted(os.path.join(S.USER_DECKS_DIR, f) for f in files)


def _deck_button_label(source) -> str:
    try:
        return deck_label(resolve_deck(source))
    except Exception:
        return str(source)


class MenuScene(Scene):
    def __init__(self):
        self.decks = list(_PRESET_DECKS)
        # Match main.py's own CLI defaults (--p1-deck fignor --p2-deck igor)
        # rather than whatever "cinder"/"fignor" happen to sort to first.
        self.p1_deck_i = self.decks.index("fignor") if "fignor" in self.decks else 0
        self.p2_deck_i = self.decks.index("igor") if "igor" in self.decks else min(1, len(self.decks) - 1)
        self.p1_seat_i = 0  # human
        self.p2_seat_i = 1  # bot
        self.first_i = 0
        self.format_i = 0  # archon
        self.fixed_seed = False
        self.buttons = {}
        self.particles = ParticleSystem()
        self._spawn_timer = 0.0

        # Layout metadata computed by _layout_buttons() and consumed by
        # draw() -- kept as instance state so the two never disagree about
        # where anything actually is.
        self.panel_rect = pygame.Rect(0, 0, 0, 0)
        self.field_labels = []      # [(text, x, y)]
        self.section_headers = []   # [(text, center_x, y)]
        self.dividers = []          # [(x0, x1, y)]
        self.hint_pos = (0, 0)

    def on_enter(self) -> None:
        self._refresh_decks()

    def _refresh_decks(self) -> None:
        """Re-reads the deck list (bundled presets + user decks). Called on
        entry, and explicitly by DeckBuilderScene's Back button -- returning
        to an already-on-the-stack MenuScene doesn't re-trigger on_enter,
        but a deck may have just been saved or deleted there."""
        self.decks = _PRESET_DECKS + _user_deck_files()
        self.p1_deck_i = min(self.p1_deck_i, len(self.decks) - 1)
        self.p2_deck_i = min(self.p2_deck_i, len(self.decks) - 1)
        self._layout_buttons()

    def _layout_buttons(self) -> None:
        """Lays out the whole setup panel as a top-to-bottom stack of
        sections (player columns, format/first/seed, start/quit, secondary
        nav), each full-width within the panel's padded content area. Every
        section's height is computed from its own contents, so later
        sections can never collide with earlier ones or overflow the panel
        -- unlike the old fixed row_h stride, which let the Start/Quit/nav
        buttons run past both the panel's right and bottom edges."""
        cx = S.CANVAS_W // 2
        self.buttons.clear()
        self.field_labels = []
        self.section_headers = []
        self.dividers = []

        PAD = 36
        LABEL_H = 18
        LABEL_GAP = 6
        FIELD_H = 42

        panel_w, panel_h = 880, 600
        panel_x, panel_y = cx - panel_w // 2, 195
        self.panel_rect = pygame.Rect(panel_x, panel_y, panel_w, panel_h)

        content_x0 = panel_x + PAD
        content_x1 = panel_x + panel_w - PAD
        content_w = content_x1 - content_x0

        def field(x, w, y, label_text, key, button_label, cb):
            self.field_labels.append((label_text, x, y))
            by = y + LABEL_H + LABEL_GAP
            self.buttons[key] = (Button(pygame.Rect(x, by, w, FIELD_H), button_label), cb)
            return by + FIELD_H

        # -- Section 1: Player 1 / Player 2 columns (deck, then seat) ------
        col_gap = 40
        col_w = (content_w - col_gap) // 2
        col1_x, col2_x = content_x0, content_x0 + col_w + col_gap

        y = panel_y + PAD
        header_y = y
        self.section_headers.append(("Player 1", col1_x + col_w // 2, header_y))
        self.section_headers.append(("Player 2", col2_x + col_w // 2, header_y))
        y = header_y + 24 + 14

        deck_bottom = field(col1_x, col_w, y, "Deck", "p1_deck", _deck_button_label(self.decks[self.p1_deck_i]), self._cycle_p1_deck)
        field(col2_x, col_w, y, "Deck", "p2_deck", _deck_button_label(self.decks[self.p2_deck_i]), self._cycle_p2_deck)
        y = deck_bottom + 20

        seat_bottom = field(col1_x, col_w, y, "Seat", "p1_seat", SEATS[self.p1_seat_i].capitalize(), self._cycle_p1_seat)
        field(col2_x, col_w, y, "Seat", "p2_seat", SEATS[self.p2_seat_i].capitalize(), self._cycle_p2_seat)
        y = seat_bottom + 34

        self.dividers.append((content_x0, content_x1, y))
        y += 22

        # -- Section 2: Format / First player / Seed, three sub-columns ----
        sub_gap = 30
        sub_w = (content_w - 2 * sub_gap) // 3
        sub_xs = [content_x0, content_x0 + sub_w + sub_gap, content_x0 + 2 * (sub_w + sub_gap)]

        format_bottom = field(sub_xs[0], sub_w, y, "Format", "format", FORMAT_LABELS[FORMATS[self.format_i]], self._cycle_format)
        field(sub_xs[1], sub_w, y, "First player", "first", FIRST_LABELS[FIRST_CHOICES[self.first_i]], self._cycle_first)
        field(sub_xs[2], sub_w, y, "Seed", "seed", "Fixed (42)" if self.fixed_seed else "Random", self._toggle_seed)
        y = format_bottom + 22

        self.dividers.append((content_x0, content_x1, y))
        y += 22

        # -- Deck/house hint line, then Start/Quit ---------------------------
        self.hint_pos = (cx, y)
        y += 18 + 16

        start_w, quit_w, gap = 220, 220, 40
        total = start_w + quit_w + gap
        sx = content_x0 + (content_w - total) // 2
        self.buttons["start"] = (Button(pygame.Rect(sx, y, start_w, 54), "Start Game", primary=True), self._start)
        self.buttons["quit"] = (Button(pygame.Rect(sx + start_w + gap, y, quit_w, 54), "Quit"), self.app.quit)
        y += 54

        # -- Secondary nav row, pinned to the panel's bottom edge -----------
        nav_h = 48
        nav_y = panel_y + panel_h - PAD - nav_h
        self.dividers.append((content_x0, content_x1, (y + nav_y) // 2))

        nav_gap = 24
        nav_w = (content_w - 2 * nav_gap) // 3
        nav_xs = [content_x0, content_x0 + nav_w + nav_gap, content_x0 + 2 * (nav_w + nav_gap)]
        self.buttons["deck_builder"] = (Button(pygame.Rect(nav_xs[0], nav_y, nav_w, nav_h), "Deck Builder"), self._deck_builder)
        self.buttons["alliance_builder"] = (Button(pygame.Rect(nav_xs[1], nav_y, nav_w, nav_h), "Alliance"), self._alliance_builder)
        self.buttons["history"] = (Button(pygame.Rect(nav_xs[2], nav_y, nav_w, nav_h), "Past Games"), self._history)

    def _cycle_p1_deck(self):
        self.p1_deck_i = (self.p1_deck_i + 1) % len(self.decks)
        self._layout_buttons()

    def _cycle_p2_deck(self):
        self.p2_deck_i = (self.p2_deck_i + 1) % len(self.decks)
        self._layout_buttons()

    def _cycle_p1_seat(self):
        self.p1_seat_i = (self.p1_seat_i + 1) % len(SEATS)
        self._layout_buttons()

    def _cycle_p2_seat(self):
        self.p2_seat_i = (self.p2_seat_i + 1) % len(SEATS)
        self._layout_buttons()

    def _cycle_format(self):
        self.format_i = (self.format_i + 1) % len(FORMATS)
        self._layout_buttons()

    def _cycle_first(self):
        self.first_i = (self.first_i + 1) % len(FIRST_CHOICES)
        self._layout_buttons()

    def _toggle_seed(self):
        self.fixed_seed = not self.fixed_seed
        self._layout_buttons()

    def _history(self):
        from .history_scene import HistoryScene

        self.app.push(HistoryScene())

    def _deck_builder(self):
        from .deck_builder_scene import DeckBuilderScene

        self.app.push(DeckBuilderScene())

    def _alliance_builder(self):
        from .alliance_builder_scene import AllianceBuilderScene

        self.app.push(AllianceBuilderScene())

    def _start(self):
        fmt = FORMATS[self.format_i]
        settings = MatchSettings(
            p1_deck=self.decks[self.p1_deck_i],
            p2_deck=self.decks[self.p2_deck_i],
            p1_seat=SEATS[self.p1_seat_i],
            p2_seat=SEATS[self.p2_seat_i],
            format=fmt,
            first_player=FIRST_CHOICES[self.first_i],
            seed=42 if self.fixed_seed else None,
            max_turns=None,
        )
        if fmt == "archon":
            from .game_scene import GameScene

            self.app.push(GameScene(settings))
        else:
            from .match_scene import MatchScene

            self.app.push(MatchScene(settings))

    # ------------------------------------------------------------- events ----

    def handle_event(self, event: pygame.event.Event) -> None:
        mouse = self.mouse
        for button, cb in self.buttons.values():
            button.update_hover(mouse)
            if button.clicked(event):
                self.app.assets.play("click", 0.3)
                cb()
                return

    def update(self, dt_ms: float) -> None:
        self.particles.update(dt_ms)
        self._spawn_timer -= dt_ms
        if self._spawn_timer <= 0:
            self._spawn_timer = 220
            x = random.uniform(0, S.CANVAS_W)
            self.particles.particles.append(
                Particle(
                    x, S.CANVAS_H + 10, random.uniform(-6, 6), random.uniform(-70, -30),
                    life_ms=6000, size=random.uniform(1.5, 3), color=S.AEMBER_GLOW, gravity=-4, shrink=False,
                )
            )

    # --------------------------------------------------------------- draw ----

    def draw(self, surface: pygame.Surface) -> None:
        surface.fill(S.BG_DEEP)
        self.particles.draw(surface)

        title_font = self.app.assets.font("cinzel", 54)
        title = title_font.render("KEYFORGE", True, S.TEXT)
        surface.blit(title, title.get_rect(center=(S.CANVAS_W // 2, 110)))
        sub_font = self.app.assets.font("cinzel", 22)
        sub = sub_font.render("Archon · Reversal · Adaptive", True, S.AEMBER)
        surface.blit(sub, sub.get_rect(center=(S.CANVAS_W // 2, 160)))

        draw_panel(surface, self.panel_rect, alpha=200, border=S.TEXT_FAINT)

        header_font = self.app.assets.font("cinzel", 19)
        for text, center_x, y in self.section_headers:
            img = header_font.render(text, True, S.AEMBER)
            surface.blit(img, img.get_rect(midtop=(center_x, y)))

        label_font = self.app.assets.font("inter", 15)
        for text, x, y in self.field_labels:
            img = label_font.render(text, True, S.TEXT_DIM)
            surface.blit(img, (x, y))

        for x0, x1, y in self.dividers:
            pygame.draw.line(surface, (*S.TEXT_FAINT, 90), (x0, y), (x1, y), 1)

        p1_deck = resolve_deck(self.decks[self.p1_deck_i])
        p2_deck = resolve_deck(self.decks[self.p2_deck_i])
        p1_houses = " / ".join(h.value for h in p1_deck.houses())
        p2_houses = " / ".join(h.value for h in p2_deck.houses())
        hint_text = f"{deck_label(p1_deck)} · {p1_houses}      {deck_label(p2_deck)} · {p2_houses}"
        house_hint = self.app.assets.font("inter", 13).render(hint_text, True, S.TEXT_FAINT)
        hx, hy = self.hint_pos
        surface.blit(house_hint, house_hint.get_rect(midtop=(hx, hy)))

        for key, (button, _cb) in self.buttons.items():
            button.draw(surface, self.app.assets)

        hint = self.app.assets.font("inter", 13).render(
            "F11 fullscreen · F3 FPS · Esc back", True, S.TEXT_FAINT
        )
        surface.blit(hint, (S.CANVAS_W // 2 - hint.get_width() // 2, S.CANVAS_H - 26))
