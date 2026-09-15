"""Title / setup screen: pick each seat's deck and type, who goes first,
and start a game."""

from __future__ import annotations

import random

import pygame

from .. import settings as S
from ..anim.particles import Particle, ParticleSystem
from ..app import Scene
from ..engine_bridge import MatchSettings
from ..sprites.widgets import Button, draw_panel

DECKS = ["fignor", "igor"]
SEATS = ["human", "bot"]
FIRST_CHOICES = [None, 1, 2]
FIRST_LABELS = {None: "Random", 1: "Player 1", 2: "Player 2"}


class MenuScene(Scene):
    def __init__(self):
        self.p1_deck_i = 0
        self.p2_deck_i = 1
        self.p1_seat_i = 0  # human
        self.p2_seat_i = 1  # bot
        self.first_i = 0
        self.fixed_seed = False
        self.buttons = {}
        self.particles = ParticleSystem()
        self._spawn_timer = 0.0

    def on_enter(self) -> None:
        self._layout_buttons()

    def _layout_buttons(self) -> None:
        cx = S.CANVAS_W // 2
        top = 300
        row_h = 64
        bw, bh = 170, 42

        def row(y, key_prefix, label_left, cycle_cb, value_label):
            self.buttons[f"{key_prefix}_cycle"] = (
                Button(pygame.Rect(cx + 40, y, bw, bh), value_label),
                cycle_cb,
            )

        self.buttons.clear()
        y = top
        self.buttons["p1_deck"] = (Button(pygame.Rect(cx + 40, y, bw, bh), DECKS[self.p1_deck_i].capitalize()), self._cycle_p1_deck)
        y += row_h
        self.buttons["p1_seat"] = (Button(pygame.Rect(cx + 40, y, bw, bh), SEATS[self.p1_seat_i].capitalize()), self._cycle_p1_seat)
        y += row_h + 20
        self.buttons["p2_deck"] = (Button(pygame.Rect(cx + 40, y, bw, bh), DECKS[self.p2_deck_i].capitalize()), self._cycle_p2_deck)
        y += row_h
        self.buttons["p2_seat"] = (Button(pygame.Rect(cx + 40, y, bw, bh), SEATS[self.p2_seat_i].capitalize()), self._cycle_p2_seat)
        y += row_h + 20
        self.buttons["first"] = (Button(pygame.Rect(cx + 40, y, bw, bh), FIRST_LABELS[FIRST_CHOICES[self.first_i]]), self._cycle_first)
        y += row_h
        self.buttons["seed"] = (Button(pygame.Rect(cx + 40, y, bw, bh), "Fixed (42)" if self.fixed_seed else "Random"), self._toggle_seed)
        y += row_h + 30
        self.buttons["start"] = (Button(pygame.Rect(cx - 110, y, 220, 54), "Start Game", primary=True), self._start)

    def _cycle_p1_deck(self):
        self.p1_deck_i = (self.p1_deck_i + 1) % len(DECKS)
        self._layout_buttons()

    def _cycle_p2_deck(self):
        self.p2_deck_i = (self.p2_deck_i + 1) % len(DECKS)
        self._layout_buttons()

    def _cycle_p1_seat(self):
        self.p1_seat_i = (self.p1_seat_i + 1) % len(SEATS)
        self._layout_buttons()

    def _cycle_p2_seat(self):
        self.p2_seat_i = (self.p2_seat_i + 1) % len(SEATS)
        self._layout_buttons()

    def _cycle_first(self):
        self.first_i = (self.first_i + 1) % len(FIRST_CHOICES)
        self._layout_buttons()

    def _toggle_seed(self):
        self.fixed_seed = not self.fixed_seed
        self._layout_buttons()

    def _start(self):
        from .game_scene import GameScene

        settings = MatchSettings(
            p1_deck=DECKS[self.p1_deck_i],
            p2_deck=DECKS[self.p2_deck_i],
            p1_seat=SEATS[self.p1_seat_i],
            p2_seat=SEATS[self.p2_seat_i],
            first_player=FIRST_CHOICES[self.first_i],
            seed=42 if self.fixed_seed else None,
            max_turns=None,
        )
        self.app.push(GameScene(settings))

    # ------------------------------------------------------------- events ----

    def handle_event(self, event: pygame.event.Event) -> None:
        mouse = pygame.mouse.get_pos()
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
        sub = sub_font.render("Phase 1.1 · Archon", True, S.AEMBER)
        surface.blit(sub, sub.get_rect(center=(S.CANVAS_W // 2, 160)))

        panel_rect = pygame.Rect(S.CANVAS_W // 2 - 320, 260, 640, 520)
        draw_panel(surface, panel_rect, alpha=200, border=S.TEXT_FAINT)

        label_font = self.app.assets.font("inter", 16)
        cx = S.CANVAS_W // 2
        labels = [
            (300, "Player 1 deck"), (364, "Player 1 seat"),
            (448, "Player 2 deck"), (512, "Player 2 seat"),
            (596, "First player"), (660, "Seed"),
        ]
        for y, text in labels:
            img = label_font.render(text, True, S.TEXT_DIM)
            surface.blit(img, (cx - 300, y + 10))

        house_hint = self.app.assets.font("inter", 13).render(
            "Fignor · Dis / Logos / Shadows      Igor · Dis / Logos / Shadows", True, S.TEXT_FAINT
        )
        surface.blit(house_hint, (cx - 300, 726))

        for key, (button, _cb) in self.buttons.items():
            button.draw(surface, self.app.assets)

        hint = self.app.assets.font("inter", 13).render(
            "F11 fullscreen · F3 FPS · Esc back", True, S.TEXT_FAINT
        )
        surface.blit(hint, (S.CANVAS_W // 2 - hint.get_width() // 2, S.CANVAS_H - 26))
