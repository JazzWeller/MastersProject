"""Game-over overlay: winner, reason, keys/turns, Rematch / Main menu."""

from __future__ import annotations

import pygame

from .. import settings as S
from ..app import Scene
from ..engine_bridge import MatchSettings
from ..sprites.widgets import Button, draw_panel


class GameOverScene(Scene):
    def __init__(self, result: dict, settings: MatchSettings):
        self.result = result or {}
        self.settings = settings
        self.buttons = {}

    def on_enter(self) -> None:
        w, h = 200, 50
        cx = S.CANVAS_W // 2
        y = S.CANVAS_H // 2 + 70
        self.buttons["rematch"] = Button(pygame.Rect(cx - w - 10, y, w, h), "Rematch", primary=True)
        self.buttons["menu"] = Button(pygame.Rect(cx + 10, y, w, h), "Main Menu")

    def handle_event(self, event: pygame.event.Event) -> None:
        mouse = pygame.mouse.get_pos()
        for b in self.buttons.values():
            b.update_hover(mouse)
        if self.buttons["rematch"].clicked(event):
            from .game_scene import GameScene

            self.app.pop()  # this scene
            self.app.pop()  # the finished GameScene
            new_settings = MatchSettings(**{**self.settings.__dict__, "seed": None})
            self.app.push(GameScene(new_settings))
        elif self.buttons["menu"].clicked(event):
            from .menu_scene import MenuScene

            self.app.pop()  # this scene
            self.app.pop()  # the finished GameScene
            if not self.app.scenes:  # launched directly (no menu underneath) -> make one
                self.app.push(MenuScene())
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.app.pop()

    def draw(self, surface: pygame.Surface) -> None:
        rect = pygame.Rect(0, 0, 560, 320)
        rect.center = (S.CANVAS_W // 2, S.CANVAS_H // 2 - 20)
        draw_panel(surface, rect, alpha=245, border=S.KEY_GOLD)

        winner = self.result.get("winner")
        reason = self.result.get("reason", "")
        title_font = self.app.assets.font("cinzel", 30)
        if winner:
            title = f"Player {winner} wins!"
        else:
            title = "It's a draw"
        img = title_font.render(title, True, S.KEY_GOLD)
        surface.blit(img, img.get_rect(center=(rect.centerx, rect.top + 60)))

        sub_font = self.app.assets.font("inter", 16)
        sub = sub_font.render(f"Reason: {reason}" if reason else "", True, S.TEXT_DIM)
        surface.blit(sub, sub.get_rect(center=(rect.centerx, rect.top + 100)))

        turns = self.result.get("turns")
        if turns is not None:
            t = sub_font.render(f"Turns played: {turns}", True, S.TEXT_DIM)
            surface.blit(t, t.get_rect(center=(rect.centerx, rect.top + 128)))

        for b in self.buttons.values():
            b.draw(surface, self.app.assets)
