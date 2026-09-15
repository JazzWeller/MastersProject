"""Hot-seat "pass the device" screen: fully hides the board (the App only
ever draws the top of the scene stack, so nothing underneath leaks) between
turns whenever the pending decision belongs to the *other* human player."""

from __future__ import annotations

import pygame

from .. import settings as S
from ..app import Scene
from ..sprites.widgets import Button


class CurtainScene(Scene):
    def __init__(self, player: int, on_ready):
        self.player = player
        self.on_ready = on_ready
        self.button: Button = None

    def on_enter(self) -> None:
        self.button = Button(pygame.Rect(S.CANVAS_W // 2 - 110, S.CANVAS_H // 2 + 50, 220, 50), "I'm Ready", primary=True)

    def _continue(self) -> None:
        self.on_ready()
        self.app.pop()

    def handle_event(self, event: pygame.event.Event) -> None:
        self.button.update_hover(pygame.mouse.get_pos())
        if self.button.clicked(event):
            self._continue()
        elif event.type == pygame.KEYDOWN and event.key in (pygame.K_RETURN, pygame.K_SPACE):
            self._continue()

    def draw(self, surface: pygame.Surface) -> None:
        title = self.app.assets.font("cinzel", 40).render(f"Pass to Player {self.player}", True, S.TEXT)
        surface.blit(title, title.get_rect(center=(S.CANVAS_W // 2, S.CANVAS_H // 2 - 50)))
        sub = self.app.assets.font("inter", 16).render(
            "Make sure the other player can't see the screen.", True, S.TEXT_DIM
        )
        surface.blit(sub, sub.get_rect(center=(S.CANVAS_W // 2, S.CANVAS_H // 2 - 10)))
        self.button.update_hover(pygame.mouse.get_pos())
        self.button.draw(surface, self.app.assets)
