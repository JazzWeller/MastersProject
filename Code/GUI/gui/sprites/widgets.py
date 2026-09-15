"""Small reusable UI widgets: a clickable Button and a couple of panel
drawing helpers. Deliberately minimal — this project needs buttons, not a
widget toolkit.
"""

from __future__ import annotations

from typing import Optional

import pygame

from .. import settings as S


def draw_panel(surface: pygame.Surface, rect: pygame.Rect, alpha: int = 220, border=None, radius=10) -> None:
    panel = pygame.Surface(rect.size, pygame.SRCALPHA)
    pygame.draw.rect(panel, (*S.PANEL, alpha), panel.get_rect(), border_radius=radius)
    if border:
        pygame.draw.rect(panel, (*border, 255), panel.get_rect(), width=2, border_radius=radius)
    surface.blit(panel, rect)


class Button:
    def __init__(
        self,
        rect: pygame.Rect,
        text: str,
        enabled: bool = True,
        primary: bool = False,
        danger: bool = False,
    ):
        self.rect = rect
        self.text = text
        self.enabled = enabled
        self.primary = primary
        self.danger = danger
        self.hovered = False

    def update_hover(self, mouse_pos) -> None:
        self.hovered = self.enabled and self.rect.collidepoint(mouse_pos)

    def clicked(self, event: pygame.event.Event) -> bool:
        if not self.enabled:
            return False
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            return self.rect.collidepoint(event.pos)
        return False

    def draw(self, surface: pygame.Surface, assets) -> None:
        if not self.enabled:
            bg, fg, border = S.PANEL, S.TEXT_FAINT, S.TEXT_FAINT
        elif self.danger:
            bg = S.DANGER if self.hovered else S.PANEL_LIGHT
            fg = S.WHITE
            border = S.DANGER
        elif self.primary:
            bg = S.AEMBER_GLOW if self.hovered else S.AEMBER
            fg = S.BLACK
            border = S.AEMBER_GLOW
        else:
            bg = S.PANEL_LIGHT if self.hovered else S.PANEL
            fg = S.TEXT
            border = S.TEXT_FAINT

        pygame.draw.rect(surface, bg, self.rect, border_radius=8)
        pygame.draw.rect(surface, border, self.rect, width=2, border_radius=8)
        font = assets.font("inter", 16, bold=self.primary)
        img = font.render(self.text, True, fg)
        surface.blit(img, img.get_rect(center=self.rect.center))


class Chip:
    """A small pill of text (status effect, house count, etc.)."""

    def __init__(self, text: str, color=None):
        self.text = text
        self.color = color or S.TEXT_DIM

    def size(self, assets, font_size=13) -> pygame.Rect:
        font = assets.font("inter", font_size, bold=True)
        w, h = font.size(self.text)
        return pygame.Rect(0, 0, w + 16, h + 10)

    def draw(self, surface: pygame.Surface, assets, x, y, font_size=13) -> pygame.Rect:
        font = assets.font("inter", font_size, bold=True)
        img = font.render(self.text, True, S.TEXT)
        w, h = img.get_width() + 16, img.get_height() + 10
        rect = pygame.Rect(x, y, w, h)
        pygame.draw.rect(surface, (*self.color, 60), rect, border_radius=h // 2)
        pygame.draw.rect(surface, self.color, rect, width=1, border_radius=h // 2)
        surface.blit(img, img.get_rect(center=rect.center))
        return rect
