"""A scrollable panel of game-log sentences (from option_labels.describe_log_event)."""

from __future__ import annotations

from typing import List

import pygame

from .. import settings as S
from .widgets import draw_panel


def _wrap(text: str, font, max_w: int) -> List[str]:
    words = text.split(" ")
    lines, cur = [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if font.size(trial)[0] <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines or [""]


class LogPanel:
    def __init__(self):
        self.scroll = 0  # lines scrolled up from the bottom

    def handle_event(self, event: pygame.event.Event, rect: pygame.Rect) -> None:
        if event.type == pygame.MOUSEWHEEL:
            if rect.collidepoint(pygame.mouse.get_pos()):
                self.scroll = max(0, self.scroll - event.y * 3)

    def draw(self, surface: pygame.Surface, assets, rect: pygame.Rect, sentences: List[str]) -> None:
        draw_panel(surface, rect, alpha=190)
        title_font = assets.font("cinzel", 16)
        title = title_font.render("Log", True, S.TEXT)
        surface.blit(title, (rect.left + 10, rect.top + 8))

        body = pygame.Rect(rect.left + 10, rect.top + 34, rect.width - 20, rect.height - 44)
        font = assets.font("inter", 13)
        line_h = font.get_height() + 3

        wrapped: List[str] = []
        for s in sentences:
            wrapped.extend(_wrap(s, font, body.width))

        max_scroll = max(0, len(wrapped) - 1)
        self.scroll = max(0, min(self.scroll, max_scroll))
        visible_n = max(1, body.height // line_h)
        end = len(wrapped) - self.scroll
        start = max(0, end - visible_n)
        shown = wrapped[start:end]

        clip = surface.get_clip()
        surface.set_clip(body)
        y = body.top
        for line in shown:
            img = font.render(line, True, S.TEXT_DIM)
            surface.blit(img, (body.left, y))
            y += line_h
        surface.set_clip(clip)

        if self.scroll > 0:
            hint = assets.font("inter", 11).render("↓ scroll for latest", True, S.TEXT_FAINT)
            surface.blit(hint, (body.left, rect.bottom - 16))
