"""A scrollable panel of game-log sentences (from option_labels.describe_log_event),
colored by event category and split into turns by option_labels.log_event_category
/ GameScene._sync_log's synthetic turn-separator lines."""

from __future__ import annotations

from typing import List, Tuple

import pygame

from .. import settings as S
from .widgets import draw_panel

_CATEGORY_COLOR = {
    "aember": S.AEMBER,
    "key": S.KEY_GOLD,
    "purge": S.PURGE,
    "damage": S.DANGER,
    "heal": S.HEAL,
    "turn": S.TEXT_FAINT,
    "neutral": S.TEXT_DIM,
}


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

    def handle_event(self, event: pygame.event.Event, rect: pygame.Rect, mouse_pos) -> None:
        if event.type == pygame.MOUSEWHEEL:
            if rect.collidepoint(mouse_pos):
                self.scroll = max(0, self.scroll - event.y * 3)

    def draw(self, surface: pygame.Surface, assets, rect: pygame.Rect, sentences: List[tuple], mouse_pos=(-1, -1)):
        """`sentences` are (text, category) or (text, category, iid) tuples.
        Returns [(line rect, iid)] for lines that name a card, so the scene
        can zoom that card when a line is hovered."""
        draw_panel(surface, rect, alpha=190)
        title_font = assets.font("cinzel", 16)
        title = title_font.render("Log", True, S.TEXT)
        surface.blit(title, (rect.left + 10, rect.top + 8))

        body = pygame.Rect(rect.left + 10, rect.top + 34, rect.width - 20, rect.height - 54)
        font = assets.font("inter", 13)
        turn_font = assets.font("inter", 12, bold=True)
        line_h = font.get_height() + 3

        wrapped: List[tuple] = []
        for entry in sentences:
            s, category = entry[0], entry[1]
            iid = entry[2] if len(entry) > 2 else None
            if category == "turn":
                wrapped.append((s, category, None))
                continue
            for line in _wrap(s, font, body.width):
                wrapped.append((line, category, iid))

        max_scroll = max(0, len(wrapped) - 1)
        self.scroll = max(0, min(self.scroll, max_scroll))
        visible_n = max(1, body.height // line_h)
        end = len(wrapped) - self.scroll
        start = max(0, end - visible_n)
        shown = wrapped[start:end]

        targets = []
        clip = surface.get_clip()
        surface.set_clip(body)
        y = body.top
        for line, category, iid in shown:
            color = _CATEGORY_COLOR.get(category, S.TEXT_DIM)
            line_rect = pygame.Rect(body.left, y, body.width, line_h)
            if iid is not None:
                targets.append((line_rect, iid))
                if line_rect.collidepoint(mouse_pos):
                    pygame.draw.rect(surface, S.PANEL_LIGHT, line_rect, border_radius=3)
            if category == "turn":
                pygame.draw.line(surface, S.TEXT_FAINT, (body.left, y + line_h // 2), (body.left + 18, y + line_h // 2), 1)
                img = turn_font.render(line.strip("— "), True, color)
                surface.blit(img, (body.left + 24, y))
            else:
                img = font.render(line, True, color)
                surface.blit(img, (body.left, y))
            y += line_h
        surface.set_clip(clip)

        hint_font = assets.font("inter", S.MIN_FONT)
        hint_text = "scrolled up: wheel down for latest" if self.scroll > 0 else "hover a line to see its card"
        hint = hint_font.render(hint_text, True, S.TEXT_FAINT)
        surface.blit(hint, (body.left, rect.bottom - 18))
        return targets
