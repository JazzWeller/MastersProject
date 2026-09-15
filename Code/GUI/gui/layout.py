"""Turns board bands (settings.py) + card counts into concrete on-screen
rects and per-card slot transforms, for one viewer.

Pure geometry: no pygame display calls, no engine imports, so it's cheap
to unit test headlessly. `pygame.Rect` is used only as a plain (x, y, w, h)
container.
"""

from __future__ import annotations

from typing import List, Tuple

import pygame

from . import settings as S

PILE_ORDER_TOP_DOWN = ("deck", "discard", "archive", "purged")


class Layout:
    def __init__(self, viewer: int):
        self.viewer = viewer

    def is_bottom(self, pid: int) -> bool:
        return pid == self.viewer

    # --------------------------------------------------------------- bands ----

    def _band(self, pid: int, bottom_band, top_band) -> Tuple[int, int]:
        return bottom_band if self.is_bottom(pid) else top_band

    def hand_rect(self, pid: int) -> pygame.Rect:
        y0, y1 = self._band(pid, S.BAND_YOUR_HAND, S.BAND_OPP_HAND)
        return pygame.Rect(S.PLAY_X, y0, S.PLAY_W, y1 - y0)

    def hud_rect(self, pid: int) -> pygame.Rect:
        y0, y1 = self._band(pid, S.BAND_YOUR_HUD, S.BAND_OPP_HUD)
        return pygame.Rect(S.PLAY_X, y0, S.PLAY_W, y1 - y0)

    def creature_row_rect(self, pid: int) -> pygame.Rect:
        y0, y1 = self._band(pid, S.BAND_YOUR_CREATURES, S.BAND_OPP_CREATURES)
        return pygame.Rect(S.PLAY_X, y0, S.PLAY_W, y1 - y0)

    def artifact_row_rect(self, pid: int) -> pygame.Rect:
        y0, y1 = self._band(pid, S.BAND_YOUR_ARTIFACTS, S.BAND_OPP_ARTIFACTS)
        return pygame.Rect(S.PLAY_X, y0, S.PLAY_W, y1 - y0)

    def prompt_rect(self) -> pygame.Rect:
        y0, y1 = S.BAND_PROMPT
        return pygame.Rect(S.PLAY_X, y0, S.PLAY_W, y1 - y0)

    def action_bar_rect(self) -> pygame.Rect:
        y0, y1 = S.BAND_ACTION_BAR
        return pygame.Rect(S.PLAY_X, y0, S.PLAY_W, y1 - y0)

    def pile_rect(self, pid: int, kind: str) -> pygame.Rect:
        bottom = self.is_bottom(pid)
        y0, y1 = S.BAND_PILES_YOU if bottom else S.BAND_PILES_OPP
        order = list(PILE_ORDER_TOP_DOWN)
        if bottom:
            order.reverse()  # deck ends up nearest your hand
        slot = order.index(kind)
        y = y0 + slot * (S.PILE_SLOT_H + S.PILE_GAP)
        x = S.LEFT_COL_X + (S.LEFT_COL_W - S.PILE_ICON_W) // 2
        return pygame.Rect(x, y, S.PILE_ICON_W, min(S.PILE_SLOT_H, y1 - y))

    def zoom_rect(self) -> pygame.Rect:
        y0, y1 = S.SIDE_ZOOM
        return pygame.Rect(S.SIDE_X + 10, y0, S.SIDE_W - 20, y1 - y0)

    def log_rect(self) -> pygame.Rect:
        y0, y1 = S.SIDE_LOG
        return pygame.Rect(S.SIDE_X + 10, y0, S.SIDE_W - 20, y1 - y0)

    # -------------------------------------------------------------- slots ----

    def fan_slots(
        self, n: int, rect: pygame.Rect, card_w: int, card_h: int
    ) -> List[Tuple[float, float, float]]:
        """Center (x, y, rotation_degrees) for `n` cards fanned in `rect`."""
        if n <= 0:
            return []
        max_spacing = card_w * 0.62
        if n > 1:
            spacing = min(max_spacing, (min(rect.width, S.HAND_FAN_MAX_SPREAD) - card_w) / (n - 1))
            spacing = max(spacing, card_w * 0.14)
        else:
            spacing = 0.0
        total_w = spacing * (n - 1) + card_w
        start_x = rect.centerx - total_w / 2 + card_w / 2
        base_y = rect.centery
        center_i = (n - 1) / 2
        slots = []
        for i in range(n):
            x = start_x + i * spacing
            d = i - center_i
            lift = 0.0 if center_i == 0 else min(S.HAND_FAN_LIFT, abs(d) * (S.HAND_FAN_LIFT / max(center_i, 1)))
            y = base_y + lift
            rot = max(-30.0, min(30.0, -d * 4.5))
            slots.append((x, y, rot))
        return slots

    def row_slots(self, n: int, rect: pygame.Rect, card_w: int, card_h: int) -> List[Tuple[float, float]]:
        """Center (x, y) for `n` cards evenly spaced (with overlap if crowded)
        in a non-fanned row."""
        if n <= 0:
            return []
        gap = 14
        if n > 1:
            spacing = min(card_w + gap, (rect.width - card_w) / (n - 1))
        else:
            spacing = 0.0
        total_w = spacing * (n - 1) + card_w
        start_x = rect.centerx - total_w / 2 + card_w / 2
        y = rect.centery
        return [(start_x + i * spacing, y) for i in range(n)]
