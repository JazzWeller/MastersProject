"""Turns board bands (settings.py) + card counts into concrete on-screen
rects and per-card slot transforms, for one viewer.

Pure geometry: no pygame display calls, no engine imports, so it's cheap
to unit test headlessly. `pygame.Rect` is used only as a plain (x, y, w, h)
container.

Invariant (enforced by tests/test_snapshot_layout.py): every card a slot
function returns lies entirely inside its own band, and no two bands
overlap.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import pygame

from . import settings as S

PILE_ORDER_TOP_DOWN = ("deck", "discard", "archive", "purged")


def rotated_half_extents(w: float, h: float, rot_deg: float) -> Tuple[float, float]:
    """Half width/height of the axis-aligned box around a w*h card rotated by rot_deg."""
    r = math.radians(abs(rot_deg))
    return (w * math.cos(r) + h * math.sin(r)) / 2, (w * math.sin(r) + h * math.cos(r)) / 2


class Layout:
    def __init__(self, viewer: int):
        self.viewer = viewer

    def is_bottom(self, pid: int) -> bool:
        return pid == self.viewer

    # --------------------------------------------------------------- bands ----

    def _band(self, pid: int, bottom_band, top_band) -> Tuple[int, int]:
        return bottom_band if self.is_bottom(pid) else top_band

    def _band_rect(self, band) -> pygame.Rect:
        y0, y1 = band
        return pygame.Rect(S.PLAY_X, y0, S.PLAY_W, y1 - y0)

    def hand_rect(self, pid: int) -> pygame.Rect:
        return self._band_rect(self._band(pid, S.BAND_YOUR_HAND, S.BAND_OPP_HAND))

    def hud_rect(self, pid: int) -> pygame.Rect:
        return self._band_rect(self._band(pid, S.BAND_YOUR_HUD, S.BAND_OPP_HUD))

    def board_rect(self, pid: int) -> pygame.Rect:
        return self._band_rect(self._band(pid, S.BAND_YOUR_BOARD, S.BAND_OPP_BOARD))

    def creature_row_rect(self, pid: int) -> pygame.Rect:
        b = self.board_rect(pid)
        return pygame.Rect(b.left, b.top, S.CREATURE_LANE_W, b.height)

    def artifact_row_rect(self, pid: int) -> pygame.Rect:
        b = self.board_rect(pid)
        return pygame.Rect(b.left + S.CREATURE_LANE_W + S.LANE_DIVIDER_W, b.top, S.ARTIFACT_LANE_W, b.height)

    def prompt_rect(self) -> pygame.Rect:
        return self._band_rect(S.BAND_PROMPT)

    def action_bar_rect(self) -> pygame.Rect:
        return self._band_rect(S.BAND_ACTION_BAR)

    def all_band_rects(self) -> List[Tuple[str, pygame.Rect]]:
        return [
            ("opp hand", self.hand_rect(3 - self.viewer)),
            ("opp hud", self.hud_rect(3 - self.viewer)),
            ("opp board", self.board_rect(3 - self.viewer)),
            ("prompt", self.prompt_rect()),
            ("your board", self.board_rect(self.viewer)),
            ("your hud", self.hud_rect(self.viewer)),
            ("your hand", self.hand_rect(self.viewer)),
            ("action bar", self.action_bar_rect()),
        ]

    def pile_rect(self, pid: int, kind: str) -> pygame.Rect:
        bottom = self.is_bottom(pid)
        y0, y1 = S.BAND_PILES_YOU if bottom else S.BAND_PILES_OPP
        order = list(PILE_ORDER_TOP_DOWN)
        if bottom:
            order.reverse()  # deck ends up nearest your hand
        slot = order.index(kind)
        y = y0 + S.PILE_GAP + slot * (S.PILE_SLOT_H + S.PILE_GAP)  # label sits in the gap above
        x = S.LEFT_COL_X + (S.LEFT_COL_W - S.PILE_ICON_W) // 2
        return pygame.Rect(x, y, S.PILE_ICON_W, S.PILE_ICON_H)

    def zoom_rect(self) -> pygame.Rect:
        y0, y1 = S.SIDE_ZOOM
        return pygame.Rect(S.SIDE_X + 10, y0, S.SIDE_W - 20, y1 - y0)

    def log_rect(self) -> pygame.Rect:
        y0, y1 = S.SIDE_LOG
        return pygame.Rect(S.SIDE_X + 10, y0, S.SIDE_W - 20, y1 - y0)

    # -------------------------------------------------------------- slots ----

    def hand_card_size(self, pid: int) -> Tuple[int, int]:
        if self.is_bottom(pid):
            return S.HAND_CARD_W, S.HAND_CARD_H
        return S.OPP_HAND_CARD_W, S.OPP_HAND_CARD_H

    def hand_slots(self, pid: int, n: int) -> List[Tuple[float, float, float]]:
        """(x, y, rotation) for each of `n` cards in `pid`'s hand."""
        rect = self.hand_rect(pid)
        w, h = self.hand_card_size(pid)
        if not self.is_bottom(pid):
            return [(x, y, 0.0) for x, y in self.row_slots(n, rect, w, h, gap=6)]
        # The fan's baseline sits so the highest (outermost, lifted, tilted)
        # card touches the band's top padding and the lowest stays inside.
        _, far_half = rotated_half_extents(w, h, S.HAND_FAN_MAX_ROT)
        base_y = rect.top + 4 + far_half + S.HAND_FAN_LIFT
        return self.fan_slots(n, rect, w, h, base_y=base_y)

    def fan_slots(
        self, n: int, rect: pygame.Rect, card_w: int, card_h: int, arc_up: bool = True, base_y: float = None
    ) -> List[Tuple[float, float, float]]:
        """Center (x, y, rotation_degrees) for `n` cards fanned in `rect`.
        Outer cards lift toward the board (up for `arc_up`) and tilt slightly."""
        if n <= 0:
            return []
        max_spacing = card_w * 0.72
        usable = min(rect.width, S.HAND_FAN_MAX_SPREAD)
        if n > 1:
            spacing = min(max_spacing, (usable - card_w) / (n - 1))
        else:
            spacing = 0.0
        total_w = spacing * (n - 1) + card_w
        start_x = rect.centerx - total_w / 2 + card_w / 2
        if base_y is None:
            base_y = rect.centery
        center_i = (n - 1) / 2
        sign = -1.0 if arc_up else 1.0
        slots = []
        for i in range(n):
            x = start_x + i * spacing
            d = i - center_i
            lift = 0.0 if center_i == 0 else min(S.HAND_FAN_LIFT, abs(d) * (S.HAND_FAN_LIFT / max(center_i, 1)))
            y = base_y + sign * lift
            rot_mag = max(-S.HAND_FAN_MAX_ROT, min(S.HAND_FAN_MAX_ROT, -d * S.HAND_FAN_ROT_SLOPE))
            rot = rot_mag if arc_up else -rot_mag
            slots.append((x, y, rot))
        return slots

    def row_slots(self, n: int, rect: pygame.Rect, card_w: int, card_h: int, gap: int = 12) -> List[Tuple[float, float]]:
        """Center (x, y) for `n` cards in a straight row inside `rect` (with
        padding). Cards spread with `gap` between them and compress
        horizontally -- never vertically, never outside the rect -- when there
        are too many to fit."""
        if n <= 0:
            return []
        inner = rect.inflate(-2 * S.LANE_PAD, 0)
        if n > 1:
            spacing = min(card_w + gap, (inner.width - card_w) / (n - 1))
        else:
            spacing = 0.0
        total_w = spacing * (n - 1) + card_w
        start_x = inner.centerx - total_w / 2 + card_w / 2
        y = rect.centery
        return [(start_x + i * spacing, y) for i in range(n)]

    def flank_ghost_slots(self, pid: int, n_creatures: int) -> dict:
        """Where to show the "put it here" slots for a new creature: just past
        each end of the current row, so they never cover a creature already
        in play. Clamped inside the lane; if the row is full the ghost sits
        over the end card, which is where the new card will squeeze in."""
        rect = self.creature_row_rect(pid)
        if n_creatures == 0:
            x, y = self.row_slots(1, rect, S.BOARD_CARD_W, S.BOARD_CARD_H)[0]
            return {"left": (x, y), "right": (x, y)}
        slots = self.row_slots(n_creatures, rect, S.BOARD_CARD_W, S.BOARD_CARD_H)
        step = S.BOARD_CARD_W + 12
        lo = rect.left + S.LANE_PAD + S.BOARD_CARD_W / 2
        hi = rect.right - S.LANE_PAD - S.BOARD_CARD_W / 2
        y = slots[0][1]
        return {"left": (max(lo, slots[0][0] - step), y), "right": (min(hi, slots[-1][0] + step), y)}

    def upgrade_slot(self, host_x: float, host_y: float, index: int) -> Tuple[float, float, int, int]:
        """An upgrade sits as a small tab inside its host's top-right corner."""
        x = host_x + S.BOARD_CARD_W / 2 - S.UPGRADE_TAB_W / 2 - 4 - index * (S.UPGRADE_TAB_W * 0.45)
        y = host_y - S.BOARD_CARD_H / 2 + S.UPGRADE_TAB_H / 2 + 4
        return x, y, S.UPGRADE_TAB_W, S.UPGRADE_TAB_H
