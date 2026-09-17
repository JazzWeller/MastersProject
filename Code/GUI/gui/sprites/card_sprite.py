"""CardSprite: the on-screen representation of one physical card
(`instance_id`-keyed, reused across frames so tweens can move it smoothly).

Its `x`, `y`, `rot`, `scale`, `alpha` attributes are exactly what
`gui/anim/tween.py` animates (`Tween(sprite, "x", 400, 300)` etc.), so this
class deliberately has no animation logic of its own — it just draws
whatever state it's currently in.
"""

from __future__ import annotations

import math
from typing import Optional

import pygame

from .. import settings as S
from ..assets import AssetCache
from ..snapshot import CardState


class CardSprite:
    def __init__(self, iid: int, assets: AssetCache):
        self.iid = iid
        self.assets = assets

        self.x = 0.0
        self.y = 0.0
        self.rot = 0.0       # degrees, counter-clockwise (pygame's convention)
        self.scale = 1.0
        self.alpha = 255.0

        self.w = S.BOARD_CARD_W
        self.h = S.BOARD_CARD_H

        self.face_up = True
        self.image_path: Optional[str] = None
        self.exhausted = False
        self.glow: Optional[str] = None   # None | "legal" | "selected" | "target" | "hint"
        self.unusable_reason: Optional[str] = None  # set while it's your turn and this card can't be used
        self.hint: Optional[str] = None             # informational tooltip (e.g. "can't play, can discard")
        self.pick_number: Optional[int] = None      # multi-select / ordering badge
        self.visible = True

        self.card_state: Optional[CardState] = None

    def set_size(self, w: int, h: int) -> None:
        self.w, self.h = w, h

    def snap_to(self, x: float, y: float, rot: float = 0.0) -> None:
        self.x, self.y, self.rot = x, y, rot

    def apply_state(self, cs: CardState) -> None:
        self.card_state = cs
        self.face_up = cs.face_up
        self.image_path = cs.image if cs.face_up else None
        self.exhausted = cs.exhausted and cs.zone in ("play_creature", "play_artifact")

    def rect(self) -> pygame.Rect:
        w, h = self.w * self.scale, self.h * self.scale
        return pygame.Rect(int(self.x - w / 2), int(self.y - h / 2), int(w), int(h))

    def contains_point(self, px: float, py: float) -> bool:
        """Point-in-rotated-rectangle, matching how `draw` renders the card.

        `pygame.transform.rotozoom(img, rot, ...)` turns the image
        counter-clockwise on screen by `rot` degrees. With y pointing down,
        mapping a screen offset back into the card's own frame therefore
        rotates by +rot. (It used -rot, which tilted every hitbox the wrong
        way -- Code/PLAYTEST_FIX_PLAN.md M1.)"""
        if not self.visible:
            return False
        rad = math.radians(self.rot)
        dx, dy = px - self.x, py - self.y
        lx = dx * math.cos(rad) - dy * math.sin(rad)
        ly = dx * math.sin(rad) + dy * math.cos(rad)
        return abs(lx) <= (self.w * self.scale) / 2 and abs(ly) <= (self.h * self.scale) / 2

    # ------------------------------------------------------------- drawing ----

    def _face_surface(self) -> pygame.Surface:
        size = (int(self.w), int(self.h))
        if self.face_up:
            return self.assets.card_face(self.image_path, size)
        return self.assets.card_back(size)

    def _glow_color(self):
        return {
            "legal": S.GLOW_LEGAL,
            "selected": S.GLOW_SELECTED,
            "target": S.GLOW_TARGET,
            "hint": S.TEXT_DIM,
        }.get(self.glow)

    def draw(self, surface: pygame.Surface) -> None:
        if not self.visible or self.alpha <= 1:
            return
        img = self._face_surface()
        shade = 0
        if self.exhausted:
            shade = 110
        if self.unusable_reason:
            shade = max(shade, 120)
        if shade:
            dark = pygame.Surface(img.get_size(), pygame.SRCALPHA)
            pygame.draw.rect(dark, (0, 0, 0, shade), dark.get_rect(), border_radius=max(4, int(self.w * 0.07)))
            img = img.copy()
            img.blit(dark, (0, 0))

        color = self._glow_color()
        if color is not None:
            pad = 8
            glow = pygame.Surface((img.get_width() + pad * 2, img.get_height() + pad * 2), pygame.SRCALPHA)
            radius = max(4, int(self.w * 0.09))
            for i, a in ((0, 70), (1, 120), (2, 190)):
                r = glow.get_rect().inflate(-i * 5, -i * 5)
                pygame.draw.rect(glow, (*color, a), r, width=3, border_radius=radius)
            glow.blit(img, (pad, pad))
            img = glow

        transformed = pygame.transform.rotozoom(img, self.rot, self.scale) if (self.rot or self.scale != 1.0) else img
        if self.alpha < 255:
            transformed = transformed.copy()
            transformed.set_alpha(int(self.alpha))
        surface.blit(transformed, transformed.get_rect(center=(self.x, self.y)))

        self._draw_tokens(surface)

    def _draw_tokens(self, surface: pygame.Surface) -> None:
        cs = self.card_state
        base = self.rect()
        if self.pick_number is not None:
            self._badge(surface, (base.centerx, base.top + 16), S.WHITE, str(self.pick_number),
                        self.assets.font("inter", 16, bold=True), fg=S.BLACK)
        if cs is None or not self.face_up or self.scale < 0.3 or self.w < 90:
            return
        font = self.assets.font("inter", 13, bold=True)

        if cs.zone == "play_creature":
            remaining = cs.power - cs.damage
            # Power (remaining after damage) bottom-left, damage beside it.
            color = S.DANGER if cs.damage else (40, 36, 52)
            self._badge(surface, (base.left + 16, base.bottom - 16), color, str(remaining), font)
            if cs.damage:
                self._badge(surface, (base.left + 44, base.bottom - 16), (60, 20, 24), f"-{cs.damage}", self.assets.font("inter", 12, bold=True))
            if cs.armor:
                self._badge(surface, (base.left + 16, base.bottom - 42), (60, 70, 90), f"A{cs.armor}", self.assets.font("inter", 12, bold=True))
            if cs.aember_captured > 0:
                self._badge(surface, (base.right - 16, base.bottom - 16), S.AEMBER, str(cs.aember_captured), font, fg=S.BLACK)
            chips = []
            if cs.elusive:
                chips.append(("Elusive", S.PURGE))
            if cs.skirmish:
                chips.append(("Skirmish", S.HEAL))
            y = base.top + 12
            for text, chip_color in chips:
                self._pill(surface, (base.left + 6, y), chip_color, text)
                y += 18
        if self.exhausted:
            self._pill(surface, (base.centerx, base.centery), (30, 26, 40), "Exhausted", center=True)

    def _badge(self, surface, center, color, text, font, fg=S.WHITE):
        t = font.render(text, True, fg)
        r = t.get_rect(center=center)
        bubble = r.inflate(12, 6)
        bubble.width = max(bubble.width, bubble.height)
        bubble.center = center
        pygame.draw.rect(surface, color, bubble, border_radius=bubble.height // 2)
        pygame.draw.rect(surface, S.BLACK, bubble, width=1, border_radius=bubble.height // 2)
        surface.blit(t, t.get_rect(center=bubble.center))

    def _pill(self, surface, pos, color, text, center=False):
        font = self.assets.font("inter", 12, bold=True)
        t = font.render(text, True, S.WHITE)
        r = pygame.Rect(0, 0, t.get_width() + 10, t.get_height() + 2)
        if center:
            r.center = pos
        else:
            r.topleft = pos
        pill = pygame.Surface(r.size, pygame.SRCALPHA)
        pygame.draw.rect(pill, (*color, 220), pill.get_rect(), border_radius=r.height // 2)
        surface.blit(pill, r)
        surface.blit(t, t.get_rect(center=r.center))
