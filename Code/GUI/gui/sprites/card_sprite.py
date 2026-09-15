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
        self.rot = 0.0       # degrees, clockwise
        self.scale = 1.0
        self.alpha = 255.0

        self.w = S.BOARD_CARD_W
        self.h = S.BOARD_CARD_H

        self.face_up = True
        self.image_path: Optional[str] = None
        self.dim = False
        self.glow: Optional[str] = None   # None | "legal" | "selected" | "target"
        self.selected = False
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
        self.dim = cs.exhausted and cs.zone in ("play_creature", "play_artifact")

    def rect(self) -> pygame.Rect:
        w, h = self.w * self.scale, self.h * self.scale
        return pygame.Rect(int(self.x - w / 2), int(self.y - h / 2), int(w), int(h))

    def contains_point(self, px: float, py: float) -> bool:
        if not self.visible:
            return False
        rad = math.radians(-self.rot)
        dx, dy = px - self.x, py - self.y
        lx = dx * math.cos(rad) - dy * math.sin(rad)
        ly = dx * math.sin(rad) + dy * math.cos(rad)
        return abs(lx) <= (self.w * self.scale) / 2 and abs(ly) <= (self.h * self.scale) / 2

    # ------------------------------------------------------------- drawing ----

    def _face_surface(self) -> pygame.Surface:
        size = (self.w, self.h)
        if self.face_up:
            return self.assets.card_face(self.image_path, size)
        return self.assets.card_back(size)

    def _glow_color(self):
        return {
            "legal": S.GLOW_LEGAL,
            "selected": S.GLOW_SELECTED,
            "target": S.GLOW_TARGET,
        }.get(self.glow)

    def draw(self, surface: pygame.Surface) -> None:
        if not self.visible or self.alpha <= 1:
            return
        img = self._face_surface()
        if self.dim:
            dark = pygame.Surface(img.get_size(), pygame.SRCALPHA)
            dark.fill((0, 0, 0, 100))
            img = img.copy()
            img.blit(dark, (0, 0))

        color = self._glow_color()
        if color is not None:
            pad = 10
            glow = pygame.Surface((img.get_width() + pad * 2, img.get_height() + pad * 2), pygame.SRCALPHA)
            radius = max(4, int(self.w * 0.09))
            for i, a in ((0, 70), (1, 110), (2, 170)):
                rect = glow.get_rect().inflate(-i * 6, -i * 6)
                pygame.draw.rect(glow, (*color, a), rect, width=3, border_radius=radius)
            combined = pygame.Surface(glow.get_size(), pygame.SRCALPHA)
            combined.blit(glow, (0, 0))
            combined.blit(img, (pad, pad))
            img = combined

        transformed = pygame.transform.rotozoom(img, self.rot, self.scale) if (self.rot or self.scale != 1.0) else img
        if self.alpha < 255:
            transformed = transformed.copy()
            transformed.set_alpha(int(self.alpha))
        rect = transformed.get_rect(center=(self.x, self.y))
        surface.blit(transformed, rect)

        self._draw_tokens(surface)

    def _draw_tokens(self, surface: pygame.Surface) -> None:
        cs = self.card_state
        if cs is None or not self.face_up or self.scale < 0.3:
            return
        base_rect = self.rect()
        font = self.assets.font("inter", max(11, int(self.h * 0.14)), bold=True)

        if cs.type == "Creature":
            if cs.damage > 0:
                self._badge(surface, base_rect.bottomleft, S.DANGER, f"-{cs.damage}", font, anchor="bottomleft")
            if cs.aember_captured > 0:
                self._badge(surface, base_rect.bottomright, S.AEMBER, str(cs.aember_captured), font, anchor="bottomright")
            chips = []
            if cs.elusive:
                chips.append(("E", S.PURGE))
            if cs.skirmish:
                chips.append(("S", S.HEAL))
            for i, (letter, color) in enumerate(chips):
                cx = base_rect.left + 10 + i * 18
                cy = base_rect.top + 10
                pygame.draw.circle(surface, color, (cx, cy), 8)
                pygame.draw.circle(surface, S.BLACK, (cx, cy), 8, width=1)
                t = self.assets.font("inter", 11, bold=True).render(letter, True, S.WHITE)
                surface.blit(t, t.get_rect(center=(cx, cy)))

    def _badge(self, surface, pos, color, text, font, anchor="center"):
        t = font.render(text, True, S.WHITE)
        r = t.get_rect()
        setattr(r, anchor, pos)
        pad = 4
        bubble = pygame.Rect(r.left - pad, r.top - pad, r.width + pad * 2, r.height + pad * 2)
        pygame.draw.ellipse(surface, color, bubble)
        pygame.draw.ellipse(surface, S.BLACK, bubble, width=1)
        surface.blit(t, r)
