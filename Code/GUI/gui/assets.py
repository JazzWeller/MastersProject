"""Loads and caches everything drawn on screen: card art (rounded, scaled),
a procedurally-drawn card back, fonts, small procedural icons, and optional
sound effects.

Nothing here talks to the engine; it only knows about file paths and pixels.
"""

from __future__ import annotations

import math
import os
from collections import OrderedDict
from typing import Dict, Optional, Tuple

import pygame

from . import settings as S

# Raw card art loaded at full resolution (~200x280, ~0.2MB each as an alpha
# surface) is the expensive thing to cache -- with 370 Phase 3 cards across
# 7 houses browsable in the deck builder, caching every one forever would
# hold ~83MB of surfaces that are mostly never drawn again. The much smaller
# scaled/rounded `_card_sized` cache (bounded by the handful of on-screen
# sizes actually used) is left uncapped.
_RAW_ART_CACHE_LIMIT = 120


def _rounded_mask(size: Tuple[int, int], radius: int) -> pygame.Surface:
    w, h = size
    mask = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.draw.rect(mask, (255, 255, 255, 255), (0, 0, w, h), border_radius=radius)
    return mask


def _apply_rounded_corners(surface: pygame.Surface, radius: int) -> pygame.Surface:
    size = surface.get_size()
    out = surface.convert_alpha()
    mask = _rounded_mask(size, radius)
    out.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
    return out


class AssetCache:
    def __init__(self):
        pygame.font.init()
        self._raw_art: "OrderedDict[str, Optional[pygame.Surface]]" = OrderedDict()
        self._card_sized: Dict[Tuple[str, int, int], pygame.Surface] = {}
        self._card_back: Dict[Tuple[int, int], pygame.Surface] = {}
        self._fonts: Dict[Tuple[str, int, bool], pygame.font.Font] = {}
        self._icons: Dict[Tuple[str, int], pygame.Surface] = {}
        self._sounds: Dict[str, "pygame.mixer.Sound"] = {}
        self._missing_art_logged = set()
        self.muted = False
        self._load_sounds()

    # ------------------------------------------------------------ fonts ----

    def font(self, family: str = "inter", size: int = 18, bold: bool = False) -> pygame.font.Font:
        key = (family, size, bold)
        f = self._fonts.get(key)
        if f is not None:
            return f
        path = S.CINZEL_PATH if family == "cinzel" else S.INTER_PATH
        try:
            f = pygame.font.Font(path, size)
        except (FileNotFoundError, OSError):
            f = pygame.font.SysFont(None, size)
        if bold and family != "cinzel":
            f.set_bold(True)
        self._fonts[key] = f
        return f

    # -------------------------------------------------------- card faces ----

    def _raw(self, rel_path: str) -> Optional[pygame.Surface]:
        if rel_path in self._raw_art:
            self._raw_art.move_to_end(rel_path)
            return self._raw_art[rel_path]
        full = os.path.join(S.CARD_ART_DIR, rel_path)
        surf = None
        try:
            surf = pygame.image.load(full).convert_alpha()
        except (FileNotFoundError, pygame.error):
            if rel_path not in self._missing_art_logged:
                self._missing_art_logged.add(rel_path)
        self._raw_art[rel_path] = surf
        if len(self._raw_art) > _RAW_ART_CACHE_LIMIT:
            self._raw_art.popitem(last=False)
        return surf

    def card_face(self, rel_path: Optional[str], size: Tuple[int, int]) -> pygame.Surface:
        """The face of a card, scaled to `size` with rounded corners. Falls
        back to a placeholder if the art file is missing (never crashes)."""
        w, h = size
        key = (rel_path or "", w, h)
        cached = self._card_sized.get(key)
        if cached is not None:
            return cached

        raw = self._raw(rel_path) if rel_path else None
        radius = max(4, int(w * 0.07))
        if raw is not None:
            scaled = pygame.transform.smoothscale(raw, (w, h))
            face = _apply_rounded_corners(scaled, radius)
        else:
            face = self._placeholder_face(size, radius, rel_path or "?")
        # subtle border
        pygame.draw.rect(face, (0, 0, 0, 140), face.get_rect(), width=max(1, w // 60), border_radius=radius)
        self._card_sized[key] = face
        return face

    def _placeholder_face(self, size, radius, label: str) -> pygame.Surface:
        w, h = size
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(surf, (60, 52, 82, 255), (0, 0, w, h), border_radius=radius)
        pygame.draw.rect(surf, S.TEXT_FAINT, (0, 0, w, h), width=2, border_radius=radius)
        f = self.font("inter", max(10, w // 9))
        name = os.path.splitext(os.path.basename(label))[0].replace("-", " ")
        words = name.split()
        y = h // 2 - 10 * max(1, len(words) // 2)
        for word in words[:4]:
            txt = f.render(word, True, S.TEXT_DIM)
            surf.blit(txt, (w // 2 - txt.get_width() // 2, y))
            y += f.get_height()
        return surf

    def card_back(self, size: Tuple[int, int]) -> pygame.Surface:
        cached = self._card_back.get(size)
        if cached is not None:
            return cached
        w, h = size
        radius = max(4, int(w * 0.07))
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(surf, S.CARD_BACK_BASE, (0, 0, w, h), border_radius=radius)
        inset = max(4, int(w * 0.07))
        inner = pygame.Rect(inset, inset, w - 2 * inset, h - 2 * inset)
        pygame.draw.rect(surf, S.CARD_BACK_EDGE, inner, border_radius=max(2, radius - 4))
        pygame.draw.rect(surf, S.CARD_BACK_FILIGREE, inner, width=max(1, w // 60), border_radius=max(2, radius - 4))
        # central gem (diamond)
        cx, cy = w // 2, h // 2
        gem_r = int(min(w, h) * 0.16)
        pts = [(cx, cy - gem_r), (cx + gem_r, cy), (cx, cy + gem_r), (cx - gem_r, cy)]
        pygame.draw.polygon(surf, S.CARD_BACK_FILIGREE, pts)
        pygame.draw.polygon(surf, S.CARD_BACK_EDGE, pts, width=max(1, w // 90))
        # corner flourishes
        for ang in (45, 135, 225, 315):
            rad = math.radians(ang)
            fx = cx + math.cos(rad) * gem_r * 2.6
            fy = cy + math.sin(rad) * gem_r * 2.6
            pygame.draw.circle(surf, S.CARD_BACK_FILIGREE, (int(fx), int(fy)), max(2, w // 40))
        pygame.draw.rect(surf, (0, 0, 0, 140), (0, 0, w, h), width=max(1, w // 60), border_radius=radius)
        self._card_back[size] = surf
        return surf

    # ---------------------------------------------------------------- icons ----

    def house_emblem(self, house: str, size: int) -> pygame.Surface:
        key = (f"house:{house}", size)
        cached = self._icons.get(key)
        if cached is not None:
            return cached
        color = S.HOUSE_COLORS.get(house, S.TEXT_DIM)
        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        c = size // 2
        r = int(size * 0.46)
        pygame.draw.circle(surf, (*color, 60), (c, c), r)
        pygame.draw.circle(surf, color, (c, c), r, width=max(2, size // 14))
        if house == "Dis":
            pts = [(c, c - r * 0.6), (c + r * 0.6, c + r * 0.4), (c - r * 0.6, c + r * 0.4)]
            pygame.draw.polygon(surf, color, pts)
        elif house == "Logos":
            pygame.draw.circle(surf, color, (c, c), int(r * 0.35), width=max(2, size // 16))
            pygame.draw.ellipse(
                surf, color,
                (c - r * 0.75, c - r * 0.28, r * 1.5, r * 0.56), width=max(1, size // 20)
            )
        elif house == "Shadows":
            pygame.draw.circle(surf, color, (int(c + r * 0.22), c), int(r * 0.55))
            pygame.draw.circle(surf, (0, 0, 0, 0), (int(c + r * 0.5), c), int(r * 0.5))
        elif house == "Brobnar":
            w = max(2, size // 10)
            pygame.draw.line(surf, color, (c - r * 0.55, c + r * 0.55), (c + r * 0.55, c - r * 0.55), width=w)
            pygame.draw.line(surf, color, (c - r * 0.55, c - r * 0.55), (c + r * 0.55, c + r * 0.55), width=w)
        elif house == "Mars":
            pygame.draw.circle(surf, color, (c, c), max(2, int(r * 0.22)))
            for ang in (0, 90, 180, 270):
                rad = math.radians(ang)
                x0, y0 = c + math.cos(rad) * r * 0.45, c + math.sin(rad) * r * 0.45
                x1, y1 = c + math.cos(rad) * r * 0.85, c + math.sin(rad) * r * 0.85
                pygame.draw.line(surf, color, (x0, y0), (x1, y1), width=max(1, size // 18))
        elif house == "Sanctum":
            arm_w = max(2, int(r * 0.34))
            pygame.draw.rect(surf, color, (c - arm_w / 2, c - r * 0.62, arm_w, r * 1.24))
            pygame.draw.rect(surf, color, (c - r * 0.62, c - arm_w / 2, r * 1.24, arm_w))
        elif house == "Untamed":
            toe_r = max(2, int(r * 0.2))
            for ang in (200, 270, 340):
                rad = math.radians(ang)
                tx = c + math.cos(rad) * r * 0.45
                ty = c + math.sin(rad) * r * 0.45
                pygame.draw.circle(surf, color, (int(tx), int(ty)), toe_r)
            pygame.draw.circle(surf, color, (c, int(c + r * 0.4)), max(3, int(r * 0.32)))
        self._icons[key] = surf
        return surf

    def key_icon(self, size: int, forged: bool) -> pygame.Surface:
        """Forged: solid bright gold with a faint glow halo. Unforged: a
        dim, fully hollow outline -- deliberately drawn as different shapes
        (not just different colors of the same fill) so the two states stay
        legible even at HUD size."""
        key = (f"key:{forged}", size)
        cached = self._icons.get(key)
        if cached is not None:
            return cached
        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        c = size // 2
        r = int(size * 0.26)
        shaft_w = max(2, round(size * 0.16))
        head_cy = int(size * 0.30)
        shaft_bottom = int(size * 0.86)
        tooth_w = max(3, int(size * 0.3))
        tooth_h = max(2, shaft_w - 1)

        if forged:
            glow = pygame.Surface((size, size), pygame.SRCALPHA)
            pygame.draw.circle(glow, (*S.KEY_GOLD, 70), (c, head_cy), r + 3)
            surf.blit(glow, (0, 0))
            pygame.draw.circle(surf, S.KEY_GOLD, (c, head_cy), r)
            pygame.draw.circle(surf, S.AEMBER_GLOW, (c, head_cy), max(1, r - r // 2), width=1)
            pygame.draw.rect(surf, S.KEY_GOLD, (c - shaft_w // 2, head_cy, shaft_w, shaft_bottom - head_cy))
            pygame.draw.rect(surf, S.KEY_GOLD, (c, shaft_bottom - tooth_h, tooth_w, tooth_h))
            pygame.draw.rect(surf, S.KEY_GOLD, (c, shaft_bottom - tooth_h * 2 - 2, int(tooth_w * 0.6), tooth_h))
        else:
            color = S.KEY_EMPTY
            pygame.draw.circle(surf, color, (c, head_cy), r, width=max(1, size // 14))
            pygame.draw.line(surf, color, (c, head_cy + r - 1), (c, shaft_bottom), width=max(1, size // 14))
            pygame.draw.line(surf, color, (c, shaft_bottom), (c + tooth_w // 2, shaft_bottom), width=max(1, size // 14))

        self._icons[key] = surf
        return surf

    def chain_icon(self, size: int) -> pygame.Surface:
        key = ("chain", size)
        cached = self._icons.get(key)
        if cached is not None:
            return cached
        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        r = size // 4
        pygame.draw.circle(surf, S.TEXT_DIM, (int(size * 0.35), size // 2), r, width=max(2, size // 12))
        pygame.draw.circle(surf, S.TEXT_DIM, (int(size * 0.65), size // 2), r, width=max(2, size // 12))
        self._icons[key] = surf
        return surf

    def aember_gem(self, size: int) -> pygame.Surface:
        key = ("gem", size)
        cached = self._icons.get(key)
        if cached is not None:
            return cached
        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        c = size // 2
        pts = [(c, 0), (size, int(size * 0.38)), (int(size * 0.8), size), (int(size * 0.2), size), (0, int(size * 0.38))]
        pygame.draw.polygon(surf, S.AEMBER, pts)
        pygame.draw.polygon(surf, S.AEMBER_GLOW, pts, width=max(1, size // 14))
        self._icons[key] = surf
        return surf

    # --------------------------------------------------------------- sound ----

    def _load_sounds(self) -> None:
        if not os.path.isdir(S.SOUNDS_DIR):
            return
        try:
            if pygame.mixer.get_init() is None:
                pygame.mixer.init()
        except pygame.error:
            return
        for fname in os.listdir(S.SOUNDS_DIR):
            if fname.lower().endswith((".ogg", ".wav")):
                name = os.path.splitext(fname)[0]
                try:
                    self._sounds[name] = pygame.mixer.Sound(os.path.join(S.SOUNDS_DIR, fname))
                except pygame.error:
                    pass

    def play(self, name: str, volume: float = 1.0) -> None:
        if self.muted:
            return
        snd = self._sounds.get(name)
        if snd is not None:
            snd.set_volume(volume)
            snd.play()
