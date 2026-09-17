"""Screen-space overlays that aren't tied to a particular card: floating
damage/heal numbers, toasts (bot move previews), and the turn/key banner.

Each has its own tiny `update(dt_ms) -> bool done` / `draw(surface, assets)`
so a scene can just hold a list of them and prune finished ones.
"""

from __future__ import annotations

from dataclasses import dataclass

import pygame

from .. import settings as S


@dataclass
class FloatingText:
    x: float
    y: float
    text: str
    color: tuple
    life_ms: float = 700
    rise: float = 46.0
    age_ms: float = 0.0
    big: bool = False

    def update(self, dt_ms: float) -> bool:
        self.age_ms += dt_ms
        return self.age_ms >= self.life_ms

    def draw(self, surface: pygame.Surface, assets) -> None:
        t = max(0.0, min(1.0, self.age_ms / self.life_ms))
        y = self.y - self.rise * t
        alpha = int(255 * (1 - max(0.0, t - 0.6) / 0.4)) if t > 0.6 else 255
        size = 22 if self.big else 16
        font = assets.font("inter", size, bold=True)
        img = font.render(self.text, True, self.color)
        img.set_alpha(max(0, alpha))
        outline = font.render(self.text, True, S.BLACK)
        outline.set_alpha(max(0, alpha))
        rect = img.get_rect(center=(self.x, y))
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            surface.blit(outline, rect.move(dx, dy))
        surface.blit(img, rect)


@dataclass
class Toast:
    x: float
    y: float
    text: str
    life_ms: float = 1400
    age_ms: float = 0.0

    def update(self, dt_ms: float) -> bool:
        self.age_ms += dt_ms
        return self.age_ms >= self.life_ms

    def draw(self, surface: pygame.Surface, assets) -> None:
        t = self.age_ms / self.life_ms
        alpha = 255
        if t < 0.12:
            alpha = int(255 * (t / 0.12))
        elif t > 0.8:
            alpha = int(255 * (1 - (t - 0.8) / 0.2))
        font = assets.font("inter", 16, bold=True)
        img = font.render(self.text, True, S.TEXT)
        pad_x, pad_y = 14, 8
        box = pygame.Surface((img.get_width() + pad_x * 2, img.get_height() + pad_y * 2), pygame.SRCALPHA)
        pygame.draw.rect(box, (*S.PANEL_LIGHT, 235), box.get_rect(), border_radius=10)
        pygame.draw.rect(box, (*S.AEMBER, 200), box.get_rect(), width=2, border_radius=10)
        box.blit(img, (pad_x, pad_y))
        box.set_alpha(max(0, alpha))
        surface.blit(box, box.get_rect(center=(self.x, self.y)))


@dataclass
class Banner:
    text: str
    subtext: str = ""
    life_ms: float = 900
    age_ms: float = 0.0
    color: tuple = S.AEMBER

    def update(self, dt_ms: float) -> bool:
        self.age_ms += dt_ms
        return self.age_ms >= self.life_ms

    def draw(self, surface: pygame.Surface, assets, cx: float, cy: float) -> None:
        t = self.age_ms / self.life_ms
        slide = max(0.0, 1 - min(1.0, t / 0.25))
        alpha = 255
        if t > 0.75:
            alpha = int(255 * (1 - (t - 0.75) / 0.25))
        font = assets.font("cinzel", 26)
        img = font.render(self.text, True, S.TEXT)
        simg = assets.font("inter", 14).render(self.subtext, True, S.TEXT_DIM) if self.subtext else None
        content_w = max(img.get_width(), simg.get_width() if simg else 0)
        box = pygame.Rect(0, 0, content_w + 60, img.get_height() + (50 if simg else 26))
        box.center = (int(cx - 260 * slide), int(cy))
        panel = pygame.Surface(box.size, pygame.SRCALPHA)
        pygame.draw.rect(panel, (*S.PANEL, 235), panel.get_rect(), border_radius=8)
        pygame.draw.rect(panel, (*self.color, 220), panel.get_rect(), width=2, border_radius=8)
        panel.blit(img, img.get_rect(midtop=(box.width // 2, 10)))
        if simg:
            panel.blit(simg, simg.get_rect(midtop=(box.width // 2, img.get_height() + 16)))
        panel.set_alpha(max(0, alpha))
        surface.blit(panel, box)
