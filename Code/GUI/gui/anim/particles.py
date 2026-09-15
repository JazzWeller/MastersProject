"""A small, dependency-free particle system: sparks, embers, and a swirl,
plus a couple of one-shot "burst" helpers used by the Director.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple

import pygame

from .. import settings as S


@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    life_ms: float
    age_ms: float = 0.0
    size: float = 3.0
    color: Tuple[int, int, int] = S.AEMBER
    gravity: float = 0.0
    fade: bool = True
    shrink: bool = True

    def update(self, dt_ms: float) -> bool:
        self.age_ms += dt_ms
        dt_s = dt_ms / 1000.0
        self.vy += self.gravity * dt_s
        self.x += self.vx * dt_s
        self.y += self.vy * dt_s
        return self.age_ms >= self.life_ms

    def draw(self, surface: pygame.Surface) -> None:
        t = max(0.0, min(1.0, self.age_ms / self.life_ms))
        alpha = int(255 * (1 - t)) if self.fade else 255
        size = max(0.5, self.size * (1 - t * 0.7)) if self.shrink else self.size
        if alpha <= 2 or size < 0.5:
            return
        surf = pygame.Surface((int(size * 2) + 2, int(size * 2) + 2), pygame.SRCALPHA)
        pygame.draw.circle(surf, (*self.color, alpha), surf.get_rect().center, size)
        surface.blit(surf, surf.get_rect(center=(self.x, self.y)))


class ParticleSystem:
    """A single global pool. `emit_*` helpers queue particles; `update`/`draw`
    run every frame regardless of whether the Animator is mid-beat, so a
    burst can keep drifting after its triggering beat finishes."""

    def __init__(self):
        self.particles: List[Particle] = []

    def update(self, dt_ms: float) -> None:
        self.particles = [p for p in self.particles if not p.update(dt_ms)]

    def draw(self, surface: pygame.Surface) -> None:
        for p in self.particles:
            p.draw(surface)

    def clear(self) -> None:
        self.particles.clear()

    # ------------------------------------------------------------- emitters ----

    def emit_sparks(self, x, y, color=S.AEMBER, n=14, speed=140, life=550):
        for _ in range(n):
            ang = random.uniform(0, math.tau)
            spd = random.uniform(speed * 0.3, speed)
            self.particles.append(
                Particle(
                    x, y, math.cos(ang) * spd, math.sin(ang) * spd - 40,
                    life_ms=random.uniform(life * 0.6, life), size=random.uniform(2, 4),
                    color=color, gravity=260,
                )
            )

    def emit_embers(self, x, y, n=18, life=650):
        for _ in range(n):
            ang = random.uniform(0, math.tau)
            spd = random.uniform(40, 160)
            self.particles.append(
                Particle(
                    x, y, math.cos(ang) * spd, math.sin(ang) * spd,
                    life_ms=random.uniform(life * 0.5, life), size=random.uniform(2, 5),
                    color=random.choice([S.DANGER, (255, 140, 60), (120, 30, 20)]),
                    gravity=40,
                )
            )

    def emit_swirl(self, x, y, n=16, life=600, color=S.PURGE, radius=40):
        for i in range(n):
            ang = (i / n) * math.tau
            r = radius * random.uniform(0.6, 1.0)
            px, py = x + math.cos(ang) * r, y + math.sin(ang) * r
            vx, vy = (x - px) * 1.6, (y - py) * 1.6
            self.particles.append(
                Particle(px, py, vx, vy, life_ms=life, size=random.uniform(2, 4), color=color, fade=True)
            )

    def emit_confetti(self, x, y, n=40, spread=500):
        colors = [S.AEMBER, S.KEY_GOLD, S.HOUSE_COLORS["Dis"], S.HOUSE_COLORS["Logos"], S.HOUSE_COLORS["Shadows"]]
        for _ in range(n):
            vx = random.uniform(-spread, spread) * 0.4
            vy = random.uniform(-spread, -spread * 0.3)
            self.particles.append(
                Particle(x, y, vx, vy, life_ms=random.uniform(900, 1600), size=random.uniform(2, 4),
                          color=random.choice(colors), gravity=320, shrink=False)
            )

    def emit_burst_ring(self, x, y, color=S.KEY_GOLD, n=24, speed=220, life=700):
        for i in range(n):
            ang = (i / n) * math.tau
            self.particles.append(
                Particle(x, y, math.cos(ang) * speed, math.sin(ang) * speed, life_ms=life,
                          size=random.uniform(2, 5), color=color, gravity=0)
            )
