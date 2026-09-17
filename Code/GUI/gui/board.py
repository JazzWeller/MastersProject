"""Board: the sprite pool + per-player HUD state + particle/overlay lists
for one viewer's rendering of a game in progress. Owns *where things are*;
`gui/anim/director.py` owns *how they get there*.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pygame

from . import settings as S
from .anim.particles import ParticleSystem
from .anim.tween import Delay, Parallel, Playable, Tween
from .assets import AssetCache
from .layout import Layout
from .snapshot import BoardSnapshot, CardState
from .sprites.card_sprite import CardSprite
from .sprites.hud import PlayerHUDState
from .sprites.overlays import Banner, FloatingText, Toast


class Board:
    def __init__(self, assets: AssetCache, viewer: int, spectating: bool = False):
        self.assets = assets
        self.viewer = viewer
        self.spectating = spectating  # True when no seat is human (bot vs. bot)
        self.layout = Layout(viewer)
        self.sprites: Dict[int, CardSprite] = {}
        self.hud_states: Dict[int, PlayerHUDState] = {1: PlayerHUDState(), 2: PlayerHUDState()}
        self.particles = ParticleSystem()
        self.floaters: List[FloatingText] = []
        self.toasts: List[Toast] = []
        self.banners: List[Banner] = []
        self.snapshot: Optional[BoardSnapshot] = None

    # ---------------------------------------------------------- perspective ----

    def player_label(self, pid: int) -> str:
        """'You' / 'Your opponent' relative to the viewer, or 'Player N'
        while spectating a bot-vs-bot game (nobody to call "you")."""
        if self.spectating:
            return f"Player {pid}"
        return "You" if pid == self.viewer else "Your opponent"

    def is_second_person(self, pid: int) -> bool:
        """True if `player_label(pid)` reads as "You" (so a caller needs the
        "you"-conjugated verb form: "choose"/"have", not "chooses"/"has")."""
        return (not self.spectating) and pid == self.viewer

    # ------------------------------------------------------------ sprites ----

    def sprite_for(self, iid: int) -> CardSprite:
        sprite = self.sprites.get(iid)
        if sprite is None:
            sprite = CardSprite(iid, self.assets)
            self.sprites[iid] = sprite
        return sprite

    def slot(self, cs: CardState, snap: BoardSnapshot) -> Tuple[float, float, float, int, int]:
        """Where (center x, y, rotation degrees, w, h) `cs` belongs right now."""
        L = self.layout
        pid = cs.owner
        if cs.zone == "hand":
            rect = L.hand_rect(pid)
            slots = L.fan_slots(cs.zone_count, rect, S.HAND_CARD_W, S.HAND_CARD_H, arc_up=L.is_bottom(pid))
            x, y, rot = slots[cs.index]
            return x, y, rot, S.HAND_CARD_W, S.HAND_CARD_H
        if cs.zone in ("play_creature", "play_artifact"):
            rect = L.creature_row_rect(pid) if cs.zone == "play_creature" else L.artifact_row_rect(pid)
            slots = L.row_slots(cs.zone_count, rect, S.BOARD_CARD_W, S.BOARD_CARD_H)
            x, y = slots[cs.index]
            rot = 90.0 if (cs.exhausted and cs.zone == "play_creature") else 0.0
            return x, y, rot, S.BOARD_CARD_W, S.BOARD_CARD_H
        if cs.zone == "upgrade":
            host = snap.cards.get(cs.host_iid)
            if host is not None:
                hx, hy, hrot, hw, hh = self.slot(host, snap)
                ox = -16 + cs.index * 12
                return hx + ox, hy - 20, 0.0, int(S.BOARD_CARD_W * 0.55), int(S.BOARD_CARD_H * 0.55)
            return 0.0, 0.0, 0.0, S.BOARD_CARD_W, S.BOARD_CARD_H
        if cs.zone in ("discard", "archive", "purged", "deck"):
            rect = L.pile_rect(pid, cs.zone)
            return float(rect.centerx), float(rect.centery), 0.0, rect.width, rect.height
        return 0.0, 0.0, 0.0, S.BOARD_CARD_W, S.BOARD_CARD_H

    # -------------------------------------------------------------- visual ----

    def apply_visual(self, snap: BoardSnapshot) -> None:
        """Sync everything that ISN'T tweened (face, dim, tokens). Call this
        once per director build, before laying out position beats, so a
        card's face is already correct while it's mid-flight."""
        self.snapshot = snap
        for iid, cs in snap.cards.items():
            self.sprite_for(iid).apply_state(cs)

    def snap_all_instant(self, snap: BoardSnapshot) -> None:
        self.apply_visual(snap)
        for iid, cs in snap.cards.items():
            sprite = self.sprites[iid]
            x, y, rot, w, h = self.slot(cs, snap)
            sprite.set_size(w, h)
            sprite.snap_to(x, y, rot)
            sprite.scale = 1.0
            sprite.alpha = 255.0
            sprite.glow = None
        for pid, ps in snap.players.items():
            self.hud_states[pid].displayed_aember = float(ps.aember)

    def settle_beat(self, snap: BoardSnapshot, duration: float = S.T_SETTLE) -> Playable:
        """Tween every card to its snapshot-driven slot and the HUD counters
        to their true values. Always safe to call: the final catch-all so
        the board can never drift out of sync with the engine."""
        parts: List[Playable] = []
        for iid, cs in snap.cards.items():
            sprite = self.sprites[iid]
            x, y, rot, w, h = self.slot(cs, snap)
            sprite.set_size(w, h)
            sprite.glow = None
            parts.append(
                Parallel(
                    Tween(sprite, "x", x, duration),
                    Tween(sprite, "y", y, duration),
                    Tween(sprite, "rot", rot, duration),
                    Tween(sprite, "scale", 1.0, duration),
                    Tween(sprite, "alpha", 255.0, duration),
                )
            )
        for pid, ps in snap.players.items():
            parts.append(Tween(self.hud_states[pid], "displayed_aember", float(ps.aember), duration))
        return Parallel(*parts) if parts else Delay(0)

    # ------------------------------------------------------------- overlays ----

    def update(self, dt_ms: float) -> None:
        self.particles.update(dt_ms)
        self.floaters = [f for f in self.floaters if not f.update(dt_ms)]
        self.toasts = [t for t in self.toasts if not t.update(dt_ms)]
        self.banners = [b for b in self.banners if not b.update(dt_ms)]

    def draw_overlays(self, surface: pygame.Surface) -> None:
        self.particles.draw(surface)
        for f in self.floaters:
            f.draw(surface, self.assets)
        for t in self.toasts:
            t.draw(surface, self.assets)
        if self.banners:
            cx = S.PLAY_X + S.PLAY_W / 2
            cy = 60
            for b in self.banners:
                b.draw(surface, self.assets, cx, cy)
