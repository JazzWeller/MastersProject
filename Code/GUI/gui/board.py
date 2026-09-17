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

# Draw order, back to front. Hit-testing walks the same order in reverse, so
# whatever is drawn on top is what the mouse gets (Code/PLAYTEST_FIX_PLAN.md M2).
ZONE_DRAW_RANK = {
    "deck": 0, "discard": 1, "archive": 2, "purged": 3,
    "play_artifact": 4, "play_creature": 5, "upgrade": 6, "hand": 7,
}


def draw_key(cs: CardState) -> int:
    return ZONE_DRAW_RANK.get(cs.zone, 0) * 1000 + cs.index


class Board:
    def __init__(self, assets: AssetCache, viewer: int, spectating: bool = False):
        self.assets = assets
        self.viewer = viewer
        self.spectating = spectating  # True when no seat is human (bot vs. bot, replays)
        self.layout = Layout(viewer)
        self.sprites: Dict[int, CardSprite] = {}
        self.hud_states: Dict[int, PlayerHUDState] = {1: PlayerHUDState(), 2: PlayerHUDState()}
        self.particles = ParticleSystem()
        self.floaters: List[FloatingText] = []
        self.toasts: List[Toast] = []
        self.banners: List[Banner] = []
        self.snapshot: Optional[BoardSnapshot] = None
        # Filled in by draw_hud each frame: where each player's aember number
        # and key icons actually are, for animations and hover tooltips.
        self.hud_rects: Dict[int, dict] = {1: {}, 2: {}}
        self.hidden_iids: set = set()  # sprites a full-screen overlay has taken over

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

    def set_viewer(self, viewer: int) -> None:
        self.viewer = viewer
        self.layout = Layout(viewer)

    # ------------------------------------------------------------ sprites ----

    def sprite_for(self, iid: int) -> CardSprite:
        sprite = self.sprites.get(iid)
        if sprite is None:
            sprite = CardSprite(iid, self.assets)
            self.sprites[iid] = sprite
        return sprite

    def draw_order(self, snap: Optional[BoardSnapshot] = None) -> List[CardState]:
        snap = snap or self.snapshot
        if snap is None:
            return []
        return sorted(snap.cards.values(), key=draw_key)

    def card_at(self, pos, face_up_only: bool = False) -> Optional[int]:
        """The topmost visible card under `pos` -- the single answer every
        hover, click and inspect uses."""
        x, y = pos
        for cs in reversed(self.draw_order()):
            if cs.iid in self.hidden_iids:
                continue
            sprite = self.sprites.get(cs.iid)
            if sprite is None or not sprite.visible or sprite.alpha <= 1:
                continue
            if face_up_only and not sprite.face_up:
                continue
            if sprite.contains_point(x, y):
                return cs.iid
        return None

    def click_target_at(self, pos) -> Optional[int]:
        """Like `card_at`, but an upgrade forwards the click to the creature
        it's attached to: the upgrade covers part of its host, and is never a
        legal target on its own (Code/PLAYTEST_FIX_PLAN.md M5)."""
        iid = self.card_at(pos)
        if iid is None or self.snapshot is None:
            return iid
        cs = self.snapshot.cards.get(iid)
        if cs is not None and cs.zone == "upgrade" and cs.host_iid is not None:
            return cs.host_iid
        return iid

    def slot(self, cs: CardState, snap: BoardSnapshot) -> Tuple[float, float, float, int, int]:
        """Where (center x, y, rotation degrees, w, h) `cs` belongs right now."""
        L = self.layout
        pid = cs.owner
        if cs.zone == "hand":
            slots = L.hand_slots(pid, cs.zone_count)
            x, y, rot = slots[cs.index]
            w, h = L.hand_card_size(pid)
            return x, y, rot, w, h
        if cs.zone in ("play_creature", "play_artifact"):
            # build_snapshot sets `owner` to the player whose play area the
            # card is in, which is what index/zone_count are relative to.
            rect = L.creature_row_rect(pid) if cs.zone == "play_creature" else L.artifact_row_rect(pid)
            slots = L.row_slots(cs.zone_count, rect, S.BOARD_CARD_W, S.BOARD_CARD_H)
            x, y = slots[cs.index]
            return x, y, 0.0, S.BOARD_CARD_W, S.BOARD_CARD_H
        if cs.zone == "upgrade":
            host = snap.cards.get(cs.host_iid)
            if host is not None:
                hx, hy, _hrot, _hw, _hh = self.slot(host, snap)
                x, y, w, h = L.upgrade_slot(hx, hy, cs.index)
                return x, y, 0.0, w, h
            return 0.0, 0.0, 0.0, S.UPGRADE_TAB_W, S.UPGRADE_TAB_H
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
            # Over the middle of the opponent's board: clear of the prompt
            # band, the action bar and any decision modal (which open lower).
            cx = S.PLAY_X + S.PLAY_W / 2
            cy = self.layout.board_rect(3 - self.viewer).centery - 30
            for b in self.banners[-2:]:
                b.draw(surface, self.assets, cx, cy)
                cy += 74
