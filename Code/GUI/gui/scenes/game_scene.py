"""The main board scene: renders the Board, drives the EngineBridge and
Director/Animator, and routes input through the DecisionPanel."""

from __future__ import annotations

from typing import Dict, Optional

import pygame

from .. import settings as S
from ..anim.animator import Animator
from ..anim.director import Director
from ..app import Scene
from ..board import Board
from ..decision.panel import DecisionPanel
from ..engine_bridge import EngineBridge, MatchSettings
from ..option_labels import describe_option
from ..sprites.hud import draw_hud
from ..sprites.log_panel import LogPanel
from ..sprites.overlays import Toast
from ..sprites.piles import draw_pile
from ..sprites.widgets import Button, draw_panel

_ZONE_DRAW_RANK = {
    "deck": 0, "discard": 1, "archive": 2, "purged": 3,
    "hand": 4, "play_artifact": 5, "play_creature": 6, "upgrade": 7,
}


def _felt(surface: pygame.Surface) -> None:
    rect = pygame.Rect(0, 0, S.BOARD_W, S.CANVAS_H)
    steps = 40
    for i in range(steps):
        t = i / steps
        color = tuple(int(a + (b - a) * t) for a, b in zip(S.FELT_TOP, S.FELT_BOTTOM))
        band = pygame.Rect(0, int(rect.height * i / steps), rect.width, int(rect.height / steps) + 1)
        pygame.draw.rect(surface, color, band)
    side = pygame.Rect(S.SIDE_X, 0, S.SIDE_W, S.CANVAS_H)
    pygame.draw.rect(surface, S.PANEL, side)


class GameScene(Scene):
    def __init__(self, settings: MatchSettings):
        self.settings = settings
        self.bridge: Optional[EngineBridge] = None
        self.board: Optional[Board] = None
        self.animator = Animator()
        self.panel = DecisionPanel()
        self.log_panel = LogPanel()
        self.viewer = 1
        self.human_seats = []
        self.last_snapshot = None
        self.log_sentences = []
        self._bot_pending_choice = None
        self._bot_think_timer = 0.0
        self._game_over_pushed = False
        self.hover_iid: Optional[int] = None
        self.pinned_iid: Optional[int] = None
        self.browsing: Optional[tuple] = None  # (pid, kind) while a pile browser modal is open
        self.decklist_pid: Optional[int] = None  # which player's full 36-card pool is shown, or None
        self.browse_scroll = 0
        self.reveal_hands = False  # bot-vs-bot spectate: show both hands face up (R to toggle)
        self.show_help = False

    # -------------------------------------------------------------- setup ----

    def on_enter(self) -> None:
        from .curtain_scene import CurtainScene  # local import: avoid a cycle

        self._CurtainScene = CurtainScene
        self.bridge = EngineBridge(self.settings)
        self.human_seats = [pid for pid, s in self.bridge.seats.items() if s == "human"]
        self.viewer = self.human_seats[0] if self.human_seats else 1
        self.board = Board(self.app.assets, self.viewer, spectating=not self.human_seats)
        snap = self.bridge.snapshot(self.viewer)
        self.board.snap_all_instant(snap)
        self.last_snapshot = snap
        self._sync_log()
        self._sync_reveal_hands()
        self._maybe_show_curtain()

    def _sync_reveal_hands(self) -> None:
        """Bot-vs-bot spectate only: with `reveal_hands` on, show both hands
        face up. Never applies with a human seat at the table."""
        if self.last_snapshot is None:
            return
        can_reveal = self.reveal_hands and not self.human_seats
        for cs in self.last_snapshot.cards.values():
            if cs.zone != "hand":
                continue
            sprite = self.board.sprites.get(cs.iid)
            if sprite is None:
                continue
            if can_reveal:
                sprite.face_up = True
                sprite.image_path = cs.image
            else:
                sprite.face_up = cs.face_up
                sprite.image_path = cs.image if cs.face_up else None

    # ------------------------------------------------------------ hot-seat ----

    def _maybe_show_curtain(self) -> None:
        if self.bridge.is_over or len(self.human_seats) < 2:
            return
        d = self.bridge.pending_decision
        if d is None or d.player == self.viewer:
            return
        if self.app.scenes and isinstance(self.app.scenes[-1], self._CurtainScene):
            return
        self.app.push(self._CurtainScene(d.player, on_ready=self._switch_viewer))

    def _switch_viewer(self) -> None:
        d = self.bridge.pending_decision
        if d is None:
            return
        self.viewer = d.player
        self.board.viewer = self.viewer
        from ..layout import Layout

        self.board.layout = Layout(self.viewer)
        snap = self.bridge.snapshot(self.viewer)
        self.board.snap_all_instant(snap)
        self.last_snapshot = snap
        self._sync_log()
        self._sync_reveal_hands()
        self.panel.decision = None

    # ---------------------------------------------------------------- log ----

    def _sync_log(self) -> None:
        from ..option_labels import describe_log_event

        self.log_sentences = []
        for ev in self.bridge.game.log.events:
            text = describe_log_event(ev, self.viewer, spectating=not self.human_seats)
            if text:
                self.log_sentences.append(text)

    # -------------------------------------------------------------- update ----

    def update(self, dt_ms: float) -> None:
        self.board.update(dt_ms)
        self.animator.update(dt_ms)
        self._update_hover()

        if self.animator.is_busy:
            return

        if self.bridge.is_over:
            self._on_game_over()
            return

        d = self.bridge.pending_decision
        if d is None:
            return

        if len(self.human_seats) == 2 and d.player != self.viewer:
            self._maybe_show_curtain()
            return

        if self.bridge.seat_is_bot(d.player):
            self._step_bot(dt_ms)
            return

        if self.panel.decision is not d:
            view = self.bridge.game.view_for(d.player)
            self.panel.on_enter(d, view, self.board)
            self.panel.sync_glow(self.board)
        else:
            self.panel.sync_glow(self.board)

        if self.panel.result is not None:
            self._submit(self.panel.result)

    def _on_game_over(self) -> None:
        if not self._game_over_pushed:
            self._game_over_pushed = True
            self.animator.enqueue(Director.game_over_beat(self.board, self.last_snapshot))
            return
        if not self.animator.is_busy:
            from .game_over_scene import GameOverScene

            self.app.push(GameOverScene(self.bridge.game.result, self.settings))

    def _step_bot(self, dt_ms: float) -> None:
        d = self.bridge.pending_decision
        if self._bot_pending_choice is None:
            choice = self.bridge.bot_choice(d.player)
            self._bot_pending_choice = choice
            self._bot_think_timer = S.T_BOT_THINK
            self._preview_bot_choice(d, choice)
            return
        self._bot_think_timer -= dt_ms
        if self._bot_think_timer <= 0:
            choice = self._bot_pending_choice
            self._bot_pending_choice = None
            self._submit(choice)

    def _preview_bot_choice(self, decision, choice) -> None:
        view = self.bridge.game.view_for(decision.player)
        label = describe_option(choice, decision, view)
        who = self.board.player_label(decision.player)
        card = getattr(choice, "card", choice if hasattr(choice, "instance_id") else None)
        pos = None
        if card is not None:
            sprite = self.board.sprites.get(getattr(card, "instance_id", None))
            if sprite is not None:
                sprite.glow = "target"
                pos = (sprite.x, max(60, sprite.y - 70))
        if pos is None:
            prompt = self.board.layout.prompt_rect()
            pos = (prompt.centerx, prompt.centery)
        self.board.toasts.append(Toast(pos[0], pos[1], f"{who}: {label}"))

    def _submit(self, choice) -> None:
        for sprite in self.board.sprites.values():
            if sprite.glow == "target":
                sprite.glow = None
        before, events, after = self.bridge.submit(choice, self.viewer)
        beat = Director.build(self.board, before, after, events)
        self.animator.enqueue(beat)
        self.last_snapshot = after
        self._sync_log()
        self._sync_reveal_hands()
        self.panel.decision = None
        self.panel.result = None

    # ------------------------------------------------------------- hover ----

    def _update_hover(self) -> None:
        mx, my = pygame.mouse.get_pos()
        best_iid, best_rank = None, -1
        for iid, cs in self.last_snapshot.cards.items():
            sprite = self.board.sprites.get(iid)
            if sprite is None or not sprite.visible or not sprite.face_up:
                continue
            if sprite.contains_point(mx, my):
                rank = _ZONE_DRAW_RANK.get(cs.zone, 0) * 1000 + cs.index
                if rank > best_rank:
                    best_rank, best_iid = rank, iid
        self.hover_iid = best_iid

    # ------------------------------------------------------------- events ----

    def handle_event(self, event: pygame.event.Event) -> None:
        if self.decklist_pid is not None:
            self._handle_decklist_event(event)
            return
        if self.browsing is not None:
            self._handle_browser_event(event)
            return

        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            if self.show_help:
                self.show_help = False
            elif not self.animator.is_busy:
                self.app.pop()
            return

        if event.type == pygame.KEYDOWN and event.key in (pygame.K_1, pygame.K_2, pygame.K_3):
            self.animator.speed = {pygame.K_1: 0.5, pygame.K_2: 1.0, pygame.K_3: 2.0}[event.key]
            return

        if event.type == pygame.KEYDOWN and event.key in (pygame.K_SLASH, pygame.K_h) and not (event.mod & pygame.KMOD_CTRL):
            self.show_help = not self.show_help
            return

        if event.type == pygame.KEYDOWN and event.key == pygame.K_d:
            self.decklist_pid = self.viewer
            self.browse_scroll = 0
            return

        if event.type == pygame.KEYDOWN and event.key == pygame.K_r and not self.human_seats:
            self.reveal_hands = not self.reveal_hands
            self._sync_reveal_hands()
            return

        if event.type == pygame.KEYDOWN and event.key == pygame.K_m:
            self.app.assets.muted = not self.app.assets.muted
            return

        if self.show_help:
            return

        if self.animator.is_busy:
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self.animator.skip()
            return

        self.log_panel.handle_event(event, self.board.layout.log_rect())

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
            self.pinned_iid = self.hover_iid if self.pinned_iid != self.hover_iid else None
            return

        if self.bridge.is_over:
            return
        d = self.bridge.pending_decision
        if d is None or self.bridge.seat_is_bot(d.player) or (len(self.human_seats) == 2 and d.player != self.viewer):
            return

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and not self.panel.modal_open:
            if self._maybe_click_pile(event.pos):
                return

        mouse_pos = event.pos if hasattr(event, "pos") else pygame.mouse.get_pos()
        self.panel.handle_event(event, self.board, mouse_pos, self.board.layout)

    def _maybe_click_pile(self, pos) -> bool:
        L = self.board.layout
        ps = self.last_snapshot.players
        for pid in (1, 2):
            for kind in ("deck", "discard", "archive", "purged"):
                rect = L.pile_rect(pid, kind)
                if not rect.collidepoint(pos):
                    continue
                count = getattr(ps[pid], f"{kind}_count")
                if count == 0:
                    return True
                visible = kind != "deck" and (kind != "archive" or pid == self.viewer)
                if visible:
                    self.browsing = (pid, kind)
                    self.browse_scroll = 0
                return True
        return False

    def _handle_browser_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.MOUSEWHEEL:
            self.browse_scroll = max(0, self.browse_scroll - event.y)
        elif event.type == pygame.MOUSEBUTTONDOWN or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            self.browsing = None

    def _handle_decklist_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.MOUSEWHEEL:
            self.browse_scroll = max(0, self.browse_scroll - event.y)
            return
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.decklist_pid = None
            return
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            w, h = 780, 600
            rect = pygame.Rect((S.CANVAS_W - w) // 2, (S.CANVAS_H - h) // 2, w, h)
            tab1 = pygame.Rect(rect.left + 16, rect.top + 10, 110, 28)
            tab2 = pygame.Rect(rect.left + 132, rect.top + 10, 110, 28)
            if tab1.collidepoint(event.pos):
                self.decklist_pid, self.browse_scroll = 1, 0
            elif tab2.collidepoint(event.pos):
                self.decklist_pid, self.browse_scroll = 2, 0
            elif not rect.collidepoint(event.pos):
                self.decklist_pid = None

    # --------------------------------------------------------------- draw ----

    def draw(self, surface: pygame.Surface) -> None:
        _felt(surface)
        L = self.board.layout
        snap = self.last_snapshot

        for pid in (1, 2):
            for kind in ("deck", "discard", "archive", "purged"):
                rect = L.pile_rect(pid, kind)
                ps = snap.players[pid]
                count = getattr(ps, f"{kind}_count")
                draw_pile(surface, self.app.assets, rect, kind, count, hovered=rect.collidepoint(pygame.mouse.get_pos()))

        for pid in (1, 2):
            effects = [e for e in snap.active_effects if e["player_affected"] == pid]
            is_you = pid == self.viewer and pid in self.human_seats
            label = f"Player {pid}" + (" (You)" if is_you else "")
            draw_hud(surface, self.app.assets, L.hud_rect(pid), snap.players[pid], self.board.hud_states[pid], snap.active_player == pid, label, effects)

        for cs in sorted(snap.cards.values(), key=lambda c: _ZONE_DRAW_RANK.get(c.zone, 0) * 1000 + c.index):
            sprite = self.board.sprites.get(cs.iid)
            if sprite is not None:
                sprite.draw(surface)

        self.board.draw_overlays(surface)
        self.panel.draw(surface, self.app.assets, L)
        self._draw_zoom(surface, L)
        self.log_panel.draw(surface, self.app.assets, L.log_rect(), self.log_sentences)

        if self.browsing is not None:
            self._draw_browser(surface)
        if self.decklist_pid is not None:
            self._draw_decklist(surface)
        if self.show_help:
            self._draw_help(surface)

        mute_txt = "M unmute" if self.app.assets.muted else "M mute"
        hint = self.app.assets.font("inter", 11).render(
            f"D decklists · H help · {mute_txt}" + ("  ·  R reveal hands" if not self.human_seats else ""),
            True, S.TEXT_FAINT,
        )
        surface.blit(hint, (S.LEFT_COL_X, S.CANVAS_H - 16))

    def _draw_zoom(self, surface, L) -> None:
        rect = L.zoom_rect()
        draw_panel(surface, rect, alpha=190)
        iid = self.pinned_iid or self.hover_iid
        cs = self.last_snapshot.cards.get(iid) if iid is not None else None
        title_font = self.app.assets.font("cinzel", 15)
        title = title_font.render("Pinned" if self.pinned_iid else "Card", True, S.TEXT_DIM)
        surface.blit(title, (rect.left + 10, rect.top + 8))
        if cs is None or not cs.face_up:
            hint = self.app.assets.font("inter", 13).render("Hover a card to inspect it", True, S.TEXT_FAINT)
            surface.blit(hint, hint.get_rect(center=(rect.centerx, rect.centery)))
            return
        img_w, img_h = rect.width - 20, int((rect.width - 20) * (S.CARD_ART_H / S.CARD_ART_W))
        img = self.app.assets.card_face(cs.image, (img_w, img_h))
        img_rect = img.get_rect(midtop=(rect.centerx, rect.top + 32))
        surface.blit(img, img_rect)
        info_font = self.app.assets.font("inter", 13)
        y = img_rect.bottom + 8
        for line in (cs.name, f"{cs.house} · {cs.type}"):
            t = info_font.render(line, True, S.TEXT)
            surface.blit(t, (rect.left + 10, y))
            y += t.get_height() + 2
        if cs.type == "Creature":
            t = info_font.render(f"Power {cs.power}   Damage {cs.damage}   Armor {cs.armor}", True, S.TEXT_DIM)
            surface.blit(t, (rect.left + 10, y))

    def _dim_backdrop(self, surface) -> None:
        dim = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 150))
        surface.blit(dim, (0, 0))

    def _draw_card_grid(self, surface, rect: pygame.Rect, body_top: int, cards) -> None:
        """Shared scrollable grid used by the pile browser and the decklist
        viewer. `self.browse_scroll` (rows) drives the mouse-wheel scroll."""
        body = pygame.Rect(rect.left + 14, rect.top + body_top, rect.width - 28, rect.bottom - rect.top - body_top - 16)
        cols = max(1, body.width // 112)
        rows_total = (len(cards) + cols - 1) // cols
        rows_visible = max(1, body.height // 154)
        max_scroll = max(0, rows_total - rows_visible)
        self.browse_scroll = max(0, min(self.browse_scroll, max_scroll))

        clip = surface.get_clip()
        surface.set_clip(body)
        for i, card in enumerate(cards):
            gx = i % cols
            gy = i // cols - self.browse_scroll
            cell = pygame.Rect(body.left + gx * 112, body.top + gy * 154, 100, 140)
            if cell.bottom < body.top or cell.top > body.bottom:
                continue
            img = self.app.assets.card_face(card.image, (100, 140))
            surface.blit(img, cell)
        surface.set_clip(clip)

        if max_scroll > 0:
            hint = self.app.assets.font("inter", 11).render(
                f"scroll for more ({self.browse_scroll + 1}/{max_scroll + 1})", True, S.TEXT_FAINT
            )
            surface.blit(hint, (body.left, rect.bottom - 14))

    def _draw_browser(self, surface) -> None:
        pid, kind = self.browsing
        player = self.bridge.game.players[pid]
        zone = getattr(player, kind)
        cards = zone.cards()

        self._dim_backdrop(surface)
        w, h = 700, 560
        rect = pygame.Rect((S.CANVAS_W - w) // 2, (S.CANVAS_H - h) // 2, w, h)
        draw_panel(surface, rect, alpha=245, border=S.AEMBER)
        if self.board.spectating:
            who = f"Player {pid}'s"
        else:
            who = "Your" if pid == self.viewer else "Opponent's"
        title = self.app.assets.font("cinzel", 18).render(f"{who} {kind} ({len(cards)})", True, S.TEXT)
        surface.blit(title, (rect.left + 16, rect.top + 10))
        self._draw_card_grid(surface, rect, 44, cards)

    def _draw_decklist(self, surface) -> None:
        pid = self.decklist_pid
        cards = sorted(self.bridge.game.players[pid].all_cards, key=lambda c: (c.house.value, c.name))

        self._dim_backdrop(surface)
        w, h = 780, 600
        rect = pygame.Rect((S.CANVAS_W - w) // 2, (S.CANVAS_H - h) // 2, w, h)
        draw_panel(surface, rect, alpha=245, border=S.AEMBER)
        title = self.app.assets.font("cinzel", 18).render(f"Decklist ({len(cards)} cards, unordered)", True, S.TEXT)
        surface.blit(title, (rect.left + 260, rect.top + 12))

        for tab_pid, x in ((1, rect.left + 16), (2, rect.left + 132)):
            tab = pygame.Rect(x, rect.top + 10, 110, 28)
            active = tab_pid == pid
            pygame.draw.rect(surface, S.AEMBER if active else S.PANEL_LIGHT, tab, border_radius=6)
            pygame.draw.rect(surface, S.AEMBER, tab, width=1, border_radius=6)
            tf = self.app.assets.font("inter", 13, bold=True)
            label = tf.render(f"Player {tab_pid}", True, S.BLACK if active else S.TEXT)
            surface.blit(label, label.get_rect(center=tab.center))

        self._draw_card_grid(surface, rect, 50, cards)

    def _draw_help(self, surface) -> None:
        self._dim_backdrop(surface)
        w, h = 560, 420
        rect = pygame.Rect((S.CANVAS_W - w) // 2, (S.CANVAS_H - h) // 2, w, h)
        draw_panel(surface, rect, alpha=250, border=S.AEMBER)
        title = self.app.assets.font("cinzel", 20).render("Controls", True, S.TEXT)
        surface.blit(title, (rect.left + 18, rect.top + 14))

        lines = [
            ("Click a glowing card", "play / discard / reap / fight / use it"),
            ("O", "open the full list of legal choices"),
            ("Hover / right-click", "zoom a card / pin the zoom"),
            ("Click a pile", "browse discard, purged, or your archive"),
            ("D", "view either player's full 36-card decklist"),
            ("M", "mute / unmute sound"),
            ("Space", "skip the current animation"),
            ("1 / 2 / 3", "animation speed 0.5x / 1x / 2x"),
            ("F11 / F3", "fullscreen / show FPS"),
            ("Esc", "back"),
        ]
        if not self.human_seats:
            lines.insert(4, ("R", "spectating: reveal both hands"))

        y = rect.top + 56
        label_font = self.app.assets.font("inter", 14, bold=True)
        desc_font = self.app.assets.font("inter", 13)
        for key, desc in lines:
            k = label_font.render(key, True, S.AEMBER)
            surface.blit(k, (rect.left + 20, y))
            d = desc_font.render(desc, True, S.TEXT_DIM)
            surface.blit(d, (rect.left + 230, y + 1))
            y += 32
