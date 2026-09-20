"""The main board scene: renders the Board, drives the EngineBridge and
Director/Animator, and routes input through the DecisionPanel."""

from __future__ import annotations

from types import SimpleNamespace
from typing import List, Optional, Tuple

import pygame

from .. import settings as S
from ..anim.animator import Animator
from ..anim.director import Director
from ..app import Scene
from ..board import Board, ZONE_DRAW_RANK, draw_key
from ..decision.panel import DecisionPanel
from ..engine_bridge import EngineBridge, MatchSettings
from ..option_labels import describe_log_event, describe_option, log_event_category, log_event_iid
from ..sprites.hud import draw_hud
from ..sprites.log_panel import LogPanel
from ..sprites.overlays import Banner, Toast
from ..sprites.piles import draw_pile
from ..sprites.widgets import draw_panel

from keyforge.cards.card_data import get_card_def
from keyforge.enums import DecisionKind

_ZONE_DRAW_RANK = ZONE_DRAW_RANK  # kept for older imports

QUEUED_CLICK_MS = 1500
LEAVE_CONFIRM_MS = 3000  # a second Esc within this window leaves the game


def _card_info_from_state(cs) -> SimpleNamespace:
    cdef = get_card_def(cs.name)
    return SimpleNamespace(
        name=cs.name, house=cs.house, type=cs.type, image=cs.image,
        power=cs.power, armor=cs.armor, damage=cs.damage,
        text=cdef.text, errata=cdef.errata,
    )


def _card_info_from_engine_card(card) -> SimpleNamespace:
    to = card.type_object
    return SimpleNamespace(
        name=card.name, house=card.house.value, type=card.type.value, image=card.card_def.image,
        power=getattr(to, "base_power", 0), armor=getattr(to, "base_armor", 0), damage=getattr(to, "damage", 0),
        text=card.card_def.text, errata=card.card_def.errata,
    )


def _wrap_text(text: str, font, max_w: int) -> List[str]:
    words, lines, cur = text.split(" "), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if font.size(trial)[0] <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def _felt(surface: pygame.Surface) -> None:
    rect = pygame.Rect(0, 0, S.BOARD_W, S.CANVAS_H)
    steps = 40
    for i in range(steps):
        t = i / steps
        color = tuple(int(a + (b - a) * t) for a, b in zip(S.FELT_TOP, S.FELT_BOTTOM))
        band = pygame.Rect(0, int(rect.height * i / steps), rect.width, int(rect.height / steps) + 1)
        pygame.draw.rect(surface, color, band)
    pygame.draw.rect(surface, S.PANEL, pygame.Rect(S.SIDE_X, 0, S.SIDE_W, S.CANVAS_H))


class GameScene(Scene):
    record_history = True

    def __init__(self, settings: MatchSettings):
        self.settings = settings
        self.bridge: Optional[EngineBridge] = None
        self.board: Optional[Board] = None
        self.animator = Animator()
        self.panel = DecisionPanel()
        self.log_panel = LogPanel()
        self.viewer = 1
        self.human_seats: List[int] = []
        self.last_snapshot = None
        self.log_sentences = []
        self._bot_pending_choice = None
        self._bot_think_timer = 0.0
        self._game_over_pushed = False
        self.hover_iid: Optional[int] = None
        self._hover_info: Optional[SimpleNamespace] = None
        self.browsing: Optional[tuple] = None  # (pid, kind) while a pile browser is open
        self.decklist_pid: Optional[int] = None
        self.browse_scroll = 0
        self.reveal_hands = False
        self.show_help = False
        self.inspect_info = None
        self._inspect_card_rect: Optional[pygame.Rect] = None
        self._announced_first = False
        self._leave_armed_ms = 0.0
        self._queued_click: Optional[Tuple[tuple, int, float]] = None
        # Rebuilt every draw; read on the next update/event.
        self._overlay_targets: List[Tuple[pygame.Rect, int]] = []
        self._log_targets: List[Tuple[pygame.Rect, int]] = []
        self._tooltip_targets: List[Tuple[pygame.Rect, List[str]]] = []

    # -------------------------------------------------------------- setup ----

    def _make_bridge(self):
        history = self.app.history if self.record_history else None
        return EngineBridge(self.settings, history=history)

    def on_enter(self) -> None:
        from .curtain_scene import CurtainScene  # local import: avoid a cycle

        self._CurtainScene = CurtainScene
        self.bridge = self._make_bridge()
        self.human_seats = [pid for pid, s in self.bridge.seats.items() if s == "human"]
        self.viewer = self.human_seats[0] if self.human_seats else 1
        self.board = Board(self.app.assets, self.viewer, spectating=not self.human_seats)
        self._resync_board()
        self._maybe_show_curtain()

    def on_exit(self) -> None:
        if self.bridge is not None:
            self.bridge.close(abandoned=True)

    def _resync_board(self) -> None:
        snap = self.bridge.snapshot(self.viewer)
        self.board.snap_all_instant(snap)
        self.last_snapshot = snap
        self._sync_log()
        self._sync_reveal_hands()

    def _sync_reveal_hands(self) -> None:
        """Spectating only: with `reveal_hands` on, show both hands face up.
        Never applies with a human seat at the table."""
        if self.last_snapshot is None:
            return
        can_reveal = self.reveal_hands and not self.human_seats
        for cs in self.last_snapshot.cards.values():
            if cs.zone not in ("hand", "archive"):
                continue
            sprite = self.board.sprites.get(cs.iid)
            if sprite is None:
                continue
            face = cs.face_up or can_reveal
            sprite.face_up = face
            sprite.image_path = cs.image if face else None

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
        self.board.set_viewer(self.viewer)
        self._resync_board()
        self.panel.reset()

    # ---------------------------------------------------------------- log ----

    def _sync_log(self) -> None:
        """(text, category, iid) per line. A synthetic "turn" separator goes
        before each turn's first line; the house choice (or forced house) is
        the first event of every turn."""
        self.log_sentences = []
        spectating = not self.human_seats
        for ev in self.bridge.game.log.events:
            if ev.kind == "turn_start":
                who = self.board.player_label(ev.data["player"]) if self.board else f"Player {ev.data['player']}"
                whose = "Your" if who == "You" else (f"{who}'s")
                self.log_sentences.append((f"— Turn {ev.data['turn']} · {whose} turn —", "turn", None))
                continue
            text = describe_log_event(ev, self.viewer, spectating=spectating)
            if text:
                self.log_sentences.append((text, log_event_category(ev), log_event_iid(ev, self.viewer)))

    # -------------------------------------------------------------- update ----

    def _is_my_decision(self, d) -> bool:
        return (
            d is not None
            and not self.bridge.seat_is_bot(d.player)
            and not (len(self.human_seats) == 2 and d.player != self.viewer)
        )

    def update(self, dt_ms: float) -> None:
        self.board.update(dt_ms)
        self._leave_armed_ms = max(0.0, self._leave_armed_ms - dt_ms)
        self.animator.update(dt_ms)
        self._update_hover()
        if self._queued_click is not None:
            pos, iid, age = self._queued_click
            self._queued_click = (pos, iid, age + dt_ms)

        if self.animator.is_busy:
            return

        if self.bridge.is_over:
            self._on_game_over()
            return

        self._announce_first_player()

        d = self.bridge.pending_decision
        if d is None:
            return

        if len(self.human_seats) == 2 and d.player != self.viewer:
            self._maybe_show_curtain()
            return

        if self.bridge.seat_is_bot(d.player):
            self._clear_unusable()
            self._step_bot(dt_ms)
            return

        if self.panel.decision is not d:
            view = self.bridge.game.view_for(d.player)
            self.panel.on_enter(d, view, self.board)
            self._sync_unusable(d)
        self.panel.sync_glow(self.board)
        self._replay_queued_click()

        if self.panel.result is not None:
            self._submit(self.panel.result)

    def _announce_first_player(self) -> None:
        game = self.bridge.game
        if self._announced_first or game.turn_number < 1:
            return
        self._announced_first = True
        first = game.first_player
        label = self.board.player_label(first)
        verb = "go" if self.board.is_second_person(first) else "goes"
        self.board.banners.append(
            Banner(f"{label} {verb} first", "First turn: only one card may be played or discarded.", life_ms=2600)
        )

    def _on_game_over(self) -> None:
        if not self._game_over_pushed:
            self._game_over_pushed = True
            self.bridge.close(abandoned=False)
            self.animator.enqueue(Director.game_over_beat(self.board, self.last_snapshot))
            return
        if not self.animator.is_busy and not isinstance(self.app.scenes[-1], _game_over_cls()):
            if self.app.scenes[-1] is self:
                self.app.push(_game_over_cls()(self.bridge.game, self.settings, self))

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
        if isinstance(choice, list):
            label = ", ".join(describe_option(c, decision, view) for c in choice) or "nothing"
        who = self.board.player_label(decision.player)
        card = getattr(choice, "card", choice if hasattr(choice, "instance_id") else None)
        pos = None
        if card is not None:
            sprite = self.board.sprites.get(getattr(card, "instance_id", None))
            if sprite is not None:
                sprite.glow = "target"
                pos = (sprite.x, max(40, sprite.y - sprite.h / 2 - 20))
        if pos is None:
            prompt = self.board.layout.prompt_rect()
            pos = (prompt.centerx, prompt.centery)
        self.board.toasts = [t for t in self.board.toasts if t.age_ms < t.life_ms * 0.5]
        self.board.toasts.append(Toast(pos[0], pos[1], f"{who}: {label}"))

    def _submit(self, choice) -> None:
        for sprite in self.board.sprites.values():
            if sprite.glow == "target":
                sprite.glow = None
        self._clear_unusable()
        before, events, after = self.bridge.submit(choice, self.viewer)
        beat = Director.build(self.board, before, after, events)
        self.animator.enqueue(beat)
        self.last_snapshot = after
        self._sync_log()
        self._sync_reveal_hands()
        self.panel.reset()

    # ------------------------------------------------- "why can't I use it" ----

    def _clear_unusable(self) -> None:
        for sprite in self.board.sprites.values():
            sprite.unusable_reason = None
            sprite.hint = None

    def _sync_unusable(self, d) -> None:
        """Dim cards you can't use right now, with the reason on hover (U2/U18)."""
        self._clear_unusable()
        if d.kind != DecisionKind.CHOOSE_ACTION:
            return
        game = self.bridge.game
        pid = d.player
        player = game.players[pid]
        options = self.panel.card_option_map
        for card in player.hand.cards():
            sprite = self.board.sprites.get(card.instance_id)
            if sprite is None:
                continue
            reason = game.why_not_playable(pid, card)
            if reason is None:
                continue
            if card.instance_id in options:  # can still be discarded
                sprite.unusable_reason = None
                sprite.hint = f"Can't play: {reason}. You can still discard it."
            else:
                sprite.unusable_reason = reason
        for card in list(player.play_area.creatures) + list(player.play_area.artifacts):
            sprite = self.board.sprites.get(card.instance_id)
            if sprite is None:
                continue
            if card.instance_id in options:
                # Still usable, but stun will consume it with no effect, or
                # it can only fight this turn -- worth a hover hint even
                # though it isn't dimmed.
                if card.stunned:
                    sprite.hint = f"{card.name} is stunned: this use will have no effect."
                elif player.get_can_only_fight(game):
                    sprite.hint = f"{card.name} can only be used to fight this turn."
                continue
            sprite.unusable_reason = game.why_not_usable(pid, card)

    # ------------------------------------------------------------- hover ----

    def _overlay_blocks_board(self) -> bool:
        return bool(
            self.panel.full_screen
            or (self.panel.decision is not None and self.panel.modal_open and self.panel._modal_is_grid())
            or self.browsing or self.decklist_pid is not None or self.show_help or self.inspect_info
        )

    def _sync_hidden(self) -> None:
        """Cards a full-screen review (mulligan / archive) is showing large are
        taken off the board -- neither drawn nor hit-tested there."""
        hidden = set()
        if self.panel.full_screen and self.last_snapshot is not None:
            zone = "hand" if self.panel.mulligan_open else "archive"
            hidden = {cs.iid for cs in self.last_snapshot.zone_cards(self.panel.decision.player, zone)}
        self.board.hidden_iids = hidden

    def _update_hover(self, pos=None) -> None:
        """One answer to "what card is the mouse over", used by the zoom panel,
        right/middle-click inspect and tooltips."""
        pos = self.mouse if pos is None else pos
        self._sync_hidden()
        self.hover_iid, self._hover_info = None, None
        for rect, iid in reversed(self._overlay_targets):
            if rect.collidepoint(pos):
                self._set_hover(iid, allow_hidden=True)
                return
        for rect, iid in self._log_targets:
            if rect.collidepoint(pos):
                self._set_hover(iid, allow_hidden=True)
                return
        if self._overlay_blocks_board():
            return
        if self.panel.decision is not None and self.panel.modal_open:
            modal = self.panel._modal_rect(self.app.assets)
            if modal.collidepoint(pos):
                return
        iid = self.board.card_at(pos)
        if iid is not None:
            self._set_hover(iid, allow_hidden=False)

    def _set_hover(self, iid: int, allow_hidden: bool) -> None:
        info = self._card_info(iid, allow_hidden)
        if info is not None:
            self.hover_iid, self._hover_info = iid, info

    def _card_info(self, iid: int, allow_hidden: bool) -> Optional[SimpleNamespace]:
        cs = self.last_snapshot.cards.get(iid) if self.last_snapshot else None
        sprite = self.board.sprites.get(iid)
        if cs is not None and (cs.face_up or (sprite is not None and sprite.face_up)):
            return _card_info_from_state(cs)
        if not allow_hidden:
            return None
        for player in self.bridge.game.players.values():
            for card in player.all_cards:
                if card.instance_id == iid:
                    return _card_info_from_engine_card(card)
        return None

    # ------------------------------------------------------------- events ----

    def handle_event(self, event: pygame.event.Event) -> None:
        if self.inspect_info is not None:
            self._handle_inspector_event(event)
            return
        if self.decklist_pid is not None:
            self._handle_decklist_event(event)
            return
        if self.browsing is not None:
            self._handle_browser_event(event)
            return

        if event.type == pygame.KEYDOWN:
            if self._handle_key(event):
                return

        if self.show_help:
            if event.type == pygame.MOUSEBUTTONDOWN:
                self.show_help = False
            return

        # Inspect: right-click or middle-click whatever is hovered, or click
        # the zoom panel itself (U4 -- most trackpads have no middle button).
        if event.type == pygame.MOUSEBUTTONDOWN and event.button in (2, 3):
            self._update_hover(event.pos)  # the hover from the last frame may be a card the mouse has since left
            if self._hover_info is not None:
                self._open_inspector(self._hover_info)
            return
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.board.layout.zoom_rect().collidepoint(event.pos):
            if self._hover_info is not None or self._zoom_info is not None:
                self._open_inspector(self._zoom_info or self._hover_info)
            return

        self.log_panel.handle_event(event, self.board.layout.log_rect(), self.mouse)

        # Piles are public information: browsable at any time (U14).
        if (
            event.type == pygame.MOUSEBUTTONDOWN and event.button == 1
            and not self.panel.modal_open and not self.panel.full_screen
            and self._maybe_click_pile(event.pos)
        ):
            return

        if self.animator.is_busy:
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # Remember it: applied once the animation settles if the same
                # card is still under the cursor (M6).
                self._queued_click = (event.pos, self.board.click_target_at(event.pos), 0.0)
            return

        if self.bridge.is_over:
            return
        d = self.bridge.pending_decision
        if not self._is_my_decision(d):
            return
        mouse_pos = event.pos if hasattr(event, "pos") else self.mouse
        self.panel.handle_event(event, self.board, mouse_pos, self.board.layout)

    def _handle_key(self, event) -> bool:
        key = event.key
        if key == pygame.K_ESCAPE:
            if self.show_help:
                self.show_help = False
                return True
            if self.panel.chooser_iid is not None or self.panel.modal_open:
                return False  # the panel closes its own popups
            if self.animator.is_busy:
                return True
            if self.human_seats and not self.bridge.is_over and self._leave_armed_ms <= 0:
                # One stray Esc (e.g. meant for a popup that already closed)
                # used to throw the whole game away.
                self._leave_armed_ms = LEAVE_CONFIRM_MS
                rect = self.board.layout.action_bar_rect()
                self.board.toasts.append(
                    Toast(rect.centerx, rect.top - 40, "Press Esc again to leave this game (it's saved as unfinished)", life_ms=LEAVE_CONFIRM_MS)
                )
                return True
            self.app.pop()
            return True
        if key == pygame.K_SPACE and self.animator.is_busy:
            self.animator.skip()
            return True
        if key in (pygame.K_1, pygame.K_2, pygame.K_3):
            self.animator.speed = {pygame.K_1: 0.5, pygame.K_2: 1.0, pygame.K_3: 2.0}[key]
            return True
        if key in (pygame.K_SLASH, pygame.K_h) and not (event.mod & pygame.KMOD_CTRL):
            self.show_help = not self.show_help
            return True
        if key == pygame.K_d:
            self.decklist_pid = self.viewer
            self.browse_scroll = 0
            return True
        if key == pygame.K_r and not self.human_seats:
            self.reveal_hands = not self.reveal_hands
            self._sync_reveal_hands()
            return True
        if key == pygame.K_m:
            self.app.assets.muted = not self.app.assets.muted
            return True
        return False

    def _replay_queued_click(self) -> None:
        if self._queued_click is None:
            return
        pos, iid, age = self._queued_click
        self._queued_click = None
        if age > QUEUED_CLICK_MS or iid is None or self.board.click_target_at(pos) != iid:
            return
        if iid in self.panel.card_option_map and not self.panel.modal_open and not self.panel.full_screen:
            self.panel.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=pos, button=1), self.board, pos, self.board.layout)

    def _maybe_click_pile(self, pos) -> bool:
        L = self.board.layout
        ps = self.last_snapshot.players
        for pid in (1, 2):
            for kind in ("deck", "discard", "archive", "purged"):
                rect = L.pile_rect(pid, kind).inflate(8, 20)
                if not rect.collidepoint(pos):
                    continue
                count = getattr(ps[pid], f"{kind}_count")
                if count and self._pile_browsable(pid, kind):
                    self.browsing = (pid, kind)
                    self.browse_scroll = 0
                return True
        return False

    def _pile_browsable(self, pid: int, kind: str) -> bool:
        if kind == "deck":
            return False
        if kind == "archive":
            return pid == self.viewer and (self.human_seats or self.reveal_hands) or (not self.human_seats and self.reveal_hands)
        return True

    def _browser_rect(self) -> pygame.Rect:
        w, h = 800, 640
        return pygame.Rect((S.CANVAS_W - w) // 2, (S.CANVAS_H - h) // 2, w, h)

    def _decklist_rect(self) -> pygame.Rect:
        w, h = 850, 660
        return pygame.Rect((S.CANVAS_W - w) // 2, (S.CANVAS_H - h) // 2, w, h)

    def _handle_grid_event(self, event, rect, body_top, cards, close) -> None:
        if event.type == pygame.MOUSEWHEEL:
            self.browse_scroll = max(0, self.browse_scroll - event.y)
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            close()
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button in (1, 2, 3):
            card = self._card_grid_cell_at(rect, body_top, cards, event.pos)
            if card is not None:
                self._open_inspector(_card_info_from_engine_card(card))
            elif not rect.collidepoint(event.pos):
                close()

    def _handle_browser_event(self, event: pygame.event.Event) -> None:
        pid, kind = self.browsing
        cards = getattr(self.bridge.game.players[pid], kind).cards()

        def close():
            self.browsing = None

        self._handle_grid_event(event, self._browser_rect(), 44, cards, close)

    def _handle_decklist_event(self, event: pygame.event.Event) -> None:
        rect = self._decklist_rect()
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for tab_pid, tab in self._decklist_tabs(rect):
                if tab.collidepoint(event.pos):
                    self.decklist_pid, self.browse_scroll = tab_pid, 0
                    return
        cards = self._decklist_cards()

        def close():
            self.decklist_pid = None

        self._handle_grid_event(event, rect, 50, cards, close)

    def _decklist_tabs(self, rect):
        return [(1, pygame.Rect(rect.left + 16, rect.top + 10, 110, 28)), (2, pygame.Rect(rect.left + 132, rect.top + 10, 110, 28))]

    def _decklist_cards(self):
        return sorted(self.bridge.game.players[self.decklist_pid].all_cards, key=lambda c: (c.house.value, c.name))

    # -------------------------------------------------------- inspector ----

    def _open_inspector(self, info: SimpleNamespace) -> None:
        self.app.assets.play("click", 0.2)
        self.inspect_info = info

    def _close_inspector(self) -> None:
        self.inspect_info = None
        self._inspect_card_rect = None

    def _handle_inspector_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.MOUSEBUTTONDOWN or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            self._close_inspector()

    def _draw_inspector(self, surface: pygame.Surface) -> None:
        info = self.inspect_info
        self._dim_backdrop(surface, 190)
        card_h = min(S.INSPECT_MAX_H, S.CANVAS_H - 140)
        card_w = int(card_h * (S.CARD_ART_W / S.CARD_ART_H))
        panel_w = 380
        gap = 28
        total_w = card_w + gap + panel_w
        left = S.PLAY_X + (S.PLAY_W - total_w) // 2
        card_rect = pygame.Rect(left, 0, card_w, card_h)
        card_rect.centery = S.CANVAS_H // 2 - 30
        self._inspect_card_rect = card_rect
        surface.blit(self.app.assets.card_face(info.image, card_rect.size), card_rect)
        pygame.draw.rect(surface, S.AEMBER, card_rect.inflate(6, 6), width=2, border_radius=10)
        cx = card_rect.centerx
        name = self.app.assets.font("cinzel", 22).render(info.name, True, S.TEXT)
        surface.blit(name, name.get_rect(midtop=(cx, card_rect.bottom + 12)))
        sub_txt = f"{info.house} · {info.type}"
        if info.type == "Creature":
            sub_txt += f"   Power {info.power}" + (f"   Armor {info.armor}" if info.armor else "") + (f"   Damage {info.damage}" if info.damage else "")
        sub = self.app.assets.font("inter", 15).render(sub_txt, True, S.TEXT_DIM)
        surface.blit(sub, sub.get_rect(midtop=(cx, card_rect.bottom + 42)))
        hint = self.app.assets.font("inter", S.MIN_FONT).render("Click anywhere or press Esc to close", True, S.TEXT_FAINT)
        surface.blit(hint, hint.get_rect(midtop=(cx, card_rect.bottom + 66)))
        self._draw_inspector_text_panel(surface, info, card_rect, gap, panel_w)

    def _draw_inspector_text_panel(self, surface: pygame.Surface, info, card_rect: pygame.Rect, gap: int, panel_w: int) -> None:
        """The canonical rules text beside the card art, plus a note
        whenever it differs from what's printed on the card (errata cards,
        Safe Place, and the general Fight/Reap wording change)."""
        text_rect = pygame.Rect(card_rect.right + gap, card_rect.top, panel_w, card_rect.height)
        draw_panel(surface, text_rect, alpha=235, border=S.AEMBER)
        pad = 18
        ty = text_rect.top + pad
        header = self.app.assets.font("cinzel", 16).render("Official text", True, S.TEXT)
        surface.blit(header, (text_rect.left + pad, ty))
        ty += header.get_height() + 10

        body_font = self.app.assets.font("inter", 15)
        for line in _wrap_text(info.text or "(no ability text)", body_font, text_rect.width - pad * 2):
            img = body_font.render(line, True, S.TEXT)
            surface.blit(img, (text_rect.left + pad, ty))
            ty += img.get_height() + 4

        if info.errata:
            ty += 14
            badge_font = self.app.assets.font("inter", 13, bold=True)
            badge = badge_font.render("Differs from the printed card art:", True, S.AEMBER)
            surface.blit(badge, (text_rect.left + pad, ty))
            ty += badge.get_height() + 4
            note_font = self.app.assets.font("inter", 13)
            for line in _wrap_text(info.errata, note_font, text_rect.width - pad * 2):
                img = note_font.render(line, True, S.TEXT_DIM)
                surface.blit(img, (text_rect.left + pad, ty))
                ty += img.get_height() + 2

    def draw_crisp(self, window_surface: pygame.Surface) -> None:
        if self.inspect_info is None or self._inspect_card_rect is None:
            return
        win_rect = self.app.canvas_to_window_rect(self._inspect_card_rect)
        if win_rect.width <= 0 or win_rect.height <= 0:
            return
        window_surface.blit(self.app.assets.card_face(self.inspect_info.image, win_rect.size), win_rect)

    # --------------------------------------------------------------- draw ----

    def _forge_turns(self, pid: int) -> List[int]:
        return [e.data.get("turn") for e in self.bridge.game.log.events if e.kind == "forge_key" and e.data.get("player") == pid]

    def _draw_lanes(self, surface) -> None:
        """Faint panels behind each creature/artifact lane so the rows read
        as separate areas (L2)."""
        L = self.board.layout
        snap = self.last_snapshot
        font = self.app.assets.font("inter", 13)
        for pid in (1, 2):
            for zone, rect, empty in (
                ("play_creature", L.creature_row_rect(pid), "Creatures"),
                ("play_artifact", L.artifact_row_rect(pid), "Artifacts"),
            ):
                panel = pygame.Surface(rect.size, pygame.SRCALPHA)
                pygame.draw.rect(panel, (255, 255, 255, 10), panel.get_rect(), border_radius=10)
                pygame.draw.rect(panel, (255, 255, 255, 28), panel.get_rect(), width=1, border_radius=10)
                surface.blit(panel, rect)
                if not snap.zone_cards(pid, zone):
                    who = self.board.player_label(pid)
                    label = f"No {empty.lower()}" if self.board.spectating else (f"Your {empty.lower()}" if pid == self.viewer else f"Opponent's {empty.lower()}")
                    txt = font.render(label + (" — none in play" if not self.board.spectating else " in play"), True, S.TEXT_FAINT)
                    surface.blit(txt, txt.get_rect(center=rect.center))

    def _draw_prompt_status(self, surface) -> None:
        """Left of the prompt band: turn number and the first-turn rule status (U1)."""
        L = self.board.layout
        rect = L.prompt_rect()
        game = self.bridge.game
        font = self.app.assets.font("inter", 13, bold=True)
        turn = font.render(f"Turn {max(1, game.turn_number)}" if game.turn_number else "Setup", True, S.TEXT_DIM)
        surface.blit(turn, turn.get_rect(midleft=(rect.left + 8, rect.centery)))
        first = game.first_player
        if game.turn_number == 1 and game.active_player_id == first and not game.is_over:
            used = game.players[first].cards_played_or_discarded_this_turn >= 1
            text = "First turn: card limit used" if used else "First turn: 1 card to play or discard"
            chip_font = self.app.assets.font("inter", S.MIN_FONT, bold=True)
            t = chip_font.render(text, True, S.BLACK)
            r = t.get_rect().inflate(14, 6)
            r.midleft = (turn.get_rect(midleft=(rect.left + 8, rect.centery)).right + 12, rect.centery)
            pygame.draw.rect(surface, S.AEMBER_GLOW if not used else S.TEXT_DIM, r, border_radius=r.height // 2)
            surface.blit(t, t.get_rect(center=r.center))

    def draw(self, surface: pygame.Surface) -> None:
        _felt(surface)
        L = self.board.layout
        snap = self.last_snapshot
        self._tooltip_targets = []
        mouse = self.mouse

        for pid in (1, 2):
            for kind in ("deck", "discard", "archive", "purged"):
                rect = L.pile_rect(pid, kind)
                count = getattr(snap.players[pid], f"{kind}_count")
                browsable = self._pile_browsable(pid, kind)
                draw_pile(surface, self.app.assets, rect, kind, count, hovered=rect.collidepoint(mouse), browsable=browsable)

        self._draw_lanes(surface)
        game = self.bridge.game
        for pid in (1, 2):
            effects = [e for e in snap.active_effects if e["player_affected"] == pid]
            is_you = pid == self.viewer and pid in self.human_seats
            label = f"Player {pid}" + (" (You)" if is_you else "")
            tag = "First player" if game.turn_number <= 2 and pid == snap.first_player else ""
            rects = draw_hud(
                surface, self.app.assets, L.hud_rect(pid), snap.players[pid], self.board.hud_states[pid],
                snap.active_player == pid, label, effects, tag=tag, forge_turns=self._forge_turns(pid),
            )
            self.board.hud_rects[pid] = rects
            for kr, tip in rects.get("keys", []):
                self._tooltip_targets.append((kr, [tip]))
            for cr, lines in rects.get("chips", []):
                self._tooltip_targets.append((cr, lines))
            if "cost" in rects:
                ps = snap.players[pid]
                tip = f"Forging a key costs {ps.key_cost} Æmber" if ps.can_forge else "An effect prevents forging a key"
                self._tooltip_targets.append((rects["cost"], [tip]))

        self._draw_prompt_status(surface)

        self._sync_hidden()
        hidden = self.board.hidden_iids
        hovered_card = None if self._overlay_blocks_board() else self.board.card_at(mouse)
        for cs in sorted(snap.cards.values(), key=draw_key):
            if cs.iid in hidden:
                continue
            sprite = self.board.sprites.get(cs.iid)
            if sprite is not None:
                sprite.draw(surface)
                reason = sprite.unusable_reason or sprite.hint
                if reason and cs.iid == hovered_card:
                    self._tooltip_targets.append((sprite.rect(), [reason]))

        self.board.draw_overlays(surface)
        self.panel.draw(surface, self.app.assets, L, mouse)
        self._overlay_targets = list(self.panel.hover_targets)
        self._draw_zoom(surface, L)
        self._log_targets = self.log_panel.draw(surface, self.app.assets, L.log_rect(), self.log_sentences, mouse)
        self._draw_action_hints(surface)

        if self.browsing is not None:
            self._draw_browser(surface)
        if self.decklist_pid is not None:
            self._draw_decklist(surface)
        if self.show_help:
            self._draw_help(surface)
        if self.inspect_info is not None:
            self._draw_inspector(surface)
        else:
            self._draw_tooltip(surface, mouse)

    def _draw_action_hints(self, surface) -> None:
        rect = self.board.layout.action_bar_rect()
        mute_txt = "M unmute" if self.app.assets.muted else "M mute"
        text = f"Right-click a card to read it · D decklists · H help · {mute_txt}" + ("  · R reveal hands" if not self.human_seats else "")
        hint = self.app.assets.font("inter", S.MIN_FONT).render(text, True, S.TEXT_FAINT)
        surface.blit(hint, hint.get_rect(midleft=(rect.left + 4, rect.centery)))

    def _draw_tooltip(self, surface, mouse) -> None:
        lines = None
        for rect, tip_lines in self._tooltip_targets:
            if rect.collidepoint(mouse):
                lines = tip_lines
        if not lines:
            return
        font = self.app.assets.font("inter", 13)
        imgs = [font.render(line, True, S.TEXT) for line in lines]
        w = max(i.get_width() for i in imgs) + 20
        h = sum(i.get_height() + 2 for i in imgs) + 12
        box = pygame.Rect(0, 0, w, h)
        box.topleft = (mouse[0] + 16, mouse[1] + 18)
        if box.right > S.CANVAS_W - 6:
            box.right = mouse[0] - 10
        if box.bottom > S.CANVAS_H - 6:
            box.bottom = mouse[1] - 10
        draw_panel(surface, box, alpha=245, border=S.TEXT_FAINT, radius=6)
        y = box.top + 6
        for img in imgs:
            surface.blit(img, (box.left + 10, y))
            y += img.get_height() + 2

    @property
    def _zoom_info(self) -> Optional[SimpleNamespace]:
        return getattr(self, "_zoom_info_cache", None)

    def _draw_zoom(self, surface, L) -> None:
        rect = L.zoom_rect()
        draw_panel(surface, rect, alpha=190)
        info = self._hover_info
        self._zoom_info_cache = info
        title = self.app.assets.font("cinzel", 15).render("Card", True, S.TEXT_DIM)
        surface.blit(title, (rect.left + 10, rect.top + 8))
        if info is None:
            font = self.app.assets.font("inter", 13)
            for i, line in enumerate(("Hover any card to see it here.", "Right-click (or click here) to", "open it full size.")):
                t = font.render(line, True, S.TEXT_FAINT)
                surface.blit(t, t.get_rect(center=(rect.centerx, rect.centery - 20 + i * 20)))
            return
        img_w = rect.width - 20
        img_h = int(img_w * (S.CARD_ART_H / S.CARD_ART_W))
        img = self.app.assets.card_face(info.image, (img_w, img_h))
        img_rect = img.get_rect(midtop=(rect.centerx, rect.top + 32))
        surface.blit(img, img_rect)
        info_font = self.app.assets.font("inter", 13)
        y = img_rect.bottom + 6
        stats = f"{info.house} · {info.type}"
        if info.type == "Creature":
            stats += f" · Power {info.power}" + (f" · Damage {info.damage}" if info.damage else "")
        for line in (info.name, stats):
            t = info_font.render(line, True, S.TEXT)
            surface.blit(t, (rect.left + 10, y))
            y += t.get_height() + 2

    def _dim_backdrop(self, surface, alpha=150) -> None:
        dim = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        dim.fill((0, 0, 0, alpha))
        surface.blit(dim, (0, 0))

    _GRID_STRIDE_X = S.BROWSER_CARD_W + 16
    _GRID_STRIDE_Y = S.BROWSER_CARD_H + 34

    def _card_grid_body(self, rect: pygame.Rect, body_top: int) -> pygame.Rect:
        return pygame.Rect(rect.left + 14, rect.top + body_top, rect.width - 28, rect.bottom - rect.top - body_top - 22)

    def _card_grid_cells(self, rect, body_top, cards):
        body = self._card_grid_body(rect, body_top)
        cols = max(1, body.width // self._GRID_STRIDE_X)
        rows_total = (len(cards) + cols - 1) // cols
        rows_visible = max(1, body.height // self._GRID_STRIDE_Y)
        max_scroll = max(0, rows_total - rows_visible)
        self.browse_scroll = max(0, min(self.browse_scroll, max_scroll))
        cells = []
        for i, card in enumerate(cards):
            gx, gy = i % cols, i // cols - self.browse_scroll
            cell = pygame.Rect(body.left + gx * self._GRID_STRIDE_X, body.top + gy * self._GRID_STRIDE_Y, S.BROWSER_CARD_W, S.BROWSER_CARD_H)
            if body.top <= cell.top and cell.bottom <= body.bottom:
                cells.append((cell, card))
        return body, cells, max_scroll

    def _card_grid_cell_at(self, rect: pygame.Rect, body_top: int, cards, pos):
        _body, cells, _ = self._card_grid_cells(rect, body_top, cards)
        for cell, card in cells:
            if cell.collidepoint(pos):
                return card
        return None

    def _draw_card_grid(self, surface, rect: pygame.Rect, body_top: int, cards) -> None:
        body, cells, max_scroll = self._card_grid_cells(rect, body_top, cards)
        name_font = self.app.assets.font("inter", S.MIN_FONT)
        mouse = self.mouse
        for cell, card in cells:
            hovered = cell.collidepoint(mouse)
            surface.blit(self.app.assets.card_face(card.image, cell.size), cell)
            self._overlay_targets.append((cell, card.instance_id))
            if hovered:
                pygame.draw.rect(surface, S.AEMBER, cell, width=2, border_radius=6)
            name = name_font.render(card.name, True, S.TEXT if hovered else S.TEXT_DIM)
            if name.get_width() > cell.width:
                name = pygame.transform.smoothscale(name, (cell.width, name.get_height()))
            surface.blit(name, (cell.left, cell.bottom + 3))
        hint_font = self.app.assets.font("inter", S.MIN_FONT)
        text = "Click a card to read it full size · Esc to close"
        if max_scroll > 0:
            text = f"Scroll for more (row {self.browse_scroll + 1} of {max_scroll + 1}) · " + text
            track = pygame.Rect(rect.right - 12, body.top, 5, body.height)
            pygame.draw.rect(surface, S.PANEL_LIGHT, track, border_radius=3)
            thumb_h = max(24, body.height // (max_scroll + 1))
            thumb_y = track.top + int((body.height - thumb_h) * self.browse_scroll / max_scroll)
            pygame.draw.rect(surface, S.AEMBER, pygame.Rect(track.left, thumb_y, 5, thumb_h), border_radius=3)
        surface.blit(hint_font.render(text, True, S.TEXT_FAINT), (body.left, rect.bottom - 20))

    def _draw_browser(self, surface) -> None:
        pid, kind = self.browsing
        cards = getattr(self.bridge.game.players[pid], kind).cards()
        self._dim_backdrop(surface)
        rect = self._browser_rect()
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
        cards = self._decklist_cards()
        self._dim_backdrop(surface)
        rect = self._decklist_rect()
        draw_panel(surface, rect, alpha=245, border=S.AEMBER)
        title = self.app.assets.font("cinzel", 18).render(f"Decklist ({len(cards)} cards, unordered)", True, S.TEXT)
        surface.blit(title, (rect.left + 260, rect.top + 12))
        for tab_pid, tab in self._decklist_tabs(rect):
            active = tab_pid == pid
            pygame.draw.rect(surface, S.AEMBER if active else S.PANEL_LIGHT, tab, border_radius=6)
            pygame.draw.rect(surface, S.AEMBER, tab, width=1, border_radius=6)
            label = self.app.assets.font("inter", 13, bold=True).render(f"Player {tab_pid}", True, S.BLACK if active else S.TEXT)
            surface.blit(label, label.get_rect(center=tab.center))
        self._draw_card_grid(surface, rect, 50, cards)

    def _help_lines(self):
        lines = [
            ("Click a glowing card", "play / discard / reap / fight / use it"),
            ("Dimmed card", "can't be used right now — hover it to see why"),
            ("O", "the full list of legal choices"),
            ("Right-click a card", "read it full size (also: click the zoom panel)"),
            ("Hover a card or log line", "show it in the zoom panel"),
            ("Click a pile", "browse discard, purged, or your archive — any time"),
            ("D", "view either player's full 36-card decklist"),
            ("M", "mute / unmute sound"),
            ("Space", "skip the current animation"),
            ("1 / 2 / 3", "animation speed 0.5x / 1x / 2x"),
            ("F11 / F3", "fullscreen / show FPS"),
            ("Esc", "close a popup, or leave the game"),
        ]
        if not self.human_seats:
            lines.insert(6, ("R", "spectating: reveal both hands"))
        return lines

    def _draw_help(self, surface) -> None:
        self._dim_backdrop(surface)
        lines = self._help_lines()
        w, h = 680, 80 + 30 * len(lines)
        rect = pygame.Rect((S.CANVAS_W - w) // 2, (S.CANVAS_H - h) // 2, w, h)
        draw_panel(surface, rect, alpha=250, border=S.AEMBER)
        surface.blit(self.app.assets.font("cinzel", 20).render("Controls", True, S.TEXT), (rect.left + 18, rect.top + 14))
        y = rect.top + 56
        label_font = self.app.assets.font("inter", 14, bold=True)
        desc_font = self.app.assets.font("inter", 14)
        for key, desc in lines:
            surface.blit(label_font.render(key, True, S.AEMBER), (rect.left + 20, y))
            surface.blit(desc_font.render(desc, True, S.TEXT_DIM), (rect.left + 250, y))
            y += 30


def _game_over_cls():
    from .game_over_scene import GameOverScene

    return GameOverScene
