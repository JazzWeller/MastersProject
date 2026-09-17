"""DecisionPanel: the one piece of UI that answers every `Decision` the
engine can produce for a human seat.

Two complementary interaction paths, both always available:
  1. Direct manipulation — options that reference a `Card` sitting somewhere
     clearly clickable (hand, creature row, artifact row) make that card's
     sprite glow; clicking it picks the option (or, if a card has more than
     one legal option, opens a small scoped menu).
  2. The **Options** panel (bound to the O key and a button, and opened
     automatically for decisions with nothing clickable on the board, e.g.
     MULLIGAN or CHOOSE_HOUSE) lists every current option as a row — a card
     thumbnail grid for Card options sitting in a pile, plain labeled
     buttons otherwise. This is also the guaranteed fallback: it can answer
     *any* decision, so there's always a way forward.
Both paths feed the same `self.picked` accumulator, so multi-select
(CHOOSE_CARDS with max_n > 1) and ordering (ORDER_EFFECTS) work the same
way regardless of which path was used to make each pick.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pygame

from .. import settings as S
from ..option_labels import describe_option, describe_option_short
from ..sprites.widgets import Button, draw_panel

from keyforge.actions import DiscardCard, EndTurn
from keyforge.cards.card import Card
from keyforge.effects.effect_object import TriggerEffect
from keyforge.enums import DecisionKind

_ON_BOARD_ZONES = {"hand", "play_creature", "play_artifact"}


def _option_card(opt: Any) -> Optional[Card]:
    if isinstance(opt, Card):
        return opt
    c = getattr(opt, "card", None)
    return c if isinstance(c, Card) else None


class OptionRow:
    __slots__ = ("option", "label", "card")

    def __init__(self, option, label, card):
        self.option = option
        self.label = label
        self.card = card


class DecisionPanel:
    def __init__(self):
        self.decision = None
        self.view = None
        self.card_option_map: Dict[int, List[Any]] = {}
        self.picked: List[Any] = []
        self.result: Optional[Any] = None

        self.modal_open = False
        self.modal_rows: List[OptionRow] = []
        self.modal_scroll = 0

        self._scoped_card_iid: Optional[int] = None  # set when the modal was opened by clicking one card
        self._board = None  # set by on_enter; used only to play a click sound on picks

        # The action chooser: a compact popup anchored beside one clicked
        # card, showing that card *once* plus a labeled button per option
        # ("Play" / "Discard" / ...). This is what a card with more than one
        # legal option opens now, instead of the generic Options modal (which
        # used to render one identical thumbnail per option -- see
        # UX_FIX_PLAN.md F3/A3).
        self.chooser_iid: Optional[int] = None
        self.chooser_opts: List[Any] = []

        # MULLIGAN gets its own full-hand review screen (B3) instead of the
        # generic Yes/No modal, which used to dim the board over a hand the
        # player couldn't actually see or hover. See UX_FIX_PLAN.md F6/B3.
        self.mulligan_open = False

        # A one-click misfire guard (D7): discarding a card, and ending your
        # turn while other legal actions still exist, both arm instead of
        # submitting immediately -- the same click again (or the button
        # re-clicked) confirms it.
        self._armed_action: Any = None

    # --------------------------------------------------------------- setup ----

    def on_enter(self, decision, view, board) -> None:
        self.decision = decision
        self.view = view
        self.picked = []
        self.result = None
        self._scoped_card_iid = None
        self.card_option_map = {}
        self._board = board
        self.chooser_iid = None
        self.chooser_opts = []
        self.mulligan_open = False
        self._armed_action = None

        for opt in decision.options:
            card = _option_card(opt)
            if card is None:
                continue
            cs = board.snapshot.cards.get(card.instance_id) if board.snapshot else None
            if cs is not None and cs.zone in _ON_BOARD_ZONES:
                self.card_option_map.setdefault(card.instance_id, []).append(opt)

        if decision.kind == DecisionKind.MULLIGAN:
            self.mulligan_open = True
            self.modal_open = False
            self.modal_rows = []
            return

        has_clickable = bool(self.card_option_map)
        self.modal_open = not has_clickable
        if self.modal_open:
            self._open_modal(decision.options)
        else:
            self.modal_rows = []

    def _open_modal(self, options, scoped_iid: Optional[int] = None) -> None:
        self._close_chooser()
        self.mulligan_open = False
        self.modal_rows = [OptionRow(o, describe_option(o, self.decision, self.view), _option_card(o)) for o in options]
        self.modal_scroll = 0
        self.modal_open = True
        self._scoped_card_iid = scoped_iid

    def _close_modal(self) -> None:
        self.modal_open = False
        self._scoped_card_iid = None

    # ------------------------------------------------------------- picking ----

    def _submit(self, choice) -> None:
        self.result = choice

    def _finish_value(self, picked_list: List[Any]):
        d = self.decision
        if d.kind in (DecisionKind.CHOOSE_CARDS, DecisionKind.ORDER_EFFECTS):
            return list(picked_list)
        return picked_list[0]

    def _needs_confirm(self, opt: Any) -> bool:
        if isinstance(opt, DiscardCard):
            return True
        if isinstance(opt, EndTurn) and self.decision is not None and len(self.decision.options) > 1:
            return True
        return False

    def _pick(self, opt: Any) -> None:
        if opt is not self._armed_action and self._needs_confirm(opt):
            self._armed_action = opt
            if self._board is not None:
                self._board.assets.play("click", 0.15)
            return
        self._armed_action = None
        if self._board is not None:
            self._board.assets.play("click", 0.3)
        d = self.decision
        if d.kind == DecisionKind.ORDER_EFFECTS:
            if opt in self.picked:
                return
            self.picked.append(opt)
            if len(self.picked) >= len(d.options):
                self._submit(self._finish_value(self.picked))
            return

        if d.max_n <= 1:
            self._submit(self._finish_value([opt]))
            return

        # multi-select: toggle
        if opt in self.picked:
            self.picked = [p for p in self.picked if p is not opt]
            return
        if len(self.picked) >= d.max_n:
            return
        self.picked.append(opt)
        if len(self.picked) == d.max_n:
            self._submit(self._finish_value(self.picked))

    def confirm(self) -> None:
        d = self.decision
        if d is not None and d.min_n <= len(self.picked) <= d.max_n:
            self._submit(self._finish_value(self.picked))

    def choose_none(self) -> None:
        if self.decision is not None and self.decision.min_n == 0:
            self._submit(self._finish_value([]))

    # ------------------------------------------------------------ board ui ----

    def sync_glow(self, board) -> None:
        for iid, sprite in board.sprites.items():
            if sprite.glow == "legal":
                sprite.glow = None
        for iid in self.card_option_map:
            sprite = board.sprites.get(iid)
            if sprite is None:
                continue
            sprite.glow = "selected" if any(o in self.picked for o in self.card_option_map[iid]) else "legal"

    def click_board(self, mx: float, my: float, board) -> bool:
        """Returns True if the click was consumed (hit a clickable card)."""
        for iid in sorted(self.card_option_map, key=lambda i: -board.sprites[i].scale if i in board.sprites else 0):
            sprite = board.sprites.get(iid)
            if sprite is None or not sprite.visible:
                continue
            if sprite.contains_point(mx, my):
                opts = self.card_option_map[iid]
                if len(opts) == 1:
                    self._pick(opts[0])
                else:
                    self._open_chooser(iid, opts)
                return True
        return False

    # ------------------------------------------------------ action chooser ----

    def _open_chooser(self, iid: int, opts: List[Any]) -> None:
        self.chooser_iid = iid
        self.chooser_opts = opts

    def _close_chooser(self) -> None:
        self.chooser_iid = None
        self.chooser_opts = []

    def _chooser_rect(self, board) -> pygame.Rect:
        w, h = 300, 56 + 42 * len(self.chooser_opts)
        sprite = board.sprites.get(self.chooser_iid)
        cx, cy = (sprite.x, sprite.y) if sprite is not None else (S.PLAY_X + S.PLAY_W // 2, S.CANVAS_H // 2)
        # Anchor beside the card: to the right if there's room, else the
        # left; vertically centered on it but clamped fully on-canvas.
        x = cx + 70 if cx + 70 + w < S.BOARD_W - 10 else cx - 70 - w
        x = max(S.PLAY_X, min(x, S.PLAY_X + S.PLAY_W - w))
        y = max(10, min(cy - h // 2, S.CANVAS_H - h - 10))
        return pygame.Rect(int(x), int(y), w, h)

    def _chooser_option_rows(self, rect: pygame.Rect) -> List[pygame.Rect]:
        rows = []
        y = rect.top + 56
        for _ in self.chooser_opts:
            rows.append(pygame.Rect(rect.left + 88, y, rect.width - 88 - 14, 34))
            y += 42
        return rows

    def _handle_chooser_click(self, mouse_pos, board) -> None:
        rect = self._chooser_rect(board)
        if not rect.collidepoint(mouse_pos):
            self._close_chooser()
            return
        for row, opt in zip(self._chooser_option_rows(rect), self.chooser_opts):
            if row.collidepoint(mouse_pos):
                self._pick(opt)
                if self._armed_action is not opt:
                    # Not just armed for confirmation -- either it submitted
                    # outright, or it never needed confirming.
                    self._close_chooser()
                return

    def _draw_chooser(self, surface: pygame.Surface, assets, board, mouse_pos=(-1, -1)) -> None:
        rect = self._chooser_rect(board)
        draw_panel(surface, rect, alpha=250, border=S.AEMBER)

        sprite = board.sprites.get(self.chooser_iid)
        cs = sprite.card_state if sprite is not None else None
        card_w, card_h = 64, 90
        card_rect = pygame.Rect(rect.left + 12, rect.top + 12, card_w, card_h)
        if cs is not None and cs.image:
            img = assets.card_face(cs.image, (card_w, card_h))
            surface.blit(img, card_rect)
        name_font = assets.font("inter", 12, bold=True)
        name = cs.name if cs is not None else "Card"
        words = name.split()
        line = ""
        y = card_rect.bottom + 4
        for w in words:
            trial = f"{line} {w}".strip()
            if name_font.size(trial)[0] > card_w and line:
                t = name_font.render(line, True, S.TEXT_DIM)
                surface.blit(t, (card_rect.left, y))
                y += t.get_height()
                line = w
            else:
                line = trial
        if line:
            t = name_font.render(line, True, S.TEXT_DIM)
            surface.blit(t, (card_rect.left, y))

        label_font = assets.font("inter", 14, bold=True)
        for row, opt in zip(self._chooser_option_rows(rect), self.chooser_opts):
            armed = opt is self._armed_action
            hovered = row.collidepoint(mouse_pos)
            if armed:
                bg, border, fg = S.DANGER, S.DANGER, S.WHITE
            elif hovered:
                bg, border, fg = S.AEMBER, S.AEMBER, S.BLACK
            else:
                bg, border, fg = S.PANEL_LIGHT, S.AEMBER, S.TEXT
            pygame.draw.rect(surface, bg, row, border_radius=6)
            pygame.draw.rect(surface, border, row, width=1, border_radius=6)
            label = "Confirm?" if armed else describe_option_short(opt)
            img = label_font.render(label, True, fg)
            surface.blit(img, img.get_rect(center=row.center))

        hint_font = assets.font("inter", 11)
        hint = hint_font.render("Esc to cancel", True, S.TEXT_FAINT)
        surface.blit(hint, (rect.left + 12, rect.bottom - 16))

    # ---------------------------------------------------------- mulligan ----

    def _mulligan_hand(self, board) -> List:
        if board.snapshot is None:
            return []
        return board.snapshot.zone_cards(self.decision.player, "hand")

    def _mulligan_card_rects(self, board, layout) -> List[tuple]:
        hand = self._mulligan_hand(board)
        row = pygame.Rect(S.PLAY_X, 300, S.PLAY_W, S.MULLIGAN_CARD_H)
        slots = layout.row_slots(len(hand), row, S.MULLIGAN_CARD_W, S.MULLIGAN_CARD_H)
        out = []
        for cs, (x, y) in zip(hand, slots):
            rect = pygame.Rect(0, 0, S.MULLIGAN_CARD_W, S.MULLIGAN_CARD_H)
            rect.center = (int(x), int(y))
            out.append((rect, cs))
        return out

    def _mulligan_buttons(self) -> List[tuple]:
        n = len(self._mulligan_hand(self._board)) if self._board is not None else 0
        mull_opt = next((o for o in self.decision.options if o is True), None)
        keep_opt = next((o for o in self.decision.options if o is False), None)
        y = 620
        w, h = 260, 50
        cx = S.PLAY_X + S.PLAY_W // 2
        keep_rect = pygame.Rect(cx - w - 12, y, w, h)
        mull_rect = pygame.Rect(cx + 12, y, w, h)
        keep = Button(keep_rect, "Keep This Hand", primary=True)
        mull = Button(mull_rect, f"Mulligan (draw {max(0, n - 1)})", danger=True)
        return [(keep, keep_opt), (mull, mull_opt)]

    def _handle_mulligan_click(self, mouse_pos, board, layout) -> None:
        for button, opt in self._mulligan_buttons():
            button.update_hover(mouse_pos)
            fake_up = pygame.event.Event(pygame.MOUSEBUTTONUP, pos=mouse_pos, button=1)
            if button.clicked(fake_up) and opt is not None:
                self._pick(opt)
                return

    def _draw_mulligan(self, surface, assets, board, layout, mouse_pos=(-1, -1)) -> None:
        dim = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 215))
        surface.blit(dim, (0, 0))

        hand = self._mulligan_hand(board)
        title_font = assets.font("cinzel", 24)
        title = title_font.render("Mulligan your hand?", True, S.TEXT)
        surface.blit(title, title.get_rect(center=(S.PLAY_X + S.PLAY_W // 2, 60)))

        sub_font = assets.font("inter", 14)
        houses: Dict[str, int] = {}
        for cs in hand:
            houses[cs.house] = houses.get(cs.house, 0) + 1
        summary = " · ".join(f"{n} {h}" for h, n in sorted(houses.items()))
        sub = sub_font.render(
            f"{len(hand)} cards" + (f"  ({summary})" if summary else ""), True, S.TEXT_DIM
        )
        surface.blit(sub, sub.get_rect(center=(S.PLAY_X + S.PLAY_W // 2, 92)))

        for rect, cs in self._mulligan_card_rects(board, layout):
            hovered = rect.collidepoint(mouse_pos)
            if cs.face_up and cs.image:
                img = assets.card_face(cs.image, (rect.width, rect.height))
            else:
                img = assets.card_back((rect.width, rect.height))
            if hovered:
                glow = rect.inflate(10, 10)
                pygame.draw.rect(surface, S.AEMBER, glow, width=3, border_radius=10)
            surface.blit(img, rect)
            name_font = assets.font("inter", 12, bold=True)
            name = name_font.render(cs.name, True, S.TEXT)
            surface.blit(name, name.get_rect(midtop=(rect.centerx, rect.bottom + 4)))

        for button, _opt in self._mulligan_buttons():
            button.update_hover(mouse_pos)
            button.draw(surface, assets)

        hint = sub_font.render(
            "A mulligan shuffles your hand back in and draws one fewer card.  (O for a plain list)",
            True, S.TEXT_FAINT,
        )
        surface.blit(hint, hint.get_rect(center=(S.PLAY_X + S.PLAY_W // 2, 690)))

    # ---------------------------------------------------------- action bar ----

    def _end_turn_option(self):
        if self.decision is not None and self.decision.kind == DecisionKind.CHOOSE_ACTION:
            for opt in self.decision.options:
                if isinstance(opt, EndTurn):
                    return opt
        return None

    def action_bar_buttons(self, layout) -> List[tuple]:
        """[(Button, callback)] for whatever's relevant to the current
        decision. Rebuilt on demand (cheap) so draw() and handle_event()
        always agree on hit boxes."""
        if self.decision is None:
            return []
        rect = layout.action_bar_rect()
        buttons = []
        x = rect.right
        h = rect.height

        def add_right(text, cb, primary=False, danger=False, enabled=True, w=132):
            nonlocal x
            x -= w + 8
            b = Button(pygame.Rect(x, rect.top, w, h), text, enabled=enabled, primary=primary, danger=danger)
            buttons.append((b, cb))

        end_turn = self._end_turn_option()
        if end_turn is not None:
            armed = end_turn is self._armed_action
            label = "Confirm End Turn?" if armed else "End Turn"
            add_right(label, lambda: self._pick(end_turn), primary=not armed, danger=armed, w=150 if armed else 132)

        d = self.decision
        if d.kind != DecisionKind.ORDER_EFFECTS and d.max_n > 1:
            can_confirm = d.min_n <= len(self.picked) <= d.max_n
            add_right("Confirm", self.confirm, primary=True, enabled=can_confirm)
        if d.min_n == 0 and not self.picked:
            add_right("Choose none", self.choose_none)

        add_right("Options (O)", lambda: self._open_modal(self.decision.options), w=118)
        return buttons

    # ------------------------------------------------------------- events ----

    def handle_event(self, event: pygame.event.Event, board, mouse_pos, layout) -> Optional[Any]:
        if self.decision is None:
            return None

        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE and self.chooser_iid is not None:
            self._close_chooser()
            return self.result

        if event.type == pygame.KEYDOWN and event.key in (pygame.K_F1, pygame.K_o):
            self._close_chooser()
            if self.modal_open and self._scoped_card_iid is None:
                self._close_modal()
            else:
                self._open_modal(self.decision.options)
            return self.result

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for button, cb in self.action_bar_buttons(layout):
                button.update_hover(mouse_pos)
            fake_up = pygame.event.Event(pygame.MOUSEBUTTONUP, pos=mouse_pos, button=1)
            for button, cb in self.action_bar_buttons(layout):
                if button.clicked(fake_up):
                    cb()
                    return self.result
            if self.mulligan_open:
                self._handle_mulligan_click(mouse_pos, board, layout)
            elif self.chooser_iid is not None:
                self._handle_chooser_click(mouse_pos, board)
            elif self.modal_open:
                self._handle_modal_click(mouse_pos)
            else:
                self.click_board(*mouse_pos, board)
        elif event.type == pygame.MOUSEWHEEL and self.modal_open:
            self.modal_scroll = max(0, self.modal_scroll - event.y)

        return self.result

    def _handle_modal_click(self, mouse_pos) -> None:
        rect = self._modal_rect()
        if not rect.collidepoint(mouse_pos):
            if self._scoped_card_iid is not None:
                self._close_modal()
            return
        row_h = 40
        grid = self._modal_is_grid()
        body = pygame.Rect(rect.left + 14, rect.top + 40, rect.width - 28, rect.height - 80)
        if grid:
            cols = max(1, body.width // 106)
            for i, row in enumerate(self.modal_rows):
                gx = i % cols
                gy = i // cols - self.modal_scroll
                cell = pygame.Rect(body.left + gx * 106, body.top + gy * 150, 96, 140)
                if cell.collidepoint(mouse_pos):
                    self._pick(row.option)
                    if self._armed_action is row.option:
                        return
                    if self.decision.max_n <= 1 or self.decision.kind == DecisionKind.ORDER_EFFECTS:
                        self._close_modal()
                    return
        else:
            for i, row in enumerate(self.modal_rows):
                y = body.top + (i - self.modal_scroll) * (row_h + 6)
                cell = pygame.Rect(body.left, y, body.width, row_h)
                if cell.collidepoint(mouse_pos):
                    self._pick(row.option)
                    if self._armed_action is row.option:
                        return
                    if self.decision.max_n <= 1 or self.decision.kind == DecisionKind.ORDER_EFFECTS:
                        self._close_modal()
                    return

    # --------------------------------------------------------------- draw ----

    def _modal_rect(self) -> pygame.Rect:
        """Sized to its content rather than a fixed 620x520: a 3-option
        house pick shouldn't black out the whole board the way a 30-card
        pile browse legitimately does. See UX_FIX_PLAN.md D1."""
        grid = self._modal_is_grid()
        n = len(self.modal_rows)
        if grid:
            w = 620
            cols = max(1, (w - 28) // 106)
            rows = max(1, -(-n // cols)) if n else 1
            h = min(560, 80 + rows * 150)
        else:
            w = 460
            h = min(560, 80 + max(1, n) * 46)
        h = max(150, h)
        x = S.PLAY_X + (S.PLAY_W - w) // 2
        y = max(20, (S.CANVAS_H - h) // 2)
        return pygame.Rect(x, y, w, h)

    def _modal_is_grid(self) -> bool:
        """A card-thumbnail grid only makes sense when every row is a
        *distinct* physical card -- otherwise (e.g. Play/Discard on the same
        card, reached via O instead of a card-click) it renders the same
        card's art twice with no way to tell the rows apart. See
        UX_FIX_PLAN.md F3."""
        if not self.modal_rows or any(r.card is None for r in self.modal_rows):
            return False
        seen = set()
        for r in self.modal_rows:
            if r.card.instance_id in seen:
                return False
            seen.add(r.card.instance_id)
        return True

    def draw(self, surface: pygame.Surface, assets, layout, mouse_pos=(-1, -1)) -> None:
        if self.decision is None:
            return
        if self.mulligan_open:
            self._draw_mulligan(surface, assets, self._board, layout, mouse_pos)
            return
        if not self.modal_open:
            self._draw_prompt(surface, assets, layout)
        for button, _cb in self.action_bar_buttons(layout):
            button.update_hover(mouse_pos)
            button.draw(surface, assets)
        if self.chooser_iid is not None:
            self._draw_chooser(surface, assets, self._board, mouse_pos)
        elif self.modal_open:
            self._draw_modal(surface, assets)

    def _draw_prompt(self, surface, assets, layout) -> None:
        rect = layout.prompt_rect()
        font = assets.font("inter", 18, bold=True)
        img = font.render(self.decision.prompt, True, S.TEXT)
        surface.blit(img, img.get_rect(center=rect.center))
        hint_font = assets.font("inter", 12)
        hint = hint_font.render("Press O for options", True, S.TEXT_FAINT)
        surface.blit(hint, (rect.right - hint.get_width(), rect.bottom - 2))

    def _draw_modal(self, surface, assets) -> None:
        rect = self._modal_rect()
        grid = self._modal_is_grid()
        dim = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 140 if grid else 80))
        surface.blit(dim, (0, 0))
        draw_panel(surface, rect, alpha=245, border=S.AEMBER)

        title_font = assets.font("cinzel", 18)
        title = title_font.render(self.decision.prompt, True, S.TEXT)
        surface.blit(title, (rect.left + 16, rect.top + 10))

        body = pygame.Rect(rect.left + 14, rect.top + 40, rect.width - 28, rect.height - 80)
        clip = surface.get_clip()
        surface.set_clip(body)

        if grid:
            cols = max(1, body.width // 106)
            for i, row in enumerate(self.modal_rows):
                gx = i % cols
                gy = i // cols - self.modal_scroll
                cell = pygame.Rect(body.left + gx * 106, body.top + gy * 150, 96, 140)
                if cell.bottom < body.top or cell.top > body.bottom:
                    continue
                picked = row.option in self.picked
                armed = row.option is self._armed_action
                img = assets.card_face(row.card.image, (96, 134))
                surface.blit(img, cell)
                if armed:
                    pygame.draw.rect(surface, S.DANGER, cell, width=3, border_radius=6)
                    tag_font = assets.font("inter", 11, bold=True)
                    tag = tag_font.render("Confirm?", True, S.WHITE)
                    tag_bg = tag.get_rect(center=cell.center).inflate(10, 6)
                    pygame.draw.rect(surface, S.DANGER, tag_bg, border_radius=4)
                    surface.blit(tag, tag.get_rect(center=cell.center))
                elif picked:
                    pygame.draw.rect(surface, S.GLOW_SELECTED, cell, width=3, border_radius=6)
                else:
                    pygame.draw.rect(surface, S.TEXT_FAINT, cell, width=1, border_radius=6)
        else:
            row_h = 40
            for i, row in enumerate(self.modal_rows):
                y = body.top + (i - self.modal_scroll) * (row_h + 6)
                if y + row_h < body.top or y > body.bottom:
                    continue
                cell = pygame.Rect(body.left, y, body.width, row_h)
                picked = row.option in self.picked
                armed = row.option is self._armed_action
                bg = S.DANGER if armed else (S.PANEL_LIGHT if picked else S.PANEL)
                pygame.draw.rect(surface, bg, cell, border_radius=6)
                pygame.draw.rect(surface, S.DANGER if armed else (S.GLOW_SELECTED if picked else S.TEXT_FAINT), cell, width=1, border_radius=6)
                font = assets.font("inter", 15)
                label = f"Confirm? ({row.label})" if armed else row.label
                img = font.render(label, True, S.WHITE if armed else S.TEXT)
                surface.blit(img, (cell.left + 10, cell.centery - img.get_height() // 2))

        surface.set_clip(clip)

        footer_font = assets.font("inter", 12)
        if self.decision.kind == DecisionKind.ORDER_EFFECTS:
            note = f"Click in the order they should resolve ({len(self.picked)}/{len(self.decision.options)})"
        elif self.decision.max_n > 1:
            note = f"Selected {len(self.picked)} (need {self.decision.min_n}-{self.decision.max_n})"
        else:
            note = ""
        if note:
            img = footer_font.render(note, True, S.TEXT_DIM)
            surface.blit(img, (rect.left + 16, rect.bottom - 30))
