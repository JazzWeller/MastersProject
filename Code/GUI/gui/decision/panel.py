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
from ..option_labels import describe_option
from ..sprites.widgets import Button, draw_panel

from keyforge.actions import EndTurn
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

    # --------------------------------------------------------------- setup ----

    def on_enter(self, decision, view, board) -> None:
        self.decision = decision
        self.view = view
        self.picked = []
        self.result = None
        self._scoped_card_iid = None
        self.card_option_map = {}
        self._board = board

        for opt in decision.options:
            card = _option_card(opt)
            if card is None:
                continue
            cs = board.snapshot.cards.get(card.instance_id) if board.snapshot else None
            if cs is not None and cs.zone in _ON_BOARD_ZONES:
                self.card_option_map.setdefault(card.instance_id, []).append(opt)

        has_clickable = bool(self.card_option_map)
        self.modal_open = not has_clickable
        if self.modal_open:
            self._open_modal(decision.options)
        else:
            self.modal_rows = []

    def _open_modal(self, options, scoped_iid: Optional[int] = None) -> None:
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

    def _pick(self, opt: Any) -> None:
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
                    self._open_modal(opts, scoped_iid=iid)
                return True
        return False

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
            add_right("End Turn", lambda: self._pick(end_turn), primary=True)

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

        if event.type == pygame.KEYDOWN and event.key in (pygame.K_F1, pygame.K_o):
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
            if self.modal_open:
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
        grid = all(r.card is not None for r in self.modal_rows) and len(self.modal_rows) > 0
        body = pygame.Rect(rect.left + 14, rect.top + 40, rect.width - 28, rect.height - 80)
        if grid:
            cols = max(1, body.width // 106)
            for i, row in enumerate(self.modal_rows):
                gx = i % cols
                gy = i // cols - self.modal_scroll
                cell = pygame.Rect(body.left + gx * 106, body.top + gy * 150, 96, 140)
                if cell.collidepoint(mouse_pos):
                    self._pick(row.option)
                    if self.decision.max_n <= 1 or self.decision.kind == DecisionKind.ORDER_EFFECTS:
                        self._close_modal()
                    return
        else:
            for i, row in enumerate(self.modal_rows):
                y = body.top + (i - self.modal_scroll) * (row_h + 6)
                cell = pygame.Rect(body.left, y, body.width, row_h)
                if cell.collidepoint(mouse_pos):
                    self._pick(row.option)
                    if self.decision.max_n <= 1 or self.decision.kind == DecisionKind.ORDER_EFFECTS:
                        self._close_modal()
                    return

    # --------------------------------------------------------------- draw ----

    def _modal_rect(self) -> pygame.Rect:
        w, h = 620, 520
        return pygame.Rect(S.PLAY_X + (S.PLAY_W - w) // 2, (S.CANVAS_H - h) // 2, w, h)

    def draw(self, surface: pygame.Surface, assets, layout) -> None:
        if self.decision is None:
            return
        self._draw_prompt(surface, assets, layout)
        mouse_pos = pygame.mouse.get_pos()
        for button, _cb in self.action_bar_buttons(layout):
            button.update_hover(mouse_pos)
            button.draw(surface, assets)
        if self.modal_open:
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
        dim = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 140))
        surface.blit(dim, (0, 0))
        draw_panel(surface, rect, alpha=245, border=S.AEMBER)

        title_font = assets.font("cinzel", 18)
        title = title_font.render(self.decision.prompt, True, S.TEXT)
        surface.blit(title, (rect.left + 16, rect.top + 10))

        body = pygame.Rect(rect.left + 14, rect.top + 40, rect.width - 28, rect.height - 80)
        clip = surface.get_clip()
        surface.set_clip(body)

        grid = all(r.card is not None for r in self.modal_rows) and len(self.modal_rows) > 0
        if grid:
            cols = max(1, body.width // 106)
            for i, row in enumerate(self.modal_rows):
                gx = i % cols
                gy = i // cols - self.modal_scroll
                cell = pygame.Rect(body.left + gx * 106, body.top + gy * 150, 96, 140)
                if cell.bottom < body.top or cell.top > body.bottom:
                    continue
                picked = row.option in self.picked
                img = assets.card_face(row.card.image, (96, 134))
                surface.blit(img, cell)
                if picked:
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
                bg = S.PANEL_LIGHT if picked else S.PANEL
                pygame.draw.rect(surface, bg, cell, border_radius=6)
                pygame.draw.rect(surface, S.GLOW_SELECTED if picked else S.TEXT_FAINT, cell, width=1, border_radius=6)
                font = assets.font("inter", 15)
                img = font.render(row.label, True, S.TEXT)
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
