"""DecisionPanel: the one piece of UI that answers every `Decision` the
engine can produce for a human seat.

Interaction paths, all feeding the same `self.picked` accumulator:

1. **Board clicks.** Options that reference a card sitting in a hand or a
   play area make that card glow; clicking it picks the option, or opens a
   small chooser beside it when the card has several ("Play" / "Discard").
   Clicks resolve to the card drawn on top (`Board.click_target_at`).
2. **Dedicated screens** for decisions that need context to answer well:
   MULLIGAN and TAKE_ARCHIVE show the cards in question large; CHOOSE_FLANK
   shows clickable ghost slots at both ends of your creature row.
3. **The Options modal** (O, or the action-bar button, and opened by
   default when nothing else fits) lists every option as a labeled row or a
   readable card grid. It can answer *any* decision, so it's the guaranteed
   fallback.

Every card image the panel draws is registered in `hover_targets` so the
scene's zoom panel and inspector work on it exactly like on a board card.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pygame

from .. import settings as S
from ..option_labels import describe_option, describe_option_short
from ..sprites.widgets import Button, draw_panel

from keyforge.actions import DiscardCard, EndTurn
from keyforge.cards.card import Card
from keyforge.enums import CardType, DecisionKind, House

_ON_BOARD_ZONES = {"hand", "play_creature", "play_artifact", "upgrade"}

GRID_CELL_W, GRID_CELL_H = S.BROWSER_CARD_W, S.BROWSER_CARD_H
GRID_STRIDE_X, GRID_STRIDE_Y = GRID_CELL_W + 16, GRID_CELL_H + 30
ROW_H, ROW_STRIDE = 42, 48
REVIEW_CARD_W, REVIEW_CARD_H = S.MULLIGAN_CARD_W, S.MULLIGAN_CARD_H


def _option_card(opt: Any) -> Optional[Card]:
    if isinstance(opt, Card):
        return opt
    c = getattr(opt, "card", None)
    return c if isinstance(c, Card) else None


def _wrap(text: str, font, max_w: int) -> List[str]:
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


class OptionRow:
    __slots__ = ("option", "label", "card", "detail")

    def __init__(self, option, label, card, detail=""):
        self.option = option
        self.label = label
        self.card = card
        self.detail = detail


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

        self._scoped_card_iid: Optional[int] = None
        self._board = None

        # The action chooser: a compact popup beside one clicked card that
        # shows the card once plus a labeled button per option.
        self.chooser_iid: Optional[int] = None
        self.chooser_opts: List[Any] = []

        # Full-screen review of the cards a decision is about.
        self.mulligan_open = False
        self.archive_open = False
        # CHOOSE_FLANK: ghost slots on the board instead of a modal.
        self.flank_open = False

        # One-click misfire guard: Discard, and End Turn while other actions
        # exist, arm on the first click and need a second click on the *same*
        # control; any other click disarms (Code/PLAYTEST_FIX_PLAN.md U12).
        self._armed_action: Any = None
        self._last_pick: Any = None

        self.hover_house: Optional[House] = None
        # [(rect, iid)] for every card image drawn by the panel this frame.
        self.hover_targets: List[Tuple[pygame.Rect, int]] = []

    # --------------------------------------------------------------- setup ----

    def reset(self) -> None:
        """Forget the current decision and every popup that belonged to it."""
        self.decision = None
        self.result = None
        self.picked = []
        self.card_option_map = {}
        self.modal_open = False
        self.modal_rows = []
        self.chooser_iid = None
        self.chooser_opts = []
        self.mulligan_open = self.archive_open = self.flank_open = False
        self._armed_action = None
        self.hover_house = None
        self.hover_targets = []

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
        self.archive_open = False
        self.flank_open = False
        self._armed_action = None
        self.hover_house = None
        self.modal_open = False
        self.modal_rows = []

        for opt in decision.options:
            card = _option_card(opt)
            if card is None:
                continue
            cs = board.snapshot.cards.get(card.instance_id) if board.snapshot else None
            if cs is not None and cs.zone in _ON_BOARD_ZONES:
                self.card_option_map.setdefault(card.instance_id, []).append(opt)

        if decision.kind == DecisionKind.MULLIGAN:
            self.mulligan_open = True
            return
        if decision.kind == DecisionKind.TAKE_ARCHIVE:
            self.archive_open = True
            return
        if decision.kind == DecisionKind.CHOOSE_FLANK:
            self.flank_open = True
            return
        # A decision whose options aren't all on the board (e.g. a mix of
        # hand cards and discard-pile cards) opens the full list, so nothing
        # is reachable only via O.
        all_on_board = bool(self.card_option_map) and all(
            _option_card(o) is not None and _option_card(o).instance_id in self.card_option_map
            for o in decision.options
            if not isinstance(o, EndTurn)
        )
        # CHOOSE_ACTION never opens the list by itself: End Turn lives on the
        # action bar, and a modal holding only "End Turn" would just cover
        # the board (O still opens the full list).
        if not all_on_board and decision.kind != DecisionKind.CHOOSE_ACTION:
            self._open_modal(decision.options)

    def _open_modal(self, options, scoped_iid: Optional[int] = None) -> None:
        self._close_chooser()
        self.mulligan_open = self.archive_open = self.flank_open = False
        self.modal_rows = [
            OptionRow(o, describe_option(o, self.decision, self.view), _option_card(o), self._row_detail(o))
            for o in options
        ]
        self.modal_scroll = 0
        self.modal_open = True
        self._scoped_card_iid = scoped_iid

    def _close_modal(self) -> None:
        self.modal_open = False
        self._scoped_card_iid = None
        self.hover_house = None

    def _row_detail(self, opt) -> str:
        """Extra context under a modal row: for a house choice, what that
        house would let you do this turn (U8)."""
        if not isinstance(opt, House) or self.view is None or self.decision.kind != DecisionKind.CHOOSE_HOUSE:
            return ""
        me = self.view.me()
        in_hand = sum(1 for c in (me.hand or []) if c.house == opt)
        ready = sum(1 for c in me.creatures + me.artifacts if c.house == opt and not c.Exhausted)
        return f"{in_hand} in hand · {ready} ready in play"

    # ------------------------------------------------------------- picking ----

    def _submit(self, choice) -> None:
        self.result = choice

    def _finish_value(self, picked_list: List[Any]):
        if self.decision.kind in (DecisionKind.CHOOSE_CARDS, DecisionKind.ORDER_EFFECTS):
            return list(picked_list)
        return picked_list[0]

    def _needs_confirm(self, opt: Any) -> bool:
        if isinstance(opt, DiscardCard):
            return True
        if isinstance(opt, EndTurn) and self.decision is not None and len(self.decision.options) > 1:
            return True
        return False

    def _pick(self, opt: Any) -> None:
        self._last_pick = opt
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
            if any(p is opt for p in self.picked):
                return
            self.picked.append(opt)
            if len(self.picked) >= len(d.options):
                self._submit(self._finish_value(self.picked))
            return

        if d.max_n <= 1:
            self._submit(self._finish_value([opt]))
            return

        # multi-select: toggle
        if any(p is opt for p in self.picked):
            self.picked = [p for p in self.picked if p is not opt]
            return
        if len(self.picked) >= d.max_n:
            return
        self.picked.append(opt)
        if len(self.picked) == d.max_n:
            self._submit(self._finish_value(self.picked))

    def undo(self) -> None:
        if self.picked:
            self.picked.pop()

    def confirm(self) -> None:
        d = self.decision
        if d is not None and d.min_n <= len(self.picked) <= d.max_n:
            self._submit(self._finish_value(self.picked))

    def choose_none(self) -> None:
        if self.decision is not None and self.decision.min_n == 0:
            self._submit(self._finish_value([]))

    # ------------------------------------------------------------ board ui ----

    def sync_glow(self, board) -> None:
        for sprite in board.sprites.values():
            if sprite.glow in ("legal", "selected", "hint"):
                sprite.glow = None
            sprite.pick_number = None
        for iid, opts in self.card_option_map.items():
            sprite = board.sprites.get(iid)
            if sprite is None:
                continue
            picked_at = next((i for i, p in enumerate(self.picked) if any(p is o for o in opts)), None)
            if picked_at is not None:
                sprite.glow = "selected"
                if self.decision.max_n > 1 or self.decision.kind == DecisionKind.ORDER_EFFECTS:
                    sprite.pick_number = picked_at + 1
            else:
                sprite.glow = "legal"
        if self.hover_house is not None and board.snapshot is not None:
            for cs in board.snapshot.cards.values():
                if cs.owner == self.decision.player and cs.house == self.hover_house.value and cs.zone in ("hand", "play_creature", "play_artifact"):
                    sprite = board.sprites.get(cs.iid)
                    if sprite is not None and sprite.glow is None:
                        sprite.glow = "legal"

    def click_board(self, pos, board) -> bool:
        """Returns True if the click hit a card with a legal option."""
        iid = board.click_target_at(pos)
        if iid is None or iid not in self.card_option_map:
            return False
        opts = self.card_option_map[iid]
        if len(opts) == 1:
            self._pick(opts[0])
        else:
            self._open_chooser(iid, opts)
        return True

    # ------------------------------------------------------ action chooser ----

    def _open_chooser(self, iid: int, opts: List[Any]) -> None:
        self.chooser_iid = iid
        self.chooser_opts = opts

    def _close_chooser(self) -> None:
        self.chooser_iid = None
        self.chooser_opts = []

    def _chooser_rect(self, board) -> pygame.Rect:
        w, h = 300, max(130, 60 + 42 * len(self.chooser_opts))
        sprite = board.sprites.get(self.chooser_iid)
        cx, cy = (sprite.x, sprite.y) if sprite is not None else (S.PLAY_X + S.PLAY_W // 2, S.CANVAS_H // 2)
        half_w = (sprite.w / 2 if sprite is not None else 60) + 10
        x = cx + half_w if cx + half_w + w < S.BOARD_W - 10 else cx - half_w - w
        x = max(S.PLAY_X, min(x, S.PLAY_X + S.PLAY_W - w))
        y = max(10, min(cy - h // 2, S.CANVAS_H - h - 10))
        return pygame.Rect(int(x), int(y), w, h)

    def _chooser_option_rows(self, rect: pygame.Rect) -> List[pygame.Rect]:
        return [pygame.Rect(rect.left + 88, rect.top + 14 + i * 42, rect.width - 102, 34) for i in range(len(self.chooser_opts))]

    def _handle_chooser_click(self, mouse_pos, board) -> None:
        rect = self._chooser_rect(board)
        if not rect.collidepoint(mouse_pos):
            # Close it, then treat the click as the board click it was meant
            # as -- clicking card B while A's chooser is open used to only
            # close the chooser, which read as the click being ignored (M4).
            self._close_chooser()
            self.click_board(mouse_pos, board)
            return
        for row, opt in zip(self._chooser_option_rows(rect), self.chooser_opts):
            if row.collidepoint(mouse_pos):
                self._pick(opt)
                if self._armed_action is not opt:
                    self._close_chooser()
                return

    def _draw_chooser(self, surface, assets, board, mouse_pos) -> None:
        rect = self._chooser_rect(board)
        draw_panel(surface, rect, alpha=250, border=S.AEMBER)
        sprite = board.sprites.get(self.chooser_iid)
        cs = sprite.card_state if sprite is not None else None
        card_rect = pygame.Rect(rect.left + 12, rect.top + 14, 64, 90)
        if cs is not None and cs.image:
            surface.blit(assets.card_face(cs.image, card_rect.size), card_rect)
            self.hover_targets.append((card_rect, cs.iid))
        name_font = assets.font("inter", S.MIN_FONT, bold=True)
        y = card_rect.bottom + 4
        for line in _wrap(cs.name if cs else "Card", name_font, 70):
            t = name_font.render(line, True, S.TEXT_DIM)
            surface.blit(t, (card_rect.left, y))
            y += t.get_height()

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
            label = "Click again to confirm" if armed else describe_option_short(opt)
            img = label_font.render(label, True, fg)
            surface.blit(img, img.get_rect(center=row.center))

    # -------------------------------------------------------- review screens ----

    def _review_cards(self, board) -> List:
        if board.snapshot is None or self.decision is None:
            return []
        zone = "hand" if self.mulligan_open else "archive"
        return board.snapshot.zone_cards(self.decision.player, zone)

    def _review_card_rects(self, board, layout) -> List[tuple]:
        cards = self._review_cards(board)
        row = pygame.Rect(S.PLAY_X, 150, S.PLAY_W, REVIEW_CARD_H)
        slots = layout.row_slots(len(cards), row, REVIEW_CARD_W, REVIEW_CARD_H)
        out = []
        for cs, (x, y) in zip(cards, slots):
            r = pygame.Rect(0, 0, REVIEW_CARD_W, REVIEW_CARD_H)
            r.center = (int(x), int(y))
            out.append((r, cs))
        return out

    def _review_buttons(self) -> List[tuple]:
        yes = next((o for o in self.decision.options if o is True), None)
        no = next((o for o in self.decision.options if o is False), None)
        cx = S.PLAY_X + S.PLAY_W // 2
        w, h, y = 280, 50, 150 + REVIEW_CARD_H + 60
        left = pygame.Rect(cx - w - 12, y, w, h)
        right = pygame.Rect(cx + 12, y, w, h)
        if self.mulligan_open:
            n = len(self._review_cards(self._board)) if self._board else 0
            return [(Button(left, "Keep This Hand", primary=True), no),
                    (Button(right, f"Mulligan (draw {max(0, n - 1)})", danger=True), yes)]
        return [(Button(left, "Take Archive Into Hand", primary=True), yes),
                (Button(right, "Leave It Archived"), no)]

    def _handle_review_click(self, mouse_pos) -> None:
        fake_up = pygame.event.Event(pygame.MOUSEBUTTONUP, pos=mouse_pos, button=1)
        for button, opt in self._review_buttons():
            if button.clicked(fake_up) and opt is not None:
                self._pick(opt)
                return

    def _draw_review(self, surface, assets, board, layout, mouse_pos) -> None:
        dim = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 215))
        surface.blit(dim, (0, 0))
        cards = self._review_cards(board)
        cx = S.PLAY_X + S.PLAY_W // 2
        if self.mulligan_open:
            title_txt = "Mulligan your hand?"
            hint_txt = "A mulligan shuffles your hand back into your deck and draws one fewer card."
        else:
            title_txt = "Take your archive into your hand?"
            hint_txt = "You can take all archived cards now, or leave them for a later turn."
        title = assets.font("cinzel", 24).render(title_txt, True, S.TEXT)
        surface.blit(title, title.get_rect(center=(cx, 70)))
        houses: Dict[str, int] = {}
        for cs in cards:
            houses[cs.house] = houses.get(cs.house, 0) + 1
        summary = " · ".join(f"{n} {h}" for h, n in sorted(houses.items()))
        sub_font = assets.font("inter", 15)
        sub = sub_font.render(f"{len(cards)} card{'s' if len(cards) != 1 else ''}" + (f"  ({summary})" if summary else ""), True, S.TEXT_DIM)
        surface.blit(sub, sub.get_rect(center=(cx, 104)))

        for rect, cs in self._review_card_rects(board, layout):
            hovered = rect.collidepoint(mouse_pos)
            img = assets.card_face(cs.image, rect.size) if cs.face_up and cs.image else assets.card_back(rect.size)
            if hovered:
                pygame.draw.rect(surface, S.AEMBER, rect.inflate(10, 10), width=3, border_radius=10)
            surface.blit(img, rect)
            if cs.face_up:
                self.hover_targets.append((rect, cs.iid))
            name = assets.font("inter", 13, bold=True).render(cs.name if cs.face_up else "", True, S.TEXT)
            surface.blit(name, name.get_rect(midtop=(rect.centerx, rect.bottom + 6)))

        for button, _opt in self._review_buttons():
            button.update_hover(mouse_pos)
            button.draw(surface, assets)
        hint = sub_font.render(hint_txt + "   Right-click a card to read it full size.", True, S.TEXT_FAINT)
        surface.blit(hint, hint.get_rect(center=(cx, 150 + REVIEW_CARD_H + 140)))

    # --------------------------------------------------------- flank slots ----

    def flank_rects(self, board, layout) -> Dict[str, pygame.Rect]:
        pid = self.decision.player
        n = len(board.snapshot.zone_cards(pid, "play_creature")) if board.snapshot else 0
        out = {}
        for side, (x, y) in layout.flank_ghost_slots(pid, n).items():
            r = pygame.Rect(0, 0, S.BOARD_CARD_W, S.BOARD_CARD_H)
            r.center = (int(x), int(y))
            if any(r == other for other in out.values()):
                continue  # no creatures yet: both flanks are the same slot
            out[side] = r
        return out

    def _draw_flank(self, surface, assets, board, layout, mouse_pos) -> None:
        font = assets.font("inter", 14, bold=True)
        for side, rect in self.flank_rects(board, layout).items():
            hovered = rect.collidepoint(mouse_pos)
            panel = pygame.Surface(rect.size, pygame.SRCALPHA)
            pygame.draw.rect(panel, (*S.AEMBER, 90 if hovered else 45), panel.get_rect(), border_radius=10)
            surface.blit(panel, rect)
            pygame.draw.rect(surface, S.AEMBER, rect, width=3 if hovered else 2, border_radius=10)
            for i, line in enumerate((f"{side.capitalize()}", "flank")):
                t = font.render(line, True, S.TEXT)
                surface.blit(t, t.get_rect(center=(rect.centerx, rect.centery - 10 + i * 20)))

    def _handle_flank_click(self, mouse_pos, board, layout) -> None:
        for side, rect in self.flank_rects(board, layout).items():
            if rect.collidepoint(mouse_pos) and side in self.decision.options:
                self._pick(side)
                return

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
            only = len(self.decision.options) == 1
            label = "Click again to end turn" if armed else "End Turn"
            add_right(label, lambda: self._pick(end_turn), primary=not armed or only, danger=armed, w=230 if armed else 132)

        d = self.decision
        if d.kind != DecisionKind.ORDER_EFFECTS and d.max_n > 1:
            can_confirm = d.min_n <= len(self.picked) <= d.max_n
            add_right(f"Confirm ({len(self.picked)}/{d.max_n})", self.confirm, primary=True, enabled=can_confirm, w=150)
        if d.min_n == 0 and not self.picked and d.kind == DecisionKind.CHOOSE_CARDS:
            add_right("Choose none", self.choose_none)
        if self.picked and (d.max_n > 1 or d.kind == DecisionKind.ORDER_EFFECTS):
            add_right("Undo", self.undo, w=90)

        add_right("All options (O)", lambda: self._open_modal(self.decision.options), w=150)
        return buttons

    # ------------------------------------------------------------- events ----

    def handle_event(self, event: pygame.event.Event, board, mouse_pos, layout) -> Optional[Any]:
        if self.decision is None:
            return None

        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            if self.chooser_iid is not None:
                self._close_chooser()
            elif self.modal_open and self._default_view_available():
                self._close_modal()
                self._restore_default_view()
            self._armed_action = None
            return self.result

        if event.type == pygame.KEYDOWN and event.key in (pygame.K_F1, pygame.K_o):
            self._close_chooser()
            if self.modal_open:
                if self._default_view_available():
                    self._close_modal()
                    self._restore_default_view()
            else:
                self._open_modal(self.decision.options)
            return self.result

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            prev_armed = self._armed_action
            self._last_pick = None
            self._dispatch_click(mouse_pos, board, layout)
            if prev_armed is not None and self._armed_action is prev_armed and self._last_pick is not prev_armed:
                self._armed_action = None
        elif event.type == pygame.MOUSEWHEEL and self.modal_open:
            self.modal_scroll = max(0, min(self._max_modal_scroll(), self.modal_scroll - event.y))
        elif event.type == pygame.MOUSEMOTION and self.modal_open and self.decision.kind == DecisionKind.CHOOSE_HOUSE:
            row = self._modal_row_at(mouse_pos)
            self.hover_house = row.option if row is not None and isinstance(row.option, House) else None

        return self.result

    def _dispatch_click(self, mouse_pos, board, layout) -> None:
        fake_up = pygame.event.Event(pygame.MOUSEBUTTONUP, pos=mouse_pos, button=1)
        if not (self.mulligan_open or self.archive_open):
            for button, cb in self.action_bar_buttons(layout):
                if button.clicked(fake_up):
                    self._last_pick = self._armed_action if "end turn" in button.text.lower() else None
                    cb()
                    return
        if self.mulligan_open or self.archive_open:
            self._handle_review_click(mouse_pos)
        elif self.chooser_iid is not None:
            self._handle_chooser_click(mouse_pos, board)
        elif self.modal_open:
            self._handle_modal_click(mouse_pos, board)
        elif self.flank_open:
            self._handle_flank_click(mouse_pos, board, layout)
        else:
            self.click_board(mouse_pos, board)

    def _default_view_available(self) -> bool:
        """Whether closing the Options modal leaves a usable view behind."""
        d = self.decision
        return bool(self.card_option_map) or d.kind in (
            DecisionKind.MULLIGAN, DecisionKind.TAKE_ARCHIVE, DecisionKind.CHOOSE_FLANK
        )

    def _restore_default_view(self) -> None:
        d = self.decision
        self.mulligan_open = d.kind == DecisionKind.MULLIGAN
        self.archive_open = d.kind == DecisionKind.TAKE_ARCHIVE
        self.flank_open = d.kind == DecisionKind.CHOOSE_FLANK

    # -------------------------------------------------------------- modal ----

    def _modal_is_grid(self) -> bool:
        """A card grid only when every row is a *distinct* card; otherwise the
        same art would repeat with nothing to tell the rows apart (F3)."""
        if not self.modal_rows or any(r.card is None for r in self.modal_rows):
            return False
        ids = [r.card.instance_id for r in self.modal_rows]
        return len(ids) == len(set(ids))

    def _modal_title_lines(self, assets, width) -> List[str]:
        return _wrap(self.decision.prompt, assets.font("cinzel", 18), width - 32)

    def _modal_rect(self, assets=None) -> pygame.Rect:
        """Sized to its content (and title), not a fixed box (D1/U23)."""
        grid = self._modal_is_grid()
        n = len(self.modal_rows)
        if grid:
            w = 4 * GRID_STRIDE_X + 44
            rows = max(1, -(-n // 4))
            body_h = min(2, rows) * GRID_STRIDE_Y
        else:
            w = 520
            body_h = min(9, max(1, n)) * ROW_STRIDE
        title_lines = 1
        if assets is not None:
            title_lines = len(self._modal_title_lines(assets, w))
        header = 16 + title_lines * 26 + 8
        h = header + body_h + 50
        x = S.PLAY_X + (S.PLAY_W - w) // 2
        y = max(10, (S.CANVAS_H - h) // 2)
        return pygame.Rect(x, y, w, h)

    def _modal_body(self, rect, assets=None) -> pygame.Rect:
        lines = len(self._modal_title_lines(assets, rect.width)) if assets is not None else 1
        top = rect.top + 16 + lines * 26 + 8
        return pygame.Rect(rect.left + 22, top, rect.width - 44, rect.bottom - 50 - top)

    def _visible_capacity(self) -> int:
        return 4 * 2 if self._modal_is_grid() else 9

    def _max_modal_scroll(self) -> int:
        if self._modal_is_grid():
            rows = -(-len(self.modal_rows) // 4)
            return max(0, rows - 2)
        return max(0, len(self.modal_rows) - 9)

    def _modal_cells(self, body) -> List[Tuple[pygame.Rect, OptionRow]]:
        cells = []
        if self._modal_is_grid():
            for i, row in enumerate(self.modal_rows):
                gx, gy = i % 4, i // 4 - self.modal_scroll
                cell = pygame.Rect(body.left + gx * GRID_STRIDE_X, body.top + gy * GRID_STRIDE_Y, GRID_CELL_W, GRID_CELL_H)
                if body.top - 1 <= cell.top and cell.bottom <= body.bottom + 1:
                    cells.append((cell, row))
        else:
            for i, row in enumerate(self.modal_rows):
                y = body.top + (i - self.modal_scroll) * ROW_STRIDE
                cell = pygame.Rect(body.left, y, body.width, ROW_H)
                if body.top - 1 <= cell.top and cell.bottom <= body.bottom + 1:
                    cells.append((cell, row))
        return cells

    def _modal_row_at(self, pos) -> Optional[OptionRow]:
        assets = self._board.assets if self._board is not None else None
        rect = self._modal_rect(assets)
        for cell, row in self._modal_cells(self._modal_body(rect, assets)):
            if cell.collidepoint(pos):
                return row
        return None

    def _handle_modal_click(self, mouse_pos, board) -> None:
        assets = board.assets if board is not None else None
        rect = self._modal_rect(assets)
        if not rect.collidepoint(mouse_pos):
            if self._default_view_available():
                self._close_modal()
                self._restore_default_view()
                if self.card_option_map:
                    self.click_board(mouse_pos, board)
            return
        row = self._modal_row_at(mouse_pos)
        if row is None:
            return
        self._pick(row.option)
        if self._armed_action is row.option:
            return
        if self.result is not None:
            self._close_modal()

    def _draw_modal(self, surface, assets, mouse_pos) -> None:
        rect = self._modal_rect(assets)
        grid = self._modal_is_grid()
        dim = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 150 if grid else 90))
        surface.blit(dim, (0, 0))
        draw_panel(surface, rect, alpha=248, border=S.AEMBER)

        title_font = assets.font("cinzel", 18)
        for i, line in enumerate(self._modal_title_lines(assets, rect.width)):
            surface.blit(title_font.render(line, True, S.TEXT), (rect.left + 16, rect.top + 14 + i * 26))

        body = self._modal_body(rect, assets)
        is_order = self.decision.kind == DecisionKind.ORDER_EFFECTS
        name_font = assets.font("inter", 13)
        row_font = assets.font("inter", 15)
        detail_font = assets.font("inter", S.MIN_FONT)
        for cell, row in self._modal_cells(body):
            picked_at = next((i for i, p in enumerate(self.picked) if p is row.option), None)
            armed = row.option is self._armed_action
            hovered = cell.collidepoint(mouse_pos)
            if grid:
                surface.blit(assets.card_face(row.card.image, cell.size), cell)
                self.hover_targets.append((cell, row.card.instance_id))
                border = S.DANGER if armed else (S.GLOW_SELECTED if picked_at is not None else (S.AEMBER if hovered else S.TEXT_FAINT))
                pygame.draw.rect(surface, border, cell, width=3 if (armed or picked_at is not None or hovered) else 1, border_radius=6)
                name = name_font.render(row.card.name, True, S.TEXT if hovered else S.TEXT_DIM)
                if name.get_width() > cell.width:
                    name = pygame.transform.smoothscale(name, (cell.width, name.get_height()))
                surface.blit(name, (cell.left, cell.bottom + 4))
                if armed:
                    self._tag(surface, assets, cell.center, "Click again", S.DANGER)
                elif picked_at is not None:
                    self._tag(surface, assets, (cell.centerx, cell.top + 18), str(picked_at + 1), S.WHITE, fg=S.BLACK)
            else:
                bg = S.DANGER if armed else (S.PANEL_LIGHT if (picked_at is not None or hovered) else S.PANEL)
                pygame.draw.rect(surface, bg, cell, border_radius=6)
                pygame.draw.rect(surface, S.GLOW_SELECTED if picked_at is not None else (S.AEMBER if hovered else S.TEXT_FAINT), cell, width=1, border_radius=6)
                x = cell.left + 12
                if isinstance(row.option, House):
                    pygame.draw.circle(surface, S.HOUSE_COLORS.get(row.option.value, S.TEXT_DIM), (x + 8, cell.centery), 8)
                    x += 24
                if is_order:
                    num = str(picked_at + 1) if picked_at is not None else "·"
                    t = row_font.render(num, True, S.AEMBER)
                    surface.blit(t, t.get_rect(center=(x + 8, cell.centery)))
                    x += 26
                label = f"Click again to confirm: {row.label}" if armed else row.label
                img = row_font.render(label, True, S.WHITE if armed else S.TEXT)
                if row.detail:
                    surface.blit(img, (x, cell.top + 3))
                    surface.blit(detail_font.render(row.detail, True, S.TEXT_DIM), (x, cell.top + 23))
                else:
                    surface.blit(img, (x, cell.centery - img.get_height() // 2))
                if row.card is not None:
                    self.hover_targets.append((cell, row.card.instance_id))

        # scroll indicator (U24)
        max_scroll = self._max_modal_scroll()
        footer_font = assets.font("inter", S.MIN_FONT)
        if max_scroll > 0:
            track = pygame.Rect(rect.right - 12, body.top, 5, body.height)
            pygame.draw.rect(surface, S.PANEL_LIGHT, track, border_radius=3)
            frac = body.height * self._visible_capacity() / max(len(self.modal_rows) + (3 if grid else 0), 1)
            thumb_h = max(24, min(body.height, int(frac)))
            thumb_y = track.top + int((body.height - thumb_h) * self.modal_scroll / max_scroll)
            pygame.draw.rect(surface, S.AEMBER, pygame.Rect(track.left, thumb_y, 5, thumb_h), border_radius=3)
            more = footer_font.render("Scroll for more", True, S.TEXT_FAINT)
            surface.blit(more, (rect.right - more.get_width() - 20, rect.bottom - 30))

        if is_order:
            note = f"Click in the order they should resolve ({len(self.picked)}/{len(self.decision.options)}) — Undo on the bar below"
        elif self.decision.max_n > 1:
            note = f"Selected {len(self.picked)} of {self.decision.max_n}" + (f" (at least {self.decision.min_n})" if self.decision.min_n else "")
        elif self._default_view_available():
            note = "Esc or O to close this list"
        else:
            note = ""
        if note:
            surface.blit(footer_font.render(note, True, S.TEXT_DIM), (rect.left + 16, rect.bottom - 30))

    def _tag(self, surface, assets, center, text, color, fg=S.WHITE):
        font = assets.font("inter", 13, bold=True)
        t = font.render(text, True, fg)
        r = t.get_rect(center=center).inflate(12, 6)
        pygame.draw.rect(surface, color, r, border_radius=r.height // 2)
        surface.blit(t, t.get_rect(center=r.center))

    # --------------------------------------------------------------- draw ----

    def prompt_text(self) -> str:
        d = self.decision
        if d is None:
            return ""
        if d.kind == DecisionKind.CHOOSE_ACTION and len(d.options) == 1:
            return "No more plays available — End Turn"
        text = d.prompt
        if d.max_n > 1 and d.kind == DecisionKind.CHOOSE_CARDS:
            text += f"  ({len(self.picked)} of {d.max_n} chosen)"
        return text

    @property
    def full_screen(self) -> bool:
        return self.decision is not None and (self.mulligan_open or self.archive_open)

    def draw(self, surface: pygame.Surface, assets, layout, mouse_pos=(-1, -1)) -> None:
        self.hover_targets = []
        if self.decision is None:
            return
        if self.full_screen:
            self._draw_review(surface, assets, self._board, layout, mouse_pos)
            return
        if not self.modal_open:
            self._draw_prompt(surface, assets, layout)
        if self.flank_open:
            self._draw_flank(surface, assets, self._board, layout, mouse_pos)
        for button, _cb in self.action_bar_buttons(layout):
            button.update_hover(mouse_pos)
            button.draw(surface, assets)
        if self.chooser_iid is not None:
            self._draw_chooser(surface, assets, self._board, mouse_pos)
        elif self.modal_open:
            self._draw_modal(surface, assets, mouse_pos)

    def _draw_prompt(self, surface, assets, layout) -> None:
        rect = layout.prompt_rect()
        font = assets.font("inter", 18, bold=True)
        # Left 340px of the band belong to the turn / first-turn status.
        area = pygame.Rect(rect.left + 340, rect.top, rect.width - 340, rect.height)
        img = font.render(self.prompt_text(), True, S.TEXT)
        if img.get_width() > area.width - 20:
            img = assets.font("inter", 14, bold=True).render(self.prompt_text(), True, S.TEXT)
        surface.blit(img, img.get_rect(center=area.center))
