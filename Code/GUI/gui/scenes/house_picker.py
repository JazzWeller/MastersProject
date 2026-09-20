"""Shared "pick 3 of the 7 houses" modal, used by both the Deck Builder and
Alliance Builder scenes (Code/PHASE_3_PLAN.md Milestone B: a deck is any 3
distinct houses out of 7, not a fixed Dis/Logos/Shadows triple).
"""

from __future__ import annotations

from typing import Callable, Dict, List

import pygame

from .. import settings as S
from ..sprites.widgets import draw_panel
from keyforge.enums import House

ALL_HOUSES = tuple(House)
PICK_COUNT = 3


class HousePicker:
    """Modal state + geometry + drawing for choosing exactly `PICK_COUNT`
    distinct houses. The owning scene supplies `panel_rect_fn` (its own
    panel, so the modal centers over it) and `on_confirm` (called with the
    chosen houses once exactly `PICK_COUNT` are picked and confirmed)."""

    def __init__(self, panel_rect_fn: Callable[[], pygame.Rect], on_confirm: Callable[[List[House]], None],
                 title: str = "Choose 3 houses"):
        self._panel_rect_fn = panel_rect_fn
        self.on_confirm = on_confirm
        self.title = title
        self.is_open = False
        self.selection: List[House] = []
        self.dismissable = True

    def open(self, current: List[House], dismissable: bool) -> None:
        self.selection = list(current)
        self.dismissable = dismissable
        self.is_open = True

    def toggle(self, house: House) -> None:
        if house in self.selection:
            self.selection.remove(house)
        elif len(self.selection) < PICK_COUNT:
            self.selection.append(house)

    # ------------------------------------------------------------ geometry ----

    def rect(self) -> pygame.Rect:
        p = self._panel_rect_fn()
        w, h = 900, 380
        return pygame.Rect(p.centerx - w // 2, p.centery - h // 2, w, h)

    def button_rects(self) -> Dict[House, pygame.Rect]:
        r = self.rect()
        cols = 4
        bw, bh = 200, 70
        gap_x, gap_y = 16, 16
        n = len(ALL_HOUSES)
        full_rows = n // cols
        rects: Dict[House, pygame.Rect] = {}
        for i, house in enumerate(ALL_HOUSES):
            row, col = divmod(i, cols)
            row_count = cols if row < full_rows else (n % cols or cols)
            row_w = row_count * bw + (row_count - 1) * gap_x
            start_x = r.centerx - row_w // 2
            x = start_x + col * (bw + gap_x)
            y = r.top + 76 + row * (bh + gap_y)
            rects[house] = pygame.Rect(x, y, bw, bh)
        return rects

    def confirm_rect(self) -> pygame.Rect:
        r = self.rect()
        return pygame.Rect(r.centerx - 160, r.bottom - 56, 150, 40)

    def cancel_rect(self) -> pygame.Rect:
        r = self.rect()
        return pygame.Rect(r.centerx + 10, r.bottom - 56, 150, 40)

    # --------------------------------------------------------------- input ----

    def handle_event(self, event: pygame.event.Event) -> None:
        """Consumes every event while open (it's a blocking modal)."""
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            if self.dismissable:
                self.is_open = False
            return
        if event.type != pygame.MOUSEBUTTONUP or event.button != 1:
            return
        for house, rect in self.button_rects().items():
            if rect.collidepoint(event.pos):
                self.toggle(house)
                return
        if self.confirm_rect().collidepoint(event.pos) and len(self.selection) == PICK_COUNT:
            self.is_open = False
            self.on_confirm(list(self.selection))
            return
        if self.dismissable and self.cancel_rect().collidepoint(event.pos):
            self.is_open = False

    # ---------------------------------------------------------------- draw ----

    def draw(self, surface: pygame.Surface, assets, mouse) -> None:
        if not self.is_open:
            return
        dim = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 170))
        surface.blit(dim, (0, 0))
        r = self.rect()
        draw_panel(surface, r, alpha=248, border=S.AEMBER)
        title = assets.font("cinzel", 20).render(self.title, True, S.TEXT)
        surface.blit(title, title.get_rect(midtop=(r.centerx, r.top + 16)))
        sub = assets.font("inter", 14).render(f"{len(self.selection)}/{PICK_COUNT} selected", True, S.TEXT_DIM)
        surface.blit(sub, sub.get_rect(midtop=(r.centerx, r.top + 46)))
        for house, rect in self.button_rects().items():
            selected = house in self.selection
            color = S.HOUSE_COLORS.get(house.value, S.TEXT_FAINT)
            bg = S.PANEL_LIGHT if selected else S.PANEL
            pygame.draw.rect(surface, bg, rect, border_radius=8)
            pygame.draw.rect(surface, color, rect, width=3 if selected else 1, border_radius=8)
            emblem = assets.house_emblem(house.value, 28)
            surface.blit(emblem, emblem.get_rect(midleft=(rect.left + 14, rect.centery)))
            label = assets.font("inter", 16, bold=selected).render(house.value, True, S.TEXT)
            surface.blit(label, label.get_rect(midleft=(rect.left + 52, rect.centery)))
        confirm = self.confirm_rect()
        ready = len(self.selection) == PICK_COUNT
        pygame.draw.rect(surface, S.AEMBER if ready else S.PANEL_LIGHT, confirm, border_radius=8)
        ctxt = assets.font("inter", 15, bold=True).render("Confirm", True, S.BLACK if ready else S.TEXT_FAINT)
        surface.blit(ctxt, ctxt.get_rect(center=confirm.center))
        if self.dismissable:
            cancel = self.cancel_rect()
            hovered = cancel.collidepoint(mouse)
            pygame.draw.rect(surface, S.PANEL_LIGHT if hovered else S.PANEL, cancel, border_radius=8)
            pygame.draw.rect(surface, S.TEXT_FAINT, cancel, width=1, border_radius=8)
            cxt = assets.font("inter", 15).render("Cancel", True, S.TEXT)
            surface.blit(cxt, cxt.get_rect(center=cancel.center))
