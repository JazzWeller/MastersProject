"""Deck Builder: build or edit a 3-house, 12-card-per-house deck from the
full card pool, validate it, and save it to `Code/GUI/data/decks/` (Code/
PHASE_2_PLAN.md Milestone E). Reached from the main menu; `AllianceBuilder
Scene` reuses its house-tab/grid layout for picking whole pods instead of
individual cards.
"""

from __future__ import annotations

import os
import random
import re
from types import SimpleNamespace
from typing import Dict, List, Optional

import pygame

from .. import settings as S
from ..app import Scene
from ..sprites.widgets import Button, draw_panel

from keyforge.cards.card_data import CARD_DEFS, get_card_def
from keyforge.cards.decks import (
    CARDS_PER_POD,
    DECKS as BUNDLED_DECKS,
    Deck,
    deck_label,
    load_deck_json,
    random_deck,
    save_deck_json,
    validate_deck,
)
from keyforge.enums import CardType, House

HOUSES = (House.DIS, House.LOGOS, House.SHADOWS)
TYPE_FILTERS = (None, CardType.CREATURE, CardType.ACTION, CardType.ARTIFACT, CardType.UPGRADE)
TYPE_LABELS = {None: "All", CardType.CREATURE: "Creature", CardType.ACTION: "Action",
               CardType.ARTIFACT: "Artifact", CardType.UPGRADE: "Upgrade"}

GRID_STRIDE_X, GRID_STRIDE_Y = 156, 232
GRID_CELL_W, GRID_CELL_H = S.BROWSER_CARD_W, S.BROWSER_CARD_H

_NAMES_BY_HOUSE: Dict[House, List[str]] = {}
for _name, _cdef in CARD_DEFS.items():
    _NAMES_BY_HOUSE.setdefault(_cdef.house, []).append(_name)
for _house in HOUSES:
    _NAMES_BY_HOUSE[_house].sort(key=lambda n: (CARD_DEFS[n].type.value, n))


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "deck"


def _info_from_name(name: str) -> SimpleNamespace:
    cdef = get_card_def(name)
    return SimpleNamespace(
        name=cdef.name, house=cdef.house.value, type=cdef.type.value, image=cdef.image,
        power=cdef.power, armor=cdef.armor, damage=0, text=cdef.text, errata=cdef.errata,
    )


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


class DeckBuilderScene(Scene):
    def __init__(self, deck: Optional[Deck] = None, saved_path: Optional[str] = None):
        self.name = deck.name if deck is not None else "New Deck"
        self.pods: Dict[House, List[str]] = {h: list(deck.pods.get(h, [])) if deck is not None else [] for h in HOUSES}
        self.saved_path = saved_path
        self.active_house = House.DIS
        self.type_filter: Optional[CardType] = None
        self.scroll = 0
        self.editing_name = False
        self.name_buffer = ""
        self.message: Optional[str] = None
        self.message_is_error = False
        self.picker_open = False  # the Load picker
        self.confirm_delete: Optional[str] = None
        self.inspect_info: Optional[SimpleNamespace] = None
        self.rng = random.Random()

    def on_enter(self) -> None:
        os.makedirs(S.USER_DECKS_DIR, exist_ok=True)

    def on_exit(self) -> None:
        if self.editing_name:
            pygame.key.stop_text_input()

    # -------------------------------------------------------------- data ----

    def _house_cards(self, house: House) -> List[str]:
        names = _NAMES_BY_HOUSE[house]
        if self.type_filter is None:
            return names
        return [n for n in names if CARD_DEFS[n].type == self.type_filter]

    def _pod(self) -> List[str]:
        return self.pods[self.active_house]

    def _add_card(self, name: str) -> None:
        pod = self._pod()
        if len(pod) >= CARDS_PER_POD:
            self._set_message(f"{self.active_house.value}'s pod is already full (12/12).", error=True)
            return
        pod.append(name)
        self.app.assets.play("click", 0.15)

    def _remove_one(self, name: str) -> None:
        pod = self._pod()
        if name in pod:
            pod.remove(name)
            self.app.assets.play("click", 0.15)

    def _current_deck(self) -> Deck:
        return Deck(name=self.name, pods={h: list(cards) for h, cards in self.pods.items()})

    def _errors(self) -> List[str]:
        return validate_deck(self._current_deck())

    def _set_message(self, text: str, error: bool = False) -> None:
        self.message, self.message_is_error = text, error

    def _random_fill(self) -> None:
        deck = random_deck(self.rng, name=self.name)
        self.pods = {h: list(deck.pods[h]) for h in HOUSES}
        self._set_message("Filled all three houses with random cards.")

    def _user_deck_files(self) -> List[str]:
        try:
            files = [f for f in os.listdir(S.USER_DECKS_DIR) if f.endswith(".json")]
        except FileNotFoundError:
            return []
        return sorted(os.path.join(S.USER_DECKS_DIR, f) for f in files)

    def _save(self) -> None:
        errors = self._errors()
        if errors:
            self._set_message(f"Can't save: {errors[0]}", error=True)
            return
        path = self.saved_path or os.path.join(S.USER_DECKS_DIR, f"{_slugify(self.name)}.json")
        save_deck_json(self._current_deck(), path)
        self.saved_path = path
        self._set_message(f"Saved to {os.path.basename(path)}.")

    def _load(self, path: str) -> None:
        try:
            deck = load_deck_json(path)
        except Exception as exc:
            self._set_message(f"Couldn't load that deck: {exc}", error=True)
            return
        self.name = deck.name
        self.pods = {h: list(deck.pods.get(h, [])) for h in HOUSES}
        self.saved_path = path
        self.picker_open = False
        self._set_message(f"Loaded {deck.name}.")

    def _load_preset(self, key: str) -> None:
        deck = BUNDLED_DECKS[key]
        self.name = f"{deck.name} (copy)"
        self.pods = {h: list(deck.pods.get(h, [])) for h in HOUSES}
        self.saved_path = None
        self.picker_open = False
        self._set_message(f"Loaded a copy of {deck.name}. Save it to keep your changes.")

    def _delete(self, path: str) -> None:
        try:
            os.remove(path)
        except OSError as exc:
            self._set_message(f"Couldn't delete: {exc}", error=True)
            return
        if self.saved_path == path:
            self.saved_path = None
        self.confirm_delete = None

    # ------------------------------------------------------------ geometry ----

    def _panel(self) -> pygame.Rect:
        return pygame.Rect(24, 16, S.CANVAS_W - 48, S.CANVAS_H - 32)

    def _grid_rect(self) -> pygame.Rect:
        p = self._panel()
        return pygame.Rect(p.left + 16, p.top + 172, p.width - 420, p.height - 192)

    def _pod_rect(self) -> pygame.Rect:
        p = self._panel()
        g = self._grid_rect()
        return pygame.Rect(g.right + 16, p.top + 172, p.right - g.right - 32, p.height - 192)

    def _name_rect(self) -> pygame.Rect:
        p = self._panel()
        return pygame.Rect(p.left + 20, p.top + 12, 420, 40)

    def _top_buttons(self) -> Dict[str, Button]:
        p = self._panel()
        y = p.top + 12
        return {
            "back": Button(pygame.Rect(p.right - 604, y, 100, 40), "Back"),
            "random": Button(pygame.Rect(p.right - 496, y, 130, 40), "Random Fill"),
            "load": Button(pygame.Rect(p.right - 358, y, 100, 40), "Load"),
            "save": Button(pygame.Rect(p.right - 250, y, 120, 40), "Save", primary=True, enabled=not self._errors()),
            "delete": Button(pygame.Rect(p.right - 122, y, 100, 40), "Delete", danger=True, enabled=self.saved_path is not None),
        }

    def _house_tab_rects(self) -> Dict[House, pygame.Rect]:
        p = self._panel()
        y = p.top + 64
        w = 240
        return {h: pygame.Rect(p.left + 20 + i * (w + 10), y, w, 40) for i, h in enumerate(HOUSES)}

    def _type_filter_rects(self) -> Dict[Optional[CardType], pygame.Rect]:
        p = self._panel()
        y = p.top + 116
        x = p.left + 20
        rects = {}
        for tf in TYPE_FILTERS:
            w = 90
            rects[tf] = pygame.Rect(x, y, w, 32)
            x += w + 8
        return rects

    def _grid_cols(self) -> int:
        return max(1, self._grid_rect().width // GRID_STRIDE_X)

    def _grid_cells(self) -> List[tuple]:
        rect = self._grid_rect()
        cols = self._grid_cols()
        names = self._house_cards(self.active_house)
        cells = []
        for i, name in enumerate(names):
            gx, gy = i % cols, i // cols - self.scroll
            cell = pygame.Rect(rect.left + gx * GRID_STRIDE_X, rect.top + gy * GRID_STRIDE_Y, GRID_CELL_W, GRID_CELL_H)
            if rect.top - 1 <= cell.top and cell.bottom <= rect.bottom + 1:
                cells.append((cell, name))
        return cells

    def _max_scroll(self) -> int:
        cols = self._grid_cols()
        rows = -(-len(self._house_cards(self.active_house)) // cols)
        visible_rows = max(1, self._grid_rect().height // GRID_STRIDE_Y)
        return max(0, rows - visible_rows)

    # -------------------------------------------------------------- input ----

    def handle_event(self, event: pygame.event.Event) -> None:
        if self.editing_name:
            self._handle_name_edit(event)
            return
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            if self.picker_open:
                self.picker_open = False
                self.confirm_delete = None
            elif self.inspect_info is not None:
                self.inspect_info = None
            else:
                self._back()
            return
        if event.type == pygame.MOUSEWHEEL and not self.picker_open:
            self.scroll = max(0, min(self._max_scroll(), self.scroll - event.y))
            return
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
            self._handle_right_click(event.pos)
            return
        if event.type != pygame.MOUSEBUTTONUP or event.button != 1:
            return
        if self.inspect_info is not None:
            self.inspect_info = None
            return
        if self.picker_open:
            self._handle_picker_click(event)
            return
        for key, button in self._top_buttons().items():
            if button.clicked(event):
                self._handle_top_button(key)
                return
        if self._name_rect().collidepoint(event.pos):
            self.editing_name = True
            self.name_buffer = self.name
            pygame.key.start_text_input()
            return
        for house, rect in self._house_tab_rects().items():
            if rect.collidepoint(event.pos):
                self.active_house, self.scroll = house, 0
                self.app.assets.play("click", 0.2)
                return
        for tf, rect in self._type_filter_rects().items():
            if rect.collidepoint(event.pos):
                self.type_filter, self.scroll = tf, 0
                self.app.assets.play("click", 0.2)
                return
        for cell, name in self._grid_cells():
            if cell.collidepoint(event.pos):
                self._add_card(name)
                return
        for row_rect, name, _count in self._pod_rows():
            remove_rect = pygame.Rect(row_rect.right - 28, row_rect.top + 4, 22, row_rect.height - 8)
            if remove_rect.collidepoint(event.pos):
                self._remove_one(name)
                return

    def _back(self) -> None:
        self.app.pop()
        # Returning to an already-on-the-stack MenuScene doesn't re-trigger
        # its on_enter, so refresh its deck list explicitly -- a deck may
        # have just been saved or deleted here.
        if self.app.scenes:
            under = self.app.scenes[-1]
            refresh = getattr(under, "_refresh_decks", None)
            if callable(refresh):
                refresh()

    def _handle_top_button(self, key: str) -> None:
        self.app.assets.play("click", 0.3)
        if key == "back":
            self._back()
        elif key == "random":
            self._random_fill()
        elif key == "load":
            self.picker_open = True
        elif key == "save":
            self._save()
        elif key == "delete" and self.saved_path is not None:
            self._delete(self.saved_path)

    def _handle_right_click(self, pos) -> None:
        for cell, name in self._grid_cells():
            if cell.collidepoint(pos):
                self.inspect_info = _info_from_name(name)
                return
        for row_rect, name, _count in self._pod_rows():
            if row_rect.collidepoint(pos):
                self.inspect_info = _info_from_name(name)
                return

    def _handle_name_edit(self, event: pygame.event.Event) -> None:
        if event.type == pygame.TEXTINPUT:
            if len(self.name_buffer) < 40:
                self.name_buffer += event.text
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_BACKSPACE:
                self.name_buffer = self.name_buffer[:-1]
            elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                self.name = self.name_buffer.strip() or "New Deck"
                self.editing_name = False
                pygame.key.stop_text_input()
            elif event.key == pygame.K_ESCAPE:
                self.editing_name = False
                pygame.key.stop_text_input()

    # --------------------------------------------------------- load picker ----

    def _picker_rect(self) -> pygame.Rect:
        p = self._panel()
        w, h = 640, 560
        return pygame.Rect(p.centerx - w // 2, p.centery - h // 2, w, h)

    def _picker_rows(self) -> List[tuple]:
        rows = [("preset", key, deck.name) for key, deck in sorted(BUNDLED_DECKS.items(), key=lambda kv: kv[1].name)]
        rows += [("user", path, os.path.splitext(os.path.basename(path))[0]) for path in self._user_deck_files()]
        return rows

    def _picker_row_rect(self, i: int) -> pygame.Rect:
        r = self._picker_rect()
        return pygame.Rect(r.left + 16, r.top + 60 + i * 44, r.width - 32, 38)

    def _handle_picker_click(self, event: pygame.event.Event) -> None:
        r = self._picker_rect()
        close = pygame.Rect(r.right - 34, r.top + 8, 26, 26)
        if close.collidepoint(event.pos):
            self.picker_open = False
            return
        if not r.collidepoint(event.pos):
            self.picker_open = False
            return
        for i, (kind, ref, _label) in enumerate(self._picker_rows()):
            row = self._picker_row_rect(i)
            if not row.collidepoint(event.pos):
                continue
            if kind == "preset":
                self._load_preset(ref)
                return
            delete_rect = pygame.Rect(row.right - 84, row.top + 3, 76, row.height - 6)
            if kind == "user" and delete_rect.collidepoint(event.pos):
                if self.confirm_delete == ref:
                    self._delete(ref)
                else:
                    self.confirm_delete = ref
                return
            self._load(ref)
            return

    def _pod_rows(self) -> List[tuple]:
        """(row_rect, card_name, count) for each distinct card currently in
        the active house's pod, most-copies first."""
        pod = self._pod()
        counts: Dict[str, int] = {}
        for name in pod:
            counts[name] = counts.get(name, 0) + 1
        rect = self._pod_rect()
        y = rect.top + 44
        rows = []
        for name, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            rows.append((pygame.Rect(rect.left, y, rect.width, 30), name, count))
            y += 32
        return rows

    # --------------------------------------------------------------- draw ----

    def draw(self, surface: pygame.Surface) -> None:
        surface.fill(S.BG_DEEP)
        assets = self.app.assets
        p = self._panel()
        draw_panel(surface, p, alpha=235, border=S.TEXT_FAINT)

        name_rect = self._name_rect()
        title_text = (self.name_buffer if self.editing_name else self.name) or " "
        cursor = "|" if self.editing_name and (pygame.time.get_ticks() // 500) % 2 == 0 else ""
        name_img = assets.font("cinzel", 26).render(title_text + cursor, True, S.TEXT)
        pygame.draw.rect(surface, S.PANEL_LIGHT if self.editing_name else S.PANEL, name_rect, border_radius=6)
        pygame.draw.rect(surface, S.AEMBER if self.editing_name else S.TEXT_FAINT, name_rect, width=1, border_radius=6)
        surface.blit(name_img, name_img.get_rect(midleft=(name_rect.left + 10, name_rect.centery)))
        if not self.editing_name:
            hint = assets.font("inter", S.MIN_FONT).render("Click to rename", True, S.TEXT_FAINT)
            surface.blit(hint, (name_rect.left + 4, name_rect.bottom + 2))

        mouse = self.mouse
        for button in self._top_buttons().values():
            button.update_hover(mouse)
            button.draw(surface, assets)

        errors = self._errors()
        status_font = assets.font("inter", 14, bold=True)
        if errors:
            status = status_font.render(f"Not yet a legal deck: {errors[0]}", True, S.NOTE)
        else:
            status = status_font.render("Legal deck — ready to save.", True, S.HEAL)
        surface.blit(status, (p.left + 20, p.top + 40))

        for house, rect in self._house_tab_rects().items():
            active = house == self.active_house
            n = len(self.pods[house])
            ok = n == CARDS_PER_POD
            bg = S.PANEL_LIGHT if active else S.PANEL
            pygame.draw.rect(surface, bg, rect, border_radius=8)
            pygame.draw.rect(surface, S.HOUSE_COLORS.get(house.value, S.TEXT_FAINT), rect, width=2 if active else 1, border_radius=8)
            dot_color = S.HEAL if ok else S.NOTE
            pygame.draw.circle(surface, dot_color, (rect.left + 18, rect.centery), 5)
            label = assets.font("inter", 16, bold=active).render(f"{house.value}  {n}/{CARDS_PER_POD}", True, S.TEXT)
            surface.blit(label, label.get_rect(midleft=(rect.left + 32, rect.centery)))

        for tf, rect in self._type_filter_rects().items():
            active = tf == self.type_filter
            pygame.draw.rect(surface, S.AEMBER if active else S.PANEL, rect, border_radius=6)
            pygame.draw.rect(surface, S.AEMBER if active else S.TEXT_FAINT, rect, width=1, border_radius=6)
            fg = S.BLACK if active else S.TEXT_DIM
            label = assets.font("inter", 13, bold=active).render(TYPE_LABELS[tf], True, fg)
            surface.blit(label, label.get_rect(center=rect.center))

        self._draw_grid(surface, assets, mouse)
        self._draw_pod(surface, assets, mouse)

        if self.message:
            color = S.DANGER if self.message_is_error else S.TEXT_DIM
            m = assets.font("inter", 14).render(self.message, True, color)
            surface.blit(m, m.get_rect(bottomleft=(p.left + 20, p.bottom - 12)))

        if self.picker_open:
            self._draw_picker(surface, assets, mouse)
        if self.inspect_info is not None:
            self._draw_inspector(surface, assets)

    def _draw_grid(self, surface, assets, mouse) -> None:
        rect = self._grid_rect()
        counts: Dict[str, int] = {}
        for name in self._pod():
            counts[name] = counts.get(name, 0) + 1
        name_font = assets.font("inter", 12)
        for cell, name in self._grid_cells():
            hovered = cell.collidepoint(mouse)
            surface.blit(assets.card_face(CARD_DEFS[name].image, cell.size), cell)
            border = S.AEMBER if hovered else S.TEXT_FAINT
            pygame.draw.rect(surface, border, cell, width=2 if hovered else 1, border_radius=6)
            label = name_font.render(name, True, S.TEXT if hovered else S.TEXT_DIM)
            if label.get_width() > cell.width:
                label = pygame.transform.smoothscale(label, (cell.width, label.get_height()))
            surface.blit(label, (cell.left, cell.bottom + 4))
            n = counts.get(name, 0)
            if n:
                badge_font = assets.font("inter", 13, bold=True)
                t = badge_font.render(f"x{n}", True, S.BLACK)
                r = t.get_rect(center=(cell.right - 16, cell.top + 16)).inflate(10, 6)
                pygame.draw.rect(surface, S.AEMBER, r, border_radius=r.height // 2)
                surface.blit(t, t.get_rect(center=r.center))
        max_scroll = self._max_scroll()
        if max_scroll > 0:
            track = pygame.Rect(rect.right - 8, rect.top, 4, rect.height)
            pygame.draw.rect(surface, S.PANEL_LIGHT, track, border_radius=2)
            thumb_h = max(24, int(rect.height / (max_scroll + rect.height / GRID_STRIDE_Y)))
            thumb_y = track.top + int((rect.height - thumb_h) * self.scroll / max_scroll)
            pygame.draw.rect(surface, S.AEMBER, pygame.Rect(track.left, thumb_y, 4, thumb_h), border_radius=2)

    def _draw_pod(self, surface, assets, mouse) -> None:
        rect = self._pod_rect()
        draw_panel(surface, rect, alpha=200, border=S.TEXT_FAINT)
        title = assets.font("cinzel", 16).render(f"{self.active_house.value} pod ({len(self._pod())}/{CARDS_PER_POD})", True, S.TEXT)
        surface.blit(title, (rect.left + 12, rect.top + 10))
        rows = self._pod_rows()
        if not rows:
            hint = assets.font("inter", 14).render("Click a card on the left to add it.", True, S.TEXT_FAINT)
            surface.blit(hint, (rect.left + 12, rect.top + 50))
        row_font = assets.font("inter", 14)
        for row_rect, name, count in rows:
            hovered = row_rect.collidepoint(mouse)
            if hovered:
                pygame.draw.rect(surface, S.PANEL_LIGHT, row_rect, border_radius=4)
            text = f"{name}" + (f"  x{count}" if count > 1 else "")
            label = row_font.render(text, True, S.TEXT)
            if label.get_width() > row_rect.width - 40:
                label = pygame.transform.smoothscale(label, (row_rect.width - 40, label.get_height()))
            surface.blit(label, (row_rect.left + 6, row_rect.centery - label.get_height() // 2))
            remove_rect = pygame.Rect(row_rect.right - 28, row_rect.top + 4, 22, row_rect.height - 8)
            pygame.draw.rect(surface, S.DANGER if remove_rect.collidepoint(mouse) else S.PANEL, remove_rect, border_radius=4)
            x_img = assets.font("inter", 13, bold=True).render("x", True, S.WHITE)
            surface.blit(x_img, x_img.get_rect(center=remove_rect.center))

    def _draw_picker(self, surface, assets, mouse) -> None:
        dim = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 150))
        surface.blit(dim, (0, 0))
        r = self._picker_rect()
        draw_panel(surface, r, alpha=248, border=S.AEMBER)
        title = assets.font("cinzel", 18).render("Load a deck", True, S.TEXT)
        surface.blit(title, (r.left + 16, r.top + 16))
        close = pygame.Rect(r.right - 34, r.top + 8, 26, 26)
        pygame.draw.rect(surface, S.PANEL_LIGHT if close.collidepoint(mouse) else S.PANEL, close, border_radius=6)
        x_img = assets.font("inter", 14, bold=True).render("x", True, S.TEXT)
        surface.blit(x_img, x_img.get_rect(center=close.center))
        rows = self._picker_rows()
        row_font = assets.font("inter", 15)
        for i, (kind, ref, label) in enumerate(rows):
            row = self._picker_row_rect(i)
            if row.top > r.bottom - 30:
                break
            hovered = row.collidepoint(mouse)
            pygame.draw.rect(surface, S.PANEL_LIGHT if hovered else S.PANEL, row, border_radius=6)
            tag = "Preset" if kind == "preset" else "Your deck"
            t = row_font.render(f"{label}  ", True, S.TEXT)
            tag_img = assets.font("inter", S.MIN_FONT).render(tag, True, S.TEXT_FAINT)
            surface.blit(t, (row.left + 10, row.centery - t.get_height() // 2))
            surface.blit(tag_img, (row.left + 10 + t.get_width(), row.centery - tag_img.get_height() // 2))
            if kind == "user":
                delete_rect = pygame.Rect(row.right - 84, row.top + 3, 76, row.height - 6)
                confirming = self.confirm_delete == ref
                pygame.draw.rect(surface, S.DANGER if delete_rect.collidepoint(mouse) or confirming else S.PANEL, delete_rect, border_radius=4)
                dtxt = assets.font("inter", 12, bold=True).render("Confirm?" if confirming else "Delete", True, S.WHITE)
                surface.blit(dtxt, dtxt.get_rect(center=delete_rect.center))
        if not rows:
            t = row_font.render("No decks yet.", True, S.TEXT_FAINT)
            surface.blit(t, t.get_rect(center=(r.centerx, r.top + 120)))

    def _draw_inspector(self, surface, assets) -> None:
        info = self.inspect_info
        dim = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 190))
        surface.blit(dim, (0, 0))
        card_h = 520
        card_w = int(card_h * (S.CARD_ART_W / S.CARD_ART_H))
        panel_w = 360
        gap = 24
        total_w = card_w + gap + panel_w
        left = (S.CANVAS_W - total_w) // 2
        card_rect = pygame.Rect(left, (S.CANVAS_H - card_h) // 2, card_w, card_h)
        surface.blit(assets.card_face(info.image, card_rect.size), card_rect)
        pygame.draw.rect(surface, S.AEMBER, card_rect.inflate(6, 6), width=2, border_radius=10)
        cx = card_rect.centerx
        name = assets.font("cinzel", 20).render(info.name, True, S.TEXT)
        surface.blit(name, name.get_rect(midtop=(cx, card_rect.bottom + 10)))
        sub_txt = f"{info.house} · {info.type}"
        if info.type == "Creature":
            sub_txt += f"   Power {info.power}" + (f"   Armor {info.armor}" if info.armor else "")
        sub = assets.font("inter", 14).render(sub_txt, True, S.TEXT_DIM)
        surface.blit(sub, sub.get_rect(midtop=(cx, card_rect.bottom + 38)))

        text_rect = pygame.Rect(card_rect.right + gap, card_rect.top, panel_w, card_h)
        draw_panel(surface, text_rect, alpha=240, border=S.AEMBER)
        pad = 16
        ty = text_rect.top + pad
        header = assets.font("cinzel", 15).render("Official text", True, S.TEXT)
        surface.blit(header, (text_rect.left + pad, ty))
        ty += header.get_height() + 8
        body_font = assets.font("inter", 14)
        for line in _wrap(info.text or "(no ability text)", body_font, text_rect.width - pad * 2):
            img = body_font.render(line, True, S.TEXT)
            surface.blit(img, (text_rect.left + pad, ty))
            ty += img.get_height() + 4
        if info.errata:
            ty += 12
            badge = assets.font("inter", 12, bold=True).render("Differs from the printed card art:", True, S.AEMBER)
            surface.blit(badge, (text_rect.left + pad, ty))
            ty += badge.get_height() + 4
            note_font = assets.font("inter", 12)
            for line in _wrap(info.errata, note_font, text_rect.width - pad * 2):
                img = note_font.render(line, True, S.TEXT_DIM)
                surface.blit(img, (text_rect.left + pad, ty))
                ty += img.get_height() + 2
        hint = assets.font("inter", S.MIN_FONT).render("Click or Esc to close", True, S.TEXT_FAINT)
        surface.blit(hint, hint.get_rect(midtop=(cx, card_rect.bottom + 64)))
