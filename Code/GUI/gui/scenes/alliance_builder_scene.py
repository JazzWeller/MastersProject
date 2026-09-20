"""Alliance Builder: assemble an `AllianceDeck` by picking 3 distinct houses
(of the 7, Code/PHASE_3_PLAN.md Milestone B) and, for each, which saved deck
(preset or your own) to take that house's pod from, then preview and save
it (Code/PHASE_2_PLAN.md v2.1 Milestone F). Reached from the main menu,
alongside the single-deck Deck Builder.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import pygame

from .. import settings as S
from ..app import Scene
from ..sprites.widgets import Button, draw_panel
from .deck_builder_scene import _info_from_name, _slugify, _wrap
from .house_picker import HousePicker

from keyforge.cards.decks import (
    DECKS as BUNDLED_DECKS,
    AllianceDeck,
    build_alliance_deck,
    deck_label,
    load_deck_json,
    resolve_deck,
    save_deck_json,
    validate_deck,
)
from keyforge.enums import House

COLUMN_GAP = 20
ROW_H = 26


class AllianceBuilderScene(Scene):
    def __init__(self):
        # A different bundled preset per house by default, so the initial
        # preview is already a real (if arbitrary) alliance once houses are
        # chosen.
        self.name = "New Alliance"
        self.chosen_houses: List[House] = []
        self.house_sources: Dict[House, str] = {}
        self._default_sources = sorted(BUNDLED_DECKS.keys())
        self.saved_path: Optional[str] = None
        self.editing_name = False
        self.name_buffer = ""
        self.message: Optional[str] = None
        self.message_is_error = False
        self.picker_open: Optional[House] = None  # which house's source picker is open
        self.load_picker_open = False
        self.confirm_delete: Optional[str] = None
        self.inspect_info = None
        self.house_picker = HousePicker(self._panel, self._on_houses_confirmed)
        self.house_picker.open([], dismissable=False)

    def _on_houses_confirmed(self, houses: List[House]) -> None:
        self.chosen_houses = houses
        old_sources = self.house_sources
        self.house_sources = {}
        for i, house in enumerate(houses):
            if house in old_sources:
                self.house_sources[house] = old_sources[house]
            elif self._default_sources:
                self.house_sources[house] = self._default_sources[i % len(self._default_sources)]

    def on_enter(self) -> None:
        os.makedirs(S.USER_DECKS_DIR, exist_ok=True)

    def on_exit(self) -> None:
        if self.editing_name:
            pygame.key.stop_text_input()

    # -------------------------------------------------------------- data ----

    def _resolved(self, house: House):
        try:
            return resolve_deck(self.house_sources[house])
        except Exception:
            return None

    def _pod(self, house: House) -> List[str]:
        deck = self._resolved(house)
        if deck is None:
            return []
        return list(deck.pods.get(house, []))

    def _current_deck(self) -> AllianceDeck:
        return build_alliance_deck(self.name, dict(self.house_sources))

    def _errors(self) -> List[str]:
        try:
            return validate_deck(self._current_deck())
        except Exception as exc:
            return [str(exc)]

    def _set_message(self, text: str, error: bool = False) -> None:
        self.message, self.message_is_error = text, error

    def _user_deck_files(self) -> List[str]:
        try:
            files = [f for f in os.listdir(S.USER_DECKS_DIR) if f.endswith(".json")]
        except FileNotFoundError:
            return []
        return sorted(os.path.join(S.USER_DECKS_DIR, f) for f in files)

    def _source_options(self) -> List[tuple]:
        """(kind, ref, label) for every deck a house's pod could be taken
        from: every bundled preset, then every saved user deck."""
        rows = [("preset", key, deck.name) for key, deck in sorted(BUNDLED_DECKS.items(), key=lambda kv: kv[1].name)]
        for path in self._user_deck_files():
            try:
                deck = load_deck_json(path)
            except Exception:
                continue
            rows.append(("user", path, deck_label(deck)))
        return rows

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
        self.chosen_houses = list(deck.houses())
        self.house_sources = {}
        if isinstance(deck, AllianceDeck) and deck.sources:
            # Re-resolve each house's ORIGIN deck by name, falling back to
            # keeping the saved pod's own deck as its own source if the
            # origin can no longer be found (renamed, deleted).
            for house in self.chosen_houses:
                origin_name = deck.sources.get(house)
                match = self._find_source_by_name(origin_name) if origin_name else None
                self.house_sources[house] = match if match is not None else path
        else:
            for house in self.chosen_houses:
                self.house_sources[house] = path
        self.saved_path = path
        self.load_picker_open = False
        self.house_picker.is_open = False
        self._set_message(f"Loaded {deck.name}.")

    def _find_source_by_name(self, name: str) -> Optional[str]:
        for kind, ref, label in self._source_options():
            if label == name:
                return ref
        return None

    def _delete(self, path: str) -> None:
        try:
            os.remove(path)
        except OSError as exc:
            self._set_message(f"Couldn't delete: {exc}", error=True)
            return
        if self.saved_path == path:
            self.saved_path = None
        self.confirm_delete = None

    def _back(self) -> None:
        self.app.pop()
        if self.app.scenes:
            under = self.app.scenes[-1]
            refresh = getattr(under, "_refresh_decks", None)
            if callable(refresh):
                refresh()

    # ------------------------------------------------------------ geometry ----

    def _panel(self) -> pygame.Rect:
        return pygame.Rect(24, 16, S.CANVAS_W - 48, S.CANVAS_H - 32)

    def _name_rect(self) -> pygame.Rect:
        p = self._panel()
        return pygame.Rect(p.left + 20, p.top + 12, 420, 40)

    def _top_buttons(self) -> Dict[str, Button]:
        p = self._panel()
        y = p.top + 12
        return {
            "back": Button(pygame.Rect(p.right - 480, y, 100, 40), "Back"),
            "load": Button(pygame.Rect(p.right - 372, y, 100, 40), "Load"),
            "save": Button(pygame.Rect(p.right - 264, y, 120, 40), "Save", primary=True, enabled=not self._errors()),
            "delete": Button(pygame.Rect(p.right - 136, y, 100, 40), "Delete", danger=True, enabled=self.saved_path is not None),
        }

    def _column_rect(self, i: int) -> pygame.Rect:
        p = self._panel()
        top = p.top + 100
        w = (p.width - 40 - COLUMN_GAP * 2) // 3
        return pygame.Rect(p.left + 20 + i * (w + COLUMN_GAP), top, w, p.bottom - top - 20)

    def _source_button_rect(self, house: House) -> pygame.Rect:
        col = self._column_rect(self.chosen_houses.index(house))
        return pygame.Rect(col.left, col.top + 34, col.width, 34)

    def _pod_row_rects(self, house: House) -> List[pygame.Rect]:
        col = self._column_rect(self.chosen_houses.index(house))
        top = col.top + 80
        return [pygame.Rect(col.left, top + i * ROW_H, col.width, ROW_H - 2) for i in range(len(self._pod(house)))]

    def _houses_button_rect(self) -> pygame.Rect:
        p = self._panel()
        return pygame.Rect(p.right - 604, p.top + 60, 120, 30)

    # -------------------------------------------------------------- input ----

    def handle_event(self, event: pygame.event.Event) -> None:
        if self.editing_name:
            self._handle_name_edit(event)
            return
        if self.house_picker.is_open:
            self.house_picker.handle_event(event)
            return
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            if self.picker_open is not None:
                self.picker_open = None
            elif self.load_picker_open:
                self.load_picker_open = False
                self.confirm_delete = None
            elif self.inspect_info is not None:
                self.inspect_info = None
            else:
                self._back()
            return
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
            self._handle_right_click(event.pos)
            return
        if event.type != pygame.MOUSEBUTTONUP or event.button != 1:
            return
        if self.inspect_info is not None:
            self.inspect_info = None
            return
        if self.picker_open is not None:
            self._handle_source_picker_click(event)
            return
        if self.load_picker_open:
            self._handle_load_picker_click(event)
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
        if self._houses_button_rect().collidepoint(event.pos):
            self.house_picker.open(self.chosen_houses, dismissable=True)
            self.app.assets.play("click", 0.2)
            return
        for house in self.chosen_houses:
            if self._source_button_rect(house).collidepoint(event.pos):
                self.picker_open = house
                return

    def _handle_top_button(self, key: str) -> None:
        self.app.assets.play("click", 0.3)
        if key == "back":
            self._back()
        elif key == "load":
            self.load_picker_open = True
        elif key == "save":
            self._save()
        elif key == "delete" and self.saved_path is not None:
            self._delete(self.saved_path)

    def _handle_right_click(self, pos) -> None:
        for house in self.chosen_houses:
            for row_rect, name in zip(self._pod_row_rects(house), self._pod(house)):
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
                self.name = self.name_buffer.strip() or "New Alliance"
                self.editing_name = False
                pygame.key.stop_text_input()
            elif event.key == pygame.K_ESCAPE:
                self.editing_name = False
                pygame.key.stop_text_input()

    # ------------------------------------------------------- source picker ----

    def _picker_rect(self) -> pygame.Rect:
        p = self._panel()
        w, h = 560, 520
        return pygame.Rect(p.centerx - w // 2, p.centery - h // 2, w, h)

    def _picker_row_rect(self, i: int) -> pygame.Rect:
        r = self._picker_rect()
        return pygame.Rect(r.left + 16, r.top + 60 + i * 40, r.width - 32, 34)

    def _handle_source_picker_click(self, event: pygame.event.Event) -> None:
        house = self.picker_open
        r = self._picker_rect()
        close = pygame.Rect(r.right - 34, r.top + 8, 26, 26)
        if close.collidepoint(event.pos) or not r.collidepoint(event.pos):
            self.picker_open = None
            return
        for i, (kind, ref, _label) in enumerate(self._source_options()):
            if self._picker_row_rect(i).collidepoint(event.pos):
                self.house_sources[house] = ref
                self.picker_open = None
                self._set_message(f"{house.value}'s pod now comes from {deck_label(resolve_deck(ref))}.")
                return

    def _handle_load_picker_click(self, event: pygame.event.Event) -> None:
        r = self._picker_rect()
        close = pygame.Rect(r.right - 34, r.top + 8, 26, 26)
        if close.collidepoint(event.pos):
            self.load_picker_open = False
            return
        if not r.collidepoint(event.pos):
            self.load_picker_open = False
            return
        for i, path in enumerate(self._user_deck_files()):
            row = self._picker_row_rect(i)
            if not row.collidepoint(event.pos):
                continue
            delete_rect = pygame.Rect(row.right - 84, row.top + 2, 76, row.height - 4)
            if delete_rect.collidepoint(event.pos):
                if self.confirm_delete == path:
                    self._delete(path)
                else:
                    self.confirm_delete = path
                return
            self._load(path)
            return

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
            hint = assets.font("inter", S.MIN_FONT).render("Click to rename · an Alliance deck", True, S.TEXT_FAINT)
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
        surface.blit(status, (p.left + 20, p.top + 60))

        hb = self._houses_button_rect()
        hovered = hb.collidepoint(mouse)
        pygame.draw.rect(surface, S.PANEL_LIGHT if hovered else S.PANEL, hb, border_radius=6)
        pygame.draw.rect(surface, S.AEMBER if hovered else S.TEXT_FAINT, hb, width=1, border_radius=6)
        hlabel = assets.font("inter", 13).render("Change houses", True, S.TEXT)
        surface.blit(hlabel, hlabel.get_rect(center=hb.center))

        for i, house in enumerate(self.chosen_houses):
            self._draw_column(surface, assets, mouse, house, i)

        if self.message:
            color = S.DANGER if self.message_is_error else S.TEXT_DIM
            m = assets.font("inter", 14).render(self.message, True, color)
            surface.blit(m, m.get_rect(bottomleft=(p.left + 20, p.bottom - 12)))

        if self.picker_open is not None:
            self._draw_source_picker(surface, assets, mouse)
        if self.load_picker_open:
            self._draw_load_picker(surface, assets, mouse)
        if self.inspect_info is not None:
            self._draw_inspector(surface, assets)
        self.house_picker.draw(surface, assets, mouse)

    def _draw_column(self, surface, assets, mouse, house: House, i: int) -> None:
        col = self._column_rect(i)
        draw_panel(surface, col, alpha=210, border=S.HOUSE_COLORS.get(house.value, S.TEXT_FAINT))
        head = assets.font("cinzel", 18).render(house.value, True, S.TEXT)
        surface.blit(head, (col.left + 12, col.top + 6))

        btn = self._source_button_rect(house)
        hovered = btn.collidepoint(mouse)
        pygame.draw.rect(surface, S.PANEL_LIGHT if hovered else S.PANEL, btn, border_radius=6)
        pygame.draw.rect(surface, S.AEMBER if hovered else S.TEXT_FAINT, btn, width=1, border_radius=6)
        source_label = deck_label(self._resolved(house)) if self._resolved(house) is not None else "(missing)"
        label = assets.font("inter", 14, bold=True).render(f"Source: {source_label}", True, S.TEXT)
        surface.blit(label, label.get_rect(midleft=(btn.left + 10, btn.centery)))

        row_font = assets.font("inter", 13)
        pod = self._pod(house)
        for row_rect, name in zip(self._pod_row_rects(house), pod):
            row_hovered = row_rect.collidepoint(mouse)
            if row_hovered:
                pygame.draw.rect(surface, S.PANEL_LIGHT, row_rect, border_radius=4)
            img = row_font.render(name, True, S.TEXT if row_hovered else S.TEXT_DIM)
            if img.get_width() > row_rect.width - 12:
                img = pygame.transform.smoothscale(img, (row_rect.width - 12, img.get_height()))
            surface.blit(img, (row_rect.left + 6, row_rect.centery - img.get_height() // 2))
        if not pod:
            hint = assets.font("inter", 13).render("Source deck not found.", True, S.DANGER)
            surface.blit(hint, (col.left + 12, col.top + 84))
        count_txt = assets.font("inter", S.MIN_FONT).render(f"{len(pod)}/12 cards", True, S.TEXT_FAINT)
        surface.blit(count_txt, count_txt.get_rect(bottomright=(col.right - 10, col.bottom - 6)))

    def _draw_source_picker(self, surface, assets, mouse) -> None:
        dim = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 150))
        surface.blit(dim, (0, 0))
        r = self._picker_rect()
        draw_panel(surface, r, alpha=248, border=S.AEMBER)
        title = assets.font("cinzel", 18).render(f"Pick {self.picker_open.value}'s source deck", True, S.TEXT)
        surface.blit(title, (r.left + 16, r.top + 16))
        close = pygame.Rect(r.right - 34, r.top + 8, 26, 26)
        pygame.draw.rect(surface, S.PANEL_LIGHT if close.collidepoint(mouse) else S.PANEL, close, border_radius=6)
        x_img = assets.font("inter", 14, bold=True).render("x", True, S.TEXT)
        surface.blit(x_img, x_img.get_rect(center=close.center))
        row_font = assets.font("inter", 15)
        for i, (kind, ref, label) in enumerate(self._source_options()):
            row = self._picker_row_rect(i)
            if row.top > r.bottom - 30:
                break
            hovered = row.collidepoint(mouse)
            current = self.house_sources.get(self.picker_open) == ref
            bg = S.AEMBER if current else (S.PANEL_LIGHT if hovered else S.PANEL)
            fg = S.BLACK if current else S.TEXT
            pygame.draw.rect(surface, bg, row, border_radius=6)
            tag = "Preset" if kind == "preset" else "Your deck"
            t = row_font.render(f"{label}  ", True, fg)
            surface.blit(t, (row.left + 10, row.centery - t.get_height() // 2))
            tag_img = assets.font("inter", S.MIN_FONT).render(tag, True, S.BLACK if current else S.TEXT_FAINT)
            surface.blit(tag_img, (row.left + 10 + t.get_width(), row.centery - tag_img.get_height() // 2))

    def _draw_load_picker(self, surface, assets, mouse) -> None:
        dim = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 150))
        surface.blit(dim, (0, 0))
        r = self._picker_rect()
        draw_panel(surface, r, alpha=248, border=S.AEMBER)
        title = assets.font("cinzel", 18).render("Load a saved alliance/deck", True, S.TEXT)
        surface.blit(title, (r.left + 16, r.top + 16))
        close = pygame.Rect(r.right - 34, r.top + 8, 26, 26)
        pygame.draw.rect(surface, S.PANEL_LIGHT if close.collidepoint(mouse) else S.PANEL, close, border_radius=6)
        x_img = assets.font("inter", 14, bold=True).render("x", True, S.TEXT)
        surface.blit(x_img, x_img.get_rect(center=close.center))
        files = self._user_deck_files()
        row_font = assets.font("inter", 15)
        if not files:
            t = row_font.render("No saved decks yet.", True, S.TEXT_FAINT)
            surface.blit(t, t.get_rect(center=(r.centerx, r.top + 120)))
        for i, path in enumerate(files):
            row = self._picker_row_rect(i)
            if row.top > r.bottom - 30:
                break
            hovered = row.collidepoint(mouse)
            pygame.draw.rect(surface, S.PANEL_LIGHT if hovered else S.PANEL, row, border_radius=6)
            try:
                label = deck_label(load_deck_json(path))
            except Exception:
                label = os.path.basename(path)
            t = row_font.render(label, True, S.TEXT)
            surface.blit(t, (row.left + 10, row.centery - t.get_height() // 2))
            delete_rect = pygame.Rect(row.right - 84, row.top + 2, 76, row.height - 4)
            confirming = self.confirm_delete == path
            pygame.draw.rect(surface, S.DANGER if delete_rect.collidepoint(mouse) or confirming else S.PANEL, delete_rect, border_radius=4)
            dtxt = assets.font("inter", 12, bold=True).render("Confirm?" if confirming else "Delete", True, S.WHITE)
            surface.blit(dtxt, dtxt.get_rect(center=delete_rect.center))

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
