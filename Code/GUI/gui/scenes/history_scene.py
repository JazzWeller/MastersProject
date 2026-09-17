"""Past Games: every recorded game, newest first, with replay and delete
(Code/PLAYTEST_FIX_PLAN.md H3)."""

from __future__ import annotations

from typing import List, Optional

import pygame

from .. import settings as S
from ..app import Scene
from ..sprites.widgets import Button, draw_panel

PAGE_SIZE = 12
ROW_H = 46


class HistoryScene(Scene):
    def __init__(self):
        self.page = 0
        self.rows = []
        self.total = 0
        self.selected: Optional[int] = None
        self.confirm_delete: Optional[int] = None
        self.error: Optional[str] = None

    def on_enter(self) -> None:
        self._load()

    def _load(self) -> None:
        history = self.app.history
        if history is None:
            self.error = "The game history database couldn't be opened."
            self.rows, self.total = [], 0
            return
        self.total = history.count()
        max_page = max(0, (self.total - 1) // PAGE_SIZE)
        self.page = max(0, min(self.page, max_page))
        self.rows = history.list(limit=PAGE_SIZE, offset=self.page * PAGE_SIZE)

    # ------------------------------------------------------------ geometry ----

    def _panel(self) -> pygame.Rect:
        return pygame.Rect(120, 90, S.CANVAS_W - 240, 720)

    def _row_rect(self, i: int) -> pygame.Rect:
        p = self._panel()
        return pygame.Rect(p.left + 20, p.top + 70 + i * ROW_H, p.width - 40, ROW_H - 6)

    def _row_buttons(self, i: int) -> List[tuple]:
        r = self._row_rect(i)
        replay = Button(pygame.Rect(r.right - 230, r.top + 3, 110, r.height - 6), "Replay", primary=True)
        confirming = self.confirm_delete == self.rows[i].id
        delete = Button(pygame.Rect(r.right - 112, r.top + 3, 108, r.height - 6), "Confirm?" if confirming else "Delete", danger=True)
        return [("replay", replay), ("delete", delete)]

    def _nav_buttons(self) -> dict:
        p = self._panel()
        y = p.bottom - 60
        return {
            "back": Button(pygame.Rect(p.left + 20, y, 140, 44), "Back"),
            "prev": Button(pygame.Rect(p.centerx - 150, y, 140, 44), "Newer", enabled=self.page > 0),
            "next": Button(pygame.Rect(p.centerx + 10, y, 140, 44), "Older", enabled=(self.page + 1) * PAGE_SIZE < self.total),
        }

    # -------------------------------------------------------------- input ----

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.app.pop()
            return
        if event.type == pygame.MOUSEWHEEL:
            self._page(-event.y)
            return
        if event.type != pygame.MOUSEBUTTONUP or event.button != 1:
            return
        nav = self._nav_buttons()
        if nav["back"].clicked(event):
            self.app.pop()
            return
        if nav["prev"].clicked(event):
            self._page(-1)
            return
        if nav["next"].clicked(event):
            self._page(1)
            return
        for i, row in enumerate(self.rows):
            for kind, button in self._row_buttons(i):
                if not button.clicked(event):
                    continue
                self.app.assets.play("click", 0.3)
                if kind == "replay":
                    self._replay(row.id)
                elif self.confirm_delete == row.id:
                    self.app.history.delete(row.id)
                    self.confirm_delete = None
                    self._load()
                else:
                    self.confirm_delete = row.id
                return
        self.confirm_delete = None

    def _page(self, delta: int) -> None:
        new = self.page + delta
        if new < 0 or new * PAGE_SIZE >= max(self.total, 1):
            return
        self.page = new
        self.confirm_delete = None
        self._load()

    def _replay(self, game_id: int) -> None:
        from .replay_scene import ReplayScene

        try:
            scene = ReplayScene.from_history(self.app.history, game_id)
        except Exception as exc:
            self.error = f"Game #{game_id} couldn't be loaded: {exc}"
            return
        self.app.push(scene)

    # --------------------------------------------------------------- draw ----

    def draw(self, surface: pygame.Surface) -> None:
        surface.fill(S.BG_DEEP)
        p = self._panel()
        draw_panel(surface, p, alpha=230, border=S.TEXT_FAINT)
        title = self.app.assets.font("cinzel", 30).render("Past Games", True, S.TEXT)
        surface.blit(title, (p.left + 20, p.top + 16))
        count = self.app.assets.font("inter", 14).render(
            f"{self.total} recorded game{'s' if self.total != 1 else ''} · page {self.page + 1} of {max(1, -(-self.total // PAGE_SIZE))}",
            True, S.TEXT_DIM,
        )
        surface.blit(count, count.get_rect(topright=(p.right - 20, p.top + 28)))

        head = self.app.assets.font("inter", 13, bold=True)
        cols = (0, 50, 210, 400, 590)
        header_y = p.top + 52
        for x, text in zip(cols, ("#", "Played", "Decks", "Seats", "Result")):
            surface.blit(head.render(text, True, S.TEXT_FAINT), (p.left + 32 + x, header_y))

        body = self.app.assets.font("inter", 15)
        mouse = self.mouse
        if not self.rows:
            msg = self.error or "No games recorded yet. Every game you play is saved here automatically."
            t = body.render(msg, True, S.TEXT_DIM)
            surface.blit(t, t.get_rect(center=(p.centerx, p.top + 200)))
        for i, row in enumerate(self.rows):
            r = self._row_rect(i)
            pygame.draw.rect(surface, S.PANEL_LIGHT if r.collidepoint(mouse) else S.PANEL, r, border_radius=6)
            result_color = S.KEY_GOLD if row.status == "finished" and row.winner else S.TEXT_DIM
            cells = (
                (str(row.id), S.TEXT_DIM),
                (row.started_at.replace("T", " ")[:16], S.TEXT),
                (f"{row.p1_deck.capitalize()} vs {row.p2_deck.capitalize()}", S.TEXT),
                (f"{row.p1_seat.capitalize()} vs {row.p2_seat.capitalize()}", S.TEXT),
                (f"{row.result_text} · {row.turns} turns · keys {row.p1_keys}-{row.p2_keys}", result_color),
            )
            for x, (text, color) in zip(cols, cells):
                img = body.render(text, True, color)
                surface.blit(img, img.get_rect(midleft=(p.left + 32 + x, r.centery)))
            for _kind, button in self._row_buttons(i):
                button.update_hover(mouse)
                button.draw(surface, self.app.assets)

        for button in self._nav_buttons().values():
            button.update_hover(mouse)
            button.draw(surface, self.app.assets)
        if self.error and self.rows:
            t = body.render(self.error, True, S.DANGER)
            surface.blit(t, t.get_rect(midbottom=(p.centerx, p.bottom - 70)))
