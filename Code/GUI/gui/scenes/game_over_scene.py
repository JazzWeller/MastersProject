"""Game over: drawn *over* the final board (not instead of it), with how the
game was won -- each player's keys, Æmber, and the turn every key was
forged -- plus View Board, Watch Replay, Rematch and Main Menu.
See Code/PLAYTEST_FIX_PLAN.md W2 / U15."""

from __future__ import annotations

from typing import Optional

import pygame

from .. import settings as S
from ..app import Scene
from ..engine_bridge import MatchSettings
from ..sprites.widgets import Button, draw_panel


class GameOverScene(Scene):
    overlay = True  # App draws the finished GameScene underneath

    def __init__(self, game, settings: MatchSettings, game_scene=None):
        self.game = game
        self.result = (game.result if game is not None else None) or {}
        self.settings = settings
        self.game_scene = game_scene
        self.buttons = {}
        self.board_view = False

    # -------------------------------------------------------------- setup ----

    def _panel_rect(self) -> pygame.Rect:
        rect = pygame.Rect(0, 0, 620, 400)
        rect.center = (S.PLAY_X + S.PLAY_W // 2, S.CANVAS_H // 2)
        return rect

    def on_enter(self) -> None:
        rect = self._panel_rect()
        w, h = 136, 46
        y = rect.bottom - h - 22
        x = rect.left + 22
        names = [("board", "View Board", False), ("replay", "Watch Replay", False), ("rematch", "Rematch", True), ("menu", "Main Menu", False)]
        for key, label, primary in names:
            self.buttons[key] = Button(pygame.Rect(x, y, w, h), label, primary=primary)
            x += w + 10
        self.buttons["replay"].enabled = self._history_id() is not None

    def _history_id(self) -> Optional[int]:
        bridge = getattr(self.game_scene, "bridge", None)
        return getattr(bridge, "last_history_id", None)

    # -------------------------------------------------------------- input ----

    def handle_event(self, event: pygame.event.Event) -> None:
        if self.board_view:
            if event.type == pygame.MOUSEBUTTONDOWN or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                self.board_view = False
            return
        for b in self.buttons.values():
            b.update_hover(self.mouse)
        if self.buttons["board"].clicked(event):
            self.board_view = True
        elif self.buttons["replay"].clicked(event):
            self._watch_replay()
        elif self.buttons["rematch"].clicked(event):
            from .game_scene import GameScene

            self.app.assets.play("click", 0.3)
            self._leave_game()
            self.app.push(GameScene(MatchSettings(**{**self.settings.__dict__, "seed": None})))
        elif self.buttons["menu"].clicked(event):
            from .menu_scene import MenuScene

            self.app.assets.play("click", 0.3)
            self._leave_game()
            if not self.app.scenes:
                self.app.push(MenuScene())
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.board_view = True

    def _leave_game(self) -> None:
        self.app.pop()  # this scene
        if self.app.scenes and self.app.scenes[-1] is self.game_scene:
            self.app.pop()  # the finished GameScene

    def _watch_replay(self) -> None:
        from .replay_scene import ReplayScene

        game_id = self._history_id()
        if game_id is None or self.app.history is None:
            return
        self.app.assets.play("click", 0.3)
        self._leave_game()
        self.app.push(ReplayScene.from_history(self.app.history, game_id))

    # --------------------------------------------------------------- draw ----

    def _title(self) -> str:
        winner = self.result.get("winner")
        if not winner:
            return "It's a draw"
        humans = [pid for pid in (1, 2) if getattr(self.settings, f"p{pid}_seat") == "human"]
        if len(humans) == 1:
            return "You win!" if winner == humans[0] else "You lose"
        return f"Player {winner} wins!"

    def draw(self, surface: pygame.Surface) -> None:
        if self.board_view:
            hint = self.app.assets.font("inter", 14, bold=True).render("Viewing the final board — click anywhere to return", True, S.BLACK)
            r = hint.get_rect().inflate(24, 12)
            r.midtop = (S.PLAY_X + S.PLAY_W // 2, S.BAND_PROMPT[0])
            pygame.draw.rect(surface, S.AEMBER_GLOW, r, border_radius=r.height // 2)
            surface.blit(hint, hint.get_rect(center=r.center))
            return

        dim = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 150))
        surface.blit(dim, (0, 0))
        rect = self._panel_rect()
        draw_panel(surface, rect, alpha=248, border=S.KEY_GOLD)
        cx = rect.centerx

        title = self.app.assets.font("cinzel", 32).render(self._title(), True, S.KEY_GOLD)
        surface.blit(title, title.get_rect(center=(cx, rect.top + 44)))
        reason = self.result.get("reason", "")
        turns = self.result.get("turns")
        sub = self.app.assets.font("inter", 15).render(
            (f"By {reason}" if reason else "") + (f" · {turns} turns played" if turns else ""), True, S.TEXT_DIM
        )
        surface.blit(sub, sub.get_rect(center=(cx, rect.top + 82)))

        if self.game is not None:
            head = self.app.assets.font("inter", 13, bold=True)
            body = self.app.assets.font("inter", 15)
            cols = (rect.left + 30, rect.left + 250, rect.left + 330, rect.left + 420)
            y = rect.top + 118
            for x, text in zip(cols, ("Player", "Keys", "Æmber", "Forged on turns")):
                surface.blit(head.render(text, True, S.TEXT_FAINT), (x, y))
            y += 26
            for pid in (1, 2):
                player = self.game.players[pid]
                forged = [str(e.data.get("turn", "?")) for e in self.game.log.events if e.kind == "forge_key" and e.data.get("player") == pid]
                seat = getattr(self.settings, f"p{pid}_seat")
                deck = getattr(self.settings, f"p{pid}_deck").capitalize()
                won = self.result.get("winner") == pid
                color = S.KEY_GOLD if won else S.TEXT
                cells = (f"Player {pid} ({seat}, {deck})", f"{player.keys} / 3", str(player.aember), ", ".join(forged) or "—")
                for x, text in zip(cols, cells):
                    surface.blit(body.render(text, True, color), (x, y))
                y += 30
            if self.game.first_player:
                note = self.app.assets.font("inter", 13).render(f"Player {self.game.first_player} went first.", True, S.TEXT_FAINT)
                surface.blit(note, (cols[0], y + 6))

        for b in self.buttons.values():
            b.update_hover(self.mouse)
            b.draw(surface, self.app.assets)
