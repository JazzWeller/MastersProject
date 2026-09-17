"""Step-by-step replay of a recorded game (Code/PLAYTEST_FIX_PLAN.md H4).

A `GameScene` whose bridge is a `ReplayBridge`: every seat is a "replay"
seat, so nothing asks for input, and the recorded choice for each decision
is fed in only when the viewer steps forward or presses play. Stepping
backwards (or jumping) rebuilds the game from the start instantly -- the
engine replays thousands of decisions per second -- and snaps the board.
"""

from __future__ import annotations

from typing import List, Optional

import pygame

from .. import settings as S
from ..engine_bridge import MatchSettings, ReplayBridge
from ..option_labels import describe_option
from ..sprites.overlays import Banner, Toast
from ..sprites.widgets import Button, draw_panel
from .game_scene import GameScene

PLAY_STEP_MS = 650


class ReplayScene(GameScene):
    record_history = False

    def __init__(self, config, record: List, settings: Optional[MatchSettings] = None, title: str = "Replay"):
        super().__init__(settings or MatchSettings())
        self._config = config
        self._record = record
        self._label_settings = settings
        self.title = title
        self.playing = False
        self._step_request = 0
        self._play_timer = 0.0
        self.reveal_hands = True
        self._ended_announced = False

    @classmethod
    def from_history(cls, history, game_id: int) -> "ReplayScene":
        config, record = history.load(game_id)
        summary = history.get(game_id)
        settings = MatchSettings(
            p1_deck=summary.p1_deck, p2_deck=summary.p2_deck,
            p1_seat=summary.p1_seat, p2_seat=summary.p2_seat,
            first_player=config.first_player, seed=config.seed, max_turns=config.max_turns,
        )
        title = f"Replay · Game #{game_id} · {summary.started_at.replace('T', ' ')}"
        return cls(config, record, settings, title)

    def _make_bridge(self):
        return ReplayBridge(self._config, self._record)

    def on_enter(self) -> None:
        super().on_enter()
        self._announced_first = True  # the banner would repeat on every rebuild

    # ------------------------------------------------------------ stepping ----

    def _step_bot(self, dt_ms: float) -> None:
        bridge: ReplayBridge = self.bridge
        if bridge.at_end:
            self.playing = False
            if not self._ended_announced:
                self._ended_announced = True
                self.board.banners.append(Banner("End of recording", "This game was not finished." if not bridge.is_over else "", life_ms=2500))
            return
        if self.playing:
            self._play_timer -= dt_ms
            if self._play_timer > 0:
                return
            self._play_timer = PLAY_STEP_MS
        elif self._step_request <= 0:
            return
        else:
            self._step_request -= 1
        d = bridge.pending_decision
        choice = bridge.bot_choice(d.player)
        self._preview_bot_choice(d, choice)
        self._submit(choice)

    def _on_game_over(self) -> None:
        self.playing = False
        if not self._game_over_pushed:
            self._game_over_pushed = True
            result = self.bridge.game.result or {}
            winner = result.get("winner")
            text = f"Player {winner} won" if winner else "Draw"
            self.board.banners.append(Banner(text, f"by {result.get('reason', '')}", color=S.KEY_GOLD, life_ms=3000))

    def jump_to(self, position: int) -> None:
        bridge: ReplayBridge = self.bridge
        position = max(0, min(position, bridge.length))
        self.animator.clear()
        self.board.toasts.clear()
        self.board.banners.clear()
        self.playing = False
        self._step_request = 0
        self._bot_pending_choice = None
        bridge.rebuild_to(position)
        self._game_over_pushed = False
        self._ended_announced = False
        self.panel.reset()
        self._resync_board()

    def step(self, n: int = 1) -> None:
        if n > 0:
            if self.animator.is_busy:
                self.animator.skip()
            self._step_request += n
        else:
            self.jump_to(self.bridge.position + n)

    def _turn_jump(self, direction: int) -> None:
        starts = self.bridge.turn_start_positions()
        pos = self.bridge.position
        if direction > 0:
            target = next((s for s in starts if s > pos), self.bridge.length)
        else:
            earlier = [s for s in starts if s < pos]
            target = earlier[-1] if earlier else 0
        self.jump_to(target)

    def toggle_play(self) -> None:
        if self.bridge.at_end:
            self.jump_to(0)
        self.playing = not self.playing
        self._play_timer = 0

    def switch_perspective(self) -> None:
        self.viewer = 3 - self.viewer
        self.board.set_viewer(self.viewer)
        self.animator.skip()
        self._resync_board()

    # -------------------------------------------------------------- input ----

    def _control_buttons(self) -> List[tuple]:
        rect = self.board.layout.action_bar_rect()
        specs = [
            ("|<", lambda: self.jump_to(0), 50),
            ("< Turn", lambda: self._turn_jump(-1), 80),
            ("< Step", lambda: self.step(-1), 80),
            ("Pause" if self.playing else "Play", self.toggle_play, 90),
            ("Step >", lambda: self.step(1), 80),
            ("Turn >", lambda: self._turn_jump(1), 80),
            (">|", lambda: self.jump_to(self.bridge.length), 50),
            (f"View as P{3 - self.viewer}", self.switch_perspective, 110),
            ("Hide hands" if self.reveal_hands else "Show hands", self._toggle_reveal, 110),
        ]
        buttons = []
        x = rect.left
        for label, cb, w in specs:
            buttons.append((Button(pygame.Rect(x, rect.top, w, rect.height), label, primary=label in ("Play", "Pause")), cb))
            x += w + 6
        return buttons

    def _toggle_reveal(self) -> None:
        self.reveal_hands = not self.reveal_hands
        self._sync_reveal_hands()

    def _scrub_rect(self) -> pygame.Rect:
        rect = self.board.layout.action_bar_rect()
        left = rect.left + 860
        return pygame.Rect(left, rect.centery - 5, rect.right - left - 8, 10)

    def handle_event(self, event: pygame.event.Event) -> None:
        if self.inspect_info is None and self.browsing is None and self.decklist_pid is None and not self.show_help:
            if event.type == pygame.KEYDOWN:
                keys = {
                    pygame.K_SPACE: self.toggle_play,
                    pygame.K_RIGHT: lambda: self.step(1),
                    pygame.K_LEFT: lambda: self.step(-1),
                    pygame.K_PAGEDOWN: lambda: self._turn_jump(1),
                    pygame.K_DOWN: lambda: self._turn_jump(1),
                    pygame.K_PAGEUP: lambda: self._turn_jump(-1),
                    pygame.K_UP: lambda: self._turn_jump(-1),
                    pygame.K_HOME: lambda: self.jump_to(0),
                    pygame.K_END: lambda: self.jump_to(self.bridge.length),
                    pygame.K_v: self.switch_perspective,
                }
                if event.key in keys:
                    keys[event.key]()
                    return
                if event.key == pygame.K_ESCAPE:
                    self.app.pop()
                    return
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                fake_up = pygame.event.Event(pygame.MOUSEBUTTONUP, pos=event.pos, button=1)
                for button, cb in self._control_buttons():
                    if button.clicked(fake_up):
                        self.app.assets.play("click", 0.25)
                        cb()
                        return
                scrub = self._scrub_rect().inflate(0, 20)
                if scrub.collidepoint(event.pos) and self.bridge.length:
                    frac = (event.pos[0] - scrub.left) / max(1, scrub.width)
                    self.jump_to(round(frac * self.bridge.length))
                    return
        super().handle_event(event)

    # --------------------------------------------------------------- draw ----

    def draw(self, surface: pygame.Surface) -> None:
        super().draw(surface)
        if self.inspect_info is not None:
            return
        # Replace the prompt band with replay status.
        prompt = self.board.layout.prompt_rect()
        cover = pygame.Rect(prompt.left + 330, prompt.top, prompt.width - 330, prompt.height)
        draw_panel(surface, cover, alpha=230, radius=8)
        bridge: ReplayBridge = self.bridge
        d = bridge.pending_decision
        font = self.app.assets.font("inter", 15, bold=True)
        if d is not None and not bridge.at_end:
            nxt = describe_option(bridge.bot_choice(d.player), d, bridge.game.view_for(d.player))
            status = f"Next: Player {d.player} — {d.prompt}: {nxt}"
        elif bridge.is_over:
            result = bridge.game.result or {}
            status = f"Game over — " + (f"Player {result.get('winner')} won by {result.get('reason')}" if result.get("winner") else "draw")
        else:
            status = "End of recording (game unfinished)"
        img = font.render(status, True, S.TEXT)
        if img.get_width() > cover.width - 20:
            img = self.app.assets.font("inter", 13, bold=True).render(status, True, S.TEXT)
        surface.blit(img, img.get_rect(midleft=(cover.left + 12, cover.centery)))

        # Title strip over the action-bar hint text.
        bar = self.board.layout.action_bar_rect()
        pygame.draw.rect(surface, S.FELT_BOTTOM, bar)
        for button, _cb in self._control_buttons():
            button.update_hover(self.mouse)
            button.draw(surface, self.app.assets)
        scrub = self._scrub_rect()
        pygame.draw.rect(surface, S.PANEL_LIGHT, scrub, border_radius=5)
        if bridge.length:
            frac = bridge.position / bridge.length
            pygame.draw.rect(surface, S.AEMBER, pygame.Rect(scrub.left, scrub.top, int(scrub.width * frac), scrub.height), border_radius=5)
            for s in bridge.turn_start_positions()[1:-1]:
                x = scrub.left + int(scrub.width * s / bridge.length)
                pygame.draw.line(surface, S.TEXT_FAINT, (x, scrub.top - 3), (x, scrub.bottom + 3))
        label = self.app.assets.font("inter", S.MIN_FONT).render(
            f"{self.title}   ·   decision {bridge.position} / {bridge.length}   ·   Space play · ←/→ step · ↑/↓ turn · V switch view · Esc back",
            True, S.TEXT_DIM,
        )
        surface.blit(label, label.get_rect(midbottom=(bar.centerx, bar.top - 2)))

    def _draw_action_hints(self, surface) -> None:
        pass
