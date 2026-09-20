"""Match play (Reversal / Adaptive): orchestrates a sequence of GameScenes
plus the match-level decisions between games (first-player choice, chain
bidding), and a final Match Over screen. See PHASE_2_PLAN.md v1.2/v1.3.

`MatchScene` owns the single `MatchBridge` for the whole match and decides
what to show next each time control returns to it: a `MatchGameScene` while
a sub-game is being played, a `MatchInterstitialScene` right after one
finishes, a `FirstPlayerScene` or `BiddingScene` for the two new match-level
decisions, or a `MatchOverScene` once the match itself is over.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

import pygame

from .. import settings as S
from ..anim.director import Director
from ..app import Scene
from ..engine_bridge import MatchBridge, MatchSettings
from ..sprites.widgets import Button, draw_panel
from .game_scene import GameScene

from keyforge.cards.decks import deck_label

from keyforge.enums import DecisionKind


class _SubGameBridgeView:
    """Presents one sub-game of a `Match` through the same interface
    `GameScene` expects from an `EngineBridge` -- `.is_over` /
    `.pending_decision` / `.game` describe *this* game, not the whole
    match, so `GameScene` needs no match-awareness at all. `.submit` /
    `.bot_choice` still delegate to the shared `MatchBridge` (and so to the
    `Match`), which is what actually advances match state."""

    def __init__(self, match_bridge: MatchBridge):
        self._mb = match_bridge

    @property
    def game(self):
        return self._mb.game

    @property
    def is_over(self) -> bool:
        g = self._mb.game
        return g is None or g.is_over

    @property
    def pending_decision(self):
        g = self._mb.game
        return None if g is None else g.pending_decision

    @property
    def seats(self):
        return self._mb.seats

    def seat_is_bot(self, pid: int) -> bool:
        return self._mb.seat_is_bot(pid)

    def any_human(self) -> bool:
        return self._mb.any_human()

    def snapshot(self, viewer: int):
        return self._mb.snapshot(viewer)

    def bot_choice(self, pid: int):
        return self._mb.bot_choice(pid)

    def submit(self, choice, viewer: int):
        return self._mb.submit(choice, viewer)

    def close(self, abandoned: bool = True) -> None:
        pass  # the MatchBridge is owned and closed by MatchScene

    @property
    def last_history_id(self):
        return self._mb.last_history_id


class MatchGameScene(GameScene):
    """One game within a Match: reuses all of GameScene's board rendering
    and input handling unchanged, but is driven by the match's shared
    MatchBridge (via `_SubGameBridgeView`) instead of creating its own
    EngineBridge, and hands control back to MatchScene instead of pushing
    the standalone GameOverScene when the sub-game ends."""

    record_history = False  # the MatchBridge already records through the match

    def __init__(self, match_bridge: MatchBridge, settings: MatchSettings, match_scene: "MatchScene"):
        super().__init__(settings)
        self.match_bridge = match_bridge
        self.match_scene = match_scene

    def _make_bridge(self):
        return _SubGameBridgeView(self.match_bridge)

    def on_exit(self) -> None:
        pass  # MatchScene owns and closes the shared MatchBridge

    def _on_game_over(self) -> None:
        if not self._game_over_pushed:
            self._game_over_pushed = True
            self.animator.enqueue(Director.game_over_beat(self.board, self.last_snapshot))
            return
        if not self.animator.is_busy and self.app.scenes and self.app.scenes[-1] is self:
            self.app.pop()
            self.match_scene.child_returned()


class MatchScene(Scene):
    def __init__(self, settings: MatchSettings):
        self.settings = settings
        self.bridge: Optional[MatchBridge] = None
        self._pushed_child = False
        self._interstitial_shown_for = 0

    def on_enter(self) -> None:
        history = self.app.history
        self.bridge = MatchBridge(self.settings, history=history)
        self._advance()

    def on_exit(self) -> None:
        if self.bridge is not None:
            self.bridge.close(abandoned=True)

    def child_returned(self) -> None:
        self._pushed_child = False
        self._advance()

    def _push(self, scene: Scene) -> None:
        self._pushed_child = True
        self.app.push(scene)

    def _advance(self) -> None:
        if self._pushed_child:
            return
        match = self.bridge.match
        if match.is_over:
            self._push(MatchOverScene(self.bridge, self.settings, self))
            return
        finished = len(match.games)
        if finished > self._interstitial_shown_for:
            self._interstitial_shown_for = finished
            self._push(MatchInterstitialScene(self.bridge, self.settings, self))
            return
        d = match.pending_decision
        if d is None:
            return
        if d.kind == DecisionKind.CHOOSE_FIRST_PLAYER:
            self._push(FirstPlayerScene(self.bridge, self.settings, self))
        elif d.kind == DecisionKind.BID_CHAINS:
            self._push(BiddingScene(self.bridge, self.settings, self))
        else:
            self._push(MatchGameScene(self.bridge, self.settings, self))

    def draw(self, surface: pygame.Surface) -> None:
        surface.fill(S.BG_DEEP)


def _panel(rect_size) -> pygame.Rect:
    rect = pygame.Rect(0, 0, *rect_size)
    rect.center = (S.CANVAS_W // 2, S.CANVAS_H // 2)
    return rect


class MatchInterstitialScene(Scene):
    """Between games: the score so far, and a hint at what's next."""

    def __init__(self, bridge: MatchBridge, settings: MatchSettings, match_scene: MatchScene):
        self.bridge = bridge
        self.settings = settings
        self.match_scene = match_scene
        self.button: Optional[Button] = None

    def on_enter(self) -> None:
        rect = _panel((640, 420))
        self.button = Button(pygame.Rect(0, 0, 220, 54), "Continue", primary=True)
        self.button.rect.center = (rect.centerx, rect.bottom - 50)
        # No one to click "Continue" in a bot-vs-bot (or spectated) match --
        # advance on a short timer instead, same as a bot's own "thinking" delay.
        self._auto_timer = S.T_BOT_THINK if not self.bridge.any_human() else None

    def update(self, dt_ms: float) -> None:
        if self._auto_timer is not None:
            self._auto_timer -= dt_ms
            if self._auto_timer <= 0:
                self._continue()

    def _continue(self) -> None:
        self.app.pop()
        self.match_scene.child_returned()

    def handle_event(self, event: pygame.event.Event) -> None:
        self.button.update_hover(self.mouse)
        if self.button.clicked(event):
            self.app.assets.play("click", 0.3)
            self._continue()
        elif event.type == pygame.KEYDOWN and event.key in (pygame.K_RETURN, pygame.K_SPACE, pygame.K_ESCAPE):
            self._continue()

    def draw(self, surface: pygame.Surface) -> None:
        surface.fill(S.BG_DEEP)
        match = self.bridge.match
        rect = _panel((640, 420))
        draw_panel(surface, rect, alpha=248, border=S.KEY_GOLD)

        last = match.games[-1]
        title = self.app.assets.font("cinzel", 30).render(f"Game {len(match.games)} complete", True, S.KEY_GOLD)
        surface.blit(title, title.get_rect(center=(rect.centerx, rect.top + 46)))

        body = self.app.assets.font("inter", 16)
        head = self.app.assets.font("inter", 13, bold=True)
        y = rect.top + 96
        winner_text = (
            f"Player {last.winner} won ({last.reason}, {last.turns} turns)"
            if last.winner else f"No winner ({last.reason})"
        )
        surface.blit(body.render(winner_text, True, S.TEXT), (rect.left + 34, y))
        y += 30
        decks_text = f"Decks this game: P1 {deck_label(last.seat_decks[1]).capitalize()} · P2 {deck_label(last.seat_decks[2]).capitalize()}"
        surface.blit(self.app.assets.font("inter", 13).render(decks_text, True, S.TEXT_DIM), (rect.left + 34, y))
        y += 44

        surface.blit(head.render("Score", True, S.TEXT_FAINT), (rect.left + 34, y))
        y += 24
        score_text = f"Player 1: {match.score.get(1, 0)}      Player 2: {match.score.get(2, 0)}"
        surface.blit(body.render(score_text, True, S.TEXT), (rect.left + 34, y))
        y += 50

        d = match.pending_decision
        if d is not None:
            if d.kind == DecisionKind.CHOOSE_FIRST_PLAYER:
                next_text = f"Next: Player {d.player} (lost that game) chooses who goes first."
            elif d.kind == DecisionKind.BID_CHAINS:
                next_text = f"Next: bidding for {match.bidding_deck}, the deck that won both games."
            else:
                next_text = "Next: the next game begins."
            note = self.app.assets.font("inter", 14).render(next_text, True, S.TEXT_DIM)
            surface.blit(note, (rect.left + 34, y))

        self.button.update_hover(self.mouse)
        self.button.draw(surface, self.app.assets)


class FirstPlayerScene(Scene):
    """The previous game's loser chooses to go first or second next game."""

    def __init__(self, bridge: MatchBridge, settings: MatchSettings, match_scene: MatchScene):
        self.bridge = bridge
        self.settings = settings
        self.match_scene = match_scene
        self.decision = None
        self.buttons = {}
        self._bot_choice = None
        self._bot_timer = 0.0

    def on_enter(self) -> None:
        self.decision = self.bridge.pending_decision
        rect = _panel((560, 260))
        w, h = 200, 54
        self.buttons["first"] = Button(pygame.Rect(rect.centerx - w - 10, rect.bottom - h - 30, w, h), "Go First", primary=True)
        self.buttons["second"] = Button(pygame.Rect(rect.centerx + 10, rect.bottom - h - 30, w, h), "Go Second")
        if self.bridge.seat_is_bot(self.decision.player):
            self._bot_choice = self.bridge.bot_choice(self.decision.player)
            self._bot_timer = S.T_BOT_THINK

    def update(self, dt_ms: float) -> None:
        if self._bot_choice is not None:
            self._bot_timer -= dt_ms
            if self._bot_timer <= 0:
                self._submit(self._bot_choice)

    def _submit(self, choice) -> None:
        self.bridge.submit(choice, self.decision.player)
        self.app.pop()
        self.match_scene.child_returned()

    def handle_event(self, event: pygame.event.Event) -> None:
        if self._bot_choice is not None:
            return
        for key, b in self.buttons.items():
            b.update_hover(self.mouse)
            if b.clicked(event):
                self.app.assets.play("click", 0.3)
                self._submit(key)
                return

    def draw(self, surface: pygame.Surface) -> None:
        surface.fill(S.BG_DEEP)
        rect = _panel((560, 260))
        draw_panel(surface, rect, alpha=248, border=S.TEXT_FAINT)
        title = self.app.assets.font("cinzel", 26).render(f"Player {self.decision.player} lost that game", True, S.TEXT)
        surface.blit(title, title.get_rect(center=(rect.centerx, rect.top + 54)))
        sub = self.app.assets.font("inter", 15).render("Choose whether to go first or second next game.", True, S.TEXT_DIM)
        surface.blit(sub, sub.get_rect(center=(rect.centerx, rect.top + 94)))
        if self._bot_choice is not None:
            note = self.app.assets.font("inter", 14).render("Thinking…", True, S.TEXT_FAINT)
            surface.blit(note, note.get_rect(center=(rect.centerx, rect.top + 130)))
        for b in self.buttons.values():
            b.update_hover(self.mouse)
            b.draw(surface, self.app.assets)


class BiddingScene(Scene):
    """Ascending chain auction for the deck that won both games 1 and 2."""

    def __init__(self, bridge: MatchBridge, settings: MatchSettings, match_scene: MatchScene):
        self.bridge = bridge
        self.settings = settings
        self.match_scene = match_scene
        self.decision = None
        self.buttons = {}
        self.pass_button: Optional[Button] = None
        self._bot_choice = None
        self._bot_timer = 0.0

    def _panel_rect(self) -> pygame.Rect:
        return _panel((8 * 64 + 60, 420))

    def on_enter(self) -> None:
        self.decision = self.bridge.pending_decision
        raises = [o for o in self.decision.options if isinstance(o, int)]
        rect = self._panel_rect()
        cols = 8
        bw, bh, gap = 56, 40, 8
        start_x = rect.centerx - (cols * (bw + gap) - gap) // 2
        top = rect.top + 180
        self.buttons = {}
        for i, n in enumerate(raises):
            col, row = i % cols, i // cols
            x = start_x + col * (bw + gap)
            y = top + row * (bh + gap)
            self.buttons[n] = Button(pygame.Rect(x, y, bw, bh), str(n))
        self.pass_button = Button(pygame.Rect(rect.centerx - 90, rect.bottom - 60, 180, 46), "Pass", danger=True)
        if self.bridge.seat_is_bot(self.decision.player):
            self._bot_choice = self.bridge.bot_choice(self.decision.player)
            self._bot_timer = S.T_BOT_THINK

    def update(self, dt_ms: float) -> None:
        if self._bot_choice is not None:
            self._bot_timer -= dt_ms
            if self._bot_timer <= 0:
                self._submit(self._bot_choice)

    def _submit(self, choice) -> None:
        self.bridge.submit(choice, self.decision.player)
        self.app.pop()
        self.match_scene.child_returned()

    def handle_event(self, event: pygame.event.Event) -> None:
        if self._bot_choice is not None:
            return
        self.pass_button.update_hover(self.mouse)
        if self.pass_button.clicked(event):
            self.app.assets.play("click", 0.3)
            self._submit("pass")
            return
        for n, b in self.buttons.items():
            b.update_hover(self.mouse)
            if b.clicked(event):
                self.app.assets.play("click", 0.3)
                self._submit(n)
                return

    def draw(self, surface: pygame.Surface) -> None:
        surface.fill(S.BG_DEEP)
        rect = self._panel_rect()
        draw_panel(surface, rect, alpha=248, border=S.AEMBER)
        match = self.bridge.match

        title = self.app.assets.font("cinzel", 24).render(f"Bidding for {match.bidding_deck}", True, S.AEMBER)
        surface.blit(title, title.get_rect(center=(rect.centerx, rect.top + 36)))
        sub = self.app.assets.font("inter", 14).render(self.decision.prompt, True, S.TEXT_DIM)
        surface.blit(sub, sub.get_rect(center=(rect.centerx, rect.top + 68)))
        turn_txt = f"Player {self.decision.player} to act" + (" (bot)" if self.bridge.seat_is_bot(self.decision.player) else "")
        turn = self.app.assets.font("inter", 14, bold=True).render(turn_txt, True, S.TEXT)
        surface.blit(turn, turn.get_rect(center=(rect.centerx, rect.top + 94)))
        note = self.app.assets.font("inter", 12).render(
            "More chains taken on shrink your opening hand next game (1 fewer card per 6 chains, rounded up).",
            True, S.TEXT_FAINT,
        )
        surface.blit(note, note.get_rect(center=(rect.centerx, rect.top + 122)))

        for n, b in self.buttons.items():
            b.update_hover(self.mouse)
            b.draw(surface, self.app.assets)
        self.pass_button.update_hover(self.mouse)
        self.pass_button.draw(surface, self.app.assets)

        if self._bot_choice is not None:
            think = self.app.assets.font("inter", 14).render("Thinking…", True, S.TEXT_FAINT)
            surface.blit(think, think.get_rect(center=(rect.centerx, rect.bottom - 96)))


class MatchOverScene(Scene):
    def __init__(self, bridge: MatchBridge, settings: MatchSettings, match_scene: MatchScene):
        self.bridge = bridge
        self.settings = settings
        self.match_scene = match_scene
        self.buttons = {}

    def _panel_rect(self) -> pygame.Rect:
        return _panel((680, 480))

    def on_enter(self) -> None:
        rect = self._panel_rect()
        w, h = 180, 46
        y = rect.bottom - h - 24
        x = rect.left + 24
        for key, label, primary in (("rematch", "New Match", True), ("menu", "Main Menu", False)):
            self.buttons[key] = Button(pygame.Rect(x, y, w, h), label, primary=primary)
            x += w + 12

    def _leave(self) -> None:
        self.app.pop()  # this scene
        if self.app.scenes and self.app.scenes[-1] is self.match_scene:
            self.app.pop()  # the MatchScene

    def handle_event(self, event: pygame.event.Event) -> None:
        for b in self.buttons.values():
            b.update_hover(self.mouse)
        if self.buttons["rematch"].clicked(event):
            self.app.assets.play("click", 0.3)
            self._leave()
            self.app.push(MatchScene(replace(self.settings, seed=None)))
        elif self.buttons["menu"].clicked(event):
            self.app.assets.play("click", 0.3)
            self._leave()
            from .menu_scene import MenuScene

            if not self.app.scenes:
                self.app.push(MenuScene())
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.app.assets.play("click", 0.3)
            self._leave()
            from .menu_scene import MenuScene

            if not self.app.scenes:
                self.app.push(MenuScene())

    def _title(self) -> str:
        result = self.bridge.match.result or {}
        winner = result.get("winner")
        if not winner:
            return "Match undecided"
        humans = [pid for pid in (1, 2) if getattr(self.settings, f"p{pid}_seat") == "human"]
        if len(humans) == 1:
            return "You win the match!" if winner == humans[0] else "You lose the match"
        return f"Player {winner} wins the match!"

    def draw(self, surface: pygame.Surface) -> None:
        surface.fill(S.BG_DEEP)
        rect = self._panel_rect()
        draw_panel(surface, rect, alpha=248, border=S.KEY_GOLD)
        match = self.bridge.match
        cx = rect.centerx

        title = self.app.assets.font("cinzel", 32).render(self._title(), True, S.KEY_GOLD)
        surface.blit(title, title.get_rect(center=(cx, rect.top + 44)))
        fmt_txt = self.app.assets.font("inter", 14).render(f"{match.format.capitalize()} match", True, S.TEXT_DIM)
        surface.blit(fmt_txt, fmt_txt.get_rect(center=(cx, rect.top + 76)))

        head = self.app.assets.font("inter", 13, bold=True)
        body = self.app.assets.font("inter", 15)
        cols = (rect.left + 30, rect.left + 180, rect.left + 320, rect.left + 460)
        y = rect.top + 112
        for x, text in zip(cols, ("Game", "Winner", "Turns", "Decks (P1 / P2)")):
            surface.blit(head.render(text, True, S.TEXT_FAINT), (x, y))
        y += 26
        for i, g in enumerate(match.games, 1):
            won_text = f"Player {g.winner}" if g.winner else "—"
            decks_text = f"{deck_label(g.seat_decks[1]).capitalize()} / {deck_label(g.seat_decks[2]).capitalize()}"
            cells = (f"Game {i}", won_text, str(g.turns), decks_text)
            for x, text in zip(cols, cells):
                surface.blit(body.render(text, True, S.TEXT), (x, y))
            y += 28

        y += 10
        score_text = f"Final score — Player 1: {match.score.get(1, 0)}   Player 2: {match.score.get(2, 0)}"
        surface.blit(body.render(score_text, True, S.KEY_GOLD), (cols[0], y))
        y += 30

        if match.bid is not None:
            b = match.bid
            bid_text = (
                f"Bid: Player {b.winner} won the right to pilot {b.deck} at {b.amount} chains "
                f"(owner: Player {b.owner})"
            )
            surface.blit(self.app.assets.font("inter", 13).render(bid_text, True, S.TEXT_DIM), (cols[0], y))

        for b in self.buttons.values():
            b.update_hover(self.mouse)
            b.draw(surface, self.app.assets)
