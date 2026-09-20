"""Regression tests for the usability items in Code/PLAYTEST_FIX_PLAN.md,
each exercised through a real GameScene (U*, W*, M6)."""

import os
import re
import unittest

import tests.helpers  # noqa: F401

import pygame

from bots.heuristic_bot import HeuristicBot
from keyforge.actions import EndTurn, PlayCard
from keyforge.decision import Decision
from keyforge.effects.effect_object import TriggerEffect
from keyforge.enums import DecisionKind, House

from gui import settings as S
from gui.app import App
from gui.engine_bridge import MatchSettings
from gui.scenes.game_over_scene import GameOverScene
from gui.scenes.game_scene import GameScene
from gui.sprites.hud import draw_hud, PlayerHUDState


def _scene(p1="human", p2="bot", seed=5, **kw):
    app = App(window_size=(1600, 900))
    app.push(GameScene(MatchSettings(p1_seat=p1, p2_seat=p2, seed=seed, **kw)))
    return app, app.scenes[-1]


def _run_until(app, g, predicate, answer=None, frames=40000):
    """Update until `predicate(g, decision)` holds at a human decision; other
    human decisions are answered by `answer` (default: heuristic bot)."""
    bot = HeuristicBot(0)
    for _ in range(frames):
        g.update(60)
        app.draw_scenes() if _ % 50 == 0 else None
        if g.animator.is_busy or g.bridge.is_over:
            continue
        d = g.bridge.pending_decision
        if d is None or g.bridge.seat_is_bot(d.player) or g.panel.decision is not d:
            continue
        if predicate(g, d):
            return d
        choice = answer(g, d) if answer else bot.decide(g.bridge.game.view_for(d.player), d)
        g.panel._submit(choice)
    raise AssertionError("condition never reached")


def _synthetic(g, kind, options, prompt="Test", min_n=1, max_n=1):
    d = Decision(g.viewer, kind, prompt, options, min_n, max_n)
    g.panel.reset()
    g.panel.on_enter(d, g.bridge.game.view_for(g.viewer), g.board)
    return d


class TestFirstPlayerAndUnusableCards(unittest.TestCase):
    def test_first_player_is_announced_and_tagged(self):  # U1
        app, g = _scene(first_player=2)
        seen = set()
        tagged = False
        for _ in range(3000):
            g.update(40)
            app.draw_scenes()
            seen.update(b.text for b in g.board.banners)
            tagged = tagged or "tag" in g.board.hud_rects[2]
            d = g.bridge.pending_decision
            if d is not None and not g.bridge.seat_is_bot(d.player) and g.panel.decision is d and not g.animator.is_busy:
                if d.kind == DecisionKind.MULLIGAN:
                    g.panel._submit(False)
                else:
                    break
        self.assertIn("Your opponent goes first", seen)
        self.assertTrue(tagged)

    def test_off_house_hand_cards_are_dimmed_with_a_reason(self):  # U2
        app, g = _scene()
        _run_until(app, g, lambda g, d: d.kind == DecisionKind.CHOOSE_ACTION)
        game = g.bridge.game
        me = game.players[g.viewer]
        off = [c for c in me.hand.cards() if c.house != me.selected_house]
        if not off:
            self.skipTest("no off-house cards in hand")
        sprite = g.board.sprites[off[0].instance_id]
        self.assertIn("active house", sprite.unusable_reason)
        on = [c for c in me.hand.cards() if game.why_not_playable(g.viewer, c) is None]
        for c in on:
            self.assertIsNone(g.board.sprites[c.instance_id].unusable_reason)

    def test_cannot_use_cards_gives_a_specific_reason(self):
        # Skippy Timehog: "Your opponent cannot use any cards next turn."
        app, g = _scene()
        d = _run_until(app, g, lambda g, d: d.kind == DecisionKind.CHOOSE_ACTION and g.bridge.game.players[g.viewer].play_area.creatures)
        game = g.bridge.game
        me = game.players[g.viewer]
        creature = me.play_area.creatures[0]
        creature.Exhausted = False
        g.panel.card_option_map.pop(creature.instance_id, None)  # not currently a legal option
        me.CannotUseCards = True
        try:
            g._sync_unusable(d)
            sprite = g.board.sprites[creature.instance_id]
            self.assertIn("can't be used", sprite.unusable_reason)
        finally:
            me.CannotUseCards = False

    def test_stunned_but_usable_creature_gets_a_hint_not_a_dim(self):
        app, g = _scene()
        d = _run_until(app, g, lambda g, d: d.kind == DecisionKind.CHOOSE_ACTION and g.bridge.game.players[g.viewer].play_area.creatures)
        game = g.bridge.game
        me = game.players[g.viewer]
        creature = me.play_area.creatures[0]
        creature.Exhausted = False
        creature.stunned = True
        g.panel.card_option_map[creature.instance_id] = [object()]  # still a legal (if pointless) option
        try:
            g._sync_unusable(d)
            sprite = g.board.sprites[creature.instance_id]
            self.assertIsNone(sprite.unusable_reason)
            self.assertIn("stunned", sprite.hint)
        finally:
            creature.stunned = False

    def test_artifact_use_toll_gives_a_specific_reason_when_unaffordable(self):
        # Tentacus: "Your opponent must pay you 1Æ in order to use an artifact."
        from keyforge.cards.card import Card
        from keyforge.cards.card_data import get_card_def
        from keyforge.effects.effect_object import DurationEffect, INFINITE
        from gui.sprites.card_sprite import CardSprite

        app, g = _scene()
        d = _run_until(app, g, lambda g, d: d.kind == DecisionKind.CHOOSE_ACTION)
        game = g.bridge.game
        me = game.players[g.viewer]
        artifact = Card(get_card_def("Pocket Universe"), g.viewer)
        me.play_area.add_artifact(artifact)
        artifact.Exhausted = False
        me.aember = 0
        g.board.sprites[artifact.instance_id] = CardSprite(artifact.instance_id, g.app.assets)
        game.active_effects.add(
            DurationEffect(artifact, 3 - g.viewer, INFINITE, g.viewer, "ArtifactUseToll", "=", (1, 3 - g.viewer))
        )
        g._sync_unusable(d)
        sprite = g.board.sprites[artifact.instance_id]
        self.assertIn("Costs", sprite.unusable_reason)

    def test_only_end_turn_left_says_so(self):  # U2
        app, g = _scene()
        _run_until(app, g, lambda g, d: d.kind == DecisionKind.CHOOSE_ACTION)
        _synthetic(g, DecisionKind.CHOOSE_ACTION, [EndTurn()], "Choose an action")
        self.assertIn("No more plays", g.panel.prompt_text())
        g.panel._pick(g.panel.decision.options[0])
        self.assertIsNotNone(g.panel.result)  # no confirm needed when it's the only option


class TestDecisionScreens(unittest.TestCase):
    def setUp(self):
        self.app, self.g = _scene()
        _run_until(self.app, self.g, lambda g, d: d.kind == DecisionKind.CHOOSE_ACTION)

    def test_take_archive_shows_the_archive(self):  # U7
        g = self.g
        me = g.bridge.game.players[g.viewer]
        card = me.hand.cards()[0]
        me.hand.remove(card)
        me.archive.add(card)
        g._resync_board()
        _synthetic(g, DecisionKind.TAKE_ARCHIVE, [True, False], "Take your archive into your hand?")
        self.assertTrue(g.panel.archive_open)
        self.app.draw_scenes()
        self.assertTrue(any(iid == card.instance_id for _r, iid in g.panel.hover_targets))
        take, _ = g.panel._review_buttons()[0]
        g.panel._dispatch_click(take.rect.center, g.board, g.board.layout)
        self.assertIs(g.panel.result, True)

    def test_flank_choice_uses_board_slots(self):  # U11
        g = self.g
        me = g.bridge.game.players[g.viewer]
        if not me.play_area.creatures:
            creature = next(c for c in me.deck.cards() if c.type.value == "Creature")
            me.deck.remove(creature)
            me.play_area.add_creature(creature, "left")
            g._resync_board()
        _synthetic(g, DecisionKind.CHOOSE_FLANK, ["left", "right"], "Choose a flank")
        self.assertTrue(g.panel.flank_open)
        self.assertFalse(g.panel.modal_open)
        right = g.panel.flank_rects(g.board, g.board.layout)["right"]
        g.panel._dispatch_click(right.center, g.board, g.board.layout)
        self.assertEqual(g.panel.result, "right")

    def test_effect_ordering_is_numbered_with_undo(self):  # U10
        g = self.g
        opts = ["effect", "check"]
        d = _synthetic(g, DecisionKind.ORDER_EFFECTS, opts, "Wild Wormhole: choose what resolves first", 2, 2)
        self.assertTrue(g.panel.modal_open)
        labels = [r.label for r in g.panel.modal_rows]
        self.assertTrue(all("effect" not in l or "Play" in l for l in labels), labels)
        g.panel._pick("check")
        self.app.draw_scenes()
        self.assertTrue(any(label.lower() == "undo" for label in (b.text for b, _ in g.panel.action_bar_buttons(g.board.layout))))
        g.panel.undo()
        self.assertEqual(g.panel.picked, [])
        g.panel._pick("check")
        g.panel._pick("effect")
        self.assertEqual(g.panel.result, ["check", "effect"])

    def test_trigger_effect_labels_name_what_they_do(self):  # U10
        g = self.g
        card = g.bridge.game.players[g.viewer].all_cards[0]
        card.name = "Library Access"
        trig = TriggerEffect(card, g.viewer, "card_played", lambda *a: None)
        from gui.option_labels import describe_option

        self.assertEqual(describe_option(trig), "Library Access: draw a card")

    def test_armed_end_turn_disarms_on_any_other_click(self):  # U12
        g = self.g
        end = next(o for o in g.bridge.pending_decision.options if isinstance(o, EndTurn))
        g.panel.on_enter(g.bridge.pending_decision, g.bridge.game.view_for(g.viewer), g.board)
        if len(g.panel.decision.options) == 1:
            self.skipTest("End Turn is the only option")
        g.panel._pick(end)
        self.assertIs(g.panel._armed_action, end)
        g.panel.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(S.PLAY_X + 20, S.BAND_PROMPT[0] + 5), button=1), g.board, (S.PLAY_X + 20, S.BAND_PROMPT[0] + 5), g.board.layout)
        self.assertIsNone(g.panel._armed_action)
        self.assertIsNone(g.panel.result)

    def test_multi_select_shows_a_counter(self):  # U13
        g = self.g
        creatures = g.bridge.game.players[g.viewer].hand.cards()[:3]
        _synthetic(g, DecisionKind.CHOOSE_CARDS, creatures, "Pick cards", 0, 2)
        self.assertIn("(0 of 2 chosen)", g.panel.prompt_text())
        g.panel._pick(creatures[0])
        g.panel.sync_glow(g.board)
        self.assertIn("(1 of 2 chosen)", g.panel.prompt_text())
        self.assertEqual(g.board.sprites[creatures[0].instance_id].pick_number, 1)

    def test_house_rows_explain_each_house(self):  # U8
        g = self.g
        _synthetic(g, DecisionKind.CHOOSE_HOUSE, [House.DIS, House.LOGOS, House.SHADOWS], "Choose your house")
        details = [r.detail for r in g.panel.modal_rows]
        self.assertTrue(all("in hand" in d and "ready in play" in d for d in details), details)


class TestBoardAndOverlays(unittest.TestCase):
    def test_piles_can_be_browsed_during_the_bots_turn(self):  # U14
        app, g = _scene(p1="human", p2="bot", seed=8, first_player=2)
        for _ in range(4000):
            g.update(30)
            d = g.bridge.pending_decision
            if d is not None and g.bridge.seat_is_bot(d.player) and g.bridge.game.players[1].discard.cards() + g.bridge.game.players[2].discard.cards():
                break
            if d is not None and not g.bridge.seat_is_bot(d.player) and g.panel.decision is d and not g.animator.is_busy:
                g.panel._submit(HeuristicBot(1).decide(g.bridge.game.view_for(d.player), d))
        pid = 1 if g.bridge.game.players[1].discard.cards() else 2
        if not g.bridge.game.players[pid].discard.cards():
            self.skipTest("no discard pile yet")
        g.animator.clear()
        rect = g.board.layout.pile_rect(pid, "discard")
        g.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=rect.center, button=1))
        self.assertEqual(g.browsing, (pid, "discard"))

    def test_own_hand_is_sorted_with_the_active_house_first(self):  # U19
        app, g = _scene()
        _run_until(app, g, lambda g, d: d.kind == DecisionKind.CHOOSE_ACTION)
        hand = g.last_snapshot.zone_cards(g.viewer, "hand")
        active = g.bridge.game.players[g.viewer].selected_house.value
        keys = [(cs.house != active, cs.house, cs.type, cs.name) for cs in hand]
        self.assertEqual(keys, sorted(keys))

    def test_hud_chips_collapse_into_a_plus_n_chip(self):  # U20
        app, g = _scene()
        effects = [{"variable": "KeyForgeCost", "op": "+", "value": i, "remaining_duration": 2,
                    "source_name": f"Card {i}", "_house": "Dis"} for i in range(12)]
        rect = g.board.layout.hud_rect(1)
        rects = draw_hud(app.canvas, app.assets, rect, g.last_snapshot.players[1], PlayerHUDState(), True, "Player 1", effects)
        chip_rects = rects["chips"]
        self.assertTrue(all(rect.contains(r) for r, _ in chip_rects))
        last_lines = chip_rects[-1][1]
        self.assertGreater(len(last_lines), 1)  # the "+N" chip lists every hidden effect

    def test_no_font_below_the_minimum(self):  # U21
        gui_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gui")
        offenders = []
        for root, _dirs, files in os.walk(gui_dir):
            for name in files:
                if name.endswith(".py"):
                    with open(os.path.join(root, name), encoding="utf-8") as fh:
                        text = fh.read()
                    for m in re.finditer(r'font\("inter",\s*(\d+)', text):
                        if int(m.group(1)) < S.MIN_FONT:
                            offenders.append((name, m.group(0)))
        self.assertEqual(offenders, [])

    def test_hovering_a_log_line_shows_its_card(self):  # U22
        app, g = _scene()
        _run_until(app, g, lambda g, d: d.kind == DecisionKind.CHOOSE_ACTION and any(t[2] for t in g.log_sentences))
        app.draw_scenes()
        self.assertTrue(g._log_targets)
        rect, iid = g._log_targets[-1]
        app.mouse_canvas = rect.center
        g._update_hover()
        self.assertEqual(g.hover_iid, iid)
        self.assertIsNotNone(g._hover_info)

    def test_right_click_and_zoom_panel_open_the_inspector(self):  # U4
        app, g = _scene()
        _run_until(app, g, lambda g, d: d.kind == DecisionKind.CHOOSE_ACTION)
        cs = next(c for c in g.last_snapshot.zone_cards(g.viewer, "hand") if g.board.card_at((g.board.sprites[c.iid].x, g.board.sprites[c.iid].y)) == c.iid)
        sp = g.board.sprites[cs.iid]
        app.mouse_canvas = (sp.x, sp.y)
        g._update_hover()
        g.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(sp.x, sp.y), button=3))
        self.assertEqual(g.inspect_info.name, cs.name)
        g._close_inspector()
        app.draw_scenes()
        g.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=g.board.layout.zoom_rect().center, button=1))
        self.assertEqual(g.inspect_info.name, cs.name)

    def test_inspector_shows_official_text_and_an_errata_note(self):
        from keyforge.cards.card import Card
        from keyforge.cards.card_data import get_card_def
        from gui.scenes.game_scene import _card_info_from_engine_card

        app, g = _scene()
        _run_until(app, g, lambda g, d: d.kind == DecisionKind.CHOOSE_ACTION)

        errata_card = Card(get_card_def("Ozmo, Martianologist"), 1)
        info = _card_info_from_engine_card(errata_card)
        self.assertTrue(info.text)
        self.assertTrue(info.errata)
        g._open_inspector(info)
        app.draw_scenes()  # must not crash while rendering the text panel

        plain_card = Card(get_card_def("Charette"), 1)
        info2 = _card_info_from_engine_card(plain_card)
        self.assertTrue(info2.text)
        self.assertIsNone(info2.errata)
        g._open_inspector(info2)
        app.draw_scenes()

    def test_click_during_an_animation_is_applied_afterwards(self):  # M6
        app, g = _scene()
        d = _run_until(app, g, lambda g, d: d.kind == DecisionKind.CHOOSE_ACTION and any(len(o) == 1 and isinstance(o[0], PlayCard) for o in g.panel.card_option_map.values()))
        iid = next(i for i, o in g.panel.card_option_map.items() if len(o) == 1 and isinstance(o[0], PlayCard) and g.board.card_at((g.board.sprites[i].x, g.board.sprites[i].y)) == i)
        sp = g.board.sprites[iid]
        from gui.anim.tween import Delay

        g.panel.reset()
        g.animator.enqueue(Delay(200))
        g.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(sp.x, sp.y), button=1))
        self.assertIsNotNone(g._queued_click)
        for _ in range(10):
            g.update(40)
            if g.panel.result is not None or g.bridge.pending_decision is not d:
                break
        played = g.bridge.pending_decision is not d or (g.panel.result is not None and g.panel.result.card.instance_id == iid)
        self.assertTrue(played)


class TestGameOverAndHud(unittest.TestCase):
    def setUp(self):
        self._think = S.T_BOT_THINK
        S.T_BOT_THINK = 1

    def tearDown(self):
        S.T_BOT_THINK = self._think

    def test_hud_matches_the_engine_and_is_never_covered(self):  # W4
        app, g = _scene(p1="bot", p2="bot", seed=31)
        checked = 0
        for frame in range(80000):
            top = app.scenes[-1]
            if isinstance(top, GameOverScene):
                break
            g.update(40)
            if g.animator.is_busy or frame % 7:
                continue
            app.draw_scenes()
            for pid in (1, 2):
                player = g.bridge.game.players[pid]
                self.assertEqual(round(g.board.hud_states[pid].displayed_aember), player.aember)
                self.assertEqual(g.last_snapshot.players[pid].keys, player.keys)
                rects = g.board.hud_rects[pid]
                for r in [rects["aember"]] + [kr for kr, _ in rects["keys"]]:
                    for point in (r.center, r.topleft, r.bottomright):
                        self.assertIsNone(g.board.card_at(point), f"a card covers player {pid}'s HUD at {point}")
                checked += 1
        self.assertGreater(checked, 20)  # a sanity floor: the game length depends on the seed
        go = app.scenes[-1]
        self.assertIsInstance(go, GameOverScene)
        winner = go.result.get("winner")
        if go.result.get("reason") == "3 keys":
            forged = [e for e in go.game.log.events if e.kind == "forge_key" and e.data["player"] == winner]
            self.assertEqual(len(forged), 3)

    def test_game_over_title_is_personal_and_shows_forge_turns(self):  # W2
        app, g = _scene(p1="human", p2="bot", seed=4)
        go = GameOverScene(g.bridge.game, g.settings, g)
        go.result = {"winner": 1, "reason": "3 keys", "turns": 20}
        app.push(go)
        self.assertEqual(go._title(), "You win!")
        go.result = {"winner": 2, "reason": "3 keys", "turns": 20}
        self.assertEqual(go._title(), "You lose")
        app.draw_scenes()
        go.handle_event(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=go.buttons["board"].rect.center, button=1))
        self.assertTrue(go.board_view)
        app.draw_scenes()


class TestLiveSessionFindings(unittest.TestCase):
    """Found by watching screenshots of GUI games as they were played."""

    def test_right_click_inspects_the_card_under_the_click_not_the_last_hover(self):
        app, g = _scene()
        _run_until(app, g, lambda g, d: d.kind == DecisionKind.CHOOSE_ACTION and len(g.last_snapshot.zone_cards(g.viewer, "hand")) >= 2)
        reachable = [c for c in g.last_snapshot.zone_cards(g.viewer, "hand")
                     if g.board.card_at((g.board.sprites[c.iid].x, g.board.sprites[c.iid].y)) == c.iid]
        first, second = reachable[0], reachable[-1]
        a, b = g.board.sprites[first.iid], g.board.sprites[second.iid]
        app.mouse_canvas = (a.x, a.y)
        g._update_hover()  # last frame's hover: the first card
        app.mouse_canvas = (b.x, b.y)  # the mouse moved and clicked before the next frame
        g.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(b.x, b.y), button=3))
        self.assertEqual(g.inspect_info.name, second.name)

    def test_escape_needs_a_second_press_to_leave_a_game(self):
        app, g = _scene()
        _run_until(app, g, lambda g, d: d.kind == DecisionKind.CHOOSE_ACTION)
        esc = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE, mod=0)
        g.handle_event(esc)
        self.assertIs(app.scenes[-1], g)
        self.assertTrue(any("Esc again" in t.text for t in g.board.toasts))
        g.update(4000)  # the prompt expires
        g.handle_event(esc)
        self.assertIs(app.scenes[-1], g)
        g.handle_event(esc)
        self.assertNotIn(g, app.scenes)

    def test_spectating_escape_leaves_at_once(self):
        app, g = _scene(p1="bot", p2="bot")
        g.animator.skip() if g.animator.is_busy else None
        while g.animator.is_busy:
            g.update(5000)
        g.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE, mod=0))
        self.assertNotIn(g, app.scenes)

    def test_new_log_sentences(self):
        from types import SimpleNamespace
        from gui.option_labels import describe_log_event

        def say(kind, viewer=1, **data):
            return describe_log_event(SimpleNamespace(kind=kind, data=data), viewer)

        self.assertEqual(say("duration_effect", card="Miasma", iid=1, variable="CanKeyForge", op="=", value=False, player=2, affected=[1]),
                         "Miasma: You can't forge a key next turn.")
        self.assertEqual(say("duration_effect", viewer=2, card="Miasma", iid=1, variable="CanKeyForge", op="=", value=False, player=2, affected=[1]),
                         "Miasma: Your opponent can't forge a key next turn.")
        self.assertEqual(say("capture_released", card="Old Bruno", iid=1, amount=3, player=1),
                         "Old Bruno leaves play: its 3 captured Æmber goes to you.")
        self.assertEqual(say("gain_chains", card="Gateway to Dis", iid=1, n=3, total=3, player=2),
                         "Your opponent gains 3 chains from Gateway to Dis (3 total).")
        self.assertEqual(say("shed_chain", fewer=1, total=2, player=1),
                         "You draw 1 fewer card because of chains, and shed one (2 left).")


class TestShortfallsAreExplained(unittest.TestCase):
    def test_sentence_resolves_players_for_each_reader(self):
        from types import SimpleNamespace
        from gui.option_labels import describe_log_event, log_event_category

        ev = SimpleNamespace(kind="shortfall", data=dict(
            card="Urchin", iid=7, player=1, reason="steals nothing: {pos:2} Æmber pool is empty", short="Nothing to steal"))
        self.assertEqual(describe_log_event(ev, 1), "Urchin steals nothing: your opponent's Æmber pool is empty.")
        self.assertEqual(describe_log_event(ev, 2), "Urchin steals nothing: your Æmber pool is empty.")
        self.assertEqual(describe_log_event(ev, 1, spectating=True), "Urchin steals nothing: Player 2's Æmber pool is empty.")
        self.assertEqual(log_event_category(ev), "shortfall")

    def test_board_shows_a_lingering_popup(self):
        from keyforge.log import LogEvent
        from gui.anim.director import Director

        app, g = _scene()
        _run_until(app, g, lambda g, d: d.kind == DecisionKind.CHOOSE_ACTION)
        snap = g.last_snapshot
        iid = snap.zone_cards(g.viewer, "hand")[0].iid
        ev = LogEvent("shortfall", dict(card="X", iid=iid, player=1, reason="does nothing: test", short="Nothing to steal"))
        beat = Director.build(g.board, snap, snap, [ev])
        g.animator.enqueue(beat)
        g.update(50)
        texts = [t.text for t in g.board.toasts]
        self.assertIn("Nothing to steal", texts)
        popup = next(t for t in g.board.toasts if t.text == "Nothing to steal")
        self.assertGreaterEqual(popup.life_ms, 2000)


if __name__ == "__main__":
    unittest.main()
