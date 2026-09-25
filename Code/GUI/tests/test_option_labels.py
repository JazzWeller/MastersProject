"""Every option type in gui/decision/panel.py's vocabulary, and every log
event kind the engine actually emits, must produce a non-empty label."""

import unittest

import tests.helpers  # noqa: F401

from bots.random_bot import RandomBot
from keyforge.actions import DiscardCard, EndTurn, Fight, PlayCard, Reap, UseAction, UseOmni
from keyforge.config import GameConfig
from keyforge.decision import Decision
from keyforge.enums import DecisionKind, House
from keyforge.game import Game

from keyforge.log import LogEvent

from gui.option_labels import describe_log_event, describe_option


def _first_creature(game, pid):
    for p in game.players.values():
        if p.id == pid:
            for c in p.hand.cards() + list(p.play_area.creatures):
                if c.type.value == "Creature":
                    return c
    return None


class TestOptionLabels(unittest.TestCase):
    def test_static_option_types(self):
        game = Game(GameConfig(decks=("fignor", "igor"), seed=1))
        card = None
        for p in game.players.values():
            if p.hand.cards():
                card = p.hand.cards()[0]
                break
        self.assertIsNotNone(card)

        samples = [
            EndTurn(),
            PlayCard(card),
            DiscardCard(card),
            UseAction(card),
            UseOmni(card),
            Reap(card),
            Fight(card),
            card,
            House.DIS,
            House.LOGOS,
            House.SHADOWS,
            True,
            False,
            "reap",
            "fight",
            "action",
            "effect",
            "check",
            "left",
            "right",
            1,
            2,
        ]
        for opt in samples:
            label = describe_option(opt)
            self.assertIsInstance(label, str)
            self.assertTrue(label.strip(), f"empty label for {opt!r}")

    def test_choose_number_options_are_not_misread_as_player_ids(self):
        # Regression: describe_option's generic "int -> discard pile owner"
        # branch (Creeping Oblivion's CHOOSE_CARDS pile choice, options 1/2)
        # must not swallow CHOOSE_NUMBER's raw power values (Dance of Doom
        # can offer any power in play, e.g. 7) -- with a real view passed
        # (as the decision panel always does), that used to raise a KeyError
        # doing view.players[7].
        game = Game(GameConfig(decks=("fignor", "igor"), seed=1))
        view = game.view_for(1)
        decision = Decision(player=1, kind=DecisionKind.CHOOSE_NUMBER, prompt="Dance of Doom: choose a number", options=[2, 4, 7])
        for opt in decision.options:
            label = describe_option(opt, decision, view)
            self.assertEqual(label, str(opt))

    def test_choose_mode_options_render_their_own_text(self):
        decision = Decision(
            player=1, kind=DecisionKind.CHOOSE_MODE, prompt="Knowledge is Power: choose one",
            options=["Archive a card", "Gain 1Æ per archived card"],
        )
        for opt in decision.options:
            self.assertEqual(describe_option(opt, decision), opt)

    def test_every_log_event_kind_has_a_sentence_or_is_intentionally_silent(self):
        # Kinds describe_log_event deliberately returns None for (nothing
        # interesting to tell a player) — everything else must render.
        silent_ok = {
            "turn_start",  # rendered by GameScene as a turn separator line, not a sentence
            "destroyed_in_fight",  # internal marker for Warchest-style queries; the same
            # destruction already gets its own "destroyed" log entry with a real sentence
            "mulligan_decision",  # Agent Interface Plan Milestone C: fires on both branches so
            "archive_decision",  # the observation layer sees a decline too; "mulligan"/"take_archive"
            # already narrate the case where something actually happened.
        }

        seen_kinds = set()
        # A spread of decks: the original Phase 1 pair, two of the Phase 2
        # curated decks (take_control, stun, under_card, move_aember, ...),
        # and the four Phase 3 presets (armor, capture, reveal, ready, ...).
        deck_pairs = [
            ("fignor", "igor"), ("wraith", "cinder"), ("riftwalker", "gambit"),
            ("stonewall", "starfall"), ("vigil", "thornwood"),
        ]
        for seed in range(10):
            decks = deck_pairs[seed % len(deck_pairs)]
            game = Game(GameConfig(decks=decks, seed=seed, max_turns=30))
            bots = {1: RandomBot(seed=seed), 2: RandomBot(seed=seed + 1000)}
            while not game.is_over:
                d = game.pending_decision
                choice = bots[d.player].decide(game.view_for(d.player), d)
                game.submit(choice)
            for ev in game.log.events:
                seen_kinds.add(ev.kind)
                text = describe_log_event(ev, viewer=1)
                if ev.kind in silent_ok:
                    continue
                self.assertIsNotNone(text, f"no sentence for log event kind {ev.kind!r}: {ev.data}")
                self.assertTrue(text.strip())

        # Sanity: the fuzz run actually exercised a good spread of events.
        self.assertGreater(len(seen_kinds), 10, seen_kinds)

    def test_spectating_never_says_you(self):
        # Bot-vs-bot spectate: nobody is "you", every player is addressed
        # in the third person as "Player N".
        for seed in range(6):
            game = Game(GameConfig(decks=("fignor", "igor"), seed=seed, max_turns=20))
            bots = {1: RandomBot(seed=seed), 2: RandomBot(seed=seed + 1000)}
            while not game.is_over:
                d = game.pending_decision
                choice = bots[d.player].decide(game.view_for(d.player), d)
                game.submit(choice)
            for ev in game.log.events:
                text = describe_log_event(ev, viewer=1, spectating=True)
                if text is None:
                    continue
                self.assertNotIn("You", text, (ev.kind, ev.data, text))
                self.assertNotIn("your", text.lower(), (ev.kind, ev.data, text))

    def test_phase_3_log_event_sentences(self):
        # Milestone D added log kinds (damage prevention/redirection, Blood
        # Money's place_aember, Mimicry's copy) that the fuzz-coverage test
        # above only reaches if the right cards happen to come up -- check
        # each directly instead.
        cases = [
            (LogEvent("damage_prevented", {"card": "Protectrix", "iid": 1, "amount": 3}),
             "Protectrix can't be dealt damage: 3 damage is prevented."),
            (LogEvent("damage_redirected", {"card": "Shadow Self", "iid": 1, "to": "Snudge", "to_iid": 2, "amount": 2}),
             "2 of Shadow Self's damage is redirected to Snudge."),
            (LogEvent("place_aember", {"card": "Blood Money", "iid": 1, "amount": 2}),
             "2 Æmber is placed on Blood Money."),
            (LogEvent("mimicry_copy", {"card": "Mimicry", "iid": 1, "copied": "Lash of Broken Dreams"}),
             "Mimicry copies Lash of Broken Dreams."),
            (LogEvent("reveal", {"player": 1, "cards": ["Battle Fleet"], "iids": [1]}),
             "You reveal Battle Fleet from your hand."),
            (LogEvent("reveal", {"player": 1, "cards": [], "iids": []}),
             "You reveal no cards from your hand."),
        ]
        for event, expected in cases:
            self.assertEqual(describe_log_event(event, viewer=1), expected)

    def test_you_is_never_mis_conjugated(self):
        # Every "You <verb>" sentence must use the you-form of the verb
        # (no stray third-person "-s"), for both possible viewers.
        bad_verbs = {
            "mulligans", "forges", "chooses", "takes", "discards", "plays",
            "uses", "reaps", "gains", "loses", "draws", "shuffles", "archives",
        }
        for seed in range(6):
            game = Game(GameConfig(decks=("fignor", "igor"), seed=seed, max_turns=20))
            bots = {1: RandomBot(seed=seed), 2: RandomBot(seed=seed + 1000)}
            while not game.is_over:
                d = game.pending_decision
                choice = bots[d.player].decide(game.view_for(d.player), d)
                game.submit(choice)
            for viewer in (1, 2):
                for ev in game.log.events:
                    text = describe_log_event(ev, viewer=viewer, spectating=False)
                    if text is None or not text.startswith("You "):
                        continue
                    verb = text.split(" ")[1].rstrip(".")
                    self.assertNotIn(verb, bad_verbs, (viewer, ev.kind, ev.data, text))


if __name__ == "__main__":
    unittest.main()
