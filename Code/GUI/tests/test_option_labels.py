"""Every option type in gui/decision/panel.py's vocabulary, and every log
event kind the engine actually emits, must produce a non-empty label."""

import unittest

import tests.helpers  # noqa: F401

from bots.random_bot import RandomBot
from keyforge.actions import DiscardCard, EndTurn, Fight, PlayCard, Reap, UseAction, UseOmni
from keyforge.config import GameConfig
from keyforge.enums import House
from keyforge.game import Game

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

    def test_every_log_event_kind_has_a_sentence_or_is_intentionally_silent(self):
        # Kinds describe_log_event deliberately returns None for (nothing
        # interesting to tell a player) — everything else must render.
        silent_ok = set()

        seen_kinds = set()
        for seed in range(10):
            game = Game(GameConfig(decks=("fignor", "igor"), seed=seed, max_turns=30))
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
