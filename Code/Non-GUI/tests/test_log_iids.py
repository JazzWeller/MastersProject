"""Every log event that names a card also carries that card's instance id,
so the GUI can tell physical copies of the same-named card apart."""

import unittest

from keyforge.config import GameConfig
from keyforge.game import Game
from keyforge.log import GameLog

from bots.random_bot import RandomBot

# Events keyed by a *pair* of card fields use their own iid field names
# instead of the single card/iid convention.
PAIR_IID_EVENTS = {
    "fight": ("attacker", "attacker_iid", "target", "target_iid"),
}
# Events that name a card but currently have no meaningful iid (list-valued
# or genuinely cardless) are intentionally excluded.
EXEMPT_EVENTS = {"choose_house", "forge_key", "mulligan", "take_archive"}


class TestLogIids(unittest.TestCase):
    def test_every_card_event_has_an_iid(self):
        for seed in range(15):
            game = Game(GameConfig(decks=("fignor", "igor"), seed=seed, max_turns=40))
            bots = {1: RandomBot(seed=seed), 2: RandomBot(seed=seed + 1000)}
            while not game.is_over:
                d = game.pending_decision
                choice = bots[d.player].decide(game.view_for(d.player), d)
                game.submit(choice)

            for event in game.log.events:
                data = event.data
                if event.kind in PAIR_IID_EVENTS:
                    card_key, iid_key, card_key2, iid_key2 = PAIR_IID_EVENTS[event.kind]
                    self.assertIn(iid_key, data, event)
                    self.assertIn(iid_key2, data, event)
                    continue
                if event.kind in EXEMPT_EVENTS:
                    continue
                if "card" in data:
                    self.assertIn("iid", data, f"{event} missing iid")
                if "iids" in data:
                    self.assertIsInstance(data["iids"], list)


class TestGameLogByKind(unittest.TestCase):
    """`GameLog.by_kind` (Milestone L: game.py's forged_key_on_turn and
    friends scan one kind instead of the whole log) must stay in exact sync
    with `events`, for every caller -- including a test that pokes
    `log.add(...)` directly rather than going through an engine call site,
    same as tests/test_rules_phase2.py's TestUnforgeQuery already does."""

    def test_by_kind_mirrors_events_added_through_add(self):
        log = GameLog()
        log.add("forge_key", player=1, turn=1)
        log.add("gain", player=1, amount=1)
        log.add("forge_key", player=2, turn=2)
        self.assertEqual(log.by_kind["forge_key"], [log.events[0], log.events[2]])
        self.assertEqual(log.by_kind["gain"], [log.events[1]])

    def test_an_unseen_kind_is_empty_not_a_keyerror(self):
        log = GameLog()
        self.assertEqual(log.by_kind["never_added"], [])


if __name__ == "__main__":
    unittest.main()
