"""Every log event that names a card also carries that card's instance id,
so the GUI can tell physical copies of the same-named card apart."""

import unittest

from keyforge.config import GameConfig
from keyforge.game import Game

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


if __name__ == "__main__":
    unittest.main()
