"""The GUI's bot seat takes any registered agent (`--bot`), handed a
capability matching that agent's registered privilege -- so a search agent
actually searches in the GUI instead of silently falling back."""

import unittest

import tests.helpers  # noqa: F401  (sets up sys.path + dummy SDL driver)

from gui.engine_bridge import SEAT_BOT, EngineBridge, MatchBridge, MatchSettings


class TestBridgeAgents(unittest.TestCase):
    def _play(self, bridge, limit=60):
        steps = 0
        self.kinds = set()
        while not bridge.is_over and steps < limit:
            d = bridge.pending_decision
            self.kinds.add(d.kind.name)
            choice = bridge.bot_choice(d.player)
            self.assertTrue(d.validate(choice))
            bridge.submit(choice, viewer=1)
            steps += 1
        return steps

    def test_a_search_agent_seat_searches_through_the_bridge(self):
        settings = MatchSettings(p1_seat=SEAT_BOT, p2_seat=SEAT_BOT, seed=5, max_turns=6, bot_agent="search-within-turn")
        bridge = EngineBridge(settings)
        for bot in bridge.bots.values():
            bot.search.settings.simulations = 4
        self.assertGreater(self._play(bridge), 0)
        searched = sum(b.search_stats["decisions_searched"] for b in bridge.bots.values())
        self.assertGreater(searched, 0, "the search agent fell back instead of searching")

    def test_match_bridge_handles_between_game_decisions(self):
        settings = MatchSettings(p1_seat=SEAT_BOT, p2_seat=SEAT_BOT, seed=6, max_turns=80, format="adaptive", bot_agent="search-within-turn")
        bridge = MatchBridge(settings)
        for bot in bridge.bots.values():
            bot.search.settings.simulations = 2
        self._play(bridge, limit=5000)
        self.assertTrue(bridge.is_over)
        self.assertIn("CHOOSE_FIRST_PLAYER", self.kinds)  # a between-games decision, with no live game to fork


if __name__ == "__main__":
    unittest.main()
