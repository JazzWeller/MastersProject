"""Agent Observation Plan, Milestone O0: the content-level leak test.

Milestone C's fuzz test compares only the *kinds* of history events with the
log's own visibility, so it could not see that a `draw` event named the
drawn cards to both players. These tests compare contents: every card
identity a viewer is shown, in `Observation.history` and in
`PlayerView.log_tail`, must be one that viewer was entitled to when the event
happened (tests/_entitlement.py records that, privileged, during the game).
"""

import random
import unittest

from bots.heuristic_bot import HeuristicBot
from bots.random_bot import RandomBot
from keyforge.cards.decks import DECKS, random_deck
from keyforge.config import GameConfig
from keyforge.game import Game
from keyforge.observation import build_observation

from tests._entitlement import check_history, recorded_game


def _play(game, bots, recorder, on_decision=None):
    while not game.is_over:
        d = game.pending_decision
        if on_decision is not None:
            on_decision(game)
        game.submit(bots[d.player].decide(game.view_for(d.player), d))


def _shown_history(game, viewer):
    """`(log index, kind, data)` for each entry of `viewer`'s observation
    history, aligned with the full log."""
    obs = build_observation(game, viewer)
    indices = [i for i, e in enumerate(game.log.events) if viewer in e.visible_to]
    assert len(indices) == len(obs.history)
    return [(i, h.kind, h.data) for i, h in zip(indices, obs.history)]


def _shown_tail(game, viewer):
    view = game.view_for(viewer)
    indices = [i for i, e in enumerate(game.log.events) if viewer in e.visible_to][-len(view.log_tail):] if view.log_tail else []
    return [(i, e.kind, e.data) for i, e in zip(indices, view.log_tail)]


def _games(n_random=40):
    """Fignor/Igor, every bundled preset pairing, and random legal decks,
    played by HeuristicBot and RandomBot."""
    deck_rng = random.Random(7)
    presets = sorted(DECKS)
    configs = []
    for seed in range(6):
        configs.append(((("fignor", "igor")), seed, "heuristic"))
    for i, name in enumerate(presets):
        configs.append(((name, presets[(i + 3) % len(presets)]), 100 + i, "heuristic" if i % 2 else "random"))
    for i in range(n_random):
        configs.append(((random_deck(deck_rng, "R1"), random_deck(deck_rng, "R2")), 200 + i, "heuristic" if i % 2 else "random"))
    return configs


class TestHistoryContentsAreEntitled(unittest.TestCase):
    def _run(self, decks, seed, bot_kind):
        game, recorder = recorded_game(GameConfig(decks=decks, seed=seed, max_turns=80))
        if bot_kind == "heuristic":
            bots = {1: HeuristicBot(seed=seed), 2: HeuristicBot(seed=seed + 1)}
        else:
            bots = {1: RandomBot(seed=seed), 2: RandomBot(seed=seed + 1)}
        tails = []

        def sample_tail(g):
            if len(g.choice_record) % 7 == 0:
                for viewer in (1, 2):
                    tails.append((viewer, _shown_tail(g, viewer)))

        _play(game, bots, recorder, on_decision=sample_tail)
        leaks = []
        for viewer in (1, 2):
            leaks += check_history(game, recorder, viewer, _shown_history(game, viewer))
        for viewer, shown in tails:
            leaks += check_history(game, recorder, viewer, shown)
        return leaks

    def test_observation_history_and_log_tail_name_only_entitled_cards(self):
        for decks, seed, bot_kind in _games():
            with self.subTest(seed=seed):
                leaks = self._run(decks, seed, bot_kind)
                self.assertEqual(leaks[:5], [], f"{len(leaks)} leaks, seed {seed}")

    def test_the_owner_still_sees_their_own_draws(self):
        game = Game(GameConfig(decks=("fignor", "igor"), seed=3, max_turns=80))
        bots = {1: HeuristicBot(seed=3), 2: HeuristicBot(seed=4)}
        for _ in range(60):
            d = game.pending_decision
            game.submit(bots[d.player].decide(game.view_for(d.player), d))
        for viewer in (1, 2):
            obs = build_observation(game, viewer)
            own = [h for h in obs.history if h.kind == "draw" and h.data["player"] == viewer]
            theirs = [h for h in obs.history if h.kind == "draw" and h.data["player"] != viewer]
            self.assertTrue(own and theirs)
            self.assertTrue(all(isinstance(i, int) for h in own for i in h.data["iids"]))
            # The fact and the count stay public; the identities don't.
            self.assertTrue(all(h.data["n"] >= 1 and "iids" not in h.data for h in theirs))

    def test_the_raw_log_is_unchanged(self):
        """The rules and the GUI read the raw log, which stays complete."""
        game = Game(GameConfig(decks=("fignor", "igor"), seed=3, max_turns=80))
        draws = [e for e in game.log.events if e.kind == "draw"]
        self.assertTrue(draws and all(isinstance(i, int) for e in draws for i in e.data["iids"]))


if __name__ == "__main__":
    unittest.main()
