"""Agent Observation Plan, Milestone O1: the zone journal and the per-viewer
projection.

The acceptance checks themselves live in tools/o1_acceptance.py (which runs
them over 1,000 games per pool); these run them on a few games, plus the
registry test for the log's redaction schemas and some properties of the
projection by example.
"""

import ast
import os
import unittest

from keyforge.config import GameConfig
from keyforge.game import Game
from keyforge.log import EVENT_SCHEMAS, SITE_RESTRICTED
from keyforge.observation import build_observation
from keyforge.projection import PRIVATE, PUBLIC, REVEALED, Projector
from bots.heuristic_bot import HeuristicBot
from tools import o1_acceptance

_PACKAGE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "keyforge")


def _log_calls():
    """(path, line, kinds, restricted) for every `*.log.add(...)` in the
    engine source."""
    for root, dirs, files in os.walk(_PACKAGE):
        dirs[:] = [d for d in dirs if d not in ("compiled", "__pycache__")]
        for name in files:
            if not name.endswith(".py"):
                continue
            path = os.path.join(root, name)
            with open(path, encoding="utf-8") as f:
                tree = ast.parse(f.read())
            for n in ast.walk(tree):
                if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "add"
                        and isinstance(n.func.value, ast.Attribute) and n.func.value.attr == "log"):
                    arg = n.args[0] if n.args else None
                    if isinstance(arg, ast.IfExp):
                        kinds = [arg.body, arg.orelse]
                    else:
                        kinds = [arg]
                    restricted = any(k.arg == "visible_to" for k in n.keywords)
                    yield path, n.lineno, [k.value if isinstance(k, ast.Constant) else None for k in kinds], restricted


def _played(config, max_decisions=None):
    game = Game(config)
    bots = {1: HeuristicBot(seed=config.seed), 2: HeuristicBot(seed=config.seed + 1)}
    n = 0
    while not game.is_over and (max_decisions is None or n < max_decisions):
        d = game.pending_decision
        game.submit(bots[d.player].decide(game.view_for(d.player), d))
        n += 1
    return game


class TestEventSchemas(unittest.TestCase):
    def test_every_event_kind_the_engine_logs_has_a_schema(self):
        calls = list(_log_calls())
        self.assertGreater(len(calls), 100)
        for path, line, kinds, restricted in calls:
            for kind in kinds:
                self.assertIsNotNone(kind, f"{path}:{line}: an event kind that isn't a literal")
                self.assertIn(kind, EVENT_SCHEMAS, f"{path}:{line}: event kind {kind!r} has no redaction schema")
                if restricted:
                    self.assertIn(kind, SITE_RESTRICTED, f"{path}:{line}: {kind!r} narrows who sees it at its call "
                                                          "site but isn't declared to")

    def test_a_draw_is_redacted_by_its_schema(self):
        game = _played(GameConfig(decks=("fignor", "igor"), seed=1), max_decisions=5)
        draws = [e for e in game.log.events if e.kind == "draw"]
        self.assertTrue(draws)
        for e in draws:
            self.assertEqual(e.private, {"iids": frozenset({e.data["player"]})})


class TestAcceptanceSample(unittest.TestCase):
    """Completeness, sigma-invariance, no over-hiding and determinism on a
    few games per pool (the full sweep: `python -m tools.o1_acceptance`)."""

    def test_a_few_games_per_pool(self):
        for pool in o1_acceptance.POOLS:
            for i in range(3):
                with self.subTest(pool=pool, game=i):
                    config = o1_acceptance.game_config(pool, i)
                    self.assertEqual(o1_acceptance.check_game(config, i), [])

    def test_sigma_catches_a_leak(self):
        """The check has teeth: a projection that shows decks is caught."""
        config = o1_acceptance.game_config("phase1", 1)
        played = o1_acceptance.Played(config, o1_acceptance.bots_for(1, config.seed))
        saved = Projector.visible_zone
        Projector.visible_zone = lambda self, zone, pid, owner: zone[0] == "deck" or saved(self, zone, pid, owner)
        try:
            import random
            self.assertNotEqual(o1_acceptance.check_sigma(played, random.Random(0)), [])
        finally:
            Projector.visible_zone = saved


class TestProjection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.game = _played(GameConfig(decks=("fignor", "igor"), seed=4))

    def test_incremental_equals_from_scratch(self):
        config = GameConfig(decks=("igor", "fignor"), seed=9)
        game = Game(config)
        bots = {1: HeuristicBot(seed=9), 2: HeuristicBot(seed=10)}
        snapshots = []
        while not game.is_over:
            if len(game.choice_record) % 17 == 0:
                snapshots.append((len(game.projected(1)), list(game.projected(1))))
            d = game.pending_decision
            game.submit(bots[d.player].decide(game.view_for(d.player), d))
        final = game.projected(1)
        self.assertEqual(final, Projector(1).update(game.journal, game.log.events))
        for n, prefix in snapshots:
            self.assertEqual(final[:n], prefix)

    def test_draws_are_the_drawers(self):
        for v in (1, 2):
            draws = [x for x in self.game.projected(v) if x[0] == "zone" and x[8] == "draw_top/add"]
            mine = [x for x in draws if x[6] == ("deck", v)]
            theirs = [x for x in draws if x[6] == ("deck", 3 - v)]
            self.assertTrue(mine and theirs)
            self.assertTrue(all(x[3] is not None and x[-1] == PRIVATE for x in mine))
            self.assertTrue(all(x[3] is None and x[-1] == PUBLIC for x in theirs))

    def test_a_card_played_from_hand_is_revealed(self):
        for v in (1, 2):
            plays = [x for x in self.game.projected(v) if x[0] == "zone" and x[6][0] == "hand"
                     and x[7][0] in ("battleline", "artifacts", "limbo")]
            mine = [x for x in plays if x[6][1] == v]
            theirs = [x for x in plays if x[6][1] != v]
            self.assertTrue(mine and theirs)
            self.assertTrue(all(x[3] is not None and x[-1] == REVEALED for x in mine))
            self.assertTrue(all(x[3] is not None and x[-1] == PUBLIC for x in theirs))

    def test_moves_name_what_caused_them(self):
        causes = [e[10] for e in self.game.journal if e[10] is not None]
        self.assertTrue(causes)
        for c in causes:
            self.assertIn(c, self.game._cards_by_id)

    def test_decisions_show_the_choice_to_the_chooser_only_when_hidden(self):
        decisions = {v: [x for x in self.game.projected(v) if x[0] == "decision"] for v in (1, 2)}
        self.assertEqual(len(decisions[1]), len(decisions[2]))
        self.assertEqual(len(decisions[1]), len(self.game.choice_record))
        hidden_somewhere = 0
        for a, b in zip(decisions[1], decisions[2]):
            chooser, other = (a, b) if a[2] == 1 else (b, a)
            for mine, theirs in zip(chooser[8], other[8]):
                if len(mine) == 3 and theirs[1] is None:
                    hidden_somewhere += 1
                    self.assertIsNotNone(mine[1])
                    self.assertEqual(chooser[-1], PRIVATE)
        self.assertGreater(hidden_somewhere, 0)

    def test_observation_history_is_the_projection(self):
        obs = build_observation(self.game, 1)
        streams = {h.stream for h in obs.history}
        self.assertEqual(streams, {"log", "zone", "decision"})
        self.assertEqual(len(obs.history), len(self.game.projected(1)))


if __name__ == "__main__":
    unittest.main()
