"""Agent Observation Plan, Milestone O2: the knowledge tracker.

The soundness fuzz and the comparison with the consistent worlds live in
tools/o2_acceptance.py (run over 1,000 games per pool); these run them on a
few games, find the plan's five scenarios in fuzz games and compare the
tracker with the worlds right after each, and check what a copy carries.
"""

import unittest

from bots.heuristic_bot import HeuristicBot
from keyforge.config import GameConfig
from keyforge.game import Game
from tools import o2_acceptance
from tools.o1_acceptance import POOLS, bots_for, game_config


def _scenarios(game, start):
    """The plan's scenario kinds among the journal entries from `start`."""
    names = {i: c.name for i, c in game._cards_by_id.items()}
    entries = game.journal.entries
    out = set()
    for e in entries[start:]:
        cause = names.get(e[10])
        frm, to = e[5], e[6]
        if cause == "Lights Out" and frm[0] == "battleline" and to[0] == "hand":
            out.add("lights_out")
        if cause == "Snudge" and frm[0] == "artifacts" and to[0] == "hand":
            out.add("snudge")
        if to[0] == "deck" and e[9] == "top" and frm[0] != "setup":
            out.add("put_on_top")
        if e[7] == "take_all/shuffle_in" and frm[0] == "discard":
            out.add("reshuffle")
        if frm[0] == "hand" and to[0] == "archive" and any(
                f[7] == "reveal_hand" and f[5] == frm for f in entries[:e[0]]):
            out.add("reveal_then_archive")
    return out


class TestSoundness(unittest.TestCase):
    def test_a_few_games_per_pool(self):
        for pool in POOLS:
            for i in range(3):
                with self.subTest(pool=pool, game=i):
                    self.assertEqual(o2_acceptance.check_game(game_config(pool, i), i, brute_every=25 if i == 0 else 0), [])


class TestScenarios(unittest.TestCase):
    """A Lights Out bounce, an artifact returned by Snudge, an exhausted deck
    reshuffled, a card put on top of a deck, a hand revealed and later
    partly archived: found in fuzz games, and at the decision after each the
    tracker's masks are exactly the consistent worlds' (`world_masks`).
    (Elsewhere a mask can be wider than the worlds -- the tracker counts
    per zone, the worlds per card; O3's group counts close that.)"""

    KINDS = {"lights_out", "snudge", "put_on_top", "reshuffle", "reveal_then_archive"}

    def test_each_scenario_matches_the_worlds(self):
        found = {}
        checked = 0
        for i in range(400):
            if set(found) == self.KINDS and min(found.values()) >= 2:
                break
            config = game_config("random", 1000 + i)
            game = Game(config)
            bots = bots_for(i, config.seed)
            mark = 0
            while not game.is_over:
                n = len(game.journal.entries)
                kinds = _scenarios(game, mark) if n > mark else set()
                kinds = {k for k in kinds if found.get(k, 0) < 2}
                mark = n
                if kinds:
                    for v in (1, 2):
                        problems = o2_acceptance.check_brute_force(game, v)
                        self.assertEqual(problems, [], f"game {i}: {sorted(kinds)}")
                        checked += 1
                    for k in kinds:
                        found[k] = found.get(k, 0) + 1
                d = game.pending_decision
                game.submit(bots[d.player].decide(game.view_for(d.player), d))
        self.assertEqual(set(found), self.KINDS, found)
        self.assertGreater(checked, 0)


class TestTracker(unittest.TestCase):
    def test_a_copy_carries_the_tracker_and_continues_identically(self):
        config = GameConfig(decks=("fignor", "igor"), seed=11)
        game = Game(config)
        bots = {1: HeuristicBot(seed=11), 2: HeuristicBot(seed=12)}
        n = 0
        while n < 60 or not game.copy_anywhere:  # (native execution copies at boundaries only)
            d = game.pending_decision
            game.submit(bots[d.player].decide(game.view_for(d.player), d))
            n += 1
        game.knowledge(1)
        copy = game.copy()
        self.assertIsNot(copy.__dict__["_trackers"][1], game.__dict__["_trackers"][1])
        record = []
        for _ in range(60):
            if game.is_over:
                break
            d = game.pending_decision
            game.submit(bots[d.player].decide(game.view_for(d.player), d))
            record.append(game.choice_record[-1])
        from keyforge.replay import decode_choice
        for encoded in record:
            copy.submit(decode_choice(copy.pending_decision, encoded))
        self.assertEqual(copy.knowledge(1).mask, game.knowledge(1).mask)
        fresh = Game(config)
        for encoded in game.choice_record:
            fresh.submit(decode_choice(fresh.pending_decision, encoded))
        self.assertEqual(fresh.knowledge(1).mask, game.knowledge(1).mask)

    def test_the_export(self):
        game = Game(GameConfig(decks=("fignor", "igor"), seed=2))
        bots = {1: HeuristicBot(seed=2), 2: HeuristicBot(seed=3)}
        for _ in range(40):
            d = game.pending_decision
            game.submit(bots[d.player].decide(game.view_for(d.player), d))
        t = game.knowledge(1)
        mine = [c for c in game.players[1].hand.cards()]
        theirs = [c for c in game.players[2].hand.cards()]
        for c in mine:
            x = t.export(c.instance_id, game.turn_number)
            self.assertEqual((x["kind"], x["zones"]), ("exact", (("hand", 1),)))
        for c in theirs:
            x = t.export(c.instance_id, game.turn_number)
            self.assertIn(("hand", 2), x["zones"])


if __name__ == "__main__":
    unittest.main()
