"""Real deck lists and mavericks: a maverick plays as its pod's house through
games and forks, the import refuses what the engine can't build, train and test
never share a deck, and evaluation uses every test deck once."""

from __future__ import annotations

import csv
import json
import os
import random
import tempfile
import unittest
from unittest import mock

import run
from keyforge.cards.card_data import CARD_DEFS
from keyforge.cards.decks import DECKS, Deck, build_deck, deck_from_dict, deck_to_dict, validate_deck
from keyforge.config import GameConfig
from keyforge.enums import House
from keyforge.game import Game
from bots.heuristic_bot import HeuristicBot
from sim import decklists
from sim.bc_corpus import decks_for
from tools import eval_v2, import_decks


def _pod(house, n=12):
    return [name for name, c in CARD_DEFS.items() if c.house == house][:n]


def _maverick_deck(name="Mav"):
    """A Dis / Logos / Shadows deck whose Dis pod holds a Sanctum card."""
    pods = {House.DIS: _pod(House.DIS, 11) + ["Bulwark"], House.LOGOS: _pod(House.LOGOS), House.SHADOWS: _pod(House.SHADOWS)}
    return Deck(name=name, pods=pods)


class Mavericks(unittest.TestCase):
    def test_a_maverick_is_legal_and_of_its_pods_house(self):
        deck = _maverick_deck()
        self.assertEqual(validate_deck(deck), [])
        self.assertEqual(CARD_DEFS["Bulwark"].house, House.SANCTUM)
        cards = build_deck(deck, 1)
        self.assertEqual([c for c in cards if c.name == "Bulwark"][0].house, House.DIS)
        self.assertEqual([c.name for c in cards], deck.all_card_names())

    def test_its_house_survives_a_dict_and_every_kind_of_fork(self):
        deck = deck_from_dict(deck_to_dict(_maverick_deck()))
        game = Game(GameConfig(decks=(deck, "igor"), seed=3, max_turns=200))
        forks = [game.fork(), game.fork_determinized(game.pending_decision.player, random.Random(1), sampler="chance_exact")]
        for g in [game] + forks:
            bulwark = [c for c in g._cards_by_id.values() if c.name == "Bulwark"][0]
            self.assertEqual(bulwark.house, House.DIS)

    def test_no_player_is_offered_a_house_outside_their_deck(self):
        game = Game(GameConfig(decks=(_maverick_deck(), _maverick_deck("B")), seed=7, max_turns=200))
        bots = {1: HeuristicBot(seed=1), 2: HeuristicBot(seed=2)}
        while not game.is_over:
            d = game.pending_decision
            if d.kind.name == "CHOOSE_HOUSE":
                self.assertNotIn(House.SANCTUM, d.options)
            game.submit(bots[d.player].decide(game.view_for(d.player), d))

    def test_presets_have_no_mavericks(self):
        for name, deck in DECKS.items():
            for house, names in deck.pods.items():
                for card in names:
                    self.assertEqual(CARD_DEFS[card].house, house, f"{name}: {card}")


class Import(unittest.TestCase):
    FIELDS = ["Name", "Houses", "Sas Rating", "Master Vault Link", "House 1 Cards", "House 2 Cards", "House 3 Cards"]

    def _csv(self, rows):
        f = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8-sig", newline="")
        self.addCleanup(os.remove, f.name)
        with f:
            w = csv.DictWriter(f, fieldnames=self.FIELDS)
            w.writeheader()
            w.writerows(rows)
        return f.name

    def _row(self, k, sas=70.0, dis=None):
        d = dis or _pod(House.DIS)
        d = list(d)
        rng = random.Random(k)
        logos = [rng.choice(_pod(House.LOGOS, 30)) for _ in range(12)]
        return {"Name": f"Deck “{k}”", "Houses": "Dis | Logos | Shadows", "Sas Rating": str(sas),
                "Master Vault Link": f"https://www.keyforgegame.com/decks/{k:08d}-0000-0000-0000-000000000000",
                "House 1 Cards": "|".join(d), "House 2 Cards": "|".join(logos), "House 3 Cards": "|".join(_pod(House.SHADOWS))}

    def test_it_keeps_a_maverick_and_refuses_an_unknown_card(self):
        rows = import_decks.read_rows(self._csv([self._row(1, dis=_pod(House.DIS, 11) + ["Bulwark"])]))
        self.assertEqual(rows[0]["pods"]["Dis"][-1], "Bulwark")
        self.assertEqual(rows[0]["name"], "Deck “1”")
        with self.assertRaisesRegex(ValueError, "Not A Card"):
            import_decks.read_rows(self._csv([self._row(1, dis=_pod(House.DIS, 11) + ["Not A Card"])]))

    def test_the_same_deck_twice_is_kept_once(self):
        self.assertEqual(len(import_decks.read_rows(self._csv([self._row(1), self._row(1)]))), 1)

    def test_the_split_ignores_file_order_and_covers_every_band(self):
        rows = [self._row(k, sas=45.0 + (k % 6) * 10) for k in range(600)]
        decks = import_decks.read_rows(self._csv(rows))
        import_decks.split(decks)
        first = {d["id"]: d["split"] for d in decks}
        again = import_decks.read_rows(self._csv(rows[::-1]))
        import_decks.split(again)
        self.assertEqual(first, {d["id"]: d["split"] for d in again})
        for b in {import_decks.band(d["sas"]) for d in decks}:
            members = [d for d in decks if import_decks.band(d["sas"]) == b]
            self.assertAlmostEqual(sum(d["split"] == "test" for d in members) / len(members), 1 / 6, delta=0.01)

    def test_bands(self):
        self.assertEqual([import_decks.band(s) for s in (44, 50, 51, 60, 61, 91, 100)],
                         ["0-50", "0-50", "51-60", "51-60", "61-70", "91-100", "91-100"])


class Lists(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        rows = [dict(deck_to_dict(_maverick_deck(f"D{k}")), split="test" if k < 10 else "train") for k in range(40)]
        with open(os.path.join(self.dir.name, "t.json"), "w", encoding="utf-8") as f:
            json.dump({"decks": rows}, f)
        for p in (mock.patch.object(decklists, "DECKLIST_DIR", self.dir.name), mock.patch.dict(decklists._LOADED, clear=True)):
            p.start()
            self.addCleanup(p.stop)

    def test_training_only_draws_the_train_split(self):
        pool = {"deck_source": "list", "deck_list": "t", "split": "train"}
        test = {d.name for d in decklists.decklist("t", "test")}
        for k in range(200):
            a, b = decks_for(pool, k, random.Random(k))
            self.assertNotEqual(a.name, b.name)
            self.assertFalse({a.name, b.name} & test)

    def test_evaluation_uses_every_deck_once_before_repeating(self):
        names = [d.name for k in range(5) for d in decklists.pair_for("t", "test", k)]
        self.assertEqual(sorted(names), sorted(d.name for d in decklists.decklist("t", "test")))

    def test_eval_v2_pairs_and_names(self):
        class Args:
            deck_list, deck_split, deck_set, deck_seed, first_seed, unit = "t", "test", 0, 2, 100, 1

        self.assertEqual(eval_v2.unit_decks(Args, 103), decklists.pair_for("t", "test", 3))
        self.assertIn("_t-train", eval_v2.out_name("a", "b", 100, 100, 1, False, 0, 2, "t", "train"))
        self.assertEqual(eval_v2.out_name("a", "b", 100, 100, 1), "a__b__100sims_100x1.json")


class Methods(unittest.TestCase):
    def test_real_runs_never_use_random_decks(self):
        for arm in ("v1", "v2"):
            step = run.METHODS[f"real-{arm}"]["steps"][0]
            override = json.loads(step[step.index("--override") + 1])
            self.assertEqual(override["pool"], {"deck_source": "list", "deck_list": "phase3", "split": "train"})
            self.assertEqual(step[step.index("--init-run") + 1], f"m11-{arm}")

    def test_eval_real_measures_the_gap_and_the_baseline(self):
        steps = run.METHODS["eval-real"]["steps"]
        self.assertEqual([s[s.index("--deck-split") + 1] for s in steps[:2]], ["test", "train"])
        self.assertIs(steps[2], run._BASELINE_STEP)
        self.assertNotIn("--deck-list", steps[2])
        self.assertIn("v1:real-v1", steps[2])


if __name__ == "__main__":
    unittest.main()
