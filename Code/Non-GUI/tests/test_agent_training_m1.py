"""Agent Training Plan, Milestone M1 (Code/AGENT_TRAINING_PLAN.md): the
tensor specification (agent/spec.py), the redacted info-set extract
(keyforge/infoset.py) and the encoder (agent/features.py).

The plan's five acceptance criteria, each a class below: no leak,
determinism, throughput (measured by tools/bench_engine.py --encoding, not
asserted here -- a wall-clock bound is flaky on a shared machine), coverage
of every DecisionKind, and pool-independence. Plus the reserved-slot rule.

The full-scale no-leak run (1,000 seeded games) is `python -m
tools.check_encoder --games 1000`; the suite runs a smaller sample of the
same check.
"""

import os
import random
import re
import subprocess
import sys
import unittest
from collections import Counter

from agent import spec
from agent.features import encode, static_table
from bots.random_bot import RandomBot
from bots.heuristic_bot import HeuristicBot
from keyforge.cards.card_data import CARD_DEFS
from keyforge.cards.decks import Deck, build_alliance_deck, random_deck, resolve_deck
from keyforge.config import GameConfig
from keyforge.enums import DecisionKind, House, Resample
from keyforge.game import Game
from keyforge.infoset import (
    KNOWN_TO_ME,
    ZONE,
    ZONE_NAMES,
    _option_entry,
    build_infoset,
    infoset_for,
)
from keyforge.match import Match, MatchConfig
from tools.check_encoder import check_entitlement

_NON_GUI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _play(config, bots, on_decision):
    game = Game(config)
    while not game.is_over:
        d = game.pending_decision
        on_decision(game, d)
        game.submit(bots[d.player].decide(game.view_for(d.player), d))
    return game


def _option_multiset(e):
    w = spec.OPTION.width
    return sorted((bytes(e.options[k * w : (k + 1) * w].tobytes()), e.pointers[k]) for k in range(e.n_options))


class TestNoLeak(unittest.TestCase):
    def test_every_entity_is_placed_only_where_the_viewer_is_entitled(self):
        deck_rng = random.Random(7)
        for seed in range(40):
            if seed % 2:
                decks = (random_deck(deck_rng, "R1"), random_deck(deck_rng, "R2"))
                bots = {1: RandomBot(seed=seed), 2: RandomBot(seed=seed + 1)}
            else:
                decks = ("fignor", "igor")
                bots = {1: HeuristicBot(seed=seed), 2: HeuristicBot(seed=seed + 1)}

            def check(game, _d):
                for viewer in (1, 2):
                    problems = check_entitlement(game, viewer, build_infoset(game, viewer))
                    self.assertEqual(problems, [], f"seed {seed}, turn {game.turn_number}")

            _play(GameConfig(decks=decks, seed=seed, max_turns=40), bots, check)

    def test_encoding_is_a_function_of_the_information_set_alone(self):
        """The strongest leak check available: `fork_determinized` re-deals
        exactly the cards the viewer can't see. If any hidden fact reached
        the encoding, some re-deal would change it."""
        checked = 0
        for seed in range(12):
            rng = random.Random(seed)
            bots = {1: HeuristicBot(seed=seed), 2: HeuristicBot(seed=seed + 1)}
            counter = [0]

            def check(game, d):
                counter[0] += 1
                if counter[0] % 9:
                    return
                nonlocal checked
                for viewer in (1, 2):
                    base = encode(build_infoset(game, viewer))
                    for resample in (Resample.OWN_DECK, Resample.OPPONENT_PRIVATE, Resample.ALL):
                        fork = game.fork_determinized(viewer, random.Random(rng.getrandbits(32)), resample=resample)
                        other = encode(build_infoset(fork, viewer))
                        self.assertEqual(base.card_ids, other.card_ids)
                        self.assertEqual(base.zones, other.zones, f"seed {seed} {resample}")
                        self.assertEqual(base.flags, other.flags, f"seed {seed} {resample}")
                        self.assertEqual(base.inplay_index, other.inplay_index)
                        self.assertEqual(base.inplay, other.inplay)
                        self.assertEqual(base.globals, other.globals, f"seed {seed} {resample}")
                        self.assertEqual(_option_multiset(base), _option_multiset(other))
                        checked += 1

            _play(GameConfig(decks=("fignor", "igor"), seed=seed, max_turns=40), bots, check)
        self.assertGreater(checked, 300)

    def test_opponent_hidden_cards_are_indistinguishable(self):
        """Every opp_unseen entity carries exactly the same per-decision
        bytes -- no field (known_to_them, controller, ...) quietly
        distinguishes a card in their hand from one in their deck."""
        game = Game(GameConfig(decks=("fignor", "igor"), seed=3))
        bots = {1: HeuristicBot(seed=3), 2: HeuristicBot(seed=4)}
        for _ in range(60):
            d = game.pending_decision
            game.submit(bots[d.player].decide(game.view_for(d.player), d))
        info = build_infoset(game, 1)
        unseen = [i for i, z in enumerate(info.zones) if z == ZONE["opp_unseen"]]
        self.assertTrue(unseen)
        self.assertEqual({info.flags[i] for i in unseen}, {0})
        self.assertFalse(any(i in info.inplay for i in unseen))


class TestMasterplanLeak(unittest.TestCase):
    """Found while building M1: a card facedown beneath Masterplan leaked
    its identity to the opponent through the log and through the
    observation's `under_cards`."""

    def _game(self):
        igor = resolve_deck("igor")
        pods = {h: list(v) for h, v in igor.pods.items()}
        pods[House.SHADOWS][0] = "Masterplan"
        deck = Deck(name="IgorMP", pods=pods)
        game = Game(GameConfig(decks=(deck, "fignor"), seed=5, first_player=1, setup_script=[("put_artifact", 1, "Masterplan")]))
        game.submit(False)  # mulligans; the setup script runs after them
        game.submit(False)
        return game

    def test_facedown_card_is_hidden_from_the_opponent_everywhere(self):
        from keyforge.effects.named import shadows
        from keyforge.observation import build_observation
        from tests.helpers import drive

        game = self._game()
        mp = next(c for c in game.players[1].play_area.artifacts if c.name == "Masterplan")
        under = game.players[1].hand.cards()[0]
        drive(shadows.masterplan_play(game, mp), [[under]])
        self.assertEqual(mp.under_cards, [under])
        event = game.log.by_kind["under_card"][-1]
        self.assertEqual(event.visible_to, frozenset({1}))

        opp_obs = build_observation(game, 2)
        mp_state = next(a for a in opp_obs.players[1].artifacts if a["name"] == "Masterplan")
        self.assertEqual(mp_state["under_cards"], [{"__card__": None}])
        own_obs = build_observation(game, 1)
        mp_state = next(a for a in own_obs.players[1].artifacts if a["name"] == "Masterplan")
        self.assertEqual(mp_state["under_cards"], [{"__card__": under.instance_id}])

        opp_info = build_infoset(game, 2)
        own_info = build_infoset(game, 1)
        i = opp_info.index_of_iid[under.instance_id]
        self.assertEqual(ZONE_NAMES[opp_info.zones[i]], "opp_unseen")
        j = own_info.index_of_iid[under.instance_id]
        self.assertEqual(ZONE_NAMES[own_info.zones[j]], "my_under")
        self.assertEqual(check_entitlement(game, 2, opp_info), [])
        self.assertEqual(check_entitlement(game, 1, own_info), [])


class TestDeterminism(unittest.TestCase):
    def test_same_bytes_across_processes_and_hash_seeds(self):
        from tests._encoding_worker import digest

        here = digest(3)
        for hashseed in ("0", "12345"):
            env = dict(os.environ, PYTHONHASHSEED=hashseed)
            out = subprocess.run(
                [sys.executable, os.path.join(_NON_GUI, "tests", "_encoding_worker.py"), "3"],
                cwd=_NON_GUI, env=env, capture_output=True, text=True, timeout=300,
            )
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertEqual(out.stdout.strip(), here, f"PYTHONHASHSEED={hashseed}")

    def test_static_table_is_complete_and_deterministic(self):
        table = static_table()
        self.assertEqual(len(table), 370)
        for vid, row in enumerate(table):
            self.assertEqual(len(row), spec.STATIC.width)
            self.assertGreater(sum(row), 0, f"vocab id {vid} has an all-zero static row")


def _drive_until_kinds(target_kinds, *, max_games):
    """Random decks from all 7 houses with RandomBot, plus Adaptive matches
    for the between-games kinds, until every kind in `target_kinds` has
    been encoded (or `max_games` is spent). Returns {kind: count}."""
    seen = Counter()
    deck_rng = random.Random(11)
    for g in range(max_games):
        if not (set(target_kinds) - set(seen)):
            break
        d1, d2 = random_deck(deck_rng, "R1"), random_deck(deck_rng, "R2")
        if g % 3 == 0:
            # Heuristic play, so games conclude and a split (the same deck
            # winning both) reaches the chain bid.
            obj = Match(MatchConfig(format="adaptive", decks=(d1, d2), seed=g, max_turns=80))
            bots = {1: HeuristicBot(seed=g), 2: HeuristicBot(seed=g + 1)}
        else:
            obj = Game(GameConfig(decks=(d1, d2), seed=g, max_turns=40))
            bots = {1: RandomBot(seed=g), 2: RandomBot(seed=g + 1)}
        while not obj.is_over:
            dec = obj.pending_decision
            e = encode(infoset_for(obj, dec.player))
            assert e.n_options == len(dec.options)
            seen[dec.kind] += 1
            obj.submit(bots[dec.player].decide(obj.view_for(dec.player), dec))
    return seen


class TestCoverage(unittest.TestCase):
    def test_every_decision_kind_encodes(self):
        seen = _drive_until_kinds(tuple(DecisionKind), max_games=400)
        missing = [k.name for k in DecisionKind if k not in seen]
        self.assertEqual(missing, [], f"never reached: {missing} (seen: {dict(seen)})")

    def test_unknown_option_type_is_a_hard_error(self):
        game = Game(GameConfig(decks=("fignor", "igor"), seed=1))
        with self.assertRaises(ValueError):
            _option_entry(game.pending_decision, object(), {})

    def test_every_duration_effect_variable_has_a_slot(self):
        pattern = re.compile(r"(?:DurationEffect\([^)]*?|duration_effect\(|duration_effects_for\(|_apply\(game, )\"([A-Z][A-Za-z]+)\"")
        found = set()
        for root, _dirs, files in os.walk(os.path.join(_NON_GUI, "keyforge")):
            for name in files:
                if name.endswith(".py"):
                    with open(os.path.join(root, name), "r", encoding="utf-8") as f:
                        found.update(pattern.findall(f.read()))
        found.discard("Game")
        missing = sorted(found - set(spec.EFFECT_VARIABLES))
        self.assertEqual(missing, [], "new duration-effect variables need a slot in spec.EFFECT_VARIABLES")

    def test_every_keyword_in_the_pool_has_a_slot(self):
        for cdef in CARD_DEFS.values():
            for k in tuple(cdef.keywords) + tuple(cdef.grants_keywords):
                self.assertIn(k, spec.KEYWORDS, cdef.name)


class TestPoolIndependence(unittest.TestCase):
    def _shapes(self, decks, seed=1, n=80):
        game = Game(GameConfig(decks=decks, seed=seed, max_turns=40))
        bots = {1: RandomBot(seed=seed), 2: RandomBot(seed=seed + 1)}
        shapes = set()
        for _ in range(n):
            if game.is_over:
                break
            d = game.pending_decision
            e = encode(build_infoset(game, d.player))
            shapes.add((len(e.card_ids), len(e.zones), len(e.globals), len(e.options) // max(e.n_options, 1)))
            self.assertEqual(check_entitlement(game, d.player, build_infoset(game, d.player)), [])
            game.submit(bots[d.player].decide(game.view_for(d.player), d))
        return shapes

    def test_same_shapes_for_presets_random_and_alliance_decks(self):
        expected = {(72, 72, spec.GLOBAL.width, spec.OPTION.width)}
        rng = random.Random(3)
        alliance = build_alliance_deck("A", {House.BROBNAR: "stonewall", House.MARS: "starfall", House.DIS: "vigil"})
        for decks in (("fignor", "igor"), ("stonewall", "starfall"), ("vigil", "thornwood"),
                      (random_deck(rng, "R1"), random_deck(rng, "R2")), (alliance, "igor")):
            with self.subTest(decks=[getattr(d, "name", d) for d in decks]):
                shapes = self._shapes(decks)
                self.assertTrue(shapes <= expected | {(72, 72, spec.GLOBAL.width, spec.OPTION.width)}, shapes)


# Pinned widths per FEATURE_VERSION. Changing any width without bumping
# FEATURE_VERSION (and adding its row here) fails this test -- the
# reserved-slot rule, enforced. Consuming a reserved slot keeps widths, so
# it never trips this.
PINNED = {
    1: {"static": 122, "entity": 86, "inplay": 37, "global": 222, "option": 50},
}


class TestReservedSlots(unittest.TestCase):
    def test_widths_are_pinned_to_the_feature_version(self):
        widths = {b.name: b.width for b in spec.BLOCKS}
        self.assertIn(spec.FEATURE_VERSION, PINNED, "new FEATURE_VERSION: pin its widths here")
        self.assertEqual(widths, PINNED[spec.FEATURE_VERSION])

    def test_every_block_ends_in_reserved_slots_and_reports_them(self):
        report = spec.reserved_report()
        self.assertEqual(report["entity"], 16)
        self.assertEqual(report["global"], 24)
        self.assertEqual(report["option"], 8)
        self.assertGreaterEqual(report["keyword_bits"], 0)
        self.assertEqual(report["intents"], 8)
        for name, remaining in report.items():
            self.assertGreaterEqual(remaining, 0, name)
        for block in (spec.STATIC, spec.ENTITY, spec.GLOBAL, spec.OPTION):
            self.assertTrue(block.fields[-1][0].startswith("reserved"), block.name)

    def test_consuming_a_reserved_slot_keeps_the_shape_hash(self):
        """A minor change -- a reserved field renamed into use, same width
        -- must leave the index-bearing shape (and so checkpoint
        compatibility) intact; widening a field must not."""
        fields = list(spec.GLOBAL.fields)
        renamed = fields[:-1] + [("new_counter", 1), ("reserved_global", fields[-1][1] - 1)]
        self.assertEqual(spec.Block("g", renamed).width, spec.GLOBAL.width)
        widened = [(n, w + 1) if n == "turn" else (n, w) for n, w in fields]
        self.assertNotEqual(spec.Block("g", widened).width, spec.GLOBAL.width)

    def test_stamp_round_trips_and_refuses_a_major_mismatch(self):
        stamp = spec.stamp()
        spec.check_stamp(stamp)
        bad = dict(stamp, feature_version=stamp["feature_version"] + 1)
        with self.assertRaises(spec.FeatureVersionMismatch):
            spec.check_stamp(bad)
        minor = dict(stamp, feature_minor=stamp["feature_minor"] + 1)
        spec.check_stamp(minor)  # allowed by default
        with self.assertRaises(spec.FeatureVersionMismatch):
            spec.check_stamp(minor, allow_minor=False)


class TestDenseForm(unittest.TestCase):
    def test_dense_rows_are_static_then_entity_and_flags_decode(self):
        game = Game(GameConfig(decks=("fignor", "igor"), seed=2))
        info = build_infoset(game, game.pending_decision.player)
        e = encode(info)
        entities, globals_, options, pointers = e.dense()
        w = spec.entity_width()
        self.assertEqual(len(entities), 72 * w)
        table = static_table()
        for i in range(72):
            row = entities[i * w : (i + 1) * w]
            self.assertEqual(list(row[: spec.STATIC.width]), list(table[e.card_ids[i]]))
            zone_block = row[spec.STATIC.width : spec.STATIC.width + spec.ENTITY.span("zone")[1]]
            self.assertEqual(sum(zone_block), 1.0)
            self.assertEqual(zone_block[info.zones[i]], 1.0)
            known = row[spec.STATIC.width + spec.ENTITY.offset("flags")]
            self.assertEqual(known, 1.0 if info.flags[i] & KNOWN_TO_ME else 0.0)
        self.assertEqual(len(pointers), len(game.pending_decision.options))

    def test_prefix_and_stop_for_the_sequential_treatment(self):
        game = Game(GameConfig(decks=("fignor", "igor"), seed=2))
        bots = {1: HeuristicBot(seed=1), 2: HeuristicBot(seed=2)}
        while not (game.pending_decision.kind == DecisionKind.CHOOSE_ACTION and len(game.pending_decision.options) > 2):
            d = game.pending_decision
            game.submit(bots[d.player].decide(game.view_for(d.player), d))
        info = build_infoset(game, game.pending_decision.player)
        plain = encode(info)
        seq = encode(info, prefix=[0], with_stop=True)
        self.assertEqual(seq.n_options, plain.n_options + 1)
        self.assertEqual(seq.pointers[-1], -1)
        stop_row = seq.options[(seq.n_options - 1) * spec.OPTION.width : seq.n_options * spec.OPTION.width]
        self.assertEqual(stop_row[spec.OPTION.offset("verb") + spec.STOP_VERB], 1.0)
        if plain.pointers[0] >= 0:
            self.assertEqual(seq.selected[plain.pointers[0]], 1)


if __name__ == "__main__":
    unittest.main()
