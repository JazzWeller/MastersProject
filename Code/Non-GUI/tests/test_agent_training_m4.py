"""Agent Training Plan, Milestones M4 and M5 (Code/AGENT_TRAINING_PLAN.md):
the shared search core and the two regimes, with the no-network heuristic
evaluator -- everything the plan says to get right *before* a network
exists, because a wrong implementation produces an agent that merely looks
mediocre rather than one that crashes.

Mechanics are checked on a tiny, exactly solvable game (Nim, with an
optional hidden coin), where the right answers are known: backup signs
(forced win -> +1, forced loss -> -1, from either seat), availability
counts under determinization, subtree reuse, batching. KeyForge positions
built with `setup_script` check that the regimes find a known best move and
see a forced loss; a conformance block runs the search agent through the
driver at every privilege level.
"""

import random
import unittest

from agent.search.core import Search, SearchSettings, action_key, drive, run_searches
from agent.search.full_game import FullGame
from agent.search.leaf import HeuristicEvaluator
from agent.search.policies import heuristic_policy
from agent.search.within_turn import WithinTurn
from agent.agents.search_agent import SearchAgent
from bots.base import Budget
from bots.heuristic_bot import HeuristicBot
from keyforge.capabilities import make_capability
from keyforge.cards.decks import Deck
from keyforge.config import GameConfig
from keyforge.enums import DecisionKind, House, PrivilegeLevel, Resample
from keyforge.game import Game
from sim.driver import run_games


# ------------------------------------------------------------------ toy game
class _Decision:
    def __init__(self, player, options):
        self.player = player
        self.options = options
        self.kind = DecisionKind.CHOOSE_ACTION
        self.min_n = self.max_n = 1


class Nim:
    """Take 1 or 2 stones (3 as well, for the player to move, while the
    hidden `coin` is set and it's their first move); taking the last stone
    wins. `sub_moves` = 2 makes each turn two consecutive decisions by the
    same player (take 1-2, then take 0-1), for subtree-reuse tests."""

    def __init__(self, pile, to_move=1, coin=False, sub_moves=1):
        self.pile = pile
        self.to_move = to_move
        self.coin = coin
        self.sub_moves = sub_moves
        self.step = 0
        self.moves_made = 0
        self.turn_number = 1
        self.winner = None

    @property
    def is_over(self):
        return self.winner is not None

    @property
    def pending_decision(self):
        if self.is_over:
            return None
        if self.sub_moves == 2 and self.step == 1:
            return _Decision(self.to_move, [n for n in (0, 1) if n <= self.pile])
        opts = [n for n in (1, 2) if n <= self.pile]
        if self.coin and self.moves_made == 1 and 3 <= self.pile:
            opts.append(3)
        return _Decision(self.to_move, opts)

    def submit(self, n):
        self.pile -= n
        self.moves_made += 1
        if self.pile == 0 and (n > 0 or self.step == 1):
            self.winner = self.to_move
            return
        if self.sub_moves == 2 and self.step == 0:
            self.step = 1
            return
        self.step = 0
        self.to_move = 3 - self.to_move
        self.turn_number += 1

    def outcome_for(self, pid):
        return 0 if self.winner is None else (1 if self.winner == pid else -1)

    def copy(self):
        g = Nim(self.pile, self.to_move, self.coin, self.sub_moves)
        g.step, g.moves_made, g.turn_number, g.winner = self.step, self.moves_made, self.turn_number, self.winner
        return g


class NimCap:
    """A search capability over Nim: each determinization re-flips the
    hidden coin with probability `p_coin` (the searcher can't see it)."""

    def __init__(self, game, viewer, p_coin=0.0):
        self.game = game
        self.viewer = viewer
        self.p_coin = p_coin
        self.coins = []

    def fork_determinized(self, rng, resample=Resample.ALL, *, backend="replay"):
        g = self.game.copy()
        g.coin = rng.random() < self.p_coin
        self.coins.append(g.coin)
        return g

    def infoset(self):
        return self.game


class TerminalOnly:
    """Uniform priors, value 0 at every non-terminal leaf: the only signal
    is who actually wins, so backed-up values are exact."""

    def job(self, world, search, actor, actions, need_priors, to_value_state):
        if to_value_state:
            search.regime.value_state(search, world)
        priors = [1.0 / len(actions)] * len(actions) if need_priors else None
        return priors, float(world.outcome_for(search.searcher)) if world.is_over else 0.0
        yield  # pragma: no cover


def _nim_search(sims=1500, **kw):
    settings = SearchSettings(simulations=sims, leaves_in_flight=kw.pop("leaves_in_flight", 1), **kw)
    return Search(FullGame(), TerminalOnly(), lambda w, d: d.options[0], settings, seed=3)


def _search(search, game, viewer, p_coin=0.0):
    cap = NimCap(game, viewer, p_coin)
    return drive(search.search_gen(cap, game.pending_decision), None), cap


class TestBackupSigns(unittest.TestCase):
    def test_forced_win_goes_to_plus_one_from_either_seat(self):
        for seat in (1, 2):
            with self.subTest(seat=seat):
                game = Nim(4, to_move=seat)  # take 1 -> the opponent faces 3: lost
                res, _ = _search(_nim_search(), game, seat)
                self.assertEqual(res.choices[res.chosen], 1)
                self.assertGreater(res.value, 0.9)
                self.assertGreater(res.q[res.chosen], 0.9)

    def test_forced_loss_goes_to_minus_one_from_either_seat(self):
        for seat in (1, 2):
            with self.subTest(seat=seat):
                game = Nim(6, to_move=seat)  # multiples of 3 are lost
                res, _ = _search(_nim_search(), game, seat)
                self.assertLess(res.value, -0.9)
                for q in res.q:
                    self.assertLess(q, -0.8)

    def test_immediate_win_is_exactly_plus_one(self):
        res, _ = _search(_nim_search(sims=50), Nim(2, to_move=1), 1)
        self.assertEqual(res.choices[res.chosen], 2)
        self.assertEqual(res.q[res.chosen], 1.0)


class TestAvailabilityCounts(unittest.TestCase):
    def test_a_sometimes_legal_option_is_counted_only_when_legal(self):
        """At the opponent's node, "take 3" is legal only in determinizations
        where the hidden coin came up (p = 0.5). Its availability count must
        track exactly that -- about half of the node's visits -- while an
        always-legal option's count equals every visit."""
        game = Nim(9, to_move=1)
        search = _nim_search(sims=2000)
        res, cap = _search(search, game, 1, p_coin=0.5)
        worst_ratio = []
        for root_key, root_edge in search.root.edges.items():
            child = root_edge.child
            if child is None or ("value", 3) not in child.edges:
                continue
            always = child.edges[("value", 1)].A
            sometimes = child.edges[("value", 3)].A
            self.assertGreater(always, 100)
            ratio = sometimes / always
            worst_ratio.append(ratio)
            self.assertAlmostEqual(ratio, 0.5, delta=4 * (0.25 / always) ** 0.5 + 0.02)
        self.assertTrue(worst_ratio, "the coin-only option never appeared")
        self.assertAlmostEqual(sum(cap.coins) / len(cap.coins), 0.5, delta=0.06)

    def test_root_availability_equals_iterations(self):
        game = Nim(7, to_move=1)
        search = _nim_search(sims=300)
        res, _ = _search(search, game, 1)
        for edge in search.root.edges.values():
            self.assertEqual(edge.A, 300)


class TestSubtreeReuse(unittest.TestCase):
    def test_reused_root_carries_its_statistics_and_agrees_with_a_fresh_search(self):
        # Turns remove 1-3 in this variant, so multiples of 4 are lost; 9 is
        # won, and whichever first sub-move is taken, exactly one second
        # sub-move leaves the opponent a multiple of 4.
        game = Nim(9, to_move=1, sub_moves=2)
        search = _nim_search(sims=1200)
        res, _ = _search(search, game, 1)
        key = res.keys[res.chosen]
        child_visits = search.root.edges[key].child.visits
        search.note_own_action(key, game.turn_number)
        game.submit(res.choices[res.chosen])  # the first sub-move; same player decides again
        reused, _ = _search(search, game, 1)
        self.assertEqual(reused.stats["reused_visits"], child_visits)
        self.assertGreater(child_visits, 0)
        fresh, _ = _search(_nim_search(sims=1200 + child_visits), game, 1)
        self.assertEqual(reused.choices[reused.chosen], fresh.choices[fresh.chosen])

    def test_an_opponent_decision_in_between_prevents_reuse(self):
        game = Nim(8, to_move=1, sub_moves=2)
        search = _nim_search(sims=200)
        res, _ = _search(search, game, 1)
        search.note_own_action(res.keys[res.chosen], game.turn_number)
        search.note_foreign_decision()
        game.submit(res.choices[res.chosen])
        again, _ = _search(search, game, 1)
        self.assertEqual(again.stats["reused_visits"], 0)


class TestBatching(unittest.TestCase):
    def test_virtual_loss_waves_still_solve_the_position(self):
        for lif in (1, 4, 16):
            with self.subTest(leaves_in_flight=lif):
                res, _ = _search(_nim_search(sims=2000, leaves_in_flight=lif), Nim(4, to_move=1), 1)
                self.assertEqual(res.choices[res.chosen], 1)

    def test_run_searches_batches_every_generator_into_one_call_per_round(self):
        class CountingEvaluator(TerminalOnly):
            def job(self, world, search, actor, actions, need_priors, to_value_state):
                priors = [1.0 / len(actions)] * len(actions) if need_priors else None
                answers = yield [("req", id(world))]
                return priors, float(world.outcome_for(search.searcher)) if world.is_over else answers[0]

        class Client:
            calls = 0
            sizes = []

            def predict_many(self, reqs):
                Client.calls += 1
                Client.sizes.append(len(reqs))
                return [0.0] * len(reqs)

        gens = []
        for pile in (5, 7, 8):
            s = Search(FullGame(), CountingEvaluator(), lambda w, d: d.options[0], SearchSettings(simulations=40, leaves_in_flight=4), seed=pile)
            gens.append(s.search_gen(NimCap(Nim(pile), 1), Nim(pile).pending_decision))
        results = run_searches(gens, Client())
        self.assertEqual(len(results), 3)
        self.assertTrue(all(sum(r.visits) > 0 for r in results))
        # One call per round serves all three searches: far fewer calls
        # than evaluations, and the first round carries all three roots.
        self.assertEqual(Client.sizes[0], 3)
        self.assertLess(Client.calls, sum(Client.sizes))


class TestBeliefSampling(unittest.TestCase):
    """M6: sampled worlds always satisfy the count constraint, and follow
    the marginals."""

    def test_exact_hand_count_distinct_cards_from_the_candidates(self):
        from agent.search.leaf import sample_hand

        rng = random.Random(0)
        marginals = [0.0] * 72
        candidates = list(range(36, 72))
        for i in candidates:
            marginals[i] = 0.9 if i < 42 else 0.05
        counts = [0] * 72
        for _ in range(2000):
            hand = sample_hand(marginals, candidates, 6, rng)
            self.assertEqual(len(hand), 6)
            self.assertEqual(len(set(hand)), 6)
            self.assertTrue(set(hand) <= set(candidates))
            for i in hand:
                counts[i] += 1
        likely = sum(counts[36:42]) / 2000
        self.assertGreater(likely, 4.0)  # the six high-marginal cards dominate the samples

    def test_never_more_than_the_candidates(self):
        from agent.search.leaf import sample_hand

        self.assertEqual(sorted(sample_hand([0.5] * 72, [3, 9], 6, random.Random(1))), [3, 9])


# --------------------------------------------------------- KeyForge positions
INERT = Deck(name="Inert", pods={
    House.BROBNAR: ["Troll"] * 12,
    House.SANCTUM: ["Bulwark"] * 12,
    House.UNTAMED: ["Niffle Ape"] * 12,
})


def _position(setup, first_player=1, seed=11):
    """A game at the first player's first decision, with the given setup:
    both decks inert (no card moves Æmber except by reaping)."""
    game = Game(GameConfig(decks=(INERT, INERT), seed=seed, first_player=first_player, setup_script=setup))
    game.submit(False)
    game.submit(False)
    return game


def _search_agent_choice(game, regime, sims=200, seed=1):
    agent = SearchAgent(regime, settings=SearchSettings(simulations=sims), seed=seed)
    agent.on_game_start(game.pending_decision.player, {})
    cap = make_capability(PrivilegeLevel.SEARCH, game, game.pending_decision.player)
    choice = agent.decide(game.view_for(game.pending_decision.player), game.pending_decision, None, cap)
    return choice, agent.last_result


class TestKnownPositions(unittest.TestCase):
    RACE = [("set_keys", 1, 2), ("set_aember", 1, 5), ("set_keys", 2, 2), ("set_aember", 2, 5),
            ("put_creature", 1, "Troll", "left"), ("put_creature", 2, "Troll", "left")]

    def test_both_regimes_pick_the_house_that_wins_the_race(self):
        """Both players sit at 2 keys and 5 Æmber with a ready Troll. Brobnar
        (reap with Troll -> 6 Æmber -> forge first) wins; any other house
        loses the race."""
        for regime in ("within_turn", "full_game"):
            with self.subTest(regime=regime):
                game = _position(self.RACE)
                self.assertEqual(game.pending_decision.kind, DecisionKind.CHOOSE_HOUSE)
                choice, res = _search_agent_choice(game, regime)
                self.assertEqual(choice, House.BROBNAR, res and list(zip(res.keys, res.visits)))

    def test_within_turn_search_then_reaps(self):
        game = _position(self.RACE)
        game.submit(House.BROBNAR)
        d = game.pending_decision
        self.assertEqual(d.kind, DecisionKind.CHOOSE_ACTION)
        choice, res = _search_agent_choice(game, "within_turn")
        # Reaping now, or after playing the one card the first turn allows,
        # both reach 6 Æmber; ending the turn at 5 never does.
        self.assertNotEqual(type(choice).__name__, "EndTurn")
        end = next(i for i, c in enumerate(res.choices) if type(c).__name__ == "EndTurn")
        self.assertGreater(res.q[res.chosen], res.q[end])

    def test_forced_loss_is_exactly_minus_one_within_the_turn(self):
        """The opponent sits at 2 keys and 10 Æmber; nothing in an inert
        deck can stop them forging at the start of their turn, so every
        within-turn leaf is the lost game itself."""
        setup = [("set_keys", 2, 2), ("set_aember", 2, 10), ("put_creature", 1, "Troll", "left")]
        game = _position(setup)
        _choice, res = _search_agent_choice(game, "within_turn", sims=60)
        self.assertEqual(res.value, -1.0)
        self.assertTrue(all(q == -1.0 for q, n in zip(res.q, res.visits) if n))
        _choice, res = _search_agent_choice(_position(setup), "full_game", sims=200)
        self.assertLess(res.value, -0.5)


class TestSearchAgentConformance(unittest.TestCase):
    """The interface plan's Milestone K contract, for the search agent, at
    every privilege level (with a small budget, so it stays fast)."""

    def _run(self, privilege, regime="within_turn", n=2, seed0=0):
        agents = {}

        def make_agents(i):
            a = SearchAgent(regime, settings=SearchSettings(simulations=12), seed=100 + i)
            agents[i] = a
            return {1: a, 2: HeuristicBot(seed=i)}

        results = run_games(
            n, lambda i: GameConfig(decks=("fignor", "igor"), seed=seed0 + i, max_turns=40), make_agents,
            privilege={1: privilege}, concurrency=1,
        )
        return results, agents

    def test_legal_and_terminating_at_every_privilege_level(self):
        for privilege in (PrivilegeLevel.OBSERVATION, PrivilegeLevel.SEARCH, PrivilegeLevel.PRIVILEGED):
            for regime in ("within_turn", "full_game"):
                with self.subTest(privilege=privilege.value, regime=regime):
                    results, agents = self._run(privilege, regime)
                    for r in results:
                        self.assertIsNone(r.forfeit, r.forfeit)
                        self.assertIsNotNone(r.reason)
                    searched = any(a.last_result is not None for a in agents.values())
                    self.assertEqual(searched, privilege != PrivilegeLevel.OBSERVATION)

    def test_reproducible_from_its_seed(self):
        a, _ = self._run(PrivilegeLevel.SEARCH, n=1, seed0=5)
        b, _ = self._run(PrivilegeLevel.SEARCH, n=1, seed0=5)
        self.assertEqual(a[0].choice_record, b[0].choice_record)

    def test_deciding_never_mutates_the_live_game(self):
        game = Game(GameConfig(decks=("fignor", "igor"), seed=7, max_turns=30))
        agent = SearchAgent("full_game", settings=SearchSettings(simulations=10), seed=1)
        other = HeuristicBot(seed=2)
        agent.on_game_start(1, {})
        steps = 0
        while not game.is_over and steps < 60:
            d = game.pending_decision
            if d.player == 1:
                before = game.state_hash()
                choice = agent.decide(game.view_for(1), d, Budget(simulations=10), make_capability(PrivilegeLevel.SEARCH, game, 1))
                self.assertEqual(before, game.state_hash())
                self.assertTrue(d.validate(choice))
            else:
                choice = other.decide(game.view_for(2), d)
            game.submit(choice)
            steps += 1

    def test_exact_fork_needs_the_privileged_capability(self):
        game = Game(GameConfig(decks=("fignor", "igor"), seed=3))
        game.submit(False)
        game.submit(False)
        agent = SearchAgent("within_turn", settings=SearchSettings(simulations=5, resample=None), seed=1)
        d = game.pending_decision
        with self.assertRaises(PermissionError):
            agent.decide(game.view_for(d.player), d, None, make_capability(PrivilegeLevel.SEARCH, game, d.player))
        agent.decide(game.view_for(d.player), d, None, make_capability(PrivilegeLevel.PRIVILEGED, game, d.player))

    def test_registered_agents_resolve(self):
        import agent.agents.registry_entries  # noqa: F401
        from bots.registry import make_agent, privilege_of

        self.assertEqual(privilege_of("search-within-turn"), PrivilegeLevel.SEARCH)
        self.assertEqual(privilege_of("search-full-game-privileged"), PrivilegeLevel.PRIVILEGED)
        a = make_agent("search-within-turn", seed=1, simulations=7)
        self.assertEqual(a.search.settings.simulations, 7)


if __name__ == "__main__":
    unittest.main()
