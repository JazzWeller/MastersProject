"""The shared search core (Agent Training Plan, Milestone M4): information-
set MCTS with PUCT, availability counts, virtual loss, root noise, subtree
reuse and batched leaf evaluation -- one implementation serving both search
regimes (M5) and every leaf estimator (M6).

**Tree.** A node is the next *branching* decision point after a path of
actions from the root; an edge is keyed by the action's `action_key` (the
option key, or the canonical multi-select key), never by an index -- so the
"same" action lines up across determinizations even though option lists
and orders differ between worlds. Per the interface plan's tree-node
guidance, nodes hold statistics only, never a live `Game`: every iteration
re-derives its state by determinizing the root and replaying the selected
actions.

**Selection (PUCT for information sets).**

    Q(s,a) = W(s,a) / N(s,a)                        (0 when N = 0)
    U(s,a) = c_puct * P(s,a) * sqrt(A(s,a)) / (1 + N(s,a))

over the actions legal in *this iteration's* world, where `A` -- the
availability count -- is the number of iterations in which the action was
legal (Cowling, Powley & Whitehouse, 2012). Without it, actions that are
only sometimes legal are systematically over-explored. `W` is stored from
the perspective of the player acting at the node, so the backup negates the
searcher's value at every node the opponent acts at.

**One iteration:** determinize the root (`fork_determinized` via the
search capability -- never the true hidden state), descend by PUCT applying
actions to the world, stop at a node that needs an evaluation (unexpanded,
or reached in a world with a newly-legal action), a regime leaf (M5) or the
end of the game; evaluate; back up; count availability at every node
visited.

**Batching.** Up to `leaves_in_flight` iterations descend before any is
evaluated (virtual loss keeps them apart), and their evaluation jobs are
batched. Jobs are generators that yield `agent.agents.requests.Request`
lists, so `run_searches` can drive many searches -- one per concurrent
game -- through a single `InferenceClient.predict_many` per round.

**No-network debug mode** (the plan's "build this first"): the heuristic
evaluator in `leaf.py` -- uniform priors, `bots.baseline_evaluator`'s value
-- needs no requests at all.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Sequence, Tuple

from keyforge.encoding import option_key
from keyforge.enums import DecisionKind, Resample

from ..multiselect import enumerate_candidates

MULTI_KINDS = (DecisionKind.CHOOSE_CARDS, DecisionKind.ORDER_EFFECTS)


def action_key(kind: DecisionKind, choice: Any) -> Any:
    """The tree-edge key for a submitted `choice` -- also what subtree reuse
    matches observed choices against. A multi-select submission keys by its
    member option keys: in order for ORDER_EFFECTS, as a set for
    CHOOSE_CARDS."""
    if isinstance(choice, list):
        keys = tuple(option_key(None, c) for c in choice)
        if kind == DecisionKind.ORDER_EFFECTS:
            return ("multi", keys)
        return ("multi", tuple(sorted(keys, key=repr)))
    return option_key(None, choice)


def legal_actions(decision, enumerate_cap: int) -> Optional[List[Tuple[Any, Any, Any]]]:
    """[(key, choice, index form)] for every legal submission of `decision`
    -- the index form is the option index (single choice) or the tuple of
    option indices (multi-select), which is what the network's heads score.
    None if a multi-select space exceeds `enumerate_cap` (the caller falls
    back to its fixed policy, never truncates)."""
    if decision.kind in MULTI_KINDS:
        ordered = decision.kind == DecisionKind.ORDER_EFFECTS
        try:
            cands = enumerate_candidates(ordered, len(decision.options), decision.min_n, decision.max_n, enumerate_cap)
        except ValueError:
            return None
        out = []
        for cand in cands:
            choice = [decision.options[i] for i in cand]
            out.append((action_key(decision.kind, choice), choice, cand))
        return out
    return [(option_key(decision, o), o, i) for i, o in enumerate(decision.options)]


def map_choice(decision, fork_choice: Any) -> Any:
    """A choice made against a *fork's* copy of `decision` (different Card
    objects), mapped onto `decision`'s own options by option key."""
    by_key = {option_key(decision, o): o for o in decision.options}
    if isinstance(fork_choice, list):
        return [by_key[option_key(None, c)] for c in fork_choice]
    return by_key[option_key(None, fork_choice)]


def _release(world) -> None:
    """Frees a finished simulation's world promptly (`Game.release`); a
    test double without it is left alone."""
    release = getattr(world, "release", None)
    if release is not None:
        release()


class Edge:
    __slots__ = ("N", "W", "P", "A", "vloss", "child")

    def __init__(self):
        self.N = 0
        self.W = 0.0
        self.P: Optional[float] = None
        self.A = 0
        self.vloss = 0
        self.child: Optional["Node"] = None


class Node:
    __slots__ = ("edges", "evaluated", "visits")

    def __init__(self):
        self.edges: Dict[Any, Edge] = {}
        self.evaluated = False
        self.visits = 0


@dataclass
class SearchSettings:
    simulations: int = 100
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.8
    dirichlet_eps: float = 0.25
    root_noise: bool = False
    virtual_loss: int = 1
    leaves_in_flight: int = 8
    # What each iteration's world resamples. `None` = the EXACT fork (the
    # privileged capability only) -- the hidden-information diagnostic's
    # upper bound, never an ordinary agent.
    resample: Optional[Resample] = Resample.ALL
    fork_backend: str = "auto"  # replay | copy | auto (copy at boundary decisions)
    enumerate_cap: int = 1024
    temperature: float = 0.0
    reuse: bool = True
    max_depth: int = 400  # safety net per descent, in applied decisions


@dataclass
class SearchResult:
    keys: List[Any]  # root actions, in `decision.options` / candidate order
    choices: List[Any]
    index_forms: List[Any]
    visits: List[int]
    priors: List[float]
    q: List[float]
    chosen: int
    value: float  # root value estimate, searcher's perspective
    stats: Dict[str, float] = field(default_factory=dict)

    @property
    def policy(self) -> List[float]:
        total = sum(self.visits)
        return [v / total for v in self.visits] if total else [1.0 / len(self.visits)] * len(self.visits)


@dataclass
class _Descent:
    path: List[Tuple[Node, Edge, int]]
    value: Optional[float] = None  # set when resolved without an evaluation (terminal)
    job: Any = None  # evaluation generator
    node: Optional[Node] = None  # node receiving priors, if any
    actions: Optional[List[Tuple[Any, Any, Any]]] = None


def dirichlet(rng: random.Random, alpha: float, n: int) -> List[float]:
    xs = [rng.gammavariate(alpha, 1.0) for _ in range(n)]
    total = sum(xs) or 1.0
    return [x / total for x in xs]


class Search:
    """One searcher's tree. `regime` (M5) decides which decisions branch
    and where leaves are; `evaluator` (M6) produces priors and values;
    `fixed_policy(world, decision) -> choice` resolves decisions the regime
    doesn't branch on."""

    def __init__(self, regime, evaluator, fixed_policy: Callable, settings: Optional[SearchSettings] = None, *, seed: Optional[int] = None):
        self.regime = regime
        self.evaluator = evaluator
        self.fixed_policy = fixed_policy
        self.settings = settings or SearchSettings()
        self.rng = random.Random(seed)
        self.root: Optional[Node] = None
        self.searcher: int = 0
        self.root_turn: int = 0
        self._reuse_from: Optional[Tuple[Node, Any, int]] = None  # (old root, key taken, turn)
        self._foreign_since_last = False

    # ------------------------------------------------------------ reuse --
    def note_own_action(self, key: Any, turn: int) -> None:
        self._reuse_from = (self.root, key, turn) if self.root is not None else None
        self._foreign_since_last = False

    def note_foreign_decision(self) -> None:
        """Another player's decision happened since our last search: its
        choice may be hidden information, so the tree is not re-rooted
        through it (keying reuse on a hidden choice would condition our
        statistics on it)."""
        self._foreign_since_last = True

    def _reuse_root(self, turn: int) -> Optional[Node]:
        if not self.settings.reuse or self._reuse_from is None or self._foreign_since_last:
            return None
        old_root, key, old_turn = self._reuse_from
        if old_turn != turn or old_root is None:
            return None
        edge = old_root.edges.get(key)
        return edge.child if edge is not None else None

    # ------------------------------------------------------------ search --
    def search_gen(self, capability, decision, budget=None) -> Generator[List[Any], List[Any], SearchResult]:
        """The search as a generator: yields lists of evaluation requests,
        receives their results, returns the `SearchResult`."""
        s = self.settings
        self.searcher = capability.viewer
        world0 = self._world(capability, random.Random(0))
        self.root_turn = world0.turn_number
        reused = self._reuse_root(self.root_turn)
        self.root = reused if reused is not None else Node()
        self._reuse_from = None
        stats = {"simulations": 0, "forks": 0, "fork_seconds": 0.0, "evaluations": 0, "terminal": 0, "max_depth": 0,
                 "depth_total": 0, "reused_visits": self.root.visits, "fallback_fixed": 0}

        root_actions = legal_actions(decision, s.enumerate_cap)
        if root_actions is None:
            stats["fallback_fixed"] += 1
            fork_choice = self.fixed_policy(world0, world0.pending_decision)
            choice = map_choice(decision, fork_choice)
            key = action_key(decision.kind, choice)
            return SearchResult([key], [choice], [None], [1], [1.0], [0.0], 0, 0.0, stats)
        # Root priors from the root information set (the searcher's own
        # decision -- identical in every determinization).
        root_job = self.evaluator.job(world0, self, self.searcher, root_actions, need_priors=True, to_value_state=False)
        root_priors, root_value = yield from self._run_one(root_job)
        stats["evaluations"] += 1
        root = self.root
        if s.root_noise:
            noise = dirichlet(self.rng, s.dirichlet_alpha, len(root_actions))
            root_priors = [(1 - s.dirichlet_eps) * p + s.dirichlet_eps * n for p, n in zip(root_priors, noise)]
        for (key, _c, _i), p in zip(root_actions, root_priors):
            e = root.edges.get(key)
            if e is None:
                e = root.edges[key] = Edge()
            e.P = p
        root.evaluated = True

        sims = (budget.simulations if budget is not None and budget.simulations else None) or s.simulations
        deadline = None
        if budget is not None and budget.wall_clock_seconds:
            deadline = time.perf_counter() + 0.8 * budget.wall_clock_seconds
        done = 0
        while done < sims:
            if deadline is not None and time.perf_counter() > deadline and done > 0:
                break
            wave = min(s.leaves_in_flight, sims - done)
            descents: List[_Descent] = []
            worlds = []
            for _ in range(wave):
                t0 = time.perf_counter()
                world = self._world(capability, self.rng)
                stats["forks"] += 1
                stats["fork_seconds"] += time.perf_counter() - t0
                d = self._descend(world, stats)
                if d.value is not None:
                    self._backup(d.path, d.value)
                    stats["terminal"] += 1
                    done += 1
                    _release(world)
                else:
                    descents.append(d)
                    worlds.append(world)
            if descents:
                results = yield from self._run_jobs_gen([d.job for d in descents])
                for d, (priors, value) in zip(descents, results):
                    if d.node is not None and priors is not None:
                        self._set_priors(d.node, d.actions, priors)
                    self._backup(d.path, value)
                    stats["evaluations"] += 1
                done += len(descents)
            # the jobs played their worlds to the value state: done with them
            for world in worlds:
                _release(world)
        stats["simulations"] = done
        cache = getattr(self.evaluator, "cache", None)
        if cache is not None:
            stats["cache_hits"] = cache.hits
            stats["cache_misses"] = cache.misses
        return self._result(decision, root_actions, root_priors, root_value, stats)

    def _world(self, capability, rng):
        s = self.settings
        if s.resample is None:
            if not hasattr(capability, "fork"):
                raise PermissionError("an exact-fork search needs the privileged capability")
            return capability.fork()
        return capability.fork_determinized(rng, s.resample, backend=s.fork_backend)

    def _run_one(self, job):
        results = yield from self._run_jobs_gen([job])
        return results[0]

    def _run_jobs_gen(self, jobs: List[Generator]) -> Generator[List[Any], List[Any], List[Tuple[Optional[List[float]], float]]]:
        """Advances every job generator in lockstep: each round, all their
        requests go out as one list; every job receives its own slice."""
        results: List[Any] = [None] * len(jobs)
        pending: Dict[int, List[Any]] = {}
        for i, job in enumerate(jobs):
            try:
                pending[i] = next(job)
            except StopIteration as stop:
                results[i] = stop.value
        while pending:
            order = list(pending)
            flat = [r for i in order for r in pending[i]]
            answers = (yield flat) if flat else []
            at = 0
            nxt: Dict[int, List[Any]] = {}
            for i in order:
                n = len(pending[i])
                try:
                    nxt[i] = jobs[i].send(answers[at : at + n])
                except StopIteration as stop:
                    results[i] = stop.value
                at += n
            pending = nxt
        return results

    def _descend(self, world, stats) -> _Descent:
        s = self.settings
        node = self.root
        path: List[Tuple[Node, Edge, int]] = []
        depth = 0
        while True:
            if world.is_over:
                self._note_depth(stats, depth)
                return _Descent(path, value=float(world.outcome_for(self.searcher)))
            d = world.pending_decision
            how = self.regime.classify(self, world, d)
            if how == "leaf":
                self._note_depth(stats, depth)
                return _Descent(path, job=self.evaluator.job(world, self, d.player, None, need_priors=False, to_value_state=False))
            if how == "fixed" or depth >= s.max_depth:
                world.submit(self.fixed_policy(world, d))
                depth += 1
                continue
            actions = legal_actions(d, s.enumerate_cap)
            if actions is None:
                stats["fallback_fixed"] += 1
                world.submit(self.fixed_policy(world, d))
                depth += 1
                continue
            if len(actions) == 1:
                world.submit(actions[0][1])  # forced: no node
                depth += 1
                continue
            for key, _c, _i in actions:
                e = node.edges.get(key)
                if e is None:
                    e = node.edges[key] = Edge()
                e.A += 1
            if not node.evaluated or any(node.edges[k].P is None for k, _c, _i in actions):
                self._note_depth(stats, depth)
                job = self.evaluator.job(world, self, d.player, actions, need_priors=True, to_value_state=True)
                return _Descent(path, job=job, node=node, actions=actions)
            key, choice = self._select(node, actions, d.player)
            edge = node.edges[key]
            edge.vloss += s.virtual_loss
            path.append((node, edge, d.player))
            world.submit(choice)
            depth += 1
            if edge.child is None:
                edge.child = Node()
            node = edge.child

    @staticmethod
    def _note_depth(stats, depth):
        stats["depth_total"] += depth
        if depth > stats["max_depth"]:
            stats["max_depth"] = depth

    def _select(self, node: Node, actions, actor: int):
        c = self.settings.c_puct
        best, best_key, best_choice = -math.inf, None, None
        ties = 0
        rand = self.rng.random
        for key, choice, _i in actions:
            e = node.edges[key]
            n = e.N + e.vloss
            # Virtual visits count as losses for the player to move here.
            q = (e.W - e.vloss) / n if n > 0 else 0.0
            u = c * (e.P or 0.0) * math.sqrt(e.A) / (1 + n)
            score = q + u
            if score > best:
                best, best_key, best_choice, ties = score, key, choice, 1
            elif score == best:
                # Exact ties (every unvisited action under uniform priors)
                # are broken uniformly at random -- never "the first
                # option", which is End Turn at every CHOOSE_ACTION.
                ties += 1
                if rand() * ties < 1.0:
                    best_key, best_choice = key, choice
        return best_key, best_choice

    def _set_priors(self, node: Node, actions, priors: List[float]) -> None:
        for (key, _c, _i), p in zip(actions, priors):
            e = node.edges[key]
            if e.P is None:
                e.P = p
        node.evaluated = True

    def _backup(self, path, value: float) -> None:
        """`value` is from the searcher's perspective; each edge stores it
        from the perspective of the player who chose it."""
        vl = self.settings.virtual_loss
        for node, edge, actor in path:
            edge.N += 1
            edge.W += value if actor == self.searcher else -value
            edge.vloss -= vl
            node.visits += 1

    def _result(self, decision, root_actions, root_priors, root_value, stats) -> SearchResult:
        root = self.root
        keys, choices, forms, visits, q = [], [], [], [], []
        for key, choice, form in root_actions:
            e = root.edges[key]
            keys.append(key)
            choices.append(choice)
            forms.append(form)
            visits.append(e.N)
            q.append(e.W / e.N if e.N else 0.0)
        chosen = self._choose(visits, root_priors, q)
        total = sum(visits)
        value = sum(v * qq for v, qq in zip(visits, q)) / total if total else root_value
        return SearchResult(keys, choices, forms, visits, list(root_priors), q, chosen, value, stats)

    def _choose(self, visits: List[int], priors: List[float], q: Optional[List[float]] = None) -> int:
        tau = self.settings.temperature
        if tau <= 0 or sum(visits) == 0:
            # Most visits; ties (common at tiny budgets) go to the better
            # value estimate, then the prior -- the option order only last.
            qq = q or [0.0] * len(visits)
            return max(range(len(visits)), key=lambda i: (visits[i], qq[i] if visits[i] else -2.0, priors[i], -i))
        weights = [v ** (1.0 / tau) for v in visits]
        r = self.rng.random() * sum(weights)
        for i, w in enumerate(weights):
            r -= w
            if r <= 0:
                return i
        return len(weights) - 1


def drive(gen: Generator, client) -> Any:
    """Runs one search generator to completion against `client` (an
    `InferenceClient`, or None when the evaluator never yields requests)."""
    try:
        reqs = next(gen)
        while True:
            answers = client.predict_many(reqs) if reqs else []
            reqs = gen.send(answers)
    except StopIteration as stop:
        return stop.value


def run_searches(gens: Sequence[Generator], client) -> List[Any]:
    """Drives several search generators at once, batching every round's
    requests from all of them into one `predict_many` -- the self-play
    actor's K concurrent games sharing one GPU call (M7)."""
    results: List[Any] = [None] * len(gens)
    pending: Dict[int, List[Any]] = {}
    for i, g in enumerate(gens):
        try:
            pending[i] = next(g)
        except StopIteration as stop:
            results[i] = stop.value
    while pending:
        order = list(pending)
        flat = [r for i in order for r in pending[i]]
        answers = client.predict_many(flat) if flat else []
        at = 0
        nxt = {}
        for i in order:
            n = len(pending[i])
            try:
                nxt[i] = gens[i].send(answers[at : at + n])
            except StopIteration as stop:
                results[i] = stop.value
            at += n
        pending = nxt
    return results


def run_searches_grouped(pairs: Sequence[Tuple[Generator, Any]]) -> List[Any]:
    """`run_searches` for generators that answer to different clients (two
    networks playing each other: gating, the M9 matrix). Each round, every
    client gets one `predict_many` with all of its generators' requests."""
    results: List[Any] = [None] * len(pairs)
    pending: Dict[int, List[Any]] = {}
    for i, (g, _c) in enumerate(pairs):
        try:
            pending[i] = next(g)
        except StopIteration as stop:
            results[i] = stop.value
    while pending:
        by_client: Dict[int, List[int]] = {}
        for i in pending:
            by_client.setdefault(id(pairs[i][1]), []).append(i)
        nxt = {}
        for ids in by_client.values():
            client = pairs[ids[0]][1]
            flat = [r for i in ids for r in pending[i]]
            answers = client.predict_many(flat) if (flat and client is not None) else []
            at = 0
            for i in ids:
                n = len(pending[i])
                try:
                    nxt[i] = pairs[i][0].send(answers[at : at + n])
                except StopIteration as stop:
                    results[i] = stop.value
                at += n
        pending = nxt
    return results
