"""A minimal, real search agent built entirely from existing Milestone D/G
infrastructure (Agent Interface Plan): for each candidate `CHOOSE_ACTION`
option, samples a few forks (determinized under `SEARCH`, exact under
`PRIVILEGED`), applies the option, rolls out to the end of that turn with
a `HeuristicBot` policy, and scores the result with the no-network
heuristic value function (Milestone J). Every other decision kind is
delegated straight to the rollout policy.

Not competitive with a trained agent -- it exists to give Milestone J's
diagnostics (`tools/diag_hidden_info.py`, `tools/diag_search_curve.py`)
something real to measure, and to prove the search seams (`fork`/
`fork_determinized`, `run_until`, capabilities, `option_key`) compose the
way the plan says they should. `n_samples` is this bot's "simulation
count" knob.
"""

from __future__ import annotations

import random as _random
from typing import Optional

from keyforge.encoding import option_key
from keyforge.enums import DecisionKind, Resample
from keyforge.game import until_end_of_turn

from .base import Controller
from .baseline_evaluator import heuristic_value
from .heuristic_bot import HeuristicBot


class DeterminizedRolloutBot(Controller):
    def __init__(self, seed=None, n_samples: int = 8, resample: Resample = Resample.ALL):
        self.rng = _random.Random(seed)
        self.n_samples = max(1, n_samples)
        self.resample = resample
        self._rollout_policy = HeuristicBot(seed=seed)

    def decide(self, view, decision, budget=None, capability=None):
        if decision.kind != DecisionKind.CHOOSE_ACTION or len(decision.options) <= 1 or capability is None:
            return self._rollout_policy.decide(view, decision, budget, capability)
        scores = [self._score_option(decision.player, decision, option, capability) for option in decision.options]
        best = max(range(len(decision.options)), key=lambda i: scores[i])
        return decision.options[best]

    def _branch(self, capability):
        """Privileged: the exact fork. Search: a determinized one.
        Observation-only (or no capability at all): can't fork -- callers
        must not reach here in that case."""
        if hasattr(capability, "fork"):
            return capability.fork()
        if hasattr(capability, "fork_determinized"):
            return capability.fork_determinized(self.rng, resample=self.resample)
        return None

    def _score_option(self, pid: int, decision, option, capability) -> float:
        target_key = option_key(decision, option)
        total = 0.0
        samples = 0
        for _ in range(self.n_samples):
            branch = self._branch(capability)
            if branch is None or branch.is_over:
                continue
            bd = branch.pending_decision
            match = next((o for o in bd.options if option_key(bd, o) == target_key), None)
            if match is None:
                continue  # shouldn't happen for a faithful fork, but never crash a search over it
            branch.submit(match)
            if not branch.is_over:
                branch.run_until(
                    until_end_of_turn(branch),
                    lambda g, d: self._rollout_policy.decide(g.view_for(d.player), d),
                )
            score = branch.outcome_for(pid) if branch.is_over else heuristic_value(branch.view_for(pid))
            total += score
            samples += 1
        return total / samples if samples else 0.0
