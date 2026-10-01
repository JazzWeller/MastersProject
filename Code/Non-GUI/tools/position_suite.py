#!/usr/bin/env python3
"""The multi-select position suite (Agent Training Plan, M3 Screen 3's
second metric, reported in M9): on a fixed suite of CHOOSE_CARDS /
ORDER_EFFECTS positions, how often does each treatment -- enumerate,
sequential, independent top-k -- pick what a high-budget reference search
picks? Per-decision agreement is far more sensitive per unit of compute
than win rates (a 1-point effect needs ~10,000 games per pairing).

    python -m tools.position_suite --checkpoint <bc.kfc> --positions 300 --reference-sims 1000

Positions come from HeuristicBot mirror games and are stored as replay
data `(config, choice-record prefix)`, so the suite is reproducible and
can be re-scored against any later checkpoint. The reference is the M5
full-game search (`--reference-regime`) with the heuristic evaluator, or
with the checkpoint's network (`--reference-leaf student`).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import List

import torch

from agent.agents.net_agent import NetAgent
from agent.agents.search_agent import SearchAgent
from agent.search.core import SearchSettings, action_key
from bots.heuristic_bot import HeuristicBot
from bots.inference_client import InProcessInferenceClient
from bots.option_space import choice_space_size
from keyforge.capabilities import make_capability
from keyforge.config import GameConfig
from keyforge.enums import DecisionKind, PrivilegeLevel
from keyforge.game import Game
from keyforge.replay import config_from_dict, config_to_dict, replay
from ml.checkpoints import load_model
from ml.infer_server import TorchModel

MULTI = (DecisionKind.CHOOSE_CARDS, DecisionKind.ORDER_EFFECTS)


def build_suite(n_positions: int, seed: int = 0, decks=("fignor", "igor")) -> List[dict]:
    suite = []
    g = 0
    while len(suite) < n_positions:
        game = Game(GameConfig(decks=decks if g % 2 == 0 else decks[::-1], seed=seed * 100_003 + g, max_turns=200))
        bots = {1: HeuristicBot(seed=g), 2: HeuristicBot(seed=g + 1)}
        while not game.is_over and len(suite) < n_positions:
            d = game.pending_decision
            if d.kind in MULTI and choice_space_size(d) >= 2:
                suite.append({"config": config_to_dict(game.config), "prefix": list(game.choice_record), "kind": d.kind.name,
                              "space": choice_space_size(d)})
            game.submit(bots[d.player].decide(game.view_for(d.player), d))
        g += 1
    return suite


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--positions", type=int, default=300)
    parser.add_argument("--reference-sims", type=int, default=1000)
    parser.add_argument("--reference-regime", default="full_game")
    parser.add_argument("--reference-leaf", choices=("heuristic", "student"), default="heuristic")
    parser.add_argument("--suite", default=None, help="a saved suite JSON (built and saved here if missing)")
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.suite and os.path.exists(args.suite):
        suite = json.load(open(args.suite))
    else:
        suite = build_suite(args.positions)
        if args.suite:
            json.dump(suite, open(args.suite, "w"))
    net, _meta, _ = load_model(args.checkpoint)
    model = TorchModel(net, args.device if torch.cuda.is_available() else "cpu")
    client = InProcessInferenceClient(model)
    treatments = {t: NetAgent(client, multi_select=t, seed=1) for t in ("enumerate", "sequential", "topk")}
    hits = {t: 0 for t in treatments}
    share = {t: 0.0 for t in treatments}
    by_kind = {}
    t0 = time.time()
    for i, pos in enumerate(suite):
        game = replay(config_from_dict(pos["config"]), pos["prefix"])
        d = game.pending_decision
        pid = d.player
        ref = SearchAgent(args.reference_regime, leaf=args.reference_leaf, client=client if args.reference_leaf != "heuristic" else None,
                          settings=SearchSettings(simulations=args.reference_sims), seed=i)
        ref.on_game_start(pid, {})
        ref.decide(game.view_for(pid), d, None, make_capability(PrivilegeLevel.SEARCH, game, pid))
        res = ref.last_result
        total = sum(res.visits) or 1
        visit_share = {k: v / total for k, v in zip(res.keys, res.visits)}
        ref_key = res.keys[res.chosen]
        cap = make_capability(PrivilegeLevel.OBSERVATION, game, pid)
        kind = by_kind.setdefault(pos["kind"], {t: [0, 0] for t in treatments})
        for t, agent in treatments.items():
            choice = agent.decide(None, d, None, cap)
            key = action_key(d.kind, choice)
            hit = key == ref_key
            hits[t] += hit
            share[t] += visit_share.get(key, 0.0)
            kind[t][0] += hit
            kind[t][1] += 1
    n = len(suite)
    out = {
        "positions": n, "reference": f"{args.reference_regime}/{args.reference_leaf}@{args.reference_sims}",
        "agreement": {t: round(hits[t] / n, 4) for t in treatments},
        "reference_visit_share": {t: round(share[t] / n, 4) for t in treatments},
        "by_kind": {k: {t: round(h / m, 4) if m else None for t, (h, m) in v.items()} for k, v in by_kind.items()},
        "seconds": round(time.time() - t0, 1),
    }
    print(json.dumps(out, indent=1))
    if args.out:
        json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
