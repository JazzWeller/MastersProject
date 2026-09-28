"""Playing strength of a checkpoint (Agent Training Plan, M3 gate G1 and
the M9 anchors): a network agent, batched across every concurrent game
through one in-process model, against any registered agent over paired
seeds (sim/paired.py -- both seat orders x both deck assignments).

    python -m ml.evaluate <checkpoint.kfc> --opponent heuristic --seeds 2000
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Optional

import torch

from agent.agents.net_agent import NetAgent
from bots.inference_client import InProcessInferenceClient
from bots.registry import make_agent
from sim.paired import run_paired

from .infer_server import TorchModel


def net_agent_from_model(model: TorchModel, *, mode: str = "policy", multi_select: Optional[str] = None, **kw) -> NetAgent:
    net_cfg = model.net.net_cfg
    return NetAgent(
        InProcessInferenceClient(model), mode=mode, multi_select=multi_select or net_cfg.get("multi_select", "enumerate"),
        policy_head=net_cfg.get("policy_head", "pointer"), enumerate_cap=int(net_cfg.get("enumerate_cap", 1024)), **kw,
    )


def evaluate_model(
    model: TorchModel, opponent: str = "heuristic", *, n_seeds: int = 2000, first_seed: int = 1_000_000,
    deck_x="fignor", deck_y="igor", max_turns: int = 200, concurrency: int = 256, multi_select: Optional[str] = None,
    mode: str = "policy",
) -> dict:
    agent = net_agent_from_model(model, mode=mode, multi_select=multi_select, seed=first_seed)
    seeds = list(range(first_seed, first_seed + n_seeds))
    t0 = time.perf_counter()
    rep, _results = run_paired(
        lambda i: agent, lambda i: make_agent(opponent, seed=first_seed + i), seeds, deck_x, deck_y,
        max_turns=max_turns, concurrency=concurrency,
    )
    out = rep.summary()
    out.update(opponent=opponent, seconds=round(time.perf_counter() - t0, 1), evaluations=model.evaluations)
    return out


def main():
    from .checkpoints import load_model

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("--opponent", default="heuristic")
    parser.add_argument("--seeds", type=int, default=2000)
    parser.add_argument("--multi-select", default=None)
    parser.add_argument("--mode", default="policy")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    net, _meta, _ = load_model(args.checkpoint)
    model = TorchModel(net, args.device)
    print(json.dumps(evaluate_model(model, args.opponent, n_seeds=args.seeds, multi_select=args.multi_select, mode=args.mode), indent=1))


if __name__ == "__main__":
    main()
