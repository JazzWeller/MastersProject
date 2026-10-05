#!/usr/bin/env python3
"""The v1 baselines (Agent Observation Plan, Milestone O0, item 3), on fixed
seeds, under `$KEYFORGE_DATA/runs/obs-baseline/`:

- value log-loss by turn bucket, for turn-boundary positions and mid-turn
  positions separately;
- belief log-loss against "uniform over consistent worlds";
- encode time per decision (extract + encode, CPU).

Generalization to held-out cards is Screen 5, rerun into the same run by

    python -m tools.run_screens --run obs-baseline --stages generalization --data <Tier 0 corpus>

and BC top-1 per decision kind is Tier 0's own (also reported here, from the
same evaluation).

    python -m tools.obs_baseline --data /root/keyforge-data/bc/eaf9fce66a855b3e/encoded
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import time

import torch

from agent.features import encode
from agent.telemetry import Run
from bots.heuristic_bot import HeuristicBot
from keyforge.config import GameConfig
from keyforge.game import Game
from keyforge.infoset import build_infoset
from ml.bc_train import evaluate
from ml.checkpoints import CheckpointStore, load_model
from ml.dataset import Corpus
from sim import data_root


def encode_timing(n_games: int, seed: int = 0) -> dict:
    """Extract and encode at every decision of `n_games` HeuristicBot
    Fignor/Igor games, timed separately; the engine's own cost per decision
    alongside, for scale."""
    extract_us, encode_us, engine_us = [], [], []
    for g in range(n_games):
        game = Game(GameConfig(decks=("fignor", "igor"), seed=seed + g, max_turns=200))
        bots = {1: HeuristicBot(seed=seed + g), 2: HeuristicBot(seed=seed + g + 1)}
        while not game.is_over:
            d = game.pending_decision
            t0 = time.perf_counter()
            info = build_infoset(game, d.player)
            t1 = time.perf_counter()
            encode(info)
            t2 = time.perf_counter()
            choice = bots[d.player].decide(game.view_for(d.player), d)
            t3 = time.perf_counter()
            game.submit(choice)
            t4 = time.perf_counter()
            extract_us.append((t1 - t0) * 1e6)
            encode_us.append((t2 - t1) * 1e6)
            engine_us.append((t4 - t3) * 1e6)

    def summary(xs):
        return {"mean": round(statistics.fmean(xs), 1), "median": round(statistics.median(xs), 1)}

    return {"games": n_games, "decisions": len(extract_us), "extract_us": summary(extract_us),
            "encode_us": summary(encode_us), "engine_submit_us": summary(engine_us)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="obs-baseline")
    parser.add_argument("--config", default="tier0_bc.json")
    parser.add_argument("--data", required=True, help="the Tier 0 encoded corpus")
    parser.add_argument("--checkpoint-run", default="tier0", help="the run whose main checkpoint is evaluated")
    parser.add_argument("--encode-games", type=int, default=40)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    run = Run.resume_or_create(args.run, args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_path = os.path.join(run.root, "baseline.json")
    report = json.load(open(out_path)) if os.path.exists(out_path) else {"config_hash": run.config_hash}

    src_root = os.path.join(data_root.resolve("runs"), args.checkpoint_run)
    screens = json.load(open(os.path.join(src_root, "screens.json")))
    ckpt = screens["main"]["checkpoint"]
    net, _meta, _ = load_model((CheckpointStore(os.path.join(src_root, "checkpoints")), ckpt[:16]), device=str(device))
    corpus = Corpus.from_dir(data_root.resolve(args.data))
    val_idx = corpus.indices(split=1)
    t0 = time.perf_counter()
    rep = evaluate(net, corpus, val_idx, policy_head=net.net_cfg.get("policy_head", "pointer"),
                   cap=1024, device=device)
    report["checkpoint"] = {"run": args.checkpoint_run, "hash": ckpt, "data": args.data, "val_positions": int(len(val_idx)),
                            "eval_seconds": round(time.perf_counter() - t0, 1)}
    report["top1_by_kind"] = rep["top1_by_kind"]
    report["multi_select_exact"] = rep["multi_select_exact"]
    report["value"] = rep["value"]
    report["belief"] = rep["belief"]
    run.journal("obs_baseline_eval", game_index=0, checkpoint=ckpt)

    random.seed(0)
    report["encode_timing"] = encode_timing(args.encode_games)
    run.journal("obs_baseline_encode", game_index=0, **report["encode_timing"]["extract_us"])
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    v = report["value"]
    print(json.dumps({"value": {k: v[k] for k in ("logloss", "constant_logloss")},
                      "by_position": {k: {kk: vv for kk, vv in p.items() if kk != "by_turn"} for k, p in v["by_position"].items()},
                      "belief": {k: report["belief"][k] for k in ("n", "logloss", "uniform_logloss")},
                      "encode_timing": report["encode_timing"]}, indent=1))
    print(f"report: {out_path}")


if __name__ == "__main__":
    main()
