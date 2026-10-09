#!/usr/bin/env python3
"""Screen 5 with v2 (Agent Observation Plan, O10: generalization to held-out
cards). WSL, CUDA.

    python -m tools.screen5_v2 --run screen5-v2 [--games 6000] [--stages data,v1,v2]

As the training plan's Screen 5: random legal decks over every card, 20% of
the cards held out of the training games entirely; evaluation games each
include held-out cards, and their decisions are split by whether an option
involves one. Trained on the same games:
- v1 (`identity: both`), the baseline, rerun on today's engine;
- v2 with all static card meaning, and with attributes only (O10's A7).
Every random-deck game has its own deck pair, so the value head can memorize
outcomes from the decklists: its log-loss on the training corpus's own
validation games is the honest one.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np


def _involves_v1(corpus, idx, held_ids):
    from tools.run_screens import _involves

    return _involves(corpus, idx, held_ids)


def _involves_v2(corpus, held_ids) -> np.ndarray:
    """Per record of an (unpacked) v2 corpus: does any option point at an
    entity whose card is held out?"""
    from agent import spec_v2 as S
    from ml.dataset_v2 import _NARROW_IDX

    card_col = _NARROW_IDX.index(S.ENTITY.i["card"])
    ptr_col = S.OPTION.i["pointer"]
    held = np.asarray(sorted(held_ids))
    out = []
    for sh in corpus.shards:
        a = sh.a
        cards = a["ent_i"][:, :, card_col]
        is_held = np.isin(cards, held)
        off = a["option_off"]
        ptr = a["option_i"][:, ptr_col]
        rec = np.repeat(np.arange(sh.size), np.diff(off))
        ok = ptr >= 0
        hit = np.zeros(sh.size, bool)
        hit[rec[ok][is_held[rec[ok], ptr[ok]]]] = True
        out.append(hit)
        sh.release()
    return np.concatenate(out)


def _metrics(rep: dict) -> dict:
    t1 = rep["top1_by_kind"]
    ms = rep["multi_select_exact"]
    return {"CHOOSE_ACTION_top1": t1.get("CHOOSE_ACTION", {}).get("accuracy"),
            "CHOOSE_HOUSE_top1": t1.get("CHOOSE_HOUSE", {}).get("accuracy"),
            "CHOOSE_CARDS_enumerate": ms["enumerate"]["CHOOSE_CARDS"]["accuracy"],
            "value_logloss": rep["value"]["logloss"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="screen5-v2")
    parser.add_argument("--games", type=int, default=6000)
    parser.add_argument("--stages", default="data,v1,v2")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=None, help="default: the config's (4)")
    args = parser.parse_args()
    stages = set(args.stages.split(","))

    import copy

    import torch

    from agent import config as config_mod
    from ml import resume
    from keyforge.cards.vocabulary import CARD_VOCAB
    from sim import data_root
    from sim.bc_corpus import held_out_cards
    from tools.run_screens import stage_data

    cfg = config_mod.resolve("tier0_generalization.json")
    if args.epochs:
        cfg["bc"]["epochs"] = args.epochs
    held = held_out_cards(cfg["seed"])
    held_ids = {CARD_VOCAB[n] for n in held}
    pool = dict(cfg["pool"], deck_source="random")
    out_dir = data_root.resolve(os.path.join("runs", args.run))
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, "screen5_v2.json")
    report = json.load(open(report_path)) if os.path.exists(report_path) else {}

    def save():
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1, sort_keys=True)

    t0 = time.time()
    train_v1 = stage_data(copy.deepcopy(cfg), args.workers, tag="gen-train", counts={"heuristic": args.games},
                          excluded=sorted(held), pool=pool)
    eval_v1 = stage_data(copy.deepcopy(cfg), args.workers, tag="gen-eval", counts={"heuristic": max(500, args.games // 4)},
                         required=sorted(held), pool=pool)
    from ml.dataset_v2 import CorpusV2, PackedCorpusV2, encode_corpus_v2, pack_corpus_v2

    roots = {}
    for name, enc in (("train", train_v1), ("eval", eval_v1)):
        root = os.path.dirname(enc)
        encode_corpus_v2(os.path.join(root, "records"), os.path.join(root, "encoded_v2"), workers=args.workers)
        roots[name] = root
    # training draws from the packed form (each game's positions spread out)
    pack_corpus_v2(os.path.join(roots["train"], "encoded_v2"), os.path.join(roots["train"], "packed_v2"), workers=args.workers)
    report["data"] = {"held_out_cards": len(held), "games": args.games, "seconds": round(time.time() - t0)}
    save()
    device = torch.device("cuda")

    if "v1" in stages and "v1" not in report:
        from ml.bc_train import evaluate, train
        from ml.dataset import Corpus

        tr, ev = Corpus.from_dir(train_v1), Corpus.from_dir(eval_v1)
        idx = ev.indices()
        inv = _involves_v1(ev, idx, held_ids)
        icfg = config_mod._deep_merge(cfg, {"network": {"identity": "both"}})
        state = os.path.join(out_dir, "state_v1.pt")  # pausable (ml/resume.py)
        model, rep = train(icfg, tr, device=device, state_path=state)
        report["v1"] = {"seen": _metrics(evaluate(model, ev, idx[~inv], policy_head="pointer", cap=1024, device=device)),
                        "held_out": _metrics(evaluate(model, ev, idx[inv], policy_head="pointer", cap=1024, device=device)),
                        "value_logloss_unseen_games": rep["value"]["logloss"],
                        "eval_involving_held_out": int(inv.sum()), "eval_positions": int(len(idx))}
        save()
        resume.finished(state)

    if "v2" in stages:
        from ml.bc_train_v2 import evaluate as evaluate_v2
        from ml.bc_train_v2 import train as train_v2

        tr = PackedCorpusV2(os.path.join(roots["train"], "packed_v2"), history="none")
        ev = CorpusV2.from_dir(os.path.join(roots["eval"], "encoded_v2"), history="none")
        inv = _involves_v2(ev, held_ids)
        idx = np.arange(ev.size)
        for name, static in (("v2_all_static", ["attr", "text", "sig", "embedding"]), ("v2_attr_only", ["attr"])):
            if name in report:
                continue
            vcfg = config_mod.resolve("tier0b_bc.json", {"network": {"history_arch": "none", "static": static},
                                                         "bc": {"epochs": cfg["bc"]["epochs"]}})
            state = os.path.join(out_dir, f"state_{name}.pt")
            model, rep = train_v2(vcfg, tr, device=device, state_path=state)
            report[name] = {"seen": _metrics(evaluate_v2(model, ev, idx[~inv], cap=1024, device=device)),
                            "held_out": _metrics(evaluate_v2(model, ev, idx[inv], cap=1024, device=device)),
                            "value_logloss_unseen_games": rep["value"]["logloss"],
                            "eval_involving_held_out": int(inv.sum()), "eval_positions": int(len(idx))}
            save()
            resume.finished(state)
    print(json.dumps(report, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
