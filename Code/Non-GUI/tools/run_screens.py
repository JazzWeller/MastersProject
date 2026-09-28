#!/usr/bin/env python3
"""Tier 0 end to end (Agent Training Plan, Milestone M3 and gate G1): every
cheap screen, from an empty data root to a written report.

    python -m tools.run_screens --run tier0 [--config tier0_bc.json] [--stages all]

Stages, each idempotent (a finished stage's outputs are reused):

1. `data`     -- sim.bc_corpus records, then ml.dataset encoded shards,
                 under $KEYFORGE_DATA/bc/<corpus hash>/ (shared by every run
                 with the same pool/counts/seed/engine).
2. `main`     -- Screens 1-3 (+ belief/oracle) on the configured network.
3. `g1`       -- the BC agent, no search, vs HeuristicBot over paired seeds.
                 Gate G1: >= 40%.
4. `ablations`-- Screen 4: pointer vs fixed head, identity id/attr/both,
                 2/4/6 layers, without DecisionIntent.
5. `generalization` -- Screen 5: random legal decks with 20% of the 370
                 cards held out of training entirely; imitation accuracy on
                 decisions involving held-out cards vs seen ones, for each
                 identity encoding.

Writes `screens.json` and `screens.md` into the run directory, and journals
every stage (with its verdict, for G1).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import time

import numpy as np
import torch

from agent import config as config_mod
from agent.telemetry import Run
from ml.bc_train import evaluate, train
from ml.checkpoints import CheckpointStore, load_model, save_model
from ml.dataset import Corpus, encode_corpus
from ml.evaluate import evaluate_model
from ml.infer_server import TorchModel
from sim import data_root
from sim.bc_corpus import generate, held_out_cards

ABLATIONS = {
    "fixed_vocabulary_head": {"network": {"policy_head": "fixed"}},
    "identity_attributes_only": {"network": {"identity": "attr"}},
    "identity_id_only": {"network": {"identity": "id"}},
    "layers_2": {"network": {"layers": 2}},
    "layers_6": {"network": {"layers": 6}},
    "no_decision_intent": {"network": {"ablate_globals": ["intent"]}},
}


def corpus_dir(cfg: dict, tag: str = "") -> str:
    key = json.dumps({"pool": cfg["pool"], "bc": {k: cfg["bc"][k] for k in ("heuristic_games", "epsilon_games", "random_games", "epsilon")},
                      "seed": cfg["seed"], "engine": cfg["code"]["rules_hash"], "tag": tag}, sort_keys=True)
    return os.path.join(data_root.resolve("bc"), hashlib.sha256(key.encode()).hexdigest()[:16])


def stage_data(cfg: dict, workers: int, tag: str = "", counts=None, excluded=(), required=(), pool=None) -> str:
    root = corpus_dir(cfg, tag)
    bc = cfg["bc"]
    counts = counts or {"heuristic": bc["heuristic_games"], "epsilon": bc["epsilon_games"], "random": bc["random_games"]}
    t0 = time.perf_counter()
    generate(os.path.join(root, "records"), counts, run_seed=cfg["seed"], pool=pool or cfg["pool"], epsilon=bc["epsilon"],
             workers=workers, excluded=excluded, required=required)
    t1 = time.perf_counter()
    encode_corpus(os.path.join(root, "records"), os.path.join(root, "encoded"), workers=workers)
    print(f"[data] {root}: records {t1 - t0:.0f}s, encoding {time.perf_counter() - t1:.0f}s", flush=True)
    return os.path.join(root, "encoded")


def _key_metrics(report: dict) -> dict:
    top1 = report["top1_by_kind"]
    ms = report["multi_select_exact"]
    return {
        "CHOOSE_ACTION_top1": top1.get("CHOOSE_ACTION", {}).get("accuracy"),
        "CHOOSE_HOUSE_top1": top1.get("CHOOSE_HOUSE", {}).get("accuracy"),
        "CHOOSE_CARDS_enumerate": ms["enumerate"]["CHOOSE_CARDS"]["accuracy"],
        "CHOOSE_CARDS_sequential": ms["sequential"]["CHOOSE_CARDS"]["accuracy"],
        "CHOOSE_CARDS_topk": ms["topk"]["CHOOSE_CARDS"]["accuracy"],
        "value_logloss": report["value"]["logloss"],
        "value_constant_logloss": report["value"]["constant_logloss"],
    }


def _involves(corpus: Corpus, idx: np.ndarray, card_ids: set) -> np.ndarray:
    """Per index: does any option point at an entity whose card is in
    `card_ids`?"""
    out = np.zeros(len(idx), dtype=bool)
    for j, gi in enumerate(idx):
        s = int(np.searchsorted(corpus.offsets, gi, side="right") - 1)
        a = corpus.shards[s].a
        i = int(gi - corpus.offsets[s])
        lo, hi = a["opt_off"][i], a["opt_off"][i + 1]
        ptr = a["pointers"][lo:hi]
        ids = a["card_ids"][i]
        out[j] = any(p >= 0 and int(ids[p]) in card_ids for p in ptr)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--config", default="tier0_bc.json")
    parser.add_argument("--stages", default="all", help="comma list: data,main,g1,ablations,generalization")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--g1-seeds", type=int, default=2000)
    parser.add_argument("--ablation-epochs", type=int, default=None, help="default: the config's epochs")
    parser.add_argument("--generalization-games", type=int, default=6000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    stages = {"data", "main", "g1", "ablations", "generalization"} if args.stages == "all" else set(args.stages.split(","))

    run = Run.resume_or_create(args.run, args.config)
    cfg = run.config
    workers = args.workers or cfg["bc"]["workers"]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    report_path = os.path.join(run.root, "screens.json")
    report = json.load(open(report_path)) if os.path.exists(report_path) else {"config_hash": run.config_hash}
    store = CheckpointStore(os.path.join(run.root, "checkpoints"))

    def save_report():
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)
        write_markdown(run, report)

    data_dir = stage_data(cfg, workers) if ("data" in stages or not os.path.isdir(corpus_dir(cfg) + "/encoded")) else corpus_dir(cfg) + "/encoded"
    corpus = Corpus.from_dir(data_dir)
    report["corpus"] = {"dir": data_dir, "positions": corpus.size}
    run.journal("screens_data_ready", game_index=0, positions=corpus.size, dir=data_dir)

    if "main" in stages and "main" not in report:
        model, rep = train(cfg, corpus, device=device, metrics=run.metrics("learner"))
        ckpt, digest = save_model(model, store, config_hash=run.config_hash, extra={"kind": "bc", "stage": "main"})
        rep["checkpoint"] = ckpt
        report["main"] = rep
        run.journal("screens_main_done", game_index=0, checkpoint=ckpt)
        save_report()

    if "g1" in stages and "g1" not in report:
        net, _meta, _ = load_model((store, report["main"]["checkpoint"][:16]))
        g1 = evaluate_model(TorchModel(net, str(device)), "heuristic", n_seeds=args.g1_seeds, deck_x=cfg["pool"]["decks"][0], deck_y=cfg["pool"]["decks"][1])
        g1_random = evaluate_model(TorchModel(net, str(device)), "random", n_seeds=max(50, args.g1_seeds // 10))
        g1["verdict"] = "PASS" if g1["a_score"] >= 0.40 else "FAIL"
        report["g1"] = {"vs_heuristic": g1, "vs_random": g1_random}
        run.journal("gate_G1", game_index=0, verdict=g1["verdict"], score=g1["a_score"], games=g1["games"])
        save_report()

    if "ablations" in stages:
        report.setdefault("ablations", {})
        for name, over in ABLATIONS.items():
            if name in report["ablations"]:
                continue
            acfg = config_mod._deep_merge(cfg, over)
            if args.ablation_epochs:
                acfg["bc"]["epochs"] = args.ablation_epochs
            model, rep = train(acfg, corpus, device=device)
            report["ablations"][name] = {"override": over, **_key_metrics(rep), "params": rep["training"]["params"]}
            run.journal("screen4_ablation", game_index=0, ablation=name, **{k: v for k, v in _key_metrics(rep).items()})
            save_report()

    if "generalization" in stages and "generalization" not in report:
        held = held_out_cards(cfg["seed"])
        gcfg = copy.deepcopy(cfg)
        gpool = dict(cfg["pool"], deck_source="random")
        n = args.generalization_games
        train_dir = stage_data(gcfg, workers, tag="gen-train", counts={"heuristic": n}, excluded=sorted(held), pool=gpool)
        eval_dir = stage_data(gcfg, workers, tag="gen-eval", counts={"heuristic": max(500, n // 4)}, required=sorted(held), pool=gpool)
        tr = Corpus.from_dir(train_dir)
        ev = Corpus.from_dir(eval_dir)
        from keyforge.cards.vocabulary import CARD_VOCAB

        held_ids = {CARD_VOCAB[n_] for n_ in held}
        all_idx = ev.indices()
        involves = _involves(ev, all_idx, held_ids)
        gen = {"held_out_cards": len(held), "train_positions": tr.size, "eval_positions": ev.size,
               "eval_involving_held_out": int(involves.sum())}
        for ident in ("both", "attr", "id"):
            icfg = config_mod._deep_merge(gcfg, {"network": {"identity": ident}})
            model, _rep = train(icfg, tr, device=device)
            ph = icfg["network"].get("policy_head", "pointer")
            cap = int(icfg["network"].get("enumerate_cap", 1024))
            seen = evaluate(model, ev, all_idx[~involves], policy_head=ph, cap=cap, device=device)
            unseen = evaluate(model, ev, all_idx[involves], policy_head=ph, cap=cap, device=device)
            gen[ident] = {"seen": _key_metrics(seen), "held_out": _key_metrics(unseen)}
            run.journal("screen5_identity", game_index=0, identity=ident,
                        seen=gen[ident]["seen"]["CHOOSE_ACTION_top1"], held_out=gen[ident]["held_out"]["CHOOSE_ACTION_top1"])
        report["generalization"] = gen
        save_report()

    save_report()
    print(f"report: {report_path}")


def write_markdown(run: Run, report: dict) -> None:
    lines = [f"# Tier 0 screens -- run `{run.run_id}`", "", f"Config hash `{run.config_hash[:16]}`.", ""]
    if "corpus" in report:
        lines += [f"Corpus: {report['corpus']['positions']:,} labelled decisions.", ""]
    main = report.get("main")
    if main:
        t = main["training"]
        lines += ["## Screens 1-3 (main network)", "",
                  f"{t['params']:,} parameters, {t['steps']} steps, {t['seconds']}s.", "",
                  "| Decision kind | top-1 | n |", "|---|---|---|"]
        for k, v in main["top1_by_kind"].items():
            lines.append(f"| {k} | {v['accuracy']:.3f} | {v['n']} |")
        lines += ["", "| Multi-select treatment | CHOOSE_CARDS exact | ORDER_EFFECTS exact |", "|---|---|---|"]
        for t_, d in main["multi_select_exact"].items():
            cc, oe = d["CHOOSE_CARDS"], d["ORDER_EFFECTS"]
            fmt = lambda x: "--" if x["accuracy"] is None else f"{x['accuracy']:.3f} (n={x['n']})"
            lines.append(f"| {t_} | {fmt(cc)} | {fmt(oe)} |")
        v = main["value"]
        lines += ["", f"Value: log-loss {v['logloss']} vs constant {v['constant_logloss']} (MSE {v['mse']}).", "",
                  "| Turns | n | log-loss | constant | accuracy |", "|---|---|---|---|---|"]
        for row in v["by_turn"]:
            lines.append(f"| {row['turns']} | {row['n']} | {row['logloss']} | {row['constant_logloss']} | {row['accuracy']} |")
        if main.get("belief"):
            b = main["belief"]
            lines += ["", f"Belief head: log-loss {b['logloss']} vs uniform-over-consistent-worlds {b['uniform_logloss']} (n={b['n']:,}).", "",
                      "| Belief bin | n | predicted | observed |", "|---|---|---|---|"]
            for row in b.get("reliability", []):
                lines.append(f"| {row['bin']} | {row['n']} | {row['predicted']} | {row['observed']} |")
            lines.append("")
        o = main["oracle"]
        lines += [f"Oracle value head: log-loss {o['logloss']} (student {v['logloss']}).", ""]
    g1 = report.get("g1")
    if g1:
        h = g1["vs_heuristic"]
        r = g1["vs_random"]
        lines += ["## Gate G1", "",
                  f"BC agent (no search) vs HeuristicBot: **{h['a_score']:.3f}** +/- {h['standard_error']:.3f} over {h['games']:,} paired games "
                  f"-- **{h['verdict']}** (need >= 0.40). Vs RandomBot: {r['a_score']:.3f} over {r['games']:,}.", ""]
    abl = report.get("ablations")
    if abl:
        lines += ["## Screen 4 (ablations)", "", "| Variant | CHOOSE_ACTION | CHOOSE_CARDS (enum) | value log-loss | params |", "|---|---|---|---|---|"]
        if main:
            km = _key_metrics(main)
            lines.append(f"| main | {km['CHOOSE_ACTION_top1']} | {km['CHOOSE_CARDS_enumerate']} | {km['value_logloss']} | {main['training']['params']:,} |")
        for name, m in abl.items():
            lines.append(f"| {name} | {m['CHOOSE_ACTION_top1']} | {m['CHOOSE_CARDS_enumerate']} | {m['value_logloss']} | {m['params']:,} |")
        lines.append("")
    gen = report.get("generalization")
    if gen:
        lines += ["## Screen 5 (unseen cards)", "",
                  f"{gen['held_out_cards']} cards held out of training; {gen['eval_involving_held_out']:,} of {gen['eval_positions']:,} "
                  "evaluation decisions offer one.", "", "| Identity | CHOOSE_ACTION seen | CHOOSE_ACTION held-out | CHOOSE_CARDS seen | CHOOSE_CARDS held-out |",
                  "|---|---|---|---|---|"]
        for ident in ("both", "attr", "id"):
            if ident in gen:
                s, u = gen[ident]["seen"], gen[ident]["held_out"]
                lines.append(f"| {ident} | {s['CHOOSE_ACTION_top1']} | {u['CHOOSE_ACTION_top1']} | {s['CHOOSE_CARDS_enumerate']} | {u['CHOOSE_CARDS_enumerate']} |")
    with open(os.path.join(run.root, "screens.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
