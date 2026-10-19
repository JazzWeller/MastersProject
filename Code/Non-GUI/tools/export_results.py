#!/usr/bin/env python3
"""Copies the small, reviewable part of finished runs into the repository,
so results made on the cluster come back through git.

    python -m tools.export_results tier2-v2-s0 tier2-v1-s0 [--out results/tier2]

For each run, `<out>/<run id>/` gets:
- `config.json`, `journal.jsonl`, `best.json`, and the gate verdicts (`gates/`);
- `monitor.txt` / `monitor.json`, the monitor's screen and its numbers;
- `learner_metrics.jsonl.gz`, the learner's metric records only (the
  actors' are most of `metrics.jsonl`, and the monitor already sums them);
- `checkpoints/`, the best promoted network and the final one.

Every evaluation result under `$KEYFORGE_DATA/evals/` naming one of these runs
goes to `<out>/evals/`. Shards, logs and other checkpoints stay where they
are. Run it again to refresh: files are overwritten.
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import json
import os
import shutil

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _copy(src: str, dst: str) -> bool:
    if not os.path.exists(src):
        return False
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)
    return True


def export_run(run_id: str, out: str) -> list:
    from agent.telemetry import Run
    from tools import monitor

    run = Run.open(run_id)
    dst = os.path.join(out, run_id)
    os.makedirs(dst, exist_ok=True)
    done = []
    for name in ("config.json", "journal.jsonl", "best.json"):
        if _copy(os.path.join(run.root, name), os.path.join(dst, name)):
            done.append(name)
    gates = os.path.join(run.root, "gates")
    if os.path.isdir(gates):
        for n in sorted(os.listdir(gates)):
            if n.endswith(".json"):
                _copy(os.path.join(gates, n), os.path.join(dst, "gates", n))
        done.append("gates/")
    summary = monitor.summarize(run, monitor.PLAN_GAMES_PER_HOUR)
    with open(os.path.join(dst, "monitor.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, default=str)
    with open(os.path.join(dst, "monitor.txt"), "w", encoding="utf-8") as f:
        f.write(monitor.render(summary) + "\n")
    done.append("monitor")
    if os.path.exists(run.metrics_path):
        with open(run.metrics_path, "r", encoding="utf-8") as f, \
                gzip.open(os.path.join(dst, "learner_metrics.jsonl.gz"), "wt", encoding="utf-8") as g:
            for line in f:
                with contextlib.suppress(ValueError):
                    if json.loads(line).get("source") == "learner":
                        g.write(line)
        done.append("learner_metrics")
    store = os.path.join(run.root, "checkpoints")
    wanted = {}
    with contextlib.suppress(OSError, ValueError, KeyError):
        with open(os.path.join(run.root, "best.json"), "r", encoding="utf-8") as f:
            wanted["best"] = json.load(f)["checkpoint"]
    finals = [e for e in run.journal_entries() if e["event"] == "run_finished"]
    if finals:
        wanted["final"] = finals[-1]["final"]
    for label, h in wanted.items():
        for n in os.listdir(store) if os.path.isdir(store) else ():
            if n.startswith(h):
                _copy(os.path.join(store, n), os.path.join(dst, "checkpoints", n))
                done.append(f"{label} checkpoint")
    return done


def main():
    from sim import data_root

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--out", default=os.path.join(HERE, "results", "tier2"))
    args = parser.parse_args()
    for rid in args.runs:
        print(rid, ":", ", ".join(export_run(rid, args.out)))
    evals = data_root.resolve("evals")
    n = 0
    if os.path.isdir(evals):
        for name in sorted(os.listdir(evals)):
            if name.endswith(".json") and any(r.replace(":", "-") in name for r in args.runs):
                _copy(os.path.join(evals, name), os.path.join(args.out, "evals", name))
                n += 1
    print(f"{n} evaluation results; written under {args.out}")


if __name__ == "__main__":
    main()
