#!/usr/bin/env python3
"""Suspension-point coverage of the reference engine (Agent Observation Plan,
Part R, R0).

A suspension point is a `yield` or `yield from` inside a generator function
of `keyforge_ref/`, found by an AST scan. Stubs' unreachable yields (the
`return; yield` idiom that makes a never-pausing function a generator) are
listed separately and not counted.

Coverage is measured by playing the diff harness's fuzz corpus in lockstep
(`tools.diff_engines`; the bots read `keyforge`, the reference plays the same
choices) with `sys.monitoring` line events enabled only on the reference's
code objects, each line disabled after its first hit. A point counts as
reached once its line runs.

    python -m tools.ref_coverage --games 20000 --workers 10 --out <dir>

Writes `coverage.json` (every point, reached or not) and `coverage.md` (per
routine, and every point the fuzz missed).
"""

from __future__ import annotations

import argparse
import ast
import json
import multiprocessing as mp
import os
import sys
import time
from typing import Dict, List, Set, Tuple

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

REF_DIR = os.path.join(_HERE, "keyforge_ref")


class _Scanner(ast.NodeVisitor):
    """Collects, per generator function, its yield sites. A function's own
    yields only: a nested function is its own routine."""

    def __init__(self, relpath: str):
        self.relpath = relpath
        self.stack: List[str] = []
        self.kinds: List[str] = []
        self.routines: List[dict] = []

    def _visit_function(self, node):
        nested = "function" in self.kinds
        self.stack.append(node.name)
        self.kinds.append("function")
        qualname = ".".join(self.stack)
        points, dead = [], []
        for child, after_return in _own_yields(node):
            entry = {"line": child.lineno, "col": child.col_offset, "kind": "yield_from" if isinstance(child, ast.YieldFrom) else "yield"}
            (dead if after_return else points).append(entry)
        if points or dead:
            self.routines.append({"file": self.relpath, "routine": qualname, "line": node.lineno,
                                  "points": points, "stub_yields": dead, "nested": nested})
        for stmt in node.body:
            self.visit(stmt)
        self.stack.pop()
        self.kinds.pop()

    visit_FunctionDef = _visit_function
    visit_AsyncFunctionDef = _visit_function

    def visit_ClassDef(self, node):
        self.stack.append(node.name)
        self.kinds.append("class")
        self.generic_visit(node)
        self.stack.pop()
        self.kinds.pop()


def _own_yields(func):
    """`(yield node, is_dead)` for every yield in `func` itself (not in
    nested functions, lambdas or classes). Dead: an expression statement
    `yield` that directly follows a `return` in the same block."""
    out = []

    def walk_block(stmts):
        after_return = False
        for s in stmts:
            if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(s, ast.Expr) and isinstance(s.value, (ast.Yield, ast.YieldFrom)) and after_return:
                out.append((s.value, True))
                continue
            walk_stmt(s)
            if isinstance(s, ast.Return):
                after_return = True

    def walk_stmt(s):
        for field, value in ast.iter_fields(s):
            if isinstance(value, list) and value and isinstance(value[0], ast.stmt):
                walk_block(value)
            elif isinstance(value, list):
                for v in value:
                    if isinstance(v, ast.AST):
                        walk_expr_or_handler(v)
            elif isinstance(value, ast.AST):
                walk_expr_or_handler(value)

    def walk_expr_or_handler(node):
        if isinstance(node, (ast.Lambda, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return
        if isinstance(node, ast.ExceptHandler):
            walk_block(node.body)
            return
        if isinstance(node, (ast.Yield, ast.YieldFrom)):
            out.append((node, False))
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.stmt):
                walk_stmt(child)
            else:
                walk_expr_or_handler(child)

    walk_block(func.body)
    return out


# The machine itself (Part R) and its generated routines: not rules code.
NOT_RULES = ("vm.py", "compiled")


def scan(root: str = REF_DIR) -> List[dict]:
    routines = []
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__" and d not in NOT_RULES)
        for name in sorted(files):
            if not name.endswith(".py") or name in NOT_RULES:
                continue
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, root).replace(os.sep, "/")
            with open(path, encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=path)
            s = _Scanner(rel)
            s.visit(tree)
            routines.extend(s.routines)
    return routines


# ------------------------------------------------------------ measuring ----

_HITS: Set[Tuple[str, int]] = set()


_INSTALLED = []


def _install_monitor(root: str):
    if _INSTALLED:
        return
    _INSTALLED.append(root)
    mon = sys.monitoring
    tool = mon.PROFILER_ID
    mon.use_tool_id(tool, "ref_coverage")
    root = os.path.normcase(os.path.abspath(root))

    def on_start(code, offset):
        if os.path.normcase(code.co_filename).startswith(root):
            mon.set_local_events(tool, code, mon.events.LINE)
        return mon.DISABLE

    def on_line(code, line):
        _HITS.add((os.path.relpath(code.co_filename, root).replace(os.sep, "/"), line))
        return mon.DISABLE

    mon.register_callback(tool, mon.events.PY_START, on_start)
    mon.register_callback(tool, mon.events.LINE, on_line)
    mon.set_events(tool, mon.events.PY_START)


def _worker(args):
    items, base_seed = args
    from tools import diff_engines as de

    _install_monitor(REF_DIR)
    a, b = de.Engine("keyforge"), de.Engine("keyforge_ref")
    for i in items:
        de.play(a, b, de.fuzz_spec(i, base_seed), hash_every=10**9)
    return sorted(_HITS)


def measure(games: int, workers: int, base_seed: int = 0, chunk: int = 50) -> Set[Tuple[str, int]]:
    idx = list(range(games))
    tasks = [(idx[i : i + chunk], base_seed) for i in range(0, len(idx), chunk)]
    hits: Set[Tuple[str, int]] = set()
    if workers <= 1:
        for t in tasks:
            hits.update(map(tuple, _worker(t)))
    else:
        with mp.get_context("spawn").Pool(workers) as pool:
            for r in pool.imap_unordered(_worker, tasks):
                hits.update(map(tuple, r))
    return hits


def report(routines: List[dict], hits: Set[Tuple[str, int]], games: int, seconds: float) -> dict:
    n_points = n_hit = 0
    missed = []
    for r in routines:
        for p in r["points"]:
            p["reached"] = (r["file"], p["line"]) in hits
            n_points += 1
            n_hit += p["reached"]
            if not p["reached"]:
                missed.append({"file": r["file"], "routine": r["routine"], **p})
        r["reached"] = sum(p["reached"] for p in r["points"])
    pausing = [r for r in routines if r["points"]]
    return {
        "games": games, "seconds": round(seconds, 1),
        "generator_functions": len(routines), "pausing": len(pausing),
        "stubs": len([r for r in routines if not r["points"]]),
        "suspension_points": n_points, "reached": n_hit,
        "routines_fully_reached": sum(1 for r in pausing if r["reached"] == len(r["points"])),
        "routines_never_reached": sum(1 for r in pausing if r["reached"] == 0),
        "missed": missed, "routines": routines,
    }


def write_markdown(rep: dict, path: str) -> None:
    lines = [
        "# Reference-engine suspension-point coverage (Part R, R0)", "",
        f"{rep['games']:,} fuzz games ({rep['seconds']} s). {rep['generator_functions']} generator functions: "
        f"{rep['pausing']} pausing, {rep['stubs']} stubs.", "",
        f"**{rep['reached']} of {rep['suspension_points']} suspension points reached**; "
        f"{rep['routines_fully_reached']} routines fully reached, {rep['routines_never_reached']} never reached.", "",
        "## Missed points", "", "| File | Routine | Line | Kind |", "|---|---|---|---|",
    ]
    for m in rep["missed"]:
        lines.append(f"| {m['file']} | `{m['routine']}` | {m['line']} | {m['kind']} |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--games", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=".")
    args = parser.parse_args()
    routines = scan()
    t0 = time.perf_counter()
    hits = measure(args.games, args.workers, args.seed)
    rep = report(routines, hits, args.games, time.perf_counter() - t0)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "coverage.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1)
    write_markdown(rep, os.path.join(args.out, "coverage.md"))
    print(f"{rep['reached']}/{rep['suspension_points']} suspension points reached; "
          f"{rep['pausing']} pausing routines, {rep['stubs']} stubs; {len(rep['missed'])} missed")


if __name__ == "__main__":
    main()
