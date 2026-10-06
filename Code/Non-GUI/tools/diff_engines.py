#!/usr/bin/env python3
"""Drive two engines in lockstep and compare them after every decision
(Agent Observation Plan, Part R, R0).

An engine is a package name, optionally with an execution mode:
`keyforge_ref` (the engine frozen before Part R), `keyforge` (today's,
native), `keyforge@compiled` (Part R's explicit stack). The first engine
named is the one the bots read: each choice is made on it, then submitted to
both by its option index, so the bots never touch the second engine's
objects.

After every submit the two must agree on:
- the pending decision: player, kind, options (canonical), min_n/max_n,
  intent, affects, source card and optional;
- every new log event: kind, data, who may see it and its private fields;
- the journal's new entries, once both engines keep one (Milestone O1);
- the keyed-RNG counters and the choice record;
- in compiled execution, that nothing on the stack is a NativeFrame;
- and, every `--hash-every` decisions and at the end, the whole canonical
  state (`state_hash`'s input, minus the engine version string).

Corpora (`--corpus`):
- `golden`: the 22 recorded games of tests/golden_corpus.json, replayed.
- `fuzz`: `--games` games, cycling through bots (RandomBot, HeuristicBot,
  an epsilon-greedy HeuristicBot), deck sources (Fignor/Igor, every bundled
  preset, random legal decks, alliance decks) and formats (single Archon
  games; Reversal and Adaptive matches).

    python -m tools.diff_engines --a keyforge --b keyforge_ref --corpus golden
    python -m tools.diff_engines --a keyforge --b keyforge_ref --corpus fuzz --games 100000 --workers 10

Exits non-zero on the first divergence in any game, printing where.
"""

from __future__ import annotations

import argparse
import importlib
import json
import multiprocessing as mp
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

GOLDEN_PATH = os.path.join(_HERE, "tests", "golden_corpus.json")


class Divergence(AssertionError):
    pass


class Engine:
    """One engine package (and execution mode), with just the entry points
    the harness needs."""

    def __init__(self, spec: str):
        self.spec = spec
        package, _, mode = spec.partition("@")
        self.package = package
        self.mode = mode or None
        self.game_mod = importlib.import_module(f"{package}.game")
        self.match_mod = importlib.import_module(f"{package}.match")
        self.config_mod = importlib.import_module(f"{package}.config")
        self.decks_mod = importlib.import_module(f"{package}.cards.decks")
        self.hash_mod = importlib.import_module(f"{package}.state_hash")
        self.replay_mod = importlib.import_module(f"{package}.replay")

    def _deck(self, d):
        return d if isinstance(d, str) else self.decks_mod.deck_from_dict(d)

    def _kwargs(self) -> dict:
        return {"execution": self.mode} if self.mode else {}

    def new(self, spec: dict):
        decks = tuple(self._deck(d) for d in spec["decks"])
        if spec.get("format", "game") == "game":
            config = self.config_mod.GameConfig(
                decks=decks, seed=spec["seed"], max_turns=spec.get("max_turns"), first_player=spec.get("first_player"),
                starting_chains=spec.get("starting_chains"), setup_script=spec.get("setup_script"),
            )
            return self.game_mod.Game(config, **self._kwargs())
        config = self.match_mod.MatchConfig(format=spec["format"], decks=decks, seed=spec["seed"],
                                            max_turns=spec.get("max_turns"), first_player=spec.get("first_player"))
        return self.match_mod.Match(config, **self._kwargs())

    # ------------------------------------------------------- snapshots ----

    def decision(self, d) -> Any:
        if d is None:
            return None
        canon = self.hash_mod.canonicalize_decision(d)
        canon.update({
            "intent": d.intent.name if d.intent is not None else None,
            "affects": d.affects.value if d.affects is not None else None,
            "source": getattr(d.source_card, "instance_id", None) if d.source_card is not None else None,
            "optional": d.optional,
        })
        return canon

    def event(self, e) -> Any:
        return {
            "kind": e.kind,
            "data": self.hash_mod.canonicalize(e.data),
            "visible_to": sorted(e.visible_to),
            "private": {k: sorted(v) for k, v in sorted(getattr(e, "private", {}).items())},
        }

    def state(self, game) -> dict:
        state = self.hash_mod.canonical_game_state(game)
        state.pop("engine_version", None)
        return state


def approximate(state):
    """A canonical state in the frozen reference's approximations, for
    comparing an exact-hash engine with it (Part R, R1 made the hash exact):
    a callable is `"<callable>"`, an effect's handler isn't hashed, a
    condition is only whether there is one, and pending cleanups are
    counted."""

    def walk(x):
        if isinstance(x, dict):
            if "__fn__" in x or "__method__" in x or "__callable__" in x:
                return "<callable>"
            return {k: walk(v) for k, v in x.items()}
        if isinstance(x, list):
            return [walk(v) for v in x]
        return x

    out = walk(state)
    if not isinstance(out, dict):
        return out
    eff = out.get("active_effects")
    if isinstance(eff, dict):
        for kind in ("trigger", "instead", "modifier"):
            for e in eff.get(kind, []):
                e.pop("handler", None)
        for e in eff.get("duration", []):
            if not isinstance(e.get("conditional"), bool):
                e["conditional"] = e.get("conditional") is not None
    if isinstance(out.get("pending_cleanups"), list):
        out["pending_cleanups"] = len(out["pending_cleanups"])
    return out


def _game_of(obj):
    """The `Game` a handle is currently playing: itself, or a match's
    current game."""
    return getattr(obj, "current_game", obj) if not hasattr(obj, "players") else obj


def _vm_enabled() -> bool:
    from keyforge import vm
    return vm._ENABLED


class Lockstep:
    def __init__(self, a: Engine, b: Engine, spec: dict):
        self.a, self.b = a, b
        self.spec = spec
        self.ga = a.new(spec)
        self.gb = b.new(spec)
        self.n = 0
        self._log_seen: Dict[int, int] = {}
        self._journal_seen: Dict[int, int] = {}
        self.compare(full=True)

    def fail(self, what: str, va, vb):
        raise Divergence(f"{self.spec.get('name', '')} after {self.n} choices: {what} differs\n  {self.a.spec}: {va}\n  {self.b.spec}: {vb}")

    def _compare_game(self, full: bool):
        ga, gb = _game_of(self.ga), _game_of(self.gb)
        if (ga is None) != (gb is None):
            self.fail("current game", ga, gb)
        if ga is None:
            return
        key = len(getattr(self.ga, "finished_games", ()))
        # New log events.
        start = self._log_seen.get(key, 0)
        ea, eb = ga.log.events, gb.log.events
        if len(ea) != len(eb):
            self.fail("log length", len(ea), len(eb))
        for i in range(start, len(ea)):
            xa, xb = self.a.event(ea[i]), self.b.event(eb[i])
            if self.a.package != self.b.package:
                xa, xb = approximate(xa), approximate(xb)
            if xa != xb:
                self.fail(f"log event {i}", xa, xb)
        self._log_seen[key] = len(ea)
        ja, jb = getattr(ga, "journal", None), getattr(gb, "journal", None)
        if ja is not None and jb is not None:
            start = self._journal_seen.get(key, 0)
            if len(ja) != len(jb):
                self.fail("journal length", len(ja), len(jb))
            for i in range(start, len(ja)):
                if tuple(ja[i]) != tuple(jb[i]):
                    self.fail(f"journal entry {i}", ja[i], jb[i])
            self._journal_seen[key] = len(ja)
        for eng, g in ((self.a, ga), (self.b, gb)):
            m = getattr(g, "_machine", None)
            # (inside `vm.disabled()` every frame is native by design)
            if eng.mode == "compiled" and m is not None and not m.copyable and _vm_enabled():
                self.fail("a NativeFrame in compiled execution", [type(f).__name__ for f in m.stack], "")
        if ga._rng_counters != gb._rng_counters:
            self.fail("rng counters", ga._rng_counters, gb._rng_counters)
        if ga.choice_record != gb.choice_record:
            self.fail("game choice record", ga.choice_record[-3:], gb.choice_record[-3:])
        if full:
            sa, sb = self.a.state(ga), self.b.state(gb)
            if self.a.package != self.b.package:
                sa, sb = approximate(sa), approximate(sb)
            if sa != sb:
                for k in sa:
                    if sa.get(k) != sb.get(k):
                        self.fail(f"state[{k!r}]", sa.get(k), sb.get(k))
                self.fail("state keys", sorted(sa), sorted(sb))

    def compare(self, full: bool = False):
        if self.ga.is_over != self.gb.is_over:
            self.fail("is_over", self.ga.is_over, self.gb.is_over)
        da, db = self.a.decision(self.ga.pending_decision), self.b.decision(self.gb.pending_decision)
        if self.a.package != self.b.package:
            da, db = approximate(da), approximate(db)
        if da != db:
            self.fail("pending decision", da, db)
        if self.ga.choice_record != self.gb.choice_record:
            self.fail("choice record", self.ga.choice_record[-3:], self.gb.choice_record[-3:])
        if hasattr(self.ga, "score"):
            if self.ga.score != self.gb.score:
                self.fail("match score", self.ga.score, self.gb.score)
            ra = [(r.winner, r.turns, r.reason) for r in self.ga.games]
            rb = [(r.winner, r.turns, r.reason) for r in self.gb.games]
            if ra != rb:
                self.fail("match game records", ra, rb)
        if self.ga.is_over and getattr(self.ga, "result", None) != getattr(self.gb, "result", None):
            if not hasattr(self.ga, "score"):  # a match result holds a BidRecord of each engine's own class
                self.fail("result", self.ga.result, self.gb.result)
        self._compare_game(full)

    def step(self, choice, full: bool):
        self.ga.submit(choice)
        self.gb.submit_index(self.ga.choice_record[-1])
        self.n += 1
        self.compare(full)


# ------------------------------------------------------------- policies ----


def _bots(kind: str, seed: int):
    from bots.heuristic_bot import HeuristicBot
    from bots.random_bot import RandomBot
    from sim.bc_corpus import EpsilonGreedy

    if kind == "random":
        return {1: RandomBot(seed=seed), 2: RandomBot(seed=seed + 1)}
    if kind == "heuristic":
        return {1: HeuristicBot(seed=seed), 2: HeuristicBot(seed=seed + 1)}
    if kind == "epsilon":
        return {1: EpsilonGreedy(seed, 0.2), 2: EpsilonGreedy(seed + 1, 0.2)}
    raise ValueError(kind)


def play(a: Engine, b: Engine, spec: dict, *, hash_every: int = 25) -> int:
    """Plays one spec in lockstep; returns the number of decisions compared.
    Raises `Divergence` on the first difference."""
    ls = Lockstep(a, b, spec)
    if "choice_record" in spec:
        for encoded in spec["choice_record"]:
            ls.ga.submit_index(encoded)
            ls.gb.submit_index(encoded)
            ls.n += 1
            ls.compare(full=ls.n % hash_every == 0)
    else:
        bots = _bots(spec["bots"], spec["seed"])
        while not ls.ga.is_over:
            d = ls.ga.pending_decision
            choice = bots[d.player].decide(ls.ga.view_for(d.player), d)
            ls.step(choice, full=(ls.n + 1) % hash_every == 0)
    ls.compare(full=True)
    if "state_hash" in spec and hasattr(ls.ga, "state_hash"):
        if ls.ga.state_hash() != spec["state_hash"] and a.package == "keyforge" and not a.mode:
            raise Divergence(f"{spec.get('name')}: final state hash differs from the golden corpus")
    return ls.n


# --------------------------------------------------------------- corpora ----


def golden_specs() -> List[dict]:
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        entries = json.load(f)
    specs = []
    for e in entries:
        specs.append({
            "name": e["name"], "decks": e["decks"], "seed": e["seed"], "max_turns": e["max_turns"],
            "first_player": e["first_player"], "starting_chains": e["starting_chains"],
            "setup_script": e["setup_script"], "choice_record": e["choice_record"], "state_hash": e["state_hash"],
        })
    return specs


def fuzz_spec(i: int, base_seed: int = 0) -> dict:
    """The `i`-th fuzz game: deterministic in `i`, cycling bots, deck
    sources and formats so every combination recurs."""
    from keyforge.cards.decks import DECKS, build_alliance_deck, deck_to_dict, random_deck
    from keyforge.enums import House

    rng = random.Random(base_seed * 1_000_003 + i)
    bots = ("random", "heuristic", "epsilon")[i % 3]
    source = ("presets", "fignor_igor", "random", "alliance")[(i // 3) % 4]
    fmt = ("game", "game", "game", "reversal", "adaptive")[(i // 12) % 5]
    presets = sorted(DECKS)
    if source == "fignor_igor":
        decks = ["fignor", "igor"]
    elif source == "presets":
        decks = [rng.choice(presets), rng.choice(presets)]
    elif source == "random":
        decks = [deck_to_dict(random_deck(rng, "R1")), deck_to_dict(random_deck(rng, "R2"))]
    else:
        decks = []
        for side in ("A1", "A2"):
            houses = rng.sample(list(House), 3)
            sources = {h: rng.choice([p for p in presets if h in DECKS[p].pods] or presets) for h in houses}
            sources = {h: s for h, s in sources.items() if h in DECKS[s].pods}
            if len(sources) < 3:
                decks.append(deck_to_dict(random_deck(rng, side)))
            else:
                decks.append(deck_to_dict(build_alliance_deck(side, sources)))
    return {"name": f"fuzz-{base_seed}-{i}", "format": fmt, "decks": decks, "seed": rng.getrandbits(31),
            "max_turns": 120, "bots": bots}


# ------------------------------------------------------------------ main ----

_ENGINES: Dict[str, Engine] = {}


def _engine(spec: str) -> Engine:
    if spec not in _ENGINES:
        _ENGINES[spec] = Engine(spec)
    return _ENGINES[spec]


def _worker(args):
    a_spec, b_spec, kind, items, hash_every, base_seed = args
    a, b = _engine(a_spec), _engine(b_spec)
    decisions = 0
    for item in items:
        spec = fuzz_spec(item, base_seed) if kind == "fuzz" else item
        try:
            decisions += play(a, b, spec, hash_every=hash_every)
        except Divergence as exc:
            return {"ok": False, "error": str(exc), "spec": spec, "decisions": decisions}
        except Exception as exc:  # an engine crash is a divergence too, unless both crash the same way
            import traceback

            return {"ok": False, "error": f"{spec.get('name')}: {type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                    "spec": spec, "decisions": decisions}
    return {"ok": True, "decisions": decisions, "games": len(items)}


def run(a_spec: str, b_spec: str, corpus: str, *, games: int = 1000, workers: int = 1, hash_every: int = 25,
        base_seed: int = 0, start: int = 0, chunk: int = 50) -> dict:
    if corpus == "golden":
        items: List[Any] = golden_specs()
        chunks = [items[i : i + 4] for i in range(0, len(items), 4)]
    else:
        idx = list(range(start, start + games))
        chunks = [idx[i : i + chunk] for i in range(0, len(idx), chunk)]
    tasks = [(a_spec, b_spec, corpus, c, hash_every, base_seed) for c in chunks]
    t0 = time.perf_counter()
    total = {"games": 0, "decisions": 0, "failures": []}
    if workers <= 1:
        results = map(_worker, tasks)
    else:
        pool = mp.get_context("spawn").Pool(workers)
        results = pool.imap_unordered(_worker, tasks)
    for r in results:
        if r["ok"]:
            total["games"] += r["games"]
            total["decisions"] += r["decisions"]
        else:
            total["failures"].append(r)
            break
    if workers > 1:
        pool.terminate()
    total["seconds"] = round(time.perf_counter() - t0, 1)
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", default="keyforge")
    parser.add_argument("--b", default="keyforge_ref")
    parser.add_argument("--corpus", choices=("golden", "fuzz"), default="golden")
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--hash-every", type=int, default=25)
    args = parser.parse_args()
    out = run(args.a, args.b, args.corpus, games=args.games, workers=args.workers, hash_every=args.hash_every,
              base_seed=args.seed, start=args.start)
    if out["failures"]:
        f = out["failures"][0]
        print(f["error"])
        print(json.dumps(f["spec"])[:2000])
        sys.exit(1)
    print(f"{args.a} == {args.b} on {args.corpus}: {out['games']} games, {out['decisions']:,} decisions, {out['seconds']} s")


if __name__ == "__main__":
    main()
