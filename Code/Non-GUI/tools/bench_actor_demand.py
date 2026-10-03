#!/usr/bin/env python3
"""Actor-shaped throughput benchmark (Agent Observation Plan, O10): how fast
the self-play actors would ask the GPU for evaluations, measured without any
self-play.

    python -m tools.bench_actor_demand --checkpoint <tier0.kfc> --config tier2_selfplay_within_turn.json

Shaped like `agent/selfplay.py`'s actors:
- the config's number of worker processes and concurrent games per worker;
- its search settings (playout-cap randomization, root noise, leaves in
  flight, resample, move temperature);
- every worker batching all of its games' leaves into one call per round, to
  one shared inference server (`ml/infer_server.py`) on the GPU.

The searcher plays HeuristicBot, alternating seats and deck assignments, so
no game is self-play. Nothing is learned and no shard is written.

Each worker warms up, then measures over a fixed window: the network
evaluations it asked for (the evaluator's cache misses, i.e. what actually
reaches the GPU), the calls carrying them and the time spent waiting on
them, searched decisions, simulations and finished games. The parent
samples GPU utilization with `nvidia-smi` over the same window and writes
the totals under `$KEYFORGE_DATA/bench/actor_demand/`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import subprocess
import sys
import time

_NON_GUI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _CountingClient:
    """Wraps an `InferenceClient`: counts what it is asked for and how long
    each answer takes."""

    def __init__(self, inner):
        self.inner = inner
        self.calls = 0
        self.requests = 0
        self.wait_seconds = 0.0

    def predict_many(self, requests):
        t0 = time.perf_counter()
        answers = self.inner.predict_many(requests)
        self.wait_seconds += time.perf_counter() - t0
        self.calls += 1
        self.requests += len(requests)
        return answers


def _seed(run_seed: int, worker: int, index: int) -> int:
    digest = hashlib.sha256(f"bench_actor_demand|{run_seed}|{worker}|{index}".encode()).digest()
    return int.from_bytes(digest[:8], "big") >> 1


def _connect(host: str, port: int, authkey: str, timeout: float = 180.0):
    """A client to the server, once it answers (it may still be loading)."""
    from bots.inference_client import RemoteInferenceClient

    client = RemoteInferenceClient((host, port), authkey.encode())
    deadline = time.time() + timeout
    while True:
        try:
            client.predict_many([])
            return client
        except (ConnectionError, OSError):
            if time.time() > deadline:
                raise
            time.sleep(1.0)


def run_worker(args) -> dict:
    """One actor-shaped worker: `games_per_worker` concurrent games of the
    config's search against HeuristicBot, for `warmup + measure` seconds.
    Returns the counts over the measured window only."""
    from agent.config import resolve
    from agent.search.core import Search, SearchSettings, legal_actions, run_searches
    from agent.search.full_game import FullGame
    from agent.search.leaf import make_evaluator
    from agent.search.policies import make_policy
    from agent.search.within_turn import WithinTurn
    from agent.selfplay import SelfPlaySettings, _Cap, _pick
    from bots.heuristic_bot import HeuristicBot
    from keyforge.config import GameConfig
    from keyforge.enums import Resample
    from keyforge.game import Game

    s = SelfPlaySettings.from_config(resolve(args.config))
    concurrency = args.games_per_worker or s.concurrency
    client = _CountingClient(_connect(args.host, args.port, args.authkey))

    class Slot:
        __slots__ = ("game", "searcher", "search", "bot", "rng", "decisions")

    def advance(sl: Slot) -> None:
        """Plays forced decisions and HeuristicBot's decisions until the
        searcher has a real choice, or the game ends."""
        g = sl.game
        while not g.is_over:
            d = g.pending_decision
            acts = legal_actions(d, s.enumerate_cap)
            if acts is not None and len(acts) == 1:
                g.submit(acts[0][1])
            elif d.player != sl.searcher:
                g.submit(sl.bot.decide(g.view_for(d.player), d))
            else:
                return

    def new_slot(index: int) -> Slot:
        seed = _seed(args.seed, args.worker, index)
        sl = Slot()
        decks = s.decks if index % 2 == 0 else (s.decks[1], s.decks[0])
        sl.game = Game(GameConfig(decks=decks, seed=seed, max_turns=s.max_turns))
        sl.searcher = 1 if (index // 2) % 2 == 0 else 2  # every seat x deck combination
        policy = make_policy(s.rollout, seed=seed, client=client)
        regime = WithinTurn(policy) if s.regime == "within_turn" else FullGame(policy, quiet_leaves=s.quiet_leaves)
        settings = SearchSettings(
            simulations=s.sims_full, c_puct=s.c_puct, dirichlet_alpha=s.dirichlet_alpha, dirichlet_eps=s.dirichlet_eps,
            root_noise=True, leaves_in_flight=s.leaves_in_flight, resample=Resample(s.resample),
            enumerate_cap=s.enumerate_cap, reuse=False,
        )
        sl.search = Search(regime, make_evaluator(s.leaf, seed=seed), policy, settings, seed=seed)
        sl.bot = HeuristicBot(seed=seed ^ 0x5EED)
        sl.rng = random.Random(seed ^ 0xA5)
        sl.decisions = 0
        advance(sl)
        return sl

    t_start = time.perf_counter()
    t_measure = t_start + args.warmup
    t_end = t_measure + args.measure
    next_index = 0
    slots = []
    for _ in range(concurrency):
        slots.append(new_slot(next_index))
        next_index += 1
    games = searches = simulations = rounds = 0
    base = None
    while True:
        now = time.perf_counter()
        if base is None and now >= t_measure:
            base = (now, client.requests, client.calls, client.wait_seconds, searches, simulations, games, rounds)
        if now >= t_end:
            break
        for i in range(len(slots)):
            while slots[i].game.is_over:
                games += 1
                slots[i] = new_slot(next_index)
                next_index += 1
        gens = []
        for sl in slots:
            full = sl.rng.random() < s.full_fraction
            sl.search.settings.simulations = s.sims_full if full else s.sims_small
            gens.append(sl.search.search_gen(_Cap(sl.game, sl.searcher), sl.game.pending_decision))
        results = run_searches(gens, client)
        rounds += 1
        for sl, res in zip(slots, results):
            tau = s.tau_high if sl.decisions < s.temperature_moves else s.tau_low
            k = _pick(res.visits, tau, sl.rng, res.q)
            sl.decisions += 1
            searches += 1
            simulations += res.stats.get("simulations", 0)
            sl.game.submit(res.choices[k])
            advance(sl)
    t0, r0, c0, w0, se0, si0, g0, ro0 = base if base is not None else (t_start, 0, 0, 0.0, 0, 0, 0, 0)
    return {
        "worker": args.worker,
        "seconds": time.perf_counter() - t0,
        "requests": client.requests - r0,
        "calls": client.calls - c0,
        "wait_seconds": client.wait_seconds - w0,
        "searches": searches - se0,
        "simulations": simulations - si0,
        "games": games - g0,
        "rounds": rounds - ro0,
        "concurrency": concurrency,
    }


def _start_gpu_sampler(path: str):
    """`nvidia-smi` sampling GPU utilization and memory every 500 ms into
    `path`; None where there is no `nvidia-smi`."""
    try:
        out = open(path, "w")
        return subprocess.Popen(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits", "-lms", "500"],
            stdout=out, stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return None


def _gpu_summary(path: str) -> dict:
    util, mem = [], []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                    util.append(int(parts[0]))
                    mem.append(int(parts[1]))
    if not util:
        return {"samples": 0}
    ordered = sorted(util)
    return {
        "samples": len(util),
        "utilization_mean": round(statistics.mean(util), 1),
        "utilization_median": statistics.median(util),
        "utilization_p90": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))],
        "memory_used_mib_max": max(mem),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="the network the server loads (e.g. the Tier 0 BC checkpoint)")
    parser.add_argument("--config", default="tier2_selfplay_within_turn.json", help="the self-play config whose actors to imitate")
    parser.add_argument("--workers", type=int, default=None, help="default: the config's selfplay.workers")
    parser.add_argument("--games-per-worker", type=int, default=None, help="default: the config's selfplay.games_per_worker")
    parser.add_argument("--warmup", type=float, default=60.0, help="seconds before measuring")
    parser.add_argument("--measure", type=float, default=480.0, help="seconds measured")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6170)
    parser.add_argument("--authkey", default="keyforge")
    parser.add_argument("--out", default=None, help="result file; default $KEYFORGE_DATA/bench/actor_demand/<config>-<time>.json")
    parser.add_argument("--worker", type=int, default=None, help=argparse.SUPPRESS)  # internal: run as one worker
    args = parser.parse_args()

    if args.worker is not None:
        print(json.dumps(run_worker(args)), flush=True)
        return

    from agent.config import resolve
    from keyforge.version import ENGINE_VERSION
    from sim.data_root import resolve as data_path

    cfg = resolve(args.config)
    workers = args.workers or int(cfg["selfplay"]["workers"])
    per_worker = args.games_per_worker or int(cfg["selfplay"]["games_per_worker"])
    out = args.out or data_path(os.path.join("bench", "actor_demand", f"{cfg['name']}-{time.strftime('%Y%m%d-%H%M%S')}.json"))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    stem = out[:-5] if out.endswith(".json") else out

    with open(stem + ".server.log", "a") as log:
        server = subprocess.Popen(
            [sys.executable, "-m", "ml.infer_server", args.checkpoint, "--host", args.host, "--port", str(args.port),
             "--authkey", args.authkey],
            cwd=_NON_GUI, stdout=log, stderr=subprocess.STDOUT,
        )
    procs, sampler, results = [], None, []
    try:
        _connect(args.host, args.port, args.authkey)
        cmd = [
            sys.executable, "-m", "tools.bench_actor_demand", "--checkpoint", args.checkpoint, "--config", args.config,
            "--games-per-worker", str(per_worker), "--warmup", str(args.warmup), "--measure", str(args.measure),
            "--seed", str(args.seed), "--host", args.host, "--port", str(args.port), "--authkey", args.authkey,
        ]
        for w in range(workers):
            with open(f"{stem}.worker{w}.log", "a") as err:
                procs.append(subprocess.Popen(cmd + ["--worker", str(w)], cwd=_NON_GUI, stdout=subprocess.PIPE, stderr=err, text=True))
        time.sleep(args.warmup)
        sampler = _start_gpu_sampler(stem + ".gpu.csv")
        for w, p in enumerate(procs):
            stdout, _ = p.communicate()
            if p.returncode != 0:
                raise RuntimeError(f"worker {w} exited with {p.returncode}; see {stem}.worker{w}.log")
            results.append(json.loads(stdout.strip().splitlines()[-1]))
    finally:
        if sampler is not None:
            sampler.terminate()
        for p in procs:
            if p.poll() is None:
                p.kill()
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()

    def total(key):
        return sum(r[key] for r in results)

    seconds = statistics.mean(r["seconds"] for r in results)
    requests, searches, games = total("requests"), total("searches"), total("games")
    searches_per_game = searches / games if games else None
    summary = {
        "config": cfg["name"],
        "regime": cfg["search"]["regime"],
        "checkpoint": args.checkpoint,
        "engine_version": ENGINE_VERSION,
        "workers": workers,
        "games_per_worker": per_worker,
        "warmup_seconds": args.warmup,
        "measured_seconds": round(seconds, 1),
        "evaluations_per_second": round(requests / seconds, 1),
        "evaluations_per_second_per_worker": round(requests / seconds / workers, 1),
        "requests_per_call": round(requests / total("calls"), 1) if total("calls") else None,
        "inference_wait_fraction": round(total("wait_seconds") / total("seconds"), 3),
        "searches_per_second": round(searches / seconds, 2),
        "simulations_per_second": round(total("simulations") / seconds, 1),
        "evaluations_per_simulation": round(requests / total("simulations"), 3) if total("simulations") else None,
        "games_per_hour_vs_heuristic": round(games / seconds * 3600, 1),
        "searches_per_game_one_seat": round(searches_per_game, 1) if searches_per_game else None,
        # Self-play searches both seats' decisions; at the same search rate
        # it finishes roughly half as many games.
        "selfplay_games_per_hour_estimate": round(searches / seconds * 3600 / (2 * searches_per_game), 1) if searches_per_game else None,
        "gpu": _gpu_summary(stem + ".gpu.csv"),
        "per_worker": results,
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_worker"}, indent=1))
    print(f"written: {out}")


if __name__ == "__main__":
    main()
