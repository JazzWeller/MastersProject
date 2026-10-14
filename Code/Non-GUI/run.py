#!/usr/bin/env python3
"""One entry point for everything that runs on the cluster (or anywhere).

    python run.py                 # list the methods and what each one asks for
    python run.py <method>        # run one

On a Slurm machine (`sbatch` on PATH) a method submits itself as a batch job
with that method's CPU / memory / GPU / time request; anywhere else it runs
right here with the same worker counts scaled to the machine. Inside a job
the same file runs the steps, forwards SIGTERM to the running step, and on a
pause (exit 75, see ml/resume.py) requeues the job, so a time limit costs a
few minutes, not the run. Rerunning a method resumes it: every training and
evaluation step is resumable by run id.

Site settings (partition, account, modules, venv, data root, GPU type) live
in `cluster.json` next to this file (copy `cluster.example.json`), or in the
environment as KF_PARTITION, KF_CPU_PARTITION, KF_ACCOUNT, KF_QOS, KF_GRES, KF_MODULES, KF_VENV,
KF_DATA, KF_MAX_TIME. Nothing else needs editing.

Keep METHODS current: a new tool or stage gets a row here, with its resources.
Standard library only.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
EXIT_PAUSED = 75  # ml/resume.py: state saved, same command resumes
MAX_REQUEUES = 50
ARCHS = ("none", "summary", "turn_tokens", "joint", "stream")  # network.history_arch (configs/tier0b_bc.json)

# --------------------------------------------------------------------- methods
# cpus / mem_gb / gpu / hours are the Slurm request. "{w}" in a step expands to
# the worker count: the allocated CPUs less `reserve` (the learner, the server
# and the main process), capped at `max_w`, and at one worker per `gb_per_w` GB.
# Steps run in order from this directory and stop at the first failure.
PY = "python"
DATA = "bc/tier0b"  # under $KEYFORGE_DATA


def _bc(arch):
    return dict(
        desc=f"Tier 0b behaviour cloning, history_arch={arch} (run hpc-{arch}); needs `data`",
        cpus=8, mem_gb=64, gpu=True, hours=48,
        steps=[[PY, "-m", "ml.bc_train_v2", "--config", "tier0b_bc.json", "--run", f"hpc-{arch}",
                "--data", f"{DATA}/packed_v2", "--override", json.dumps({"network": {"history_arch": arch}})]])


def _eval(arch):
    return dict(
        desc=f"search strength of hpc-{arch} vs HeuristicBot (100 sims, 100 paired seeds, within-turn)",
        cpus=32, mem_gb=64, gpu=True, hours=12, max_w=16,
        steps=[[PY, "-m", "tools.eval_v2", "--a", f"v2:hpc-{arch}", "--b", "heuristic", "--sims", "100",
                "--seeds", "100", "--workers", "{w}"]])


METHODS = {
    "setup": dict(
        desc="login node: create the venv, install numpy + torch, compile the engine (needs internet)",
        local=True, steps=[]),  # handled in setup()
    "check": dict(
        desc="15 min: python, torch, the GPU, bf16, compiled engine",
        cpus=4, mem_gb=16, gpu=True, hours=0.25, steps=[[PY, "run.py", "--check"]]),
    "test": dict(
        desc="the unit and interaction suites",
        cpus=8, mem_gb=32, gpu=True, hours=3, steps=[[PY, "-m", "unittest", "discover", "-s", "tests"]]),
    "data": dict(
        desc="Tier 0b corpus: games -> v2 shards -> packed (~150 GB scratch; resumable)",
        cpus=32, mem_gb=128, gpu=False, hours=12, max_w=32, gb_per_w=4,
        steps=[[PY, "-m", "sim.bc_corpus", "--config", "tier0b_bc.json", "--out", f"{DATA}/records", "--workers", "{w}"],
               [PY, "-m", "ml.dataset_v2", f"{DATA}/records", f"{DATA}/encoded_v2", "--workers", "{w}"],
               [PY, "-m", "ml.dataset_v2", f"{DATA}/encoded_v2", f"{DATA}/packed_v2", "--pack", "--workers", "{w}"]]),
    "tier0": dict(
        desc="Tier 0 (v1): data, screens 1-3, gate G1 (run `tier0`)",
        cpus=16, mem_gb=64, gpu=True, hours=24, max_w=16, gb_per_w=4,
        steps=[[PY, "-m", "tools.run_screens", "--run", "tier0", "--config", "tier0_bc.json",
                "--stages", "data,main,g1", "--workers", "{w}"]]),
    "screen5": dict(
        desc="Screen 5: generalization to held-out cards, v1 vs v2 (run screen5-v2)",
        cpus=32, mem_gb=128, gpu=True, hours=48, max_w=16, gb_per_w=4,
        steps=[[PY, "-m", "tools.screen5_v2", "--run", "screen5-v2", "--workers", "{w}"]]),
    "probe": dict(
        desc="linear probes and targeted-position probes on the trained hpc-* networks",
        cpus=8, mem_gb=64, gpu=True, hours=10,
        steps=[[PY, "-m", "tools.probe_v2", "--runs", ",".join(f"hpc-{a}" for a in ARCHS)],
               [PY, "-m", "tools.probe_targeted", "--data", f"{DATA}/packed_v2",
                "--runs", ",".join(f"hpc-{a}" for a in ARCHS)]]),
    "search-eval": dict(
        desc="gate G2: within-turn search + heuristic evaluator vs HeuristicBot",
        cpus=32, mem_gb=48, gpu=False, hours=12, max_w=32, gb_per_w=2,
        steps=[[PY, "-m", "tools.eval_search", "--agent", "search-within-turn", "--sims", "200", "--seeds", "100",
                "--gate", "0.55", "--workers", "{w}"]]),
    "diag": dict(
        desc="G3 diagnostics: hidden-information gap and the search-vs-simulations curve (heuristic evaluator)",
        cpus=32, mem_gb=48, gpu=False, hours=24, max_w=32, gb_per_w=2,
        steps=[[PY, "-m", "tools.diag_hidden_info", "--regime", "within_turn", "--sims", "100", "--workers", "{w}"],
               [PY, "-m", "tools.diag_search_curve", "--regime", "full_game", "--sims", "1", "10", "100", "1000",
                "--workers", "{w}"]]),
    "bench-engine": dict(
        desc="engine throughput (decisions/s, fork costs)",
        cpus=4, mem_gb=8, gpu=False, hours=1, steps=[[PY, "-m", "tools.bench_engine"]]),
    "bench-actors": dict(
        desc="actor-shaped search throughput on a trained net (not self-play); needs bc-none",
        cpus=16, mem_gb=48, gpu=True, hours=1, max_w=16,
        steps=[[PY, "-m", "tools.bench_actor_demand", "--v2-run", "hpc-none", "--workers", "{w}"]]),
    "status": dict(desc="login node: every run's derived numbers (tools.monitor)", local=True, steps=[[PY, "-m", "tools.monitor"]]),
    "pause": dict(desc="login node: ask this user's running jobs to save state and stop (rerun a method to resume)",
                  local=True, steps=[]),
    "ladder": dict(desc="login node: submit data, then bc-<arch> for every arch in parallel, then probe, with dependencies",
                   local=True, steps=[]),
}
for _a in ARCHS:
    METHODS[f"bc-{_a}"] = _bc(_a)
    METHODS[f"eval-{_a}"] = _eval(_a)
# Self-play: the smoke run only, so far (the observation plan's gate is open for it; Tier 2/3 come after it).
# "{port}" is a per-job port for the learner's inference server, so two jobs on one node never collide.
METHODS["smoke-selfplay"] = dict(
    desc="the training plan's self-play smoke run: server + actors + learner + gating, minutes (run smoke-sp)",
    cpus=8, mem_gb=32, gpu=True, hours=2,
    steps=[[PY, "-m", "ml.selfplay_train", "--run", "smoke-sp", "--config", "smoke_selfplay.json", "--port", "{port}"],
           [PY, "-m", "tools.monitor", "smoke-sp"]])


# ---------------------------------------------------------------------- config
def settings() -> dict:
    cfg = {}
    path = os.path.join(HERE, "cluster.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    for key, env in (("partition", "KF_PARTITION"), ("cpu_partition", "KF_CPU_PARTITION"), ("account", "KF_ACCOUNT"), ("qos", "KF_QOS"), ("gres", "KF_GRES"),
                     ("modules", "KF_MODULES"), ("venv", "KF_VENV"), ("data", "KF_DATA"), ("max_time", "KF_MAX_TIME")):
        if os.environ.get(env):
            cfg[key] = os.environ[env]
    cfg.setdefault("gres", "gpu:1")
    cfg.setdefault("venv", os.path.join(HERE, ".venv-kf"))
    cfg.setdefault("data", os.environ.get("KEYFORGE_DATA")
                   or os.path.join(os.environ.get("SCRATCH", os.path.expanduser("~")), "keyforge-data"))
    cfg.setdefault("logs", os.path.join(cfg["data"], "slurm"))
    return cfg


def venv_python(cfg) -> str:
    for rel in ("bin/python", "Scripts/python.exe"):
        p = os.path.join(cfg["venv"], *rel.split("/"))
        if os.path.exists(p):
            return p
    return sys.executable


def hhmmss(hours: float, cap: str = None) -> str:
    s = int(hours * 3600)
    if cap:
        parts = [int(x) for x in cap.replace("-", ":").split(":")]
        days, rest = (parts[0], parts[1:]) if "-" in cap else (0, parts)
        rest = (rest + [0, 0, 0])[:3] if "-" in cap else ([0] * (3 - len(rest)) + rest)
        s = min(s, days * 86400 + rest[0] * 3600 + rest[1] * 60 + rest[2])
    return f"{s // 86400}-{s % 86400 // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


# --------------------------------------------------------------------- running
def workers(m: dict, cpus: int, mem_gb: float) -> int:
    w = cpus - m.get("reserve", 2)
    w = min(w, m.get("max_w", w), int(mem_gb // m.get("gb_per_w", 1)))
    return max(1, w)


def allocation(m: dict):
    """(cpus, mem_gb) this process may use: the Slurm allocation inside a job, else the machine."""
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK") or 0)
    if not cpus:
        cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    mem = m.get("mem_gb", 8)
    if not os.environ.get("SLURM_JOB_ID"):
        try:
            mem = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30
        except (ValueError, OSError, AttributeError):
            pass
    return cpus, mem


def run_steps(name: str, m: dict, cfg: dict) -> int:
    env = dict(os.environ, KEYFORGE_DATA=cfg["data"], PYTHONUNBUFFERED="1")
    os.makedirs(cfg["data"], exist_ok=True)
    cpus, mem = allocation(m)
    w = workers(m, cpus, mem)
    child = {"p": None}

    def forward(signum, _frame):
        env["KF_TERMINATED"] = "1"
        child["terminated"] = True
        if child["p"] and child["p"].poll() is None:
            child["p"].send_signal(signal.SIGTERM)

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    py = venv_python(cfg)
    print(f"[run.py] {name}: {cpus} cpus, {mem:.0f} GB, workers={w}, data={cfg['data']}", flush=True)
    for step in m["steps"]:
        port = 20000 + int(os.environ.get("SLURM_JOB_ID") or os.getpid()) % 20000
        argv = [py if a == PY else a.replace("{w}", str(w)).replace("{port}", str(port)) for a in step]
        if argv[1:3] == ["run.py", "--check"] and argv[0] == sys.executable:
            return check()  # already the venv's python (or no venv)
        print(f"[run.py] $ {shlex.join(argv)}", flush=True)
        child["p"] = subprocess.Popen(argv, cwd=HERE, env=env)
        while True:
            try:
                code = child["p"].wait()
                break
            except KeyboardInterrupt:
                pass
        if code != 0:
            return code
    return 0


def check() -> int:
    ok = True
    print("python", sys.version.split()[0])
    try:
        import numpy
        import torch

        print("numpy", numpy.__version__, "| torch", torch.__version__, "| cuda", torch.version.cuda)
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            print(f"gpu: {p.name}, {p.total_memory / 2 ** 30:.0f} GiB, capability {p.major}.{p.minor}, "
                  f"bf16 {torch.cuda.is_bf16_supported()}, devices {torch.cuda.device_count()}")
            x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
            print("bf16 matmul ok", float((x @ x).float().abs().mean()) > 0)
        else:
            print("NO CUDA: training and the GPU evaluations will not run")
            ok = False
    except Exception as e:  # noqa: BLE001
        print("torch/numpy problem:", e)
        ok = False
    r = subprocess.run([sys.executable, "-m", "tools.compile_engine", "--check"], cwd=HERE)
    print("compiled engine", "up to date" if r.returncode == 0 else "OUT OF DATE: run `python run.py setup`")
    return 0 if ok and r.returncode == 0 else 1


def setup(cfg: dict) -> int:
    venv = cfg["venv"]
    if not os.path.exists(os.path.join(venv, "bin", "python")):
        subprocess.check_call([sys.executable, "-m", "venv", venv])
    py = venv_python(cfg)
    pip = [py, "-m", "pip", "install", "--upgrade", "pip", "numpy"]
    subprocess.check_call(pip)
    torch_cmd = [py, "-m", "pip", "install", "torch"] + (["--index-url", cfg["torch_index"]] if cfg.get("torch_index") else [])
    subprocess.check_call(torch_cmd)
    subprocess.check_call([py, "-m", "tools.compile_engine"], cwd=HERE)
    return subprocess.call([py, "run.py", "--check"], cwd=HERE)


# ------------------------------------------------------------------- slurm side
def sbatch_script(name: str, m: dict, cfg: dict, argv_method: str) -> str:
    lines = ["#!/bin/bash", f"#SBATCH --job-name=kf-{name}", "#SBATCH --nodes=1", "#SBATCH --ntasks=1",
             f"#SBATCH --cpus-per-task={m['cpus']}", f"#SBATCH --mem={int(m['mem_gb'])}G",
             f"#SBATCH --time={hhmmss(m['hours'], cfg.get('max_time'))}",
             f"#SBATCH --output={cfg['logs']}/%x-%j.out", "#SBATCH --requeue", "#SBATCH --signal=B:TERM@300"]
    if m.get("gpu"):
        lines.append(f"#SBATCH --gres={cfg['gres']}")
    if cfg.get("partition") or cfg.get("cpu_partition"):  # CPU-only methods go to cpu_partition when one is set
        lines.append(f"#SBATCH --partition={cfg.get('partition') if m.get('gpu') else cfg.get('cpu_partition') or cfg.get('partition')}")
    for key in ("account", "qos"):
        if cfg.get(key):
            lines.append(f"#SBATCH --{key}={cfg[key]}")
    for mod in str(cfg.get("modules", "")).replace(",", " ").split():
        lines.append(f"module load {mod}")
    lines += [f"cd {shlex.quote(HERE)}", f"KF_IN_JOB=1 python3 run.py {argv_method}"]
    return "\n".join(lines) + "\n"


def submit(name: str, m: dict, cfg: dict, after: list = None) -> str:
    os.makedirs(cfg["logs"], exist_ok=True)
    cmd = ["sbatch", "--parsable"]
    if after:
        cmd.append("--dependency=afterok:" + ":".join(after))
    out = subprocess.run(cmd, input=sbatch_script(name, m, cfg, name), text=True, capture_output=True)
    if out.returncode != 0:
        sys.exit(f"sbatch failed: {out.stderr.strip()}")
    return out.stdout.strip().split(";")[0]


def requeue_if_paused(name: str, code: int, cfg: dict) -> bool:
    jid = os.environ.get("SLURM_JOB_ID")
    if code != EXIT_PAUSED or not jid:
        return False
    marker = os.path.join(cfg["logs"], f"pause-{jid}")
    if os.path.exists(marker):
        print("[run.py] paused on request; rerun the method to resume", flush=True)
        os.remove(marker)
        return True
    if int(os.environ.get("SLURM_RESTART_COUNT", "0")) >= MAX_REQUEUES:
        return False
    print("[run.py] state saved; requeueing", flush=True)
    return subprocess.call(["scontrol", "requeue", jid]) == 0


def pause(cfg: dict) -> int:
    out = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h", "-t", "R", "-o", "%i %j"],
                         capture_output=True, text=True).stdout.split("\n")
    os.makedirs(cfg["logs"], exist_ok=True)
    for line in filter(None, out):
        jid, jname = line.split(None, 1)
        if jname.startswith("kf-"):
            open(os.path.join(cfg["logs"], f"pause-{jid}"), "w").close()
            subprocess.call(["scancel", "--signal=TERM", "--batch", jid])
            print("pausing", jid, jname)
    return 0


def ladder(cfg: dict) -> int:
    data = submit("data", METHODS["data"], cfg)
    rungs = [submit(f"bc-{a}", METHODS[f"bc-{a}"], cfg, [data]) for a in ARCHS]
    probe = submit("probe", METHODS["probe"], cfg, rungs)
    print(f"submitted: data {data} -> {' '.join(rungs)} -> probe {probe}")
    return 0


# ------------------------------------------------------------------------ main
def listing() -> None:
    print(__doc__.split("Keep METHODS")[0].rstrip(), "\n")
    print(f"{'method':<14}{'cpus':>5}{'mem':>6}{'gpu':>5}{'hours':>7}  what")
    for k, m in METHODS.items():
        if m.get("local"):
            print(f"{k:<14}{'-':>5}{'-':>6}{'-':>5}{'-':>7}  {m['desc']}")
        else:
            print(f"{k:<14}{m['cpus']:>5}{m['mem_gb']:>5}G{'1' if m['gpu'] else '0':>5}{m['hours']:>7g}  {m['desc']}")


def main(argv) -> int:
    if len(argv) == 2 and argv[1] == "--check":
        return check()
    if len(argv) != 2 or argv[1] in ("-h", "--help", "list") or argv[1] not in METHODS:
        listing()
        return 0 if len(argv) < 2 or argv[1] in ("-h", "--help", "list") else 2
    name, m, cfg = argv[1], METHODS[argv[1]], settings()
    if name == "setup":
        return setup(cfg)
    if name == "pause":
        return pause(cfg)
    if name == "ladder":
        return ladder(cfg)
    in_job = bool(os.environ.get("KF_IN_JOB"))
    if shutil.which("sbatch") and not in_job and not m.get("local"):
        print(f"submitted batch job {submit(name, m, cfg)} ({name}); log: {cfg['logs']}/kf-{name}-<jobid>.out")
        return 0
    code = run_steps(name, m, cfg)
    if in_job and requeue_if_paused(name, code, cfg):
        return 0
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv))
