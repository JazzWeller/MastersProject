"""The torch side of the Agent Training Plan (Code/AGENT_TRAINING_PLAN.md):
networks, batching, training loops, checkpoints, the inference server.

CPython + torch only (WSL with CUDA for training; a CPU build on Windows
for GUI play). **Never imported by an engine worker** -- `agent/` and
everything under `keyforge`/`bots`/`sim` stay torch-free, which is what
keeps `os.fork()` (CUDA doesn't survive it) and PyPy available to them.
Workers reach a network only through `bots.inference_client`.
"""
