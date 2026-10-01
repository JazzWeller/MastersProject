"""Agents, features and search for the Agent Training Plan
(Code/AGENT_TRAINING_PLAN.md).

**Never imports `torch`** -- not here, not in any submodule. Engine workers
import this package, and they must stay forkable (`os.fork()` does not
survive CUDA initialization; see keyforge/branching_fork.py) and runnable
under PyPy (which can't run PyTorch). Everything torch-side lives in the
sibling `ml` package and reaches these workers only through
`bots.inference_client.InferenceClient`. `tests/test_agent_training_m0.py`
enforces the rule on every module under `agent/`.
"""
