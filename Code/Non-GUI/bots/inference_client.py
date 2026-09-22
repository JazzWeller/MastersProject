"""InferenceClient abstraction (Agent Interface Plan, Milestone H): the same
client interface serves driver-level batching and in-search leaf batching,
so a search agent needs no separate code path when it moves from an
in-process model to a remote inference server.

Two implementations:
- `InProcessInferenceClient` wraps a plain callable -- CPU, small networks,
  no separate process.
- `RemoteInferenceClient`/`InferenceServer` talk over a stdlib
  `multiprocessing.connection` pipe, so many worker processes can share one
  model-owning server process without any engine code depending on how that
  model is implemented.

**The CUDA rule** (only the inference server process ever initializes CUDA,
since CUDA does not survive `fork`) is a deployment constraint on whatever
real model eventually plugs into `InferenceServer` -- there is no GPU or
`torch` in this environment to enforce it against. What this module *can*
and does guarantee in code: nothing in `keyforge`, `bots`, or `sim` imports
`torch`, so an engine worker pool built from these modules alone never
risks initializing CUDA by accident.

`InferenceServer.serve_forever` handles each accepted connection in its own
thread, with the model call itself under one lock -- correct (one client's
batch is never interleaved with another's) and simple, but it does not
implement cross-request dynamic batching (collecting several workers'
concurrent requests into a single larger model call). That's a real
throughput feature worth adding once there's an actual model whose batching
sweet spot can be measured -- premature here.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from multiprocessing.connection import AuthenticationError, Client, Listener
from typing import Any, Callable, List, Tuple

Prediction = Tuple[List[float], float]  # (policy, value)


class InferenceClient(ABC):
    @abstractmethod
    def predict_many(self, observations: List[Any]) -> List[Prediction]:
        """One `(policy, value)` per observation, in the same order."""
        raise NotImplementedError

    def predict(self, observation: Any) -> Prediction:
        return self.predict_many([observation])[0]


class InProcessInferenceClient(InferenceClient):
    """Wraps a plain `observation -> (policy, value)` callable."""

    def __init__(self, model: Callable[[Any], Prediction]):
        self._model = model

    def predict_many(self, observations: List[Any]) -> List[Prediction]:
        return [self._model(o) for o in observations]


class InferenceServer:
    """Owns a model (in a real deployment: GPU-resident) and serves
    `predict_many` requests from any number of `RemoteInferenceClient`s over
    one `multiprocessing.connection.Listener`.

    `address` is a `(host, port)` pair for a TCP listener, or a single
    string for a platform pipe/socket path -- whatever
    `multiprocessing.connection.Listener` itself accepts. `serve_forever`
    blocks the calling thread/process; run it in the dedicated inference
    server process the plan calls for, never inside an engine worker.
    """

    def __init__(self, address: Any, authkey: bytes, model: Callable[[Any], Prediction]):
        self._address = address
        self._authkey = authkey
        self._model = model
        self._lock = threading.Lock()
        self._listener: Listener | None = None
        self._stop = threading.Event()

    @property
    def address(self) -> Any:
        """The listener's actual bound address -- useful when `address` was
        given as `("localhost", 0)` and the OS picked the port."""
        return self._listener.address if self._listener is not None else self._address

    def _handle(self, conn) -> None:
        try:
            observations = conn.recv()
            with self._lock:
                results = [self._model(o) for o in observations]
            conn.send(results)
        finally:
            conn.close()

    def serve_forever(self) -> None:
        self._listener = Listener(self._address, authkey=self._authkey)
        try:
            while not self._stop.is_set():
                try:
                    conn = self._listener.accept()
                except AuthenticationError:
                    continue  # one bad client must not take the server down
                except OSError:
                    break  # listener closed from another thread (stop())
                threading.Thread(target=self._handle, args=(conn,), daemon=True).start()
        finally:
            self._listener.close()

    def stop(self) -> None:
        """Unblocks a `serve_forever` running on another thread."""
        self._stop.set()
        if self._listener is not None:
            self._listener.close()


class RemoteInferenceClient(InferenceClient):
    """Talks to one `InferenceServer` -- a fresh connection per call, so
    concurrent callers (e.g. several driver worker processes) never share a
    socket."""

    def __init__(self, address: Any, authkey: bytes):
        self._address = address
        self._authkey = authkey

    def predict_many(self, observations: List[Any]) -> List[Prediction]:
        conn = Client(self._address, authkey=self._authkey)
        try:
            conn.send(observations)
            return conn.recv()
        finally:
            conn.close()
