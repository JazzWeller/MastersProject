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
real model plugs into `InferenceServer` -- enforced in code the only way it
can be without a GPU in most environments: nothing in `keyforge`, `bots.
base`, `bots.registry`, or `sim` imports `torch`, so an engine worker pool
built from those modules alone never risks initializing CUDA by accident.
`bots/torch_model.py` is the one module that DOES import it, for a real
GPU-resident model (this machine has one -- an RTX 5060 Ti; see that
module's own tests, which actually run on it).

`InferenceServer` does real cross-request dynamic batching: every accepted
connection's request lands in one shared queue; a single batching thread
drains it -- everything already queued, then whatever else arrives within
`batch_window_seconds`, up to `max_batch_size` -- into ONE call to the
model, then hands each connection back its own slice of the results. A
model that exposes `predict_many` (any real batched model, e.g.
`bots.torch_model.TorchInferenceModel`) gets one real batched call across
however many requests coalesced; a plain `observation -> (policy, value)`
callable (every test double in this codebase, and the simplest possible
real model) still works, just without a batching win, via a per-item loop
over the same coalesced list.
"""

from __future__ import annotations

import queue
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from multiprocessing.connection import AuthenticationError, Client, Listener
from typing import Any, Callable, List, Optional, Tuple

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


@dataclass
class _PendingRequest:
    observations: List[Any]
    event: threading.Event = field(default_factory=threading.Event)
    result: Optional[Any] = None  # ("ok", [Prediction, ...]) or ("error", message) once `event` is set


class InferenceServer:
    """Owns a model (in a real deployment: GPU-resident) and serves
    `predict_many` requests from any number of `RemoteInferenceClient`s over
    one `multiprocessing.connection.Listener`, batching concurrent requests
    into real, single model calls (see this module's own docstring).

    `address` is a `(host, port)` pair for a TCP listener, or a single
    string for a platform pipe/socket path -- whatever
    `multiprocessing.connection.Listener` itself accepts. `serve_forever`
    blocks the calling thread/process; run it in the dedicated inference
    server process the plan calls for, never inside an engine worker.
    """

    def __init__(
        self, address: Any, authkey: bytes, model: Callable[[Any], Prediction],
        *, max_batch_size: int = 128, batch_window_seconds: float = 0.01,
    ):
        self._address = address
        self._authkey = authkey
        self._model = model
        self._model_predict_many = getattr(model, "predict_many", None)
        self._max_batch_size = max_batch_size
        self._batch_window_seconds = batch_window_seconds
        self._queue: "queue.Queue[Optional[_PendingRequest]]" = queue.Queue()
        self._listener: Listener | None = None
        self._stop = threading.Event()

    @property
    def address(self) -> Any:
        """The listener's actual bound address -- useful when `address` was
        given as `("localhost", 0)` and the OS picked the port."""
        return self._listener.address if self._listener is not None else self._address

    def _predict(self, observations: List[Any]) -> List[Prediction]:
        if self._model_predict_many is not None:
            return list(self._model_predict_many(observations))
        return [self._model(o) for o in observations]

    def _handle(self, conn) -> None:
        try:
            observations = conn.recv()
            request = _PendingRequest(observations)
            self._queue.put(request)
            request.event.wait()
            status, payload = request.result
            # A model error is sent back as data (a tagged tuple), not
            # raised here and the connection dropped -- the latter would
            # leave the client's own `conn.recv()` seeing a bare EOFError,
            # with no way to tell "the model raised" from "the network
            # dropped" or from any other reason the connection might close.
            conn.send((status, payload))
        finally:
            conn.close()

    def _batch_loop(self) -> None:
        """The one thread that ever calls the model: pulls the first
        request (blocking), then keeps coalescing whatever else is already
        queued or arrives within `batch_window_seconds`, up to
        `max_batch_size` observations total, into one `_predict` call."""
        while True:
            first = self._queue.get()
            if first is None:
                return
            batch = [first]
            total = len(first.observations)
            deadline = time.monotonic() + self._batch_window_seconds
            while total < self._max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    nxt = self._queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if nxt is None:
                    self._queue.put(None)  # let the next iteration see the stop signal
                    break
                batch.append(nxt)
                total += len(nxt.observations)

            all_observations = [o for req in batch for o in req.observations]
            try:
                all_results = self._predict(all_observations)
                i = 0
                for req in batch:
                    n = len(req.observations)
                    req.result = ("ok", all_results[i : i + n])
                    i += n
            except Exception as e:  # noqa: BLE001 -- every waiting connection must be released
                message = f"{type(e).__name__}: {e}"
                for req in batch:
                    req.result = ("error", message)
            for req in batch:
                req.event.set()

    def serve_forever(self) -> None:
        # `Listener`'s own default backlog is 1 -- fine for occasional
        # inter-process connections, much too small for a burst of self-play
        # workers all connecting within the same batch window: a queued-but-
        # not-yet-accepted connection can stall for hundreds of ms (measured
        # on Windows), which starves this very batch window of exactly the
        # concurrent requests it exists to coalesce.
        self._listener = Listener(self._address, authkey=self._authkey, backlog=128)
        batch_thread = threading.Thread(target=self._batch_loop, daemon=True)
        batch_thread.start()
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
            self._queue.put(None)
            batch_thread.join(timeout=5)

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
            status, payload = conn.recv()
            if status == "error":
                raise RuntimeError(f"InferenceServer's model raised: {payload}")
            return payload
        finally:
            conn.close()
