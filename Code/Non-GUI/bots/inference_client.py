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
The real network (`ml/infer_server.py`) lives in the torch-only `ml`
package and is only ever loaded in the server process.

`InferenceServer` does real cross-request dynamic batching: every request,
on any connection, lands in one shared queue; a single batching thread
drains it -- everything already queued, then whatever else arrives within
`batch_window_seconds`, up to `max_batch_size` -- into ONE call to the
model, then hands each connection back its own slice of the results. A
model that exposes `predict_many` (any real batched model, e.g. the one
`ml/infer_server.py` serves) gets one real batched call across
however many requests coalesced; a plain `observation -> (policy, value)`
callable (every test double in this codebase, and the simplest possible
real model) still works, just without a batching win, via a per-item loop
over the same coalesced list.

Connections are **persistent**: a `RemoteInferenceClient` connects and
authenticates once, then sends every request over that one connection, and
the server answers request after request on it. A fresh connection per call
made every call pay a TCP connect plus the authentication handshake, all
handshakes queued behind the server's single accepting thread (Agent
Observation Plan, O8).
"""

from __future__ import annotations

import os
import queue
import socket
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from multiprocessing.connection import AuthenticationError, Client, Listener
from typing import Any, Callable, List, Optional, Tuple

Prediction = Tuple[List[float], float]  # (policy, value)


def _no_delay(conn) -> None:
    """Turns Nagle's algorithm off on a TCP connection.

    `multiprocessing.connection` writes a large message's 4-byte length
    header and its payload separately. On a long-lived connection the
    receiver's delayed ACK then holds back the payload's last segment by
    ~40 ms per message: 52 ms against 11 ms per round trip, measured in WSL
    with a real request batch. A fresh connection per call never showed it,
    because TCP starts every connection in quick-ACK mode. A pipe or Unix
    socket address has no such option and is left as it is."""
    try:
        s = socket.socket(fileno=conn.fileno())
    except (OSError, ValueError):
        return
    try:
        if s.family in (socket.AF_INET, socket.AF_INET6):
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except OSError:
        pass
    finally:
        s.detach()  # the descriptor still belongs to `conn`


class InferenceClient(ABC):
    @abstractmethod
    def predict_many(self, observations: List[Any]) -> List[Prediction]:
        """One `(policy, value)` per observation, in the same order."""
        raise NotImplementedError

    def predict(self, observation: Any) -> Prediction:
        return self.predict_many([observation])[0]


class InProcessInferenceClient(InferenceClient):
    """Wraps a plain `observation -> (policy, value)` callable -- or, if the
    model exposes its own batched `predict_many` (any real network, e.g.
    `ml.infer_server.TorchModel`), calls that once per batch instead, the
    same way `InferenceServer` does."""

    def __init__(self, model: Callable[[Any], Prediction]):
        self._model = model
        self._batched = getattr(model, "predict_many", None)

    def predict_many(self, observations: List[Any]) -> List[Prediction]:
        if self._batched is not None:
            return list(self._batched(observations))
        return [self._model(o) for o in observations]


@dataclass
class _PendingRequest:
    observations: List[Any]
    event: threading.Event = field(default_factory=threading.Event)
    queued: float = field(default_factory=time.monotonic)
    result: Optional[Any] = None  # ("ok", [Prediction, ...]) or ("error", message) once `event` is set


class InferenceServer:
    """Owns a model (in a real deployment: GPU-resident) and serves
    `predict_many` requests from any number of `RemoteInferenceClient`s over
    one `multiprocessing.connection.Listener`, batching concurrent requests
    into real, single model calls (see this module's own docstring).

    Each accepted connection gets a thread that answers request after
    request until the client closes it. `connections_accepted` counts the
    connections ever accepted (a persistent client opens one).

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
        # Guards the hand-off between connection threads and shutdown: once
        # `_closed`, nothing more is queued, so no request can wait on a
        # batching thread that has already exited.
        self._state_lock = threading.Lock()
        self._closed = False
        self.connections_accepted = 0
        # Opt-in timing (`$KEYFORGE_SERVER_STATS`, a JSON-lines file: the
        # running totals appended every 10 s, so a window can be taken):
        # where the server's time goes, for the actor benchmarks.
        self._stats_path = os.environ.get("KEYFORGE_SERVER_STATS")
        self.stats = {"batches": 0, "calls": 0, "observations": 0, "predict_seconds": 0.0,
                      "queue_wait_seconds": 0.0, "window_seconds": 0.0, "since": time.time()}

    @property
    def address(self) -> Any:
        """The listener's actual bound address -- useful when `address` was
        given as `("localhost", 0)` and the OS picked the port."""
        return self._listener.address if self._listener is not None else self._address

    def _predict(self, observations: List[Any]) -> List[Prediction]:
        if self._model_predict_many is not None:
            return list(self._model_predict_many(observations))
        return [self._model(o) for o in observations]

    def _enqueue(self, request: _PendingRequest) -> bool:
        """Queues `request` for the batching thread; False once the server
        is shutting down (the batching thread may already be gone)."""
        with self._state_lock:
            if self._closed:
                return False
            self._queue.put(request)
            return True

    def _close_queue(self) -> None:
        """From now on nothing is queued; the batching thread answers what
        is already queued, then exits."""
        with self._state_lock:
            if not self._closed:
                self._closed = True
                self._queue.put(None)

    def _handle(self, conn) -> None:
        """Serves one connection: request after request until the client
        closes it, the connection breaks, or the server shuts down."""
        try:
            while True:
                try:
                    observations = conn.recv()
                except (EOFError, OSError):
                    return  # the client closed the connection, or it broke
                request = _PendingRequest(observations)
                if not self._enqueue(request):
                    return  # shutting down: closing tells the client to go elsewhere
                request.event.wait()
                # A model error is sent back as data (a tagged tuple), not
                # raised here and the connection dropped -- the latter would
                # leave the client's own `conn.recv()` seeing a bare EOFError,
                # with no way to tell "the model raised" from "the network
                # dropped" or from any other reason the connection might close.
                try:
                    conn.send(request.result)
                except OSError:
                    return  # the client went away while its answer was computed
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
            t_first = time.monotonic()
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
            t_predict = time.monotonic()
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
            if self._stats_path:
                st = self.stats
                st["batches"] += 1
                st["calls"] += len(batch)
                st["observations"] += len(all_observations)
                st["predict_seconds"] += time.monotonic() - t_predict
                st["window_seconds"] += t_predict - t_first
                st["queue_wait_seconds"] += sum(t_first - r.queued for r in batch if t_first > r.queued)
                if time.time() - st.get("written", 0) > 10:
                    st["written"] = time.time()
                    import json

                    with open(self._stats_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(st) + "\n")

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
                except (AuthenticationError, EOFError, OSError):
                    # A client that fails or drops during the handshake must
                    # not take the server down. stop() lands here too: it
                    # closes the listener, and wakes the accept with a bare
                    # connection.
                    if self._stop.is_set():
                        break
                    continue
                if self._stop.is_set():
                    conn.close()
                    break
                self.connections_accepted += 1
                _no_delay(conn)
                threading.Thread(target=self._handle, args=(conn,), daemon=True).start()
        finally:
            self._listener.close()
            # Everything queued before the sentinel is still answered; nothing
            # can be queued after it (see `_enqueue`).
            self._close_queue()
            batch_thread.join(timeout=5)

    def stop(self) -> None:
        """Stops a `serve_forever` running on another thread: requests stop
        being queued at once, and the accepting loop is woken. Closing the
        listener wakes a blocked `accept()` on Windows but not on Linux, so
        for a TCP address a bare connection is made to it first."""
        self._stop.set()
        self._close_queue()
        listener = self._listener
        if listener is None:
            return
        try:
            address = listener.address
        except AttributeError:  # already closed
            return
        if isinstance(address, tuple):
            try:
                socket.create_connection(address, timeout=1.0).close()
            except OSError:
                pass
        listener.close()


class RemoteInferenceClient(InferenceClient):
    """Talks to one `InferenceServer` over one persistent connection, opened
    on first use and reused for every call.

    - **Threads:** calls on one client are serialized by a lock -- a
      connection carries one request at a time.
    - **Processes:** a client that reaches another process (a fork, or
      pickling -- the connection itself is never pickled) opens its own
      connection there; two processes never share a socket.
    - **Failures:** the first connection attempt raises at once (so a caller
      waiting for the server to come up sees the refusal, as before). Once
      connected, a broken connection -- the server restarted -- is reopened
      and the request sent again, retrying for up to `reconnect_seconds`.
      A resend is always safe: inference has no side effects.
    """

    def __init__(self, address: Any, authkey: bytes, *, reconnect_seconds: float = 30.0):
        self._address = address
        self._authkey = authkey
        self._reconnect_seconds = reconnect_seconds
        self._conn = None
        self._pid: Optional[int] = None  # the process `_conn` belongs to
        self._lock = threading.Lock()

    def __getstate__(self):
        return {"address": self._address, "authkey": self._authkey, "reconnect_seconds": self._reconnect_seconds}

    def __setstate__(self, state):
        self.__init__(state["address"], state["authkey"], reconnect_seconds=state["reconnect_seconds"])

    def _connection(self):
        if self._conn is None or self._pid != os.getpid():
            self._drop()
            self._conn = Client(self._address, authkey=self._authkey)
            self._pid = os.getpid()
            _no_delay(self._conn)
        return self._conn

    def _drop(self) -> None:
        """Forgets the connection, closing this process's copy of it (a
        copy inherited through a fork closes only the child's descriptor)."""
        conn, self._conn = self._conn, None
        if conn is not None:
            try:
                conn.close()
            except OSError:
                pass

    def _round_trip(self, observations: List[Any]):
        connected = self._conn is not None and self._pid == os.getpid()
        try:
            conn = self._connection()
            conn.send(observations)
            return conn.recv()
        except (EOFError, OSError) as first:
            self._drop()
            if not connected:
                raise
            deadline = time.monotonic() + self._reconnect_seconds
            while True:
                try:
                    conn = self._connection()
                    conn.send(observations)
                    return conn.recv()
                except (EOFError, OSError):
                    self._drop()
                    if time.monotonic() >= deadline:
                        raise ConnectionError(
                            f"lost the inference server at {self._address!r} and could not reconnect "
                            f"within {self._reconnect_seconds:g} s"
                        ) from first
                    time.sleep(0.25)

    def predict_many(self, observations: List[Any]) -> List[Prediction]:
        with self._lock:
            status, payload = self._round_trip(observations)
        if status == "error":
            raise RuntimeError(f"InferenceServer's model raised: {payload}")
        return payload

    def close(self) -> None:
        """Closes the connection; the next call opens a new one."""
        with self._lock:
            self._drop()
