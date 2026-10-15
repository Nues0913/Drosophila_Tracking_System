"""Scalar environments stepped concurrently in worker processes.

Each slot is an ordinary FlyTrackEnv owned by one worker process. The trainer
still decides every reset and action, so episode content depends only on seeds
and actions, never on worker count or timing. ``workers=0`` keeps all slots
in-process with identical semantics (tests, debugging, monkeypatched worlds).
"""

from __future__ import annotations

import multiprocessing
import os
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

_WORKERS: int | None = None
_THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def set_default_workers(workers: int | None) -> None:
    """CLI hook; library callers and tests default to in-process environments."""
    global _WORKERS
    if workers is not None and workers < 0:
        raise ValueError("workers must be nonnegative")
    _WORKERS = workers


def default_workers() -> int:
    return 0 if _WORKERS is None else _WORKERS


def _worker(connection, factories: dict[int, Callable[[], Any]]) -> None:
    envs = {slot: factory() for slot, factory in factories.items()}
    try:
        while True:
            command, payload = connection.recv()
            try:
                if command == "close":
                    break
                result = _apply(envs, command, payload)
            except BaseException as error:  # noqa: BLE001 - forwarded to the trainer
                connection.send(("error", repr(error)))
                continue
            connection.send(("ok", result))
    finally:
        for env in envs.values():
            env.close()
        connection.close()


def _apply(envs: dict[int, Any], command: str, payload):
    if command == "step":
        return [envs[slot].step(action) for slot, action in payload]
    if command == "reset":
        slot, seed = payload
        return envs[slot].reset(seed=seed)
    if command == "remake":
        slot, factory = payload
        envs[slot].close()
        envs[slot] = factory()
        return None
    if command == "replay":
        slot, actions = payload
        result = None
        for action in actions:
            result = envs[slot].step(np.asarray(action, dtype=np.float32))
        return result
    if command == "get":
        slot, name = payload
        return getattr(envs[slot], name)
    if command == "set":
        slot, name, value = payload
        setattr(envs[slot], name, value)
        return None
    raise ValueError(f"Unknown environment pool command {command}")


class EnvPool:
    def __init__(self, factories: Sequence[Callable[[], Any]], workers: int | None = None):
        workers = default_workers() if workers is None else workers
        self.size = len(factories)
        if self.size < 1:
            raise ValueError("EnvPool needs at least one environment")
        self.workers = min(workers, self.size)
        self._closed = False
        if self.workers == 0:
            self._envs = {slot: factory() for slot, factory in enumerate(factories)}
            return
        self._owner = [slot % self.workers for slot in range(self.size)]
        context = multiprocessing.get_context("spawn")
        self._connections, self._processes = [], []
        saved = {name: os.environ.get(name) for name in _THREAD_VARIABLES}
        try:
            # Spawned children inherit this environment: one BLAS thread each.
            for name in _THREAD_VARIABLES:
                os.environ[name] = "1"
            for worker in range(self.workers):
                parent, child = context.Pipe()
                owned = {
                    slot: factories[slot]
                    for slot in range(self.size)
                    if slot % self.workers == worker
                }
                process = context.Process(target=_worker, args=(child, owned), daemon=True)
                process.start()
                child.close()
                self._connections.append(parent)
                self._processes.append(process)
        finally:
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def _receive(self, worker: int):
        status, result = self._connections[worker].recv()
        if status != "ok":
            raise RuntimeError(f"Environment worker {worker} failed: {result}")
        return result

    def _call(self, slot: int, command: str, payload):
        if self.workers == 0:
            return _apply(self._envs, command, payload)
        worker = self._owner[slot]
        self._connections[worker].send((command, payload))
        return self._receive(worker)

    def reset(self, slot: int, seed: int):
        return self._call(slot, "reset", (slot, seed))

    def remake(self, slot: int, factory: Callable[[], Any]) -> None:
        self._call(slot, "remake", (slot, factory))

    def replay(self, slot: int, actions):
        return self._call(slot, "replay", (slot, actions))

    def get(self, slot: int, name: str):
        return self._call(slot, "get", (slot, name))

    def set(self, slot: int, name: str, value) -> None:
        self._call(slot, "set", (slot, name, value))

    def step_async(self, actions: dict[int, np.ndarray]) -> None:
        self._pending = list(actions)
        if self.workers == 0:
            self._results = _apply(self._envs, "step", list(actions.items()))
            return
        grouped: dict[int, list] = {}
        for slot, action in actions.items():
            grouped.setdefault(self._owner[slot], []).append((slot, action))
        self._grouped = grouped
        for worker, payload in grouped.items():
            self._connections[worker].send(("step", payload))

    def step_wait(self) -> dict[int, tuple]:
        if self.workers == 0:
            return dict(zip(self._pending, self._results, strict=True))
        results = {}
        for worker, payload in self._grouped.items():
            for (slot, _), result in zip(payload, self._receive(worker), strict=True):
                results[slot] = result
        return results

    def step(self, actions: dict[int, np.ndarray]) -> dict[int, tuple]:
        """Step the given slots concurrently; returns slot -> (obs, reward, term, trunc, info)."""
        self.step_async(actions)
        return self.step_wait()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.workers == 0:
            for env in self._envs.values():
                env.close()
            return
        for connection in self._connections:
            try:
                connection.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for process in self._processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
        for connection in self._connections:
            connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
