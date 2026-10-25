"""Real-Time Chunking (RTC) helpers for Kai0 absolute dual-arm joint eval.

Unlike ``shellgame_rtc`` (EEF-relative), Kai0 actions are absolute joints
``[T, 14]`` (left q6+grip, right q6+grip). Leftover prefixes need no frame
transform — just the unexecuted suffix, zero-padded to the full horizon.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from typing import Callable
import time

import numpy as np


def estimate_delay_steps(latency_s: float, frequency: float, *, floor: int = 1) -> int:
    return max(int(floor), int(np.ceil(float(latency_s) * float(frequency))))


@dataclass
class JointRTCChunk:
    """One absolute-joint action chunk from the policy."""

    actions: np.ndarray  # (H, action_dim)
    created_monotonic: float

    @property
    def horizon(self) -> int:
        return int(len(self.actions))


def leftover_absolute(
    chunk: JointRTCChunk,
    *,
    start_index: int,
    horizon: int,
) -> np.ndarray:
    """Unexecuted absolute-joint suffix, zero-padded to ``horizon``."""
    actions = np.asarray(chunk.actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[0] == 0:
        dim = int(actions.shape[1]) if actions.ndim == 2 and actions.shape[1] else 14
        return np.zeros((horizon, dim), dtype=np.float32)
    dim = int(actions.shape[1])
    leftover = actions[max(0, int(start_index)) :]
    out = np.zeros((horizon, dim), dtype=np.float32)
    n = min(len(leftover), horizon)
    if n > 0:
        out[:n] = leftover[:n]
    return out


def build_joint_chunk(actions: np.ndarray) -> JointRTCChunk:
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected actions [T, D], got {arr.shape}")
    return JointRTCChunk(actions=arr, created_monotonic=time.monotonic())


class JointRTCInferenceWorker:
    """Background absolute-joint RTC inference (paper Algorithm 1 style)."""

    def __init__(
        self,
        infer_fn: Callable[..., tuple[np.ndarray, float]],
        *,
        frequency: float,
        s_min: int,
        delay_buffer_size: int = 8,
        initial_delay_steps: int = 4,
        action_dim: int = 14,
        action_horizon: int = 50,
    ) -> None:
        self._infer_fn = infer_fn
        self.frequency = float(frequency)
        self.s_min = int(s_min)
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kai0-rtc")
        self._future: Future | None = None
        self._closed = False
        self._delay_buf: deque[int] = deque(
            [int(initial_delay_steps)], maxlen=int(delay_buffer_size)
        )
        self.last_latency_s: float = float(initial_delay_steps) / self.frequency

    def close(self, *, wait: bool = True, timeout_s: float = 30.0) -> bool:
        """Stop accepting jobs. Returns True if no in-flight infer remains."""
        self._closed = True
        future = self._future
        still_running = False
        if wait and future is not None and not future.done():
            try:
                future.result(timeout=timeout_s)
            except FuturesTimeoutError:
                still_running = True
                print(
                    f"[Kai0-RTC] timed out waiting {timeout_s:.1f}s for in-flight infer; "
                    "skipping blocking shutdown"
                )
            except Exception:
                pass
        elif future is not None and not future.done():
            still_running = True
        self._future = None
        self._executor.shutdown(wait=wait and not still_running, cancel_futures=True)
        return not still_running

    @property
    def delay_estimate(self) -> int:
        return int(max(self._delay_buf))

    @property
    def busy(self) -> bool:
        return self._future is not None and not self._future.done()

    def maybe_start(
        self,
        *,
        executed_since_chunk: int,
        current_chunk: JointRTCChunk,
        get_obs_bundle: Callable[[], dict],
    ) -> None:
        if self._closed or self.busy:
            return
        s = max(self.delay_estimate, self.s_min)
        if executed_since_chunk < s:
            return
        d = self.delay_estimate
        obs_bundle = get_obs_bundle()
        # Pad to full model action_horizon (not trimmed chunk length).
        leftover = leftover_absolute(
            current_chunk,
            start_index=executed_since_chunk,
            horizon=self.action_horizon,
        )
        started = time.monotonic()

        def _job():
            actions, latency = self._infer_fn(
                obs_bundle=obs_bundle,
                rtc_prev_chunk=leftover,
                rtc_inference_delay=d,
                rtc_execution_horizon=s,
            )
            return actions, latency, started, s, d

        self._future = self._executor.submit(_job)

    def poll(self) -> tuple[JointRTCChunk, int, int] | None:
        """Return (new_chunk, observed_delay_steps, execution_horizon) when ready."""
        if self._future is None or not self._future.done():
            return None
        actions, latency, _started, s, _d_est = self._future.result()
        self._future = None
        observed_delay = estimate_delay_steps(latency, self.frequency)
        self._delay_buf.append(observed_delay)
        self.last_latency_s = float(latency)
        chunk = build_joint_chunk(actions)
        n = chunk.horizon
        if n == 0:
            return chunk, observed_delay, s
        start_index = min(max(0, observed_delay), n - 1)
        trimmed = JointRTCChunk(
            actions=chunk.actions[start_index:],
            created_monotonic=chunk.created_monotonic,
        )
        return trimmed, observed_delay, s
