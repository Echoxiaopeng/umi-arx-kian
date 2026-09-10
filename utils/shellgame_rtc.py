"""Real-Time Chunking (RTC) control helpers for ARX5 ShellGame eval."""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from typing import Callable
import time

import numpy as np

from utils.shellgame_openpi import relative_chunk_to_world
from utils.shellgame_openpi import world_chunk_to_relative


@dataclass
class RTCChunk:
    """One decoded policy chunk plus the anchor used to produce it."""

    relative_actions: np.ndarray  # (H, 10) current-relative EEF10
    world_link6: np.ndarray  # (H, 7) world xyz+rotvec+gripper (link6)
    anchor_link6: np.ndarray  # (6,)
    created_monotonic: float


def estimate_delay_steps(latency_s: float, frequency: float, *, floor: int = 1) -> int:
    return max(int(floor), int(np.ceil(float(latency_s) * float(frequency))))


def leftover_relative_for_new_anchor(
    chunk: RTCChunk,
    *,
    start_index: int,
    new_anchor_link6: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """Re-express unexecuted world waypoints in the new observation's relative frame."""
    leftover_world = chunk.world_link6[start_index:]
    if len(leftover_world) == 0:
        return np.zeros((horizon, chunk.relative_actions.shape[1]), dtype=np.float32)
    relative = world_chunk_to_relative(leftover_world, new_anchor_link6)
    out = np.zeros((horizon, relative.shape[1]), dtype=np.float32)
    out[: len(relative)] = relative
    return out


def build_rtc_chunk(
    relative_actions: np.ndarray,
    anchor_link6: np.ndarray,
) -> RTCChunk:
    relative = np.asarray(relative_actions, dtype=np.float64)
    if relative.size == 0:
        relative = relative.reshape(0, 10)
        world = np.zeros((0, 7), dtype=np.float64)
    else:
        world = relative_chunk_to_world(relative, anchor_link6)
    return RTCChunk(
        relative_actions=relative.astype(np.float32),
        world_link6=world,
        anchor_link6=np.asarray(anchor_link6, dtype=np.float64).reshape(6),
        created_monotonic=time.monotonic(),
    )


class RTCInferenceWorker:
    """Background inference with delay tracking (paper Algorithm 1 style)."""

    def __init__(
        self,
        infer_fn: Callable[..., tuple[np.ndarray, np.ndarray, float]],
        *,
        frequency: float,
        s_min: int,
        delay_buffer_size: int = 8,
        initial_delay_steps: int = 4,
    ) -> None:
        self._infer_fn = infer_fn
        self.frequency = float(frequency)
        self.s_min = int(s_min)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rtc-infer")
        self._future: Future | None = None
        self._closed = False
        self._delay_buf: deque[int] = deque([int(initial_delay_steps)], maxlen=int(delay_buffer_size))
        self.last_latency_s: float = float(initial_delay_steps) / self.frequency

    def close(self, *, wait: bool = True, timeout_s: float = 30.0) -> bool:
        """Stop accepting jobs. Returns True if no in-flight infer remains (ws free)."""
        self._closed = True
        future = self._future
        still_running = False
        if wait and future is not None and not future.done():
            try:
                future.result(timeout=timeout_s)
            except FuturesTimeoutError:
                still_running = True
                print(
                    f"[RTC] timed out waiting {timeout_s:.1f}s for in-flight infer; "
                    "skipping blocking shutdown"
                )
            except Exception:
                # Shutdown path: ignore infer errors; caller must not race recv.
                pass
        elif future is not None and not future.done():
            still_running = True
        self._future = None
        # If infer is still inside ws.recv(), wait=True would hang forever.
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
        current_chunk: RTCChunk,
        get_obs_bundle: Callable[[], dict],
    ) -> None:
        if self._closed or self.busy:
            return
        s = max(self.delay_estimate, self.s_min)
        if executed_since_chunk < s:
            return
        d = self.delay_estimate
        obs_bundle = get_obs_bundle()
        leftover = leftover_relative_for_new_anchor(
            current_chunk,
            start_index=executed_since_chunk,
            new_anchor_link6=obs_bundle["anchor_link6"],
            horizon=len(current_chunk.relative_actions),
        )
        started = time.monotonic()

        def _job():
            relative, anchor_link6, latency = self._infer_fn(
                obs_bundle=obs_bundle,
                rtc_prev_chunk=leftover,
                rtc_inference_delay=d,
                rtc_execution_horizon=s,
            )
            return relative, anchor_link6, latency, started, s, d

        self._future = self._executor.submit(_job)

    def poll(self) -> tuple[RTCChunk, int, int] | None:
        """Return (new_chunk, observed_delay_steps, execution_horizon) when ready."""
        if self._future is None or not self._future.done():
            return None
        relative, anchor_link6, latency, started, s, d_est = self._future.result()
        self._future = None
        observed_delay = estimate_delay_steps(latency, self.frequency)
        # Paper records the index into the new chunk when it arrives (= delay).
        self._delay_buf.append(observed_delay)
        self.last_latency_s = float(latency)
        chunk = build_rtc_chunk(relative, anchor_link6)
        # Consume the guaranteed-executed prefix of the new chunk.
        n = len(chunk.relative_actions)
        if n == 0:
            return chunk, observed_delay, s
        start_index = min(max(0, observed_delay), n - 1)
        trimmed = RTCChunk(
            relative_actions=chunk.relative_actions[start_index:],
            world_link6=chunk.world_link6[start_index:],
            anchor_link6=chunk.anchor_link6,
            created_monotonic=chunk.created_monotonic,
        )
        return trimmed, observed_delay, s
