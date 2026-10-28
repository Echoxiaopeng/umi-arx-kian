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
import threading
import time

import numpy as np


def estimate_delay_steps(latency_s: float, frequency: float, *, floor: int = 1) -> int:
    return max(int(floor), int(np.ceil(float(latency_s) * float(frequency))))


class StreamActionBuffer:
    """Official kai0 ARX-style action queue (async infer + control pop).

    Matches ``arx_openpi_inference_rtc.StreamActionBuffer``: latency trim via
    ``k``/``max_k``, temporal overlap blend into ``cur_chunk``, control loop
    pops one action per tick.
    """

    def __init__(
        self,
        *,
        state_dim: int = 14,
        smooth_method: str = "temporal",
    ) -> None:
        self.state_dim = int(state_dim)
        self.smooth_method = str(smooth_method)
        self.lock = threading.Lock()
        self.cur_chunk: deque[np.ndarray] = deque()
        self.k = 0
        self.last_action: np.ndarray | None = None

    def __len__(self) -> int:
        with self.lock:
            return len(self.cur_chunk)

    def integrate_new_chunk(
        self,
        actions_chunk: np.ndarray,
        max_k: int,
        min_m: int = 8,
    ) -> None:
        with self.lock:
            if actions_chunk is None or len(actions_chunk) == 0:
                return
            max_k = max(0, int(max_k))
            min_m = max(1, int(min_m))
            drop_n = min(self.k, max_k)
            if drop_n >= len(actions_chunk):
                return
            arr = np.asarray(actions_chunk, dtype=np.float64)
            new_chunk = [arr[i].copy() for i in range(drop_n, len(arr))]

            if str(self.smooth_method).lower() == "raw":
                self.cur_chunk = deque(new_chunk)
                self.k = 0
                return

            if len(self.cur_chunk) == 0 and self.last_action is not None:
                old_list = [
                    np.asarray(self.last_action, dtype=np.float64).copy()
                    for _ in range(min_m)
                ]
                self.last_action = None
            else:
                old_list = list(self.cur_chunk)
                if 0 < len(old_list) < min_m:
                    tail = np.asarray(old_list[-1], dtype=np.float64).copy()
                    old_list.extend(
                        [tail.copy() for _ in range(min_m - len(old_list))]
                    )
                elif len(old_list) == 0:
                    self.cur_chunk = deque(new_chunk)
                    self.k = 0
                    return

            new_list = list(new_chunk)
            overlap_len = min(len(old_list), len(new_list))
            if overlap_len <= 0:
                self.cur_chunk = deque(new_list)
                self.k = 0
                return
            if len(old_list) > len(new_list):
                old_list = old_list[: len(new_list)]
                overlap_len = len(new_list)

            if overlap_len == 1:
                w_old = np.array([1.0], dtype=np.float64)
            else:
                w_old = np.linspace(1.0, 0.0, overlap_len, dtype=np.float64)
            w_new = 1.0 - w_old
            smoothed = [
                w_old[i] * np.asarray(old_list[i], dtype=np.float64)
                + w_new[i] * np.asarray(new_list[i], dtype=np.float64)
                for i in range(overlap_len)
            ]
            combined = smoothed + new_list[overlap_len:]
            self.cur_chunk = deque([a.copy() for a in combined])
            self.k = 0

    def pop_next_action(self) -> np.ndarray | None:
        with self.lock:
            if len(self.cur_chunk) == 0:
                return None
            if len(self.cur_chunk) == 1:
                self.last_action = np.asarray(self.cur_chunk[0], dtype=np.float64).copy()
            act = np.asarray(self.cur_chunk.popleft(), dtype=np.float64)
            self.k += 1
            return act

    def peek_actions(self, n: int) -> np.ndarray | None:
        """Return up to ``n`` upcoming actions without consuming (for ZMQ lookahead)."""
        with self.lock:
            if len(self.cur_chunk) == 0 or n <= 0:
                return None
            n = min(int(n), len(self.cur_chunk))
            return np.stack(
                [np.asarray(self.cur_chunk[i], dtype=np.float64) for i in range(n)],
                axis=0,
            )

    def clear(self) -> None:
        with self.lock:
            self.cur_chunk.clear()
            self.last_action = None
            self.k = 0


def round_up_to_batch(steps: int, batch: int) -> int:
    """Round ``steps`` up to a multiple of ``batch`` (batch<=1 => no-op)."""
    steps = max(1, int(steps))
    batch = int(batch)
    if batch <= 1:
        return steps
    return ((steps + batch - 1) // batch) * batch


# Delay is aligned to at most this many steps. Using steps_per_inference (e.g. 32)
# as the align quantum would inflate delay≈3 → 32, leave only ~18 actions after
# trim, then deadlock because s_min/delay > remaining horizon.
# Keep ≤4: RTT≈0.12s@20Hz is ~3 steps; aligning to 8 wasted ~5 actions on trim.
DEFAULT_MAX_DELAY_ALIGN = 4


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
) -> tuple[np.ndarray, int]:
    """Unexecuted absolute-joint suffix, padded to ``horizon``.

    Returns ``(padded, real_len)``. Tail padding repeats the last real action
    (never zeros): kai0's Pi0RTC clamps ``execute_horizon`` to
    ``prev.shape[1]`` *before* padding, but a client-side zero-pad to full H
    makes the server think the whole horizon is valid and guides toward 0 —
    which yanks the arms at every swap when ``s_min`` is large.
    """
    actions = np.asarray(chunk.actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[0] == 0:
        dim = int(actions.shape[1]) if actions.ndim == 2 and actions.shape[1] else 14
        return np.zeros((horizon, dim), dtype=np.float32), 0
    dim = int(actions.shape[1])
    leftover = actions[max(0, int(start_index)) :]
    real_len = int(len(leftover))
    out = np.zeros((horizon, dim), dtype=np.float32)
    n = min(real_len, horizon)
    if n > 0:
        out[:n] = leftover[:n]
        if n < horizon:
            out[n:] = leftover[n - 1]
    return out, real_len


def build_joint_chunk(actions: np.ndarray) -> JointRTCChunk:
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected actions [T, D], got {arr.shape}")
    return JointRTCChunk(actions=arr, created_monotonic=time.monotonic())


def temporal_smooth_integrate(
    old_actions: np.ndarray | None,
    new_chunk: JointRTCChunk,
    *,
    min_m: int = 10,
    last_action: np.ndarray | None = None,
) -> JointRTCChunk:
    """Kai0 official ARX ``StreamActionBuffer`` overlap blend (client-side).

    Linearly mixes the unexecuted old suffix with the start of ``new_chunk``
    over ``overlap_len = min(len(old), len(new))``. A short old suffix is
    padded to ``min_m`` with hold-last; if the old suffix is empty, pads from
    ``last_action`` (same as official ``last_action`` path).
    """
    if new_chunk.horizon == 0:
        return new_chunk
    new_arr = np.asarray(new_chunk.actions, dtype=np.float32)
    min_m = max(1, int(min_m))

    if old_actions is None or len(np.asarray(old_actions).reshape(-1, new_arr.shape[-1])) == 0:
        if last_action is None:
            return new_chunk
        last = np.asarray(last_action, dtype=np.float32).reshape(-1)
        if last.shape != new_arr.shape[1:]:
            return new_chunk
        old_list = [last.copy() for _ in range(min_m)]
    else:
        old_arr = np.asarray(old_actions, dtype=np.float32)
        if old_arr.ndim != 2 or old_arr.shape[1] != new_arr.shape[1]:
            return new_chunk
        old_list = [old_arr[i].copy() for i in range(len(old_arr))]
        if 0 < len(old_list) < min_m:
            tail = old_list[-1].copy()
            old_list.extend([tail.copy() for _ in range(min_m - len(old_list))])

    new_list = [new_arr[i].copy() for i in range(len(new_arr))]
    overlap_len = min(len(old_list), len(new_list))
    if overlap_len <= 0:
        return new_chunk
    if len(old_list) > len(new_list):
        old_list = old_list[: len(new_list)]
        overlap_len = len(new_list)

    if overlap_len == 1:
        w_old = np.array([1.0], dtype=np.float32)
    else:
        w_old = np.linspace(1.0, 0.0, overlap_len, dtype=np.float32)
    w_new = 1.0 - w_old
    smoothed = [
        w_old[i] * np.asarray(old_list[i], dtype=np.float32)
        + w_new[i] * np.asarray(new_list[i], dtype=np.float32)
        for i in range(overlap_len)
    ]
    combined = np.stack(smoothed + new_list[overlap_len:], axis=0).astype(np.float32)
    return JointRTCChunk(actions=combined, created_monotonic=new_chunk.created_monotonic)


def apply_rtc_client_smooth(
    new_chunk: JointRTCChunk,
    *,
    method: str = "temporal",
    old_chunk: JointRTCChunk | None = None,
    old_cursor: int = 0,
    last_commanded: np.ndarray | None = None,
    prev_commanded: np.ndarray | None = None,
    blend_steps: int = 10,
) -> JointRTCChunk:
    """Dispatch client-side RTC boundary smoothing.

    - ``temporal``: official kai0 ARX StreamActionBuffer overlap blend
    - ``c1``: local C1 coast + skip0 (``stitch_chunk_to_state``)
    - ``raw``: no client blend (server RTC guidance only)
    """
    key = str(method).strip().lower()
    if key in {"raw", "none", "off"}:
        return new_chunk
    if key in {"c1", "stitch"}:
        prev_vel = None
        if last_commanded is not None and prev_commanded is not None:
            prev_vel = (
                np.asarray(last_commanded, dtype=np.float32)
                - np.asarray(prev_commanded, dtype=np.float32)
            )
        return stitch_chunk_to_state(
            new_chunk,
            last_commanded,
            blend_steps=max(1, int(blend_steps)),
            prev_velocity=prev_vel,
        )
    # Default: official temporal overlap.
    old_actions = None
    if old_chunk is not None and old_chunk.horizon > 0:
        start = max(0, min(int(old_cursor), old_chunk.horizon))
        if start < old_chunk.horizon:
            old_actions = old_chunk.actions[start:]
    return temporal_smooth_integrate(
        old_actions,
        new_chunk,
        min_m=max(1, int(blend_steps)),
        last_action=last_commanded,
    )


def stitch_chunk_to_state(
    chunk: JointRTCChunk,
    last_commanded: np.ndarray | None,
    *,
    blend_steps: int = 16,
    prev_velocity: np.ndarray | None = None,
    keep_anchor: bool = False,
) -> JointRTCChunk:
    """Temporal blend a new chunk onto the last commanded joints (kai0-style).

    Builds a C0 anchor at ``last``, optionally C1-coasts with ``prev_velocity``,
    then by default **drops** the duplicate ``actions[0] == last`` so the first
    scheduled waypoint continues motion (avoids a one-step dq=0 hitch at every
    mid-chunk RTC swap).

    Set ``keep_anchor=True`` when resuming from standstill (e.g. DAgger R after
    drag): dropping the anchor would turn rest → sudden first step into a jerk.

    Without ``prev_velocity``, uses a position-only linear fade (C0).
    """
    if last_commanded is None or chunk.horizon == 0:
        return chunk
    last = np.asarray(last_commanded, dtype=np.float32).reshape(-1)
    actions = np.asarray(chunk.actions, dtype=np.float32).copy()
    if last.shape != actions.shape[1:]:
        return chunk
    n = min(max(1, int(blend_steps)), len(actions))
    if n == 1:
        actions[0] = last
        return JointRTCChunk(actions=actions, created_monotonic=chunk.created_monotonic)

    vel = None
    if prev_velocity is not None:
        vel = np.asarray(prev_velocity, dtype=np.float32).reshape(-1)
        if vel.shape != last.shape:
            vel = None

    if vel is None:
        # C0 only: fade from last pose toward the new chunk.
        w_old = np.linspace(1.0, 0.0, n, dtype=np.float32)
        for i in range(n):
            actions[i] = w_old[i] * last + (1.0 - w_old[i]) * actions[i]
        actions[0] = last
    else:
        # C1: coast with incoming velocity, smoothstep into the new chunk.
        # t=0 → fully coast (pos=last, step≈vel); t=1 → fully new.
        for i in range(n):
            t = float(i) / float(n - 1)
            w_new = t * t * (3.0 - 2.0 * t)  # smoothstep
            coast = last + float(i) * vel
            actions[i] = (1.0 - w_new) * coast + w_new * actions[i]
        actions[0] = last  # exact C0; step0→1 ≈ vel when blend is long

    # Drop the duplicate start pose (== last). Re-scheduling it yields a one-step
    # dq=0 hitch (fast→stop→go) at every swap; the next waypoint already carries C1.
    # Keep it when starting from rest (drag→policy) so the first command is hold.
    if (
        not keep_anchor
        and len(actions) >= 2
        and float(np.max(np.abs(actions[0] - last))) < 1e-5
    ):
        actions = actions[1:]

    return JointRTCChunk(actions=actions, created_monotonic=chunk.created_monotonic)


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
        schedule_batch_size: int = 1,
        max_delay_align: int = DEFAULT_MAX_DELAY_ALIGN,
        soft_min: int = 20,
    ) -> None:
        self._infer_fn = infer_fn
        self.frequency = float(frequency)
        self.s_min = int(s_min)
        self.soft_min = max(0, int(soft_min))
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.schedule_batch_size = max(1, int(schedule_batch_size))
        # Align delay to min(batch, max_delay_align), never to a huge batch like 32.
        self.delay_align = max(
            1, min(int(self.schedule_batch_size), int(max_delay_align))
        )
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kai0-rtc")
        self._future: Future | None = None
        self._closed = False
        init_d = min(
            round_up_to_batch(int(initial_delay_steps), self.delay_align),
            self.delay_align,
        )
        self._delay_buf: deque[int] = deque([init_d], maxlen=int(delay_buffer_size))
        self.last_latency_s: float = float(initial_delay_steps) / self.frequency
        # Set while an infer is in flight / awaiting commit_end before swap.
        self._pending_start_executed: int | None = None
        self._pending_delay: int | None = None
        # Completed infer held until cursor reaches start+delay (do not early-swap).
        self._ready_result: tuple | None = None

    def capped_delay(self) -> int:
        """Latency delay, rounded up to delay_align but never above delay_align max quantum."""
        d = round_up_to_batch(self.delay_estimate, self.delay_align)
        # Hard cap: keep enough of the 50-step chunk after trim.
        cap = min(self.delay_align, max(1, self.action_horizon // 3))
        return min(d, cap)

    def plan_horizons(self, chunk_horizon: int) -> tuple[int, int, int]:
        """Return ``(delay, switch_s, infer_start_cursor)``.

        Open-loop-like cadence: swap after ``switch_s ≈ s_min`` steps (capped so
        ``switch_s + 0 <= H`` via ``switch_s <= H - d``). Infer starts at
        ``switch_s - d`` so the delay prefix fits and leftover is as long as
        possible for soft weights ``[d, exec_h)`` where ``exec_h = leftover``.

        Do **not** cap to ``(H+d)//2`` — that forced ~25-step chunks and made
        boundaries more frequent (felt worse than open-loop).
        """
        d = self.capped_delay()
        h = max(1, int(chunk_horizon))
        # commit_end = start + d = switch_s, need switch_s <= h
        switch_s = min(max(d, self.s_min), max(d, h - d))
        start_trigger = max(0, int(switch_s) - int(d))
        return int(d), int(switch_s), int(start_trigger)

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
        self._pending_start_executed = None
        self._pending_delay = None
        self._ready_result = None
        self._executor.shutdown(wait=wait and not still_running, cancel_futures=True)
        return not still_running

    @property
    def delay_estimate(self) -> int:
        return int(max(self._delay_buf))

    @property
    def busy(self) -> bool:
        return self._future is not None and not self._future.done()

    @property
    def awaiting_swap(self) -> bool:
        """Infer finished (or in flight) but delay prefix not yet fully executed."""
        return self.commit_end is not None

    @property
    def commit_end(self) -> int | None:
        """Max old-chunk index to execute before applying the new chunk."""
        if self._pending_start_executed is None or self._pending_delay is None:
            return None
        return int(self._pending_start_executed) + int(self._pending_delay)

    def maybe_start(
        self,
        *,
        executed_since_chunk: int,
        current_chunk: JointRTCChunk,
        get_obs_bundle: Callable[[], dict],
        force: bool = False,
    ) -> None:
        if self._closed or self.busy or self._ready_result is not None:
            return
        if self.commit_end is not None and not force:
            # Still draining the previous infer's delay prefix.
            return
        horizon = int(current_chunk.horizon)
        d, switch_s, start_trigger = self.plan_horizons(horizon)
        if not force and executed_since_chunk < start_trigger:
            return
        start_executed = int(executed_since_chunk)
        d_pending = int(d)
        # commit_end = start + d should equal switch_s when started on time.
        if start_executed + d_pending > horizon:
            d_pending = max(0, horizon - start_executed)
        obs_bundle = get_obs_bundle()
        leftover, real_len = leftover_absolute(
            current_chunk,
            start_index=executed_since_chunk,
            horizon=self.action_horizon,
        )
        # Guidance horizon: soft is [d, exec_h). Real leftover alone is often
        # short (soft ≈ H - switch_s ≈ 8–10). Extend into the hold-last pad so
        # the server keeps guiding toward the old end pose for soft_min steps.
        soft_min = int(getattr(self, "soft_min", 20))
        exec_h = max(d_pending, real_len if real_len > 0 else d_pending)
        exec_h = min(
            self.action_horizon,
            max(exec_h, d_pending + max(0, soft_min)),
        )
        if force:
            exec_h = max(d_pending, min(exec_h, max(1, int(executed_since_chunk))))
        s = exec_h
        started = time.monotonic()
        self._pending_start_executed = start_executed
        self._pending_delay = d_pending

        def _job():
            actions, latency = self._infer_fn(
                obs_bundle=obs_bundle,
                rtc_prev_chunk=leftover,
                rtc_inference_delay=d_pending,
                rtc_execution_horizon=s,
            )
            return actions, latency, started, s, d_pending, start_executed

        self._future = self._executor.submit(_job)

    def poll(
        self, executed_since_chunk: int | None = None
    ) -> tuple[JointRTCChunk, int, int] | None:
        """Return (new_chunk, trim_steps, execution_horizon) when ready to swap.

        Holds the completed infer until ``executed_since_chunk >= start + delay``
        so we do not apply a new chunk before the RTC-committed prefix runs.
        """
        if self._future is not None and self._future.done():
            self._ready_result = self._future.result()
            self._future = None

        if self._ready_result is None:
            return None

        actions, latency, _started, s, d_est, start_executed = self._ready_result
        commit = int(start_executed) + int(d_est)
        if executed_since_chunk is not None and int(executed_since_chunk) < commit:
            # Infer done early; keep holding until delay prefix is executed.
            return None

        self._ready_result = None
        self._pending_start_executed = None
        self._pending_delay = None

        observed_delay = estimate_delay_steps(latency, self.frequency)
        steps_during = 0
        if executed_since_chunk is not None:
            steps_during = max(0, int(executed_since_chunk) - int(start_executed))
        trim = max(observed_delay, steps_during, int(d_est))
        trim = round_up_to_batch(trim, self.delay_align)
        cap = min(self.delay_align, max(1, self.action_horizon // 3))
        trim = min(trim, cap)
        self._delay_buf.append(trim)
        self.last_latency_s = float(latency)

        chunk = build_joint_chunk(actions)
        n = chunk.horizon
        if n == 0:
            return chunk, trim, s
        start_index = min(max(0, trim), n - 1)
        trimmed = JointRTCChunk(
            actions=chunk.actions[start_index:],
            created_monotonic=chunk.created_monotonic,
        )
        return trimmed, trim, s
