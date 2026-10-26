#!/usr/bin/env python3
"""Dry-run the RTC control loop against the policy server — no robot motion.

Simulates cursor / commit_end / trim / stitch exactly like ``run_rtc``, but never
calls ``schedule_joint_waypoint``. Safe while cameras/ZMQ are up.

  conda activate umi-arx
  python scripts/test_kai0_rtc_dry_loop.py --policy-ip 127.0.0.1 --policy-port 8000
"""

from __future__ import annotations

from pathlib import Path
import sys
import time

import click
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CLIENT_SRC = ROOT / "packages" / "openpi-client" / "src"
if CLIENT_SRC.is_dir():
    sys.path.insert(0, str(CLIENT_SRC))

from openpi_client.websocket_client_policy import WebsocketClientPolicy  # noqa: E402
from utils.kai0_rtc import JointRTCInferenceWorker  # noqa: E402
from utils.kai0_rtc import build_joint_chunk  # noqa: E402
from utils.kai0_rtc import estimate_delay_steps  # noqa: E402
from utils.kai0_rtc import leftover_absolute  # noqa: E402
from utils.kai0_rtc import stitch_chunk_to_state  # noqa: E402

ACTION_DIM = 14
DEFAULT_PROMPT = "Flatten and fold the cloth."
# Dataset init median (same as eval defaults) — no arm read required.
SEED_STATE = np.array(
    [
        0.025, 0.637, 0.694, -0.621, 0.026, -0.053, 0.084,
        0.052, 0.633, 0.693, -0.632, 0.068, 0.030, 0.067,
    ],
    dtype=np.float32,
)


def fake_obs(state: np.ndarray, prompt: str) -> dict:
    black = np.zeros((224, 224, 3), dtype=np.uint8)
    return {
        "observation.images.top_head": black,
        "observation.images.hand_left": black,
        "observation.images.hand_right": black.copy(),
        "observation.state": np.asarray(state, dtype=np.float32),
        "prompt": prompt,
    }


def with_rtc(obs: dict, *, prev, delay: int, exec_h: int, max_gw: float) -> dict:
    payload = dict(obs)
    payload["enable_rtc"] = True
    payload["inference_delay"] = int(delay)
    payload["execute_horizon"] = int(exec_h)
    payload["max_guidance_weight"] = float(max_gw)
    if prev is not None:
        payload["prev_action_chunk"] = np.asarray(prev, dtype=np.float32)
    return payload


def max_step(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


@click.command()
@click.option("--policy-ip", default="127.0.0.1", show_default=True)
@click.option("--policy-port", default=8000, type=int, show_default=True)
@click.option("--prompt", default=DEFAULT_PROMPT, show_default=True)
@click.option("--frequency", default=20.0, type=float, show_default=True)
@click.option("--s-min", default=40, type=int, show_default=True)
@click.option("--steps-per-inference", default=8, type=int, show_default=True)
@click.option("--max-guidance-weight", default=0.5, type=float, show_default=True)
@click.option("--n-swaps", default=4, type=int, show_default=True, help="RTC swaps to complete.")
@click.option("--max-joint-step", default=0.04, type=float, show_default=True, help="rad/step at 20Hz*0.8")
@click.option("--server-timeout", default=120.0, type=float, show_default=True)
def main(
    policy_ip: str,
    policy_port: int,
    prompt: str,
    frequency: float,
    s_min: int,
    steps_per_inference: int,
    max_guidance_weight: float,
    n_swaps: int,
    max_joint_step: float,
    server_timeout: float,
) -> None:
    print(f"[dry] no robot motion; connecting ws://{policy_ip}:{policy_port}")
    policy = WebsocketClientPolicy(
        host=policy_ip,
        port=policy_port,
        ping_timeout=server_timeout,
        close_timeout=server_timeout,
    )
    meta = policy.get_server_metadata()
    print(f"[dry] metadata supports_rtc={meta.get('supports_rtc')} config={meta.get('config')}")
    if not meta.get("supports_rtc"):
        raise SystemExit("FAIL: server needs --rtc")

    state = SEED_STATE.copy()
    obs0 = fake_obs(state, prompt)

    print("[dry] warm + seeded RTC chunk...")
    t0 = time.monotonic()
    warm = policy.infer(obs0)
    warm_lat = time.monotonic() - t0
    raw = np.asarray(warm["actions"], dtype=np.float32)
    H = int(raw.shape[0])
    if raw.shape != (H, ACTION_DIM) or not np.all(np.isfinite(raw)):
        raise SystemExit(f"FAIL: bad warm actions {raw.shape}")

    seed_prev = np.tile(state[None, :], (H, 1))
    t0 = time.monotonic()
    r0 = policy.infer(
        with_rtc(
            obs0,
            prev=seed_prev,
            delay=0,
            exec_h=min(s_min, H),
            max_gw=max_guidance_weight,
        )
    )
    lat0 = time.monotonic() - t0
    a0 = np.asarray(r0["actions"], dtype=np.float32)
    if a0.shape != (H, ACTION_DIM) or not np.all(np.isfinite(a0)):
        raise SystemExit("FAIL: bad first RTC chunk")
    print(f"[dry] warm_lat={warm_lat:.3f}s rtc0_lat={lat0:.3f}s H={H}")

    init_delay = max(1, estimate_delay_steps(max(lat0, warm_lat), frequency))
    rtc_batch = min(int(steps_per_inference), 8)
    current = build_joint_chunk(a0)
    cursor = 0
    last_commanded = state.astype(np.float64)
    swaps = 0
    step_jumps: list[float] = []
    swap_jumps: list[float] = []
    stall_guard = 0

    def get_obs_bundle() -> dict:
        # Advance proprio along last commanded (no cameras / no arms).
        return {"obs": None, "state": last_commanded.astype(np.float32)}

    def infer_fn(
        *,
        obs_bundle: dict,
        rtc_prev_chunk: np.ndarray,
        rtc_inference_delay: int,
        rtc_execution_horizon: int,
    ) -> tuple[np.ndarray, float]:
        st = obs_bundle["state"]
        started = time.monotonic()
        out = policy.infer(
            with_rtc(
                fake_obs(st, prompt),
                prev=rtc_prev_chunk,
                delay=rtc_inference_delay,
                exec_h=rtc_execution_horizon,
                max_gw=max_guidance_weight,
            )
        )
        return np.asarray(out["actions"], dtype=np.float32), time.monotonic() - started

    worker = JointRTCInferenceWorker(
        infer_fn,
        frequency=frequency,
        s_min=s_min,
        delay_buffer_size=16,
        initial_delay_steps=init_delay,
        action_dim=ACTION_DIM,
        action_horizon=H,
        schedule_batch_size=rtc_batch,
    )
    print(
        f"[dry] loop s_min={s_min} delay~{worker.capped_delay()} "
        f"batch={rtc_batch} target_swaps={n_swaps}"
    )

    t_loop = time.monotonic()
    while swaps < n_swaps:
        if time.monotonic() - t_loop > 180:
            raise SystemExit("FAIL: dry loop timed out (possible deadlock)")

        polled = worker.poll(executed_since_chunk=cursor)
        if polled is not None:
            swap_at = cursor
            new_chunk, trim, s_used = polled
            new_chunk = stitch_chunk_to_state(new_chunk, last_commanded, blend_steps=16)
            jump = max_step(new_chunk.actions[0], last_commanded)
            swap_jumps.append(jump)
            print(
                f"[dry] swap#{swaps+1} cursor={swap_at} trim={trim} s={s_used} "
                f"H={new_chunk.horizon} stitch0_jump={jump:.4f} "
                f"lat={worker.last_latency_s:.3f}s"
            )
            if not np.all(np.isfinite(new_chunk.actions)):
                raise SystemExit("FAIL: NaN/Inf after swap")
            current = new_chunk
            cursor = 0
            swaps += 1
            stall_guard = 0
            if swaps >= n_swaps:
                break

        if cursor >= current.horizon:
            if worker.awaiting_swap:
                release_at = max(cursor, worker.commit_end or cursor)
                done = worker.poll(executed_since_chunk=release_at)
                if done is not None:
                    new_chunk, trim, s_used = done
                    new_chunk = stitch_chunk_to_state(
                        new_chunk, last_commanded, blend_steps=16
                    )
                    jump = max_step(new_chunk.actions[0], last_commanded)
                    swap_jumps.append(jump)
                    print(
                        f"[dry] swap#{swaps+1} (horizon flush) cursor={cursor} "
                        f"trim={trim} s={s_used} H={new_chunk.horizon} "
                        f"stitch0_jump={jump:.4f}"
                    )
                    current = new_chunk
                    cursor = 0
                    swaps += 1
                    stall_guard = 0
                    continue
                stall_guard += 1
                if stall_guard > 500:
                    raise SystemExit(
                        f"FAIL: deadlock cursor={cursor} H={current.horizon} "
                        f"commit_end={worker.commit_end}"
                    )
                time.sleep(0.005)
                continue
            worker.maybe_start(
                executed_since_chunk=max(cursor, 1),
                current_chunk=current,
                get_obs_bundle=get_obs_bundle,
                force=True,
            )
            time.sleep(0.005)
            continue

        if worker.commit_end is not None and cursor >= worker.commit_end:
            while worker.awaiting_swap:
                done = worker.poll(executed_since_chunk=cursor)
                if done is not None:
                    new_chunk, trim, s_used = done
                    new_chunk = stitch_chunk_to_state(
                        new_chunk, last_commanded, blend_steps=16
                    )
                    jump = max_step(new_chunk.actions[0], last_commanded)
                    swap_jumps.append(jump)
                    soft = max(0, int(s_used) - int(trim))
                    print(
                        f"[dry] swap#{swaps+1} cursor={cursor} trim={trim} "
                        f"s={s_used} soft={soft} H={new_chunk.horizon} "
                        f"stitch0_jump={jump:.4f} lat={worker.last_latency_s:.3f}s"
                    )
                    if not np.all(np.isfinite(new_chunk.actions)):
                        raise SystemExit("FAIL: NaN/Inf after swap")
                    current = new_chunk
                    cursor = 0
                    swaps += 1
                    stall_guard = 0
                    break
                time.sleep(0.005)
            continue

        if worker.commit_end is not None:
            remaining_commit = worker.commit_end - cursor
        else:
            _d, _switch_s, s_trigger = worker.plan_horizons(current.horizon)
            remaining_commit = (
                s_trigger - cursor if cursor < s_trigger else worker.delay_align
            )

        remaining = current.horizon - cursor
        n = min(rtc_batch, remaining, remaining_commit)
        if n <= 0:
            stall_guard += 1
            if stall_guard > 500:
                raise SystemExit(
                    f"FAIL: stall n=0 cursor={cursor} H={current.horizon} "
                    f"commit_end={worker.commit_end} busy={worker.busy}"
                )
            time.sleep(0.005)
            continue
        stall_guard = 0

        batch = current.actions[cursor : cursor + n].astype(np.float64)
        # Same per-step joint clamp as eval guarded_chunk (joints only).
        prev = last_commanded.copy()
        for i in range(len(batch)):
            for off in (0, 7):
                delta = batch[i, off : off + 6] - prev[off : off + 6]
                batch[i, off : off + 6] = prev[off : off + 6] + np.clip(
                    delta, -max_joint_step, max_joint_step
                )
            jump = max_step(batch[i], prev)
            step_jumps.append(jump)
            prev = batch[i]
        last_commanded = batch[-1]
        cursor += n
        worker.maybe_start(
            executed_since_chunk=cursor,
            current_chunk=current,
            get_obs_bundle=get_obs_bundle,
        )
        # Simulate wall time of batch at control rate (shortened for dry speed).
        time.sleep(min(0.02, n / frequency * 0.25))

    worker.close(wait=True, timeout_s=5.0)

    sj = np.asarray(step_jumps) if step_jumps else np.array([0.0])
    wj = np.asarray(swap_jumps) if swap_jumps else np.array([0.0])
    print("\n[dry] summary")
    print(f"  swaps_ok={swaps}/{n_swaps}")
    print(
        f"  step max|dq| p50={np.median(sj):.4f} p90={np.percentile(sj,90):.4f} "
        f"max={sj.max():.4f} (limit={max_joint_step})"
    )
    print(
        f"  swap stitch0 jump p50={np.median(wj):.4f} max={wj.max():.4f} "
        f"(expect ~0 after stitch)"
    )
    if sj.max() > max_joint_step + 1e-6:
        raise SystemExit("FAIL: guarded step exceeded max_joint_step")
    if wj.max() > 1e-3:
        raise SystemExit(f"FAIL: stitch left jump {wj.max():.4f}")
    print(
        "\nPASS: RTC dry loop completed without deadlock; actions finite; "
        "stitch C0; steps within guard. Safe to try real arms with same flags."
    )


if __name__ == "__main__":
    main()
