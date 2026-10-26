#!/usr/bin/env python3
"""Smoke-test Kai0 RTC without robots or cameras.

Talks only to the policy websocket. Requires the server started with --rtc::

  # terminal A (kai0 repo)
  uv run scripts/serve_kai0_policy.py --checkpoint <jax_ckpt> --rtc --port 8000

  # terminal B
  conda activate umi-arx
  python scripts/test_kai0_rtc_smoke.py --policy-ip 127.0.0.1 --policy-port 8000
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
from utils.kai0_rtc import build_joint_chunk  # noqa: E402
from utils.kai0_rtc import estimate_delay_steps  # noqa: E402
from utils.kai0_rtc import leftover_absolute  # noqa: E402

ACTION_DIM = 14
DEFAULT_PROMPT = "Flatten and fold the cloth."


def fake_obs(state: np.ndarray, prompt: str) -> dict:
    """Minimal Kai0 observation: 224x224 RGB + 14-D joint state."""
    black = np.zeros((224, 224, 3), dtype=np.uint8)
    return {
        "observation.images.top_head": black,
        "observation.images.hand_left": black,
        "observation.images.hand_right": black.copy(),
        "observation.state": np.asarray(state, dtype=np.float32),
        "prompt": prompt,
    }


def with_rtc(
    obs: dict,
    *,
    prev: np.ndarray | None,
    delay: int,
    exec_h: int,
    max_gw: float,
) -> dict:
    payload = dict(obs)
    payload["enable_rtc"] = True
    payload["inference_delay"] = int(delay)
    payload["execute_horizon"] = int(exec_h)
    payload["max_guidance_weight"] = float(max_gw)
    if prev is not None:
        payload["prev_action_chunk"] = np.asarray(prev, dtype=np.float32)
    return payload


@click.command()
@click.option("--policy-ip", default="127.0.0.1", show_default=True)
@click.option("--policy-port", default=8000, type=int, show_default=True)
@click.option("--prompt", default=DEFAULT_PROMPT, show_default=True)
@click.option("--frequency", default=20.0, type=float, show_default=True)
@click.option("--s-min", default=8, type=int, show_default=True)
@click.option("--max-guidance-weight", default=0.5, type=float, show_default=True)
@click.option("--server-timeout", default=120.0, type=float, show_default=True)
def main(
    policy_ip: str,
    policy_port: int,
    prompt: str,
    frequency: float,
    s_min: int,
    max_guidance_weight: float,
    server_timeout: float,
) -> None:
    print(f"Connecting ws://{policy_ip}:{policy_port} ...")
    policy = WebsocketClientPolicy(
        host=policy_ip,
        port=policy_port,
        ping_timeout=server_timeout,
        close_timeout=server_timeout,
    )
    meta = policy.get_server_metadata()
    print(f"metadata={meta}")
    if not meta.get("supports_rtc"):
        raise SystemExit(
            "FAIL: server metadata supports_rtc is not true.\n"
            "Restart with: uv run scripts/serve_kai0_policy.py "
            "--checkpoint <jax_ckpt> --rtc --port 8000"
        )
    print("OK: supports_rtc=true")

    # Neutral-ish dual-arm joint seed (same dim as Kai0).
    state = np.array(
        [
            0.02, 0.64, 0.69, -0.62, 0.03, -0.05, 0.08,
            0.05, 0.63, 0.69, -0.63, 0.07, 0.03, 0.07,
        ],
        dtype=np.float32,
    )
    obs0 = fake_obs(state, prompt)

    # --- round 1: warm + seeded RTC ---
    print("\n[1] warm infer (no RTC fields)...")
    t0 = time.monotonic()
    warm = policy.infer(obs0)
    warm_lat = time.monotonic() - t0
    warm_act = np.asarray(warm["actions"], dtype=np.float32)
    print(f"    latency={warm_lat:.3f}s shape={warm_act.shape}")
    if warm_act.ndim != 2 or warm_act.shape[1] != ACTION_DIM:
        raise SystemExit(f"FAIL: unexpected warm actions shape {warm_act.shape}")
    if not np.all(np.isfinite(warm_act)):
        raise SystemExit("FAIL: warm actions contain NaN/Inf")
    H = int(warm_act.shape[0])

    seed_prev = np.tile(state[None, :], (H, 1))
    print(f"[2] RTC infer with tiled-state prev (H={H})...")
    t0 = time.monotonic()
    r1 = policy.infer(
        with_rtc(
            obs0,
            prev=seed_prev,
            delay=0,
            exec_h=min(s_min, H),
            max_gw=max_guidance_weight,
        )
    )
    lat1 = time.monotonic() - t0
    a1 = np.asarray(r1["actions"], dtype=np.float32)
    print(f"    latency={lat1:.3f}s shape={a1.shape} first={np.round(a1[0], 3)}")
    if a1.shape != warm_act.shape:
        raise SystemExit(f"FAIL: RTC actions shape {a1.shape} != warm {warm_act.shape}")
    if not np.all(np.isfinite(a1)):
        raise SystemExit("FAIL: RTC actions contain NaN/Inf")

    # --- round 2: leftover + delay (simulates overlapping replan) ---
    cursor = min(s_min, H - 1)
    delay = estimate_delay_steps(lat1, frequency)
    s = max(delay, s_min)
    chunk = build_joint_chunk(a1)
    leftover, real_len = leftover_absolute(chunk, start_index=cursor, horizon=H)
    s = min(s, max(delay + 1, real_len))
    print(
        f"[3] RTC replan leftover cursor={cursor} delay={delay} "
        f"exec_h={s} real_len={real_len} "
        f"leftover_finite={bool(np.all(np.isfinite(leftover)))}..."
    )
    # Slightly nudge state as if the arm moved along the chunk.
    state2 = a1[min(cursor, len(a1) - 1)].astype(np.float32)
    obs1 = fake_obs(state2, prompt)
    t0 = time.monotonic()
    r2 = policy.infer(
        with_rtc(
            obs1,
            prev=leftover,
            delay=delay,
            exec_h=min(s, H),
            max_gw=max_guidance_weight,
        )
    )
    lat2 = time.monotonic() - t0
    a2 = np.asarray(r2["actions"], dtype=np.float32)
    print(f"    latency={lat2:.3f}s shape={a2.shape} first={np.round(a2[0], 3)}")
    if a2.shape != a1.shape or not np.all(np.isfinite(a2)):
        raise SystemExit("FAIL: second RTC infer bad shape or NaN")

    # Overlap continuity check: guided region should stay near leftover.
    d = min(delay, H - 1)
    e = min(s, H)
    if e > d:
        overlap_err = float(np.mean(np.abs(a2[d:e] - leftover[d:e])))
        print(
            f"[4] overlap MAE on [{d},{e}) vs leftover: {overlap_err:.4f} "
            f"(lower => RTC guidance pulling toward prev)"
        )
    else:
        overlap_err = float("nan")
        print("[4] skip overlap MAE (delay >= exec_h)")

    print(
        "\nPASS: RTC websocket path works "
        f"(supports_rtc, two RTC infers OK"
        + (f", overlap_MAE={overlap_err:.4f}" if np.isfinite(overlap_err) else "")
        + "). No robot was used."
    )


if __name__ == "__main__":
    main()
