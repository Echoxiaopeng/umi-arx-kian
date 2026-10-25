#!/usr/bin/env python3
"""Offline-check a Kai0 flatten/fold joint policy against a bimanual zarr dump.

The Kai0 Agilex/FlattenFold contract is absolute dual-arm joints:
  state/action = [Lq1..Lq6, Lg, Rq1..Rq6, Rg]  (14-D)

``collect_bimanual_*`` stores TCP ``action`` (xyz+rotvec+gripper)×2, so this
script builds the ground-truth action horizon from ``robot*_joint_pos`` and
``robot*_gripper_width`` instead of the TCP action field.

Example (policy server already running on :8000)::

  conda activate umi-arx
  cd /home/lenovo/kian/umi-arx-kian
  python scripts/offline_eval_kai0_bimanual_zarr.py \\
    --demo-dir data/collect_bimanual_10_16 \\
    --policy-ip 127.0.0.1 --policy-port 8000 \\
    --num-samples 24 --stride 3
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import click
import numpy as np
import zarr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CLIENT_SRC = ROOT / "packages" / "openpi-client" / "src"
if CLIENT_SRC.is_dir():
    sys.path.insert(0, str(CLIENT_SRC))

from openpi_client.websocket_client_policy import WebsocketClientPolicy  # noqa: E402

ACTION_DIM = 14
DEFAULT_PROMPT = "Flatten and fold the cloth."


def to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        # collect_bimanual stores float32 in [0, 1]
        if float(np.nanmax(image)) <= 1.5:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HxWx3 RGB, got {image.shape}")
    return np.ascontiguousarray(image)


def joint_state_at(data, index: int) -> np.ndarray:
    left = np.asarray(data["robot0_joint_pos"][index], dtype=np.float32).reshape(6)
    right = np.asarray(data["robot1_joint_pos"][index], dtype=np.float32).reshape(6)
    lg = float(np.asarray(data["robot0_gripper_width"][index]).reshape(-1)[0])
    rg = float(np.asarray(data["robot1_gripper_width"][index]).reshape(-1)[0])
    return np.concatenate([left, [lg], right, [rg]], axis=0).astype(np.float32)


def joint_horizon(data, start: int, horizon: int, stride: int) -> np.ndarray | None:
    last = start + (horizon - 1) * stride
    if last >= len(data["robot0_joint_pos"]):
        return None
    rows = [joint_state_at(data, start + k * stride) for k in range(horizon)]
    return np.stack(rows, axis=0)


def sample_indices(
    ends: np.ndarray,
    num_samples: int,
    horizon: int,
    stride: int,
    *,
    all_frames: bool = False,
) -> list[int]:
    starts = np.concatenate([[0], ends[:-1]])
    candidates: list[int] = []
    for start, end in zip(starts, ends, strict=True):
        # Need room for a full action horizon inside the episode.
        usable = int(end - ((horizon - 1) * stride + 1))
        if usable <= start:
            continue
        if all_frames:
            lo, hi = int(start), usable
            step = 1
        else:
            # Keep away from episode edges for sparse sampling.
            lo = int(start + 30)
            hi = int(usable - 5)
            if hi <= lo:
                continue
            step = max(1, (hi - lo) // max(1, num_samples))
        candidates.extend(range(lo, hi, step))
    if not candidates:
        raise click.ClickException("No valid sample indices in dataset")
    if all_frames or len(candidates) <= num_samples:
        return candidates
    rng = np.random.default_rng(0)
    chosen = rng.choice(candidates, size=num_samples, replace=False)
    return sorted(int(x) for x in chosen)


def summarize(pred: np.ndarray, gt: np.ndarray) -> dict:
    err = pred - gt
    abs_err = np.abs(err)
    joint_mask = np.ones(ACTION_DIM, dtype=bool)
    joint_mask[[6, 13]] = False
    return {
        "mae_all": float(abs_err.mean()),
        "mae_joints": float(abs_err[..., joint_mask].mean()),
        "mae_grippers": float(abs_err[..., [6, 13]].mean()),
        "rmse_all": float(np.sqrt((err**2).mean())),
        "mae_per_dim": abs_err.mean(axis=(0, 1)).tolist(),
        "pred_mean": pred.mean(axis=(0, 1)).tolist(),
        "gt_mean": gt.mean(axis=(0, 1)).tolist(),
        "pred_std": pred.std(axis=(0, 1)).tolist(),
        "gt_std": gt.std(axis=(0, 1)).tolist(),
        "pred_min": pred.min(axis=(0, 1)).tolist(),
        "pred_max": pred.max(axis=(0, 1)).tolist(),
        "gt_min": gt.min(axis=(0, 1)).tolist(),
        "gt_max": gt.max(axis=(0, 1)).tolist(),
        "frac_joint_err_lt_0.1": float((abs_err[..., joint_mask] < 0.1).mean()),
        "frac_joint_err_lt_0.3": float((abs_err[..., joint_mask] < 0.3).mean()),
        "max_abs_err": float(abs_err.max()),
    }


@click.command()
@click.option(
    "--demo-dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    default=ROOT / "data" / "collect_bimanual_10_16",
    show_default=True,
)
@click.option("--policy-ip", default="127.0.0.1", show_default=True)
@click.option("--policy-port", default=8000, type=int, show_default=True)
@click.option("--prompt", default=DEFAULT_PROMPT, show_default=True)
@click.option("--num-samples", default=24, type=int, show_default=True)
@click.option(
    "--all-frames",
    is_flag=True,
    default=False,
    help="Evaluate every frame that can fit a full action horizon (ignores --num-samples).",
)
@click.option(
    "--horizon",
    default=50,
    type=int,
    show_default=True,
    help="Action horizon expected from the Kai0 policy.",
)
@click.option(
    "--stride",
    default=3,
    type=int,
    show_default=True,
    help="Dataset is ~60 Hz; stride=3 approximates a 20 Hz control rate.",
)
@click.option("--server-timeout", default=600.0, type=float, show_default=True)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=ROOT / "data_local" / "kai0_offline_eval" / "collect_bimanual_10_16.json",
    show_default=True,
)
def main(
    demo_dir: Path,
    policy_ip: str,
    policy_port: int,
    prompt: str,
    num_samples: int,
    all_frames: bool,
    horizon: int,
    stride: int,
    server_timeout: float,
    output: Path,
) -> None:
    zarr_path = demo_dir / "replay_buffer.zarr"
    if not zarr_path.exists():
        raise click.ClickException(f"Missing {zarr_path}")
    root = zarr.open(str(zarr_path), mode="r")
    data = root["data"]
    ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    indices = sample_indices(
        ends,
        num_samples=num_samples,
        horizon=horizon,
        stride=stride,
        all_frames=all_frames,
    )

    print(f"Dataset {zarr_path}")
    print(
        f"  episodes={len(ends)} frames={int(ends[-1])} samples={len(indices)}"
        f"{' (all usable frames)' if all_frames else ''}"
    )
    print(f"  GT horizon built from joints with stride={stride} (~{60/stride:.1f} Hz)")
    print(f"Connecting ws://{policy_ip}:{policy_port}")
    client = WebsocketClientPolicy(
        host=policy_ip,
        port=policy_port,
        ping_timeout=server_timeout,
        close_timeout=server_timeout,
    )
    print("metadata:", client.get_server_metadata())

    preds = []
    gts = []
    latencies = []
    per_sample = []

    for sample_id, index in enumerate(indices):
        gt = joint_horizon(data, index, horizon=horizon, stride=stride)
        if gt is None:
            continue
        state = joint_state_at(data, index)
        obs = {
            "observation.images.top_head": to_uint8_rgb(data["camera1_rgb"][index]),
            "observation.images.hand_left": to_uint8_rgb(data["camera0_rgb"][index]),
            "observation.images.hand_right": to_uint8_rgb(data["camera2_rgb"][index]),
            "observation.state": state,
            "prompt": prompt,
        }
        t0 = time.monotonic()
        result = client.infer(obs)
        latency = time.monotonic() - t0
        pred = np.asarray(result["actions"], dtype=np.float64)
        if pred.ndim != 2 or pred.shape[1] < ACTION_DIM:
            raise RuntimeError(f"Bad action shape at t={index}: {pred.shape}")
        pred = pred[:horizon, :ACTION_DIM]
        if pred.shape[0] != horizon:
            raise RuntimeError(f"Expected horizon {horizon}, got {pred.shape[0]}")

        mae = float(np.mean(np.abs(pred - gt)))
        preds.append(pred)
        gts.append(gt)
        latencies.append(latency)
        entry = {
            "index": int(index),
            "latency_s": latency,
            "mae": mae,
        }
        if not all_frames:
            entry.update(
                {
                    "state": state.tolist(),
                    "pred0": pred[0].tolist(),
                    "gt0": gt[0].tolist(),
                }
            )
        per_sample.append(entry)
        log_every = 100 if all_frames else 1
        if (sample_id + 1) % log_every == 0 or sample_id == 0 or sample_id + 1 == len(indices):
            print(
                f"[{sample_id+1}/{len(indices)}] t={index} "
                f"latency={latency:.3f}s mae={mae:.4f} "
                f"pred0_grip=({pred[0,6]:.3f},{pred[0,13]:.3f}) "
                f"gt0_grip=({gt[0,6]:.3f},{gt[0,13]:.3f})"
            )

    pred_arr = np.stack(preds, axis=0)
    gt_arr = np.stack(gts, axis=0)
    stats = summarize(pred_arr, gt_arr)
    stats.update(
        {
            "num_samples": int(pred_arr.shape[0]),
            "all_frames": bool(all_frames),
            "horizon": horizon,
            "stride": stride,
            "prompt": prompt,
            "demo_dir": str(demo_dir.resolve()),
            "mean_latency_s": float(np.mean(latencies)),
            "p50_latency_s": float(np.median(latencies)),
            "p90_latency_s": float(np.percentile(latencies, 90)),
            "samples": per_sample,
            "note": (
                "GT actions are future joint+gripper states from the zarr, "
                "not the TCP action field. Dataset ~60 Hz; stride approximates "
                "policy control rate."
            ),
        }
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("\n=== Offline summary ===")
    print(f"samples={stats['num_samples']} horizon={horizon} stride={stride}")
    print(
        f"MAE all={stats['mae_all']:.4f}  joints={stats['mae_joints']:.4f}  "
        f"grippers={stats['mae_grippers']:.4f}  RMSE={stats['rmse_all']:.4f}"
    )
    print(
        f"joint |err|<0.1 rad: {stats['frac_joint_err_lt_0.1']*100:.1f}%  "
        f"|err|<0.3 rad: {stats['frac_joint_err_lt_0.3']*100:.1f}%"
    )
    print(
        f"latency mean={stats['mean_latency_s']:.3f}s  "
        f"p50={stats['p50_latency_s']:.3f}s  p90={stats['p90_latency_s']:.3f}s"
    )
    print("pred joint range (min..max over samples/horizon):")
    for dim, name in enumerate(
        [*(f"Lq{i}" for i in range(6)), "Lg", *(f"Rq{i}" for i in range(6)), "Rg"]
    ):
        print(
            f"  {name:3s} pred[{stats['pred_min'][dim]:+.3f},{stats['pred_max'][dim]:+.3f}] "
            f"gt[{stats['gt_min'][dim]:+.3f},{stats['gt_max'][dim]:+.3f}] "
            f"mae={stats['mae_per_dim'][dim]:.4f}"
        )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
