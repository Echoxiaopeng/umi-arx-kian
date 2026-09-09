#!/usr/bin/env python3
"""Evaluate the served ShellGame checkpoint against a raw real-robot demo.

The script reproduces the zarr -> relative EEF10 conversion used by training,
uploads the exact 241-frame visual history once, and compares sampled 16-step
action chunks with the converted ground truth at several grasp frames.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import zarr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CLIENT_SRC = ROOT / "packages" / "openpi-client" / "src"
if CLIENT_SRC.is_dir():
    sys.path.insert(0, str(CLIENT_SRC))

from openpi_client.websocket_client_policy import WebsocketClientPolicy  # noqa: E402
from utils.shellgame_lerobot_convert import build_source_indices  # noqa: E402
from utils.shellgame_lerobot_convert import detect_grasp_start  # noqa: E402
from utils.shellgame_lerobot_convert import relative_episode  # noqa: E402
from utils.shellgame_openpi import ACTION_HORIZON  # noqa: E402
from utils.shellgame_openpi import HISTORY_FRAMES  # noqa: E402
from utils.shellgame_openpi import SWAP_FRAMES  # noqa: E402
from utils.shellgame_openpi import ShellGameHistory  # noqa: E402
from utils.shellgame_openpi import ShellGamePolicyClient  # noqa: E402
from utils.shellgame_openpi import image_to_uint8  # noqa: E402

DEFAULT_PROMPT = "The shell game has ended. Grasp and lift the cup containing the ball."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--demo-dir",
        type=Path,
        default=ROOT / "data" / "cup_demo",
    )
    parser.add_argument("--policy-ip", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=8000)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--samples-per-frame", type=int, default=3)
    parser.add_argument("--frame-step", type=int, default=10)
    parser.add_argument(
        "--norm-stats",
        type=Path,
        default=Path(
            "/home/lenovo/kian/openpi-umi-icrl/checkpoints/"
            "pi0_mem_semantic_action_shellgame_real_wrist_eef10/"
            "semantic_action_306_degap_pi05_seed42_v1/21000/21000/"
            "assets/norm_stats.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data_local" / "shellgame_eval" / "offline_checkpoint_21000.json",
    )
    return parser.parse_args()


def load_episode(demo_dir: Path) -> dict[str, np.ndarray]:
    root = zarr.open(str(demo_dir / "replay_buffer.zarr"), mode="r")
    end = int(np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)[0])
    data = root["data"]
    keys = (
        "robot0_eef_pos",
        "robot0_eef_rot_axis_angle",
        "robot0_gripper_width",
        "action",
        "camera0_rgb",
    )
    return {key: np.asarray(data[key][:end]) for key in keys}


def make_history(images: np.ndarray, poses6: np.ndarray) -> ShellGameHistory:
    frames = [image_to_uint8(frame) for frame in images[:HISTORY_FRAMES]]
    if len(frames) != HISTORY_FRAMES:
        raise ValueError(f"Need {HISTORY_FRAMES} history images, got {len(frames)}")
    history = ShellGameHistory()
    history.initial_frame = frames[0]
    history.episode_start_pose6 = np.asarray(poses6[0], dtype=np.float64).copy()
    history.swap_segments = [
        frames[1 + segment * SWAP_FRAMES : 1 + (segment + 1) * SWAP_FRAMES]
        for segment in range(3)
    ]
    if not history.ready or history.frame_count != HISTORY_FRAMES:
        raise RuntimeError(f"Invalid reconstructed history: {history.frame_count}")
    return history


def cosine(a: np.ndarray, b: np.ndarray, threshold: float = 1e-5) -> np.ndarray:
    a_norm = np.linalg.norm(a, axis=-1)
    b_norm = np.linalg.norm(b, axis=-1)
    valid = (a_norm > threshold) & (b_norm > threshold)
    out = np.full(a_norm.shape, np.nan, dtype=np.float64)
    out[valid] = np.sum(a[valid] * b[valid], axis=-1) / (a_norm[valid] * b_norm[valid])
    return out


def finite_mean(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(finite.mean()) if finite.size else None


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def main() -> None:
    args = parse_args()
    if args.samples_per_frame <= 0 or args.frame_step <= 0:
        raise ValueError("samples-per-frame and frame-step must be positive")

    episode = load_episode(args.demo_dir.expanduser().resolve())
    positions = np.asarray(episode["robot0_eef_pos"], dtype=np.float64)
    rotvecs = np.asarray(episode["robot0_eef_rot_axis_angle"], dtype=np.float64)
    poses6 = np.concatenate([positions, rotvecs], axis=-1)
    gripper = np.asarray(episode["robot0_gripper_width"], dtype=np.float64).reshape(-1)
    raw_actions = np.asarray(episode["action"], dtype=np.float64)
    images = np.asarray(episode["camera0_rgb"])

    grasp_start, used_fallback = detect_grasp_start(positions, raw_actions)
    source_indices = build_source_indices(len(positions), grasp_start=grasp_start)
    state_pos, state_rot6d, state_gripper, gt_actions = relative_episode(
        positions, rotvecs, gripper, raw_actions, source_indices
    )
    state = np.concatenate([state_pos, state_rot6d, state_gripper], axis=-1)

    label_path = args.demo_dir / "labels.jsonl"
    label = json.loads(label_path.read_text(encoding="utf-8").splitlines()[0])
    slot_names = ("left", "middle", "right")
    expected_final_cup = slot_names[int(label["final_cup"])]

    history = make_history(images, poses6)
    client = WebsocketClientPolicy(
        host=args.policy_ip,
        port=args.policy_port,
        ping_timeout=600,
        close_timeout=600,
    )
    metadata = client.get_server_metadata()
    policy = ShellGamePolicyClient(client, cached_infer=True)
    if not policy.cached_infer_enabled:
        raise RuntimeError("Policy server does not support cached ShellGame inference")

    print(f"server metadata: {metadata}", flush=True)
    print(
        f"raw_frames={len(positions)} grasp_start={grasp_start} "
        f"converted_frames={len(source_indices)} fallback={used_fallback}",
        flush=True,
    )
    cache_started = time.monotonic()
    cache_result = policy.upload_history(history, prompt=args.prompt)
    print(
        f"history cache: {time.monotonic() - cache_started:.3f}s; "
        f"memory={cache_result.get('memory')}",
        flush=True,
    )

    first_output = HISTORY_FRAMES - 1
    early_indices = [first_output + offset for offset in (0, 1, 2, 3, 5)]
    eval_indices = sorted(
        set(
            index
            for index in [
                *early_indices,
                *range(first_output, len(source_indices), args.frame_step),
            ]
            if index < len(source_indices)
        )
    )
    if eval_indices[-1] != len(source_indices) - 1:
        eval_indices.append(len(source_indices) - 1)

    rows: list[dict] = []
    all_pred: list[np.ndarray] = []
    all_gt: list[np.ndarray] = []
    for output_index in eval_indices:
        source_index = int(source_indices[output_index])
        chunk_indices = np.minimum(
            output_index + np.arange(ACTION_HORIZON), len(gt_actions) - 1
        )
        gt = np.asarray(gt_actions[chunk_indices], dtype=np.float64)
        predictions = []
        timings = []
        memories = []
        for _ in range(args.samples_per_frame):
            started = time.monotonic()
            result = policy.infer_step(
                history,
                current_frame=images[source_index],
                current_pose6=poses6[source_index],
                gripper_width=float(gripper[source_index]),
                prompt=args.prompt,
            )
            timings.append(time.monotonic() - started)
            pred = np.asarray(result["actions"], dtype=np.float64)
            if pred.shape != gt.shape:
                raise ValueError(f"Policy returned {pred.shape}, expected {gt.shape}")
            predictions.append(pred)
            memories.append(result.get("memory"))

        pred_samples = np.stack(predictions, axis=0)
        pred = pred_samples.mean(axis=0)
        xyz_error = pred[:, :3] - gt[:, :3]
        xyz_cos = cosine(pred[:, :3], gt[:, :3])
        row = {
            "output_index": output_index,
            "source_index": source_index,
            "pred_step0_xyz_m": pred[0, :3].tolist(),
            "gt_step0_xyz_m": gt[0, :3].tolist(),
            "pred_chunk_xyz_mean_m": pred[:, :3].mean(axis=0).tolist(),
            "gt_chunk_xyz_mean_m": gt[:, :3].mean(axis=0).tolist(),
            "pred_step0_norm_mm": float(np.linalg.norm(pred[0, :3]) * 1000),
            "gt_step0_norm_mm": float(np.linalg.norm(gt[0, :3]) * 1000),
            "xyz_rmse_mm": float(np.sqrt(np.mean(np.square(xyz_error))) * 1000),
            "xyz_cosine_mean": finite_mean(xyz_cos),
            "latency_s_mean": float(np.mean(timings)),
            "sample_xyz_std_mm": float(np.mean(np.std(pred_samples[:, :, :3], axis=0)) * 1000),
            "memory": memories[-1],
        }
        rows.append(row)
        all_pred.append(pred)
        all_gt.append(gt)
        print(
            f"out={output_index:3d} src={source_index:3d} "
            f"p0_mm={np.round(pred[0, :3] * 1000, 2)} "
            f"gt0_mm={np.round(gt[0, :3] * 1000, 2)} "
            f"rmse={row['xyz_rmse_mm']:.2f}mm cos={row['xyz_cosine_mean']}",
            flush=True,
        )

    pred = np.stack(all_pred, axis=0)
    gt = np.stack(all_gt, axis=0)
    error = pred - gt
    zero_error = gt
    xyz_cos = cosine(pred[..., :3].reshape(-1, 3), gt[..., :3].reshape(-1, 3))

    norm_payload = json.loads(args.norm_stats.read_text(encoding="utf-8"))["norm_stats"]
    action_std = np.asarray(norm_payload["actions"]["std"], dtype=np.float64)
    state_mean = np.asarray(norm_payload["state"]["mean"], dtype=np.float64)
    state_std = np.asarray(norm_payload["state"]["std"], dtype=np.float64)
    state_z = (state[eval_indices] - state_mean) / (state_std + 1e-6)
    normalized_error = error[..., :10] / (action_std[:10] + 1e-6)
    xyz_rmse_mm = float(np.sqrt(np.mean(np.square(error[..., :3]))) * 1000)
    zero_rmse_mm = float(np.sqrt(np.mean(np.square(zero_error[..., :3]))) * 1000)
    early_count = sum(index <= first_output + 50 for index in eval_indices)
    early_cos = cosine(
        pred[:early_count, :, :3].reshape(-1, 3),
        gt[:early_count, :, :3].reshape(-1, 3),
    )
    summary = {
        "n_eval_frames": len(eval_indices),
        "samples_per_frame": args.samples_per_frame,
        "xyz_rmse_mm": xyz_rmse_mm,
        "xyz_zero_baseline_rmse_mm": zero_rmse_mm,
        "xyz_rmse_vs_zero_ratio": xyz_rmse_mm / zero_rmse_mm,
        "xyz_mae_mm_by_dim": (np.mean(np.abs(error[..., :3]), axis=(0, 1)) * 1000).tolist(),
        "xyz_pred_abs_mean_mm_by_dim": (
            np.mean(np.abs(pred[..., :3]), axis=(0, 1)) * 1000
        ).tolist(),
        "xyz_gt_abs_mean_mm_by_dim": (
            np.mean(np.abs(gt[..., :3]), axis=(0, 1)) * 1000
        ).tolist(),
        "xyz_direction_cosine_mean": finite_mean(xyz_cos),
        "early_xyz_direction_cosine_mean": finite_mean(early_cos),
        "xyz_direction_opposite_fraction": float(np.mean(xyz_cos[np.isfinite(xyz_cos)] < 0)),
        "xyz_sign_agreement_fraction_by_dim": np.mean(
            np.sign(pred[..., :3]) == np.sign(gt[..., :3]), axis=(0, 1)
        ).tolist(),
        "normalized_rmse_by_dim": np.sqrt(
            np.mean(np.square(normalized_error), axis=(0, 1))
        ).tolist(),
        "step0_pred_xyz_mean_mm": (pred[:, 0, :3].mean(axis=0) * 1000).tolist(),
        "step0_gt_xyz_mean_mm": (gt[:, 0, :3].mean(axis=0) * 1000).tolist(),
        "step0_y_positive_fraction": float(np.mean(pred[:, 0, 1] > 0)),
        "step0_z_negative_fraction": float(np.mean(pred[:, 0, 2] < 0)),
        "state_max_abs_z_by_dim": np.max(np.abs(state_z), axis=0).tolist(),
        "state_fraction_abs_z_gt_3": float(np.mean(np.abs(state_z) > 3)),
        "expected_final_cup": expected_final_cup,
        "memory_predicted_final_cup": rows[-1]["memory"].get("predicted_final_cup")
        if rows[-1]["memory"]
        else None,
    }
    payload = {
        "demo_dir": str(args.demo_dir.expanduser().resolve()),
        "norm_stats": str(args.norm_stats.expanduser().resolve()),
        "server": {"host": args.policy_ip, "port": args.policy_port, "metadata": metadata},
        "conversion": {
            "raw_frames": len(positions),
            "grasp_start": grasp_start,
            "converted_frames": len(source_indices),
            "used_fallback": used_fallback,
        },
        "summary": summary,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )
    print(
        "\nSUMMARY",
        json.dumps(summary, indent=2, ensure_ascii=False, default=json_default),
        flush=True,
    )
    print(f"saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
