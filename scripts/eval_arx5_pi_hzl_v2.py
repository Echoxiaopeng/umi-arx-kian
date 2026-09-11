#!/usr/bin/env python3
'''Deploy the real-wrist Pi0 semantic-memory policy on one ARX5 arm.
# 推理 (M6 cached + RTC soft inpainting)
.venv/bin/python scripts/mem/serve_shellgame_real_m6_cached.py \
--checkpoint /home/lenovo/kian/openpi-umi-icrl/checkpoints/cup_grasp_0907_prompt_only_h16_from14999_v1/12000   \
--condition-mode prompt_only   \
--action-horizon 16   \
--port 8017

# 中间层 (RTC)
deactivate
conda activate umi-arx

python scripts/eval_arx5_pi_hzl_v2.py \
  --policy-ip 127.0.0.1 \
  --policy-port 8017 \
  --robot-ip 127.0.0.1 \
  --robot-port 8765 \
  --init-pose-frame direct \
  --direct-zmq-origin 0.094905,-0.002323,0.158023 \
  --eef-offset 0.145 \
  --rtc \
  --rtc-s-min 8 \
  --steps-per-inference 8 \
  --max-duration 600 \
  --action-lead-time 0.05 \
  --frequency 20 \
  --server-warmup-timeout 600 \
  --grasp-only \
  --grasp-cup left \
  --cached-infer 2>&1 | tee shellgame_grasp_h16_rtc_trace.log




python scripts/eval_arx5_pi_hzl_v2.py \
  --policy-ip 127.0.0.1 \
  --policy-port 8017 \
  --robot-ip 127.0.0.1 \
  --robot-port 8765 \
  --init-pose-frame direct \
  --direct-zmq-origin 0.094905,-0.002323,0.158023 \
  --eef-offset 0.145 \
  --steps-per-inference 16 \
  --max-duration 600 \
  --action-lead-time 0.0 \
  --frequency 20 \
  --auto-capture \
  --server-warmup-timeout 600 \
  --grasp-only \
  --grasp-cup middle \
  --cached-infer 2>&1 | tee shellgame_grasp_h16_rtc_trace.log


Controls
--------
F (auto, default): one F saves the initial frame and continuously records all 3×80 swaps; policy auto-starts at 241
F (manual, --manual-capture): capture initial frame, then each F starts one swap; press C to run policy
C: start policy (manual capture, or --grasp-only after dummy history is seeded)
S: stop policy and return to human control
P: in human mode, return to init_pose and reset history for the next round
Q: quit
--grasp-only --grasp-cup {left,middle,right}: skip F/swap capture; seed dummy 241 frames;
  F cycles left→middle→right (shown on screen); C starts grasp with the selected cup.
  Server skips MEM and freezes that cup prompt (restart serve_shellgame_real_m6_cached.py
  so force_cup support is loaded).

Startup defaults to the Direct-collected init pose and converts it to ZMQ before moving.
After init move, a throw-away 242-frame request warms the policy server (JAX compile)
before F/history collection; use --skip-server-warmup to disable.

RTC (Black et al. 2025) overlaps inference with execution and soft-inpaints the next
chunk against the unexecuted leftover, so chunk boundaries stay continuous under delay.
'''

from __future__ import annotations

from multiprocessing.managers import SharedMemoryManager
import os
from pathlib import Path
import sys
import time

import click
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
OPENPI_CLIENT_SRC = ROOT_DIR / "packages" / "openpi-client" / "src"
if OPENPI_CLIENT_SRC.is_dir():
    sys.path.insert(0, str(OPENPI_CLIENT_SRC))
os.chdir(ROOT_DIR)

from utils.shellgame_openpi import ACTION_DIM  # noqa: E402
from utils.shellgame_openpi import HISTORY_FRAMES  # noqa: E402
from utils.shellgame_openpi import NUM_SWAPS  # noqa: E402
from utils.shellgame_openpi import SWAP_FRAMES  # noqa: E402
from utils.shellgame_openpi import TOTAL_MODEL_FRAMES  # noqa: E402
from utils.shellgame_openpi import ShellGameHistory  # noqa: E402
from utils.shellgame_openpi import ShellGamePolicyClient  # noqa: E402
from utils.shellgame_openpi import direct_link6_pose_to_zmq_eef  # noqa: E402
from utils.shellgame_openpi import eef_pose_to_link6  # noqa: E402
from utils.shellgame_openpi import link6_actions_to_eef  # noqa: E402
from utils.shellgame_openpi import parse_csv_vector  # noqa: E402
from utils.shellgame_openpi import relative_chunk_to_world  # noqa: E402
from utils.shellgame_openpi import relative_pose10  # noqa: E402
from utils.shellgame_openpi import run_contract_self_test  # noqa: E402
from utils.shellgame_openpi import select_safe_waypoint_count  # noqa: E402
from utils.shellgame_rtc import RTCInferenceWorker  # noqa: E402
from utils.shellgame_rtc import build_rtc_chunk  # noqa: E402

DEFAULT_EEF_OFFSET = 0.145
DEFAULT_ZMQ_GRIPPER_MIN = 0.0
DEFAULT_ZMQ_GRIPPER_MAX = 0.085
# Direct SingleArmClient full-open width (real_pos=5 → width 0.11 m).
DEFAULT_DIRECT_GRIPPER_FULL_OPEN = 0.11
# Direct-collected link6 init pose (requires --direct-zmq-origin or --calibrate-direct-origin).
DEFAULT_INIT_POSE = "-0.006958,-0.002573,0.120019,2.464006,-2.497232,1.524022"
DEFAULT_PROMPT = "The shell game has ended. Grasp and lift the cup containing the ball."
CUP_DISPLAY_NAMES = ("左杯/left", "中杯/middle", "右杯/right")
CUP_NAMES = ("left", "middle", "right")
N_CUPS = 3
DEFAULT_CUP_OVERLAY_NORM: tuple[tuple[float, float], ...] = (
    (0.25, 0.42),  # 0 left — same defaults as collect_drag_umi_cup
    (0.50, 0.38),  # 1 middle
    (0.75, 0.42),  # 2 right
)
CUP_OVERLAY_REF_SIZE = (224, 224)
DEFAULT_VIS_SCALE = 3.0
DEFAULT_PANEL_HEIGHT = 300


def cup_name(idx: int) -> str:
    if 0 <= idx < len(CUP_NAMES):
        return CUP_NAMES[idx]
    return str(idx)


def parse_cup_overlay_xy(text: str) -> list[tuple[float, float]]:
    """Parse 'x0,y0;x1,y1;x2,y2' (pixels at 224x224 reference)."""
    parts = [part.strip() for part in text.split(";") if part.strip()]
    if len(parts) != N_CUPS:
        raise click.BadParameter(
            f"--cup-overlay-xy needs {N_CUPS} points x,y;x,y;x,y, got {len(parts)}"
        )
    out: list[tuple[float, float]] = []
    for part in parts:
        values = [float(value) for value in part.split(",")]
        if len(values) != 2:
            raise click.BadParameter(f"--cup-overlay-xy each point needs x,y, got: {part!r}")
        out.append((values[0], values[1]))
    return out


def resolve_cup_overlay_positions(
    width: int,
    height: int,
    cup_overlay_xy: str | None,
) -> list[tuple[int, int]]:
    ref_w, ref_h = CUP_OVERLAY_REF_SIZE
    if cup_overlay_xy:
        ref_points = parse_cup_overlay_xy(cup_overlay_xy)
        return [
            (int(round(x * width / ref_w)), int(round(y * height / ref_h)))
            for x, y in ref_points
        ]
    return [
        (int(round(nx * width)), int(round(ny * height)))
        for nx, ny in DEFAULT_CUP_OVERLAY_NORM
    ]


def draw_cup_slot_markers(
    vis: np.ndarray,
    positions: list[tuple[int, int]],
    *,
    highlight_cups: set[int] | None = None,
) -> None:
    """Draw 0/1/2 slot markers on the preview (camera-image coordinates)."""
    highlight_cups = highlight_cups or set()
    radius = max(12, min(vis.shape[0], vis.shape[1]) // 12)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for cup_id, (x, y) in enumerate(positions):
        cx, cy = int(x), int(y)
        active = cup_id in highlight_cups
        ring = (0, 255, 255) if active else (255, 255, 255)
        fill = (0, 180, 255) if active else (40, 40, 40)
        cv2.circle(vis, (cx, cy), radius, ring, 2, lineType=cv2.LINE_AA)
        cv2.circle(vis, (cx, cy), max(radius - 3, 1), fill, -1, lineType=cv2.LINE_AA)
        label = str(cup_id)
        scale = max(0.55, radius / 22.0)
        thickness = max(1, int(round(scale * 2)))
        (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
        cv2.putText(
            vis,
            label,
            (cx - tw // 2, cy + th // 2),
            font,
            scale,
            (255, 255, 255),
            thickness,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            cup_name(cup_id)[0].upper(),
            (cx - 4, cy + radius + 14),
            font,
            0.35,
            ring,
            1,
            lineType=cv2.LINE_AA,
        )


def memory_highlight_cups(memory: dict | None) -> set[int]:
    if not memory or "predicted_final_cup" not in memory:
        return set()
    return {int(memory["predicted_final_cup"])}


def memory_panel_lines(memory: dict | None) -> list[str]:
    if not memory or "predicted_final_cup" not in memory:
        return ["MEM: waiting for prediction (auto-starts after capture / press C in manual)"]
    final_index = int(memory["predicted_final_cup"])
    probabilities = np.asarray(
        memory["predicted_final_cup_probabilities"],
        dtype=np.float64,
    )
    lines = [
        (
            f"FINAL PREDICT: cup {final_index} ({CUP_DISPLAY_NAMES[final_index]})  "
            f"conf={probabilities[final_index]:.3f}"
        ),
        "  " + ", ".join(
            f"{CUP_DISPLAY_NAMES[index]}={probability:.3f}"
            for index, probability in enumerate(probabilities)
        ),
    ]
    if "initial_cup" in memory:
        initial_index = int(memory["initial_cup"])
        lines.append(
            f"INIT PREDICT: cup {initial_index} ({CUP_DISPLAY_NAMES[initial_index]})"
        )
    if "stage_cups" in memory:
        stage_text = " -> ".join(
            f"{int(index)}({cup_name(int(index))[0].upper()})"
            for index in memory["stage_cups"]
        )
        lines.append(f"STAGES: {stage_text}")
    return lines


def print_memory_prediction(memory: dict, *, source: str) -> None:
    """Print model-native MEM classification without inferring it from actions."""
    final_index = int(memory["predicted_final_cup"])
    probabilities = np.asarray(
        memory["predicted_final_cup_probabilities"],
        dtype=np.float64,
    )
    probability_text = ", ".join(
        f"{CUP_DISPLAY_NAMES[index]}={probability:.3f}"
        for index, probability in enumerate(probabilities)
    )
    forced = bool(memory.get("forced_cup", False))
    tag = f"[MEM:{source}|forced]" if forced else f"[MEM:{source}]"
    print(
        f"{tag} 最终方块位置={CUP_DISPLAY_NAMES[final_index]} "
        f"confidence={probabilities[final_index]:.3f} | {probability_text}"
    )

    if forced:
        prompt = memory.get("direction_prompt")
        if prompt:
            print(f"{tag} direction_prompt={prompt!r}")
        return

    if "initial_cup" in memory:
        initial_index = int(memory["initial_cup"])
        initial_probabilities = np.asarray(memory["initial_cup_probabilities"], dtype=np.float64)
        print(
            f"{tag} 初始位置={CUP_DISPLAY_NAMES[initial_index]} "
            f"confidence={initial_probabilities[initial_index]:.3f}"
        )
    if "stage_cups" in memory:
        stage_text = " -> ".join(
            CUP_DISPLAY_NAMES[int(index)] for index in memory["stage_cups"]
        )
        print(f"{tag} 三阶段位置={stage_text}")
    if "swap_pair_names" in memory:
        print(f"{tag} 三次交换={memory['swap_pair_names']}")


def make_eval_vis(
    frame: np.ndarray,
    *,
    status: str,
    help_text: str,
    detail_lines: list[str] | None = None,
    memory: dict | None = None,
    cup_overlay_positions: list[tuple[int, int]] | None = None,
    vis_scale: float = DEFAULT_VIS_SCALE,
    panel_height: int = DEFAULT_PANEL_HEIGHT,
) -> np.ndarray:
    """Enlarged camera preview with cup markers + bottom status panel (collector style)."""
    cam = np.asarray(frame)
    if cam.dtype != np.uint8:
        if float(np.nanmax(cam)) <= 1.5:
            cam = (np.clip(cam, 0, 1) * 255).astype(np.uint8)
        else:
            cam = np.clip(cam, 0, 255).astype(np.uint8)
    cam = np.ascontiguousarray(cam)

    scale = max(float(vis_scale), 1.0)
    if abs(scale - 1.0) > 1e-6:
        cam = cv2.resize(cam, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    if cup_overlay_positions is not None:
        scaled_positions = [
            (int(round(x * scale)), int(round(y * scale)))
            for x, y in cup_overlay_positions
        ]
        draw_cup_slot_markers(
            cam,
            scaled_positions,
            highlight_cups=memory_highlight_cups(memory),
        )

    h, w = cam.shape[:2]
    panel_h = max(int(panel_height), 220)
    panel = np.full((panel_h, w, 3), 28, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    status_color = (0, 200, 255)
    if status.upper().startswith("POLICY") or status.upper().startswith("READY"):
        status_color = (80, 255, 120)
    elif status.upper().startswith("REC") or status.upper().startswith("AUTO"):
        status_color = (0, 0, 255)

    cv2.putText(panel, status, (14, 40), font, 1.15, status_color, 2, cv2.LINE_AA)
    cv2.putText(panel, help_text, (14, 78), font, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

    y = 118
    for line in memory_panel_lines(memory):
        if y + 28 >= panel_h - 8:
            break
        color = (0, 255, 255) if line.startswith("FINAL") else (80, 255, 120)
        cv2.putText(panel, line, (14, y), font, 0.82, color, 2, cv2.LINE_AA)
        y += 34

    if detail_lines:
        for line in detail_lines:
            if not line:
                continue
            if y + 26 >= panel_h - 8:
                break
            cv2.putText(panel, line, (14, y), font, 0.72, (200, 200, 200), 2, cv2.LINE_AA)
            y += 30

    return np.concatenate([cam, panel], axis=0)


def zmq_gripper_to_direct(
    width: float | np.ndarray,
    *,
    gripper_min: float,
    gripper_max: float,
    direct_full_open: float,
) -> np.ndarray:
    """Map ZMQ physical opening (m) into Direct training width (m)."""
    zmq = np.asarray(width, dtype=np.float64)
    zmq_span = gripper_max - gripper_min
    return (zmq - gripper_min) * float(direct_full_open) / zmq_span


def direct_gripper_to_zmq(
    width: float | np.ndarray,
    *,
    gripper_min: float,
    gripper_max: float,
    direct_full_open: float,
) -> np.ndarray:
    """Map Direct training width (m) into ZMQ command opening (m)."""
    direct = np.asarray(width, dtype=np.float64)
    zmq_span = gripper_max - gripper_min
    return gripper_min + direct * zmq_span / float(direct_full_open)


def ease_in_out_quad(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    return np.where(value < 0.5, 2.0 * value * value, 1.0 - ((-2.0 * value + 2.0) ** 2) / 2.0)


def current_robot_pose_and_gripper(env) -> tuple[np.ndarray, float]:
    state = env.get_robot_state()[0]
    pose = np.asarray(state["ActualTCPPose"], dtype=np.float64).reshape(6)
    gripper = float(state["gripper_position"])
    return pose, gripper


def hold_current_pose(env) -> None:
    """Cancel queued ZMQ waypoints by driving each controller to its measured pose."""
    states = env.get_robot_state()
    for robot, state in zip(env.robots, states, strict=True):
        pose = np.asarray(state["ActualTCPPose"], dtype=np.float64).reshape(6)
        gripper = float(state["gripper_position"])
        robot.servoL(
            pose=pose,
            gripper_pos=gripper,
            duration=max(0.1, 1.0 / float(robot.frequency)),
        )


def move_to_pose(
    env,
    target_pose6: np.ndarray,
    *,
    duration: float,
    frequency: float,
    target_gripper: float | None,
    gripper_min: float,
    gripper_max: float,
) -> None:
    """Move to a fixed pose using smooth translation and shortest-path rotation."""
    start_pose, start_gripper = current_robot_pose_and_gripper(env)
    target_pose = np.asarray(target_pose6, dtype=np.float64).reshape(6)
    target_gripper_value = (
        start_gripper if target_gripper is None else float(np.clip(target_gripper, gripper_min, gripper_max))
    )
    steps = max(2, round(duration * frequency))
    linear = np.linspace(0.0, 1.0, steps)
    blend = ease_in_out_quad(linear)

    positions = start_pose[:3] + blend[:, None] * (target_pose[:3] - start_pose[:3])
    rotation_keyframes = Rotation.from_rotvec(np.stack([start_pose[3:], target_pose[3:]], axis=0))
    rotations = Slerp([0.0, 1.0], rotation_keyframes)(blend).as_rotvec()
    grippers = start_gripper + blend * (target_gripper_value - start_gripper)

    actions = np.concatenate([positions, rotations, grippers[:, None]], axis=-1)
    start_time = time.time() + 0.25
    timestamps = start_time + np.arange(steps, dtype=np.float64) / frequency
    env.exec_actions(actions=actions, timestamps=timestamps, compensate_latency=False)
    time.sleep(max(0.0, timestamps[-1] - time.time()) + 0.3)


def _fmt_xyz(xyz: np.ndarray, precision: int = 4) -> str:
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1)
    return "[" + ", ".join(f"{float(v):+.{precision}f}" for v in xyz[:3]) + "]"


def _fmt_pose6(pose6: np.ndarray, precision: int = 4) -> str:
    pose6 = np.asarray(pose6, dtype=np.float64).reshape(6)
    return (
        f"xyz={_fmt_xyz(pose6[:3], precision)} "
        f"rot={_fmt_xyz(pose6[3:], precision)}"
    )


def rotation_error_deg(pose6_a: np.ndarray, pose6_b: np.ndarray) -> float:
    """Geodesic angle between two TCP rotvec poses (degrees)."""
    a = np.asarray(pose6_a, dtype=np.float64).reshape(6)
    b = np.asarray(pose6_b, dtype=np.float64).reshape(6)
    return float(
        np.rad2deg(
            (Rotation.from_rotvec(a[3:6]) * Rotation.from_rotvec(b[3:6]).inv()).magnitude()
        )
    )


def print_init_link6_alignment(
    label: str,
    target_link6_direct: np.ndarray,
    live_link6: np.ndarray,
    *,
    target_zmq_eef: np.ndarray | None = None,
    live_zmq_eef: np.ndarray | None = None,
    direct_origin3: np.ndarray | None = None,
    pos_tol_mm: float = 30.0,
    rot_tol_deg: float = 5.0,
) -> None:
    """Compare live pose against init (ZMQ command + Direct training link6 frame)."""
    target_direct = np.asarray(target_link6_direct, dtype=np.float64).reshape(6)
    live = np.asarray(live_link6, dtype=np.float64).reshape(6)

    target_link6_zmq = target_direct.copy()
    if direct_origin3 is not None:
        target_link6_zmq[:3] = target_direct[:3] + np.asarray(direct_origin3, dtype=np.float64).reshape(3)

    link6_pos_err_mm = float(np.linalg.norm(live[:3] - target_link6_zmq[:3]) * 1000.0)
    rot_err_deg = rotation_error_deg(target_direct, live)

    zmq_pos_err_mm: float | None = None
    if target_zmq_eef is not None and live_zmq_eef is not None:
        target_zmq = np.asarray(target_zmq_eef, dtype=np.float64).reshape(6)
        live_zmq = np.asarray(live_zmq_eef, dtype=np.float64).reshape(6)
        zmq_pos_err_mm = float(np.linalg.norm(live_zmq[:3] - target_zmq[:3]) * 1000.0)

    pos_err_mm = zmq_pos_err_mm if zmq_pos_err_mm is not None else link6_pos_err_mm
    status = "OK" if pos_err_mm <= pos_tol_mm and rot_err_deg <= rot_tol_deg else "MISALIGNED"
    zmq_note = f"ZMQ |Δxyz|={zmq_pos_err_mm:.1f} mm" if zmq_pos_err_mm is not None else "ZMQ n/a"
    origin_note = (
        f"origin={_fmt_xyz(direct_origin3)}"
        if direct_origin3 is not None
        else "origin=n/a"
    )
    print(
        f"[INIT:{label}] {status} {zmq_note}  link6|Δxyz|={link6_pos_err_mm:.1f} mm "
        f"|Δrot|={rot_err_deg:.2f} deg (tol {pos_tol_mm:.0f} mm / {rot_tol_deg:.1f} deg)  {origin_note}"
    )
    if target_zmq_eef is not None and live_zmq_eef is not None:
        print(
            f"[INIT:{label}] target ZMQ eef {_fmt_pose6(target_zmq)}  "
            f"live {_fmt_pose6(live_zmq)}"
        )
    print(
        f"[INIT:{label}] target link6 (direct+origin) {_fmt_pose6(target_link6_zmq)}  "
        f"live link6 {_fmt_pose6(live)}  "
        f"direct-only {_fmt_pose6(target_direct)}"
    )


def print_action_debug(
    *,
    label: str,
    raw_actions: np.ndarray,
    anchor_link6: np.ndarray,
    anchor_eef: np.ndarray,
    world_link6: np.ndarray,
    world_eef: np.ndarray,
    selected_eef: np.ndarray | None,
    selected_action_indices: np.ndarray | None,
    episode_start_link6: np.ndarray | None,
    state_rel_xyz: np.ndarray | None,
    gripper_width: float,
    gripper_width_zmq: float | None,
    eef_offset: float,
    gripper_scale: bool,
    preview: int = 4,
) -> None:
    """Print layered action diagnostics to separate policy vs transform vs command issues."""
    n = min(int(preview), len(raw_actions))
    raw = np.asarray(raw_actions, dtype=np.float64)
    w_link6 = np.asarray(world_link6, dtype=np.float64)
    w_eef = np.asarray(world_eef, dtype=np.float64)
    a_link6 = np.asarray(anchor_link6, dtype=np.float64).reshape(6)
    a_eef = np.asarray(anchor_eef, dtype=np.float64).reshape(6)

    rel_xyz = raw[:n, :3]
    d_link6_all = w_link6[:, :3] - a_link6[:3]
    d_eef_all = w_eef[:, :3] - a_eef[:3]
    d_link6 = d_link6_all[:n]
    d_eef = d_eef_all[:n]
    tcp_z_world = Rotation.from_rotvec(a_link6[3:6]).apply(
        np.array([0.0, 0.0, 1.0], dtype=np.float64)
    )
    offset_shift = tcp_z_world * float(eef_offset)

    # Round-trip sanity: identity relative action should recover the anchors.
    identity = np.zeros((len(raw), ACTION_DIM), dtype=np.float64)
    identity[:, 3] = 1.0
    identity[:, 7] = 1.0
    identity_world_link6 = relative_chunk_to_world(identity, a_link6)
    identity_world_eef = link6_actions_to_eef(identity_world_link6, eef_offset)
    id_err_link6 = float(np.linalg.norm(identity_world_link6[0, :3] - a_link6[:3]))
    id_err_eef = float(np.linalg.norm(identity_world_eef[0, :3] - a_eef[:3]))

    print(f"[DEBUG:{label}] ---------- action pipeline ----------")
    if episode_start_link6 is not None:
        print(f"[DEBUG:{label}] episode_start_link6 {_fmt_pose6(episode_start_link6)}")
    if state_rel_xyz is not None:
        grip_note = (
            f"gripper_direct={gripper_width:.4f} zmq={gripper_width_zmq:.4f}"
            if gripper_scale and gripper_width_zmq is not None
            else f"gripper={gripper_width:.4f}"
        )
        print(
            f"[DEBUG:{label}] state_sent_to_policy rel_xyz={_fmt_xyz(state_rel_xyz)} "
            f"{grip_note}"
        )
    print(f"[DEBUG:{label}] anchor_link6 {_fmt_pose6(a_link6)}")
    print(f"[DEBUG:{label}] anchor_eef   {_fmt_pose6(a_eef)}")
    print(
        f"[DEBUG:{label}] tcp_z_world(link6)={_fmt_xyz(tcp_z_world)} "
        f"eef_offset_shift={_fmt_xyz(offset_shift)} "
        f"(anchor_eef should ≈ anchor_link6 + shift)"
    )
    print(
        f"[DEBUG:{label}] identity round-trip err: "
        f"link6={id_err_link6:.6f}m eef={id_err_eef:.6f}m"
    )
    print(
        f"[DEBUG:{label}] raw_rel_xyz[0:{n}] (policy TCP-relative translation): "
        + " | ".join(_fmt_xyz(rel_xyz[i]) for i in range(n))
    )
    print(
        f"[DEBUG:{label}] raw_rel_xyz span over horizon: "
        f"min={_fmt_xyz(raw[:, :3].min(axis=0))} max={_fmt_xyz(raw[:, :3].max(axis=0))} "
        f"last={_fmt_xyz(raw[-1, :3])} grip[0]={raw[0, 9]:.4f}→[{raw[-1, 9]:.4f}]"
    )
    print(
        f"[DEBUG:{label}] world_link6 Δxyz vs anchor[0:{n}]: "
        + " | ".join(_fmt_xyz(d_link6[i]) for i in range(n))
    )
    print(
        f"[DEBUG:{label}] world_eef   Δxyz vs anchor[0:{n}]: "
        + " | ".join(_fmt_xyz(d_eef[i]) for i in range(n))
    )
    # Dominant motion axis of first commanded eef delta — "drop" usually means large -Z.
    first_deef = d_eef[0]
    dominant = int(np.argmax(np.abs(first_deef)))
    axis_name = "XYZ"[dominant]
    print(
        f"[DEBUG:{label}] first eef Δ={_fmt_xyz(first_deef)} "
        f"|Δ|={float(np.linalg.norm(first_deef)):.4f}m "
        f"dominant={axis_name}{'↓' if first_deef[dominant] < 0 else '↑'} "
        f"Δz={first_deef[2]:+.4f}m"
    )

    selected_indices = set()
    if selected_action_indices is not None:
        selected_indices = {
            int(index) for index in np.asarray(selected_action_indices).reshape(-1)
        }
    step_d_eef = np.diff(
        np.concatenate([a_eef[None, :3], w_eef[:, :3]], axis=0),
        axis=0,
    )
    print(
        f"[TRACE:{label}] full horizon ({len(raw)} rows): "
        "idx mode raw_rel_mm world_delta_mm step_delta_mm abs_z "
        "grip_policy grip_cmd"
    )
    for index in range(len(raw)):
        if selected_action_indices is None:
            mode = "PRED"
        else:
            mode = "EXEC" if index in selected_indices else "SKIP"
        print(
            f"[TRACE:{label}] {index:02d} {mode} "
            f"raw={_fmt_xyz(raw[index, :3] * 1000.0, 1)}mm "
            f"worldΔ={_fmt_xyz(d_eef_all[index] * 1000.0, 1)}mm "
            f"stepΔ={_fmt_xyz(step_d_eef[index] * 1000.0, 1)}mm "
            f"z={w_eef[index, 2]:+.4f}m "
            f"grip_policy={raw[index, 9]:.4f} "
            f"grip_cmd={w_eef[index, 6]:.4f}"
        )

    def first_index(mask: np.ndarray) -> int | None:
        matches = np.flatnonzero(mask)
        return None if not len(matches) else int(matches[0])

    min_grip_index = int(np.argmin(raw[:, 9]))
    close_crossings = ", ".join(
        f"≤{threshold:.3f}:{first_index(raw[:, 9] <= threshold)}"
        for threshold in (0.095, 0.090, 0.085, 0.080)
    )
    lift_crossings = ", ".join(
        f"+{threshold_mm:.0f}mm:{first_index(d_eef_all[:, 2] >= threshold_mm / 1000.0)}"
        for threshold_mm in (10.0, 20.0, 40.0)
    )
    print(
        f"[TRACE:{label}] horizon summary: world_z_delta "
        f"min={d_eef_all[:, 2].min() * 1000.0:+.1f}mm "
        f"max={d_eef_all[:, 2].max() * 1000.0:+.1f}mm "
        f"last={d_eef_all[-1, 2] * 1000.0:+.1f}mm; "
        f"lift crossings [{lift_crossings}]"
    )
    print(
        f"[TRACE:{label}] gripper summary: current_policy={gripper_width:.4f}, "
        f"pred_min={raw[min_grip_index, 9]:.4f}@{min_grip_index}, "
        f"pred_last={raw[-1, 9]:.4f}; crossings [{close_crossings}]"
    )
    if selected_action_indices is not None:
        executed_text = (
            ",".join(str(index) for index in sorted(selected_indices))
            if selected_indices
            else "none"
        )
        first_lift = first_index(d_eef_all[:, 2] >= 0.010)
        first_close = first_index(raw[:, 9] <= 0.090)
        print(
            f"[TRACE:{label}] executed action indices={executed_text}; "
            f"first_10mm_lift={'none' if first_lift is None else first_lift} "
            f"({'EXEC' if first_lift in selected_indices else 'SKIP/NONE'}), "
            f"first_grip≤0.090={'none' if first_close is None else first_close} "
            f"({'EXEC' if first_close in selected_indices else 'SKIP/NONE'})"
        )
    if selected_eef is not None and len(selected_eef):
        sel = np.asarray(selected_eef, dtype=np.float64)
        sel_d = sel[:, :3] - a_eef[:3]
        print(
            f"[DEBUG:{label}] submitted {len(sel)} eef waypoints Δxyz: "
            + " | ".join(_fmt_xyz(sel_d[i]) for i in range(min(n, len(sel))))
        )
        print(
            f"[DEBUG:{label}] submitted eef[0] xyz={_fmt_xyz(sel[0, :3])} "
            f"grip={float(sel[0, 6]):.4f}"
        )
    print(f"[DEBUG:{label}] ----------------------------------------")
    sys.stdout.flush()


def warmup_policy_server(
    *,
    policy_client: ShellGamePolicyClient,
    env,
    wrist_camera_index: int,
    prompt: str,
    current_link6_pose: np.ndarray,
    gripper_scale: bool,
    gripper_min: float,
    gripper_max: float,
    direct_gripper_full_open: float,
) -> float:
    """Warm the policy server before grasp control (cache+step or full infer)."""
    camera_key = f"camera{wrist_camera_index}_rgb"
    obs = env.get_obs()
    frame = obs[camera_key][-1]
    gripper_width_zmq = float(np.asarray(obs["robot0_gripper_width"][-1]).reshape(-1)[0])
    if gripper_scale:
        gripper_width = float(
            zmq_gripper_to_direct(
                gripper_width_zmq,
                gripper_min=gripper_min,
                gripper_max=gripper_max,
                direct_full_open=direct_gripper_full_open,
            )
        )
    else:
        gripper_width = gripper_width_zmq

    warmup_history = ShellGameHistory()
    warmup_history.seed_dummy_history(frame, current_link6_pose)
    started = time.monotonic()
    if policy_client.cached_infer_enabled:
        print(
            f"[warmup] uploading {HISTORY_FRAMES} history frames + one step infer "
            "(first run may take 1-3 min while the model compiles; arm will not move)..."
        )
        sys.stdout.flush()
        policy_client.upload_history(warmup_history, prompt=prompt)
        result = policy_client.infer_step(
            warmup_history,
            current_frame=frame,
            current_pose6=current_link6_pose,
            gripper_width=gripper_width,
            prompt=prompt,
        )
    else:
        print(
            f"[warmup] sending {TOTAL_MODEL_FRAMES} frames to policy server "
            "(first run may take 1-3 min while the model compiles; arm will not move)..."
        )
        sys.stdout.flush()
        result = policy_client.infer_full(
            warmup_history,
            current_frame=frame,
            current_pose6=current_link6_pose,
            gripper_width=gripper_width,
            prompt=prompt,
        )
    latency = time.monotonic() - started
    raw_actions = np.asarray(result["actions"], dtype=np.float64)
    if raw_actions.ndim != 2 or raw_actions.shape[1] != ACTION_DIM or len(raw_actions) not in (16, 32):
        raise ValueError(f"Warmup returned unsupported action shape {raw_actions.shape}; expected (16|32, 10)")
    print(f"[warmup] policy server ready in {latency:.1f}s")
    sys.stdout.flush()
    policy_client.reset_session()
    return latency


def run_policy(
    *,
    env,
    policy_client: ShellGamePolicyClient,
    history: ShellGameHistory,
    wrist_camera_index: int,
    prompt: str,
    frequency: float,
    steps_per_inference: int,
    action_lead_time: float,
    max_duration: float,
    max_inference_latency: float,
    max_pos_speed: float,
    max_rot_speed: float,
    workspace_min: np.ndarray,
    workspace_max: np.ndarray,
    gripper_min: float,
    gripper_max: float,
    eef_offset: float,
    gripper_scale: bool,
    direct_gripper_full_open: float,
    key_counter,
    observe_only: bool,
    warmup_done: bool,
    vis_scale: float = DEFAULT_VIS_SCALE,
    panel_height: int = DEFAULT_PANEL_HEIGHT,
    cup_overlay_positions: list[tuple[int, int]] | None = None,
    rtc: bool = False,
    rtc_s_min: int = 8,
    rtc_delay_buffer_size: int = 8,
    rtc_initial_delay_steps: int = 4,
    rtc_max_guidance_weight: float = 5.0,
    force_cup: str | None = None,
) -> tuple[bool, bool, dict | None]:
    """Run one policy episode. Returns (quit_requested, warmup_done, last_memory)."""
    from peripherals.keystroke_counter import KeyCode

    if not history.ready:
        print("History is incomplete; refusing to start policy")
        return False, warmup_done, None

    camera_key = f"camera{wrist_camera_index}_rgb"
    last_infer_frame: np.ndarray | None = None
    last_memory: dict | None = None

    def show_policy_view(
        frame: np.ndarray,
        *,
        status: str,
        help_text: str,
        detail_lines: list[str] | None = None,
    ) -> None:
        view = make_eval_vis(
            frame,
            status=status,
            help_text=help_text,
            detail_lines=detail_lines,
            memory=last_memory,
            cup_overlay_positions=cup_overlay_positions,
            vis_scale=vis_scale,
            panel_height=panel_height,
        )
        cv2.imshow("ARX5 ShellGame", view[..., ::-1])
        cv2.pollKey()

    def latest_obs_timestamp(obs: dict) -> float | None:
        if "timestamp" not in obs:
            return None
        values = np.asarray(obs["timestamp"]).reshape(-1)
        if not len(values):
            return None
        value = float(values[-1])
        return value if np.isfinite(value) else None

    def _gripper_to_policy(gripper_width_zmq: float) -> float:
        if gripper_scale:
            return float(
                zmq_gripper_to_direct(
                    gripper_width_zmq,
                    gripper_min=gripper_min,
                    gripper_max=gripper_max,
                    direct_full_open=direct_gripper_full_open,
                )
            )
        return float(gripper_width_zmq)

    def _world_eef_from_link6(world_link6: np.ndarray) -> np.ndarray:
        world_actions = link6_actions_to_eef(world_link6, eef_offset)
        if gripper_scale:
            world_actions[:, 6] = direct_gripper_to_zmq(
                world_actions[:, 6],
                gripper_min=gripper_min,
                gripper_max=gripper_max,
                direct_full_open=direct_gripper_full_open,
            )
        world_actions[:, 6] = np.clip(world_actions[:, 6], gripper_min, gripper_max)
        return world_actions

    def infer_once(
        *,
        label: str = "infer",
        rtc_prev_chunk: np.ndarray | None = None,
        rtc_inference_delay: int = 0,
        rtc_execution_horizon: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict, float, float, float]:
        nonlocal last_infer_frame, last_memory
        obs = env.get_obs()
        obs_received_wall = time.time()
        frame = np.asarray(obs[camera_key][-1])
        obs_timestamp = latest_obs_timestamp(obs)
        obs_age_before_ms = (
            None if obs_timestamp is None else (obs_received_wall - obs_timestamp) * 1000.0
        )
        frame_float = frame.astype(np.float32, copy=False)
        if frame_float.ndim == 3 and frame_float.shape[-1] >= 3:
            frame_mean = frame_float[..., :3].mean(axis=(0, 1))
            frame_std = float(frame_float[..., :3].std())
        else:
            frame_mean = np.array([frame_float.mean()] * 3, dtype=np.float32)
            frame_std = float(frame_float.std())
        frame_delta = (
            None
            if last_infer_frame is None or last_infer_frame.shape != frame.shape
            else float(
                np.mean(
                    np.abs(
                        frame_float
                        - last_infer_frame.astype(np.float32, copy=False)
                    )
                )
            )
        )
        last_infer_frame = frame.copy()
        anchor_eef = np.concatenate([obs["robot0_eef_pos"][-1], obs["robot0_eef_rot_axis_angle"][-1]], axis=-1).astype(
            np.float64
        )
        anchor_link6 = eef_pose_to_link6(anchor_eef, eef_offset)
        gripper_width_zmq = float(np.asarray(obs["robot0_gripper_width"][-1]).reshape(-1)[0])
        gripper_width = _gripper_to_policy(gripper_width_zmq)
        if policy_client.cached_infer_enabled:
            print(f"[policy] {label}: sending 1 current frame (cached history on server)")
        else:
            print(
                f"[policy] {label}: sending {TOTAL_MODEL_FRAMES} frames to policy server "
                f"(first call may take 1-3 min while the model compiles)..."
            )
        sys.stdout.flush()
        started = time.monotonic()
        result = policy_client.infer_step(
            history,
            current_frame=frame,
            current_pose6=anchor_link6,
            gripper_width=gripper_width,
            prompt=prompt,
            rtc_prev_chunk=rtc_prev_chunk,
            rtc_inference_delay=rtc_inference_delay,
            rtc_execution_horizon=rtc_execution_horizon,
            rtc_max_guidance_weight=rtc_max_guidance_weight,
        )
        latency = time.monotonic() - started
        obs_age_after_ms = (
            None if obs_timestamp is None else (time.time() - obs_timestamp) * 1000.0
        )
        print(
            f"[TRACE:{label}] observation timestamp="
            f"{'n/a' if obs_timestamp is None else f'{obs_timestamp:.6f}'}, "
            f"age_before_infer="
            f"{'n/a' if obs_age_before_ms is None else f'{obs_age_before_ms:.1f}ms'}, "
            f"age_after_infer="
            f"{'n/a' if obs_age_after_ms is None else f'{obs_age_after_ms:.1f}ms'}, "
            f"frame_mean={_fmt_xyz(frame_mean, 1)}, frame_std={frame_std:.1f}, "
            f"frame_MAD_from_previous="
            f"{'n/a' if frame_delta is None else f'{frame_delta:.2f}'}"
        )
        raw_actions = np.asarray(result["actions"], dtype=np.float64)
        if raw_actions.ndim != 2 or raw_actions.shape[1] != ACTION_DIM or len(raw_actions) not in (16, 32):
            raise ValueError(f"Policy returned unsupported action shape {raw_actions.shape}; expected (16|32, 10)")
        if "memory" in result:
            last_memory = result["memory"]
            print_memory_prediction(result["memory"], source=label)
        return raw_actions, anchor_link6, anchor_eef, obs, latency, gripper_width, gripper_width_zmq

    if policy_client.cached_infer_enabled:
        if force_cup is not None:
            print(
                f"[cache] uploading {HISTORY_FRAMES} history frames "
                f"(force_cup={force_cup}, skip MEM)..."
            )
        else:
            print(f"[cache] uploading {HISTORY_FRAMES} history frames to policy server...")
        sys.stdout.flush()
        cache_started = time.monotonic()
        cache_result = policy_client.upload_history(
            history, prompt=prompt, force_cup=force_cup
        )
        cache_latency = time.monotonic() - cache_started
        print(f"[cache] history cached in {cache_latency:.3f}s")
        if "memory" in cache_result:
            last_memory = cache_result["memory"]
            print_memory_prediction(cache_result["memory"], source="history-upload")
            live_obs = env.get_obs()
            show_policy_view(
                live_obs[camera_key][-1],
                status="READY MEM" if force_cup is None else f"FORCE {force_cup.upper()}",
                help_text="History uploaded; starting policy  |  S: stop  Q: quit",
                detail_lines=[f"cache upload {cache_latency:.3f}s"],
            )

    if not warmup_done and not observe_only:
        print("Compiling/warming policy with a throw-away inference; no action will be sent...")
        _, _, _, _, warmup_latency, _, _ = infer_once(label="warmup")
        warmup_done = True
        print(f"Policy warm-up complete in {warmup_latency:.3f}s; taking a fresh observation")

    if observe_only:
        actions, anchor_link6, anchor_eef, _, latency, gripper_width, gripper_width_zmq = infer_once(
            label="observe-only"
        )
        world_link6 = relative_chunk_to_world(actions, anchor_link6)
        world_eef = link6_actions_to_eef(world_link6, eef_offset)
        if gripper_scale:
            world_eef[:, 6] = direct_gripper_to_zmq(
                world_eef[:, 6],
                gripper_min=gripper_min,
                gripper_max=gripper_max,
                direct_full_open=direct_gripper_full_open,
            )
        world_eef[:, 6] = np.clip(world_eef[:, 6], gripper_min, gripper_max)
        state_rel = None
        if history.episode_start_pose6 is not None:
            state_rel = relative_pose10(history.episode_start_pose6, anchor_link6)[:3]
        print(f"[observe-only] inference={latency:.3f}s, raw={actions.shape}")
        print_action_debug(
            label="observe-only",
            raw_actions=actions,
            anchor_link6=anchor_link6,
            anchor_eef=anchor_eef,
            world_link6=world_link6,
            world_eef=world_eef,
            selected_eef=None,
            selected_action_indices=None,
            episode_start_link6=history.episode_start_pose6,
            state_rel_xyz=state_rel,
            gripper_width=gripper_width,
            gripper_width_zmq=gripper_width_zmq,
            eef_offset=eef_offset,
            gripper_scale=gripper_scale,
        )
        return False, warmup_done, last_memory

    if rtc:
        if not policy_client.rtc_enabled:
            print("[RTC] WARNING: server metadata lacks supports_rtc; continuing anyway")
        print(
            f"[RTC] enabled: s_min={rtc_s_min}, delay_buf={rtc_delay_buffer_size}, "
            f"d_init={rtc_initial_delay_steps}, max_gw={rtc_max_guidance_weight}"
        )
        episode_started = False
        quit_requested = False
        episode_start_time = time.monotonic()
        worker: RTCInferenceWorker | None = None

        def _get_obs_bundle() -> dict:
            obs = env.get_obs()
            frame = np.asarray(obs[camera_key][-1])
            anchor_eef = np.concatenate(
                [obs["robot0_eef_pos"][-1], obs["robot0_eef_rot_axis_angle"][-1]],
                axis=-1,
            ).astype(np.float64)
            anchor_link6 = eef_pose_to_link6(anchor_eef, eef_offset)
            gripper_width_zmq = float(np.asarray(obs["robot0_gripper_width"][-1]).reshape(-1)[0])
            return {
                "obs": obs,
                "frame": frame,
                "anchor_eef": anchor_eef,
                "anchor_link6": anchor_link6,
                "gripper_width": _gripper_to_policy(gripper_width_zmq),
                "gripper_width_zmq": gripper_width_zmq,
            }

        def _rtc_infer_fn(*, obs_bundle, rtc_prev_chunk, rtc_inference_delay, rtc_execution_horizon):
            started = time.monotonic()
            result = policy_client.infer_step(
                history,
                current_frame=obs_bundle["frame"],
                current_pose6=obs_bundle["anchor_link6"],
                gripper_width=obs_bundle["gripper_width"],
                prompt=prompt,
                rtc_prev_chunk=rtc_prev_chunk,
                rtc_inference_delay=rtc_inference_delay,
                rtc_execution_horizon=rtc_execution_horizon,
                rtc_max_guidance_weight=rtc_max_guidance_weight,
            )
            latency = time.monotonic() - started
            actions = np.asarray(result["actions"], dtype=np.float64)
            if "memory" in result:
                nonlocal last_memory
                last_memory = result["memory"]
            return actions, obs_bundle["anchor_link6"], latency

        try:
            actions, anchor_link6, anchor_eef, obs, latency, gripper_width, gripper_width_zmq = (
                infer_once(label="rtc-init")
            )
            if latency > max_inference_latency:
                raise RuntimeError(
                    f"Inference latency {latency:.3f}s exceeds limit {max_inference_latency:.3f}s"
                )
            current = build_rtc_chunk(actions, anchor_link6)
            cursor = 0
            worker = RTCInferenceWorker(
                _rtc_infer_fn,
                frequency=frequency,
                s_min=rtc_s_min,
                delay_buffer_size=rtc_delay_buffer_size,
                initial_delay_steps=max(rtc_initial_delay_steps, int(np.ceil(latency * frequency))),
            )
            print(f"[RTC] initial chunk H={len(current.relative_actions)}, latency={latency:.3f}s")

            while True:
                for key in key_counter.get_press_events():
                    if key == KeyCode(char="s"):
                        print("S pressed: stopping policy")
                        return False, warmup_done, last_memory
                    if key == KeyCode(char="q"):
                        quit_requested = True
                        return True, warmup_done, last_memory

                polled = worker.poll()
                if polled is not None:
                    current, observed_delay, s_used = polled
                    cursor = 0
                    print(
                        f"[RTC] swapped chunk: delay={observed_delay}, s={s_used}, "
                        f"remaining={len(current.relative_actions)}, "
                        f"latency={worker.last_latency_s:.3f}s"
                    )

                if cursor >= len(current.world_link6):
                    print("[RTC] exhausted chunk; forcing replan")
                    worker.maybe_start(
                        executed_since_chunk=max(cursor, worker.s_min),
                        current_chunk=current,
                        get_obs_bundle=_get_obs_bundle,
                    )
                    time.sleep(0.01)
                    continue

                live = env.get_obs()
                live_eef = np.concatenate(
                    [live["robot0_eef_pos"][-1], live["robot0_eef_rot_axis_angle"][-1]],
                    axis=-1,
                ).astype(np.float64)
                world_eef = _world_eef_from_link6(current.world_link6[cursor : cursor + 1])
                safe_count, safety_reason = select_safe_waypoint_count(
                    world_eef,
                    anchor_pose6=live_eef,
                    dt=1.0 / frequency,
                    max_pos_speed=max_pos_speed,
                    max_rot_speed=max_rot_speed,
                    workspace_min=workspace_min,
                    workspace_max=workspace_max,
                    max_waypoints=1,
                )
                if safe_count == 0:
                    print(f"[SAFETY][RTC] rejected waypoint: {safety_reason}")
                    time.sleep(1.0 / frequency)
                    continue

                first_timestamp = time.time() + action_lead_time
                if not episode_started:
                    env.start_episode(time.time())
                    episode_started = True
                env.exec_actions(
                    actions=world_eef[:1],
                    timestamps=np.asarray([first_timestamp], dtype=np.float64),
                    compensate_latency=False,
                    dynamic_latency=False,
                )
                cursor += 1
                worker.maybe_start(
                    executed_since_chunk=cursor,
                    current_chunk=current,
                    get_obs_bundle=_get_obs_bundle,
                )

                show_policy_view(
                    live[camera_key][-1],
                    status="RTC CONTROL",
                    help_text="S: stop  Q: quit",
                    detail_lines=[
                        f"cursor {cursor}/{len(current.relative_actions)}, "
                        f"delay~{worker.delay_estimate}, "
                        f"last_infer {worker.last_latency_s:.3f}s",
                    ],
                )

                deadline = first_timestamp + 0.5 / frequency
                while time.time() < deadline:
                    for key in key_counter.get_press_events():
                        if key == KeyCode(char="s"):
                            print("S pressed: stopping policy")
                            return False, warmup_done, last_memory
                        if key == KeyCode(char="q"):
                            quit_requested = True
                            return True, warmup_done, last_memory
                    time.sleep(0.005)

                if time.monotonic() - episode_start_time >= max_duration:
                    print("Maximum policy duration reached")
                    break
        finally:
            # Wait for in-flight RTC infer so reset_session does not race ws.recv().
            if worker is not None:
                worker.close(wait=True, timeout_s=max(5.0, max_inference_latency + 1.0))
            policy_client.reset_session()
            hold_current_pose(env)
            time.sleep(0.15)
            if episode_started:
                env.end_episode()
            print("Policy stopped; human control restored. Restore the cups, then press P.")
        return quit_requested, warmup_done, last_memory

    episode_started = False
    quit_requested = False
    episode_start_time = time.monotonic()
    infer_step = 0
    previous_command: dict[str, object] | None = None
    try:
        while True:
            for key in key_counter.get_press_events():
                if key == KeyCode(char="s"):
                    print("S pressed: stopping policy")
                    return False, warmup_done, last_memory
                if key == KeyCode(char="q"):
                    quit_requested = True
                    return True, warmup_done, last_memory

            actions, anchor_link6, anchor_eef, obs, inference_latency, gripper_width, gripper_width_zmq = (
                infer_once(label="policy")
            )
            infer_step += 1
            print(f"Inference latency: {inference_latency:.3f}s")
            if inference_latency > max_inference_latency:
                raise RuntimeError(
                    f"Inference latency {inference_latency:.3f}s exceeds limit {max_inference_latency:.3f}s"
                )

            if previous_command is not None:
                previous_anchor = np.asarray(previous_command["anchor_eef"], dtype=np.float64)
                previous_target = np.asarray(previous_command["target_eef"], dtype=np.float64)
                previous_deadline = float(previous_command["deadline"])
                obs_timestamp = latest_obs_timestamp(obs)
                feedback_time = time.time() if obs_timestamp is None else obs_timestamp
                intended_delta = previous_target[:3] - previous_anchor[:3]
                observed_delta = anchor_eef[:3] - previous_anchor[:3]
                target_error = anchor_eef[:3] - previous_target[:3]
                print(
                    f"[TRACE:step{infer_step}] previous chunk feedback "
                    f"(observation-aligned): obs_vs_deadline="
                    f"{(feedback_time - previous_deadline) * 1000.0:+.1f}ms, "
                    f"intendedΔ={_fmt_xyz(intended_delta * 1000.0, 1)}mm, "
                    f"observedΔ={_fmt_xyz(observed_delta * 1000.0, 1)}mm, "
                    f"target_error={_fmt_xyz(target_error * 1000.0, 1)}mm "
                    f"|err|={np.linalg.norm(target_error) * 1000.0:.1f}mm, "
                    f"rot_err={rotation_error_deg(anchor_eef, previous_target[:6]):.2f}deg"
                )
                print(
                    f"[TRACE:step{infer_step}] previous gripper feedback: "
                    f"target_cmd={previous_target[6]:.4f}, "
                    f"observed_zmq={gripper_width_zmq:.4f}, "
                    f"error={gripper_width_zmq - previous_target[6]:+.4f}m"
                )

            world_link6 = relative_chunk_to_world(actions, anchor_link6)
            world_actions = link6_actions_to_eef(world_link6, eef_offset)
            if gripper_scale:
                world_actions[:, 6] = direct_gripper_to_zmq(
                    world_actions[:, 6],
                    gripper_min=gripper_min,
                    gripper_max=gripper_max,
                    direct_full_open=direct_gripper_full_open,
                )
            world_actions[:, 6] = np.clip(world_actions[:, 6], gripper_min, gripper_max)

            count = min(steps_per_inference, len(actions))
            safe_count, safety_reason = select_safe_waypoint_count(
                world_actions,
                anchor_pose6=anchor_eef,
                dt=1.0 / frequency,
                max_pos_speed=max_pos_speed,
                max_rot_speed=max_rot_speed,
                workspace_min=workspace_min,
                workspace_max=workspace_max,
                max_waypoints=count,
            )

            state_rel = None
            if history.episode_start_pose6 is not None:
                state_rel = relative_pose10(history.episode_start_pose6, anchor_link6)[:3]

            if safe_count == 0:
                print(f"[SAFETY] rejected chunk, holding pose: {safety_reason}")
                print_action_debug(
                    label=f"step{infer_step}-REJECTED",
                    raw_actions=actions,
                    anchor_link6=anchor_link6,
                    anchor_eef=anchor_eef,
                    world_link6=world_link6,
                    world_eef=world_actions,
                    selected_eef=None,
                    selected_action_indices=None,
                    episode_start_link6=history.episode_start_pose6,
                    state_rel_xyz=state_rel,
                    gripper_width=gripper_width,
                    gripper_width_zmq=gripper_width_zmq,
                    eef_offset=eef_offset,
                    gripper_scale=gripper_scale,
                )
                continue
            if safety_reason is not None:
                print(
                    f"[SAFETY] truncated {count}->{safe_count} waypoints: {safety_reason}"
                )
            selected = world_actions[:safe_count]
            selected_action_indices = np.arange(safe_count, dtype=np.int64)
            first_timestamp = time.time() + action_lead_time
            timestamps = first_timestamp + np.arange(safe_count, dtype=np.float64) / frequency
            fresh = timestamps > time.time()
            selected = selected[fresh]
            selected_action_indices = selected_action_indices[fresh]
            timestamps = timestamps[fresh]
            if not len(selected):
                raise RuntimeError("All action waypoints became stale before submission")

            print_action_debug(
                label=f"step{infer_step}",
                raw_actions=actions,
                anchor_link6=anchor_link6,
                anchor_eef=anchor_eef,
                world_link6=world_link6,
                world_eef=world_actions,
                selected_eef=selected,
                selected_action_indices=selected_action_indices,
                episode_start_link6=history.episode_start_pose6,
                state_rel_xyz=state_rel,
                gripper_width=gripper_width,
                gripper_width_zmq=gripper_width_zmq,
                eef_offset=eef_offset,
                gripper_scale=gripper_scale,
            )

            submit_wall = time.time()
            print(
                f"[TRACE:step{infer_step}] schedule: action_indices="
                f"{','.join(str(int(index)) for index in selected_action_indices)}, "
                f"submit_to_first={(timestamps[0] - submit_wall) * 1000.0:+.1f}ms, "
                f"submit_to_last={(timestamps[-1] - submit_wall) * 1000.0:+.1f}ms, "
                f"waypoint_span={(timestamps[-1] - timestamps[0]) * 1000.0:.1f}ms; "
                "the next inference starts after the last submitted deadline"
            )

            if not episode_started:
                env.start_episode(time.time())
                episode_started = True
            env.exec_actions(
                actions=selected,
                timestamps=timestamps,
                compensate_latency=False,
                dynamic_latency=False,
            )
            print(f"Submitted {len(selected)} safe waypoints")
            previous_command = {
                "anchor_eef": anchor_eef.copy(),
                "target_eef": selected[-1].copy(),
                "deadline": float(timestamps[-1]),
            }

            show_policy_view(
                obs[f"camera{wrist_camera_index}_rgb"][-1],
                status="POLICY CONTROL",
                help_text="S: stop  Q: quit",
                detail_lines=[
                    f"inference {inference_latency:.3f}s, execute {safe_count}/{len(actions)}",
                ],
            )

            deadline = timestamps[-1]
            while time.time() < deadline:
                for key in key_counter.get_press_events():
                    if key == KeyCode(char="s"):
                        print("S pressed: stopping policy")
                        return False, warmup_done, last_memory
                    if key == KeyCode(char="q"):
                        quit_requested = True
                        return True, warmup_done, last_memory
                time.sleep(0.01)

            if time.monotonic() - episode_start_time >= max_duration:
                print("Maximum policy duration reached")
                break
    finally:
        policy_client.reset_session()
        # A fresh servoL target truncates the controller's queued
        # interpolation, so S does not continue executing the old chunk.
        hold_current_pose(env)
        time.sleep(0.15)
        if episode_started:
            env.end_episode()
        print("Policy stopped; human control restored. Restore the cups, then press P.")
    return quit_requested, warmup_done, last_memory


@click.command()
@click.option("--output", "-o", default="data_local/shellgame_eval", show_default=True)
@click.option("--policy-ip", default="127.0.0.1", show_default=True)
@click.option("--policy-port", default=8000, show_default=True, type=int)
@click.option("--robot-ip", default="127.0.0.1", show_default=True)
@click.option("--robot-port", default=8765, show_default=True, type=int)
@click.option("--camera-reorder", default="0", show_default=True)
@click.option("--wrist-camera-index", default=0, show_default=True, type=int)
@click.option("--frequency", default=20.0, show_default=True, type=float)
@click.option("--steps-per-inference", default=8, show_default=True, type=click.IntRange(1, 32))
@click.option("--action-lead-time", default=0.10, show_default=True, type=float)
@click.option("--max-duration", default=30.0, show_default=True, type=float)
@click.option("--max-inference-latency", default=3.0, show_default=True, type=float)
@click.option(
    "--rtc/--no-rtc",
    default=False,
    show_default=True,
    help="Enable Real-Time Chunking (async infer + soft prefix inpainting).",
)
@click.option("--rtc-s-min", default=8, show_default=True, type=click.IntRange(1, 32))
@click.option("--rtc-delay-buffer-size", default=8, show_default=True, type=click.IntRange(1, 64))
@click.option("--rtc-initial-delay-steps", default=4, show_default=True, type=click.IntRange(1, 32))
@click.option("--rtc-max-guidance-weight", default=5.0, show_default=True, type=float)
@click.option(
    "--image-key-style",
    type=click.Choice(["m6", "legacy"], case_sensitive=False),
    default="m6",
    show_default=True,
    help="Cached image key naming: m6=left_wrist_0_rgb_0_*, legacy=left_wrist_0_rgb_*.",
)
@click.option("--prompt", default=DEFAULT_PROMPT, show_default=True)
@click.option(
    "--init-pose",
    default=DEFAULT_INIT_POSE,
    show_default=True,
    type=str,
    help=(
        "启动后缓动到的初始 TCP 位姿 x,y,z,rx,ry,rz。"
        "默认是 Direct link6 位姿，需配合 --init-pose-frame=direct "
        "与 --direct-zmq-origin 或 --calibrate-direct-origin。"
    ),
)
@click.option(
    "--init-pose-frame",
    type=click.Choice(["zmq", "direct"], case_sensitive=False),
    default="direct",
    show_default=True,
    help="init-pose 坐标系: zmq=eef 绝对位姿; direct=Direct 采集 link6 位姿。",
)
@click.option(
    "--direct-zmq-origin",
    default=None,
    type=str,
    help=(
        "Direct→ZMQ 的 link6 原点平移 x,y,z (米)。"
        "init-pose-frame=direct 时必填，除非使用 --calibrate-direct-origin。"
    ),
)
@click.option(
    "--calibrate-direct-origin",
    is_flag=True,
    help=(
        "启动时用当前 ZMQ 位姿标定 direct-zmq-origin: "
        "origin = zmq_link6_now - direct_init_xyz。"
        "请先把臂放到与 --init-pose 对应的 Direct 物理构型。"
    ),
)
@click.option("--init-gripper", default=0.085, type=float)
@click.option("--init-duration", default=4.0, show_default=True, type=float)
@click.option("--gripper-min-width", default=DEFAULT_ZMQ_GRIPPER_MIN, show_default=True, type=float)
@click.option("--gripper-max-width", default=DEFAULT_ZMQ_GRIPPER_MAX, show_default=True, type=float)
@click.option(
    "--gripper-scale/--no-gripper-scale",
    default=True,
    show_default=True,
    help=(
        "Direct-collected training uses width up to ~0.11 m; ZMQ uses ~0.085 m. "
        "When enabled: obs zmq→direct for policy, cmd direct→zmq before clip "
        "(same as replay_cup_demo_grasp --gripper-map=scale)."
    ),
)
@click.option(
    "--direct-gripper-full-open",
    default=DEFAULT_DIRECT_GRIPPER_FULL_OPEN,
    show_default=True,
    type=float,
    help="Direct collector full-open width (m) for --gripper-scale.",
)
@click.option(
    "--workspace-min",
    default=None,
    type=str,
    help="Optional EEF workspace lower bound x,y,z. Omitted means no lower-bound check.",
)
@click.option(
    "--workspace-max",
    default=None,
    type=str,
    help="Optional EEF workspace upper bound x,y,z. Omitted means no upper-bound check.",
)
@click.option(
    "--max-pos-speed",
    default=None,
    type=float,
    help="Optional per-waypoint translation speed limit in m/s. Omitted disables this check.",
)
@click.option(
    "--max-rot-speed",
    default=None,
    type=float,
    help="Optional per-waypoint rotation speed limit in rad/s. Omitted disables this check.",
)
@click.option("--camera-obs-latency", default=0.17, show_default=True, type=float)
@click.option("--robot-obs-latency", default=0.20, show_default=True, type=float)
@click.option(
    "--eef-offset",
    default=DEFAULT_EEF_OFFSET,
    show_default=True,
    type=float,
    help="ZMQ eef 相对 Direct link6 沿 TCP Z 的偏移（米）；设为 0 关闭。",
)
@click.option("--observe-only", is_flag=True, help="Run inference and print targets without commanding the robot.")
@click.option(
    "--observe-only-quick",
    is_flag=True,
    help="With --observe-only: seed dummy 241-frame history and infer once right after startup.",
)
@click.option(
    "--skip-server-warmup",
    is_flag=True,
    help="Do not compile/warm the policy server at startup (first C may take 1-3 min or hit WS timeout).",
)
@click.option(
    "--server-warmup-timeout",
    default=600.0,
    show_default=True,
    type=float,
    help="WebSocket ping/close timeout (s) while waiting for server warmup inference.",
)
@click.option(
    "--cached-infer/--no-cached-infer",
    default=True,
    show_default=True,
    help="Upload 241-frame history once, then send only current frame+state each step.",
)
@click.option(
    "--auto-capture/--manual-capture",
    default=True,
    show_default=True,
    help=(
        "auto: one F saves the initial frame, continuously records all three "
        "80-frame swaps, then starts policy automatically. "
        "manual: F for initial frame, each F starts the next swap, then press C."
    ),
)
@click.option(
    "--grasp-only/--full-shellgame",
    default=False,
    show_default=True,
    help=(
        "Skip shell-game F/swap capture. Seeds a dummy 241-frame history and "
        "requires --grasp-cup so the server skips MEM and freezes that cup prompt."
    ),
)
@click.option(
    "--grasp-cup",
    default=None,
    type=click.Choice(["left", "middle", "right"], case_sensitive=False),
    help="Cup to grasp when using --grasp-only (also usable to force_cup on real history).",
)
@click.option(
    "--vis-scale",
    default=DEFAULT_VIS_SCALE,
    show_default=True,
    type=float,
    help="Preview upsample factor (collector-style enlarged view).",
)
@click.option(
    "--panel-height",
    default=DEFAULT_PANEL_HEIGHT,
    show_default=True,
    type=int,
    help="Bottom status panel height in pixels.",
)
@click.option(
    "--cup-overlay-xy",
    default=None,
    type=str,
    help="Optional cup marker pixels at 224x224: 'x0,y0;x1,y1;x2,y2'.",
)
@click.option("--self-test", is_flag=True, help="Validate transforms without opening cameras or robot hardware.")
def main(
    output: str,
    policy_ip: str,
    policy_port: int,
    robot_ip: str,
    robot_port: int,
    camera_reorder: str,
    wrist_camera_index: int,
    frequency: float,
    steps_per_inference: int,
    action_lead_time: float,
    max_duration: float,
    max_inference_latency: float,
    rtc: bool,  # noqa: FBT001
    rtc_s_min: int,
    rtc_delay_buffer_size: int,
    rtc_initial_delay_steps: int,
    rtc_max_guidance_weight: float,
    image_key_style: str,
    prompt: str,
    init_pose: str | None,
    init_pose_frame: str,
    direct_zmq_origin: str | None,
    calibrate_direct_origin: bool,  # noqa: FBT001
    init_gripper: float | None,
    init_duration: float,
    gripper_min_width: float,
    gripper_max_width: float,
    gripper_scale: bool,  # noqa: FBT001
    direct_gripper_full_open: float,
    workspace_min: str | None,
    workspace_max: str | None,
    max_pos_speed: float | None,
    max_rot_speed: float | None,
    camera_obs_latency: float,
    robot_obs_latency: float,
    eef_offset: float,
    observe_only: bool,  # noqa: FBT001
    observe_only_quick: bool,  # noqa: FBT001
    skip_server_warmup: bool,  # noqa: FBT001
    server_warmup_timeout: float,
    cached_infer: bool,  # noqa: FBT001
    auto_capture: bool,  # noqa: FBT001
    grasp_only: bool,  # noqa: FBT001
    grasp_cup: str | None,
    vis_scale: float,
    panel_height: int,
    cup_overlay_xy: str | None,
    self_test: bool,  # noqa: FBT001
) -> None:
    if self_test:
        run_contract_self_test()
        return
    if observe_only_quick and not observe_only:
        raise click.BadParameter("--observe-only-quick requires --observe-only")
    if grasp_only and observe_only:
        raise click.BadParameter("--grasp-only cannot be combined with --observe-only")
    if grasp_only and not grasp_cup:
        raise click.BadParameter("--grasp-only requires --grasp-cup {left,middle,right}")
    if grasp_cup is not None:
        grasp_cup = grasp_cup.lower()
    force_cup = grasp_cup

    from modules.arx5_env import Arx5Env
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    from peripherals.keystroke_counter import KeyCode
    from peripherals.keystroke_counter import KeystrokeCounter
    from utils.other_util import precise_wait

    if frequency <= 0 or init_duration <= 0:
        raise click.BadParameter("frequency and init-duration must be positive")
    if not 0 <= gripper_min_width < gripper_max_width:
        raise click.BadParameter("invalid gripper width range")
    if gripper_scale and direct_gripper_full_open <= 0:
        raise click.BadParameter("--direct-gripper-full-open must be positive when --gripper-scale")
    if max_pos_speed is not None and max_pos_speed <= 0:
        raise click.BadParameter("--max-pos-speed must be positive when configured")
    if max_rot_speed is not None and max_rot_speed <= 0:
        raise click.BadParameter("--max-rot-speed must be positive when configured")
    if server_warmup_timeout <= 0:
        raise click.BadParameter("--server-warmup-timeout must be positive")
    if eef_offset < 0:
        raise click.BadParameter("--eef-offset must be non-negative")
    if vis_scale < 1.0:
        raise click.BadParameter("--vis-scale must be >= 1")
    if panel_height < 120:
        raise click.BadParameter("--panel-height must be >= 120")
    init_pose_frame = init_pose_frame.lower()

    init_pose6_raw = parse_csv_vector(init_pose, size=6, name="--init-pose")
    direct_origin3: np.ndarray | None = None
    if direct_zmq_origin is not None:
        direct_origin3 = parse_csv_vector(
            direct_zmq_origin, size=3, name="--direct-zmq-origin"
        )
    if init_pose_frame == "direct":
        if direct_origin3 is None and not calibrate_direct_origin:
            raise click.BadParameter(
                "init-pose-frame=direct requires --direct-zmq-origin "
                "or --calibrate-direct-origin"
            )
    elif init_pose_frame != "zmq":
        raise click.BadParameter(f"unsupported init-pose-frame: {init_pose_frame}")
    elif calibrate_direct_origin:
        raise click.BadParameter("--calibrate-direct-origin only works with init-pose-frame=direct")

    init_pose6 = init_pose6_raw
    if init_pose_frame == "direct" and direct_origin3 is not None:
        init_pose6 = direct_link6_pose_to_zmq_eef(
            init_pose6_raw,
            direct_origin3,
            eef_offset,
        )
        print(
            "[ALIGN] converted Direct init pose → ZMQ eef: "
            f"{np.array2string(init_pose6, precision=6, separator=',')}"
        )
        print(
            f"[ALIGN] direct-zmq-origin={np.array2string(direct_origin3, precision=6, separator=',')}"
        )
    if init_pose_frame == "direct":
        target_link6_pose6 = init_pose6_raw.copy()
    else:
        target_link6_pose6 = eef_pose_to_link6(init_pose6_raw, eef_offset)
    workspace_min3 = (
        np.full(3, -np.inf, dtype=np.float64)
        if workspace_min is None
        else parse_csv_vector(workspace_min, size=3, name="--workspace-min")
    )
    workspace_max3 = (
        np.full(3, np.inf, dtype=np.float64)
        if workspace_max is None
        else parse_csv_vector(workspace_max, size=3, name="--workspace-max")
    )
    if np.any(workspace_min3 >= workspace_max3):
        raise click.BadParameter("workspace-min must be smaller than workspace-max")
    max_pos_speed = np.inf if max_pos_speed is None else max_pos_speed
    max_rot_speed = np.inf if max_rot_speed is None else max_rot_speed

    print(
        "[SAFETY] eval guards: "
        f"workspace_min={'disabled' if workspace_min is None else workspace_min}, "
        f"workspace_max={'disabled' if workspace_max is None else workspace_max}, "
        f"max_pos_speed={'disabled' if np.isinf(max_pos_speed) else f'{max_pos_speed:g} m/s'}, "
        f"max_rot_speed={'disabled' if np.isinf(max_rot_speed) else f'{max_rot_speed:g} rad/s'}"
    )

    output_path = Path(output).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    camera_order = [int(item.strip()) for item in camera_reorder.split(",")]

    print(f"Connecting to policy server ws://{policy_ip}:{policy_port}")
    client = WebsocketClientPolicy(
        host=policy_ip,
        port=policy_port,
        ping_timeout=server_warmup_timeout,
        close_timeout=server_warmup_timeout,
    )
    print("Policy server connected")
    policy_client = ShellGamePolicyClient(
        client,
        cached_infer=cached_infer,
        image_key_style=image_key_style,
    )
    if cached_infer:
        if policy_client.cached_infer_enabled:
            print("[cache] server supports cached ShellGame inference")
        else:
            print("[cache] server lacks cached infer support; falling back to full 242-frame infer")
    if rtc:
        print(
            f"[RTC] client flag on (server_supports_rtc={policy_client.rtc_enabled}, "
            f"image_key_style={image_key_style})"
        )
    print(
        "[ALIGN] Direct/ZMQ world-origin difference cancels in relative SE(3); "
        f"link6<->eef offset={eef_offset:.3f}m along TCP Z"
    )
    if gripper_scale:
        zmq_span = gripper_max_width - gripper_min_width
        factor = zmq_span / direct_gripper_full_open
        print(
            f"[GRIP] scale enabled: Direct↔ZMQ ×{factor:.4f} "
            f"(full-open {direct_gripper_full_open:.3f}m → ZMQ {gripper_max_width:.3f}m); "
            "obs zmq→direct, cmd direct→zmq"
        )
    else:
        print("[GRIP] scale disabled: gripper passed through in ZMQ units")
    if observe_only:
        if observe_only_quick:
            print(
                "[observe-only] quick mode: will infer once immediately after env ready "
                "(dummy 241-frame history, no robot motion)."
            )
        else:
            print(
                "[observe-only] full mode: press F → capture 3×80 swap frames → C "
                "to print inference targets (no robot motion)."
            )
    if grasp_only:
        print(
            f"[grasp-only] skip shell-game capture; dummy 241 history + force_cup={force_cup}. "
            "F: cycle cup (L/M/R)  C: start grasp  P: reset  Q: quit"
        )
    elif auto_capture:
        print(
            "[capture] auto mode: one F starts continuous recording "
            f"(1 + {NUM_SWAPS * SWAP_FRAMES} frames), then policy starts automatically"
        )
    else:
        print(
            "[capture] manual mode: F for initial frame, then each F starts one swap; "
            "press C to start policy"
        )

    robots_config = [
        {
            "robot_type": "arx5",
            "robot_ip": robot_ip,
            "robot_port": robot_port,
            "robot_obs_latency": robot_obs_latency,
            "robot_action_latency": 0.01,
        }
    ]

    history = ShellGameHistory(auto_capture=auto_capture)
    # Arx5Env preview resolution is fixed at 224x224 in this script.
    cup_overlay_positions = resolve_cup_overlay_positions(224, 224, cup_overlay_xy)
    last_memory: dict | None = None
    warmup_done = False
    dt = 1.0 / frequency
    print(
        f"[vis] scale={vis_scale:.1f}x panel={panel_height}px "
        f"cup_markers={cup_overlay_positions}"
    )

    with SharedMemoryManager() as shm_manager:  # noqa: SIM117
        with (
            KeystrokeCounter() as key_counter,
            Arx5Env(
                output_dir=output_path,
                robots_config=robots_config,
                frequency=frequency,
                obs_image_resolution=(224, 224),
                max_obs_buffer_size=120,
                obs_float32=False,
                camera_reorder=camera_order,
                init_joints=False,
                enable_multi_cam_vis=False,
                camera_obs_latency=camera_obs_latency,
                camera_obs_horizon=1,
                robot_obs_horizon=1,
                max_pos_speed=max_pos_speed,
                max_rot_speed=max_rot_speed,
                shm_manager=shm_manager,
            ) as env,
        ):
            cv2.setNumThreads(2)
            print(f"Waiting for camera and ARX5 ZMQ controller tcp://{robot_ip}:{robot_port}...")
            while not env.is_ready:
                time.sleep(0.1)

            if init_pose_frame == "direct" and calibrate_direct_origin:
                live_eef, _ = current_robot_pose_and_gripper(env)
                live_link6 = eef_pose_to_link6(live_eef, eef_offset)
                direct_origin3 = live_link6[:3] - init_pose6_raw[:3]
                init_pose6 = direct_link6_pose_to_zmq_eef(
                    init_pose6_raw,
                    direct_origin3,
                    eef_offset,
                )
                print(
                    "[ALIGN] calibrated direct-zmq-origin from current pose: "
                    f"{np.array2string(direct_origin3, precision=6, separator=',')}"
                )
                print(
                    "[ALIGN] converted Direct init pose → ZMQ eef: "
                    f"{np.array2string(init_pose6, precision=6, separator=',')}"
                )

            if observe_only:
                live_eef, _ = current_robot_pose_and_gripper(env)
                target_link6_pose6 = eef_pose_to_link6(live_eef, eef_offset)
                init_pose6 = live_eef.copy()
                print("[observe-only] environment ready; keeping the current robot pose")
            else:
                if np.any(init_pose6[:3] < workspace_min3) or np.any(init_pose6[:3] > workspace_max3):
                    raise click.BadParameter("init-pose is outside the configured workspace")

                print("Environment ready; moving to init_pose")
                move_to_pose(
                    env,
                    init_pose6,
                    duration=init_duration,
                    frequency=frequency,
                    target_gripper=init_gripper,
                    gripper_min=gripper_min_width,
                    gripper_max=gripper_max_width,
                )
                live_eef, _ = current_robot_pose_and_gripper(env)
                print_init_link6_alignment(
                    "after-move",
                    target_link6_pose6,
                    eef_pose_to_link6(live_eef, eef_offset),
                    target_zmq_eef=init_pose6,
                    live_zmq_eef=live_eef,
                    direct_origin3=direct_origin3,
                )
            current_link6_pose = eef_pose_to_link6(live_eef, eef_offset)
            if not skip_server_warmup:
                warmup_policy_server(
                    policy_client=policy_client,
                    env=env,
                    wrist_camera_index=wrist_camera_index,
                    prompt=prompt,
                    current_link6_pose=current_link6_pose,
                    gripper_scale=gripper_scale,
                    gripper_min=gripper_min_width,
                    gripper_max=gripper_max_width,
                    direct_gripper_full_open=direct_gripper_full_open,
                )
                warmup_done = True
            if observe_only_quick:
                obs = env.get_obs()
                camera_key = f"camera{wrist_camera_index}_rgb"
                frame = obs[camera_key][-1]
                current_pose = np.concatenate(
                    [obs["robot0_eef_pos"][-1], obs["robot0_eef_rot_axis_angle"][-1]],
                    axis=-1,
                )
                current_link6_pose = eef_pose_to_link6(current_pose, eef_offset)
                history.seed_dummy_history(frame, current_link6_pose)
                print(
                    "[observe-only] dummy history seeded; running one alignment inference..."
                )
                run_policy(
                    env=env,
                    policy_client=policy_client,
                    history=history,
                    wrist_camera_index=wrist_camera_index,
                    prompt=prompt,
                    frequency=frequency,
                    steps_per_inference=steps_per_inference,
                    action_lead_time=action_lead_time,
                    max_duration=max_duration,
                    max_inference_latency=max_inference_latency,
                    max_pos_speed=max_pos_speed,
                    max_rot_speed=max_rot_speed,
                    workspace_min=workspace_min3,
                    workspace_max=workspace_max3,
                    gripper_min=gripper_min_width,
                    gripper_max=gripper_max_width,
                    eef_offset=eef_offset,
                    gripper_scale=gripper_scale,
                    direct_gripper_full_open=direct_gripper_full_open,
                    key_counter=key_counter,
                    observe_only=True,
                    warmup_done=True,
                    vis_scale=vis_scale,
                    panel_height=panel_height,
                    cup_overlay_positions=cup_overlay_positions,
                    rtc=False,
                )
                print("[observe-only] done.")
                cv2.destroyAllWindows()
                return

            if auto_capture:
                f_hint = "Press F to auto-capture; policy starts when history is ready."
            else:
                f_hint = "Press F to capture the initial frame."
            if grasp_only:
                obs = env.get_obs()
                camera_key = f"camera{wrist_camera_index}_rgb"
                frame = obs[camera_key][-1]
                current_pose = np.concatenate(
                    [obs["robot0_eef_pos"][-1], obs["robot0_eef_rot_axis_angle"][-1]],
                    axis=-1,
                )
                current_link6_pose = eef_pose_to_link6(current_pose, eef_offset)
                history.seed_dummy_history(frame, current_link6_pose)
                print(
                    f"[grasp-only] seeded dummy {HISTORY_FRAMES}-frame history; "
                    f"cup={force_cup}. F cycles cup, C starts grasp."
                )
            elif observe_only:
                print(f"At current pose. {f_hint}")
            else:
                print(f"At init_pose. {f_hint}")
            if observe_only and not grasp_only:
                print(
                    "[observe-only] waiting for 241-frame history before inference; "
                    "or restart with --observe-only-quick"
                )

            quit_requested = False
            while not quit_requested:
                target_pose, target_gripper = current_robot_pose_and_gripper(env)
                start_policy = False
                iteration = 0
                loop_start = time.monotonic()
                was_ready = history.ready

                while not start_policy and not quit_requested:
                    cycle_end = loop_start + (iteration + 1) * dt
                    obs = env.get_obs()
                    camera_key = f"camera{wrist_camera_index}_rgb"
                    frame = obs[camera_key][-1]
                    current_pose = np.concatenate(
                        [obs["robot0_eef_pos"][-1], obs["robot0_eef_rot_axis_angle"][-1]], axis=-1
                    )
                    current_link6_pose = eef_pose_to_link6(current_pose, eef_offset)
                    message = ""

                    for key in key_counter.get_press_events():
                        if key == KeyCode(char="q"):
                            quit_requested = True
                            break
                        if key == KeyCode(char="f"):
                            if grasp_only:
                                assert force_cup is not None
                                cup_idx = CUP_NAMES.index(force_cup)
                                force_cup = CUP_NAMES[(cup_idx + 1) % len(CUP_NAMES)]
                                message = (
                                    f"[grasp-only] cup → {CUP_DISPLAY_NAMES[CUP_NAMES.index(force_cup)]} "
                                    f"({force_cup})"
                                )
                                print(message)
                                continue
                            message = history.press_f(frame, current_link6_pose)
                            print_init_link6_alignment(
                                "press-F",
                                target_link6_pose6,
                                current_link6_pose,
                                target_zmq_eef=init_pose6,
                                live_zmq_eef=current_pose,
                                direct_origin3=direct_origin3,
                            )
                            print(message)
                        elif key == KeyCode(char="c"):
                            if history.ready:
                                start_policy = True
                            else:
                                message = f"History incomplete: {history.frame_count}/241"
                                print(message)
                        elif key == KeyCode(char="p"):
                            reset_target = "current pose" if observe_only else "init_pose"
                            print(f"P pressed: clearing history and returning to {reset_target}")
                            policy_client.reset_session()
                            history.reset()
                            last_memory = None
                            if not observe_only:
                                move_to_pose(
                                    env,
                                    init_pose6,
                                    duration=init_duration,
                                    frequency=frequency,
                                    target_gripper=init_gripper,
                                    gripper_min=gripper_min_width,
                                    gripper_max=gripper_max_width,
                                )
                            target_pose, target_gripper = current_robot_pose_and_gripper(env)
                            loop_start = time.monotonic()
                            iteration = 0
                            if grasp_only:
                                seed_obs = env.get_obs()
                                seed_frame = seed_obs[camera_key][-1]
                                seed_pose = np.concatenate(
                                    [
                                        seed_obs["robot0_eef_pos"][-1],
                                        seed_obs["robot0_eef_rot_axis_angle"][-1],
                                    ],
                                    axis=-1,
                                )
                                seed_link6 = eef_pose_to_link6(seed_pose, eef_offset)
                                history.seed_dummy_history(seed_frame, seed_link6)
                                message = (
                                    f"[grasp-only] re-seeded dummy history; "
                                    f"cup={force_cup}. F cycles cup, C starts grasp."
                                )
                            else:
                                message = (
                                    "At init_pose; press F to auto-capture"
                                    if auto_capture
                                    else "At init_pose; press F for the new initial frame"
                                )
                            print(message)

                    capture_message = history.capture_tick(frame)
                    if capture_message and (history.ready or history.frame_count % 10 == 0 or not history.collecting):
                        print(capture_message)
                    # Only auto-start on the ready transition, not while already ready after S.
                    if (
                        auto_capture
                        and not grasp_only
                        and history.ready
                        and not was_ready
                        and not quit_requested
                    ):
                        print("Auto-capture complete: starting policy")
                        start_policy = True
                    was_ready = history.ready
                    if start_policy or quit_requested:
                        break

                    # Hold pose before F locks the viewpoint; no SpaceMouse teleop.
                    if not observe_only and not history.locked:
                        command = np.concatenate([target_pose, [target_gripper]])
                        env.exec_actions(
                            actions=[command],
                            timestamps=[time.time() + dt],
                            compensate_latency=False,
                        )

                    if grasp_only:
                        help_text = f"F: cycle cup  C: grasp  P: reset  Q: quit"
                    elif auto_capture:
                        help_text = "F: auto-capture→policy  P: reset  Q: quit"
                    else:
                        help_text = "F: capture  C: policy  P: reset  Q: quit"
                    remaining = ""
                    if history.collecting:
                        swap_left = SWAP_FRAMES - len(history.active_segment or [])
                        if auto_capture:
                            swaps_left = NUM_SWAPS - len(history.swap_segments) - 1
                            total_left = swap_left + swaps_left * SWAP_FRAMES
                            remaining = (
                                f"swap remaining {swap_left / frequency:.1f}s | "
                                f"total remaining {total_left / frequency:.1f}s"
                            )
                        else:
                            remaining = f"remaining {swap_left / frequency:.1f}s"
                    status_label = history.status()
                    if grasp_only:
                        assert force_cup is not None
                        cup_idx = CUP_NAMES.index(force_cup)
                        status_label = (
                            f"GRASP-ONLY  cup={CUP_DISPLAY_NAMES[cup_idx]} ({force_cup})"
                        )
                    elif history.ready:
                        status_label = "READY 241/241"
                    elif history.collecting:
                        status_label = (
                            f"REC {len(history.swap_segments) + 1}/{NUM_SWAPS} "
                            f"{len(history.active_segment or [])}/{SWAP_FRAMES}"
                        )
                    detail_lines = [line for line in (remaining, message) if line]
                    if grasp_only and force_cup is not None:
                        detail_lines = [
                            f"selected: {force_cup.upper()}  "
                            f"[{' | '.join(n.upper() if n == force_cup else n for n in CUP_NAMES)}]",
                            *detail_lines,
                        ]
                    view = make_eval_vis(
                        frame,
                        status=status_label,
                        help_text=help_text,
                        detail_lines=detail_lines,
                        memory=last_memory,
                        cup_overlay_positions=cup_overlay_positions,
                        vis_scale=vis_scale,
                        panel_height=panel_height,
                    )
                    cv2.imshow("ARX5 ShellGame", view[..., ::-1])
                    cv2.pollKey()
                    precise_wait(cycle_end)
                    iteration += 1

                if quit_requested:
                    break
                quit_requested, warmup_done, last_memory = run_policy(
                    env=env,
                    policy_client=policy_client,
                    history=history,
                    wrist_camera_index=wrist_camera_index,
                    prompt=prompt,
                    frequency=frequency,
                    steps_per_inference=steps_per_inference,
                    action_lead_time=action_lead_time,
                    max_duration=max_duration,
                    max_inference_latency=max_inference_latency,
                    max_pos_speed=max_pos_speed,
                    max_rot_speed=max_rot_speed,
                    workspace_min=workspace_min3,
                    workspace_max=workspace_max3,
                    gripper_min=gripper_min_width,
                    gripper_max=gripper_max_width,
                    eef_offset=eef_offset,
                    gripper_scale=gripper_scale,
                    direct_gripper_full_open=direct_gripper_full_open,
                    key_counter=key_counter,
                    observe_only=observe_only,
                    warmup_done=warmup_done,
                    vis_scale=vis_scale,
                    panel_height=panel_height,
                    cup_overlay_positions=cup_overlay_positions,
                    rtc=rtc,
                    rtc_s_min=rtc_s_min,
                    rtc_delay_buffer_size=rtc_delay_buffer_size,
                    rtc_initial_delay_steps=rtc_initial_delay_steps,
                    rtc_max_guidance_weight=rtc_max_guidance_weight,
                    force_cup=force_cup,
                )

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
