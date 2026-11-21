"""Detect and trim leftover end-of-episode gripper open-close gestures.

Used by ``scripts/trim_end_grip_gesture_lerobot.py`` and the data-management web UI.
Trims parquet/zarr + videos so vision training stays consistent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np

ProgressCb = Optional[Callable[[dict], None]]


def _state(w: float, close_thresh: float, open_thresh: float) -> Optional[str]:
    if w <= close_thresh:
        return "C"
    if w >= open_thresh:
        return "O"
    return None


def _iter_edge_events(
    g: np.ndarray,
    *,
    close_thresh: float,
    open_thresh: float,
    start: int = 0,
    end: Optional[int] = None,
) -> list[tuple[int, str, str]]:
    events: list[tuple[int, str, str]] = []
    state: Optional[str] = None
    n = int(g.shape[0]) if end is None else int(end)
    for i in range(max(0, int(start)), n):
        ns = _state(float(g[i]), close_thresh, open_thresh)
        if ns is None:
            continue
        if state is None:
            state = ns
            continue
        if ns != state:
            events.append((i, state, ns))
            state = ns
    return events


def _closing_onset_from_full_open(
    g: np.ndarray,
    open_i: int,
    close_i: int,
    open_thresh: float,
) -> int:
    """全开平台末端 → 开始闭合的拐点（保留到该帧之前）。"""
    open_i = int(open_i)
    close_i = int(close_i)
    if close_i <= open_i:
        return int(open_i)
    peak_i = open_i + int(np.argmax(g[open_i : close_i + 1]))
    plateau = float(g[peak_i])
    # 相对高台下降一点即视为离开全开（比等跌破 open_thresh 更早、更贴拐点）
    delta = max(0.004, 0.06 * max(plateau - float(open_thresh), 0.01))
    i = peak_i
    while i < close_i and float(g[i]) >= plateau - delta:
        i += 1
    if i <= peak_i:
        i = peak_i
        while i < close_i and float(g[i]) >= open_thresh:
            i += 1
    return int(i)


def find_end_grip_gesture_cut(
    grip: np.ndarray,
    *,
    fps: float = 20.0,
    close_thresh: float = 0.0295,
    open_thresh: float = 0.0665,
    window_s: float = 1.5,
    search_s: float = 3.0,
    min_keep: int = 30,
) -> Optional[int]:
    """Return keep-end for leftover end open-close; keep ``[0, cut)``.

    定位末尾短窗开合中「从全开到闭合」的拐点：先找到靠近结尾的 O→C，
    再取配对张开段上离开全开平台、开始往下合的那一帧作为 ``cut``。
    这样会裁掉结尾的「夹」动作，但不会从张开点长扫进回零。
    """
    g = np.asarray(grip, dtype=np.float64).reshape(-1)
    n = int(g.shape[0])
    if n < min_keep + 5:
        return None

    max_drop = int(round((window_s + 0.75) * fps))
    scan0 = max(0, n - int(round((search_s + window_s + 0.5) * fps)))
    near_close0 = n - int(round(search_s * fps))

    events = _iter_edge_events(
        g, close_thresh=close_thresh, open_thresh=open_thresh, start=scan0
    )
    if not events:
        return None

    close_events = [
        e for e in events if e[1] == "O" and e[2] == "C" and e[0] >= near_close0
    ]

    open_i: Optional[int] = None
    close_i: Optional[int] = None

    if close_events:
        close_i = int(close_events[-1][0])
        for e in reversed(events):
            if e[0] >= close_i:
                continue
            if e[1] == "C" and e[2] == "O":
                open_i = int(e[0])
                break
        if open_i is None:
            j = close_i
            lo = max(0, close_i - max_drop)
            while j > lo and float(g[j]) < open_thresh:
                j -= 1
            while j > lo and float(g[j]) >= open_thresh:
                j -= 1
            open_i = j + 1
            if open_i >= close_i:
                return None
            mid = g[open_i:close_i]
            if mid.size == 0 or float(np.max(mid)) < open_thresh:
                return None
    else:
        # 未跌破 close_thresh：结尾仍可能有「全开后往下夹」
        open_events = [
            e
            for e in events
            if e[1] == "C"
            and e[2] == "O"
            and e[0] >= n - int(round((window_s + 1.0) * fps))
        ]
        if not open_events:
            return None
        open_i = int(open_events[-1][0])
        close_i = n - 1
        peak = float(np.max(g[open_i:]))
        if peak < open_thresh:
            return None
        # 末帧需明显低于全开平台，确有往下合的趋势
        if float(g[-1]) > peak - max(0.008, 0.1 * (peak - close_thresh)):
            return None

    assert open_i is not None and close_i is not None

    # 张开平台可以稍长；闭合段本身应短。总跨度限制放宽到 search 窗
    if (close_i - open_i) > int(round(search_s * fps)):
        return None

    # 双开合：短窗内还有上一次完整开合 → 用第一次的全开→闭合拐点
    prev_close = None
    prev_open = None
    for e in reversed(events):
        if e[0] >= open_i:
            continue
        if prev_close is None and e[1] == "O" and e[2] == "C":
            prev_close = int(e[0])
            continue
        if prev_close is not None and e[1] == "C" and e[2] == "O":
            prev_open = int(e[0])
            break
    if (
        prev_open is not None
        and prev_close is not None
        and (open_i - prev_open) <= int(round(window_s * fps))
        and (prev_close - prev_open) <= int(round(search_s * fps))
    ):
        open_i = int(prev_open)
        close_i = int(prev_close)

    cut = _closing_onset_from_full_open(g, open_i, close_i, open_thresh)

    drop = n - cut
    if drop < 3:
        return None
    if drop > max_drop:
        return None
    if cut < min_keep or cut <= 0 or cut >= n:
        return None
    return int(cut)


def default_grip_thresholds(
    gripper_min: float = 0.011,
    gripper_max: float = 0.085,
) -> tuple[float, float]:
    span = float(gripper_max) - float(gripper_min)
    close_thresh = float(gripper_min) + 0.25 * span
    open_thresh = float(gripper_min) + 0.75 * span
    return close_thresh, open_thresh


def _load_lerobot_grip(pq_path: Path, grip_idx: int) -> np.ndarray:
    import pyarrow.parquet as pq

    table = pq.read_table(pq_path, columns=["action"])
    acts = [
        np.asarray(x, dtype=np.float64) for x in table.column("action").to_pylist()
    ]
    if not acts:
        return np.zeros((0,), dtype=np.float64)
    arr = np.stack(acts, axis=0)
    if arr.ndim != 2 or arr.shape[1] <= grip_idx:
        raise ValueError(f"action dim too small in {pq_path}: {arr.shape}")
    return arr[:, int(grip_idx)]


def _load_zarr_grip(env, episode_id: int, grip_idx: int) -> np.ndarray:
    rb = env.replay_buffer
    sl = rb.get_episode_slice(int(episode_id))
    # Prefer dedicated gripper key when present
    robot_idx = int(grip_idx) // 7
    gkey = f"robot{robot_idx}_gripper_width"
    data = rb.data
    if gkey in data:
        return np.asarray(data[gkey][sl], dtype=np.float64).reshape(-1)
    if "action" in data:
        act = np.asarray(data["action"][sl], dtype=np.float64)
        if act.ndim == 2 and act.shape[1] > grip_idx:
            return act[:, int(grip_idx)].reshape(-1)
    raise KeyError(f"no gripper channel for ep{episode_id} (idx={grip_idx})")


def plan_end_grip_trims(
    env,
    *,
    grip_idx: int = 13,
    gripper_min: float = 0.011,
    gripper_max: float = 0.085,
    window_s: float = 1.5,
    search_s: float = 3.0,
    episode_id: Optional[int] = None,
    fps: Optional[float] = None,
) -> list[dict]:
    """Scan dataset; return plans ``[{episode_id,n,cut,drop}, ...]``."""
    close_thresh, open_thresh = default_grip_thresholds(gripper_min, gripper_max)
    fmt = str(getattr(env, "dataset_format", "zarr") or "zarr")
    plans: list[dict] = []

    if fmt == "lerobot":
        from modules.lerobot_dataset_io import (
            episode_parquet_path,
            list_episode_ids,
            load_info,
        )

        root = Path(env.output_dir)
        info = load_info(root)
        fps_v = float(fps if fps is not None else (info.get("fps") or 20.0))
        ids = (
            [int(episode_id)]
            if episode_id is not None
            else list_episode_ids(root, info)
        )
        for eid in ids:
            pq_path = episode_parquet_path(root, eid, info)
            if not pq_path.is_file():
                continue
            grip = _load_lerobot_grip(pq_path, grip_idx)
            n = int(grip.shape[0])
            cut = find_end_grip_gesture_cut(
                grip,
                fps=fps_v,
                close_thresh=close_thresh,
                open_thresh=open_thresh,
                window_s=window_s,
                search_s=search_s,
            )
            if cut is None:
                continue
            c = int(cut)
            plans.append(
                {
                    "episode_id": int(eid),
                    "n": n,
                    "cut": c,
                    "drop": n - c,
                    # 网页裁剪标记：保留 inclusive [start, end]
                    "start": 0,
                    "end": max(0, c - 1),
                }
            )
        return plans

    # zarr
    rb = env.replay_buffer
    n_eps = int(rb.n_episodes)
    fps_v = float(fps if fps is not None else 20.0)
    if episode_id is not None:
        eids = [int(episode_id)]
    else:
        eids = list(range(n_eps))
    for eid in eids:
        if eid < 0 or eid >= n_eps:
            continue
        grip = _load_zarr_grip(env, eid, grip_idx)
        n = int(grip.shape[0])
        cut = find_end_grip_gesture_cut(
            grip,
            fps=fps_v,
            close_thresh=close_thresh,
            open_thresh=open_thresh,
            window_s=window_s,
            search_s=search_s,
        )
        if cut is None:
            continue
        c = int(cut)
        plans.append(
            {
                "episode_id": int(eid),
                "n": n,
                "cut": c,
                "drop": n - c,
                "start": 0,
                "end": max(0, c - 1),
            }
        )
    return plans


def apply_end_grip_trims(
    env,
    *,
    grip_idx: int = 13,
    gripper_min: float = 0.011,
    gripper_max: float = 0.085,
    window_s: float = 1.5,
    search_s: float = 3.0,
    episode_id: Optional[int] = None,
    max_passes: int = 1,
    progress_cb: ProgressCb = None,
    trim_fn: Optional[Callable[..., bool]] = None,
) -> dict:
    """Scan + trim. Default one pass to avoid eating return-home.

    Returns summary dict.
    """
    if trim_fn is None:
        raise ValueError("trim_fn is required (e.g. trim_saved_episode)")

    total_trimmed = 0
    total_drop = 0
    passes_done = 0
    last_plans: list[dict] = []

    for pass_i in range(max(1, int(max_passes))):
        plans = plan_end_grip_trims(
            env,
            grip_idx=grip_idx,
            gripper_min=gripper_min,
            gripper_max=gripper_max,
            window_s=window_s,
            search_s=search_s,
            episode_id=episode_id,
        )
        last_plans = plans
        if not plans:
            break
        passes_done = pass_i + 1
        # high → low so lerobot global index rewrites stay cheap
        ordered = sorted(plans, key=lambda p: int(p["episode_id"]), reverse=True)
        for j, plan in enumerate(ordered):
            eid = int(plan["episode_id"])
            cut = int(plan["cut"])
            drop = int(plan["drop"])
            if progress_cb is not None:
                progress_cb(
                    {
                        "phase": "end_grip_trim",
                        "frac": (j + 0.01) / max(len(ordered), 1),
                        "message": (
                            f"末尾开合裁剪 pass{passes_done} "
                            f"ep{eid} drop {drop} ({j + 1}/{len(ordered)})"
                        ),
                        "episode_id": eid,
                        "drop": drop,
                        "cut": cut,
                        "close_anchor": plan.get("close"),
                    }
                )

            def _cb(info, _eid=eid):
                if progress_cb is None or not isinstance(info, dict):
                    return
                payload = dict(info)
                msg = str(payload.get("message") or "")
                if msg and not msg.startswith("末尾开合"):
                    payload["message"] = f"ep{_eid}: {msg}"
                progress_cb(payload)

            ok = bool(trim_fn(env, eid, 0, cut, progress_cb=_cb))
            if ok:
                total_trimmed += 1
                total_drop += drop

        # single-episode mode: one pass enough
        if episode_id is not None:
            break

    if progress_cb is not None:
        progress_cb(
            {
                "phase": "done",
                "frac": 1.0,
                "message": (
                    f"末尾开合裁剪完成 · 处理 {total_trimmed} 次 · "
                    f"共丢 {total_drop} 帧 · {passes_done} 轮"
                ),
            }
        )
    return {
        "n_trimmed": int(total_trimmed),
        "n_drop_frames": int(total_drop),
        "n_passes": int(passes_done),
        "remaining": len(last_plans) if passes_done >= max_passes else 0,
        "preview": last_plans if total_trimmed == 0 else [],
    }
