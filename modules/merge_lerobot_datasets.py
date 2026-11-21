"""Merge multiple LeRobot v2.1 datasets into one training-ready dataset.

Copies / hardlinks videos, rewrites parquet ``episode_index`` / ``index`` /
``task_index``, and rebuilds ``meta/{info,episodes,tasks,episodes_stats}.json(l)``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from modules.lerobot_dataset_io import (
    LEROBOT_CAM_KEYS,
    episode_parquet_path,
    episode_video_path,
    is_lerobot_dataset,
    list_episode_ids,
    load_info,
)

ProgressCb = Optional[Callable[[dict], None]]


def _emit(progress_cb: ProgressCb, **payload) -> None:
    if progress_cb is not None:
        progress_cb(dict(payload))


def _require_pq():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "合并 LeRobot 需要 pyarrow。请在当前环境安装: pip install pyarrow"
        ) from exc
    return pa, pq


def _video_keys_from_info(info: dict) -> list[str]:
    feats = info.get("features") or {}
    keys = [
        str(k)
        for k, v in feats.items()
        if isinstance(v, dict) and str(v.get("dtype") or "") == "video"
    ]
    if keys:
        return keys
    return list(LEROBOT_CAM_KEYS)


def _feature_sig(info: dict) -> dict:
    feats = info.get("features") or {}
    out = {}
    for k, v in feats.items():
        if not isinstance(v, dict):
            continue
        out[str(k)] = {
            "dtype": v.get("dtype"),
            "shape": list(v.get("shape") or []),
        }
    return out


def validate_lerobot_sources(sources: list[Path]) -> dict:
    """Validate sources are merge-compatible. Returns ref info from first dataset."""
    if len(sources) < 2:
        raise ValueError("请至少选择 2 个 LeRobot 数据集进行合并")

    ref_info: Optional[dict] = None
    ref_sig: Optional[dict] = None
    ref_fps: Optional[float] = None
    ref_robot: Optional[str] = None
    ref_vkeys: Optional[list[str]] = None

    for raw in sources:
        root = Path(raw).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"目录不存在: {root}")
        if not is_lerobot_dataset(root):
            raise ValueError(f"不是 LeRobot 数据集（缺 meta/info.json）: {root}")
        info = load_info(root)
        ids = list_episode_ids(root, info)
        if not ids:
            raise ValueError(f"数据集没有 episode: {root}")
        fps = float(info.get("fps") or 20.0)
        robot = str(info.get("robot_type") or "")
        sig = _feature_sig(info)
        vkeys = _video_keys_from_info(info)
        if ref_info is None:
            ref_info = info
            ref_sig = sig
            ref_fps = fps
            ref_robot = robot
            ref_vkeys = vkeys
            continue
        if abs(fps - float(ref_fps)) > 1e-6:
            raise ValueError(
                f"fps 不一致: {root} fps={fps} vs 参考 {ref_fps}"
            )
        if robot and ref_robot and robot != ref_robot:
            raise ValueError(
                f"robot_type 不一致: {root}={robot} vs 参考 {ref_robot}"
            )
        if sig != ref_sig:
            raise ValueError(
                f"features 不一致，无法安全合并: {root}\n"
                f"  此集: {sorted(sig.keys())}\n"
                f"  参考: {sorted((ref_sig or {}).keys())}"
            )
        if sorted(vkeys) != sorted(ref_vkeys or []):
            raise ValueError(
                f"视频键不一致: {root} {vkeys} vs 参考 {ref_vkeys}"
            )
        # spot-check first episode files exist
        eid0 = int(ids[0])
        pq0 = episode_parquet_path(root, eid0, info)
        if not pq0.is_file():
            raise FileNotFoundError(f"缺少 parquet: {pq0}")
        for vk in vkeys:
            vp = episode_video_path(root, eid0, vk, info)
            if not vp.is_file():
                raise FileNotFoundError(f"缺少视频: {vp}")

    assert ref_info is not None
    return ref_info


def _load_tasks(root: Path) -> list[dict]:
    path = Path(root) / "meta" / "tasks.jsonl"
    rows: list[dict] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _load_episode_rows(root: Path) -> dict[int, dict]:
    path = Path(root) / "meta" / "episodes.jsonl"
    out: dict[int, dict] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            out[int(row["episode_index"])] = row
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    return out


def _load_stats_rows(root: Path) -> dict[int, dict]:
    path = Path(root) / "meta" / "episodes_stats.jsonl"
    out: dict[int, dict] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            out[int(row["episode_index"])] = row
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    return out


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os_link = getattr(__import__("os"), "link")
        os_link(str(src), str(dst))
    except OSError:
        shutil.copy2(src, dst)


def _rewrite_parquet_ids(
    src_pq: Path,
    dst_pq: Path,
    *,
    episode_index: int,
    index_start: int,
    task_index: int,
) -> int:
    pa, pq = _require_pq()
    table = pq.read_table(src_pq)
    T = int(table.num_rows)
    if T <= 0:
        raise ValueError(f"空 parquet: {src_pq}")
    cols = {name: table[name] for name in table.column_names}
    cols["episode_index"] = pa.array(
        np.full(T, int(episode_index), dtype=np.int64), type=pa.int64()
    )
    cols["index"] = pa.array(
        np.arange(int(index_start), int(index_start) + T, dtype=np.int64),
        type=pa.int64(),
    )
    cols["task_index"] = pa.array(
        np.full(T, int(task_index), dtype=np.int64), type=pa.int64()
    )
    if "frame_index" in cols:
        cols["frame_index"] = pa.array(np.arange(T, dtype=np.int64), type=pa.int64())
    dst_pq.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst_pq.with_suffix(".merge_tmp.parquet")
    pq.write_table(pa.table(cols), tmp)
    tmp.replace(dst_pq)
    return T


def default_merge_out_dir(sources: list[Path], name: str = "") -> Path:
    """Sibling dir next to first source: ``{parent}/{name or merged_lerobot}``."""
    src0 = Path(sources[0]).expanduser().resolve()
    parent = src0.parent
    raw = str(name or "").strip()
    if raw:
        p = Path(raw).expanduser()
        if p.is_absolute():
            return p.resolve()
        # bare name → under first source's parent
        if "/" not in raw and "\\" not in raw:
            return (parent / raw).resolve()
        return (parent / p).resolve()
    return (parent / "merged_lerobot").resolve()


def merge_lerobot_datasets(
    sources: list[str | Path],
    out_dir: str | Path,
    *,
    overwrite: bool = False,
    progress_cb: ProgressCb = None,
) -> dict:
    """Merge sources in order into ``out_dir``. Returns summary dict."""
    src_paths = [Path(s).expanduser().resolve() for s in sources]
    dst = Path(out_dir).expanduser().resolve()
    ref_info = validate_lerobot_sources(src_paths)

    if dst in src_paths:
        raise ValueError(f"输出目录不能与任一源目录相同: {dst}")
    for s in src_paths:
        if dst == s or s in dst.parents:
            raise ValueError(f"输出目录不能落在源数据集内部: {dst}")

    total_eps_plan = 0
    for root in src_paths:
        total_eps_plan += len(list_episode_ids(root, load_info(root)))
    total_eps_plan = max(int(total_eps_plan), 1)

    def _prog(frac: float, message: str, **extra) -> None:
        _emit(
            progress_cb,
            phase="merge",
            frac=float(max(0.0, min(1.0, frac))),
            message=str(message),
            n_keys=total_eps_plan,
            **extra,
        )

    _prog(0.0, f"准备合并 · {len(src_paths)} 源 / {total_eps_plan} ep → {dst.name}")

    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"输出目录已存在（可勾选覆盖）: {dst}")
        _prog(0.01, f"清理已有输出目录…\n{dst}")
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    vkeys = _video_keys_from_info(ref_info)
    chunks_size = int(ref_info.get("chunks_size") or 1000)
    fps = float(ref_info.get("fps") or 20.0)

    # task registry: task text -> index
    task_to_idx: dict[str, int] = {}
    tasks_out: list[dict] = []

    def _task_index_for(text: str) -> int:
        t = str(text or "").strip() or "Flatten and fold the cloth."
        if t not in task_to_idx:
            task_to_idx[t] = len(task_to_idx)
            tasks_out.append({"task_index": task_to_idx[t], "task": t})
        return int(task_to_idx[t])

    # seed tasks from all sources (stable order by first appearance)
    for root in src_paths:
        for row in _load_tasks(root):
            _task_index_for(str(row.get("task") or ""))

    episodes_meta: list[dict] = []
    stats_meta: list[dict] = []
    global_index = 0
    out_eid = 0
    done_eps = 0
    n_vid = max(len(vkeys), 1)

    for src_i, root in enumerate(src_paths):
        info = load_info(root)
        ids = list_episode_ids(root, info)
        ep_rows = _load_episode_rows(root)
        st_rows = _load_stats_rows(root)
        src_tasks = {
            int(r.get("task_index", 0)): str(r.get("task") or "")
            for r in _load_tasks(root)
        }
        if not src_tasks:
            src_tasks = {0: "Flatten and fold the cloth."}

        for eid in ids:
            # 进度：本条开始时的基数 + parquet/视频子步骤
            base = float(done_eps) / float(total_eps_plan)
            step = 1.0 / float(total_eps_plan)
            _prog(
                base + 0.05 * step,
                f"合并 {done_eps + 1}/{total_eps_plan} ({int(base * 100)}%) · "
                f"{root.name} ep{eid} → ep{out_eid} · 写 parquet…",
                key=f"src{src_i}/ep{eid}",
                key_i=done_eps + 1,
            )

            src_pq = episode_parquet_path(root, eid, info)
            if not src_pq.is_file():
                raise FileNotFoundError(f"缺少 parquet: {src_pq}")

            # resolve task text for this episode
            ep_row = ep_rows.get(int(eid), {})
            task_list = ep_row.get("tasks") or []
            if isinstance(task_list, list) and task_list:
                task_text = str(task_list[0])
            else:
                task_text = src_tasks.get(0, "Flatten and fold the cloth.")
            tidx = _task_index_for(task_text)

            dst_info_stub = {
                "chunks_size": chunks_size,
                "data_path": ref_info.get(
                    "data_path",
                    "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                ),
                "video_path": ref_info.get(
                    "video_path",
                    "videos/chunk-{episode_chunk:03d}/{video_key}/"
                    "episode_{episode_index:06d}.mp4",
                ),
            }
            dst_pq = episode_parquet_path(dst, out_eid, dst_info_stub)
            T = _rewrite_parquet_ids(
                src_pq,
                dst_pq,
                episode_index=out_eid,
                index_start=global_index,
                task_index=tidx,
            )

            for vi, vk in enumerate(vkeys):
                src_v = episode_video_path(root, eid, vk, info)
                if not src_v.is_file():
                    raise FileNotFoundError(f"缺少视频: {src_v}")
                dst_v = episode_video_path(dst, out_eid, vk, dst_info_stub)
                _prog(
                    base + (0.25 + 0.7 * (vi + 1) / n_vid) * step,
                    f"合并 {done_eps + 1}/{total_eps_plan} "
                    f"({int((base + 0.5 * step) * 100)}%) · "
                    f"{root.name} ep{eid} · 视频 {vk.split('.')[-1]} "
                    f"({vi + 1}/{n_vid})",
                    key=f"src{src_i}/ep{eid}/{vk}",
                    key_i=done_eps + 1,
                )
                _link_or_copy(src_v, dst_v)

            episodes_meta.append(
                {
                    "episode_index": int(out_eid),
                    "tasks": [task_text],
                    "length": int(T),
                }
            )
            st = st_rows.get(int(eid))
            if isinstance(st, dict):
                st_out = dict(st)
                st_out["episode_index"] = int(out_eid)
                stats = st_out.setdefault("stats", {})
                for key in ("observation.state", "action"):
                    if key not in stats:
                        stats[key] = {
                            "min": [],
                            "max": [],
                            "mean": [],
                            "std": [],
                            "count": [int(T)],
                        }
                    else:
                        stats[key]["count"] = [int(T)]
                stats_meta.append(st_out)
            else:
                stats_meta.append(
                    {
                        "episode_index": int(out_eid),
                        "stats": {
                            "observation.state": {
                                "min": [],
                                "max": [],
                                "mean": [],
                                "std": [],
                                "count": [int(T)],
                            },
                            "action": {
                                "min": [],
                                "max": [],
                                "mean": [],
                                "std": [],
                                "count": [int(T)],
                            },
                        },
                    }
                )

            global_index += T
            out_eid += 1
            done_eps += 1
            _prog(
                float(done_eps) / float(total_eps_plan),
                f"合并 {done_eps}/{total_eps_plan} "
                f"({int(100 * done_eps / total_eps_plan)}%) · "
                f"{root.name} ep{eid} → ep{out_eid - 1} (T={T})",
                key=f"src{src_i}/ep{eid}",
                key_i=done_eps,
            )

    _prog(0.98, f"写 meta（info / episodes / tasks / stats）· {out_eid} ep…")

    meta_dir = dst / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    total_eps = len(episodes_meta)
    total_frames = int(global_index)
    total_chunks = int((total_eps + chunks_size - 1) // max(chunks_size, 1)) if total_eps else 1

    info_out = {
        "codebase_version": ref_info.get("codebase_version") or "v2.1",
        "robot_type": ref_info.get("robot_type") or "agilex",
        "total_episodes": total_eps,
        "total_frames": total_frames,
        "total_tasks": len(tasks_out),
        "total_videos": total_eps * len(vkeys),
        "total_chunks": max(total_chunks, 1),
        "chunks_size": chunks_size,
        "fps": int(fps) if abs(float(fps) - round(float(fps))) < 1e-9 else float(fps),
        "splits": {"train": f"0:{total_eps}"},
        "data_path": ref_info.get(
            "data_path",
            "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        ),
        "video_path": ref_info.get(
            "video_path",
            "videos/chunk-{episode_chunk:03d}/{video_key}/"
            "episode_{episode_index:06d}.mp4",
        ),
        "features": ref_info.get("features") or {},
    }
    with open(meta_dir / "info.json", "w", encoding="utf-8") as f:
        json.dump(info_out, f, indent=2)

    with open(meta_dir / "episodes.jsonl", "w", encoding="utf-8") as f:
        for row in episodes_meta:
            f.write(json.dumps(row) + "\n")

    with open(meta_dir / "tasks.jsonl", "w", encoding="utf-8") as f:
        for row in tasks_out:
            f.write(json.dumps(row) + "\n")

    with open(meta_dir / "episodes_stats.jsonl", "w", encoding="utf-8") as f:
        for row in stats_meta:
            f.write(json.dumps(row) + "\n")

    # sanity: must be detectable as lerobot
    if not is_lerobot_dataset(dst):
        raise RuntimeError(f"合并后校验失败（不是合法 LeRobot）: {dst}")

    _emit(
        progress_cb,
        phase="done",
        frac=1.0,
        message=(
            f"合并完成 · {total_eps} ep / {total_frames} frames · "
            f"{len(src_paths)} 源 → {dst}"
        ),
    )
    return {
        "out_dir": str(dst),
        "n_sources": len(src_paths),
        "n_episodes": total_eps,
        "n_frames": total_frames,
        "n_tasks": len(tasks_out),
        "video_keys": vkeys,
    }
