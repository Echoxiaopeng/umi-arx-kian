"""Split a DAgger UMI dataset into human-intervention and policy-only folders.

Human path:
  - take contiguous intervention==1 segments
  - trim leading + trailing still plateaus via dual-arm eef_pos
    (enter-drag settle / exit-drag wait before resume)
  - write new replay_buffer.zarr (+ all-ones intervention sidecars)

Policy path:
  - copy full episodes whose intervention labels are all 0
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import zarr

from modules.replay_buffer import ReplayBuffer

ProgressCb = Optional[Callable[[dict], None]]

# Still-plateau trim at human-segment edges (eef_pos)
SPEED_THR = 0.0010
MIN_PLATEAU = 15
SEARCH_HORIZON = 100
MIN_MOVE_AFTER = 20
SMOOTH_W = 5
MIN_KEEP = 30
MIN_EP_DISP = 0.02


def human_segments(labels: np.ndarray) -> list[tuple[int, int]]:
    lab = np.asarray(labels, dtype=np.uint8).reshape(-1)
    segs: list[tuple[int, int]] = []
    in_h = False
    start = 0
    for t, v in enumerate(lab):
        if v and not in_h:
            in_h = True
            start = t
        elif (not v) and in_h:
            in_h = False
            segs.append((start, t))
    if in_h:
        segs.append((start, len(lab)))
    return segs


def arm_speed(p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    return np.maximum(
        np.linalg.norm(np.diff(p0, axis=0, prepend=p0[:1]), axis=-1),
        np.linalg.norm(np.diff(p1, axis=0, prepend=p1[:1]), axis=-1),
    )


def _smooth(x: np.ndarray, w: int = SMOOTH_W) -> np.ndarray:
    if w <= 1:
        return x
    ker = np.ones(w, dtype=np.float64) / w
    return np.convolve(x, ker, mode="same")


def leading_plateau_trim(p0: np.ndarray, p1: np.ndarray | None = None) -> int:
    """End index of last leading still-plateau based on eef_pos speed."""
    if p1 is None:
        p1 = p0
    spd = _smooth(arm_speed(p0, p1), SMOOTH_W)
    T = len(spd)
    if T == 0:
        return 0
    still = spd < SPEED_THR
    runs: list[tuple[int, int]] = []
    t = 0
    while t < T:
        if still[t]:
            a = t
            while t < T and still[t]:
                t += 1
            runs.append((a, t))
        else:
            t += 1
    cands: list[tuple[int, int]] = []
    for a, b in runs:
        if (b - a) < MIN_PLATEAU or a > SEARCH_HORIZON:
            continue
        after = spd[b : b + MIN_MOVE_AFTER]
        if len(after) < MIN_MOVE_AFTER:
            continue
        if float(np.mean(after > SPEED_THR)) >= 0.7:
            cands.append((a, b))
    if cands:
        return int(cands[-1][1])
    move = ~still
    t = 0
    while t < T:
        if move[t]:
            a = t
            while t < T and move[t]:
                t += 1
            if (t - a) >= MIN_MOVE_AFTER:
                return 0 if a == 0 else int(a)
        else:
            t += 1
    return 0


def trailing_plateau_trim(p0: np.ndarray, p1: np.ndarray | None = None) -> int:
    """Number of trailing still frames to drop (exit-drag / pre-resume pause)."""
    if p1 is None:
        p1 = p0
    if len(p0) == 0:
        return 0
    return int(leading_plateau_trim(p0[::-1], p1[::-1]))


def edge_still_trims(
    p0: np.ndarray, p1: np.ndarray | None = None, *, min_keep: int = MIN_KEEP
) -> tuple[int, int]:
    """Leading and trailing still cuts that leave at least ``min_keep`` frames."""
    if p1 is None:
        p1 = p0
    T = int(len(p0))
    if T <= 0:
        return 0, 0
    lead = leading_plateau_trim(p0, p1)
    trail = trailing_plateau_trim(p0, p1)
    keep = T - lead - trail
    if keep >= min_keep:
        return int(lead), int(trail)
    # Prefer keeping motion: shrink trailing first, then leading.
    need = min_keep - keep
    shrink_t = min(trail, need)
    trail -= shrink_t
    need -= shrink_t
    if need > 0:
        lead = max(0, lead - need)
    if T - lead - trail < 1:
        return 0, 0
    return int(lead), int(trail)


def max_disp(p0: np.ndarray, p1: np.ndarray) -> float:
    if len(p0) == 0:
        return 0.0
    d0 = float(np.linalg.norm(p0 - p0[0], axis=-1).max())
    d1 = float(np.linalg.norm(p1 - p1[0], axis=-1).max()) if len(p1) else 0.0
    return max(d0, d1)


def dataset_has_intervention(src: str | Path) -> bool:
    inter = Path(src) / "intervention"
    if not inter.is_dir():
        return False
    return any(inter.glob("episode_*.npy"))


def inspect_intervention(src: str | Path) -> dict:
    """Quick stats for UI: n_ep, n_with_labels, n_policy, n_human_eps, unlabeled."""
    src = Path(src)
    zarr_path = src / "replay_buffer.zarr"
    inter_dir = src / "intervention"
    out = {
        "has_intervention": False,
        "n_episodes": 0,
        "n_labeled": 0,
        "n_policy_only": 0,
        "n_human_eps": 0,
        "n_unlabeled": 0,
        "n_mismatch": 0,
        "human_steps": 0,
        "policy_steps": 0,
    }
    if not zarr_path.exists() or not inter_dir.is_dir():
        return out
    buf = ReplayBuffer.create_from_path(str(zarr_path), mode="r")
    ends = np.asarray(buf.episode_ends[:], dtype=np.int64)
    ep_lens = np.diff(np.concatenate([[0], ends]))
    n_ep = len(ep_lens)
    out["n_episodes"] = int(n_ep)
    out["has_intervention"] = True
    mapping = build_label_mapping(src, ep_lens)
    for ep in range(n_ep):
        fid = mapping.get(ep)
        if fid is None:
            out["n_unlabeled"] += 1
            continue
        lab = np.load(inter_dir / f"episode_{fid}.npy")
        if len(lab) != int(ep_lens[ep]):
            out["n_mismatch"] += 1
            continue
        out["n_labeled"] += 1
        h = int(np.asarray(lab).sum())
        if h == 0:
            out["n_policy_only"] += 1
            out["policy_steps"] += int(ep_lens[ep])
        else:
            out["n_human_eps"] += 1
            out["human_steps"] += h
    return out


def build_label_mapping(
    src: str | Path, ep_lens: np.ndarray
) -> dict[int, int | None]:
    """Map episode id -> intervention file id.

    Prefer identity when lengths match. For early abort-induced shifts,
    greedily assign unused same-length files nearest to the episode id.
    """
    src = Path(src)
    inter_dir = src / "intervention"
    n = len(ep_lens)
    labs: dict[int, np.ndarray] = {}
    for p in inter_dir.glob("episode_*.npy"):
        try:
            fid = int(p.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        labs[fid] = np.load(p)

    mapping: dict[int, int | None] = {i: None for i in range(n)}
    used: set[int] = set()

    # Pass 1: exact identity matches
    for ep in range(n):
        if ep in labs and len(labs[ep]) == int(ep_lens[ep]):
            mapping[ep] = ep
            used.add(ep)

    # Pass 2: fill remaining by nearest same-length unused file
    for ep in range(n):
        if mapping[ep] is not None:
            continue
        T = int(ep_lens[ep])
        cands = [j for j, lab in labs.items() if j not in used and len(lab) == T]
        if not cands:
            continue
        cands.sort(key=lambda j: (abs(j - ep), j))
        j = cands[0]
        # Avoid stealing a file that is the identity match for another ep
        # which still needs it (already assigned in pass 1).
        mapping[ep] = j
        used.add(j)

    return mapping


def _create_empty_dst(path: Path) -> ReplayBuffer:
    if path.exists():
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    store = zarr.DirectoryStore(str(path))
    root = zarr.group(store=store, overwrite=True)
    return ReplayBuffer.create_empty_zarr(root=root)


def _emit(cb: ProgressCb, **info) -> None:
    if cb is not None:
        cb(dict(info))


def extract_dagger_split(
    src: str | Path,
    *,
    dst_human: str | Path | None = None,
    dst_policy: str | Path | None = None,
    what: str = "both",
    overwrite: bool = False,
    trim_still: bool = True,
    progress_cb: ProgressCb = None,
) -> dict:
    """Extract human and/or policy subsets.

    ``what``: ``human`` | ``policy`` | ``both``.
    Returns a result dict with paths and counts.
    """
    src = Path(src).expanduser().resolve()
    what = str(what).lower().strip()
    if what not in ("human", "policy", "both"):
        raise ValueError(f"what must be human|policy|both, got {what!r}")

    src_zarr = src / "replay_buffer.zarr"
    inter_dir = src / "intervention"
    if not src_zarr.exists():
        raise FileNotFoundError(f"missing {src_zarr}")
    if not inter_dir.is_dir():
        raise FileNotFoundError(f"missing intervention dir: {inter_dir}")

    if dst_human is None:
        dst_human = src.parent / f"{src.name}_human"
    else:
        dst_human = Path(dst_human).expanduser().resolve()
    if dst_policy is None:
        dst_policy = src.parent / f"{src.name}_policy"
    else:
        dst_policy = Path(dst_policy).expanduser().resolve()

    do_human = what in ("human", "both")
    do_policy = what in ("policy", "both")

    for dst, enable in ((dst_human, do_human), (dst_policy, do_policy)):
        if not enable:
            continue
        if dst.exists() and any(dst.iterdir()):
            if not overwrite:
                raise FileExistsError(
                    f"输出目录已存在且非空: {dst}（可勾选覆盖，或换路径）"
                )
            shutil.rmtree(dst)

    _emit(
        progress_cb,
        phase="start",
        frac=0.0,
        message=f"打开数据集 {src.name}…",
    )
    src_buf = ReplayBuffer.create_from_path(str(src_zarr), mode="r")
    ends = np.asarray(src_buf.episode_ends[:], dtype=np.int64)
    ep_lens = np.diff(np.concatenate([[0], ends]))
    n_ep = len(ep_lens)
    mapping = build_label_mapping(src, ep_lens)

    labs: dict[int, np.ndarray] = {}
    for fid in set(v for v in mapping.values() if v is not None):
        labs[int(fid)] = np.load(inter_dir / f"episode_{int(fid)}.npy")

    policy_eps: list[int] = []
    human_plan: list[tuple[int, int, list[tuple[int, int]]]] = []
    unlabeled: list[int] = []
    for ep in range(n_ep):
        fid = mapping.get(ep)
        if fid is None:
            unlabeled.append(ep)
            continue
        lab = labs[int(fid)]
        if len(lab) != int(ep_lens[ep]):
            unlabeled.append(ep)
            continue
        h = int(np.asarray(lab).sum())
        if h == 0:
            policy_eps.append(ep)
        else:
            human_plan.append((ep, int(fid), human_segments(lab)))

    chunks = src_buf.get_chunks()
    compressors = src_buf.get_compressors()
    result: dict = {
        "src": str(src),
        "what": what,
        "unlabeled_episodes": unlabeled,
        "n_src_episodes": n_ep,
        "human": None,
        "policy": None,
    }

    # -------- human --------
    if do_human:
        _emit(
            progress_cb,
            phase="human",
            frac=0.05,
            message=f"提取人工示教（{len(human_plan)} eps）→ {dst_human.name}",
        )
        dst_human.mkdir(parents=True, exist_ok=True)
        human_buf = _create_empty_dst(dst_human / "replay_buffer.zarr")
        human_manifest = []
        human_dropped = []
        kept_steps = 0
        trim_steps = 0
        n_plan = max(len(human_plan), 1)
        for pi, (ep_id, fid, segs) in enumerate(human_plan):
            sl = src_buf.get_episode_slice(ep_id)
            for seg_i, (a, b) in enumerate(segs):
                seg = {
                    k: np.asarray(v)
                    for k, v in src_buf.get_steps_slice(
                        sl.start + a, sl.start + b, copy=True
                    ).items()
                }
                T0 = b - a
                if trim_still and "robot0_eef_pos" in seg:
                    p0 = seg["robot0_eef_pos"]
                    p1 = seg.get("robot1_eef_pos", p0)
                    trim_l, trim_t = edge_still_trims(p0, p1)
                else:
                    trim_l, trim_t = 0, 0
                kept = {k: v[trim_l : T0 - trim_t] for k, v in seg.items()}
                keep_n = T0 - trim_l - trim_t
                disp = 0.0
                if keep_n > 0 and "robot0_eef_pos" in kept:
                    disp = max_disp(
                        kept["robot0_eef_pos"],
                        kept.get("robot1_eef_pos", kept["robot0_eef_pos"]),
                    )
                if keep_n < MIN_KEEP or (trim_still and disp < MIN_EP_DISP):
                    human_dropped.append(
                        {
                            "src_episode_id": ep_id,
                            "seg_index": seg_i,
                            "orig_seg_steps": T0,
                            "trim_leading": trim_l,
                            "trim_trailing": trim_t,
                            "keep": keep_n,
                            "max_disp": disp,
                        }
                    )
                    continue
                new_id = human_buf.n_episodes
                human_buf.add_episode(
                    kept, chunks=chunks, compressors=compressors
                )
                kept_steps += keep_n
                trim_steps += trim_l + trim_t
                human_manifest.append(
                    {
                        "new_episode_id": int(new_id),
                        "src_episode_id": int(ep_id),
                        "label_file_id": int(fid),
                        "seg_index": int(seg_i),
                        "src_local_start": int(a),
                        "src_local_end": int(b),
                        "trim_leading": int(trim_l),
                        "trim_trailing": int(trim_t),
                        "kept_steps": int(keep_n),
                        "max_disp_m": float(disp),
                    }
                )
            frac = 0.05 + 0.55 * float(pi + 1) / float(n_plan)
            _emit(
                progress_cb,
                phase="human",
                frac=frac,
                message=(
                    f"人工示教 {pi + 1}/{len(human_plan)} · "
                    f"已写 {len(human_manifest)} 段"
                ),
            )

        inter_out = dst_human / "intervention"
        inter_out.mkdir(parents=True, exist_ok=True)
        for e in human_manifest:
            np.save(
                inter_out / f"episode_{e['new_episode_id']}.npy",
                np.ones((e["kept_steps"],), dtype=np.uint8),
            )
        human_meta = {
            "src": str(src),
            "dst": str(dst_human),
            "method": (
                "intervention==1 segments"
                + (
                    " + eef_pos leading/trailing still-plateau trim"
                    if trim_still
                    else ""
                )
            ),
            "trim_params": {
                "enabled": bool(trim_still),
                "edges": ["leading", "trailing"],
                "speed_thr_m_per_frame": SPEED_THR,
                "min_plateau": MIN_PLATEAU,
                "search_horizon": SEARCH_HORIZON,
                "min_move_after": MIN_MOVE_AFTER,
                "min_keep": MIN_KEEP,
                "min_ep_disp_m": MIN_EP_DISP,
            },
            "n_src_episodes": n_ep,
            "n_human_episodes": len(human_manifest),
            "n_dropped": len(human_dropped),
            "total_kept_steps": kept_steps,
            "total_trim_steps": trim_steps,
            "unlabeled_episodes": unlabeled,
            "label_mapping": {str(k): v for k, v in mapping.items()},
            "dropped": human_dropped,
            "episodes": human_manifest,
        }
        with open(
            dst_human / "human_extract_manifest.json", "w", encoding="utf-8"
        ) as f:
            json.dump(human_meta, f, indent=2, ensure_ascii=False)
        result["human"] = {
            "dst": str(dst_human),
            "n_episodes": len(human_manifest),
            "n_steps": kept_steps,
            "n_dropped": len(human_dropped),
            "trim_steps": trim_steps,
        }

    # -------- policy --------
    if do_policy:
        _emit(
            progress_cb,
            phase="policy",
            frac=0.65,
            message=f"提取纯模型（{len(policy_eps)} eps）→ {dst_policy.name}",
        )
        dst_policy.mkdir(parents=True, exist_ok=True)
        policy_buf = _create_empty_dst(dst_policy / "replay_buffer.zarr")
        policy_manifest = []
        policy_steps = 0
        n_pol = max(len(policy_eps), 1)
        for pi, ep_id in enumerate(policy_eps):
            ep = {
                k: np.asarray(v)
                for k, v in src_buf.get_episode(ep_id, copy=True).items()
            }
            new_id = policy_buf.n_episodes
            policy_buf.add_episode(ep, chunks=chunks, compressors=compressors)
            n = len(next(iter(ep.values())))
            policy_steps += n
            policy_manifest.append(
                {
                    "new_episode_id": int(new_id),
                    "src_episode_id": int(ep_id),
                    "label_file_id": int(mapping[ep_id]),
                    "n_steps": int(n),
                }
            )
            frac = 0.65 + 0.30 * float(pi + 1) / float(n_pol)
            _emit(
                progress_cb,
                phase="policy",
                frac=frac,
                message=(
                    f"纯模型 {pi + 1}/{len(policy_eps)} · "
                    f"已写 {len(policy_manifest)} 条"
                ),
            )

        inter_out = dst_policy / "intervention"
        inter_out.mkdir(parents=True, exist_ok=True)
        for e in policy_manifest:
            np.save(
                inter_out / f"episode_{e['new_episode_id']}.npy",
                np.zeros((e["n_steps"],), dtype=np.uint8),
            )
        policy_meta = {
            "src": str(src),
            "dst": str(dst_policy),
            "method": "full episodes with intervention all 0 (policy-only)",
            "n_src_episodes": n_ep,
            "n_policy_episodes": len(policy_manifest),
            "total_steps": policy_steps,
            "unlabeled_skipped": unlabeled,
            "episodes": policy_manifest,
        }
        with open(
            dst_policy / "policy_extract_manifest.json", "w", encoding="utf-8"
        ) as f:
            json.dump(policy_meta, f, indent=2, ensure_ascii=False)
        result["policy"] = {
            "dst": str(dst_policy),
            "n_episodes": len(policy_manifest),
            "n_steps": policy_steps,
        }

    _emit(progress_cb, phase="done", frac=1.0, message="DAgger 拆分完成")
    return result


def default_split_paths(src: str | Path) -> tuple[Path, Path]:
    src = Path(src).expanduser().resolve()
    return (
        src.parent / f"{src.name}_human",
        src.parent / f"{src.name}_policy",
    )


def transition_still_drop_mask(
    p0: np.ndarray,
    p1: np.ndarray,
    labels: np.ndarray,
    *,
    trim_policy_resume: bool = True,
    min_keep_seg: int = MIN_KEEP,
) -> np.ndarray:
    """Bool mask over one episode: True = drop transition still frames.

    Drops:
      - leading / trailing still inside each intervention==1 segment
        (enter-drag settle / exit-drag pause)
      - leading still of the policy run right after a human segment
        (soft-resume standstill), when ``trim_policy_resume``
    Keeps both policy and human motion; does not split episodes.
    """
    lab = np.asarray(labels, dtype=np.uint8).reshape(-1)
    T = len(lab)
    drop = np.zeros(T, dtype=bool)
    if T == 0:
        return drop
    p0 = np.asarray(p0)
    p1 = np.asarray(p1)
    for a, b in human_segments(lab):
        tl, tt = edge_still_trims(p0[a:b], p1[a:b], min_keep=min_keep_seg)
        if tl > 0:
            drop[a : a + tl] = True
        if tt > 0:
            drop[b - tt : b] = True
        if not trim_policy_resume or b >= T or int(lab[b]) != 0:
            continue
        c = b
        while c < T and int(lab[c]) == 0:
            c += 1
        pol_n = c - b
        if pol_n <= min_keep_seg:
            continue
        pt = leading_plateau_trim(p0[b:c], p1[b:c])
        if pol_n - pt < min_keep_seg:
            pt = max(0, pol_n - min_keep_seg)
        if pt > 0:
            drop[b : b + pt] = True
    return drop


def trim_dagger_transition_stills(
    src: str | Path,
    dst: str | Path | None = None,
    *,
    overwrite: bool = False,
    trim_policy_resume: bool = True,
    progress_cb: ProgressCb = None,
) -> dict:
    """Rewrite a mixed DAgger dataset with transition stills removed.

    Preserves episode order and both policy / human frames (minus still
    plateaus at enter-drag, exit-drag, and optional policy soft-resume).
    Writes ``replay_buffer.zarr`` + ``intervention/``; does not rewrite
    ``videos/`` (zarr RGB is the training source).
    """
    src = Path(src).expanduser().resolve()
    if dst is None:
        dst = src
    else:
        dst = Path(dst).expanduser().resolve()

    src_zarr = src / "replay_buffer.zarr"
    inter_dir = src / "intervention"
    if not src_zarr.exists():
        raise FileNotFoundError(f"missing {src_zarr}")
    if not inter_dir.is_dir():
        raise FileNotFoundError(f"missing intervention dir: {inter_dir}")

    in_place = dst.resolve() == src.resolve()
    build = (
        src.parent / f".{src.name}_stilltrim_building"
        if in_place
        else dst
    )
    if build.exists() and any(build.iterdir()):
        if not overwrite and not in_place:
            raise FileExistsError(
                f"输出目录已存在且非空: {build}（可 --overwrite）"
            )
        shutil.rmtree(build)
    build.mkdir(parents=True, exist_ok=True)

    _emit(
        progress_cb,
        phase="start",
        frac=0.0,
        message=f"打开数据集 {src.name}…",
    )
    src_buf = ReplayBuffer.create_from_path(str(src_zarr), mode="r")
    ends = np.asarray(src_buf.episode_ends[:], dtype=np.int64)
    ep_lens = np.diff(np.concatenate([[0], ends]))
    n_ep = len(ep_lens)
    mapping = build_label_mapping(src, ep_lens)

    labs: dict[int, np.ndarray] = {}
    for fid in {v for v in mapping.values() if v is not None}:
        labs[int(fid)] = np.load(inter_dir / f"episode_{int(fid)}.npy")

    chunks = src_buf.get_chunks()
    compressors = src_buf.get_compressors()
    out_buf = _create_empty_dst(build / "replay_buffer.zarr")
    inter_out = build / "intervention"
    inter_out.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    unlabeled: list[int] = []
    total_in = 0
    total_out = 0
    total_drop_h_lead = 0
    total_drop_h_trail = 0
    total_drop_pol = 0

    for ep in range(n_ep):
        sl = src_buf.get_episode_slice(ep)
        ep_data = {
            k: np.asarray(v)
            for k, v in src_buf.get_steps_slice(
                sl.start, sl.stop, copy=True
            ).items()
        }
        T = int(ep_lens[ep])
        total_in += T
        fid = mapping.get(ep)
        drop_h_lead = drop_h_trail = drop_pol = 0

        if fid is None or int(fid) not in labs or len(labs[int(fid)]) != T:
            unlabeled.append(ep)
            keep_data = ep_data
            keep_lab = None
            n_keep = T
        else:
            lab = np.asarray(labs[int(fid)], dtype=np.uint8).reshape(-1)
            p0 = ep_data["robot0_eef_pos"]
            p1 = ep_data.get("robot1_eef_pos", p0)
            drop = transition_still_drop_mask(
                p0, p1, lab, trim_policy_resume=trim_policy_resume
            )
            # Count drop categories for logging
            for a, b in human_segments(lab):
                tl, tt = edge_still_trims(p0[a:b], p1[a:b])
                drop_h_lead += int(tl)
                drop_h_trail += int(tt)
                if trim_policy_resume and b < T and int(lab[b]) == 0:
                    c = b
                    while c < T and int(lab[c]) == 0:
                        c += 1
                    pol_n = c - b
                    if pol_n > MIN_KEEP:
                        pt = leading_plateau_trim(p0[b:c], p1[b:c])
                        if pol_n - pt < MIN_KEEP:
                            pt = max(0, pol_n - MIN_KEEP)
                        drop_pol += int(pt)
            keep = ~drop
            n_keep = int(keep.sum())
            if n_keep <= 0:
                # Degenerate — keep original
                keep_data = ep_data
                keep_lab = lab
                n_keep = T
                drop_h_lead = drop_h_trail = drop_pol = 0
            else:
                keep_data = {k: v[keep] for k, v in ep_data.items()}
                keep_lab = lab[keep]

        new_id = out_buf.n_episodes
        out_buf.add_episode(
            keep_data, chunks=chunks, compressors=compressors
        )
        if keep_lab is None:
            # unlabeled: no sidecar
            pass
        else:
            np.save(
                inter_out / f"episode_{new_id}.npy",
                np.asarray(keep_lab, dtype=np.uint8),
            )
        total_out += n_keep
        total_drop_h_lead += drop_h_lead
        total_drop_h_trail += drop_h_trail
        total_drop_pol += drop_pol
        manifest.append(
            {
                "new_episode_id": int(new_id),
                "src_episode_id": int(ep),
                "label_file_id": None if fid is None else int(fid),
                "src_steps": int(T),
                "kept_steps": int(n_keep),
                "drop_human_leading": int(drop_h_lead),
                "drop_human_trailing": int(drop_h_trail),
                "drop_policy_resume": int(drop_pol),
            }
        )
        frac = 0.05 + 0.9 * float(ep + 1) / float(max(n_ep, 1))
        _emit(
            progress_cb,
            phase="trim",
            frac=frac,
            message=f"裁静止 {ep + 1}/{n_ep} · 已写 {new_id + 1} eps",
        )

    meta = {
        "src": str(src),
        "dst": str(dst),
        "method": (
            "mixed DAgger: drop enter/exit-drag still"
            + (" + policy soft-resume still" if trim_policy_resume else "")
            + " (keep policy+human, no split)"
        ),
        "trim_params": {
            "speed_thr_m_per_frame": SPEED_THR,
            "min_plateau": MIN_PLATEAU,
            "search_horizon": SEARCH_HORIZON,
            "min_move_after": MIN_MOVE_AFTER,
            "min_keep": MIN_KEEP,
            "trim_policy_resume": bool(trim_policy_resume),
        },
        "n_src_episodes": n_ep,
        "n_out_episodes": len(manifest),
        "total_src_steps": total_in,
        "total_kept_steps": total_out,
        "drop_human_leading": total_drop_h_lead,
        "drop_human_trailing": total_drop_h_trail,
        "drop_policy_resume": total_drop_pol,
        "unlabeled_episodes": unlabeled,
        "videos_note": "videos/ not rewritten; use zarr camera_*_rgb",
        "episodes": manifest,
    }
    with open(build / "still_trim_manifest.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    if in_place:
        bak_z = src / "replay_buffer.zarr.bak_before_still_trim"
        bak_i = src / "intervention.bak_before_still_trim"
        if bak_z.exists():
            shutil.rmtree(bak_z)
        if bak_i.exists():
            shutil.rmtree(bak_i)
        (src / "replay_buffer.zarr").rename(bak_z)
        (src / "intervention").rename(bak_i)
        (build / "replay_buffer.zarr").rename(src / "replay_buffer.zarr")
        (build / "intervention").rename(src / "intervention")
        shutil.move(
            str(build / "still_trim_manifest.json"),
            str(src / "still_trim_manifest.json"),
        )
        if build.exists():
            shutil.rmtree(build, ignore_errors=True)
        final_dst = src
    else:
        final_dst = build

    _emit(progress_cb, phase="done", frac=1.0, message="静止裁剪完成")
    return {
        "src": str(src),
        "dst": str(final_dst),
        "n_episodes": len(manifest),
        "src_steps": total_in,
        "kept_steps": total_out,
        "drop_human_leading": total_drop_h_lead,
        "drop_human_trailing": total_drop_h_trail,
        "drop_policy_resume": total_drop_pol,
        "unlabeled_episodes": unlabeled,
        "bak_zarr": str(src / "replay_buffer.zarr.bak_before_still_trim")
        if in_place
        else None,
    }
