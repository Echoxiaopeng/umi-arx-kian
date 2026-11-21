#!/usr/bin/env python3
"""Trim leftover end-of-episode gripper open-close from a LeRobot dataset.

默认 ``trim``：从开合起点删掉 parquet + 视频（训练图/动作一致）。
也可用网页数据管理「一键裁剪末尾开合」。

Example:
  conda run -n umi-arx python scripts/trim_end_grip_gesture_lerobot.py \\
    /path/to/dataset_lerobot --dry-run

  conda run -n umi-arx python scripts/trim_end_grip_gesture_lerobot.py \\
    /path/to/dataset_lerobot --apply
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.end_grip_gesture_trim import (  # noqa: E402
    default_grip_thresholds,
    find_end_grip_gesture_cut,
)
from modules.lerobot_dataset_io import (  # noqa: E402
    OfflineLerobotEnv,
    episode_parquet_path,
    list_episode_ids,
    load_info,
    trim_lerobot_episode,
)


def _load_grip(pq_path: Path, grip_idx: int) -> np.ndarray:
    import pyarrow.parquet as pq

    table = pq.read_table(pq_path, columns=["action"])
    acts = [
        np.asarray(x, dtype=np.float64) for x in table.column("action").to_pylist()
    ]
    if not acts:
        return np.zeros((0,), dtype=np.float64)
    arr = np.stack(acts, axis=0)
    return arr[:, int(grip_idx)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--grip-idx", type=int, default=13)
    ap.add_argument("--gripper-min", type=float, default=0.011)
    ap.add_argument("--gripper-max", type=float, default=0.085)
    ap.add_argument("--window-s", type=float, default=1.5)
    ap.add_argument("--search-s", type=float, default=3.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if args.dry_run == args.apply:
        raise SystemExit("请指定 --dry-run 或 --apply 之一")

    root = Path(args.dataset)
    info = load_info(root)
    fps = float(info.get("fps") or 20.0)
    close_thresh, open_thresh = default_grip_thresholds(
        args.gripper_min, args.gripper_max
    )
    ids = list_episode_ids(root, info)

    print(f"dataset: {root}")
    print(f"episodes: {len(ids)}  fps={fps}  grip_idx={args.grip_idx}")
    print(f"thresh: close≤{close_thresh:.4f}  open≥{open_thresh:.4f}")
    print(f"run: {'DRY-RUN' if args.dry_run else 'APPLY'}")

    plans: list[tuple[int, int, int, int]] = []
    for eid in ids:
        pq_path = episode_parquet_path(root, eid, info)
        if not pq_path.is_file():
            continue
        grip = _load_grip(pq_path, args.grip_idx)
        n = int(grip.shape[0])
        cut = find_end_grip_gesture_cut(
            grip,
            fps=fps,
            close_thresh=close_thresh,
            open_thresh=open_thresh,
            window_s=float(args.window_s),
            search_s=float(args.search_s),
        )
        if cut is None:
            continue
        plans.append((int(eid), n, int(cut), n - int(cut)))

    if args.limit and args.limit > 0:
        plans = plans[: int(args.limit)]

    if not plans:
        print("未发现需要裁剪的末尾夹爪开合。")
        return

    drops = np.asarray([p[3] for p in plans], dtype=np.float64)
    print(
        f"将裁剪 {len(plans)}/{len(ids)} 条 · "
        f"丢帧 min/mean/max = {drops.min():.0f}/{drops.mean():.1f}/{drops.max():.0f} "
        f"({drops.mean() / fps:.2f}s avg)"
    )
    for eid, n, cut, drop in plans[:20]:
        print(f"  ep{eid}: keep [0,{cut})  drop {drop} / {n}  ({drop / fps:.2f}s)")
    if len(plans) > 20:
        print(f"  ... +{len(plans) - 20} more")

    if args.dry_run:
        report = root / "trim_end_grip_gesture_dryrun.json"
        report.write_text(
            json.dumps(
                [
                    {
                        "episode_index": eid,
                        "n": n,
                        "keep_end": cut,
                        "drop": drop,
                    }
                    for eid, n, cut, drop in plans
                ],
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"dry-run report → {report}")
        return

    # multi-pass via OfflineLerobotEnv if available
    try:
        from modules.lerobot_dataset_io import OfflineLerobotEnv  # type: ignore
    except Exception:
        OfflineLerobotEnv = None  # type: ignore

    if OfflineLerobotEnv is not None:
        from modules.end_grip_gesture_trim import apply_end_grip_trims

        env = OfflineLerobotEnv(root)

        def _trim(env_, eid, start, end, progress_cb=None):
            return trim_lerobot_episode(
                Path(env_.output_dir), int(eid), int(start), int(end), progress_cb
            )

        summary = apply_end_grip_trims(
            env,
            grip_idx=args.grip_idx,
            gripper_min=args.gripper_min,
            gripper_max=args.gripper_max,
            window_s=float(args.window_s),
            search_s=float(args.search_s),
            trim_fn=_trim,
            progress_cb=lambda p: print(p.get("message") or ""),
        )
        print(
            f"完成：trimmed={summary['n_trimmed']} drop_frames={summary['n_drop_frames']} "
            f"passes={summary['n_passes']}"
        )
        return

    ok = 0
    for i, (eid, n, cut, drop) in enumerate(reversed(plans), 1):
        print(f"[{i}/{len(plans)}] trim ep{eid}: [0,{cut}) drop {drop}")
        trim_lerobot_episode(root, eid, 0, cut)
        ok += 1
    print(f"完成：已裁剪 {ok} 条 episode。")


if __name__ == "__main__":
    main()
