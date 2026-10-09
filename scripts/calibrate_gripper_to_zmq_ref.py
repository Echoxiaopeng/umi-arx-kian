#!/usr/bin/env python3
"""把 as_zmq 数据集的夹爪开度仿射对齐到 collect_demo_zmq。

用同一零件抓取时的实测开度做对应点，再加张开锚点（p99），拟合:
    g' = a * g + b
写回 robot0_gripper_width 与 action[:, 6]，并 clip 到参考范围。

默认测点（继电器 / 电感 / 电容）:
  demo_as_zmq:  0.0277, 0.0257, 0.0222
  pro_as_zmq:   0.0183, 0.0164, 0.0115
  collect_demo_zmq (目标): 0.0095, 0.0068, 0.0008

用法:
  # dry-run
  python scripts/calibrate_gripper_to_zmq_ref.py \\
      -i data/collect_demo_as_zmq/replay_buffer.zarr \\
      --preset demo --dry_run

  # 写盘
  python scripts/calibrate_gripper_to_zmq_ref.py \\
      -i data/collect_demo_as_zmq/replay_buffer.zarr \\
      --preset demo

  python scripts/calibrate_gripper_to_zmq_ref.py \\
      -i data/collect_demo_pro_as_zmq/replay_buffer.zarr \\
      --preset pro
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import numpy as np
import zarr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_REF = ROOT / "data" / "collect_demo_zmq" / "replay_buffer.zarr"

# 抓取开度测点: (relay, inductor, capacitor)
PRESETS = {
    "demo": {
        "src": (0.0277, 0.0257, 0.0222),
        "dst": (0.0095, 0.0068, 0.0008),
        "labels": ("relay", "inductor", "capacitor"),
    },
    "pro": {
        "src": (0.0183, 0.0164, 0.0115),
        "dst": (0.0095, 0.0068, 0.0008),
        "labels": ("relay", "inductor", "capacitor"),
    },
}

OBJECT_NAMES = ("relay", "inductor", "capacitor")


def _parse_triplet(text: str | None, default: tuple[float, float, float]) -> np.ndarray:
    if text is None or str(text).strip() == "":
        return np.asarray(default, dtype=np.float64)
    vals = [float(x) for x in str(text).split(",") if x.strip() != ""]
    if len(vals) != 3:
        raise click.ClickException(f"需要 3 个数（逗号分隔），收到 {len(vals)}: {text}")
    return np.asarray(vals, dtype=np.float64)


def _grip_array(zarr_path: Path) -> np.ndarray:
    root = zarr.open(str(zarr_path), mode="r")
    return np.asarray(root["data/robot0_gripper_width"][:], dtype=np.float64)


def _percentile(arr: np.ndarray, q: float) -> float:
    return float(np.percentile(np.asarray(arr, dtype=np.float64), q))


def fit_affine(src_pts: np.ndarray, dst_pts: np.ndarray) -> tuple[float, float]:
    """最小二乘 y = a x + b。"""
    src_pts = np.asarray(src_pts, dtype=np.float64).reshape(-1)
    dst_pts = np.asarray(dst_pts, dtype=np.float64).reshape(-1)
    if src_pts.shape != dst_pts.shape or src_pts.size < 2:
        raise ValueError("仿射拟合至少需要 2 对点")
    A = np.vstack([src_pts, np.ones_like(src_pts)]).T
    a, b = np.linalg.lstsq(A, dst_pts, rcond=None)[0]
    return float(a), float(b)


def apply_affine(g: np.ndarray, a: float, b: float, clip_min: float, clip_max: float) -> np.ndarray:
    out = a * np.asarray(g, dtype=np.float64) + b
    return np.clip(out, clip_min, clip_max)


def summarize(name: str, g: np.ndarray) -> str:
    g = np.asarray(g, dtype=np.float64)
    return (
        f"{name}: min={g.min():.4f} max={g.max():.4f} mean={g.mean():.4f} "
        f"p5={np.percentile(g, 5):.4f} p99={np.percentile(g, 99):.4f}"
    )


@click.command()
@click.option(
    "--input",
    "-i",
    "input_zarr",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="待标定的 as_zmq replay_buffer.zarr",
)
@click.option(
    "--ref",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=DEFAULT_REF,
    show_default=True,
    help="参考 collect_demo_zmq replay_buffer.zarr",
)
@click.option(
    "--preset",
    type=click.Choice(["demo", "pro", "custom"]),
    default="demo",
    show_default=True,
    help="使用内置测点表；custom 时必须提供 --src_grasp/--dst_grasp",
)
@click.option(
    "--src_grasp",
    default="",
    help="源抓取开度 relay,inductor,capacitor（覆盖 preset）",
)
@click.option(
    "--dst_grasp",
    default="",
    help="目标抓取开度 relay,inductor,capacitor（覆盖 preset）",
)
@click.option(
    "--open_percentile",
    default=99.0,
    show_default=True,
    type=float,
    help="张开锚点百分位（源 pX → 参考 pX）",
)
@click.option(
    "--no_open_anchor",
    is_flag=True,
    default=False,
    help="不加入张开锚点，仅用 3 个抓取点拟合",
)
@click.option(
    "--clip_min",
    default=-0.001,
    show_default=True,
    type=float,
)
@click.option(
    "--clip_max",
    default=0.085,
    show_default=True,
    type=float,
)
@click.option(
    "--dry_run",
    is_flag=True,
    default=False,
    help="只打印拟合与映射统计，不写盘",
)
def main(
    input_zarr: Path,
    ref: Path,
    preset: str,
    src_grasp: str,
    dst_grasp: str,
    open_percentile: float,
    no_open_anchor: bool,
    clip_min: float,
    clip_max: float,
    dry_run: bool,
):
    input_zarr = input_zarr.resolve()
    ref = ref.resolve()

    if preset == "custom":
        if not src_grasp.strip() or not dst_grasp.strip():
            raise click.ClickException("preset=custom 时需要同时提供 --src_grasp 与 --dst_grasp")
        src_pts = _parse_triplet(src_grasp, (0.0, 0.0, 0.0))
        dst_pts = _parse_triplet(dst_grasp, (0.0, 0.0, 0.0))
        labels = OBJECT_NAMES
    else:
        cfg = PRESETS[preset]
        src_pts = _parse_triplet(src_grasp or None, cfg["src"])
        dst_pts = _parse_triplet(dst_grasp or None, cfg["dst"])
        labels = cfg["labels"]

    g_src = _grip_array(input_zarr)
    g_ref = _grip_array(ref)

    fit_src = list(src_pts)
    fit_dst = list(dst_pts)
    open_src = open_dst = None
    if not no_open_anchor:
        open_src = _percentile(g_src, open_percentile)
        open_dst = _percentile(g_ref, open_percentile)
        fit_src.append(open_src)
        fit_dst.append(open_dst)

    a, b = fit_affine(np.asarray(fit_src), np.asarray(fit_dst))
    mapped_grasp = apply_affine(src_pts, a, b, clip_min, clip_max)
    grasp_err = mapped_grasp - dst_pts
    grasp_rmse = float(np.sqrt(np.mean(grasp_err**2)))

    g_mapped = apply_affine(g_src, a, b, clip_min, clip_max)

    print("=" * 64)
    print("Calibrate gripper → collect_demo_zmq")
    print(f"  input : {input_zarr}")
    print(f"  ref   : {ref}")
    print(f"  preset: {preset}")
    print(f"  affine: g' = {a:.6f} * g + ({b:.6f})")
    print(f"  clip  : [{clip_min}, {clip_max}]")
    if open_src is not None:
        print(
            f"  open anchor: src p{open_percentile:g}={open_src:.4f} → "
            f"ref p{open_percentile:g}={open_dst:.4f}"
        )
    else:
        print("  open anchor: disabled")
    print("-" * 64)
    for name, s, d, m in zip(labels, src_pts, dst_pts, mapped_grasp):
        print(f"  {name:10s} src={s:.4f} → mapped={m:.4f}  target={d:.4f}  err={m-d:+.5f}")
    print(f"  grasp RMSE = {grasp_rmse:.6f}")
    print(summarize("before", g_src))
    print(summarize("after ", g_mapped))
    print(summarize("ref   ", g_ref))
    print("=" * 64)

    if grasp_rmse >= 1e-3:
        print(f"[warn] grasp RMSE {grasp_rmse:.6f} >= 1e-3")

    meta = {
        "input": str(input_zarr),
        "ref": str(ref),
        "preset": preset,
        "affine": {"a": a, "b": b},
        "clip": {"min": clip_min, "max": clip_max},
        "open_anchor": None
        if open_src is None
        else {
            "percentile": open_percentile,
            "src": open_src,
            "dst": open_dst,
        },
        "grasp_points": {
            "labels": list(labels),
            "src": src_pts.tolist(),
            "dst": dst_pts.tolist(),
            "mapped": mapped_grasp.tolist(),
            "rmse": grasp_rmse,
        },
        "stats": {
            "before": {
                "min": float(g_src.min()),
                "max": float(g_src.max()),
                "mean": float(g_src.mean()),
                "p99": _percentile(g_src, 99),
            },
            "after": {
                "min": float(g_mapped.min()),
                "max": float(g_mapped.max()),
                "mean": float(g_mapped.mean()),
                "p99": _percentile(g_mapped, 99),
            },
            "ref": {
                "min": float(g_ref.min()),
                "max": float(g_ref.max()),
                "mean": float(g_ref.mean()),
                "p99": _percentile(g_ref, 99),
            },
        },
        "dry_run": bool(dry_run),
    }

    if dry_run:
        print("dry_run: 未写盘")
        meta_path = input_zarr.parent / "gripper_calib_meta.dry_run.json"
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"Wrote {meta_path}")
        return

    root = zarr.open(str(input_zarr), mode="r+")
    action = np.asarray(root["data/action"][:], dtype=np.float64)
    if action.ndim != 2 or action.shape[1] < 7:
        raise click.ClickException(f"action shape 异常: {action.shape}")
    if action.shape[0] != g_mapped.shape[0]:
        raise click.ClickException(
            f"长度不一致: grip={g_mapped.shape[0]} action={action.shape[0]}"
        )

    root["data/robot0_gripper_width"][:] = g_mapped
    action[:, 6] = g_mapped
    root["data/action"][:] = action

    # verify
    g2 = np.asarray(root["data/robot0_gripper_width"][:])
    a6 = np.asarray(root["data/action"][:, 6])
    sync_err = float(np.abs(g2 - a6).max())
    print(f"[write] action sync max|diff| = {sync_err:.3e}")
    if sync_err > 1e-12:
        raise click.ClickException("写盘后 action[:,6] 与 gripper_width 不同步")

    meta["action_sync_max_abs"] = sync_err
    meta_path = input_zarr.parent / "gripper_calib_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {meta_path}")
    print("Done.")


if __name__ == "__main__":
    main()
