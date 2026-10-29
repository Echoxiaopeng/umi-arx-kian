#!/usr/bin/env python3
"""Parse eval logs and plot RTC vs open-loop commanded trajectories (OpenCV)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import numpy as np

EVAL = Path(__file__).resolve().parents[1] / "data_local" / "kai0_eval"
OUT = EVAL / "traj_compare"

WP_RE = re.compile(
    r"\[Arx5Controller\] schedule_joint_waypoint -> \[([^\]]+)\] @ ([0-9.]+)"
)


def parse_log(path: Path) -> dict | None:
    text = path.read_text(errors="replace").splitlines()
    argv = ""
    for line in text[:12]:
        if line.startswith("[log] argv="):
            argv = line[len("[log] argv=") :]

    waypoints: list[tuple[float, np.ndarray]] = []
    swaps_t: list[float] = []
    pending_swap = False
    for line in text:
        if "[RTC] swap #" in line:
            pending_swap = True
        m = WP_RE.search(line)
        if not m:
            continue
        parts = [float(x) for x in m.group(1).split() if x]
        if len(parts) != 7:
            continue
        t = float(m.group(2))
        waypoints.append((t, np.asarray(parts, dtype=np.float64)))
        if pending_swap:
            swaps_t.append(t)
            pending_swap = False

    by_t: dict[float, list[np.ndarray]] = {}
    for t, q in waypoints:
        by_t.setdefault(round(t, 6), []).append(q)

    # Pair arms by temporal continuity: log order of L/R at the same
    # timestamp is not stable, so pick the assignment closest to previous.
    rows: list[tuple[float, np.ndarray]] = []
    prev14: np.ndarray | None = None
    for t in sorted(by_t):
        qs = by_t[t]
        if len(qs) < 2:
            continue
        a, b = qs[0], qs[1]
        cand_ab = np.concatenate([a, b])
        cand_ba = np.concatenate([b, a])
        if prev14 is None:
            q14 = cand_ab
        else:
            dab = float(np.max(np.abs(cand_ab - prev14)))
            dba = float(np.max(np.abs(cand_ba - prev14)))
            q14 = cand_ab if dab <= dba else cand_ba
        # Drop impossible one-step teleport (>0.5 rad) as log seam / reseed.
        if prev14 is not None and float(np.max(np.abs(q14 - prev14))) > 0.5:
            continue
        rows.append((t, q14))
        prev14 = q14
    if not rows:
        return None

    t0 = rows[0][0]
    T = np.asarray([r[0] - t0 for r in rows], dtype=np.float64)
    Q = np.stack([r[1] for r in rows], axis=0)
    swaps = np.asarray(
        [s - t0 for s in swaps_t if s >= rows[0][0]], dtype=np.float64
    )

    dQ = np.diff(Q, axis=0)
    idx = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
    step = np.max(np.abs(dQ[:, idx]), axis=1)
    active = np.where(step > 1e-4)[0]
    if len(active) == 0:
        return None
    i0 = int(active[0])
    i1 = int(active[-1]) + 1
    travel = np.cumsum(step)
    start_candidates = np.where(travel >= travel[i0] + 0.15)[0]
    start = int(start_candidates[0]) if len(start_candidates) else i0
    end = i1

    T = T[start : end + 1]
    Q = Q[start : end + 1]
    if len(T) > 2:
        T = T - T[0]
    swaps = swaps[
        (swaps >= (rows[start][0] - t0))
        & (swaps <= (rows[min(end, len(rows) - 1)][0] - t0))
    ]
    swaps = swaps - (rows[start][0] - t0)
    return {
        "path": path.name,
        "argv": argv,
        "rtc": "--rtc" in argv,
        "T": T,
        "Q": Q,
        "swaps": swaps,
    }


def joint_step_stats(Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = np.diff(Q, axis=0)
    idx = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
    step = np.max(np.abs(d[:, idx]), axis=1)
    dd = np.diff(d, axis=0)
    accel = np.max(np.abs(dd[:, idx]), axis=1)
    return step, accel


def _put_text(img: np.ndarray, text: str, org: tuple[int, int], scale=0.45) -> None:
    cv2.putText(
        img,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (30, 30, 30),
        1,
        cv2.LINE_AA,
    )


def draw_lines(
    panels: list[dict],
    *,
    width: int = 1400,
    panel_h: int = 220,
    title: str,
    out_path: Path,
) -> None:
    margin_l, margin_r, margin_t, margin_b = 70, 30, 50, 40
    n = len(panels)
    h = margin_t + margin_b + n * panel_h
    img = np.full((h, width, 3), 255, dtype=np.uint8)
    _put_text(img, title, (margin_l, 28), 0.6)

    for i, panel in enumerate(panels):
        y0 = margin_t + i * panel_h
        x0, x1 = margin_l, width - margin_r
        y1 = y0 + panel_h - 28
        cv2.rectangle(img, (x0, y0), (x1, y1), (230, 230, 230), 1)
        _put_text(img, panel["title"], (x0 + 4, y0 + 16), 0.42)

        xs = np.asarray(panel["x"], dtype=np.float64)
        ys = np.asarray(panel["y"], dtype=np.float64)
        if len(xs) < 2:
            continue
        xmax = float(panel.get("xmax", xs[-1]))
        ymin = float(panel.get("ymin", 0.0))
        ymax = float(panel.get("ymax", max(ys.max(), 1e-6)))
        if ymax <= ymin:
            ymax = ymin + 1e-6

        def px(x: float) -> int:
            return int(x0 + (np.clip(x, 0, xmax) / max(xmax, 1e-9)) * (x1 - x0 - 1))

        def py(y: float) -> int:
            return int(y1 - ((np.clip(y, ymin, ymax) - ymin) / (ymax - ymin)) * (y1 - y0 - 1))

        # grid
        for g in range(5):
            yy = y0 + int((g / 4) * (y1 - y0))
            cv2.line(img, (x0, yy), (x1, yy), (245, 245, 245), 1)
        for g in range(9):
            xx = x0 + int((g / 8) * (x1 - x0))
            cv2.line(img, (xx, y0), (xx, y1), (245, 245, 245), 1)

        for s in panel.get("vlines", []):
            if 0 <= s <= xmax:
                xx = px(float(s))
                cv2.line(img, (xx, y0), (xx, y1), (180, 180, 180), 1)

        for href, hcol in panel.get("hlines", []):
            yy = py(float(href))
            cv2.line(img, (x0, yy), (x1, yy), hcol, 1, cv2.LINE_AA)

        color = panel.get("color", (30, 120, 200))
        pts = np.column_stack([xs, ys])
        # downsample for drawing
        if len(pts) > 2000:
            idx = np.linspace(0, len(pts) - 1, 2000).astype(int)
            pts = pts[idx]
        for a, b in zip(pts[:-1], pts[1:]):
            if a[0] > xmax and b[0] > xmax:
                continue
            cv2.line(
                img,
                (px(float(a[0])), py(float(a[1]))),
                (px(float(b[0])), py(float(b[1]))),
                color,
                1,
                cv2.LINE_AA,
            )

        _put_text(img, f"{ymax:.3f}", (8, y0 + 14), 0.35)
        _put_text(img, f"{ymin:.3f}", (8, y1 - 4), 0.35)
        _put_text(img, "0s", (x0, y1 + 14), 0.35)
        _put_text(img, f"{xmax:.0f}s", (x1 - 40, y1 + 14), 0.35)

    cv2.imwrite(str(out_path), img)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    files = {
        "openloop": EVAL / "run_20261029_101233.log",
        "rtc_smooth": EVAL / "run_20261029_104503.log",
        "rtc_accel4": EVAL / "run_20261029_105440.log",
    }
    if not files["openloop"].exists():
        files["openloop"] = EVAL / "run_20261029_095020.log"

    colors_bgr = {
        "openloop": (180, 100, 40),
        "rtc_smooth": (40, 140, 255),
        "rtc_accel4": (40, 40, 220),
    }
    labels = {
        "openloop": "Open-loop SPI=50",
        "rtc_smooth": "RTC timing-fix (no accel clip)",
        "rtc_accel4": "RTC accel=4",
    }

    runs: dict[str, dict] = {}
    for key, path in files.items():
        print(f"parse {key}: {path.name}")
        r = parse_log(path)
        if r is None:
            print("  FAIL")
            continue
        print(
            f"  n={len(r['T'])} t={r['T'][-1]:.1f}s rtc={r['rtc']} "
            f"swaps={len(r['swaps'])}"
        )
        runs[key] = r

    if not runs:
        raise SystemExit("no runs parsed")

    # step size figure
    panels = []
    for key in ["openloop", "rtc_smooth", "rtc_accel4"]:
        if key not in runs:
            continue
        r = runs[key]
        step, _ = joint_step_stats(r["Q"])
        xmax = min(40.0, float(r["T"][-1]))
        panels.append(
            {
                "title": (
                    f"{labels[key]} |Δq|_∞  "
                    f"p50={np.median(step):.4f} p95={np.percentile(step,95):.4f} "
                    f"max={step.max():.4f}"
                ),
                "x": r["T"][1:],
                "y": step,
                "xmax": xmax,
                "ymin": 0.0,
                "ymax": max(0.08, float(np.percentile(step, 99))),
                "vlines": r["swaps"].tolist(),
                "color": colors_bgr[key],
            }
        )
    p1 = OUT / "step_size_compare.png"
    draw_lines(
        panels,
        title="Commanded step size |Δq|_∞ (gray=RTC swap)",
        out_path=p1,
    )
    print("wrote", p1)

    # accel figure
    panels = []
    for key in ["openloop", "rtc_smooth", "rtc_accel4"]:
        if key not in runs:
            continue
        r = runs[key]
        _, accel = joint_step_stats(r["Q"])
        xmax = min(40.0, float(r["T"][-1]))
        hlines = []
        if key == "rtc_accel4":
            hlines = [(0.01, (0, 0, 0))]
        panels.append(
            {
                "title": (
                    f"{labels[key]} |Δ(Δq)|_∞  "
                    f"p50={np.median(accel):.4f} p95={np.percentile(accel,95):.4f} "
                    f"max={accel.max():.4f}"
                ),
                "x": r["T"][2:],
                "y": accel,
                "xmax": xmax,
                "ymin": 0.0,
                "ymax": max(0.05, float(np.percentile(accel, 99))),
                "vlines": r["swaps"].tolist(),
                "hlines": hlines,
                "color": colors_bgr[key],
            }
        )
    p2 = OUT / "accel_compare.png"
    draw_lines(
        panels,
        title="Commanded accel |ΔΔq|_∞ (black dashed≈accel clip for a=4)",
        out_path=p2,
    )
    print("wrote", p2)

    # relative position overlay (first 15s) — two joints stacked
    panels = []
    for j, jname in [(1, "left q1 Δ"), (2, "left q2 Δ")]:
        # draw each series into one panel by combining? Better separate panels per mode
        for key in ["openloop", "rtc_smooth", "rtc_accel4"]:
            if key not in runs:
                continue
            r = runs[key]
            m = r["T"] <= 15.0
            if not np.any(m):
                continue
            q = r["Q"][m, j] - r["Q"][m, j][0]
            panels.append(
                {
                    "title": f"{labels[key]} · {jname}",
                    "x": r["T"][m],
                    "y": q,
                    "xmax": 15.0,
                    "ymin": float(q.min() - 0.02),
                    "ymax": float(q.max() + 0.02),
                    "vlines": (
                        [s for s in r["swaps"] if s <= 15]
                        if key.startswith("rtc")
                        else []
                    ),
                    "color": colors_bgr[key],
                }
            )
    p3 = OUT / "position_relative_15s.png"
    draw_lines(
        panels,
        title="Relative joint position first 15s (different episodes; shape only)",
        out_path=p3,
        panel_h=160,
    )
    print("wrote", p3)

    # zoom around swap for accel4
    if "rtc_accel4" in runs and len(runs["rtc_accel4"]["swaps"]) >= 3:
        r = runs["rtc_accel4"]
        s = float(r["swaps"][min(5, len(r["swaps"]) - 1)])
        mask = (r["T"] >= s - 0.6) & (r["T"] <= s + 1.2)
        panels = []
        for j, jname in [(1, "Lq1"), (2, "Lq2"), (8, "Rq1")]:
            panels.append(
                {
                    "title": f"RTC accel=4 {jname} around swap@{s:.2f}s",
                    "x": r["T"][mask],
                    "y": r["Q"][mask, j],
                    "xmax": float(r["T"][mask][-1]),
                    "ymin": float(r["Q"][mask, j].min() - 0.01),
                    "ymax": float(r["Q"][mask, j].max() + 0.01),
                    "vlines": [s],
                    "color": (40, 40, 220),
                }
            )
        # also step around swap
        step, accel = joint_step_stats(r["Q"])
        t_step = r["T"][1:]
        m2 = (t_step >= s - 0.6) & (t_step <= s + 1.2)
        panels.append(
            {
                "title": f"|Δq| around swap@{s:.2f}s",
                "x": t_step[m2],
                "y": step[m2],
                "xmax": float(t_step[m2][-1]) if np.any(m2) else s + 1.2,
                "ymin": 0.0,
                "ymax": max(0.05, float(step[m2].max()) if np.any(m2) else 0.05),
                "vlines": [s],
                "color": (40, 40, 220),
            }
        )
        p4 = OUT / "rtc_accel4_swap_zoom.png"
        draw_lines(
            panels,
            title="Zoom: RTC accel=4 around one swap",
            out_path=p4,
            panel_h=180,
        )
        print("wrote", p4)

    summary = {}
    for key, r in runs.items():
        step, accel = joint_step_stats(r["Q"])
        summary[key] = {
            "file": r["path"],
            "duration_s": float(r["T"][-1]),
            "n": int(len(r["T"])),
            "n_swaps": int(len(r["swaps"])),
            "step_p50": float(np.median(step)),
            "step_p95": float(np.percentile(step, 95)),
            "step_max": float(step.max()),
            "accel_p50": float(np.median(accel)),
            "accel_p95": float(np.percentile(accel, 95)),
            "accel_max": float(accel.max()),
            "frac_accel_gt_0.02": float(np.mean(accel > 0.02)),
            "frac_accel_gt_0.01": float(np.mean(accel > 0.01)),
        }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
