#!/usr/bin/env python3
"""用 robot0 eef_pos/rot/gripper 覆盖写回尖刺 action（pose-as-action 采数）。

示例:
  python scripts/fix_replay_action_from_eef.py -o data/collect_demo_zmq
  python scripts/fix_replay_action_from_eef.py -o data/collect_demo_zmq --all
  python scripts/fix_replay_action_from_eef.py -o data/collect_demo_zmq --episodes 0,2,5
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.collect_web_ui import episode_action_qc, rebuild_spiked_actions  # noqa: E402
from modules.replay_buffer import ReplayBuffer  # noqa: E402


@click.command()
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="采数输出目录（含 replay_buffer.zarr）",
)
@click.option(
    "--episodes",
    default="",
    show_default=False,
    help="逗号分隔 episode id；空=全部",
)
@click.option(
    "--all",
    "force_all",
    is_flag=True,
    help="即使 health=OK 也强制用 eef 覆写 action",
)
@click.option(
    "--dry_run",
    is_flag=True,
    help="只打印 QC，不写盘",
)
def main(output: str, episodes: str, force_all: bool, dry_run: bool) -> None:
    zarr_path = Path(output) / "replay_buffer.zarr"
    if not zarr_path.exists():
        raise SystemExit(f"missing {zarr_path}")

    mode = "r" if dry_run else "r+"
    rb = ReplayBuffer.create_from_path(str(zarr_path), mode=mode)
    n = int(rb.n_episodes)
    if episodes.strip():
        eids = [int(x) for x in episodes.split(",") if x.strip() != ""]
    else:
        eids = list(range(n))

    print(f"zarr={zarr_path}  episodes={n}  check={eids}  dry_run={dry_run}")
    if dry_run:
        for eid in eids:
            sl = rb.get_episode_slice(eid)
            qc = episode_action_qc(rb, sl)
            print(
                f"  ep {eid}: {qc['health']}  "
                f"max|Δ|={qc['max_pos_err']:.4f}m"
            )
        return

    summary = rebuild_spiked_actions(
        rb,
        episode_ids=eids,
        only_if_spikes=not force_all,
    )
    print(
        f"checked={summary['n_checked']}  "
        f"rewrote={summary['n_rewrote']}"
    )
    for info in summary["results"]:
        print(f"  {info.get('message')}")


if __name__ == "__main__":
    main()
