#!/usr/bin/env python3
"""CLI: split a DAgger dataset into human / policy folders.

Example:
  python scripts/extract_dagger_split.py \\
      -i /media/lenovo/data/kian/kai0_dagger_11_06 \\
      --what both --overwrite
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.dagger_extract import (  # noqa: E402
    default_split_paths,
    extract_dagger_split,
    inspect_intervention,
)


@click.command()
@click.option(
    "-i",
    "--input",
    "src",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    required=True,
    help="含 replay_buffer.zarr + intervention/ 的 DAgger 目录",
)
@click.option(
    "--human-out",
    type=click.Path(path_type=Path),
    default=None,
    help="人工示教输出目录（默认 {input}_human）",
)
@click.option(
    "--policy-out",
    type=click.Path(path_type=Path),
    default=None,
    help="纯模型输出目录（默认 {input}_policy）",
)
@click.option(
    "--what",
    type=click.Choice(["human", "policy", "both"], case_sensitive=False),
    default="both",
    show_default=True,
)
@click.option("--overwrite", is_flag=True, help="覆盖已存在的输出目录")
@click.option(
    "--no-trim-still",
    is_flag=True,
    help="不去掉人工段开头/结尾静止（进出示教停顿）",
)
def main(
    src: Path,
    human_out: Path | None,
    policy_out: Path | None,
    what: str,
    overwrite: bool,
    no_trim_still: bool,
) -> None:
    stats = inspect_intervention(src)
    click.echo(
        f"[inspect] eps={stats['n_episodes']} labeled={stats['n_labeled']} "
        f"human_eps={stats['n_human_eps']} policy_only={stats['n_policy_only']} "
        f"unlabeled={stats['n_unlabeled']}"
    )
    if human_out is None or policy_out is None:
        d_h, d_p = default_split_paths(src)
        human_out = human_out or d_h
        policy_out = policy_out or d_p

    def _prog(info: dict) -> None:
        msg = info.get("message", "")
        frac = info.get("frac")
        if isinstance(frac, (int, float)):
            click.echo(f"[{100 * float(frac):5.1f}%] {msg}")
        else:
            click.echo(msg)

    result = extract_dagger_split(
        src,
        dst_human=human_out,
        dst_policy=policy_out,
        what=what,
        overwrite=overwrite,
        trim_still=not no_trim_still,
        progress_cb=_prog,
    )
    click.echo(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
