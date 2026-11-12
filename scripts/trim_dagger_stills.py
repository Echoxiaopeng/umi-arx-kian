#!/usr/bin/env python3
"""Trim enter/exit-drag (and soft-resume) still frames from a mixed DAgger set.

Keeps both policy and human data in the same episodes — does not split.

Example:
  python scripts/trim_dagger_stills.py \\
      -i /media/lenovo/data/kian/kai0_dagger_11_06 --in-place --overwrite
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.dagger_extract import trim_dagger_transition_stills  # noqa: E402


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
    "-o",
    "--output",
    "dst",
    type=click.Path(path_type=Path),
    default=None,
    help="输出目录（默认与 --in-place 互斥；不设则必须 --in-place）",
)
@click.option(
    "--in-place",
    is_flag=True,
    help="写回原目录（原 zarr/intervention 改名为 *.bak_before_still_trim）",
)
@click.option("--overwrite", is_flag=True, help="覆盖已存在的输出目录")
@click.option(
    "--no-policy-resume-trim",
    is_flag=True,
    help="不裁恢复策略后的 soft-resume 静止",
)
def main(
    src: Path,
    dst: Path | None,
    in_place: bool,
    overwrite: bool,
    no_policy_resume_trim: bool,
) -> None:
    if in_place and dst is not None:
        raise click.UsageError("--in-place 与 --output 不能同时用")
    if not in_place and dst is None:
        raise click.UsageError("请指定 --output 或 --in-place")

    def _prog(info: dict) -> None:
        msg = info.get("message", "")
        frac = info.get("frac")
        if isinstance(frac, (int, float)):
            click.echo(f"[{100 * float(frac):5.1f}%] {msg}")
        else:
            click.echo(msg)

    result = trim_dagger_transition_stills(
        src,
        dst=None if in_place else dst,
        overwrite=overwrite or in_place,
        trim_policy_resume=not no_policy_resume_trim,
        progress_cb=_prog,
    )
    click.echo(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
