"""Persist sync address book under the repo (git-friendly, no passwords)."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_REL = Path("config") / "sync_remotes.json"


def default_sync_remotes_path() -> Path:
    override = str(os.environ.get("UMI_SYNC_REMOTES_PATH") or "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return (_REPO_ROOT / _DEFAULT_REL).resolve()


def _normalize_entry(raw: Any) -> dict[str, str] | None:
    if isinstance(raw, str):
        dest = raw.strip()
        if not dest:
            return None
        return {"dest": dest, "label": ""}
    if isinstance(raw, dict):
        dest = str(raw.get("dest") or "").strip()
        if not dest:
            return None
        label = str(raw.get("label") or "").strip()
        return {"dest": dest, "label": label}
    return None


def normalize_remotes(items: Any) -> list[dict[str, str]]:
    """Deduplicate by dest (last wins); drop empties."""
    if not isinstance(items, list):
        return []
    by_dest: dict[str, dict[str, str]] = {}
    order: list[str] = []
    for raw in items:
        ent = _normalize_entry(raw)
        if ent is None:
            continue
        dest = ent["dest"]
        if dest not in by_dest:
            order.append(dest)
        by_dest[dest] = ent
    return [by_dest[d] for d in order]


def load_sync_remotes(path: Path | None = None) -> list[dict[str, str]]:
    p = Path(path) if path is not None else default_sync_remotes_path()
    if not p.is_file():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(data, list):
        return normalize_remotes(data)
    if isinstance(data, dict):
        return normalize_remotes(data.get("remotes") or [])
    return []


def save_sync_remotes(
    remotes: list[dict[str, str]] | list[Any],
    path: Path | None = None,
) -> list[dict[str, str]]:
    p = Path(path) if path is not None else default_sync_remotes_path()
    cleaned = normalize_remotes(remotes)
    payload = {"version": 1, "remotes": cleaned}
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    fd, tmp_name = tempfile.mkstemp(
        prefix=".sync_remotes_", suffix=".json", dir=str(p.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, p)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return cleaned


def ensure_sync_remotes_file(path: Path | None = None) -> Path:
    """Create empty template if missing; return path."""
    p = Path(path) if path is not None else default_sync_remotes_path()
    if not p.is_file():
        save_sync_remotes([], p)
    return p
