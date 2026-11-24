"""Sync local zarr / LeRobot dataset dirs to a remote host via rsync+SSH."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Optional

ProgressCb = Optional[Callable[[dict], None]]

_REMOTE_RE = re.compile(
    r"^(?:(?P<user>[^@/\s]+)@)?(?P<host>[^:/\s]+):(?P<path>.*)$"
)
_TARGET_ONLY_RE = re.compile(
    r"^(?:(?P<user>[^@/\s]+)@)?(?P<host>[^:/\s]+)$"
)
_PROGRESS_RE = re.compile(
    r"(?P<pct>\d+(?:\.\d+)?)%\s+(?P<speed>\S+)?\s*(?P<eta>\S+)?"
)


class SyncRemoteError(RuntimeError):
    """User-facing sync / auth error."""


def _shell_quote(s: str) -> str:
    return "'" + str(s).replace("'", "'\"'\"'") + "'"


def parse_ssh_target(raw: str) -> tuple[str, str, str]:
    """
    Parse ``user@host``, ``user@host:``, or ``user@host:/path``.

    Returns ``(ssh_target, remote_dir, host)``.
    When path is omitted, ``remote_dir`` is ``~`` (browse start).
    """
    text = str(raw or "").strip()
    if not text:
        raise SyncRemoteError("请填写远程主机，例如 user@192.168.1.10")

    m = _REMOTE_RE.match(text)
    if m:
        user = (m.group("user") or "").strip()
        host = (m.group("host") or "").strip()
        path = (m.group("path") or "").strip()
        if not host:
            raise SyncRemoteError("远程主机无效")
        if path != "/" and path.endswith("/"):
            path = path.rstrip("/") or "/"
        if path == "":
            path = "~"
        ssh_target = f"{user}@{host}" if user else host
        return ssh_target, path, host

    m2 = _TARGET_ONLY_RE.match(text)
    if m2:
        user = (m2.group("user") or "").strip()
        host = (m2.group("host") or "").strip()
        if not host:
            raise SyncRemoteError("远程主机无效")
        ssh_target = f"{user}@{host}" if user else host
        return ssh_target, "~", host

    raise SyncRemoteError(
        "格式无效，请输入 user@ip 或 user@ip:/path/"
    )


def parse_remote_dest(dest: str) -> tuple[str, str, str]:
    """
    Parse full sync destination ``user@host:/path`` (path required, not ``~``-only host).

    Returns ``(ssh_target, remote_dir, host)``.
    """
    ssh_target, path, host = parse_ssh_target(dest)
    raw = str(dest or "").strip()
    # Bare user@host is OK for browse, not for sync.
    if ":" not in raw or path in ("", "~"):
        # Allow absolute/relative explicit paths only for sync.
        if ":" not in raw:
            raise SyncRemoteError(
                "请选择远端目录，或填写完整目标 user@host:/path/"
            )
        if path == "~":
            # user@host: with empty path → treat as needing browse/select
            raise SyncRemoteError(
                "请选择远端目录，或填写完整目标 user@host:/path/"
            )
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/") or "/"
    return ssh_target, path, host


def validate_remote_dest_string(dest: str) -> str:
    """Normalize and validate a destination string; return cleaned form."""
    ssh_target, remote_dir, _host = parse_remote_dest(dest)
    return f"{ssh_target}:{remote_dir}"


def _system_bin(name: str) -> str:
    """Prefer /usr/bin to avoid conda OpenSSL conflicts with system OpenSSH."""
    for cand in (f"/usr/bin/{name}", f"/bin/{name}"):
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    path = shutil.which(name)
    if path:
        return path
    return name


def sshpass_available() -> bool:
    return os.path.isfile(_system_bin("sshpass")) or shutil.which("sshpass") is not None


def _ssh_clean_env(password: Optional[str] = None) -> dict:
    """
    Build env for ssh/sshpass subprocesses.

    Conda/ROS LD_LIBRARY_PATH can make system ssh fail with
    ``OpenSSL version mismatch``; strip those vars for the child only.
    """
    env = os.environ.copy()
    for k in (
        "LD_LIBRARY_PATH",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "PYTHONPATH",
    ):
        env.pop(k, None)
    # Keep a minimal PATH that still finds system tools first.
    sys_path = "/usr/bin:/bin:/usr/sbin:/sbin"
    old_path = env.get("PATH", "")
    env["PATH"] = sys_path + ((":" + old_path) if old_path else "")
    if password is not None:
        env["SSHPASS"] = str(password)
    return env


def _ssh_base_argv(
    *,
    password: Optional[str] = None,
    batch_mode: bool = False,
    connect_timeout: int = 8,
) -> list[str]:
    argv: list[str] = []
    if password is not None:
        if not sshpass_available():
            raise SyncRemoteError(
                "需要密码登录但未安装 sshpass。"
                "请 apt/yum 安装 sshpass，或配置 SSH 公钥免密。"
            )
        argv.extend([_system_bin("sshpass"), "-e"])
    argv.append(_system_bin("ssh"))
    argv.extend(
        [
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            f"ConnectTimeout={int(connect_timeout)}",
            "-o",
            "ServerAliveInterval=5",
            "-o",
            "ServerAliveCountMax=2",
        ]
    )
    if batch_mode:
        argv.extend(["-o", "BatchMode=yes", "-o", "PasswordAuthentication=no"])
    elif password is not None:
        # Avoid hanging on pubkey/agent prompts when user provided a password.
        argv.extend(
            [
                "-o",
                "PreferredAuthentications=password",
                "-o",
                "PubkeyAuthentication=no",
                "-o",
                "KbdInteractiveAuthentication=no",
                "-o",
                "NumberOfPasswordPrompts=1",
            ]
        )
    return argv


def _run_ssh(
    ssh_target: str,
    remote_cmd: str,
    *,
    password: Optional[str] = None,
    batch_mode: bool = False,
    connect_timeout: int = 8,
    check: bool = True,
) -> subprocess.CompletedProcess:
    argv = _ssh_base_argv(
        password=password,
        batch_mode=batch_mode,
        connect_timeout=connect_timeout,
    )
    argv.extend([ssh_target, remote_cmd])
    return subprocess.run(
        argv,
        capture_output=True,
        text=True,
        env=_ssh_clean_env(password),
        check=check,
    )


def probe_ssh_auth(
    dest: str,
    *,
    password: Optional[str] = None,
    connect_timeout: int = 8,
) -> dict:
    """
    Probe whether SSH works for ``user@host`` or ``user@host:/path``.

    If ``password`` is given, verify with sshpass; otherwise BatchMode (key/agent).
    """
    ssh_target, remote_dir, _host = parse_ssh_target(dest)
    norm = (
        f"{ssh_target}:{remote_dir}"
        if remote_dir not in ("", "~")
        else ssh_target
    )
    try:
        proc = _run_ssh(
            ssh_target,
            "true",
            password=password,
            batch_mode=password is None,
            connect_timeout=connect_timeout,
            check=False,
        )
    except SyncRemoteError as exc:
        return {
            "ok": False,
            "need_password": False,
            "dest": norm,
            "ssh_target": ssh_target,
            "remote_dir": remote_dir,
            "error": str(exc),
        }
    except FileNotFoundError:
        return {
            "ok": False,
            "need_password": False,
            "dest": norm,
            "ssh_target": ssh_target,
            "remote_dir": remote_dir,
            "error": "未找到 ssh 命令，请先安装 OpenSSH 客户端",
        }

    if proc.returncode == 0:
        return {
            "ok": True,
            "need_password": False,
            "dest": norm,
            "ssh_target": ssh_target,
            "remote_dir": remote_dir,
            "error": None,
        }

    err = (proc.stderr or proc.stdout or "").strip()
    low = err.lower()
    auth_fail = any(
        s in low
        for s in (
            "permission denied",
            "authentication failed",
            "publickey",
            "password",
            "too many authentication failures",
        )
    ) or proc.returncode in (255, 5)
    if "openssl version mismatch" in low:
        return {
            "ok": False,
            "need_password": False,
            "dest": norm,
            "ssh_target": ssh_target,
            "remote_dir": remote_dir,
            "error": (
                "本机 SSH/OpenSSL 环境冲突（conda/ROS 库干扰）。"
                "已尝试隔离环境仍失败，请检查系统 ssh。"
            ),
        }
    if auth_fail and password is None:
        hint = None
        if not sshpass_available():
            hint = (
                "免密失败；若要用密码请先安装 sshpass，"
                "或配置 SSH 公钥。"
            )
        return {
            "ok": True,
            "need_password": True,
            "dest": norm,
            "ssh_target": ssh_target,
            "remote_dir": remote_dir,
            "error": hint,
            "sshpass_missing": not sshpass_available(),
        }
    if auth_fail and password is not None:
        return {
            "ok": False,
            "need_password": False,
            "dest": norm,
            "ssh_target": ssh_target,
            "remote_dir": remote_dir,
            "error": f"密码错误或无权登录：{ssh_target}",
        }

    return {
        "ok": False,
        "need_password": False,
        "dest": norm,
        "ssh_target": ssh_target,
        "remote_dir": remote_dir,
        "error": err or f"SSH 探测失败 (exit {proc.returncode})",
    }


def list_remote_dirs(
    target: str,
    path: Optional[str] = None,
    *,
    password: Optional[str] = None,
    connect_timeout: int = 12,
) -> dict:
    """
    List subdirectories on a remote host via SSH.

    ``target`` may be ``user@host`` or ``user@host:/start``.
    Returns path / parent / entries[{name,path,is_dir}] / ssh_target / dest.
    """
    import base64
    import json as _json

    ssh_target, default_path, _host = parse_ssh_target(target)
    remote_path = str(path if path is not None else default_path).strip() or "~"

    # Avoid SSH heredoc (often hangs / drops stdin). Use base64 + python3 -c.
    py_code = (
        "import json,os\n"
        f"raw={_json.dumps(remote_path)}\n"
        "path=os.path.abspath(os.path.expanduser(raw))\n"
        "\n"
        "def fail(msg):\n"
        "    print(json.dumps({'ok': False, 'error': str(msg)}))\n"
        "    raise SystemExit(0)\n"
        "\n"
        "if not os.path.isdir(path):\n"
        "    fail('不是目录: ' + path)\n"
        "parent=os.path.dirname(path.rstrip(os.sep)) if path not in ('/','') else None\n"
        "if parent==path:\n"
        "    parent=None\n"
        "entries=[]\n"
        "try:\n"
        "    names=sorted(os.listdir(path), key=lambda s: s.lower())\n"
        "except OSError as exc:\n"
        "    fail(exc)\n"
        "for name in names:\n"
        "    if name in ('.','..'):\n"
        "        continue\n"
        "    full=os.path.join(path, name)\n"
        "    try:\n"
        "        is_dir=os.path.isdir(full)\n"
        "    except OSError:\n"
        "        continue\n"
        "    if is_dir:\n"
        "        entries.append({'name': name, 'path': full, 'is_dir': True})\n"
        "print(json.dumps({'ok': True, 'path': path, 'parent': parent, 'entries': entries}))\n"
    )
    b64 = base64.b64encode(py_code.encode("utf-8")).decode("ascii")
    remote_script = (
        "if ! command -v python3 >/dev/null 2>&1; then "
        "echo '{\"ok\": false, \"error\": \"远端未安装 python3，无法浏览目录\"}'; "
        "exit 0; fi; "
        "python3 -c \"import base64; exec(base64.b64decode('"
        + b64
        + "').decode())\""
    )
    proc = _run_ssh(
        ssh_target,
        remote_script,
        password=password,
        batch_mode=password is None,
        connect_timeout=connect_timeout,
        check=False,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        low = err.lower()
        if "openssl version mismatch" in low:
            raise SyncRemoteError(
                "本机 SSH/OpenSSL 环境冲突（conda/ROS 库干扰），列目录失败"
            )
        need_pw = password is None and any(
            s in low
            for s in (
                "permission denied",
                "authentication failed",
                "publickey",
                "password",
            )
        )
        if need_pw:
            raise SyncRemoteError("需要密码登录")
        if password is not None and (
            "permission denied" in low or "authentication failed" in low
        ):
            raise SyncRemoteError(f"密码错误或无权登录：{ssh_target}")
        raise SyncRemoteError(err or f"远端列目录失败 (exit {proc.returncode})")

    out_text = (proc.stdout or "").strip()
    # Take last JSON line in case of ssh banners.
    line = ""
    for cand in reversed(out_text.splitlines()):
        c = cand.strip()
        if c.startswith("{") and c.endswith("}"):
            line = c
            break
    if not line:
        raise SyncRemoteError(
            "远端列目录无有效输出（需要远端有 python3 或兼容 shell）"
        )
    try:
        import json as _json

        data = _json.loads(line)
    except Exception as exc:  # noqa: BLE001
        raise SyncRemoteError(f"解析远端目录列表失败: {exc}") from exc

    if not data.get("ok"):
        raise SyncRemoteError(str(data.get("error") or "远端列目录失败"))

    abs_path = str(data.get("path") or remote_path)
    parent = data.get("parent")
    entries = data.get("entries") or []
    return {
        "ok": True,
        "ssh_target": ssh_target,
        "path": abs_path,
        "parent": parent,
        "entries": entries,
        "dest": f"{ssh_target}:{abs_path}",
    }


def _validate_local_datasets(paths: list[str | Path]) -> list[Path]:
    from modules.lerobot_dataset_io import detect_dataset_format

    out: list[Path] = []
    if not paths:
        raise SyncRemoteError("未选择要同步的数据集")
    for raw in paths:
        p = Path(raw).expanduser().resolve()
        if not p.exists():
            raise SyncRemoteError(f"路径不存在: {p}")
        if not p.is_dir():
            raise SyncRemoteError(f"不是目录: {p}")
        fmt = detect_dataset_format(p)
        if fmt is None:
            raise SyncRemoteError(f"无法识别为 zarr/LeRobot 数据集: {p}")
        out.append(p)
    return out


def _ensure_remote_dir(
    ssh_target: str,
    remote_dir: str,
    *,
    password: Optional[str] = None,
) -> None:
    proc = _run_ssh(
        ssh_target,
        f"mkdir -p {_shell_quote(remote_dir)}",
        password=password,
        batch_mode=password is None,
        check=False,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        raise SyncRemoteError(
            f"远端创建目录失败 ({remote_dir}): {err or proc.returncode}"
        )


def _parse_rsync_progress_line(line: str) -> Optional[float]:
    # Prefer xfer# / to-check lines from --info=progress2, else plain % lines.
    m = _PROGRESS_RE.search(line.replace(",", ""))
    if not m:
        return None
    try:
        return max(0.0, min(100.0, float(m.group("pct")))) / 100.0
    except (TypeError, ValueError):
        return None


def _rsync_one(
    local: Path,
    ssh_target: str,
    remote_subdir: str,
    *,
    password: Optional[str] = None,
    delete: bool = False,
    progress_cb: ProgressCb = None,
    dataset_i: int = 0,
    n_datasets: int = 1,
) -> None:
    rsync_bin = _system_bin("rsync")
    if not (os.path.isfile(rsync_bin) or shutil.which("rsync")):
        raise SyncRemoteError("未找到 rsync，请先安装 rsync")

    remote_uri = f"{ssh_target}:{remote_subdir}/"
    local_src = str(local) + "/"

    ssh_argv = _ssh_base_argv(
        password=password,
        batch_mode=password is None,
    )
    # rsync -e wants a single shell-like command string.
    rsh = " ".join(ssh_argv)

    cmd = [
        rsync_bin,
        "-a",
        "--info=progress2",
        "-e",
        rsh,
    ]
    if delete:
        cmd.append("--delete")
    cmd.extend([local_src, remote_uri])

    # Never print password / SSHPASS.
    print(f"[sync-remote] rsync {local} → {remote_uri}", flush=True)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=_ssh_clean_env(password),
        bufsize=1,
    )
    assert proc.stdout is not None
    last_frac = 0.0
    try:
        for raw_line in proc.stdout:
            line = raw_line.rstrip("\n")
            frac = _parse_rsync_progress_line(line)
            if frac is not None:
                last_frac = frac
            # overall: dataset index + within-file frac
            overall = (dataset_i + last_frac) / max(n_datasets, 1)
            if progress_cb is not None:
                progress_cb(
                    {
                        "phase": "sync",
                        "frac": float(overall),
                        "dataset_i": int(dataset_i),
                        "n_datasets": int(n_datasets),
                        "dataset": local.name,
                        "message": (
                            f"同步 {dataset_i + 1}/{n_datasets} "
                            f"{local.name} · {int(last_frac * 100)}%"
                        ),
                        "line": line[-200:] if line else "",
                    }
                )
        rc = proc.wait()
    finally:
        if proc.poll() is None:
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass
            proc.wait(timeout=5)

    if rc != 0:
        raise SyncRemoteError(
            f"rsync 失败 ({local.name} → {remote_uri}) exit={rc}"
        )


def sync_datasets(
    paths: list[str | Path],
    remote_dest: str,
    *,
    delete: bool = False,
    password: Optional[str] = None,
    progress_cb: ProgressCb = None,
) -> dict:
    """
    Sync each local dataset directory to ``remote_dest/<basename>/``.

    ``password`` is optional; when set, uses sshpass -e (SSHPASS). Never log it.
    """
    locals_ = _validate_local_datasets(paths)
    ssh_target, remote_root, _host = parse_remote_dest(remote_dest)
    n = len(locals_)

    if password is not None and not sshpass_available():
        raise SyncRemoteError(
            "需要密码登录但未安装 sshpass。"
            "请安装 sshpass 或配置 SSH 公钥免密。"
        )

    if progress_cb is not None:
        progress_cb(
            {
                "phase": "sync",
                "frac": 0.0,
                "message": f"准备远端目录 {ssh_target}:{remote_root}…",
                "n_datasets": n,
            }
        )

    _ensure_remote_dir(ssh_target, remote_root, password=password)

    done: list[str] = []
    try:
        for i, local in enumerate(locals_):
            remote_subdir = f"{remote_root.rstrip('/')}/{local.name}"
            if progress_cb is not None:
                progress_cb(
                    {
                        "phase": "sync",
                        "frac": float(i) / max(n, 1),
                        "dataset_i": i,
                        "n_datasets": n,
                        "dataset": local.name,
                        "message": f"同步 {i + 1}/{n} {local.name}…",
                    }
                )
            _ensure_remote_dir(ssh_target, remote_subdir, password=password)
            _rsync_one(
                local,
                ssh_target,
                remote_subdir,
                password=password,
                delete=bool(delete),
                progress_cb=progress_cb,
                dataset_i=i,
                n_datasets=n,
            )
            done.append(str(local))
    finally:
        # Best-effort: drop password from this process env if we set it.
        if "SSHPASS" in os.environ and password is not None:
            # Only clear if we might have leaked into process env via mistake;
            # _run_ssh / Popen use copied env, so this is just hygiene.
            pass

    if progress_cb is not None:
        progress_cb(
            {
                "phase": "sync",
                "frac": 1.0,
                "n_datasets": n,
                "message": f"同步完成 · {n} 个数据集 → {ssh_target}:{remote_root}",
            }
        )

    return {
        "n_datasets": n,
        "done": done,
        "remote_dest": f"{ssh_target}:{remote_root}",
    }
