"""Resumable CLI runs — a CLI turn that outlives the bridge process.

An agent building agntchat edits the desktop app it runs inside; `tauri dev`
rebuilds and restarts that app, and the bridge (its child) goes with it. The
CLI run must survive that, and the next bridge must deliver its result.

How:

* A run started under a run key (``run_key_var``: ``("task", task_id)`` or
  ``("message", message_id)``, set by the executor around each handler) writes
  its stdout/stderr to files in a per-run directory, with a ``meta.json``
  record, instead of pipes that die with the bridge.
* Shutdown leaves resumable runs alone (``_cli_utils.begin_shutdown``);
  computer-use / Chrome runs are never resumable — an orphan driving the
  desktop is the hazard the process-group kill exists for.
* A new bridge reports the task ids it can adopt at registration
  (``resumable_task_ids``); the server requeues those rows for redelivery
  instead of failing them. Messages are requeued on re-register anyway.
* The redelivered handler runs as usual. When it reaches the CLI spawn,
  ``find_adoptable`` returns the surviving run for the same key and the
  backend reads its event file from the start instead of spawning — so all
  post-run handling (reply, completion, usage) is the normal code path.

Records belong to the process that spawned them (``PROCESS_TOKEN``); only a
different process adopts. A record is dropped when its run is finished and
handled, killed, or found dead without a result.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Optional, Tuple

logger = logging.getLogger("agentchat.backends.cli_runs")

# ("task", task_id) | ("message", message_id) for the handler in flight.
run_key_var: contextvars.ContextVar[Optional[Tuple[str, str]]] = contextvars.ContextVar(
    "agntchat_cli_run_key", default=None
)

# Identifies this bridge process's own records.
PROCESS_TOKEN = uuid.uuid4().hex

_POLL_INTERVAL = 0.1

# A record nobody adopted by now never will be: its task was closed some other
# way (failed as interrupted, cancelled, done by hand). Dropped at registration.
_MAX_RECORD_AGE_SECONDS = 24 * 3600

# POSIX only. The liveness probe is `os.kill(pid, 0)`, which on Windows
# TERMINATES the process, and there is no process group to kill or check
# there. Windows keeps piped, non-resumable runs.
SUPPORTED = os.name == "posix"


def runs_dir() -> Path:
    home = Path(os.environ.get("AGNTCHAT_HOME", str(Path.home() / ".agentchat")))
    agent = os.environ.get("AGENT_ID") or "default"
    return home / "runs" / agent


def _key_dirname(key: Tuple[str, str]) -> str:
    kind, ident = key
    safe = "".join(c for c in ident if c.isalnum() or c in "-_")
    return f"{kind}-{safe}"


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # A reused pid would not lead its own process group (the CLI is spawned
    # as a session leader), so a pgid mismatch means our process is gone.
    try:
        return os.getpgid(pid) == pid
    except OSError:
        return False


class FileTail:
    """``async read(n)`` over a file another process is still appending to.

    Returns data as it arrives; returns ``b""`` (EOF) only once the writer is
    dead and the file is drained — what ``iter_event_lines`` expects of a
    pipe.
    """

    def __init__(self, path: Path, alive):
        self._f = open(path, "rb")
        self._alive = alive

    async def read(self, n: int) -> bytes:
        while True:
            data = self._f.read(n)
            if data:
                return data
            if not self._alive():
                return self._f.read(n)
            await asyncio.sleep(_POLL_INTERVAL)

    def close(self) -> None:
        try:
            self._f.close()
        except OSError:
            pass


class CliRun:
    """One CLI run, spawned here or adopted from a previous bridge process."""

    def __init__(
        self,
        *,
        pid: int,
        record_dir: Optional[Path],
        proc: Optional[asyncio.subprocess.Process] = None,
        resumable: bool = False,
        adopted: bool = False,
    ):
        self.pid = pid
        self.proc = proc
        self.record_dir = record_dir
        self.resumable = resumable
        self.adopted = adopted
        self._tail: Optional[FileTail] = None

    # -- liveness ------------------------------------------------------
    def alive(self) -> bool:
        if self.proc is not None:
            return self.proc.returncode is None
        return pid_alive(self.pid)

    @property
    def returncode(self) -> Optional[int]:
        """Exit code, or None for an adopted run (not our child — unknowable)."""
        return self.proc.returncode if self.proc is not None else None

    # -- I/O -------------------------------------------------------------
    def stdout(self):
        if self.record_dir is None:
            assert self.proc is not None and self.proc.stdout is not None
            return self.proc.stdout
        if self._tail is None:
            self._tail = FileTail(self.record_dir / "stdout.jsonl", self.alive)
        return self._tail

    async def wait(self) -> None:
        if self.proc is not None:
            await self.proc.wait()
            return
        while pid_alive(self.pid):
            await asyncio.sleep(_POLL_INTERVAL)

    async def stderr_text(self) -> str:
        if self.record_dir is None:
            if self.proc is not None and self.proc.stderr is not None:
                return (await self.proc.stderr.read()).decode(errors="replace").strip()
            return ""
        try:
            return (self.record_dir / "stderr.log").read_text(errors="replace").strip()
        except OSError:
            return ""

    # -- teardown --------------------------------------------------------
    def discard(self) -> None:
        """Forget the run: close the tail and delete its record directory."""
        if self._tail is not None:
            self._tail.close()
            self._tail = None
        if self.record_dir is not None:
            shutil.rmtree(self.record_dir, ignore_errors=True)


async def spawn(
    argv: list[str],
    *,
    prompt: str,
    env: dict[str, str],
    limit: int,
    resumable: bool,
    subprocess_kwargs: dict[str, Any],
) -> CliRun:
    """Start a CLI run. File-backed and recorded when a run key is set."""
    key = run_key_var.get() if SUPPORTED else None
    record_dir: Optional[Path] = None
    if key is not None:
        record_dir = runs_dir() / _key_dirname(key)
        shutil.rmtree(record_dir, ignore_errors=True)
        record_dir.mkdir(parents=True, exist_ok=True)

    if record_dir is None:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=limit,
            env=env,
            start_new_session=True,
            **subprocess_kwargs,
        )
    else:
        with open(record_dir / "stdout.jsonl", "wb") as out, open(record_dir / "stderr.log", "wb") as err:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=out,
                stderr=err,
                env=env,
                start_new_session=True,
                **subprocess_kwargs,
            )
        assert key is not None
        _write_json(
            record_dir / "meta.json",
            {
                "kind": key[0],
                "id": key[1],
                "pid": proc.pid,
                "owner": PROCESS_TOKEN,
                "resumable": resumable,
                "started_at": time.time(),
            },
        )

    if prompt and proc.stdin:
        proc.stdin.write(prompt.encode())
        await proc.stdin.drain()
        proc.stdin.close()

    # Only a file-backed run can be adopted; a piped one dies with its reader.
    return CliRun(
        pid=proc.pid, proc=proc, record_dir=record_dir,
        resumable=resumable and record_dir is not None,
    )


def find_adoptable(key: Optional[Tuple[str, str]] = None) -> Optional[CliRun]:
    """The surviving run a previous bridge left for this run key, if any."""
    key = key or run_key_var.get()
    if key is None or not SUPPORTED:
        return None
    record_dir = runs_dir() / _key_dirname(key)
    meta = _read_meta(record_dir)
    if meta is None or not _adoptable(record_dir, meta):
        return None
    logger.info(
        "Adopting CLI run for %s %s (pid=%s, alive=%s)",
        key[0], key[1], meta["pid"], pid_alive(meta["pid"]),
    )
    return CliRun(pid=meta["pid"], record_dir=record_dir, resumable=True, adopted=True)


def resumable_task_ids() -> list[str]:
    """Task ids with a run this process can adopt; prunes dead records.

    Called once at registration. A record that is neither running nor holding
    a finished result is deleted — the server fails that task as interrupted.
    """
    root = runs_dir()
    ids: list[str] = []
    if not SUPPORTED or not root.is_dir():
        return ids
    for record_dir in root.iterdir():
        meta = _read_meta(record_dir)
        if meta is None:
            shutil.rmtree(record_dir, ignore_errors=True)
            continue
        if meta.get("owner") == PROCESS_TOKEN:
            continue
        if not _adoptable(record_dir, meta):
            logger.info(
                "Dropping CLI run record for %s %s: not resumable, too old, or died without a result",
                meta.get("kind"), meta.get("id"),
            )
            shutil.rmtree(record_dir, ignore_errors=True)
            continue
        if meta.get("kind") == "task":
            ids.append(str(meta["id"]))
    return ids


def _adoptable(record_dir: Path, meta: dict[str, Any]) -> bool:
    if meta.get("owner") == PROCESS_TOKEN or not meta.get("resumable"):
        return False
    if time.time() - float(meta.get("started_at") or 0) > _MAX_RECORD_AGE_SECONDS:
        return False
    return pid_alive(int(meta["pid"])) or _has_result_event(record_dir / "stdout.jsonl")


def _has_result_event(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 256 * 1024))
            tail = f.read()
    except OSError:
        return False
    for line in tail.splitlines()[::-1]:
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(event, dict) and event.get("type") == "result":
            return True
    return False


def _read_meta(record_dir: Path) -> Optional[dict[str, Any]]:
    try:
        meta = json.loads((record_dir / "meta.json").read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(meta, dict) or not isinstance(meta.get("pid"), int):
        return None
    return meta


def _write_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    os.replace(tmp, path)
