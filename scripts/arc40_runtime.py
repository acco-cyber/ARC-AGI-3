"""Small standard-library helpers for the pinned ARC3 Kaggle runtime.

Nothing here installs packages, starts a server, or submits a notebook.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def _probe_gpu(*, expected_python=(3, 12), minimum_memory_mib=90_000, timeout=10):
    """Fail before package install/model load on the wrong image or accelerator."""
    if tuple(sys.version_info[:2]) != tuple(expected_python):
        raise RuntimeError(
            f"Pinned offline wheels require Python {expected_python[0]}.{expected_python[1]}; "
            f"observed {sys.version.split()[0]}"
        )
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=timeout,
    )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(f"Expected exactly one GPU; observed {len(lines)}")
    name, sep, memory = lines[0].rpartition(",")
    if not sep or "RTX PRO 6000" not in name.upper():
        raise RuntimeError(f"Pinned Pennyroyal runtime requires RTX PRO 6000; observed {lines[0]}")
    try:
        memory_mib = int(float(memory.strip()))
    except ValueError as exc:
        raise RuntimeError(f"Could not parse GPU memory: {lines[0]}") from exc
    if memory_mib < minimum_memory_mib:
        raise RuntimeError(f"Expected the 96 GB accelerator; observed {memory_mib} MiB")
    return {"python": sys.version.split()[0], "gpu_name": name.strip(), "memory_mib": memory_mib}


def validate_indexed_weights(root):
    """Validate lightweight checkpoint metadata and every referenced shard path."""
    root = Path(root)
    index = json.loads((root / "model.safetensors.index.json").read_text())
    names = sorted(set(index["weight_map"].values()))
    if not names:
        raise RuntimeError(f"Empty weight index: {root}")
    for name in names:
        rel = Path(name)
        if rel.is_absolute() or ".." in rel.parts:
            raise RuntimeError(f"Unsafe shard path in checkpoint: {name}")
        path = root / rel
        if not path.is_file() or path.stat().st_size < 8:
            raise RuntimeError(f"Missing or empty checkpoint shard: {path}")
    return names


class OwnedServer:
    """Track only a Popen launched with start_new_session=True.

    Register immediately after Popen, before the server can fail. The session
    ID and group ID must equal that owned child's PID. No global process-name
    kills are used, and the notebook's own group can never be a target.
    """
    def __init__(self, proc):
        self.proc = proc
        self.pgid = proc.pid
        self.closed = False
        if (os.getpgid(proc.pid) != proc.pid or os.getsid(proc.pid) != proc.pid
                or proc.pid in (os.getpid(), os.getpgrp())):
            raise RuntimeError("Server must be an owned child launched in a new session")

    def _group_exists(self):
        try:
            os.killpg(self.pgid, 0)
            return True
        except ProcessLookupError:
            return False

    def _signal(self, sig):
        try:
            os.killpg(self.pgid, sig)
            return True
        except ProcessLookupError:
            return False

    def shutdown(self, *, grace_s=8.0, kill_wait_s=3.0):
        if grace_s < 0 or kill_wait_s < 0:
            raise ValueError("Cleanup timeouts must be nonnegative")
        if self.closed:
            return {"pid": self.proc.pid, "terminated": False, "killed": False,
                    "returncode": self.proc.returncode}
        # Verify the live leader before first signal. A previously registered
        # group remains ours if the leader has already exited but workers remain.
        if self.proc.poll() is None:
            if (os.getpgid(self.proc.pid) != self.pgid
                    or os.getsid(self.proc.pid) != self.pgid):
                raise RuntimeError("Owned server changed process group/session")
        terminated = self._signal(signal.SIGTERM)
        deadline = time.monotonic() + grace_s
        while self._group_exists() and time.monotonic() < deadline:
            self.proc.poll()
            time.sleep(0.05)
        killed = self._signal(signal.SIGKILL) if self._group_exists() else False
        try:
            self.proc.wait(timeout=kill_wait_s)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("Owned inference server survived TERM/KILL cleanup") from exc
        self.closed = True
        return {"pid": self.proc.pid, "terminated": terminated, "killed": killed,
                "returncode": self.proc.returncode}


# Backwards-compatible name for local tests and notebook integration.
gpu_preflight = _probe_gpu


def stop_owned_server(proc, *, grace_s=8.0, kill_wait_s=3.0):
    """Clean up a live proc created with start_new_session=True.

    Prefer registering OwnedServer(proc) immediately after Popen, so workers
    remain attributable if the launcher exits before notebook teardown.
    """
    if proc.poll() is not None:
        proc.wait()
        return {"pid": proc.pid, "terminated": False, "killed": False,
                "returncode": proc.returncode}
    return OwnedServer(proc).shutdown(grace_s=grace_s, kill_wait_s=kill_wait_s)


def preflight_gpu(*, report_path, **kwargs):
    report_path = Path(report_path)
    report = {"model_inference_attempted": False, "passed": False}
    try:
        report.update(_probe_gpu(**kwargs))
        report["passed"] = True
    except Exception as exc:
        report.update(error_type=type(exc).__name__, error=str(exc))
        report_path.write_text(json.dumps(report, indent=2))
        raise RuntimeError("GPU preflight failed before model load: " + str(exc)) from exc
    report_path.write_text(json.dumps(report, indent=2))
    return report


def register_owned_server(proc):
    import atexit
    owner = OwnedServer(proc)
    atexit.register(owner.shutdown)
    # IPython can stop on a later cell error while keeping the kernel alive.
    # Release our GPU process then, not only at interpreter exit.
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is not None:
            def cleanup_on_error(result):
                if result.error_before_exec is not None or result.error_in_exec is not None:
                    owner.shutdown()
            shell.events.register('post_run_cell', cleanup_on_error)
    except ImportError:
        pass
    return owner.shutdown


def wait_for_server_ready(proc, url, *, timeout_s=300.0):
    from urllib.request import urlopen
    deadline = time.monotonic() + timeout_s
    while True:
        if proc.poll() is not None:
            raise RuntimeError("Inference server exited before readiness")
        try:
            with urlopen(url, timeout=min(5.0, max(0.1, deadline-time.monotonic()))) as response:
                if response.status == 200:
                    return
        except Exception:
            pass
        if time.monotonic() >= deadline:
            raise RuntimeError("Inference server readiness timed out")
        time.sleep(min(3.0, max(0, deadline-time.monotonic())))
