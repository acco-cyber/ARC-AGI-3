import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from arc40_runtime import OwnedServer, gpu_preflight, validate_indexed_weights


class PreflightTests(unittest.TestCase):
    def test_wrong_python_is_rejected_before_nvidia_smi(self):
        with patch("arc40_runtime.sys.version_info", (3, 11, 0)), patch("arc40_runtime.subprocess.run") as run:
            with self.assertRaisesRegex(RuntimeError, "Python 3.12"):
                gpu_preflight()
            run.assert_not_called()

    def test_wrong_gpu_and_memory_are_rejected(self):
        for text in ["Tesla P100, 16280\n", "NVIDIA RTX PRO 6000, 24000\n",
                     "NVIDIA RTX PRO 6000, 97887\nNVIDIA RTX PRO 6000, 97887\n"]:
            result = subprocess.CompletedProcess([], 0, stdout=text, stderr="")
            with self.subTest(text=text), patch("arc40_runtime.sys.version_info", (3, 12, 0)), patch("arc40_runtime.subprocess.run", return_value=result):
                with self.assertRaises(RuntimeError):
                    gpu_preflight()

    def test_shard_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "../outside"}}))
            with self.assertRaisesRegex(RuntimeError, "Unsafe shard"):
                validate_indexed_weights(tmp)



class LifecycleTests(unittest.TestCase):
    def _spawn(self, code):
        return subprocess.Popen([sys.executable, "-c", code], start_new_session=True,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def test_shutdown_ignores_unrelated_group_and_is_idempotent(self):
        owned = self._spawn("import time; time.sleep(60)")
        unrelated = self._spawn("import time; time.sleep(60)")
        try:
            server = OwnedServer(owned)
            info = server.shutdown(grace_s=0.2)
            self.assertTrue(info["terminated"])
            self.assertIsNotNone(owned.poll())
            self.assertIsNone(unrelated.poll())
            self.assertFalse(server.shutdown(grace_s=0)["terminated"])
        finally:
            if owned.poll() is None:
                os.killpg(owned.pid, signal.SIGKILL)
                owned.wait()
            os.killpg(unrelated.pid, signal.SIGTERM)
            unrelated.wait()

    def test_sigkill_fallback_reaps_term_ignoring_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            ready = Path(tmp, "ready")
            code = f"import signal,time,pathlib; signal.signal(signal.SIGTERM, signal.SIG_IGN); pathlib.Path({str(ready)!r}).touch(); time.sleep(60)"
            proc = self._spawn(code)
            try:
                deadline = time.monotonic() + 3
                while not ready.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(ready.exists())
                info = OwnedServer(proc).shutdown(grace_s=0.1)
                self.assertTrue(info["killed"])
                self.assertEqual(proc.returncode, -signal.SIGKILL)
            finally:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()

    def test_orphan_term_ignoring_worker_is_killed(self):
        with tempfile.TemporaryDirectory() as tmp:
            ready = Path(tmp, "ready")
            child = f"import signal,time,pathlib; signal.signal(signal.SIGTERM,signal.SIG_IGN); pathlib.Path({str(ready)!r}).touch(); time.sleep(60)"
            parent = f"import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',{child!r}]); time.sleep(60)"
            proc = self._spawn(parent)
            try:
                deadline = time.monotonic() + 3
                while not ready.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(ready.exists())
                info = OwnedServer(proc).shutdown(grace_s=0.1)
                self.assertTrue(info["killed"])
                self.assertIsNotNone(proc.poll())
            finally:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()


if __name__ == "__main__":
    unittest.main()
