#!/usr/bin/env python3
"""Focused tests for the opt-in hosted Docker matrix harness."""

import unittest
import tempfile
from pathlib import Path

import run_hosted_docker as hosted_docker


DETECTED = {
    "ncpu": 64,
    "mem_avail_mib": 200_000,
    "ulimit_nproc": 4096,
}


class HostedDockerTests(unittest.TestCase):
    def test_default_limits_are_bounded(self):
        limits = hosted_docker.compute_limits(DETECTED)
        self.assertEqual(limits.cpus, 8)
        self.assertEqual(limits.memory_mib, 24576)
        self.assertEqual(limits.pids, 2048)

    def test_unsafe_overrides_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "--cpus"):
            hosted_docker.compute_limits(DETECTED, cpus=65)
        with self.assertRaisesRegex(ValueError, "--memory-mib"):
            hosted_docker.compute_limits(DETECTED, memory_mib=120_000)
        with self.assertRaisesRegex(ValueError, "--pids-limit"):
            hosted_docker.compute_limits(DETECTED, pids=4096)

    def test_gpu_command_exposes_one_device_and_all_limits(self):
        limits = hosted_docker.compute_limits(DETECTED)
        command = hosted_docker.container_command(
            ["docker"], "jax", "gpu", "image", "host", limits,
            Path("/tmp/setup.py"), "abc123", 2, "case",
        )
        rendered = " ".join(str(value) for value in command)
        self.assertIn("--gpus device=2", rendered)
        self.assertIn("--cpus 8", rendered)
        self.assertIn("--memory 24576m", rendered)
        self.assertIn("--pids-limit 2048", rendered)
        self.assertIn("D2L_HOSTED_DEVICE=gpu", rendered)
        self.assertIn("--core-only", rendered)
        self.assertIn("--provider-compatible", rendered)

    def test_cpu_command_hides_gpus(self):
        limits = hosted_docker.compute_limits(DETECTED)
        command = hosted_docker.container_command(
            ["docker"], "tensorflow", "cpu", "image", "host", limits,
            Path("/tmp/setup.py"), "abc123", 0, "case",
        )
        rendered = " ".join(str(value) for value in command)
        self.assertIn("CUDA_VISIBLE_DEVICES=-1", rendered)
        self.assertIn("JAX_PLATFORMS=cpu", rendered)
        self.assertNotIn("--gpus", command)
        self.assertIn("check_pip_delta.py", rendered)
        self.assertIn("cp /repo/d2l/tensorflow.py", rendered)

    def test_published_command_does_not_overlay_worktree_helper(self):
        limits = hosted_docker.compute_limits(DETECTED)
        command = hosted_docker.container_command(
            ["docker"], "jax", "cpu", "image", "host", limits,
            Path("/tmp/setup.py"), "abc123", 0, "case", source="published",
        )
        self.assertNotIn("cp /repo/d2l/jax.py", " ".join(command))

    def test_notebook_preflight_checks_complete_framework_inventory(self):
        limits = hosted_docker.compute_limits(DETECTED)
        command = hosted_docker.container_command(
            ["docker"], "jax", "cpu", "image", "host", limits,
            Path("/tmp/setup.py"), "abc123", 0, "case",
            notebook_preflight=True, result_dir=Path("/tmp/results"),
        )
        rendered = " ".join(command)
        self.assertIn("check_hosted_notebooks.py jax", rendered)
        self.assertIn("--execute-setup", rendered)
        self.assertIn("/tmp/results:/results", rendered)

    def test_runtime_log_scanner_rejects_cuda_and_python_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "case.log"
            path.write_text(
                "normal output\n"
                "2026-01-01: E external/local_xla/cuda_platform.cc:51] "
                "failed call to cuInit: UNKNOWN ERROR (303)\n"
                "AttributeError: missing API\n"
            )
            errors = hosted_docker.runtime_log_errors(path)
        self.assertEqual(len(errors), 2)

    def test_runtime_log_scanner_accepts_clean_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "case.log"
            path.write_text("Hosted runtime contract OK\n")
            self.assertEqual(hosted_docker.runtime_log_errors(path), [])

    def test_runtime_log_scanner_accepts_exact_ipykernel_shutdown_bug(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "case.log"
            path.write_text(
                "Exception ignored in atexit callback: <function shutdown>\n"
                "Traceback (most recent call last):\n"
                "  File \"/usr/lib/python3.12/logging/__init__.py\", line 2265, in shutdown\n"
                "    h.close()\n"
                "  File \"/usr/local/lib/python3.12/dist-packages/ipykernel/iostream.py\", line 446, in close\n"
                "    self.watch_fd_thread.join()\n"
                "AttributeError: 'OutStream' object has no attribute 'watch_fd_thread'\n"
            )
            self.assertEqual(hosted_docker.runtime_log_errors(path), [])

    def test_runtime_log_scanner_rejects_outstream_error_outside_shutdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "case.log"
            path.write_text(
                "Traceback (most recent call last):\n"
                "AttributeError: 'OutStream' object has no attribute 'watch_fd_thread'\n"
            )
            self.assertEqual(len(hosted_docker.runtime_log_errors(path)), 2)


if __name__ == "__main__":
    unittest.main()
