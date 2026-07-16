#!/usr/bin/env python3
"""Run an occasional Colab CPU/GPU compatibility matrix in Docker.

The normal hosted-notebook checks are intentionally fast and run in the local
framework environments.  This opt-in harness instead starts from Google's
current Colab CPU and GPU images, executes the same generated setup cell that
is published with notebooks, and then runs a small optimizer/device contract.

Cases are serial by design.  Every container has CPU, RAM, PID, shared-memory,
thread, and GPU limits, and is removed after the case.  The large provider
images are retained unless ``--prune-other-image`` is requested explicitly.
"""

from __future__ import annotations

import argparse
import json
import os
import selectors
import shutil
import subprocess
import sys
import time
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from build_hosted_notebooks import _setup_cell, git_revision
from detect_resources import _detect


ROOT = Path(__file__).resolve().parents[1]
FRAMEWORKS = ("pytorch", "tensorflow", "jax")
DEVICES = ("cpu", "gpu")
DEFAULT_IMAGES = {
    "cpu": "us-docker.pkg.dev/colab-images/public/cpu-runtime:latest",
    "gpu": "us-docker.pkg.dev/colab-images/public/runtime:latest",
}
GIB = 1024 ** 3
RUNTIME_ERROR_PATTERNS = (
    re.compile(r"(?:^|\s)E(?:\d{4})?\s+external/", re.IGNORECASE),
    re.compile(r"failed call to cuInit", re.IGNORECASE),
    re.compile(r"^Traceback \(most recent call last\):"),
    re.compile(r"\b(?:VersionError|AttributeError|RuntimeError):"),
)


@dataclass(frozen=True)
class Limits:
    cpus: int
    memory_mib: int
    pids: int
    shm_mib: int
    tmp_mib: int


def compute_limits(detected: dict, cpus: int | None = None,
                   memory_mib: int | None = None,
                   pids: int | None = None) -> Limits:
    """Return conservative per-container limits and reject unsafe overrides."""
    host_cpus = max(1, int(detected.get("ncpu") or 1))
    available = int(detected.get("mem_avail_mib") or 0)
    host_nproc = int(detected.get("ulimit_nproc") or 4096)
    safe_memory = max(4096, available * 50 // 100) if available else 24576
    safe_pids = max(128, min(4096, host_nproc * 75 // 100))

    wanted_cpus = cpus if cpus is not None else min(8, host_cpus)
    wanted_memory = memory_mib if memory_mib is not None else min(24576, safe_memory)
    wanted_pids = pids if pids is not None else min(2048, safe_pids)
    if not 1 <= wanted_cpus <= host_cpus:
        raise ValueError(f"--cpus must be between 1 and {host_cpus}")
    if not 1024 <= wanted_memory <= safe_memory:
        raise ValueError(
            f"--memory-mib must be between 1024 and {safe_memory} "
            "(50% of currently available RAM)"
        )
    if not 64 <= wanted_pids <= safe_pids:
        raise ValueError(
            f"--pids-limit must be between 64 and {safe_pids} "
            "(capped at 4096 and 75% of RLIMIT_NPROC)"
        )
    return Limits(
        cpus=wanted_cpus,
        memory_mib=wanted_memory,
        pids=wanted_pids,
        shm_mib=min(2048, max(512, wanted_memory // 8)),
        tmp_mib=min(12288, max(4096, wanted_memory // 2)),
    )


def command_ok(command: list[str]) -> bool:
    try:
        return subprocess.run(
            command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=20,
        ).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def docker_prefix() -> list[str]:
    candidates = (["docker"], ["sudo", "-n", "docker"])
    for candidate in candidates:
        if command_ok([*candidate, "info"]):
            return list(candidate)
    raise RuntimeError(
        "Docker is unavailable. Start the daemon and grant this user Docker "
        "access or passwordless `sudo docker`."
    )


def image_present(docker: list[str], image: str) -> bool:
    return command_ok([*docker, "image", "inspect", image])


def image_digest(docker: list[str], image: str) -> str:
    result = subprocess.run(
        [*docker, "image", "inspect", image, "--format", "{{json .RepoDigests}}"],
        capture_output=True, text=True, check=True,
    )
    digests = json.loads(result.stdout)
    return digests[0] if digests else image


def stream_command(command: list[str], log, timeout: int | None = None,
                   timeout_cleanup: list[str] | None = None) -> int:
    """Stream a subprocess to the terminal and log, with a real wall timeout."""
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    assert process.stdout is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    started = time.monotonic()
    while process.poll() is None:
        for key, _ in selector.select(timeout=1):
            line = key.fileobj.readline()
            if line:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
        if timeout and time.monotonic() - started > timeout:
            message = f"Timed out after {timeout} seconds\n"
            print(message, end="", flush=True)
            log.write(message)
            if timeout_cleanup:
                subprocess.run(
                    timeout_cleanup, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL, timeout=30,
                )
            process.kill()
            process.wait()
            return 124
    for line in process.stdout:
        print(line, end="", flush=True)
        log.write(line)
    return process.returncode


def runtime_log_errors(path: Path) -> list[str]:
    """Return unexpected error-level diagnostics from a completed case log."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    ignored = set()
    for start, line in enumerate(lines):
        if not line.startswith("Exception ignored in atexit callback:"):
            continue
        stop = min(len(lines), start + 16)
        block = lines[start:stop]
        try:
            end = next(
                start + offset for offset, value in enumerate(block)
                if value == ("AttributeError: 'OutStream' object has no "
                             "attribute 'watch_fd_thread'")
            )
        except StopIteration:
            continue
        exact_shutdown = lines[start:end + 1]
        if (any("logging/__init__.py" in value and "shutdown" in value
                for value in exact_shutdown) and
                any("ipykernel/iostream.py" in value
                    for value in exact_shutdown)):
            ignored.update(range(start, end + 1))

    findings = []
    for offset, line in enumerate(lines):
        if offset in ignored:
            continue
        if any(pattern.search(line) for pattern in RUNTIME_ERROR_PATTERNS):
            findings.append(f"line {offset + 1}: {line}")
    return findings


def pull_image(docker: list[str], image: str, log_dir: Path) -> None:
    log_path = log_dir / ("pull-" + image.rsplit("/", 1)[-1].replace(":", "-") + ".log")
    with log_path.open("w", encoding="utf-8") as log:
        code = stream_command([*docker, "pull", image], log)
    if code:
        raise RuntimeError(f"docker pull failed for {image}; see {log_path}")


def choose_network(docker: list[str], image: str, requested: str,
                   limits: Limits) -> str:
    if requested != "auto":
        return requested
    name = f"d2l-hosted-network-probe-{os.getpid()}"
    probe = [
        *docker, "run", "--rm", "--name", name, "--network", "bridge", "--cpus", "1",
        "--memory", "1024m", "--pids-limit", "128", "--read-only",
        "--tmpfs", "/tmp:rw,nosuid,nodev,size=256m", "--entrypoint",
        "python3", image, "-c",
        "import socket; socket.getaddrinfo('pypi.org', 443)",
    ]
    if command_ok(probe):
        return "bridge"
    subprocess.run(
        [*docker, "rm", "-f", name], stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, timeout=30,
    )
    if sys.platform.startswith("linux"):
        print("Docker bridge DNS failed; using host networking for this run.")
        return "host"
    raise RuntimeError("Docker bridge DNS failed and host networking is unavailable")


def setup_script(framework: str, revision: str, path: Path) -> None:
    source = "".join(_setup_cell(framework, revision)["source"])
    path.write_text(source, encoding="utf-8")


def container_command(docker: list[str], framework: str, device: str,
                      image: str, network: str, limits: Limits,
                      setup_path: Path, revision: str, gpu: int,
                      name: str, source: str = "worktree",
                      notebook_preflight: bool = False,
                      result_dir: Path | None = None) -> list[str]:
    command = [
        *docker, "run", "--rm", "--name", name,
        "--network", network,
        "--cpus", str(limits.cpus),
        "--memory", f"{limits.memory_mib}m",
        "--pids-limit", str(limits.pids),
        "--shm-size", f"{limits.shm_mib}m",
        "--tmpfs", f"/tmp:rw,nosuid,nodev,size={limits.tmp_mib}m",
        "-e", "PIP_NO_CACHE_DIR=1",
        "-e", f"D2L_HOSTED_DEVICE={device}",
        "-e", f"OMP_NUM_THREADS={limits.cpus}",
        "-e", f"OPENBLAS_NUM_THREADS={limits.cpus}",
        "-e", f"MKL_NUM_THREADS={limits.cpus}",
        "-e", f"NUMEXPR_MAX_THREADS={limits.cpus}",
        "-e", "MPLBACKEND=Agg",
        "-e", "TF_CPP_MIN_LOG_LEVEL=1",
        "-e", "TF_FORCE_GPU_ALLOW_GROWTH=true",
        "-e", "XLA_PYTHON_CLIENT_PREALLOCATE=false",
        "-v", f"{ROOT}:/repo:ro",
        "-v", f"{setup_path}:/harness/setup.py:ro",
        "-w", "/work",
    ]
    if result_dir is not None:
        command.extend(["-v", f"{result_dir}:/results"])
    if device == "gpu":
        command.extend(["--gpus", f"device={gpu}"])
    else:
        command.extend(["-e", "CUDA_VISIBLE_DEVICES=-1", "-e", "JAX_PLATFORMS=cpu"])
    d2l_root = f"/work/.d2l-hosted/{revision}"
    helper = "torch" if framework == "pytorch" else framework
    candidate_overlay = ""
    if source == "worktree":
        candidate_overlay = (
            f"  cp /repo/d2l/__init__.py {d2l_root}/d2l/__init__.py\n"
            f"  cp /repo/d2l/{helper}.py {d2l_root}/d2l/{helper}.py\n"
        )
    preflight = ""
    if notebook_preflight:
        preflight_root = "/repo" if source == "worktree" else d2l_root
        report = f"/results/preflight-{device}-{framework}.json"
        preflight = (
            "if [ $status -eq 0 ]; then\n"
            f"  python3 /repo/tools/check_hosted_notebooks.py {framework} "
            f"--root /repo/_hosted_notebooks --execute-setup "
            f"--d2l-root {preflight_root} --output {report} || status=$?\n"
            "fi\n"
        )
    shell = (
        "status=0\n"
        "python3 -m pip check > /tmp/d2l-pip-before 2>&1 || true\n"
        "python3 /harness/setup.py || status=$?\n"
        "if [ $status -eq 0 ]; then\n"
        + candidate_overlay +
        f"  python3 /repo/tools/check_hosted_runtime.py {framework} "
        f"--device {device} --core-only --provider-compatible "
        f"--d2l-root {d2l_root} || status=$?\n"
        "fi\n"
        + preflight +
        "python3 -m pip check > /tmp/d2l-pip-after 2>&1 || true\n"
        "pip_status=0\n"
        "python3 /repo/tools/check_pip_delta.py /tmp/d2l-pip-before "
        "/tmp/d2l-pip-after || pip_status=$?\n"
        "if [ $status -eq 0 ] && [ $pip_status -ne 0 ]; then status=$pip_status; fi\n"
        "exit $status"
    )
    command.extend(["--entrypoint", "/bin/bash", image, "-lc", shell])
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--framework", action="append", choices=FRAMEWORKS,
        help="framework to test; repeat as needed (default: all)",
    )
    parser.add_argument(
        "--device", action="append", choices=DEVICES,
        help="device class to test; repeat as needed (default: CPU and GPU)",
    )
    parser.add_argument("--cpu-image", default=DEFAULT_IMAGES["cpu"])
    parser.add_argument("--gpu-image", default=DEFAULT_IMAGES["gpu"])
    parser.add_argument("--pull", choices=("missing", "always", "never"),
                        default="missing")
    parser.add_argument(
        "--prune-other-image", action="store_true",
        help="remove the other Colab image before a pull on disk-constrained hosts",
    )
    parser.add_argument("--network", choices=("auto", "bridge", "host"),
                        default="auto")
    parser.add_argument("--gpu", type=int, default=0,
                        help="expose only this physical GPU (default: 0)")
    parser.add_argument("--cpus", type=int)
    parser.add_argument("--memory-mib", type=int)
    parser.add_argument("--pids-limit", type=int)
    parser.add_argument("--timeout", type=int, default=2700,
                        help="seconds per case (default: 2700)")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--source", choices=("worktree", "published"), default="worktree",
        help="test local candidate d2l helpers or the revision fetched by the notebook",
    )
    parser.add_argument(
        "--notebook-preflight", action="store_true",
        help="compile/import/API-check every public notebook in each selected framework",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    detected = _detect()
    try:
        limits = compute_limits(
            detected, args.cpus, args.memory_mib, args.pids_limit
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    frameworks = list(dict.fromkeys(args.framework or FRAMEWORKS))
    devices = list(dict.fromkeys(args.device or DEVICES))
    images = {"cpu": args.cpu_image, "gpu": args.gpu_image}
    revision = git_revision(ROOT)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    log_dir = args.log_dir or ROOT / "logs" / "hosted-docker" / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)

    docker = ["docker"] if args.dry_run else docker_prefix()
    if "gpu" in devices and not args.dry_run:
        gpu_count = int(detected.get("num_gpus") or 0)
        if not 0 <= args.gpu < gpu_count:
            raise SystemExit(f"--gpu must be between 0 and {gpu_count - 1}")
        info = subprocess.run(
            [*docker, "info", "--format", "{{json .Runtimes}}"],
            capture_output=True, text=True, check=True,
        ).stdout
        if '"nvidia"' not in info:
            raise SystemExit(
                "Docker has no NVIDIA runtime; install/configure "
                "nvidia-container-toolkit first"
            )

    print(f"Hosted Docker matrix: frameworks={frameworks} devices={devices}")
    print(f"Limits per serial case: {limits}")
    print(f"Logs: {log_dir}")
    results = []
    for device in devices:
        image = images[device]
        if not args.dry_run:
            present = image_present(docker, image)
            if args.pull == "always" or (args.pull == "missing" and not present):
                if args.prune_other_image:
                    other = images["gpu" if device == "cpu" else "cpu"]
                    if other != image and image_present(docker, other):
                        print(f"Removing other provider image to free space: {other}")
                        subprocess.run([*docker, "image", "rm", other], check=True)
                free_gib = shutil.disk_usage("/var/lib/docker").free / GIB
                print(f"Docker storage free before pull: {free_gib:.1f} GiB")
                pull_image(docker, image, log_dir)
            elif args.pull == "never" and not present:
                raise SystemExit(f"required image is missing: {image}")
            digest = image_digest(docker, image)
            network = choose_network(docker, image, args.network, limits)
        else:
            digest, network = image, args.network

        for framework in frameworks:
            setup_path = log_dir / f"setup-{device}-{framework}.py"
            setup_script(framework, revision, setup_path)
            name = f"d2l-hosted-{device}-{framework}-{os.getpid()}"
            command = container_command(
                docker, framework, device, image, network, limits,
                setup_path, revision, args.gpu, name,
                source=args.source,
                notebook_preflight=args.notebook_preflight,
                result_dir=log_dir,
            )
            log_path = log_dir / f"{device}-{framework}.log"
            print(f"\n=== {framework}/{device} ===")
            started = time.monotonic()
            if args.dry_run:
                print(" ".join(command))
                code = 0
            else:
                with log_path.open("w", encoding="utf-8") as log:
                    code = stream_command(
                        command, log, timeout=args.timeout,
                        timeout_cleanup=[*docker, "rm", "-f", name],
                    )
            log_errors = [] if args.dry_run else runtime_log_errors(log_path)
            if log_errors:
                print("Unexpected error diagnostics in case log:")
                for finding in log_errors:
                    print(f"  {finding}")
                if code == 0:
                    code = 1
            result = {
                "framework": framework,
                "device": device,
                "status": "pass" if code == 0 else "fail",
                "exit_code": code,
                "duration_seconds": round(time.monotonic() - started, 1),
                "image": digest,
                "network": network,
                "source": args.source,
                "notebook_preflight": args.notebook_preflight,
                "preflight_report": (
                    str(log_dir / f"preflight-{device}-{framework}.json")
                    if args.notebook_preflight else None
                ),
                "limits": asdict(limits),
                "log": str(log_path),
                "log_errors": log_errors,
            }
            results.append(result)
            print(f"{result['status'].upper()}: {framework}/{device} "
                  f"({result['duration_seconds']}s)")
            if code and args.fail_fast:
                break
        if args.fail_fast and results and results[-1]["exit_code"]:
            break

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "revision": revision,
        "host": detected,
        "results": results,
    }
    summary_path = log_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("\nHosted Docker matrix summary")
    for result in results:
        print(f"  {result['framework']:10s} {result['device']:3s} "
              f"{result['status']:4s} {result['duration_seconds']:7.1f}s")
    print(f"Summary: {summary_path}")
    return int(any(result["exit_code"] for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
