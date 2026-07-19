#!/usr/bin/env python3
"""Occasionally execute every public hosted notebook in Colab GPU containers.

This is the expensive release-quality sweep, not the normal hosted preflight.
For ordinary notebooks it keeps four persistent containers alive, one pinned
to each GPU, and dispatches one notebook at a time to each.  Two-GPU notebooks
run afterwards in two containers pinned to GPU pairs.  Runs are resumable and
retain complete logs plus executed notebooks under one results directory.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_hosted_notebooks import git_revision
from export_hosted_env import load_profile
from run_hosted_docker import DEFAULT_IMAGES, docker_prefix, image_digest, runtime_log_errors
from runtime_env import MULTI_GPU_NOTEBOOKS, SHARED_DATA_NOTEBOOKS


ROOT = Path(__file__).resolve().parents[1]
FRAMEWORKS = ("pytorch", "tensorflow", "jax")
GPU_IDS = (0, 1, 2, 3)
_state_lock = threading.Lock()
_print_lock = threading.Lock()


def run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(command, text=True, **kwargs)


def image_id(docker: list[str], image: str) -> str:
    return run(
        [*docker, "image", "inspect", image, "--format", "{{.Id}}"],
        capture_output=True, check=True,
    ).stdout.strip().removeprefix("sha256:")


def prepared_tag(docker: list[str], base: str, framework: str) -> str:
    env = load_profile(framework)["environment_sha256"][:12]
    return f"d2l-hosted-full-{framework}:{image_id(docker, base)[:12]}-{env}"


def image_exists(docker: list[str], image: str) -> bool:
    return run([*docker, "image", "inspect", image],
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def prepare_image(docker: list[str], base: str, framework: str, tag: str,
                  run_dir: Path, rebuild: bool, gpu: int = 0) -> None:
    if image_exists(docker, tag) and not rebuild:
        print(f"Using prepared image {tag}", flush=True)
        return
    if image_exists(docker, tag):
        run([*docker, "image", "rm", tag], check=True)
    name = f"d2l-full-prepare-{framework}-{os.getpid()}"
    log_path = run_dir / "prepare" / f"{framework}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        *docker, "run", "-d", "--name", name, "--network", "host",
        "--gpus", f"device={gpu}", "--cpus", "12", "--memory", "40g",
        "--pids-limit", "1536", "--shm-size", "4g",
        "--tmpfs", "/tmp:rw,nosuid,nodev,size=12g",
        "-e", "D2L_HOSTED_DEVICE=gpu", "-e", "PIP_NO_CACHE_DIR=1",
        "-e", "TF_CPP_MIN_LOG_LEVEL=1", "-e", "MPLBACKEND=Agg",
        "-e", "XLA_PYTHON_CLIENT_PREALLOCATE=false",
        "-e", "USER=d2l", "-e", "LOGNAME=d2l",
        "-e", "XDG_CACHE_HOME=/tmp/d2l-cache",
        "-e", "TORCH_HOME=/tmp/d2l-cache/torch",
        "-e", "TORCHINDUCTOR_CACHE_DIR=/tmp/d2l-cache/torchinductor",
        "-e", "NUMBA_CACHE_DIR=/tmp/d2l-cache/numba",
        "-v", f"{ROOT}:/repo:ro", "--entrypoint", "sleep", base, "infinity",
    ]
    run(command, check=True, capture_output=True)
    try:
        check = [
            *docker, "exec", name, "python3",
            "/repo/tools/check_hosted_notebooks.py", framework,
            "--root", "/repo/_hosted_notebooks", "--execute-setup",
            "--d2l-root", "/repo",
        ]
        with log_path.open("w", encoding="utf-8") as log:
            result = run(check, stdout=log, stderr=subprocess.STDOUT, timeout=2400)
        findings = runtime_log_errors(log_path)
        if result.returncode or findings:
            raise RuntimeError(
                f"cannot prepare {framework}: exit={result.returncode}, "
                f"log_errors={findings[:3]}; see {log_path}"
            )
        run([*docker, "commit", name, tag], check=True, capture_output=True)
        print(f"Prepared {tag}", flush=True)
    finally:
        run([*docker, "rm", "-f", name], stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL)


def load_state(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"created_at": datetime.now(timezone.utc).isoformat(), "results": {}}


def save_state(path: Path, state: dict) -> None:
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def prepare_result_tree(run_dir: Path, framework: str, revision: str,
                        notebooks: list[str]) -> None:
    root = run_dir / "executed" / framework
    root.mkdir(parents=True, exist_ok=True)
    helper = "torch" if framework == "pytorch" else framework
    for chapter in sorted({Path(rel).parent for rel in notebooks}):
        package = root / chapter / ".d2l-hosted" / revision / "d2l"
        package.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "d2l" / "__init__.py", package / "__init__.py")
        shutil.copy2(ROOT / "d2l" / f"{helper}.py", package / f"{helper}.py")


def start_container(docker: list[str], image: str, framework: str,
                    device_spec: str, suffix: str, run_dir: Path) -> str:
    name = f"d2l-full-{framework}-{suffix}-{os.getpid()}"
    executed = run_dir / "executed" / framework
    (executed / "data").mkdir(parents=True, exist_ok=True)
    (executed / "img").mkdir(parents=True, exist_ok=True)
    gpu_request = (f'"device={device_spec}"' if ',' in device_spec
                   else f"device={device_spec}")
    command = [
        *docker, "run", "-d", "--name", name, "--network", "host",
        "--gpus", gpu_request,
        "--cpus", "12", "--memory", "40g", "--pids-limit", "1024",
        "--shm-size", "8g", "--tmpfs", "/tmp:rw,nosuid,nodev,size=12g",
        "-e", "D2L_HOSTED_DEVICE=gpu", "-e", "MPLBACKEND=Agg",
        "-e", "TF_CPP_MIN_LOG_LEVEL=1", "-e", "TF_FORCE_GPU_ALLOW_GROWTH=true",
        "-e", "XLA_PYTHON_CLIENT_PREALLOCATE=false",
        "-e", "USER=d2l", "-e", "LOGNAME=d2l",
        "-e", "XDG_CACHE_HOME=/tmp/d2l-cache",
        "-e", "TORCH_HOME=/tmp/d2l-cache/torch",
        "-e", "TORCHINDUCTOR_CACHE_DIR=/tmp/d2l-cache/torchinductor",
        "-e", "NUMBA_CACHE_DIR=/tmp/d2l-cache/numba",
        "-e", "OMP_NUM_THREADS=8", "-e", "OPENBLAS_NUM_THREADS=8",
        "-e", "MKL_NUM_THREADS=8", "-e", "TF_NUM_INTRAOP_THREADS=8",
        "-e", "TF_NUM_INTEROP_THREADS=2",
        "-v", f"{ROOT}:/repo:ro", "-v", f"{run_dir}:/results",
        "-v", f"{ROOT / 'data'}:/results/executed/{framework}/data",
        "-v", f"{ROOT / 'img'}:/results/executed/{framework}/img:ro",
        "--entrypoint", "sleep", image, "infinity",
    ]
    try:
        run(command, check=True, capture_output=True)
    except subprocess.CalledProcessError:
        run([*docker, "rm", "-f", name], stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL)
        raise
    return name


def stop_containers(docker: list[str], names: list[str]) -> None:
    for name in names:
        run([*docker, "rm", "-f", name], stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL)


def output_has_error(path: Path) -> bool:
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return True
    return any(
        output.get("output_type") == "error"
        for cell in notebook.get("cells", [])
        for output in cell.get("outputs", [])
    )


def execute_one(docker: list[str], container: str, framework: str, rel: str,
                run_dir: Path, timeout: int) -> dict:
    output = run_dir / "executed" / framework / rel
    output.parent.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "logs" / framework / Path(rel).with_suffix(".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    inside = f"/results/executed/{framework}/{rel}"
    source = f"/repo/_hosted_notebooks/{framework}/{rel}"
    directory = str(Path(inside).parent)
    shell = (
        "set -e\n"
        "mkdir -p /tmp/d2l-home " + shlex.quote(directory) + "\n"
        f"cp {shlex.quote(source)} {shlex.quote(inside)}\n"
        f"cd {shlex.quote(directory)}\n"
        f"timeout --signal=TERM --kill-after=30 {timeout}s "
        "python3 -m jupyter nbconvert --to notebook --execute --inplace "
        f"--ExecutePreprocessor.timeout={max(60, timeout - 120)} "
        "--ExecutePreprocessor.kernel_name=python3 " + shlex.quote(Path(rel).name)
    )
    command = [
        *docker, "exec", "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", "HOME=/tmp/d2l-home", "-e", "USER=d2l", "-e", "LOGNAME=d2l",
        container, "/bin/bash", "-lc", shell,
    ]
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        try:
            result = run(command, stdout=log, stderr=subprocess.STDOUT,
                         timeout=timeout + 120)
            code = result.returncode
        except subprocess.TimeoutExpired:
            code = 124
            log.write(f"Host timeout after {timeout + 120} seconds\n")
    findings = runtime_log_errors(log_path)
    notebook_error = output_has_error(output)
    status = "pass" if code == 0 and not findings and not notebook_error else "fail"
    return {
        "framework": framework, "notebook": rel, "status": status,
        "exit_code": code, "duration_seconds": round(time.monotonic() - started, 1),
        "container": container, "log": str(log_path), "output": str(output),
        "log_errors": findings, "notebook_error_output": notebook_error,
    }


def run_batch(docker: list[str], image: str, framework: str, notebooks: list[str],
              device_specs: list[str], run_dir: Path, state_path: Path,
              state: dict, timeout: int) -> None:
    pending = [rel for rel in notebooks if not (
        state["results"].get(f"{framework}/{rel}", {}).get("status") == "pass"
        and (run_dir / "executed" / framework / rel).is_file()
    )]
    if not pending:
        return
    role = "pair" if any("," in spec for spec in device_specs) else "worker"
    containers = [
        start_container(docker, image, framework, spec,
                        f"{role}-{index}", run_dir)
        for index, spec in enumerate(device_specs)
    ]
    available: queue.Queue[str] = queue.Queue()
    for container in containers:
        available.put(container)

    def task(rel: str) -> dict:
        container = available.get()
        try:
            return execute_one(docker, container, framework, rel, run_dir, timeout)
        finally:
            available.put(container)

    try:
        with ThreadPoolExecutor(max_workers=len(containers)) as pool:
            futures = {pool.submit(task, rel): rel for rel in pending}
            for future in as_completed(futures):
                result = future.result()
                key = f"{framework}/{result['notebook']}"
                with _state_lock:
                    state["results"][key] = result
                    save_state(state_path, state)
                    done = sum(1 for value in state["results"].values()
                               if value.get("framework") == framework)
                with _print_lock:
                    print(
                        f"[{framework} {done}/{len(notebooks)}] "
                        f"{result['status'].upper():4s} "
                        f"{result['duration_seconds']:7.1f}s {result['notebook']}",
                        flush=True,
                    )
    finally:
        stop_containers(docker, containers)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", action="append", choices=FRAMEWORKS)
    parser.add_argument("--image", default=DEFAULT_IMAGES["gpu"])
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--rebuild-images", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--prepare-gpu", type=int, choices=GPU_IDS, default=0,
                        help="GPU used while preparing dependency layers")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--limit", type=int,
                        help="execute only the first N notebooks per framework (canary)")
    parser.add_argument(
        "--notebook", action="append",
        help="execute only this framework-relative notebook (repeatable)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frameworks = list(dict.fromkeys(args.framework or FRAMEWORKS))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = (args.run_dir or ROOT / "logs" / "hosted-full" / timestamp).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "state.json"
    state = load_state(state_path)
    state.update({"revision": git_revision(ROOT), "frameworks": frameworks,
                  "base_image": args.image, "run_dir": str(run_dir)})
    save_state(state_path, state)
    inventories = {}
    for framework in frameworks:
        root = ROOT / "_hosted_notebooks" / framework
        all_notebooks = sorted(p.relative_to(root).as_posix()
                               for p in root.glob("chapter_*/*.ipynb"))
        if args.notebook:
            requested = set(args.notebook)
            available = set(all_notebooks)
            unknown = sorted(requested - available)
            if unknown:
                raise SystemExit(
                    f"{framework}: unknown notebook(s): {', '.join(unknown)}"
                )
            all_notebooks = [rel for rel in all_notebooks if rel in requested]
        if args.limit is not None:
            all_notebooks = all_notebooks[:args.limit]
        regular = [rel for rel in all_notebooks if rel not in MULTI_GPU_NOTEBOOKS]
        multi = [rel for rel in all_notebooks if rel in MULTI_GPU_NOTEBOOKS]
        inventories[framework] = (all_notebooks, regular, multi)
        print(f"{framework}: {len(all_notebooks)} total = "
              f"{len(regular)} single-GPU + {len(multi)} two-GPU", flush=True)
    if args.list:
        return 0

    docker = docker_prefix()
    if not image_exists(docker, args.image):
        raise SystemExit(f"missing base image: {args.image}")
    state["base_digest"] = image_digest(docker, args.image)
    save_state(state_path, state)
    for framework in frameworks:
        all_notebooks, regular, multi = inventories[framework]
        tag = prepared_tag(docker, args.image, framework)
        prepare_image(docker, args.image, framework, tag, run_dir,
                      args.rebuild_images, args.prepare_gpu)
        if args.prepare_only:
            continue
        prepare_result_tree(run_dir, framework, state["revision"], all_notebooks)
        run_batch(docker, tag, framework, regular,
                  [str(gpu) for gpu in GPU_IDS], run_dir, state_path, state,
                  args.timeout)
        run_batch(docker, tag, framework, multi,
                  ["0,1", "2,3"], run_dir, state_path, state, args.timeout)

    if args.prepare_only:
        print(f"Prepared hosted images for {', '.join(frameworks)}", flush=True)
        return 0

    expected_keys = {
        f"{framework}/{rel}"
        for framework in frameworks
        for rel in inventories[framework][0]
    }
    scoped = {key: value for key, value in state["results"].items()
              if key in expected_keys}
    failures = [value for value in scoped.values()
                if value.get("status") != "pass"]
    expected = len(expected_keys)
    passed = sum(1 for value in scoped.values() if value.get("status") == "pass")
    print(f"Full hosted sweep: {passed}/{expected} passed, "
          f"{len(failures)} failed; state={state_path}", flush=True)
    return int(passed != expected or bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
