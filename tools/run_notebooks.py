#!/usr/bin/env python3
"""Execute Jupyter notebooks for a given framework.

Runs each notebook in _notebooks/<framework>/ using nbconvert's execute
preprocessor.  Produces executed notebooks in-place and writes a summary
report to stdout.

With --parallel N, runs N notebooks concurrently, pinning each to a single
GPU via CUDA_VISIBLE_DEVICES (round-robin over --num-gpus).  CPU-only
notebooks run on N additional workers concurrently (no GPU allocated).
Notebooks that require multiple GPUs (see MULTI_GPU_NOTEBOOKS) are run
serially after the parallel batch, with all GPUs visible.

Usage:
    python tools/run_notebooks.py pytorch                          # sequential
    python tools/run_notebooks.py pytorch --parallel 4 --num-gpus 4
    python tools/run_notebooks.py pytorch --glob "chapter_linear*/**"
    python tools/run_notebooks.py pytorch --list                   # dry-run
"""

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from runtime_env import (
    GPU_KEYWORDS, MULTI_GPU_NOTEBOOKS, setup_framework_env,
    MAX_CPUS_PER_GPU_WORKER, MAX_CPUS_PER_CPU_WORKER,
    make_cpu_affinity_fn, worker_cpu_set,
)


NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "_notebooks"

_print_lock = threading.Lock()


def notebook_uses_gpu(nb_path):
    """Return True if any code cell contains GPU-related keywords."""
    try:
        with open(nb_path) as f:
            nb = json.load(f)
        for cell in nb.get("cells", []):
            if cell["cell_type"] == "code":
                src = "".join(cell["source"])
                if any(kw in src for kw in GPU_KEYWORDS):
                    return True
    except Exception:
        pass
    return False


def find_notebooks(framework, glob_pattern=None, files=None):
    fw_dir = NOTEBOOKS_DIR / framework
    if not fw_dir.is_dir():
        print(f"Error: {fw_dir} does not exist. Run ./build.sh notebooks first.",
              file=sys.stderr)
        sys.exit(1)

    if files:
        nbs = sorted(fw_dir / f for f in files)
        missing = [nb for nb in nbs if not nb.is_file()]
        if missing:
            print(f"Error: notebooks not found: {missing}", file=sys.stderr)
            sys.exit(1)
    elif glob_pattern:
        nbs = sorted(fw_dir.glob(glob_pattern))
        nbs = [nb for nb in nbs if nb.suffix == ".ipynb"]
    else:
        nbs = sorted(fw_dir.rglob("*.ipynb"))
    return nbs




def execute_notebook(nb_path, timeout=600, kernel="python3", cuda_devices=None,
                     cpu_affinity=None):
    """Execute a single notebook in-place via jupyter nbconvert.

    cuda_devices: str or None.  If set, passed as CUDA_VISIBLE_DEVICES.
    cpu_affinity: set of CPU indices, or None for no restriction.
    Returns (success: bool, elapsed: float, stderr: str).
    """
    env = os.environ.copy()
    if cuda_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_devices
        if cuda_devices == "":
            from runtime_env import CPU_ONLY_ENV
            env.update(CPU_ONLY_ENV)

    t0 = time.time()
    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "jupyter", "nbconvert",
                "--to", "notebook",
                "--execute",
                "--inplace",
                f"--ExecutePreprocessor.timeout={timeout}",
                f"--ExecutePreprocessor.kernel_name={kernel}",
                str(nb_path),
            ],
            capture_output=True,
            text=True,
            timeout=timeout + 120,
            env=env,
            preexec_fn=make_cpu_affinity_fn(cpu_affinity),
        )
    except subprocess.TimeoutExpired as e:
        elapsed = time.time() - t0
        return False, elapsed, f"TIMEOUT after {elapsed:.0f}s: {e}"

    elapsed = time.time() - t0
    if result.returncode == 0:
        subprocess.run(
            [sys.executable, "-m", "jupyter", "trust", str(nb_path)],
            capture_output=True, timeout=30,
        )
        return True, elapsed, result.stderr
    err = result.stderr.strip() or result.stdout.strip()
    return False, elapsed, err


def _shorten_error(err):
    if not err:
        return "unknown"
    # Keep the last ~25 lines — the tail normally contains the exception
    # type + message, which is what we need to diagnose the failure.
    lines = err.splitlines()
    return "\n".join(lines[-25:])


def _write_error_log(fw_root, rel, stderr):
    """Persist full stderr next to the notebook tree for post-hoc analysis."""
    log_dir = fw_root.parent / "errors" / fw_root.name
    log_path = log_dir / (rel + ".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(stderr or "")


def _run_one(idx, total, nb, rel, timeout, cuda_devices, cpu_affinity=None):
    with _print_lock:
        if cuda_devices is None:
            gpu_tag = "[all GPUs]"
        elif cuda_devices == "":
            gpu_tag = "[CPU]"
        else:
            gpu_tag = f"[GPU {cuda_devices}]"
        print(f"[{idx}/{total}] START {gpu_tag} {rel}", flush=True)
    ok, elapsed, stderr = execute_notebook(nb, timeout=timeout, cuda_devices=cuda_devices,
                                           cpu_affinity=cpu_affinity)
    status = "OK" if ok else "FAIL"
    with _print_lock:
        print(f"[{idx}/{total}] {status} ({elapsed:.1f}s) {rel}", flush=True)
        if not ok:
            short = _shorten_error(stderr)
            print(f"  -- error --\n{short}\n  -- end --", flush=True)
            _write_error_log(nb.parent.parent, rel, stderr)
    return ok, elapsed, stderr


def run_sequential(nbs, timeout, continue_on_error, framework):
    fw_root = NOTEBOOKS_DIR / framework
    passed, failed, errors = 0, 0, []
    for i, nb in enumerate(nbs, 1):
        rel = str(nb.relative_to(fw_root))
        ok, elapsed, stderr = _run_one(i, len(nbs), nb, rel, timeout, cuda_devices=None)
        if ok:
            passed += 1
        else:
            failed += 1
            errors.append((rel, stderr))
            if not continue_on_error:
                break
    return passed, failed, errors




def run_parallel(gpu_nbs, cpu_nbs, timeout, gpu_workers, cpu_workers, num_gpus, framework):
    """Run GPU notebooks on gpu_workers (round-robin GPUs) and CPU notebooks
    on cpu_workers (CUDA_VISIBLE_DEVICES='') concurrently."""
    fw_root = NOTEBOOKS_DIR / framework
    passed, failed, errors = 0, 0, []
    total = len(gpu_nbs) + len(cpu_nbs)

    gpu_pool = queue.Queue()
    for g in range(num_gpus):
        gpu_pool.put(str(g))

    total_workers = gpu_workers + cpu_workers
    _cpu_worker_id = [0]
    _cpu_id_lock = threading.Lock()

    def _run_gpu(idx, nb, rel):
        gpu = gpu_pool.get()
        try:
            cpus = worker_cpu_set(int(gpu), total_workers, MAX_CPUS_PER_GPU_WORKER)
            return _run_one(idx, total, nb, rel, timeout, gpu, cpu_affinity=cpus)
        finally:
            gpu_pool.put(gpu)

    def _run_cpu(idx, nb, rel):
        with _cpu_id_lock:
            wid = gpu_workers + _cpu_worker_id[0]
            _cpu_worker_id[0] = (_cpu_worker_id[0] + 1) % cpu_workers
        cpus = worker_cpu_set(wid, total_workers, MAX_CPUS_PER_CPU_WORKER)
        return _run_one(idx, total, nb, rel, timeout, cuda_devices="", cpu_affinity=cpus)

    with ThreadPoolExecutor(max_workers=gpu_workers) as gpu_exec, \
         ThreadPoolExecutor(max_workers=cpu_workers) as cpu_exec:
        futures = {}
        idx = 0
        for nb in gpu_nbs:
            idx += 1
            rel = str(nb.relative_to(fw_root))
            fut = gpu_exec.submit(_run_gpu, idx, nb, rel)
            futures[fut] = (nb, rel)
        for nb in cpu_nbs:
            idx += 1
            rel = str(nb.relative_to(fw_root))
            fut = cpu_exec.submit(_run_cpu, idx, nb, rel)
            futures[fut] = (nb, rel)

        for fut in as_completed(futures):
            nb, rel = futures[fut]
            ok, elapsed, stderr = fut.result()
            if ok:
                passed += 1
            else:
                failed += 1
                errors.append((rel, stderr))
    return passed, failed, errors


def run_multigpu_serial(multi_gpu_nbs, timeout, framework):
    fw_root = NOTEBOOKS_DIR / framework
    passed, failed, errors = 0, 0, []
    total = len(multi_gpu_nbs)
    for i, nb in enumerate(multi_gpu_nbs, 1):
        rel = str(nb.relative_to(fw_root))
        # cuda_devices=None => inherit; all GPUs visible.
        ok, elapsed, stderr = _run_one(i, total, nb, rel, timeout, cuda_devices=None)
        if ok:
            passed += 1
        else:
            failed += 1
            errors.append((rel, stderr))
    return passed, failed, errors


def main():
    parser = argparse.ArgumentParser(description="Execute d2l notebooks")
    parser.add_argument("framework", choices=["pytorch", "tensorflow", "jax", "mxnet"])
    parser.add_argument("--timeout", type=int, default=3600,
                        help="Per-cell timeout in seconds (default: 3600)")
    parser.add_argument("--glob", type=str, default=None,
                        help="Glob pattern to select notebooks (e.g. 'chapter_linear*/**')")
    parser.add_argument("--files", nargs="*", default=None,
                        help="Specific notebook paths (relative to _notebooks/<fw>/)")
    parser.add_argument("--list", action="store_true",
                        help="List notebooks without executing")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="Continue running after a notebook fails (sequential mode)")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Run N notebooks concurrently (single-GPU notebooks only)")
    parser.add_argument("--num-gpus", type=int, default=1,
                        help="Number of GPUs available for round-robin pinning")
    parser.add_argument("--skip-multi-gpu", action="store_true",
                        help="Do not run notebooks that require multiple GPUs")
    args = parser.parse_args()

    setup_framework_env(args.framework)

    nbs = find_notebooks(args.framework, args.glob, args.files)
    if not nbs:
        print("No notebooks found.")
        return

    fw_root = NOTEBOOKS_DIR / args.framework
    print(f"Found {len(nbs)} notebooks for {args.framework}")

    if args.list:
        for nb in nbs:
            print(f"  {nb.relative_to(NOTEBOOKS_DIR)}")
        return

    # Split into multi-GPU / single-GPU / CPU-only
    single_gpu, cpu_only, multi_gpu = [], [], []
    for nb in nbs:
        rel = str(nb.relative_to(fw_root))
        if rel in MULTI_GPU_NOTEBOOKS:
            multi_gpu.append(nb)
        elif notebook_uses_gpu(nb):
            single_gpu.append(nb)
        else:
            cpu_only.append(nb)

    print(f"  {len(single_gpu)} GPU notebooks, {len(cpu_only)} CPU-only notebooks, "
          f"{len(multi_gpu)} multi-GPU notebooks")
    if args.skip_multi_gpu:
        print("  (multi-GPU notebooks will be skipped)")
        multi_gpu = []

    t0 = time.time()
    all_passed, all_failed, all_errors = 0, 0, []

    if args.parallel > 1:
        gpu_workers = args.parallel
        cpu_workers = args.parallel
        print(f"\n=== Parallel phase: {gpu_workers} GPU workers + {cpu_workers} CPU workers "
              f"across {args.num_gpus} GPUs ===")
        p, f, e = run_parallel(single_gpu, cpu_only, args.timeout,
                               gpu_workers, cpu_workers, args.num_gpus, args.framework)
    else:
        print(f"\n=== Sequential phase ===")
        p, f, e = run_sequential(single_gpu + cpu_only, args.timeout, args.continue_on_error, args.framework)
    all_passed += p; all_failed += f; all_errors += e

    if multi_gpu:
        print(f"\n=== Multi-GPU phase (serial, all GPUs visible): {len(multi_gpu)} notebooks ===")
        p, f, e = run_multigpu_serial(multi_gpu, args.timeout, args.framework)
        all_passed += p; all_failed += f; all_errors += e

    total_elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Results: {all_passed} passed, {all_failed} failed "
          f"(total elapsed {total_elapsed/60:.1f} min)")
    if all_errors:
        print(f"\nFailed notebooks:")
        for rel, _ in all_errors:
            print(f"  - {rel}")
    print(f"{'='*70}")

    sys.exit(1 if all_failed else 0)


if __name__ == "__main__":
    main()
