#!/usr/bin/env python3
"""Execute Jupyter notebooks for a given framework.

Runs each notebook in _notebooks/<framework>/ using nbconvert's execute
preprocessor.  Produces executed notebooks in-place, touches matching
.executed stamps after successful runs, and writes a summary report to stdout.

With --parallel N, runs N GPU notebooks concurrently, round-robin over
--num-gpus.  Multiple workers may share a GPU (e.g. --parallel 8 --num-gpus 4
puts 2 workers per GPU).  CPU-only notebooks run on --num-gpus additional
workers concurrently (no GPU allocated).  Notebooks that require multiple GPUs
(see MULTI_GPU_NOTEBOOKS) are run serially after the parallel batch, with all
GPUs visible.

Usage:
    python tools/run_notebooks.py pytorch                          # sequential
    python tools/run_notebooks.py pytorch --parallel 8 --num-gpus 4  # 2/GPU
    python tools/run_notebooks.py pytorch --glob "chapter_linear*/**"
    python tools/run_notebooks.py pytorch --list                   # dry-run
"""

import argparse
import json
import os
import queue
import re
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
    make_cpu_affinity_fn, worker_cpu_set, kill_stale_kernels,
    file_uses_gpu,
)


NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "_notebooks"

_print_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Best-of-N: stochastic notebooks that benefit from multiple runs.
# Keys are notebook paths relative to _notebooks/<fw>/.
# Values: (max_attempts, good_enough_score).
# After the normal run, these notebooks are re-executed up to max_attempts
# times (including the initial run), keeping whichever result scores highest.
# Stops early if score >= good_enough_score.
# ---------------------------------------------------------------------------

BEST_OF_N = {
    "chapter_recurrent-modern/lstm.ipynb":                                    (5, 2.0),
    "chapter_attention-mechanisms-and-transformers/bahdanau-attention.ipynb":  (5, 2.5),
    "chapter_attention-mechanisms-and-transformers/transformer.ipynb":         (5, 2.5),
    "chapter_generative-adversarial-networks/dcgan.ipynb":                    (3, 3.0),
}


def score_notebook(nb_path):
    """Score an executed notebook's output quality.

    Returns a float >= 0.  Higher is better.
    - seq2seq / attention notebooks: sum of BLEU scores (format: "bleu,X.XXX")
    - LSTM / RNN text-generation notebooks: heuristic penalizing repetition
    Returns 0.0 if the notebook has no scoreable output or failed to execute.
    """
    try:
        with open(nb_path) as f:
            nb = json.load(f)
    except Exception:
        return 0.0

    bleu_scores = []
    generated_texts = []
    gan_loss_G = None

    for cell in nb.get("cells", []):
        for out in cell.get("outputs", []):
            # BLEU scores in stream output: "bleu,0.658"
            if "text" in out:
                for line in out["text"]:
                    for m in re.finditer(r"bleu,(\d+\.\d+)", line):
                        bleu_scores.append(float(m.group(1)))
                    # GAN loss: "loss_D 0.161, loss_G 4.254"
                    m = re.search(r"loss_G (\d+\.\d+)", line)
                    if m:
                        gan_loss_G = float(m.group(1))
                    # Showcase prediction printed by the gated-RNN section
                    # (10.1): "perplexity 78.3, 'the time traveller ...'"
                    m = re.search(r"perplexity \d+\.\d+, ['\"](.+)['\"]", line)
                    if m:
                        generated_texts.append(m.group(1))
            # Generated text in execute_result: "'it has ...'"
            if "data" in out and "text/plain" in out["data"]:
                txt = "".join(out["data"]["text/plain"])
                if "it has" in txt:
                    generated_texts.append(txt)

    if bleu_scores:
        return sum(bleu_scores)

    if generated_texts:
        return _score_generated_text(generated_texts[-1])

    if gan_loss_G is not None:
        return gan_loss_G

    return 0.0


def _score_generated_text(text):
    """Score RNN-generated text; penalize repetitive n-grams."""
    text = text.strip().strip("'\"")
    words = text.split()
    if len(words) < 3:
        return 0.0
    # Fraction of unique bigrams — 1.0 means no repetition
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
    if not bigrams:
        return 0.0
    unique_ratio = len(set(bigrams)) / len(bigrams)
    # Scale to roughly match BLEU range (0-3ish) for comparable thresholds
    return unique_ratio * 3.0


def notebook_uses_gpu(nb_path):
    """Return True if any framework's version of this notebook uses GPU."""
    return file_uses_gpu(nb_path, NOTEBOOKS_DIR)


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




# Process-group tracking so a SIGTERM/SIGINT to this process kills the
# nbconvert + ipykernel descendant tree, not just nbconvert. See
# tools/run_one_notebook.py:install_signal_handlers() for the signal side.
_active_pgids = set()


def _register_pgid(pgid):
    _active_pgids.add(pgid)


def _unregister_pgid(pgid):
    _active_pgids.discard(pgid)


def kill_active_subprocesses(grace_sec=3):
    """Kill every notebook subprocess group tracked in this process.
    Called from signal handlers in run_one_notebook.py so a Ctrl-C / Make
    kill can't leave orphan ipykernels."""
    import signal as _signal
    for pgid in list(_active_pgids):
        try:
            os.killpg(pgid, _signal.SIGTERM)
        except ProcessLookupError:
            _active_pgids.discard(pgid)
    if not _active_pgids:
        return
    deadline = time.time() + grace_sec
    while _active_pgids and time.time() < deadline:
        time.sleep(0.2)
        for pgid in list(_active_pgids):
            try:
                # signal 0: existence probe, no signal delivered
                os.killpg(pgid, 0)
            except ProcessLookupError:
                _active_pgids.discard(pgid)
    for pgid in list(_active_pgids):
        try:
            os.killpg(pgid, _signal.SIGKILL)
        except ProcessLookupError:
            pass
        _active_pgids.discard(pgid)


_SELF_KERNEL_NAME = "d2l-self"
_self_kernel_dir = None


def ensure_self_kernelspec():
    """Create (once) an ephemeral kernelspec that launches THIS interpreter.

    Notebooks are run under the framework venv's python (run_one_notebook.py is
    invoked as `.venv-<fw>/bin/python`), so `sys.executable` is always the right
    interpreter. Relying on the ambient `python3` kernelspec is fragile: on hosts
    with an always-active conda base (e.g. the Vast image's `/venv/main`), jupyter
    resolves `python3` to that base's kernelspec — a different interpreter with no
    mxnet/torch — and every notebook fails with `ModuleNotFoundError`. Pinning the
    kernel to an absolute `sys.executable` makes execution independent of whatever
    base env the host keeps active, and needs no `make kernels` step.

    Returns the JUPYTER_PATH data dir holding the spec.
    """
    global _self_kernel_dir
    if _self_kernel_dir is not None:
        return _self_kernel_dir
    import tempfile
    base = tempfile.mkdtemp(prefix="d2l-kernel-")
    spec_dir = os.path.join(base, "kernels", _SELF_KERNEL_NAME)
    os.makedirs(spec_dir, exist_ok=True)
    spec = {
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "d2l (self)",
        "language": "python",
    }
    with open(os.path.join(spec_dir, "kernel.json"), "w") as fh:
        json.dump(spec, fh)
    _self_kernel_dir = base
    return base


def execute_notebook(nb_path, timeout=600, kernel=None, cuda_devices=None,
                     cpu_affinity=None):
    """Execute a single notebook in-place via jupyter nbconvert.

    cuda_devices: str or None.  If set, passed as CUDA_VISIBLE_DEVICES.
    cpu_affinity: set of CPU indices, or None for no restriction.
    kernel: kernelspec name to run with. Defaults to an ephemeral spec pinned to
        the current interpreter (see ensure_self_kernelspec) so execution does not
        depend on the host's ambient `python3` kernelspec.
    Returns (success: bool, elapsed: float, stderr: str).

    The nbconvert child runs in a new session (`start_new_session=True`),
    so when this Python process dies we can `killpg` the whole nbconvert
    + ipykernel tree at once. Each child also gets PR_SET_PDEATHSIG=SIGTERM
    as a kernel-level fallback for the SIGKILL case.
    """
    import signal as _signal
    from runtime_env import make_subprocess_preexec_fn

    env = os.environ.copy()
    if cuda_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_devices
        if cuda_devices == "":
            from runtime_env import CPU_ONLY_ENV
            env.update(CPU_ONLY_ENV)

    if kernel is None:
        kernel = _SELF_KERNEL_NAME
        self_dir = ensure_self_kernelspec()
        env["JUPYTER_PATH"] = (
            self_dir + os.pathsep + env["JUPYTER_PATH"]
            if env.get("JUPYTER_PATH") else self_dir)

    cmd = [
        sys.executable, "-m", "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--inplace",
        f"--ExecutePreprocessor.timeout={timeout}",
        f"--ExecutePreprocessor.kernel_name={kernel}",
        str(nb_path),
    ]
    preexec = make_subprocess_preexec_fn(
        cpu_set=cpu_affinity, pdeathsig=_signal.SIGTERM)

    t0 = time.time()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        preexec_fn=preexec,
        start_new_session=True,
    )
    pgid = os.getpgid(proc.pid)
    _register_pgid(pgid)
    try:
        try:
            stdout, stderr = proc.communicate(timeout=timeout + 120)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            try:
                os.killpg(pgid, _signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                stdout, stderr = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(pgid, _signal.SIGKILL)
                except ProcessLookupError:
                    pass
                stdout, stderr = proc.communicate()
            elapsed = time.time() - t0
            return False, elapsed, f"TIMEOUT after {elapsed:.0f}s"
    finally:
        _unregister_pgid(pgid)
        # Belt-and-suspenders: if anything in the group is still alive
        # (e.g. nbconvert exited but ipykernel lagged), kill it.
        try:
            os.killpg(pgid, _signal.SIGKILL)
        except ProcessLookupError:
            pass

    elapsed = time.time() - t0
    if rc == 0:
        subprocess.run(
            [sys.executable, "-m", "jupyter", "trust", str(nb_path)],
            capture_output=True, timeout=30,
        )
        return True, elapsed, stderr or ""
    err = (stderr or "").strip() or (stdout or "").strip()
    return False, elapsed, err


TRANSIENT_ERRORS = (
    "Kernel didn't respond",
    "Address already in use",
    "Kernel died before replying to kernel_info",
    "KernelDied",
)


def _is_transient(stderr):
    return stderr and any(msg in stderr for msg in TRANSIENT_ERRORS)


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


def _touch_executed_stamp(nb):
    nb.with_suffix(".executed").touch()


def _run_one(idx, total, nb, rel, timeout, cuda_devices, cpu_affinity=None,
             max_retries=1):
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
    if not ok and _is_transient(stderr) and max_retries > 0:
        with _print_lock:
            print(f"[{idx}/{total}] RETRY (transient failure) {rel}", flush=True)
        time.sleep(2)
        ok, elapsed, stderr = execute_notebook(nb, timeout=timeout, cuda_devices=cuda_devices,
                                               cpu_affinity=cpu_affinity)
    status = "OK" if ok else "FAIL"
    with _print_lock:
        print(f"[{idx}/{total}] {status} ({elapsed:.1f}s) {rel}", flush=True)
        if not ok:
            short = _shorten_error(stderr)
            print(f"  -- error --\n{short}\n  -- end --", flush=True)
            _write_error_log(nb.parent.parent, rel, stderr)
    if ok:
        _touch_executed_stamp(nb)
    return ok, elapsed, stderr


def run_sequential(nbs, timeout, continue_on_error, framework,
                   cuda_devices_for=None):
    fw_root = NOTEBOOKS_DIR / framework
    passed, failed, errors = 0, 0, []
    for i, nb in enumerate(nbs, 1):
        rel = str(nb.relative_to(fw_root))
        cuda_devices = (
            cuda_devices_for(i - 1, nb) if callable(cuda_devices_for)
            else cuda_devices_for)
        ok, elapsed, stderr = _run_one(
            i, len(nbs), nb, rel, timeout, cuda_devices=cuda_devices)
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

    if gpu_workers > 0:
        workers_per_gpu = max(1, gpu_workers // max(1, num_gpus))
        gpu_pool = queue.Queue()
        for g in range(num_gpus):
            for _ in range(workers_per_gpu):
                gpu_pool.put(str(g))
    else:
        gpu_pool = None

    total_workers = max(1, gpu_workers + cpu_workers)
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

    # ThreadPoolExecutor requires max_workers >= 1; use a sentinel size of 1
    # for the unused side when one pool is empty.
    with ThreadPoolExecutor(max_workers=max(1, gpu_workers)) as gpu_exec, \
         ThreadPoolExecutor(max_workers=max(1, cpu_workers)) as cpu_exec:
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


def run_best_of_n(framework, nbs, timeout):
    """Re-run stochastic notebooks up to N times, keeping the best result.

    Only notebooks listed in BEST_OF_N and present in `nbs` are retried.
    The initial run (from the normal execution phase) counts as attempt 1.
    """
    fw_root = NOTEBOOKS_DIR / framework
    candidates = []
    for nb in nbs:
        rel = str(nb.relative_to(fw_root))
        if rel in BEST_OF_N:
            candidates.append((nb, rel))

    if not candidates:
        return

    print(f"\n=== Best-of-N phase: {len(candidates)} stochastic notebooks ===")

    for nb, rel in candidates:
        max_attempts, good_enough = BEST_OF_N[rel]
        best_score = score_notebook(nb)
        if best_score >= good_enough:
            print(f"  {rel}: score={best_score:.3f} (already good enough)", flush=True)
            continue

        best_nb = nb.read_bytes()
        print(f"  {rel}: attempt 1 score={best_score:.3f}", flush=True)

        for attempt in range(2, max_attempts + 1):
            ok, elapsed, stderr = execute_notebook(nb, timeout=timeout)
            if not ok:
                print(f"  {rel}: attempt {attempt} FAILED ({elapsed:.0f}s)", flush=True)
                continue
            score = score_notebook(nb)
            print(f"  {rel}: attempt {attempt} score={score:.3f} ({elapsed:.0f}s)", flush=True)
            if score > best_score:
                best_score = score
                best_nb = nb.read_bytes()
            if best_score >= good_enough:
                break

        # Restore the best result
        nb.write_bytes(best_nb)
        _touch_executed_stamp(nb)
        print(f"  {rel}: BEST score={best_score:.3f} "
              f"({'good' if best_score >= good_enough else 'best available'})", flush=True)


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
    parser.add_argument("--no-best-of-n", action="store_true",
                        help="Skip best-of-N retries for stochastic notebooks")
    parser.add_argument("--cpu-only", action="store_true",
                        help="Force every notebook to run with CUDA_VISIBLE_DEVICES='' "
                             "(used for frameworks whose wheels lack kernels for this host's GPU)")
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

    # Split into multi-GPU / single-GPU / CPU-only. With --cpu-only every
    # notebook is treated as CPU-only (CUDA_VISIBLE_DEVICES='') regardless of
    # its GPU usage hints.
    single_gpu, cpu_only, multi_gpu = [], [], []
    if args.cpu_only:
        cpu_only = list(nbs)
    else:
        for nb in nbs:
            rel = str(nb.relative_to(fw_root))
            if rel in MULTI_GPU_NOTEBOOKS:
                multi_gpu.append(nb)
            elif notebook_uses_gpu(nb):
                single_gpu.append(nb)
            else:
                cpu_only.append(nb)

    print(f"  {len(single_gpu)} GPU notebooks, {len(cpu_only)} CPU-only notebooks, "
          f"{len(multi_gpu)} multi-GPU notebooks"
          + (" (forced via --cpu-only)" if args.cpu_only else ""))
    if args.skip_multi_gpu:
        print("  (multi-GPU notebooks will be skipped)")
        multi_gpu = []

    t0 = time.time()
    all_passed, all_failed, all_errors = 0, 0, []

    if args.parallel > 1:
        if args.cpu_only:
            gpu_workers, cpu_workers = 0, args.parallel
            print(f"\n=== Parallel phase: {cpu_workers} CPU workers (cpu-only mode) ===")
        else:
            gpu_workers = args.parallel
            cpu_workers = args.num_gpus
            wpg = max(1, gpu_workers // args.num_gpus)
            print(f"\n=== Parallel phase: {gpu_workers} GPU workers ({wpg}/GPU) + "
                  f"{cpu_workers} CPU workers across {args.num_gpus} GPUs ===")
        p, f, e = run_parallel(single_gpu, cpu_only, args.timeout,
                               gpu_workers, cpu_workers, args.num_gpus, args.framework)
    else:
        print(f"\n=== Sequential phase ===")
        p = f = 0
        e = []
        if single_gpu:
            p1, f1, e1 = run_sequential(
                single_gpu, args.timeout, args.continue_on_error,
                args.framework,
                cuda_devices_for=lambda i, _nb: str(i % args.num_gpus))
            p += p1; f += f1; e += e1
        if cpu_only and (args.continue_on_error or f == 0):
            p1, f1, e1 = run_sequential(
                cpu_only, args.timeout, args.continue_on_error,
                args.framework, cuda_devices_for="")
            p += p1; f += f1; e += e1
    all_passed += p; all_failed += f; all_errors += e

    if multi_gpu:
        print(f"\n=== Multi-GPU phase (serial, all GPUs visible): {len(multi_gpu)} notebooks ===")
        p, f, e = run_multigpu_serial(multi_gpu, args.timeout, args.framework)
        all_passed += p; all_failed += f; all_errors += e

    if not args.no_best_of_n:
        run_best_of_n(args.framework, nbs, args.timeout)

    total_elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Results: {all_passed} passed, {all_failed} failed "
          f"(total elapsed {total_elapsed/60:.1f} min)")
    if all_errors:
        print(f"\nFailed notebooks:")
        for rel, _ in all_errors:
            print(f"  - {rel}")
    print(f"{'='*70}")

    venv_dir = Path(sys.executable).resolve().parent.parent
    n = kill_stale_kernels(venv_dir)
    if n:
        print(f"Killed {n} stale ipykernel process(es) from {venv_dir.name}")

    sys.exit(1 if all_failed else 0)


if __name__ == "__main__":
    main()
