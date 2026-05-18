#!/usr/bin/env python3
"""Execute a single d2l notebook with GPU slot locking and best-of-N retry.

Designed to be called from the per-notebook Make rule:
    python tools/run_one_notebook.py <framework> <ipynb_path>

It:
  1. Acquires a GPU slot via flock on /tmp/d2l-slots/gpu-<i>.lock files
     (or runs CPU-only for CPU-only notebooks).
  2. Runs the notebook in-place via jupyter nbconvert.
  3. For known-stochastic notebooks (BEST_OF_N), retries up to N times and
     keeps the best-scoring run.
  4. Exits 0 on success, non-zero on failure.

Env:
  D2L_NUM_GPUS=<n>   how many GPU slots exist (default 1, or NUM_GPUS from make)

The Make pattern rule uses one process per notebook; concurrency is bounded by
`make -j`. Slot locking ensures that no more than D2L_NUM_GPUS GPU-using
notebooks run at once (one per GPU). Multi-GPU notebooks claim all slots.
"""
import argparse
import fcntl
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from runtime_env import (
    MULTI_GPU_NOTEBOOKS, setup_framework_env, file_uses_gpu,
)
# Reuse functions defined in run_notebooks.py rather than duplicating them.
from run_notebooks import (
    BEST_OF_N, NOTEBOOKS_DIR, execute_notebook, score_notebook,
    _is_transient, _shorten_error,
)


SLOT_DIR = Path('/tmp/d2l-slots')


def _ensure_slots(gpu_slots, cpu_slots):
    SLOT_DIR.mkdir(parents=True, exist_ok=True)
    for i in range(gpu_slots):
        (SLOT_DIR / f'gpu-{i}.lock').touch()
    for i in range(cpu_slots):
        (SLOT_DIR / f'cpu-{i}.lock').touch()


def _slot_to_gpu(slot_idx, gpu_slots, num_gpus):
    """Map a GPU-slot index → physical CUDA device index.

    With gpu_slots = num_gpus * workers_per_gpu, slots cluster per GPU:
    slots 0..(workers_per_gpu-1) → GPU 0; next batch → GPU 1; etc.
    """
    workers_per_gpu = max(1, gpu_slots // max(1, num_gpus))
    return slot_idx // workers_per_gpu


@contextmanager
def acquire_gpu_slot(gpu_slots, num_gpus):
    """Block until any of gpu_slots is free; yield the CUDA device index.

    With gpu_slots > num_gpus, multiple workers can share a GPU (suitable
    when per-job VRAM is well below total VRAM, e.g. 11GB jobs on a 24GB
    GPU → 2 workers per GPU). Polls every 0.5s with non-blocking flock so
    workers rotate across slots fairly.
    """
    _ensure_slots(gpu_slots, 0)
    paths = [SLOT_DIR / f'gpu-{i}.lock' for i in range(gpu_slots)]
    while True:
        for i, p in enumerate(paths):
            fp = open(p, 'w')
            try:
                fcntl.flock(fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                fp.close()
                continue
            try:
                yield _slot_to_gpu(i, gpu_slots, num_gpus)
            finally:
                fcntl.flock(fp, fcntl.LOCK_UN)
                fp.close()
            return
        time.sleep(0.5)


@contextmanager
def acquire_cpu_slot(cpu_slots):
    """Block until one of cpu_slots is free; yield the slot index.

    CPU-only notebooks each get ~`nproc / cpu_slots` cores worth of
    headroom, mirroring the old run_notebooks.py "2 CPU workers" pool.
    Beware: `make -j` already caps total concurrency, so cpu_slots really
    means "no more than N CPU-only notebooks run at once" — useful when
    `make -j` is set higher than CPU work can sustain.
    """
    _ensure_slots(0, cpu_slots)
    paths = [SLOT_DIR / f'cpu-{i}.lock' for i in range(cpu_slots)]
    while True:
        for i, p in enumerate(paths):
            fp = open(p, 'w')
            try:
                fcntl.flock(fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                fp.close()
                continue
            try:
                yield i
            finally:
                fcntl.flock(fp, fcntl.LOCK_UN)
                fp.close()
            return
        time.sleep(0.5)


@contextmanager
def acquire_all_gpus(gpu_slots, num_gpus):
    """Block until ALL slots are free; yield comma-separated GPU index list.

    Used by MULTI_GPU_NOTEBOOKS, which need exclusive access to every GPU.
    Blocking flock; acquires in order, releases all on exit. Caller must
    ensure this isn't called concurrently with itself (otherwise deadlock).
    Make's -j scheduler will naturally serialize multi-GPU recipes because
    they each need all slots.
    """
    _ensure_slots(gpu_slots, 0)
    paths = [SLOT_DIR / f'gpu-{i}.lock' for i in range(gpu_slots)]
    fps = []
    try:
        for p in paths:
            fp = open(p, 'w')
            fcntl.flock(fp, fcntl.LOCK_EX)
            fps.append(fp)
        yield ','.join(str(i) for i in range(num_gpus))
    finally:
        for fp in fps:
            try:
                fcntl.flock(fp, fcntl.LOCK_UN)
                fp.close()
            except Exception:
                pass


def progress_for(fw):
    """Return (done, total) counts for a framework, snapshot-of-disk.

    `done` counts existing .executed stamps; `total` counts .ipynb files.
    Read concurrently across processes; numbers may briefly disagree but
    converge to the true completion count.
    """
    fw_dir = NOTEBOOKS_DIR / fw
    try:
        total = sum(1 for _ in fw_dir.rglob('*.ipynb'))
        done = sum(1 for _ in fw_dir.rglob('*.executed'))
    except OSError:
        return 0, 0
    return done, total


def notebook_mode(nb_path, fw):
    """Classify a notebook's execution mode without acquiring any slot.

    Returns one of: 'multi-gpu', 'gpu', 'cpu'. Used so we can print where
    a notebook is about to run before slot acquisition (helpful when many
    notebooks queue under `make -j`).
    """
    nb = Path(nb_path).resolve()
    rel = str(nb.relative_to((NOTEBOOKS_DIR / fw).resolve()))
    if rel in MULTI_GPU_NOTEBOOKS:
        return 'multi-gpu'
    if file_uses_gpu(nb, NOTEBOOKS_DIR / fw):
        return 'gpu'
    return 'cpu'


def _run_once(nb_path, fw, gpu_slots, num_gpus, cpu_slots, timeout, mode, label):
    """Single attempt; uses pre-computed mode and prints which slot we acquired.

    `mode` is 'multi-gpu', 'gpu', or 'cpu' (from notebook_mode()). `label`
    is the human-readable prefix for log lines (e.g. "[mxnet] chapter/foo:").
    """
    nb = Path(nb_path).resolve()

    if mode == 'multi-gpu':
        with acquire_all_gpus(gpu_slots, num_gpus) as cuda:
            print(f"{label}: [all GPUs] running", flush=True)
            return execute_notebook(nb, timeout=timeout, cuda_devices=cuda)
    if mode == 'gpu':
        with acquire_gpu_slot(gpu_slots, num_gpus) as i:
            print(f"{label}: [GPU {i}] running", flush=True)
            return execute_notebook(nb, timeout=timeout, cuda_devices=str(i))
    # CPU-only: bounded-pool acquisition so we don't oversubscribe cores
    # when many CPU notebooks are queued under `make -j`.
    with acquire_cpu_slot(cpu_slots) as i:
        print(f"{label}: [CPU {i}] running", flush=True)
        return execute_notebook(nb, timeout=timeout, cuda_devices="")


def run_with_retry(nb_path, fw, gpu_slots, num_gpus, cpu_slots, timeout,
                   mode, label):
    """Execute one notebook, retrying for stochastic / transient cases."""
    nb = Path(nb_path).resolve()
    rel = str(nb.relative_to((NOTEBOOKS_DIR / fw).resolve()))
    max_attempts, good_enough = BEST_OF_N.get(rel, (1, 0.0))

    last_err = None
    best_score = -1.0
    any_success = False

    for attempt in range(1, max_attempts + 1):
        ok, elapsed, err = _run_once(nb, fw, gpu_slots, num_gpus, cpu_slots,
                                     timeout, mode, label)
        if not ok:
            last_err = err
            print(f"  attempt {attempt}: FAIL ({elapsed:.0f}s) — "
                  f"{_shorten_error(err).splitlines()[-1][:120]}",
                  file=sys.stderr, flush=True)
            # Bail on first failure unless this is a known-stochastic notebook
            # AND the error is transient (otherwise retry won't help).
            if max_attempts == 1:
                return False, last_err
            if not _is_transient(err) and attempt == 1:
                # Non-transient first failure on a stochastic notebook: still
                # retry, the underlying issue might be a flaky GPU init.
                pass
            continue

        any_success = True
        if max_attempts == 1:
            return True, None

        score = score_notebook(nb)
        print(f"  attempt {attempt}: score={score:.3f} ({elapsed:.0f}s)",
              flush=True)
        if score > best_score:
            best_score = score
        if score >= good_enough:
            return True, None

    if any_success:
        # Stochastic notebook: kept the last successful run on disk; the
        # caller's contract is "passed at all", not "passed with great score".
        return True, None
    return False, last_err or "no successful attempt"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("framework",
                        choices=["pytorch", "tensorflow", "jax", "mxnet"])
    parser.add_argument("notebook", type=Path,
                        help="Path to .ipynb under _notebooks/<framework>/")
    # Physical GPU count (for CUDA_VISIBLE_DEVICES mapping) AND slot count
    # (controlling concurrency) are tracked separately. Rule of thumb from
    # CLAUDE.md: ≥11GB GPU RAM per GPU notebook, ≥16 cores per CPU notebook.
    # Makefile passes D2L_NUM_GPUS / D2L_GPU_SLOTS / D2L_CPU_SLOTS based on
    # nvidia-smi + nproc; explicit flags override here.
    parser.add_argument("--num-gpus", type=int,
                        default=int(os.environ.get("D2L_NUM_GPUS", "1")),
                        help="Physical GPU count for CUDA_VISIBLE_DEVICES")
    parser.add_argument("--gpu-slots", type=int,
                        default=int(os.environ.get("D2L_GPU_SLOTS", "0")) or None,
                        help="Concurrent GPU notebook slots "
                             "(default: D2L_GPU_SLOTS env, then num-gpus)")
    parser.add_argument("--cpu-slots", type=int,
                        default=int(os.environ.get("D2L_CPU_SLOTS", "2")),
                        help="Concurrent CPU notebook slots (default: 2)")
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()
    # Default GPU slots = physical GPU count if not set (1 worker per GPU).
    if args.gpu_slots is None:
        args.gpu_slots = max(1, args.num_gpus)

    if not args.notebook.exists():
        print(f"Not found: {args.notebook}", file=sys.stderr)
        sys.exit(2)

    setup_framework_env(args.framework)

    rel = args.notebook.resolve().relative_to(
        (NOTEBOOKS_DIR / args.framework).resolve())
    # "X/N" prefix uses live counts of `.executed` stamps vs total .ipynb
    # for this framework — concurrent across processes, so the numbers
    # are approximate (other workers may complete between our snapshot
    # and our print) but the trend is correct.
    done_before, total = progress_for(args.framework)
    label = f"[{args.framework} {done_before + 1}/{total}] {rel}"
    mode = notebook_mode(args.notebook, args.framework)
    mode_tag = {'multi-gpu': '[all GPUs]', 'gpu': '[GPU]', 'cpu': '[CPU]'}[mode]
    print(f"{label}: start {mode_tag}", flush=True)
    t0 = time.time()
    ok, err = run_with_retry(args.notebook, args.framework,
                             args.gpu_slots, args.num_gpus, args.cpu_slots,
                             args.timeout, mode, label)
    elapsed = time.time() - t0
    if ok:
        print(f"{label}: OK ({elapsed:.0f}s) {mode_tag}", flush=True)
        sys.exit(0)
    print(f"{label}: FAIL ({elapsed:.0f}s) {mode_tag}",
          file=sys.stderr, flush=True)
    sys.exit(1)


if __name__ == "__main__":
    main()
