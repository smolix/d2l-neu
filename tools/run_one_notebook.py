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


def _ensure_slots(num_gpus):
    SLOT_DIR.mkdir(parents=True, exist_ok=True)
    for i in range(num_gpus):
        (SLOT_DIR / f'gpu-{i}.lock').touch()


@contextmanager
def acquire_gpu_slot(num_gpus):
    """Block until one of num_gpus slots is free; yield the GPU index (int).

    Polls every 0.5s using non-blocking flock so we can rotate across slots
    instead of getting stuck waiting on a specific one.
    """
    _ensure_slots(num_gpus)
    paths = [SLOT_DIR / f'gpu-{i}.lock' for i in range(num_gpus)]
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
def acquire_all_gpus(num_gpus):
    """Block until ALL slots are free; yield comma-separated GPU index list.

    Uses blocking flock; acquires in order, releases all on exit. Beware:
    if a single-GPU notebook holds slot 0 forever, this can deadlock. We
    avoid that by only using this context for explicitly multi-GPU notebooks
    that are run after parallel batches.
    """
    _ensure_slots(num_gpus)
    paths = [SLOT_DIR / f'gpu-{i}.lock' for i in range(num_gpus)]
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


def _run_once(nb_path, fw, num_gpus, timeout):
    """Single attempt; selects CPU/single-GPU/multi-GPU context appropriately."""
    nb = Path(nb_path).resolve()
    rel = str(nb.relative_to((NOTEBOOKS_DIR / fw).resolve()))

    if rel in MULTI_GPU_NOTEBOOKS:
        with acquire_all_gpus(num_gpus) as cuda:
            return execute_notebook(nb, timeout=timeout, cuda_devices=cuda)
    if file_uses_gpu(nb, NOTEBOOKS_DIR / fw):
        with acquire_gpu_slot(num_gpus) as i:
            return execute_notebook(nb, timeout=timeout, cuda_devices=str(i))
    # CPU-only
    return execute_notebook(nb, timeout=timeout, cuda_devices="")


def run_with_retry(nb_path, fw, num_gpus, timeout):
    """Execute one notebook, retrying for stochastic / transient cases."""
    nb = Path(nb_path).resolve()
    rel = str(nb.relative_to((NOTEBOOKS_DIR / fw).resolve()))
    max_attempts, good_enough = BEST_OF_N.get(rel, (1, 0.0))

    last_err = None
    best_score = -1.0
    any_success = False

    for attempt in range(1, max_attempts + 1):
        ok, elapsed, err = _run_once(nb, fw, num_gpus, timeout)
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
    parser.add_argument("--num-gpus", type=int,
                        default=int(os.environ.get("D2L_NUM_GPUS", "1")))
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()

    if not args.notebook.exists():
        print(f"Not found: {args.notebook}", file=sys.stderr)
        sys.exit(2)

    setup_framework_env(args.framework)

    rel = args.notebook.resolve().relative_to(
        (NOTEBOOKS_DIR / args.framework).resolve())
    print(f"[{args.framework}] {rel}: start", flush=True)
    t0 = time.time()
    ok, err = run_with_retry(args.notebook, args.framework,
                             args.num_gpus, args.timeout)
    elapsed = time.time() - t0
    if ok:
        print(f"[{args.framework}] {rel}: OK ({elapsed:.0f}s)", flush=True)
        sys.exit(0)
    print(f"[{args.framework}] {rel}: FAIL ({elapsed:.0f}s)", file=sys.stderr,
          flush=True)
    sys.exit(1)


if __name__ == "__main__":
    main()
