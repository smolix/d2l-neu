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
import signal
import sys
import time
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from runtime_env import (
    HEAVY_GPU_NOTEBOOKS, MULTI_GPU_NOTEBOOKS, setup_framework_env, file_uses_gpu,
)
# Reuse functions defined in run_notebooks.py rather than duplicating them.
from run_notebooks import (
    BEST_OF_N, NOTEBOOKS_DIR, execute_notebook, score_notebook,
    _is_transient, _shorten_error, kill_active_subprocesses,
)


# Kill descendant nbconvert + ipykernel groups on Make/Ctrl-C kill, so the
# user doesn't end up with orphan processes pinning GPU memory after every
# interrupt. execute_notebook starts each child in its own session
# (start_new_session=True) and registers the pgid; we kill that group here.
_received_signal = None


def _signal_handler(signo, frame):
    global _received_signal
    _received_signal = signo
    kill_active_subprocesses(grace_sec=3)
    # Exit with the conventional 128 + signo. Use os._exit so atexit /
    # finalizers can't reanimate or block; we've already cleaned up our
    # subprocess group.
    os._exit(128 + signo)


def install_signal_handlers():
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        try:
            signal.signal(sig, _signal_handler)
        except (ValueError, OSError):
            pass


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


# MXNet 2.0's cuDNN loader races with itself when several mxnet processes
# init the GPU backend in the same wall-clock window — we get
# CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED. With 3 mxnet jobs/GPU, the issue
# pops up whenever multiple GPU slots free up together (e.g. a multi-GPU
# job exits and the queue dumps several mxnet notebooks at once).
#
# Fix: serialize ONLY the brief cuDNN sub-library load window. Hold an
# exclusive flock just long enough for `import mxnet` + first CUDA touch
# to clear (~5 s heuristic, well under typical notebook runtimes), then
# release so subsequent mxnet starts can stagger through. A background
# timer thread releases the lock; the notebook proceeds independently.
# If only ONE mxnet starts at a time, the lock is uncontended → no delay.
_MXNET_CUDNN_INIT_HOLD_SEC = 5.0


def serialize_mxnet_cudnn_init():
    """Acquire the mxnet cuDNN init lock, schedule auto-release.

    Returns a callable that releases early (safe to call multiple times).
    Released automatically after _MXNET_CUDNN_INIT_HOLD_SEC, or when the
    process exits (kernel cleans up flock).
    """
    import threading
    lock_path = SLOT_DIR / 'mxnet-cudnn-init.lock'
    SLOT_DIR.mkdir(parents=True, exist_ok=True)
    lock_path.touch()
    fp = open(lock_path, 'w')
    fcntl.flock(fp.fileno(), fcntl.LOCK_EX)

    released = threading.Event()

    def release():
        if released.is_set():
            return
        released.set()
        try:
            fcntl.flock(fp.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            fp.close()
        except OSError:
            pass

    timer = threading.Timer(_MXNET_CUDNN_INIT_HOLD_SEC, release)
    timer.daemon = True
    timer.start()
    return release


# Per-framework concurrency cap, layered ON TOP OF the global GPU/CPU slot
# pools. Today only JAX uses this: with the d2l.Module.plot path requiring
# the CPU backend, every JAX process keeps the 64-thread tf_XLAEigen pool
# alive, so we throttle JAX further than other frameworks to stay under
# the user thread limit. Set D2L_<FW>_GPU_SLOTS / D2L_<FW>_CPU_SLOTS=0 (or
# leave unset) to disable the cap for a framework.
@contextmanager
def acquire_fw_cap(framework, kind, count):
    """Block until one of `count` framework-`kind` slots is free.
    `kind` is 'gpu' or 'cpu'. No-op when count <= 0."""
    if count <= 0:
        yield
        return
    SLOT_DIR.mkdir(parents=True, exist_ok=True)
    paths = [SLOT_DIR / f'{framework}-{kind}-{i}.lock' for i in range(count)]
    for p in paths:
        p.touch()
    while True:
        for p in paths:
            fp = open(p, 'w')
            try:
                fcntl.flock(fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                fp.close()
                continue
            try:
                yield
            finally:
                fcntl.flock(fp, fcntl.LOCK_UN)
                fp.close()
            return
        time.sleep(0.5)


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
def acquire_n_gpu_slots(n, gpu_slots, num_gpus):
    """Acquire `n` adjacent slots on a single GPU; yield that GPU's
    physical device index.

    Used for memory-heavy single-GPU notebooks (HEAVY_GPU_NOTEBOOKS):
    locking n adjacent flocks in the same per-GPU range reduces the
    parallelism on that GPU from workers_per_gpu to (workers_per_gpu /
    n) for the duration of this notebook, giving it an n× share of
    GPU VRAM. Falls back to acquire_gpu_slot's behavior if n <= 1 or
    workers_per_gpu < n.
    """
    if n <= 1:
        with acquire_gpu_slot(gpu_slots, num_gpus) as i:
            yield i
        return
    _ensure_slots(gpu_slots, 0)
    workers_per_gpu = max(1, gpu_slots // max(1, num_gpus))
    if workers_per_gpu < n:
        # Not enough slots/GPU to satisfy the request — fall back to a
        # single slot rather than deadlocking against ourselves.
        with acquire_gpu_slot(gpu_slots, num_gpus) as i:
            yield i
        return
    # Try each GPU's slot range. Slots cluster per GPU: GPU 0 owns
    # 0..workers_per_gpu-1, GPU 1 owns workers_per_gpu..2*workers_per_gpu-1,
    # etc. Within one GPU's range we attempt to non-blocking-flock n
    # slots; if we can't get all n, release everything and try the
    # next GPU. Loops until we succeed (≈ acquire_gpu_slot's polling
    # loop).
    while True:
        for gpu_idx in range(max(1, num_gpus)):
            start = gpu_idx * workers_per_gpu
            paths = [SLOT_DIR / f'gpu-{start + j}.lock' for j in range(n)]
            held = []
            for p in paths:
                fp = open(p, 'w')
                try:
                    fcntl.flock(fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    fp.close()
                    break
                held.append(fp)
            if len(held) == n:
                try:
                    yield gpu_idx
                finally:
                    for fp in held:
                        try:
                            fcntl.flock(fp, fcntl.LOCK_UN)
                            fp.close()
                        except Exception:
                            pass
                return
            # Couldn't get all n on this GPU; release partial holds.
            for fp in held:
                try:
                    fcntl.flock(fp, fcntl.LOCK_UN)
                    fp.close()
                except Exception:
                    pass
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
    if file_uses_gpu(nb, NOTEBOOKS_DIR):
        return 'gpu'
    return 'cpu'


def _fw_cap(framework, kind):
    """Read D2L_<FW>_<KIND>_SLOTS env (e.g. D2L_JAX_GPU_SLOTS). Returns
    0 (= no cap) when unset or non-positive."""
    key = f'D2L_{framework.upper()}_{kind.upper()}_SLOTS'
    try:
        return max(0, int(os.environ.get(key, '0')))
    except ValueError:
        return 0


def _maybe_serialize_mxnet(fw):
    """Return an early-release callable for mxnet (to be invoked once the
    subprocess has had a chance to load cuDNN), or a no-op for everyone
    else."""
    if fw == 'mxnet':
        return serialize_mxnet_cudnn_init()
    return lambda: None


def _run_once(nb_path, fw, gpu_slots, num_gpus, cpu_slots, timeout, mode, label):
    """Single attempt; uses pre-computed mode and prints which slot we acquired.

    `mode` is 'multi-gpu', 'gpu', or 'cpu' (from notebook_mode()). `label`
    is the human-readable prefix for log lines (e.g. "[mxnet] chapter/foo:").

    For frameworks with a tighter concurrency cap (today: JAX), acquires
    the framework-specific slot FIRST, then the global pool slot, so a
    capped framework can't tie up a global slot waiting on its own cap.
    For mxnet, an additional cuDNN-init flock prevents the multi-process
    library-loader race; held for a few seconds after exec_notebook
    spawns the child, then released.
    """
    nb = Path(nb_path).resolve()

    if mode == 'multi-gpu':
        with acquire_fw_cap(fw, 'gpu', _fw_cap(fw, 'gpu')):
            with acquire_all_gpus(gpu_slots, num_gpus) as cuda:
                print(f"{label}: [all GPUs] running", flush=True)
                release_init = _maybe_serialize_mxnet(fw)
                try:
                    return execute_notebook(nb, timeout=timeout, cuda_devices=cuda)
                finally:
                    release_init()
    if mode == 'gpu':
        rel = str(nb.relative_to((NOTEBOOKS_DIR / fw).resolve()))
        heavy_n = HEAVY_GPU_NOTEBOOKS.get((fw, rel), 1)
        with acquire_fw_cap(fw, 'gpu', _fw_cap(fw, 'gpu')):
            with acquire_n_gpu_slots(heavy_n, gpu_slots, num_gpus) as i:
                tag = f"GPU {i}" if heavy_n == 1 else f"GPU {i} ×{heavy_n}"
                print(f"{label}: [{tag}] running", flush=True)
                release_init = _maybe_serialize_mxnet(fw)
                try:
                    return execute_notebook(nb, timeout=timeout, cuda_devices=str(i))
                finally:
                    release_init()
    # CPU-only: bounded-pool acquisition so we don't oversubscribe cores
    # when many CPU notebooks are queued under `make -j`.
    with acquire_fw_cap(fw, 'cpu', _fw_cap(fw, 'cpu')):
        with acquire_cpu_slot(cpu_slots) as i:
            print(f"{label}: [CPU {i}] running", flush=True)
            release_init = _maybe_serialize_mxnet(fw)
            try:
                return execute_notebook(nb, timeout=timeout, cuda_devices="")
            finally:
                release_init()


def run_with_retry(nb_path, fw, gpu_slots, num_gpus, cpu_slots, timeout,
                   mode, label):
    """Execute one notebook, retrying for stochastic / transient cases."""
    nb = Path(nb_path).resolve()
    rel = str(nb.relative_to((NOTEBOOKS_DIR / fw).resolve()))
    max_attempts, good_enough = BEST_OF_N.get(rel, (1, 0.0))

    last_err = None
    best_score = -1.0
    any_success = False

    # On failure, also dump the full stderr to a per-notebook log so the
    # cause of a DeadKernelError (which has no Python traceback) can be
    # diagnosed after the fact. Cleaned up on success.
    err_log = (NOTEBOOKS_DIR.parent / 'logs' / 'nb-errors' / fw / rel).with_suffix('.log')

    for attempt in range(1, max_attempts + 1):
        ok, elapsed, err = _run_once(nb, fw, gpu_slots, num_gpus, cpu_slots,
                                     timeout, mode, label)
        if not ok and err:
            err_log.parent.mkdir(parents=True, exist_ok=True)
            with err_log.open('a') as fh:
                fh.write(f'\n=== attempt {attempt} @ {time.strftime("%H:%M:%S")} '
                         f'({elapsed:.0f}s) ===\n')
                fh.write(err + '\n')
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
    # Install before any subprocess is spawned so a Ctrl-C / Make kill at
    # any point cleans up the nbconvert + ipykernel descendant tree.
    install_signal_handlers()

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
    label = f"[{args.framework}] {rel}"
    mode = notebook_mode(args.notebook, args.framework)
    mode_tag = {'multi-gpu': '[all GPUs]', 'gpu': '[GPU]', 'cpu': '[CPU]'}[mode]
    print(f"{label}: start {mode_tag}", flush=True)
    t0 = time.time()
    ok, err = run_with_retry(args.notebook, args.framework,
                             args.gpu_slots, args.num_gpus, args.cpu_slots,
                             args.timeout, mode, label)
    elapsed = time.time() - t0
    # Progress counter ONLY on the end line — it's a "stamps now on disk vs
    # total" snapshot. With parallel workers and brief gen_notebooks wipes
    # the count can fluctuate non-monotonically; printing it on the END
    # line (after our own stamp is written by the make recipe) makes it
    # stable enough to be useful as a "remaining work" hint.
    done_after, total = progress_for(args.framework)
    # +1 because our own .executed stamp lands AFTER this script exits.
    prog = f"({done_after + (1 if ok else 0)}/{total} {args.framework} done)"
    if ok:
        print(f"{label}: OK ({elapsed:.0f}s) {mode_tag} {prog}", flush=True)
        sys.exit(0)
    print(f"{label}: FAIL ({elapsed:.0f}s) {mode_tag} {prog}",
          file=sys.stderr, flush=True)
    sys.exit(1)


if __name__ == "__main__":
    main()
