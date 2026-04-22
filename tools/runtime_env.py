"""Shared runtime environment setup for executing d2l notebooks and slides.

Provides framework-specific environment variables (thread limits, memory
caps, NVIDIA library paths) used by both run_notebooks.py and gen_slides.py.
"""

import glob as _glob
import os
import resource
import sys
from pathlib import Path


# Keywords that indicate a notebook/slide needs GPU access.
GPU_KEYWORDS = ("gpu(", "cuda", "GPU", "num_gpus", "try_gpu", "try_all_gpus",
                "device(", "/GPU:", "/device:GPU",
                "Trainer(", "d2l.train")

# Per-framework thread-limiting env vars.
THREAD_LIMIT_ENV = {
    "pytorch": {
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
    },
    "mxnet": {
        "OMP_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
        "MXNET_CPU_WORKER_NTHREADS": "4",
        "MXNET_GPU_WORKER_NTHREADS": "2",
    },
    "jax": {
        "OMP_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
        "TF_NUM_INTRAOP_THREADS": "4",
        "TF_NUM_INTEROP_THREADS": "2",
    },
    "tensorflow": {
        "OMP_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
        "TF_NUM_INTRAOP_THREADS": "4",
        "TF_NUM_INTEROP_THREADS": "2",
    },
}

# Per-framework runtime env vars (memory management, JIT flags, etc.).
FRAMEWORK_ENV = {
    "jax": {
        "XLA_PYTHON_CLIENT_MEM_FRACTION": ".40",
    },
    "tensorflow": {
        "TF_CPP_MIN_LOG_LEVEL": "2",
        "TF_XLA_FLAGS": "--tf_xla_auto_jit=2",
    },
    "mxnet": {
        "MXNET_CUDNN_LIB_CHECKING": "0",
    },
}

# Env vars to suppress CUDA noise when running CPU-only (CUDA_VISIBLE_DEVICES="").
# Only apply these when GPU is deliberately hidden — NOT for GPU workloads.
CPU_ONLY_ENV = {
    "TF_CPP_MIN_LOG_LEVEL": "3",
    "JAX_PLATFORMS": "cpu",
}

# Notebooks that use multiple GPUs or test GPU availability across devices.
MULTI_GPU_NOTEBOOKS = {
    "chapter_builders-guide/use-gpu.ipynb",
    "chapter_computational-performance/multiple-gpus.ipynb",
    "chapter_computational-performance/multiple-gpus-concise.ipynb",
    "chapter_computational-performance/auto-parallelism.ipynb",
    "chapter_computational-performance/async-computation.ipynb",
    "chapter_optimization/minibatch-sgd.ipynb",
}


def nvidia_lib_path(venv_root=None):
    """Build LD_LIBRARY_PATH entries from pip-installed nvidia-* packages.

    If *venv_root* is given, search that venv; otherwise use the venv
    of the current interpreter.
    """
    if venv_root is None:
        venv_root = os.path.dirname(os.path.dirname(os.path.abspath(sys.executable)))
    base = os.path.join(str(venv_root), "lib", "python*", "site-packages",
                        "nvidia", "*", "lib")
    dirs = sorted(os.path.abspath(d) for d in _glob.glob(base))
    return ":".join(dirs) if dirs else ""


def setup_framework_env(framework, venv_root=None):
    """Apply framework-specific env vars to os.environ (idempotent).

    Sets thread limits, memory fractions, LD_LIBRARY_PATH for NVIDIA libs,
    and raises the nproc soft limit for XLA thread pools.
    """
    for k, v in FRAMEWORK_ENV.get(framework, {}).items():
        os.environ.setdefault(k, v)
    for k, v in THREAD_LIMIT_ENV.get(framework, {}).items():
        os.environ.setdefault(k, v)

    nv_libs = nvidia_lib_path(venv_root)
    if nv_libs:
        os.environ["LD_LIBRARY_PATH"] = (
            nv_libs + ":" + os.environ.get("LD_LIBRARY_PATH", ""))

    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
        if hard > soft:
            resource.setrlimit(resource.RLIMIT_NPROC, (hard, hard))
    except (ValueError, OSError):
        pass


# ── CPU affinity helpers ──────────────────────────────────────────────
# XLA sizes its Eigen and internal thread pools from the CPU affinity mask,
# so restricting visible cores is the only reliable way to cap per-process
# thread count.

MAX_CPUS_PER_GPU_WORKER = 32
MAX_CPUS_PER_CPU_WORKER = 16

_HOST_CPUS = sorted(os.sched_getaffinity(0))


def make_cpu_affinity_fn(cpu_set):
    """Return a preexec_fn that pins the child process to *cpu_set*."""
    if cpu_set is None:
        return None
    frozen = frozenset(cpu_set)
    def _set():
        try:
            os.sched_setaffinity(0, frozen)
        except OSError:
            pass
    return _set


def worker_cpu_set(worker_id, num_workers, max_cpus):
    """Return a set of CPU indices for *worker_id*.

    Distributes *max_cpus* cores per worker, strided across host CPUs so
    that adjacent workers overlap minimally.
    """
    n = len(_HOST_CPUS)
    if n <= max_cpus:
        return set(_HOST_CPUS)
    stride = n // num_workers
    start = worker_id * stride
    return {_HOST_CPUS[(start + i) % n] for i in range(min(max_cpus, n))}


# ── Stale kernel cleanup ────────────────────────────────────────────

def kill_stale_kernels(venv_dir):
    """Kill orphaned ipykernel processes from a previous run of *venv_dir*.

    After notebook/slide execution, jupyter kernels occasionally survive
    their parent nbconvert/quarto process and hold GPU memory.  This finds
    any ``ipykernel_launcher`` processes whose executable lives under
    *venv_dir* and sends them SIGKILL.  Returns the number of processes
    killed.
    """
    import signal
    venv_dir = str(Path(venv_dir).resolve())
    killed = 0
    try:
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                exe = (entry / "exe").resolve()
            except (OSError, PermissionError):
                continue
            if not str(exe).startswith(venv_dir):
                continue
            try:
                cmdline = (entry / "cmdline").read_bytes().split(b"\x00")
            except (OSError, PermissionError):
                continue
            if any(b"ipykernel_launcher" in arg for arg in cmdline):
                pid = int(entry.name)
                try:
                    os.kill(pid, signal.SIGKILL)
                    killed += 1
                except OSError:
                    pass
    except (OSError, PermissionError):
        pass
    return killed
