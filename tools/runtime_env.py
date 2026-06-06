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

# (framework, relative-path) → number of global GPU slots to hold for
# this single notebook. Used for notebooks that overflow the standard
# per-process VRAM budget. Each entry causes run_one_notebook to flock
# N adjacent slots on a single GPU instead of 1, so the notebook gets
# N × GPU_MIB_PER_LIGHT MiB of headroom.
#
# tensorflow/chapter_computer-vision/ssd.ipynb: anchor-box generation
# stacks ~1000 tensors via tf.stack/Pack and OOMs with the default 8 GB
# budget; 2 slots (~16 GB) is enough.
HEAVY_GPU_NOTEBOOKS = {
    ("tensorflow", "chapter_computer-vision/ssd.ipynb"): 2,
    # bert-pretraining: hidden=128, seq=64, batch=512 lands a
    # BatchMatMulV2 of ~512×64×128×128 that OOMs the 8 GB budget.
    ("tensorflow", "chapter_natural-language-processing-pretraining/bert-pretraining.ipynb"): 2,
    # fine-tuning: ResNet18 features at 224×224, batch=128, XLA-fused
    # train step allocates ~9.9 GiB in one go and OOMs the 8 GB budget.
    ("tensorflow", "chapter_computer-vision/fine-tuning.ipynb"): 2,
}


def _text_has_gpu_keywords(text):
    """Return True if text contains any GPU-related keywords."""
    return any(kw in text for kw in GPU_KEYWORDS)


def file_uses_gpu(filepath, siblings_root):
    """Return True if this file or any framework sibling uses GPU keywords.

    ``filepath`` is e.g. ``_notebooks/jax/chapter_cv/foo.ipynb`` or
    ``_slides/pytorch/chapter_cv/foo.qmd``.  ``siblings_root`` is the
    parent that contains framework directories (``_notebooks`` or
    ``_slides``).  If the PyTorch or MXNet version of the same file
    contains GPU keywords, the JAX version is GPU too.
    """
    filepath = Path(filepath)
    siblings_root = Path(siblings_root)
    try:
        text = filepath.read_text(encoding="utf-8")
    except Exception:
        text = ""
    if _text_has_gpu_keywords(text):
        return True
    try:
        rel = filepath.relative_to(siblings_root)
    except ValueError:
        return False
    parts = rel.parts  # (framework, chapter, file, ...)
    if len(parts) >= 2:
        for sibling in siblings_root.iterdir():
            if sibling.is_dir() and sibling.name != parts[0]:
                candidate = sibling / Path(*parts[1:])
                if candidate.is_file():
                    try:
                        if _text_has_gpu_keywords(
                                candidate.read_text(encoding="utf-8")):
                            return True
                    except Exception:
                        pass
    return False


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
    # Signal headless capture so d2l.ProgressBoard skips its live (interactive)
    # animation frames and records exactly one final figure per cell. Interactive
    # sessions never call this, so they keep the live curve.
    os.environ.setdefault("D2L_NB_CAPTURE", "1")

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

# os.sched_getaffinity is Linux-only. Off Linux (macOS/Windows) CPU pinning is
# a no-op — make_cpu_affinity_fn is only ever called with a non-None cpu_set on
# the Linux execution box — so fall back to the full CPU set just to keep this
# module importable for the CPU-only render/scan path.
try:
    _HOST_CPUS = sorted(os.sched_getaffinity(0))
except AttributeError:
    _HOST_CPUS = list(range(os.cpu_count() or 1))


def make_cpu_affinity_fn(cpu_set):
    """Return a preexec_fn that pins the child process to *cpu_set*."""
    if cpu_set is None:
        return None
    frozen = frozenset(cpu_set)
    def _set():
        try:
            os.sched_setaffinity(0, frozen)
        except (OSError, AttributeError):  # AttributeError: not Linux
            pass
    return _set


# Linux-specific belt-and-suspenders against orphan subprocesses: ask the
# kernel to deliver a signal to a child when its parent dies. Combined with
# start_new_session=True (so we can kill the whole group from the parent),
# this guarantees nbconvert / ipykernel chains can't survive their parent
# even on SIGKILL. See tools/run_one_notebook.py for the parent-side
# install_signal_handlers().
_PR_SET_PDEATHSIG = 1
try:
    _LIBC = __import__('ctypes').CDLL("libc.so.6", use_errno=True)
except OSError:
    _LIBC = None


def _prctl_pdeathsig(sig):
    if _LIBC is None:
        return
    try:
        _LIBC.prctl(_PR_SET_PDEATHSIG, sig, 0, 0, 0)
    except OSError:
        pass


def make_subprocess_preexec_fn(cpu_set=None, pdeathsig=None):
    """Compose a subprocess preexec_fn: CPU affinity + PR_SET_PDEATHSIG.

    `pdeathsig` is a signal number (e.g. signal.SIGTERM) to deliver to the
    child when this Python process dies. None disables the prctl call.
    """
    cpu_fn = make_cpu_affinity_fn(cpu_set)
    if cpu_fn is None and pdeathsig is None:
        return None
    def _setup():
        if cpu_fn is not None:
            cpu_fn()
        if pdeathsig is not None:
            _prctl_pdeathsig(pdeathsig)
    return _setup


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
