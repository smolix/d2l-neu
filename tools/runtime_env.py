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
        "XLA_PYTHON_CLIENT_MEM_FRACTION": ".70",
    },
    "tensorflow": {
        "TF_FORCE_GPU_ALLOW_GROWTH": "true",
        "TF_XLA_FLAGS": "--tf_xla_auto_jit=2",
    },
}

# Notebooks that use multiple GPUs or test GPU availability across devices.
MULTI_GPU_NOTEBOOKS = {
    "chapter_builders-guide/use-gpu.ipynb",
    "chapter_computational-performance/multiple-gpus.ipynb",
    "chapter_computational-performance/multiple-gpus-concise.ipynb",
    "chapter_computational-performance/auto-parallelism.ipynb",
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
