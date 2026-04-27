#!/usr/bin/env python3
"""Measure thread count of a JAX process under different env var configs.

Usage:
    python tools/test_jax_threads.py [config_name]

Configs: baseline, xla_flag, omp_thread_pool, both
If no config given, runs all sequentially.
"""
import os
import subprocess
import sys
import time

CONFIGS = {
    "baseline": {
        # Current production settings from run_notebooks.py
        "OMP_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
        "TF_NUM_INTRAOP_THREADS": "4",
        "TF_NUM_INTEROP_THREADS": "2",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": ".70",
    },
    "xla_compilation_parallelism": {
        "OMP_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
        "TF_NUM_INTRAOP_THREADS": "4",
        "TF_NUM_INTEROP_THREADS": "2",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": ".70",
        "XLA_FLAGS": "--xla_gpu_force_compilation_parallelism=4",
    },
    "omp_thread_pool": {
        "OMP_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
        "TF_NUM_INTRAOP_THREADS": "4",
        "TF_NUM_INTEROP_THREADS": "2",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": ".70",
        # Limit the Eigen thread pool that XLA uses internally
        "TF_INTRA_OP_PARALLELISM_THREADS": "4",
        "TF_INTER_OP_PARALLELISM_THREADS": "2",
    },
    "xla_cpu_threads": {
        "OMP_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
        "TF_NUM_INTRAOP_THREADS": "4",
        "TF_NUM_INTEROP_THREADS": "2",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": ".70",
        "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false",
    },
    "combined": {
        "OMP_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
        "TF_NUM_INTRAOP_THREADS": "4",
        "TF_NUM_INTEROP_THREADS": "2",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": ".70",
        "XLA_FLAGS": "--xla_gpu_force_compilation_parallelism=4 --xla_cpu_multi_thread_eigen=false",
        "TF_INTRA_OP_PARALLELISM_THREADS": "4",
        "TF_INTER_OP_PARALLELISM_THREADS": "2",
    },
}

# The JAX workload: import, create array on GPU, run a JIT compile + execute
WORKLOAD = r"""
import os, threading, time

# Count threads at various stages
def count_threads():
    try:
        with open(f'/proc/{os.getpid()}/status') as f:
            for line in f:
                if line.startswith('Threads:'):
                    return int(line.split()[1])
    except:
        return threading.active_count()

print(f"PRE-IMPORT threads: {count_threads()}")

import jax
import jax.numpy as jnp

print(f"POST-IMPORT threads: {count_threads()}")
print(f"  devices: {jax.devices()}")

# Force a JIT compilation
@jax.jit
def f(x):
    return jnp.dot(x, x.T) + jnp.sin(x)

x = jnp.ones((256, 256))
_ = f(x).block_until_ready()

print(f"POST-JIT threads: {count_threads()}")

# Run a few more compilations to see if threads grow
for size in [512, 1024]:
    y = jnp.ones((size, size))
    _ = f(y).block_until_ready()

print(f"POST-MULTI-JIT threads: {count_threads()}")

# Small pause to let any background threads settle
time.sleep(2)
print(f"SETTLED threads: {count_threads()}")
"""


def run_config(name, env_vars):
    print(f"\n{'='*60}")
    print(f"CONFIG: {name}")
    for k, v in sorted(env_vars.items()):
        print(f"  {k}={v}")
    print(f"{'='*60}")

    env = os.environ.copy()
    # Clear any pre-existing XLA/thread vars
    for k in list(env.keys()):
        if k.startswith(("XLA_", "TF_", "OMP_", "MKL_", "OPENBLAS_")):
            del env[k]
    env.update(env_vars)
    env["CUDA_VISIBLE_DEVICES"] = "0"

    try:
        result = subprocess.run(
            [sys.executable, "-c", WORKLOAD],
            capture_output=True, text=True, timeout=120, env=env,
        )
        print(result.stdout)
        if result.returncode != 0:
            print(f"FAILED (exit {result.returncode})")
            # Print last 10 lines of stderr
            err_lines = result.stderr.strip().splitlines()
            for line in err_lines[-10:]:
                print(f"  {line}")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("TIMEOUT")
        return False


def main():
    if len(sys.argv) > 1:
        names = sys.argv[1:]
    else:
        names = list(CONFIGS.keys())

    for name in names:
        if name not in CONFIGS:
            print(f"Unknown config: {name}")
            print(f"Available: {', '.join(CONFIGS.keys())}")
            sys.exit(1)

    results = {}
    for name in names:
        ok = run_config(name, CONFIGS[name])
        results[name] = ok

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, ok in results.items():
        status = "OK" if ok else "FAIL"
        print(f"  {name}: {status}")


if __name__ == "__main__":
    main()
