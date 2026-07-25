#!/usr/bin/env python3
"""Investigate JAX thread pools: what are the 220 threads and can we shrink them?"""
import os
import subprocess
import sys

# Test: does limiting visible CPUs via sched_setaffinity reduce XLA thread pools?
WORKLOAD_AFFINITY = r"""
import os, threading, time

def count_threads():
    with open(f'/proc/{os.getpid()}/status') as f:
        for line in f:
            if line.startswith('Threads:'):
                return int(line.split()[1])

def list_threads():
    pid = os.getpid()
    tid_dir = f'/proc/{pid}/task'
    threads = []
    for tid in sorted(os.listdir(tid_dir)):
        try:
            with open(f'{tid_dir}/{tid}/comm') as f:
                name = f.read().strip()
            threads.append((tid, name))
        except:
            pass
    return threads

AFFINITY = os.environ.get("_TEST_AFFINITY")
if AFFINITY:
    cpus = set(range(int(AFFINITY)))
    os.sched_setaffinity(0, cpus)
    print(f"Set CPU affinity to {len(cpus)} cores: {cpus}")

print(f"Visible CPUs: {len(os.sched_getaffinity(0))}")
print(f"PRE-IMPORT threads: {count_threads()}")

import jax
import jax.numpy as jnp

print(f"POST-IMPORT threads: {count_threads()}")

@jax.jit
def f(x):
    return jnp.dot(x, x.T) + jnp.sin(x)

x = jnp.ones((256, 256))
_ = f(x).block_until_ready()

n = count_threads()
print(f"POST-JIT threads: {n}")

for size in [512, 1024]:
    y = jnp.ones((size, size))
    _ = f(y).block_until_ready()

time.sleep(2)
n = count_threads()
print(f"SETTLED threads: {n}")

# Categorize threads by name prefix
threads = list_threads()
from collections import Counter
names = Counter(name for _, name in threads)
print(f"\nThread name histogram ({len(threads)} total):")
for name, cnt in names.most_common(20):
    print(f"  {cnt:4d}  {name}")
"""

def run(label, extra_env=None):
    print(f"\n{'='*60}")
    print(f"TEST: {label}")
    print(f"{'='*60}")

    env = os.environ.copy()
    for k in list(env.keys()):
        if k.startswith(("XLA_", "TF_", "OMP_", "MKL_", "OPENBLAS_")):
            del env[k]
    env.update({
        "OMP_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
        "TF_NUM_INTRAOP_THREADS": "4",
        "TF_NUM_INTEROP_THREADS": "2",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": ".70",
        "CUDA_VISIBLE_DEVICES": "0",
    })
    if extra_env:
        env.update(extra_env)

    result = subprocess.run(
        [sys.executable, "-c", WORKLOAD_AFFINITY],
        capture_output=True, text=True, timeout=120, env=env,
    )
    print(result.stdout)
    if result.returncode != 0:
        err = result.stderr.strip().splitlines()
        for line in err[-10:]:
            print(f"  {line}")
    return result.returncode == 0


# Test 1: baseline with thread names
run("Baseline (64 cores)")

# Test 2-5: restrict CPU affinity
for ncpus in [16, 8, 4, 2]:
    run(f"Affinity={ncpus} cores", {"_TEST_AFFINITY": str(ncpus)})


if __name__ == "__main__":
    pass
