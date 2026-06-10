#!/usr/bin/env python3
"""Detect host resources (GPU, CPU, RAM, ulimits) and derive build parallelism.

The Makefile consults this at the start of a build to size the global GPU/CPU
job pools and the per-framework concurrency caps to the *actual* hardware, so
the same Makefile runs unchanged on a 4×24 GB / 64-core server and on a 16 GB
laptop. Every notebook job is budgeted a footprint (VRAM / CPU cores / RAM);
the number of concurrent jobs is the largest count that fits ALL of the
detected constraints simultaneously (VRAM, system RAM, cores, and — for
DataLoader-heavy MXNet — the open-file / process ulimits).

Modes:
  --report        human-readable resource + parallelism summary (default)
  --make          emit `KEY=VALUE` lines for every derived knob (Make reads these)
  --get KEY       print a single derived value (e.g. GPU_SLOTS)

Footprints are overridable from the environment (e.g. GPU_MIB_PER_LIGHT=8192)
so the Makefile / operator can retune without editing this file. Results are
cached in /tmp for a few seconds so the Makefile can call --get many times
during one parse without re-shelling nvidia-smi each time.
"""
import json
import os
import re
import subprocess
import sys
import time

CACHE = "/tmp/.d2l_resources.json"
CACHE_TTL = 8.0  # seconds


def _macos_mem():
    """Return (total_mib, avail_mib) on macOS, or (0, 0) elsewhere.

    macOS has no ``/proc/meminfo``. Total RAM comes from ``sysctl
    hw.memsize``; the allocatable-without-swapping estimate is the sum of
    free + inactive + speculative + purgeable pages reported by ``vm_stat``
    (the closest analogue to Linux's ``MemAvailable``). Best-effort: any
    failure falls back to treating all RAM as available, or (0, 0) if even
    the total is unreadable.
    """
    if sys.platform != "darwin":
        return 0, 0
    try:
        total = int(subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=5).stdout.strip())
    except Exception:
        return 0, 0
    total_mib = total // (1024 * 1024)
    avail_mib = total_mib  # conservative fallback if vm_stat is unreadable
    try:
        out = subprocess.run(
            ["vm_stat"], capture_output=True, text=True, timeout=5).stdout
        page = 4096
        m = re.search(r"page size of (\d+) bytes", out)
        if m:
            page = int(m.group(1))
        counts = {}
        for line in out.splitlines():
            mm = re.match(r'"?([A-Za-z][\w \-]+?)"?:\s+(\d+)\.', line)
            if mm:
                counts[mm.group(1).strip()] = int(mm.group(2))
        free_pages = (counts.get("Pages free", 0)
                      + counts.get("Pages inactive", 0)
                      + counts.get("Pages speculative", 0)
                      + counts.get("Pages purgeable", 0))
        if free_pages:
            avail_mib = min(total_mib, free_pages * page // (1024 * 1024))
    except Exception:
        pass
    return total_mib, avail_mib

# ── Per-job footprints (overridable via env) ─────────────────────────────
# VRAM (MiB) a single notebook of each class needs at peak. "LIGHT" =
# pytorch/tensorflow/mxnet (their env tuning caps resident VRAM); JAX keeps
# the CPU backend + XLA pools alive so it is heavier.
def _envint(name, default):
    try:
        return max(1, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default

GPU_MIB_PER_LIGHT = _envint("GPU_MIB_PER_LIGHT", 7680)    # 7.5 GiB
GPU_MIB_PER_JAX   = _envint("GPU_MIB_PER_jax",   11776)   # 11.5 GiB
CPU_PER_LIGHT     = _envint("CPU_PER_LIGHT",     8)       # 8 cores / CPU-only job
CPU_PER_JAX       = _envint("CPU_PER_jax",       8)       # (>=8 cores -> 1 job/8 cores)
RAM_MIB_PER_LIGHT = _envint("RAM_MIB_PER_LIGHT", 4096)    # ~4 GiB / job
RAM_MIB_PER_JAX   = _envint("RAM_MIB_PER_jax",   8192)    # JAX CPU backend
RAM_HEADROOM_PCT  = _envint("RAM_HEADROOM_PCT",  85)      # leave 15% for OS/cache
# MXNet DataLoader workers are file-descriptor / process hungry; budget per job
# so the open-file (-n) and process (-u) ulimits also bound MXNet concurrency.
FD_PER_MXNET_JOB   = _envint("FD_PER_MXNET_JOB",   1024)
PROC_PER_MXNET_JOB = _envint("PROC_PER_MXNET_JOB", 128)


def _detect():
    # GPUs: count + minimum VRAM (workers_per_gpu must be uniform).
    num_gpus, min_gpu_mib = 0, 0
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15).stdout
        mibs = [int(x) for x in out.split() if x.strip().isdigit()]
        if mibs:
            num_gpus, min_gpu_mib = len(mibs), min(mibs)
    except Exception:
        mibs = []

    # CPU: respect cgroup / taskset affinity when available.
    try:
        ncpu = len(os.sched_getaffinity(0))
    except Exception:
        ncpu = os.cpu_count() or 16

    # RAM: MemAvailable is the kernel's estimate of allocatable-without-swap.
    # Linux exposes it via /proc/meminfo; macOS has no /proc, so fall back to
    # sysctl + vm_stat (_macos_mem). Either way a 0 here means "RAM unknown",
    # which derive() treats as "do not bound concurrency on RAM".
    mem_total_mib = mem_avail_mib = 0
    try:
        with open("/proc/meminfo") as f:
            info = {}
            for line in f:
                k, _, v = line.partition(":")
                info[k.strip()] = int(v.split()[0])  # KiB
        mem_total_mib = info.get("MemTotal", 0) // 1024
        mem_avail_mib = info.get("MemAvailable", info.get("MemFree", 0)) // 1024
    except Exception:
        mem_total_mib, mem_avail_mib = _macos_mem()

    # ulimits.
    try:
        import resource
        nofile = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
        nproc = resource.getrlimit(resource.RLIMIT_NPROC)[0]
        INF = resource.RLIM_INFINITY
        nofile = 1 << 20 if nofile == INF else nofile
        nproc = 1 << 20 if nproc == INF else nproc
    except Exception:
        nofile, nproc = 8192, 4096

    return dict(num_gpus=num_gpus, min_gpu_mib=min_gpu_mib, gpu_mibs=mibs,
                ncpu=ncpu,
                mem_total_mib=mem_total_mib, mem_avail_mib=mem_avail_mib,
                ulimit_nofile=int(nofile), ulimit_nproc=int(nproc))


def _floor_div(a, b):
    return a // b if b > 0 else 0


def derive(d):
    """Turn detected resources into a parallelism plan + provenance."""
    num_gpus = d["num_gpus"] or 1            # slot→device mapping needs ≥1
    min_mib = d["min_gpu_mib"]
    ncpu = d["ncpu"]
    ram_budget = d["mem_avail_mib"] * RAM_HEADROOM_PCT // 100

    def vram_slots(per):                      # total GPU jobs that fit VRAM
        return max(1, num_gpus * _floor_div(min_mib, per)) if min_mib else 1

    def ram_jobs(per):                        # total jobs that fit RAM budget
        return max(1, _floor_div(ram_budget, per)) if ram_budget else 1 << 20

    def core_slots(per):
        return max(1, _floor_div(ncpu, per))

    # Global pools, sized for the lightest footprint.
    raw_gpu = vram_slots(GPU_MIB_PER_LIGHT)
    raw_cpu = core_slots(CPU_PER_LIGHT)
    total_ram_jobs = ram_jobs(RAM_MIB_PER_LIGHT)
    # GPU jobs have priority for the RAM budget (the GPUs are the scarce
    # resource); trim the CPU pool first, then the GPU pool, if RAM is tight.
    gpu_slots = min(raw_gpu, total_ram_jobs)
    cpu_slots = max(1, min(raw_cpu, total_ram_jobs - gpu_slots)) \
        if total_ram_jobs > gpu_slots else 1
    gpu_slots = max(1, gpu_slots)

    # Per-framework GPU caps (acquire_fw_cap), each bounded by its own VRAM &
    # RAM footprint and never exceeding the global pool.
    jax_gpu = min(vram_slots(GPU_MIB_PER_JAX), ram_jobs(RAM_MIB_PER_JAX),
                  gpu_slots)
    jax_cpu = min(core_slots(CPU_PER_JAX), cpu_slots)
    # MXNet additionally bounded by the file-descriptor / process ulimits,
    # because its Gluon DataLoader spawns FD/process-hungry workers.
    mxnet_fd = _floor_div(d["ulimit_nofile"], FD_PER_MXNET_JOB)
    mxnet_proc = _floor_div(d["ulimit_nproc"], PROC_PER_MXNET_JOB)
    mxnet_gpu = max(1, min(gpu_slots, mxnet_fd or gpu_slots,
                           mxnet_proc or gpu_slots))

    # Multi-GPU notebooks use exactly 2 GPUs at <=GPU_MIB_PER_LIGHT each, so we
    # group GPUs into disjoint pairs and memory-pack per_pair notebooks onto
    # each pair (verified: 3 fit on a 24 GiB card across all 4 frameworks).
    pairs = num_gpus // 2
    per_pair = vram_slots(GPU_MIB_PER_LIGHT) // num_gpus if min_mib else 1
    per_pair = max(1, per_pair)
    multigpu_slots = max(1, pairs * per_pair)
    # jax preallocates a fixed VRAM fraction; cap it so per_pair jax processes
    # fit one card with ~10% headroom (per_pair=3 -> 0.30; =2 -> 0.45; =1 -> 0.90).
    jax_mgpu_frac = f"{0.9 / per_pair:.2f}"

    # ── Authoritative model for the unified scheduler ────────────────────────
    # 1 GPU slot per GPU_MIB_PER_LIGHT (7.5 GiB) of EACH GPU's VRAM, summed
    # across (possibly heterogeneous) GPUs; 1 CPU slot per CPU_PER_LIGHT cores
    # (min 1). The scheduler tracks slots PER physical GPU (not a flat pool).
    gpu_mibs = d.get("gpu_mibs") or []
    per_gpu_slots = [max(0, m // GPU_MIB_PER_LIGHT) for m in gpu_mibs]
    gpu_slots_total = sum(per_gpu_slots) or gpu_slots   # fall back to flat calc
    cpu_slots_cores = max(1, _floor_div(ncpu, CPU_PER_LIGHT))

    return dict(
        NUM_GPUS=num_gpus,
        GPU_SLOTS=gpu_slots_total,
        GPU_SLOTS_PER=",".join(str(s) for s in per_gpu_slots),
        GPU_VRAM_PER=",".join(str(m) for m in gpu_mibs),
        GPU_MIB_PER_SLOT=GPU_MIB_PER_LIGHT,
        CPU_SLOTS=cpu_slots_cores,
        JAX_GPU_SLOTS=max(1, jax_gpu),
        JAX_CPU_SLOTS=max(1, jax_cpu),
        MXNET_GPU_SLOTS=mxnet_gpu,
        MXNET_CPU_SLOTS=max(1, min(core_slots(CPU_PER_LIGHT), cpu_slots)),
        NUM_GPU_PAIRS=pairs,
        MULTIGPU_PER_PAIR=per_pair,
        MULTIGPU_SLOTS=multigpu_slots,
        JAX_MGPU_MEM_FRACTION=jax_mgpu_frac,
    ), dict(
        raw_gpu_vram=raw_gpu, raw_cpu_cores=raw_cpu,
        total_ram_jobs=total_ram_jobs, ram_budget_mib=ram_budget,
        mxnet_fd_cap=mxnet_fd, mxnet_proc_cap=mxnet_proc,
    )


def _load():
    try:
        if time.time() - os.path.getmtime(CACHE) < CACHE_TTL:
            with open(CACHE) as f:
                return json.load(f)
    except Exception:
        pass
    d = _detect()
    plan, prov = derive(d)
    blob = dict(detected=d, plan=plan, prov=prov)
    try:
        with open(CACHE, "w") as f:
            json.dump(blob, f)
    except Exception:
        pass
    return blob


def main(argv):
    mode = argv[1] if len(argv) > 1 else "--report"
    blob = _load()
    plan, d, prov = blob["plan"], blob["detected"], blob["prov"]

    if mode == "--get":
        key = argv[2]
        print(plan.get(key, ""))
        return 0
    if mode == "--make":
        for k, v in plan.items():
            print(f"{k}={v}")
        return 0

    # --report (human)
    gpu_line = (f"{d['num_gpus']} × {d['min_gpu_mib']} MiB"
                if d["num_gpus"] else "none (CPU-only)")
    print("══ Build resource detection ══")
    print(f"  GPU    : {gpu_line}")
    print(f"  CPU    : {d['ncpu']} cores")
    print(f"  RAM    : {d['mem_avail_mib']//1024} GiB available "
          f"/ {d['mem_total_mib']//1024} GiB total "
          f"(budget {prov['ram_budget_mib']//1024} GiB)")
    print(f"  ulimit : nofile={d['ulimit_nofile']} nproc={d['ulimit_nproc']}")
    print("══ Parallelism ══")
    print(f"  GPU_SLOTS       = {plan['GPU_SLOTS']}  "
          f"(per-GPU [{plan['GPU_SLOTS_PER']}] @ {plan['GPU_MIB_PER_SLOT']} MiB/slot)")
    print(f"  CPU_SLOTS       = {plan['CPU_SLOTS']}  "
          f"(1 per {CPU_PER_LIGHT} cores)")
    print(f"  JAX_GPU_SLOTS   = {plan['JAX_GPU_SLOTS']}   "
          f"JAX_CPU_SLOTS = {plan['JAX_CPU_SLOTS']}")
    print(f"  MXNET_GPU_SLOTS = {plan['MXNET_GPU_SLOTS']}  "
          f"(FD-fit {prov['mxnet_fd_cap']}, proc-fit {prov['mxnet_proc_cap']}; "
          f"heavy notebooks claim ≥2 slots)")
    print(f"  MULTIGPU_SLOTS  = {plan['MULTIGPU_SLOTS']}  "
          f"({plan['NUM_GPU_PAIRS']} GPU pairs × {plan['MULTIGPU_PER_PAIR']} "
          f"packed/pair; jax mem-fraction {plan['JAX_MGPU_MEM_FRACTION']})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
