#!/usr/bin/env python3
"""Unified, resource-aware notebook execution scheduler.

This replaces the old "two background `make -jN` queues + per-notebook flock"
orchestration in `run-all-notebooks`. It cleanly separates the two concerns the
old design tangled together:

  * **Scheduling** — *which* work is allowed to run. The only scheduling
    constraint is that the four framework variants of a given notebook NEVER run
    concurrently (a per-relpath mutex, applied to ALL notebooks). That sequences
    a notebook's frameworks one at a time, which structurally prevents the
    shared-dataset reorg race (kaggle-cifar10/-dog copy into the same
    data/.../train_valid_test/ tree) without any special-casing. Different
    notebooks run concurrently.

  * **Resource allocation** — *how many* run at once, on which device. Three
    independent slot pools sized from detect_resources (same model the validated
    build used, so VRAM behaviour is unchanged):
        - per-GPU light slots:  GPU_SLOTS // NUM_GPUS on each physical GPU;
          a light GPU notebook takes `heavy_n` slots on ONE GPU.
        - CPU slots:            CPU_SLOTS, each pinned to a core group.
        - multi-GPU pairs:      NUM_GPU_PAIRS × MULTIGPU_PER_PAIR packing slots;
          a multi-GPU notebook takes one packing slot on a pair → 2 device idxs.
    Per-framework GPU caps (JAX/MXNet) bound a single framework's concurrent GPU
    notebooks. Multi-GPU items are held until light-GPU work drains so the two
    never double-book a physical GPU (CPU work overlaps throughout).

Each dispatch shells `make <stamp>` with D2L_ASSIGNED_CUDA / D2L_ASSIGNED_CPU_CORES
set, so it reuses the per-framework EXEC_RULE env (venv, LD_LIBRARY_PATH, thread
/ memory tuning) and stamp/log handling verbatim; run_one_notebook sees the
assignment and skips its own flock (see run_one_notebook._run_once assigned mode).

Env in (set by the Makefile recipe, falling back to detect_resources):
  D2L_NUM_GPUS D2L_GPU_SLOTS D2L_CPU_SLOTS D2L_NUM_GPU_PAIRS D2L_MULTIGPU_PER_PAIR
  D2L_MULTIGPU_SLOTS D2L_JAX_GPU_SLOTS D2L_MXNET_GPU_SLOTS D2L_JAX_MGPU_MEM_FRACTION

Usage:
  notebook_scheduler.py [--frameworks pytorch,jax,...] [--dry-run] [--jobs-cap N]
"""
import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from runtime_env import MULTI_GPU_NOTEBOOKS, HEAVY_GPU_NOTEBOOKS, file_uses_gpu

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = ROOT / "_notebooks"
FRAMEWORKS = ["pytorch", "tensorflow", "jax", "mxnet"]


def _int(env, default):
    try:
        return int(os.environ.get(env, "") or default)
    except ValueError:
        return default


def _detect(key):
    """Fallback to detect_resources --get KEY when an env knob is absent."""
    try:
        out = subprocess.run([sys.executable, str(ROOT / "tools/detect_resources.py"),
                              "--get", key], capture_output=True, text=True, timeout=30)
        return out.stdout.strip()
    except Exception:
        return ""


class Item:
    __slots__ = ("fw", "rel", "nb", "stamp", "mode", "heavy_n")

    def __init__(self, fw, rel, nb, stamp, mode, heavy_n):
        self.fw, self.rel, self.nb, self.stamp = fw, rel, nb, stamp
        self.mode, self.heavy_n = mode, heavy_n  # mode: 'cpu'|'gpu'|'multigpu'

    @property
    def label(self):
        return f"[{self.fw}] {self.rel}"


def _stale(nb: Path, stamp: Path) -> bool:
    """True if the notebook needs (re)running: stamp missing, or older than the
    notebook or any content dep listed in its `.d` file (matches make's deps)."""
    if not stamp.exists():
        return True
    st = stamp.stat().st_mtime
    if nb.stat().st_mtime > st:
        return True
    d = stamp.with_suffix(".d")
    if d.exists():
        try:
            text = d.read_text()
        except OSError:
            return True
        # `.d` is make syntax: "target: dep1 dep2 ...". Check every dep mtime.
        deps = text.replace("\\\n", " ").split(":", 1)
        if len(deps) == 2:
            for tok in deps[1].split():
                p = (ROOT / tok)
                try:
                    if p.stat().st_mtime > st:
                        return True
                except OSError:
                    continue
    return False


def build_worklist(frameworks, force_all=False):
    items = []
    for fw in frameworks:
        root = NOTEBOOKS_DIR / fw
        if not root.is_dir():
            continue
        for nb in sorted(root.rglob("*.ipynb")):
            if ".ipynb_checkpoints" in nb.parts:
                continue
            rel = str(nb.relative_to(root))
            stamp = nb.with_suffix(".executed")
            if not force_all and not _stale(nb, stamp):
                continue
            if rel in MULTI_GPU_NOTEBOOKS:
                mode = "multigpu"
            elif file_uses_gpu(nb, NOTEBOOKS_DIR):
                mode = "gpu"
            else:
                mode = "cpu"
            heavy_n = HEAVY_GPU_NOTEBOOKS.get((fw, rel), 1)
            items.append(Item(fw, rel, nb, stamp, mode, heavy_n))
    return items


class Scheduler:
    def __init__(self, args):
        self.args = args
        self.num_gpus = _int("D2L_NUM_GPUS", 0) or int(_detect("NUM_GPUS") or 0)
        self.gpu_slots = _int("D2L_GPU_SLOTS", 0) or int(_detect("GPU_SLOTS") or 0)
        self.cpu_slots = max(1, _int("D2L_CPU_SLOTS", 0) or int(_detect("CPU_SLOTS") or 2))
        self.pairs = _int("D2L_NUM_GPU_PAIRS", 0) or int(_detect("NUM_GPU_PAIRS") or 0)
        self.per_pair = _int("D2L_MULTIGPU_PER_PAIR", 0) or int(_detect("MULTIGPU_PER_PAIR") or 1)
        self.jax_mgpu_frac = os.environ.get("D2L_JAX_MGPU_MEM_FRACTION") or _detect("JAX_MGPU_MEM_FRACTION") or "0.30"
        self.fw_cap = {}
        jcap = _int("D2L_JAX_GPU_SLOTS", 0)
        mcap = _int("D2L_MXNET_GPU_SLOTS", 0) or int(_detect("MXNET_GPU_SLOTS") or 0)
        if jcap:
            self.fw_cap["jax"] = jcap
        if mcap:
            self.fw_cap["mxnet"] = mcap

        self.per_gpu = max(1, self.gpu_slots // self.num_gpus) if self.num_gpus else 0
        # Resource pools (mutated only under self.lock).
        self.gpu_free = [self.per_gpu] * self.num_gpus
        self.pair_free = [self.per_pair] * self.pairs
        cores = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(os.cpu_count() or 1))
        per = max(1, len(cores) // self.cpu_slots)
        self.core_groups = [cores[i * per:(i + 1) * per] or [cores[i % len(cores)]]
                            for i in range(self.cpu_slots)]
        self.cpu_free = list(range(self.cpu_slots))
        self.fw_gpu_running = {fw: 0 for fw in FRAMEWORKS}

        self.lock = threading.Condition()
        self.running_rel = set()          # relpaths with a framework in flight
        self.light_gpu_pending = 0        # incl. heavy single-GPU
        self.light_gpu_inflight = 0
        self.results = []                 # (item, rc, elapsed)
        self.inflight = 0

    # ---- resource reservation (called under self.lock) ----
    def _reserve(self, it):
        """Return an assignment dict and mutate pools, or None if not runnable
        now. Assignment keys: cuda (str), cpu_cores (list|None), and private
        bookkeeping under '_'."""
        if it.rel in self.running_rel:
            return None
        fw = it.fw
        if it.mode == "cpu":
            if not self.cpu_free:
                return None
            slot = self.cpu_free.pop()
            return {"cuda": "", "cpu_cores": self.core_groups[slot], "_": ("cpu", slot)}
        if it.mode == "gpu":
            cap = self.fw_cap.get(fw)
            if cap is not None and self.fw_gpu_running[fw] >= cap:
                return None
            # place on the GPU with the most free slots that fits heavy_n
            best = max(range(self.num_gpus), key=lambda g: self.gpu_free[g], default=-1)
            if best < 0 or self.gpu_free[best] < it.heavy_n:
                return None
            self.gpu_free[best] -= it.heavy_n
            self.fw_gpu_running[fw] += 1
            return {"cuda": str(best), "cpu_cores": None, "_": ("gpu", best, it.heavy_n)}
        # multigpu: held until light-GPU work fully drains (no double-booking)
        if self.light_gpu_pending or self.light_gpu_inflight:
            return None
        cap = self.fw_cap.get(fw)
        if cap is not None and self.fw_gpu_running[fw] >= cap:
            return None
        p = max(range(self.pairs), key=lambda i: self.pair_free[i], default=-1)
        if p < 0 or self.pair_free[p] <= 0:
            return None
        self.pair_free[p] -= 1
        self.fw_gpu_running[fw] += 1
        env = {}
        if fw == "jax":
            env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = self.jax_mgpu_frac
        return {"cuda": f"{2 * p},{2 * p + 1}", "cpu_cores": None,
                "extra_env": env, "_": ("mgpu", p)}

    def _release(self, it, asg):
        kind = asg["_"][0]
        if kind == "cpu":
            self.cpu_free.append(asg["_"][1])
        elif kind == "gpu":
            _, g, n = asg["_"]
            self.gpu_free[g] += n
            self.fw_gpu_running[it.fw] -= 1
        elif kind == "mgpu":
            self.pair_free[asg["_"][1]] += 1
            self.fw_gpu_running[it.fw] -= 1
        self.running_rel.discard(it.rel)

    # ---- execution ----
    def _make_env(self, asg):
        env = dict(os.environ)
        env["D2L_ASSIGNED_CUDA"] = asg["cuda"]
        if asg.get("cpu_cores"):
            env["D2L_ASSIGNED_CPU_CORES"] = ",".join(str(c) for c in asg["cpu_cores"])
        else:
            env.pop("D2L_ASSIGNED_CPU_CORES", None)
        # The scheduler owns concurrency, so the inner `make <stamp>` must NOT
        # try to join this process's parent jobserver (we run it as a plain
        # single-target build).
        for k in ("MAKEFLAGS", "MFLAGS", "MAKELEVEL"):
            env.pop(k, None)
        # Hand resource knobs to the inner make as env (origin=environment ⇒ its
        # `?=` is skipped), so it doesn't re-run detect_resources / nvidia-smi on
        # every dispatch. In assigned mode run_one_notebook ignores these (no
        # flock), but EXEC_RULE still evaluates them.
        for k, v in (("NUM_GPUS", self.num_gpus), ("GPU_SLOTS", self.gpu_slots),
                     ("CPU_SLOTS", self.cpu_slots), ("NUM_GPU_PAIRS", self.pairs),
                     ("MULTIGPU_PER_PAIR", self.per_pair),
                     ("MULTIGPU_SLOTS", self.pairs * self.per_pair),
                     ("JAX_MGPU_MEM_FRACTION", self.jax_mgpu_frac)):
            env.setdefault(k, str(v))
        if "mxnet" in self.fw_cap:
            env.setdefault("MXNET_GPU_SLOTS", str(self.fw_cap["mxnet"]))
        env.update(asg.get("extra_env", {}))
        return env

    def _worker(self, it, asg):
        t0 = time.time()
        rc = 0
        if self.args.dry_run:
            time.sleep(0.05)
        else:
            cmd = ["make", "--no-print-directory", str(it.stamp.relative_to(ROOT))]
            try:
                cp = subprocess.run(cmd, cwd=str(ROOT), env=self._make_env(asg))
                rc = cp.returncode
            except Exception as e:
                print(f"{it.label}: scheduler error: {e}", file=sys.stderr, flush=True)
                rc = 1
        elapsed = time.time() - t0
        with self.lock:
            self._release(it, asg)
            if it.mode in ("gpu",):
                self.light_gpu_inflight -= 1
            self.results.append((it, rc, elapsed))
            self.inflight -= 1
            done = len(self.results)
            self.lock.notify_all()
        tag = ("GPU " + asg["cuda"]) if asg["cuda"] else "CPU"
        print(f"{it.label}: {'OK' if rc == 0 else 'FAIL'} ({elapsed:.0f}s) "
              f"[{tag}] ({done}/{self.total} done)", flush=True)

    def _track(self):
        """Record peak resource usage (called under lock) for the dry-run check."""
        for g in range(self.num_gpus):
            self.peak["gpu_per"][g] = max(self.peak["gpu_per"][g], self.per_gpu - self.gpu_free[g])
        self.peak["cpu"] = max(self.peak["cpu"], self.cpu_slots - len(self.cpu_free))
        for p in range(self.pairs):
            self.peak["pair"][p] = max(self.peak["pair"][p], self.per_pair - self.pair_free[p])
        for fw in FRAMEWORKS:
            self.peak["fw"][fw] = max(self.peak["fw"][fw], self.fw_gpu_running[fw])
        if (self.light_gpu_pending or self.light_gpu_inflight):
            self.peak["mgpu_during_light"] = max(
                self.peak["mgpu_during_light"], sum(self.per_pair - f for f in self.pair_free))

    def run(self, items):
        self.total = len(items)
        pending = list(items)
        self.light_gpu_pending = sum(1 for it in items if it.mode == "gpu")
        n_cpu = sum(1 for it in items if it.mode == "cpu")
        n_mg = sum(1 for it in items if it.mode == "multigpu")
        print(f"=== scheduler: {self.total} notebooks "
              f"({self.light_gpu_pending} gpu, {n_cpu} cpu, {n_mg} multi-gpu) | "
              f"GPUs={self.num_gpus}×{self.per_gpu}slot, CPU={self.cpu_slots}, "
              f"pairs={self.pairs}×{self.per_pair}, caps={self.fw_cap} ===",
              flush=True)
        # invariant trackers (dry-run self-check): peak usage must stay within pools
        self.peak = {"gpu_per": [0] * self.num_gpus, "cpu": 0, "pair": [0] * self.pairs,
                     "fw": {fw: 0 for fw in FRAMEWORKS}, "mgpu_during_light": 0}
        with self.lock:
            while pending or self.inflight:
                progressed = False
                for it in list(pending):
                    asg = self._reserve(it)
                    if asg is None:
                        continue
                    pending.remove(it)
                    self.running_rel.add(it.rel)
                    if it.mode == "gpu":
                        self.light_gpu_pending -= 1
                        self.light_gpu_inflight += 1
                    self.inflight += 1
                    self._track()
                    threading.Thread(target=self._worker, args=(it, asg),
                                     daemon=True).start()
                    progressed = True
                if not progressed:
                    self.lock.wait()
        failed = [it for it, rc, _ in self.results if rc != 0]
        print(f"=== scheduler done: {self.total - len(failed)}/{self.total} ok, "
              f"{len(failed)} failed ===", flush=True)
        for it in failed:
            print(f"  FAILED: {it.label}", flush=True)
        if self.args.dry_run:
            p = self.peak
            print(f"peak: gpu_per={p['gpu_per']} (<= {self.per_gpu}), "
                  f"cpu={p['cpu']} (<= {self.cpu_slots}), "
                  f"pair={p['pair']} (<= {self.per_pair}), "
                  f"fw_gpu={p['fw']} (caps={self.fw_cap}), "
                  f"mgpu_during_light={p['mgpu_during_light']} (must be 0)",
                  flush=True)
            ok = (all(x <= self.per_gpu for x in p["gpu_per"]) and p["cpu"] <= self.cpu_slots
                  and all(x <= self.per_pair for x in p["pair"])
                  and all(p["fw"][f] <= self.fw_cap.get(f, 10 ** 9) for f in FRAMEWORKS)
                  and p["mgpu_during_light"] == 0)
            print(f"INVARIANTS: {'OK' if ok else 'VIOLATED'}", flush=True)
        return 1 if failed else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frameworks", default=",".join(FRAMEWORKS),
                    help="comma-separated subset to run")
    ap.add_argument("--dry-run", action="store_true",
                    help="plan + simulate dispatch (no notebook execution)")
    ap.add_argument("--force-all", action="store_true",
                    help="include every notebook regardless of stamp freshness")
    args = ap.parse_args()
    fws = [f for f in args.frameworks.split(",") if f.strip()]
    items = build_worklist(fws, force_all=args.force_all)
    if not items:
        print("scheduler: nothing to run (all stamps fresh)", flush=True)
        return 0
    return Scheduler(args).run(items)


if __name__ == "__main__":
    sys.exit(main())
