#!/usr/bin/env python3
"""Unified notebook execution scheduler.

Model (see CLAUDE.md / docs/build-system.md §6.7):

* **Resources.** 1 GPU slot per GPU_MIB_PER_SLOT (7.5 GiB) of EACH GPU's VRAM,
  tracked PER physical GPU (heterogeneous GPUs just contribute different slot
  counts, e.g. 3×24 GiB + 1×32 GiB → 3+3+3+4 = 13). 1 CPU slot per 8 cores
  (min 1).

* **Per-notebook requirement** (tools/runtime_env.notebook_resource):
  default 1 CPU *or* 1 GPU slot; a notebook may instead need 2 slots on one GPU
  (memory-heavy), 1 slot on each of two GPUs ("2x1", data-parallel), or 2 slots
  on each of two GPUs ("2x2", a heavier framework's data-parallel variant).

* **Scheduling.** A single continuous dispatch over ONE work list ordered
  framework-by-framework (all of pytorch's notebooks, then jax's, …). Items are
  dispatched the moment their required slots are free — 1- and 2-GPU and CPU
  notebooks all mixed — with NO barrier between frameworks. The framework-grouped
  order separates a notebook's framework variants by ~one framework's worth of
  dispatches (~130), so with a ~12-20 slot pool the same notebook never runs in
  two frameworks at once: cross-framework contention (shared data/ reorg, lib
  rebuild) is avoided by ordering, not by a mutex or a barrier.

Each dispatch shells `make <stamp>` with the chosen device(s) in
D2L_ASSIGNED_CUDA (reusing the per-framework EXEC_RULE env / stamp / log) and
`--old-file` guards so an inner make never rebuilds the shared d2l library;
d2l bytecode is pre-compiled once so the first import burst doesn't race.

Env in (Makefile SCHED_ENV, else detect_resources):
  D2L_GPU_SLOTS_PER  e.g. "3,3,3,3"   (per-GPU slot capacity)
  D2L_GPU_VRAM_PER   e.g. "24564,..." (per-GPU VRAM MiB, for jax mem fraction)
  D2L_CPU_SLOTS      e.g. "8"
  D2L_GPU_MIB_PER_SLOT e.g. "7680"
  D2L_JAX_TOTAL_SLOTS e.g. "8"       (combined GPU + CPU JAX jobs)

Usage: notebook_scheduler.py [--frameworks a,b] [--dry-run] [--force-all]
"""
import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from runtime_env import notebook_resource, file_uses_gpu

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = ROOT / "_notebooks"
FRAMEWORKS = ["pytorch", "tensorflow", "jax", "mxnet"]


def _detect(key):
    try:
        return subprocess.run([sys.executable, str(ROOT / "tools/detect_resources.py"),
                               "--get", key], capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except Exception:
        return ""


def _intlist(s):
    return [int(x) for x in s.split(",") if x.strip()]


class Item:
    __slots__ = ("fw", "rel", "nb", "stamp", "req")

    def __init__(self, fw, rel, nb, stamp, req):
        self.fw, self.rel, self.nb, self.stamp, self.req = fw, rel, nb, stamp, req

    @property
    def label(self):
        return f"[{self.fw}] {self.rel}"

    @property
    def slots(self):  # total slots, for reporting / big-first tie-break
        if self.req[0] == "cpu":
            return 1
        return self.req[1] * self.req[2]


def _stale(nb: Path, stamp: Path) -> bool:
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
        parts = text.replace("\\\n", " ").split(":", 1)
        if len(parts) == 2:
            for tok in parts[1].split():
                try:
                    if (ROOT / tok).stat().st_mtime > st:
                        return True
                except OSError:
                    continue
    return False


def parse_files(spec):
    """Parse a whitespace/comma-separated list of source paths into a set of
    (chapter, stem) pairs. Accepts `chapter_x/foo.md`, `chapter_x/foo.ipynb`,
    or `chapter_x/foo`. Returns None ('all') for an empty/None spec."""
    if not spec:
        return None
    pairs = set()
    for tok in spec.replace(",", " ").split():
        p = Path(tok)
        chapter = p.parent.name or None
        pairs.add((chapter, p.stem))
    return pairs or None


def build_worklist(frameworks, force_all=False, files=None):
    """Items in framework-grouped order (framework outer, relpath inner).

    `files` (a set of (chapter, stem) from parse_files) restricts the worklist to
    a subset of source notebooks — used by `make refresh-stale` to re-execute
    only the audit-stale set in parallel instead of the whole book."""
    items = []
    for fw in frameworks:                       # framework is the OUTER ordering
        root = NOTEBOOKS_DIR / fw
        if not root.is_dir():
            continue
        for nb in sorted(root.rglob("*.ipynb")):  # relpath inner ordering
            if ".ipynb_checkpoints" in nb.parts:
                continue
            rel = str(nb.relative_to(root))
            if files is not None and (nb.parent.name, nb.stem) not in files:
                continue
            stamp = nb.with_suffix(".executed")
            if not force_all and not _stale(nb, stamp):
                continue
            req = notebook_resource(fw, rel, file_uses_gpu(nb, NOTEBOOKS_DIR))
            items.append(Item(fw, rel, nb, stamp, req))
    return items


class Scheduler:
    def __init__(self, args):
        self.args = args
        self.gpu_cap = _intlist(os.environ.get("D2L_GPU_SLOTS_PER")
                                or _detect("GPU_SLOTS_PER"))
        self.gpu_vram = _intlist(os.environ.get("D2L_GPU_VRAM_PER")
                                 or _detect("GPU_VRAM_PER"))
        try:
            self.cpu_slots = max(1, int(os.environ.get("D2L_CPU_SLOTS")
                                        or _detect("CPU_SLOTS") or 1))
        except ValueError:
            self.cpu_slots = 1
        try:
            self.mib_per_slot = int(os.environ.get("D2L_GPU_MIB_PER_SLOT")
                                    or _detect("GPU_MIB_PER_SLOT") or 7680)
        except ValueError:
            self.mib_per_slot = 7680
        self.num_gpus = len(self.gpu_cap)
        if len(self.gpu_vram) != self.num_gpus:        # vram unknown → assume uniform
            self.gpu_vram = [self.mib_per_slot * c for c in self.gpu_cap]

        # pools (mutated under lock)
        self.gpu_free = list(self.gpu_cap)
        cores = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") \
            else list(range(os.cpu_count() or 1))
        per = max(1, len(cores) // self.cpu_slots)
        self.core_groups = [cores[i * per:(i + 1) * per] or [cores[i % len(cores)]]
                            for i in range(self.cpu_slots)]
        self.cpu_free = list(range(self.cpu_slots))

        self.lock = threading.Condition()
        self.inflight = 0
        self.results = []
        self.peak_gpu = [0] * self.num_gpus
        self.peak_cpu = 0
        self.fw_inflight = {fw: 0 for fw in FRAMEWORKS}
        self.peak_fw = {fw: 0 for fw in FRAMEWORKS}
        self.fw_total_cap = {}
        for fw in FRAMEWORKS:
            key = f"{fw.upper()}_TOTAL_SLOTS"
            raw = os.environ.get(f"D2L_{key}") or _detect(key)
            try:
                if int(raw) > 0:
                    self.fw_total_cap[fw] = int(raw)
            except (TypeError, ValueError):
                pass

        # Pre-set the inner `make <stamp>`'s ?= resource vars in the child env so
        # it doesn't re-run detect_resources/nvidia-smi on every dispatch.
        # run_one_notebook ignores them in assigned mode, but EXEC_RULE still
        # evaluates them.
        self.inner_env = {
            "NUM_GPUS": str(self.num_gpus or 1),
            "GPU_SLOTS": str(sum(self.gpu_cap) or 1),
            "CPU_SLOTS": str(self.cpu_slots),
            "NUM_GPU_PAIRS": _detect("NUM_GPU_PAIRS") or "0",
            "MULTIGPU_PER_PAIR": _detect("MULTIGPU_PER_PAIR") or "1",
            "MULTIGPU_SLOTS": _detect("MULTIGPU_SLOTS") or "1",
            "JAX_MGPU_MEM_FRACTION": _detect("JAX_MGPU_MEM_FRACTION") or "0.30",
            "MXNET_GPU_SLOTS": _detect("MXNET_GPU_SLOTS") or "8",
            "JAX_GPU_SLOTS": _detect("JAX_GPU_SLOTS") or "8",
            "JAX_TOTAL_SLOTS": _detect("JAX_TOTAL_SLOTS") or "1",
        }

    # ---- reservation (under lock); returns assignment dict or None ----
    def _reserve(self, it):
        cap = self.fw_total_cap.get(it.fw)
        if cap is not None and self.fw_inflight[it.fw] >= cap:
            return None
        if it.req[0] == "cpu":
            if not self.cpu_free:
                return None
            slot = self.cpu_free.pop()
            self.fw_inflight[it.fw] += 1
            return {"cuda": "", "cpu_cores": self.core_groups[slot],
                    "_": ("cpu", slot), "_fw": it.fw}
        _, ngpu, spg = it.req
        # GPUs with at least spg free, most-free first (spread load)
        cand = sorted((g for g in range(self.num_gpus) if self.gpu_free[g] >= spg),
                      key=lambda g: self.gpu_free[g], reverse=True)
        if len(cand) < ngpu:
            return None
        chosen = sorted(cand[:ngpu])
        for g in chosen:
            self.gpu_free[g] -= spg
        env = {}
        if it.fw == "jax":   # jax preallocates: cap to the slots we reserved
            vram = min(self.gpu_vram[g] for g in chosen)
            frac = min(0.95, max(0.05, (spg * self.mib_per_slot) / vram))
            env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"{frac:.2f}"
        self.fw_inflight[it.fw] += 1
        return {"cuda": ",".join(str(g) for g in chosen), "cpu_cores": None,
                "extra_env": env, "_": ("gpu", chosen, spg), "_fw": it.fw}

    def _release(self, asg):
        self.fw_inflight[asg["_fw"]] -= 1
        kind = asg["_"][0]
        if kind == "cpu":
            self.cpu_free.append(asg["_"][1])
        else:
            _, chosen, spg = asg["_"]
            for g in chosen:
                self.gpu_free[g] += spg

    def _track(self):
        for g in range(self.num_gpus):
            self.peak_gpu[g] = max(self.peak_gpu[g], self.gpu_cap[g] - self.gpu_free[g])
        self.peak_cpu = max(self.peak_cpu, self.cpu_slots - len(self.cpu_free))
        for fw in FRAMEWORKS:
            self.peak_fw[fw] = max(self.peak_fw[fw], self.fw_inflight[fw])

    # ---- execution ----
    def _make_env(self, asg):
        env = dict(os.environ)
        env["D2L_ASSIGNED_CUDA"] = asg["cuda"]
        if asg.get("cpu_cores"):
            env["D2L_ASSIGNED_CPU_CORES"] = ",".join(str(c) for c in asg["cpu_cores"])
        else:
            env.pop("D2L_ASSIGNED_CPU_CORES", None)
        for k in ("MAKEFLAGS", "MFLAGS", "MAKELEVEL"):
            env.pop(k, None)
        for k, v in self.inner_env.items():
            env.setdefault(k, v)
        env.update(asg.get("extra_env", {}))
        return env

    def _worker(self, it, asg):
        t0 = time.time()
        rc = 0
        if self.args.dry_run:
            time.sleep(0.05)
        else:
            # -o guards: a per-notebook `make` must never rebuild the SHARED
            # d2l lib / preprocess / notebook set (built once upfront); else
            # concurrent inner makes corrupt d2l/*.py mid-import.
            cmd = ["make", "--no-print-directory",
                   "-o", ".preprocess.stamp", "-o", "d2l/.built",
                   "-o", f"d2l/{it.fw}.py",
                   "-o", f"_notebooks/{it.fw}/.generated",
                   str(it.stamp.relative_to(ROOT))]
            try:
                rc = subprocess.run(cmd, cwd=str(ROOT), env=self._make_env(asg)).returncode
            except Exception as e:
                print(f"{it.label}: scheduler error: {e}", file=sys.stderr, flush=True)
                rc = 1
        el = time.time() - t0
        with self.lock:
            self._release(asg)
            self.results.append((it, rc, el))
            self.inflight -= 1
            done = len(self.results)
            self.lock.notify_all()
        tag = ("GPU " + asg["cuda"]) if asg["cuda"] else "CPU"
        print(f"{it.label}: {'OK' if rc == 0 else 'FAIL'} ({el:.0f}s) [{tag}] "
              f"({done}/{self.total} done)", flush=True)

    def run(self, items):
        self.total = len(items)
        pending = list(items)                    # already framework-grouped
        n_cpu = sum(1 for it in items if it.req[0] == "cpu")
        print(f"=== scheduler: {self.total} notebooks ({self.total - n_cpu} gpu, "
              f"{n_cpu} cpu) | GPUs per-slot {self.gpu_cap}, CPU {self.cpu_slots} ===",
              flush=True)
        if not self.args.dry_run:
            self._warm_pyc({it.fw for it in items})
        with self.lock:
            while pending or self.inflight:
                progressed = False
                # First-fit in framework order: dispatch every item whose slots
                # are free right now; skip (don't block on) ones that don't fit.
                for it in list(pending):
                    asg = self._reserve(it)
                    if asg is None:
                        continue
                    pending.remove(it)
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
            ok = all(self.peak_gpu[g] <= self.gpu_cap[g] for g in range(self.num_gpus)) \
                 and self.peak_cpu <= self.cpu_slots \
                 and all(self.peak_fw[fw] <= cap
                         for fw, cap in self.fw_total_cap.items())
            print(f"peak: gpu={self.peak_gpu} (caps {self.gpu_cap}), "
                  f"cpu={self.peak_cpu} (<= {self.cpu_slots}), "
                  f"framework={self.peak_fw} (caps {self.fw_total_cap}) — "
                  f"INVARIANTS {'OK' if ok else 'VIOLATED'}", flush=True)
        return 1 if failed else 0

    def _warm_pyc(self, frameworks):
        py = next((ROOT / f".venv-{fw}/bin/python" for fw in frameworks
                   if (ROOT / f".venv-{fw}/bin/python").exists()), None)
        if py is None:
            return
        try:
            subprocess.run([str(py), "-m", "compileall", "-q", "-f", "d2l"],
                           cwd=str(ROOT), timeout=180,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("scheduler: pre-compiled d2l bytecode (warm import cache)", flush=True)
        except Exception as e:
            print(f"scheduler: warm-pyc skipped ({e})", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frameworks", default=",".join(FRAMEWORKS))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force-all", action="store_true")
    ap.add_argument("--files", default=None,
                    help="whitespace/comma-separated subset of source paths "
                         "(chapter_x/foo.md) to run; default: all stale notebooks")
    args = ap.parse_args()
    fws = [f for f in args.frameworks.split(",") if f.strip()]
    items = build_worklist(fws, force_all=args.force_all,
                           files=parse_files(args.files))
    if not items:
        print("scheduler: nothing to run (all stamps fresh)", flush=True)
        return 0
    return Scheduler(args).run(items)


if __name__ == "__main__":
    sys.exit(main())
