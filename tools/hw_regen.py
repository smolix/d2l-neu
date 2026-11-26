#!/usr/bin/env python3
"""Run one regeneration of chapter 22 (Hardware for Training and Inference)
without a notebook.

    python tools/hw_regen.py <name> [--processes 3] [--replace] [--config C]
    python tools/hw_regen.py --prefetch

`<name>` is a key of `d2l.HW_ARTIFACTS` (box-microbench, box-workloads,
box-moe, box-storage, box-p2p, box-net, box-coll, box-power,
box-clocksweep; the last only through tools/hw_clock_sweep.sh, which
locks clocks with `sudo -n nvidia-smi`). The script
changes into chapter_lm-hardware (data paths are `../data/...` as in the
notebooks), sets `D2L_RERUN=<name>` and calls the name's driver from
`d2l.HW_JOBS` with its keyword arguments (the page's calls on the full
configuration) in `--processes` separate processes, one after the other
(the microbenchmark contract: the headline is the median of the
per-process medians). Each process writes its entries under
`../data/lm/rerun/ch22/<next key>/proc<i>/`; the script runs the driver's
`<driver>_prepare(folder, n)` if the library defines one (entries that must
agree between processes before they can be combined, such as box-coll's
dropped pairs), then combines them
(`d2l._hw_combine`) into `<next key>/<entry>.json.gz`, checks that no
measurement saw a foreign process on its cards, runs the driver's
`<driver>_finish(folder)` if the library defines one (files derived from
the combined entries, such as box-p2p's `alpha_beta.json`), and packs
`<next key>.tar` (or `.jsonl.gz`). It never executes a notebook and never
edits a tracked file. Then: `make upload-data FILES=... PREFIX=lm/`, one
`DATA_HUB` statement with the printed sha1 in the section's -lib.md file,
`make lib`, and a normal run.

Run it under the GPU-lock helper with the locks the name needs (one whole
card; all four for box-storage, box-p2p, box-coll and box-moe; two for
box-net). Exit status: 0 done, 2 usage or missing driver, 3 a measurement
was contended (re-run on a quiet box), 1 failure.

`--prefetch` downloads every registered chapter-22 artifact and the pinned
model configurations quietly, so a capture run never downloads. `--out DIR`
writes under DIR instead of ../data/lm/rerun/ch22 (the campaign launcher).
`--config C` runs the driver with config=C for a campaign elsewhere: the
output is the campaign log that `d2l.HW_CONFIG_LOGS[(name, C)]` names (for
example `box-microbench --config t2` writes `lm-hw-a100-microbench-v1`), its
records carry that configuration's tier, and the driver refuses a GPU that
is not the campaign's chip-table row. `--config box` is the default (no
configuration). `--config smoke` (box-clocksweep only) is a short test run:
it needs `--out`, and its folder is combined but never packed, so a smoke
never reaches an upload. A dry run (`D2L_HW_DRYRUN=1`) is never packed
either, and its combined entries record `dryrun` in their meta. SIGTERM, SIGINT and SIGHUP are passed on to the
running process (once), so its `finally` blocks run (box-clocksweep resets
the clock there) before this script exits.
"""
import argparse
import hashlib
import os
import shutil
import signal
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHAPTER = os.path.join(REPO, 'chapter_lm-hardware')
SMOKE_NAMES = ('box-clocksweep',)  # names whose driver has a 'smoke' config


def sha1(path):
    with open(path, 'rb') as f:
        return hashlib.sha1(f.read()).hexdigest()


def prefetch(d2l):
    for registry in (d2l.HW_ARTIFACTS, d2l.HW_CAMPAIGNS, d2l.HW_STATIC):
        for name, (key, kind, *_) in sorted(registry.items()):
            if key not in d2l.DATA_HUB:
                continue
            t = time.time()
            with d2l._quiet():
                path = d2l.download_keyed(key)
            if kind in ('tar', 'jsonl') and registry is not d2l.HW_STATIC:
                d2l._hw_unpack(path, kind)
            print(f'{name}: {os.path.getsize(path) / 1e6:.1f} MB '
                  f'({time.time() - t:.0f} s)', flush=True)
    models = sorted({m for m in d2l.HF_REVISIONS if m.startswith('Qwen/Qwen3-')
                     and m.endswith(('0.6B', '1.7B', '8B'))}
                    | set(d2l.HW_REVISIONS))
    for model in models:
        d2l.hw_config(model)
        print(f'{model}: configuration', flush=True)
    return 0


def child(d2l, name):
    driver, kwargs = d2l.HW_JOBS[name]
    kw = kwargs()
    if os.environ.get('D2L_HW_CONFIG'):
        kw['config'] = os.environ['D2L_HW_CONFIG']
    elif os.environ.get('D2L_HW_SMOKE') == '1':
        kw['config'] = 'smoke'
    getattr(d2l, driver)(**kw)
    return 0


def dryrun():
    """D2L_HW_DRYRUN=1: the privileged nvidia-smi calls were printed, not
    run (tools/hw_clock_sweep.sh), so nothing may be packed."""
    return os.environ.get('D2L_HW_DRYRUN') == '1'


def combine(d2l, folder, n):
    """Combine proc0..proc{n-1} into folder; return the contended count."""
    procs = [os.path.join(folder, f'proc{i}') for i in range(n)]
    entries = sorted(f for f in os.listdir(procs[0]) if f.endswith('.json.gz'))
    contended = 0
    for f in entries:
        parts = [d2l._hw_read(os.path.join(p, f)) for p in procs]
        configs = [m['config'] for _, m in parts]
        if any(c != configs[0] for c in configs):
            raise RuntimeError(f'{f}: the processes ran different configs')
        meta = dict(parts[0][1])
        env = dict(meta['env'])
        env['contended'] = any(m['env']['contended'] for _, m in parts)
        env['neighbours_busy'] = any(m['env']['neighbours_busy']
                                     for _, m in parts)
        env['seconds'] = round(sum(m['env']['seconds'] for _, m in parts), 1)
        env['n_proc'] = n
        contended += env['contended']
        meta['env'] = env
        if dryrun():
            meta['dryrun'] = True
        d2l._hw_write(os.path.join(folder, f),
                      d2l._hw_combine([r for r, _ in parts]), meta)
    return contended, entries


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('name', nargs='?')
    p.add_argument('--processes', type=int, default=3)
    p.add_argument('--replace', action='store_true',
                   help='delete an earlier unregistered regeneration folder')
    p.add_argument('--prefetch', action='store_true')
    p.add_argument('--out')
    p.add_argument('--config', help='a campaign configuration of the driver')
    p.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.config == 'box':
        a.config = None  # the default: the box run
    smoke = a.config == 'smoke'
    if smoke and not a.child:
        if a.name not in SMOKE_NAMES:
            print(f'{a.name} has no smoke configuration', file=sys.stderr)
            return 2
        if not a.out:
            print('--config smoke writes only under --out DIR', file=sys.stderr)
            return 2
    os.chdir(CHAPTER)
    if a.out:
        os.environ['D2L_HW_OUT'] = os.path.abspath(os.path.join(REPO, a.out))
    if smoke:
        os.environ.pop('D2L_HW_CONFIG', None)
        os.environ['D2L_HW_SMOKE'] = '1'
    elif a.config:
        os.environ['D2L_HW_CONFIG'] = a.config
    elif not a.child:
        os.environ.pop('D2L_HW_CONFIG', None)
    if not a.child and not smoke:
        os.environ.pop('D2L_HW_SMOKE', None)
    if a.prefetch:
        os.environ.pop('D2L_RERUN', None)
        from d2l import torch as d2l
        return prefetch(d2l)
    if not a.name:
        p.error('give a regeneration name or --prefetch')
    os.environ['D2L_RERUN'] = a.name
    from d2l import torch as d2l
    if a.name not in d2l.HW_JOBS:
        print(f'{a.name} is not a regeneration; one of {sorted(d2l.HW_JOBS)}',
              file=sys.stderr)
        return 2
    driver = d2l.HW_JOBS[a.name][0]
    if not hasattr(d2l, driver):
        print(f'{a.name}: driver d2l.{driver} is not defined yet',
              file=sys.stderr)
        return 2
    if (a.config and not smoke
            and (a.name, a.config) not in d2l.HW_CONFIG_LOGS):
        print(f'{a.name} has no campaign configuration {a.config!r}',
              file=sys.stderr)
        return 2
    if a.child:
        return child(d2l, a.name)
    key, kind = d2l._hw_regen_key(a.name)
    os.environ.pop('D2L_HW_PROC', None)
    folder = d2l._hw_regen_dir(a.name)
    out = folder + d2l._HW_EXT[kind]
    if os.path.exists(folder) or os.path.exists(out):
        if not a.replace:
            print(f'{folder} exists (an earlier regeneration); pass --replace '
                  'to delete it', file=sys.stderr)
            return 2
        shutil.rmtree(folder, ignore_errors=True)
        if os.path.exists(out):
            os.remove(out)
    t = time.time()
    running, signalled = [], []

    def forward(signum, frame):  # once: the process's own reset must finish
        if signalled:
            return
        signalled.append(signum)
        for proc in running:
            if proc.poll() is None:
                proc.send_signal(signum)
    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(s, forward)
    for i in range(a.processes):
        if signalled:
            break
        env = dict(os.environ, D2L_RERUN=a.name, D2L_HW_PROC=str(i))
        print(f'{a.name}: process {i + 1} of {a.processes}', flush=True)
        proc = subprocess.Popen([sys.executable, os.path.abspath(__file__),
                                 a.name, '--child'], env=env)
        running[:] = [proc]
        if signalled:  # it came between the check and the start
            proc.send_signal(signalled[0])
        returncode = proc.wait()
        if signalled:
            print(f'{a.name}: stopped by signal {signalled[0]} in process '
                  f'{i + 1} (exit {returncode})', file=sys.stderr)
            return 128 + signalled[0]
        if returncode != 0:
            print(f'{a.name}: process {i + 1} failed (exit {returncode})',
                  file=sys.stderr)
            return 1
    if signalled:
        print(f'{a.name}: stopped by signal {signalled[0]}', file=sys.stderr)
        return 128 + signalled[0]
    prepare = getattr(d2l, f'{driver}_prepare', None)
    if prepare is not None:  # make the processes' entries combinable
        prepare(folder, a.processes)
    contended, entries = combine(d2l, folder, a.processes)
    if contended:
        print(f'{a.name}: {contended} of {len(entries)} entries saw a foreign '
              'process on their cards; not packed: re-run on a quiet box',
              file=sys.stderr)
        return 3
    finish = getattr(d2l, f'{driver}_finish', None)
    if finish is not None:  # files written from the combined entries
        finish(folder)
    if smoke or dryrun():
        print(f'{a.name} ({"smoke" if smoke else "dry run"}): {len(entries)} '
              f'entries in {folder}; not packed', flush=True)
        return 0
    d2l._hw_pack(folder, kind, out)
    print(f'{a.name}: {len(entries)} entries in {(time.time() - t) / 60:.1f} '
          f'min', flush=True)
    print(f'wrote {out}\n  size {os.path.getsize(out) / 1e6:.2f} MB, '
          f'sha1 {sha1(out)}')
    print(f'register: d2l.DATA_HUB[{os.path.basename(folder)!r}] = (\n'
          f'    d2l.LM_DATA_URL + {os.path.basename(out)!r}, '
          f'{sha1(out)!r})')
    return 0


if __name__ == '__main__':
    sys.exit(main())
