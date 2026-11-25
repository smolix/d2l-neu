#!/usr/bin/env python3
"""Run one long regeneration job of chapter 21 without a notebook.

    python tools/lm_regen.py <name> [--resume] [--only RUN]
    python tools/lm_regen.py --prefetch

`<name>` is a key of `d2l.PRETRAIN_JOBS` (path (b) of the chapter's rule
R1: jobs over 30 minutes, or ones that hold whole cards for hours). The
script sets `D2L_RERUN=<name>`, changes into chapter_lm-pretraining (data
paths are `../data/lm/...` as in the notebooks), and calls the job's driver
with its keyword arguments, which are the page's call written a second
time; the result lands in `../data/lm/rerun/` under the next version of its
key, and the next normal run checks the page's configuration against it.
`--only RUN` limits a multi-run job to one run (a preset or a job folder
name such as `ladder-54m-s0`), so groups can run on disjoint GPUs; the
merge happens in the call that finds every run's files. `--merge` computes
nothing: it reads the parts' files (each written by a `--only` run or a
part job) and writes the merged result. `--resume` is the
default behaviour of the drivers (job folders resume from their last
state) and is accepted for the launcher's command line.

`--prefetch` downloads every registered chapter-21 artifact and the pinned
Hub weights and data of `d2l.PRETRAIN_SOURCES` into HF_HOME, quietly, so a
capture run never downloads.

Run it under the GPU-lock helper (gpu_run.sh); it exits non-zero on failure.
"""
import argparse
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHAPTER = os.path.join(REPO, 'chapter_lm-pretraining')


def prefetch(d2l):
    keys = sorted({k for k, _ in d2l.PRETRAIN_ARTIFACTS.values()
                   if k in d2l.DATA_HUB})
    for key in keys:
        t = time.time()
        with d2l._quiet():
            path = d2l.download_keyed(key)
        print(f'{key}: {os.path.getsize(path) / 1e6:.1f} MB '
              f'({time.time() - t:.0f} s)', flush=True)
    from huggingface_hub import snapshot_download
    for name, (kind, repo, rev, _) in sorted(d2l.PRETRAIN_SOURCES.items()):
        if kind != 'model':
            continue  # datasets are read per file by the builders
        t = time.time()
        with d2l._quiet():
            snapshot_download(repo, revision=rev, repo_type=kind)
        print(f'{name}: {repo}@{rev[:10]} ({time.time() - t:.0f} s)',
              flush=True)
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('name', nargs='?')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--only')
    p.add_argument('--merge', action='store_true')
    p.add_argument('--prefetch', action='store_true')
    a = p.parse_args()
    os.chdir(CHAPTER)
    if a.prefetch:
        from d2l import torch as d2l
        return prefetch(d2l)
    if not a.name:
        p.error('give a job name or --prefetch')
    os.environ['D2L_RERUN'] = a.name
    if a.only:
        os.environ['D2L_PRETRAIN_ONLY'] = a.only
    if a.merge:
        os.environ['D2L_PRETRAIN_MERGE'] = '1'
    from d2l import torch as d2l
    if a.name not in d2l.PRETRAIN_JOBS:
        raise SystemExit(f'{a.name} is not a job; one of '
                         f'{sorted(d2l.PRETRAIN_JOBS)}')
    driver, kwargs = d2l.PRETRAIN_JOBS[a.name]
    t = time.time()
    result = getattr(d2l, driver)(**kwargs())
    label = (result.attrs.get('label', '') if hasattr(result, 'attrs')
             else getattr(result, 'label', ''))
    print(f'{a.name}: {driver} done in {(time.time() - t) / 60:.1f} min'
          f'{"; " + label if label else ""}', flush=True)
    # pending: a run's files are missing (expected with --only)
    return 2 if label.endswith('pending') and not a.only else 0


if __name__ == '__main__':
    sys.exit(main())
