#!/usr/bin/env python3
"""One pretraining job of chapter 21 (sec_lm_reference_model), under torchrun.

    torchrun --standalone --nproc-per-node 4 tools/lm_pretrain.py \\
        --preset r2 --out data/lm/rerun/ch21/r2-s0 --resume auto
    torchrun ... tools/lm_pretrain.py --config <job>/config.json --resume auto
    torchrun ... tools/lm_pretrain.py --preset r2 --probe --out S/logs/probe
    python tools/lm_pretrain.py --package <job>     # job folder -> artifact files

The job runs `d2l.pretrain` on `d2l.PackedStream` over the preset's shards
(`'4b'` = the reference run's 3.2B-token epoch, `'100m'` = one shard), with
validation on rank 0, bf16 checkpoints, full states (resumable), and the
preset's branches. `--branch FROM:DECAY` replaces the branches. A finished
job writes `DONE` in its folder; a non-finite loss exits 3 (the launcher does
not restart it; rank 0 writes `NONFINITE`). `--package` writes `<job>/package/` (the log as
`log.jsonl.gz` with every branch's rows, the checkpoints, `config.json`) and
prints each file's sha1 to stderr.

Paths in the data folder resolve as in the notebooks, relative to the
chapter folder (`../data/lm/...`), so the job changes into
chapter_lm-pretraining first.
"""
import argparse
import dataclasses
import gzip
import hashlib
import json
import os
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHAPTER = os.path.join(REPO, 'chapter_lm-pretraining')


def parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--preset', help='an r2_config preset')
    p.add_argument('--config', help='a job config.json (written by launch_pretrain)')
    p.add_argument('--seed', type=int)
    p.add_argument('--probe', action='store_true', help='the D1 throughput probe')
    p.add_argument('--branch', action='append', default=None,
                   help='FROM:DECAY (repeatable); replaces the preset branches')
    p.add_argument('--resume', default='auto', help="'auto', 'none' or a state file")
    p.add_argument('--out', help='job folder (default from the config)')
    p.add_argument('--steps', type=int)
    p.add_argument('--lr', type=float)
    p.add_argument('--qk-clip', type=float, help='QK-clip threshold tau')
    p.add_argument('--micro-batch', type=int)
    p.add_argument('--package', metavar='JOB', help='package a finished job folder')
    p.add_argument('--allow-nonfinite', action='store_true',
                   help='exit 0 after a non-finite loss (instability arms)')
    return p.parse_args()


def sha1(path):
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def package(job):
    from d2l import torch as d2l
    out = os.path.join(job, 'package')
    os.makedirs(out, exist_ok=True)
    log = d2l._read_job_logs(job)
    path = os.path.join(out, 'log.jsonl.gz')
    with open(os.path.join(job, 'config.json')) as f:
        config = json.load(f)
    with gzip.open(path, 'wt', encoding='utf-8') as f:
        f.write(json.dumps({'_meta': dict(config=config, label='')}) + '\n')
        for rec in log.to_dict('records'):
            f.write(json.dumps(d2l._jsonable(rec)) + '\n')
    files = [path]
    shutil.copy(os.path.join(job, 'config.json'), out)
    files.append(os.path.join(out, 'config.json'))
    ckpt = os.path.join(job, 'ckpt')
    for name in sorted(os.listdir(ckpt)) if os.path.isdir(ckpt) else []:
        dst = os.path.join(out, name)
        if not os.path.exists(dst):
            os.link(os.path.join(ckpt, name), dst)
        files.append(dst)
    for f in files:
        print(f'{sha1(f)}  {os.path.getsize(f):>12}  {f}', file=sys.stderr)


def main():
    a = parse()
    if a.package:
        return package(os.path.abspath(a.package))
    out = os.path.abspath(a.out) if a.out else None
    os.chdir(CHAPTER)
    import torch
    from d2l import torch as d2l
    if a.config:
        with open(a.config) as f:
            cfg = d2l._cfg_from_json(json.load(f))
    elif a.preset:
        cfg = d2l.r2_config(a.preset)
    else:
        raise SystemExit('give --preset or --config')
    world = int(os.environ.get('WORLD_SIZE', 1))
    kw = dict(ddp=world > 1)
    for field, value in (('seed', a.seed), ('steps', a.steps),
                         ('peak_lr', a.lr), ('micro_batch', a.micro_batch),
                         ('out_dir', out)):
        if value is not None:
            kw[field] = value
    if a.branch is not None:
        kw['branches'] = tuple(tuple(int(x) for x in b.split(':'))
                               for b in a.branch)
        kw['state_steps'] = tuple(sorted(set(cfg.state_steps) | {
            f for f, _ in kw['branches']}))
    cfg = dataclasses.replace(cfg, **kw)
    if a.probe:
        rows = d2l.probe_throughput(cfg, out=cfg.out_dir)
        if int(os.environ.get('RANK', 0)) == 0:
            for r in rows:
                print(json.dumps(d2l._jsonable(r)))
        return 0
    if cfg.out_dir is None:
        raise SystemExit('give --out (or a config with out_dir)')
    os.makedirs(cfg.out_dir, exist_ok=True)
    rank, world, device = d2l._ddp_setup() if cfg.ddp else (
        0, 1, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    if rank == 0:
        with open(os.path.join(cfg.out_dir, 'config.json'), 'w') as f:
            json.dump(d2l._cfg_json(cfg), f, indent=1)
    torch.manual_seed(cfg.seed)
    model = d2l.PackedGPT.from_arch(cfg.arch).to(device)
    stream = d2l.PackedStream(d2l._train_shards(cfg), cfg.arch.T,
                              cfg.global_batch, seed=cfg.data_seed,
                              rank=rank, world=world)
    hooks = []
    if a.qk_clip:
        def clip(step, raw, opt, batch, row, tau=a.qk_clip):
            if cfg.stats_every and (step + 1) % cfg.stats_every == 0:
                d2l.qk_clip_(raw, d2l.attn_logit_stats(raw, batch), tau)
        hooks.append(clip)
    val = d2l._val_fn(cfg) if cfg.val_every else None
    log = d2l.pretrain(model, stream, cfg, val_fn=val, hooks=hooks,
                       resume=a.resume)
    finite = bool(len(log) == 0 or log['loss'].map(
        lambda x: x == x and abs(x) != float('inf')).all())
    if rank == 0:  # torchrun turns any rank's exit code into 1: a marker
        marker = 'DONE' if finite or a.allow_nonfinite else 'NONFINITE'
        open(os.path.join(cfg.out_dir, marker), 'w').close()
    if world > 1:
        import torch.distributed as dist
        dist.barrier()
        dist.destroy_process_group()
    return 0 if finite or a.allow_nonfinite else 3


if __name__ == '__main__':
    sys.exit(main())
