#!/usr/bin/env python3
"""Run the proxy campaign of the data-mixtures section (sec_lm_data_mixtures).

One launcher for both tiers: the T1 regeneration of the notebook (mix-t1,
one RTX 4090) and the T2 campaign package (a faster node). From an entry of
d2l.MIX_CONFIGS it builds the run list (d2l.mix_run_specs) and runs it:

  1. the learning-rate runs (uniform mixture, seed 0); the best final mean
     bits per byte wins, ties (within lr_tie) to the smaller rate;
  2. the pure-domain, training, held-out and uniform runs; the DoReMi proxy
     once uniform-wsd-s0 has finished (its final checkpoint is the
     reference model);
  3. the arms: regression-selected (fit_mixture_regression(select) on the
     finished training runs, choose_mixture for the target: chosen by code,
     never by hand), DoReMi (its averaged weights) and staged (uniform, then
     the selected mixture from stage_frac on).

One worker process per entry of --gpus (an id, or 'cpu') pulls runs from a
file-locked queue in OUT; every run writes OUT/runs/<run_id>.json by an
atomic rename and checkpoints every ckpt_every steps to OUT/ckpt/<id>.pt;
--resume skips finished runs and resumes checkpoints. The launcher stops its
workers at --max-hours (default for t2: the campaign file's stop_hours, else
1.5 x the smoke's predicted GPU-hours on these cards / the number of
workers; 1 h for the other configurations); a later --resume continues.
At the end it merges everything into OUT/lm-mix-runs-<config>-v1.json.

--smoke runs only the smoke of --config; --smoke-t2 runs the smoke of the t2
configuration first (on the pools of --config) and then the campaign. The
smoke trains 100 steps (20 for the test configuration), kills a copy after
its step-50 checkpoint and resumes it, with deterministic kernels, in bf16
(within 0.01 nats of an uninterrupted run) and in fp32 (within 1e-6),
records how far two uninterrupted runs on the campaign's kernels drift
apart (bf16 kernels need not repeat bit for bit), measures tokens/s with and without torch.compile,
MFU against the device's dense bf16 peak and peak memory, and predicts the
hours of the t2 campaign.

Usage (from the repository, .venv-lm):
  python tools/run_mix_campaign.py --config t1 --gpus 0 --out DIR --smoke-t2
  python tools/run_mix_campaign.py --config t2 --gpus 0,1,2,3 --out DIR
The test configuration 'tiny' (CPU, seconds per run) exists for
tools/test_lm_mixtures.py and smoke runs.
"""
import argparse
import fcntl
import json
import math
import os
import pathlib
import platform
import signal
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Dense bf16 tensor-core peaks with fp32 accumulation (FLOP/s), from the
# vendors' datasheets: NVIDIA A100 Tensor Core GPU datasheet (BF16 312
# TFLOPS dense); NVIDIA Ada Lovelace architecture whitepaper, GeForce RTX
# 4090 (FP16/BF16 tensor with FP32 accumulate 165.2 TFLOPS dense); NVIDIA
# H100 Tensor Core GPU datasheet, SXM (BF16 989 TFLOPS dense).
PEAK_BF16 = {'A100': 312e12, 'RTX 4090': 165.2e12, 'H100': 989e12}
A100_PEAK = PEAK_BF16['A100']


def d2l_module():
    sys.path.insert(0, str(ROOT))
    from d2l import torch as d2l
    return d2l


def configs(d2l):
    """d2l.MIX_CONFIGS plus the test configuration 'tiny' (CPU, fp32): the
    t1 design and file names at a 2-block, d = 32 proxy, 2^15 tokens and
    the first 64 windows of each validation set."""
    c = dict(d2l.MIX_CONFIGS)
    c['tiny'] = dict(
        c['t1'], name='tiny', runs_key='lm-mix-runs-tiny-v1',
        proxy=d2l.ProxyCfg(32, 2, 2, seq_len=64, batch=8, micro_batch=8,
                           dtype='float32'),
        tokens=2**15, lr_tokens=2**14, curve_windows=16, ckpt_every=32,
        val_windows=64)
    return c


def atomic_json(path, obj):
    tmp = f'{path}.tmp{os.getpid()}'
    with open(tmp, 'w') as f:
        json.dump(obj, f, sort_keys=True)
    os.replace(tmp, path)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def byte_lengths(d2l, tokenizer):
    """Bytes per token id of the tokenizer and the SHA-256 of its file."""
    import hashlib
    if tokenizer is None:
        tokenizer = d2l.download_keyed('lm-r2-tokenizer-v1')
        tok = d2l.r2_tokenizer(raw=True)
    else:
        tok = d2l._r2_from_file(tokenizer, raw=True)
    sha = hashlib.sha256(pathlib.Path(tokenizer).read_bytes()).hexdigest()
    return d2l.token_byte_lengths(tok), sha


def load_pools(d2l, c, args):
    pools, vals = d2l.mix_pools(c, args.pools)
    missing = [d for d in c['domains'] if d not in pools or d not in vals]
    if missing:
        sys.exit(f'pools missing for {missing} under {args.pools}/{c["name"]}'
                 f': python tools/build_mix_pools.py --config {c["name"]}')
    if c.get('val_windows'):  # the test configuration only
        n = c['val_windows'] * c['proxy'].seq_len + 1
        vals = {d: d2l.TokenPool(v.ids[:n], v.meta) for d, v in vals.items()}
    byte_len, sha = byte_lengths(d2l, args.tokenizer)
    for P in list(pools.values()) + list(vals.values()):
        if P.meta['tokenizer_sha256'] != sha:
            sys.exit(f'pool {P.meta["domain"]}/{P.meta["split"]} was built '
                     'with another tokenizer')
    return pools, vals, byte_len


def peak_of(name):
    return next((v for k, v in PEAK_BF16.items() if k in name), None)


# ----------------------------------------------------------------- queue


class Queue:
    """Runs of the plan in OUT/runs, claimed under a file lock."""
    def __init__(self, out, specs, only=None):
        self.out, self.specs, self.only = out, specs, only
        (out / 'runs').mkdir(parents=True, exist_ok=True)
        (out / 'ckpt').mkdir(exist_ok=True)

    def done(self):
        return {p.stem for p in (self.out / 'runs').glob('*.json')}

    def claim(self):
        """A runnable spec (now claimed), 'wait', or None when finished."""
        with open(self.out / 'queue.lock', 'w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            done = self.done()
            claimed = {p.stem for p in (self.out / 'runs').glob('*.claim')}
            todo = [s for s in self.specs if s['run_id'] not in done and (
                self.only is None or s['run_id'] in self.only)]
            for s in todo:
                if s['run_id'] in claimed:
                    continue
                if all(d in done for d in s['depends']):
                    (self.out / 'runs' / f'{s["run_id"]}.claim').write_text(
                        str(os.getpid()))
                    return s
            return 'wait' if todo and claimed else None

    def finish(self, spec, record):
        rid = spec['run_id']
        atomic_json(self.out / 'runs' / f'{rid}.json', record)
        (self.out / 'runs' / f'{rid}.claim').unlink(missing_ok=True)

    def cached(self, name, compute):
        """A shared derived result (learning rate, selection), computed once
        under the lock."""
        path = self.out / f'{name}.json'
        with open(self.out / 'derived.lock', 'w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not path.exists():
                atomic_json(path, compute())
            return read_json(path)


def lr_choice(d2l, c, q):
    def compute():
        runs = [read_json(q.out / 'runs' / f'{s["run_id"]}.json')
                for s in q.specs if s['arm'] == 'lr']
        mean = {f'{lr:g}': sum(r['final'][d]['bpb'] for d in c['domains'])
                / len(c['domains']) for lr, r in zip(c['lr_grid'], runs)}
        best = min(mean.values())
        chosen = min(lr for lr in c['lr_grid']
                     if mean[f'{lr:g}'] <= best + c['lr_tie'])
        return dict(grid=list(c['lr_grid']), mean_bpb=mean, chosen=chosen,
                    tokens=c['lr_tokens'], rule='best final mean bits per '
                    f'byte; ties within {c["lr_tie"]} to the smaller rate')
    return q.cached('lr_choice', compute)


def selection(d2l, c, q):
    import numpy as np

    def compute():
        ids = [s['run_id'] for s in q.specs if s['arm'] == 'train']
        runs = [read_json(q.out / 'runs' / f'{i}.json') for i in ids]
        W = np.array([r['weights'] for r in runs])
        L = np.array([[r['final'][d]['bpb'] for d in c['domains']]
                      for r in runs])
        f = d2l.fit_mixture_regression(W, L, c['select'])
        w, pred = d2l.choose_mixture(f, c['targets'][c['target']],
                                     max_weight=c['max_weight'], within=W)
        params = None if f.params is None else np.asarray(f.params).tolist()
        return dict(kind=c['select'], target=c['target'], weights=w.tolist(),
                    predicted=pred, train_runs=ids, params=params)
    return q.cached('selection', compute)


# ---------------------------------------------------------------- worker


def run_one(d2l, c, q, spec, pools, vals, byte_len):
    import torch
    t0 = time.time()
    D = c['domains']
    lr = (c['lr_grid'][[s['run_id'] for s in q.specs
                        if s['arm'] == 'lr'].index(spec['run_id'])]
          if spec['arm'] == 'lr' else lr_choice(d2l, c, q)['chosen'])
    cfg = c['proxy']._replace(lr=lr, schedule=spec['schedule'])
    weights, stages = spec['weights'], spec['stages']
    if spec['arm'] in ('regsel', 'staged'):
        sel = selection(d2l, c, q)['weights']
        if spec['arm'] == 'regsel':
            weights = sel
        else:
            stages = [[c['stage_frac'], sel]]
    if spec['arm'] == 'doremi':
        weights = read_json(q.out / 'runs' / 'doremi-proxy.json')[
            'doremi']['weights']
    spec = dict(spec, weights=list(weights), stages=stages)
    as_dict = lambda w: dict(zip(D, (float(x) for x in w)))
    device = d2l.try_gpu()
    if spec['arm'] == 'doremi-weights':
        result = run_doremi(d2l, c, q, cfg, pools, vals, byte_len, device)
    else:
        ckpt = q.out / 'ckpt' / f'{spec["run_id"]}.pt'
        result = d2l.train_proxy_lm(
            pools, as_dict(weights), spec['tokens'], cfg, seed=spec['seed'],
            eval_sets=vals, stages=[(f, as_dict(w)) for f, w in stages or []],
            log_every=c['log_every'], eval_at=spec['eval_at'],
            curve_windows=c['curve_windows'], byte_lengths=byte_len,
            ckpt=str(ckpt), ckpt_every=c['ckpt_every'],
            resume=str(ckpt) if ckpt.exists() else None,
            compile=c.get('compile', False), device=device)
    record = d2l.campaign_record(spec, result, cfg)
    record.update(wall=time.time() - t0, device=(
        torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'))
    q.finish(spec, record)
    if spec['run_id'] != 'uniform-wsd-s0':  # the DoReMi reference stays
        (q.out / 'ckpt' / f'{spec["run_id"]}.pt').unlink(missing_ok=True)
    print(f'{spec["run_id"]}: {record["wall"]:.0f} s, final '
          + ', '.join(f'{d} {record["final"][d]["bpb"]:.4f}' for d in D),
          flush=True)


def run_doremi(d2l, c, q, cfg, pools, vals, byte_len, device):
    """DoReMi's weights: a fresh proxy against the final checkpoint of
    uniform-wsd-s0 (the uniform-mixture proxy) as the reference model."""
    import torch
    D, L = c['domains'], cfg.seq_len
    ref = d2l._proxy_model(cfg, 0)
    ref.load_state_dict(torch.load(q.out / 'ckpt' / 'uniform-wsd-s0.pt',
                                   map_location='cpu',
                                   weights_only=False)['model'])
    ref = ref.to(device)
    proxy = d2l._proxy_model(cfg, 0).to(device)
    b = cfg.batch // len(D)  # sequences per domain and step
    steps = c['tokens'] // (cfg.batch * L)
    streams = {d: d2l._ProxyStream({d: pools[d]}, {d: 1.0}, steps * b * L, L,
                                   b, seed=0) for d in D}
    batches = lambda k: {d: torch.from_numpy(s.batch(k)[0]).to(device)
                         for d, s in streams.items()}
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    w, trace = d2l.doremi_weights(
        ref, proxy, batches, steps, cfg, **c['doremi'],
        lr=lambda k: cfg.lr * d2l._lr_at(k, steps, cfg))
    seconds = time.perf_counter() - t0
    final = {d: d2l._proxy_eval(proxy, vals[d].ids, cfg, byte_len)
             for d in D}
    return dict(doremi=dict(weights=w.tolist(), trace=trace,
                            ref_run='uniform-wsd-s0', sequences_per_domain=b,
                            **c['doremi']),
                final=final, steps=steps, seconds=seconds,
                n_params=d2l.num_params(proxy), log=[], evals=[],
                peak_mem=(torch.cuda.max_memory_allocated(device) / 2**30
                          if device.type == 'cuda' else None))


def worker(args):
    d2l = d2l_module()
    c = configs(d2l)[args.config]
    plan = read_json(args.out / 'plan.json')
    q = Queue(args.out, plan['specs'], plan['only'])
    pools, vals, byte_len = load_pools(d2l, c, args)
    while True:
        spec = q.claim()
        if spec is None:
            return
        if spec == 'wait':
            time.sleep(2)
            continue
        run_one(d2l, c, q, spec, pools, vals, byte_len)


# ----------------------------------------------------------------- smoke


class deterministic:
    """Deterministic kernels (the math attention kernel, deterministic
    algorithms; cuBLAS needs CUBLAS_WORKSPACE_CONFIG, which spawn sets), so
    a resumed run can be compared with an uninterrupted one exactly."""
    def __enter__(self):
        import torch
        from torch.nn.attention import SDPBackend, sdpa_kernel
        torch.use_deterministic_algorithms(True, warn_only=True)
        self.kernel = sdpa_kernel(SDPBackend.MATH)
        self.kernel.__enter__()

    def __exit__(self, *exc):
        import torch
        self.kernel.__exit__(*exc)
        torch.use_deterministic_algorithms(False)


def smoke_child(args):
    """Train the smoke run to its half-way checkpoint, then go on training
    (writing nothing more) until the parent kills it: the checkpoint the
    parent resumes is always the step-half one, however fast the steps."""
    d2l = d2l_module()
    c, cfg, pools, steps = smoke_setup(d2l, args)
    per_step = cfg.batch * cfg.seq_len
    run = lambda **kw: d2l.train_proxy_lm(
        pools, uniform(c), c['tokens'], cfg, seed=0, log_every=1,
        max_epochs=10**6, **kw)
    with deterministic():
        run(stop_tokens=steps // 2 * per_step, ckpt=args.smoke_child,
            ckpt_every=steps // 2)
        run(stop_tokens=steps * per_step, resume=args.smoke_child)


def uniform(c):
    return {d: 1 / len(c['domains']) for d in c['domains']}


def smoke_setup(d2l, args):
    allc = configs(d2l)
    c = allc[args.smoke_config]
    cfg = c['proxy']._replace(lr=1e-3, schedule='cosine')
    pools, _, _ = load_pools(d2l, allc[args.config], args)
    return c, cfg, pools, 20 if args.smoke_config == 'tiny' else 100


def smoke(args):
    """The smoke record of args.smoke_config (see the module docstring)."""
    import torch
    d2l = d2l_module()
    c, cfg, pools, steps = smoke_setup(d2l, args)
    per_step, half = cfg.batch * cfg.seq_len, steps // 2
    device = d2l.try_gpu()
    common = dict(seed=0, log_every=1, max_epochs=10**6,
                  stop_tokens=steps * per_step, device=device)
    losses = lambda r: {s: l for s, _, l, _ in r['log']}
    rate = lambda r: ((r['log'][-1][1] - r['log'][half - 1][1])
                      / (r['log'][-1][3] - r['log'][half - 1][3]))
    train = lambda cfg, **kw: (d2l.lm_free(), d2l.train_proxy_lm(
        pools, uniform(c), c['tokens'], cfg, **{**common, **kw}))[1]
    a = train(cfg)  # uninterrupted, the kernels of the campaign
    a2 = train(cfg)  # again: these bf16 kernels need not repeat bit for bit
    with deterministic():
        d = train(cfg)
    ckpt = args.out / 'ckpt' / 'smoke.pt'
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.unlink(missing_ok=True)
    d2l.lm_free()  # the child needs the card
    child = subprocess.Popen(
        [sys.executable, __file__, '--config', args.config, '--out',
         str(args.out), '--pools', str(args.pools), '--smoke-config',
         args.smoke_config, '--smoke-child', str(ckpt)]
        + (['--tokenizer', args.tokenizer] if args.tokenizer else []))
    while not ckpt.exists() and child.poll() is None:
        time.sleep(0.2)
    child.kill()
    child.wait()
    assert ckpt.exists(), 'the smoke child wrote no checkpoint'
    assert torch.load(ckpt, map_location='cpu', weights_only=False)[
        'step'] == half, 'the resume would not start half way'
    with deterministic():
        b = train(cfg, resume=str(ckpt))
    ld, lb = losses(d), losses(b)
    diff16 = max(abs(ld[s] - lb[s]) for s in lb)
    la, la2 = losses(a), losses(a2)
    noise16 = max(abs(la[s] - la2[s]) for s in la2)
    cfg32 = cfg._replace(dtype='float32')
    with deterministic():
        f_full = train(cfg32)
        train(cfg32, stop_tokens=half * per_step, ckpt=str(ckpt))
        f_res = train(cfg32, resume=str(ckpt))
    lf, lr_ = losses(f_full), losses(f_res)
    diff32 = max(abs(lf[s] - lr_[s]) for s in lr_)
    ckpt.unlink(missing_ok=True)
    compiled = train(cfg, compile=True) if device.type == 'cuda' else None
    name = (torch.cuda.get_device_name(0) if device.type == 'cuda'
            else platform.processor() or 'cpu')
    n = a['n_params']
    per_token = d2l.proxy_flops(cfg, 1, n)
    peak = peak_of(name)
    tps = rate(a)
    tps_c = rate(compiled) if compiled else None
    # compilation time: the compiled run's first half beyond its steady rate
    compile_s = (None if compiled is None else max(
        0.0, compiled['log'][half - 1][3] - half * per_step / tps_c))
    t2 = configs(d2l)['t2']
    use_c = bool(t2.get('compile')) and compiled is not None
    run_tps = tps_c if use_c else tps  # the kernels the campaign runs
    with torch.device('meta'):
        n2 = d2l.num_params(d2l._proxy_model(t2['proxy']))
    flop, n_compiled = 0.0, 0
    for s in d2l.mix_run_specs(t2):  # DoReMi: two more forward passes
        f = d2l.proxy_flops(t2['proxy'], s['tokens'], n2)
        flop += f * (10 / 6 if s['arm'] == 'doremi-weights' else 1)
        n_compiled += use_c and s['arm'] != 'doremi-weights'
    compile_h = n_compiled * (compile_s or 0.0) / 3600  # once per run
    mfu = lambda r: None if peak is None or r is None else r * per_token / peak
    pred_a100 = (None if peak is None else
                 flop / (A100_PEAK * mfu(run_tps)) / 3600 + compile_h)
    record = dict(
        config=args.smoke_config, device=name, steps=steps, checkpoint=half,
        tokens_per_step=per_step, n_params=n, flop_per_token=per_token,
        tokens_per_s=tps, tokens_per_s_compiled=tps_c, mfu=mfu(tps),
        mfu_compiled=mfu(tps_c), device_peak_bf16=peak,
        peak_mem_gib=a['peak_mem'], resume_bf16_max_diff=diff16,
        rerun_bf16_max_diff=noise16, resume_bf16_ok=diff16 <= 0.01,
        resume_fp32_max_diff=diff32, campaign_runs=len(d2l.mix_run_specs(t2)),
        campaign_flop=flop, campaign_compile=use_c, compile_seconds=compile_s,
        compile_hours=compile_h, mfu_campaign=mfu(run_tps),
        predicted_a100_hours=pred_a100,
        predicted_hours_here=flop / (run_tps * per_token) / 3600 + compile_h,
        # trained: a twice, d, the child (about half), b, three fp32
        # parts, compiled
        smoke_flop=per_token * per_step * (5 * steps + 4 * half),
        mfu_assumption='MFU measured here, assumed equal on an A100; '
        'hours are GPU-hours of one card (compile time once per run)',
        pools=f'the {args.config} pools (the smoke stream may repeat them)',
        losses=[la[s] for s in sorted(la)])
    atomic_json(args.out / f'smoke-{args.smoke_config}.json', record)
    print(json.dumps({k: v for k, v in record.items() if k != 'losses'},
                     indent=1), flush=True)


# ------------------------------------------------------------------ main


def environment(d2l):
    import torch
    git = lambda *a: subprocess.run(['git', '-C', str(ROOT), *a],
                                    capture_output=True, text=True).stdout
    return dict(torch=torch.__version__, cuda=torch.version.cuda,
                python=platform.python_version(), host=platform.node(),
                git_commit=git('rev-parse', 'HEAD').strip(),
                git_dirty=bool(git('status', '--porcelain', '--',
                                   'd2l', 'tools').strip()))


def merge(d2l, c, args, specs, t_start):
    D = c['domains']
    pools, vals = d2l.mix_pools(c, args.pools)
    keep = ('n_tokens', 'n_docs', 'n_bytes_nfc', 'n_bytes_scored', 'sha256',
            'tokenizer_sha256', 'source', 'revision', 'files')
    runs = [read_json(args.out / 'runs' / f'{s["run_id"]}.json')
            for s in specs]
    doremi = next(r['doremi'] for r in runs if r['arm'] == 'doremi-weights')
    smoke_path = args.out / f'smoke-{"t2" if args.smoke_t2 else c["name"]}.json'
    merged = dict(
        schema='d2l-mix-runs/1', config=c['name'], cfg=c['proxy']._asdict(),
        mix_config=json.loads(json.dumps(c)),
        tokens=c['tokens'], domains=list(D),
        targets={k: list(v) for k, v in c['targets'].items()},
        pools={d: {k: pools[d].meta[k] for k in keep} for d in D},
        val={d: {k: vals[d].meta[k] for k in keep} for d in D},
        lr_choice=read_json(args.out / 'lr_choice.json'),
        selection=read_json(args.out / 'selection.json'), doremi=doremi,
        runs=runs, smoke=(read_json(smoke_path) if smoke_path.exists()
                          else None),
        environment=environment(d2l), wall_hours=(time.time() - t_start)
        / 3600)
    path = args.out / f'lm-mix-runs-{c["name"]}-v1.json'
    atomic_json(path, merged)
    print(f'merged {len(runs)} runs into {path}', flush=True)


def device_name(gpu):
    """nvidia-smi's name of a GPU id (without initializing CUDA here)."""
    if gpu == 'cpu':
        return 'cpu'
    r = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,'
                        'noheader', '-i', gpu], capture_output=True, text=True)
    return r.stdout.strip() or None


def default_hours(name, out, gpus):
    """The wall-clock stop of a campaign: for t2 the campaign file's
    stop_hours, else 1.5 x the smoke's predicted GPU-hours on this node's
    cards divided by the number of workers; 1 h for the other configs."""
    if name != 't2':
        return 1.0
    try:
        import yaml
        entry = next(e for e in yaml.safe_load(
            (ROOT / 'tools' / 'lm_campaigns.yaml').read_text())['campaigns']
            if e['id'] == '20.4-mixtures-t2')
        if entry.get('stop_hours'):
            return float(entry['stop_hours'])
    except (ImportError, OSError, StopIteration, KeyError):
        pass
    smoke_file = out / 'smoke-t2.json'
    if not smoke_file.exists():
        return 8.0
    sm, dev = read_json(smoke_file), device_name(gpus[0])
    if dev == sm['device']:
        hours = sm['predicted_hours_here']
    elif dev and peak_of(dev) and sm.get('mfu_campaign'):
        hours = (sm['campaign_flop'] / (peak_of(dev) * sm['mfu_campaign'])
                 / 3600 + sm['compile_hours'])
    else:
        return 8.0
    return 1.5 * hours / len(gpus)


def spawn(args, extra, gpu, threads):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='' if gpu == 'cpu' else gpu)
    if '--smoke-run' in extra:  # deterministic cuBLAS for the fp32 check
        env.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if gpu == 'cpu' and 'OMP_NUM_THREADS' not in os.environ:
        env['OMP_NUM_THREADS'] = str(threads)
    cmd = [sys.executable, __file__, '--config', args.config, '--out',
           str(args.out), '--pools', str(args.pools)] + extra + (
        ['--tokenizer', args.tokenizer] if args.tokenizer else [])
    return subprocess.Popen(cmd, env=env, start_new_session=True)


def stop_all(procs):
    for p in procs:
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGTERM)
    for p in procs:
        p.wait()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--config', required=True)
    ap.add_argument('--gpus', default='0',
                    help="comma-separated GPU ids or 'cpu', one worker each")
    ap.add_argument('--out', required=True)
    ap.add_argument('--pools', default=str(ROOT / 'data' / 'lm' / 'mix'))
    ap.add_argument('--tokenizer', default=None,
                    help='tokenizer.json of the pools in place of the '
                    'reference one (tests and smoke runs only)')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--smoke-t2', action='store_true')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--max-hours', type=float, default=None)
    ap.add_argument('--runs', default=None, help='only these run ids')
    ap.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    ap.add_argument('--smoke-config', default=None, help=argparse.SUPPRESS)
    ap.add_argument('--smoke-run', action='store_true',
                    help=argparse.SUPPRESS)
    ap.add_argument('--smoke-child', default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    args.out = pathlib.Path(args.out).resolve()
    args.pools = pathlib.Path(args.pools).resolve()
    if args.tokenizer:
        args.tokenizer = str(pathlib.Path(args.tokenizer).resolve())
    os.chdir(ROOT / 'tools')  # registered keys download to ROOT/data/lm
    if args.worker:
        return worker(args)
    if args.smoke_child:
        return smoke_child(args)
    if args.smoke_run:
        return smoke(args)
    d2l = d2l_module()
    c = configs(d2l)[args.config]
    gpus = args.gpus.split(',')
    args.out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    hours = args.max_hours or default_hours(args.config, args.out, gpus)
    if args.smoke or args.smoke_t2:
        name = args.config if args.smoke else 't2'
        p = spawn(args, ['--smoke-run', '--smoke-config', name], gpus[0],
                  os.cpu_count())
        if p.wait():
            sys.exit(f'smoke of {name} failed')
        if args.smoke:
            return
    specs = d2l.mix_run_specs(c)
    only = args.runs.split(',') if args.runs else None
    plan = dict(config=args.config, specs=specs, only=only)
    plan_path = args.out / 'plan.json'
    if plan_path.exists() and not args.resume:
        sys.exit(f'{plan_path} exists: pass --resume to continue that run')
    if plan_path.exists() and read_json(plan_path)['specs'] != json.loads(
            json.dumps(specs)):
        sys.exit(f'{plan_path} holds another plan')
    atomic_json(plan_path, plan)
    for claim in (args.out / 'runs').glob('*.claim'):
        claim.unlink()  # no worker is alive at this point
    threads = max(1, (os.cpu_count() or 1) // len(gpus))
    procs = [spawn(args, ['--worker'], g, threads) for g in gpus]

    def on_signal(signum, frame):  # the workers run in sessions of their own
        stop_all(procs)
        sys.exit(f'stopped by signal {signum}; rerun with --resume')
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    print(f'{len(specs)} runs on {gpus}; stop at {hours:.1f} h', flush=True)
    while any(p.poll() is None for p in procs):
        time.sleep(5)
        if any(p.returncode not in (None, 0) for p in procs):
            stop_all(procs)
            sys.exit('a worker failed: see its output; rerun with --resume')
        if time.time() - t_start > hours * 3600:
            stop_all(procs)
            sys.exit(f'stopped at {hours:.1f} h; rerun with --resume')
    if any(p.returncode for p in procs):
        sys.exit('a worker failed: see its output; rerun with --resume')
    if only is None:
        merge(d2l, c, args, specs, t_start)


if __name__ == '__main__':
    main()
