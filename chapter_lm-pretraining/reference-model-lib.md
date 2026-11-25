# The Reference Pretraining Run: Library Internals (build-only, unlisted)

<!--
This file is not listed in `_quarto.yml`/`CHAPTER_NUMBERING` and is never
rendered or executed as a notebook: `make lib` (`tools/build_lib.py`) only
scans it for `#@save` blocks, and `gen_notebooks.py` excludes it via
`LIB_ONLY_FILES` (appended after the numbered chapters). It holds the code
:numref:`sec_lm_reference_model` explains in a sentence each but does not
show (docs/code-guide.md 7.1):

- 21.2's registry entries (`PRETRAIN_ARTIFACTS`, `PRETRAIN_JOBS`; the
  `DATA_HUB` entries follow the uploads);
- the checkpoint schedules of the presets (`_preset_saves`), the field
  reader behind `flops_per_token`, the peak FLOP/s of the build GPU;
- `_PretrainRun`, everything `pretrain` does besides the step: device and
  DDP, compilation, AdamW, micro-batches and the global valid count, timing
  and MFU, update-RMS and attention-logit probes, validation, bf16
  checkpoints, full states (save, rolling retention, resume), branches and
  the log; the job launcher behind `launch_pretrain` and the D1 throughput
  probe;
- the readers of committed checkpoints (`load_reference_model`) and the
  Llama-format conversion for the lm-evaluation-harness cross-check.
-->

<!-- Not rendered; it sets the "Defined in" suffix of the docstrings. -->
:label:`sec_lm_reference_model`

## Registries

```{.python .input #reference-model-lib-artifacts}
%%tab pytorch
#@save
d2l.PRETRAIN_ARTIFACTS['reference-run-log'] = (
    'lm-r2-train-log-v1', 'jsonl')
d2l.PRETRAIN_ARTIFACTS['reference-evals'] = ('lm-r2-evals-v1', 'parquet')
d2l.PRETRAIN_ARTIFACTS['ladder-logs'] = ('lm-ladder-logs-v1', 'jsonl')
d2l.PRETRAIN_ARTIFACTS['ladder-evals'] = ('lm-ladder-evals-v1', 'parquet')
d2l.PRETRAIN_ARTIFACTS['t1-runs'] = ('lm-r2t1-log-v1', 'jsonl')
d2l.PRETRAIN_ARTIFACTS['instability-t1'] = ('lm-r2-inst-t1-v1', 'parquet')
d2l.PRETRAIN_ARTIFACTS['instability-t2'] = ('lm-r2-inst-t2-v1', 'parquet')
d2l.PRETRAIN_ARTIFACTS['spike-replay'] = ('lm-r2-spike-v1', 'parquet')
d2l.PRETRAIN_ARTIFACTS['throughput-probe'] = ('lm-r2-probe-v1', 'json-table')
d2l.PRETRAIN_ARTIFACTS['val-domains'] = ('lm-fwedu-val-domains-v1', 'jsonl')
```

```{.python .input #reference-model-lib-jobs}
%%tab pytorch
#@save
d2l.PRETRAIN_JOBS['t1-runs'] = ('launch_pretrain', lambda: dict(
    cfgs=[d2l.r2_config('t1', seed=s) for s in (0, 1, 2)], n_gpus=1))
d2l.PRETRAIN_JOBS['reference-run'] = ('launch_pretrain', lambda: dict(
    cfgs=d2l.r2_config('r2'), n_gpus=4))
d2l.PRETRAIN_JOBS['ladder'] = ('launch_pretrain', lambda: dict(
    cfgs=[d2l.r2_config(p) for p in ('ladder-25m', 'ladder-54m',
                                     'ladder-88m')], n_gpus=4))
```

## Presets, Fields and Peaks

```{.python .input #reference-model-lib-presets}
%%tab pytorch
#@save
def _preset_saves(preset):
    """What a preset's run saves: bf16 checkpoints, pinned full states
    (branch points and the end of the stable phase), rolling states, and
    the branches (from_step, decay_steps) that finish 10N and 5N budgets."""
    r2_ckpts = (500, 1000, 2000, 3000, 4000, 4500, 4800, 5100, 5400, 5700,
                6000)
    ladder = {'ladder-25m': ((816, 144), (408, 72)),
              'ladder-54m': ((1760, 310), (880, 155)),
              'ladder-88m': ((2857, 504), (1428, 252))}
    _, steps, _, decay, _, _ = d2l.R2_SCHEDULES[preset]
    if preset == 'r2':
        return dict(ckpt_steps=r2_ckpts, state_steps=(1275, 2550, 5100),
                    state_every=500, branches=((2550, 450), (1275, 225)))
    if preset in ladder:
        b = ladder[preset]
        return dict(ckpt_steps=(steps,), state_steps=tuple(sorted(
            f for f, _ in b)), state_every=500, branches=b)
    if preset == 't1':
        return dict(ckpt_steps=(steps,), state_steps=(steps - decay,))
    return dict(ckpt_steps=(steps,), state_every=500 if steps > 1500 else 0)

#@save
def _arch_fields(arch):
    """(L, d, H, H_kv, d_h, f, V, tied) of a Transformers config (dict or
    object; its names are read first) or of an Arch."""
    get = arch.get if isinstance(arch, dict) else (
        lambda k: getattr(arch, k, None))
    if get('num_hidden_layers') is not None:
        L, d, H = (get('num_hidden_layers'), get('hidden_size'),
                   get('num_attention_heads'))
        return (L, d, H, get('num_key_value_heads') or H,
                get('head_dim') or d // H, get('intermediate_size'),
                get('vocab_size'), bool(get('tie_word_embeddings')))
    return (arch.L, arch.d, arch.H, arch.H_kv, arch.d_h, arch.f, arch.vocab,
            arch.tie)

#@save
GPU_PEAK_BF16 = {  # dense bf16 FLOP/s with fp32 accumulation, per GPU
    'NVIDIA GeForce RTX 4090': 165e12,
    'NVIDIA A100-SXM4-40GB': 312e12, 'NVIDIA A100-SXM4-80GB': 312e12,
    'NVIDIA H100 80GB HBM3': 989e12}

#@save
def _peak_flops(device):
    if device.type != 'cuda':
        return None
    return GPU_PEAK_BF16.get(torch.cuda.get_device_name(device))

#@save
def _cfg_json(cfg):
    """A PretrainConfig as JSON values (the arch as a dict)."""
    out = d2l._jsonable(cfg)
    out['arch'] = cfg.arch._asdict()
    return out

#@save
def _cfg_from_json(d):
    d = dict(d)
    d['arch'] = d2l.Arch(**d['arch'])
    for k in ('betas', 'ckpt_steps', 'state_steps'):
        d[k] = tuple(d[k])
    d['branches'] = tuple(tuple(b) for b in d['branches'])
    return d2l.PretrainConfig(**d)

#@save
def _train_shards(cfg):
    """The training shards of a config: '100m' (one shard) or '4b' (the
    first 32 of the 40 shards, 3.2B tokens: the reference run's epoch)."""
    with d2l._quiet():
        if cfg.data == '100m':
            return d2l.fwedu_shards('100m')
        if cfg.data == '4b':
            return d2l.fwedu_shards('4b', max_tokens=3_145_728_001)
    raise ValueError(f'unknown data {cfg.data!r}')
```

## The Run Behind `pretrain`

```{.python .input #reference-model-lib-run}
%%tab pytorch
#@save
_Segment = collections.namedtuple('_Segment', 'name start total decay')

#@save
class _PretrainRun:
    """Everything pretrain does besides the step (see the file header)."""
    def __init__(self, model, stream, cfg, val_fn, resume):
        self.cfg, self.stream, self.val_fn = cfg, stream, val_fn
        self.rank, self.world, self.device = (
            d2l._ddp_setup() if cfg.ddp else (0, 1, next(
                model.parameters()).device))
        if (stream.rank, stream.world) != (self.rank, self.world):
            raise ValueError(f'stream is rank {stream.rank} of {stream.world}'
                             f', the run {self.rank} of {self.world}')
        self.raw = model.to(self.device)
        cuda = self.device.type == 'cuda'
        self.optimizer = torch.optim.AdamW(
            d2l.param_groups(self.raw, cfg.wd), lr=cfg.peak_lr,
            betas=cfg.betas, eps=cfg.eps, fused=cuda)
        net = self.raw
        if cfg.ddp:
            net = torch.nn.parallel.DistributedDataParallel(
                net, device_ids=[self.device] if cuda else None,
                gradient_as_bucket_view=True)
        self.model = torch.compile(net) if cfg.compile else net
        self.accum = cfg.accum(self.world)
        self.step_tokens = cfg.global_batch * cfg.arch.T
        self.peak = _peak_flops(self.device)
        self.out, self.rows, self.halt = cfg.out_dir, [], False
        self.grad_hist = collections.deque(maxlen=100)
        self._mem_states, self._expect = {}, None
        self.start = 0
        path = self._resume_path(resume)
        if path is not None:
            self.start = self._load_state(path)
            self._expect = self.start + 1
        self.log_path = self._log_file(cfg.preset)
        if self.log_path and self.rank == 0:
            d2l._truncate_log(self.log_path, self.start)

    # -- segments: the main run, then each branch from its saved state --
    def segments(self):
        cfg = self.cfg
        if self.start < cfg.steps:
            yield _Segment(cfg.preset, self.start, cfg.steps, cfg.decay)
        for frm, dec in cfg.branches:
            name = f'b{frm + dec}'
            if self.halt or self._branch_done(name, frm + dec):
                continue
            self._load_state(self._state_source(frm))
            self.log_path = self._log_file(name)
            if self.log_path and self.rank == 0:
                d2l._truncate_log(self.log_path, frm)
            yield _Segment(name, frm, frm + dec, dec)

    def batch(self, step):
        """This rank's batch at step, its micro-batches on the device, and
        the step's global count of valid targets (one all-reduce)."""
        b = self.stream.batch(step)
        cuda = self.device.type == 'cuda'
        dev = {k: (v.pin_memory() if cuda else v).to(self.device,
                                                      non_blocking=True)
               for k, v in b.items()}
        n_valid = d2l._all_reduce_sum((dev['labels'] != -100).sum())
        m = self.cfg.micro_batch
        micro = [{k: v[i:i + m] for k, v in dev.items()}
                 for i in range(0, len(dev['labels']), m)]
        assert len(micro) == self.accum, (len(micro), self.accum)
        return dev, micro, n_valid

    def snapshot(self, step):
        """Weights before the step, at probe steps (for the update RMS)."""
        every = self.cfg.probe_every
        if not every or (step + 1) % every:
            return None
        return {n: p.detach().clone() for n, p in self.raw.named_parameters()}

    def record(self, seg, row, batch, before, wait, t0):
        """Complete row in place (one host sync), probe, validate, save."""
        cfg, done = self.cfg, row['step']
        for k, v in row.items():
            if torch.is_tensor(v):
                row[k] = v.item()
        row['step_s'] = time.perf_counter() - t0
        row['data_wait_s'], row['tok_s'] = wait, self.step_tokens / row[
            'step_s']
        row['mfu'] = (d2l.mfu(row['tok_s'], cfg.arch, self.peak, self.world)
                      if self.peak else float('nan'))
        row['run'], row['seed'] = seg.name, cfg.seed
        if self._expect is not None:  # the scheduler did not restart at 0
            assert done == self._expect, (done, self._expect)
            self._expect = None
        hist = sorted(self.grad_hist)
        row['spike'] = bool(len(hist) >= 10 and row['grad_norm'] > 10 * hist[
            len(hist) // 2])
        self.grad_hist.append(row['grad_norm'])
        if before is not None:
            row.update(d2l._update_rms(self.raw, before, self.optimizer))
        if cfg.stats_every and done % cfg.stats_every == 0:
            for l, s in d2l.attn_logit_stats(self.raw, batch).items():
                row[f'max_logit_{l}'] = float(s.max_logit.max())
        last = done == seg.total
        if self.val_fn is not None and self.rank == 0 and (
                last or (cfg.val_every and done % cfg.val_every == 0)):
            row['val_loss'] = self.val_fn(self.raw)
        self.stream.step = done
        self._save(seg, done, last or not math.isfinite(row['loss']))
        if self.rank == 0:
            self.rows.append(dict(row))
            if self.log_path:
                with open(self.log_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(d2l._jsonable(row)) + '\n')

    def stopped(self, row):
        """True (and no further segment) after a non-finite loss."""
        if not math.isfinite(row['loss']):
            self.halt = True
        return self.halt

    def finish(self):
        if self.rank != 0:
            return pd.DataFrame()
        if self.out:
            df = d2l._read_job_logs(self.out)
        else:
            df = pd.DataFrame(self.rows)
        df.attrs['label'] = ''
        return df

    # -- checkpoints and states --
    def _log_file(self, name):
        if not self.out:
            return None
        os.makedirs(self.out, exist_ok=True)
        return os.path.join(self.out, 'log.jsonl' if name == self.cfg.preset
                            else f'log-{name}.jsonl')

    def _save(self, seg, done, last):
        cfg, main = self.cfg, seg.name == self.cfg.preset
        if self.rank == 0 and self.out and ((main and done in cfg.ckpt_steps)
                                            or (last and not main)):
            name = f's{done}' if main else f'{seg.name}-s{done}'
            d2l._save_bf16(os.path.join(self.out, 'ckpt',
                                        name + '.safetensors'),
                           self.raw, done)
        pinned = done in cfg.state_steps or (main and last)
        rolling = cfg.state_every and done % cfg.state_every == 0
        if main and (pinned or rolling):
            self._save_state(done, pinned)

    def _save_state(self, step, pinned):
        import torch.distributed as dist
        rng = dict(cpu=torch.get_rng_state(), cuda=(
            torch.cuda.get_rng_state(self.device)
            if self.device.type == 'cuda' else None))
        rngs = [rng]
        if self.world > 1:
            rngs = [None] * self.world
            dist.all_gather_object(rngs, rng)
        if self.rank == 0:
            stream = dict(self.stream.state_dict(), step=step)
            state = dict(model=self.raw.state_dict(),
                         optim=self.optimizer.state_dict(), step=step,
                         stream=stream, rng=rngs, cfg=_cfg_json(self.cfg))
            if not self.out:
                import copy
                self._mem_states[step] = copy.deepcopy(
                    {k: v for k, v in state.items()})
            else:
                d2l._write_state(self.out, step, state, pinned,
                                 self.cfg.state_steps)
        if self.world > 1:
            dist.barrier()

    def _load_state(self, src):
        state = (src if isinstance(src, dict) else
                 torch.load(src, map_location='cpu', weights_only=False))
        self.raw.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optim'])
        self.stream.load_state_dict(state['stream'])
        rng = state['rng'][self.rank if len(state['rng']) == self.world
                           else 0]
        torch.set_rng_state(rng['cpu'])
        if rng['cuda'] is not None and self.device.type == 'cuda':
            torch.cuda.set_rng_state(rng['cuda'], self.device)
        return state['step']

    def _resume_path(self, resume):
        if resume in (None, 'none'):
            return None
        if resume != 'auto':
            return resume
        steps = d2l._state_steps(self.out)
        main = [s for s in steps if s <= self.cfg.steps]
        return d2l._state_path(self.out, max(main)) if main else None

    def _state_source(self, step):
        if step in self._mem_states:
            return self._mem_states[step]
        if self.out and os.path.exists(d2l._state_path(self.out, step)):
            return d2l._state_path(self.out, step)
        raise FileNotFoundError(f'no saved state at step {step} to branch '
                                'from (add it to state_steps)')

    def _branch_done(self, name, total):
        return bool(self.out) and os.path.exists(os.path.join(
            self.out, 'ckpt', f'{name}-s{total}.safetensors'))
```

```{.python .input #reference-model-lib-state}
%%tab pytorch
#@save
def _state_path(out, step):
    return os.path.join(out, 'state', f's{step}.pt')

#@save
def _state_steps(out):
    """Steps of the full states saved under out/state, ascending."""
    folder = os.path.join(out or '', 'state')
    if not out or not os.path.isdir(folder):
        return []
    return sorted(int(f[1:-3]) for f in os.listdir(folder)
                  if f.startswith('s') and f.endswith('.pt'))

#@save
def _write_state(out, step, state, pinned, pinned_steps):
    """Write a full state through a temporary file; keep the pinned states,
    the final one and the newest two of the rest."""
    path = _state_path(out, step)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path + '.tmp')
    os.replace(path + '.tmp', path)
    keep = set(pinned_steps) | ({step} if pinned else set())
    rolling = [s for s in _state_steps(out) if s not in keep]
    for s in rolling[:-2]:
        os.remove(_state_path(out, s))

#@save
def _save_bf16(path, model, step):
    """bf16 weights with the model's config() and step as metadata."""
    from safetensors.torch import save_file
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {k: v.detach().to(torch.bfloat16).contiguous().cpu()
             for k, v in model.state_dict().items()}
    save_file(state, path + '.tmp', metadata=dict(
        config=json.dumps(model.config()), step=str(step)))
    os.replace(path + '.tmp', path)

#@save
def _load_bf16(path, dtype=torch.bfloat16, device='cpu'):
    """The PackedGPT a _save_bf16 file holds, in eval mode."""
    from safetensors import safe_open
    from safetensors.torch import load_file
    with safe_open(path, 'pt') as f:
        config = json.loads(f.metadata()['config'])
    model = d2l._packed_from_config(config, meta=True)
    state = {k: v.to(dtype) for k, v in load_file(path,
                                                  device=str(device)).items()}
    model.load_state_dict(state, assign=True)
    return model.eval()

#@save
def _truncate_log(path, step):
    """Keep the rows of a jsonl log up to step (a resumed run rewrites the
    rest)."""
    if not os.path.exists(path):
        return
    with open(path, encoding='utf-8') as f:
        rows = [line for line in f if json.loads(line)['step'] <= step]
    with open(path + '.tmp', 'w', encoding='utf-8') as f:
        f.writelines(rows)
    os.replace(path + '.tmp', path)

#@save
def _read_job_logs(out):
    """All rows of a job folder: the main log, then each branch's."""
    files = ['log.jsonl'] + sorted(f for f in os.listdir(out)
                                   if f.startswith('log-')
                                   and f.endswith('.jsonl'))
    rows = []
    for name in files:
        path = os.path.join(out, name)
        if os.path.exists(path):
            with open(path, encoding='utf-8') as f:
                rows += [json.loads(line) for line in f]
    return pd.DataFrame(rows)

#@save
def _module_kind(name):
    """'W_q', ..., 'W_2', 'token_emb' or 'norm' for a parameter name."""
    parts = name.split('.')
    if parts[0] in ('token_emb', 'pos_emb', 'lm_head'):
        return parts[0]
    return 'norm' if 'norm' in parts[-2] else parts[-2]

#@save
def _update_rms(model, before, optimizer):
    """Median over layers, per module kind, of the RMS of the step's AdamW
    update in units of the learning rate (weight decay taken out)."""
    group = {id(p): g for g in optimizer.param_groups for p in g['params']}
    kinds = collections.defaultdict(list)
    for name, p in model.named_parameters():
        g = group[id(p)]
        u = (before[name] * (1 - g['lr'] * g['weight_decay']) - p.detach()
             ) / g['lr']
        kinds[_module_kind(name)].append(u.float().pow(2).mean().sqrt())
    med = {k: torch.stack(v).median() for k, v in kinds.items()}
    return {f'rms_{k}': v for k, v in zip(med, torch.stack(
        list(med.values())).tolist())}
```

```{.python .input #reference-model-lib-val}
%%tab pytorch
#@save
def _val_fn(cfg, batch_size=None):
    """Validation loss on the first cfg.val_tokens tokens of the validation
    shard, as windows of arch.T tokens with the training targets."""
    with d2l._quiet():
        shards = d2l.fwedu_shards('val')
    T, eos = cfg.arch.T, shards.eos_id
    n = cfg.val_tokens // T
    windows = np.stack([shards.read(i * T, T + 1) for i in range(n)])
    bs = batch_size or cfg.micro_batch

    def val(model):
        dev = next(model.parameters()).device
        total, count = 0.0, 0
        with torch.inference_mode(), torch.autocast(
                dev.type, torch.bfloat16, enabled=cfg.dtype == 'bf16'):
            for i in range(0, n, bs):
                W = torch.from_numpy(windows[i:i + bs]).long().to(dev)
                X, Y = W[:, :-1], W[:, 1:].clone()
                Y[X == eos] = -100
                ctx = d2l.PackCtx(*d2l.doc_ids_from_eos(X.int(), eos))
                H, Wt = model(X, ctx, return_hidden=True)
                s, c = d2l.chunked_lm_loss(H, Wt, Y)
                total, count = total + s.float(), count + c
        return float(total / count)
    return val
```

## Probes of the Attention Logits

```{.python .input #reference-model-lib-probes}
%%tab pytorch
#@save
def _probe_inputs(batch, max_seqs, model):
    """Token ids and PackCtx of the first max_seqs sequences of batch, on
    the model's device."""
    dev = next(d2l._unwrap(model).parameters()).device
    X = batch['input_ids'][:max_seqs].to(dev)
    ctx = d2l.PackCtx(batch['doc_ids'][:max_seqs].to(dev),
                      batch['pos_ids'][:max_seqs].to(dev))
    return X, ctx

#@save
def _attention_inputs(model, X, ctx):
    """(attention module, its input, its RoPE tables) for every block of
    one eager forward pass."""
    raw, seen = d2l._unwrap(model), []

    def grab(module, args, kwargs):
        seen.append((module, args[0].detach(), kwargs.get('tables')))
    hooks = [blk.attention.register_forward_pre_hook(grab, with_kwargs=True)
             for blk in raw.blks]
    try:
        with torch.no_grad(), torch.autocast(X.device.type, torch.bfloat16,
                                             enabled=X.is_cuda):
            raw(X, ctx)
    finally:
        for h in hooks:
            h.remove()
    return seen

#@save
def _chunked_logit_stats(q, k, ctx, chunk=512):
    """Per head: the largest logit q.k / sqrt(d_h) a query gives a key it
    may see (same document, not later), and the mean entropy of its
    attention, from fp32 scores, chunk queries at a time."""
    B, H, T, d_h = q.shape
    q, k = q.float(), k.float()
    mx = torch.full((H,), -math.inf, device=q.device)
    ent = torch.zeros(H, device=q.device)
    j = torch.arange(T, device=q.device)
    for a in range(0, T, chunk):
        s = q[:, :, a:a + chunk] @ k.transpose(-2, -1) / math.sqrt(d_h)
        i = torch.arange(a, min(a + chunk, T), device=q.device)
        ok = j[None, :] <= i[:, None]
        if ctx is not None:
            d = ctx.doc_ids
            ok = ok & (d[:, i, None] == d[:, None, :])[:, None]
        s = s.masked_fill(~ok, -math.inf)
        mx = torch.maximum(mx, s.amax((0, 2, 3)))
        p = torch.softmax(s, -1)
        ent += -(p * torch.log(p.clamp_min(1e-30))).sum(-1).sum((0, 2))
    return mx, ent / (B * T)
```

## Jobs: `launch_pretrain` and the Throughput Probe

```{.python .input #reference-model-lib-launch}
%%tab pytorch
#@save
_RUN_NAMES = {  # presets of one launch_pretrain call: its regeneration name
    ('t1',): 't1-runs', ('r2',): 'reference-run',
    ('ladder-25m', 'ladder-54m', 'ladder-88m'): 'ladder'}

#@save
def launch_pretrain(cfgs, n_gpus=4):
    """The logs of pretraining runs (one config or a list: seeds, ladder
    sizes) as one DataFrame with columns run and seed: the committed result
    of this call's configuration, or, under D2L_RERUN, its runs launched
    with torchrun on n_gpus GPUs (tools/lm_pretrain.py, one job folder per
    run, resumable)."""
    cfgs = [cfgs] if isinstance(cfgs, d2l.PretrainConfig) else list(cfgs)
    presets = tuple(sorted({c.preset for c in cfgs}))
    if presets not in _RUN_NAMES:
        raise ValueError(f'no committed run for presets {presets}')
    name = _RUN_NAMES[presets]
    config = dict(n_gpus=n_gpus, runs=[_cfg_json(c) for c in cfgs])
    log = d2l._pretrain_result(
        name, lambda: _launch_jobs(cfgs, n_gpus), config,
        fallback=None if name == 't1-runs' else 't1-runs')
    if log.attrs.get('label', '').startswith('T2 run pending'):
        label = log.attrs['label']
        log = log[log['seed'] == 0].reset_index(drop=True)
        log.attrs['label'] = label
    return log

#@save
def _job_root():
    return os.path.abspath(os.path.join('..', 'data', 'lm', 'rerun', 'ch21'))

#@save
def _launch_jobs(cfgs, n_gpus, max_restarts=3):
    """Run every config's job that is not done (D2L_PRETRAIN_ONLY limits
    this process to one preset or job); the merged logs once all are done,
    else None."""
    import subprocess
    import sys
    only = os.environ.get('D2L_PRETRAIN_ONLY')
    repo = os.path.dirname(os.path.dirname(os.path.abspath(d2l.__file__)))
    for cfg in cfgs:  # extract the manifests before any rank starts
        _train_shards(cfg)
    logs, world = [], max(1, min(n_gpus, torch.cuda.device_count()))
    for cfg in cfgs:
        job = os.path.join(_job_root(), f'{cfg.preset}-s{cfg.seed}')
        done, bad = os.path.join(job, 'DONE'), os.path.join(job, 'NONFINITE')
        if not os.path.exists(done) and (not only or only in (
                cfg.preset, os.path.basename(job))):
            os.makedirs(job, exist_ok=True)
            with open(os.path.join(job, 'config.json'), 'w') as f:
                json.dump(_cfg_json(dataclasses.replace(
                    cfg, out_dir=job, ddp=world > 1)), f, indent=1)
            cmd = [sys.executable, '-m', 'torch.distributed.run',
                   '--standalone', f'--nproc-per-node={world}',
                   os.path.join(repo, 'tools', 'lm_pretrain.py'), '--config',
                   os.path.join(job, 'config.json'), '--resume', 'auto']
            for attempt in range(max_restarts + 1):
                with open(os.path.join(job, 'train.log'), 'a') as out:
                    rc = subprocess.call(cmd, cwd=repo, stdout=out,
                                         stderr=subprocess.STDOUT)
                if rc == 0 or os.path.exists(done) or os.path.exists(bad):
                    break
            if os.path.exists(bad):
                raise FloatingPointError(f'{job}: non-finite loss')
            if not os.path.exists(done):
                raise RuntimeError(f'{job}: failed {max_restarts + 1} times '
                                   f'(exit {rc}); see train.log')
        if os.path.exists(done):
            logs.append(_read_job_logs(job))
    if len(logs) < len(cfgs):
        return None
    df = pd.concat(logs, ignore_index=True)
    df.attrs['label'] = ''
    return df
```

```{.python .input #reference-model-lib-probe}
%%tab pytorch
#@save
def probe_throughput(cfg, micro_batches=(8, 4, 2), warmup=5, steps=20,
                     out=None):
    """The D1 probe: for each micro-batch size, compile time, median step
    time (IQR), tokens/s, MFU, peak memory per rank, the all-reduce share
    (steps without gradient sync), validation and checkpoint times, and
    the initial loss and gradient norm. One dict per micro-batch size."""
    import tempfile
    rows = []
    rank, world, device = d2l._ddp_setup() if cfg.ddp else (0, 1, torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'))
    shards = _train_shards(cfg)
    for m in micro_batches:
        c = dataclasses.replace(cfg, micro_batch=m, val_every=0, ckpt_steps=(),
                                state_steps=(), state_every=0, branches=(),
                                out_dir=None, probe_every=0, stats_every=0)
        row = dict(preset=cfg.preset, gpus=world, micro=m)
        try:
            row.update(_probe_one(c, shards, rank, world, device, warmup,
                                  steps))
        except torch.OutOfMemoryError:
            row['status'] = 'oom'
        rows.append(row)
        torch._dynamo.reset()
        d2l.lm_free()
    if rank == 0 and out:
        with tempfile.TemporaryDirectory(dir=out) as tmp:
            rows.append(_probe_io(cfg, tmp, device))
        d2l.write_jsonl(os.path.join(
            out, f'probe-{cfg.preset}-g{world}.jsonl'), rows)
    return rows

#@save
def _probe_one(cfg, shards, rank, world, device, warmup, steps):
    torch.manual_seed(cfg.seed)
    model = d2l.PackedGPT.from_arch(cfg.arch).to(device)
    stream = d2l.PackedStream(shards, cfg.arch.T, cfg.global_batch,
                              seed=cfg.data_seed, rank=rank, world=world)
    run = _PretrainRun(model, stream, cfg, None, None)
    torch.cuda.reset_peak_memory_stats(device)

    def timed(step):
        for g in run.optimizer.param_groups:
            g['lr'] = d2l.wsd_lr(step, cfg.steps, cfg.warmup, cfg.decay,
                                 cfg.peak_lr)
        torch.cuda.synchronize(device)
        t = time.perf_counter()
        _, micro, n = run.batch(step)
        out = d2l.pretrain_step(run.model, run.optimizer, micro, cfg, n)
        loss, norm = out['loss'].item(), out['grad_norm'].item()
        torch.cuda.synchronize(device)
        return time.perf_counter() - t, loss, norm
    t0, loss0, norm0 = timed(0)
    for s in range(1, warmup):
        timed(s)
    ts = np.array([timed(warmup + s)[0] for s in range(steps)])
    d2l._FORCE_NO_SYNC[0] = world > 1
    try:
        t_ns = np.median([timed(warmup + steps + s)[0]
                          for s in range(steps // 2)]) if world > 1 else None
    finally:
        d2l._FORCE_NO_SYNC[0] = False
    gib = 2 ** 30
    mem = torch.tensor([torch.cuda.max_memory_allocated(device) / gib,
                        torch.cuda.max_memory_reserved(device) / gib],
                       device=device)
    mem = d2l._all_reduce_max(mem).tolist()
    step_s = float(np.median(ts))
    tok_s = cfg.global_batch * cfg.arch.T / step_s
    peak = _peak_flops(device)
    return dict(accum=cfg.accum(world), first_step_s=t0, step_s=step_s,
                step_iqr_s=float(np.subtract(*np.percentile(ts, [75, 25]))),
                tok_s=tok_s, mfu=d2l.mfu(tok_s, cfg.arch, peak, world)
                if peak else None, peak_alloc_gib=mem[0],
                peak_reserved_gib=mem[1],
                allreduce_share=None if t_ns is None else (step_s - t_ns)
                / step_s, init_loss=loss0, init_grad_norm=norm0, status='ok')

#@save
def _probe_io(cfg, tmp, device):
    """Seconds for one validation pass and for the two checkpoint kinds."""
    torch.manual_seed(0)
    model = d2l.PackedGPT.from_arch(cfg.arch).to(device)
    opt = torch.optim.AdamW(d2l.param_groups(model, cfg.wd), lr=1e-4,
                            fused=device.type == 'cuda')
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    opt.step()
    val = _val_fn(cfg)
    t = time.perf_counter()
    val_loss = val(model)
    t_val = time.perf_counter() - t
    t = time.perf_counter()
    _save_bf16(os.path.join(tmp, 'ckpt.safetensors'), model, 0)
    t_bf16 = time.perf_counter() - t
    t = time.perf_counter()
    _write_state(tmp, 0, dict(model=model.state_dict(),
                              optim=opt.state_dict()), True, ())
    t_state = time.perf_counter() - t
    return dict(preset=cfg.preset, io=True, val_s=t_val, val_loss=val_loss,
                ckpt_bf16_s=t_bf16, state_s=t_state)
```

## Reading Committed Checkpoints

```{.python .input #reference-model-lib-load}
%%tab pytorch
#@save
REFERENCE_STEPS = (500, 1000, 2000, 3000, 4000, 4500, 4800, 5100, 5400, 5700,
                   6000)

#@save
def _reference_key(preset, step):
    if preset == 'r2':
        step = 6000 if step is None else step
        if step not in REFERENCE_STEPS:
            raise ValueError(f'step is one of {REFERENCE_STEPS}')
        return f'lm-r2-s{step}-v1'
    if preset == 't1':
        if step not in (None, 1526):
            raise ValueError('the t1 checkpoint is step 1526')
        return 'lm-r2t1-s1526-v1'
    if preset in ('r2-16k-pi', 'r2-16k-yarn'):
        return f'lm-{preset}-v1'
    raise ValueError(f'unknown preset {preset!r}')

#@save
def load_reference_model(step=None, dtype=torch.bfloat16, device='cpu',
                         preset='r2'):
    """A committed checkpoint of the reference run (preset 'r2', steps
    REFERENCE_STEPS, default 6000), of the t1 run ('t1'), or of a 16K arm
    ('r2-16k-pi', 'r2-16k-yarn'), as a PackedGPT in eval mode. Its
    tokenizer is d2l.r2_tokenizer()."""
    key = _reference_key(preset, step)
    local = os.path.join(d2l._rerun_dir(), d2l._next_key(key)
                         + '.safetensors')
    if d2l._regenerating() and os.path.exists(local):
        path = local
    elif key in d2l.DATA_HUB:
        with d2l._quiet():
            path = d2l.download_keyed(key)
    else:
        raise KeyError('reference checkpoints not committed yet')
    return _load_bf16(path, dtype, device)

#@save
def packed_to_hf(model):
    """A Transformers LlamaForCausalLM with the weights of a PackedGPT
    without QK-norm (RoPE pairs moved back to Llama's split halves)."""
    import transformers
    c = model.config()
    if c['qk_norm'] or c['softcap'] or c['pos'] != 'rope':
        raise ValueError('Llama has no QK-norm, softcap or learned positions')
    hf_cfg = transformers.LlamaConfig(
        vocab_size=c['vocab_size'], hidden_size=c['num_hiddens'],
        intermediate_size=c['ffn_hidden'], num_hidden_layers=c['num_blks'],
        num_attention_heads=c['num_heads'],
        num_key_value_heads=c['num_kv_heads'], head_dim=c['head_dim'],
        rms_norm_eps=c['norm_eps'], rope_theta=c['rope_theta'],
        max_position_embeddings=c['max_len'],
        tie_word_embeddings=c['tie_embeddings'], attention_bias=False,
        mlp_bias=False, bos_token_id=None, eos_token_id=0, pad_token_id=1)
    d_h = c['head_dim']
    inv = torch.argsort(d2l.qwen3_rope_permutation(d_h))

    def unpermute(W):
        return W.unflatten(0, (-1, d_h))[:, inv].flatten(0, 1)
    names = {'token_emb': 'model.embed_tokens', 'norm': 'model.norm',
             'lm_head': 'lm_head'}
    layer = {'norm1': 'input_layernorm', 'norm2': 'post_attention_layernorm',
             'attention.W_q': 'self_attn.q_proj',
             'attention.W_k': 'self_attn.k_proj',
             'attention.W_v': 'self_attn.v_proj',
             'attention.W_o': 'self_attn.o_proj', 'ffn.W_g': 'mlp.gate_proj',
             'ffn.W_1': 'mlp.up_proj', 'ffn.W_2': 'mlp.down_proj'}
    state = {}
    for k, v in model.state_dict().items():
        mod = k[:-len('.weight')]
        if mod in names:
            state[names[mod] + '.weight'] = v
            continue
        _, i, rest = mod.split('.', 2)
        if rest in ('attention.W_q', 'attention.W_k'):
            v = unpermute(v)
        state[f'model.layers.{i}.{layer[rest]}.weight'] = v
    hf = transformers.LlamaForCausalLM(hf_cfg)  # (its RoPE buffer is built)
    missing, unexpected = hf.load_state_dict(state, strict=False)
    if unexpected or set(missing) - {'lm_head.weight'}:
        raise ValueError(f'packed_to_hf: {missing} {unexpected}')
    return hf.to(next(model.parameters()).dtype).eval()
```
