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
- the readers of committed checkpoints (`load_reference_model`,
  `load_reference_models`) and the Llama-format conversion for the
  lm-evaluation-harness cross-check;
- the drivers behind the page's one-line calls: the opening's
  `gradient_event`, the plot helpers (`plot_run_log`, `plot_traces`,
  `plot_ladder_runs`), the checkpoint evaluations, the instability arms and the
  spike protocol. While the reference run is pending, each shows the T1
  configuration under a printed label (the T1 run keeps checkpoints at the
  reference run's fractions of training for this).
-->

<!-- Not rendered; it sets the "Defined in" suffix of the docstrings. -->
:label:`sec_lm_reference_model`

## Registries

The hosted inputs and results (d2l-data R2), one entry per upload: the
t1 runs' log (three seeds) with seed 0's eleven bf16 checkpoints and its
plateau state (the stand-ins while the reference run is pending). The
per-domain validation texts (built by `tools/build_pretraining_data.py
--stage domains`) are FineWeb-Edu file 012 by URL host, about 2M R2 tokens
per domain (FineWeb-Edu: ODC-By 1.0, Hugging Face), and the book's held-out
sections (CC BY-SA 4.0).

```{.python .input #reference-model-lib-hub}
%%tab pytorch
#@save
d2l.DATA_HUB['lm-fwedu-val-domains-v1'] = (
    d2l.LM_DATA_URL + 'lm-fwedu-val-domains-v1.jsonl.gz',
    '180e3865876bb019eea808b4222a090bedda9aaa')
d2l.DATA_HUB['lm-r2t1-log-v1'] = (
    d2l.LM_DATA_URL + 'lm-r2t1-log-v1.jsonl.gz',
    'cb398bde00c2415fd1e82a69501cce64e57341e9')
d2l.DATA_HUB['lm-r2t1-s127-v1'] = (
    d2l.LM_DATA_URL + 'lm-r2t1-s127-v1.safetensors',
    '45d9c3c86ee988d2279bfab40a0412fd050561cf')
d2l.DATA_HUB['lm-r2t1-s254-v1'] = (
    d2l.LM_DATA_URL + 'lm-r2t1-s254-v1.safetensors',
    'ab02e70bbbc42ee92832ec812194bbf9e2412cbc')
d2l.DATA_HUB['lm-r2t1-s509-v1'] = (
    d2l.LM_DATA_URL + 'lm-r2t1-s509-v1.safetensors',
    'faa881f42c6066194bbc2f76e24b928cb71bdf55')
d2l.DATA_HUB['lm-r2t1-s763-v1'] = (
    d2l.LM_DATA_URL + 'lm-r2t1-s763-v1.safetensors',
    '1eeb53e34acbaa8242896f70b16967d2e1c29aec')
d2l.DATA_HUB['lm-r2t1-s1017-v1'] = (
    d2l.LM_DATA_URL + 'lm-r2t1-s1017-v1.safetensors',
    '0edbc49afe5387eb08a4f04cc39720d69c6d4455')
d2l.DATA_HUB['lm-r2t1-s1144-v1'] = (
    d2l.LM_DATA_URL + 'lm-r2t1-s1144-v1.safetensors',
    '69db306e06d1abf391be788ad5d64d46c77e5169')
d2l.DATA_HUB['lm-r2t1-s1221-v1'] = (
    d2l.LM_DATA_URL + 'lm-r2t1-s1221-v1.safetensors',
    '47b2715c2c4af418e4f449beffb504faaf4118e7')
d2l.DATA_HUB['lm-r2t1-s1297-v1'] = (
    d2l.LM_DATA_URL + 'lm-r2t1-s1297-v1.safetensors',
    'c2446e3d272e84204c053b417acc02d21eee4ad7')
d2l.DATA_HUB['lm-r2t1-s1373-v1'] = (
    d2l.LM_DATA_URL + 'lm-r2t1-s1373-v1.safetensors',
    '91593410d65e6edb0a1b8681a2a20344e7be04c3')
d2l.DATA_HUB['lm-r2t1-s1450-v1'] = (
    d2l.LM_DATA_URL + 'lm-r2t1-s1450-v1.safetensors',
    '3216071fcd7a5eb6f0afd7da3fd6c19545b1bcff')
d2l.DATA_HUB['lm-r2t1-s1526-v1'] = (
    d2l.LM_DATA_URL + 'lm-r2t1-s1526-v1.safetensors',
    'ab1404181189d79fc45fd581a2e9f32fe0f64bfc')
d2l.DATA_HUB['lm-r2t1-s1297-state-v1'] = (
    d2l.LM_DATA_URL + 'lm-r2t1-s1297-state-v1.pt',
    '9e635a58da51f219333f35c2a1417a06124488ba')
d2l.DATA_HUB['lm-r2t1-evals-v1'] = (
    d2l.LM_DATA_URL + 'lm-r2t1-evals-v1.json.gz',
    '72c3ecec0b84a0a9bd899e6dd8ba1128ce559e42')
d2l.DATA_HUB['lm-r2-spike-v1'] = (
    d2l.LM_DATA_URL + 'lm-r2-spike-v1.json.gz',
    'b5616a662d8eeb4a04c41cc3e3637f4d80563aa3')
d2l.DATA_HUB['lm-r2-inst-t1-v1'] = (
    d2l.LM_DATA_URL + 'lm-r2-inst-t1-v1.json.gz',
    '7095f284a524f7a21512483cfe5ecc7093762638')
d2l.DATA_HUB['lm-r2-train-log-v1'] = (
    d2l.LM_DATA_URL + 'lm-r2-train-log-v1.jsonl.gz',
    '8c955acf12a939307fed602bd22fa6ce769c5ed3')
d2l.DATA_HUB['lm-r2-config-v1'] = (
    d2l.LM_DATA_URL + 'lm-r2-config-v1.json',
    '4eb73e0e30ca4aaeb6005ff62693f1ca7554a44f')
d2l.DATA_HUB['lm-r2-probe-v1'] = (
    d2l.LM_DATA_URL + 'lm-r2-probe-v1.json.gz',
    'f4dd9bc981c7984d120a896c10ae8370ce6b8ec2')
d2l.DATA_HUB['lm-r2-s500-v1'] = (
    d2l.LM_DATA_URL + 'lm-r2-s500-v1.safetensors',
    'fc7ea408faea5c936763b42f96ee5547bc9f62d3')
d2l.DATA_HUB['lm-r2-s1000-v1'] = (
    d2l.LM_DATA_URL + 'lm-r2-s1000-v1.safetensors',
    '9fc7ef006390315141af3c16aaf313d420ade8d7')
d2l.DATA_HUB['lm-r2-s2000-v1'] = (
    d2l.LM_DATA_URL + 'lm-r2-s2000-v1.safetensors',
    'f2befa5e2554b6be91766a772396ee10ed1cb2c5')
d2l.DATA_HUB['lm-r2-s3000-v1'] = (
    d2l.LM_DATA_URL + 'lm-r2-s3000-v1.safetensors',
    'a665e2983d26dffb546c92b46c79ee041fb0c7b9')
d2l.DATA_HUB['lm-r2-s4000-v1'] = (
    d2l.LM_DATA_URL + 'lm-r2-s4000-v1.safetensors',
    'f1623a5452ffc204e3803903f458f542bb04ab8e')
d2l.DATA_HUB['lm-r2-s4500-v1'] = (
    d2l.LM_DATA_URL + 'lm-r2-s4500-v1.safetensors',
    '4d6d5ddfe8ea57a67ac953b6f9321b038f681845')
d2l.DATA_HUB['lm-r2-s4800-v1'] = (
    d2l.LM_DATA_URL + 'lm-r2-s4800-v1.safetensors',
    '401899e18609a7d6911e53c6afdfd4a35d92a77d')
d2l.DATA_HUB['lm-r2-s5100-v1'] = (
    d2l.LM_DATA_URL + 'lm-r2-s5100-v1.safetensors',
    '8d48b8fd79c5b69f68cc2fd51eb87128cf8fae89')
d2l.DATA_HUB['lm-r2-s5400-v1'] = (
    d2l.LM_DATA_URL + 'lm-r2-s5400-v1.safetensors',
    '9f6bc7b3822086e5014e1d5d592e78ab37c80f41')
d2l.DATA_HUB['lm-r2-s5700-v1'] = (
    d2l.LM_DATA_URL + 'lm-r2-s5700-v1.safetensors',
    '7511a32055eb951e5852a9df6e8b864b18710b23')
d2l.DATA_HUB['lm-r2-s6000-v1'] = (
    d2l.LM_DATA_URL + 'lm-r2-s6000-v1.safetensors',
    '869d5f6637f455108557ee25a02c6a1678dba888')
d2l.DATA_HUB['lm-r2-s5100-state-v1'] = (
    d2l.LM_DATA_URL + 'lm-r2-s5100-state-v1.pt',
    'ffb9e95f1b0867206c5dd33db60d80a77e259f99')
d2l.DATA_HUB['lm-r2-evals-v1'] = (
    d2l.LM_DATA_URL + 'lm-r2-evals-v1.json.gz',
    '78ae3caa3a7573df21fceafdd84d2045cf2a57ea')
d2l.DATA_HUB['lm-ladder-logs-v1'] = (
    d2l.LM_DATA_URL + 'lm-ladder-logs-v1.jsonl.gz',
    '4a64125165abd77b05dcd0e2c5e53b485547ec67')
d2l.DATA_HUB['lm-ladder-config-v1'] = (
    d2l.LM_DATA_URL + 'lm-ladder-config-v1.json',
    '28ba56be7ef0e427ae982096fdac0fff1aa3b38d')
d2l.DATA_HUB['lm-r2-inst-t2-v1'] = (
    d2l.LM_DATA_URL + 'lm-r2-inst-t2-v1.json.gz',
    '3296eb030f2cd5c5d3d725293d7b0839f61c100c')
```

```{.python .input #reference-model-lib-artifacts}
%%tab pytorch
#@save
d2l.PRETRAIN_ARTIFACTS['reference-run-log'] = (
    'lm-r2-train-log-v1', 'jsonl')
d2l.PRETRAIN_ARTIFACTS['reference-evals'] = (
    'lm-r2-evals-v1', 'json-table')
d2l.PRETRAIN_ARTIFACTS['t1-evals'] = ('lm-r2t1-evals-v1', 'json-table')
d2l.PRETRAIN_ARTIFACTS['ladder-logs'] = ('lm-ladder-logs-v1', 'jsonl')
d2l.PRETRAIN_ARTIFACTS['ladder-evals'] = (
    'lm-ladder-evals-v1', 'json-table')
d2l.PRETRAIN_ARTIFACTS['t1-runs'] = ('lm-r2t1-log-v1', 'jsonl')
d2l.PRETRAIN_ARTIFACTS['instability-t1'] = (
    'lm-r2-inst-t1-v1', 'json-table')
d2l.PRETRAIN_ARTIFACTS['instability-t2'] = (
    'lm-r2-inst-t2-v1', 'json-table')
d2l.PRETRAIN_ARTIFACTS['spike-replay'] = ('lm-r2-spike-v1', 'json-table')
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
d2l.PRETRAIN_JOBS['reference-evals'] = ('reference_evals', lambda: dict(
    steps=d2l.REFERENCE_STEPS,
    tasks=('hellaswag', 'arc_easy', 'piqa', 'sciq', 'winogrande')))
d2l.PRETRAIN_JOBS['t1-evals'] = ('reference_evals', lambda: dict(
    steps=d2l.T1_STEPS,
    tasks=('hellaswag', 'arc_easy', 'piqa', 'sciq', 'winogrande'),
    preset='t1'))
d2l.PRETRAIN_JOBS['ladder-evals'] = ('ladder_evals', lambda: dict(
    ladder=d2l._job_log('ladder'), log=d2l._job_log('reference-run'),
    tasks=('hellaswag', 'arc_easy', 'piqa', 'sciq', 'winogrande')))
d2l.PRETRAIN_JOBS['inst-t2'] = ('instability_arms', lambda: dict(
    cfg=d2l.r2_config('r2'), lrs=(6e-3,), variants=('none', 'qk_clip'),
    steps=500, tau=d2l.tau_rule(d2l._job_log('reference-run'), 300, 500)))
d2l.PRETRAIN_JOBS['inst-t1-none'] = (
    'instability_arms', lambda: d2l._inst_t1_job())
d2l.PRETRAIN_JOBS['inst-t1-qkclip'] = (
    'instability_arms', lambda: d2l._inst_t1_job())
d2l.PRETRAIN_JOBS['inst-t1-qknorm'] = (
    'instability_arms', lambda: d2l._inst_t1_job())
d2l.PRETRAIN_JOBS['inst-t1-softcap'] = (
    'instability_arms', lambda: d2l._inst_t1_job())
d2l.PRETRAIN_JOBS['spike-replay'] = ('spike_replay', lambda: dict(
    cfg=d2l.r2_config('t1', clip=None), bad_step=300, kind='random',
    replays=(('from 250, same batches', 250, (), None),
             ('from 250, skip 300', 250, (300,), None),
             ('from 50, same batches', 50, (), None),
             ('clip 1.0', 0, (), 1.0))))
```

## Presets, Fields and Peaks

```{.python .input #reference-model-lib-presets}
%%tab pytorch
#@save
REFERENCE_STEPS = (500, 1000, 2000, 3000, 4000, 4500, 4800, 5100, 5400, 5700,
                   6000)

#@save
T1_STEPS = (127, 254, 509, 763, 1017, 1144, 1221, 1297, 1373, 1450,
            1526)  # the same fractions of the t1 run's 1,526 steps

#@save
def _preset_saves(preset):
    """What a preset's run saves: bf16 checkpoints, pinned full states
    (branch points and the end of the stable phase), rolling states, and
    the branches (from_step, decay_steps) that finish 10N and 5N budgets;
    for r2 also its logit-statistics interval (10 steps, so that the
    change points of 'Who Moves First' resolve to 10 steps)."""
    ladder = {'ladder-25m': ((816, 144), (408, 72)),
              'ladder-54m': ((1760, 310), (880, 155)),
              'ladder-88m': ((2857, 504), (1428, 252))}
    _, steps, _, decay, _, _ = d2l.R2_SCHEDULES[preset]
    if preset == 'r2':
        return dict(ckpt_steps=REFERENCE_STEPS, state_steps=(1275, 2550,
                                                             5100),
                    state_every=500, branches=((2550, 450), (1275, 225)),
                    stats_every=10)
    if preset in ladder:
        b = ladder[preset]
        return dict(ckpt_steps=(steps,), state_steps=tuple(sorted(
            f for f, _ in b)), state_every=500, branches=b)
    if preset == 't1':  # checkpoints for the stand-in while r2 is pending
        return dict(ckpt_steps=T1_STEPS, state_steps=(steps - decay,))
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
        tied = get('tie_word_embeddings')  # absent: Transformers' default
        return (L, d, H, get('num_key_value_heads') or H,
                get('head_dim') or d // H, get('intermediate_size'),
                get('vocab_size'), True if tied is None else bool(tied))
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
        self.gpu = (torch.cuda.get_device_name(self.device) if cuda
                    else 'cpu')
        self.out, self.rows, self.halt = cfg.out_dir, [], False
        self.grad_hist = collections.deque(maxlen=100)
        self._mem_states, self._expect = {}, None
        self.start = 0
        path = self._resume_path(resume)
        if path is not None:
            self.start = self._load_state(path)
        self.log_path = self._log_file(cfg.preset)
        if self.log_path and self.rank == 0:
            d2l._truncate_log(self.log_path, self.start)

    # -- segments: the main run, then each branch from its saved state --
    def segments(self):
        cfg = self.cfg
        if self.start < cfg.steps:
            self._expect = self.start + 1  # checked by the segment's 1st row
            yield _Segment(cfg.preset, self.start, cfg.steps, cfg.decay)
        for frm, dec in cfg.branches:
            name = f'b{frm + dec}'
            if self.halt or self._branch_done(name, frm + dec):
                continue
            self._load_state(self._state_source(frm))
            self.log_path = self._log_file(name)
            if self.log_path and self.rank == 0:
                d2l._truncate_log(self.log_path, frm)
            self._expect = frm + 1
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
        row['world'], row['gpu'], row['job_start'] = (self.world, self.gpu,
                                                      self.start)
        if self._expect is not None:  # the segment starts where it should
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
        row['time'] = time.time()  # wall clock after the step's I/O
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
def _val_fn(cfg, batch_size=None, inference=True):
    """Validation loss on the first cfg.val_tokens tokens of the validation
    shard, as windows of arch.T tokens with the training targets; under
    torch.no_grad instead of inference mode when not inference (for layers
    that refuse inference tensors)."""
    with d2l._quiet():
        shards = d2l.fwedu_shards('val')
    T, eos = cfg.arch.T, shards.eos_id
    n = cfg.val_tokens // T
    windows = np.stack([shards.read(i * T, T + 1) for i in range(n)])
    bs = batch_size or cfg.micro_batch

    def val(model):
        dev = next(model.parameters()).device
        total, count = 0.0, 0
        grad_off = torch.inference_mode() if inference else torch.no_grad()
        with grad_off, torch.autocast(
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
    run, resumable). While the reference run is pending its log is the t1
    run's seed 0; a pending result prints its label."""
    cfgs = [cfgs] if dataclasses.is_dataclass(cfgs) else list(cfgs)
    presets = tuple(sorted({c.preset for c in cfgs}))
    if presets not in _RUN_NAMES:
        raise ValueError(f'no committed run for presets {presets}')
    name = _RUN_NAMES[presets]
    config = dict(n_gpus=n_gpus, runs=[_cfg_json(c) for c in cfgs])
    log = d2l._pretrain_result(
        name, lambda: _launch_jobs(cfgs, n_gpus), config,
        fallback='t1-runs' if name == 'reference-run' else None)
    if log.attrs.get('label', '').startswith('T2 run pending'):
        label = log.attrs['label']
        log = log[log['seed'] == 0].reset_index(drop=True)
        log.attrs['label'] = label
    return d2l._show_label(log)

#@save
def reference_run_log():
    """The reference run's log: launch_pretrain(r2_config('r2'),
    n_gpus=4)."""
    return launch_pretrain(d2l.r2_config('r2'), n_gpus=4)

#@save
def _job_root():
    return os.path.abspath(os.path.join('..', 'data', 'lm', 'rerun', 'ch21'))

#@save
def _launch_jobs(cfgs, n_gpus, max_restarts=3, jobs=None, extra=None):
    """Run every config's job that is not done (D2L_PRETRAIN_ONLY limits
    this process to one preset or job); the merged logs (a column job names
    each run's folder) once all are done, else None. jobs: folder names
    (default <preset>-s<seed>); extra: more lm_pretrain.py arguments per
    config."""
    import subprocess
    import sys
    only = os.environ.get('D2L_PRETRAIN_ONLY')
    repo = os.path.dirname(os.path.dirname(os.path.abspath(d2l.__file__)))
    for cfg in cfgs:  # extract the manifests before any rank starts
        _train_shards(cfg)
    logs, world = [], max(1, min(n_gpus, torch.cuda.device_count()))
    jobs = jobs or [f'{c.preset}-s{c.seed}' for c in cfgs]
    extra = extra or [[] for _ in cfgs]
    for cfg, name, args in zip(cfgs, jobs, extra):
        job = os.path.join(_job_root(), name)
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
                   os.path.join(job, 'config.json'), '--resume', 'auto',
                   *map(str, args)]
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
            logs.append(_read_job_logs(job).assign(job=name))
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
def _reference_key(preset, step):
    if preset == 'r2':
        step = 6000 if step is None else step
        if step not in REFERENCE_STEPS:
            raise ValueError(f'step is one of {REFERENCE_STEPS}')
        return f'lm-r2-s{step}-v1'
    if preset == 't1':
        step = 1526 if step is None else step
        if step not in T1_STEPS:
            raise ValueError(f'step is one of {T1_STEPS}')
        return f'lm-r2t1-s{step}-v1'
    if preset in ('r2-16k-pi', 'r2-16k-yarn'):
        return f'lm-{preset}-v1'
    raise ValueError(f'unknown preset {preset!r}')

#@save
def _job_checkpoint(preset, step):
    """The bf16 checkpoint a campaign job folder holds for a committed
    key, read only by the checkpoint evaluations that follow the run before
    its upload (E1: reference-evals, t1-evals); None otherwise."""
    folders = {'r2': ('r2-s0', 6000), 't1': ('t1-s0', 1526)}
    evals = {'r2': 'reference-evals', 't1': 't1-evals'}
    if preset not in folders or not d2l._regenerates(evals[preset]):
        return None
    job, last = folders[preset]
    return os.path.join(_job_root(), job, 'ckpt',
                        f's{last if step is None else step}.safetensors')

#@save
def load_reference_model(step=None, dtype=torch.bfloat16, device='cpu',
                         preset='r2'):
    """A committed checkpoint of the reference run (preset 'r2', steps
    REFERENCE_STEPS, default 6000), of the t1 run ('t1', steps T1_STEPS,
    default 1526), or of a 16K arm ('r2-16k-pi', 'r2-16k-yarn'), as a
    PackedGPT in eval mode. Its tokenizer is d2l.r2_tokenizer()."""
    key = _reference_key(preset, step)
    local = os.path.join(d2l._rerun_dir(), d2l._next_key(key)
                         + '.safetensors')
    job = _job_checkpoint(preset, step)
    if d2l._regenerating() and os.path.exists(local):
        path = local
    elif job and os.path.exists(job):
        path = job  # the evaluations right after the run (E1)
    elif key in d2l.DATA_HUB:
        with d2l._quiet():
            path = d2l.download_keyed(key)
    else:
        raise KeyError('reference checkpoints not committed yet')
    return _load_bf16(path, dtype, device)

#@save
def _reference_pending():
    """Whether the reference run's results are pending: a regeneration
    run, or the campaign declared pending."""
    return bool(d2l._regenerating()) or (
        'reference-run' in d2l.PRETRAIN_PENDING)

#@save
T1_STAND_IN = ('T2 run pending; T1 checkpoints at the same fractions of '
               'training')

#@save
def load_reference_models(steps, dtype=torch.bfloat16, device='cpu'):
    """{step: model} of the reference run's checkpoints at steps (of
    REFERENCE_STEPS), each as load_reference_model gives it; while the run
    is pending, the t1 run's checkpoints at the same fractions of training
    (T1_STEPS), under a printed label."""
    if _reference_key('r2', steps[0]) in d2l.DATA_HUB or not (
            _reference_pending()):
        return {s: load_reference_model(s, dtype, device) for s in steps}
    print(f'[{T1_STAND_IN}]')
    t1 = dict(zip(REFERENCE_STEPS, T1_STEPS))
    return {s: load_reference_model(t1[s], dtype, device, preset='t1')
            for s in steps}

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

## The Section's Measurements

Validation loss and bits per byte, the run-log dashboard, the checkpoint
evaluations (E1), the instability arms and the spike protocol: the drivers
behind the page's one-line calls. Each decides between computing and
loading through `d2l._pretrain_result` (`data-path-lib.md`).

```{.python .input #reference-model-lib-val-bpb}
%%tab pytorch
#@save
_VAL_COUNTS = {}  # (T, tokens): (scored targets, the UTF-8 bytes they spell)

#@save
def val_loss(model, cfg, tokens=None):
    """Validation loss (nats per scored target) of model on the first tokens
    (default cfg.val_tokens) of the validation shard, in windows of
    cfg.arch.T tokens with the training targets; on the GPU if there is one
    (the model moves there)."""
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    c = dataclasses.replace(cfg, val_tokens=tokens or cfg.val_tokens)
    return _val_fn(c)(model.to(dev))

#@save
def val_bpb(loss, cfg, tokens=None):
    """Bits per UTF-8 byte of a val_loss(model, cfg, tokens) value (or an
    array of them): loss x scored targets / (ln 2 x their bytes)."""
    T, n = cfg.arch.T, tokens or cfg.val_tokens
    if (T, n) not in _VAL_COUNTS:
        with d2l._quiet():
            shards = d2l.fwedu_shards('val')
            lengths = d2l.token_byte_lengths(d2l.r2_tokenizer())
        W = np.stack([shards.read(i * T, T + 1) for i in range(n // T)])
        scored = W[:, :-1] != shards.eos_id
        _VAL_COUNTS[(T, n)] = (int(scored.sum()), int(lengths[
            W[:, 1:][scored].astype(np.int64)].sum()))
    count, n_bytes = _VAL_COUNTS[(T, n)]
    return loss * count / (math.log(2) * n_bytes)

#@save
def _job_log(name):
    """The log or result a later job of the campaign reads: the committed
    result of regeneration name if registered, else the file its
    regeneration wrote (before the upload)."""
    artifact = d2l._RERUN_ARTIFACT[name]
    key, kind = d2l.PRETRAIN_ARTIFACTS[artifact]
    if key in d2l.DATA_HUB:
        return d2l.load_artifact(artifact)
    path = os.path.join(d2l._rerun_dir(), key + d2l._ARTIFACT_EXT[kind])
    if not os.path.exists(path):
        raise FileNotFoundError(f'{name} has not run yet (no {path})')
    return d2l._read_result(path, kind)

#@save
def _as_tuple(cls, result, tables, defaults=None):
    """A computed, stored or pending result as the namedtuple cls; a pending
    result (an empty table) gives empty tables (with the columns tables
    names, a dict, so that the page's code runs on them) and the defaults
    (else None) for the other fields."""
    label = d2l._label_of(result)
    if isinstance(result, pd.DataFrame):
        result = None
    fields = {f: getattr(result, f, None) for f in cls._fields if f != 'label'}
    for f, v in (defaults or {}).items():
        if fields[f] is None:
            fields[f] = v
    for f in tables:
        if fields[f] is None:
            cols = tables[f] if isinstance(tables, dict) else []
            fields[f] = pd.DataFrame(columns=cols)
    return cls(**fields, label=label)
```

```{.python .input #reference-model-lib-dashboard}
%%tab pytorch
#@save
_PANEL_LABELS = {'loss': 'train loss', 'grad_norm': 'grad norm\n(pre-clip)',
                 'rms': 'update RMS\n(units of lr)',
                 'max_logit': 'largest\nattn. logit', 'mfu': 'MFU'}

#@save
def plot_run_log(log, cols, marks=(), figsize=(6, 7)):
    """Stacked panels of a run's log against steps (its main run, not the
    branches): 'loss', 'grad_norm' (log scale), 'rms' (update RMS per module
    kind), 'max_logit' (largest over layers), 'mfu', or any column; marks
    are steps drawn as dashed lines (phase boundaries)."""
    main = log[log['run'] == log['run'].iloc[0]]
    d2l.use_svg_display()
    fig, axes = d2l.plt.subplots(len(cols), 1, sharex=True, figsize=figsize)
    for ax, col in zip(np.atleast_1d(axes), cols):
        if col == 'rms':
            kinds = sorted(c for c in main if c.startswith('rms_'))
            sub = main.dropna(subset=kinds, how='all')
            for k in kinds:
                ax.plot(sub['step'], sub[k], lw=0.8, label=k[4:])
            ax.legend(ncol=5, fontsize=6, loc='upper right')
        else:
            y = (main.filter(like='max_logit_').max(axis=1)
                 if col == 'max_logit' else main[col])
            ok = y.notna()
            ax.plot(main['step'][ok], y[ok], lw=0.6)
        if col == 'grad_norm':
            ax.set_yscale('log')
            d2l.set_log_ticks(ax.yaxis)
        for m in marks:
            ax.axvline(m, ls='--', color='black', lw=0.8)
        ax.set_ylabel(_PANEL_LABELS.get(col, col), fontsize=8)
    np.atleast_1d(axes)[-1].set_xlabel('step')
    if d2l._label_of(log):  # a stand-in says so
        fig.suptitle(f'[{d2l._label_of(log)}]', fontsize=8)
    fig.align_ylabels()

#@save
def plot_traces(traces, cols, by, logy=(), steps=None, figsize=(5, 4)):
    """Stacked panels of per-step traces against step, one line per group
    of the columns by (named by their values): cols as in plot_run_log
    ('loss', 'grad_norm', 'max_logit') or any column; logy: the columns on
    a log scale; steps: the (first, last) step shown."""
    d2l.use_svg_display()
    fig, axes = d2l.plt.subplots(len(cols), 1, sharex=True, figsize=figsize,
                                 squeeze=False)
    axes = axes[:, 0]
    for key, g in traces.groupby(list(np.atleast_1d(by)), sort=False):
        if steps is not None:
            g = g[g['step'].between(*steps)]
        name = ' '.join(f'{k:g}' if isinstance(k, float) else str(k)
                        for k in np.atleast_1d(key))
        for ax, col in zip(axes, cols):
            ok = g[col].notna()
            ax.plot(g['step'][ok], g[col][ok], lw=0.8, label=name)
    for ax, col in zip(axes, cols):
        ax.set_ylabel(_PANEL_LABELS.get(col, col), fontsize=8)
        if col in logy:
            ax.set_yscale('log')
            d2l.set_log_ticks(ax.yaxis)
    if len(traces):
        axes[0].legend(fontsize=7)
    axes[-1].set_xlabel('step')
    fig.align_ylabels()

#@save
def plot_ladder_runs(*logs, figsize=(5, 3)):
    """Validation loss against tokens of every job in the logs (column
    job): the main run as a line, the ends of its branches as dots in the
    same colour."""
    d2l.use_svg_display()
    d2l.set_figsize(figsize)
    ax = d2l.plt.gca()
    jobs = [g for log in logs if 'job' in log for _, g in log.groupby('job')]
    for i, g in enumerate(jobs):
        first = g['run'].iloc[0]  # the main run's rows come first
        main = g[g['run'] == first].dropna(subset=['val_loss'])
        ends = g[g['run'] != first].groupby('run').last()
        ax.plot(main['tokens'], main['val_loss'], color=f'C{i}',
                label=g['job'].iloc[0].rsplit('-s', 1)[0])
        ax.plot(ends['tokens'], ends['val_loss'], 'o', color=f'C{i}')
    ax.set_xscale('log')
    d2l.set_log_ticks(ax.xaxis)
    ax.set_xlabel('tokens')
    ax.set_ylabel('validation loss')
    if jobs:
        ax.legend(fontsize=7)
```

```{.python .input #reference-model-lib-event}
%%tab pytorch
#@save
GradientEvent = collections.namedtuple('GradientEvent', [
    'step', 'hour', 'ratio', 'mild', 'source', 'at', 'max_logit', 'layer',
    'dloss', 'nonfinite', 'label'])

#@save
def gradient_event(log, mild=3, delta=0.3, window=100, after=200):
    """The largest gradient event of a run's log: the step whose pre-clip
    norm is largest against the median of the window steps before it, its
    hour into the run and that ratio. Read there (source: the run), or, if
    the ratio is below mild, at the pre-registered fallback (source: the
    unclipped arm of instability_arms at ten times the peak rate, at its
    first step more than delta above the baseline's loss): the largest
    attention logit and its layer at the nearest probe, and the loss change
    over the next `after` steps against the `after` before it (nonfinite:
    the first non-finite step, if any)."""
    run = log[log['run'] == log['run'].iloc[0]].set_index('step')
    g = run['grad_norm']
    ratio = g / g.rolling(window).median().shift()
    step = int(ratio.idxmax())
    head = dict(step=step, hour=(run['time'].loc[step] - run['time'].iloc[0])
                / 3600, ratio=float(ratio[step]),
                mild=bool(ratio[step] < mild), label=d2l._label_of(log))
    if not head['mild']:
        return GradientEvent(**head, source=f'{log["run"].iloc[0]} run',
                             at=step, **_read_event(run, step, after))
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):  # its label, once
        arms = d2l.instability_arms(  # the page's ten-x call
            d2l.r2_config('r2'), lrs=(6e-3,), variants=('none', 'qk_clip'),
            steps=500, tau=d2l.tau_rule(log, 300, 500))
    if arms.label and arms.label != head['label']:
        print(f'[{arms.label}]')
    t = arms.traces.set_index('step')
    arm = t[t['variant'] == 'none']
    if arm.empty:
        return GradientEvent(**head, source='unclipped arm (pending)',
                             at=None, max_logit=math.nan, layer=None,
                             dloss=math.nan, nonfinite=None)
    gap = arm['loss'] - t.loc[t['variant'] == 'baseline', 'loss']
    over = gap.index[~(gap <= delta)]  # non-finite, or above the margin
    at = int(over[0] if len(over) else gap.idxmax())
    source = f'unclipped arm at lr {arm["lr"].iloc[0]:g}'
    return GradientEvent(**head, source=source, at=at,
                         **_read_event(arm, at, after))

#@save
def _read_event(run, at, after):
    """max_logit, layer, dloss and nonfinite of gradient_event at step at
    of run (indexed by step)."""
    logits = run.filter(like='max_logit_').dropna(how='all')
    out = dict(max_logit=math.nan, layer=None, nonfinite=None)
    if len(logits):
        top = logits.loc[logits.index[np.abs(logits.index - at).argmin()]]
        top = top.astype(float)
        out.update(max_logit=float(top.max()),
                   layer=int(top.idxmax().rsplit('_', 1)[1]) + 1)
    loss = run['loss'].astype(float)
    bad = loss.index[(loss.index >= at) & ~np.isfinite(loss.to_numpy())]
    if len(bad):
        out['nonfinite'] = int(bad[0])
    out['dloss'] = float(loss.loc[at + 1:at + after].mean()
                         - loss.loc[at - after + 1:at].mean())
    return out
```

```{.python .input #reference-model-lib-evals}
%%tab pytorch
#@save
CheckpointEvals = collections.namedtuple(
    'CheckpointEvals', ['table', 'items', 'domains', 'chance', 'label'])

#@save
LadderEvals = collections.namedtuple('LadderEvals',
                                     ['points', 'items', 'label'])

#@save
FULL_VAL_TOKENS = 2 ** 23  # the whole validation shard, in 4,096-token windows

#@save
def domain_texts():
    """The per-domain validation texts (FineWeb-Edu file 012 by URL-host
    class, and the held-out book sections): doc_id, domain, text."""
    return d2l.load_artifact('val-domains')

#@save
def _suite_chance(suite):
    """Accuracy of guessing uniformly, per task: the mean of 1 / choices."""
    out = {}
    for t in suite.tasks:
        k = [len(it.get('choices') or it.get('prompts') or ())
             for it in suite.items(t.id)]
        out[t.id] = float(np.mean([1 / n for n in k if n]))
    return out

#@save
def _score_point(model, suite, tokenizer, cfg, tokens, texts=None):
    """Validation loss and BPB, per-item suite scores and per-domain BPB of
    one model: (row, items, domain rows)."""
    loss = val_loss(model, cfg, tokens)
    with d2l._quiet():
        res = suite.run(model, tokenizer)
    items = res.per_item[['task', 'item_id', 'score']]
    row = dict(val_loss=loss, bpb=val_bpb(loss, cfg, tokens),
               **items.groupby('task')['score'].mean())
    doms = []
    for dom, g in ([] if texts is None else texts.groupby('domain')):
        with d2l._quiet():
            doms.append(dict(domain=dom, bpb=d2l.bits_per_byte(
                model, tokenizer, list(g['text']), ctx_len=cfg.arch.T)))
    return row, items, doms

#@save
def _eval_setup(tasks):
    if not torch.cuda.is_available():
        raise RuntimeError('the checkpoint evaluations run on a GPU')
    with d2l._quiet():
        return d2l.r2_tokenizer(), d2l.EvalSuite(list(tasks), name='r2',
                                                 batch_size=32)

#@save
def _reference_evals(steps, tasks, preset='r2'):
    tok, suite = _eval_setup(tasks)
    cfg, texts = d2l.r2_config(preset), domain_texts()
    rows, items, doms = [], [], []
    for s in steps:
        model = load_reference_model(step=s, device='cuda', preset=preset)
        row, it, dm = _score_point(model, suite, tok, cfg, cfg.val_tokens,
                                   texts)
        rows.append(dict(step=s, **row))
        items.append(it.assign(step=s))
        doms += [dict(step=s, **d) for d in dm]
        del model
        d2l.lm_free()
    return CheckpointEvals(pd.DataFrame(rows), pd.concat(items, ignore_index=
                           True), pd.DataFrame(doms), _suite_chance(suite), '')

#@save
def reference_evals(steps, tasks, preset='r2'):
    """The reference run's checkpoints at steps (or the t1 run's, preset
    't1'), scored on validation loss and bits per byte (2^21 tokens), on the
    suite tasks (full item sets, per-item scores kept) and per domain:
    CheckpointEvals(table, items, domains, chance, label); table has one
    row per step. While the reference run is pending: the t1 run's
    checkpoints at the same fractions of training, its steps written as the
    reference run's, under a printed label."""
    config = dict(steps=list(steps), tasks=list(tasks))
    name = 'reference-evals' if preset == 'r2' else f'{preset}-evals'
    res = d2l._pretrain_result(name, lambda: _reference_evals(
        steps, tasks, preset), config,
        fallback='t1-evals' if preset == 'r2' else None)
    tables = dict(table=['step', 'val_loss', 'bpb', *tasks],
                  items=['task', 'item_id', 'score', 'step'],
                  domains=['step', 'domain', 'bpb'])
    res = _as_tuple(CheckpointEvals, res, tables, dict(
        chance=dict.fromkeys(tasks, math.nan)))
    if res.label.startswith('T2 run pending'):  # t1 steps -> r2 steps
        r2_step = dict(zip(T1_STEPS, REFERENCE_STEPS))
        renamed = {f: getattr(res, f) for f in tables}
        for f, df in renamed.items():
            if 'step' in df:
                renamed[f] = df.assign(step=df['step'].map(r2_step))
        res = res._replace(label=T1_STAND_IN, **renamed)
    return d2l._show_label(res)

#@save
def _ladder_points(logs):
    """(job, run, step, finished, source) of every point ladder_evals
    scores: per job, the finished main run and branches (bf16 checkpoints)
    and the plateau states its branches start from."""
    pts = []
    for job in sorted(set(logs['job'])):
        cfg = d2l.r2_config(job.rsplit('-s', 1)[0])
        folder = os.path.join(_job_root(), job)
        pts.append((job, cfg.preset, cfg.steps, True, os.path.join(
            folder, 'ckpt', f's{cfg.steps}.safetensors')))
        for frm, dec in cfg.branches:
            end = frm + dec
            pts.append((job, f'b{end}', end, True, os.path.join(
                folder, 'ckpt', f'b{end}-s{end}.safetensors')))
            pts.append((job, cfg.preset, frm, False, _state_path(folder,
                                                                 frm)))
    return pts

#@save
def _ladder_evals(logs, tasks):
    tok, suite = _eval_setup(tasks)
    rows, items = [], []
    for job, run, step, finished, src in _ladder_points(logs):
        cfg = d2l.r2_config(job.rsplit('-s', 1)[0])
        if finished:
            model = _load_bf16(src, device='cuda')
        else:
            model = d2l.PackedGPT.from_arch(cfg.arch)
            model.load_state_dict(torch.load(src, map_location='cpu',
                                             weights_only=False)['model'])
            model = model.to('cuda', torch.bfloat16).eval()
        P = sum(p.numel() for p in model.parameters())
        tokens = step * cfg.global_batch * cfg.arch.T
        row, it, _ = _score_point(model, suite, tok, cfg, FULL_VAL_TOKENS)
        rows.append(dict(size=cfg.preset, P=P, run=run, step=step,
                         tokens=tokens, tokens_per_param=tokens / P,
                         finished=finished, **row))
        items.append(it.assign(size=cfg.preset, run=run, step=step))
        del model
        d2l.lm_free()
    return LadderEvals(pd.DataFrame(rows), pd.concat(items, ignore_index=
                       True), '')

#@save
def ladder_evals(ladder, log, tasks):
    """Full-validation loss (2^23 tokens) and per-item suite scores at the
    finished points (main run and its two branches) and the plateau points
    (the branches' starting states) of every ladder size and of the
    reference run (log): LadderEvals(points, items, label)."""
    logs = pd.concat([ladder, log], ignore_index=True)
    config = dict(jobs=sorted(set(logs['job'])) if 'job' in logs else [],
                  tasks=list(tasks))
    res = d2l._pretrain_result('ladder-evals', lambda: _ladder_evals(
        logs, tasks), config)
    return d2l._show_label(_as_tuple(LadderEvals, res, dict(
        points=['size', 'P', 'run', 'step', 'tokens', 'tokens_per_param',
                'finished', 'val_loss', 'bpb', *tasks],
        items=['task', 'item_id', 'score', 'size', 'run', 'step'])))
```

```{.python .input #reference-model-lib-instability}
%%tab pytorch
#@save
InstabilityRuns = collections.namedtuple(
    'InstabilityRuns', ['table', 'traces', 'tau', 'label'])

#@save
_INST_PARTS = {'none': 'inst-t1-none', 'qk_clip': 'inst-t1-qkclip',
               'qk_norm': 'inst-t1-qknorm', 'softcap': 'inst-t1-softcap'}

#@save
INST_SOFTCAP = 50.0  # Gemma 2's attention-logit cap

#@save
_INST_TABLES = dict(  # the columns of an InstabilityRuns' tables
    table=['variant', 'lr', 'seed', 'final_loss', 'max_logit',
           'first_nonfinite', 'val_loss', 'steps'],
    traces=['step', 'loss', 'grad_norm', 'max_logit', 'variant', 'lr',
            'seed'])

#@save
def _qk_clip_hook(tau):
    """A pretrain hook: QK-clip at tau after every optimizer step."""
    def hook(step, model, optimizer, batch, row):
        d2l.qk_clip_(model, d2l.attn_logit_stats(model, batch), tau)
    return hook

#@save
def _inst_config(cfg, lr, seed, steps, **kw):
    """An arm: cfg's warmup, then a constant learning rate (no decay), logit
    statistics every 10 steps, validation at the end only, nothing saved."""
    args = dict(steps=steps, peak_lr=lr, decay=0, seed=seed, stats_every=10,
                probe_every=0, val_every=0, ckpt_steps=(), state_steps=(),
                state_every=0, branches=(), out_dir=None)
    return dataclasses.replace(cfg, **dict(args, **kw))

#@save
def _arm_summary(log, variant, lr, seed, tail=20):
    """One arm's row: final loss (mean of the last tail steps; inf after a
    non-finite loss), largest logit, first non-finite step, final
    validation loss."""
    loss = log['loss'].to_numpy(float)
    bad = np.flatnonzero(~np.isfinite(loss))
    first = int(log['step'].iloc[bad[0]]) if len(bad) else None
    logits = log.filter(like='max_logit_').to_numpy(float)
    val = log['val_loss'].dropna() if 'val_loss' in log else pd.Series([])
    return dict(variant=variant, lr=lr, seed=seed,
                final_loss=(math.inf if first is not None
                            else float(loss[-tail:].mean())),
                max_logit=float(np.nanmax(logits)) if np.isfinite(
                    logits).any() else math.nan, first_nonfinite=first,
                val_loss=float(val.iloc[-1]) if len(val) else math.nan,
                steps=int(log['step'].max()))

#@save
def _arm_trace(log, variant, lr, seed):
    """Per step: loss, pre-clip norm, the largest logit per layer and over
    layers (at the statistics steps)."""
    t = log[['step', 'loss', 'grad_norm']
            + [c for c in log if c.startswith('max_logit_')]].copy()
    t['max_logit'] = log.filter(like='max_logit_').max(axis=1)
    return t.assign(variant=variant, lr=lr, seed=seed)

#@save
def _inst_run(cfg, variant, lr, seed, steps, tau, shards, val_fn, device):
    """One T1 arm in this process."""
    arch = cfg.arch._replace(qk_norm=variant == 'qk_norm' or cfg.arch.qk_norm)
    extra = (dict(softcap=INST_SOFTCAP, backend='math')
             if variant == 'softcap' else {})
    c = _inst_config(cfg, lr, seed, steps, arch=arch)
    torch.manual_seed(seed)
    model = d2l.PackedGPT.from_arch(arch, **extra).to(device)
    stream = d2l.PackedStream(shards, arch.T, c.global_batch,
                              seed=c.data_seed)
    hooks = [_qk_clip_hook(tau)] if variant == 'qk_clip' else []
    return d2l.pretrain(model, stream, c, val_fn=val_fn, hooks=hooks)

#@save
def _inst_t1(cfg, lrs, variant, seeds, steps, tau, device=None):
    """The T1 grid of one variant, on one GPU (device: tests only)."""
    if device is None and not torch.cuda.is_available():
        raise RuntimeError('the instability arms train on a GPU')
    device, shards = torch.device(device or 'cuda'), _train_shards(cfg)
    val = _val_fn(dataclasses.replace(cfg, val_tokens=2 ** 20))
    rows, traces = [], []
    for lr in lrs:
        for seed in seeds:
            log = _inst_run(cfg, variant, lr, seed, steps, tau, shards, val,
                            device)
            rows.append(_arm_summary(log, variant, lr, seed))
            traces.append(_arm_trace(log, variant, lr, seed))
            torch._dynamo.reset()
            d2l.lm_free()
    return InstabilityRuns(pd.DataFrame(rows), pd.concat(
        traces, ignore_index=True), tau, '')

#@save
def _inst_t2(cfg, lrs, variants, steps, tau):
    """The T2 arms as torchrun jobs on every GPU, and the reference run's
    first steps as the baseline (same recipe at its own learning rate)."""
    if 'softcap' in variants:
        raise ValueError('the torchrun job has no softcap variant')
    base = _job_log('reference-run')
    base = base[(base['run'] == cfg.preset) & (base['step'] <= steps)]
    cfgs, jobs, extra, arms = [], [], [], []
    for lr in lrs:
        for v in variants:
            arch = cfg.arch._replace(qk_norm=v == 'qk_norm')
            cfgs.append(_inst_config(cfg, lr, cfg.seed, steps, arch=arch,
                                     val_every=steps))
            jobs.append(f'inst-t2-{v.replace("_", "")}-{lr:g}')
            extra.append(['--allow-nonfinite'] + (
                ['--qk-clip', tau] if v == 'qk_clip' else []))
            arms.append((v, lr))
    logs = _launch_jobs(cfgs, torch.cuda.device_count(), jobs=jobs,
                        extra=extra)
    if logs is None:
        return None
    rows = [_arm_summary(base, 'baseline', cfg.peak_lr, cfg.seed)]
    traces = [_arm_trace(base, 'baseline', cfg.peak_lr, cfg.seed)]
    for (v, lr), job in zip(arms, jobs):
        g = logs[logs['job'] == job].reset_index(drop=True)
        rows.append(_arm_summary(g, v, lr, cfg.seed))
        traces.append(_arm_trace(g, v, lr, cfg.seed))
    return InstabilityRuns(pd.DataFrame(rows), pd.concat(
        traces, ignore_index=True), tau, '')

#@save
def _merge_arms(parts):
    runs = list(parts.values())
    return InstabilityRuns(
        pd.concat([r.table for r in runs], ignore_index=True),
        pd.concat([r.traces for r in runs], ignore_index=True), runs[0].tau,
        '')

#@save
T1_STAND_IN_LR = 1.6e-2  # the T1 grid's rate shown while inst-t2 is pending

#@save
def _t1_stand_in(res):
    """The T1 grid's arms shown in place of a pending inst-t2: the t1 peak
    rate as the baseline and the none and qk_clip arms at T1_STAND_IN_LR
    (seed 0), under the pending label."""
    t1_lr = d2l.r2_config('t1').peak_lr
    out = []
    for df in (res.table, res.traces):
        df = df[df['seed'] == 0]
        base = df[(df['variant'] == 'none') & (df['lr'] == t1_lr)]
        arms = df[df['variant'].isin(['none', 'qk_clip'])
                  & (df['lr'] == T1_STAND_IN_LR)]
        out.append(pd.concat([base.assign(variant='baseline'), arms],
                             ignore_index=True))
    return InstabilityRuns(*out, res.tau, d2l._label_of(res))

#@save
def instability_arms(cfg, lrs, variants, seeds=(0,), steps=400, tau=None):
    """Short runs of cfg at each learning rate, per variant ('none';
    'qk_clip', QK-clip at tau after every step; 'qk_norm'; 'softcap',
    attention logits capped at INST_SOFTCAP) and seed, warmup as cfg and a
    constant rate after it: InstabilityRuns(table, traces, tau, label), one
    table row per arm (final loss = mean of the last 20 steps, inf after a
    non-finite loss), traces per step. The t1 preset trains here on one GPU
    (a regeneration name per variant); the reference preset launches
    torchrun jobs (inst-t2) and adds the reference run's first steps as the
    baseline."""
    if 'qk_clip' in variants and tau is None:  # tau comes from tau_rule
        raise ValueError('the qk_clip variant needs tau')
    config = dict(cfg=_cfg_json(cfg), lrs=list(lrs), variants=list(variants),
                  seeds=list(seeds), steps=steps, tau=tau)
    if cfg.preset == 't1':
        parts = {_INST_PARTS[v]: (lambda v=v: _inst_t1(
            cfg, lrs, v, seeds, steps, tau), dict(config, variants=[v]))
            for v in variants}
        res = d2l._pretrain_merged(None, parts, _merge_arms, config)
        return d2l._show_label(_as_tuple(InstabilityRuns, res, _INST_TABLES))
    res = d2l._pretrain_result('inst-t2', lambda: _inst_t2(
        cfg, lrs, variants, steps, tau), config, fallback='instability-t1')
    res = _as_tuple(InstabilityRuns, res, _INST_TABLES)
    if res.label.startswith('T2 run pending') and len(res.table):
        res = _t1_stand_in(res)
    return d2l._show_label(res)
```

```{.python .input #reference-model-lib-spike}
%%tab pytorch
#@save
SpikeReplay = collections.namedtuple('SpikeReplay',
                                     ['table', 'traces', 'floor', 'label'])

#@save
class _BadBatchStream(d2l.PackedStream):
    """A PackedStream whose batch of log step bad_step (1-based, as the log
    counts) is make_bad_batch(kind); skipping that step removes it, and
    bad_step=None gives the plain stream."""
    def __init__(self, shards, seq_len, global_batch, bad_step, kind, vocab,
                 **kw):
        super().__init__(shards, seq_len, global_batch, **kw)
        self.bad_step, self.kind, self.vocab = bad_step, kind, vocab

    def batch(self, step):
        b = super().batch(step)
        if (self.bad_step is not None
                and self.physical_index(step) == self.bad_step - 1):
            b = d2l.make_bad_batch(b, self.kind, self.vocab,
                                   seed=self.bad_step)
        return b

#@save
def _spike(cfg, bad_step, kind, replays, steps=400, window=(200, 400),
           device=None):
    """The spiked run (full states kept at the replays' starts), each
    replay (from a state of the spiked run, skip set in log steps, clip),
    and two same-state replays without the bad batch (the floor); device:
    tests only."""
    import shutil
    if device is None and not torch.cuda.is_available():
        raise RuntimeError('the spike protocol trains on a GPU')
    device = device or 'cuda'
    shards, folder = _train_shards(cfg), os.path.join(_job_root(), 'spike')
    shutil.rmtree(folder, ignore_errors=True)  # this driver's own states
    starts = tuple(sorted({s for _, s, _, _ in replays if s}))
    base = _inst_config(cfg, cfg.peak_lr, cfg.seed, steps, stats_every=0)

    def run(start=0, skip=(), clip=None, bad=bad_step, save=False):
        c = dataclasses.replace(base, clip=base.clip if clip is None
                                else clip, state_steps=starts if save
                                else (), out_dir=folder if save else None)
        torch.manual_seed(cfg.seed)
        model = d2l.PackedGPT.from_arch(cfg.arch).to(device)
        stream = _BadBatchStream(shards, cfg.arch.T, cfg.global_batch, bad,
                                 kind, cfg.arch.vocab, seed=cfg.data_seed,
                                 skip=[s - 1 for s in skip])
        state = None
        if start:
            state = torch.load(_state_path(folder, start), map_location='cpu',
                               weights_only=False)
            state['stream'] = dict(state['stream'], skip=[s - 1 for s in skip])
        log = d2l.pretrain(model, stream, c, resume=state)
        torch._dynamo.reset()
        return log.set_index('step')
    spiked = run(save=True)
    logs = {'spiked run': spiked}
    for label, start, skip, clip in replays:
        logs[label] = run(start, skip, clip)
    floor = (run(starts[-1], bad=None)['loss']
             - run(starts[-1], bad=None)['loss']).abs().max()
    rows, traces = [], []
    for label, g in logs.items():
        near = g.loc[bad_step - 5:bad_step + 5, 'grad_norm']
        after = (g['loss'] - spiked['loss']).loc[bad_step + 1:steps]
        rows.append(dict(run=label, loss_at_bad=g['loss'].get(bad_step),
                         peak_grad_norm=near.max(),
                         max_dloss=math.nan if g is spiked
                         else after.abs().max()))
        traces.append(g.loc[window[0]:window[1], ['loss', 'grad_norm']]
                      .reset_index().assign(run=label))
    return SpikeReplay(pd.DataFrame(rows), pd.concat(traces, ignore_index=
                       True), float(floor), '')

#@save
def spike_replay(cfg, bad_step=300, kind='random', replays=()):
    """The spike protocol at cfg: a bad batch (make_bad_batch(kind)) at log
    step bad_step of a run with a constant rate after warmup, then each
    replay (label, from_step, skipped steps, clip or None) from the spiked
    run's state at from_step (0: from the start), and the replay floor (two
    same-state replays without the bad batch): SpikeReplay(table, traces,
    floor, label)."""
    config = dict(cfg=_cfg_json(cfg), bad_step=bad_step, kind=kind,
                  replays=d2l._jsonable(replays))
    res = d2l._pretrain_result('spike-replay', lambda: _spike(
        cfg, bad_step, kind, replays), config)
    return d2l._show_label(_as_tuple(SpikeReplay, res, dict(
        table=['run', 'loss_at_bad', 'peak_grad_norm', 'max_dloss'],
        traces=['step', 'loss', 'grad_norm', 'run']), dict(floor=math.nan)))
```

```{.python .input #reference-model-lib-inst-jobs}
%%tab pytorch
#@save
def _inst_t1_job():
    """The page's T1 grid call (cell 25) for tools/lm_regen.py, with tau
    from the t1 run's seed-0 log by the page's rule."""
    t1 = _job_log('t1-runs')
    return dict(cfg=d2l.r2_config('t1', clip=None),
                lrs=(1e-3, 2e-3, 4e-3, 8e-3, 1.6e-2, 3.2e-2),
                variants=('none', 'qk_clip', 'qk_norm', 'softcap'),
                seeds=(0, 1), steps=400,
                tau=d2l.tau_rule(t1[t1['seed'] == 0], 76, 400))
```
