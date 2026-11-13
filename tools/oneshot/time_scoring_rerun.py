"""Time one batch of every job of 18.4's regeneration run and pick its cut.

`chapter_lm-pretrained/scoring.md` regenerates its artifacts under
`D2L_RERUN=1`, and such a run must finish within 30 minutes measured (ruling
2026-11-12). Its cuts apply in a declared order, `D2L_RERUN_CUT=k` applying
the first k:

  1. the reasoning arm on 100 subset items instead of 200;
  2. GSM8K samples capped at 320 new tokens instead of 384;
  3. no fp32 batch-size re-scorings.

This script picks k from wall clock alone, before the run, so that no
accuracy has been seen when the cut is chosen. It executes the notebook's own
setup cells (subsets, prompts, samplers, arms, formats), then times one batch
of 50 prompts of every generation job and 50 items of every re-scoring,
loading each model as the run does. It extrapolates to the full run, adds the
normal build's live part (`--live-min`, the spec's 10 min unless measured),
and prints the smallest k whose estimate fits `--ceiling-min`. It prints
times and memory only: no reply is parsed or scored.

    cd <worktree>
    CUDA_VISIBLE_DEVICES=<gpu> HF_HOME=$HOME/.cache/hf-lm \\
        .venv-lm/bin/python tools/oneshot/time_scoring_rerun.py
    # --dry: build the prompts and print the plan without loading a model
    # --only arc-generations: probe and estimate only the jobs of the
    #   artifacts that D2L_RERUN will name (default 1, all four)

A generation batch is modelled as prefill plus a fixed time per new token:
the probe also runs each batch with one new token (the prefill), and the
estimate scales the decode part to the length the probed batch reached
("measured"), or to the full token budget ("at cap", an upper bound).
"""
import argparse
import contextlib
import io
import math
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
SOURCE = os.path.join(ROOT, 'chapter_lm-pretrained', 'scoring.md')
CELLS = ['scoring-imports', 'scoring-imports-extras', 'scoring-data-hub',
         'scoring-download', 'scoring-setup', 'scoring-artifacts',
         'scoring-data', 'scoring-format-print', 'scoring-format-merges',
         'scoring-gsm8k-prompts', 'scoring-arc-arms', 'scoring-grid-def']
BATCH = 50  # prompts per generation batch, as the notebook's cells
N_SAMPLES, N_GSM, N_ARC = 8, 100, 500  # GSM8K samples and problems; ARC
CUTS = {0: (200, 384, True), 1: (100, 384, True), 2: (100, 320, True),
        3: (100, 320, False)}  # k: reasoning items, GSM8K budget, fp32 runs


def cell(source, cid):
    """The code of the cell with id cid, without its %%tab line."""
    m = re.search(r'```\{\.python \.input #' + re.escape(cid) +
                  r'\}\n(.*?)\n```', source, re.S)
    if m is None:
        raise KeyError(cid)
    return re.sub(r'^%%tab [^\n]*\n', '', m[1])


def notebook_state():
    """Run the notebook's setup cells, as a normal build would, and return
    their namespace (CUT 0, RERUN off; their printed output is dropped)."""
    for var in ('D2L_RERUN', 'D2L_RERUN_CUT'):
        os.environ.pop(var, None)
    with open(SOURCE) as f:
        source = f.read()
    ns = {'__name__': '__scoring__'}
    os.chdir(os.path.dirname(SOURCE))  # the cells read ../data/lm
    with contextlib.redirect_stdout(io.StringIO()):
        for cid in CELLS:
            exec(compile(cell(source, cid), cid, 'exec'), ns)
        runs = cell(source, 'scoring-batch-size')  # its list of runs
        exec(runs[:runs.index("\nif 'arc-scores' in RERUN:")], ns)
    return ns


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--dry', action='store_true',
                   help='print the plan without loading a model')
    p.add_argument('--live-min', type=float, default=10.0,
                   help='minutes of the normal build inside the run')
    p.add_argument('--ceiling-min', type=float, default=30.0)
    p.add_argument('--only', default='1', metavar='D2L_RERUN',
                   help='the value D2L_RERUN will have: 1 (all) or a '
                   'comma-separated list of artifacts; only their jobs '
                   'are probed and estimated')
    args = p.parse_args(argv)
    ns = notebook_state()
    torch, d2l = ns['torch'], ns['d2l']
    names = args.only.split(',')
    only = set(ns['ARTIFACTS'] if names == ['1'] else names)
    if not only <= set(ns['ARTIFACTS']):
        p.error(f'--only names one of {ns["ARTIFACTS"]}, or 1')
    print('regenerating:', ', '.join(sorted(only)) + (
        ' (bpb is part of the live run)' if 'bpb' in only else ''))
    PAIR, ARMS, GSM_SAMPLER = ns['PAIR'], ns['ARMS'], ns['GSM_SAMPLER']
    gsm_budget = GSM_SAMPLER['max_new_tokens']
    reason_budget = ARMS['reasoning'][2]['max_new_tokens']
    arc_models = list(dict.fromkeys(ARMS['direct'][0] + ARMS['reasoning'][0]))
    if 'gsm8k-samples' in only:
        print(f'GSM8K pool: {PAIR}, {N_SAMPLES} x '
              f'{math.ceil(N_GSM / BATCH)} batches per model, budget '
              f'{gsm_budget}')
    for arm, (models, n, s) in ARMS.items() if 'arc-generations' in only \
            else ():
        print(f'ARC {arm}: {models}, {math.ceil(n / BATCH)} batches per '
              f'model, budget {s["max_new_tokens"]}')
    if 'arc-scores' in only:
        print('batch-size runs:', [(m, str(dt), bs) for m, dt, bs, _ in
                                   ns['BATCH_RUNS']])
    if args.dry:
        return
    if not torch.cuda.is_available():
        print('WARNING: no GPU; these times say nothing about the run')
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (
        lambda: None)

    def timed(f):
        sync()
        tic = time.time()
        out = f()
        sync()
        return out, time.time() - tic

    def load(m, **kw):
        (model, tok), t = timed(lambda: d2l.load_hf(m, **kw))
        return model, tok, t

    def probe_generate(model, tok, prompts, settings):
        """(prefill s, s per new token, tokens reached) of one batch."""
        one = dict(settings, max_new_tokens=1)
        _, t1 = timed(lambda: d2l.generate(model, tok, prompts,
                                           batch_size=BATCH, **one))
        out, t = timed(lambda: d2l.generate(model, tok, prompts,
                                            batch_size=BATCH, **settings))
        reached = max(len(g.token_ids) for g in out)
        return t1, max(t - t1, 0.0) / max(reached - 1, 1), reached

    def batch_time(probe, budget, cap=False):
        t1, step, reached = probe
        return t1 + step * ((budget if cap else min(reached, budget)) - 1)

    rows, loads = [], {'gsm': 0.0, 'arc': 0.0}
    gsm, arc = {}, {}
    for m in PAIR if 'gsm8k-samples' in only else ():
        model, tok, t = load(m)
        loads['gsm'] += t
        gsm[m] = probe_generate(model, tok, ns['GSM_PROMPTS'][:BATCH],
                                dict(GSM_SAMPLER, seed=0))
        rows.append((f'GSM8K {m}', gsm[m], ns['peak']()))
        del model
        ns['free']()
    for m in arc_models if 'arc-generations' in only else ():
        model, tok, t = load(m)
        loads['arc'] += t
        for arm, (models, n, settings) in ARMS.items():
            if m in models:
                arc[arm, m] = probe_generate(
                    model, tok, ns['ARC_CHAT'][:BATCH],
                    dict(settings, chat=True))
                rows.append((f'ARC {arm} {m}', arc[arm, m], ns['peak']()))
        del model
        ns['free']()
    rescore = []  # (fp32?, seconds for the full run incl. its load)
    for m, dtype, size, col in (ns['BATCH_RUNS'] if 'arc-scores' in only
                                else ()):
        model, tok, t = load(m, dtype=dtype)
        for fmt in ('cloze', 'letter0'):
            prompts, choices = zip(*ns['FORMATS'][fmt][:BATCH])
            _, s = timed(lambda: d2l.score_choices(
                model, tok, prompts, choices, batch_size=size))
            t += s * N_ARC / BATCH
        rescore.append((dtype == torch.float32, t))
        print(f're-score {m} {dtype} batch {size}: {t / 60:.1f} min '
              f'(est.), {ns["peak"]()}')
        del model
        ns['free']()
    for name, (t1, step, reached), mem in rows:
        print(f'{name}: prefill {t1:.1f} s, {1e3 * step:.0f} ms/token, '
              f'reached {reached} tokens, {mem}')

    def estimate(k, cap):
        n_reason, budget, fp32 = CUTS[k]
        g = sum(N_SAMPLES * math.ceil(N_GSM / BATCH) *
                batch_time(gsm[m], budget, cap) for m in gsm)
        a = sum(math.ceil((N_ARC if arm == 'direct' else n_reason) / BATCH)
                * batch_time(probe, ARMS[arm][2]['max_new_tokens'], cap)
                for (arm, m), probe in arc.items())
        r = sum(t for is_fp32, t in rescore if fp32 or not is_fp32)
        return (args.live_min * 60 + loads['gsm'] + loads['arc'] + g + a +
                r) / 60
    print(f'\nestimated run (live part {args.live_min:g} min included):')
    choice = None
    for k in CUTS:
        mid, top = estimate(k, False), estimate(k, True)
        print(f'  D2L_RERUN_CUT={k}: {mid:.1f} min measured lengths, '
              f'{top:.1f} min at the token caps')
        if choice is None and mid <= args.ceiling_min:
            choice = k
    if choice is None:
        print(f'no cut level fits {args.ceiling_min:g} min: escalate')
    else:
        print(f'choose D2L_RERUN_CUT={choice} (the fewest cuts that fit '
              f'{args.ceiling_min:g} min)')
    print(f'(budgets probed: GSM8K {gsm_budget}, reasoning {reason_budget})')


if __name__ == '__main__':
    sys.exit(main())
