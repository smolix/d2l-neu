"""Time one batch of every job of 18.4's regeneration run before it runs.

`chapter_lm-pretrained/scoring.md` regenerates its measured results under
`D2L_RERUN` (`1` for all four, or a comma-separated subset of `bpb`,
`gsm8k-samples`, `arc-generations`, `arc-scores`). A regeneration run has a
30-minute target and a 60-minute hard limit, the scheduler's per-notebook
timeout (ruling 2026-11-13). The section declares no cuts: its settings are
the ones the registered artifacts record. A run that would exceed the hard
limit is split into per-artifact runs (`D2L_RERUN=<one artifact>`), each of
which repeats the normal build's live part.

This script executes the notebook's own setup cells (subsets, prompts,
samplers, arms with their batch sizes, formats), then times one batch of
every generation job and 50 items of every re-scoring, loading each model
as the run does. It extrapolates to the full run, adds the normal build's
live part (`--live-min`), and checks the estimate against `--target-min`
and `--limit-min`: within the target, run as one; over the target but with
its at-cap estimate within the limit, run as one (the log says it is over
the target); otherwise it prints the per-artifact estimates and whether
each fits the limit, or says to escalate. It prints times, memory and the
share of the probed replies cut at the token budget (whether the budget
binds), never an accuracy.

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
         'scoring-setup', 'scoring-data', 'scoring-format-print',
         'scoring-format-merges', 'scoring-gsm8k-prompts', 'scoring-arc-arms',
         'scoring-grid-def']
GSM_BATCH = 50  # prompts per GSM8K batch, as the notebook's cell
N_SAMPLES, N_GSM, N_ARC = 8, 100, 500  # GSM8K samples and problems; ARC
PROBE = 50  # items per probed re-scoring


def cell(source, cid):
    """The code of the cell with id cid, without its %%tab line."""
    m = re.search(r'```\{\.python \.input #' + re.escape(cid) +
                  r'\}\n(.*?)\n```', source, re.S)
    if m is None:
        raise KeyError(cid)
    return re.sub(r'^%%tab [^\n]*\n', '', m[1])


def notebook_state():
    """Run the notebook's setup cells, as a normal build would, and return
    their namespace (RERUN off; their printed output is dropped)."""
    os.environ.pop('D2L_RERUN', None)
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
    p.add_argument('--target-min', type=float, default=30.0)
    p.add_argument('--limit-min', type=float, default=60.0)
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
        p.error(f'--only names one of {tuple(ns["ARTIFACTS"])}, or 1')
    print('regenerating:', ', '.join(sorted(only)) + (
        ' (bpb is part of the live run)' if 'bpb' in only else ''))
    PAIR, ARMS, GSM_SAMPLER = ns['PAIR'], ns['ARMS'], ns['GSM_SAMPLER']
    gsm_budget = GSM_SAMPLER['max_new_tokens']
    arc_models = list(dict.fromkeys(ARMS['direct'][0] + ARMS['reasoning'][0]))
    if 'gsm8k-samples' in only:
        print(f'GSM8K pool: {PAIR}, {N_SAMPLES} x '
              f'{math.ceil(N_GSM / GSM_BATCH)} batches per model, budget '
              f'{gsm_budget}')
    for arm, (models, n, size, s) in ARMS.items() if 'arc-generations' in \
            only else ():
        print(f'ARC {arm}: {models}, {math.ceil(n / size)} batches of {size} '
              f'per model, budget {s["max_new_tokens"]}')
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

    def probe_generate(model, tok, prompts, settings, size):
        """(prefill s, s per new token, tokens reached, share cut at the
        budget) of one batch."""
        one = dict(settings, max_new_tokens=1)
        _, t1 = timed(lambda: d2l.generate(model, tok, prompts,
                                           batch_size=size, **one))
        out, t = timed(lambda: d2l.generate(model, tok, prompts,
                                            batch_size=size, **settings))
        reached = max(len(g.token_ids) for g in out)
        cut = sum(g.finish_reason == 'length' for g in out) / len(out)
        return t1, max(t - t1, 0.0) / max(reached - 1, 1), reached, cut

    def batch_time(probe, budget, cap=False):
        t1, step, reached, _ = probe
        return t1 + step * ((budget if cap else min(reached, budget)) - 1)

    rows, loads = [], {'gsm8k-samples': 0.0, 'arc-generations': 0.0}
    gsm, arc = {}, {}
    for m in PAIR if 'gsm8k-samples' in only else ():
        model, tok, t = load(m)
        loads['gsm8k-samples'] += t
        gsm[m] = probe_generate(model, tok, ns['GSM_PROMPTS'][:GSM_BATCH],
                                dict(GSM_SAMPLER, seed=0), GSM_BATCH)
        rows.append((f'GSM8K {m}', gsm[m], d2l.lm_peak()))
        del model
        d2l.lm_free()
    for m in arc_models if 'arc-generations' in only else ():
        model, tok, t = load(m)
        loads['arc-generations'] += t
        for arm, (models, n, size, settings) in ARMS.items():
            if m in models:
                arc[arm, m] = probe_generate(
                    model, tok, ns['ARC_CHAT'][:size],
                    dict(settings, chat=True), size)
                rows.append((f'ARC {arm} {m}', arc[arm, m], d2l.lm_peak()))
        del model
        d2l.lm_free()
    rescore = 0.0  # seconds for the full re-scorings, loads included
    for m, dtype, size, col in (ns['BATCH_RUNS'] if 'arc-scores' in only
                                else ()):
        model, tok, t = load(m, dtype=dtype)
        for fmt in ('cloze', 'letter0'):
            prompts, choices = zip(*ns['FORMATS'][fmt][:PROBE])
            _, s = timed(lambda: d2l.score_choices(
                model, tok, prompts, choices, batch_size=size))
            t += s * N_ARC / PROBE
        rescore += t
        print(f're-score {m} {dtype} batch {size}: {t / 60:.1f} min '
              f'(est.), {d2l.lm_peak()}')
        del model
        d2l.lm_free()
    for name, (t1, step, reached, cut), mem in rows:
        print(f'{name}: prefill {t1:.1f} s, {1e3 * step:.0f} ms/token, '
              f'reached {reached} tokens, {cut:.0%} of the batch cut at the '
              f'budget, {mem}')

    def minutes(cap):
        """Minutes per artifact, without the live part."""
        g = sum(N_SAMPLES * math.ceil(N_GSM / GSM_BATCH) *
                batch_time(gsm[m], gsm_budget, cap) for m in gsm)
        a = sum(math.ceil(ARMS[arm][1] / ARMS[arm][2]) *
                batch_time(probe, ARMS[arm][3]['max_new_tokens'], cap)
                for (arm, m), probe in arc.items())
        return {'bpb': 0.0,
                'gsm8k-samples': (loads['gsm8k-samples'] + g) / 60,
                'arc-generations': (loads['arc-generations'] + a) / 60,
                'arc-scores': rescore / 60}
    mid, top = minutes(False), minutes(True)
    total = [args.live_min + sum(e[k] for k in only) for e in (mid, top)]
    print(f'\nestimated run, live part ({args.live_min:g} min) included: '
          f'{total[0]:.1f} min at the measured lengths, {total[1]:.1f} min '
          f'at the token caps')
    if total[0] <= args.target_min:
        print(f'within the {args.target_min:g}-min target: run '
              f'D2L_RERUN={args.only} as one job')
    elif total[1] <= args.limit_min:
        print(f'WARNING: over the {args.target_min:g}-min target, within the '
              f'{args.limit_min:g}-min hard limit at the caps: run '
              f'D2L_RERUN={args.only} as one job; its log says it is over '
              'the target')
    else:
        print(f'over the {args.limit_min:g}-min hard limit as one job; '
              'per-artifact runs:')
        fits = True
        for k in sorted(only):
            t = [args.live_min + e[k] for e in (mid, top)]
            ok = t[1] <= args.limit_min
            fits &= ok
            print(f'  D2L_RERUN={k}: {t[0]:.1f} min measured, {t[1]:.1f} min '
                  f'at the caps' + ('' if ok else ' (over the limit)'))
        if not fits:
            print('an artifact alone does not fit the hard limit: escalate')
    for arm, (_, n, size, s) in ARMS.items() if arc else ():
        cut = [arc[arm, m][3] for m in ARMS[arm][0]]
        print(f'(ARC {arm}: budget {s["max_new_tokens"]}, probed batch cut '
              f'at the budget {min(cut):.0%} to {max(cut):.0%})')


if __name__ == '__main__':
    sys.exit(main())
