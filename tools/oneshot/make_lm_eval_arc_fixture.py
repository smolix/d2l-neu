#!/usr/bin/env python3
"""Make lm-eval-reference-arc-v1.json, the harness reference of sec_lm_scoring.

lm-evaluation-harness 0.4.13 scores Qwen3-0.6B-Base (at its HF_REVISIONS pin)
on the first 200 ARC-Challenge test items with its stock `arc_challenge` task
(`Question: {q}\\nAnswer:`, each option after a space, `acc` = argmax of the
summed log-likelihood, `acc_norm` = argmax of the log-likelihood divided by the
option's length in characters), batch size 1. This script runs the harness in
a throwaway environment, reads its per-sample log and writes the fixture that
cell `scoring-harness-fixture` compares d2l.score_choices against.

The fixture holds ids and numbers only, no question or option text (ARC is
CC BY-SA 4.0; the notebook downloads the text itself): per item the ARC id, the
harness's doc index, the gold index, per option the summed log-likelihood and
the option's length in characters and in UTF-8 bytes, the two predictions and
the two harness scores. The harness's is-greedy flag is not kept: in 0.4.13 it
came out True for all 799 options, multi-token ones with log-likelihoods near
-25 included, so it carries no information. The header records the
harness, Transformers, torch and datasets versions, the model revision, the
dataset commit the harness resolved (asserted equal to the notebook's pin), the
device, dtype and the exact command.

The throwaway environment (not one of the book's venvs; delete it afterwards):

    E=~/.cache/d2l-throwaway/lmeval-0.4.13
    uv venv -p 3.12 $E/.venv
    VIRTUAL_ENV=$E/.venv uv pip install --index-url \\
        https://download.pytorch.org/whl/cpu torch==2.11.0
    VIRTUAL_ENV=$E/.venv uv pip install -c <(printf 'torch==2.11.0\\n'\\
        'transformers==5.22.2\\n') 'lm-eval[hf]==0.4.13'

Run (stdlib only on this side; the harness runs in the throwaway env):

    HF_HOME=/home/smola/.cache/hf-lm CUDA_VISIBLE_DEVICES= python3 \\
        tools/oneshot/make_lm_eval_arc_fixture.py --python $E/.venv/bin/python

fp32 on the CPU takes a few minutes. `--dtype bfloat16 --device cuda` makes the
bf16 counterpart on a GPU host (a separate file and key, see --out).
"""
import argparse
import datetime
import glob
import hashlib
import json
import math
import os
import pathlib
import re
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODEL = 'Qwen/Qwen3-0.6B-Base'
DATASET = 'allenai/ai2_arc'
HARNESS = '0.4.13'

VERSIONS = r'''
import json, sys, datasets, lm_eval, torch, transformers
from huggingface_hub import HfApi
print(json.dumps({
    'lm_eval': lm_eval.__version__, 'transformers': transformers.__version__,
    'torch': torch.__version__, 'datasets': datasets.__version__,
    'python': sys.version.split()[0],
    'dataset_main_sha': HfApi().dataset_info(%r).sha}))
''' % DATASET


def pins():
    """Model and dataset pins as written in the chapter source."""
    hf = (ROOT / 'chapter_lm-pretrained/hf-models.md').read_text()
    model = re.search(r"'%s': '([0-9a-f]{40})'" % re.escape(MODEL), hf)[1]
    sc = (ROOT / 'chapter_lm-pretrained/scoring.md').read_text()
    data = re.search(r"datasets/allenai/ai2_arc/resolve/'\s*'([0-9a-f]{40})/",
                     sc)[1]
    return model, data


def env_versions(python, env):
    out = subprocess.run([python, '-c', VERSIONS], env=env, check=True,
                         capture_output=True, text=True).stdout
    return json.loads(out.strip().splitlines()[-1])


def convert(sample):
    """One harness sample -> one fixture row (ids and numbers only)."""
    doc = sample['doc']
    texts = list(doc['choices']['text'])
    gold = int(sample['target'])
    ll = [float(r[0]) for r in sample['filtered_resps']]
    n_chars = [len(t) for t in texts]
    n_bytes = [len(t.encode('utf-8')) for t in texts]
    # The harness's argmax (numpy, first index on ties).
    pred = max(range(len(ll)), key=lambda i: (ll[i], -i))
    norm = [l / c for l, c in zip(ll, n_chars)]
    pred_norm = max(range(len(norm)), key=lambda i: (norm[i], -i))
    acc, acc_norm = float(sample['acc']), float(sample['acc_norm'])
    assert acc == float(pred == gold), (doc['id'], acc, pred, gold)
    assert acc_norm == float(pred_norm == gold), (doc['id'], acc_norm)
    return {'id': doc['id'], 'doc_id': int(sample['doc_id']), 'gold': gold,
            'labels': list(doc['choices']['label']), 'll': ll,
            'n_chars': n_chars, 'n_bytes': n_bytes,
            'pred_acc': pred, 'pred_acc_norm': pred_norm,
            'acc': acc, 'acc_norm': acc_norm}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--python', required=True,
                    help="the throwaway env's python (has lm-eval 0.4.13)")
    ap.add_argument('--dtype', default='float32',
                    choices=['float32', 'bfloat16'])
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--limit', type=int, default=200)
    ap.add_argument('--threads', type=int, default=16,
                    help='torch CPU threads (OMP_NUM_THREADS)')
    ap.add_argument('--out', default=None, help='default: data/lm/authoring/'
                    'lm-eval-reference-arc-v1.json (fp32), '
                    '...-arc-bf16-v1.json (bf16)')
    args = ap.parse_args()
    out = pathlib.Path(args.out or ROOT / 'data/lm/authoring' / (
        'lm-eval-reference-arc-v1.json' if args.dtype == 'float32'
        else 'lm-eval-reference-arc-bf16-v1.json'))
    model_rev, data_rev = pins()
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads),
               MKL_NUM_THREADS=str(args.threads), TOKENIZERS_PARALLELISM='0',
               HF_DATASETS_TRUST_REMOTE_CODE='0')
    if args.device == 'cpu':
        env['CUDA_VISIBLE_DEVICES'] = ''
    if 'HF_HOME' not in env:
        sys.exit('set HF_HOME (the local Hugging Face cache) first')

    before = env_versions(args.python, env)
    assert before['lm_eval'] == HARNESS, before
    assert before['dataset_main_sha'] == data_rev, (
        f"{DATASET} main is {before['dataset_main_sha']}, the notebook pins "
        f'{data_rev}: the harness would read other data')

    model_args = f'pretrained={MODEL},revision={model_rev},dtype={args.dtype}'
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [args.python, '-m', 'lm_eval', 'run', '--model', 'hf',
               '--model_args', model_args, '--device', args.device,
               '--tasks', 'arc_challenge',
               '--limit', str(args.limit), '--batch_size', '1',
               '--log_samples', '--output_path', tmp, '--seed', '0']
        print('running:', ' '.join(cmd[1:]), flush=True)
        t0 = datetime.datetime.now(datetime.timezone.utc)
        subprocess.run(cmd, env=env, check=True)
        seconds = (datetime.datetime.now(datetime.timezone.utc)
                   - t0).total_seconds()
        samples = glob.glob(f'{tmp}/**/samples_arc_challenge_*.jsonl',
                            recursive=True)
        results = glob.glob(f'{tmp}/**/results_*.json', recursive=True)
        assert len(samples) == 1 and len(results) == 1, (samples, results)
        rows = [json.loads(line) for line in open(samples[0])]
        res = json.load(open(results[0]))
    after = env_versions(args.python, env)
    assert after['dataset_main_sha'] == data_rev, after

    rows = sorted((convert(s) for s in rows), key=lambda r: r['doc_id'])
    assert [r['doc_id'] for r in rows] == list(range(args.limit))
    for r in rows:
        assert all(math.isfinite(x) for x in r['ll']), r['id']
    agg = res['results']['arc_challenge']
    fixture = {
        'name': out.stem, 'made': t0.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'producer': 'tools/oneshot/make_lm_eval_arc_fixture.py',
        'task': 'arc_challenge (lm-evaluation-harness stock task)',
        'prompt': 'Question: {question}\\nAnswer:',
        'continuation': "' ' + option text",
        'acc': 'argmax of summed log-likelihood',
        'acc_norm': 'argmax of log-likelihood / len(option text) '
                    '(characters, without the leading space)',
        'model': MODEL, 'model_revision': model_rev,
        'dataset': DATASET, 'dataset_config': 'ARC-Challenge',
        'split': 'test', 'items': f'first {args.limit} in dataset order',
        'dataset_revision': data_rev,
        'dataset_revision_note': 'the harness reads main; main resolved to '
                                 'this commit before and after the run',
        'dtype': args.dtype, 'device': args.device, 'batch_size': 1,
        'versions': {k: v for k, v in before.items()
                     if k != 'dataset_main_sha'},
        'command': ' '.join(cmd[2:cmd.index('--output_path')]),
        'seconds': round(seconds, 1),
        'summary': {'acc': agg['acc,none'], 'acc_norm': agg['acc_norm,none'],
                    'n': len(rows)},
        'licence': 'scores of an Apache-2.0 model; no ARC text',
        'not_kept': "the harness's is_greedy flag (True for every option in "
                    'this run, so uninformative)',
        'items_data': rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(fixture, indent=1, ensure_ascii=False) + '\n'
    out.write_text(text)
    sha1 = hashlib.sha1(text.encode()).hexdigest()
    print(f'wrote {out} ({len(text.encode())} bytes, sha1 {sha1}); '
          f"acc {agg['acc,none']:.4f}, acc_norm {agg['acc_norm,none']:.4f}, "
          f'{seconds:.0f} s')


if __name__ == '__main__':
    main()
