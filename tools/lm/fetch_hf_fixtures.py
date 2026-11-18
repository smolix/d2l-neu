"""Fetch the Hugging Face metadata fixtures of the Language Models tests.

Writes tools/lm/fixtures/hf_models/<org>/<repo>/config.json for each model of
CONFIGS and header.json (tensor name -> dtype and shape, read from the
safetensors headers by HTTP range requests, no tensor data) for each model of
HEADERS, all at the commits pinned in d2l.HF_REVISIONS. The files are
Apache-2.0 metadata of the model repositories; README.md in the fixture
directory carries the attribution. Re-run only when a pin moves:

    .venv-lm/bin/python tools/lm/fetch_hf_fixtures.py
"""
import json
import pathlib
import re
import struct

import requests

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / 'tools/lm/fixtures/hf_models'
CONFIGS = ['Qwen/Qwen3.5-2B', 'Qwen/Qwen3-0.6B', 'Qwen/Qwen3-0.6B-Base',
           'Qwen/Qwen3-1.7B-Base', 'Qwen/Qwen3-4B', 'Qwen/Qwen3-8B',
           'google/gemma-4-E2B-it']
HEADERS = ['Qwen/Qwen3.5-2B', 'google/gemma-4-E2B-it']
HEADERS_UA = {'User-Agent': 'd2l-fixtures'}


def revisions():
    """HF_REVISIONS as written in the chapter source (no d2l import)."""
    src = (ROOT / 'chapter_lm-pretrained/hf-models.md').read_text()
    block = src[src.index('HF_REVISIONS = {'):]
    block = block[:block.index('}') + 1]
    return dict(re.findall(r"'([^']+)': '([0-9a-f]{40})'", block))


def get(model_id, rev, name, byte_range=None):
    headers = dict(HEADERS_UA)
    if byte_range:
        headers['Range'] = 'bytes=%d-%d' % byte_range
    r = requests.get(f'https://huggingface.co/{model_id}/resolve/{rev}/{name}',
                     headers=headers, timeout=60)
    r.raise_for_status()
    return r.content


def header(model_id, rev):
    files = [s['rfilename'] for s in requests.get(
        f'https://huggingface.co/api/models/{model_id}/revision/{rev}',
        headers=HEADERS_UA, timeout=60).json()['siblings']]
    if 'model.safetensors.index.json' in files:
        index = json.loads(get(model_id, rev, 'model.safetensors.index.json'))
        shards = sorted(set(index['weight_map'].values()))
    else:
        shards = ['model.safetensors']
    tensors = {}
    for shard in shards:
        n = struct.unpack('<Q', get(model_id, rev, shard, (0, 7)))[0]
        h = json.loads(get(model_id, rev, shard, (8, 8 + n - 1)))
        h.pop('__metadata__', None)
        tensors.update({k: {'dtype': v['dtype'], 'shape': v['shape']}
                        for k, v in h.items()})
    return {'model_id': model_id, 'revision': rev, 'shards': shards,
            'tensors': dict(sorted(tensors.items()))}


def main():
    revs = revisions()
    for model_id in CONFIGS:
        d = OUT / model_id
        d.mkdir(parents=True, exist_ok=True)
        (d / 'config.json').write_bytes(get(model_id, revs[model_id],
                                            'config.json'))
    for model_id in HEADERS:
        h = header(model_id, revs[model_id])
        (OUT / model_id / 'header.json').write_text(
            json.dumps(h, indent=0, separators=(',', ':')) + '\n')
        print(model_id, len(h['tensors']), 'tensors')


if __name__ == '__main__':
    main()
