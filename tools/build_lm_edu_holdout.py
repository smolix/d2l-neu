#!/usr/bin/env python3
"""Build lm-edu-holdout-v1: the rows of the released FineWeb-Edu annotations
(HuggingFaceFW/fineweb-edu-llama3-annotations at the pinned revision) that the
released FineWeb-Edu classifier did NOT train on.

Why (sec_lm_quality_filtering, review finding F2): the released classifier
was trained on these annotations, so its scores on a random annotation row are
in-sample. 20.2 draws its validation and test rows from the classifier's
hold-out instead, so the paired comparison with our heads is out-of-sample
for both.

How the hold-out is reproduced:
  * The classifier's training code (github.com/huggingface/cosmopedia,
    classification/train_edu_bert.py) clips the score to 0..5 and calls
    datasets' train_test_split(train_size=0.9, seed=42,
    stratify_by_column='score') on the annotation dataset as first uploaded.
  * The first upload (revision c130af06…, 2024-06-03 12:17) has 468,668 rows:
    ceil(0.1 * 468,668) = 46,867, the hold-out size on the classifier's card.
    The pinned release (72df4c92…, same day 18:29) is that upload minus 1,244
    rows (1,243 of score 5 and one of score 0), in the same order (aligned
    here on url and dump).
  * The split below is datasets 2.19.1's stratified split
    (src/datasets/utils/stratify.py, Apache-2.0), copied without change of
    behaviour. Its per-class supports equal the card's (5,694 / 26,512 /
    10,322 / 3,407 / 807 / 125); 46,744 of its rows survive in the release.

Verification (--verify, one GPU, about 2 minutes per run; 2026-11-16): the
released classifier in bf16 on the 46,744 rows gives the card's confusion
matrix (rows 0-4) up to a summed absolute difference of 90 counts, spread in
both directions (bf16 rounding near the .5 boundaries); two control samples
of the same per-class size drawn outside the hold-out differ by 1,686 and
1,790. In fp32 the difference is 304: the card's numbers were computed in
bf16.

Output: data/lm/rerun/lm-edu-holdout-v1.json.gz (deterministic bytes): the
sorted release row indices and the provenance above. Upload with
make upload-data and register its sha1 next to _edu_annot_sample.

    .venv-lm/bin/python tools/build_lm_edu_holdout.py [--verify]
"""
import argparse
import gzip
import io
import json
import os
import sys
from math import floor

import numpy as np
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = 'HuggingFaceFW/fineweb-edu-llama3-annotations'
TRAINED_ON = 'c130af068a7e93e4dc1bdf4c8ed42c1c1b47396a'  # first upload
OUT = os.path.join(ROOT, 'data', 'lm', 'rerun', 'lm-edu-holdout-v1.json.gz')
CARD_SUPPORT = [5694, 26512, 10322, 3407, 807, 125]
CARD_CM = np.array([[2791, 2858, 45, 0, 0, 0], [919, 22343, 3180, 69, 1, 0],
                    [3, 3225, 6330, 757, 7, 0], [1, 66, 1473, 1694, 173, 0],
                    [0, 4, 98, 420, 283, 2], [0, 0, 18, 85, 21, 1]])


def approximate_mode(class_counts, n_draws, rng):
    """datasets 2.19.1 utils/stratify.py (from scikit-learn)."""
    continuous = n_draws * class_counts / class_counts.sum()
    floored = np.floor(continuous)
    need_to_add = int(n_draws - floored.sum())
    if need_to_add > 0:
        remainder = continuous - floored
        values = np.sort(np.unique(remainder))[::-1]
        for value in values:
            (inds,) = np.where(remainder == value)
            add_now = min(len(inds), need_to_add)
            inds = rng.choice(inds, size=add_now, replace=False)
            floored[inds] += 1
            need_to_add -= add_now
            if need_to_add == 0:
                break
    return floored.astype(np.int64)


def stratified_split(y, n_train, n_test, rng):
    """The first split of datasets 2.19.1's
    stratified_shuffle_split_generate_indices."""
    _, y_indices = np.unique(y, return_inverse=True)
    class_counts = np.bincount(y_indices)
    class_indices = np.split(np.argsort(y_indices, kind='mergesort'),
                             np.cumsum(class_counts)[:-1])
    n_i = approximate_mode(class_counts, n_train, rng)
    t_i = approximate_mode(class_counts - n_i, n_test, rng)
    train, test = [], []
    for i in range(len(class_counts)):
        permutation = rng.permutation(class_counts[i])
        perm = class_indices[i].take(permutation, mode='clip')
        train.extend(perm[:n_i[i]])
        test.extend(perm[n_i[i]:n_i[i] + t_i[i]])
    return rng.permutation(train), rng.permutation(test)


def columns(open_file):
    """score and url|dump of the eight files, in order."""
    scores, keys = [], []
    for k in range(8):
        with open_file(f'data/train-{k:05d}-of-00008.parquet') as f:
            t = pq.ParquetFile(f).read(columns=['score', 'metadata'])
        scores.append(t.column('score').to_numpy())
        m = t.column('metadata').combine_chunks()
        keys += [f'{u}|{d}' for u, d in zip(m.field('url').to_pylist(),
                                            m.field('dump').to_pylist())]
    return np.concatenate(scores), keys


def build():
    import huggingface_hub
    sys.path.insert(0, ROOT)
    from d2l import torch as d2l
    release = d2l.LM_DATA_SOURCES['edu-annotations'][2]
    fs = huggingface_hub.HfFileSystem()
    s_old, k_old = columns(lambda p: fs.open(
        f'datasets/{REPO}@{TRAINED_ON}/{p}', 'rb'))
    s_new, k_new = columns(lambda p: open(huggingface_hub.hf_hub_download(
        REPO, p, revision=release, repo_type='dataset'), 'rb'))
    assert len(s_new) == d2l._EDU_ANNOT_ROWS
    y = np.clip(s_old.astype(int), 0, 5)
    n_train = floor(0.9 * len(y))
    _, test = stratified_split(y, n_train, len(y) - n_train,
                               np.random.default_rng(42))
    support = np.bincount(y[test], minlength=6).tolist()
    assert len(test) == 46_867 and support == CARD_SUPPORT, support
    old2new, j = np.full(len(y), -1), 0  # the release keeps the order
    for i in range(len(y)):
        if j < len(k_new) and k_old[i] == k_new[j] and s_old[i] == s_new[j]:
            old2new[i], j = j, j + 1
    assert j == len(k_new), 'the release is not the first upload minus rows'
    removed = np.bincount(y[old2new < 0], minlength=6).tolist()
    rows = np.sort(old2new[test][old2new[test] >= 0])
    print(f'hold-out {len(test):,} rows, supports {support}; release = first '
          f'upload minus {int((old2new < 0).sum()):,} rows (by score '
          f'{removed}); {len(rows):,} hold-out rows in the release')
    doc = dict(
        dataset=REPO, release_revision=release, trained_on_revision=TRAINED_ON,
        split=("train_test_split(train_size=0.9, seed=42, stratify_by_column="
               "'score'), datasets 2.19.1, score clipped to 0..5"),
        source='github.com/huggingface/cosmopedia classification/'
               'train_edu_bert.py',
        holdout_size=len(test), holdout_support=support,
        removed_in_release_by_score=removed, n_rows=len(rows),
        rows=rows.tolist())
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode='wb', mtime=0) as g:
        g.write(json.dumps(doc, sort_keys=True).encode('utf-8'))
    with open(OUT, 'wb') as f:
        f.write(buf.getvalue())
    import hashlib
    print(f'wrote {OUT}, sha1 {hashlib.sha1(buf.getvalue()).hexdigest()}')
    return rows


def verify(rows):
    """The released classifier's confusion matrix on the hold-out (bf16)
    against the card's, rows 0-4 (the release keeps 2 of the 125 score-5
    hold-out rows)."""
    import huggingface_hub
    import torch
    from d2l import torch as d2l
    release = d2l.LM_DATA_SOURCES['edu-annotations'][2]
    texts, scores, start = [], [], 0
    for k in range(8):
        p = pq.ParquetFile(huggingface_hub.hf_hub_download(
            REPO, f'data/train-{k:05d}-of-00008.parquet', revision=release,
            repo_type='dataset'))
        n = p.metadata.num_rows
        sel = rows[(rows >= start) & (rows < start + n)] - start
        t = p.read(columns=['text', 'score']).take(sel)
        texts += t.column('text').to_pylist()
        scores.append(t.column('score').to_numpy())
        start += n
    y = np.clip(np.concatenate(scores), 0, 5)
    s = d2l.released_edu_scores(texts, batch=128, dtype=torch.bfloat16)
    cm = np.zeros((6, 6), int)
    np.add.at(cm, (y, np.round(s).clip(0, 5).astype(int)), 1)
    print('confusion matrix (rows: Llama score 0-4, columns: prediction):')
    print(cm[:5])
    print('summed |card - ours| over rows 0-4:',
          int(np.abs(CARD_CM[:5] - cm[:5]).sum()))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--verify', action='store_true',
                    help='score the hold-out with the released classifier')
    rows = build()
    if ap.parse_args().verify:
        verify(rows)
