#!/usr/bin/env python3
"""Fixed inputs of sec_lm_d2l_checkpoint (18.2) and of assignment A1.

Three parts, each run on its own:

prompts (CPU, tokenizer only; this stage)
    data/lm/authoring/lm-qwen3-port-prompts-v1.json: the 32 x 64 token grid on
    which cells 13-24 compare the port with Transformers, the 28 round-trip
    texts that cell 19 generates from, and a 4,096-token prompt of
    contiguous book prose. Eight categories of four prompts, in this order
    (cell 21 uses the first prompt of each):
      english   book prose: first paragraph of at least 80 tokens of four
                sections
      python    the d2l library source (d2l/torch.py) from four definitions
      chinese   four short passages written for this fixture (below)
      latex     display equations of four math-appendix sections, joined
      arithmetic  lines `a + b = c`, operands by a fixed rule
      languages German, French, Spanish and Japanese translations of the
                opening of :numref:`sec_perf_model`, written for this fixture
      random    32 ids drawn uniformly from [0, 151643) with seed 18,
                then repeated (the induction pattern); ids only
      chat      the first exercise of four sections as a user message,
                rendered by the chat template (enable_thinking=False)
    Round-trip rule: for every category but `random` the stored text is
    tok.decode(ids) and tok(text).input_ids == ids holds; where a window
    [s, s + 64) fails it, s moves forward one token at a time. For `chat` the
    window runs over the message's tokens and the template's tokens wrap it.
    Book sources are read at COMMIT (a commit of main) with `git show`.

logit-ref (GPU, stage B2; never run on a CPU for the real model)
    data/lm/authoring/lm-qwen3-0p6b-logit-ref-v1.pt: per position of the grid,
    the top-100 logits of Transformers Qwen3-0.6B in fp64 (SDPA, batches of
    4) stored as fp32, their ids (int32), the logsumexp (fp32) and the input
    ids, with the model revision, Transformers and torch versions and the
    GPU name. A1's autograder reads it; the notebook does not.

a1-corrupt DIR (CPU or GPU; author only; DIR outside the repository)
    Three corrupted copies of Qwen3-0.6B's safetensors state dict for A1's
    `first_divergence` task, and DIR/answers.json with the (layer, capture
    point) each must be localized to. Never uploaded or committed.

    HF_HOME=/home/smola/.cache/hf-lm CUDA_VISIBLE_DEVICES= \\
        .venv-lm/bin/python tools/lm/make_port_fixtures.py prompts
    HF_HOME=... .venv-lm/bin/python tools/lm/make_port_fixtures.py logit-ref
    HF_HOME=... .venv-lm/bin/python tools/lm/make_port_fixtures.py \\
        a1-corrupt ~/a1-hidden

Each part prints the file's size and sha1 for its DATA_HUB entry.
"""
import argparse
import hashlib
import json
import pathlib
import re
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/lm'))
import book_text  # noqa: E402

MODEL = 'Qwen/Qwen3-0.6B'
COMMIT = 'b4d28000a49956d76ef1a8c2f5a0d7a135609efc'  # main, 2026-11-12
OUT = ROOT / 'data/lm/authoring'
T, LONG_T, RANDOM_SEED, VOCAB_REGULAR = 64, 4096, 18, 151643

ENGLISH = ['chapter_introduction/index.md',
           'chapter_linear-regression/linear-regression.md',
           'chapter_convolutional-neural-networks/why-conv.md',
           'chapter_attention/what-attention-computes.md']
PYTHON = ['class Trainer(', 'def masked_softmax(',
          'class MultiHeadAttention(', 'def rope(']
LATEX = ['chapter_mdl-linear-algebra/mdl-eigendecomposition.md',
         'chapter_mdl-calculus/mdl-single-variable-calculus.md',
         'chapter_mdl-probability-statistics/mdl-maximum-likelihood.md',
         'chapter_mdl-information-theory/mdl-information-theory.md']
CHAT = ['chapter_linear-regression/linear-regression.md',
        'chapter_convolutional-neural-networks/why-conv.md',
        'chapter_recurrent-neural-networks/rnn.md',
        'chapter_computational-performance/performance-model.md']
LONG = 'chapter_introduction/index.md'

# Written for this fixture (the book's own text, CC BY-SA 4.0).
CHINESE = [
    '深度学习的核心思想是用数据来训练模型，而不是靠人手写规则。一个神经网络由许多层'
    '组成，每一层都把输入变换成新的表示。训练时，我们先计算模型在一批样本上的损失，'
    '再用反向传播求出每个参数的梯度，最后沿着梯度的反方向更新参数。只要数据足够多、'
    '模型足够大，这个简单的过程就能学到相当复杂的函数。',
    '语言模型的任务是预测下一个词元。给定前面的文本，模型输出词表上每个词元的概率，'
    '然后我们可以从这个分布中采样，或者直接选取概率最大的词元。把新生成的词元接到'
    '输入后面，再重复这一步，模型就能写出一整段话。温度越高，采样结果越多样；温度'
    '越低，结果越确定，也越容易重复。',
    '注意力机制让模型在处理一个位置时，可以直接查看序列中的所有其他位置。每个位置'
    '都会生成查询、键和值三个向量。查询与键的点积决定了权重，权重经过归一化之后再'
    '对值进行加权求和。与循环神经网络相比，注意力可以并行计算，因此在现代硬件上'
    '训练得更快，但它的计算量会随着序列长度的平方增长。',
    '在比较两个模型时，只看一个准确率数字是不够的。测试集只是从所有可能问题中抽取的'
    '一个样本，换一组题目，结果就可能不同。更稳妥的做法是在同一组题目上评估两个'
    '模型，计算它们逐题得分的差，然后用自助法估计这个差的置信区间。如果区间包含零，'
    '我们就不能断言哪个模型更好。',
]
# Translations of the opening of sec_perf_model
# (chapter_computational-performance/performance-model.md), written for this
# fixture.
LANGUAGES = {
    'de': 'Jedes Netz, das wir bisher trainiert haben, verbringt den größten '
          'Teil seiner Rechenarbeit mit Matrixprodukten, und wir haben '
          'angenommen, dass eine GPU sie mit der Geschwindigkeit berechnet, '
          'die ihr Datenblatt angibt. Diese Annahme ist oft falsch. Ein '
          'Produkt zweier kleiner Matrizen erreicht weniger als zwei Prozent '
          'der Spitzenleistung, obwohl die Operation und der Chip dieselben '
          'sind.',
    'fr': "Chaque réseau que nous avons entraîné jusqu'ici consacre "
          "l'essentiel de ses calculs à des produits matriciels, et nous "
          "avons supposé qu'un GPU les calcule à la vitesse indiquée par sa "
          'fiche technique. Cette hypothèse est souvent fausse. Le produit '
          'de deux petites matrices atteint moins de deux pour cent de la '
          "performance maximale, bien que l'opération et la puce soient les "
          'mêmes.',
    'es': 'Cada red que hemos entrenado hasta ahora dedica la mayor parte de '
          'su aritmética a productos de matrices, y hemos supuesto que una '
          'GPU los calcula a la velocidad que indica su especificación. Esa '
          'suposición suele ser falsa. El producto de dos matrices pequeñas '
          'alcanza menos del dos por ciento del rendimiento máximo, aunque la '
          'operación y el chip son los mismos.',
    'ja': 'これまでに訓練したどのネットワークも、計算の大部分を行列積に費やして'
          'おり、私たちはGPUが仕様書に書かれた速度でそれを計算すると仮定してきた。'
          'しかし、この仮定はしばしば誤りである。小さな行列どうしの積は、演算も'
          'チップも同じであるにもかかわらず、ピーク性能の2パーセントにも届かない。'
          '同じ演算がなぜこれほど違う速さで動くのか、そしてプログラムがどちらの'
          '場合にあるのかをどう見分けるのか。',
}


def sha1_of(path):
    return hashlib.sha1(pathlib.Path(path).read_bytes()).hexdigest()


def tokenizer():
    from transformers import AutoTokenizer
    from d2l import torch as d2l
    rev = d2l.HF_REVISIONS[MODEL]
    return AutoTokenizer.from_pretrained(MODEL, revision=rev), rev


def encode(tok, text):
    return tok(text, add_special_tokens=False).input_ids


def window(tok, ids, n, wrap=lambda w: w):
    """The first window [s, s + n) of ids whose wrapped decode round-trips."""
    for s in range(len(ids) - n + 1):
        w = wrap(ids[s:s + n])
        text = tok.decode(w)
        if encode(tok, text) == w:
            return s, w, text
    raise ValueError('no window round-trips')


def first_exercise(md):
    """Text of the first top-level item of the Exercises section, up to its
    first sub-item, with whitespace collapsed."""
    sec = md.split('## Exercises', 1)[1]
    item = re.search(r'^1\. (.*?)(?=^\s+1\. |^\s*$)', sec, flags=re.S | re.M)
    return ' '.join(item[1].split())


def library_source(src, start):
    return src[src.index('\n' + start) + 1:]


def arithmetic(row):
    lines = []
    for j in range(16):
        i = 16 * row + j
        a, b = (37 * i + 11) ** 2 % 9973, (53 * i + 29) ** 2 % 9967
        lines.append(f'{a} + {b} = {a + b}')
    return '\n'.join(lines) + '\n'


def make_prompts(out):
    import transformers
    tok, rev = tokenizer()
    show = lambda p: book_text.git_show(ROOT, COMMIT, p)  # noqa: E731
    rows = []

    def add(category, source, text, min_tokens=T):
        ids = encode(tok, text)
        assert len(ids) >= min_tokens, (category, source, len(ids))
        s, w, decoded = window(tok, ids, T)
        rows.append({'category': category, 'source': source, 'start': s,
                     'ids': w, 'text': decoded})

    for path in ENGLISH:
        para = next(p for p in book_text.paragraphs(book_text.prose(show(path)))
                    if len(encode(tok, p)) >= 80)
        add('english', path, para)
    lib = show('d2l/torch.py')
    for start in PYTHON:
        add('python', f'd2l/torch.py: {start}', library_source(lib, start))
    for i, text in enumerate(CHINESE):
        add('chinese', f'written for this fixture ({i})', text)
    for path in LATEX:
        add('latex', path, '\n'.join(book_text.display_math(show(path))[:12]))
    for r in range(4):
        add('arithmetic', f'a = (37i + 11)^2 mod 9973, b = (53i + 29)^2 mod '
            f'9967, i = {16 * r}..{16 * r + 15}', arithmetic(r))
    for lang, text in LANGUAGES.items():
        add('languages', f'translation ({lang}) of the opening of '
            'chapter_computational-performance/performance-model.md', text)
    rng = np.random.default_rng(RANDOM_SEED)
    for k in range(4):
        half = rng.integers(0, VOCAB_REGULAR, size=T // 2).tolist()
        rows.append({'category': 'random', 'source': f'draw {k + 1} of '
                     f'default_rng({RANDOM_SEED}).integers(0, {VOCAB_REGULAR}'
                     f', 32), repeated', 'start': 0, 'ids': half + half,
                     'text': None})
    for path in CHAT:
        question = first_exercise(show(path))
        msg = [{'role': 'user', 'content': '\0'}]
        shell = tok.apply_chat_template(msg, tokenize=False,
                                        add_generation_prompt=True,
                                        enable_thinking=False)
        pre, post = (encode(tok, part) for part in shell.split('\0'))
        n = T - len(pre) - len(post)
        content = encode(tok, question)
        assert len(content) >= n, (path, len(content), n)
        s, w, decoded = window(tok, content, n, lambda c: pre + c + post)
        # The rendered message of the cut question equals the window.
        cut = tok.decode(content[s:s + n])
        assert encode(tok, tok.apply_chat_template(
            [{'role': 'user', 'content': cut}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)) == w
        rows.append({'category': 'chat', 'source': f'{path}: exercise 1',
                     'start': s, 'ids': w, 'text': decoded})
    assert len(rows) == 32 and all(len(r['ids']) == T for r in rows)

    long_text = book_text.prose(show(LONG))
    long_ids = encode(tok, long_text)
    assert len(long_ids) >= LONG_T, len(long_ids)
    s, long_w, long_decoded = window(tok, long_ids, LONG_T)

    doc = {
        'name': 'lm-qwen3-port-prompts-v1',
        'producer': 'tools/lm/make_port_fixtures.py prompts',
        'tokenizer': {'model': MODEL, 'revision': rev,
                      'transformers': transformers.__version__},
        'commit': book_text.git_commit(ROOT, COMMIT)[0],
        'licence': f'{book_text.PROSE_LICENCE}; library source '
                   f'{book_text.CODE_LICENCE}',
        'categories': ['english', 'python', 'chinese', 'latex', 'arithmetic',
                       'languages', 'random', 'chat'],
        'round_trip': 'for rows with text: tok(text, add_special_tokens='
                      'False).input_ids == ids; the window start moves '
                      'forward one token until this holds',
        'ids': [r['ids'] for r in rows],
        'texts': [r['text'] for r in rows if r['text'] is not None],
        'text_rows': [i for i, r in enumerate(rows) if r['text'] is not None],
        'rows': [{k: r[k] for k in ('category', 'source', 'start')}
                 for r in rows],
        'long': {'source': LONG, 'processing': 'tools/lm/book_text.prose',
                 'start': s, 'ids': long_w, 'text': long_decoded},
    }
    out = pathlib.Path(out or OUT / 'lm-qwen3-port-prompts-v1.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, ensure_ascii=False, separators=(',', ':'))
                   + '\n')
    return out


def check_prompts(path):
    """Re-tokenize every stored text and compare (the round-trip rule)."""
    tok, _ = tokenizer()
    doc = json.loads(pathlib.Path(path).read_text())
    assert len(doc['ids']) == 32 and len(doc['texts']) == 28
    for i, text in zip(doc['text_rows'], doc['texts']):
        assert encode(tok, text) == doc['ids'][i], i
    assert encode(tok, doc['long']['text']) == doc['long']['ids']
    assert len(doc['long']['ids']) == LONG_T


def make_logit_ref(out, prompts, model_id=MODEL, device='cuda', top=100):
    """fp64 Transformers top-k logits on the grid (see the docstring)."""
    import torch
    import transformers
    from d2l import torch as d2l
    if device == 'cpu' and model_id == MODEL:
        sys.exit('the real reference is made on the GPU host (stage B2)')
    ids = torch.tensor(json.loads(pathlib.Path(prompts).read_text())['ids'])
    model, _ = d2l.load_hf(model_id, dtype=torch.float64, device_map=device)
    assert model.config._attn_implementation == 'sdpa'
    values, indices, lse = [], [], []
    with torch.no_grad():
        for batch in ids.split(4):
            logits = model(batch.to(device)).logits
            v, i = logits.topk(top, dim=-1)
            values.append(v.float().cpu())
            indices.append(i.int().cpu())
            lse.append(logits.logsumexp(-1).float().cpu())
    gpu = (torch.cuda.get_device_name() if device.startswith('cuda')
           else 'cpu')
    ref = {'input_ids': ids.int(), 'topk_logits': torch.cat(values),
           'topk_ids': torch.cat(indices), 'logsumexp': torch.cat(lse),
           'meta': {'model': model_id,
                    'revision': getattr(model, 'revision', None)
                    or d2l.HF_REVISIONS.get(model_id),
                    'dtype': 'float64', 'attn': 'sdpa', 'batch_size': 4,
                    'top_k': top, 'stored_as': 'float32',
                    'transformers': transformers.__version__,
                    'torch': str(torch.__version__), 'device': gpu,
                    'prompts_sha1': sha1_of(prompts)}}
    out = pathlib.Path(out or OUT / 'lm-qwen3-0p6b-logit-ref-v1.pt')
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ref, out)
    return out


# A1's hidden bugs: (Transformers tensor, edit, expected (layer, point)).
A1_BUGS = {
    'a1_bug1': ('model.layers.5.self_attn.v_proj.weight', 'swap_kv_heads_2_5',
                (5, 'attn_out')),
    'a1_bug2': ('model.layers.13.mlp.gate_proj.weight', 'swap_with_up_proj',
                (13, 'mlp_out')),
    'a1_bug3': ('model.layers.21.self_attn.k_norm.weight', 'scale_1.05',
                (21, 'attn_out')),
}


def corrupt(state, cfg, bug):
    """A copy of a Transformers Qwen3 state dict with one A1 bug."""
    import torch
    name, edit, _ = A1_BUGS[bug]
    state = dict(state)
    w = state[name].clone()
    if edit == 'swap_kv_heads_2_5':
        d_h = cfg['head_dim']
        a, b = slice(2 * d_h, 3 * d_h), slice(5 * d_h, 6 * d_h)
        w[a], w[b] = state[name][b].clone(), state[name][a].clone()
    elif edit == 'swap_with_up_proj':
        up = name.replace('gate_proj', 'up_proj')
        w, state[up] = state[up].clone(), w
    elif edit == 'scale_1.05':
        w = (w.double() * 1.05).to(w.dtype)
    assert not torch.equal(w, state[name]), bug
    state[name] = w
    return state


def make_a1_corrupt(out_dir, model_id=MODEL):
    import shutil
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file, save_file
    from d2l import torch as d2l
    out_dir = pathlib.Path(out_dir).expanduser().resolve()
    assert ROOT not in out_dir.parents and out_dir != ROOT, (
        'A1 bugs stay outside the repository')
    rev = d2l.HF_REVISIONS.get(model_id)
    files = {f: hf_hub_download(model_id, f, revision=rev)
             for f in ('model.safetensors', 'config.json')}
    state = load_file(files['model.safetensors'])
    cfg = json.loads(pathlib.Path(files['config.json']).read_text())
    answers = {}
    for bug, (name, edit, (layer, point)) in A1_BUGS.items():
        d = out_dir / bug
        d.mkdir(parents=True, exist_ok=True)
        save_file(corrupt(state, cfg, bug), str(d / 'model.safetensors'),
                  metadata={'format': 'pt'})
        shutil.copy(files['config.json'], d / 'config.json')
        answers[bug] = {'tensor': name, 'edit': edit, 'layer': layer,
                        'point': point}
    (out_dir / 'answers.json').write_text(json.dumps(
        {'model': model_id, 'revision': rev, 'bugs': answers}, indent=1))
    return out_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='part', required=True)
    p = sub.add_parser('prompts')
    p.add_argument('--out')
    p = sub.add_parser('logit-ref')
    p.add_argument('--out')
    p.add_argument('--prompts', default=str(
        OUT / 'lm-qwen3-port-prompts-v1.json'))
    p.add_argument('--device', default='cuda')
    p = sub.add_parser('a1-corrupt')
    p.add_argument('dir')
    args = ap.parse_args()
    if args.part == 'prompts':
        out = make_prompts(args.out)
        check_prompts(out)
    elif args.part == 'logit-ref':
        out = make_logit_ref(args.out, args.prompts, device=args.device)
    else:
        print('wrote', make_a1_corrupt(args.dir))
        return
    print(f'wrote {out} ({out.stat().st_size} bytes, sha1 {sha1_of(out)})')


if __name__ == '__main__':
    main()
