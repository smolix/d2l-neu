# Tokenizers at Scale
:label:`sec_lm_tokenizers`

<!-- Author brief, copied verbatim from the Language Models part plan
(docs/llm/language_models_part_plan.md, "3.6 Tokenizers at Scale · T0").
Blueprint: docs/llm/blueprints/ch03_data.md section 3.6.
The plan is a specification, not book text; this comment is replaced
when the section is written.

**Idea:** Vocabulary size is a compute choice, and the unigram model is the alternative to merges.
**Opening:** The byte-level BPE of `sec_text-sequence` splits a Hindi sentence into three times the tokens of its English translation, so the same sentence costs three times the compute and fits a third as much into the context. The section builds the alternative to merges, measures what one tokenizer costs each language, chooses a vocabulary size from a cost model, freezes the tokenizer the reference model uses, and closes on the question of extending a vocabulary after training.
**Teaches:**
- The unigram model: piece probabilities estimated by expectation-maximization with pruning, Viterbi segmentation (`d2l.viterbi_segment`), subword regularization by sampling segmentations; SentencePiece as the production implementation; the one difference from merges (a probabilistic lexicon instead of a merge order).
- Vocabulary size against embedding and softmax cost and against sequence length: the elasticity of compression in V and the cost model that picks V for a parameter budget; the vocabulary fit of released models.
- Fertility and the tokenization premium by language (`d2l.fertility_table`) on parallel passages, and bits per character as the comparison that is fair across tokenizers.
- Extension of a vocabulary with new pieces and initialization of the new embedding rows (random, mean, decomposition over existing pieces; `d2l.extend_vocab`), with the softmax mass of the new rows at step zero as the check.
- The reference tokenizer (byte-level BPE, 32,768 pieces, Qwen3-style single-digit pre-tokenization, EOS id 0, no BOS, ids 0 to 31 reserved for specials) and the shard format Chapter 4 reads (`d2l.write_shards`, `d2l.TokenShards`).
**Derivation:** none.
**Notebook:**
1. Train SentencePiece unigram, SentencePiece BPE and byte-level BPE with the reference regex at a fixed vocabulary on a 10-language FineWeb-2 slice (5 MB per language, 90% train); implement Viterbi from scratch and check it against SentencePiece's segmentation on 1,000 sentences.
2. Compute fertility, bytes per token and the premium relative to English on parallel passages for the three tokenizers and for the GPT-2, o200k and Qwen3 tokenizers; plot bits per character per language.
3. Train one small `d2l.GPT` per tokenizer on the same 45 MB of text (a byte budget, so token counts differ and are recorded) with two seeds, and evaluate bits per byte per language on the held-out 10% and on the parallel passages.
4. Extend a small released decoder's vocabulary with Hindi pieces under three initializations and continue training briefly on Hindi; report Hindi bits per byte at step 0 and early steps, English retention, and the new tokens' softmax mass at step 0.
5. Verify that the freshly trained reference tokenizer reproduces the committed `tokenizer.json` hash; write a 100-million-token FineWeb-Edu shard with `d2l.write_shards` and read it back with `d2l.TokenShards`, checking EOS placement and the resume cursor.
6. Claim that survives noise: the tokenizer type (unigram against BPE at equal V) matters less for per-language bits per byte than the premium spread across languages does (differences smaller than the larger of the two seed ranges are not interpreted; passages are bootstrapped for the macro-average); informed initializations reach lower Hindi bits per byte at step 0 than random and the arms converge with training (step-0 values are deterministic and bootstrapped over passages).
7. Reused pieces: `d2l.BPETokenizer` and `GPT2_PATTERN` (`sec_text-sequence`, recapped in one paragraph), `d2l.GPT`, `d2l.train_proxy_lm`, `d2l.ProxyCfg`, `d2l.bpb_from_nll`, `d2l.temperature_weights` (3.4), `d2l.CorpusManifest`, `d2l.docs_hash` (3.1), `d2l.load_hf` (1.1); new `d2l.UnigramTokenizer`, `d2l.viterbi_segment`, `d2l.fertility_table`, `d2l.R2_SPLIT_REGEX`, `d2l.train_hf_bpe`, `d2l.r2_tokenizer`, `d2l.extend_vocab`, `d2l.write_shards`, `d2l.TokenShards`.
**Figures:** `fig_lm_fertility_by_language` (plot: bits per byte against the tokenization premium per language and tokenizer, with lines joining a language across tokenizers); `fig_lm_unigram_viterbi` (the lattice of segmentations of one word with piece log-probabilities and the best path).
**Restate / forward:** `subsec_fertility-digits-glitch` (fertility differs by script and digits chunk unevenly under a merge-based tokenizer); `sec_mdl-coding-compression` (bits per character converts a per-token loss into a tokenizer-free quantity, `eq_mdl-bpc`). Forward: 4.1 (the stream that reads these shards).
**Exercises:** 5 to 7. Load-bearing: derive the vocabulary size that balances embedding and softmax cost against sequence length for a stated parameter budget and compression elasticity (derive); measure digit chunking under the three tokenizers on a numeric passage and the effect on a small arithmetic probe (measure); check that `d2l.TokenShards.locate(i)` returns a cursor from which `iter_docs` resumes at document i, and that id 0 appears only at document ends (check); open: edit the merge table instead of appending pieces so the extended tokenizer stays BPE-consistent, and compare segmentation of mixed Hindi-English text.
**Writer notes:**
- The reference tokenizer is a one-way door (blueprint README Decision 1): byte-level BPE with vocabulary exactly 32,768; normalizer NFC; pre-tokenizer `Split(R2_SPLIT_REGEX, 'isolated')` then `ByteLevel(add_prefix_space=False, use_regex=False)` with Qwen3's regex (contractions, letters with an optional leading non-letter, single digits, punctuation runs, newlines, whitespace); specials `<|endoftext|>` = 0 (EOS and document delimiter; no BOS), `<|pad|>` = 1, `<|im_start|>` = 2, `<|im_end|>` = 3, `<|reserved_4|>` to `<|reserved_31|>` = 4 to 31 (so later chat, tool or thinking tokens need no retraining); merges start at id 32 in trainer order (do not assume id minus 32 is a byte value); trained on 200,000 train-bucket documents disjoint from validation; only `<|endoftext|>` occurs in pretraining, so a chapter that starts using a reserved id initializes its row first. Rejected alternative: reuse GPT-2's tokenizer (a 53% larger embedding for a 150M model and worse digit and whitespace handling).
- The shard format is normative for 4.1's `d2l.PackedStream` and 4.2: `manifest.json` in the `d2l-corpus/1` schema with tokenizer hash and format; `{split}-{k:05d}.bin` raw little-endian ids, `uint16` when every id is at most 65,535 and `uint32` otherwise (Qwen's 151,936 in 4.4 needs `uint32`), recorded in `format.dtype`; EOS appended after every document and nowhere else; documents never span shards; `{split}-{k:05d}.idx` as `uint32` offsets of length documents plus one; `.meta.parquet` for audits; `TokenShards.locate(i)` returns the resume cursor; `verify()` recomputes hashes; `fetch=True` downloads missing shards from the manifest's `DATA_HUB` entries, which pin a commit of a d2l-owned Hugging Face dataset repository (never `main`).
- Fertility and digit chunking are restated from `subsec_fertility-digits-glitch` and bits per character from `sec_mdl-coding-compression`, not re-measured on the GPT-2 tokenizer; the section's own content is the unigram model, the vocabulary cost model, extension and initialization, and the reference tokenizer and shards; FineWeb-2 is named as the multilingual corpus; the experiment with small LMs is T1 while the rest is T0 (the section's tier stays T0 in the table, with the T1 cells on committed results by default).
- Data: FineWeb-2 configs for nine languages plus FineWeb-Edu English, 5 MB each (ODC-By); parallel passages from `facebook/belebele` (CC BY-SA 4.0; the language codes differ from FineWeb-2's, for example `zho_Hans` against `cmn_Hani`; passages are translations of the same text, so token counts compare); Hindi documents from FineWeb-2 for the extension and 200 held-out English documents for retention; reference tokenizers GPT-2 and o200k through `tiktoken`, Qwen3-0.6B's `tokenizer.json`, and a small released decoder with a byte-level tokenizer for the extension arm.
- SentencePiece settings: `byte_fallback=True`, identity normalization after the notebook's own NFC (its default applies NFKC-like rules silently), `split_digits=True`, `character_coverage=0.9995`; the pure-Python `d2l.UnigramTokenizer` is the fallback if wheels fail.
- Pin the `tokenizers` library to a 0.x release (the blueprint pinned at least 0.21): the documentation of the 1.0 release candidate, read on 6 November 2026, says its bindings load and run a `tokenizer.json` but do not yet expose building a tokenizer from components, editing or training, which `d2l.train_hf_bpe` and the reference-tokenizer cell need.
- Pitfalls to state: comparing perplexity across tokenizers; counting bytes before normalization for one tokenizer and after for another; training the tokenizer on text the model never sees (glitch tokens); choosing V by habit; the number of new tokens is not the benefit; mean initialization makes every new row identical and can hijack the softmax at step 0 (check the mass); forgetting the output matrix when embeddings are untied and `config.vocab_size`; adding pieces that contain ASCII changes English segmentation; `uint16` with V over 65,536; EOS inside a document.
- Hosting: the 4-billion-token shard set is about 8 GB; if the bucket cannot host it, ship the manifest and build script and host 1 billion tokens.
- Committed artifacts: `lm-r2-tokenizer` (`tokenizer.json`, 2 to 3 MB), `lm-fwedu-val` (at least 2^23 tokens plus the NFC text), `lm-fwedu-shards-100m`, `lm-fwedu-manifest-4b` (40 shards of 10^8 tokens with URLs and hashes), the two result tables as JSON.
- Do not re-teach: the BPE merge rule and pre-tokenization (`sec_text-sequence`), the code-length view of a unigram model beyond one sentence (`sec_mdl-information_theory`), the GPT-2 fertility table.
**Sources:**
- Papers: [SentencePiece](https://aclanthology.org/D18-2012/) (the unigram and BPE trainer with byte fallback); [Subword Regularization](https://aclanthology.org/P18-1007/) (the unigram model, Viterbi and sampled segmentations); [Scaling Laws with Vocabulary](https://arxiv.org/abs/2407.13623) (optimal vocabulary size as a function of compute: the cost model); [Language Model Tokenizers Introduce Unfairness Between Languages](https://arxiv.org/abs/2305.15425) (the tokenization premium by language); [Byte Pair Encoding is Suboptimal for Language Model Pretraining](https://arxiv.org/abs/2004.03720) (unigram against BPE in their setting, cited with that limit); [WECHSEL](https://aclanthology.org/2022.naacl-main.293/) (embedding initialization for a new vocabulary by decomposition over the old); [Neural Machine Translation of Rare Words with Subword Units](https://aclanthology.org/P16-1162/) (BPE's origin, one sentence); [Byte Latent Transformer](https://arxiv.org/abs/2412.09871) (the tokenizer-free alternative, one pointer to Chapter 14's Further Reading).
- Code and documentation: [tokenizers](https://huggingface.co/docs/tokenizers/index) and its [pipeline guide](https://huggingface.co/docs/tokenizers/pipeline) (normalizer, pre-tokenizer, model, decoder: the reference tokenizer is built with these); [tokenization algorithms summary](https://huggingface.co/docs/transformers/tokenizer_summary); [SentencePiece](https://github.com/google/sentencepiece) (the trainer and its normalization defaults); [minbpe](https://github.com/karpathy/minbpe) (the from-scratch BPE oracle); [FineWeb-2 dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) (the per-language slices).
- Courses, tutorials and blogs: [Hugging Face LLM Course, chapter 6](https://huggingface.co/learn/llm-course/en/chapter6/1) (training a tokenizer with the library); [CS336](https://cs336.stanford.edu/) (the tokenization lecture and assignment A1's tokenizer).
-->

<!-- Prose: the opening (spec 20.6 part 1): the byte-level BPE of
sec_text-sequence splits a Hindi passage into several times the tokens of
its English translation (the table below); one paragraph recaps
sec_text-sequence (byte-level BPE, pre-tokenization, specials,
subsec_fertility-digits-glitch) and says what is new here: the unigram
model, what a tokenizer's training data does to each language, the size of
the vocabulary, changing a trained model's vocabulary, and the frozen
reference tokenizer with its shards. -->

```{.python .input #tokenizers-imports}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #tokenizers-extras}
%%tab pytorch
import collections
import copy
import gzip
import hashlib
import itertools
import json
import os
import shutil
import tempfile
import textwrap
import time
import unicodedata
import warnings
import pandas as pd
import regex
import sentencepiece as spm
import tiktoken
import transformers
from scipy.interpolate import PchipInterpolator
from torch.utils.checkpoint import checkpoint
```

<!-- Prose: what the notebook computes and where: tokenizer training,
Viterbi, fertility, the vocabulary cost model and the shards run on the CPU
in every build; the small-LM comparison and the vocabulary extension are
T1 results regenerated on one GPU (D2L_RERUN=tok-lm, tok-extend) and read
from registered files otherwise. -->

```{.python .input #tokenizers-config}
%%tab pytorch
LANGS = ('eng_Latn', 'deu_Latn', 'rus_Cyrl', 'arb_Arab', 'hin_Deva',
         'cmn_Hani', 'jpn_Jpan', 'tha_Thai', 'swh_Latn', 'kor_Hang')
BELEBELE = {l: 'zho_Hans' if l == 'cmn_Hani' else l for l in LANGS}
V_SMALL = 16_384            # the small tokenizers' vocabulary
ALPHAS = (1.0, 0.3, 0.0)     # language weighting of tokenizer text
TOK_PROXY = d2l.ProxyCfg(num_hiddens=256, num_heads=4, num_blks=4,
                         seq_len=512, vocab_size=V_SMALL)
SEEDS, LOG_EVERY, LIVE_TOKENS = (0, 1), 4, 2**19
LM_EPOCHS = 1.0              # passes of each small LM over its 45 MB
EXT_MODEL, N_NEW = 'Qwen/Qwen3-0.6B-Base', 4_096
EXT_TRAIN = dict(embed_steps=50, embed_lr=3e-3, lr=1e-5, warmup=10,
                 steps=150, batch=8, micro_batch=4, seq_len=1024,
                 eval_at=(0, 25, 50, 100, 150), seeds=(0, 1),
                 loss_chunk=1024, final_lr_frac=0.1, curve_docs=100,
                 pool_tokens=2**22)
N_TOK_DOCS, R2_VOCAB = 80_000, 32_768  # the reference tokenizer
ARTIFACTS = {  # name: (registered key, file a regeneration writes)
    'tok-lm': ('lm-data-tok-results-v1', 'tok-lm.json'),
    'tok-extend': ('lm-data-tok-results-v1', 'tok-extend.json')}
RERUN = d2l.lm_rerun(ARTIFACTS)
GPU = torch.cuda.is_available()
assert GPU or not RERUN, 'the regenerations train on a GPU'
assert GPU or not os.environ.get('CUDA_VISIBLE_DEVICES'), (
    'a GPU slot without CUDA: the small CPU path must not be captured')
SMALL = not GPU  # the CPU path: step-0 diagnostics on fewer texts
DEVICE = d2l.try_gpu()
N_EXT = (dict(hindi=8, passages=24, english=8, contexts=4, ctx_len=256)
         if SMALL else dict(hindi=500, passages=488, english=200,
                            contexts=200, ctx_len=1024))
RERUN_DIR = os.path.join('..', 'data', 'lm', 'rerun')

def say(*args, sep=' '):
    """print, each line wrapped to the page's 100 columns."""
    print('\n'.join(l if len(l) <= 100 else textwrap.fill(
        l, 100, subsequent_indent='    ', break_long_words=False,
        break_on_hyphens=False) for l in sep.join(map(str, args)).split('\n')))

with torch.device('meta'):
    N_PARAMS = d2l.num_params(d2l._proxy_model(TOK_PROXY))
TIER = (f'at T1 scale ({N_PARAMS / 1e6:.1f}M-parameter LMs, one epoch over '
        f'45 MB; {EXT_MODEL.split("/")[1]}, {EXT_TRAIN["steps"]} steps of '
        f'{EXT_TRAIN["batch"]} x {EXT_TRAIN["seq_len"]:,} tokens)')
say(f'small-LM comparison and extension {TIER.replace(" scale", "")}')
say(f'GPU={GPU}, SMALL={SMALL}, regenerating:',
    ', '.join(sorted(RERUN)) or 'none')
```

<!-- Prose: the registry: Belebele (CC BY-SA 4.0) is downloaded from its
pinned commit, never re-hosted; the ten-language slice is FineWeb-2 and
FineWeb-Edu text (ODC-By); the reference tokenizer and its shards are built
once by tools/build_r2_shards.py, never by this notebook. -->

<!-- Prose: the results of the two GPU regenerations (the six small LMs
and the eight extension runs) are one registered file,
lm-data-tok-results-v1, made in one run (D2L_RERUN=1, one whole RTX 4090,
2026-11-21); a later single-part regeneration reads the other part from
it. -->

```{.python .input #tokenizers-data-hub}
%%tab pytorch
#@save
d2l.DATA_HUB['belebele-eng_Latn'] = (  # FLORES passages, CC BY-SA 4.0
    'https://huggingface.co/datasets/facebook/belebele/resolve/'
    '7899cdfa4e1e0d733fd77c848e2c273cb1d32be2/data/eng_Latn.jsonl',
    '39c6796587fef3057a50ff60ecfa3d99c65ada99')
d2l.DATA_HUB['belebele-deu_Latn'] = (
    'https://huggingface.co/datasets/facebook/belebele/resolve/'
    '7899cdfa4e1e0d733fd77c848e2c273cb1d32be2/data/deu_Latn.jsonl',
    '2492011aa4718237cfaf71256150acb453184401')
d2l.DATA_HUB['belebele-rus_Cyrl'] = (
    'https://huggingface.co/datasets/facebook/belebele/resolve/'
    '7899cdfa4e1e0d733fd77c848e2c273cb1d32be2/data/rus_Cyrl.jsonl',
    '9c17bbd2594ec9ea831ad887587a4a401f80c48d')
d2l.DATA_HUB['belebele-arb_Arab'] = (
    'https://huggingface.co/datasets/facebook/belebele/resolve/'
    '7899cdfa4e1e0d733fd77c848e2c273cb1d32be2/data/arb_Arab.jsonl',
    'd409a6fda63a97566583558886f0414c8edd152a')
d2l.DATA_HUB['belebele-hin_Deva'] = (
    'https://huggingface.co/datasets/facebook/belebele/resolve/'
    '7899cdfa4e1e0d733fd77c848e2c273cb1d32be2/data/hin_Deva.jsonl',
    'e7017e822412dd49ef24f24419bc3e2e45175285')
d2l.DATA_HUB['belebele-zho_Hans'] = (
    'https://huggingface.co/datasets/facebook/belebele/resolve/'
    '7899cdfa4e1e0d733fd77c848e2c273cb1d32be2/data/zho_Hans.jsonl',
    'f7da1fed3c0c6d7e43a324bf8739231f2b346b6e')
d2l.DATA_HUB['belebele-jpn_Jpan'] = (
    'https://huggingface.co/datasets/facebook/belebele/resolve/'
    '7899cdfa4e1e0d733fd77c848e2c273cb1d32be2/data/jpn_Jpan.jsonl',
    '2a3755b2eaea864e066f4414c3f64e0a53cca41a')
d2l.DATA_HUB['belebele-tha_Thai'] = (
    'https://huggingface.co/datasets/facebook/belebele/resolve/'
    '7899cdfa4e1e0d733fd77c848e2c273cb1d32be2/data/tha_Thai.jsonl',
    '4aae5d586fe29db59bf12bfcbc9b75e816ca4380')
d2l.DATA_HUB['belebele-swh_Latn'] = (
    'https://huggingface.co/datasets/facebook/belebele/resolve/'
    '7899cdfa4e1e0d733fd77c848e2c273cb1d32be2/data/swh_Latn.jsonl',
    '1c327092d52031a5d7dfffe5b7419e0f2f584bc8')
d2l.DATA_HUB['belebele-kor_Hang'] = (
    'https://huggingface.co/datasets/facebook/belebele/resolve/'
    '7899cdfa4e1e0d733fd77c848e2c273cb1d32be2/data/kor_Hang.jsonl',
    '17fa36b78cef39784d497a7cf40caf678e83c955')
d2l.DATA_HUB['lm-data-fw2-slice-v1'] = (  # FineWeb-2, FineWeb-Edu: ODC-By
    d2l.LM_DATA_URL + 'lm-data-fw2-slice-v1.tar',
    '80ddb29c077df2fa73fb85f45197e32459ba4f56')
d2l.DATA_HUB['lm-r2-tokenizer-v1'] = (  # the reference tokenizer
    d2l.LM_DATA_URL + 'lm-r2-tokenizer-v1.json',
    '4e8dbe43379e205000c68cbec6798278761a38cd')
d2l.DATA_HUB['lm-fwedu-val-v1'] = (  # FineWeb-Edu validation shard: ODC-By
    d2l.LM_DATA_URL + 'lm-fwedu-val-v1.tar',
    '2675d35e9e0fb6a8e5b287c3928f5c014a8a34ab')
d2l.DATA_HUB['lm-fwedu-shards-100m-v1'] = (  # train-00000: ODC-By
    d2l.LM_DATA_URL + 'lm-fwedu-shards-100m-v1.tar',
    '66992765074d705d749c304b54f06fe5586f0c0f')
d2l.DATA_HUB['lm-fwedu-manifest-4b-v1'] = (  # 40 shards under fwedu-r2-v1/
    d2l.LM_DATA_URL + 'lm-fwedu-manifest-4b-v1.json',
    '2d97947273a7df83f494ad20bbb561bfbd2b336c')
d2l.DATA_HUB['lm-data-tok-results-v1'] = (  # small LMs and extension runs
    d2l.LM_DATA_URL + 'lm-data-tok-results-v1.json',
    'b0eb51e54b42e5273dc9b17ba416f9f998531548')

SLICE_KEY, RES_KEY = 'lm-data-fw2-slice-v1', 'lm-data-tok-results-v1'
R2_KEYS = ('lm-r2-tokenizer-v1', 'lm-fwedu-val-v1',
           'lm-fwedu-shards-100m-v1', 'lm-fwedu-manifest-4b-v1')
for name, keys in (('this section', (SLICE_KEY, RES_KEY) + R2_KEYS),
                   ('upstream', ('lm-mix-runs-t1-v1',))):
    say(f'{name}: registered', ', '.join(
        k for k in keys if k in d2l.DATA_HUB) or 'none', '| pending:',
        ', '.join(k for k in keys if k not in d2l.DATA_HUB) or 'none')

def open_tar(key):
    """The folder of a registered tar, extracted afresh whenever the
    registered sha1 differs from the one stamped at the last extraction."""
    path = d2l.download_keyed(key)
    root = os.path.join(os.path.dirname(path), 'files')
    stamp, old = os.path.join(root, '.sha1'), None
    if os.path.exists(stamp):
        with open(stamp) as f:
            old = f.read()
    if old != d2l.DATA_HUB[key][1]:
        shutil.rmtree(root, ignore_errors=True)
        os.makedirs(root)
        d2l.extract(path, root)
        with open(stamp, 'w') as f:
            f.write(d2l.DATA_HUB[key][1])
    return root

def write_json(path, obj):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, sort_keys=True, ensure_ascii=False)

ROOT = open_tar(SLICE_KEY)
SLICE = pd.read_parquet(os.path.join(ROOT, 'slice.parquet'))
HINDI = pd.read_parquet(os.path.join(ROOT, 'hindi.parquet'))
with open(os.path.join(ROOT, 'meta.json')) as f:
    SLICE_META = json.load(f)
TRAIN, HELD = ({l: SLICE[(SLICE.lang == l) & (SLICE.part == p)].text.tolist()
                for l in LANGS} for p in ('train', 'heldout'))
assert all(t == unicodedata.normalize('NFC', t) for t in SLICE.text)
RESULTS = None  # the registered results of both regenerations
if RES_KEY in d2l.DATA_HUB:
    with open(d2l.lm_artifact(RES_KEY), encoding='utf-8') as f:
        RESULTS = json.load(f)
assert RESULTS is not None or RERUN in (set(), set(ARTIFACTS)), (
    'the first generation makes both parts in one run (D2L_RERUN=1); a '
    'single part reads the other one from the registered results')
print(pd.DataFrame({
    'train docs': {l: len(TRAIN[l]) for l in LANGS},
    'train MB': {l: sum(len(t.encode()) for t in TRAIN[l]) / 1e6
                 for l in LANGS},
    'held-out docs': {l: len(HELD[l]) for l in LANGS}}).round(2))
print('Hindi extension documents:', HINDI.part.value_counts().to_dict())
```

<!-- Prose: fertility and premium defined (restating
subsec_fertility-digits-glitch in one sentence): bytes per token on the
same passages, and the premium as a language's tokens over English's for
translations of the same text. -->

```{.python .input #tokenizers-fertility-def}
%%tab pytorch
#@save
def fertility_table(encoders, parallel, ref='eng_Latn'):
    """Per encoder and language: passages, UTF-8 bytes, tokens, bytes per
    token, tokens per passage, the share of byte-fallback tokens and the
    premium, the tokens a language needs over the tokens ref needs for
    translations of the same passages. encoders maps a name to encode(str)
    -> ids, or to (encode, fallback_ids); parallel maps a language to its
    passages, aligned across languages."""
    rows = []
    for name, enc in encoders.items():
        enc, fallback = enc if isinstance(enc, tuple) else (enc, ())
        fallback = set(fallback)
        for lang, passages in parallel.items():
            ids = [enc(p) for p in passages]
            n = sum(map(len, ids))
            n_bytes = sum(len(p.encode('utf-8')) for p in passages)
            rows.append(dict(
                encoder=name, lang=lang, passages=len(passages),
                bytes=n_bytes, tokens=n, bytes_per_token=n_bytes / n,
                tokens_per_passage=n / len(passages),
                fallback_share=sum(i in fallback for x in ids for i in x)
                / n))
    df = pd.DataFrame(rows)
    ref_tokens = df[df.lang == ref].set_index('encoder').tokens
    df['premium'] = df.tokens / df.encoder.map(ref_tokens)
    return df
```

<!-- Prose: the four released tokenizers (GPT-2 and o200k_base through
tiktoken, Qwen3 and Qwen3.5 from their pinned tokenizer files) on the 488
distinct Belebele passages; Belebele's zho_Hans is FineWeb-2's cmn_Hani. -->

```{.python .input #tokenizers-opening}
%%tab pytorch
def belebele(lang):
    """The distinct passages of a Belebele language by link (a passage's
    link is the same in every language)."""
    with open(d2l.download_keyed('belebele-' + BELEBELE[lang]),
              encoding='utf-8') as f:
        return {r['link']: r['flores_passage'] for r in map(json.loads, f)}

def hf_tokenizer(model_id):
    return transformers.AutoTokenizer.from_pretrained(
        model_id, revision=d2l.HF_REVISIONS[model_id])

def encoder(tok):
    """(encode, fallback ids) of any tokenizer the section uses; special
    strings in a text stay text."""
    if hasattr(tok, 'encode_as_ids'):  # SentencePiece
        return tok.encode, d2l._fallback_ids(tok)
    return (lambda s: d2l._doc_token_ids(tok, [s])[0],
            d2l._fallback_ids(tok))

by_link = {lang: belebele(lang) for lang in LANGS}
LINKS = sorted(by_link['eng_Latn'])
assert all(sorted(p) == LINKS for p in by_link.values())
BELE = {lang: [p[k] for k in LINKS] for lang, p in by_link.items()}
RELEASED = {'GPT-2': tiktoken.get_encoding('gpt2'),
            'o200k_base': tiktoken.get_encoding('o200k_base'),
            'Qwen3': hf_tokenizer('Qwen/Qwen3-0.6B-Base'),
            'Qwen3.5': hf_tokenizer('Qwen/Qwen3.5-0.8B-Base')}
FERT = d2l.fertility_table({k: encoder(t) for k, t in RELEASED.items()},
                           BELE)
say(f'{len(LINKS)} passages per language')
say(FERT.pivot(index='lang', columns='encoder', values='premium')
    .loc[list(LANGS)].round(2))
say(FERT.pivot(index='lang', columns='encoder', values='bytes_per_token')
    .loc[list(LANGS)].round(2))
say(FERT.pivot(index='lang', columns='encoder', values='fallback_share')
    .loc[list(LANGS)].round(3))
PREM = FERT.set_index(['encoder', 'lang']).premium
# C1: English-centred tokenizers charge non-Latin scripts several times
assert min(PREM['GPT-2', l] for l in ('hin_Deva', 'tha_Thai')) > 3
assert PREM['Qwen3.5', 'hin_Deva'] < PREM['Qwen3', 'hin_Deva']
say(f"Hindi premium: Qwen3 {PREM['Qwen3', 'hin_Deva']:.1f}, "
    f"Qwen3.5 {PREM['Qwen3.5', 'hin_Deva']:.1f}")
```

## The Unigram Model
:label:`subsec_lm_tok_unigram`

<!-- Prose: piece probabilities p(x), a segmentation's probability as the
product, the string's probability as the sum over segmentations; the
lattice (fig_lm_unigram_viterbi: positions as nodes, pieces as edges);
forward and backward sums; the posterior edge count
alpha_i p(x) beta_j / P(s) and the EM update (eq_lm_tok_em, derived with a
Lagrange multiplier); Viterbi as the max-product version; pruning by the
likelihood a piece's removal costs; sampling with P(s)^alpha for subword
regularization. -->

![The segmentation lattice of one word: positions are nodes, pieces are
edges labelled with their log-probabilities, and the Viterbi path is
highlighted.](../img/mdl-lm-data-unigram-viterbi.svg)
:label:`fig_lm_unigram_viterbi`

```{.python .input #tokenizers-viterbi-def}
%%tab pytorch
#@save
def _unigram_lattice(s, logp, max_len):
    """Edges (i, j, piece) of the segmentation lattice of s, by start i:
    s[i:j] is a piece of logp."""
    return [(i, j, s[i:j]) for i in range(len(s))
            for j in range(i + 1, min(len(s), i + max_len) + 1)
            if s[i:j] in logp]

#@save
def _unigram_fb(s, logp, max_len):
    """log P(s), summed over all segmentations of s, and each piece's
    posterior expected count in s (forward-backward on the lattice)."""
    n, edges = len(s), _unigram_lattice(s, logp, max_len)
    fwd, bwd = [-math.inf] * (n + 1), [-math.inf] * (n + 1)
    fwd[0] = bwd[n] = 0.0
    for i, j, x in edges:  # by start: fwd[i] is complete when read
        fwd[j] = d2l._logaddexp(fwd[j], fwd[i] + logp[x])
    for i, j, x in reversed(edges):  # by start, descending
        bwd[i] = d2l._logaddexp(bwd[i], logp[x] + bwd[j])
    counts = collections.Counter()
    for i, j, x in edges:
        counts[x] += math.exp(fwd[i] + logp[x] + bwd[j] - fwd[n])
    return fwd[n], counts

#@save
def viterbi_segment(text, logp, max_len, *, unk=None):
    """The most probable segmentation of text into pieces of logp (piece
    -> log-probability) of at most max_len characters; ties go to fewer
    pieces. With unk, a character that is no piece is a piece of
    log-probability unk (SentencePiece's rule for unknown characters)."""
    n = len(text)
    best, back = [(-math.inf, 0)] * (n + 1), [0] * (n + 1)
    best[0] = (0.0, 0)  # (log-probability, minus the number of pieces)
    for j in range(1, n + 1):
        for i in range(max(0, j - max_len), j):
            lp = logp.get(text[i:j], unk if j == i + 1 else None)
            if lp is not None and best[i][0] > -math.inf:
                cand = (best[i][0] + lp, best[i][1] - 1)
                if cand > best[j]:
                    best[j], back[j] = cand, i
    if best[n][0] == -math.inf:
        raise ValueError(f'no segmentation of {text!r}')
    out, j = [], n
    while j > 0:
        out.append(text[back[j]:j])
        j = back[j]
    return out[::-1]

#@save
def _unigram_sample(s, logp, max_len, alpha, rng):
    """A segmentation of s drawn with probability proportional to
    P(segmentation)^alpha: forward filtering, then backward sampling."""
    fwd, into = [-math.inf] * (len(s) + 1), collections.defaultdict(list)
    fwd[0] = 0.0
    for i, j, x in _unigram_lattice(s, logp, max_len):
        fwd[j] = d2l._logaddexp(fwd[j], fwd[i] + alpha * logp[x])
        into[j].append((i, x))
    out, j = [], len(s)
    while j > 0:
        w = np.array([fwd[i] + alpha * logp[x] for i, x in into[j]])
        p = np.exp(w - w.max())
        i, x = into[j][rng.choice(len(w), p=p / p.sum())]
        out.append(x)
        j = i
    return out[::-1]

#@save
class UnigramTokenizer:
    """A unigram language model over pieces (Kudo 2018), trained by EM
    with pruning: seed the vocabulary with every character and the
    seed_size most frequent substrings (scored by count times length),
    then alternate em_iters EM steps with keeping prune_frac of the pieces
    until vocab_size remain. Words are split at whitespace and marked by a
    leading U+2581, as SentencePiece marks them."""
    def __init__(self, vocab_size, seed_size=100_000, em_iters=2,
                 prune_frac=0.8, max_piece_len=12):
        self.vocab_size, self.seed_size = vocab_size, seed_size
        self.em_iters, self.prune_frac = em_iters, prune_frac
        self.max_len, self.pieces, self.history = max_piece_len, {}, []

    @staticmethod
    def _words(sentences):
        return collections.Counter('▁' + w for s in sentences
                                   for w in s.split())

    def _em_step(self, words, logp):
        """E: expected piece counts under logp; M: probabilities
        proportional to them. Returns the new log-probabilities and the
        corpus log-likelihood under the old ones."""
        counts, loglik = collections.Counter(), 0.0
        for w, c in words.items():
            logz, cw = d2l._unigram_fb(w, logp, self.max_len)
            loglik += c * logz
            for x, v in cw.items():
                counts[x] += c * v
        for x in logp:  # a character stays a piece, however rare
            if len(x) == 1 and counts[x] <= 0:
                counts[x] = 1e-6
        log_total = math.log(sum(counts.values()))
        return ({x: math.log(counts[x]) - log_total for x in logp
                 if counts[x] > 0}, loglik)

    def _prune(self, words, logp, keep):
        """The keep pieces whose removal costs the most likelihood: a
        piece's loss is its count in the Viterbi segmentations times what
        its log-probability gains over the best segmentation of its own
        string without it. Characters are always kept."""
        freq = collections.Counter()
        for w, c in words.items():
            for x in d2l.viterbi_segment(w, logp, self.max_len):
                freq[x] += c
        loss, others = {}, dict(logp)
        for x, lp in logp.items():
            if len(x) > 1:
                del others[x]
                alt = d2l.viterbi_segment(x, others, self.max_len)
                loss[x] = freq[x] * (lp - sum(others[a] for a in alt))
                others[x] = lp
        chars = [x for x in logp if len(x) == 1]
        ranked = sorted(loss, key=lambda x: (-loss[x], x))
        kept = set(chars) | set(ranked[:max(keep - len(chars), 0)])
        return {x: lp for x, lp in logp.items() if x in kept}

    def train(self, sentences):
        words = self._words(sentences)
        subs, chars = collections.Counter(), collections.Counter()
        for w, c in words.items():
            for i in range(len(w)):
                chars[w[i]] += c
                for j in range(i + 2, min(len(w), i + self.max_len) + 1):
                    subs[w[i:j]] += c
        seed = sorted(subs, key=lambda x: (-subs[x] * len(x), x))
        seed = {x: subs[x] * len(x) for x in seed[:self.seed_size]}
        seed.update(chars)
        total = sum(seed.values())
        logp = {x: math.log(v / total) for x, v in seed.items()}
        while True:
            for _ in range(self.em_iters):
                logp, loglik = self._em_step(words, logp)
                self.history.append((len(logp), loglik))
            if len(logp) <= self.vocab_size:
                break
            logp = self._prune(words, logp, max(
                self.vocab_size, int(len(logp) * self.prune_frac)))
        self.pieces = logp
        return self

    def encode(self, text):
        return [x for w in text.split() for x in d2l.viterbi_segment(
            '▁' + w, self.pieces, self.max_len)]

    def encode_sample(self, text, alpha=0.2, seed=0):
        """A segmentation drawn with probability proportional to
        P(segmentation)^alpha, word by word (subword regularization)."""
        rng = np.random.default_rng(seed)
        return [x for w in text.split() for x in d2l._unigram_sample(
            '▁' + w, self.pieces, self.max_len, alpha, rng)]
```

<!-- Prose: the checks of the derivation, each against brute force:
forward-backward against enumerating every segmentation, Viterbi against
the enumerated argmax, EM's likelihood against its own previous value,
sampled segmentations against their posterior probabilities. -->

```{.python .input #tokenizers-em-checks}
%%tab pytorch
def segmentations(s, logp, max_len):
    """Every segmentation of s into pieces of logp (brute force)."""
    if not s:
        yield []
        return
    for j in range(1, min(max_len, len(s)) + 1):
        if s[:j] in logp:
            for rest in segmentations(s[j:], logp, max_len):
                yield [s[:j]] + rest

def random_pieces(rng, n=12, max_len=4):
    pieces = set('abc') | {''.join(rng.choice(list('abc'), k)) for k in
                           rng.integers(2, max_len + 1, n)}
    w = rng.random(len(pieces))
    return dict(zip(sorted(pieces), np.log(w / w.sum())))

rng, err, ties = np.random.default_rng(0), 0.0, 0
for _ in range(200):  # forward-backward and Viterbi against enumeration
    logp = random_pieces(rng)
    s = ''.join(rng.choice(list('abc'), rng.integers(1, 13)))
    segs = list(segmentations(s, logp, 4))
    lps = np.array([sum(logp[x] for x in g) for g in segs])
    logz, counts = d2l._unigram_fb(s, logp, 4)
    brute = collections.Counter()
    for g, lp in zip(segs, lps):
        for x in g:
            brute[x] += math.exp(lp - np.logaddexp.reduce(lps))
    err = max(err, abs(logz - np.logaddexp.reduce(lps)),
              *(abs(brute[x] - counts[x]) for x in set(brute) | set(counts)))
    v = d2l.viterbi_segment(s, logp, 4)
    top = [len(g) for g, lp in zip(segs, lps) if lp >= lps.max() - 1e-12]
    ties += len(top) > 1
    assert ''.join(v) == s and len(v) == min(top)
    assert sum(logp[x] for x in v) >= lps.max() - 1e-12
assert err < 1e-9
say(f'forward-backward vs enumeration: max difference {err:.1e}; '
    f'Viterbi = argmax on 200 strings ({ties} with tied maxima)')

toy = [''.join(rng.choice(list('abc'), rng.integers(3, 9)))
       for _ in range(5000)]
uni = d2l.UnigramTokenizer(vocab_size=0, max_piece_len=3)
words = collections.Counter(toy)
logp = random_pieces(np.random.default_rng(1), n=20, max_len=3)
lls = []
for _ in range(6):  # EM never lowers the corpus log-likelihood
    logp, ll = uni._em_step(words, logp)
    lls.append(ll)
assert all(b >= a - 1e-9 for a, b in zip(lls, lls[1:]))
say('EM log-likelihood:', ', '.join(f'{ll:,.1f}' for ll in lls))

s, alpha = 'abcabc', 0.2  # sampled segmentations against posteriors
logp = random_pieces(np.random.default_rng(2), n=40)
segs = [tuple(g) for g in segmentations(s, logp, 4)]
w = alpha * np.array([sum(logp[x] for x in g) for g in segs])
post = dict(zip(segs, np.exp(w - np.logaddexp.reduce(w))))
draws, rng = 10_000, np.random.default_rng(0)
freq = collections.Counter(tuple(d2l._unigram_sample(s, logp, 4, alpha, rng))
                           for _ in range(draws))
z = max(abs(freq[g] / draws - p) / math.sqrt(p * (1 - p) / draws)
        for g, p in post.items())
assert z < 3
say(f'{len(segs)} segmentations of {s!r}: sampled frequencies within '
    f'{z:.1f} standard errors of their posteriors')
```

<!-- Prose: the from-scratch trainer on 1 MB of English against
SentencePiece's unigram trainer on the same text; SentencePiece's pieces
and scores loaded into viterbi_segment reproduce its segmentations; where
the two vocabularies agree; words with two plausible segmentations; five
sampled segmentations of one sentence. -->

```{.python .input #tokenizers-unigram-train}
%%tab pytorch
def first_bytes(texts, n_bytes):
    """Documents in order until n_bytes UTF-8 bytes."""
    out, b = [], 0
    for t in texts:
        if b >= n_bytes:
            break
        out.append(t)
        b += len(t.encode('utf-8'))
    return out

def lines(texts):
    return [l for t in texts for l in t.split('\n') if l.strip()]

eng_1mb = first_bytes(TRAIN['eng_Latn'], 1_000_000)
held_lines = lines(HELD['eng_Latn'])[:1000]
tic = time.time()
uni = d2l.UnigramTokenizer(4096).train(lines(eng_1mb))
t_ours = time.time() - tic
tic = time.time()
sp_uni = d2l.sp_train(eng_1mb, 'unigram', 4096)
t_sp = time.time() - tic
say(f'from scratch: {len(uni.pieces):,} pieces in {t_ours:.0f} s '
    f'({len(uni.history) // uni.em_iters} rounds); SentencePiece: '
    f'{t_sp:.1f} s')

def sp_viterbi(sp, text):
    """SentencePiece's segmentation recomputed by viterbi_segment from its
    pieces and scores: spaces become U+2581 (with the dummy prefix), a
    character that is no piece is spelled by its byte pieces."""
    logp, unk = d2l._sp_pieces_scores(sp)
    s = '▁' + text.replace(' ', '▁')
    out = []
    for x in d2l.viterbi_segment(s, logp, 16, unk=unk):
        out += ([x] if x in logp else
                [f'<0x{b:02X}>' for b in x.encode('utf-8')])
    return out

same = [sp_viterbi(sp_uni, l) == sp_uni.encode(l, out_type=str)
        for l in held_lines]
say(f'viterbi_segment reproduces SentencePiece on {sum(same)} of '
    f'{len(same)} held-out lines')
for l, ok in list(zip(held_lines, same)):
    if not ok:  # the first disagreements and their cause
        a, b = sp_viterbi(sp_uni, l), sp_uni.encode(l, out_type=str)
        k = next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)
        say('  ours', a[k:k + 4], '| SentencePiece', b[k:k + 4])
assert sum(same) >= 0.99 * len(same)  # C2

logp_sp, _ = d2l._sp_pieces_scores(sp_uni)
shared = set(uni.pieces) & set(logp_sp)
say(f'vocabulary overlap: {len(shared):,} of {len(uni.pieces):,} pieces '
    f'({len(shared) / len(uni.pieces):.0%})')
two = []  # frequent words whose two best segmentations are close
for w, _ in d2l.UnigramTokenizer._words(held_lines).most_common():
    if not 7 <= len(w) <= 13:
        continue
    segs = sorted(((sum(uni.pieces[x] for x in g), g) for g in
                   segmentations(w, uni.pieces, 12)), reverse=True)[:2]
    if len(segs) == 2 and segs[0][0] - segs[1][0] < 1.0:
        two.append((w, segs))
    if len(two) == 5:
        break
for w, segs in two:
    say(w, ' | '.join(f'{" ".join(g)} ({lp:.1f})' for lp, g in segs))
sentence = held_lines[0][:80]
say('Viterbi:', ' '.join(uni.encode(sentence)))
for seed in range(5):
    say(f'sample {seed}:', ' '.join(uni.encode_sample(sentence, 0.2, seed)))
```

<!-- Prose: SentencePiece's defaults change text before it is segmented
(nmt_nfkc folds ligatures, superscripts and full-width forms), which is why
the notebook normalizes to NFC itself and passes identity; with
character_coverage below 1 the rarest characters fall back to bytes. -->

```{.python .input #tokenizers-sp-pitfalls}
%%tab pytorch
probe = ['ﬁnance', 'x² + y²', 'ＡＢＣ１２',
         'café', 'Ⅳ ①']
nfkc = spm.SentencePieceNormalizer(rule_name='nmt_nfkc')
say(pd.DataFrame({'text': probe,
                  'nmt_nfkc': [nfkc.normalize(s) for s in probe],
                  'changed': [nfkc.normalize(s) != s for s in probe]}))
jpn = first_bytes(TRAIN['jpn_Jpan'], 1_000_000)
rows = {}
for cov in (0.9995, 1.0):
    sp = d2l.sp_train(jpn, 'unigram', 4096, character_coverage=cov)
    fb = d2l._fallback_ids(sp)
    ids = [i for p in BELE['jpn_Jpan'] for i in sp.encode(p)]
    chars = {p for p in d2l._sp_pieces_scores(sp)[0] if len(p) == 1}
    rows[cov] = dict(single_characters=len(chars),
                     fallback_share=sum(i in fb for i in ids) / len(ids),
                     tokens=len(ids))
say(pd.DataFrame(rows).T.round(4).rename_axis('character_coverage'))
```

## Training Data Decides Who Pays
:label:`subsec_lm_tok_premium`

<!-- Prose: a tokenizer is a compressor fitted to its training text; three
byte-level BPEs at V = 16,384 whose text weights the languages by
temperature_weights (restated from sec_lm_data_mixtures) with alpha 1, 0.3
and 0 on FineWeb-2's document counts (English is given German's count, an
assumption the text states); what each alpha does to each language's
premium on Belebele. -->

<!-- Prose: the reference tokenizer's pipeline in the Hugging Face
tokenizers library (normalizer, pre-tokenizer, model, decoder; the library's
pipeline guide): NFC, Qwen3's split rule with single digits, byte-level BPE,
the special tokens first; train_hf_bpe builds it for any vocabulary size and
special-token list, and the section uses it at 16,384, 65,536 and 32,768. -->

```{.python .input #tokenizers-bpe-def}
%%tab pytorch
#@save
R2_SPLIT_REGEX = (r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+"""
                  r"""|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)"""
                  r"""|\s+""")

#@save
def train_hf_bpe(texts, vocab_size, specials, regex=R2_SPLIT_REGEX,
                 min_frequency=2):
    """A byte-level BPE (Hugging Face tokenizers) trained on texts: NFC,
    the split regex, bytes as 256 base symbols, the special tokens first
    (ids 0, 1, ...), no BOS or EOS added on encode. Special-token strings
    inside a text are encoded as text (encode_special_tokens is set; the
    flag is not saved in tokenizer.json, so loaders set it again)."""
    from tokenizers import (Regex, Tokenizer, decoders, models, normalizers,
                            pre_tokenizers, trainers)
    tok = Tokenizer(models.BPE(dropout=None, byte_fallback=False,
                               unk_token=None))
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(regex), behavior='isolated', invert=False),
        pre_tokenizers.ByteLevel(add_prefix_space=False, trim_offsets=False,
                                 use_regex=False)])
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, min_frequency=min_frequency,
        special_tokens=list(specials), show_progress=False,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(texts, trainer=trainer)
    if tok.get_vocab_size() != vocab_size:
        raise ValueError(f'trained {tok.get_vocab_size()} pieces, not '
                         f'{vocab_size}: too little text for this vocabulary')
    tok.encode_special_tokens = True
    return tok
```

```{.python .input #tokenizers-premium}
%%tab pytorch
SIZES = {l: SLICE_META['fw2_docs'][l] for l in LANGS if l != 'eng_Latn'}
SIZES = {'eng_Latn': SIZES['deu_Latn'], **SIZES}  # English: German's count
SCRIPTS = {'Latin': r'\p{Latin}', 'Cyrillic': r'\p{Cyrillic}',
           'Arabic': r'\p{Arabic}', 'Devanagari': r'\p{Devanagari}',
           'Han': r'\p{Han}', 'Kana': r'[\p{Hiragana}\p{Katakana}]',
           'Thai': r'\p{Thai}', 'Hangul': r'\p{Hangul}'}

def script_counts(tok):
    """Pieces per script: a piece counts for a script when every letter
    in it belongs to that script."""
    out = collections.Counter()
    for b in d2l._vocab_bytes(tok):
        try:
            letters = regex.findall(r'\p{L}', b.decode('utf-8'))
        except (AttributeError, UnicodeDecodeError):  # special, partial
            continue
        for name, rx in SCRIPTS.items():
            if letters and all(regex.fullmatch(rx, c) for c in letters):
                out[name] += 1
    return out

ALPHA_TOK, WEIGHTS, rows = {}, {}, {}
for a in ALPHAS:
    w = d2l.temperature_weights(SIZES, a)
    budget = {l: w[l] / max(w.values()) * sum(len(t.encode())
                                              for t in TRAIN[l])
              for l in LANGS}  # the most weighted language uses all of it
    texts = [t for l in LANGS for t in first_bytes(TRAIN[l], budget[l])]
    tic = time.time()
    ALPHA_TOK[a] = d2l.train_hf_bpe(texts, V_SMALL, ['<|endoftext|>'])
    WEIGHTS[a] = w
    rows[a] = dict(MB=sum(len(t.encode()) for t in texts) / 1e6,
                   seconds=time.time() - tic, **script_counts(ALPHA_TOK[a]))
say(pd.DataFrame(WEIGHTS).loc[list(LANGS)].round(4).rename(
    columns=lambda a: f'weight, alpha={a}'))
say(pd.DataFrame(rows).T.round(1).rename_axis('alpha'))
FERT_A = d2l.fertility_table(
    {f'alpha={a}': encoder(t) for a, t in ALPHA_TOK.items()}, BELE)
PREM_A = FERT_A.pivot(index='lang', columns='encoder',
                      values='premium').loc[list(LANGS)]
say(PREM_A.round(2))
for l in ('swh_Latn', 'hin_Deva', 'tha_Thai'):  # C3: the smallest corpora
    p = [PREM_A.loc[l, f'alpha={a}'] for a in ALPHAS]
    say(f'{l}: premium ' + ' -> '.join(f'{x:.2f}' for x in p) +
        (' (falls as alpha falls)' if p[0] > p[1] > p[2] else
         ' (not monotone in alpha)'))
```

<!-- Prose: the comparison that is fair across tokenizers is bits per byte
(eq_lm_tok_bpb: bpb = L / (b-bar ln 2), restating subsec_bits-per-byte and
eq_mdl-bpc); three tokenizers at V = 16,384 trained on the same 45 MB
(SentencePiece unigram, SentencePiece BPE, the byte-level BPE with
alpha = 0), one small LM per tokenizer and seed for one epoch over the same
text (a byte budget, so token counts differ and are recorded), the 20.4
recipe and learning rate; every result here is at T1 scale. -->

```{.python .input #tokenizers-small-lms-def}
%%tab pytorch
def fingerprint(tok):
    if hasattr(tok, 'id_to_piece'):
        return d2l._sp_fingerprint(tok)
    return hashlib.sha256(tok.to_str(pretty=False).encode()).hexdigest()

def token_ids(tok, texts):
    if hasattr(tok, 'id_to_piece'):
        return tok.encode(list(texts))
    return d2l._doc_token_ids(tok, texts)

def eos_id(tok):
    return (tok.eos_id() if hasattr(tok, 'eos_id')
            else tok.token_to_id('<|endoftext|>'))

def as_pool(tok, texts):
    """Every text's ids followed by EOS, in one uint16 array."""
    return np.array([i for ids in token_ids(tok, texts)
                     for i in ids + [eos_id(tok)]], dtype=np.uint16)

@torch.no_grad()
def doc_nll(model, docs, eos, cfg, batch=32):
    """Summed NLL (nats) of every document, scored on its own after one
    EOS of context, in windows of seq_len tokens that advance by half a
    window (each token scored once, the later ones with at least half a
    window of context)."""
    L, windows = cfg.seq_len, []
    for k, ids in enumerate(docs):
        seq, done = [eos] + list(ids), 0  # targets seq[1:done + 1] scored
        while done < len(seq) - 1:
            end = min(L if done == 0 else done + L // 2, len(seq) - 1)
            windows.append((k, seq[max(end - L, 0):end + 1], end - done))
            done = end
    windows.sort(key=lambda w: len(w[1]))
    nll, model = np.zeros(len(docs)), model.eval()
    device = next(model.parameters()).device
    for a in range(0, len(windows), batch):
        ws = windows[a:a + batch]
        T = max(len(x) for _, x, _ in ws)
        X = torch.tensor([x + [0] * (T - len(x)) for _, x, _ in ws],
                         device=device)
        lp = d2l._per_token_nll(model, X, cfg).double().cpu().numpy()
        for r, (k, x, m) in enumerate(ws):
            nll[k] += lp[r, len(x) - 1 - m:len(x) - 1].sum()
    return nll

def lm_config(lr):
    """The small-LM runs' configuration, as their results file stores it."""
    return json.loads(json.dumps(dict(
        proxy=TOK_PROXY._replace(lr=lr)._asdict(), lr_from='lm-mix-runs-t1-v1',
        seeds=SEEDS, epochs=LM_EPOCHS, log_every=LOG_EVERY,
        vocab_size=V_SMALL)))

def train_small_lm(pool, seed, lr, stop_tokens=None):
    """One small LM on a token pool: one epoch (or stop_tokens); returns
    train_proxy_lm's record, the model and its configuration."""
    cfg = TOK_PROXY._replace(lr=lr)
    per_step = cfg.batch * cfg.seq_len
    tokens = int(LM_EPOCHS * (len(pool) - 1)) // per_step * per_step
    with tempfile.TemporaryDirectory() as d:
        ckpt = os.path.join(d, 'last.pt')
        out = d2l.train_proxy_lm(
            {'slice': pool}, {'slice': 1.0}, tokens, cfg, seed=seed,
            log_every=LOG_EVERY, stop_tokens=stop_tokens, ckpt=ckpt,
            ckpt_every=10**9, device=DEVICE)
        model = d2l._proxy_model(cfg, seed).to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE,
                                         weights_only=False)['model'])
    return out, model, cfg
```

<!-- Prose: the three tokenizers at V = 16,384 on the same 45 MB
(SentencePiece unigram, SentencePiece BPE, and the byte-level BPE with
alpha = 0, which was trained on exactly this text above): their token
counts for the same bytes, and the fingerprints with which a normal build
checks that its tokenizers are the runs'. -->

```{.python .input #tokenizers-small-lms-data}
%%tab pytorch
LM_TEXT = [t for l in LANGS for t in TRAIN[l]]
tic = time.time()
LM_TOKS = {'SP unigram': d2l.sp_train(LM_TEXT, 'unigram', V_SMALL),
           'SP BPE': d2l.sp_train(LM_TEXT, 'bpe', V_SMALL),
           'HF BPE': ALPHA_TOK[0.0]}
say(f'two SentencePiece models trained in {time.time() - tic:.0f} s')
POOLS = {k: as_pool(t, LM_TEXT) for k, t in LM_TOKS.items()}
EVAL_IDS = {k: {'heldout': {l: token_ids(t, HELD[l]) for l in LANGS},
                'belebele': {l: token_ids(t, BELE[l]) for l in LANGS}}
            for k, t in LM_TOKS.items()}
N_BYTES = {'heldout': {l: [len(x.encode()) for x in HELD[l]] for l in LANGS},
           'belebele': {l: [len(x.encode()) for x in BELE[l]]
                        for l in LANGS}}
TOK_INFO = {k: dict(fingerprint=fingerprint(t), train_tokens=len(POOLS[k]),
                    eos=eos_id(t), n_tokens={s: {l: [len(x) for x in v]
                                                 for l, v in d.items()}
                                             for s, d in EVAL_IDS[k].items()})
            for k, t in LM_TOKS.items()}
TEXT_HASH = d2l.docs_hash([(str(i), t) for i, t in enumerate(LM_TEXT)])
say(f'{sum(len(t.encode()) for t in LM_TEXT) / 1e6:.1f} MB of training '
    'text:', ', '.join(f'{k} {len(p):,} tokens' for k, p in POOLS.items()))
```

<!-- Prose: one small LM per tokenizer and seed for one epoch over that
text (a byte budget, so the runs read different numbers of tokens), the
20.4 recipe and learning rate; every result here is at T1 scale. A normal
build reads the registered runs, checks that they used this build's
tokenizers, text and configuration, and retrains the start of one run. -->

```{.python .input #tokenizers-small-lms}
%%tab pytorch
if 'tok-lm' in RERUN:
    with open(d2l.lm_artifact('lm-mix-runs-t1-v1')) as f:
        LR = json.load(f)['lr_choice']['chosen']
    runs = []
    for name in LM_TOKS:
        for seed in SEEDS:
            out, model, cfg = train_small_lm(POOLS[name], seed, LR)
            nll = {s: {l: np.round(doc_nll(model, v, TOK_INFO[name]['eos'],
                                           cfg), 4).tolist()
                       for l, v in d.items()}
                   for s, d in EVAL_IDS[name].items()}
            runs.append(dict(
                tokenizer=name, seed=seed, nll=nll, log=out['log'],
                **{k: out[k] for k in ('steps', 'n_params', 'throughput',
                                       'peak_mem', 'seconds')}))
            say(f'{name}, seed {seed}: {out["steps"]} steps of '
                f'{cfg.batch * cfg.seq_len:,} tokens, {out["seconds"]:.0f}'
                f' s, final loss {out["log"][-1][2]:.3f}', d2l.lm_peak())
            del model
            d2l.lm_free()
    TOKLM = json.loads(json.dumps(dict(
        schema='d2l-tok-lm/1', tier=TIER, text_hash=TEXT_HASH,
        config=lm_config(LR), tokenizers=TOK_INFO, n_bytes=N_BYTES,
        runs=runs)))  # as the file will read back
    d2l.lm_save_rerun('tok-lm.json', lambda p: write_json(p, TOKLM))
elif RESULTS is None:
    raise KeyError(f'{RES_KEY} is not registered: regenerate it with '
                   'D2L_RERUN=1, upload and register it')
else:
    TOKLM = RESULTS['tok-lm']
LR = TOKLM['config']['proxy']['lr']
# The runs used this build's tokenizers, text, bytes and configuration
assert TOKLM['config'] == lm_config(LR) and TOKLM['tier'] == TIER
assert TOKLM['text_hash'] == TEXT_HASH
assert TOKLM['n_bytes'] == json.loads(json.dumps(N_BYTES))
for name, info in json.loads(json.dumps(TOK_INFO)).items():
    assert TOKLM['tokenizers'][name] == info, f'{name} differs from its run'
say(f'{len(TOKLM["runs"])} runs {TIER}, learning rate {LR:g} (from '
    f'{TOKLM["config"]["lr_from"]})')
if GPU and 'tok-lm' not in RERUN:  # live check: the start of one run
    out, model, _ = train_small_lm(POOLS['HF BPE'], 0, LR,
                                   stop_tokens=LIVE_TOKENS)
    reg = {s: l for s, _, l, _ in next(
        r for r in TOKLM['runs'] if r['tokenizer'] == 'HF BPE'
        and r['seed'] == 0)['log']}
    diff = max(abs(l - reg[s]) for s, _, l, _ in out['log'] if s in reg)
    say(f'live check, HF BPE seed 0, first {LIVE_TOKENS:,} tokens: '
        f'largest loss difference {diff:.4f} nats', d2l.lm_peak())
    assert diff < 0.02
    del model
    d2l.lm_free()
```

<!-- Prose: D1 numerically: three ways to the same bits per byte, and why
perplexity per token cannot be compared across tokenizers (the pitfall). -->

```{.python .input #tokenizers-bpb-identity}
%%tab pytorch
def run_of(name, seed):
    return next(r for r in TOKLM['runs']
                if r['tokenizer'] == name and r['seed'] == seed)

LN2, rows = math.log(2), {}
for name in LM_TOKS:  # 10 held-out documents per language: 100
    nll = np.concatenate([run_of(name, 0)['nll']['heldout'][l][:10]
                          for l in LANGS])
    N = np.concatenate([TOKLM['tokenizers'][name]['n_tokens']['heldout'][l]
                        [:10] for l in LANGS])
    B = np.concatenate([TOKLM['n_bytes']['heldout'][l][:10] for l in LANGS])
    L, bbar = nll.sum() / N.sum(), B.sum() / N.sum()
    ways = (L / (bbar * LN2), nll.sum() / (B.sum() * LN2),
            np.sum(B / B.sum() * nll / (B * LN2)))
    assert max(ways) - min(ways) < 1e-9
    rows[name] = dict(loss_per_token=L, perplexity=math.exp(L),
                      bytes_per_token=bbar, bpb=ways[0],
                      max_difference=max(ways) - min(ways))
say(f'100 held-out documents, seed 0 {TIER}')
say(pd.DataFrame(rows).T.round(4))
```

<!-- Prose: the plot below puts each language at its premium under each
tokenizer and its bits per byte on the Belebele passages; what C4 says,
in the words the numbers allow. -->

```{.python .input #tokenizers-bpb-premium}
%%tab pytorch
def lang_bpb(name, seed, split, lang):
    nll = np.sum(run_of(name, seed)['nll'][split][lang])
    return nll / (np.sum(TOKLM['n_bytes'][split][lang]) * LN2)

TOKS = list(LM_TOKS)
tab = {}
for name in TOKS:
    n = {l: sum(TOKLM['tokenizers'][name]['n_tokens']['belebele'][l])
         for l in LANGS}
    for l in LANGS:
        b = [lang_bpb(name, s, 'belebele', l) for s in SEEDS]
        h = [lang_bpb(name, s, 'heldout', l) for s in SEEDS]
        tab[name, l] = dict(premium=n[l] / n['eng_Latn'],
                            bpb_belebele=np.mean(b), seed_range=np.ptp(b),
                            bpb_heldout=np.mean(h), heldout_range=np.ptp(h))
TAB = pd.DataFrame(tab).T.rename_axis(['tokenizer', 'lang'])
say(f'Belebele and held-out bits per byte {TIER}')
WIDE = TAB.unstack(0).loc[list(LANGS)].round(3)
for part in (['premium', 'bpb_belebele', 'seed_range'],
             ['bpb_heldout', 'heldout_range']):
    say(WIDE[part].to_string())

def passage_bpb(name):
    """Per-passage bits per byte, averaged over languages and seeds."""
    B = np.array([TOKLM['n_bytes']['belebele'][l] for l in LANGS], float)
    return np.mean([np.array([run_of(name, s)['nll']['belebele'][l]
                              for l in LANGS]) / (B * LN2) for s in SEEDS],
                   axis=(0, 1))

macro = {name: passage_bpb(name) for name in TOKS}
for a, b in itertools.combinations(TOKS, 2):
    r = d2l.paired_bootstrap(macro[a], macro[b])
    say(f'macro bits per byte, {a} - {b}: {r.mean:+.4f} '
        f'[{r.ci[0]:+.4f}, {r.ci[1]:+.4f}] over {r.n} passages')
bpb = TAB.bpb_belebele.unstack(0).loc[list(LANGS)]
prem = TAB.premium.unstack(0).loc[list(LANGS)]
rng_ = TAB.seed_range.unstack(0).loc[list(LANGS)]
for l in LANGS:  # C4, per language: unigram against BPE at equal V
    d = bpb.loc[l, 'SP unigram'] - bpb.loc[l, 'SP BPE']
    noise = rng_.loc[l, ['SP unigram', 'SP BPE']].max()
    say(f'{l}: unigram - BPE {d:+.3f} bits per byte (seed range '
        f'{noise:.3f}): ' + ('not distinguished' if abs(d) <= noise
                             else 'differs'))
effect = (bpb['SP unigram'] - bpb['SP BPE']).abs() / bpb['SP BPE']
spread = prem.max() / prem.min()  # per tokenizer, across languages
say(f'unigram against BPE moves a language by at most {effect.max():.1%} '
    f'of its bits per byte ({effect.idxmax()}); under one tokenizer the '
    f'premium spans a factor of {spread.min():.2f}-{spread.max():.2f} '
    'across languages')

d2l.plot([prem[t].to_numpy() for t in TOKS] +
         [prem.loc[l].to_numpy() for l in LANGS],
         [bpb[t].to_numpy() for t in TOKS] +
         [bpb.loc[l].to_numpy() for l in LANGS],
         xlabel='premium (tokens relative to English)',
         ylabel='bits per byte (Belebele)', legend=TOKS,
         fmts=('o', 's', '^') + ('C7-',) * len(LANGS), figsize=(5, 3.5))
for l in LANGS:
    d2l.plt.gca().annotate(l[:3], (prem.loc[l].max(), bpb.loc[l].iloc[-1]),
                           xytext=(4, 0), textcoords='offset points',
                           fontsize=8)
```

## Vocabulary Size
:label:`subsec_lm_tok_vocab`

<!-- Prose: the per-byte compute model C(V) = 6 (A + V d) / b-bar(V)
(eq_lm_tok_vstar): five lines from df/dV = 0 to the elasticity of
compression equal to the softmax share, V* = eps A / (d (1 - eps)) at
constant eps; the measured b-bar(V) from prefixes of one BPE (the prefix
property of sec_text-sequence); Tao et al. (Scaling Laws with Vocabulary)
point the same way, larger models deserve larger vocabularies, and also
let the vocabulary change the loss, which this model holds fixed; where
released models sit; the practical rules (multiples of 64 or 128, reserved
ids, tied embeddings). -->

```{.python .input #tokenizers-vocab}
%%tab pytorch
R2 = d2l.r2_tokenizer()
R2_RAW = d2l.r2_tokenizer(raw=True)
SHARDS = d2l.fwedu_shards('100m')

def shard_texts(shards, n_bytes, start=0):
    """Decoded documents of a shard set, in order, until n_bytes bytes."""
    out, b = [], 0
    for _, ids in shards.iter_docs(start):
        if b >= n_bytes:
            break
        out.append(R2_RAW.decode(ids[:-1].tolist()))
        b += len(out[-1].encode())
    return out

tic = time.time()
ENG = shard_texts(SHARDS, sum(len(t.encode()) for t in LM_TEXT))
BIG = {'English': d2l.train_hf_bpe(ENG, 65_536, ['<|endoftext|>']),
       'ten languages': d2l.train_hf_bpe(LM_TEXT, 65_536,
                                         ['<|endoftext|>'])}
say(f'two BPEs of 65,536 pieces, on {len(ENG):,} English documents and '
    f'on the slice ({sum(len(t.encode()) for t in ENG) / 1e6:.0f} and '
    f'{sum(len(t.encode()) for t in LM_TEXT) / 1e6:.0f} MB): '
    f'{time.time() - tic:.0f} s')
GRID = [2**k for k in range(10, 17)]
EVAL = {'English': HELD['eng_Latn'],
        'ten languages': [t for l in LANGS for t in HELD[l]]}

def bytes_per_token(tok, texts):
    return (sum(len(t.encode()) for t in texts)
            / sum(len(e.ids) for e in tok.encode_batch(texts)))

BBAR = pd.DataFrame({k: [bytes_per_token(d2l._bpe_prefix(BIG[k], V),
                                         EVAL[k]) for V in GRID]
                     for k in BIG}, index=GRID)
# log b-bar as a smooth monotone function of log V: its value and its
# slope (the elasticity) come from the same curve
LOG_BBAR = {k: PchipInterpolator(np.log(GRID), np.log(BBAR[k]))
            for k in BIG}
bbar = lambda k, V: np.exp(LOG_BBAR[k](np.log(V)))
elasticity = lambda k, V: LOG_BBAR[k].derivative()(np.log(V))
d, A = 1024, 12 * 24 * 1024**2   # the reference transformer of D3
share = lambda V: V * d / (A + V * d)
cost = lambda V, bb: 6 * (A + V * d) / bb  # FLOP per byte
say(pd.concat([BBAR.round(3).add_prefix('bytes/token, '),
               pd.DataFrame({k: elasticity(k, np.array(GRID, float))
                             for k in BIG}, index=GRID
                            ).round(3).add_prefix('elasticity, '),
               pd.Series({V: share(V) for V in GRID},
                         name='softmax share').round(3)], axis=1))

V = np.geomspace(1_000, 200_000, 20_001)  # D3 check on b-bar = c V^0.1
v_star = 0.1 * A / (d * 0.9)
v_grid = V[np.argmin(cost(V, V**0.1))]
assert abs(v_grid / v_star - 1) < 0.01
say(f'synthetic b-bar = c V^0.1: grid minimum {v_grid:,.0f}, formula '
    f'{v_star:,.0f}')
FINE = np.geomspace(GRID[0], GRID[-1], 2_001)
for k in BIG:  # the measured minimum and the elasticity crossing
    v_min = FINE[np.argmin(cost(FINE, bbar(k, FINE)))]
    gap = elasticity(k, FINE) - share(FINE)
    cross = FINE[np.argmax(gap < 0)] if (gap < 0).any() else None
    say(f'{k}: cost minimum at V = {v_min:,.0f}; elasticity falls to '
        'the softmax share at ' + (f'V = {cross:,.0f}' if cross else
                                   'no V in the grid'))
```

<!-- Prose: where released tokenizers sit: each one's bytes per token on
the same English held-out text, and the share of per-byte compute its
embedding and softmax take at its own width (parameter counts from the
pinned configurations; no weights are loaded); the reference tokenizer at
the width of the next chapter's reference model. -->

```{.python .input #tokenizers-vocab-released}
%%tab pytorch
def released_params(model_id):
    """(vocabulary rows, width, parameters) from the pinned configuration,
    on the meta device: no weights are loaded."""
    cfg = transformers.AutoConfig.from_pretrained(
        model_id, revision=d2l.HF_REVISIONS[model_id]).get_text_config()
    with torch.device('meta'), warnings.catch_warnings():
        warnings.simplefilter('ignore')  # kernels a CPU cannot run
        m = transformers.AutoModelForCausalLM.from_config(cfg)
    return cfg.vocab_size, cfg.hidden_size, d2l.num_params(m)

FIT = {'GPT-2': (50_257, 768, 124_439_808),  # OpenAI's released config
       'Qwen3': released_params('Qwen/Qwen3-0.6B-Base'),
       'Qwen3.5': released_params('Qwen/Qwen3.5-0.8B-Base'),
       'reference': (32_768, 768, 138_442_752),  # chap. 21's model
       'o200k_base': (200_019, None, None)}
TOKENIZERS = {**RELEASED, 'reference': R2}
rows = {}
for k, (Vk, dk, n) in FIT.items():
    bb = sum(len(t.encode()) for t in HELD['eng_Latn']) / sum(map(
        len, d2l._doc_token_ids(TOKENIZERS[k], HELD['eng_Latn'])))
    row = dict(V=Vk, bytes_per_token=bb)
    if dk:
        Ak = n - Vk * dk  # tied embeddings: one V x d matrix
        row.update(d=dk, non_embedding_M=Ak / 1e6,
                   softmax_share=Vk * dk / (Ak + Vk * dk),
                   GFLOP_per_kB=6 * (Ak + Vk * dk) / bb * 1e3 / 1e9)
    rows[k] = row
say('English held-out text; tied embeddings')
say(pd.DataFrame(rows).T.round(3))

fig, axes = d2l.plt.subplots(1, 2, figsize=(7, 2.8))
d2l.plot(GRID, [BBAR[k].to_numpy() for k in BIG], 'vocabulary size V',
         'bytes per token', legend=list(BIG), xscale='log', axes=axes[0],
         fmts=('-o', '--s'))
for k in TOKENIZERS:
    axes[0].plot(rows[k]['V'], rows[k]['bytes_per_token'], 'k^',
                 markersize=4)
    axes[0].annotate(k, (rows[k]['V'], rows[k]['bytes_per_token']),
                     xytext=(3, -9), textcoords='offset points', fontsize=7)
c = cost(FINE, bbar('English', FINE)) / 1e9
d2l.plot(FINE, c, 'vocabulary size V', 'GFLOP per byte (d = 1024)',
         xscale='log', axes=axes[1])
_ = axes[1].plot(FINE[np.argmin(c)], c.min(), 'ko')
```

### Numbers, Whitespace and Code
:label:`subsec_lm_tok_digits`

<!-- Prose: three digit rules in released regexes (GPT-2 lets BPE carve
numbers by frequency, o200k_base caps runs at three digits, Qwen3, Qwen3.5
and the reference tokenizer split every digit); indentation and newline
runs on the book's own library code (d2l/torch.py, MIT-0). -->

```{.python .input #tokenizers-digits-code}
%%tab pytorch
VB = {k: d2l._vocab_bytes(t) for k, t in TOKENIZERS.items()}

def pieces(k, s):
    return '|'.join(VB[k][i].decode('utf-8', 'replace')
                    for i in d2l._doc_token_ids(TOKENIZERS[k], [s])[0])

say(pd.DataFrame({s: {k: pieces(k, s) for k in TOKENIZERS}
                  for s in ('12345678', '3.14159', '1,234,567')}))
NUMBERS = ('In 2019 the plant produced 1,234,567 tonnes, up 12.5% from '
           '1,097,393 in 2018; the price rose from $3,456.78 to $3,901.25 '
           'per unit, and 47 of its 365 days ran at 98.6% capacity.\n'
           '| year | output | share |\n| 2016 | 864,512 | 31.4% |\n'
           '| 2017 | 970,031 | 33.9% |\n| 2018 | 1,097,393 | 36.2% |\n')
with open(d2l.__file__, encoding='utf-8') as f:
    CODE = ''.join(itertools.islice(f, 2000))
rows = {}
for k, tok in TOKENIZERS.items():
    ids = d2l._doc_token_ids(tok, [NUMBERS])[0]
    digit_tokens = [i for i in ids if any(48 <= c <= 57 for c in VB[k][i])]
    code = d2l._doc_token_ids(tok, [CODE])[0]
    spaces = [len(b) for b in VB[k] if b and b.strip(b' ') == b'']
    rows[k] = dict(digits_per_token=sum(c.isdigit() for c in NUMBERS)
                   / len(digit_tokens),
                   code_tokens_per_line=len(code) / CODE.count('\n'),
                   code_bytes_per_token=len(CODE.encode()) / len(code),
                   longest_space_token=max(spaces, default=0))
say(pd.DataFrame(rows).T.round(2))
```

## Extending a Trained Vocabulary
:label:`subsec_lm_tok_extend`

<!-- Prose: Qwen3-0.6B-Base fragments Hindi (its premium in the opening
table) while Qwen3.5's larger vocabulary and its regex, which keeps
combining marks with their letters, do not; the motivating contrast. Add
4,096 Hindi pieces; extend_vocab with three initializations (random from
the model's initializer, the mean of the old rows with a little noise, the
mean of the old embeddings of the pieces the old tokenizer splits the new
token into); 19.1's view of initialization as placement in embedding space
(one sentence); FOCUS and WECHSEL place new rows by cross-lingual
similarity and are described, not implemented. -->


```{.python .input #tokenizers-extend-def}
%%tab pytorch
#@save
def extend_vocab(model, tokenizer, new_pieces, init='random', seed=0):
    """Add new_pieces to a Transformers tokenizer (as added tokens, which
    are matched before the model's own segmentation) and rows for them to
    the model's input and, if untied, output matrices, initialized by
    init: 'random' (normal, the config's initializer_range), 'mean' (the
    mean of the old rows plus noise of 1% of their standard deviation, so
    that no two rows are equal) or 'decomp' (the mean of the old rows of
    the pieces the old tokenizer splits the new piece into). Returns
    (model, tokenizer) with config.vocab_size updated."""
    if init not in ('random', 'mean', 'decomp'):
        raise ValueError(f'unknown init {init!r}')
    pieces, n_old = list(new_pieces), len(tokenizer)
    parts = tokenizer(pieces, add_special_tokens=False)['input_ids']
    if tokenizer.add_tokens(pieces) != len(pieces):
        raise ValueError('a new piece is in the vocabulary already')
    out = model.get_output_embeddings()
    tied = out is None or out.weight is model.get_input_embeddings().weight
    old = [model.get_input_embeddings().weight] + ([] if tied
                                                   else [out.weight])
    old = [W.detach()[:n_old].float().cpu() for W in old]
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    mats = [model.get_input_embeddings().weight] + (
        [] if tied else [model.get_output_embeddings().weight])
    g = torch.Generator().manual_seed(seed)
    std = model.config.get_text_config().initializer_range
    k, d = len(pieces), old[0].shape[1]
    with torch.no_grad():
        for W, W0 in zip(mats, old):
            if init == 'random':
                new = torch.randn(k, d, generator=g) * std
            elif init == 'mean':
                new = W0.mean(0) + 0.01 * W0.std() * torch.randn(
                    k, d, generator=g)
            else:
                new = torch.stack([W0[ids].mean(0) for ids in parts])
            W[n_old:n_old + k] = new.to(W.device, W.dtype)
    model.config.get_text_config().vocab_size = len(tokenizer)
    return model, tokenizer
```

<!-- Prose: the new pieces come from a byte-level BPE of 8,192 pieces on
the 30,000 Hindi documents, with a split rule that keeps letters and their
combining marks together and never puts a space in a piece (a piece with an
ASCII character would change how English is segmented); the first 4,096
pieces by merge order that hold at least two Devanagari characters and are
not already in Qwen3's vocabulary. -->

```{.python .input #tokenizers-extend-build}
%%tab pytorch
HI_REGEX = r"""[\p{L}\p{M}]+|\p{N}|[^\s\p{L}\p{M}\p{N}]+|\s+"""
HI_TRAIN = HINDI[HINDI.part == 'ext_train'].text.tolist()
HI_HELD = HINDI[HINDI.part == 'ext_heldout'].text.tolist()
tic = time.time()
hi_bpe = d2l.train_hf_bpe(HI_TRAIN, 8192, ['<|endoftext|>'], regex=HI_REGEX)
QWEN = RELEASED['Qwen3']
in_qwen, cand, NEW = QWEN.get_vocab(), 0, []
for i, b in enumerate(d2l._vocab_bytes(hi_bpe)):  # ids in merge order
    try:
        s = b.decode('utf-8')
    except (AttributeError, UnicodeDecodeError):  # special, partial
        continue
    if (len(b) > 1 and not any(c < 128 for c in b)
            and len(regex.findall(r'\p{Devanagari}', s)) >= 2):
        cand += 1
        if hi_bpe.id_to_token(i) not in in_qwen:
            NEW.append(s)
say(f'Hindi BPE in {time.time() - tic:.0f} s: {cand:,} Devanagari '
    f'pieces, {len(NEW):,} new to Qwen3; the first {N_NEW:,} are added')
assert len(NEW) >= N_NEW
NEW = NEW[:N_NEW]
EXT_TOK = copy.deepcopy(QWEN)
EXT_TOK.add_tokens(NEW)
ascii_lines = [l for l in lines(HELD['eng_Latn']) if l.isascii()][:1000]
assert len(ascii_lines) == 1000
assert (QWEN(ascii_lines, add_special_tokens=False)['input_ids']
        == EXT_TOK(ascii_lines, add_special_tokens=False)['input_ids'])
say(f'1,000 ASCII lines: the same ids before and after extension')
hb = sum(len(t.encode()) for t in HI_HELD)
for name, tok in (('Qwen3', QWEN), ('Qwen3 + Hindi', EXT_TOK),
                  ('Qwen3.5', RELEASED['Qwen3.5'])):
    n = sum(map(len, d2l._doc_token_ids(tok, HI_HELD)))
    say(f'{name}: {hb / n:.2f} bytes per token on 500 held-out Hindi '
        'documents')
say('new pieces:', ' '.join(NEW[:12]), '...', ' '.join(NEW[-6:]))
```

<!-- Prose: the step-0 diagnostics, before any training: how much
probability the new rows take from English text (it should be about zero;
much more means they hijack the softmax) and how much they get on Hindi
text, where about half of the next tokens are new ones (identical mean rows
give the new tokens one shared logit, so they can neither win mass nor be
told apart as inputs), Hindi bits per byte on held-out
documents and on Belebele's Hindi passages, English bits per byte (the
English ids are unchanged, so only the softmax normalizer can move it), and
the decomposition identity: with tied embeddings, a new token's logit is
the mean of its pieces' logits. -->

```{.python .input #tokenizers-extend-step0-def}
%%tab pytorch
@torch.no_grad()
def new_mass(model, tok, texts, lo):
    """Mean probability of the ids lo and above (the new tokens) over the
    positions of the first 256 tokens of each text."""
    out = []
    for ids in d2l._doc_token_ids(tok, texts):
        x = torch.tensor([ids[:256]], device=model.device)
        p = model(input_ids=x).logits[0].float().softmax(-1)
        out.append(p[:, lo:].sum(-1).mean().item())
    return np.array(out)

def bpb(model, tok, texts):
    return d2l.bits_per_byte(model, tok, texts, ctx_len=N_EXT['ctx_len'])

def step0(model, tok):
    """The step-0 diagnostics of one model and tokenizer: new-token mass on
    English and on Hindi contexts, Hindi, Belebele and English bits per
    byte, Hindi bytes per token, and per-passage Belebele bits per byte."""
    per = np.array([bpb(model, tok, [p]) for p in PASSAGES])
    w = np.array([len(p.encode()) for p in PASSAGES])
    k = N_EXT['contexts']
    en, hi = ((new_mass(model, tok, EN_DOCS[:k], LO),
               new_mass(model, tok, HI_DOCS[:k], LO)) if len(tok) > LO
              else (np.zeros(1), np.zeros(1)))
    n = sum(map(len, d2l._doc_token_ids(tok, HI_DOCS)))
    return dict(new_mass=float(en.mean()), new_mass_max=float(en.max()),
                hindi_new_mass=float(hi.mean()),
                hindi_bpb=bpb(model, tok, HI_DOCS),
                belebele_bpb=float((per * w).sum() / w.sum()),
                english_bpb=bpb(model, tok, EN_DOCS),
                hindi_bytes_per_token=sum(len(t.encode())
                                          for t in HI_DOCS) / n,
                per_passage=per.round(5).tolist())

@torch.no_grad()
def decomp_check(model, tok, n=64):
    """Largest |logit(new) - mean(logit(its pieces))| over the first n
    new tokens and the positions of two English contexts, and the
    rounding bound of the stored rows' dtype."""
    W = model.get_output_embeddings().weight
    parts = QWEN(NEW[:n], add_special_tokens=False)['input_ids']
    worst = 0.0
    for ids in d2l._doc_token_ids(tok, EN_DOCS[:2]):
        x = torch.tensor([ids[:128]], device=W.device)
        h = model.get_decoder()(input_ids=x).last_hidden_state[0].float()
        new = h @ W[LO:LO + n].float().T
        mean = torch.stack([(h @ W[p].float().T).mean(1) for p in parts], 1)
        tol = (1e-5 if W.dtype == torch.float32 else 2**-8) * (
            h.abs() @ W[LO:LO + n].float().abs().T) + 1e-5
        assert ((new - mean).abs() <= tol).all()
        worst = max(worst, (new - mean).abs().max().item())
    return worst
```

<!-- Prose: the four arms at step 0 (the unextended control and the three
initializations, seed 0) on the same texts: Hindi held-out documents,
Belebele's Hindi passages, English validation documents. -->

```{.python .input #tokenizers-extend-step0}
%%tab pytorch
BASE, _ = d2l.load_hf(EXT_MODEL, dtype=torch.float32 if SMALL
                      else torch.bfloat16)
VAL = d2l.fwedu_shards('val')
with gzip.open(os.path.join(VAL.root, 'val.jsonl.gz'), 'rt',
               encoding='utf-8') as f:
    VAL_TEXT = [json.loads(line) for line in f]
EN_DOCS = [r['text'] for r in VAL_TEXT[:N_EXT['english']]]
HI_DOCS, PASSAGES = HI_HELD[:N_EXT['hindi']], BELE['hin_Deva'][
    :N_EXT['passages']]
LO = len(QWEN)  # the first new id

tic = time.time()
STEP0 = {'control': step0(BASE, QWEN)}
for init in ('random', 'mean', 'decomp'):
    model, tok = d2l.extend_vocab(copy.deepcopy(BASE), copy.deepcopy(QWEN),
                                  NEW, init=init, seed=0)
    assert model.config.get_text_config().vocab_size == len(tok) == LO + N_NEW
    STEP0[init] = step0(model, tok)
    if init == 'decomp':
        say(f'decomposition identity: largest logit difference '
            f'{decomp_check(model, tok):.2e} ({model.dtype})')
    del model
    d2l.lm_free()
say(f'step 0 on {len(HI_DOCS)} Hindi documents, {len(PASSAGES)} Belebele '
    f'passages, {len(EN_DOCS)} English documents'
    + (' (the small CPU path)' if SMALL else ''),
    f'{time.time() - tic:.0f} s', d2l.lm_peak())
say(pd.DataFrame({k: {c: v for c, v in s.items() if c != 'per_passage'}
                  for k, s in STEP0.items()}).T.round(4))
per = {k: np.array(s['per_passage']) for k, s in STEP0.items()}
for init in ('mean', 'decomp'):  # C5 at step 0, over passages
    r = d2l.paired_bootstrap(per[init], per['random'])
    say(f'{init} - random, Belebele Hindi bits per byte: {r.mean:+.3f} '
        f'[{r.ci[0]:+.3f}, {r.ci[1]:+.3f}];',
        'lower than random' if r.ci[1] < 0 else 'not lower than random',
        '| new-token mass', 'below' if STEP0[init]['new_mass']
        < STEP0['random']['new_mass'] else 'NOT below', "random's")
```

<!-- Prose: continued training on Hindi (control: the unextended model
with its own tokenizer; and the three initializations), 150 steps of
8 x 1,024 tokens in two phases, the usual vocabulary-extension recipe
(author ruling R4, 2026-11-20): for 50 steps only the new rows train
(body and old rows frozen; the control has no new rows, so it waits),
then every parameter at a low rate; each phase warms up over 10 steps
and decays by a cosine; AdamW. Why not the spec's single phase at 1e-4:
re-warming an annealed base model at that rate made the control itself
worse at Hindi (0.627 -> 0.766 bits per byte after 10 steps, 2026-11-20
probe), so "the arms converge" would partly have been the control
degrading. The two rates were settled on Hindi documents outside the
training pool and outside HI_HELD (2026-11-21 probe), so no reported
metric was tuned: the new rows' rate 3e-3 had the lowest mean Hindi
bits per byte of the three initializations after the first phase (grid
3e-4 to 3e-2; at 3e-2 the new rows hijack English, 8-10 bits per byte);
the body rate 1e-5 is the largest at which the control ends within 1% of
its start on both Hindi and English (3e-6 to 1e-4; the mean over the four
arms alone kept falling up to the spec's 1e-4, where the control lost
0.04 bits per byte on Hindi and 0.07 on English). Two data-order seeds; every arm reads the same
documents, so the extended arms see more text per step; bf16 autocast
over fp32 weights, gradient checkpointing, and the loss in chunks of
1,024 positions so the 8,192 x 155,765 logit matrix never exists
(D2L_RERUN=tok-extend, measured time and peak printed). -->

```{.python .input #tokenizers-extend-train-def}
%%tab pytorch
def hindi_docs_for(tok, n_tokens):
    """The first Hindi training documents whose tokens under tok reach
    n_tokens."""
    k, n = 0, 0
    for ids in d2l._doc_token_ids(tok, HI_TRAIN[:5000]):
        k, n = k + 1, n + len(ids) + 1
        if n >= n_tokens:
            return HI_TRAIN[:k]
    raise ValueError('too few Hindi documents for the training pool')

def chunk_ce(h, W, y):
    """Summed cross-entropy of one chunk of positions; checkpointed, so
    its logits are recomputed in the backward pass, never stored."""
    with torch.autocast(h.device.type, dtype=torch.bfloat16,
                        enabled=h.device.type == 'cuda'):
        z = h @ W.T
    return F.cross_entropy(z.float(), y, reduction='sum')

def phase_lr(step):
    """(first phase?, learning rate) at step: the new rows alone at
    embed_lr for the first embed_steps steps, then every parameter at lr;
    each phase warms up over warmup steps and decays by a cosine."""
    E = EXT_TRAIN
    first = step < E['embed_steps']
    s, n = ((step, E['embed_steps']) if first else
            (step - E['embed_steps'], E['steps'] - E['embed_steps']))
    sched = d2l.ProxyCfg(1, 1, 1, warmup_frac=E['warmup'] / n,
                         final_lr_frac=E['final_lr_frac'])
    return first, (E['embed_lr'] if first else E['lr']) * d2l._lr_at(
        s, n, sched)

def train_ext(arm, seed, docs):
    """Continued training of one arm on docs: the new rows alone (body and
    old rows frozen; the control has none, so it waits), then everything
    at a low rate. Returns the Hindi curve, English before and after, the
    batch losses (None while the control waits) and the cost."""
    E = EXT_TRAIN
    model, tok = d2l.load_hf(EXT_MODEL, dtype=torch.float32)
    n_old = len(tok)
    if arm != 'control':
        model, tok = d2l.extend_vocab(model, tok, NEW, init=arm, seed=seed)
    eos = d2l._lm_boundary_id(tok)
    pool = np.array([i for ids in d2l._doc_token_ids(tok, docs)
                     for i in ids + [eos]], dtype=np.uint32)
    stream = d2l._ProxyStream({'hi': pool}, {'hi': 1.0}, E['steps'] * E[
        'batch'] * E['seq_len'], E['seq_len'], E['batch'], seed)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    W = model.get_output_embeddings().weight
    assert W is model.get_input_embeddings().weight, 'tied embeddings'
    for p in model.parameters():  # first phase: the new rows only
        p.requires_grad_(p is W)
    rows = torch.arange(len(W), device=W.device)[:, None] >= n_old
    hook = W.register_hook(lambda g: g * rows)
    opt = torch.optim.AdamW([W], lr=E['embed_lr'], betas=(0.9, 0.95),
                            weight_decay=0.0)
    curve, losses = [], []
    evaluate = lambda texts: bpb(model.eval(), tok, texts)
    english = {0: evaluate(EN_DOCS)}
    curve.append((0, evaluate(HI_HELD[:E['curve_docs']])))
    tic = time.time()
    for step in range(E['steps']):
        if step == E['embed_steps']:  # second phase: every parameter
            hook.remove()
            for p in model.parameters():
                p.requires_grad_(True)
            opt = torch.optim.AdamW(model.parameters(), lr=E['lr'],
                                    betas=(0.9, 0.95), weight_decay=0.0)
        first, lr = phase_lr(step)
        if first and arm == 'control':  # no new rows to train
            losses.append(None)
        else:
            model.train()
            for g in opt.param_groups:
                g['lr'] = lr
            X = torch.from_numpy(stream.batch(step)[0]).to(model.device)
            total = 0.0
            for Xm in X.split(E['micro_batch']):
                with torch.autocast(X.device.type, dtype=torch.bfloat16,
                                    enabled=X.device.type == 'cuda'):
                    H = model.get_decoder()(
                        input_ids=Xm[:, :-1]).last_hidden_state
                h, y = H.flatten(0, 1), Xm[:, 1:].flatten()
                loss = sum(checkpoint(
                    chunk_ce, h[i:i + E['loss_chunk']], W,
                    y[i:i + E['loss_chunk']], use_reentrant=False)
                    for i in range(0, len(y), E['loss_chunk'])) / (
                        X.shape[0] * E['seq_len'])
                loss.backward()
                total += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            losses.append(round(total, 4))
        if step + 1 in E['eval_at']:
            curve.append((step + 1, evaluate(HI_HELD[:E['curve_docs']])))
    english[E['steps']] = evaluate(EN_DOCS)
    out = dict(arm=arm, seed=seed, curve=curve, english=english,
               losses=losses, seconds=time.time() - tic,
               train_tokens=len(pool), peak=d2l.lm_peak())
    del model, opt
    d2l.lm_free()
    return out
```

<!-- Prose: the eight runs (four arms, two data-order seeds), read from the
registered results unless regenerated: Hindi bits per byte along training
and English bits per byte before and after. -->

```{.python .input #tokenizers-extend-train}
%%tab pytorch
if 'tok-extend' in RERUN:
    docs = hindi_docs_for(QWEN, EXT_TRAIN['pool_tokens'])
    runs = []
    for arm in ('control', 'random', 'mean', 'decomp'):
        for seed in EXT_TRAIN['seeds']:
            runs.append(train_ext(arm, seed, docs))
            r = runs[-1]
            say(f'{arm}, seed {seed}: Hindi bits per byte '
                + ' -> '.join(f'{v:.3f}' for _, v in r['curve'])
                + f', {r["seconds"]:.0f} s, {r["peak"]}')
    TOKEXT = dict(schema='d2l-tok-extend/1', tier=TIER, model=EXT_MODEL,
                  revision=d2l.HF_REVISIONS[EXT_MODEL], config=EXT_TRAIN,
                  n_eval=N_EXT, new_pieces=NEW, n_docs=len(docs),
                  step0=STEP0, runs=runs)
    TOKEXT = json.loads(json.dumps(TOKEXT))
    d2l.lm_save_rerun('tok-extend.json', lambda p: write_json(p, TOKEXT))
elif RESULTS is None:
    raise KeyError(f'{RES_KEY} is not registered: regenerate it with '
                   'D2L_RERUN=1, upload and register it')
else:
    TOKEXT = RESULTS['tok-extend']
assert TOKEXT['new_pieces'] == NEW, 'the new pieces differ from the runs'
assert {k: TOKEXT[k] for k in ('tier', 'model', 'revision', 'config')} == (
    json.loads(json.dumps(dict(tier=TIER, model=EXT_MODEL, config=EXT_TRAIN,
                               revision=d2l.HF_REVISIONS[EXT_MODEL])))), (
    'the registered runs used another model or configuration')
ARMS = ('control', 'random', 'mean', 'decomp')
steps = [s for s, _ in TOKEXT['runs'][0]['curve']]
curves = {a: np.array([[v for _, v in r['curve']] for r in TOKEXT['runs']
                       if r['arm'] == a]) for a in ARMS}
say(f'continued training on Hindi {TIER}')
say(pd.DataFrame({a: dict(zip(steps, c.mean(0))) for a, c in
                  curves.items()}).round(3).rename_axis('step'))
say(pd.DataFrame([dict(arm=r['arm'], seed=r['seed'], **{
    f'English bpb, step {s}': v for s, v in r['english'].items()})
    for r in TOKEXT['runs']]).round(3).to_string(index=False))
d2l.plot_intervals(steps, {a: (c.mean(0), c.min(0), c.max(0))
                           for a, c in curves.items()}, 'step',
                   'Hindi bits per byte', kind='band',
                   marks={'seeds': ('.', {a: (np.tile(steps, len(c)),
                                              c.ravel())
                                          for a, c in curves.items()})})
```

### Byte Patches
:label:`subsec_lm_tok_patches`

<!-- Prose: a tokenizer is a frozen compression decision; the Byte Latent
Transformer replaces fixed tokens by patches whose boundaries a small byte
model sets where its entropy jumps (cited, not claimed). The toy below uses
a character trigram model on Hindi text and cuts where the next
character's entropy rises, and compares patch and token lengths in UTF-8
bytes (a byte-level token can be part of a character); it says nothing
about BLT's quality. -->

```{.python .input #tokenizers-patches}
%%tab pytorch
text = ''.join(HI_TRAIN[:2000])
nxt = collections.defaultdict(collections.Counter)
for a, b, c in zip(text, text[1:], text[2:]):
    nxt[a + b][c] += 1
def entropy(ctx):
    n = np.array(list(nxt[ctx].values()), float)
    return float(-(n / n.sum() * np.log2(n / n.sum())).sum()) if len(n) else 8.
para = next(p for t in HI_HELD for p in t.split('\n') if len(p) > 300)
H = np.array([entropy(para[i - 2:i]) for i in range(2, len(para))])
cuts = [0] + [i + 2 for i in range(1, len(H)) if H[i] > H[i - 1] + 0.5]
patches = [len(para[a:b].encode()) for a, b in zip(cuts, cuts[1:]
                                                   + [len(para)])]
toks = {k: [len(VB[k][i]) for i in
            d2l._doc_token_ids(TOKENIZERS[k], [para])[0]]
        for k in ('Qwen3', 'Qwen3.5', 'reference')}
say(f'UTF-8 bytes per patch and per token, one Hindi paragraph of '
    f'{len(para.encode()):,} bytes')
say(pd.DataFrame({'entropy patches': pd.Series(patches).describe(),
                  **{f'{k} tokens': pd.Series(v).describe()
                     for k, v in toks.items()}}).round(2))
```

## The Reference Tokenizer and Its Shards
:label:`subsec_lm_tok_reference`

<!-- Prose: the frozen decisions, as a table (tab_lm_tok_shard_format, the
chapter specification's part 1.6): byte-level BPE, exactly 32,768 pieces,
NFC, Qwen3's split rule with single digits, ids 0-31 the special tokens
(<|endoftext|> = 0 is the document end, no BOS), trained on the first
80,000 documents of the shuffled training stream; special-token strings in
a document are text, never special ids (a page about <|endoftext|> must
not end early); the shards. -->

:The reference tokenizer and its shards (normative; the next chapter reads them).
:label:`tab_lm_tok_shard_format`

| Field | Value |
|---|---|
| Tokenizer | byte-level BPE, 32,768 pieces, NFC, single digits, no BOS |
| Special ids | 0 end of text (the EOS after every document), 1 padding, 2 and 3 chat-turn delimiters, 4-31 reserved |
| `.bin` | little-endian ids, `uint16` for at most 65,536 ids, else `uint32` |
| `.idx` | `uint32` offsets, one per document plus the token count |
| `.meta.parquet` | `doc_id`, `n_tokens`, then `url`, `dump`, `bucket`, `edu_score`, `int_score`, `gpt2_tokens`, `src_file`, `src_row` |
| `manifest.json` | `d2l-corpus/1`: sources, stages, tokenizer hash, format, splits |

<!-- Prose: r2_tokenizer loads the committed file, checks its SHA-256 and
wraps it for Transformers with EOS id 0, no BOS, and special strings in
text kept as text (chapter spec part 1.6a). -->

```{.python .input #tokenizers-r2-tok-def}
%%tab pytorch
#@save
R2_TOKENIZER_SHA256 = (  # of lm-r2-tokenizer-v1 (tools/build_r2_shards.py)
    'b1b4b45c36aac256ae4f4bd5e5fd595395242fc6949299cb0e2e25d2a5a656e3')

#@save
def r2_tokenizer(raw=False):
    """The reference tokenizer (artifact lm-r2-tokenizer-v1; its file is
    checked against R2_TOKENIZER_SHA256): a Transformers
    PreTrainedTokenizerFast with EOS <|endoftext|> = 0, pad <|pad|> = 1,
    no BOS, and special-token strings in text encoded as text; raw=True
    gives the tokenizers.Tokenizer itself."""
    if 'lm-r2-tokenizer-v1' not in d2l.DATA_HUB or not d2l.R2_TOKENIZER_SHA256:
        raise KeyError('lm-r2-tokenizer-v1 is not registered yet: it is '
                       'built by tools/build_r2_shards.py')
    return d2l._r2_from_file(d2l.download_keyed('lm-r2-tokenizer-v1'),
                             raw=raw, sha256=d2l.R2_TOKENIZER_SHA256)
```

<!-- Prose: retraining from the registered documents reproduces the
committed tokenizer.json byte for byte; the wrapper's contract (what the
next chapters call); the reference tokenizer's tokens against GPT-2's on
the same documents (the dataset's own token_count column). -->

```{.python .input #tokenizers-r2-retrain}
%%tab pytorch
say(type(R2).__name__, f'| {len(R2):,} ids | eos {R2.eos_token_id} | '
    f'bos {R2.bos_token_id} | pad {R2.pad_token_id} | SHA-256 '
    f'{d2l.R2_TOKENIZER_SHA256[:16]}...')
assert R2.eos_token_id == 0 and R2.bos_token_id is None
assert d2l._lm_boundary_id(R2) == 0 and len(R2) == R2_VOCAB
assert R2.encode('a b', add_special_tokens=False) == R2(
    'a b', add_special_tokens=False)['input_ids'] == R2_RAW.encode('a b').ids
SPECIAL_TEXT = ('A page about chat templates: <|endoftext|> ends a document, '
                '<|pad|> pads a batch, <|im_start|> and <|im_end|> wrap a '
                'turn.')
ids = R2.encode(SPECIAL_TEXT, add_special_tokens=False)
assert min(ids) >= 32 and R2.decode(ids) == SPECIAL_TEXT
say(f'special strings in text: {len(ids)} ordinary tokens, none below 32')

assert SHARDS.n_docs >= N_TOK_DOCS
tic = time.time()
DOCS = R2_RAW.decode_batch([SHARDS.doc(i)[:-1].tolist()
                            for i in range(N_TOK_DOCS)])
retrained = d2l.train_hf_bpe(DOCS, R2_VOCAB, d2l.R2_SPECIALS)
sha = hashlib.sha256(retrained.to_str(pretty=False).encode()).hexdigest()
say(f'retrained on the first {N_TOK_DOCS:,} training documents '
    f'({sum(map(len, DOCS)) / 1e6:.0f}M characters) in '
    f'{time.time() - tic:.0f} s: SHA-256 {sha[:16]}...',
    'equal to the committed file' if sha == d2l.R2_TOKENIZER_SHA256
    else 'DIFFERENT')
assert sha == d2l.R2_TOKENIZER_SHA256
texts = [r['text'] for r in VAL_TEXT[:10_000]]
assert R2_RAW.decode_batch([e.ids for e in R2_RAW.encode_batch(
    texts, add_special_tokens=False)]) == texts
assert [e.ids for e in R2_RAW.encode_batch(DOCS[:10_000])] == [
    SHARDS.doc(i)[:-1].tolist() for i in range(min(10_000, N_TOK_DOCS))]
meta = SHARDS.meta(0).iloc[:N_TOK_DOCS]
ratio = (meta.n_tokens - 1).sum() / meta.gpt2_tokens.sum()
say(f'round trips: {len(texts):,} validation texts and '
    f'{min(10_000, N_TOK_DOCS):,} training documents; reference tokens '
    f'per GPT-2 token: {ratio:.3f}')
del DOCS, retrained
```

<!-- Prose: write_shards and TokenShards: each document is its ids and
one EOS, documents never span shards, a shard closes at the first document
boundary past shard_tokens; TokenShards finds documents by global index and
tokens by global position, and verify() recomputes every hash and the EOS
and offset invariants. -->

```{.python .input #tokenizers-shards-def}
%%tab pytorch
#@save
def write_shards(docs, tokenizer, out_dir, split, shard_tokens=100_000_000,
                 eos_id=None, meta_fn=None, manifest=None, *, batch_docs=1000,
                 max_shards=None):
    """Tokenize documents into shards out_dir/{split}-{k:05d}.bin|.idx|
    .meta.parquet and return the split's manifest entry. docs yields
    (doc_id, text) or (doc_id, text, meta); each document becomes its ids
    and one EOS (eos_id, default the tokenizer's <|endoftext|>); special
    strings in a text stay text, and a special id inside a document is an
    error. A shard closes at the first document boundary at or after
    shard_tokens; ids are uint16 for vocabularies up to 65,536, else uint32.
    meta_fn(doc_id, text, meta) gives a document's extra meta columns
    (default: meta itself). With max_shards, writing stops when that many
    shards are closed (the rest of docs is not read). The entry is stored
    in manifest (a CorpusManifest, created if None), which is saved as
    out_dir/manifest.json."""
    import itertools
    os.makedirs(out_dir, exist_ok=True)
    vocab, eos, special = d2l._tok_info(tokenizer)
    eos = eos if eos_id is None else eos_id
    if eos is None:
        raise ValueError('the tokenizer has no <|endoftext|>: pass eos_id')
    dtype = np.dtype('uint16' if vocab <= 2**16 else 'uint32')
    special = np.array(sorted(special | {eos}), dtype=np.int64)
    shards, w, docs = [], None, iter(docs)
    while (len(shards) != max_shards
           and (batch := list(itertools.islice(docs, batch_docs)))):
        batch = [(d[0], d[1], d[2] if len(d) > 2 else None) for d in batch]
        for (doc_id, text, meta), ids in zip(
                batch, d2l._doc_token_ids(tokenizer, [b[1] for b in batch])):
            ids = np.asarray(ids, dtype=np.int64)
            bad = np.isin(ids, special)
            if bad.any():
                raise ValueError(f'document {doc_id}: special id '
                                 f'{int(ids[bad][0])} inside the text')
            if w is None:
                w = d2l._ShardWriter(out_dir, f'{split}-{len(shards):05d}',
                                     dtype)
            extra = meta_fn(doc_id, text, meta) if meta_fn else (meta or {})
            w.add(doc_id, np.append(ids, eos).astype(dtype), extra)
            if w.n_tokens >= shard_tokens:
                shards.append(w.close())
                w = None
                if len(shards) == max_shards:
                    break
    if w is not None:
        shards.append(w.close())
    entry = dict(n_tokens=sum(s['n_tokens'] for s in shards),
                 n_docs=sum(s['n_docs'] for s in shards), shards=shards)
    manifest = manifest or d2l.CorpusManifest(os.path.basename(
        os.path.normpath(out_dir)))
    manifest.splits = dict(manifest.splits or {}, **{split: entry})
    fmt = dict(dtype=dtype.name, eos_id=int(eos), endianness='little',
               doc_end='eos', idx_dtype='uint32',
               meta_required=['doc_id', 'n_tokens'])
    if manifest.format not in (None, fmt):
        raise ValueError(f'manifest format {manifest.format} != {fmt}')
    manifest.format = fmt
    manifest.tokenizer = manifest.tokenizer or dict(vocab_size=vocab)
    manifest.save(os.path.join(out_dir, 'manifest.json'))
    return entry

#@save
class TokenShards:
    """Read-only access to a shard set written by write_shards: documents
    by global index, token ranges by global position, per-shard meta, and
    verify(), which recomputes every file's SHA-256 and the EOS and offset
    invariants. max_tokens keeps the smallest prefix of shards holding at
    least that many tokens; fetch=True downloads missing files from the
    manifest's url and checks sha1 and SHA-256. A pickled copy (a data
    loader's worker) carries no memory maps; it opens its own."""
    def __init__(self, root, split='train', fetch=False, max_tokens=None):
        self.root, self.split, self.fetch = root, split, fetch
        with open(os.path.join(root, 'manifest.json'), encoding='utf-8') as f:
            self.manifest = json.load(f)
        shards = self.manifest['splits'][split]['shards']
        if max_tokens is not None:
            total = np.cumsum([s['n_tokens'] for s in shards])
            shards = shards[:int(np.searchsorted(total, max_tokens)) + 1]
        self.shards = shards
        fmt = self.manifest['format']
        self.dtype = np.dtype(fmt['dtype']).newbyteorder('<')
        self.eos_id = fmt['eos_id']
        self.vocab_size = (self.manifest.get('tokenizer') or {}).get(
            'vocab_size')
        self._docs = np.cumsum([0] + [s['n_docs'] for s in shards])
        self._toks = np.cumsum([0] + [s['n_tokens'] for s in shards])
        self.n_docs, self.n_tokens = int(self._docs[-1]), int(self._toks[-1])
        self.n_shards = len(shards)
        self._mm, self._idx = {}, {}

    def __getstate__(self):  # a worker process reopens the memory maps
        return dict(self.__dict__, _mm={})

    def _path(self, k, ext):
        s = self.shards[k]
        path = os.path.join(self.root, s['file'] + d2l._SHARD_EXT[ext])
        if not os.path.exists(path):
            if not self.fetch or 'url' not in s:
                raise FileNotFoundError(path)
            d2l._fetch_shard_file(s, ext, path)
        return path

    def shard(self, k):
        if k not in self._mm:
            self._mm[k] = np.memmap(self._path(k, 'bin'), dtype=self.dtype,
                                    mode='r')
        return self._mm[k]

    def offsets(self, k):
        if k not in self._idx:
            self._idx[k] = np.fromfile(self._path(k, 'idx'), dtype='<u4')
        return self._idx[k]

    def locate(self, i):
        """(shard, document within it, offset of its first token there)."""
        if not 0 <= i < self.n_docs:
            raise IndexError(i)
        k = int(np.searchsorted(self._docs, i, side='right')) - 1
        j = i - int(self._docs[k])
        return k, j, int(self.offsets(k)[j])

    def locate_token(self, t):
        """(shard, position within it) of global token position t."""
        if not 0 <= t < self.n_tokens:
            raise IndexError(t)
        k = int(np.searchsorted(self._toks, t, side='right')) - 1
        return k, t - int(self._toks[k])

    def doc(self, i):
        k, j, a = self.locate(i)
        return np.asarray(self.shard(k)[a:self.offsets(k)[j + 1]],
                          dtype=np.int32)

    def iter_docs(self, start=0):
        for i in range(start, self.n_docs):
            yield i, self.doc(i)

    def read(self, start, n):
        """n tokens from global position start, across shard boundaries."""
        if start < 0 or start + n > self.n_tokens:
            raise IndexError((start, n))
        out = []
        while n > 0:
            k, a = self.locate_token(start)
            take = min(n, self.shards[k]['n_tokens'] - a)
            out.append(np.asarray(self.shard(k)[a:a + take], dtype=np.int32))
            start, n = start + take, n - take
        return np.concatenate(out) if out else np.zeros(0, dtype=np.int32)

    def meta(self, k):
        return pd.read_parquet(self._path(k, 'meta'))

    def verify(self):
        """Recompute every file's SHA-256 and check that offsets start at 0,
        increase and end at the shard's token count, that EOS ends every
        document and occurs nowhere else, and that the meta rows match."""
        for k, s in enumerate(self.shards):
            for ext in d2l._SHARD_EXT:
                path = self._path(k, ext)
                if d2l._sha256_file(path) != s['sha256'][ext]:
                    raise ValueError(f'{path}: SHA-256 differs from the '
                                     'manifest')
            ids, idx = self.shard(k), self.offsets(k)
            name = s['file']
            if (len(idx) != s['n_docs'] + 1 or idx[0] != 0
                    or idx[-1] != len(ids) or len(ids) != s['n_tokens']
                    or (np.diff(idx.astype(np.int64)) <= 0).any()):
                raise ValueError(f'{name}.idx: offsets do not match the '
                                 'shard')
            if (np.asarray(ids[idx[1:].astype(np.int64) - 1]) != self.eos_id
                    ).any() or int((ids == self.eos_id).sum()) != s['n_docs']:
                raise ValueError(f'{name}.bin: EOS not exactly at the '
                                 'document ends')
            m = self.meta(k)
            if len(m) != s['n_docs'] or (m['n_tokens'].to_numpy()
                                         != np.diff(idx)).any():
                raise ValueError(f'{name}.meta.parquet: rows do not match '
                                 'the offsets')
        return True

#@save
def fwedu_shards(which='100m', fetch=True, max_tokens=None):
    """The reference shards of FineWeb-Edu under the reference tokenizer:
    which='val' (2^23 validation tokens), '100m' (train-00000, 10^8 tokens)
    or '4b' (the 40 training shards, downloaded one by one when first read).
    Each lives in ../data/lm/fwedu-r2-v1/<which>/, extracted on the first
    call (make that call once before starting several processes)."""
    keys = {'val': 'lm-fwedu-val-v1', '100m': 'lm-fwedu-shards-100m-v1',
            '4b': 'lm-fwedu-manifest-4b-v1'}
    if which not in keys:
        raise ValueError(f'which is one of {tuple(keys)}')
    key = keys[which]
    if key not in d2l.DATA_HUB:
        raise KeyError(f'{key} is not registered yet: it is built by '
                       'tools/build_r2_shards.py')
    root = os.path.join('..', 'data', 'lm', 'fwedu-r2-v1', which)
    path, sha1 = d2l.download_keyed(key), d2l.DATA_HUB[key][1]
    stamp = os.path.join(root, '.extracted-' + key)
    old = None
    if os.path.exists(stamp):
        with open(stamp) as f:
            old = f.read()
    if old != sha1:
        shutil.rmtree(root, ignore_errors=True)  # afresh when the
        os.makedirs(root)                        # registered sha1 changes
        if which == '4b':  # the manifest only; shards come when read
            shutil.copyfile(path, os.path.join(root, 'manifest.json'))
        else:
            d2l.extract(path, root)
        with open(stamp, 'w') as f:
            f.write(sha1)
    return d2l.TokenShards(root, 'val' if which == 'val' else 'train',
                           fetch=fetch, max_tokens=max_tokens)
```

<!-- Prose: write_shards and TokenShards (shown above): each document is
its ids and one EOS, documents never span shards, a shard closes at the
first document boundary past shard_tokens; the checks below on a small
temporary set, the uint32 path the next chapter uses for Qwen3, and the
registered shards. -->

```{.python .input #tokenizers-shards-demo}
%%tab pytorch
docs = [(r['doc_id'], r['text']) for r in VAL_TEXT[:2000]]
docs.append(('fixture-special-strings', SPECIAL_TEXT))
with tempfile.TemporaryDirectory() as tmp:
    entry = d2l.write_shards(docs, R2, tmp, 'val', shard_tokens=500_000)
    ts = d2l.TokenShards(tmp, 'val')
    ids = d2l._doc_token_ids(R2, [t for _, t in docs])
    assert ts.n_docs == len(docs) and ts.n_shards > 1
    assert all(np.array_equal(ts.doc(i), ids[i] + [0])
               for i in range(len(docs)))
    for k, s in enumerate(entry['shards']):
        x, idx = np.asarray(ts.shard(k)), ts.offsets(k)
        n = np.diff(idx)
        assert (x == 0).sum() == (x < 32).sum() == s['n_docs']
        assert np.array_equal(idx, np.concatenate([[0], np.cumsum(n)]))
    assert ts.verify()
    i = ts.n_docs // 2 + 1  # a resume cursor
    k, j, off = ts.locate(i)
    first = next(ts.iter_docs(start=i))
    assert first[0] == i and np.array_equal(
        first[1], ts.shard(k)[off:ts.offsets(k)[j + 1]])
    t = entry['shards'][0]['n_tokens'] - 5  # across the first boundary
    assert np.array_equal(ts.read(t, 10), np.concatenate(
        [ts.shard(0)[-5:], ts.shard(1)[:5]]))
    say(f'{ts.n_docs:,} documents, {ts.n_tokens:,} tokens in '
        f'{ts.n_shards} shards; doc {i} is shard {k}, document {j}, '
        f'token {off:,}; meta columns: {list(ts.meta(0).columns)}')
    q = d2l.write_shards(docs[:100] + docs[-1:], QWEN,
                         os.path.join(tmp, 'qwen'), 'val')
    tq = d2l.TokenShards(os.path.join(tmp, 'qwen'), 'val')
    assert tq.dtype == np.uint32 and tq.eos_id == 151643 and tq.verify()
    say(f'Qwen3: {tq.dtype}, EOS {tq.eos_id}, {tq.n_tokens:,} tokens; '
        f'the fixture ends in {tq.doc(tq.n_docs - 1)[-3:].tolist()}')
for name, ts in (('train', SHARDS), ('val', VAL)):
    i = ts.n_docs - 7
    k, j, off = ts.locate(i)  # the resume cursor of document i
    assert np.array_equal(next(ts.iter_docs(start=i))[1], ts.shard(k)[
        off:ts.offsets(k)[j + 1]]) and ts.verify()
    say(f'{name}: {ts.n_docs:,} documents, {ts.n_tokens:,} tokens, '
        f'{ts.n_shards} shard(s), {ts.dtype}, EOS {ts.eos_id}, '
        f'vocabulary {ts.vocab_size:,}; verified')
say(json.dumps(SHARDS.manifest['format']))
```

<!-- Prose: FineWeb-Edu is deduplicated within each crawl dump, not across
dumps, so the same page can sit in validation and in training: the
val_dedup stage dropped validation documents that repeat a training
document's text or URL; what is left of the overlap, measured with exact
50-token spans against all of train-00000 (20.3's idea of exact n-gram
matching on token ids). -->

```{.python .input #tokenizers-leakage}
%%tab pytorch
stage = next(s for s in VAL.manifest['stages'] if s['name'] == 'val_dedup')
say(f"val_dedup: {stage['n_in']:,} validation documents in, "
    f"{stage['n_out']:,} out; dropped {stage['drops']}")
N = 50
tic = time.time()
v_ids = np.concatenate([np.asarray(VAL.shard(k)) for k in
                        range(VAL.n_shards)])
t_ids = np.asarray(SHARDS.shard(0))
hv, okv = d2l._rolling_ngram_hashes(v_ids, N, eos=0)
ht, okt = d2l._rolling_ngram_hashes(t_ids, N, eos=0)
t_pos = np.flatnonzero(okt)
order = np.argsort(ht[t_pos], kind='stable')
t_sorted = ht[t_pos][order]
j = np.minimum(np.searchsorted(t_sorted, hv), len(t_sorted) - 1)
starts = np.flatnonzero(okv & (t_sorted[j] == hv))
t_start = t_pos[order[j[starts]]]
span = np.arange(N)
same = (v_ids[starts[:, None] + span] == t_ids[t_start[:, None] + span]
        ).all(1)  # the hash match confirmed on the ids
starts = starts[same]
v_off = np.concatenate([[0], np.cumsum([VAL.shard(k).shape[0]
                                        for k in range(VAL.n_shards)])])
doc_starts = np.concatenate([VAL.offsets(k)[:-1].astype(np.int64)
                             + v_off[k] for k in range(VAL.n_shards)])
hit_doc = np.searchsorted(doc_starts, starts, 'right') - 1
hit_docs = np.unique(hit_doc)
cover = np.zeros(len(v_ids) + 1, np.int64)
np.add.at(cover, starts, 1)
np.add.at(cover, starts + N, -1)
covered = (np.cumsum(cover)[:-1] > 0).sum()
f_docs = len(hit_docs) / VAL.n_docs
say(f'{len(starts):,} validation 50-grams also in train-00000 '
    f'({SHARDS.shards[0]["n_docs"]:,} documents), {time.time() - tic:.0f}'
    f' s: {len(hit_docs):,} of {VAL.n_docs:,} validation documents '
    f'({f_docs:.2%}), {covered / VAL.n_tokens:.3%} of their tokens')
docs_of = collections.defaultdict(set)  # shared span -> its documents
for h, k in zip(hv[starts].tolist(), hit_doc.tolist()):
    docs_of[h].add(k)
first = {h: s for s, h in zip(starts[::-1], hv[starts][::-1].tolist())}
shown = []  # the most shared spans, one per group of documents
for h in sorted(docs_of, key=lambda h: (-len(docs_of[h]), first[h])):
    if not any(docs_of[h] & g for g in shown) and len(shown) < 3:
        shown.append(docs_of[h])
        s = first[h]
        say(f'{len(docs_of[h])} documents: '
            f'{R2_RAW.decode(v_ids[s:s + N].tolist())!r}'[:100])
if f_docs > 0.05:
    say('more than 5% of validation documents overlap: inspect the spans'
        ' above before reading this as boilerplate')
```

<!-- Prose: the chapter's manifest: what the next chapter reads, with
its hash, its sources and stages, and the sizes of the four artifacts. -->

```{.python .input #tokenizers-chapter-manifest}
%%tab pytorch
M = d2l.CorpusManifest.load(os.path.join(SHARDS.root, 'manifest.json'))
say(f'{M.name} ({M.SCHEMA}), SHA-256 {M.sha256()[:16]}...')
src = M.sources[0]
say(pd.DataFrame([{k: src.get(k) for k in ('id', 'uri', 'revision',
                                           'license', 'n_docs')}]))
short = lambda files: ', '.join(f.split('/')[-1][:3] for f in files)
say(f"validation files {short(src['val_files'])}; train scope "
    f"{short(src['train_scope'])}; reserved {short(src['reserved'])}")
say(pd.DataFrame([{k: s.get(k) for k in ('name', 'n_in', 'n_out',
                                         'tokens_out', 'drops')}
                  for s in M.stages]).to_string(index=False))
say('tokenizer:', {k: v for k, v in M.tokenizer.items()
                   if k != 'specials'})
for key in R2_KEYS:
    if key not in d2l.DATA_HUB:
        say(f'{key}: pending')
        continue
    path = d2l.download_keyed(key)
    line = f'{key}: {os.path.getsize(path) / 1e6:,.2f} MB'
    if key == 'lm-fwedu-manifest-4b-v1':  # shards are fetched when read
        with open(path) as f:
            split = json.load(f)['splits']['train']
        gb = sum(s['bytes']['bin'] for s in split['shards']) / 1e9
        line += (f", listing {split['n_tokens']:,} tokens in "
                 f"{len(split['shards'])} shards ({gb:.1f} GB of .bin)")
    say(line)
```

```{.python .input #tokenizers-results}
%%tab pytorch
PARTS = {'tok-lm': TOKLM, 'tok-extend': TOKEXT}
if RERUN:  # the registered file holds both parts
    merged = dict(schema='d2l-tok-results/1', **PARTS)
    d2l.lm_save_rerun(RES_KEY + '.json', lambda p: write_json(p, merged))
    say(f'regenerated: {", ".join(sorted(RERUN))}' + (
      '; the other part is the registered one'
      if len(RERUN) < len(ARTIFACTS) else ''))
else:
    reg = RESULTS['tok-extend']['step0']
    diff = {k: max(abs(STEP0[k][c] - reg[k][c]) for c in
                   ('hindi_bpb', 'belebele_bpb', 'english_bpb'))
            for k in STEP0}
    say('small-LM runs: tokenizers, texts and byte counts equal to the '
        'registered runs')
    if SMALL:
        say('step 0: the small CPU path scores fewer texts than the '
            'registered results; not compared')
    else:
        say('step 0, largest bits-per-byte difference from the registered '
            'values:', {k: round(v, 4) for k, v in diff.items()})
        assert max(diff.values()) < 0.01
```

## Summary

<!-- Prose: 3-6 sentences (W1 rule). -->

## Exercises

<!-- Prose: the seven exercises of spec 20.6 part 8: (1) same bits per
byte, different perplexities; (2) TokenShards.locate and iter_docs resume,
id 0 only at document ends; (3) the vocabulary size of D3 for a stated
parameter budget and elasticity, and its untied variant; (4) EM never
decreases the corpus log-likelihood; (5) digit chunking under three
tokenizers and a small arithmetic probe with d2l.score_choices on
Qwen3-0.6B-Base; (6) 1,024, 4,096 and 8,192 new Hindi tokens with the
decomposition initialization; (7) editing the merge table instead of
appending pieces. -->
