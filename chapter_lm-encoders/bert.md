# Pretrained Encoders
:label:`sec_lm_bert`

The previous chapter labeled news headlines by prompting a decoder
(:numref:`sec_lm_prompting`). With a few solved examples in its prompt, the
decoder beat a classifier fitted to the same examples, but it read those
examples again for every query. Now suppose the task does not stop after two
hundred headlines. A support team sorts ten thousand messages an hour by
topic, and over the months thousands of labeled examples pile up. Should we
keep prompting, or can a model learn from the labels and then label the
stream cheaply?

The bidirectional encoder of :numref:`sec_transformer` points to an answer.
It learned to predict hidden bytes from both sides, and to do so it computed
at every position a vector that describes the text around it. BERT
:cite:`Devlin.Chang.Lee.ea.2018` trained the same kind of model on subword
tokens and a few billion words. It then added a small layer for each task,
the task's *head*, and trained all weights briefly on the task's labeled
examples, a step called *fine-tuning*. With one head per task, BERT beat the
specialized models of its time across a whole suite of language-understanding
tasks. Encoders of this kind, with
about a hundred million parameters, still do much of the routine labeling of
text in filters, taggers and search engines. In this section we load a
released encoder, fine-tune it for three tasks, and measure on the same items
when it beats a prompted decoder, and at what cost.

```{.python .input #bert-imports}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #bert-extras}
%%tab pytorch
import copy
import json
import pandas as pd
import time
import transformers
import zipfile
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors
from seqeval.metrics import f1_score as seqeval_f1
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
```

The encoder is ModernBERT-base :cite:`Warner.Chaffin.Clavie.ea.2024`,
released in 2024, and the decoder is Qwen3-1.7B, the instruct model of
:numref:`sec_lm_prompting`. As in the previous chapter, the expensive results
were computed once and stored. Running this notebook reads them and repeats
one piece of each live; the environment variable `D2L_RERUN` names the
results to recompute. The configuration cell collects the settings of every
run, each explained where it is first used.

```{.python .input #bert-config}
%%tab pytorch
ENC, DEC = 'answerdotai/ModernBERT-base', 'Qwen/Qwen3-1.7B'
GPU = torch.cuda.is_available()
SMALL = not GPU  # the CPU path: registered results, shorter live checks
ARTIFACTS = {  # name: (registered key, the file a regeneration writes)
    'bert-sweep': ('lm-enc-bert-sweep-v1', 'lm-enc-bert-sweep.json'),
    'bert-ckpt': ('lm-enc-modernbert-mnli-v1', 'lm-enc-modernbert-mnli.zip'),
    'bert-tagging': ('lm-enc-bert-tagging-v1', 'lm-enc-bert-tagging.json'),
    'bert-decoder': ('lm-enc-bert-decoder-v1', 'lm-enc-bert-decoder.json')}
RERUN = d2l.lm_rerun(ARTIFACTS)  # D2L_RERUN: the results to regenerate
assert 'bert-ckpt' not in RERUN or 'bert-sweep' in RERUN, \
    'the sweep run writes the checkpoint: D2L_RERUN=bert-sweep'
RERUN |= {'bert-ckpt'} if 'bert-sweep' in RERUN else set()
assert GPU or not RERUN, 'regenerating the results needs a GPU'
N_MNLI, N_SST = (250, 1000, 4000, 16000), (100, 500, 2000, 8000)
SEEDS, SHOTS, LRS = (0, 1, 2), (0, 4, 8), (2e-5, 5e-5, 8e-5)
MAXLEN = {'mnli': 128, 'sst2': 64}
N_EVAL = {'mnli': 2000, 'sst2': 872}  # validation items every model scores
N_CARVE, N_HYP, EPOCHS, BATCH = 2000, 5000, 2, 32
SIZES = {'mnli': N_MNLI, 'sst2': N_SST}
LABELS = {'mnli': ('entailment', 'neutral', 'contradiction'),
          'sst2': ('negative', 'positive')}
LIVE = (dict(n=250, items=200, dec_items=8, tag_train=1000, tag_items=500)
        if SMALL else dict(n=1000, items=2000, dec_items=2000,
                           tag_train=None, tag_items=None))  # None: all
TAG_LR = 5e-5  # the tagger's learning rate, fixed (not tuned)
VERBAL = {'mnli': ('yes', 'maybe', 'no'), 'sst2': ('negative', 'positive')}
FMT = {'mnli': '{x}\nAnswer: {y}', 'sst2': 'Review: {x}\nSentiment: {y}'}
DRAWS = {0: 1, 4: 1, 8: 3}  # demonstration sets per shot count
N_POOL, MIN_WORDS = 1000, {'mnli': 1, 'sst2': 8}  # demonstrations: the pool
CONFIG = {  # what each result is made with; reading one checks it
    'bert-sweep': dict(model=ENC, revision=d2l.HF_ENCODER_REVISIONS[ENC],
                       sizes=SIZES, seeds=SEEDS, lrs=LRS, maxlen=MAXLEN,
                       epochs=EPOCHS, batch=BATCH, n_carve=N_CARVE,
                       n_hyp=N_HYP, head='pretrained dense and norm'),
    'bert-tagging': dict(model=ENC, revision=d2l.HF_ENCODER_REVISIONS[ENC],
                         lr=TAG_LR, epochs=EPOCHS, batch=BATCH),
    'bert-decoder': dict(model=DEC, revision=d2l.HF_REVISIONS[DEC],
                         shots=SHOTS, draws=DRAWS, fmt=FMT, verbal=VERBAL,
                         norm='token', n_pool=N_POOL, min_words=MIN_WORDS,
                         demos='label round-robin, one slice per draw')}

def result_path(name):
    """The file this run regenerates for name, else the registered one."""
    key, file = ARTIFACTS[name]
    return d2l.lm_artifact(key, file if name in RERUN else None)

def read_result(name):
    """The results of name, checked against this notebook's settings."""
    with open(result_path(name)) as f:
        result = json.load(f)
    assert result['config'] == json.loads(json.dumps(CONFIG[name])), \
        f'{name} was made with other settings: D2L_RERUN={name}'
    return result

print(pd.DataFrame({'model': [ENC, DEC], 'revision': [
    d2l.HF_ENCODER_REVISIONS[ENC][:12], d2l.HF_REVISIONS[DEC][:12]]},
                   index=['encoder', 'decoder']))
print(f'GPU: {GPU}; regenerating: {", ".join(sorted(RERUN)) or "none"}')
```

The configuration reads the encoder's commit from `HF_ENCODER_REVISIONS`,
which pins encoders as :numref:`sec_lm_hf_models` pinned the decoders.

```{.python .input #bert-revisions}
%%tab pytorch
#@save
HF_ENCODER_REVISIONS = {
    'answerdotai/ModernBERT-base': '8949b909ec900327062f0ebf497f51aef5e6f0c8'}
```

The datasets come from pinned commits, and the section's four stored
results are registered beside them.

```{.python .input #bert-data-hub}
%%tab pytorch
#@save
# Several of these files share a name: fetch them with d2l.download_keyed
d2l.DATA_HUB['glue-mnli-train'] = (
    'https://huggingface.co/datasets/nyu-mll/glue/resolve/'
    'bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c/mnli/'
    'train-00000-of-00001.parquet',
    '29c714bf5419e6a351ba0fc979f000dcc1ccf671')
d2l.DATA_HUB['glue-mnli-validation-matched'] = (
    'https://huggingface.co/datasets/nyu-mll/glue/resolve/'
    'bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c/mnli/'
    'validation_matched-00000-of-00001.parquet',
    '190b5f94bc4126d3747a817da7c0da9df619dfbc')
d2l.DATA_HUB['glue-sst2-train'] = (
    'https://huggingface.co/datasets/nyu-mll/glue/resolve/'
    'bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c/sst2/'
    'train-00000-of-00001.parquet',
    '12cd5d3347611f133d19fd975afe391fa457553f')
d2l.DATA_HUB['glue-sst2-validation'] = (
    'https://huggingface.co/datasets/nyu-mll/glue/resolve/'
    'bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c/sst2/'
    'validation-00000-of-00001.parquet',
    'bc4155c5b85fcdda2784ea3154a9cfd00bf2009a')
d2l.DATA_HUB['conll2003-train'] = (
    'https://huggingface.co/datasets/tner/conll2003/resolve/'
    'b18612dee0007b1f7129731dbf2f5f2ed4039ad3/dataset/train.json',
    '09a4703c703992a75087ece5b5eab2e0621b0590')
d2l.DATA_HUB['conll2003-valid'] = (
    'https://huggingface.co/datasets/tner/conll2003/resolve/'
    'b18612dee0007b1f7129731dbf2f5f2ed4039ad3/dataset/valid.json',
    '3f9256142388f6da88b2269541fab5b8dbfe49f0')
d2l.DATA_HUB['conll2003-labels'] = (
    'https://huggingface.co/datasets/tner/conll2003/resolve/'
    'b18612dee0007b1f7129731dbf2f5f2ed4039ad3/dataset/label.json',
    'd4fd1344c9af2613a3094dd2c84ff594e5215af0')
d2l.DATA_HUB['lm-enc-bert-sweep-v1'] = (  # D2L_RERUN=bert-sweep, RTX 4090
    d2l.LM_DATA_URL + 'lm-enc-bert-sweep-v1.json',
    '06440f0e5acbd7b91deb6b66f8402f8bea8d6155')
d2l.DATA_HUB['lm-enc-modernbert-mnli-v1'] = (  # its MNLI checkpoint, bf16
    d2l.LM_DATA_URL + 'lm-enc-modernbert-mnli-v1.zip',
    '36365804df9df70b812f52595ee6d092639285c1')
d2l.DATA_HUB['lm-enc-bert-tagging-v1'] = (  # D2L_RERUN=bert-tagging
    d2l.LM_DATA_URL + 'lm-enc-bert-tagging-v1.json',
    'f0f0f41d18ed997462891d0c6dece556ad57b680')
d2l.DATA_HUB['lm-enc-bert-decoder-v1'] = (  # D2L_RERUN=bert-decoder
    d2l.LM_DATA_URL + 'lm-enc-bert-decoder-v1.json',
    '22b140629d3aa486a601068dce61b8a0f0301beb')
```

`load_encoder`, the encoder's counterpart of `d2l.load_hf`
(:numref:`sec_lm_hf_models`), loads a pinned encoder, bare or with a
masked-token, sentence or token head, and reports which weights go unused
and which start at random.

```{.python .input #bert-load-encoder}
%%tab pytorch
#@save
def load_encoder(model_id, revision=None, *, head=None, num_labels=None,
                 dtype=torch.bfloat16, device=None, attn_implementation='sdpa',
                 padding_side=None):
    """A pinned encoder from the Hugging Face Hub and its tokenizer, in eval
    mode: the bare model (head=None), or with a masked-token head ('mlm'), a
    sequence classifier ('seq_cls') or a token classifier ('token_cls');
    the pin comes from HF_ENCODER_REVISIONS, then HF_REVISIONS."""
    import logging
    import transformers
    revision = revision or d2l.HF_ENCODER_REVISIONS.get(
        model_id, d2l.HF_REVISIONS.get(model_id))
    if revision is None:
        raise KeyError(f'{model_id} is not pinned: pass revision=')
    auto = {None: transformers.AutoModel,
            'mlm': transformers.AutoModelForMaskedLM,
            'seq_cls': transformers.AutoModelForSequenceClassification,
            'token_cls': transformers.AutoModelForTokenClassification}[head]
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    extra = {} if num_labels is None else {'num_labels': num_labels}
    report = logging.getLogger('transformers.modeling_utils')
    quiet = lambda record: 'LOAD REPORT' not in record.getMessage()
    report.addFilter(quiet)  # its key table; the summary below replaces it
    try:
        model, info = auto.from_pretrained(
            model_id, revision=revision, dtype=dtype, use_safetensors=True,
            attn_implementation=attn_implementation, output_loading_info=True,
            **extra)
    finally:
        report.removeFilter(quiet)
    model = model.to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id, revision=revision)
    if padding_side is not None:
        tokenizer.padding_side = padding_side
    model.revision = revision
    print(f'{model_id}@{revision[:12]}: {type(model).__name__}\n    '
          f'{dtype}, {device}, attention '
          f'{model.config._attn_implementation}')
    for what, keys in (('not used', info['unexpected_keys']),
                       ('new, at random', info['missing_keys'])):
        if keys:
            print(f'    {what}: {", ".join(sorted(keys))}')
    return model.eval(), tokenizer
```

Which tasks should we measure? Sentiment is everywhere in the text a
decoder is pretrained on. *Natural language inference* is not. In MNLI
(Williams, Nangia and Bowman, 2018) each item pairs a *premise* with a
*hypothesis*, labeled *entailment* if the premise implies the hypothesis,
*contradiction* if it rules the hypothesis out, and *neutral* otherwise.
Where one label ends and the next begins is a convention of the dataset's
annotators. Sorting messages by topic lies somewhere between the two. A third
task, tagging every name in a sentence with its type, needs a label per word,
which a prompt handles badly.

We look at the answer first, on MNLI: the encoder, fine-tuned on 16,000
labeled pairs, against Qwen3-1.7B prompted with 8 solved pairs, on the same
2,000 validation pairs. Accuracies are means over three training seeds and
over three sets of solved examples; times are per thousand examples on one
RTX 4090.

```{.python .input #bert-opening}
%%tab pytorch
if RERUN & {'bert-sweep', 'bert-decoder'}:
    print('regenerating: the table reads the registered results')
else:
    sweep, decoder = read_result('bert-sweep'), read_result('bert-decoder')
    y = np.array(sweep['eval']['mnli']['label'])
    mnli_runs = [r for r in sweep['runs'] if r['task'] == 'mnli']
    n = max(r['n'] for r in mnli_runs)
    enc = np.mean([np.mean(np.array(r['pred']) == y) for r in mnli_runs
                   if r['n'] == n])  # the mean over seeds
    dec = np.mean([np.mean(np.array(p) == y)  # over demonstration draws
                   for p in decoder['pred']['mnli']['8']])
    print(pd.DataFrame({
        'labelled examples': [n, 8], 'accuracy': [enc, dec],
        'ms per 1,000 examples': [sweep['timing']['mnli'],
                                  decoder['timing']['mnli']['8']]},
        index=[f'{ENC.split("/")[1]}, fine-tuned ({len(SEEDS)} seeds)',
               f'{DEC.split("/")[1]}, 8 shots ({DRAWS[8]} draws)']).round(
        {'accuracy': 3, 'ms per 1,000 examples': 0}).to_string())
```

The encoder labels 0.84 of the pairs correctly and the decoder 0.63. The
encoder takes less than half a second per thousand pairs, the decoder 48
seconds: about a hundredth of the time. Yet the encoder learned from 16,000
labeled pairs and the decoder from 8. The rest of the section builds the
encoder's row of this table and then asks how many labels the encoder needs
before it passes the decoder.

## A Released Masked-Token Encoder
:label:`subsec_lm_bert_encoder`

### Token Vectors

The encoder of :numref:`sec_transformer` was a stack of transformer blocks
without the causal mask, trained to predict masked bytes from everything that
remained. Context on both sides lowered the loss of a masked byte by about a
third against context on one side (:numref:`subsec_mlm-loss-by-position`).
ModernBERT is a model of this kind at scale: it reads subword tokens, has 22
blocks instead of 4, and was pretrained on trillions of tokens. We pass it a
premise and a hypothesis together, without a head.

```{.python .input #bert-raw}
%%tab pytorch
encoder, tok = d2l.load_encoder(ENC, dtype=torch.float32)  # fp32: trained
dev = next(encoder.parameters()).device
pair = tok('A man plays a guitar on stage.', 'Someone is making music.',
           return_tensors='pt').to(dev)
d2l.print_wrapped(tok.convert_ids_to_tokens(pair['input_ids'][0].tolist()),
                  width=96)
print('fields:', list(pair))  # no token_type_ids
with torch.inference_mode():
    H = encoder(**pair).last_hidden_state
print('hidden states:', tuple(H.shape))  # (batch, tokens, width)
print('embedding modules:', [n for n, _ in encoder.embeddings.named_children()])
c = encoder.config
print(pd.Series({
    'layers': c.num_hidden_layers, 'width': c.hidden_size,
    'heads': c.num_attention_heads, 'GeGLU width': c.intermediate_size,
    'positions': c.max_position_embeddings,
    'global attention every': c.global_attn_every_n_layers,
    'local window': c.local_attention,
    'RoPE theta (global, local)': tuple(c.rope_parameters[t]['rope_theta']
        for t in ('full_attention', 'sliding_attention')),
    'vocabulary': c.vocab_size, 'classifier pooling': c.classifier_pooling,
    'parameters': f'{sum(p.numel() for p in encoder.parameters()):,}'}
                ).to_string())
```

The tokenizer writes the pair as `[CLS] premise [SEP] hypothesis [SEP]`, as
BERT did. `[SEP]` is a separator that closes each text. `[CLS]` is a special
token at the start, whose final vector BERT trained on a second objective,
*next-sentence prediction*: did the second text follow the first in the
source? BERT also added a *segment embedding*, a learned vector that marked
each token as part of the first or the second text (`token_type_ids`).
ModernBERT has none. Its tokenizer returns token ids and an attention mask,
and its embedding module holds a token table, a norm and dropout, so the
separator alone marks where the hypothesis begins.

The output has shape (1, 16, 768), one vector of 768 numbers per token.
Unlike the word vectors of :numref:`sec_lm_word_embeddings`, these depend on
the whole input. "Bank" beside a river and "bank" beside a teller get
different vectors, and the encoder need not choose between the two senses.
Since the pair enters as one input, every layer also lets each token of the
hypothesis attend to every token of the premise:
the two texts are compared word by word before anything is pooled. An
encoder that reads both texts of a pair together is called a
*cross-encoder*. Encoding each text alone and comparing the vectors, which
the next sections build, is much faster when one text must be compared with
many.

### Masked-Token Prediction

The checkpoint ships with its pretraining head: a dense layer, a GELU, a norm
and a projection onto the vocabulary. We mask one token in each of three
sentences and print the five most probable fills.

```{.python .input #bert-mlm-fill}
%%tab pytorch
mlm, _ = d2l.load_encoder(ENC, head='mlm', dtype=torch.float32)
texts = ['The capital of France is [MASK].',
         'He drank a cup of [MASK] before work.',
         'The encoder reads the words on [MASK] sides of the gap.']
X = tok(texts, padding=True, return_tensors='pt').to(dev)
with torch.inference_mode():  # one masked position per text, in order
    P = mlm(**X).logits[X['input_ids'] == tok.mask_token_id].softmax(-1)
for text, p, i in zip(texts, *P.topk(5)):
    print(text + '\n   ', ', '.join(
        f'{tok.decode(j, clean_up_tokenization_spaces=False).strip()} {q:.2f}'
        for q, j in zip(p.tolist(), i.tolist())))
MLM_HEAD = {k: v.detach().clone()  # dense and norm: the sentence head's start
            for k, v in mlm.head.state_dict().items()}
del mlm
d2l.lm_free()
```

The model puts 0.92 on "Paris", splits the cup between coffee (0.61) and
tea (0.21), and fills the third gap with "both" at 0.94, a word that only the
token after the gap pins down. The head predicts words. It knows nothing of
entailment or sentiment, so each task below gets a new head. We keep the
pretrained dense layer and norm, from which Transformers' sentence classifier
starts as well.

### The 80/10/10 Corruption

In :numref:`sec_transformer` every selected byte became the mask symbol. BERT
also selects 15% of the tokens for prediction, but it replaces only 80% of
them by `[MASK]`, replaces 10% by a random token and leaves 10% unchanged.
The loss is computed at all selected positions. Why not mask them all? First,
`[MASK]` never occurs in the text the encoder is later fine-tuned on. Second,
a model asked only about positions showing `[MASK]` would have no reason to
keep, elsewhere, a vector that describes the token there, and those are the
vectors fine-tuning uses. When any ordinary-looking token may be asked about,
the model must keep a good guess of the true token everywhere. `mlm_corrupt`
implements the rule and never selects a special token.

```{.python .input #bert-mlm-corrupt}
%%tab pytorch
#@save
def mlm_corrupt(ids, mask_id, vocab_size, special_ids, p=0.15,
                probs=(0.8, 0.1, 0.1), generator=None):
    """BERT's masked-token corruption of ids: each token that is not in
    special_ids is selected with probability p; a selected token becomes
    mask_id with probability probs[0], a uniformly random token with
    probs[1], and stays with probs[2]. Returns the corrupted ids and the
    labels: the original id where selected, -100 (not scored) elsewhere."""
    special = torch.isin(ids, torch.as_tensor(special_ids, device=ids.device))
    u, v = (torch.rand(ids.shape, generator=generator, device=ids.device)
            for _ in range(2))
    selected = (u < p) & ~special
    noise = torch.randint(vocab_size, ids.shape, generator=generator,
                          device=ids.device)
    out = torch.where(selected & (v < probs[0]), mask_id, ids)
    out = torch.where(selected & (v >= probs[0]) & (v < probs[0] + probs[1]),
                      noise, out)
    return out, torch.where(selected, ids, -100)
```

A million token ids through the corruption check the shares.

```{.python .input #bert-mlm-stats}
%%tab pytorch
g = torch.Generator().manual_seed(0)
ids = torch.randint(50_000, (10_000, 100), generator=g)  # ordinary tokens
ids[:, 0], ids[:, -1] = tok.cls_token_id, tok.sep_token_id
X, labels = d2l.mlm_corrupt(ids, tok.mask_token_id, len(tok),
                            tok.all_special_ids, generator=g)
special = torch.isin(ids, torch.tensor(tok.all_special_ids))
sel = labels != -100
masked, kept = X[sel] == tok.mask_token_id, X[sel] == ids[sel]
shares = {'selected, of ordinary tokens': sel[~special].float().mean(),
          '[MASK], of selected': masked.float().mean(),
          'random, of selected': (~masked & ~kept).float().mean(),
          'kept, of selected': kept.float().mean()}
shares = {k: v.item() for k, v in shares.items()}
print(pd.Series(shares).round(4).to_string())
print('special tokens selected:', (sel & special).sum().item())
assert not (sel & special).any() and all(abs(shares[k] - v) < 2e-3 for k, v
                                         in zip(shares, (0.15, 0.8, 0.1, 0.1)))
```

The corruption selects 0.1506 of the ordinary tokens, and of those it masks
0.798, randomizes 0.101 and keeps 0.101. So 12% of all tokens show `[MASK]`,
and only 1.5% are random tokens that the model must recognize as wrong.

### Relatives and ModernBERT

Three successors of BERT are worth knowing, each for what it changed
(:numref:`tab_lm_bert_relatives`).

:Three successors of BERT and what each changed.
:label:`tab_lm_bert_relatives`

| model | what it changes |
|---|---|
| RoBERTa :cite:`Liu.Ott.Goyal.ea.2019` | drops next-sentence prediction, draws new masks each time a sequence is read (as calling `mlm_corrupt` once per batch does), and trains longer on far more text |
| ELECTRA :cite:`clark2019electra` | a small generator fills the masked positions with plausible tokens, and the encoder predicts for *every* token whether it was replaced, so every position supplies a training signal, not 15% |
| T5 :cite:`raffel2020exploring` | an encoder--decoder: whole spans are masked, the decoder writes the missing spans, and every task, classification included, becomes text in and text out |

RoBERTa showed that BERT had been undertrained rather than misdesigned.
ELECTRA gets more learning out of each token of compute. T5 points toward the
decoders of the previous chapter, which do everything by writing text.

ModernBERT keeps BERT's bidirectional attention, its special tokens and its
masked-token objective, and changes the block. The feed-forward layer is
gated, a GeGLU :cite:`Shazeer.2020` of width 1,152, and no linear layer or
norm inside the blocks has a bias term. The classification head pools by the
mean, a choice we return to below.

The other changes serve long inputs. An input may have 8,192 tokens, sixteen
times BERT's 512, and attention alternates between two kinds of layers.
Every third layer, starting with the first, is *global*: every token attends
to all others. The other layers are *local*: each token attends only to the
tokens within a window of $w = 128$ positions, 64 on either side, as in the
sliding windows of :numref:`sec_attention-at-scale`.

Positions enter as rotations of queries and keys (:numref:`subsec_pos-rope`),
with base 10,000 in local layers and 160,000 in global ones. The larger base
rotates more slowly, so that positions thousands of tokens apart stay
distinguishable. *Unpadding* packs a batch of texts into one row without pad
tokens; it needs FlashAttention's kernels :cite:`Dao.2024`. With PyTorch's
fused attention (`attention sdpa` in the load report) batches are padded, and
only the speed differs.

The sentence pairs of this section are short, but the next sections encode
documents of thousands of tokens. How much does the window save there? A
global layer scores all $L^2$ pairs of positions of an input of $L$ tokens,
and a local layer about $Lw$ pairs. With $n_g$ global and $n_l$ local layers,
the attention work relative to a model with only global layers is

$$
\rho(L) = \frac{n_g L^2 + n_l L w}{(n_g + n_l) L^2} = \frac{n_g L + n_l w}{(n_g + n_l) L}.
$$
:eqlabel:`eq_lm_bert_attention_share`

For long inputs the local term vanishes and $\rho$ approaches
$n_g / (n_g + n_l)$: the global layers set the cost. With ModernBERT's
$n_g = 8$, $n_l = 14$ and $w = 128$ the formula gives 0.374 at
$L = 8{,}192$. The cell counts the pairs from boolean window masks.

```{.python .input #bert-attention-cost}
%%tab pytorch
glob = [i for i, t in enumerate(c.layer_types) if t == 'full_attention']
n_g, n_l = len(glob), c.num_hidden_layers - len(glob)
w, half = c.local_attention, c.local_attention // 2  # |i - j| <= 64
print('global layers:', glob)
rows = {}
for L in (512, 8192):
    band = torch.ones(L, L, dtype=torch.bool).triu(-half).tril(half)
    pairs = n_g * L * L + n_l * band.sum().item()  # summed over layers
    rows[L] = {'pairs, global layer': L * L, 'pairs, local layer':
               band.sum().item(), 'share of full, counted':
               pairs / (c.num_hidden_layers * L * L), 'share, formula':
               (n_g * L + n_l * w) / ((n_g + n_l) * L)}
print(pd.DataFrame.from_dict(rows, orient='index').round(3).to_string())
```

The count agrees with the formula at 8,192 tokens. At 512 it is a little
lower, 0.514 against 0.523, because windows are cut off at the ends of the
input. On short inputs nothing is saved: when $L$ is at most 65, a window of
64 tokens on either side covers the whole input, and the sentence pairs of
this section have about 40 tokens.

## Sentence and Pair Classification
:label:`subsec_lm_bert_heads`

### The Pooled Head

A classifier needs one vector per input. BERT used the final vector of
`[CLS]`, which its next-sentence objective had trained. ModernBERT averages
the final vectors over the tokens, leaving out padding. Its configuration
records this (`classifier_pooling: mean`) and Transformers' sequence
classifier follows the field, so read it rather than assume `[CLS]`. The
dense layer and norm that start the head come from the masked-token head,
which saw one token's vector at a time. Neither pooling is what they were
fitted to, and only an experiment says which serves better (Exercise
:exref:`ex_lm_bert-pooling`).

Let $\mathbf{h}_1, \ldots, \mathbf{h}_T$ be the final vectors of the $T$
tokens that are not padding. The head maps their mean to one logit per class:

$$
\bar{\mathbf{h}} = \frac{1}{T} \sum_{t=1}^{T} \mathbf{h}_t, \qquad
\mathbf{z} = \mathbf{W}_\textrm{c}\, \textrm{LN}\big(\textrm{GELU}(\mathbf{W}_\textrm{d} \bar{\mathbf{h}})\big) + \mathbf{b}_\textrm{c}.
$$

That is a mean, a dense layer $\mathbf{W}_\textrm{d}$ of size $768 \times 768$,
a GELU, a layer norm, and a linear map with bias $\mathbf{b}_\textrm{c}$ to
the logits. A dropout layer sits before the last map, but ModernBERT's
configuration sets its rate, like all its dropout rates, to zero.
`PooledClassifier` computes this, with first-token pooling as an option that
needs the padding on the right. Its submodules carry the names of
Transformers' `ModernBertForSequenceClassification`, so loading that model's
weights into ours is a matter of renaming.

```{.python .input #bert-pooled-classifier}
%%tab pytorch
#@save
class PooledClassifier(nn.Module):
    """A sentence or sentence-pair classifier on an encoder: the final
    hidden states pooled (the mean over the attention mask, or the first
    token), then dense, GELU, layer norm, dropout and a linear map to
    num_labels logits. Biases and the norm follow the encoder's
    configuration, as in Transformers' ModernBERT head."""
    def __init__(self, encoder, num_labels, pooling='mean', dropout=0.0):
        super().__init__()
        if pooling not in ('mean', 'cls'):
            raise ValueError(f'unknown pooling: {pooling!r}')
        c = encoder.config
        bias, d = getattr(c, 'classifier_bias', True), c.hidden_size
        self.encoder, self.pooling = encoder, pooling
        self.head = nn.Module()
        self.head.dense = nn.Linear(d, d, bias=bias)
        self.head.norm = nn.LayerNorm(d, eps=getattr(c, 'norm_eps', 1e-5),
                                      bias=getattr(c, 'norm_bias', True))
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(d, num_labels)
        p = next(encoder.parameters())  # the head lives where the encoder is
        self.head.to(p.device, p.dtype)
        self.classifier.to(p.device, p.dtype)

    def forward(self, input_ids, attention_mask):
        H = self.encoder(input_ids=input_ids,
                         attention_mask=attention_mask).last_hidden_state
        if self.pooling == 'cls':
            if not attention_mask[:, 0].all():
                raise ValueError('cls pooling needs right padding')
            h = H[:, 0]
        else:  # padding positions do not count
            m = attention_mask.unsqueeze(-1).to(H.dtype)
            h = (H * m).sum(1) / m.sum(1)
        h = self.head.norm(F.gelu(self.head.dense(h)))
        return self.classifier(self.drop(h))
```

The same encoder serves all three tasks of this section. Only the input and
the head change (:numref:`tab_lm_bert_heads`).

:One encoder, three heads. Every head starts with the dense layer, GELU and norm of the pretrained masked-token head.
:label:`tab_lm_bert_heads`

| task | input | vectors read | output |
|---|---|---|---|
| sentence (SST-2) | `[CLS] sentence [SEP]` | the mean over tokens | 2 logits |
| pair (MNLI) | `[CLS] premise [SEP] hypothesis [SEP]` | the mean over tokens | 3 logits |
| tagging (CoNLL-2003) | `[CLS] sentence [SEP]` | each word's first token | 9 logits per word |

<!-- Placeholder (figure): ![One encoder and three heads: a pooled
sentence head (mean, dense, GELU, norm, linear), a pair read jointly as
[CLS] premise [SEP] hypothesis [SEP] with no segment ids, and a tagging head
that labels each word's first token with B- and I- tags and skips the
others.](../img/mdl-lm-encoders-encoder-heads.svg)
:label:`fig_lm_encoder_heads` (generator tools/gen_mdl_lm_encoders_figures.py,
written in the figures step; when it lands, cite it beside
tab_lm_bert_heads in the sentence above). -->

We will fine-tune our own `PooledClassifier`, so we first check that it
computes what the library computes. We load Transformers' sequence
classifier with a new three-way head and copy its weights into a
`PooledClassifier` under the renamed keys; the load is strict, so it fails if
any name does not map. Then we compare the logits on eight padded MNLI pairs
in fp32 and in bf16.

```{.python .input #bert-head-check}
%%tab pytorch
mnli_val = pd.read_parquet(d2l.download_keyed('glue-mnli-validation-matched'))
torch.manual_seed(0)  # the head is new: any initialization serves
ref, _ = d2l.load_encoder(ENC, head='seq_cls', num_labels=3,
                          dtype=torch.float32)
clf = d2l.PooledClassifier(copy.deepcopy(encoder), 3,
                           pooling=ref.config.classifier_pooling)
clf.load_state_dict({k.replace('model.', 'encoder.', 1): v
                     for k, v in ref.state_dict().items()})
same = all(torch.equal(ref.head.state_dict()[k], v)
           for k, v in MLM_HEAD.items())
print('dense and norm as in the masked-token head:', same)
X = tok(list(mnli_val.premise[:8]), list(mnli_val.hypothesis[:8]),
        padding=True, truncation='only_first', max_length=MAXLEN['mnli'],
        return_tensors='pt').to(dev)
diff = {}
with torch.inference_mode():
    for dtype in (torch.float32, torch.bfloat16):
        ref, clf = ref.to(dtype), clf.to(dtype)
        diff[str(dtype)] = (ref(**X).logits - clf(**X)).abs().max().item()
print('max |logit difference|:', diff)
assert same and diff['torch.float32'] <= 1e-4 and diff['torch.bfloat16'] <= 2e-2
del ref, clf
d2l.lm_free()
```

Both differences are 0.0: the two models run the same operations in the
same order on the same batch, so even bf16 agrees bit for bit. The first
output line adds that the library's classifier takes its dense layer and
norm from the masked-token head, as ours does, so only its final map is
new.

### Fine-Tuning
:label:`subsec_lm_bert_finetune`

Fine-tuning trains every weight of the encoder and the head on the labeled
examples with the cross-entropy loss, briefly and with small steps.
`finetune_classifier` uses AdamW :cite:`Loshchilov.Hutter.2019` with weight
decay 0.01 on the weight matrices, a learning rate that rises linearly over
the first 10% of the steps and then falls linearly to zero, and gradients
clipped at norm 1. Warmup and clipping protect the pretrained weights from
the large gradients of a new layer that starts at random. We train two epochs
in batches of 32 and choose the learning rate among $2 \times 10^{-5}$,
$5 \times 10^{-5}$ and $8 \times 10^{-5}$ on 2,000 held-out training pairs,
never on the validation items. Products run in bf16 with fp32 weights. Each
batch is padded to its own longest example, not to the limit of 128 tokens,
which would spend most of the arithmetic on pad tokens.

The fp32 weights, their gradients and AdamW's two moment estimates take 16
bytes per parameter, about 2.4 GB for 149 million parameters. Activations
come on top and grow with the batch and the length (Exercise
:exref:`ex_lm_bert-memory`).

```{.python .input #bert-finetune}
%%tab pytorch
#@save
def finetune_classifier(model, train, eval, lr, epochs, batch_size, seed,
                        warmup=0.1, device=None):
    """Fine-tune model on train, a list of (input_ids, label) or, for token
    tagging, (input_ids, labels) with a label per token (-100: not scored),
    then predict eval. model(input_ids=, attention_mask=) returns logits or
    an output with .logits. AdamW (weight decay 0.01 on matrices), linear
    warmup over the share warmup of the steps then linear decay, gradients
    clipped to norm 1, bf16 autocast on a GPU; each batch is padded to its
    longest sequence (with id 0: the attention mask hides it, so any id
    serves); seed fixes the data order (and dropout, if the model has
    any). Returns the training losses, the eval accuracy and predictions
    (a label per example, or per scored token), the training seconds and
    the peak GPU memory in GiB."""
    device = torch.device(device or ('cuda' if torch.cuda.is_available()
                                     else 'cpu'))
    cuda = device.type == 'cuda'

    def batches(data, order):
        for i in range(0, len(order), batch_size):
            rows = [data[j] for j in order[i:i + batch_size]]
            T = max(len(x) for x, _ in rows)
            X = torch.tensor([x + [0] * (T - len(x)) for x, _ in rows])
            M = torch.tensor([[1] * len(x) + [0] * (T - len(x))
                              for x, _ in rows])
            y = torch.tensor([y if isinstance(y, int) else
                              y + [-100] * (T - len(y)) for _, y in rows])
            yield X.to(device), M.to(device), y.to(device)

    def logits(X, M):
        with torch.autocast(device.type, torch.bfloat16, enabled=cuda):
            out = model(input_ids=X, attention_mask=M)
        return getattr(out, 'logits', out).float()

    model.to(device).train()
    torch.manual_seed(seed)  # dropout, if the model has any
    order = torch.Generator().manual_seed(seed)
    steps = epochs * math.ceil(len(train) / batch_size)
    n_warm = max(1, round(warmup * steps))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([
        {'params': [p for p in params if p.ndim >= 2], 'weight_decay': 0.01},
        {'params': [p for p in params if p.ndim < 2], 'weight_decay': 0.0}],
        lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (
        (s + 1) / n_warm if s < n_warm else
        max(0.0, (steps - s) / max(1, steps - n_warm))))
    if cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    t0, losses = time.perf_counter(), []
    for _ in range(epochs):
        perm = torch.randperm(len(train), generator=order).tolist()
        for X, M, y in batches(train, perm):
            loss = F.cross_entropy(logits(X, M).flatten(0, -2), y.flatten(),
                                   ignore_index=-100)
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            losses.append(loss.detach())
    if cuda:
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - t0
    model.eval()
    with torch.inference_mode():
        out = [(logits(X, M).argmax(-1), y)
               for X, M, y in batches(eval, range(len(eval)))]
    hits = sum(((p == y) & (y != -100)).sum().item() for p, y in out)
    total = sum((y != -100).sum().item() for _, y in out)
    preds = [v for p, y in out for v in (p.tolist() if y.ndim == 1 else [
        r[y_r != -100].tolist() for r, y_r in zip(p, y)])]
    return dict(loss=torch.stack(losses).tolist(),
                eval_acc=hits / total if total else float('nan'),
                eval_pred=preds, seconds=seconds,
                peak_gib=(torch.cuda.max_memory_allocated(device) / 2**30
                          if cuda else float('nan')))
```

The last helper counts the operations of a forward pass, which the cost
comparison at the end of the section needs; its formula is derived there.

```{.python .input #bert-cost}
%%tab pytorch
#@save
def cost_per_example(n_params, n_tokens, n_embed=0):
    """Forward FLOPs per example, 2 (n_params - n_embed) n_tokens: a
    multiply and an add per weight and token, embedding lookups excluded."""
    return 2 * (n_params - n_embed) * n_tokens
```

### Data and Sentence Pairs

The second task, SST-2 (Socher et al., 2013), holds sentences from movie
reviews labeled positive or negative. For MNLI, two details of the pair input
matter. The separator is the only marker of the boundary, so a pipeline that
drops it or swaps the texts gives the model an input it was not fine-tuned
on. And a pair over 128 tokens is cut from the end of the premise
(`truncation='only_first'`); cut from the end of the pair, it could lose the
"not" that decides the label.

All splits come from one permutation of each training set: nested prefixes
for training, the next 2,000 items held out for the learning rate, and, from
the far end, the decoder's solved examples, balanced over the labels in three
disjoint sets. Every
model is scored on the same items: 2,000 MNLI validation pairs and all 872
SST-2 validation sentences. MNLI's annotators wrote one hypothesis per label
for each premise, so pairs that share a premise are not independent. Our
intervals therefore resample whole premises, keeping their hypotheses
together, so that three related pairs do not count as three independent
pieces of evidence (:numref:`sec_lm_eval_suites`).

```{.python .input #bert-data}
%%tab pytorch
def read(key):
    return pd.read_parquet(d2l.download_keyed(key))

def encode(task, df, hypothesis_only=False):
    """(input_ids, label) per row: a pair cut from the premise's end, a
    sentence (or a hypothesis alone) cut at MAXLEN."""
    if task == 'sst2' or hypothesis_only:
        text = df['hypothesis' if hypothesis_only else 'sentence'].str.strip()
        ids = tok(list(text), truncation=True, max_length=MAXLEN[task])
    else:
        ids = tok(list(df.premise.str.strip()), list(df.hypothesis.str.strip()),
                  truncation='only_first', max_length=MAXLEN[task])
    return list(zip(ids['input_ids'], df.label.tolist()))

def demonstrations(task, pool, draw, k=max(SHOTS)):
    """Demonstration set draw: k rows of pool, labels in round-robin order
    (rotated by draw), each label's rows from a slice no other draw uses;
    the first j rows are the j-shot set."""
    C = len(LABELS[task])
    per = -(-k // C)  # rows of each label a draw uses at most
    return pd.DataFrame([pool[pool.label == (draw + i) % C].iloc[
        draw * per + i // C] for i in range(k)])

train = {'mnli': read('glue-mnli-train'), 'sst2': read('glue-sst2-train')}
val = {'mnli': mnli_val.iloc[np.random.default_rng(0).permutation(
    len(mnli_val))], 'sst2': read('glue-sst2-validation')}
DATA = {}
for task, sizes in SIZES.items():
    perm = np.random.default_rng(0).permutation(len(train[task]))
    rows = train[task].iloc[perm[:max(sizes) + N_CARVE]]
    ev = val[task].iloc[:N_EVAL[task]]
    assert max(sizes) + N_CARVE + N_POOL <= len(perm)  # all disjoint
    pool = train[task].iloc[perm[-N_POOL:]]  # the permutation's end
    pool = pool[pool[{'mnli': 'premise', 'sst2': 'sentence'}[task]]
                .str.split().str.len() >= MIN_WORDS[task]]
    DATA[task] = {'perm': perm, 'train_df': rows.iloc[:max(sizes)],
                  'eval_df': ev, 'train': encode(task, rows.iloc[:max(sizes)]),
                  'carve': encode(task, rows.iloc[max(sizes):]),
                  'eval': encode(task, ev), 'eval_idx': ev.idx.tolist(),
                  'cluster': pd.factorize(ev.premise)[0] if task == 'mnli'
                  else None,  # MNLI: items that share a premise
                  'shots': [demonstrations(task, pool, d)
                            for d in range(max(DRAWS.values()))]}
assert all(x[0] == tok.cls_token_id and x[-1] == tok.sep_token_id and
           x.count(tok.sep_token_id) == 2 for part in ('train', 'carve', 'eval')
           for x, _ in DATA['mnli'][part])  # [CLS] a [SEP] b [SEP]
DATA['mnli']['hyp_train'] = encode('mnli', train['mnli'].iloc[
    DATA['mnli']['perm'][:N_HYP]], hypothesis_only=True)
DATA['mnli']['hyp_eval'] = encode('mnli', DATA['mnli']['eval_df'],
                                  hypothesis_only=True)
print(pd.DataFrame({task: {k: str(v) for k, v in {  # a column per task
    'largest subset': len(D['train']), 'carved (learning rate)':
    len(D['carve']), 'validation items': len(D['eval']),
    'cut at MAXLEN': sum(len(x) == MAXLEN[task] for x, _ in D['train']),
    'majority rate (validation)': D['eval_df'].label.value_counts(
        normalize=True).max().round(3), 'premises (validation)': 'n/a'
    if D['cluster'] is None else len(set(D['cluster']))}.items()}
    for task, D in DATA.items()}).to_string())
```

Only 51 of the 16,000 training pairs lose the end of their premise, and no
SST-2 sentence is cut. A classifier that ignores its input scores the majority
rate, 0.363 on MNLI and 0.509 on SST-2. The 2,000 MNLI items share 1,635
premises.

The helpers build a fresh classifier for each run, with the pretrained dense
layer and norm and a new final layer initialized from the run's seed, and
predict with it, time it and store it.

```{.python .input #bert-run-defs}
%%tab pytorch
def pad(seqs):
    """Token-id lists padded to the longest, and the mask, on dev."""
    T = max(map(len, seqs))
    X = torch.tensor([s + [tok.pad_token_id] * (T - len(s)) for s in seqs])
    M = torch.tensor([[1] * len(s) + [0] * (T - len(s)) for s in seqs])
    return X.to(dev), M.to(dev)

def finetune(task, train, ev, lr, seed):
    """A fresh copy of the encoder under the pretrained dense and norm
    and a new final layer, initialized as in Transformers from seed."""
    c = encoder.config
    model = d2l.PooledClassifier(copy.deepcopy(encoder), len(LABELS[task]),
                                 pooling=c.classifier_pooling)
    model.head.load_state_dict(MLM_HEAD)
    torch.manual_seed(seed)  # the new final layer
    std, cut = c.hidden_size ** -0.5, c.initializer_cutoff_factor
    nn.init.trunc_normal_(model.classifier.weight, std=std, a=-cut * std,
                          b=cut * std)
    nn.init.zeros_(model.classifier.bias)
    return model, d2l.finetune_classifier(model, train, ev, lr, EPOCHS,
                                          BATCH, seed)

def predict(model, data, batch_size=64):
    with torch.inference_mode():
        return torch.cat([model(*pad([x for x, _ in data[i:i + batch_size]]))
                          .argmax(-1) for i in range(0, len(data),
                                                     batch_size)]).tolist()

def ms_per_1k(f, n, warmup=3, repeats=10):
    with torch.inference_mode():
        return 1e6 * d2l.Benchmark(f, warmup, repeats).time / n

def write_json(obj):
    def write(path):
        with open(path, 'w') as f:
            json.dump(obj, f)
    return write

def save_ckpt(model, task):
    """A writer for the classifier as a zip: bf16 weights (safetensors),
    the encoder's configuration and the head's settings."""
    head = dict(num_labels=len(LABELS[task]), pooling=model.pooling,
                labels=LABELS[task], base=ENC, revision=encoder.revision)
    members = {'model.safetensors': save_safetensors({
        k: v.to(torch.bfloat16).contiguous().cpu()
        for k, v in model.state_dict().items()}),
               'config.json': model.encoder.config.to_json_string().encode(),
               'head.json': json.dumps(head).encode()}
    return lambda path: d2l.lm_write_zip(path, members)

def load_ckpt(path, dtype):
    with zipfile.ZipFile(path) as z:
        config = transformers.AutoConfig.for_model(**json.loads(
            z.read('config.json')))
        head, state = (json.loads(z.read('head.json')),
                       load_safetensors(z.read('model.safetensors')))
    model = d2l.PooledClassifier(transformers.AutoModel.from_config(
        config, attn_implementation='sdpa'), head['num_labels'],
                                 head['pooling'])
    model.load_state_dict(state)
    return model.to(dev, dtype).eval()
```

### Seeds and Sample Size

How many labeled pairs does the encoder need? Guess before reading the table:
at what size does it clearly beat the majority rate of 0.363 on MNLI? The
sweep fine-tunes a fresh encoder at every size with three seeds. A seed fixes
the new layer's initial weights and the order of the batches; ModernBERT uses
no dropout. The learning rate is chosen per task at the largest size. All
four stored results of this section took about 27 minutes to compute on one
RTX 4090.

```{.python .input #bert-sweep}
%%tab pytorch
if 'bert-sweep' in RERUN:
    t0 = time.perf_counter()
    out = {'config': CONFIG['bert-sweep'],
           'eval': {t: {'idx': D['eval_idx'], 'label': [y for _, y in D[
               'eval']]} for t, D in DATA.items()},
           'lr': {}, 'runs': [], 'hyp_only': [], 'timing': {}}
    for task, D in DATA.items():  # learning rate on the carved items
        nc, carved, best = len(D['carve']), {}, None
        y_carve = np.array([y for _, y in D['carve']])
        for lr in LRS:  # the validation items ride along (not used here)
            model, r = finetune(task, D['train'], D['carve'] + D['eval'],
                                lr, 0)
            carved[lr] = float(np.mean(np.array(r['eval_pred'][:nc]) ==
                                       y_carve))
            if best is None or carved[lr] > carved[best[0]]:
                best = (lr, model, r)
            del model
            d2l.lm_free()
        lr = best[0]
        out['lr'][task] = {'chosen': lr, 'carved_acc': list(carved.values())}
        for n in SIZES[task]:
            for seed in SEEDS:
                if (n, seed) == (max(SIZES[task]), 0):  # the chosen run
                    _, model, r = best
                    best, r['eval_pred'] = None, r['eval_pred'][nc:]
                    r['eval_acc'] = float(np.mean(np.array(r['eval_pred']) ==
                                                  out['eval'][task]['label']))
                else:
                    model, r = finetune(task, D['train'][:n], D['eval'], lr,
                                        seed)
                out['runs'].append(dict(
                    task=task, n=n, seed=seed, lr=lr, acc=r['eval_acc'],
                    pred=r['eval_pred'], seconds=r['seconds'],
                    peak_gib=r['peak_gib']))
                if (n, seed) == (max(SIZES[task]), 0):  # timed in bf16
                    model = model.to(torch.bfloat16)
                    X = pad([x for x, _ in D['eval'][:BATCH]])
                    out['timing'][task] = ms_per_1k(lambda: model(*X), BATCH)
                    if task == 'mnli':  # what the saved weights predict
                        out['ckpt_pred'] = predict(model, D['eval'])
                        d2l.lm_save_rerun(ARTIFACTS['bert-ckpt'][1],
                                          save_ckpt(model, task))
                del model
                d2l.lm_free()
    for seed in SEEDS:  # MNLI from the hypothesis alone
        model, r = finetune('mnli', DATA['mnli']['hyp_train'], DATA['mnli'][
            'hyp_eval'], out['lr']['mnli']['chosen'], seed)
        out['hyp_only'].append(dict(seed=seed, acc=r['eval_acc'],
                                    pred=r['eval_pred']))
        del model
        d2l.lm_free()
    out['minutes'] = (time.perf_counter() - t0) / 60
    d2l.lm_save_rerun(ARTIFACTS['bert-sweep'][1], write_json(out))
    print(f'regenerated in {out["minutes"]:.1f} min; {d2l.lm_peak()}')
SWEEP = read_result('bert-sweep')
assert all(SWEEP['eval'][t]['idx'] == D['eval_idx'] for t, D in DATA.items())
runs = pd.DataFrame(SWEEP['runs'])
grid = runs.groupby(['task', 'n']).acc.agg(['mean', 'min', 'max'])
grid['range'] = grid['max'] - grid['min']
print('learning rates:', {t: v['chosen'] for t, v in SWEEP['lr'].items()})
print(grid[['mean', 'range']].round(3).to_string())
for task in SIZES:  # the seed range at the smallest size against size steps
    g = grid.loc[task]
    print(f'{task}: seed range at n = {g.index[0]}: {g["range"].iloc[0]:.3f}; '
          f'median gain per size step: {np.median(np.diff(g["mean"])):.3f}')
```

Both tasks chose the largest learning rate, $8 \times 10^{-5}$. SST-2 climbs
as one would expect, from 0.58 with 100 sentences to 0.93 with 8,000. MNLI does not. With 250
pairs the encoder reaches 0.376, hardly above the majority rate. With 1,000
it averages 0.47, but its seeds lie 0.158 apart: some runs have begun to
learn the task and others have not. From 4,000 pairs on it is a good
classifier, at 0.75 and then 0.84, and at 16,000 its seeds agree to within
0.003.

With few labels, the seed moves accuracy as much as the next step up in data.
From one size of the grid to the next, accuracy gains a median of 0.098 on
MNLI and 0.114 on SST-2. The seeds lie 0.158 apart at 1,000 MNLI pairs and
0.142 apart at 100 SST-2 sentences. Dodge et al. (2020) found the same for
BERT on four tasks of GLUE, the benchmark suite on which BERT made its name:
the seed alone moved validation scores about as much as newer methods claimed
to gain. Two fine-tuning methods compared at a few hundred examples with one
seed each may differ only in their seeds.

A small range is no comfort either. At 250 pairs the three seeds agree to
within 0.046, but only because all three sit near the majority rate.
Agreement among seeds shows that a result is reproducible. It does not show
that the result is good.

The left end of the MNLI curve measures the recipe as well as the encoder.
Two epochs of 250 pairs in batches of 32 are 16 optimizer steps, at a
learning rate chosen for 16,000 pairs and at the top of the grid. A schedule
and rate chosen for each size, with the present ones among the candidates,
can be expected to raise the small sizes' accuracies (Exercise
:exref:`ex_lm_bert-schedule`). The comparisons below are those of this
recipe, and at small sizes it errs in the decoder's favor.

Two live checks tie the stored results to the code. The first retrains one
cell of the grid, 1,000 MNLI pairs with seed 0.

```{.python .input #bert-sweep-live}
%%tab pytorch
n, k = LIVE['n'], LIVE['items']
y = np.array(SWEEP['eval']['mnli']['label'][:k])
model, r = finetune('mnli', DATA['mnli']['train'][:n], DATA['mnli']['eval'][
    :k], SWEEP['lr']['mnli']['chosen'], 0)
reg = np.array(runs.query('task == "mnli" and n == @n and seed == 0')
               .pred.iloc[0][:k])
live = np.array(r['eval_pred'])
gap, bound = abs(np.mean(live == y) - np.mean(reg == y)), max(
    0.02, grid.loc[('mnli', n), 'range'])  # the registered seeds' spread
print(f'n = {n}, seed 0, {k} items: accuracy {np.mean(live == y):.3f} live, '
      f'{np.mean(reg == y):.3f} registered\n  (bound on the difference '
      f'{bound:.3f}); same prediction on {np.mean(live == reg):.3f}; '
      f'{r["seconds"]:.0f} s')
assert SMALL or gap <= bound
del model
d2l.lm_free()
```

The live run reaches 0.529, as the stored run did, with the same prediction
on all 2,000 items. The second check loads the stored classifier, whose
weights were saved in bf16, and predicts the validation items. It stays
loaded for the timing below.

```{.python .input #bert-ckpt-live}
%%tab pytorch
clf16 = load_ckpt(result_path('bert-ckpt'),
                  torch.bfloat16 if GPU else torch.float32)
k = LIVE['items']
y = np.array(SWEEP['eval']['mnli']['label'][:k])
trained = np.array(runs.query('task == "mnli" and seed == 0').sort_values(
    'n').pred.iloc[-1][:k])  # the fp32 weights at the end of training
pred = np.array(predict(clf16, DATA['mnli']['eval'][:k]))
agree = np.mean(pred == np.array(SWEEP['ckpt_pred'][:k]))
print(f'registered checkpoint, {k} items: accuracy {np.mean(pred == y):.3f}; '
      f'same prediction\n  as when it was saved on {agree:.3f}, as its fp32 '
      f'training run on {np.mean(pred == trained):.3f}\n  {d2l.lm_peak()}')
assert SMALL or agree >= 0.99
```

The stored classifier scores 0.838 and repeats the predictions it made when
it was saved. With the fp32 weights at the end of training it agrees on 0.996
of the items, the few pairs that bf16 rounding moves across the decision
boundary.

### The Hypothesis-Only Probe
:label:`subsec_lm_bert_hypothesis_only`

Is 0.84 on MNLI reading comprehension? A simple test removes the premise. A
model that sees only the hypothesis cannot know whether it follows from a
premise it never read, so it should score the majority rate, 0.363.
Gururangan et al. (2018) ran this test on the inference datasets of the time
and found scores far above chance: annotators asked for a contradiction often
wrote a negation such as "nobody" or "never", and those asked for a neutral
hypothesis often added a detail that the premise did not mention. We fine-tune our encoder on 5,000 hypotheses
alone, with three seeds.

```{.python .input #bert-hypothesis-only}
%%tab pytorch
y = np.array(SWEEP['eval']['mnli']['label'])
majority = np.bincount(y).max() / len(y)
rows = {}
for r in SWEEP['hyp_only']:
    b = d2l.bootstrap_ci(np.array(r['pred']) == y,
                         cluster=DATA['mnli']['cluster'])
    rows[f'seed {r["seed"]}'] = {'accuracy': b.mean, 'CI low': b.ci[0],
                                 'CI high': b.ci[1]}
print(f'majority-class rate: {majority:.3f}')
print(pd.DataFrame(rows).T.round(3).to_string())
```

The seeds reach 0.523, 0.528 and 0.542, and their intervals start at 0.50 or
above. Without the premise the encoder still scores about 0.17 above the
majority rate. The full model's 0.84 can therefore mix two skills, comparing
the two texts and recognizing how annotators phrased each label, and its
accuracy alone cannot separate them. An inference score reported without a
hypothesis-only baseline does not say how much of it is reading.

## Token Tagging
:label:`subsec_lm_bert_tagging`

### Words and Sub-Word Tokens

A pooled head averages the token vectors away. A token head classifies each
of them. CoNLL-2003 (Tjong Kim Sang and De Meulder, 2003) marks every name
in a set of news sentences as a person (PER), a location (LOC), an
organization (ORG) or miscellaneous (MISC). The copy we load uses the *IOB2*
scheme: the first word of a name gets `B-` and its type, the other words of
the name get `I-` and the type, and all other words get `O`, nine tags in
all. Labels belong to words, and a word may be split into several tokens. The
usual rule, and ours, gives the word's label to its first token and $-100$ to
the rest, so the loss skips them and the prediction for a word is read at its
first token.

```{.python .input #bert-align}
%%tab pytorch
#@save
def align_token_labels(word_ids, word_labels, scheme='first'):
    """Token labels from word labels; word_ids gives each token's word
    (None for a special token). scheme='first' labels the first sub-token
    of each word and gives the others -100 (not scored); 'all' gives every
    sub-token its word's label."""
    if scheme not in ('first', 'all'):
        raise ValueError(f'unknown scheme: {scheme!r}')
    labels, prev = [], None
    for w in word_ids:
        first = w is not None and w != prev
        labels.append(word_labels[w] if first or (
            w is not None and scheme == 'all') else -100)
        prev = w
    return labels
```

### Span F1

Accuracy per word would be a poor measure. Most words lie outside any name,
and a tagger that answered `O` everywhere would look good. The shared task
scores whole names instead: a predicted name counts as correct only if its
type and both of its ends match a gold name. *Precision* is the share of
predicted names that are correct, *recall* the share of gold names that are
found, and *span F1* their harmonic mean. An `I-` tag after an `O`, or after
a tag of another type, starts a new name, as the official scorer and the
library `seqeval` count it.

```{.python .input #bert-span-f1}
%%tab pytorch
#@save
def span_f1(pred_tags, gold_tags, id2label):
    """Entity-level precision, recall and F1 of IOB2 tag sequences (a list
    of label ids per sentence), counted as conlleval and seqeval count
    them: an entity starts at a B- tag, or at an I- tag after O or after
    another type, and a predicted entity is correct if its type and both
    ends match; per_type holds the three numbers per entity type."""
    def entities(seqs):
        found = set()
        for s, seq in enumerate(seqs):
            start, kind = None, None
            for i, t in enumerate([id2label[t] for t in seq] + ['O']):
                bio, _, typ = t.partition('-')
                if start is not None and (bio != 'I' or typ != kind):
                    found.add((kind, s, start, i))
                    start = None
                if bio == 'B' or (bio == 'I' and start is None):
                    start, kind = i, typ
        return found

    def prf(pred, gold):
        tp = len(pred & gold)
        p = tp / len(pred) if pred else 0.0
        r = tp / len(gold) if gold else 0.0
        return dict(precision=p, recall=r, f1=2 * p * r / (p + r) if tp
                    else 0.0)
    pred, gold = entities(pred_tags), entities(gold_tags)
    kinds = sorted({e[0] for e in pred | gold})
    return dict(**prf(pred, gold), per_type={k: prf(
        {e for e in pred if e[0] == k}, {e for e in gold if e[0] == k})
        for k in kinds})
```

CoNLL-2003 comes as lists of words with their tags. We join the words with
spaces, tokenize the text, and map every token to its word through the
characters it covers. The tokenizer's own notion of a word would not do: it
splits at punctuation, so "U.S." would count as four words.

```{.python .input #bert-conll-data}
%%tab pytorch
conll = {s: d2l.read_jsonl(d2l.download_keyed(f'conll2003-{s}'))
         for s in ('train', 'valid')}
with open(d2l.download_keyed('conll2003-labels')) as f:
    id2label = {i: name for name, i in json.load(f).items()}

def encode_words(words, tags):
    """Token ids of the words joined by spaces, each token's word (through
    its character offsets), and the aligned token labels."""
    word_of_char = np.repeat(np.arange(len(words)), [len(w) + 1 for w in words])
    enc = tok(' '.join(words), return_offsets_mapping=True)
    word_ids = [None if s == e else int(word_of_char[e - 1])
                for s, e in enc['offset_mapping']]
    return enc['input_ids'], word_ids, d2l.align_token_labels(word_ids, tags)

TAGGED = {s: [encode_words(w, t) for w, t in zip(df.tokens, df.tags)]
          for s, df in conll.items()}
assert all(sum(l != -100 for l in labels) == len(t) for s, df in conll.items()
           for (_, _, labels), t in zip(TAGGED[s], df.tags))  # a label per word
print({s: len(v) for s, v in TAGGED.items()}, 'sentences')
i = next(i for i, (x, w, _) in enumerate(TAGGED['train'])  # a word split up
         if len(x) < 14 and len(set(w)) < len(w) - 1)
x, w, labels = TAGGED['train'][i]
print(pd.DataFrame({'token': tok.convert_ids_to_tokens(x),
                    'word': ['-' if v is None else v for v in w],
                    'label': [id2label.get(l, '-') for l in labels]}
                   ).T.to_string(header=False))
```

The training set has 14,041 sentences and the validation set 3,250. In the
example, "Blackburn" becomes `ĠBlack`, which carries `I-PER`, and `burn`,
which is not scored.

### The Token Head

Transformers' token classifier applies the sentence head's dense layer, GELU
and norm at every position, then a linear map to the nine tags. We fine-tune
it with the same function for two epochs at a fixed learning rate of
$5 \times 10^{-5}$. Running the notebook trains seed 0 live and compares its
span F1 with the three stored seeds and with `seqeval`.

```{.python .input #bert-tagging}
%%tab pytorch
def train_tagger(seed, train, valid):
    torch.manual_seed(seed)  # the new token head
    tagger, _ = d2l.load_encoder(ENC, head='token_cls', num_labels=len(
        id2label), dtype=torch.float32)
    r = d2l.finetune_classifier(tagger, [(x, l) for x, _, l in train],
                                [(x, l) for x, _, l in valid], TAG_LR,
                                EPOCHS, BATCH, seed)
    del tagger
    d2l.lm_free()
    return r

gold = conll['valid'].tags.tolist()
if 'bert-tagging' in RERUN:
    tagging = []
    for seed in SEEDS:
        r = train_tagger(seed, TAGGED['train'], TAGGED['valid'])
        tagging.append(dict(seed=seed, seconds=r['seconds'], peak_gib=r[
            'peak_gib'], **d2l.span_f1(r['eval_pred'], gold, id2label)))
    d2l.lm_save_rerun(ARTIFACTS['bert-tagging'][1], write_json(
        {'config': CONFIG['bert-tagging'], 'runs': tagging}))
TAGGING = read_result('bert-tagging')
r = train_tagger(0, TAGGED['train'][:LIVE['tag_train']],
                 TAGGED['valid'][:LIVE['tag_items']])
gold = gold[:LIVE['tag_items']]
f1 = d2l.span_f1(r['eval_pred'], gold, id2label)
def names(seqs):
    return [[id2label[t] for t in s] for s in seqs]

oracle = seqeval_f1(names(gold), names(r['eval_pred']))
print(f'live, seed 0, {len(gold)} sentences: precision {f1["precision"]:.3f}, '
      f'recall {f1["recall"]:.3f}, F1 {f1["f1"]:.4f}; seqeval F1 {oracle:.4f}')
print(pd.DataFrame(f1['per_type']).round(3).to_string())
seeds_f1 = {x['seed']: x['f1'] for x in TAGGING['runs']}
bound = max(0.01, np.ptp(list(seeds_f1.values())))  # the seeds' spread
print(f'registered, {len(seeds_f1)} seeds: F1 {min(seeds_f1.values()):.3f} '
      f'to {max(seeds_f1.values()):.3f}, seed 0 {seeds_f1[0]:.4f} (bound on '
      f'the live difference {bound:.3f})\n  {d2l.lm_peak()}')
assert abs(f1['f1'] - oracle) < 1e-9
assert SMALL or abs(f1['f1'] - seeds_f1[0]) <= bound
```

The tagger finds 95% of the names (recall 0.951), 94% of the names it
predicts are right (precision 0.939), and its span F1 of 0.945 equals
`seqeval`'s. The stored seeds lie between 0.945 and 0.951, so the token head
reaches a span F1 of about 0.95. Locations and persons are nearly solved, at
0.97. Organizations reach 0.92; many of them, sports clubs above all, carry
the name of a city. The catch-all MISC of nationalities, events and titles
reaches 0.87. All of this comes from a small head on one vector per token,
fine-tuned together with the encoder. Nobody wrote a rule about capital
letters or a list of cities.

## Encoder or Prompted Decoder
:label:`subsec_lm_encoder_vs_decoder`

### The Prompted Decoder

Back to the opening question: when does the fine-tuned encoder beat a
prompted decoder, and at what cost? The decoder labels the validation items from $k$ = 0, 4 or 8 solved examples in its prompt. As in
:numref:`sec_lm_scoring`, each label word (`yes`, `maybe` or `no` for MNLI,
`negative` or `positive` for SST-2) is appended to the prompt in turn, and the
word with the highest log-probability per token wins. The choice of examples
moves few-shot accuracy much as a seed moves fine-tuning, so 8 shots are
scored with three disjoint sets of examples. The 4-shot prompts use the first
four of one set, a single draw.

In bf16 an item's score depends on the other items in its batch
(:numref:`sec_lm_scoring`), so the live check re-scores all 2,000 MNLI items
in the batches of the stored run. It sets aside *near ties*, items whose two
best labels differ by at most 0.5 nats per token, where rounding could flip
the choice, and requires every other prediction to repeat.

```{.python .input #bert-decoder}
%%tab pytorch
TIE = 0.5  # nats per token: margins bf16 moves when the batches differ

def query(task, row):
    if task == 'mnli':
        return (f'{row.premise.strip()}\nQuestion: {row.hypothesis.strip()} '
                'Yes, no, or maybe?')
    return row.sentence.strip()

def classify(task, k, rows, draw=0):
    """Scores (items, labels): each label word's log-probability per token
    after a prompt with k demonstrations of set draw."""
    demos = [(query(task, r), VERBAL[task][r.label])
             for r in DATA[task]['shots'][draw].iloc[:k].itertuples()]
    prompts = [d2l.fewshot_prompt(demos, query(task, r), fmt=FMT[task])
               for r in rows.itertuples()]
    return d2l.score_choices(dec, dtok, prompts, [' ' + v for v in VERBAL[
        task]], norm='token', batch_size=16).numpy()

dec, dtok = d2l.load_hf(DEC, dtype=torch.bfloat16 if GPU else torch.float32)
if 'bert-decoder' in RERUN:
    out = {'config': CONFIG['bert-decoder'],
           'eval': {t: D['eval_idx'] for t, D in DATA.items()},
           'pred': {}, 'timing': {}}
    for task, D in DATA.items():
        scores = {k: [classify(task, k, D['eval_df'], d)
                      for d in range(DRAWS[k])] for k in SHOTS}
        out['pred'][task] = {k: [s.argmax(-1).tolist() for s in v]
                             for k, v in scores.items()}
        if task == 'mnli':  # for the live check
            out['scores'] = np.round(scores[8][0], 4).tolist()
        out['timing'][task] = {k: ms_per_1k(lambda: classify(
            task, k, D['eval_df'].iloc[:BATCH]), BATCH, 1, 3) for k in SHOTS}
    d2l.lm_save_rerun(ARTIFACTS['bert-decoder'][1], write_json(out))
DECODER = read_result('bert-decoder')
assert all(DECODER['eval'][t] == D['eval_idx'] for t, D in DATA.items())
rows = {}
for task in DATA:
    for k in SHOTS:
        for d, pred in enumerate(DECODER['pred'][task][str(k)]):
            b = d2l.bootstrap_ci(np.array(pred) == np.array(SWEEP['eval'][
                task]['label']), cluster=DATA[task]['cluster'])
            rows[task, k, d] = {'accuracy': b.mean, 'CI low': b.ci[0],
                                'CI high': b.ci[1]}
print(pd.DataFrame(rows).T.rename_axis(['task', 'shots', 'draw']).round(3)
      .to_string())
m = LIVE['dec_items']
live = classify('mnli', 8, DATA['mnli']['eval_df'].iloc[:m])
reg = np.array(DECODER['scores'][:m])
top2 = np.sort(reg, -1)[:, -2:]
clear = top2[:, 1] - top2[:, 0] > TIE  # not a near tie
agree = np.mean(live.argmax(-1)[clear] == reg.argmax(-1)[clear])
print(f'live, 8 shots, {m} MNLI items: largest score change '
      f'{np.abs(live - reg).max():.4f} nats per token\n  {np.sum(~clear)} '
      f'near ties (margin <= {TIE}); same prediction on {agree:.3f} of the '
      f'others\n  {d2l.lm_peak()}')
assert SMALL or agree == 1.0
```

The live scores repeat the stored ones to 0.0001 nats per token, and every
prediction outside the 124 near ties is unchanged. On MNLI the decoder
reaches 0.40 without examples, a little above the majority rate, and 0.62 to
0.65 with four or eight. On SST-2 it starts at 0.71 and reaches 0.935 to
0.946. Sentiment is everywhere in the text a decoder is pretrained on, so a
few examples suffice to name the task. Which pairs MNLI calls entailment,
neutral or contradiction is a convention of its annotators, and eight
examples convey it only in part.

### Baselines on Word Counts

Is pretraining needed at all? The classical floor is naive Bayes on word
counts (:numref:`sec_mdl-naive_bayes`), which treats the words of a text as
independent once the class is known: training is counting, and prediction is
multiplying. A pair becomes two bags of words, with premise and hypothesis
words kept apart by a prefix. (Before transformers, the usual learned
sentence classifier slid a one-dimensional convolution over the word vectors
of a sentence and kept the maximum of each filter over the positions, a
detector of informative phrases wherever they occur :cite:`Kim.2014`; we do
not train one here.)

```{.python .input #bert-naive-bayes}
%%tab pytorch
words = CountVectorizer().build_analyzer()  # lowercase words

def bag(task, df):
    if task == 'sst2':
        return [words(s) for s in df.sentence]
    return [['p:' + w for w in words(p)] + ['h:' + w for w in words(h)]
            for p, h in zip(df.premise, df.hypothesis)]

NB = {}
for task, D in DATA.items():
    for n in SIZES[task]:
        vec = CountVectorizer(analyzer=lambda ws: ws)
        X = vec.fit_transform(bag(task, D['train_df'].iloc[:n]))
        nb = MultinomialNB().fit(X, D['train_df'].label.iloc[:n])
        NB[task, n] = nb.score(vec.transform(bag(task, D['eval_df'])),
                               D['eval_df'].label)
print(pd.Series(NB).unstack(0).round(3).to_string())
```

Naive Bayes reaches 0.53 to 0.78 on SST-2 as its data grow, but only 0.38 to
0.42 on MNLI, barely above the majority rate. Word counts carry sentiment.
They carry little entailment, which depends on how the words of the
hypothesis relate to those of the premise, and two bags of words cannot
represent that.

### Cost per Example

The encoder reads about 40 tokens per MNLI pair. The decoder reads its eight
solved examples again for every item, and three times over, since
`score_choices` runs the prompt once for each label word. How much
arithmetic is that? Every weight matrix multiplies the vector of every token,
and each weight contributes a multiply and an add per token. A model with $N$
parameters, $N_\textrm{emb}$ of them in the embedding table, spends about

$$
C = 2\, (N - N_\textrm{emb})\, L
$$
:eqlabel:`eq_lm_bert_flops`

floating-point operations on an example of $L$ tokens. Embedding lookups are
not products, and attention's $L^2$ term is small at a few hundred tokens.
The cost thus grows with the weights each token passes through and with the
number of tokens read. A training step costs about three forward passes,
since the backward pass computes two products for each one of the forward
pass. `cost_per_example`, defined with the fine-tuning helpers, evaluates
:eqref:`eq_lm_bert_flops`. The cell times both models on 32 MNLI items in
bf16 and compares the measured ratio with the estimate.

```{.python .input #bert-timing}
%%tab pytorch
B = BATCH if GPU else 4  # decoder items timed
warm, reps = (3, 10) if GPU else (0, 1)
X = pad([x for x, _ in DATA['mnli']['eval'][:BATCH]])
items = DATA['mnli']['eval_df'].iloc[:B]
ms = {'encoder': ms_per_1k(lambda: clf16(*X), BATCH, warm, reps),
      'decoder': ms_per_1k(lambda: classify('mnli', 8, items), B,
                           min(warm, 1), max(1, reps // 3))}
demos = [(query('mnli', r), VERBAL['mnli'][r.label])
         for r in DATA['mnli']['shots'][0].iloc[:8].itertuples()]
tokens = {'encoder': X[1].sum(1).float().mean().item(),  # without padding
          'decoder': np.mean([sum(len(dtok(
    d2l.fewshot_prompt(demos, query('mnli', r), fmt=FMT['mnli']) + ' ' + v)
    ['input_ids']) for v in VERBAL['mnli']) for r in items.itertuples()])}
flops = {'encoder': d2l.cost_per_example(
    sum(p.numel() for p in clf16.parameters()), tokens['encoder'],
    clf16.encoder.get_input_embeddings().weight.numel()),
         'decoder': d2l.cost_per_example(
    sum(p.numel() for p in dec.parameters()), tokens['decoder'],
    dec.get_input_embeddings().weight.numel())}
print(pd.DataFrame({'tokens per example': tokens, 'GFLOPs per example':
                    {m: f / 1e9 for m, f in flops.items()},
                    'ms per 1,000, live': ms, 'ms per 1,000, registered': {
                        'encoder': SWEEP['timing']['mnli'],
                        'decoder': DECODER['timing']['mnli']['8']}}
                   ).round(1).to_string())
print(f'decoder / encoder: measured {ms["decoder"] / ms["encoder"]:.0f}x, '
      f'FLOPs {flops["decoder"] / flops["encoder"]:.0f}x; '
      f'{torch.cuda.get_device_name() if GPU else "CPU"}, attention '
      f'{clf16.encoder.config._attn_implementation}')
del dec, clf16
d2l.lm_free()
```

The decoder reads 1,543 tokens per item against the encoder's 40, about 38
times as many. It spends 2.8 GFLOPs per token against the encoder's 0.22,
about 13 times as much, which is the ratio of their non-embedding weights.
Together that makes 486 times the encoder's operations per example. The clock
gives about 100 times: under half a second against 46 to 48 seconds per
thousand items. Per second, then, the decoder computes four to five times as
much: batches of 32 short pairs give the encoder small matrix products that
leave the GPU partly idle, while the decoder's long prompts fill it. Most of
the decoder's cost goes to reading the same solved examples again. A server
that computes their keys and values once and reuses them removes most of it
(Exercise :exref:`ex_lm_bert-prefix`).

### Learning Curves and Crossing Points

The plot shows accuracy against labeled examples (left) and against seconds
per thousand examples (right), both on log scales, MNLI on top. The bands
span the encoder's seeds and the decoder's three sets of 8 solved examples;
at 0 and 4 shots they are the interval of one set. The dashed line is the
majority rate.

```{.python .input #bert-curve}
%%tab pytorch
def correct(pred, task):
    return np.array(pred) == np.array(SWEEP['eval'][task]['label'])

def decoder_band(task, k):
    """Accuracy at k shots: mean, low, high over the demonstration sets, or
    the bootstrap interval of a single set."""
    preds = DECODER['pred'][task][str(k)]
    if len(preds) > 1:
        accs = [correct(p, task).mean() for p in preds]
        return np.mean(accs), min(accs), max(accs)
    b = d2l.bootstrap_ci(correct(preds[0], task), cluster=DATA[task]['cluster'])
    return b.mean, *b.ci

_, axes = d2l.plt.subplots(2, 2, figsize=(8, 6.4))
for ax, task in zip(axes, ('mnli', 'sst2')):
    acc = runs.query('task == @task').groupby('n').acc
    sizes, big = acc.mean().index.to_numpy(), acc.get_group(acc.mean().index[-1])
    dec_acc = {k: decoder_band(task, k) for k in SHOTS}
    series = {'encoder (seed range)': (acc.mean(), acc.min(), acc.max())}
    series |= {f'decoder, {k} shots' + (f' ({DRAWS[k]} sets)' if DRAWS[k] > 1
                                        else ''): tuple([v] * len(sizes) for v
                                                        in b)
               for k, b in dec_acc.items()}
    series['naive Bayes'] = ([NB[task, n] for n in sizes],) * 3
    majority = np.bincount(SWEEP['eval'][task]['label']).max() / len(
        SWEEP['eval'][task]['label'])
    d2l.plot_intervals(sizes, series, 'labelled examples', f'{task} accuracy',
                       kind='band', xscale='log', reference=majority,
                       axes=ax[0])
    sec = [SWEEP['timing'][task] / 1e3] + [DECODER['timing'][task][str(
        k)] / 1e3 for k in SHOTS]  # x: one cost per model
    none = [np.nan] * len(SHOTS)
    d2l.plot_intervals(sec, {
        f'encoder, n = {sizes[-1]:,}': ([big.mean(), *none],
                                       [big.min(), *none], [big.max(), *none]),
        'decoder, 0, 4, 8 shots': tuple([np.nan] + [b[i] for b in
                                                    dec_acc.values()]
                                        for i in range(3))},
        'seconds per 1,000 examples', f'{task} accuracy', xscale='log',
        reference=majority, axes=ax[1])
    ax[1].set_xscale('log')  # standard log ticks, not one per cost
for top, bottom in zip(*axes):  # the top row's legends serve both rows
    top.legend(fontsize='small')
    bottom.get_legend().remove()
```

On MNLI the encoder's curve climbs through the decoder's lines between 1,000
and 4,000 pairs. On SST-2 it meets them only at the right edge. On the right,
the encoder sits two decades to the left of the 8-shot decoder, at equal or
higher accuracy.

Where does the encoder pass the decoder? We call it ahead at a size only if
no choice of seed and no choice of solved examples reverses the comparison,
and the gap exceeds either kind of spread. The rule, fixed before the runs,
pairs each of the three seeds with each of the three sets of examples. For
each pairing the cell computes the paired bootstrap interval of the
difference in accuracy, which scores both models on the same resampled items
(by premise on MNLI), so that the difficulty they share cancels. A size is
*passed* when

- all nine intervals exclude zero on the encoder's side, and
- the seed range and the range over the sets are both narrower than the mean
  gap.

The *crossing point* is the smallest size from which every larger size is
passed.

```{.python .input #bert-crossing}
%%tab pytorch
def crossing(task):
    """Per size: the encoder against 8 shots for every (seed, demonstration
    set) pair; the crossing is the smallest size from which on every size
    is 'encoder ahead'."""
    cl, rows = DATA[task]['cluster'], {}
    decs = [correct(p, task) for p in DECODER['pred'][task]['8']]
    dec_range = np.ptp([d.mean() for d in decs])
    for n, group in runs.query('task == @task').groupby('n'):
        diffs = [d2l.paired_bootstrap(correct(p, task), d8, cluster=cl)
                 for p in group.pred for d8 in decs]
        accs, gap = group.acc.to_numpy(), np.mean([d.mean for d in diffs])
        signs = {int(np.sign(d.ci[0])) if d.ci[0] * d.ci[1] > 0 else 0
                 for d in diffs}
        clear = len(signs) == 1 and max(np.ptp(accs), dec_range) < abs(gap)
        rows[n] = {'seeds': len(accs), 'encoder': accs.mean(),
                   'seed range': np.ptp(accs), 'gap to 8 shots': gap,
                   'CIs from': min(d.ci[0] for d in diffs),
                   'CIs to': max(d.ci[1] for d in diffs), 'verdict': (
                       {1: 'encoder ahead', -1: 'decoder ahead'}[signs.pop()]
                       if clear and 0 not in signs else 'within noise')}
    table = pd.DataFrame(rows).T.infer_objects()
    ahead = (table.verdict == 'encoder ahead').to_numpy()
    on = np.logical_and.accumulate(ahead[::-1])[::-1]  # and every larger n
    return table, dec_range, table.index[on][0] if on.any() else None

for task in ('mnli', 'sst2'):
    table, dec_range, n = crossing(task)
    print(f'{task}: the encoder passes 8 shots from n = '
          f'{n or "none measured"} on\n  range over the {DRAWS[8]} '
          f'demonstration sets {dec_range:.3f}')
    print(table.to_string(float_format=lambda v: f'{v:.3f}'))
```

On MNLI the decoder leads at 250 pairs, by 0.255 (the nine intervals lie
between 0.19 and 0.32). At 1,000 pairs it is still ahead in all nine
pairings, by 0.16 on average with intervals from 0.06 to 0.30. The rule
withholds a verdict there only because the seeds lie as far apart, 0.158, as
the gap is wide. From 4,000 on the encoder leads, by 0.12 (the nine intervals
lie between 0.02 and 0.20) and by 0.21 at 16,000 (between 0.17 and 0.25).
With this recipe, the fine-tuned encoder passes the 8-shot decoder on MNLI
once it has seen between 1,000 and 4,000 labeled pairs. On SST-2 the decoder
leads up to 2,000 sentences, still by 0.046 there (the nine intervals lie
between 0.006 and 0.088). At 8,000 the gap is within noise, $-0.009$ with
intervals from $-0.037$ to 0.022: the encoder catches up on sentiment but
does not pass.

These crossing points count labels, which cost annotators' time that eight
solved examples do not; that is why the plot measures both models against
labeled examples. Where the line runs depends on how much of the task the
decoder already knows. The cost per item does not: at every size the encoder labels an item in about a hundredth of the decoder's time.

### Choice by Situation

:numref:`tab_lm_bert_when` turns these measurements into advice.

:A fine-tuned encoder or a prompted decoder, by situation.
:label:`tab_lm_bert_when`

| situation | prefer | because |
|---|---|---|
| no labels, or a few hundred | prompted decoder | at 250 MNLI pairs and at 100 to 2,000 SST-2 sentences the decoder was ahead |
| thousands of labels for a fixed task | fine-tuned encoder | on MNLI it passed 8 shots from 4,000 pairs; on SST-2 it caught up at 8,000 |
| many items, high throughput | fine-tuned encoder | about a hundredth of the time per item |
| a label set that changes often | prompted decoder | a new label is a new prompt, not a new training run |
| a label for every word | fine-tuned encoder | one vector per token, read by a small head |
| an answer that must be written or explained | decoder | an encoder head outputs one of a fixed set of labels |

Each task in this section got a model of its own, trained on its own labels.
Search needs something else. A question must be compared with millions of
passages, far too many to pair with each one in a cross-encoder, and there
are no labels to fine-tune on. It needs one vector per text, computed once,
that serves every query. Averaging the token vectors of this encoder is the
obvious candidate, and :numref:`sec_lm_text_embeddings` measures how well it
retrieves and what training it needs.

## Summary

A masked-token encoder computes one vector per token from the text on both
sides, and a small head, fine-tuned together with all of the encoder's
weights, turns these vectors into a label for a sentence, for a pair of
sentences read together, or for every word. Fine-tuned this way,
ModernBERT-base reached 0.84 on MNLI with 16,000 pairs, 0.93 on SST-2 with
8,000 sentences, and a span F1 of about 0.95 on CoNLL-2003. With few labels
the seed moved accuracy as much as the next step up in data, and a model that
never saw the premise still reached about 0.53 on MNLI against a majority
rate of 0.36. Against Qwen3-1.7B with 8 solved examples, the encoder, trained
for two epochs at every size, passed on MNLI between 1,000 and 4,000 labeled
pairs and caught up on SST-2 at 8,000, at about a hundredth of the time per
example. Prompting suits few labels and changing tasks; a fine-tuned encoder
suits a fixed task with thousands of labels and many items to label.

## Exercises

1. [code] **Peak memory and batch size.** Fine-tune the MNLI classifier on
   4,000 pairs for one epoch with batch sizes 8, 16, 32 and 64. Pad every
   batch to 128 tokens, so that the length does not vary, and record
   `peak_gib` from `finetune_classifier`.
    1. Fit the peak as a linear function of the batch size. Compare the
       intercept with the 16 bytes per parameter of the weights, gradients and
       AdamW moments, and the slope with an estimate of the activations kept
       per sequence: how many vectors of width 768 or 1,152 per token and
       layer does the slope correspond to?
    1. Predict the largest batch that fits on a 24 GB card, and check it.
    1. Under autocast most saved activations are bf16 (layer norms run and
       save in fp32). How large would the slope be if all were kept in fp32,
       and which batch would then fit?
   :exlabel:`ex_lm_bert-memory`

1. **The masked-token loss under 80/10/10.** Let $p = 0.15$ be the share of
   selected tokens, $V$ the vocabulary size, and $\mathcal{S}$ the selected
   positions; the loss is $-\sum_{t \in \mathcal{S}} \log q_t(x_t)$, where
   $q_t$ is the model's distribution at position $t$ given the corrupted
   input.
    1. Show that the gradient of the loss with respect to the logits at an
       unselected position is zero. How, then, does the vector at such a
       position receive any gradient at all?
    1. A position shows an ordinary token $v$ whose frequency in the data is
       $f$. Compute, as functions of $f$, $p$ and $V$, the probability that
       the position is scored and the probability that its target is $v$
       given that it is scored. Evaluate both for $V = 50{,}368$, once with
       $f = 1/V$ (tokens drawn uniformly, as in the cell above) and once with
       $f = 10^{-3}$.
    1. Use the result to explain why a model trained this way predicts the
       token it sees with high probability at most positions, and why this
       is useful for fine-tuning. What would the loss teach the model at a position
       showing $v$ if every selected token were replaced by `[MASK]`?

1. [code] **What the hypothesis gives away.** Fit naive Bayes on the bags of
   MNLI hypothesis words alone, with the vectorizer of the naive Bayes cell
   and the 5,000 training hypotheses of the hypothesis-only encoder.
    1. Compare its accuracy per class with that of the hypothesis-only
       encoder of :numref:`subsec_lm_bert_hypothesis_only`. Which class does
       each predict best?
    1. List the ten words with the largest ratio
       $p(\textrm{word} \mid \textrm{class}) / p(\textrm{word})$ for each
       class. What do they say about how the hypotheses were written?
    1. On the validation pairs the full encoder gets right, how often does
       the hypothesis-only encoder get them right too? What does this suggest
       about the share of the 0.84 that needs the premise?

1. [code] **First token or mean.** Fine-tune `PooledClassifier` with
   `pooling='cls'` on 4,000 MNLI pairs with three seeds, and compare it with
   the stored mean-pooling runs by paired bootstrap on the same items.
   Before you run it, write down which result would change your mind about
   reading `classifier_pooling` from the configuration, and which would
   not. Then fine-tune both poolings once more with a freshly initialized
   dense layer and norm. Does the pretrained pair help either pooling?
   :exlabel:`ex_lm_bert-pooling`

1. **Prefix caching and break-even.** The 8-shot decoder of this section
   reads three copies of a prompt whose first part, the solved examples, is
   the same for every item.
    1. With the shared prefix cached, how many tokens does each item still
       cost? Measure the length of the prefix and of a query, and recompute
       the ratio of :eqref:`eq_lm_bert_flops` against the encoder.
    1. Qwen3-1.7B stores keys and values for 8 heads of width 128 in each of
       28 layers, $2 \cdot 28 \cdot 8 \cdot 128$ numbers of two bytes per
       token (:numref:`subsec_lm_icl_vs_probe`). How much memory does the
       cached prefix take, and how does it compare with the cache of one
       batch of 32 queries?
    1. Fine-tuning costs about three forward passes per example and pass
       over the data. Using :eqref:`eq_lm_bert_flops`, after how many
       queries is fine-tuning on 4,000 pairs plus encoder inference
       cheaper in operations than the cached 8-shot prompt? Is the answer
       different if you count annotators' time for the 4,000 labels?
   :exlabel:`ex_lm_bert-prefix`

1. [code] **A schedule for small data.** The recipe trains two passes at
   every size, so 250 MNLI pairs get 16 steps.
    1. At 250 and 1,000 pairs, train with three seeds for a fixed number of
       steps instead (try 300 and 1,000), choosing the learning rate on the
       held-out pairs at that size.
    1. Recompute the crossing with the 8-shot decoder. Does it move below
       1,000 pairs?
    1. Did the seed range at 1,000 pairs shrink? What does that say about the
       source of the spread in the stored grid?
   :exlabel:`ex_lm_bert-schedule`

1. [code] **Extractive question answering.** Put a start logit and an end
   logit on every context token of a pair `[CLS] question [SEP] context [SEP]`
   and fine-tune for one epoch on 10,000 questions of SQuAD v1.1, whose
   answers are spans of the context. Predict the span with the highest sum of
   start and end logits whose end does not precede its start, and report
   exact match and token-level F1 on its validation set. Collect the failed
   questions whose answer is a name, and run the tagger of
   :numref:`subsec_lm_bert_tagging` over their contexts. Does it mark the
   answer as a name?

1. [code] **A CRF on the token head.** The token head predicts each word's
   tag on its own, so it may output `I-ORG` after `B-PER`. Add a linear-chain
   conditional random field on top of the logits at first tokens, train it
   jointly by maximizing the log-likelihood of the gold tag sequence, and
   decode with the Viterbi algorithm. Measure the change in span F1 over
   three seeds against the stored range of 0.945 to 0.951, and count the
   invalid tag transitions with and without the CRF.

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §19.2]{.kicker}

Pretrained Encoders<br>**masked tokens · ModernBERT · three heads · seeds · encoder or prompt**
:::
:::

::: {.slide title="A Fine-Tuned Encoder Labels MNLI Better Than a Prompt, in a Hundredth of the Time"}
[The problem]{.kicker}

MNLI: does a premise entail, contradict, or leave open a hypothesis? The
same 2,000 validation pairs for both models, one RTX 4090.

| model | labeled examples | accuracy | seconds per 1,000 pairs |
|---|---|---|---|
| ModernBERT-base, fine-tuned (mean of 3 seeds) | 16,000 | 0.84 | 0.4 |
| Qwen3-1.7B, 8 solved examples (mean of 3 sets) | 8 | 0.63 | 48 |

::: {.d2l-note .warn}
The encoder learned from 16,000 labels, the decoder from 8. How many labels
does the encoder need before it passes the prompt?
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- say what a masked-token encoder computes, and why BERT masks only 80% of
  the tokens it scores;
- read an encoder's configuration and compute what alternating attention
  saves;
- fine-tune a sentence, a pair and a token head, and check one against the
  library;
- read a seed range and a hypothesis-only baseline before trusting a score;
- choose between a fine-tuned encoder and a prompted decoder from labels and
  cost.

**Running example:** ModernBERT-base (149 million parameters) against
Qwen3-1.7B on MNLI (pairs), SST-2 (sentiment) and CoNLL-2003 (names).
:::

::: {.slide title="Labeling a Stream of Text Is an Encoder's Job"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| movement | section |
|---|---|
| vectors | Word Embeddings |
| | **Pretrained Encoders** |
| | Text Embeddings |
| finding | Sparse, Dense and Approximate Retrieval |
| using what is found | Retrieval-Augmented Generation |
:::

::: {.col .narrow}
- **Uses:** a bidirectional encoder trained on masked bytes; few-shot
  prompts read by label log-probabilities.
- **Skips:** pretraining an encoder; tuning only part of the weights.
:::
:::
:::

::: {.slide title="A Masked-Token Encoder Fills a Gap from Both Sides"}
[A released encoder]{.kicker}

No causal mask: every token attends to every token. Pretraining hides
tokens and predicts them, $\max \sum_{t \in \mathcal{M}} \log p(x_t \mid \mathbf{x}_{\setminus \mathcal{M}})$,
with $\mathcal{M}$ the hidden positions.

| text | five most probable fills |
|---|---|
| The capital of France is `[MASK]`. | Paris 0.92, Lyon 0.04, Nancy 0.02, Nice 0.01, Orleans 0.00 |
| He drank a cup of `[MASK]` before work. | coffee 0.61, tea 0.21, water 0.08, milk 0.04, wine 0.01 |
| The encoder reads the words on `[MASK]` sides of the gap. | both 0.94, the 0.03, opposite 0.01, two 0.01, all 0.01 |

::: {.d2l-note}
"both" needs the word after the gap. The head predicts words, not labels.
:::
:::

::: {.slide title="BERT Scores 15% of Tokens, and Only 80% of Those Show [MASK]"}
| of the selected 15% | becomes | why |
|---|---|---|
| 80% | `[MASK]` | the prediction task |
| 10% | a random token | a visible token may be wrong |
| 10% | itself | a visible token may be scored |

Measured on a million token ids: selected 0.1506; of those, `[MASK]` 0.798,
random 0.101, kept 0.101; special tokens never.

::: {.d2l-note}
`[MASK]` never occurs in fine-tuning text. Every position must keep a guess
of its own token, because any of them may be asked.
:::
:::

::: {.slide title="Three Successors Changed the Training, the Objective, the Architecture"}
| model | the change |
|---|---|
| RoBERTa | no next-sentence prediction; new masks each time a text is read; longer training on far more text |
| ELECTRA | a small generator replaces tokens; the encoder says for **every** token whether it was replaced |
| T5 | encoder--decoder; masked spans are written by the decoder; every task is text to text |

- RoBERTa: BERT was undertrained, not misdesigned.
- ELECTRA: a training signal at every position, not 15%.
- T5: the step toward decoders that do everything by writing text.

Next-sentence prediction: BERT's second objective, on the `[CLS]` vector;
did text B follow text A?
:::

::: {.slide title="ModernBERT Keeps BERT's Objective and Takes a Decoder's Block"}
::: {.cols .vc}
::: {.col}
| field | ModernBERT-base |
|---|---|
| layers, width, heads | 22, 768, 12 |
| feed-forward | GeGLU, width 1,152 |
| positions | 8,192, rotary |
| rotary base | 10,000 local, 160,000 global |
| attention | every third layer global (all tokens), others local (window of 128, 64 each side) |
| biases | none in the blocks |
| vocabulary | 50,368 |
| pooling in its head | mean |
| parameters | 149,014,272 |
:::

::: {.col .narrow}
- **Unpadding**: a batch is packed into one row; needs FlashAttention.
  With PyTorch's fused attention the batches are padded: same results,
  other speed.
- A larger rotary base keeps far positions apart.
:::
:::
:::

::: {.slide title="No Segment Ids: the Separator Alone Marks the Second Text"}
`[CLS] A Ġman Ġplays Ġa Ġguitar Ġon Ġstage . [SEP] Someone Ġis Ġmaking Ġmusic . [SEP]`

- `[CLS]` opens the input; `[SEP]` closes each text.
- Fields: `input_ids`, `attention_mask`; no `token_type_ids`, no segment
  table.
- Output (1, 16, 768): a vector per token, from the whole pair; "bank"
  by a river and "bank" by a teller get different vectors.
- Read jointly, every hypothesis token attends to every premise token: a
  **cross-encoder**.

::: {.d2l-note .rule}
Keep the separator and the order. Cut long pairs from the premise's end.
:::
:::

::: {.slide title="Alternating Attention Scores 37% of the Pairs at 8,192 Tokens"}
$$\rho(L) = \frac{n_g L^2 + n_l L w}{(n_g + n_l) L^2} = \frac{n_g L + n_l w}{(n_g + n_l) L}$$

$n_g = 8$ global and $n_l = 14$ local layers, window $w = 128$ (64 on each
side), input of $L$ tokens. Pairs counted from boolean masks:

| $L$ | per global layer | per local layer | share, counted | share, formula |
|---|---|---|---|---|
| 512 | 262,144 | 61,888 | 0.514 | 0.523 |
| 8,192 | 67,108,864 | 1,052,608 | 0.374 | 0.374 |

::: {.d2l-note}
For long inputs $\rho \to n_g / (n_g + n_l) = 8/22$: the global layers set
the cost.
:::
:::

::: {.slide title="Check: When Does the Window Save Nothing?"}
[Check]{.kicker}

A local layer lets token $i$ attend to $j$ when $|i - j| \le 64$. An MNLI
pair has about 40 tokens. What does alternating attention save on it?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** Nothing. For $L \le 65$ every window covers the whole input,
and every local layer acts as a global one. The saving comes only on long
documents.
:::
:::
:::

::: {.slide title="One Encoder Serves Three Tasks; Only the Input and the Head Change"}
[Sentence and pair classification]{.kicker}

| task | input | vectors read | output |
|---|---|---|---|
| sentence (SST-2) | `[CLS] sentence [SEP]` | mean over tokens | 2 logits |
| pair (MNLI) | `[CLS] premise [SEP] hypothesis [SEP]` | mean over tokens | 3 logits |
| tagging (CoNLL-2003) | `[CLS] sentence [SEP]` | each word's first token | 9 logits per word |

::: {.d2l-note}
Each head: dense, GELU and norm taken from the pretrained masked-token
head, then a new linear map. Fine-tuning trains all weights.
:::
:::

::: {.slide title="Our Pooled Head Reproduces the Library's Logits"}
$$\bar{\mathbf{h}} = \frac{1}{T}\sum_{t=1}^{T} \mathbf{h}_t, \qquad \mathbf{z} = \mathbf{W}_\textrm{c}\, \textrm{LN}\big(\textrm{GELU}(\mathbf{W}_\textrm{d}\bar{\mathbf{h}})\big) + \mathbf{b}_\textrm{c}$$

$\mathbf{h}_t$: final vector of token $t$, padding left out.
$\mathbf{W}_\textrm{d}$: $768 \times 768$. $\mathbf{W}_\textrm{c}$,
$\mathbf{b}_\textrm{c}$: one row per class.

- Library weights loaded by renaming, strictly.
- Eight padded MNLI pairs: largest logit difference 0.0 in fp32 and in bf16,
  the same operations in the same order.
- Neither pooling is what the pretrained dense and norm were fitted to: they
  saw one masked token's vector at a time.

::: {.d2l-note .rule}
Read `classifier_pooling` from the configuration: this checkpoint pools by
the mean.
:::
:::

::: {.slide title="Fine-Tuning Trains Every Weight, in Small Steps, for Two Epochs"}
- AdamW, weight decay 0.01 on matrices.
- Warmup over 10% of the steps, linear decay, clipping at norm 1: the new
  layer starts random.
- Batch 32, two epochs; learning rate from $\{2, 5, 8\} \times 10^{-5}$ on
  held-out training pairs. Chosen: $8 \times 10^{-5}$.
- bf16 products, fp32 weights; each batch padded to its own longest pair.
- A seed fixes the new layer's initialization and the batch order (no
  dropout in ModernBERT).

::: {.d2l-note}
16 bytes per parameter for weights, gradients and two moments: 2.4 GB for
149 million, plus activations.
:::
:::

::: {.slide title="With Few Labels, the Seed Moves Accuracy as Much as the Next Step Up in Data"}
[Seeds and sample size]{.kicker}

| MNLI pairs | 250 | 1,000 | 4,000 | 16,000 |
|---|---|---|---|---|
| mean of 3 seeds | 0.376 | 0.474 | 0.752 | 0.839 |
| seed range | 0.046 | 0.158 | 0.102 | 0.003 |

| SST-2 sentences | 100 | 500 | 2,000 | 8,000 |
|---|---|---|---|---|
| mean of 3 seeds | 0.583 | 0.782 | 0.895 | 0.932 |
| seed range | 0.142 | 0.065 | 0.030 | 0.010 |

- Median gain per size step: MNLI 0.098, SST-2 0.114. Seed range: 0.158 at
  1,000 MNLI pairs, 0.142 at 100 SST-2 sentences.
- At 250 pairs every seed is near the majority rate, 0.363: a small range, a
  poor model.
- Two epochs at every size: 250 pairs get 16 optimizer steps.

::: {.d2l-note}
Agreement among seeds shows that a result is reproducible, not that it is
good.
:::
:::

::: {.slide title="Without the Premise, the Encoder Still Scores 0.53 on MNLI"}
Trained on 5,000 hypotheses alone; intervals resample premises.

@!bert-hypothesis-only

- Annotators' phrasing predicts the label: negations for contradictions,
  added details for neutral pairs.
- Without the premise: about 0.17 above the majority rate of 0.363.

::: {.d2l-note .rule}
Report an inference score beside a hypothesis-only baseline.
:::
:::

::: {.slide title="A Small Head on Token Vectors Tags Names at a Span F1 of 0.95"}
[Token tagging]{.kicker}

| token | `[CLS]` | `Peter` | `ĠBlack` | `burn` | `[SEP]` |
|---|---|---|---|---|---|
| label | not scored | `B-PER` | `I-PER` | not scored | not scored |

- A word's tag goes to its first token. A name counts only if its type and
  both ends match.
- Seed 0: precision 0.939, recall 0.951, span F1 0.945, equal to `seqeval`.
  Three seeds: 0.945 to 0.951.
- By type: LOC 0.97, PER 0.97, ORG 0.92, MISC 0.87.

::: {.d2l-note}
Only a small head, fine-tuned with the encoder: nobody wrote a rule about
capital letters or a list of cities.
:::
:::

::: {.slide title="Check: How Many Tokens Does the 8-Shot Decoder Read per Pair?"}
[Check]{.kicker}

The prompt holds 8 solved MNLI pairs and the query, about 500 tokens. The
answer is read by appending each of the three label words `yes`, `maybe`,
`no` and scoring it. The encoder reads about 40 tokens per pair.

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** Three copies of the prompt: 1,543 tokens measured, about 38
times the encoder's 40. Most of them are the same solved examples, read
again for every item.
:::
:::
:::

::: {.slide title="Operations Predict 500 Times the Cost; the Clock Shows 100"}
[Cost per example]{.kicker}

$C = 2\,(N - N_\textrm{emb})\,L$: a multiply and an add per weight and
token, for $N$ parameters with $N_\textrm{emb}$ in the embedding table and
$L$ tokens.

| per MNLI pair | encoder | 8-shot decoder |
|---|---|---|
| tokens | 40.3 | 1,543 |
| GFLOPs | 8.9 | 4,350 |
| seconds per 1,000, measured | about 0.43 | about 47 |

- Ratio: 486 in operations (38 times the tokens, 13 times the GFLOPs per
  token), about 100 on the clock.
- Short pairs in batches of 32 leave the GPU partly idle; long prompts fill
  it.
:::

::: {.slide title="The Encoder Passes 8 Shots on MNLI Between 1,000 and 4,000 Labels"}
[Encoder or prompted decoder]{.kicker}

::: {.cols .vc}
::: {.col .fig .big}
@!bert-curve
:::

::: {.col}
- Ahead = every seed × every set of examples agrees, and the gap exceeds
  both spreads.
- MNLI, gap to 8 shots: $-0.26$ at 250; $-0.16$ at 1,000 (seed range
  0.158: no verdict); $+0.12$ at 4,000; $+0.21$ at 16,000.
- SST-2: $-0.046$ at 2,000; $-0.009$ at 8,000, within noise.
- 8 shots: MNLI 0.62 to 0.65, SST-2 0.935 to 0.946.
:::
:::

::: {.d2l-note}
On MNLI, thousands of labels, not hundreds, before fine-tuning wins.
:::
:::

::: {.slide title="Check: Prompt or Fine-Tune?"}
[Check]{.kicker}

Case A: 50 labeled support messages, and new topics every month. Case B:
20,000 labeled pairs for a fixed task, and a million new items a day.

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** A: prompt a decoder. With a few hundred labels or fewer the
decoder was ahead, and a new topic is a new prompt. B: fine-tune an
encoder. From 4,000 MNLI pairs it led the 8-shot prompt, at about a
hundredth of the time per item.
:::
:::
:::

::: {.slide title="Wrap-up"}
- A masked-token encoder gives one vector per token from both sides; a
  fine-tuned head turns them into a label per sentence, pair or word.
- 80/10/10 corruption keeps every position's vector about its own token.
- ModernBERT-base: 0.84 on MNLI (16,000 pairs), 0.93 on SST-2 (8,000), span
  F1 about 0.95 on CoNLL-2003.
- Seed range up to 0.158 at 1,000 pairs, against a median gain of 0.098 per
  size step; a hypothesis-only model reaches 0.53 against a majority rate of
  0.36.
- Against 8-shot Qwen3-1.7B, two epochs per size: passes on MNLI between
  1,000 and 4,000 labels, catches up on SST-2 at 8,000, at about a hundredth
  of the time.

**Next:** one vector per text, for search without a head.
:::
