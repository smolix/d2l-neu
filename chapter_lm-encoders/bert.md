# Pretrained Encoders
:label:`sec_lm_bert`

<!-- Author brief, copied verbatim from the Language Models part plan
(docs/llm/language_models_part_plan.md, "2.2 Pretrained Encoders · T1").
Blueprint: docs/llm/blueprints/ch02_encoders.md section 2.2.
The plan is a specification, not book text; this comment is replaced
when the section is written.

**Idea:** A masked-token encoder gives one vector per token, and a task head is fine-tuned per task.
**Opening:** Ten thousand sentences an hour cost more to prompt a decoder than to label. The reader has a bidirectional encoder from `sec_transformer` trained on characters; a released encoder of the same shape, pretrained on trillions of tokens, gives a vector per token that a small head turns into a label. The section fine-tunes it for three tasks, measures when it beats few-shot prompting, and closes on the question of one vector for every task without a head.
**Teaches:**
- The encoder and its masked-token objective restated in one paragraph; ModernBERT worked in full (RoPE, alternating local and global attention, unpadding, 8K context, mean pooling in its head); RoBERTa, ELECTRA and T5 as relatives with their one difference each (no next-sentence prediction and more data; replaced-token detection; span corruption).
- Three heads: sentence classification (sentiment), sentence pairs (natural language inference, with the separator as the only marker since ModernBERT has no segment ids), and token classification (named-entity tagging with word-to-token alignment and span F1), the literal demonstration of one vector per token.
- When an encoder beats prompting: cost per thousand examples, latency, and the labelled data the encoder sees; naive Bayes (`sec_mdl-naive_bayes`) as the classical baseline on word counts; one sentence on 1-D convolutions over tokens as the pre-transformer sentence encoder.
- Fine-tuning hygiene: seeds and data order move small-data accuracy by more than method gaps, so three seeds per cell; a hypothesis-only probe guards against reading an annotation artifact as skill.
**Derivation:** none. Forward FLOPs per example as 2 × non-embedding parameters × tokens, for the cost comparison (`d2l.cost_per_example`).
**Notebook:**
1. Load `answerdotai/ModernBERT-base`, replicate its classification head from scratch (`d2l.PooledClassifier`: pool, linear, GELU, layer norm, dropout, linear) and check logit equality with the library's head.
2. Fine-tune on nested training subsets of a natural-language-inference set (250 to 16,000 examples) and a sentiment set (100 to 8,000), 2 epochs, three seeds for the smaller sizes; evaluate on fixed validation items.
3. Fine-tune a token head on a CoNLL-style tagging set with `d2l.align_token_labels` and report span F1 (`d2l.span_f1`, checked against an oracle library).
4. Score few-shot Qwen3-1.7B on the same validation items with `d2l.score_choices` over verbalized labels at 0, 4 and 8 shots; time encoder and decoder per thousand examples with `d2l.Benchmark` and compare with the FLOP estimate.
5. Claim that survives noise: the encoder matches few-shot prompting at a tenth of the cost per example once it has seen on the order of a thousand labelled examples (the crossing point is measured; a crossing is reported only if the paired interval excludes zero and the seed range is narrower than the gap); the token head reaches a stated span F1 on the tagging set.
6. Reused pieces: `d2l.load_hf`, `d2l.score_choices`, `d2l.paired_bootstrap`, `d2l.Benchmark`; new `d2l.mlm_corrupt`, `d2l.finetune_classifier`.
**Figures:** `fig_lm_encoder_heads` (one encoder, three heads: pooled sentence head, pair input with separator, per-token tagging head); `fig_lm_encoder_vs_prompting` (plot: accuracy against labelled examples for the encoder with the decoder's few-shot lines, and accuracy against cost per thousand examples).
**Restate / forward:** `sec_transformer` (a bidirectional encoder predicts masked tokens from both sides; `subsec_mlm-loss-by-position` showed the loss falls with available context); `sec_mdl-naive_bayes` (a bag-of-words Bayes classifier is the floor); `sec_attention-at-scale` (sliding windows are local attention). Forward: 2.3 (contrastive training of the same encoder).
**Exercises:** 5 to 7. Load-bearing: measure peak memory against batch size for the fine-tuning run and locate the activation term (the exercise `cloud-instances.md:212` re-points to; measure); derive the masked-position loss and its expected gradient on the unmasked positions under the 80/10/10 corruption (derive); check the hypothesis-only baseline on the inference set and say what a high score without the premise shows (check); open: replace the tagging head's first-sub-token rule by a CRF layer and measure the span F1 change.
**Writer notes:**
- The token-classification head (named-entity tagging) is the third head; the BERT-recipe bullets that `sec_transformer` owns are not repeated here; the text-CNN sentence and the naive-Bayes link are included. Keep the sentence and pair heads as in the blueprint.
- Extractive question answering (a start and an end logit per context token, the fourth task of the retiring `finetuning-bert.md`) is an exercise built on the tagging head, not a fourth notebook head: 2.5 answers questions by retrieval and generation. Next-sentence prediction appears only in the RoBERTa clause. Mine `docs/exercise-catalog/19-nlp-applications.md` for exercises (its hypothesis-only exercise is already load-bearing here).
- Exports: `d2l.PooledClassifier(encoder, num_labels, pooling='mean', dropout=0.0)`, `d2l.align_token_labels(word_ids, word_labels, scheme='first')`, `d2l.span_f1(pred_tags, gold_tags, id2label)`, `d2l.finetune_classifier(model, train, eval, lr, epochs, batch_size, seed, warmup=0.1, device=None)`, `d2l.cost_per_example(n_params, n_tokens, n_embed=0)`, `d2l.mlm_corrupt`.
- Reference and tolerance: the replicated head equals `modeling_modernbert.py`'s head to bf16 rounding; `span_f1` equals `seqeval` on the validation split; read `config.classifier_pooling` (mean for ModernBERT) rather than assuming CLS.
- Silent breakages: dropping the separator or swapping premise and hypothesis; truncation that cuts the hypothesis (truncate the first segment); padding to the dataset's longest sequence instead of the batch's; FlashAttention-2 absent on the build box (the SDPA path pads; results unchanged, speed differs, say which kernel ran).
- Data: `nyu-mll/glue` configs `mnli` and `sst2` (mixed-licence derivatives, downloaded at run time, never redistributed); a CoNLL-2003 copy on the Hub (`tner/conll2003`; research terms, run-time download) with a hand-made 500-sentence fallback shipped with the book if the Hub copy changes. Learning rate chosen on a split carved from train, never on validation.
- Memory: fp32 weights, gradients and two Adam moments for 150M parameters are about 2.4 GB; activations at batch 32 × 128 tokens keep the run near 6 GiB; Qwen3-1.7B scoring is about 4 GiB; both fit a 15 GiB slot sequentially.
- Pitfalls to state: a single seed on 250 examples can differ from another seed by more than the method gap; comparing an encoder that saw n labels with a decoder that saw k shots as if the budgets were equal; the decoder's cost includes re-encoding the shots on every call; NLI accuracy without a hypothesis-only baseline is not reading comprehension.
- Do not re-teach: the encoder block and masked objective (`sec_transformer`), BPE and special tokens (`sec_text-sequence`), RoPE, sliding windows.
- Committed artifacts: `d2l-modernbert-mnli` (head and encoder at the largest size, bf16) and the results JSON so the opening cell is instant.
**Sources:**
- Papers: [BERT](https://arxiv.org/abs/1810.04805) (input layout, masked tokens, the fine-tuning recipe); [RoBERTa](https://arxiv.org/abs/1907.11692) (dynamic masking, no next-sentence prediction); [ELECTRA](https://arxiv.org/abs/2003.10555) (replaced-token detection); [T5](https://arxiv.org/abs/1910.10683) (span corruption); [ModernBERT](https://arxiv.org/abs/2412.13663) (the architecture list and training recipe); [Introduction to the CoNLL-2003 Shared Task](https://aclanthology.org/W03-0419/) (the tagging task and its span-level evaluation).
- Code and documentation: [`answerdotai/ModernBERT-base`](https://huggingface.co/answerdotai/ModernBERT-base) (model card and configuration) and its repository [AnswerDotAI/ModernBERT](https://github.com/AnswerDotAI/ModernBERT) (fine-tuning recipes); Transformers' `modeling_modernbert.py` (the head to replicate).
- Courses, tutorials and blogs: [ModernBERT blog](https://huggingface.co/blog/modernbert); Jay Alammar, [The Illustrated BERT](https://jalammar.github.io/illustrated-bert/); [CS224N](https://web.stanford.edu/class/cs224n/) (pretraining lecture); [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/chapter1/1) (the fine-tuning chapter).
-->

<!-- Placeholder (stage 2, cells step): every code cell of the section is
written; the prose around the cells comes with the section. Cell ids follow
the section specification. -->

<!-- Placeholder (prose, opening): labelling ten thousand sentences an hour by prompting a decoder costs more than labelling them with a small model; the reader's bidirectional encoder from sec_transformer, now released and pretrained on far more text; the table below is read from the registered results. -->

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

<!-- Placeholder (prose, opening): the two models, the flags of the CPU path and of regeneration (D2L_RERUN names the results to regenerate; a normal build reads the registered ones and checks them live), and the grid of training sizes, seeds, shots and learning rates. -->

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
        if SMALL else dict(n=1000, items=2000, dec_items=200,
                           tag_train=None, tag_items=None))  # None: all

def result_path(name):
    """The file this run regenerates for name, else the registered one."""
    key, file = ARTIFACTS[name]
    return d2l.lm_artifact(key, file if name in RERUN else None)

print(pd.DataFrame({'model': [ENC, DEC], 'revision': [
    d2l.HF_ENCODER_REVISIONS[ENC][:12], d2l.HF_REVISIONS[DEC][:12]]},
                   index=['encoder', 'decoder']))
print(f'GPU: {GPU}; regenerating: {", ".join(sorted(RERUN)) or "none"}')
```

<!-- Placeholder (stage 2): the encoder is pinned like the decoders of the previous chapter. The encoders, embedders and rerankers of this chapter keep their pins in a registry of their own, so that adding one does not change `d2l.HF_REVISIONS`. -->

```{.python .input #bert-revisions}
%%tab pytorch
#@save
HF_ENCODER_REVISIONS = {
    'answerdotai/ModernBERT-base': '8949b909ec900327062f0ebf497f51aef5e6f0c8'}
```

<!-- Placeholder (stage 2): GLUE (MNLI, SST-2) and CoNLL-2003, pinned to commits and downloaded at run time. The four results of the regeneration runs (the sweep, the MNLI checkpoint, the tagging seeds, the decoder's predictions) are added to this cell when they are registered. -->

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
```

<!-- Placeholder (stage 2): a pinned encoder with one of three heads; load_hf of the previous chapter loads causal models only. The load line names the weights the checkpoint has but the model does not use, and the new weights a head starts from at random. -->

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
    print(f'{model_id}@{revision[:12]}: {type(model).__name__}, '
          f'{dtype}, {device}, attention '
          f'{model.config._attn_implementation}')
    for what, keys in (('not used', info['unexpected_keys']),
                       ('new, at random', info['missing_keys'])):
        if keys:
            print(f'    {what}: {", ".join(sorted(keys))}')
    return model.eval(), tokenizer
```

<!-- Placeholder (prose, opening): read the registered results: the encoder fine-tuned on 16,000 labelled pairs against Qwen3-1.7B with 8 demonstrations in its prompt, on the same 2,000 MNLI validation items. -->

```{.python .input #bert-opening}
%%tab pytorch
if RERUN & {'bert-sweep', 'bert-decoder'}:
    print('regenerating: the table reads the registered results')
else:
    with open(result_path('bert-sweep')) as f:
        sweep = json.load(f)
    with open(result_path('bert-decoder')) as f:
        decoder = json.load(f)
    y = np.array(sweep['eval']['mnli']['label'])
    mnli_runs = [r for r in sweep['runs'] if r['task'] == 'mnli']
    n = max(r['n'] for r in mnli_runs)
    enc = np.mean([np.mean(np.array(r['pred']) == y) for r in mnli_runs
                   if r['n'] == n])  # the mean over seeds
    dec = np.mean(np.array(decoder['pred']['mnli']['8']) == y)
    print(pd.DataFrame({
        'labelled examples': [n, 8], 'accuracy': [enc, dec],
        'ms per 1,000 examples': [sweep['timing']['mnli'],
                                  decoder['timing']['mnli']['8']]},
        index=[f'{ENC.split("/")[1]}, fine-tuned', f'{decoder["config"][
            "model"].split("/")[1]}, 8 shots in the prompt']).round(
        {'accuracy': 3, 'ms per 1,000 examples': 0}).to_string())
```

<!-- Placeholder (prose, ## A Released Masked-Token Encoder): a one-paragraph recap of the bidirectional encoder of sec_transformer and of subsec_mlm-loss-by-position; then the released encoder: a premise-hypothesis pair through its tokenizer, [CLS] a [SEP] b [SEP] with no token_type_ids, one vector per token, and no segment table among the embedding modules; its configuration below. -->

```{.python .input #bert-raw}
%%tab pytorch
encoder, tok = d2l.load_encoder(ENC, dtype=torch.float32)  # fp32: trained
dev = next(encoder.parameters()).device
pair = tok('A man plays a guitar on stage.', 'Someone is making music.',
           return_tensors='pt').to(dev)
print(tok.convert_ids_to_tokens(pair['input_ids'][0].tolist()))
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

<!-- Placeholder (prose, ## A Released Masked-Token Encoder): what the pretrained masked-token head predicts at one masked position per sentence, the five most probable fills. -->

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
    print(f'{text:58s}', ', '.join(
        f'{tok.decode(j, clean_up_tokenization_spaces=False).strip()} {q:.2f}'
        for q, j in zip(p.tolist(), i.tolist())))
del mlm
d2l.lm_free()
```

<!-- Placeholder (prose, ## A Released Masked-Token Encoder): BERT's corruption: 15% of the tokens are selected for prediction; of those, 80% become [MASK], 10% a random token and 10% stay, so the model cannot rely on seeing [MASK] at the positions it predicts and keeps a useful vector at every position. -->

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

<!-- Placeholder (prose, ## A Released Masked-Token Encoder): a million tokens through the corruption: the shares it selects and how it changes them. -->

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

<!-- Placeholder (prose, ## A Released Masked-Token Encoder): RoBERTa, ELECTRA and T5, one difference each; then ModernBERT from its configuration: RoPE with two thetas, GeGLU, no biases, 8,192 positions, mean pooling in its head, unpadding (the FlashAttention path; this build runs SDPA, as the load line says), and alternating attention: every third layer global, the others a window of 128 tokens. The cell counts the query-key pairs the two kinds of layer score, from boolean masks, against the formula (n_g L + n_l w) / ((n_g + n_l) L). -->

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

<!-- Placeholder (prose, ## Three Heads): the sentence head written out: pool the final hidden states (mean over the attention mask, as config.classifier_pooling says, or the first token), then dense, GELU, layer norm, dropout and a linear map to the logits; its submodule names follow ModernBertForSequenceClassification, so loading that head's weights is a rename. Figure fig_lm_encoder_heads. -->

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

    def forward(self, input_ids, attention_mask):
        H = self.encoder(input_ids=input_ids,
                         attention_mask=attention_mask).last_hidden_state
        if self.pooling == 'cls':
            h = H[:, 0]
        else:  # padding positions do not count
            m = attention_mask.unsqueeze(-1).to(H.dtype)
            h = (H * m).sum(1) / m.sum(1)
        h = self.head.norm(F.gelu(self.head.dense(h)))
        return self.classifier(self.drop(h))
```

<!-- Placeholder (prose, ## Three Heads): the check: Transformers' sequence classifier with a new head, its weights renamed into PooledClassifier (model. becomes encoder.; a strict load proves the names map one to one), logits on eight padded MNLI pairs in fp32 and in bf16. -->

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
X = tok(list(mnli_val.premise[:8]), list(mnli_val.hypothesis[:8]),
        padding=True, truncation='only_first', max_length=MAXLEN['mnli'],
        return_tensors='pt').to(dev)
diff = {}
with torch.inference_mode():
    for dtype in (torch.float32, torch.bfloat16):
        ref, clf = ref.to(dtype), clf.to(dtype)
        diff[str(dtype)] = (ref(**X).logits - clf(**X)).abs().max().item()
print('max |logit difference|:', diff)
assert diff['torch.float32'] <= 1e-4 and diff['torch.bfloat16'] <= 2e-2
del ref, clf
d2l.lm_free()
```

<!-- Placeholder (prose, ## Fine-Tuning and Its Variance): the recipe: AdamW with weight decay 0.01 on the matrices, a linear warmup over the first 10% of the steps and a linear decay, gradient clipping at norm 1, bf16 autocast on a GPU with fp32 master weights; batches padded to their own longest sequence, not the dataset's. The same function trains the token head: a label per token, -100 where nothing is scored. -->

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
    longest sequence; seed fixes the data order and dropout. Returns the
    training losses, the eval accuracy and predictions (a label per
    example, or per scored token), the training seconds and the peak GPU
    memory in GiB."""
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
    torch.manual_seed(seed)  # dropout
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

<!-- Placeholder (prose, ## Encoder or Prompted Decoder): forward FLOPs per example: two (a multiply and an add) per weight per token; an embedding lookup is not a matrix product, and attention's own L^2 term is left out at these lengths. -->

```{.python .input #bert-cost}
%%tab pytorch
#@save
def cost_per_example(n_params, n_tokens, n_embed=0):
    """Forward FLOPs per example, 2 (n_params - n_embed) n_tokens: a
    multiply and an add per weight and token, embedding lookups excluded."""
    return 2 * (n_params - n_embed) * n_tokens
```

<!-- Placeholder (prose, ## Fine-Tuning and Its Variance): the data: nested training subsets (each a prefix of one seed-0 permutation of the training set), 2,000 items after the largest subset carved off for choosing the learning rate (never the validation items), and fixed validation items: the first 2,000 of a seed-0 shuffle of MNLI validation_matched, all 872 of SST-2. A pair is encoded jointly, premise first, and cut from the premise's end (truncation='only_first'), so the hypothesis is never cut. -->

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

train = {'mnli': read('glue-mnli-train'), 'sst2': read('glue-sst2-train')}
val = {'mnli': mnli_val.iloc[np.random.default_rng(0).permutation(
    len(mnli_val))], 'sst2': read('glue-sst2-validation')}
DATA = {}
for task, sizes in SIZES.items():
    perm = np.random.default_rng(0).permutation(len(train[task]))
    rows = train[task].iloc[perm[:max(sizes) + N_CARVE]]
    ev = val[task].iloc[:N_EVAL[task]]
    DATA[task] = {'perm': perm, 'train_df': rows.iloc[:max(sizes)],
                  'eval_df': ev, 'train': encode(task, rows.iloc[:max(sizes)]),
                  'carve': encode(task, rows.iloc[max(sizes):]),
                  'eval': encode(task, ev), 'eval_idx': ev.idx.tolist(),
                  'shots': train[task].iloc[perm[-max(SHOTS):]]}
DATA['mnli']['hyp_train'] = encode('mnli', train['mnli'].iloc[
    DATA['mnli']['perm'][:N_HYP]], hypothesis_only=True)
DATA['mnli']['hyp_eval'] = encode('mnli', DATA['mnli']['eval_df'],
                                  hypothesis_only=True)
print(pd.DataFrame.from_dict({task: {
    'largest subset': len(D['train']), 'carved (learning rate)':
    len(D['carve']), 'validation items': len(D['eval']),
    'cut at MAXLEN': sum(len(x) == MAXLEN[task] for x, _ in D['train']),
    'majority rate (validation)': D['eval_df'].label.value_counts(
        normalize=True).max().round(3)} for task, D in DATA.items()},
                             orient='index').to_string())
```

<!-- Placeholder (prose, ## Fine-Tuning and Its Variance): helpers for the runs: a fresh copy of the pretrained encoder under a new head whose initialization the seed fixes, prediction in batches padded to their own longest pair, a timer per 1,000 examples, and the fine-tuned classifier as a zip of bf16 weights. -->

```{.python .input #bert-run-defs}
%%tab pytorch
def pad(seqs):
    """Token-id lists padded to the longest, and the mask, on dev."""
    T = max(map(len, seqs))
    X = torch.tensor([s + [tok.pad_token_id] * (T - len(s)) for s in seqs])
    M = torch.tensor([[1] * len(s) + [0] * (T - len(s)) for s in seqs])
    return X.to(dev), M.to(dev)

def finetune(task, train, ev, lr, seed):
    torch.manual_seed(seed)  # the new head's initialization
    model = d2l.PooledClassifier(copy.deepcopy(encoder), len(LABELS[task]))
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

<!-- Placeholder (prose, ## Fine-Tuning and Its Variance): the sweep, regenerated under D2L_RERUN=bert-sweep (about 20 minutes on one card): the learning rate per task chosen on the carved items at the largest size, then every size with three seeds, a hypothesis-only model, timings and the MNLI checkpoint. A normal build reads the registered results and checks them twice: it retrains one cell of the grid and compares the predictions, and it loads the registered checkpoint and predicts the validation items. -->

```{.python .input #bert-sweep}
%%tab pytorch
if 'bert-sweep' in RERUN:
    t0 = time.perf_counter()
    out = {'config': dict(model=ENC, revision=encoder.revision, sizes=SIZES,
                          seeds=SEEDS, lrs=LRS, maxlen=MAXLEN, epochs=EPOCHS,
                          batch=BATCH, n_carve=N_CARVE, n_hyp=N_HYP),
           'eval': {t: {'idx': D['eval_idx'], 'label': [y for _, y in D[
               'eval']]} for t, D in DATA.items()},
           'lr': {}, 'runs': [], 'hyp_only': [], 'timing': {}}
    for task, D in DATA.items():  # learning rate on the carved items
        acc = {lr: finetune(task, D['train'], D['carve'], lr, 0)[1][
            'eval_acc'] for lr in LRS}
        lr = max(acc, key=acc.get)
        out['lr'][task] = {'chosen': lr, 'carved_acc': list(acc.values())}
        for n in SIZES[task]:
            for seed in SEEDS:
                model, r = finetune(task, D['train'][:n], D['eval'], lr, seed)
                out['runs'].append(dict(
                    task=task, n=n, seed=seed, lr=lr, acc=r['eval_acc'],
                    pred=r['eval_pred'], seconds=r['seconds'],
                    peak_gib=r['peak_gib']))
                if (n, seed) == (max(SIZES[task]), 0):  # timed in bf16
                    model = model.to(torch.bfloat16)
                    X = pad([x for x, _ in D['eval'][:BATCH]])
                    out['timing'][task] = ms_per_1k(lambda: model(*X), BATCH)
                    if task == 'mnli':
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
with open(result_path('bert-sweep')) as f:
    SWEEP = json.load(f)
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

<!-- Placeholder (prose, ## Fine-Tuning and Its Variance): live check (i): one cell of the grid retrained, seed 0, with the registered learning rate; GPU kernels are not deterministic, so the predictions agree on most items, not all. -->

```{.python .input #bert-sweep-live}
%%tab pytorch
n, k = LIVE['n'], LIVE['items']
y = np.array(SWEEP['eval']['mnli']['label'][:k])
model, r = finetune('mnli', DATA['mnli']['train'][:n], DATA['mnli']['eval'][
    :k], SWEEP['lr']['mnli']['chosen'], 0)
reg = np.array(runs.query('task == "mnli" and n == @n and seed == 0')
               .pred.iloc[0][:k])
live = np.array(r['eval_pred'])
print(f'n = {n}, seed 0, {k} items: accuracy {np.mean(live == y):.3f} live, '
      f'{np.mean(reg == y):.3f} registered; same prediction on '
      f'{np.mean(live == reg):.3f}; {r["seconds"]:.0f} s')
del model
d2l.lm_free()
```

<!-- Placeholder (prose, ## Fine-Tuning and Its Variance): live check (ii): the registered checkpoint (bf16 weights) predicts the validation items; it stays loaded for the timing below. -->

```{.python .input #bert-ckpt-live}
%%tab pytorch
clf16 = load_ckpt(result_path('bert-ckpt'),
                  torch.bfloat16 if GPU else torch.float32)
y = np.array(SWEEP['eval']['mnli']['label'][:k])
reg = np.array(runs.query('task == "mnli" and seed == 0').sort_values('n')
               .pred.iloc[-1][:k])
pred = np.array(predict(clf16, DATA['mnli']['eval'][:k]))
agree = np.mean(pred == reg)
print(f'registered checkpoint, {k} items: accuracy {np.mean(pred == y):.3f}; '
      f'same prediction as its training run on {agree:.3f}; {d2l.lm_peak()}')
assert SMALL or agree >= 0.99
```

<!-- Placeholder (prose, ## Fine-Tuning and Its Variance): the hypothesis-only probe (Gururangan et al.): the same encoder trained on the hypothesis alone, never seeing the premise; accuracy with a bootstrap interval per seed against the majority-class rate. -->

```{.python .input #bert-hypothesis-only}
%%tab pytorch
y = np.array(SWEEP['eval']['mnli']['label'])
majority = np.bincount(y).max() / len(y)
rows = {}
for r in SWEEP['hyp_only']:
    b = d2l.bootstrap_ci(np.array(r['pred']) == y)
    rows[f'seed {r["seed"]}'] = {'accuracy': b.mean, 'CI low': b.ci[0],
                                 'CI high': b.ci[1]}
print(f'majority-class rate: {majority:.3f}')
print(pd.DataFrame(rows).T.round(3).to_string())
```

<!-- Placeholder (prose, ## Three Heads): token tagging: a word may be several tokens; the first sub-token carries the word's label, the others are not scored (-100), and so are the special tokens. -->

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

<!-- Placeholder (prose, ## Three Heads): span F1: an entity counts only if its type and both ends match; a run of I- tags after O or after another type starts an entity, as conlleval and seqeval count. -->

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

<!-- Placeholder (prose, ## Three Heads): CoNLL-2003 comes as words with IOB2 tags; the words are joined by spaces and each token is mapped to its word through character offsets (the tokenizer's own word ids split at punctuation, so "U.S." would be four words). -->

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

<!-- Placeholder (prose, ## Three Heads): the token head (ModernBertForTokenClassification: the same dense, GELU and norm at every position, then a linear map to nine tags), fine-tuned for two epochs on the training sentences; predictions at first sub-tokens are the words' tags. The regeneration (D2L_RERUN=bert-tagging, about 4 minutes) trains three seeds; the normal build trains seed 0 live. -->

```{.python .input #bert-tagging}
%%tab pytorch
def train_tagger(seed, train, valid):
    torch.manual_seed(seed)  # the new token head
    tagger, _ = d2l.load_encoder(ENC, head='token_cls', num_labels=len(
        id2label), dtype=torch.float32)
    r = d2l.finetune_classifier(tagger, [(x, l) for x, _, l in train],
                                [(x, l) for x, _, l in valid], 5e-5, EPOCHS,
                                BATCH, seed)
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
        {'config': dict(model=ENC, revision=encoder.revision, lr=5e-5,
                        epochs=EPOCHS, batch=BATCH), 'runs': tagging}))
with open(result_path('bert-tagging')) as f:
    TAGGING = json.load(f)
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
seeds_f1 = [x['f1'] for x in TAGGING['runs']]
print(f'registered, {len(seeds_f1)} seeds: F1 {min(seeds_f1):.3f} to '
      f'{max(seeds_f1):.3f}; {d2l.lm_peak()}')
assert abs(f1['f1'] - oracle) < 1e-9
```

<!-- Placeholder (prose, ## Encoder or Prompted Decoder): the decoder labels the same validation items from k demonstrations drawn once from the training set (nested: the 4 shots are the first 4 of the 8), each answer scored as the log-probability of its label word (d2l.score_choices). The regeneration (D2L_RERUN=bert-decoder, about 10 minutes) scores every item at 0, 4 and 8 shots for both tasks; the normal build re-scores the first items at 8 shots. -->

```{.python .input #bert-decoder}
%%tab pytorch
VERBAL = {'mnli': ('yes', 'maybe', 'no'), 'sst2': ('negative', 'positive')}
FMT = {'mnli': '{x}\nAnswer: {y}', 'sst2': 'Review: {x}\nSentiment: {y}'}

def query(task, row):
    if task == 'mnli':
        return (f'{row.premise.strip()}\nQuestion: {row.hypothesis.strip()} '
                'Yes, no, or maybe?')
    return row.sentence.strip()

def classify(task, k, rows):
    demos = [(query(task, r), VERBAL[task][r.label])
             for r in DATA[task]['shots'].iloc[:k].itertuples()]
    prompts = [d2l.fewshot_prompt(demos, query(task, r), fmt=FMT[task])
               for r in rows.itertuples()]
    return d2l.score_choices(dec, dtok, prompts, [' ' + v for v in VERBAL[
        task]], norm='token', batch_size=16).argmax(-1).tolist()

dec, dtok = d2l.load_hf(DEC, dtype=torch.bfloat16 if GPU else torch.float32)
if 'bert-decoder' in RERUN:
    out = {'config': dict(model=DEC, revision=dec.revision, shots=SHOTS,
                          fmt=FMT, verbal=VERBAL, norm='token'),
           'eval': {t: D['eval_idx'] for t, D in DATA.items()},
           'pred': {}, 'timing': {}}
    for task, D in DATA.items():
        out['pred'][task] = {k: classify(task, k, D['eval_df']) for k in SHOTS}
        out['timing'][task] = {k: ms_per_1k(lambda: classify(
            task, k, D['eval_df'].iloc[:BATCH]), BATCH, 1, 3) for k in SHOTS}
    d2l.lm_save_rerun(ARTIFACTS['bert-decoder'][1], write_json(out))
with open(result_path('bert-decoder')) as f:
    DECODER = json.load(f)
assert DECODER['config']['model'] == DEC and all(
    DECODER['eval'][t] == D['eval_idx'] for t, D in DATA.items())
rows = {}
for task in DATA:
    for k in SHOTS:
        b = d2l.bootstrap_ci(np.array(DECODER['pred'][task][str(k)]) == np.array(
            SWEEP['eval'][task]['label']))
        rows[task, k] = {'accuracy': b.mean, 'CI low': b.ci[0], 'CI high': b.ci[1]}
print(pd.DataFrame(rows).T.round(3).to_string())
m = LIVE['dec_items']
live = classify('mnli', 8, DATA['mnli']['eval_df'].iloc[:m])
agree = np.mean(np.array(live) == np.array(DECODER['pred']['mnli']['8'][:m]))
print(f'live, 8 shots, {m} MNLI items: same prediction on {agree:.3f}; '
      f'{d2l.lm_peak()}')
assert SMALL or agree >= 0.99
```

<!-- Placeholder (prose, ## Encoder or Prompted Decoder): the classical floor: naive Bayes on word counts (sec_mdl-naive_bayes), trained on the same nested subsets; a pair is two bags, premise words and hypothesis words kept apart by a prefix. Before transformers, a one-dimensional convolution over token embeddings was the usual learned sentence encoder; it sits between this floor and the fine-tuned encoder. -->

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

<!-- Placeholder (prose, ## Encoder or Prompted Decoder): measured cost per example on this card (bf16, SDPA): a batch of 32 MNLI pairs through the fine-tuned encoder, and 32 items through the 8-shot decoder, which re-reads its demonstrations for every item and every label word; against the FLOP estimate 2 N L with N the non-embedding parameters and L the tokens computed per example (padding included for the encoder's batch). -->

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
         for r in DATA['mnli']['shots'].iloc[:8].itertuples()]
tokens = {'encoder': X[0].shape[1], 'decoder': np.mean([sum(len(dtok(
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

<!-- Placeholder (prose, ## Encoder or Prompted Decoder): the learning curve (left: accuracy against labelled examples, the encoder's band spanning its three seeds, the decoder's k-shot lines with their bootstrap intervals, naive Bayes, the majority rate dashed) and the cost plot (right: accuracy against seconds per 1,000 examples on a log scale, the tables' milliseconds per 1,000 divided by 1,000); MNLI above, SST-2 below. -->

```{.python .input #bert-curve}
%%tab pytorch
def correct(pred, task):
    return np.array(pred) == np.array(SWEEP['eval'][task]['label'])

_, axes = d2l.plt.subplots(2, 2, figsize=(8, 6.4))
for ax, task in zip(axes, ('mnli', 'sst2')):
    acc = runs.query('task == @task').groupby('n').acc
    sizes, big = acc.mean().index.to_numpy(), acc.get_group(acc.mean().index[-1])
    dec = {k: d2l.bootstrap_ci(correct(DECODER['pred'][task][str(k)], task))
           for k in SHOTS}
    flat = lambda v: [v] * len(sizes)
    series = {'encoder (seed range)': (acc.mean(), acc.min(), acc.max())}
    series |= {f'decoder, {k} shots': (flat(b.mean), flat(b.ci[0]),
                                        flat(b.ci[1])) for k, b in dec.items()}
    series['naive Bayes'] = ([NB[task, n] for n in sizes],) * 3
    majority = np.bincount(SWEEP['eval'][task]['label']).max() / len(
        SWEEP['eval'][task]['label'])
    d2l.plot_intervals(sizes, series, 'labelled examples', f'{task} accuracy',
                       kind='band', xscale='log', reference=majority,
                       axes=ax[0])
    sec = [SWEEP['timing'][task] / 1e3] + [DECODER['timing'][task][str(
        k)] / 1e3 for k in SHOTS]  # x: one cost per model
    none = [np.nan] * len(SHOTS)
    d2l.plot_intervals([float(f'{v:.2g}') for v in sec], {
        f'encoder, n = {sizes[-1]:,}': ([big.mean(), *none],
                                       [big.min(), *none], [big.max(), *none]),
        'decoder, 0, 4, 8 shots': ([np.nan] + [b.mean for b in dec.values()],
                                   [np.nan] + [b.ci[0] for b in dec.values()],
                                   [np.nan] + [b.ci[1] for b in dec.values()])},
        'seconds per 1,000 examples', f'{task} accuracy', xscale='log',
        reference=majority, axes=ax[1])
for top, bottom in zip(*axes):  # the top row's legends serve both rows
    top.legend(fontsize='small')
    bottom.get_legend().remove()
```

<!-- Placeholder (prose, ## Encoder or Prompted Decoder): where the encoder passes the 8-shot decoder: per training size and seed, a paired bootstrap of per-item correctness on the same validation items; a size counts as passed when every seed's interval excludes zero on the encoder's side and the seed range is narrower than the mean gap. -->

```{.python .input #bert-crossing}
%%tab pytorch
def crossing(task):
    dec8, rows = correct(DECODER['pred'][task]['8'], task), {}
    for n, group in runs.query('task == @task').groupby('n'):
        diffs = [d2l.paired_bootstrap(correct(p, task), dec8)
                 for p in group.pred]
        accs, gap = group.acc.to_numpy(), np.mean([d.mean for d in diffs])
        signs = {int(np.sign(d.ci[0])) if d.ci[0] * d.ci[1] > 0 else 0
                 for d in diffs}
        clear = len(signs) == 1 and np.ptp(accs) < abs(gap)
        rows[n] = {'seeds': len(accs), 'encoder': accs.mean(),
                   'seed range': np.ptp(accs), 'gap to 8 shots': gap,
                   'CIs from': min(d.ci[0] for d in diffs),
                   'CIs to': max(d.ci[1] for d in diffs), 'verdict': (
                       {1: 'encoder ahead', -1: 'decoder ahead'}[signs.pop()]
                       if clear and 0 not in signs else 'within noise')}
    table = pd.DataFrame(rows).T.infer_objects()
    return table, next((n for n, v in table.verdict.items()
                        if v == 'encoder ahead'), None)

for task in ('mnli', 'sst2'):
    table, n = crossing(task)
    print(f'{task}: the encoder passes 8 shots at n = {n or "none measured"}')
    print(table.to_string(float_format=lambda v: f'{v:.3f}'))
```

<!-- Placeholder (prose, ## Encoder or Prompted Decoder): when each wins: a short Markdown table tab_lm_bert_when (labels available, throughput, a label set that changes); then the pitfalls; Summary; Exercises. One model per task: the next section asks for one vector per text that serves every task without a head. -->
