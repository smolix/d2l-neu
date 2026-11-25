# Long-Context Training
:label:`sec_lm_long_context`

<!-- Author brief, copied verbatim from the Language Models part plan
(docs/llm/language_models_part_plan.md, "4.5 Long-Context Training · T2").
Blueprint: docs/llm/blueprints/ch04_pretraining.md section 4.6.
The plan is a specification, not book text; this comment is replaced
when the section is written.

**Idea:** A longer window is trained in stages on data that needs it, and measured on tasks that need it, because perplexity at a length is not use of that length.
**Opening:** After the RoPE extension of `sec_positional-information` the reference model accepts 16K tokens, and a cell shows its loss by position bucket flat beyond 4K while a recall probe that asks for a value stored 12K tokens earlier sits at chance. The reader knows which frequencies interpolation compresses and why YaRN leaves the fast pairs alone; what remains is the training and the measurement. The closing question is the road from 16K to a million tokens.
**Teaches:**
- Progressive length schedules (a stage at 8K before 16K; the attention share of FLOPs rises from 0.45 at 4K to 0.77 at 16K, so the schedule saves compute as well as helping optimization) and the learning-rate recipe for extension (a low constant rate with a short decay).
- Data with real long-range dependencies: books and long papers mixed half and half with short replay under document masks, against concatenated unrelated documents that teach no long-range dependence and cost 2.3 times the FLOPs per token.
- Effective context measured three ways at 4K, 8K, 12K and 16K: perplexity by position bucket, a synthetic recall task over a length-by-depth grid (RULER's design), and the long-context gain on repeated rare words (the loss drop from the distant occurrence); effective context as the largest length whose lower confidence bound clears a threshold; natural tasks (HELMET) stated as the next tier, not run.
- The gap to the million-token flagships and what closes it, in one paragraph: sparse and linear mixing (14.5, 14.6), context parallelism (6.5), and KV tiers (5.4, 13.6); no new experiment.
**Derivation:** none (position interpolation, NTK-aware scaling and YaRN are restated, not re-derived). The one small computation kept: the attention share $a(T)$ of $F(T) = 6N_{\textrm{mm}} + 12LdT$ for the reference model gives $F = 1.51$, $2.19$ and $3.55$ GFLOP per token at 4K, 8K and 16K, so 40M of 100M tokens spent at 8K save 15% of the stage's compute; checked against `FlopCounterMode` within 3%.
**Notebook:**
1. Question: extended from 4K to 16K with interpolation and with YaRN on long-dependency data, which scheme keeps more of the 4K accuracy, and does perplexity or recall say so first?
2. Model and data: the reference model's final checkpoint (`lm-r2-s6000`) with `d2l.rope_inv_freq(scheme=...)` and `rope_mscale=d2l.yarn_mscale(4)` on PG-19 books (`deepmind/pg19`, Apache-2.0, public-domain texts) plus long peS2o papers, half long and half FineWeb-Edu replay with document masks; three trained arms of 100M tokens at $T = 16384$: interpolation, YaRN, and YaRN progressive (40M tokens at 8K with scale 2, then 60M at 16K with scale 4); peak learning rate 1e-4, warmup 20, constant, final 15% linear to zero; batch 262,144 tokens, micro-batch 2; about 2.1 to 2.7 h on one A100 40 GB plus 30 minutes of evaluation (fallback 50M tokens per arm). Zero-shot cells for each scheme precede training. T1: the `t1` checkpoint from 1K to 4K with 16M tokens per arm (`lm-pg19-30m`, `lm-fwedu-replay-12m`), three seeds for interpolation and YaRN.
3. Check: `d2l.rope_inv_freq` at scale 1 returns the base frequencies to `1e-7`; interpolation by position scaling equals dividing the frequencies by $s$; YaRN's ramp gives the expected pair counts (9 interpolated, 12 blended, 11 untouched for the reference model's 32 pairs at 4K); the frequencies match Hugging Face's `rope_utils` YaRN initialization with `truncate=False` to `1e-6`; an oracle copier scores 100% on the recall task at every length; the bucketed losses average to the overall loss to `1e-6`.
4. Claim that survives noise: YaRN keeps more of the 4K perplexity and recall at 16K than interpolation, stated with the T1 three-seed interval and the T2 single run's item bootstrap labelled "single run"; perplexity recovers before recall and before the long-context gain, so perplexity overstates effective context; the progressive arm's standing against one-shot is reported without a presumed direction.
5. Reused: `d2l.pretrain`, `d2l.PackedStream`, `d2l.load_reference_model`, `d2l.score_choices` (1.4) for the recall candidates, `d2l.paired_bootstrap`; a RULER single-needle cross-check on `Qwen/Qwen3-1.7B` at 4K to 16K as an optional cell (its KV cache at 32K is 3.8 GB).
**Figures:** `fig_lm_length_schedule` (the two-stage schedule and the data mixture per stage); `fig_lm_effective_context` (plot: recall over the length-by-depth grid for the two schemes, with the effective-context threshold); `fig_lm_lc_ppl_position` (plot: loss by position bucket, zero-shot and trained).
**Restate / forward:** `sec_positional-information` (`subsec_pos-context-extension`: only the slow RoPE pairs meet new angles beyond the training length; interpolation compresses all frequencies, YaRN compresses only the slow ones and adds an attention temperature; the NTK-aware base change is the exercise there). Forward: 14.5 (learned sparse attention as what makes a million tokens affordable).
**Exercises:** 5 to 7. Derive: the attention-entropy change under the YaRN temperature; the number of pairs that complete a rotation within 4K (23 of 32) and the overhang of NTK-aware scaling for the slowest pair. Measure: recall against needle depth at each length; the long-data share at $\{0, 25, 50, 75, 100\}$%. Open: 32K with scale 8 from the 16K checkpoint, and whether the recall threshold moves.
**Writer notes:**
- "Scored on RULER-style and document-QA tasks" is implemented as the synthetic recall grid plus the long-context gain; a document-QA family would need an evaluation set the part does not own (HELMET is stated, not run). Do not reuse another model's effective-context threshold; thresholds (recall at least 0.5, gain positive) are protocol parameters.
- Pitfalls: serving with a different scaling than training (store the scheme and scale in the checkpoint config; `lm-r2-16k-pi` and `lm-r2-16k-yarn` carry it, Chapter 12 loads one); applying the YaRN temperature twice (`rope_mscale` and the softmax scale); YaRN's $\alpha = 1$, $\beta = 32$ were tuned on Llama's head dimension 128 and the reference model has 32 pairs, so report `rope_mscale=1` as well if the temperature hurts; a recall task with several correct answers.
- Risk: a 138M model may show no usable recall at 16K. Fallback ladder: recall at 4K, 6K and 8K; more tokens at 8K; report perplexity and gain only and say recall was at chance. Verify the task with the oracle copier and a model trained from scratch at 4K first.
- Hugging Face's `rope_utils` rounds the ramp edges to integer pair indices by default (`truncate=True`); the check prints the difference and passes with `truncate=False`.
- Do not re-teach: RoPE and the three extension schemes (`sec_positional-information`, positional-information.md:730-830 and the exercise at :990), online softmax (`sec_attention-at-scale`) and sliding windows (`sec_windowed-linear-attention`). New here: schedules, data, the three measurements, the trained comparison.
- Blueprint: ch04 §4.6 (four schemes including NTK-aware, D1's pair arithmetic with checks (a) to (g), D2, the evaluation families E1 to E3, `tab_lm_lc_recipes`). The NTK-aware trained arm is an exercise; the zero-shot cells stay, and the section adds the million-token paragraph.
**Sources:**
- Papers: [How to Train Long-Context Language Models (Effectively)](https://arxiv.org/abs/2410.02660) (data and schedule choices measured on downstream tasks); [YaRN](https://arxiv.org/abs/2309.00071) (the scheme and its temperature; restated); [Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595) (position interpolation; restated); [RULER](https://arxiv.org/abs/2404.06654) (synthetic tasks over length and depth); [HELMET](https://arxiv.org/abs/2410.02694) (application-oriented long-context evaluation); [Lost in the Middle](https://arxiv.org/abs/2307.03172) (position sensitivity of retrieval); [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) (staged context extension in a production recipe).
- Code and documentation: [HELMET code](https://github.com/princeton-nlp/HELMET) (task construction for the next evaluation tier); [FlexAttention API](https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html) (document masks at 16K).
- Courses, tutorials and blogs: [SmolLM3 blog](https://huggingface.co/blog/smollm3) (a documented long-context extension stage with its data mixture).
-->

<!-- Placeholder (prose): opening. The reader has the reference model, trained at 4K, and the RoPE extension schemes of subsec_pos-context-extension. Scaled by interpolation, the model accepts 16K tokens; the cell shows the control (recall of a value stored 2K back inside the trained window), then the loss by position bucket at 16K and recall of a value stored 12K back. Opening question: extended from 4K to 16K with interpolation and with YaRN on long-dependency data, which scheme keeps more of the 4K accuracy, and does perplexity or recall say so first? -->

```{.python .input #long-context-imports}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #long-context-imports-extra}
%%tab pytorch
import pandas as pd
```

```{.python .input #long-context-opening}
%%tab pytorch
model = d2l.lc_reference_model()  # trained at L = model.max_len tokens
tok, L, s, depth = d2l.lc_tokenizer(), model.max_len, 4, 0.25
control = d2l.make_recall_task(tok, L, depth=0.5, n_items=50)
k = d2l.score_recall(model, tok, control).sum()
print(f'control, trained window ({L / 1024:g}K, {L / 2048:g}K back): '
      f'{k} / 50')
d_head = model.blks[0].attention.head_dim
model.set_rope(d2l.rope_inv_freq(d_head, scheme='pi', s=s, L=L),
               scheme='pi', scale=s)  # model.max_len is now s * L
books = d2l.test_books(32, s * L)
pos = d2l.nll_by_position(model, books, range(0, s * L + 1, L))
print('loss by position (nats): ' + ', '.join(
    f'{b} {v:.2f}' for b, v in zip(pos['bucket'], pos['nll'])))
far = d2l.make_recall_task(tok, s * L, depth=depth, n_items=50)
k = d2l.score_recall(model, tok, far).sum()
print(f'recall {(1 - depth) * s * L / 1024:g}K back: {k} / 50 (chance 1/8)')
rising = (pos['lo'].iloc[1:] > pos['hi'].iloc[0]).any()
above = d2l.wilson_ci(k, 50)[0] > 1 / 8
print(f'reading: loss {"rising" if rising else "flat"} beyond {L / 1024:g}K,'
      f' recall {"above" if above else "at"} chance')
```

<!-- Placeholder (prose): reading of the opening: the model accepts the long window and its loss beyond 4K stays flat (or rises; the cell decides from the bucket intervals), yet the value stored 12K back is recalled at chance while the control inside the trained window is recalled. Perplexity at a length is not use of that length. Roadmap of the section: the schemes without training, what training needs (schedule, data), three measurements, the trained arms. -->

## Extension without Training

<!-- Placeholder (prose): restate in one sentence subsec_pos-context-extension (only the slow pairs meet new angles beyond the trained length; PI compresses all frequencies, NTK-aware scaling raises the base, YaRN compresses only the slow pairs and adds a temperature on queries and keys). The function below computes the four sets of frequencies; YaRN's ramp runs over the pair index between the pair that turns beta = 32 times within L and the pair that turns alpha = 1 time. -->

```{.python .input #long-context-rope-inv-freq}
%%tab pytorch
#@save
def rope_inv_freq(d_head, base=10000.0, scheme='none', s=1.0, L=4096,
                  alpha=1.0, beta=32.0):
    """RoPE frequencies of the d_head / 2 pairs for a window s times the
    trained length L: 'none' (the base frequencies), 'pi' (all divided by
    s), 'ntk' (the base raised by s^(d / (d - 2))), or 'yarn' (pairs that
    turn more than beta times within L kept, less than alpha times divided
    by s, a linear ramp over the pair index between)."""
    if scheme not in ('none', 'pi', 'ntk', 'yarn'):
        raise ValueError(f'unknown scheme {scheme!r}')
    if scheme == 'ntk':
        base = base * s ** (d_head / (d_head - 2))
    theta = 1.0 / base ** (torch.arange(0, d_head, 2, dtype=torch.float32,
                                        device='cpu') / d_head)
    if scheme == 'pi':
        return theta / s
    if scheme == 'yarn':  # the pair index m at which a pair turns r times
        m = lambda r: d_head * math.log(L / (2 * math.pi * r)) / (
            2 * math.log(base))
        ramp = ((torch.arange(d_head // 2) - m(beta))
                / (m(alpha) - m(beta))).clamp(0, 1)
        return theta * (ramp / s + 1 - ramp)
    return theta

#@save
def yarn_mscale(s):
    """YaRN's scale of queries and keys, 0.1 ln s + 1 (1 for s <= 1): the
    attention logits grow by its square, the softmax scale stays."""
    return 0.1 * math.log(s) + 1.0 if s > 1 else 1.0
```

<!-- Placeholder (prose): the checks below: scale 1 gives the base frequencies for every scheme; interpolation is division by s; the YaRN pair counts for the reference model's 32 pairs at L = 4,096 (derived in the text); the temperature; and agreement with Transformers' YaRN initialization (whose default truncate=True rounds the ramp edges to whole pairs). The plot shows each scheme's frequency ratio per pair. -->

```{.python .input #long-context-scheme-checks}
%%tab pytorch
d_head, L_r2 = 64, 4096  # the reference model's head width and length
theta = d2l.rope_inv_freq(d_head)
same = max((d2l.rope_inv_freq(d_head, scheme=k, s=1) - theta).abs().max()
           for k in ('pi', 'ntk', 'yarn'))
print(f's = 1: max |Δ| {same:.1e}')
pi = d2l.rope_inv_freq(d_head, scheme='pi', s=4)
print(f'PI equals θ/s: {torch.equal(pi, theta / 4)}')
yarn = d2l.rope_inv_freq(d_head, scheme='yarn', s=4, L=L_r2)
ratio = yarn / theta
print(f'YaRN pairs: {(ratio == 1 / 4).sum()} interpolated, '
      f'{((ratio > 1 / 4) & (ratio < 1)).sum()} blended, '
      f'{(ratio == 1).sum()} untouched')
print(f'yarn_mscale(4) = {d2l.yarn_mscale(4):.4f}')
hf, _ = d2l.transformers_yarn(d_head, s=4, L=L_r2, truncate=False)
rounded, _ = d2l.transformers_yarn(d_head, s=4, L=L_r2, truncate=True)
print(f'vs Transformers (truncate=False): {(yarn - hf).abs().max():.1e}; '
      f'truncate=True differs by {(yarn - rounded).abs().max():.1e}')
ntk = d2l.rope_inv_freq(d_head, scheme='ntk', s=4)
d2l.plot(torch.arange(d_head // 2), [pi / theta, ntk / theta, ratio],
         'pair index m', 'θ′ / θ', legend=['PI', 'NTK', 'YaRN'],
         ylim=(0, 1.05), figsize=(4, 2.5))
```

<!-- Placeholder (prose): zero-shot, the schemes only change which angles the model sees; whether a model uses 16K tokens is decided by training and measured below. -->

## What the Training Needs

### The Cost of Length

<!-- Placeholder (prose): the attention share a(T) of F(T) = 6 N_mm + 12 L H d_h T for the reference model (the dense count of the attention scores; kernels that skip the masked causal half do about half of that work, printed beside it); the ×2.35 of a 16K window holds for windows without document masks (one long document, or unrelated documents attended across); a progressive schedule (a stage at 8K before 16K) saves compute as well as easing optimization; the learning-rate recipe for extension (a low constant rate, fresh optimizer moments, a short final decay). -->

```{.python .input #long-context-attention-share}
%%tab pytorch
arch, lengths = d2l.r2_config('r2').arch, (4096, 8192, 16_384)
F_mm = d2l.flops_per_token(arch)  # the weight matrices, no attention scores
cost = {T: d2l.flops_per_token(arch, T) for T in lengths}  # dense scores
half = {T: d2l.flops_per_token(arch, T, causal_half=True) for T in lengths}
flops = ' / '.join(f'{cost[T] / 1e9:.2f}' for T in lengths)
share = ' / '.join(f'{1 - F_mm / cost[T]:.2f}' for T in lengths)
share_half = ' / '.join(f'{1 - F_mm / half[T]:.2f}' for T in lengths)
print(f'F, dense attention: {flops} GFLOP per token at 4K / 8K / 16K')
print(f'attention share: {share} (causal half: {share_half})')
stage, total = 40e6, 100e6  # tokens at 8K, all tokens of the arm
saving = stage * (cost[8192] - cost[16_384]) / (total * cost[16_384])
print(f'{stage / 1e6:.0f}M of {total / 1e6:.0f}M tokens at 8K: '
      f"{saving:+.0%} of the arm's compute")
ratio = d2l.flop_counter_per_token(arch, 16_384) / cost[16_384]
print(f'FlopCounterMode at 16K (dense scores as well): ratio {ratio:.3f}')
print('16K windows, no document mask: '
      f'×{cost[16_384] / cost[4096]:.2f} the FLOPs of 4K windows')
```

### Data with Long-Range Dependencies

<!-- Placeholder (prose): books and long papers have dependencies across thousands of tokens; concatenated unrelated documents attended across (no document mask) cost 2.35 times the FLOPs per token of 4K windows and teach no long-range dependence (ProLong's finding, cited, not measured here). The arms mix half long data and half short replay under document masks, so the short-context skills keep being trained; fig_lm_length_schedule shows the two stages, the learning rate and the mixture. -->

<!-- Figure (generator): fig_lm_length_schedule, img/mdl-lm-pretraining-length-schedule.svg (tools/gen_mdl_lm_pretraining_figures.py): the context-length staircase (8K at scale 2, then 16K at scale 4) against tokens, the learning rate beneath, and each stage's long/short mixture as a stacked bar. -->

## Three Ways to Measure Context

### Recall over Length and Depth

<!-- Placeholder (prose): a synthetic recall task in RULER's design: a haystack of key-value lines, one queried key whose line sits at a chosen depth, eight candidate values from the haystack. A model that uses its window recalls at every depth; one that only accepts it recalls near the end and at chance elsewhere. The task must have exactly one correct answer (keys and values are distinct). -->

```{.python .input #long-context-recall-task}
%%tab pytorch
#@save
def make_recall_task(tokenizer, length, depth, n_pairs=None, n_candidates=8,
                     n_items=200, seed=0):
    """Key-value recall items of at most length tokens for score_choices:
    lines 'The value of <key> is <value>.' with distinct two-word keys and
    five-digit values, the queried line at fraction depth of the prompt,
    then the query 'The value of <key> is'; the choices are n_candidates
    distinct values of the haystack, the gold value among them once."""
    g = d2l.recall_generator(seed, length, depth)
    out = dict(prompts=[], choices=[], gold=[], depth_actual=[])
    for _ in range(n_items):
        keys, values, lines, start = d2l.recall_haystack(
            tokenizer, length, g, n_pairs, min_lines=n_candidates)
        q = round(depth * (len(lines) - 1))  # the queried line
        others = values[:q] + values[q + 1:]
        picks = torch.randperm(len(others), generator=g)[:n_candidates - 1]
        choices = [f' {values[q]}'] + [f' {others[i]}' for i in picks]
        order = torch.randperm(n_candidates, generator=g).tolist()
        out['prompts'].append(d2l.RECALL_HEAD + ''.join(lines)
                              + f'The value of {keys[q]} is')
        out['choices'].append([choices[j] for j in order])
        out['gold'].append(order.index(0))
        out['depth_actual'].append(start[q] / start[-1])
    return out
```

<!-- Placeholder (prose): before measuring a model, check the task: one item's size, and an oracle that copies the value stored under the queried key must score 100% at every length. -->

```{.python .input #long-context-task-checks}
%%tab pytorch
N = 20  # items per length
task = d2l.make_recall_task(tok, 16_384, depth=0.5, n_items=N)
prompt, choices = task['prompts'][0], task['choices'][0]
print(f'item: {len(tok(prompt)["input_ids"]):,} tokens, '
      f'{prompt.count(" is "):,} pairs, query depth '
      f'{task["depth_actual"][0]:.2f}, {len(choices)} candidates')

def copier(prompt, choices):  # the value stored under the queried key
    key = prompt.rsplit('The value of ', 1)[1].removesuffix(' is')
    value = prompt.split(f'The value of {key} is ', 1)[1].split('.', 1)[0]
    return choices.index(' ' + value)

hits = {}
for length in (4096, 8192, 16_384):
    t = d2l.make_recall_task(tok, length, depth=0.5, n_items=N)
    hits[length] = sum(copier(p, c) == g for p, c, g in
                       zip(t['prompts'], t['choices'], t['gold'])) / N
print('oracle copier: ' + ', '.join(f'{h:.0%}' for h in hits.values())
      + ' at 4K, 8K, 16K')
```

### Loss by Position and the Long-Context Gain

<!-- Placeholder (prose): two measurements on natural text. The loss by position bucket averages the next-token loss over positions 0-4K, 4-8K, ...; intervals resample whole windows (books), since positions within a book are not independent. The long-context gain isolates use of distant context: on rare tokens that occurred earlier in the window at a gap beyond the short context, the loss with the whole window minus the loss with only the last short_len / 2 to short_len tokens (a sliding window). -->

```{.python .input #long-context-measures}
%%tab pytorch
#@save
def nll_by_position(model, windows, edges, ctx=None):
    """Mean next-token loss (nats) of windows (n, T + 1) in each position
    bucket [edges[k], edges[k + 1]), with a 95% interval from resampling
    whole windows: DataFrame(bucket, nll, lo, hi)."""
    nll = d2l.window_nlls(model, windows, ctx).double()  # (n, T)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        part = nll[:, lo:hi]
        rows.append((d2l.bucket_name(lo, hi), *d2l.window_ci(
            part.sum(1), torch.full((len(part),), float(hi - lo)))))
    return pd.DataFrame(rows, columns=['bucket', 'nll', 'lo', 'hi'])

#@save
def long_context_gain(model, windows, short_len, long_len, min_gap,
                      rare_ids, edges):
    """Loss drop (nats) from seeing the whole window of long_len tokens
    instead of the last short_len / 2 to short_len tokens, on rare tokens
    whose previous occurrence lies at least min_gap back, per gap bucket
    (edges[k], edges[k + 1]]: DataFrame(bucket, gain, lo, hi, n)."""
    windows = torch.as_tensor(windows)[:, :long_len + 1]
    gain = (d2l.sliding_nlls(model, windows, short_len)
            - d2l.window_nlls(model, windows))
    gap = d2l.repeat_gaps(windows, rare_ids)  # 0: no earlier occurrence
    pairs = list(zip(edges[:-1], edges[1:]))
    sums, counts = d2l.gain_sums(gain, gap, pairs, min_gap)
    rows = [(d2l.bucket_name(lo, hi), *d2l.window_ci(sums[:, k],
                                                     counts[:, k]),
             int(counts[:, k].sum())) for k, (lo, hi) in enumerate(pairs)]
    return pd.DataFrame(rows, columns=['bucket', 'gain', 'lo', 'hi', 'n'])
```

<!-- Placeholder (prose): two checks: the bucket means (equal widths) average to the loss over the whole window; on a synthetic repeat (distinct random tokens, then the same tokens again), an induction head as a model gains about ln V nats from the long window, and a model that ignores its context gains nothing. -->

```{.python .input #long-context-measure-checks}
%%tab pytorch
edges = range(0, s * L + 1, L)
parts = d2l.nll_by_position(model, books[:4], edges)
whole = d2l.nll_by_position(model, books[:4], [0, s * L])
print('bucket means average to the total: |Δ| '
      f'{abs(parts["nll"].mean() - whole["nll"][0]):.1e}')
V, T, short = 2048, 2048, 256  # vocabulary, window, short context
torch.manual_seed(0)
x = torch.stack([torch.randperm(V)[:T // 2] for _ in range(8)])
repeat = torch.cat([x, x, x[:, :1]], 1)  # every token again T / 2 later
gain = {name: d2l.long_context_gain(
            m, repeat, short_len=short, long_len=T, min_gap=short,
            rare_ids=torch.arange(V), edges=[short, T])['gain'][0]
        for name, m in (('oracle', d2l.CopyOracle(V)),
                        ('unigram', d2l.UnigramModel(V)))}
print(f'long-context gain: oracle {gain["oracle"]:+.2f} nats, '
      f'unigram {gain["unigram"]:.3f}')
```

### The Test Data and the Zero-Shot Baseline

<!-- Placeholder (prose): the long half (PG-19 books in file order, then peS2o papers of at least 8,192 tokens; licences), the short replay (a FineWeb-Edu shard the reference run never read) and the 64 held-out PG-19 test books cut to 16,385 tokens (the table counts their EOS as well, 16,386). -->

```{.python .input #long-context-data}
%%tab pytorch
long = d2l.long_data('t2')
print(long.table.to_string(index=False, formatters={
    'documents': '{:,}'.format, 'tokens': '{:,}'.format,
    'median length': '{:,.0f}'.format}))
```

<!-- Placeholder (prose): every scheme at scale 4 without training: loss per position bucket on 64 test books and recall at 4K, 8K and 16K (five depths, 100 items each). The none row's recall at 4K is the control: the task is solved inside the trained window before recall at 16K measures anything. The mscale = 1 row: YaRN's alpha and beta were tuned for head dimension 128, the reference model has 32 pairs. -->

```{.python .input #long-context-zeroshot}
%%tab pytorch
SCHEMES = {'none': dict(scheme='none', s=4), 'PI': dict(scheme='pi', s=4),
           'NTK': dict(scheme='ntk', s=4), 'YaRN': dict(scheme='yarn', s=4),
           'YaRN, mscale 1': dict(scheme='yarn', s=4, mscale=1.0)}
zs = d2l.lc_zeroshot('r2', schemes=SCHEMES, lengths=(4096, 8192, 16_384))
d2l.show_lc_zeroshot(zs)
```

## The Trained Arms

### The Length Schedule

<!-- Placeholder (prose): three arms of 100M tokens each from the reference model's final checkpoint: interpolation and YaRN one-shot at 16K, and YaRN progressive (40M tokens at 8K with scale 2, then 60M at 16K with scale 4). Peak rate 1e-4, warmup 20 steps, constant, linear decay over the final 15%; fresh AdamW moments; half long data, half replay. -->

```{.python .input #long-context-schedule}
%%tab pytorch
ARMS = {  # stages: (window T, RoPE scale s, share of the tokens)
    'PI': dict(scheme='pi', stages=((16_384, 4, 1.0),)),
    'YaRN': dict(scheme='yarn', stages=((16_384, 4, 1.0),)),
    'YaRN-prog': dict(scheme='yarn', stages=((8_192, 2, 0.4),
                                             (16_384, 4, 0.6)))}
TOKENS = 100_000_000
print(f'{"arm":<10} {"stage 1":<20} {"stage 2":<20} tokens')
for arm, a in ARMS.items():
    stages = [f'{T // 1024}K × {share * TOKENS / 1e6:.0f}M, s = {sc}'
              for T, sc, share in a['stages']] + ['-']
    print(f'{arm:<10} {stages[0]:<20} {stages[1]:<20} '
          f'{TOKENS / 1e6:.0f}M')
```

### Three Arms at 16K

<!-- Placeholder (prose): the result table: val loss at 4K (and its change from the base), loss in the last position bucket, recall at 16K with its Wilson interval (one run per arm: a single run), and the effective context. The plot shows the training loss; the dotted line marks the progressive arm's switch to 16K. -->

```{.python .input #long-context-arms}
%%tab pytorch
lc = d2l.lc_arms(ARMS, base='r2', tokens=TOKENS, seq_len=16_384,
                 batch_tokens=262_144, peak_lr=1e-4, warmup=20,
                 decay_frac=0.15, long_share=0.5)
d2l.show_lc_table(lc)
d2l.plot_lc_train(lc.logs)
```

<!-- Placeholder (prose): loss by position bucket, the zero-shot PI and YaRN models against the three trained arms. -->

```{.python .input #long-context-position}
%%tab pytorch
d2l.show_lc_position(lc)
```

<!-- Placeholder (prose): recall over the length-by-depth grid (100 items per cell); the effective context is the largest length whose recall, pooled over depths, has a lower 95% bound of at least 0.5 and whose long-context gain at gaps (L/2, L] is positive. Both thresholds are protocol parameters, not another model's. -->

```{.python .input #long-context-recall-grid}
%%tab pytorch
THRESHOLD = 0.5  # lower 95% bound of recall (the effective-context rule)
effective = d2l.effective_context(lc.recall, lc.gain, threshold=THRESHOLD)
d2l.show_lc_recall(lc, effective, threshold=THRESHOLD)
```

<!-- Placeholder (prose): the long-context gain per gap bucket: how much the distant occurrence of a rare token lowers its loss. -->

```{.python .input #long-context-gain}
%%tab pytorch
d2l.show_lc_gain(lc)
```

### Which Measure Moves First

<!-- Placeholder (prose): from the arms' periodic evaluations (every 64 steps, 16 books, 250 recall items at 16K): the first step at which the loss in the last bucket comes within MARGIN of the base's loss on its trained window, recall at 16K clears the rule, and the gain at the longest gaps is positive. State the order per arm as measured; 'perplexity first' only if it holds in every arm. -->

```{.python .input #long-context-first-steps}
%%tab pytorch
MARGIN = 0.05  # nats above the base's loss on its trained window
d2l.show_lc_first_steps(lc, margin=MARGIN, threshold=THRESHOLD)
```

### The Cost at Short Context

<!-- Placeholder (prose): what the extension costs at short context: the val loss at 4K and the suite mean (five tasks, 2,000 items each) minus the base's, with intervals paired by window and by item. -->

```{.python .input #long-context-short-cost}
%%tab pytorch
d2l.show_lc_short(lc)
```

### Seeds at Small Scale

<!-- Placeholder (prose): one run per arm cannot separate a scheme difference from run-to-run noise; the T1 configuration repeats the comparison on the t1 checkpoint (trained at 1K) extended to 4K with three seeds of the data order (16M tokens per arm). The base row's recall at 1K is the control. -->

```{.python .input #long-context-t1}
%%tab pytorch
ARMS_T1 = {  # the t1 checkpoint (trained at 1K) to 4K; seeds: data order
    'PI': dict(scheme='pi', stages=((4096, 4, 1.0),), seeds=(0, 1, 2)),
    'YaRN': dict(scheme='yarn', stages=((4096, 4, 1.0),), seeds=(0, 1, 2)),
    'YaRN-prog': dict(scheme='yarn', stages=((2048, 2, 0.4),
                                             (4096, 4, 0.6)), seeds=(0,))}
t1 = d2l.lc_arms(ARMS_T1, base='t1', tokens=16_000_000, seq_len=4096,
                 batch_tokens=65_536)
d2l.show_lc_table(t1)
```

<!-- Placeholder (prose): the scheme difference YaRN minus PI with intervals: at T1 a t interval over the three seeds, at T2 a paired resampling of items and books within the single run (labelled so). A direction is claimed only where the T1 interval excludes zero; the T2 single run is never the only evidence. -->

```{.python .input #long-context-scheme-difference}
%%tab pytorch
for tier, runs in (('T1', t1), ('T2', lc)):  # T2 never uses the stand-in
    d2l.show_scheme_difference(runs, tier, 'YaRN', 'PI')
```

## Toward a Million Tokens

<!-- Placeholder (prose): one paragraph, forward references only: sparse and linear mixing (chap_lm_architectures), context parallelism (chap_lm_parallelism), KV tiers (chap_lm_hardware, chap_lm_serving); HELMET as the next evaluation tier; no experiment. -->

## Summary

<!-- Placeholder (prose): summary bullets. -->

## Exercises

<!-- Placeholder (prose): 5 to 7 exercises: the attention-entropy change under the YaRN temperature; the pairs that complete a rotation within 4K (23 of 32) and the NTK overhang of the slowest pair; recall against needle depth at each length (the committed grid, re-plotted); the long-data share at {0, 25, 50, 75, 100}% (T1, about 25 minutes); the NTK-aware arm trained (T1); RULER single-needle on Qwen/Qwen3-1.7B at 4K to 16K; open: 32K with scale 8 from the 16K checkpoint. -->
