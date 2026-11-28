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

<!-- Placeholder (prose): opening. The reader has the reference model, trained at 4K, and the RoPE extension schemes of subsec_pos-context-extension. Scaled by interpolation, the model accepts 16K tokens; the cell shows, unscaled at 4K, needle recall 128-256 tokens back against 2-4K back (claim 1b, the opening's contrast: the base retrieves near and not across its own trained window: lower Wilson bound 128-256 back at least 0.5, upper bound 2-4K back below 0.5; otherwise the opening uses the 16K contrast alone), then with PI at 16K the loss by position bucket, recall 128-256 back with the query at 16K (whether PI zero-shot still copies at short range at all) and recall 8-16K back. Opening question: extended from 4K to 16K with interpolation and with YaRN on long-dependency data, which scheme keeps more of the 4K accuracy, and does perplexity or recall say so first? The opening states claims 1 and 1b from this cell as measured; their readings are in the placeholder after it. -->

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
model = d2l.load_reference_model(device=d2l.try_gpu())  # trained at 4K
tok, L, s = d2l.lc_tokenizer(), model.max_len, 4
pct = lambda b: f'{b.rate:.0%} [{b.lo:.0%}, {b.hi:.0%}]'  # Wilson 95%
bins = d2l.recall_by_gap(d2l.needle_recall(
    model, d2l.needle_windows(tok, L, 100)), L).set_index('gap_lo')
print(f'trained window ({L // 1024}K): recall {pct(bins.loc[128])} '
      f'128-256 back, {pct(bins.loc[L // 2])} 2-4K back (chance 12.5%)')
d_head = model.blks[0].attention.head_dim
model.set_rope(d2l.rope_inv_freq(d_head, scheme='pi', s=s, L=L),
               scheme='pi', scale=s)  # model.max_len is now s * L
books = d2l.test_books(32, s * L)
pos = d2l.nll_by_position(model, books, range(0, s * L + 1, L))
print('PI at 16K, no training: loss by position ' + ', '.join(
    f'{b} {v:.2f}' for b, v in zip(pos['bucket'], pos['nll'])))
bins = d2l.recall_by_gap(d2l.needle_recall(
    model, d2l.needle_windows(tok, s * L, 100)), s * L).set_index('gap_lo')
near, far = bins.loc[128], bins.loc[s * L // 2]
print(f'recall at 128-256 back, query at 16K: {pct(near)}')
print(f'recall at 8K-16K back: {pct(far)}')
rising = (pos['lo'].iloc[1:] > pos['hi'].iloc[0]).any()
side = lambda b: 'above' if b.lo > 1 / 8 else 'at'
print(f'reading: loss {"rising" if rising else "flat"} beyond 4K; recall '
      f'{side(near)} chance 128-256 back, {side(far)} chance 8-16K back')
```

<!-- Placeholder (prose): reading of the opening: the window a model accepts, the window it was trained on and the distance it retrieves across are three numbers. Inside its trained window the base copies a value 128-256 tokens back and fails 2-4K back (as measured). With PI at 16K the loss beyond 4K stays flat (or rises; the cell decides from the bucket intervals); recall 128-256 back with the query at 16K says whether interpolation without training still copies at short range, and only if it does is chance-level recall 8-16K back a statement about distance (otherwise PI broke copying at every gap; state which, as measured). Perplexity at a length is not use of that length. Roadmap of the section: the schemes without training, what training needs (schedule, data), three measurements, the trained arms. Claims 1 and 1b, live half (cell 11 adds the paired interval of loss 12-16K minus 0-4K and 250-window recalls): claim 1 holds if the buckets are flat and recall 8-16K back is at chance, else "rises by x nats" or "above chance (k of n)"; claim 1b holds if the lower Wilson bound 128-256 back is at least 0.5 (reach, ruling (a)) and the upper bound 2-4K back is below 0.5 (resolved below reach), else "the base model retrieves across its window" if the 2-4K lower bound is at least 0.5, or "not resolved 2-4K back" if its interval contains 0.5, and the opening uses the 16K contrast alone. Word the 2-4K bin from its interval: "far below reach" (or "not detectable" if its lower bound is at most 0.25), and "at chance" only if the interval contains 12.5% (the smoke's lower bound sits just above it). -->

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

<!-- Placeholder (prose): the checks below: scale 1 gives the base frequencies for every scheme; interpolation is division by s; the YaRN pair counts for the reference model's 32 pairs at L = 4,096 (derived in the text); the temperature; and agreement with Transformers' YaRN initialization (whose default truncate=True rounds the ramp edges to whole pairs). The plot shows each scheme's frequency ratio per pair. Claim 2: the cell asserts the derived values (scale 1 and Transformers within 1e-6, interpolation exact, 9 / 12 / 11 pairs); there is no fallback. -->

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
pairs = [int(m.sum()) for m in (ratio == 1 / 4, (ratio > 1 / 4) & (ratio < 1),
                                ratio == 1)]
print('YaRN pairs: {} interpolated, {} blended, {} untouched'.format(*pairs))
print(f'yarn_mscale(4) = {d2l.yarn_mscale(4):.4f}')
hf, _ = d2l.transformers_yarn(d_head, s=4, L=L_r2, truncate=False)
rounded, _ = d2l.transformers_yarn(d_head, s=4, L=L_r2, truncate=True)
print(f'vs Transformers (truncate=False): {(yarn - hf).abs().max():.1e}; '
      f'truncate=True differs by {(yarn - rounded).abs().max():.1e}')
assert same < 1e-6 and (yarn - hf).abs().max() < 1e-6  # as derived
assert torch.equal(pi, theta / 4) and pairs == [9, 12, 11]
ntk = d2l.rope_inv_freq(d_head, scheme='ntk', s=4)
d2l.plot(torch.arange(d_head // 2), [pi / theta, ntk / theta, ratio],
         'pair index m', 'θ′ / θ', legend=['PI', 'NTK', 'YaRN'],
         ylim=(0, 1.05), figsize=(4, 2.5))
```

<!-- Placeholder (prose): zero-shot, the schemes only change which angles the model sees; whether a model uses 16K tokens is decided by training and measured below. -->

## What the Training Needs

### The Cost of Length

<!-- Placeholder (prose): the attention share a(T) of F(T) = 6 N_mm + 12 L H d_h T for the reference model (the dense count of the attention scores; kernels that skip the masked causal half do about half of that work, printed beside it); the ×2.35 of a 16K window holds for windows without document masks (one long document, or unrelated documents attended across); a progressive schedule (a stage at 8K before 16K) saves compute as well as easing optimization; the learning-rate recipe for extension (a low constant rate, fresh optimizer moments, a short final decay). Claim 3: the shares 0.45 / 0.62 / 0.77 and the 15% saving are exact arithmetic; the cell asserts the profiler ratio within 3%. Claim 10, cost half: the ×2.35 line. -->

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
assert abs(ratio - 1) < 0.03  # the count agrees with the profiler
print('16K windows, no document mask: '
      f'×{cost[16_384] / cost[4096]:.2f} the FLOPs of 4K windows')
```

### Data with Long-Range Dependencies

<!-- Placeholder (prose): books and long papers have dependencies across thousands of tokens; concatenated unrelated documents attended across (no document mask) cost 2.35 times the FLOPs per token of 4K windows and teach no long-range dependence (ProLong's finding, cited, not measured here). The arms mix half long data and half short replay under document masks, so the short-context skills keep being trained; fig_lm_length_schedule shows the two stages, the learning rate and the mixture. Claim 10, second half (unrelated documents teach no long-range dependence): cited, not measured. -->

<!-- Figure (generator): fig_lm_length_schedule, img/mdl-lm-pretraining-length-schedule.svg (tools/gen_mdl_lm_pretraining_figures.py): the context-length staircase (8K at scale 2, then 16K at scale 4) against tokens, the learning rate beneath, and each stage's long/short mixture as a stacked bar. -->

## Three Ways to Measure Context

### Recall over Length and Depth

<!-- Placeholder (prose): the one-hop needle probe (rulings (a)-(c) of 2026-11-25; the recall proposal): eight lines `<key> <value>` of common single-token words inserted at line breaks of a held-out book; the query is a line break and one key, the answer the next token, chosen among the window's eight values (chance 1/8). Recall is indexed by the gap, the distance back from the query to the value, in octave bins, not by the window length and depth. Why one hop: the first build's template ("The value of <key> is") asked the model to find a two-word key and then copy a five-digit value several tokens long, a two-hop task that the reference model failed inside its trained window, so the probe failed for a reason other than context. Keys and values never occur in the book text of their window, and the needle positions are the same for every model of one tokenizer. -->

```{.python .input #long-context-recall-task}
%%tab pytorch
#@save
def needle_windows(tokenizer, L, n, seed=0, n_needles=8):
    """n needle windows of L tokens: test-book text with n_needles lines
    '<key> <value>' inserted after line breaks. The prefix holds the first
    L - 2 tokens; the query of needle j is '\\n<key j>' at positions L - 2
    and L - 1, its answer value j. A needle's gap runs from its value to
    the query (L - 1), log-uniform in (32, L - 64]. Returns
    Needles(prefix (n, L - 2), keys (n, k), values (n, k), gaps (n, k), nl)."""
    nl, = d2l._encode(tokenizer, '\n')
    key_pool, value_pool = d2l.needle_pools(tokenizer)
    text = d2l.test_books(n, L - 3 - 3 * n_needles, seed, tokenizer)
    ends, out = d2l._line_ends(tokenizer), ([], [], [], [])
    for i, t in enumerate(text):
        g = torch.Generator().manual_seed(seed * 1_000_003 + L * 1009 + i)
        gaps, at = d2l._needle_slots(ends[t], L, n_needles, g)
        words = d2l._text_words(tokenizer, t)
        keys = d2l._needle_pick(key_pool, words, t, n_needles, g)
        values = d2l._needle_pick(value_pool, words, t, n_needles, g)
        lines = torch.tensor([[k, v, nl] for (_, k), (_, v) in
                              zip(keys, values)])
        parts, prev = [], 0
        for j, a in enumerate(at.tolist()):  # after text token a
            parts += [t[prev:a + 1], lines[j]]
            prev = a + 1
        for x, v in zip(out, (torch.cat(parts + [t[prev:]]), lines[:, 0],
                              lines[:, 1], gaps)):
            x.append(v)
    return d2l.Needles(*map(torch.stack, out), nl)
```

<!-- Placeholder (prose): before measuring a model, check the probe: one window's shape and its gaps; the pilot's three checks of a miss (an oracle that copies the token after the last earlier occurrence of the queried key scores 100% at every length; every pool word is one token in the form the window uses, a key on its own after a line break, a value with its leading space; no key or value occurs in its window's text); and the octave bins of gap partition the queries. -->

```{.python .input #long-context-task-checks}
%%tab pytorch
nd = {n: d2l.needle_windows(tok, n, 20) for n in (4096, 16_384)}
w, (keys, values) = nd[16_384], d2l.needle_pools(tok)
print(f'window: {w.prefix.shape[1] + 2:,} tokens, {w.keys.shape[1]} '
      f'needles, gaps {int(w.gaps.min()):,} to {int(w.gaps.max()):,}; '
      f'{len(keys):,} keys, {len(values):,} values')
oracle = {}
for length, w in nd.items():
    checks = d2l.needle_diagnostics(tok, w)
    assert all(checks.values()), checks  # the pilot's checks of a miss
    rec = d2l.needle_oracle(w)
    assert d2l.recall_by_gap(rec, length)['n'].sum() == len(rec)  # bins
    oracle[length] = rec['correct'].mean()
print('oracle copier: ' + ', '.join(f'{h:.0%}' for h in oracle.values())
      + ' at 4K, 16K; the pilot checks hold at both')
```

<!-- Placeholder (prose): the format table: the first build's template against the one-hop needle on the same book windows of 1,024 tokens (item i of both formats: the same book, offset, relative line positions and candidate order; 200 items, the queried line at mid depth), each candidate scored by its summed log-probability, for the oracle copier, the t1 and r2 final checkpoints and Qwen3-0.6B-Base; the last column is template minus one-hop paired by item with its interval (claim 1c). State as measured whether the reference model fails the template and solves the one-hop needle inside its trained window, and whether Qwen3 solves both (it is the check that the template is solvable at all). Claim 1c: if Qwen3 scores below 90% on either format, "the reference scores x% on the one-hop format", as measured; the oracle copier is 100% on Qwen3's own items, so the shortfall is the model's, not the probe's. -->

```{.python .input #long-context-format-table}
%%tab pytorch
probe = d2l.lc_probe_runs(t1_steps=(509, 1017, 1526),
                          r2_steps=(500, 1000, 2000, 3000, 4000, 6000))
d2l.show_lc_format(probe)
```

<!-- Placeholder (prose): reach during pretraining: recall by gap at the trained length for the committed checkpoints of t1 (at 1K) and r2 (at 4K). The steps between which copying appears (from a checkpoint whose 32-64 interval contains chance, 1/8, to the first with a lower 95% bound above 0.25; 'by step b' alone when the checkpoint before is already above chance), r2's reach at each checkpoint from step 1,000 to step 6,000 and whether it is non-decreasing (within one bin), and its recall 256-1K back at step 6,000 minus at step 1,000, paired by query (claim 11, fixed steps), as measured; the induction-head literature (cited) places the onset early in training. Claim 12 is an observation on two runs, not a law in steps: copying appears between steps 500 and 1,000 in each run, stated per model as measured. Claim 11's fallback: "reach was x at step 1,000 and y at the end". -->

```{.python .input #long-context-reach-train}
%%tab pytorch
d2l.show_reach_train(probe)
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
gap = abs(parts['nll'].mean() - whole['nll'][0])
print(f'bucket means average to the total: |Δ| {gap:.1e}')
assert gap < 1e-6
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
assert gain['oracle'] > 0.99 * math.log(V) and abs(gain['unigram']) < 1e-3
```

### The Test Data and the Zero-Shot Baseline

<!-- Placeholder (prose): the long half (PG-19 books in file order, then peS2o papers of at least 8,192 tokens; licences), the short replay (a FineWeb-Edu shard the reference run never read) and the 100 held-out PG-19 test books, whole (their EOS counted), from which every evaluation cuts windows at seeded random offsets, the same for every model of one tokenizer. -->

```{.python .input #long-context-data}
%%tab pytorch
long = d2l.long_data('t2')
print(long.table.to_string(index=False, formatters={
    'documents': '{:,}'.format, 'tokens': '{:,}'.format,
    'median length': '{:,.0f}'.format}))
```

<!-- Placeholder (prose): every scheme at scale 4 without training: the losses of the first and last position buckets on 64 test windows of 16K and their difference on the same windows (window-bootstrap interval; claim 1 for PI), needle recall at 16K in the gap bin 128-256 and in the top octave 8-16K (250 windows, ± half the 95% Wilson interval) and the reach. The line under the table is the control, the unscaled base at 4K, with its two recalls as Wilson intervals [lo, hi] (claim 1b is read on the bounds against 0.5): its recall 128-256 back shows the probe solved inside the trained window before recall at 16K measures anything; its top octave (2-4K back) shows how far the base's own copying reaches, which need not be the whole trained window (claim 1b). The two lines after it compare YaRN with PI and with NTK on the same windows (loss 0-4K paired by window, near recall over gaps 32-512 paired by query, window-resampled 95% intervals; claim 4): state the order as measured, and "not resolved" where the paired interval contains 0. The mscale = 1 row: YaRN's alpha and beta were tuned for head dimension 128, the reference model has 32 pairs. -->

```{.python .input #long-context-zeroshot}
%%tab pytorch
SCHEMES = {'none': dict(scheme='none', s=4), 'PI': dict(scheme='pi', s=4),
           'NTK': dict(scheme='ntk', s=4), 'YaRN': dict(scheme='yarn', s=4),
           'YaRN, mscale 1': dict(scheme='yarn', s=4, mscale=1.0)}
zs = d2l.lc_zeroshot('r2', schemes=SCHEMES, lengths=(4096, 16_384))
d2l.show_lc_zeroshot(zs)
```

## The Trained Arms

### The Length Schedule

<!-- Placeholder (prose): three arms of 100M tokens each from the reference model's final checkpoint: interpolation and YaRN one-shot at 16K (three seeds of the data order each), and YaRN progressive (40M tokens at 8K with scale 2, then 60M at 16K with scale 4; one seed). Peak rate 1e-4, warmup 20 steps, constant, linear decay over the final 15%; fresh AdamW moments; half long data, half replay. -->

```{.python .input #long-context-schedule}
%%tab pytorch
ARMS = {  # stages: (window T, RoPE scale s, share of the tokens)
    'PI': dict(scheme='pi', stages=((16_384, 4, 1.0),), seeds=(0, 1, 2)),
    'YaRN': dict(scheme='yarn', stages=((16_384, 4, 1.0),), seeds=(0, 1, 2)),
    'YaRN-prog': dict(scheme='yarn', stages=((8_192, 2, 0.4),
                                             (16_384, 4, 0.6)), seeds=(0,))}
TOKENS = 100_000_000
print(f'{"arm":<10} {"stage 1":<20} {"stage 2":<20} {"tokens":<7} seeds')
for arm, a in ARMS.items():
    stages = [f'{T // 1024}K × {share * TOKENS / 1e6:.0f}M, s = {sc}'
              for T, sc, share in a['stages']] + ['-']
    print(f'{arm:<10} {stages[0]:<20} {stages[1]:<20} '
          f'{TOKENS / 1e6:<7.0f} {len(a["seeds"])}')
```

### Three Arms at 16K

<!-- Placeholder (prose): the result table (seeds averaged): val loss at 4K (and its change from the base), loss in the last position bucket, the loss of the second half of each window with the whole 16K window minus with its last 4K [95% CI], near recall (gaps 32-512) and far recall (4-16K back) at 16K, the reach (every octave bin up to it has a lower 95% Wilson bound of at least 0.5) and the detectable range (above 0.25). Each arm's row is the mean over the seeds named beside it (PI and YaRN three, YaRN-prog one). The base row is the zero-shot PI model at 16K with the base's own val loss; the Qwen3-0.6B-Base row (trained to 32K) has recall only, its losses are in its own tokens. The line under the table compares progressive and one-shot YaRN at seed 0 of each (paired by window and query), a single-seed comparison (claim 7); its bounds get extra decimals where they would round to 0, so a difference below the printed precision shows as such. The plot shows the training loss of each arm's first seed; the dotted line marks the progressive arm's switch to 16K. Claim 6b reads the ctx column (16K minus 4K on the same target tokens): an interval below 0 after training, against the base row's zero-shot value; else "no measurable benefit from context beyond 4K". Claim 13 also reads the Qwen3 row. -->

```{.python .input #long-context-arms}
%%tab pytorch
lc = d2l.lc_arms(ARMS, base='r2', tokens=TOKENS, seq_len=16_384,
                 batch_tokens=262_144, peak_lr=1e-4, warmup=20,
                 decay_frac=0.15, long_share=0.5)
d2l.show_lc_table(lc)
d2l.plot_lc_train(lc.logs)
```

<!-- Placeholder (prose): left, loss by position bucket, the zero-shot PI and YaRN models against the three trained arms; right, the loss against the context given (each target of the second half sees c / 2 to c tokens back, minus its loss given 256), in bits per byte so that Qwen3's other tokenizer is comparable. A flat right panel means the extra context is accepted, not used. -->

```{.python .input #long-context-position}
%%tab pytorch
d2l.show_lc_loss(lc)
```

<!-- Placeholder (prose): recall against the gap (octave bins, 500 windows per length, seeds pooled per query, Wilson bands; dashed chance, dotted 0.5 and 0.25), at 16K and inside the trained window at 4K, with Qwen3-0.6B-Base in grey. Reach is the upper edge of the largest bin up to which every bin's lower 95% Wilson bound is at least 0.5; "detectable to" uses 0.25. Both thresholds are protocol parameters, not another model's; the lines below are seed 0's. Each arm's line adds its far recall (4-16K back) minus the unscaled base's recall in its own top bin (2-4K back at 4K), an unpaired difference with a window-resampled interval (claim 8); then Qwen3-0.6B-Base's recall per bin at 16K with Wilson intervals (claim 13: per-bin recall as measured; the pilot found the fallback likely, so say "the reference falls to x% at gaps beyond y" from lc-reference's 250 windows and never call Qwen3 a 90% ceiling). The per-bin Wilson bands treat queries as independent and do not account for the eight queries of a window sharing its text or windows from one book overlapping; the differences resample whole windows. Claim 8's fallback: if far recall stays at chance, "extension made 16K positions usable and left reach at x". -->

```{.python .input #long-context-recall-grid}
%%tab pytorch
#@save
def reach(bins, lo=0.5, strict=False):
    """The upper edge of the largest octave bin of gap up to which every
    bin's lower 95% Wilson bound is at least lo (above lo if strict); 0
    when the first bin fails."""
    b = bins.sort_values('gap_lo')
    ok = (b['lo'] > lo if strict else b['lo'] >= lo).to_numpy()
    k = len(ok) if ok.all() else int(ok.argmin())  # bins that pass
    return int(b['gap_hi'].iloc[k - 1]) if k else 0

d2l.show_lc_recall(lc, reach)
```

<!-- Placeholder (prose): the long-context gain per octave of gap from 256 (the needle's axis): how much the distant occurrence of a rare token lowers its loss, against a sliding context of 256 tokens; seeds pooled per window. -->

```{.python .input #long-context-gain}
%%tab pytorch
d2l.show_lc_gain(lc)
```

### Which Measure Moves First

<!-- Placeholder (prose): from the periodic needle evaluations of the seed-1 and seed-2 arms (every 64 steps and before the first: 50 needle windows at 16K, 16 test windows; the evaluation draws nothing from the training generators, so these seeds train as seed 0's recipe would): the first tokens at which the loss in the last bucket comes within MARGIN of the base's loss on its trained window (the same 16 windows), near recall has a lower Wilson bound of 0.5, far recall (4-16K back) one above 0.25, and the gain beyond 4K is positive. Seed 0 and the progressive arm were trained before this evaluation existed and have no timeline (were they re-trained, a progressive arm's first-stage evaluations run at its 8K window and are not counted here). The cell names those arms on its `no timeline:` line without a reason; the prose gives it. A criterion met at the evaluation before the first step prints `at start`: it is not a move made by training, so claim 6 orders only the criteria met after training starts, and the prose says which ones held at the start. State the order per arm as measured; 'perplexity first' only if it holds in every arm and seed. -->

```{.python .input #long-context-first-steps}
%%tab pytorch
MARGIN = 0.05  # nats above the base's loss on its trained window
d2l.show_lc_first_steps(lc, margin=MARGIN)
```

### The Cost at Short Context

<!-- Placeholder (prose): what the extension costs at short context: the val loss at 4K, the suite mean (five tasks, 2,000 items each) and near recall at 4K minus the base's, with intervals paired by window, item and query (seeds averaged). Claim 9: a cost is stated where a paired interval excludes 0; else "no measurable cost". -->

```{.python .input #long-context-short-cost}
%%tab pytorch
d2l.show_lc_short(lc)
```

### Seeds at Small Scale

<!-- Placeholder (prose): three seeds at T2 bound a scheme difference only loosely; the T1 configuration repeats the comparison on the t1 checkpoint (trained at 1K) extended to 4K with five seeds of the data order (16M tokens per arm). Each arm's row is the mean over the seeds named beside it. The base row's val loss is the base's own at 1K; its 4K columns are the base scaled by PI without training (250 needle windows). The last column is recall 32-64 back at 1K, descriptive: the base's row is the 1K control (how well the base copies inside its trained window, as measured), the arms' rows show what extension to 4K left of that short-range copying, each row with its own interval, so the rows are not a paired comparison; the paired short-context cost (claim 9) is the previous cell's. Then the scheme difference YaRN minus PI per seed, at T1 (five seeds) and at T2 (three): the loss in the last bucket, near and far recall. Every seed is scored on the same evaluation windows, so a t interval over the seeds measures only how much training seeds vary on these windows; the line prints for each measure the wider of that interval and the interval from resampling the windows (seeds held fixed), named in parentheses (seeds or windows). A direction is claimed only where both intervals exclude zero; otherwise 'differ by less than x' (claim 5). A bound printed with more decimals than the others is nonzero but smaller than the usual precision; its sign decides whether the interval excludes zero. -->

```{.python .input #long-context-t1}
%%tab pytorch
ARMS_T1 = {  # the t1 checkpoint (trained at 1K) to 4K; seeds: data order
    'PI': dict(scheme='pi', stages=((4096, 4, 1.0),), seeds=(0, 1, 2, 3, 4)),
    'YaRN': dict(scheme='yarn', stages=((4096, 4, 1.0),),
                 seeds=(0, 1, 2, 3, 4)),
    'YaRN-prog': dict(scheme='yarn', stages=((2048, 2, 0.4),
                                             (4096, 4, 0.6)), seeds=(0,))}
t1 = d2l.lc_arms(ARMS_T1, base='t1', tokens=16_000_000, seq_len=4096,
                 batch_tokens=65_536)
d2l.show_lc_seeds(t1, lc, 'YaRN', 'PI')  # T2 never uses the stand-in
```

## Toward a Million Tokens

<!-- Placeholder (prose): one paragraph, forward references only: sparse and linear mixing (chap_lm_architectures), context parallelism (chap_lm_parallelism), KV tiers (chap_lm_hardware, chap_lm_serving); HELMET as the next evaluation tier; no experiment. -->

## Summary

<!-- Placeholder (prose): summary bullets. -->

## Exercises

<!-- Placeholder (prose): 5 to 7 exercises: the attention-entropy change under the YaRN temperature; the pairs that complete a rotation within 4K (23 of 32) and the NTK overhang of the slowest pair; recall against needle depth at each length (the committed grid, re-plotted); the long-data share at {0, 25, 50, 75, 100}% (T1, about 25 minutes); the NTK-aware arm trained (T1); RULER single-needle on Qwen/Qwen3-1.7B at 4K to 16K; open: 32K with scale 8 from the 16K checkpoint. -->
