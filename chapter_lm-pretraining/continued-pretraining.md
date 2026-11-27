# Continued Pretraining and Mid-Training
:label:`sec_lm_continued_pretraining`

<!-- Author brief, copied verbatim from the Language Models part plan
(docs/llm/language_models_part_plan.md, "4.4 Continued Pretraining and Mid-Training · T2").
Blueprint: docs/llm/blueprints/ch04_pretraining.md section 4.5.
The plan is a specification, not book text; this comment is replaced
when the section is written.

**Idea:** A new domain is bought with forgetting of the old ones unless replay and a re-warmed schedule pay for it.
**Opening:** Qwen3-0.6B-Base continued on the book's own text for a few million tokens learns it, and a cell shows its loss on general web text rising at the same time while one item of the Chapter 1 suite drops. The reader knows warmup and decay from `sec_scheduler`; here the schedule restarts on a model that has already decayed to zero, with cold optimizer moments. The closing question is what a released model's stages look like from the outside, which the SmolLM3 reading answers.
**Teaches:**
- Re-warming and re-decaying the learning rate: a restart from a decayed checkpoint at the old peak with no warmup spikes; warmup tames cold moments; the loss bump after re-warming scales with the peak rate and heals during the decay.
- Replay as the price of retention: a mixture stream whose per-step domain counts are a seeded multinomial, exact and resumable; the forgetting-against-gain frontier, concave, with replay above the early-stopping path.
- Branching from a plateau against restoring a decayed checkpoint: the reference model's committed branch point (`lm-r2-s5100-state`) as the T1 case.
- Mid-training stages before post-training (reasoning, code, long context, a decay on high-quality data) as the production form of the same mechanism; SmolLM3-3B as the open exemplar whose stage checkpoints, configurations and logs are public.
- Reading an open trajectory: SmolLM3-3B's pretraining-stage, long-context and mid-training checkpoints scored with the Chapter 1 suite and per-domain bits per byte in one cell, read from a committed table with two checkpoints re-scored live.
**Derivation:** Forgetting against gain in a quadratic model. With domain and general Hessians $H_d$, $H_g$ and optima $\theta_d$, $\theta_g$, the replay solution at share $r$ is $\theta_r = ((1 - r)H_d + rH_g)^{-1}((1 - r)H_d\theta_d + rH_g\theta_g)$; in one dimension with $u = r h_g / ((1 - r)h_d + r h_g)$ the forgetting is $F = \tfrac12 h_g \Delta^2 (1 - u)^2$ and the gain $G = \tfrac12 h_d \Delta^2 (1 - u^2)$, so the normalized frontier is $\tilde G = 2\sqrt{\tilde F} - \tilde F$, concave with slope $1/\sqrt{\tilde F} - 1$, and early stopping traces the same curve. Smallest case: two dimensions with $H_g = \textrm{diag}(10, 0.1)$, $H_d = I$, $\theta_d = (1, 1)$, where at forgetting 0.2 replay gains 0.66 against 0.36 for early stopping. The notebook checks the one-dimensional identity to `1e-9` and the two-dimensional points at four forgetting levels.
**Notebook:**
1. Question: with and without replay, how do domain loss, general loss and the suite move when Qwen3-0.6B-Base is continued on the book corpus plus a technical corpus?
2. Model and data: `Qwen/Qwen3-0.6B-Base` (596,049,920 parameters, Apache-2.0, revision pinned) re-wrapped by `d2l.PackedGPT.adopt`; the domain stream is 90% scientific prose from the open S2ORC-derived corpus `allenai/peS2o` (ODC-By) and 10% the book corpus `d2l.book_corpus()` with every fifth section held out; FineWeb-Edu shards for replay and for the general validation loss; $T = 2048$, batch 262,144 tokens, 305 steps (80M tokens per arm), AdamW (0.9, 0.95), weight decay 0.1, warmup 10 steps then cosine to 10% of peak. Arms: A peak 1e-4 without replay; B peak 1e-4 with 25% replay; C peak 1e-5 without replay; A' is A with another data-order seed. About 3.2 to 4.1 h on one A100 40 GB plus evaluation (fallback: drop A' or 40M tokens per arm). T1: the reference model as base, 12M tokens per arm at $T = 1024$ (`lm-pes2o-12m`, `lm-fwedu-replay-12m`), arm C replaced by a branch from the plateau checkpoint with no re-warm; three seeds of arm A.
3. Check: `d2l.mix_streams` realizes the weights within three binomial standard errors over 10,000 steps and resumes exactly; the adopted model's logits match `d2l.load_qwen3` within bf16 tolerance on four sequences; the document-level cluster bootstrap recovers a synthetic 0.05-nat shift.
4. Claim that survives noise: without replay, domain loss falls and general loss rises, more at the higher rate; replay recovers most of the general-loss gap at a small cost in domain gain, with arm B above the line from the base through arm C on the frontier; differences below the seed arm's spread are "not resolved". Equal-token accounting is stated: B's 60M domain tokens against A at 60M.
5. SmolLM3 cell on a committed artifact: `lm-smollm3-trajectory-v1` (per-item suite scores and per-domain bits per byte for the ends of pretraining stages 1, 2 and 3, the two long-context stages and the mid-training checkpoint, all branches of `HuggingFaceTB/SmolLM3-3B-checkpoints`, Apache-2.0, revisions pinned), with two checkpoints re-scored live at T1 (about 6 GB each in bf16, within a 15 GiB slot).
6. Reused: `d2l.EvalSuite`, `d2l.paired_bootstrap`, `d2l.bits_per_byte`, `d2l.PackedStream`, `d2l.pretrain`, `d2l.load_reference_model(step=5100)`.
**Figures:** `fig_lm_replay_tradeoff` (plot: domain gain against forgetting for the arms and the base, with the quadratic frontier); `fig_lm_smollm3_stages` (plot: suite accuracy and per-domain bits per byte across the SmolLM3 stage checkpoints); `fig_lm_cpt_curves` (plot: domain, general and book losses against tokens for the three arms, with the re-warm bump).
**Restate / forward:** `sec_scheduler` (warmup tames a cold optimizer state, which a restart recreates; the plateau can be branched); `subsec_adamw-memory` (the moments restored from the branch point are what makes the branch cheap). Forward: none.
**Exercises:** 5 to 7. Derive: the frontier's concavity; the warmup excess in the noise-ball model ($O(\eta^2)$ against $O(\eta)$); the cold-moment spike with no warmup. Measure: forgetting against replay share at $\{0, 0.1, 0.25, 0.5\}$ at T1; the bump against peak rate. Open: a mid-training mixture for long context, as the bridge to 4.5.
**Writer notes:**
- Tokenizations and documents must be identical across arms; replay tokens are not domain tokens; restoring a decayed checkpoint at the old peak with no re-warm is the pitfall the opening shows; four passes over the book corpus overfit it, so report the held-out sections separately; Qwen3's vocabulary needs `uint32` shards, regenerated by a script and not hosted (the T1 slices are hosted).
- The SmolLM3 reading (2026-11-24) replaces the earlier plan's Olmo 3 7B checkpoints and the blueprint's OLMo 2 revisions (`allenai/OLMo-2-0425-1B` stage checkpoints): SmolLM3-3B is already the part's NoPE row (14.4), fits a T1 slot, and publishes everything the cell needs. `HuggingFaceTB/SmolLM3-3B-checkpoints` (verified 2026-11-24) has one branch per checkpoint: `stage1-step-*` to `stage3-step-*` every 40,000 steps (94.4B tokens), `lc-4k-to-32k-step-*` and `lc-32k-to-64k-step-*`, and `it-mid-training`, `it-SFT`, `it-soup-APO` and `it-LC-expert` for post-training; the stage configurations with their data weights are the `HuggingFaceTB/smollm3-configs` dataset, and the training logs are public on Weights & Biases. The cell scores the base-model branches; the post-training branches are named once, as the stages that follow. The comparison with the reference model is not an ablation (five things differ); the cell reads capability against bits per byte on offset curves.
- Fallback if `d2l.PackedGPT.adopt` fails the 1.2 name check: train the Hugging Face `Qwen3ForCausalLM` with `position_ids` (document masks then need the packed FlashAttention path; without it, train unmasked and say so).
- Licences: peS2o (ODC-By, verified 2026-10-29) and FineWeb-Edu shards ship with attribution; peS2o has no field metadata, so "technical" means scientific prose; Chapter 3's mathematics and code pools are the alternatives.
- Do not re-teach: schedules and branching (`sec_scheduler`, `subsec_scheduler-branching`), the noise-ball argument (`sec_sgd`), the data mixture (3.4). New here: re-warming, replay as an exact stream, the frontier, stages, the open trajectory.
- Blueprint: ch04 §4.5 (arms, budgets, D1 and D2, `tab_lm_cpt_stages` with OLMo 2, SmolLM3, Llama 3 and Qwen3 stages; the stage table stays, with Olmo 3 in place of OLMo 2 and SmolLM3 as the exemplar row).
**Sources:**
- Papers: [Don't Stop Pretraining](https://arxiv.org/abs/2004.10964) (domain- and task-adaptive pretraining); [Simple and Scalable Strategies to Continually Pre-train Large Language Models](https://arxiv.org/abs/2403.08763) (re-warming, re-decaying and replay at scale); [Olmo 3](https://arxiv.org/abs/2512.13961) (another open run with stage checkpoints and logs, cited for its stage table row); [2 OLMo 2 Furious](https://arxiv.org/abs/2501.00656) (the two-stage recipe with a decay on a curated mixture); [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) (annealing on high-quality data); [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) (general, reasoning and long-context stages).
- Code and documentation: [OLMo-core](https://github.com/allenai/OLMo-core) (the stage configurations as code); [SmolLM3-3B-checkpoints](https://huggingface.co/HuggingFaceTB/SmolLM3-3B-checkpoints) (the stage checkpoints of the trajectory, one branch each) and [smollm3-configs](https://huggingface.co/datasets/HuggingFaceTB/smollm3-configs) (the stage configurations); [Tulu 3 SFT mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) (named once as the post-training stage that follows).
- Courses, tutorials and blogs: [SmolLM3 blog](https://huggingface.co/blog/smollm3) (the documented three-stage recipe whose checkpoints the cell reads); [SmolLM3 training logs](https://wandb.ai/huggingface/SmolLM3-training-logs) (the run's public logs); [Olmo 3 release post](https://allenai.org/blog/olmo3) (the model flow with every stage public).
-->

<!-- Placeholder (prose): opening. The reader has a pretrained base model (Qwen3-0.6B-Base) and a target domain: reasoning traces, a question followed by a long think-aloud solution in chat markers (the base already predicts them better than web text, about 1.3 against 2.5 nats/token in the fix-pass probe; quote the arms' own step-0 numbers. The probe scored each conversation's first 2,048 tokens, question in view; the held-out trace set scores higher for the base, about 1.67 in the F22 pilot, because its documents run up to about 17K tokens and are cut into 2,048-token windows, so most windows start mid-trace with neither the question nor the earlier reasoning in view: say that the campaign's trace loss is mostly a no-context mid-trace loss, F22 review finding 2) (ruling 2026-11-25 ~16:30: traces, which every reader can read; the book is no longer trained, only evaluated). Continue its training for a few million tokens and the held-out trace loss falls, while the loss on general web text rises and the suite moves; the cell below shows arm A of this section's runs at its first suite evaluation (15 steps, 3.9M tokens). The reader knows warmup and decay (sec_scheduler); here the schedule restarts on a model that has already decayed, with cold optimizer moments. Opening question: with and without replay, how do domain loss, general loss and the suite move, and what does a released model's sequence of stages look like from outside? -->

```{.python .input #continued-pretraining-imports}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #continued-pretraining-imports-extra}
%%tab pytorch
import pandas as pd
```

```{.python .input #continued-pretraining-opening}
%%tab pytorch
runs = d2l.cpt_qwen_runs()
look = runs.logs.set_index(['arm', 'step']).reindex([('base', 0), ('A', 15)])
suite = d2l.suite_mean(runs.items).reindex([('base', 0), ('A', 15)])
trace, web = look['domain_loss'].to_numpy(), look['general_loss'].to_numpy()
print(f'after {look["tokens"].iloc[1] / 1e6:.1f}M trace tokens (arm A): '
      f'held-out trace loss {trace[0]:.2f} → {trace[1]:.2f} nats/token')
print(f'general web text: {web[0]:.2f} → {web[1]:.2f}')
print(f'suite ({len(d2l.CPT_TASKS)} tasks, mean accuracy): '
      f'{100 * suite.iloc[0]:.1f}% → {100 * suite.iloc[1]:.1f}%')
```

<!-- Placeholder (prose): read the three lines (quote them at the printed precision); the section's question and its plan: the schedule (re-warming), the data (replay as an exact mixture, the forgetting frontier), the four arms at T2 and the branch point at T1, then the stages of a released model (SmolLM3). Licence note for OpenThoughts3-1.2M (Apache-2.0; attribution text in the manifest), FineWeb-Edu (ODC-By) and the book (CC BY-SA 4.0). -->

## Re-warming and the Cold Optimizer

<!-- Placeholder (prose): restate sec_scheduler in one sentence (warmup tames a cold optimizer state; a decayed model sits in a narrow valley). A restart from a decayed checkpoint creates both conditions at once: fresh AdamW moments (Qwen3 ships none) and a learning rate jumping from about zero to a peak. The noise-ball picture (sec_sgd) in one sentence: the loss excess scales with the rate, so re-warming costs a bump that the re-decay heals. Branching from a plateau checkpoint (subsec_scheduler-branching) keeps the moments (subsec_adamw-memory): the T1 branch arm tests it in the third section. -->

### Gain Against Forgetting in a Quadratic Model

<!-- Placeholder (prose): the derivation (brief's Derivation): domain and general quadratics, replay solution theta_r, the 1-D normalized frontier G = 2 sqrt(F) - F (concave, slope 1/sqrt(F) - 1), early stopping traces the same curve in 1-D; in 2-D with H_g = diag(10, 0.1), H_d = I, theta_d = (1, 1), theta_g = 0 replay moves along the flat direction of the general loss first and beats early stopping. Proposition/proof style, short. -->

```{.python .input #continued-pretraining-quadratic}
%%tab pytorch
r = np.linspace(0, 1, 1001)  # 1-D: theta_g = 0, theta_d = 1, h_g = 10, h_d = 1
th = (1 - r) / ((1 - r) + 10 * r)  # the replay solution at share r
F1, G1 = th ** 2, 1 - (1 - th) ** 2  # normalized forgetting and gain
err = np.abs(G1 - (2 * np.sqrt(F1) - F1)).max()
h, th_d = np.array([10.0, 0.1]), np.array([1.0, 1.0])  # diag(H_g), theta_d
r = np.linspace(0, 1, 200_001)[:, None]
paths = {'replay': (1 - r) * th_d / ((1 - r) + r * h),  # argmin of the mix
         'early stopping': (1 - r) * th_d}  # gradient flow on L_d from 0
forget = {k: 0.5 * (h * th ** 2).sum(1) for k, th in paths.items()}
gain = {k: 1 - 0.5 * ((th - th_d) ** 2).sum(1) for k, th in paths.items()}
at = lambda k, f: np.interp(f, forget[k][::-1], gain[k][::-1])
print(f'1-D frontier identity: max error {err:.1e}')
print('forgetting  replay gain  early-stopping gain')
for f in (0.05, 0.1, 0.2, 0.4):
    print(f'{f:>10}  {at("replay", f):11.3f}  {at("early stopping", f):19.3f}')
assert err < 1e-9
assert (round(at('replay', 0.2), 2), round(at('early stopping', 0.2), 2)) == (
    0.66, 0.36)
keep = {k: forget[k] <= 0.5 for k in paths}
d2l.plot([forget[k][keep[k]] for k in paths],
         [gain[k][keep[k]] for k in paths], 'forgetting', 'domain gain',
         legend=list(paths))
```

<!-- Placeholder (prose): define both axes first: the gain is normalized (the domain loss's excess at the base, L_d(0) = 1, so gain 1 means the domain optimum), while the forgetting is the general loss's rise in its own units (not normalized: L_g(theta_d) = 5.05 in the 2-D model), so the table's numbers depend on that choice. Then read the table: at forgetting 0.2 replay keeps 0.66 of gain against 0.36 for early stopping (quote the printed values); why (replay spends its forgetting along the general loss's flat direction); the frontier is concave, so the first units of forgetting buy the most gain. What the model leaves out (curvature changes, data overlap); the real arms test whether replay lands above the line from the base through the low-rate arm. -->

## Replay as an Exact Mixture and the Forgetting Frontier

<!-- Placeholder (prose): replay mixes general data into the domain stream. A naive mixture samples documents with probability r and drifts; this one draws each step's per-stream counts from a multinomial seeded by (seed, step), takes each stream's next windows in that stream's own permuted order (the order PackedStream uses), and keeps per-stream cursors as prefix sums computed from the seed alone, so the state is (seed, step) and a resumed run reads the same batches. Restate the data mixture of sec_lm_data_mixtures in one sentence. -->

```{.python .input #continued-pretraining-mixstream}
%%tab pytorch
class MixStream:  #@save
    """Batches of global_batch windows from several streams: step s draws
    each stream's count from a multinomial seeded by (seed, s) and takes
    that stream's next windows; the cursors follow from the seed alone."""
    def __init__(self, streams, weights, global_batch, seed=0, rank=0,
                 world=1):
        self.streams, self.global_batch = streams, global_batch
        self.weights = np.asarray(weights, dtype=float) / np.sum(weights)
        self.seq_len = streams[0].seq_len
        self.eos_id = streams[0].shards.eos_id
        self.seed, self.rank, self.world, self.step = seed, rank, world, 0
        self._cum = np.zeros((1, len(streams)), dtype=np.int64)

    def counts(self, step):
        rng = np.random.default_rng([self.seed, step])
        return rng.multinomial(self.global_batch, self.weights)

    def cursors(self, step):
        """Windows each stream has given before step (prefix sums)."""
        while len(self._cum) <= step:  # extend the table, doubling it
            n = len(self._cum)
            c = np.stack([self.counts(s) for s in range(n - 1, 2 * n - 1)])
            self._cum = np.concatenate([self._cum,
                                        self._cum[-1] + c.cumsum(0)])
        return self._cum[step]

    def batch(self, step):
        W = np.concatenate([d2l._stream_windows(s, int(a), int(k)) for s, a, k
                            in zip(self.streams, self.cursors(step),
                                   self.counts(step))])
        W = W[np.random.default_rng([self.seed, step, 1]).permutation(len(W))]
        return d2l._window_batch(W[self.rank::self.world], self.eos_id)
```

```{.python .input #continued-pretraining-mix-streams}
%%tab pytorch
def mix_streams(streams, weights, global_batch, seed=0, rank=0,  #@save
                world=1):
    """A MixStream: global_batch windows per step from streams in the
    shares weights (rank's slice of each batch)."""
    return MixStream(streams, weights, global_batch, seed, rank, world)
```

<!-- Placeholder (prose): the library adds state_dict/load_state_dict (seed, step and the streams' layouts) and iteration. The check: two stand-in streams (the FineWeb-Edu validation shard under two data seeds) mixed 3:1 at 128 windows per step; the realized share over 10,000 steps against three binomial standard errors; a run that stops after 4,321 steps and resumes from its state reads the same next 100 batches. -->

```{.python .input #continued-pretraining-mix-check}
%%tab pytorch
val = d2l.reference_shards('val')
streams = lambda: [d2l.PackedStream(val, 256, 128, seed=s) for s in (0, 1)]
mix = d2l.mix_streams(streams(), (0.75, 0.25), 128)
counts = np.stack([mix.counts(s) for s in range(10_000)])
share = counts[:, 1].sum() / counts.sum()
se = math.sqrt(0.25 * 0.75 / counts.sum())
print(f'replay share over 10,000 steps: {share:.5f} (target 0.25, '
      f'3 SE = {3 * se:.4f})')
mix.step = 4_321  # a run that stopped here saved mix.state_dict()
resumed = d2l.mix_streams(streams(), (0.75, 0.25), 128)
resumed.load_state_dict(mix.state_dict())
a, b = iter(mix), iter(resumed)
same = all(torch.equal(next(a)['input_ids'], next(b)['input_ids'])
           for _ in range(100))
print(f'resumed at step 4,321: next 100 batches '
      f'{"identical" if same else "DIFFERENT"}')
assert abs(share - 0.25) < 3 * se and same
```

### Measuring Gain and Forgetting

<!-- Placeholder (prose): the two numbers per arm: domain gain = base loss minus arm loss on the domain validation set, forgetting = arm loss minus base loss on the general set, both in nats per token. Each is a ratio (summed NLL over scored tokens), and tokens within a document are correlated, so the interval resamples whole documents; base and arms share each set's resamples, which makes the differences paired. -->

```{.python .input #continued-pretraining-forgetting-gain}
%%tab pytorch
def forgetting_gain(base, docs, n_boot=2000, seed=0):  #@save
    """Per arm: domain gain (base minus arm loss on the domain set) and
    forgetting (arm minus base loss on the general set), in nats per token,
    with 95% intervals from resampling documents (the same resamples for
    the base and every arm); fg.attrs['draws'] keeps the replicates."""
    draws = {}
    for i, (s, key, sign) in enumerate((('domain', 'gain', 1),
                                        ('general', 'forget', -1))):
        ids = base.loc[base['set'] == s, 'doc_id'].to_numpy()
        if len(ids) == 0:  # a pending result
            continue
        idx = np.stack([*d2l.cluster_resample(ids, n_boot, seed + i),
                        np.arange(len(ids))])  # the point estimate last
        def loss(df):  # token-weighted: summed NLL over scored tokens
            d = df[df['set'] == s].set_index('doc_id').loc[ids]
            nll, n = d['sum_nll'].to_numpy(), d['n_tokens'].to_numpy()
            return nll[idx].sum(1) / n[idx].sum(1)
        for arm, df in docs.groupby('arm', sort=False):
            draws.setdefault(arm, {})[key] = sign * (loss(base) - loss(df))
    rows = [dict(arm=arm, **{f'{k}{q}': v for k, x in d.items() for q, v in
                             zip(('', '_lo', '_hi'), (x[-1], *np.percentile(
                                 x[:-1], [2.5, 97.5])))})
            for arm, d in draws.items()]
    fg = pd.DataFrame(rows, columns=['arm', 'gain', 'gain_lo', 'gain_hi',
                                     'forget', 'forget_lo', 'forget_hi'])
    fg.attrs['draws'] = draws
    return fg
```

<!-- Placeholder (prose): the check on synthetic documents: 400 documents per set with lengths from 100 to 3,000 tokens and per-document losses around 2.5 nats; the arm is 0.05 nats per token better on the domain set (with per-document noise) and unchanged on the general set. -->

```{.python .input #continued-pretraining-estimator-check}
%%tab pytorch
rng = np.random.default_rng(0)
n_tok = rng.integers(100, 3000, 800)
per_tok = 2.5 + 0.3 * rng.standard_normal(800)
synth = pd.DataFrame(dict(set=np.repeat(['domain', 'general'], 400),
                          doc_id=np.tile(np.arange(400), 2), n_tokens=n_tok,
                          sum_nll=per_tok * n_tok))
shift = np.where(synth['set'] == 'domain', 0.05, 0.0)
noise = 0.05 * rng.standard_normal(800) * (synth['set'] == 'domain')
arm = synth.assign(arm='X', sum_nll=(per_tok - shift + noise) * n_tok)
est = d2l.forgetting_gain(synth, arm).iloc[0]
print(f'planted 0.050: estimated {est.gain:.3f} '
      f'[{est.gain_lo:.3f}, {est.gain_hi:.3f}]')
assert est.gain_lo < 0.05 < est.gain_hi
```

## The Four Arms on Qwen3-0.6B-Base, and the Branch Point at T1

<!-- Placeholder (prose): the base: Qwen3-0.6B-Base (596,049,920 parameters, Apache-2.0) re-wrapped by PackedGPT.adopt so that the training loop of sec_lm_reference_model runs on it with document masks. The check: without a context the adopted model reproduces load_qwen3's logits; with a one-document mask (the packed attention path) it matches the causal path within bf16 rounding. Fallback named in the brief if adopt failed (it does not; say so only if the output shows otherwise). -->

```{.python .input #continued-pretraining-adopt}
%%tab pytorch
qwen = d2l.load_qwen3('Qwen/Qwen3-0.6B-Base', device=d2l.try_gpu())
packed = d2l.PackedGPT.adopt(qwen)
X = d2l.qwen3_val_tokens(n=4, T=512).to(d2l.try_gpu())
one_doc = d2l.PackCtx(torch.zeros_like(X, dtype=torch.int32),
                      torch.arange(512, device=X.device).int().expand_as(X))
with torch.no_grad():
    ref, plain, masked = qwen(X), packed(X), packed(X, one_doc)
nll = lambda z: F.cross_entropy(z[:, :-1].flatten(0, 1).float(),
                                X[:, 1:].flatten())
print(f'adopt, no context: '
      f'{"identical logits" if torch.equal(ref, plain) else "DIFFERENT"}')
print(f'adopt, one-document mask vs causal: loss {nll(masked):.4f} vs '
      f'{nll(plain):.4f} nats/token (bf16)')
assert torch.equal(ref, plain) and abs(nll(masked) - nll(plain)) < 2e-2
del qwen, packed, ref, plain, masked
d2l.lm_free()
```

<!-- Placeholder (prose): the data (ruling 2026-11-25 ~16:30: reasoning traces, which every reader can read). The domain stream is reasoning traces from OpenThoughts3-1.2M (Apache-2.0; licence and attribution in the manifest): a pinned subset, every 10th of its 120 files (120,000 conversations, the dataset's math/code/science mix), one document per conversation in ChatML markers with the content verbatim (the question, then the assistant's reply opening a <think> block; not Qwen3's full chat template), so the arms are a small causal twin of SmolLM3's reasoning stage. Most replies stop at the dataset's generation cap of about 16K tokens: about two thirds of the training documents (and of their tokens) never close the <think> block and end mid-sentence, and some that close it are cut in the answer (quote the per-ending counts from the manifest; the author's ruling on keeping them is pending, F20 review finding 1). Decontaminated against GSM8K test and MATH-500 (13-gram; quote the counts from the manifest). The held-out traces are 500 conversations whose questions no training document asks (question-disjoint, near-copies of training text replaced); their mix is math-heavier than training (about 75% math against 67%, since code questions carry many annotations and held-out skips questions that training asks; quote the manifest's counts); val is their first 2.2M tokens; the arms see about 84M trace tokens once each, no repetition. The book is no longer trained: only its held-out sections are evaluated (old-domain technical prose). Replay and the general validation set are FineWeb-Edu as before. Qwen3's vocabulary needs uint32 shards, which a script regenerates; a normal run reads the committed summary below and opens no shard. -->

```{.python .input #continued-pretraining-data}
%%tab pytorch
data = d2l.cpt_data('t2')
print(data.table.rename(columns=dict(tokens='tokens (Qwen)')).to_string(
    index=False, formatters={'documents': '{:,}'.format,
                             'tokens (Qwen)': '{:,}'.format}))
```

### The Arms

<!-- Placeholder (prose): the arm protocol as a markdown table (T = 2,048; 128 x 2,048 = 262,144 tokens per step; 305 steps = 79,953,920 tokens; AdamW (0.9, 0.95), eps 1e-8, weight decay 0.1, clip 1.0; no z-loss, since the base was not trained with the book's z-loss; bf16 autocast with fp32 master weights; loss chunk 2,048; document masks; warmup 10 steps then cosine to 10% of the peak; fresh AdamW moments). Arms: A peak 1e-4 without replay; B peak 1e-4 with 25% replay; C peak 1e-5 without replay; A′ is A under another data seed, the yardstick for "not resolved". Evaluations: every 3 steps for the first 30 (the bump), then every 15 on 0.5M tokens of each validation set; at the end the full sets (2M tokens each), the suite with full item sets and GSM8K in two formats (500 items each, base and every arm). The domain stream is the reasoning traces alone, mixed with replay in the shares (1 − r, r); the book is never trained. Each arm's job folder records its configuration, its base and its data (the manifest's layout hashes), so a folder trained on another corpus is refused. Runtime on the 4090s and the A100 package (the campaign's measured numbers). -->

```{.python .input #continued-pretraining-arms}
%%tab pytorch
ARMS = {'A': dict(peak_lr=1e-4, replay=0.0, data_seed=0),
        'B': dict(peak_lr=1e-4, replay=0.25, data_seed=0),
        'C': dict(peak_lr=1e-5, replay=0.0, data_seed=0),
        'A′': dict(peak_lr=1e-4, replay=0.0, data_seed=1)}
runs = d2l.cpt_arms(ARMS, base='Qwen/Qwen3-0.6B-Base', tokens=80_000_000,
                    seq_len=2048, batch_tokens=262_144, warmup=10,
                    final_frac=0.1)
cols = {'arm': 'arm', 'domain_loss': 'trace loss',
        'general_loss': 'general loss', 'book_loss': 'book loss',
        'suite': 'suite mean', 'gsm8k_chat': 'GSM8K chat'}
percent = lambda v: '–' if pd.isna(v) else f'{100 * v:.1f}%'
print(runs.table[list(cols)].rename(columns=cols).to_string(
    index=False, float_format='{:.3f}'.format,
    formatters={'suite mean': percent, 'GSM8K chat': percent}))
d2l.plot_cpt_curves(runs)
```

<!-- Placeholder (prose): read the table and the curves (claim 4): without replay the held-out trace loss falls and the general loss rises, more at the higher rate (A against C); quote the differences only where they exceed the A/A′ spread, else "not resolved". The held-out book (never trained, old-domain English technical prose) is a second general set: say whether it moves with FineWeb-Edu. GSM8K is read in the next cell. -->

### What the Arms Learned, on a Test Set

<!-- Placeholder (prose): a lower loss on held-out traces is not yet a skill. GSM8K (chapter 18) asks the question directly, in two formats: the 8-shot completion format of a base model (chapter 18's gsm8k_cot, 256 new tokens) and one ChatML user turn with room to think (4,096 new tokens), the format of the traces; greedy, the same 500 items for every model; the answer is the boxed number (else the last number) after the reply's </think>, and a reply cut at the budget or one that never closes its think block counts as no answer (the pre-registered "chat" column). The "forced" column (budget forcing, s1, Muennighoff et al. 2025; F21 review finding 1) closes the think block of such a reply, adds "The final answer is \boxed{" and lets the model write the number (at most 32 tokens): every item then has an answer, and "cut" says how many needed the push. Which of the two chat columns claim 13 reads, and whether the budget stays at 4,096, is the author's ruling before F23 (pending). Most training traces stop at the dataset's generation cap without closing </think> (about two thirds), so an arm may learn to think past any budget: the columns "chat cut" and "chat no </think>" show it, and "8-shot chat tokens" shows chat markers leaking into the completion format (Olmo 3 §3.5.4). Claim 13: quote changes against the base only where the paired interval excludes 0; "no measurable change at 80M tokens" if none does; a chat drop that comes with more never-closed think blocks than the base's is a format effect of the truncated traces, not forgetting; a completion drop that comes with chat tokens is a format effect. Under the T1 fallback the cell prints one line (GSM8K is scored for the Qwen arms only). -->

```{.python .input #continued-pretraining-gsm8k}
%%tab pytorch
tables = d2l.gsm8k_tables(runs.gsm8k)
if tables is None:  # the T1 stand-in
    print('GSM8K: scored for the Qwen arms only')
else:
    print(tables[0].to_string(index=False))
    print('change against the base, points [95% CI, paired items]')
    print(tables[1].to_string(index=False))
```

### The Re-Warm Bump

```{.python .input #continued-pretraining-rewarm-bump}
%%tab pytorch
def rewarm_bump(runs, within=40):  #@save
    """Per arm: the bump, the largest rise of the general loss over the
    arm's own loss at step 0 in the first `within` steps (nats per token);
    the step of that peak; and how much of it the end has healed (peak
    minus final loss). A peak on the window's last evaluation is no
    transient (the loss is still rising: forgetting) and has NaN bump."""
    rows = []
    logs = runs.logs.dropna(subset=['general_loss'])
    for arm, g in logs[logs['arm'] != 'base'].groupby('arm', sort=False):
        loss = g.set_index('step')['general_loss'].sort_index()
        early = loss.loc[:within]
        if 0 not in loss.index or len(early) < 2:
            continue
        peak = early.iloc[1:].idxmax()
        rise = early.index[-1] > peak  # falls again inside the window
        rows.append(dict(arm=arm, step=peak,
                         bump=loss[peak] - loss[0] if rise else math.nan,
                         healed=loss[peak] - loss.iloc[-1] if rise
                         else math.nan))
    return pd.DataFrame(rows, columns=['arm', 'bump', 'step', 'healed'])
```

```{.python .input #continued-pretraining-bump-table}
%%tab pytorch
bump = d2l.rewarm_bump(runs).merge(runs.table[['arm', 'peak_lr']], on='arm')
nats = lambda v, none: none if np.isnan(v) else f'{v:.3f}'
bump['bump'] = bump['bump'].map(lambda v: nats(v, 'no transient'))
bump['healed'] = bump['healed'].map(lambda v: nats(v, '–'))
cols = {'arm': 'arm', 'peak_lr': 'peak lr', 'bump': 'bump (nats)',
        'step': 'step of the peak', 'healed': 'healed by the end'}
print(bump[list(cols)].rename(columns=cols).to_string(
    index=False, formatters={'peak lr': '{:.0e}'.format}))
```

<!-- Placeholder (prose): claim 8, T2 half: each bump is measured from the arm's own loss at step 0 (the training model before its first update), so a start that differs from the base costs nothing here. The bump scales with the peak rate (A against C) and heals during the decay ("healed by the end" is the peak minus the final general loss); "no transient" marks an arm whose general loss is still rising at the window's end (forgetting, not a bump). Compare against the A/A′ spread; "no bump resolved" for arms inside it. -->

### Replay on the Frontier

```{.python .input #continued-pretraining-frontier}
%%tab pytorch
fg = d2l.forgetting_gain(runs.base_docs, runs.docs)
ci = lambda r, k: f'{r[k]:.3f} [{r[k + "_lo"]:.3f}, {r[k + "_hi"]:.3f}]'
print(f'{"arm":<4} {"gain [95% CI]":<23} forgetting [95% CI]')
for r in fg.to_dict('records'):
    print(f'{r["arm"]:<4} {ci(r, "gain"):<23} {ci(r, "forget")}')
seed = fg.set_index('arm').reindex(['A', 'A′'])
print(f'seed spread (A vs A′): gain {abs(seed.gain.diff().iloc[-1]):.3f}, '
      f'forgetting {abs(seed.forget.diff().iloc[-1]):.3f}')
pct = lambda e: f'{e.value:.0%} [{e.lo:.0%}, {e.hi:.0%}]'
kept, forgot = d2l.replay_fractions(fg, 'B', of='A')
print(kept.note or f"B: {pct(kept)} of A's gain, {pct(forgot)} of A's "
      'forgetting')
d2l.plot_tradeoff(fg, chord=('C', 'A'))
```

<!-- Placeholder (prose): claim 5: B's forgetting as a fraction of A's and B's gain as a fraction of A's, as printed with their intervals (paired: the same document resamples); "most" only if the forgetting fraction's interval lies under 50%, "small cost" only if the kept-gain interval lies over 75%; else the measured fractions. The plot: the dashed line from the base through C to A is the no-replay lower bound the next cell tests B against. Under the T1 fallback there is no arm C: the cell says so. -->

<!-- Placeholder (prose): claim 6, the test (investigation of 2026-11-25: the line from the base through C, extended past C, lies above a concave frontier, so it tested nothing). Without replay the arms trace a frontier through the base that is concave (quadratic model: slope 1/sqrt(F) - 1), so the chord between two no-replay points lies on or under it: B, whose forgetting falls between C's and A's, is measured against the chord from C to A at its own forgetting. Pre-registered out-of-range cases: if B forgets less than C, the reference is the segment from the base to C (also a lower bound there); if B forgets more than A, nothing measured bounds the frontier and the cell says so, no verdict. If B does not forget at all (point estimate at or below 0), the segment's extension below 0 bounds the frontier from above, not below: no verdict either (F21 review finding 5). Wording (F21 review finding 4: the chord is a lower bound, so only "below" places B relative to the frontier itself): "above the measured no-replay frontier (the C → A chord)", or "beats every interpolation of the no-replay runs", only if the interval excludes 0 and the excess exceeds the A/A′ spread, never "above the no-replay frontier" plain; "on the chord within the seed spread" otherwise; "below the chord" if the interval lies under 0, and then the reading is "a no-replay run between C and A does at least as well" (not "C is the better trade-off": the run that beats B is the no-replay run at B's forgetting); never "above the line". The second line gives the seed spread and the excess against the chord to A′ instead (when B forgets less than C, both references are the base → C segment, and the line says there is no A′ chord). -->

```{.python .input #continued-pretraining-frontier-chord}
%%tab pytorch
ex = d2l.frontier_excess(fg, 'B', chord=('C', 'A'))
print(ex.note or f'B − no-replay reference ({ex.ref}): {ex.value:+.3f} '
      f'[{ex.lo:+.3f}, {ex.hi:+.3f}] nats/token')
alt = d2l.frontier_excess(fg, 'B', chord=('C', 'A′'))
chord = (f'B − {alt.ref}: {alt.value:+.3f}' if 'chord' in alt.ref else
         'C → A′ chord:' if alt.note else
         'C → A′ chord: none, B forgets less than C')
print(f'seed spread (A vs A′): gain {abs(seed.gain.diff().iloc[-1]):.3f}; '
      + chord)
if alt.note:
    print(f'  {alt.note}')
```

### Equal-Token Accounting

<!-- Placeholder (prose): replay tokens are not domain tokens: B saw 25% fewer domain tokens than A in the same 80M. A and A′ were evaluated on the full sets at step 229, where they had seen as many domain tokens as B in its whole run; compare B there, never interpolated from the 15-step grid. -->

```{.python .input #continued-pretraining-equal-tokens}
%%tab pytorch
eq = runs.equal.set_index('arm').reindex(['A', 'A′'])
B = runs.table.set_index('arm').reindex(['B']).iloc[0]
m = eq['domain_tokens'].iloc[0] / 1e6
d, g = eq['domain_loss'].to_numpy(), eq['general_loss'].to_numpy()
print(f'domain loss at {m:.3g}M domain tokens: A {d[0]:.3f} '
      f'(A′ {d[1]:.3f}), B {B.domain_loss:.3f}')
print(f'general loss at the same point: A {g[0]:.3f} (A′ {g[1]:.3f}), '
      f'B {B.general_loss:.3f}')
lr = runs.logs.set_index(['arm', 'step'])['lr'] / B.peak_lr
print(f'learning rate there, share of the peak: A '
      f'{lr.get(("A", eq["step"].iloc[0]), math.nan):.0%}, '
      f'B {lr.get(("B", B.steps), math.nan):.0%}')
```

<!-- Placeholder (prose): claim 7: read the two pairs against the A/A′ spread: at equal domain tokens, what replay costs in domain loss and what it buys in general loss. State the confound the third line prints: A at the equal-token step is still at about a quarter of its peak rate, while B has decayed to 10%, so A still carries part of the noise-ball excess that B has shed; this favours B by an amount comparable to the effect (a few hundredths of a nat). Draw no conclusion from a difference of that size. -->

### The Suite

```{.python .input #continued-pretraining-suite-deltas}
%%tab pytorch
last = runs.items.groupby('arm')['step'].transform('max')
wide = runs.items[runs.items['step'] == last].pivot_table(
    index=['task', 'item_id'], columns='arm', values='score')
tasks = wide.index.get_level_values('task')
w = 1 / tasks.map(tasks.value_counts()).to_numpy()
w = w / w.mean()  # every task weighs the same in the suite mean
print(f'{"arm":<4} {"Δ suite mean (points)":>22}  95% CI')
for arm in [a for a in ('A', 'B', 'C') if a in wide]:
    r = d2l.paired_bootstrap(wide[arm] * w, wide['base'] * w)
    print(f'{arm:<4} {100 * r.mean:>+22.1f}  [{100 * r.ci[0]:+.1f}, '
          f'{100 * r.ci[1]:+.1f}]')
```

<!-- Placeholder (prose): claim 9: which suite changes are resolved (CI excludes 0); "not resolved" otherwise. The suite measures general ability, and its items are not the domain. -->

### The Branch Point at T1

<!-- Placeholder (prose): the same machinery on the reference model (sec_lm_reference_model) at T1: 12M tokens per arm at T = 1,024 from the hosted R2-tokenized slices: the same reasoning traces as plain text (User: … Assistant: …, no chat tokens; same question split) and FineWeb-Edu. Arm A under three data seeds (the spread that decides "not resolved"), B with 25% replay, the cold restart (the final checkpoint at the base run's own peak 6e-4, no warmup, fresh moments: the pitfall), the branch from the plateau state at step 5,100 (lm-r2-s5100-state: weights and AdamW moments restored, the rate continuing at 6e-4 and decaying linearly to zero), and cold-plateau, the branch with fresh moments (same weights, rate and schedule; fix pass, so that the two differ only in the moments). This is also the configuration every claim of the T2 arms falls back to while they are pending. -->

```{.python .input #continued-pretraining-t1-arms}
%%tab pytorch
ARMS_T1 = {'A': dict(peak_lr=3e-4, replay=0.0, data_seed=0),
           'A′': dict(peak_lr=3e-4, replay=0.0, data_seed=1),
           'A″': dict(peak_lr=3e-4, replay=0.0, data_seed=2),
           'B': dict(peak_lr=3e-4, replay=0.25, data_seed=0),
           # peak_lr None: the base run's own peak (6e-4)
           'cold': dict(peak_lr=None, replay=0.0, data_seed=0, warmup=0),
           'cold-plateau': dict(peak_lr=None, replay=0.0, data_seed=0,
                                warmup=0, start='plateau',
                                schedule='linear', moments='fresh'),
           'branch': dict(peak_lr=None, replay=0.0, data_seed=0, warmup=0,
                          start='plateau', schedule='linear')}
t1 = d2l.cpt_arms(ARMS_T1, base='r2', tokens=12_000_000, seq_len=1024,
                  batch_tokens=65_536)
table = t1.table.merge(d2l.rewarm_bump(t1)[['arm', 'bump']], on='arm',
                       how='left')
cols = dict(arm='arm', bump='bump', domain_loss='trace',
            general_loss='general', book_loss='book')
print(table[list(cols)].rename(columns=cols).to_string(
    index=False, na_rep='–', float_format='{:.3f}'.format))
gaps = d2l.early_gaps(t1, [('cold-plateau', 'branch'), ('cold', 'A')],
                      seeds=['A', 'A′', 'A″'])
print('largest general-loss gap over steps 1-40 (nats/token)')
print(gaps.to_string(index=False, float_format='{:.3f}'.format))
```

<!-- Placeholder (prose): the table's units (nats per token; "–" is an arm with no transient bump, or the base). Claims 8b and 8c are read from the second table (F21 review finding 3: on traces the arms' general loss may still rise at step 40, so the bump is "–" and a bump difference undefined; the gaps are defined either way). Claim 8c (fix pass: run B compared the branch with A at another rate, which tested nothing): branch and cold-plateau start from the same plateau weights with the same data order, rate and schedule and differ only in the AdamW moments, restored against fresh, so forgetting cancels in their paired curves; the statistic is the largest gap cold-plateau − branch in general loss over steps 1-40, against the yardstick, the largest gap between A's three seeds over the same steps ("the restored moments removed / reduced the early rise by x" if the gap exceeds the yardstick; "no difference resolved" otherwise). Mechanism for the prose: the s5100 moments were estimated at 128 x 4,096 = 524K tokens per step, the branch steps at 64 x 1,024 = 65K, so its gradient noise is larger than the restored second moments expect and it takes larger steps until v adapts (beta2 = 0.95, about 20 steps); run B's branch bump of 0.195 fits that. Claim 8b: the cold restart (final checkpoint, fresh moments, the base run's peak 6e-4 with no warmup) against A at matched steps, the largest gap cold − A over steps 1-40 against the same yardstick ("no spike resolved" if inside it; the gap includes cold's doubled peak rate, so a resolved gap reads "the cold restart's general loss rose faster", and "spike" only if cold's bump is a transient in the first table). Branching from a plateau is the cheap way to continue a run that one controls; a released model ships no moments, which is the Qwen case. -->

## Mid-Training Stages and the SmolLM3 Trajectory

<!-- Placeholder (prose): production runs continue pretraining in stages before post-training: a decay on high-quality data (Llama 3's annealing, OLMo's two-stage recipe), reasoning and code mixtures, long context. The stage table tab_lm_cpt_stages (markdown): Olmo 3, SmolLM3 (the exemplar row), Llama 3, Qwen3, with each stage's tokens, data emphasis and context length. SmolLM3-3B publishes a checkpoint at the end of every stage (HuggingFaceTB/SmolLM3-3B-checkpoints), its stage configurations (smollm3-configs) and its training logs; token counts come from the configuration files (2,359,296 tokens per step) and the blog. The post-training branches (it-SFT, it-soup-APO, it-LC-expert) are the stages that follow; this cell scores the base-model branches. -->

```{.python .input #continued-pretraining-smollm3}
%%tab pytorch
SMOLLM3_STAGES = ('stage1', 'stage2', 'stage3', 'lc32k', 'lc64k', 'mid')
traj = d2l.smollm3_trajectory(SMOLLM3_STAGES)
short = dict(hellaswag='hswag', arc_easy='arc-e', arc_challenge='arc-c',
             winogrande='wino')
print(traj.table.rename(columns=short).to_string(
    index=False, float_format=lambda v: f'{100 * v:.1f}'))
print(traj.bpb.to_string(index=False, float_format='{:.3f}'.format))
d2l.plot_smollm3(traj)
```

<!-- Placeholder (prose): read the two tables and the plot. The token counts are reconstructions from the stage configurations (smollm3-configs: 2,359,296 tokens per step; the stage-end steps) and the SmolLM3 blog (stages end at 8T, 10T and 11.1T; two long-context stages of about 50B tokens each; mid-training 140B): cite both, and quote them as the blog rounds them. The next cell gives the changes between adjacent checkpoints with intervals. -->

```{.python .input #continued-pretraining-stage-deltas}
%%tab pytorch
deltas = d2l.stage_deltas(traj)
ci = lambda r, k: f'{r[k]:+.1f} [{r[k + "_lo"]:+.1f}, {r[k + "_hi"]:+.1f}]'
domains = ('web', 'science', 'book')
row = lambda name, cells: f'{name:<14}' + ''.join(
    f'{c:<22}' for c in cells).rstrip()
print(row('transition', ['Δ suite mean, points [95% CI]']))
for r in deltas.to_dict('records'):
    print(row(r['transition'], [ci(r, 'suite')]))
print(row('transition', [f'Δ BPB {d}, %' for d in domains]))
for r in deltas.to_dict('records'):
    print(row(r['transition'], [ci(r, d) for d in domains]))
```

<!-- Placeholder (prose): claim 10: what each stage moves. Quote only changes between adjacent checkpoints whose interval excludes 0 (suite: items paired, every task weighing the same; bits per byte: relative change, documents paired); the rest "unchanged within the interval". Expected reading (verify against the tables): the decay and mid-training stages move the suite more than their BPB change suggests; the long-context stages change short-context scores little. -->

<!-- Placeholder (prose): two checkpoints re-scored live on 500 items per task (seed 0) against the committed full-set values: the committed table is what the run measured, and the reader can check it on one GPU. -->

```{.python .input #continued-pretraining-smollm3-live}
%%tab pytorch
live = d2l.smollm3_trajectory(['stage3', 'mid'], n=500)
acc = traj.table.set_index('checkpoint')[list(d2l.SMOLLM3_TASKS)]
full = acc.mean(axis=1)
print(f'{"checkpoint":<14}{"live suite mean (n = 500) [95% CI]":<38}'
      'committed')
for name, g in live.items.groupby('checkpoint', sort=False):
    w = 1 / g.groupby('task')['score'].transform('size')
    r = d2l.bootstrap_ci(g['score'] * w / w.mean())
    ci = f'{100 * r.mean:.1f} [{100 * r.ci[0]:.1f}, {100 * r.ci[1]:.1f}]'
    print(f'{name:<14}{ci:<38}{100 * full.get(name, math.nan):.1f}')
```

<!-- Placeholder (prose): claim 11: the committed value lies inside the live interval (or report the difference and its cause: batch drift, see score_choices). Then the capability-against-BPB reading. -->

```{.python .input #continued-pretraining-capability-bpb}
%%tab pytorch
ref = d2l.reference_evals(d2l.REFERENCE_STEPS, tasks=(
    'hellaswag', 'arc_easy', 'piqa', 'sciq', 'winogrande'))
d2l.plot_capability_bpb(ref, traj)
```

<!-- Placeholder (prose): claim 12: the reference model's checkpoints and SmolLM3's lie on offset curves of capability against bits per byte; this is not an ablation, since things differ (name them: parameters, tokenizer, data mixture, tokens seen, context length), and a sixth: SmolLM3's web BPB is measured on FineWeb-Edu validation documents, and FineWeb-Edu is part of SmolLM3's pretraining mix, so these documents or near-duplicates are probably its training data, while the reference model's validation set is disjoint from its training data. The two BPBs are also measured differently: the reference model on the first 2^21 tokens in packed windows, SmolLM3 on the first 500 documents, each scored on its own at a context of 2,048. Likewise for the arms: Qwen3-0.6B-Base may have seen FineWeb-Edu text (the general loss) and older d2l.ai text (the held-out book), which can inflate forgetting on memorized documents. Hand-off: the long-context stages are what sec_lm_long_context trains. -->

## Summary

<!-- Placeholder (prose): summary bullets. -->

## Exercises

<!-- Placeholder (prose): 5-7 exercises per spec H: derive the frontier's concavity; the warmup excess in the noise-ball model (O(eta^2) against O(eta)); the cold-moment spike with no warmup (with the cold arm of the T1 cell); measure forgetting against replay share {0, 0.1, 0.25, 0.5} at T1 (about 15 minutes, four runs); measure the bump against the peak rate at T1; open: a mid-training mixture for long context, the bridge to sec_lm_long_context. -->
