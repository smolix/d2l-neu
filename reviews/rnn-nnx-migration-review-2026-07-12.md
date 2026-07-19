# Review: JAX linen → NNX migration, chapters 9–10 (RNN chapters)

*2026-07-12. Three parallel Opus reviews of commit `01a4f3e` ("Migrate JAX backend
to Flax NNX") + outputs refresh `94f7419`, scoped to
`chapter_recurrent-neural-networks/` and `chapter_recurrent-modern/`. Diff base
`6ae5172`. Findings verified against the outputs store, `d2l/jax.py`, live
runtime checks in `.venv-jax` (jax 0.10.2 / flax 0.12.7), and rendered SVGs.*

**Bottom line:** the migration is semantically faithful where it was applied —
gate algebra, state threading, parameter registration, masked loss,
teacher-forcing wiring, RNG hygiene, and the `#@save` APIs the attention chapter
subclasses all check out, and JAX↔PyTorch perplexity parity holds (scratch 7.3
vs 7.4; concise 7.3 vs 7.6). Two MUST-FIX regressions and two SHOULD-FIXes.
Per Alex's 2026-07-12 direction, all fixes are folded into the ch. 9–10
modernization (branch `rnn-ssm-modernization`) rather than hotfixed on main;
the tracer-leak fix is pulled forward into Phase 1 because every new section's
training figures depend on it.

## MUST-FIX

### 1. Every JAX RNN-LM training figure renders blank (tracer leak kills the ProgressBoard thread)
Affected store outputs: `rnn-scratch`, `rnn-concise` (ch. 9); `lstm` ×2, `gru` ×2,
`deep-rnn` ×2 (ch. 10). Visually confirmed: JAX figures are empty axes
(y 0.0–1.0, no curves, no legend); PyTorch counterparts show train_ppl/val_ppl.

Root cause: post-migration `RNNLMScratch`/`RNNLM` (`d2l/jax.py:902-910, 955-972`)
call `self.plot('ppl', d2l.exp(l), ...)` from inside `training_step`/
`validation_step`, which now run under the `@nnx.jit`-decorated
`_trainer_train_step` (`d2l/jax.py:522-527`). The plotted value is a JAX tracer;
the ProgressBoard's background drawing thread later evaluates the deferred
`lambda: d2l.numpy(d2l.to(v, cpu))` on the leaked tracer, raises, and dies — so
*no* points (not even the trainer's eager per-batch loss) are ever accumulated.
This is a regression: the linen base plotted outside jit and its `d2l/jax.py`
carried an explicit comment guarding against exactly this ("tracer-leak errors
from closure side-effects inside jit"). `Classifier`/regression models are
unaffected because `fit_epoch` plots them eagerly (`d2l/jax.py:420, 430-433`).

Fix: never queue a tracer — have the jitted step return the loss and plot
eagerly outside jit (the Classifier pattern), preserving ppl (not raw loss) as
the plotted metric for parity with the PyTorch tab. Fix the `#@save` blocks,
`make lib`, re-execute the affected JAX notebooks, recapture scoped
`--frameworks jax`. **Routed to: Phase 1 (verified pattern must exist before
any new section trains an LM).** The same pattern must be used in the new 9.5.

### 2. `bi-rnn.md` JAX tab silently skipped — still linen, broken under the NNX base class
`chapter_recurrent-modern/bi-rnn.md:116-153`: no `nnx` import; `BiRNNScratch`
still uses the linen dataclass-field + `setup()` idiom. Since `d2l.Module` is
now `nnx.Module`, `setup()` never runs and the class is inert/broken. It escapes
the build only because no JAX cell ever instantiates it (the store shows the
cell with no output). Shipped reference code is nonetheless broken for readers.

Fix: **routed to Phase 3 (10.1 rewrite)** — bi-rnn.md is absorbed into the new
`lstm.md` §10.1 where the bidirectional subsection is a concise
`bidirectional=True` shapes demo; the linen scratch class disappears entirely.
Must not survive into the rewritten chapter.

## SHOULD-FIX

### 3. JAX seq2seq/Bahdanau showcase translations regressed (BLEU 3/3 → 1/3)
`seq2seq.md` cell `seq2seq-evaluation-of-predicted-sequences-2` (and the same
sentence regressing in `bahdanau-attention.md`): linen store had 3/3 BLEU 1.000;
NNX store has "i'm calm ." → "je suis en retard ." (0.548) and "i'm home ." →
"je suis malade ." (0.512); PyTorch remains 3/3. No logic defect found (masked
loss, teacher forcing, state wiring, dropout toggling, greedy loop all correct);
most consistent with weaker default init — the flax tabs never applied the
explicit Xavier init that pytorch/mxnet apply via `init_seq2seq`. Alex
explicitly flagged this class of result as "should be fixed".

Fix: **routed to Phase 3 (10.2 rewrite)** — give the JAX (and TF, if kept) tab
explicit Xavier init to match, retrain, and only bless showcase outputs that
translate correctly. Re-check `bahdanau-attention.md` outputs after the shared
classes change (library rebuild ⇒ its notebooks are "plausibly affected").

### 4. `rnn-concise.md` JAX tab omits the untrained-model prediction the prose promises
The other three tabs end with `model.predict('it has', 20, data.vocab)` before
training; JAX ends at model construction. Was a linen limitation (predict needed
trained params); NNX removed it (verified: untrained predict runs fine).

Fix: **routed to Phase 2 (9.5 merge)** — the merged section's JAX tab includes
the untrained predict for cross-tab parity.

## NITs (fold in where touched, else ignore)

- `rnn-scratch.md:629-681`: `clip_gradients` is dead code in the JAX path
  (clipping actually applied via `optax.clip_by_global_norm` in `fit()`); the
  shared prose "invoked by the `fit_epoch` method" is false for JAX. Fix the
  prose in the 9.5 rewrite.
- `deep-rnn.md:163`: `StackedRNNScratch` layer 1 seeds `nnx.Rngs(1)`, same seed
  as `RNNLMScratch`'s head stream. Harmless (different shapes); avoid seed
  overlap in new code.
- `seq2seq.md:743`: redundant `d2l.reshape(Y, -1)` on an already-flat array
  (pre-existing, carried from linen).
- `bi-rnn.md:118`: unused `jnp` import (dies with the file).

## Verified clean (load-bearing absences)

- **deep-rnn.md `trainer.fit()` incident did NOT recur**: all four tabs' fit
  calls are live in both scratch and concise cells, with real executed outputs;
  JAX prediction on par with PyTorch.
- No flax-linen leftovers anywhere in ch. 9–10 except finding #2; no
  `.apply/init_with_output/TrainState/self.param(` residue.
- Files with JAX tabs but no `nnx` (sequence, text-sequence, language-model,
  rnn, machine-translation-and-dataset, beam-search) are genuinely
  data-only/pure-jnp and were re-executed, not skipped.
- Param registration correct throughout (all weights are `nnx.Param` and receive
  gradients — verified by `nnx.state(..., nnx.Param)` leaf counts at runtime).
- Encoder→decoder state wiring, context repetition, masked cross-entropy,
  greedy predict-loop threading, and the `predict_step` 2-tuple contract the
  attention chapter unpacks: all correct and API-compatible.
- Freshness gate clean at review time (586 manifests, 0 stale).

## Framework policy amendment (Alex, 2026-07-12)

For the ch. 9–10 restructure: **PyTorch and JAX are must-have**; omit MXNet
wherever generating it is not reasonably possible, and the same applies to
TensorFlow. Carryover sections that already have working 4-tab code keep it;
new SSM/Mamba sections are pytorch+jax (+tf only if cheap). "Improve, don't
just preserve": where better showcase code or implementation improvements
exist during the restructure, make them (the BLEU regression above is the
named example).
