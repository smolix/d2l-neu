# Chapters 9-10 Modernization: Final Report

*2026-07-13. Branch `rnn-ssm-modernization` (base `4453a99` on main). Executed per
docs/rnn-ssm-modernization-{overview,outline,handoff}.md by a Fable orchestrator
with Fable section-writers, Opus reviewers/fixers, and Sonnet mechanical agents.
Companion review record: reviews/rnn-nnx-migration-review-2026-07-12.md.*

## Status: complete

All six phases done. 13 section files match the outline's structure and cell
inventory; 8 old files deleted with labels re-anchored (zero dangling refs,
verified corpus-wide); `_quarto.yml` + `CHAPTER_NUMBERING` consistent; the
outputs store is captured for every touched file, pruned of deleted ones, and
`--verify-fresh` is clean (576 manifests). Full book renders green: HTML,
4 PDFs (~53 MB), all slide decks (0 failures across 4 frameworks). All 24
generated figures pass the style audit; both chapter generators are
byte-idempotent. No em-dashes; no PDF tripwires.

## Chapter 9 - Sequence Models and Language Models

| § | File | Status |
|---|---|---|
| 9.0 | index.md | Rewritten (691 w) |
| 9.1 | sequence.md | Rewritten (3,109 w); kept working 4-tab linear-AR code, MLP became an exercise |
| 9.2 | text-sequence.md | Rewritten (~3,700 w); byte-level BPE from scratch; `from_tiktoken('gpt2')` reproduces tiktoken **token-for-token** (asserted in-notebook, 4 fw) |
| 9.3 | language-model.md | Rewritten (3,582 w); n-gram sampling progression; ppl/bpb reversal table; TimeMachine emits BPE ids (char escape hatch kept) |
| 9.4 | rnn.md | Revised (2,330 w); constant-memory paragraph seeds the ch10 arc |
| 9.5 | rnn-implementation.md | NEW merge of rnn-scratch+rnn-concise (3,168 w); embedding replaces one-hot; val ppl 89-97 scratch / 100-111 concise, bpb ~2.35 vs char-trigram 2.68 |
| 9.6 | bptt.md | Revised (2,205 w); randomized truncation cut; Jacobian demo (rho 0.9/1.0/1.1 -> 0.015/1/45.3); gradient-pathologies bridge label for 10.3 |
| 9.7 | decoding.md | NEW (~4,000 w); greedy/beam/temperature/top-k/top-p/min-p; `#@save` sample_next/generate/beam_search |

## Chapter 10 - Gated and Linear Recurrence

| § | File | Status |
|---|---|---|
| 10.0 | index.md | Rewritten (587 w); gate/linearize/select framing |
| 10.1 | lstm.md | Rewritten (4,366 w), absorbs gru/deep-rnn/bi-rnn; GRU 75-82 ppl beats vanilla, LSTM 91-97 matches it at the shared recipe (honest numbers); broken linen BiRNNScratch eliminated |
| 10.2 | seq2seq.md | Rewritten (2,736 w from 8,314 across 3 files); shared 4k BPE MTFraEng; chrF taught, BLEU demoted; bottleneck demo chrF 0.17->0.07 vs length |
| 10.3 | ssm.md | NEW (~4,600 w); minGRU -> scans -> ZOH -> HiPPO -> S4D; conv==scan==loop to ~1e-7; sMNIST S4D 81-83% vs init-sensitive LSTM 60-83% |
| 10.4 | mamba.md | NEW (4,050 w); selective copy (S4D 0.55-0.66 / LSTM 0.97-0.99 / Mamba 1.00); capstone gate held |

**Capstone (val ppl / bpb / params / s-per-epoch):** pytorch LSTM 93.1/2.35/298k/0.7 ·
minGRU 82.5/2.29/215k/0.8 · Mamba 78.4/2.26/489k/4.9; tensorflow Mamba 69.8;
jax minGRU 84.9 edges Mamba 88.8 (stated honestly in prose and slides).
Mamba <= LSTM held in all three frameworks. 10.3/10.4 are pytorch+jax+tf
(mxnet omitted, exclusion verified end-to-end).

## Library surface (all four backends, per docs/rnn-ssm-modernization-apis.md)

`BPETokenizer` (+`from_tiktoken`, compat shims), `sample_next`/`generate`/
`beam_search`, `associative_scan` (pytorch), `chrf`, re-hosted
RNNScratch/RNNLMScratch/RNN/RNNLM (embedding contract), LSTMScratch/LSTM/GRU
(num_inputs uniform), Encoder/Decoder/EncoderDecoder/Seq2Seq/MTFraEng (attention-
chapter contracts preserved, grep-verified). Dead classes (GRUScratch,
StackedRNNScratch, BiRNNScratch) removed with zero remaining consumers.
tiktoken 0.13.0 is a new shared dependency.

## Adversarial review outcome (3 Opus reviewers over the full branch)

Zero equation errors (every formula checked against S4/S4D/HiPPO/Mamba/Feng/
Popovic/Holtzman sources); zero dangling references; strong tab parity; worked
examples re-derived by hand. All 20 prose/slide-vs-store drift findings fixed
(4cedfa7). One MUST-FIX was mine to own and is corrected in 35cd820: the
attention chapter had never actually been re-executed against the new BPE
MTFraEng (see "build-system trap" below); after adapting its eval cells to
decode+chrF and genuinely re-executing, the transformer showcases are
1.000/1.000/1.000 chrF on all four frameworks.

## Build-system trap found and fixed

Store staleness (lib fingerprints in outputs/ manifests) and scheduler stamp
freshness can disagree: a lib-stale notebook with unchanged source keeps a
fresh stamp, never re-runs, and capture blesses old outputs under new
fingerprints - `refresh-stale` did exactly this to bahdanau/transformer
(52c663b's message wrongly claimed re-execution; corrected in 35cd820).
Fixed: `audit_outputs.py --stale-stamps` + stamp removal in `refresh-stale`;
documented in docs/build-system.md. A follow-up hardening pass (capture-side
guard, end-to-end trap simulation) is queued post-release.

## Notable deviations from the outline (full list in agent reports)

- 9.1 keeps the linear AR model (MLP as exercise) - preserves working 4-tab code.
- 9.2 compression sweep by prefix-slicing one 4,096-merge train - stronger
  demonstration of the rank-order property than retraining per size.
- 9.5 uses 50k/5k Time Machine windows (defaults memorize) and BPE-level ppl
  ~90 with a bpb aside - honest framing replaces the old char-ppl ~7 story.
- 10.2 greedy decoding uses the shared `predict_step` rather than a per-token
  step_fn (stable across all four tabs); beam table on pytorch+jax.
- 10.3 minGRU keeps tanh on the candidate (pure-linear degrades to ppl 134
  under the shared recipe; disclosed, made an exercise); classifier trainers
  add clip=1 (TF LSTM collapses otherwise; S4D measured clip-invariant).
- 10.4 Mamba alone gets dropout 0.3 + Adam 3e-4 (overfits otherwise; framed as
  a capability teaching point); jax copy-model embeds via one-hot matmul
  (nnx.Embed scatter-add on a 10-row table was 120x slower - the "1-hour jax
  notebook" performance bug, root-caused by ablation).

## Known gotchas recorded for the repo

- Keras 3 does not track plain `tf.Variable` attributes on Layers (silently
  frozen params); use `tf.keras.Variable`. Worth a build-docs note.
- JAX `lax.scan` retraces per prefix length; generation cells should use
  fixed-shape or recurrent-path stepping.
- `make figures` drifts 8 pre-existing SVGs (matplotlib version) outside these
  chapters - do not commit that drift.
- GPU pinning for run_one_notebook.py uses `D2L_ASSIGNED_CUDA`, not
  CUDA_VISIBLE_DEVICES.

## Open questions for Alex

1. **10.2 seq2seq showcase on tf/mxnet**: grammatical-but-wrong French at the
   1,024-pair scale (pytorch 3/3, jax 2/3 perfect). Prose now says this
   honestly. Accept, or drop those two showcase tabs?
2. **Bahdanau tf/mxnet**: same class of imperfection post-BPE (pytorch/jax
   perfect). The attention chapter rewrite is a separate later project; is the
   honest-imperfect interim acceptable?
3. The Time Machine BPE vocab (1,024) is trained in-notebook in ~15-45 s per
   notebook that needs it; acceptable, or worth caching as a data asset later?

## Release state

Per Alex's 2026-07-13 directive (supersedes the handoff's "do not push"):
a clean-rebuild release pipeline (make clean, full 4-framework re-run, html/
pdf/slides/notebook-zips, R2 upload, commit + push) runs after this report,
followed by the build-trap hardening agent. Branch and HEAD at report time:
`rnn-ssm-modernization` @ 35cd820 (this report's commit follows it).
