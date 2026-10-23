# Slide Review of Basics, Advanced, and the Mathematics Appendix

Date: 2026-10-23. Source: `main` at `732c1431`; decks built 2026-09-28,
whose slide blocks match the current sources.
Standard: `docs/slide-guide.md`. Proposed outlines for the weakest deck of
each chapter: `slide-outlines-2026-10.md`.

## Method

Eight reviews screenshotted every PyTorch deck in scope (146 decks, about
2,240 slides), read every `<!-- slides -->` block, and measured overflow with
a browser check. They compared the decks with the MLSS 2026 deck for content:
whether a lecturer could teach the section from the deck alone.

## Main Findings

1. **No deck is built to teach on its own.** None of the 146 decks states
   what the student will be able to do or what it skips. There are almost no
   check or estimate slides. Most "Motivation" slides list topics. Several
   recaps introduce content that no slide showed.
2. **No running examples.** Numbers are recomputed in every deck on a new toy
   problem and never carried through a chapter. Good examples already exist
   in the text (the coin, FrozenLake, GPT-2 small, the three-mode mixture, the
   $\kappa=20$ valley) but appear on few slides.
3. **Code instead of evidence.** In ch. 6–8, 52% of slides are code; across
   the book several hundred slides show more than ten lines, up to 71, often
   imports, seeds, helpers, or plotting code. The one slide where code is the
   idea (a loss, an update, a sampler) is often buried in a wall.
4. **About 15% of slides overflow or clip,** often hiding the very claim of
   the slide: the decode-cost plot in `ssm`, the bf16 result in
   `memory-precision`, the TD update line in `qlearning`, the saddle figure in
   `optimization-intro`. About 30 figures render too small to read.
5. **Titles are labels.** Between 11% and 32% of titles state a claim,
   depending on the chapter ("Training", "Results", "Setup").
6. **Book markup leaks onto slides.** `tools/gen_slides.py` translates
   directives only in titles, so `:numref:`, `:eqref:`, and `:cite:` print as
   literal text in slide bodies: 99 occurrences in the two RL chapters alone,
   in at least 21 decks overall. Hardcoded section numbers and "exercise N"
   references are often stale.
7. **The north-star decks look good but teach the API.** The Preliminaries
   decks (`ndarray`, `linear-algebra`, `calculus`) have clean diagrams but say
   how far more often than why, with no running example and no estimates.

## Chapters

Verdicts run from 1 (fine) to 5 (rewrite); the column gives the range and the
weakest deck, whose proposed outline is in `slide-outlines-2026-10.md`.

| Ch. | Decks | Slides | Verdicts | Weakest deck | Main problem |
|---|--:|--:|---|---|---|
| 1 Preliminaries | 7 | 131 | 2–3 | `linear-algebra` | API tour; ~6 slides repeat `ndarray`; no link to layers or cost |
| 2 Linear Regression | 7 | 112 | 2–4 | `oo-design` + `synthetic` + `scratch` | four plumbing decks for one lecture |
| 3 Linear Classification | 7 | 135 | 2–4 | `environment-and-distribution-shift` | 14 of 19 content slides text only |
| 4 Multilayer Perceptron | 7 | 134 | 2–4 | `generalization-deep` | survey prose, no measured evidence |
| 5 Computation | 0 | 0 | — | — | no decks; two lectures proposed |
| 6 CNNs | 6 | 75 | 2–4 | `lenet` | accuracy never stated; MLP comparison asserted |
| 7 Modern ConvNets | 8 | 108 | 2–4 | `batch-norm` | no axes diagram; claim contradicts the stored runs |
| 8 Sequence Models | 7 | 83 | 3–4 | `rnn` | imports slide, no shapes, stale references |
| 9 Optimization | 12 | 146 | 2–5 | `minibatch-sgd` | four setup slides; claims never computed |
| 10 Attention | 6 | 81 | 2–3 | `attention-scoring` | topic titles, code walls, no shape diagram |
| 11 Transformers | 7 | 94 | 2–4 | `gpt` | 55-line class slide; no model diagram |
| 12 State Space Models | 7 | 112 | 2–5 | `ssm` | 18 code cells; decode-cost plot clipped |
| 13 Computational Performance | 7 | 62 | 2–5 | `memory-precision` | four of six slides are clipped code |
| 14 Reinforcement Learning | 7 | 71 | 3–4 | `qlearning` | the update line is clipped off its slide |
| 15 Deep RL | 7 | 83 | 3–5 | `sac` | no diagram; seven hardcoded section numbers |
| 16 GANs | 7 | 82 | 3–4 | `dcgan` | no architecture diagram; 4 overflows |
| 17 Diffusion Models | 11 | 98 | 2–4 | `ddpm` | four equation-only slides; undefined symbols |
| 24 Linear Algebra | 3 | 77 | 2–4 | `geometry-linear-algebraic-ops` | tours five topics without a problem |
| 25 Calculus | 4 | 103 | 2–4 | `multivariable` | backprop repeated from the next deck |
| 26 Optimization | 5 | 118 | 2–4 | `adaptive-stochastic-methods` | opens mid-proof with undefined symbols |
| 27 Probability and Statistics | 7 | 164 | 2–4 | `concentration-generalization` | book text pasted; one figure in 27 slides |
| 28 Information Theory | 3 | 71 | 3–4 | `divergences-distances` | eight slides of printed checks |
| 29 Dynamics | 4 | 96 | 3–4 | `sdes` | code walls hide their outputs |

Strong decks to keep as models: `probability` (ch. 1; the HIV example),
`efficient-convnets` (ch. 7), `batch-size` and `scaling` (ch. 9), `hybrids`
(ch. 12; costs with units), `hardware` (ch. 13), `langevin` (ch. 17),
`gradient-based-optimization` and `numerical-stability-conditioning` (ch. 26;
"The math is right; the loss is NaN").

## Pipeline Fixes

These change every deck and should come first.

1. **Resolve directives in slide bodies** in `tools/gen_slides.py`:
   `:numref:` to the section's name, `:cite:` to a footer source, `:eqref:` to
   a displayed equation or nothing. Exercise references already render
   (`:exref:`, added 2026-10-23).
2. **An overflow gate** in `make slides` (the reviews' browser check), and a
   minimum rendered width for plots.
3. **Callout size.** Callouts render at 0.74em (`_d2l-slides.scss`, near line
   490), so the takeaway is the smallest text on the slide.
4. **`r-stretch`** shrinks figures that are followed by text and lets others
   overflow; one rule for both.
5. **Small defects:** covers in ch. 16–17 print the title twice; a fragment
   marker inside a `.col` prints as ". . ."; a `\"` inside `title=` truncates
   a title; inline math such as `$\beta < $` breaks.

## Decisions Needed

- **Granularity.** One deck per section (most reviews), or one deck per
  lecture that merges thin sections (the Basics review). Merging needs
  `gen_slides.py` to build a deck from several files, or the slides of thin
  sections moved into the file of the idea they serve.
- **Chapter opener decks** from each `index.md`.
- **Running examples,** per `docs/slide-guide.md` §5.

## Order of Work

1. Pipeline fixes 1–5 above.
2. The opener and closer template on every deck, with claim titles. This
   makes every deck usable as a lecture frame, even before its body is
   rebuilt.
3. Rebuild the weakest deck of each chapter to its outline, together with
   that chapter's text, code, and figure revision.
