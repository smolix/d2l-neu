# Chapter 5: Computation

Sources: `chapter_builders-guide/` (index and eight sections). Reviewed 2026-11-09
at `201f2cb6`. Two reviewers read the chapter in full (sections 5.1 to 5.4 and
5.5 to 5.8); this file combines their chapter-level observations. Detail and
finding IDs are in the per-notebook files:

- [index](ch05-computation/index.md)
- [model-construction](ch05-computation/model-construction.md) (MC)
- [parameters-state-memory](ch05-computation/parameters-state-memory.md) (PSM)
- [init](ch05-computation/init.md) (INI)
- [custom-layers](ch05-computation/custom-layers.md) (CL)
- [numerics](ch05-computation/numerics.md) (NUM)
- [saving-loading](ch05-computation/saving-loading.md) (SAV)
- [gpus-devices-memory](ch05-computation/gpus-devices-memory.md) (GPU)
- [reproducibility-inspection](ch05-computation/reproducibility-inspection.md) (REP)

## Verdict

This is a better chapter than a chapter of "software craft with no new model"
has any right to be, and it is the one place in Basics where the book has a
method of its own: predict a number from a short argument, then measure it.
Where the chapter does that (128 GB for the 8-billion-parameter model, the
$2^{16}$ blow-up of an uncorrected residual stack and its $\sqrt{e}$ bound,
variance growing by 1.5 per block, the safetensors header read by hand, the
jump after resuming with a fresh Adam explained from the update equation) it
reads like a textbook and is a pleasure. Where it does not, it reads like
framework documentation: device naming, type promotion rules, generator
taxonomies, registration demos, and tab blocks that open "X has no ...". The
split is roughly 60 to 40. No section is weak throughout, and none is at the
level of the model texts throughout. The single most important fix is that
several sections teach a remedy without ever showing the failure it cures.

## Scores

Scale: 5 = at home in Cover and Thomas or MacKay; 4 = good textbook prose with
local flaws; 3 = competent but ordinary, or good with a structural problem.

| Section | Explanation | Thought | Pedagogy | Language | Enjoyment |
|---|---|---|---|---|---|
| index | 4 | 4 | 4 | 4 | 4 |
| 5.1 model-construction | 4 | 3 | 4 | 4 | 3 |
| 5.2 parameters-state-memory | 4 | 3 | 3 | 4 | 4 |
| 5.3 init | 4 | 4 | 3 | 4 | 4 |
| 5.4 custom-layers | 4 | 4 | 4 | 4 | 4 |
| 5.5 numerics | 4 | 4 | 3 | 4 | 4 |
| 5.6 saving-loading | 4 | 4 | 4 | 4 | 4 |
| 5.7 gpus-devices-memory | 4 | 3 | 3 | 3 | 3 |
| 5.8 reproducibility-inspection | 3 | 3 | 3 | 3 | 3 |

## Is the length earned?

The chapter is the longest of the part: about 49,000 words of prose in the
source. That figure overstates what any one reader meets, because much of it
is per-framework tab prose. A PyTorch reader sees roughly 2,300 to 3,300 words
per section, about 23,000 in all, which is within the writing guide's budget
per section and still makes this one of the longest chapters of the part for
its reader.

Both reviewers reached the same answer independently. A quarter to a third
can go with no loss, and all of it comes from the documentation side. Halving
the chapter would begin to remove the worked experiments, which are what the
chapter is worth reading for. The cuts they agree on:

- 5.1: the "Access by Path" cells; three of the four registration cells.
- 5.2: the meta-device subsection (October asked for this too); the freeze
  side-effect demo; EMA; the mixed-precision preview paragraph (PSM-5).
- 5.3: the custom distribution and direct-edit toys carried over from the
  first edition; half of lazy initialization.
- 5.5: the TF32 subsection, the fp8 row, the detail of promotion rules.
- 5.6: freezing (already in 5.2).
- 5.7: the device-object tour, pinned memory, the unused `Benchmark` class.
- 5.8: the worker-initializer cell and the backward-hook signature prose.

The cost of tabs is larger here than anywhere else in the part. Sections 5.1
and 5.2 have 18 to 19 tab splits each, many of them one sentence naming an
API; the second half has 216 tab blocks, 31 of which open "X has no ...".
Identical paragraphs are repeated inside each tab (NUM-7, GPU-16), and a
sentence is split across code cells (GPU-4). For the reader of one framework
this produces a stop-and-go rhythm that the other chapters do not have. The
chapter would read better if the shared idea were always stated once in the
common prose and each tab held only code plus, at most, one sentence that is
true of that framework alone.

## Arc and order

The idea that every operation (optimizer, initializer, checkpoint, device
move) is a walk over a tree of modules holds 5.1 to 5.4 together, and each
section hands off to the next. The index states this well. The second half
has a second thread, the 8-billion-parameter model whose memory the reader
budgets, which appears in most section openings. Three things weaken the arc:

1. **The roadmap is flat.** The index lists eight "asks" sentences. The
   chapter has three movements: build a model (5.1 to 5.4), choose a number
   format (5.5), run it (5.6 to 5.8). Saying so would tell the reader why
   initialization, a mathematical sequel to chapter 4, sits inside a software
   chapter.
2. **The running example is never introduced and never closed.** The 8B model
   is used throughout but does not appear in the index, and the question that
   opens 5.7, "Will a given model fit?", is answered "no" and dropped (GPU-3).
   A closing paragraph that settles the budget would give the chapter an end.
3. **The chapter stops on a section summary.** There is no closing synthesis
   and no hand-off to convolutional networks. A capstone in 5.8 that uses a
   seed, a checkpoint and the NaN finder together on one small failure would
   close the chapter and tie its second half to its first.

Section order is sound. Two placements deserve a second look: hooks in 5.8
repeat the measurement of 5.3 and may belong there; and the last third of 5.2
(tying, freezing, EMA) depends on language models, embeddings, fine-tuning,
batch normalization and Adam, none of which the reader has met (PSM-9 to
PSM-11). Keep tying, with its gradient check; move or cut the rest.

## Most important findings

1. **Remedies without failures; claims without runs.** This is the pattern
   that most separates the chapter from the bar.
   - `init` never trains a model, yet says "Initial activations this large
     make optimization impractical" (INI structural).
   - The straight-through estimator is checked for its gradient but never
     used to learn anything (CL-8).
   - The partly registered model, ranked the worst case, is introduced with
     "Suppose" and never run (MC-3).
   - Numerics never trains with bf16 weights, never times mixed precision
     ("We did not time the runs"), and never shows loss scaling rescuing a
     run (NUM-8, NUM-9, NUM-10).
   - Reproducibility describes data-loader worker failures and GPU
     nondeterminism, but its cells use `num_workers=0` and print an error
     message (REP-5, REP-7).
   - "Resumes exactly" is promised and not tested against an uninterrupted
     run (SAV-7).
2. **The reproducibility section does not answer its own question.** It
   opens by asking whether a 0.3-point gain is real, and concludes about
   seed-to-seed spread, without ever measuring that spread (REP-1). Its final
   gradient cell is left uninterpreted (REP-13). This is the weakest section
   and also the cheapest to repair: five seeds of one small model.
3. **Prose that narrates other frameworks' outputs.** In "Accumulation in
   fp32" the shared paragraph explains a left-to-right sum that "stalls at 4"
   and an fp32 total "about 2% low". The stored PyTorch output is `9784` and
   `10004.0439`; only the MXNet tab shows the behaviour described. The
   paragraph hedges across four summation orders instead of showing one
   effect (NUM-12, NUM-13). Use an explicit loop, which behaves the same in
   every framework, and an increment that fp16 stores exactly. See also
   SAV-10.
4. **`parameters-state-memory` and `gpus-devices-memory` are several
   sections each.** The first is three (PSM-9 to PSM-11); the second is four,
   with a twelve-cell API tour in "Devices and Placement" (GPU-9 and its
   structural findings).
5. **`init` peaks in the middle.** The residual-network derivation, the best
   passage of the first half, is followed by first-edition toys: std 0.01,
   the constant 42, the $\pm[5, 10]$ distribution, `+= 1` (INI-4, INI-11).
   Reorder so that the residual result ends the section, and demonstrate
   `apply` with He initialization.
6. **The PyTorch reader is promised an example only other tabs deliver.**
   "Forward Computation" announces a data-dependent loop that exists in the
   TensorFlow and JAX tabs only (MC-5).
7. **The same thing is taught several times.** Registration and buffers three
   times (the `alpha` constant, `Whitener`, `RandomFeatures`; MC-7, CL-5);
   the $16N$ memory accounting five times across 5.2, 5.5, 5.6 and 5.7;
   "Training Memory" as a heading in both 5.2 and 5.7; freezing in 5.2 and
   5.6; `ResidualBlock` redefined in each section.

## Recurring patterns

- **Terminology drift.** "body" (5.1, 5.2, code) against "branch" (5.3);
  $\alpha$, $g$ and $\mathbf{g}$ for scales; Adam's state as "running
  averages", "moment estimates" and "moments"; "kernel" for a GPU routine, a
  weight matrix and, one chapter later, a convolution filter; coined
  "specimen" and "treatments".
- **Forward-defined material.** Tokens, embeddings, GPT-2, Llama, fine-tuning
  and Adam appear before the reader has met them. October's finding is milder
  now but not closed.
- **Cells that print `True`.** Several demonstrations show that nothing
  happened. A check that can only succeed teaches less than a number that
  could have come out otherwise.
- **Helpers defined for later chapters.** `Benchmark` and others are saved
  here and not used here.
- **Resource lists** still carry em dashes and an older, chattier voice than
  the body.

## What to keep as the model

"Initialization of Residual Networks" in `init`, "Activation Statistics" in
`reproducibility-inspection`, the `Tripwire` pickle and the hand-read
safetensors header in `saving-loading`, "Layers in a Plain List" in
`model-construction`, and the RMSNorm example in `custom-layers`. Each poses a
concrete question, predicts the answer in a few lines, and lets the notebook
confirm or surprise. The rest of the chapter, and the rest of the part,
should be written this way.

## Not checked

Only PyTorch code and stored outputs were compared with the prose; the other
tabs were read but not verified. Rendered HTML pages were not inspected;
figures were rendered from the SVG sources.
