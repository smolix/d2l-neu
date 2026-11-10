# Adam and AdamW

Source: `chapter_optimization/adam.md` (lines 1 to 1968, the slides after
the marker not read). About 8,100 words of prose including 1,400 of
exercises, and about 800 lines of code (both tabs). Stored outputs checked:
`outputs/pytorch` and `outputs/jax`, `chapter_optimization/adam.json`, with
the loss curves and both decay grids rendered.

## Verdict

The first 380 lines are close to the bar: AdaGrad, RMSProp and Adam are
built one from the other, each motivated by a defect of the one before, with
numbers the reader can check and several sentences worth keeping. After that
the file is three sections in one. It builds the chapter's test harness (two
models, a dataset, six saved utilities, 370 lines of code) between the
definition of Adam and the first test of it; it then teaches AdamW, weight
decay at scale, decay exemptions and optimizer memory, which is a second
subject with its own label (`sec_adamw`) that a dozen other files already
cite as a section. Until the "book-pass revision" commit AdamW was a file of
its own, and it should be again. The deeper problem is in the experiments.
**The comparison that the section is organized around does not show what
the section is about**: on both test models tuned Adam beats tuned SGD by
the same modest margin, the text says so ("On our two small models Adam's
speedup is modest and of similar size"), and the explanation of why Adam
wins on transformers is then carried by three papers retold. The decay grid
has the same fault in a milder form: its headline conclusion rests on
minima that move by one grid step, in a direction that differs between the
two tabs. Fixing the experiments, or cutting them down to what they show,
matters more than any prose edit.

## Scores

| Dimension | Score | Note |
|---|---|---|
| Clarity of explanation | 4 | AdaGrad to Adam and the $\epsilon$ paragraph are very clear; the scale invariance that explains Adam's learning rate is never stated |
| Clarity of thought and focus | 3 | Three subjects in one file; two central experiments are read for more than they show; $\lambda$ is given two opposite meanings in adjacent subsections |
| Pedagogy | 3 | Good ladder of methods, good from-scratch code; a 370-line harness in the middle; papers retold where an experiment should be |
| Language | 4 | Plain and exact, almost free of mannerisms; one object has eight names |
| Ease and enjoyment | 3 | The first quarter is a pleasure; the reader then wades through infrastructure and hedged results |

## What works

- **The motivation of AdaGrad.** "This is the normal state of affairs in
  language: we will see the word *learning* far more often than the word
  *preconditioning*", and "One clock cannot serve both populations." A
  concrete case, a small joke that carries the point, and a sentence a reader
  will remember. Protect both.
- **Each method is introduced by the failure of the previous one, with a
  number.** AdaGrad's first step is $\eta$ in every coordinate and twenty
  steps cover "at most $\eta \sum_{t=1}^{20} t^{-1/2} \approx 7.6\,\eta =
  3.0$", which "is not enough to cover the distance of $5$" (checked: the sum
  is 7.60, and the stored run moves $x_1$ by 2.62). RMSProp's first step is
  $3.2\,\eta$ because the average starts at zero. Adam's correction removes
  that. This is how a family of methods should be taught (guide 4.3).
- **The bias correction is explained, not asserted**: the weights total
  $1 - \beta_2^t$, "after ten steps $\mathbf{v}_t$ holds about $1\%$ of its
  stationary value" (checked), and the from-scratch check prints `first step
  / lr: 1.0000 to 1.0000`. That cell is a check on an implementation with a
  second purpose, and it is worth its place.
- **The $\epsilon$ paragraph** in "Limits of the Variance Estimate". The
  ceiling $\eta/\epsilon$ on a coordinate's rate, and the limit in which
  Adam becomes SGD with momentum at rate $(1-\beta_1)\eta/\epsilon$, end in
  a judgment: "which makes $\epsilon$ a tuning knob rather than a numerical
  detail." This is the voice the guide asks for.
- **The sign-descent identity** is derived in one line from the update
  ($\beta_1 = \beta_2 = 0$, $\epsilon = 0$) and gives the reader the right
  mental model: "Adam is a smoothed sign method."
- **The collapse of coupled decay** (cell `adamw-one-number-two-meanings`)
  is the most striking experiment of the section: the same $\lambda = 0.1$
  leaves AdamW indistinguishable from no decay and sends Adam with an
  $\ell_2$ penalty to a loss of 3.15, and the next cell shows that this is
  the entropy of the byte frequencies, 3.17. Reading a loss level against a
  computed baseline is good practice.
- **"Penalizing the loss *is* shrinking the weights"**, with the one-line
  identity :eqref:`eq_l2-is-decay` before the Adam case. Short and right.
- **Protocol stated before results**: same initialization, constant rate,
  four rates per method, "Below, 'tuned' means the best of these four
  rates", and the citation of Schmidt et al. on tuning effort. The honesty
  is real; the trouble is what the protocol then yields (ADAM-12).
- **Use of the appendix follows decision 3.** The equations of AdaGrad,
  RMSProp, Adam and the coupled-versus-decoupled display appear in both
  places, all small. The regret bound, the Reddi construction and the
  fixed-point proposition are in the appendix only. All four pointers
  (`subsec_mdl-per-coordinate` twice, `subsec_mdl-adam-counterexample`,
  `subsec_mdl-decoupled-weight-decay`) lead to passages that hold what the
  text says they hold.

## Structural findings

### The file is three sections

Prose words and code lines by part (before the slides):

| Part | Lines | Prose | Code lines |
|---|---|---|---|
| Introduction, "Per-Coordinate Learning Rates" | 1 to 382 | 1,600 | 136 |
| "TinyLM and FashionCNN" | 383 to 942 | 1,080 | 369 |
| "Strengths and Limits of Adam" | 943 to 1253 | 1,450 | 126 |
| "Decoupled Weight Decay", "Optimizer-State Memory" | 1254 to 1764 | 2,330 | 165 |
| Summary, exercises | 1765 to 1968 | 1,620 | 0 |

Recommended split, with the seams:

1. **Adam** (`sec_adam`): the introduction, "Per-Coordinate Learning Rates",
   "Strengths and Limits of Adam", its own summary, exercises 1 to 6. About
   3,700 words of prose. Its one question: what do per-coordinate rates
   achieve, and how does Adam set them?
2. **AdamW and weight decay** (`sec_adamw`): seam at the heading "##
   Decoupled Weight Decay" (line 1254). It takes "Decoupled Weight Decay"
   with its five subsections, a shortened memory paragraph, and exercises 7
   to 10. About 3,000 words. Its one question: what should weight decay
   mean under a per-coordinate method, and how is it set? The label
   `sec_adamw` already sits on that heading and is cited as a section from
   `weight-decay.md`, `muon.md`, `scaling.md`, `practice.md`, `lstm.md`,
   the chapter introduction and the Language Models skeletons; at present a
   `sec_` label resolves to a subsection. The subject was a separate file
   (`adamw.md`) until the commit "Ch. 9 Optimization: book-pass revision";
   the cell IDs still begin `adamw-`.
3. **The test harness**: seam at "## TinyLM and FashionCNN" (line 383) to
   the end of `evaluate_accuracy` (line 942), plus `lr_sweep`. It defines
   `WikiText103`, `TinyLM`, `FashionCNN`, `train_steps`, `lm_perplexity`,
   `evaluate_accuracy`, `final_loss`, `smooth` and `lr_sweep`, nearly all
   `#@save`, and `subsec_tinylm` is cited from chapters 10, 11 and 12 and
   from four later sections of this chapter. A reader who looks up "the
   `TinyLM` of :numref:`subsec_tinylm`" from the attention chapter lands in
   the middle of the Adam section. Give it a short section of its own ("Two
   Test Models", or similar): what the models are, why a transformer
   appears two chapters early, the first training run, and the parameter
   groups. Two placements are possible. Directly before Adam keeps the
   current order. Before Momentum would also let `momentum.md` test
   momentum on a network, which it cannot do now (see the review of
   `momentum.md`, MOM-9). This is an author decision.

As it stands the reader meets Adam's update at line 226 and the first
evidence about Adam at line 1036, about ten screens and 370 lines of code
later.

### The comparison does not show the phenomenon

See ADAM-12. The introduction promises "To see where its advantage comes
from, we build two small models ... and compare Adam with tuned SGD with
momentum on both." What the stored runs show: Adam ahead by 0.06 nats on the
language model and by 0.05 on the CNN, reaching SGD's final loss at step
1,275 of 2,000 on both (PyTorch). The two models do not differ. The section
then explains a difference between transformers and CNNs that its own
experiment did not produce, by retelling Kunstner et al. (2023), Kunstner et
al. (2024) and Zhang et al. (2024), and closes with "How much each finding
explains for `TinyLM` is open". Guide section 9 is explicit: when an
experiment does not show what the prose says, "fix the experiment or remove
it. Do not keep it and explain what it fails to show."

### Missing

- **The scale invariance of Adam's update** and what it says about the
  learning rate (ADAM-6). The appendix calls it "the central result of this
  section"; the main text has the ingredients and never states it.
- **That a normalized step does not shrink with the gradient**, so Adam at
  a constant $\eta$ does not settle (ADAM-4). The Summary's last sentence
  ("What Adam does not remove is the choice of $\eta$ over time") arrives
  without this.
- **Adam on the valley.** The running example is used for AdaGrad and
  RMSProp and dropped when the title method arrives.

### Superfluous or misplaced

- The CNN comparison (about 60 lines of code, 300 words) yields the same
  result as the language-model comparison and could be a three-row table
  and a paragraph.
- "Weight Decay in Large-Scale Training" is 500 words of results from three
  papers with no experiment (ADAM-19).
- "Optimizer-State Memory" repeats a worked example that
  `training-systems.md` also carries, and describes a memory layout that
  chapter 13 does not use (ADAM-21).

## Passage-level findings

**ADAM-1** (minor, thought; repetition). Introduction. Anchor: "twenty steps
take $x_1$ from $-5$". The valley's stability limit, the factor 0.92 and the
value $-0.94$ were derived in `gd.md` and again, with the same numbers, at
the top of `momentum.md` one section earlier. Guide 7.7: a recurring number
is derived once. One clause with a reference is enough here ("a single rate
is capped by the steepest direction, :numref:`sec_momentum`"), and the new
content, "the inverse of that coordinate's curvature, a single step would
reach the minimum", can then lead the paragraph.

**ADAM-2** (moderate, pedagogy). "AdaGrad". Anchor: "Because the scaling
adapts, we can compensate with a learning rate at which". The cell that
follows (`adam-per-coordinate-learning-rates-2`, $\eta = 2$) is the only
cell in the section whose output is never read: the next line is the heading
"RMSProp". The stored output is $x_1 = -0.002$ after 20 steps. Read it, and
say what it teaches: under AdaGrad the learning rate is a travel budget
(about $7.6\,\eta$ in 20 steps), so it must be chosen from the distance to
cover, not from the curvature.

**ADAM-3** (moderate, correctness; confident, recomputed). "RMSProp".
Anchor: "RMSProp gets close to the minimum where AdaGrad stalled". True at
step 20, the horizon of `train_2d`, and misleading beyond it. I reran the
cell's rule in NumPy for 200 steps: RMSProp reaches $10^{-6}$ around step
30, then leaves again; by step 100 both coordinates are back at 0.1 to 0.25
from the minimum, and the run ends oscillating at
$|x_i| \approx 0.2 = \eta/2$ without settling. AdaGrad at the
same $\eta = 0.4$, which "stalls", is at 0.05 after 200 steps and converges.
On this problem the 20-step picture reverses the long-run facts. The cause
is the property the section needs later: a step normalized by the recent
gradient size has length about $\eta$ however small the gradient is, so at a
constant $\eta$ the iterate cannot settle closer than about $\eta$. Fix:
show 200 steps (or print step 20 and step 200), state the property here,
and let it motivate both "Adam is a smoothed sign method" and the hand-off
to schedules. AdaGrad's "stalls far from the minimum" should become "is
still far from the minimum after 20 steps".

**ADAM-4** (minor, correctness). Anchors: "ten at the customary $\beta_2 =
0.9$" and "the same overshoot as RMSProp's". The value 0.9 is Hinton's
lecture value and the Optax and Keras default; PyTorch's `RMSprop` defaults
to 0.99. Say "at the original $\beta_2 = 0.9$". And the two overshoots agree
only because $0.1/\sqrt{0.001} = 1/\sqrt{0.1}$ for the particular defaults;
"the same" suggests a reason. Write "about as large as".

**ADAM-5** (moderate, explanation). "Adam". Anchor: "Each coordinate thus
moves by about $\eta$ times the ratio of its average". This sentence is one
step from the result that makes Adam usable and stops. Add it; it is small,
so by decision 3 it belongs in both the main text and the appendix. Draft:
"Multiply the loss by 1,000 and Adam takes the same steps:
$\hat{\mathbf{m}}_t$ and $\sqrt{\hat{\mathbf{v}}_t}$ both grow by the
factor, and their ratio does not change. The same holds for each coordinate
separately. The learning rate of Adam is therefore a distance in parameter
space, at most about $\eta$ per coordinate and step, and not a multiplier
of the gradient. That is why one default, $10^{-3}$, serves models whose
gradients differ by orders of magnitude, and why the good rates in the
comparisons below are a hundred times smaller than SGD's." The last clause
answers a question every reader has at the two `lr_sweep` grids (SGD: 0.03
to 1; Adam: 0.0003 to 0.01), which the text never addresses.

**ADAM-6** (minor, language). Anchor: "are exact rather than". "Exact" and
"exactly" are used six times for the bias correction (also "Adam corrects
this startup error exactly", "corrects both averages exactly" in the
Summary). The claim that follows is narrower: unbiased in expectation when
the squared gradients have a steady mean. And the step
$\hat{m}/\sqrt{\hat{v}}$ stays biased, as the appendix points out. Say once
what is true: "The corrections are not a heuristic. If the squared
gradients have a steady mean, dividing by $1 - \beta_2^t$ removes the
deficit at every $t$."

**ADAM-7** (moderate, language; terminology). $\hat{\mathbf{v}}_t$ and the
division by its root go under eight names: "scale", "second-moment
estimate", "scale estimate", "typical gradient size", "variance estimate",
"denominator", "preconditioner", "per-coordinate scaling". Anchor for the
worst case: "Limits of the Variance Estimate" (heading and label
`subsec_adam-variance-estimate`). $\mathbf{v}_t$ averages squared gradients,
so it estimates the second moment, which includes the squared mean; it is
not a variance, and the distinction matters for the sign-descent reading
(when the mean dominates, the ratio is $\pm 1$). Choose "second-moment
estimate" for $\mathbf{v}$ and "scale" for its root, and use "preconditioner"
only where the valley picture is meant.

**ADAM-8** (major, pedagogy; structure). "TinyLM and FashionCNN". Anchor:
"A comparison of optimizers needs models on which the optimizers behave".
See the first structural finding. Within the part: (a) the transformer code
cannot be read by this reader (query, key and value projections, head
reshapes, `scaled_dot_product_attention` with `is_causal`); the text says so
honestly ("an optimizer does not need to know what attention does"), which
is a reason to keep the model out of the argument's way in a section of its
own; (b) five tab notes carry framework plumbing (default initializers,
`approximate=False`, `d2l.Embed` and XLA's gather), which guide section 9
sends to code comments; (c) "Three helpers read a run" and the two
evaluation functions are introduced before anything needs them. In a
harness section they can be listed in a small table (name, what it returns,
first use).

**ADAM-9** (minor, correctness; confident). "The Language Model". Anchor:
"On the 100,000 characters of *The Time". The Basics part gives the length
as "some 170,000" characters (`text-sequence.md`; the slide there says
173,428). With 8.2 million predictions, each character would be a target
about 47 times, not 80. The argument is unaffected.

**ADAM-10** (minor, explanation). "Training for a Fixed Number of Steps".
Anchor: "Positions labeled $-100$ add no loss". The reader of this section
has no use for masked labels; the feature serves the masked language model
of chapter 11. In a harness section say that in a clause; here it reads as
an unexplained magic number.

**ADAM-11** (moderate, correctness; not checked). "Sources of the Gap".
Anchor: "Every lowercase letter appears more than twice in an average
minibatch". This sentence and the next two give four statistics of the byte
frequencies (more than twice per minibatch; "About a hundred byte values";
"less than once in ten minibatches"; "less than 0.1% of the text"). No cell
prints any of them. The frequency vector `freq` is computed 250 lines later
for another purpose. I could not check them (the data is not on this
machine). Compute `freq` here and print the four numbers, or reduce the
passage to the one claim the argument needs (rare bytes carry a negligible
share of the loss).

**ADAM-12** (major, pedagogy and thought). "Comparison on the Language
Model" to "Sources of the Gap". Anchor: "On our two small models Adam's
speedup is modest and of similar size." Stored results (PyTorch; JAX within
0.02): language model, SGD 1.654 at $\eta = 0.3$ with $\eta = 1$ divergent,
Adam 1.592, gap 0.062 nats, crossing at step 1,275; CNN, SGD 0.175, Adam
0.128, gap 0.047, crossing at step 1,275. The prose gives the run-to-run
variation as "a few hundredths of a nat", with SGD "up to about a tenth of a
nat higher". So:

- the gap on the language model is one to three times the noise, and the
  section's quantitative claim ("about 1.3 to 2.5 times as fast") is as wide
  as that implies;
- the two models, chosen to contrast, give the same answer;
- SGD's best rate is the last grid point before divergence, on a grid with
  factors of three, so its true optimum is somewhere in $[0.3, 1)$ and has
  not been located. The prose reads this as a finding ("No learning rate in
  SGD's grid closes the gap: the best one is the last stable one"); it is
  also an untuned baseline at the one place where tuning matters.

Direction, for the development machine:

1. Make the experiment show a mechanism instead of a small gap. Exercise 3
   already describes the experiment the body needs: the loss on frequent and
   on rare tokens under each optimizer, with a 1,024-token BPE vocabulary so
   that rare tokens carry a real share of the text. If SGD stalls on the
   rare group and Adam does not, that one plot replaces the second and third
   paragraphs of "Sources of the Gap". A second, cheap arm is sign descent
   with momentum next to Adam and SGD: if it tracks Adam, the sentence "Adam
   is a smoothed sign method" has evidence on this model.
2. Add one or two rates between 0.3 and 1 to SGD's grid, so that "tuned"
   holds at the edge.
3. Reduce the CNN comparison to a table row and a paragraph unless it can be
   made to differ from the language model.
4. If no small experiment separates the two architectures, say in two
   sentences that the decisive gap appears at scale, cite the three papers
   in one paragraph, and do not present the comparison as an explanation.

**ADAM-13** (moderate, thought; repetition). The same two ranges are stated
three and four times: "about 1.3 to 2.5 times as fast" (language model, CNN,
Summary) and "about 0.03 to 0.2" nats with "3 to 20%" (language model, CNN
paragraph twice, Summary). The three paragraphs after the CNN run ("Two
things repeat from the language model.", "Both losses are cross-entropies in
nats per prediction", "Measured in time, the question our protocol asks")
spend 300 words to say that the CNN result equals the language-model
result. One paragraph.

**ADAM-14** (minor, correctness). Anchor: "Third, they come from the
architecture." The first two findings carry citations; the third, the
heterogeneity of curvature across the blocks of a transformer and its
absence in CNNs, is the result of Zhang et al. (2024), which is cited only
in the lead paragraph for a different claim. Cite it here. I believe the
attribution is right (the paper compares block-wise Hessian spectra of
transformers and CNNs); the three attributions to Kunstner et al. match my
knowledge of those papers.

**ADAM-15** (minor, correctness). Anchor: "the better model assigns, on
average, $e^\Delta$ times as much". The factor $e^\Delta$ is the ratio of
geometric means of the probabilities, not of their averages. Write "in
geometric mean", or "per prediction, in the sense of the geometric mean".

**ADAM-16** (minor, language). "Decoupled Weight Decay". Anchor: "Adam is
almost always trained with weight decay". Models are trained; Adam is used.
Also in the Summary, "tuned Adam reaches tuned SGD's final training loss
about 1.3 to 2.5 times as fast" drops the definition that "tuned" means the
best of four rates on a grid; in a summary it reads as a general fact.

**ADAM-17** (moderate, correctness; medium confidence, not run). "Coupled
and Decoupled Decay Compared". Anchors: "As training makes progress the loss
gradients shrink" and "it has lost what it learned about the context in its
first hundred". Two problems with the reading of
`adamw-one-number-two-meanings`.
(a) The stored curve does not show a model that learned and then forgot.
The coupled run is half a nat above the others in the first 25-step block
(about 3.75 against 3.2, read off the stored plot), dips to 3.07 near step 50, and settles at 3.15 to 3.25.
Its best value is a tenth of a nat below the entropy of the byte
frequencies (3.17): it never learned much about the context.
(b) The explanation needs gradients that have shrunk by orders of magnitude
through training, but the damage is done in the first 50 steps. A simpler
account fits the numbers: the entries of the weight matrices start below
$1/\sqrt{128} \approx 0.09$ in magnitude; wherever $\lambda x$ exceeds the
loss gradient of a coordinate, Adam moves that coordinate toward zero by
about $\eta = 0.003$ per step, which erases it in at most 30 steps. That
would happen from the first step, for every coordinate whose loss gradient
is below $0.1\,|x|$, with no need for training to make progress. I cannot
confirm which account holds without running the model. Add a measurement
(the share of coordinates with $|\lambda x| > |g|$ at steps 1, 50 and 500,
or the norm of the matrices over the run) and write the explanation that it
supports. The text's next sentence, that a dominated coordinate "shrinks by
about $\eta$ per step whatever the value of $\lambda$", is right and is the
core of either account.

**ADAM-18** (major, thought). Same subsection, the grid. Anchor: "a column
for AdamW, a whole grid for the coupled form". The stored tables:

| | $\eta = 0.001$ | $\eta = 0.003$ | $\eta = 0.01$ |
|---|---|---|---|
| Coupled, best $\lambda$ (PyTorch) | $10^{-3}$ (2.09) | $3\times10^{-3}$ (2.35) | $10^{-3}$ (2.21) |
| Coupled, best $\lambda$ (JAX) | $10^{-3}$ (2.09) | $3\times10^{-3}$ (2.34) | $3\times10^{-3}$ (2.34) |
| AdamW, best $\lambda$ (both) | 3 | 3 | 3 |

- The coupled minimum moves by one grid step, not monotonically in $\eta$,
  and at $\eta = 0.01$ the two tabs disagree about where it is: the entry at
  $\lambda = 10^{-3}$ is 2.21 in PyTorch and 2.62 in JAX. A difference of
  0.4 nats between two runs of the same cell is larger than the effect the
  conclusion rests on (decision 5: no conclusion on a difference smaller
  than the noise).
- AdamW's minimum sits at $\lambda = 3$ on a four-point grid with steep
  sides (2.96 or more at $\lambda = 1$, 2.55 at $\lambda = 10$); "3 or 10"
  in the prose concedes that it is located to within a factor of three.
- The set-up needs a paragraph of confounders ("we therefore make the model
  overfit on purpose"), which guide section 9 names as the mark of a
  research experiment. The details are harsher than the prose says. The
  8,000 training windows overlap with stride one, so they cover about 8,060
  characters and each character is a target about 400 times ("the model sees
  each window about six times" is true and understates it); the 2,000
  held-out windows cover about 2,060 characters.
- The best decoupled $\lambda$ is 3, thirty times "the common choice
  $\lambda = 0.1$" of the previous page. A reader may take that home.
- The fixed-point paragraph ("A fixed point explains why the best $\lambda$
  of AdamW stays put here") cannot explain a difference between the two
  forms: the fixed point of Adam with an $\ell_2$ penalty is the minimizer
  of the penalized loss, which does not depend on $\eta$ either (appendix,
  "Fixed Points", part (a) of the proposition).
- The next subsection then says that at scale the opposite rule holds (keep
  $\eta\lambda$ fixed) and that Loshchilov and Hutter's own result is that
  rule in another parameterization.

A first-time reader leaves with three statements about how $\lambda$
relates to $\eta$ and no way to tell which applies to their run. Direction:
keep the three-curve experiment, which shows beyond doubt that one $\lambda$
means two things. Replace the grid by a demonstration that isolates the
mechanism, for instance the two-coordinate pure-noise run of
:numref:`subsec_mdl-decoupled-weight-decay` (15 lines of NumPy, decay rates
a factor of 100 apart under one $\lambda$; small, so duplicating it follows
decision 3), or cut the grid and cite the separability result. Then state
the two regimes once, in one place: where decay selects a solution,
$\lambda$ is the quantity to hold; where the decayed weights are
scale-invariant, $\eta\lambda$ is. Exercise 8 can keep the grid.

**ADAM-19** (moderate, thought; focus). "Weight Decay in Large-Scale
Training". Anchor: "Gradient noise makes the norms grow, decay makes them
shrink". The subsection reports Kosson et al., D'Angelo et al. and Bergsma
et al. in 500 words; nothing in it is derived or run, and its central
quantitative fact is left to exercise 9. That fact is two lines and should
be in the body: an update $\eta\mathbf{u}$ orthogonal to $\mathbf{x}$ with
$\|\mathbf{u}\|^2 \approx d$ gives
$\|\mathbf{x}_{t+1}\|^2 = (1-\eta\lambda)^2\|\mathbf{x}_t\|^2 + \eta^2 d$,
so the norm settles at $\|\mathbf{x}\|^2 \approx \eta d/(2\lambda)$ and the
angle per step at $\eta\sqrt{d}/\|\mathbf{x}\| = \sqrt{2\eta\lambda}$
(checked). With it, "the product $\eta\lambda$ sets the *effective* learning
rate" is a result and not a report. Three details:
(a) "Gradient noise makes the norms grow" is imprecise. Any update
orthogonal to the weight lengthens it, by Pythagoras; noise is not needed.
(b) "Small weights also keep low-precision training in a safe numerical
range." is a dangling sentence; give it a reference and a reason or cut it.
(c) The horizon $1/(\eta\lambda)$ is presented as the section's own and the
finding is credited to Bergsma et al. (2025). I believe the description of
that paper is right (the best timescale constant at a fixed ratio of tokens
to parameters, a power law otherwise) but could not confirm it; the
averaging-horizon view is, to my knowledge, due to Wang and Aitchison
(2024), who are not cited. Unverified.

**ADAM-20** (moderate, pedagogy). "Parameters Exempt from Decay". Anchor:
"has nothing to prevent and cannot help". The cell
`adamw-what-not-to-decay-2` trains for 2,000 steps to show that nothing
happens (stored perplexities 5.00 against 4.91 in PyTorch, 4.97 against 4.90
in JAX; "within a few percent ... in either direction" is a fair range).
Keep the cell that builds the two parameter groups, which is code the
reader will reuse, and replace the run by one sentence. The OLMo 2 anecdote
is a good concrete case; I believe it is accurate (that paper removed decay
from the embeddings for the stated reason) but could not check the wording.
Unverified.

**ADAM-21** (moderate, correctness; confident on the first point).
"Optimizer-State Memory". Anchor: "The forward and backward passes use bf16
weights and gradients, 2 bytes each". This is not the mixed-precision recipe
the book uses. `fast-transformer.md` states it: "The weights, their
gradients, and Adam's moment estimates stay in fp32", with the casts applied
per operation by `torch.autocast`. Under that recipe no bf16 copy of the
weights is stored: the 16 bytes are 4 + 4 + 8, the same as full fp32, and
the saving is in the activations. A stored half-precision model beside an
fp32 master copy describes other training stacks. The totals in the table
happen to agree (16), but the breakdown, the row labels and the 20-byte row
do not describe what the reader will run. Also, the 7-billion-parameter
ledger (112 GB) is worked a second time in `training-systems.md` with the
same numbers. Give the ledger one home (chapter 13 or the tools appendix),
make it agree with the recipe of chapter 13, and leave here one paragraph:
Adam's two buffers are 8 of the 16 bytes that each parameter costs in
training, which is why optimizer state is the first target when memory runs
out. The cell `adamw-optimizer-state-and-memory` multiplies three constants
and can go.

**ADAM-22** (minor, language; notation). $\mathbf{v}$ was the velocity in
`momentum.md` and is the second-moment average here, with $\mathbf{m}$ the
momentum; $\lambda$ was curvature there and is the decay coefficient here;
the weights are $\mathbf{x}$ here, $\mathbf{w}$ in
:numref:`sec_weight_decay` and in the appendix's decay section; the update
index moves from $\mathbf{x}_t \leftarrow \mathbf{x}_{t-1}$ to
$\mathbf{x}_{t+1} = \mathbf{x}_t$. Adam's letters are the literature's and
should stay, with one sentence where $\mathbf{m}$ and $\mathbf{v}$ are
introduced ("we follow the Adam paper: $\mathbf{m}$ is the momentum average,
and $\mathbf{v}$ no longer denotes a velocity").

## Code and experiments

- `adam-per-coordinate-learning-rates-2`: output never read (ADAM-2).
- `adam-rmsprop-forgetting-on-purpose-1`: read for more than 20 steps show
  (ADAM-3).
- `adam-adam-both-moments-debiased-2`, `adamw-adamw-from-scratch-2`:
  implementation checks that also show the first-step property. Both
  outputs match the prose in both tabs. Keep.
- `adam-a-tiny-language-model-6`: "about 5" and "about 2.3 bits" match (4.84
  and 4.88; $\log_2 4.84 = 2.27$).
- `adam-the-race-on-the-language-model-1` to `-3`,
  `adam-the-same-race-on-a-cnn-2` to `-4`: ranges in the prose contain the
  stored values. The comparison does not separate the two models
  (ADAM-12). At $\eta = 0.3$ the CNN run under SGD stays at 2.305 in
  PyTorch and trains to 0.325 in JAX; the prose covers both ("Often the
  loss stays at" the value of a uniform guess, "and when the run does
  train" it ends higher).
- `adamw-one-number-two-meanings`: striking and sound; the explanation of
  it is unsupported (ADAM-17).
- `adamw-a-grid-twice-2` to `-4`, `adam-coupled-and-decoupled-decay-compared-1`:
  conclusion rests on differences within the noise; tabs disagree at one
  entry by 0.4 nats (ADAM-18).
- `adamw-what-not-to-decay-2`: a run that shows nothing by design
  (ADAM-20).
- `adamw-optimizer-state-and-memory`: arithmetic on constants; layout
  disagrees with chapter 13 (ADAM-21).
- Numbers no cell prints: the four byte-frequency statistics (ADAM-11); the
  421,642 parameters of `FashionCNN` (correct by hand: 320 + 18,496 +
  401,536 + 1,290).
- No TensorFlow or MXNet material. Twelve tab notes; those on default
  initializers, `approximate=False`, `d2l.Embed` and image layout are
  plumbing that belongs in comments. The JAX tab builds AdamW as a chain,
  the PyTorch tab as a subclass; the shared sentence "shrink the weights,
  then take the Adam step" is true of both.
- Not checked: the JAX code was read but not executed; no PyTorch or JAX on
  this machine, so none of the network runs was reproduced.

## Exercises

Ten, about 1,400 words. The level is high and several are among the best in
the chapter: 4 (per-group learning rates for SGD) tests the block
explanation directly, 8 (the fixed-product ridge) and 9 (rotational
equilibrium) turn the two regimes of weight decay into computations, and 2
(bias correction by initialization) is a good short problem. Changes:

- **Exercise 3** is the experiment the body lacks. Promote it (ADAM-12) and
  leave a variation as the exercise.
- **Exercises 5 and 7 cannot be done without the appendix**: 5 extends the
  function `online` of :numref:`subsec_mdl-adam-counterexample`, 7 extends
  "The experiment of :numref:`subsec_mdl-decoupled-weight-decay`". The
  appendix is optional reading; either mark both as appendix exercises and
  move them there, or make them self-contained (the loss sequence of 5 is
  three lines). In 5, the parenthesis "The function `online` below takes the
  maximum" refers to a function that the exercise has not yet introduced.
  In 7, "starts two parameters at $x_i = 1$" describes an experiment that
  the appendix runs on two groups of $10^4$ weights called $w$.
- **Exercise 9, part 1** contains the two-line derivation that ADAM-19 moves
  into the body; keep the simulation (part 3) as the exercise.
- **Add** a short exercise on scale invariance: show that Adam with
  $\epsilon = 0$ takes the same steps on $f$ and on $c f$, and find the size
  of $\epsilon$ relative to the gradients at which this stops being true.
- After the split, exercises 1 to 6 go with Adam and 7 to 10 with AdamW.

## Still open from October

"The test model `TinyLM` is built inside the Adam section from ch. 11
material; four later sections depend on it": open, and now three chapters
depend on it. "$\lambda$ is curvature, then a decay rate, then weight
decay": open. The sentences the October review asked to keep ("One clock
cannot serve both populations") are intact, and the mannerisms the writing
guide quotes from the old `adam.md` and `adamw.md` are gone.

## For the chapter review

- **Split into three** (Adam; AdamW and weight decay; the test models), with
  the seams at lines 383, 943 and 1254. `sec_adamw` and `subsec_tinylm` are
  already cited as if they were sections. Where the test models go (before
  Adam, or before Momentum) is an author decision that also affects
  `momentum.md`.
- **Order and arc**: Momentum to Adam works well. `adam.md` reuses the leaky
  average, the window $1/(1-\beta)$ and the effective step from
  `momentum.md` correctly and in one clause each. The valley is re-derived
  at the top of both sections.
- **Against the appendix** (`mdl-adaptive-stochastic-methods.md`): the
  division of labour follows decision 3 and all pointers are accurate. Two
  small results that the appendix has and the main text needs: scale
  invariance (ADAM-5) and the fixed points of the two decay forms
  (ADAM-18). Two exercises depend on appendix code.
- **Forward dependencies**: `TinyLM` uses chapter 11 material by design;
  nothing depends on the Language Models skeletons. "Optimizer-State
  Memory" points to :numref:`sec_training_systems` and
  :numref:`chap_performance` and disagrees with the latter about where bf16
  is stored.
- **Patterns to check across the chapter**: (1) experiments whose result is
  within run-to-run noise, kept and explained instead of redesigned (the
  later sections reuse `lr_sweep` on the same two models, so `muon.md`,
  `batch-size.md` and `practice.md` should be read for the same fault);
  (2) paper retelling in "at scale" subsections; (3) numbers in prose that
  no cell prints; (4) one object under many names; (5) tab notes that carry
  plumbing.
