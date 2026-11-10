# Scaling Up

Source: `chapter_optimization/scaling.md`, lines 1 to 686 (slides ignored). About 2,850 words of body prose plus 800 words of exercises; 9 code cells per framework.

## Verdict

The first half of this section is the best explanation of the maximal update parametrization I know at textbook level: a sweep that shows the optimum drifting with width, a four-line argument that explains the drift from the sign of an outer product, and a check that takes seconds. A reader could reconstruct all of it a week later. The second half does not hold up as well. The headline measurement of the section, the cost of transferring a learning rate from width 128 to width 1,024, comes out as stated in the stored PyTorch run (20% under standard parametrization, 5% under muP) and reversed in the stored JAX run (2% against 6%). The prose anticipates this with "an occasional run brings the two close together", which means the section's main claim rests on a difference within the run-to-run noise. In addition, the statement that muP leaves the initialization unchanged does not match the published rules for the output layer, and the description of the falling logit curve does not match what the plot shows for a base at the widest member. The single most important fix: carry the argument on what the experiment shows robustly in both tabs (the optimum moves 4 to 8 times under standard parametrization and at most 2 times under muP; the activations stop growing), and demote or redesign the percent-above-best comparison.

## Scores

| Dimension | Score |
|---|---|
| Clarity of explanation | 4 |
| Clarity of thought and focus | 3 |
| Pedagogy | 4 |
| Language | 4 |
| Ease and enjoyment | 4 |

## What works

- **The opening** gives the reader a reason to care in six sentences and names the failure concretely: "the large run diverges or trains slowly, and the error shows only after much of the budget is spent."
- **"Note where the width lives."** (lines 56 to 58). The observation that the input matrix has fan-in 784 and the output matrix fan-out 10 at every width, "so only the two $n \times n$ matrices in the middle grow in both dimensions", is set up before it is needed and pays off three times (the first-step argument, the table, the flat layer-1 curve). This is how a good lecturer plants a fact.
- **The first Adam step** (lines 224 to 258). "All $n$ terms of the sum have the same sign, because the update is built from the very activations that it multiplies. Their sizes add up instead of cancelling". This is the sentence of the section. The derivation uses only the Adam bias correction and the outer-product gradient, both of which the reader has. It also says honestly where it stops: "A minibatch gradient is a sum of such outer products, and its sign no longer factorizes, so this argument holds only to leading order."
- **The implementation is a nine-line subclass** with the two rules marked in comments, and "At the base width $m = 1$, and `MuMLP` is `MLP`" gives the reader a sanity check for free.
- **The coordinate check is read with judgement**: "Growth is the unambiguous failure. A flat or falling curve says only that the forward pass is stable, because the check reads the sizes of activations, not whether features are learned." This is the authors judging their own tool, which the voice model asks for. The list of errors it catches ("an initializer with a fixed standard deviation, like the 0.02 common in transformer code") is the practical detail that matters.
- **The section is short** (2,850 words) and has one question. Nothing here is a survey until the last three paragraphs.

## Structural findings

1. **The transfer experiment is too small for its claim.** An eightfold range of widths on a three-layer MLP gives an optimum that moves by 4 to 8 under standard parametrization, and the width-1,024 loss curve is flat enough near its minimum that a learning rate 4 times too large costs 2% in one stored run and 20% in the other. The percent-above-best statistic compares two single-seed numbers on a factor-of-two grid. The authors know (exercise 1 exists to "measure the spread"), but the body prose still leads with it and the Summary generalizes from it. For the writing pass: lead with the position of the optimum and the alignment of the curves, both visible in both tabs. For the experiment pass: extend the sweep to width 4,096 (a 32-fold range, where a $1/n$ drift is unmistakable and the coordinate check already goes), or average three seeds in the cell.
2. **"The Spectral View" is two sections' worth of claims in 400 words.** It introduces a new condition, cites a paper for its equivalence to muP, hands the one calculation to an exercise, then turns to Muon and corrects a constant of :numref:`sec_muon`, and closes with three sentences on depth, weight decay and "the coordinate check is worth running". The Muon link is the reason the subsection exists and is worth having. The spectral condition needs either its three-line check in the text (the rank-one sign matrix has spectral norm $\eta\sqrt{k_{\textrm{out}} k_{\textrm{in}}}$; this is small and is the same calculation as :eqref:`eq_scaling_first_step`) or to be cut to the two sentences about Muon.
3. **What the reader would look for and not find.** (a) What to do with a real model: the section says transformers need "one further rule" and that embeddings are treated apart, then sends the reader to a blog post. One short table row each for embeddings, attention logits and LayerNorm gains would make the section usable on `TinyLM`, which the rest of the chapter is built on. (b) Whether to use muP at all: the honest position, stated in the last paragraph and in exercise 7, is that practice is divided and that many large runs use standard parametrization with retuning or fitted power laws. That judgement belongs in the body, in the authors' voice, not in an exercise preamble.
4. **Order is right.** Sweep, explanation, rule, quick check, transfer, then the link to Muon. Do not reorder.

## Passage-level findings

**SCALE-1** (minor, explanation). Under "A Family of Widths". Anchor: "Up to constants it is what default initializers do and what the".
The reader may wonder why the He factor of 2 is dropped for a ReLU network. Exercise 5 mentions that variance $1/\textrm{fan-in}$ "halves the second moment of the ReLU activations at every layer". With three layers this is harmless, but say so in a clause here, since the section is about getting scales right.

**SCALE-2** (minor, thought). Under "A Family of Widths". Anchor: "The score of a run is its cross-entropy
on the whole training set after the last step.".
The score is a single iterate after 400 noisy steps at a constant rate, so near the stability edge it carries the noise of the last step. This is part of why the transfer cost fluctuates (SCALE-9). A mean over the last 50 steps, or an evaluation at two nearby steps, would steady it. Experiment pass.

**SCALE-3** (minor, correctness). Under "Learning-Rate Sweep". Anchor: "the drift
continues at larger widths :cite:`Yang.Hu.Babuschkin.ea.2022`".
Supported by the cited paper for transformers and MLPs, as far as I recall its figures. Fine. But the sentence before it, "several times smaller", can be sharper using the section's own prediction: the stored runs give factors of 8 (PyTorch) and 4 (JAX) for an 8-fold width, against the $1/n$ of the argument that follows. Say after the derivation that the measured drift is "between half of and all of" the predicted factor, and why it can fall short (the input layer and biases do not scale; the minibatch sum is only partly coherent).

**SCALE-4** (minor, explanation). Under "The First Adam Step". Anchor: "that receives activations $\mathbf{h}$ whose entries have typical size".
$\overline{|h|}$ is introduced as "typical size" and then used as the mean of $|h_j|$ in :eqref:`eq_scaling_first_step`. Define it as the mean absolute entry. Also say that the change is measured on the same example that produced the gradient; that is what makes the signs line up, and the coordinate check does exactly this (it evaluates on the batch it trained on).

**SCALE-5** (moderate, explanation). Under "The First Adam Step". Anchor: "Doubling the width doubles the change, so the largest stable
learning rate falls in proportion to $1/n$".
Two things are skipped. First, why a change of order $\eta n$ in the pre-activations is a problem: the reader has to supply that the pre-activations themselves are of order one at every width (variance $1/\textrm{fan-in}$ guarantees it), so the *relative* change grows like $n$. One sentence. Second, the argument is for the first step; that the stable learning rate over 400 steps follows the same law is an extrapolation, which the sweep supports. Say that it is one.

**SCALE-6** (moderate, correctness; sure of the published rules, less sure how much it matters at these widths). Under "The Rules". Anchor: "The initialization column is the one that does *not* change: variance".
For the output matrix this is not what muP prescribes. In Yang and Hu's table the output weights have variance $\propto 1/\textrm{fan-in}^2$ with no multiplier; in the equivalent formulation with a $1/\textrm{fan-in}$ multiplier (the one this section uses, and the one in the Dey et al. guide, as far as I recall it) the output weights have a variance that is *fixed* at its base-width value and does not shrink with width. `MuMLP` keeps variance $1/n$ and adds the $1/m$ multiplier, so its output layer starts smaller than muP's by a factor $\sqrt{m}$ in standard deviation: initial logits scale as $1/m$, not $1/\sqrt{m}$. This variant is harmless and arguably deliberate (the muP authors recommend a zero-initialized readout, and this moves toward it for $m > 1$), and the learning-rate rules are correct. But the sentence "variance $\propto 1/\textrm{fan-in}$ is right at every width" and the table entry "unchanged" for the output matrix state as the rule what is a deviation from it. Fix: in the table write the output row as it is implemented and add one sentence: "muP proper holds the output matrix's initial variance fixed as the width grows; we keep $1/\textrm{fan-in}$, which makes the initial logits smaller still and changes nothing in what follows." Check against the two sources before editing.

**SCALE-7** (minor, focus). Under "The Rules". Anchor: "SGD has its own column of
exponents, and several algebraically equivalent formulations are in use.".
Three sentences in a row (embeddings, attention logits, SGD) each open a door and close it. See structural finding 3: either give the transformer rows, since the chapter's main model is a transformer, or cut all three to one pointer.

**SCALE-8** (moderate, explanation). Under "The Rules". Anchor: "A base at width 128 would instead fix the constants where
they matter least".
The paragraph on the choice of base (lines 350 to 362) is the hardest in the section and the least concrete. "fix the constants where they matter least" is unexplained, and the measured consequence ("one to three hundredths of a nat above tuned standard parametrization") is a number no cell prints, about an experiment the reader has not seen, pointing to an exercise. A reader about to use muP needs to know one thing: the rules give exponents, and the base width fixes prefactors that still need tuning once. Suggested replacement for the last four sentences: "We put the base at the widest member, 1,024, so that `MuMLP` there is the standard network and no extra constants need tuning. The price is that the narrow members run with hidden learning rates and logits up to eight times larger than a standard network's. An exercise moves the base to 128 and finds what changes."

**SCALE-9** (major, correctness; stored output contradicts the prose in one tab). Under "Learning-Rate Transfer". Anchor: "Under muP the transferred learning rate typically lands within a few".
Stored outputs of cell `scaling-learning-rate-transfer-1`: PyTorch prints 20% (SP) and 5% (muP); JAX prints 2% (SP) and 6% (muP). In the JAX tab the reader sees standard parametrization transfer *better* than muP directly above a paragraph that says the opposite, and "Under standard parametrization it typically lands 10% to 30% above it" is contradicted by the number on the page. The next sentence, "Under muP the best loss also falls from width 128 to width 1,024", also fails in the JAX output (0.304, 0.299, 0.286, 0.293: it rises at the last width). The guide's rule applies: an experiment must show what the prose says it shows, in every tab. What both tabs do show: under standard parametrization the grid optimum moves from $7.8 \times 10^{-3}$ or $3.9 \times 10^{-3}$ at width 128 to $9.8 \times 10^{-4}$ at width 1,024, a factor of 4 to 8; under muP it moves by one grid step, from $4.9 \times 10^{-4}$ to $9.8 \times 10^{-4}$. Rewrite the paragraph around that: "Under standard parametrization the best learning rate fell by a factor of four to eight between widths 128 and 1,024. Under muP it stays within one grid step. What this is worth in loss depends on how flat the wide network's curve is near its minimum, and on a family this small it is often flat enough that the cost is a few percent either way." Then the percent line in the cell can go, or stay as a number the prose reads cautiously.

**SCALE-10** (moderate, thought). Under "Learning-Rate Transfer". Anchor: "The narrower curves
have moved to meet it, and the minimum no longer drifts with width".
Look at the rendered muP sweep in both tabs. The left arms and the minima align, as stated. The right-hand walls do not: they are now in the *opposite* order from the standard sweep. Width 128 leaves the chart at a learning rate two to four times lower than width 1,024. So muP with this base has overcorrected the stability edge for the narrow networks, which is what one expects if the minibatch update is less coherent than the single-example argument assumes (the measured drift under SP was sometimes 4, the correction is always 8). The reader will see this in the plot and the text is silent. It is also the more interesting result: it says the exponent 1 is an asymptotic statement and that at widths of a few hundred the effective exponent is smaller. Two sentences. This is the kind of thing an experiment teaches beyond the theorem.

**SCALE-11** (moderate, explanation). Under "The Coordinate Check". Anchor: "$1/m$ multiplier keeps the logits of the wider networks small at first, and".
With the base at 1,024, four of the six plotted widths have $m < 1$, so there the multiplier *enlarges* the logits, up to eightfold. The rendered plot shows a mean absolute logit of about 10 at width 128 and about 6 at width 256 after one step. A ten-class softmax with logits of that size is saturated, and the cross-entropy of the narrow model starts far above $\log 10$. So the sentence describes the right half of the curve and the reader is looking at the left half; and "this is by design" covers a start that no one would design. Fix: say what the plot shows ("with the base at 1,024 the narrow networks start with logits several times larger than a standard network's; this is the constant-factor cost of our choice of base, see above"), and connect it to SCALE-10, since saturated logits are a likely reason the narrow muP networks destabilize earlier.

**SCALE-12** (minor, thought). Under "The Coordinate Check". Anchor: "The plot shows :eqref:`eq_scaling_first_step` at work.".
The equation predicts growth proportional to $n$. The plot (log-log) shows layer 2 growing about sixfold and layer 3 about twentyfold over a 32-fold range of width, flat until about width 500, because below that the order-one initial activation hides an update of order $\eta n \overline{|h|}$ with $\eta = 2^{-8}$. The prose says "grow with width" and stops. Give the reader the reading: where the curves leave the flat part ($\eta n \overline{|h|} \approx 1$, about $n = 500$ to 1,000 here) and that the slope beyond is close to one for layer 2 and steeper for layer 3. That turns a picture into a check of the formula, and it predicts the learning rate at which width 1,024 becomes unstable.

**SCALE-13** (minor, language). Under "The Coordinate Check". Anchor: "they grow to order one through learning rather than through size. Growth is the unambiguous failure.".
"Growth" in two senses in adjacent sentences (logits growing through learning, good; activations growing with width, bad). Write "Activations that grow with width are the unambiguous failure."

**SCALE-14** (minor, focus). Under "Learning-Rate Transfer". Anchor: "sweeping a 40-million-parameter proxy, spent about 7% of the pretraining".
Correct as a report of the paper's claim (40 million parameter proxy, 6.7 billion target, 7% of pretraining cost, better than published numbers). It is the right example and the right length. But it sits at the end of a 180-word paragraph that also holds the transfer result, the retuning remark, a second citation of the same paper, and the limits of transfer. Split after "muP saves it."; give the GPT-3 case and the "not every hyperparameter" caveat their own short paragraph.

**SCALE-15** (moderate, explanation). Under "The Spectral View". Anchor: "for the weights and for their updates alike".
The condition is motivated in one sentence and then applied only to updates. For the weights it is not checked, and for the output matrix it fails under the section's own standard initialization ($10 \times n$ with variance $1/n$ has spectral norm near 1, not $\sqrt{10/n}$), which is the spectral-language reason the output layer needs its $1/m$. Either drop "for the weights" or use it: it explains the third row of the table in one line.

**SCALE-16** (minor, thought). Under "The Spectral View". Anchor: "was chosen instead to
match the root-mean-square entry of an AdamW update".
Correct and consistent with `muon.md` (line 465). This sentence quietly tells the reader that the Muon of two sections ago does not have the width-independence that its derivation suggested. Good that it is said; it should also be said, in one clause with a forward reference, in :numref:`sec_muon`. Chapter-level item.

**SCALE-17** (moderate, focus). Under "The Spectral View". Anchor: "Width is only one direction of scaling.".
The closing paragraph raises depth, batch size, training length and a live dispute about weight decay in two sentences and ends on advice. The weight-decay question is the most important caveat about muP in current practice and is handed to exercise 7. Give it three sentences in the body with the authors' judgement of where things stand (the claim, as I understand the cited paper: with decoupled weight decay the update sizes settle to an equilibrium set by $\eta\lambda$ after an initial phase, so muP's role would be mostly that of an implicit warmup; unverified in detail). Then the section ends on its last result, not on "worth running on any family of models one intends to scale".

**SCALE-18** (minor, thought). Under "Summary". Anchor: "the best learning rate then stays within a
grid step or two of its width-128 value".
Accurate for both tabs and the right claim to make. The Summary is otherwise a list of everything mentioned (eight sentences, including the Muon remark). Trim to: the drift and its cause, the rules, the check, the measured result.

## Code and experiments

- **`scaling-learning-rate-transfer-1`**: stored JAX output contradicts the prose (SCALE-9). This is the main defect of the section.
- **`scaling-the-sweep`** and **`scaling-learning-rate-transfer`**: 72 training runs, single seed, last-iterate score. The optimum positions are the robust output; the prose claims about them match both tabs.
- **`scaling-the-coordinate-check-2` and `-3`**: these are checks that can fail and that teach how to debug, not confirmations. Keep. The prose reads the muP logit curve for the wrong half of the range (SCALE-11) and underreads the SP plot (SCALE-12).
- **`coord_check` default `lr=2**-8`** is called "a learning rate in the middle of the grid"; the middle of `lrs` is $2^{-9}$. Trivial.
- **Number no cell prints:** "one to three hundredths of a nat above tuned standard parametrization" for a base at width 128 (SCALE-8).
- **JAX tab**: `configure_adam` needs `optax.multi_transform` and a path-labelling function, 7 lines against PyTorch's 5. The comment "Rule 2: hidden LR / m" sits on the label function, not on the line that divides. Move it to the `'hidden': optax.adam(lr / self.m)` line.
- No TensorFlow or MXNet material. No figure-drawing code.

## Exercises

Seven problems, 800 words; heavy for a 2,850-word body (28%) but each has a question. Assessment:

- Exercise 1 (transfer across seeds) is the body's missing error bar. If the experiment pass adds seeds to the cell, this exercise goes.
- Exercise 2 (breaking muP) is excellent, especially part 3, where the check passes and the model is still wrong. Keep.
- Exercise 3 (choosing the base) carries the explanation that SCALE-8 says the body lacks. Keep, and move its conclusion into the text.
- Exercise 4 (fitting the drift) is good and honest about resolution. Keep.
- Exercise 6 (first update in spectral norm) is the derivation behind "The Spectral View"; answer $\eta\sqrt{k_{\textrm{out}} k_{\textrm{in}}}$. Keep; consider doing part 1 in the text.
- Exercise 7 (weight decay): ten times longer runs (4,000 steps, two widths, nine rates, two optimizers, 36 runs) to probe a claim the exercise itself says the model cannot test well ("`MLP` has no normalization layer"). High cost for an inconclusive result. Cut, or reduce to the pencil part.
- Add: a short exercise applying the rules to `TinyLM` (which parameters are input-like, hidden, output-like; what the attention scaling changes), since the body does not cover transformers.

## Still open from October

Nothing specific to this file was listed. The chapter-level item "TinyLM is built inside the Adam section; four later sections depend on it" does not touch this section, which uses its own MLP.

## For the chapter review

- Assumes: Adam's bias correction and first step (`adam.md`), Xavier and He initialization from Basics (`sec_numerical_stability`), the spectral norm and Muon's update scale (`muon.md`). No appendix reference is needed or made.
- Overlap: none with the batch-size section beyond a shared protocol idiom. The link to Muon in "The Spectral View" corrects an impression left by `muon.md` and should be reflected there.
- Of the chapter's three closing sections this is the shortest and the most focused, and the one whose core (lines 188 to 258) meets the bar. Its weakness is the chapter's general one in a specific form: a single-seed sweep on a small model carries a quantitative claim that the stored outputs of the two tabs do not agree on.
- The part introduction's one sentence about training ("A learning rate tuned for one model may make a model ten times wider diverge") is this section. The SP sweep supports it: at the width-128 optimum, width 1,024 is on the unstable side in the stored PyTorch run. Good.
- For the part: check every section's shared prose against the JAX outputs as well as the PyTorch ones. I found one direct contradiction in three sections by doing so.
