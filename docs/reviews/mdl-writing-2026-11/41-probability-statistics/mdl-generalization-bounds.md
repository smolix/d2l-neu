# Uniform Convergence and Generalization

Source: `chapter_mdl-probability-statistics/mdl-generalization-bounds.md`.
About 6,400 words of prose in the body (introduction 390, "Uniform
Convergence" 1,250, "Rademacher Complexity" 2,810, "Interpolation and Double
Descent" 1,760, summary 220) plus 1,900 words of exercises. Six code cells,
two generated figures, ten exercises. Anchors below are verbatim quotes
(blockquote lines); the heading they sit under follows in parentheses.

## Verdict

The middle of this section, "Rademacher Complexity", is very good: a
definition motivated by a question a practitioner understands, three tiny
cases computed by hand, a symmetrization proof in three named steps, an
exact computation for the linear class with an analysis of where the slack
is, and a margin bound with numbers. It is deeper than the main text and it
is the reason to keep the section. The parts on either side are weaker for
different reasons. The first part re-teaches, at the same level and with
nearly the same numbers, what :numref:`chap_classification_generalization`
already teaches and proves. The third part reruns the double-descent
experiment of the main text; its conclusions are stated more strongly than
the experiment supports, and its closing explanation of benign overfitting
appeals to the same formula that explained the catastrophe at the threshold
without saying what differs. I found no error in a proposition or proof and
every quoted number checks, but several inferences are looser than the
mathematics around them. The single most important fix is the third part:
say what the sweep does and does not show (GEN-17), and either explain
benign overfitting properly or reduce it to one honest sentence (GEN-19).

## Scores

| Dimension | Score |
|---|---|
| Clarity of explanation | 4 |
| Clarity of thought | 3 |
| Pedagogy | 4 |
| Language | 4 |
| Ease and enjoyment | 4 |

"Rademacher Complexity" alone: 5 for explanation, 4 elsewhere.

## What works

Preserve these in revision.

- **The framing sentence of the section**: "A test-set certificate and a
  learner differ only in the order of two events: choosing the function and
  drawing the sample." And its resolution: "Choosing after seeing the data is
  safe if the guarantee was made for all candidates before seeing it, because
  the one we pick is one of them." These two sentences are the theory.
- **The figure of the 1,000 coins** (:numref:`fig_mdl-gen-best-of-many`): an
  expected histogram on a log scale, with the single-coin band, the 3.3 coins
  expected beyond its edge, and the wider band of the union bound. It shows
  why the rare tail is populated, which the main text's running-maximum plot
  does not. Rendered, it is clean and legible.
- **The conservatism is measured**: "the exact probability that some coin
  falls outside 27 to 73 heads is about 0.17%, against the 5% the bound
  allows. But it is valid, which the single-coin band was not." (I confirm
  0.17%.)
- **The definition is preceded by its question** ("how well can the class
  fit pure noise?") and followed at once by the smallest cases: two constant
  functions at $n=2$ ($\tfrac12$), $n=4$ ($\tfrac38$), large $n$
  ($\sqrt{2/(\pi n)}$, 0.0796 against 0.0798 at $n=100$). "Even a class with
  only two members correlates with noise at the rate $1/\sqrt n$" is the kind
  of remark that makes a definition stick.
- **The proof of the theorem** names its three moves (ghost sample, sign
  swap, split), says what each achieves ("The unknown risk has been replaced
  by a second sample"; "The split produces the factor $2$"), and gives the
  intuition after the proof, not in place of it. McDiarmid is stated as what
  it is and deferred honestly.
- **"The theorem can be checked at its ends"**: one member gives one-sided
  Hoeffding, a finite class gives 0.078 against the union bound's 0.024,
  "The Rademacher route is weaker for finite classes, but it also applies to
  infinite ones."
- **The slack analysis for the linear class**: "Only one step of the proof
  loses anything", with the loss computed (11% in the plane, from
  $\sqrt\pi/2$; about 1% in twenty dimensions by the thin shell) and then
  confirmed by the cell (0.126, 0.140, 0.141 against 0.141). This is a model
  of how to treat a bound.
- **The objection paragraph**: "A reader may object that the proposition
  needs $B$ fixed before the data, while the norm of a trained weight vector
  is known only afterwards. The objection is right, and a union bound over
  radii answers it." The reader's question, asked and answered inside the
  argument.
- **The peak is not blamed on noise**: "The peak is not caused by the label
  noise", supported by the clean and noise columns of the table. I reran the
  sweep with `noise = 0` and with a target in the span of the first ten
  features; both claims of that paragraph hold (peak of the same size; no
  peak).
- **The closing thought**: every bound covers the worst function of a class
  fixed before the data, while the function we use is the one an algorithm
  reached.

## Structural findings

**S1. Two sections or one told twice?** Two. This section and
:numref:`sec_mdl-concentration-generalization` have different subjects (an
average fixed in advance; a function chosen afterwards) and the boundary
between them is in the right place. The alternative, moving the finite-class
bound into the concentration section as its closing application, would make
that section three subjects long and would leave the best-of-many coins
without the result they motivate. What the pair repeats is small and
listed in the other file's chapter note: the coin teaser, the contrast of
Chebyshev and Hoeffding under a union bound, the definition of the union
bound, and the single-function half-width (restated here correctly, in one
display with a reference). The duplication that matters is with the main
text (S2, S3).

**S2. "Uniform Convergence" repeats the main text at the same level.**
:numref:`chap_classification_generalization` already contains: Hoeffding
with the 95% radius; the union bound over $k$ classifiers with its proof
and the radii 0.014, 0.023 and 0.030 for one, a thousand and a million
classifiers on 10,000 examples; a selection experiment (the best of 10,000
random guessers); the Blum–Hardt majority attack run in code; and the
finite-class uniform-convergence bound with the sentence "it holds in
particular for $f_\mathcal{S}$, however the learner made its choice". This
part gives the same proof, the radii 0.0136, 0.0238 (for 2,000) and 0.0296,
a selection experiment (the best of 1,000 coins), and the attack as an
exercise. The main text says this section "proves the finite-class bound";
the main text has already proved it, since Hoeffding plus the union bound
is the whole proof. What is new here is the figure, the formula for $n$ with
"doubling the number of candidates adds about 3,500 validation examples",
the Chebyshev contrast, and the exact conservatism. Recommendation: cut the
part to about 500 words. Recall the main-text result in a clause with its
reference, keep the coins and the figure as the one illustration, state the
proposition without ceremony, keep the three new items and the leaderboard
caveat. The reader arriving from the main text then meets something new
within a page.

**S3. "Interpolation and Double Descent" is a second topic, and partly a
second telling.** The main text (:numref:`subsec_double_descent`) freezes a
hidden layer, sweeps the width through $h=n$, prints the smallest singular
value and the norm at the threshold, and gives the nesting argument in one
sentence ("give the new unit zero weight"). This part repeats the
construction on synthetic data, restates nesting as a Proposition with a
Proof, and adds three things: the SVD formula for the fit, the clean and
noise parts of the test error, and the "capacity product". The first two are
worth having. The link to the rest of the section is the third, and the text
concedes that it "certifies nothing". So about 1,760 words serve mainly as a
counterexample to the bounds, and much of that length re-establishes what
the main text showed. Recommendation: keep the part here (the section needs
its limit shown, and the main text sends readers to
`subsec_mdl-double-descent-example`), but cut it to what is new, about
1,000 words: the SVD formula, one table, the clean and noise split, the
caveats of GEN-17 and GEN-18. Nesting becomes one sentence with a reference.

**S4. The section's promise about deep learning rests on an exercise.** The
text's only norm-based statement about networks is negative (random labels).
The positive result a deep learning reader would look for, that a two-layer
ReLU network has complexity at most $2B_1B_2r/\sqrt n$ with no dependence on
the width, is the last exercise. One sentence in "The Linear Class" stating
that result and pointing to the exercise would give the weight-decay remark
something to stand on beyond linear models.

**S5. Notation against the main text.** The main text writes
$\epsilon(f)$ and $\epsilon_\mathcal{S}(f)$ for population and empirical
error, calls the half-width the "radius", and writes $v$ for the VC
dimension. This section writes $R(f)$, $\hat R(f)$, "half-width" and
$d_{\mathrm{VC}}$, announces "the notation of learning", and gives no
mapping. A reader following the main text's reference lands here and has to
work it out. One sentence of translation at the first definition suffices.

**S6. Code.** The cells compute what the prose reads, and the readings agree
with the stored outputs (median 0.66 in [0.64, 0.69]; half-widths 0.136 and
0.230; 97% stored against "about 96%" in the prose, exact value 96.3%;
0.126/0.140/0.141; mean norm 1.00, largest 1.48, $r=11.7$, product 2.5; the
table at seven feature counts). The prose correctly quotes ranges for the
unseeded sweep. One cell verifies something that could not have come out
otherwise (GEN-13). The boilerplate sentence "The code in this section is
plain NumPy, the same in every framework tab." and the unused framework
imports are as in the previous section.

## Passage-level findings

**GEN-1** (moderate, thought: repetition). Introduction and "The Best of
Many Coins".
> about 66 heads, although every coin is fair and any one coin, named in

The coin experiment is told in full in the introduction, result included,
and then run 110 lines later as if its outcome were open ("asks how far the
best of the 1,000 coins strays"). The number 66 appears six times in this
file (twice in the introduction, then in the cell's reading, the figure
caption, the finite-class check and the summary) and once more in the
previous section's teaser; the band 37 to 63 is derived twice. Pose the experiment in the introduction as a question the
reader can guess at ("How many heads does the best coin show?") and let the
cell answer it, or give the answer in the introduction and make the cell's
paragraph two sentences.

**GEN-2** (major, pedagogy: overlap). "Finite Classes and the Union Bound".
> The proposition thus turns the test-set-reuse warning of

The claim is that the proposition "turns the ... warning ... into a number".
The main text already has the number, the proof and the experiment (S2). A
reader who has done the main text reads 1,250 words to learn three new
things. Shorten as in S2, and change this sentence to say what is added: the
sample-size form and the measured conservatism. (For the chapter review: the
main text's phrase that this section "proves the finite-class bound" should
change too.)

**GEN-3** (minor, thought). "Finite Classes and the Union Bound".
> coin, at 66, is inside. The bound is conservative: the exact probability that

A sentence on where the factor of 30 between 0.17% and 5% is lost would
teach more than the number alone. It is lost almost entirely in Hoeffding's
bound for one coin (a coin leaves 27 to 73 heads with probability
$1.7\times10^{-6}$, where the bound allows $5\times10^{-5}$), not in the
union bound, which is nearly exact for rare independent events. The reader
would otherwise blame the union bound.

**GEN-4** (moderate, thought). "Finite Classes and the Union Bound".
> The guarantee then degrades much faster, as the analysis of adaptive data

"Much faster" is not quantified, although the section's own exercise gives
the rate: the attack moves the reported loss by an amount of order
$\sqrt{k/n}$ after $k$ submissions, against $\sqrt{\log k/n}$ for a fixed
list. One clause with the two rates makes the paragraph's point exact.

**GEN-5** (minor, language). "Rademacher Complexity", opening.
> part is due to Koltchinskii and Panchenko and to

One attribution has a citation and the other does not. Cite both or name
neither.

**GEN-6** (moderate, explanation). "Correlation with Random Signs".
> average. Below, predictors take values in $[-1,1]$ and losses in $[0,1]$.

The convention is broken by the section's main example. The linear class
$\{\langle\mathbf w,\mathbf x\rangle:\|\mathbf w\|\leq B\}$ takes values in
$[-Br,Br]$, and two subsections later its complexity is 2.5 and "at least
1", after the reader was told that 1 is the maximum, "reached by the class
of all functions into $[-1,1]$". The sentence "A predictor class that fits
every sign pattern on the sample has $\widehat{\mathfrak R}_S(\mathcal F)=1$"
is likewise true only for classes into $[-1,1]$. Drop the blanket
convention. State the range where a result needs it (the scale paragraph,
Massart's lemma, the 0-1 loss identity), and say once that for an unbounded
class the complexity can exceed 1 and that fitting every sign pattern means
at least 1.

**GEN-7** (moderate, correctness; fairly sure). "The Uniform Convergence
Theorem" and its uses.
> R(g) \le \hat R_S(g) + 2\,\mathfrak R_n(\mathcal G)

The theorem is stated with the expected complexity $\mathfrak R_n$, an
average over samples. Everything the section then computes is the empirical
complexity $\widehat{\mathfrak R}_S$ of the sample at hand. For the linear
class with $\|\mathbf x\|\leq r$ for every input this is harmless, because
the bound $Br/\sqrt n$ holds for every sample, and the margin bound says so
("With $\|\mathbf x\|\le r$ for every input"). But the two numerical
subsections take $r$ to be the largest row norm of a Gaussian feature
matrix, which is read off the sample and has no bound in advance, and the
closing remark of the definition ("how freely the class can vary on the
actual sample points") suggests a data-dependent guarantee that the stated
theorem does not give. State after the theorem, in one sentence, that the
same bound holds with $\widehat{\mathfrak R}_S$ in place of
$\mathfrak R_n$ at the price of a confidence term
$3\sqrt{\log(2/\delta)/(2n)}$ (a second use of McDiarmid). Then the
empirical computations are covered, and "its $B$ was read off the fit" is
the only objection left to the capacity product.

**GEN-8** (minor, explanation). "The Uniform Convergence Theorem".
> on $n$ points gives $\mathfrak R_n\lesssim\sqrt{d_{\mathrm{VC}}\log n/n}$

The symbol $\lesssim$ is not defined, and the VC dimension is $v$ in the
main text (S5). Write "at most a constant times".

**GEN-9** (minor, pedagogy). "The Linear Class", margin bound.
> is its training ramp loss plus two terms we can evaluate. For $Br/\gamma=10$,

The reader has no way to tell whether $Br/\gamma=10$ is typical, generous or
absurd for a real classifier; the number is chosen to make the bound come
out at 9.5%. The paragraph ends by saying that fitting random labels "forces
$Br/\gamma$ up", which the next subsection could confirm with its own
numbers but does not connect. Say that 10 is an illustration, and tie the
remark forward: the sign fits of the next cell need $Br/\sqrt n$ of at
least 1 at margin $\gamma=1$, which for $n=50$ is $Br/\gamma\geq7$, already
past the point where :eqref:`eq_mdl-margin-bound` says anything at that
sample size. (Measuring $Br/\gamma$ for a trained linear classifier on real
data would be the convincing version; that is a new experiment and is
deferred under the current rules.)

**GEN-10** (minor, explanation). "The Linear Class".
> not bounded by $1$: on the class they reach about $1+Br$, so they must be

"Must be divided by that maximum" leaves the consequence unsaid: the
complexity term then carries the factor $1/(1+Br)$ and the bound is on the
rescaled loss, which is why the ramp loss is the cleaner route. One clause.

**GEN-11** (moderate, explanation). "Rademacher Complexity in Numbers",
second experiment.
> by construction. The informative number is the norm the fits need. Since

The inverse-Wishart mean is heavy machinery for a prediction (1.01) that
plays almost no part in the argument, and the same derivation is the
section's own exercise ("Use this identity to predict the typical
$\|\mathbf w\|$"). Leave the identity to the exercise, and in the text say
only that the fits have norm about 1 while the rows of
$\boldsymbol\Phi$ have length about 10.

**GEN-12** (moderate, explanation). Same subsection.
> and 2.8. A ball that holds an interpolant of every one of the $2^{50}$ sign

The paragraph argues backwards. It reports 2 to 2.8 for a ball that holds
2,000 fits, says a ball for all $2^{50}$ would be larger, says its value
"must be at least 1", and only then gives the reason. The reader cannot
tell what the cell established. Put the logic first: a class that
interpolates every sign pattern has $\widehat{\mathfrak R}_S\geq1$, the
proposition gives $\widehat{\mathfrak R}_S\leq Br/\sqrt n$, hence any ball
containing all the interpolants has $Br/\sqrt n\geq1$. Then say what the
cell adds to that necessity: how close to 1 real fits come (a typical fit
has $\|\mathbf w\|r/\sqrt n\approx1.0\times10/7.1\approx1.4$; the ball
around the 2,000 fits gives 2 to 2.8).

**GEN-13** (minor, pedagogy). Same cell.
> W = np.linalg.pinv(Phi) @ signs.T             # min-norm fit of each sign vector

Following GEN-12: as written, the cell's printed product (2.5) confirms an
inequality that holds by logic. It becomes informative if the prose reads it
as "interpolating noise costs a norm only 1.4 to 2.5 times the minimum the
bound allows", that is, the bound $Br/\sqrt n$ is nearly tight for this
class even at complexity 1.

**GEN-14** (moderate, thought). "Rademacher Complexity in Numbers", networks.
> architecture, trained this way" therefore fits every sign pattern on those

The main text's experiment fit one draw of random ten-class labels on 1,000
images. From that, the text concludes that the class "fits every sign
pattern on those inputs" and that its complexity "is about 1". The
conclusion is reasonable; the inference as written is not. And "this
architecture, trained this way" is not a class fixed before the data in the
sense the theorem requires (trained on which labels?). The sound argument is
short and should replace it: take the class of all functions the
architecture can represent. Training on labels $\boldsymbol\varepsilon$
returns a member whose correlation with $\boldsymbol\varepsilon$ is about 1,
so the supremum is at least that, and if training succeeds for typical
sign vectors (the experiment tried one labeling, and
:citet:`zhang2021understanding` many) the average over
$\boldsymbol\varepsilon$ is about 1.

**GEN-15** (minor, thought: repetition). "The Minimum-Norm Interpolant".
> **Proposition (nesting).** *For nested features and $p\ge n$, with

The main text gives this argument in one sentence ("an exact fit with $h$
hidden units remains an exact fit with $h+1$ (give the new unit zero
weight), so the smallest norm among the exact fits cannot grow") and then
refers the reader here. Here it returns as a Proposition with a Proof of the
same content. Keep it as a sentence with the reference. The remark after it,
that the norm grows with $p$ below the threshold because least squares is
not an interpolant, is new and should stay.

**GEN-16** (moderate, correctness; fairly sure). "A Double-Descent Sweep".
> shape of the test curve, but it certifies nothing, for two reasons. First,

The threshold 1 has no meaning for this experiment. "A complexity bound of
$1$ or more ... the resulting bound on the test error is empty" is a
statement about a loss in $[0,1]$: twice a loss-class complexity of
$\tfrac12$ exhausts the range. The sweep is a regression with squared loss
and real targets of variance 1; the theorem does not apply to it at all, and
the product $\|\mathbf w\|r/\sqrt n$ changes if the targets are rescaled.
The first "reason" is therefore an analogy with the classification bound,
not a consequence of it, and "stays above 1" is repeated as a finding in the
closing paragraph and in the summary ("while its capacity product stays
above $1$"). Say that the product is a diagnostic borrowed from the linear
classification bound, that no bound of this section covers the squared loss
on unbounded targets, and give the reasons it certifies nothing as: no
applicable theorem, $B$ read off the fit, and $r$ read off the sample
(GEN-7). The observation that it tracks the test curve can stay.

**GEN-17** (major, thought). "A Double-Descent Sweep", conclusion.
> the experiment is the most overparameterized one, fitting noisy data exactly,

The comparison the conclusion rests on is weaker than it reads. The targets
have variance 1 ($\|\boldsymbol\beta\|=1$, standard Gaussian inputs), so a
test error of 1 is the error of predicting zero. The text reports "about
$1$ at $p=2$" and an underparameterized minimum of "roughly $0.6$ to
$0.8$" without saying that this means the best classical model explains a
quarter to a third of the variance. The classical regime is weak by
construction: 40 points, a linear target in 15 dimensions, and random ReLU
features that represent it badly. Ordinary least squares on the 15 raw
inputs, also a classical model, has expected test error
$\sigma_\xi^2\,d/(n-d-1)=0.01\cdot15/24\approx0.006$, about nine times
better than the $p=400$ interpolant's 0.05. So the sweep shows that within
one nested family of random features, more features past the threshold are
better and the widest model is best. It does not show that interpolation
beats a well-specified small model, which "roughly ten times below the best
underparameterized model" and the summary's "ten times better than the best
classical fit" invite the reader to conclude. Add two sentences of candour
where the first descent is described: name 1 as the trivial level, and say
what least squares on the raw inputs would achieve. Rephrase the conclusion
and the summary as statements about this feature family. The section's real
point (the training error cannot tell $p=42$ from $p=400$; the norm and
$\sigma_{\min}$ can) loses nothing.

**GEN-18** (minor, correctness; sure). "A Double-Descent Sweep".
> the cell to the next, because the peak is heavy-tailed: for a square Gaussian

The fact is true for a square Gaussian matrix; the matrix here has ReLU
features. Write "as for a square Gaussian matrix, where
$\mathbb{E}[1/\sigma_{\min}^2]$ is infinite". Similarly "invertible with
overwhelming probability" two subsections earlier could simply be "almost
always invertible".

**GEN-19** (major, explanation). "Benign Overfitting".
> interpolated noise across the many directions with small $\sigma_i$, and each

This paragraph is the answer to the puzzle the subsection poses, and it does
not explain. Three problems. (i) It cites :eqref:`eq_mdl-svd-interpolant`,
which says that a component along a direction with small $\sigma_i$ is
*amplified* by $1/\sigma_i$. The same equation was used 150 lines earlier to
explain the peak ("the smallest singular value ... amplifies most"). The
reader is told that small singular values are the disease at $p=n$ and the
cure at $p=400$, by the same formula, with no account of the difference.
The missing step: a noise component along $\mathbf u_i$ enters the weights
as $\langle\mathbf u_i,\boldsymbol\xi\rangle/\sigma_i$ along $\mathbf v_i$,
and what it does to a test prediction is that weight times the test
feature's component along $\mathbf v_i$. Far past the threshold a small
$\sigma_i$ reflects a direction in which the features truly vary little, so
test features are small there too and the two factors cancel. At $p=n$ the
smallest singular value is small by an accident of the sample, test features
are not small in that direction, and nothing cancels. (ii) Nothing in the
notebook shows the spectrum the explanation relies on. I computed the
median singular values at $p=400$: one near 53, about fifteen between 26 and
9, then a tail falling from 8 to 2.7. That is the claimed shape, and
printing five of these numbers would turn an assertion into an observation
(a changed cell, so deferred; the prose can at least say it). (iii) "Far
past the threshold, interpolating the noise adds about the noise variance
itself to the test error" is presented as a small cost. It means the
predictions carry as much noise as one training label, with no averaging at
all, where a well-specified least-squares fit would carry
$\sigma_\xi^2d/n$. "Benign" here means bounded, not vanishing, and the text
should say so. If the full explanation is too long for the appendix, cut the
paragraph to one sentence that states the condition and cites
:citet:`Bartlett.Long.Lugosi.Tsigler.2020`, and do not claim that the
equation "explains why".

**GEN-20** (minor, language). Closing paragraph of "Benign Overfitting".
> We can now say how much more data a choice among a fixed list of candidates

Three "We can" sentences recap the section immediately before a ten-sentence
summary that recaps it again. The guide asks that results be restated in
the summary only, and names "We can now" as a formula to avoid; it also
closes the two preceding sections of the chapter. Keep the last two
sentences of the paragraph, which belong to the argument, and delete the
recap.

**GEN-21** (minor, language: pattern). Throughout.
> which says nothing about a loss in $[0,1]$. Hoeffding gives $t=0.024$ for the

Nine variations of one verdict in the body: "says nothing" (four times),
"certifies nothing" (three), "guarantees nothing", "gives nothing". Other
instances: "and the inequality above, which holds for each $f$ separately,
says nothing about it"; "the complexity term is $1$ and the bound says
nothing"; "and :eqref:`eq_mdl-rademacher-bound` certifies nothing". Each is
accurate; together they become a refrain. Vary by saying what the bound
does say in each case (an error below 100%; a deviation that may be as
large as the range).

**GEN-22** (minor, language). Summary. Ten sentences that list the section
in order, including the coin count for the fifth time. Reduce to the four
answers: what uniform convergence is and why selection needs it, how the
finite and the Rademacher bound achieve it, what the linear class's
complexity is, and where the argument stops.

## Exercises

Ten problems, about 29% of the body. The set is strong and better graded
than most. "Thresholds on the line" is the best: exact Rademacher
complexities by enumeration, a random walk, and a comparison with Massart
that asks what the lemma ignores. "Massart's lemma and $\ell_1$ balls",
"Margin bound", "Ranking configurations on a holdout" (the winner's curse
with an exact binomial check) and "Norms, not widths" are all worth their
place; the last holds the section's most useful statement about networks
and should be mentioned in the text (S4).

- **"Leaderboard attack", part (a)** asks the reader to implement the attack
  that the main text already runs in a cell
  (:numref:`chap_classification_generalization`, "Adaptive Overfitting").
  Start the problem from that cell and keep the comparison with
  :eqref:`eq_mdl-finite-class`, the constant in (b) and the Ladder in (c).
- **"McDiarmid's inequality", part (b)** is the step the proof in the text
  performs. It can stay as a check; part (c), the median, is the
  instructive one.
- **"Fitting random signs", part (b)** repeats the text's inverse-Wishart
  prediction. If GEN-11 moves the derivation out of the text, the
  duplication disappears.
- **"Benign spectra"** is a reproduction of a research paper in three
  parts. It is marked extended and is well specified, but its two effective
  ranks are written $\rho_k$ and $\varrho_k$, which are hard to tell apart
  in print; rename one ($r_k$ and $R_k$, as in the paper).
- **"The square case", part (c)** carries two claims the text makes without
  showing (no noise leaves the peak; a target in the span of ten features
  removes it). I confirmed both by rerunning the sweep; the problem is
  answerable and good.

A problem that asks what least squares on the raw inputs achieves in the
sweep would follow from GEN-17.

## Still open from October

The October review treated this material as part of one concentration
section and asked for the "benign-overfitting survey" to be cut. The survey
is gone; what remains is one explanatory paragraph that does too little
(GEN-19). Nothing else from October applies here.

## For the chapter review

- **Order and boundary.** The section follows the concentration section and
  needs from it only Hoeffding's inequality, the union bound and, for one
  remark, the thin shell. The boundary between the two is right. They are
  two sections. What they repeat between them is small; see the list in the
  other file.
- **Main-text overlap is the structural issue of the pair.** Part one here
  duplicates :numref:`chap_classification_generalization` (same proof,
  nearly the same numbers, same kind of experiment, same attack), and part
  three partly duplicates :numref:`subsec_double_descent`. The main text
  points here several times, once with a promise ("proves the finite-class
  bound") that it has already kept itself. The appendix should start from
  the main-text results and spend its length on Rademacher complexity, the
  margin bound and the clean and noise split. That would bring the section
  from 6,400 to about 4,800 words.
- **The experiment claims more than it shows** (GEN-17, GEN-19, GEN-16).
  This is the fault the Basics review found most often, and it occurs here
  in the one part of the section that is an experiment. I suspect it recurs
  wherever the appendix reruns a main-text experiment.
- **What it sets up.** The further-reading notebook continues from the
  section's closing question (PAC-Bayes, stability, covering numbers), and
  the hand-over is clean. The next section, naive Bayes, has no connection
  to this one beyond one exercise; the chapter's arc ends here, and naive
  Bayes reads as an appendix to the chapter.
- **Shared mannerisms.** "We can now" closings in three consecutive
  sections; a verdict phrase repeated as a refrain ("says nothing",
  "certifies nothing"); summaries that list every result; notation that
  differs from the main text's for the same objects without a mapping.
- **Correctness.** No errors in propositions or proofs; all numbers verified
  (binomial tails, half-widths, Rademacher values, the margin-bound
  arithmetic, the inverse-Wishart prediction). Three places where the
  reasoning outruns the mathematics: the expected against the empirical
  complexity (GEN-7), the threshold 1 applied to a regression (GEN-16), and
  the inference from one random labeling to every sign pattern (GEN-14).
