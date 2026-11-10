# Concentration Inequalities

Source: `chapter_mdl-probability-statistics/mdl-concentration-generalization.md`.
About 5,600 words of prose in the body (introduction 380, "Exponential Tail
Bounds" 3,080, "Probability in High Dimension" 2,030, summary 150) plus 1,500
words of exercises. Six code cells, two generated figures, ten exercises.
Anchors below are verbatim quotes (blockquote lines); the heading they sit
under follows in parentheses.

## Verdict

This is one of the better sections I have read in the book, and its first
half is close to the bar. The route from Markov to Chernoff to Hoeffding is
told as one argument with one running example (a classifier with 830 errors
on 10,000 images), one small case worked to the end before the general proof
(the fair coin), and a number after every result. The proofs are the short
ones, and every figure I recomputed is right: I found no mathematical error
in a statement or proof, and all quoted numbers agree with the stored
outputs and with independent computation. The second half (thin shell,
near-orthogonality, consequences) is good but looser, and carries most of
the repetition. The single most important thing to fix is the middle of the
section, the subsection "Sub-Gaussian and Sub-Exponential Variables": in 900
words it stacks two definitions, three propositions and a counterexample
with no problem the reader feels, the pace of the section collapses there,
and one application (Bernstein's interval) quietly assumes the true variance
is known. Reordering that block so each tool arrives when a number needs it
would bring the whole section to the level of its first 2,000 words.

## Scores

| Dimension | Score |
|---|---|
| Clarity of explanation | 4 |
| Clarity of thought | 4 |
| Pedagogy | 4 |
| Language | 4 |
| Ease and enjoyment | 4 |

The first half alone ("Chebyshev's Rate" through "Hoeffding's Lemma and
Hoeffding's Inequality") would get 5 for explanation and pedagogy.

## What works

Preserve these in revision.

- **The opening example poses the problem in numbers before any machinery.**
  830 mistakes on 10,000 images; Chebyshev gives ±2.2% at 95% and "±500%,
  which says nothing" at one in a million; the exponential bound gives ±1.4%
  and ±2.7%. All four numbers check. The reader knows what the section will
  buy before the first equation.
- **The majorant view** ("The Chernoff Method"): Markov, Chebyshev and
  Chernoff as three functions lying above the indicator, with the bound read
  as the average of the function "where the mass of $X$ sits". This is the
  right picture, the figure draws exactly it, and it makes the choice of
  $\lambda$ a visible trade-off. Cover and Thomas do not explain Chernoff
  this well.
- **"No polynomial majorant factorizes this way: $\mathbb{E}[S^4]$ is not the
  product of the fourth moments."** One sentence that says why the
  exponential and not a high power. Keep it verbatim.
- **The fair coin worked to the end before the general proof**, including
  the series comparison with its one-line reason ("the even factors of
  $(2k)!$ alone give $2^kk!$"). The general proof then reads "This is the
  coin's parabola with $(b-a)^2$ in place of $1$", which is how a proof
  should lean on its example.
- **Hoeffding's lemma by the tilted variance.** The proof is the short one,
  the preparatory proposition (variance of a bounded variable) is proved in
  four lines with its equality case, and "The tilt moves no mass outside
  $[a,b]$" is the whole idea in eight words.
- **The constants are translated, not just quoted**: "Hoeffding's half-width
  is the Gaussian one with $\hat\sigma$ replaced by its worst case
  $(b-a)/2=0.5$ and the quantile $1.96$ replaced by $\sqrt{2\log(2/\delta)}=2.72$."
  The reader learns where the factor 2.5 comes from and when it will be
  smaller.
- **Reading the slope of the log plot**: Hoeffding's exponent 0.02 per flip
  against the exact 0.0201, identified as $D_{\textrm{KL}}(0.6\|0.5)$. A
  number computed two ways that almost agree, with the reason for the gap
  sent to an exercise.
- **Candour about what the bounds do not give**: the chapter's coin (9 of
  13) where Hoeffding gives 0.77 and "not even the exact tail rules out a
  fair coin"; "The bound captures the form of the number we started from,
  but not its size" ($5\times10^{-11}$ against $10^{-110}$).
- **The thin-shell opening**: density largest at the origin, yet the
  probability of landing within half the typical length is $10^{-110}$. "No
  sample we could ever draw will contain such a point."
- **The $d=2$ catch in the measurement**: the mean $|\cos|$ is 0.64, not the
  formula's 0.56, and the text says why ($2/\pi$, uniform angle). A text that
  explains the row that does not fit earns trust for the rows that do.
- **Depth against width in the initialization paragraph**: log-variance
  $5K/d$, so "The variance calculation therefore describes a typical network
  only while $K\ll d$". This is a real consequence a reader of the
  initialization section does not have, in three sentences.

## Structural findings

**S1. The section holds two subjects, and that is acceptable, but the join
needs to be shorter.** Part one bounds the error of a test-set average. Part
two is the geometry of random vectors in high dimension. They share the
Chernoff method and are joined by the chi-square. A lookup reader who
arrives from the linear algebra chapter (cosines of random directions) or
from a later chapter wanting the thin shell lands 600 lines into a section
titled "Concentration Inequalities" and can in fact read from "Probability
in High Dimension" onward using only two displayed results from part one.
I would not split the section: each half is the right length and the second
is the first use of the first. What makes the whole feel long is the bridge
(see S2). Add one sentence at the head of "Probability in High Dimension"
naming the two results it uses, :eqref:`eq_mdl-hoeffding` and
:eqref:`eq_mdl-subexponential-tail`, so the lookup reader knows the entry
cost.

**S2. "Sub-Gaussian and Sub-Exponential Variables" is where the section
loses its reader.** Up to this point every tool was introduced by a number
that needed it. Here the order is: definition of sub-Gaussian, three cases,
closure under averaging, a counterexample, definition of sub-exponential
with two unexplained parameters, a proposition with proof, then Bernstein
(stated, proof in an exercise), then at last a number. The reader has been
waiting for Bernstein since the sentence "Bernstein's inequality below
repairs this" 140 lines earlier, and has no reason yet to want
sub-exponential variables at all; the reason is the thin shell, 100 lines
later. Suggested order: (1) Bernstein directly after the 8.3% example, with
its numbers (the 2% classifier) and the caveat of CON-9; (2) the name
"sub-Gaussian" as a remark of one paragraph (it is used later only for the
random sign and in the next section's Massart exercise); (3) the
sub-exponential definition and tail bound moved to the start of "The Thin
Shell", introduced by the question that needs it: the terms $x_i^2$ are
unbounded, Hoeffding does not apply, what does? The counterexample
($x^2$ is not sub-Gaussian) then motivates the definition instead of
preceding it.

**S3. The motivation in the introduction belongs to the next section.** The
fourth paragraph says one-in-a-million guarantees are needed because "A
learner compares thousands of models and keeps the best". Nothing in this
section delivers that; the closing paragraph says so and hands over to
:numref:`sec_mdl-generalization-bounds`, which then tells the story in full.
Within this section the same point is made a second time in "Consequences
for Deep Learning" (CON-17). State the motive in one clause with a forward
reference and spend the saved words on the motive this section does deliver:
sizing a test set, and the geometry of random vectors.

**S4. The coin table is printed twice and read three times.** The cell in
"Chebyshev's Rate" prints exact against Chebyshev; the cell in "The Fair
Coin in Numbers" prints the same two columns plus Hoeffding. The prose
reads the comparison at $n=1{,}000$ three times ("Chebyshev's Rate", "The
Fair Coin", "The Fair Coin in Numbers"): $2.7\times10^{-10}$ against $0.025$
twice, "eight orders of magnitude" twice, "a factor of 15" twice, and $2e^{-2}\approx0.27$ twice. The first
table earns its place as motivation. In the second subsection keep the plot
and the new observations (Chebyshev tighter at $n=100$, the slopes, the
13-flip coin) and drop the repeated readings.

**S5. The pairwise union bound is derived twice in 120 lines, and a third
time in an exercise.** "Near-Orthogonality of Random Directions" counts
$m^2/2$ pairs and gets $m\leq\sqrt\delta\,e^{d\varepsilon^2/4}$. "Distances
and nearest neighbors" counts $n^2/2$ pairs and gets
$d\geq(8/\varepsilon^2)\log(n^2/\delta)$. The Johnson–Lindenstrauss exercise
asks for the second again with $k$ for $d$. One derivation in the text is
enough; the distances paragraph can cite it and give only its result
(CON-17). The October review asked for the initialization and
nearest-neighbour material to be cut. I disagree for initialization (it has
the best new idea of the second half) and agree in part for nearest
neighbours: keep the conclusion and the pointer to the exercise, drop the
display.

**S6. Code.** All six cells compute something the prose reads, and the
readings match the stored outputs (first table, second table, half-widths
±1.36/±0.76/±0.54 and ±1.36/±0.39/±0.27, norms and cosines, shell masses
14.3%/46.5%/95.3%/100% and contrasts 36.9/1.16/0.27/0.075). The `d2l.plot`
call teaches a computed result. Two small points: the imports cell loads
`torch`, `nn`, `F` (and their counterparts in the other tabs) for a section
that uses none of them, and the sentence "The code in this section is plain
NumPy, the same in every framework tab." is about the book's machinery, not
the subject; the same sentence opens the next section. If the four tabs must
exist, drop the sentence. The "contrast" experiment (second half of the cell
under "The Thin Shell in Numbers") introduces a coined quantity used for one
paragraph; it is the only evidence for the nearest-neighbour remark, so it
stays or goes with that remark.

## Passage-level findings

**CON-1** (minor, thought). Introduction.
> Guarantees at one in a million arise in practice. A learner compares

The step from "thousands of models" to "one in a million" is not given
(1,000 models at overall 5% is $5\times10^{-5}$ each; a million is a
different pool). The reader either does not notice or cannot reconstruct
it. Give the arithmetic in a clause or soften to "far below 5%". See S3: the
paragraph should shrink to a sentence with a forward reference.

**CON-2** (minor, explanation). "The Chernoff Method" and
:numref:`fig_mdl-conc-majorants`.
> inequality uses $g(x)=x/t$ for a nonnegative $X$. Chebyshev's applies

The text says Chebyshev applies $(x/t)^2$ to the deviation $X-\mu$, but
panel (a) draws all three curves over the same axis for the same $X\geq0$
with its mass near 1 and $t=2$; the orange curve there is $(x/t)^2$ applied
to $X$ itself. A careful reader checks the figure against the sentence and
they do not match. Either say "in the figure all three are drawn for one
nonnegative $X$" or change the sentence to "Chebyshev's uses $(x/t)^2$,
applied in practice to the deviation $X-\mu$".

**CON-3** (moderate, explanation). "The Chernoff Method".
> the mean of $n=100$ fair coins at $t=0.1$, where the next subsection shows

Panel (b) and its numbers ($-10\lambda+12.5\lambda^2$, minimum $-2$ at
$\lambda=0.4$, $e^{-0.875}$ at $\lambda=0.1$) are discussed one subsection
before the reader can derive the parabola. At this point the reader has
:eqref:`eq_mdl-chernoff` and no moment generating function for a sum. The
panel then has to be re-read after "The Fair Coin". Move the paragraph on
panel (b) to after the display that produces
$\exp(-\lambda nt+n\lambda^2/8)$, where it becomes a picture of the step
just taken ("The exponent is a parabola in $\lambda$").

**CON-4** (moderate, explanation). "The Chernoff Method" and "The Fair Coin".
> Why is the exponential majorant so effective? The function

The intuition for the exponential is given in two halves 90 lines apart:
first "Raising $\lambda$ pushes the exponential down there, but makes it
grow faster beyond $t$, where there is little mass", then this paragraph
after the coin's bound. The second is the better one (value 1 at the
threshold, $e^{-\lambda nt}$ at the mean). Its last sentence, "as long as
the sum rarely strays far from its mean, and the product rule says how
rarely", is circular as an explanation: the thing being bounded is how
rarely the sum strays. Say instead what limits $\lambda$: the function also
grows like $e^{\lambda(S-nt)}$ beyond the threshold, and the product of the
moment generating functions, $e^{n\lambda^2/8}$, is the price of that
growth; the best $\lambda$ balances $e^{-\lambda nt}$ against it. Then the
earlier half-sentence can go.

**CON-5** (minor, pedagogy). "Hoeffding's Lemma and Hoeffding's Inequality",
proof of the lemma.
> The second derivative is a variance. Let $q_\lambda$ be the *tilted* law,

The reader met exactly this fact three sections earlier:
:numref:`sec_mdl-exponential-family` shows that the second derivative of a
log-partition function is a variance, and $\psi$ is the log-partition
function of the family obtained by tilting the law of $X$. One clause with
the reference turns a new computation into a recognition and ties the
chapter together. Do not remove the displayed computation; a lookup reader
needs it.

**CON-6** (minor, explanation). Proof of Hoeffding's inequality.
> argument applied to $-Z_i$ bounds the lower tail. The *union bound* says

The union bound is defined here, for two tails. The two tail events are
disjoint for $t>0$, so their probabilities simply add, as the coin's proof
said ("adding the two tails gives"). The definition is spent where it is not
needed, used in earnest 400 lines later for $m^2/2$ pairs without a
reminder, and then defined again in the next section. Write "adding the two
tails gives the factor 2" here, and define the union bound at the count of
random directions, its first real use.

**CON-7** (moderate, pedagogy). Statement of Hoeffding's inequality.
> independent random variables with $X_i\in[a,b]$, and let

The proposition is stated for a common range. Two hundred lines later the
text needs unequal ranges for $\sum_iu_ix_i$, states the general form in a
display inside the near-orthogonality argument, and refers to an exercise
for the proof. The next section's McDiarmid exercise also assumes the
unequal form. The proof does not change: the product of the lemma's bounds
has $\sum_i(b_i-a_i)^2$ where it now has $n(b-a)^2$. State the proposition
with ranges $[a_i,b_i]$ from the start, or add a two-line remark after the
proof, and keep the exercise for parts (b) and (c) only.

**CON-8** (minor, explanation). "The Fair Coin in Numbers".
> The chapter's coin, 9 heads in 13 flips, shows the other end. Its deviation

"The other end" of what is not said (small $n$), and "the chapter's coin"
assumes the reader came through the chapter introduction. Draft: "At small
$n$ no bound helps. The coin of the chapter introduction landed heads 9
times in 13 flips, a deviation of $t=2.5/13$ from $\tfrac12$ ..."

**CON-9** (moderate, correctness; fairly sure). "Sub-Gaussian and
Sub-Exponential Variables", Bernstein's interval.
> Gaussian interval is $\pm0.27\%$, so Bernstein gives up a factor of $1.4$ in

Bernstein's inequality needs a bound $\sigma^2$ on the true variance. The
cell computes `var = p * (1 - p)` from the measured error rate, so the
half-width $\pm0.39\%$ is not "valid at every $n$" in the sense claimed for
Hoeffding, which needs nothing but the range. This is the same substitution
of $\hat\sigma$ for $\sigma$ that the section criticized in the Gaussian
interval. The repair is cheap and worth a sentence because it shows the
habit of mind: the true rate is at most $\hat p+t$, so solve with variance
$(\hat p+t)(1-\hat p-t)$. I computed the self-consistent half-widths:
$\pm0.43\%$ instead of $\pm0.39\%$ at 2%, and $\pm0.79\%$ instead of
$\pm0.76\%$ at 8.3% (ignoring the small cost of using the bound twice). The
conclusion stands, the claim becomes true. Either add this, or say plainly
that the comparison assumes the variance known.

**CON-10** (moderate, explanation). "Sub-Gaussian and Sub-Exponential
Variables", notation.
> stops working. A random variable $X$ is **sub-Gaussian** with *variance

The letter $\sigma^2$ is the true variance in Chebyshev's bound, the
variance proxy here, and the true variance again in Bernstein's inequality
40 lines later. For the 0/1 loss the two differ ($p(1-p)$ against the proxy
$\tfrac14$ from Hoeffding's lemma), and that difference is the whole reason
Bernstein exists. Sharing the letter hides the point. Say in one sentence
that the proxy is never smaller than the variance (compare the two sides of
:eqref:`eq_mdl-subgaussian` to second order in $\lambda$) and can be much
larger, and consider a different symbol for the proxy. The parameters
$(\nu,\alpha)$ of the sub-exponential definition also arrive unexplained;
"$\nu^2$ is the proxy near zero, $1/\alpha$ the width of the window" is said
only after the display and only in part.

**CON-11** (minor, thought). After Bernstein's inequality.
> a bounded variable sub-exponential with $\nu^2=2\sigma^2$ and $\alpha=2M/3$.

This sentence and the next ("Bernstein's inequality is thus an instance of
... up to constants") are a remark for the expert. The reader cannot use
them, and exercise part (d) asks for exactly this. Cut both sentences from
the text.

**CON-12** (minor, thought). End of part one.
> work, and there the variance-aware bound matters most.

True, and it is the place to say what the section's opening question has
become: for the 8.3% classifier the answer is now $\pm0.76\%$, not
$\pm1.36\%$. The running example is dropped here without its final number
being named as the answer.

**CON-13** (moderate, correctness; sure). After the norm-concentration
proposition.
> Chernoff parameter reaches the edge of its window and the tail becomes

The proposition is about the norm, and this sentence says that for
$\varepsilon>1$ "the tail becomes $e^{-d\varepsilon/8}$". That is the tail
of the squared norm $\|\mathbf x\|^2/d$. For the norm itself the Gaussian
rate persists: $z-1\geq\varepsilon$ means $z^2-1\geq\varepsilon(2+\varepsilon)$,
so the same computation gives $e^{-d\varepsilon(2+\varepsilon)/8}\leq
e^{-d\varepsilon^2/8}$ for every $\varepsilon>1$, and the restriction
$\varepsilon\leq1$ in :eqref:`eq_mdl-norm-concentration` is not needed for
the upper tail (the lower tail is empty for $\varepsilon>1$). The further
reading says the same from the other side: the norm is 1-Lipschitz and has a
Gaussian tail at every scale. As written, a reader concludes that the norm
has a heavier tail far out, which is false. Fix: "For the squared norm the
tail beyond $\varepsilon=1$ is only exponential, $e^{-d\varepsilon/8}$. The
norm keeps its Gaussian tail, because a norm exceeding $(1+\varepsilon)\sqrt d$
means a squared norm exceeding $(1+2\varepsilon+\varepsilon^2)d$."

**CON-14** (minor, thought). "The Thin Shell".
> The thin shell is the simplest instance of *concentration of measure*: a

The term is named and never used again. Either connect it (McDiarmid's
inequality in :numref:`sec_mdl-generalization-bounds` is this statement made
precise, and the Lipschitz version is in the further reading) or cut the
paragraph.

**CON-15** (moderate, pedagogy). "Near-Orthogonality of Random Directions".
> For vectors of random signs the statement holds without approximation. Let

The argument is told twice: first for Gaussian vectors, ending in the
hand-wave "probability of order $e^{-d\varepsilon^2/2}$", then for sign
vectors, exactly. The exact case uses Hoeffding's inequality just proved and
a norm that is exactly $\sqrt d$; it is the natural first case. Lead with
it, and give the Gaussian case one sentence: the inner product with a unit
vector is a standard Gaussian, and the norm is within a few percent of
$\sqrt d$ by the thin shell, so the same rate holds approximately. The
subsection also opens by restating the linear algebra result with its
Chebyshev bound; one clause with the reference is enough.

**CON-16** (minor, correctness; sure). "Near-Orthogonality of Random
Directions".
> At $d=784$ and $\delta=0.05$, about 570 directions are pairwise

570 and $10^7$ are what a loose union bound guarantees, not estimates of how
many nearly orthogonal directions there are. "About 570 directions are
pairwise within" reads as the latter. Write "the bound guarantees at least
570 directions with all pairwise $|\cos|<0.2$, and $10^7$ with
$|\cos|<0.3$".

**CON-17** (moderate, thought). "Consequences for Deep Learning", distances.
> points. A union bound turns one exponential tail into a guarantee for many

This sentence and the one after it (Chebyshev would need $d$ proportional to
$n^2$) are the thesis of the next section, which states it again for
classes ("A polynomial tail makes the sample size grow like
$|\mathcal F|$, an exponential tail like $\log|\mathcal F|$"). Per S5 the
display above it repeats the pair count of the previous subsection. Reduce
the paragraph to: distances between $n$ independent Gaussian points all lie
in $(1\pm\varepsilon)\sqrt{2d}$ once $d$ exceeds a multiple of $\log n$ (by
the count used for directions), the measured contrast shows the effect,
real embeddings escape it because of low intrinsic dimension, and the
exercise turns it into a construction.

**CON-18** (moderate, explanation). "Consequences for Deep Learning",
initialization.
> approximation. A variance is an average over draws of the weights, and

This is the idea of the paragraph (an expectation over initializations
against one drawn network) and it is compressed until it is hard to parse.
The sentence before it also mixes two approximations (input coordinates of
unit variance, and $\|\mathbf x\|^2/d\approx1$). Draft: "The initialization
argument computes an expectation over all draws of the weights. We train
one draw. Concentration says when the two agree: for one layer of width
$d$ the squared norm of a single draw is within about $\sqrt{5/d}$ of its
expectation."

**CON-19** (minor, language). Closing paragraph.
> We can now size a test set from a required precision and failure

"We can now" closes this section, the previous one ("We can now estimate
any posterior expectation") and the next ("We can now say how much more
data"). The guide names this formula as one to avoid. The teaser that
follows gives away the next section's opening experiment, number included
(66% heads); that section then tells it from the start. End on the
distinction (fixed before the data against chosen after) and the question,
without the number.

**CON-20** (minor, language). Summary. Eight sentences that list every
result. The guide asks for three to six that answer what, why, how, and
where it is used. The last question (initialization, embeddings, test-set
sizing) is missing from the summary although the introduction promised it.

## Exercises

Ten problems, about 26% of the body, graded from routine to a full proof.
This is a good set. The best are "Comparing two classifiers on one test set"
(it answers the chapter introduction's question whether a second model that
scores a little higher is really better, and deserves a sentence in the text
pointing to it), "Chernoff bound for coins" (closes the loop on the 0.0201),
"Activation norms through depth", and the Johnson–Lindenstrauss problem with
its question about which part of the bound is loose. I checked that the
two-classifier problem is answerable as posed (half-widths 0.63%, 0.88% and
1.24% at $q=0.05$, 0.10 and 0.20) and that the Bernstein problem's steps
close (the exponent is exactly $-nt^2/(2c)$; at $\sigma^2=\tfrac14$
Bernstein gives 1.37% against Hoeffding's 1.36%).

- **"Mode versus mass", parts (a) and (b)** ask the reader to show two facts
  the text derives in full under "The Thin Shell" (the ratio $e^{-d/2}$ with
  its value at 784, and the density $r^{d-1}e^{-r^2/2}$ with its peak). Cut
  both parts or replace them; parts (c) and (d) carry the problem.
- **"Hoeffding with unequal ranges", part (c)** repeats the Gaussian
  inner-product bound derived in the text. If CON-7 is adopted, part (a)
  becomes the text's and the problem reduces to (b); merge it into the
  random-signs problem.
- **"Sub-Gaussian random signs", part (a)** is one line from the averaging
  paragraph of the text. Part (b) is the real problem.
- **"Bernstein's inequality", part (d)** stays if CON-11 removes the
  sentence from the text.

Nothing obvious is missing. A short problem on the empirical-variance
version of Bernstein would follow from CON-9 but is not needed.

## Still open from October

The October review covered this file before it was divided in two. Its
complaint about hedges ("about nine hedges") and a broken sentence no longer
applies; I found neither. Its cut list (initialization and nearest-neighbour
material, Bernstein) was not carried out. I would keep Bernstein and
initialization and shorten the nearest-neighbour paragraph (S5, CON-17). Its
remark that near-orthogonality is taught three times across the linear
algebra and probability chapters is reduced but not gone: the subsection
here still opens by restating the linear algebra result and then proves the
tail twice (CON-15).

## For the chapter review

- **Two sections, not one told twice.** This section and
  :numref:`sec_mdl-generalization-bounds` divide cleanly at "a quantity fixed
  before the data" against "a function chosen after it", and that is the
  right boundary. The overlap between them is local and easy to remove: the
  best-of-1,000-coins teaser at the end of this section, the Chebyshev
  against Hoeffding contrast under a union bound (told here for distances,
  there for classes), the definition of the union bound (given in both), and
  the introduction's many-models motive. Each should have one home; for all
  four it is the next section. The serious duplication of the pair is with
  the main text, and it sits in the other section.
- **Fit with the chapter.** It assumes Chebyshev and the three-point law from
  :numref:`sec_mdl-random_variables` and the Gaussian interval from
  :numref:`sec_mdl-statistics`, and says so in its first paragraph. It could
  lean on :numref:`sec_mdl-exponential-family` for the log-partition
  function (CON-5) and does not. The previous section hands over with "how
  far such an average can be trusted", which this section answers for
  independent draws only; a clause noting that MCMC averages are not
  independent would keep that promise honest.
- **Against the main text.** The Hoeffding statement, the 18,445 and the
  9,604 are in :numref:`chap_classification_generalization`; here they are
  proved and cited in one sentence, which is the right relation between
  appendix and main text. This section is deeper than the main text
  throughout.
- **Suspected chapter-wide patterns.** Sections close with a "We can now ..."
  paragraph that restates results ahead of the summary; the boilerplate
  sentence about NumPy and framework tabs; an imports cell that loads
  unused framework modules; exercise parts that ask for facts the text has
  already derived; $\sigma^2$ carrying several meanings.
- **Correctness.** No errors in statements or proofs. Two statements need
  repair: the tail of the norm beyond $\varepsilon=1$ (CON-13) and the
  variance assumed known in Bernstein's interval (CON-9). One wording
  overstates a bound as a count (CON-16).
