# Integral Calculus

Source: `chapter_mdl-calculus/mdl-integral-calculus.md` (lines 1 to 1607, up to
the slides marker). About 7,100 words of prose and mathematics in the body and
summary, plus about 1,900 words in 12 exercises; 13 code cells (two of them
with four framework tabs).

## Verdict

This is one of the better sections a reviewer could hope to find in a
mathematics appendix. It opens on a question the reader can feel (the bell
curve has height 0.40 at zero, yet $P(Z=0)=0$; where do $1/\sqrt{2\pi}$ and
$0.68$ come from?), keeps one line of argument for 1,300 lines, and answers
both halves of the question with a proof and a number. Almost every cell
follows the method the Basics review asked for: predict a number by a short
argument, run the code, read the number. I recomputed every worked value,
closed form and exercise answer I could (list at the end) and found no
mathematical error of substance, only four loose statements. Against Cover and
Thomas it falls short in three ways. It is two sections in one (integration,
then Monte Carlo and gradient estimators), at roughly twice the guide's length
ceiling. Its first third re-teaches school calculus to a reader the chapter
introduction says already has it. And the prose, although clean, refers to its
own layout constantly ("the opening constant", "the smallest case", "the
Gaussian integral subsection"), which reads as stage directions and fails the
reader who lands mid-section from a reference. The single most important fix
is structural: split the section after the Gaussian integral, and decide
whether the second half lives here or in chapter 41, which already proves the
same $\sigma^2/n$ result.

## Scores

| Dimension | Score |
|---|---|
| Clarity of explanation | 4 |
| Clarity of thought | 4 |
| Pedagogy | 4 |
| Language | 4 |
| Ease and enjoyment | 3 |

## What works

- **The opening** (lines 11 to 24). "Its value at $z=0$ is $0.40$, yet
  $P(Z=0)=0$" is a real puzzle, stated in two sentences, and the section pays
  it off twice (`eq_mdl-normal-normalizer` and the $\operatorname{erf}(1/\sqrt2)=0.683$
  line). Keep it.
- **Predict, then run.** The left-rule bias $\tfrac\epsilon2(f(a)-f(b))$ is
  predicted once and then explains four separate outputs: the $0.2\,\epsilon$
  column, the $\epsilon/4$ gap in the cumulative check, the residue $1.0005$
  in the improper integral, and the unexpectedly good grid sum of the
  Gaussian ("for an even integrand on a symmetric interval the left rule's
  first-order bias ... vanishes"). The three estimator standard deviations
  $2$, $7.18$, $4.90$ are derived in closed form and the stored output
  reproduces them (2.000, 7.116, 4.842). This is the book's method at its
  best.
- **The Gaussian integral in three steps**, with the remark "This factor $r$
  is the missing factor $x$ of the substitution example." That one sentence
  ties the substitution subsection to the polar trick and explains why the
  detour works. The paragraph "Three points make the argument rigorous" is
  honest about what the three-step argument skipped, in four sentences.
- **Counterexamples that are sharp and short**: $\int_{-1}^1x^{-2}\,dx=-2$ by
  careless evaluation; $(x^2-y^2)/(x^2+y^2)^2$ with iterated integrals
  $\pm\pi/4$; the uniform density on $[0,\theta]$, used three times (Leibniz
  boundary term, a score with nonzero mean, exercise 8); the indicator
  $\mathbf 1(x>0)$ where the pathwise estimator returns $0$; and
  $\sin(\omega x)$, where the score estimator wins. The last one prevents the
  usual folklore ("pathwise always has lower variance") and is the best
  passage of the second half.
- **Quotable sentences.** "Under the bell weight, multiplying by $z$ has the
  same effect as differentiating." "The **Monte Carlo** estimate is a Riemann
  sum whose slices are placed by chance". "The gradient of log-sum-exp is the
  softmax; the gradient of log-integral-exp is an expectation." Protect these.
- **Figures** carry weight and are referenced where used. The FTC sliver, the
  polar cell and the Leibniz boundary strip each replace a paragraph.
- **Hypotheses are stated and then shown to matter**, which is rare. Fubini
  gets absolute integrability and a counterexample; the pathwise identity gets
  a differentiability condition and a counterexample.

## Structural findings

**S1. The section is two sections (major).** Lines 82 to 836 are integral
calculus: Riemann sums, the fundamental theorem, improper integrals, parts,
substitution, multiple integrals, the Gaussian integral. Lines 838 to 1284 are
probability and stochastic optimization: densities, expectations, Monte Carlo
error, grids against sampling, the Leibniz rule, score-function and pathwise
estimators, baselines, a variance comparison. Each half has its own running
example ($x/(1+x^2)$; then $\mathbb E[X^2]$ with $X\sim\mathcal N(\theta,1)$),
its own motivating question and its own closing. The body is about 7,100
words against the guide's hard ceiling of 4,500 (section 4.4). Split after the
Gaussian integral. The second section ("Expectations and their gradients", or
similar) then opens on the training problem that currently appears only at
line 993, which is also what the October review complained about.

**S2. The second half depends on chapter 41, which comes later and repeats
it (major).** The proposition on the variance of the sample mean uses
variance, independence and covariance, none of which the appendix has defined
by chapter 39. `chapter_mdl-probability-statistics/mdl-random-variables.md`
(lines 1125 to 1145) proves the identical result as `eq_mdl-var_mean` and
draws the identical Monte Carlo conclusion, and its "Probability as Area" and
"Densities" subsections re-teach the density definition given here. The
change-of-variables theorem is also stated twice (here and at line 1381
there). Decide on one home for each: (a) densities, expectations and the
$\sigma/\sqrt n$ rate belong in chapter 41, and this section should use them
by reference; (b) the gradient estimators have no better home than here
(`mdl-bayesian-computation.md:1134` cites them) but they then need a one-line
statement of what is borrowed from chapter 41. A front-to-back reader
currently meets $\textrm{Var}$ in a proposition 2 chapters before its
definition.

**S3. The first third is below the stated reader (moderate).** The chapter
introduction says "We assume school calculus and the Basics part". Riemann
sums of $x^2$ with the formula for $\sum i^2$, signed area with "one flip, two
flips", the fundamental theorem with a full squeeze proof, parts and
substitution with a sliver argument: all of this is first-year calculus,
taught well, at full length (about 2,900 words, lines 82 to 600). Later the same section quotes
dominated convergence and asks the reader to implement REBAR's toy problem.
The level is not consistent. A reader with college calculus needs the
definition, the theorem and its one-picture proof, and the two reversal rules;
that fits in half the space. Candidates to cut or compress: the "smallest
case" derivation (keep the result and the $1/N$ error); the Signed Area
subsection (two sentences in the definition); the cumulative-sum check
(see IC-9); the second paragraph on the Dirichlet function (one clause and the
pointer to further reading).

**S4. Why a deep learning reader needs each topic is stated unevenly
(moderate).** The introduction motivates the section with VAEs and policy
gradients, and the last third delivers. In between, improper integrals, parts
and substitution each do have a concrete use that the text names (heavy
tails; Stein and score matching; the density change under a map), but
Fubini's theorem, the Riemann-integrability discussion and the left-rule error
analysis do not. Fubini needs one sentence of use (marginals, and the
exchange of expectation and integral that `mdl-divergences-distances.md:1261`
relies on). The quadrature error analysis is numerical analysis for its own
sake; it earns its place only as the foil for Monte Carlo, and the section
should say so when it first appears.

**S5. Two framework-tabbed cells do work that needs no framework
(moderate).** The only cells with four tabs check $G'=f$ on a grid and check
that $\partial_\theta(\theta+z)^2=2x$ ("max |pathwise - 2x| = 0.0"). Both
verify one-line derivatives. Because of them the section imports four
frameworks, enables x64 in JAX, and shows the reader a tab strip twice in a
section that is otherwise pure NumPy. Either drop both (the prose derivations
suffice) or give the second one a job: differentiate a $g$ the reader cannot
do by hand. This changes cells, so it is deferred under the author's decision
on experiments; the prose can already stop presenting them as evidence.

**S6. Closing paragraph and summary say the same thing (minor).** The
paragraph beginning "We can now evaluate an integral exactly" (line 1276) is a
summary, and the Summary section follows eight lines later with the same
content. Guide section 7.7 lists exactly this. Keep only the last sentence of
the closing paragraph (what remains open: the constant $\sigma$, and where
the book takes it up) and move it to the end of the summary.

**S7. What is missing.** (a) The multidimensional integration by parts
(divergence theorem), which score matching and the Fokker-Planck derivation
actually use; `mdl-fokker-planck-probability-flow.md:251` has to say "the
one-dimensional rule". One paragraph after `eq_mdl-stein-1d` would do.
(b) A sentence connecting `eq_mdl-change_var_nd` to normalizing flows by name,
since that is the deep learning use; the text says only "tells how a density
changes under a map". (c) Importance sampling is alluded to in the last
sentence but not named; name it.

## Passage-level findings

**IC-1** (moderate, language / explanation). Whole section; 11 instances.
Anchors: "the opening constant" (4 times), "the plan of the opening reaches it
through", "which answers the first half of the opening question", "the puzzle
of the opening has a precise form", "opening probability into an error
function", "the density $p$ of the opening", "the second half of the opening
question". Related: "the smallest case" (4), "The Gaussian integral subsection
obtains", "in the subsection on densities", "the training problem".
The text names things by where they stand in the section instead of by what
they are. A front-to-back reader has to remember what "the opening constant"
was 700 lines ago; a reader who arrives at the Gaussian integral from a
reference in chapter 41 has never seen the opening. Fix: name the object.
"The opening constant" becomes "the normalizer $1/\sqrt{2\pi}$ of the bell
curve"; "the smallest case" becomes "$\int_0^1x^2\,dx$"; "the Gaussian
integral subsection" becomes a `:numref:` or `:eqref:`. One back-reference to
the opening, at the point where $0.683$ appears, is enough.

**IC-2** (minor, explanation). Opening. Anchor: "The earlier chapters used
continuous random variables without asking how".
In chapter 39 of the appendix, "the earlier chapters" reads as chapter 38
(linear algebra), which used none. Write "The main text used continuous
random variables ..." and name one place by label.

**IC-3** (moderate, language). Opening, roadmap paragraph. Anchor: "Along the
way the section answers both halves of the opening question: the constant".
The roadmap paragraph (lines 26 to 36) is eleven lines that preview every
subsection, then explain where each half of the question will be answered.
The reader has not yet seen the terms ("no elementary reverse", "average
random samples", "the same two tools"), so the preview cannot be understood
on first reading and is not needed on second. Cut to three sentences: slices
and their limit; the fundamental theorem and why it fails for the bell curve;
sampling and the derivative of an expectation. Drop the sentence quoted.

**IC-4** (moderate, pedagogy). The Definite Integral, Riemann Sums. Anchor:
"Both facts follow from a".
The left-rule bias $\tfrac\epsilon2(f(a)-f(b))$ is the most used tool in the
first half (it explains four outputs), yet its derivation is delegated to
exercise 1(a). The derivation is two lines: on a slice, $f(x)\approx
f(x_i)+f'(x_i)(x-x_i)$, so the slice integral exceeds the rectangle by
$\tfrac{\epsilon^2}2f'(x_i)$, and summing gives
$\tfrac\epsilon2\int f'=\tfrac\epsilon2(f(b)-f(a))$, by the fundamental
theorem. Put it in the text, after the fundamental theorem where it becomes
available, and let the exercise ask for the midpoint rule only. As written,
the reader is told a formula "follows" from an expansion they have not seen,
using a theorem that comes a subsection later.

**IC-5** (minor, explanation). Riemann Sums. Anchor: "integrable, and so is
every bounded function with finitely many jumps".
The citation for these two facts is Folland, a graduate measure theory text.
A reader who follows it will not find a statement at this level. Cite Rudin's
*Principles* (already in the bibliography as `Rudin.1976`) for Riemann
integrability and keep Folland for Fubini and dominated convergence.

**IC-6** (minor, thought). Riemann Sums. Anchor: "Not every bounded function
has an integral. The function that is $1$ on the".
The Dirichlet function is hygiene with no later use in this section; the only
function in the book it is contrasted with is none. The October review listed
"Dirichlet functions" under real-analysis hygiene. Reduce to one clause
("some bounded functions, such as the indicator of the rationals, have no
Riemann integral; the Lebesgue integral of
:numref:`sec_mdl-calculus-further-reading` repairs this") or cut.

**IC-7** (minor, pedagogy). Signed Area. Anchor: "Each flip, reflecting across".
The flip-counting ("one flip", "two flips, which cancel") spends a paragraph
and a figure on $\int_b^a=-\int_a^b$. For this reader, the convention and its
reason ("keeps additivity true for every order of $a$, $b$, $c$", which is
well put) are enough. The last paragraph of the subsection, on why the
$n$-dimensional rule has an absolute value, is the part worth keeping and
could move to where that rule appears.

**IC-8** (minor, language). The Theorem. Anchor: "$G(x)=\tfrac12\log(1+x^2)$
the derivative". The sentence "the chain rule gives $G(x)=\ldots$ the
derivative $G'(x)=\ldots$" does not parse on first reading (gives what to
what?). Write "For the running example take $G(x)=\tfrac12\log(1+x^2)$. By
the chain rule $G'(x)=\ldots$".

**IC-9** (moderate, pedagogy). The Theorem. Anchor: "We can check $G'=f$ on
all of $[0,2]$ without repeating the derivation."
Two cells follow. The first uses automatic differentiation to confirm a
derivative just computed by hand in one line; it prints `5.55e-17` and
teaches nothing about integrals (see S5). The second (cumulative sums against
$G(x)-G(0)$) does demonstrate the theorem, and its reading ("largest where
$f$ peaks, at $x=1$ ... which gives the printed $\tfrac\epsilon4$") is
excellent. Keep the second, drop the first.

**IC-10** (minor, correctness; confident). Improper Integrals. Anchor: "A
**heavy-tailed** density is one whose tail is not bounded".
The standard definition is on the tail probability: $X$ is heavy-tailed if
$e^{\lambda x}P(X>x)$ is unbounded for every $\lambda>0$ (equivalently the
moment generating function is infinite for every $\lambda>0$). Stated on the
density, as here, a density with ever thinner tall spikes on a light tail
would qualify. State it on $P(X>x)$, or say "we call a density heavy-tailed
if ..." and add "for densities with a monotone tail the two agree". The term is defined a second time in
`mdl-random-variables.md:702` ("when its tails decay more slowly than every
exponential"), there on the tails. Keep one definition (that one) and cite
it from here.

**IC-11** (minor, correctness; confident). Improper Integrals. Anchor:
"neither change touches the other".
Not exactly: the discretization bias is
$\tfrac\epsilon2\bigl(f(0)-f(b)\bigr)=\tfrac\epsilon2(1-e^{-b})$, which
depends on $b$, as the sentence two lines above itself shows with the arrow
"$\to\tfrac\epsilon2$". The stored output shows it too: the gap between the
two columns is $0.000316$ at $b=1$ and $0.000500$ at $b=20$. Write "to
leading order the two errors add: a larger $b$ removes the first, a smaller
$\epsilon$ the second".

**IC-12** (minor, language). Improper Integrals. Anchor: "the area out to
$b$, followed as $b$ recedes". "Followed as $b$ recedes" is an odd phrase for
a limit. "The area out to $b$, in the limit of large $b$."

**IC-13** (minor, correctness; confident). Integration by Parts. Anchor:
"integration by parts on $[-b,b]$ and $b\to\infty$ give".
Two small gaps. The paragraph above (Improper Integrals) insisted that a
whole-line integral needs both one-sided limits and that "a symmetric limit
is not enough"; here the symmetric limit is used without comment. And the
condition $g\,w\to0$ alone does not make the two integrals converge. Add the
hypothesis "and both integrals converge" (true for every polynomial $g$, the
only case used), and take the limits on $[-a,b]$.

**IC-14** (moderate, explanation). Substitution. Anchor: "A thin slice
explains the factor. In the $x$-integral on the right of".
The sliver paragraph is hard to follow because it names sides of an equation
("on the right of", "on the left", and the caption's "the $x$ side (the
right-hand side of the formula)") instead of saying what is being compared.
The idea is simple: a slice of width $\epsilon$ in $x$ covers a slice of
width $\epsilon\,u'(x)$ in $y$, at the same height $f(u(x))$. Draft: "Cut
$[a,b]$ into slices of width $\epsilon$. The map $u$ sends the slice at $x$
to a slice of width about $\epsilon\,u'(x)$ at $y=u(x)$, where the integrand
has height $f(y)=f(u(x))$. Its area, $f(u(x))\,u'(x)\,\epsilon$, is what the
$x$-integral must add up." The proof by the chain rule already precedes it,
so this paragraph is intuition and should be the shorter of the two.

**IC-15** (minor, thought). Substitution, last paragraph. Anchor: "If $u$
decreases, then $u(b)<u(a)$, so the limits on the left of".
The same point (signed form, unsigned form, which one generalizes) is made
three times: here, in Signed Area ("In one dimension the order of the limits
carries the sign"), and after `eq_mdl-change_var_nd` ("The formula uses
$|\det\mathbf J_{\boldsymbol\phi}|$ and not"), and then again in exercise 5.
Keep it once, where the $n$-dimensional formula is stated, and let the
exercise do the rest.

**IC-16** (minor, explanation). Double Integrals and Fubini's Theorem. Anchor:
"This section quotes three theorems without proof, all from".
This useful statement of candour is buried in the middle of the paragraph
that introduces Fubini, between "it needs a hypothesis" and the hypothesis.
Move it to the start of "Multiple Integrals", as its own sentence, so that
the Fubini paragraph runs: reordering is trivial for finite sums; in the
limit it needs absolute integrability; statement.

**IC-17** (minor, correctness; confident). Change of Variables in Many
Dimensions. Anchor: "= \int_{\Omega} f(\boldsymbol{\phi}(\mathbf{x}))\,".
The formula uses $\mathbf x$ for the variable on both sides: on the left
$\mathbf x\in\boldsymbol\phi(\Omega)$, on the right $\mathbf x\in\Omega$. It is
formally correct (dummy variables) but the explanation that follows ("A tiny
cube of volume $d\mathbf x$ at $\mathbf x$ is sent to ...") then has the same
symbol in the domain and the image. The one-dimensional rule used $y$ and
$x$, and `mdl-random-variables.md:1387` writes the same theorem with
$\mathbf x$ and $\mathbf y$. Use $\mathbf y=\boldsymbol\phi(\mathbf x)$ on the
left here too. Also, in step II of the Gaussian integral the map is applied
to $(r,\theta)$, a third naming.

**IC-18** (minor, explanation). Change of Variables. Anchor: "Read with $f$ a
density, the same factor tells how a density changes under a map".
This is the deep learning use of the theorem (normalizing flows, the
reparameterized density of a VAE) and it gets half a sentence and a pointer.
Say it in a full sentence and name flows; see S7(b).

**IC-19** (minor, explanation). The Gaussian Integral. Anchor: "The value over
the whole line". The paragraph introduces the error function, its scale, the
library call, and only then says that the value over the whole line needs a
detour. The reader wants the plan first. Reorder: no elementary
antiderivative; the value over the whole line can still be found, by a detour
through two dimensions; the antiderivative itself is named erf and is in
`math.erf`.

**IC-20** (minor, correctness; confident). The Gaussian Integral, after the
cell. Anchor: "by $b=3$ the volume outside the square is below $10^{-3}$".
True but loose: the output gives $\pi-3.141454=1.4\times10^{-4}$. Say "about
$10^{-4}$".

**IC-21** (moderate, thought). Integration in Probability, Monte Carlo
Estimation. Anchor: "**Proposition (Variance of the sample mean).**".
See S2. In addition, the proposition's proof uses "the variance of a sum is
the sum of all pairwise covariances", a fact the appendix states only in
chapter 41. If the result stays here, say "we borrow two facts from
:numref:`sec_mdl-random_variables`" and state them; better, cite
`eq_mdl-var_mean` and delete the proposition.

**IC-22** (minor, thought). Monte Carlo Estimation. Anchor: "When $g(X)$ has
finite variance this is the weak law of large numbers proved in". Weak law,
strong law, which one the book proves, and under which moment condition: none
of it is used. One clause: "by the law of large numbers
(:numref:`sec_mdl-statistics`)".

**IC-23** (minor, thought). Monte Carlo Estimation. Anchor: "(The expected
absolute error is". A parenthesis explaining that the mean absolute error is
$0.8$ of the root-mean-square error is a distraction at the point where the
key message ("The rate does not mention the dimension of $x$") should land.
That message is the best sentence of the paragraph and is currently last,
after the parenthesis. Cut the parenthesis, or move it to where the table is
read.

**IC-24** (moderate, pedagogy; claim against stored output). Monte Carlo
Estimation, plot cell. Anchor: "Across four decades the measured error tracks
the slope $-\tfrac12$ of the".
The stored PyTorch figure does not show this clearly. The curve is the error
of one running average, so its values at neighbouring $n$ are strongly
correlated: it sits above the line from $n=10$ to $150$, then below it by up
to a factor of ten from $n=300$ to $900$, with three spikes down to $10^{-4}$
where the error changes sign. A reader sees a noisy curve near a line, not a
slope. The claim "random excursions on both sides" is also only loosely true
(most of the range is below, as it should be, since the mean absolute error
is $0.8\sigma/\sqrt n$ and the path is one draw). The honest experiment is the
root-mean-square error over, say, 200 independent runs, which lies on the
line; exercise 9 asks for exactly that with 20 repetitions. Under the
author's decision the cell change is deferred; the prose can be corrected now:
"A single running average wanders around the prediction, because its errors
at nearby $n$ share most of their samples; the envelope falls with slope
$-\tfrac12$." The table before it already makes the point adequately, so the
plot could also go.

**IC-25** (minor, explanation). The Leibniz Rule. Anchor: "The swap needs
$f(\cdot,\theta)$ integrable and $f(x,\cdot)$".
"The swap" is used seven times as a noun for differentiation under the
integral sign, from this sentence on, without being introduced as a name. On
first use write "this exchange of derivative and integral, which we call the
swap, needs ...", or use "differentiation under the integral" throughout.
The conditions paragraph itself is dense: three regimes in four sentences.
Since only the Gaussian case is used, state the dominated condition once and
say that it holds whenever a Gaussian factor multiplies a polynomial.

**IC-26** (minor, pedagogy). The Leibniz Rule. Anchor: "Consider a problem
from training." The motivating problem is good, but the Leibniz rule proper
(moving limits) is then needed only for the uniform counterexample; the
training problem needs only the swap. The subsection title and the
proposition make the moving-limit rule look like the main result. Retitle
"Differentiating under the integral", present the swap as the result, and the
Leibniz rule as the correction when the support moves.

**IC-27** (minor, thought). The Leibniz Rule. Anchor: "The swap also gives the
continuous version of this chapter's log-sum-exp."
A good link to the chapter's recurring function, placed as an afterthought at
the end of a subsection about boundary terms. It also uses "completing the
square gives $\log\mathbb{E}[e^{\theta Z}]=\theta^2/2$" without showing the
one line. Either give it its own short paragraph heading and the one line, or
move it to :numref:`sec_mdl-exponential-family`, where the same derivative is
$A'(\eta)$.

**IC-28** (minor, correctness; fairly confident). Score-Function and Pathwise
Gradients. Anchor: "Lipschitz $g$ that is differentiable almost everywhere
suffices".
A Lipschitz function is always differentiable almost everywhere
(Rademacher, cited in the further reading), so the second condition is
redundant, and the statement as a whole needs one more hypothesis: the
distribution of $T_{\boldsymbol\theta}(\mathbf z)$ must have a density, so
that the kinks are hit with probability zero. Write "(a Lipschitz $g$
suffices when $X$ has a density)".

**IC-29** (minor, explanation). Score-Function and Pathwise Gradients. Anchor:
"**The training problem, both ways.**" This paragraph delivers nine numbers
and three formulas in eleven lines, with the derivation of all of them sent
to exercise 11(b). The pathwise variance is immediate ($\textrm{Var}(2X)=4$)
and should be shown; for the score estimator, show the one expansion
$\mathbb E[(\theta+z)^4z^2]=\theta^4+18\theta^2+15$ so that the reader sees
where $\mathbb E[z^6]=15$ enters. I verified all three closed forms and
$b^\star=\theta^2+3$.

**IC-30** (minor, claim against stored output). After the cell
`integral-score-wins`. Anchor: "about $0.55$".
The stored output prints `sd 0.54`. The exact value is $0.545$, so runs print
either. Write "about $0.5$ against $0.7$" in keeping with the house rule on
precision that survives re-execution. Similarly "within a percent or two" for
the sample standard deviations is at the edge: the stored run is off by 0.9%
and 1.2%; "to within a few percent" is safe.

**IC-31** (minor, language). Pattern: the phrase "The cell" opens the
description of a cell eight times ("The cell below checks the formula",
"The cell refines the slices", "the cell compares it", "The cell evaluates",
"The cell computes the grid sum", "The cell checks the three numbers", "The
cell estimates", "The cell compares the two"), always in the order
describe-cell, cell, read-output. The better instances lead with the question
("How fast do the sums approach $\tfrac12\log5$?", "Do the squares exhaust
the plane as step I claims, and is the limit $\pi$?"). Make all of them
questions or predictions, and vary the verb.

**IC-32** (minor, language). Pattern: "In words" twice in one subsection
(Anchors: "In words, the integral is the value that the slice sums approach";
"In words: to integrate $f$, find a function whose derivative is $f$"). The
restatements are good; the label is not needed.

## Exercises

Twelve exercises, about 1,900 words. They are the strongest set I have seen in
this book: nearly all extend the text (GELU's slope range from
$\Phi'=\phi$; the volume of the unit ball from the Gaussian integral; the
tail formula $\mathbb E[X]=\int P(X>t)\,dt$ by Tonelli; the double cover in
$u(x)=x^2$) instead of repeating it, and I checked the answers of 1 to 8,
11(b) and 12(a, b). Problems:

- **Length and number.** Exercises 4(d), 9, 10 and 12 are small projects with
  five to eight questions each. 4(d) alone asks for a radial integral, a
  surface area, a volume, six evaluations, a limit, a grid count and a moral.
  Split 4(d) into its own exercise ("The volume of the unit ball"), which is
  also the one a reader will want to find.
- **Exercise 11 is mistitled and partly redundant.** "Estimator variance for
  a quartic" has four parts, of which (b) is about $X^2$ (the text's example)
  and (d) is about the indicator. Part (d) asks the reader to show exactly
  what the text has already shown in full, with the same $\theta=1.5$ and the
  same value $0.1295$. Cut (d) or reduce it to "verify by simulation".
- **Exercise 1(a)** carries the derivation the text relies on (IC-4); once
  that moves into the text, 1(a) reduces to the prediction question.
- **Exercise 12** reuses symbols: $b$ is the Bernoulli variable here and the
  baseline in the text; $z$ is the relaxed sample here and the Gaussian noise
  in the text; $\sigma$ is the logistic function here and a standard
  deviation in the text. Rename the Bernoulli variable and say "$\sigma$ is
  the logistic function" once. It also cites three papers inside an exercise
  statement; that framing duplicates the further-reading notebook
  ("Gradient Estimators beyond Score and Pathwise") and could shrink to one
  sentence.
- **Grading.** The marks ● and ●● appear on 9 to 12 only; exercises 4 and 8
  are at least as hard as 9.
- **Missing.** Nothing tests Fubini's failure (the $\pm\pi/4$ example) or the
  log-integral-exp derivative; one short exercise on the latter ("show that
  the second derivative is the variance under the tilted density") would tie
  to the exponential family.

## Still open from October

The October review's one specific complaint ("states its motivation 650 lines
in") is fixed for the first half: the section now opens with it. It is still
true of the second half, whose motivating problem first appears at line 993
(S1). "Real-analysis hygiene ... Dirichlet functions, Fubini counterexamples"
is partly open: the Fubini counterexample earns its place, the Dirichlet
function does not (IC-6).

## What I verified

Recomputed by hand or numerically: the $\sum i^2$ expansion and its three
values; the left-rule bias and all four outputs it explains; $\int_0^1
xe^{-x^2}=0.316$; the power-law tests at infinity and at zero; the Stein
moments 1, 3, 15; the Fubini counterexample ($\partial_y\,y/(x^2+y^2)=f$,
$\pm\pi/4$, divergence of $\iint|f|$); the polar Jacobian and all normalizers;
$\operatorname{erf}(1/\sqrt2)=0.6827$; $\sigma\approx0.2$ for $e^{-U^2}$; the
tie of exponents at $d=4$; the Leibniz uniform example; the tilted Gaussian;
the nonzero mean of the uniform score; the indicator example
($\varphi(1.5)=0.1295$); the three estimator variances and $b^\star$; the
$\sin$ example (true means, standard deviations $0.657$ and $0.545$ at
$\omega=1$, $2.12$ and $0.71$ at $\omega=3$). All stored PyTorch outputs agree
with the prose except as noted in IC-24 and IC-30. All `:numref:` targets and
citation keys exist. I rendered the ten figures; they match their captions
(the only blemish is the very small "$\epsilon$" and "$\epsilon\,u'(x)$"
labels in the left panel of `mdl-cal-cov-jacobian.svg`). Not checked: the
TensorFlow, JAX and MXNet tabs and their stored outputs.

## For the chapter review

- **Fit.** The section is the chapter's only one on integration and is
  self-contained within the chapter: it uses the derivative table, the mean
  value theorem and the multivariable chain rule by label, and the
  log-sum-exp thread once. It is the longest dependency of chapters 41 to 43
  (cited from random variables, maximum likelihood, Bayesian computation,
  divergences, Fokker-Planck, ODE solvers).
- **Order problem across chapters.** Its second half is probability, placed
  two chapters before probability, and duplicated there (S2). The chapter
  review should decide where densities, Monte Carlo error and gradient
  estimators live; my recommendation is a split into two sections, with the
  second trimmed against chapter 41.
- **Level.** The first third is school calculus at full length while the
  chapter introduction assumes school calculus. If the other sections of the
  chapter show the same pattern (the October review said "first-year calculus
  with proofs"), it is a chapter-level decision how much of that to keep.
- **Pattern to check elsewhere:** positional self-reference ("the opening
  constant", "the smallest case", "the X subsection") in place of names and
  labels (IC-1). "The smallest case" also occurs in
  `mdl-random-variables.md`, so it may be a part-wide coined phrase.
- **Pattern worth spreading:** one predicted quantity that explains several
  outputs (the left-rule bias; the three standard deviations).
- **Framework tabs** contribute nothing here (S5); the section could be
  framework-free.
