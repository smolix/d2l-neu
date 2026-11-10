# Chapter 39, Calculus and Automatic Differentiation: writing review

Sources: the seven `.md` files of `chapter_mdl-calculus/`, each read in full by a section reviewer up to the slides marker. About 34,000 words of prose and 10,000 words of exercises (55 problems). This file is assembled from the section reviews; I opened the sources only to check the cross-section claims marked "checked" below.

Section reviews:
[index](39-calculus/index.md) ·
[single-variable](39-calculus/mdl-single-variable-calculus.md) ·
[multivariable](39-calculus/mdl-multivariable-calculus.md) ·
[matrix calculus](39-calculus/mdl-matrix-calculus.md) ·
[automatic differentiation](39-calculus/mdl-matrix-calculus-autodiff.md) ·
[integral](39-calculus/mdl-integral-calculus.md) ·
[further reading](39-calculus/further-reading.md)

## Verdict

This is a good chapter, correct almost everywhere, and in places at the bar: the dual-number passage, implicit differentiation with the unconverged solver, "Saddles in High Dimension", the Gaussian integral, and the comparison of score and pathwise estimators would stand in MacKay. The reviewers recomputed nearly every worked number, proof and exercise answer and found no error in a derivation or a proof; the correctness findings are two false or self-contradicting statements (MC-11, MVC-12), two loose statements in the further reading (FR-4, FR-2), and a dozen small imprecisions. The method is the right one and is applied in every section: one worked number, a prediction from a short argument, a cell that prints it, an explanation of the residue. What keeps the chapter from the bar is not explanation but economy and ownership. The same things are taught two to four times: the backward pass in three sections, the Hessian of log-sum-exp in three, the adjoint in three, the cost of probing in four places, and the descent lemma, Newton's convergence, the subgradient, densities and the variance of a sample mean again in chapters 40 and 41. Two sections are really two (multivariable, integral), the first third of two sections re-teaches school calculus that the introduction says is assumed, and every section ends twice and frames almost every cell as a question whose answer is known. The single most important decision is to assign each repeated result one home, inside the chapter and against chapters 40 and 41, before any section is revised.

## Scores

1 to 5 as the section reviewers gave them; 5 means the page could stand in Cover and Thomas or MacKay.

| Section | Explanation | Thought | Pedagogy | Language | Enjoyment |
|---|---|---|---|---|---|
| [index](39-calculus/index.md) | 4 | 4 | 4 | 4 | 4 |
| [mdl-single-variable-calculus](39-calculus/mdl-single-variable-calculus.md) | 4 | 4 | 4 | 4 | 3 |
| [mdl-multivariable-calculus](39-calculus/mdl-multivariable-calculus.md) | 4 | 3 | 4 | 4 | 3 |
| [mdl-matrix-calculus](39-calculus/mdl-matrix-calculus.md) | 4 | 4 | 4 | 4 | 3 |
| [mdl-matrix-calculus-autodiff](39-calculus/mdl-matrix-calculus-autodiff.md) | 4 | 4 | 4 | 4 | 4 |
| [mdl-integral-calculus](39-calculus/mdl-integral-calculus.md) | 4 | 4 | 4 | 4 | 3 |
| [further-reading](39-calculus/further-reading.md) | 4 | 4 | 4 | 4 | 4 |

Thirty of the thirty-five scores are 4, so the table hardly separates the sections; the verdicts do. Read from the prose, the order is: automatic differentiation first (its reviewer would give 5 for explanation and enjoyment to four of its subsections), then integral and matrix calculus, then single-variable, then multivariable, whose 3 for thought is a judgement on its order and not on its paragraphs. The integral section's 3 for enjoyment sits beside a verdict that calls it "one of the better sections a reviewer could hope to find"; that reviewer marked down for length and self-reference more heavily than the others did.

## Correctness

No proof or derivation is wrong. In order of severity:

1. **MC-11** (moderate; confidence high; checked at `mdl-matrix-calculus.md:584`). "This $1\times1$ check catches wrong shapes" says the opposite of what the check can do. With every matrix $1\times1$ there are no shapes, orders or transposes left; the check catches a wrong coefficient or a missing term. The slide title repeats the claim.
2. **MVC-12** (moderate; sure; checked at `mdl-multivariable-calculus.md:688`). The text teaches that $\partial f/\partial p$ is the edge derivative ($1$) and that the total sensitivity ($7$) is something else, then defines the adjoint as $\bar a=\partial f/\partial a$, "the total derivative". The defining equation is false for the node that made the distinction. The automatic differentiation section defines $\bar v=\partial y/\partial v$ the same way (`:377`), so the fix has to be made in both.
3. **FR-4** (moderate; confident). Quasi-Monte Carlo: the bound $N^{-1}(\log N)^d$ beats $N^{-1/2}$ only for $N$ exponentially large in $d$, not "once $N$ is large compared with" the dimension.
4. **MC-7** (moderate, in part correctness; confidence high). "$\mathbf W^K$ grows or shrinks like $|\lambda_{\max}|^K$" is the asymptotic rate, not a bound; it needs "for large $K$", and products of different factors are governed by singular values.
5. **FR-2** (minor; confident). The rate $RG/\sqrt T$ is attached to square-summable step sizes; it belongs to the constant step $R/(G\sqrt T)$, which violates that condition.
6. **AD-7** (minor; confidence high, code was run). The stated visiting order $y,r,t,u,v$ is not what `backprop` does ($y,r,t,v,u$). The adjoints in the table are right.
7. **IC-28** (minor; fairly confident). "Lipschitz $g$ that is differentiable almost everywhere" is redundant (Rademacher) and lacks the hypothesis that $X$ has a density.
8. **IC-10** (minor; confident). "Heavy-tailed" is defined on the density instead of on the tail probability, and differently from chapter 41.
9. **IC-13** (minor; confident). Stein's identity is derived with a symmetric limit the section has just ruled out, and without assuming that both integrals converge.
10. **MVC-18** (minor; sure it is inconsistent). The Lagrange remainder is restated with a hypothesis different from the one proved in the previous section and attributed to that section.
11. **MVC-10** (minor; sure). The heuristic before the Lagrange proposition uses differentiability of $g$ one paragraph before it is assumed.
12. **IC-17** (minor; confident). The change-of-variables formula uses $\mathbf x$ for the variable in the domain and in the image.
13. Single-variable exercise 9(a) (no ID; minor; sure) needs "three times continuously differentiable" to merge the two remainder points.

Stored outputs against prose, and numbers no cell prints:

- **IC-24**: "Across four decades the measured error tracks the slope $-\tfrac12$"; the stored figure shows one correlated running average that sits a factor of ten off the line for a decade.
- **IC-11** (minor; confident): "neither change touches the other"; the stored columns differ by $0.000316$ at $b=1$ and $0.000500$ at $b=20$.
- **IC-30**: prose "about $0.55$", output `sd 0.54`; "within a percent or two" is at the edge of what the store shows.
- **IC-20**: "below $10^{-3}$" where the output gives $1.4\times10^{-4}$.
- **SVC-1** (minor; sure): the gap "equals" $0.00125$; it is $0.0012495$, and the cell prints $0.1249$.
- **AD-5**: the cost of a gradient is "about three runs" in the opening, "at most five" by Baur and Strassen, and measured as two ($802$ against $401$), with no sentence saying that the three count different things.
- **SVC-15**: the Newton results for $x_0=1.3$ and for damping $\alpha=\tfrac12$ are quoted, but the cell runs only $x_0=1.4$. **SVC-18**: the values for $\beta=64$ refer to a figure that draws $\beta=1,4,16$.

Not verified by any reviewer: the Baur and Strassen constant 5 (AD review, "Could not check"); the proportions quoted from Bertoin et al. (FR-7, and single-variable exercise 12(b)); the TensorFlow, JAX and MXNet stores (only PyTorch was compared, plus JAX for the $|x|$ cell).

## Arc and order

The chapter has one line and the order of sections is right: the local model of a function in one variable and in many, the derivative of a layer and of a program, the integral and the derivative of an expectation. The introduction poses the two questions the chapter answers. The line is blurred in three places.

- **Multivariable calculus is two sections.** It finishes the local-model story (gradient, Hessian, critical points) and starts the derivative-of-a-computation story (sum over paths, backward pass), and it interleaves them: "Critical Points" promises a test that arrives 570 lines later, after 3,000 words on the chain rule (MVC-8, structural finding 1 of that review). Split it into "Gradients and the Chain Rule" and "Second Derivatives and Critical Points", or reorder so that critical points, Hessian, test, saddles and Lagrange multipliers form one movement.
- **Integral calculus is two sections**, at 7,100 words: integration up to the Gaussian integral, then densities, Monte Carlo and gradient estimators (S1). The second half is probability placed two chapters before probability, and it uses variance and covariance before the appendix defines them (S2, IC-21; checked at `mdl-integral-calculus.md:928`).
- **The boundary between matrix calculus and automatic differentiation leaks in both directions.** The two were evidently one section. Matrix calculus names the modes and counts their cost; automatic differentiation restates the product under a second label, counts again with other numbers, and re-derives the weight gradient (AD-3, AD-10; checked: `eq_mdl-jacobian-product` and `eq_mdl-ad-jacobian-product` both exist).
- **Single-variable calculus is half optimization.** Gradient descent, the descent lemma, Newton's method with its error recursion and subgradients take about 2,900 words, and chapter 40 proves all three again (structural finding 1 of that review, SVC-14).
- **Level.** The introduction says "We assume school calculus and the Basics part". The first 1,000 words of the single-variable section (SVC-2), "Partial Derivatives" (MVC-2) and the first 2,900 words of the integral section (S3) teach school calculus at full length; later the same sections quote dominated convergence and random-matrix theory. The introduction's second promise, that log-sum-exp "recurs throughout", is kept in parts of each section and dropped for the chain rule, the Lagrange example, and reverse mode (index review; MVC-11; AD-8).

If both splits are made the chapter has seven sections of 3,500 to 5,500 words, each with one question, which is the better shape.

## Most important findings

1. **Give the backward pass one account at three levels, each building on the last.** Multivariable keeps the scalar picture (sum over paths, edge against total, adjoint recursion); matrix calculus owns the Jacobian product, the cost by order of multiplication, and the layer rules; automatic differentiation owns seeds, the two engines, measured cost, memory and compositions. Cut "Two Orders of One Product" and "Parameter Gradients" to references. AD-3, AD-10, MVC-15, MC-12, AD-2, IDX-2.
2. **Fix the two statements that are wrong as written.** MC-11 (the $1\times1$ check) and MVC-12 (the adjoint written as a partial derivative). Both sit at the exact points where the chapter warns the reader about a common mistake.
3. **Split or reorder the multivariable section** so that critical points are not cut in two by backpropagation. MVC-8, structural finding 1 of that review.
4. **Split the integral section after the Gaussian integral and settle the second half against chapter 41.** S1, S2, IC-21.
5. **Decide what the single-variable section owns against chapter 40.** Keep the one-variable descent lemma and Newton as "minimize the parabola" with its divergence example; give the error recursion and the subdifferential one home. SVC-14, SVC-19, structural finding 1.
6. **Derive log-sum-exp's Hessian once.** It is derived in three sections (checked: `eq_mdl-lse-hessian`, `eq_mdl-softmax-jacobian`, `eq_mdl-ad-lse-hessian`), with the same "two checks against intuition" twice. AD-15, MC-17, MC-18.
7. **Make the checks able to fail, and stop asking questions with one answer.** MC-16 is the sharpest case: the cell that guards against a wrong transpose uses shapes for which a wrong transpose raises an error. Also SVC-4, MVC-4, AD-14, AD-16, SVC-22, IC-9, S5 of the integral review.
8. **Compress the school calculus.** SVC-2, MVC-2, S3, IC-7, and the real-analysis remnants SVC-12 and IC-6.
9. **Put the chapter's own function where the generic example now stands.** The Lagrange example as entropy-regularized maximization, whose solution is the softmax and whose value is log-sum-exp (MVC-11); reverse mode run on log-sum-exp (AD-8); the softplus for the best parabola (SVC-11).
10. **Replace positions by names and labels.** "The opening constant", "the smallest case", "the previous section", "below" (IC-1, MC-4, SVC-21, MC-13), and correct the main-text reference that sends readers to the wrong section for the layer rules (see Connection to the main text).

## Repetition

Inside the chapter, with a proposed home:

| Taught more than once | Where | Home |
|---|---|---|
| Cost of probing one weight at a time | introduction ("a hundred million"), multivariable opening ($10^6$) and twice more, automatic differentiation opening ($10^8$); also `sec_calculus` and `sec_autograd` (MVC-1, index review) | Automatic differentiation, whose opening numbers are paid off later ("the 30 ms of the opening"); the introduction keeps its one sentence; multivariable cites. The reviewers disagree here, see below. |
| Forward against reverse, and the cost argument | multivariable "Backpropagation", matrix calculus "The Order of Multiplication", automatic differentiation "Two Orders of One Product" and "Cost of the Two Modes" (MVC-15, AD-3) | Count for a chain by order: matrix calculus. Cost in sweeps and measured cost: automatic differentiation. Multivariable keeps only 24 against 12 multiplications. |
| Definition of the adjoint | `eq_mdl-adjoint`, matrix calculus ("call it the *adjoint*"), "The Tape" (MC-12, AD-2; checked) | Multivariable; the other two cite and extend. |
| Weight gradient as outer product | "Backward Rules of a Layer", "Parameter Gradients", `sec_backprop` (AD-10) | Matrix calculus, `subsec_mdl-layer-vjp`. |
| Hessian of log-sum-exp, $\mathbf H\mathbf 1=\mathbf 0$, variance argument | three sections (AD-15, MC-18) | Derivation: multivariable. Variance and covariance reading: matrix calculus. |
| Vanishing and exploding product $\mathbf W^K$ | matrix calculus, automatic differentiation closing (MC-7, AD-17) | Matrix calculus, after the product formula. |
| Counterexample $2xy/(x^2+y^2)$ | multivariable and matrix calculus (MC-4) | Multivariable; the second use is already a recall and needs only the label. |
| Signed against unsigned change of variables | three places and an exercise (IC-15) | Where the $n$-dimensional formula is stated. |
| Truncation against rounding in finite differences | single-variable exercise 9, multivariable exercise 7, `sec_calculus` | Multivariable exercise 7. |
| Count of forward against backward multiplications as an exercise | matrix calculus 4(b), 5(c), tensor-tape exercise (d) | One of them. |
| Survey facts (NP-completeness, recursive checkpointing, subgradient step sizes, forward gradients) | main sections and further reading | Further reading. |
| Classification of critical points; "a layer maps a vector to a vector"; the gradient of $\mathbf x^\top\mathbf A\mathbf x$ | five, four and four times within one section (MVC-19, MC-1, MC-9) | Once each. |

Against other appendix chapters: the descent lemma, Newton's quadratic convergence with damping, and subgradients (chapter 40; SVC-14); the Lagrange condition (stated here and in `mdl-constrained-optimization-duality.md`, which defers the proof here, an acceptable division); stable log-sum-exp (`subsec_mdl-stable-softmax`; MC-18); densities, the variance of the sample mean as `eq_mdl-var_mean`, the change-of-variables theorem and the definition of heavy tails (chapter 41; S2, IC-10, IC-17). Proposed: chapter 39 owns the one-variable descent lemma and the gradient estimators; chapter 40 owns Newton's convergence theory and the subdifferential; chapter 41 owns densities and the $\sigma/\sqrt n$ rate.

Against the main text: the difference quotient, the derivative table and the $\cos\theta$ argument for steepest descent (`sec_calculus`; SVC-2, MVC-5); the framework `jvp` and `vjp` calls (`sec_autograd`; AD-14); the outer-product rule (`sec_backprop`; AD-10); the fused softmax primitive (MC-18). In each case the main text keeps its version and the appendix recalls it in a clause.

## Recurring patterns

**Every body ends twice, with the same words.** All five sections close on a paragraph beginning "We can now ..." directly before a Summary that lists the same results (checked; SVC-20, MVC-22, MC-19, AD-17, S6). In each, the bridging sentences that follow the recap are good and should be all that remains.

**A question before every cell.** Eight in single-variable, seven in multivariable, fourteen in matrix calculus and automatic differentiation together (nine before eleven output cells in the latter); the integral section uses "The cell ..." eight times instead (IC-31). The form works where the reader could be wrong and flattens where the answer is "yes, the framework agrees" (SVC-22, AD-16).

**Verification cells that cannot fail.** Autograd confirming a hand derivative (SVC-4, MVC-4a, IC-9 and the pathwise cell in S5); a formula evaluated at its own roots (MVC-4b); four of five cells in matrix calculus (MC-16); the framework Jacobian printed a third time (AD-14). A plot-only cell remains at SVC-10. The cells that set the standard print a number the reader must think about: gap$/\epsilon^2$, the tenfold error ratio, $80{,}200$ against $802$, the unconverged solver, the three estimator standard deviations.

**Positions in place of names.** 17 "below" and "above" in single-variable (SVC-21); nine "the previous section" in matrix calculus (MC-4); eleven "the opening ..." and four "the smallest case" in integral (IC-1); "the first exercise" (MC-5). An appendix read by lookup needs labels.

**"The step that is not routine."** Four times in the chapter (checked), once on a step that is routine (AD-11, MC-20).

**Results restated in place.** Three to five statements of one result within a section (MVC-3, MVC-19, SVC-9, MC-1, MC-8).

**Notation drift.** $\partial$ for both edge and total derivatives (MVC-12); weights multiplying from the left and from the right, switching three times (MC-14); $m$ as output count, output width and mask (MC-14); $f$, $x$, $y$, $a$, $s$, $\varphi$ reused for unrelated objects across the first two sections (MVC-14); $\dot v$ introduced after forward mode is finished (AD-2); $\mathbf x$ on both sides of a change of variables (IC-17); $b$, $z$, $\sigma$ reused in integral exercise 12; machine precision written two ways in two exercises.

**Framework tabs** are light throughout and handled honestly (the JAX value for $|x|$, the MXNet notes). The integral section's two tabbed cells need no framework (S5), and one tab sentence repeats shared prose verbatim (AD-9).

## Connection to the main text

- **Wrong address** (checked at `chapter_multilayer-perceptrons/backprop.md:214`): `sec_backprop` says the appendix derives the layer rules "for general tensor operations" in `sec_mdl-matrix-calculus-autodiff`. They are derived in `subsec_mdl-layer-vjp` of the matrix calculus section.
- The body of the matrix calculus section has no reference to the main text, although it derives what `sec_backprop`, `subsec_softmax_and_derivatives` and `subsec_softmax-implementation-revisited` use (structural finding 3 of that review).
- `sec_backprop` mentions trading computation for memory with no pointer to `subsec_mdl-checkpointing` (AD-13).
- Two sections of the book are titled "Automatic Differentiation"; the appendix one needs a title that says what it adds. The attribution of the tape to Wengert should match `sec_autograd` (AD-6).
- Uses named without a label: "The earlier chapters used continuous random variables" (IC-2); remedies that "belong to optimization and to the design of architectures" (AD-17); convexity (SVC-17); "the section" in further reading (FR-5); normalizing flows, never named where the change of variables is stated (IC-18).
- Missing labels on likely targets: "The Descent Lemma"; "Backpropagation", "The Chain Rule as a Sum over Paths", "Symmetry of the Hessian", "The Second-Order Taylor Approximation" (lookup findings of the first two section reviews).
- Within the appendix: chapter 40 cites its own descent lemma and not this chapter's (checked in `mdl-convexity.md`); the Fokker-Planck section has to fall back on "the one-dimensional rule" because integration by parts in several dimensions is absent here (S7).
- The appendix is deeper than the main text nearly everywhere; the automatic differentiation section is the model of the relation (the main text calls the tool, the appendix builds it in forty lines).

## What to cut, what is missing

Cut or compress, about 4,000 words, none of it argument:

- Single-variable: the difference-quotient opening (SVC-2), the autograd check cell (SVC-4), the plot-only Taylor cell (SVC-10), the real-analytic digression (SVC-12), whatever moves to chapter 40.
- Multivariable: "Partial Derivatives" to three sentences (MVC-2), the recipe list and $\cos\theta$ derivation (MVC-5), "Tangent Planes", the general cost statement (MVC-15), the bullet list of cases (MVC-19).
- Matrix calculus: the index derivation (MC-9), the repeated motivations (MC-1, MC-8), the fused-primitive sentences (MC-18).
- Automatic differentiation: about 700 words of "Two Orders of One Product" and "Parameter Gradients" (AD-3, AD-10), the third derivation of the Hessian (AD-15), the duplicate Jacobian cell (AD-14).
- Integral: flip counting (IC-7), the Dirichlet function (IC-6), the roadmap (IC-3), the variance proposition in favour of `eq_mdl-var_mean` (IC-21), exercise 11(d).
- All five: the "We can now" recap.

Missing:

- Fermat's theorem stated as a lemma (SVC-7), and the power rule for real exponents, one line (SVC-5).
- The softmax as the solution of a constrained problem (MVC-11).
- The bias rule for a minibatch, and the differential of an elementwise function among the rules (matrix calculus structural finding 7, MC-10).
- Reverse mode run on log-sum-exp (AD-8); one sentence on why a program with branches is an expression (AD-1); why "adjoint" has its name (MC-12).
- The two-line derivation of the left-rule bias, which explains four outputs and is delegated to an exercise (IC-4).
- Integration by parts in several dimensions, normalizing flows by name, importance sampling by name (S7).
- What was found in training at kinks, in two sentences, instead of in an extended exercise (SVC-19).

## What is best and should set the model

- **One number explained three times.** The gap $0.00125$ in the single-variable section: measured, read as a Taylor coefficient, used in the descent lemma.
- **A prediction paid back 800 lines later.** The linear-model gap $1.786\times10^{-4}$ against the Hessian's $1.778\times10^{-4}$, with the residue accounted for.
- **One small network worked three ways** to the same numbers in the matrix calculus section, and the sentence "a gradient has the shape of its variable".
- **Dual numbers**, from $(3+\varepsilon)^2=9+6\varepsilon$ to $\varepsilon$ as a $2\times2$ matrix; **the non-vacuous test**; and **the unconverged solver**, an experiment whose result is not obvious.
- **"Saddles in High Dimension"**: a heuristic, the statement that it overestimates, the exact value, the reason, and the limit of the claim.
- **One predicted quantity that explains several outputs**: the left-rule bias and the three estimator standard deviations in the integral section.
- **Counterexamples that arrive with the hypothesis**: $|x|$ for the mean value theorem, the convex function on which Newton diverges, $\mathrm{ReLU}(x)-\mathrm{ReLU}(-x)$, the Fubini example with $\pm\pi/4$, $\sin(\omega x)$ where the score estimator wins.
- **Sentences that judge**: "Passes and work are different measures"; "Both directions are this one sentence"; "Calculus answers these questions, although it was created for others".
- **The exercises.** Every section review singles them out, three as the best set in the book: they extend the text, state a deep learning reason, and give numbers by which a reader can tell success.
- **The further-reading notebook** as a template for the other chapters.

## Where the section reviews disagree

- **Home of the probing argument.** The index review and MVC-1 put it in the chapter introduction; the automatic differentiation review makes that section its home and would leave the introduction "the question without the numbers". I side with the latter, since only that section pays the numbers off.
- **Home of the cost-by-order argument.** The multivariable review wants it in automatic differentiation, with matrix calculus "only" naming the order; the matrix calculus and automatic differentiation reviews agree that matrix calculus owns the product and the count. The table above follows the latter two.
- **"The step that is not routine."** The single-variable review praises it as a sentence with an author in it; MC-20 and AD-11 report it as a formula. Both are right: keep the two uses that name a real step.
- **A proposed exercise already answered.** The multivariable review suggests an exercise that $\operatorname{diag}(\mathbf p)-\mathbf p\mathbf p^\top\succeq0$; the matrix calculus section proves it in the text by the variance argument.
- **Two splits at once.** The multivariable and integral reviews each propose a split without seeing the other. Together they change the chapter from five sections to seven, and the route paragraph (IDX-1) must be rewritten after that decision, not before.

## Could not check

Rendered pages; the TensorFlow, JAX and MXNet stores; the Baur and Strassen constant; the figures attributed to Bertoin et al.; the problem numbers in the MIT attributions; the proposed code changes (MC-16, AD-8, IC-24), each of which the section review marks as deferred under the author's decision on experiments.
