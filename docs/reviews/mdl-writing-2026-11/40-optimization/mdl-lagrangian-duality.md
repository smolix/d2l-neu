# Duality

Source: `chapter_mdl-optimization/mdl-lagrangian-duality.md`. About 7,500 words of prose and mathematics to the summary (9,100 in the file with code), plus 2,300 words of exercises. Five `##` parts: the dual problem, the convex conjugate, strong duality, multipliers as prices, worked examples.

## Verdict

The better of the two sections, and in its first part at the level of the model texts. It opens on a question a practitioner has: a solver stops at $(0.55, 0.835)$ with value $8.01$; "Is that good enough, or is there a feasible point with value $7$?" It answers with a table of numbers ($q(3) = 7.875$, $q(4) = 8$) before any definition, and only then generalizes. The proof of strong duality through the set $\mathcal{A}$ of attainable (constraint, objective) pairs is the right short proof and says afterwards where each hypothesis was used. The SVM example ends on something most textbooks omit: the dual solution is a segment, and "margin $1$ does not imply $\alpha_i > 0$". I recomputed every worked number, the stored outputs, and the numbers quoted in exercises 3, 4(c), 7 and 10; all are right, and I found no false mathematical statement. What keeps it from a 5 overall is order and length. The conjugate (2,000 words) is placed between weak and strong duality, so the reader who has just been told that a gap can exist waits 240 lines to learn when it does not. The section re-derives things from its neighbour (the price computation, Slater's condition) and from itself (the lower edge of $\mathcal{A}$ is computed three times in three notations without the reader being told it is one curve). **The most important fix:** move "The Convex Conjugate" after "Multipliers as Prices", and state once that the shadow-price inequality is the supporting line of the figure.

On the question put to this review: this is a separate section from Constrained Optimization, and the cut between them is in the right place. See "For the chapter review" for the leaks.

## Scores

| Clarity of explanation | Clarity of thought | Pedagogy | Language | Ease and enjoyment |
|:-:|:-:|:-:|:-:|:-:|
| 5 | 4 | 4 | 4 | 4 |

## What works

- **The opening problem.** A stopping point with numbers, a reason the old test fails ("The gradient at the true optimum is $(-2.4, -3.2)$, not zero"), and a reason brute force fails ("the disk has infinitely many points"). The section then keeps its promise twice: "The solver's $8.01$ is therefore within $0.01$ of optimal. Even $\lambda = 3$, picked without solving anything, proves that it is within $0.135$." That last sentence is the one a reader quotes a week later.
- **"Why is $\lambda = 4$ the right value?"** The paragraph that reads too small a multiplier as an infeasible minimizer ($\mathbf{x}(1)$ has norm $2.5$) and too large a one as a minimizer "pushed deep inside" is the intuition for complementary slackness, given before the term is needed.
- **"A value $-\infty$ is a true but useless bound."** The author judging, in one short sentence.
- **The conjugate as a supporting line**, started in one dimension with slope $2$ and the parabola, with $|x|$ as the case where the supremum is infinite, and the double well as the case where Fenchel-Young equality fails. Definition, example, boundary, in that order.
- **"Why exponentials?"** The log-sum-exp and entropy subsection answers a question about softmax that the main text leaves open, and the sentence "Without the entropy term the maximizer of $\mathbf{p}^\top\mathbf{x}$ is a one-hot vector" makes the role of the entropy concrete.
- **The strong duality proof** and the paragraph after it: "The proof used convexity once, to make $\mathcal{A}$ convex, and the strictly feasible point once, to rule out a vertical line, which no finite multiplier can represent." The proof header is honest about its scope ("one inequality constraint ..., $p^\star$ finite").
- **KKT necessity closes the loop** with the previous section in six lines ("The proposition also proves the necessity of the KKT conditions").
- **The SVM non-uniqueness passage** ("The printout also shows a limit of reading the dual"), with the rank argument and exercise 4(c) to compute the two ends of the segment.
- **The gap example** is the smallest possible, computed by hand, and the text checks that Slater's point exists so that the reader attributes the gap to the right cause.
- Figures (rendered and checked) carry the argument: the supporting-line pair for the conjugate, the two sets $\mathcal{A}$, the basins.

## Structural findings

1. **Order: the conjugate interrupts the main line.** "The Dual Problem" ends: "The difference $p^\star - d^\star \ge 0$ is the **duality gap**. It was zero for the clipping problem, and the last worked example of this section shows a problem where it is not." The reader's question is now "when is it zero?". The next 240 lines are the conjugate, Fenchel-Young, biconjugates, a list of pairs, and log-sum-exp. The conjugate is a tool for *computing* duals; nothing in "Strong Duality" or "Multipliers as Prices" uses it. Move it to follow "Multipliers as Prices", directly before "Worked Examples", where its opening sentence ("Computing $q$ needs an unconstrained minimization for every value of the multipliers") is the natural lead-in to computing three duals. The only repair needed is the first sentence of "Strong Duality", which cites the half-plane (DUA-5). The line of argument then reads: bound, when the bound is tight, what the tight multipliers mean, how to compute duals, three duals.
2. **One curve, three derivations, no identification.** The lower boundary of the clipping problem's set $\mathcal{A}$ is derived after the Slater proof as $t = \tfrac12(5 - \sqrt{1 + 2u})^2$ with $\rho$ for the radius; the previous section derives the same thing as $p^\star(s) = \tfrac12(5 - \sqrt{2s})^2$; "Shadow Prices" derives it a third time with $s = \tfrac12 + u$ and $r$. The text never says that the lower edge of $\mathcal{A}$ *is* the function $p^\star(u)$ of the shadow-price proposition, and so misses the picture that makes that proposition obvious: the supporting line $t + \lambda^\star u \ge p^\star$ of :numref:`fig_mdl-opt-primal-dual-gap` is the inequality :eqref:`eq_mdl-opt-shadow-price` (DUA-9).
3. **Repetition across the boundary** with Constrained Optimization: the price computation and the scaling remark (DUA-10), Slater's condition (DUA-7). And one missing sentence: the KKT sufficiency proof of the previous section was already weak duality (DUA-4).
4. **Repetition across the part.** "Among all distributions with the same expected score $\mathbf{p}^\top\mathbf{x}$, it has the largest entropy" and all of exercise 2 are the maximum-entropy proposition of `subsec_mdl-max-entropy-derivation` (chapter 41), which is not cited (DUA-6).
5. **Lookup use.** The section is reached from the main text at `subsec_mdl-weight-decay-duality` and from chapter 42 at `subsec_mdl-convex-conjugate`. Both subsections stand up reasonably well, but the section says "the previous section" eight times without a label (DUA-2), which a reader arriving by reference cannot follow.
6. **Length.** 7,500 words is long but most of it is used. What can go without loss: the first code cell's second half and the sentence it repeats (DUA-3), the third derivation of the clipping curve, the recap paragraph (DUA-16), about 500 words in all. The offset-folding passage of the SVM (DUA-12) is the one stretch that drags.

## Passage-level findings

**DUA-1** (moderate, explanation). Opening, second paragraph.

> minimized the squared distance $\tfrac12\|\mathbf{x} - \mathbf{x}_0\|^2$ to

In the previous section the point being clipped is $\mathbf{y}$ throughout ("Clipping the gradient $\mathbf{y} = (3, 4)$", $\mathbf{x} = \mathbf{y}/(1 + \lambda)$). Here the same point is $\mathbf{x}_0$, in the opening, the code, and exercise 10, and the formula $\mathbf{x}(\lambda) = \mathbf{x}_0/(1 + \lambda)$ is derived again. The example is the bridge between the two sections; it should not change its letters at the bridge. Use $\mathbf{y}$ here (the conjugate's slope variable $\mathbf{y}$ is a clash either way; if that decides it, change the previous section to $\mathbf{x}_0$, but pick one).

**DUA-2** (moderate, explanation; pattern, 8 instances). Throughout.

> function (:numref:`sec_mdl-convexity`). The previous section showed that a

> previous section proved that the KKT conditions are sufficient for convex

> qualification announced in the previous section: one point strictly inside

"The previous section" appears eight times, mostly without a label. The appendix is read by lookup, and `subsec_mdl-weight-decay-duality` is the target of a reference from `chapter_linear-regression/weight-decay.md`: a reader who lands there is told that "the previous section proved" something. Replace each by the label of the subsection meant (:numref:`subsec_mdl-kkt-conditions`, :numref:`subsec_mdl-closest-point`), once per subsection.

**DUA-3** (minor, pedagogy and language). "A Lower Bound from One Multiplier", around the first cell.

> Do these values, and the best bound of $8$, follow from the definition

> touching it once. Every value is a bound, the best bound is the optimum, and

The question is not a real one: the cell minimizes the Lagrangian through the closed-form minimizer $\mathbf{x}_0/(1 + \lambda)$, so it evaluates the same algebra the reader has just done, and its printed values are the ones already listed in the prose. The plot is worth having (a concave curve under the line $p^\star = 8$, touching once). The sentence after it then repeats, word for word, the sentence forty lines earlier ("The best bound equals the optimum, and the $\lambda$ that attains it is the KKT multiplier"). Direction: introduce the cell as "The cell plots $q$" and end on what the plot adds, concavity, which the next subsection proves. Delete the repeated sentence.

**DUA-4** (moderate, pedagogy). "The Lagrange Dual Function", after the weak duality proof.

> This is the inequality that made every value $q(\lambda)$ of the clipping

Add the link to the previous section here: its proof of KKT sufficiency was this inequality at $\boldsymbol{\lambda}^\star$ (the function $\varphi$ there is $\mathcal{L}(\cdot, \boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)$, its minimum is $q(\boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star)$, and complementary slackness made the bound tight). One sentence turns two sections with a shared example into two sections with a shared idea.

**DUA-5** (minor, pedagogy). "Duals of Linearly Constrained Problems".

> with $\mathbf{a} = (1, 2)$ and $b = -1$. The origin violates the constraint

The previous section's closest-point example used $\mathbf{a} = (3, 4)$, $b = 5$ and landed on $(0.6, 0.8)$ "once more"; this paragraph says it is "the previous section's closest-point example" and uses different numbers, and exercise 6(b) cites the same subsection with a third set ($\mathbf{a} = (1, 2)$, $b = 1$). With $\mathbf{a}^\top\mathbf{x} \ge 5$ and $\mathbf{a} = (3, 4)$ the dual is $q(\lambda) = 5\lambda - \tfrac{25}{2}\lambda^2$, $\lambda^\star = 0.2$, $d^\star = \tfrac12$, and the reader recognizes both the point and the multiplier $0.2$ from the earlier section. Use the same numbers in all three places.

**DUA-6** (moderate, thought). "Log-Sum-Exp and Entropy".

> over the simplex. Equivalently, among all distributions with the same

This is the maximum-entropy characterization that `chapter_mdl-probability-statistics/mdl-exponential-family.md` states and proves as a proposition (label `subsec_mdl-max-entropy-derivation`), including the softmax case ("Among all distributions on the classes with a given average logit ... the one of largest entropy is then $p_k \propto e^{\eta z_k}$"). Neither section cites the other, and exercise 2 here asks the reader to derive it once more. Cite the label in the text. Keep exercise 2 only for what is new relative to chapter 41: part (c), the multiplier as the price of the score constraint and the temperature limit; reduce (a) and (b) to a reference.

**DUA-7** (moderate, correctness; sure). "Slater's Condition".

> interior of $\mathcal{D}$: $g_i(\bar{\mathbf{x}}) < 0$ for every $g_i$ that

Three points. (i) This is the second definition of Slater's condition in the chapter and it differs from the first (the previous section requires strict inequality for every $g_i$); see CON-6 in that review. Define once, refer here. (ii) The text then calls it "the constraint qualification announced in the previous section", but that section did more than announce it: it defined it and stated its consequence. (iii) The SVM paragraph argues Slater by scaling a separating hyperplane "until all margins exceed $1$", which is the strict form; under the refined form just stated, the constraints are affine and feasibility is enough, as the water-filling paragraph correctly says. Make the SVM sentence match: "the constraints are affine, so separability is all Slater's condition needs". Also "interior of $\mathcal{D}$" should be "relative interior" to be exact, or the proposition should assume $\mathcal{D}$ open; minor.

**DUA-8** (minor, explanation). "Supporting Lines".

> Two conventions hold from here on. In this section a star $\star$ marks an

The convention is announced fifteen lines after $f^*$ first appears and 300 lines after $p^\star$ and $\mathbf{x}^\star$, and "from here on" is not true of the star. It is also in the chapter's notation table. Move the sentence about $\star$ and $*$ to the line that defines $f^*$ (one clause), and keep the paragraph for the one convention that is new, extended-valued $f$.

**DUA-9** (moderate, pedagogy). After the Slater proof, and "Shadow Prices".

> $\|\mathbf{x}\| = \rho$ the smallest objective is $\tfrac12(5 - \rho)^2$ and the

> For the clipping problem, relaxing $g(\mathbf{x}) \le 0$ to

The first passage computes the lower edge of $\mathcal{A}$ and its slope $-4$ at $u = 0$. The second, 200 lines later, computes $p^\star$ as a function of the relaxation and its derivative $-4$. They are the same curve: the lowest $t$ that $\mathcal{A}$ reaches at constraint level $u$ is $p^\star(u)$. Say so, and the shadow-price proposition gets its picture for free: the best supporting line lies below $\mathcal{A}$, so $p^\star(u) \ge p^\star(0) - \lambda^\star u$ for every $u$, with slope $-\lambda^\star$ where the edge is smooth. Suggested sentence before the proposition: "In :numref:`fig_mdl-opt-primal-dual-gap` the lower edge of $\mathcal{A}$ is the graph of $p^\star(u)$, and the supporting line of slope $-\lambda^\star$ lies below it. That is the proposition." Then the clipping paragraph can cite the edge computed earlier instead of recomputing it. Also rename $\rho$ to $r$: the previous section uses $\rho$ for the contraction factor and this section uses $r$ for the radius everywhere else.

**DUA-10** (moderate, thought). "Shadow Prices".

> The multiplier depends on how the constraint is written. Here $\lambda^\star$

This paragraph, the one before it, and the one after it (the closest-point example as the equality case) are the home of material the previous section also teaches in two paragraphs with the same numbers ($\partial p^\star/\partial b = 0.2$; $p^\star(s)$ with derivative $-4$; multiplier $-0.2$ against $-1$). This is the right home, since here the claim is proved. The fix is in the other section (CON-3, CON-4); here, after that fix, the closest-point paragraph should carry the numbers ($b$ from $5$ to $5.1$, value up by $0.02$) that are removed there.

**DUA-11** (moderate, explanation). "Weight Decay and Norm Constraints".

> For a *convex* loss the correspondence holds in both directions. From

The main text (`weight-decay.md`) sends the reader here for the correspondence "($\|\mathbf{w}\| \le s$ for ridge, $\|\mathbf{w}\|_1 \le s$ for lasso)", and the previous section says the lasso weight "is written $\lambda$ because it plays the role of the multiplier of a constraint $\|\mathbf{w}\|_1 \le c$". This subsection proves the $\ell_2$ case only and does not mention that the argument is general. It is: both directions use only that the penalty is convex, that the KKT conditions are sufficient, and that Slater's condition holds for a positive budget (with subgradients in place of the gradient of the penalty). Add two sentences saying so for a convex penalty $\Omega$ and naming the $\ell_1$ case, so that both inbound references are honoured.

**DUA-12** (minor, pedagogy). "The Support Vector Machine Dual".

> This variant also penalizes the offset, so its boundary can differ slightly

The subsection derives the standard dual with its equality constraint (35 lines), then changes the problem to remove that constraint (20 lines), then solves the changed problem. The equality-constrained dual is never computed, and the reader is told that the problem solved "can differ slightly" from the one derived. The effective-domain lesson is worth its place (it was promised: "in the SVM dual below, this restriction turns into an equality constraint"). The folding is there only so that projected ascent on the orthant applies. Shorten it to three sentences, and say plainly which problem the cell solves and why: "To keep the dual feasible set an orthant, the cell solves the variant with the offset folded into the weights."

**DUA-13** (minor, pedagogy). "Water-Filling".

> The second example has a closed-form solution with a geometric picture. A

A fine example and the October review asked to keep it; the cell that checks $\nu^\star = 0.697674$ by re-solving at $P \pm 10^{-4}$ is the best experiment in the section, because the number checked was predicted by a proposition. But it is the third threshold-by-KKT solution the reader meets (simplex projection, soft-thresholding, now this), the text notes the kinship in two places ("turned upside down", "the same threshold structure upside down"), and no deep learning use is named. One of the two kinship remarks can go, and one clause should say why it is here: it is the standard example in which a multiplier has units an engineer cares about.

**DUA-14** (minor, language; pattern, 5 instances). Before each cell.

> duality gap of $\tfrac14$. Does a grid computation over the domain confirm

Every experiment is introduced by a question whose answer is yes: "Do these values ... follow from the definition", "Is $\mathrm{lse}(\mathbf{x})$ the maximum ...", "Does the constrained problem ... return the penalized solution", "Does the dual solution give a separating hyperplane", "Does a grid computation ... confirm the gap?". Once is a device, five times is a template, and a question the reader cannot get wrong is not a question. Keep it where something could go either way (the weight-decay cell, where the multiplier is recovered from a different algorithm), and elsewhere say what the cell computes.

**DUA-15** (minor, language; pattern, 4 instances). After displayed definitions.

> Thus $q$ asks how low $f$ can go once the constraints are replaced by

"Thus ..." opens the reading sentence after four definitions (also "Thus $f^*(\mathbf{y})$ is minus the intercept", "Thus log-sum-exp is the largest value", "Thus $p^\star(\mathbf{u}, \mathbf{v})$ is the optimal value"). The sentences are good; nothing has been deduced, so "Thus" is wrong in each, and the repetition shows the equation protocol being applied. Drop the word.

**DUA-16** (moderate, language). Last paragraph of the body.

> We can now bound a constrained optimum from below with any nonnegative

The first sentence is a recap in the chapter's fixed closing formula, and the Summary follows directly. The rest of the paragraph is a real closing (what remains open: nonconvex gaps, non-convergent descent-ascent; where convex subproblems occur in deep learning) and should stay. Delete the first sentence and open with "Two questions remain open."

**DUA-17** (minor, explanation). Exercise 6(b).

> $\mathbf{a} = (1, 2)$, $b = 1$, and the start $\mathbf{x} = \mathbf{0}$,

Cites "the closest-point problem of :numref:`subsec_mdl-closest-point`", which has $\mathbf{a} = (3, 4)$ and $b = 5$. See DUA-5.

**DUA-18** (minor, pedagogy). Imports cell.

> from torch.nn import functional as F

As in the previous section, the code is NumPy and `d2l.plot` only; `torch`, `nn`, `F`, `math` and the other tabs' framework imports are unused.

## Notation

Not a defect by itself, but a reviser should know the load. Within this one section: $\lambda$ is a multiplier, the weight-decay coefficient, and (as $\lambda_{\max}$) an eigenvalue; $p$ is the number of equality constraints, $\mathbf{p}$ a probability vector and a marginal, $p^\star$ the primal value, $P$ a power; $\theta$ is a convex weight in the Slater proof and the water level; $\nu$ is an equality multiplier whose reciprocal is $\theta$; $\rho$ is a radius here and a contraction factor one section earlier; $q$ is the dual function (and a difference quotient in Convexity). The chapter table covers $\lambda$. The two cheap fixes are $\rho \to r$ (DUA-9) and $\mathbf{x}_0 \to \mathbf{y}$ (DUA-1).

## Exercises

Eleven problems, in band, graded, and the strongest exercise set I have seen in the book: each one is a small piece of the subject (a dual certificate for basis pursuit, the leave-one-out bound from support vectors, the gap as a property of the formulation, descent-ascent on a bilinear game and on a Lagrangian, Sinkhorn as block-coordinate dual ascent, the ridge dual, $q' = g(\mathbf{x}(\lambda))$). Stated answers that I checked are correct (basis pursuit $p^\star = d^\star = 1$; the two segment endpoints in 4(c); dual ascent $5.07$, $4.02$, $4.0001$; local minima $1.16$ and $3.11$ in 7).

- **Exercise 2**: overlaps chapter 41's maximum-entropy proposition (DUA-6). Keep part (c).
- **Exercise 6(b)**: numbers disagree with the subsection it cites (DUA-17).
- **Exercise 11** re-solves by the dual the half-plane problem that the previous section solved by KKT and by PGD. Good as a bridge; it should cite that passage, and part (c)'s question ("Does the ill-conditioning make this constrained problem hard to solve through its dual?") is the best line in it.
- **To receive**: the Moreau decomposition exercise from the previous section, which needs the conjugate.
- Exercise labels are a mix of `ex_mdl-convexity-`, `ex_mdl-constrained-optimization-duality-` and `ex_mdl-lagrangian-duality-`; invisible to the reader.

## Still open from October

The October review asked to cut "most conjugates" and the problem-class table. The table is gone and the list of conjugate pairs is down to four, two of which are used only in exercises; that is acceptable. It asked to keep water-filling; it is kept. Its general remark on this chapter, "Every opening lists questions instead of posing a problem", no longer applies here: this opening poses a problem.

## For the chapter review

- **Fit.** Sixth of seven sections. It needs the KKT conditions and the clipping example from Constrained Optimization, the supporting-hyperplane fact and the softmax Hessian from Convexity, and hands to Numerical Stability with one sentence. It is the most self-contained theory section of the chapter and the one with the clearest single question.
- **Two sections or one told twice?** Two. Constrained Optimization is primal (recognize an optimum, compute it by projection); this one is the lower bound, when it is tight, and what the multipliers mean. No result is proved in both. What is told twice is small and fixable: the multiplier as a price with the same worked derivative (home: here), the scaling of multipliers (home: here), the definition of Slater's condition (home: there, proof here), and the Lagrangian's definition (needed here for the domain $\mathcal{D}$; shorten). What is missing at the boundary is the sentence that the KKT sufficiency proof was already weak duality. The shared example changes its symbol from $\mathbf{y}$ to $\mathbf{x}_0$ at the boundary. An alternative division (multipliers, KKT and duality in one section; projection and proximal algorithms in the other) would be tidier on paper but would separate the clipping example from the projection that produces it; I do not recommend it.
- **Titles and labels.** The titles ("Constrained Optimization", "Duality") do not overlap; the file names and labels do (`mdl-constrained-optimization-duality` holds no duality, and `subsec_mdl-lagrangian-duality` is a subsection of `sec_mdl-lagrangian-duality`). Chapter 42 cites `subsec_mdl-lagrangian-duality` twice, so any rename needs a grep.
- **Order within the section.** Move the conjugate after the price material (Structural 1). This is the one reordering in my half of the chapter that I would press for.
- **Patterns for the chapter and part.** (a) The mathematics is careful: across both sections I found no false statement after recomputing every number, only an inconsistent definition and an existence argument with a gap. (b) The prose is clean and mostly even; the sentences with voice are few and are quoted above so they can be protected. (c) Templates: the closing "We can now ...", the question before each cell, "Thus" after each definition. (d) Cross-part duplication is by topic, not by sentence: Lagrange's condition (chapters 39 and 40), maximum entropy and softmax (chapters 40 and 41). A pass that greps each proposition title across the six chapters would find the rest.
- **Could not check.** The TensorFlow, JAX and MXNet outputs (the code cells are untabbed NumPy, so they should not differ); the citations' theorem and section numbers, except that the quoted bounds are the standard ones.
