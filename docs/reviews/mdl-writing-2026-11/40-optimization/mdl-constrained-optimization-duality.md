# Constrained Optimization

Source: `chapter_mdl-optimization/mdl-constrained-optimization-duality.md`. About 6,500 words of prose and mathematics to the summary (8,000 in the file with code), plus 2,100 words of exercises. Four `##` parts: Lagrange multipliers (1,100 words), KKT (1,500), projections and projected gradient descent (2,600), proximal operators and coordinate descent (2,400).

## Verdict

A good section, close to the bar in its first three parts. One worked instance, clipping $(3, 4)$ to the unit disk, is solved in the opening and then carried through everything: it is the tangency picture, the active case of the KKT split, the residual $(2.4, 3.2)$ of the projection inequality, and the supporting line $3x_1 + 4x_2 = 5$ of the hyperplane example. I recomputed every hand number in the section and every claim about a stored output; all are right. The proofs are short and each is the short one (KKT sufficiency in four lines, nonexpansiveness in five, the PGD contraction in three steps with the one non-routine step named). The faults are of structure, not of sentences. The fourth part (lasso, proximal gradient, coordinate descent) is a second topic: it is about nonsmooth penalties, not about constraints, it runs a third of the section, and its last subsection ends by saying that the method "is not used for dense end-to-end network training". And the section teaches three things that the next section teaches again (the multiplier as a price, with the same derivative of the same function; that a multiplier depends on how the constraint is written; Slater's condition). **The most important fix:** cut coordinate descent to a paragraph and move the price material to the Duality section, so that this section is "recognize a constrained optimum, then compute it by projecting", in about 5,000 words.

On the question put to this review: this section and Duality are two sections, not one told twice. The division is sound (this one finds and recognizes the primal optimum; the next one bounds it from below and prices the constraints). The boundary leaks in four places, listed under Structural findings and again in "For the chapter review".

## Scores

| Clarity of explanation | Clarity of thought | Pedagogy | Language | Ease and enjoyment |
|:-:|:-:|:-:|:-:|:-:|
| 4 | 4 | 4 | 4 | 3 |

## What works

- **The opening.** "Clipping the gradient $\mathbf{y} = (3, 4)$ to norm at most $1$ returns $(0.6, 0.8)$" and then "$f$ has gradient $(-2.4, -3.2)$ there": the reader sees with numbers that the zero-gradient test has failed before any machinery appears, and the third paragraph cancels the gradient with "$4$ times the outward normal". This is the smallest case solved completely. Protect it.
- **The running example is reused, not restated.** The multiplier $4$ returns as the ratio of two vector lengths, as the residual of the projection ("the multiplier times the constraint normal once more"), and the far point with $\nu = -6$ explains in advance why inequality multipliers must be nonnegative ("the multiplier $-6$ would give the far point $(-0.6, -0.8)$ of the circle, which is not a minimum over the disk").
- **Necessary versus sufficient is taught by counterexample each time**: the maximum on the circle, $-x^2$ on $[-1, 1]$ with a KKT point that "is the *maximum*", the constraint $x_1^2 + x_2^2 = 0$ with no multiplier, the two-point set $\{-1, 1\}$ where projection is discontinuous.
- **"it touches $\mathbf{x}^\star$ but does no work"** for an active constraint with zero multiplier. A sentence with voice that is literally true.
- **The simplex projection** is the best subsection. The hand example ($\tau_1 = -0.1$, $\tau_2 = 0.2$, $\tau_3 \approx 0.067$ fails) turns into the algorithm, the two facts that justify the algorithm are proved in four lines, and the code cell makes a point that is easy to miss: stationarity, complementary slackness and dual feasibility hold for a wrong $\tau$ as well, "so these residuals cannot confirm the solution", and the variational inequality at the $n$ vertices is what certifies it.
- **The subgradient experiment** asks a real question ("Does it find the five zeros?") whose answer is no, and the explanation (a fixed step uses one element of $[-\lambda, \lambda]$ and none balances the smooth part) leads to the prox. This is the predict, run, be surprised pattern the Basics review asked for.
- **The remark that gradient clipping is not PGD** on any parameter constraint, with an exercise that makes the reader find the fixed points of both, prevents a confusion the opening example invites.
- All figures carry weight and match the text (rendered and checked): tangency with the maximum marked, the active-set pair, the obtuse-angle and nonexpansive pair, the threshold bars against softmax.

## Structural findings

1. **The fourth part is a different section.** "Proximal Operators and Coordinate Descent" opens "A hard wall is not the only way to restrict weights" and from there on nothing is constrained. Soft-thresholding, the prox and ISTA earn their place: the prox is the projection generalized, `chapter_linear-regression/weight-decay.md` points here for it, and exact zeros from $\ell_1$ is something a deep learning reader looks for. "Coordinate Descent" (750 words, one cell, one plot) does not: it introduces coordinate-wise smoothness constants, Gauss-Seidel, the glmnet update, and a randomized rate, for a method the text itself says is not used to train networks. The October review listed it under "Cut". Recommended: keep "Soft-Thresholding and the Proximal Operator" and "Proximal Gradient Descent"; reduce coordinate descent to one paragraph (the lasso coordinate step is a soft-threshold; it needs separability, with the $\max(|x|, |y|)$ counterexample) and keep exercise 10 for the rest; retitle the part "Penalties with Kinks: the Proximal Step" or similar.
2. **Boundary with Duality, leak 1: the multiplier as a price is taught here twice and there a third time.** Here: the last paragraph of "The Closest Point on a Hyperplane" and the last paragraph of "Active and Inactive Constraints", which computes $p^\star(s) = \tfrac12(5 - \sqrt{2s})^2$ and its derivative $-4$. Duality, "Shadow Prices", computes the same function and the same derivative again. Keep one sentence here ("the multiplier will turn out to be the rate at which the optimum changes when the constraint is moved, :numref:`subsec_mdl-shadow-prices`"), and move both computations there (CON-3).
3. **Leak 2: "a multiplier depends on how its constraint is written"** is shown here on the hyperplane (multiplier $-0.2$ against $-1$) and again in Duality on the disk (multiplier $4$ against $2$). One home; it belongs with the price reading.
4. **Leak 3: Slater's condition is defined in both sections, with different hypotheses** (CON-6). Here every $g_i$ must be strictly negative; there only the non-affine ones.
5. **Leak 4, a missed connection: the proof of KKT sufficiency is weak duality at $\boldsymbol{\lambda}^\star$.** The chain $f(\mathbf{x}) \ge \varphi(\mathbf{x}) \ge \varphi(\mathbf{x}^\star) = f(\mathbf{x}^\star)$ says that the unconstrained minimum of the Lagrangian is a lower bound on $f$ over the feasible set, which is the whole idea of the next section. Neither section says so. One sentence after the proof would make the next section's opening feel inevitable (CON-7).
6. **The Lagrange condition is proved in the calculus chapter and retold here.** `subsec_mdl-lagrange` states the same proposition with the same hypotheses and a full proof, with the opposite sign and the symbol $\mu$ (which in this chapter is the strong-convexity constant). This section restates the proposition, retells the proof in a paragraph, and then spends a paragraph reconciling the two conventions (CON-2).
7. **Deep learning uses are named for the opening and the simplex, thin elsewhere.** Projected gradient descent has one famous use in deep learning, adversarial examples under an $\ell_\infty$ budget, where the projection is a coordinatewise clip. It is not mentioned here (the Duality section cites the same paper for min-max). One clause in the "PGD is practical whenever" paragraph would do.
8. **Closing and summary say the same thing** (CON-18).

## Passage-level findings

**CON-1** (minor, language). Opening, first paragraph.

> a hard limit by a cost and leads to a close relative

"A close relative" of what? The sentence interrupts a paragraph about constraints to announce the lasso and does not say what the relative is. Either drop it (the third paragraph introduces the proximal step properly: "When a penalty with a kink, such as the absolute value, replaces the hard wall") or say it: "A lasso penalty charges for the absolute value of every weight instead of forbidding large ones; its optimum is found by the same kind of step."

**CON-2** (moderate, thought). "Tangency and the Lagrange Condition".

> The full argument is in :numref:`subsec_mdl-lagrange`, which writes the

By this point the section has stated the proposition and given the proof in seven lines ("The proof makes the tangency argument precise ..."), so the reader has the argument twice in the part and then a paragraph about why the two statements differ in sign ($\nu = -\mu$). Per guide 7.7 the fact has one home. Direction: keep the tangency paragraph and figure (they are better here than in the calculus chapter, because the level sets cross and the maximum is marked), state the proposition, and replace the proof paragraph and the reconciliation paragraph by "This is the Lagrange condition of :numref:`subsec_mdl-lagrange`, proved there; with $h = g - c$ and $\nu = -\mu$ it is the same statement." Better still, change the calculus chapter to the plus-sign form so that no reconciliation is needed (for the chapter 39 reviser; the October review flagged the opposite signs).

**CON-3** (moderate, thought). "The Closest Point on a Hyperplane" and "Active and Inactive Constraints".

> a multiplier is called a *shadow price*.

> The multiplier is again a price, now in the units in which $g$ is written,

Two paragraphs teach the sensitivity reading by example, twenty and eighty lines before the KKT conditions are stated, and the Duality section then proves it as a proposition and recomputes $p^\star(s) = \tfrac12(5 - \sqrt{2s})^2$ with derivative $-4$ ("Shadow Prices", "For the clipping problem, relaxing"). A front-to-back reader meets the same derivative of the same function twice in two sections. Move the second paragraph to Duality entirely. Reduce the first to one sentence with a forward reference, keeping the number: "Raising $b$ from $5$ to $5.1$ raises the optimal value by about $0.02 = -\nu^\star \cdot 0.1$; :numref:`subsec_mdl-shadow-prices` shows that this is what a multiplier always measures."

**CON-4** (minor, thought). "The Closest Point on a Hyperplane".

> multiplier depends on how its constraint is written

True and worth saying once. It is said again in Duality ("The multiplier depends on how the constraint is written. Here $\lambda^\star$"). Keep the one that sits next to the price reading, since the scaling only matters once the multiplier has units.

**CON-5** (minor, explanation). "The Karush--Kuhn--Tucker Conditions".

> The first-order optimality conditions are the four

The four conditions are presented as *the* optimality conditions; that they hold at a minimum only under a qualification, and suffice only under convexity, arrives sixty lines later. A reader who lands here by lookup takes away "minimum if and only if KKT". Add one clause at the statement: "At a minimum where a constraint qualification holds (next subsection), there are multipliers satisfying the four ...".

**CON-6** (moderate, correctness; sure). "Constraint Qualifications and Sufficiency".

> convex and every $h_j$ is affine. It satisfies **Slater's condition**

Here Slater's condition requires $g_i(\bar{\mathbf{x}}) < 0$ for all $i$. The Duality section states it as a proposition with a weaker hypothesis ("$g_i(\bar{\mathbf{x}}) < 0$ for every $g_i$ that is not affine, $g_i(\bar{\mathbf{x}}) \le 0$ for the affine ones") and calls it "the constraint qualification announced in the previous section". Both statements are true, but they are two definitions of one named condition, and exercise 3 of Duality then appeals to "Slater's condition in its refined form" as if the reader had met two. Define it once. Suggested: define the refined form here in one sentence, state the consequence by citation as now, and let Duality refer back for the definition and add the proof.

**CON-7** (moderate, pedagogy). After the proof of KKT sufficiency.

> The function $\varphi$ is $f$ plus a penalty that is nonpositive on the

This paragraph says the proof again in words (five sentences for a four-line proof) and then restates the conclusion. What the reader needs at this point is the idea the proof contains: the unconstrained minimum of the Lagrangian, for fixed nonnegative multipliers, is a lower bound on $f$ over the feasible set, and at the KKT multipliers the bound is attained. Replace the paragraph by two sentences that say this and point to :numref:`sec_mdl-lagrangian-duality`, which starts from that bound. Keep the final sentence ("For convex problems the KKT conditions are thus sufficient, and under Slater's condition also necessary").

**CON-8** (moderate, thought; verified numerically). "Projected Gradient Descent", after the PGD cell.

> is a worst case per step, and a few steps contract far more. The run reaches

The prose reads the stored output correctly (ratio $0.998002$, 974 steps, final gradient $(0.999, 0.999)$) but does not explain it, and the explanation is not what a reader would guess. I traced the run. The iterates alternate: a step from the line $x_1 + x_2 = 1$ flips the sign of $x_2$ and lands inside the half-space, the next step flips it back and is projected onto the line. For almost the whole run the distance shrinks by $0.998$ or $0.9973$ per step; ten steps contract by more than 10 percent; and at step 974 the ratio is $6 \times 10^{-14}$: the run lands on $\mathbf{x}^\star$ in one step. The reason is a coincidence of the chosen numbers. On the line, the quadratic has curvature $(1 + 1000)/2 = 500.5$ along $(1, -1)/\sqrt2$, and $\eta = 2/1001$ is exactly its inverse, so the first time two consecutive steps are both projected the method takes an exact Newton step along the constraint. So "974 steps" is a finite-termination count, not a rate, and the comparison with "$7{,}660$ steps" compares a guarantee with an accident. Direction (prose only, the experiment can stay): say what the run does. "The iterates bounce: every other step leaves the boundary. Once a step stays on the line, the run finishes at once, because on the line the quadratic has curvature $500.5$ and $\eta = 2/1001$ is its inverse. The proposition bounds each step; it does not predict this." If a later pass may change the cell, a start on the boundary or $\eta = 1/1000$ gives a run whose rate can be read against $\rho$.

**CON-9** (minor, pedagogy). "Projected Gradient Descent", last paragraph.

> PGD is practical whenever $\Pi_C$ is cheap. Nonnegativity constraints project

The two examples (orthant, max-norm rows) are fair but dated. Add the use a deep learning reader has met: projected gradient ascent on an input perturbation with $\|\boldsymbol{\delta}\|_\infty \le \epsilon$, where the projection is a coordinatewise clip to $[-\epsilon, \epsilon]$ (:cite:`Madry.Makelov.Schmidt.ea.2018`, already in the bibliography and cited in the next section).

**CON-10** (minor, correctness; sure). "Projection onto the Simplex".

> positions. Softmax is the same projection with an entropy term in place of the

Softmax is not a projection of $\mathbf{y}$ onto $\Delta$ in any distance the text has defined, and "in place of the squared distance" does not parse into a formula. The precise statement is one line and is proved in the next section: sparsemax maximizes $\mathbf{y}^\top\mathbf{x} - \tfrac12\|\mathbf{x}\|^2$ over $\Delta$, softmax maximizes $\mathbf{y}^\top\mathbf{x} + H(\mathbf{x})$ (:eqref:`eq_mdl-opt-lse-entropy`). Write that, with the forward reference.

**CON-11** (minor, explanation). "Soft-Thresholding and the Proximal Operator".

> weight is written $\lambda$ because it plays the role of the multiplier of a

A promise that is never kept. The Duality section proves the penalty and constraint correspondence for the squared $\ell_2$ norm only, and `chapter_linear-regression/weight-decay.md` cites that subsection for the lasso as well. Either add a forward reference here and one sentence there (see DUA-13 in the Duality review), or drop the clause.

**CON-12** (minor, explanation). Same subsection.

> A first method to try is subgradient descent. A *subgradient* of $|w|$ is a slope

The subgradient, the subdifferential of $|x|$ at $0$, and the test $0 \in \partial f$ are defined in :numref:`sec_mdl-convexity` ("Subgradients"), with a figure. Here they are defined again without a reference, and "A point is a minimizer of a convex function if and only if $0$ lies in its set of subgradients there" is restated as if new. One clause with the reference is enough, and it helps the lookup reader more than a second informal definition.

**CON-13** (moderate, explanation). Same subsection, after the definition of the prox.

> lower semicontinuous and grows quadratically. It therefore attains its minimum

The level changes for one paragraph: "proper closed convex", "lower semicontinuous", "compact sublevel set", "the Weierstrass theorem holds for lower semicontinuous functions". None of these terms is used before or after in the section (the projection's existence was argued with a closed ball, in plain words), and the argument has a gap: the objective grows quadratically only because a proper closed convex $\psi$ lies above some affine function, which is not said (correctness, minor; sure). Direction: "The prox is well defined whenever $\psi$ is convex, finite somewhere, and has a closed epigraph: the objective is then strongly convex, so it has exactly one minimizer :cite:`Parikh.Boyd.2014`." Two sentences, same content, no detour.

**CON-14** (minor, thought). "Proximal Gradient Descent", after the ISTA cell.

> nonzero coordinates sit within about $0.1$ of the planted values. They cannot

The stored output says $0.040$; over 200 reruns I saw $0.02$ to $0.06$. "Within about $0.1$" is true but two to four times looser than the data, and the next sentence explains a shrinkage of "$\approx 0.03$", so the reader wonders where the rest of the $0.1$ comes from. Write "within about $0.05$".

**CON-15** (moderate, thought). "Coordinate Descent".

> step gains less than a gradient step, whose factor is $1 - \mu/L$ with

The paragraph sets up a comparison and does not finish it. The reader has been told $L_j \approx 100$ and $L \approx 130$ to $190$ twice ("Both numbers return below"), and expects them to predict something. The prediction is one line: $d$ coordinate steps shrink the gap by about $1 - d\mu/\sum_j L_j = 1 - \mu/\bar L$ with $\bar L$ the mean of the $L_j$, against $1 - \mu/L$ for one gradient step of the same cost, so a pass of coordinate descent is worth about $L/\bar L \approx 1.5$ gradient steps. Instead the text retreats to "A fair comparison therefore counts work, memory traffic, and sparsity rather than iterations", and the experiment then shows a factor of two to six, which the quoted rate (for randomized steps of length $1/L_j$, not cyclic exact minimization) does not explain. If the subsection stays, finish the comparison and say that exact minimization and fresh residuals account for the rest. If it is cut to a paragraph (Structural 1), this goes with it.

**CON-16** (minor, thought). "Coordinate Descent", second to last paragraph.

> Coordinate descent remains the method of choice for sparse linear models,

"Embedding tables" are trained by sparse stochastic gradient updates, not by coordinate descent; the claim is at best loose (alternating least squares for matrix factorization may be what is meant, and that is the third item of the list). The sentence that follows concedes that the method is not used for network training. Drop "embedding tables".

**CON-17** (minor, pedagogy). "Coordinate Descent", opening.

> Suppose $f$ has coordinate-wise smoothness constants $L_j$, so that moving along

The subsection opens with the general definition, then a general step, then the quadratic, then Gauss-Seidel, and reaches the lasso (the only case the reader has a stake in) after 35 lines. If the subsection is kept, start from the lasso: fix all coordinates but one, the subproblem is the scalar problem just solved, so the exact step is a soft-threshold. The general $L_j$ and Gauss-Seidel can follow in three sentences or go to exercise 10.

**CON-18** (moderate, language). Last paragraph of the body.

> We can now write the KKT conditions of a convex problem, solve them by a case

A recap paragraph with the fixed formula "We can now" (the same formula closes Convexity, Adaptive Methods and Duality in this chapter), followed forty words later by a Summary that lists the same results. Guide 12.2: the body ends on its last result or open question, and the Summary is the only restatement. Keep only the last three sentences, which are the real closing: "Every multiplier so far, however, was found by solving for $\mathbf{x}$ first. ... The clipping problem, with $\lambda^\star = 4$ and $p^\star = 8$, is its first example."

**CON-19** (minor, pedagogy). Imports cell.

> from torch.nn import functional as F

All code in the section is NumPy plus one `d2l.plot`. The four-tab imports cell loads `torch`, `nn`, `F`, `math` (and `gluon`, `optax`, `flax.nnx`, `tf` in the other tabs), none of which is used. The tabs do not fragment the prose, since the imports cell is the only tabbed cell, but the reader of a section whose code is ten short NumPy cells is shown framework imports for no reason. Import `numpy` and `d2l` only.

## Exercises

Ten problems, in band, graded (four marked with a dot), and well above the level of the Basics exercises: nearly every one extends the text instead of drilling it. Exercise 1 (three ways a qualification fails, ending with a non-differentiable $p^\star$) and exercise 5 (second-order conditions on the sphere, ending with PGD as power iteration) are excellent. I checked the stated numbers in exercises 3(d), 6 and 10(c); they are right.

- **Exercise 8 (Moreau decomposition) belongs in Duality.** It defines the convex conjugate, which the text introduces one section later, and uses two unproved facts about $\partial\psi^*$. Move it after the conjugate is taught and drop the in-exercise definition; the sentence in the text that points to it ("the prox of a norm and the projection onto a related ball add up to the identity") becomes a forward reference.
- **Exercise 10(c)** states the theory and its numbers ($0.947$, $0.897$, $0.973$) and asks the reader to "run both ... and measure". The reader verifies a printed answer. Either ask for the prediction (give the Jacobi factor formula, ask for the two rates) or cut the part with the coordinate-descent subsection.
- **Exercise 2** carries the stale label `ex_mdl-constrained-optimization-duality-write-all-four-kkt`; harmless to the reader.
- If coordinate descent is cut to a paragraph, exercise 10 keeps the material for those who want it; say so in the paragraph.

## Still open from October

The October review asked to cut coordinate descent (still present, 750 words) and noted that the Lagrange multiplier has opposite signs in the calculus and optimization chapters (now reconciled by a paragraph, CON-2, but both conventions remain). The LP/QP/SOCP/SDP table it asked to cut is gone, to the Further Reading notebook.

## For the chapter review

- **Fit.** The section follows Adaptive Methods, whose closing hands over with one sentence, and it leans on Convexity for the first-order inequality and (silently) for subgradients, and on Gradient-Based Optimization for the descent lemma and the per-mode factor. It reads on its own up to the proximal part.
- **Boundary with Duality.** Two sections, correctly divided in principle: this one is primal (recognize the optimum, compute it), Duality is the lower bound and its consequences. Four leaks: the price reading and the scaling remark are taught in both (move to Duality); Slater's condition is defined in both with different hypotheses (define once, here); the KKT sufficiency proof is weak duality at $\boldsymbol{\lambda}^\star$ and neither section says so. The file name and label `sec_mdl-constrained-optimization-duality` still carry "duality" from before the split; a brief in `chapter_lm-rl/dpo.md` already cites this label for the penalty and constraint correspondence, which lives in the other section. Renaming is optional and has costs; the mis-citation should be fixed either way.
- **Length and proportion.** About 6,500 words; 5,000 would lose nothing a deep learning reader needs. The cut is coordinate descent, the retold Lagrange proof, the two price paragraphs, and the recap.
- **Patterns to check chapter-wide.** (a) The closing formula "We can now ..." ends at least four sections of this chapter. (b) Unused framework imports in a NumPy-only section. (c) An experiment whose printed numbers are read correctly but whose behaviour is not explained (CON-8); worth checking the other sections' runs on the $\kappa = 1000$ bowl for the same thing. (d) Mathematical care is high: I found no false statement, one inconsistent definition and one gap in an existence argument.
