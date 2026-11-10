# Further Reading (Optimization)

Source: `chapter_mdl-optimization/further-reading.md`. About 1,500 words, no code, no figures, no exercises; 13 topics under four headings. Short review, as assigned: is it a useful and accurate reading guide?

## Verdict

Yes on accuracy, mostly on usefulness. I checked each technical statement against what I know of the cited work (the Chebyshev bound $2\rho^k/(1+\rho^{2k})$, the performance-estimation rate $(1-\mu/L)^2$, the central-path gap $m/s$, the Fenchel dual of $f(\mathbf x)+h(\mathbf A\mathbf x)$, the Frank-Wolfe step and gap, the divergence of simultaneous descent and ascent on $xy$, the growth factor $2^{n-1}$) and found no error. The notebook follows the form of guide 12.3 closely: within length, in chapter order, every topic opens with the gap it fills relative to a named subsection and states one change, every citation carries a clause, and the opening says that nothing depends on it. Its weakness is density. Each topic compresses a field into 80 to 130 words, and several use terms the appendix never defines, so it reads as notes for someone who already knows the literature and less as a guide for the reader of this chapter. The most useful fix is not more text but one sentence per topic saying who should follow the pointer and why (what it explains about training networks), plus a gloss for the half dozen undefined terms.

## Scores

| Dimension | Score | Note |
|---|---|---|
| Clarity of explanation | 3 | Correct and compact; too compressed for a first-time reader in about half the topics. |
| Clarity of thought | 4 | Each topic has one job and is tied to a subsection. |
| Pedagogy | 3 | A good map; little guidance on which road to take. |
| Language | 4 | Plain and exact. A few sentences are clipped past comfort. |
| Ease and enjoyment | 3 | Three topics are a pleasure (edge of stability, interpolation, probabilistic rounding); the rest are entries. |

## What works

- **Every topic is anchored.** "The noise floor of :numref:`subsec_mdl-stochastic-gradients` is proportional to $\eta$. A decaying step removes it but makes late progress slow. Averaging the iterates removes it differently." Gap, known cure, new cure, in three sentences. This is the pattern at its best.
- **"Self-Stabilization at the Edge of Stability"** says exactly what the chapter's quadratic model cannot explain ("but not what brings the sharpness back down") and what one added term changes. It ends on a sentence a reader will remember: "Without the cubic term the oscillation simply grows."
- **"Probabilistic Rounding-Error Analysis"** closes the loop with the main section: the $\sqrt n\,u$ model "fails when the rounding errors are correlated, as when many equal small terms are added to a large running sum and every addition rounds the same way", which is the $10^7\times0.1$ experiment of :numref:`sec_mdl-numerical-stability-conditioning`. Name that experiment explicitly; at present the reader has to notice.
- **"Interpolation"** gives the reason before the result ("The floor exists because the per-example gradients do not vanish at the minimizer").
- The division of labour with the chapter introduction's resource list and with :numref:`sec_optimization-further-reading` is stated in the opening and respected; I found no topic repeated from the introduction's list.

## Structural findings

**S1. No ranking.** Thirteen topics get the same weight and the same register. Guide 10.2 asks the authors to judge. A reader of a deep learning book should be told that edge of stability, interpolation, averaging, the Fisher and Gauss-Newton matrices, and saddle-point methods bear directly on training networks, and that performance estimation, interior-point methods, ADMM, and Frank-Wolfe are classical optimization they will need only for particular problems. A sentence in the opening, or an order within each group that puts the deep learning topics first, would do it.

**S2. Undefined terms.** Used here without a definition or a reference: "a nonlinearity in a feedback loop" and "GMRES" (defined nowhere in the appendix, as far as a search shows); "nuclear-norm ball" (defined only in the linear-algebra further reading and an exercise); "Gram matrix" (linear-algebra chapter), "natural parameter" (:numref:`sec_mdl-exponential-family`), and "regret" (:numref:`sec_mdl-adaptive-stochastic-methods`), which are defined elsewhere but not referenced. "Strong growth condition" is glossed in place, which is the model. Each of the others needs three to six words or a reference.

**S3. Coverage.** The selection is sound. Checked against the chapter: quasi-Newton, variance reduction, the PL condition, and implicit bias are taught in the sections, so their absence here is correct. Topics a reader might still look for and not find: lower bounds for first-order methods on general smooth convex functions (the Chebyshev topic covers quadratics only), sharpness-aware minimization as the practical sequel to the edge of stability, and schedule-free or parameter-free step sizes. None is necessary; the guide says to use this notebook sparingly. If one is added, sharpness-aware minimization has the clearest tie to a subsection.

## Passage-level findings

**FR-1** · minor · language · opening
> "the chapter, topics they had no room for: optimal methods on quadratics and"

The opening is one 100-word paragraph whose first two sentences list the chapter and then the notebook, topic by topic. It is accurate and it is the "list of topics" opening the October review objected to in the appendix. Since the four headings already show the route, cut the list and use the space for the ranking of S1.

**FR-2** · moderate · thought · under "Growth Factors and Iterative Refinement"
> "The mixed-precision refinement of"

The sentence continues with a reference to :numref:`sec_mdl-numerical-stability-conditioning` and "converges only when $\kappa(\mathbf{A})\,u_{32}$ is well below one". That section does not teach iterative refinement in its text. It appears only in its last exercise (marked ●●), where the convergence condition is the thing the student is asked to derive. A reader who did not do the exercise cannot follow this topic, and one who is about to do it has the answer given away. Either introduce the method here in two sentences ("solve in float32, compute the residual in float64, solve for a correction, repeat") and refer to the exercise as the place to try it, or state the condition without attributing it to the section.

**FR-3** · minor · explanation · under "Optimal Polynomials and Conjugate Gradients"
> "In high dimension no method of this kind beats, in the worst case, the"

Two things are compressed here. The reader is not told how the Chebyshev maximum relates to the heavy-ball rate: one clause, "about $2\rho^k$, the same $\rho$ as heavy ball", makes the comparison visible. And "in high dimension" needs its reason (in dimension $d$ any such method finishes in $d$ steps, so the bound binds only for $k<d$), which the next sentence about conjugate gradients half supplies. Also name the quadratic's minimizer or the error $\mathbf e_0$: the first sentence writes $\tfrac12\mathbf x^\top\mathbf A\mathbf x$ and then "the error $p_k(\mathbf A)\mathbf e_0$" with $[\mu,L]$ undefined in this file.

**FR-4** · minor · explanation · under "Rates Certified by Semidefinite Programs"
> "rate, and a dual solution is a proof. For gradient descent with $\eta = 1/L$"

A good sentence that deserves its consequence: the reader should be told that $(1-\mu/L)^2$ against the chapter's $1-\mu/L$ means the hand proof loses a factor of two in the exponent, that is, the exact worst case needs half as many steps as the chapter's bound suggests. Without that the two numbers are just two numbers.

**FR-5** · minor · language · under "The Fisher and the Gauss–Newton Matrix"
> "approximate a curvature matrix. K-FAC's is the Fisher"

"K-FAC's is the Fisher" does not parse on first reading. Write "K-FAC approximates the Fisher matrix". The topic also never says why the distinction matters to a practitioner: the empirical Fisher is what is cheap to compute and what much code uses under the name "Fisher". One clause.

**FR-6** · minor · explanation · under "Interpolation"
> "least squares do, that covariance is zero, and constant-step SGD converges"

"converges linearly, as gradient descent does" holds under the curvature assumptions of the cited result (strong convexity or a PL-type condition), which over-parameterized networks satisfy only locally if at all. Add "on strongly convex or PL objectives (:numref:`sec_mdl-convexity`)". Confidence: high that the qualifier is needed for the statement to be true as written.

**FR-7** · minor · language · under "Operator Splitting and ADMM"
> "and a quadratic penalty $\tfrac{\rho}{2}\|\mathbf{A}\mathbf{x} - \mathbf{z}\|^2$"

$\rho$ is the contraction factor in the first topic of the same file and throughout the chapter, and the penalty weight here. The clash is inherited from the literature; a different letter costs nothing. Likewise $s$ for the barrier parameter is a good choice (it avoids $t$) and could be said to be the "$t$" of Boyd and Vandenberghe for readers who follow the citation.

**FR-8** · minor · thought · under "Saddle-Point Methods"
> "descent and ascent spirals away from the saddle of $xy$ for every step size."

True and vivid ($|1+i\eta|>1$). Give the reader those four symbols; it is the shortest proof in the notebook and shows why the trial step of the extragradient method helps. The reference to GAN training should carry the label of the GAN chapter.

## Exercises

None, correctly.

## Still open from October

Not covered by name in the October review. Its complaint about appendix openings that are lists of topics applies mildly to the first paragraph (FR-1).

## For the chapter review

- The notebook does what guide 12.3 asks and is consistent with the chapter introduction, which says that interior-point and operator-splitting methods are left out and whose resource list points here in its first sentence.
- It depends on the sections only through labels, with one exception: FR-2 relies on an exercise of the preceding section.
- It is the one place in the chapter where the authors could say what matters for deep learning and what is classical background, and it does not (S1). If the other further-reading notebooks of the part are equally even in tone, that is a part-level finding: accurate maps without a recommended route.
- Notation: $\rho$ is used with two meanings here, and the Fisher topic introduces $\mathbf F$, $\mathbf J$, $\mathbf H_{\mathrm{out}}$ in one sentence; the chapter's notation table has none of them, which is fine for a notebook nothing depends on.
