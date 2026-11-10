# Chapter 40, Optimization: writing review

Sources: the nine section reviews under `40-optimization/`, read in full. The chapter itself is `chapter_mdl-optimization/` (introduction, seven sections, further reading), about 55,000 words of prose and 15,000 words of exercises by the section reviewers' counts. I opened the chapter sources and `chapter_optimization/` only to check cross-section claims; what I checked is marked "verified in the source". I did not rescore or re-review any section.

Section reviews:
[index](40-optimization/index.md) ·
[mdl-gradient-based-optimization](40-optimization/mdl-gradient-based-optimization.md) ·
[mdl-convexity](40-optimization/mdl-convexity.md) ·
[mdl-stochastic-gradient-methods](40-optimization/mdl-stochastic-gradient-methods.md) ·
[mdl-adaptive-stochastic-methods](40-optimization/mdl-adaptive-stochastic-methods.md) ·
[mdl-constrained-optimization-duality](40-optimization/mdl-constrained-optimization-duality.md) ·
[mdl-lagrangian-duality](40-optimization/mdl-lagrangian-duality.md) ·
[mdl-numerical-stability-conditioning](40-optimization/mdl-numerical-stability-conditioning.md) ·
[further-reading](40-optimization/further-reading.md)

Finding IDs: GD, CV, SGM, ADA, CON, DUA, NSC, FR, IX, all in the files above. The tags CON and FR may recur in other chapters; here they always mean `40-optimization/mdl-constrained-optimization-duality.md` and `40-optimization/further-reading.md`. Four reviews also number their structural findings S1, S2, ...; I cite those with the tag of the file, for example "SGM S1" (stochastic), "ADA S1" (adaptive), "NSC S1" (numerical stability), "FR S1" (further reading). The reviews of gradient descent, convexity, constrained optimization and duality list structural findings without IDs; I cite them as "GD structural", "CV structural", "CON structural 1", "DUA structural 1".

## Verdict

This is a strong chapter, and in places it is at the bar. Four reviewers independently recomputed several hundred quoted numbers and checked every proof; they found no false theorem and no stored output that contradicts the prose, and each of them names pages they would be content to find in a good optimization or numerical-analysis text: the first three parts of gradient descent, the opening and first part of duality, the first half of numerical stability, the scale-invariance and rotated-valley pages of adaptive methods. The method the Basics review asked for (predict a number from a short argument, then print it) is practised here more consistently than anywhere in the main text, and the exercises are called the best in the book by all four reviewers.

What keeps the chapter below the bar is not the sentences. It is, first, the division of labour: the chapter teaches again, at the same level, things that chapter 9 already teaches (one experiment, four derivations, one exercise, two openers), things other appendix chapters already prove (the Lagrange condition, maximum entropy, the two-line system, the ridge filter factors), and things its own neighbouring sections teach (the rates, the price reading of a multiplier, Slater's condition, the step count 6,908). Second, three sections are two sections each, and in all three the weaker half is the one whose use in deep learning the text itself concedes is small. Third, a handful of experiments show less than the prose says, or something other than it says. Fourth, every section ends twice, with the same formula. The correctness findings are few and local; five need fixing before anything else. What matters most is the first point: decide, item by item, what chapter 9 owns and what this chapter owns, because that decision removes about a tenth of the chapter and makes the rest read as the proof behind chapter 9 instead of a second telling of it.

## Scores

1 to 5 as the section reviewers gave them; 5 means the reviewer would be happy to find the page in Cover and Thomas or MacKay.

| Section | Explanation | Thought | Pedagogy | Language | Enjoyment |
|---|---|---|---|---|---|
| [index](40-optimization/index.md) | 4 | 4 | 3 | 4 | 3 |
| [Gradient Descent and Curvature](40-optimization/mdl-gradient-based-optimization.md) | 4 | 4 | 4 | 4 | 4 |
| [Convexity](40-optimization/mdl-convexity.md) | 4 | 4 | 3 | 4 | 3 |
| [Stochastic Gradient Methods](40-optimization/mdl-stochastic-gradient-methods.md) | 4 | 4 | 4 | 4 | 3 |
| [Adaptive Methods](40-optimization/mdl-adaptive-stochastic-methods.md) | 4 | 3 | 4 | 4 | 4 |
| [Constrained Optimization](40-optimization/mdl-constrained-optimization-duality.md) | 4 | 4 | 4 | 4 | 3 |
| [Duality](40-optimization/mdl-lagrangian-duality.md) | 5 | 4 | 4 | 4 | 4 |
| [Numerical Stability and Conditioning](40-optimization/mdl-numerical-stability-conditioning.md) | 4 | 4 | 5 | 4 | 4 |
| [Further Reading](40-optimization/further-reading.md) | 3 | 4 | 3 | 4 | 3 |

The gradient descent reviewer adds that the first three parts alone would score 5, 4, 5, 4, 4 and the Newton to trust-region parts 3 on pedagogy and enjoyment. Calibration across the four reviewers is close. One visible difference in weighting: cells that confirm a theorem on random inputs cost Convexity a point on pedagogy (3), while the duality reviewer records the same habit as a minor language pattern (DUA-14) and gives pedagogy 4. Read the two pedagogy scores as nearer to each other than they look.

## Correctness

No section review found an error in a theorem or a proof that changes a result. The findings below are all of dimension `correctness` in the section files, ordered by severity, followed by places where prose and stored output disagree and by quoted numbers that no cell prints.

Of substance (fix first):

1. **GD-4** (moderate; sure, verified numerically). The divergence of the fixed step $\eta = 0.3$ on $\tfrac14 x^4$ is explained by "the local stability ceiling is $2/27$". The true threshold from $x_0 = 3$ is $2/9$; the section's own exercise 4(b) asks the reader to prove this, so text and exercise contradict each other. Verified in the source: the claim is at line 390, the exercise at line 1600, and the slide repeats the claim.
2. **NSC-17** (moderate; sure of the logic, numbers platform dependent). The Hilbert example attributes the gap between $\mathbf 1$ and the exact solution of the stored system to the rounding of the matrix entries. Since $\mathbf b$ is built from the stored $\mathbf H$, the gap can only come from rounding in the product `H @ x_true`. The three quoted deviations are computed by no cell and did not reproduce on the reviewer's machine.
3. **GD-8** (moderate; fairly sure). The edge-of-stability subsection says the per-mode recursion "explains why the sharpness stops at the ceiling". It explains the oscillation, not the return of the sharpness. Verified in the source: chapter 9 (`optimization-intro.md` line 390) and the chapter's own Further Reading (line 55, "but not what brings the sharpness back down") both say the quadratic model does not explain this. The chapter contradicts itself here.
4. **CON-6 and DUA-7** (moderate; sure). Slater's condition is defined twice with different hypotheses: strict inequality for every $g_i$ in Constrained Optimization, strict only for the non-affine $g_i$ in Duality, which then calls it "the constraint qualification announced in the previous section" and has an exercise appeal to "its refined form". Verified in the source (constrained lines 393 to 397; duality lines 538 to 548 and 1307). Both statements are true; one name has two definitions. DUA-7 also notes that the SVM paragraph argues the strict form where the affine form suffices, and that "interior of $\mathcal D$" should be "relative interior".
5. **SGM-8** (moderate, correctness of presentation; confident). The proposition on the constant in a $1/k$ schedule is followed by a leading-order matching that is closed with $\blacksquare$, while two of its claims ("for every $c > 0$, $e_k \to 0$" and the $k^{-2c\mu}$ rate) are argued only by approximation. The complete proof is in exercise 2 and is correct. A reader cannot tell what has been proved.
6. **NSC-11** (moderate; fairly sure). "The measured error is $3\times10^{-9}$" for the forward difference at $h = 10^{-8}$: no cell measures it, the reviewer gets $1.4\times10^{-8}$, and the quantity is rounding noise.

Minor (one clause each):

- **SGM-7** (confident): the $1/k$ proposition is silent on the boundary case $2c\mu = 1$, where $e_k = \Theta(\log k/k)$, and exercise 2(d) runs exactly that case.
- **GD-7** (sure): "Raising $\eta$ lowers the left endpoint's value and raises the right one's" holds only between $1/\lambda_{\max}$ and $1/\lambda_{\min}$.
- **GD-12** (fairly sure): Nesterov's rate $1 - 1/\sqrt\kappa$ is for $f - f^\star$, while the proposition just above measures distance; the units change silently.
- **GD-13** (sure): the lower bound holds only in dimension large compared with the number of steps; the exercise says so, the text does not.
- **GD-14** (sure): gradient descent iterates change under every non-orthogonal $\mathbf T$, not only those that are "not a multiple of an orthogonal matrix".
- **CON-10** (sure): "Softmax is the same projection with an entropy term in place of the squared distance" is not a formula; the exact statement is one line and is proved in Duality.
- **CON-13** (minor part; sure): the existence argument for the prox uses quadratic growth without saying why it holds.
- **ADA-24** (confident): the Gauss-Newton matrix is not "the positive semidefinite part of the Hessian".
- **ADA-20** (confident): the fixed-point proposition characterizes fixed points but the text then says both methods "settle" there; existence and stability are not claimed, and the dichotomy is stated only for $\epsilon \to 0$.
- **ADA-17** (not fully checked): three attributions to Reddi, Kale and Kumar need checking against the paper; the paper's third result is stochastic, not periodic.
- **NSC-1** (certain): the bound $|\delta| \le u$ is not attained in the model $\mathrm{fl}(x) = x(1+\delta)$.
- **NSC-8** (low to moderate confidence): `log1p` does form $1 + x$ in common libraries.
- Exercise 2 part 3 of numerical stability (no ID; high confidence, see "Exercises" in that file): none of the test points makes the naive form overflow; $t = 800$ should be $-800$.
- **FR-6** (dimension explanation; high confidence): "constant-step SGD converges linearly" under interpolation needs "on strongly convex or PL objectives".

Prose that the stored output or the run does not support:

- **CON-8** (verified numerically by the reviewer): the projected-gradient run ends in 974 steps because $\eta = 2/1001$ happens to be the inverse curvature along the constraint, so the run takes an exact Newton step once two consecutive steps are projected. The prose compares this accident with the guaranteed 7,660 steps as if it were a rate.
- **ADA S1, ADA-1, ADA-12**: the headline "48 steps" sets optimally tuned gradient descent and heavy ball against Adam at a hand-picked $\eta = 0.1$; the count depends on the start in units of $\eta$ (1 step at $\eta = 1$, 4,145 steps from $(100, 100)$).
- **ADA S2, ADA-13**: the table shows full Adam at 7,908 steps on the rotated valley, "more than gradient descent"; the next sentence retunes it to about 620.
- **ADA S3, ADA-22**: the warmup subsection opens by blaming the unestimated preconditioner and its experiment then shows the preconditioner is not the cause.
- **SGM S3, SGM-17**: a measured slope of $-0.6$ is read as "consistent with" an upper bound of slope $-1/2$; any slope at or below $-1/2$ would be.
- **CON-14**: "within about $0.1$" where the output says $0.040$. **SGM-20**: "about $0.008$" printed as $0.0066$ directly above. **NSC-18**: "of order $10^{-16}$ in every row" where one row prints `0.0e+00`. **NSC-3**: the opening promises three `NaN` that no cell prints. **NSC-4**: the same probability is $3.7\times10^{-44}$ and $3.8\times10^{-44}$ six lines apart (both right, unexplained).
- **DUA-5, DUA-17**: the "previous section's closest-point example" is cited with different numbers from the ones that section uses, twice.

Numbers quoted in prose that no cell prints (all recomputed and found right by the reviewers unless listed above): the 40, 48, 54 steps at ratios $1$, $10^3$, $10^5$ (ADA-10); four facts about the rotated-valley run (ADA-14); the cosine-decay check in a parenthesis (ADA-21); $2.7\times10^{-3}$, the $\Theta(\log K/K)$ rate and "more than an order of magnitude" handed to exercises (SGM-25, the third not run by the reviewer); the finite-difference error and the stored-system deviations (NSC-11, NSC-17, NSC S4).

## Arc and order

The chapter has one line of argument and the introduction states it in a sentence the index reviewer asks to keep verbatim: the steepest curvature sets the largest step, the ratio of curvatures sets the number of steps, convexity decides whether the stopping point is the best one. The seven sections fall into three movements: exact gradients (gradient descent, convexity), noisy gradients (stochastic, adaptive), and constraints and arithmetic (constrained, duality, numerical stability). The introduction lists them as seven sentences of one shape and does not group them (IX-4, with a draft). All reviewers find the order of sections right, including gradient descent before convexity. No section should move.

Three sections are really two:

- **Gradient Descent and Curvature** changes subject after Nesterov, from "how far can a gradient step go" to curvature models: Newton, secant, BFGS, trust regions, 2,800 words (GD structural, GD-16). Split it off as a short section or prune it. The labels there are cited from chapter 9, so the material stays in the chapter.
- **Numerical Stability and Conditioning** is floating point and stable formulas (4,600 words) followed by the conditioning of linear systems (2,750 words), which leans on three other sections (NSC S1, NSC S2). The reviewer prefers cutting the second half to what is new over splitting.
- **Constrained Optimization**: the fourth part (lasso, prox, coordinate descent) is about penalties with kinks, not constraints (CON structural 1). Keep soft-thresholding and proximal gradient, cut coordinate descent to a paragraph.

Stochastic, Adaptive and Duality are each one section. The constrained and duality reviewer was asked whether those two are one section told twice and answers no: one is primal (recognize an optimum, compute it by projection), the other is the lower bound and what the multipliers mean. I agree on the evidence given.

Inside sections, two reorderings are pressed for. In Duality the conjugate (2,000 words) sits between weak and strong duality, so the reader who has just been told a gap can exist waits 240 lines to learn when it does not; move it after "Multipliers as Prices" (DUA structural 1). In Convexity the answer to the opening question ("local minima are global") arrives 4,400 words in, though its proof needs only the chord inequality (CV structural, CV-3); bring it forward, shrink the catalogue of sets, and make Jensen the third "consequence" that the text promises and does not deliver (CV-9).

Where boundaries leak:

- **Gradient descent to Convexity.** The table of rates is stated in the first and proved in the second, with strong convexity defined differently in each, and the two sections tell the "factor of the square" differently (GD-10). The counts 6,908 (six digits of distance, optimal step) and "about 6,900" (three digits of gap, the bound, $\eta = 1/L$) are different quantities that look like one number (CV-12).
- **Constrained to Duality.** Four leaks, listed identically by both reviews: the multiplier as a price (CON-3, DUA-10), the dependence of a multiplier on how the constraint is written (CON-4), Slater's condition (CON-6, DUA-7), and a missing sentence: the KKT sufficiency proof is weak duality at $\boldsymbol\lambda^\star$ and neither section says so (CON-7, DUA-4). The shared example changes its letter at the boundary, from $\mathbf y$ to $\mathbf x_0$ (DUA-1).
- **Uses before definitions.** Adaptive Methods uses projection, "feasible" and KKT (the counterexample, exercises 7 and 10) one section before they are defined (ADA exercises). The trust-region paragraph calls its shift "a Lagrange multiplier" two sections early with no label (GD-18). Exercise 8 of Constrained Optimization needs the conjugate, taught one section later (CON exercises).
- **Numerical stability to gradient descent.** The ridge subsection re-proves the contraction factor and derives the step-count law, a question the first section owns (NSC-21).

## Most important findings

1. **Settle the division of labour with chapter 9, item by item.** The stated division (chapter 9 measures on networks, this chapter proves) is sound and is kept for the stability ceiling, the $\kappa$ law, the heavy-ball tuning, the rates and the Adam counterexample. It breaks in about a dozen places; the list with a proposed home for each is under "Repetition" below. GD-9, SGM S1, ADA S5, IX-1, IX-2, CV structural.
2. **Fix the five correctness findings of substance.** GD-4, NSC-17, GD-8, CON-6 with DUA-7, SGM-8.
3. **Make three experiments say what they show.** Adam's 48 steps (ADA S1, ADA S2, ADA-10: the row 40, 48, 54 is the clean evidence and deserves the cell), the warmup subsection (ADA S3), and the projected-gradient run (CON-8). Add the $\tanh$ neuron (SGM S3). In each case the prose can be corrected now and the cell later.
4. **Split or prune the three double sections,** cutting in each the half the text itself says matters little for deep learning: BFGS and the second-order tail (GD-16, GD-17), coordinate descent (CON-15, CON-16, CON-17), the retold conditioning material (NSC-20, NSC-21).
5. **Repair the boundary between Constrained Optimization and Duality.** One home for the price computation and the scaling remark (Duality), one definition of Slater's condition (Constrained, with the proof in Duality), one letter for the clipped point, and the sentence that links KKT sufficiency to weak duality. Say once that the lower edge of $\mathcal A$ is the graph of $p^\star(u)$, which gives the shadow-price proposition its picture (DUA-9). CON-3, CON-4, CON-6, CON-7, DUA-1, DUA-4, DUA-9, DUA-10.
6. **Reorder Duality and Convexity** as above (DUA structural 1; CV structural, CV-1).
7. **Tell the rates once.** One table, in Convexity; one comparison of bound, quadratic at $1/L$ and quadratic at the optimal step (13,800, 6,900 and 3,454 steps for six digits of the gap); one statement of which factor of two is the proof's and which is the step's; the iteration-bound proposition in the first section. GD-10, CV-12, NSC-21, FR-4.
8. **Let cells find something out, and print what the prose quotes.** Three of seven cells in Convexity confirm what was just proved and a fourth is redeemed only by what it happens to find, and five cells in Duality are introduced by a question whose answer is fixed (CV-6, CV-7, CV-13, DUA-3, DUA-14); at least ten measured values are quoted with no cell (list above). The model for the repair is in the same chapter: the first half of numerical stability.
9. **Give each section one ending.** All seven sections close the body with "We can now ..." or "You can now ..." and then a Summary that repeats it; three also give the results away in the opening. GD-19, CV-16, SGM-23, ADA-27, CON-18, DUA-16, NSC-24; SGM S6, ADA S7.
10. **Name the deep learning use where it is missing.** Control variates are the baseline of policy gradients, and interpolation is why constant-step SGD works on large networks (SGM-18, SGM S4); implicit bias matters because overparameterized networks interpolate (CV-15); projected gradient ascent under an $\ell_\infty$ budget is adversarial training (CON-9); an ill-conditioned solve occurs in GPTQ and K-FAC damping (NSC S6); the penalty and constraint correspondence must cover $\ell_1$, since the main text cites it for the lasso (DUA-11, CON-11); Further Reading should rank its thirteen topics by relevance to training networks (FR S1).

## Repetition

### With chapter 9 (`chapter_optimization`): one list, with a proposed home

Two rules decide almost every case, and both are in the section reviews. An experiment on a network belongs to chapter 9; if the appendix runs it too, cut it here (index review, "For the chapter review", citing guide 7.6). A derivation or proof belongs here; chapter 9 states the result with its intuition and points here (SGM S1). The arrangement for heavy ball is the model: `momentum.md` states the optimal tuning and cites `subsec_mdl-momentum-acceleration` for the proof (GD structural).

| # | Item | Where it is taught twice | Proposed home | What the other side keeps | Findings |
|---|---|---|---|---|---|
| 1 | Edge-of-stability experiment (25-parameter $\tanh$ network, 16 points of $\sin 3x$, $\eta = 0.4$ and $0.25$) | `optimization-intro.md` and "The Edge of Stability" here. Verified in the source: same network, data and steps; the appendix subsection does not cite chapter 9 and has no label | Chapter 9 | One paragraph here (what the per-mode recursion says at the ceiling, with chapter 9's careful statement and the Damian et al. citation) plus a label, because ADA-23 and the Further Reading topic need a target. Removes two cells and about 450 words | GD-9, GD-8, ADA-23 |
| 2 | Newton step from the Taylor expansion, and its $O(d^2)$, $O(d^3)$ cost | `gd.md` and "Newton's Method" here | Chapter 9 for step and cost | Here: the quadratic-convergence proof, affine invariance, the saddle that attracts; cite step and cost in a clause each | GD structural |
| 3 | Noise-ball recursion and its fixed point $\eta\sigma^2/(\lambda(2 - \eta\lambda))$ | `sgd.md` "The Noise Ball" and "The Noise Floor" here, same steps, nearly the same words. Verified in the source: both derive the same formula, and `sgd.md` contains no reference to this chapter at all | Here | Chapter 9 states the formula and the picture and points to `subsec_mdl-stochastic-gradients`. Also align the two valleys (curvatures $0.2$ and $4$ there, $1$ and $1000$ here) and the step index ($t$ there, $k$ here) | SGM S1 |
| 4 | Robbins-Monro conditions with their glosses, and the remark that the square-summable one is not needed on a quadratic | `sgd.md`, `lr-scheduler.md` "Progress and Noise", and here | Here | A sentence and a pointer in chapter 9 | SGM S1, ADA-5 |
| 5 | Schedule comparison on a quadratic valley | `lr-scheduler.md` "Four Schedules on the Valley" (simulated) and "Schedule Shapes under the Exact Recursion" here (exact, no random numbers) | Here, since the exact recursion is the better experiment | Chapter 9 keeps schedules on networks and cites the exact ranking. This one changes chapter 9 most; it is the author's call | SGM verdict, SGM "For the chapter review" |
| 6 | AdaGrad, RMSProp, Adam derived in order | `adam.md` and three subsections here | Chapter 9 | Here: the three update equations with one sentence each, since the propositions are about them and some readers start here | ADA S5 |
| 7 | Bias-correction computation (the $1\%$ after ten steps, the first-step factor $3.2$) | `adam.md` and "Adam and Its Bias Correction" here | Chapter 9 (the reviewer leaves the choice open; the computation is needed to use Adam, so I propose the main text) | Here only what is new: the mis-scaling peak of $6.6$ at $t = 12$ | ADA S5, ADA-7 |
| 8 | Coupled against decoupled decay: the display and the sentence about large and small gradients | `adam.md` and "Coupled and Decoupled Decay" here | Chapter 9 for display and intuition | Here: the equation, the noise experiment, the fixed-point proposition | ADA S5 |
| 9 | Why warmup | `lr-scheduler.md` (curvature) and "Warmup" here (momentum average) | Each keeps its mechanism | Fix the pointer: verified in the source, `lr-scheduler.md` line 544 cites `subsec_mdl-schedules-warmup` for "those estimates are unreliable in the first steps", which this subsection's experiment refutes | ADA S3, ADA-22 |
| 10 | Muon and the structured preconditioners | `muon.md` teaches Muon from scratch; "K-FAC, Shampoo and Muon" here surveys three methods | Chapter 9 for Muon | Here: the identity $(\mathbf G\mathbf G^\top)^{-1/4}\mathbf G(\mathbf G^\top\mathbf G)^{-1/4} = \mathbf U\mathbf V^\top$ and the form $\mathbf P_{\mathrm l}^{-a}\mathbf G\mathbf P_{\mathrm r}^{-a}$; halve the rest. Verified in the source: `muon.md` twice promises a "ladder" at `subsec_mdl-preconditioning-ladder`, and the word survives here only in the label | ADA S6 |
| 11 | Permutation-symmetry argument for nonconvexity, and its exercise | `optimization-intro.md` "The Role of Convexity" with `ex_optimization-intro-permutation-symmetric-minima`; "Nonconvex Networks" here with `ex_mdl-convexity-permutation-symmetry` | Here (the version with two ReLU units computing $|x|$ is the more concrete) | Chapter 9 cites it and keeps its $d!$ count; drop part (d) of the exercise here, which repeats chapter 9's parts (a) and (b) | CV structural, CV exercises |
| 12 | Opening sentence ("Every network in this book was trained ...") and the Cauchy 1847 anecdote | Both chapter introductions. Verified in the source | Chapter 9 for Cauchy | Here: open on what chapter 9 left unproved; a history of the theory (Polyak, Nemirovski and Yudin) if one is wanted | IX-1, IX-2 |
| 13 | Noise scale and batch size on networks | `batch-size.md` and exercise 1(c) of Stochastic Gradient Methods | Chapter 9 | The exercise keeps (a) and (b) and points to `sec_batch_size` | SGM exercises |
| 14 | Heavy-ball optimal tuning; the stability ceiling; the $\kappa$ law | Stated in chapter 9, proved here | Already right | Keep as the model | GD structural |

Three smaller items of the same kind involve other main-text chapters. The max shift for log-sum-exp is explained in `subsec_softmax-implementation-revisited`, in `sec_numerics` and here; the reviewer accepts this because only this section has the bounds, but the three texts disagree on small things (`bf16` against `bfloat16`; adding against subtracting $10^{-3}$, hence thresholds $2^{-8}$ against $2^{-9}$) (NSC S5, NSC-2, NSC-7). The without-replacement factor $(n - b)/(n - 1)$ is credited to `sec_sgd`; a two-line exercise deriving it fits here (SGM exercises). And "Numerical Stability" names both this section and `sec_numerical_stability` of chapter 4, which is about Jacobian products; a clause in each opening would prevent a wrong landing (NSC S5).

### With other appendix chapters

- **The Lagrange condition** is stated and proved in the calculus chapter (`subsec_mdl-lagrange`) with the opposite sign and the symbol $\mu$, which here is the strong-convexity constant. Constrained Optimization restates it, retells the proof, and spends a paragraph reconciling the conventions. Home: the calculus chapter for the proof, ideally changed to the plus-sign form; here the tangency figure and a citation (CON-2, CON structural 6).
- **Maximum entropy and softmax.** The statement that softmax has the largest entropy at a given expected score, and all of exercise 2 of Duality, are the proposition of `subsec_mdl-max-entropy-derivation` in chapter 41, which is not cited. Home: chapter 41; keep exercise 2(c) here (DUA-6).
- **The two-line system and the amplification bound** are in `subsec_mdl-condition-number` (chapter 38) with the same matrix, right-hand side, singular values and $\kappa$. Home: chapter 38 for the numbers; here the picture of two lines and the reading as small backward error (NSC S2, NSC-15, NSC-16).
- **The ridge filter factors** are `eq_mdl-ridge-filter` in chapter 38, with $\gamma$, $\mathbf x$ and no factor $\tfrac12$; here a proposition with proof, with $\lambda$ and $\mathbf w$. Home: chapter 38; here one sentence and the new observation that follows (NSC-20).

### Inside the chapter

- **The rates table** and the definition of strong convexity: in the first section and in Convexity. Home: Convexity (GD-10).
- **Proof slack against the quadratic** is told three ways: "the same units" (gradient descent), "the proof gives away a square" (Convexity), and $(1 - \mu/L)^2$ in Further Reading. Home: the rate-check passage of Convexity (GD-10, CV-12, FR-4).
- **The step count 6,908** and the contraction factor: derived in the first section, where the reviewer of the last counted it at least seven times, quoted in Convexity and Stochastic, and re-proved as a proposition in numerical stability. Home: the first section (NSC S2, NSC-21, CV-12).
- **The multiplier as a price,** with $p^\star(s) = \tfrac12(5 - \sqrt{2s})^2$ and its derivative $-4$. Verified in the source: computed in Constrained Optimization (line 302), as the lower edge of $\mathcal A$ in Duality (line 604), and again in "Shadow Prices" (line 805), in three notations. Home: Duality, once, with the edge and the function identified (CON-3, DUA-9, DUA-10).
- **"A multiplier depends on how its constraint is written."** Home: Duality (CON-4).
- **Slater's condition.** Home: definition in Constrained Optimization, in the refined form; proof in Duality (CON-6, DUA-7).
- **Subgradients** are defined in Convexity with a figure and again, informally and without a reference, in Constrained Optimization (CON-12).
- **Opening experiments told twice:** the bowl and the double well in Convexity (CV-1), the $0.053$ and $0.23$ in Stochastic (SGM-5), the outcomes of the rotated valley and the counterexample in the opening of Adaptive (ADA S7, where 48 appears seven times).
- Smaller: the operational reason for WSD three times (SGM-12); two kinship remarks for water-filling (DUA-13); a sentence repeated word for word forty lines apart (DUA-3).

## Recurring patterns

**Double endings, 7 of 7 sections.** Verified by grep: "You can now predict ..." closes the first and the last section, "We can now ..." the five between, and a Summary follows each. In five of the seven the second half of the closing paragraph is a real ending (what the section did not cover, what remains open); every reviewer says to keep that half and delete the recap.

**Lead-ins to code.** "The cell ..." opens ten lead-ins in the first section (GD-20). In Convexity six of seven cells are introduced by a question followed by "The cell ...", four of them yes-or-no questions whose answer the preceding proof has fixed (CV-18). Duality has five such questions (DUA-14). The lead-ins that work ask for a quantity ("by how much?") or state a prediction.

**Cells that cannot fail.** Convexity: `convexity-three-lenses`, `convexity-jensen-mc` ("AM >= GM in every draw: True"), `convexity-lse-hessian` (CV-6, CV-7, CV-13). Duality: the first cell evaluates the closed form just derived (DUA-3). Stochastic: the $\tanh$ neuron (SGM S3). Exercise 10(c) of Constrained Optimization states the numbers and asks the reader to measure them. Against these stands the chapter's own counterexample, the subgradient cell whose answer to "Does it find the five zeros?" is no.

**The featured method is not tuned like the baseline, or wins by accident.** ADA S1, ADA S2, CON-8. The adaptive reviewer asked that other sections be checked for this; the constrained reviewer found the second case independently.

**Small templates, with the reviewers' counts.** "the previous section" without a label, 8 times in Duality (DUA-2); references by description instead of label, 4 in the first section (GD-15); "Thus" after a definition, 4 (DUA-15); "the step that is not routine", 3 (SGM-9); "Intuitively,", 3 (SGM-10); "part" for a `##` unit, 5 (NSC-6); two families of format names in one section, fp32 12 times and float32 14 times (NSC-7).

**Results handed to exercises.** Ten in Stochastic (SGM-25), at least six in Adaptive (ADA "For the chapter review"), four in numerical stability. Acceptable in an appendix when the text says "the exercise computes"; not when the text then also gives the answer away (ADA-14 with exercise 8(b)) or when Further Reading relies on an exercise as if it were text (FR-2).

**Unused framework imports.** Four of the seven sections (Stochastic, Adaptive, Constrained, Duality) are NumPy only and still carry a four-tab imports cell that loads a framework no cell uses (SGM "Code", ADA "Code", CON-19, DUA-18).

**Stale identifiers,** invisible to the reader and a trap for an editing agent: exercise labels and cell IDs that carry the names of other notebooks (SGM "Code" and exercises, CON exercises, DUA exercises); the file and label `mdl-constrained-optimization-duality`, which holds no duality; `subsec_mdl-preconditioning-ladder`. Difficulty marks are missing on some exercises in Stochastic and Adaptive.

**Notation drift.**
- Ridge or weight-decay coefficient: $\nu$ in the first section, where the chapter table reserves $\nu$ for equality multipliers; $\lambda$ everywhere else in the chapter; $\gamma$ in chapter 38 (GD-3, IX-6, NSC-20).
- The clipped point: $\mathbf y$ in Constrained Optimization, $\mathbf x_0$ in Duality (DUA-1). The closest-point example has three sets of numbers (DUA-5, DUA-17).
- $\rho$: contraction factor in the chapter, a radius once in Duality, the ADMM penalty in Further Reading (DUA-9, FR-7).
- $\mu$: strong-convexity constant here, the Lagrange multiplier in chapter 39 (CON-2).
- Step index $k$ in the first three sections, $t$ in Adaptive and in chapter 9; $\sigma^2$ and $\boldsymbol\Sigma$ change meaning between Stochastic and Adaptive; the constant $A$ shares a section with the data matrix `A` (SGM-24, ADA-26). The table admits $k$ against $t$.
- $f'$ for a second function (ADA-8); $\star$ against $*$ announced late (DUA-8); "sign step" used in three senses (ADA-11); "transient" for two different terms (SGM-15); the schedule called $c/k$ and $c/(k + k_0)$ (SGM-13).
- Test problems multiply: four quadratics in the first section and a fifth in chapter 9 (GD structural, GD-6), five problems in Stochastic (SGM S2), a figure at $\kappa = 20$ where the text uses $10^3$ (ADA-15, NSC-16). The introduction's "running example of every section" overstates: the $\kappa = 1000$ bowl is worked in Stochastic, Adaptive, numerical stability and once in Constrained Optimization, quoted but not worked in the first two sections (IX-3), and absent from Duality, whose example is the clipping problem.

**Voice.** No reviewer reports the voiceless register of the Basics chapters as a chapter-wide fault. Each quotes sentences with a voice that are also literally true ("it touches $\mathbf x^\star$ but does no work"; "A value $-\infty$ is a true but useless bound"; "a negative output is the exact variance of no data at all") and asks that they be protected. Enjoyment drops to 3 where a section turns to inventory: the catalogue of sets, variance reduction, coordinate descent, the survey of preconditioners, the even-toned entries of Further Reading.

## Connection to the main text

Stale or wrong pointers:

- `lr-scheduler.md` line 544 cites `subsec_mdl-schedules-warmup` for a claim the subsection refutes (ADA S3). Verified.
- `muon.md` lines 141 and 925 promise a "ladder" that the cited subsection no longer contains (ADA S6). Verified.
- `chapter_linear-regression/weight-decay.md` cites `subsec_mdl-weight-decay-duality` for the ridge and the lasso correspondence; only $\ell_2$ is proved (DUA-11). Constrained Optimization promises the same for $\ell_1$ and no section delivers it (CON-11).
- A brief in `chapter_lm-rl/dpo.md` cites `sec_mdl-constrained-optimization-duality` for the penalty and constraint correspondence, which lives in the other section (CON "For the chapter review").
- Further Reading attributes iterative refinement to the numerical-stability section, where it exists only as the last exercise (FR-2).

Missing pointers from here to the main text. By my grep of chapter 9's section labels before the slides marker, the first section cites chapter 9 once (`sec_muon`), Convexity once (`sec_adam`), and Constrained Optimization, Duality and numerical stability not at all. Wanted: `sec_optimization-intro` from the edge-of-stability paragraph (GD-9); `sec_scheduler` from the line-search remark (GD-5); `sec_baselines` from control variates (SGM-18); the GAN chapter from saddle-point methods (FR-8); a deep learning site for an ill-conditioned solve (NSC S6); adversarial examples from projected gradient descent (CON-9).

Missing pointers from the main text to here: `sgd.md` derives the noise ball with no reference to this chapter (item 3 of the table). The introduction says chapter 9 "left the proofs to this chapter" and does not say which section proves what; one sentence would give the reader arriving from chapter 9 a path (index review, structural). The scope sentence "We assume gradients, Hessians and eigenvalues" should carry the two labels where a reader finds them (IX-5).

Lookup use. The first section stands up well to a reader landing on one of its subsections, because each proposition restates its recurrence (GD structural). Duality does not, because of "the previous section" (DUA-2). In Stochastic a paragraph that defines the noise model of the valley sits 160 lines before the valley (SGM S5, SGM-3), and its figure shows a warmup that the text has not defined and the cell omits (SGM-11).

## What to cut

About 6,000 words by the reviewers' estimates, none of it argument:

- Gradient descent: the edge-of-stability experiment down to a paragraph (GD-9); the BFGS update, its two bullet proofs, the Wolfe paragraph and the six-dimensional cell, or else a two-sentence motivation in their place (GD-16, GD-17); the "You can now" paragraph (GD-19); the forward-stated rates (GD-10). About 900 words and three cells.
- Convexity: polyhedra, the PSD cone and the hull out of "Convex Sets"; the second telling of the opening experiment; the boundary hypotheses of the uniqueness proposition; the one-dimensional supporting-line proof to an exercise (CV-3, CV-1, CV-11, CV-8, CV-5). About 850 words.
- Stochastic: the shared core with chapter 9 only if chapter 9 becomes its home, which I do not propose; a third of variance reduction; the second $0.053$ example; the recap (SGM S7). About 800 words.
- Adaptive: the duplicated derivations and displays; half of "K-FAC, Shampoo and Muon"; two paragraphs of the opening (ADA S5, ADA S6, ADA S7). About 900 words.
- Constrained: coordinate descent to a paragraph; the retold Lagrange proof and the reconciliation paragraph; the two price paragraphs; the prox existence detour; the recap (CON structural 1, CON-2, CON-3, CON-13, CON-18). About 1,500 words.
- Duality: the second half of the first cell, the third derivation of the clipping curve, the offset-folding passage of the SVM, the recap (DUA-3, DUA-9, DUA-12, DUA-16). About 500 words.
- Numerical stability: the filter-factor and iteration-bound propositions and the restated two-line numbers (NSC-20, NSC-21, NSC-15). About 1,000 words.
- Further Reading: the list of topics in the opening (FR-1).

One caution where two reviews pull apart. The convexity review would move polyhedra and the hull into Constrained Optimization, which its own reviewer wants shorter by 1,500 words. Unless the constrained section turns out to need them, cut them instead of moving them.

The October review asked to cut BFGS, trust regions, the edge of stability and coordinate descent. All four are still present. The current reviewers keep the trust region, because it completes the saddle example, and repeat the other three.

## What is missing

- A full-matrix, rotation-equivariant preconditioner applied to the rotated valley: the section shows the failure and promises a repair it does not make (ADA S4, ADA-16).
- The cell that prints 40, 48, 54 (ADA-10).
- The reason implicit bias matters for networks that interpolate (CV-15), and that interpolation removes the noise floor (SGM-18).
- Control variates as the baseline of policy gradients (SGM-18).
- Adversarial training as the deep learning use of projected gradient descent (CON-9).
- The $\ell_1$ case of the penalty and constraint correspondence (DUA-11).
- The sentence that $\kappa(\mathbf A)$ is the condition number $\kappa_f$ of the map $\mathbf b \mapsto \mathbf A^{-1}\mathbf b$; without it the last section is two arguments (NSC S3, NSC-15).
- A proper paragraph on the gradient check and the step $h \approx \sqrt u$ (NSC-10); one sentence on why parallel sums are not reproducible across runs and devices (NSC "For the chapter review").
- An honest account of Welford's stability, marked as a heuristic with the citation for the proof (NSC-12).
- A Jensen exercise on a loss, for example ensembling (CV exercises); an exercise for the without-replacement factor (SGM exercises).
- In the introduction: a grouped route and a reading path from chapter 9 (IX-4). In Further Reading: a ranking, glosses for half a dozen undefined terms, and perhaps sharpness-aware minimization as the sequel to the edge of stability (FR S1, FR S2, FR S3).

## What is best and should set the model

- **Gradient descent, parts 1 to 3.** The opening promises three counts (69, 6,908, 315), the text derives each and a cell prints each. The proof of the optimal step by a weighted average and the heavy-ball law from "the product of the two roots is $\beta$" are the house style for proofs at its best, with the figure drawn from the proof.
- **Numerical stability, up to "Summation Order".** A prediction before nearly every cell and a printed number after it: the multiple of 128, the increment 0.125, $149\log 2 = 103.2789$. The diagnosis of $\log(1 + x)$, which builds a tool to answer "is the function hard or is the formula to blame?", is named the best page of the section.
- **Duality, the opening and first part.** A solver stops at $8.01$; "Is that good enough?"; a table of bounds before any definition; and "Even $\lambda = 3$, picked without solving anything, proves that it is within $0.135$." Also the strong-duality proof with the paragraph that says where each hypothesis was used, and the SVM passage on a dual solution that is a segment.
- **Constrained optimization, one example carried through.** Clipping $(3, 4)$ to the unit disk is the tangency picture, the active KKT case, the projection residual and the supporting line. Necessary against sufficient is taught by a counterexample each time. The simplex projection, whose cell shows that three KKT residuals vanish for a wrong threshold too, is the best subsection.
- **Adaptive methods, the symmetry and where it ends.** The scale-invariance proposition, the rotated valley with its two-cycle at $250\,\eta^2$, the counterexample read in one period, the fixed points of AdamW. The reviewer notes these are not in the standard texts.
- **Stochastic methods, one recursion used four times.** $e_{k+1} = \rho^2 e_k + \eta^2\sigma^2$ gives the floor, the time to reach it, the $1/k$ constant and the exact schedule comparison. The result that the stiff coordinate holds two thirds of the loss floor is the kind of surprise the bar asks for.
- **Convexity, the opening and the close.** A share of runs predicted before running ($0.458$), the chord gap that returns as the variance in Jensen, the log-sum-exp Hessian read as a covariance, and "Beyond Convexity", the part a deep learning reader came for.
- **Candour about limits,** in every section: where a model stops, where a bound is loose, where a hypothesis fails. The introduction promises it and the sections keep the promise.
- **The exercises.** Ten to twelve per section, graded, almost none a drill, several of them small research results made accessible. Each reviewer calls the set the best in the book. They should be the model for the other appendix chapters.

## Where section reviews disagree or need reconciling

- **Edge of stability.** GD-9 would reduce the subsection to a paragraph; ADA-23 wants the warmup subsection to cite it by label as the appendix's own account of the curvature mechanism, and the Further Reading topic on self-stabilization builds on it. Verified in the source: the subsection has no label today. Resolution: keep one paragraph, corrected as in GD-8, and give it a label.
- **Where the convex-sets catalogue goes** (above, under "What to cut").
- **Iteration-bound proposition.** NSC-21 would move it into the first section, which its own reviewer finds too long at 9,300 words. Compatible if the edge-of-stability experiment and the rates table leave that section first.
- **Who points where for the noise ball.** The stochastic review says chapter 2 and chapter 9 both point to this section for the noise cloud. Verified in the source: `sgd.md`, where chapter 9 derives it, has no reference to this chapter; the references from chapter 9 to this section are in `lr-scheduler.md` and its further reading.
- **Home of the bias correction** is left open by ADA S5; I propose chapter 9 (table, item 7).

## Could not check

I verified in the sources only the cross-section claims marked as such. The section reviewers report as unchecked: the TensorFlow, JAX and MXNet outputs (cells in four sections are untabbed NumPy and should not differ); three attributions to Reddi, Kale and Kumar (ADA-17); the claim that the convex bound is loose "by more than an order of magnitude" (SGM-25); platform-dependent numbers in NSC-11 and NSC-17; the citations' theorem numbers in Duality. Proposed cell changes are untested, and under the author's decision that experiments wait they are recorded as prose fixes now and cell changes later.
