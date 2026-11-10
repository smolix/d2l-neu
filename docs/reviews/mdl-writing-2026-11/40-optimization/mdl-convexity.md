# Convexity

Source: `chapter_mdl-optimization/mdl-convexity.md`, about 8,800 words of prose before the exercises and 2,500 words of exercises (code excluded; 7 PyTorch cells besides the imports).

## Verdict

A correct, well-organized account of convexity for a deep learning reader, with a good opening, several memorable moments and a strong last part, and with a long conventional middle. The opening experiment (500 runs on a bowl and on a tilted double well, 46% of them stranded one unit above the optimum) poses the question of the section in a way a reader feels, and the last part ("Beyond Convexity": the two-weight network $(ab - 1)^2$, the PL inequality, implicit bias) answers the objection every deep learning reader has, that their losses are not convex. Between the two, the section follows the order of a convex-analysis course (sets, three characterizations, strict and strong, subgradients, Jensen) for 4,400 words before the reader gets the first consequence that the opening promised. I found no mathematical errors: I checked every proof and recomputed every worked number against the stored outputs. The weaknesses are of pedagogy. Several cells confirm by sampling a statement that was proved a page earlier ("AM $\ge$ GM holds in every draw"), which teaches nothing; the text itself says "Sampling can refute convexity but cannot prove it" and then never lets sampling refute anything. And the opening experiment is told twice. The most important fix is the order and economy of the first half: bring the first consequence (local minima are global) forward, trim the catalogue of sets, and make each cell find something out.

## Scores

| Clarity of explanation | Clarity of thought | Pedagogy | Language | Ease and enjoyment |
|---|---|---|---|---|
| 4 | 4 | 3 | 4 | 3 |

## What works

- **The opening experiment, and the prediction attached to it.** The cell computes the local maximum of $\phi$ as a root of $\phi'$, predicts the share of runs that go right, $(1.5 - 0.127)/3 \approx 0.458$, "a prediction made before running anything", and compares it with the observed share. A number is predicted from a short argument and then measured.
- **A thread from the chord gap to the variance.** Under "The Chord Condition" the gap for $x^2$ is computed as $\theta(1-\theta)(x-y)^2$, with the remark "This gap will return as the variance in Jensen's inequality." It does, under "Jensen's Inequality" ("is that distribution's variance") and again in the numerical check ("the vector form of the gap computed for $x^2$ above"). This is how a section acquires unity.
- **Log-sum-exp.** "The curvature of log-sum-exp in a direction $\mathbf{v}$ is the variance of the entry $v_Y$ picked out by a random class." The proof by reading the Hessian as a covariance is the elegant one, it gives the flat direction $\mathbf{1}$ for free, and the consequences (cross-entropy convex in the logits, softmax regression convex in the weights, shift invariance) are the ones the reader needs. The exercise that generalizes it to log-partition functions is the right next step.
- **The counterexample that returns the section's own example.** In the monotone-composition rule: "$h(t) = t^2$ is convex but not nondecreasing, and with the convex $g(x) = x^2 - 1$ it composes to the double well $(x^2 - 1)^2$, which becomes $\phi$ after adding $x/2$." The reader learns where $\phi$ came from and why the hypothesis is needed, in one sentence.
- **"The proof gives away a square."** The rate-check cell measures $0.81$ where the proposition promises $0.9$; the text explains why in the eigenvector coordinates, says which sharper result is known, and gives an exercise that proves a weaker version. Bound, measurement, explanation of the gap: exactly right.
- **The proofs are the short ones.** Jensen by one supporting hyperplane ("the supporting hyperplane at $\mathbf{m}$ holds for every outcome at once"); local implies global in three lines followed by the sentence "In words: ..."; the smooth convex rate with the candid signpost "The non-routine step is to complete the square".
- **Small true sentences.** "every probability distribution is an average of certainties"; "The condition and the conclusion are the same inequality, so each implies the other"; *the smallest "deep" network*; "which gives a rate for the training loss and says nothing about generalization".
- **$x^2$ against $x^4$.** One loop, two functions, the product $x_k\sqrt{k} \to 1$ printed and then proved in an exercise. It shows what strong convexity is for better than the definition does.

## Structural findings

- **The payoff is 4,400 words from the question.** The opening asks when the end of a run is the best point. The answer ("local minima are global") arrives under "Consequences of Convexity", after the whole apparatus, although its proof uses nothing but the chord inequality and would fit directly after "The Chord Condition". Recommended order: opening; chord condition, with the local-global proposition and the double-well cell immediately after it; then first- and second-order tests; then strong convexity and the rates together (the $x^2$ against $x^4$ cell is the natural lead-in to the linear rate, and today the two are 500 lines apart); then subgradients and Jensen; then the calculus, log-sum-exp, and "Beyond Convexity" unchanged. Convex sets shrink to what the argument uses (next finding).
- **"Convex Sets" is a catalogue placed before the reader needs it.** It runs 870 words: four families, the two meanings of $\succeq$, intersections, affine maps, polyhedra, the convex hull. This section then uses sets three times (the domain must be convex, sublevel sets, the minimizer set), and the epigraph once. Polyhedra, the PSD cone as an intersection of half-spaces, and the hull are material for :numref:`sec_mdl-constrained-optimization-duality`, where feasible sets are the subject. Keep the definition, the annulus, the simplex, norm balls and the intersection proposition (about 350 words); move the rest to the constrained section, or leave it and accept a slow start. See CV-3.
- **Four of seven cells verify instead of finding out.** `convexity-three-lenses` samples chords and tangents of least squares, whose convexity the text proved in one line ($\|\mathbf{X}\mathbf{v}\|^2 \ge 0$). `convexity-jensen-mc` prints "AM >= GM in every draw: True" and "KL >= 0 for every pair: True". `convexity-lse-hessian` confirms a derivative. `convexity-rate-check` asks "Does the measured contraction obey the bound" (it must). The last of these is redeemed by what it finds (the square); the first three are not. The house rule is that code computes or demonstrates something the prose discusses; a theorem checked on random inputs demonstrates only that the code has no bug. See CV-6, CV-7, CV-13. Changes to cells are deferred by the author's decision 3; the writing pass can at least stop presenting these as tests whose outcome is open.
- **The opening experiment is told twice** (CV-1).
- **Jensen's inequality is filed under "Convex Functions" though it is a consequence**, and it is the passage most cited from outside (`mdl-information-theory.md`, `mdl-divergences-distances.md`, `mdl-maximum-likelihood.md` all point to `subsec_mdl-jensen`). A reader who lands there needs the subgradient, defined two subsections up, and nothing else; that works. It would sit more naturally as the first subsection of "Consequences of Convexity", whose own introduction promises "three answers" and delivers two (CV-9).
- **Overlap.** With the main text: `chapter_optimization/optimization-intro.md` ("The Role of Convexity") makes the permutation-symmetry argument in a paragraph and sets it as an exercise (`ex_optimization-intro-permutation-symmetric-minima`); this section makes it again under "Nonconvex Networks" and sets it again (`ex_mdl-convexity-permutation-symmetry`). The appendix version is the more concrete one (two ReLU units computing $|x|$); chapter 9 could cite it in a clause and keep its $d!$ count, or this section could drop its exercise part (d), which repeats chapter 9's part (a) and (b). With the previous section: the descent lemma, $L$-smoothness and $\kappa$ are restated in a sentence each and cited, which is the right amount; the table of rates is stated there and proved here (GD-10 in the previous file).
- **Length.** 8,800 words. The reordering above removes "A Bowl and a Double Well" as a separate subsection, the trimmed sets save about 500 words, and CV-11 and CV-14 another 350. Nothing in "Beyond Convexity" should be cut; it is the part a deep learning reader came for.

## Passage-level findings

**CV-1** (moderate, pedagogy). Opening, second paragraph; and "A Bowl and a Double Well".
Quote: "roughly half of the runs, 46% in expectation, stop at $x \approx 0.93$, where"
The opening reports the experiment in full (500 runs, 46%, $x \approx 0.93$, $\phi \approx 0.48$, $-0.51$, $-1.06$) with no code. Seven hundred lines later "A Bowl and a Double Well" runs it and reports the same numbers again ("at $x \approx -1.0575$ with $\phi \approx -0.515$ and at $x \approx 0.9304$ with $\phi \approx 0.483$, a gap of almost one unit"). Guide §7.7: a result is told once. The loop needs no theory, so the cell can sit in the opening, where the reader would see the split happen before being told what it means; "Consequences" then needs one clause ("which is what the 500 runs on the bowl showed"). If the cell must stay where it is, reduce the opening to the phenomenon and one number: "about half of the runs stop in a well whose value is almost one unit above the best".

**CV-2** (minor, thought). Opening, third paragraph.
Quote: "this property are called *convex*. For a convex function, information at"
The phrase "this property" follows a sentence that names two (chords above the graph for the bowl, chords that dip below for the well). Write "Functions whose chords all lie on or above the graph are called *convex*."

**CV-3** (moderate, pedagogy). Under "Definition and Examples".
Quote: "Four families of convex sets appear throughout this book."
A list of four families with a use each ("the $\ell_2$ balls of weight decay and gradient clipping", "attention weights, class probabilities, and mixture weights", "covariance matrices and the Hessians of convex functions"), then a paragraph on notation, then polyhedra and hulls. None of it is wrong and the uses are well chosen, but the reader was promised an answer about minima and is given an inventory. See the structural finding. If the catalogue stays, cut the paragraph "The symbol $\succeq$ (and its mirror $\preceq$) carries two senses" to a clause at the first matrix use; the chapter's notation table is the place for it.

**CV-4** (minor, language). Under "The Chord Condition".
Quote: "about convex sets into facts about convex functions. The first such fact"
"This correspondence turns facts about convex sets into facts about convex functions. The first such fact runs the other way". A fact that runs the other way is not such a fact. Write: "The link also runs from functions to sets."

**CV-5** (minor, explanation). Under "The Second-Order Condition".
Quote: "(on a closed full-dimensional domain, test the interior and require"
A four-line parenthesis about closed and lower-dimensional domains sits inside the statement of the test, before the reader has seen the test used once. The October review called this kind of thing "Real-analysis hygiene". The losses of this book live on $\mathbb{R}^d$. Move the remark after the three examples, as one sentence, or to an exercise; keep the nice example ($x^2 - y^2$ on the line $y = 0$).

**CV-6** (moderate, pedagogy). Under "A Numerical Check of the Three Tests".
Quote: "Do the three tests agree on a concrete loss? The cell checks them on the"
Three problems. The check comes two subsections after the proposition it illustrates (strong convexity and subgradients intervene). Its outcome is not in doubt: the same loss was shown convex in one line under "The Second-Order Condition". And its best sentence, "Sampling can refute convexity but cannot prove it", describes something the cell does not do. Fix (cell change, deferred): run the same three tests on a function where they find something, the double well $\phi$ or $(ab - 1)^2$, print the worst chord gap (positive) and the points that produce it, and keep least squares as the control. In the writing pass: move the subsection up to follow the proposition, and replace the question by the statement of what the gaps should equal ($-\tfrac12\theta(1-\theta)\|\mathbf{X}(\mathbf{u} - \mathbf{v})\|^2$), which the text gives only afterwards and the cell could verify to rounding, a sharper test than a sign.

**CV-7** (moderate, pedagogy). Under "Jensen's Inequality", the cell and its reading.
Quote: "Monte Carlo error but not this gap. AM $\ge$ GM holds in every draw, and the"
Two of the four printed lines are theorems confirmed on random inputs. The first line ($1.65$ against $1.00$, with "more samples shrink the Monte Carlo error but not this gap") and the last (the divergence falling a hundredfold per tenfold $\varepsilon$, explained by $\tfrac{\varepsilon^2}{2}\sum_x (q(x) - p(x))^2/p(x)$) both teach something. Drop the two bulk checks from the prose now and from the cell later, and let the lead-in ask the question the cell answers: how does the bound $0$ get approached?

**CV-8** (minor, pedagogy). Under "Subgradients".
Quote: "**Proposition (supporting lines in one dimension).** *Let $f$ be convex on an"
Of the proofs in the section this is the one that least earns its 230 words. It establishes subgradients in one dimension; the two places that use existence (Jensen for random vectors, the uniqueness proposition) need $d$ dimensions, which the text then cites without proof. The three-slope inequality is elegant but is not used again. Options: state the $d$-dimensional fact with its citation and move the one-dimensional proof to an exercise (it is a good one), or keep it and say plainly that it is shown because it is short, not because it suffices.

**CV-9** (minor, thought). Opening of "Consequences of Convexity".
Quote: "mean? For convex functions there are three answers. A local minimum is"
Two answers follow (local is global; steps add up to rates). Either Jensen was the third and has moved, or the count is wrong. Fix the count, or adopt the structural suggestion and make Jensen the third.

**CV-10** (minor, thought). Under "Local and Global Minima".
Quote: "The proposition has three corollaries. The set of minimizers is convex: if $\mathbf{x}$ and $\mathbf{y}$ both attain"
The three statements are consequences of the chord inequality and of the first-order condition, as their one-line proofs say; none uses the proposition. Call them "three more facts of the same kind". The third ("a small gradient means a small gap") is then applied to the previous section's guarantee without the hypothesis it needs, that $\|\mathbf{x}_k - \mathbf{x}^\star\|$ stays bounded; the text proves this 150 lines later ("the iterates move monotonically closer to every minimizer"). Add a forward clause, or move the remark after the rate proof.

**CV-11** (minor, explanation). Under "Local and Global Minima", after the uniqueness proof.
Quote: "function may jump upward: $f(x) = x + \tfrac12 x^2$ on $(0, 1]$ with"
The uniqueness proposition carries the hypotheses "continuous", "closed", "nonempty interior", and a paragraph with a boundary counterexample, to cover a case (a convex function that jumps at the boundary of its domain) that no loss in this book presents. The proof is 200 words, mostly to avoid assuming differentiability. For this reader, state it for $f$ on $\mathbb{R}^d$ (or continuous on a closed convex set), prove existence from :eqref:`eq_mdl-opt-strong-convexity` in three lines (the quadratic floor makes the sublevel set bounded), and move the general hypotheses and the counterexample to exercise 3, which already studies existence.

**CV-12** (moderate, thought). Under "Convergence Rates", and the closing paragraph.
Quote: "and $(1 - 10^{-3})^k \le 10^{-3}$ requires $k \approx 6{,}900$ steps."
The previous section's signature number is 6,908: the steps gradient descent needs on the running example for six digits of *distance* at the *optimal* step. Here "about 6,900" is the number of steps the *bound* requires for three digits of the *gap* at $\eta = 1/L$, and the closing paragraph repeats it ("or about $6{,}900$ on the running example for $\epsilon = 10^{-3}$"). The two agree by the accident that $\log 10^3 = \tfrac12\log 10^6$. A reader will take them for the same count and conclude that bound and measurement coincide, which is the opposite of what the rate-check passage then shows. Fix: use the previous section's target here ("six digits of the gap: the bound requires about 13,800 steps, the quadratic at $\eta = 1/L$ about 6,900, and at the optimal step 3,454, as :numref:`sec_mdl-gradient-based-optimization` found") so that the three numbers are placed side by side once, and delete the count from the closing paragraph.

**CV-13** (minor, pedagogy). Under "Log-Sum-Exp and the Softmax Covariance", the cell.
Quote: "Does the covariance reading hold numerically? The cell computes the"
It was just proved, so it does. The cell is short and its three prints are harmless, but the question is not a real one. A use of the same lines that finds something out: print the largest eigenvalue for peaked and for uniform logits and compare with the bound $\tfrac12$ of exercise 6(b) (uniform over two classes attains it), which is the smoothness constant of cross-entropy and is used by the next sections. In the writing pass, replace the question by "The cell confirms the three facts at six random logits."

**CV-14** (minor, explanation). Under "The Polyak–Łojasiewicz Condition".
Quote: "$x_0 = 4.5$ with the step $\tfrac1{16}$, half of $1/L$ for $L = 8$, the"
The proposition is for $\eta = 1/L$; the cell uses $1/(2L)$ without a reason, and the text must then introduce a second guarantee, $1 - \tfrac{3\mu}{4L}$, that the reader has to derive from the descent lemma unaided. Either say why ("at $\eta = 1/8$ the local factor $1 - f''(0)\eta$ is $0$ and the last ratios are not informative") and show the one-line derivation of $\tfrac{3\mu}{4L}$, or run the cell at $\eta = 1/L$ (deferred).

**CV-15** (minor, pedagogy). Under "Implicit Bias".
Quote: "which one it reaches: it has an **implicit bias**. The simplest case is"
The subsection is 250 words, states two results and sends the reader to three exercises and another section's exercise. The row-space argument is complete in three sentences and is good. What is missing is the reason a deep learning reader should care: networks with more weights than examples interpolate, so *which* interpolant the optimizer picks decides how the model generalizes (and the per-coordinate steps of Adam change the pick, as exercise 8(c) shows). One sentence saying so would turn a list of pointers into the end of the argument. Without it, this reads as a survey tail of the kind guide §12.2 sends to Further Reading.

**CV-16** (moderate, language). Last paragraph of the body.
Quote: "We can now certify that a loss is convex from its parts, read one gradient"
The body closes with a recap in the "We can now ..." formula that the guide excludes (§12.2), the previous section closes with "You can now predict ...", and a Summary follows in both. The second half of this paragraph ("Every result here assumes the exact gradient and weights free to range over all of $\mathbb{R}^d$ ...") is a real closing: it says what the section did not cover and where that goes. Keep that half; delete the recap.

**CV-17** (minor, thought). Summary.
Quote: "convex because its Hessian is a covariance. Network losses are generically nonconvex, but the PL"
"the PL inequality keeps the linear rate" reads as if network losses satisfied it. The body is careful: PL "can also hold on part of the space", and for wide networks "near interpolating solutions". Write "where a PL inequality holds, the linear rate survives without convexity".

**CV-18** (minor, language; pattern, 6 instances). Question-then-cell lead-ins.
Quotes: "Do the three tests agree on a concrete loss? The cell checks them on the"; "Does the measured contraction obey the bound, and by how much? The cell"; "Does the covariance reading hold numerically? The cell computes the"; "What does gradient descent lose when a function is strictly but not strongly"
Six of the seven cells are introduced by a question followed by "The cell ..." (the exception is the double-well cell). Four of the six are yes-or-no questions whose answer the preceding proof has fixed. The two that work ("What does gradient descent lose ...", "by how much?") ask for a quantity. Keep those; for the rest, state the prediction and what would count as a surprise.

## Exercises

Twelve problems, at the top of the 8 to 12 band, and the set is good: most extend the text, several are small research results made accessible (Nesterov smoothing of the maximum with the optimal temperature; the imbalance $a^2 - b^2$ conserved by gradient flow on $(ab - 1)^2$; implicit bias under momentum, preconditioning and coordinate descent; the max-margin direction of Soudry et al. with a heuristic derivation and a run). Difficulty is graded and marked. I checked the stated identities and constants in exercises 4, 6, 7 and 10; they are correct.

- Exercise 8(c) is the best deep learning connection in the section (a diagonal preconditioner changes which interpolant is reached, hence a different implicit bias for Adam). The text under "Implicit Bias" should mention its conclusion, not only point to it (CV-15).
- Exercise 9 (max-margin, marked extended) is long, with a $10^6$-step run; it is worth keeping as the one project.
- Exercise 11(d) repeats parts (a) and (b) of chapter 9's `ex_optimization-intro-permutation-symmetric-minima`. Drop (d) here or cite the chapter 9 exercise.
- Missing: nothing on Jensen applied to a loss (for example, that the loss of an averaged prediction is at most the average loss, the argument behind ensembling and behind dropout-as-averaging). One short problem of that kind would connect Jensen to the main text more directly than AM-GM does.

## Still open from October

The October review's points that touch this section are resolved: $\kappa$ is now introduced with a sentence that reconciles it with the previous section ("here it is defined for every smooth strongly convex function"), subgradients cite the calculus chapter instead of re-teaching it, and the opening poses a problem. "Real-analysis hygiene" survives in two places (CV-5, CV-11).

## For the chapter review

- The section depends on the previous one for the descent lemma, $L$-smoothness and the stationarity guarantee, and on :numref:`sec_mdl-single_variable_calculus` for Taylor's theorem and one-dimensional subgradients. It supplies Jensen's inequality to three other chapters, the rates and the PL inequality to the stochastic section, strong convexity and subgradients to the constrained and duality sections, and the log-sum-exp Hessian to the numerical-stability section.
- Order within the chapter is right: gradient descent first gives the reader a reason to ask what convexity adds. The handover of the rates needs tidying (GD-10, CV-12): one place for the table, one comparison of bound and quadratic, one use of each count.
- Suspected chapter-wide patterns, seen in both sections I read: a recap paragraph ("You can now", "We can now") before the Summary; code introduced by "The cell ..." after a question whose answer is already known; cells that confirm a theorem on random inputs; unseeded cells, here handled well (the prose reports signs and predictions that do not depend on the draw).
- Material shared with chapter 9 should have one home: the permutation-symmetry argument and exercise (here, with chapter 9 citing it) and the role of convexity as a local model (chapter 9, with this section's "Beyond Convexity" as its proof).
- What a reader might look for and not find: quasi-convexity is rightly absent; conjugates are in the duality section; the one gap is a sentence on why overparameterized networks make the choice among minimizers matter (CV-15).
