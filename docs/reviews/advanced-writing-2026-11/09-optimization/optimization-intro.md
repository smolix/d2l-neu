# Landscapes

Source: `chapter_optimization/optimization-intro.md` (lines 1 to 627, up to the slides marker). About 3,000 words of body prose plus 1,100 words of exercises; 5 pre-generated figures, 4 code cells.

## Verdict

This is the best of the three opening sections of the chapter, and two of its passages meet the bar: the ill-conditioned valley (factor per coordinate, a sweep over learning rates, the floor of 47 steps, then the condition number that explains the floor) and the edge-of-stability experiment on a 25-parameter network, where two runs that differ only in $\eta$ end at sharpness 5 and 8. Both follow the method the earlier reviews asked for: predict a number, run, read the output. The weakness is structural. The section does three jobs: a catalogue of classical hazards (local minima, the cubic, the saddle, the plateau) pitched below a reader who has trained convnets; the full analysis of gradient descent on a quadratic, which is the content the next section (`gd.md`) announces as its own; and a preview of `sgd.md` without an experiment. It then ends three times (convexity, "Three Decisions", Summary). The most important thing to fix: decide who owns gradient descent on a quadratic. I recommend that this section keeps the valley and the edge of stability, that `gd.md` stops re-teaching them, and that the hazards catalogue and the noise preview shrink to make room.

## Scores

| Dimension | Score |
|---|---|
| Clarity of explanation | 4 |
| Clarity of thought and focus | 3 |
| Pedagogy | 4 |
| Language | 4 |
| Ease and enjoyment | 4 |

## What works

- **The valley** ("An Ill-Conditioned Valley"). "The steep coordinate sets a ceiling ... The flat coordinate sets the pace" is the sentence a reader will remember. The first cell is read for what it shows (factor $-0.8$, factor $0.91$, "every factor of ten along $x_1$ takes about 24 steps"; I recomputed 24.4 and $x_1 = -0.295$ after 30 steps). The second cell answers a question the reader has just been made to ask ("Could another learning rate do better?"), and the table's jump from 47 steps at $\eta = 0.476$ to 113 at $0.49$ is a small surprise that the prose explains. Protect all of this.
- **The edge of stability.** A phenomenon from a 2021 paper is reproduced on a network small enough to compute the full Hessian, in a cell of 20 lines, with the ceilings $2/\eta$ drawn in the plot. The stored outputs (PyTorch: 3.86 to 5.00 and 8.02; JAX: 3.89 to 5.00 and 7.99) match the prose. "each as sharp as its own learning rate allows" is exact and quotable. The candour of "Most results quoted in this chapter assume monotone descent on a fixed surface, and the edge of stability lies outside that assumption" is the voice the author asked for.
- **The permutation argument against global convexity** ("The Role of Convexity"): average the $d!$ copies of a minimum and obtain a network with one distinct unit. Short, concrete, and backed by a graded exercise.
- **The coin-flip heuristic is stated as a heuristic** and its independence assumption is handed to an exercise that tests it for $k = 2$.
- The opening paragraph states the three difficulties (flat, uneven, noisy) in plain words before any formula.

## Structural findings

1. **Gradient descent is analysed here before it is taught.** The update rule is introduced in a subordinate clause in the middle of "An Ill-Conditioned Valley", and within forty lines the section has derived the stability ceiling, the optimal learning rate $2/(\lambda_{\min}+\lambda_{\max})$, the factor $(\kappa-1)/(\kappa+1)$ and the step count $(\kappa/2)\log(1/\epsilon)$. The next section then opens "This section asks what decides which learning rates work, and how fast the best of them can go", which this section has just answered, and restarts from $f(x) = x^2$. A reader meets the hard version first and the easy version second. Either move the valley analysis to `gd.md` (and leave here only the picture of uneven curvature), or keep it here and cut `gd.md` down to what is new there (the descent guarantee, the general quadratic, Newton, preconditioning). I recommend the second: the valley is this section's strongest material and the running example of the chapter, and the edge-of-stability experiment needs the ceiling $2/\lambda_{\max}$ to have been derived. See GD-1 in the `gd.md` review.

2. **The hazards catalogue is below the reader and too long for its weight.** "Critical Points and Plateaus" spends about 700 words and four figures on definitions (local and global minimum on $x\cos(\pi x)$, the stationary inflection of $x^3$, the saddle $x^2 - y^2$, $\tanh$ at $x = 4$), and then says that "what limits the speed of routine training is usually something else". The section's own judgement is that this material matters less; its length should say the same. Keep the eigenvalue test, the coin-flip argument with the Dauphin measurement, and the plateau paragraph (which connects to vanishing gradients). Cut the local-minimum figure and the cubic (LAND-4, LAND-5). That frees about 350 words.

3. **"Noisy Gradients" is a preview with no example.** It states the $1/b$ law, the noise ball, the need for decay and the role of batch size in three paragraphs. Each of these is derived and measured in `sgd.md`, and the first two were already taught in :numref:`subsec_linear-regression-minibatch-sgd`. Under guide 7.7 this should be one short paragraph that names the second effect and points forward. The saddle-escape paragraph (Lee et al.) is the only new content and belongs with the saddles (LAND-14).

4. **Three endings.** "The Role of Convexity" is announced as closing the section; "Three Decisions" follows with a second roadmap of the chapter (the chapter introduction already lists each section's question); the Summary follows that, and its last sentence is a third roadmap. Keep one. "Three Decisions" has a real idea (direction, step size over time, treatment of noise) that could be the last paragraph of the body; its second paragraph, which walks through the sections again, duplicates `index.md` and can go (LAND-16).

5. **"The Goal of Optimization" carries the imports cell in the middle of its argument** and a figure that draws what :numref:`sec_generalization_basics` already taught. Two sentences and the reference would do (LAND-2, LAND-3).

6. **Not here and looked for:** the bridge from the diagonal valley to a general quadratic. The section defines $\kappa$ by Hessian eigenvalues but never says that gradient descent on any quadratic splits into independent coordinates along the eigenvectors, which is why the two-coordinate valley is general. It is three sentences, small enough to duplicate from the appendix (LAND-9).

## Passage-level findings

**LAND-1** (major, thought). Under "An Ill-Conditioned Valley".
Anchor: "simplest optimizer, steps against the gradient with a learning rate $\eta$:"
Gradient descent, the subject of the next section, is introduced in passing and then analysed completely. See structural finding 1. Fix: if the analysis stays here, say so in the section opening ("we analyse the simplest optimizer on it now, because the result organizes the chapter") and have `gd.md` cite it instead of restarting.

**LAND-2** (minor, pedagogy). Under "The Goal of Optimization".
Anchor: "A one-dimensional example shows the difference. Let $f$ be a smooth risk"
The imports cells sit between the paragraph that distinguishes risk from empirical risk and the example that illustrates it, 170 lines before the first cell that uses them. The JAX tab also has a second imports cell (`functools`, `random`), against the one-imports-cell rule, and `random` is not used. Move the imports to just before the first valley cell and merge the JAX cells.

**LAND-3** (moderate, thought). Under "The Goal of Optimization".
Anchor: "No optimizer can close this gap, because it comes from sampling the data,"
True of the gap between the two functions, but the sentence that follows ("we therefore set generalization aside") is contradicted by the section's own best result: the two edge-of-stability runs "end in regions of different curvature", and the local-minima paragraph says noise selects basins. The optimizer does choose among minimizers of the empirical risk, and that choice affects the risk. Fix: say that the gap between $f$ and $g$ is not the optimizer's doing, that which minimizer of $g$ is reached is, and that the chapter measures the training objective and returns to this choice only in passing. The figure `mdl-opt-risk-gap` is a drawn toy that repeats :numref:`sec_generalization_basics`; cut it.

**LAND-4** (moderate, pedagogy). Under "Local Minima".
Anchor: "has one of each (:numref:`fig_mdl-opt-local-minima`)."
A reader who has trained ResNets does not need a figure to learn what a local minimum is. The subsection's useful content is one sentence (the gradient cannot tell a local minimum from the global one) and one forward pointer. Fold both into the opening of "Critical Points and Plateaus" and drop the function and the figure. Exercise 1 already uses $x\cos(\pi x)$ and can define it.

**LAND-5** (minor, explanation). Under "Saddle Points".
Anchor: "*stationary inflection*. A true saddle, which curves up in one direction and"
The subsection defines a saddle, illustrates it with a non-example ($x^3$), corrects itself ("Strictly speaking"), declares 'we use "saddle" for both', and four paragraphs later calls the indefinite case "strict". The cubic costs a figure and a terminological detour and is needed only by Exercise 1. Start with $x^2 - y^2$; mention the degenerate case in the fourth bullet of the eigenvalue test, where it belongs.

**LAND-6** (minor, correctness; unverified in detail). Under "Saddle Points".
Anchor: "heuristic where it matters: in high dimensions the critical points a method"
The evidence cited (Dauphin et al. 2014) is, as far as I recall, critical points located by a Newton-type search on small MLPs, plus random-matrix arguments. "the critical points a method meets on its way down are overwhelmingly saddles" is stronger than that: a gradient method does not meet critical points, it passes regions of small gradient. I believe the qualitative claim (index falls with loss) is right. Soften to "the critical points found at high loss are saddles with many descending directions; those found at low loss have few or none".

**LAND-7** (minor, thought). Under "Plateaus".
Anchor: "and at low loss most critical points are good. And what limits the speed"
Circular: a critical point at low loss is good by definition. What the Dauphin result supports is "critical points with high loss are saddles, which a noisy method leaves, so the minima a run can settle in have low loss".

**LAND-8** (minor, correctness). Under "An Ill-Conditioned Valley".
Anchor: "this is $10 \log 100 \approx 46$, the count we measured."
The table prints 47, and the text says 47 three times. $10 \ln 100 = 46.05$ and the exact count is $\lceil \ln 100 / \ln(21/19) \rceil = 47$. Write "about 46, against the 47 we measured". Also say once that $\log$ is the natural logarithm.

**LAND-9** (moderate, explanation). Under "An Ill-Conditioned Valley".
Anchor: "eigenvalues of the Hessian. The valley's Hessian is diagonal, so its"
The definition of $\kappa$ jumps from two coordinates to Hessian eigenvalues without the step that connects them. Draft: "Nothing depended on the valley being aligned with the axes. For any quadratic $\frac12 \mathbf{x}^\top \mathbf{H}\mathbf{x}$ with positive definite $\mathbf{H}$, write $\mathbf{x}$ in the basis of eigenvectors of $\mathbf{H}$: each coordinate is then multiplied by $1 - \eta\lambda_i$ per step, exactly as $x_1$ and $x_2$ were. The largest eigenvalue sets the ceiling and the smallest sets the pace." Then the formulas that follow are earned. Only one of this section and `gd.md` should carry it (GD-7).

**LAND-10** (moderate, correctness). Under "An Ill-Conditioned Valley".
Anchor: "gradient descent grows linearly with $\kappa$, and the $\kappa$ of a deep"
For a network the quantity just defined does not exist in a useful sense: the same sentence says the Hessian has "a bulk of eigenvalues near zero", and the previous subsection said it has negative ones, so $\lambda_{\max}/\lambda_{\min}$ is infinite or negative. The text slides to "its largest curvature exceeds the typical one by factors of hundreds or more" without saying that it has changed the definition. Say it: "A network's Hessian has eigenvalues near zero and below it, so $\kappa$ as defined is not finite. The useful comparison is between the few large eigenvalues and the bulk ..."

**LAND-11** (moderate, correctness of a reading). Under "The Edge of Stability".
Anchor: "Meanwhile the loss keeps falling by orders of magnitude, though not on"
The cell prints the loss at the start and at the end and the sharpness every 500 steps. Nothing in the output shows that the loss rises on single steps; the claim is stated twice (here and in the paragraph that introduces Cohen et al.) and is the defining property of the regime. The appendix's NumPy rerun of the same experiment (:numref:`subsec_mdl-edge-of-stability`) does print the fraction of late steps on which the loss rose. Prose fix now: attribute the non-monotone loss to the paper, not to "our" run. Deferred experiment fix: record the loss per step and print that fraction here, where the experiment is taught, and drop the duplicate run from the appendix.

**LAND-12** (moderate, correctness; unverified, medium confidence). Under "The Edge of Stability".
Anchor: "warmup keeps $\eta$ small until the sharpness has fallen"
As I understand Kalra and Barkeshli (2024) and Gilmer et al. (2021), the mechanism is not that sharpness falls by itself while warmup waits. The rising learning rate pushes the sharpness down: at each stage the run sits near $2/\eta_t$ and the self-stabilization described in the previous paragraph lowers the sharpness as $\eta_t$ grows, so that the target rate becomes stable. That reading also follows directly from what the reader has just seen. I could not check the paper. If the reading is right, rewrite as "warmup raises $\eta$ gradually, and at each stage the same self-correction lowers the sharpness to about $2/\eta$, so the run arrives at the target rate on a surface flat enough for it."

**LAND-13** (minor, language). Under "The Edge of Stability".
Anchor: "oscillation into a drift toward lower sharpness. The quadratic analysis"
Four consecutive sentences say that the quadratic model omits the third-order term ("predicts the oscillation but not what follows", "which the quadratic model drops", "The quadratic analysis misses this change", "lies outside that assumption"). Two are enough.

**LAND-14** (moderate, thought). Under "Noisy Gradients".
Anchor: "a *noise ball*, whose size grows with $\eta$. Convergence therefore needs a"
This subsection asserts, without example or measurement, what `sgd.md` derives and measures two sections later and what Basics already taught. Reduce to one paragraph: the gradient is an estimate from $b$ examples, its error does not vanish at the minimum, so a constant learning rate stalls at a distance that :numref:`sec_sgd` computes. Move the saddle-escape paragraph (Lee et al.) to "Saddle Points", where the reader is asking whether saddles trap gradient descent.

**LAND-15** (minor, language). Under "Local Minima" and "Noisy Gradients".
Anchor: "Noise is not only a nuisance. It can push the iterate off a saddle point or"
The same remark was made 300 lines earlier ("why minibatch noise is not purely a nuisance"). State it once, where the saddle-escape result is given.

**LAND-16** (moderate, thought). Under "Three Decisions".
Anchor: "What we have seen suggests that any optimizer can be described by three"
A second closing and a second chapter roadmap. The three decisions are a useful frame; the paragraph beginning "Each method in this chapter is a particular set of these decisions" re-lists the sections that `index.md` has listed. Keep the first paragraph as the end of the body, cut the second, and remove the roadmap sentence from the end of the Summary.

**LAND-17** (moderate, correctness). Under "Summary".
Anchor: "constant learning rate from converging. Real networks also adapt their"
"Real networks ... train at the edge of stability" generalizes from full-batch gradient descent on a 25-parameter network (and from a paper about full-batch gradient descent) to training in general. With minibatches, momentum or Adam the threshold and the behaviour differ, and the body says only that the dependence "can run the other way". Write "Under full-batch gradient descent, a network's sharpness rises to about $2/\eta$ and stays there".

**LAND-18** (minor, language; pattern across the three sections). The largest curvature has four names in the first three sections: $h_2$ (here and in `gd.md`), $\lambda_{\max}$ (here), "the sharpness" (here), and $L$ (`gd.md`, `sgd.md`), and the curvature along a coordinate is $h_j$ here and $\lambda$ in `sgd.md`. Pick $\lambda$ for curvatures and eigenvalues throughout, keep "sharpness" as the name of $\lambda_{\max}$ for a network, and say once that $L$ in the smoothness bound is the largest curvature.

## Code and experiments

- `optimization-intro-an-ill-conditioned-valley` and `-1`: both teach; outputs match the prose in both tabs (49 steps at 0.45, 47 at 0.476, 113 at 0.49, never at 0.5).
- `optimization-intro-the-edge-of-stability-2`: outputs match. The clause "though not on every step" is not shown by any output (LAND-11). "usually below 5" and "In almost every run" are ranges over unseeded starts and are fine under the author's decision 5.
- The same experiment is run a second time in the appendix in NumPy (`mdl-gradient-based-optimization-the-edge-of-stability-1`). One run should own it; the main text is the natural owner (see "For the chapter review").
- No cell here merely confirms a derived result. The sweep could in principle be called a confirmation of $2/(\lambda_{\min}+\lambda_{\max})$, but it comes before the formula and motivates it, which is the right order.

## Exercises

Six problems, graded, all tied to the text. Exercise 4 (the scalar network $ab$ whose minimizers have sharpness $a^2 + b^2$, run at four learning rates) is the best in the three sections: it reproduces the selection of flat minima by the learning rate in two parameters. Exercise 2 (escape time from the saddle, with and without noise) repairs the missing example of LAND-14. Exercise 6 (Wigner) is at appendix level; keep it, it is marked hard and its part 3 has a clean answer (I get $(2-\sqrt2)/4 \approx 0.146$ against $0.25$). Exercise 3 part 2 asks the reader to derive what the body quotes and then to "compare your derivation with the appendix's"; drop the comparison clause, since the main text must not need the appendix. No exercise needs cutting.

## Still open from October

The October item "the first section calls helpers that the next section defines" is fixed (the section now defines its own `gd_valley`). "The test valley three times" is still open: the valley is fully introduced here, re-introduced in `gd.md` and again in `sgd.md` and `momentum.md`, each time with its curvatures and the 47 steps. The $1/b$ law is still stated here, derived in `sgd.md`, and taught in Basics.

## For the chapter review

- This section and `gd.md` overlap on gradient descent on a quadratic; one must own it (structural finding 1, GD-1). The number 47 appears three times here, three times in `gd.md` and again in the opening of `momentum.md`.
- Against the appendix: the stability ceiling, optimal step and $(\kappa-1)/(\kappa+1)$ are small results stated here and proved in :numref:`subsec_mdl-quadratic-model`; that is the duplication rule working as intended, and the references (`subsec_mdl-quadratic-model`, `eq_mdl-opt-optimal-step`, `sec_mdl-convexity`) point to passages that hold the cited content. The edge-of-stability experiment is run in both places, which is duplication of an experiment, not of a small result.
- Assumes from Basics: empirical risk and risk, vanishing gradients, activation functions. All references resolve. It does not need the appendix to be read.
- Sets up: the valley and the condition number for every later section; the ceiling $2/\lambda_{\max}$ for warmup in `lr-scheduler.md`.
- Suspected chapter pattern: each section opening re-tells the valley and its numbers instead of citing them, and the classical d2l material (hazards catalogue here, one-dimensional traces in `gd.md`) survives at its old length beside new material of a higher level.
