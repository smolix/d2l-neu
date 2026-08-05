# Exercise Catalog: chapter_optimization

Chapter overview (12 files, 69 existing exercises surveyed against verified external
sources — every URL below was directly fetched, not inferred):

EPFL's CS-439 "Optimization for Machine Learning" (Flammarion & Jaggi), checked
directly across all 11 Spring-2025 problem sets, is convex-theory-heavy
(convexity, GD/projected-GD, subgradient, Newton/quasi-Newton, Frank-Wolfe,
coordinate descent, lower bounds) and matched only gd.md well. Simon Prince's
*Understanding Deep Learning* (official Answer Booklet, verified) was the best
match for the chapter's foundational sections, with direct problem-for-problem
overlaps in optimization-intro.md and momentum.md. CS231n Assignment 2
(verified via three independent starter-code mirrors spanning academic years)
remains the standing source for implementing momentum/RMSProp/Adam from
scratch. Stanford CS336 ("Language Modeling from Scratch," verified via its
own PDFs and GitHub repos) was the single best-matched course for the second
half of the chapter — its Assignment 1 supplies AdamW, gradient clipping, and
a cosine-with-warmup schedule almost verbatim, and its Assignment 3 supplies a
Chinchilla-style scaling-law fit. Microsoft's `mup` repo/paper grounded
scaling.md's coordinate check. Two sections have essentially no external
homework tradition, and that absence is itself the finding: minibatch-sgd.md's
vectorization/cache material belongs to systems courses, not ML-optimization
courses; muon.md — checked directly against Keller Jordan's original post and
against a live 2025 optimization course's full syllabus — has not yet entered
academic coursework anywhere. Most files (adam, adamw, muon, batch-size,
scaling, practice, and the strong halves of momentum/sgd/minibatch-sgd) were
already excellent per the prior style review and are kept essentially intact;
external material is layered on top, never substituted in.

---

## chapter_optimization/optimization-intro.md — Landscapes

**Topic:** What optimization does and doesn't accomplish for deep learning: vanishing gradients, curvature/condition number, noise, and convexity as local approximation and vocabulary.
**Current exercises:** 5; disposition: keep 3, rewrite 1, drop 1 — the two problematic items are exactly the file's two "Can you...?"/pure-brainstorm items (ex3's sub-question, ex5); the analytical items (permutation-symmetry, Wigner symmetry, valley/condition-number) are strong and kept.

**External sources found:**
- Simon J. D. Prince, *Understanding Deep Learning* (MIT Press, 2023), official Answer Booklet, Problem 6.2 — prove a linear-regression least-squares loss is convex by showing the Hessian's trace and determinant are both positive — https://github.com/udlbook/udlbook (`UDL_Answer_Booklet_Students.pdf`)
- Prince, *Understanding Deep Learning*, Problem 6.6 — classify seven labeled points on a given nonconvex surface as local minimum, global minimum, or neither — same source
- Prince, *Understanding Deep Learning*, Problem 6.8 — can non-stochastic gradient descent with a fixed learning rate escape a local minimum, and why — same source
- Prince, *Understanding Deep Learning*, Problem 6.11 — state the Hessian's dimensions for a one-million-parameter model, motivating why full second-order methods are infeasible — same source
- No external homework tradition was found for the specific symmetric-random-matrix/Wigner-semicircle argument for saddle-point prevalence, or for the permutation-symmetry argument for combinatorial minima multiplicity — these appear to be this book's own framing devices rather than standard course exercises; note explicitly as a finding.

**Proposed problem set** (6 problems):
1. [conceptual] **Permutation-symmetric minima.** Consider an MLP with one hidden layer of $d$ units and one output. Show that every local minimum belongs to a family of at least $d!$ distinct parameter settings computing the identical function, by exhibiting the hidden-unit permutation that generates them, and give one explicit example for $d=3$.
   *Provenance: original (retained, ex1).*
1. [conceptual] **Symmetric random matrices and saddles.** Let $\mathbf{M}$ be symmetric with i.i.d., sign-symmetric entries $M_{ij}=M_{ji}$, $p_{ij}(x)=p_{ij}(-x)$.
    1. Prove the eigenvalue distribution is sign-symmetric: $P(\lambda>0)=P(\lambda<0)$ for any eigenvector.
    1. Explain why this does not imply $P(\lambda>0)=0.5$.
   *Provenance: original (retained, ex2).*
1. [short-code] **Naming the saddle-balancing act.** Explain in one paragraph why balancing a ball on a saddle is unstable in some directions and stable in others. Then implement $f(x_1,x_2)=x_1^2-x_2^2$ and show numerically that gradient descent started exactly on the ridge ($x_2=0$) needs $10^6\times$ more iterations to escape a $10^{-6}$ perturbation in $x_2$ than a $10^{-2}$ perturbation, reporting both iteration counts.
   *Provenance: original (rewrite of ex3, removing the "Can you exploit this?" filler question and adding a measured deliverable).*
1. [short-code] **Rescaling an ill-conditioned valley.** For $f(\mathbf{x})=0.1x_1^2+2x_2^2$:
    1. Find the largest learning rate for which gradient descent still converges; verify with `d2l.train_2d`.
    1. At $\eta=0.45$, compute the per-step shrinkage of $|x_1|$, $|x_2|$ and the steps needed to shrink $|x_1|$ by $100\times$; check numerically.
    1. For $f(\mathbf{x})=\frac{\lambda_{\min}}{2}x_1^2+\frac{\lambda_{\max}}{2}x_2^2$ at the best stable rate, show step count grows linearly in $\kappa$.
    1. Find the rescaling $\tilde x_1=\alpha x_1$ that perfectly conditions the valley, and name the later section that estimates such a rescaling from gradients alone.
   *Provenance: original (retained, ex4).*
1. [conceptual] **Classifying a nonconvex surface.** Given a plotted 2-D nonconvex surface with several marked points, classify each as a local minimum, the global minimum, or neither, justifying each with the local gradient and curvature.
   *Provenance: adapted from Prince, Problem 6.6 (overlap medium — same task type, different surface).*
1. [short-code] **Why Newton doesn't scale.** State the fp32 memory required to store a dense Hessian at $p\in\{10^3,10^6,10^9\}$ parameters, then confirm the $p=10^6$ figure by instantiating (not inverting) a Hessian-shaped tensor of that size and measuring its footprint.
   *Provenance: adapted from Prince, Problem 6.11 (overlap high).*

---

## chapter_optimization/gd.md — Gradient Descent

**Topic:** 1-D and multivariate gradient descent, Newton's method, and diagonal preconditioning.
**Current exercises:** 5; disposition: keep 2, rewrite 3 — ex1 (no metric), ex2 (line-search/binary-search terminology conflation), and ex5 (ambiguous "the algorithm above") are the file's three clarity flags; each is salvageable with a concrete deliverable rather than a drop.

**External sources found:**
- EPFL CS-439 "Optimization for Machine Learning" (Nicolas Flammarion, with Martin Jaggi), Problem Set 7 "Newton," Spring 2025 — assigns Newton's-method and quasi-Newton exercises, plus a coding task adapting a fixed-point-iteration solver to Newton's update $x_{t+1}=x_t-f'(x_t)/f''(x_t)$ — https://github.com/epfml/OptML_course/raw/master/labs/ex07/exercise07.pdf
- Stephen Boyd & Lieven Vandenberghe, *Convex Optimization* (2004), Exercises 9.30–9.31, per the companion "Additional Exercises" notes (used at Stanford EE364a/UCLA EE236b/MIT 6.975) — 9.30 has students implement and compare the gradient method, steepest descent, and Newton's method with backtracking line search on a random unconstrained problem; 9.31 compares Newton's method recomputing the Hessian factorization every step against reusing a stale factorization for $N=1,15,30$ steps, plotted against flop count rather than iteration count — http://web.mit.edu/~jadbabai/www/EE605/additional_exercises.pdf
- Checked but not usable: Nocedal & Wright, *Numerical Optimization* — confirmed the book's line-search and Newton/quasi-Newton chapters exist, but no specific numbered exercise could be verified from the available text within this session; omitted rather than guessed.

**Proposed problem set** (6 problems):
1. [short-code] **Learning-rate sweep with a stopping metric.** For $f(x)=x^2$ and $f(x)=|x|^{1.5}$, run gradient descent from $x_0=10$ at $\eta\in\{0.01,0.1,0.5,0.9,1.1\}$ for 100 steps. Report, per pair, steps to $|x|<10^{-3}$ or "diverged," as a 2$\times$5 table.
   *Provenance: original (rewrite of ex1, replacing "experiment... and observe" with a scored grid).*
1. [short-code] **Line search vs. binary search.** Implement bisection minimization of a convex $f$ over $[a,b]$ using only the sign of $f'$ to choose $[a,(a+b)/2]$ or $[(a+b)/2,b]$.
    1. State why this needs the derivative's sign, unlike bisecting on $f$ values directly.
    1. Derive the interval-width convergence rate.
    1. Apply it to minimize $\log(\exp(x)+\exp(-2x-3))$ to $10^{-6}$ accuracy and report the iteration count.
   *Provenance: original (rewrite of ex2, disambiguating "line search" from "binary search").*
1. [conceptual] **An objective gradient descent hates.** Design $f:\mathbb{R}^2\to\mathbb{R}$ on which gradient descent is exceedingly slow by scaling the coordinates differently, and report your example's Hessian condition number.
   *Provenance: original (retained, ex3).*
1. [short-code] **A lightweight diagonal Newton.** Precondition gradient descent with (a) $\mathrm{diag}(H)^{-1}$ and (b) $|\mathrm{diag}(H)|^{-1}$; apply both to the previous problem's objective and compare iteration counts against plain gradient descent.
   *Provenance: original (retained, ex4).*
1. [short-code] **Stale Hessians and the flop budget.** Implement Newton's method on a random unconstrained least-squares-with-smoothing problem, comparing three variants: refactoring the Hessian every step, every 5 steps, and every 20 steps. Plot loss against estimated flop count ($n^3/3$ per factorization, $2n^2$ per solve), not iteration count.
   *Provenance: adapted from Boyd & Vandenberghe, Exercise 9.31 (overlap medium — cite on adoption).*
1. [conceptual] **Which algorithm, and does rotation matter?** Take problem 4's preconditioned Newton method, apply it to a convex and a nonconvex 2-D objective, then rotate each by 45° and reapply. State which variant (diagonal or absolute-diagonal) is rotation-invariant and why, using the definition of $\mathrm{diag}(H)$.
   *Provenance: original (rewrite of ex5, naming the specific algorithm and the invariance question the ambiguity was gesturing at).*

---

## chapter_optimization/sgd.md — Stochastic Gradient Descent

**Topic:** Stochastic gradient estimates, dynamic learning rates, and the variance/batch-size relationship.
**Current exercises:** 6; disposition: keep 4, rewrite 2 — this is a strong, largely well-specified set; only ex4 (open-ended "how would you change the solver") and ex5 ("Can you change $f$...") needed concrete deliverables.

**External sources found:**
- Simon J. D. Prince, *Understanding Deep Learning*, Problem 6.9 — given 1,000 SGD iterations on a dataset of size 100 with batch size 20, compute the number of epochs — https://github.com/udlbook/udlbook (`UDL_Answer_Booklet_Students.pdf`)
- No further external tradition found: Robbins–Monro-style convergence theory is covered at a theorem level in Boyd/Nocedal, but hands-on "measure the noise ball" exercises like this section's own are not a standard course assignment — they appear original to this book's empirical style, and are worth keeping as-is.

**Proposed problem set** (6 problems):
1. [short-code] **Schedule race to the optimum.** Run SGD on this section's toy quadratic under constant, $1/\sqrt{t}$, and $1/t$ rates for a fixed iteration budget, plotting distance to $(0,0)$ vs. iteration for all three. State which wins at this budget and whether the ranking would flip at 10$\times$ the budget.
   *Provenance: original (retained, ex1).*
1. [conceptual] **Noise as a random design matrix.** Prove that for $f(x_1,x_2)=x_1^2+2x_2^2$, adding normal noise to the gradient is equivalent to minimizing $f(\mathbf{x},\mathbf{w})=(x_1-w_1)^2+2(x_2-w_2)^2$ with $\mathbf{w}$ normally distributed.
   *Provenance: original (retained, ex2).*
1. [short-code] **With vs. without replacement.** Compare SGD's convergence sampling with replacement against one shuffled pass per epoch, for a fixed number of epochs. Report the final loss gap and relate it to the $1-e^{-1}$ coverage argument from this section.
   *Provenance: original (retained, ex3).*
1. [short-code] **A coordinate with a runaway gradient.** Construct a 2-D quadratic where one coordinate's gradient is consistently $100\times$ larger. Report plain SGD's largest stable rate, then the diagonal-preconditioned (:numref:`sec_gd`) rate and its speedup on the slow coordinate.
   *Provenance: original (rewrite of ex4, replacing "how would you change the solver" with a built-and-measured fix).*
1. [conceptual] **An objective that hides no minima.** Count the local minima of $f(x)=x^2(1+\sin x)$ on $[-10,10]$ and explain why the count grows unboundedly as the interval widens. Construct a related 1-D function on which any derivative-free search must inspect every local minimum before certifying the global one, and state what forces this.
   *Provenance: original (rewrite of ex5, removing the "Can you" filler and naming the deliverable).*
1. [short-code] **Variance after training.** Repeat this section's gradient-variance-vs-batch-size measurement at the parameters reached after training with any chapter optimizer. State whether the $1/b$ slope still holds and whether the noise level (not just slope) changed, with an explanation.
   *Provenance: original (retained, ex6).*

---

## chapter_optimization/minibatch-sgd.md — Minibatches

**Topic:** Vectorization, cache effects, and minibatch gradients as the computational half of the SGD/batch-size story.
**Current exercises:** 5; disposition: keep 3, rewrite 2 — ex1 (no metric) and ex3 ("what happens?") needed quantified deliverables; ex2, ex4, and ex5 are already precise.

**External sources found:**
- No strong academic-homework tradition was found for the vectorization/cache/arithmetic-intensity framing itself — this is systems/HPC-course content, not typically an ML-optimization-course exercise. The closest analogue is Stanford CS336 "Language Modeling from Scratch," Assignment 2 (systems) — its benchmarking harness explicitly warns that CUDA calls are asynchronous and that skipping a warm-up iteration biases the first timed measurement — https://github.com/stanford-cs336/spring2024-assignment2-systems (overlap low: different measurement target, same methodological point)
- Note explicitly: this section's own exercises are already close to the strongest treatment found anywhere for this specific topic; the gap is structural (wrong discipline of course), not a quality gap.

**Proposed problem set** (6 problems):
1. [short-code] **Batch size and learning rate, jointly.** Sweep $b\in\{1,10,100,1000\}$ with learning rate scaled by $\sqrt{b}$ relative to a $b{=}10,\eta{=}0.01$ baseline. Report time to loss 0.3 in both epochs and wall-clock seconds for each of the 4 runs, and state which unit tells the more useful story.
   *Provenance: original (rewrite of ex1, adding a scaling rule and a concrete target).*
1. [short-code] **Where block width saturates.** Vary the blocked-matmul benchmark's block width over $\{1,4,16,64,256\}$, time each, and identify where throughput saturates and why it saturates before width 256. Repeat the element/column/full comparison at $4096\times4096$ on a GPU.
   *Provenance: original (retained, ex2).*
1. [short-code] **With-replacement minibatches.** Compare minibatch SGD against sampling each minibatch with replacement from the full training set, for a fixed number of gradient steps. Report the final loss gap and the fraction of examples never seen by the with-replacement variant, explained via the $1-e^{-1}$ coverage argument.
   *Provenance: original (rewrite of ex3, replacing "what happens?" with a quantified comparison).*
1. [conceptual] **Duplicated data, three ways.** A silently duplicated dataset (every example appears twice). Explain, separately for SGD, minibatch SGD, and full-batch GD, whether each method's step count or per-step cost changes, and why.
   *Provenance: original (retained, ex4).*
1. [short-code] **Gradient accumulation, verified.** Implement gradient accumulation on `train_ch11`: sum gradients over $k$ minibatches of size $b$, update once with their average. Verify the loss trajectory matches a direct batch-$kb$ run against examples processed, then compare wall-clock time.
   *Provenance: original (retained, ex5).*
1. [short-code] **A warm-up-free benchmark.** Repeat problem 2's block-width benchmark recording the *first* timed call at each width with no warm-up iteration. Report how much the first call overstates steady-state time, and explain the mechanism.
   *Provenance: inspired by Stanford CS336, Assignment 2 (systems) (overlap low — different benchmark target, same warm-up caveat).*

---

## chapter_optimization/momentum.md — Momentum

**Topic:** Heavy-ball momentum as a leaky gradient average, and Nesterov's look-ahead correction.
**Current exercises:** 5; disposition: keep 3, rewrite 2 — ex1 (no metric) and ex5 (trailing "experiment with the parameters") are the only soft items; ex2, ex3, ex4 are precise and kept.

**External sources found:**
- Simon J. D. Prince, *Understanding Deep Learning*, Problem 6.7 — explains gradient descent's right-angle valley oscillation and asks for a fix; the official answer names both Newton's method and a momentum term — https://github.com/udlbook/udlbook (`UDL_Answer_Booklet_Students.pdf`)
- Prince, *Understanding Deep Learning*, Problem 6.10 — show the momentum term $m_t$ is an infinite weighted sum of past gradients and derive the weighting coefficients — a direct match to this section's "leaky average" framing — same source
- Stanford CS231n, Assignment 2 (`cs231n/optim.py`, function `sgd_momentum`) — students implement the heavy-ball update from its docstring and compare it with plain SGD on a deep network — verified via the assignment page and three independent starter-code mirrors with identical function signatures across academic years — https://cs231n.github.io/assignments2024/assignment2/

**Proposed problem set** (6 problems):
1. [short-code] **Momentum hyperparameters, scanned.** On $f(\mathbf{x})=0.1x_1^2+2x_2^2$, run a $4\times4$ grid of $(\beta,\eta)\in\{0,0.5,0.9,0.99\}\times\{0.1,0.2,0.4,0.8\}$ and record iterations to $\|\mathbf{x}_t\|\le10^{-3}$ (or "diverged") as a heatmap.
   *Provenance: original (rewrite of ex1, replacing "observe and analyze" with a scanned grid).*
1. [short-code] **Momentum across many eigenvalues.** For $f(\mathbf{x})=\frac{1}{2}\sum_i\lambda_ix_i^2$, $\lambda_i=2^{-i}$, run GD and momentum from $x_i=1$ and plot each coordinate's decrease under both.
   *Provenance: original (retained, ex2).*
1. [conceptual] **Two parametrizations, one trajectory.** PyTorch's `nesterov=True` computes $\mathbf{v}_t=\beta\mathbf{v}_{t-1}+\mathbf{g}_t$, $\mathbf{x}_t=\mathbf{x}_{t-1}-\eta(\mathbf{g}_t+\beta\mathbf{v}_t)$. By a change of variables, show this matches :eqref:`eq_nesterov`'s iterates and identify which point the framework's velocity corresponds to.
   *Provenance: original (retained, ex3).*
1. [short-code] **Finding $\beta^\star$.** For $f(x)=\frac{\lambda}{2}x^2$ at fixed $\eta$, sweep $\beta\in[0,1)$, measure iterations to $|x_t|\le10^{-6}|x_0|$, locate $\beta^\star$, and compare against the theoretical value.
   *Provenance: original (retained, ex4).*
1. [short-code] **Momentum meets minibatch noise.** Combine momentum with minibatch SGD at $b\in\{8,32,128,512\}$, $\eta$ fixed. Plot final loss vs. batch size and mark the batch size below which momentum's variance-smoothing benefit visibly degrades.
   *Provenance: original (rewrite of ex5, replacing "experiment with the parameters" with a specific sweep and a named failure point).*
1. [conceptual] **Deriving the leaky average.** Show that $\mathbf{v}_t=\beta\mathbf{v}_{t-1}+\nabla f(\mathbf{x}_{t-1})$ unrolls to $\sum_{i=0}^{t-1}\beta^i\nabla f(\mathbf{x}_{t-1-i})$, and derive the averaging window $\sum_i\beta^i=\frac{1}{1-\beta}$ this section's summary asserts without proof.
   *Provenance: adapted from Prince, Problem 6.10 (overlap high — same derivation, different notation).*

---

## chapter_optimization/adam.md — Adam

**Topic:** AdaGrad → RMSProp → Adam, bias correction, and Adam's failure modes (AMSGrad, Yogi, Adadelta).
**Current exercises:** 9; disposition: keep 8, rewrite 1 — this is the chapter's longest and, apart from ex1, most rigorously specified set; kept almost entirely, two related items merged into one to fit the format cap.

**External sources found:**
- Stanford CS231n, Assignment 2 (`cs231n/optim.py`) — students implement `adam` and `rmsprop` from their update-rule docstrings and compare optimizers on a deep network; a verified public solutions notebook's "Inline Question 2" asks why AdaGrad's monotonically growing accumulator shrinks updates to nothing, and whether Adam has the same problem — closely matching this section's own $\epsilon$/AMSGrad/Yogi discussion — https://cs231n.github.io/assignments2024/assignment2/
- No course exercise was found assigning AMSGrad, Yogi, or Adadelta as separate implementation problems the way this section already does; these appear to be this book's own extension beyond the standard "implement the big three" assignment.

**Proposed problem set** (8 problems):
1. [short-code] **Tuning Adam by hand.** Sweep $\eta\in\{10^{-4},10^{-3},10^{-2},10^{-1}\}$ for from-scratch Adam on the airfoil data, holding other hyperparameters fixed. Report final loss and steps to within 10% of the best loss found, per $\eta$.
   *Provenance: original (rewrite of ex1, replacing "observe and analyze" with a scored grid).*
1. [conceptual] **Bias correction by a different init.** Rewrite the moment updates of :eqref:`eq_adam-moments` so no explicit bias correction is needed, by initializing with the first gradient rather than zero. What is lost?
   *Provenance: original (retained, ex2).*
1. [short-code] **The forgetting-rate sweep.** Rerun tuned Adam on `TinyLM` with $\beta_2\in\{0.9,0.99,0.999,0.9999\}$. Which direction hurts more, and how does it relate to the window $1/(1-\beta_2)$?
   *Provenance: original (retained, ex3).*
1. [short-code] **The $\epsilon$ ceiling.** Sweep $\epsilon\in\{10^{-8},10^{-6},10^{-4},10^{-2},1\}$ for Adam on `TinyLM` at fixed $\eta$; explain the trend at both ends via the $\eta/\epsilon$ step ceiling.
   *Provenance: original (retained, ex4).*
1. [short-code] **Two fixes for Adam's failure mode.**
    1. Implement Yogi's capped-shrinkage update; compare with Adam on the airfoil data and construct a gradient stream where Adam diverges but Yogi converges.
    1. Implement AMSGrad's running maximum $\hat{\mathbf{v}}_t^{\max}$ in place of $\hat{\mathbf{v}}_t$; verify it matches Adam on the airfoil data, then test it on the non-convergence construction of :numref:`subsec_mdl-per-coordinate`.
   *Provenance: original (retained, merging ex5+ex6).*
1. [short-code] **Adadelta from scratch.** Implement Adadelta (learning rate replaced by the square root of a second exponential average of squared updates). Where does the first update's scale come from, absent an explicit learning rate?
   *Provenance: original (retained, ex7).*
1. [conceptual] **Rotating the per-coordinate advantage away.** Rotate the toy problem 45° to $f(\mathbf{x})=0.1(x_1+x_2)^2+2(x_1-x_2)^2$ and rerun `adagrad_2d`/`rmsprop_2d`. How much of the advantage over GD survives?
   *Provenance: original (retained, ex8).*
1. [extended] **Frequent vs. rare tokens.** Following :citet:`Kunstner.Yadav.Milligan.ea.2024`, log `TinyLM`'s loss separately for frequent and rare characters under tuned SGD and tuned Adam. Which optimizer progresses on the rare half, and by how much?
   *Provenance: original (retained, ex9; already adapted from Kunstner et al. in-book).*

---

## chapter_optimization/adamw.md — AdamW

**Topic:** Decoupled weight decay vs. $\ell_2$ regularization under Adam's preconditioner.
**Current exercises:** 6; disposition: keep 6, rewrite 0, drop 0 — the most rigorous file in the chapter per the prior style review (zero clarity flags); kept in full, with one genuinely new addition rather than a replacement.

**External sources found:**
- Stanford CS336 "Language Modeling from Scratch" (Percy Liang & Tatsunori Hashimoto), Assignment 1 (basics), Spring 2024, Problem "adamw" (2 pts) — implement AdamW as a `torch.optim.Optimizer` subclass following Algorithm 2 of Loshchilov & Hutter (2019) — https://github.com/stanford-cs336/spring2024-assignment1-basics (`cs336_spring2024_assignment1_basics.pdf`)
- Same assignment, Problem "adamwAccounting" (2 pts) — compute AdamW's peak memory and per-step FLOPs, decomposed by component — closely matches this section's own "Memory Required for Optimizer State" material — same source

**Proposed problem set** (7 problems):
1. [short-code] **Coupled vs. decoupled, isolated.** On a two-parameter toy with pure-noise gradients $g_i\sim\mathcal{N}(0,\sigma_i^2)$, $\sigma=(10,0.1)$, track $|x_i|$ under Adam-with-$\ell_2$ vs. AdamW at matched $(\eta,\lambda)$; verify AdamW's trajectory against $(1-\eta\lambda)^t$.
   *Provenance: original (retained, ex1).*
1. [conceptual] **Does momentum break the equivalence?** Under SGD with momentum, is the $\ell_2$ penalty still exactly equivalent to decoupled decay? Trace where $\lambda\mathbf{x}_t$ ends up inside the momentum buffer, then check experimentally.
   *Provenance: original (retained, ex2).*
1. [short-code] **Riding the fixed-product ridge.** Fix $\eta\lambda=3\times10^{-3}$ and rerun the decoupled sweep along $(\eta,\lambda)\in\{(10^{-3},3),(3\times10^{-3},1),(10^{-2},0.3)\}$. How constant is the held-out loss, and what breaks at the extremes?
   *Provenance: original (retained, ex3).*
1. [short-code] **Verifying the exemptions.** Apply decay to every parameter and track a rare token's embedding-row norm over training; relate this to OLMo 2's reported embedding instability and the LayerNorm gradient's $1/\|\mathbf{x}\|$ dependence.
   *Provenance: original (retained, ex4).*
1. [short-code] **Where activations overtake optimizer state.** Extend the accounting cell to activations: estimate `TinyLM`'s retained bf16 backward-pass activation memory per block, and find the batch size at which it overtakes optimizer state.
   *Provenance: original (retained, ex5).*
1. [short-code] **Decay independent of the schedule.** Rerun the decoupled grid with `weight_decay=wd/lr` so decay runs at a rate independent of $\eta$, matching Loshchilov & Hutter's original scaling. Does the best column stay put?
   *Provenance: original (retained, ex6).*
1. [short-code] **The full memory ledger.** Build a per-parameter byte ledger for `TinyLM` under AdamW (parameters, gradients, both moment buffers at true precision) against the section's $\approx20$-byte figure; repeat for SGD-with-momentum and Adam-with-$\ell_2$, confirming decoupling changes none of the totals.
   *Provenance: adapted from Stanford CS336, Assignment 1, Problem "adamwAccounting" (overlap medium — same accounting exercise, applied to this book's testbed and extended to a 3-optimizer comparison).*

---

## chapter_optimization/lr-scheduler.md — Schedules

**Topic:** Learning-rate schedules: constant baseline, polynomial/square-root decay, warmup, warmup–stable–decay, schedule-free SGD, SGLD.
**Current exercises:** 6; disposition: keep 5, rewrite 1 — only ex6's "read about SGLD and relate..." phrasing needed a concrete deliverable; the rest are already well-specified.

**External sources found:**
- Stanford CS336, Assignment 1 (basics), Problem "learning_rate_schedule" — implement the cosine-annealing-with-linear-warmup schedule used to train LLaMA (Touvron et al. 2023), parametrized by current step, max/min rate, warmup steps $T_w$, and cosine-annealing steps $T_c$ — https://github.com/stanford-cs336/spring2024-assignment1-basics
- Hägele, Bakouch, Kosson et al. (2024) and Defazio et al. (2024, schedule-free) and Welling & Teh (2011, SGLD) are already the book's own citations for ex4/ex5/ex6; no separate course assignment reproducing any of the three was found — this section's exercises are already close to the current research frontier on genuinely unsettled questions (WSD vs. cosine), which the section's own prose says explicitly.

**Proposed problem set** (7 problems):
1. [short-code] **The constant baseline.** Train at $\eta\in\{0.03,0.1,0.3,0.5\}$ under a constant schedule; report final test accuracy and describe the train–test gap's trend as $\eta$ shrinks.
   *Provenance: original (retained, ex1).*
1. [short-code] **Polynomial decay's exponent.** Implement $\eta_t=\eta_0(\beta t+1)^{-\alpha}$ ($\alpha=0.5$ recovers square-root decay); try $\alpha\in\{0.25,1,2\}$ and describe the early-progress/late-noise tradeoff.
   *Provenance: original (retained, ex2).*
1. [short-code] **Finding the stability ceiling.** At the demo's target rate, vary warmup from 1 to 10 epochs; then raise the target rate until no warmup length saves the run.
   *Provenance: original (retained, ex3).*
1. [extended] **An accuracy-vs-budget curve from one run.** Extend the plateau run to 60 epochs and branch 6-epoch decays at epochs 30, 40, 50; plot final accuracy against branch point.
   *Provenance: original (retained, ex4; already adapted from Hägele et al. in-book).*
1. [short-code] **Schedule-free descent on a noisy quadratic.** Implement schedule-free SGD on $f(\mathbf{x})=\frac{1}{2}\mathbf{x}^\top\mathrm{diag}(1,10)\mathbf{x}$ with Gaussian noise; plot $f(\mathbf{z}_t)$ and $f(\mathbf{x}_t)$ over 500 steps at constant $\eta$.
   *Provenance: original (retained, ex5).*
1. [conceptual] **From noise floor to sampler.** Using the noise-floor expression from :numref:`sec_sgd`, derive the injected-noise scale SGLD requires at a given learning rate, and state the one condition under which SGLD's noise and SGD's noise floor coincide.
   *Provenance: original (rewrite of ex6, replacing "read about... and relate" with a specific derivation).*
1. [short-code] **Cosine annealing with warmup.** Implement the LLaMA-style cosine-with-warmup schedule and add it to this section's testbed alongside the constant, polynomial, and WSD schedules already measured. Does it beat problem 2's polynomial-decay optimum at matched compute?
   *Provenance: adapted from Stanford CS336, Assignment 1, Problem "learning_rate_schedule" (overlap high — same schedule formula, applied to this book's testbed).*

---

## chapter_optimization/muon.md — Muon

**Topic:** Steepest descent under a chosen norm; the spectral-norm case (Muon) via Newton–Schulz orthogonalization; comparison with AdamW and Lion.
**Current exercises:** 6; disposition: keep 6, rewrite 0, drop 0 — zero clarity flags, the most rigorously specified file in the chapter; kept in full.

**External sources found:**
- Keller Jordan, "Muon: An optimizer for hidden layers in neural networks" (blog post, Dec. 8, 2024) — introduces the tuned quintic Newton–Schulz iteration (coefficients $3.4445,-4.7750,2.0315$), argues hidden-layer momentum updates are often near-low-rank so orthogonalizing amplifies underrepresented directions, and reports CIFAR-10-speedrun/NanoGPT results. Directly fetched and confirmed the post contains **no exercises or reader prompts** — exposition, not coursework — https://kellerjordan.github.io/posts/muon/
- EPFL CS-439 "Optimization for Machine Learning" (Flammarion & Jaggi), Spring 2025 — all 11 problem sets checked directly; none mentions Muon, spectral-norm steepest descent, or matrix orthogonalization — https://github.com/epfml/OptML_course
- No university course, textbook, or problem set of any kind was found assigning Muon, Newton–Schulz orthogonalization, or norm-dependent steepest descent as student exercises. This is a genuine, actively-checked finding, not a search gap: Muon (introduced Dec. 2024) is covered by primary sources — the original post, the follow-up paper, and production reports — not yet by pedagogy built on top of them.

**Proposed problem set** (7 problems):
1. [conceptual] **The sign-descent limit of Adam.** Set $\beta_1=\beta_2=0$ in :eqref:`eq_adam-moments`/:eqref:`eq_adam-update` to show the update becomes $\eta\,\mathrm{sign}(\mathbf{g}_t)$ as $\epsilon\to0$. Which norm ball in :eqref:`eq_muon-ball` does this solve, and what do the two moving averages restore?
   *Provenance: original (retained, ex1).*
1. [short-code] **Verifying the RMS-matching factor.** Show $\|\mathbf{U}\mathbf{V}^\top\|_F=\sqrt{\min(m,n)}$ for a rank-$\min(m,n)$ matrix; instrument an AdamW run of `TinyLM` to measure its actual update RMS and compare against the constant $0.2$ in :eqref:`eq_muon-update`.
   *Provenance: original (retained, ex2).*
1. [short-code] **How many Newton–Schulz steps are enough?** Rerun the hybrid with `num_iters=1` and `10`, measuring final loss and wall-clock time; plot the quintic $p(x)$ of :eqref:`eq_muon-quintic` to see what one application does to the spectrum.
   *Provenance: original (retained, ex3).*
1. [short-code] **Orthogonalizing the embedding table.** Move the embedding table and output head into the Muon group and rerun the sweep; using the one-hot-input argument, explain the effect on rare tokens' rows.
   *Provenance: original (retained, ex4).*
1. [short-code] **Lion in six lines.** Implement Lion and compare it with AdamW and the hybrid on `TinyLM` at matched four-point tuning; report optimizer state per parameter for all three.
   *Provenance: original (retained, ex5).*
1. [conceptual] **Completing the spectral-step bound.** Show that $\|\mathbf{A}\|_2\le1$ implies every diagonal entry of $\mathbf{U}^\top\mathbf{A}\mathbf{V}$ has absolute value at most 1, and identify when equality holds for all entries simultaneously.
   *Provenance: original (retained, ex6).*
1. [short-code] **How low-rank are real gradients?** Measure the singular-value spectrum of a hidden weight matrix's momentum buffer in `TinyLM` at three training points (init, 10%, end); report what fraction of the Frobenius norm the top 10% of singular values carry at each, and whether concentration grows or shrinks over training.
   *Provenance: inspired by Keller Jordan's Muon post (overlap low — the post makes the qualitative claim; this is a new quantitative measurement on this book's own testbed).*

---

## chapter_optimization/batch-size.md — Batch Size

**Topic:** The gradient-noise scale, the steps-to-target-vs-batch-size hyperbola, learning-rate scaling rules, and growing the batch during training.
**Current exercises:** 6; disposition: keep 6, rewrite 0, drop 0 — zero clarity flags, and this section's exercises already extend the field's primary methodology further than any external source found.

**External sources found:**
- Sam McCandlish, Jared Kaplan, Dario Amodei et al. (OpenAI), "An Empirical Model of Large-Batch Training" (arXiv:1812.06162, 2018) — defines the gradient noise scale and the steps-to-target hyperbola this section directly operationalizes; the field's primary source, confirmed by direct fetch.
- Stanford CS336, Assignment 3 (scaling) — its "training API" exposes batch size as one of several queryable hyperparameters when fitting scaling laws, but checked directly and confirmed batch size is not this assignment's dedicated subject (the phrase appears exactly once, naming a hyperparameter, not a topic) — https://github.com/stanford-cs336/spring2024-assignment3-scaling
- No course assignment was found that builds a homework problem directly on the gradient-noise-scale methodology itself. This section's own exercises already appear to be at or beyond the frontier of what is taught anywhere else on this specific topic — a strength worth preserving rather than a gap to fill with weaker adapted material.

**Proposed problem set** (6 problems):
1. [short-code] **Noise scale over training.** Extend the noise-scale cell to 3,000 steps, measuring every 500; plot noise scale against training loss at each checkpoint and state where doublings would land under "double $b$ when $b_{\textrm{noise}}$ overtakes it."
   *Provenance: original (retained, ex1).*
1. [short-code] **Past the elbow.** Extend the `TinyLM` sweep to $b=1024$ under the square-root rule (step count should flatten or rise); rerun at $b=1024$ with $\eta$ held at its $b=256$ value and explain both via :eqref:`eq_steps-examples`.
   *Provenance: original (retained, ex2).*
1. [short-code] **Factor-of-two check.** Estimate a run-averaged $b_{\textrm{noise}}$ for `TinyLM` and compare it against the steps-to-target curve's elbow (where examples-to-target reach twice its minimum).
   *Provenance: original (retained, ex3).*
1. [short-code] **Retuning instead of scaling.** For each CNN-sweep batch size, run a three-point learning-rate grid around the scaling rule's value and keep the best steps-to-target; state whether the elbow moves.
   *Provenance: original (retained, ex4).*
1. [extended] **Time-optimal vs. compute-optimal.** Model per-step cost as $t(b)=t_0(1+b/b_{\textrm{sat}})$ (measure both constants with `d2l.Timer`); combine with measured $S(b)$ to plot time-to-target against compute-to-target and identify both optima.
   *Provenance: original (retained, ex5).*
1. [conceptual] **Which population is noisiest?** Restrict `grad_sq_norm` to `TinyLM`'s embeddings, matrices, and vectors in turn, measuring three separate noise scales; relate the noisiest population to the sparse gradients that motivated AdaGrad.
   *Provenance: original (retained, ex6).*

---

## chapter_optimization/scaling.md — Scaling Up

**Topic:** Hyperparameter transfer across width via the maximal update parametrization (muP), verified with coordinate checks and transfer sweeps.
**Current exercises:** 4; disposition: keep 4, rewrite 0, drop 0 — smallest set in the chapter but zero clarity flags; kept whole, extended with two new problems drawn from directly-verified external sources.

**External sources found:**
- Greg Yang et al. (Microsoft), "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer" (arXiv:2203.03466, 2022) and the `microsoft/mup` repository — define the coordinate-check diagnostic this section uses: track the average $\ell_1$ norm of activation coordinates across widths over several training steps; stable coordinates confirm a correct implementation, exploding/vanishing ones reveal a bug — https://github.com/microsoft/mup
- Stanford CS336, Assignment 3 (scaling), Problem "chinchilla_isoflops" (5 pts) — reproduce Hoffmann et al.'s (2022) IsoFLOPs method to fit scaling laws for compute-optimal model/dataset size from provided runs, then predict the optimum at a held-out FLOPs budget — https://github.com/stanford-cs336/spring2024-assignment3-scaling
- Jordan Hoffmann et al. (DeepMind), "Training Compute-Optimal Large Language Models" (the Chinchilla paper, arXiv:2203.15556, 2022) — the scaling-law method CS336's assignment operationalizes, and the implicit reference for this section's "fitted scaling laws" strand of production practice.

**Proposed problem set** (6 problems):
1. [short-code] **Depth and the coordinate check.** Generalize `MLP` from 3 hidden layers to $L\in\{3,6,12,24\}$ at fixed width; run the coordinate check under both parametrizations and state what muP's width scaling does and does not promise across depth.
   *Provenance: original (retained, ex1).*
1. [short-code] **Transfer, verified directly.** Under each parametrization, sweep for the best rate at width 256, apply it unchanged at width 1,024, and compare against sweeping directly at 1,024; report all four optima.
   *Provenance: original (retained, ex2).*
1. [short-code] **Breaking muP on purpose.** Delete the $1/m$ logit scaling from `MuMLP` (keeping the hidden-rate rule) and rerun the coordinate check; repeat keeping the logit multiplier but giving hidden matrices the full rate.
   *Provenance: original (retained, ex3).*
1. [short-code] **Weight decay vs. parametrization.** Replace Adam with weight-decayed AdamW in the standard-parametrization sweep and train several times longer; relate any reduced drift to :citet:`Kosson.Welborn.Liu.ea.2025`'s stabilization argument.
   *Provenance: original (retained, ex4).*
1. [short-code] **A compute-optimal width, fitted.** Using problem 2's width sweep as small-scale data, fit an IsoFLOPs-style law relating held-out loss to width at fixed compute; predict the optimal width for a budget $8\times$ larger than any tested, train there, and report the prediction error.
   *Provenance: adapted from Stanford CS336, Assignment 3, Problem "chinchilla_isoflops" (overlap medium — same fitting method, applied to width instead of joint model/dataset size).*
1. [conceptual] **How many steps does the diagnostic need?** The coordinate check here reads activation scale at one early point in training. Argue, from the `mup` repo's practice of tracking coordinate size over *several* steps, why a single-step check can pass by coincidence on a broken implementation, and design a scaling bug invisible at step 0 but visible by step 10.
   *Provenance: adapted from the `microsoft/mup` coordinate-check methodology / Yang et al. (overlap medium — same diagnostic, applied to a constructed failure case not in the original source).*

---

## chapter_optimization/practice.md — Practice

**Topic:** Combining optimizer, schedule, and noise-control choices into a stable large-scale recipe: gradient clipping, weight averaging, and tuning protocol.
**Current exercises:** 6; disposition: keep 6, rewrite 0, drop 0 — the best-specified file in the chapter per the prior style review; kept whole, extended with one new problem.

**External sources found:**
- Stanford CS336, Assignment 1 (basics), Problem "gradient_clipping" (1 pt) — implement global-$\ell_2$-norm gradient clipping (scale by $\|\mathbf{g}\|_2/(M+\epsilon)$ above threshold $M$, PyTorch default $\epsilon=10^{-6}$) as a standalone function — https://github.com/stanford-cs336/spring2024-assignment1-basics
- Frank Schmidt, Felix Schneider, Philipp Hennig, "Descending through a Crowded Valley" (ICML 2021) — already the book's own citation for ex6; this is also the closest external analogue to ex1's "keep a tuning log" protocol, and no course assignment reproducing its benchmarking protocol as homework was found — ex1 appears to be an original contribution rather than an adaptation.

**Proposed problem set** (7 problems):
1. [short-code] **A ten-run budget, logged.** Using AdamW on `TinyLM`, reach training loss 1.1 in as few steps as possible tuning only the learning rate, with a budget of ten runs. Log, per run, the rate, steps, final loss, and the one-sentence conclusion drawn before the next run; report the log, not just the winner.
   *Provenance: original (retained, ex1).*
1. [short-code] **Rescaling the clipping demo.** Change the clipping demo's batch size from 64 to 256 and repeat the four-point AdamW learning-rate sweep; compare the optimum's shift against the square-root and linear scaling rules of :numref:`sec_batch_size`.
   *Provenance: original (retained, ex2).*
1. [short-code] **Guard, brake, or absent.** Sweep clipping threshold $\theta\in\{0.01,0.1,0.5,1,4,\infty\}$, recording final loss and clip-fire fraction; identify the three regimes and show that firing on every step is normalized gradient descent with step size $\eta\theta$.
   *Provenance: original (retained, ex3).*
1. [conceptual] **The decay timescale, computed.** Compute $\tau=B/(\eta\lambda D)$ for the DeepSeek-V3 and OLMo 2 rows of :numref:`tab_practice_recipes`; state what fraction of each dataset the averaging horizon spans, and what should happen to $\lambda$ if batch size doubled and $\tau$ is preserved.
   *Provenance: original (retained, ex4).*
1. [short-code] **EMA at both extremes.** Sweep EMA decay $\alpha\in\{0.9,0.99,0.999,0.9999\}$ in the weight-averaging demo; relate the window $1/(1-\alpha)$ to both failure ends within the 15-epoch budget.
   *Provenance: original (retained, ex5).*
1. [short-code] **Defaults vs. a four-run budget.** Run SGD-with-momentum, Adam, and AdamW at framework defaults on `TinyLM` for 2,000 steps; compare the best of the three against the grid-tuned Adam baseline and state which strategy a 4-run budget favors.
   *Provenance: original (retained, ex6).*
1. [short-code] **Clipping, built from scratch.** Implement your own global-$\ell_2$-norm clipping function matching problem 3's convention ($\epsilon=10^{-6}$); verify it reproduces problem 3's clip-fired counts at $\theta\in\{0.1,1,4\}$.
   *Provenance: adapted from Stanford CS336, Assignment 1, Problem "gradient_clipping" (overlap medium — same clipping convention, implemented independently and cross-checked).*
