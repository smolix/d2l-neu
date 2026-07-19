# Review of the Optimization Chapter

## Scope and review standard

This review covers the current local sources and executed output for all twelve sections in `chapter_optimization`:

1. Landscapes
2. Gradient Descent
3. Stochastic Gradient Descent
4. Minibatches
5. Momentum
6. Adam
7. AdamW
8. Schedules
9. Muon
10. Batch Size
11. Scaling Up
12. Practice

The local working tree contains uncommitted edits to several of these files. I treated the working-tree Markdown as the current text and the locally rendered book as the source for inspecting figures and executed outputs. The review does not propose edits to those chapter files.

I checked every proposed omission against:

- the optimization appendix (`chapter_mdl-optimization`), especially *Gradient-Based Optimization*, *Adaptive Stochastic Methods*, *Convex Sets and Convex Functions*, *Constrained Optimization and Duality*, and *Numerical Stability and Conditioning*;
- the Basics part, especially linear regression and weight decay, backpropagation, initialization, generalization, dropout, normalization, mixed precision, parameter/state memory, reproducibility, and modern training recipes.

The intended standard is a top-five-university course: equations must be correct under their stated assumptions; experiments must distinguish illustration from evidence; code must teach reusable habits; and current-practice claims must be dated, sourced, and calibrated. Systems and distributed optimization are intentionally out of scope here. ZeRO, FSDP, DeepSpeed, optimizer-state sharding, communication, and parallel training should receive at most a short forward reference to the future systems chapter.

## Overall assessment

The chapter has an unusually strong ambition and a good high-level arc. It connects classical conditioning and stochastic gradients to AdamW, schedules, Muon, batch-size scaling, and muP; this is much closer to current practice than most textbooks. The best diagrams—the norm-ball diagram, the Newton--Schulz spectrum plot, the clipping diagnostic, and the schedule “river”—could anchor an excellent course.

It is not yet reliable enough to publish as authoritative teaching material. Several central mathematical statements are false or materially incomplete. The most serious are the fixed-radius steepest-descent formulation in the Muon section, the minibatch factorization used to motivate muP, the incomplete Adam batch-size scaling rule, the interpretation of AdamW as restoring an objective-level regularizer, and repeated confusion between mean-square and root-mean-square noise radii. The experimental sections also infer general conclusions from one seed and very coarse sweeps, sometimes while using the test set for selection. Those are not cosmetic defects: they teach students an invalid standard of empirical reasoning.

The chapter is also too long and repetitive for its conceptual payload. It repeatedly trains the same small testbeds, produces many nearly identical loss curves, and restates conclusions in prose, summaries, slide blocks, and exercises. A tighter core path, with clearly marked frontier/optional material, would improve both comprehension and maintainability.

## Priority corrections

### P0: correct before publication

| Location | Problem | Required correction |
|---|---|---|
| Muon, “Steepest Descent Under a Ball” | The optimization problem $\arg\min_{\|d\|\leq\eta}\langle g,d\rangle$ gives $d=-\eta g/\|g\|_2$ under the Euclidean norm, not the ordinary gradient-descent step $-\eta g$. The figure/table show the normalized result while the prose says it recovers gradient descent. | Separate direction selection from step magnitude, or formulate Euclidean GD through the regularized local model $\arg\min_d \langle g,d\rangle + \|d\|_2^2/(2\eta)$. For a general norm, introduce the squared-norm regularizer and its duality map. If the fixed ball is retained, explicitly call the result normalized steepest descent and explain how ordinary GD chooses a gradient-dependent radius. |
| Scaling Up, “Why It Moves” | $g=\delta h^\top$ and `sign(g)=sign(delta) sign(h)^T` hold for one example. A minibatch gradient is a sum of outer products, and the sign does not factorize. The text applies the single-example identity to batch-512 training as if it were exact. | Make the toy derivation explicitly single-example, then explain why it is only intuition for a batch. Prefer a short, correct tensor-program scaling derivation for the actual model. Do not infer the batch update by taking the sign of a factored single-example gradient. |
| Scaling Up, muP output layer | The table says biases are not width-scaled, but the implementation divides the entire output—including the output bias—by width. | Implement the output as `linear_without_bias(h) / width_multiplier + bias`, remove the bias, or change and justify the stated parametrization. Validate all parameter groups against the official `mup` rules. |
| Batch Size, Adam scaling | The cited SDE square-root rule is presented as changing only the Adam learning rate. The rule also changes $\beta_1$, $\beta_2$, and $\epsilon$ under its assumptions. | State the complete transformation, including its admissible range and assumptions, and distinguish it from the common heuristic “only multiply the learning rate by $\sqrt{k}$.” Malladi et al. give $\eta' = \sqrt{k}\eta$, $\beta_i'=1-k(1-\beta_i)$, and $\epsilon'=\epsilon/\sqrt{k}$ in the relevant convention. |
| Landscapes and SGD | The chapter alternates between saying the constant-step noise-ball “radius” is proportional to $\eta$ and deriving parameter variance proportional to $\eta$. | Use precise quantities: mean-square error/covariance is $O(\eta)$ in the small-step strongly convex model, while RMS distance is $O(\sqrt{\eta})$. Give the exact one-dimensional stationary variance $\eta\sigma^2/[\lambda(2-\eta\lambda)]$ before its small-$\eta$ approximation. |
| SGD, Robbins--Monro schedule | $\alpha=1/2$ is called a standard schedule satisfying the Robbins--Monro conditions, but $\sum_t t^{-2\alpha}$ diverges at $\alpha=1/2$. | State that the classical almost-sure conditions require $1/2<\alpha\leq1$, plus assumptions on the objective and noise. Explain separately why $1/\sqrt{t}$ appears in finite-horizon convex regret/convergence bounds, often with averaging. |
| AdamW, penalty decomposition | The coupled-Adam update is algebraically split into a data-gradient term and $\lambda x/\sqrt{v}$ even though both moment estimates are built from $g+\lambda x$. This is not an exact decomposition. | Label it as a schematic approximation or write the actual recurrence with moments of the penalized gradient. Then explain the mechanism without pretending that $v$ is the moment of the unregularized gradient. |
| AdamW, interpretation | AdamW is said to restore the “intended semantics” of $\ell_2$ regularization. Decoupled decay restores multiplicative shrinkage, but generally does not optimize the same penalized objective or have the same stationary points. | Distinguish three ideas: an $\ell_2$ penalty in the objective, multiplicative parameter decay, and a Gaussian-prior/MAP interpretation. The appendix already makes this distinction; cross-reference it here. |
| Muon, Newton--Schulz | Frobenius normalization is said to put every singular value in $(0,1]$ and the iteration is said to make every singular value one. Zero singular values remain zero; a rank-deficient matrix does not become full rank. | Say “all nonzero singular values are at most one” and describe convergence to the polar factor on the supported subspaces. Discuss rank deficiency and numerical damping explicitly. |
| Muon, scratch versus library | The scratch algorithm uses classical momentum, while canonical/current Muon implementations commonly use a Nesterov-style momentum update. The library comparison therefore is not necessarily the same algorithm. | Implement the canonical form used in the experiment or name the scratch variant and compare it separately. Document momentum convention, weight decay, RMS rescaling, orthogonalization coefficients, matrix selection, and tensor layout. |
| Practice, clipping and Adam | “Adam’s normalization already caps every coordinate near $\eta$” and therefore clipping cannot rescue an excessive Adam learning rate is false. $\hat m/\sqrt{\hat v}$ is not universally bounded by one, and clipping changes the moments if applied before the optimizer. | Remove the cap claim. Specify whether raw global gradients are clipped before moment updates, show the clip fraction and norm distribution, and explain that clipping can stabilize Adam without making an otherwise poor learning rate optimal. |
| Practice, EMA with BatchNorm | The chapter warns that BatchNorm statistics should be recomputed for checkpoint averaging, but not for EMA. Averaged weights and averaged running statistics are not generally self-consistent. | Recompute BatchNorm statistics for EMA/SWA weights before final evaluation, or copy and clearly justify live statistics. PyTorch’s averaging utilities explicitly provide `update_bn` for this reason. |

### P1: high-value corrections and qualifications

- Replace the claim that “every practical gradient is a minibatch estimate.” Full-batch methods, gradient accumulation, deterministic objectives, and curvature/Hessian-vector products are real counterexamples.
- Distinguish a stationary inflection such as $x^3$ from a strict saddle. State the convention if “saddle” is being used for every non-minimizing stationary point.
- State that a positive-semidefinite Hessian is inconclusive in the second-order test; it proves a strict local minimum only when positive definite, while zero eigenvalues require higher-order analysis.
- Remove the balanced-independent-coin model for Hessian eigenvalue signs. The Hessian is conditioned on the point being critical and its eigenvalues are highly structured. A Wigner-matrix exercise does not establish a fact about trained neural networks.
- Qualify all saddle-escape claims. Gradient descent with random initialization avoids strict saddles under regularity and step-size conditions; SGD escape results require assumptions on smoothness and noise. Noise does not automatically cross the barrier out of a genuine local minimum.
- Replace the condition-number contraction in Landscapes with the optimal quadratic result $\eta_*=2/(L+\mu)$ and factor $(\kappa-1)/(\kappa+1)$, already derived in the appendix. The current $1-2/\kappa$ expression describes a near-stability-edge choice, not the best generic rate.
- Do not quote a neural-network “condition number” without defining the subspace and point at which it is measured. Deep-network Hessians are commonly indefinite and singular, so global $L/\mu$ may be undefined.
- State the sampling assumptions behind unbiased minibatch gradients and $1/b$ variance. Sampling without replacement has a finite-population correction, while augmentation, dropout, and BatchNorm alter the stochastic objective.
- Replace “SGD performs a random walk near the optimum” with the stable autoregressive/stationary-distribution picture on a strongly convex quadratic.
- Correct “momentum is a leaky average.” The implemented $v_t=\beta v_{t-1}+g_t$ is an unnormalized exponentially weighted sum. Define whichever timescale is used: e-folding time $-1/\log\beta$, mean age $\beta/(1-\beta)$, or effective sample size $(1+\beta)/(1-\beta)$.
- Give the complete heavy-ball optimum: both $\eta_*=4/(\sqrt L+\sqrt\mu)^2$ and $\beta_*=((\sqrt\kappa-1)/(\sqrt\kappa+1))^2$. “Critical damping” is mode-specific; one $\beta$ cannot critically damp every eigenmode.
- Stop describing AdaGrad/RMSProp/Adam’s second moment as a Hessian-diagonal estimate. It is a gradient-second-moment statistic. Connections to the Fisher require a probabilistic model and expectation under the model/data distribution; they are not a general Hessian identity.
- Qualify Adam bias correction: it exactly removes the zero-initialization weighting under stationary moments, not nonstationarity in the current gradient distribution.
- Explain epsilon placement and dtype. `sqrt(v)+eps` and `sqrt(v+eps_root)` are materially different for small gradients and mixed precision.
- Correct the TinyLM claim that the character vocabulary is nearly balanced. Character frequencies are strongly skewed.
- Replace the claimed universal $\beta_2=0.95$. It is common in several transformer-pretraining recipes, while Adam’s framework default and influential recipes also use $0.999$. Give a dated workload table rather than a universal prescription.
- Correct optimizer-memory accounting. Mixed precision has multiple regimes; a bfloat16 parameter does not necessarily imply an FP32 master copy or FP32 gradient accumulator. Separate model parameters, gradients, master weights, $m/v$ state, and temporary `foreach`/fused buffers.
- Use optimization steps or tokens, not epochs, as the primary schedule variable. Epochs may remain in the classroom testbed, but current large-scale practice and budget transfer are step/token based.
- Describe schedule-free optimization as an algorithm with coupled iterates and train/eval semantics, not merely “evaluate a running average.”
- Treat the gradient-noise scale $\operatorname{tr}\Sigma/\|g\|^2$ as a proxy for, not the definition of, critical batch size. Critical batch size depends on optimizer, curvature, target, and efficiency metric.
- In the two-batch noise-scale estimator, discuss negative estimates, uncertainty, checkpoint dependence, correlated sequences, and units (examples versus tokens).
- Correct the historical statement that McCandlish et al. measured GPT-3-scale batches; the work predates GPT-3 and extrapolated across workloads.
- Explain that gradient accumulation is arithmetically equivalent to a larger batch only under controlled sample ordering, loss normalization, stochastic layers, batch-coupled layers, clipping, and update placement.
- Correct the ordinary spectral-norm scaling claim in Scaling Up. For a fan-in-normalized Gaussian matrix, the largest singular value scales like $1+\sqrt{n_{out}/n_{in}}$ up to constants/concentration, not simply $\sqrt{n_{out}/n_{in}}$. If an RMS-to-RMS operator norm is intended, define it explicitly.
- Do not claim a raw-activation coordinate check catches every muP bug. It should track activation coordinates and, crucially, changes from initialization across multiple early steps and widths. Vanishing changes are a failure even if activations remain $O(1)$.
- Correct the EMA “bias correction” analogy. An EMA initialized at $x_0$ has weights summing to one; its early issue is lag/window composition, not Adam’s zero-initialized moment deficit.
- Restrict JAX EMA state traversal to floating parameter/batch-stat leaves. Do not average RNG state, counters, integer state, or other non-parameter collections.

### P2: polish, evidence, and maintainability

- Remove “honest” throughout. It appears repeatedly as a confidence cue (“honest cartoon,” “honest answer,” “honest summary,” “retune honestly”) without adding a testable qualification. Replace it with the actual limitation, protocol, or uncertainty.
- Audit absolute terms such as “every,” “never,” “exactly,” “universal,” “settled,” and “standard.” Retain them only for identities or directly verified scope-limited statements.
- Remove the stray `%load_ext d2lbook.tab`/`interact_select` fenced cells at the start of `adam.md` and `adamw.md`; they are preprocessing artifacts, not reader-facing content.
- Date all “frontier” sections. A textbook should say “as of mid-2026” and separate durable mechanics from rapidly changing empirical rankings.
- Do not call a four-point learning-rate grid “tuned” without qualification. Use “best point in this coarse grid,” then either refine locally or lower the strength of the conclusion.
- Avoid claims such as “cannot be tuned away,” “never lost,” “same across seeds,” or “within run-to-run noise” unless the displayed experiment actually has repeated seeds and uncertainty.

## What is already covered elsewhere

The practical chapter should not duplicate the following material. It should add a one-sentence reminder and a precise cross-reference where needed.

| Topic | Existing treatment | Recommendation here |
|---|---|---|
| Smoothness, descent lemma, nonconvex stationarity | Appendix: *Gradient-Based Optimization* | Use the result and state assumptions; do not reproduce the proof. |
| Optimal step on a strongly convex quadratic, condition number, edge of stability | Appendix: *Gradient-Based Optimization* | Correct the practical cartoon and link to the exact derivation. |
| Momentum acceleration and Nesterov analysis | Appendix: *Gradient-Based Optimization* | Keep implementation and physical intuition; link to the rate proof. |
| Stochastic convergence/noise ball/step-size decay | Appendix: *Gradient-Based Optimization* | Keep the exact 1-D stationary calculation and practical schedules; link to the theorem. |
| Newton, quasi-Newton, and trust-region methods | Appendix: *Gradient-Based Optimization* | Give a fair practical map and link; no new long theory section. |
| AdaGrad, RMSProp, Adam, Adam failures/AMSGrad | Appendix: *Adaptive Stochastic Methods* | Keep executable mechanics and deep-learning evidence; eliminate duplicated/inconsistent claims. |
| AdamW, schedules, warmup, SVRG | Appendix: *Adaptive Stochastic Methods* | Cross-reference. Do not call SVRG “missing.” |
| K-FAC, Shampoo, Muon, muP hierarchy | Appendix: *Adaptive Stochastic Methods* | Use the practical chapter for implementation and evidence, not a second survey. |
| Convexity, subgradients, proximal and coordinate methods | Appendix: *Convex Sets and Convex Functions* | A short “where these tools matter” link is enough. |
| Constraints, KKT, projection, duality | Appendix: *Constrained Optimization and Duality* | Do not expand the practical chapter. |
| Floating point, stable losses, mixed precision, conditioning | Optimization appendix and Computation/Builder’s Guide | Link from the stability kit and optimizer-memory discussion. |
| Generalization, weight decay, early stopping, implicit regularization | Linear Regression and MLP Basics | Distinguish optimization behavior from generalization, then link. |
| Backpropagation and automatic differentiation | Preliminaries/MLP Basics | Assume it. |
| Initialization and normalization | MLP and Computation Basics | Refer to these when explaining warmup or stability; do not reteach. |
| Gradient clipping for recurrent models | Sequence-model Basics | The Practice section can generalize the technique and discuss modern large-model diagnostics. |
| Cosine/warmup/EMA in a modern vision recipe | Modern ConvNets | Use it as prior experience and focus here on mechanism and controlled comparison. |
| Reproducibility, checkpointing, state, and memory | Computation/Builder’s Guide | Require those practices in experiments and link to implementation details. |

This cross-check changes the main recommendation: the chapter is not missing more classical optimization theory. It is missing precision at the boundary between the appendix’s theory and the practical experiments.

## Section-by-section review

### Chapter introduction / index

**What works**

- The three recurring choices—direction, step scale, and noise management—are a useful organizing device.
- The reading list is more current and useful than a conventional optimizer catalog.

**Problems**

- The three-choice taxonomy becomes mathematically inconsistent in Muon because a fixed norm ball controls both direction and magnitude. Ordinary GD, normalized GD, sign descent, and Muon cannot all be described as changing only the norm while keeping a common radius without careful normalization.
- “Every gradient is estimated from a minibatch, never computed exactly” is false.
- “Roughly the order history made them” is not a useful promise and is not quite accurate.
- “Muon is the first credible challenger” is marketing language. Shampoo and other matrix preconditioners predate it, and benchmark outcomes depend on workload and tuning protocol. MLCommons AlgoPerf’s 2025 self-tuning track, for example, was won by schedule-free AdamW, while Shampoo won an external-tuning setting; there is no single universal winner ([AlgoPerf 2025 results](https://proceedings.iclr.cc/paper_files/paper/2025/hash/6bdde0373d53d4a501249547084bed43-Abstract-Conference.html)).
- The resources list describes repository records as all documented and reproducible. A speedrun repository is valuable evidence, but reproducibility still depends on hardware, software versions, rules, and independent reproduction.

**Suggested revision**

- Reframe the chapter around four questions: objective/gradient estimator; update geometry/preconditioner; step-size/time schedule; and validation/diagnostics. Parameterization and batch size affect all four.
- Mark AdamW as the default practical baseline, Muon and muP as an advanced/current-practice module, and the final claims as dated.

### Landscapes

**Mathematical accuracy**

- The local/global objective distinction is good, but the optimization objective and generalization metric should be separated more cleanly. Early stopping and validation are model selection, not merely alternative optimization goals.
- $x^3$ is a stationary inflection. Some nonconvex-optimization papers use “saddle” broadly for a stationary non-minimum, but students will reasonably expect a direction of positive and a direction of negative curvature. Define the convention and include a true two-dimensional strict saddle such as $x^2-y^2$.
- The Hessian test is incomplete at semidefinite critical points. A positive-definite Hessian implies a strict local minimum; an indefinite Hessian implies a saddle; a semidefinite Hessian is inconclusive.
- The independent balanced coin-flip story for thousands of Hessian eigenvalue signs should be removed. It is neither a model of a Hessian conditioned on criticality nor evidence about neural networks. Wigner’s semicircle law describes a random-matrix ensemble, not trained-network critical points.
- The text overstates the prevalence and isolation of local minima. Overparameterized networks have symmetries, flat directions, and often low-loss connections between solutions ([Garipov et al., 2018](https://arxiv.org/abs/1802.10026)).
- “Only noise” can leave a saddle/local minimum is false. With suitable random initialization and step sizes, deterministic gradient descent avoids strict saddles almost surely under regularity assumptions ([Lee et al., 2016](https://proceedings.mlr.press/v49/lee16.html)). Conversely, SGD escape theorems require conditions such as gradient/Hessian smoothness and dispersive noise ([Fang et al., 2019](https://proceedings.mlr.press/v99/fang19a.html)). Neither result says ordinary noise reliably crosses arbitrary local-minimum barriers.
- The condition-number rate should match the appendix’s exact result. A local neural-network Hessian can have zero and negative eigenvalues, so say explicitly when the strongly convex quadratic model applies.
- The stochastic-gradient formula needs iid/uniform sampling assumptions and a finite-population correction for sampling without replacement.

**Teaching and figures**

- Present a four-panel taxonomy: local minimum, local maximum, strict saddle, and degenerate stationary point. Put gradient and Hessian eigenvalues directly on the panels.
- Replace the hard-to-read 3-D saddle surface with contours plus a small cross-section. A contour plot shows descent and ascent directions more clearly and prints better.
- Add a visual boundary around the “local quadratic model”: one panel where it predicts dynamics well, one where flat/negative curvature invalidates the bowl story.
- The ill-conditioned valley is the most useful bridge in the section. Label $L$, $\mu$, the two eigendirections, the stable-step ceiling, and the predicted contraction on the same figure.

### Gradient Descent

**Mathematical accuracy**

- State the differentiability/smoothness conditions behind the Taylor remainder and the existence of a sufficiently small descent step.
- “Newton solves the step-size problem” is true for an exact positive-definite quadratic, not generally. Newton is a root-finding method for the gradient; outside a suitable local basin it requires damping, line search, a trust region, or a positive-definite model.
- Taking an absolute Hessian is a pedagogical toy, not a general repair. Mention modified Cholesky, damping, Gauss--Newton, or trust regions, then link to the appendix.
- The dense $O(d^2)$ storage/$O(d^3)$ factorization statement is correct for explicit dense Newton, but “nothing rescues it” is too broad. Hessian-vector products, conjugate-gradient/Hessian-free methods, L-BFGS, Gauss--Newton, K-FAC, and Shampoo avoid explicit dense factorization. They do not make exact Newton cheap, but they are important distinctions.
- Line searches need not always evaluate the full dataset; stochastic and minibatch line searches exist. The relevant practical concern is noisy, expensive function comparison.
- A raw inverse diagonal Hessian can be zero or negative on nonconvex objectives. It is not automatically a safe preconditioner.
- The exercise that says “binary search” should specify whether it is bisection on a monotone derivative or a derivative-free interval method such as golden-section search, and state unimodality assumptions.

**Code and figures**

- The learning-rate plots auto-scale independently, making stable, oscillatory, and divergent runs look deceptively comparable. Use common axes or explicit inset/escape annotations.
- Put $\eta$, the stability limit, iteration count, and optimum on each plot. Several trajectories are otherwise indistinguishable.
- The nonconvex Newton examples expand to enormous axes after divergence, hiding the local mechanism. Clip/annotate escaped iterates and retain a readable local inset.
- Most one- and two-dimensional algorithm mechanics are framework-independent. NumPy would be simpler and would avoid duplicated PyTorch/JAX cells; reserve framework tabs for actual model training.

### Stochastic Gradient Descent

**Mathematical accuracy**

- Near a strongly convex quadratic optimum, constant-step SGD is a stable AR(1)-type process, not an unbounded random walk. Derive the exact stationary variance and then show its small-step approximation.
- Robbins--Monro conditions are not sufficient by themselves. State unbiased/martingale-difference noise, bounded second moment (or an appropriate growth condition), regularity of the objective, and stability/boundedness assumptions before promising convergence.
- Separate almost-sure asymptotic schedules from finite-horizon $O(1/\sqrt{T})$ results and iterate averaging.
- “$O(1)$ per step” means independent of dataset size, not independent of model size.
- Shuffling without replacement is not simply iid sampling in a different order; it changes dependence and often improves finite-sum behavior. Label the standard dataloader practice separately from the iid proof.
- The claim that variance fell “exactly” as $1/b$ by three orders of magnitude is stronger than the displayed Monte Carlo experiment. Batch 1 to 512 is 2.71 orders, and one draw per size has sampling error.
- “100 times the compute buys 10 times quieter” concerns per-step example count under a simple model, not necessarily wall time or end-to-end sample efficiency.

**Teaching and figures**

- Replace three similar trajectory plots with one phase portrait containing the optimum, deterministic path, noisy paths, and a stationary covariance ellipse.
- Plot empirical variance with error bars across repeated minibatch draws and overlay the $1/b$ prediction. Include the finite-population correction if batches are drawn without replacement.
- Use the exact 1-D result to connect learning rate, curvature, and noise in one memorable diagram; that will prevent the radius-versus-squared-radius error from recurring later.

### Minibatches

**Mathematical and systems accuracy**

- The opening distributed example belongs in the future systems chapter and is also too restrictive. Global batch need not exceed world size because devices can process microbatches, accumulate gradients, or have idle/uneven work. Retain only a forward reference.
- The hardware argument should use arithmetic intensity and a roofline-style model rather than date-sensitive peak FLOP numbers alone.
- The vectorization benchmark is not a valid cross-framework/device benchmark. GPU work is asynchronous unless synchronized; JAX includes compilation unless warmup is separated; tiny matrices measure dispatch; `time.time` is a weak timer; and the JAX elementwise path uses a NumPy host buffer while the other paths may execute on an accelerator.
- Claims such as “option 2 halves traffic” and “batch 64 is as efficient as full batch” are machine-, layout-, and cache-dependent.
- The optimizer comparisons use different learning rates and epoch budgets. The scratch and concise PyTorch/JAX runs also use different epoch counts and compilation costs. They cannot support a framework or optimizer ranking.
- The epoch-axis calculation mishandles a partial last batch. Count examples/updates explicitly.
- Standardizing the entire 1,500-row dataset, including the response, is acceptable for a pure optimization demonstration only if it is labeled as such. It would be leakage in a predictive evaluation.
- Gradient accumulation equivalence needs the caveats listed under P1, especially BatchNorm, dropout, loss reduction, clipping, and optimizer-step placement.

**Code and figures**

- Build one sound benchmark helper: warmup; synchronize; measure compile separately; use `perf_counter`; repeat; report median and spread; print device, dtype, shapes, versions, and backend.
- Do not plot five nearly identical loss curves. Use one controlled plot with matched update/example budgets and one throughput plot with uncertainty.
- The final wall-clock plot uses a range much larger than the data and compresses the result into the left edge. Tighten it or use a log scale.
- Rename helpers by task. Reusing `train_lm` for a CNN later in the chapter teaches a misleading abstraction.

### Momentum

**Mathematical accuracy**

- Distinguish an unnormalized momentum buffer from a normalized EMA. If noise reduction is discussed, compare equal effective update scales: for independent gradient noise, $\operatorname{Var}(v)=\sigma^2/(1-\beta^2)$ for the unnormalized buffer, whereas the normalized EMA has variance $\sigma^2(1-\beta)/(1+\beta)$.
- “Noise reduction at no cost” is therefore wrong; the signal gain and effective learning rate change too.
- Define the chosen effective-window convention instead of equating all notions with $1/(1-\beta)$.
- State both optimal heavy-ball hyperparameters and the strongly convex quadratic assumptions.
- Critical damping is per eigenmode and depends jointly on $\eta$, $\beta$, and $\lambda$. Avoid phrases such as “$\beta=0.8$ is twice critical.”
- Nesterov has several equivalent-looking but not identical implementation conventions. Show the variable transformation that relates the textbook look-ahead form to PyTorch/JAX implementations and state initialization details.
- Claims that Nesterov’s correction is drowned out by minibatch noise need evidence and workload scope.

**Teaching and figures**

- The damping diagram should label $\eta\lambda$, roots/spectral radius, and which eigenmode is being shown.
- Show normalized-EMA and unnormalized-buffer impulse responses side by side. This would resolve several later Adam confusions.
- Consolidate the repeated airfoil curves into a controlled table/plot with the same budget and multiple seeds.

### Adam

**Mathematical accuracy**

- Recast AdaGrad as adaptive diagonal scaling driven by cumulative squared gradients, not a Hessian estimator. Mention the special Fisher connection only with its assumptions.
- AdaGrad’s accumulator is nondecreasing, but it need not grow linearly when gradients vanish. Its effective step is not inevitably $t^{-1/2}$.
- “Exactly right for convex, exactly wrong for nonconvex” should become a scoped statement about regret guarantees and continued adaptation in nonstationary deep training.
- Bias correction should be derived as correction for zero-initialized exponential weights under stationarity, then qualified for changing distributions.
- Epsilon is not an incidental implementation detail. Explain epsilon placement, scale, and mixed-precision implications.
- The “per-coordinate ceiling $\eta/\epsilon$” applies to the coefficient multiplying the first moment in one convention, not a universal bound on the parameter update.
- Framework implementations differ in epsilon semantics, AMSGrad, weight-decay coupling, dtype, fused/foreach paths, and capturability. Do not say they apply the displayed equation “exactly” without specifying flags and versions.

**Evidence and teaching**

- The TinyLM arrives as a black box in an optimizer chapter. Add a compact architecture/dataflow diagram and a parameter-count/state table; or use a small MLP for the first optimizer race and keep TinyLM as the realistic extension.
- The LM comparison is one seed, a coarse learning-rate grid, constant schedule, fixed short horizon, training loss only, and no controlled weight decay/clipping. It cannot establish that SGD “cannot be tuned,” that Adam wins by a stable percentage, or that the cause is heavy-tailed gradients.
- If heavy tails are part of the explanation, actually plot per-block gradient-coordinate or norm distributions, with a robust tail diagnostic.
- The CNN experiment repeats the same limitations. Use validation, not test, for optimizer selection and report test once.
- Add a one-step unit test comparing the scratch optimizer to the framework optimizer on a fixed gradient sequence. This is more educational than another loss curve.

### AdamW

**Mathematical accuracy**

- Correct the coupled-moment algebra and objective-versus-decay distinction described in P0.
- $\ell_2$ penalty and multiplicative decay are exactly equivalent for plain SGD under a matching learning-rate convention, not generally for momentum/adaptive methods.
- Decoupling does not eliminate learning-rate/weight-decay interaction. The shrink per step is $1-\eta_t\lambda$, so the schedule and total integrated learning rate matter. This currently conflicts with the heatmap interpretation.
- The statement that most transformer matrices are scale-invariant because they are followed by normalization is inaccurate for common pre-LN residual architectures. Identify the specific reparameterizations where scale invariance holds.
- “Do not decay embeddings/norms/biases” is a common heuristic, not a universal law. Embedding treatment is workload- and implementation-dependent.

**Experiments and figures**

- The PyTorch/JAX experiments use different learning rates while the prose suggests a common setting. Explain framework-specific calibration or use a controlled common protocol.
- A $3\times3$ single-seed heatmap does not demonstrate independence or behavior “across seeds.” Refine the grid near the optimum and repeat seeds.
- Use a shared color normalization and colorbar across coupled and decoupled heatmaps. Independent scales make the same color mean different losses. Mark minima with a high-contrast symbol rather than subtle bold text.
- Report validation loss/accuracy, not only training dynamics. Weight decay is precisely where optimization and generalization must be separated.
- Present memory accounting as a table of concrete precision regimes. PyTorch notes that `foreach` can consume extra peak memory ([AdamW documentation](https://docs.pytorch.org/docs/main/generated/torch.optim.AdamW.html)); include temporary memory separately from persistent state.

### Schedules

**Mathematical and experimental accuracy**

- Use step/token schedules as the primary API and explain epoch schedules as a pedagogical simplification.
- The section repeatedly checks test accuracy while comparing schedules. That is test-set tuning. Create train/validation/test splits, select with validation, and evaluate test once.
- Reset model initialization and data order for paired comparisons, or repeat independent seeds. Current PyTorch runs do not establish “identical network,” “reproducible comparison,” or “within run-to-run noise.”
- Square-root decay has guarantees only under specified convex/stochastic assumptions, often with averaging. “Neural networks want the opposite” is too broad.
- Cosine has more than one meaningful parameter: peak, floor/final rate, horizon, warmup, and often cycles/restarts.
- Present warmup mechanisms as hypotheses supported in particular regimes, not a settled single cause. Separate optimizer-moment startup, normalization/activation transients, curvature/sharpness evolution, and large-batch scaling.
- The $7.5$ learning-rate failure is a contrived point. Map a small target-LR-by-warmup grid to show the actual stability boundary.
- WSD is operationally useful, but claims that it “took over,” can plateau indefinitely, or appears in every published curve are too strong. The WSD paper says continuation is possible in principle, not that an unchanged run is indefinitely optimal ([Hägele et al., 2024](https://arxiv.org/abs/2410.05192)).
- Schedule-free methods use coupled iterates and explicit train/eval behavior; give the algorithm, not just the averaging analogy ([Defazio et al., 2024](https://arxiv.org/abs/2405.15682)).
- A “current frontier” section should be dated and periodically reviewed. As of mid-2026, Hyperball is directly relevant to the Muon/schedule/scale narrative and reports joint weight/update norm control ([Kosson et al., 2026](https://arxiv.org/abs/2606.16899)). Treat it as emerging evidence, not a new default.

**Figures**

- There are roughly seventeen similar train/test panels. Consolidate them into: (1) schedule shapes on a common step axis; (2) a controlled validation comparison with uncertainty; (3) one warmup stability map; and (4) one full WSD branching timeline.
- Do not share a y-axis between loss and accuracy. Small schedule differences become invisible and the units are unrelated.
- The branching plots restart the epoch axis, so the reader cannot see the branch in the parent run. Overlay branches on absolute training time/tokens and mark the fork checkpoint.
- Keep the river diagram, but enlarge labels and connect each branch to the corresponding empirical timeline.

### Muon

**Mathematical accuracy**

- Correct the fixed-ball inconsistency and rank-deficient Newton--Schulz claims in P0.
- “Adam is smoothed $\ell_\infty$ steepest descent” is an analogy, exact only in a limiting no-momentum/no-memory setting. Say so.
- Spectral norm controls worst-case amplification over dense inputs; it is not uniquely the “natural” matrix geometry. Data covariance, RMS-to-RMS geometry, and natural-gradient/Fisher geometry may be more relevant depending on the layer.
- An embedding is a linear map from one-hot vectors, so its operator interpretation does not disappear. The relevant induced geometry differs (for example, max row norm for an $\ell_1$-like input set), and spectral orthogonalization may be inappropriate.
- A Frobenius-normalized Newton--Schulz implementation should transpose tall matrices or choose the smaller Gram matrix to avoid unnecessarily large intermediates.
- The prose says bfloat16 is used but the scratch code does not consistently cast accordingly.
- The $0.2\sqrt{\max(m,n)}$ consistent-RMS factor is empirical, not a theorem that learning rates transfer directly. Optax describes it as an empirical convention ([Optax Muon documentation](https://optax.readthedocs.io/en/latest/api/generated/optax.contrib.muon.html)).
- “Fifteen matmuls” and “about one percent overhead” depend on the polynomial degree, number of iterations, fusion, shape, and hardware. Report measured overhead for the shown model/device.
- “Muon is Shampoo without memory” is a useful heuristic only in a restricted instantaneous/full-rank sense. State damping, pseudoinverse, accumulation, and rank differences.
- Convolution layouts differ between PyTorch and JAX/Flax. Define the flattening/dimension numbers rather than relying on a shape rule that silently changes the operator.

**Implementation and evidence**

- Current stable PyTorch documentation exposes `torch.optim.Muon` for 2-D parameters with Nesterov momentum enabled by default and nonzero default weight decay; code and prose should pin explicit arguments rather than inherit version defaults ([PyTorch Muon documentation](https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html)).
- Build parameter groups from named roles, not only tensor rank. Attention/output/embedding matrices may all be 2-D but require different treatment.
- Add a scratch-versus-library one-step conformance test for both PyTorch and JAX, including tall/wide/rank-deficient matrices.
- The four-point one-seed races do not justify “never lost.” Report best-in-grid, uncertainty, validation metric, equal token/compute/wall budgets, and local refinements.
- Separate evidence tiers: speedrun demonstrations; controlled small-model ablations; large open training reports; and independent benchmarks. “Old Optimizer, New Norm” explicitly discusses the optimizer equivalence after simplifying moment machinery, not a universal optimizer ranking ([Bernstein & Newhouse, 2024](https://arxiv.org/abs/2409.20325)).
- The norm-ball figure is conceptually strong, but it currently exposes the Euclidean normalization contradiction. Fixing the derivation will make it one of the chapter’s best figures.

### Batch Size

**Mathematical accuracy**

- Present $B_{noise}=\operatorname{tr}\Sigma/\|g\|^2$ as the “simple noise scale” proxy. OpenAI’s presentation itself calls the mapping to useful batch size approximate ([McCandlish et al., 2018](https://openai.com/index/how-ai-training-scales/)).
- “Below the noise scale, mostly noise” is about expected squared norm, not whether the stochastic direction is useful. “Above it, essentially exact” also overstates the result.
- Add uncertainty to the two-batch estimator. Its signal estimate can be negative and its noise estimate unstable. Repeated independent draws or bootstrap intervals are required.
- The language-model windows overlap and are correlated, violating the simple iid-sequence model. Report both sequence and token batch sizes.
- The steps-to-target sweep should not be described as validation of a hyperbola without fitting the curve, showing confidence intervals, and estimating the noise scale at corresponding checkpoints.
- Shallue et al. independently tune learning rate, momentum, and schedule at each batch size; the current sweep does not. Their study also cautions against attributing generalization degradation to batch size before retuning ([Shallue et al., 2019](https://www.jmlr.org/papers/v20/18-789.html)).
- Linear and square-root learning-rate rules are regime-dependent heuristics. Include the full Adam SDE transformation and explain where its transformed $\beta$ values become invalid.
- Batch-size ramping is a useful pattern, not a universal frontier standard. Its value depends on changing noise scale, throughput, schedule, optimizer, and data mixture.

**Figures**

- Overlay a fitted steps-to-target model and a vertical band for the independently estimated noise scale. With only four or five points, do not describe a sharply verified elbow.
- Add repeated-seed error bars. The “perfect scaling” line should not be anchored to a noisy single smallest-batch observation.
- Put examples, tokens, optimizer steps, and wall time in separate, explicitly labeled views; they answer different questions.
- Keep distributed scaling to one forward-reference paragraph, as requested.

### Scaling Up / muP

**Mathematical accuracy**

- Fix the single-example factorization, bias scaling, spectral-norm formula, and coordinate-check criteria described above.
- Validate the simplified table against Tensor Programs V and the official Microsoft implementation. The core claim—stable hyperparameter transfer across width—is supported by the muP literature ([Tensor Programs V](https://arxiv.org/abs/2203.03466), [Microsoft `mup`](https://github.com/microsoft/mup)), but exact rules depend on parameter role, width multipliers, initialization, optimizer, and readout.
- “Flat or falling activations means stable” is wrong. Falling update magnitudes can indicate lazy/frozen feature learning even when raw activations remain bounded.
- A one-step raw-activation check may miss a bug whose effect appears over several updates. Track coordinate distributions and $\Delta h=h_t-h_0$ for several early steps.
- “Tune small, transfer big is universal” should become a scoped engineering goal. Transfer has been demonstrated for specified scaling families, not every architecture, depth, data mixture, batch, schedule, or regularizer.
- Do not imply that Muon RMS scaling automatically satisfies muP’s spectral conditions. That connection is active research and depends on width/shape conventions.

**Experiments and figures**

- The standard-parametrization sweep does not visually establish all of the claimed learning-rate drift. Print the numeric minima, refine around each optimum, and show seed variability.
- Evaluate transfer on validation performance and at more than one training horizon; short-horizon optima may differ from converged optima.
- Plot update-to-weight ratios and activation changes by layer/width, not only raw activation magnitude.
- Use a base-width vertical reference and confidence bands. Make “transfer” visually mean that one hyperparameter selected at the base width remains competitive at larger widths.
- Treat the 2025/2026 spectral optimizer literature as dated evidence. Hyperball is especially relevant because it changes both weight and update norm control; include it in a small “open questions” box rather than rewriting the chapter around a very recent result.

### Practice

**Accuracy and evidence**

- The recipe table has undisclosed fields. Absence in a paper/report does not imply the default or establish consensus. Mark cells “not reported,” and infer only from disclosed configurations.
- A global clipping threshold of one is not invariant to parameterization, model size, loss reduction, or units. Teach monitoring: pre-clip norm distribution, fraction of clipped steps, scale factor, and per-block diagnostics.
- Specify clipping before optimizer moments/preconditioning and after unscaling in mixed precision.
- QK normalization can constrain one source of logit growth but does not absolutely bound attention logits; alignment and learned scales still matter.
- A report of “zero spikes” is not causal evidence that MuonClip alone produced the result.
- EMA should recompute BatchNorm statistics, and the JAX state selection must exclude non-floating state.
- Do not evaluate EMA every epoch on the test set. Select decay/evaluation protocol on validation and test once.
- “EMA adds one to three points” needs multiple seeds and workload scope.
- “Change one thing at a time” is a debugging heuristic, not a general tuning strategy. Optimizer, learning rate, batch size, schedule, and decay interact. Recommend staged joint tuning or low-discrepancy/random search after a diagnostic baseline.
- The dismissals of SAM, SVRG, LARS, and LAMB are too absolute. State the regimes where each is relevant and why it is not in the core recipe. The appendix already teaches SVRG theory; no duplicate section is needed.
- Model soups and diffusion EMA are useful context but can be shortened or moved to a note; they interrupt the chapter’s optimizer-selection conclusion.

**Suggested practical endpoint**

End with a reproducible decision sheet rather than a universal recipe:

1. Define data, model, objective, validation metric, token/example/compute budget, and precision.
2. Start with an explicit AdamW baseline and explicit parameter groups.
3. Sweep peak learning rate first on a validation budget; record divergence and clip fraction.
4. Jointly refine schedule, weight decay, and batch size.
5. Compare a challenger only under equal tuning budget and at least three seeds near the apparent optimum.
6. Report training/validation curves, throughput, persistent/peak memory, tokens/examples, and uncertainty.
7. Evaluate test once after protocol selection.
8. Escalate to Muon/muP when scale and matrix-heavy architecture make their engineering cost worthwhile.

## Topics that should be added to this chapter

These additions are not already adequately covered in the appendix or Basics, and they improve the practical chapter without drifting into distributed systems.

### 1. A precise “optimizer experiment” protocol

This is the largest genuine omission. The chapter repeatedly compares optimizers but never fully teaches how to make the comparison valid. Add a short early box and enforce it throughout:

- same initialization/data-order pair for paired diagnostics, plus independent-seed repeats for conclusions;
- train/validation/test separation;
- equal tuning budget and a declared search space per optimizer;
- equal examples/tokens, compute, or wall time—state which;
- local refinement after a coarse sweep;
- divergence handling declared before running;
- confidence intervals or all seed traces;
- persistent state and peak-memory accounting;
- exact software version, device, dtype, and optimizer flags.

This would turn the current repeated caveats into a transferable scientific skill.

### 2. Parameter groups as part of the algorithm

AdamW and Muon both depend on which tensors receive decay or matrix orthogonalization, yet parameter grouping is treated as housekeeping. Add one compact diagram of a transformer/MLP parameter tree showing:

- matrix weights sent to Muon;
- embeddings/output heads/norm scales/biases sent to AdamW;
- decay and no-decay groups;
- group-specific learning rates and dtype/state.

Include assertions that every trainable parameter belongs to exactly one group. This is algorithmic correctness, not a systems topic.

### 3. Update diagnostics

The chapter plots loss frequently but rarely plots the quantities its explanations invoke. Add reusable diagnostics for:

- gradient norm and clip fraction;
- update norm and update-to-weight ratio by block;
- Adam $\sqrt{\hat v}$ distribution/effective step coefficients;
- momentum-buffer norm;
- Muon singular-value spectrum before/after orthogonalization;
- activation and activation-change coordinates across widths;
- validation loss versus training loss.

One dashboard applied to AdamW and Muon would teach more than several standalone races.

### 4. Optimizer correctness tests

Before expensive training, teach three cheap tests:

- a fixed-gradient-sequence recurrence test with hand-computed expected updates;
- scratch-versus-framework one-step equivalence, including parameter groups and epsilon convention;
- quadratic sanity tests for convergence, stability boundary, momentum roots, and scale equivariance.

These tests are especially valuable for Colab notebooks and changing framework APIs.

### 5. A dated evidence map, not a winner list

Add a half-page table with columns: method, geometry/state, strongest evidence type, workloads, tuning protocol, cost, and open uncertainty. Include SGD+momentum, AdamW, Shampoo, Muon, schedule-free AdamW, and emerging Hyperball. Keep K-FAC/Shampoo theory in the appendix; here the goal is to teach evidence calibration.

### 6. Time-scale conventions

Use a small common box to define:

- optimizer step;
- microbatch, batch, accumulated/global batch;
- example and token;
- epoch;
- schedule horizon;
- EMA/momentum half-life or e-folding time.

Many current ambiguities arise because experiments silently switch among these units.

## Topics not to add here

- Do not add a second derivation-heavy survey of convex optimization, constraints, proximal methods, quasi-Newton methods, trust regions, or variance reduction. The appendix already covers them.
- Do not add a long mixed-precision tutorial. Link to the numerical-stability and Computation chapters; give only optimizer-specific state/epsilon/clipping implications.
- Do not add a second generalization chapter. Use validation properly and link to the Basics treatment of weight decay, early stopping, and implicit regularization.
- Do not expand ZeRO, FSDP, DeepSpeed, tensor/data/pipeline parallelism, all-reduce, or communication overlap. Add one forward reference to the future systems/distributed-optimization chapter.

## Figure and visualization redesign

### Current recurring problems

- Independent auto-scaling makes algorithm comparisons visually misleading.
- Many panels lack the optimizer, learning rate, seed count, budget, or device in either the graphic or caption.
- Single trajectories are presented as method comparisons without uncertainty.
- Loss and accuracy sometimes share an axis.
- The same airfoil/CNN curve is redrawn many times with little new information.
- Heatmaps use independent color scales and no colorbar.
- WSD branches restart time, concealing their relation to the parent trajectory.
- Batch-size plots show neither an estimated noise-scale band nor a fitted saturation model.
- muP plots inspect raw activations instead of the update/feature changes central to the claim.

### Proposed visual grammar

Each major idea should have at most three visual layers:

1. **Mechanism schematic:** clean geometry, annotated equation, no experimental clutter.
2. **Controlled diagnostic:** a quadratic or fixed small model where the prediction is directly measured.
3. **Real-workload result:** validation metric, declared budget, multiple seeds, uncertainty, and complete caption.

Use consistent colors throughout: SGD, momentum, AdamW, Muon, theory/reference. Use common axes when comparing trajectories; use log axes only when they reveal a stated scaling relation. Every empirical caption should include framework, device, dtype, seed count, train/validation split, budget, and whether hyperparameters were independently tuned.

### Figures worth retaining and improving

- Retain the norm-ball diagram after correcting the Euclidean derivation.
- Retain the Newton--Schulz singular-value plot; add zero singular values and tall/wide cases.
- Retain the clipping stability plot; add clip fraction and pre-clip norms.
- Retain the schedule river; align it with an absolute-time empirical branching plot.
- Retain the ill-conditioned valley; annotate eigenmodes and predicted rates.

### Figures to consolidate or replace

- Collapse the numerous airfoil loss traces into one controlled optimizer panel.
- Collapse schedule results into four figures described in the Schedules review.
- Replace the AdamW heatmaps with shared-scale heatmaps plus uncertainty or a response surface with contours.
- Replace the SGD trajectory triptych with one phase portrait and stationary ellipse.
- Replace the minibatch timing demo with a proper roofline/throughput diagnostic.
- Replace raw muP activation panels with per-layer activation-change/update-ratio plots.

## Code-quality review

### Framework choice

Use NumPy for one-dimensional functions, quadratics, recurrence demonstrations, spectral calculations, and plotting whenever autodiff or accelerator execution is not the lesson. PyTorch/JAX tabs are appropriate for model training, optimizer APIs, JIT semantics, and parameter-tree handling. This will remove substantial duplicated code and make the mathematical cells easier to inspect.

### Reproducibility and state

- Avoid mutable module-level variables such as `eta`, `beta`, counters, and global JAX keys. Pass configuration and PRNG keys explicitly.
- Make cells safely rerunnable and avoid dependence on execution order.
- Use immutable result records containing config, seed, metrics, timing, device, and versions.
- Split train, validation, and test once and name them correctly.
- Reset seeds/initial weights deliberately for paired comparisons, and use independent seeds for uncertainty.

### Benchmarking

- Warm up compiled/device code and report compilation separately.
- Synchronize accelerators around timing.
- Use `time.perf_counter`, repeated measurements, and median/quantiles.
- Avoid comparing host NumPy loops to accelerator kernels as though the framework were the causal variable.
- Record actual examples/tokens processed, including partial batches.
- Report both persistent and peak memory.

### Optimizer implementation

- Add recurrence-level unit tests for scratch implementations.
- Pin framework optimizer arguments rather than relying on changing defaults.
- Document epsilon placement, momentum convention, decay convention, dtype, and parameter grouping.
- Assert that every parameter belongs to exactly one optimizer group.
- Handle tall/wide/rank-deficient matrices in Muon and framework-specific convolution layouts.
- In JAX, filter state collections by semantic type rather than applying tree operations to every leaf.

### Abstraction and duplication

- Do not use a language-model training helper name for CNN training.
- Factor common experiment code into small, transparent helpers, but keep optimizer recurrences visible.
- Avoid copying TinyLM/CNN definitions across sections if the saved `d2l` versions are authoritative; conversely, do not hide the architecture when it is necessary to interpret an experiment.
- Remove source/preprocessor artifacts from reader-facing Markdown.

## Writing and pedagogy

### Tone

The prose is energetic, but it often tries to manufacture confidence with “honest,” “settled,” “exactly,” “never,” and “universal.” Scientific authority should come from conditions and evidence. Prefer:

- “under the quadratic model”;
- “in this four-point grid”;
- “for these three seeds”;
- “reported by these two pretraining runs”;
- “an empirical convention as of mid-2026”;
- “the mechanism is an active hypothesis.”

### Definitions before slogans

Several memorable slogans are introduced before their domain of validity: noise ball, critical damping, natural matrix geometry, critical batch size, maximal update, and schedule-free averaging. Put a boxed definition/assumptions line before each slogan. Students should be able to answer “what precisely is measured?” and “under which model is this true?”

### Separate identity, theorem, heuristic, and report

Use visually distinct labels:

- **Identity:** algebraically true.
- **Theorem:** assumptions and conclusion.
- **Heuristic:** useful mechanism without a guarantee.
- **Experimental result:** protocol and uncertainty.
- **Industry report:** attributed observation, not independent causal proof.

This single change would prevent many of the chapter’s strongest overclaims.

### Reduce repetition

The chapter repeats the same idea in prose, output commentary, summary, slides, and exercises. Aim for one main explanation, one diagnostic figure, one real result, and one exercise. Slides should be generated from the same claims or kept deliberately shorter to avoid drift.

## Recommended chapter architecture

The current twelve sections can remain as files, but the learning path should be explicit.

### Core path

1. **Landscapes and evidence protocol** — local models, curvature, stochastic gradients, and how optimizer claims will be tested.
2. **Gradient descent and SGD** — exact quadratic dynamics, stationary noise, and schedules.
3. **Minibatches and momentum** — statistical/compute tradeoff and filtering/acceleration.
4. **Adam and AdamW** — adaptive moments, bias correction, decoupled decay, parameter groups, and memory.
5. **Schedules, batch size, and practice** — validation-based tuning, diagnostics, and a reproducible recipe.

### Advanced/current-practice path

6. **Muon** — update geometry, polar orthogonalization, parameter roles, and calibrated evidence.
7. **Scaling Up / muP** — parametrization, coordinate checks, and transfer.

Muon and muP should be clearly marked optional/advanced. A student should be able to complete the core path and run a defensible AdamW experiment without first accepting emerging optimizer claims.

## Verification plan for the rewrite

### Mathematical checks

- Symbolically or numerically verify every displayed optimizer recurrence on a two-coordinate quadratic.
- Test the exact SGD stationary variance against simulation.
- Verify heavy-ball roots/rates at the stated optimal $\eta$ and $\beta$.
- Check Adam/AdamW scratch updates against hand calculations and framework implementations.
- Test Muon Newton--Schulz on square, tall, wide, and rank-deficient matrices.
- Check muP scaling rules parameter group by parameter group, including output bias.
- Verify the full SDE batch-scaling transformation and identify invalid transformed hyperparameters.

### Empirical checks

- Use validation for all model/schedule/optimizer choices; test once.
- Run at least three seeds near each reported optimum; more if intervals overlap materially.
- Refine coarse grids locally before using “tuned.”
- Report equal token/example and, where relevant, equal compute/wall-clock comparisons.
- Include error bars/all traces and full experiment metadata.
- Measure the proposed mechanism: do not invoke heavy tails, noise scale, singular values, update ratios, or activation stability without plotting/estimating them.

### Code checks

- Execute notebooks from a fresh environment and in top-to-bottom order.
- Rerun key cells to test idempotence.
- Test CPU and accelerator timing paths with synchronization.
- Pin/version-check newer optimizer APIs such as PyTorch Muon.
- Add lightweight recurrence/conformance tests that run without training full notebooks.

## Primary references used to verify this review

- Lee et al., “Gradient Descent Only Converges to Minimizers” ([PMLR, 2016](https://proceedings.mlr.press/v49/lee16.html)).
- Fang, Lin, and Zhang, “Sharp Analysis for Nonconvex SGD Escaping from Saddle Points” ([PMLR, 2019](https://proceedings.mlr.press/v99/fang19a.html)).
- Shallue et al., “Measuring the Effects of Data Parallelism on Neural Network Training” ([JMLR, 2019](https://www.jmlr.org/papers/v20/18-789.html)).
- Malladi et al., “SDEs and Scaling Rules for Adaptive Gradient Algorithms” ([arXiv, 2022](https://arxiv.org/abs/2205.10287)); the authors’ practical statement of the full Adam rule is also useful ([Malladi, 2024](https://sadhikamalladi.github.io/blog/2024/01/22/SDEs-ScalingRules/)).
- Yang et al., “Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer” ([arXiv, 2022](https://arxiv.org/abs/2203.03466)) and the official [`mup` implementation](https://github.com/microsoft/mup).
- Bernstein and Newhouse, “Old Optimizer, New Norm” ([arXiv, 2024](https://arxiv.org/abs/2409.20325)).
- Hägele et al., “Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations” / WSD ([arXiv, 2024](https://arxiv.org/abs/2410.05192)).
- Defazio et al., “The Road Less Scheduled” ([arXiv, 2024](https://arxiv.org/abs/2405.15682)).
- PyTorch’s current [Muon](https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html), [AdamW](https://docs.pytorch.org/docs/main/generated/torch.optim.AdamW.html), and [optimizer averaging](https://docs.pytorch.org/docs/main/optim.html) documentation.
- Optax’s current [Muon documentation](https://optax.readthedocs.io/en/latest/api/generated/optax.contrib.muon.html).
- Dahl et al./MLCommons, AlgoPerf competition results ([ICLR, 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/6bdde0373d53d4a501249547084bed43-Abstract-Conference.html)).
- Kosson et al., “Fantastic Pretraining Optimizers and Where to Find Them II: Hyperball” ([arXiv, 2026](https://arxiv.org/abs/2606.16899)). This is very recent and should be treated as emerging evidence rather than settled practice.

## Bottom line

The chapter already contains almost all of the right broad topics. Its priority is not adding more optimizer names or duplicating the appendix. It needs mathematical repair at a handful of load-bearing points, a much stronger empirical protocol, a reduction in repeated figures, and clearer separation of theorem, intuition, and current industry report. With those changes, the chapter can become exceptional: it would teach not just optimizer formulas, but how to reason correctly about optimization claims in modern deep learning.
