# Deferred Curriculum Additions for the Mathematics Appendix

Status: planning note. Do not implement these additions until the existing text
has been reviewed and the scope has been approved.

The appendix already covers the mathematical foundations of deep learning in
considerable depth. The additions below address gaps that matter in current
teaching and practice. They should not become a survey of every mathematical
topic used in machine learning. Each addition should either complete an argument
that the book already begins, explain a method students will encounter in modern
training systems, or connect material that is currently presented in separate
chapters.

## Highest-priority additions

Two topics have dedicated notebooks. The entries below record the later
expansions proposed for those notebooks; none should be added until the scope is
approved.

### Adaptive Methods and the Dynamics of Training

Existing home:
`chapter_mdl-optimization/mdl-adaptive-stochastic-methods.md`

Place it after the existing notebook on gradient-based optimization. It should
assume the treatment of stochastic gradients and smooth optimization already
given there.

The notebook should cover:

1. A nonconvex SGD stationarity result, such as an
   $O(K^{-1/2})$ bound for a randomly selected iterate. State the smoothness,
   unbiasedness, and bounded-variance assumptions explicitly. This is the
   stochastic counterpart to the deterministic result already used to explain
   neural-network training.
2. AdaGrad as a coordinatewise or diagonal preconditioner. Derive its update
   from accumulated squared gradients and explain the geometry behind the
   effective learning rates. A regret result may be included if it can be
   proved without turning the section into an online-learning detour.
3. RMSProp as a change from cumulative to exponentially weighted second
   moments, followed by Adam's first- and second-moment estimates. Derive the
   bias-correction factors rather than merely stating the update.
4. The fact that Adam does not inherit a general convergence guarantee merely
   from its use of moving averages. Include a small version of the Reddi et al.
   counterexample and explain what AMSGrad changes.
5. AdamW and decoupled weight decay. Distinguish an $L_2$ penalty from weight
   decay once the optimizer applies a nonuniform preconditioner. Cross-reference
   the discussion of MAP estimation and regularization.
6. Learning-rate schedules: the behavior of $1/k$ schedules, warmup, cosine
   decay, and warmup-stable-decay schedules. Separate proven claims from
   empirical conventions.
7. A short preconditioning ladder connecting diagonal methods to K-FAC,
   Shampoo, and Muon. These methods need orientation and references, not full
   derivations. Cross-reference the existing Newton--Schulz material for Muon.

The executable material should include:

- SGD, Adam, and AdamW on the same ill-conditioned quadratic, with trajectories
  shown in the eigenbasis of the Hessian;
- the Adam nonconvergence example;
- the effect of coupled and decoupled weight decay under a diagonal
  preconditioner; and
- at least one schedule comparison in which the quantity being compared is
  mathematically defined before the experiment.

Exercises should ask students to derive the bias correction, analyze a
two-coordinate AdaGrad problem, and identify when weight decay and an $L_2$
penalty are equivalent.

### Concentration and Generalization

Existing home:
`chapter_mdl-probability-statistics/mdl-concentration-generalization.md`

Place it after the statistics notebook. The proposed sequence is random
variables, distributions, likelihood, statistics, concentration and
generalization, then naive Bayes as a capstone using those tools.

The notebook should cover:

1. Moment-generating functions and the Chernoff method, introduced for the
   purpose of deriving tail bounds rather than as an isolated definition.
2. Hoeffding's inequality with a short proof and all boundedness and
   independence assumptions stated. The main text already uses Hoeffding, so
   this section should repair that missing mathematical handoff.
3. Sub-Gaussian random variables and the relationship among variance proxies,
   tail bounds, and sums of independent variables.
4. Concentration in high dimensions: norm concentration for an isotropic
   Gaussian and near-orthogonality of random vectors. Reuse and cross-reference
   the geometric result in the linear algebra chapter.
5. The union bound and uniform convergence for a finite hypothesis class. Give
   one complete generalization proof rather than presenting a bound as a black
   box.
6. Rademacher complexity, including one calculation or bound for a linear
   function class. This should fulfill the promise made in the main
   generalization chapter.
7. The limits of the classical account: interpolation, double descent, and
   benign overfitting. Clearly distinguish theorem, stylized model, and
   empirical observation. This should become the book's single substantial
   treatment of double descent; other chapters should point here.

The principal experiment should compare Chebyshev, Hoeffding, and empirical
tails. A second experiment may produce a small random-features double-descent
curve, provided its finite-sample behavior is explained cautiously and the run
is inexpensive on CPU.

Exercises should include deriving a tail bound for an average, applying a union
bound to model selection, and computing an empirical Rademacher complexity in a
simple finite setting.

## Additions within existing notebooks

These topics are important but do not warrant separate files.

### Calculus and automatic differentiation

Add an introduction to the implicit function theorem and differentiation
through equations. Derive

$$
\frac{\partial x^*}{\partial \theta}
=-
\left(\frac{\partial g}{\partial x}\right)^{-1}
\frac{\partial g}{\partial \theta}
$$

for a solution defined by $g(x^*,\theta)=0$. Verify it on a small linear or
nonlinear system and connect it to bilevel optimization, deep equilibrium
models, and the adjoint method. The discussion must state the local
invertibility assumptions.

Also add the Leibniz rule for integrals with moving limits and name the Gamma
function where it is already used implicitly.

### Linear algebra

Add rank--nullity, projection onto a subspace, and the projection matrix
$QQ^\top$ for an orthonormal basis $Q$. Connect this to the least-squares hat
matrix. Include a compact Gram--Schmidt or QR experiment that checks
orthogonality and reconstruction.

A short Marchenko--Pastur remark may accompany the existing random-matrix
discussion. It should explain what question the law answers and give a reading
pointer; a full random-matrix treatment is not needed.

### Convex optimization and duality

Add proximal operators as generalized projections, derive soft thresholding as
the proximal operator of the $L_1$ norm, and show one ISTA update.

Add the saddle-point interpretation of Lagrangian duality. This should connect
the existing primal and dual constructions to minimax problems and provide a
careful pointer to adversarial training and GANs. Do not imply that generic GAN
objectives satisfy convex--concave minimax assumptions.

### Numerical computation

Chapter 6 owns the main treatment of floating-point formats, including FP8,
scaling, and the distinction among storage, arithmetic, and accumulation. The
appendix should cross-reference that chapter and focus on error analysis: add
summation error, pairwise or compensated summation, and Kahan summation near the
existing discussion of cancellation. Introduce stochastic rounding and state
what bias it removes and what variance it introduces.

### Information theory and compression

Extend the existing coding discussion with a definition of entropy rate for a
sequence source. Explain precisely when a language model can supply
probabilities to an arithmetic coder and which costs are excluded from the
idealized cross-entropy calculation.

Add a short treatment of empirical cross-entropy scaling laws. Present them as
observed regularities over specified model, data, and compute regimes, not as
universal laws. Include MDL and two-part codes in further reading.

Introduce proper scoring rules around the classification-loss discussion. Add
a brief comparison of Renyi or alpha divergences with KL where divergence
families are already discussed.

### Probability, statistics, and calibration

Add a calibration subsection with a reliability diagram and a quantitative
calibration measure. Explain the population conditions under which log loss is
strictly proper and why finite data, misspecification, regularization,
optimization error, and distribution shift can still produce a miscalibrated
model.

Add a bootstrap confidence interval for the existing naive Bayes experiment so
that its reported accuracy is accompanied by uncertainty. A short variational
inference paragraph may extend the existing ELBO discussion, with a pointer to
the VAE chapter.

### Differential equations, diffusion, and flow matching

Add the algebraic dictionary among score prediction, noise prediction,
$x_0$ prediction, velocity prediction, and $v$ prediction for Gaussian paths.
Specify the parameterization before writing conversion formulas. Include the
log-SNR clock and explain which transformations amount to a change of time and
which change the training objective.

Connect optimization to dynamics: gradient descent as forward Euler applied to
gradient flow, the learning-rate stability limit on a quadratic, and momentum
as a discretization of the heavy-ball ODE. Cross-reference the optimization
chapter rather than repeating its convergence proofs.

Extend the Langevin discussion with the Metropolis-adjusted Langevin algorithm.
The purpose is to show how a Metropolis--Hastings correction restores the target
stationary distribution after discretization. A general survey of MCMC belongs
in a later edition.

### Attention and transformer mathematics

Thread this topic through existing chapters instead of creating a separate
notebook now:

- connect near-orthogonality to the $1/\sqrt{d}$ scaling of dot-product
  attention, while being precise about the distributional assumptions;
- add an exercise on the row-wise softmax Jacobian and the Jacobian of an
  attention map;
- identify log-sum-exp and softmax conjugacy where it bears directly on
  attention; and
- add an outlook reference on attention as an interacting-particle system.

A full treatment of attention dynamics should wait until the presentation is
stable enough for textbook use.

## Reading-list and exercise material

The following topics should initially receive only a short remark, exercise, or
reading reference:

- $\mu$P, tensor programs, and width-dependent parameter scaling;
- neural tangent kernels;
- broader approximation theory;
- CUR decompositions and nuclear-norm methods;
- the KSG mutual-information estimator;
- the delta method; and
- fuller treatments of MCMC and variational inference.

Optimal transport already has enough coverage in the appendix and does not need
a new section.

## Editorial requirements

Any later implementation should follow these constraints:

1. Introduce every mathematical object before using it. State hypotheses next
   to the theorem or proposition that needs them.
2. Separate exact results, asymptotic results, modeling assumptions, and
   empirical observations. Avoid presenting a useful heuristic as a theorem.
3. Use executable cells to test central quantitative claims. A plot should
   answer a stated question rather than merely decorate the text.
4. Keep examples small enough to execute on CPU. New framework-specific code
   should support PyTorch, TensorFlow, JAX, and MXNet; MXNet remains as a legacy
   tab.
5. Avoid repeating material from the main book. Replace duplicate explanations
   with a short recall and a precise cross-reference.
6. Cite primary sources for modern results and a stable textbook source for
   established theory. Scaling-law and optimizer claims need dates and scope.
7. Add exercises that require derivation, diagnosis, or interpretation, not
   only substitution into a displayed formula.
8. Update slides only after the notebook prose and executable examples have
   passed review.

## Suggested order of work

When curriculum work resumes, use separate review checkpoints:

1. Review and approve the detailed outline for adaptive optimization.
2. Write and review that notebook before capturing outputs or preparing slides.
3. Review and approve the detailed outline for concentration and
   generalization.
4. Write and review that notebook and consolidate the existing double-descent
   discussions.
5. Add the implicit-differentiation and diffusion-parameterization material.
6. Add the remaining short sections in small, independently reviewable groups.
7. Add reading-list items only after the core additions have settled.

This ordering does not authorize implementation. It records the proposed scope
so that the additions can be evaluated before they alter the textbook.
