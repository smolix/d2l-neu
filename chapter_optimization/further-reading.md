# Further Reading
:label:`sec_optimization-further-reading`

This section collects foundational textbooks, survey papers, and recent research extending the methods of this chapter. Mathematical proofs of the convergence rates cited in the text are given in :numref:`chap_mdl-optimization` (with additional theory references in :numref:`sec_mdl-optimization-further-reading`), automated hyperparameter optimization is covered in :numref:`chap_hyperopt`, and distributed multi-device training systems are covered in :numref:`chap_performance`.

## Foundational Textbooks, Surveys, and Playbooks

* **Convex and Numerical Optimization:** :citet:`Boyd.Vandenberghe.2004` (*[Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/)*) is the standard reference for conditioning, gradient descent, Newton's method (:numref:`sec_gd`), and steepest descent under general norms (:numref:`sec_muon`). :citet:`Nocedal.Wright.2006` (*Numerical Optimization*) provides a comprehensive treatment of Newton, quasi-Newton, line-search, and trust-region methods.
* **Stochastic Optimization:** :citet:`Bottou.Curtis.Nocedal.2018` (*[Optimization Methods for Large-Scale Machine Learning](https://arxiv.org/abs/1606.04838)*) surveys stochastic gradient noise balls, step-size decay (:numref:`sec_sgd`), and variance reduction.
* **Interactive Guides and Practitioner Playbooks:** Gabriel Goh's *[Why Momentum Really Works](https://distill.pub/2017/momentum/)* (Distill, 2017) provides interactive visualizations of critical damping and the $\sqrt{\kappa}$ acceleration law (:numref:`sec_momentum`); Jeremy Bernstein's *[Deriving Muon](https://jeremybernste.in/writing/deriving-muon)* and :citet:`Bernstein.Newhouse.2024` unify steepest descent across norm balls (:numref:`sec_muon`); EleutherAI's *[Practitioner's Guide to Maximal Update Parameterization](https://blog.eleuther.ai/mutransfer/)* details $\mu\text{P}$ coordinate checks (:numref:`sec_scaling`); and Google Research's *[Deep Learning Tuning Playbook](https://github.com/google-research/tuning_playbook)* :cite:`Godbole.Dahl.Gilmer.ea.2023` formalizes the distinction between scientific, nuisance, and fixed hyperparameters used throughout :numref:`sec_test_models` and :numref:`sec_practice`.

## The Loss Surface

### Visualizing High-Dimensional Loss Landscapes and Mode Connectivity
:label:`subsec_optimization-fr-connected-minima`

Low-dimensional cross-sections of neural network loss surfaces can be misleading unless perturbation directions are normalized filter-by-filter to remove positive-homogeneity scale invariances ($c\,\mathbf{W}_1$ and $c^{-1}\mathbf{W}_2$ compute the same ReLU network). Using filter-normalized slices, :citet:`Li.Xu.Taylor.ea.2018` showed that residual connections (:numref:`sec_resnet`) transform jagged, non-convex surfaces into smooth basins.

While linear interpolation between two independently trained networks crosses a high-loss barrier, :citet:`Garipov.Izmailov.Podoprikhin.ea.2018` and :citet:`Draxler.Veschgini.Salmhofer.ea.2018` discovered **mode connectivity**: simple piecewise-linear curves with a single bend connect independent minima along paths of uniformly low training and test loss. :citet:`Entezari.Sedghi.Saukh.ea.2022` and :citet:`Ainsworth.Hayase.Srinivasa.2023` demonstrated that after aligning hidden units via permutation symmetry (:numref:`sec_gd`), the linear barrier between wide networks nearly vanishes.

### Flat Minima and Sharpness-Aware Minimization (SAM)
:label:`subsec_optimization-fr-flat-minima`

Motivated by the hypothesis that flatter minima generalize better, **Sharpness-Aware Minimization (SAM)** :cite:`Foret.Kleiner.Mobahi.ea.2021` minimizes the worst-case loss inside an $\ell_2$ ball around the current weights by taking a gradient-ascent perturbation step before evaluating the descent gradient (at $2\times$ the gradient cost per update). Note, however, that raw Hessian sharpness is not invariant to layerwise reparametrization :cite:`Dinh.Pascanu.Bengio.ea.2017`.

## Classical Second-Order and Variance-Reduced Methods

### Quasi-Newton, Line Search, and Trust-Region Methods
:label:`subsec_optimization-fr-quasi-newton`

**Quasi-Newton methods** approximate the inverse Hessian $\mathbf{H}^{-1}$ from successive secant pairs $(\mathbf{x}_{t+1} - \mathbf{x}_t, \mathbf{g}_{t+1} - \mathbf{g}_t)$ using only first-order gradients. Limited-memory BFGS (**L-BFGS**, :citet:`Liu.Nocedal.1989`) stores the $m$ most recent pairs in $O(md)$ memory and is the method of choice for deterministic smooth problems (:numref:`subsec_mdl-quasi-newton`), though it is sensitive to minibatch gradient differences. **Line search** (choosing $\eta_t$ dynamically via Armijo sufficient-decrease checks) and **trust-region methods** (:numref:`subsec_mdl-trust-region`; see :citet:`Kovalev.2025` for a trust-region analysis of Muon) similarly excel in deterministic optimization.

### Hessian-Free and Saddle-Free Newton Methods
:label:`subsec_optimization-fr-hessian-free`

To apply second-order methods to deep networks without forming the $d \times d$ Hessian, **Hessian-free optimization** :cite:`Martens.2010` solves $\mathbf{H}\mathbf{d} = -\mathbf{g}$ approximately via conjugate gradients using Hessian–vector products (:numref:`subsec_mdl-hvp`) on the positive-semidefinite Gauss–Newton matrix, while **Saddle-Free Newton** :cite:`Dauphin.Pascanu.Gulcehre.ea.2014` preconditions via $|\mathbf{H}|^{-1}$ on a low-dimensional Krylov subspace so that negative-curvature directions repel rather than attract iterates.

### Stochastic Convergence Rates and Variance Reduction

For general convex objectives, minibatch SGD achieves the minimax-optimal $O(1/\sqrt{T})$ rate :cite:`Nemirovski.Yudin.1983` (:numref:`subsec_mdl-fixed-budget-rates`), while on non-convex smooth objectives it guarantees $\min_{t \le T} \mathbb{E}[\|\nabla f(\mathbf{x}_t)\|^2] = O(1/\sqrt{T})$ :cite:`Ghadimi.Lan.2013` (:numref:`subsec_mdl-nonconvex-sgd`). Polyak–Ruppert iterate averaging :cite:`Polyak.Juditsky.1992` achieves optimal asymptotic covariance (:numref:`subsec_practice-weight-averaging`). On finite datasets, **SVRG** :cite:`Johnson.Zhang.2013` (:numref:`sec_mdl-variance-reduction`) eliminates the stationary noise ball via periodic full-dataset control variates, though data augmentation and rapidly shifting features limit its effectiveness in deep learning.

## Matrix and Structural Optimizers

* **Natural Gradient, K-FAC, and SOAP:** The **natural gradient** :cite:`Amari.1998` preconditions updates by the inverse Fisher information matrix to measure step length in output-distribution space. **K-FAC** :cite:`Martens.Grosse.2015` approximates layerwise Fisher blocks as Kronecker products of input covariances and output-gradient covariances (:numref:`subsec_mdl-preconditioning-ladder`), while **SOAP** :cite:`Vyas.Morwani.Zhao.ea.2024` runs Adam in the eigenbasis of Shampoo's left and right preconditioners :cite:`Gupta.Koren.Singer.2018`.
* **Understanding and Scaling Muon:** :citet:`Wang.Zhang.Li.ea.2025` trace Muon's speedup over AdamW to associative-memory projections (attention value/output and MLP matrices) on tail features, mirroring Adam's advantage over SGD on rare tokens :cite:`Kunstner.Chen.Lavington.ea.2023,Kunstner.Yadav.Milligan.ea.2024`, while :citet:`Shah.Polloreno.Stratos.ea.2025` find that Muon remains data-efficient at larger batch sizes than AdamW. **Scion** :cite:`Pethick.Xie.Antonakopoulos.ea.2025` extends operator-norm steepest descent to every layer (including embeddings and readout heads) without falling back to AdamW.
* **Efficient and Distributed Orthogonalization:** **Polar Express** :cite:`Amsel.Persson.Musco.ea.2026` and :citet:`Cesista.You.Jordan.2025` optimize per-step Newton–Schulz polynomial coefficients (:numref:`fig_opt_ns_maps`), :citet:`Kim.Oh.2026` analyze convergence under inexact $q$-step orthogonalization, and **Dion** :cite:`Ahn.Xu.Abreu.ea.2025` orthogonalizes low-rank momentum approximations to reduce cross-device communication in sharded training.

## Schedules, Averaging, and Large-Scale Benchmarks

* **Warm Restarts, Schedule-Free Optimization, and Model Soups:** Cosine annealing was originally introduced with periodic warm restarts (**SGDR**, :citet:`Loshchilov.Hutter.2016`), enabling **snapshot ensembles** :cite:`Huang.Li.Pleiss.ea.2017` that average predictions across cycles, whereas **model soups** :cite:`Wortsman.Ilharco.Gadre.ea.2022` average the weights of models fine-tuned from a shared initialization. **Schedule-Free SGD and AdamW** :cite:`Defazio.Yang.Mehta.ea.2024` eliminate predetermined horizon schedules $T$ by evaluating gradients at an interpolation between the live iterate and the running iterate average.
* **Batch-Size Ramps and Scaling Laws:** :citet:`Malladi.Lyu.Panigrahi.ea.2022` derive full SDE scaling rules for Adam moments $(\beta_1, \beta_2, \epsilon)$ as batch size grows; :citet:`Zhang.Morwani.Vyas.ea.2024` and :citet:`Merrill.Arora.Groeneveld.ea.2025` show that the critical batch size grows primarily with tokens processed rather than model size; and :citet:`Dey.Gosal.Chen.ea.2023`, :citet:`Bi.Chen.Chen.ea.2024`, and :citet:`Meta.2025` document hyperparameter transfer across scales in production language models.
* **Optimizer Benchmarks and Protocols:** Apparent gains from new optimizers (such as LARS/LAMB :cite:`You.Gitman.Ginsburg.2017` or NorMuon :cite:`Li.Liu.Liang.ea.2026`) often shrink or vanish once baseline nuisance hyperparameters are tuned with equal effort :cite:`Nado.Gilmer.Shallue.ea.2021,Schmidt.Schneider.Hennig.2021,Wen.Hall.Ma.ea.2025`. Standardized benchmarks such as **AlgoPerf** :cite:`Dahl.Schneider.Nado.ea.2023,Kasimbeg.Schneider.Eschenhagen.ea.2025`, language-model benchmarking suites :cite:`Semenov.Pagliardini.Jaggi.2025`, and the **NanoGPT speedrun** optimization track :cite:`Jordan.2024,Jordan.2026` enforce explicit tuning budgets and statistical significance thresholds. Finally, the public **OPT-175B training logbook** :cite:`zhang2022opt` provides an unfiltered day-by-day record of mid-run loss spikes, checkpoint rewinds, and hardware recovery at scale.

[Discussions](https://d2l.discourse.group/t/36851)
