# Singular Value Decomposition and Low-Rank Approximation
:label:`sec_mdl-svd-low-rank`

The eigendecomposition of :numref:`sec_mdl-eigendecompositions` is powerful but
picky: it needs a square matrix, and to be fully well-behaved (an orthonormal
eigenbasis, real eigenvalues) it really wants a *symmetric* one. The *singular
value decomposition* (SVD) removes every one of those restrictions. It applies
to **every** matrix---rectangular, rank-deficient, non-symmetric---and factors it
into a clean rotate--scale--rotate form whose "scale" factors, the *singular
values*, are the right generalization of eigenvalues. This single tool
underwrites a remarkable share of practical deep learning: principal component
analysis, the Eckart--Young low-rank truncation that justifies LoRA and other
low-rank adapters, the condition number that predicts numerical trouble, and the
spectral diagnostics used to inspect trained weights. This section builds the SVD
from the eigendecomposition we already have, states the results a deep-learning
practitioner needs, and works PCA as the central example.

::: {.callout-important title="Status: section is a plan (outline only)"}
This file is a **detailed table of contents** for §1.3. Each subsection below is
a planned-outline stub, not finished prose. Subsections, key results, diagrams,
worked examples, and exercises are specified so the design can be reviewed before
any prose or code is written. Notation follows the chapter contract: $\Sigma$ is
reserved for singular values, $\boldsymbol{\Lambda}$ for eigenvalues
(:numref:`sec_mdl-eigendecompositions`).
:::

## The SVD: Rotate--Scale--Rotate

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
**Body framing:** Every matrix, viewed as a linear map, does the *same* three
things in sequence: rotate (or reflect) the input, stretch along orthogonal axes,
then rotate again. The SVD makes those three steps explicit and unifies the
"matrices skew/rotate/scale the grid" picture from
:numref:`sec_mdl-geometry-linear-algebraic-ops`.
**Outline:** 1. State $\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ for any $m\times n$ matrix: $\mathbf{U}$ ($m\times m$) and $\mathbf{V}$ ($n\times n$) orthogonal, $\boldsymbol{\Sigma}$ ($m\times n$) diagonal with $\sigma_1 \ge \sigma_2 \ge \cdots \ge 0$ · 2. Geometric reading: $\mathbf{V}^\top$ rotates input axes to align with the right singular vectors, $\boldsymbol{\Sigma}$ scales by $\sigma_i$, $\mathbf{U}$ rotates into the output frame · 3. Full vs. *thin/economy* SVD ($r = \operatorname{rank}\mathbf{A}$ terms) · 4. The dyadic sum $\mathbf{A} = \sum_{i=1}^{r}\sigma_i\,\mathbf{u}_i\mathbf{v}_i^\top$, setting up truncation in §1.3.4.
**Key results to state:** $\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$; $\mathbf{A}\mathbf{v}_i = \sigma_i\mathbf{u}_i$; $\mathbf{A} = \sum_i \sigma_i\mathbf{u}_i\mathbf{v}_i^\top$.
**Diagrams:** `fig_mdl-svd-rotate-scale-rotate` — the unit circle in $\mathbb{R}^2$ mapped by a rectangular/asymmetric $\mathbf{A}$ to an ellipse, shown as three labeled stages (rotate by $\mathbf{V}^\top$, scale by $\sigma_1,\sigma_2$, rotate by $\mathbf{U}$), with right singular vectors on the input and left singular vectors on the output.
**Worked example(s):** compute the `svd` of a small $3\times 2$ matrix and verify $\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top = \mathbf{A}$ numerically across the four frameworks.
**Exercises (draft):** (1) SVD of $\operatorname{diag}(3,1)$ by inspection; (2) for the rotation $\begin{bmatrix}0&-2\\1&0\end{bmatrix}$ show the singular values are $\{2,1\}$ even though $|\lambda|=\sqrt{2}$, illustrating $\sigma \neq |\lambda|$.
**Prereqs / cross-refs:** matrices as linear maps and the grid picture (:numref:`sec_mdl-geometry-linear-algebraic-ops`); the spectral theorem (:numref:`sec_mdl-eigendecompositions`).
:::

## SVD via the Eigendecomposition of $\mathbf{A}^\top\mathbf{A}$

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
**Body framing:** The SVD is not a new mystery---it is the spectral theorem
applied to the symmetric PSD matrix $\mathbf{A}^\top\mathbf{A}$. This both *proves
existence* (Gram matrices are always symmetric PSD, so the eigendecomposition we
already have never fails) and gives a recipe for computing it by hand.
**Outline:** 1. $\mathbf{A}^\top\mathbf{A}$ is symmetric PSD (callback to the PSD subsection of :numref:`sec_mdl-eigendecompositions`) · 2. Its orthonormal eigenvectors are the right singular vectors $\mathbf{v}_i$; eigenvalues are $\sigma_i^2$, so $\sigma_i = \sqrt{\lambda_i} \ge 0$ · 3. Left singular vectors $\mathbf{u}_i = \mathbf{A}\mathbf{v}_i/\sigma_i$ (equivalently eigenvectors of $\mathbf{A}\mathbf{A}^\top$) · 4. Why this *always* works where plain eigendecomposition can fail (the defective-matrix example from the previous section).
**Key results to state:** $\mathbf{A}^\top\mathbf{A} = \mathbf{V}\boldsymbol{\Sigma}^\top\boldsymbol{\Sigma}\mathbf{V}^\top$; $\sigma_i = \sqrt{\lambda_i(\mathbf{A}^\top\mathbf{A})}$; $\mathbf{u}_i = \mathbf{A}\mathbf{v}_i/\sigma_i$.
**Diagrams:** reuses `fig_mdl-svd-rotate-scale-rotate`; optionally annotate $\mathbf{V}$ as the eigenbasis of $\mathbf{A}^\top\mathbf{A}$.
**Worked example(s):** for the $3\times 2$ matrix above, eigendecompose $\mathbf{A}^\top\mathbf{A}$ by hand and confirm $\sigma_i^2$ match the squared singular values from the library `svd`.
**Exercises (draft):** (1) show every singular value is $\ge 0$ while eigenvalues of a general matrix can be negative or complex; (2) prove $\mathbf{A}$ and $\mathbf{A}^\top$ share the same nonzero singular values.
**Prereqs / cross-refs:** PSD and the spectral theorem (:numref:`sec_mdl-eigendecompositions`); the defective-matrix "escape hatch" promised there.
:::

## Rank, Range, and Null Space from the Spectrum

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
**Body framing:** The SVD reads off the rank and the four fundamental subspaces
directly from the singular values---and it gives a *numerically honest* notion of
rank, which the exact, exponential-cost definition from
:numref:`sec_mdl-geometry-linear-algebraic-ops` could not.
**Outline:** 1. $\operatorname{rank}\mathbf{A} = $ number of nonzero $\sigma_i$ (resolves the "more efficient way to compute rank" promise) · 2. Orthonormal bases for the four subspaces: range from $\{\mathbf{u}_i: \sigma_i>0\}$, null space from $\{\mathbf{v}_i: \sigma_i=0\}$, and the cousins for $\mathbf{A}^\top$ · 3. *Numerical rank*: count $\sigma_i$ above a tolerance, since floating-point noise makes exact zeros rare · 4. Why this matters: effective dimensionality of features/weights.
**Key results to state:** $\operatorname{rank}\mathbf{A} = |\{i:\sigma_i>0\}|$; $\operatorname{range}\mathbf{A} = \operatorname{span}\{\mathbf{u}_i:\sigma_i>0\}$; $\operatorname{null}\mathbf{A} = \operatorname{span}\{\mathbf{v}_i:\sigma_i=0\}$.
**Diagrams:** none new (small table of the four subspaces vs. singular vectors).
**Worked example(s):** build a deliberately rank-2 $4\times 4$ matrix, show two singular values are $\sim 10^{-16}$ (numerical zero), and recover the rank with a tolerance.
**Exercises (draft):** (1) given a list of singular values, state rank and nullity; (2) explain why thresholding $\sigma_i$ is more reliable than testing for exact zeros.
**Prereqs / cross-refs:** rank and linear dependence (:numref:`sec_mdl-geometry-linear-algebraic-ops`); §1.3.1 (singular vectors).
:::

## Eckart--Young: Optimal Low-Rank Approximation

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
**Body framing:** This is the theorem that makes the SVD indispensable: keeping
only the top $k$ singular triples gives the *provably best* rank-$k$ approximation
of a matrix, in both the spectral and Frobenius norms. It is the mathematical
guarantee behind PCA compression and LoRA.
**Outline:** 1. Define the truncation $\mathbf{A}_k = \sum_{i=1}^{k}\sigma_i\mathbf{u}_i\mathbf{v}_i^\top$ · 2. State the Eckart--Young--Mirsky theorem: $\mathbf{A}_k$ minimizes $\|\mathbf{A}-\mathbf{B}\|$ over all $\operatorname{rank}\mathbf{B}\le k$ · 3. Exact error: $\|\mathbf{A}-\mathbf{A}_k\|_2 = \sigma_{k+1}$ and $\|\mathbf{A}-\mathbf{A}_k\|_F^2 = \sum_{i>k}\sigma_i^2$ · 4. The *energy ratio* $\sum_{i\le k}\sigma_i^2 / \sum_i \sigma_i^2$ as a "how much did we keep" dial.
**Key results to state:** $\min_{\operatorname{rank}\mathbf{B}\le k}\|\mathbf{A}-\mathbf{B}\|_2 = \sigma_{k+1}$, attained at $\mathbf{A}_k$; $\|\mathbf{A}-\mathbf{A}_k\|_F^2 = \sum_{i>k}\sigma_i^2$.
**Diagrams:** `fig_mdl-eckart-young-truncation` — left: a bar chart of the singular-value spectrum with the kept/discarded split highlighted; right: an image reconstructed at increasing $k$ (e.g. $k=1,5,20,$ full), showing perceptual quality rising as discarded energy shrinks.
**Worked example(s):** take an image as a matrix, plot its singular-value spectrum, reconstruct at several $k$, and verify the measured reconstruction error equals $\sigma_{k+1}$ (spectral) / $\sqrt{\sum_{i>k}\sigma_i^2}$ (Frobenius).
**Exercises (draft):** (1) prove the spectral-norm error $\|\mathbf{A}-\mathbf{A}_k\|_2=\sigma_{k+1}$; (2) find the smallest $k$ capturing 95% of the Frobenius energy for a given spectrum.
**Prereqs / cross-refs:** §1.3.1 (dyadic form); feeds PCA (§1.3.5) and modern DL (§1.3.8); cite Eckart--Young (1936) and Mirsky (1960).
:::

## PCA as the Worked Example

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
**Body framing:** Principal component analysis is exactly Eckart--Young applied to
centered data: the principal directions are the right singular vectors of the
centered data matrix, equivalently the eigenvectors of the covariance, and the
top-$k$ projection is the optimal linear dimensionality reduction.
**Outline:** 1. Center the data $\tilde{\mathbf{X}} = \mathbf{X} - \bar{\mathbf{x}}$ · 2. Principal directions = right singular vectors $\mathbf{v}_i$ of $\tilde{\mathbf{X}}$ = eigenvectors of the covariance $\tfrac{1}{n}\tilde{\mathbf{X}}^\top\tilde{\mathbf{X}}$ (callback to PSD covariance) · 3. Variance explained by component $i$ is $\sigma_i^2/n$; ranked variance = the scree plot · 4. Top-$k$ projection is the best rank-$k$ linear reduction by Eckart--Young · 5. The whitening transform as a sequel.
**Key results to state:** principal axes $= \mathbf{v}_i$; explained variance $= \sigma_i^2/n$; PCA projection $\mathbf{z} = \mathbf{V}_k^\top(\mathbf{x}-\bar{\mathbf{x}})$ minimizes reconstruction MSE among rank-$k$ linear maps.
**Diagrams:** reuses `fig_mdl-eckart-young-truncation` for the scree plot; a 2-D scatter with the two principal axes drawn as arrows scaled by $\sigma_i$.
**Worked example(s):** tiny PCA on a 2-D correlated point cloud---center, `svd`, plot principal axes, project to 1-D, and report variance explained; cross-check $\mathbf{v}_i$ against `eig` of the covariance.
**Exercises (draft):** (1) show PCA directions are eigenvectors of the covariance; (2) relate explained-variance ratio to the energy ratio of §1.3.4; (3) why centering matters (what the first singular vector captures if you skip it).
**Prereqs / cross-refs:** covariance as symmetric PSD (:numref:`sec_mdl-eigendecompositions`); Eckart--Young (§1.3.4); forward-ref to the statistics/dimensionality material in later chapters.
:::

## Pseudoinverse and Least Squares

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
**Body framing:** When $\mathbf{A}\mathbf{x}=\mathbf{b}$ has no solution (too many
equations) or infinitely many (too few), the SVD delivers the principled answer:
the Moore--Penrose pseudoinverse gives the minimum-norm least-squares solution,
and it is far more numerically stable than forming the normal equations.
**Outline:** 1. Define $\mathbf{A}^{+} = \mathbf{V}\boldsymbol{\Sigma}^{+}\mathbf{U}^\top$, inverting only the nonzero $\sigma_i$ · 2. $\hat{\mathbf{x}} = \mathbf{A}^{+}\mathbf{b}$ minimizes $\|\mathbf{A}\mathbf{x}-\mathbf{b}\|_2$, and among minimizers has the smallest norm · 3. Contrast with the normal equations $\mathbf{A}^\top\mathbf{A}\mathbf{x}=\mathbf{A}^\top\mathbf{b}$, which *square* the condition number (forward-ref §1.3.7) · 4. Truncated pseudoinverse as regularization (drop tiny $\sigma_i$).
**Key results to state:** $\mathbf{A}^{+} = \mathbf{V}\boldsymbol{\Sigma}^{+}\mathbf{U}^\top$ with $\Sigma^{+}_{ii}=1/\sigma_i$ for $\sigma_i>0$, else $0$; $\hat{\mathbf{x}}=\mathbf{A}^{+}\mathbf{b}$ is the min-norm least-squares solution.
**Diagrams:** none new.
**Worked example(s):** solve an overdetermined least-squares problem two ways---via `pinv` and via the library `lstsq`---and compare against the (worse-conditioned) normal-equations solution; print the residuals.
**Exercises (draft):** (1) prove $\mathbf{A}^{+}\mathbf{b}$ is the minimum-norm least-squares solution; (2) show $\mathbf{A}^{+}=\mathbf{A}^{-1}$ when $\mathbf{A}$ is square invertible; (3) why truncating small $\sigma_i$ stabilizes the solution.
**Prereqs / cross-refs:** invertibility and "avoid forming $\mathbf{A}^{-1}$" (:numref:`sec_mdl-geometry-linear-algebraic-ops`); condition number (§1.3.7); cite :cite:`Golub.Van-Loan.1996`.
:::

## Condition Number and Numerical Stability

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
**Body framing:** The *condition number* $\kappa = \sigma_1/\sigma_r$ is the one
number that predicts numerical pain: how much input error a linear solve can
amplify, and---foreshadowing the optimization chapter---how badly gradient descent
will zig-zag. It is the SVD's most practical single scalar.
**Outline:** 1. Define $\kappa(\mathbf{A}) = \sigma_{\max}/\sigma_{\min}$ (over nonzero singular values) · 2. Error amplification: relative output error bounded by $\kappa \times$ relative input error · 3. Geometric picture: a large $\kappa$ means very elongated level sets · 4. Forward-ref: the same $\kappa$ sets gradient descent's contraction rate, so a quadratic with ratio $\kappa$ between curvatures converges slowly along the flat axis.
**Key results to state:** $\kappa = \sigma_1/\sigma_r$; relative error $\lesssim \kappa \cdot$ (relative perturbation); a well-conditioned $\kappa\approx 1$ vs. ill-conditioned $\kappa\gg 1$.
**Diagrams:** `fig_mdl-condition-number-contours` — side-by-side quadratic-bowl contour plots for a well-conditioned ($\kappa\approx 1$, near-circular contours) and an ill-conditioned ($\kappa\gg 1$, elongated contours) matrix, with a gradient-descent trajectory zig-zagging across the narrow valley in the latter.
**Worked example(s):** build matrices with prescribed $\sigma_1,\sigma_r$, compute $\kappa$ via `cond`, perturb $\mathbf{b}$ and measure how the solution moves vs. the $\kappa$ bound.
**Exercises (draft):** (1) show $\kappa$ of an orthogonal matrix is $1$; (2) relate $\kappa(\mathbf{A}^\top\mathbf{A}) = \kappa(\mathbf{A})^2$ and explain why the normal equations are worse; (3) sketch why $\kappa$ predicts GD step-count.
**Prereqs / cross-refs:** §1.3.6 (least squares); forward-ref to gradient descent and numerical-stability sections in later chapters.
:::

## SVD in Modern Deep Learning

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
**Body framing:** Close the loop by showing where the SVD actually shows up in
2020s deep learning: low-rank adapters (LoRA), weight-spectrum diagnostics, and
the near-low-rank structure of attention.
**Outline:** 1. *LoRA*: freeze a pretrained weight $\mathbf{W}$ and learn a low-rank update $\Delta\mathbf{W} = \mathbf{B}\mathbf{A}$ with rank $r \ll \min(m,n)$, trading $mn$ parameters for $r(m+n)$; Eckart--Young says rank $r$ is the most-expressive cheap update · 2. *Weight-spectrum diagnostics*: plotting trained-layer singular values to read effective rank and detect heavy-tailed spectra · 3. *Attention rank*: empirical near-low-rank structure of attention matrices and linear-attention approximations · 4. Pointer to randomized SVD for large matrices.
**Key results to state:** parameter count $r(m+n)$ vs. $mn$; LoRA as a learned rank-$r$ correction; "effective rank" via the singular-value energy ratio.
**Diagrams:** reuses `fig_mdl-eckart-young-truncation` for the weight-spectrum view; a small schematic of $\mathbf{W} + \mathbf{B}\mathbf{A}$.
**Worked example(s):** take a trained (or synthetic) weight matrix, plot its singular-value spectrum, and report the rank capturing 95% of spectral energy plus the parameter saving a LoRA of that rank would give.
**Exercises (draft):** (1) compute the LoRA rank needed for 95% spectral energy of a given matrix and the resulting parameter saving; (2) show that adding a rank-$r$ update can change at most $r$ singular values' worth of behavior; (3) estimate FLOP/parameter savings of LoRA vs. full fine-tuning for a given layer shape.
**Prereqs / cross-refs:** Eckart--Young (§1.3.4); main-book attention and parameter-efficient fine-tuning chapters; cite the LoRA paper (Hu et al., 2021) and randomized-SVD references (Halko, Martinsson, Tropp, 2011).
:::

## Summary

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
**Planned bullets:** Every matrix factors as $\mathbf{A}=\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ (rotate--scale--rotate). · Singular values are $\sqrt{\text{eigenvalues of }\mathbf{A}^\top\mathbf{A}}$; the SVD always exists because Gram matrices are symmetric PSD. · Rank, range, and null space read off the spectrum; numerical rank thresholds the $\sigma_i$. · Eckart--Young: top-$k$ truncation is the optimal rank-$k$ approximation, error $\sigma_{k+1}$. · PCA *is* Eckart--Young on centered data. · The pseudoinverse gives min-norm least squares; $\kappa=\sigma_1/\sigma_r$ predicts both numerical error and GD speed. · SVD powers PCA, LoRA, and weight/attention spectral analysis.
:::

## Exercises

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
**Draft exercises (section-level):**
1. Compute the SVD of $\operatorname{diag}(3,1)$ by inspection; then show the rotation $\begin{bmatrix}0&-2\\1&0\end{bmatrix}$ has singular values $\{2,1\}$ even though its eigenvalue magnitudes are $\sqrt 2$ --- i.e. $\sigma \neq |\lambda|$ for non-symmetric matrices.
2. Prove the Eckart--Young spectral-norm error $\|\mathbf{A}-\mathbf{A}_k\|_2 = \sigma_{k+1}$.
3. Prove $\mathbf{A}^{+}\mathbf{b}$ is the minimum-norm least-squares solution of $\mathbf{A}\mathbf{x}=\mathbf{b}$.
4. For a given weight matrix, find the LoRA rank achieving 95% spectral energy and compute the resulting parameter saving relative to a full update.
:::

## Discussion

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
**Plan:** A short closing pointer plus per-framework Discussions blocks
(one each for mxnet, pytorch, tensorflow, jax) following the chapter
convention, to be filled in when the section is authored.
:::

<!-- slides -->

::: {.callout-note title="⟢ Planned — slides (outline only, not yet written)"}
**Planned slide deck (mirrors the eigendecomposition deck style):**

- *The SVD picture* — $\mathbf{A}=\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ as rotate--scale--rotate; `fig_mdl-svd-rotate-scale-rotate`; reference the §1.3.1 code cell.
- *Where the singular values come from* — $\sigma_i=\sqrt{\lambda_i(\mathbf{A}^\top\mathbf{A})}$; the §1.3.2 verification cell.
- *Eckart--Young / low-rank* — `fig_mdl-eckart-young-truncation`; the image-reconstruction cell.
- *PCA* — principal axes from the SVD; the 2-D PCA cell.
- *Least squares & conditioning* — pseudoinverse, $\kappa=\sigma_1/\sigma_r$; `fig_mdl-condition-number-contours`; the `pinv`/`lstsq`/`cond` cell.
- *SVD in modern DL* — LoRA and weight spectra; the weight-spectrum cell.
- *Recap* — every matrix factors; truncation is optimal; $\kappa$ is the one number to watch.

(Code cells will carry stable `#<id>` tags and `%%tab` structure once written;
slide outputs will come from `tools/inject_outputs.py slides`.)
:::
