# Further Reading
:label:`sec_mdl-linear-algebra-further-reading`

The chapter stops where linear algebra turns into numerical analysis,
probability, or current research. This section says what lies past that
point, in the order of the chapter's sections; where to learn the chapter's
own material is the business of the list in its introduction
(:numref:`chap_mdl-linear-algebra`). The entries are not of equal weight.
For deep learning, read random projections, iterative solvers, random
matrices and trainable depth, and sketching first, then the last two
entries, which lead back into the main text. Tensor decompositions, the QR
algorithm, the Kreiss theorem, and mixing times are for the curious.
Nothing else in the book depends on this section.

## Geometry and Linear Maps

### Other Inner Products: Kernel Methods

The nearest-mean rule of :numref:`subsec_mdl-nearest-mean` compares an image
with two class means through one dot product, so its decision boundary is a
hyperplane. It fails when a class is spread out or curved, as on t-shirts
against shirts. Kernel methods keep the rule and change the inner product:
$\mathbf{x}\cdot\mathbf{x}_i$ is replaced by a function
$k(\mathbf{x},\mathbf{x}_i)$ that is an inner product in some other,
possibly infinite-dimensional, space. The class mean becomes a mean in that
space, every training point votes with weight $k(\mathbf{x},\mathbf{x}_i)$,
and the boundary can bend. Choosing which training points to keep, and with
what weights, leads to the support vector machine.
:citet:`Scholkopf.Smola.2002` start their book from exactly this classifier
and develop the theory of kernels from it.

### Random Projections

:numref:`sec_mdl-geometry-linear-algebraic-ops` showed that random
directions in $\mathbb{R}^d$ are nearly orthogonal. Can this shrink the
dimension of a dataset without changing its geometry? The
Johnson--Lindenstrauss lemma :cite:`Johnson.Lindenstrauss.1984` says yes:
for $n$ points, a scaled random Gaussian map to $k$ of order
$\varepsilon^{-2}\log n$ dimensions preserves all $\binom{n}{2}$ pairwise
distances within a factor $1\pm\varepsilon$, with high probability. Unlike
principal component analysis (:numref:`subsec_mdl-pca`), the map does not
look at the data, and the target dimension does not depend on $d$, so the
lemma pays off when $d$ is far larger than $\varepsilon^{-2}\log n$.
:citet:`Dasgupta.Gupta.2003` give a short proof with explicit constants, and
:citet:`Vershynin.2018` places the lemma among the concentration results for
random vectors.

### Solving Linear Systems at Scale

:numref:`subsec_mdl-numerical-issues` solved systems by an LU factorization
and warned that the inverse of a large sparse matrix is dense.
:citet:`Trefethen.Bau.1997` explain why elimination with partial pivoting is
stable in practice although it is not in the worst case, and
:citet:`Higham.2002` gives the full rounding-error analysis. Solvers for
sparse matrices never form the inverse, and often not the LU factors either.
Sparse direct methods reorder the unknowns so that elimination creates few
new nonzeros. Iterative methods touch the matrix only through products
$\mathbf{A}\mathbf{x}$, as power iteration does
(:numref:`subsec_mdl-power-iteration`). For a symmetric positive definite
matrix, the conjugate gradient method of :citet:`Hestenes.Stiefel.1952`
shrinks the error after $k$ steps by at least the factor
$2\bigl((\sqrt\kappa-1)/(\sqrt\kappa+1)\bigr)^k$ in the norm defined by the
matrix, where $\kappa$ is the condition number of
:numref:`subsec_mdl-condition-number`. :citet:`Golub.Van-Loan.1996` treat
the iterative methods in detail.

### Contraction Order and Tensor Decompositions

The `einsum` of :numref:`subsec_mdl-einsum` fixes the result of a product of
several factors but not the order of the pairwise contractions. For
$n\times n$ matrices, $(\mathbf{A}\mathbf{B})\mathbf{x}$ costs about $n^3$
multiplications and $\mathbf{A}(\mathbf{B}\mathbf{x})$ costs $2n^2$. Finding
the best order for many factors is a search over parenthesizations, which
`np.einsum(..., optimize=True)` and the opt_einsum package
:cite:`Smith.Gray.2018` carry out. Low rank also extends beyond matrices. A
three-way tensor can be written as a sum of outer products
$\mathbf{a}_i\otimes\mathbf{b}_i\otimes\mathbf{c}_i$ (the CP decomposition),
as a small core tensor multiplied by a matrix along each mode (Tucker), or as
a chain of three-way cores (the tensor train of :citet:`Oseledets.2011`).
Eckart--Young has no analogue for tensors: computing the CP rank is NP-hard,
and a best approximation of a given CP rank need not exist. The survey of
:citet:`Kolda.Bader.2009` covers the CP and Tucker decompositions and their
algorithms.

## Eigenvalues and Spectra

### How Libraries Compute Eigenvalues

No formula in radicals gives the roots of a general polynomial of degree five
or more, so every eigensolver iterates, and none forms the characteristic
polynomial of :numref:`subsec_mdl-finding-eigenvalues`, whose roots are
highly sensitive to its coefficients. `eig` first reduces the matrix by
orthogonal similarity transformations to Hessenberg form, zero below the
first subdiagonal, at a cost of $O(n^3)$. The shifted QR algorithm then
factors $\mathbf{H}-\mu\mathbf{I}=\mathbf{Q}\mathbf{R}$, forms
$\mathbf{R}\mathbf{Q}+\mu\mathbf{I}$, and repeats. Each step is a similarity
transformation, so the eigenvalues stay the same, while the subdiagonal
entries go to zero and the diagonal converges to the spectrum (a complex
pair of a real matrix remains as a $2\times2$ block). For a
symmetric matrix the Hessenberg form is tridiagonal, which `eigh` exploits.
:citet:`Trefethen.Bau.1997` derive the algorithm from power iteration
applied to whole subspaces, and :citet:`Golub.Van-Loan.1996` give the
production details.

### Pseudospectra and the Kreiss Matrix Theorem

:numref:`subsec_mdl-pseudospectra` showed that the spectrum of a non-normal
matrix does not predict its transient growth. The Kreiss matrix theorem makes
the pseudospectra quantitative. With the Kreiss constant
$K(\mathbf{A})=\sup_{|z|>1}(|z|-1)\|(z\mathbf{I}-\mathbf{A})^{-1}\|_2$, it
brackets the largest power,
$K(\mathbf{A})\le\sup_k\|\mathbf{A}^k\|_2\le e\,n\,K(\mathbf{A})$ for an
$n\times n$ matrix. The lower bound is an exercise of
:numref:`sec_mdl-eigendecompositions`; the upper bound says that a matrix
whose pseudospectra stay close to the unit disc cannot grow much.
:citet:`Trefethen.Embree.2005` develop the subject and its applications,
among them the transition to turbulence in fluid flows whose linearization
has only stable eigenvalues.

### Mixing Times and the Spectral Gap

The error of the two-page chain of :numref:`sec_mdl-eigendecompositions`
shrinks by a factor of 4 per click, because its second eigenvalue is $1/4$.
How long does a general chain take? For a reversible chain, let
$\lambda_\star$ be the largest modulus among the eigenvalues other than 1.
The number of steps needed to come within total variation distance
$\varepsilon$ of the stationary distribution is bounded above and below by
multiples of $1/(1-\lambda_\star)$, the multiples depending on
$\varepsilon$ and on the smallest stationary probability. For a lazy chain,
one that stays put with probability at least $1/2$, no eigenvalue is
negative and $\lambda_\star$ is the second eigenvalue $\lambda_2$.
Cheeger's inequality bounds the gap $1-\lambda_2$ from both sides by the
chain's *conductance*, the smallest probability flow out of a set relative
to the set's mass, so the gap can be read from the shape of the graph.
:citet:`Levin.Peres.2017` develop this theory, including the convergence
facts that :numref:`subsec_mdl-pagerank` cites from
:citet:`Horn.Johnson.2012`.

### Random Matrices and Trainable Depth

:numref:`subsec_mdl-jacobian-products` showed that the spectral radii of the
factors do not control a product of Jacobians. At initialization, what does?
:citet:`Saxe.McClelland.Ganguli.2014` solve the learning dynamics of deep
linear networks exactly and find that orthogonal initial weights make the
learning time independent of depth, where scaled Gaussian weights do not.
:citet:`Pennington.Schoenholz.Ganguli.2017` use free probability to compute
the distribution of all singular values of the input--output Jacobian of a
deep nonlinear network. *Dynamical isometry*, all these singular values near
1, is reachable with orthogonal weights and tanh units but not with ReLU, and
networks that have it train faster at large depth.

### Spectra of Trained Weights

The Marchenko--Pastur law of :numref:`subsec_mdl-sample-covariance-mp`
describes the spectrum of a matrix of independent noise, and the weights of
an untrained network with independent random initialization follow it.
:citet:`Martin.Mahoney.2021` compare the spectra of trained weight matrices
with this null model and find that training produces outliers and a heavy,
power-law tail. They read the exponent of that tail as a measure of implicit
self-regularization. The
departure from Marchenko--Pastur is observed across many models; using the
exponent to rank models by quality without test data rests on an empirical
correlation, not a theorem.

## Low Rank Beyond the Truncated SVD

### Matrix Completion and the Nuclear Norm

A ratings matrix is observed in a few percent of its entries, and
Eckart--Young (:numref:`subsec_mdl-eckart-young`) needs the whole matrix.
Matrix completion minimizes the *nuclear norm*
$\|\mathbf{Z}\|_*=\sum_i\sigma_i$, the convex stand-in for rank in the way
that $\ell_1$ stands in for sparsity, subject to agreeing with the observed
entries. :citet:`Candes.Recht.2009` prove that this convex program recovers
a low-rank matrix exactly, with high probability, from far fewer than all
$n^2$ entries chosen at random, provided its singular vectors are spread out
over the coordinates. Soft-Impute :cite:`Mazumder.Hastie.Tibshirani.2010`,
the algorithm of the matrix-completion exercise of
:numref:`sec_mdl-svd-low-rank`, solves the penalized form by alternating a
fill-in step with soft-thresholding of the singular values.

### CUR Decompositions and Sketching

The factors of the truncated SVD mix all rows and columns, so they are hard
to interpret, and a full SVD of an $m\times n$ matrix costs
$O(mn\min(m,n))$. A CUR decomposition keeps actual columns $\mathbf{C}$ and
rows $\mathbf{R}$ of the data and fits a small core $\mathbf{U}$ between
them. :numref:`subsec_mdl-eckart-young` showed the naive choice, the columns
of largest norm, losing to the truncation by a wide margin.
:citet:`Mahoney.Drineas.2009` sample columns and rows by their *leverage
scores*, the squared lengths of their components along the top singular
vectors, and come within a small constant factor of the Eckart--Young error
with high probability. The range finder of
:numref:`subsec_mdl-spectrum-control` :cite:`Halko.Martinsson.Tropp.2011` is
one instance of *sketching*: multiply by a random matrix and work with the
small result. :citet:`Woodruff.2014` applies the same idea to least squares
and regression, which are solved to within a factor $1+\varepsilon$ on a
sketch with far fewer rows than the data.

### Adapter Variants

LoRA (:numref:`subsec_mdl-low-rank-updates`) starts its update
$\mathbf{B}\mathbf{A}$ with $\mathbf{B}=\mathbf{0}$ and a random
$\mathbf{A}$, so the first updates see the input only through a random
$k$-dimensional subspace, unrelated to the pretrained weight. In PiSSA
:cite:`Meng.Wang.Zhang.2024`, $\mathbf{B}\mathbf{A}$ starts as the rank-$k$
truncated SVD of the pretrained weight and the residual is frozen, so
training adjusts the principal directions from the first step. DoRA
:cite:`Liu.Wang.Yin.ea.2024` writes each column of the weight as a length
times a unit direction, trains the lengths directly, and applies the
low-rank update to the directions only.

### Low Rank in Attention

Attention over $n$ tokens forms an $n\times n$ matrix, and several methods
taught in the attention chapters replace it, or the cache behind it, by a
low-rank factorization. Linear attention (:numref:`sec_attention-at-scale`)
writes the scores as $\phi(\mathbf{q})\cdot\phi(\mathbf{k})$, so the matrix
has rank at most the feature dimension, and multi-head latent attention
(:numref:`sec_kv-cache`) :cite:`DeepSeek-AI.2024` caches one low-dimensional
latent per token. Linformer :cite:`Wang.Li.Khabsa.ea.2020` starts from the
observation that trained attention matrices have rapidly decaying singular
values and projects the keys and values along the sequence axis to a fixed
length.

### Steepest Descent under Other Norms

:numref:`subsec_mdl-orthogonalized-updates` identified the polar factor
$\mathbf{U}\mathbf{V}^\top$ as the steepest-descent direction when steps are
measured in the spectral norm. :citet:`Bernstein.Newhouse.2024` turn this
into a way of classifying optimizers. With their moving averages switched
off, Adam becomes sign descent, which is steepest descent under the
$\ell_\infty$ norm, and Shampoo becomes steepest descent under the spectral
norm. The norm, chosen per layer, is then the design decision, and the
update rule follows from it. The optimizer built on the spectral norm is
taught in :numref:`sec_muon`.

[Discussions](https://d2l.discourse.group/t/36856)
