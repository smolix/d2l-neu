# Further Reading
:label:`sec_mdl-linear-algebra-further-reading`

Every chapter has to stop somewhere, and this one stops right where linear
algebra begins to merge into numerical analysis, probability, and active
research. What follows is a guide to the territory just past that border,
arranged in the same order as the chapter's sections. (If instead you are
looking for books or lectures covering the chapter's own material, see the
reading list in :numref:`chap_mdl-linear-algebra`.)

Not all of these entries are equally urgent. If your main interest is deep
learning, start with random projections, iterative solvers, random matrices
and trainable depth, and sketching, then jump to the final two entries, which
connect directly back to the main text. Tensor decompositions, the QR
algorithm, the Kreiss theorem, and Markov mixing times are here for curious
readers who want to see how the ideas travel. Nothing elsewhere in the book
depends on this section.

## Geometry and Linear Maps

### Other Inner Products: Kernel Methods

The nearest-mean rule of :numref:`subsec_mdl-nearest-mean` classifies a test
image by comparing it with two class averages through a single dot product,
which forces the decision boundary to be a flat hyperplane. That simplicity
breaks down whenever a class is curved or split into clusters, as happens when
distinguishing t-shirts from shirts. Kernel methods keep the exact same
geometric rule and change only the inner product: we replace
$\mathbf{x}\cdot\mathbf{x}_i$ by a similarity function
$k(\mathbf{x},\mathbf{x}_i)$ that corresponds to a genuine inner product in
some other, possibly infinite-dimensional, feature space. The class average
becomes a mean in that hidden space, every training example votes with weight
$k(\mathbf{x},\mathbf{x}_i)$, and the resulting boundary can bend freely in the
original input space. Asking which training points actually need nonzero
weights leads directly to the support vector machine.
:citet:`Scholkopf.Smola.2002` open their book with this very classifier and
build the full theory of kernel methods from it.

### Random Projections

In :numref:`sec_mdl-geometry-linear-algebraic-ops`, we saw that random
directions in high-dimensional space $\mathbb{R}^d$ are almost orthogonal to
one another. Surprisingly, that geometric quirk lets us shrink the dimension of
a dataset without looking at the data at all. The Johnson--Lindenstrauss lemma
:cite:`Johnson.Lindenstrauss.1984` guarantees that for any collection of $n$
points, multiplying by a suitably scaled random Gaussian matrix into $k$
dimensions of order $\varepsilon^{-2}\log n$ preserves all $\binom{n}{2}$
pairwise distances within a factor of $1\pm\varepsilon$, with high probability.
Unlike principal component analysis (:numref:`subsec_mdl-pca`), a random
projection is completely data-independent, and its target dimension $k$ does
not even depend on the ambient dimension $d$. The trick therefore pays off most
when $d$ is vastly larger than $\varepsilon^{-2}\log n$.
:citet:`Dasgupta.Gupta.2003` give a remarkably short proof with explicit
constants, while :citet:`Vershynin.2018` situates the lemma inside the broader
theory of concentration for random vectors.

### Solving Linear Systems at Scale

In :numref:`subsec_mdl-numerical-issues`, we solved linear systems via LU
factorization and noted that the inverse of a large sparse matrix is typically
dense. Why does Gaussian elimination with partial pivoting work so reliably in
floating-point arithmetic, even though pathological worst-case matrices exist?
:citet:`Trefethen.Bau.1997` give an illuminating explanation of this gap
between worst-case and practical stability, and :citet:`Higham.2002` provides
the definitive rounding-error analysis.

At large scale, sparse solvers never form $\mathbf{A}^{-1}$, and often avoid
forming dense LU factors altogether. Sparse direct methods carefully reorder
the unknowns so that elimination introduces as few new nonzeros as possible.
Iterative methods go a step further and access the matrix solely through
matrix--vector products $\mathbf{A}\mathbf{x}$, much like power iteration
(:numref:`subsec_mdl-power-iteration`). When $\mathbf{A}$ is symmetric positive
definite, the conjugate gradient method of :citet:`Hestenes.Stiefel.1952`
shrinks the error after $k$ steps by at least a factor of
$2\bigl((\sqrt\kappa-1)/(\sqrt\kappa+1)\bigr)^k$ in the norm induced by
$\mathbf{A}$, where $\kappa$ is the condition number from
:numref:`subsec_mdl-condition-number`. :citet:`Golub.Van-Loan.1996` treat
conjugate gradient and its relatives in full detail.

### Contraction Order and Tensor Decompositions

The `einsum` notation of :numref:`subsec_mdl-einsum` specifies the mathematical
result of contracting several tensors, but it leaves open the order in which
pairwise contractions are carried out. That choice can make a dramatic
difference: for $n\times n$ matrices, computing $(\mathbf{A}\mathbf{B})\mathbf{x}$
takes roughly $n^3$ multiplications, whereas $\mathbf{A}(\mathbf{B}\mathbf{x})$
takes only $2n^2$. When many factors are involved, choosing the cheapest order
becomes a search over parenthesizations, which `np.einsum(..., optimize=True)`
and the `opt_einsum` package :cite:`Smith.Gray.2018` automate.

The idea of low-rank approximation also extends beyond matrices to higher-order
tensors, though with a sobering catch. A three-way tensor can be written as a
sum of rank-one outer products
$\mathbf{a}_i\otimes\mathbf{b}_i\otimes\mathbf{c}_i$ (the CP decomposition), as
a small core tensor multiplied by a factor matrix along each mode (the Tucker
decomposition), or as a chain of three-way cores (the tensor train of
:citet:`Oseledets.2011`). Unfortunately, nothing as clean as the Eckart--Young
theorem holds in general for tensors: determining the CP rank is NP-hard, and a
best approximation of a given CP rank may not even exist. The survey by
:citet:`Kolda.Bader.2009` is the standard guide to CP and Tucker decompositions
and their algorithms.

## Eigenvalues and Spectra

### Computing Eigenvalues in Practice

There is no formula in radicals for the roots of a general polynomial of degree
five or higher, so every practical eigensolver must iterate. Worse still, no
production library forms the characteristic polynomial of
:numref:`subsec_mdl-finding-eigenvalues`, because the roots of a polynomial can
be extraordinarily sensitive to tiny rounding errors in its coefficients.
Instead, `eig` first applies orthogonal similarity transformations to reduce
the matrix to Hessenberg form (zero everywhere below the first subdiagonal),
which takes $O(n^3)$ operations. It then runs the shifted QR algorithm: factor
$\mathbf{H}-\mu\mathbf{I}=\mathbf{Q}\mathbf{R}$, form
$\mathbf{R}\mathbf{Q}+\mu\mathbf{I}$, and repeat. Every step is a similarity
transformation, so the eigenvalues never change, while the subdiagonal entries
shrink toward zero and the diagonal reveals the spectrum (complex conjugate
pairs of a real matrix settle into $2\times 2$ diagonal blocks). For symmetric
matrices, the Hessenberg form is tridiagonal, and `eigh` exploits that
structure for major speedups. :citet:`Trefethen.Bau.1997` derive the QR
algorithm cleanly from power iteration on subspaces, and
:citet:`Golub.Van-Loan.1996` supply the production engineering details.

### Pseudospectra and the Kreiss Matrix Theorem

In :numref:`subsec_mdl-pseudospectra`, we saw that for a non-normal matrix,
eigenvalues inside the unit disc do not prevent large transient growth in
$\|\mathbf{A}^k\|_2$. The Kreiss matrix theorem turns the visual intuition of
pseudospectra into a quantitative bound. Defining the Kreiss constant as
$K(\mathbf{A})=\sup_{|z|>1}(|z|-1)\|(z\mathbf{I}-\mathbf{A})^{-1}\|_2$, the
theorem brackets the worst-case power norm for any $n\times n$ matrix:
$K(\mathbf{A})\le\sup_k\|\mathbf{A}^k\|_2\le e\,n\,K(\mathbf{A})$. You can
prove the lower bound in an exercise in :numref:`sec_mdl-spectra-in-practice`;
the upper bound guarantees that a matrix whose pseudospectra hug the unit disc
closely cannot experience unbounded transient growth.
:citet:`Trefethen.Embree.2005` develop this theory thoroughly, including its
classic application to fluid flows that transition to turbulence even when
every linearized eigenvalue is strictly stable.

### Mixing Times and the Spectral Gap

In the two-page Markov chain of :numref:`sec_mdl-eigendecompositions`, the
distance to the stationary distribution shrinks by a factor of 4 at every click
because the second eigenvalue is $1/4$. How many steps does a general Markov
chain need to mix? For a reversible chain, let $\lambda_\star$ be the largest
modulus among all eigenvalues other than $1$. The number of steps required to
come within total variation distance $\varepsilon$ of the stationary
distribution is bounded both above and below by multiples of
$1/(1-\lambda_\star)$, where the constants depend on $\varepsilon$ and on the
smallest stationary probability. For a lazy chain—one that stays at its current
state with probability at least $1/2$—all eigenvalues are nonnegative, so
$\lambda_\star$ is simply the second eigenvalue $\lambda_2$. Cheeger's
inequality then links the spectral gap $1-\lambda_2$ from both sides to the
chain's *conductance*, the smallest probability flow leaving any subset
relative to its stationary mass. In other words, you can read the convergence
rate directly from the bottlenecks of the graph. :citet:`Levin.Peres.2017`
present this theory with great clarity, alongside the convergence results that
:numref:`subsec_mdl-pagerank` cites from :citet:`Horn.Johnson.2012`.

### Random Matrices and Trainable Depth

In :numref:`subsec_mdl-jacobian-products`, we observed that the spectral radii
of individual layer Jacobians do not control the behavior of a long product of
Jacobians. What, then, determines whether a very deep network can be trained
from initialization? :citet:`Saxe.McClelland.Ganguli.2014` solve the learning
dynamics of deep linear networks exactly and show that orthogonal initial
weights keep the training time independent of depth, whereas scaled Gaussian
weights cause learning to slow down as depth grows. Extending the analysis to
nonlinear networks, :citet:`Pennington.Schoenholz.Ganguli.2017` use free
probability to compute the full singular value distribution of the
input--output Jacobian. They show that *dynamical isometry*, the regime in
which all singular values cluster tightly near $1$, is achievable with
orthogonal weights and $\tanh$ activations but not with ReLU, and that networks
initialized in this regime train substantially faster at large depth.

### Spectra of Trained Weights

The Marchenko--Pastur law of :numref:`subsec_mdl-sample-covariance-mp`
describes the singular value spectrum of a matrix of pure independent noise,
which is what the weight matrices of a freshly initialized network look like.
:citet:`Martin.Mahoney.2021` compare the spectra of trained neural network
weights against this random-matrix baseline and observe that training produces
both sharp outliers and a heavy, power-law tail. They interpret the tail
exponent as a signature of implicit self-regularization. While this departure
from Marchenko--Pastur appears consistently across many architectures, using
the tail exponent to rank model quality without evaluating on test data remains
an empirical correlation rather than a proven theorem.

## Low Rank Beyond the Truncated SVD

### Matrix Completion and the Nuclear Norm

A movie-ratings matrix might have only a few percent of its entries observed,
whereas the Eckart--Young theorem (:numref:`subsec_mdl-eckart-young`) requires
knowing the entire matrix up front. Matrix completion bridges this gap by
minimizing the *nuclear norm* $\|\mathbf{Z}\|_*=\sum_i\sigma_i$, which serves
as a convex stand-in for matrix rank in the same way that the $\ell_1$ norm
stands in for vector sparsity, subject to matching all observed entries.
:citet:`Candes.Recht.2009` prove that this convex optimization problem recovers
an underlying low-rank matrix exactly, with high probability, from far fewer
than $n^2$ randomly observed entries, provided its singular vectors are spread
out across the coordinates rather than concentrated on a few spikes.
Soft-Impute :cite:`Mazumder.Hastie.Tibshirani.2010`, the algorithm you
implement in the matrix-completion exercise of :numref:`sec_mdl-svd-low-rank`,
solves the penalized version of this problem by alternating a fill-in step with
soft-thresholding of the singular values.

### CUR Decompositions and Sketching

Although the truncated SVD is optimal in approximation error, its singular
vectors mix together all rows and columns, making them difficult to interpret,
and computing a full SVD of an $m\times n$ matrix costs $O(mn\min(m,n))$
operations. A CUR decomposition instead builds an approximation from actual
columns $\mathbf{C}$ and actual rows $\mathbf{R}$ of the data matrix, linked by
a small core matrix $\mathbf{U}$. In :numref:`subsec_mdl-eckart-young`, we saw
that the naive strategy of picking the columns with the largest norms loses
badly to the truncated SVD. :citet:`Mahoney.Drineas.2009` show how to fix this:
sampling columns and rows according to their *leverage scores*, the squared
lengths of their projections onto the top singular vectors, gets within a small
constant factor of the Eckart--Young error with high probability.

More broadly, the randomized range finder of
:numref:`subsec_mdl-spectrum-control` :cite:`Halko.Martinsson.Tropp.2011` is
one example of *sketching*: multiply a large matrix by a thin random matrix and
solve the problem on the much smaller sketch. :citet:`Woodruff.2014` surveys
how this same idea accelerates least squares and regression, returning a
solution within a $1+\varepsilon$ factor of optimal from a sketch with far
fewer rows than the original dataset.

### Adapter Variants

LoRA (:numref:`subsec_mdl-low-rank-updates`) initializes its low-rank update
$\mathbf{B}\mathbf{A}$ with $\mathbf{B}=\mathbf{0}$ and a random
$\mathbf{A}$, which means the earliest gradient steps see the input only
through a random $k$-dimensional subspace unrelated to the pretrained weight.
Several variants change this initialization or parameterization in simple ways.
PiSSA :cite:`Meng.Wang.Zhang.2024` initializes $\mathbf{B}\mathbf{A}$ to the
rank-$k$ truncated SVD of the pretrained weight and freezes the residual, so
fine-tuning immediately adapts the principal directions of the original matrix.
DoRA :cite:`Liu.Wang.Yin.ea.2024` decomposes each weight column into a scalar
magnitude and a unit direction, trains the magnitudes directly, and applies the
low-rank update only to the directions.

### Low Rank in Attention

Self-attention over a sequence of $n$ tokens forms an $n\times n$ score matrix,
and several efficient architectures in the attention chapters replace either
that matrix or its key--value cache with a low-rank factorization. Linear
attention (:numref:`sec_windowed-linear-attention`) expresses the attention
scores as feature dot products $\phi(\mathbf{q})\cdot\phi(\mathbf{k})$,
capping the rank of the score matrix at the feature dimension. Multi-head
latent attention (:numref:`sec_kv-cache`) :cite:`DeepSeek-AI.2024` compresses
the cache by storing a single low-dimensional latent vector per token instead
of separate keys and values for every head. Linformer
:cite:`Wang.Li.Khabsa.ea.2020` starts from the empirical observation that
trained attention matrices have rapidly decaying singular values, and projects
the keys and values along the sequence dimension down to a fixed length.

### Steepest Descent under Other Norms

In :numref:`subsec_mdl-orthogonalized-updates`, we computed the polar factor
$\mathbf{U}\mathbf{V}^\top$ of a gradient matrix. Geometrically, this matrix is
the steepest-descent direction when step sizes are measured in the spectral
norm rather than the Euclidean norm. :citet:`Bernstein.Newhouse.2024` turn that
observation into a unifying lens for deep-learning optimizers: with their
moving averages turned off, Adam reduces to sign descent (steepest descent
under the $\ell_\infty$ norm), while Shampoo reduces to steepest descent under
the spectral norm. From this viewpoint, choosing the right norm for each layer
is the core design decision, and the update rule follows automatically. The
spectral-norm optimizer built on this idea is developed in :numref:`sec_muon`.

[Discussions](https://d2l.discourse.group/t/36856)
