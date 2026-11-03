# Eigendecompositions
:label:`sec_mdl-eigendecompositions`

In :numref:`sec_mdl-linear-maps` we asked what a matrix does to space when it
is applied once. Many computations apply the same matrix again and again. A
deep linear network with equal layers computes $\mathbf A^{50}\mathbf x$
after 50 layers, and gradient descent on a quadratic multiplies its error by
one fixed matrix per step. In each case we want to know whether the signal
grows, dies out, or settles, and how quickly. Multiplying out the powers gives
the numbers but does not explain them.

A random walk between two web pages is the smallest instance of the question.
From page 1 a surfer clicks to page 1 or to page 2 with probability $1/2$
each; from page 2 she stays with probability $3/4$ and returns to page 1 with
probability $1/4$. Let $\mathbf p_k$ hold the probabilities of being on each
page after $k$ clicks. One click multiplies it by a fixed matrix,

$$
\mathbf p_{k+1}=\mathbf P\mathbf p_k,
\qquad
\mathbf P=\begin{pmatrix}1/2&1/4\\1/2&3/4\end{pmatrix},
$$

whose columns are the click probabilities from each page, so each column sums
to one. Starting on page 1, $\mathbf p_0=(1,0)$, the distribution reads
$(0.5,0.5)$, $(0.375,0.625)$, $(0.344,0.656)$, $(0.336,0.664)$. The changes
shrink from click to click, as if the walk were settling on a fixed
distribution. Where is the surfer after 50 clicks, and how fast does the
answer settle? PageRank asks the same question of the whole web.

The answer is to write the starting vector in the coordinates of a few
special directions, the eigenvectors, along which the matrix acts as
multiplication by a number. Each coordinate then evolves by its own scalar
power $\lambda^k$.

In :numref:`sec_linear-algebra` we met eigenvectors as the directions that a
matrix only stretches: the symmetric
$\mathbf S=\left(\begin{smallmatrix}2&1\\1&2\end{smallmatrix}\right)$
stretches $(1,1)$ by 3 and $(1,-1)$ by 1, and for a symmetric matrix
$\mathbf A^k\mathbf x=\sum_i c_i\lambda_i^k\mathbf v_i$. This section drops
the symmetry. We find the eigenvalues of any square matrix and see when a full
set of eigenvectors exists, and what replaces it when it does not. We then
prove what symmetry guarantees, locate and compute eigenvalues without solving
polynomials, and study matrices whose eigenvalues mislead. We compute with
NumPy throughout; its `eig` returns complex eigenvalues when a real matrix has
them.

```{.python .input #eigendecomposition-imports}
#@tab mxnet
import numpy as np
```

```{.python .input #eigendecomposition-imports}
#@tab pytorch
import numpy as np
```

```{.python .input #eigendecomposition-imports}
#@tab tensorflow
import numpy as np
```

```{.python .input #eigendecomposition-imports}
#@tab jax
import numpy as np
```

## Eigenvalues and Diagonalization

### Eigenvectors and Eigenvalues
:label:`subsec_mdl-eigen-definition`

Some inputs are only rescaled by $\mathbf P$, as if it were a number. The
vector $(1,2)$ is mapped to
$\mathbf P(1,2)=(\tfrac12+\tfrac12,\ \tfrac12+\tfrac32)=(1,2)$, and $(1,-1)$
is mapped to $(\tfrac12-\tfrac14,\ \tfrac12-\tfrac34)=\tfrac14(1,-1)$. Along
the first direction $\mathbf P$ multiplies by 1, along the second by $1/4$.

In general, a nonzero vector $\mathbf v$ is an *eigenvector* of a square
matrix $\mathbf A$, with *eigenvalue* $\lambda$, if

$$
\mathbf{A}\mathbf{v} = \lambda \mathbf{v}.
$$
:eqlabel:`eq_mdl-eigpair`

The matrix only stretches an eigenvector by the factor $\lambda$, and reverses
it when $\lambda<0$. For $\mathbf P$, the pairs are $(1,(1,2))$ and
$(\tfrac14,(1,-1))$. The zero vector is excluded as an eigenvector because it
satisfies :eqref:`eq_mdl-eigpair` for every $\lambda$. All solutions of
$\mathbf A\mathbf v=\lambda\mathbf v$ for one $\lambda$, together with
$\mathbf 0$, form a subspace, the *eigenspace* of $\lambda$.

The eigenvalue $\lambda=0$ is allowed. It means
$\mathbf A\mathbf v=\mathbf 0$ for some $\mathbf v\neq\mathbf 0$, so a square
matrix is singular if and only if 0 is one of its eigenvalues. Not every real
matrix has a real eigenvector: a rotation by $90^\circ$ turns every direction.
Its eigenvalues are complex, and we return to them at the end of this part.

A $2\times2$ invertible matrix maps the unit circle to an ellipse. Are the
axes of that ellipse the eigenvectors? :numref:`fig_mdl-la-eig-ellipse`
compares two matrices. For the symmetric $\mathbf S$ the eigenvectors lie on
the axes. For the upper-triangular
$\left(\begin{smallmatrix}1&1\\0&2\end{smallmatrix}\right)$ they do not: its
eigenvectors are $(1,0)$ with $\lambda=1$ and $(1,1)$ with $\lambda=2$, while
the ellipse axes point at $58^\circ$ and $-32^\circ$. The ellipse axes are the
directions of largest and smallest stretch; the eigenvectors are the
directions that are not turned. For symmetric matrices the two coincide, as
the spectral theorem of :numref:`subsec_mdl-spectral-theorem` proves. For
other matrices they need not.

![The unit circle (dashed) and its image (blue). Left: the symmetric $\left(\begin{smallmatrix}2&1\\1&2\end{smallmatrix}\right)$; its unit eigenvectors (green) are the ellipse axes, and their images (red) are scaled by 3 and 1 (for $\lambda=1$ the two arrows coincide). Right: the upper-triangular $\left(\begin{smallmatrix}1&1\\0&2\end{smallmatrix}\right)$; its eigenvectors $(1,0)$ and $(1,1)$ are not the ellipse axes (grey).](../img/mdl-la-eig-ellipse.svg)
:label:`fig_mdl-la-eig-ellipse`

### The Characteristic Polynomial
:label:`subsec_mdl-finding-eigenvalues`

We found the eigenvectors of $\mathbf P$ by inspection. To find the
eigenvalues of any matrix, move $\lambda\mathbf v$ to the left side of
:eqref:`eq_mdl-eigpair` and factor out the vector:

$$
(\mathbf{A} - \lambda \mathbf{I})\mathbf{v} = \mathbf 0.
$$
:eqlabel:`eq_mdl-eigvalue_der`

A nonzero solution $\mathbf v$ exists if and only if
$\mathbf A-\lambda\mathbf I$ has a nontrivial null space, that is, if and only
if it is singular and $\det(\mathbf A-\lambda\mathbf I)=0$. Expanding the
determinant gives a polynomial of degree $n$ in $\lambda$, the
*characteristic polynomial* $p(\lambda)=\det(\mathbf A-\lambda\mathbf I)$.
Its roots are the eigenvalues. For each root, solving
:eqref:`eq_mdl-eigvalue_der` gives the eigenspace.

For the surfer's matrix,

$$
\det(\mathbf P-\lambda\mathbf I)
=\Bigl(\tfrac12-\lambda\Bigr)\Bigl(\tfrac34-\lambda\Bigr)-\tfrac14\cdot\tfrac12
=\lambda^2-\tfrac54\lambda+\tfrac14
=(\lambda-1)\Bigl(\lambda-\tfrac14\Bigr),
$$

so the eigenvalues are 1 and $1/4$, the two factors we found by inspection.
The eigenvectors follow from one row of each singular system. For $\lambda=1$
the first row of $\mathbf P-\mathbf I$ reads $-\tfrac12v_1+\tfrac14v_2=0$, so
$v_2=2v_1$ and $\mathbf v=(1,2)$. For $\lambda=\tfrac14$ the first row of
$\mathbf P-\tfrac14\mathbf I$ reads $\tfrac14v_1+\tfrac14v_2=0$, so
$\mathbf v=(1,-1)$. (Hand computations are easier on $4\mathbf P$, which has
integer entries, the same eigenvectors, and eigenvalues 4 and 1.) Each
eigenvector is determined only up to a nonzero factor: if $\mathbf v$ is an
eigenvector, so is $c\mathbf v$ for any $c\neq0$, since both sides of
:eqref:`eq_mdl-eigpair` scale by $c$.

Does the library agree? The cell computes the eigenpairs of $\mathbf P$ and
checks the defining equation for all of them at once.

```{.python .input #eigendecomposition-an-example}
P = np.array([[0.5, 0.25], [0.5, 0.75]])
lam, V = np.linalg.eig(P)
print('eigenvalues:', lam)
print('eigenvectors (columns):\n', V)
print('max |P V - V diag(lam)|:', np.abs(P @ V - V * lam).max())
```

The eigenvalues are $1/4$ and 1, in the order the library chose. The columns
are parallel to $(1,-1)$ and $(1,2)$, scaled to length one, with a sign that
the library also chose. The residual is at the level of rounding error.

Libraries do not solve the characteristic polynomial. For $n\ge5$ there is
no general formula for its roots in radicals, and the roots are highly
sensitive to rounding errors in the coefficients. Instead, `eig` applies a sequence of orthogonal changes of basis that drives
the matrix to triangular form without changing its eigenvalues. A triangular
matrix shows its eigenvalues on the diagonal, since
$\det(\mathbf T-\lambda\mathbf I)=\prod_i(t_{ii}-\lambda)$ by the triangular
determinant rule. For matrices too large to factor at all, power iteration
(:numref:`subsec_mdl-power-iteration`) needs only matrix--vector products.

### Diagonalization
:label:`subsec_mdl-diagonalization`

We return to the opening question, $\mathbf P^{50}\mathbf p_0$. Collect the two
eigenvectors as the columns of $\mathbf V$ and the eigenvalues on the diagonal
of $\boldsymbol\Lambda$:

$$
\mathbf V=\begin{pmatrix}1&1\\2&-1\end{pmatrix},
\qquad
\boldsymbol\Lambda=\begin{pmatrix}1&0\\0&1/4\end{pmatrix}.
$$

The two eigenvector equations $\mathbf P\mathbf v_i=\lambda_i\mathbf v_i$,
written side by side as columns, read
$\mathbf P\mathbf V=\mathbf V\boldsymbol\Lambda$. Since
$\det\mathbf V=-1-2=-3\neq0$, the matrix $\mathbf V$ is invertible, with
$\mathbf V^{-1}=\tfrac13\left(\begin{smallmatrix}1&1\\2&-1\end{smallmatrix}\right)$,
and multiplying by $\mathbf V^{-1}$ on the right gives

$$
\mathbf{P} = \mathbf{V} \boldsymbol{\Lambda} \mathbf{V}^{-1}.
$$
:eqlabel:`eq_mdl-eig_decomp`

In words: change to eigenvector coordinates ($\mathbf V^{-1}$), scale each
coordinate by its eigenvalue ($\boldsymbol\Lambda$), and change back
($\mathbf V$). A square matrix $\mathbf A$ that can be written as
$\mathbf A=\mathbf V\boldsymbol\Lambda\mathbf V^{-1}$, with an invertible
$\mathbf V$ and a diagonal $\boldsymbol\Lambda$, is called *diagonalizable*;
the factorization is its *eigendecomposition*. It exists if and only if the
matrix has $n$ linearly independent eigenvectors to fill the columns of
$\mathbf V$.

Powers of a diagonalizable matrix reduce to powers of numbers. In a product of
$k$ copies of
$\mathbf V\boldsymbol\Lambda\mathbf V^{-1}$, each inner pair
$\mathbf V^{-1}\mathbf V$ cancels:

$$
\mathbf{A}^k = \overbrace{(\mathbf{V}\boldsymbol{\Lambda} \mathbf{V}^{-1})\cdots(\mathbf{V}\boldsymbol{\Lambda} \mathbf{V}^{-1})}^{\textrm{$k$ times}} =  \mathbf{V}\overbrace{\boldsymbol{\Lambda}\cdots\boldsymbol{\Lambda}}^{\textrm{$k$ times}}\mathbf{V}^{-1} = \mathbf{V}\boldsymbol{\Lambda}^k \mathbf{V}^{-1}.
$$

A power of the matrix is the same power of each eigenvalue. Applied to a
vector $\mathbf x$, with $\mathbf c=\mathbf V^{-1}\mathbf x$ the coordinates
of $\mathbf x$ in the eigenbasis, so that
$\mathbf x=\sum_i c_i\mathbf v_i$, this reads

$$
\mathbf A^k\mathbf x=\mathbf V\boldsymbol\Lambda^k\mathbf c
=\sum_{i=1}^n c_i\lambda_i^k\mathbf v_i .
$$
:eqlabel:`eq_mdl-eig-power-expansion`

Each eigen-coordinate evolves on its own, as the scalar sequence
$c_i\lambda_i^k$. For the surfer, $\mathbf c=\mathbf V^{-1}\mathbf p_0
=\tfrac13(1,2)=(\tfrac13,\tfrac23)$, and therefore

$$
\mathbf p_k=\mathbf P^k\mathbf p_0
=\frac13\begin{pmatrix}1\\2\end{pmatrix}
+\frac23\Bigl(\frac14\Bigr)^k\begin{pmatrix}1\\-1\end{pmatrix}.
$$
:eqlabel:`eq_mdl-markov-closed-form`

This answers the opening question. The distribution approaches
$\boldsymbol\pi=(1/3,2/3)$: in the long run the surfer spends twice as much
time on page 2 as on page 1. The error is $\tfrac23\,4^{-k}$ in each entry and
points along the eigenvector $(1,-1)$. Every click divides it by 4, so after
10 clicks it is $6\times10^{-7}$ and after 50 clicks below $10^{-30}$. At $k=3$ the formula
gives $(\tfrac13+\tfrac1{96},\ \tfrac23-\tfrac1{96})=(0.34375,0.65625)$, the
third iterate of the opening. :numref:`fig_mdl-la-markov-iterates` shows the
geometry: every $\mathbf p_k$ has entries summing to one, so the iterates
slide along the line $x+y=1$ toward the point where the eigenvector of
$\lambda=1$ meets it.

![The surfer's distributions $\mathbf p_k$ slide along $x+y=1$ toward $\boldsymbol\pi=(1/3,2/3)$, where the eigenvector $(1,2)$ of $\lambda=1$ meets the line. The error points along the eigenvector $(1,-1)$ of $\lambda=1/4$ and shrinks by a factor of 4 per click.](../img/mdl-la-markov-iterates.svg)
:label:`fig_mdl-la-markov-iterates`

Does the closed form reproduce 50 matrix multiplications? The cell compares
the two and measures how much the error shrinks per click.

```{.python .input #mdl-eigendecomposition-diagonalization}
p0 = np.array([1.0, 0.0])
c = np.linalg.solve(V, p0)              # coordinates of p0 in the eigenbasis
print('P^50 p0             :', np.linalg.matrix_power(P, 50) @ p0)
print('sum c_i lam_i^50 v_i:', V @ (c * lam ** 50))
pi = np.array([1 / 3, 2 / 3])
errs = np.array([np.abs(np.linalg.matrix_power(P, k) @ p0 - pi).max()
                 for k in range(8)])
print('error ratios:', np.round(errs[1:] / errs[:-1], 4))
```

Both computations give $(1/3,2/3)$ to every printed digit, and the error
shrinks by the factor $0.25$ at every click, the eigenvalue of the direction
in which it lies. The eigenvalue 1 belongs to the part of $\mathbf p_0$ that
survives, the eigenvalue $1/4$ to the part that dies out.

The same recipe handles other functions of a diagonalizable matrix. If no
eigenvalue is zero, $\mathbf A^{-1}=\mathbf V\boldsymbol\Lambda^{-1}\mathbf V^{-1}$:
invert each eigenvalue. The *matrix exponential*
$e^{\mathbf A t}=\sum_{j\ge0}(\mathbf A t)^j/j!$ solves the differential
equation $\dot{\mathbf x}=\mathbf A\mathbf x$, and for diagonalizable
$\mathbf A$ it is computed as
$e^{\mathbf A t}=\mathbf V e^{\boldsymbol\Lambda t}\mathbf V^{-1}$, since each
term of the series is a power. The scalar factors $e^{\lambda_i t}$ then
decay if and only if the real parts of the eigenvalues are negative, which is
how the stability of linear differential equations is read.

**Similarity.** The eigendecomposition says that $\mathbf P$ and
$\boldsymbol\Lambda$ are the same map written in two bases. Two matrices
related by $\mathbf A'=\mathbf B\mathbf A\mathbf B^{-1}$ for an invertible
$\mathbf B$ are called *similar*. The next proposition shows that a change of
basis cannot change the eigenvalues. It uses the multiplicativity of the
determinant from :numref:`sec_mdl-linear-maps`,

$$
\det(\mathbf X\mathbf Y)=\det\mathbf X\,\det\mathbf Y,
$$

which also gives $\det(\mathbf B^{-1})=1/\det\mathbf B$ from
$\mathbf B\mathbf B^{-1}=\mathbf I$.

**Proposition (similar matrices share the characteristic polynomial).** *For
every square $\mathbf A$ and every invertible $\mathbf B$,*

$$
\det(\mathbf B\mathbf A\mathbf B^{-1}-\lambda\mathbf I)=\det(\mathbf A-\lambda\mathbf I).
$$
:eqlabel:`eq_mdl-similar-charpoly`

**Proof.** Write $\lambda\mathbf I=\mathbf B(\lambda\mathbf I)\mathbf B^{-1}$
and factor:
$\det(\mathbf B\mathbf A\mathbf B^{-1}-\lambda\mathbf I)
=\det\bigl(\mathbf B(\mathbf A-\lambda\mathbf I)\mathbf B^{-1}\bigr)
=\det\mathbf B\,\det(\mathbf A-\lambda\mathbf I)\,\det\mathbf B^{-1}
=\det(\mathbf A-\lambda\mathbf I)$. $\blacksquare$

Similar matrices therefore have the same eigenvalues with the same
multiplicities, and every quantity computed from the characteristic
polynomial is a *similarity invariant*. The two most useful ones are the
determinant and the trace. For a diagonalizable matrix,

$$
\det(\mathbf{A}) = \det(\mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^{-1})
  = \det(\mathbf{V})\det(\boldsymbol{\Lambda})\det(\mathbf{V}^{-1})
  = \det(\boldsymbol{\Lambda})
  = \lambda_1 \lambda_2 \cdots \lambda_n .
$$
:eqlabel:`eq_mdl-det-eig`

The determinant is the product of the eigenvalues: whatever $\mathbf V$ does
to volumes, $\mathbf V^{-1}$ undoes, so only the scaling by
$\boldsymbol\Lambda$ remains. The trace is cyclic,

$$
\operatorname{tr}(\mathbf X\mathbf Y)=\operatorname{tr}(\mathbf Y\mathbf X),
$$

for any $\mathbf X$ and $\mathbf Y$ whose products are square (both sides
equal $\sum_{i,j}x_{ij}y_{ji}$). With $\mathbf X=\mathbf V$ and
$\mathbf Y=\boldsymbol\Lambda\mathbf V^{-1}$,

$$
\operatorname{tr}(\mathbf{A})
  = \operatorname{tr}(\mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^{-1})
  = \operatorname{tr}(\boldsymbol{\Lambda}\mathbf{V}^{-1}\mathbf{V})
  = \operatorname{tr}(\boldsymbol{\Lambda})
  = \lambda_1+\cdots+\lambda_n .
$$
:eqlabel:`eq_mdl-tr-eig`

The trace is the sum of the eigenvalues. On the surfer's matrix both identities
check by eye: $\operatorname{tr}\mathbf P=\tfrac12+\tfrac34=\tfrac54=1+\tfrac14$
and $\det\mathbf P=\tfrac38-\tfrac18=\tfrac14=1\cdot\tfrac14$. For a $2\times2$
matrix the two numbers determine both eigenvalues, as the roots of
$\lambda^2-(\operatorname{tr})\lambda+\det$.

Both identities hold for every square matrix, diagonalizable or not, if the
eigenvalues are counted over $\mathbb C$ with their multiplicity as roots. The
*fundamental theorem of algebra*, which we use without proof, says that a
polynomial of degree $n$ factors completely over $\mathbb C$, so
$p(\lambda)=\prod_{i=1}^n(\lambda_i-\lambda)$. Setting $\lambda=0$ gives
$\det\mathbf A=\prod_i\lambda_i$. On the right, the coefficient of $\lambda^{n-1}$ is
$(-1)^{n-1}\sum_i\lambda_i$. On the left, only the product of the diagonal
entries of $\mathbf A-\lambda\mathbf I$ reaches degree $n-1$, so the
coefficient is $(-1)^{n-1}\sum_i a_{ii}$. Hence
$\operatorname{tr}\mathbf A=\sum_i\lambda_i$.
The complex eigenvalues of a real matrix come in conjugate pairs, so the
product and the sum are real.

Finally, the rank of a diagonalizable matrix is the number of its nonzero
eigenvalues, counted with multiplicity: multiplying by the invertible
$\mathbf V$ and $\mathbf V^{-1}$ changes no dimensions, so
$\operatorname{rank}\mathbf A=\operatorname{rank}\boldsymbol\Lambda$.
Exercise :exref:`ex_mdl-eigendecomposition-rank-zero-eigenvalues` treats the
general case.

### Algebraic and Geometric Multiplicity
:label:`subsec_mdl-multiplicity`

The surfer's matrix had two distinct eigenvalues and hence two independent
eigenvectors. When does a matrix have $n$ independent eigenvectors, so that
$\mathbf V$ is invertible? The answer is a count with two notions of
multiplicity. The number of times $\lambda$ appears as a root of
$p(\lambda)$ is its *algebraic multiplicity* $a(\lambda)$. The dimension of
its eigenspace, the number of independent eigenvectors it contributes, is its
*geometric multiplicity* $g(\lambda)$.

**Proposition.** *For every eigenvalue $\lambda$ of a square matrix,
$1\le g(\lambda)\le a(\lambda)$.*

**Proof.** Since $\mathbf A-\lambda\mathbf I$ is singular, its null space is
not zero, so $g\ge1$. For the upper bound, take a basis
$\mathbf v_1,\ldots,\mathbf v_g$ of the eigenspace, extend it to a basis of
$\mathbb C^n$, and collect the basis as the columns of $\mathbf B$. The first
$g$ columns of $\mathbf A\mathbf B$ are $\lambda\mathbf v_1,\ldots,\lambda\mathbf v_g$,
so in this basis the matrix is block upper triangular,

$$
\mathbf B^{-1}\mathbf A\mathbf B=\begin{pmatrix}\lambda\mathbf I_g&\mathbf X\\\mathbf 0&\mathbf Y\end{pmatrix}.
$$

By :eqref:`eq_mdl-similar-charpoly`, $p(t)=(\lambda-t)^g\det(\mathbf Y-t\mathbf I)$,
so $(\lambda-t)^g$ divides the characteristic polynomial and $a\ge g$. $\blacksquare$

An eigenvalue can never have more independent eigenvectors than its count as
a root, but it can have fewer. Eigenvectors of different eigenvalues are
independent (the next proposition, applied to the components of a dependence
relation in each eigenspace), so a matrix has $\sum_\lambda g(\lambda)$
independent eigenvectors in total. Over $\mathbb C$ the algebraic
multiplicities add up to $n$, so this total is at most $n$. It follows that *the eigenvectors of $\mathbf A$ span
$\mathbb C^n$ if and only if $g(\lambda)=a(\lambda)$ for every complex
eigenvalue $\lambda$*.

A real matrix has a basis of real eigenvectors if and
only if, in addition, all of its eigenvalues are real. The rotation by
$90^\circ$, $\mathbf Q_{90^\circ}=\left(\begin{smallmatrix}0&-1\\1&0\end{smallmatrix}\right)$,
shows that the second condition is needed: its eigenvalues $\pm i$ each have
$g=a=1$, so it has a complex eigenbasis, but no real eigenvector.

**Proposition (distinct eigenvalues give independent eigenvectors).**
*Eigenvectors belonging to distinct eigenvalues are linearly independent. In
particular, an $n\times n$ matrix with $n$ distinct eigenvalues is
diagonalizable.*

**Proof.** Suppose, for contradiction, that some eigenvectors
$\mathbf{v}_1,\ldots,\mathbf{v}_m$ with distinct eigenvalues
$\lambda_1,\ldots,\lambda_m$ are dependent, and take a dependence relation
$\sum_{i=1}^{m} c_i\mathbf{v}_i = \mathbf 0$ with the fewest nonzero
coefficients; relabel so that $c_m\neq0$. Such a relation has at least two
nonzero coefficients, because $c_m\mathbf{v}_m=\mathbf 0$ alone would force
$\mathbf{v}_m=\mathbf 0$. Apply $\mathbf{A}$ to the relation and subtract
$\lambda_m$ times the relation:

$$
\mathbf 0 = \mathbf{A}\!\sum_i c_i\mathbf{v}_i - \lambda_m\!\sum_i c_i\mathbf{v}_i
   = \sum_{i=1}^{m-1} c_i(\lambda_i-\lambda_m)\,\mathbf{v}_i .
$$

The $\mathbf{v}_m$ term cancels, so this relation is shorter. It is still
nontrivial: the second nonzero coefficient $c_i$, $i<m$, survives as
$c_i(\lambda_i-\lambda_m)\neq0$ because the eigenvalues are distinct. This
contradicts minimality. With $n$ distinct eigenvalues we obtain $n$
independent eigenvectors, a basis of $\mathbb C^n$ (of $\mathbb R^n$ when the
eigenvalues are real), so $\mathbf{V}$ is invertible. $\blacksquare$

When eigenvalues repeat, diagonalizability can fail. The simplest example is
the *shear*

$$
\mathbf{A} = \begin{pmatrix}
1 & 1 \\
0 & 1
\end{pmatrix}.
$$
:eqlabel:`eq_mdl-defective-shear`

Its characteristic polynomial is $(1-\lambda)^2$, so $\lambda=1$ has algebraic
multiplicity 2. But $(\mathbf{A}-\mathbf{I})\mathbf{v}=\mathbf 0$ reads
$\left(\begin{smallmatrix}0&1\\0&0\end{smallmatrix}\right)\mathbf v=\mathbf 0$,
that is, $v_2=0$, so the eigenspace is the line spanned by $(1,0)$. The geometric
multiplicity is 1, there is no eigenbasis, and the shear is not
diagonalizable. A matrix with $g(\lambda)<a(\lambda)$ for some eigenvalue is
called *defective*. :numref:`fig_mdl-la-defective-shear` shows the shortage:
the shear slides every horizontal line sideways by an amount proportional to
its height, and the horizontal axis is the only line through the origin that
keeps its direction. A diagonalizable $2\times2$ matrix with a repeated
eigenvalue $\lambda$ would be $\lambda\mathbf I$ and would keep every
direction.

![The defective shear acting on the plane. Every horizontal line slides to the right by an amount proportional to its height, so the grid tilts, but the $x$-axis (green) is carried to itself: $\mathbf{A}\mathbf{e}_1=\mathbf{e}_1$, the only eigendirection, with $\lambda=1$. The other basis vector picks up a horizontal component, $\mathbf{A}\mathbf{e}_2=(1,1)$, and no second direction is preserved: geometric multiplicity 1 against algebraic multiplicity 2.](../img/mdl-la-defective-shear.svg)
:label:`fig_mdl-la-defective-shear`

Powers of the shear show what defectiveness does to iteration:
$\mathbf A^k=\left(\begin{smallmatrix}1&k\\0&1\end{smallmatrix}\right)$, so
$\mathbf A^k\mathbf e_2=(k,1)$ grows without bound although the only
eigenvalue is 1. No eigen-expansion can produce the factor $k$; the next
subsection explains where it comes from.

### The Jordan Normal Form
:label:`subsec_mdl-jordan`

When the geometric multiplicity falls short, no eigenbasis exists. What
structure remains? Over the complex numbers, every square matrix is similar to
a block-diagonal matrix whose diagonal blocks are *Jordan blocks*

$$
\mathbf{J}_m(\lambda) =
\begin{pmatrix}
\lambda & 1      &        &        \\
        & \lambda& \ddots &        \\
        &        & \ddots & 1      \\
        &        &        & \lambda
\end{pmatrix},
$$

$m\times m$ blocks with one eigenvalue $\lambda$ repeated down the diagonal and
1s just above it. This is the *Jordan normal form* :cite:`Horn.Johnson.2012`,
which we use without proof. By :eqref:`eq_mdl-similar-charpoly` it carries the
same eigenvalues as the original matrix.

The form displays both multiplicities, because each block contributes one
eigenvector: within a block, $(\mathbf{J}_m(\lambda)-\lambda\mathbf{I})\mathbf{v}=\mathbf 0$
reads $v_2=\cdots=v_m=0$, so $\mathbf v$ is a multiple of the first basis
vector. The geometric multiplicity of $\lambda$ is therefore the number of
blocks carrying $\lambda$, the algebraic multiplicity is their total size, and
a matrix is diagonalizable if and only if every block is $1\times1$. The shear
:eqref:`eq_mdl-defective-shear` is already in Jordan form: it is the single
block $\mathbf{J}_2(1)$.

What does a Jordan block do under iteration? Write
$\mathbf{J}_m(\lambda)=\lambda\mathbf{I}+\mathbf{N}$, where $\mathbf{N}$ has
1s just above the diagonal and zeros elsewhere. $\mathbf N$ is a shift: it
sends each basis vector to the previous one, so $\mathbf N^m=\mathbf 0$ (a
matrix with a vanishing power is called *nilpotent*). Because $\mathbf{I}$ and
$\mathbf{N}$ commute, the binomial theorem applies, and for $m=2$ every term
beyond the first two contains $\mathbf{N}^2=\mathbf 0$:

$$
\mathbf{J}_2(\lambda)^k = (\lambda\mathbf{I}+\mathbf{N})^k
  = \lambda^k\mathbf{I} + k\lambda^{k-1}\mathbf{N}.
$$

The eigenvalue still decides between growth and decay in the long run, but it
is multiplied by a polynomial in $k$ (of degree $m-1$ for a block of size $m$).
With $|\lambda|=1$, as for the shear, the iterates drift linearly. With
$|\lambda|$ slightly below 1, the corner entry $k\lambda^{k-1}$ first grows,
peaks near $k\approx-1/\log|\lambda|$, and only then decays. For $\lambda=0.99$
this puts the peak near $k\approx100$, where the corner entry is
$100\cdot0.99^{99}\approx37$. The cell checks the prediction.

```{.python .input #mdl-eigendecomposition-jordan-power}
J = np.array([[0.99, 1.], [0., 0.99]])   # the Jordan block J_2(0.99)
for k in [1, 50, 100, 200, 800]:
    Jk = np.linalg.matrix_power(J, k)
    pred = k * 0.99 ** (k - 1)   # predicted corner entry, k * lambda^(k-1)
    print(f'k = {k:3d}   ||J^k|| = {np.linalg.norm(Jk, 2):8.3f}   '
          f'k * 0.99^(k-1) = {pred:8.3f}')
```

Every eigenvalue of $\mathbf{J}$ is $0.99$, yet the norm of its powers climbs to
about 37 at $k=100$ and is still 27 at $k=200$ before the factor $0.99^k$
dominates. The eigenvalues alone do not reveal this transient growth, to which
:numref:`subsec_mdl-transient-growth` returns.

Jordan blocks are one way in which a real matrix falls short of a real
eigenbasis: it has too few eigenvectors. The second way is different in kind.
A real matrix can have no real eigenvector at all in some plane, because it
rotates that plane.

### Complex Eigenvalues
:label:`subsec_mdl-complex-rotation`

Consider the planar rotation by an angle $\theta$,

$$
\mathbf{Q}_\theta = \begin{pmatrix}\cos\theta & -\sin\theta\\
                            \sin\theta & \phantom{-}\cos\theta\end{pmatrix}.
$$

If $\theta$ is not a multiple of $\pi$, $\mathbf{Q}_\theta$ turns every nonzero
real vector, so it has no real eigenvector. Its characteristic polynomial
$\lambda^2-2\cos\theta\,\lambda+1$ has the complex conjugate roots
$\lambda=\cos\theta\pm i\sin\theta=e^{\pm i\theta}$. The modulus
$|\lambda|=1$ records that $\mathbf{Q}_\theta$ preserves lengths, and the
argument $\pm\theta$ records the angle. The cell confirms this for
$\theta=\pi/6$.

```{.python .input #eigendecomposition-complex-rotation}
theta = np.pi / 6
Q = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])
lam_Q = np.linalg.eigvals(Q)
print('eigenvalues of Q:', np.round(lam_Q, 4))
print('e^(+-i theta):   ', np.round(np.exp([1j * theta, -1j * theta]), 4))
print('moduli:', np.round(np.abs(lam_Q), 4))
```

The chapter's running $2\times2$ matrix has no visible rotation in its
entries:

$$
\mathbf A=\begin{pmatrix}1&2\\-1&3\end{pmatrix},
\qquad \operatorname{tr}\mathbf A=4,\quad \det\mathbf A=5 .
$$

Its characteristic polynomial is $\lambda^2-4\lambda+5$, with roots
$\lambda=2\pm i=\sqrt5\,e^{\pm i\theta}$ and $\theta=\arctan\tfrac12\approx26.6^\circ$.
The eigenvector for $2+i$ comes from the second row of
$\mathbf A-(2+i)\mathbf I$, which reads $-w_1+(1-i)w_2=0$. With $w_2=1$,

$$
\mathbf w=\begin{pmatrix}1-i\\1\end{pmatrix}
=\underbrace{\begin{pmatrix}1\\1\end{pmatrix}}_{\mathbf u}
+i\underbrace{\begin{pmatrix}-1\\0\end{pmatrix}}_{\mathbf v}.
$$

Where is the rotation hidden? Take any real matrix with an eigenpair
$\mathbf{A}\mathbf{w}=\lambda\mathbf{w}$, $\lambda=a+ib$, $b\neq0$, and split
the eigenvector into real and imaginary parts, $\mathbf{w}=\mathbf{u}+i\mathbf{v}$.
Expanding $\mathbf{A}(\mathbf{u}+i\mathbf{v})=(a+ib)(\mathbf{u}+i\mathbf{v})$
and matching real and imaginary parts gives

$$
\mathbf{A}\mathbf{u} = a\mathbf{u} - b\mathbf{v},
\qquad
\mathbf{A}\mathbf{v} = b\mathbf{u} + a\mathbf{v}.
$$

Both images land back in $\operatorname{span}\{\mathbf{u},\mathbf{v}\}$, so
this span is an invariant subspace. It is a plane. If $\mathbf u$ and
$\mathbf v$ were dependent, their span would be a line (not zero, since
$\mathbf w\neq\mathbf 0$) mapped into itself, on which $\mathbf A$ multiplies by
a real number $\mu$. Then $\mathbf A\mathbf w=\mu\mathbf w$, and $\lambda=\mu$
would be real, contradicting $b\neq0$. Collecting the two identities as
columns with $\mathbf{B}=[\mathbf{v}\;\;\mathbf{u}]$ gives
$\mathbf{A}\mathbf{B}=\mathbf{B}\left(\begin{smallmatrix}a & -b\\ b & \phantom{-}a\end{smallmatrix}\right)$,
that is,

$$
\mathbf{B}^{-1}\mathbf{A}\mathbf{B}
= \begin{pmatrix} a & -b\\ b & \phantom{-}a \end{pmatrix}
= \underbrace{\sqrt{a^2+b^2}}_{r}\,
  \begin{pmatrix} \cos\theta & -\sin\theta\\ \sin\theta & \phantom{-}\cos\theta \end{pmatrix},
\qquad \theta=\arg(a+ib).
$$
:eqlabel:`eq_mdl-rotation-scaling-block`

In the basis $(\mathbf v,\mathbf u)$ of its invariant plane, the matrix is a
rotation--scaling block $r\mathbf Q_\theta$: it scales by the modulus and
rotates by the argument. The block does not depend on the scale or phase of
$\mathbf w$, since the derivation used only $\mathbf A\mathbf w=\lambda\mathbf w$.
The correspondence also reads from right to left. Identify
$(x,y)\in\mathbb{R}^2$ with the complex number $x+iy$. The block maps
$(x,y)$ to $(ax-by,\,bx+ay)$, which is multiplication by $a+ib$.

For the running matrix, $\mathbf B=\left(\begin{smallmatrix}-1&1\\0&1\end{smallmatrix}\right)$
and $\mathbf B^{-1}\mathbf A\mathbf B=\left(\begin{smallmatrix}2&-1\\1&2\end{smallmatrix}\right)$:
scale by $\sqrt5$ and rotate by $26.6^\circ$. The cell computes the block
from `eig` and follows the iterates $\mathbf A^k(1,0)$.

```{.python .input #mdl-eigendecomposition-complex-eigenvalues-are-rotations-1}
A = np.array([[1., 2.], [-1., 3.]])
lam, V = np.linalg.eig(A)
j = np.argmax(lam.imag)                  # the eigenvalue a + ib with b > 0
r, theta = np.abs(lam[j]), np.angle(lam[j])
u, v = V[:, j].real, V[:, j].imag
B = np.column_stack([v, u])              # basis (v, u) of the invariant plane
print(f'lambda = {lam[j]:.4f}   r = {r:.4f}   theta = {theta:.4f}')
print('B^-1 A B =\n', np.round(np.linalg.inv(B) @ A @ B, 4))
xs = [np.array([1., 0.])]
for _ in range(8):
    xs.append(A @ xs[-1])
norms = np.linalg.norm(xs, axis=1)
print('norm ratios:', np.round(norms[1:] / norms[:-1], 2))
```

The block is $\left(\begin{smallmatrix}2&-1\\1&2\end{smallmatrix}\right)$,
whatever eigenvector `eig` returned. The norm ratios show what "scale by
$\sqrt5\approx2.24$" means for the iterates
$(1,0)\to(1,-1)\to(-1,-4)\to(-9,-11)\to(-31,-24)$. The ratios swing between
about 1.4 and 3.5 instead of staying at 2.24. The scaling is uniform only in
the coordinates $(\mathbf v,\mathbf u)$, which are not orthonormal. In the
original coordinates the iterates spiral outward along stretched ellipses,
and a full turn takes $2\pi/\theta\approx13.6$ steps. Only the average rate
$\|\mathbf A^k\mathbf x\|^{1/k}$ settles at $\sqrt5$ (it is 2.25 at $k=20$).
The motion is a rigid rotation and scaling only when the matrix is normal, a
property defined in :numref:`subsec_mdl-normal-matrices`.

Allowing rotation--scaling blocks next to the Jordan blocks gives the *real
Jordan form* :cite:`Horn.Johnson.2012`: every real square matrix is similar,
over $\mathbb R$, to a block-diagonal matrix built from four kinds of blocks.
There are pure stretch directions (real eigenvalues with enough
eigenvectors), Jordan blocks (real eigenvalues with too few), rotation--scaling
planes (complex conjugate pairs), and, for a complex pair that is itself
defective, rotation--scaling blocks chained by identity blocks above the
diagonal. The $4\times4$ matrix
$\left(\begin{smallmatrix}\mathbf Q_{90^\circ}&\mathbf I\\\mathbf 0&\mathbf Q_{90^\circ}\end{smallmatrix}\right)$
is the smallest example: its eigenvalues $\pm i$ each have $a=2$ and $g=1$. In
continuous time, $\dot{\mathbf x}=\mathbf A\mathbf x$, the same rotation--scaling
block produces a spiral, with the real part $a$ setting growth or decay and
the imaginary part $b$ the angular speed.

## Symmetric Matrices and Quadratic Forms

### The Spectral Theorem
:label:`subsec_mdl-spectral-theorem`

The shear and the rotation showed the two ways a real matrix can lack a real
eigenbasis: too few eigenvectors, or complex ones. For which matrices is a
real, orthonormal eigenbasis guaranteed? Symmetry rules out both failures.
The symmetric $\mathbf S$ of :numref:`sec_linear-algebra` shows what to
expect. Its unit eigenvectors
$(1,1)/\sqrt2$ and $(1,-1)/\sqrt2$ are perpendicular, and with them

$$
\mathbf S=\begin{pmatrix}2&1\\1&2\end{pmatrix}
=\underbrace{\frac1{\sqrt2}\begin{pmatrix}1&1\\1&-1\end{pmatrix}}_{\mathbf V}
\begin{pmatrix}3&0\\0&1\end{pmatrix}
\underbrace{\frac1{\sqrt2}\begin{pmatrix}1&1\\1&-1\end{pmatrix}}_{\mathbf V^\top}.
$$

The eigenvector matrix is orthogonal, $\mathbf V^\top\mathbf V=\mathbf I$, so
its inverse is its transpose. The spectral theorem states that this holds for
every symmetric matrix.

**Theorem (spectral theorem).** *Every real symmetric matrix $\mathbf{A}$ has
real eigenvalues and an orthonormal basis of real eigenvectors. Equivalently,*

$$
\mathbf{A} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^\top ,
$$
:eqlabel:`eq_mdl-spectral`

*where $\boldsymbol{\Lambda}=\operatorname{diag}(\lambda_1,\ldots,\lambda_n)$
is real and $\mathbf{V}$ is orthogonal.*

We prove three claims in turn: the eigenvalues are real, eigenvectors of
distinct eigenvalues are orthogonal, and an orthonormal eigenbasis exists. For
complex vectors we use the inner product
$\langle\mathbf x,\mathbf y\rangle=\mathbf x^*\mathbf y$, where $\mathbf x^*$
is the conjugate transpose; on real vectors it is the dot product. The only
fact we borrow is that the characteristic polynomial has at least one complex
root (the fundamental theorem of algebra), so at least one eigenpair exists
over $\mathbb{C}$.

**Proof, part (i): the eigenvalues are real.** Let
$\mathbf{A}\mathbf{v}=\lambda\mathbf{v}$ with $\mathbf{v}\neq\mathbf 0$, allowing
complex entries. Since $\mathbf{A}$ is real and symmetric,
$\mathbf A^*=\mathbf A^\top=\mathbf A$, so

$$
\lambda\,\mathbf{v}^*\mathbf{v}
   = \mathbf{v}^*\mathbf{A}\mathbf{v}
   = (\mathbf{A}\mathbf{v})^*\mathbf{v}
   = \bar\lambda\,\mathbf{v}^*\mathbf{v}.
$$

Because $\mathbf{v}^*\mathbf{v}=\|\mathbf v\|^2>0$, we may divide by it and
obtain $\lambda=\bar\lambda$: the eigenvalue is real. The eigenvector can then
be taken real as well, since it solves the real system
$(\mathbf A-\lambda\mathbf I)\mathbf v=\mathbf 0$. $\blacksquare$

**Proof, part (ii): eigenvectors of distinct eigenvalues are orthogonal.** Let
$\mathbf{A}\mathbf{u}=\lambda\mathbf{u}$ and
$\mathbf{A}\mathbf{v}=\mu\mathbf{v}$ with $\lambda\neq\mu$, both real by (i),
and $\mathbf u$, $\mathbf v$ real. Using $\mathbf{A}=\mathbf{A}^\top$ to move
$\mathbf{A}$ from one side of the inner product to the other,

$$
\lambda\langle\mathbf{u},\mathbf{v}\rangle
  = \langle\mathbf{A}\mathbf{u},\mathbf{v}\rangle
  = \langle\mathbf{u},\mathbf{A}\mathbf{v}\rangle
  = \mu\langle\mathbf{u},\mathbf{v}\rangle,
$$

so $(\lambda-\mu)\langle\mathbf{u},\mathbf{v}\rangle=0$, and since
$\lambda\neq\mu$, $\langle\mathbf{u},\mathbf{v}\rangle=0$. $\blacksquare$

**Proof, part (iii): an orthonormal eigenbasis.** We induct on $n$; for $n=1$
there is nothing to prove. For $n>1$, part (i) gives a real eigenpair
$(\lambda_1,\mathbf{v}_1)$ with $\|\mathbf{v}_1\|=1$. Let
$U=\mathbf{v}_1^{\perp}$ be the orthogonal complement, of dimension $n-1$. The
matrix maps $U$ into itself: for any $\mathbf{x}\in U$,

$$
\langle\mathbf{A}\mathbf{x},\mathbf{v}_1\rangle
  = \langle\mathbf{x},\mathbf{A}\mathbf{v}_1\rangle
  = \lambda_1\langle\mathbf{x},\mathbf{v}_1\rangle = 0 .
$$

Let the columns of $\mathbf Q\in\mathbb R^{n\times(n-1)}$ be an orthonormal
basis of $U$. Since the columns of $\mathbf A\mathbf Q$ lie in $U$, we can write
$\mathbf A\mathbf Q=\mathbf Q\mathbf C$ with
$\mathbf C=\mathbf Q^\top\mathbf A\mathbf Q$, the matrix of $\mathbf A$
restricted to $U$. It is symmetric,
$\mathbf C^\top=\mathbf Q^\top\mathbf A^\top\mathbf Q=\mathbf C$, and of size
$n-1$, so by the inductive hypothesis
$\mathbf C=\mathbf V'\boldsymbol\Lambda'\mathbf V'^\top$ with $\mathbf V'$
orthogonal. The columns of $\mathbf Q\mathbf V'$ are orthonormal, since
$(\mathbf Q\mathbf V')^\top\mathbf Q\mathbf V'=\mathbf V'^\top\mathbf V'=\mathbf I$.
They lie in $U$, hence are orthogonal to $\mathbf v_1$. They are also
eigenvectors, since
$\mathbf A\mathbf Q\mathbf V'=\mathbf Q\mathbf C\mathbf V'=\mathbf Q\mathbf V'\boldsymbol\Lambda'$.
Together with $\mathbf v_1$ they form the orthogonal matrix
$\mathbf V=[\mathbf v_1\;\mathbf Q\mathbf V']$, and
$\mathbf A\mathbf V=\mathbf V\boldsymbol\Lambda$. $\blacksquare$

The factorization :eqref:`eq_mdl-spectral` reads as three steps. First
$\mathbf V^\top$ turns the orthonormal eigenvectors onto the coordinate axes.
This is a rotation, possibly combined with a reflection; for $\mathbf S$
above, $\mathbf V$ has determinant $-1$. Then $\boldsymbol\Lambda$ stretches
each axis by its eigenvalue, and $\mathbf V$ turns the axes back. This is the
left panel of :numref:`fig_mdl-la-eig-ellipse`, now proved for every symmetric
matrix.

Every matrix $\mathbf A$, square or not, yields a symmetric matrix
$\mathbf A^\top\mathbf A$. Its quadratic form
$\mathbf x^\top\mathbf A^\top\mathbf A\mathbf x=\|\mathbf A\mathbf x\|^2$ is
never negative, a property that the next subsection names and tests.

### Positive Definiteness
:label:`subsec_mdl-psd`

Is the quadratic form $\mathbf{x}^\top\mathbf{A}\mathbf{x}$ ever negative?
The question decides whether a critical point of a loss is a minimum, whether
a matrix can be a covariance, and whether a least-squares problem has a
unique solution. A symmetric matrix $\mathbf{A}$ is *positive semidefinite*
(PSD), written $\mathbf{A}\succeq0$, if

$$
\mathbf{x}^\top\mathbf{A}\mathbf{x} \ge 0 \quad\textrm{for all } \mathbf{x},
$$

and *positive definite* (PD), written $\mathbf{A}\succ0$, if the inequality
is strict for every $\mathbf{x}\neq\mathbf 0$. For example,
$\mathbf x^\top\mathbf S\mathbf x=2x_1^2+2x_1x_2+2x_2^2=x_1^2+x_2^2+(x_1+x_2)^2$
is positive unless $\mathbf x=\mathbf 0$, so $\mathbf S\succ0$, while
$\left(\begin{smallmatrix}1&0\\0&-1\end{smallmatrix}\right)$ is neither, since
$\mathbf e_2$ gives $-1$. Completing squares by hand does not scale, but the
eigenvalues decide the question for every symmetric matrix.

**Proposition (definiteness from eigenvalues).** *For symmetric $\mathbf{A}$,*

$$
\mathbf{A}\succeq0 \iff \lambda_i\ge0\ \textrm{for all } i,
\qquad
\mathbf{A}\succ0 \iff \lambda_i>0\ \textrm{for all } i .
$$

**Proof.** Write $\mathbf{A}=\mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^\top$ by
the spectral theorem and change variables to the eigen-coordinates
$\mathbf{y}=\mathbf{V}^\top\mathbf{x}$, a rotation (possibly with a
reflection) that preserves lengths. Then

$$
\mathbf{x}^\top\mathbf{A}\mathbf{x}
  = \mathbf{x}^\top\mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^\top\mathbf{x}
  = \mathbf{y}^\top\boldsymbol{\Lambda}\mathbf{y}
  = \sum_{i=1}^{n}\lambda_i\,y_i^2
  = \sum_{i=1}^{n}\lambda_i\,(\mathbf{v}_i^\top\mathbf{x})^2 .
$$
:eqlabel:`eq_mdl-quadform`

The quadratic form is a weighted sum of squares with the eigenvalues as
weights. If every $\lambda_i\ge0$, the sum is non-negative. Conversely,
$\mathbf{x}=\mathbf{v}_j$ gives $\mathbf y=\mathbf e_j$ and
$\mathbf{v}_j^\top\mathbf{A}\mathbf{v}_j=\lambda_j$, which forces
$\lambda_j\ge0$. The PD statement is the same argument with strict
inequalities, using that $\mathbf y\neq\mathbf 0$ whenever
$\mathbf x\neq\mathbf 0$. $\blacksquare$

For $\mathbf S$ and $\mathbf x=\mathbf e_1$, the eigen-coordinates are
$\mathbf y=\mathbf V^\top\mathbf e_1=(1/\sqrt2,1/\sqrt2)$, and
:eqref:`eq_mdl-quadform` gives $3\cdot\tfrac12+1\cdot\tfrac12=2$, which is
$\mathbf e_1^\top\mathbf S\mathbf e_1=s_{11}$. Three consequences cover most
places where definiteness appears in deep learning.

* **PD implies invertible.** If $\mathbf{A}\succ0$ and
  $\mathbf{A}\mathbf{x}=\mathbf 0$, then
  $\mathbf{x}^\top\mathbf{A}\mathbf{x}=0$, which forces $\mathbf{x}=\mathbf 0$.
  Equivalently, all $\lambda_i>0$, so by :eqref:`eq_mdl-det-eig`
  $\det\mathbf A=\prod_i\lambda_i>0$.
* **Gram and covariance matrices are PSD.** For any matrix $\mathbf{X}$,
  $\mathbf{x}^\top\mathbf{X}^\top\mathbf{X}\mathbf{x}=\|\mathbf{X}\mathbf{x}\|^2\ge0$,
  so $\mathbf X^\top\mathbf X\succeq0$, and it is PD if and only if
  $\mathbf{X}$ has full column rank (so that $\mathbf{X}\mathbf{x}=\mathbf 0$
  forces $\mathbf{x}=\mathbf 0$). A sample covariance is such a Gram matrix
  divided by the number of samples, so it is always PSD.
* **A PD Hessian gives a strict local minimum.** Let $f$ be twice
  continuously differentiable with $\nabla f(\mathbf x^*)=\mathbf 0$, and let
  its Hessian $\nabla^2 f(\mathbf x^*)$, the symmetric matrix of second
  partial derivatives, be PD. For a small displacement $\boldsymbol\delta$,
  Taylor's theorem gives
  $f(\mathbf x^*+\boldsymbol\delta)=f(\mathbf x^*)+\tfrac12\boldsymbol\delta^\top\nabla^2 f(\mathbf x^*)\boldsymbol\delta+o(\|\boldsymbol\delta\|^2)$,
  and by :eqref:`eq_mdl-quadform` the quadratic term is at least
  $\tfrac12\lambda_{\min}\|\boldsymbol\delta\|^2>0$. So $\mathbf x^*$ is a
  strict local minimum. The condition is sufficient, not necessary:
  $f(x)=x^4$ has a strict minimum at 0 with Hessian 0.

Positive definiteness also has an algorithmic certificate, the *Cholesky
factorization*: $\mathbf{A}\succ0$ if and only if
$\mathbf{A}=\mathbf{L}\mathbf{L}^\top$ for a lower-triangular $\mathbf{L}$ with
positive diagonal. One direction is immediate: such an $\mathbf{L}$ is
invertible, so $\mathbf{x}^\top\mathbf{A}\mathbf{x}=\|\mathbf{L}^\top\mathbf{x}\|^2>0$
for $\mathbf{x}\neq\mathbf 0$. Attempting the factorization is the standard
numerical test for definiteness: it costs about $n^3/3$ floating-point
operations, against about $4n^3/3$ for the eigenvalues of a symmetric matrix
alone. The same factor turns independent standard Gaussian samples
$\mathbf z$ into samples $\mathbf L\mathbf z$ with covariance
$\mathbf L\mathbf L^\top=\mathbf A$.

Geometrically, the signs of the eigenvalues set the shape of the surface
$z=\mathbf{x}^\top\mathbf{A}\mathbf{x}$, seen along the eigenvector axes. All
positive eigenvalues give a bowl (PD, a strict minimum at the origin), a zero
eigenvalue gives a trough that is flat along its eigenvector (PSD, not
strict), and mixed signs give a saddle (indefinite).
:numref:`fig_mdl-la-psd` shows all three.

![The quadratic-form surface $z=\mathbf{x}^\top\mathbf{A}\mathbf{x}$ for three diagonal matrices takes its shape from the eigenvalue signs: a bowl when all $\lambda_i>0$ (positive definite), a trough when one $\lambda_i=0$ (positive semidefinite), and a saddle for mixed signs (indefinite).](../img/mdl-la-psd.svg)
:label:`fig_mdl-la-psd`

Does the rank condition for Gram matrices show up in their eigenvalues? We
build $\mathbf X^\top\mathbf X$ from a tall $\mathbf{X}$ with independent
columns, which should be PD, and from one whose second column is twice the
first, which should be PSD with a zero eigenvalue.

```{.python .input #eigendecomposition-psd-gram}
X_full = np.array([[1., 0.], [1., 1.], [0., 1.], [2., 1.]])  # rank 2
X_dep = np.array([[1., 2.], [1., 2.], [0., 0.], [2., 4.]])   # col 2 = 2*col 1
for X, name in [(X_full, 'full column rank'), (X_dep, 'dependent columns')]:
    eigvals = np.linalg.eigvalsh(X.T @ X).round(4)
    eigvals[np.abs(eigvals) < 1e-9] = 0.0  # clean up tiny round-off / -0.0
    print(f'{name}: eigenvalues of X^T X = {eigvals}')
```

With independent columns both eigenvalues are positive, so the Gram matrix is
PD. With dependent columns one eigenvalue is zero: the form vanishes along
$(2,-1)$, the direction that $\mathbf X$ maps to zero.

### The Rayleigh Quotient
:label:`subsec_mdl-rayleigh`

Identity :eqref:`eq_mdl-quadform` does more than test signs: it identifies the
largest and smallest eigenvalues as the solutions of optimization problems.
Define the *Rayleigh quotient* of a symmetric $\mathbf{A}$,

$$
R(\mathbf{x}) = \frac{\mathbf{x}^\top\mathbf{A}\mathbf{x}}{\mathbf{x}^\top\mathbf{x}},
\qquad \mathbf{x}\neq\mathbf 0 .
$$

It is the value of the quadratic form per unit squared length, and on an
eigenvector it equals the eigenvalue. In this subsection the eigenvalues are
ordered by their signed values, $\lambda_1\ge\cdots\ge\lambda_n$
(:numref:`subsec_mdl-power-iteration` orders them by modulus).

**Proposition (Rayleigh).** *For symmetric $\mathbf{A}$ with eigenvalues
$\lambda_1\ge\cdots\ge\lambda_n$,*

$$
\lambda_1 = \max_{\mathbf{x}\neq\mathbf 0} R(\mathbf{x}),
\qquad
\lambda_n = \min_{\mathbf{x}\neq\mathbf 0} R(\mathbf{x}),
$$

*attained at the corresponding eigenvectors.*

**Proof.** Scaling $\mathbf{x}$ leaves $R$ unchanged, so restrict to unit
vectors. With $\mathbf{y}=\mathbf{V}^\top\mathbf{x}$ we have
$\|\mathbf y\|=\|\mathbf x\|=1$ and, by :eqref:`eq_mdl-quadform`,
$R(\mathbf x)=\sum_i\lambda_i y_i^2$: an average of the eigenvalues with
non-negative weights $y_i^2$ that sum to 1. An average lies between the
smallest and the largest value, so $\lambda_n\le R(\mathbf x)\le\lambda_1$,
and the bounds are attained by putting all weight on one coordinate, at
$\mathbf x=\mathbf v_1$ or $\mathbf x=\mathbf v_n$. $\blacksquare$

For $\mathbf S$ the proposition predicts that $R$ ranges over $[1,3]$, with
the maximum at the eigenvector direction $45^\circ$ and the minimum at
$135^\circ$. The cell sweeps a unit vector around half the circle and checks
both the values and the angles.

```{.python .input #mdl-eigendecomposition-the-rayleigh-quotient-eigenvalues-as-extreme-stretches}
S = np.array([[2., 1.], [1., 2.]])
thetas = np.linspace(0, np.pi, 721)   # R has period pi: x and -x agree
xs = np.stack([np.cos(thetas), np.sin(thetas)])   # unit vectors, all angles
R = np.sum(xs * (S @ xs), axis=0)     # R(x) = x^T S x  (denominator is 1)
print(f'sweep of R:  min = {R.min():.6f}, max = {R.max():.6f}')
print(f'eigvalsh:    {np.linalg.eigvalsh(S).round(6)}')
print(f'argmax angle = {thetas[R.argmax()]:.4f}, '
      f'argmin angle = {thetas[R.argmin()]:.4f}')
print(f'pi/4 = {np.pi / 4:.4f}, 3 pi/4 = {3 * np.pi / 4:.4f}')
```

The sweep never leaves $[1,3]$, its extremes are the two eigenvalues, and they
occur at $\pi/4$ and $3\pi/4$, the eigenvector directions. Every direction in
between mixes the two eigenvalues in proportion to its squared
eigen-coordinates, as the proof says.

Maximizing over subspaces recovers every eigenvalue in between: the
*Courant--Fischer theorem* states that
$\lambda_j=\max_{\dim U=j}\ \min_{\mathbf 0\neq\mathbf x\in U}R(\mathbf x)$,
the best guaranteed stretch over all $j$-dimensional subspaces $U$
(Exercise :exref:`ex_mdl-eigendecomposition-computed-eigenvalue-accuracy`
proves it).

Two uses of the extreme case recur in the book. First, for every
matrix $\mathbf A$, the largest Rayleigh quotient of $\mathbf A^\top\mathbf A$
is $\max_{\mathbf x\neq\mathbf 0}\|\mathbf A\mathbf x\|^2/\|\mathbf x\|^2$, the
square of the largest factor by which $\mathbf A$ stretches a vector. Second,
gradient descent on a convex quadratic $f(\mathbf x)=\tfrac12\mathbf x^\top\mathbf A\mathbf x$
with $\mathbf A\succ0$ updates $\mathbf x\leftarrow(\mathbf I-\eta\mathbf A)\mathbf x$.
By :eqref:`eq_mdl-eig-power-expansion` each eigen-coordinate is multiplied by
$1-\eta\lambda_i$ per step, so the iteration converges from every start if and
only if $|1-\eta\lambda_i|<1$ for all $i$, that is, $0<\eta<2/\lambda_{\max}$.
The largest eigenvalue is the curvature constant $L$ that limits the step
size, and the smallest sets the slowest rate. Their ratio
$\kappa=\lambda_{\max}/\lambda_{\min}$, the *condition number*, measures how
elongated the bowl is.

### Sample Covariance and the Marchenko--Pastur Law
:label:`subsec_mdl-sample-covariance-mp`

When we look for structure in data, we look at the large eigenvalues of its
sample covariance. To tell which eigenvalues are large, we need to know what
they are when the data have no structure at all. Draw $n$ samples of pure
noise in $d$ dimensions, with independent coordinates of mean 0 and
variance 1, so that the true covariance is $\mathbf I$, and stack them as
the rows of $\mathbf X\in\mathbb R^{n\times d}$.
Do the eigenvalues of the sample covariance
$\mathbf C=\tfrac1n\mathbf X^\top\mathbf X$ sit near 1? For a unit vector
$\mathbf w$, the Rayleigh quotient $\mathbf w^\top\mathbf C\mathbf w$ is the
sample variance of the data along $\mathbf w$. The extreme eigenvalues of
$\mathbf C$ are therefore the sample variances of the most and the least
variable directions, the quantities that principal component analysis reads
as structure.

They do not sit near 1. Let $d,n\to\infty$ with the *aspect ratio*
$c=d/n\le1$ fixed. Then the eigenvalues of $\mathbf C$ fill the interval

$$
\bigl[(1-\sqrt c)^2,\ (1+\sqrt c)^2\bigr],
$$

the *Marchenko--Pastur bulk* :cite:`Marchenko.Pastur.1967`, with a density
that the theorem also gives. For $c>1$ the matrix has rank at most $n<d$, so in
addition $d-n$ eigenvalues are exactly zero. With four samples per dimension,
$c=1/4$, the noise eigenvalues already spread over $[0.25,2.25]$: the sample
variance along the most variable direction is more than twice the true
variance, along the least variable one a quarter of it. The cell checks this
for $n=4000$ and $d=1000$.

```{.python .input #mdl-eigendecomposition-sample-covariance-and-the-marchenko-pastur-law}
n, d = 4000, 1000                       # aspect ratio c = d / n = 1/4
X = np.random.randn(n, d)               # pure noise: true covariance I
ev = np.linalg.eigvalsh(X.T @ X / n)
c = d / n
print(f'eigenvalues of C: min {ev.min():.3f}, max {ev.max():.3f}')
print(f'Marchenko-Pastur edges: {(1 - c ** 0.5) ** 2:.3f}, '
      f'{(1 + c ** 0.5) ** 2:.3f}')
```

On every run the smallest and largest eigenvalues land within a few
hundredths of the predicted edges 0.25 and 2.25. An eigenvalue inside the
bulk is therefore indistinguishable from noise. Only an eigenvalue clearly
above the upper edge indicates a direction of real structure, and Exercise
:exref:`ex_mdl-eigendecomposition-spiked-covariance` computes how strong a
direction must be to show up.

## Locating and Computing Eigenvalues

### Gershgorin Discs
:label:`subsec_mdl-gershgorin`

Can we say where the eigenvalues of a matrix lie without computing them? For
a diagonal matrix they are the diagonal entries. If the off-diagonal entries
are small, the eigenvalues should stay near the diagonal, and the following
theorem says how near.

**Theorem (Gershgorin** :cite:`Gershgorin.1931`**).** *Let
$\mathbf{A}=(a_{ij})$ be any $n\times n$ matrix and let
$r_i=\sum_{j\neq i}|a_{ij}|$ be the sum of the absolute values of the
off-diagonal entries in row $i$. Let $\mathcal{D}_i$ be the disc in the complex
plane with centre $a_{ii}$ and radius $r_i$. Then every eigenvalue of
$\mathbf{A}$ lies in the union $\bigcup_i\mathcal{D}_i$.*

**Proof.** Let $\mathbf{A}\mathbf{v}=\lambda\mathbf{v}$ with
$\mathbf{v}\neq\mathbf 0$, and pick the index $i$ of an entry of largest
magnitude, so that $|v_i|\ge|v_j|$ for all $j$ and $v_i\neq0$. Row $i$ of
$\mathbf{A}\mathbf{v}=\lambda\mathbf{v}$ reads

$$
\sum_j a_{ij}v_j = \lambda v_i
\quad\Longrightarrow\quad
(\lambda - a_{ii})\,v_i = \sum_{j\neq i} a_{ij} v_j .
$$

Divide by $v_i$ and take absolute values; since each $|v_j/v_i|\le1$,

$$
|\lambda - a_{ii}|
  = \Bigl|\sum_{j\neq i} a_{ij}\,\tfrac{v_j}{v_i}\Bigr|
  \le \sum_{j\neq i} |a_{ij}| = r_i ,
$$

so $\lambda\in\mathcal{D}_i$. $\blacksquare$

The proof places each eigenvalue in the disc of the row where its eigenvector
is largest, which we do not know in advance; the theorem therefore guarantees
only the union. Consider the non-symmetric matrix

$$
\mathbf A=\begin{pmatrix}4&1&0\\-1&4&0.5\\0.5&0&1\end{pmatrix}.
$$
:eqlabel:`eq_mdl-gershgorin-example`

The discs have centres 4, 4, 1 and radii 1, 1.5, 0.5. The two discs centred
at 4 overlap (one contains the other), and the disc centred at 1 is disjoint
from them, since $|4-1|=3>1.5+0.5$. The eigenvalues are
$3.99\pm0.96i$ and $1.03$: the complex pair lies inside the two discs centred
at 4, and the real eigenvalue inside the small disc.
:numref:`fig_mdl-la-gershgorin` draws the discs and the eigenvalues.

![Gershgorin discs of the $3\times3$ matrix in the text (centres at the diagonal entries, radii the sums of the absolute off-diagonal entries of each row) and its eigenvalues (crosses). The two discs centred at 4 overlap and together hold two eigenvalues, the complex pair; the disjoint disc centred at 1 holds the third.](../img/mdl-la-gershgorin.svg)
:label:`fig_mdl-la-gershgorin`

The split, two eigenvalues in the overlapping pair and one in the small disc,
is not a coincidence. *If a union of $m$ discs is
connected and disjoint from the other discs, it contains exactly $m$
eigenvalues*, counted with multiplicity. To see why, let $\mathbf D$ be the
diagonal part of $\mathbf A$ and consider
$\mathbf A(t)=\mathbf D+t(\mathbf A-\mathbf D)$ for $t\in[0,1]$. At $t=0$ the
eigenvalues are the centres, one per disc. As $t$ grows, the discs of
$\mathbf A(t)$ grow with radii $tr_i$ but keep their centres, so the
components stay separated. The eigenvalues are the roots of a polynomial
whose coefficients depend continuously on $t$, so they move continuously. An
eigenvalue cannot jump the gap between two components, so each component
keeps the number it started with. Here the overlapping pair holds two
eigenvalues and the small disc one. The cell computes centres, radii, and
eigenvalues and checks the theorem.

```{.python .input #eigendecomposition-gershgorin-circle-theorem}
A = np.array([[4.0, 1.0, 0.0],
              [-1.0, 4.0, 0.5],
              [0.5, 0.0, 1.0]])
centres = np.diag(A)
radii = np.abs(A).sum(axis=1) - np.abs(centres)
print('centres:', centres, '  radii:', radii)
for z in np.linalg.eigvals(A):
    print(f'eigenvalue {z:.3f}   in the union of the discs: '
          f'{bool((np.abs(z - centres) <= radii).any())}')
```

All three eigenvalues lie in the union, as the theorem guarantees. The
complex pair is at distance 0.96 from 4, inside the disc of radius 1 but close
to its rim, and well inside the disc of radius 1.5.

A corollary needs no computation at all. If $\mathbf{A}$ is *strictly
diagonally dominant* with positive diagonal, $a_{ii}>r_i$ for every $i$, then
every disc lies in the open right half-plane, 0 is not an eigenvalue, and
$\mathbf A$ is invertible. If such a matrix is also symmetric, its
eigenvalues are real and positive, so it is positive definite. Strictly
diagonally dominant matrices are also the ones on which Gaussian elimination
needs no row exchanges and the simple Jacobi iteration for
$\mathbf A\mathbf x=\mathbf b$ converges.

### Power Iteration
:label:`subsec_mdl-power-iteration`

A matrix with a billion rows cannot be factored, but it can be multiplied by a
vector if it is sparse. What does $\mathbf A^k\mathbf v_0$ tell us about its
eigenvalues? The expansion :eqref:`eq_mdl-eig-power-expansion` answers this.
Let $\mathbf A$ be diagonalizable with eigenvectors $\mathbf v_i$, ordered by
modulus with a *strictly dominant* eigenvalue,
$|\lambda_1|>|\lambda_2|\ge\cdots\ge|\lambda_n|$. (For a real matrix, strict
dominance forces $\lambda_1$ to be real, since a complex eigenvalue would share
its modulus with its conjugate.) Write $\mathbf v_0=\sum_i c_i\mathbf v_i$ and
factor out $\lambda_1^k$:

$$
\mathbf{A}^k\mathbf{v}_0
  = \sum_{i=1}^{n} c_i\lambda_i^k\mathbf{v}_i
  = \lambda_1^k\Bigl(c_1\mathbf{v}_1
       + \sum_{i\ge2} c_i\Bigl(\tfrac{\lambda_i}{\lambda_1}\Bigr)^{k}\mathbf{v}_i\Bigr).
$$
:eqlabel:`eq_mdl-power-iter`

Every ratio $|\lambda_i/\lambda_1|$ is below 1, so if $c_1\neq0$ the bracket
converges to $c_1\mathbf v_1$. The direction of $\mathbf A^k\mathbf v_0$
therefore approaches the eigenvector $\mathbf v_1$, with an error that decays like
$|\lambda_2/\lambda_1|^k$, and the length grows by a factor approaching
$|\lambda_1|$ per step. Multiplying repeatedly and rescaling to length 1 after
each step, so that the numbers stay in range, is *power iteration*
:cite:`Golub.Van-Loan.1996`. For a symmetric matrix, the Rayleigh quotient
of the iterate estimates $\lambda_1$ and converges faster than the direction,
at the squared rate $|\lambda_2/\lambda_1|^{2k}$. The eigenvectors are
orthogonal, so a misalignment of size $\delta$ changes the Rayleigh quotient only by an amount
of order $\delta^2$
(Exercise :exref:`ex_mdl-eigendecomposition-power-iteration-rates`).

:numref:`sec_linear-algebra` computed the iterates of $\mathbf S$ from $\mathbf e_1$:
$(2,1)$, $(5,4)$, $(14,13)$, in general $\bigl((3^k+1)/2,\,(3^k-1)/2\bigr)$.
The expansion explains them. With the unit eigenvectors
$\mathbf v_1=(1,1)/\sqrt2$ and $\mathbf v_2=(1,-1)/\sqrt2$, we have
$\mathbf e_1=(\mathbf v_1+\mathbf v_2)/\sqrt2$, so

$$
\mathbf S^k\mathbf e_1=\frac{3^k\mathbf v_1+\mathbf v_2}{\sqrt2},
\qquad
\tan\angle(\mathbf S^k\mathbf e_1,\mathbf v_1)=3^{-k},
\qquad
3-R(\mathbf S^k\mathbf e_1)=\frac{2}{9^k+1}.
$$

The direction error falls by the factor $|\lambda_2/\lambda_1|=1/3$ per step:
the sine of the angle to $\mathbf v_1$ is 0.316, 0.110, 0.037 for $k=1,2,3$.
The error of the Rayleigh quotient falls by $1/9$: it is 0.2, 0.024, 0.0027.
:numref:`fig_mdl-la-power-iter` plots both on a logarithmic scale.

![Power iteration on $\mathbf S=\left(\begin{smallmatrix}2&1\\1&2\end{smallmatrix}\right)$ from $\mathbf e_1$. The direction error falls by a factor of 3 per step and the error of the eigenvalue estimate by a factor of 9: the squared rate, because the eigenvectors of a symmetric matrix are orthogonal. Dashed lines: $3^{-k}$ and $2\cdot9^{-k}$.](../img/mdl-la-power-iter.svg)
:label:`fig_mdl-la-power-iter`

Two conditions can fail. The start needs $c_1\neq0$; a random start satisfies
this with probability one. And the dominance must be strict. If the dominant
eigenvalues are a complex pair of equal modulus, the dominant part of the
iterate keeps rotating. The ratio of consecutive norms then oscillates forever,
as it did for the running matrix $\mathbf A$ (ratios between 1.4 and 3.5,
period about 13.6 steps). Even then, the average rate
$\|\mathbf A^k\mathbf v_0\|^{1/k}$ converges to the largest modulus.

Which matrices are safe? A matrix with positive entries has a real, simple,
strictly dominant eigenvalue, by the Perron--Frobenius theorem of the next
subsection. The cell runs 200 steps of power iteration on a random $5\times5$
matrix with entries uniform in $[0,1]$, from a random start, and compares the
final norm ratio with the largest eigenvalue modulus.

```{.python .input #eigendecomposition-power-iteration}
A = np.random.rand(5, 5)                 # positive entries
v = np.random.randn(5)
for _ in range(200):
    Av = A @ v
    ratio = np.linalg.norm(Av) / np.linalg.norm(v)
    v = Av / np.linalg.norm(Av)
mods = np.sort(np.abs(np.linalg.eigvals(A)))[::-1]
print(f'norm ratio after 200 steps = {ratio:.10f}')
print(f'largest |eigenvalue|       = {mods[0]:.10f}')
print(f'|lambda_2| / |lambda_1|    = {mods[1] / mods[0]:.3f}')
```

The norm ratio agrees with the largest eigenvalue modulus in every printed
digit, on every run. For such matrices the gap ratio $|\lambda_2/\lambda_1|$
is typically between 0.1 and 0.4, so 200 steps shrink the direction error below
rounding level. With Gaussian entries instead, about four draws in ten have a
complex dominant pair, and the ratio never settles.

### PageRank
:label:`subsec_mdl-pagerank`

For non-negative matrices the dominant eigenvector is often the quantity we
want, and power iteration is how it is computed. A non-negative matrix is
*irreducible* if, in the graph with an edge $j\to i$ whenever $a_{ij}>0$,
every node reaches every other. The **Perron--Frobenius theorem**
:cite:`Horn.Johnson.2012` states that such a matrix has a real, positive
eigenvalue equal to the largest modulus of its eigenvalues, that this
eigenvalue is simple, and that its eigenvector can be chosen with positive
entries. If every entry of the matrix is positive, this
eigenvalue is also strictly dominant: every other eigenvalue has smaller
modulus.

Google's original PageRank :cite:`Brin.Page.1998` is such an eigenvector.
Consider three pages with links $1\to2$, $2\to1$, and $2\to3$. A surfer
follows a random outgoing link; page 3 has none, so we repair it by letting
the surfer jump from page 3 to a uniformly random page. Here $\mathbf P$ is
the repaired link matrix, column-stochastic with column $j$ holding the click
probabilities from page $j$:

$$
\mathbf P=\begin{pmatrix}0&\tfrac12&\tfrac13\\1&0&\tfrac13\\0&\tfrac12&\tfrac13\end{pmatrix}.
$$

The long-run fraction of time on each page is the stationary distribution
$\boldsymbol\pi$ with $\mathbf P\boldsymbol\pi=\boldsymbol\pi$. By hand, rows 1
and 3 give $\pi_1=\tfrac12\pi_2+\tfrac13\pi_3=\pi_3$, and then row 1 gives
$\pi_2=\tfrac43\pi_3$; normalizing to sum 1 gives
$\boldsymbol\pi=(0.3,0.4,0.3)$. Page 2 ranks first because page 1 sends it
every click, while page 2 splits its clicks between pages 1 and 3. The
eigenvalues of $\mathbf P$ are $1$, $-\tfrac23$ and $0$, so power iteration from any distribution converges at the rate $2/3$ per step.

Every column-stochastic matrix has 1 as an eigenvalue of largest modulus.
First, because the columns sum to one,
$\mathbf 1^\top\mathbf P=\mathbf 1^\top$, so 1 is an eigenvalue of
$\mathbf P^\top$, and hence of $\mathbf P$: a matrix and its transpose have
the same characteristic polynomial, since $\det\mathbf X^\top=\det\mathbf X$.
Second, Gershgorin's theorem applied to $\mathbf P^\top$, whose rows are the
columns of $\mathbf P$, puts every eigenvalue in a disc with centre
$p_{ii}\ge0$ and radius $1-p_{ii}$, which lies inside the unit disc.

The web contains link cycles and groups of pages from which the surfer
cannot escape. The repaired matrix can then fail to be irreducible or have
several eigenvalues of modulus 1. A cycle of four pages has eigenvalues
$\pm1$ and $\pm i$, and power iteration on it never settles
(Exercise :exref:`ex_mdl-eigendecomposition-pagerank-damping`). PageRank
therefore iterates the *damped* matrix

$$
\mathbf G=\alpha\mathbf P+(1-\alpha)\tfrac1n\mathbf 1\mathbf 1^\top,
\qquad \alpha=0.85,
$$

which models a surfer who, at each click, jumps to a uniformly random page
with probability $0.15$. Every entry of $\mathbf G$ is positive, so
Perron--Frobenius applies by construction. :citet:`Haveliwala.Kamvar.2003`
show that $|\lambda_2(\mathbf G)|\le\alpha$, so the error of power iteration
shrinks by at least the factor $0.85$ per step. On the three-page example the
eigenvalues of $\mathbf G$ are 1, $-0.567$ and 0, which apart from 1 are
$\alpha$ times those of $\mathbf P$. The stationary distribution moves
slightly toward uniform, to $(0.303,0.394,0.303)$.

At web scale $\mathbf P$ is sparse and stored as one entry per link, and the
dense $\mathbf G$ is never formed. For a distribution $\mathbf p$,
$\mathbf 1^\top\mathbf p=1$ gives
$\mathbf G\mathbf p=\alpha\mathbf P\mathbf p+(1-\alpha)\tfrac1n\mathbf 1$, so
each step needs one sparse product. The repair of pages without links is a
rank-one term handled the same way.

## Iteration and Stability

### The Spectral Radius
:label:`subsec_mdl-spectral-radius`

Power iteration measured one number, the largest eigenvalue modulus. It has a
name, the *spectral radius* $\rho(\mathbf A)=\max_i|\lambda_i|$, and it
controls the long-run growth rate of the powers of one fixed matrix. Gelfand's
formula makes this precise: for every square matrix and every matrix norm,

$$
\lim_{k\to\infty}\|\mathbf A^k\|^{1/k}=\rho(\mathbf A).
$$

Exercise :exref:`ex_mdl-eigendecomposition-spectral-radius-norms` proves the
easy half, $\rho(\mathbf A)\le\|\mathbf A^k\|^{1/k}$ for every $k$. The formula
says nothing about a fixed finite $k$, and we have already seen two matrices
where that matters. The shear has $\rho=1$, yet its powers
$\left(\begin{smallmatrix}1&k\\0&1\end{smallmatrix}\right)$ grow without bound,
so $\rho\le1$ does not bound the iterates. The Jordan block $\mathbf J_2(0.99)$
has $\rho<1$, yet its powers reach norm 37 first, so $\rho<1$ does not bound
the transient. When do the eigenvalues control every step, not only the
limit?

### Normal Matrices
:label:`subsec_mdl-normal-matrices`

Both counterexamples, the shear and the Jordan block, have too few
eigenvectors to form a basis. We therefore look first at the opposite
extreme, matrices with an orthonormal basis of eigenvectors. Such a matrix
acts on perpendicular directions independently, so its eigenvalues should
tell the whole story. A matrix is *normal* when it commutes with its
transpose, $\mathbf A^\top\mathbf A=\mathbf A\mathbf A^\top$ (for complex
matrices, with the conjugate transpose $\mathbf A^*$ in place of
$\mathbf A^\top$). Symmetric
matrices, orthogonal matrices, and the rotation--scaling blocks
$r\mathbf Q_\theta$ are normal. The running matrix
$\mathbf A=\left(\begin{smallmatrix}1&2\\-1&3\end{smallmatrix}\right)$ is not:
$\mathbf A\mathbf A^\top=\left(\begin{smallmatrix}5&5\\5&10\end{smallmatrix}\right)$
while $\mathbf A^\top\mathbf A=\left(\begin{smallmatrix}2&-1\\-1&13\end{smallmatrix}\right)$.
This is why its iterates did not move on circles.

The *complex spectral theorem* :cite:`Horn.Johnson.2012` states that a matrix
is normal if and only if it has an orthonormal eigenbasis over $\mathbb C$,
$\mathbf A=\mathbf U\boldsymbol\Lambda\mathbf U^*$ with $\mathbf U$ unitary
($\mathbf U^*\mathbf U=\mathbf I$). Its proof is the induction of part (iii)
above with conjugate transposes. The one new step is that for a normal matrix
$\mathbf A\mathbf v=\lambda\mathbf v$ implies $\mathbf A^*\mathbf v=\bar\lambda\mathbf v$,
because normality gives $\|(\mathbf A-\lambda\mathbf I)\mathbf x\|=\|(\mathbf A-\lambda\mathbf I)^*\mathbf x\|$
for every $\mathbf x$.

Write
$\|\mathbf X\|_2=\max_{\|\mathbf x\|=1}\|\mathbf X\mathbf x\|$ for the
*spectral norm*, the largest factor by which $\mathbf X$ stretches a vector.
Since a unitary change of basis preserves lengths, for a normal matrix

$$
\|\mathbf A^k\|_2
=\|\mathbf U\boldsymbol\Lambda^k\mathbf U^*\|_2
=\max_i|\lambda_i|^k
=\rho(\mathbf A)^k .
$$
:eqlabel:`eq_mdl-normal-power-norm`

For a normal matrix the eigenvalues control the largest possible
amplification at every step, not only its long-run rate. A diagonalizable
matrix that is not normal has a non-orthogonal eigenvector matrix in
$\mathbf A=\mathbf V\boldsymbol\Lambda\mathbf V^{-1}$, and submultiplicativity
of the norm gives only

$$
\|\mathbf A^k\|_2
\le \underbrace{\|\mathbf V\|_2\|\mathbf V^{-1}\|_2}_{\kappa_2(\mathbf V)}
   \rho(\mathbf A)^k .
$$
:eqlabel:`eq_mdl-nonnormal-power-bound`

The factor $\kappa_2(\mathbf V)$, the condition number of the eigenvector
matrix, is 1 for an orthogonal $\mathbf V$ and large when eigenvectors are
nearly parallel. The growth it allows comes from cancellation. When
eigenvectors are nearly parallel, an ordinary input can only be written as a combination of them with
large coefficients that nearly cancel. The modes then decay at different rates,
the cancellation is undone, and the iterate grows before the common decay
takes over.

### Transient Growth
:label:`subsec_mdl-transient-growth`

The smallest example of such growth is the upper-triangular matrix

$$
\mathbf M=\begin{pmatrix}0.9&4\\0&0.8\end{pmatrix}.
$$
:eqlabel:`eq_mdl-nonnormal-example`

Its eigenvalues are its diagonal entries 0.9 and 0.8, both inside the unit
circle, and they are distinct, so $\mathbf M$ is diagonalizable. We find its
eigenvectors and expand $\mathbf e_2=(0,1)$ in them. For $\lambda=0.9$, the
second row of $\mathbf M-0.9\mathbf I$ reads $-0.1v_2=0$, so
$\mathbf v_1=(1,0)$. For $\lambda=0.8$, the first row of $\mathbf M-0.8\mathbf I$
reads $0.1v_1+4v_2=0$, so $\mathbf v_2=(-40,1)$. The two lines are
$\arctan(1/40)\approx1.4^\circ$ apart. The unit input is
$\mathbf e_2=40\mathbf v_1+\mathbf v_2$, and by
:eqref:`eq_mdl-eig-power-expansion`

$$
\mathbf M^k\mathbf e_2=40\cdot0.9^k\,\mathbf v_1+0.8^k\,\mathbf v_2
=\begin{pmatrix}40\,(0.9^k-0.8^k)\\0.8^k\end{pmatrix}.
$$
:eqlabel:`eq_mdl-transient-closed-form`

A vector of length 1 is the difference of two components of length about 40.
Both decay, but at different rates, so their difference first grows. The
first entry $40(0.9^k-0.8^k)$ peaks where its derivative in $k$ vanishes,
where $(9/8)^k=\ln0.8/\ln0.9$, at $k\approx6.4$, with value about 10.8.
:numref:`fig_mdl-la-transient-cancel` draws the two components at $k=0$ and
$k=6$. The bound :eqref:`eq_mdl-nonnormal-power-bound` with unit
eigenvectors has $\kappa_2(\mathbf V)=80$ and gives
$\|\mathbf M^6\|_2\le80\cdot0.9^6\approx42.5$. The bound holds, but it is four
times the actual value 10.8.

![Growth of $\mathbf M^k\mathbf e_2$ before decay. The unit vector $\mathbf e_2$ is the difference of two eigen-components of length about 40 that nearly cancel (left). Six steps later the components have decayed by $0.9^6$ and $0.8^6$, the cancellation is gone, and the result is ten times longer (right). The vertical axis is stretched about 30-fold; drawn to scale, $\mathbf v_1$ and $\mathbf v_2$ are $1.4^\circ$ apart.](../img/mdl-la-transient-cancel.svg)
:label:`fig_mdl-la-transient-cancel`

The cell checks the derivation and compares $\mathbf M$ with the diagonal
matrix $\boldsymbol\Lambda=\operatorname{diag}(0.9,0.8)$, which has the same
eigenvalues and is normal.

```{.python .input #mdl-eigendecomposition-nonnormal-transient}
M = np.array([[0.9, 4.0], [0.0, 0.8]])
Lam = np.diag([0.9, 0.8])                   # same eigenvalues, normal
lam, V = np.linalg.eig(M)                   # unit eigenvectors as columns
print('eigenvalues:', lam)
print('coefficients of e_2 on the unit eigenvectors:',
      np.linalg.solve(V, [0., 1.]))
ks = np.arange(61)
norm_M = [np.linalg.norm(np.linalg.matrix_power(M, k), 2) for k in ks]
norm_L = [np.linalg.norm(np.linalg.matrix_power(Lam, k), 2) for k in ks]
k_peak = int(np.argmax(norm_M))
print(f'max ||M^k|| = {norm_M[k_peak]:.3f} at k = {k_peak};  '
      f'max ||Lambda^k|| = {max(norm_L):.3f}')
print(f'kappa_2(V) = {np.linalg.cond(V):.1f}')
```

On the unit eigenvectors, $\mathbf e_2$ has two coefficients of size about 40,
the lengths of $40\mathbf v_1$ and $\mathbf v_2$. The largest
$\|\mathbf M^k\|_2$ is 10.8 at $k=6$, while $\boldsymbol\Lambda$ never exceeds
its starting norm 1. With the same eigenvalues, the normal matrix contracts
from the first step, while $\mathbf M$ first grows ten-fold.
:numref:`fig_mdl-la-transient-growth` (left) shows the two curves on a
logarithmic scale; after the peak both fall with the same slope $\log0.9$, as
Gelfand's formula requires.

![Left: $\|\mathbf M^k\|_2$ and $\|\boldsymbol\Lambda^k\|_2$ for the same eigenvalues 0.9 and 0.8; $\mathbf M$ grows to 10.8 at $k=6$ before both decay with slope $\log 0.9$. Right: the $\varepsilon$-pseudospectra for $\varepsilon=0.05,0.1,0.2,0.4$; around $\boldsymbol\Lambda$ they are discs of radius $\varepsilon$, around $\mathbf M$ they reach far past the unit circle.](../img/mdl-la-transient-growth.svg)
:label:`fig_mdl-la-transient-growth`

### Pseudospectra
:label:`subsec_mdl-pseudospectra`

A second symptom of non-normality is that the eigenvalues are sensitive to
perturbations of the matrix. For $\varepsilon>0$, the
*$\varepsilon$-pseudospectrum* collects every number that becomes an
eigenvalue under a perturbation $\mathbf E$ of size at most $\varepsilon$. It
has an equivalent description through the *resolvent*
$(z\mathbf I-\mathbf A)^{-1}$:

$$
\Lambda_\varepsilon(\mathbf A)
=\{z\in\mathbb C:z\text{ is an eigenvalue of }\mathbf A+\mathbf E
       \text{ for some }\|\mathbf E\|_2\le\varepsilon\}
=\{z\in\mathbb C:\|(z\mathbf I-\mathbf A)^{-1}\|_2\ge1/\varepsilon\},
$$
:eqlabel:`eq_mdl-pseudospectrum`

with the convention that the resolvent norm is infinite at an eigenvalue.
A large resolvent norm at $z$ means that a small perturbation moves an
eigenvalue to $z$, and Exercise
:exref:`ex_mdl-eigendecomposition-pseudospectra-transient-growth` proves that
the two sets agree. For a normal matrix the resolvent norm is the reciprocal
of the distance from $z$ to the nearest eigenvalue, so $\Lambda_\varepsilon$
is the union of discs of radius $\varepsilon$ around the eigenvalues. For
$\mathbf M$ it is far larger. The cell evaluates both resolvents at $z=1$, a
point on the unit circle.

```{.python .input #mdl-eigendecomposition-pseudospectra}
I = np.eye(2)
print(f'||(I - M)^-1||      = {np.linalg.norm(np.linalg.inv(I - M), 2):.1f}')
print(f'||(I - Lambda)^-1|| = {np.linalg.norm(np.linalg.inv(I - Lam), 2):.1f}')
```

For $\boldsymbol\Lambda$ the resolvent norm at $z=1$ is 10, the reciprocal of
the distance 0.1 to the eigenvalue 0.9. For $\mathbf M$ it is 200.3, so a
perturbation of size $1/200.3\approx0.005$ already puts an eigenvalue of
$\mathbf M$ on the unit circle, where $\boldsymbol\Lambda$ needs 0.1. The right
panel of :numref:`fig_mdl-la-transient-growth` draws the level sets. Where
pseudospectra reach far beyond the eigenvalues, the computed eigenvalues may
be inaccurate, since rounding errors act as small perturbations. The iterates
may also grow transiently, as they did for $\mathbf M$.

### Products of Jacobians
:label:`subsec_mdl-jacobian-products`

A recurrent network computes its hidden state as
$\mathbf h_t=\phi(\mathbf W\mathbf h_{t-1}+\cdots)$, with the same weights
$\mathbf W$ at every step, and backpropagation through time
multiplies the per-step Jacobians
$\mathbf D_t=\partial\mathbf h_t/\partial\mathbf h_{t-1}$:

$$
\nabla_{\mathbf{h}_0}\mathcal{L}
   = \mathbf{D}_1^\top\mathbf{D}_2^\top\cdots\mathbf{D}_T^\top
   \,\nabla_{\mathbf{h}_T}\mathcal{L}.
$$

The gradient that reaches the first state is the last one multiplied by $T$
matrices. These matrices differ from step to step, because the nonlinearity
$\phi$ is evaluated at different points, so the theory of powers of one matrix
does not apply. The spectral radii of the factors say nothing about the
product. Take

$$
\mathbf D_1=\begin{pmatrix}0&2\\0&0\end{pmatrix},
\qquad
\mathbf D_2=\begin{pmatrix}0&0\\2&0\end{pmatrix},
\qquad
\mathbf D_2\mathbf D_1=\begin{pmatrix}0&0\\0&4\end{pmatrix}.
$$

Both factors are nilpotent with $\rho=0$, and their product has $\rho=4$:
$\mathbf D_1$ moves the second coordinate into the first, doubled, and
$\mathbf D_2$ moves it back, doubled again.

Three quantities describe iterated linear maps, each answering a different
question. The spectral
radius gives the asymptotic rate of the powers of one fixed map. The spectral
norm bounds every finite product,
$\|\mathbf D_T\cdots\mathbf D_1\|_2\le\prod_t\|\mathbf D_t\|_2$, here
$4\le2\cdot2$. The *top Lyapunov exponent*

$$
\mu=\lim_{T\to\infty}\frac1T\log\|\mathbf D_T\cdots\mathbf D_1\mathbf x_0\|
$$

gives the average growth rate per step of a long product of changing maps.
For independent, identically distributed random factors (under a mild moment
condition), :citet:`Furstenberg.Kesten.1960` show that the same limit with the
matrix norm in place of the vector exists with probability one; for Gaussian
factors it equals $\mu$ for every fixed $\mathbf x_0$. In general one reads the
definition as a $\limsup$.

A gradient that is multiplied by maps with $\mu>0$ explodes, and one with
$\mu<0$ vanishes :cite:`bengio1994learning,Pascanu.Mikolov.Bengio.2013`. The
standard remedies act on these quantities rather than on eigenvalues.
Orthogonal initialization :cite:`Saxe.McClelland.Ganguli.2014` makes every
singular value of the initial weight matrix equal to 1, so a linear recurrence
preserves the length of every vector, a stronger property than $\rho=1$.
Gradient clipping caps the norm of the gradient after it has been computed. Gates, as in LSTMs and GRUs, change
the Jacobians themselves so that some directions pass through with factors
close to 1.

### Random Matrices at Initialization
:label:`subsec_mdl-random-matrices`

How large are these quantities for the random weight matrices of an
untrained network? Let $\mathbf A$ be $n\times n$ with independent standard
Gaussian entries, the real *Ginibre ensemble* :cite:`Ginibre.1965`. As
$n\to\infty$, the eigenvalues divided by $\sqrt n$ fill the unit disc in the
complex plane uniformly: this is the *circular law*, which
:citet:`Tao.Vu.2010` prove for independent, identically distributed entries
of mean 0 and variance 1. The circular law describes where almost all eigenvalues are; it does not
exclude a few outliers. That none escapes, so that
$\rho(\mathbf A)/\sqrt n\to1$ with probability one, is a separate theorem
:cite:`Geman.1986`. The largest singular value behaves differently:
$\sigma_{\max}(\mathbf A)/\sqrt n\to2$ :cite:`Yin.Bai.Krishnaiah.1988`. The
cell checks both for $n=1000$.

```{.python .input #mdl-eigendecomposition-random-matrices-at-initialization}
n = 1000
A = np.random.randn(n, n)
print(f'rho(A) / sqrt(n)       = {np.abs(np.linalg.eigvals(A)).max() / n ** 0.5:.3f}')
print(f'sigma_max(A) / sqrt(n) = {np.linalg.norm(A, 2) / n ** 0.5:.3f}')
```

For large $n$ the spectral radius is within a few percent of $\sqrt n$ and
the largest stretch close to $2\sqrt n$: a random matrix stretches some
vectors twice as much as its eigenvalues suggest. Initialization schemes divide
random weights by $\sqrt{\textrm{fan-in}}$ for a different reason: with
entries of variance $1/n$, $\mathbb{E}\|\mathbf A\mathbf x\|^2=\|\mathbf x\|^2$
for every fixed $\mathbf x$, so squared lengths are preserved on average.
A spectral radius near 1 is a side effect, and by
:numref:`subsec_mdl-jacobian-products` it does not control a product of such
matrices. In fact the product of
independent factors with this scaling shrinks a typical vector, $\mu<0$, even
though every factor preserves squared length on average
(Exercise :exref:`ex_mdl-eigendecomposition-lyapunov-random-products`).

We can now compute the eigenvalues of a small matrix by hand and write
$\mathbf A^k\mathbf x$ in closed form whenever an eigenbasis exists. We can
read the sign of a quadratic form off the eigenvalues of a symmetric matrix,
bound eigenvalues without computing them, and compute the dominant one by
iteration. Two questions remain open. Eigenvalues say little about
finite-step stretch: $\mathbf M$ has $\rho=0.9$ and grows ten-fold before it
decays, and a random matrix stretches twice as much as its spectral radius.
And eigenvalues do not exist for rectangular matrices at all. The singular
values of :numref:`sec_mdl-svd-low-rank` answer both questions.

## Summary

Repeated application of one matrix, as in a Markov chain, a deep linear
network, or gradient descent on a quadratic, is governed by its eigenvalues.
In eigenvector coordinates, $\mathbf A^k\mathbf x=\sum_i c_i\lambda_i^k\mathbf v_i$,
so each coordinate evolves as a scalar power. The surfer's distribution
approaches $(1/3,2/3)$ with an error that shrinks by a factor of 4 per click.
Over $\mathbb C$, a matrix is diagonalizable if and only if every eigenvalue
has as many independent eigenvectors as its multiplicity as a root, and a
real eigenbasis needs real eigenvalues as well. Otherwise Jordan blocks add
polynomial factors, and complex pairs act as rotation--scaling blocks.
Symmetric matrices always have real eigenvalues and an orthonormal
eigenbasis. Their quadratic forms are then weighted sums of squares, their
definiteness is decided by the eigenvalue signs, and the extreme eigenvalues
are the extremes of the Rayleigh quotient. The characteristic polynomial, Gershgorin
discs, and power iteration compute or locate eigenvalues, from $2\times2$
matrices to PageRank. Eigenvalues mislead for defective, rotating, and
non-normal matrices at finite times, and for products of different maps,
where singular values and Lyapunov exponents take their place.

## Exercises

1. **Trace, determinant, and definiteness.** The identities
   :eqref:`eq_mdl-det-eig` and :eqref:`eq_mdl-tr-eig` fix both eigenvalues of
   a $2\times2$ matrix through their sum and product.
    1. Find the eigenvalues of
       $\left(\begin{smallmatrix}4&1\\1&4\end{smallmatrix}\right)$ from its
       trace and determinant alone, without forming the characteristic
       polynomial. Then find its eigenvectors. Do the same for the surfer's
       matrix $\mathbf P$, whose trace is $5/4$ and determinant $1/4$.
    1. Show that a symmetric $2\times2$ matrix is positive definite if and
       only if its trace and its determinant are both positive. Which sign
       conditions on trace and determinant characterize positive
       semidefiniteness?
    1. Each of the matrices
       $\left(\begin{smallmatrix}2&1\\1&2\end{smallmatrix}\right)$,
       $\left(\begin{smallmatrix}1&2\\2&1\end{smallmatrix}\right)$, and
       $\left(\begin{smallmatrix}1&1\\1&1\end{smallmatrix}\right)$ is the
       Hessian of the quadratic $f(\mathbf{x})=\tfrac12\mathbf{x}^\top\mathbf{A}\mathbf{x}$
       at its critical point $\mathbf{0}$. Use (b) to decide for each $f$
       whether the origin is a strict minimum, a minimum that is not strict,
       or a saddle point. For the saddle point, give a direction along which
       $f$ decreases.
    1. Show that the Cholesky factor $\mathbf{L}$ of
       $\left(\begin{smallmatrix}a&b\\b&c\end{smallmatrix}\right)$ has the
       diagonal entries $\sqrt a$ and $\sqrt{(ac-b^2)/a}$. Why do the
       conditions under which a factor with positive diagonal exists coincide
       with the test of (b)?
   :exlabel:`ex_mdl-eigendecomposition-trace-determinant`

1. **Rank and zero eigenvalues.** The section shows that the rank of a
   diagonalizable matrix equals the number of its nonzero eigenvalues,
   counted with multiplicity.
    1. Find the rank and the eigenvalues of
       $\mathbf{N}=\left(\begin{smallmatrix}0&1\\0&0\end{smallmatrix}\right)$.
       Which step of the argument for diagonalizable matrices fails for
       $\mathbf{N}$?
    1. Let $g_0$ be the geometric multiplicity of the eigenvalue $0$ of an
       $n\times n$ matrix $\mathbf{A}$, with $g_0=0$ when $0$ is not an
       eigenvalue. Show that $\operatorname{rank}\mathbf{A}=n-g_0$ for every
       square matrix. Conclude that the rank is at least the number of nonzero
       eigenvalues counted with algebraic multiplicity, with equality if and
       only if the eigenvalue $0$ is not defective.
    1. The *singular values* of a matrix are
       $\sigma_i=\sqrt{\lambda_i(\mathbf{A}^\top\mathbf{A})}$, the square roots
       of the eigenvalues of the PSD matrix $\mathbf A^\top\mathbf A$. Find the
       singular values of $\mathbf{N}$ and of the Jordan block
       $\mathbf{J}_m(0)$ of :numref:`subsec_mdl-jordan`. Compare the number of
       nonzero singular values with the rank and with the number of nonzero
       eigenvalues.
   :exlabel:`ex_mdl-eigendecomposition-rank-zero-eigenvalues`

1. **A defective limit.** For $\varepsilon\ge0$, let

   $$
   \mathbf{A}_\varepsilon=\begin{pmatrix}2&1\\\varepsilon&2\end{pmatrix}.
   $$

    1. Find the eigenvalues and eigenvectors of $\mathbf{A}_\varepsilon$ for
       $\varepsilon>0$ and for $\varepsilon=0$. Which multiplicities differ at
       $\varepsilon=0$?
    1. Collect the unit eigenvectors of $\mathbf{A}_\varepsilon$ in the columns
       of $\mathbf{V}_\varepsilon$. How do the angle between them and the
       condition number $\kappa_2(\mathbf{V}_\varepsilon)$ of
       :eqref:`eq_mdl-nonnormal-power-bound` behave as $\varepsilon\to0$?
       Show that $\sqrt\varepsilon\,\kappa_2(\mathbf V_\varepsilon)$ tends to
       a constant as $\varepsilon\to0$, and find it.
    1. By how much do the eigenvalues of $\mathbf{A}_0$ move when its
       lower-left entry changes from $0$ to $\varepsilon=10^{-10}$? Compare
       with the eigenvalues of the symmetric matrix
       $2\mathbf{I}+\varepsilon\left(\begin{smallmatrix}0&1\\1&0\end{smallmatrix}\right)$.
       This is the smallest case of Exercise
       :exref:`ex_mdl-eigendecomposition-computed-eigenvalue-accuracy`a.
   :exlabel:`ex_mdl-eigendecomposition-defective-limit`

1. **Spectral radius and norms.** The spectral radius bounds every power of
   a matrix from below, and for normal matrices it also bounds them from
   above.
    1. Show that $\rho(\mathbf{A})\le\|\mathbf{A}^k\|_2^{1/k}$ for every
       square matrix $\mathbf{A}$ and every $k\ge1$.
    1. The chain :eqref:`eq_mdl-normal-power-norm` uses two facts without
       proof. Show that $\|\mathbf{U}\mathbf{X}\mathbf{U}^*\|_2=\|\mathbf{X}\|_2$
       for every unitary $\mathbf{U}$ and every square $\mathbf{X}$, and that
       the spectral norm of a diagonal matrix is the largest modulus of its
       diagonal entries.
    1. Show that the converse of :eqref:`eq_mdl-normal-power-norm` fails:
       find a $3\times3$ matrix that is not normal but satisfies
       $\|\mathbf{A}^k\|_2=\rho(\mathbf{A})^k$ for every $k\ge1$.
    1. For a given $t>0$, construct a diagonalizable $2\times2$ matrix
       with spectral radius $1/2$ and $\|\mathbf{A}\|_2>t$.
   :exlabel:`ex_mdl-eigendecomposition-spectral-radius-norms`

1. **The centering matrix.** Let $\mathbf{1}\in\mathbb{R}^n$ be the all-ones
   vector and $\mathbf{H}=\mathbf{I}-\tfrac1n\mathbf{1}\mathbf{1}^\top$.
    1. Show that $n\mathbf{I}-\mathbf{1}\mathbf{1}^\top$ is positive
       semidefinite by writing
       $\mathbf{x}^\top(n\mathbf{I}-\mathbf{1}\mathbf{1}^\top)\mathbf{x}$ as a
       sum of squares of the differences $x_i-x_j$. Then find all eigenvalues
       of $n\mathbf{I}-\mathbf{1}\mathbf{1}^\top$ and their eigenvectors.
    1. Show that $\mathbf{H}$ is symmetric with $\mathbf{H}^2=\mathbf{H}$, and
       deduce its eigenvalues and its rank from (a).
    1. Let the rows of $\mathbf{X}\in\mathbb{R}^{n\times d}$ be $n$ samples.
       Subtracting the sample mean from every row gives $\mathbf{H}\mathbf{X}$,
       so that the sample covariance is
       $\mathbf{C}=\tfrac1n\mathbf{X}^\top\mathbf{H}\mathbf{X}$.
       Show that $\mathbf{C}\succeq0$ and
       $\operatorname{rank}\mathbf{C}\le n-1$. How many zero
       eigenvalues does the sample covariance of $512$-dimensional embeddings
       computed from a minibatch of $256$ examples have at least? This is the
       case $c=d/n>1$ of :numref:`subsec_mdl-sample-covariance-mp`. Which
       operations on $\mathbf{C}$ does it rule out?

   *Adapted from Gilbert Strang,
   [MIT 18.065 Problem Sets](https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/cd4c57b7e61b8ca9fdd3511a30aca052_MIT18_065S18PSets.pdf),
   Problems for Lecture 13, Problem 4.*
   :exlabel:`ex_mdl-eigendecomposition-centering-matrix`

1. [code] **Power-iteration rates.** Let $\mathbf{S}$ be the matrix
   of :numref:`fig_mdl-la-power-iter`.
    1. By :eqref:`eq_mdl-power-iter`, the misalignment between the iterate and
       $\mathbf{v}_1$ decays at the rate $|\lambda_2/\lambda_1|=1/3$. How many
       iterations reduce it by a factor of $100$?
    1. Let $\mathbf{A}$ be symmetric with unit eigenvector $\mathbf{v}_1$ for
       $\lambda_1$, and let $\mathbf{x}=\mathbf{v}_1+\delta\mathbf{u}$ for a unit
       vector $\mathbf{u}$ orthogonal to $\mathbf{v}_1$. Show that the Rayleigh
       quotient of :numref:`subsec_mdl-rayleigh` satisfies
       $R(\mathbf{x})-\lambda_1=O(\delta^2)$, with no term linear in $\delta$.
       At which rate does the Rayleigh quotient of the power iterates
       therefore approach $\lambda_1$? Where does the argument use that
       $\mathbf A$ is symmetric?
    1. Repeat the experiment of :numref:`fig_mdl-la-power-iter` for
       $\left(\begin{smallmatrix}1&0.1\\0.1&1\end{smallmatrix}\right)$, which
       has the same eigenvectors as $\mathbf S$ and the eigenvalues 1.1 and
       0.9, from a random start. Plot the sine of the angle to $\mathbf v_1$
       and the error of the Rayleigh quotient on a logarithmic scale for 30
       steps. Predict both slopes and the number of steps each error needs to
       shrink by a factor of 100, and explain why this matrix converges so much
       more slowly than $\mathbf S$.
   :exlabel:`ex_mdl-eigendecomposition-power-iteration-rates`

1. [code] **Accuracy of computed eigenvalues.** ● The algorithm behind
   `np.linalg.eigvals` is backward stable: it returns the exact eigenvalues of
   a matrix $\mathbf{A}+\mathbf{E}$ with $\|\mathbf{E}\|_2$ of the order of
   $\epsilon_{\textrm{mach}}\|\mathbf{A}\|_2$, where
   $\epsilon_{\textrm{mach}}\approx2.2\times10^{-16}$ in double precision
   :cite:`Golub.Van-Loan.1996`. Exercise
   :exref:`ex_mdl-eigendecomposition-defective-limit` shows how strongly such
   a perturbation can move a defective eigenvalue.
    1. Let $\mathbf{e}_1,\ldots,\mathbf{e}_m$ be the standard basis vectors.
       Show that $\mathbf{J}_m(\lambda)+\varepsilon\,\mathbf{e}_m\mathbf{e}_1^\top$,
       which places $\varepsilon$ in the lower-left corner of the Jordan
       block, has the eigenvalues $\lambda+\varepsilon^{1/m}\omega$, where
       $\omega$ ranges over the $m$-th roots of unity.
    1. For $m=1,\ldots,4$, form $\mathbf{Q}\mathbf{J}_m(2)\mathbf{Q}^\top$ with
       an orthogonal $\mathbf{Q}$ taken from the QR factorization of a
       Gaussian matrix, and record the median over $20$ draws of
       $\max_i|\hat\lambda_i-2|$ for the computed eigenvalues
       $\hat\lambda_i$. Compare with $\epsilon_{\textrm{mach}}^{1/m}$. Repeat
       with the symmetric matrix $\mathbf{Q}(2\mathbf{I})\mathbf{Q}^\top$.
    1. Let $\mathbf{A}$ and $\mathbf{E}$ be symmetric. Use the Rayleigh
       proposition of :numref:`subsec_mdl-rayleigh` to show that
       $\lambda_{\max}(\mathbf{A}+\mathbf{E})\le\lambda_{\max}(\mathbf{A})+\|\mathbf{E}\|_2$
       and
       $\lambda_{\min}(\mathbf{A}+\mathbf{E})\ge\lambda_{\min}(\mathbf{A})-\|\mathbf{E}\|_2$.
       How does this explain the symmetric result of (b)?
    1. Prove the Courant--Fischer theorem
       $\lambda_j=\max_{\dim U=j}\ \min_{\mathbf 0\neq\mathbf x\in U}R(\mathbf x)$
       for symmetric $\mathbf A$ with $\lambda_1\ge\cdots\ge\lambda_n$. For the
       upper bound, intersect $U$ with
       $\operatorname{span}(\mathbf v_j,\ldots,\mathbf v_n)$, which has
       dimension $n-j+1$, so the intersection is at least a line; for the
       lower bound take $U=\operatorname{span}(\mathbf v_1,\ldots,\mathbf v_j)$.
       Deduce Weyl's inequality
       $|\lambda_j(\mathbf A+\mathbf E)-\lambda_j(\mathbf A)|\le\|\mathbf E\|_2$
       for every $j$, which extends (c) to all eigenvalues.
   :exlabel:`ex_mdl-eigendecomposition-computed-eigenvalue-accuracy`

1. [code] **Pseudospectra and transient growth.** ● Let $\mathbf{M}$ be the
   non-normal matrix :eqref:`eq_mdl-nonnormal-example` and
   $\boldsymbol\Lambda=\operatorname{diag}(0.9,0.8)$ the normal matrix with the
   same eigenvalues, as in :numref:`subsec_mdl-transient-growth`.
    1. Prove the equality of the two sets in :eqref:`eq_mdl-pseudospectrum`.
       First show that $\|(z\mathbf I-\mathbf A)^{-1}\|_2=1/\sigma_{\min}(z\mathbf I-\mathbf A)$,
       where $\sigma_{\min}(\mathbf X)=\min_{\|\mathbf x\|=1}\|\mathbf X\mathbf x\|$.
       For one inclusion, take unit vectors $\mathbf u$, $\mathbf v$ with
       $(z\mathbf{I}-\mathbf{A})\mathbf{v}=\sigma\mathbf{u}$,
       $\sigma=\sigma_{\min}(z\mathbf I-\mathbf A)$, and show that $z$ is an
       eigenvalue of $\mathbf{A}+\sigma\mathbf{u}\mathbf{v}^*$. For the other,
       apply $(z\mathbf I-\mathbf A)\mathbf x=\mathbf E\mathbf x$ to an
       eigenvector $\mathbf x$ of $\mathbf A+\mathbf E$. Using the resolvent
       norms printed in :numref:`subsec_mdl-pseudospectra`, how large a
       perturbation makes $z=1$ an eigenvalue of a perturbed $\mathbf{M}$, and
       how large for $\boldsymbol\Lambda$?
    1. Suppose that $\|\mathbf{A}^k\|_2\le K_0$ for all $k\ge0$. Show that for
       $|z|>1$ the series $\sum_{k\ge0}\mathbf{A}^k/z^{k+1}$ converges to
       $(z\mathbf{I}-\mathbf{A})^{-1}$, and conclude that

       $$
       \sup_{k\ge0}\|\mathbf{A}^k\|_2\;\ge\;K(\mathbf{A})=\sup_{|z|>1}\,(|z|-1)\,\|(z\mathbf{I}-\mathbf{A})^{-1}\|_2 .
       $$

       The quantity $K(\mathbf{A})$ is called the Kreiss constant. Restate the
       bound in terms of pseudospectra: if $\Lambda_\varepsilon(\mathbf{A})$
       contains a point at distance $\delta$ outside the unit circle, how
       large must some power of $\mathbf{A}$ be?
    1. Evaluate $K(\mathbf{M})$ numerically on a grid of points $z$ with
       $|z|>1$, and compare it with the largest $\|\mathbf{M}^k\|_2$ printed
       in :numref:`subsec_mdl-transient-growth`. Find $K(\boldsymbol\Lambda)$
       by hand.
   :exlabel:`ex_mdl-eigendecomposition-pseudospectra-transient-growth`

1. [code] **Shifted inverse iteration.** ● Power iteration finds only the
   dominant eigenvalue. *Inverse iteration* with shift $\mu$ runs power
   iteration with $(\mathbf{A}-\mu\mathbf{I})^{-1}$ instead, solving one linear
   system per step :cite:`Golub.Van-Loan.1996`. Parts (a) to (c) use the
   symmetric matrix

   $$
   \mathbf A_4=\begin{pmatrix}1&0.1&0.1&0.1\\0.1&3&0.2&0.3\\0.1&0.2&5&0.5\\0.1&0.3&0.5&9\end{pmatrix},
   $$

   whose Gershgorin centres are 1, 3, 5, 9 and whose eigenvalues are about
   0.99, 2.97, 4.95, and 9.08.
    1. Show that $(\mathbf{A}-\mu\mathbf{I})^{-1}$ has the same eigenvectors as
       $\mathbf{A}$, and find its eigenvalues. Which eigenpair of $\mathbf{A}$
       does inverse iteration find, and at which rate does it converge?
    1. Run inverse iteration on $\mathbf A_4$ with each of the four disc centres
       as the shift. How many iterations does each run need until the sine of
       the angle between the iterate and the eigenvector returned by
       `np.linalg.eigh` falls below $10^{-10}$? Compare with plain power
       iteration for the largest eigenvalue.
    1. *Rayleigh quotient iteration* replaces the fixed shift by the Rayleigh
       quotient of the current iterate at every step. Repeat (b) with the disc
       centres as initial shifts. Relate the iteration counts to Exercise
       :exref:`ex_mdl-eigendecomposition-power-iteration-rates`b.
    1. Run inverse iteration on the $3\times3$ matrix
       :eqref:`eq_mdl-gershgorin-example` with the real shift $\mu=4$. Why does
       it never converge? Replace the shift by $4+i$ (complex arithmetic) and
       explain the observed rate of about 0.02 per step from the distances
       $|\lambda-\mu|$ of the three eigenvalues.
   :exlabel:`ex_mdl-eigendecomposition-shifted-inverse-iteration`

1. [code] **Damping in PageRank.** ● Let $\mathbf{P}$ be a column-stochastic
   $n\times n$ matrix and
   $\mathbf{G}=\alpha\mathbf{P}+(1-\alpha)\tfrac1n\mathbf{1}\mathbf{1}^\top$
   with $0<\alpha<1$, the damped matrix of :numref:`subsec_mdl-pagerank`.
    1. Let $\mathbf{P}\mathbf{x}=\lambda\mathbf{x}$. Show that
       $\mathbf{1}^\top\mathbf{x}=0$ whenever $\lambda\neq1$, and that
       $\mathbf{G}\mathbf{x}=\alpha\lambda\mathbf{x}$ whenever
       $\mathbf{1}^\top\mathbf{x}=0$.
    1. The eigenvalue $1$ of $\mathbf{P}$ can be repeated, as it is when the
       links split into disconnected pockets. You may use that every
       column-stochastic $\mathbf P$ has a non-negative eigenvector for the
       eigenvalue 1 (a stationary distribution). Show that the vectors
       $\mathbf{x}$ of this eigenspace with $\mathbf{1}^\top\mathbf{x}=0$ form a
       subspace of dimension one less than the eigenspace. Conclude that for
       diagonalizable $\mathbf{P}$ the eigenvalues of $\mathbf{G}$ are $1$ and
       $\alpha\lambda_2(\mathbf{P}),\ldots,\alpha\lambda_n(\mathbf{P})$, which
       gives the bound $|\lambda_2(\mathbf{G})|\le\alpha$ of
       :citet:`Haveliwala.Kamvar.2003`.
    1. Four pages link in a cycle, $1\to2\to3\to4\to1$. Find the eigenvalues
       of their link matrix $\mathbf{P}$. Why does undamped power iteration
       started at page $1$ never converge?
    1. With $\alpha=0.85$ and the cycle of (c), run power iteration on
       $\mathbf{G}$ from page $1$, and count the iterations until the $\ell_1$
       distance to the stationary distribution falls below $10^{-8}$. Compare
       with the prediction $0.85^k\le10^{-8}$ from (b).
    1. Check numerically on the three-page example of
       :numref:`subsec_mdl-pagerank` that the eigenvalues of $\mathbf G$ are
       $\alpha$ times those of $\mathbf P$, apart from the eigenvalue 1, and
       that the stationary distribution moves from $(0.3,0.4,0.3)$ to about
       $(0.303,0.394,0.303)$. Why does damping move it toward uniform?
   :exlabel:`ex_mdl-eigendecomposition-pagerank-damping`

1. [code] **Spikes above the bulk.** ●
   :numref:`subsec_mdl-sample-covariance-mp` treats the Marchenko--Pastur bulk
   as the range of pure-noise eigenvalues. Suppose that the data contain one
   real direction: the $n$ samples have independent Gaussian coordinates of
   variance $1$, except the first coordinate, whose variance is $1+\beta$. For
   $n\to\infty$ with $c=d/n$ fixed, :citet:`Baik.Silverstein.2006` show that
   the largest eigenvalue of $\mathbf{C}=\tfrac1n\mathbf{X}^\top\mathbf{X}$
   converges to

   $$
   \lambda_{\max}\to\begin{cases}(1+\sqrt c)^2, & \beta\le\sqrt c,\\
   (1+\beta)\left(1+c/\beta\right), & \beta>\sqrt c.\end{cases}
   $$

    1. For $n=2000$ and $d=500$, compute the largest eigenvalue of
       $\mathbf{C}$ for $\beta\in\{0,0.3,0.5,0.7,1,2\}$, averaged over five
       draws, and compare with the prediction.
    1. Fix $d=500$ and $\beta=0.4$. Use the formula to find the sample size
       $n$ above which the largest eigenvalue leaves the bulk. Then increase
       the number of samples through $n\in\{500,1000,2000,4000,8000\}$, and
       compare the largest eigenvalue with the bulk edge, with the prediction,
       and with the largest eigenvalue for $\beta=0$ at the same $n$. Why is
       the separation hard to see at $n=4000$?
    1. A practitioner keeps the principal components whose eigenvalues exceed
       the bulk edge. How large must the variance of a direction be at
       $c=1/4$ for it to be kept? How many samples per dimension are
       needed to detect a direction whose variance exceeds the others by
       $10\%$?
   :exlabel:`ex_mdl-eigendecomposition-spiked-covariance`

1. [code] **Lyapunov exponents of random products.** ●● Let
   $\mathbf{D}_1,\mathbf{D}_2,\ldots$ be independent $n\times n$ matrices with
   independent $\mathcal{N}(0,1/n)$ entries, a linear model of the Jacobians
   of a deep network at fan-in initialization, and let $\mathbf{x}_0$ be a
   unit vector. By :numref:`subsec_mdl-random-matrices`, each factor has
   spectral radius close to $1$ for large $n$.
   :numref:`subsec_mdl-jacobian-products` summarizes the growth of the
   product by its top Lyapunov exponent

   $$
   \mu=\lim_{T\to\infty}\frac1T\log\|\mathbf{D}_T\cdots\mathbf{D}_1\mathbf{x}_0\|,
   $$

   which exists with probability one by the theorem of Furstenberg and Kesten.
    1. Show that $\mathbb{E}\|\mathbf{D}\mathbf{x}\|^2=\|\mathbf{x}\|^2$ for
       every fixed $\mathbf{x}$, so that this scaling preserves squared norms
       on average.
    1. Let $\mathbf{u}_0=\mathbf{x}_0$ and
       $\mathbf{u}_t=\mathbf{x}_t/\|\mathbf{x}_t\|$ for the iterates
       $\mathbf{x}_t=\mathbf{D}_t\cdots\mathbf{D}_1\mathbf{x}_0$. Given
       $\mathbf{D}_1,\ldots,\mathbf{D}_{t-1}$, what is the distribution of
       $n\|\mathbf{D}_t\mathbf{u}_{t-1}\|^2$? Use the answer to write
       $\log\|\mathbf{D}_T\cdots\mathbf{D}_1\mathbf{x}_0\|$ as a sum of
       independent terms. With $\mathbb{E}\log\chi^2_n=\psi(n/2)+\log2$, where
       $\psi$ is the digamma function, derive the closed form of
       :citet:`Cohen.Newman.1984`

       $$
       \mu=\tfrac12\bigl(\psi(n/2)+\log2-\log n\bigr).
       $$

    1. Show that $\mu<0$ for every $n$, although (a) holds. Evaluate $\mu$ in
       closed form for $n=1$, using $\psi(1/2)=-\gamma_{\textrm{E}}-2\log2$
       with Euler's constant $\gamma_{\textrm{E}}\approx0.5772$. Using
       $\psi(x)=\log x-\tfrac1{2x}+O(x^{-2})$, show that $\mu\approx-1/(2n)$
       for large $n$.
    1. Simulate $T=10^4$ steps with renormalization for
       $n\in\{1,2,5,50\}$, and compare the measured exponent with (b) and with
       the logarithms of the spectral radius and of the largest singular value
       of single factors. By which factor does a typical input shrink in a
       linear network of width $50$ and depth $100$ at this initialization?
       What does the comparison say about the three quantities of
       :numref:`subsec_mdl-jacobian-products`?
   :exlabel:`ex_mdl-eigendecomposition-lyapunov-random-products`

:begin_tab:`mxnet`
[Discussions](https://d2l.discourse.group/t/411)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/t/1086)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://d2l.discourse.group/t/1087)
:end_tab:

:begin_tab:`jax`
[Discussions](https://d2l.discourse.group/t/1087)
:end_tab:

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §22.2]{.kicker}

Eigendecompositions<br>
**the directions a matrix only stretches · powers · symmetry · iteration**
:::
:::

::: {.slide title="Where Is the Surfer After 50 Clicks?"}
[The problem]{.kicker}

Two web pages. From page 1: stay or move, $1/2$ each. From page 2: stay
$3/4$, back to page 1 $1/4$.

$$\mathbf p_{k+1}=\mathbf P\mathbf p_k,\qquad
\mathbf P=\begin{pmatrix}1/2&1/4\\1/2&3/4\end{pmatrix}$$

$(1,0)\to(0.5,0.5)\to(0.375,0.625)\to(0.344,0.656)\to\ ?$

::: {.d2l-note .warn}
Where does $\mathbf P^{50}\mathbf p_0$ land, and how fast does it settle?
The same question: a 50-layer linear network, gradient descent on a
quadratic, PageRank.
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- compute the eigenvalues and eigenvectors of a $2\times2$ matrix by hand;
- write $\mathbf A^k\mathbf x$ in closed form when an eigenbasis exists;
- decide positive definiteness from eigenvalue signs;
- bound eigenvalues without computing them, and compute the largest one;
- name the cases where eigenvalues mislead.

**Running example:** the surfer's $\mathbf P$. **Idea:** in eigenvector
coordinates a matrix is a list of numbers.
:::

::: {.slide title="Eigenvalues Turn Repeated Maps into Scalar Powers"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| Vectors, Angles, and Projections | lengths, angles, projections, hyperplanes |
| Matrices as Linear Maps | columns, rank, inverses, determinants |
| **Eigendecompositions** | **powers of one square map, symmetric matrices** |
| Singular Value Decomposition | two bases, best low rank, least squares, $\kappa$ |
:::

::: {.col .narrow}
- **Uses:** determinants, $\det(\mathbf X\mathbf Y)=\det\mathbf X\det\mathbf Y$.
- **Skips:** production eigensolvers.
- Complex numbers only where a real matrix forces them.
:::
:::
:::

::: {.slide title="Eigenvalues Are the Roots of det(A − λI)"}
$\mathbf A\mathbf v=\lambda\mathbf v$ with $\mathbf v\neq\mathbf 0$ needs
$\mathbf A-\lambda\mathbf I$ singular:

$$\det(\mathbf P-\lambda\mathbf I)=\lambda^2-\tfrac54\lambda+\tfrac14
=(\lambda-1)\bigl(\lambda-\tfrac14\bigr).$$

- $\lambda=1$: row 1 gives $-\tfrac12v_1+\tfrac14v_2=0$, so $\mathbf v_1=(1,2)$.
- $\lambda=\tfrac14$: row 1 gives $v_1+v_2=0$, so $\mathbf v_2=(1,-1)$.

::: {.d2l-note .rule}
Eigenvectors are fixed only up to scale; $\lambda=0$ means singular.
:::
:::

::: {.slide title="Eigenvectors Are the Ellipse Axes Only for Symmetric Matrices"}
@fig:mdl-la-eig-ellipse

Left $\left(\begin{smallmatrix}2&1\\1&2\end{smallmatrix}\right)$: eigenvectors
(green) on the axes. Right
$\left(\begin{smallmatrix}1&1\\0&2\end{smallmatrix}\right)$: eigenvectors
$(1,0)$, $(1,1)$; axes (grey) at $58^\circ$ and $-32^\circ$.
:::

::: {.slide title="Diagonalization Turns P^k into Scalar Powers"}
$\mathbf P=\mathbf V\boldsymbol\Lambda\mathbf V^{-1}$ with
$\mathbf V=\left(\begin{smallmatrix}1&1\\2&-1\end{smallmatrix}\right)$,
$\boldsymbol\Lambda=\operatorname{diag}(1,\tfrac14)$; then
$\mathbf A^k\mathbf x=\sum_i c_i\lambda_i^k\mathbf v_i$, $\mathbf c=\mathbf V^{-1}\mathbf x$:

$$\mathbf p_k=\tfrac13\begin{pmatrix}1\\2\end{pmatrix}+\tfrac23\bigl(\tfrac14\bigr)^k\begin{pmatrix}1\\-1\end{pmatrix}$$

@!mdl-eigendecomposition-diagonalization

::: {.d2l-note .rule}
Limit $(1/3,2/3)$; the error shrinks by 4 per click.
:::
:::

::: {.slide title="Check: Eigenvalues from Trace and Determinant"}
[Check]{.kicker}

For a $2\times2$ matrix the eigenvalues are the roots of
$\lambda^2-(\operatorname{tr})\lambda+\det$.

1. $\operatorname{tr}=5/4$, $\det=1/4$: which eigenvalues?
2. $\operatorname{tr}=8$, $\det=15$: which eigenvalues?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** (1) $1$ and $1/4$, the surfer's $\mathbf P$. (2) $3$ and $5$:
their sum is 8, their product 15.
:::
:::
:::

::: {.slide title="A Change of Basis Cannot Change the Eigenvalues"}
For invertible $\mathbf B$:

$$\det(\mathbf B\mathbf A\mathbf B^{-1}-\lambda\mathbf I)
=\det\mathbf B\,\det(\mathbf A-\lambda\mathbf I)\,\det\mathbf B^{-1}
=\det(\mathbf A-\lambda\mathbf I).$$

- So $\det\mathbf A=\prod_i\lambda_i$ and $\operatorname{tr}\mathbf A=\sum_i\lambda_i$.
- On $\mathbf P$: $\det=\tfrac14=1\cdot\tfrac14$, $\operatorname{tr}=\tfrac54=1+\tfrac14$.
- Over $\mathbb C$ with multiplicity, both hold for every square matrix.
:::

::: {.slide title="A Repeated Eigenvalue Can Have Too Few Eigenvectors"}
::: {.cols .vc}
::: {.col}
Shear $\left(\begin{smallmatrix}1&1\\0&1\end{smallmatrix}\right)$:
$\lambda=1$ is a double root, but only $(1,0)$ keeps its direction.

- algebraic multiplicity 2, geometric 1: *defective*;
- $\mathbf A^k=\left(\begin{smallmatrix}1&k\\0&1\end{smallmatrix}\right)$ grows.
:::

::: {.col .fig .big}
![](../img/mdl-la-defective-shear.svg)
:::
:::

::: {.d2l-note .rule}
Eigenbasis of $\mathbb C^n$ iff $g=a$ for every eigenvalue; a real one needs
real eigenvalues too.
:::
:::

::: {.slide title="A Jordan Block Grows Like kλ^(k−1) Before It Decays"}
$\mathbf J_2(\lambda)=\lambda\mathbf I+\mathbf N$, $\mathbf N^2=\mathbf 0$, so
$\mathbf J_2(\lambda)^k=\lambda^k\mathbf I+k\lambda^{k-1}\mathbf N$.
For $\lambda=0.99$ the peak is near $k\approx-1/\log\lambda\approx100$:

@!mdl-eigendecomposition-jordan-power

::: {.d2l-note .rule}
Every eigenvalue is 0.99, yet the norm reaches 37.
:::
:::

::: {.slide title="A Complex Pair Is a Rotation and Scaling in Some Plane"}
$\mathbf A=\left(\begin{smallmatrix}1&2\\-1&3\end{smallmatrix}\right)$,
$\lambda=2\pm i=\sqrt5\,e^{\pm i\,26.6^\circ}$. Split the eigenvector,
$\mathbf w=\mathbf u+i\mathbf v$, and use the basis $\mathbf B=[\mathbf v\;\mathbf u]$:

@!mdl-eigendecomposition-complex-eigenvalues-are-rotations-1

::: {.d2l-note .rule}
In that basis: scale $\sqrt5$, turn $26.6^\circ$. In the original one the
norm ratio swings between 1.4 and 3.5.
:::
:::

::: {.slide title="Symmetric Matrices Have Real Eigenvalues (I)"}
$\mathbf A=\mathbf A^\top$ real, $\mathbf A\mathbf v=\lambda\mathbf v$, $\mathbf v\in\mathbb C^n$:

$$\lambda\|\mathbf v\|^2=\mathbf v^*\mathbf A\mathbf v=(\mathbf A\mathbf v)^*\mathbf v=\bar\lambda\|\mathbf v\|^2
\ \Rightarrow\ \lambda=\bar\lambda.$$

Distinct eigenvalues give orthogonal eigenvectors:
$\lambda\langle\mathbf u,\mathbf v\rangle=\langle\mathbf A\mathbf u,\mathbf v\rangle=\langle\mathbf u,\mathbf A\mathbf v\rangle=\mu\langle\mathbf u,\mathbf v\rangle$.

::: {.d2l-note}
Instance: $\left(\begin{smallmatrix}2&1\\1&2\end{smallmatrix}\right)$ has $\lambda=3,1$
with eigenvectors $(1,1)\perp(1,-1)$.
:::
:::

::: {.slide title="…and an Orthonormal Eigenbasis (II)"}
Induction: take a unit eigenvector $\mathbf v_1$; $U=\mathbf v_1^\perp$ is
mapped into itself, since
$\langle\mathbf A\mathbf x,\mathbf v_1\rangle=\lambda_1\langle\mathbf x,\mathbf v_1\rangle=0$.

- orthonormal basis $\mathbf Q$ of $U$; $\mathbf Q^\top\mathbf A\mathbf Q$ is symmetric, size $n-1$;
- its eigenvectors, mapped back by $\mathbf Q$, finish the basis.

::: {.d2l-note .rule}
$\mathbf A=\mathbf V\boldsymbol\Lambda\mathbf V^\top$, $\mathbf V^\top\mathbf V=\mathbf I$:
turn onto the axes, scale, turn back.
:::
:::

::: {.slide title="Eigenvalue Signs Decide the Shape of xᵀAx"}
$\mathbf x^\top\mathbf A\mathbf x=\sum_i\lambda_i(\mathbf v_i^\top\mathbf x)^2$, a
weighted sum of squares. For $\mathbf S$, $\mathbf e_1$: $2=3\cdot\tfrac12+1\cdot\tfrac12$.

@fig:mdl-la-psd

PD iff all $\lambda_i>0$; PSD iff all $\lambda_i\ge0$.
:::

::: {.slide title="The Rayleigh Quotient Lies Between λmin and λmax"}
$R(\mathbf x)=\mathbf x^\top\mathbf S\mathbf x/\mathbf x^\top\mathbf x$ is an
average of the eigenvalues with weights $y_i^2$:

@!mdl-eigendecomposition-the-rayleigh-quotient-eigenvalues-as-extreme-stretches

::: {.d2l-note .rule}
Extremes 1 and 3 at $135^\circ$ and $45^\circ$. Gradient descent on
$\tfrac12\mathbf x^\top\mathbf A\mathbf x$ needs $\eta<2/\lambda_{\max}$.
:::
:::

::: {.slide title="Pure-Noise Covariances Fill [0.25, 2.25] When d = n/4"}
$n$ samples of $d$ independent unit-variance coordinates; true covariance
$\mathbf I$, but $\mathbf C=\tfrac1n\mathbf X^\top\mathbf X$ has eigenvalues
in $\bigl[(1-\sqrt c)^2,(1+\sqrt c)^2\bigr]$, $c=d/n$:

@!mdl-eigendecomposition-sample-covariance-and-the-marchenko-pastur-law

::: {.d2l-note .rule}
An eigenvalue inside the bulk is noise; only one above the edge is structure.
:::
:::

::: {.slide title="Check: Positive Definite Without Eigenvalues?"}
[Check]{.kicker}

Is $\left(\begin{smallmatrix}2&1\\1&2\end{smallmatrix}\right)$ positive
definite? Decide without computing eigenvalues, in two different ways.

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** Trace $4>0$ and determinant $3>0$, so both eigenvalues are
positive. Or: Gershgorin puts both real eigenvalues in $[2-1,2+1]=[1,3]$.
:::
:::
:::

::: {.slide title="Every Eigenvalue Lies in the Union of the Gershgorin Discs"}
@fig:mdl-la-gershgorin

Centres $a_{ii}$, radii $r_i=\sum_{j\neq i}|a_{ij}|$. A connected group of
$m$ discs holds $m$ eigenvalues: the pair $3.99\pm0.96i$ near 4, and $1.03$.
:::

::: {.slide title="Power Iteration Converges at Rate |λ₂/λ₁|"}
@fig:mdl-la-power-iter

$\mathbf S^k\mathbf e_1=(3^k\mathbf v_1+\mathbf v_2)/\sqrt2$: direction error
$\times\tfrac13$, eigenvalue estimate $\times\tfrac19$ per step. Needs
$c_1\neq0$ (almost every start) and a strictly dominant $|\lambda_1|$.
:::

::: {.slide title="PageRank Is the Eigenvector for λ = 1"}
Links $1\to2$, $2\to1$, $2\to3$; page 3 jumps anywhere:
$\mathbf P\boldsymbol\pi=\boldsymbol\pi$ gives $\boldsymbol\pi=(0.3,0.4,0.3)$.

- columns sum to 1, so $\lambda=1$; Gershgorin keeps $|\lambda|\le1$;
- damping $\mathbf G=0.85\,\mathbf P+0.05\,\mathbf 1\mathbf 1^\top$ is positive;
- Haveliwala and Kamvar: $|\lambda_2(\mathbf G)|\le0.85$;
- sparse $\mathbf P$ is stored, dense $\mathbf G$ never formed.
:::

::: {.slide title="Same Eigenvalues, Ten-Fold Growth Before Decay"}
$\mathbf M=\left(\begin{smallmatrix}0.9&4\\0&0.8\end{smallmatrix}\right)$:
$\mathbf v_1=(1,0)$, $\mathbf v_2=(-40,1)$, $1.4^\circ$ apart, and
$\mathbf e_2=40\mathbf v_1+\mathbf v_2$.

@fig:mdl-la-transient-cancel

Peak $\|\mathbf M^6\|_2=10.8$; $\operatorname{diag}(0.9,0.8)$ never exceeds 1.
:::

::: {.slide title="Two Factors with ρ = 0, a Product with ρ = 4"}
$\mathbf D_1=\left(\begin{smallmatrix}0&2\\0&0\end{smallmatrix}\right)$,
$\mathbf D_2=\left(\begin{smallmatrix}0&0\\2&0\end{smallmatrix}\right)$,
$\mathbf D_2\mathbf D_1=\operatorname{diag}(0,4)$.

- spectral radius: rate of powers of one fixed map;
- spectral norm: bounds finite products, $\le\prod_t\|\mathbf D_t\|_2$;
- Lyapunov exponent: average growth of long changing products.

::: {.d2l-note .rule}
Orthogonal init sets every singular value to 1; clipping caps the gradient
norm; gates change the Jacobians.
:::
:::

::: {.slide title="Check: Does ρ = 1 Bound the Shear?"}
[Check]{.kicker}

The shear $\left(\begin{smallmatrix}1&1\\0&1\end{smallmatrix}\right)$ has
spectral radius 1. Is $\|\mathbf A^k\|_2$ bounded in $k$?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** No: $\mathbf A^k=\left(\begin{smallmatrix}1&k\\0&1\end{smallmatrix}\right)$
and $\|\mathbf A^k\mathbf e_2\|=\sqrt{k^2+1}$. The shear is defective, so no
eigen-expansion applies.
:::
:::
:::

::: {.slide title="Wrap-up"}
- Surfer: $\mathbf p_k\to(1/3,2/3)$, error $\times\tfrac14$ per click.
- Jordan block $\mathbf J_2(0.99)$: norm 37 at $k=100$.
- $\left(\begin{smallmatrix}1&2\\-1&3\end{smallmatrix}\right)$: $\sqrt5$ and $26.6^\circ$ in the right basis.
- Symmetric: $\mathbf V\boldsymbol\Lambda\mathbf V^\top$; signs give definiteness.
- Noise covariance at $d=n/4$: eigenvalues in $[0.25,2.25]$.
- $\mathbf M$: same eigenvalues as $\operatorname{diag}(0.9,0.8)$, peak 10.8.

**Next:** singular values measure finite-step stretch for every matrix.
:::
