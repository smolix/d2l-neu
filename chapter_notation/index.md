# Notation
:label:`chap_notation`

This book uses the following notational conventions. Some entries describe a
class of symbols, while others define one particular symbol. The indefinite
article “a” marks a convention: for example, “$x$: a scalar” means that
lowercase letters generally denote scalars. The definite article marks a fixed
meaning: “$\mathbb{R}$: the set of real numbers” refers specifically to
$\mathbb{R}$.

A few letters carry more than one meaning. The form of the symbol tells them
apart: the sigmoid $\sigma(\cdot)$ takes an argument, a standard deviation
$\sigma_X$ names a random variable, a singular value $\sigma_i(\mathbf{X})$
names a matrix, and the noise level $\sigma_t$ of a diffusion model carries a
time. The entries below point out such cases. Chapters that need many symbols
of their own list them in a table at their start: optimization
(:numref:`tab_mdl-opt-notation`), information theory
(:numref:`tab_mdl-it-notation`), dynamics (:numref:`tab_mdl-dyn-notation`),
reinforcement learning (:numref:`tab_rl_notation`), and generative
adversarial networks (:numref:`tab_gan_notation`). The last sections of this
page give the core symbols of several later chapters.



## Numerical Objects

* $x$: a scalar
* $\mathbf{x}$: a vector
* $(x_1, \ldots, x_n)$: the column vector with entries $x_1, \ldots, x_n$, written inline; a matrix row is listed the same way, as in "the rows $(0, 1, 2)$ and $(3, 4, 5)$"
* $\mathbf{X}$: a matrix
* $\mathsf{X}$: a tensor with more than two axes
* $\mathbf{I}$: the identity matrix (of some given dimension), i.e., a square matrix with $1$ on all diagonal entries and $0$ on all off-diagonals
* $\mathbf{0}$, $\mathbf{1}$: the vectors of all zeros and of all ones (of some given dimension)
* $x_i$: the $i^\textrm{th}$ element of vector $\mathbf{x}$
* $x_{ij}$, $[\mathbf{X}]_{i,j}$: the element of matrix $\mathbf{X}$ in row $i$ and column $j$



## Set Theory

* $\mathcal{X}$: a set
* $\mathbb{R}$: the set of real numbers
* $\mathbb{R}^n$: the set of $n$-dimensional vectors of real numbers
* $\mathbb{R}^{a\times b}$: the set of real matrices with $a$ rows and $b$ columns
* $|\mathcal{X}|$: cardinality (number of elements) of set $\mathcal{X}$
* $\mathcal{A}\cup\mathcal{B}$: union of sets $\mathcal{A}$ and $\mathcal{B}$
* $\mathcal{A}\cap\mathcal{B}$: intersection of sets $\mathcal{A}$ and $\mathcal{B}$
* $\mathcal{A}\setminus\mathcal{B}$: the elements of $\mathcal{A}$ that do not belong to $\mathcal{B}$



## Functions and Operators

* $f(\cdot)$: a function
* $\log(\cdot)$: the natural logarithm (base $e$)
* $\log_2(\cdot)$: logarithm to base $2$
* $\exp(\cdot)$: the exponential function
* $\sigma(\cdot)$: the sigmoid (logistic) function, $\sigma(x) = 1/(1 + e^{-x})$
* $\mathrm{softmax}(\mathbf{o})$: the vector with entries $\exp(o_i) / \sum_j \exp(o_j)$
* $\mathbf{1}(\cdot)$: the indicator function, equal to $1$ when its Boolean argument is true and $0$ otherwise (unlike the vector $\mathbf{1}$, it takes an argument)
* $\operatorname{argmax}_x f(x)$, $\operatorname{argmin}_x f(x)$: a value of $x$ at which $f$ is largest, smallest
* $\sum$: summation over a collection of elements
* $\prod$: product over a collection of elements
* $O(\cdot)$: an upper bound up to a constant factor, as in a cost of $O(n^2)$ operations
* $\propto$: proportional to; the two sides differ by a factor that does not depend on the variable of interest
* $\stackrel{\textrm{def}}{=}$: an equality asserted as a definition of the symbol on the left-hand side



## Linear Algebra

* $(\cdot)^\top$: transpose of a vector or a matrix
* $\mathbf{X}^{-1}$: inverse of matrix $\mathbf{X}$
* $\mathbf{X}^{+}$: Moore–Penrose pseudoinverse of matrix $\mathbf{X}$
* $\langle \mathbf{x}, \mathbf{y} \rangle$: inner (dot) product of vectors $\mathbf{x}$ and $\mathbf{y}$, also written $\mathbf{x}^\top \mathbf{y}$
* $\odot$: Hadamard (elementwise) product
* $[\cdot, \cdot]$: concatenation
* $\|\cdot\|_p$: $\ell_p$ norm
* $\|\cdot\|$: $\ell_2$ norm
* $\|\mathbf{X}\|_2$: spectral norm of matrix $\mathbf{X}$, its largest singular value
* $\|\mathbf{X}\|_F$: Frobenius norm of matrix $\mathbf{X}$, the $\ell_2$ norm of its entries
* $\|\mathbf{X}\|_*$: nuclear norm of matrix $\mathbf{X}$, the sum of its singular values
* $\operatorname{diag}(\mathbf{x})$: the diagonal matrix with the entries of $\mathbf{x}$ on its diagonal
* $\operatorname{rank}\mathbf{X}$, $\operatorname{tr}\mathbf{X}$, $\det\mathbf{X}$: rank, trace, and determinant of matrix $\mathbf{X}$
* $\lambda_i$, $\lambda_{\max}(\mathbf{X})$: an eigenvalue of a square matrix, and the largest eigenvalue of a symmetric matrix $\mathbf{X}$
* $\sigma_i(\mathbf{X})$: the $i^\textrm{th}$ largest singular value of matrix $\mathbf{X}$ (not the sigmoid $\sigma(\cdot)$)
* $\kappa(\mathbf{X})$: condition number of matrix $\mathbf{X}$, the ratio of its largest to its smallest nonzero singular value
* $\rho(\mathbf{X})$: spectral radius of a square matrix $\mathbf{X}$, the largest modulus of its eigenvalues (not the correlation $\rho(X, Y)$ below)
* $\mathbf{X}\succeq 0$, $\mathbf{X}\succ 0$: the symmetric matrix $\mathbf{X}$ is positive semidefinite, positive definite



## Calculus

* $\frac{dy}{dx}$, $f'(x)$: derivative of $y$ with respect to $x$, derivative of $f$ at $x$
* $\frac{\partial y}{\partial x}$: partial derivative of $y$ with respect to $x$
* $\nabla_{\mathbf{x}} y$: gradient of $y$ with respect to $\mathbf{x}$
* $\nabla^2 f$: the Hessian of $f$, the matrix of its second partial derivatives
* $\int_a^b f(x) \,dx$: definite integral of $f$ from $a$ to $b$ with respect to $x$
* $\int f(x) \,dx$: indefinite integral of $f$ with respect to $x$



## Probability and Information Theory

* $X$: a random variable
* $P$: a probability distribution
* $X \sim P$: the random variable $X$ follows distribution $P$
* $P(X=x)$: the probability that random variable $X$ takes the value $x$
* $P(X \mid Y)$: the conditional probability distribution of $X$ given $Y$
* $p(\cdot)$: a probability density function (PDF) associated with distribution $P$
* $\mathbb{E}[X]$: expectation of a random variable $X$
* $\mathbb{E}_{x \sim P}[f(x)]$: expectation of $f(x)$ for $x$ drawn from $P$
* $X \perp Y$: random variables $X$ and $Y$ are independent
* $X \perp Y \mid Z$: random variables $X$ and $Y$ are conditionally independent given $Z$
* $\sigma$, $\sigma_X$: a standard deviation, and that of random variable $X$
* $\textrm{Var}(X)$: variance of random variable $X$, equal to $\sigma^2_X$
* $\textrm{Cov}(X, Y)$: covariance of random variables $X$ and $Y$
* $\boldsymbol{\Sigma}$: a covariance matrix
* $\rho(X, Y)$: the Pearson correlation coefficient, $\frac{\textrm{Cov}(X, Y)}{\sigma_X \sigma_Y}$
* $\mathcal{N}(\mu, \sigma^2)$: the Gaussian (normal) distribution with mean $\mu$ and variance $\sigma^2$; $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ is its multivariate form
* $H(X)$: entropy of random variable $X$
* $D_{\textrm{KL}}(P\|Q)$: the KL divergence (relative entropy) from distribution $P$ to distribution $Q$



## Data, Models, and Training

* $n$: the number of examples in a dataset
* $d$: the number of input features
* $\mathbf{x}^{(i)}$, $y^{(i)}$: the features and the label of example $i$
* $\mathcal{D}$: a dataset, such as $\{(\mathbf{x}^{(i)}, y^{(i)})\}_{i=1}^n$
* $\mathbf{X} \in \mathbb{R}^{n \times d}$: a design matrix or minibatch, with one example per row
* $\mathcal{B}$: a minibatch of examples, with $|\mathcal{B}|$ its size
* $\hat{y}$: a prediction of $y$
* $\mathbf{w}$, $b$: the weights and the bias of a linear model
* $\mathbf{W}^{(\ell)}$, $\mathbf{b}^{(\ell)}$: the weights and the bias of layer $\ell$, as in $\mathbf{W}^{(1)}$
* $\theta$, $\boldsymbol{\theta}$: the parameters of a model, in bold when treated as one vector
* $l$, $\ell$: a loss, often that of one example
* $L$: a loss averaged over a dataset or minibatch
* $\eta$: the learning rate (step size) of a gradient method
* $\lambda$: the coefficient of a weight-decay penalty $\frac{\lambda}{2}\|\mathbf{w}\|^2$
* $t$, $T$: a time step, and the length of a sequence
* $\mathbf{x}_t$, $\mathbf{h}_t$: the input and the hidden state at time step $t$



## Attention and Transformers

These symbols belong to :numref:`chap_attention` and are used again in :numref:`chap_transformers` and the chapters that build on it.

* $\mathbf{q}$, $\mathbf{k}_i$, $\mathbf{v}_i$: a query, and the key and value of pair $i$ (:numref:`sec_queries-keys-values`)
* $a(\mathbf{q}, \mathbf{k})$: an attention scoring function, $\mathbf{q}^\top \mathbf{k} / \sqrt{d}$ for scaled dot-product attention
* $\alpha(\mathbf{q}, \mathbf{k}_i)$: the attention weight of key $i$, the softmax of the scores over the keys
* $\mathbf{Q}$, $\mathbf{K}$, $\mathbf{V}$: the matrices of $n$ queries, $m$ keys, and $m$ values, one per row, which attention maps to $\mathrm{softmax}(\mathbf{Q}\mathbf{K}^\top / \sqrt{d})\,\mathbf{V}$ (:numref:`sec_attention-scoring-functions`)
* $d$, $d_v$: the width of queries and keys, and of values
* $H$, $d_h$: the number of attention heads and the width of each, $d_h = d/H$ (:numref:`sec_multihead-attention`)



## State Space Models

These symbols belong to :numref:`chap_modern_rnn`.

* $\mathbf{A}$, $\mathbf{B}$, $\mathbf{C}$, $D$: the state matrix, input map, readout, and skip term of a state space model (:numref:`subsec_state-space`)
* $N$: the size of the state of one channel
* $\lambda_n$: the rate of state $n$ of a diagonal state matrix; the state decays by $e^{\Delta \lambda_n}$ per step
* $\Delta$, $\Delta_t$: the step size of the discretization (:numref:`subsec_zoh`), and its input-dependent form in Mamba (:numref:`sec_mamba`)
* $\mathbf{S}_t$: a matrix-valued state, written with $\mathbf{v}_t \mathbf{k}_t^\top$ and read with a query $\mathbf{q}_t$ (:numref:`sec_matrix-state`)
* $\beta_t$: the write strength of the delta rule (:numref:`sec_deltanet`)



## Reinforcement Learning

These symbols belong to :numref:`chap_reinforcement_learning` and :numref:`chap_deep_rl`; :numref:`tab_rl_notation` lists them all with the section that defines each.

* $\mathcal{S}$, $\mathcal{A}$: the sets of states and actions
* $s_t$, $a_t$, $r_t$: the state, action, and reward at step $t$
* $P(s' \mid s, a)$: the transition kernel, the probability of reaching state $s'$ from state $s$ under action $a$
* $r(s, a)$: the expected reward of action $a$ at state $s$
* $\gamma$: the discount factor
* $\tau$, $R(\tau)$: a trajectory $(s_0, a_0, r_0, s_1, \ldots)$, and its discounted return $\sum_t \gamma^t r_t$
* $\pi(a \mid s)$: a policy, the probability of action $a$ at state $s$; $\pi(s)$ is the action of a deterministic policy
* $\pi_\theta$, $J(\theta)$: a policy with parameters $\theta$, and its expected return
* $V^\pi(s)$, $Q^\pi(s, a)$: the expected return under $\pi$ from state $s$, and from state $s$ after action $a$
* $V^*$, $Q^*$: the value functions of an optimal policy
* $A^\pi(s, a)$: the advantage $Q^\pi(s, a) - V^\pi(s)$
* $\delta_t$: the temporal-difference error of the transition at step $t$
* $\epsilon$: the exploration rate of $\epsilon$-greedy action selection



## Generative Adversarial Networks

These symbols belong to :numref:`chap_gans`; :numref:`tab_gan_notation` lists them all with the section that defines each and the letters the papers use.

* $p$, $q$, $m$: the data density, the generator's density, and their mixture $\tfrac12(p + q)$
* $x \sim p$, $x' \sim q$: a real and a generated sample
* $y$: the source label of a sample, $y = 1$ for real and $y = 0$ for generated
* $z$, $G$: the latent variable and the generator, $x' = G(z)$
* $D$: the critic's realness logit; $\sigma(D)$ is its probability that a sample is real (papers say discriminator)
* $V$: the value function of the log-loss game
* $\rho = p/q$, $\lambda = \log \rho$: the density ratio and its logarithm
* $\eta = p/(p + q)$: the class posterior; in this chapter the step size is $h$, not $\eta$
* $f$, $D_f$: a divergence function and its $f$-divergence
* $\gamma$, $R_1$, $R_2$: the penalty weight and the zero-centered gradient penalties at the data and at the generated samples
* $c$: a condition, such as a class label



## Diffusion Models

These symbols belong to :numref:`chap_diffusion` and follow the dynamics appendix (:numref:`tab_mdl-dyn-notation`). One clock serves the whole book: data sit at time $t = 0$, noise at $t = 1$, and samplers run from $t = 1$ down to $0$. Flow-matching papers often reverse the clock (:numref:`sec_diffusion-gaussian-paths`).

* $\mathbf{x}_0$, $\boldsymbol{\epsilon}$: a clean data point and standard Gaussian noise, $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
* $\mathbf{x}_t = \alpha_t \mathbf{x}_0 + \sigma_t \boldsymbol{\epsilon}$: the noisy point at time $t$ on a Gaussian path
* $\alpha_t$, $\sigma_t$: the signal and the noise coefficient of the path (not a step size or a standard deviation of the data)
* $\lambda_t = \log(\alpha_t^2 / \sigma_t^2)$: the log signal-to-noise ratio (log-SNR), which falls from $+\infty$ at the data to $-\infty$ at the noise
* $p_t$, $\nabla \log p_t$: the density of $\mathbf{x}_t$ and its score
* $\hat{\mathbf{x}}_0$, $\hat{\boldsymbol{\epsilon}}$: the posterior means $\mathbb{E}[\mathbf{x}_0 \mid \mathbf{x}_t]$ and $\mathbb{E}[\boldsymbol{\epsilon} \mid \mathbf{x}_t]$, which a denoiser predicts
* $\mathbf{u}_t$, $\mathbf{u}_{\boldsymbol{\theta}}$, $\hat{\mathbf{u}}$: the velocity that carries the samples along the path, a network that learns it, and that network's prediction
* $\mathbf{v} = \alpha_t \boldsymbol{\epsilon} - \sigma_t \mathbf{x}_0$: the target of $\mathbf{v}$-prediction (:numref:`sec_diffusion-gaussian-paths`)
* $\tilde{\mathbf{u}}$: the guided velocity (:numref:`subsec_diffusion-images-guided`)
* $\bar{\mathbf{u}}(\mathbf{x}_t, s, t)$, $\mathbf{u}_{\textrm{ref}}$: the average velocity from time $t$ to $s$, and the reference velocity (a teacher's $\hat{\mathbf{u}}$ or the example's $\boldsymbol{\epsilon} - \mathbf{x}_0$) in its loss (:numref:`sec_diffusion-few-step`)
* $c$, $\gamma$: a condition and the guidance scale (:numref:`subsec_diffusion-images-guided`)



[Discussions](https://d2l.discourse.group/t/25)
