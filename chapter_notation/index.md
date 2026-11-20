# Notation
:label:`chap_notation`

Clear, consistent notation makes it much easier to connect mathematical ideas to code across a book that spans linear algebra, calculus, probability, optimization, and modern neural architectures. Throughout this book, we adhere to a uniform typographic hierarchy: lowercase italic letters ($x$) denote scalars, bold lowercase letters ($\mathbf{x}$) denote vectors, bold uppercase letters ($\mathbf{X}$) denote matrices, sans-serif uppercase letters ($\mathsf{X}$) denote higher-order tensors, calligraphic letters ($\mathcal{X}$) denote sets, and uppercase italic letters ($X$) denote random variables.

In the reference lists below, the indefinite article *"a"* indicates a general typographic convention---for example, "$x$: a scalar" means that lowercase italic letters generally represent scalars---whereas the definite article *"the"* indicates a symbol with a specific fixed meaning, such as "$\mathbb{R}$: the set of real numbers".

Because the alphabet is finite, a few standard letters carry more than one meaning across subfields. In every case, the syntactic form and context of the symbol disambiguate it: for example, the sigmoid function $\sigma(\cdot)$ takes a parenthesized argument, a standard deviation $\sigma_X$ carries a random variable as a subscript, a singular value $\sigma_i(\mathbf{X})$ is indexed and applied to a matrix, and the noise schedule $\sigma_t$ of a diffusion model is indexed by time $t$. We note potential ambiguities explicitly below. Chapters that introduce a larger domain-specific vocabulary also open with a dedicated notation table, including optimization (:numref:`tab_mdl-opt-notation`), information theory (:numref:`tab_mdl-it-notation`), dynamics (:numref:`tab_mdl-dyn-notation`), reinforcement learning (:numref:`tab_rl_notation`), and generative adversarial networks (:numref:`tab_gan_notation`). The final sections of this chapter summarize the core notation for several advanced topics.



## Numerical Objects

* $x$: a scalar
* $\mathbf{x}$: a vector (by default a column vector)
* $(x_1, \ldots, x_n)$: the column vector with entries $x_1, \ldots, x_n$ written inline; a matrix row is written the same way in running text, as in "the rows $(0, 1, 2)$ and $(3, 4, 5)$"
* $\mathbf{X}$: a matrix
* $\mathsf{X}$: a tensor with more than two axes
* $\mathbf{I}$: the identity matrix of a contextually determined dimension (a square matrix with $1$ on the main diagonal and $0$ everywhere else)
* $\mathbf{0}$, $\mathbf{1}$: the vectors of all zeros and all ones of a contextually determined dimension
* $x_i$: the $i^\textrm{th}$ element of vector $\mathbf{x}$
* $x_{ij}$, $[\mathbf{X}]_{i,j}$: the element of matrix $\mathbf{X}$ at row $i$ and column $j$



## Set Theory

* $\mathcal{X}$: a set
* $\mathbb{R}$: the set of real numbers
* $\mathbb{R}^n$: the set of $n$-dimensional real vectors
* $\mathbb{R}^{a\times b}$: the set of real matrices with $a$ rows and $b$ columns
* $|\mathcal{X}|$: the cardinality (number of elements) of set $\mathcal{X}$
* $\mathcal{A}\cup\mathcal{B}$: the union of sets $\mathcal{A}$ and $\mathcal{B}$
* $\mathcal{A}\cap\mathcal{B}$: the intersection of sets $\mathcal{A}$ and $\mathcal{B}$
* $\mathcal{A}\setminus\mathcal{B}$: the set difference (the elements of $\mathcal{A}$ that do not belong to $\mathcal{B}$)



## Functions and Operators

* $f(\cdot)$: a function
* $\log(\cdot)$: the natural logarithm (base $e$)
* $\log_2(\cdot)$: the base-$2$ logarithm
* $\exp(\cdot)$: the exponential function
* $\sigma(\cdot)$: the logistic sigmoid function, $\sigma(x) = 1/(1 + e^{-x})$
* $\mathrm{softmax}(\mathbf{o})$: the softmax vector with entries $\exp(o_i) / \sum_j \exp(o_j)$
* $\mathbf{1}(\cdot)$: the indicator function, equal to $1$ when its Boolean argument is true and $0$ otherwise (distinguished from the all-ones vector $\mathbf{1}$ by taking an argument)
* $\operatorname{argmax}_x f(x)$, $\operatorname{argmin}_x f(x)$: a value of $x$ that maximizes or minimizes $f(x)$
* $\sum$: summation over a collection of elements
* $\prod$: product over a collection of elements
* $O(\cdot)$: an asymptotic upper bound up to a constant factor, as in a cost of $O(n^2)$ operations
* $\propto$: proportional to (the two sides differ by a constant factor independent of the variable of interest)
* $\stackrel{\textrm{def}}{=}$: equality by definition of the symbol on the left-hand side



## Linear Algebra

* $(\cdot)^\top$: the transpose of a vector or matrix
* $\mathbf{X}^{-1}$: the inverse of square matrix $\mathbf{X}$
* $\mathbf{X}^{+}$: the Moore–Penrose pseudoinverse of matrix $\mathbf{X}$
* $\langle \mathbf{x}, \mathbf{y} \rangle$: the inner (dot) product of vectors $\mathbf{x}$ and $\mathbf{y}$, also written $\mathbf{x}^\top \mathbf{y}$
* $\odot$: the Hadamard (elementwise) product
* $[\cdot, \cdot]$: concatenation of vectors or matrices
* $\|\cdot\|_p$: the $\ell_p$ vector norm
* $\|\cdot\|$: the Euclidean ($\ell_2$) norm
* $\|\mathbf{X}\|_2$: the spectral (operator) norm of matrix $\mathbf{X}$, equal to its largest singular value
* $\|\mathbf{X}\|_F$: the Frobenius norm of matrix $\mathbf{X}$, equal to the $\ell_2$ norm of its entries
* $\|\mathbf{X}\|_*$: the nuclear (trace) norm of matrix $\mathbf{X}$, equal to the sum of its singular values
* $\operatorname{diag}(\mathbf{x})$: the diagonal matrix with the entries of vector $\mathbf{x}$ on its main diagonal
* $\operatorname{rank}\mathbf{X}$, $\operatorname{tr}\mathbf{X}$, $\det\mathbf{X}$: the rank, trace, and determinant of matrix $\mathbf{X}$
* $\lambda_i$, $\lambda_{\max}(\mathbf{X})$: an eigenvalue of a square matrix, and the largest eigenvalue of a symmetric matrix $\mathbf{X}$
* $\sigma_i(\mathbf{X})$: the $i^\textrm{th}$ largest singular value of matrix $\mathbf{X}$ (not to be confused with the sigmoid function $\sigma(\cdot)$)
* $\kappa(\mathbf{X})$: the condition number of matrix $\mathbf{X}$, equal to the ratio of its largest to its smallest nonzero singular value
* $\rho(\mathbf{X})$: the spectral radius of a square matrix $\mathbf{X}$, equal to the largest absolute value (modulus) among its eigenvalues (not to be confused with the correlation coefficient $\rho(X, Y)$ below)
* $\mathbf{X}\succeq 0$, $\mathbf{X}\succ 0$: the symmetric matrix $\mathbf{X}$ is positive semidefinite ($\mathbf{X}\succeq 0$) or positive definite ($\mathbf{X}\succ 0$)



## Calculus

* $\frac{dy}{dx}$, $f'(x)$: the derivative of $y$ with respect to $x$, and the derivative of $f$ evaluated at $x$
* $\frac{\partial y}{\partial x}$: the partial derivative of $y$ with respect to $x$
* $\nabla_{\mathbf{x}} y$: the gradient of scalar $y$ with respect to vector $\mathbf{x}$
* $\nabla^2 f$: the Hessian matrix of $f$, containing its second-order partial derivatives
* $\int_a^b f(x) \,dx$: the definite integral of $f$ from $a$ to $b$ with respect to $x$
* $\int f(x) \,dx$: the indefinite integral (antiderivative) of $f$ with respect to $x$



## Probability and Information Theory

* $X$: a random variable
* $P$: a probability distribution
* $X \sim P$: the random variable $X$ follows distribution $P$
* $P(X=x)$: the probability assigned to the event that random variable $X$ takes the value $x$
* $P(X \mid Y)$: the conditional probability distribution of $X$ given $Y$
* $p(\cdot)$: a probability density function (PDF) or probability mass function (PMF) associated with distribution $P$
* $\mathbb{E}[X]$: the expectation of random variable $X$
* $\mathbb{E}_{x \sim P}[f(x)]$: the expectation of $f(x)$ when $x$ is drawn from distribution $P$
* $X \perp Y$: random variables $X$ and $Y$ are statistically independent
* $X \perp Y \mid Z$: random variables $X$ and $Y$ are conditionally independent given $Z$
* $\sigma$, $\sigma_X$: a standard deviation in general, and the standard deviation of random variable $X$ in particular
* $\textrm{Var}(X)$: the variance of random variable $X$, equal to $\sigma^2_X$
* $\textrm{Cov}(X, Y)$: the covariance of random variables $X$ and $Y$
* $\boldsymbol{\Sigma}$: a covariance matrix
* $\rho(X, Y)$: the Pearson correlation coefficient between $X$ and $Y$, equal to $\frac{\textrm{Cov}(X, Y)}{\sigma_X \sigma_Y}$
* $\mathcal{N}(\mu, \sigma^2)$: the univariate Gaussian (normal) distribution with mean $\mu$ and variance $\sigma^2$; $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ denotes its multivariate form
* $H(X)$: the Shannon entropy (or differential entropy) of random variable $X$
* $D_{\textrm{KL}}(P\|Q)$: the Kullback–Leibler (KL) divergence (relative entropy) from distribution $P$ to distribution $Q$



## Data, Models, and Training

* $n$: the number of examples in a dataset
* $d$: the number of input features (dimensionality of an example)
* $\mathbf{x}^{(i)}$, $y^{(i)}$: the feature vector and target label of the $i^\textrm{th}$ example
* $\mathcal{D}$: a dataset, such as $\{(\mathbf{x}^{(i)}, y^{(i)})\}_{i=1}^n$
* $\mathbf{X} \in \mathbb{R}^{n \times d}$: a design matrix or minibatch matrix, arranged with one example per row
* $\mathcal{B}$: a minibatch of examples, of size $|\mathcal{B}|$
* $\hat{y}$: a model's prediction of target $y$
* $\mathbf{w}$, $b$: the weight vector and scalar bias of a linear model
* $\mathbf{W}^{(\ell)}$, $\mathbf{b}^{(\ell)}$: the weight matrix and bias vector of layer $\ell$, as in $\mathbf{W}^{(1)}$
* $\theta$, $\boldsymbol{\theta}$: the parameters of a model (written in bold $\boldsymbol{\theta}$ when collected into a single vector)
* $l$, $\ell$: a per-example loss function
* $L$: the empirical loss averaged over a dataset or minibatch
* $\eta$: the learning rate (step size) of a gradient-based optimizer
* $\lambda$: the regularization strength of an $\ell_2$ weight-decay penalty $\frac{\lambda}{2}\|\mathbf{w}\|^2$
* $t$, $T$: a discrete or continuous time step $t$, and the total length $T$ of a sequence
* $\mathbf{x}_t$, $\mathbf{h}_t$: the input and hidden state at time step $t$



## Attention and Transformers

These symbols are introduced in :numref:`chap_attention` and used throughout :numref:`chap_transformers` and the chapters that build on it.

* $\mathbf{q}$, $\mathbf{k}_i$, $\mathbf{v}_i$: a query vector, and the key and value vectors of item $i$ (:numref:`sec_queries-keys-values`)
* $a(\mathbf{q}, \mathbf{k})$: an attention scoring function, such as $\mathbf{q}^\top \mathbf{k} / \sqrt{d}$ for scaled dot-product attention
* $\alpha(\mathbf{q}, \mathbf{k}_i)$: the attention weight assigned to key $i$, obtained by applying the softmax function to the scores across all keys
* $\mathbf{Q}$, $\mathbf{K}$, $\mathbf{V}$: the stacked matrices of $n$ queries, $m$ keys, and $m$ values (one per row), which scaled dot-product attention maps to $\mathrm{softmax}(\mathbf{Q}\mathbf{K}^\top / \sqrt{d})\,\mathbf{V}$ (:numref:`sec_attention-scoring-functions`)
* $d$, $d_v$: the dimension (width) of queries and keys ($d$), and of values ($d_v$)
* $H$, $d_h$: the number of attention heads and the per-head dimension, $d_h = d/H$ (:numref:`sec_multihead-attention`)



## State Space Models

These symbols are introduced in :numref:`chap_modern_rnn`.

* $\mathbf{A}$, $\mathbf{B}$, $\mathbf{C}$, $D$: the state transition matrix, input projection, output readout, and feedthrough (skip) term of a state space model (:numref:`subsec_state-space`)
* $N$: the latent state dimension per channel
* $\lambda_n$: the $n^\textrm{th}$ diagonal entry (rate) of a diagonal state matrix, whose corresponding state component decays by $e^{\Delta \lambda_n}$ per step
* $\Delta$, $\Delta_t$: the discretization step size (:numref:`subsec_zoh`), and its input-dependent (selective) counterpart in Mamba (:numref:`sec_mamba`)
* $\mathbf{S}_t \in \mathbb{R}^{d_v \times d_k}$: a matrix-valued recurrent state updated via outer products $\mathbf{v}_t \mathbf{k}_t^\top$ and read with a query $\mathbf{q}_t$ (:numref:`sec_matrix-state`); here the key dimension is written $d_k$ alongside the value dimension $d_v$
* $\beta_t$: the write strength (learning rate) of the delta update rule (:numref:`sec_deltanet`)



## Reinforcement Learning

These symbols are used across :numref:`chap_reinforcement_learning` and :numref:`chap_deep_rl`; :numref:`tab_rl_notation` provides a complete listing along with the section where each symbol is defined.

* $\mathcal{S}$, $\mathcal{A}$: the state space and action space
* $s_t$, $a_t$, $r_t$: the state, action, and scalar reward at time step $t$
* $P(s' \mid s, a)$: the Markov transition kernel giving the probability of transitioning to state $s'$ from state $s$ under action $a$
* $r(s, a)$: the expected immediate reward for taking action $a$ in state $s$
* $\gamma$: the discount factor, $\gamma \in [0, 1)$
* $\tau$, $R(\tau)$: a trajectory $(s_0, a_0, r_0, s_1, \ldots)$ and its discounted return $R(\tau) = \sum_t \gamma^t r_t$
* $\pi(a \mid s)$: a stochastic policy giving the probability of selecting action $a$ in state $s$; for a deterministic policy, $\pi(s)$ denotes the chosen action
* $\pi_\theta$, $J(\theta)$: a parameterized policy with parameters $\theta$, and its expected discounted return
* $V^\pi(s)$, $Q^\pi(s, a)$: the state-value and action-value functions under policy $\pi$ (the expected return starting from state $s$, and starting from state $s$ after taking action $a$)
* $V^*$, $Q^*$: the optimal state-value and action-value functions
* $A^\pi(s, a)$: the advantage function, $A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$
* $\delta_t$: the temporal-difference (TD) error at step $t$
* $\epsilon$: the exploration probability in $\epsilon$-greedy action selection



## Generative Adversarial Networks

These symbols belong to :numref:`chap_gans`; :numref:`tab_gan_notation` lists every symbol along with its defining section and common alternatives in the literature.

* $p$, $q$, $m$: the true data density, the generator's model density, and their equal mixture $m = \tfrac12(p + q)$
* $x \sim p$, $x' \sim q$: a real sample drawn from the data distribution and a synthetic sample drawn from the generator
* $y$: the binary source label of a sample ($y = 1$ for real data and $y = 0$ for generated samples)
* $z$, $G$: the latent noise variable and the generator network, producing $x' = G(z)$
* $D$: the critic's (or discriminator's) realness logit, so that $\sigma(D(x))$ represents the estimated probability that sample $x$ is real
* $V$: the value function of the minimax log-loss game
* $\rho = p/q$, $\lambda = \log \rho$: the likelihood ratio (density ratio) and log-density ratio
* $\eta = p/(p + q)$: the Bayes-optimal class posterior probability that a sample is real (note that within :numref:`chap_gans`, the optimizer step size is written $h$ rather than $\eta$)
* $f$, $D_f$: a divergence function and its $f$-divergence
* $\Phi$, $d_{\mathrm{Rp}}$: the pairing (relativistic) objective and its value at the optimal critic
* $\gamma$, $R_1$, $R_2$: the gradient-penalty weight and the zero-centered gradient penalties evaluated on real data ($R_1$) and generated samples ($R_2$)
* $c$: a conditioning variable, such as a class label or text prompt



## Diffusion Models

These symbols belong to :numref:`chap_diffusion` and match the conventions of the dynamics appendix (:numref:`tab_mdl-dyn-notation`). Throughout this book, we use a single unified time direction: clean data reside at $t = 0$, pure noise resides at $t = 1$, and generative samplers integrate backward from $t = 1$ down to $t = 0$. (Note that some flow-matching papers reverse this convention; see :numref:`sec_diffusion-gaussian-paths`.)

* $\mathbf{x}_0$, $\boldsymbol{\epsilon}$: a clean data point and an independent standard Gaussian noise vector, $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
* $\mathbf{x}_t = \alpha_t \mathbf{x}_0 + \sigma_t \boldsymbol{\epsilon}$: the noisy state at time $t \in [0, 1]$ along a Gaussian probability path
* $\alpha_t$, $\sigma_t$: the signal and noise schedule coefficients along the path (distinct from a learning rate or the standard deviation of the data)
* $\lambda_t = \log(\alpha_t^2 / \sigma_t^2)$: the log signal-to-noise ratio (log-SNR), which decreases monotonically from $+\infty$ at clean data ($t=0$) to $-\infty$ at pure noise ($t=1$)
* $p_t$, $\nabla \log p_t$: the marginal density of $\mathbf{x}_t$ at time $t$ and its score function
* $\hat{\mathbf{x}}_0$, $\hat{\boldsymbol{\epsilon}}$: the posterior expectations $\mathbb{E}[\mathbf{x}_0 \mid \mathbf{x}_t]$ and $\mathbb{E}[\boldsymbol{\epsilon} \mid \mathbf{x}_t]$ predicted by a denoising network
* $\mathbf{u}_t$, $\mathbf{u}_{\boldsymbol{\theta}}$, $\hat{\mathbf{u}}$: the probability-flow velocity field that transports samples along the path, a neural network parameterized by $\boldsymbol{\theta}$ trained to approximate it, and that network's predicted velocity
* $\mathbf{v} = \alpha_t \boldsymbol{\epsilon} - \sigma_t \mathbf{x}_0$: the target vector in $\mathbf{v}$-prediction (:numref:`sec_diffusion-gaussian-paths`)
* $\tilde{\mathbf{u}}$: the classifier-free guided velocity field (:numref:`subsec_diffusion-images-guided`)
* $\bar{\mathbf{u}}(\mathbf{x}_t, s, t)$, $\mathbf{u}_{\textrm{ref}}$: the average velocity between times $t$ and $s$, and the reference velocity (either a teacher model's prediction $\hat{\mathbf{u}}$ or the conditional velocity $\boldsymbol{\epsilon} - \mathbf{x}_0$) used in few-step distillation losses (:numref:`sec_diffusion-few-step`)
* $c$, $\gamma$: a conditioning signal and the guidance scale (:numref:`subsec_diffusion-images-guided`)



[Discussions](https://d2l.discourse.group/t/25)
