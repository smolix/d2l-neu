# Notation
:label:`chap_notation`

This book uses the following notational conventions. Some entries describe a
class of symbols, while others define one particular symbol. The indefinite
article “a” marks a convention: for example, “$x$: a scalar” means that
lowercase letters generally denote scalars. The definite article marks a fixed
meaning: “$\mathbb{Z}$: the set of integers” refers specifically to
$\mathbb{Z}$.



## Numerical Objects

* $x$: a scalar
* $\mathbf{x}$: a vector
* $(x_1, \ldots, x_n)$: the column vector with entries $x_1, \ldots, x_n$, written inline; the entries of a matrix row are listed the same way, as in "the rows $(0, 1, 2)$ and $(3, 4, 5)$"
* $\mathbf{X}$: a matrix
* $\mathsf{X}$: a general tensor
* $\mathbf{I}$: the identity matrix (of some given dimension), i.e., a square matrix with $1$ on all diagonal entries and $0$ on all off-diagonals
* $x_i$, $[\mathbf{x}]_i$: the $i^\textrm{th}$ element of vector $\mathbf{x}$
* $x_{ij}$, $x_{i,j}$, $[\mathbf{X}]_{ij}$, $[\mathbf{X}]_{i,j}$: the element of matrix $\mathbf{X}$ in row $i$ and column $j$



## Set Theory


* $\mathcal{X}$: a set
* $\mathbb{Z}$: the set of integers
* $\mathbb{Z}^+$: the set of positive integers
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
* $\mathbf{1}$: the vector of all ones (of some given dimension); unlike the indicator function below, it takes no argument
* $\mathbf{1}(\cdot)$: the indicator function, equal to $1$ when its Boolean argument is true and $0$ otherwise
* $\mathbf{1}_{\mathcal{X}}(z)$: the set-membership indicator, equal to $1$ when $z\in\mathcal{X}$ and $0$ otherwise
* $\mathbf{(\cdot)}^\top$: transpose of a vector or a matrix
* $\mathbf{X}^{-1}$: inverse of matrix $\mathbf{X}$
* $\odot$: Hadamard (elementwise) product
* $[\cdot, \cdot]$: concatenation
* $\|\cdot\|_p$: $\ell_p$ norm
* $\|\cdot\|$: $\ell_2$ norm
* $\|\mathbf{X}\|_2$: spectral norm of matrix $\mathbf{X}$, its largest singular value
* $\|\mathbf{X}\|_F$: Frobenius norm of matrix $\mathbf{X}$, the $\ell_2$ norm of its entries
* $\|\mathbf{X}\|_*$: nuclear norm of matrix $\mathbf{X}$, the sum of its singular values
* $\sigma_i(\mathbf{X})$: the $i^\textrm{th}$ largest singular value of matrix $\mathbf{X}$
* $\mathbf{X}^{+}$: Moore–Penrose pseudoinverse of matrix $\mathbf{X}$
* $\kappa(\mathbf{X})$: condition number of matrix $\mathbf{X}$, the ratio of its largest to its smallest nonzero singular value
* $\rho(\mathbf{X})$: spectral radius of a square matrix $\mathbf{X}$, the largest modulus of its eigenvalues (not the correlation $\rho(X, Y)$ below)
* $\operatorname{rank}\mathbf{X}$, $\operatorname{tr}\mathbf{X}$, $\det\mathbf{X}$: rank, trace, and determinant of matrix $\mathbf{X}$
* $\mathbf{X}\succeq 0$, $\mathbf{X}\succ 0$: the symmetric matrix $\mathbf{X}$ is positive semidefinite, positive definite
* $\langle \mathbf{x}, \mathbf{y} \rangle$: inner (dot) product of vectors $\mathbf{x}$ and $\mathbf{y}$
* $\sum$: summation over a collection of elements
* $\prod$: product over a collection of elements
* $\stackrel{\textrm{def}}{=}$: an equality asserted as a definition of the symbol on the left-hand side



## Calculus

* $\frac{dy}{dx}$: derivative of $y$ with respect to $x$
* $\frac{\partial y}{\partial x}$: partial derivative of $y$ with respect to $x$
* $\nabla_{\mathbf{x}} y$: gradient of $y$ with respect to $\mathbf{x}$
* $\int_a^b f(x) \;dx$: definite integral of $f$ from $a$ to $b$ with respect to $x$
* $\int f(x) \;dx$: indefinite integral of $f$ with respect to $x$



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
* $\sigma_X$: standard deviation of random variable $X$
* $\textrm{Var}(X)$: variance of random variable $X$, equal to $\sigma^2_X$
* $\textrm{Cov}(X, Y)$: covariance of random variables $X$ and $Y$
* $\rho(X, Y)$: the Pearson correlation coefficient, $\frac{\textrm{Cov}(X, Y)}{\sigma_X \sigma_Y}$
* $H(X)$: entropy of random variable $X$
* $D_{\textrm{KL}}(P\|Q)$: the KL divergence (relative entropy) from distribution $P$ to distribution $Q$



## Reinforcement Learning

These symbols belong to :numref:`chap_reinforcement_learning` and the chapter after it; :numref:`tab_rl_notation` lists them with the section that defines each.

* $\mathcal{S}$, $\mathcal{A}$: the sets of states and actions
* $s_t$, $a_t$, $r_t$: the state, action, and reward at step $t$
* $P(s' \mid s, a)$: the transition kernel, the probability of reaching state $s'$ from state $s$ under action $a$
* $r(s, a)$: the expected reward of action $a$ at state $s$
* $\gamma$: the discount factor
* $\mu_0$: the start-state distribution (for language models, the distribution of prompts)
* $\tau$: a trajectory $(s_0, a_0, r_0, s_1, \ldots)$
* $R(\tau)$: the discounted return $\sum_t \gamma^t r_t$ of a trajectory $\tau$
* $\pi(a \mid s)$: a policy, the probability of action $a$ at state $s$; $\pi(s)$ is the action of a deterministic policy
* $\pi_\theta$: a policy with parameters $\theta$
* $J(\theta)$: the expected return of $\pi_\theta$
* $V^\pi(s)$, $Q^\pi(s, a)$: the expected return under $\pi$ from state $s$, and from state $s$ after action $a$
* $V^*$, $Q^*$: the value functions of an optimal policy
* $A^\pi(s, a)$: the advantage $Q^\pi(s, a) - V^\pi(s)$
* $\mathcal{T}$: the Bellman operator
* $d^\pi_t$, $d^\pi_\gamma$: the distribution of the state at step $t$ under $\pi$, and its discounted average over steps
* $\delta$, $\delta_t$: the temporal-difference error of one transition, of the transition at step $t$
* $\alpha$: the step size of a temporal-difference update
* $\eta$: the step size of a policy update
* $\eta_V$: the step size of a value-baseline update
* $\epsilon$: the exploration rate of $\epsilon$-greedy action selection
* $\varepsilon$: the per-step probability that a cloned policy disagrees with the expert (not the exploration rate $\epsilon$)
* $\kappa$: the weight of an uncertainty bonus (not the condition number $\kappa(\mathbf{X})$)
* $\hat{G}_t$: the reward-to-go, the discounted sum of the rewards from step $t$ on
* $b$: a baseline subtracted from a return
* $K$: the number of responses sampled per prompt
* $\lambda$: the weight of the $\lambda$-return
* $w$, $w^-$: the parameters of a value network, and of its frozen or Polyak-averaged copy
* $n$: the depth of an $n$-step target, or of a plan
* $\rho_t$: the importance ratio $\pi_\theta(a_t \mid s_t) / \pi_{\theta_{\textrm{old}}}(a_t \mid s_t)$ at step $t$
* $\epsilon_{\textrm{clip}}$: the half-width of the clip band of a probability ratio
* $\Delta_{\textrm{KL}}$: a KL budget per update
* $\pi_{\textrm{ref}}$: a reference policy
* $\beta$: the weight of a KL penalty to a reference policy; soft actor-critic's temperature
* $\hat{r}$: a fitted reward
* $\nu$: the Polyak rate of a target network
* $\hat{f}_\phi$: a learned transition model



## Generative Adversarial Networks

These symbols belong to :numref:`chap_gans`; :numref:`tab_gan_notation` lists them with the section that defines each and the letters the papers use. Where the chapter departs from the usual letter, a parenthesis gives the papers' letter and the reason.

* $p$, $q$, $m$: the data density, the generator's density, and their balanced mixture $\tfrac12(p + q)$
* $x \sim p$, $x' \sim q$: a real and a generated sample (papers often write $y$ or $\hat x$ for the generated sample; $y$ is the source label here)
* $y$: the source label of a sample, $y = 1$ for real and $y = 0$ for generated
* $z$, $G$, $d_z$: the latent variable, the generator $x' = G(z)$, and the latent dimension (papers often write $k$; $k$ is a kernel here)
* $D$: the critic's realness logit; $\sigma(D)$ is its probability that a sample is real (papers say discriminator; "critic" also covers the rankers and witness functions of later sections, and R3GAN's fakeness logit is $-D$)
* $V$, $V_\ell$: the value function of the log-loss game, and of the game with payoff $\ell$
* $\rho = p/q$, $\lambda = \log \rho$: the density ratio and its logarithm, the optimal log-loss critic
* $\mu$: a mean, of a Gaussian or of features; the separation of $\mathcal{N}(0, 1)$ and $\mathcal{N}(\mu, 1)$; $\mu(c)$ is a conditional mean (papers often write $\delta$ or $d$ for the separation; both are taken here)
* $\delta_\theta$: the point mass at $\theta$
* $b$, $b_q$: an additive shift, of a critic $D + b$, of the linear generator $z A_q + b_q$, of a feature map $b(c)$, or of a band
* $d(p, q)$, $d_x$: a discrepancy between two distributions, and the data dimension (papers write $d$ for the dimension; $d$ is the discrepancy here)
* $w_{\textrm{sat}}$, $w_{\textrm{ns}}$, $w_{\textrm{Rp}}$: the per-sample weights of the saturating, non-saturating and pairing generator updates
* $\eta = p/(p + q)$: the class posterior (the step size, often $\eta$ in papers, is $h$ here)
* $\ell$, $L(\eta)$, $\Delta_\ell$: the payoff of a score ($\ell = \log \sigma$ for the log loss), its conditional Bayes risk, and its Bayes-risk gap (papers state the loss $-\ell$)
* $T$, $\mathcal{T}$, $\mathcal{F}$, $\ell_p$, $\ell_q$: a critic or test function, its class, the function class of an integral probability metric, and the scores of real and generated samples in the general adversarial objective (papers write $f$ for the test functions of an integral probability metric; $f$ is the divergence function here)
* $f$, $D_f$: a divergence function and its $f$-divergence
* $k$, $s$: a kernel and its bandwidth (papers write $\sigma$ or $\ell$ for the bandwidth; both are taken here)
* $\Phi$, $d_{\textrm{Rp}}$: the pairing objective and its value at the optimal critic (Jolicoeur-Martineau's divergence is $2\, d_{\textrm{Rp}}$)
* $P = p \otimes q$, $Q = q \otimes p$: the two orderings of a real--generated pair
* $(a_1, a_2)$, $\pi$: an ordered pair holding one real and one generated sample, and a permutation that pairs real with generated samples
* $\mathcal{E}_{\textrm{GAN}}$, $\mathcal{E}_{\textrm{diag}}$, $\mathcal{E}_{\textrm{all}}$: the classical, paired and all-pairs losses at the critic's best response, as functions of the generated points
* $v$, $\epsilon$: the direction and the size of a small change $q = p\,(1 + \epsilon v)$ (papers write $h$ for the direction; $h$ is the step size here)
* $\gamma$, $R_1$, $R_2$: the penalty weight and the zero-centered gradient penalties at the data and generated samples
* $h$, $\nu_{1,2}$, $\psi$, $\mathbf{u}$: the step size, the eigenvalues, the critic's slope and the gradient field of the Dirac-GAN, the two-point game (Mescheder et al. write $\lambda_{1,2}$ and $v$; $\lambda$ is the log ratio and $v$ a direction here)
* $\omega$: a wavenumber or angular frequency (papers write $k$ for the wavenumber; $k$ is a kernel here)
* $\phi$: a feature map
* $p_{\textrm{aug}}$: the probability of applying an augmentation (ADA writes $p$, the data density here)
* $c$, $e_c$, $a(c)$, $b(c)$, $\psi_0$, $\zeta_c$: a condition, its embedding, a modulation's scale and shift, the unconditional head of a projection critic, and the critic's free constant per class (papers write $\gamma(c)$, $\beta(c)$ for the scale and shift; $\gamma$ is the penalty weight here)
* $g$: a predictor of a sample $x$ from a condition $c$ (regression papers write $f: x \mapsto y$; $f$ is the divergence function and $y$ the source label here, so the condition-and-sample letters of conditional generation are kept)
* $\xi$: standard normal noise (papers write $\epsilon$, a small number here)
* $\vartheta$: a phase (papers write $\varphi$, too close to the feature map $\phi$)
* $\mathrm{JS}(p, q)$, $D_{\textrm{J}}$, $H_b(t)$: the Jensen--Shannon divergence, the symmetrized KL divergence $D_{\textrm{KL}}(p\|q) + D_{\textrm{KL}}(q\|p)$, and the entropy of a coin with bias $t$



[Discussions](https://d2l.discourse.group/t/25)
