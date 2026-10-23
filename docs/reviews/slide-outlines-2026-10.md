# Proposed Lecture Outlines, 2026-10

Companion to `slide-review-2026-10.md`. For the weakest deck of each chapter,
the reviewers proposed an outline to `docs/slide-guide.md`: one claim per
slide, with the evidence that carries it. These are starting points for
rebuilding each deck together with its chapter; numbers come from the current
text and stored outputs and must be re-checked against the outputs when the
deck is built. Covers and dividers are omitted.

## Ch. 1 `linear-algebra` (running example: $X \in \mathbb{R}^{256\times784}$, $W\in\mathbb{R}^{784\times10}$)

1. One matrix product turns 256 images into 2,560 class scores — shape diagram.
2. Goals: read shapes, predict product shape and cost, measure vectors, find dominant directions; skip inverses — box.
3. A row is a vector, a batch a matrix, an image batch has order 4 — rank ladder.
4. Reductions delete an axis; `keepdims` broadcasts back — diagram; standardize $X$ in one line.
5. The normalized dot product is a cosine — angle diagram with four values.
6. Cauchy–Schwarz keeps $|\cos\theta|\le 1$ — three-line proof.
7. $Ax$ takes one dot product per row: a linear layer is ten templates — columns of $W$.
8. Matrices move space — unit square under rotation, scaling, shear.
9. $AB$ costs $2mkn$ FLOPs: about 4 MFLOP here; $X$ is 0.8 MB — calculation.
10. Check: does $(256\times784)(10\times784)$ multiply?
11. $\ell_2=5$, $\ell_1=7$ for $(3,-4)$ — two rulers.
12. Norms reappear as weight decay, sparsity, and clipping — table.
13. Eigenvectors are only stretched — ellipse with axes.
14. Power iteration finds $\lambda_{\max}$ — four lines of code and a growth plot.
15. Products of $L$ matrices scale like $|\lambda|^L$: $1.1^{50}\approx117$, $0.9^{50}\approx0.005$.
16. Check: the cost of a $4096^3$ matmul ($\approx1.4\times10^{11}$).
17. Recap: shape rule, cost rule, cosine, eigen growth.

## Ch. 2 Linear regression in practice (replaces `oo-design`, `synthetic-regression-data`, `linear-regression-scratch`)

1. If you know the answer, you can test the training loop — generator and scatter.
2. Goals and scope (framework internals skipped).
3. Every model plugs into Module, DataModule, and Trainer — three boxes.
4. 1,000 examples in batches of 32 make 32 batches, the last of 8 — shuffle diagram.
5. The sampler is six lines — trimmed code.
6. The forward pass is one matrix–vector product — shapes $(32\times2)(2\times1)$.
7. The gradient is the error-weighted input — derivation and picture.
8. Check: $\partial\ell/\partial w$ for $x=(1,2)$ and $\hat y-y=0.5$.
9. SGD steps, then zeroes gradients — five lines.
10. The noise floor is $\sigma^2/2=5\times10^{-5}$ — calculation.
11. The training loss reaches the floor — log-scale plot.
12. $\hat w$ lands within about $10^{-4}$ of $[2,-3.4]$, as $\sigma/\sqrt n$ predicts — table and estimate.
13. Too large a step diverges, too small crawls — three curves.
14. Logging must not stall the step — queue diagram.
15. Recap.

## Ch. 3 `environment-and-distribution-shift`

1. Right on the test set, wrong in deployment — two panels.
2. Goals and scope.
3. Nothing transfers without an assumption — label-flip counterexample.
4. Three shifts, three invariants — table of what moves and what stays.
5–7. Covariate, label, and concept shift — one picture each.
8. Check: classify a new camera, an outbreak, new diagnostic criteria.
9. Reweighting fixes the risk: $E_p[\ell]=E_q[\beta\ell]$ — derivation.
10. A source-against-target classifier estimates $\beta$ — derivation.
11. Running demo: source $N(0,I)$, target shifted to $(2,0)$ — scatter.
12. The learned log-ratio matches the true $2x_1-2$ — table.
13. Weights explode where the source is thin — density-ratio figure.
14. Target accuracy rises from about 50% to about 93% — bars.
15. Label shift: solve $C\,p(y)=\mu(\hat y)$ — worked $2\times2$ case.
16. Concept shift needs fresh labels; predictions feed back — diagrams.
17. Recap: shift → fix table.

## Ch. 4 `generalization-deep`

1. Our MLP has 3.4 parameters per training image and still generalizes.
2. Goals; scope (no NTK or PAC-Bayes).
3. The same network fits shuffled labels — accuracy plot (new run).
4. A capacity bound says nothing here: $\sqrt{d/n}\approx1.8$.
5. Every learner bets on an inductive bias — table.
6. The classical U-curve — measured polynomial curve.
7. Past interpolation, test error falls again — measured min-norm curve.
8. The peak sits where parameters equal examples — conditioning.
9. Check: how can more data hurt?
10. Gradient descent from zero picks the minimum-norm interpolant — diagram.
11. Logistic gradient descent converges to the max-margin separator — diagram.
12. Clean labels are fit first, noisy ones later — plot with 20% label noise.
13. Early stopping keeps the clean fit — the same plot with the stop.
14. Recap: known and open.

## Ch. 5 Computation (no decks today; two lectures)

**A. A model is a module tree plus state:** the module tree; parameters
against buffers; 16 bytes per parameter for fp32 Adam; tied and frozen
parameters; activation spread against depth (why scaled initialization);
RMSNorm as the worked custom layer; checkpoints store state, not code.

**B. Numbers, devices, memory:** range against precision for fp32, tf32,
bf16, fp16, fp8; overflow and loss scaling; what fills GPU memory (16 bytes
per parameter plus activations); checkpointing; asynchronous execution;
seeds and the cost of determinism; hooks and a NaN finder. Running example:
an 8B-parameter model, 16 GB in bf16 but about 130 GB for full Adam
fine-tuning; check slide "Will it fit?".

## Ch. 6 `lenet` (running example: one 28×28 image)

1. LeNet maps a 28×28 image to 10 scores with 62k weights — diagram.
2. Goals: compute shapes, parameters, and multiply-adds of any conv stack.
3. The encoder trades space for channels — shape table.
4. Check: $6\times14\times14$ through sixteen $5\times5$ filters.
5. One layer holds 78% of the weights — parameter table.
6. The convolutions hold 4% of the weights and do 86% of the multiply-adds — table.
7. LeNet needs 3× fewer weights than the MLP — bars.
8. Ten lines of `Sequential`.
9. Sigmoid LeNet can stall before it learns — training curve.
10. LeNet against the MLP after 10 epochs — rounded table.
11. Global average pooling would cut LeNet to 2.7k weights — calculation.
12. Check: a $3\times3$, 64→128 conv on $56\times56$.
13. Recap: three formulas; next, AlexNet.

## Ch. 7 `batch-norm` (running example: LeNet, batch 128)

1. Sigmoid LeNet stalls; with batch norm it does not — two curves.
2. Goals and scope (the covariate-shift debate skipped).
3. BN standardizes each channel over a minibatch — $(N,C,H\!\cdot\!W)$ cube.
4. One channel's mean pools 73,728 numbers — calculation.
5. $\gamma$ and $\beta$ can undo the standardization.
6. Training uses batch statistics, inference running averages — two paths.
7. Check: BN in train mode with a batch of one.
8. The core is five lines.
9. BN adds four numbers per channel — 452 parameters.
10. At inference BN folds into the convolution — algebra.
11. LeNet with BN reaches about 85% where plain LeNet reaches about 70% — table.
12. With two images per GPU, BN's error rises and GroupNorm's does not — Wu and He (2018).
13. LayerNorm, InstanceNorm, GroupNorm move the averaging axes — four cubes.
14. Choose by whether the batch is reliable — table.
15. Recap.

## Ch. 8 `rnn` (running example: BPE-1024, embedding 64, hidden 128)

1. A window forgets everything older than τ; a state does not.
2. Goals and scope.
3. One update rule at every step — equation with shapes.
4. Unrolled, an RNN is a deep network with tied weights.
5. The state is 512 bytes for any prefix length.
6. One matmul on $[x_t,h_{t-1}]$ equals two — three lines of code.
7. An RNN language model is embedding, recurrence, head — shape diagram.
8. 89% of the parameters face the vocabulary — 222k total.
9. Per token, the head costs 5× the recurrence.
10. Check: which terms grow 4× when the hidden size doubles?
11. Training predicts every next token from the true prefix.
12. A uniform guess has perplexity 1,024.
13. Ten epochs reach about 2.4 bits per byte, below the BPE trigram's 2.6.
14. Generation feeds back its own outputs; greedy text loops.
15. Gradients cross every step: $\rho^{40}$ is 0.015 or 45.
16. Recap and map.

## Ch. 9 `minibatch-sgd`

1. The batch is a hardware decision before it is a statistical one.
2. One update, three prices — cost, variance, updates per epoch for $b=1,100,1500$.
3. A GPU does hundreds of FLOPs per byte it fetches — calculation.
4. Arithmetic intensity decides who waits — roofline sketch.
5. Python dispatch costs microseconds; a dot product nanoseconds — calculation.
6. The same work runs about 500× apart — log bar chart.
7. Check: how many columns per call before dispatch stops dominating?
8. A minibatch gradient is one matmul — shape diagram.
9. Averaging $b$ gradients cuts variance $b$-fold — formula and plot.
10. On the wall clock, $b\approx10$–100 wins — overlay of loss against time.
11. Every optimizer is (params, states, hyperparameters) — six lines.
12. Both effects saturate — throughput and steps against $b$.
13. Recap.

## Ch. 10 `attention-scoring`

1. Goals and scope.
2. The Gaussian kernel leaves a dot product — derivation.
3. At $d=64$ a typical score is ±8, not ±1.
4. Unscaled scores collapse softmax entropy — plot.
5. Dividing by $\sqrt d$ flattens it — takeaway.
6. Attention is two matmuls and a softmax — shape diagram.
7. One GPT-2 head at $n=1024$: 1M scores, 4 MiB, 0.27 GFLOP.
8. Check: score memory at $n=8192$ (256 MiB).
9. A ragged batch needs a key mask; query $t$ sees keys $\le t$ — two masks.
10. Masks compose by AND under broadcasting — diagram.
11. Use `finfo.min`: $-10^6$ overflows fp16 and $-\infty$ gives NaN rows — table.
12. `masked_softmax` in six lines; masked weights are exactly zero.
13. An empty row is a bug — example.
14. Additive scoring lost to the dot product on hardware — table.
15. Recap.

## Ch. 11 `gpt` (running example: GPT-2 small)

1. Goals: count parameters, estimate cost with $6ND$, diagnose a gap, load a checkpoint.
2. A GPT is an embedding, $L$ identical blocks, and a tied head — stack diagram.
3. Tying saves 39M of GPT-2's 124M.
4. Each block costs $12d^2$ — bar chart, 85M total.
5. Check: estimate GPT-2 medium (355M).
6. Causality is a flag on the fused kernel; positions are a table or a rotation.
7. The forward pass in eight lines.
8. $6ND$: $5\times10^{14}$ here, $7\times10^{18}$ for GPT-2 — log bars.
9. Validation turns up while training loss falls — memorization by calculation.
10. Post-norm sticks at the unigram entropy — two curves.
11. Temperature and top-$k$ reshape one distribution — three panels.
12. Naive generation costs about $N\cdot T^2$ — leads into the KV cache.
13. A checkpoint loads with a name map and one transpose; perplexity confirms it.
14. Recap.

## Ch. 12 State Space Models (two lectures)

Running example: a Mamba-2 layer's state (about 1.3 MB) against a GQA cache
of 4 KB per token. Teaching task: associative recall.

*Lecture 1.* Attention's memory grows with context, a recurrent layer's does
not; goals; past about 320 tokens the state is the smaller memory; every model
here is a transition, a write, and a readout; gates fixed vanishing gradients
but kept training sequential; taking the state out of the nonlinearity makes
the update affine; affine maps compose, so $T$ steps take $\log_2T$ rounds;
the step size Δ is a gate; check: the half-life of $a=0.99$; fixed dynamics
weigh the past by position, not content; input-dependent Δ solves selective
copying; selectivity breaks the convolution but keeps the scan.

*Lecture 2.* Linear attention, Mamba-2, and GLA are one matrix-state
recurrence; an additive memory holds about $d_k$ pairs; a gated recurrence is
masked attention in another order; chunks turn the recurrence into matmuls;
adding cannot re-bind a key; the delta rule is one gradient step on the recall
loss; every layer solves one regression with a different solver; one
attention layer in four restores recall; recap table of state, update,
readout, cost.

## Ch. 13 `memory-precision` (running example: a 1B-parameter GPT on one 24 GB GPU)

1. A 1B model fills a 24 GB GPU before it sees a token — 16 B per parameter.
2. Goals and scope.
3. Training memory is $16P$ of state plus activations — anatomy and table.
4. Adam costs twice the weights — SGD 8P, momentum 12P, Adam 16P.
5. Activations grow with tokens × layers × width — about 1.4 MB per token.
6. Peak memory comes at the start of backward — measured trace.
7. Check: how many tokens fit in the remaining 8 GB?
8. bf16 keeps fp32's range and gives up precision — bit layouts.
9. Only fp16 needs loss scaling — range diagram.
10. Checkpointing buys about 17× more tokens for about ⅓ more compute.
11. Accumulation gives a large batch at small memory, never faster.
12. Sharding: 16 GB over 4 GPUs is 4 GB each.
13. Check: does context 4096 × batch 8 train?
14. Symptom → fix → price — table; recap.

## Ch. 14 `qlearning` (running example: slippery FrozenLake, $\gamma=0.95$)

1. Goals: write the TD update, explain its convergence, predict ε-greedy regret.
2. Value iteration needs $P$; an agent only sees transitions — backup diagrams.
3. Running example card: $V^*(s_0)=0.180$; the optimal policy wins 73.6%.
4. Replace the expectation by one sample — one transition worked with numbers.
5. Mask the bootstrap on termination, not truncation — two timelines.
6. The algorithm is five lines.
7. In expectation, a TD step is a value-iteration step.
8. Q-learning ends within 0.02 of value iteration — table.
9. Exploring at ε=0.05 costs the behavior policy 20 points — bars.
10. Check: which step-size schedule for 4,000 episodes?
11. A bandit prices learning as regret; fixed ε pays a linear tax.
12. Optimism and Thompson sampling — regret bars, log scale.
13. The max makes Q-learning off-policy; SARSA is on-policy.
14. The same max overestimates — dot plot.
15. Recap; next, a network replaces the table.

## Ch. 15 `sac` (running example: Pendulum)

1. Goal: swing up Pendulum in under 10k steps where REINFORCE failed in 480k.
2. Running example card and budget.
3. Two independent choices: replay reuses data, pathwise gradients cut variance — 2×2 grid placing the methods.
4. Add entropy to every reward — the objective.
5. With a uniform reference, the entropy bonus is the KL penalty already seen.
6. The soft target subtracts $\alpha\log\pi$ at the next state.
7. Policy improvement is a KL projection onto $\exp(Q/\alpha)/Z$ — plot.
8. Reparameterizing sends the gradient through the critic: variance 21.5 against 1.0.
9. Squashing needs a change-of-variables term — density picture and a stable form.
10. Twin critics against overestimation; Polyak targets with a 139-update half-life.
11. The whole update in eight lines.
12. SAC crosses −200 in 5–8k steps — return plot.
13. Check: with $\alpha=0$, what algorithm remains?
14. Recap and place on the data-rules map.

## Ch. 16 `dcgan` (running example: 40,597 sprites)

1. Goals: build, diagnose, and score an image GAN.
2. DCGAN made the classic loss train by constraining the architecture — block diagram.
3. Architecture cannot remove rotation or mode-dropping.
4. A modern backbone: learned constant, bilinear resampling, no batch norm — diagrams.
5. The whole loss in eight lines.
6. Same backbone, same budget, only the loss differs — the classic loss collapses.
7. Check: which run would you stop, from the curves alone?
8. FID is a $W_2$ distance between feature Gaussians — worked 1-D example.
9. KID, its bias, and what the feature network can see — table.
10. Precision and recall separate fidelity from coverage.
11. No copying: nearest-neighbour distances — pairs.
12. Recap: five rules for an image GAN.

## Ch. 17 `ddpm` (running example: the three-mode mixture)

1. Goals: write the DDPM loss and sampler.
2. One process replaces the noise ladder — diagram.
3. Each step shrinks by $\sqrt{1-\beta_t}$ and adds variance $\beta_t$ — defines $\alpha_t$, $\bar\alpha_t$.
4. Any step is one draw: $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\,\varepsilon$.
5. After 200 steps the data are scaled by 0.006 — table.
6. Check: at which $t$ do signal and noise have equal variance?
7. The reverse chain is a latent-variable model; the bound splits into three parts.
8. Given $x_0$, the reverse step is a known Gaussian — diagram.
9. Matching means means predicting ε.
10. Training is five lines.
11. ε-prediction is a scaled score.
12. Sampling is 200 Langevin-like steps — six lines.
13. Ancestral sampling recovers the mode weights — table.
14. Recap: four equations; sources.

## Ch. 24 `geometry-linear-algebraic-ops` (running examples: t-shirts against trousers; $A=[[1,2],[-1,3]]$)

An image is a point and an arrow in $\mathbb{R}^{784}$; $v\cdot w=\|v\|\|w\|\cos\theta$;
Cauchy–Schwarz by a discriminant; projection worked by hand; $P=QQ^\top$ and
least squares; random directions have cosine about $\pm1/\sqrt d$; check: two
random 784-dimensional vectors; why attention divides by $\sqrt d$; signed
distance to a hyperplane; one dot product per image is right about 92% of the
time; a matrix is fixed by where it sends $e_1,e_2$; dependent columns
collapse the plane; $\det A$ is a signed area (5); area scalings multiply;
$\det=0$ means no inverse (use `solve`); orthogonal matrices preserve lengths;
every product is one `einsum` pattern; check: $[[1,1],[2,2]]$ by hand; recap.

## Ch. 25 `multivariable` (running example: $L(z)=\log(e^{z_1}+e^{z_2})$ at $z_0=(0,\log2)$)

The gradient's dot product with a step is the first-order change;
$\nabla L=(1/3,2/3)$ by hand; the linear prediction misses by $1.8\times10^{-4}$;
directional slope; steepest descent; the gradient is normal to level sets; a
zero gradient is necessary, not sufficient; Lagrange tangency; check: maximize
$x+2y$ on the circle; the chain rule sums over paths; backprop on a five-node
graph by hand; forward against reverse cost; `backward()` runs the same sweep;
the Hessian by hand explains the $1.8\times10^{-4}$ gap; eigenvalue signs
classify critical points; most high-dimensional critical points are saddles;
check: classify $x^2-3xy+y^2$; recap.

## Ch. 26 `adaptive-stochastic-methods` (running problem: $\tfrac12(x_1^2+1000x_2^2)$ and its noisy version)

One step size is held hostage by the stiffest weight (GD 6,160 steps, Adam
344); minibatch gradients are unbiased with variance $\sigma^2/b$; noise adds a
term to the descent lemma; balancing gives the $1/\sqrt K$ rate; 10× smaller
gradient norm costs 100× the steps (log–log plot); check; per-coordinate steps
solve a diagonal quadratic; AdaGrad, RMSProp, bias correction (the first step
is 3.16× too large without it); Adam in six lines; rotating the problem
removes Adam's advantage; Reddi's counterexample; $\ell_2$ through Adam against
AdamW; a constant step stalls at a noise floor; warm-up, stable, decay; recap
with the preconditioner ladder.

## Ch. 27 `concentration-generalization` (running example: test accuracy to ±1% at 95%)

Chebyshev needs 50,000 test examples; polynomial tails make high confidence
unaffordable (table); Chernoff in four derivation steps; Hoeffding certifies
±1% with 18,445 examples; check: ±0.5% at 99%; on a fair coin Chebyshev is
eight orders of magnitude loose; a 784-dimensional Gaussian lives on a shell
of radius 28; random directions are nearly orthogonal; picking the best model
on the data breaks Hoeffding; 2,000 configurations widen ±1.4% to ±2.4%;
Rademacher complexity; norm-bounded classes cost $Br/\sqrt n$; networks that
fit random labels make class bounds vacuous; test error peaks at $p=n$;
minimum norm shrinks past the peak; recap.

## Ch. 28 `divergences-distances` (running pair: $P=(0.6,0.3,0.1)$, $Q=(0.2,0.5,0.3)$)

Three families ask three questions (ratio, test function, mass moved); Jensen
proves f-divergences non-negative; on one pair the family spans a factor of
10 (KL 0.396, $\chi^2$ 1.013, TV 0.40, JS 0.091); check: TV by hand; near
$P=Q$ every f-divergence is a scaled $\chi^2$; processing loses
distinguishability; every f-divergence is a game against a critic; the
optimal discriminator gives Jensen–Shannon; forward KL covers modes, reverse
KL picks one; on disjoint supports JS freezes while $W_1$ grows; $W_1$ is the
area between CDFs (0.6); Pinsker; MMD without a critic; Fisher divergence
ignores the normalizer; each generative objective is a divergence (table);
recap.

## Ch. 29 `sdes` (running example: an OU process on bimodal data)

Diffusion needs a noise process that forgets any dataset; Brownian motion
takes $\sqrt{\Delta t}$ steps and has no derivative; squared increments add up
to time; Itô's table; Itô's lemma in two derivation steps; $d(W^2)=2W\,dW+dt$;
check: $E[W_t^4]$; the Itô integral's mean and variance; Euler–Maruyama in six
lines; strong against weak error and their orders (log–log plot); OU balances
drift and noise at $\sigma^2/2\theta$; Itô solves OU in two lines; by $t=1$ the
modes have merged (table); $\sigma^2=2\theta$ is DDPM's forward marginal;
recap.
