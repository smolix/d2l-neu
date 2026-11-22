# Multilayer Perceptrons
:label:`chap_perceptrons`

In this chapter, we introduce your first truly *deep* neural networks. In :numref:`chap_classification`, softmax regression assigned a fixed weight to each of the 784 pixels of a Fashion-MNIST garment image and summed the evidence—meaning every pixel contributed the exact same vote toward a class regardless of what surrounded it. Yet a bright pixel might belong to a sleeve, a handbag strap, or a collar, and only its neighbors can reveal which. Even simpler, no straight line—no single linear layer—can compute the exclusive-OR (XOR) of two binary bits, a barrier Marvin Minsky and Seymour Papert famously highlighted in 1969 for Frank Rosenblatt's single-layer *perceptron*.

*Multilayer perceptrons* (MLPs) break through this linear barrier by stacking one or more fully connected *hidden layers*—each pairing an affine transformation with a nonlinear activation function—before the output layer, and training all layers jointly end to end via *backpropagation* :cite:`Rumelhart.Hinton.Williams.ea.1988`. While the cross-entropy loss, minibatch SGD optimizer, and `Trainer` loop from :numref:`chap_classification` carry straight over, adding hidden layers introduces three fundamental deep-learning challenges: computing gradients efficiently across hundreds of thousands of parameters, choosing initial weight scales so signals and gradients neither vanish nor explode across depth, and understanding why heavily overparameterized networks generalize instead of merely memorizing their training data.

This chapter unfolds in seven sections:

- **Multilayer Perceptrons (:numref:`sec_mlp`):** We show why linear models fail on feature interactions like XOR, solve XOR with a two-unit ReLU hidden layer, examine universal approximation and the exponential piece-multiplying power of depth, and compare standard activation functions (ReLU, GELU, sigmoid, and $\tanh$).
- **Implementation (:numref:`sec_mlp-implementation`):** We implement a 256-unit ReLU MLP both from scratch and concisely on Fashion-MNIST, cutting roughly a fifth of softmax regression's classification errors ($\sim 87\%$ vs. $\sim 84\%$ accuracy) at virtually identical GPU epoch time and visualizing the localized garment-part detectors learned by the hidden units.
- **Backpropagation (:numref:`sec_backprop`):** We derive the three matrix-calculus local rules of reverse-mode differentiation through an MLP, verify our hand-coded backward pass against `autograd` to $10^{-14}$ relative error, and tally the exact activation memory and FLOP cost of training versus inference.
- **Numerical Stability and Initialization (:numref:`sec_numerical_stability`):** We analyze how activations and gradients scale across $K$ layers, derive *Xavier* and *He* initialization to prevent vanishing and exploding gradients in a 10-layer MLP, and prove why random initialization is required to break hidden-unit permutation symmetry.
- **Generalization in Deep Learning (:numref:`sec_generalization_deep`):** We demonstrate random-label memorization, *double descent* past the interpolation threshold, the *implicit bias* of gradient descent (minimum-norm least squares and maximum-margin classification), and *early stopping* under label noise.
- **Dropout (:numref:`sec_dropout`):** We introduce inverted dropout masking and rescaling, implement it from scratch and concisely in a two-hidden-layer MLP, and test how it slows the memorization of corrupted labels.
- **Predicting House Prices on Kaggle (:numref:`sec_kaggle_house`):** We put the entire toolkit to work end to end on a real tabular regression competition (Ames Housing), covering leakage-free preprocessing, log-RMSE evaluation, and 5-fold cross-validated model selection.

```toc
:maxdepth: 2

mlp
mlp-implementation
backprop
numerical-stability-and-init
generalization-deep
dropout
kaggle-house-price
```

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §4]{.kicker}

Multilayer Perceptrons: Breaking the Linear Barrier<br>
**hidden layers · backpropagation · variance-preserving initialization · generalization · dropout · Kaggle**
:::
:::

::: {.slide title="Why Linear Classifiers Hit a Ceiling on Real Tasks"}
[The Central Problem]{.kicker}

::: {.cols .vc}
::: {.col}
- **No feature interactions (XOR):** Flipping input bit $x_1$ in exclusive-OR reverses the output depending on $x_2$; no single hyperplane can separate all four corners of $\{0, 1\}^2$.
- **Context-blind pixel weights:** In Fashion-MNIST, a bright pixel contributes the exact same vote toward a class whether its neighbors form a collar, a sleeve, or a handbag strap, capping linear accuracy at $\sim 84\%$.
- **The deep-learning solution:** Insert hidden layers $\mathbf{H} = \phi(\mathbf{X}\mathbf{W}^{(1)} + \mathbf{b}^{(1)})$ that learn nonlinear feature combinations jointly with the output classifier ($\sim 87\%$ accuracy).
:::

::: {.col .fig .big}
@fig:ch04-mlp-xor-ceiling
:::
:::
:::

::: {.slide title="Three Fundamental Challenges of Deep Networks"}
[What Depth Introduces]{.kicker}

::: {.cols .vc}
::: {.col}
1. **Efficient gradient computation (§4.3):** Finite differences would require $203{,}530$ forward passes per step; *backpropagation* computes every gradient in one reverse sweep by caching forward activations in GPU RAM.
2. **Numerical stability across depth (§4.4):** Across $K$ layers, signals and gradients pass through a product of $K$ weight and slope matrices, requiring variance-preserving *Xavier* and *He* initialization.
3. **Generalization past interpolation (§4.5–§4.6):** Overparameterized MLPs can memorize random noise yet generalize on real data via SGD's implicit bias, early stopping, and *dropout*.
:::

::: {.col .fig .big}
@fig:ch04-mlp-three-challenges
:::
:::
:::

::: {.slide title="Running Benchmark: Fashion-MNIST Across Chapter 4"}
[Running Example]{.kicker}

::: {.cols .vc}
::: {.col}
- **Softmax baseline ($784 \to 10$):** $7{,}850$ parameters ($0.8\textrm{ MB}$ activations at $n = 256$) plateau at $\sim 84\%$ validation accuracy.
- **1- and 2-hidden-layer ReLU MLPs:** Expand to $203{,}530$–$269{,}322$ parameters and reach $\sim 87\%$ accuracy at nearly the exact same GPU wall-clock time per epoch.
- **10-layer depth test ($795{,}658$ params):** Stays frozen at $10\%$ chance under $\mathcal{N}(0, 0.01^2)$ initialization, but trains cleanly to $\sim 85\%$ under He initialization ($\tau = \sqrt{2/n_{\textrm{in}}}$).
:::

::: {.col .fig .big}
@fig:ch04-mlp-benchmark-ladder
:::
:::
:::

::: {.slide title="Architectural & Mathematical Roadmap of Chapter 4"}
[Chapter Roadmap]{.kicker}

::: {.cols .vc}
::: {.col}
- **Representation & features (§§4.1–4.2):** Nonlinear activations prevent linear collapse, fold space into $2^m$ pieces with depth, and learn localized garment-part filters.
- **Optimization & stability (§§4.3–4.4):** Three local VJP rules ($\mathbf{X}^\top\mathbf{G}$, $\mathbf{G}\mathbf{W}^\top$, $\odot\,\phi'$) run in $\sim 2\times$ forward FLOPs; setting $h\tau^2\mathbb{E}[\phi'(z)^2] = 1$ prevents vanishing/exploding products.
- **Generalization & tabular ML (§§4.5–4.7):** Double descent, minimum-norm implicit bias, inverted dropout $h' = \xi h / (1 - p)$, and leakage-free 5-fold Kaggle cross-validation.
:::

::: {.col .fig .big}
@fig:ch04-mlp-chapter-roadmap
:::
:::
:::

::: {.slide title="Concept Check & Study Guide for Chapter 4"}
[Check & Synthesis]{.kicker}

Before diving into §4.1, test your intuition on two foundational questions that connect the sections ahead:

1. **Linear collapse:** If we stack 10 fully connected layers of width $256$ without activation functions ($\phi(z) = z$), does the network have more expressive power than a single $784 \times 10$ linear layer?
2. **Memory vs. parameters:** When training a deep network on minibatches of size $n = 256$, why can activation memory easily exceed parameter memory even when every hidden layer is narrower than the input?

::: {.d2l-note .fragment .fade-up}
**Answers:**

1. **No:** $(\mathbf{X}\mathbf{W}^{(1)})\mathbf{W}^{(2)}\cdots\mathbf{W}^{(10)}$ multiplies out to a single matrix $\mathbf{X}\mathbf{W}$ of rank at most $256$ (and at most $10$ at the output). Nonlinear activations are essential to prevent algebraic collapse.
2. **Batch scaling:** A weight matrix $\mathbf{W}^{(k)} \in \mathbb{R}^{h \times h}$ stores $h^2$ floats independent of batch size, whereas its cached forward activation $\mathbf{H}^{(k)} \in \mathbb{R}^{n \times h}$ stores $nh$ floats per layer, growing linearly with both batch size $n$ and depth $K$.
:::
:::

