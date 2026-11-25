# Multilayer Perceptrons
:label:`chap_perceptrons`

In this chapter, we introduce your first *deep* neural networks. In :numref:`chap_classification`, softmax regression assigned a fixed weight to each of the 784 pixels of a Fashion-MNIST garment image and summed the evidence, so every pixel cast the same vote toward a class regardless of what surrounded it. Yet a bright pixel might belong to a sleeve, a handbag strap, or a collar, and only its neighbors can tell which. Even on four binary points, no single linear layer can compute the exclusive-OR (XOR) of two bits, the limitation Marvin Minsky and Seymour Papert highlighted in 1969 for Frank Rosenblatt's single-layer *perceptron*.

*Multilayer perceptrons* (MLPs) overcome this linear barrier by stacking one or more fully connected *hidden layers*—each pairing an affine transformation with a nonlinear activation function—before the output layer, and training all layers jointly via *backpropagation* :cite:`Rumelhart.Hinton.Williams.ea.1988`. We begin by showing why linear models fail on feature interactions like XOR, how hidden layers fold input space into exponentially many linear regions with depth, and how common activation functions (ReLU, GELU, sigmoid, and $\tanh$) behave under differentiation (:numref:`sec_mlp`). Implementing a 256-unit ReLU MLP from scratch and concisely on Fashion-MNIST (:numref:`sec_mlp-implementation`) eliminates roughly a fifth of softmax regression's classification errors ($\sim 87\%$ vs. $\sim 84\%$ accuracy) at nearly the same GPU epoch time, replacing blurred whole-garment templates with localized contrast detectors.

Adding hidden layers also raises three foundational questions of deep learning that occupy the rest of the chapter. First, :numref:`sec_backprop` derives the three matrix-calculus rules of reverse-mode differentiation through an MLP, checks them against automatic differentiation to machine precision, and accounts for training's activation-memory and FLOP budgets. Second, :numref:`sec_numerical_stability` tracks signal and gradient variance across $K$ layers, derives *Xavier* and *He* initialization to prevent vanishing and exploding gradients in a 10-layer MLP, and shows why random weights are needed to break hidden-unit symmetry. Third, :numref:`sec_generalization_deep` and :numref:`sec_dropout` examine why heavily overparameterized networks generalize—covering random-label memorization, double descent, the implicit bias of gradient descent, early stopping, and multiplicative noise regularization via *dropout*—before :numref:`sec_kaggle_house` puts the full toolkit to work on Kaggle's Ames Housing regression benchmark.

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
1. **Efficient gradient computation:** Finite differences would require $203{,}530$ forward passes per step; *backpropagation* computes every gradient in one reverse sweep by caching forward activations in GPU RAM.
2. **Numerical stability across depth:** Across $K$ layers, signals and gradients pass through a product of $K$ weight and slope matrices, requiring variance-preserving *Xavier* and *He* initialization.
3. **Generalization past interpolation:** Overparameterized MLPs can memorize random noise yet generalize on real data via SGD's implicit bias, early stopping, and *dropout*.
:::

::: {.col .fig .big}
@fig:ch04-mlp-three-challenges
:::
:::
:::

::: {.slide title="Running Benchmark: Fashion-MNIST Across the Chapter"}
[Running Example]{.kicker}

::: {.cols .vc}
::: {.col}
- **Softmax baseline ($784 \to 10$):** $7{,}850$ parameters ($0.8\textrm{ MB}$ activations at $n = 256$) plateau at $\sim 84\%$ validation accuracy.
- **1- and 2-hidden-layer ReLU MLPs:** Expand to $203{,}530$–$269{,}322$ parameters and reach $\sim 87\%$ accuracy at nearly the exact same GPU wall-clock time per epoch.
- **10-layer depth test ($795{,}658$ params):** Stays frozen at $10\%$ chance under $\mathcal{N}(0, 0.01^2)$ initialization, but trains cleanly to $85\text{--}87\%$ under He initialization ($\tau = \sqrt{2/n_{\textrm{in}}}$).
:::

::: {.col .fig .big}
@fig:ch04-mlp-benchmark-ladder
:::
:::
:::

::: {.slide title="Architectural & Mathematical Roadmap of the Chapter"}
[Chapter Roadmap]{.kicker}

::: {.cols .vc}
::: {.col}
- **Representation & features:** Nonlinear activations prevent linear collapse, fold space into $2^m$ pieces with depth, and learn contrast patterns in place of whole-garment templates.
- **Optimization & stability:** Three local VJP rules ($\mathbf{X}^\top\mathbf{G}$, $\mathbf{G}\mathbf{W}^\top$, $\odot\,\phi'$) run in $\sim 2\times$ forward FLOPs; setting $h\tau^2\mathbb{E}[\phi'(z)^2] = 1$ prevents vanishing/exploding products.
- **Generalization & tabular ML:** Double descent, minimum-norm implicit bias, inverted dropout $h' = \xi h / (1 - p)$, and leakage-free 5-fold Kaggle cross-validation.
:::

::: {.col .fig .big}
@fig:ch04-mlp-chapter-roadmap
:::
:::
:::

::: {.slide title="Concept Check & Study Guide for the Chapter"}
[Check & Synthesis]{.kicker}

Before diving into the first section, test your intuition on two foundational questions that connect the sections ahead:

1. **Linear collapse:** If we stack 10 fully connected layers of width $256$ without activation functions ($\phi(z) = z$), does the network have more expressive power than a single $784 \times 10$ linear layer?
2. **Memory vs. parameters:** When training a deep network on minibatches of size $n = 256$, why can activation memory easily exceed parameter memory even when every hidden layer is narrower than the input?

::: {.d2l-note .fragment .fade-up}
**Answers:**

1. **No:** $(\mathbf{X}\mathbf{W}^{(1)})\mathbf{W}^{(2)}\cdots\mathbf{W}^{(10)}$ multiplies out to a single matrix $\mathbf{X}\mathbf{W}$ of rank at most $256$ (and at most $10$ at the output). Nonlinear activations are essential to prevent algebraic collapse.
2. **Batch scaling:** A weight matrix $\mathbf{W}^{(k)} \in \mathbb{R}^{h \times h}$ stores $h^2$ floats independent of batch size, whereas its cached forward activation $\mathbf{H}^{(k)} \in \mathbb{R}^{n \times h}$ stores $nh$ floats per layer, growing linearly with both batch size $n$ and depth $K$.
:::
:::

