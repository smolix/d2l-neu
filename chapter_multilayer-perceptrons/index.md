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

In Chapter 3, softmax regression scored each class by taking a fixed affine weighted sum of the $784$ input pixels, baking in two fundamental flaws:

- **No feature interactions (XOR):** Flipping input bit $x_1$ in exclusive-OR reverses the output depending on $x_2$; no single hyperplane can separate all four corners of $\{0, 1\}^2$.
- **Context-blind pixel weights:** In Fashion-MNIST, a bright pixel contributes the exact same vote toward a class whether its neighbors form a collar, a sleeve, or a handbag strap, capping linear accuracy at $\sim 84\%$.
- **The deep-learning solution:** Insert one or more fully connected *hidden layers* $\mathbf{H} = \phi(\mathbf{X}\mathbf{W}^{(1)} + \mathbf{b}^{(1)})$ that learn nonlinear feature combinations jointly with the output classifier.

::: {.d2l-note}
Adding a single 256-unit ReLU hidden layer lifts Fashion-MNIST accuracy to $\sim 87\%$ (cutting errors by $\sim 20\%$), while multiplying parameters 26-fold to $203{,}530$.
:::
:::

::: {.slide title="Three Fundamental Challenges of Deep Networks"}
[What Depth Introduces]{.kicker}

Moving from a single linear layer to stacked nonlinear layers preserves our cross-entropy loss and minibatch SGD loop, but introduces three new core problems:

1. **Efficient gradient computation (§4.3):** Finite differences would require $203{,}530$ forward passes per step; *backpropagation* computes every gradient in one reverse sweep at the cost of caching forward activations in GPU RAM.
2. **Numerical stability across depth (§4.4):** Across $K$ layers, signals and gradients pass through a product of $K$ weight and slope matrices, vanishing to $0$ or exploding to `NaN` unless initial weight variance matches the fan-in and activation (*Xavier* and *He* initialization).
3. **Generalization past interpolation (§4.5–§4.6):** With $203{,}530$ parameters for $60{,}000$ images ($3.4$ parameters per image), the MLP can memorize random noise, yet generalizes on real data via the implicit bias of gradient descent, early stopping, and *dropout*.
:::

::: {.slide title="Running Benchmark: Fashion-MNIST Across Chapter 4"}
[Running Example]{.kicker}

Throughout this chapter, we benchmark our models head-to-head on Fashion-MNIST ($d = 784$ pixels, $q = 10$ classes, minibatch size $n = 256$, learning rate $\eta = 0.1$):

| Architecture | Parameters | Multiply-Adds / Image | Activation Memory ($n = 256$) | Validation Accuracy |
|:--|--:|--:|--:|--:|
| Softmax regression ($784 \to 10$) | $7{,}850$ | $7{,}840$ | $0.8\textrm{ MB}$ | $\sim 84\%$ (30 epochs) |
| 1-hidden-layer MLP ($784 \to 256 \to 10$) | $203{,}530$ | $203{,}264$ | $1.1\textrm{ MB}$ | $\sim 87\%$ (30 epochs) |
| 2-hidden-layer Dropout MLP ($784 \to 256 \to 256 \to 10$) | $269{,}322$ | $268{,}800$ | $1.3\textrm{ MB}$ | $\sim 87\%$ (20 epochs) |
| 10-layer MLP with $\mathcal{N}(0, 0.01^2)$ init | $795{,}658$ | $793{,}088$ | $3.4\textrm{ MB}$ | $10\%$ (frozen at $\log 10$) |
| 10-layer MLP with He init ($\tau = \sqrt{2/n_{\textrm{in}}}$) | $795{,}658$ | $793{,}088$ | $3.4\textrm{ MB}$ | $\sim 85\%$ (10 epochs) |

::: {.d2l-note}
On a GPU, the 256-unit MLP trains in nearly the exact same wall-clock time per epoch as softmax regression because data loading and kernel launch overhead dominate dense matrix arithmetic at width $256$.
:::
:::

::: {.slide title="Architectural & Mathematical Roadmap of Chapter 4"}
[Chapter Roadmap]{.kicker}

Each section solves one concrete link in the chain from representation power to real-world tabular deployment:

| Section | Core Question | Key Mathematical / Systems Takeaway |
|:--|:--|:--|
| **§4.1 Multilayer Perceptrons** | What functions can hidden layers represent? | Without nonlinear $\phi$, layers collapse; depth folds space into $2^m$ pieces with $2m$ units. |
| **§4.2 Implementation** | What do hidden units actually learn on images? | Pre-training sanity checks ($\text{loss} \approx \log 10$, $50\%$ active ReLUs); localized garment-part filters. |
| **§4.3 Backpropagation** | How does one pass compute $203{,}530$ derivatives? | Three local rules ($\mathbf{X}^\top\mathbf{G}$, $\mathbf{G}\mathbf{W}^\top$, $\odot\,\phi'$) at $\sim 2\times$ forward FLOPs and $O(nKh)$ memory. |
| **§4.4 Numerical Stability & Init** | Why do 10-layer gradients vanish or explode? | Per-layer variance factor $h\tau^2\mathbb{E}[\phi'(z)^2] = 1$ yields Xavier ($2/(n_{\textrm{in}}+n_{\textrm{out}})$) and He ($2/n_{\textrm{in}}$). |
| **§4.5 Generalization in Deep Learning** | Why doesn't a memorizing network overfit? | Random-label memorization breaks VC bounds; double descent at $h = n$; minimum-norm implicit bias. |
| **§4.6 Dropout** | How does random masking curb memorization? | Inverted dropout $h' = \xi h / (1 - p)$ keeps $\mathbb{E}[h'] = h$ at test time and slows noisy-label overfitting. |
| **§4.7 Kaggle House Prices** | How does the full pipeline run on messy tables? | Leakage-free standardization/imputation, log-RMSE target centering, and shared 5-fold cross-validation. |
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

