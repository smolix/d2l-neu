# Numerical Stability and Initialization

Source: `chapter_multilayer-perceptrons/numerical-stability-and-init.md`. About 3,200 words of prose, 950 words of exercises, four code cells, two stored plots, no generated figure.

## Verdict

The section has a clear idea (the gradient reaching layer $k$ is a product of per-layer factors, and the initial weight scale decides whether the factors are near 1) and a fine central device: one function that measures mean squares layer by layer, used once to show the problem and once to show the cure, with predictions ($0.0128^9 \approx 10^{-17}$) that the measurement confirms. The He derivation is short and has a real insight in it. Two things hold it back. First, the section opens with a claim about training ("does not learn") and never trains anything: every measurement is of a random network on random inputs at initialization, so the reader never sees the cure work on a task. Second, the key formula, the per-layer factor of the gradient, is asserted by a compressed sketch with a forward pointer, and the place it points to derives the forward case and says "the same computation shows" for the backward one. The most important fix is a small training experiment that pays off the opening.

## Scores

| Clarity of explanation | Clarity of thought | Pedagogy | Language | Ease and enjoyment |
|---|---|---|---|---|
| 3 | 4 | 3 | 4 | 3 |

## What works

- **Prediction, then measurement.** "On a logarithmic axis the curves are straight lines: every layer multiplies the mean square by the same factor." The stored output gives `9.1e-18` and `7.6e+18` at layer 1 against predicted $10^{-17}$ and $10^{19}$. The reader sees a formula predict a number to within rounding. More of the book's experiments should have this shape.
- **The He derivation.** "the inputs enter only through their mean square" is the observation that makes the ReLU case a two-line extension of the linear case, and the text names it as the thing the earlier argument "used". This is the Strang move: keep the form of the simple solution.
- **The sigmoid curve with $h\tau^2 = 1$** isolates the activation's contribution from the weight scale, and the text explains why larger weights do not rescue it ("they push more pre-activations into the tails").
- **Symmetry among Hidden Units.** A complete small argument (identical units receive identical gradients, so they stay identical), a check that prints `0.00` against `0.44`, and a clean separation from the variance argument: "The variance argument fixes the scale of the initial weights. It does not say why they should be random".
- **The limit is stated.** "This holds at initialization, but training changes the weights and breaks the assumption, and nothing then keeps the factor of each layer near 1." One sentence that tells the reader what the section did not solve and where the book goes next.
- Symbols avoid the usual clash: $\tau^2$ for the weight variance and $\gamma^2$ for the input variance leave $\sigma$ to the sigmoid.

## Structural findings

- **No training run.** See NS-1. The chapter has now asserted three times that small weights stop a deep network from training (closing of `mlp-implementation.md`, its exercise 5, and the opening here) and has not shown it once. All four code cells measure statistics of random matrices.
- **Order: formula, measurement, derivation.** The per-layer factor is stated (sketch), then measured, then, two subsections later, derived for the forward pass under the name Xavier. The reader meets the central computation twice, the first time too fast to follow and the second time for a different quantity. A cleaner order: (1) the measurement with small and large weights, read off as "each layer multiplies by a constant"; (2) the question "which constant?"; (3) the variance computation, once, forward and backward; (4) Xavier and He as its two solutions; (5) the measurement again. This also removes the forward reference "(:numref:`subsec_xavier` derives this)".
- **"Default Initialization" sits before the derivation that makes it readable.** It tells the reader that the framework default "is not designed for a deep stack of ReLU layers" before the reader knows what a design for ReLU layers is, and two of the four tab notes name things defined later ("the Xavier initialization derived below", "LeCun normal"). After the He subsection the same facts become a useful comparison: PyTorch's default variance $1/(3n_\textrm{in})$ is one sixth of He's $2/n_\textrm{in}$.
- **Notation changes from the previous section.** See NS-3.
- **The title.** "Numerical Stability" suggests rounding and overflow, which get two sentences. The section is about the scale of signals and gradients. "Initialization" alone, or "Gradient Scale and Initialization", would name the subject.

## Passage-level findings

- **NS-1** (major, pedagogy: missing demonstration)
  - Anchor (section opening): "A network with ten hidden layers of width 256, initialized the same way,"
  - The sentence ends "does not learn". The section never shows this, and never shows that Xavier or He initialization makes the same network learn. A reader finishes with three formulas and two plots of mean squares of random matrices, and has to believe that these translate into training. The exercise in `mlp-implementation.md` hands the experiment to the reader.
  - Fix: add one experiment at the end of "Activations and Gradients across Depth": the ten-hidden-layer MLP on Fashion-MNIST for a few epochs with the three initializations already used, one plot of training loss. Expected: flat at $\log 10$ for $\mathcal{N}(0, 0.01^2)$, slow or flat for Xavier, falling for He. Two sentences read it. This is the payoff of the section and of the thread started in `mlp-implementation.md`.

- **NS-2** (major, explanation: the central step is skipped)
  - Anchor (Products of Jacobians): "the cross terms of the square of this sum vanish, so the sum has $h$ times"
  - Anchor (Xavier Initialization): "over the $n_\textrm{out}$ outputs of the layer instead. The same computation"
  - The factor $h\tau^2\,\mathbb{E}[\phi'(z)^2]$ is the result the whole section rests on. It is justified by one long sentence (a sum of $h$ terms, cross terms vanish, one term has $\tau^2 \mathbb{E}[\phi'(z)^2]$ times the mean square above) that a first-time reader cannot follow at reading speed, with a pointer forward. The Xavier subsection then derives the *forward* variance carefully and dismisses the backward one with "The same computation shows". So the backward factor is never derived. The argument also uses an assumption that is not stated: that the slopes $\phi'(z)$ and the gradient from above can be treated as independent of the weights they are multiplied by. That is an approximation (the slopes depend on those same weights through the forward pass), and it is the reason the text says "about".
  - Fix: after reordering (see above), do the forward computation as now, then write the backward one out in three lines for one entry of the gradient, $g_i = \sum_j w_{ij}\,\phi'(z_j)\,g'_j$, state the independence assumption in one sentence and say that it is an approximation which the measurement then tests.

- **NS-3** (moderate, explanation: notation)
  - Anchor (Products of Jacobians): "function and then by the transposed weights."
  - The sentence says the backward step multiplies "by the transposed weights". The display that follows, written for a single example as a column vector, has $\mathbf{W}^{(k)}$ untransposed, introduces $\nabla_{\mathbf{h}} L$ in place of the previous section's $\partial J / \partial \mathbf{H}$, a new matrix $\mathbf{M}^{(k)}$ with $\operatorname{diag}$, and then the sentence "The transpose of $\mathbf{M}^{(k)}$ is the Jacobian of layer $k$". The reader has to hold three conventions (batch rows against columns, which side the weights multiply on, what is transposed) to connect this to the rule learned twenty minutes earlier. Meanwhile the code computes `G = (G * dphi(Z)) @ W.T`, which is the previous section's rule verbatim.
  - Fix: stay in the batch form of `sec_backprop`: $\mathbf{G}^{(k-1)} = \bigl(\mathbf{G}^{(k)} \odot \phi'(\mathbf{Z}^{(k)})\bigr)\mathbf{W}^{(k)\top}$, which is also the line of code. Say in words that this is multiplication by the layer's transposed Jacobian, and unroll it in words ("one such step per layer"). Keep $\mathbf{M}^{(k)}$ only if exercise 1 needs it, and define it there.

- **NS-4** (moderate, pedagogy: two scales for one quantity)
  - Anchor (section opening): "shrinks the magnitude of the gradient about ninefold. By the time it reaches"
  - The opening speaks of magnitude: ninefold per layer, "more than $10^8$" in all. The body speaks of mean squares: $0.0128$ per layer, $10^{-17}$ in all. The exploding case does the conversion once ("about $3 \times 10^9$ times larger in magnitude"). A reader comparing the opening with the plot finds $10^8$ against $10^{17}$ and has to work out that one is the square of the other.
  - Fix: use mean squares throughout and say once that the typical size of an entry is the square root. In the opening: "each multiplication shrinks the mean square of the gradient about eightyfold".

- **NS-5** (moderate, pedagogy: order)
  - Anchor (Default Initialization): "is not designed for a deep stack of ReLU layers, so this book sets the"
  - See the structural finding. As placed, the subsection is a claim the reader cannot check plus four tab notes of library facts. I also could not verify "this book sets the initialization of its models explicitly" beyond this chapter; `mlp-implementation.md` does, with a scale this section then shows to be poor.
  - Fix: move after "He Initialization" and turn each tab note into a comparison with $2/n_\textrm{in}$.

- **NS-6** (minor, language)
  - Anchor (Products of Jacobians): "As for the repeated matrix products of :numref:`sec_linear-algebra`, with"
  - "As for the repeated matrix products ..., with $\rho = 0.9$ the gradient shrinks 200-fold" does not parse on first reading: "as for" reads as "regarding". The numbers are good (I checked: $0.9^{50} \approx 1/194$, $1.1^{50} \approx 117$).
  - Fix: "We saw the same effect for repeated matrix products in :numref:`sec_linear-algebra`. With $\rho = 0.9$ ..."

- **NS-7** (minor, explanation)
  - Anchor (before the first code cell): "activations of the last hidden layer with random weights $v_{ij}$ of mean"
  - Three sentences construct a loss $L = \sum v_{ij}[\mathbf{H}^{(K-1)}]_{ij}$ whose only purpose is that the backward pass starts from a random matrix of mean square 1.
  - Fix: say that directly: "We start the backward pass at the last hidden layer from a random gradient of mean square 1, so that the plot shows only what the layers below do to it."

- **NS-8** (moderate, material written for the authors)
  - Anchor (He Initialization): "logits have twice the mean square of the activations below them. In the"
  - Five sentences reconcile this section's count of where the unmatched factor of 2 falls with the count in the He et al. paper ("In the count above it is the output layer ... In the count of pre-activations of ... it is the first layer"). A first-time reader does not know there were two counts and does not need to. This is a paper audit (writing guide §4.5).
  - Fix: keep one sentence: "One weight layer has no ReLU after it, so the scale is off there by a single factor of 2, which does not compound with depth."

- **NS-9** (minor, pedagogy: forward reference)
  - Anchor (Symmetry among Hidden Units): "already differ between units. Dropout, introduced later in this chapter,"
  - An aside about a method two sections ahead, attached to the end of the result. It is true and it distracts.
  - Fix: delete, or move to `dropout.md` as one sentence there.

- **NS-10** (minor, correctness of emphasis)
  - Anchor (Activations and Gradients across Depth): "last layer are about $10^{-19}$, and the network is useless before training"
  - This is the stronger and simpler statement of why the small-weight network fails: its output does not depend on its input to within float precision. It arrives as the last clause of the last measurement. The opening and the first half of the section explain the failure through the gradient alone.
  - Fix: say early that both passes scale together for ReLU layers ("the same factor shrinks the signal on the way up and the gradient on the way down"), so the reader holds one picture.

## Exercises

Seven problems, mathematically the richest set in the chapter and all tied to the section. "Variance under ReLU" (why the mean square and not the variance) and "Tanh networks" ($m' \approx m - 2m^2$, hence decay like $1/(2k)$; I checked the expansion) are real extensions that a strong student will enjoy. "Fan-in versus averaged fans" has a surprising answer. "Other symmetries" extends the symmetry argument.

- The set is heavy on analysis: five of seven are pencil-and-paper with spectral norms, variances or expansions. For the intended reader that is steep. If NS-1 is not adopted in the text, add a [code] exercise that trains the deep network under the three initializations; if it is adopted, the reader has the experiment.
- "Layerwise adaptive scaling" introduces a named method (LARS) with a citation that the chapter never uses again. It is a good experiment, but it reads as a paper pointer. Consider stating it without the name.

## Still open from October

October's "keep" (the derivation of He initialization) is preserved and is still the best passage. Nothing else was raised for this file.
