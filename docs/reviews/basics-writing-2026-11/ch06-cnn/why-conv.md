# From Fully Connected Layers to Convolutions

Source: `chapter_convolutional-neural-networks/why-conv.md`, about 2,650 words of prose, 6 exercises (about 800 words).

## Verdict

The argument is now correct, complete, and economical: a fully connected layer on a grid, reindexed by offsets, is translation equivariant if and only if its weights depend on the offset alone; locality then truncates the offsets; what is left is a convolution. The proof is short and its key step (one bright pixel reveals the weights) is a good one. The prose is plain and almost free of mannerisms. Against Cover and Thomas the section holds up as *text*. Against "The Illustrated Transformer" it does not, because the derivation is carried by four-index algebra in two dimensions while the one figure that could carry it (the $8\times8$ weight matrix in one dimension) is introduced in a single sentence and never used again. The single most important fix: run the whole derivation first on the eight-sample, one-dimensional signal of `fig_why_conv_weight_sharing`, where a weight has two indices and the proof is a picture (each column of the matrix is the response to one bright sample, and equivariance says each column is the previous one moved down by one), and only then lift it to images. The second fix is the Waldo photograph, which shows a crowd in which everyone is dressed as Waldo and therefore contradicts the sentence it illustrates.

## Scores

| Clarity of explanation | Clarity of thought | Pedagogy | Language | Ease and enjoyment |
|---|---|---|---|---|
| 4 | 4 | 3 | 4 | 3 |

## What works

- The parameter ladder $10^{12} \to 4\times10^6 \to$ a few hundred, announced in the opening and paid off step by step. Each number is derived where it appears ("$1999^2 \approx 4 \times 10^6$", "9 weights for $\Delta = 1$ and 121 for $\Delta = 5$"). This is the Strang habit of finishing the example with numbers.
- The reindexing is explained instead of asserted: "The second line only renames the weights. It measures the position of a pixel relative to the hidden unit". The October complaint that the key step was asserted is closed.
- The sentence that reads the proof: "In words, a single bright pixel reveals the weights." Then the sentence that reads the result: "Every hidden unit computes the same weighted sum of the pixels around it, so the layer compares one pattern, $\mathbf{K}$, with the image at every location." Both are Cover-and-Thomas quality.
- The paragraph beginning "The reduction from $10^{12}$ to a few hundred parameters comes from restricting the class of functions, not from a more economical way to represent the same functions." It states the cost of the assumption, with the failure case. Preserve it.
- Equivariance and invariance are kept apart from the first page, with the Waldo score map (equivariant) against the question "is Waldo in the scene?" (invariant). Most textbooks blur the two.
- The exercises on audio, spectrograms, text, and the nine-cell grid test whether the reader understands *when* the two assumptions hold. The rotation exercise is the best in the chapter.

## Structural findings

- **The derivation is done in the hardest case first.** The guide's own sequence (smallest case, worked completely, then the general form) is not followed. The reader meets a fourth-order tensor $\mathsf{W}$, its reindexed twin $\mathsf{V}$, and an if-and-only-if proof with six indices before seeing any instance with numbers. The one-dimensional figure sits between the setup and the proof as an aside. See WHY-5, WHY-6, WHY-7.
- **Three requirements are listed, two are used.** The numbered list after the Waldo photo has three items; the section then speaks of "the two observations" and "the two constraints". The third (deeper layers combine local responses) is a consequence that the section returns to only in one sentence at the end of "Locality". See WHY-3.
- **The infinite-grid paragraph arrives one section early.** It is needed by the proposition, not by the definitions. Placed where it is, it interrupts the Waldo intuition with a technicality. See WHY-4.
- **"Channels" is taught twice in the chapter.** The subsection here (about 420 words, with the four-index kernel and a parameter count) and `channels.md` cover the same definition in two index orders. See WHY-12.
- **The term "inductive bias" is never introduced.** The paragraph on restricting the function class describes it exactly. The reader will meet the term in every paper on the subject (and in `generalization-deep.md`); one clause here would connect them.
- **The summary introduces a new claim.** See WHY-14.

## Passage-level findings

**WHY-1** (minor, pedagogy). Opening:

> in no particular order. If we permuted the pixels of every image in the same
> way, it would learn just as well.

Second of three statements of this fact (see IDX-1). The chapter introduction made the point one page earlier. Fix: one clause ("As the chapter introduction noted, an MLP ignores which pixels are neighbors.") and move straight to the cost.

**WHY-2** (moderate, explanation). Opening, third paragraph:

> We start from a layer that maps
> a $1000 \times 1000$ image to a hidden representation of the same size,
> which has $10^{12}$ weights.

The paragraph before counted $10^9$ weights for "a hidden layer of only 1000 units". The reader now gets a different layer with a different count and no reason why the hidden representation should have the size of the image; every hidden layer they have built so far was smaller than its input. The reason exists and is the heart of the section: we want one hidden unit *per location*, a map of scores like the Waldo detector's. Fix: say so. "An MLP's hidden units have no location. To ask that a shifted image give a shifted response, each hidden unit must sit somewhere, so we give the hidden layer one unit per pixel. Such a layer has $10^6 \times 10^6 = 10^{12}$ weights." This also removes the need for the $10^9$ and the $3\times10^9$ figures, which are not used again.

**WHY-3** (minor, thought). Under "Equivariance and Invariance":

> This suggests three requirements for the layers of a vision network:

Three are listed; from the next section on the text says "the two observations" and "the two constraints". Fix: list two requirements and put the third as a sentence after the list ("Deeper layers then combine these local responses ..."), or keep three and say "the first two constrain a single layer; the third is about stacking, to which we return at the end of the section".

**WHY-4** (minor, pedagogy). Under "Equivariance and Invariance":

> A shift needs a convention at the border of the image, where content leaves
> the frame and nothing enters it.

Correct and careful, but it is a hypothesis of the proposition, and it lands between the Waldo picture and the derivation. Fix: move this paragraph to just before the proposition (it replaces the proposition's first sentence, which restates it).

**WHY-5** (major, pedagogy). Under "Constraints on the Fully Connected Layer":

> :numref:`fig_why_conv_weight_sharing` shows what the two constraints do to
> the weights of a layer on a one-dimensional signal of eight samples, where
> the weights form an ordinary $8 \times 8$ matrix.

This is the only sentence in the section that refers to the figure. The figure is the best explanatory device in the chapter, and the text walks past it. The numbers printed in it (64, 15, 3 weights) appear nowhere in the prose; panel (b), "constant along each diagonal", is never tied to the statement being proved; the reindexing from $\mathsf{W}$ to $\mathsf{V}$ is never shown on it. Fix: make the one-dimensional case the main line.

1. Write the layer for eight samples: $h_i = u_i + \sum_k W_{i,k} x_k$, 64 weights, panel (a).
2. Rename by offset, $V_{i,a} = W_{i,i+a}$: "offset $a$ is the $a$-th diagonal of the matrix".
3. Feed in a signal with a single 1 at sample $k$: the response is column $k$ of $\mathbf{W}$. Shift the 1 to sample $k+1$: the response is column $k+1$. Equivariance says the second response is the first moved down by one. So every column is the previous column moved down by one, which is the statement that $\mathbf{W}$ is constant along each diagonal: panel (b), 15 weights, one per offset from $-7$ to $7$.
4. Locality sets all but the offsets $-1, 0, 1$ to zero: panel (c), 3 weights.
5. "Nothing changes for images except that positions and offsets become pairs": state the two-dimensional layer, the proposition, and the counts $10^{12}$, $4\times10^6$, $(2\Delta+1)^2$.

The formal proof then confirms what the reader has seen, instead of being the only route to it. A small added panel or second figure showing step 3 (two impulse inputs and the two columns they pick out) would complete it. This is what an "Illustrated" treatment would do, and it costs perhaps 250 words, recovered by shortening the two-dimensional algebra.

**WHY-6** (moderate, explanation). Figure `fig_why_conv_weight_sharing` itself (rendered and inspected):

> Equal colors are equal weights; white is zero.

In panel (b) fifteen diagonals share one blue ramp, so several neighboring diagonals differ by a shade that the eye cannot separate, and "equal colors are equal weights" cannot be checked for most of the matrix. Panel (c) works because it has only three colors. Fix (in the generator): use a qualitative or diverging palette with clearly distinct neighbors in (b), or print the offset ($-7 \ldots 7$) in each cell of (b) and the labels $w_{-1}, w_0, w_1$ in (c). Label the axes "input sample $k$" and "hidden unit $i$" to match the text.

**WHY-7** (moderate, explanation). Under "Constraints on the Fully Connected Layer", equation `eq_mlp_fourth_order`:

> $$
> \begin{aligned} [\mathbf{H}]_{i, j} &= [\mathbf{U}]_{i, j} + \sum_k \sum_l [\mathsf{W}]_{i, j, k, l}  [\mathbf{X}]_{k, l}\\ &=  [\mathbf{U}]_{i, j} + \sum_a \sum_b [\mathsf{V}]_{i, j, a, b}  [\mathbf{X}]_{i+a, j+b}.\end{aligned}
> $$

The text before it prepares the first line well. The second line arrives inside the same display, and $\mathsf{V}$, $a$, and $b$ are explained only afterwards. A first-time reader stops at the second line. Fix: display the first line alone and read it; then one sentence of motive ("both constraints concern where a pixel lies *relative to* the hidden unit, so we relabel each weight by that offset"); then the second line. If WHY-5 is adopted, the reader has already seen the relabeling as "diagonals" and this becomes easy.

**WHY-8** (minor, language). Under "Translation Equivariance":

> We first impose translation equivariance :cite:`Zhang.ea.1988`. We show that
> a layer of the form :eqref:`eq_mlp_fourth_order` is equivariant exactly when

The citation hangs on a sentence about what "we" do, so the reader cannot tell what Zhang et al. contributed. The same happens with "Time-delay neural networks ... were among the first models to share weights" at the end of the subsection and "go back to the Neocognitron" at the end of "Locality" (which the chapter introduction has already said). Three history sentences are scattered through the derivation. Fix: gather them into one sentence after the convolutional layer has been named ("Shared local detectors go back to the Neocognitron; time-delay networks for speech and the shift-invariant networks of Zhang et al. shared weights across positions in the same way."), or cut them, since the chapter introduction carries the history.

**WHY-9** (minor, explanation). Proposition:

> scalar $u$ and a matrix $\mathbf{K}$ such that $[\mathbf{U}]_{i,j} = u$ and

The proposition is an "if and only if", and the reader is not told which half matters. Say it in one sentence before the statement: "That shared weights give equivariance is a one-line check. The converse is the point: equivariance *forces* the weights to be shared." (A small precision: the summary says "as we proved for linear layers", but the layer has a bias and is affine; "for layers of this form" is safer.)

**WHY-10** (moderate, explanation). Under "Locality", the paragraph on restricting the function class:

> do not, for instance when the same local pattern calls for a different
> response at different locations, the model may fail even to fit its training
> data.

The claim is right and important, and it is abstract. One concrete case would fix it in memory, and the section already owns one, in exercise 4: a digit in one of nine cells of a grid, where the task is to name the cell. Or: passport photographs, where an eye is always in the upper half and the same dark blob lower down is a nostril. Fix: add one such example in a sentence, and name the idea: "Assumptions of this kind, built into the architecture before any data are seen, are called an *inductive bias*."

**WHY-11** (minor, thought). Under "Convolution and Cross-Correlation":

> $f, g: \mathbb{R}^d \to \mathbb{R}$ :cite:`Rudin.1973` is

The continuous integral and the citation of Rudin serve only to explain the name. The reader needs the discrete two-dimensional form and the flip. Fix: start from "$(f * g)(i) = \sum_a f(a) g(i-a)$" with the sentence "It measures the overlap between $f$ and a copy of $g$ that is flipped and shifted", mention the integral in a clause, and drop the citation. The subsection is then about 120 words. Note also that `conv-layer.md` has a subsection with the same title that repeats the conclusion; see CONV-4.

**WHY-12** (moderate, thought). Under "Channels":

> therefore a third-order tensor $\mathsf{X}$ of shape height × width × 3, for
> instance $1024 \times 1024 \times 3$, with entries $[\mathsf{X}]_{i,j,c}$.

Three small problems in one place. (a) The running image has been $1000\times1000$ for the whole section; $1024$ is a stray. (b) The layout here is channels-last with kernel $[\mathsf{K}]_{a,b,c,d}$; `channels.md` uses channels-first, "$3\times 224\times 224$" and $c_\textrm{o}\times c_\textrm{i}\times k_\textrm{h}\times k_\textrm{w}$, and has to add "(there called $d$)" to reconcile them. (c) The subsection defines input channels, output channels, feature maps, and the parameter count, all of which `channels.md` defines again. Fix: pick one index order for the chapter (channels-first matches the default PyTorch tab and the LeNet figure's "6@28×28"), and cut this subsection to the idea only: a color image has three numbers per pixel, a layer wants many detectors, so both input and output become stacks of maps and the kernel gains two indices; display `eq_conv-layer-channels` once; leave the counting to `sec_channels`.

**WHY-13** (withdrawn). **Withdrawn 2026-11-09 (author):** the Waldo photographs are a deliberate joke (a crowd in which everyone is dressed as Waldo) and stay as they are. The separate suggestion of a small figure of equivariance itself is still open. Original text of the finding follows.

(major, pedagogy). Figure `fig_waldo_mask` and its caption (both Waldo images were inspected):

> ![A Waldo detector: the response of a kernel that matches Waldo peaks at his location (image courtesy of William Murphy (Infomatique)).](../img/waldo-mask.jpg)

The photograph is of a gathering at which every person in the crowd wears Waldo's striped shirt, hat, and glasses. The text says "somewhere among them is Waldo" and that a kernel matching "Waldo's appearance yields a map that peaks where he stands". In this picture a detector for Waldo's appearance would fire on every face. The second image is not a response map either: it is the same photograph washed out, with two rectangles left at full contrast, one of them around a woman. A reader who looks closely (and the first caption invites them to: "Can you find Waldo") finds that the illustration contradicts the idea. The figure is also placed after the paragraph on one- and three-dimensional grids, two paragraphs away from its reference, although its natural home is the sentence in the first section, "its map of scores should move with Waldo". Fix, in order of preference: (1) replace both images by a generated three-panel figure that shows equivariance directly: a small image with one distinctive patch, the response map of a matching kernel with one bright peak, and the same two panels with the patch moved, so the peak moves with it. This is the "commuting square" picture of $f(T_v\mathbf{X}) = T_v f(\mathbf{X})$, and the section has no picture of its central definition. (2) If the photograph stays for charm, keep only the first image, say honestly in the caption that everyone here is dressed as Waldo, and drop the "detector" image.

**WHY-14** (moderate, thought). Summary:

> function class. On an infinite grid, a stack of convolutional layers followed
> by a head that aggregates over all locations, for instance by a maximum,
> cannot represent a classifier whose
> label changes when the same object moves from the left half of the image to
> the right.

A new and fairly subtle statement, made for the first time in the summary, in a sentence of 46 words. The body said only that the model "may fail even to fit its training data". Fix: put the example in the body (WHY-10) and let the summary say "The savings come from restricting the function class: a convolutional layer cannot give the same local pattern different responses at different locations."

**WHY-15** (minor, explanation). Under "Channels", last paragraph before the figure:

> :numref:`sec_sentiment_cnn` classifies text with one-dimensional
> convolutions.

The reference points to a chapter in the Attic (the retired NLP Applications chapter). A forward reference from Basics should not send the reader there. Cut the sentence; the paragraph stands without it.

## Exercises

Strong. Six problems, graded, each tied to the section's idea. Notes:

- Exercise 1.2 (cross-correlation is not commutative) is well posed now that the operation is defined in the problem.
- Exercise 5 (boundary effects) is three counting parts; the third (corner against center after $L$ layers) is the one that teaches, and it anticipates `img_conv_reuse` in the padding section. Parts 1 and 2 could merge.
- Exercise 6 (rotation equivariance) is excellent and should set the model for a marked hard problem: it repeats the section's argument for a different group and ends with a design insight.
- Missing: one problem that makes the reader *do* the one-dimensional derivation by hand (write the $4\times4$ matrix of a layer on four samples with periodic boundary, impose equivariance, count the free weights). If WHY-5 is adopted the text does this and the exercise can ask for the periodic case.

## Still open from October

Closed: "The key step of the derivation is asserted" (now proved) and "the tensor $\mathsf V$ changes order three times" (it is now defined once and used consistently). The legacy prose is gone. Residue: the index order of the channel kernel still differs between this section and `channels.md` (WHY-12).
