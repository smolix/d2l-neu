# Multilayer Perceptrons

Source: `chapter_multilayer-perceptrons/mlp.md`. About 3,600 words of prose, 970 words of exercises, two code cells, four generated figures plus `mlp.svg`.

## Verdict

The best section of the chapter and the one closest to the bar. It has a single line of argument that a reader can retell: a weighted sum cannot make one input's effect depend on another; four points (XOR) already show it; two ReLU units fix it by folding the input; the same fold, written as a tent and composed with itself, shows what depth adds. Every abstraction has a small case worked to numbers, and the figures carry explanation. What keeps it from a 5 is that the argument never touches real data (the reader is told a hidden layer "learns features" and never sees one), a few paragraphs carry catalogue material that the argument does not need (GELU variants), and two key steps are compressed into single sentences. The single most important fix: connect the toy to Fashion-MNIST, here or in the next section, by showing what one or two trained hidden units respond to.

## Scores

| Clarity of explanation | Clarity of thought | Pedagogy | Language | Ease and enjoyment |
|---|---|---|---|---|
| 4 | 5 | 4 | 4 | 4 |

## What works

- **XOR, carried to the end.** Truth table, a three-line proof by adding inequalities, the midpoint picture ("it would have to be positive and not positive at once"), a hand-built two-unit network, a code cell that prints `x`, `h`, `o` for the four inputs, and a figure with input space and hidden space side by side. This is the Strang method: smallest case, solved completely, with numbers.
- **The fold.** "The second unit *folds* the line of values of $s$ at $s = 1$, so that its two ends reach the same output." This one image is then reused for depth: "It is the XOR network read along the diagonal". The reader meets the tent map as something already understood. Preserve this link in any revision.
- **Width adds, depth multiplies.** "For $m = 10$, this means 20 units in ten hidden layers against at least 1,023 in one." A number the reader will remember, followed at once by the honest limit: "Depth does not help with every function, and nothing guarantees that training finds such compact solutions."
- **Universal approximation by hinges.** The one-input argument (a ReLU unit is a hinge, a sum of hinges is a polyline, a polyline follows any continuous function) is a real proof sketch that a first-time reader can follow, and the figure shows the error shrinking.
- **"Limits of the Theorem".** Three open questions (width, training, generalization) that are the agenda of the rest of the chapter. This paragraph does more to hold the chapter together than any roadmap.
- **The role of the nonlinearity.** The collapse to $\mathbf{X}\mathbf{W} + \mathbf{b}$ is shown by substitution in one display, with the remark that the hidden layer "adds parameters but no new functions".
- The derivative table printed by autograd (`0.0177`, `0.1966`, `0.2500` ...) matches the prose: saturation at $\pm 4$ gives 0.018 and 0.0013, as stated.

## Structural findings

- **No learned feature is ever shown.** The section's claim is that the hidden layer learns a representation. The evidence is a hand-built network on four points. Neither this section nor the next shows what a trained hidden unit computes on an image (for instance, the 28×28 weight image of a few units, or the two classes one unit separates). A reader therefore understands why hidden layers *can* help and has to take on trust that training finds such features. One figure would close the gap. It belongs at the end of `mlp-implementation.md` (see IMPL-1), with a forward pointer from the sentence "training chooses them instead".
- **The activation-function section interrupts the argument.** The line of thought runs limits → hidden layer → XOR → deeper networks → expressive power. Between "Deeper Networks" and "Expressive Power" sit about 900 words on ReLU, GELU, SiLU, sigmoid and tanh. The material on sigmoid saturation is needed (it sets up vanishing gradients), but the reader loses the thread just before the universal-approximation question. Consider moving "Activation Functions" after "Expressive Power", so the section ends on the question of training (which activation lets gradients through), which is where the chapter goes next.
- **Monotonicity is introduced and not used.** The first subsection names two limits, monotonicity and absence of interaction. The body temperature example is good, and the text says it can be repaired by hand. Everything after is about interaction. That is fine, but the heading "Monotonicity and Interactions" promises two topics of equal weight. One sentence stating that interaction is the limit that matters for the rest of the section would make the scope decision explicit.
- Length is right for the content. Nothing structural should be cut except the items in MLP-6.

## Passage-level findings

- **MLP-1** (moderate, pedagogy)
  - Anchor (section opening): "an image, whether a bright pixel belongs to a sleeve, a strap, or the"
  - The opening paragraph repeats the chapter introduction almost word for word (same 83%, same sleeve, strap and background, same closing question), and the first subsection says it a third time: "Whether a bright pixel is evidence for a shirt or for a bag depends". Three statements of one idea in the first 700 words.
  - Fix: cut the first paragraph to two sentences that point back ("Softmax regression adds up evidence pixel by pixel. We now make precise what that rules out, on the smallest example where it fails.") and start with XOR. Keep the shirt-or-bag sentence in "Monotonicity and Interactions", where it does work.

- **MLP-2** (minor, explanation)
  - Anchor (XOR Problem): "The proof has a picture. Both pairs of points have the midpoint"
  - There is no picture at this point. The figure with the four points comes two subsections later and does not draw the two diagonals or their common midpoint.
  - Fix: either add the two diagonals and the midpoint to the left panel of `mdl-mlp-xor.svg` and reference it here, or write "The proof can be seen geometrically."

- **MLP-3** (minor, language and correctness)
  - Anchor (The Model): "an early linear classifier, of which an MLP stacks several layers."
  - The sentence does not parse cleanly ("of which an MLP stacks several layers"), and it is loose: an MLP does not stack perceptrons, whose units are thresholds, but layers of units that each compute a weighted sum followed by a nonlinearity.
  - Fix: "The name comes from the perceptron, an early classifier that thresholds a weighted sum. Each hidden unit resembles one, with a smoother nonlinearity, and an MLP arranges such units in layers."

- **MLP-4** (minor, explanation)
  - Anchor (The Model): "its layer. With a superscript numbering the layers, the hidden layer has"
  - The layout sentence carries a lot: rows of $\mathbf{X}$ are examples, weight matrices have one row per input, superscripts number layers, and four shapes, all before the display. Then the single-example form uses $\mathbf{W}^{(1)\top}\mathbf{x}$. A first-time reader will wonder why the transpose appears.
  - Fix: after the single-example formula, add one clause: "the transpose appears because $\mathbf{x}$ is a column while each example is a row of $\mathbf{X}$."

- **MLP-5** (moderate, correctness of wording)
  - Anchor (Activation Functions): "Every weight into unit $j$ is scaled by the same factor $\phi'(z_j)$."
  - It is the gradient of every weight, not the weight, that carries the factor. As written the sentence says something false, and the next sentence ("none of these weights moves") depends on reading it correctly.
  - Fix: "The gradient of every weight into unit $j$ contains the same factor $\phi'(z_j)$."

- **MLP-6** (moderate, pedagogy: density)
  - Anchor (ReLU and GELU): "libraries compute it by default with an approximation built from the tanh,"
  - Anchor (ReLU and GELU): "subsection, is used in a gated form, SwiGLU, by many recent language models"
  - The GELU paragraph introduces, in eight sentences, the GELU, the normal CDF, its minimum of $-0.17$, a library approximation and its error bound of $5 \cdot 10^{-4}$, the SiLU, a "gated form" called SwiGLU, and a forward reference to the transformer block. None of the last four is used again in the chapter. This is the catalogue style that the writing guide §4.3 and §6.1 rule out, and the only place in the section where the text stops explaining and starts listing.
  - Fix: keep the definition, the one-sentence description (matches ReLU far from zero, rounds the corner) and "many transformer models use it". Cut the tanh approximation (it can live as the existing code comment in the JAX tab) and the SiLU and SwiGLU sentence.

- **MLP-7** (minor, thought)
  - Anchor (Sigmoid and Tanh): "weights of 1, the gradient therefore shrinks by a factor of at least"
  - Good, concrete, and correct. But the same chain is then the subject of exercise 4 and of the next-but-one section's opening. Here it would be enough to state the factor and the pointer. Not a defect, a note for whoever balances repetition across the chapter.

- **MLP-8** (minor, correctness: text and figure disagree)
  - Anchor (Universal Approximation): "line of slope $a$ to the right of it. A sum of $h$ hinges and a constant is"
  - The text counts a constant plus $h$ hinges, giving $h + 1$ pieces with a flat first piece. The figure caption says "a line plus the three hinges is a polyline with 4 pieces", and its first piece slopes. A sloped first piece needs either a line term, which a hidden layer does not have, or a fourth hinge at the left end, which the "Conversely" sentence indeed uses. A careful reader counting units gets 3 in the caption and 4 in the text.
  - Fix: make the caption say "a constant, a hinge at the left end, and the three hinges shown" or redraw with a flat first piece.

- **MLP-9** (moderate, pedagogy)
  - Anchor (Universal Approximation): "every continuous function. This result is the *universal approximation"
  - The paragraph proves the one-input case, states the general theorem, cites two papers, explains why polynomials are excluded, and only then says that what was just stated is the universal approximation theorem. The theorem, the most quotable result of the section, is named in passing in the ninth sentence of a paragraph.
  - Fix: split in two. End the first paragraph with the one-input conclusion. Open the second with the name and the statement ("The *universal approximation theorem* says that the same holds for any number of inputs ..."), then the polynomial remark, then the exercise pointer.

- **MLP-10** (moderate, explanation)
  - Anchor (Depth): "layer of two units, since the output map of one layer and the affine map of"
  - This clause justifies the claim that $m$ compositions need only $m$ hidden layers of two units, which is the whole point of the comparison with $2^m - 1$. It is too compressed: the reader has to see that $g$ ends in an affine combination of its two ReLUs, and that the next copy of $g$ starts with an affine map, and that two affine maps in a row are one.
  - Fix: "Each copy of $g$ is a hidden layer of two ReLU units followed by a weighted sum. The next copy begins with a weighted sum of its own, and two weighted sums in a row are one, as we saw for networks without activation. So each further composition adds one hidden layer of two units and nothing else."

- **MLP-11** (minor, language)
  - A tic to watch: "whatever" as a universal quantifier appears six times in this section ("whatever the other inputs are" twice in the first 90 lines, "whatever the loss", "whatever threshold") and thirteen times in the chapter. Vary with "for every", "regardless of", or restructure.

## Exercises

Seven problems, well graded, and several are excellent: "XOR by hand" part (b) (one ReLU unit cannot do it) tests exactly the understanding the section builds; "Depth and linear pieces" extends the tent to three units; "Weight scale and saturation" anticipates the next sections with a one-dimensional iteration.

- "One hidden layer suffices" is a hard, good problem, but its preamble describes a two-hidden-layer construction ("the first builds steps along the coordinate axes from pairs of ReLU units, and the second combines them into localized bumps") that the section never presented. A reader cannot picture it from one sentence. Either drop the preamble and state the task directly, or add the bump construction as part (a).
- "Linear networks" is cited from the body as the proof that a hidden layer can remove functions. Good use of an exercise.
- No exercise asks the reader to train anything, which is right for this section.

## Still open from October

The October items for this file are closed: the six plotting cells are gone (replaced by one generated figure and one cell that computes derivatives), and the citation catalogues are reduced to the GELU paragraph (MLP-6), which is the remaining instance.
