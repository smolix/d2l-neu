# Slide outline — §5.1 `mlp.md` (Multilayer Perceptrons)

**Status: ready to design.** No staleness. One per-framework gap (`mlp-xor`,
PyTorch-only) — handle with `only="pytorch"`.

This is the chapter opener: from "why linear isn't enough" → hidden layers →
the collapse argument → activations → universal approximation. It is a
*concept + activation-gallery* deck, not a training deck. The existing
`<!-- slides -->` block (lines 721–913) is a strong source of ideas but is
**all prose + math + activation plots, with no diagram**; the north-star
rewrite should lead the conceptual slides with the chapter's two figures
(`mdl-mlp-xor.svg`, `mlp.svg`) and build the activation gallery as compact
In/Out cards.

## Notebook cells (all 4 fw unless noted)

| cell id | kind | tabs | shows |
|---|---|---|---|
| `mlp-multilayer-perceptrons` | text (no output) | all 4 | imports |
| `mlp-xor` | text | **pytorch only** | hand-built ReLU net computes XOR; 4×3 truth-table tensor |
| `mlp-relu-function-1` | svg plot | all 4 | ReLU curve |
| `mlp-relu-function-2` | svg plot | all 4 | ReLU derivative (step) |
| `mlp-sigmoid-function-1` | svg plot | all 4 | sigmoid curve |
| `mlp-sigmoid-function-2` | svg plot | all 4 | sigmoid derivative (peak 0.25) |
| `mlp-tanh-function-1` | svg plot | all 4 | tanh curve |
| `mlp-tanh-function-2` | svg plot | all 4 | tanh derivative |

## Figures available

- `img/mdl-mlp-xor.svg` — **two-panel** matplotlib figure: (a) input space, 4
  corners coloured by XOR label, the two classes on opposite diagonals (no line
  separates); (b) hidden space after `ReLU(xW+b)`, the two class-1 corners
  folded onto $(1,0)$, now linearly separable by $h_1-2h_2$. This *is* the XOR
  payoff. Reuse as a Markdown image.
- `img/mlp.svg` — the classic MLP-with-one-hidden-layer schematic (4 inputs, 5
  hidden, 3 outputs). Legacy book figure; reuse for the architecture slide.
- No diagram-engine (`@fig:`) module exists for this chapter yet. Decision for
  the author: reuse the committed `mdl-`/legacy SVGs as plain `![](...)` images
  (fast, already house-style), **or** author a `diagrams/mlp.mjs` engine module
  (`mlp-arch`, `mlp-collapse`, `mlp-xor-fold`) for exact-font inline `@fig:`.
  **Recommendation: reuse the existing SVGs** — they are already polished and
  are the book figures; an engine module is optional polish, not required.

## Proposed slide list

1. **Cover** — `.cover`. Kicker "Dive into Deep Learning · §5.1". Title:
   *Multilayer Perceptrons* / "stacking linear layers with a nonlinearity — the
   first deep network."

2. **Why linear models aren't enough** (`title`, `.cols .vc`) — kicker
   *Motivation*. Left col: the three failure cases as a tight list (temperature
   → health risk is non-monotonic; cat-vs-dog pixel meaning is contextual; XOR
   is provably unsolvable). Right col `.fig`: `![](../img/mdl-mlp-xor.svg)` panel
   — show the *input-space* half as the hook ("no line separates these"). One
   `.d2l-note`: linearity ⇒ monotonicity, a strong assumption.
   *(Replaces existing slide 1; adds the picture it was missing.)*

3. **Divider 01 — From Linear to Nonlinear.**

4. **Architecture: the hidden layer** (`title`, `.cols .vc`) — left `.col .fig
   .big`: `![](../img/mlp.svg)` (4→5→3). Right col: the one-hidden-layer math
   $$\mathbf{H}=\mathbf{X}\mathbf{W}^{(1)}+\mathbf{b}^{(1)},\quad
     \mathbf{O}=\mathbf{H}\mathbf{W}^{(2)}+\mathbf{b}^{(2)}.$$
   Caption: two layers, two weight matrices, two biases. "Looks like progress."

5. **Stacking alone buys nothing** (`title`) — the collapse. Show the algebra
   $\mathbf{O}=\mathbf{X}\mathbf{W}^{(1)}\mathbf{W}^{(2)}+\dots=\mathbf{X}\mathbf{W}+\mathbf{b}$
   with the `\underbrace`. `. . .` fragment → the punchline `.d2l-note .warn`:
   "A composition of affine maps is affine. The hidden layer adds **zero**
   expressive power." This is the load-bearing idea of the section.

6. **The missing ingredient: a nonlinearity** (`title`) — insert $\sigma$:
   $$\mathbf{H}=\sigma(\mathbf{X}\mathbf{W}^{(1)}+\mathbf{b}^{(1)}),\quad
     \mathbf{O}=\mathbf{H}\mathbf{W}^{(2)}+\mathbf{b}^{(2)}.$$
   One line of caption: now the composition can curve/fold the decision
   surface. (No code — this is the pivot slide.)

7. **A concrete win: XOR** (`title`, `.cols .vc`) — kicker *From Linear to
   Nonlinear*. Left `.col .fig .big`: full two-panel `![](../img/mdl-mlp-xor.svg)`
   (input → hidden fold). Right col: one-line statement of the hand-built
   weights $\mathbf{W}^{(1)}=\left(\begin{smallmatrix}1&1\\1&1\end{smallmatrix}\right)$,
   $\mathbf{b}^{(1)}=(0,-1)$, output $h_1-2h_2$. Caption: the hidden layer
   *re-represents* the inputs so a line works.

8. **XOR, verified** (`title`, `only="pytorch"`) — the hand-built net actually
   computes XOR on all four inputs:
   `@mlp-xor` → the 4×3 truth table `[[0,0,0],[0,1,1],[1,0,1],[1,1,0]]`.
   Caption: third column = XOR of first two. "We *constructed* these weights;
   the rest of the book is about *learning* them."
   **Gap:** no jax/tf/mxnet sibling. Either scope `only="pytorch"` (deck for
   other frameworks skips the verification, keeps the figure on slide 7), **or**
   port `mlp-xor` to the other three before their decks (see omissions).

9. **Universal approximation** (`title`) — one slide, prose-light. The theorem
   (Cybenko 1989; any sane activation): one hidden layer, enough units →
   approximates any continuous function. `. . .` fragment → three caveats as a
   tight list (existence ≠ findability; may not generalize; width can be
   *exponential*). `.d2l-note`: "Depth trades width for parameter efficiency —
   the modern reason for deep nets." No code.

10. **Divider 02 — Activation Functions.**

11. **ReLU — the modern default** (`title`, `.cols .vc`) — left col: math
    $\operatorname{ReLU}(x)=\max(0,x)$ + `@!mlp-relu-function-1` (output-only
    plot — the curve *is* the point, drop the 2-line code). Right col `.narrow`
    `.d2l-note`: three reasons (no right-saturation → gradient 1; cheap; sparse
    activations). *Curate:* use `@!` not `@` so the plot leads, not the matplotlib.

12. **ReLU's derivative & the dead unit** (`title`, `.cols .vc`) — left:
    `@!mlp-relu-function-2` (step function); $\operatorname{ReLU}'(x)=\mathbb 1[x>0]$.
    Right `.narrow`: the *dead ReLU* — a unit pushed negative for every example
    gets zero gradient forever; fix is LeakyReLU/PReLU
    $\max(0,x)+\alpha\min(0,x)$. (Merges existing "ReLU's derivative" + "Dead
    ReLU" slides — one idea: the flat-left consequence.)

13. **Sigmoid — squashes to (0,1)** (`title`, `.cols .vc`) — left:
    $\sigma(x)=1/(1+e^{-x})$ + `@!mlp-sigmoid-function-1`. Right `.narrow`: today
    used for *output layers* (binary prob) and *gates* (LSTM/GRU/attention), not
    hidden layers.

14. **Why sigmoid hurts deep nets** (`title`, `.cols .vc`) — left:
    `@!mlp-sigmoid-function-2`; $\sigma'(x)=\sigma(x)(1-\sigma(x))$, max 0.25.
    Right `.narrow` `.d2l-note .warn`: in a 10-layer stack the backward pass
    multiplies $\le0.25$ per layer → $\approx 4^{-10}\approx10^{-6}$. The
    vanishing-gradient problem ReLU solved. (Forward-points to §5.4.)

15. **Tanh — the zero-centered cousin** (`title`, `.cols .vc`) — left:
    $\tanh(x)=2\sigma(2x)-1$ + `@!mlp-tanh-function-1`; right `.narrow`: range
    $(-1,1)$, zero-centered (mildly helps optimization), default in RNN cells.
    Optionally fold the tanh-derivative `@!mlp-tanh-function-2` as a `. . .`
    fragment ("still saturates at both tails — same issue as sigmoid").

16. **Activation cheat-sheet** (`title`) — the existing markdown table
    (ReLU/LeakyReLU/GELU/Sigmoid/Tanh/Softmax: range, saturates?, use case).
    Keep — it is a genuine one-glance summary. Add a one-line default:
    "ReLU hidden, GELU for Transformers, sigmoid/softmax at outputs."

17. **Recap** (`title`) — MLP = affine layers + elementwise nonlinearity; the
    nonlinearity is essential (collapse otherwise); one wide hidden layer is a
    universal approximator, depth makes it efficient; ReLU is the default;
    sigmoid/tanh persist in output/gate/RNN roles. Pointer: "the rest of this
    chapter trains MLPs — forward/backprop, init, regularization."

## Per-framework notes

- **`mlp-xor` is PyTorch-only** (`#@tab pytorch`, the only tagged variant). The
  verification slide (#8) must be `only="pytorch"`, **or** the cell ported to
  jax/tf/mxnet first. The XOR *figure* (slides 2, 7) is framework-agnostic, so
  the conceptual XOR story survives in every deck even without the code cell.
- All activation plots (`relu/sigmoid/tanh -1/-2`) exist for all four
  frameworks and are pure plot swaps — **no `only=`/`except=` needed**; captions
  stay framework-neutral. Using `@!` (output-only) sidesteps the small per-fw
  code differences (`torch.relu` vs `jax.nn.relu` vs `tf.nn.relu` vs `npx.relu`)
  cleanly, keeping the gallery uniform across decks.
- No JAX-immutability / TF-Variable framing issues here.

## Must port before non-pytorch decks
- **`mlp-xor`** → add `#@tab jax`, `#@tab tensorflow`, `#@tab mxnet` variants
  (trivial: `@`/`maximum`/`relu` equivalents on a 4×2 input), or accept
  `only="pytorch"` for slide 8.
