# Review — chapter_multilayer-perceptrons/backprop.md  (§5.3 "Forward Propagation, Backward Propagation, and Computational Graphs")

**Role in the chapter:** The math-heavy core that takes the lid off the
`loss.backward()` call: it derives forward and backward propagation by hand on
one concrete example (a one-hidden-layer MLP with $\ell_2$ regularization),
introduces the computational-graph picture, and explains why training costs more
memory than inference. It is the *promised payoff* of the autograd preview in
Preliminaries — `chapter_preliminaries/autograd.md:184` literally says "We unpack
computational graphs and backpropagation in full in :numref:`sec_backprop`" — and
its `prod`-operator notation is reused downstream in
`chapter_recurrent-neural-networks/bptt.md`.

**Verdict:** The derivation is **correct, complete, and clean** — every chain-rule
step is explicit, the regularizer is threaded through both gradients properly, and
the memory point is made well. But it falls short of best-in-class on exactly the
axis the research digest flagged: **it tells the algorithm symbolically and never
once shows numbers flowing.** Worse, the section that is *advertised* as the deep
treatment ships a **dated, monochrome, forward-only figure** (`forward.svg`),
while the *preview* of the same idea in `autograd.md` already has a gorgeous
modern house-style graph with colored boxes and a blue backward pass
(`autograd-comp-graph.svg`). The single highest-value change is to **add one
worked numeric example with a matching color-coded forward/backward figure** — the
CS231n/Nielsen pedagogical anchor — which both fixes the missing-numbers gap and
brings the figure up to the standard the rest of the book already set.

**Grade:** **B.** Correct and assignable, but currently the *least vivid*
treatment of backprop a top-tier student will have seen, and visibly less polished
than its own Preliminaries preview. The numeric worked example + figure lift it to
A−/A.

**Top priorities (ranked):**
1. [P1] **BP-1** — Add a fully worked numeric example (small graph $e=(a+b)\cdot c$ *then* a tiny 2-2-1 net) showing real numbers forward and gradients backward. The canonical anchor; the digest calls it out explicitly.
2. [P1] **BP-2** — Replace/redraw `forward.svg` in the modern house style of `autograd-comp-graph.svg` (colored boxes, a *backward* pass overlaid in blue), so the "full" figure isn't poorer than the preview.
3. [P1] **BP-3** — Add a "what the framework actually does" paragraph tying the hand derivation to autograd: this `prod`-and-chain-rule recipe *is* reverse-mode AD; forward-point to `autograd.md` §"Forward versus Reverse Mode" (don't re-derive it).
4. [P2] **BP-4** — Sharpen the memory subsection with a concrete cost statement ($O(\text{depth} \times \text{batch} \times \text{width})$ activations) and a one-line forward pointer to gradient checkpointing as the standard remedy.
5. [P2] **BP-5** — Promote the exercises from mechanical to graded: tie Ex.1 to the worked example, ask for the numeric check, add a checkpointing / recomputation question.
6. [P2] **BP-6** — Currency & prose micro-fixes (James Brown aside, "Training Neural Networks" heading, intro framing, `prod` definition placement).

---

## 1. Coverage

### Add

**A worked numeric example (the one real gap — P1, `BP-1`).** This is the digest's
headline finding and I concur strongly. The section currently derives
$\partial J/\partial \mathbf W^{(2)}$, $\partial J/\partial\mathbf h$,
$\partial J/\partial\mathbf z$, $\partial J/\partial\mathbf W^{(1)}$ purely
symbolically. A student can follow every line and still not *feel* the algorithm.
Every best-in-class treatment anchors backprop with numbers:

- **CS231n** (`cs231n.github.io/optimization-2/`) — the $f(x,y,z)=(x+y)z$ circuit
  with green forward values and red backward gradients; "add gates distribute, max
  gates route, multiply gates swap." The single most reused teaching device for
  backprop.
- **Nielsen** Ch. 2 — derives BP1–BP4 and walks them on a concrete net.
- **Goodfellow** Ch. 6.5 — frames the same as reverse-mode AD.

Recommended (drafted in §4, `BP-1`): a *two-stage* example. **(i)** a minimal
$e=(a+b)\cdot c$ gate-graph with $a=2,b=1,c=-3$ to teach the "local gradient ×
incoming gradient" mechanic in the cleanest possible setting; **(ii)** a tiny
2-input → 2-hidden (ReLU) → 1-output net with specific weights, run forward to a
numeric loss and backward to numeric $\partial L/\partial\mathbf W^{(1)}$,
$\partial L/\partial\mathbf W^{(2)}$ — i.e. the *same* network the symbolic
derivation just produced, now instantiated. This makes the abstract `prod`
formulas land. The numbers can be verified against autograd, which doubles as a
new exercise (`BP-5`).

**A bridge sentence connecting the hand derivation to autograd (P1, `BP-3`).** The
intro says we "just invoked the backpropagation function" and now go "under the
hood," but the section never closes the loop by saying *the recipe you just did by
hand is exactly what `.backward()` runs* — accumulate a gradient at the output,
walk the graph in reverse multiplying local Jacobians (`prod`). One paragraph does
this and forward-points to `autograd.md`'s "Forward versus Reverse Mode"
subsection (which already exists and covers `jacfwd`/`jacrev`, `jvp`/`vjp`) for the
generality. **Keep this light** — per the task brief and scope map, reverse-mode
generality is owned by `autograd.md`; do not re-derive it here.

**A sharper memory statement (P2, `BP-4`).** The current memory paragraph
(lines 263–275) is qualitatively right but vague: "roughly proportional to the
number of network layers and the batch size." A top course states the cost
crisply — peak activation memory is $O(\text{depth}\times\text{batch}\times\text{width})$
because every intermediate $\mathbf z^{(l)},\mathbf h^{(l)}$ must be retained until
its gradient is consumed — and names the standard fix: **gradient (activation)
checkpointing**, which trades compute for memory by recomputing activations in the
backward pass. A one-line mention + forward pointer is in scope (it directly
answers Ex.5 about graphs too large for a GPU); a full treatment is not.

### Remove / trim

Nothing should be **cut** — the section is lean. Two small trims:

- **The James Brown aside (lines 45–47)**, "in the eternal words of funk virtuoso
  James Brown, you must 'pay the cost to be the boss'." It is charming but it
  *apologizes* for the material ("This may seem tedious"). A best-in-class text
  doesn't pre-concede that its core derivation is a slog; it makes the derivation
  feel inevitable. Recommend replacing with a one-line statement of *why* the
  step-by-step pays off (you will recognize this exact computation inside every
  autograd engine). See `BP-6`.
- The `prod` operator paragraph (lines 143–152) is good but currently floats in
  the middle of the Backpropagation section *after* the abstract chain-rule
  statement. Minor reorder noted below.

### Reorder / restructure

The four `##` headings are **Forward Propagation / Computational Graph of Forward
Propagation / Backpropagation / Training Neural Networks**. This is logical but:

- **"Computational Graph of Forward Propagation" as a standalone `##` is thin** —
  it is two sentences plus one figure. It would read better folded into "Forward
  Propagation" as a `###`, or (better) renamed and *expanded* to carry the worked
  example and the new figure (see `BP-1`/`BP-2`), becoming the section's visual
  centerpiece rather than an afterthought.
- **"Training Neural Networks" (line 240) is a misleading heading** for what is
  really "Why forward and backward interleave, and the memory consequence." It
  doesn't teach *training* (no SGD, no loop); a reader scanning the ToC expects
  the training algorithm. Rename to **"Interdependence and Memory"** or
  **"Putting It Together: The Training Loop's Memory Cost."** (`BP-6`).

Proposed spine (3 top-level `##`, each substantive):
1. **Forward Propagation** (current content + fold in the small computational-graph
   intro as a `###`).
2. **Backpropagation** (current derivation + new `### A worked example` carrying
   `BP-1` + the `### From the chain rule to autograd` bridge `BP-3`).
3. **Interdependence and Memory** (renamed "Training Neural Networks", + `BP-4`).

This keeps the file lean while giving the worked example and the bridge a home.

#### Currency check (written ~2021)

This is a **timeless** section — the math of backprop has not changed — so there
is little to update for currency, which is appropriate for an MLP-level
derivation. Two points:

- **No dated APIs or framework code** — there are *zero* code cells (confirmed:
  all four `outputs/<fw>/.../backprop.json` manifests are empty). So the
  MXNet-retirement and Flax/Keras-3/PyTorch-2.x currency concerns that dominate
  the rest of the chapter **do not apply here.** Worth stating in the orchestrator
  summary so this file isn't swept into a blanket MXNet-tab decision.
- The only currency lift is `BP-3`/`BP-4`: framing the derivation as reverse-mode
  AD and pointing at checkpointing — both modern standard vocabulary a 2026 reader
  expects, neither requiring new in-scope content.

---

## 2. Teaching quality

### Structure & flow

Mostly good (see Reorder above). The logical spine forward → graph → backward →
interdependence is the right one and mirrors how CMU 11-785 and CS231n sequence
it. The weak links are the thin standalone graph section and the mislabeled
"Training" heading. The derivation itself flows cleanly: each gradient is
introduced with "Next, we …" and a one-line motivation, and the shapes
($\in\mathbb R^q$, $\in\mathbb R^{q\times h}$, etc.) are annotated throughout —
**this shape-annotation is a real strength; keep it.**

### Figures

There is **one** figure, and it is the section's biggest teaching-quality liability.

| Figure | Source | Verdict |
|---|---|---|
| `:numref:`fig_forward`` "Computational graph of forward propagation" | `../img/forward.svg` (line 117) | **Pre-generated (good — no inline drawing code), but dated and incomplete.** |

Details:

- **Provenance / inline-drawing check: passes.** `forward.svg` is a committed
  static SVG (a legacy fig2dev/transfig-style export — monochrome, generic vector
  glyphs, no `<style>`/font block). There is **no figure-drawing matplotlib in any
  teaching cell** (there are no cells at all). So it does not violate the
  "no inline schematic drawing" rule. Good.
- **But it is visibly poorer than the book's own newer graph.** Compare to
  `img/autograd-comp-graph.svg` (added in commit `7a8e326`, used at
  `autograd.md:178`): that figure uses the house style — Source Sans 3 / JetBrains
  Mono fonts, **purple input box, blue intermediate box, green output box**, black
  forward arrows, **a blue backward pass**, and a worked chain-rule caption
  (`∂y/∂x = (∂y/∂a)(∂a/∂x) = 2·2x = 4x`). `forward.svg` is grey squares-and-circles
  with no color, no semantic legend, and **shows only the forward pass** — the one
  thing the section is *about* (backprop) is not in the figure at all.
- **The pedagogical miss:** this is the file `autograd.md` explicitly forward-points
  to as the "full" treatment, yet its figure is *less* informative than the preview.
  A best-in-class version has a **single graph that shows both passes** —
  forward values flowing up-right in black, gradients flowing back down-left in
  blue — exactly the CS231n green-forward/red-backward device, in the book's house
  colors. That is `BP-2`, and it should be **co-designed with the worked example
  `BP-1`** so the figure carries the actual numbers.

**Figure recommendation (P1, `BP-2`):** author a new house-style SVG
`img/backprop-graph.svg` (matching `autograd-comp-graph.svg` exactly — same fonts,
same box-color semantics) for the one-hidden-layer net, with the **forward pass in
black and the backward pass overlaid in blue**, annotated with the worked numbers
from `BP-1`. Either redraw `forward.svg` in place or add the new figure and demote
the old one. Like `autograd-comp-graph.svg`, this is a **directly-committed SVG**,
not a `gen_mdl_*figures.py` product — there is no generator for either graph, so
the `mdl-figure` skill (matplotlib) does *not* apply; hand-author the SVG.

### Prose & clarity

The prose breathes — short paragraphs, no walls of math, every equation
interpreted. A few specifics:

- **Strong:** the intro (lines 12–25) — "academic papers had to allocate numerous
  pages to deriving update rules" — is exactly the right motivation, and the line
  "if you want to go beyond a shallow understanding" earns the section.
- **Awkward grammar, line 175–176:** "we compute the gradient of the objective
  function with respect to **variable of** the output layer $\mathbf o$" — missing
  article; → "with respect to the output-layer variable $\mathbf o$." (`BP-6`)
- **Line 164:** "The order of calculations **are** reversed" → "is reversed."
  (`BP-6`)
- **Line 83–84:** "As we will see the definition of $\ell_2$ regularization to be
  introduced later" is tangled (it both says "as we will see" and "to be
  introduced later"). The forward-ref to `sec_weight_decay` is already on line 35;
  → "Recall the $\ell_2$ regularization term (:numref:`sec_weight_decay`): given
  the hyperparameter $\lambda$, …". (`BP-6`) Note: `weight-decay.md` is §3.7,
  *earlier* in the book than this §5.3, so "introduced later" is actually
  **backwards** — it was introduced earlier. The intro on line 35 correctly says
  "introduced in", but line 84 says "to be introduced later." Fix the
  contradiction.
- **`prod` operator (143–152):** clear and the right level of abstraction. One
  improvement: state once, explicitly, that for the linear layers in *this* example
  `prod` is just (possibly transposed) matrix multiplication, so the reader doesn't
  carry the abstraction as a mystery. (The text says this generally; tie it to the
  concrete steps.)

### Exercises

The five exercises (lines 288–296) are **decent and conceptual** — the bias-term
extension (Ex.2) with "draw the graph + derive both passes" is genuinely good and
mirrors a CMU-style problem; the second-derivative question (Ex.4) and the
multi-GPU partition (Ex.5) probe real understanding. Gaps for a top problem set:

- **No exercise uses numbers** — once `BP-1` lands, add: "Verify the gradients in
  the worked example by hand, then confirm them with autograd
  (:numref:`sec_autograd`)." This closes the loop and rewards the numeric example.
- **Ex.3 (memory footprint)** is good but should ask for the *asymptotic* answer
  ($O(\cdot)$ in depth/width/batch), not just "compute," and connect to
  checkpointing (`BP-4`).
- Consider adding a **vanishing/exploding-gradient seed**: "Using the backward
  equations, explain why a deep stack of layers with $\phi'<1$ everywhere drives
  $\partial J/\partial\mathbf z$ toward zero. Which section addresses this?"
  (forward-points to `numerical-stability-and-init.md`, the very next file) — this
  ties §5.3 to §5.4 and shows the *payoff* of having the backward equations in
  hand. (`BP-5`)

No exercise should be cut.

---

## 3. Code & examples

### Does the code teach?

**There is no code in this file** — it is pure theory + one figure. Confirmed by
inspection of the source (no ```` ```{.python .input} ```` fences) and by the four
empty executed-output manifests (`outputs/{pytorch,jax,tensorflow,mxnet}/chapter_multilayer-perceptrons/backprop.json`).
This is an appropriate choice for a derivation section. The optional `BP-1` worked
example can be presented as **inline numeric prose / a small static table**, not a
code cell — keeping the section framework-agnostic. (If a code cell is ever added
to *verify* the numbers with autograd, it should be a single untagged all-framework
cell using `d2l` helpers; but inline numbers are the cleaner choice here.)

### PyTorch / JAX / TensorFlow / MXNet

Not applicable — no per-framework code. The section correctly stays at the
mathematical level that is identical across frameworks, which is the right call for
backprop (the *concept* is framework-independent; the *API* lives in
`autograd.md`). **No MXNet-retirement action is needed for this file** (no MXNet
code to tombstone).

### Cross-framework consistency & d2l conventions

- **Notation is internally consistent** and matches the book: $\mathbf W^{(1)}$,
  $\mathbf W^{(2)}$, $\phi$, $\odot$, Frobenius norm $\|\cdot\|_\textrm F$, the
  `prod` operator. The `prod` operator is **defined here and reused in
  `bptt.md`** (`chapter_recurrent-neural-networks/bptt.md:298`) — so its definition
  is load-bearing book-wide; **do not remove or rename it.**
- Both display-equation labels resolve: `:eqlabel:`eq_forward-s`` (line 89) ↔
  `:eqref:` (line 252); `:eqlabel:`eq_backprop-J-h`` (line 200) ↔ `:eqref:`
  (line 258). Clean.
- The chain-rule dependency is satisfied upstream: `calculus.md` (§2.4) develops
  the scalar and multivariate chain rule and the matrix–vector form
  $\nabla_{\mathbf x} y = \mathbf A\,\nabla_{\mathbf u} y$ (lines 359–393) and
  promises backprop will formalize it. The activation derivative $\phi'$ used on
  line 224 is established in `mlp.md` (§5.1), which plots ReLU/sigmoid/tanh and
  their derivatives. So all prerequisites are in place — `BP-3` should *cite* these
  (`:numref:`subsec_calculus-grad``), not re-derive.

---

## 4. Implementation spec (downstream agents act on THIS)

### BP-1 — Worked numeric backprop example  ·  [P1] · [L] · [authored]
- **Type:** coverage / teaching
- **Where:** `chapter_multilayer-perceptrons/backprop.md` — insert a new `###`
  subsection inside the Backpropagation section, **after** the final gradient
  equation (the block ending at line 236, `…= \frac{\partial J}{\partial \mathbf{z}} \mathbf{x}^\top + \lambda \mathbf{W}^{(1)}.`)
  and **before** the `## Training Neural Networks` heading (line 240).
- **Change:** Add the worked example below verbatim (numbers chosen so they are
  clean by hand; the agent should double-check the arithmetic when integrating and
  may adjust the figure reference once `BP-2` lands):

  ```markdown
  ### A worked example

  Symbols can hide what backpropagation actually *does*, so let us push real
  numbers through a graph. We follow the rule that gave us every equation above:
  at each node, multiply the gradient arriving from downstream by the node's
  *local* derivative.

  Start with the simplest non-trivial graph, $e = (a + b)\,c$, evaluated at
  $a = 2$, $b = 1$, $c = -3$. The **forward pass** computes the intermediate
  $d = a + b = 3$ and then $e = d\,c = -9$. For the **backward pass** we seed
  $\partial e/\partial e = 1$ and walk back. The multiply node $e = d\,c$ has
  local derivatives $\partial e/\partial d = c = -3$ and $\partial e/\partial c = d = 3$.
  The add node $d = a + b$ has $\partial d/\partial a = \partial d/\partial b = 1$,
  so it simply *passes its incoming gradient through* to both inputs. Chaining,

  $$\frac{\partial e}{\partial a} = \frac{\partial e}{\partial d}\frac{\partial d}{\partial a} = -3,\quad
    \frac{\partial e}{\partial b} = -3,\quad
    \frac{\partial e}{\partial c} = 3.$$

  This is the whole algorithm in miniature: *add* nodes broadcast the upstream
  gradient unchanged, *multiply* nodes scale it by the other input. Every backward
  equation in this section is an instance of this one move.

  Now run the same machinery on the network of :numref:`fig_forward`, shrunk to
  $d = h = 2$, $q = 1$, with ReLU activation $\phi(z) = \max(0, z)$ and (for
  clarity) no regularization, $\lambda = 0$. Take

  $$\mathbf{x} = \begin{bmatrix} 1 \\ 2 \end{bmatrix},\quad
    \mathbf{W}^{(1)} = \begin{bmatrix} 1 & -1 \\ 0 & \phantom{-}1 \end{bmatrix},\quad
    \mathbf{W}^{(2)} = \begin{bmatrix} 2 & -1 \end{bmatrix},\quad y = 0,$$

  and squared-error loss $L = \tfrac12 (o - y)^2$. **Forward:**
  $\mathbf{z} = \mathbf{W}^{(1)}\mathbf{x} = [-1,\ 2]^\top$, so
  $\mathbf{h} = \phi(\mathbf{z}) = [0,\ 2]^\top$ (the first unit is dead),
  $o = \mathbf{W}^{(2)}\mathbf{h} = -2$, and $L = \tfrac12(-2)^2 = 2$.
  **Backward:** $\partial L/\partial o = o - y = -2$. Then

  $$\frac{\partial L}{\partial \mathbf{W}^{(2)}} = \frac{\partial L}{\partial o}\,\mathbf{h}^\top = -2\,[0,\ 2] = [0,\ -4],$$

  $$\frac{\partial L}{\partial \mathbf{h}} = {\mathbf{W}^{(2)}}^\top \frac{\partial L}{\partial o} = [-4,\ 2]^\top,\qquad
    \frac{\partial L}{\partial \mathbf{z}} = \frac{\partial L}{\partial \mathbf{h}} \odot \phi'(\mathbf{z}) = [-4,\ 2]^\top \odot [0,\ 1]^\top = [0,\ 2]^\top,$$

  using $\phi'(z) = \mathbf{1}[z > 0]$, which is exactly where the *dead* first
  unit blocks the gradient. Finally

  $$\frac{\partial L}{\partial \mathbf{W}^{(1)}} = \frac{\partial L}{\partial \mathbf{z}}\,\mathbf{x}^\top
    = \begin{bmatrix} 0 \\ 2 \end{bmatrix}[1,\ 2]
    = \begin{bmatrix} 0 & 0 \\ 2 & 4 \end{bmatrix}.$$

  Notice that the row of $\partial L/\partial \mathbf{W}^{(1)}$ feeding the dead
  unit is entirely zero: no gradient, no learning signal — the concrete face of
  the "dying ReLU" we met in :numref:`sec_mlp`. You can confirm every number here
  in a few lines with automatic differentiation (:numref:`sec_autograd`); doing so
  is a good way to convince yourself the framework is running exactly this
  computation.
  ```
- **Touches:** Pairs with `BP-2` (the figure should carry these numbers). The
  `:numref:`sec_mlp`` and `:numref:`sec_autograd`` references must resolve — verify
  the labels (`mlp.md` is `sec_mlp`; `autograd.md` is `sec_autograd`).
- **Done when:** Renders in HTML and PDF with `make html` clean; the arithmetic is
  internally consistent ($L=2$; $\partial L/\partial\mathbf W^{(2)}=[0,-4]$;
  $\partial L/\partial\mathbf W^{(1)}=\big(\begin{smallmatrix}0&0\\2&4\end{smallmatrix}\big)$);
  no `$`-before-digit PDF tripwire introduced.
- **Depends on:** none (can land before the figure; figure then illustrates it).

### BP-2 — Redraw the computational-graph figure in the house style, with a backward pass  ·  [P1] · [M] · [authored]
- **Type:** figure
- **Where:** `chapter_multilayer-perceptrons/backprop.md:117` —
  `![Computational graph of forward propagation.](../img/forward.svg)` +
  `:label:`fig_forward`` (line 118).
- **Change:** Author a new house-style SVG matching `img/autograd-comp-graph.svg`
  **exactly** in styling (fonts: `Source Sans 3` / `JetBrains Mono`; rounded
  `rect` nodes; **purple `#7D12BA` input, blue `#2196F3` intermediate, green
  `#43A047` output**; black `#15181C` forward arrows; **blue `#2196F3` backward
  arrows**; "forward →" / "← backward" corner labels). Depict the
  one-hidden-layer-with-$\ell_2$ graph of this section: nodes
  $\mathbf x \to \mathbf z \to \mathbf h \to \mathbf o \to L$, with
  $\mathbf W^{(1)},\mathbf W^{(2)}$ feeding in and $s$ joining at $J = L + s$, the
  **forward pass in black and the backward gradients overlaid in blue** (label key
  edges with the local derivatives, e.g. $\partial\mathbf o/\partial\mathbf h = {\mathbf W^{(2)}}^\top$).
  Update the caption to **"Computational graph for the one-hidden-layer MLP with
  $\ell_2$ regularization. The forward pass (black) flows from $\mathbf x$ to the
  objective $J$; backpropagation (blue) walks the same graph in reverse,
  multiplying local derivatives via the chain rule."** Keep `:label:`fig_forward``
  unchanged (it is referenced at lines 108, 156 here and from
  `hyperopt-intro.md:343` and `bptt.md`).
- **Touches:** New committed file `img/forward.svg` (overwrite in place) or
  `img/backprop-graph.svg` (new — then update the `![...]` path). This is a
  **directly-committed SVG** like `autograd-comp-graph.svg`; there is **no
  matplotlib generator** for it, so the `mdl-figure` skill does not apply —
  hand-author the SVG. Run the `figure-style-audit` skill afterward (caption
  integrity, attached `:label:`, byte-idempotent).
- **Done when:** Figure renders in HTML and PDF; shows *both* passes; visually
  matches `autograd-comp-graph.svg`'s style; `:numref:`fig_forward`` still resolves
  at all four call sites (this file ×2, `hyperopt-intro.md`, `bptt.md`).
- **Depends on:** Coordinate numbers with `BP-1` if the figure annotates the worked
  example (recommended).

### BP-3 — Bridge paragraph: this derivation *is* reverse-mode autodiff  ·  [P1] · [S] · [authored]
- **Type:** coverage / currency
- **Where:** `chapter_multilayer-perceptrons/backprop.md` — end of the
  Backpropagation section, after the worked example (`BP-1`) if present, else after
  line 236.
- **Change:** Add this paragraph verbatim:

  ```markdown
  What we have just done by hand is precisely what a deep learning framework does
  when you call `backward()`: it records the computational graph during the forward
  pass, seeds a gradient of $1$ at the scalar objective, and sweeps the graph in
  reverse, multiplying the local derivative at each node — our $\textrm{prod}$ — to
  accumulate the gradient with respect to every parameter in a *single* pass. This
  output-to-input sweep is *reverse-mode* automatic differentiation, and it is cheap
  exactly when there are many parameters and one scalar loss — the deep learning
  regime. We use it throughout the book and developed the mechanics, including when
  the opposite *forward mode* is preferable, in :numref:`sec_autograd`.
  ```
- **Touches:** none. `:numref:`sec_autograd`` must resolve (it does — `autograd.md`).
- **Done when:** Renders clean; does not re-derive reverse-mode (one paragraph,
  forward-pointer only — honoring the scope map).
- **Depends on:** none (reads better after `BP-1`).

### BP-4 — Sharpen the memory subsection + checkpointing pointer  ·  [P2] · [S] · [authored]
- **Type:** coverage / teaching
- **Where:** `chapter_multilayer-perceptrons/backprop.md:271–275`, the sentences
  "Besides, the size of such intermediate values is roughly proportional to the
  number of network layers and the batch size. Thus, training deeper networks using
  larger batch sizes more easily leads to *out-of-memory* errors."
- **Change:** Replace with:

  ```markdown
  Concretely, peak activation memory scales as
  $O(\textrm{depth} \times \textrm{batch size} \times \textrm{width})$: every
  intermediate $\mathbf{z}$ and $\mathbf{h}$ must be kept alive until the backward
  pass consumes it. This is why training deeper networks with larger batches so
  readily triggers *out-of-memory* errors, and why a standard remedy — *gradient
  (activation) checkpointing* — keeps only a subset of activations and *recomputes*
  the rest during the backward pass, trading extra computation for memory.
  ```
- **Touches:** none.
- **Done when:** Renders clean; the $O(\cdot)$ expression has no `$`-before-digit
  issue; "gradient checkpointing" is named.
- **Depends on:** none.

### BP-5 — Upgrade exercises (numeric verification, asymptotics, vanishing-gradient seed)  ·  [P2] · [S] · [authored]
- **Type:** teaching
- **Where:** `chapter_multilayer-perceptrons/backprop.md:288–296` (the `## Exercises`
  list).
- **Change:**
  - Append a new item after current Ex.1: `1. Verify by hand the forward values
    and the gradients $\partial L/\partial \mathbf{W}^{(1)}$ and
    $\partial L/\partial \mathbf{W}^{(2)}$ in the worked example of this section,
    then reproduce them with automatic differentiation (:numref:`sec_autograd`). Do
    they match?` (only if `BP-1` landed).
  - Edit Ex.3 (line 292) `old → new`:
    `Compute the memory footprint for training and prediction in the model described in this section.`
    → `Give the memory footprint for training versus prediction in the model of this section as a function of depth, width, and batch size. Why is the training cost larger, and what does gradient checkpointing buy you?`
  - Append: `1. Using the backward equations, explain why a deep stack of layers
    whose activation satisfies $\phi'(z) < 1$ everywhere drives the gradient
    $\partial J/\partial \mathbf{z}$ toward zero as it propagates back. Which
    section of this chapter studies this failure mode and its remedies?`
    (forward-points to `numerical-stability-and-init.md`).
- **Touches:** none. The new Ex.1-followup `:numref:`sec_autograd`` and the
  vanishing-gradient item's implicit pointer to §5.4 should resolve.
- **Done when:** Renders clean; numeric-verification exercise is present iff `BP-1`
  is; no exercise was removed.
- **Depends on:** `BP-1` (for the numeric-verification item only).

### BP-6 — Prose, heading, and framing micro-fixes  ·  [P2] · [S] · [mechanical]
- **Type:** prose
- **Where / Change (apply each `old → new`):**
  - `backprop.md:164` — `The order of calculations are reversed` → `The order of calculations is reversed`.
  - `backprop.md:175–176` — `with respect to variable of the output layer $\mathbf{o}$` → `with respect to the output-layer variable $\mathbf{o}$`.
  - `backprop.md:83–85` — `As we will see the definition of $\ell_2$ regularization\nto be introduced later,\ngiven the hyperparameter $\lambda$,` → `Recall the $\ell_2$ regularization term (:numref:`sec_weight_decay`): given the hyperparameter $\lambda$,` (removes the tangled phrasing **and** fixes the backwards "to be introduced later" — `sec_weight_decay` is §3.7, *earlier* than this §5.3).
  - `backprop.md:45–48` — replace `This may seem tedious but in the eternal words\nof funk virtuoso James Brown,\nyou must "pay the cost to be the boss".` → `Working through it once pays off: you will recognize this exact computation running inside every automatic-differentiation engine.`
  - `backprop.md:240` — heading `## Training Neural Networks` → `## Interdependence and Memory`.
- **Touches:** none. `:numref:`sec_weight_decay`` resolves (`weight-decay.md:7`).
- **Done when:** All five strings replaced; `make html` clean; ToC shows the
  renamed heading.
- **Depends on:** none.

---

## 5. Keep — what is already excellent (do not lose this)

- **The derivation is correct and complete.** Every chain-rule step is explicit,
  the regularizer is threaded through *both* parameter gradients
  (:eqref:`eq_backprop-J-h` and the $\mathbf W^{(1)}$ analogue), and the
  output→hidden→input ordering is right. Do not "simplify" it.
- **Shape annotations on every gradient** ($\in\mathbb R^q$, $\in\mathbb R^{q\times h}$,
  $\in\mathbb R^h$, $\in\mathbb R^{h\times d}$) — a genuine pedagogical strength
  that prevents the most common student error. Keep them.
- **The `prod` operator** as a notational device that hides transpose/shape
  bookkeeping — clean, and **reused in `bptt.md`**, so it is load-bearing.
  Preserve the definition.
- **The intro framing** (lines 12–25): the historical "papers spent pages deriving
  update rules" motivation is exactly right and earns the section.
- **The memory insight** (lines 263–270): correctly identifies *why* training costs
  more memory than inference (intermediates retained until backward completes) —
  `BP-4` only sharpens it, does not replace the idea.
- **No code, no MXNet baggage:** the framework-agnostic level is the right choice
  for a backprop derivation; this file needs **no** MXNet-tab or API-currency
  surgery.
- **Correct, in-scope forward/back-pointers:** to `sec_weight_decay` (L2),
  `sec_autograd` (the recipe-in-practice), and the chain rule in `calculus.md` —
  this file *consumes* upstream results rather than re-deriving them, which is
  exactly the discipline the review guide asks for.
