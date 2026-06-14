# Slide outline — §5.3 `backprop.md` (Forward/Backward Propagation & Computational Graphs)

**Status: ready to design — but unusual.** This section has **NO
`<!-- slides -->` block today** and **NO Python code cells at all** (zero cells
in every framework's `outputs/` JSON). It is pure math + two computational-graph
figures + a hand-worked numerical example. So this deck is **100% diagram- and
math-driven, framework-agnostic, with no `@id` placeholders.** That is exactly
the north-star ideal (cf. calculus: "6 geometric diagrams carry the chapter,
code is incidental") — here there is no code at all, by design.

**Not stale.** The content is current: it derives backprop via the chain rule
on a one-hidden-layer net with $\ell_2$ regularization, then walks a concrete
2→2→1 ReLU example by hand (with a *dead* unit), and ties it to reverse-mode
autograd. The "dying ReLU" callback to §5.1 and the autograd pointer to §2.5 are
good forward/back-links.

## Cells

None. Nothing to inject; no per-framework variants; **no per-framework gaps**.
The deck will be identical for all four frameworks (a single shared block, no
`only=`/`except=`).

## Figures available

- `img/forward.svg` — legacy book figure: computational graph of the **forward**
  pass (squares = variables, circles = operators; input lower-left, output
  upper-right) for the one-hidden-layer regularized net. Reuse for the
  forward-graph slide.
- `img/mdl-mlp-backprop-graph.svg` — **the hero figure** (matplotlib,
  house-style): the worked 2→2→1 ReLU example as a computational graph, forward
  values in **black**, backward gradients in **blue**, with the dead first unit
  ($z_1=-1$) zeroing the top row of $\partial L/\partial\mathbf W^{(1)}$. This
  single figure carries the whole section.
- No `@fig:` engine module. **Strong candidate for an engine diagram** if the
  author wants the forward+backward *animation* feel via fragments, but the
  committed `mdl-mlp-backprop-graph.svg` already shows both directions in one
  static image and is the book figure — recommend reusing it. (Note: the design
  spec calls out `autograd-comp-graph` as the analogous engine diagram for §2.5;
  a `diagrams/backprop.mjs` mirroring it is the optional upgrade path.)

## The worked numbers (from the source, for caption/figure fidelity)

Toy graph $e=(a+b)c$ at $a{=}2,b{=}1,c{=}-3$: forward $d{=}3, e{=}-9$; backward
$\partial e/\partial a=\partial e/\partial b=-3$, $\partial e/\partial c=3$.
Rule: **add nodes pass the gradient through; multiply nodes scale by the other
input.**

Network: $\mathbf x=[1,2]^\top$,
$\mathbf W^{(1)}=\left(\begin{smallmatrix}1&-1\\0&1\end{smallmatrix}\right)$,
$\mathbf W^{(2)}=[2,-1]$, $y=0$, squared-error, ReLU, $\lambda=0$.
Forward: $\mathbf z=[-1,2]^\top$, $\mathbf h=[0,2]^\top$ (first unit dead),
$o=-2$, $L=2$. Backward: $\partial L/\partial o=-2$;
$\partial L/\partial\mathbf W^{(2)}=[0,-4]$;
$\partial L/\partial\mathbf h=[-4,2]^\top$,
$\partial L/\partial\mathbf z=[0,2]^\top$ (dead unit blocks gradient);
$\partial L/\partial\mathbf W^{(1)}=\left(\begin{smallmatrix}0&0\\2&4\end{smallmatrix}\right)$.

## Proposed slide list

1. **Cover** — `.cover`. Kicker "§5.3". Title: *Forward & Backward Propagation* /
   "what `backward()` actually does, by hand."

2. **Why look under the hood** (`title`) — autograd usually hides this; but to
   go beyond a shallow understanding you should know how the gradient is
   computed. Frame the destination: a one-hidden-layer net with weight decay; we
   want $\partial J/\partial\mathbf W^{(1)},\partial J/\partial\mathbf W^{(2)}$.
   (Short, motivational; no math wall.)

3. **Divider 01 — The Forward Pass.**

4. **Forward propagation as a graph** (`title`, `.cols .vc`) — left `.col .fig
   .big`: `![](../img/forward.svg)`. Right col: the five forward equations,
   condensed —
   $\mathbf z=\mathbf W^{(1)}\mathbf x$, $\mathbf h=\phi(\mathbf z)$,
   $\mathbf o=\mathbf W^{(2)}\mathbf h$, $L=l(\mathbf o,y)$,
   $J=L+\tfrac\lambda2(\|\mathbf W^{(1)}\|_F^2+\|\mathbf W^{(2)}\|_F^2)$.
   Caption: store every intermediate; arrows flow right-and-up.

5. **Divider 02 — Backpropagation.**

6. **One rule: the chain rule, in reverse** (`title`) — the $\textrm{prod}$
   operator and
   $\partial Z/\partial X=\textrm{prod}(\partial Z/\partial Y,\partial Y/\partial X)$.
   `. . .` fragment → the slogan as `.d2l-note .rule`: "**add** nodes broadcast
   the upstream gradient unchanged; **multiply** nodes scale it by the other
   input. Every backward equation is an instance of this one move."

7. **The simplest graph** (`title`, `.cols .vc`) — $e=(a+b)c$ at $(2,1,-3)$.
   Left col: forward $d=3,e=-9$. `. . .` fragment / right col: backward seeds
   $\partial e/\partial e=1$, gives $\partial e/\partial a=\partial e/\partial b=-3$,
   $\partial e/\partial c=3$. Caption: this *is* the whole algorithm in
   miniature. (A tiny inline 3-node sketch would help — could be a small engine
   diagram or hand SVG; otherwise the math carries it.)

8. **A real network, by the numbers** (`title`, `.cols .vc`) — **the hero
   slide.** Left `.col .fig .big`: `![](../img/mdl-mlp-backprop-graph.svg)`.
   Right col: the setup ($\mathbf x,\mathbf W^{(1)},\mathbf W^{(2)},y$, ReLU,
   $\lambda=0$) and the forward result $\mathbf h=[0,2]^\top$ (first unit dead),
   $o=-2$, $L=2$. Caption: forward in black.

9. **Gradients flow back** (`title`, `.cols .vc`) — reuse the same figure (or a
   `. . .`-revealed second copy emphasising the blue arrows). Right col: the
   backward chain $\partial L/\partial o=-2 \to \partial L/\partial\mathbf
   W^{(2)}=[0,-4] \to \partial L/\partial\mathbf h=[-4,2]^\top \to \partial
   L/\partial\mathbf z=[0,2]^\top \to \partial L/\partial\mathbf
   W^{(1)}=\left(\begin{smallmatrix}0&0\\2&4\end{smallmatrix}\right)$.
   `.d2l-note .warn`: the dead unit's row is **all zero** — the concrete face of
   "dying ReLU" from §5.1. (This is the emotional payoff: the abstract chain rule
   becomes visible numbers, and the dead unit's zero row is a *picture* of no
   learning signal.)

10. **From the chain rule to autograd** (`title`) — what we just did by hand =
    what `backward()` does: record the graph forward, seed 1 at the scalar loss,
    sweep in reverse multiplying local derivatives, accumulate every gradient in
    *one* pass. `.d2l-note`: this is **reverse-mode** AD — cheap exactly when
    there are many params and one scalar loss (the deep-learning regime). Pointer
    to §2.5 for forward vs reverse mode.

11. **Forward and backward are interdependent** (`title`) — training alternates
    them: backward reuses forward's stored intermediates (e.g. $\mathbf h$ in
    $\partial J/\partial\mathbf W^{(2)}$), which is *why training needs far more
    memory than prediction* — intermediates scale with depth × batch size →
    OOM on deep nets / big batches. (Genuinely useful systems intuition; keep.)

12. **Recap** (`title`) — forward = compute + store intermediates input→output;
    backprop = chain rule output→input, one local-derivative multiply per node;
    add broadcasts / multiply scales; the dead ReLU zeros its gradient row;
    `backward()` is exactly this reverse-mode sweep; training holds intermediates
    in memory until the backward pass completes.

## Per-framework notes

- **Fully framework-agnostic.** No code cells, no injected outputs, no `#@tab`
  anything. One shared `<!-- slides -->` block renders identically for
  pytorch/jax/tensorflow/mxnet. **Zero `only=`/`except=` slides.**

## Must port before non-pytorch decks
- **None** (no code).

## Author note
This is the cleanest north-star fit in the chapter: a math-and-figure deck with
no notebook coupling. The only real authoring choice is whether the toy-graph
slide (#7) and the forward/backward split (#8–9) want small inline engine
diagrams for the fragment-reveal of the backward sweep, or whether the committed
`forward.svg` + `mdl-mlp-backprop-graph.svg` (both already house-style book
figures) suffice. Recommend the latter for speed.
