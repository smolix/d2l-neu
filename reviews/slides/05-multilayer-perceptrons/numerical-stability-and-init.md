# Slide outline — §5.4 `numerical-stability-and-init.md`

**Status: ready to design.** No staleness. No per-framework gaps — every cited
cell exists in all four frameworks. Mostly math + two demo cells.

This deck is *failure-modes → variance analysis → fixes*: vanishing/exploding
gradients, symmetry breaking, then Xavier/He init. The existing block
(lines 470–626) is strong and complete but **text/math-heavy with no figure** —
the north-star upgrade wants a diagram of the **gradient-as-product-of-Jacobians**
spine (the one idea the whole section turns on) and lets the two demo cells
(sigmoid plot, exploding random-matrix product) carry their slides as In/Out
cards. Only two cells produce visible output.

## Notebook cells (all 4 fw)

| cell id | kind | tabs | shows |
|---|---|---|---|
| `numerical-stability-and-init-numerical-stability-and-initialization` | text (no out) | all 4 | imports |
| `numerical-stability-and-init-vanishing-gradients` | **svg plot** | all 4 | sigmoid + its gradient on $[-8,8]$ |
| `numerical-stability-and-init-exploding-gradients` | **text** | all 4 | one $4\times4$ Gaussian, then product of 100 → entries $\sim10^{24}$ |

## Output facts (for caption fidelity)

- `exploding-gradients`: prints "a single matrix" (a tame $4\times4$) then
  "after multiplying 100 matrices" with entries on the order of $10^{22}$–$10^{25}$
  (pytorch ~$10^{24}$, jax ~$10^{22}$ — same story, different seed). The point is
  the explosion, not the digits — trim to a few rows with `output-lines` if wide.
- `vanishing-gradients`: the sigmoid curve flat at both tails, gradient peaking
  at 0.25 near 0 and ~0 elsewhere.

## Figures available

- **No existing figure for this section** (no `mdl-mlp-stability-*.svg`). The
  conceptual spine — *the gradient is a product of $L-\ell$ Jacobians, which
  shrinks (<1 spectral radius) or grows (>1)* — has **no picture yet** and is the
  best diagram opportunity in the deck.
  - **Recommend a new diagram** (engine `diagrams/mlp.mjs` id
    `mlp-jacobian-product`, or a `tools/gen_mdl_mlp_*` matplotlib figure):
    a layer chain $\mathbf h^{(0)}\!\to\!\dots\!\to\!\mathbf h^{(L)}$ with the
    backward arrows labelled $\mathbf M^{(\ell)}$, and two annotated regimes
    (×0.8 per layer → vanish; ×1.2 per layer → explode). This is the slide-2
    hero and would teach the section's core idea visually.
  - If no new figure is authored, slide 2 carries the product formula as math
    only (acceptable, weaker).
- The two demo plots are *computed* data plots (loss-curve-like), correctly left
  as injected cell outputs (`@!`), not pre-generated figures.

## Proposed slide list

1. **Cover** — `.cover`. Kicker "§5.4". Title: *Numerical Stability &
   Initialization* / "why deep nets wouldn't train before we got init right."

2. **The whole problem in one picture** (`title`, `.cols .vc`) — **hero.** Left
   `.col .fig .big`: `@fig:mlp-jacobian-product` (NEW) or the product formula
   $$\frac{\partial\mathcal L}{\partial\mathbf W^{(\ell)}}=
     \frac{\partial\mathcal L}{\partial\mathbf h^{(L)}}\,
     \mathbf M_L\cdots\mathbf M_{\ell+1}\,
     \frac{\partial\mathbf h^{(\ell)}}{\partial\mathbf W^{(\ell)}}.$$
   Right col: the gradient is a **product of $L-\ell$ Jacobians**; spectral
   radius $<1$ everywhere → shrinks geometrically (**vanishing**), $>1$ → grows
   geometrically (**exploding**). `.d2l-note`: this single fact drives the rest
   of the deck. *(Replaces the existing "chain rule turns the gradient into a
   product" slide — same idea, now with a picture.)*

3. **Setup** (`title`) — `@numerical-stability-and-init-numerical-stability-and-initialization`.

4. **Divider 01 — How Gradients Misbehave.**

5. **Vanishing: sigmoid is the culprit** (`title`, `.cols .vc`) — left:
   `@!numerical-stability-and-init-vanishing-gradients` (output-only — the curve
   is the point). Right `.narrow` `.d2l-note .warn`: $\sigma'$ peaks at 0.25,
   collapses at the tails; a 10-layer stack → $0.25^{10}\approx10^{-6}$,
   gradients at layer 1 a millionth of those near the output. ReLU fixes it
   (derivative exactly 1 where active).

6. **Exploding: random-matrix products** (`title`, `.cols .vc`) — left `.col`
   (wide, for the matrix output): `@numerical-stability-and-init-exploding-gradients`
   — show the code and the $10^{24}$ blow-up after 100 mults. Right `.narrow`:
   Gaussian matrices have spectral radius $>1$ → the product diverges; the same
   on gradients in a poorly-scaled deep net → loss goes NaN in a few hundred
   steps. (Consider `output-lines="6"` to keep the matrix dumps from
   overflowing; trim the "single matrix" preamble if needed.)

7. **Crash modes you'll actually see** (`title`) — keep the existing slide's
   tight list: loss spikes mid-training (exploding on a bad batch); NaN from step
   1 (exploding init); loss won't move (vanishing, or LR too small). Practical,
   memorable — keep.

8. **Symmetry breaking** (`title`) — set every weight to constant $c$: every
   hidden unit computes the same thing → same gradient → stays the same forever →
   an $h$-unit layer acts like 1 unit. `. . .` fragment → `.d2l-note .rule`:
   initialize **randomly**; even tiny noise breaks the permutation symmetry. (SGD
   alone won't; dropout would — forward-link to §5.6.) Could pair with a small
   2-unit "mirror" sketch if a figure is authored, but math/prose is fine here.

9. **Divider 02 — Variance-Preserving Init.**

10. **Keep variance constant through depth** (`title`) — the forward derivation,
    condensed: $o_i=\sum_j w_{ij}x_j$; with $w\sim\mathcal N(0,\sigma^2)$, inputs
    var $\gamma^2$ → $\mathbb E[o_i]=0$, $\operatorname{Var}[o_i]=n_{\text{in}}\sigma^2\gamma^2$.
    Preserve variance forward ⇒ $\boxed{\sigma^2=1/n_{\text{in}}}$. `. . .`
    fragment: backward gives $\sigma^2=1/n_{\text{out}}$ — **can't satisfy both**.

11. **Xavier and He** (`title`, `.cols`) — two columns, one each:
    **Xavier/Glorot (2010)** $\sigma^2=\dfrac{2}{n_{\text{in}}+n_{\text{out}}}$,
    averages forward+backward, for tanh/sigmoid;
    **Kaiming/He (2015)** $\sigma^2=\dfrac{2}{n_{\text{in}}}$, compensates for
    ReLU *halving* post-activation variance, default for ReLU nets/Transformers.
    `.d2l-note`: rule of thumb — **Xavier for tanh/sigmoid, He for ReLU**; both
    ship as framework defaults; bias starts at 0. (Note `. . .` is top-level
    only — do NOT put a fragment inside the `.col`s.)

12. **Beyond init** (`title`) — keep the existing "modern building blocks" slide,
    trimmed: BatchNorm/LayerNorm (re-normalize during training), residual
    connections (direct gradient path, $h^{(\ell+1)}=h^{(\ell)}+f(h^{(\ell)})$),
    mixed precision. These largely remove the init burden in very deep nets
    (forward-link to §8 Modern CNNs / ResNet). One slide, not two.

13. **Recap** (`title`) — gradients are products of per-layer Jacobians → vanish
    or explode without care; vanishing = saturating activations + small weights;
    exploding = large weights → NaN; two fixes: non-saturating activations (ReLU)
    + variance-preserving init (Xavier/He); random init breaks symmetry, zero
    init collapses layers; BatchNorm + residuals + init together reach 100+
    layers.

## Per-framework notes

- **No gaps, no scoped slides needed.** All three cells exist in all four
  frameworks. The exploding-matrix code differs cosmetically per fw
  (`torch.normal` / `np.random.normal` / `tf.random.normal` /
  `jax.random.normal(d2l.get_key())`) — pure `#@tab` swap, neutral caption.
- The exploding-gradients *output values* differ by seed across frameworks (and
  JAX prints a numpy-style array) but the teaching point (10^2x+ blow-up) is
  identical — caption must not quote a specific magnitude as if universal; say
  "entries explode to ~$10^{20}$+".

## Must port before non-pytorch decks
- **None.**

## New-figure recommendation (optional but high-value)
`mlp-jacobian-product` — the layer chain with backward Jacobian labels and the
two ×<1 / ×>1 regimes. This is the section's central idea and currently has no
picture; it would lift the deck from "good math deck" to north-star.
