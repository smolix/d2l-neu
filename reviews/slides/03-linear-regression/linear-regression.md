# Slide outline — §3.1 `linear-regression`

**Source:** `chapter_linear-regression/linear-regression.md`
**Status:** ready to build (no stale content). Existing `<!-- slides -->` block is
legacy (mostly equation/text walls, no cover/divider/kicker/`@fig:`); treat as a
source of ideas and rebuild to the north-star bar.
**Frameworks:** all four cells present & identical in structure. One framework-aware
caption needed (JAX vectorization timing). No per-framework cell-id gaps.

## What this section teaches
The linear model $\hat y=\mathbf w^\top\mathbf x+b$, squared-error loss, the two ways
to fit it (closed form vs minibatch SGD), why vectorization matters, and the
Gaussian-noise / MLE justification for squared loss. This is the conceptual cover of
the whole chapter — heavy on *ideas and pictures*, light on code (only two
genuinely-teaching code groups exist: the vectorization benchmark and the normal
density plot).

## Code-cell inventory (notebook order)
| id | what | output | use |
|---|---|---|---|
| `linear-regression` | imports | none | skip (setup) |
| `…-vectorization-for-speed-1` | `n=1000`, two `ones` vectors | none | `@-` setup on the bench slide |
| `…-vectorization-for-speed-2` | Python for-loop add + timing | `'0.01228 sec'` (pt); JAX `'1.01515 sec'` | show — loop |
| `…-vectorization-for-speed-3` | `a+b` + timing | `'0.00021 sec'` | show — vectorized |
| `…-the-normal-distribution-and-squared-loss-1` | `def normal(...)` | none | show (small, teaches the density formula in code) |
| `…-the-normal-distribution-and-squared-loss-2` | plot 3 normal densities | `[plot]` | show — the bell curves |

## Diagrams
- **NEW `linear-regression-single-neuron`** — *reuse the book idea, redraw in engine
  style.* The chapter already has `img/singleneuron.svg` (inputs $x_1..x_d$ wired to
  one output $o_1$) and `img/fit-linreg.svg` (points scattered about a fitted line).
  For the deck author a clean engine version: $d$ input chips → weighted edges
  ($w_1..w_d$) → sum node $\Sigma{+}b$ → output $\hat y$. This is *the* picture of the
  section. (Could instead inline `img/singleneuron.svg` via a plain image line as a
  fast path, but engine fonts/look are preferred.)
- **NEW `linear-regression-fit-line`** (optional, opener) — a 1-D scatter with the
  fitted line and one vertical residual highlighted; mirrors `fig_fit_linreg`. Use on
  the "what regression is" opener. Reusing `img/fit-linreg.svg` inline is acceptable.
- **NEW `linear-regression-squared-vs-gaussian`** — pairs the squared-loss parabola
  $\tfrac12(\hat y-y)^2$ with the Gaussian bell $\exp(-\tfrac12(\cdot)^2)$ so the MLE
  equivalence is *seen*, not just asserted. Pairs with the normal-density plot slide.
- Inline computed plot: the normal densities come from cell `…-squared-loss-2`
  (`@!…` output-only or `@…`).

## Slide list

1. **Cover** — `::: {.cover}` kicker "Dive into Deep Learning · §3.1"; title
   "Linear Regression — the model every deep net generalizes." Teaser figure:
   `@fig:linear-regression-single-neuron` (or `@!…-squared-loss-2`).
2. **Why start here? (opener)** — `.cols`: left, the house-price story in 3 bullets
   (features → weighted sum → price; fit on data; predict unseen); right
   `@fig:linear-regression-fit-line`. One idea: regression draws the line.
3. **Divider 01 — The Model.**
4. **The linear model** — `.cols .vc`: left the equation
   $\hat y=\mathbf w^\top\mathbf x+b$ and the design-matrix form
   $\hat{\mathbf y}=\mathbf X\mathbf w+b$; right `@fig:linear-regression-single-neuron`.
   `.d2l-note`: weights = feature influence, bias = offset; affine, not linear.
   No code.
5. **Squared-error loss** — the per-example $\ell^{(i)}=\tfrac12(\hat y^{(i)}-y^{(i)})^2$
   and the mean $L(\mathbf w,b)$. One line: convex in $(\mathbf w,b)$ → every local min
   is global. `.d2l-note .warn`: quadratic penalty → outlier-sensitive. No code.
6. **Divider 02 — Two Ways to Fit.**
7. **Closed form** — $\mathbf w^*=(\mathbf X^\top\mathbf X)^{-1}\mathbf X^\top\mathbf y$;
   one-line geometric reading (orthogonal projection of $\mathbf y$ onto the column
   space; residual ⟂ every feature). `.d2l-note`: unique iff $\mathbf X$ full rank;
   doesn't generalize past linear models. No code. *(Optional NEW diagram
   `linear-regression-projection`: $\mathbf y$, its projection onto a 2-D column space,
   the perpendicular residual — strongly recommended, this is a geometric idea; the
   engine's `dotProduct` projection drawing is the template.)*
8. **Minibatch SGD** — the update
   $(\mathbf w,b)\leftarrow(\mathbf w,b)-\tfrac{\eta}{|\mathcal B|}\sum\nabla\ell$;
   3 bullets (sample $\mathcal B$ / gradient of mean loss / step downhill).
   `. . .` fragment: hyperparameters ($\eta$, $|\mathcal B|$) are user-set, tuned on
   validation. No code (the recipe is the point; code arrives in §3.4).
9. **Divider 03 — Vectorize.**
10. **Loop vs. vectorized add** — `.cols`: left the loop cell
    `@…-vectorization-for-speed-2` (`@-…-1` as silent setup above it); right the
    one-call cell `@…-vectorization-for-speed-3`. Caption: same answer, ~50× faster
    (pt). **Framework note:** JAX's loop is ~1.0 s (immutable `.at[].set()` per
    element) — keep the caption "orders of magnitude," not a hard number, so it reads
    right on every tab. `.d2l-note .rule`: push inner loops into compiled kernels.
11. **Divider 04 — Why Squared Loss?** (or fold into slide 12's kicker).
12. **Squared loss = Gaussian MLE** — `.cols .vc`: left the noise model
    $y=\mathbf w^\top\mathbf x+b+\epsilon,\ \epsilon\sim\mathcal N(0,\sigma^2)$ and the
    one-line punchline (minimizing squared error = maximizing Gaussian
    log-likelihood); right `@fig:linear-regression-squared-vs-gaussian`. Show the
    `normal` definition cell `@…-squared-loss-1` (small, illustrative).
13. **Seeing the bells** — `@…-squared-loss-2` (the three normal densities plot).
    Caption: squared loss assumes errors look like one of these bells centered at the
    prediction; shifting the mean slides it, more variance flattens it.
14. **Recap** — model / loss (convex) / optimizer (SGD; closed form exists but
    doesn't generalize) / vectorize everything / squared loss = MLE under Gaussian
    noise (template for matching loss to noise model).

## Notes & flags
- **Drift in the legacy block:** it says "10 000-element vectors"; the notebook uses
  `n = 1000`. Use 1000 in the new captions.
- **No biology slide.** The McCulloch–Pitts neuron / `fig_Neuron` photo is book prose,
  not a teaching idea for the deck — omit (consistent with "code teaches; figures are
  schematic"; the anatomical photo is explicitly kept as-is in the book but is not
  slide material).
- **No per-framework framing divergence** beyond the JAX timing caption — this is a
  code/output swap section, so **zero `only=`/`except=` slides** (like linear-algebra).
  Keep slide 10's caption framework-neutral.
- Per-framework omissions: none (all 6 cells exist in all four tabs).
