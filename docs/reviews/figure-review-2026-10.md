# Figure Review of Basics, Advanced, and the Mathematics Appendix

Date: 2026-10-23. Source: `main` at `732c1431`.
Standards: `CLAUDE.md` (figure house style), `docs/figure-style-guide.md`,
`docs/writing-guide.md`.

## Method

Eight reviews rendered every figure in the 23 chapters: the static figures
referenced in the text, and the notebook output plots (PyTorch store).
Reviewers inspected contact sheets and full-size renders, read the prose
around each figure, and searched each section for ideas that the text
explains without a picture. In all, 335 static figures and 332 output plots
were reviewed.

## Main Findings

1. **About sixty figures contradict or fail to show what the text claims.**
   These are the most harmful defects, because a reader trusts the picture.
   They are listed below and should be fixed before anything cosmetic.
2. **Most output plots hide their own result.** The axes, scales, or layout
   bury the effect the prose describes: linear axes for a noise floor, two
   comparisons in separately auto-scaled plots, a legend over the diverging
   curves, a log axis with one tick label. The cause is mostly four plotting
   helpers (`d2l.plot`, `ProgressBoard`, `show_heatmaps`, `show_grid`) whose
   defaults are wrong for teaching. Fixing the helpers repairs dozens of plots
   at once, at the cost of re-capturing the affected notebooks.
3. **Static figures are consistent within a chapter but not across the
   book.** Each chapter has its own generator. Nearly all write sizes in `pt`
   (so figures display 33% larger than designed and their type shrinks when
   scaled to the column), use tab10 hex colors rather than the figstyle
   tokens, and set labels at 10–11 pt. About 15 gallery and panorama figures
   are 800–1,000 pt wide and become illegible at column width. The JS diagram
   engine used in the Preliminaries puts nine figures below 12 px and does
   not outline its text, so the PDF build falls back to other fonts.
4. **Labels collide with lines** in about 60 static figures, the defect
   `CLAUDE.md` calls the most common.
5. **Many results are printed, not plotted.** About 40 places print a table
   or a list of numbers where the prose describes a curve or a comparison.
6. **Duplicates.** At least 20 figures repeat another figure: the zig-zag
   valley appears four times across ch. 24 and 26, the OU paths three times
   in ch. 29, sixteen identical airfoil loss curves in ch. 9, six activation
   plots in ch. 4.
7. **Important pictures are missing.** Several sections that teach a
   structural or geometric idea have no figure at all: batch normalization,
   SAC, DeltaNet, the concentration section (about 1,000 lines), numerical
   stability (two figures in 1,250 lines). The highest-value additions are
   listed per chapter below.

## Figures That Contradict the Text

| Chapter | Figure or plot | Problem |
|---|---|---|
| 5 | `bg-safetensors-layout` (`saving-loading.md:338`) | shows a data region before `hidden.weight`, whose offsets start at 0 |
| 5 | capstone plot (`gpus-devices-memory.md:1716`) | trains one epoch: two points and empty validation entries |
| 5 | `bg-activation-checkpoint` | states a fixed 1.3× overhead that the prose says is not fixed |
| 6 | LeNet training plot (`lenet.md:310`) | about 0.70 validation accuracy, below the MLP's 0.82, while the summary claims the advantage of image structure |
| 6 | `lenet-vert.svg` | prose asks readers to check shapes against it; it shows none |
| 7 | `arch-anynet` | caption uses $c_i$, figure $w_i$ |
| 8 | RNN perplexity plot (`rnn-implementation.md:662`) | starts at $10^7$; prose says "near the vocabulary size" |
| 2 | polynomial fitting (`generalization.md:376`) | y-axis spans 28 decades; the U-shape the text cites is invisible |
| 2 | scratch and concise training plots | linear axes hide the $\sigma^2/2$ noise floor the text describes |
| 2 | bias–variance plot | bias² climbs back at degree 14, a numerical artifact contradicting the text |
| 3 | confusion matrix (scratch) | ticks at even classes only; prose cites classes 3, 5, 7, 9 |
| 4 | `forward-graph` (`backprop.md:128`) | prose says circles are operators; there are no circles |
| 9 | `branching-off-the-plateau` (`lr-scheduler.md:719–790`) | prose describes branched runs; the plot repeats one series three times |
| 11 | scaling-law plot (`scaling-laws.md:416`) | the extrapolated line the prose discusses is not drawn |
| 12 | Δ boxplot (`mamba.md:1056`, prose `:1106`) | **prose says Δ opens on symbols and stays low on filler; the plot shows the opposite in both frameworks** |
| 12 | LSTM run (`ssm.md:1119`) | prose says clipping stabilizes the LSTM; the clipped run still spikes |
| 12 | `three-answers` (`mamba.md:1124`) | puts minGRU under fixed, content-blind dynamics; `ssm.md:456` says its decay is input-dependent |
| 13 | GPU sweep (`performance-model.md:385`) | prose reads it against a roofline that is not drawn |
| 13 | memory snapshot (`memory-precision.md:124`) | "allocated (MB)" goes to −100 MB; it plots a change, not an allocation |
| 13 | `roofline` | elementwise op at intensity ≈ 1; the text gives 0.25 |
| 13 | `float-formats` | tf32 drawn as 19 bits; the text says the figure shows 32-bit storage |
| 14 | regret plot (`qlearning.md:341`) | linear axes; prose describes "the logarithmic plot" |
| 14 | occupancy grids (`imitation.md:353`) | two identical-looking maps; prose says they localize a difference at state 13 |
| 14 | `roadmap` | caption's arrow to PPO ends in empty space |
| 15 | DQN runs (`dqn.md:429, :521`) | **prose says some seeds settle above the ceiling of 100; stored estimates are 93.8, 98.3, 93.0** |
| 15 | `kl-tilting` | caption says a dip survives in both rows; the bottom row is all 0.00 |
| 15 | PPO diagnostics (`ppo.md:450`) | prose describes KL rising and entropy falling; one shared axis shows neither |
| 15 | `deadly-triad` | in-figure sentence contradicts the caption's hedge |
| 16 | 25-Gaussians plot (`convergence.md:499`) | titles say "25/25 modes"; panels show a near-uniform square |
| 16 | `mdl-gan-pairing` | figure writes $+c$, prose $b$ |
| 17 | `mdl-diffusion-low-density` (`langevin.md:282`) | a random "schematic" field overstates a failure that the notebook can show for real |
| 24 | `eig-ellipse` (`mdl-eigendecomposition.md:96`) | caption promises green eigenvectors and red images; neither is drawn |
| 24 | `gershgorin` (`:1128`) | caption says crosses are eigenvalues; crosses are disc centres |
| 24 | `null-collapse` | nilpotent $B$ makes null space equal column space; the text apologizes |
| 24 | `hyperplane`, `power-iter` | captions describe elements the figures lack |
| 25 | `gd-step`, `gradient-field` | the drop and the normals the captions describe are not visible |
| 26 | `local-equals-global` (`mdl-convexity.md:554`) | **"local" and "global" labels are swapped** |
| 26 | `convex-vs-nonconvex-set` (`:75`) | the non-convex example does not show non-convexity; simplex drawn filled |
| 26 | `kkt-active-set` (`mdl-constrained-optimization-duality.md:278`) | draws $+\nabla f$; text says $-\nabla f$ |
| 26 | `schedule-zoo` | caption claims a warm-up the runs lack |
| 27 | CLT plot (`mdl-distributions.md:561`) | promised $N(0,1)$ overlay missing: `d2l.plot` silently drops the fifth series |
| 27 | family tree (`:19`) | caption promises arrows and an envelope that are not drawn |
| 27 | `bootstrap`, `fisher`, `chebyshev` | captions describe layouts or ellipses that differ from the figures |
| 28 | log-N ceiling (`mdl-mutual-information.md:727`) | the $\ln N$ lines, the point of the plot, are not drawn |
| 29 | FM paths | drawn as tubes that nearly touch, undercutting "never crosses" |

The ch. 12 Δ result matters most: it decides how the rewritten SSM chapter
explains selectivity. Either the analysis changes (per-channel Δ, or the
effective retention $e^{\Delta a}$ on the channels the readout uses) or the
claim does.

## Infrastructure Fixes

These fixes change many figures at once.

**Plotting helpers** (the affected notebooks must be re-captured):

- `d2l.plot`: cycle line formats instead of silently dropping series beyond
  four (a bug, *verified*); log axes with ticks at the data and a plain
  formatter (nine plots show overlapping or single tick labels); a fixed
  color per method name across a chapter; mathtext labels.
- `ProgressBoard`: a y-axis label, decade ticks on log axes, separate axes for
  loss and accuracy, and an overlay mode for comparing runs on one zoomed
  axis (four paired comparisons in ch. 7 alone).
- `show_heatmaps`: integer ticks and token tick labels; shared color limits
  across panels; a diverging colormap for signed data; one colormap.
- `show_grid` and `plot_curves` (RL): a light colormap, state indices, a
  colorbar, fixed colors per method, interquartile bands instead of
  overlapping min–max bands.

**Static figures:**

- Write unitless SVG sizes in the shared save functions (one change fixes
  about 50 figures per chapter group).
- Raise the label floor to 13 pt in every generator, and cap canvas width at
  the column.
- Move all generators to the figstyle color tokens.
- JS diagram engine: a 12 px minimum and outlined text.

**Decisions needed:**

- **Axis color.** `CLAUDE.md` rule 1 says coordinate axes are black;
  `docs/figure-style-guide.md` §7.3 and `figstyle.axis_cross` use muted grey.
  About 8 Preliminaries figures and every schematic in ch. 24–26 use grey.
- **Static versus computed.** `docs/figure-style-guide.md` §3 still lists
  tab10 as an anti-pattern although the 2026-08 palette revision restored
  tab10 hues. One of the two needs updating.

## Chapter Notes

Counts are static figures and output plots, and how many of each have
defects. "Add" lists the most valuable new figures first.

### Ch. 1 Preliminaries (20 / 3; 14 with defects)

Three generations of figures in one chapter. Nine JS figures below 12 px;
six repeat their caption inside the image. **Add:** axis reductions and
`keepdims` (the top beginner confusion); finite-difference error against
$h$; detached versus attached graphs; forward versus reverse mode as row- or
column-wise Jacobian fill.

### Ch. 2 Linear Regression (8 / 9; 12)

Training and polynomial plots hide their claims (above); ridge contours are
circles where the caption says ellipses. **Add:** polynomial fits of degree
1, 3, 19; least squares as projection; GD against SGD paths with the noise
ball; the outlier demo as a plot; shrinkage factors and $\mathrm{df}(\lambda)$.

### Ch. 3 Linear Classification (9 / 8; 10)

`clf-loss-accuracy` is 790 pt wide with 7 px labels; galleries use false
color and do not say which label is true. **Add:** the 2-D covariate-shift
experiment (the prose says "so the shift is visible"; nothing is plotted);
the 2×2 confusion table; binary losses against the margin; a reliability
diagram.

### Ch. 4 Multilayer Perceptron (13 / 14; 12)

Six activation plots should be one; `backprop-graph` is 847 px wide; Kaggle
screenshots are dated. **Add:** depth folds the input (a printed table now);
early stopping on noisy labels; dropout against no dropout (there is no
baseline run); the $e=(a+b)c$ graph with forward and backward values.

### Ch. 5 Computation (15 / 2; 12)

**Add:** float-format range bars with a bf16 rounding example; the real
module tree with registry names; gradient magnitudes against the fp16
underflow line before and after loss scaling; activation spread against
depth for three initializations (a printed table now).

### Ch. 6 CNNs (14 / 1; 7)

Two legacy LeNet SVGs and three rasters beside generated figures. **Add:**
the why-conv triptych (dense $10^{12}$, shared $4\times10^6$, local
$(2\Delta+1)^2$ weights); receptive-field towers for stride and dilation;
im2col; dense, grouped, and depthwise connectivity as block structure.

### Ch. 7 Modern ConvNets (17 / 14; 20)

Five panoramas too wide to read; four comparisons split across auto-scaled
plots; a RegNet raster. **Add:** the BN, LN, GN, IN axes on an
$(N,C,H\!\cdot\!W)$ cube (the batch-norm section has no figure); LeNet with
and without BN on shared axes; RepVGG kernel algebra; the ConvNeXt roadmap as
a cumulative step chart.

### Ch. 8 Sequence Models (11 / 14; 13)

The BPTT figures show no backward direction. **Add:** greedy against the
better path as probability grids; one distribution at three temperatures;
rollout error against $k$; shapes from token ids to logits.

### Ch. 9 Optimization (8 / 78; ~60)

Sixteen near-identical airfoil curves, twelve lr-scheduler boards on
different scales, six trajectory plots with different limits. **Fix:** one
overlay per comparison and shared-limit small multiples; one color per
method. **Add:** Newton–Schulz maps; steps against examples (critical batch);
the Nesterov look-ahead; preconditioning.

### Ch. 10 Attention (10 / 19; 25)

`show_heatmaps` defaults throughout; the residual-stream arrow misses its
target. **Add:** a shape diagram for $\mathrm{softmax}(QK^\top/\sqrt d)V$ and
the head split; linear attention by associativity; mask algebra; memory
traffic of a naive against a tiled kernel.

### Ch. 11 Transformers (8 / 11; 16)

`vit.svg` is from the oldest generation and contradicts the chapter's color
roles. **Add:** a whole-model GPT diagram (the chapter has none); a decoder
block with cross-attention; Chinchilla contours with iso-compute lines;
prefill against decode on a roofline.

### Ch. 12 State Space Models (14 / 28; ~34)

Three refuted claims (above). Fifteen trainer boards should be three
comparisons. **Add, for the rewrite:** discretization as a map from the left
half-plane to the unit disk; delta-rule geometry with the eigenvalue along
$k$; the effective kernel of a trained selective model against an LTI one
(the evidence the Δ boxplot fails to give); matrix-state memory anatomy;
chunked data flow.

### Ch. 13 Computational Performance (18 / 5; 14)

**Add:** a roofline with measured points (the most useful figure in the
chapter); fusion before and after with bytes counted; checkpointing memory
traces; data-parallel prediction against measurement.

### Ch. 14 Reinforcement Learning (14 / 15; 20)

`show_grid` is unreadable (green on black, arrows through letters, no
indices). Five static/notebook duplicates. **Add:** the trajectory likelihood
as a chain (model-free gradient); shortest path against the optimal policy
(the prose says "shown below"; only prints follow); termination against
truncation timelines.

### Ch. 15 Deep RL (12 / 19; 28)

`data-rules` is the worst figure in the two chapters; SAC has no figure.
**Add:** GRPO dataflow; the $N\times T$ rollout with GAE accumulated right to
left; SAC squashing and its Jacobian; soft policy improvement.

### Ch. 16 GANs (4 / 21; 18)

Colors for real and generated data change between plots. **Add:** DCGAN and
backbone architecture diagrams (two architectures, no picture); optimal
critics on separated supports (the chapter's central argument, prose only);
saturating against non-saturating weights; the three conditioning
mechanisms.

### Ch. 17 Diffusion Models (10 / 25; 24)

Nine unlabeled image grids; three frames for one running example. **Add:**
the density of $x_t$ over $(t, x)$ with forward and reverse paths; DDIM step
geometry; the noise schedule; one steps-against-error plot replacing three
printed tables.

### Ch. 24 Linear Algebra (22 / 2; 19)

Four content errors (above). **Add:** Strang's big picture of the four
subspaces with $A^+$; $Ax \parallel x$ for a non-symmetric example; powers
of a matrix (iterates, spiral, Jordan drift); least squares as projection in
$\mathbb{R}^3$.

### Ch. 25 Calculus (29 / 5; 23)

19 figures below 12 pt. **Add:** steepest descent under different norms
(the prose refers to this picture, which does not exist); Newton against
gradient step; implicit differentiation.

### Ch. 26 Optimization (17 / 6; 18)

Four content errors (above). **Add:** floating-point bit layouts; edge of
stability as a plot; projected gradient descent; L1 against L2 balls.

### Ch. 27 Probability and Statistics (30 / 12; 29)

**Add:** Chernoff majorants and the resulting tails (the concentration
section has no static figure); the thin shell of high-dimensional Gaussians;
change of variables; a reliability diagram.

### Ch. 28 Information Theory (15 / 5; 13)

The smallest fonts of any generator. **Add:** arithmetic coding as nested
intervals; the typical set; an MMD witness; mutual information against
correlation.

### Ch. 29 Dynamics (19 / 14; 26)

Seven duplicate sets with ch. 17 and within the chapter. **Add:** one-step
solver geometry (Euler and RK4 probes); the adjoint timeline; divergence as
volume change; reflow before and after.

## Order of Work

1. **Content errors** in the table above. They are cheap and they mislead.
2. **Helper fixes** (`d2l.plot`, `ProgressBoard`, `show_heatmaps`,
   `show_grid`, `plot_curves`), then re-capture. Basics notebooks need all
   four frameworks.
3. **Generator fixes** (unitless sizes, 13 pt floor, column width, tokens,
   the axis-color decision), then `make figures`.
4. **Merge duplicates and split comparisons** into overlays.
5. **New figures,** chapter by chapter, together with the text revision of
   that chapter.
