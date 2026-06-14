# Slide outline — §5.2 `mlp-implementation.md` (Implementation of MLPs)

**Status: ready to design.** No staleness. No per-framework gaps — every cited
cell exists in all four frameworks. Training now runs **30 epochs**
(`d2l.Trainer(max_epochs=30)`, line 207) — reflect the 30-epoch loss/accuracy
curves, not the old 10.

This is the *build-it-twice* deck (from scratch → concise), both training on
Fashion-MNIST. The existing block (lines 354–489) is solid and already has good
bones (an ASCII pipeline, a parameter-count slide). The north-star upgrade:
replace the ASCII data-flow box with the chapter's real architecture figure
`mdl-mlp-arch.svg`, and pair the from-scratch vs concise code as clean In/Out
cards. Output cells are training-curve SVGs.

## Notebook cells (all 4 fw)

| cell id | kind | tabs | shows |
|---|---|---|---|
| `mlp-implementation-implementation-of-multilayer-perceptrons` | text (no out) | all 4 | imports |
| `mlp-implementation-initializing-model-parameters` | text (no out) | all 4 | `MLPScratch.__init__` — W1,b1,W2,b2 |
| `mlp-implementation-model-1` | text (no out) | all 4 | hand-rolled `relu(X)` |
| `mlp-implementation-model-2` | text (no out) | untagged (all) | `forward` (flatten → relu(XW1+b1) → ·W2+b2) |
| `mlp-implementation-training` | **svg plot** | untagged (all) | from-scratch train: 30-epoch loss + train/val acc curves |
| `mlp-implementation-model-2-2` | text (no out) | all 4 | concise `MLP` (`Sequential(Flatten,Linear,ReLU,Linear)`) |
| `mlp-implementation-training-2` | **svg plot** | untagged (all) | concise train: same 30-epoch curves |

## Figures available

- `img/mdl-mlp-arch.svg` — matplotlib figure: the exact two-layer MLP of this
  section, batched input → flatten to 784 → affine to 256 → ReLU → affine to 10
  logits. **This is the slide-1 hero** (replaces the ASCII box). Reuse as
  `![](../img/mdl-mlp-arch.svg)`.
- No `@fig:` engine module. Same decision as §5.1 — recommend reusing
  `mdl-mlp-arch.svg`; an engine `diagrams/mlp.mjs` is optional.

## Proposed slide list

1. **Cover** — `.cover`. Kicker "§5.2". Title: *Implementing an MLP* /
   "the same model two ways — by hand and with `nn.Sequential`."

2. **The model we're building** (`title`, `.cols .vc`) — left `.col .fig .big`:
   `![](../img/mdl-mlp-arch.svg)`. Right col: one-line spec — 784 → 256 (ReLU) →
   10, on Fashion-MNIST. Caption: two affine layers, one ReLU; trained
   end-to-end. (Replaces the ASCII pipeline with the real figure.)

3. **Why 256 hidden units is reasonable** (`title`) — keep the existing slide's
   three bullets, tightened: ~200k params (enough to fit, fast to train);
   powers-of-2 are a kernel habit not magic; single layer because FashionMNIST
   is easy. `.d2l-note`: these are *hyperparameters* — set by hand, not learned.

4. **Divider 01 — From Scratch.**

5. **Setup** (`title`) — `@mlp-implementation-implementation-of-multilayer-perceptrons`
   (imports, no output). Keep minimal.

6. **Parameters by hand** (`title`, `.cols .vc`) — right `.col`:
   `@mlp-implementation-initializing-model-parameters`. Left col: the shapes +
   count, as the existing slide has them —
   $\mathbf{W}^{(1)}\in\mathbb R^{784\times256},\ \mathbf{W}^{(2)}\in\mathbb R^{256\times10}$,
   total $784\cdot256+256+256\cdot10+10=203{,}530$. `.d2l-note`: small Gaussian
   $\mathcal N(0,\sigma^2)$ for weights, zero for biases (why σ matters → §5.4).

7. **ReLU + forward pass** (`title`) — `@mlp-implementation-model-1` (our own
   `max(X,0)`). `. . .` fragment → the forward math
   $\mathbf{H}=\operatorname{ReLU}(\mathbf{X}\mathbf{W}^{(1)}+\mathbf{b}^{(1)})$,
   $\mathbf{O}=\mathbf{H}\mathbf{W}^{(2)}+\mathbf{b}^{(2)}$ → then
   `@mlp-implementation-model-2`. Caption: pixels flattened to a 784-vector
   first; spatial structure ignored (CNNs fix this next chapter).

8. **Train it** (`title`) — `@mlp-implementation-training`. This is an
   `@`-with-output cell: the **30-epoch** training/validation loss + accuracy
   curves. Caption: same `Trainer`, same loaders, same cross-entropy as softmax
   regression — *only the model changed*. Note the small but real gain over
   plain softmax regression. Consider `output-lines` only if a log prints; this
   cell's output is the plot.

9. **Divider 02 — The Concise Version.**

10. **Same model, less bookkeeping** (`title`) — `@mlp-implementation-model-2-2`.
    Caption: `Sequential(Flatten, Linear, ReLU, Linear)` — lazy layers infer
    input shape; `ReLU` is built in. `.d2l-note`: both versions produce the
    *same* model; the framework just removes parameter plumbing.

11. **Same training, same accuracy** (`title`) — `@mlp-implementation-training-2`
    (30-epoch curves again). Caption: identical convergence — built-in `Linear`
    + `ReLU` compute exactly what the from-scratch version did, just easier to
    read.
    *(Optional: instead of two separate training slides (#8, #11), a single
    "scratch vs concise — identical curves" slide showing both plots side by
    side could be stronger; but two slides keep each In/Out card uncramped.
    Author's call.)*

12. **What's left to learn** (`title`) — the four open questions as a tight
    list, each pointing at its deck: initialization (§5.4), generalization
    (§5.5), regularization/dropout (§5.6), backprop (§5.3). Sets up the rest of
    the chapter.

13. **Recap** (`title`) — MLP = softmax classifier + hidden layer & nonlinearity;
    from scratch = 4 tensors + hand ReLU + explicit matmuls; concise =
    `Sequential(...)`, same model; hyperparameters live outside the model class;
    beats softmax regression on FashionMNIST by a small real margin — first
    taste of "depth helps."

## Per-framework notes

- **No gaps, no scoped slides needed.** Every cited cell has all four `#@tab`
  variants (or is untagged/all). The code/output differences across frameworks
  (`nn.Parameter` vs `tf.Variable` vs flax `self.param` vs mxnet
  `np.random.randn`; `Sequential` flavours) are pure swaps the `#@tab` mechanism
  already resolves per deck — captions stay neutral.
- Training cells (`-training`, `-training-2`) are untagged → one shared cell;
  the injected SVG curve is per-framework but the slide source is identical.
- Confirm the 30-epoch curves are what the committed `outputs/` store holds
  before final render (training config was recently changed from 10→30); the
  outline assumes 30.

## Must port before non-pytorch decks
- **None.**
