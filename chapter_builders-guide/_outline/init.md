# Initialization
:label:`sec_init_v2`

> **Role.** The API companion to :numref:`sec_numerical_stability` (which
> owns the theory: symmetry breaking, vanishing/exploding activations,
> Xavier/He derivations). This section teaches the *mechanics* — how to
> apply, override, and design initialization in code — and modernizes the
> menu of schemes to what transformer-era code actually uses. Slightly
> shorter than the current section; its `net.apply(fn)` pattern is the
> load-bearing part (it is re-instantiated downstream as `init_cnn` and
> `init_seq2seq`).

## Default Initialization and When to Override It **[MOD]**

*Topics.* What you get for free (PyTorch: Kaiming-uniform for `Linear`/
`Conv`), and the checklist of situations that warrant overriding: deep
unnormalized nets, custom layers with hand-created parameters, transfer
from a different scale/dtype regime, and reproducing a paper. Kept from the
current section, tightened; the per-framework default table survives as a
reference.

*Code (PyTorch).* Inspect a fresh `nn.Linear`'s weight statistics; confirm
against the documented default.

## Applying Initializers **[KEPT]**

*Topics.* The `net.apply(fn)` idiom: a function receiving each module,
dispatching on `type(m)`, mutating in place with `nn.init.*_` — the pattern
downstream chapters wrap as `init_cnn` (Chapter 7) and `init_seq2seq`
(Chapter 9). Applying to a submodule only (`net[0].apply(...)`). In-place
semantics and the trailing-underscore convention.

*Code (PyTorch).* `init_normal` / `init_constant` / `init_xavier` variants
applied whole-net and per-block, as in the current section but on the
config-built model from :numref:`sec_model_construction_v2` (one model
reused across the chapter instead of three throwaways).

## Modern Schemes: Truncation, Depth, and Zeros **[NEW]**

*Topics.* Three ideas that postdate the current section's menu and are now
standard: (i) `trunc_normal_` — why heavy tails are clipped in ViT/BERT
lineage init; (ii) **depth-aware residual scaling** — in an $N$-block
residual network, scale the *last* layer of each block by $1/\sqrt{N}$
(GPT-2) so the residual stream's variance stays O(1) regardless of depth;
(iii) **zero-init of the final layer / gain** — start each block as an
identity-plus-nothing map so early training is stable (zero-init gamma,
FixUp lineage). Each gets two sentences of *why* anchored to the variance
argument of :numref:`sec_numerical_stability`, not a re-derivation.

*Code (PyTorch).* Initialize the residual stack three ways (default,
$1/\sqrt{N}$-scaled, zero-init last layer); one forward pass per variant
printing the output std as depth grows — a three-line empirical
demonstration that the scaled variants keep activations O(1). *(Teaching
code that computes something; no drawing.)*

## Custom Initializers **[MOD]**

*Topics.* Writing an initializer the menu does not have, kept from the
current section (its "weird mixture" example is fine pedagogy), plus direct
tensor surgery (`with torch.no_grad(): net[0].weight += 1`) and the caveat
about doing this after lazy materialization.

*Code (PyTorch).* Current section's `my_init` essentially as-is.

## Summary and Exercises

*Exercises (sketch).* (1) Measure activation std at every block for
default vs depth-scaled init at $N=2,8,32$; plot std vs depth. (2) Zero-init
*all* layers instead of just the last per block — what breaks, and how does
it relate to symmetry breaking in :numref:`sec_numerical_stability`?
(3) Write an initializer that loads weights from a dict by name — you have
just re-invented part of `load_state_dict` (forward pointer to
:numref:`sec_read_write_v2`).

> **Downstream constraints.** `init_cnn`/`init_seq2seq` are *defined
> downstream* (lenet.md, seq2seq.md), so this section's concrete examples
> are freely editable; only the `apply(fn)` pattern itself must be taught
> before Chapter 7. Preserved. Current file has no `:label:` (nothing can
> cite it) — labels here are new.

## Framework Coverage

- **JAX** — the one structural divergence in this section: there is no
  `net.apply(fn)` mutate-in-place walker (params are immutable pytrees).
  The JAX idiom is *construct with the right `kernel_init=` argument* —
  and modern schemes + custom initializers unify into one pattern (any
  `(key, shape, dtype) → array` function; `truncated_normal` and
  `variance_scaling` verified). The prose should teach this as a genuine
  idiom difference, not a translation.
- **TensorFlow** — full coverage: `initializers.TruncatedNormal` verified;
  depth-scaled/zero-init as custom `Initializer` subclasses (verified);
  post-hoc surgery via `kernel.assign(...)` (already in the current tab).
  The apply-walk becomes iterate-`model.layers`-and-`isinstance` (ALT).
- **MXNet** — full coverage with one gap: **no `TruncatedNormal`
  initializer ships** (verified in source) — ALT via the custom
  `Initializer` subclass the current tab already teaches (`MyInit`).
  Default init confirmed `Uniform(0.07)` — worth a row in the defaults
  table. Per-layer `.initialize()` covers the apply-walk role.
