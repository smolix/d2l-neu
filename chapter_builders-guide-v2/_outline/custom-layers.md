# Custom Layers and Functions
:label:`sec_custom_layers_v2`

> **Role.** The escape hatch: what to do when the framework's layer zoo does
> not have what you need. The lesson kept from the current section is the
> *composability guarantee* — a correctly written custom layer gets
> parameter tracking, serialization, and device movement for free. The
> examples are upgraded from toys (`CenteredLayer`, a hand-rolled `Linear`)
> to layers the reader will actually reuse (RMSNorm), and a subsection on
> custom autograd is added.

## Layers without Parameters **[KEPT]**

*Topics.* The minimal custom layer: subclass, write `forward`, done.
`CenteredLayer` survives — it is the right five-line first example. The
guarantee: it composes with `Sequential`, and the framework neither knows
nor cares that it is user code.

*Code (PyTorch).* `CenteredLayer` as in the current section; sandwich it in
a `Sequential`, verify the output mean is ~0 (floating-point roundoff
note kept).

## Layers with Parameters: RMSNorm **[MOD]**

*Topics.* Replaces the hand-rolled `MyDense` (which duplicates `nn.Linear`
and teaches nothing new) with **RMSNorm**: normalize by the root-mean-square
of activations, scale by a learned gain — five lines, a real layer that did
not exist when the current chapter was written, and the normalization used
by essentially every modern LLM. Teaches the same mechanics (`nn.Parameter`
creation, shape handling) on an object with a future: the transformer
chapters can import this exact layer. Note the contrast with LayerNorm
(no mean subtraction, no bias) and why that matters is deferred to the
normalization discussion in Chapter 8/11.

*Code (PyTorch).*

```python
class RMSNorm(nn.Module):  # NOT #@save — see framework-native note below
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, X):
        rms = X.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return self.gain * X * rms
```

Verify: parameters registered, works inside `Sequential`, survives a
`state_dict()` round trip, moves with `.to(device)` — the composability
guarantee demonstrated on all four axes, once, explicitly.

Close by **checking ours against the framework's own**: `nn.RMSNorm`
(torch ≥2.4, verified in 2.11), `flax.linen.RMSNorm`, and
`keras.layers.RMSNormalization` all exist (verified in the repo venvs) —
compare outputs, then state the rule: *build it to understand it, use the
native in production*. Downstream transformer chapters use the natives
(mxnet, which has none, hand-rolls this five-liner locally). Consequently
this is **not** a `#@save` — no new d2l symbol.

## State That Is Not a Parameter, Revisited **[NEW, short]**

*Topics.* A custom layer that needs a precomputed table uses
`register_buffer` (:numref:`sec_parameters_v2`): worked micro-example of a
layer with a fixed positional table or a causal mask as a buffer. One
paragraph + one code cell; the point is to close the loop between the two
sections before the transformer chapters rely on it.

## Custom Gradients: `autograd.Function` **[NEW]**

*Topics.* When `forward` alone is not enough: operations with
non-differentiable pieces or hand-optimized backward passes. The
**straight-through estimator** as the worked example — round to the nearest
integer in `forward`, pretend it was the identity in `backward` — three
honest lines each, and the standard trick behind quantization-aware
training and discrete latents (forward pointer to future generative
content). Scope note: writing fused/custom *kernels* is out of scope; this
is only about overriding the chain rule.

*Code (PyTorch).* `class RoundSTE(torch.autograd.Function)` with
`forward`/`backward` static methods; gradcheck-style verification that the
gradient flows through the rounding.

## Summary and Exercises

*Exercises (sketch).* (1) Add an optional bias to RMSNorm; verify the
state dict grows accordingly. (2) Write a `Dropout` layer from scratch that
behaves differently under `.train()`/`.eval()` — where does the mode flag
live in the module tree? (3) Implement a clamp-with-gradient
(`autograd.Function` whose backward passes gradients only inside the clamp
range); compare with `torch.clamp`'s native behavior. (4) *(Kept from
current)* A layer computing the FFT of its input — no parameters, why is it
still worth making a layer?

> **Downstream constraints.** None: no downstream file depends on this
> section's specific artifacts (the generic subclassing idiom is covered by
> 6.1). `RMSNorm` is a candidate `#@save` for reuse by future transformer
> chapters — decide at rewrite time.

## Framework Coverage

- **JAX** — full coverage. RMSNorm via `self.param(...)` (mechanics
  already proven by the current tab's `MyDense`). Custom gradients: teach
  `jax.custom_vjp` as the general mechanism (verified STE) *and* the
  three-line `x + stop_gradient(round(x) - x)` shortcut (verified,
  identical gradient) — the shortcut is the better STE example, the
  general form covers genuinely different backward formulas.
- **TensorFlow** — full coverage, all verified live: RMSNorm via
  `add_weight` in `build()` passes the four-axis composability check;
  `@tf.custom_gradient` STE works in five lines.
- **MXNet** — full coverage: `gluon.Parameter` pattern already taught;
  `mx.autograd.Function` genuinely exists and mirrors
  `torch.autograd.Function` (verified in source) with one gotcha — each
  Function *instance* is single-use (source-enforced), so instantiate per
  call. [UNVERIFIED on GPU box: `autograd.Function` inside a training
  loop.]
