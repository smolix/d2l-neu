# Modules and Model Construction
:label:`sec_model_construction_v2`

> **Role.** The foundational section: every model from Chapter 7 onward is
> built with what is taught here. Keeps the recursive module-hierarchy mental
> model from the current section, while noting that shared children form an
> object graph rather than a tree; replaces the toy
> examples with ones that point forward to the architectures the reader will
> actually build, folds in lazy initialization (currently a standalone
> section), and adds the config-driven assembly pattern that every real 2026
> codebase uses.

## The Module Abstraction **[KEPT]**

*Topics.* Layers, blocks, and whole models are the same kind of object; a
module owns (i) parameters, (ii) child modules, (iii) a `forward`
computation. The usual tree-shaped hierarchy: a model is a recursive
composition, while sharing introduces aliases in the object graph, and
everything downstream, including parameter traversal, serialization, and
device movement, is a graph traversal that handles shared children. Why coarse *blocks* (not layers) are the unit of
design, with the 2026 example in front: a Transformer is a stack of dozens
of identical blocks.

*Code (PyTorch).* Instantiate an MLP via `nn.Sequential`, call it, peek at
`net._modules`; then re-implement the same model as an `nn.Module` subclass
with an explicit `__init__`/`forward`. Show that calling the module runs
`__call__` → `forward` (and mention hooks exist in this gap, forward
pointer to :numref:`sec_repro_v2`).

```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.LazyLinear(256)
        self.out = nn.LazyLinear(10)

    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))
```

## Sequential and Friends: Containers **[MOD]**

*Topics.* `nn.Sequential` re-implemented in five lines (`MySequential`,
kept from the current section — it demystifies the container). **New:**
`nn.ModuleList` and `nn.ModuleDict`; the classic bug of storing child
modules in a plain Python list (parameters silently invisible to
`.parameters()`, the optimizer, and the checkpoint) — currently only hinted
at in an exercise, promoted to worked example because everyone hits it.

*Code (PyTorch).* `MySequential` from scratch; a module holding layers in a
Python list vs `nn.ModuleList`, comparing `sum(p.numel() for p in
net.parameters())` for both — the broken variant "trains" with zero
parameters.

## Forward Is Just Python **[MOD]**

*Topics.* Control flow, loops, and arbitrary computation in `forward`.
Replaces the current `FixedHiddenMLP` ("a weird function nobody would
compute") with examples that preview real patterns: (i) a block with a
**residual connection** (`return X + self.body(X)`), the single most
important wiring idiom of the next twenty chapters; (ii) a non-trainable
constant tensor in `forward` — reframed as "some state is not a parameter",
forward pointer to buffers in :numref:`sec_parameters_v2`.

*Code (PyTorch).* `ResidualBlock(nn.Module)`; call it, verify shape
preservation is required and why.

## Lazy Initialization: Shapes from Data **[MOD — folded from current 6.4]**
:label:`sec_lazy_init`

> **Lib constraint.** This subsection MUST keep the label `sec_lazy_init`
> and carry over the PyTorch `Module.apply_init` `#@save`
> **byte-identical**: shards stamp symbols with `Defined in
> :numref:<label>`, and `apply_init` is used by 8+ downstream files
> (alexnet, vgg, nin, googlenet, resnet, batch-norm, lenet, hyperopt-api —
> the `model.apply_init([X], init_cnn)` idiom). Changing either the block
> or the label invalidates all of their committed outputs.

*Topics.* Declaring output widths only and letting input widths be inferred
at first call (`nn.LazyLinear`, `nn.LazyConv2d`). Why the book builds models
this way from Chapter 7 on (it removes shape bookkeeping from pedagogy);
when explicit shapes are better (config-driven code, see next subsection);
the one rule: registered placeholders are not materialized arrays until the
first forward pass, so shape/value inspection and shape-dependent
initialization follow a dry run (optimizers may be constructed earlier).
Contrast: NNX linear layers take both widths and create parameters in the
constructor from an explicit RNG stream.

*Code (PyTorch).* Build with `nn.LazyLinear`, show `UninitializedParameter`
before and a real weight after a dry run `net(X)`.

## Building from a Config **[NEW]**

*Topics.* Real models are not built by hand-writing literals into
`__init__`; they are assembled from a configuration object (a `dataclass`)
that records widths, depths, and switches. One config → one architecture;
the config travels with the checkpoint (forward pointer to
:numref:`sec_read_write_v2`); stacking `num_blocks` identical residual
blocks from the config — a common builder pattern in later Transformer
implementations. The config records variable choices; topology stays in code.

*Code (PyTorch).*

```python
@dataclass
class MLPConfig:
    d_in: int = 784
    d_hidden: int = 256
    num_blocks: int = 4
    d_out: int = 10

def build(cfg: MLPConfig) -> nn.Module:
    blocks = [ResidualBlock(cfg.d_hidden) for _ in range(cfg.num_blocks)]
    return nn.Sequential(nn.Linear(cfg.d_in, cfg.d_hidden),
                         *blocks, nn.Linear(cfg.d_hidden, cfg.d_out))
```

Print the module tree; change one config field and rebuild — the point is
that architecture becomes *data*.

## Summary and Exercises

*Exercises (sketch).* (1) Store submodules in a plain list, diagnose what
breaks and why. (2) Implement a `ParallelBlock` that runs two child modules
on the same input and concatenates outputs (kept from current). (3) Extend
`MLPConfig` with an activation-function switch; discuss what belongs in a
config vs in code. (4) Why does a residual block require matching input and
output widths, and what are the two standard fixes? *(Answers preview
Chapter 8's ResNet.)*

> **Downstream constraints.** `nn.Module` subclassing + `nn.Sequential` must
> be fully taught here (37 downstream files subclass, 29 use `Sequential`).
> Lazy layers must be taught before Chapter 7 (19 downstream files build
> with `nn.Lazy*`). Both preserved. Label `sec_model_construction` and
> `subsec_model-construction-sequential` are cited from
> :numref:`sec_oo-design` — keep labels on promotion.

## Framework Coverage

- **JAX** — full coverage via NNX. Graph nodes belong in `nnx.List` or
  `nnx.Dict`; a plain Python container of modules is rejected rather than
  silently hidden. Config-driven assembly stores an ordinary dataclass as
  architecture data on an `nnx.Module`. Linear layers take explicit widths
  and create parameters in the constructor.
- **TensorFlow** — full coverage. Keras 3 auto-tracks plain lists/dicts
  (verified) — the container subsection carries an explicit "TF differs
  here" note instead of the bug demo. `build()`/first-call *is* Keras's
  always-on lazy init. Config via `build(cfg)`; one sentence on
  `get_config()` as the native idiom.
- **MXNet** — full coverage, and two wins: deferred init is native and
  best-in-class (`in_units=0`, no special `Lazy*` class), and the
  plain-list bug exists but `collect_params()` *warns by name* (verified in
  wheel source) — better diagnostics than PyTorch's silent empty list. No
  `ModuleList`/`ModuleDict` classes: ALT via named attributes /
  `register_child`.
- **Net effect**: the container subsection becomes a genuine four-way
  comparative lesson (silent failure / warned / rejected / auto-tracked) — stronger
  than the PyTorch-only version.
