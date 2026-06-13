# Review — Building Blocks (§6.1 model-construction.md + §6.5 custom-layer.md)

*Landscape review. Files: `chapter_builders-guide/model-construction.md` ("Layers
and Modules"), `chapter_builders-guide/custom-layer.md` ("Custom Layers"). Primary
lens PyTorch 2.x; book is 4-framework (pytorch/jax/tf/mxnet). This is a
**review**, not a change spec — recommendations are directional.*

---

## Executive verdict

These two sections are the chapter's spine: they teach the single most important
mental model in practical deep learning — **everything is an `nn.Module`, and
modules compose recursively** — and they teach it *correctly*. The code runs, the
outputs are sensible, the `MySequential` "build it yourself" demo is genuinely
good pedagogy, and the `FixedHiddenMLP`/`NestMLP` examples land the "`forward` is
just Python" and "modules nest to any depth" points. The slides (out of scope per
the guide, but I read them) are noticeably more modern and better-framed than the
prose they accompany — they already say things the prose doesn't (Transformers,
`state_dict`, parameter sharing as a named concept, `register_buffer` rationale).

**But measured against the "assignable at Stanford/MIT/CMU, above most courses"
bar, the prose is a faithful 2021 treatment that has not moved with the field.**
Three gaps keep it from best-in-class:

1. **The module-vs-everything-else map is incomplete.** A 2026 student learning
   "how to build models" must know `ModuleList`/`ModuleDict` and the
   *register-or-lose-your-parameters* failure mode. The chapter gestures at it
   (Exercise 1 asks "what goes wrong if you use a plain Python list?") but never
   *teaches the answer*, and never introduces the containers that solve it. This
   is the highest-value gap — it is in-lane, it is short, and every modern
   resource covers it.
2. **The "what is a module" framing is dated in two concrete ways:** it tells the
   reader a module "must possess a backpropagation method" (autograd does not work
   that way), and it leans entirely on `LazyLinear`, which PyTorch's own docs flag
   as **experimental** and which the official tutorials do **not** use.
3. **It stops one inch short of the modern payoff.** The `FixedHiddenMLP`
   while-loop is, in 2026, the canonical example of a `torch.compile` **graph
   break** — a one-paragraph forward-pointer would turn a quaint "look, control
   flow works" demo into a teachable moment about eager-vs-compiled execution. And
   the custom-layer section never mentions `autograd.Function`, the actual answer
   to "what if my op isn't differentiable by composition" — the natural ceiling of
   a custom-layers section.

**Grade: B / B+.** Correct, clean, assignable *today* in most courses. To be the
treatment a top-5 program reaches for first, it needs the container story, a
de-dating pass on the framing and APIs, and two forward-pointers (compile,
`autograd.Function`). None of these is large; all are in-lane.

Calibration note: I checked the current **official d2l.ai** version of
model-construction. It is structurally identical (same `LazyLinear`, same
`add_module`, **no** `ModuleList`/`ModuleDict`, **no** hooks, **no**
`named_modules`, **no** `torch.compile`). So this repo is not behind upstream —
but upstream itself is a 2021 artifact here, and "match upstream" is below the bar
this project set. The opportunity to *exceed* is real and unclaimed.

---

## 1. Current-state assessment

### 1.1 `model-construction.md` — "Layers and Modules" (§6.1)

**What it teaches.** The neuron → layer → module abstraction ladder; that a module
is a Python class with `__init__` + `forward`; `nn.Sequential` as the linear-chain
container; a hand-written `MLP` subclass; a hand-written `MySequential` to demystify
the container; and `FixedHiddenMLP`/`NestMLP`/`chimera` to show control flow,
constant buffers, parameter reuse, and recursive nesting. Structure: 4 `##`
sections (Custom Module / Sequential Module / Executing Code in Forward / Summary)
+ exercises. The spine is logical and the build-up is good.

**Strong, keep:**
- The **`MySequential` reimplementation** (lines 408–534) is excellent pedagogy and
  is exactly how CMU 11-785 and fast.ai teach it: "what does the framework actually
  do? almost nothing — here it is in 4 lines." Keep.
- **`FixedHiddenMLP`** (lines 568–615) makes "`forward` is ordinary Python — loops,
  conditionals, reused layers, fixed tensors" concrete. Keep the *idea*; modernize
  the framing (below).
- The PyTorch tab's use of **`register_buffer`** for `rand_weight` (line 602) is
  *correct and modern* and better than upstream d2l (which historically used a bare
  tensor or `requires_grad=False`). The inline comment ("part of module state and
  moves with `.to(device)`") is exactly right. Keep.
- Opening neuron→layer→module motivation (lines 9–78) is solid; the ResNet-152 /
  "repeating patterns" hook is good and still current.

**Dated / awkward / wrong:**

- **"a module must possess a backpropagation method, for purposes of calculating
  gradients" (lines 89–90).** This is the wrong mental model in 2026 (and was
  always shaky). `nn.Module` has **no** `backward` method; gradients come from
  autograd recording the ops executed in `forward`. The next sentence half-walks it
  back ("due to some behind-the-scenes magic … we only need to worry about
  parameters and the forward propagation method"), but the damage is done — a
  careful reader is told modules *have* a backward method, then told to ignore it.
  The five-item "what every module must do" list (lines 245–250) repeats this:
  item 3 is "Calculate the gradient of its output with respect to its input, which
  can be accessed via its backpropagation method." A module does not own this; the
  tape does. **Reframe:** a module defines `forward`; autograd supplies the backward
  pass automatically by differentiating through it. (This matters more here than
  almost anywhere, because the *whole point* of the section is "what is the minimal
  contract of a module," and the stated contract is wrong.)

- **Total reliance on `LazyLinear`** (every PyTorch cell: lines 148, 285–286, 512,
  602–603, 751–753). `LazyLinear` is convenient for a book that wants to skip
  `in_features`, but PyTorch's own docs carry a standing warning that **lazy
  modules are an experimental feature under active development whose API is likely
  to change** (`torch.nn.LazyLinear`, see Resources), and the official
  "Build the Neural Network" / "Learning PyTorch with Examples" tutorials use
  explicit `nn.Linear(28*28, 512)`. Teaching beginners that the *normal* way to
  write a linear layer is the experimental lazy variant is a currency problem: it
  is non-idiomatic relative to every external reference a student will read next,
  and it quietly couples §6.1 to §6.4 (lazy-init) before lazy init is explained.
  This is a judgment call for the authors — lazy init *is* a real feature the book
  teaches in §6.4 — but the *default* example layer arguably should be explicit, with
  `LazyLinear` introduced as the lazy-init convenience it is, in or pointing to §6.4.

- **`MySequential` PyTorch tab uses `add_module` + `self.children()`** (lines
  434–443). This *works* and the explanatory note (lines 488–493) is fine, but it
  is slightly archaic: modern PyTorch code that needs an ordered, registered list of
  submodules reaches for **`nn.ModuleList`** or **`nn.ModuleDict`**, which is the
  more transferable lesson. `add_module` is legitimate for the "build Sequential
  from scratch" demo (it shows the registration primitive), but the section never
  then says "in real code you'd use `ModuleList`" — so the student leaves without
  the tool they'll actually use.

- **Exercise 1 (line 812)** — "What kinds of problems will occur if you change
  `MySequential` to store modules in a Python list?" — asks the single most
  important question in the section *and the section never answers it in the
  prose.* The answer (parameters in a plain list are not registered, so
  `.parameters()` misses them, the optimizer never updates them, `.to(device)` and
  `state_dict()` skip them) is *the* foundational gotcha of `nn.Module`. Leaving it
  as an unanswered exercise is a real miss; it should be taught in the body, with
  the exercise then probing an edge of it.

- **MXNet presented as co-equal.** Per project memory, Apache MXNet was archived by
  the ASF in 2023; the repo keeps it as a tab (a deliberate decision), but the
  prose treats it as a live, first-class framework (the `_children`/weakref
  discussion at 408–429 + 474–486 is more elaborate than the PyTorch one). Not this
  review's call to drop the tab, but flag: the *depth* ordering is inverted — the
  legacy framework gets the most detailed treatment of registration, the primary
  framework gets the least.

### 1.2 `custom-layer.md` — "Custom Layers" (§6.5)

**What it teaches.** Two flavors of custom layer: parameterless (`CenteredLayer`,
subtract the mean) and parameterized (`MyDense`, a hand-rolled `Linear` wrapping
`nn.Parameter`). Shows they compose into `Sequential` like built-ins. Structure: 2
`##` sections (Layers without Parameters / Layers with Parameters) + summary +
exercises. Short and clean.

**Strong, keep:**
- The **two-flavor split (stateless vs stateful)** is exactly the right pedagogical
  decomposition and matches how the official PyTorch custom-modules material frames
  it. Keep.
- **`MyDense` wraps weights in `nn.Parameter`** (line 205) and the prose explains
  the registration benefit (lines 169–174). Correct and central. Keep.
- The PyTorch tab's **fan-in init** (`torch.randn(...) / in_units**0.5`, line 205)
  is a nice touch — it quietly does the right thing (Kaiming-ish scaling) and the
  comment says so. Better than a bare `randn`. Keep.

**Dated / awkward / weak:**

- **No mention of `register_buffer` in the body.** The section teaches
  `nn.Parameter` (trainable state) but never its sibling, **buffers**
  (non-trainable state that still travels with the module: running stats, masks,
  positional tables). This is a glaring omission for a *custom layers* section — the
  buffer/parameter distinction is half the API. Telling: the **slides already cover
  it** (the "When to write a custom layer" and "Recap" slides explicitly teach
  `register_buffer`), and §6.1's `FixedHiddenMLP` *uses* it — but the custom-layer
  *prose* is silent. A student reading only the body learns half the state model.

- **No `autograd.Function`, not even as a pointer.** The section's own framing is
  "the escape hatch when the standard layer zoo doesn't cover what you need"
  (slides). But `nn.Module` is only the escape hatch for *new compositions of
  existing differentiable ops*. The deeper escape hatch — a genuinely **new
  differentiable primitive** with a hand-written backward (a non-standard
  nonlinearity, a custom CUDA kernel, a straight-through estimator) — is
  `torch.autograd.Function`. A custom-layers section that never names it has a
  conceptual hole exactly where its stated purpose points. One paragraph +
  forward-pointer would close it (full treatment belongs to the autograd material,
  not here).

- **`MyDense` bakes in ReLU** (line 210, `return F.relu(linear)`). Minor, but
  pedagogically muddy: a layer named like `Linear` that secretly applies ReLU
  conflates "linear layer" with "linear + activation." The upstream rationale was
  brevity, but it teaches a slightly wrong abstraction (real `nn.Linear` is affine
  only). Worth reconsidering — at minimum the prose should flag that the activation
  is baked in for demo compactness, not because a `Linear` should do this.

- **Cross-framework drift in the `__init__`/`setup` story** (see §3): the four
  tabs define a parameterless layer four visibly different ways, more divergence
  than the frameworks require, which obscures the (identical) underlying idea.

- **Exercises are thin.** Two exercises (tensor-reduction layer; "leading half of
  the Fourier coefficients"). The Fourier one is cute but underspecified and not
  obviously *teaching* the module API. Neither exercises the buffer/parameter
  distinction, `state_dict` round-tripping of a custom layer, or shape inference —
  the things a custom-layer problem set at a top program would drill.

---

## 2. Modernization gaps (currency, 2021 → 2026)

Ordered by value. Each: *what's dated / missing*, *why it matters in 2026*,
*direction* (not a line edit).

**G1 — The container vocabulary is missing (`ModuleList`/`ModuleDict`) and the
"unregistered list" failure mode is never taught.** *Why it matters:* this is the
#1 beginner bug in PyTorch ("my model won't train / has no parameters") and the
first thing every modern guide teaches after `Sequential`. The canonical community
reference (Zuppichini, see Resources) exists *solely* to answer "Module vs
Sequential vs ModuleList vs ModuleDict." It is squarely in this section's lane —
§6.1 is literally "how do I hold a collection of layers." *Direction:* after
`MySequential`, add a short subsection: a plain `list` loses parameters (show the
empty `.parameters()`), `ModuleList` fixes it (registered, indexable, you write the
`forward`), `ModuleDict` for named/branching submodules; one decision table
(Sequential = fixed linear chain; ModuleList = you need a loop / variable depth;
ModuleDict = named/selectable branches; Module = a block of all of the above). Then
Exercise 1 becomes "verify this empirically" instead of an unanswered riddle.

**G2 — "Module owns a backward method" framing.** *Why:* false, and it's the
*definition* the section hands the reader. *Direction:* reframe the contract as
"define `forward`; autograd differentiates through it for you," and cut "must
possess a backpropagation method." Forward-point `:numref:`sec_autograd``/`sec_backprop`
for the mechanism (already referenced at line 93).

**G3 — `LazyLinear` as the default.** *Why:* officially experimental; non-idiomatic
vs every external tutorial; secretly front-loads §6.4. *Direction:* authors' call.
Conservative option: keep `LazyLinear` but add one sentence at first use ("we use
the *lazy* variant, which infers `in_features` on the first forward; see §6.4 —
PyTorch marks it experimental") so a student isn't misled about what's standard.
Bolder option: make explicit `nn.Linear(20, 256)` the default and introduce
`LazyLinear` in §6.4 where lazy init is the actual topic.

**G4 — No `torch.compile` touchpoint anywhere in the chapter's building-blocks
sections.** *Why:* in 2026, "build a model" implicitly means "build a model you can
`torch.compile`." The `FixedHiddenMLP` while-loop (data-dependent control flow) is
*the textbook graph-break case* — PyTorch's own "Common Graph Breaks" docs use
exactly this shape. *Direction:* after `FixedHiddenMLP`, add one paragraph: "this
arbitrary Python is the strength of define-by-run, *and* the reason a
data-dependent `while` causes a `torch.compile` graph break; the compiler falls
back to eager around it (or use `torch.cond` / `torch.where`)." A forward-pointer,
not a treatment. This single paragraph is the cheapest way to make the section feel
2026 rather than 2018.

**G5 — Flax `linen` + `setup()`/`@nn.compact` is the maintenance-mode JAX API.**
*Why:* the Flax team now recommends **NNX** for new projects; `linen` is legacy, and
NNX's eager, explicit-`__init__` style is closer to PyTorch (and to the rest of
this section). *Direction:* book-wide decision (affects all JAX tabs, not just
these two) — flag for the orchestrator. At minimum, a footnote that `flax.linen` is
the legacy API and `flax.nnx` is current. The chapter also uses `setup()` for `MLP`
but bare `__call__` for `CenteredLayer` — pick one style.

**G6 — TF framing.** *Why:* "TensorFlow" is now **Keras 3** (multi-backend over
TF/JAX/PyTorch). The `tf.keras` subclass + `build()` code is still valid Keras-3
idiom, so this is cosmetic, but the *label* and any "TensorFlow does X" prose are
dated. *Direction:* footnote/relabel at most; low priority.

**G7 — MXNet co-equality.** Covered in §1.1. *Direction:* book-wide; not these
files' call, but the depth-inversion (MXNet gets the most detailed registration
discussion) should be evened out if the tab stays.

---

## 3. Missing topics — what a top-5 treatment covers here that this omits

The task lists candidates; I sorted them by **whether they belong in *this lane***
(§6.1/§6.5, the "how to build models" guide *between* the MLP chapter and the
architecture chapters). Crucially, the chapter has **dedicated sibling sections**
that own several candidates — so for those the right move is a one-line pointer, not
content here:

| Candidate topic | Verdict for §6.1/§6.5 | Rationale |
|---|---|---|
| **`ModuleList` / `ModuleDict`** | **Belongs here (P1).** | No other section owns "collections of layers." It *is* the model-construction topic. See G1. |
| **Unregistered-plain-list failure mode** | **Belongs here (P1).** | The foundational `nn.Module` gotcha; already half-posed as Exercise 1. See G1. |
| **`named_children()` / `named_modules()` / `named_parameters()` / printing a model** | **Belongs here (P2).** | Introspecting the module tree is core "builder" literacy and the official tutorials teach it (print the model; iterate `named_parameters`). §6.2 owns *parameter access* specifically, but *walking the module tree* (children vs modules, recursion) is a §6.1 concept. A short subsection or a worked `for name, m in net.named_modules()` is in-lane. Coordinate with §6.2 to avoid overlap. |
| **`register_buffer` (in custom-layer body)** | **Belongs here (P1).** | Half the custom-layer state model; currently only in §6.1's example + the slides. See §1.2. |
| **`autograd.Function` (as the deeper escape hatch)** | **Pointer here (P1).** | The natural ceiling of "custom layers." Teach the *distinction* (compose existing ops → `nn.Module`; new differentiable primitive / custom backward → `autograd.Function`) in ~1 paragraph; forward-point the autograd material for the full how-to. See §1.2. |
| **Forward / backward hooks** (`register_forward_hook`, etc.) | **Pointer / brief mention, not a treatment.** | Genuinely useful (feature extraction, debugging, activation inspection), and no other section owns it. But a full hooks treatment is an advanced/diagnostics topic — risks bloating a foundations section. Recommend: one sentence + pointer ("modules expose forward/backward *hooks* for inspection and feature extraction; see [interpretability/debugging ref]"). Not a P1. |
| **Functional (`F.relu`) vs module (`nn.ReLU`) style** | **One paragraph, belongs here (P2).** | Students are immediately confused by `F.relu(...)` in `MLP.forward` vs `nn.ReLU()` in `Sequential`. The chapter *uses both* (line 291 vs 148) and never explains the choice (stateless ops → either; anything with state/params → module; `Sequential` needs modules). A short clarifying note pays for itself. |
| **When to subclass vs compose (`Sequential`)** | **Already implicit; make explicit (P2).** | The chapter shows both but never states the decision rule. MIT 6.S191 and the official tutorials state it plainly ("subclass for flexibility: custom control flow, multiple inputs/outputs, branching"). One or two sentences. |
| **`torch.compile` interaction** | **Pointer here (P1).** | See G4. The while-loop example is the perfect hook. |
| **Parameter/buffer *registration* mechanics** (how `__setattr__` auto-registers) | **Belongs here, lightly (P2).** | The "assign a Module/Parameter to an attribute and it's magically registered" behavior is the central trick; the slides assert it ("aren't ordinary fields — assigning a Module *registers* it") but the prose underplays *why*. fast.ai's from-scratch lesson teaches this by reimplementing `__setattr__`. A sentence naming the mechanism (assignment of `Module`/`Parameter` attributes is intercepted and registered) demystifies the magic. |
| **Parameter *sharing* / tied weights** | **Out of lane — owned by §6.2** (`parameters.md` has a "Tied Parameters" section). | `FixedHiddenMLP` reuses `self.linear` twice; fine to *note* it shares parameters (it already does) and forward-point §6.2. Don't teach tying here. |
| **Initialization APIs** | **Out of lane — owned by §6.3** (`init-param.md`). | Use sensible init in examples (the fan-in scaling is fine); pointer only. |
| **Serialization / `state_dict`** | **Out of lane — owned by §6.6** (`read-write.md`). | Slides mention `state_dict` as a *benefit* of registration (fine as motivation); don't teach saving/loading here. |
| **Device placement / `.to(device)`** | **Out of lane — owned by §6.7** (`use-gpu.md`). | Same: fine as a one-line benefit-of-registration, full treatment elsewhere. |

**Net:** the genuinely-missing, *in-lane* topics are: **(P1)** `ModuleList`/`ModuleDict`
+ the unregistered-list gotcha; `register_buffer` in the custom-layer body;
`autograd.Function` as a pointer; `torch.compile` graph-break pointer. **(P2)**
functional-vs-module note; subclass-vs-compose decision rule; `named_modules` tree
walk; the `__setattr__` registration mechanism in one sentence; sharper exercises.
Everything else is a forward-pointer to a sibling section that already owns it.

---

## 4. How it's taught now — researched, cited resources

Each entry: URL + one line on why it's a good model for *this* material. (All
fetched/searched June 2026.)

**Primary references — the modern idiom**

- **Official PyTorch — "Build the Neural Network"**
  (https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html).
  The canonical beginner path: `__init__`/`forward`, **explicit** `nn.Linear(28*28, 512)`
  (not lazy), `nn.Flatten`, *printing the model* and iterating `named_parameters()`
  to inspect structure, and modern device handling via `torch.accelerator`. This is
  the idiom students will see first; the chapter should not diverge from it (esp. on
  `LazyLinear` and model introspection).
- **Official PyTorch — "Models" (Intro to PyTorch series)**
  (https://docs.pytorch.org/tutorials/beginner/introyt/modelsyt_tutorial.html).
  States the registration contract precisely: "when [Parameters] are assigned as
  attributes of a Module, they are added to the list of that module's parameters" —
  the exact mechanism §6.1 underplays. Good model for *why* assignment registers.
- **Official PyTorch — `torch.nn.LazyLinear` reference**
  (https://docs.pytorch.org/docs/stable/generated/torch.nn.LazyLinear.html). The
  source for the **experimental** caveat on lazy modules (API "likely to change").
  Cite this when justifying G3.
- **Official PyTorch — "Common Graph Breaks" (torch.compile programming model)**
  (https://docs.pytorch.org/docs/stable/compile/programming_model.common_graph_breaks.html).
  Documents that data-dependent control flow (if/while on tensor values, `.item()`)
  breaks the graph, and that `torch.cond`/`torch.where` are the fixes. This *is* the
  `FixedHiddenMLP` while-loop, viewed through 2026 eyes — the basis for G4.
- **Official PyTorch — "Extending PyTorch / Defining new autograd Functions"**
  (https://docs.pytorch.org/docs/stable/notes/extending.html). The reference for
  the custom-layer ceiling: subclass `autograd.Function`, implement `forward` +
  `backward`, wrap in an `nn.Module` for reuse. Basis for the §6.5 pointer.

**The canonical "which container?" guide**

- **F. Zuppichini — "Pytorch: how and when to use Module, Sequential, ModuleList
  and ModuleDict"**
  (https://github.com/FrancescoSaverioZuppichini/Pytorch-how-and-when-to-use-Module-Sequential-ModuleList-and-ModuleDict).
  The community-standard decision guide, and the single best model for G1: "Use
  `Module` when you have a big block composed of multiple smaller blocks; `Sequential`
  to create a small block from layers; `ModuleList` when you need to iterate through
  layers; `ModuleDict` to parametrize blocks (e.g. selectable activation)." Includes
  the dynamic-depth `ModuleList`/`Sequential(*blocks)` pattern.

**Top-program courses**

- **MIT 6.S191 — Intro to Deep Learning, Lab 1 (`PT_Part1_Intro`)**
  (https://github.com/MITDeepLearning/introtodeeplearning/tree/master/lab1).
  Teaches *exactly* this duality at intro level: `nn.Sequential` first, then
  subclass `nn.Module` because it "affords the flexibility to define custom layers,
  custom training loops, custom activation functions, and custom models." A clean
  model for the **subclass-vs-compose** decision rule (G/§3).
- **CMU 11-785 — Intro to Deep Learning**
  (https://deeplearning.cs.cmu.edu/). HW1 has students *reimplement* MLP building
  blocks (Linear, activations, loss, SGD, BatchNorm) from scratch before using
  `nn.Module` — the same "build it yourself to demystify it" pedagogy as the
  chapter's `MySequential`. Validates that the build-from-scratch approach is the
  right one; suggests the chapter could go one notch deeper (reimplement the
  *registration*, à la fast.ai).
- **Stanford CS231n — PyTorch tutorial/assignment 2**
  (https://cs231n.github.io/ → assignment2 `PyTorch.ipynb`, e.g.
  https://github.com/srinadhu/CS231n/blob/master/assignment2/PyTorch.ipynb).
  The classic "Barebones → `nn.Module` → `nn.Sequential`" ladder: define layers as
  attributes in `__init__`, compose in `forward`, then show `Sequential` as the
  convenience layer. Same arc as §6.1; good calibration for depth.

**Books / long-form**

- **S. Prince — "Understanding Deep Learning"** (https://udlbook.github.io/udlbook/).
  Free, modern (MIT Press 2023), with fill-in PyTorch notebooks (e.g. "Composing
  networks"). The gold standard for *intuition-first* presentation with clean vector
  figures — a good model for how to motivate "modules compose" visually rather than
  by assertion.
- **fast.ai — "Deep Learning from the Foundations" / `001a_nn_basics`**
  (https://github.com/fastai/fastai_old/blob/master/dev_nb/001a_nn_basics.ipynb;
  course at https://course.fast.ai/). Builds `nn.Module` *itself* from scratch —
  including reimplementing the `__setattr__` parameter-registration trick. The
  deepest demystification of "why does assigning an attribute register a parameter,"
  directly relevant to the §3 `__setattr__` recommendation.
- **Zero-to-Mastery — "Learn PyTorch for Deep Learning"**
  (https://www.learnpytorch.io/, section 03/04 on building models). A widely-used,
  current (PyTorch 2.x) free course; its "what's inside `nn.Module`" sections (print
  the model, `state_dict`, `parameters()`, `to(device)`) are a good checklist of the
  introspection literacy a 2026 builder is expected to have.

**Reference docs for the cross-framework tabs**

- **Flax — "setup vs compact"** and **"Linen → NNX migration"**
  (https://flax.readthedocs.io/en/latest/guides/setup_or_nncompact.html;
  https://flax.readthedocs.io/en/latest/migrating/linen_to_nnx.html). Confirm
  `linen` is maintenance-mode and **NNX** is the recommended API for new work
  (G5), and that NNX's eager/explicit-`__init__` style mirrors PyTorch.
- **Keras 3 — "Making new layers and models via subclassing"**
  (https://keras.io/guides/making_new_layers_and_models_via_subclassing/). Current
  multi-backend Keras; confirms the `build()`/`call()` subclass pattern in the TF
  tabs is still idiomatic (G6 is cosmetic).

---

## 5. Prioritized recommendations (direction, not a line-by-line spec)

**P1 — do these to clear the bar (all in-lane, all small-to-medium):**

1. **Teach the container vocabulary and the registration gotcha** (G1, §3). Add a
   short `ModuleList`/`ModuleDict` subsection after `MySequential`, *answer*
   Exercise 1 in the body (plain list ⇒ no registered params ⇒ optimizer/​`.to`/​`state_dict`
   all miss them), and give a one-row decision table. Single highest-value change.
2. **De-date the module contract** (G2). Remove "must possess a backpropagation
   method"; reframe as "define `forward`, autograd supplies backward." Fix the
   five-item list item 3.
3. **Add `register_buffer` to the custom-layer body** (§1.2). Promote it from
   slides-only to prose; the parameter/buffer pair is the custom-layer state model.
4. **Add the `autograd.Function` pointer to §6.5** (§1.2, §3). One paragraph on the
   compose-vs-new-primitive distinction; forward-point the autograd material for
   the how-to.
5. **Add the `torch.compile` graph-break paragraph** after `FixedHiddenMLP` (G4).
   Turns the while-loop demo into a 2026 teachable moment; pointer only.
6. **Decide the `LazyLinear` story** (G3). Authors' call; at minimum add the
   one-sentence "lazy = experimental, see §6.4" caveat at first use.

**P2 — to exceed the bar (polish + literacy):**

7. **One paragraph: functional vs module style** (`F.relu` vs `nn.ReLU`), since the
   chapter uses both unexplained.
8. **State the subclass-vs-compose decision rule explicitly** (cite the pattern MIT
   6.S191 / official tutorials use).
9. **Add a `named_modules()`/`named_parameters()` "inspect the tree" snippet** (the
   official tutorials' "print the model" literacy), coordinated with §6.2 to avoid
   overlap.
10. **One sentence naming the `__setattr__` registration mechanism** (demystify the
    "magic" the slides assert).
11. **Strengthen exercises** in both files: replace/augment with problems that drill
    the buffer-vs-parameter distinction, `state_dict` round-trip of a custom layer,
    a `ModuleList`-based variable-depth net, and shape inference. The current
    Fourier-coefficients exercise is underspecified.
12. **Even out cross-framework divergence and the MXNet depth-inversion** (G5–G7);
    pick one Flax style; flag the `linen`→NNX and "TensorFlow"→Keras-3 framing as
    book-wide decisions for the orchestrator.

**Book-wide flags for the orchestrator (not these files alone):**
- `flax.linen` → `flax.nnx` currency (affects every JAX tab).
- "TensorFlow" → "Keras 3" framing (cosmetic but consistent).
- MXNet co-equality and the depth-inversion (legacy framework gets the most
  detailed registration prose).
- Scope handshake with siblings: parameter *sharing* (§6.2), init (§6.3),
  serialization (§6.6), device (§6.7) are pointers from here, not content —
  confirm the reverse sections don't duplicate the module-tree material.

---

## 6. Keep — do not lose this in any revision

- The `MySequential` "build it yourself in 4 lines" demo (§6.1) — top-tier pedagogy.
- The stateless/stateful two-flavor decomposition of custom layers (§6.5).
- `register_buffer` for `rand_weight` in `FixedHiddenMLP` and the explanatory
  comment — correct, modern, better than upstream.
- `nn.Parameter`-wrapping in `MyDense` + the registration-benefit prose.
- The fan-in init touch in `MyDense` (`/ in_units**0.5`).
- The neuron→layer→module motivation and the ResNet "repeating patterns" hook.
- The slide decks' framing (Transformers, parameter sharing, `state_dict`,
  `register_buffer` rationale, "modules all the way down") — they are *ahead* of the
  prose and should be the template the prose catches up to.
