# Review — Parameter lifecycle (§6.3–6.5): `parameters.md`, `init-param.md`, `lazy-init.md`

*Landscape review for the "Builders' Guide" parameter-lifecycle trio. Scope: the
practical "how to build/use models" guide — PyTorch 2.x primary, 4-framework book.
This is a review, not a change spec; recommendations give direction, with drafted
material where it sharpens the point. The init **theory** is owned upstream by
`chapter_multilayer-perceptrons/numerical-stability-and-init.md` (§5.4), which
explicitly forward-points "to invoking them through the parameter-initialization
API in `chap_computation`" — i.e. *here*. So this trio is the **API/mechanics
companion** to that theory, and must not re-derive it.*

---

## Executive summary

**What the trio teaches today.** Three short sections covering the imperative
mechanics of parameters: (1) `parameters.md` — accessing parameters (by index, by
`state_dict()`/`named_parameters()`), and parameter tying; (2) `init-param.md` —
built-in initializers, per-layer/per-type init, and custom initializers; (3)
`lazy-init.md` — deferred shape inference (`LazyLinear`, MXNet deferred init, the
Flax `init(rng, x)` story).

**Verdict.** The bones are sound and the *slides* (out of scope, but telling) are
already excellent and modern — they teach the parameter tree, `named_parameters`
as the optimizer/checkpoint iterator, He-vs-Xavier guidance, FixUp/skip-init, and
even name `load_state_dict` as "the ultimate init override." **The prose bodies
lag their own slides badly.** Across the three files the bodies are thin (2021
vintage), miss the concepts that make this an *assignable* treatment
(parameters-vs-buffers, `state_dict` as the serialization contract, *why* you tie
weights, what the defaults actually are and the famous PyTorch `a=sqrt(5)` wart),
and the lazy-init story is now **substantially out of date**: `LazyLinear` is
quietly deprecated upstream and the *modern* deferred-materialization story for
real models is the **`meta` device**, which the section never mentions.

**Grade: C+ / B−.** Each file is individually short and correct but reads like API
trivia, not a top-program treatment of the parameter lifecycle. The single
highest-value move is to **reframe the trio around the object that ties it
together — `nn.Module`'s state (`parameters` + `buffers` = `state_dict`) and its
lifecycle (construct → materialize shapes → initialize → train → serialize)** —
and to refresh lazy-init for the meta-device era. Done well, this clears the bar.

**Highest-value changes (ranked):**
1. **[P0] Parameters-vs-buffers + `state_dict` as the contract** (`parameters.md`).
   The chapter teaches `state_dict()` as an *access* method but never says what it
   *is*: the serialization contract = parameters ∪ buffers. Buffers (BN
   `running_mean`, masks, RoPE caches) are the canonical "saved but not trained"
   state and are entirely absent. This is the concept that makes save/load,
   `requires_grad`, and BatchNorm make sense later.
2. **[P0] Modernize the lazy-init story** (`lazy-init.md`). `nn.LazyLinear` is on a
   deprecation path; the section presents it as *the* mechanism. The real modern
   answer is two-tier: lazy modules for convenience (with the caveat) **and the
   `meta` device + `skip_init` for instantiating huge models without ever
   allocating/initializing weights** — the technique every LLM uses. Currently
   absent.
3. **[P1] Say *why* we tie, and tie *correctly*** (`parameters.md`). The tying
   section shows the mechanic (`assert net[2].weight is net[4].weight`) but the
   *motivating* case — **tied input/output embeddings** (Press & Wolf 2017),
   standard in every LM — is mentioned only in the slides. Add the canonical case
   and the gradient-accumulation consequence to the body.
4. **[P1] Fix/teach the framework defaults honestly** (`init-param.md`). The body's
   per-framework default blurbs are vague and partly wrong by 2026 (PyTorch's
   default is *not* what a reader would call "He"; it's `kaiming_uniform_(a=√5)`,
   a historical accident). State what each default actually is, and the wart.
5. **[P1] Forward-point parametrizations** (`init-param.md` or `custom-layer.md`).
   `torch.nn.utils.parametrize` (weight norm / spectral norm / orthogonal) is the
   modern home for "constrained weights," and supersedes the old
   `nn.utils.weight_norm`. A one-paragraph pointer belongs near custom init.
6. **[P1] MXNet tab** — book-wide: Apache MXNet was archived (2023). These three
   files lean heavily on MXNet idioms (`collect_params`, `force_reinit`, the `-1`
   deferred-shape story). Flag for the overview; at minimum tombstone.

**Counts:** P0 ×3, P1 ×8, P2 ×6 (detailed in §4).

---

# FILE 1 — `chapter_builders-guide/parameters.md` (§6.3 "Parameter Management")

**Role:** Teach how to *reach* the parameters of a built model — for inspection,
for optimizers/weight-decay/checkpointing, and how to *share* them. The natural
"what is the state of a module" section.

**Verdict:** Correct but minimal. It demonstrates four access idioms and a tying
trick, but never names the unifying abstraction (`Module` state = parameters +
buffers, surfaced as `state_dict`), never explains *why* you'd tie, and omits
buffers entirely. The slides for this very file already teach the better version
("A module is a tree; parameters live at the leaves … same iterator powers
optimizers, weight decay, checkpointing"). Bring the body up to the slides.

## 1. Coverage

### Add

- **[P0] Parameters vs. buffers — and `state_dict` as *the contract*.** This is the
  most important gap in the trio. The body uses `net[2].state_dict()` (line 140) as
  an *access* idiom and `named_parameters()` (line 236) as another, but never
  distinguishes them or says what `state_dict` *is*. The teaching points a top
  program expects:
  - A `Module` holds two kinds of tensor state: **parameters** (`nn.Parameter`,
    `requires_grad=True`, returned by `.parameters()`, updated by the optimizer)
    and **buffers** (registered via `register_buffer`, *not* trained, *not* in
    `.parameters()`, but *are* in `state_dict`). `state_dict() = parameters ∪
    buffers` — the complete "everything you must save to reproduce this module."
  - Canonical buffer examples: BatchNorm's `running_mean`/`running_var`, a causal
    attention mask, a RoPE frequency table, an EMA shadow copy. The unifying idea:
    *state that must round-trip through save/load but must not receive gradients.*
  - This directly sets up `read-write.md` (which saves `state_dict` to disk) and
    explains why `model-construction.md` already used `register_buffer` for
    `rand_weight` (line 602 there) without explaining the *category*. **Note the
    overlap:** `model-construction.md` introduces `register_buffer` mechanically;
    this file should own the *concept* (parameter vs buffer vs `state_dict`) and
    `model-construction` should `:numref:` to it. The overview should reconcile
    which file is the canonical home — recommend **here**, since this is the
    "module state" section.
  - **Drafted** in PARAM-1.

- **[P1] *Why* tie parameters (not just *how*).** The tying section (lines 249–351)
  is all mechanism. The body should lead with the *reason*, which currently lives
  only in the slides (lines 466–486): **tied input/output embeddings** in language
  models (Press & Wolf 2017, "Using the Output Embedding to Improve Language
  Models" — standard in GPT-2/BERT/most LMs; saves $|V|\cdot d$ parameters and
  improves perplexity), autoencoder encoder/decoder weight reuse, and recurrent
  weight reuse across time steps. One short paragraph + a `:citet:` (new bib key —
  see PARAM-3). This converts an API curiosity into a technique students will
  actually use.

- **[P2] One sentence on `requires_grad` / freezing.** Accessing parameters is the
  natural place for the single most common parameter *operation* after reading
  them: freezing (`param.requires_grad_(False)`) for fine-tuning / linear probing.
  The init slides already gesture at this ("freezing layers (`requires_grad =
  False`), discriminative learning rates"). One sentence + forward-pointer to
  transfer learning; do **not** expand into a fine-tuning tutorial (out of scope).

### Remove / trim

- **[P2] JAX tying example is a non-example.** The JAX tab (lines 320–334) builds a
  net with a "shared" `nn.Dense` but the only assertion is
  `print(len(params['params']) == 3)` — i.e. it checks the *parameter dict has 3
  entries*, not that anything is tied. In Flax, reusing the same module instance in
  a `Sequential` list does **not** tie parameters (each call site gets its own
  entry unless you explicitly share via a parent module / `param` sharing). So the
  JAX cell silently teaches the *opposite* of the section's thesis ("they are the
  same exact tensor"). Either fix it to demonstrate real Flax parameter sharing
  (shared submodule in a parent `nn.Module`) or, more honestly, add a tab note that
  Flax's functional model makes tying explicit/different and show the idiomatic
  way. **This is a correctness issue, not just trim — P1.** (PARAM-4.)

### Reorder

- Consider promoting the slide's framing sentence into the body intro: "A neural
  network is a tree of parameters; this section covers everything you do with them
  *other* than the training step — inspect, iterate, share, serialize." It orients
  the reader far better than the current "Most of the time, we will be able to
  ignore the nitty-gritty details" (lines 24–28), which undersells the section.

## 2. Teaching quality

- **Structure:** Three `##` sections (Access / Tied / Summary). Fine for the length,
  though "Parameter Access" carries 3 `###` while "Tied" has none — acceptable.
- **Prose:** The intro (lines 9–37) is soft ("Most of the time, we will be able to
  ignore…"). The slides open *much* better. Tighten to a hook (PARAM-6).
- **No figures.** The **slide's ASCII parameter tree** (slides lines 396–416) is
  genuinely clarifying and has **no body equivalent.** A small pre-generated
  schematic of a module tree (nodes = modules, leaves = parameters/buffers, with
  the two access paths "by path" vs "by traversal" annotated) would unlock the
  whole section. This is exactly the house-style *illustrative* figure (committed
  generator → `img/`), not inline matplotlib. **Recommend as P2/judgment** — high
  pedagogical value, real infra cost (the builders-guide chapter has no figure
  generator yet). (PARAM-7.)
- **Exercises (lines 359–363):** Thin. Ex. 1 (NestMLP access) and Ex. 2 (train a
  shared-param net) are fine; Ex. 3 ("Why is sharing parameters a good idea?") is
  good but unanchored — it's answered by the tied-embeddings content you'd be
  adding. Add an exercise on **buffers** (e.g. "register a non-trainable running
  count, verify it's in `state_dict` but not `parameters()`, and survives
  save/load") once PARAM-1 lands. (PARAM-8.)

## 3. Code & examples

- **PyTorch:** Idiomatic and **freshly executed** (verified outputs: access →
  `OrderedDict([('weight', tensor(...)), ('bias', ...)])`; `named_parameters` →
  `[('0.weight', Size([8,4])), …]`; tying assertions pass). The tying example
  correctly uses *the same module instance reused in `Sequential`* and asserts
  `net[2].weight is net[4].weight` — this is the **right** idiom and an improvement
  over upstream d2l (which used a single-`shared` trick). Good. **One nit:** the
  comment "We need to give the shared layer a name so that we can refer to its
  parameters" (lines 286–287) is copied from the MXNet idiom and is **misleading
  for PyTorch** — you don't name it; you reuse the object. Fix the comment.
  (PARAM-5.)
- **JAX:** The access path (`params['params']['layers_2']`, `tree_map(lambda x:
  x.shape, params)`) is correct, modern functional Flax. The **tying cell is wrong**
  (see Remove, PARAM-4). `jax.tree_util.tree_map` is current (note: newer JAX
  exposes `jax.tree.map` as the short alias — optional modernization, P2).
- **TensorFlow:** `net.get_weights()`, `net.layers[2].weights` — works, but this is
  pre-Keras-3 idiom. With Keras 3 the recommended surfaces are `layer.weights` /
  `model.get_weights()` still work, but flag the framework for the book-wide Keras-3
  pass. P2.
- **MXNet:** `collect_params()`, `net[1].params`, `.data()` — legacy; archived
  framework (see book-wide flag).
- **Conventions:** Single imports cell per framework ✓; stable cell IDs ✓; no
  figure-drawing code ✓.

---

# FILE 2 — `chapter_builders-guide/init-param.md` (§6.4 "Parameter Initialization")

**Role:** The **API companion** to the §5.4 init theory: how to *invoke* built-in
initializers, apply different schemes per layer, and write custom ones. §5.4
derives Xavier/He and hands off here for the call.

**Verdict:** Correct mechanics, but the body is vague exactly where a top course is
sharp: (1) the framework-default descriptions are hand-wavy and, for PyTorch,
*misleading* (the default is the infamous `kaiming_uniform_(a=√5)`, not "He"); (2)
it never connects to the §5.4 theory it's the companion to (no `:numref:` back to
`subsec_xavier`/`subsec_he_init`); (3) the modern "constrained-weight" mechanism
(`torch.nn.utils.parametrize`) is absent. The **slides are again far ahead** —
they have the variance argument, the He-vs-Xavier table, the defaults table, and a
"when to override" slide naming FixUp/skip-init. Lift the body toward the slides
while *forward-pointing* the theory to §5.4 rather than re-deriving it.

## 1. Coverage

### Add

- **[P1] Honest framework defaults + the PyTorch `a=√5` wart.** The per-framework
  blurbs (lines 42–69) are imprecise:
  - **PyTorch** (lines 49–53): "initializes weight and bias matrices uniformly by
    drawing from a range computed according to input and output dimension." This
    obscures the actual, much-discussed truth: `nn.Linear.reset_parameters` calls
    `kaiming_uniform_(self.weight, a=math.sqrt(5))`, and bias from
    $U(-1/\sqrt{\text{fan\_in}}, 1/\sqrt{\text{fan\_in}})$. The `a=√5` is **not**
    real Kaiming/He init for ReLU — it's a historical default with "no rigorous
    mathematical justification" (PyTorch issue #57109; community write-ups such as
    Rana, "Don't Trust PyTorch to Initialize Your Variables"). The slide's claim
    that the PyTorch default is "Kaiming-uniform" (slides line 470) is therefore
    *also* slightly misleading and should be qualified. A top program *wants*
    students to know this: the default is a reasonable scaled-uniform but you
    should pass a proper init for ReLU nets. **Drafted** in INIT-1.
  - **JAX/Flax** (lines 61–69): correctly notes `lecun_normal` as the `nn.Dense`
    default (truncated normal, $\text{std}=\sqrt{1/\text{fan\_in}}$) — good, keep.
  - **TF/Keras** (lines 56–59): Glorot-uniform default — correct; keep.
  - **MXNet** (lines 42–47): $U(-0.07,0.07)$ — correct but legacy; the slide rightly
    says "you should override."
- **[P1] Tie back to the theory (`:numref:`).** Add an opening sentence pointing to
  §5.4 (`subsec_xavier`, `subsec_he_init`) for *why* these scales: "We derived the
  variance-preserving rules in :numref:`sec_numerical_stability`; here we *call*
  them." This closes the loop the §5.4 review (NSI-1) opened — that review
  explicitly notes this file "demonstrates `xavier_uniform_` in code but contains
  no derivation," and recommends the cross-link. **Mechanical** (INIT-2).
- **[P1] Forward-point `torch.nn.utils.parametrize` (modern parametrizations).** The
  modern way to impose structure on weights — **weight norm, spectral norm,
  orthogonal** — is `register_parametrization` (PyTorch *Parametrizations*
  tutorial). It supersedes the old `nn.utils.weight_norm`/`spectral_norm` (now
  thin wrappers / partially deprecated) and is the right pointer for "I want my
  weight to always be orthogonal / unit-norm / spectrally bounded." This is
  arguably more `custom-layer.md`'s territory (it's about transforming a
  parameter on access), so **recommend a one-paragraph pointer** — placed wherever
  the overview decides — naming `register_parametrization`, the three built-ins,
  and the `cached()` / `right_inverse` API, with a link to the tutorial. Do **not**
  write a full parametrizations subsection here (out of scope for an init API
  page). (INIT-3; flag cross-file placement for the overview.)
- **[P2] A word on bias/`LayerNorm`/residual init conventions.** The body always
  zeros bias and never mentions that modern nets sometimes deviate: zero-init the
  last BN $\gamma$ in a residual block (the slide already names this — "FixUp /
  Skip-init"), or non-zero bias for specific gates. One sentence + the existing
  slide content; forward-point residual/BN to their chapters. Keep tiny.

### Remove / trim

- **[P2] The "constant init" examples are over-weighted.** Lines 163–209 spend a
  full pair of cells (×4 frameworks) initializing *everything to a constant* — which
  the section itself flags as an anti-pattern (it kills symmetry-breaking; §5.4
  teaches exactly this). The slide handles it in one line ("Constants are an
  anti-pattern but illustrate the API"). Consider collapsing to a single small cell
  with a one-line "(don't actually do this — see §5.4 symmetry-breaking)" note,
  freeing space for the defaults/parametrize content. P2/judgment.

## 2. Teaching quality

- **Structure:** `## Built-in` → `### Custom`. Reasonable. The custom-init math
  display (the heavy-tailed thresholded distribution, lines 277–285) is a fun,
  memorable example — keep.
- **Prose:** Generally fine and compact. The defaults paragraphs are the weak spot
  (see INIT-1).
- **Exercises (lines 406–408):** **One exercise only** ("Look up the online
  documentation for more built-in initializers") — far too thin for a top program.
  The §5.4 review puts the *derivation* exercises there; this file should carry the
  *API/empirical* ones: e.g. (a) initialize a 50-layer linear stack with default vs
  Xavier vs He and plot $\text{Var}[h^{(\ell)}]$ (mirrors NSI-6 but as a coding
  exercise here); (b) replace a `Linear`'s weight with an orthogonal
  parametrization and verify $W^\top W = I$ persists after an optimizer step; (c)
  inspect your framework's actual default for `Linear`/`Dense` and compare to the
  He/Xavier formula. (INIT-6.)

## 3. Code & examples

- **PyTorch:** `net.apply(init_fn)` with `isinstance(m, nn.Linear)` dispatch — the
  correct, idiomatic pattern; the slide rightly calls it "the universal pattern."
  Custom init uses `with torch.no_grad():` for the in-place mask — correct. Fresh
  outputs verified. Good.
- **JAX:** Custom-init cell (lines 357–366) has `def my_init(key, shape,
  dtype=jnp.float_)`. **`jnp.float_` was removed with NumPy 2.0** (alias of
  `float64`); on a current JAX this raises `AttributeError`. The committed output
  exists (so it was captured under an older JAX), but this is a **latent breakage**
  — replace with `jnp.float32` (or `jnp.float64`). **P1 correctness.** (INIT-4.)
  Otherwise the Flax `kernel_init=`/`bias_init=` pattern is correct and idiomatic.
- **TensorFlow:** Each example *rebuilds the whole `Sequential`* to set an
  initializer (lines 137–147, 185–196, 243–256, 344–355) because Keras sets init at
  construction. That's a real framework difference, but the repetition is heavy;
  acceptable. Keras-3 pass: `tf.random_normal_initializer` etc. still valid. P2.
- **MXNet:** `force_reinit=True`, `init.Normal`, `init.Xavier`, `MyInit` subclass —
  legacy idiom; archived framework.
- **Cross-framework:** The four diverge more than necessary (PyTorch/MXNet apply
  post-hoc; TF/Flax set at construction) — but that's intrinsic to the frameworks,
  not gratuitous. Fine.

---

# FILE 3 — `chapter_builders-guide/lazy-init.md` (§6.5 "Lazy Initialization")

**Role:** Explain why you can declare layers without input dims — deferred shape
inference materializes parameters on first forward.

**Verdict:** This is the **most dated** of the three. The concept (defer shape
inference) is sound and well explained, and the slides are excellent (the
"cascade" diagram, the CNN flatten-size motivation). But the *mechanism* it
teaches — PyTorch `nn.LazyLinear` — is now **on a deprecation path upstream**, and
the section completely **misses the modern deferred-materialization story: the
`meta` device**, which is how every large model is actually instantiated today.
For a 2026 "builders' guide," lazy-init must be a two-tier story: *convenience*
(lazy modules, caveated) and *scale* (meta device / `skip_init`).

## 1. Coverage

### Add

- **[P0] The `meta` device + `skip_init` — deferred *materialization* for big
  models.** This is the headline currency gap. The modern reason to "not allocate
  parameters yet" is not ergonomics — it's that you **cannot afford to**: a
  70B-parameter model won't fit the pattern "allocate + random-init on CPU, then
  load weights" (you'd OOM on the throwaway random init). The technique:
  - Instantiate the model on the **`meta` device** (`with torch.device('meta'):` or
    `model.to_empty(device=...)`) — tensors carry *shape/dtype but no storage*, so
    construction is ~instant and allocates nothing.
  - Then either `load_state_dict(..., assign=True)` real weights, or materialize +
    initialize on the target device. `torch.nn.utils.skip_init` is the
    smaller-scale cousin: construct a module with **uninitialized** parameters
    (skips the `reset_parameters` random fill) when you're about to overwrite them
    anyway.
  - This is the foundation of FSDP/`accelerate`/`from_pretrained(low_cpu_mem_usage=
    True)` and is squarely in scope for a *builders'* guide on parameter lifecycle.
  - **Sources:** PyTorch *"Skipping Module Parameter Initialization"* tutorial;
    `torch.nn.utils.skip_init` docs; the FSDP "delayed initialization & meta
    device" material. **Drafted pointer** in LAZY-1. Recommend a new
    `## Deferred Materialization: the meta device` subsection (PyTorch-led, with a
    note that JAX's `eval_shape`/`jax.jit` abstract-eval gives the analogous
    "shapes without data" capability, and `init_with_output`/`lazy` patterns exist).

- **[P1] State the `LazyLinear` deprecation/caveat honestly.** Upstream, the lazy
  modules (`LazyLinear`, `LazyConvNd`, …) are documented as **experimental** and are
  being **deprecated** in favor of the meta-device workflow. The book uses
  `LazyLinear` *pervasively* (it's all over `d2l/torch.py` — `LazyLinear` in the
  base `Module`, LeNet, attention, transformer FFN, etc.), so we can't just rip it
  out, but the section must (a) flag that it's experimental/being phased out, and
  (b) give the non-lazy alternative (`nn.Linear(in, out)` with explicit dims, or
  meta-device + materialize). At minimum a callout box. (LAZY-2.) **This is a
  book-wide dependency** — flag for the overview: if upstream removes lazy modules,
  `d2l/torch.py` needs a migration. The slides' "limitations" bullet ("can't
  `optim.SGD(net.parameters())` until parameters exist") is good and should move
  into the body.

- **[P2] The "infer shape without a real forward" tool.** The body's `apply_init`
  helper (lines 238–245) does a *real* `self.forward(*inputs)` to materialize. Worth
  one sentence that the principled way to get shapes without compute is a dry-run
  on the meta device or, in JAX, `jax.eval_shape` — ties the two halves of the
  section together.

### Remove / trim

- **[P2] The MXNet `-1` deferred-shape narrative** (lines 139–170) is long and
  entirely MXNet-specific (the `collect_params()` prints, the "-1 means unknown,
  `initialize()` does nothing yet" story). With MXNet archived, this is a lot of
  body real estate for a dead framework. Compress hard if MXNet is retained;
  drop if tombstoned. The §5.x reviews are flagging the same MXNet decision.

### Reorder

- Lead the section with the **motivation** (the slide's CNN flatten-size pain:
  "$16\cdot5\cdot5=400$ by-hand comments") before the mechanics. The current open
  ("it might seem that we got away with being sloppy", lines 9–28) buries the
  payoff. Then: convenience tier (lazy modules + caveat) → scale tier (meta
  device).

## 2. Teaching quality

- **Structure:** Single flat run (no `##` subsections; it's short). Adding the meta
  device gives a natural two-`##` split (Convenience / Scale).
- **Prose:** The explanation of the shape cascade (lines 205–216) is clear and good.
  The JAX `:begin_tab:` (lines 107–121) is honest and well-written — it correctly
  says Flax has no "lazy mode" in the PyTorch sense; shape inference happens at
  `net.init(rng, x)`. Keep.
- **Figures:** None in body. The slide "cascade" diagram (slides lines 302–318) is
  excellent and has no body equivalent — candidate for a pre-generated schematic
  (declare → unknown → forward → materialize), same house-style caveat as PARAM-7.
  P2/judgment.
- **Exercises (lines 261–265):** Decent and conceptual (specify some dims not
  others; mismatched dims; varying input dimensionality → "look at parameter
  tying"). Ex. 3's hint ("look at parameter tying") is a bit of a stretch. Add an
  exercise on the **meta device** once LAZY-1 lands (e.g. "instantiate a large MLP
  on `meta`, confirm no memory is allocated, then `to_empty` + init"). (LAZY-6.)

## 3. Code & examples

- **PyTorch:** `nn.LazyLinear(256)` → inspect `<UninitializedParameter>` → forward →
  `torch.Size([256, 20])`. **Outputs verified fresh** (`<UninitializedParameter>`
  then `torch.Size([256, 20])`). The mechanic is correctly shown; the *currency*
  problem is the deprecation, not the code. The `apply_init` `#@save` helper is
  fine.
- **JAX:** `net.init(d2l.get_key(), jnp.zeros((2,20)))` + `tree_flatten_with_path` —
  correct modern Flax. `apply_init` JAX variant fine.
- **TensorFlow:** `[net.layers[i].get_weights() …]` before/after build — works;
  Keras-3 pass applies. P2.
- **MXNet:** `collect_params()` / `initialize()` deferred story — legacy; archived.

---

## 4. Implementation spec (executable findings)

> IDs: `PARAM-n` (parameters.md), `INIT-n` (init-param.md), `LAZY-n`
> (lazy-init.md). Severity P0/P1/P2; effort S/M/L; tag mechanical/authored/judgment.
> **All edits are to the `.md` source; never the generated `.qmd`.**

### PARAM-1 — Parameters vs. buffers + `state_dict` as the contract · [P0] · [M] · [authored]
- **Type:** coverage
- **Where:** `chapter_builders-guide/parameters.md` — new `### Parameters and
  Buffers` subsection, placed after "All Parameters at Once" (after line 247) and
  before `## Tied Parameters`.
- **Change:** Add prose establishing: (1) a `Module` holds **parameters**
  (`nn.Parameter`, `requires_grad=True`, in `.parameters()`, optimizer-updated) and
  **buffers** (`register_buffer`, no grad, not in `.parameters()`, *are* in
  `state_dict`); (2) `state_dict() = parameters ∪ buffers` = the complete
  save/restore payload; (3) canonical buffers: BN `running_mean`/`running_var`,
  attention masks, RoPE caches, EMA shadows. Add a short PyTorch cell:
  demonstrate `register_buffer`, show the tensor appears in `state_dict()` and
  `named_buffers()` but **not** in `named_parameters()`. Per-framework tabs:
  PyTorch primary; JAX note that buffers map to non-trainable "variables"
  collections (`flax.linen` `variable`/`sow`); TF note `non_trainable_weights`;
  MXNet skip or one-line. Forward-point to `read-write.md` (save/load uses exactly
  this `state_dict`).
- **Touches:** Coordinate with `model-construction.md` (it already uses
  `register_buffer` at line 602 and lists `state_dict()` at line 893) — make this
  the canonical definition and have model-construction `:numref:` here. Flag to
  overview.
- **Done when:** Body defines parameter vs buffer vs `state_dict`; a PyTorch cell
  prints a buffer present in `state_dict()`/`named_buffers()` and absent from
  `named_parameters()`; renders clean in HTML+PDF.
- **Depends on:** overview decision on canonical home (parameters.md recommended).

### PARAM-3 — Add the "why we tie" paragraph (tied embeddings) · [P1] · [S] · [authored]
- **Type:** coverage / currency
- **Where:** `parameters.md`, top of `## Tied Parameters` (line 249), before the
  mechanics.
- **Change:** 4–6 sentences: weight tying matters because it (a) saves parameters
  and (b) can improve generalization. Canonical case: **tied input/output
  embeddings** in language models — the embedding matrix and the output softmax
  projection share weights (Press & Wolf 2017), saving $|V|\cdot d$ params and
  improving perplexity; standard in GPT-2/most LMs. Other cases: autoencoder
  encoder/decoder, recurrent weight reuse across time. Then the existing mechanics
  follow. Add the gradient note (already in the `:begin_tab:` at 343–351 — keep) up
  front: tied → gradients **accumulate** across uses.
- **Touches:** `d2l.bib` — add `Press.Wolf.2017` (EACL 2017, "Using the Output
  Embedding to Improve Language Models", arXiv:1608.05859). **New bib key.**
- **Done when:** Tying section opens with motivation + `:citet:`Press.Wolf.2017``;
  bib entry resolves; renders.
- **Depends on:** none.

### PARAM-4 — Fix the misleading JAX "tying" example · [P1] · [M] · [authored]
- **Type:** code / correctness
- **Where:** `parameters.md` JAX tab of `#parameters-tied-parameters` (lines
  320–334).
- **Change:** The current cell only asserts `len(params['params']) == 3`, which
  does **not** demonstrate tying (reusing a module instance in a Flax `Sequential`
  list does not share parameters). Either (a) rewrite to show *real* Flax parameter
  sharing — define a parent `nn.Module` that applies one shared `nn.Dense` submodule
  twice (so the param dict has a single shared entry), and assert the shapes/paths
  reflect sharing; or (b) add an explicit tab note that Flax's functional design
  makes tying explicit and different from the imperative frameworks, and show the
  idiomatic shared-submodule pattern. Prefer (a). Keep the prose thesis ("same exact
  tensor") honest for the imperative tabs only.
- **Touches:** none beyond the cell.
- **Done when:** JAX cell demonstrably ties (shared param entry / equal-by-identity
  semantics) **or** clearly states Flax's different model; executes clean in the
  committed store.
- **Depends on:** none.

### PARAM-5 — Fix the misleading PyTorch tying comment · [P2] · [S] · [mechanical]
- **Type:** prose / code comment
- **Where:** `parameters.md` PyTorch tab, lines 286–287.
- **Change:** `# We need to give the shared layer a name so that we can refer to its
  parameters` → `# Reuse the *same* module instance at two positions; PyTorch then
  treats them as one parameter set (tied, not copied).`
- **Done when:** comment matches the PyTorch idiom (no "name" language).
- **Depends on:** none.

### PARAM-6 — Tighten the intro to a hook · [P2] · [S] · [authored]
- **Where:** `parameters.md` lines 9–37.
- **Change:** Replace the soft "Most of the time we can ignore the details" opening
  with the slide's framing: a network is a *tree of parameters*; this section covers
  everything you do with them other than the gradient step — inspect, iterate,
  share, serialize. ~8 lines.
- **Done when:** intro leads with the tree framing; no "ignore the nitty-gritty"
  hedge.
- **Depends on:** none.

### PARAM-7 — Module-tree schematic (illustrative figure) · [P2] · [L] · [judgment]
- **Type:** figure
- **Where:** `parameters.md` near "Parameter Access"; mirror the slide ASCII tree
  (slides 396–416) as a clean SVG: modules = nodes, parameters/buffers = leaves,
  annotate the two access paths (by-path `net[2].weight`; by-traversal
  `named_parameters()`).
- **Touches:** needs a builders-guide figure generator (`tools/gen_mdl_*` pattern
  exists only for the math appendix today) → `make figures` → `img/<id>.svg`. **No
  inline matplotlib.** Use the `mdl-figure` skill.
- **Done when:** figure committed as `img/`, included with `:label:` + `:numref:`,
  byte-idempotent, renders HTML+PDF.
- **Depends on:** decision to stand up a chapter figure generator (infra).

### PARAM-8 — Buffer exercise · [P2] · [S] · [authored]
- **Where:** `parameters.md` exercises (after line 363).
- **Change:** Add: "Register a non-trainable counter via `register_buffer`. Verify
  it appears in `state_dict()` and `named_buffers()` but not in
  `named_parameters()`, survives a `save`/`load` round-trip, and is unchanged by an
  optimizer step."
- **Done when:** exercise present.
- **Depends on:** PARAM-1.

### INIT-1 — Honest framework defaults + PyTorch `a=√5` wart · [P1] · [M] · [authored]
- **Type:** currency / coverage
- **Where:** `init-param.md`, the per-framework `:begin_tab:` blocks (lines 42–69)
  plus a short shared paragraph.
- **Change:** Replace vague PyTorch blurb with the truth: `nn.Linear` weights are
  `nn.init.kaiming_uniform_(weight, a=math.sqrt(5))` and bias
  $U(-1/\sqrt{\text{fan\_in}}, +1/\sqrt{\text{fan\_in}})$; the `a=√5` is a historical
  default *not* equivalent to recommended He-for-ReLU init and has "no rigorous
  justification" (PyTorch issue #57109). Add one sentence: for ReLU nets, prefer
  passing `kaiming_normal_`/`kaiming_uniform_` (the §5.4 He rule) explicitly. Keep
  the correct JAX (`lecun_normal`), TF (Glorot-uniform), MXNet ($U(\pm0.07)$, legacy)
  notes. Also qualify the slide's "Kaiming-uniform … Default" claim if the slide is
  later revised.
- **Touches:** optional new bib for the Rana blog / issue (or just cite inline as a
  footnote URL).
- **Done when:** PyTorch default stated exactly with the `a=√5` caveat; other
  defaults accurate.
- **Depends on:** none.

### INIT-2 — Cross-link to the §5.4 theory · [P1] · [S] · [mechanical]
- **Type:** currency / cross-reference
- **Where:** `init-param.md` intro (after line 14, "We discussed the need for proper
  initialization in :numref:`sec_numerical_stability`.").
- **Change:** Add: "There we *derived* the variance-preserving scales — Xavier
  (:numref:`subsec_xavier`) and He (:numref:`subsec_he_init`); here we *invoke*
  them through the framework's initialization API." This closes the loop §5.4's
  NSI-1 opened.
- **Done when:** both `:numref:`s resolve; reads as the API companion to §5.4.
- **Depends on:** §5.4 NSI-1 having `subsec_he_init` label (it does — `init-param`
  can link regardless).

### INIT-3 — Forward-point `torch.nn.utils.parametrize` · [P1] · [S] · [authored]
- **Type:** coverage / currency
- **Where:** end of `### Custom Initialization` in `init-param.md` (after line 400)
  — OR `custom-layer.md` (overview decides; it's about transforming a parameter on
  access).
- **Change:** One paragraph: distinguish *custom init* (set the value once) from
  *parametrization* (impose structure on every access). Name
  `torch.nn.utils.parametrize.register_parametrization` and the three shipped
  cases — **weight norm, spectral norm, orthogonal** (`torch.nn.utils.parametrizations.{weight_norm,
  spectral_norm, orthogonal}`) — note it supersedes the old
  `nn.utils.weight_norm`/`spectral_norm`, and the `cached()` / `right_inverse` API.
  Link the PyTorch *Parametrizations* tutorial. Keep to a pointer; no full
  subsection.
- **Touches:** cross-file placement decision (flag to overview).
- **Done when:** paragraph names the API + three built-ins + the supersession; links
  the tutorial; renders.
- **Depends on:** overview placement call.

### INIT-4 — Fix `jnp.float_` (NumPy 2.0 removal) · [P1] · [S] · [mechanical]
- **Type:** code / correctness
- **Where:** `init-param.md` JAX tab `#init-param-custom-initialization-1`, line 359.
- **Change:** `def my_init(key, shape, dtype=jnp.float_):` → `def my_init(key, shape,
  dtype=jnp.float32):` (`jnp.float_` was removed with NumPy 2.0; raises
  `AttributeError` on current JAX).
- **Touches:** re-capture the JAX output for this cell (`make -B
  _notebooks/jax/...` + `capture-outputs`) — but note jax is GPU-flagged; CPU
  capture of this cell is fine.
- **Done when:** cell uses `jnp.float32`; JAX notebook executes without
  `AttributeError`.
- **Depends on:** none.

### INIT-6 — Strengthen exercises (API/empirical) · [P1] · [M] · [authored]
- **Where:** `init-param.md` exercises (lines 406–408), currently one item.
- **Change:** Add: (a) initialize a 50-layer linear stack default vs Xavier vs He,
  plot $\text{Var}[h^{(\ell)}]$ vs depth (coding companion to §5.4 NSI-6); (b)
  apply an orthogonal parametrization to a `Linear`, verify $W^\top W=I$ persists
  after an SGD step; (c) print your framework's actual `Linear`/`Dense` default and
  compare to the He/Xavier formula (surfaces the `a=√5` wart).
- **Done when:** ≥3 exercises building mechanical→empirical.
- **Depends on:** INIT-1 (wart), INIT-3 (parametrize).

### LAZY-1 — Add the meta device / `skip_init` (deferred materialization) · [P0] · [M] · [authored]
- **Type:** coverage / currency
- **Where:** `lazy-init.md` — new `## Deferred Materialization: the meta device`
  section after the existing lazy content (after line 253), before Summary.
- **Change:** Teach the *scale* tier: instantiate a model on the `meta` device
  (`with torch.device('meta'):`) so tensors carry shape/dtype but **no storage**
  (construction allocates nothing); then materialize with `to_empty(device=...)` +
  init, or `load_state_dict(..., assign=True)` real weights. Introduce
  `torch.nn.utils.skip_init` as the "construct without the throwaway random init"
  cousin (requires the module to do no compute on params in `__init__` beyond
  `torch.nn.init`). Motivate with LLMs (can't afford to random-init 70B params just
  to overwrite them) and name FSDP/`accelerate`/`from_pretrained(low_cpu_mem_usage)`
  as where this lives. Add a short PyTorch cell: build an MLP under `meta`, show a
  param's `.device` is `meta` / it has no storage, then `to_empty` + an init.
  Per-framework: PyTorch primary; one-line JAX note that `jax.eval_shape` /
  abstract eval gives the analogous "shapes without data." MXNet: skip.
- **Touches:** `d2l.bib` optional (cite a torchtitan/FSDP reference or just link
  docs). Update Summary to mention both tiers.
- **Done when:** section teaches meta-device construction + materialization +
  `skip_init`, with a runnable PyTorch cell; renders HTML+PDF; Summary updated.
- **Depends on:** none.

### LAZY-2 — Flag `LazyLinear` as experimental / deprecating + give the alternative · [P1] · [S] · [authored]
- **Type:** currency
- **Where:** `lazy-init.md`, callout near the first `nn.LazyLinear` (after line 83),
  and a line in Summary.
- **Change:** Add a short callout: PyTorch's lazy modules (`LazyLinear`,
  `LazyConvNd`, …) are **experimental and on a deprecation path**; the explicit form
  is `nn.Linear(in_features, out_features)`, and the scalable alternative for large
  models is the meta device (LAZY-1). Move the slide's good limitation note ("can't
  build `optim.SGD(net.parameters())` until parameters exist — pass data once
  first") into the body.
- **Touches:** **Book-wide:** `d2l/torch.py` uses `LazyLinear` pervasively (base
  `Module`, LeNet, attention, FFN). If upstream removes lazy modules this needs a
  migration — flag to overview, do **not** change `d2l/torch.py` in this pass.
- **Done when:** callout present; limitation note in body; overview notified of the
  `d2l/torch.py` dependency.
- **Depends on:** LAZY-1 (references the meta-device alternative).

### LAZY-6 — Meta-device exercise · [P2] · [S] · [authored]
- **Where:** `lazy-init.md` exercises (after line 265).
- **Change:** Add: "Instantiate a large MLP on the `meta` device; confirm
  construction allocates no real memory (inspect a parameter's device/storage), then
  `to_empty` it onto CPU and apply He init."
- **Done when:** exercise present.
- **Depends on:** LAZY-1.

### Cross-file / book-wide (flag to overview)
- **CF-1 [P1]** MXNet archived (2023): all three files lean on MXNet idioms
  (`collect_params`, `force_reinit`, the `-1` deferred-shape story). Decide
  tombstone vs drop, consistent with §5.x reviews.
- **CF-2 [P1]** Keras-3 / TF currency pass across the trio (`get_weights`, layer
  rebuild-to-set-init) — part of the book-wide TF pass.
- **CF-3 [P1]** Canonical home for `state_dict`/buffers concept (PARAM-1) vs
  `model-construction.md` (which already touches `register_buffer`/`state_dict`) and
  `read-write.md` (which serializes `state_dict`). Recommend: *concept* in
  `parameters.md`, *serialization* in `read-write.md`, *mechanics* cross-linked.
- **CF-4 [P1]** `LazyLinear` deprecation → `d2l/torch.py` migration risk (LAZY-2).

---

## 5. Keep — what is already excellent (do not lose this)

- **The PyTorch tying example** in `parameters.md` (reuse-the-instance,
  `assert net[2].weight is net[4].weight`) is the *correct* modern idiom and an
  improvement over upstream d2l's single-`shared` trick. Keep.
- **The custom-init example** in `init-param.md` (the heavy-tailed thresholded
  distribution) is memorable and teaches the API cleanly. Keep.
- **The lazy-init shape-cascade explanation** (prose lines 205–216) and the honest
  **JAX `:begin_tab:`** (107–121, correctly stating Flax has no PyTorch-style lazy
  mode) are clear and right. Keep.
- **All three slide decks are genuinely strong** and already contain much of the
  modern framing the bodies lack (parameter tree, `named_parameters` as the
  optimizer/checkpoint iterator, He-vs-Xavier table + defaults table, FixUp /
  skip-init, the cascade diagram, "limitations" of lazy init). They are the
  template the bodies should rise to — **do not regress the slides.**
- **Notebooks are freshly executed** (outputs verified for all three, all
  frameworks): PyTorch `<UninitializedParameter>`→`Size([256,20])`,
  `named_parameters` shapes, tying assertions, JAX/TF/MXNet outputs all sensible.
  No stale numbers found. The only latent code breakage is `jnp.float_` (INIT-4).

---

## Resources consulted (how this trio is taught now)

- **PyTorch *Parametrizations Tutorial*** — docs.pytorch.org/tutorials/intermediate/parametrizations.html
  — the modern, canonical treatment of constrained weights (`register_parametrization`,
  orthogonal/weight-norm/spectral-norm, `cached()`, `right_inverse`); supersedes
  `nn.utils.weight_norm`. *Basis for INIT-3.*
- **PyTorch *Skipping Module Parameter Initialization*** —
  docs.pytorch.org/tutorials/unstable/skip_param_init.html — and
  **`torch.nn.utils.skip_init`** docs — the official "don't pay for init you'll
  overwrite" pattern; states the no-compute-in-`__init__` requirement. *Basis for
  LAZY-1.*
- **FSDP "Delayed Initialization & meta device"** (apxml FSDP course) +
  **TorchTitan** (arXiv:2410.06511) — meta-device instantiation as the standard
  large-model workflow (flat host memory, allocate straight to sharded GPU).
  *Basis for LAZY-1.*
- **PyTorch issue #57109** and **A. Rana, "Don't Trust PyTorch to Initialize Your
  Variables"** (adityassrana.github.io) — document that `nn.Linear`'s default is
  `kaiming_uniform_(a=√5)`, *not* recommended He init, with "no rigorous
  justification." *Basis for INIT-1.*
- **CS231n "Neural Networks 2" notes** (cs231n.github.io/neural-networks-2/) —
  Stanford's standard derivation of Xavier/He and symmetry-breaking; calibrates the
  theory the §5.4 file owns (and that this trio should *point to*, not repeat).
- **Prince, *Understanding Deep Learning*, Ch. 7** (init) — He $v^2=2/N$ for ReLU,
  the forward/backward variance argument, the worked 30-/50-layer "Xavier fails,
  He trains" example; a top reference for the theory this trio defers to.
- **Press & Wolf 2017, "Using the Output Embedding to Improve Language Models"**
  (aclanthology.org/E17-2025/, arXiv:1608.05859) — the canonical *why-tie-weights*
  result (tied input/output embeddings; standard in modern LMs). *Basis for
  PARAM-3.*
- **PyTorch forums, register_buffer vs register_parameter** (ptrblck) — the standard
  explanation of buffers (BN `running_mean`) vs parameters and their `state_dict`
  membership. *Basis for PARAM-1.*
- **JAX changelog / NumPy 2.0 migration guide** — `jnp.float_` removed with NumPy
  2.0 (alias of `float64`). *Basis for INIT-4.*
