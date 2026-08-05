# Exercise Review: chapter_optimization, chapter_computational-performance

Repo: `/Users/smola/Repositories/github/d2l-neu`. All 19 files with `## Exercises`
in this chapter group were read in full, end to end (heading to end of file),
verified against source with grep/Read. No repo file was edited.

---

## chapter_computational-performance (7 files)

### file: chapter_computational-performance/compilation.md
```
heading_line: 431
n_exercises: 5
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 1 (:numref:`subsec_perf-sweep`, L449 — valid, target in performance-model.md)
subproblems: none
discussions: missing (no Discussions block at all; goes straight from L456 to `<!-- slides -->` at L457)
defects: none found
clarity:
  none — all 5 items specify a concrete action, a measurement, and a question
  tied to that measurement (e.g. ex1: rewrite control flow, confirm break count
  drops to zero, report steady-state time change).
notable: Exercises are hands-on debugging/measurement tasks tied directly to
  named APIs (`torch._dynamo.explain`, `jax.jit`, `static_argnums`). The
  cross-reference to :numref:`subsec_comp-hurts` in ex5 ("when compilation
  hurts checklist") is valid — that checklist exists in this same file (L377-389).
```

### file: chapter_computational-performance/fast-transformer.md
```
heading_line: 1290
n_exercises: 8
numbering: repeated-1
names: none (ex7, ex8 prefixed "(JAX)" as a framework marker, not a name)
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 6 (:numref:`fig_roofline`, `sec_mamba`, `sec_memory_precision` x2,
  `sec_multi_gpu`, `sec_attention-at-scale` — all verified to resolve to real
  :label: targets elsewhere in the book)
subproblems: none
discussions: missing (no Discussions block; L1336 `<!-- slides -->` follows directly)
defects:
  - L1327: ex8 uses a literal, hardcoded "§13.6" instead of a `:numref:` macro,
    inconsistent with every other cross-reference in this same exercise block
    (which all use `:numref:`/`:eqref:`). A renumbered section would silently
    break this reference.
clarity:
  none rise to a real problem — every item names a specific configuration
  (R0-R4, "Configuration 5"), all of which are established earlier in the
  file's body (verified: R0/R1/R2 at L336-626, "Configuration 5" at L857, L886).
notable: ex5 (L1306-1310) states the experimental outcome in advance
  ("Efficiency comes out markedly higher than the eager configuration's")
  before asking the reader to explain it — this pre-empts the discovery the
  exercise is supposed to produce, unlike every sibling exercise in the file,
  which asks the reader to find the number first.
```

### file: chapter_computational-performance/hardware.md
```
heading_line: 496
n_exercises: 8
numbering: repeated-1
names: none (ex7 uses a parenthetical label "(False sharing.)" — a third naming
  variant, neither italic nor bold, distinct from every other file in this group)
name_style: mixed(one parenthetical name, rest unnamed)
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 7 (:numref:`tab_gpu_specs`, `subsec_hw-shoreline`, `sec_gpt`,
  `fig_energy_ladder`, `sec_lenet`, `fig_memory_hierarchy`, `subsec_hw-bytes`
  — all verified valid)
subproblems: inline-letters(ex2)
discussions: missing (no Discussions block; `<!-- slides -->` at L533)
defects:
  - L511: unescaped, unmatched inline-math dollar sign in "At $0.30/kWh, what
    does the epoch's memory traffic cost?" — there is no closing `$` anywhere
    in the rest of the Exercises section. The next literal `$` in the file
    is on L536, inside the following section's slide text ("peak compute
    $P$"). As written, a Pandoc/KaTeX renderer will treat everything from
    "$0.30/kWh" through "$P$" — the rest of ex3, all of ex4-ex8, and the
    intervening slide prose — as a single open math span. This is the most
    severe rendering defect found in the whole chapter group.
  - L505-506: ex2 crams two sub-questions into one paragraph with inline
    lettering — "Estimate (a) the prefill arithmetic intensity ... and (b)
    the decode tokens-per-second bound" — instead of a clean nested list,
    the only clear instance of this pattern in either chapter.
clarity: none beyond the above formatting issues; every item has a concrete
  deliverable.
notable: otherwise the most cross-reference-dense file in the chapter (7
  distinct :numref: targets across 8 exercises), all verified to resolve.
```

### file: chapter_computational-performance/memory-precision.md
```
heading_line: 479
n_exercises: 5
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 4 (:numref:`sec_gpt`, `subsec_mp-anatomy`, `sec_gpt` again,
  `fig_float_formats` — all valid)
subproblems: none
discussions: missing (no Discussions block; `<!-- slides -->` at L507)
defects: none found
clarity: none — every item specifies inputs, a formula or measurement, and
  a verification step (e.g. ex1's "verify one point against
  `max_memory_allocated`").
notable: ex4 is the most elaborate item — deliberately breaks fp16 training,
  confirms the failure, then fixes it with `GradScaler` and asks why the
  failure is workload-dependent; a good example of a well-specified,
  multi-stage exercise with clear success criteria at each stage.
```

### file: chapter_computational-performance/multi-gpu-practice.md
```
heading_line: 765
n_exercises: 6
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 3 (:eqref:`eq_ring_traffic`, :eqref:`eq_dp_cost` x2 — valid,
  labeled via `:eqlabel:` in multiple-gpus.md)
subproblems: none
discussions: missing (no Discussions block; `<!-- slides -->` at L797)
defects: none found
clarity: none — all six are dense but concretely scoped (specific env vars,
  specific sweep ranges, specific numbers to reconcile against equations).
notable: ex3 (L775-783) is unusually elaborate — reproduces two distinct
  fabric behaviors and includes an explicit safety warning ("the run wedges
  within seconds (be ready to kill the launcher)"), atypical among these
  exercises but appropriately concrete rather than decorative.
```

### file: chapter_computational-performance/multiple-gpus.md
```
heading_line: 640
n_exercises: 5
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 7 (:eqref:`eq_dp_cost` x4, :eqref:`eq_ring_traffic` x2,
  :numref:`subsec_hw-interconnects`, `sec_multi_gpu_concise`, `sec_batch_size`
  — all valid; `sec_multi_gpu_concise` also appears in an unlisted
  build-only file, `legacy-multigpu-lib.md`, but that file is explicitly
  excluded from the rendered book per its own header comment, so this is not
  a real duplicate-label conflict)
subproblems: none
discussions: missing (no Discussions block; `<!-- slides -->` at L662)
defects: none found
clarity: none — each item names a specific quantity to compute or predict
  and a specific equation to check it against.
notable: none beyond the equation density.
```

### file: chapter_computational-performance/performance-model.md
```
heading_line: 640
n_exercises: 5
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 1 (:numref:`sec_linear_concise`, valid)
subproblems: none
discussions: missing (no Discussions block; `<!-- slides -->` at L664)
defects: none found
clarity: none — all five have a specific measurement and comparison.
notable: ex5 is unusually short and crisp ("Explain the difference in one
  sentence, and find one more operation with the same property") — a
  well-scoped "explain + extend" pattern, not underspecified despite its brevity.
```

---

## chapter_optimization (12 files)

### file: chapter_optimization/optimization-intro.md
```
heading_line: 330
n_exercises: 5
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 1 (:citet:`Wigner.1958`, L339)
crossrefs: 0
subproblems: nested-list(ex2, ex3, ex4) — correct 4-space indent, renders cleanly
discussions: tabbed(2 tabs: pytorch L368, jax L372)
defects: none found (math and italics balanced)
clarity:
  - ex3 (sub-item 2, L348): "Can you exploit this effect also for optimization
    algorithms?" — "Can you...?" filler-question tone violation per house
    style (docs/style-guide.md §8.8, §17.5); no concrete deliverable is named.
  - ex5 (L364-365): "What other challenges involved in deep learning
    optimization can you think of?" — a pure reading/brainstorming prompt
    with no artifact, no criterion for success, and no scope; also phrased
    as a "can you think of" filler question.
notable: ex2 and ex4's nested sub-items are a clean model of subproblem
  structure — correct 4-space nesting, one question per sub-item.
```

### file: chapter_optimization/gd.md
```
heading_line: 384
n_exercises: 5
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 0
subproblems: nested-list(ex2, ex4) — correct 4-space indent
discussions: single-link (L398, real thread `t/351`; not tabbed — the only
  chapter_optimization file with a real, working Discussions link that
  isn't pytorch/jax-tabbed)
defects: none found
clarity:
  - ex1 (L386): "Experiment with different learning rates and objective
    functions for gradient descent." — no metric, no range, no comparison;
    the canonical underspecified-exercise pattern.
  - ex2 sub-item 1 (L388): parent item says "Implement line search"; the
    sub-item asks "Do you need derivatives for binary search..." — switches
    terminology from "line search" to "binary search" without explaining the
    relationship, leaving it ambiguous whether binary search is the specific
    method intended by "line search" or a separate comparison.
  - ex6/last item (L396): "Apply the algorithm above" is ambiguous — two
    algorithms were introduced above it (line search in ex2, preconditioned
    Newton's method in ex4), and it isn't clear which "the algorithm above"
    refers to.
notable: shortest, most legacy-styled file in the group besides sgd.md/momentum.md.
```

### file: chapter_optimization/sgd.md
```
heading_line: 323
n_exercises: 6
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 0
subproblems: none
discussions: tabbed(2 tabs: pytorch L333, jax L337)
defects: none found
clarity:
  - ex5 (L329): "Can you change $f$ in such a way that to minimize it one
    needs to evaluate all the local minima?" — "Can you...?" filler-question
    phrasing (legacy d2l phrasing carried over); the sentence is also
    grammatically awkward ("in such a way that to minimize it").
notable: exercises are unwrapped single-line paragraphs (unlike the
  hard-wrapped style used throughout chapter_computational-performance),
  a formatting variant carried over from the original d2l source.
```

### file: chapter_optimization/momentum.md
```
heading_line: 388
n_exercises: 5
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 1 (:eqref:`eq_nesterov`, L392 — valid)
subproblems: none
discussions: tabbed(2 tabs: pytorch L397, jax L401)
defects: none found
clarity:
  - ex1 (L390): "Use other combinations of momentum hyperparameters and
    learning rates and observe and analyze the different experimental
    results." — no metric, no range, no criterion for what "analyze" should
    produce.
  - ex4/last item (L394): "Experiment with the parameters." (trailing
    sentence) adds another unscoped "experiment with X" instruction on top
    of two legitimate specific questions in the same item.
notable: ex3 (L393) is a strong counterexample to the above — precise sweep
  range, explicit stopping criterion ($|x_t|\le10^{-6}|x_0|$), explicit
  comparison target ($\beta^\star$).
```

### file: chapter_optimization/adam.md
```
heading_line: 891
n_exercises: 9
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 3 (:cite:`Zaheer.Reddi.Sachan.ea.2018` L908, :cite:`Zeiler.2012`
  L919, :citet:`Kunstner.Yadav.Milligan.ea.2024` L928 — mixed :cite:/:citet: usage)
crossrefs: 3 (:eqref:`eq_adam-moments`, :eqref:`eq_adam-update`,
  :numref:`subsec_mdl-per-coordinate` — all valid)
subproblems: none
discussions: tabbed(2 tabs: pytorch L934, jax L938)
defects: none found
clarity:
  - ex1 (L893-894): "Adjust the learning rate in the from-scratch Adam run on
    the airfoil data, and observe and analyze the results." — no range, no
    metric, no comparison target; the softest item in an otherwise
    tightly-specified file.
notable: longest exercise set in either chapter (9 items); ex5's "Can you
  construct a stream of gradients on which Adam diverges and Yogi converges?"
  (L912-913) uses "Can you" phrasing but does name a concrete deliverable (a
  counterexample stream), so it is a much weaker instance of the tone issue
  than sgd.md ex5 or optimization-intro.md ex3/ex5.
```

### file: chapter_optimization/adamw.md
```
heading_line: 607
n_exercises: 6
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 1 (:citet:`Loshchilov.Hutter.2019`, L639)
crossrefs: 2 (:numref:`subsec_mdl-decoupled-weight-decay`, `sec_momentum` — valid)
subproblems: none
discussions: single-link, dead pattern
defects:
  - L644: `[Discussions](https://d2l.discourse.group/)` — bare discourse-group
    URL with no thread ID, unlike every tabbed Discussions block elsewhere in
    the chapter (`.../t/NNNN`). This is a placeholder/dead link.
clarity: none — all six items are precisely scoped experiments with named
  parameters, ranges, and comparisons.
notable: technically the most rigorous file reviewed in chapter_optimization
  (explicit noise models, explicit fixed-product sweeps, explicit accounting
  of activation memory) — the dead Discussions link is the only blemish.
```

### file: chapter_optimization/batch-size.md
```
heading_line: 732
n_exercises: 6
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 4 (:eqref:`eq_steps-examples` x2, :numref:`subsec_tinylm`,
  `sec_adam` — all valid)
subproblems: none
discussions: single-link, dead pattern
defects:
  - L767: `[Discussions](https://d2l.discourse.group/)` — same bare/dead
    placeholder link as adamw.md.
clarity: none — every item specifies exact sweep values, a metric, and what
  to compare it against.
notable: none beyond the dead link.
```

### file: chapter_optimization/lr-scheduler.md
```
heading_line: 859
n_exercises: 6
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 3 (:citet:`Hagele.Bakouch.Kosson.ea.2024` L875, :cite:`Defazio.Yang.Mehta.ea.2024`
  L876, :citet:`Welling.Teh.2011` L887)
crossrefs: 1 (:numref:`sec_sgd`, valid)
subproblems: none
discussions: tabbed(2 tabs: pytorch L892, jax L896)
defects: none found
clarity:
  - ex6 (L886-889): opens "Read about stochastic gradient Langevin dynamics
    and relate the injected noise scale to..." — the "Read about X" framing
    is a reading-prompt pattern the rubric flags; it is only borderline here
    because the sentence resolves into a concrete relate-to-noise-floor task.
notable: ex5 (L876-885) is a fully-specified stochastic-optimizer
  reimplementation with explicit update equations, constants, and plotting
  instructions — a strong positive example.
```

### file: chapter_optimization/minibatch-sgd.md
```
heading_line: 634
n_exercises: 5
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 0
subproblems: none
discussions: tabbed(2 tabs: pytorch L642, jax L646)
defects: none found
clarity:
  - ex1 (L636): "Modify the batch size and learning rate and observe the rate
    of decline for the value of the objective function and the time consumed
    in each epoch." — no ranges, no explicit comparison target.
  - ex3 (L638): "Compare minibatch stochastic gradient descent with a variant
    that actually samples with replacement... What happens?" — comparison is
    named but "what happens" names no metric or success criterion.
notable: ex4's "evil genie" duplicated-dataset framing (L639) is whimsical,
  legacy-d2l phrasing that sits oddly next to the house style guide's
  restraint norms, though the technical question itself (how do SGD,
  minibatch SGD, and GD react to duplicated data) is concrete and clear.
```

### file: chapter_optimization/muon.md
```
heading_line: 861
n_exercises: 6
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 1 (:cite:`Chen.Liang.Huang.ea.2023`, L884, for Lion)
crossrefs: 6 (:eqref:`eq_adam-moments`, `eq_adam-update`, `eq_muon-ball`,
  `eq_muon-update`, `eq_muon-quintic`, `eq_muon-spectral-step` — all valid)
subproblems: none
discussions: missing
defects:
  - No Discussions block at all before `<!-- slides -->` (L897) — unlike its
    five chapter siblings (adam, adamw, batch-size, lr-scheduler,
    minibatch-sgd, momentum, optimization-intro, sgd) that have either a
    tabbed or single-link Discussions block, muon.md has none.
clarity: none — every item is a precise derivation or measurement task.
notable: densest cross-reference count in chapter_optimization (6 distinct
  equation labels across 6 exercises); technically the strongest-specified
  file in the chapter.
```

### file: chapter_optimization/practice.md
```
heading_line: 576
n_exercises: 6
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 2 (:citet:`Bergsma.Dey.Gosal.ea.2025b` L603, :citet:`Schmidt.Schneider.Hennig.2021` L611)
crossrefs: 6 (:numref:`sec_adam` x2, `sec_batch_size`, `sec_adamw`,
  `tab_practice_recipes`, :eqref:`eq_practice_clip` — all valid)
subproblems: none
discussions: missing (no Discussions block; file ends at L617, no slides
  comment or Discussions found in the remainder of the file either)
defects: none found
clarity: none — this is the best-specified file in the group. ex1 explicitly
  instructs "Keep a log ... Report the log, not just the winning
  configuration," which is a strong, unambiguous success criterion for what
  is otherwise an open-ended tuning exercise.
notable: exemplary anti-underspecification model (ex1); missing Discussions
  block, same pattern as muon.md and scaling.md.
```

### file: chapter_optimization/scaling.md
```
heading_line: 562
n_exercises: 4
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 1 (:citet:`Kosson.Welborn.Liu.ea.2025`, L580)
crossrefs: 1 (:numref:`sec_adamw`, valid)
subproblems: none
discussions: missing
defects: none found
clarity: none — smallest exercise set in the group (4 items) but each names
  a specific manipulation, sweep, or comparison.
notable: shortest exercises section in either chapter; missing Discussions
  block, same pattern as muon.md and practice.md.
```

---

## Group-level patterns (evidence summary for the aggregator)

- **Style uniformity**: all 19 files use bare, unnamed, untagged, repeated-`1.`
  numbering. Zero files in this group use the "named + tagged" or
  "named-only" styles (verified by scanning every top-level item's opening
  token for `*`, `**`, or `[` across all 19 files — zero matches).
- **Discussions block**: three states, unevenly distributed —
  10/19 files have **no Discussions block at all** (all 7
  chapter_computational-performance files, plus muon.md, practice.md,
  scaling.md in chapter_optimization); 2/19 have a **dead placeholder link**
  with no thread ID (adamw.md L644, batch-size.md L767); 7/19 have a working
  block (6 tabbed pytorch/jax: adam, lr-scheduler, minibatch-sgd, momentum,
  optimization-intro, sgd; 1 single real link: gd.md).
- **Worst defect**: hardware.md L511, an unescaped/unmatched inline-math `$`
  in "$0.30/kWh" with no closing `$` anywhere in the rest of the section —
  the renderer's next `$` is on L536, in the following section's slide text,
  so a large span of intervening prose would render as one open math span.
- **Second defect**: hardware.md ex2 (L505-506) crams two sub-questions into
  inline `(a)`/`(b)` lettering instead of a nested list — the only such
  instance in either chapter.
- **Third defect**: fast-transformer.md ex8 (L1327) uses a literal "§13.6"
  instead of `:numref:`, inconsistent with every other cross-reference in
  the same file.
- **Clarity**: chapter_computational-performance is uniformly well-specified
  — zero clarity flags across all 42 exercises. chapter_optimization
  concentrates its clarity problems in the shorter, more legacy-toned files:
  underspecified "experiment and observe" items in adam.md ex1, momentum.md
  ex1/ex4, gd.md ex1, minibatch-sgd.md ex1/ex3; ambiguous cross-item
  references in gd.md (ex2, ex6); "Can you...?" filler-question tone
  violations in optimization-intro.md (ex3, ex5) and sgd.md (ex5).
  practice.md, muon.md, and scaling.md have zero clarity flags.
- **Citations**: 8 distinct `:cite:`/`:citet:` keys across 7 files in
  chapter_optimization (adam.md, adamw.md, lr-scheduler.md, muon.md,
  optimization-intro.md, practice.md, scaling.md); chapter_computational-performance
  has none. All cross-references (`:numref:`/`:eqref:`) checked in both
  chapters resolve to real `:label:`/`:eqlabel:` targets — no broken
  cross-references found.
- **Totals**: 111 exercises across 19 files (42 in chapter_computational-performance,
  69 in chapter_optimization); 0 files with names/tags; 2 files use nested
  sub-lists (gd.md, optimization-intro.md), both correctly 4-space indented.
