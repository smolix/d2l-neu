# Exercise Review: chapter_gaussian-processes, chapter_hyperparameter-optimization, chapter_generative-adversarial-networks

Repo: `/Users/smola/Repositories/github/d2l-neu`. All profiles built from `.md` source files, full `## Exercises` section read end-to-end (heading to EOF) for every file below. Line numbers verified with `grep -n`/`sed -n`.

---

## chapter_gaussian-processes

### chapter_gaussian-processes/gp-inference.md
```
file: chapter_gaussian-processes/gp-inference.md
heading_line: 359
n_exercises: 6
numbering: sequential
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 0
subproblems: none (ex1 and ex3 bundle several inline questions in one item, but not lettered/numbered sub-parts)
discussions: single-link (begin_tab-wrapped, pytorch only, real thread https://d2l.discourse.group/t/12117)
defects: (none found — clean markup, balanced math, tight list, blank line after heading)
clarity: (none flagged — all six items have concrete asks and follow-up questions with clear deliverables)
notable: Exercise 4 asks readers to time GP regression at 10,000/20,000/40,000 training points — with exact O(n^3) cost this is a heavy compute ask for a "try this" exercise, but it does specify a concrete measurable deliverable (runtime scaling), so not flagged as a clarity defect, just flagged as resource-heavy. File carries a large (~145-line) quarto "slide" deck after the Discussions block — much larger than gp-intro.md, which has no slide deck at all.
```

### chapter_gaussian-processes/gp-intro.md
```
file: chapter_gaussian-processes/gp-intro.md
heading_line: 176
n_exercises: 7
numbering: sequential
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 0
subproblems: none (ex4 bundles 4 inline questions in one item, not lettered)
discussions: single-link (begin_tab-wrapped, pytorch only, https://d2l.discourse.group/t/12115)
defects: (none found)
clarity: (none flagged) — ex5's numeric callback ($f(x_1)=1.2$, $k(x,x_1)=0.9$) was verified against the file body (lines 129-138): the referenced worked example genuinely exists, so not a dangling reference.
notable: File ends immediately after the Discussions block (line 188 = last line) — the only one of the 13 files reviewed with no trailing slide deck.
```

### chapter_gaussian-processes/gp-priors.md
```
file: chapter_gaussian-processes/gp-priors.md
heading_line: 196
n_exercises: 5
numbering: sequential
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 0
subproblems: none
discussions: single-link (begin_tab-wrapped, pytorch only, https://d2l.discourse.group/t/12116)
defects: (none found)
clarity: (none flagged) — all 5 items are direct math questions (GP closure under sums/products, effect of kernel/amplitude) with clear expected answers.
notable: Unlike gp-inference.md/gp-intro.md, each exercise item here is followed by a blank line (loose list spacing) rather than packed tight — harmless for rendering but an inconsistent spacing convention within the same chapter. Also carries a trailing slide deck (~70 lines) like gp-inference.md.
```

---

## chapter_hyperparameter-optimization

### chapter_hyperparameter-optimization/hyperopt-api.md
```
file: chapter_hyperparameter-optimization/hyperopt-api.md
heading_line: 360
n_exercises: 2
numbering: sequential
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 2 (:numref:`sec_dropout` x2, both inside Exercise 1's sub-items)
subproblems: nested-list (exercise 1: 3 sub-items L363-365; exercise 2: 2 sub-items L367-368) — all correctly 4-space indented, verified with `xxd`.
discussions: tabbed(3 tabs: pytorch, tensorflow, jax — all pointing to the same thread https://d2l.discourse.group/t/12092)
defects: (none found — indentation, math, and code-span markup all balanced)
clarity: (none flagged) — both exercises give concrete code identifiers, exact hyperparameter ranges/values, and explicit hint pointers.
notable: none
```

### chapter_hyperparameter-optimization/hyperopt-intro.md
```
file: chapter_hyperparameter-optimization/hyperopt-intro.md
heading_line: 335
n_exercises: 3
numbering: sequential
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: "*Advanced*:" (L346, sub-item 2.5 — italic word + colon)
citations: 2 (:cite:`maclaurin-icml15` L346; :citet:`bergstra-jmlr12a` L348 — mixed styles, one :cite: one :citet:)
crossrefs: 5 (:numref:`sec_generalization_basics` L339; :numref:`sec_mlp-implementation` L341 and L344; :numref:`sec_backprop` L343; :numref:`sec_numerical_stability` L345)
subproblems: nested-list (exercise 1: 3 sub-items L338-340; exercise 2: 5 sub-items L342-346; exercise 3: 1 sub-item L348) — all 4-space indented correctly.
discussions: tabbed(3 tabs: pytorch, tensorflow, jax — all https://d2l.discourse.group/t/12090)
defects:
  - L347-348: Exercise 3 nests a single sub-item ("3.1") with no sibling "3.2" — the sub-list adds no structure since there is only one point; should be flat text under the exercise intro.
clarity:
  - ex 1 (sub-item 1, L338): "Convince yourself (by looking at the code) that this means..." — a pure verification/reading prompt with no producible artifact or check-your-work criterion; closer to "think about X" than a task.
  - ex 2 (sub-item 5, L346): "*Advanced*: Read :cite:`maclaurin-icml15` for an elegant (yet still somewhat unpractical) approach to gradient-based HPO." — literally just "read this paper," no task or deliverable at all.
notable: none
```

### chapter_hyperparameter-optimization/rs-async.md
```
file: chapter_hyperparameter-optimization/rs-async.md
heading_line: 264
n_exercises: 2
numbering: sequential
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: "*Advanced*." (L270 — on the top-level exercise itself, italic word + period; differs in placement and punctuation from hyperopt-intro.md's sub-item-level "*Advanced*:")
citations: 0
crossrefs: 4 (:numref:`sec_dropout` L266; :numref:`sec_api_hpo` L266, L268, L272)
subproblems: nested-list (exercise 1: 3 sub-items L267-269; exercise 2: 3 sub-items L271-273) — 4-space indented correctly.
discussions: single-link (begin_tab-wrapped, pytorch only, https://d2l.discourse.group/t/12093)
defects: (none found)
clarity:
  - ex 2 (sub-item 3, L273): "Compare your new LocalSearcher with RandomSearch on the DropoutMLP benchmark." — unlike Exercise 1's sub-item 3 (which names "incumbent trajectories" as the comparison artifact), this item names no metric or plot to produce; mild vagueness by contrast with its sibling exercise.
notable: This file has only a single (pytorch) Discussions tab, while its two chapter-siblings (hyperopt-api.md, hyperopt-intro.md) both have 3 tabs (pytorch/tensorflow/jax) — inconsistent Discussions coverage within the same chapter.
```

---

## chapter_generative-adversarial-networks

### chapter_generative-adversarial-networks/adversarial-losses.md
```
file: chapter_generative-adversarial-networks/adversarial-losses.md
heading_line: 271
n_exercises: 5
numbering: sequential
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 1 (:citet:`Lin.Wang.Yang.2024` L273)
crossrefs: 2 (:numref:`sec_basic_gan` L274; :numref:`fig_gan_exits` L277)
subproblems: none (ex1 enumerates three tasks to rank with inline "(a)/(b)/(c)" — this labels objects being ranked, not separate crammed sub-questions, so not counted as the inline-letter defect)
discussions: other — bare, non-tabbed link `[Discussions](https://d2l.discourse.group/)` (L279) with no thread ID, unlike every GP/HPO file's begin_tab-wrapped, thread-numbered link.
defects:
  - L279: Discussions link is a bare markdown link (not :begin_tab:-wrapped) pointing to the generic base URL `https://d2l.discourse.group/` with no thread number — reads as an unfilled placeholder.
clarity: (none flagged) — all 5 exercises are dense but concrete (rank tasks against evidence in the section, "show in three lines," modify a stated toy experiment with explicit new parameter values); the "two-band experiment" ex3/ex4 reference was verified to exist in the file body (L267).
notable: Exercises assume close familiarity with very recent (2024-2026) research named earlier in the section (DDO, ADD, DMD2, LADD, MeanFlow, sCM, ViTok) — markedly more specialized than the GP/HPO exercises, but each item is still self-contained given the section.
```

### chapter_generative-adversarial-networks/conditional.md
```
file: chapter_generative-adversarial-networks/conditional.md
heading_line: 835
n_exercises: 7
numbering: repeated-1 (all seven items literally "1." — verified byte-for-byte)
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 1 (:cite:`Brock.Donahue.Simonyan.2019` — ex7/L841)
crossrefs: 6 (:eqref:`eq_gan_cond_projection` L837; :eqref:`eq_gan_cond_value` L838, L839; :numref:`sec_basic_gan` L838, L839; :eqref:`eq_gan_kid` L840)
subproblems: none
discussions: other — bare link `[Discussions](https://d2l.discourse.group/)` (L845), no thread ID.
defects:
  - L845: same placeholder-style bare Discussions link as adversarial-losses.md.
clarity: (none flagged) — ex4's apparent forward references ("the diagnostics cell," "the alignment table," "the feature CNN") were checked against the body and confirmed grounded: L812 explicitly states "Exercise 4 extends the collapse diagnosis to per-class distribution metrics," and the alignment/diagnostics machinery is built at L541-812.
notable: This is the longest exercises set in the group (7 items, very dense multi-part prose each); despite the legacy repeated-1 numbering, clarity is high — every item names its exact deliverable (a derivation, a numerical verification recipe, or a specific comparison with a stated budget).
```

### chapter_generative-adversarial-networks/convergence.md
```
file: chapter_generative-adversarial-networks/convergence.md
heading_line: 563
n_exercises: 5
numbering: repeated-1 (all five items literally "1.")
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 4 (:eqref:`eq_gan_r1r2_sum` L566; :eqref:`eq_gan_r1r2` L566; :eqref:`eq_gan_sobolev` L567; :eqref:`eq_gan_dirac_pen` L568)
subproblems: inline-letters(4) — exercise 4 (L568) crams "(a) ... (b) ... (c) ... (d) ..." as four sub-questions inside a single paragraph rather than as nested list items.
discussions: other — bare link `[Discussions](https://d2l.discourse.group/)` (L571), no thread ID.
defects:
  - L568: exercise 4 packs four lettered sub-questions (a)-(d) inline into one paragraph instead of a nested list — exactly the "crammed inline lettering" anti-pattern the rubric calls out.
  - L571: placeholder-style bare Discussions link.
clarity: (none flagged) — despite the inline-lettering formatting issue, each of the four sub-parts of ex4 states a specific, concrete ask (explain, compute a specific ratio, name statistics, predict a direction).
notable: none beyond the shared chapter-wide Discussions issue.
```

### chapter_generative-adversarial-networks/dcgan.md
```
file: chapter_generative-adversarial-networks/dcgan.md
heading_line: 1132
n_exercises: 6
numbering: repeated-1 (all six items literally "1.")
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 1 (:citet:`Chong.Forsyth.2020` L1135)
crossrefs: 3 (:eqref:`eq_gan_fid` L1134; :numref:`sec_gan_relativistic` L1138; :numref:`subsec_gan_limited_data` L1139)
subproblems: none
discussions: other — bare link `[Discussions](https://d2l.discourse.group/)` (L1141), no thread ID.
defects:
  - L1141: placeholder-style bare Discussions link.
clarity: (none flagged) — all 6 items specify exact experimental protocol (sample sizes, budgets, which arms to compare) and a concrete comparison/deliverable.
notable: none
```

### chapter_generative-adversarial-networks/gan.md
```
file: chapter_generative-adversarial-networks/gan.md
heading_line: 661
n_exercises: 5
numbering: repeated-1 (all five items literally "1.")
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 1 (:eqref:`eq_gan_weights` L666)
subproblems: none
discussions: other — bare link `[Discussions](https://d2l.discourse.group/)` (L669), no thread ID.
defects:
  - L669: placeholder-style bare Discussions link.
clarity: (none flagged) — ex5's "the verification above" callback was checked against the body and is grounded (L494: "Exercise 5 maps this error over the plane").
notable: none
```

### chapter_generative-adversarial-networks/objectives.md
```
file: chapter_generative-adversarial-networks/objectives.md
heading_line: 767
n_exercises: 5
numbering: repeated-1 (all five items literally "1.")
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 4 instances, 3 distinct (:eqref:`eq_gan_bayes_gap` L769, L770; :eqref:`eq_mdl-f-gan-bound` L773; :numref:`sec_basic_gan` L773)
subproblems: none
discussions: other — bare link `[Discussions](https://d2l.discourse.group/)` (L775), no thread ID.
defects:
  - L775: placeholder-style bare Discussions link.
clarity: (none flagged) — all 5 items are precise derivations/verifications with named target identities to reproduce.
notable: none
```

### chapter_generative-adversarial-networks/relativistic.md
```
file: chapter_generative-adversarial-networks/relativistic.md
heading_line: 252
n_exercises: 6
numbering: repeated-1 (all six items literally "1.")
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 8 instances (:eqref:`eq_gan_V` L254, L255; :numref:`sec_basic_gan` L254, L255; :eqref:`eq_gan_rp_explicit` L257; :eqref:`eq_gan_entropy_gap` L257; :numref:`sec_mdl-infonce` L258; :eqref:`eq_gan_rp` L259)
subproblems: inline-letters(1) — exercise 6 (L259) crams "(a) ... (b) ... (c) ..." inline into one paragraph.
discussions: other — bare link `[Discussions](https://d2l.discourse.group/)` (L261), no thread ID.
defects:
  - L259: exercise 6 packs three lettered sub-parts (a)-(c) inline into a single dense paragraph instead of a nested list — same anti-pattern as convergence.md ex4.
  - L261: placeholder-style bare Discussions link.
clarity: (none flagged) — each lettered sub-part of ex6 states an exact identity to show or property to identify.
notable: This is the most math-symbol-dense exercises set in the group (up to 32 `$` per line); all math delimiters and code spans verified balanced (even counts) across every line in every GAN file — no rendering breakage despite the density.
```

---

## Group-level summary

See final message.
