# Chapter 10, Attention: writing review

Sources: the eight `.md` files of `chapter_attention/`. The six teaching sections were read in full by two section reviewers (one took the first three, the other the last three); I read `index.md` and `further-reading.md` in full and checked the cross-section claims below against the sources, the stored PyTorch outputs, `d2l.bib`, and `chapter_transformers/transformer-block.md` and `kv-cache.md`. About 20,700 words of body prose in the six sections, about 10,100 words of exercises, plus 2,300 words of further reading.

Per-file reviews:
[index](10-attention/index.md) ·
[queries-keys-values](10-attention/queries-keys-values.md) ·
[attention-scoring](10-attention/attention-scoring.md) ·
[multihead-attention](10-attention/multihead-attention.md) ·
[positional-information](10-attention/positional-information.md) ·
[attention-at-scale](10-attention/attention-at-scale.md) ·
[what-attention-computes](10-attention/what-attention-computes.md) ·
[further-reading](10-attention/further-reading.md)

## Verdict

A good chapter, current for 2026, with a real plan and several passages at the bar, and with a hole in the middle. The plan is stated in the third paragraph of the introduction: a differentiable lookup, then the three problems it creates (order, cost, and what a head computes). The second half of the chapter carries that plan out well. The RoPE derivation from a requirement, the online softmax with its sentence "Chunking changes the order of evaluation, not the definition of attention", and the circuits section, which trains a model, finds the circuit and then tries four ways to be wrong about it, are better than what the comparable texts offer. The hole is in the first half, at the step every later chapter depends on. The computation $\mathrm{softmax}(\mathbf{Q}\mathbf{K}^\top/\sqrt{d})\,\mathbf{V}$ is never run on an input whose answer the reader can predict, single-head attention with learned projections is not written down until the fourth section, and self-attention, the layer the rest of the book is built from, is taught in twelve lines and a shape check on a tensor of ones. What matters most is to teach self-attention properly, on one sentence, with numbers and a picture, before multiplying heads.

**Against The Illustrated Transformer.** The comparison is most direct in sections 2 and 3, and there the chapter is below it. Alammar's page succeeds because it takes one sentence, follows one token through query, key, value, score, softmax and sum, and draws every step with small numbers. This chapter has that quality in exactly one place, the first page of the first section (a two-record database, a query of 4, an answer of 19.5), and then abandons it at the point where the reader most needs it. In sections 2 and 3, five cells validate the central classes by `d2l.check_shape` on random or constant tensors. Everywhere else the chapter does more than The Illustrated Transformer and is more rigorous: it derives the dot product from the Gaussian kernel, proves what one head cannot do, treats masked-softmax numerics, positions, cost and circuits, none of which that page has. So the honest statement is: a deeper and more current text than The Illustrated Transformer, and a worse first explanation of self-attention. The fix is cheap relative to what is already there, because the material (the "Alice gave Bob the book" sentence, the letter-matching demonstration, `fig_qkv`) exists and is in the wrong places.

## Score table

Scores are the section reviewers' (index and further-reading are mine); 5 means the page would stand in MacKay or as an Illustrated-Transformer-quality explanation.

| Section | Explanation | Thought | Pedagogy | Language | Enjoyment |
|---|---|---|---|---|---|
| [index](10-attention/index.md) | 4 | 4 | 3 | 4 | 3 |
| [queries-keys-values](10-attention/queries-keys-values.md) | 4 | 4 | 3 | 4 | 3 |
| [attention-scoring](10-attention/attention-scoring.md) | 4 | 3 | 3 | 4 | 3 |
| [multihead-attention](10-attention/multihead-attention.md) | 3 | 3 | 3 | 4 | 3 |
| [positional-information](10-attention/positional-information.md) | 4 | 3 | 4 | 4 | 4 |
| [attention-at-scale](10-attention/attention-at-scale.md) | 4 | 3 | 3 | 4 | 3 |
| [what-attention-computes](10-attention/what-attention-computes.md) | 4 | 4 | 5 | 4 | 4 |
| [further-reading](10-attention/further-reading.md) | 4 | 3 | 3 | 4 | 3 |

The two reviewers calibrated alike; I see no systematic difference between the first three rows and the last three. Language is 4 in every row and does not discriminate: the prose is clean, plain and short-sentenced throughout, with voice on the first page of several sections and little after it. The other columns do discriminate. **Strongest:** `what-attention-computes.md` (controls that fail, an ablation that overturns the reading the heatmaps invited, an evidence table with a "remaining limitation" column) and the first half of `positional-information.md`. **Weakest:** `multihead-attention.md` (one excellent idea, out of proportion, with the most used topic least taught) and `attention-scoring.md` (no worked example of its central formula, a paper retold in place of an experiment). `attention-at-scale.md` is sound and useful and scores low on thought and pedagogy because it is three sections in one file with one rhythm.

## Technical correctness and currency

No derivation, definition or complexity count was found wrong; both reviewers recomputed the arithmetic (FLOP and parameter counts, cache sizes, the multi-head bound, RoPE periods, tile fractions) and it holds. The findings of substance are about readings of experiments and claims about papers. In order of severity:

1. **CIRC-11** (major; reviewer sure of the arithmetic, stored accuracy read from a figure). The circuits section attributes the remaining errors to a circuit that "matches on exactly one preceding token". A single-token match has a ceiling of about 0.84 on this task; the stored models reach about 0.89 to 0.93, so they resolve some ambiguous cases and the text asserts what its own Exercise 8(d) leaves open.
2. **POS-11** (moderate; fairly sure; I confirmed the code). The sinusoidal arm of the positional experiment also multiplies the token embeddings by $\sqrt{d}$, with an output head tied to the embedding, so "the positional scheme is the one constructor argument that the experiment" varies is not true, and the in-range ranking ("the sinusoidal table and RoPE give the lowest") ranks something other than positional schemes.
3. **SCORE-2** (moderate; fairly sure, simulated). The saturation experiment uses unit-variance coordinates and is read as "a network at initialization. That is when it matters". With the default initializer of the chapter's own layers the unscaled entropy at $d = 64$ is about 2.0 nats, not 0.6. The robust conclusion (sharpness independent of width) should replace the claim of near one-hot attention at the start of training.
4. **SCORE-5** (moderate; I confirmed the stored output). Prose says the unscaled Jacobian norm "first *rises*"; the cell prints 0.253, 0.219, 0.085. The rise exists between sizes the cell does not print. "About 80%", "about 90%" and "near $d = 128$" are also correct and printed nowhere.
5. **CIRC-10** (moderate). The explanation that block 2 reads a transformed copy written by block 1 is a hypothesis; no cell computes the raw-embedding count it rests on, and "as the copy matrix above showed" later cites it as established.
6. **SCORE-16** (moderate; (a) certain, (b) unverified). In Bahdanau et al. the score uses the previous decoder state, not the "current" one; and "had different widths, so instead of a dot product" attributes a motive the paper does not state as far as the reviewer knows.
7. **MHA-14** (moderate). "This is the averaging of the copy-both task on a small scale" is wrong: the two matched keys carry the same value, so nothing is lost. The true lesson is that content cannot tell two occurrences apart.
8. **MHA-12** (moderate; numbers unverified against the sources). Seven last-digit numbers from two papers and a model card inside a teaching section (47.2 against 47.1; 45.2, 41.2, 37.9); "reading it dominates the time per generated token" is asserted, not shown.
9. **QKV-4** (moderate; certain, I confirmed `d2l.bib`). `parzen1957consistent` is Parzen's paper on spectral estimates of time series, not the density-estimation paper (1962).
10. **FR-2, FR-3, IDX-4** (moderate and minor; mine). The further reading says the trained heads of the multi-head experiment "nearly duplicated each other", which the section body does not say; and it says Bahdanau's attention "was not the first differentiable alignment" while the introduction calls it "the first learned alignment".
11. **POS-15** (minor). "Close to its level at the training" length for 9.8 and 9.1 against 7.2 and 7.4; the cost of rescaling at the training length itself (8.4 against 7.2) is not mentioned.
12. **MHA-6** (minor). "More than two orders of magnitude" for a stored error of $10^{-7}$ against 0.71.
13. Minor and mostly "accurate as far as the reviewer knows, unverified": QKV-1 (convolutions and the fixed-state limit), QKV-5 ("best possible predictor"), POS-7 ($\mathbf{R}_0 = \mathbf{I}$ presented as convention), POS-17 (papers on context extension; NTK-aware scaling predates YaRN), COST-4, COST-5 (the cache explanation of the first timing step), COST-8 ("hundreds of arithmetic operations" as a universal), COST-9, COST-10 ("several times" for 19 times and 7 times), COST-11 (framework API claims not run), COST-13 (Llama 4's aim), COST-16 ("made the recurrent form competitive"), CIRC-7, CIRC-8 (block-1 head patterns described more tidily than the stored figure), CIRC-13, CIRC-15, FR-5 (benchmark ratios at two digits), FR-9, IDX-7 (RASP attributed to Rush).

**Currency.** Good. RoPE, ALiBi, NoPE, context extension, fused kernels, FlexAttention, interleaved windows, gated linear recurrences and induction heads are the 2026 picture. Three small points of superseded practice: SCORE-7 (the float16 argument without a word on bfloat16), SCORE-14 (attention dropout shown at 0.5 without saying that 0 is the usual setting today), and the sinusoidal scheme's weight in the positional section relative to its use.

**Numbers no cell prints.** SCORE-5 (three), POS-13 (the 1.1 million bytes), CIRC-10 (the raw-embedding count). The cost section's unprinted numbers are hand arithmetic and fine.

## Arc and order

**One line of argument: yes, in outline.** Lookup (1), score and masks (2), heads (3), then order (4), cost (5), interpretation (6). The handovers are mostly real: the fixed kernel's limit closes section 1 and opens section 2; the heatmap warning of section 1 returns as the thesis of section 6; `TinyCharLM` of section 4 is the subject of section 6. The order of sections is right and no reviewer proposes to change it.

**Where self-attention is taught, and where it should be.** I checked the section reviewer's finding against the sources and it holds.

- In `multihead-attention.md`, the subsection "Self-Attention" is one paragraph and :eqref:`eq_self-attention`, with $f$ "now multi-head attention", followed by a cell that checks an output shape on `torch.ones`. The word occurs once before that, in passing, in the FLOP count ("For self-attention, write the inputs ..."), 400 lines before the definition. The section's own motivating sentence ("Alice gave Bob the book") is a self-attention scenario used before the term exists (MHA-1).
- The single-head layer with learned maps, $\mathrm{softmax}\big(\mathbf{X}\mathbf{W}_q(\mathbf{X}\mathbf{W}_k)^\top/\sqrt{d}\big)\mathbf{X}\mathbf{W}_v$, is first written in `positional-information.md` as :eqref:`eq_pos-selfattn`, as the object of the permutation proof, one section after the reader needed it. (The reviewer of the first three sections says it is written "nowhere in the chapter"; that is true of those three sections, not of the chapter.)
- The comparison with convolutional and recurrent layers (path length, parallelism) is in the opening of `attention-at-scale.md`.
- `chapter_transformers/transformer-block.md` does not teach it either, and should not have to. It opens with ":numref:`chap_attention` built attention, a layer in which each position reads from all the others" and says each token "queries the others as in :numref:`sec_multihead-attention`". `positional-information.md` and `attention-at-scale.md` cite the same label as the home of self-attention.

So four places cite a home that holds twelve lines, and the three pieces a proper treatment needs are scattered over three sections. **It should be taught in this chapter, in the multi-head section, before the heads**, and that section should be renamed to say so ("Self-Attention and Multiple Heads"). The content: (1) one head with $\mathbf{W}_q, \mathbf{W}_k, \mathbf{W}_v$ and two sentences on why a token needs three roles (the draft in MHA-2); (2) the sentence "Alice gave Bob the book" run through it, with the weights of the query *gave* shown as numbers and a heatmap; (3) the causal mask in this setting, in one sentence, since section 2 has already built it; (4) the remark that the output has the shape of the input, so layers stack; (5) the comparison with recurrence and convolution, moved from the cost section and cut by a third (COST-2). Then "what one head cannot do" follows as the question the worked example raises: *gave* needs three words and has one set of weights. Chapter 11 keeps citing this section and needs no change. If the author prefers a section of its own, the seam is before "A One-Mixture Bottleneck"; I would not take it, because with the cost split below the chapter would reach eight teaching sections.

**Sections that are really two.** `attention-at-scale.md`: the seam is before "Windowed and Sparse Attention" (exact attention and its cost, against what is given up for linear cost); see structural finding 1 of that review. `multihead-attention.md` is three topics, of which one (the grouped-query evidence) leaves and one (the letter demonstration) moves to section 2. `what-attention-computes.md` changes mode in its last third and is better shortened than split.

**Boundaries that leak.**

- The letter-matching demonstration (label `subsec_cross-attention-alignment`) sits at the end of the multi-head section and uses one head, no projections and plain `d2l.DotProductAttention`. It is the worked example section 2 lacks (SCORE-12, MHA-15).
- The softmax Jacobian and saturation are stated in sections 1 and 2 and in exercises of both (QKV-2, SCORE-4). Owner: section 2.
- The bandwidth of section 1 and the factor $1/\sqrt{d}$ of section 2 are the same knob and neither section says so (QKV-6).
- Additive attention and the Bahdanau history arrive 600 lines after the question they answer (structural finding 2 of the scoring review).
- ALiBi and NoPE are introduced inside the experiment of the positional section; the explanation of why their curves are flat is in an exercise and in the further reading (POS-14, FR-8).
- Grouped-query attention, below.

**Grouped-query attention: one owner.** It is taught in `multihead-attention.md` (`subsec_kv-head-sharing`: definition, :eqref:`eq_gqa-head`, multi-query attention, a weight-copy cell, a paragraph of paper results) and again in `chapter_transformers/kv-cache.md` ("Sharing Keys and Values across Heads": definition, multi-query attention, `fig_gqa`, the cache-bytes formula, code, an experiment at four values of $H_\textrm{kv}$, uptraining). The second does not cite the first. **Owner: `kv-cache.md`.** Reasons: the motivation is the cache, which the reader of chapter 10 has not met, so the forward reference here is load-bearing (MHA-10); the figure, the formula and the experiment are already there; and `d2l.GPT` has its own `CausalAttention` with `num_kv_heads`, while no later call of `d2l.MultiHeadAttention` in the book passes that argument (I checked `transformer-block.md` and `encoders-decoders.md`). So the argument can leave the chapter 10 class, which also answers MHA-7 (the first multi-head class the reader sees carries five lines for a feature explained later). What stays in chapter 10 is what follows from its own proposition, in three sentences at the end of the multi-head material: heads need different weights but may read the same keys and values, provided the shared value head is wide enough to carry what is retrieved; with a pointer to `sec_kv-cache`. Exercise 7(c) keeps the test (the bound returns for four heads on one value head of quarter width) and adds the sharing by a subclass. What moves to `kv-cache.md`: the definition, the paper numbers (rounded), and the reference back to `subsec_one-mixture-bottleneck` as the reason sharing is safe. That section's reviewer notes that its own experiment "cannot discriminate" (all four losses within a few hundredths); the copy-both task of chapter 10 is a task on which sharing does discriminate, and one sentence there should say so. The owner must also reconcile two statements that now disagree across the chapters: chapter 10 says a from-scratch ablation found a clear loss, chapter 11 says uptraining "is how GQA spread through existing model families" (KV-11 in that review, believed false). The entry "Sharing Keys and Values" of this chapter's further reading moves to `sec_transformers-further-reading` (FR-4).

## The most important findings

1. **Self-attention is cited as taught here and is not.** Teach it on one sentence with one head and learned projections before the heads; move the layer comparison to it (MHA-1, MHA-2, MHA-13, structural findings 1 to 3 and 5 of the multi-head review, structural finding 6 of the scoring review, COST-2, IDX-3).
2. **The central formula has no worked case.** Three cells in section 2 and two in section 3 end in a shape check on random or constant tensors. Move the letter demonstration to section 2 as the test of `DotProductAttention` and add a three-token example in numbers (SCORE-12, MHA-15, MHA-13, QKV-9, structural finding 1 of the scoring review).
3. **Experiments are read for more than they show, in three sections.** The circuit is said to match one token although the model beats the one-token ceiling (CIRC-11); the positional ranking is confounded and has no baseline, and the flat curves are not explained in the text (POS-11, POS-12, POS-14); the saturation experiment is read as a statement about networks at initialization (SCORE-2); a mechanism is asserted from an unprinted count (CIRC-10).
4. **The final timing comparison of the cost section is unfair and tells a different story in each tab,** and the section ends without a judgment on which mechanism is used in practice (COST-17, COST-18, structural finding 2 of that review).
5. **Paper retellings in place of doing.** The Bahdanau subsection with a schematic "attention map" where Exercise 8(a) specifies the trained experiment (SCORE-18, structural finding 3 of the scoring review); the grouped-query results paragraph (MHA-12); the last third of the circuits section (structural finding 1 there, CIRC-14).
6. **Grouped-query attention has two homes.** Give it to `kv-cache.md` (MHA-10, MHA-11, MHA-12, MHA-7, FR-4).
7. **Split the cost section** before "Windowed and Sparse Attention" and break its one rhythm of six "maximum deviation" checks (structural findings 1 and 3 of that review, COST-12, COST-15).
8. **What multiple heads fix is stated twice with different answers** (weights or width). State the two necessary conditions first and say which experiment shows which (MHA-4, MHA-5).
9. **Notation.** $d$ means model width and head width in the same sections; $\alpha$ is a normalized weight and then an unnormalized kernel; $n$ is batch size and then the number of queries (POS-8, COST-3, QKV-3, SCORE-11).
10. **The introduction's route is a list and the further reading does not rank** (IDX-1, IDX-2, FR-1).

## Focus

- **Surveys and retellings inside teaching sections:** "Attention as Learned Alignment" in section 2; the results paragraph of "Shared Keys and Values" in section 3 (MHA-12); "Extending a Trained Model's Context" ends with four recipes from papers (POS-17), which is acceptable because the section first does the experiment itself; released models in the cost section (keep Mistral 7B and Gemma 3, which illustrate formulas; move gpt-oss, Llama 4 and the FlashAttention-2 and -3 paragraph); about 1,300 words of reporting at equal weight at the end of the circuits section. The reviewer of the last three sections notes that each ends its body with a tour of models or papers before the Summary. `further-reading.md` has matching headings for all of it.
- **Variant lists at equal weight:** four kernels with a four-panel figure of which one is used (QKV-4); five positional schemes named in the second paragraph of that section before any is needed (POS-1).
- **A second topic inside a section:** kernel regression in section 1, never tied back to attention (QKV-6); in-context regression in section 6 (structural finding 2 there, CIRC-14).
- **Openings that give away the section:** the third paragraph of section 2 and the second of section 3 list every result in heading order (SCORE-20, MHA-16).
- **What a reader can skip** is said nowhere (IDX-2).

## Experiments

- **Shape-only cells (cannot fail for almost any wrong implementation):** five. `attention-scoring-batch-matrix-multiplication`, `attention-scoring-the-dotproductattention-class-2`, `attention-scoring-from-alignment-to-attention` (SCORE-11, SCORE-12, SCORE-17); `multihead-attention-one-sequence-querying-itself`, `multihead-attention-one-sequence-querying-another` (MHA-13).
- **Cells that only confirm what was just derived or holds by construction:** `multihead-attention-one-head-must-average-1` (MHA-5), `multihead-attention-shared-keys-and-values` (MHA-11), `multihead-attention-implementation-4`, `attention-at-scale-depth-restores-the-reach` (COST-12), `what-attention-computes-in-context-regression` (CIRC-14), `what-attention-computes-where-and-what-the-qk-and-ov-circuits` (CIRC-5), `positional-information-attention-ignores-order` (POS-3; within the author's allowance of once or twice), `positional-information-sinusoidal-encodings-2` (prints binary numbers, POS-4).
- **Equivalence checks that can fail but repeat one form:** five in the cost section (structural finding 3 there). Keep the one for `chunked_attention`.
- **Redundant cells:** two in the masking part of section 2 (SCORE-9); one band heatmap in the cost section.
- **Read for more than they show:** SCORE-2, POS-11, POS-12, POS-14, CIRC-10, CIRC-11, COST-18.
- **Unfair comparison:** COST-17 (a fused vendor kernel against unfused teaching functions).
- **Stored output against prose:** SCORE-5 (contradiction as printed), MHA-6 (undersold), POS-15, COST-4, COST-10, CIRC-8, POS-3 (the JAX tab prints exactly zero where the prose says round-off). Everything else the reviewers checked agrees in both tabs, and ranges are used at the right precision, notably in the circuits section.
- **Asserted, not run:** the return of the bound for $(H, H_\textrm{kv}) = (4, 1)$ (MHA-11); the sharpening failure of linear attention (COST-15); an output of additive attention that goes unread although it holds the one interesting fact (SCORE-17).
- **Scale.** Sections 1 to 4 and 6 run on a laptop. The cost section needs a CUDA GPU with several GiB free and says so nowhere.
- **The chapter's best habit:** experiments whose outcome is not foregone. The one-block model that solves the fixed-period task and collapses when the period changes; the ablation that shows the unremarkable heads to be necessary; uniform RoPE rescaling that fails where slow-pair rescaling works; the memory prediction $8n^2$ bytes that a measurement then meets. The first three sections have one such experiment (the trained copy-both layer) and should gain a second (the toy alignment of Exercise 8(a) of section 2).

## Repetition

**Inside the chapter.**

| What | Where | Single home |
|---|---|---|
| Softmax Jacobian never zero, saturation | sections 1 and 2, exercises of both (QKV-2, SCORE-4) | section 2 |
| Sharpness of the weights (bandwidth, then $1/\sqrt{d}$) | sections 1 and 2, unconnected (QKV-6) | defined in 1, set in 2, with one back-reference |
| Proposition on one head, confirmed twice | least squares and trained layer (MHA-5) | the trained layer |
| "Chunking changes the order of evaluation" | twice within fifteen lines (COST-7) | the first |
| Evidence chain for the circuit | as it happens, in a paragraph, in the table, in the Summary (structural finding 3 of the circuits review) | the table |
| "Where it looks, what it moves" | four openings in section 6 (CIRC-9) | opening and closing |
| Released-model facts | sections 4 to 6 and the further reading | further reading, except those that illustrate a formula |
| Why flat extrapolation curves are flat | section 4 (three lines), an exercise, the further reading (POS-14, FR-8) | section 4 |
| Statement of the fixed-state problem | introduction (mystery novel) and section 1 (768 numbers) | keep both; the examples differ |

**With other Advanced chapters.** Grouped-query attention with `chapter_transformers/kv-cache.md` (owner: that section, above). Attention sinks are mentioned in sections 3 and 6 and the further reading and measured in `subsec_kv-cache-sinks`; the mentions are short and point there, which is right. The memory-traffic argument of the cost section (one paragraph, COST-8) is owned by chapter 13; the reviewer could not check that the two do not both derive it. Linear attention as a recurrence is set up here and developed in chapter 12, with two forward references that are not load-bearing (down from six in October). Exercise 4 of section 1 (a learned bandwidth) repeats `chapter_recurrent-modern/test-time-regression.md`.

**With Basics.** Batch matrix multiplication is re-taught with a display equation and a cell although the reader has used `@` on batches (SCORE-11). The recurrent-network recap in the cost section uses "recurrent Jacobians" without recalling them (COST-2). Nothing else.

**With the appendix.** None. The chapter has no reference into the appendix at all (I searched the eight sources for `mdl-` and `sec_mdl`, `eq_mdl` labels and found none) and reads without it. Under the author's rule this is the right state: its derivations are short and specific to the chapter. Two places could offer the appendix as optional depth in a clause, if suitable sections exist: the convergence statement of Nadaraya-Watson (QKV-5) and the rotation group in the RoPE derivation (POS-7).

## Recurring patterns

- **Validation by shape:** five cells in sections 2 and 3 (ten `d2l.check_shape` calls over the two tabs). None in sections 1, 4, 5, 6.
- **"Maximum deviation" checks:** five in the cost section in one sentence shape, one weight-copy check in section 3, one in section 6.
- **Roadmap paragraphs that are tables of contents:** introduction (six "asks"), section 2 (SCORE-20), section 3 (MHA-16), section 4 (POS-1).
- **Second endings and Summaries that list everything:** one "We can now ..." recap (SCORE-19; the only one in the chapter); Summaries of seven results in sections 5 and 6 (COST-19, CIRC-18).
- **Paper numbers at the last digit:** MHA-12 (seven numbers), FR-5 (five).
- **Terminology drift:** "weights", "weighting", "distribution over the keys", "mixture" for one row of weights, about 30 occurrences in section 3; "width" and "dimension"; "score", "scoring function", "compatibility function"; "attention pooling" in a heading and label but not in the body (QKV-8); coined "copy-both task", 13 times (MHA-3).
- **Notation drift:** $d$ against $d_h$ in sections 4 and 5 (POS-8, COST-3; section 6 is consistent); $\alpha$ normalized and unnormalized (QKV-3); $n$ as batch size and as number of queries (SCORE-11); $f$ against $\textrm{Attention}(\mathbf{q}, \mathcal{D})$; $c$ against $d_v$; row vectors until section 6, columns there (CIRC-3); the symbol $\overset{!}{=}$ unglossed (POS-6).
- **Tab plumbing in prose:** JAX notes of five to ten lines on `nnx.Intermediate`, a `Linear` class, `d2l.Embed`, memory measurement, weight orientation and TF32 (SCORE-13, MHA-8, COST-5, CIRC-3, CIRC-14). Five pairs of tab notes in the cost section alone. A second imports cell in the PyTorch tab of sections 5 and 6 (COST-1, CIRC-1); unused imports in section 1.
- **Exercises:** strong, heavy, and without warm-ups. About 49% of body length over the chapter (73% in section 1, 67% in section 3, 39% to 47% elsewhere). Every reviewer names exercises that hold a result the body needs: Exercise 6 and 7 of section 2 (the 3.7 nats, why additive attention barely depends on the query), Exercise 1 of section 3 (why not $H = d$), Exercise 4 of section 4, Exercises 6 and 9 of section 5, Exercises 7 and 8 of section 6. No section has a five-minute problem; three reviewers ask for one by hand on a three-token case.
- **No TensorFlow or MXNet material** in any of the eight files.

## Dependencies

- **From Basics:** softmax, the RNN language model, convolution, `d2l.Benchmark`, `d2l.peak_memory`. Adequate.
- **From chapter 9:** `d2l.WikiText103`, `d2l.train_steps`, `d2l.lm_perplexity` (from `sec_adam`), used in sections 4 and 6 without a word on what a step or the data set is. One clause each.
- **From the appendix:** nothing. The chapter reads without it.
- **Into the Attic:** "We build such an encoder--decoder in :numref:`sec_seq2seq`" points to `chapter_natural-language-processing-pretraining/seq2seq.md`, twenty-odd chapters later, as if the reader had built it (SCORE-15).
- **Forward into chapter 11:** `sec_kv-cache` from sections 2, 3 and 4. Load-bearing only for grouped-query attention (MHA-10), which the ownership decision removes.
- **Forward into chapter 12:** two, not load-bearing.
- **Forward into the Language Models part:** none in the teaching sections. Four in the further reading, all without labels (FR-7), none load-bearing.
- **What later chapters take from here:** `d2l.MultiHeadAttention` and the masks (chapter 11), `rope` and `TinyCharLM`, the residual stream of section 6 (the first paragraph of "Components of a Transformer Block"), the online softmax (cited back by the skeleton `chapter_lm-inference/decode-kernels.md`), linear attention (chapter 12). Chapter 11 also takes self-attention from here and finds twelve lines.

## Connection to the rest of the book

- `kv-cache.md` defines grouped-query attention without citing `subsec_kv-head-sharing`; after the ownership change it should cite `subsec_one-mixture-bottleneck` instead.
- `transformer-block.md` cites `sec_multihead-attention` for self-attention and `sec_what-attention-computes` for the residual stream. Both references are correct in direction; the first needs the content to exist.
- The bias default `bias=False` of `MultiHeadAttention` is justified by Exercise 1 of section 2 and by nothing in the body (MHA-7).
- Section 6 keeps the label and file name `what-attention-computes` under the title "Attention Circuits"; section 5 keeps `attention-at-scale` under "The Cost of Attention". Harmless, but the split of section 5 is the moment to align them.

## What to cut

- The batch-matrix-multiplication subsection (SCORE-11) and the two redundant masking cells (SCORE-9).
- The schematic attention map and the narration around it, replaced by the trained toy alignment (SCORE-18).
- Three of the four kernels and their figure (QKV-4).
- The results paragraph on grouped-query attention, the weight-copy cell, and `num_kv_heads` from the chapter 10 class (MHA-12, MHA-11, MHA-7).
- The first confirmation of the proposition (MHA-5).
- The binary-print cell (POS-4) and the API paragraph on `rope_tables` arguments before they are used (POS-9).
- The receptive-field cell (COST-12), four of the five equivalence paragraphs, and the released-model names that illustrate nothing (cost review, structural findings 3 and 4).
- In-context regression as a subsection (to an exercise), the paragraph before the evidence table, and most of the reporting third of section 6 (structural findings 1 to 3 there).
- From the further reading: ASIF and adaptive bandwidths (FR-6), benchmark ratios (FR-5).
- Exercises: the candidates each reviewer names (section 1: 4, part of 6; section 3: 8, part of 5; section 4: 2(c); section 5: 1, 5(c)).

## What is missing

- Self-attention on a sentence, with one head, numbers and a picture.
- A worked numeric case of :eqref:`eq_softmax_QK_V` and a meaningful first run of the class.
- The roles of the three projections, and why queries and keys have separate maps (Exercise 7(d) of section 1 has the argument).
- The link from bandwidth to $1/\sqrt{d}$ (QKV-6), and the effect of dropping the key norm (SCORE-1).
- The four-term expansion of the additive-position score, on which the case for RoPE rests (POS-5).
- A baseline (zero-block or bigram perplexity) for the positional table, and a measured number for what the flat curves read (POS-12, POS-14).
- A two-block numeric instance of the online softmax (COST-6); a cell that shows linear attention cannot sharpen (COST-15); a closing judgment on which remedy is used in practice (COST-18).
- The single-token ceiling of 0.84 and the Bayes value at the restart, both one-line results now in exercises (CIRC-11, CIRC-12); one concrete skip-trigram from text (CIRC-6).
- Reading guidance in the introduction and a ranking in the further reading (IDX-2, FR-1).
- One easy exercise per section.

## What is best and should set the model

- **The first page of `queries-keys-values.md`**: a database of two records, a query that misses, a soft answer of 19.5 the reader can recompute. This is the Illustrated-Transformer manner, and the standard the next two sections should be held to.
- **The dot product derived from the Gaussian kernel**, with the judgment that dropping the key norm "is a modeling decision rather than an approximation", and **the masked-softmax numerics**, where each choice is rejected for a stated reason.
- **The copy-both arc** of the multi-head section: a task on which one set of weights provably loses half the target, two hand-built heads with soft weights, a trained layer that finds the same.
- **The RoPE derivation**: state the requirement, read off the solution. The section then tests an extension whose outcome is not foregone.
- **Costs with numbers**: the crossover $n > 2d$, the predicted and measured $8n^2$ bytes, the tile fractions of a fused kernel, the cache fraction of interleaved windows.
- **The circuits section as a whole**: do the experiment instead of retelling the paper, include the control that fails, report the control that does not discriminate, and end with a table that says what each kind of evidence cannot show. The Bahdanau subsection, the grouped-query paragraph and the cost section's ending should be rewritten by this model.

## Could not check

Claims about papers, model cards and framework APIs were judged from the reviewers' and my own knowledge, not against the sources (MHA-12, POS-17, COST-9, COST-11, COST-13, COST-16, CIRC-13, CIRC-15, FR-5). No code was run with PyTorch or JAX; the numerical checks were NumPy reruns by the section reviewers. JAX outputs were not rendered for section 1. Whether chapter 13 derives the memory-traffic argument a second time was not checked.
