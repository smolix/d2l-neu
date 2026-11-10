# Multi-Head and Cross-Attention

Source: `chapter_attention/multihead-attention.md`, about 3,000 words of body prose, 14 cells, 8 exercises (about 2,000 words).

## Verdict

The section has one original and well-built idea: a designed task (report two stored values) on which a single weighting provably loses half the target, two hand-built heads recover it, and a trained layer finds the same solution. That is better motivation for multiple heads than the usual "different heads look at different things". Around it the section is out of proportion. Self-attention, the form of attention the rest of the book uses and the one a reader from the RNN chapters most needs explained, gets twelve lines, an equation and a shape check on a tensor of ones, at the end, after a long passage on grouped-query attention that the KV-cache section teaches again with a real experiment. Learned projections $\mathbf{W}_q, \mathbf{W}_k, \mathbf{W}_v$ appear for the first time already indexed by head. A reader who knew only the small RNN could reproduce the `MultiHeadAttention` class from this section but could not say what a self-attention layer does to a sentence. The single most important fix: teach self-attention with projections for one head on a short sentence, with a picture and numbers, before multiplying heads, and move the grouped-query material's motivation and evidence to :numref:`sec_kv-cache`.

## Scores

| Dimension | Score |
|---|---|
| Clarity of explanation | 3 |
| Clarity of thought and focus | 3 |
| Pedagogy | 3 |
| Language | 4 |
| Ease and enjoyment | 3 |

## What works

- **The designed task and its three stages** (proposition, hand-built heads, trained heads). The second cell carries the best point: "the second row does so with *soft* weights $(0.88, 0.12)$. The heads need different weightings, not one-hot ones." The trained layer then shows the same (stored weights (0.92, 0.08) and (0.07, 0.93) in PyTorch, (0.71, 0.29) and (0.00, 1.00) in JAX). Preserve this arc.
- **The proof is short and correct.** I checked the conditional mean and the error variances; the errors sum to 1 per coordinate for every $(\alpha_1, \alpha_2)$.
- **Parameter and FLOP counts** (:eqref:`eq_multihead-flops`) are derived, correct ($8nd^2 + 4n^2d$, $4d^2$), and followed by a judgment: "The number of heads changes the representation, not the leading operation count."
- **`fig_head-split`** (rendered) makes the reshape legible: slices of one projection become an axis. Together with Exercise 2(b), the `reshape` that passes every shape check and silently mixes tokens, this is the implementation detail a practitioner needs.
- **The letter demo** reads three distinct patterns off one heatmap (a single match, a split between two copies, a query with no match) and ends on a true and useful sentence: "the softmax must distribute all of its probability mass even when no key matches". I rendered the stored map; it shows what the prose says.
- The numbers for GPT-2 small's cache (18,432 numbers per token, 36 KiB, 36 MiB at 1,024 tokens) and the grouped layer's 163,840 parameters are correct.

## Structural findings

1. **Self-attention is the least taught topic of the section and the most used.** Other sections cite this one as the home of self-attention (`positional-information.md`: "the self-attention layer of :numref:`sec_multihead-attention`"; `attention-at-scale.md` likewise). What they point to is one paragraph and :eqref:`eq_self-attention`, with $f$ "now multi-head attention", checked on `torch.ones`. There is no sentence being attended over, no picture of tokens emitting queries, keys and values, no heatmap of a self-attention layer, no remark on the causal mask in this setting. The comparison with recurrent and convolutional layers (path length, parallelism) exists, but in `attention-at-scale.md`, two sections later. The Illustrated Transformer spends most of its length on exactly this step.
2. **The motivating example uses self-attention before it is defined.** "In the sentence 'Alice gave Bob the book', the verb *gave* needs three other words" is a self-attention scenario: a token of a sentence queries the other tokens of the same sentence. At that point the reader has seen attention only as a query against a database and as a decoder reading an encoder. Six hundred lines later self-attention is introduced as if new. The sentence is also dropped at once in favour of the abstract task and never returns.
3. **Projections arrive only per head.** The chapter never writes single-head attention with learned maps, $\mathrm{softmax}\big((\mathbf{X}\mathbf{W}_q)(\mathbf{X}\mathbf{W}_k)^\top/\sqrt{d}\big)\,\mathbf{X}\mathbf{W}_v$. :eqref:`eq_multihead-head` introduces $\mathbf{W}_i^{(q)}, \mathbf{W}_i^{(k)}, \mathbf{W}_i^{(v)}$ at once. Why queries and keys have different maps (so that "what I look for" can differ from "what I offer"), and what the value map is for (what a token hands over once selected, as distinct from what it is matched on), are never said in the body.
4. **Grouped-query attention is taught twice in the book.** Here: definition, :eqref:`eq_gqa-head`, multi-query attention, an equivalence cell, and a paragraph of results from three papers and a model card. In `chapter_transformers/kv-cache.md` ("Sharing Keys and Values across Heads" and `subsec_kv-cache-gqa-quality`): definition, multi-query attention, the uptraining recipe, a figure, and an experiment that trains the model at four values of $H_\textrm{kv}$. The motivation given here ("Sharing matters when a model generates text one token at a time") rests on a cache the reader has not met, so the forward reference is load-bearing. By the repetition rule one section owns this. The cache section is the natural owner of the motivation and the evidence. What belongs here is what follows from this section's own result: which heads can share without bringing the bound back.
5. **Three topics, one of them misplaced.** With the grouped-query evidence moved out and the letter demo moved to `attention-scoring.md` (it uses no heads and no projections), the section has room to do self-attention properly. Proposed order: (1) one head of self-attention with projections on a short sentence; (2) what one weighting cannot do, the designed task; (3) multi-head attention, counts, implementation, trained heads; (4) sharing keys and values, as a consequence of (2) in two paragraphs and one cell; (5) cross-attention as the same function with other arguments. If the author prefers a split, the seam is before "A One-Mixture Bottleneck": "Self-Attention" becomes a section of its own and takes the layer comparison from the opening of `attention-at-scale.md` with it.

## Passage-level findings

**MHA-1** (major, pedagogy). Opening.
Anchor: `*gave* needs three other words: who gave, who received, and what was given.`
See structural finding 2. The example is good and should be kept and used: define self-attention on this sentence first, show one head's weights for the query *gave*, and then ask what a second head would add. If the order of the section is not changed, at least add "(each word of a sentence can act as a query against the others, an arrangement called self-attention that we define below)".

**MHA-2** (major, explanation). Under "Projected Heads".
Anchor: `them into its own subspace of width $d_h$ and then runs ordinary scaled`
First appearance of learned query, key and value maps, already with a head index and a superscript. See structural finding 3. Insert before it (or at the end of `attention-scoring.md`) the single-head case with three matrices and two sentences on their roles. Draft: "A token must play three roles: it asks (query), it is found (key), and it is read (value). One vector cannot serve all three well, because what a verb looks for is not what a verb offers to others. The layer therefore computes three linear maps of each input, $\mathbf{q}_i = \mathbf{W}_q \mathbf{x}_i$, $\mathbf{k}_i = \mathbf{W}_k \mathbf{x}_i$, $\mathbf{v}_i = \mathbf{W}_v \mathbf{x}_i$, and applies :eqref:`eq_softmax_QK_V` to them."

**MHA-3** (moderate, language). Heading.
Anchor: `## A One-Mixture Bottleneck with Position-Only Keys`
A heading made of two coined phrases that the reader cannot parse before reading the subsection. The section also coins "copy-both task", used 13 times in the body and in two labels. Use a plain heading ("What One Head Cannot Do") and introduce "copy-both" once, as a name for the task, where the task is defined.

**MHA-4** (moderate, thought). Under the proposition.
Anchor: `The bound needs less than this proof used, and it does not depend on the`
After proving the bound from "one weighting", the text says the bound is really a count of dimensions ($c$ numbers in, $2c$ wanted) and holds for any $c$ numbers computed from the values. Both are true (I checked the counting argument: a readout of rank at most $c$ misses at least half of an isotropic target in $\mathbb{R}^{2c}$). But the reader is left unsure what multi-head attention fixes. If the obstacle is width, why not one wider head? The single head of the later experiment has width 16 and still fails, which is the weighting obstacle, not the width one; and the generalization "content-dependent keys ... leave the bound in place as long as the readout ... receives $c$ numbers" does not cover that head once its weights depend on the values. The closing sentence of the paragraph has the right statement ("compute several different weightings for one query, and leave room in its output for every value it retrieves") but arrives after six sentences of hedged generality. Fix: state the two necessary conditions first, as the result. Then say in one sentence each which experiment shows which: one head of full width fails (weightings), and four heads sharing one value head of quarter width fail (room). Move the "any $c$ numbers" generalization to Exercise 6, where it already is.

**MHA-5** (moderate, pedagogy; author decision 4). Under the proposition.
Anchor: `mixture of the two values, whatever they are. We check the bound by fitting`
The bound is proved, then confirmed by least squares at four splits (0.705, 0.705, 0.706, 0.707), then confirmed again by the trained one-head layer (0.71). The first confirmation teaches nothing the proof did not. The cell that follows it (two heads, soft weights) does. Fix: merge the two cells, print one single-head row as the baseline and the two two-head rows, and let the trained layer be the test of the proposition. Minor: the row with weights (1.0, 0.0) is not a softmax pair, which the proposition is stated for.

**MHA-6** (minor, correctness of prose against output). Under "Learned Heads on the Copy-Both Task".
Anchor: `than two orders of magnitude, and the layer returns both values almost`
The stored errors are $1.1 \times 10^{-7}$ and $1.2 \times 10^{-7}$ (JAX $1.2 \times 10^{-7}$, $2.2 \times 10^{-7}$): six to seven orders below 0.71, at single-precision rounding. "More than two orders" undersells a result the cell shows every time. Write "to the rounding error of single precision".

**MHA-7** (moderate, pedagogy). Under "Implementation".
Anchor: ``num_kv_heads` lets groups of heads share keys and values. We return to it in`
The first multi-head class the reader sees already carries `num_kv_heads`, `kv_hiddens`, a second assertion and `repeat_interleave`. Five of its lines serve a feature explained 200 lines later. For a reader trying to hold the layer in memory this is the wrong first version. Options, in order of preference: show the plain forward pass as a short listing in the text (project, split, attend, merge, project) before the saved class; or keep the class and add the grouping lines by `add_to_class` in the sharing subsection. Also unexplained: `bias=False`. Exercise 1 of :numref:`sec_attention-scoring-functions` shows that the key bias has no effect on the weights; one clause here would tell the reader that the default is deliberate and is the convention of current language models. And `repeat_interleave` copies keys and values $H/H_\textrm{kv}$ times, which is fine for teaching but is the opposite of what sharing is for; a comment should say so.

**MHA-8** (minor, language; tab prose). Under "Implementation".
Anchor: `The `Linear` class draws its weights and biases uniformly from`
A JAX-only class and note about initialization stand between the description of the implementation and the implementation. Move the class to where the JAX reader first needs matched initialization (or to the utilities), and keep one sentence.

**MHA-9** (minor, explanation). Under "Parameter and FLOP Counts".
Anchor: `number of heads changes the representation, not the leading operation count.`
If heads are free, why not $H = d$? The answer is in Exercise 1 (a head of width 1 can be sharp only on the largest or smallest key) and it is a result worth a sentence in the body, because it is what actually bounds $H$ in practice.

**MHA-10** (moderate, thought). Under "Shared Keys and Values".
Anchor: `Sharing matters when a model generates text one token at a time. The model`
See structural finding 4. Keep the first paragraph of the subsection (what the copy-both result implies for sharing), the definition in two sentences, and the cache factor $H/H_\textrm{kv}$ with a pointer to :numref:`sec_kv-cache`. Move the cache arithmetic for GPT-2 small to the cache section if it is not already there.

**MHA-11** (moderate, pedagogy; author decision 4). Same subsection.
Anchor: `whose key and value projections are tied within each group. We check this by`
The cell copies the weights of a grouped layer into a full layer and prints that the outputs agree (stored: `0.0e+00`). It confirms a statement that is true by construction of the class. Meanwhile the paragraph above asserts an interesting, falsifiable consequence of the section's own theory ("With four heads of width $d/4$, a single shared value head would carry only half of each value, and the bound ... returns") and leaves it to Exercise 7(c). Swap them: train the layouts $(H, H_\textrm{kv}) = (2,1)$, $(4,2)$, $(4,1)$ on the copy-both task in the body (three lines, given `train_copy_both`), read the errors against the prediction, and print the parameter counts in the same cell. Move the weight-copy check to the exercise.

**MHA-12** (moderate, thought and correctness). Same subsection.
Anchor: `Sharing keys and values lowers quality somewhat, and by how much depends on`
A paragraph of eighteen lines that retells two papers and a model card with seven quoted numbers. It is a survey inside a teaching section (guide 12.2), and the section does none of it. What I can say about the facts: the T5-XXL figures (47.2 and 1.51 s for multi-head, 47.1 and 0.28 s for 8 groups, 46.6 for multi-query, 5% further pretraining) match my memory of Table 1 of Ainslie et al.; the DeepSeek-V2 ablation (MMLU 45.2, 41.2, 37.9 for 7B models) matches my memory of that paper's appendix but I could not confirm it here, nor "at equal parameter count"; gpt-oss with 64 query and 8 key-value heads matches the released configuration as I recall it. All unverified against the sources. "Reading it dominates the time per generated token" is asserted and not shown in this chapter. Fix: replace the paragraph by two sentences ("Large models trained with a few key-value heads lose little; :numref:`subsec_kv-cache-gqa-quality` measures this on a small model and cites the large-scale evidence") and move the numbers to the cache section, rounded.

**MHA-13** (minor, pedagogy). Under "Self-Attention" and "Cross-Attention".
Anchor: `X = torch.ones((batch_size, num_queries, num_hiddens))`
Two cells that check a shape on tensors of ones (the only tests of self- and cross-attention in the section). With every input row equal, every head's weights are uniform and the output rows are identical, so the cells would pass for almost any wrong implementation. Replace with a sentence run through the layer and its heatmaps per head, even untrained, or with the trained copy-both layer's weights, which already exist.

**MHA-14** (moderate, correctness). Under "Cross-Attention between Two Words".
Anchor: `copy-both task on a small scale. The `e` query, whose letter does not occur in`
"This is the averaging of the copy-both task on a small scale" is not right. In the copy-both task the query wants two different values and the average destroys information. Here the two keys are the same letter with the same value (the demo passes the keys as values), so the average of the two values is that value and nothing is lost. What the split shows is a different fact, stated correctly in the previous clause: content alone cannot distinguish two occurrences, so position must enter (the subject of the next section). Delete the sentence or replace it with that link.

**MHA-15** (moderate, thought). Same subsection.
Anchor: `To see cross-attention at work, give every letter of the alphabet a random`
The demo uses plain `d2l.DotProductAttention`, no heads, no projections, and values equal to keys. It illustrates content matching, the topic of the previous section, where it would be the missing meaningful test of the class (see SCORE-12 there). Placed here it ends a section on multiple heads with an example that has one. Move it. What it leaves behind, the attention-sink remark and the permutation remark, are good and can close the cross-attention subsection in three sentences.

**MHA-16** (minor, language). Opening, second paragraph.
Anchor: `same number of parameters and the same leading FLOP count as one wide head,`
The second paragraph of the opening lists every result of the section in nine sentences. As in the previous section, the roadmap is a table of contents in prose. Cut to the question and the method ("we make the question precise with a small task").

Recurring small pattern: four names for one row of attention weights within this section, "set of weights", "weighting", "distribution over the keys", and "mixture" (which is also used for the weighted value). About 30 occurrences in the body ("weighting" 10 times, "mixture" 17). Use "weights" for $\boldsymbol{\alpha}$ and "mixture" only for $\sum_i \alpha_i \mathbf{v}_i$. Also $f$ here against $\textrm{Attention}(\mathbf{q}, \mathcal{D})$ in the first section, and $c$ for the value width against $d_v$ earlier.

## Code and experiments

- **Confirms only**: `multihead-attention-one-head-must-average-1` (MHA-5); `multihead-attention-shared-keys-and-values` (MHA-11, a check that cannot fail short of a bug in the copying code); `multihead-attention-implementation-4` (the parameter count $4d^2$ for four values of $H$, true by the layer's constructor; one line in another cell would do).
- **Shape-only cells**: `multihead-attention-one-sequence-querying-itself`, `multihead-attention-one-sequence-querying-another` (MHA-13).
- **Experiments that teach**: `multihead-attention-one-head-must-average-2` and `multihead-attention-learned-heads-on-the-copy-both-task-2`. The second runs in seconds, is seeded per run only by the framework default, and its prose claims (one head near 0.71; at least two clearly different weightings; not one-hot) hold in both stored tabs. The claim is at the right precision.
- **Stored output against prose**: MHA-6 (undersold). Everything else agrees.
- **Asserted, not shown**: the return of the bound with $(H, H_\textrm{kv}) = (4, 1)$; the running-time cost of many narrow heads; "reading it dominates the time per generated token".
- **Tabs**: the JAX tab adds a `Linear` class and a jitted step function; the shared prose is true of both. The JAX weight-copy code reshapes along a different axis than the PyTorch code (kernels are stored transposed), which is correct but is one more reason to drop that cell.

## Exercises

Eight problems, about 2,000 words, five marked hard: the heaviest set of the three sections, at 60% of the body. Quality is high. Exercise 1(a,b) (width-1 heads can be sharp only on an extreme key; width 2 suffices for any assignment) and Exercise 2(b) (the wrong `reshape`) are among the best in the chapter. Exercise 6(c) (best error $1 - \operatorname{rank}(\mathbf{A})/k$) is the proper general form of the proposition; I checked it.

- **Cut or halve** Exercise 8 ("Averages under key noise"): 55 lines, four parts, depends on an exercise of another section through two cross-references, and its lesson (a second query makes the average robust to key norm) is close to the hand-built two-head cell. Keep (a) and (b) if it stays.
- **Trim** Exercise 5 to (a), (b), (d). Part (c) is a second derivation of the slope found in (b).
- **Move** Exercise 7(c) into the body and 7(d), 7(e) to the cache section together with the material they belong to (MHA-10, MHA-11).
- **Add** a problem on self-attention itself, of which there is none: for a three-token sequence and given $2 \times 2$ matrices $\mathbf{W}_q, \mathbf{W}_k, \mathbf{W}_v$, compute one head's output by hand with and without a causal mask.

## Still open from October

October praised the opening for motivating something "the standard layer does not fix". That tension is now addressed in the text (the output has width $d$, so the task needs $2c \le d$) but is the source of MHA-4: the explanation of what multiple heads fix is split between weightings and width. Exercise volume remains high.

## For the chapter review

- This section is cited by later sections as the home of self-attention and does not teach it. The chapter needs a real treatment, here or as a section of its own, and the layer comparison that opens `attention-at-scale.md` (convolution, recurrence, self-attention; path length; parallelism) is part of that treatment.
- Single-head attention with learned $\mathbf{W}_q, \mathbf{W}_k, \mathbf{W}_v$ is written nowhere in the first three sections. Assign it to the end of `attention-scoring.md` or the start of this section.
- Grouped-query attention has two homes (here and `chapter_transformers/kv-cache.md`). Give the cache section the motivation, the evidence and the paper numbers; keep here only the consequence of the copy-both result.
- The letter demo belongs in `attention-scoring.md`.
- Across the three sections no attention layer is trained on data with meaning; the only training is the copy-both layer. That is defensible (the next section trains a language model) but the three sections together show the reader exactly one learned attention pattern, and it has two keys. A trained toy alignment in the scoring section (its Exercise 8) would be the second.
- Suspected part-wide pattern: paragraphs that report numbers from papers at the last digit (47.2 against 47.1; 45.2, 41.2, 37.9) inside teaching sections, and confirm-only cells of the "copy weights and compare" kind.
