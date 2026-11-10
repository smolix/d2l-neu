# Chapter 11: Transformers

Sources: `chapter_transformers/` (seven sections, an introduction and a further-reading section; about 25,000 words of body prose and about 12,000 words of exercises). Section reviews: [transformer-block](11-transformers/transformer-block.md), [gpt](11-transformers/gpt.md), [kv-cache](11-transformers/kv-cache.md), [encoders-decoders](11-transformers/encoders-decoders.md), [vision-transformer](11-transformers/vision-transformer.md), [moe](11-transformers/moe.md), [scaling-laws](11-transformers/scaling-laws.md), [index](11-transformers/index.md), [further-reading](11-transformers/further-reading.md). Finding IDs below refer to those files; "S1" and the like are the numbered structural findings of the file named with them.

## Verdict

This is a good chapter that is not yet the centre the part needs. Its strengths are real and consistent: every section opens with a problem the reader can feel, the code is compact enough to rebuild from memory (`MiniBlock`, `GPT.forward`, `DecoderBlock`), numbers are computed by hand and then printed, the arithmetic the reviewers recomputed is right almost everywhere, and the exercises are the best in the book so far. Against the bar of MacKay or the Illustrated Transformer it falls short in one recurring way: the experiments are read for more than they show. Seven sections each have a headline measurement, and in every one the reviewers found that the stored output does not support the sentence that reads it, or supports a different and better sentence that the prose does not write. The most serious case is the chapter's central argument. The block section shows that a post-norm stack collapses its tokens at initialization, under Xavier weights at 32 blocks; the GPT section then trains a six-block model with weights about three times smaller, for which the collapse rate per block is 1.01 in place of 2 and the collapse does not occur, presents its run as a test of the first result, and when the post-norm model stalls can only cite a paper for the reason. What matters most, in order: connect or separate those two experiments (TB-5, GPT-13); give `gpt.md` the three things a reader came for and does not find (the loss written down, a recurrent baseline on the same bytes, the cost of a run); make the default tab of `kv-cache.md` show the memory-bound decoding its prose argues for; and say in `encoders-decoders.md` which wiring to use today. The chapter also carries more survey than a teaching chapter should (five passages), teaches several things twice, and promises "generation" and "which wiring" without delivering either.

## Score table

Scores as the section reviewers gave them (explanation / thought and focus / pedagogy / language / enjoyment).

| Section | Expl. | Thought | Pedag. | Lang. | Enjoy. | Most important fix |
|:--|:-:|:-:|:-:|:-:|:-:|:--|
| [index](11-transformers/index.md) | 4 | 3 | 3 | 3 | 3 | Replace the seven "asks" sentences with a route that ranks (IDX-1) |
| [transformer-block](11-transformers/transformer-block.md) | 4 | 3 | 3 | 4 | 3 | State the scope of the collapse experiment (TB-5) |
| [gpt](11-transformers/gpt.md) | 3 | 3 | 3 | 4 | 4 | Recurrent baseline on the same bytes; show the objective (S1, S2) |
| [kv-cache](11-transformers/kv-cache.md) | 4 | 3 | 4 | 4 | 3 | Default tab does not show memory-bound decoding (KV-6, KV-8) |
| [encoders-decoders](11-transformers/encoders-decoders.md) | 4 | 3 | 3 | 4 | 3 | Say which wiring to choose today (ED-1) |
| [vision-transformer](11-transformers/vision-transformer.md) | 4 | 3 | 3 | 4 | 3 | Replace or re-read the position experiment (VIT-7, VIT-8) |
| [moe](11-transformers/moe.md) | 4 | 3 | 4 | 4 | 4 | Separate skew, feedback and dropping (S1, S2) |
| [scaling-laws](11-transformers/scaling-laws.md) | 4 | 3 | 3 | 4 | 4 | Read the grid along equal compute; drop the bend (SL-7, SL-8) |
| [further-reading](11-transformers/further-reading.md) | 4 | 3 | 2 | 4 | 3 | Rank the entries (FR-1) |

The scores do not discriminate: of the 35 scores the section reviewers gave, all are 3 or 4, "thought and focus" is 3 in every section and "language" is 4 in every section. That uniformity is itself a finding (every section has one experiment that is over-read, and every section is written in the same clean, low-voiced register), but it hides a clear order. **Strongest:** `moe.md` (count, build, break, repair, compare, with a mechanism for the failure and an accounting that reproduces published model sizes), the first half of `kv-cache.md` (the best systems teaching in the part, by its reviewer's judgment), and the first two thirds of `scaling-laws.md` (the $6ND$ derivation and the two-fits cell). **Weakest:** `encoders-decoders.md` (four topics, both headline experiments over-read, one of them contradicted by the stored output, and no guidance for 2026), and the pair `transformer-block.md` and `gpt.md` taken together, which matters more because they are the chapter. `gpt.md` is the only section with a 3 for explanation, and the reviewer's reasons (no loss, no shifted targets, a class carrying options for three later sections) are structural, not local.

## Technical correctness and currency

I checked two findings against the sources as asked. **Pre-norm and initialization (TB-5, GPT-13): confirmed.** `signal_stats` in `transformer-block.md` applies `xavier_uniform_` to every matrix of a 32-block stack; `GPT` in `gpt.md` draws from $\mathcal{N}(0, 0.02^2)$ with the writing matrices scaled by $1/\sqrt{2L}$ and has six blocks; exercise 7 of the block section gives the rate $1 + \alpha^4$ per block with $\alpha^2 = \sigma^2 d$, which is 2 for Xavier and $1 + (0.0004 \cdot 256)^2 \approx 1.01$ for the GPT's weights. `gpt.md` line 701 does concede "Its argument came from initialization alone" and its next sentences speak of "a deep stack", so the text does not literally claim the collapse for its own model; the fault is that it never says the measurement needed Xavier weights and 32 blocks, and that the model it is about to train has neither. I did not rerun the reviewer's NumPy simulation. **`chap_nlp_pretrain`: confirmed.** It is the label of `chapter_natural-language-processing-pretraining/index.md`, which `_quarto.yml` lists under the Attic part; `encoders-decoders.md` line 374 and `scaling-laws.md` line 505 cite it, the second explicitly as "the Language Models part". The intended targets exist as skeleton labels: `sec_lm_bert` or `chap_lm_encoders` for the first, `sec_lm_budgets` for the second.

**Errors and unsupported inferences, by severity.**

1. **TB-5 with GPT-13 (major; confidence high).** The collapse of a post-norm stack is a property of post-norm with unit-gain weights at depth 32, not of post-norm. With the next section's initialization there is no collapse, so the training experiment there is not a test of it. Affects the block section's Summary and IDX-3.
2. **GPT-13 (major; high on the stored numbers).** The post-norm run stalls at the unigram level; the reason is cited, not measured; "The third run confirms this" overstates (it changes warmup and decay at once); and the stored output shows post-norm with warmup ending level with pre-norm (1.55 against 1.54), which is the useful conclusion and is not drawn.
3. **KV-13 (major; high).** The low-rank experiment counts 1,536 ambient dimensions for keys and values that are a linear map of a 768-dimensional input. "Rank 512 ... a third of the width" is two thirds. The DeepSeek-V2 comparison has the same fault.
4. **KV-6, KV-8 (major; high, from stored output).** In the PyTorch tab the cached step is limited by kernel launches (7.2 ms against a weight read of about 0.5 ms), decode runs eleven times under the bandwidth ceiling, and batched throughput grows linearly where the formula says it saturates. The shared prose attributes the gap to weight reads. The JAX tab does show the claimed regime.
5. **ED-13 (major; high).** The alignment paragraph cites the wrong figure panel for its prediction and spends twelve lines on failed runs, while the stored outputs in both frameworks show a perfect anti-diagonal (100% of rows). How often the other outcome occurs is unverified.
6. **ED-5 (major; high).** "Conditioning on both sides of a byte lowers its loss by about a third" compares interior positions with two edge positions the masked model has almost never trained on; the chapter's own causal model reaches a lower loss with one side only. Needs a matched causal control.
7. **SL-7, SL-8 (major; recomputed).** The "bend" in the miniature scaling study is one point 0.035 to 0.04 nats off a line through three points; no floor is observed; and flattening would not show a data limit in any case, since a constant term alone bends the curve.
8. **VIT-7, VIT-8 (major; sure of the numbers).** The grid structure in the position embeddings is called "a trace" that an exercise must establish; it is twenty standard deviations from chance. The conclusion that the model has learned "little about its own geometry" does not follow from a cosine curve of a table that is still mostly its initial noise.
9. **ED-1 (major; currency).** The Summary's "encoder--decoders for translation and speech" tells a reader of 2026 that translation calls for an encoder-decoder. Decoder-only models do most text-to-text work now.
10. **TB-15 (moderate; high).** SiLU does not squash, and SwiGLU is not "a learned gate" in the sense of a factor between 0 and 1. Body and Summary.
11. **ED-10 (moderate; high).** "Needs the causal triangle *and* target padding": false under the right padding the paragraph has just specified, and contradicted by the section's own exercise 3.
12. **MOE-1 (moderate; sure).** The "ours" row of the accounting cell is the six-block width-256 model, not the two-block width-128 model "trained at the end of this section".
13. **TB-12, TB-19 (moderate).** The RMSNorm timing shows 0.74 against 0.73 ms in PyTorch and the text then asserts the advantage; the GELU against SwiGLU run shows 0.05 nats in a model still gaining 0.25 nats per 200 steps and is reported in the Summary as a result.
14. **MOE-13, MOE-15 (moderate).** A dose-response claim from two runs; a headline gain of 0.02 nats in the default tab, equal to the stated run-to-run spread, at a capacity factor that doubles the mixture's feed-forward rows while the prose says "at the same computation per token".
15. **GPT-7 (moderate).** The depth-scaling passage gives a rule, no harm it prevents, a cell that confirms arithmetic, and then half withdraws the rule; "as GPT-2 does" is the usual reading of an unreleased training code (unverified).
16. **ED-14 (moderate; mechanism confident, unmeasured).** The flat Perceiver timing is launch overhead, not $O(M^2)$ latent processing. Same fault as KV-6.
17. **SL-14, ED-7 (moderate and minor; confirmed above).** References to the retired chapter.
18. **FR-2 (moderate; checked).** Further reading says OLMo 2 uses "a fifth" placement beyond the four of `fig_norm-taxonomy`; the figure's caption lists OLMo 2 as one of the four.
19. **IDX-3, IDX-4 (moderate, minor).** The introduction says normalization "must sit" in one place "for a deep stack to train" and that a stack can "diverge in training"; the chapter shows neither.
20. **Minor:** TB-3 (positions are not supplied "around the blocks" under RoPE), TB-10 (the probe sentence describes one arm), TB-11 (claims agreement with Xiong et al. for a different observation), GPT-15 (samples exceed the trained context), GPT-17 ("five settings"), KV-14 and KV-15 (causes asserted for differences the experiment cannot resolve), KV-17 (the table says a window needs training, the experiment above it used none), ED-2 ("strongest way to represent"), ED-15 and TB-12 (`Benchmark` cited from chapter 13 when Basics defines it), VIT-4, MOE-10, SL-4, SL-13, SL-17.

**Claims about papers and systems that the reviewers believe but could not confirm.** TB-13 (QK-norm in Henry et al. used $\ell_2$ normalization with a learned scale, not RMSNorm), TB-14 (OLMo 2 and Gemma 3 placements), GPT-7 (GPT-2's $1/\sqrt{N}$), GPT-12 (WebText size, $10^{25}$ FLOPs), KV-11 (GQA "spread through existing model families" by uptraining; the reviewer believes this false, since Llama 2 70B, Mistral 7B and Llama 3 trained with it from the start), KV-16 (gate result "from 47% to 5%"), VIT-11 (seven accuracies from Hassani et al.), MOE-18 (1.81 against 1.87), SL-11 (attribution of the Kaplan and Chinchilla gap), SL-15 (OLMo 2 token count), SL-16 (several table entries), FR-3 and FR-7. These should be checked against the papers in one pass before the revision.

**One inconsistency across chapters that no section reviewer could see.** `chapter_attention/multihead-attention.md` (lines 602 to 613) reports that 7B models trained from scratch lose four MMLU points with 8 key-value groups and seven with one, and says the cost of sharing "depends on how the model was trained". `kv-cache.md` (lines 690 and 773 to 776) says sharing comes "at a usually slight cost in quality" and that GQA at $H/8$ "comes close to multi-head attention". The book gives two readings of the same evidence fifty pages apart.

**Numbers that no cell prints.** TB-19 ("stays ahead when we repeat the comparison"), KV-4 ("about a quarter of the time of a decode step"), VIT-6 (the learning-rate sweep), MOE-14 (outcomes at $c = 1.25$), ED-11 (an instability no cell shows), and the running time of every experiment in the chapter (`gpt.md` S5).

**Currency.** The components taught (pre-norm, RMSNorm, SwiGLU, RoPE, GQA, QK-norm named, MoE with bias balancing, Chinchilla and its refit) are those of open models of 2024 and 2025, and `scaling-laws.md` is current to late 2025. Superseded practice presented as current appears in three places: the wiring guidance (ED-1), token dropping presented as intrinsic to mixtures of experts (MOE-5), and uptraining as the route by which GQA spread (KV-11). The table of deployed configurations (SL-16) will age faster than anything else in the chapter and carries no date.

## Arc and order

The chapter has one line of argument and its introduction states it well: one block, and then a sequence of problems of depth and scale, each answered by one change to that block. The reader builds a block, stacks it into a language model, makes generation affordable, changes the mask and the wiring, changes the input, changes the feed-forward sublayer, and sizes the run. Every section does build on `d2l.TransformerBlock` and `d2l.GPT`, which is the chapter's main structural merit.

**Order.** Two reviewers independently recommend placing scaling laws before the mixture of experts (`moe.md` S5, `scaling-laws.md` S4): scaling laws need only the dense GPT, while the mixture section's argument that "the gain needs data" and its exercise 7 lean forward on tokens per parameter. I agree. The other positions are right.

**Sections that are really two.**

- `transformer-block.md` is one and a third: `CharLM` and the GELU against SwiGLU run (lines 632 to 756) are a language model built one section early (S1 of that file). End the section at "Shapes and Parameters".
- `kv-cache.md` is two: the cache with its cost model, and methods that shrink it. Seam at `## Sharing Keys and Values across Heads` (S1 of that file).
- `scaling-laws.md` is one section plus the chapter's closing survey. Seam at `## Configurations of Deployed Models` (S2 of that file, SL-16).
- `gpt.md` has an optional seam before `## Loading GPT-2` (S6 of that file); its reviewer would rather cut 600 to 800 words of retold material than split.

A chapter revised on these lines would read: block; GPT; the KV cache; shrinking the cache; encoders, decoders and cross-attention; vision transformer; scaling laws; mixture of experts; a short close on the modern recipe (the configurations table, dated, with a judgment about which columns matter). That is nine sections where there are seven, each shorter. The alternative for the close is to fold the table into `further-reading.md`; I prefer the short section, because a reader who has built every component deserves to see them assembled, and because the introduction's claim that "each setting is a constructor argument" is cashed there.

**One caution about load.** The reviewers of the first two sections agree that the language-model material moves from the block section into `gpt.md`, and the reviewer of `gpt.md` also asks that section to add the objective, a shifted-target cell, a mask picture, a recurrent baseline and a mechanism measurement. That is right in substance, but `gpt.md` is already the longest section. The additions fit only if the retold passages go (GPT-8, GPT-9, GPT-16), the JAX notes become comments (GPT-6), and the lean attention class replaces the present one (S3 of that file). If they do not fit, take the split before "Loading GPT-2".

**Leaking boundaries.** The causal mask, tied head and learned positions appear in `CharLM` before `gpt.md` introduces them; the block's parameter count is derived in both sections (TB-18, GPT-8); `adamw` and a model-construction shorthand are defined in both (TB-21, GPT-11); $6ND$ is derived in `gpt.md` and again in `scaling-laws.md`; sampling logic is written in `gpt.md` and again in `kv-cache.md`; the released-model table of `moe.md` overlaps the configurations table of `scaling-laws.md`. Conversely two boundaries are too tight: `encoders-decoders.md` never mentions the cache that the previous section spent 6,800 words on (ED-12), and it does not hand the vision transformer the two things it needs, pooling to one vector (ED-2) and learned queries as the interface to a language model (ED-14).

**What this chapter should own, given the attention chapter's review.** The reviewers of chapter 10 found that self-attention is cited from several places as taught in `sec_multihead-attention` and is there one paragraph and one equation, with single-head attention under learned projections written nowhere, and that grouped-query attention is taught in both chapters.

- *Self-attention.* It belongs to chapter 10, and this chapter should not repair the gap by teaching it again. `transformer-block.md` takes attention in one clause ("Each token queries the others as in :numref:`sec_multihead-attention`"), which is the right amount only if chapter 10 delivers one head of self-attention on a short sentence, with $\mathbf{W}_q, \mathbf{W}_k, \mathbf{W}_v$, a picture, and the comparison with recurrent and convolutional layers. What this chapter owns is self-attention at work: the residual stream and what the block adds around the layer (block section); the causal mask in a language model, shown as a small printed mask beside the shifted targets, since in `GPT` it is hidden inside a fused kernel (`gpt.md` S2); what removing the mask buys (encoders and decoders); and what self-attention costs during generation (the cache). If chapter 10 is not revised first, the block section needs a half-page recap of self-attention at its start; that is the fallback, not the plan.
- *Grouped-query attention.* This chapter owns it whole, in the cache section or the new "shrinking the cache" section: definition, the equation, the constructor argument, the evidence and the paper numbers. Three things follow. The first multi-head class of chapter 10 and `CausalAttention` in `gpt.md` both lose `num_kv_heads` (the reviewers of both files ask for this independently: `gpt.md` S3 and the chapter 10 review), so the option appears where it is explained. The two readings of the evidence (above) are reconciled in one place, with the from-scratch ablation included, since it is the less comfortable result. And KV-12, which says the four GQA training runs here could not have come out otherwise, has its remedy in chapter 10 already: the copy-both task there is a task on which one shared weighting provably fails, so the cache section can cite that result for "why some key-value heads must remain" instead of running a null experiment.

## The most important findings

1. **The argument for pre-norm is made twice, under two initializations, by two experiments the text does not connect** (TB-5, TB-8, TB-11, GPT-13, GPT-7, IDX-3, FR-6). Put the three-sentence derivation of exercise 7 and its scope into the block section; in `gpt.md`, measure the token spread and the gradient norms of the stalled and the warmed-up post-norm runs over the first hundred steps, state what was found, and draw the conclusion the stored output already supports: at this depth post-norm is as good once it trains, and what pre-norm buys is robustness to initialization and schedule. One editor for both sections.
2. **`gpt.md` does not let the reader answer the question the part poses** (S1, S2 and S5 of that file; IDX-2). No recurrent baseline on the same bytes and budget, no loss or shifted targets shown, no running time. The baseline is an experiment for the development machine and its outcome is open; an honest "close in loss at 128 bytes, far apart in speed and at longer context" would serve chapter 12 better than silence.
3. **The cache section's default tab does not show its thesis** (KV-6, KV-8, KV-9, IDX-5), and the same misreading of launch overhead recurs in ED-14. Redesign with a model whose weight read is large against the launch cost, or compile the step; until then say in shared prose what the solid curves show.
4. **Experiments built on a quantity that is easy to compute are read as if they measured what the model knows or why** (ED-5, ED-13, VIT-7, VIT-8, SL-7, SL-8, MOE-9, MOE-12). In four of these cases the decisive experiment already exists as an exercise (the patch permutation of `vision-transformer.md` exercise 3, the frozen router of `moe.md` exercise 4, the composed attention maps of `encoders-decoders.md` exercise 4, the sixth size of `scaling-laws.md` exercise 6), and in one it exists in the stored output unread (equal-compute comparisons in the scaling grid, S1 of that file).
5. **The low-rank cache experiment miscounts its ambient dimension** (KV-13). A one-sentence correction that makes the result clearer.
6. **No judgment about current practice where the reader most needs it** (ED-1, kv-cache S2 and KV-19, SL-16, FR-1). Which wiring; which of five cache reductions matters; which columns of the configurations table matter; which further reading to read. The chapter is candid about its own experiments and silent about the field.
7. **Library classes are shown with every later option attached** (TB-17, `gpt.md` S3, GPT-5, MOE-6, and the chapter 10 finding on `num_kv_heads`). The reader trying to rebuild the model from memory has to work out which half of the code is the model. Show the lean version; add options where they are taught.
8. **Small differences in still-improving models are reported as results** (TB-19, MOE-15, KV-14, KV-12). The chapter never gives the run-to-run spread of its small GPT where the reader can find it; one sentence in `sec_gpt`, cited from the block, cache, mixture and scaling sections, would let every such number be weighed (the suggestion is in the mixture review's notes for this file).
9. **References to a retired chapter and to a section two chapters ahead** (SL-14, ED-7, TB-16, TB-12, ED-15).
10. **Survey inside teaching sections** (TB-14, KV-16, KV-17, ED-14, VIT-5, VIT-11, MOE-17, SL-16). See Focus.

## Focus

Passages that retell papers or list variants without the section building or testing them:

| Passage | Finding | Recommendation of the section reviewer |
|:--|:--|:--|
| QK-norm and the four placements (`transformer-block.md`) | S3 of that file, TB-14 | Teach QK-norm with a ten-line demonstration, or cut to three sentences; taxonomy to further reading |
| Sink-free models, gated attention (`kv-cache.md`) | KV-16 | One sentence; rest to further reading |
| Five-row comparison table with an "evidence" column (`kv-cache.md`) | KV-17 | Three taught rows |
| Serving paragraph (`kv-cache.md`) | KV-10 | One sentence and the reference |
| Perceiver and Perceiver IO (`encoders-decoders.md`) | S2 of that file, ED-14 | One paragraph with its two real uses, or an experiment (exercise 8) |
| Register tokens (`vision-transformer.md`) | VIT-5, VIT-13 | Two sentences |
| The comparison at other scales (`vision-transformer.md`) | VIT-11 | Keep, shorten, conclusion first |
| Fine-grained and shared experts, table of four models (`moe.md`) | S4 of that file, MOE-17, MOE-20 | Keep the idea and the figure; table out |
| Configurations of deployed models (`scaling-laws.md`) | S2 of that file, SL-16 | Own short closing section |

Together these are roughly 2,500 words. Most of them are accurate and several are well told; the fault is position and weight. The proposed destination for five of them is `further-reading.md`, which can take them only if it gains a ranking (FR-1); otherwise the survey material changes address and nothing else.

## Experiments

**Cells that only confirm or cannot fail.** `transformer-block-rmsnorm` (agreement with the built-in), the two parameter-count cells of the block section (TB-18), `gpt-from-blocks-to-a-language-model-2` and `-3` (GPT-7), `encoders-decoders-assembling-the-masks-1` (ED-9), `encoders-decoders-learned-queries-1` (a shape print), `kv-cache-compressing-the-cache-further-2` (a helper self-test), and the equivalence cell for grouped layers in chapter 10. None is long; together they are a habit.

**Experiments that could not have come out otherwise.** The four GQA training runs (KV-12) and, by its reviewer's account, the GELU against SwiGLU run at 600 steps (TB-19).

**Experiments read for more than they show.** Listed under Technical correctness: TB-5, TB-12, TB-19, GPT-13, KV-6, KV-8, ED-5, ED-13, ED-14, VIT-8, MOE-9, MOE-13, MOE-15, SL-7, SL-8.

**Experiments read for less than they show.** These are cheap gains. TB-9: the post-norm gradient curve falls smoothly from $10^{-2}$ at block 1 to $10^{-9}$ at block 32, so the lower blocks train through thirty collapsed ones; the starvation is local. GPT-13 (d): post-norm with warmup ends level with pre-norm. GPT-7 (c): even with depth scaling the stream is seven times the embedding at initialization. ED-5: the loss falls to the interior level within three positions of the edge, so a byte is predicted mainly from its immediate neighbours. `scaling-laws.md` S1: at equal compute the small model trained long beats the large model trained briefly by 0.05 to 0.16 nats, which is the Chinchilla trade-off in miniature and needs no new runs. SL-4: the FLOP counter lands on the attention correction to three digits. VIT-10: the CNN leads from the first epoch.

**Stored outputs against prose.** ED-13 (perfect alignment stored, failures described); KV-6 and KV-8 (PyTorch); TB-12; MOE-1; the two frameworks disagree in substance in KV-6, KV-8 and in `moe.md` S1 (the unbalanced model is far more concentrated after training in PyTorch, barely in JAX).

**Unfair or confounded comparisons.** ED-5 (edge against interior), GPT-13 (b) (warmup and decay changed together), MOE-15 with `moe.md` S3 (capacity factor 2 doubles the mixture's rows), `moe.md` S2 (wasted capacity against skipped computation).

**For the list of experiments to redesign on the development machine:** the recurrent baseline for `gpt.md`; the mechanism measurement for GPT-13; a memory-bound decode in eager PyTorch (KV-8); a causal control for ED-5; the alignment experiment (ED-13, either option); the patch permutation and zero-position model (VIT-8); the frozen-router run and a no-drop evaluation (`moe.md` S1, S2).

**The model to copy** is the comparison of the vision transformer with a CNN in `vision-transformer.md`: same parameters, same FLOPs, same recipe, each model tuned over the same three learning rates, the confounds named, and the easy conclusion refused.

## Repetition

**Inside the chapter.**

| Taught more than once | Where | Proposed single home |
|:--|:--|:--|
| Parameter count of a block ($4d^2 + 8d^2$) | Three times in `transformer-block.md`, again in `gpt.md` and its exercise 1 (TB-18, GPT-8) | "Shapes and Parameters" of the block section; `gpt.md` owns the whole-model count |
| $6ND$ | `gpt.md` (`eq_gpt-six-nd`) and `scaling-laws.md` (`eq_six_nd`) | See the contradiction below; my recommendation is `scaling-laws.md`, with the rule stated and used in `gpt.md` |
| Causal mask, tied head, positions | `CharLM` and `gpt.md` | `gpt.md` |
| "Attention computes no nonlinear feature of a position" | Four times (TB-1) | Once in the block section, once in its Summary |
| The byte corpus and why not *The Time Machine* | `gpt.md`, `moe.md`, `scaling-laws.md` | One sentence in `gpt.md` citing `subsec_tinylm` |
| The `LMOptimizer` recipe | Three sections | `gpt.md` |
| `adamw` and model shorthands | `transformer-block.md`, `gpt.md` (TB-21, GPT-11) | The library |
| Temperature and top-$k$ code | `generate` in `gpt.md`, `sample` in `kv-cache.md` | One saved helper |
| Fixed launch cost per step | Four times in `kv-cache.md` (KV-9), again in ED-14 | Where the timing plot is first read |
| `Benchmark` described | `kv-cache.md`, `encoders-decoders.md` (ED-15) | A clause with the Basics reference |
| Registers; DeiT; continuous batching; Clark and Krajewski | Section and `further-reading.md` | `further-reading.md` |
| Released-model tables | `moe.md`, `scaling-laws.md` | The closing section |

**With other Advanced chapters.** Grouped-query attention with chapter 10 (home: this chapter, see above). The corpus rationale with `chapter_optimization/adam.md`, with different figures for the same book (100,000 characters and 80 passes there, 170,000 and fifty here; GPT-9). `CharLM` is the third toy transformer language model before the GPT, after `TinyLM` in chapter 9 and `TinyCharLM` in chapter 10 (`transformer-block.md` S1); with `CharLM` gone there are still two, and the part overview should ask whether chapters 9 and 10 can share one.

**With Basics.** Temperature and top-$k$ are told a third time (GPT-16; owner `sec_decoding`, which itself says they were met in `sec_rnn-scratch`). Bits per byte and the BPE tokenizer are cited, not retold, which is right.

**With the appendix.** None. The chapter has one reference into the appendix (Eckart-Young, `sec_mdl-svd-low-rank`, from the low-rank cache experiment), which points to the section that holds the theorem and is not needed to follow the experiment. The optimum of the Chinchilla law is a two-line calculation and is rightly kept in the main text. By the author's rule nothing here needs to move.

## Recurring patterns

- **Explanations deferred to exercises.** Five instances in the first two sections (TB-23, GPT-14), three unnamed pointers in the cache section (KV-18). Pointing to an exercise for an extension is fine; using one to close a gap the text has just opened is not.
- **Exercises that hold the section's best material.** Exercise volume is 40 to 70% of body length in every section (49% in `encoders-decoders.md`, about 70% in `vision-transformer.md`) against the guide's 30%. Their quality is high throughout; the fault is that the body's experiment is often the weaker one and the exercise the decisive one (most important finding 4). Promote, do not cut.
- **Tab prose as a second explanation, far heavier in JAX.** About 400 words in `gpt.md` (GPT-6), 630 in `kv-cache.md` (KV-3, S4 of that file), 170 in `scaling-laws.md`. `transformer-block.md`, `vision-transformer.md` and `encoders-decoders.md` have little or none, which shows it can be done.
- **Eager-PyTorch timings that measure dispatch.** KV-6, KV-8, ED-14, TB-12.
- **"We can now" closings and double endings.** TB-22, `encoders-decoders.md` S4, SL-19. The cache and vision sections have one ending; copy them.
- **Announcements and counts that do not match.** "Four ingredients" of which one is the blocks (GPT-1); "five settings" (GPT-17); "Two checks support this reading" followed by three (SL-9); "a fourth use of cross-attention" (ED-16); "each of its four parts addresses a problem met there" (GPT-10).
- **The roadmap template.** Seven "asks" sentences in the introduction (IDX-1).
- **Closing survey tables.** `kv-cache.md`, `moe.md`, `scaling-laws.md`.
- **Notation drift.** Positions counted from 1 in `gpt.md` and from 0 in the block section and the code (GPT-3); $N$ has three meanings inside `scaling-laws.md` (SL-3); two labels for one equation (`eq_gpt-six-nd`, `eq_six_nd`); "character model" for the byte-level model (KV-17); 1.4% in prose against a printed 1% (SL-2); "ridge" used in an exercise and never defined (KV-7).
- **Voice.** The reviewers agree that the prose is plain and exact and has lost the October stagecraft, and that it is low on judgment, most of all in the second halves of sections (KV-19). The sentences they single out to protect show what the register could be: "A large model offers no second try."; "a transposed square matrix still multiplies"; "generation is the training objective run forward"; "It does not."
- **Figures.** Three house-style defects: small labels in `mdl-transformers-vit.svg`, a label on an arrow in `mdl-transformers-moe.svg`, and labels crossing lines plus a caption that describes a budget line not drawn in `mdl-transformers-chinchilla.svg` (in the vision, mixture and scaling reviews). A roofline sketch (KV-7), a two-stack block diagram (ED-17) and a picture of the collapse (`transformer-block.md` S5) are the figures the chapter lacks.

## Dependencies

**What the chapter needs.** From Basics: layer normalization, residual connections and ResNet (`sec_resnet`), Xavier initialization, the RNN language model and its perplexity, BPE and bits per byte, `sec_decoding`, `Benchmark`. From chapter 9: AdamW, warmup and cosine decay, clipping, `train_steps`, `lm_perplexity`, `WikiText103`, what not to decay, the width scaling of `sec_scaling`. From chapter 10: multi-head attention with masks, `causal_mask` and `padding_mask`, RoPE and its tables, the residual-stream view of `sec_what-attention-computes`, the cost model of `sec_attention-at-scale`. All these labels resolve and point where the content is, with the exceptions below. The heaviest dependency is the one chapter 10 does not yet honour: self-attention.

**The appendix.** The chapter reads without it. One reference, correct and optional.

**Wrong or stale references.** `chap_nlp_pretrain` twice (SL-14, ED-7). `sec_perf_model` cited as the home of `Benchmark`, which Basics defines (TB-12, ED-15). `sec_lstm`, a section of chapter 12, used to explain gating as if known (TB-16). The third panel of `fig_three-wirings` cited for a prediction it does not make (ED-13). The section label `sec_transformer` for encoders and decoders, a leftover (IDX-6).

**Forward references inside the part.** `sec_fast_transformer` and `sec_compilation` (chapter 13) for JAX plumbing and for the compile-once discussion; only the PyTorch note in the cache section leans on the latter. `chap_modern_rnn` from the introduction and further reading, as a pointer.

**Forward references to the skeleton chapters.** None is load-bearing. They are frequent: about 18 mentions of "the Language Models part" across the chapter, ten of them in `further-reading.md` (FR-4), almost all without a label. Labels exist (`chap_lm_pretraining`, `chap_lm_encoders`, `chap_lm_serving`, `sec_lm_bert`, `sec_lm_budgets`); use them or drop the pointers. In the other direction, the skeleton `chapter_lm-encoders/bert.md` already plans to build on "a bidirectional encoder from `sec_transformer` trained on characters", so the masked-model part of `encoders-decoders.md` is a dependency of the next part and should not shrink.

**Connection to the main text, in the other direction.** The part introduction frames the part as the road from the recurrent network of the Basics to current language models; the chapter that completes that road never mentions the network (`gpt.md` S1, IDX-2). Chapter 12 (state space models) will want the comparison from the other side; the recurrent baseline in `gpt.md` serves both.

## What to cut, what is missing

**Cut or shrink.** `CharLM` and its run from the block section (moves, in better form, to `gpt.md`); the RMSNorm timing loop; the retold parameter count, corpus rationale and sampling paragraphs of `gpt.md`; the JAX plumbing blocks, to comments; the GQA training runs (or replace); the gate paragraph and two rows of the cache table; the mask check and the Perceiver timing; the register subsection; the capacity-factor paragraph and the model table of `moe.md`; the three-point extrapolation and the utilization sentences of `scaling-laws.md`; the "Pretraining Data" and "Loss and Abilities" entries (FR-8); the pre-Summary recaps. The reviewers' estimates add up to about 3,500 words.

**Missing.**

- The training objective written down, a batch with its shifted targets, and a small causal mask (`gpt.md` S2).
- A recurrent baseline on the same bytes (`gpt.md` S1).
- The cost of the chapter's runs in seconds and memory (`gpt.md` S5, IDX-2), and the run-to-run spread of the small GPT.
- The scope and the derivation of the collapse rate in the main text, and a picture of the collapse (TB-5; `transformer-block.md` S2, S5).
- Initialization as a topic of the block section: three schemes are in play and the reader is told about one at a time (`transformer-block.md` S4).
- A subsection "Choosing a Wiring", and the fact that decoder-only models do sequence-to-sequence work by concatenation (ED-1; the substance is already in `further-reading.md`, see FR and structural finding 2 of that review).
- Pooling per-token outputs to one vector (ED-2).
- Cached decoding in the encoder-decoder, in two sentences (ED-12).
- Prefix caching across requests, in one sentence (`kv-cache.md` S3), and an honest title for a section that contains no decoding methods.
- A roofline figure (KV-7).
- Whether the trained vision transformer uses position, and one sentence on why vision transformers matter now: they are pretrained and serve as the image encoders of vision-language models (VIT-8, `vision-transformer.md` S4).
- That token dropping is a choice of this implementation (MOE-5).
- Kaplan's answer, what a scaling law does not predict, and a unit for "0.05 nats" (`scaling-laws.md` S5, SL-12).
- Any look inside a trained model of this chapter (an attention pattern, or a reference to where chapter 10 shows one).

## What is best

- **Openings that produce the design from a concrete case.** "Given 'the European lobste', it must predict 'r'" (`gpt.md`); the assistant that would reread the whole question for every word (`kv-cache.md`); "A large model offers no second try." (`scaling-laws.md`); "the integral of" against "Act I, Scene" (`moe.md`). Every section has one. This is the chapter's signature and the model for the part.
- **Reference levels for a loss.** Uniform, unigram, bigram, model, and then the stalled run read against the unigram level (`gpt.md`). A number is never left alone.
- **Hand-computed numbers that a cell then prints, in a check that could fail.** The byte formula of the cache (72 KiB per token), the FLOP counter against $2ND$ with the attention correction, the accounting that reproduces Mixtral's and DeepSeek-V3's parameter counts.
- **The two-fits cell** of `scaling-laws.md`: a famous number that does not follow from its own published constants, shown in one cell, with the right caution after it.
- **The sink experiment** of `kv-cache.md`: perplexity 36, then 3,836 with a window, then 47 with one retained token, on a real model at laptop scale.
- **The matched-budget comparison** of `vision-transformer.md`, with its confounds named and its conclusion limited to what was measured.
- **The measurement at initialization as a method** (`transformer-block.md`): "a fixed random function, so we can *measure* how it treats signals", with the candid note on why the obvious probe would have zero gradient. The idea is first-rate even though its result needs its scope.
- **Loading GPT-2 into the class just written**, with the sentence that shape checks do not validate a checkpoint.
- **Exercises** that ask for a prediction, break the implementation on purpose, or prove something small and real: the tied head without blocks (`gpt.md` 7), time's arrow (`encoders-decoders.md` 6), width rounding (`transformer-block.md` 4), the Swin reach argument (`vision-transformer.md` 7), expected distinct experts per decode step (`moe.md` 2), the drift of the tokens-per-parameter ratio (`scaling-laws.md` 8).

## Where the section reviews disagree

- **Home of $6ND$.** GPT-12 would keep the derivation in `gpt.md` ("where the reader has just paid for a run") and have `scaling-laws.md` cite `eq_gpt-six-nd`; `scaling-laws.md` S3 would have that section own the derivation and `gpt.md` cite it. I side with the second: the scaling section gives the reason for the factor, the size of what it omits, and a check that could fail; `gpt.md` needs only the rule and its table of four runs.
- **Growth of the residual stream.** Not a disagreement between reviewers but between sections, noticed from both sides (TB-8, GPT-7 (b)): the block section says the growth "stabilizes the stack", the GPT section scales it away. Both reviewers propose the same resolution (each branch's contribution should be small relative to the stream).
- **Attribution of the Kaplan and Chinchilla gap.** SL-11 calls the section's attribution to Pearce and Song contested; `further-reading.md` presents the two analyses as complementary and the matter as closed (FR-3). The two passages should be revised together.
- **Calibration.** No reviewer is visibly harsher than another; the reviewers gave the same two columns the same score in every section, which is discussed under the score table.

Otherwise the reviews agree, and several findings were reached independently from two sides (TB-5 and GPT-13; the order of the last two sections; `Benchmark`; `num_kv_heads`; the retired chapter label).
