# Generation and the KV Cache

Source: `chapter_transformers/kv-cache.md` (read to the `<!-- slides -->` marker at line 1356). About 4,350 words of shared prose, 630 words of per-framework tab prose, 1,830 words of exercises; about 6,800 in all. Stored outputs read: `outputs/pytorch/chapter_transformers/kv-cache.json` and the JAX file, plus the two stored plots.

## Verdict

This is a strong section with one structural problem and one experimental one. The first half (why the cache is valid, how to build it, how large it is, why decode is limited by memory traffic and prefill by arithmetic, why batching saturates) is the best systems teaching I have seen in this part: a felt problem, a derivation that fits in a paragraph, a number computed by hand and then printed by a cell, and a throughput formula the reader can use the next day. It is not yet at the bar because (a) the file is two sections, "the cache and what it costs" and "shrinking the cache", joined at `## Sharing Keys and Values across Heads`, and the second half is a tour of five methods of unequal importance told at nearly equal weight; and (b) in the PyTorch tab, which is the default reading, the measurements do not show the regime the prose argues for: decode runs eleven times below the bandwidth ceiling and batched throughput grows linearly where the formula says it saturates, because the step is limited by kernel launches. The shared prose attributes the gap to weight reads anyway. The single most important fix is (b): make the default tab's experiment show memory-bound decoding, or stop saying that it does. A third item needs a correction rather than a rewrite: the low-rank experiment measures against a width of 1,536 when keys and values lie in a 768-dimensional subspace by construction.

On the two questions asked. Is the weight proportionate? About 2,900 words on the cache, its size and the cost model: yes, this is what every later discussion of serving, MoE and long context leans on. About 1,900 words plus six long exercises on compression: no, for a chapter on base architecture; GQA earns its place, the rest should shrink. Is the section really two, caching and decoding or sampling? It is two, but the seam is not there. The section contains no decoding or sampling material beyond a ten-line `sample` helper; temperature and top-$k$ are owned by `subsec_gpt-sampling` and search by `sec_decoding`. The title's "Generation" therefore promises more than the section holds. The seam is between the cache with its cost model and the methods that shrink it.

## Scores

| dimension | score | note |
|:--|:-:|:--|
| Clarity of explanation | 4 | The cache argument, the byte formula and the batching formula are each clear on first reading. The roofline paragraph is dense, and tab prose interrupts the argument three times. |
| Clarity of thought and focus | 3 | Two sections in one file; a five-row comparison table that surveys two methods the section does not teach; default-tab measurements read for a regime they do not show. |
| Pedagogy | 4 | Motivation, prediction before measurement, hand-computed numbers that cells then print. Loses a point for the GQA run that cannot discriminate and for the low-rank baseline. |
| Language | 4 | Plain, exact, almost free of the October mannerisms. Even to the point of voicelessness in the second half, where the authors never say which method matters. |
| Ease and enjoyment | 3 | The first half pulls the reader along. The second half reads as a list, and the JAX plumbing blocks (about 300 words in the first subsection alone) stop the reader who chose that tab. |

## What works

- **The opening** (lines 4 to 14). A chat assistant that would reread the whole question for every word of its answer is a problem the reader feels at once, and "It does not." followed by the one-sentence reason (causality freezes the keys and values) is the right length. Keep as is.
- **The count before the code** (lines 57 to 83). $2Nt$ per pass, $NT^2$ for naive generation, and the worked case of 3,584 prompt tokens plus 256 generated ($2.4 \times 10^{14}$ against $10^{12}$) are correct (recomputed: $2 \cdot 124\text{M} \cdot 256 \cdot 3712 = 2.36 \times 10^{14}$; $2 \cdot 124\text{M} \cdot 3840 = 9.5 \times 10^{11}$) and the end-to-end cell later uses the same case.
- **"Two details need care"** (line 91). Position offset and preallocation are the two things that go wrong in a first implementation. Exercise 2 then breaks the first one on purpose. This is the model for how implementation detail should be chosen.
- **The byte formula and its check** (`eq_kv-cache-bytes`, cell `kv-cache-the-memory-bill`). 72 KiB per token, 288 MiB at 4,096 tokens against 471 MiB of weights: all recomputed and correct, and the cell compares the formula with the allocated buffers, which is a check that could fail.
- **The batching formula** `eq_kv-cache-batch-throughput` and the crossover $b = W/(nc)$, with 53 and 7 worked out (recomputed: 52.4 and 6.5). "Batching spreads the weight reads over many sequences, but every sequence still reads its own cache" is the sentence a reader will remember.
- **The sink experiment** (cells `kv-cache-a-window-needs-a-sink-1` and `-2`). Perplexity 36, then 3,836 with the window alone, then 47 with one retained token. This is a large, surprising, reproducible effect on a real model at laptop scale, and the measurement of attention mass before the eviction prepares it. It is the most memorable experiment in the file.
- **The calibration discipline in the low-rank experiment**: fitting the basis on a second passage, saying why, and making the unfair variant an exercise (exercise 4).
- **One ending.** The body stops at the table and the Summary follows. No recap paragraph.

## Structural findings

**S1. Two sections; split at `## Sharing Keys and Values across Heads`.** The first half answers "how do we generate without recomputing, and what limits generation then?" with untrained models and timing. The second answers "how can the cache be made smaller without hurting the model?" with trained models and loss. They share one formula. Proposed cut:

- *Generation and the KV Cache* (or, more honestly, *The KV Cache*): the introduction, `From Recomputation to Caching`, `Memory and Bandwidth`. About 2,900 words. Exercises 1, 2, 3, 7.
- *Shrinking the Cache*: GQA, low-rank keys and values, windows and sinks, a shortened comparison. About 1,900 words now, less after the cuts below. Exercises 4, 5, 6, 8, 9, 10. It opens by restating `eq_kv-cache-bytes` in one clause and naming its three factors, which the current line 784 already does.

If the author prefers one file, the same seam should become the only top-level division after the cuts below, and the file should come down to about 5,000 words in all.

**S2. The second half ranks nothing.** GQA is in nearly every current open model and is a constructor argument of the chapter's block. MLA is used by one model family and needs training-time support. A sliding window with sinks matters only for unbounded streaming. Quantization and linear attention are not taught here at all. The text gives them one subsection or one table row each and never says this. A reader finishing the section cannot tell which of the five to learn. State the ranking in the first paragraph of the second half, give GQA the mechanism and the numbers, and present the other two as "what else is in the formula", each with its one experiment.

**S3. The title and the roadmap promise generation.** Nothing here concerns how tokens are chosen. Either retitle, or add the two short things a reader looking for "generation" expects and does not find: (i) the cache persists across requests, which is what makes a shared system prompt cheap (prefix caching; `further-reading.md` has the paragraph, the section has no sentence); (ii) the encoder-decoder and the speculative-decoding uses of the same idea, each in a clause. I recommend the retitle plus one sentence on (i).

**S4. Per-framework prose carries about 630 words**, most of it JAX compilation mechanics (`dynamic_update_slice`, `nnx.split`, `donate_argnames`, `fori_loop`, TF32). The writing guide (section 9) allows a sentence or two per tab and sends plumbing to comments or to the Performance chapter. For the JAX reader the first subsection has more words on XLA than on the cache. See KV-5.

**S5. The serving paragraph at the end of `Batched Decoding`** (continuous batching, paged caches) repeats `further-reading.md` lines 112 to 122 almost sentence for sentence. Keep one sentence and the reference.

## Passage-level findings

**KV-1** (minor, explanation). Under `## From Recomputation to Caching`.
Quote: "that grows with $t^2$ (:numref:`sec_attention-at-scale`). At the lengths"
The text says the attention scores stay below $2Nt$ and are "no longer small" at a few thousand tokens, and gives no number. For this model one new token costs about $4ndL$ FLOPs of scores and weighted sums, which at $n = 4032$ is $1.5 \times 10^8$ against $2N = 2.5 \times 10^8$: 60%. The reader who later sees the naive curve bend upward deserves this figure. Fix: add "for GPT-2's shape the scores reach about 60% of the dense cost at 4,000 tokens".

**KV-2** (minor, explanation). Same heading.
Quote: "about $2.4 \times 10^{14}$ operations without the cache. Reading the prompt"
The formula just derived is $NT^2$ for generation from an empty context; the worked case has a prompt, and its two numbers come from sums the text does not show (exercise 1 asks for them). One line removes the jump: "$2N \sum_{t=3585}^{3840} t \approx 2N \cdot 256 \cdot 3712$ without the cache, $2N \cdot 3840$ with it".

**KV-3** (moderate, language and focus). Under `### The Cached Forward Pass`. Pattern: tab prose as a second explanation, five blocks, about 630 words.
Quotes: "A JAX array cannot be changed in place, and a compiled function needs"; "`decode` compiles the whole generation loop. `nnx.split` separates the"; "We set matrix multiplications to full float32 precision for this"; "The naive `generate` of :numref:`subsec_gpt-sampling` keeps its context in".
Each block is accurate and each is about the framework, not about caching. The reader meets fifteen lines on XLA buffer semantics between the cached forward pass and the generation loop. Fix: keep one sentence per block in the tab ("JAX arrays are immutable, so the function returns new buffers and attention reads the whole buffer under a mask"), move the rest into code comments, and move the compile-once and donation discussion to `sec_compilation`, which owns it. The last sentence of the first JAX block (a write past the end is silently clamped) is a real pitfall and should stay, as a comment on the `dynamic_update_slice` line.

**KV-4** (minor, correctness: number no cell prints). PyTorch tab under `### The Cached Forward Pass`.
Quote: "every block saves about a quarter of the time of a decode step, which, as"
No cell measures this. Either cut the figure ("saves time on every decode step") or let an exercise measure it.

**KV-5** (moderate, pedagogy). Heading `### Checking the Cache`.
Quote: "The two paths agree to rounding error. Next we time one generation step at"
The subsection named "Checking the Cache" holds the correctness check (one cell) and then the whole timing study: the model of GPT-2's size, two timing helpers, five context lengths, a plot, and the end-to-end run. The heading hides the section's first main result. Insert `### Measuring the Speedup` at the quoted sentence. Exercise 1 and line 1166 cite the end-to-end run through `subsec_kv-cache-same-logits`; give the new subsection its own label.

**KV-6** (major, correctness: experiment read for more than it shows). End of the timing study.
Quote: "the factor of about 250 in operations. Every decoded token still reads all the weights of"
The prose explains the gap between 250 times fewer operations and a measured speedup of "more than five" by weight reads. In the stored PyTorch output the cached step takes 7.2 ms at every context from 256 to 2,048 tokens and 7.6 ms at 4,032. Reading 494 MB of weights at 1 TB/s takes about 0.5 ms. The step is therefore limited by something fourteen times larger than the weight read, namely launching several hundred kernels from Python, which the PyTorch tab note 150 lines later says outright. At 256 and 512 tokens the cached step is slower than the naive one (7.2 against 6.5 and 6.7 ms). In JAX the same loop gives 0.8 to 1.2 ms per token and a speedup of about 40 end to end (15.2 s against 0.4 s), close to what the bandwidth bound allows. So the shared sentence is true for one tab and false for the default one, and "more than five times faster" describes a 6 and a 40 with one phrase. Fix direction: say what limits each. Draft: "The cached loop is faster, but by far less than the factor of 250 in operations. A decode step cannot be faster than the time to read every weight once, about half a millisecond for this model, and an eager framework adds the cost of launching each small kernel from Python, which here is several milliseconds more. `subsec_kv-cache-prefill-decode` separates the two." Then let each tab state its own number in one sentence. See also KV-8 for the redesign.

**KV-7** (moderate, explanation). Under `### Prefill and Decode`.
Quote: "about 1 TB/s, so a computation needs about 80 FLOPs per byte to keep the"
This paragraph introduces arithmetic intensity, the roofline model, two bounds, and the ratio of 80 in seven lines. It is correct (82.6 TFLOP/s and about 1,008 GB/s for the RTX 4090) and it resolves the October finding that "roofline" was used undefined. It is still the densest paragraph in the section and has no picture and no name for the 80. Exercise 7 then calls it "the ridge" (quote: "$b \to \infty$, and compare both values with the ridge of about 80"), a term the body never defines. Fix: split into two paragraphs (the two bounds; the number for this GPU), name the crossover once ("the ridge point of this GPU, about 80 FLOPs per byte"), and consider a two-segment roofline sketch with the decode step at 0.4 and the prefill at 800 marked on it. That figure would carry the section's main idea.

**KV-8** (major, correctness: default tab does not show the claim). Under `### Prefill and Decode` and `### Batched Decoding`.
Quotes: "generates tokens at least an order of magnitude more slowly, with the same"; "At 1,024 tokens it grows little beyond a batch of 16 and"; "Here the fixed cost dominates almost the whole range. Each decode step".
Stored PyTorch output: prefill 96,792 tokens/s, decode 138 tokens/s, bandwidth ceiling 1,550 tokens/s. Batched decode: 140, 545, 2,139, 8,603 tokens/s at context 128 and 140, 535, 2,115, 6,830 at 1,024. The measured curves are straight lines of slope one, a factor of 7 to 14 under the bound at small batches (rendered plot checked). The saturation that `eq_kv-cache-batch-throughput` predicts, which is the point of the subsection, appears in one PyTorch data point (6,830 against 8,603). The sentence "it grows little beyond a batch of 16" is about the dotted bound, and a reader looking at the solid curve sees the opposite. The JAX output does show it (1,085, 3,396, 7,340, 10,059 against a ceiling near 12,000). This is the case the writing guide names in section 9: the two tabs draw different conclusions, and the experiment needs a paragraph explaining what it fails to show. Also "at least an order of magnitude" undersells both tabs: the ratio is about 700 in PyTorch and 120 in JAX; write "hundreds of times". Fix direction, for the development machine: make the per-step bytes large against the per-step launch cost in eager PyTorch. A wide, shallow random model does this (for example 12 blocks of width 3,072: about 1.4B parameters, 5.4 GB in fp32, a 5 ms weight read per step against the same few milliseconds of launches), or the decode step can be compiled once here with a forward reference to `sec_compilation`. Until then, move the admission from the tab note into the shared prose before the plot, so the reader knows what the solid lines will show.

**KV-9** (minor, language: repetition). Pattern, four statements of the fixed cost and five of the weight read.
Quotes: "its time on fixed costs, such as launching the kernels of twelve blocks,"; "bandwidth ceiling, because a fixed cost per step, such as launching the"; "cost per step of the previous subsection."; plus the PyTorch tab paragraph. The statement that every decoded token reads all the weights is made at lines 18, 495, 553, 608 and in the Summary.
The weight read is the section's thesis and may recur; the launch cost should be explained once, where it is first seen (the timing plot), and cited after that.

**KV-10** (minor, focus: duplicate). End of `### Batched Decoding`.
Quote: "every decode step, which is called *continuous batching*"
See S5. Keep: "Serving systems batch at the granularity of one decode step and page the cache (:numref:`sec_transformers-further-reading`, :numref:`sec_model_serving`)." The sentence "the Language Models part of the book builds their mechanisms" points at skeleton chapters; it is not load-bearing and can stay only if those chapters will exist at publication.

**KV-11** (moderate, correctness; confidence medium to high). Under `### Cache Size and Validation Loss`.
Quote: "the original steps recovers most of the quality, and this is how GQA"
The uptraining recipe (mean-pool, then about 5% more steps) is in Ainslie et al. The claim that this is how GQA "spread through existing model families" is, as far as I know, false: Llama 2 70B, Mistral 7B and Llama 3 were trained with GQA from the start (the Llama 2 paper reports a from-scratch ablation of MHA, MQA and GQA). Unverified from the papers in this session. Fix: "The paper proposed it as a way to convert existing checkpoints; the models that adopted GQA mostly train with it from the start."

**KV-12** (moderate, pedagogy: experiment at the wrong scale). Same heading.
Quote: "losses lie within a few hundredths of a nat, so at this scale even a single"
Stored: 1.34, 1.34, 1.35, 1.34 in both frameworks. The text reads this honestly as "no loss we can measure". But a 4.8M-parameter byte model after 1,000 steps is nowhere near the point where the number of distinct key spaces could bind, so the four training runs could not have come out otherwise, and the claim that matters ("at large scale MQA loses a little") rests on the citation. The writing guide's rule applies: an experiment that cannot show the effect is fixed or removed. Options: (a) cut the runs, keep the parameter and cache count cell, and cite; (b) replace them with a task where heads must retrieve by different keys at once (several independent key-value lookups per position) so that $H_\textrm{kv} = 1$ fails visibly and $H_\textrm{kv} = 2$ recovers; (c) move the uptraining exercise (exercise 8), which does teach something, into the body in place of the null result. I would do (a) now and (b) when experiments are redesigned.

**KV-13** (major, correctness; confidence high). Under `### Low-Rank Keys and Values`.
Quotes: "$2 \times 768 = 1536$ numbers in GPT-2, lie close to a subspace of much"; "With a basis of rank 512, a third of the width, the loss matches that of".
The keys and values of a token are a linear map (GPT-2 adds a bias) of the 768-dimensional normalized block input: `attn.qkv(blk.norm1(H))`. The 1,536-dimensional row therefore lies in a subspace of dimension at most 769 for any weights, trained or not. "Rank 512 of 1536, a third of the width" compares against a width that the data cannot fill; the honest fractions are two thirds (512 of 768), one third (256) and one sixth (128). The same holds for the DeepSeek-V2 comparison (quote: "against 32,768 numbers of keys and values per token in its 128 heads"): keys and values there are functions of a 5,120-dimensional hidden state, so the latent of 512 is a tenfold compression of the information, not sixty-fourfold. The result survives the correction and becomes clearer: caching the 768 numbers of the block input would reproduce keys and values exactly at half the bytes, at the price of recomputing the projections at every read; the experiment asks how far below 768 one can go, and the answer is that a third of it costs a few hundredths of a nat on this passage. Fix: state the 768 bound in one sentence before the experiment, change the printed denominator to 768 (or print both), and correct "a third" and "a sixth". This also sets up exercise 9, whose point is exactly the trade between stored elements and per-step FLOPs.

**KV-14** (minor, correctness: explanation without evidence). After the low-rank cell.
Quote: "full attention, even a few hundredths of a nat below it, since dropping the"
The difference is 3.55 against 3.58 on 511 predictions of one passage. "Since dropping the weakest directions also drops some noise" is a cause asserted for a difference the experiment cannot distinguish from zero. Write "the loss is indistinguishable from that of full attention".

**KV-15** (minor, correctness: explanation without evidence). After the window cell.
Quote: "The rest of the gap is what the window loses by dropping older tokens. Our"
The remaining 0.18 nat could equally come from the model meeting an attention pattern it was not trained on. The experiment does not separate the two. Write "The rest of the gap is the cost of the window: older tokens are gone, and the model was not trained on this pattern."

**KV-16** (moderate, focus: paper retelling). End of `### Sliding Windows and Attention Sinks`.
Quote: "Models can also be trained not to need a sink. gpt-oss gives every head of"
A paragraph that narrates two papers the section does nothing with, ending on a result quoted to two digits ("from 47% to 5%"). Unverified in this session; I believe both descriptions are right (gpt-oss has a learned per-head sink logit in the softmax denominator; Qiu et al. gate the attention output with a sigmoid of the layer input and report roughly 47% falling to 5% for models of 1.7B and 15B parameters). Exercise 6 already teaches the sink logit properly. Fix: one sentence ("Recent models remove the need for a sink by giving each head a learned logit that can absorb unwanted weight, which an exercise analyses") and move the rest to Further Reading.

**KV-17** (moderate, focus and correctness). `### Comparison of Cache-Reduction Methods`.
Quote: "| sliding window with sinks | cached length |"
Three problems. (i) The table surveys five methods at equal weight, two of which (quantization, linear attention) the section does not teach; the "evidence in this section" column is a note from the authors to themselves. (ii) The training-requirement cell for the window row says "train or adapt to the restricted pattern", but the experiment just above applied the window to a trained GPT-2 with no training and lost 0.18 nat, and StreamingLLM's claim is that no fine-tuning is needed (a dedicated sink token at pretraining helps further). (iii) The GQA row says "character model"; everywhere else it is the byte-level model. Fix: reduce to the three taught rows and the three columns "factor of `eq_kv-cache-bytes` reduced / stored elements / needs training?", correct the window cell to "none; better if pretrained with a sink token", and give quantization and linear attention one sentence each below the table, which the following paragraph already does.

**KV-18** (minor, pedagogy). Pattern, three occurrences.
Quotes: "cache, not the weights, fills the accelerator; an exercise works this out"; "exercise works this out."; "that of the full cache :cite:`Liu.Yuan.Jin.ea.2024`. An exercise tests why".
Unnamed pointers. The enc-dec section uses `:exlabel:`/`:exref:`; do the same here so the reader can find exercises 3, 10 and 5.

**KV-19** (minor, enjoyment: voice). Second half as a whole.
The first half has sentences with an author behind them ("It does not."; "Two details need care."). From `## Sharing Keys and Values across Heads` on, every paragraph is one register: method, citation, cell, reading. The writing guide asks the authors to judge. Places where a judgment is available and true: GQA is the cheapest factor of four to eight in the whole stack and costs nothing measurable, which is why everyone uses it; the sink result is startling (one token out of 1,024 is worth a factor of 80 in perplexity) and the text reports it in the same tone as a parameter count. Say that it is surprising, and why.

## Code and experiments

- `kv-cache-same-logits-measured-1`: a real implementation test (it can fail; exercise 2 makes it fail). Keep. Stored deviations 1.2e-06 and 7.2e-07 match "rounding error".
- `kv-cache-checking-the-cache-4`, `kv-cache-same-logits-measured-3`, `kv-cache-prefill-and-decode-1`, `kv-cache-batched-decoding`: see KV-6 and KV-8. PyTorch stored values contradict the shared prose's explanation, not its hedged numbers. Candidate for the list of experiments to redesign on the development machine.
- `kv-cache-prefill-and-decode-1`: the predicted prefill rate at peak arithmetic is $83 \times 10^{12} / 2N \approx 335{,}000$ tokens/s; measured 97,000 (PyTorch) and 122,000 (JAX), about a third of peak. The text does not compare them. One clause would close the loop on the compute-bound side as the ceiling does on the memory-bound side.
- `kv-cache-a-pluggable-implementation-1`: prints three shapes. Mild; acceptable once, since the shapes are the point of GQA.
- `kv-cache-a-pluggable-implementation-2`: parameter and byte counts (262,144 down to 147,456; 2,048 down to 256 bytes) match $4d^2 \to 2.25d^2$ and the formula. Fine.
- `kv-cache-cache-against-quality`: null result that could not have been otherwise at this scale (KV-12).
- `kv-cache-compressing-the-cache-further-2`: "helper vs model: max deviation 0.0e+00" is a helper self-test. Acceptable, one line.
- `kv-cache-low-rank-keys-and-values`: correct experiment, wrong denominator (KV-13). Note also that `calib` has 1,024 rows, so the SVD can return at most 1,024 directions; with the 768 bound this no longer matters, but the "of 1536" in the print statement should go.
- `decode` (PyTorch) runs one more `forward_cached` after sampling the last token; the logits are discarded. Harmless for timing (the helper divides by 32 and makes 32 calls) but a reader copying the loop pays one wasted step. A comment or a reordered loop fixes it.
- `sample` reimplements the temperature and top-$k$ logic of `generate` from `subsec_gpt-sampling`. If `generate` can call a saved `d2l.sample`, both sections get shorter.
- Stored plot `kv-cache-batched-decoding-1.svg` (PyTorch): the legend covers the left third of both bound curves.

## Exercises

Ten exercises, about 1,830 words, 42% of the body against the guide's 30%. Their quality is high: almost every one asks for a prediction or an explanation, several break the implementation on purpose (2, 4), and 9 (absorbed projections in MLA) and 10 (a rolling buffer with position reassignment) are small research-grade problems. Assessment:

- Keep 1, 2, 3, 7 with the first half. Exercise 3 (Llama-2-70B) recomputed: 320 KiB per token with 8 key-value heads, cache equals weights near 427,000 tokens; 14, 1 and 119 sequences of 4,096 tokens fit in the 20 GB left on two 80 GB GPUs. Consistent with the body.
- Exercise 7 part 1 recomputed: $m \le 2 \cdot 83 = 166$ tokens. Good. Define "ridge" (KV-7).
- Exercise 8 (uptraining) is the experiment the GQA subsection lacks; consider promoting it (KV-12).
- Exercise 5 (quantization) is the only treatment of a table row; fine as an exercise.
- With a split, six exercises for the second section is still many. Exercise 6 part 3 and exercise 10 part 3 could go without loss.
- No exercise asks the reader to use `eq_kv-cache-batch-throughput` on a realistic deployment (a 70B model at 16 bits, 32K contexts: at what batch does throughput saturate?). Add it as a part of exercise 3; it is the formula's main use.

## Still open from October

"Roofline is used before ch. 13 defines it" is resolved in substance: the term is now defined inline, densely (KV-7). The two `kv-cache.md` examples in the guide's table of patterns ("Formulas should be checked, not trusted:", "Why obsess over the size of a buffer") are gone.

## For the chapter review

- Position is right: after `sec_gpt` (whose `generate` it replaces) and before the wirings. It assumes `d2l.GPT` with `num_kv_heads`, `qkv`, `rope_tables`, `load_gpt2`, `LMOptimizer` and `train_steps`, and `causal_mask` from the attention chapter. It reads without the appendix; the one appendix reference (Eckart-Young, `sec_mdl-svd-low-rank`) points to the section that holds the theorem and is not needed to follow the experiment.
- It sets up the cost vocabulary (bytes moved per token, prefill against decode) that `sec_moe` and `sec_scaling-laws` should reuse rather than re-derive; the chapter reviewer should check that they do.
- The next section does not use it: `encoders-decoders.md` decodes by rerunning the decoder over the whole prefix and never mentions that cross-attention keys and values are computed once per source. One sentence there would connect the two.
- Suspected chapter-wide patterns: (1) tab prose used as a second explanation, far heavier in JAX; (2) timing experiments in eager PyTorch that measure dispatch overhead while the prose argues about arithmetic or memory (the Perceiver timing in the next section has the same problem); (3) closing survey tables with an "evidence" column; (4) exercise volume at 40 to 50% of the body.
- Forward references to the Language Models part (serving mechanisms) and to `sec_compilation` are present; only the latter carries weight, in the PyTorch tab note.
