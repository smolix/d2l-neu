# Encoders, Decoders, and Cross-Attention

Source: `chapter_transformers/encoders-decoders.md` (read to the `<!-- slides -->` marker at line 1202). About 3,100 words of prose, 1,500 words of exercises. Stored outputs read: `outputs/pytorch/chapter_transformers/encoders-decoders.json`, the JAX file, and the three stored plots. Figures `mdl-transformers-three-wirings.svg` and `mdl-transformers-latent-bottleneck.svg` rendered and inspected.

## Verdict

The section builds all three wirings from one block in compact, readable code, and two of its ideas are good enough to keep exactly: measuring what bidirectional context contributes by binning one model's loss by position, and choosing string reversal because its alignment is known in advance. Against the bar it is a competent section with three problems. First, it does not answer the question a reader in 2026 brings to it: which of the three should I use? The guidance amounts to one closing paragraph and one Summary sentence ("encoder--decoders for translation and speech"), and the section never says that decoder-only models now do most sequence-to-sequence work by putting the source in the prompt; that wiring appears only in exercise 1. Second, both headline experiments are read for more than they show. The position-binned loss compares interior positions with two edge positions that the masked model almost never trains on, and the alignment paragraph describes run-to-run failures while the stored outputs in both frameworks show a perfect anti-diagonal. Third, the learned-queries part is a description of the Perceiver with a timing of an untrained module; it teaches a complexity count and nothing about whether a bottleneck works. The most important fix is the first: add a short, opinionated subsection on choosing a wiring today, and let it replace the pre-Summary recap.

## Scores

| dimension | score | note |
|:--|:-:|:--|
| Clarity of explanation | 4 | Each wiring is explained in a paragraph and a class. Masks are explained in the wrong place. Pooling (how per-token outputs become one vector) is missing. |
| Clarity of thought and focus | 3 | Four topics (MLM, encoder-decoder, padding masks, Perceiver). The section's own question, "what does a transformer gain when it may see its whole input", is answered by a confounded measurement. |
| Pedagogy | 3 | Two well-chosen tasks; both experiments over-read; no decision guidance; a mask detour in the middle of the reversal arc. |
| Language | 4 | Plain and exact, with a few sentences that have voice. One muddled closing sentence, one wrong figure reference. |
| Ease and enjoyment | 3 | The lobster fill-in and the reversal task are pleasant. The alignment paragraph and the Perceiver part are not. |

## What works

- **The opening** (lines 4 to 11): the gap in "the ___ barked at the mail carrier" and the translator who reads the whole sentence first. Two concrete cases, two questions, eight lines.
- **"Both questions concern two choices that the GPT fixed: the mask, and the sources of the queries, keys, and values."** (line 13). This sentence organizes the section and should also organize its ending.
- **Sentences with voice**: "generation is the training objective run forward" (line 74); "The objective must hide what it asks for." (line 162). Protect both.
- **Position-binned loss as a design** (`subsec_mlm-loss-by-position`): one model, one objective, a comparison inside every batch. The idea is excellent even though the reading needs repair (ED-5).
- **The fill-in example**: "known as _he European" with 0.96 on "t", and the observation that the left context allows many words while the right leaves one. This is the Illustrated-Transformer kind of small case.
- **Reversal as a task with known alignment** (lines 385 to 397), also marked "keep" in October. The reasoning that translation gives no ground truth for attention is exactly right.
- **`DecoderBlock`** is twenty lines, and "A decoder block is the transformer block plus one sublayer" tells the reader what to look for in them.
- **Exercises 5, 6 and 8** (parallel against iterative fill, the entropy identity behind the equal edge losses, a Perceiver IO for reversal) are among the best in the chapter.

## Structural findings

**S1. The reader does not learn when each wiring is the right choice today.** What the section says: an encoder gives "one representation per token" for "classification, retrieval, and tagging"; an encoder-decoder "fits tasks whose input is fully known before generation starts"; decoder-only models "perform tasks that are specified only in the prompt". The closing paragraph and the Summary repeat this as a three-way list. What is missing, and what a practitioner needs:

- A decoder-only model handles input-to-output tasks by concatenation: source, separator, target, one stack, one causal mask (or the prefix mask of exercise 1). This is how translation, summarization and question answering are mostly done now, and it is why the encoder-decoder is no longer the default for text. The section's own figure shows it: the third panel of `fig_three-wirings`, drawn as one 6 by 6 grid, is the attention pattern of a prefix language model. The text never points this out.
- Why decoder-only won for text: every position gives a training signal (the masked objective scores 15% of them), one stack serves pretraining and every task, the cache of `sec_kv-cache` makes the prompt a one-time cost. The first of these the section could state with its own numbers.
- What an encoder-decoder still gives: a source read bidirectionally, once, by a stack that can differ in size and modality from the decoder (audio frames in Whisper), and cross-attention keys and values that are computed once per source. That is why it persists in speech and in some translation systems, and why T5-style encoders are used as text encoders for image generators.
- What encoders are for now: embeddings for retrieval and reranking, and cheap classification at high throughput, where a model of 100M to 400M parameters replaces a call to a large decoder. Here the text should also be candid that the largest embedding models are now built from decoder-only language models, often with the mask relaxed in fine-tuning (I state this from general knowledge; unverified in this session).

Proposed fix: a closing subsection, `### Choosing a Wiring`, of about 250 words, built on the two choices of line 13 (mask, source of keys and values), with a three-row table: what the model outputs, what it costs at inference, where it is the default today. It replaces the paragraph at line 1006 and lets the Summary's last sentence be cut. The dated half-sentence "encoder--decoders for translation and speech" must not survive unqualified.

**S2. The section holds four topics; the fourth can shrink.** Masked-token prediction and the encoder-decoder are the section. Padding masks are a necessary technical detail in the wrong place (ED-7). The Perceiver part (about 550 words, a figure, three cells) is a paper description; October recommended cutting "Perceiver IO and descendants", and the Perceiver IO paragraph is still there. Learned queries deserve to stay because they are the mechanism behind resamplers and query-based decoders in vision and vision-language models, which the introduction promises ("as in Perceiver-style and vision--language models") and the body never mentions again. Either deliver that connection in two sentences and cut the timing study to one cell, or cut the subsection to a paragraph and move the class to an exercise (exercise 8 already builds on it).

**S3. No link to the previous section.** After 6,800 words on not recomputing the past, the decoding loop here reruns the decoder over the whole prefix at every step, and the text says only that "the encoder runs once". Two facts would tie the sections together: the decoder's self-attention can be cached exactly as in `sec_kv-cache`, and the cross-attention keys and values depend only on the encoder output, so they are computed once per source and never grow.

**S4. Two endings.** The paragraph at line 1006 ("The three wirings differ in what they produce...") is a recap, and the Summary repeats it. See S1 for what should stand there instead.

## Passage-level findings

**ED-1** (major, thought and correctness: currency). `## Summary`, and the paragraph before it.
Quote: "encoder--decoders for translation and speech, and decoder-only models for"
See S1. As written the reader concludes that translation calls for an encoder-decoder. In current practice decoder-only models do most text-to-text work, and the encoder-decoder remains where the input is another modality or where a bidirectional read of a long source, done once, is worth a second stack. Draft for the closing subsection: "A decoder-only model can do everything the other two do: write the source, a separator, and let it continue. That is the default today, because one stack is pretrained once and every position of the text teaches it. Choose an encoder when you need vectors, not text: embeddings for search, or a label for each of a million documents, at a hundredth of the cost. Choose an encoder-decoder when the source is not text in the decoder's vocabulary (audio, an image) or when it is long, read once, and worth reading in both directions."

**ED-2** (moderate, correctness: currency and precision). Under `## Encoder, Decoder, and Encoder--Decoder Architectures`.
Quote: "strongest way to *represent* an input, and one representation per token is"
Two problems. "Strongest" is a claim the section does not support and that practice has complicated (see S1, last bullet); "a natural way" or "the direct way" is true. And classification and retrieval do not consume one representation per token; they consume one vector per input. The section never says how the per-token outputs become that vector (mean pooling or a designated token), so the reader cannot build the classifier the paragraph promises. The next section's class token will look unmotivated without it. Fix: one sentence here ("tagging uses the output at each token; classification and retrieval pool the outputs into one vector, by averaging or by reading a designated position, as :numref:`sec_vision-transformer` does").

**ED-3** (minor, language: repetition). Under `### The Masked-Token Objective`.
Quote: "This model cannot be trained on next-token prediction. With the future"
The same argument, in nearly the same words, was made 95 lines earlier ("with the future visible, next-token prediction reduces to copying"). Keep it here, where it motivates the objective, and shorten the earlier one to a clause.

**ED-4** (minor, explanation). Same heading.
Quote: "\max \; \sum_{t \in \mathcal{M}} \log p\left(x_t \mid \mathbf{x}_{\setminus \mathcal{M}}\right),"
The maximum is over nothing: no parameters appear. Write $\max_\theta$ and $p_\theta$, or drop "max" and call it the log-likelihood being maximized. The following sentence should also say that the masked positions are still present in the input as `<mask>` with their position embeddings, which is why the model knows where the gaps are. The rate of 15% arrives with no reason; exercise 2 questions it, so one clause ("BERT's choice, which an exercise varies") is enough. The sentence on the label $-100$ is plumbing and belongs in the code comment that already says it.

**ED-5** (major, pedagogy and correctness: experiment read for more than it shows). `### Loss by Available Context`.
Quotes: "position 0 or 63 sees only one side. The last position has the context"; "conditioning on both sides of a byte lowers its loss by about a third".
Stored: 2.37 at position 0, 2.36 at position 63, interior mean 1.44 (JAX: 2.37, 2.42, 1.49). The arithmetic in the prose is right (a drop of about 39%). The reading is not safe, for two reasons.
(i) Position 63 is not "the context that a causal language model has at every step". Fifteen percent of its left context is masked, and, more important, the model has been trained on that situation in one position out of 64. Under random masking an interior byte has a visible right neighbour 85% of the time; the last position never has one. The model has had about 19,000 training examples (2,000 steps of 64 windows at 15%) of one-sided prediction at each edge against over a million two-sided ones, so its edge loss measures an undertrained special case as much as the worth of the second side.
(ii) The chapter's own causal byte model reaches a validation loss of about 1.34 nats (`kv-cache-cache-against-quality`, 1,000 steps, a larger model and a longer window) with left context only. That is below this model's two-sided 1.44. The models are not matched, so this proves nothing by itself, but a reader who remembers the number will see that "one side costs 2.4 nats" cannot be a property of text.
The stored plot also shows something the prose does not mention and that is the experiment's real finding: the loss falls from 2.37 at position 0 to about 1.7 at position 1 and reaches the interior level by position 3. Two or three bytes on the missing side close most of the gap; a byte is predicted mainly from its immediate neighbours.
Fix direction: add the matched control. Train the same `TransformerEncoder` stack with a causal mask on next-byte prediction for the same 2,000 steps and print its validation loss beside the interior masked loss. Then the section can say what the second side contributes with a fair denominator (my expectation is a modest gain, far less than a third; unverified, there is no PyTorch on this machine). Report the edge values as what they are ("this model, at the two positions where it has only one side"), and add the sentence about how fast the profile falls. Rewrite the Summary sentence (quote: "third less loss than positions with only left context. An encoder--decoder") to match.

**ED-6** (minor, thought). Same heading.
Quote: "relative to one side. The two edges cost about the same, because in text"
"Because in text what follows a gap constrains it about as strongly as what precedes it" restates the observation as its cause. Exercise 6 has the real reason: for a stationary source the two conditional entropies are equal, whatever the text. Point there: "This is no accident of English; for any stationary source the two are equal (exercise :exref:...)".

**ED-7** (minor, correctness: wrong reference). End of `### Loss by Available Context`.
Quote: "pretraining and fine-tuning in full (:numref:`chap_nlp_pretrain`). Encoders"
`chap_nlp_pretrain` is the retired NLP Pretraining chapter, now in the Attic, not in the Language Models part. Either cite the Language Models chapter that will own BERT-style pretraining or say "the Attic chapter :numref:`chap_nlp_pretrain`". The ModernBERT description in the next sentence matches my knowledge (RoPE, GeGLU, alternating local and global attention, 8,192 tokens).

**ED-8** (minor, explanation). Under `### A Bidirectional Encoder`.
Quote: "positions, and the learned position table stays."
The chapter has argued for RoPE as the default, and the paragraph three screens later says ModernBERT uses it. The reader may wonder why this encoder goes back to a learned table. One clause: "we keep a learned table because `d2l.TransformerBlock` takes its positions from the embedding; RoPE would work as well".

**ED-9** (moderate, pedagogy: placement; and a check that cannot fail). `### Assembling the Masks`.
Quote: "Each row spreads its weight over the three real source positions, and the"
The subsection sits between `The Decoder Block` and `Training and Decoding` and ends by telling the reader that none of it applies to the model about to be trained ("None of this arises in the reversal model"). It interrupts the arc task, block, training, alignment at its midpoint. The content is the right kind of practical detail (three attention sites, three masks, shapes given) and should stay, after `Reading the Alignment`, as the step from the toy to a real batch. The second cell (`encoders-decoders-assembling-the-masks-1`) passes the masks to a random block and observes zeros in the masked columns, which is true by the definition of masked softmax; cut it and keep the three printed grids.

**ED-10** (moderate, correctness; confidence high). Same subsection.
Quote: "teacher-forced training needs the causal triangle *and* target padding on"
With right padding, which the paragraph has just specified, every padded target position lies after every real one, so the causal triangle already hides it from every real query. The key-side padding mask changes only the rows of padded queries, whose outputs the loss ignores. So "needs" is false for the case described, and exercise 3 part 1 asks the reader to discover exactly this. An exercise may correct a simplification; it should not contradict a statement of necessity. Fix: "Decoder self-attention takes the causal triangle. With right padding that already hides the padded keys from every real query; frameworks add the target padding mask anyway, and it becomes necessary under left padding (exercise :exref:...)".

**ED-11** (moderate, focus: material written for the authors). `### Training and Decoding`.
Quote: "Adam at a constant learning rate makes it jump now and then, to tenths of"
Four sentences about an instability no cell shows, in a run the reader will not perform, to justify a cosine schedule. This is a debugging note. Fix: "We decay the learning rate to zero with the cosine schedule of `d2l.LMOptimizer`; at a constant rate the loss, once near zero, occasionally jumps." One sentence, or an exercise if the phenomenon is worth teaching (it connects to the Adam material of `sec_adam`).

**ED-12** (minor, explanation). Same heading.
Quote: "Decoding runs the encoder once and the decoder autoregressively: feed"
See S3. Add: "Each step here reruns the decoder over the whole prefix, which at twelve tokens costs nothing. A real decoder caches its self-attention keys and values as in :numref:`sec_kv-cache`, and computes the cross-attention keys and values once, since they depend only on the encoder output."

**ED-13** (major, pedagogy and correctness: prose and stored output disagree; wrong figure reference). `### Reading the Alignment`.
Quote: "Whether the maps show the anti-diagonal predicted by the third panel of"
(i) The third panel of `fig_three-wirings` shows which keys a target query may attend to: a full light rectangle. It predicts no anti-diagonal. The prediction was made in `subsec_reversal-task`. Wrong reference.
(ii) Stored outputs, both frameworks: argmax on the true source position in 100% of rows, mean weight 0.99 (PyTorch) and 0.98 (JAX), same column for every string in 100% of rows; the stored heat map is a clean anti-diagonal. The paragraph that follows spends twelve lines on runs in which the argmax "lands there in only about half of the rows or fewer, even in none". A reader of the book sees a perfect result and an explanation of a failure that is not on the page. I could not rerun the cell to see how often the other outcome occurs (unverified).
(iii) The task was chosen because its "correct alignment is known by construction", and the paragraph then explains that the alignment is not determined after all, because the encoder can permute the letters first. Both statements cannot carry the section. The writing guide's rule applies: an experiment that needs a paragraph of confounders is fixed, not explained.
Fix direction, two options. (a) Make the claim true: remove the encoder's freedom to permute so that cross-attention must carry the whole alignment (for the reversal demonstration, an encoder of embeddings plus positions with no self-attention block, or cross-attention into the embeddings), show the anti-diagonal, and state in one sentence that with a self-attention block in the encoder the alignment can be split between two layers, which exercise 4 part 3 then investigates by composing the two maps. (b) Make the instability the lesson and show it deterministically: print the hit rate of the cross-attention alone and of the product of cross-attention and encoder self-attention over several seeds in one cell, so that the page shows both outcomes and the composed map resolving them. Option (b) is the better section, because "a single map can mislead even in a one-block model" is worth more than "attention found the alignment", but it must be on the page and not in a conditional paragraph. In either case give the range over runs once, as the house style does for random experiments.

**ED-14** (moderate, focus: paper description; cells that only confirm). `## Cross-Attention as Interface`.
Quotes: "size to data of variable size, as in Perceiver-style and vision--language"; "cross-attention grows linearly, but the fixed $O(M^2)$ latent processing".
(i) The introduction promises vision-language models; the body never returns to them. Two sentences would do it: a resampler turns a variable number of image patch features into a fixed number of tokens for a language model, and detection and segmentation decoders use one learned query per object. I state these from general knowledge of Flamingo, BLIP-2 and DETR; the editor should add the citations. It would be honest to add that many current vision-language models skip the resampler and feed projected patch tokens straight into the decoder.
(ii) Cell `encoders-decoders-learned-queries-1` prints `(1, 4096, 256) -> (1, 64, 256)`: a check that cannot fail. Cell `encoders-decoders-the-cost-curve` times an untrained module on random inputs and confirms the complexity count just stated. Nothing in the subsection shows that 64 latents can hold what a task needs; exercise 8 does (the smallest $M$ that reverses a 12-letter string), and it is the experiment this subsection should have had.
(iii) The explanation of the flat Perceiver curve is doubtful in PyTorch. Stored: 1.42, 1.42, 1.49, 1.98 ms. Two blocks over 64 latents of width 256 are microseconds of arithmetic; 1.4 ms is the cost of launching the kernels, the same fixed cost `sec_kv-cache` has just taught. Write "a fixed cost that does not depend on $N$" and do not attribute it to $O(M^2)$. (Confidence high on the mechanism, unmeasured.) The other readings match the stored output: ratios of 3.1, 3.2 and 3.7 per doubling for full self-attention, a gap of 18 times at $N = 8192$ (35 in JAX), comparable times at $N = 1024$.
Fix direction: one paragraph on learned queries with the two uses above, the class, and either the exercise-8 experiment in miniature (exact match against $M$) or no experiment. Cut the shape cell. Keep the cost comparison only as a sentence, or as one row of the table proposed in S1.

**ED-15** (minor, correctness: wrong home for a reference; repetition). `### The Cost Curve`.
Quote: "is `d2l.Benchmark` of :numref:`sec_perf_model`: it runs the function a few"
`Benchmark` is defined and saved in the Basics part (`chapter_builders-guide/gpus-devices-memory.md`, "packages these rules for later chapters"); `sec_perf_model` is two chapters ahead. And the previous section has just described the same helper in the same words. Cite the defining section in a clause and drop the description.

**ED-16** (minor, language and thought). Paragraph before `## Summary`.
Quote: "Learned queries are a fourth use of cross-attention: they decouple the"
There have been three wirings and one prior use of cross-attention, so "a fourth use of cross-attention" does not parse. It disappears with S1; if kept, write "Learned queries are a second use of cross-attention".

**ED-17** (minor, explanation). Figure `fig_three-wirings`, third panel.
Quote: "for the target's own past. The wiring fits tasks whose input is fully known"
The panel draws the encoder-decoder as one 6 by 6 grid over source and target positions. That is a clear way to show who sees whom, but it hides the difference that defines the architecture: two stacks, with every decoder block reading the output of the last encoder block. As drawn it is the mask of a single-stack prefix model. Either say so ("drawn as a single grid, this is also the mask of a decoder-only model whose prompt is read bidirectionally; the encoder-decoder differs in using two stacks, and in which layer of the source the target reads"), which gives S1 its picture for free, or add a small block diagram of the two stacks. The Illustrated Transformer's most copied picture is that block diagram; this section has no equivalent.

## Code and experiments

- `encoders-decoders-the-masked-token-objective`: 2.60, 1.82, 1.47 (JAX 2.80, 1.95, 1.54). "Far below 3.17 nats" holds; 3.17 is printed in `gpt.md` (the unigram cell), not here, so add "(:numref:`sec_gpt`)". "About half of the masked bytes" is printed by the next cell (0.56 and 0.54), not this one; move the claim after that cell.
- `encoders-decoders-what-the-second-side-is-worth-1`: see ED-5. Needs a causal control.
- `encoders-decoders-what-the-second-side-is-worth-2`: good; the prose reads only the third line. The first line ("gam_arus" filled with "p" at 0.67, where the truth is "m") is a wrong prediction the text passes over; it is a nice honest case (the model writes a plausible word, not the Latin one) and worth a clause.
- `encoders-decoders-assembling-the-masks-1`: cannot fail (ED-9). Cut.
- `encoders-decoders-training-and-decoding-1`: loss 0.007 by step 100 and 0.000 from step 400. The task is solved in the first sixth of the run; 200 steps would do, or the print could show steps 25, 50, 100 so the reader sees learning happen.
- `encoders-decoders-reading-the-alignment`: stored output contradicts the hedged prose (ED-13).
- `encoders-decoders-learned-queries-1`: shape print, cannot fail (ED-14).
- `encoders-decoders-the-cost-curve`: confirms the count; mechanism of the flat curve misattributed in PyTorch (ED-14).
- No TensorFlow or MXNet material. The single JAX tab note (43 words) is within the guide's limit.
- `MaskedBytes`, `MaskedLM`, `mask_tokens`: compact and readable. `TransformerEncoder` is reused unchanged by `EncoderDecoder`, as the text says.

## Exercises

Eight exercises, about 1,500 words, 49% of the body against the guide's 30%. Quality is high and they are well graded.

- Exercise 1 (prefix language model) carries the content that S1 says the body lacks. Keep the exercise, but the body must state the idea first.
- Exercise 3 part 1 contradicts the body's "needs" (ED-10); fix the body.
- Exercise 6 (time's arrow) is correct: by stationarity both sides equal $H(X_1, \ldots, X_{k+1}) - H(X_1, \ldots, X_k)$; for the cyclic chain both are $H(0.9, 0.1) = 0.325$ nats and the reversed chain steps backward, so a path reveals the direction although the entropies agree. This is a Cover-and-Thomas-quality problem. Its part 3 asks why the measured edges may differ; after ED-5 the body should already have said that the model is undertrained at the edges.
- Exercise 4 part 3 (composing cross-attention with encoder self-attention) is the analysis ED-13 wants in the body.
- Exercise 8 (Perceiver for reversal) is the experiment ED-14 wants in the body.
- To cut if volume must fall: exercise 2 part 2 (masking rates; three retrainings for a result the text half gives away) and exercise 7 (sorting; long, and its lesson overlaps exercise 4).
- To add: a short exercise that counts training signal per forward pass for the causal and the masked objective at equal compute, which supports the "why decoder-only won" paragraph of S1.

## Still open from October

"Cut: Perceiver IO and descendants": the descendants are gone, the Perceiver IO paragraph and the timing study remain (ED-14). "Keep: reversal as a task with known alignment": kept, but the paragraph that reads its result now undermines it (ED-13).

## For the chapter review

- Order: fine after `sec_kv-cache` and before `sec_vision-transformer`, which is the encoder wiring applied to patches. The section should hand the next one two things it currently does not: pooling to one vector (ED-2) and learned queries as the interface between an image encoder and a language model (ED-14).
- It assumes `d2l.TransformerBlock`, `d2l.MultiHeadAttention` with `attention_weights`, `causal_mask` and `padding_mask` (attention chapter), `train_steps` (`sec_adam`), `LMOptimizer` (`sec_gpt`). No appendix dependence. Forward references: `sec_perf_model` for `Benchmark` (wrong home, ED-15) and `chap_nlp_pretrain` described as the Language Models part (wrong part, ED-7); neither is load-bearing.
- The section label is still `sec_transformer`, a leftover from when this was the chapter's main section. Renaming it is a mechanical job for the editing pass (grep first).
- Overlap: the copy argument is told twice inside the section; `Benchmark` is described here and in `kv-cache.md`; the caution about reading attention maps is owned by `sec_multihead-attention` and is correctly cited, not retold.
- Suspected chapter-wide patterns, shared with `kv-cache.md`: eager-PyTorch timings whose flat part is launch overhead but is explained as arithmetic; exercises at 40 to 50% of body length that hold the section's best experiments (here exercises 1, 4 and 8 carry what the body should); and an absence of judgment about current practice. The chapter introduction says the block "now runs language, vision and speech models"; this is the section where the reader should learn in which wiring, and it does not say.
