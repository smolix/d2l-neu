# Further Reading

Source: `chapter_attention/further-reading.md`, read to the slides marker. About 2,300 words, 21 topics under five headings that follow the chapter's sections, about 55 citations. No code.

## Verdict

A careful and current annotated bibliography, better than most. Each topic starts from something the chapter taught and says what the cited work changes, the opening states the rule ("one representative of each idea" in the sections, the rest here), and it says that nothing else in the book depends on it. As far as I can tell the claims about papers and models are accurate; I could confirm none of them against the sources in this session and list the ones that carry numbers. Its weakness is the one the brief asks about: it does not rank. Twenty-one topics stand at equal weight, and a reader cannot tell the two or three that will change how they build a model (RoPE bases in released models, sinks, gated recurrences) from the ones that are history (Performer, Linformer, Reformer) or curiosities (ASIF). Two entries also state something about the chapter that the chapter does not say. The most important fix is a first paragraph, or one sentence per heading, that says what to read first and what is here only for completeness.

## Scores

| Dimension | Score |
|---|---|
| Clarity of explanation | 4 |
| Clarity of thought and focus | 3 |
| Pedagogy | 3 |
| Language | 4 |
| Ease and enjoyment | 3 |

## What works

- The opening sentence: "The sections of this chapter teach one representative of each idea: one soft lookup, one scoring function, one way to split attention into heads, a few ways to encode position, one exact kernel and two ways to avoid the quadratic cost, and one circuit traced through a small model." It tells the reader what the chapter's selection principle was.
- Judgments, where they occur, are of the right kind: "None of them is exact, and in practice they were displaced by exact attention in fused kernels, by windows, and by the gated recurrences below", and on differential attention, "No widely used open model has adopted it so far". More of the entries should end this way.
- "Sharing Keys and Values" explains why two published results differ (uptraining a converged checkpoint against training from scratch). That is the kind of note a further-reading section exists for.
- "RoPE in Trained Models" connects the fast and slow pairs of the positional section to what a trained Gemma does with them. The section's own analysis gains from it.
- "Length Generalization of ALiBi and NoPE" and "Perplexity and the Use of Long Contexts" say what the positional experiment could not: that a flat perplexity curve need not mean the context is read.

## Structural findings

1. **No ranking** (FR-1).
2. **Division of labour with the sections.** The sections still carry material that belongs here (the grouped-query results of `multihead-attention.md`, MHA-12; released models in `attention-at-scale.md`, structural finding 4 of that review; the reporting third of `what-attention-computes.md`, structural finding 1 of that review), and this file carries one explanation that belongs in a section (why the ALiBi curve is flat, FR-8 and POS-14). The headings here already match; the move is mechanical.
3. **Two topics are off the chapter's line.** Adaptive kernel bandwidths and ASIF (FR-6) concern kernel regression and representation transfer, not attention layers.
4. **One gap.** There is no entry on where self-attention came from (intra-attention in Cheng et al. 2016 and Parikh et al. 2016, the structured self-attentive embedding of Lin et al. 2017, which the multi-head section cites), although "Monotone Alignment" gives that service for cross-attention. If the chapter gains a proper treatment of self-attention, a three-sentence entry under "Heads" should accompany it.

## Passage-level findings

**FR-1** (moderate, thought). Opening.
Anchor: `cost, and the tools that analyze circuits in large models. Each topic states`
The opening lists the five groups twice (once as what the sections taught, once as what this section adds) and ranks nothing. Replace the second list by a judgment. Draft: "Three topics here bear on models you will build: the bases and frequency use of RoPE in released models, attention sinks, and the gated recurrences that made linear attention competitive. The approximations of the softmax kernel and the fixed sparse patterns are history, kept for readers who meet them in older papers. The rest is for the curious."

**FR-2** (moderate, correctness; fairly sure). Under "Redundant Heads".
Anchor: `values and that two heads can, and the trained heads of its experiment`
"the trained heads of its experiment nearly duplicated each other" is not what the body of `multihead-attention.md` reports. Its text reads the trained layer as having "at least two clearly different weightings" that "need not be one-hot"; duplicate heads appear only in an exercise ("those that nearly duplicate another head"). With four heads on a task that needs two weightings, duplication is likely and visible in the stored output, but the section does not say so. Either add the sentence to the section (it is a true and useful reading of the four-head row) or write here "with four heads, some of the trained heads repeat one another".

**FR-3** (moderate, correctness; sure of the inconsistency, the motive is unverified). Under "Monotone Alignment".
Anchor: `the first differentiable alignment. Graves's handwriting synthesis`
This corrects the chapter introduction, which calls Bahdanau et al. "the first learned alignment" (see IDX-4 in the review of `index.md`). Make the two agree. Also "a constraint inherited from the left-to-right hidden Markov models of speech recognition" attributes an origin. Graves's window moves forward because its position parameter is incremented by a positive amount; whether the paper presents that as inherited from HMMs I cannot confirm. Write "like the left-to-right hidden Markov models of speech recognition".

**FR-4** (moderate, thought). Under "Sharing Keys and Values".
Anchor: `:numref:`subsec_kv-head-sharing` quoted two results on grouped-query`
The entry exists to reconcile two numbers quoted in a teaching section, and it restates one of them at the last digit ("within a tenth of a point"). If grouped-query attention moves to `chapter_transformers/kv-cache.md`, as the chapter review proposes, this entry moves with it to `sec_transformers-further-reading`, where multi-head latent attention already is. It must then be reconciled with that section's sentence that uptraining "is how GQA spread through existing model families", which the reviewer of that section believes to be false (KV-11 there): this entry's own account, that training from scratch with few key-value heads is the harder case, sits oddly beside the fact that the models that adopted the layout mostly trained with it from the start.

**FR-5** (minor, correctness; unverified numbers). Several entries.
Anchors: `fused kernel is about a sixth faster than FlashAttention-2 on an H100.`, `factor of 2.6 against a single scale per tensor. FlexAttention`, `FlashAttention-2 supports, its kernels run at 0.68 to 1.43 times the speed`, `original budget. The second trained a 7B dense model from scratch on 1.33T`, `lengths up to 256k it extends the context to 2048k tokens.`
Each number matches my memory of the paper (sigmoid attention: 17% on H100; FlashAttention-3: 2.6 times lower error in FP8; FlexAttention: 0.68 to 1.43 times; DeepSeek-V2 ablation: 1.33T tokens; LongRoPE: 2048k), and none could be confirmed here. More to the point, a benchmark ratio at two digits on one GPU generation will be stale within the life of the book and serves no reader. Keep the qualitative statement ("somewhat faster", "roughly on par with the handwritten kernel", "by a factor of eight") and let the citation carry the digits. Also unverified: that Llama 4 applies a per-head softmax temperature at inference to its position-free layers (`Llama 4 :cite:`Meta.2025` applies the same fix at inference time to the`); I believe the released configuration has such a setting.

**FR-6** (minor, focus). Under "Bandwidths That Follow the Data".
Anchor: `works as a representation without any training: ASIF`
Adaptive kernel density estimation and ASIF are two unrelated pointers under one heading, joined by "The same lookup also works as". Neither is something a reader of an attention chapter would look for. Cut the heading, or keep one sentence on adaptive bandwidths inside "Attention as Associative Memory", where the temperature is already discussed.

**FR-7** (minor, dependencies). Four entries.
Anchors: `the length, belongs to the Language Models part of the book.`, `of the Language Models part.`, `Models part of the book treats these methods on large models.`, `Language Models part.`
Four pointers to the Language Models part, none with a label. They are not load-bearing, but they are promises about chapters that are skeletons today. Give each a label once the target section exists (`chapter_lm-inference/decode-kernels.md` already cites `eq_online-softmax` back), or drop the sentence.

**FR-8** (moderate, pedagogy). Under "Length Generalization of ALiBi and NoPE".
Anchor: `window near the training length, so the extra tokens are effectively never`
This is the explanation that decides how the extrapolation table of `positional-information.md` should be read, and the section itself gives it three lines and an exercise (POS-14, structural finding 3 of that review). Move the explanation and one measured number into the section; keep here only the citations and the NoPE results.

**FR-9** (minor, correctness). Under "Gated Recurrences and Hybrids".
Anchor: `:numref:`sec_attention-at-scale` ended with the observation that plain`
The cost section does not end with this observation (it ends with a timing figure), and "two changes made it competitive" repeats the overstatement of COST-16. Write "noted that plain linear attention models language less well than softmax attention, and that gating and a different update rule narrowed the gap".

**FR-10** (minor, language; pattern). Throughout.
Anchor: `The window of :numref:`subsec_attn-receptive-field` needs many layers before`
Nearly every entry opens with "The X of :numref:..." or ":numref:... found/proved/showed that". It is serviceable and uniform; over 21 entries the uniformity shows. Vary a third of the openings by starting from the problem ("A window needs many layers before distant tokens meet.").

## Code and experiments

None.

## Still open from October

Not applicable.

## For the chapter review

- The file is the natural receiver for the paper-reporting paragraphs that the section reviewers want out of the teaching sections (MHA-12, the gpt-oss, Llama 4 and FlashAttention-2 and -3 material of the cost section, the last third of the circuits section). After those moves it will be about 3,000 words, which makes the ranking of FR-1 necessary.
- It has no reference into the appendix and four unlabelled pointers to the Language Models part.
- It contradicts the introduction on who was first (FR-3, IDX-4).
