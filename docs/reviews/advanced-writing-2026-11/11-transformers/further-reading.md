# Further Reading

Source: `chapter_transformers/further-reading.md` (the file has no slides; read in full, 369 lines). About 2,750 words in 27 subsections under six headings, with about 73 citations. No code, no figures, no exercises.

## Verdict

As a set of notes this is well above the usual further-reading section: each entry says in one or two exact sentences what a method changes, most say how it relates to something the chapter built, and several are small explanations in their own right (the acceptance rule of speculative decoding, why the key up-projection of latent attention cannot be absorbed through a rotation). Against the bar it has two faults. It ranks nothing: 27 entries of nearly equal length, from serving engines that every deployed model depends on to a three-line note on emergent abilities, with no sentence telling the reader which five to read. And it is used as the overflow of the sections without the sections knowing: several entries hold material the section reviews say the body lacks (which wiring to use today, what learned queries are for), several repeat a section paragraph almost word for word, and three describe what a section "quotes" or "describes" in terms that will be false once the sections are revised. One statement contradicts the chapter's own figure, and one presents two rival analyses as a settled division of labour. The most important fix: open each of the six headings with one sentence of judgment (what a practitioner needs from this group, what is a research direction), and move the two entries that answer a section's main question into that section.

## Scores

| dimension | score | note |
|:--|:-:|:--|
| Clarity of explanation | 4 | Entries are exact and self-contained; a few compress a paper into a clause that only its readers can unpack. |
| Clarity of thought and focus | 3 | Organized by section, not by importance; overlaps with four sections; one contradiction with a figure. |
| Pedagogy | 2 | No ranking, no route, no statement of what is established practice and what is one paper's result. |
| Language | 4 | Plain and exact, no mannerisms. One register throughout. |
| Ease and enjoyment | 3 | Good to consult, tiring to read through. |

## What works

- **"Nothing else in the book depends on this section."** The right promise for a section of this kind, stated once.
- **Speculative decoding.** The entry gives the acceptance probability $\min(1, p(x)/q(x))$, the residual distribution, and the consequence ("leaves the output distribution exactly that of the large model"). A reader could implement it from these four sentences.
- **"A benchmark of single questions hides the cost of such evictions."** A judgment, followed by the evidence. This is the voice the rest of the section lacks.
- **Constant rates and cooldowns.** It ties a schedule to a cost the reader has just paid: one long run with cooldowns branched off replaces the fifteen separate runs of `sec_scaling-laws`.
- **Expert-choice routing** and **Sigmoid Gates and Limited Groups** each say exactly what the chapter's layer does and does not adopt.
- **Initialization of Deep Stacks.** SmallInit, OLMo 2's return to a plain normal, and muP in five sentences, each with what it changes.

## Structural findings

1. **No ranking.** The reader cannot tell that continuous batching, paged caches and prefix caching are how every serving system works, while expert-choice routing is one paper's alternative and "Loss and Abilities" is an open debate. Give each of the six headings a lead sentence that judges, and mark the handful of entries that describe settled practice. A candidate for the short list: bounds on the logits (QK-norm is in current models), serving engines, speculative decoding, uses of each wiring, self-supervised and image-text encoders, sparsity and expert width.
2. **Two entries answer questions the sections leave open, and belong in the sections.** "Uses of Each Wiring" is the answer to the main complaint about `encoders-decoders.md` (ED-1: the reader does not learn which wiring to use today); "Learned Queries as an Interface" delivers the vision-language connection that the section's introduction promises and its body drops (ED-14). Neither is optional reading. Move the substance of both into the section and leave here the comparison of Wang et al. and T5Gemma.
3. **Duplicates of section text.** "Serving Engines" repeats the closing paragraph of `Batched Decoding` in `kv-cache.md` (KV-10); "Sparsity and Expert Width" cites the two studies that `moe.md` retells (MOE-17); registers appear here and as a subsection of `vision-transformer.md` (VIT-5); the DeiT story is here and twice in that section. In each case the section reviewer recommends that this file be the single home. After that revision this file should be checked again, since its entries are written as supplements to passages that will be gone (FR-5).
4. **The chapter's resource list should move here** (see the review of `index.md`, structural finding 3). At present the chapter has an unnumbered "Resources and Further Reading" in its introduction and this numbered "Further Reading", and no section cites this one (`sec_transformers-further-reading` is referenced only from `index.md`).
5. **Promises to skeleton chapters.** "The Language Models part" is named ten times as the place where a topic is treated, never with a label. See FR-4.

## Passage-level findings

**FR-1** (major, pedagogy). Opening paragraph. Quote: "This section points to topics they had". The paragraph is two lists of seven items each (what the sections build, what this section adds) and no judgment. Replace the second list with a route: what to read if you will train models (the block and the recipe), if you will serve them (generation and the cache), if you work with images (wirings and images), and what is research in progress (most of experts and scaling).

**FR-2** (moderate, correctness; confidence high, checked against the source). Under "Other Placements and No Normalization". Quote: ":numref:`fig_norm-taxonomy` drew four placements of the normalization. OLMo". The sentence continues "2 :cite:`OLMo.2025` uses a fifth: it normalizes the output of each branch before adding it to the stream". The figure's caption in `transformer-block.md` reads "OLMo 2 normalizes what the branch writes", so OLMo 2's placement is one of the four, not a fifth. Delete the sentence, or, if the taxonomy figure moves here as the block reviewer recommends (TB-14), let this entry own all four.

**FR-3** (moderate, correctness of a claim about papers; unverified). Under "Reconciling Kaplan and Chinchilla". Quote: ":citet:`Porian.Wortsman.Jitsev.ea.2024` account for the rest: the FLOPs of". As I recall the two papers, each offers its own account of the same discrepancy (Pearce and Song: the count of non-embedding parameters at small scale; Porian and coauthors: the cost of the output layer, warmup length, and tuning by scale), and neither presents itself as covering what the other leaves. "Account for the rest" and the closing "With these corrected, the two fits agree." turn two analyses into one settled explanation. The scaling-laws reviewer raises the same point from the other side (SL-11). Write: "Two analyses explain the gap. ... Each recovers most of it with its own corrections." Check both abstracts before the revision.

**FR-4** (moderate, dependency). Pattern, ten instances. Quotes: "The Language Models part builds these"; "The Language Models part treats the design in its own section."; "The Language Models part treats sparsity in scaling laws."; "are treated in the Language Models part." Those chapters are skeletons today. None of these pointers is load-bearing, but ten unlabelled promises in one section read as deferral, and they cannot be checked. Use the labels that exist (`chap_lm_pretraining`, `chap_lm_encoders`, `chap_lm_serving`) where the target is known, and cut the pointer where it is not. One sentence in the opening paragraph ("several of these topics belong to the Language Models part") is already there and covers the rest.

**FR-5** (moderate, thought: entries that depend on passages recommended for removal). Quotes: ":numref:`subsec_kv-cache-sinks` describes the sigmoid gate of"; ":numref:`sec_scaling-laws` quotes the finding of"; "The register tokens of :numref:`sec_vision-transformer` need". The section reviews recommend cutting the gate paragraph (KV-16), the repeated-data paragraph (SL-10) and the register subsection (VIT-5) from the sections and keeping them here. Each of these entries then has to stand alone: say what the gate, the finding and the registers are, instead of referring to a description elsewhere. "Gated Attention in Full" and "Multi-Head Latent Attention in Full" should also lose the "in Full" of their headings, which presumes the partial version.

**FR-6** (moderate, thought). Under "Other Placements and No Normalization". Quote: "initialization; with these two changes, post-norm stacks of a thousand". This is the scope condition that `transformer-block.md` lacks (TB-5): post-norm is not untrainable at depth, it is sensitive to the scale of its weights. The fact sits here, where few readers will find it, while the section's Summary says that a post-norm stack "collapses its tokens geometrically". When TB-5 is fixed, cite this entry from the section in a clause.

**FR-7** (minor, correctness; unverified). Numbers and attributions that I believe but could not confirm in this session, to be checked against the papers: Gemma 2's caps of 50 and 30; "reached 80% top-1 in 300 epochs" (Beyer et al.); "a sparse model of 269B parameters at the computation per token of a 32B dense model" (ST-MoE); "$R^* \approx 15$" (Muennighoff et al.); "seven to eight bits" (Kumar et al.); "3.5T tokens" and "from about half to about a twentieth" (Qiu et al.; consistent with the "47% to 5%" in `kv-cache.md`); Qwen3-Next's 512 experts and Kimi K2's 384; the description of `Kimi.Team.2025b` as a hybrid with recurrent mixers; and the statement that RMSNorm and gated feed-forward networks are "among the few that do" transfer in Narang et al. One small imprecision I am fairly sure of: dynamic tanh is described as "an elementwise $\tanh(\alpha x)$ with a learned scalar $\alpha$"; the published layer also keeps a learned per-channel scale and shift.

**FR-8** (minor, focus). "Pretraining Data" and "Loss and Abilities". Quotes: "GPT-2 was trained on WebText, about 40 GB of web pages linked from Reddit,"; "such jumps are a property of the models or of metrics that give no credit". The first is one fact and a pointer; the second is three sentences that end in "is debated". Neither tells the reader anything to do or read. Cut the first; for the second, either say which reading the authors find more convincing or cut it.

**FR-9** (minor, thought). Under "Adam's Second Moment at Small Batch". Quote: "The runs of :numref:`sec_gpt` use PyTorch's default $\beta_2 = 0.999$,". A reader who copies the chapter's recipe to a larger run needs this in the recipe itself, not in optional reading: one clause in "The Training Recipe" of `gpt.md` ("we keep $\beta_2 = 0.999$, which suits small batches; large runs use 0.95").

## Code and experiments

None.

## Exercises

None.

## Still open from October

Nothing specific to this file.

## For the chapter review

- This file is where the section reviewers want to send five passages (taxonomy of placements, serving paragraph, gate paragraph, registers, two scaling studies of mixtures). It can take them, but only if it gains a ranking; otherwise the chapter's survey material merely changes address.
- It also holds two things the chapter's body needs (uses of each wiring, learned queries in vision-language models) and one scope condition the block section needs (DeepNet).
- Ten unlabelled pointers to the Language Models part; one contradiction with `fig_norm-taxonomy`; one contested attribution stated as settled, matching SL-11.
