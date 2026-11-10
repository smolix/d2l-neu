# Further Reading

Source: `chapter_computational-performance/further-reading.md`, read in full (the file has no slides section). About 2,800 words, 24 topics under six headings that follow the chapter's sections, about 30 citations. No code.

## Verdict

A careful section of the right kind: each topic starts from something the chapter taught and says what the cited work changes, the opening states the selection rule ("one representative of each idea"), and it says that nothing else in the book depends on it. The arithmetic I could check is right (ridge points 315, 312, 453 and 230; a dozen tokens per second for 70 GB at 819 GB/s; a third of a second for 8 GB at 23 GB/s; the bus-bandwidth factors). It has three faults. It does not rank: 24 topics stand at equal weight, and a reader cannot tell the three they will need in their first week of real training (fragmentation, compile caches, time to a target loss) from history (Theano, parameter servers) or from catalogue (four accelerators with four-digit rates). Three of its entries hold exactly what the case study lacks, so the teaching chapter ends without them while the optional section has them. And "Other Accelerators" repeats the hardware section's fault in a purer form: vendor numbers of late 2026 to four digits, which will be stale before the book is printed. The most important fix is the second: move the substance of "The Time of a Transformer Step", "Utilization as a Reported Number" and "Time to a Target Loss" into the closing of `fast-transformer.md` and keep here only the citations.

## Scores

| dimension | score | reason |
|---|---|---|
| Clarity of explanation | 4 | Each entry states a problem and a change in a few sentences. |
| Clarity of thought and focus | 3 | No ranking; a device catalogue; an entry that exists to translate the chapter's own notation. |
| Pedagogy | 3 | The reader is not told what to read first or what is here for completeness. |
| Language | 4 | Plain and exact. Many API names; a uniform entry formula. |
| Ease and enjoyment | 3 | Reads as a reference list; the entries with a judgment are the ones that stay. |

## What works

- **The opening sentence** names the chapter's selection principle: "one roofline with one memory bandwidth, one GPU and its links, one compiler per framework, three ways to save memory, one collective algorithm, and one transformer made fast on one machine."
- **"Only MFU compares systems fairly, because it credits only the work that the model needs."** A judgment with its reason.
- **"Time to a Target Loss."** "Throughput in tokens per second rewards any change that makes a step faster, even one that makes training need more steps." This is the caveat the case study needs (FT-19 in the review of `fast-transformer.md`).
- **"Unified Memory."** One machine, one division (819 GB/s over 70 GB), one verdict: such machines "trade a low compute roof for capacity, which suits decoding a single stream of a large model and suits training poorly."
- **"Parameter Servers and Asynchronous Updates"** says why the older design lost and where it survives. History with a conclusion.
- **"Writing Fused Kernels"** states correctly what the chapter's compilers do: they "fuse chains of elementwise operations and reductions, but they do not find algorithmic rewrites". This is the accurate version of a claim that `compilation.md` overstates (COMP-7 there); the two should agree on this wording.

## Structural findings

1. **No ranking** (FR-1).
2. **Three entries belong in the case study** (FR-2).
3. **A device catalogue** (FR-3). The hardware review asks for per-product detail to leave the teaching section; it should not land here at the same weight. The Tools appendix (`sec_hardware_buyers`) and the Language Models hardware chapter (`chapter_lm-hardware`, a skeleton today) are the homes for it.
4. **This file is the natural receiver** for material the section reviewers want out of the teaching sections: the format detail of HW-8 (already duplicated here under "Block-Scaled Formats in Detail"), the NCCL transport measurement of `multi-gpu-practice.md` (S3 there), TorchScript and `torch.export` (already here, under "Exporting a Captured Graph"; the compilation review asks for one sentence in the section). After those moves the ranking of FR-1 becomes necessary.
5. **One gap.** Nothing on the input pipeline (data loading as the bottleneck, DALI, decoding on the GPU), although `hardware.md` gives the host's second job two sentences and an exercise, and the case study removes the pipeline by putting the corpus on the GPU. For many readers the first overhead they meet is a starved loader. One entry under "Measuring and Modeling" would serve them.

## Passage-level findings

**FR-1** (moderate, thought). Opening.
> of the chapter, topics they had no room for. They range from finer

The opening lists the six groups twice (once as what the sections taught, once as what this section adds) and ranks nothing. Replace the second list by a judgment. Draft: "Three topics here will matter in a reader's first real training run: memory fragmentation, the cost and caching of compilation, and the difference between throughput and time to a target loss. The entries on performance models and on collective algorithms are for readers who will write kernels or run clusters. The entries on earlier frameworks and on parameter servers are history."

**FR-2** (moderate, pedagogy). "The Time of a Transformer Step", "Utilization as a Reported Number", "Time to a Target Loss".
> The tensor contractions perform over 99% of the FLOPs of a training step

The case study opens by asking where the time goes, ends at "about half of the card's bf16 peak", and never closes its account (S2 in its review). This entry holds the published form of the missing answer: nearly all FLOPs in the products, about three fifths of the time, the rest in bandwidth-bound kernels. The next entry defines the utilization number that the case study computes without naming (its exercise 2 names it), and the third states the caveat that the case study's own training check runs into (FT-19). All three are the chapter's conclusions, not further reading. Put one sentence of each into the last paragraphs of `fast-transformer.md` (the book's own split of the step beside the BERT one; the name MFU for the last column of its table; throughput is not progress), and keep the citations here.

**FR-3** (moderate, focus and currency; numbers unverified). "Other Accelerators".
> 8 TB/s and runs 2,517 dense bf16 TFLOP/s, a ridge point of about 310

Four accelerators with memory, bandwidth and rate to four digits, "In late 2026", then a second paragraph on which format runs at which rate on which product. The memory sizes and bandwidths match what I know of these parts; the throughput figures (2,517, 2,307, 1,678, 667) and the claims "fp6 runs at the fp4 rate on the MI355X but at the fp8 rate on NVIDIA's Blackwell GPUs" I cannot confirm. The second paragraph names "TPU v6e", a part the first paragraph did not introduce (it lists TPU7x). The one fact a reader should keep is in the middle: every such part has a ridge point between about 230 and 450 FLOP/byte, so the chapter's regimes carry over unchanged. Keep that sentence with the vendor names, round to two digits or drop the rates, and delete the second paragraph. "The vendors' datasheets give these values" is not a citation.

**FR-4** (minor, focus). "Block-Scaled Formats in Detail".
> :numref:`sec_hardware` explains why 4-bit formats need a scale for every

This entry and the formats paragraph of `hardware.md` (HW-8) give the same facts (blocks of 32 with a power-of-two scale, NVFP4 with blocks of 16). If HW-8 is applied, this entry is the single home and should open from the problem and not from the reference; if not, one of the two goes. The count "adds a quarter bit per value" and "half a bit per value" is correct (8/32 and 8/16) and is the kind of number worth keeping.

**FR-5** (minor, correctness: unverified and will date). "Compile Time at Scale", "Kernel Replay beyond Inference", "Choosing What to Recompute", "Fragmentation".
> kernels as CUDA graphs, which it calls command buffers, by default.

Eight framework names in four entries: `torch.compiler.nested_compile_region`, `save_cache_artifacts` and `load_cache_artifacts`, `jax_compilation_cache_dir`, `mode='max-autotune'`, `torch.compiler.cudagraph_mark_step_begin()`, `torch._functorch.config.activation_memory_budget`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. I believe each exists in current releases; none was checked against the pinned versions, one is private, and all are of the kind that gets renamed. The sentence quoted repeats COMP-13 (the default of an internal XLA flag). A further-reading section may name tools, but it should say once that the names are those of late 2026, and it should keep the idea in the sentence and the name in parentheses, as most entries already do.

**FR-6** (moderate, thought). "Bus Bandwidth".
> the time $2N/\beta$ of a large allreduce of $N$ bytes, so it is twice the

Correct (algorithm bandwidth $N/t$, bus bandwidth $2(k-1)/k$ times that, and $\beta = 2N/t$). But the entry exists because the chapter defines its link rate in a way no tool reports, and so needs a translation into the two rates that tools do report. This is the cost of MG-8: four rates for one link. If the multi-GPU sections define the link rate directly as the rate at which one device sends during a large allreduce, that is the bus bandwidth, and this entry reduces to two sentences saying what nccl-tests prints.

**FR-7** (minor, correctness). "Beyond One Machine".
> Training in fp8 and fp4 needs scales per tensor or per block and fp32

The two citations are the FP8 formats paper and the DeepSeek-V3 report (checked in `d2l.bib`); both concern fp8. Nothing cited supports training in fp4. Write "Training in fp8" or add a source for fp4. In the sentence before, tensor, pipeline and expert parallelism "split a model too large for data parallelism": the chapter has just shown sharded data parallelism holding a model that no GPU holds, so the reason is too narrow. They split a model whose step is too slow or whose communication is too large under sharded data parallelism.

**FR-8** (minor, pedagogy). "Latency Hiding".
> requests are in flight to cover the latency of each one, as the exercise

The entry leans on "the exercise on Little's law in :numref:`sec_hardware`", an unlabelled pointer to an exercise. The hardware review asks for Little's law to move into the body (its structural finding 2); if that happens this entry can cite the passage, and it then explains the one thing the body cannot: why small kernels stay below the bandwidth roof.

**FR-9** (minor, correctness: unverified numbers). Several entries.
> Fusing these operators and choosing data layouts that suit each kernel

Numbers that match my memory of the sources and were not confirmed here: Ivanov et al., over 99% of FLOPs, 61% of runtime, 1.3 times; the M3 Ultra with 512 GB at 819 GB/s; 8-bit Adam at 2 bytes per parameter; Goyal et al. with minibatches of 8,192; the speedrun on "one node of eight GPUs" with "fp8 products for some layers" (I believe the output layer). The first four are stable facts about papers and products and can stay. The speedrun's ingredients change with every record; say "at the time of writing".

**FR-10** (minor, language; pattern). Throughout.
> The break-even point of :numref:`subsec_comp-wholestep` counts the time a

Fourteen of the 23 topic entries open with a reference in their first two lines ("The X of :numref:..." or ":numref:... does Y"). It is serviceable, and over 23 entries the uniformity shows. Open a third of them from the problem, as "Host memory is larger than GPU memory by an order of magnitude and cheaper per byte" and "Throughput in tokens per second rewards any change that makes a step faster" already do; those two are the best openings in the file.

## Code and experiments

None. Arithmetic recomputed: ridge points of the four accelerators (315, 312, 453, 230, against "about 310", "about the same size", "about 450", "about 230"); 819/70 = 11.7 tokens per second; 8 GB at 23 GB/s = 0.35 s; the bus-bandwidth factors at $k = 2$.

## Still open from October

Not applicable.

## For the chapter review

- Three entries hold the conclusions of the case study (FR-2). The chapter's last teaching section should end on them.
- "Other Accelerators" is the hardware section's catalogue problem again (FR-3); a chapter-level decision on where dated product numbers live (one dated table, the Tools appendix, or the Language Models hardware chapter) should cover both.
- "Bus Bandwidth" is a symptom of the $\beta$ notation of the multi-GPU sections (FR-6).
- "Writing Fused Kernels" contradicts the "epilogue" claim of `compilation.md` and is the correct one of the two.
- No appendix reference. Two pointers to the Language Models part, both in a section that exists to point outward; not load-bearing.
- The constant 23 GB/s of the capture machine is used here once more as a fact ("at the 23 GB/s of :numref:`subsec_hw-cpu`").
