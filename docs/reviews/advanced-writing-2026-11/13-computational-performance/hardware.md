# Hardware

Source: `chapter_computational-performance/hardware.md` (lines 1 to 821; about 5,300 words of prose and exercises, of which about 1,550 are exercises; 4 PyTorch cells including imports). Stored outputs read: `outputs/pytorch/.../hardware.json`, `outputs/jax/.../hardware.json`. Figures rendered: memory hierarchy, bandwidth ladder, latency ladder, shoreline, energy ladder, float formats, PCIe topology.

## Verdict

The section has the right ambition and says so: "The numbers below are those of late 2026. They will change, but the relations between them change slowly." Where it follows that programme it is very good: the opening calculation (a 1B-parameter GPT decodes at most 500 tokens per second while the arithmetic units idle 99% of the time), the area-against-perimeter argument, the energy ladder, and the batch sweep that walks one matrix from the bandwidth roof to the compute roof. Every hardware number I can check from certain knowledge is right, and the arithmetic is right throughout. But about a third of the body is a device catalogue written at the same weight as the principles: four GPUs in a table with a footnote on GeForce accumulation rates, Gaudi 3, B300, CDNA 4, NVFP4, which GeForce cards had an NVLink connector, which cards lack peer-to-peer transfers. That material will be wrong or irrelevant within two product cycles and it is where the section drags. Meanwhile several of the principles that would outlast the devices are hidden in the exercises (the sweep time C/β is about 20 ms on all four GPUs; the power budget caps bandwidth; Little's law sets the parallelism a kernel must expose). **Most important fix:** separate the invariant from the snapshot. Keep the argument in the text, put the dated numbers in one clearly dated table, cut the per-device asides, and promote two or three of the invariants from the exercises into the body.

## Scores

| dimension | score | reason |
|---|---|---|
| Clarity of explanation | 4 | Each idea arrives with a number and usually a figure. The energy argument and the "chip boundary" rule are stated more firmly than they are explained. |
| Clarity of thought and focus | 3 | Principle and catalogue are interleaved at equal weight; the decode bound is computed three times with two answers (500 and 450); the bandwidth measurement and the SM paragraph repeat the previous section; the last part re-teaches chapter 11. |
| Pedagogy | 3 | Lines 62 to 370 are told, not shown, apart from one cell that repeats an earlier measurement. Latency and access pattern are never measured. The exercises are the best part and carry insights the body should state. |
| Language | 4 | Clean, short sentences, no mannerisms. Dense noun strings in the formats paragraph. |
| Ease and enjoyment | 3 | The opening and the shoreline are memorable; "Tensor Cores and Number Formats" and "Accelerator Specifications" read like a datasheet digest. |

## What works

- **The opening example** (lines 13 to 22). "The arithmetic for one token, about $2 \times 10^9$ FLOPs, takes the tensor cores 12 µs, while reading the 2 GB of weights takes the memory 2 ms." Two divisions and the reader sees the whole section's subject. Checked: 12.1 µs, 2.0 ms, 0.6% busy.
- **The stated programme** (lines 33 to 37): "memory is orders of magnitude slower than arithmetic, each chip boundary lowers bandwidth by roughly another order of magnitude, and compute grows faster than bandwidth, which grows faster than capacity." This is the voice the author asked for: it judges what will last.
- **Area and perimeter** (`subsec_hw-shoreline`). "Doubling the side length of a die quadruples the number of arithmetic units, but the edge only doubles", with `fig_shoreline` (16 units and 16 pads against 64 units and 32 pads). A geometric reason for a trend, in five sentences and one picture. October listed it as a keeper; agreed. The honest paragraph that follows ("A single generation need not follow the trend") is worth keeping too: it stops the reader from treating a tendency as a law.
- **"The DRAM cells are no faster; there are many more wires, and they are much shorter."** One sentence that explains HBM.
- **The batch sweep** (`hardware-batched-decoding`). Flat at 0.57 ms for B = 1 to 64, then 0.88 and 3.22 ms; 0.94 TB/s at B = 1, 171 TFLOP/s at B = 1024; crossover predicted at B = 164. "One matrix on one card covers both ends of :numref:`fig_roofline`." The cleanest confirmation of the roofline in the chapter.
- **The exercises.** 3 (format halving leaves the crossover batch unchanged except for one table entry), 5 (the ridge point of a link gives a token threshold independent of N), 6 (the DRAM energy of 2014 would put a B200 at 1.3 to 2.6 kW for memory traffic alone), 7 (sweep time) and 10 (Little's law) are original and teach something. I worked 2, 3, 5, 6, 7, 8 and 10 and found no error.

## Hardware numbers

Checked against what I know for certain; "derived" means I recomputed it from other numbers in the section.

**Confirmed.** RTX 4090: 24 GB GDDR6X, 450 W, 128 SMs, 2.52 GHz boost clock, 1.008 TB/s (384 bits at 21 Gb/s per pin), 82.6 TFLOP/s fp32, PCIe 4.0 ×16 (31.5 GB/s per direction), 72 MB of L2 (consistent with "about a tenth of a gigabyte" on the die). RTX 5090: 32 GB GDDR7, 575 W, 1.79 TB/s, PCIe 5.0, fp32 104.8 TFLOP/s so 210 at twice that. H100 SXM: 80 GB HBM3, 3.35 TB/s, 989 TFLOP/s dense bf16 and 1,979 dense fp8, 67 TFLOP/s fp32, 700 W, NVLink 4 at 900 GB/s total. B200: 8 TB/s, 180 GB per GPU in a DGX B200 (192 GB on the part), 2.25, 4.5 and 9 PFLOP/s dense in bf16, fp8 and fp4, NVLink 5 at 1.8 TB/s total, two reticle-size dies, eight HBM3e stacks of 1,024 data wires. Reticle limit 26 mm by 33 mm = 858 mm². NVMe on PCIe 5.0 ×4 at about 14 GB/s. DDR5-6400 at 51.2 GB/s per channel, so 8 to 12 channels give 0.4 to 0.6 TB/s per socket. V100 (2017) at 125 TFLOP/s, 0.9 TB/s, 32 GB. Horowitz (2014), 45 nm: 0.03 pJ, 3.7 pJ, 10 pJ, 100 pJ, 1.3 to 2.6 nJ, and the figure's 0.2 pJ and 1.1 pJ. fp4 E2M1 has fifteen distinct values from −6 to 6. Microscaling formats share one power-of-two scale per 32 values; NVFP4 uses blocks of 16. E4M3 and E5M2. The RTX 3090 and 3090 Ti were the last GeForce cards with NVLink. Gholami et al.: 3.0, 1.6 and 1.4 times per two years.

**Derived and correct.** Generational growth 2.6, 2.0 and 1.8 (cube roots of 18, 8.9 and 5.6); ridge points 139, 295, 281, 164, 117; H100 to B200 factors 2.3 and 2.4; 0.17 s and 4.4 ms for 4 GB; factor of thirty between 1 TB/s and 32 GB/s; "about as much as 500 fp32 multiplications" (350 to 700); 16 GB for Adam in fp32; 0.54 GB for the $16{,}384^2$ matrix.

**Unverified (what I believe).**
- 165 TFLOP/s bf16 and 330 TFLOP/s fp8 for the RTX 4090; 419 and 1,676 for the RTX 5090; the caption's "from vendor datasheets". NVIDIA publishes no tensor-core TFLOP/s for GeForce cards that I know of. The stored measurements (157 to 172 TFLOP/s in bf16) support the 165. Nothing in the book measures fp8 or fp4 on either card.
- "GeForce cards run fp8 and fp16 tensor arithmetic at half rate when they accumulate in fp32, and fp4 carries no such penalty." I cannot confirm this; it carries the whole explanation of the 5090's fp4 entry.
- "Intel's Gaudi 3 runs fp8 at its bf16 rate" (I believe Intel quotes one figure of about 1.8 PFLOP/s for both), "NVIDIA's B300 runs fp4 at three times its fp8 rate", "AMD CDNA 4" block scaling.
- The GPU cache and memory latencies of `fig_latency_ladder` (15 ns, 100 ns, 200 to 300 ns) attributed to Luo et al. (2024). The paper exists and benchmarks the RTX 4090, A100 and H100; I cannot confirm the values.
- "NVIDIA also does not support peer-to-peer transfers between RTX 4090 or RTX 5090 cards": true of the 4090 with the stock driver as far as I know; the 5090 I cannot confirm.
- "5–15 µs of host time" and "1–5 µs" for a kernel launch: plausible and consistent with the 12 µs measured in the previous section.
- "opening a new row takes on the order of a hundred nanoseconds": the row activation itself is nearer 15 to 50 ns; about 90 ns is the full access as a CPU sees it (the figure's own DDR5 entry). Loose, not wrong.
- "one RTX 4090 delivers more than ten times the fp32 arithmetic of a 64-core server CPU": 64 cores with two 512-bit FMA units at 3 GHz give about 12 TFLOP/s, a factor of 7 against 82.6; "more than ten" holds only for CPUs with half-width units or against the bf16 rate. Write "roughly ten times".
- `:cite:`Kung.1988`` for the systolic array: the classic reference is H. T. Kung, "Why systolic architectures?" (1982); check which work the key points to.

**No prices appear in the section** (only the 0.30 dollars per kWh of Exercise 6), which is the right choice.

**What will date fastest**, most perishable first:
1. `tab_gpu_specs` as "Representative GPUs in late 2026": its newest datacenter part was announced in 2024, and the next generation is due around publication. The B300 aside and the 5090 fp4 entry belong to the same half-life.
2. The link generations (NVLink 4 and 5, PCIe 4.0 and 5.0, "GB200 NVL72"), the NVMe figure, and the bandwidth ladder's PCIe 6.0 and 800G entries.
3. Driver policy: no peer-to-peer on the 4090 and 5090; the NCCL rates quoted downstream.
4. Format support by product (fp8 on Hopper and Ada, fp4 on Blackwell and CDNA 4, NVFP4).
5. The absolute ridge points (164, 281) and the generational factors, which shift with each new part.

**What will last:** the hierarchy and its orders of magnitude; area against perimeter; the reticle limit; energy of moving against computing (the ratios, not the 45 nm values); the latency of DRAM, "nearly constant for decades" as Exercise 10 says; 16 bytes per parameter; tokens per second ≲ β over bytes per token; and the sweep time C/β of about 20 ms, which no sentence in the body states.

## Structural findings

1. **Principle and snapshot are interleaved.** The reader cannot tell which sentences to remember. Compare "Arithmetic units fill the interior of a die, so compute scales with the die's area" with, at the same weight a page later, "Intel's Gaudi 3 runs fp8 at its bf16 rate, and NVIDIA's B300 runs fp4 at three times its fp8 rate." Direction: (a) keep `tab_gpu_specs` as the one dated snapshot, caption it with the date and say that the reader should recompute the ridge row for their own part; (b) delete or move to `sec_hardware_buyers` the asides that serve only the table (the GeForce accumulation footnote, Gaudi 3, B300, CDNA 4, which cards had NVLink connectors); (c) end each principle with the sentence that survives a new product: "whatever the part, divide P by β".

2. **Invariants hidden in the exercises should be in the body.** Three are worth a paragraph each:
   - *Sweep time* (Exercise 7): C/β is 24, 22.5, 24 and 18 ms for the four GPUs of the table. So the largest model that fits on any of them decodes a single stream at 40 to 55 tokens per second, on a consumer card or a datacenter part, in 2022 or 2026. That is the strongest "outlasts the device" fact available to the section and it is one division per column.
   - *Power caps bandwidth* (Exercise 6b): at the DRAM energy of `fig_energy_ladder`, 8 TB/s would draw 1.3 to 2.6 kW. This is the missing link in "Energy per Operation" (finding HW-6).
   - *Little's law* (Exercise 10): bandwidth times latency is the number of bytes that must be in flight, about 250 KB on the 4090, which is why a small kernel cannot reach β. This explains the SM paragraph better than the SM paragraph does.

3. **The last part, "Workloads on the Roofline", is the third telling of prefill and decode.** Chapter 11 (`subsec_kv-cache-prefill-decode`, `subsec_kv-cache-batching`) defines both, derives tokens per second ≈ bβ/(W + bnc), and measures them on this card. This section's introduction computes the single-stream bound (500 tokens per second), "Single-Stream Decoding" computes it again (450), and the Summary a third time. By the repetition rule one place owns it. Suggested ownership: chapter 11 owns prefill, decode and the formula; this section's introduction uses the bound once as motivation (it does this well); "Single-Stream Decoding" is cut to the sentence about formats; the batch sweep moves to the performance-model section as the test of the roofline (see that review, structural finding 2) or stays here with one sentence of setup. That removes about 350 words and the 500/450 discrepancy.

4. **Told, not shown, for 300 lines.** Between the imports and "The Host" there is one cell (`hardware-where-bytes-live`), and it repeats a measurement of the previous section. The access-pattern paragraph makes five claims about bursts, random reads and shuffling and measures none; Exercise 9 (a random against an identity gather on the 1 GB vector) is a three-line cell that would show the largest effect in the section. Swap: put the gather in the body in place of the repeated bandwidth cell.

5. **Is the section really two?** At 3,700 words of body it does not need splitting. But it has two subjects: why β is small relative to P (hierarchy, shoreline, energy, the GPU), and what lies outside the GPU (host, links). The second half is what the multi-GPU sections need. If the formats material grows, the natural seam is before "Data Paths outside the GPU". The formats subsection is the part with the weakest claim to its place: `sec_numerics` owns the formats, `sec_memory_precision` owns mixed precision, and the Language Models part is promised the low-precision recipes. What this section needs is two facts (a narrower format moves fewer bytes; it usually raises P; tf32 does only the second). See HW-8.

## Passage-level findings

**HW-1** (minor, thought). Introduction.
> The card therefore produces at most

"about 500 tokens per second" here, from the specified 1 TB/s; "near 450 tokens per second" under "Single-Stream Decoding", from the measured 0.9 TB/s; "roughly 450" in the Summary. One number for one quantity. Use the measured rate throughout or say "about 500" throughout; see structural finding 3 for the larger repetition.

**HW-2** (moderate, explanation). "Bandwidth and Latency".
> **each chip boundary lowers the bandwidth

The rule is bolded, repeated in the introduction, the Summary and two exercises, but what counts as a boundary is not fixed. HBM on the interposer (8 TB/s) and GDDR on the board (1 TB/s) are each one boundary from the die and differ eightfold; the example given right after the rule is a factor of thirty, not ten; NVLink (0.9 TB/s) crosses a package, a board and a cable. The underlying reason is given well a few lines later and again under "Area and Perimeter": the number of wires and their length. State the rule as that ("bandwidth is wires times rate per wire, and each step away from the die cuts the wires by one to two orders of magnitude") and give the three steps with their factors: on the package 8 TB/s, across a board 1 TB/s, across a slot or cable 30 to 900 GB/s. Exercise 2a already makes the wires-times-rate computation; one line of it belongs here.

**HW-3** (minor, explanation). Same subsection.
> takes on the order of a hundred nanoseconds, after which the following

See the number check: this is the latency of a whole random access, not of opening a row. Also, "a random read once for every few bytes it uses" and "a dataset that is read in shuffled order is best shuffled in large blocks" are advice without a size; Exercise 9c asks for the block size. Give the answer in one clause (blocks of latency times bandwidth, about 100 KB on this card by the half-bandwidth rule of Exercise 1).

**HW-4** (moderate, pedagogy, repeated experiment). "Measured Memory Bandwidth".
> multiplying a 1 GB vector by 2 reads and writes every element

The previous section already timed `x + 1`, `x * 1.5`, sine and sigmoid on a 256 MiB tensor and concluded "about 0.9 TB/s, close to the 1.008 TB/s of the specification". This cell measures the same quantity the same way and the paragraph after it ("Every large elementwise operation in a network runs at this rate, whatever it computes, because its bytes and not its arithmetic set its time") restates that section's conclusion. Cut the cell and cite the measurement, or replace it with the gather experiment (structural finding 4). Also "1 GB": the tensor is 1 GiB (1.07 GB); harmless, but the section elsewhere distinguishes carefully.

**HW-5** (minor, correctness of emphasis). "Generational Trends".
> Workloads that were compute-bound a decade ago are bandwidth-bound

True only for operations whose intensity lies between the old ridge and the new one (140 to 290 FLOP/byte), which in the skinny-product picture means batches between about 140 and 290. Most workloads did not change regime. Say that: "an operation at 200 FLOP/byte was compute-bound on a V100 and is bandwidth-bound on an H100". Also "to about 290 on the H100" here and "from 295 to 281" in the next paragraph; round once.

**HW-6** (moderate, thought). "Energy per Operation".
> Energy gives a second reason why data movement dominates.

A second reason for what? The section's question is why β is small relative to P. The paragraph shows that moving a bit costs more energy than computing with it, and then turns to "The techniques of this chapter save energy", which is a different point. The connection to bandwidth is never drawn: a chip has a fixed power budget, so the bytes per second it can afford to move are bounded by watts divided by joules per byte. Exercise 6b contains the argument (8 TB/s at 2014 DRAM energies would cost more than the whole card's 1,000 W). Put that sentence in the body; without it the subsection is an interesting table that does not answer the section's question. Note also that the ladder is from a 45 nm process of 2014; the text says "Newer processes lower every entry, but the order stays the same", which is the right caveat, but a reader computing with these values (Exercise 6) should be told that current HBM is roughly an order of magnitude cheaper per bit (unverified; this is what Exercise 6b lets them infer).

**HW-7** (moderate, focus). "Streaming Multiprocessors".
> That is
> as much microarchitecture as this book needs. When a profiler reports

(Anchor for grep: "as much microarchitecture as this book needs".) The sentence promises a stop and the next one adds *occupancy*, a term used nowhere else in the chapter (I searched all sections; "warp" recurs only in further reading). The paragraph's one consequence, "A kernel that is too small for the machine has too few warps to keep 128 SMs busy", repeats the tile paragraph of the previous section, which points forward to this one; neither place measures it. Give the SM story one home (here), drop occupancy, and connect it to Little's law (structural finding 2) so that "hide memory latency" becomes a number.

**HW-8** (moderate, focus and currency). "Tensor Cores and Number Formats".
> Formats of 8 bits and fewer cover so little range that their values are

This paragraph (E4M3 and E5M2, per-tensor scales, why fp4 fails with one scale, microscaling with blocks of 32, NVFP4 with blocks of 16, which vendors support what) is 150 words of a subject the section then hands away: "The recipes that train large models in these formats belong to the Language Models part." The reader cannot use any of it in this chapter; no cell runs fp8 or fp4, and `sec_memory_precision` skips fp8 training. It is also the most perishable prose in the section. Keep the two bold consequences (fewer bytes; usually a higher compute roof; tf32 the exception), keep the fp4 fact that explains why scaling is needed in one sentence if wanted, and move the rest to the Language Models chapter that owns low precision. The forward reference is then not load-bearing.

**HW-9** (minor, explanation). Same subsection.
> Consumer cards devote less of their area to tensor

"on the RTX 4090 the factor is two in bf16 and four in fp8" is the only justification the chapter gives for P = 165 TFLOP/s, the constant on which the previous section's roofline rests, and it is unsourced (see the number check). If the peak is derived, show the derivation here: cores times two FLOPs per fused multiply-add times clock gives 82.6 for fp32, and the tensor cores double it in bf16 with fp32 accumulation. Then the instruction "On another GPU, replace the two constants" in the previous section can point to a recipe.

**HW-10** (moderate, focus). "Accelerator Specifications".
> The RTX 5090's fp4 entry is four times its fp8 entry, not two times,

A footnote explaining an anomaly in one cell of the table, on a card the book does not use, in a format the book does not run, resting on a claim I cannot verify. It is there because Exercise 3c asks the reader to find the entry "that breaks the pattern". The exercise is good; the paragraph should be one clause in the caption or go.

**HW-11** (minor, currency). The table caption.
> :Representative GPUs in late 2026, from vendor datasheets

See "what will date fastest". Also "from vendor datasheets" is not accurate for the GeForce throughput rows (derived). Suggested caption: "Four GPUs as of late 2026. Memory, bandwidth, links and power from vendor datasheets; throughput of the GeForce cards derived as described in the text." The row "link to other GPUs" for the consumer cards says "PCIe 4.0 ×16 via host memory, 32 GB/s", while the text measures 23 GB/s and the multi-GPU section reports about 2.5 GB/s per GPU for NCCL at default settings; a reader comparing the row across columns (450, 900, 32, 64) gets a far rosier picture of the consumer cards than the chapter later delivers. Add "at most".

**HW-12** (minor, correctness). "The Host".
> 4090 delivers more than ten times the fp32 arithmetic of a 64-core server

See the number check: about seven times against a CPU with two 512-bit FMA units per core. "Roughly ten times", or compare in bf16.

**HW-13** (minor, timing that must survive re-execution). "The Host".
> The pinned copy reaches about 23 GB/s, close to the 32 GB/s that a PCIe

Stored: 23.8 (PyTorch) and 23.5 (JAX); pageable 14.3 and 8.2. Three notes. "Close to" is generous for 72%; say "about three quarters of". The pageable rates differ by 1.7 times between the tabs and the shared prose wisely gives no number; keep it that way. And 23 GB/s is a property of this host and slot; it is then reused as a constant four more times in this section (links paragraph twice, Summary, Exercise 5) and in `multi-gpu-practice.md`. On a PCIe 5.0 host or a laptop it will be another number; introduce a symbol for the host-link bandwidth once and say "23 GB/s on our machine" at each use, as the text mostly does.

**HW-14** (minor, thought). "Links between GPUs".
> A copy between two of our GPUs therefore runs at most at the rate of one

The path is two links in sequence through host memory. "At most the rate of one link" holds only if the two hops overlap; if they run one after the other the rate halves. The multi-GPU section measures the staged path; here say "at most, and only if the two hops overlap". Also in this paragraph the consequence for the reader is delayed by product history ("the RTX 3090 and 3090 Ti were the last GeForce cards with an NVLink connector"); cut the history.

**HW-15** (minor, correctness, consistency with the previous section). "Workloads on the Roofline".
> from memory for all $n$ tokens, an intensity of about $n$ FLOPs per byte.

The previous section took care over this: by `eq_matmul_intensity` the smallest dimension caps the intensity, and for a width of 2,048 "the width, not the batch, is the smallest dimension" (910 FLOP/byte at 8,192 tokens, not 8,192). For "a prompt of a thousand tokens" the exact value is about 500, because the activations are also moved. The conclusion stands (500 is above every ridge in the table), but write "up to about n, capped by the width of the layer".

**HW-16** (minor, language). "Batched Decoding".
> In a whole model the
> key–value caches add bytes that batching does not share

(Anchor for grep: "caches add bytes that batching does not share".) The paragraph ends with a list handed to another part: "The systems that do so, with their schedulers, paged caches, and speculative decoding, belong to the Language Models part." Three undefined terms in a sentence whose only job is to say "not here". End at "keep the caches small".

**HW-17** (moderate, pedagogy). "Rules of Thumb".
> With the magnitudes of :numref:`tab_rules_of_thumb`, a workload can be

This is the sentence the whole chapter needs, and it arrives as the lead-in to a table at the end of the second section: "count its FLOPs and bytes, divide by $P$ and $\beta$, and compare the result with the host cost of its operations". It is the predictive procedure that the performance-model section stops short of stating (see that review, structural finding 1). Move the procedure there, with a worked case. The table itself is useful; it would be stronger with the sweep time row (C/β ≈ 20 ms: the largest model that fits decodes at about 50 tokens per second per stream) and without the row that restates Horowitz.

**HW-18** (minor, pedagogy). Summary.
> Memory is a hierarchy in which each level farther from the arithmetic

The Summary is one paragraph of 230 words that repeats eleven numbers (0.9 TB/s, 290, 23 GB/s, 900 GB/s, 450 tokens per second, ...). A summary of a section whose thesis is that numbers change and relations stay should state the relations. Three or four sentences; the numbers are in the two tables.

## Code and experiments

- `hardware-where-bytes-live`: repeats `performance-model-three-regimes-1` (HW-4). Stored 0.92 and 0.91 TB/s; prose "about 0.9 TB/s, within about ten percent" matches.
- `hardware-the-cpu-s-role`: a real measurement with a clear reading. Prose matches the stored outputs. The JAX tab needs 100 words to explain memory kinds and why each call must block; acceptable. The cell allocates 256 MB on the device per call; fine.
- `hardware-batched-decoding`: the best cell. Prose matches both tabs ("about 0.6 ms", "Up to a batch of 64 the time barely changes", "levels off near the 165 TFLOP/s peak": stored 157 and 171, 168 and 172). The crossover "at a batch of about 164" is predicted, and the stored grid (64, 256) brackets it without resolving it; adding B = 128 would let the reader see the knee.
- No cell only confirms a theorem; no check that cannot fail. Numbers no cell prints: 500 and 450 tokens per second (prose arithmetic, fine); the 0.17 s gradient transfer (prose arithmetic).
- No TensorFlow or MXNet material. Tab fragmentation is light: two tab-specific notes after one cell.
- Missing experiments: the access pattern (Exercise 9), latency against bandwidth (Exercise 1). One of them should be in the body.

## Exercises

Ten exercises, about 1,550 words, two starred. This is the strongest exercise set I have seen in the part: nearly every one makes the reader compute a consequence the text did not state. That is also the criticism: insights the body needs are here (structural finding 2).

- **Keep as they are:** 2, 3, 5, 6, 7, 8, 10. Worked answers, for the editor: 2a gives 51.2 GB/s per DIMM, ten DIMMs, 7.8 Gb/s per wire, so the gap is the wire count. 3b gives a crossover batch of 281 on the B200 in all three formats and 117, 117, 234 on the 5090. 5b gives T ≥ (2/3)·P/β_link, about 4,800 tokens for the staged path, 1,700 for NVLink 5. 6c gives 2 J per token and 0.17 dollars per million tokens. 8a gives d* ≈ 2,000 on the 4090 and 5,700 on the B200.
- **Exercise 5c** quotes "an effective 2 GB/s, the rate that NCCL reaches on the staged path". `multi-gpu-practice.md` says "about 2.5 GB/s" per direction at k = 2 and "about 5 GB/s" for β. Use the number the later section measures, or say "about 2.5".
- **Exercise 1** is good but long in setup; 1b's question "In which regime does the curve depart from $t(n)$, and why?" depends on the on-die capacity, which the body mentions only in passing (the hierarchy paragraph and the figure). Fine as a harder item; mark it ●.
- **Exercise 4** (starved input pipeline) is the only one on the host's second job, which the body covers in two sentences. Good; consider a sentence in the body giving the rule it teaches (the slowest stage in decoded bytes per second sets the worker count).
- **Exercise 8** assumes K = 12 kernels per block per token; an eager GPT block launches several times that. The conclusion (launch-bound at d = 768) only gets stronger; say "at least".
- Ten is two or three too many for one section. If cuts are needed: 2b and 2c (geometry of two dies) and 9b.

## Still open from October

October's only notes on this section were to keep the shoreline argument (done; it is the best passage) and the chapter-wide em-dash count (this file now has none in prose outside table placeholders). Nothing else from October applies.

## For the chapter review

- **Order and seam with the previous section.** The order is right. Three items sit on the wrong side of the seam: the batch sweep (tests the roofline; belongs in the performance-model section), the bandwidth cell (duplicates it), and the predictive procedure in "Rules of Thumb" (the method; belongs there too). In return the SM and tile paragraph of the previous section belongs here.
- **Overlap with chapter 11.** Prefill, decode, the tokens-per-second bound and batched decoding are taught in `kv-cache.md` and again here. One owner is needed; I suggest chapter 11 for the phenomena and this chapter for the hardware reason.
- **Overlap inside the chapter.** The host cost of an eager operation is stated three times in this section (latency paragraph, "The Host", the rules table) and with other values in the previous one. Formats overlap with `sec_numerics` and `sec_memory_precision`. `sec_hardware_buyers` in the Tools appendix is the natural home for per-product detail.
- **What it sets up.** 23 GB/s and the PCIe topology for the two multi-GPU sections; 16 bytes per parameter for memory and precision; the format rules for mixed precision. These are used downstream as constants of "our machine"; the chapter review should check that each later use says so.
- **Suspected chapter pattern.** The chapter argues from one development machine, which makes it concrete and credible, but constants of that machine (165, 1.008, 0.9, 23, 12 µs) harden into facts as the chapter goes on. A short table "our machine, measured" near the start of the chapter, with symbols, would let every later sentence be written in terms that survive re-execution elsewhere.
- **Dependencies.** Assumes `sec_numerics` (formats), `sec_use_gpu` (pinned memory), `sec_kv-cache` (prefill and decode), `sec_perf_model`. No appendix dependence. Two forward references to the Language Models part, neither load-bearing once HW-8 and HW-16 are applied.
