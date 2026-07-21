# Ch11 (Transformers) / Ch12 (State Space Models) — Prose-vs-Captured-Output Reconciliation

Date: 2026-07-21. Scope: does every concrete number, ratio, or "we measured/the
notebook shows…" claim in the prose, summaries, and `<!-- slides -->` sections
match what the corresponding code cell actually printed, for **both** frameworks
(pytorch, jax)? Read-only review — no files edited, nothing committed. This is
the class of defect that plagued the ch13 rebuild: a prose number that
contradicts the executed cell, or a "phantom" claim with no backing cell at all.

Files reviewed (all `.md` sources, prose + slides + captured `outputs/{pytorch,jax}/...json`):
- **Ch11 Transformers:** `chapter_transformers/{index,transformer-block,gpt,kv-cache,encoders-decoders,vision-transformer,moe,scaling-laws}.md`
- **Ch12 State Space Models:** `chapter_recurrent-modern/{index,lstm,ssm,mamba,matrix-state,deltanet,test-time-regression,hybrids}.md`

Method: a Python helper (`scratchpad/recon1112/dump.py`) dumps every cell's
captured stdout text (and, for one PNG-based plot, the rendered image itself)
from `outputs/{pytorch,jax}/<chapter_dir>/<file>.json`, for both frameworks side
by side. Every sentence in every file that quotes a concrete number, ratio,
percentage, timing, or "we measured/observe/the run shows…" claim was traced to
its cell and compared against both frameworks' captured values. Hedged/
qualitative statements, forward references, and architectural/citation constants
(e.g. real GPT-2/Llama/DeepSeek specs, FLOPs formulas) were intentionally not
flagged — only claims tied to *this chapter's own executed cells*.

## Bottom line

**13 of 15 files are fully clean** — every checked numeric claim matches the
captured output, for both frameworks, within a defensible hedge. **2 genuine
defects found, both in `chapter_transformers/encoders-decoders.md`.** No
defects found anywhere in ch12 (State Space Models) — that chapter's
cross-framework claims (several of which are unusually specific, e.g. the
Mamba-vs-minGRU scoreboard flip and the hybrid recall-sweep numbers) were
verified to match the captured data with striking precision, down to the
hundredths digit in places.

| # | File | Location | Verdict | One-line issue |
|---|---|---|---|---|
| 1 | `chapter_transformers/encoders-decoders.md` | line 621 | **CONTRADICTION** | "about ten million characters" vs. ~0.9–1.8M actually processed by the printed loop |
| 2 | `chapter_transformers/encoders-decoders.md` | lines 957–958 | **CONTRADICTION** (JAX only) | "full self-attention is as fast or faster" at short inputs — false in the JAX capture (perceiver already faster at N=1024, 2048) |

Additionally, a recurring **soft/PHANTOM-adjacent pattern** appears five times
across four files: prose asserts a specific numeric outcome of *rerunning with a
different seed*, but no cell in the store executes that rerun (the project's own
policy is single seeded runs, so this is expected — but the specific numbers
quoted for the unexecuted reruns are, by definition, unverifiable from the
committed store). None of these are counted as hard contradictions because in
every case the *magnitude* claimed is plausible and, where a cross-framework
comparison exists, consistent with the actual pytorch-vs-jax spread. Listed in
§"Recurring soft-flag pattern" below for completeness.

Files confirmed **fully clean** (no contradictions, no phantoms):
`chapter_transformers/{index,transformer-block,gpt,kv-cache,vision-transformer,moe,scaling-laws}.md`,
`chapter_recurrent-modern/{index,lstm,ssm,mamba,matrix-state,deltanet,test-time-regression,hybrids}.md`.
(`ssm.md` and `scaling-laws.md` each have one soft/phantom-adjacent note, see below;
still counted "clean" for hard contradictions/phantoms.)

---

## Chapter 11 — Transformers

### `index.md` — clean
No executed cells; forward-references to later sections' numbers (e.g. "32
randomly initialized blocks") all check out against the sections they point to.

### `transformer-block.md` — clean (one soft note)

| Claim (prose) | Cell id | pytorch | jax | Verdict |
|---|---|---|---|---|
| post-LN query/key grads ~1e-9, "six orders of magnitude" below pre-LN; value/output ~0.06 | `transformer-block-signal-propagation-at-initialization` | W_q 2.9e-9 vs 1.9e-3 (ratio 6.5e5); V/O ~0.058–0.059 | W_q 5.6e-9 vs 2.3e-3 (ratio 4.1e5); V/O ~0.065–0.066 | OK — ratio is 5.6–5.8 orders, "six" is a fair rounding |
| post-LN spread "~one part in 10,000" at k=32; pre-LN "still at 0.4" | same | 1.2e-4 / 0.37 | 2.0e-4 / 0.37 | OK |
| RMSNorm timing: "nothing measurable" in pytorch, "roughly a third" in JAX | `transformer-block-rmsnorm` | LayerNorm 0.681ms, RMSNorm 0.652ms (4%) | LayerNorm 1.076ms, RMSNorm 0.732ms (32%) | OK — exact match for JAX's "a third" |
| FFN params "match to a tenth of a percent" | `transformer-block-the-feed-forward-network-2` | gelu 524288, swiglu 524544 (0.049%) | identical | OK |
| Census "$4d^2$ attention, $8d^2$ FFN, ~$12d^2$ total" | `transformer-block-shapes-and-parameters-1` | attn 262144=4·256², total/d²=12.01 | identical | OK |
| SwiGLU "ends more than a tenth of a nat ahead" | `transformer-block-the-flags-at-work-gelu-versus-swiglu` | gelu 1.31 vs swiglu 1.17 (Δ0.14) | gelu 1.41 vs swiglu 1.24 (Δ0.17) | OK, both >0.1 |

**Soft note (not counted):** "a margin that holds up across seeds (rerunning
with seeds 1 and 2 moves each number by a couple of hundredths, not the gap)" —
only seed 0 is captured in the store; no cell reruns seeds 1/2. Unverifiable but
plausible given the project's "single seeded run" policy.

### `gpt.md` — clean
Every quantitative claim checked exactly: 4.73M param model ("4.7M"), val loss
~1.49/1.47 ("around 1.5" within a few hundred steps), post-LN pinned at exactly
2.83 in **both** frameworks (matches the stated unigram entropy exactly),
124.4M-param GPT-2 ("124M"), perplexity 49 in both frameworks ("around 50"),
26.3× parameter ratio ("about 26 times"), and the FLOPs arithmetic (5×10¹⁴ /
7×10¹⁸) is internally consistent with the printed parameter/token counts.

### `kv-cache.md` — clean (two borderline-but-OK observations)
Cache-size formula (72 KiB/token → 288 MiB at n=4096) matches the allocator
exactly in **both** frameworks. GQA cache-vs-quality sweep, low-rank (rank-256/
128) perplexity, and the window+sink recovery numbers (perplexity 36→3836/3842
with 0 sinks, →47/47 with 1 sink, →43/43 with 4 sinks) all match precisely and
identically between frameworks. Two observations judged within a defensible
hedge, not flagged: (a) "about a factor of five" at n=4096 is 5.05× in pytorch
but 6.27× in jax; (b) "generates at about a hundred" tokens/s is 101 (pytorch)
but 69 (jax). Both are order-of-magnitude-correct qualitative anchors, consistent
with this book's stated policy against quoting per-run decimals.

**Soft note (not counted):** "rerunning any row with a different seed shifts it
by a few hundredths, about the spread of the whole column" (cache-vs-quality
sweep) — same unexecuted-seed pattern as above; the actual pytorch-vs-jax spread
(0.02–0.04) is consistent with the claimed magnitude.

### `encoders-decoders.md` — **2 CONTRADICTIONS FOUND**

#### Contradiction 1 — "about ten million characters"
**Location:** `chapter_transformers/encoders-decoders.md:620–621`
> "One encoder block, one decoder block, and a few hundred steps of on-line
> batches suffice — the model sees about ten million characters, none twice."

This sits directly above cell `encoders-decoders-training-and-decoding-1`, whose
full, visible loop bounds are `for step in range(600): src, dec_in, tgt =
sample_batch(128)` with `V, T = 16, 12` (strings of length 12) fixed earlier in
the file. Consuming the printed source characters only: 600 × 128 × 12 =
921,600 ≈ **0.92 million**; counting source *and* target streams together (both
length-12, distinct string instances): 1,843,200 ≈ **1.8 million**. Neither
reading reaches "about ten million" — the claim overstates by roughly **5.4× to
10.8×**, a full order of magnitude, not a rounding/hedge issue (contrast the
book's own hedge examples: "about 2× vs. captured 1.9×" is fine; this is "ten
million" vs. an actual ~1–2 million).
**Captured cells:** `encoders-decoders-training-and-decoding-1` (pytorch: `loss
at step 100/200/400/600: 0.005/0.001/0.001/0.000`; jax: `0.002/0.001/0.000/
0.000` — both confirm 600 steps ran, nothing more). No cell contradicts the
600/128/12 figures used above; the arithmetic is derived directly from the
visible, unambiguous loop bounds.

#### Contradiction 2 — "full self-attention is as fast or faster" at short inputs (JAX)
**Location:** `chapter_transformers/encoders-decoders.md:956–958`
> "The left end of the plot belongs in the reading too: at short inputs the
> bottleneck buys nothing, and full self-attention is as fast or faster."

**Captured cell:** `encoders-decoders-the-cost-curve`
| N | pytorch self-attn | pytorch perceiver | jax self-attn | jax perceiver |
|---|---|---|---|---|
| 1024 | 0.99 ms | 1.31 ms | **0.33 ms** | **0.23 ms** |
| 2048 | 2.80 ms | 1.40 ms | **0.62 ms** | **0.34 ms** |
| 4096 | 9.46 ms | 1.35 ms | 2.69 ms | 0.16 ms |
| 8192 | 34.79 ms | 1.72 ms | 10.00 ms | 0.27 ms |

For pytorch the claim holds exactly: self-attention (0.99 ms) is faster than the
perceiver (1.31 ms) at N=1024. For **jax**, the captured numbers show the
opposite at both of the two shortest lengths tested: at N=1024 self-attention
(0.33 ms) is *slower* than the perceiver (0.23 ms), and at N=2048 self-attention
(0.62 ms) is again slower than the perceiver (0.34 ms). The perceiver is faster
than full self-attention at **every** tested length in the jax capture, not just
at long inputs — directly contradicting "at short inputs... full self-attention
is as fast or faster" for that framework. (Likely cause: these are sub-millisecond
timings dominated by JAX dispatch/compile overhead rather than a real
architectural reversal, but the prose makes a specific directional claim that the
jax data does not support.)

### `vision-transformer.md` — clean
ViT val acc 0.86/0.87 ("about 86–87%"), CNN val acc 0.91/0.91 ("about 90–92%"),
both at 6.5M params exactly, position-embedding cosine similarities (neighbors
0.05/0.03, distant −0.03/−0.02, "a few hundredths positive... slightly
negative") all match precisely in both frameworks.

### `moe.md` — clean
Mixtral accounting (46.7B/12.9B "reproduces the model card exactly"), DeepSeek-V3
(656.5B/23.1B, "28-fold gap"), the ours/Mixtral ratio ("twice as sparse", 8.0x vs
4.0x), routing-collapse counts (5/16, 6/16 experts dead, "several"), and the
capstone MoE-vs-dense comparison (5.65× params at matched active, best-val
1.51/1.52 vs 1.49/1.49, "indistinguishable... near 1.5") all check out exactly
in both frameworks — including the directional claim "the training loss ends
slightly lower for the MoE, consistently across seeds and frameworks" (pytorch
0.21→0.19, jax 0.20→0.17: MoE lower in both).

**Soft note (not counted):** "rerunning, we have seen anywhere from two dead
experts to a layer served entirely by a single one" — unexecuted-seed claim,
same pattern as above.

### `scaling-laws.md` — clean (exceptionally precise matches)
$12Ld^2$ census to a fraction of a percent, embedding share (0%/32%, "nearly a
third"), profiler ratio 1.001 ("a fraction of a percent"), XLA ratio 1.10
("about ten percent above"), backward/forward = **exactly** 2.00 ("almost
exactly one to two"), achieved throughput 8.8 TFLOP/s vs. the ~83 TFLOP/s RTX
4090 spec quoted elsewhere in the same chapter ("roughly an order of magnitude
below" — 9.4×), five-size sweep (0.33M→14.2M, "factor of forty" = 42.9×), and
the recipe-table constructor calls (Mistral/Llama-3/Qwen3 all printing the
identical 4.40M/same-argument-list line, "the three modern rows are the same
argument list") all match exactly in both frameworks.

**Soft note (not counted):** "rerunning a point with a different seed moves it
by less than a hundredth of a nat" — unexecuted-seed claim; the actual
pytorch-vs-jax spread across the five sizes (0.00–0.01) is consistent with it.

---

## Chapter 12 — State Space Models

### `index.md` — clean
Every one of this file's forward-references to specific numbers in later
sections (GRU-beats-LSTM scoreboard, "kilobyte-scale state", the Mamba/JAX
scoreboard flip, the $(n-1)/d_k$ capacity law, the deltanet overwrite collapse
pattern, the "single eigenvalue" parity claim, and the "0.92 to 1.00" hybrid
recall range) were traced to their home sections and confirmed accurate — see
below. This is a well-calibrated summary page.

### `lstm.md` — clean
Scoreboard exactly as described: LSTM(scratch) 103.7/106.2 ("in the same range"
as vanilla RNN's stated 90–110), LSTM(concise) 90.7/90.9, GRU 79.2/81.8 ("posts
the best perplexity of the section... ahead of the vanilla RNN and ahead of the
LSTM" — true in both frameworks), LSTM(2 layers) 122.7/99.0 (worse than 1-layer
in both — "does not pay at all"). Bidirectional output width 256 = 2×128
("doubling to 2h") matches exactly in both frameworks.

### `ssm.md` — clean (one PHANTOM-adjacent note; two plot-only claims unverifiable from text)
minGRU vs. LSTM(scratch) via associative scan, S4D-vs-LSTM sequential-image
classification (S4D 0.808/0.827 "low-to-mid eighties", within-noise final-state
readout 0.806/0.823), the framework-flipped wall-clock story (pytorch: S4D
slower than cuDNN LSTM, 47.6s vs 4.0s/epoch; jax: S4D **faster** than sequential
LSTM, 6.1s vs 48.6s/epoch, "the parallel scan is several times faster" — 7.97×,
matches precisely) and the carried-state size (1536 bytes = "about a kibibyte" /
"kilobyte-scale", matching index.md) all check out exactly in both frameworks.

**Flagged (borderline PHANTOM):** `chapter_recurrent-modern/ssm.md:1303–1305`
> "(Checked against exact float64 stepping, each path sits about ten times
> farther from the true answer than the two sit from each other.)"

No cell anywhere in this file computes a float64 reference or reports "ten
times farther" — the only printed check is `stepped vs scanned: deviation
4.41e-04 on activations of scale 71: relative 6.21e-06` (pytorch) / `1.67e-03
... relative 2.12e-05` (jax). This parenthetical asserts a specific
quantitative comparison with no backing cell in the notebook — the closest
thing in either chapter to the "phantom measurement" pattern the task asks to
catch. Low severity (it's a plausible aside, not a fabricated headline result),
but worth a maintainer's attention since it cannot be verified from the store.

**Unverifiable from text (plot-only, no printed magnitude):** "the scan's curve
stays well below [the loop's]... by roughly an order of magnitude" and "how
wide the gap has grown... (severalfold to two orders of magnitude in our runs)"
— both describe `d2l.plot` cells with no printed numeric text; would require
rendering/measuring the SVG to check, which was out of scope for this pass.

### `mamba.md` — clean, including the critical cross-framework claim
The selective-copying baseline (S4D 0.611/0.658 "far from a solution", LSTM
0.994/0.994 "at or near perfect") is exact. **The critical scoreboard claim**
— "posted the best perplexity of the chapter in our PyTorch run (the minGRU
edged it out in the JAX run)" — was verified precisely:

| model | pytorch val ppl | jax val ppl |
|---|---|---|
| LSTM | 88.9 | 90.7 |
| minGRU | 83.5 | **85.3** |
| Mamba | **80.5** | 88.9 |

Mamba is lowest (best) in pytorch; minGRU is lowest (best) in jax, with Mamba
second — exactly the asymmetry the prose (and index.md) describes, including
"at fewer than half the parameters" (minGRU 214,851 vs. Mamba 488,739 = 44%).
The mLSTM-adjacent selective-copy retrain (Mamba 1.000/0.998, "solves the
task") and the stepped-vs-scanned logit check (relative error ~5e-7 in both
frameworks, "far more tightly than float32 rounding is obliged to deliver")
also check out.

### `matrix-state.md` — clean
The capacity-law proposition was checked **visually** (the cell emits a PNG,
not SVG, so it could be rendered directly): the measured error sits on the
dashed $(n-1)/d_k$ line "so closely the curves are hard to tell apart," and the
correlated-key (ρ=0.5) curve sits "about five-fold above" the independent
curve at small n — both confirmed by inspecting
`outputs/{pytorch,jax}/chapter_recurrent-modern/matrix-state/matrix-state-what-the-memory-costs-1.png`
(byte-identical between frameworks, since the cell is pure NumPy). The duality
check ("reduces *exactly*... at $a_t=1$") prints literally `0.00e+00` in both
frameworks. Chunked-vs-recurrence timing/memory tables, the Mamba-2 2.7B
accounting (655,360 + 15,360 elements, "1.28 MiB"), and the mLSTM stabilizer
demo (overflow at step 49/47, "a few dozen steps"; stabilized-vs-float64
relative error 2.02e-05/1.38e-06) all check out in both frameworks.

### `deltanet.md` — clean, unusually precise cross-framework fact-checking
The overwrite collapse (Hebbian 1.000→0.510→0.299→0.217, identical in both
frameworks since the cell is pure NumPy; "roughly halves by two writes," "down
to about 0.2 by eight") and the trained-model version (Hebbian 0.537/0.531 at
R=2, 0.268/0.265 at R=6; delta 1.000 throughout in both) match precisely. The
Gated DeltaNet scoreboard entry (83.7 pytorch / 74.5 jax, "mid-seventies to
mid-eighties depending on framework and seed" — each framework lands at one end
of the stated range) is exact. Most notably, the parity/state-tracking
experiment's cross-framework asymmetry claim — **"failing seeds appear from
T=16 in one framework and by T=24 in both"** — was verified precisely: jax's
first failing seed appears at T=16 (0.487, one of three seeds, vs. 1.000/1.000
for the others), while pytorch's first failure appears only at T=24 (0.517),
exactly matching the described pattern. The eigenvalue/reflection experiment
(1−β at β=0.5/1.0/1.5/2.0 → +0.5/−0.0/−0.5/−1.0; parity accuracy 1.000 at
β=2 for T=8/24/64 in both frameworks) is exact.

### `test-time-regression.md` — clean (two borderline "several(fold)" phrasings)
Learned-bandwidth experiment (w climbs from 1→1.72/2.13, learned σ=0.583/0.469
vs. hand-swept best of 0.5, "same region" — matches closely in both). Longhorn
closed-form check (~1e-6/1e-7 deviations, "floating-point tolerance"). Titans
linear-memory overwrite (1.000, 0.927, 0.185, 0.100 — identical in both
frameworks, pure NumPy) and deep-memory retrieval MSE drop (13.6×/24× "more
than an order of magnitude", both frameworks). Drift-tracking tail errors
(0.99/0.29/0.30, identical both frameworks, "several times the others'" = 3.3–
3.4×) all match precisely. Two observations judged within a defensible hedge,
not flagged: the spectrum sweep's "test error is several times the batch
solve's" is 7.0× in pytorch but 15.1× in jax, and "distance... shrinks
severalfold" is 3.6× in pytorch but 12× in jax — both stretch "several(fold)"
toward "an order of magnitude" for the jax capture specifically, but the
qualitative story (monotonic improvement, batch solve wins) is fully intact in
both, including the very specific hedge "in some of our runs slightly below
it" for the 30-pass-vs-batch test MSE, which is exactly true in jax (0.008 <
0.010) and exactly not true in pytorch (0.022 > 0.010) — a precise, correctly
hedged claim.

### `hybrids.md` — clean, including the chapter's most load-bearing claim
Parameter-matching ("a spread of 560, just under one percent": 58,544/57,984/
58,404 in both frameworks, spread exactly 560 = 0.96%) is exact. **The
centerpiece recall-sweep claim** — "in our PyTorch run it too scores 1.000 at
every load; in JAX it dips to roughly 0.92–0.94 mid-sweep and returns to
0.99–1.00 at the two largest loads" (echoed in index.md as "roughly 0.92 to
1.00 across the sweep") — was verified to match the captured sweep exactly:

| pairs | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| pytorch hybrid | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| jax hybrid | 0.980 | 0.936 | **0.923** | 0.998 | 1.000 |

jax's mid-sweep minimum (0.923–0.936) falls precisely within the stated
"0.92–0.94," and its recovery (0.998, 1.000) falls precisely within "0.99–
1.00." The pure-attention row is exactly 1.000 at every load in both
frameworks; the pure-linear row collapses to 0.301 (pytorch) / 0.376 (jax) at
64 pairs, "fewer than half." The language-modeling dissociation panel ("all
three land within roughly a tenth of a nat," "the pure-linear stack matches or
beats the pure-attention stack") is also exact: linear 1.75/1.76 < attention
1.83/1.84 in both frameworks.

---

## Recurring soft-flag pattern (not counted as contradictions)

Five places across four ch12/ch11 files make a specific numeric claim about
*rerunning with a different seed* — a robustness check whose result is not
backed by any cell in the committed store (consistent with this project's
stated policy of single seeded runs, not multi-seed sweeps):

1. `transformer-block.md` — "seeds 1 and 2... a couple of hundredths, not the gap"
2. `kv-cache.md` — "rerunning any row with a different seed shifts it by a few hundredths"
3. `moe.md` — "rerunning, we have seen anywhere from two dead experts to..."
4. `scaling-laws.md` — "rerunning a point with a different seed moves it by less than a hundredth of a nat"
5. `ssm.md`'s float64-comparison aside (listed above under its own heading; the odd one out, since it isn't a seed claim but an unexecuted reference-precision check)

None are flagged as hard contradictions: in each case the claimed magnitude is
plausible and, where a same-experiment cross-framework comparison exists (a
proxy for run-to-run variability), it is quantitatively consistent with what
is claimed. They are listed here because they are, by the letter of the task's
definition, unverifiable "no cell produces this" claims, and a future capture
refresh could not confirm or refute them without adding a multi-seed cell.

## What was deliberately not flagged

Per the task's instructions, the following were checked and judged to be
within a defensible hedge, not contradictions: several "about Nx" claims where
the two frameworks' captured ratios differ by up to ~25–50% but stay within
the same order of magnitude (kv-cache.md's "factor of five" at 6.27×;
test-time-regression.md's "several(fold)" at up to 15×; matrix-state.md's
"crosses within a few thousand tokens" against an actual crossover of ~328,
which is still *within* the stated upper bound); one alignment-hit-rate claim
("well over nine in ten") that is literally true but not "well over" for the
jax capture (92% vs. the 90% threshold); and one loss-attribution nuance in
kv-cache.md's MLA section where a −0.09 and +0.08 deviation from the same
baseline are narrated as "noise" and "measurably worsens" respectively (an
editorial-emphasis quibble, not a wrong number). None of these reverse a
claimed direction or misstate an order of magnitude, which is the bar the task
sets for a genuine defect.
