# Fixlog — reconciliation pass (prose vs. blessed outputs, 2026-07-20)

Post-recapture reconciliation of ch. 13 prose/summaries/exercises/slides
against `outputs/{pytorch,jax}/chapter_computational-performance/*.json`
(the committed store is ground truth). Scope: prose-only edits; no code
cell, `.qmd`, or figure touched. Inputs: review findings §0/§1 (incl. the
SUPERSEDED 13.7 block), the RECONCILE AFTER CAPTURE lists in
`fixlog-agentA.md` / `fixlog-agentB.md` / `fixlog-agentC.md`.

69 output-bearing cell variants checked (per cell-id per framework).
Verdicts below: **OK** = prose already agrees within the precision
policy; **ADJ** = prose adjusted this pass; **BLOCKED** = none.

## 13.1 performance-model (pt + jax)

| cell | captured | verdict |
|---|---|---|
| `counting-flops-bytes-and-arithmetic-intensity` | ridge 164 (both) | OK — "about 165" prose/caption/slide/summary |
| `measuring-without-lying-1` | pt 0.68/9.17 ms; jax 0.74/8.84 ms | OK — "close to nothing"; **guard passed**: jax honest 8.84 ≈ 10× Benchmark 0.85 |
| `measuring-without-lying-2` | pt 0.060→0.039 s (−35%); jax 0.983→0.361 s (−63%) | OK — "roughly a third" / "around two-thirds"; inversion gone in both tabs |
| `measuring-without-lying-4` | pt 0.91, jax 0.85 ms/call | OK |
| `the-sweep-mapping-our-gpu` | pt 2.2/18.2/118.3/157.6/153.1/165.8; jax 0.4/4.7/21.8/119.9/158.7/173.1 | ADJ — knee "≈2048" → "≈2048–4096, depending on the framework" (prose ×2 + sweep slide): jax reaches the roof only at 4096. n=256 "a percent or less" (pt 1.3%, jax 0.24%) and n=512 "a tenth of peak or less" (pt 11%, jax 2.8%) kept per review PM-3 calibration |
| (section opener) | gap pt 75×, jax 433× | ADJ — "up to two orders of magnitude" → "roughly two orders of magnitude" (433× exceeds "up to two"); slide title unchanged (magnitude-class) |
| `three-regimes-1` | pt all 0.14; jax 0.29/0.30/0.17/0.16 | ADJ — "the four times are nearly identical" is false in the jax tab (add ~2× sin, inverted); now "yet it runs no slower" + parenthetical attributing jax-tab scatter to un-jitted per-dispatch overhead. Slide "sin costs no more than add" holds in both tabs — unchanged |
| `three-regimes-2` | pt 0.14→1.25 (~9×); jax 0.30→1.70 (~5.7×) | OK — "factor-of-several overpayment"; per-op round-trip claim holds (jax chain ≈0.19 ms/op ≈ its unary kernels) |
| `the-profiler` | pt table warning-free; jax trace-location line | OK — acc_events fix held; no digits quoted |

## 13.2 hardware (not re-executed; prose quotes the store)

| cell | captured | verdict |
|---|---|---|
| `where-bytes-live` | pt 0.80, jax 0.91 TB/s | OK — "within tens of percent of spec" (prose + slide) |
| `the-cpu-s-role` | pt pageable 13.4 / pinned 24.3 GB/s; jax device_put 0.9 GB/s | OK for the tab prose ("roughly doubles", "near PCIe's ceiling"; jax "around one gigabyte per second"). ADJ — pinned-vs-HBM "two orders of magnitude below the GPU's own memory" → "a factor of about forty" (24.3 vs ~1000 GB/s is 1.6 orders) |
| interconnects prose (quotes 13.5) | 13.5 prints 17.28 GB/s raw, busbw ~2 (audit) | ADJ — "a few tens of GB/s" → "roughly twenty GB/s" (matches 13.5's tab wording; captured 17.3, historical 18.6–20.7); summary "one to two orders below NVLink" → "two to three orders" (raw 104×, busbw ~800×); slide "One to two orders" → "Two orders of magnitude and more" |
| rules-of-thumb table | — | OK — "PCIe per direction: tens of GB/s" (24.3), "GPU memory bandwidth 1–8 TB/s" |

## 13.3 compilation (pt + jax)

| cell | captured | verdict |
|---|---|---|
| `capture-two-philosophies-1/-2` | pt "graph breaks: 1"; jax jaxpr + trace-time print | OK — slide scoping `@…-1@pytorch`/`@…-2@jax` in place |
| `what-the-compiler-does-fusion` | pt 1.25→0.15 (8.3×); jax 1.92→0.18 (10.7×) | OK — "close to an order of magnitude" (prose/summary/slide); asserts passed silently. Cross-file: discharges 13.1's "leave it bleeding" promise |
| `whole-step-compilation-measured` | pt first 0.3 s, 1.84→1.28 ms; jax AOT 2.2 s, 19.29→0.18 ms (107×) | OK — "a fraction of a second to a couple of seconds" covers 0.3 s and 2.2 s; "about two seconds for the real Transformer" matches 13.7's 1.9 s print. ADJ — jax tab "nearly two orders of magnitude slower" → "about two orders" (107×) |
| `the-overhead-regime-capture-and-replay` | pt 2.33→0.31 ms; jax 9.42→0.43 ms | OK — **agent A's guard passed**: reduce-overhead WINS ~7.5× (no cudagraph re-record pathology); "hundred-odd launches" (120 modules) |

## 13.4 memory-precision (pt + jax)

| cell | captured | verdict |
|---|---|---|
| `measuring-memory-1` | 403 predicted / 524 allocated / 547 reserved MB | OK — "overshoots by roughly 30%" (1.30 exact); reserved > allocated as narrated |
| `measuring-memory-2` | sawtooth plot | OK |
| `measuring-memory-3` | jax temp 15 / arg+out 203 MB | OK — no digits quoted |
| `mixed-precision` | pt 22.58/17.79/9.78 (tf32→bf16 1.82×); jax 26.41/15.99/8.70 (1.84×) | OK for "one and a half to two times" (prose/summary/slides). ADJ — "tf32 buys a good fraction, and bf16 then adds roughly that much again" → "…adds at least as much again": pt's fresh fp32→tf32 step is only 1.27× vs bf16's 1.82× (step parity no longer holds; the inequality does in both tabs and in all captures to date) |
| `activation-checkpointing` | pt 4318→2306 MB (−46.6%), 59.9→71.1 ms (+19%); jax 3230→1216 MB (−62%), 50.5→58.7 ms (+16%) | OK for "~2 GB released" (2012/2014 MB) and "15–20% longer". ADJ — "falls by half or more, in the allocator's count and the compiler's plan alike" → "by nearly half in the allocator's count and by more than half in the compiler's plan" (pt is 46.6%, strictly under half); summary + slide → "cut roughly in half or better" |
| `gradient-accumulation` | pt 1.19e-07; jax 3.10e-06 | OK — "match to floating-point noise", no digits |

## 13.5 multiple-gpus (pt + jax)

| cell | captured | verdict |
|---|---|---|
| `data-parallelism-by-hand-3` | toy allreduce before/after | OK |
| `data-parallelism-by-hand-7` (invariant) | pt 9.31e-10; jax 1.86e-09 | OK — "around 10⁻⁹ against magnitudes near 10⁻²". FLAG (not blocked): the jax variant captured an XLA `cuda_timer` "Delay kernel timed out" stderr line — cosmetic, contention-dependent (agent B predicted it could land in exactly this cell); clears only on a luckier recapture |
| `data-parallelism-by-hand-5/-6` (train) | pt 2.37→2.31 s/epoch (acc .70→.79); jax 0.56→1.00 s/epoch (acc .79→.81) | ADJ — the pt capture shows 2-GPU **parity, not a slowdown**, and a 9-point accuracy scatter from its unseeded init. Applied agent B's pre-authorized fallback: "wall-clock is **worse**" → "**no better**"; "outright slower in both frameworks" → "at best on par with one GPU, and in JAX outright slower"; "accuracy is essentially unchanged" → identical-optimization + run-to-run-noise phrasing; "watch it lose" → "watch the second GPU buy us nothing". Slide "Two GPUs, No Speedup" still literally true — unchanged |
| `the-accounting` | pt star copy 17.28 GB/s / 31.1 ms; jax psum 6.53 GB/s / 82.3 ms | pt tab OK — "roughly twenty GB/s" (17.3 within the phrase's tolerance; captures range 17.3–20.7). ADJ — jax tab "only a few GB/s" → "well under ten GB/s" (prints 6.53; captures range 4.5–6.5); NVLink-gap wording corrected: pt tab "one to two orders" → "two orders" (104×), jax tab → "more than two orders" (276×), summary → "two to three orders" |
| accounting follow-up prose | 2N/β for LeNet ≈ 60 µs | OK — "a fraction of a millisecond" |

## 13.6 multi-gpu-practice (pt + jax) — NO EDITS NEEDED

| cell | captured | verdict |
|---|---|---|
| `ddp-really-run-3` (weak) | 2105 / 3714 (1.76×, 88%) / 6931 (3.29×, 82%) | OK — "about 1.8×/3.3×", "88% and 82%" exact in prose, summary, slide, exercise 5 |
| `ddp-really-run-4` (strong, k=4 first-ever run) | 2094 / 3715 (1.77×, 89%) / 5876 (2.81×, 70%) | OK — prose written to survive any outcome reads exactly right: "nearly coincide at k=2" (88 vs 89), "part company as k grows" (82 vs 70); 70% inside agent C's expected 70–80% |
| `ddp-really-run-5` (comm probe) | predicted 20 ms, measured 15 (synced 132, no_sync 117) | OK — "within tens of percent" (25%) |
| `…collectives-2` | loss 2.47; batch sharded 4-way, weight replicated | OK — prose quotes no loss value; 4-GPU visualization fine |
| `…collectives-3` (receipt) | 36 all-reduce ops; largest 6.4M floats | OK — "dozens", "exact count is a compiler artifact", "two largest carry the ~11M gradient floats" |
| `…collectives-4` (jax sweep) | 2670 / 4384 (1.64×, 82%) / 8001 (3.00×, 75%) | OK — "82% already at two GPUs"; k=4 < k=2; "a few points lower" than DDP (88→82, 82→75); "sagging, no cliff" |
| `…collectives-5` (invariant) | 1.5e-04 | OK — "about 10⁻⁴ against magnitudes near 10⁻²" |
| `sharding-…-fsdp-idea` | sketch prints | OK |

## 13.7 fast-transformer (pt only)

| cell | captured | verdict |
|---|---|---|
| `the-subject-1` | 18.9M parameters | OK — "~19M", "roughly 76 MB of gradients" |
| `rung-0-baseline-profiled` | warning-free table; mm 53% of CUDA; 660 muls/5 steps; CPU 136 vs CUDA 128 ms; R0 262,957 tok/s | OK — "about half", "hundreds of `aten::mul`", "CPU total nearly as large" |
| `rungs-…-1` (R1) | first step 1.9 s; 345,595 (1.31×) | OK — "about two seconds", "roughly 1.3×" (prose/summary/slide); 13.3's cross-ref holds |
| `rungs-…-5` (re-profile) | tail in CompiledFunction; mm share 59% (was 53%), avg 150 µs | OK — "tail folded, matmuls' share has risen" |
| `rungs-…-2` (R2) | 468,280 (1.35×) | OK — "about 1.4×" kept: printed 1.35 rounds to 1.4 and the steady state ranges 1.35–1.43 across runs (agent C's own band); no change |
| `rungs-…-3` (R3) | 622,604 (1.33×), peak 8.6 GiB, fp32 control 16.6 GiB | OK — "roughly another 1.3×", control "fits — snugly", bf16 "roughly half the footprint" |
| `rungs-…-6` (re-profile) | mm avg 509 µs (3.4× R1's) | OK — "several times longer per call" |
| `rungs-…-4` (R4) | 545,503 (0.88×), peak 3.1 GiB | OK — "cost about a tenth" (12%), "about 9 GiB down to 3", "about a third of the card"; slide "−~10%, ~3×" |
| `rungs-…-8` (R5, k=4 first-ever run) | t_comm ~34 vs t_cmp ~31 ms; k=2 floor 253k, measured 297k (1.13×); **k=4 floor 506k, measured 455k (1.73×)** | ADJ — the k=4 measurement lands **below** the no-overlap floor, contradicting "lands between the floor and the linear ceiling". Rewrote the prediction ("under half of linear at k=4"; "the measurement *can* land above the serial prediction"), the post-measurement paragraph (above the floor at k=2 via overlap; a little below at k=4 because the floor's β was calibrated at two GPUs and four ranks staging through one host path do not quite sustain it), the summary bullet, and the "Predict, Then Measure" slide. "four GPUs into roughly half of linear" → "well under half" (43%). Also reconciled the β sentence with 13.5's fresh 6.53 print: "sustains around five GB/s per device … the cell prices with 4.5, toward the conservative end of its run-to-run range" |
| `the-waterfall` | cumulative R0→R3 = 2.37× | OK — "about 2.4× in our runs — the cell prints the exact figure" (prose/summary/slide); inside agent C's ≤2.6 hedge |
| `the-waterfall-1` (learning run) | smoothed loss 3.01 → 0.07 | OK — assert passed; prose quotes no numbers, "falls very far" |

## index.md

- OK — the compile-vs-checkpointing framing already matches the measured
  story (compile a real ~1.3× win; checkpointing the negative rung when
  its constraint does not bind) — agent C's flagged IX-1 contradiction was
  already resolved before capture.
- ADJ — "one to two orders of magnitude below a datacenter NVLink fabric"
  → "roughly two orders" (raw staged copy is ~100× below 1.8 TB/s).

## BLOCKED

None. Every captured output could be covered by honest prose.

## Flags for the central pass

1. **Captured stderr noise** in jax `multiple-gpus-data-parallelism-by-hand-7`
   (XLA "Delay kernel timed out") — cosmetic, environmental; not fixable in
   prose. A quieter recapture may clear it.
2. **13.7 R5 k=4 vs the floor**: prose now says the k=4 measurement "slips
   a little below" the flat-β floor (455k vs 506k). If a future recapture
   puts k=4 back above the floor, re-touch that clause (the "lands at the
   floor, nowhere near the ceiling" frame survives either way).
3. **β = 4.5e9 code comments** in 13.6/13.7 cite 13.5, whose psum print now
   reads 6.53 GB/s (range 4.5–6.5 across captures). Prose reconciled
   ("conservative end of its run-to-run range"); the constant itself is in
   code and untouched. If 13.5 recaptures high again, consider re-basing β
   at the next code-touching pass.
4. **13.7 R2** printed 1.35×, at the very bottom of the prose's "about
   1.4×" — acceptable now; if a recapture prints ≤1.32×, drop the three
   "~1.4×" mentions to "~1.3–1.4×".

## Verification

`tools/lint_source.py` clean on all eight files. No `.qmd`, code cell,
`#@save` body, cell id, label, or figure touched — verified by reviewing
every edit hunk (all are prose paragraphs, summary bullets, or slide
text). No commits made.
