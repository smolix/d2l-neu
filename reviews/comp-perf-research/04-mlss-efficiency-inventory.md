# Inventory: smolix/mlss-efficiency — "Efficiency in LLMs" (MLSS 2026)

Source for rebuilding d2l's "Computational Performance" chapter (esp. Hardware).
Clone location: `/tmp/claude-4002/-home-smola-d2l-neu/0b526508-b3d5-4a42-8bee-4374612287b8/scratchpad/comp-perf-research/mlss-efficiency`
(read-only clone via `gh repo clone smolix/mlss-efficiency`; nothing pushed, `/home/smola/d2l-neu` untouched.)

---

## 1. Repo shape

A single LaTeX/beamer slide deck, not a paper or notebook repo.

```
mlss-efficiency/
├── main.tex                  # preamble, theme (metropolis + Boson AI skin), \input's 6 sections
├── deckstyle.sty             # factored style package (colors, tcolorbox styles, macros)
├── boson.png                 # logo
├── sections/
│   ├── 01-overview.tex       # 3,694 words / 871 lines — cost model, roofline, arithmetic intensity
│   ├── 02-hardware.tex       # 5,038 words / 1,085 lines — THE HARDWARE CHAPTER
│   ├── 03-serving.tex        # 3,760 words / 837 lines  — batching, paging, caching, spec decode
│   ├── 04-weights.tex        # 4,861 words / 1,184 lines — MoE, quantization (GPTQ/AWQ/…)
│   ├── 05-kv.tex             # 5,063 words / 1,220 lines — KV-cache compression (MLA, sparse, evict)
│   └── 06-resources.tex      # 1,539 words / 382 lines  — recap, numbers table, papers, code repos
├── figures/                  # 60 MB, 52 raster files (PNG/JPEG) — vendor photos, paper screenshots.
│                              # NO figure-generator scripts (no .py anywhere in the repo).
│                              # The actual data charts (bandwidth/latency ladders, roofline, format
│                              # bit-layouts, energy bars) are inline TikZ/pgfplots in the .tex, NOT
│                              # files — they compile at LaTeX-build time.
├── ref/
│   ├── outline.md            # 338-line slide-by-slide master outline
│   ├── style-contract.md     # binding style rules + the canonical verified-numbers cheat sheet
│   ├── gaps.md                # self-audited coverage-gap analysis vs. 9 reference sources
│   └── research/
│       ├── r1-gpu-specs.md          # 279 lines — die/process/power/memory/compute tables, sourced
│       ├── r2-bandwidth-latency.md  # 240 lines — CSV blocks: DRAM/LPDDR/GDDR/HBM/NVLink/PCIe/NIC/NVMe over time
│       ├── r3-edge-hardware.md      # 208 lines — Jetson/DGX Spark/Strix Halo/Apple Silicon/phones
│       ├── r4-formats-silicon.md    # 235 lines — IEEE-754 bit layouts, reticle limit, energy/pJ, SRAM scaling
│       ├── r5-serving.md            # 184 lines — vLLM/SGLang/pricing
│       ├── r6-weight-compression.md # 297 lines
│       ├── r7-kv-compression.md     # 365 lines
│       ├── r8-resources.md          # 132 lines — repo star counts etc.
│       ├── r9-tencent-lda.md        # 153 lines
│       ├── r10-syllabi.md           # 193 lines — gap analysis inputs
│       └── r11-additions.md         # 355 lines — later-added facts (non-NVIDIA vendors, FA blog, etc.)
└── verification/              # per-section fact-check reports (claim tables with ✅/🟡/🔴 status)
    ├── 01-overview.md … 06-resources.md, README.md   — 1,592 lines total
```

Build: `lualatex main.tex` (beamer, `aspectratio=169`). No code, no notebooks, no executable
examples anywhere — this is 100% slide prose/diagrams/tables, positioned as source material to
mine, not to import wholesale.

### Recency / hardware generation

- **Git history:** 5 commits, all June 2026 (`2026-06-13` initial commit → `2026-06-27` last
  "Improve MLSS efficiency tutorial slides"). This is freshly authored material, not a
  years-old deck.
- **Title slide, verbatim:** *"All numbers verified June 2026. They're likely wrong by
  December."* — Alex is explicit about the shelf life.
- **Hardware generation described:** current-gen **NVIDIA Blackwell (B200/B200 SXM, GB200
  NVL72)** as the primary subject, with **H100/H200 (Hopper)** as the still-relevant previous
  generation and **RTX 5090 (Blackwell, consumer)/RTX 4090 (Ada)** for the consumer tier.
  Also covers **AMD MI300X/MI325X/MI355X, Google TPU v6e/v7, AWS Trainium2, Intel Gaudi 3**,
  edge silicon (Jetson Orin Nano Super, DGX Spark/GB10, Strix Halo, Apple M4 Max/M3 Ultra,
  iPhone 17 Pro/A19 Pro), and even forward-looks at **Rubin (R200, projected)**: 35 PF dense
  NVFP4, 288 GB HBM4 @ 22 TB/s, 336B transistors, TSMC 3nm (explicitly flagged "announced, not
  shipping").
- Every fact-bearing frame carries an in-slide `\srcnote{}` citation (arXiv ID, vendor page, or
  dossier section) — the material is unusually well-provenanced for a talk deck. A parallel
  `verification/` fact-check pass (dated 2026-06-23) checked 87 hardware-chapter claims: 68 ✅,
  10 🟡 (rounding/approximation), 5 🔴 (real errors — e.g. a stale "52× PCIe vs HBM" ratio, a
  "1.4× vs 2.4×" capacity-growth mislabel). **Cross-checked against the current `sections/02-hardware.tex`: the flagged 🔴 errors are already fixed in the committed file** (capacity now reads "2.4×", PCIe ratio now reads "26×") — i.e., corrections landed after the June-23 audit, before the June-27 final commit. This is a genuinely fact-checked, low-error source.

---

## 2. Hardware chapter (`sections/02-hardware.tex`) — deep summary

**~30 frames, 1,085 lines, dense.** Framing epigraph: *"Where can the bytes live, how fast can
they move, and how much math can we do while they are moving?"* Running rule: *"hardware is
useful when it keeps bytes local, moves them fast, or does more math per byte."*

Note: the canonical **roofline / arithmetic-intensity treatment actually lives in §1
(`01-overview.tex`)**, not §2 — see §2b below; §2 supplies the raw specs the roofline model
consumes. Together they're one logical unit for a d2l "Hardware" chapter.

### 2.1 — Memory & Data Movement (hierarchy, bandwidth, latency)

**2026 Bandwidth Ladder** (log bar chart, GB/s): NVMe read 14 · NIC 800G 100 · PCIe6×16 128 ·
DDR5 socket 500 · NVLink5 1,800 · **HBM3e (B200) 8,000**.
> *"On-package HBM ~60× a PCIe link, ~80× a 800G NIC. Every chip boundary costs an order of
> magnitude. Keep the bytes home."*

**2026 Latency Ladder** (log bar chart, ns): L1/SMEM 1 ns · GPU L2 75 ns · DDR5 90 ns · HBM
access 300 ns · NVLink hop 1.5 µs · kernel launch 8 µs · RDMA RTT 5 µs · NVMe read 50 µs · WAN
RTT 60 ms.
> *"Caches cut latency; decode is bound by bandwidth. Weights read sequentially ⇒ 300 ns HBM
> latency amortizes. What bites: the ~16 GB streamed every token."*

**Bandwidth over time** (log-y line chart, 2009–2030, one line per memory family — HBM, GDDR
card, DDR/channel, LPDDR aggregate, NVLink, PCIe×16, Eth/IB, NVMe): every family doubles roughly
per generation; **HBM and LPDDR broke away** from the pack (HBM by 3D-stacking DRAM dies on the
interposer; LPDDR by widening the package/LPCAMM).

### 2.2 — The Chip (area = compute, perimeter = I/O)

**The shoreline/beachfront problem** — the single most quotable *architectural argument* in the
deck: SMs fill the die interior (compute ∝ area ∝ n²); PHYs/SerDes sit only on the die edge
(I/O ∝ perimeter ∝ n). Doubling the die side (or improving the process) buys **4× compute but
only 2× I/O**. *"Bytes-per-FLOP keeps falling. This is why the model starves."* HBM sits
micrometers from the die on a shared interposer (8 TB/s, no 20 cm PCB trace); B200 is **two
reticle-limit dies bridged** (die area capped by lithography reticle at ~858 mm²) giving one
logical 192 GB, ~8–10 TB/s GPU.

**Spec Table 1 — Die** (H100 SXM / B200 SXM / RTX 4090 / RTX 5090): process node (TSMC 4N /
4NP / 4N / 4NP), die area (814 / 2×~800 / 608 / 744 mm²), transistors (80B / 208B / 76.3B /
92.2B), TDP (700W / 1000–1200W / 450W / 575W).

**Spec Table 2 — Memory**: type (HBM3/HBM3e/GDDR6X/GDDR7), capacity (**80 / 192 / 24 / 32 GB**),
bus width (5120/8192/384/512-bit), bandwidth (**3.35 / 8.0 / 1.008 / 1.792 TB/s**). Callout:
*"Memory Capacity is the quiet crisis"* — bandwidth grew 2.4× H100→B200 and so did capacity, but
**capacity stalled at 80 GB across A100→H100** and caps at 8 HBM stacks per package; the
2025–26 DRAM squeeze (DDR5 +307% Q4'25) made every GB pricier.

**On-Chip SRAM table**: registers/SM 256 KB, L1/shared/SM 256 KB, TMEM/SM 256 KB (Blackwell
only), L2 total 50 MB (H100) / 126 MB (B200), SMs 132/148, **all on-chip SRAM ~115 MB (H100) /
~240 MB (B200)** vs. HBM weights capacity 80/192 GB — *"a 16 GB model is ~100× bigger than
on-chip SRAM; tiles, not weights."* SRAM stopped shrinking with process node (N5→N3E: bitcell
~0% smaller, logic 1.7× denser) — Blackwell's fix is TMEM + bigger L2, not bigger L1.

**One Node, One Rack**: HGX B200 node = 8×B200 + 4 NVSwitch + 2 CPU + 8 NIC, ~1.5 TB HBM3e,
1.8 TB/s/GPU NVLink. **GB200 NVL72 rack** = 72 GPU + 36 Grace CPUs in **one NVLink domain**,
13.5 TB HBM, **130 TB/s aggregate NVLink**, ~120 kW. *"72 GPUs, one 130 TB/s NVLink domain ⇒ MoE
all-to-all stays inside the rack, off the ~100 GB/s network. Large MoE was drawn to fit this
box."*

**"Useful Numbers for LLM Engineers" table** (the single best-condensed actionable table in the
chapter):

| Quantity | Value | Why it matters |
|---|---|---|
| H100 HBM bandwidth | 3.3 TB/s | decode floor: bytes / this |
| B200 HBM bandwidth | 8 TB/s | 2.4× H100 |
| H100 BF16 dense | 1 PF | prefill ceiling |
| B200 BF16 dense | 2.25 PF | 2.25× H100 |
| NVLink5 per GPU | 1.8 TB/s | GPU↔GPU, intra-node |
| PCIe6 x16 (1 dir) | 128 GB/s | 26× slower than HBM |
| 800G NIC | 100 GB/s | inter-node ceiling |
| NVMe seq read | 14 GB/s | swap / KV offload tier |
| Qwen3-8B KV/token | 147 KB | × ctx × batch |
| kernel launch overhead | 5–15 µs | batch small ops |

### 2.3 — FlashAttention (GEMM tiling as the canonical memory-hierarchy example)

Naive attention materializes the full N×N score matrix S to HBM — Θ(Nd + N²) reads/writes.
Worked numeric example: **N=40k, d=128 ⇒ S = 40k×40k×2B = 3.2 GB per head, per layer**, dragged
through HBM every forward pass. FlashAttention streams Q/K/V tiles through SRAM, keeps a
running max `m` and sum `ℓ` (online softmax, numerically stable), **never materializes S**.
Online-softmax update shown explicitly:
```
m' = max(m, x);  ℓ' = e^(m-m')·ℓ + e^(x-m');  O' = e^(m-m')·O + e^(x-m')·V,  out = O/ℓ
```
Utilization arc across generations: **FA1 ~25–40% of A100 peak → FA2 ~50–73% (~225 TF/s
training) → FA3 (Hopper, async TMA/WGMMA, ping-pong scheduling): FP16 740 TF (~75%), BF16 ~840 TF
(~85%), FP8 ~1.2 PF.** PyTorch SDPA auto-selects FA2/FA3; FlashInfer is the paged-KV,
serving-shaped cousin used by vLLM/SGLang. (Fact-check note: "memory linear in N" in the deck
text is imprecise — HBM *traffic* stays Θ(N²d²/M), only the extra O(N²) HBM *buffer* is
eliminated; ~9× fewer HBM accesses is the more defensible claim, per Theorem 2 of the FA1 paper.)

### 2.4 — Tensor Cores (matmul-shaped silicon)

A scalar FPU spends most of its energy on instruction overhead, not the multiply; a tensor core
hard-wires a systolic MAC array (same idea as TPU/Trainium). **Dense BF16 lineage: V100 125 TF →
A100 312 TF → H100 989 TF → B200 2,250 TF.** 2:4 structured sparsity doubles peak again but
requires training weights 2-of-4 zero. Per-MAC energy (amortized HMMA) ~2–3 pJ (Dally 2023).
Two "aside" frames on emulating FP64 via FP8/tensor-core Ozaki-scheme tricks (Matsuoka 2026,
arXiv 2606.06510) — niche but shows how far the format story extends.

### 2.5 — Floating Point Formats

**Format ladder** (illustrative, self-caveated as approximate): FP64 1× (8 B/elem) → FP32 4× (4
B) → FP16/BF16 16× (2 B, real H100/B200 dense 989/2,250 TF) → FP8 64× (1 B, 1,979/4,500 TF) →
FP4 256× (0.5 B, —/9,000 TF). Key argument: *"each precision halving buys ~2× on real silicon,
not 4×; big jumps come from new architectures, not formats; every halving also halves bytes
moved — low precision wins twice."*

**Spec Table 3 — compute by format** (dense, FP32-accumulate convention, the canonical table):

| Format | H100 SXM | B200 SXM | RTX 4090 | RTX 5090 |
|---|---|---|---|---|
| FP64 (TC) | 67 TF | ~40 TF | 1.3 TF† | 1.6 TF† |
| TF32 | 495 TF | 1,100 TF | 83 TF | 105 TF |
| BF16/FP16 | **989 TF** | **2,250 TF** | 165 TF | 210 TF |
| FP8 | 1,979 TF | 4,500 TF | 330 TF | 419 TF |
| FP4 | — | **9,000 TF** | — | **1,676 TF** |

† GeForce FP64 is vector, no FP64 tensor cores, 1:64 of FP32. Callout: *"Marketing doubles them
(2:4 sparsity): the 5090's '3352 AI TOPS' is FP4 sparse; dense = 1676."* Accumulator convention:
FP16/BF16 and FP8 both accumulate in FP32; FP4's 1-bit mantissa makes FP16 accumulation
sufficient.

**Bit-layout diagrams** (to-scale TikZ, sign/exponent/mantissa boxes): FP64 (1/11/52), FP32
(1/8/23), TF32 (1/8/10, stored in 32 bits but only 19 significant), BF16 (1/8/7), FP16 (1/5/10);
then sub-byte: FP8 E4M3 (max 448, forward weights/activations), FP8 E5M2 (max 57,344, gradients),
FP6 E3M2 (max 28), **FP4 E2M1 (max 6, only 16 distinct values total)**. Rule: same 8-bit exponent
(FP32/TF32/BF16) = same dynamic range, swap freely with no overflow, they differ only in mantissa
precision; FP16's 5-bit exponent trades range for precision (why training prefers BF16).

**FP4 number line** — the entire codebook is enumerable: ±{0, 0.5, 1, 1.5, 2, 3, 4, 6}, spacing
**linear (step 0.5) near zero, exponential above 2** — resolution concentrates where weights
cluster. One global scale is hopeless; per-block scaling is what makes 4-bit usable.

**Block scaling: MXFP4 vs NVFP4** — MXFP4 (OCP standard): block of 32 elements + 1-byte E8M0
(power-of-2) scale ⇒ 4.25 bits/element effective (gpt-oss ships this; 120B fits on one H100).
NVFP4 (Blackwell-native): block of 16 + FP8 E4M3 fractional scale + one FP32 per-tensor scale ⇒
4.5 bits/element, smaller block + finer scale ⇒ less quantization error. Why E4M3 beats E8M0:
power-of-2 scales can be off by up to ±50% (round up to nearest power of 2); E4M3's 3 mantissa
bits give 12.5% steps that hug the true block max. Two-level scaling: the per-block FP8 scale
handles local range, one per-tensor FP32 scale handles global magnitude so the FP8 scale never
overflows.

### 2.6 — The Landscape (energy, co-design, vendors, edge)

**Energy is the real budget** — a log-scale bar chart of pJ/operation: INT8 add 0.03 · INT8 mul
0.2 · FP16 mul 1.1 · FP32 mul 3.7 · SRAM 8KB access 10 · generic on-chip 50 · **one 64-bit DRAM
read ≈ 2,000 pJ (2 nJ)**. *"A DRAM access costs ~500 [more precisely ~540] FP32 multiplies.
Arithmetic is free; moving operands is the whole budget."* (Horowitz ISSCC 2014 @ 45nm + Dally
2023, DRAM range 1.3–2.6 nJ.)

**Convergent hardware/model co-design** diagram: hardware ships big NVLink domains → models
answer with rack-sized MoE width; hardware ships FP4 tensor units → models answer with FP4-native
training; hardware ships TMEM/decode-tuned chips → models answer with GQA/MLA/sparse attention.
*"Newest move: phase-specialized chips — GB300 NVL72 tuned for decode, prefill parts on the
roadmap."*

**Cross-vendor table** (same physics, different silicon) — memory, type, TB/s, BF16 TF, FP8 TF,
scale-up fabric:

| Chip | Mem | BW | BF16 | FP8 | Scale-up |
|---|---|---|---|---|---|
| AMD MI300X | 192 GB HBM3 | 5.3 TB/s | 1,307 | 2,615 (sparse!) | Infinity Fabric |
| AMD MI325X | 256 GB HBM3e | 6.0 TB/s | 1,307 | 2,615 (sparse!) | Infinity Fabric |
| AMD MI355X | 288 GB HBM3e | **8.0 TB/s** | 5,000* | 10,100 | Infinity Fabric |
| Google TPU v6e | 32 GB HBM | 1.6 TB/s | 918 | — | ICI 1.2 Tb/s |
| Google TPU v7 | 192 GB HBM3e | 7.4 TB/s | 2,307* | 4,614 | ICI 9.6 Tb/s |
| AWS Trainium2 | 96 GB HBM3 | 2.9 TB/s | 667 | 1,300 | NeuronLink |
| Intel Gaudi 3 | 128 GB HBM2e | 3.7 TB/s | 1,835 | 1,835 | RoCE 1.2 TB/s |
| **NVIDIA B200** | 192 GB HBM3e | **8.0** | 2,250 | 4,500 | NVLink5 1.8 TB/s |

(MI300X/MI325X FP8 column is flagged by the verification pass as mislabeled — 2,615 is the
*sparse* number, no confirmed dense FP8 spec exists for CDNA3.)

**Edge/local-inference menu** (capacity × bandwidth × price, June 2026):

| Device | Mem | BW | AI compute | Price |
|---|---|---|---|---|
| Jetson Orin Nano Super | 8 GB | 102 GB/s | 67 TOPS† | $250 |
| DGX Spark (GB10) | 128 GB | 273 GB/s | ~1 PF FP4† | $4,000 |
| Strix Halo (AI Max+ 395) | 128 GB | 256 GB/s | 50 NPU TOPS | $2,000 |
| Apple M4 Max | 128 GB | 546 GB/s | 38 TOPS | $2,000+ |
| Apple M3 Ultra | 512 GB | 819 GB/s | — | $4,000+ |
| iPhone 17 Pro (A19 Pro) | 12 GB | 76 GB/s | 38 TOPS | $1,000 |
| RTX 5090 (system) | 32 GB | 1,792 GB/s | 1,676 TF FP4 | $4–6k |
| H100 (per GPU) | 80 GB | 3,350 GB/s | 989 TF BF16 | $20–30k |

Derived decode-bound bar chart (8B-4bit, ≈4.5 GB/token swept, tok/s = BW/4.5): iPhone A19 17 ·
Orin Nano 23 · Strix Halo 48 · DGX Spark 61 · M4 Max 121 · M3 Ultra 182 · RTX 5090 398 (realized
~50–80% of these bounds in practice). Measured real data point: **DGX Spark on Llama-3.1-8B FP8:
7,991 tok/s prefill vs 20.5 tok/s decode — a ~390× gap on the same chip.**

**The SRAM-extreme pole** (Groq/Cerebras): Groq LPU 230 MB/chip @ ~80 TB/s on-chip BW (needs
15+ chips for a 7B model; rack = 576 chips); Cerebras WSE-3 44 GB/wafer @ **21 PB/s** (~6,000×
H100's HBM bandwidth) — *"you buy bandwidth with silicon area and dollars, not stacked DRAM."*

### 2.7 — Recap: the three growth rates (the chapter's thesis, stated as a rule of thumb)

> **Compute ~4×/generation** (2× architecture × 2× format) · **Bandwidth ~2×/generation** (HBM,
> NVLink, PCIe, IB) · **Capacity ~1.7×/generation** (process + stacking, worsened by the 2026
> DRAM squeeze). *"Compute outruns bandwidth outruns capacity. FLOPs-per-byte you're allowed to
> spend keeps rising — the model gets hungrier, the pipe doesn't."*
> Rubin (projected): compute ~4× (9→35 PF dense FP4), bandwidth ~2.5× (8→20 TB/s HBM4), capacity
> 1.5× (192→288 GB) ⇒ FP8 ridge point climbs 562 → ~800 FLOP/byte.

---

## 2b. The roofline / arithmetic-intensity machinery (technically §1, but inseparable from §2)

This is the load-bearing analytical framework the whole deck (and any d2l rewrite) should
adopt — it lives in `sections/01-overview.tex`, using a running example (**Qwen3-8B, 40k
context, batch 1**) that's reused throughout.

- **Opening ratio, verbatim on the title-adjacent slide:** *"4,500 TFLOP/s (FP8) ÷ 8 TB/s memory
  bandwidth ⇒ 562 FLOPs/byte. Keep the cores busy: ~500 FLOPs per byte fetched. One token
  reuses each weight once ⇒ the chip starves. Batch-1 decode is usually a memory-traffic
  problem, not a math problem."*
- **Arithmetic intensity, defined formally:** intensity = FLOPs done / bytes moved; for a matmul
  (n×k)(k×m): intensity ∝ 2nkm/(nk+km) = 2nm/(n+m) (ignoring output writes/element size).
  Prefill: each weight serves L_ctx tokens ⇒ intensity ~ L_ctx. Decode: each weight serves 1
  token ⇒ intensity ~1.
- **Ridge point = peak FLOP/s ÷ bandwidth** — worked as 4,500 TF / 8 TB/s ≈ **562** for B200 FP8;
  **234** for RTX 5090 FP8 (419 TF / 1.79 TB/s).
- **The roofline plot itself**: log-log, sloped region (bandwidth wall, slope = bandwidth) meets
  flat region (compute ceiling) at the ridge. B200 and RTX 5090 both drawn (different absolute
  numbers, identical shape). **Decode sits at intensity ~0.7, <1% of peak FLOPs — far down the
  slope. Prefill sits at the flat ceiling.**
- **Two supplementary charts** track the ridge point over hardware generations (P100→B200,
  GTX1080→RTX5090), both at fixed 16-bit precision (ridge climbs 29→295 data-center,
  28→328 consumer) and *normalized per operand byte at each generation's native format*
  (P100/FP16 14 → V100/FP16 69 → A100/BF16 77 → H100/FP8 591 → B200/FP4 2,250 — **>1,000×
  gap** when format shrinkage compounds with the raw hardware ridge climb).
- **Worked back-of-envelope table** (Qwen3-8B, 40k ctx, batch 1):
  - Prefill compute: FFN 435 TF + attn projections 121 TF + attention scores O(L²) 472 TF =
    **~1.03 PF total**. B200 time = 1.03 PF / 4,500 TF/s ≈ **0.23 s**.
  - Decode: FLOPs ≈ 2·N_params ≈ 16 GFLOP/token, but bytes touched = weights 16.4 GB + KV 6 GB
    ≈ **22 GB/token** ⇒ <1 FLOP/byte. Generation bound = 8,000 GB/s / 22 GB/tok ≈ **364 tok/s**
    (B200) vs. ≈**81 tok/s** (RTX 5090, 4.5× less bandwidth ⇒ 4.5× slower decode).
  - KV-cache byte formula, exact and reusable as-is:
    `KV/token = 2 · L · n_kv · d_head · 2B = 2·36·8·128·2 = 147 KB` (Qwen3-8B, BF16 KV);
    at 40k context ≈ 6 GB/stream.
- **Batching changes the picture**: B parallel streams share one weight read ⇒ intensity × B
  (decode marches up the roofline toward the ceiling — "how servers hit high throughput"), but
  KV scales × B too (32 streams × 6 GB = 192 GB) — *"the fundamental serving tension."*
- **Consequence 1**: prefill wants FLOP/s (compute monsters, FP8/FP4 tensor cores); decode wants
  bytes/s + capacity (fat fast HBM) ⇒ motivates **disaggregated prefill/decode serving**
  (previews DistServe arXiv 2401.09670, Mooncake arXiv 2407.00079 — covered in depth in §3).
- **Consequence 2 (the "local-inference loophole")**: at batch 1, capacity+bandwidth beats raw
  FLOPs — a 70B-Q4 model (40 GB) won't even fit on a 32 GB RTX 5090, but an M3 Ultra (512 GB)
  runs it despite half the bandwidth.

---

## 3. Other chapters — brief summaries (relevance to d2l topics requested)

### §3 Serving (`03-serving.tex`, 3,760 words) — batching, paging, caching, disaggregation, speculative decoding

Covers: continuous batching (fights GPU-time fragmentation vs. static batching), chunked prefill
(don't let one long prompt stall the queue), PagedAttention as "virtual memory for the KV
cache" (pre-vLLM systems waste 60–80% of allocated KV memory on internal fragmentation; paging
buys 2–4× throughput), RadixAttention (SGLang's prefix-sharing radix tree over token sequences,
up to 5× on shared-prefix workloads), cache-aware scheduling, the "copy vs. recompute" trade for
a 6 GB KV cache moving down the memory ladder, prompt-caching price discounts across providers
(**Anthropic 0.1× cached-read, OpenAI 0.1–0.5× model-dependent, Gemini 0.1–0.25×**), disaggregated
prefill/decode (Mooncake architecture: +59–498% effective request capacity in real traces,
>100B tok/day at Kimi), matching the GPU fleet topology to the model (NVL72 ↔ large MoE
all-to-all). **Speculative decoding gets real depth** — draft-then-verify framed as *"spend idle
FLOPs to buy tokens"* (exploits the same arithmetic-intensity thesis: verifying k drafted tokens
in one forward pass amortizes the weight sweep), a frame proving speculation is exact via
**maximal coupling** (elegant, proof-shaped — good d2l material), EAGLE-3/MTP as modern drafters,
and an explicit "when speculation backfires" frame (large-batch/compute-bound regimes reverse the
win). Ecosystem survey of vLLM/SGLang/TensorRT-LLM/Dynamo/llama.cpp/ollama/mlx with star counts.
**Relevance to d2l "compilation/kernels/fusion":** minimal — no torch.compile/CUDA-graph/kernel
content; explicitly out of scope (see §4 below). **Relevance to "benchmarking/profiling":**
thin — metrics (TTFT, TPOT, throughput) are used but no profiling *tools* frame; the deck's own
gap-analysis flags this as an acknowledged, still-open gap (§ below).

### §4 Weight Compression (`04-weights.tex`, 4,861 words) — MoE + quantization

Two orthogonal axes: fewer weights (MoE) and fewer bits (quantization). **MoE**: FFN is ~65% of
a dense model's weights (first place to hunt for savings); MoE lineage tracked by sparsity ratio
climbing from 1:4 to 1:31 across generations; DeepSeekMoE's "sparse plus [always-on] dense shared
expert" pattern; load balancing (saving all the experts from collapse); serving MoE's all-to-all
communication wants one big NVLink domain (ties back to §2's NVL72 frame). **Quantization**, in
depth and mathematically rigorous — the strongest "portable proof" content in the whole deck:
- RTN (round-to-nearest) formalized: ŵ = s·(clip(⌊w/s⌉+z, 0, 2^b−1) − z), granularity dial
  (per-tensor / per-channel / per-group g=128, the modern default).
- The outlier problem: one heavy weight stretches the shared scale, starving 127 neighbors of
  resolution — "the villain of low-bit quant," with three canonical fixes: **compensate** (GPTQ),
  **protect** (AWQ), **remove** (rotation).
- **GPTQ derived from first principles** (genuinely d2l-proof-shaped, could port near-verbatim
  into a "Proposition/Proof" block): minimize ‖WX − ŴX‖² over a calibration set ⇒
  tr[(W−Ŵ)ᵀQ(W−Ŵ)] with Q = XXᵀ (the layer's data Hessian H=2Q). Splitting free vs. already-
  quantized coordinates and minimizing over the free ones gives the closed-form update
  δ_f = −Q_ff⁻¹Q_fq·δ_q, and substituting back yields the **Schur-complement** cost
  δ_qᵀ[Q_qq − Q_fqᵀQ_ff⁻¹Q_fq]δ_q = δ_qᵀ[(Q⁻¹)_qq]⁻¹δ_q. Made fast for 175B-parameter models via
  fixed column order, 128-column blocks batched as GEMMs, and a single Cholesky factorization of
  Q⁻¹ (the needed columns *are* the Cholesky factor). Lineage traced: OBD → OBS → OBQ → GPTQ.
- AWQ: "1% of the channels carry the signal" — protect outlier channels via per-channel scaling
  instead of compensating after the fact.
- Rotations to remove outliers entirely (push mass toward Gaussian so a lattice quantizer packs
  it better than a square grid); a QTIP/trellis teaser connecting to rate-distortion theory
  (scalar quantization of Gaussian data wastes 0.5·log₂(πe/6) ≈ 0.254 bits/sample vs. the R(D)
  bound — a genuinely elegant, citable inequality).
- A useful aside: **why training in low precision is harder than inference** — tiny gradients
  underflow FP16 (needs loss scaling), FP32 master weights avoid lr·g vanishing, long dot
  products need FP32 accumulation / stochastic rounding; DeepSeek-V3's FP8 training recipe
  (E4M3 GEMMs, tile/block scaling 1×128 activations / 128×128 weights, promote to FP32 every 128
  K-elements) is a concrete "how it's actually done" case study. Inference is far more forgiving
  (forward-only, fixed weights, errors don't compound).
- Closes with "the bits-vs-quality ladder for 2026" and a practical toolbox recommendation.
**Relevance to d2l "arithmetic/precision":** this is the single richest chapter for that request
— the RTN formula, GPTQ derivation, and FP8-training recipe are all near-drop-in for a d2l
"Quantization" section (PyTorch code cells implementing RTN + a toy GPTQ pass would teach
exactly what the slides assert).

### §5 KV Compression (`05-kv.tex`, 5,063 words) — the deepest, most novel-methods-heavy chapter

Structural head-sharing (GQA/MQA) → **MLA (multi-head latent attention)**: low-rank-compresses
K/V into a shared latent, with an "absorb the up-projections at decode time" trick that avoids
ever materializing full-size K/V (DeepSeek-V2/V3's central innovation; MLA byte budget explicitly
compared to GQA). MHA2MLA retrofits MLA onto an already-trained MHA model via joint SVD of
[K_nope ‖ V]. State quantization (KIVI: quantize K and V along *opposite* axes since their
distributions differ; KVQuant: quantize keys before RoPE, since RoPE rotation harms
quantizability; TurboQuant/OSCAR: rotate toward the rate-distortion bound / the attention-
weighted covariance direction). "Reading less cache": most attention is local (sliding-window
argument), then NSA (three gated branches: compressed / selected / sliding, trained natively) and
MoBA (MoE-style routing over context blocks). Dropping tokens outright: eviction policies,
attention sinks (softmax must park its mass somewhere even when nothing is relevant — hence early
tokens get anomalously high attention), KVzip (score by reconstruction error, evict once). Ends
with **state-space models as the "escape hatch"** — eliminating the KV cache's O(context) growth
entirely by carrying a fixed-size recurrent state, and a "best of both worlds" hybrid-architecture
frame, plus a "what stacks with what" compatibility matrix across all the KV techniques covered.
**Directly relevant** to d2l's SSM/state-space chapter (ch. 13 in this repo's numbering) as an
inference-cost argument for why SSMs matter, complementary to the training-time treatment.

### §6 Wrap-Up & Resources (`06-resources.tex`, 1,539 words)

Recap map, a "ten numbers to remember" table, a curated tutorials/courses list (**How to Scale
Your Model** — Austin et al., Google DeepMind 2025, `jax-ml.github.io/scaling-book` — explicitly
the deeper roofline→parallelism→KV reference; **Ultra-Scale Playbook**, HF/nanotron, 3D
parallelism + memory budgets; **GPU MODE** lectures, CUDA→FlashAttention→quant, 100+ lectures;
**Horace He, "Making DL Go Brrrr"** — compute vs. bandwidth vs. overhead, operator fusion;
**kipply, "Transformer Inference Arithmetic"**), a papers-by-section bibliography, a
speculative-decoding resource list, and a code-repo list by category (serving engines, KV cache,
weight compression) with GitHub star counts as of June 2026.

**Explicitly stated as out of scope** (its own "What We Skipped" slide — directly answers the
"other chapters" ask):
> *"Multi-GPU parallelism (TP/PP/EP) — whole topic of its own → Ultra-Scale Playbook,
> NVIDIA/Megatron-LM. Pruning & distillation. **Kernels & compilation — CUDA graphs,
> torch.compile → gpu-mode/lectures.** Structured decoding. Multi-LoRA serving. Ring/
> context-parallel attention. Audio/video/realtime."*

So: **compilation/kernels/fusion and parallel/multi-GPU training are NOT covered in this repo** —
they're deliberately out of scope, with only pointer citations (Ultra-Scale Playbook, Megatron-LM,
GPU MODE lectures) rather than content. `ref/gaps.md` (a self-audit against 9 reference sources:
Scaling Book, CS336, Ultra-Scale Playbook, 2 surveys, vLLM/SGLang docs, GPU MODE, an EMNLP'25
inference tutorial) independently confirms this and flags **profiling/benchmarking tooling
(nsight systems, torch profiler, `vllm bench serving`, genai-perf, MLPerf Inference) and
in-node tensor parallelism as still-open gaps even for this deck's own inference-serving scope**
— i.e., if the d2l chapter needs profiling-tool coverage or TP/PP/EP training-parallelism content,
this repo is not the source; go to the Ultra-Scale Playbook / Megatron-LM / GPU MODE lectures it
points to instead.

---

## 4. Reusability assessment

### Format friction

- **All 52 image assets in `figures/` are raster PNG/JPEG** (product photos, vendor die shots,
  paper screenshots) — **not** matplotlib-generated, not SVG, no generator scripts anywhere in
  the repo. These are not portable to d2l's house style at all (`tools/gen_mdl_*_figures.py`
  byte-idempotent matplotlib→SVG convention) and in several cases are third-party/vendor IP
  (NVIDIA product photos, paper figures) that would need re-licensing or redrawing from scratch
  even if the style matched.
- **The actual data visualizations — bandwidth ladder, latency ladder, bandwidth-over-time,
  roofline plot, arithmetic-intensity-over-time, format bit-layout diagrams, FP4 number line,
  energy bar chart, tensor-core lineage — are inline TikZ/pgfplots**, not files. These carry the
  real content and **are the most valuable reusable asset class**: same underlying data,
  straightforward to redraw as matplotlib generators per d2l's house figure style (they're
  already simple bar/line/log charts, not elaborate TikZ art). This is a clean "port the numbers,
  redraw the chart" job per figure, not a format-mapping exercise.
- No code at all (no `.py` in the repo) — nothing to import as a working code cell; the GPTQ
  algorithm box and the KV-byte formula would need to be turned into actual PyTorch snippets from
  scratch, but the math itself is already correct and citable.

### Top-10 reusable assets (ranked by "port this into the d2l chapter" value)

1. **The roofline / arithmetic-intensity framework itself** (§1, `01-overview.tex`) — ridge
   point = peak FLOP/s ÷ bandwidth; intensity ∝ 2nm/(n+m) for a matmul; prefill vs. decode as
   opposite-ends-of-the-roofline case study. This is the organizing idea the whole d2l hardware
   chapter should probably adopt (the current d2l `hardware.md` has no roofline treatment at
   all).
2. **The Qwen3-8B worked numeric example** (vocab/d/layers/heads/KV formula → 8.19B params, 16.4
   GB weights, 147 KB KV/token, 6 GB @ 40k ctx) — a complete, internally-consistent running
   example ready to reuse or swap for a d2l-preferred model.
3. **The shoreline/beachfront die-area argument** (compute ∝ area ∝ n², I/O ∝ perimeter ∝ n;
   doubling die side ⇒ 4× compute but 2× I/O) — a crisp, quotable *architectural* explanation the
   current d2l hardware chapter lacks entirely.
4. **The three spec tables** (die/process/power; memory; compute-by-format) for
   H100/B200/RTX4090/RTX5090 — a ready-made, fact-checked replacement for d2l's stale Skylake/RTX
   2080 Ti/V100-era numbers.
5. **FlashAttention worked example** (naive Θ(N²) HBM round-trip, 3.2 GB S-matrix at N=40k;
   online-softmax recurrence; FA1→FA2→FA3 utilization arc 25%→73%→85%) — good notebook material:
   a PyTorch cell computing the naive-vs-tiled HBM byte count would teach exactly this.
6. **GPTQ's Schur-complement derivation** — genuinely proof-shaped math (calibration-weighted
   least squares → closed-form correction via a Hessian sub-block), a strong fit for d2l's
   "intuition-first, elegant proof" convention; portable into a quantization section for either
   this chapter or a future compression-focused one.
7. **The energy-per-operation ladder** (INT8 add 0.03 pJ → DRAM read 2,000 pJ, ~500× ratio,
   Horowitz/Dally-sourced) — a strong "why move fewer bytes" motivating figure.
8. **Number-format bit-layout diagrams and the FP4 16-value codebook** — precise, simple to
   redraw, and directly useful for any precision/quantization discussion (already d2l-appropriate
   in tone: enumerable, concrete).
9. **The GB200 NVL72 rack numbers** (72 GPU / 36 Grace, one NVLink domain, 130 TB/s, 13.5 TB HBM,
   ~120 kW) plus the cross-vendor table (AMD/Google/AWS/Intel) — good for a "current landscape"
   section, with the caveat that these numbers date fastest.
10. **The canonical-numbers cheat sheet in `ref/style-contract.md`** — a hand-curated, single-
    source-of-truth block of every verified number in the deck (specs, formulas, rules of thumb),
    already organized by topic; the fastest way to seed a d2l chapter's numeric content without
    re-deriving from the dossiers.

### What needs updating for 2026 (beyond routine "numbers age")

- Everything here is dated **June 2026**, so as of the current date (per this session) it's
  essentially current — no near-term staleness beyond the deck's own built-in caveat. The
  Rubin-generation numbers are explicitly marked "announced, not shipping" and should be flagged
  the same way if reused.
- The **DRAM-squeeze pricing note** ("DDR5 +307% Q4'25") and **prompt-caching price table**
  (provider-specific discounts) are the fastest-decaying numbers in the deck — pure market data,
  not hardware physics; a d2l chapter should either drop these or clearly timestamp them, per
  the project's own house rule (`CLAUDE.md`: "quote only the precision that survives
  re-execution," "never quote... exact... ratios" that will drift).
- Two items already flagged by the repo's own fact-check as needing a small fix if copied
  verbatim: the MI300X/MI325X FP8 column (2,615 TF is the *sparse* figure, mislabeled as dense in
  the "dense unless noted" table — footnote it or blank it) and the "memory linear in N" claim on
  the FlashAttention tiling slide (should read "~9× fewer HBM accesses; no extra O(N²) HBM
  buffer," not "linear in N" — HBM traffic is still Θ(N²d²/M), i.e., still quadratic in N with a
  smaller constant).

### Scope mismatch to flag explicitly

This deck is an **inference-serving** tutorial (batch-1/small-batch decode economics, KV-cache
economics, serving engines). d2l's "Computational Performance" chapter today (`hardware.md`,
`multiple-gpus.md`, `multiple-gpus-concise.md`, `auto-parallelism.md`, `async-computation.md`,
`hybridize.md`, `parameterserver.md`) is **training-and-general-systems** focused: multi-GPU data
parallelism, compilation/graph-mode (`hybridize.md`), async dataflow, parameter servers. The MLSS
deck's hardware chapter (die/memory/format specs, roofline, energy) transplants cleanly and is a
strict upgrade over the current stale content; its serving/KV/quantization chapters are adjacent
but topically inference-specific and would need reframing (or a new chapter) rather than a
straight port into the *training*-parallelism sections of Computational Performance. The deck
itself has almost nothing on multi-GPU *training* parallelism (TP/PP/EP, ZeRO) or on
compilation/kernel-fusion tooling (torch.compile, CUDA graphs) — both are explicitly out of scope
and only pointer-cited (Ultra-Scale Playbook, Megatron-LM, GPU MODE lectures), so those parts of
the existing d2l chapter structure will need a different primary source.
