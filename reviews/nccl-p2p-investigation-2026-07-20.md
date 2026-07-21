# NCCL PCIe allreduce investigation — why NCCL busbw is ~2.2 GB/s on the 4×4090 box

**Date:** 2026-07-20 · **Box:** `4090-0` · **Mode:** investigation only (no system
state changed, no source edited, no commit).

## TL;DR

The 2.2 GB/s NCCL busbw is **not** an IOMMU/ACS/DMA problem and **not** a P2P
problem. The discriminating measurement: a hand-rolled copy-engine bounce
(GPU→pinned host→GPU), which crosses the *same* IOMMU-translated PCIe path,
sustains **~20 GB/s** one-way. The cause is narrower: with P2P disabled (GeForce
segmentation), NCCL falls back to its **SHM transport**, and that transport's
*default* mode moves bytes with a **GPU kernel doing direct load/store to mapped
host memory** — and GPU reads-over-PCIe are latency-bound to ~2 GB/s. Setting
one env var, **`NCCL_SHM_USE_CUDA_MEMCPY=1`**, flips SHM to the DMA copy engine
(`via SHM/CE/direct`) and lifts busbw **~5×** — measured **11.2 GB/s** at k=2
same-bridge and k=4, **19.4 GB/s** at k=2 across-bridge — env-only, no reboot,
numerically verified.

Alex's hypothesis (the L40S forum thread → `iommu=pt`) is a real and related
class, but that thread's symptom was *hangs*, resolved to a stable 13 GB/s;
ours is stable-but-slow. `iommu=pt` is good hygiene here and may add a little to
the copy-engine path, but the evidence says it is **not** what caps us at 2.2.

---

## 1. System facts (read-only)

| Fact | Value |
|---|---|
| Kernel | `6.8.0-124-generic` (Ubuntu), `x86_64` |
| `/proc/cmdline` | `BOOT_IMAGE=… root=/dev/mapper/vg0-lv--root ro` — **no `iommu=`, no `intel_iommu=`, no `amd_iommu=`** |
| CPU | AMD EPYC 7502 32-core, 1 socket, SMT on (64 threads) |
| NUMA | **single node (NPS1)**, 0-63, 257 GB |
| GPUs | 4× GeForce RTX 4090 (AD102), 24 GB each |
| Driver / CUDA | **595.71.05** / CUDA 13.2 runtime |
| torch / NCCL / CUDA build | **2.11.0+cu128** / **NCCL 2.28.9** / cu12.8 |
| IOMMU | **AMD-Vi ENABLED** — `/sys/class/iommu` = `ivhd0..3`, **65 IOMMU groups** populated; **NOT in passthrough** (no `iommu=pt`) |
| `dmesg` | not readable without sudo (returns empty; `kernel.dmesg_restrict`) |
| P2P | `nvidia-smi topo -p2p rw` = **CNS** everywhere (Chipset Not Supported); `can_device_access_peer=False` — GeForce segmentation |

### PCIe topology (`nvidia-smi topo -m`, `lspci -tv`)

Single-socket EPYC exposes 4 IO dies → root complexes `0x00 / 0x40 / 0x80 / 0xc0`.
The GPUs live on two of them, one GPU per bridge:

```
root complex 0x80 ── 80:01.1 ─[81]─ GPU0        root complex 0xc0 ── c0:01.1 ─[c1]─ GPU2
                  └─ 80:03.1 ─[82]─ GPU1                          └─ c0:03.1 ─[c2]─ GPU3
```

`topo -m` distances: **GPU0↔GPU1 = PHB** and **GPU2↔GPU3 = PHB** (same root
complex, through the host bridge); **any 0/1 ↔ 2/3 = NODE** (across IO dies, over
the intra-package interconnect). **No PXB/PIX** — there is no PCIe switch and no
NVLink; every GPU-GPU byte goes up to the CPU and back. Single NUMA node, so all
`CPU Affinity = 0-63`.

**ACS:** `ACSCtl` on the bridges could **not** be read — `lspci -vvv` needs root
for extended capabilities (empty output as non-root). *Requires Alex* to confirm:
`sudo lspci -vvv -s 80:01.1 | grep -i ACSCtl` (and `80:03.1`, `c0:01.1`,
`c0:03.1`). Note: ACS only matters for *direct* P2P, which is already disabled by
the driver on GeForce — so ACS is effectively moot for this box (see §5).

---

## 2. Primary sources

| URL | One-line takeaway |
|---|---|
| [NVIDIA forum: NCCL hangs on L40S PCIe, resolved via IOMMU passthrough](https://forums.developer.nvidia.com/t/nccl-hangs-on-l40s-gpus-pcie-resolved-via-iommu-passthrough/368169) | Symptom was **hangs/deadlocks** (not slow BW) on dual-socket L40S PCIe; IOMMU was intercepting P2P traffic; fixed with **IOMMU passthrough**, bandwidth then *stabilized* ~13 GB/s. Different failure mode from ours. |
| [nccl-tests #307 — why is NCCL perf low on RTX 4090](https://github.com/NVIDIA/nccl-tests/issues/307) | 8×4090: allreduce **~5 GB/s busbw**, flat 128 MB→1 GB; NCCL routes through **PHB (CPU)**, single channel, "NVLS not available". Same class as ours, milder. |
| [nccl-tests #74 — allreduce slower with P2P-via-PCIe (A40, EPYC 7302)](https://github.com/NVIDIA/nccl-tests/issues/74) | On EPYC, PCIe-P2P can be *slower* than host-staged; disabling IOMMU in BIOS traditionally helped — but **"no effect with RTX 4090"**. |
| [nccl-tests #117 — all_reduce_perf hangs on multi-4090](https://github.com/NVIDIA/nccl-tests/issues/117) | 4090 P2P/hang issues; `NCCL_P2P_DISABLE=1`+`NCCL_SHM_DISABLE=1` (force socket) unblocks hangs. |
| [nccl #1285 — EPYC 7K62 + 4090 bandwidth too low](https://github.com/NVIDIA/nccl/issues/1285) | Same complaint (details in screenshots): "why is p2p performance so slow" on EPYC+4090. |
| [voipmonitor/nccl-tuner-amd](https://github.com/voipmonitor/nccl-tuner-amd) | NCCL's cost model is NVLink-tuned and mis-picks protocol on **AMD EPYC + PCIe**; a tuner plugin recovers up to ~38% latency. Corroborates "defaults are wrong for this topology." |
| [NCCL env docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html) | Documents `NCCL_SHM_DISABLE` ("SHM used when P2P cannot happen; host memory"), `NCCL_P2P_DISABLE`, `NCCL_P2P_LEVEL`. **`NCCL_SHM_USE_CUDA_MEMCPY` is NOT documented here** — it is a source-level knob (present in the bundled 2.28.9 lib; see §4). |

---

## 3. Measurement matrix

Microbench: `torch.distributed` ring allreduce, fp32, warmup + 15 timed iters,
`busbw = algbw · 2(k-1)/k` (nccl-tests convention). Scripts in
`…/scratchpad/nccl-probe/`. All GB/s decimal. (At k=2, busbw ≡ algbw.)

### 3a. Baseline (default NCCL) — payload sweep, confirms flat ceiling

| config | k | topo | 8 MB | 32 MB | 128 MB | 256 MB | 512 MB |
|---|---|---|---|---|---|---|---|
| default | 2 | same-bridge PHB | 2.34 | 2.22 | 2.26 | **2.24** | 2.24 |
| default | 4 | PHB+NODE (busbw) | 2.16 | 2.13 | 2.14 | **2.17** | 2.19 |

**Flat from 8 MB to 512 MB and flat from k=2 to k=4** — a transport ceiling, not
a latency/payload effect. Default k=2 across-bridge (NODE) = **3.35** (256 MB).

### 3b. Env-var matrix (k=2 same-bridge, 256 MB, busbw)

| config | busbw | note |
|---|---|---|
| baseline | 2.24 | `via SHM/direct/direct` |
| `NCCL_P2P_DISABLE=1` | 2.24 | no-op (P2P already off) ✓ |
| `NCCL_SHM_DISABLE=1` | 1.64 | forces NET/**socket** — worse |
| `NCCL_ALGO=Ring NCCL_PROTO=Simple` | 2.24 | already the default pick |
| `NCCL_ALGO=Tree` | 2.15 | — |
| `NCCL_PROTO=LL` / `LL128` | 1.30 / 1.92 | worse |
| `NCCL_BUFFSIZE=8/16/32 MB` | 2.24 / 2.17 / 2.17 | no effect |
| `NCCL_NTHREADS=640` | 2.26 | no effect |
| `NCCL_SHM_MEMCPY_MODE=3` (alone) | 2.23 | no-op without the flag below |
| **`NCCL_SHM_USE_CUDA_MEMCPY=1`** | **11.24** | `via SHM/CE/direct` — copy engine ★ |
| `…=1` + `NCCL_BUFFSIZE=16 MB` | **11.54** | best k=2 same-bridge |
| `…=1` + `NCCL_SHM_MEMCPY_MODE=3` | 11.07 | mode adds nothing over the flag |

Standard protocol/algo/buffer/thread tuning is **flat** at ~2.2. Only switching
the SHM transport to the copy engine moves it.

### 3c. The copy-engine fix across topology & scale (256 MB)

| config | k=2 same-bridge (PHB) | k=2 across-bridge (NODE) | k=4 |
|---|---|---|---|
| baseline busbw | 2.24 | 3.35 | 2.17 |
| **`SHM_USE_CUDA_MEMCPY=1`** busbw | **11.24** | **19.44** | **10.59** |
| `…=1` + `BUFFSIZE=16 MB` | 11.54 | — | **11.14** |

CE-fix payload sweep (busbw): k=2 same-bridge 9.1→11.3 (8→512 MB); k=4
9.1→10.7. Across-bridge is *faster* (19.4) than same-bridge (11.2) because the
two GPUs bounce through **independent IO-die → memory paths** instead of
contending on one host bridge — a topology effect, not a bug.

### 3d. Reference: copy-engine bounce and H2D (single process, no NCCL)

| path | GB/s |
|---|---|
| pinned H2D (256 MB) | **23.7** |
| GPU0→pinned host→GPU1 bounce, one big copy | **19.9** one-way (39.8 wire) |
| same bounce, chunked 4 / 8 / 16 MB | **19.8 / 19.9 / 19.9** one-way |

The DMA copy engine — which is *equally* IOMMU-translated — delivers ~20 GB/s
**even at 4 MB chunks**. This is the decisive control.

### 3e. What NCCL actually picks (`NCCL_DEBUG=INFO`)

- Ring algorithm, 4 channels; topology model rates the GPU-GPU PHB link at
  ~20-24 GB/s (`bw 20.0 … type PHB/PIX`) — NCCL *knows* the wire is fast.
- P2P disabled: `P2P is disabled between connected GPUs 0 and 1`.
- Default transport line: **`Channel 00 : 0[0] -> 1[1] via SHM/direct/direct`**.
- With the flag: **`via SHM/CE/direct`** + `NCCL_SHM_USE_CUDA_MEMCPY set … to 1`.
- Correctness with the flag verified: k=4 allreduce of rank-valued tensors sums
  to 10 = 1+2+3+4, `allclose=True`.

---

## 4. Root-cause analysis

**Root cause: NCCL's default SHM transport uses a GPU kernel doing direct
load/store to mapped host memory; GPU reads-over-PCIe are latency-bound to
~2 GB/s. It is not IOMMU, not ACS, not P2P, not protocol/buffer tuning.**

The discriminators, in order of force:

1. **The copy-engine bounce (§3d) hits ~20 GB/s through the same IOMMU.** If AMD-Vi
   translation (no `iommu=pt`) or ACS root-complex bouncing were the ceiling, this
   DMA path — which is also translated and also root-complex-staged — would be
   slow too. It isn't. This alone rules out IOMMU/ACS/DMA as the cause of the 10×
   gap.
2. **Flipping only the SHM copy mechanism (`…direct/direct` → `…/CE/direct`)
   recovers 5×** (§3b/§3c), with *nothing else changed*. The variable that moves
   the number is precisely "kernel load/store vs copy engine."
3. **Everything else is flat** (§3b): algo, proto, buffsize, nthreads, channels —
   all ~2.2. A protocol/tuning bug would respond to at least one of these.
4. **Flat 8 MB→512 MB and flat k=2→k=4** — a fixed per-byte transport cost, i.e. a
   transport-mode ceiling, not startup latency or channel scaling.
5. **The XLA data point resolves cleanly.** JAX/XLA also uses NCCL yet gets
   4.5-8.6 GB/s — *between* our two regimes. XLA does not sit on the pathological
   `direct/direct` path (ch13 already had to set `NCCL_LOCAL_REGISTER=0` because
   XLA drives NCCL with buffer registration). Our CE-enabled torch NCCL (11-19)
   *exceeds* XLA, confirming the default torch path is the uniquely pessimal one,
   not the wire.

Bundled-lib confirmation: `strings libnccl.so.2` (2.28.9) contains
`NCCL_SHM_USE_CUDA_MEMCPY`, `NCCL_SHM_MEMCPY_MODE`, `NCCL_P2P_USE_CUDA_MEMCPY`.
The knob is real in this build but **undocumented** in the public env-var page.

**Why the forum thread doesn't apply directly:** the L40S thread fixed *hangs*
(a DMA-remapping deadlock) with `iommu=pt`; our box does not hang — it completes
correctly, just slowly. Their post-fix 13 GB/s is roughly what a *working* SHM/CE
path yields (cf. our 11-19), suggesting their real problem was the hang, and the
bandwidth they quote is just "SHM working normally," which we reach here via the
env knob without touching the kernel.

---

## 5. Recommendations (ranked)

### [ENV-ONLY — do now, no reboot] ★ highest value

1. **`NCCL_SHM_USE_CUDA_MEMCPY=1`** — measured **5× busbw** (2.2 → 11.2 at k=2/k=4,
   19.4 across-bridge), numerically correct, applies to PyTorch DDP/FSDP and any
   NCCL user on this box. Optionally pair with **`NCCL_BUFFSIZE=16777216`** (→
   11.1-11.5, marginal). This is the single highest-value change.
   - *Risk:* low. Real transport used in production; undocumented publicly, so
     treat as "verify then adopt" (we verified allclose). Costs one extra
     copy-engine stream. Re-check if the torch/NCCL pin is bumped.
   - *Where:* export in the shell/service that launches multi-GPU jobs, or set in
     the notebook env like the existing `NCCL_LOCAL_REGISTER` line
     (`chapter_computational-performance/multi-gpu-practice.md:55`,
     `multiple-gpus.md:55`). **Not applied by this investigation** (source edits
     out of scope).
2. Ignore the other knobs — `NCCL_ALGO/PROTO/BUFFSIZE/NTHREADS/SHM_DISABLE/
   P2P_DISABLE` do nothing or hurt here (§3b). `NCCL_P2P_USE_CUDA_MEMCPY` is a
   no-op (P2P is off).

### [REQUIRES ALEX — reboot / BIOS] lower priority here

3. **`iommu=pt`** kernel param (GRUB → reboot). Forum/docs predict it fixes NCCL
   *hangs* on PCIe boxes and reduces IOTLB overhead; may add a little to the
   copy-engine path under many small DMAs. **Will not by itself fix the 2.2 GB/s**
   (the CE path already hits 20 GB/s with IOMMU translating). Worth doing as
   hygiene / latent-hang insurance, but the env fix is what recovers the 5×.
   *Risk:* low-moderate; standard for GPU bare metal; needs reboot.
4. **BIOS ACS disable** — only relevant if pursuing *actual* P2P, which needs the
   open-kernel-module 4090 P2P patch (recompiled driver, high risk, big change).
   Not recommended; host-staged CE at 11-19 GB/s is the pragmatic ceiling for this
   hardware.
5. Confirm ACS state with `sudo lspci -vvv -s 80:01.1` etc. (read-only; §1).

---

## 6. Chapter-13 impact

The chapter's default-NCCL numbers are **correct and match this box** — its own
"NCCL's own busbw reads ~2 GB/s" (`fast-transformer.md:324`) equals my measured
**2.24**. The question is which claims move if the CE knob is adopted (and ch13
outputs re-captured with it set).

**Robust regardless of the fix (structural — do not change):**
- P2P disabled / GeForce segmentation / host-staged path
  (`hardware.md:349-366`). The knob does **not** enable P2P; it only speeds the
  host bounce.
- **"Flat from 2 to 4 GPUs"** (`hardware.md:366,589`) — confirmed in *both*
  regimes (default 2.24→2.17; CE 11.24→10.59). The per-GPU staging step is still
  the ceiling.
- Raw GPU-to-GPU copy **~20 GB/s** (`hardware.md:361`) and pinned H2D ~24 GB/s
  (`hardware.md:314`) — unchanged (my 19.9 / 23.7).
- The bandwidth-ladder pedagogy (PCIe vs NVLink two-plus orders of magnitude):
  robust — even 11-19 GB/s is ~100-160× below an NVL72's 1.8 TB/s.
- The predict-then-measure `t_comm = 2N/β` method (`eq_dp_cost`): robust; only β
  changes.

**Would change if the fix is adopted (numbers, not narrative):**
- `hardware.md:362-364` "a couple of GB/s of effective bus bandwidth" and
  `481`, `588-589` → would become ~11 GB/s (still ~2× below the 20 GB/s raw copy,
  a nice teaching contrast).
- `multi-gpu-practice.md:340` `beta = 4.5e9  # (13.5)` → the effective β roughly
  doubles-to-2.5×; busbw ~2 → ~11. Comm-time predictions drop ~5×; the **88%/82%
  weak-scaling** efficiencies (`:258`, `:570`, `:669`, `:748`) would edge *up*
  (comm cheaper), most visibly for the comm-bound transformer, least for
  compute-dense ResNet-18 (comm already hidden).
- `fast-transformer.md:321-324` "around five GB/s per device … NCCL busbw ~2 GB/s"
  → both numbers rise; the 76 MB/step allreduce gets ~5× cheaper, shifting the
  transformer's predicted k=2/k=4 gains upward.

**Recommended chapter treatment (for Alex, not applied):** keep the default-NCCL
numbers — they are honest and pedagogically central (consumer GeForce boxes are
communication-starved) — and **add a footnote/aside**: the default host-staged
NCCL number is a *transport-default artifact*, not a wire limit;
`NCCL_SHM_USE_CUDA_MEMCPY=1` recovers ~5× (kernel-load/store → copy engine),
which is itself the chapter's thesis in miniature — *a transfer runs at the speed
of its slowest stage, and the framework owns some of the stages*
(`hardware.md:326-328`). Full adoption (re-capture all ch13 multi-GPU outputs
with the env set + re-derive β + the efficiency prints) is the alternative if
Alex prefers the faster box as the baseline.

---

## Appendix — reproduce

```bash
cd …/scratchpad/nccl-probe            # allreduce_bench.py, bounce.py, run_matrix.sh, run_topo.sh
V=/home/smola/d2l-neu/.venv-pytorch/bin
# baseline vs fix, k=2 same-bridge:
CUDA_VISIBLE_DEVICES=0,1 PAYLOAD_MB=256 $V/torchrun --standalone --nproc_per_node=2 allreduce_bench.py
CUDA_VISIBLE_DEVICES=0,1 PAYLOAD_MB=256 NCCL_SHM_USE_CUDA_MEMCPY=1 \
  $V/torchrun --standalone --nproc_per_node=2 allreduce_bench.py
# copy-engine reference:
CUDA_VISIBLE_DEVICES=0,1 $V/python bounce.py
# transport line:
… NCCL_DEBUG=INFO … | grep 'via SHM'   # direct/direct (default) vs CE/direct (fix)
```
