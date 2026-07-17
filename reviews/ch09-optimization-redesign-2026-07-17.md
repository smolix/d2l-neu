# Chapter 9 "Optimization Algorithms" — Redesign Proposal (2026-07-17)

Fable-authored review and notebook-lineup proposal for modernizing
`chapter_optimization/`. Inputs: a full inventory of the current chapter and of
the math-appendix optimization chapter (ch. 25, `chapter_mdl-optimization/`);
a survey of 2024–26 university curricula (Stanford CS336/CS231n, CMU
10-414/714, 11-785, 11-667, Berkeley C182, Princeton COS 514/597R, NYU, Toronto
CSC2541, MIT 6.7960, summer schools, and the ICML/NeurIPS tutorial circuit); a
survey of the 2020–26 research literature with adoption verdicts; and a survey
of the practitioner canon (Tuning Playbook, Muon/modded-nanogpt, Bernstein,
EleutherAI, Simo Ryu, CS336 Assignment 1, AlgoPerf). This document is the
input to the rewrite, not the rewrite itself.

## 1. Charter

- **Audience/bar:** teachable as-is in a CMU/Stanford/Berkeley course; a top-5
  chapter of the book.
- **Scope:** algorithms only. Distributed training, ZeRO/FSDP, parallelism live
  in ch. 11 (Computational Performance) and §29.6 (Training Systems). The only
  systems fact this chapter owns is optimizer-*state* arithmetic (bytes per
  parameter), because it is an algorithmic property.
- **Division of labor with ch. 25 (math appendix):** the appendix carries the
  proofs (descent lemma, condition-number law, momentum's √κ, Robbins–Monro,
  Ghadimi–Lan, AdaGrad/Adam derivations, bias-correction proof, Reddi
  counterexample/AMSGrad, convexity, duality, BFGS/trust regions). This chapter
  teaches the *methods, phenomena, and craft*, states results with intuition,
  and cross-references the appendix for every derivation. The appendix's own
  index already frames itself as the non-duplicative deep complement.
- **Coverage goal:** tools that serve general-purpose training now and set up
  the later image-model (ch. 16→19), LLM (ch. 17–18, 12), and beyond chapters:
  the AdamW recipe, schedules, batch-size scaling, hyperparameter transfer, and
  the matrix-preconditioned (Muon/Shampoo) family that image/text frontier
  training has begun adopting.

## 2. Verdict on the current chapter

Current lineup: intro, convexity, gd, sgd, minibatch-sgd, momentum, adagrad,
rmsprop, adadelta, adam (+Yogi), lr-scheduler. **Nothing newer than ~2019
appears anywhere.** Missing entirely: AdamW/decoupled weight decay (despite
`weight-decay.md` §2.7 forward-referencing it), warmup as a first-class topic,
WSD, batch-size scaling laws, muP, Lion, Shampoo/SOAP/Muon, edge of stability,
optimizer-state memory. Meanwhile ~40% of the chapter's length is convex
analysis and convergence proofs that ch. 25 now does better and more generally.

What is genuinely good and must survive:

| Asset | Where | Why keep |
|---|---|---|
| Ill-conditioned-valley demo (before/after momentum) | momentum.md | The single best teaching moment in the chapter |
| GFLOPS vectorization benchmark + GD/SGD/minibatch wall-clock race | minibatch-sgd.md | The most "code teaches" section; also defines the shared `train_ch11`/`get_data_ch11`/`Timer` harness |
| 1D/2D GD toys (LR too small/large, Newton-fails-nonconvex) | gd.md | Concrete, minimal; defines `train_2d`/`show_trace_2d` (`#@save`) |
| Fashion-MNIST CNN schedule harness (real model, real curves) | lr-scheduler.md | The chapter's best real-training experiment; extend, don't replace |
| "Open the black box" chapter framing | index.md | Correct framing; needs rewriting for the new split |

What goes:

- **convexity.md — retire as a section.** Ch. 25.3 (`mdl-convexity`) is a strict
  superset with better figures; an internal review (2026-06-09) already flagged
  the duplication. Salvage the two "why convexity still matters for DL"
  paragraphs (near-minimum convex behavior, SWA) into 9.1's discussion.
- **adadelta.md — cut to an exercise.** Zero 2026 syllabi teach it; zero
  adoption; it even carries a TF-specific lr=5.0 quirk as a maintenance
  liability.
- **adagrad.md / rmsprop.md — merge into the Adam notebook.** Every modern
  course compresses this genealogy to "two slides on the way to Adam"; the full
  two-derivation treatment lives in ch. 25.2.
- **Formal proofs in gd/sgd/momentum** (Newton quadratic convergence, convex
  SGD rate, eigenmode analysis) — replace with the statement + picture +
  `:numref:` into ch. 25.
- **Yogi implementation — cut to an exercise** (keep the one-paragraph "ε and
  the variance estimate can misbehave" lesson in 9.6's discussion).

## 3. What the world teaches and uses now (basis for the design)

Condensed from the three surveys; full source lists in §8.

- **Consensus curriculum 2026** (taught essentially everywhere): compressed
  classical ladder ending at **AdamW as the working default**; decoupled weight
  decay as a concept, plus its LLM-scale reinterpretation (training-loss
  effect via effective-LR/rotational equilibrium, not anti-overfit); linear
  **warmup** + cosine as default schedule; **gradient clipping** as standard
  equipment; **batch size** as a first-class topic (linear-scaling rule,
  critical batch size); **optimizer-state memory accounting** taught next to
  the update rules (CS336's graded `adamw_accounting` problem). Dropped
  everywhere: Adadelta, L-BFGS-as-advice, convexity-heavy treatments, untuned
  comparison tables.
- **The differentiator tier** (CS336 + the 2025–26 ICML/NeurIPS tutorial
  circuit + MIT 6.7960 L7 only; in *no* public Stanford/CMU/Berkeley deck yet):
  muP/hyperparameter transfer; WSD and schedule-free; the
  **steepest-descent-under-a-norm** unification (SGD ↔ ℓ₂, sign/Adam ↔ ℓ∞,
  Shampoo/Muon ↔ spectral — Bernstein & Newhouse); Muon itself.
- **Adoption reality (mid-2026):** AdamW(0.9, 0.95), wd 0.1 (not on
  embeddings/norms), clip 1.0, ~2k-step warmup, cosine-or-WSD, early batch
  ramp is the disclosed frontier recipe (Llama 3, DeepSeek-V3, OLMo 2, Qwen).
  The one production-proven challenger: **Muon for hidden 2-D matrices +
  AdamW for the rest** (Kimi K2/MuonClip 15.5T tokens zero spikes; GLM-4.5;
  DeepSeek-V4; Megatron/DeepSpeed support; `torch.optim.Muon` in core PyTorch
  since 2.9). Distributed Shampoo won AlgoPerf's external-tuning track;
  Schedule-Free AdamW won self-tuning.
- **The honesty layer the chapter must carry:** most "beats AdamW" claims
  shrink under fair tuning ("Fantastic Pretraining Optimizers", arXiv:2509.02046
  — from the group that wrote Sophia; Sophia itself is the canonical
  did-not-replicate tale); benchmark verdicts are protocol-dependent (AlgoPerf
  vs arXiv:2509.01440 vs speedruns); macro "algorithmic progress" numbers do
  not isolate the optimizer. This aligns exactly with the repo's
  results-precision policy.
- **Summer schools** (MLSS/OxML/EEML/Indaba, 2023–26): essentially no modern
  optimizer teaching — a generation behind. The gap this chapter can fill is
  real.

## 4. The chapter's organizing idea

One sentence the whole chapter hangs on:

> **An optimizer is three decisions: a descent direction (which norm measures
> "steep"?), a step size over time (schedule), and a way of living with noise
> (batching, momentum, averaging).**

9.1 introduces the frame; 9.2–9.6 fill in the classical ladder as instances;
9.7–9.8 are the step-size/decay column; 9.9 pays the frame off (change the
norm → sign/Adam and spectral/Muon fall out); 9.10–9.11 are what happens at
scale; 9.12 is the craft. Two recurring pictures: the **ill-conditioned valley**
(classical) and its modern upgrade, the **river valley** (Wen et al. 2024),
which explains warmup/stable/decay in a single figure and is reused in 9.1,
9.8, and 9.11.

## 5. Proposed notebook lineup (12 sections)

File-level plan, updated 2026-07-17 after Alex's review: **PyTorch + JAX
only** (the Advanced-part framework policy; see §6), short titles, Yogi cut
confirmed, the scaling section split in two, and optimizer-state memory
accounting moved into the AdamW section. "Keeps label" = no inbound-reference
breakage (inbound refs audited; see §6). Every section keeps/gets a
`<!-- slides -->` deck and follows the figure house style (new generator
`tools/gen_opt_figures.py`; the name avoids colliding with the appendix's mdl
generators).

### Shared testbeds

Three testbeds, escalating in realism, each inside the ≤7.5 GB / ~10-min
envelope:

1. **Airfoil regression** (existing `get_data_ch11`/`train_ch11` harness) —
   the scratch-implementation workhorse: every from-scratch optimizer trains
   here first, seconds per run.
2. **Fashion-MNIST CNN** (existing lr-scheduler harness) — schedules,
   averaging, and the "CNN side" of every CNN-vs-LM contrast.
3. **`TinyLM` — a ~0.7M-parameter char-level transformer on the ch. 8 Time
   Machine corpus (NEW).** Rationale and forward-reference plan:
   - *Why:* the phenomena that define modern optimization — the Adam-vs-SGD
     gap, Muon's advantage, heavy-tailed token noise, clipping — barely show
     on a small CNN. They show on a language model. Without one, the chapter
     demonstrates 2015 phenomena with 2026 vocabulary.
   - *Why it is pedagogically sound:* the reader arrives at ch. 9 having just
     trained character LMs *on this exact corpus* with RNNs in ch. 8 —
     task, data pipeline (`d2l.load_data_time_machine`), vocabulary, and
     perplexity are all old friends. Only the architecture is new, and it is
     used strictly as a black box.
   - *How the forward reference is handled:* one compact, visible model cell
     (~35 lines per framework: learned char + position embeddings, two
     pre-norm blocks built from the framework's attention primitive
     (`F.scaled_dot_product_attention` / `nnx.MultiHeadAttention`) plus a
     two-layer MLP, output head), introduced with an explicit callout: "this
     is the subject of :numref:`chap_attention`; for this chapter it is just
     a differentiable function with a particular *census of parameters*."
     Immediately followed by the **parameter-census cell** — a table of every
     tensor (embeddings, 2-D hidden matrices, 1-D norms/biases) with shapes
     and counts. The census, not the mechanism, is what the chapter teaches
     on: it drives the decay-exclusion rules (9.7), the memory accounting
     (9.7), the Muon matrix/non-matrix split (9.9), and muP's per-layer
     scalings (9.11). Ch. 10 then opens the box and can back-reference
     ("you already trained one of these").
   - *Mechanics:* d≈128, 2 heads, 2 blocks, seq 64–128 → ~0.5–1 M params;
     ~3k steps ≈ 1–2 min/run on the build GPUs. Defined once (in 9.6),
     `#@save`d as `d2l.TinyLM` so 9.9/9.10/9.12 reuse it. Used in: 9.6
     (Adam-vs-SGD gap, real on the LM, near-nil on the CNN), 9.9 (Muon vs
     AdamW at matched tuning), 9.10 (gradient-noise scale), 9.12 (clipping).

### 9.1 Landscapes — `optimization-intro.md` (revise; keeps `sec_optimization-intro`)
Goal: what makes deep-net optimization hard and what "solving" it means.
Keep: risk-vs-empirical-risk, local minima/saddle/vanishing-gradient toys.
Add: condition number as the first villain (valley picture); noise as the
second; the three-decision frame; a one-paragraph edge-of-stability preview
("training lives at the stability boundary, not in the descent-lemma regime",
:numref: into ch. 25.1's EoS experiment); salvaged "why convexity still
matters" paragraphs closing with a pointer to ch. 25.3.
Figures: river-valley 3-D/contour (generated, reused later); existing toys.

### 9.2 Gradient Descent — `gd.md` (trim; keeps `sec_gd`, `train_2d`/`show_trace_2d` #@save)
Keep all 1D/2D demos. Newton shrinks to "the ideal preconditioner and why it
does not scale" (statement + failure demo), pointing to ch. 25.1 for the
quadratic-convergence proof, BFGS, trust regions. Preconditioning paragraph
stays — it is the seed for 9.6 (diagonal) and 9.9 (matrix).

### 9.3 Stochastic Gradient Descent — `sgd.md` (trim; keeps `sec_sgd`)
Keep noisy-gradient demos and the why-decay-LR motivation; replace the ~80-line
convex rate proof with the stated noise-ball picture ("constant LR → plateau
at a noise floor ∝ η; decay → convergence") and :numref: to ch. 25.1/25.2.
Add: one cell measuring gradient variance vs batch size on a real small net —
the empirical hook for both 9.4 and 9.10.

### 9.4 Minibatches — `minibatch-sgd.md` (light revision; keeps `sec_minibatch_sgd`, harness #@saves; TF/MXNet tabs dropped)
Keep: GFLOPS benchmark (refresh the hardware numbers to a 2026 CPU/GPU),
`Timer`/`get_data_ch11`/`train_ch11`/`train_concise_ch11`, and the
GD/SGD/minibatch race. Add: a closing discussion distinguishing the two
reasons for batching (hardware efficiency vs gradient variance) and a
forward-pointer to 9.10 for how large is too large (critical batch size).
One-sentence nod to gradient accumulation as the systems-side equivalence
(:numref: ch. 11).

### 9.5 Momentum — `momentum.md` (revise; keeps `sec_momentum`)
Keep the valley demo and leaky-average development. Add Nesterov momentum
(both `nesterov=True` in the concise cells and the two-line look-ahead change
in the scratch cell) — a real gap today. State the √κ acceleration law and the
β timescale (1/(1−β) steps of memory); proofs → ch. 25.1. Goh's Distill
article stays cited; reproduce its critical-damping payoff as a generated
figure (rate vs β).

### 9.6 Adam — `adam.md` (absorbs adagrad/rmsprop; keeps `sec_adam`)
The compressed genealogy, from-scratch throughout, trained on the shared
harness: AdaGrad (sparse/per-coordinate motivation, the decaying-LR defect) →
RMSProp (EMA fix) → Adam (momentum + second moment + bias correction, stated;
proof → ch. 25.2). Close with two discussions: (a) *why Adam wins on
transformers* — the heterogeneity story with a real mini-experiment: train the
ch. 8 tiny char-LM with tuned SGD-momentum vs tuned AdamW and show the gap,
then the same on the Fashion-MNIST CNN where the gap nearly vanishes
(Kunstner's token-imbalance explanation in prose); (b) when Adam's variance
estimate misbehaves (ε, one paragraph; Yogi and AMSGrad as exercises,
Reddi counterexample → ch. 25.2). Retires `sec_adagrad`/`sec_rmsprop`/
`sec_adadelta` (no inbound refs outside this chapter + appendix discussions;
see §6).

### 9.7 AdamW — `adamw.md` (NEW)
The single biggest gap. Content: L2-through-Adam ≠ weight decay (the exact
two-line algebra; the 100×-disparity demo stays in ch. 25.2, referenced);
implement AdamW from scratch (the CS336 exercise, done in the book's style);
a small (LR × WD) sweep heatmap showing decoupling makes the two knobs
separately tunable; the modern reinterpretation — at scale weight decay is
effective-LR control (rotational equilibrium, qualitative) and interacts with
the schedule, not a regularizer (cite Andriushchenko; CS336 L3); practice:
what not to decay (embeddings/norms/biases — OLMo 2's reason), the
τ = B/(ηλD) timescale rule of thumb. **Also home of optimizer-state memory
accounting** (moved here from the old 9.10 plan): AdamW's two moments →
the worked bytes-per-parameter table (fp32/bf16 master-weight patterns,
≈12–20 B/param; Adafactor/8-bit states in one paragraph; sharding = ch. 11 /
§29.6 pointer), CS336-style accounting exercise on `TinyLM`'s census.
Back-references `sec_weight_decay` (§2.7), finally honoring its forward
promise.

### 9.8 Schedules — `lr-scheduler.md` (extend; keeps `sec_scheduler` + CNN harness)
Keep the schedule family and the CNN experiment. Add: **warmup** promoted to
a first-class subsection (the sharpness-reduction why, in one figure +
:numref: ch. 25.2; the "no warmup → divergence/NaN" demo); **WSD** — implement,
train the same CNN under cosine vs WSD, and demonstrate the killer property:
branch a decay off the stable plateau checkpoint and match the cosine run
without committing to a horizon up front; the **river-valley** figure as the
explanation of the WSD loss cliff; a frontier note on linear-to-zero and
schedule-free (exercise: run the schedule-free update on a quadratic and watch
the averaged iterate track an implicit decay); honest note that WSD-vs-cosine
is not settled (GLM-4.5's ablation went the other way).

### 9.9 Muon — `muon.md` (NEW; flagship)
The payoff of the chapter frame, and material no other textbook has:
- Steepest descent depends on the norm: under ℓ₂ → SGD; under ℓ∞ → sign
  descent, and Adam-without-EMA is sign descent (connects backward to 9.6);
  under the **spectral norm**, for a matrix parameter, the steepest direction
  is the orthogonalized gradient. Table: {norm, closed-form step, which
  tensors}. One generated figure: three norm balls with their steepest
  directions on the same gradient.
- **Newton–Schulz demo** (pure matmuls, works in all four frameworks): random
  matrix → 5 iterations → singular values collapse to 1; plot the spectrum
  per iteration.
- **Muon from scratch** (~15 lines): momentum → orthogonalize → step; apply to
  hidden matrices only, AdamW for embeddings/head/scalars; train the tiny
  char-LM and the CNN against the 9.6 AdamW baselines at matched tuning.
  Expect (and state only qualitatively, per the precision policy) a modest
  but visible improvement on the LM.
- Shampoo/SOAP as the Kronecker branch of the same family (prose + one
  equation; K-FAC ancestry one paragraph; derivations → ch. 25.2's
  preconditioning ladder).
- The adoption story (Kimi K2 zero-spike 15.5T tokens, GLM-4.5, DeepSeek-V4,
  `torch.optim.Muon`) **and** the honesty box: fair-tuning shrinkage
  (arXiv:2509.02046 vs the Hyperball rebuttal), AlgoPerf protocol lessons,
  Sophia as the cautionary tale. Lion appears here as the sign-family branch
  (exercise: implement Lion in 6 lines and compare).
- The modded-nanogpt speedrun timeline (45 min → ~1.3 min, annotated) as a
  generated figure — evidence culture, not hype.

### 9.10 Batch Size — `batch-size.md` (NEW)
The statistics of batching (9.4 owns the mechanics/hardware side):
- Measure the **gradient-noise scale** on the CNN and on `TinyLM`; show
  steps-to-target vs batch size flattening past the critical batch size —
  the two-experiment core of the section.
- The LR-vs-batch rules (linear for SGD, √ for adaptive) verified with a
  small sweep; where they break.
- Batch ramping as frontier practice (noise scale grows during training —
  Ai2's batch-size-warmup result), one cited paragraph.
- Interaction with schedules (9.8) and a note that Muon tolerates larger
  batches (9.9), forward pointer to data-parallel scaling in ch. 11.

### 9.11 Scaling Up — `scaling.md` (NEW)
Making small-scale tuning survive scale — the bridge to the LLM chapters:
- The problem, demonstrated: sweep LR on MLPs at widths 128→1024 under
  standard parametrization; the optimum drifts.
- **muP**: the per-layer scaling rules stated (init ∝ 1/width, hidden LR ∝
  1/width, embeddings apart); the **coordinate-check** experiment (activation
  scale vs width: SP diverges, muP flat); the LR-transfer sweep repeated
  under muP — the optimum stops moving (~24 tiny runs, minutes total).
- The spectral-condition view in one paragraph, tying muP back to 9.9's
  norm story (Muon's built-in transfer).
- What labs actually do: muP at Cerebras, scaling-law fits at DeepSeek/Qwen,
  MetaP at Meta, Moonshot's tried-and-rejected muP-for-Muon; the
  weight-decay-vs-muP debate flagged as live.

### 9.12 Practice — `practice.md` (NEW; closes the chapter)
The craft section, in the Tuning Playbook's spirit:
- The consensus recipe table with primary-source citations (Llama 3,
  DeepSeek-V3, OLMo 2, Kimi K2, nanochat) — presented as "disclosed practice",
  not gospel.
- **Gradient clipping** demo (global-norm; a high-LR LM run with vs without
  clipping), plus the stability kit in prose (z-loss, QK-norm, MuonClip's
  QK-clip; loss-spike war stories: PaLM rewind-and-skip, OPT logbook).
- **Weight averaging**: SWA/EMA/LAWA in one cell on the CNN (eval-accuracy
  bump for free); when EMA is mandatory (diffusion — forward pointer to
  ch. 16) vs checkpoint averaging (LLMs).
- **How to tune**: scientific/nuisance/fixed hyperparameter vocabulary; tiered
  budgets; "trying several optimizers at defaults ≈ heavily tuning one";
  experiment-log discipline. Exercises mirror CS336 A1: LR sweep to a target
  loss, batch-size sweep with re-tuning, an accounting problem, an
  edge-of-stability sweep.
- Closing: what we deliberately did not teach (SAM — vision-niche with a 2×
  tax; variance reduction — ch. 25.2; LARS/LAMB — superseded) and where the
  systems story continues (ch. 11, §29.6).

### index.md (rewrite)
New chapter intro stating the three-decision frame, the ladder, and the
division of labor with ch. 25 ("every proof we owe you lives there"), plus a
Resources block (Tuning Playbook, Bernstein's anthology + "Deriving Muon",
Goh's Distill momentum, CS336 A1, AlgoPerf, ch. 25's own resource list).

## 6. Cross-cutting engineering notes

- **Labels.** Kept: `sec_optimization-intro`, `sec_gd`, `sec_sgd`,
  `sec_minibatch_sgd`, `sec_momentum`, `sec_adam`, `sec_scheduler`,
  `chap_optimization` (all inbound refs stay valid, incl. `d2l/*.py`
  docstrings). Retired: `sec_convexity`, `sec_adagrad`, `sec_rmsprop`,
  `sec_adadelta` — inbound only from this chapter and from
  `chapter_mdl-optimization/*` "Discussions" paragraphs, which need a
  ~10-line cross-reference cleanup in the same PR (point them at 9.1/9.6 or
  ch. 25.3).
- **d2l library.** `#@save` set unchanged (`train_2d`, `show_trace_2d`,
  `Timer`, `get_data_ch11`, `train_ch11`, `train_concise_ch11`). Candidate new
  `#@save`: `newton_schulz` / a minimal `Muon` scratch class if later chapters
  want it (decide at implementation; nothing downstream needs it yet).
- **Frameworks — PyTorch + JAX only (Advanced-part policy, per Alex
  2026-07-17).** The Basics part and math appendix keep all four frameworks;
  the Advanced part (ch. 9–16) carries PyTorch and JAX only, and this rewrite
  drops the chapter's TF/MXNet tabs outright. Mechanically this is already
  supported — the RL chapter ships PyTorch-only today (framework coverage
  follows per-cell tabs; `outputs/` simply has no tf/mxnet trees for it).
  Concise cells use native optimizers where they exist (`torch.optim.Muon`
  ≥2.9, Optax AdamW/schedules) and keep the scratch path as the shared spine.
  **Still pin initializations explicitly in every cross-framework
  comparison** — Flax NNX defaults to LeCun init vs PyTorch's Kaiming; that
  trap cost a week of confusion in ch. 7 (see 2026-07 efficient-convnets fix).
  Side effect: the chapter's capture cost halves. Cleanup of TF/MXNet tabs in
  the *other* Advanced chapters (10, 11, 13, 15) is separate work — see open
  question in §9.
- **Compute budget.** Every notebook stays within the ≤7.5 GB / ~10-min
  envelope: testbeds are Airfoil (scratch demos), the lr-scheduler
  Fashion-MNIST CNN, and a tiny char-LM reusing ch. 8's time-machine data
  (~1–2 min/run). Single seeded runs, qualitative comparisons only, per the
  results-precision policy. Comparisons at matched tuning (small documented
  sweeps), never defaults-vs-tuned.
- **Figures.** New generator `tools/gen_opt_figures.py` (house style, via the
  `mdl-figure` skill): river-valley, norm balls, schedule shapes,
  critical-damping rate-vs-β, speedrun timeline, state-memory bars. Copies to
  `img/auto/` for any deck that `@fig:`s them.
- **Slides.** Every rewritten/new section gets a fresh deck; kickers self-heal.
- **PDF/HTML.** No numbering impact (chapter stays 9 with 11 sections).

## 7. Suggested phasing

1. **Phase 1 — the gap-closers (highest value, least risk):** 9.6 merge,
   9.7 AdamW, 9.8 schedule extension, trims in 9.2/9.3, retire
   convexity/adadelta + appendix cross-ref cleanup.
2. **Phase 2 — the differentiators:** 9.9 Muon, 9.10 Batch Size,
   9.11 Scaling Up, 9.1 reframe, `TinyLM` testbed.
3. **Phase 3 — the craft capstone:** 9.12 Practice, index rewrite, slides
   pass, PyTorch+JAX capture + `make refresh-stale`.

Each phase leaves the book green and shippable.

## 8. Key sources (curated)

**Structural templates:** CS336 Assignment 1 (stanford-cs336/assignment1-basics);
ICML 2025 tutorial "Training Neural Networks at Any Scale" + companion
arXiv:2511.11163; Princeton COS 514 optimization + schedules chapters; Grosse
CSC2541 notes; MIT 6.7960 Lecture 7 (Bernstein).
**Unification:** Bernstein & Newhouse, "Old Optimizer, New Norm"
(arXiv:2409.20325); "Modular Duality" (arXiv:2410.21265); "Deriving Muon"
(jeremybernste.in/writing/deriving-muon); spectral condition
(arXiv:2310.17813).
**Methods:** AdamW (arXiv:1711.05101); Muon (kellerjordan.github.io/posts/muon,
"Muon is Scalable" arXiv:2502.16982, MuonClip/Kimi K2 arXiv:2507.20534);
Shampoo (arXiv:1802.09568, distributed arXiv:2309.06497); SOAP
(arXiv:2409.11321); Lion (arXiv:2302.06675); Schedule-Free
(arXiv:2405.15682); WSD (MiniCPM arXiv:2404.06395; Hägele arXiv:2405.18392);
river valley (arXiv:2410.05192); warmup (arXiv:2406.09405).
**Scale:** muP (arXiv:2203.03466; EleutherAI practitioner's guide);
critical batch size (arXiv:1812.06162; data-dependence arXiv:2410.21676;
Ai2 batch-size warmup arXiv:2505.23971); rotational equilibrium
(arXiv:2305.17212); WD-vs-muP (arXiv:2510.19093).
**Honesty:** "Fantastic Pretraining Optimizers" (arXiv:2509.02046) +
Hyperball amendment (arXiv:2606.16899); AlgoPerf (arXiv:2306.07179; results
arXiv:2502.15015); "Descending through a Crowded Valley" (arXiv:2007.01547);
"No Train No Gain" (arXiv:2307.06440); benchmark plurality (arXiv:2509.01440).
**Why-Adam theory:** Kunstner (arXiv:2304.13960, arXiv:2402.19449); Hessian
heterogeneity (arXiv:2402.16788); edge of stability (arXiv:2103.00065);
heavy tails + clipping (arXiv:1912.03194).
**Craft:** Google Tuning Playbook (github.com/google-research/tuning_playbook);
Karpathy recipe + nanochat; modded-nanogpt; OLMo 2 (arXiv:2501.00656) and the
frontier-report recipe table in the research survey.

## 9. Decisions from Alex's review (2026-07-17) and remaining opens

Resolved:
1. **Frameworks:** the Advanced part carries PyTorch + JAX only; this chapter
   drops TF/MXNet outright (Basics and the math appendix keep all four).
2. **Cuts approved**, including Yogi → exercise.
3. **`TinyLM` testbed:** concrete plan in §5 (black-box model cell +
   parameter-census as the teaching object), presented for sign-off; forward
   references acceptable where understandable (per Alex).
4. **Short titles adopted:** Landscapes · Gradient Descent · Stochastic
   Gradient Descent · Minibatches · Momentum · Adam · AdamW · Schedules ·
   Muon · Batch Size · Scaling Up · Practice.
5. **Scaling section split** into 9.10 Batch Size and 9.11 Scaling Up;
   optimizer-state memory accounting moved into 9.7 AdamW.

Remaining open:
1. **TF/MXNet cleanup in the rest of the Advanced part** (10 Attention,
   11 Computational Performance, 13 SSM, 15 GANs): bulk strip now vs
   as-touched. Ch. 11 is the hard case — hybridize/async/parameterserver are
   framework-specific by construction and partly MXNet-flavored, so there it
   is closer to a rewrite than a strip.
