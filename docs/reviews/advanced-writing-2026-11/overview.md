# Writing Review of the "Advanced" Part: Overview

Date: 2026-11-10. Source: branch `advanced-writing-review`, which is
`mdl-writing-review` (PR #160) plus these review files; the nine Advanced
chapter directories are as on that branch. Scope: chapters 9 to 17, all 67
sections, the nine chapter introductions and the eight further-reading files,
about 409,000 prose words. Question asked: the same as for Basics and for the
mathematical appendix, with more weight on technical correctness and currency,
focus, and whether the experiments show what the prose says. How clear, well
thought out, well taught and enjoyable is this part, measured against Cover
and Thomas, MacKay, and The Illustrated Transformer? No source text was
changed.

Method: 24 reviewers each read two or three sections in full (prose,
derivations, both code tabs, exercises, the stored PyTorch and JAX outputs),
recomputed worked numbers in NumPy where they could, and wrote one file per
section. Nine further readers wrote the chapter files from the section files,
reviewed the introductions and further-reading files themselves, and checked
the most serious cross-section claims in the sources and the output store.
This overview is written from the chapter files and the reviewers' reports; I
verified one finding myself (the time embedding, finding 2). Not checked
anywhere: nothing was executed in PyTorch or JAX (this laptop has neither),
so every statement about a training run is read from the store; the slides;
the rendered HTML; and statements attributed to papers and systems, which the
reviewers checked from memory and marked "unverified" in the section files.

Author decisions that carry over (see `../mdl-writing-2026-11/overview.md`):
confirm-by-sampling cells only where the experiment teaches something beyond
the theorem; the size-and-kind rule for duplication with the appendix, and
the main text must read without the appendix; long sections are split; one
ending per section and no question-then-cell formula; experiments are not
changed on the laptop.

## Verdict

The Advanced part is better written than Basics was before its pass and more
current than any textbook I can compare it with. In nine chapters the
reviewers found almost no wrong mathematics: no wrong equation in chapters 15,
16 and 17, a handful of wrong statements in the others, and several hundred
worked numbers recomputed correctly. The prose is plain, the mannerisms the
October review objected to are mostly gone, the exercises are the best in the
book, and every chapter has pages that are at the bar. Those pages have one
form in common, the same in all nine chapter files: a case small enough to
compute exactly, a number predicted before the cell prints it, and a candid
sentence about what the result does not show.

The part is not yet at the bar, and the reason is also the same in all nine
chapters. **The mathematics can be trusted; the reading of the training runs
cannot.** Where the book leaves its exact small cases and trains a network,
the prose claims more than the stored outputs show: a single-seed difference
the size of the run-to-run noise is read as a win, an arm that differs in
three things is credited to one, a number comes from seeds that no cell runs,
the two framework tabs disagree under one shared paragraph, and in about a
dozen places the stored output contradicts the sentence above it. This is the
part's one large defect and it is in every chapter (finding 1).

Four smaller defects recur. The chapters teach the mechanism and stop before
telling the reader what to choose (finding 3). From the middle of most
chapters on, teaching gives way to tours of recent papers, with results
quoted and not reproduced (finding 4). The central objects of the part are
not all taught where the book says they are: self-attention gets twelve
lines, the return from step $t$ is used for six sections before it is
defined, the KL divergence is never defined in the main text (finding 5).
And a third of the sections carry a harness, a second section, or material
built for a later chapter (finding 6).

Ranked: the reinforcement-learning chapter (14) and the diffusion chapter
(17) are the strongest, then optimization (9) and deep reinforcement learning
(15, for its mathematics). The attention chapter (10) has the best single
section of the part and a hole at its centre. Transformers (11), state space
models (12) and computational performance (13) are good chapters that do not
yet deliver what their introductions promise. The GAN chapter (16) is correct
and is organized around one paper's programme that its own notebooks do not
support.

## Scores

The reviewers were asked to calibrate hard. They did, and the result does not
discriminate much: explanation is 4 almost everywhere, thought and focus 3,
language 4. Read the table for its exceptions. Means are over the sections of
a chapter (introductions and further reading excluded); full tables are in
the chapter files.

| Chapter | Words | Expl. | Thought | Pedag. | Lang. | Enjoy. | Strongest | Weakest |
|---|---:|:-:|:-:|:-:|:-:|:-:|---|---|
| [9 Optimization](09-optimization.md) | 55,500 | 4.0 | 3.2 | 3.4 | 4.0 | 3.3 | batch-size, momentum | practice, gd |
| [10 Attention](10-attention.md) | 35,400 | 3.8 | 3.3 | 3.5 | 4.0 | 3.3 | what-attention-computes | multihead-attention |
| [11 Transformers](11-transformers.md) | 41,700 | 3.9 | 3.0 | 3.3 | 4.0 | 3.4 | moe | encoders-decoders, gpt |
| [12 State Space Models](12-state-space-models.md) | 29,200 | 3.7 | 3.2 | 3.2 | 3.8 | 3.3 | deltanet, ssm | hybrids, mamba |
| [13 Computational Performance](13-computational-performance.md) | 49,500 | 3.7 | 3.1 | 3.4 | 3.7 | 3.1 | performance-model | multi-gpu-practice |
| [14 Reinforcement Learning](14-reinforcement-learning.md) | 57,600 | 4.1 | 3.2 | 3.6 | 4.0 | 3.4 | value-iter | imitation, baselines |
| [15 Deep Reinforcement Learning](15-deep-reinforcement-learning.md) | 54,500 | 3.9 | 3.4 | 3.4 | 3.8 | 3.6 | regularized, model-based | rl-sequences |
| [16 GANs](16-gans.md) | 39,600 | 3.9 | 3.3 | 3.1 | 4.0 | 3.3 | gan | objectives |
| [17 Diffusion Models](17-diffusion-models.md) | 45,600 | 3.8 | 3.4 | 3.8 | 3.9 | 3.6 | discrete-diffusion, image-diffusion | design-space, gaussian-paths |

Two sections received a 5: `value-iter.md` for explanation, and
`what-attention-computes.md` and `discrete-diffusion.md` for pedagogy. Three
sections scored 3 on every dimension or nearly: `hybrids.md`,
`rl-sequences.md`, `multihead-attention.md`. No section is below 3 anywhere.
The chapter reviewer for 15 notes one calibration difference: sections with
major experiment findings received pedagogy 4 from one reviewer and 3 from
another for the same fault.

## The main findings

### 1. Training runs are read for more than they show

Every chapter file has a section on this, with finding IDs; chapter 15 has a
table of 27 rows and chapter 13 a table of 40 experiments. The forms:

- **The stored output contradicts the sentence.** In `lstm.md` the plain RNN
  recalls a symbol at 50 steps (100 in PyTorch) where the default LSTM and
  GRU are at chance, and the introduction and Summary say the opposite
  (LSTM-11, confirmed in both stores). In `scaling.md` the cost of
  transferring a learning rate across widths is 20% under the standard
  parameterization and 5% under muP in PyTorch and 2% and 6% in JAX; the
  prose states the PyTorch result as general (SCALE-9, confirmed). In
  `dcgan.md` the 2015 DCGAN has the best score of the three arms in both tabs
  at a tenth of the compute and the prose says it "lands close" (DCG-10,
  confirmed). In `ppo.md` the reuse experiment is quoted at "roughly 25,000"
  steps where the plot gives 31,000 to 38,000, and five other places repeat
  the number (PPO-14). Also: MG-12 and MG-13 (two-GPU timings), MEM-8,
  DSP-4 (log-SNR steps), MS-3 (capacity doubles against quadruples),
  SCORE-5, BL-11, TD-6, ED-13, CIRC-11.
- **A difference the size of the noise is read as a result.** Adam against
  SGD (0.05 nats on both models, ADAM-12), Muon against AdamW (0.03 against a
  stated spread of 0.02, MUON-4), the AdamW grid (ADAM-18), the MoE gain
  (0.02 nats), GELU against SwiGLU, Mamba against the minGRU (1.00 against
  0.97), DiT-L against the U-Net (1.41 against 1.50 inside the section's own
  quoted range, LAT-14), and from section 17.4 on, Fréchet-distance
  differences smaller than the gap between the two frameworks' runs of the
  same row. This contradicts the rule already in `CLAUDE.md` ("never build a
  conclusion on a difference smaller than the run-to-run noise").
- **Arms that differ in more than the factor credited.** The actor-critic
  head-to-head (AC-7), the PPO reuse arms, the delta-rule comparison in
  `hybrids.md` (HY-11), the recipe ablation in `design-space.md` (two to four
  factors per row), the sinusoidal-position arm that also rescales embeddings
  (POS-11, confirmed in code), the 25-Gaussians arms (CONV-14), the JAX bf16
  step that also changes the attention kernel (FT-8).
- **Numbers from runs no cell prints.** "200 seeds outside the notebook"
  (`baselines.md`), "twenty seeds", "forty groups", "150 runs"
  (`convergence.md`), about a dozen results in `hybrids.md`, the trace
  numbers in `multi-gpu-practice.md`. The chapter 12 reviewer's rule is the
  right one: a result is in a cell, in an exercise, or not in the book.
- **The mechanism given is not the mechanism.** The pessimism result in
  `offline-rl.md` comes from a clip at zero in `offline_q`, by a reviewer's
  reimplementation (OFF-7, OFF-8; not repeated by anyone else). The pre-norm
  argument is shown under Xavier weights at 32 blocks and then relied on for
  a 6-block model at $\sigma = 0.02$, where the collapse does not occur
  (TB-5, GPT-13). The two-GPU slowdown is the single Python launcher, which
  the cost model omits (MG-14). The low-rank cache experiment measures rank
  against 1,536 dimensions when the keys and values lie in at most 769
  (KV-13). The "Local Minima" run in `gd.md` is the edge of stability in one
  dimension (GD-6).
- **The environment is too small to show the mechanism.** Twin critics and a
  learned temperature do not change the return on Pendulum; Penn Treebank
  puts every model of chapter 12 between 2.96 and 3.36 with the LSTM first;
  momentum is never shown to help on data (MOM-9); behaviour cloning's
  distribution shift is a fifth of the loss on the lake (IL-1).
- **Two tabs, one paragraph.** Shared prose written from one framework's
  output and false for the other: chapters 9, 11, 12, 13 and 17 each have
  cases (KV-6 and KV-8 are the most visible: the default PyTorch tab does not
  show the memory-bound decoding the section argues for).

What can be done in prose now: make each sentence say what its cell prints,
lead with what the experiment does show robustly (the reviewers name it in
most cases), and state once per chapter what its notebooks can establish.
What needs the development machine: the redesigns listed per chapter under
"Experiments".

### 2. Correctness

Wrong or overstated statements, most serious first. Confidence and details
are in the chapter files under "Technical correctness and currency".

- **Code, chapter 17: the time embedding** (DEN-5, IMG-1). `time_embedding`
  uses frequencies from 1 down to $10^{-4}$ on $t \in [0, 1]$, and every
  caller passes unscaled $t$. I recomputed it: the 128 features have singular
  values 1, 0.098, 0.0036, so the network receives in effect $t$ and $t^2$.
  It feeds `TimeMLP`, `UNet`, `UNetFlowMap` and `DiT`, so every image number
  of the chapter depends on it. `few-step.md` says the low frequencies are
  deliberate for the Jacobian-vector product; no other section says so and no
  cell tests the choice. Decide this before any re-capture of the chapter.
- **Chapter 14.** Lai and Robbins' asymptotic bound applied at $t = 2000$,
  where the section's own figure contradicts it (EXP-1); "we can therefore
  multiply any score ... by such a quantity" is false as written (BL-3); the
  one-step reading of text generation needs $\gamma = 1$ (MDP-13); with the
  potential zeroed at holes as the text requires, stepping into a hole pays
  more than reaching the goal (MDP-12, VI-15); $V^\pi$ is called "the return"
  (VI-1).
- **Chapter 16.** The body says four times that the gradient penalty keeps a
  gradient on disjoint supports and the section's exercise 2(c) proves the
  opposite for a free critic (CONV-10, confirmed); the local-convergence
  theorem is cited without hypotheses or scope (CONV-7); "attracts only if
  eigenvalues have negative real parts" (CONV-5).
- **Chapter 9.** The step ratio under momentum comes from the double root,
  error like $t\rho^t$ (MOM-5); RMSProp "gets close ... where AdaGrad
  stalled" holds at step 20 and reverses by step 200 (ADAM-3); the muP rule
  for the output layer (SCALE-6, from memory); Muon's account of the output
  head (MUON-3) and the "RMS of $0.2\eta$" claim (MUON-6); the memory ledger
  stores bf16 weights where chapter 13 keeps fp32 (ADAM-21, confirmed). Two
  exercises do not work as stated (`gd.md` 1, `muon.md` 5).
- **Chapter 11.** SwiGLU described as squashing (SiLU is unbounded, TB-15);
  target padding (ED-10); `chap_nlp_pretrain`, now in the Attic, cited from
  two sections as the current home (SL-14, ED-7, confirmed; the right targets
  exist).
- **Chapter 13.** Epilogue fusion into matrix products presented as something
  the compilers do (COMP-7, about 85% sure it is not); the data-parallel cost
  model is wrong for more than two devices (MG-9); 165 TFLOP/s is derived and
  called a specification value (PM-1, HW-9).
- **Chapters 10, 12, 15, 17.** Smaller: the saturation claim for unscaled dot
  products under the book's initializer (SCORE-2), a wrong Parzen citation
  (QKV-4, confirmed in `d2l.bib`), Bahdanau's decoder state (SCORE-16);
  Mamba's rank and tensor count (MAMBA-17), "cannot grow without bound"
  (DN-4); the 2013 and 2015 DQN papers merged (DQN-1), the SAC entropy target
  off by $\log 2$ from reference code (SAC-7); "likelihood training of
  slightly noisy data" (DSP-6, about 85% sure), the reason given for the
  stop-gradient (FEW-1).

**Claims about papers and systems** are the part's unverified surface: the
reviewers flagged roughly a hundred (model configurations, benchmark numbers,
attributions, API defaults). They were checked from memory only. The
further-reading files and the survey subsections hold most of them. Someone
should check them against the sources once, after the cuts of finding 4 have
removed half.

### 3. The chapters stop before telling the reader what to choose

The part introduction promises decisions. Chapter 9 teaches every ingredient
of "learning rate, schedule and batch size" and no page assembles them
(PRAC-24). `encoders-decoders.md` does not say which wiring to use today
(ED-1). Chapter 17 has no procedure for a sampler and step count, a target
and weighting, or a backbone (SMP-9, DSP-15, LAT-18). Chapter 13's case study
does not close its account in milliseconds. Chapter 16 does not say what
GANs keep and why they were displaced. Chapter 12 has no cost table against
attention. Drafts for several of these pages are in the section files. Each
is a page, and each would change what the reader can do after the chapter
more than any other single edit.

### 4. Focus: surveys inside teaching sections

Every section of chapter 9 from Adam on ends with a tour of recent papers;
Muon has as many words as gradient descent and SGD together, 3,000 of them
variants retold. Chapter 11 has five survey passages ("Configurations of
Deployed Models", fine-grained MoE, register tokens). `adversarial-losses.md`
teaches for half its length and then lists ten systems. `rl-sequences.md`
names nine methods in 2,350 words. `hardware.md` mixes principles with a
device catalogue, while the lasting invariants are in its exercises.
`hybrids.md` writes up research in progress as a result. The remedy the
reviewers propose is the same each time: keep the principle and one worked
instance in the section, move the rest to Further Reading with a ranking, and
say there what is settled and what is a bet.

The GAN chapter is the large case. It is organized around one 2024 programme
(relativistic pairing with two gradient penalties). Its own analysis
qualifies the pairing loss, its notebooks establish one thing (the plain
backbone collapses without penalties), and the Wasserstein objective gets
about 90 words and no experiment, although the appendix says this chapter
trains a Wasserstein critic (`mdl-divergences-distances.md`, a false
sentence on the appendix side). The chapter reviewer recommends five or six
sections in place of seven, merging `objectives.md` and `relativistic.md`.

### 5. Central objects taught thinly or in the wrong place

- **Self-attention** (chapter 10): twelve lines and a shape check on a tensor
  of ones, though `transformer-block.md` and two later sections cite the
  chapter as its home. The scaled dot-product formula is never run on an
  input with a known answer; five cells validate by shape. The material for
  the repair exists in the chapter (the "Alice gave Bob" sentence, the letter
  demonstration, the two-record database) and is in the wrong places. Against
  The Illustrated Transformer the chapter is deeper and more rigorous
  everywhere else, and a worse first explanation of this one thing.
- **The GPT section** (chapter 11) never writes the loss or shows the shifted
  targets, has no recurrent baseline on the same bytes, and gives no cost of
  a run.
- **The return from step $t$** (chapter 14): used unnamed in five sections,
  first defined in the seventh, as $\hat G_t$.
- **KL divergence**: defined nowhere in the main text before `ppo.md` and
  `regularized.md` use it; chapter 14 has only the Bernoulli case under
  another name.
- **Results that sit only in exercises**: the DAgger guarantee, the
  variance-optimal baseline, TD(0), what a target network does and does not
  fix, the non-saturating loss as $2\,\mathrm{KL}(m\|p)$, the gate identity
  that leads from the GRU to Mamba, the reward-scale experiment that
  motivates SAC's learned temperature. The exercises are excellent and are
  carrying content the body needs.
- **The appendix as a prerequisite**: `gaussian-paths.md` rests its central
  claim on a continuity equation cited only to the appendix and uses SDE
  notation the main text never introduces (GP-1, GP-3). By the author's rule
  the chapter must read without it. Elsewhere pointers are missing in the
  other direction: `ssm.md` should cite `subsec_mdl-ode-driven-linear`,
  `muon.md` the Newton-Schulz analysis, `convergence.md` the linearization
  material, the policy-gradient sections the gradient estimators.

### 6. Structure: harnesses, double sections, order

- **Sections that are two or three.** `adam.md` is Adam, a 370-line test
  harness (TinyLM, FashionCNN) and AdamW; a dozen files already cite
  `sec_adamw` and `subsec_tinylm` as sections. The chapter reviewer
  recommends a "Two Test Models" section after `sgd.md`. Also:
  `kv-cache.md` (seam at "Sharing Keys and Values across Heads"), `ssm.md`
  (at "State Space Models"), `attention-at-scale.md` (before "Windowed and
  Sparse Attention"), `gaussian-paths.md` (before "Sampling by
  Integration"), `multi-gpu-practice.md`, and `convergence.md`, where the
  reviewers disagree.
- **Harness before result.** `lstm.md` is two-fifths shared harness with 290
  lines between the task and its outcome; `imitation.md` has 190 lines of
  policy-container plumbing before the first result; `sgd.md` 190 lines of
  helpers; `mdp.md` a 44-row symbol table and `gan.md` a 20-row one with
  errors.
- **Order.** Scaling laws before MoE (two reviewers independently); Muon
  before schedules; the chapter and part end mid-paragraph in a subsection
  about BERT (DISC-16).
- **Ownership across chapters.** Grouped-query attention is taught in
  chapters 10 and 11, which read its evidence differently (owner:
  `kv-cache.md`). Prefill and decode throughput is taught three times
  (chapters 11 and 13). The zero-mean lemma and reward-to-go are derived in
  two sections of chapter 14; reward shaping is told twice; the three limits
  of chapter 14 are restated at the head of chapter 15. Notation changes
  between chapters 14 and 15 ($R(\tau)$ and $r(\tau)$, $\alpha$ and
  $\eta_V$) and inside chapter 13 ($h$ and $\beta$ each mean two things).
- **Introductions** list what each section "asks"; they should give a route
  with a ranking. **Further-reading files** do not rank their entries.
  **"We can now" recaps** before the Summary survive in several chapters.
- **Stale records.** `CLAUDE.md` calls chapter 14 PyTorch-only and chapter 17
  a placeholder; neither is true. `docs/llm/language_models_part_plan.md`
  says the DPO derivation is already in `rl-sequences.md`; the loss appears
  only in an exercise.

## What is best and should set the model

The chapter files list the passages. The ones I would show first:

- `value-iter.md`: the equation of cell 14 with the unknown on both sides,
  the stopping rule, the anatomy of an optimal policy.
- `what-attention-computes.md`: train a model, find the circuit, then try
  four ways to be wrong about it.
- `sgd.md`, the noise ball; `momentum.md`, four settings with one
  deliberately bad; `batch-size.md`, the collapse onto one curve.
- `convergence.md`, the Dirac-GAN: circle, outward spiral, damped spring.
- `dcgan.md` and `image-diffusion.md`: a score read against a real-against-
  real floor, with ranges over seeds. This is the instrument the rest of the
  part needs for finding 1.
- `discrete-diffusion.md`, the two-token table; `few-step.md`, the flow map.
- `matrix-state.md`, two dimensions and a third key; `deltanet.md`, "read
  before writing".
- `performance-model.md` and `compilation.md`, the chain: a count predicts a
  factor any hardware reproduces.
- `ppo.md`, from the two-action example to the clipped objective;
  `regularized.md`, true return against the KL budget.
- The section openings of chapter 11, each producing the design from a
  concrete case.

## Decisions needed from the author

1. **The standard for a trained comparison.** Proposed: a claim needs a
   printed number, a stated run-to-run range, and a difference larger than
   it; otherwise the sentence says what was observed in one run. Seeds stay
   single per framework (the existing rule), so most gaps of finding 1 become
   weaker sentences, not more compute. Agree?
2. **The time embedding in chapter 17**: keep low frequencies everywhere and
   say why, or use the usual scaling outside `few-step.md`. Needs a test on
   the development machine first.
3. **Chapter 16**: keep the R3GAN programme as the spine, or reduce it to one
   section and restore a Wasserstein experiment; seven sections or five to
   six.
4. **Splits**: `adam.md` into three with a harness section (and where it
   goes), and the six other seams of finding 6.
5. **Surveys**: move the paper tours to ranked Further Reading (finding 4),
   and how much of Muon, `hybrids.md` and `hardware.md`'s catalogue stays.
6. **Chapter 12's conclusion**: say once that at this scale the cheaper
   models do about as well, or redesign the experiments until the ladder
   shows.
7. **Self-attention**: taught in chapter 10 on one sentence with numbers
   (the reviewers' recommendation), with chapter 11 owning the causal mask
   and grouped-query attention.

## Recommended order of work

1. Prose pass on the laptop, chapter by chapter, as for the appendix:
   correctness items of finding 2 that are prose; every results sentence
   made to say what its cell prints; one decision page per chapter (finding
   3); surveys cut to ranked Further Reading; definitions moved from
   exercises into the body (finding 5); one ending per section; split
   markers placed. Chapters 14 and 9 first (strongest, cheapest), then 10
   and 11 (the centre), then 17, 15, 13, 12, 16.
2. Development machine: the time-embedding test and the chapter 17
   re-capture; the experiment redesigns listed in each chapter file; the
   splits; the self-attention worked case and the GPT baseline, which need
   new cells.
3. One check of the paper and system claims against their sources.
