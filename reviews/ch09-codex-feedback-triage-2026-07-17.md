# Triage of `optimization_feedback.md` (Codex review of ch. 9)

Date: 2026-07-17. Method: three independent verification passes (one per third of
the chapter) checked every review claim against the working-tree sources —
quoting the actual text/code, adjudicating the review's own math, and overlaying
house policy (CLAUDE.md results-precision rule: single seeded runs, qualitative
ranges; book-wide train/test convention; PT/JAX tab format; epochs OK in the
classroom testbed).

**Bottom line.** The review is valuable but must be applied selectively:
- ~1/3 is **right and actionable** — including 10 of its 12 P0 rows (severity
  varies; several are one-phrase fixes) and a dozen genuine overclaims that the
  house precision policy independently demands we soften.
- ~1/3 is **misread** — it demands corrections the text already makes, or gets
  our code wrong (it claims "one draw per size" where the code averages 200
  draws; asks for a log scale a plot already has; calls the repo's standard
  `%load_ext d2lbook.tab` source markup a leaked artifact — it never renders;
  claims CNN optimizer selection uses the test set where the code selects on
  training loss).
- ~1/3 is **policy-conflicted or design preference** — multi-seed sweeps with
  error bars/CIs everywhere (violates the single-seeded-run policy; the
  compute belongs to readers), book-wide train/val/test restructuring, NumPy-
  instead-of-tabs, step/token axes (the review itself concedes epochs are fine
  here), and the core/advanced chapter re-architecture.

## 1. P0 table, row by row

| # | Review P0 | Verdict | Notes |
|---|---|---|---|
| 1 | Muon fixed-ball steepest descent "recovers GD" | **REAL, med** | Our own table/figure show `−ηg/‖g‖₂` while prose/table label it "SGD"/"gradient descent" (muon.md:62-67, 148, 154, 901; index.md framing). Fixed ball ⇒ *normalized* GD. Fix the label or use the regularized model `⟨g,d⟩+‖d‖²/2η`. |
| 2 | scaling.md single-example sign factorization used on batches | **REAL, low-med** | Derivation is rank-1 (scaling.md:219-233); "single example" never said; experiments use b=512. Scope it as single-example intuition confirmed by the coordinate check. |
| 3 | muP output **bias** divided by width | **REAL, med** | Code divides the whole output incl. bias (`super().forward(X)/self.m`, scaling.md:296-297) while the table says biases unchanged (267-271). Real code/table inconsistency (bias is zero-init, so shown results barely affected; the coordinate check cannot catch it). Code fix ⇒ re-execution. |
| 4 | Adam SDE batch rule presented as LR-only | **REAL, med** | batch-size.md:528-550 attributes `η∝√b` to Malladi's SDE analysis without the β₁,β₂,ε transforms + validity range. Experiments only swept η, so results stand; the attribution is incomplete. |
| 5 | noise-ball radius ∝ η vs squared radius ∝ η | **REAL, low-med** | Three spots say "radius ∝ η" (optimization-intro.md:305, 531; sgd.md:317-318 summary); sgd.md:127-131 and batch-size.md:537-539 already say it correctly (squared radius). One-word fixes. |
| 6 | Robbins–Monro α=1/2 | **MISREAD** | sgd.md:200-204 already states α∈(1/2,1] for the classical conditions, calls α=1/2 "edge of the window," and gives the finite-horizon averaged O(1/√T) framing — exactly the demanded correction. |
| 7 | AdamW coupled-penalty decomposition not exact | **REAL, low-med** | eq_coupled-decay (adamw.md:66-85) is a schematic split presented without a "schematic" qualifier; exact treatment is in the appendix we cite. Add "schematically / to first order." |
| 8 | AdamW "restores intended ℓ2 semantics" | **MISREAD** | We claim it restores the *multiplicative-decay* (1−ηλ) semantics (correct), and adamw.md:421-460 already distinguishes objective-penalty vs decay vs prior/dynamics with the appendix cross-ref. Intro verb could be tightened (optional). |
| 9 | Newton–Schulz: zero singular values | **REAL, low** | muon.md:176-178 says "every singular value starts in (0,1]"; zeros are fixed points of the quintic. Say "all *nonzero* singular values"; note rank deficiency persists. |
| 10 | Muon scratch (classical) vs library (Nesterov default) | **REAL, med** | Scratch is classical momentum; `torch.optim.Muon` defaults `nesterov=True` (verified, torch 2.11.0) and our calls don't pin it (muon.md:695-697, 713-715), while prose says the library "mirrors our scratch version." Pin `nesterov` or state the difference. Code fix ⇒ re-execution. |
| 11 | Practice: "Adam caps every coordinate near η" | **REAL, low-med** | practice.md:227-233. `|m̂/√v̂|` can transiently reach ≈(1−β₁)/√(1−β₂)≈3.2 at (0.9,0.999). The *conclusion* (clipping won't rescue a globally-too-large Adam LR) is fine and even conceded by the review; soften the absolute. |
| 12 | Practice: EMA + BatchNorm "self-consistent" | **REAL, low-med** | practice.md:344-356 EMAs BN running stats and calls the result self-consistent; it's approximately consistent (common heuristic). Soften, or mention `update_bn`-style recomputation (the checkpoint-averaging footnote already recommends it). |

## 2. Other real issues (beyond P0)

Prose-only (no re-execution):
- **McCandlish/GPT-3 attribution** — batch-size.md:273-274 + slide 830 credit "measurements on GPT-3-scale models" to a 2018 paper that predates GPT-3. Fix wording/citation. (low-med)
- **Overlapping windows vs iid claim** — batch-size.md:104-107 says minibatches match the iid theory; `d2l.TimeMachine` uses a stride-1 window (adjacent examples overlap 63/64, verified). Add the correlated-sequences caveat; mention token vs sequence batch. (low-med)
- **"every gradient … never computed exactly"** — index.md:14-15 (+ optimization-intro.md:23-30); contradicted by the chapter's own full-batch GD demo (minibatch-sgd.md:467-478). Soften. (low-med)
- **"only noise … for free / at no charge" saddle escape** — optimization-intro.md:129, 311-313, slides 465/537. Deterministic GD avoids strict saddles a.s.; barrier-crossing isn't free. Qualify. (low-med)
- **ε placement "cosmetic"** — adam.md:284-288, in tension with the chapter's own η/ε-ceiling argument (866-872), which depends on the placement. Scope to "at this demo's scales." (low-med)
- **EMA "bias correction" analogy** — practice.md:439-445 conflates window-fill lag (EMA weights sum to 1) with Adam's zero-init moment deficit. The prescribed fix (warm up decay/start late) is right; fix the causal story. (low-med)
- **TinyLM "nearly balanced" character vocabulary** — adam.md:843-846; character frequencies are strongly skewed. Real point is "no starved rare tokens (unlike word-level)." (low)
- **muP coordinate check overclaims** — scaling.md:424-426 "flat or falling means stable … catches essentially every muP bug"; falling updates can mean frozen feature learning, and a raw-activation check misses it. Soften + one clause on Δh. (low-med)
- **scaling.md "by construction" vs muon.md's own caveat** — scaling.md:483-489 says Muon's RMS-match scale satisfies the spectral condition by construction; muon.md:284-288 correctly distinguishes RMS-match from √(fan-out/fan-in). Align. (low)
- **Conv matricization rule is PyTorch-only** — "one row per output channel" (muon.md:291-292, 536-537) doesn't describe the JAX/Flax HWIO flatten (transpose; harmless since NS commutes with ᵀ, but the stated rule is wrong for one tab). (low)
- **Embedding "no operator"** — muon.md:134-139 overstates; the induced geometry changes (max-row-norm), it doesn't vanish. Conclusion (embeddings→AdamW) stands. (low)
- **Schedule-free prose** — lr-scheduler.md:749-757 reads as post-hoc averaging; the true coupled-iterate algorithm is only in Exercise 5. One clause in prose ("gradient at an interpolated point; the average is what you evaluate"). (low-med)
- **WSD "took over" / plateau "indefinitely"** — lr-scheduler.md:766, 659; our own "not settled" two paragraphs later contradicts the first. (low)
- **β₂=0.95 "almost universally … essentially every open recipe"** — adamw.md:182-188; PaLM used 0.99. Soften to "commonly." (low)
- **"three orders of magnitude … measured exactly so"** — sgd.md:289-291 etc.; batch 1→512 is 2.7 orders. (low) — NB the review's companion claim "one draw per size" is false; the code averages 200 draws.
- **"twice the critical value"** — momentum.md:191-194; critical β is per-mode (depends on ηλ). Soften. (low)
- **"the tuned learning rate" (singular)** — adamw.md:251-253 glosses genuinely different PT/JAX rates; say "each at its own tuned rate." (low)

Code-touching (⇒ notebook re-execution + recapture for the affected files):
- muP output-bias fix (scaling.md) — P0 #3.
- Pin `nesterov` in Muon library calls (muon.md) — P0 #10.
- **Warmup "identical network" is false for PyTorch** — lr-scheduler.md:547-548 claims identical nets; the two PT cells build unseeded nets (JAX is genuinely identical via `nnx.Rngs(0)`). Seed the PT `net_fn` or soften the claim (prose-only alternative). (low-med)
- **Branch-off-plateau plots restart the epoch axis** — each `train()` re-creates the animator from 0, so the fork is invisible (lr-scheduler.md:678-725). Overlay on an absolute axis. (low-med, pedagogy)
- **AdamW heatmaps: independent color scales, no colorbar** — verified in `show_grids` (adamw.md:382-403); mitigated by printed per-cell numbers. Optional shared vmin/vmax. (low)
- **JAX EMA maps over all state leaves** — safe for this model (verified: 6 Param + 2 BatchStat, all float32) but asymmetric with the guarded PT tab; filter by type for hygiene. (low)
- Newton–Schulz could transpose tall matrices / use the smaller Gram (efficiency only). (low)

## 3. Policy-backed prose softening (overclaims the house precision rule already forbids)

- adam.md: "the gap **cannot be tuned away** / **never closes**" from a 4-point single-seed grid (L679, 888, slide 1071). → "did not close under any rate in this grid." **(med)**
- adamw.md: "the same column wins **across seeds**" — code runs one seed per framework (L409-411). → "in this single-seed run (and across both frameworks)." **(med)**
- muon.md: "the hybrid **never lost**" (summary L835, slide 976; JAX tab L525) — 4-point, 1-seed, 2 runs; the PT tab itself hedges to parity. → "matched or beat AdamW in both single-seed runs." (low-med)
- "tuned / matched tuning" for four-point grids (adam.md, muon.md, practice.md) → "best of a coarse four-point grid" at first use. (low)
- "within run-to-run noise" without a measured band (lr-scheduler.md:473-475; muon.md:505-507; practice.md:456-457) — used in the *cautious* direction, least urgent; "too close to call from a single run" avoids implying a measured spread. (low)
- P2 "honest" purge: ~18 instances; the confidence-cue uses (momentum.md:371, adam.md:812, muon.md:815, optimization-intro.md:269 …) should become the actual limitation; the "plain/faithful" uses are fine. Aligns with the writing-avoid mannerism policy.

## 4. Not applicable

**Misreads (review wrong about our text/code):** R–M α=1/2 (P0 #6); AdamW semantics (P0 #8); "one draw per size" (200-draw average, sgd.md:248-250); "use a log scale" (plot already log, minibatch-sgd.md:513-517); `%load_ext d2lbook.tab` "artifacts" (standard repo source markup, stripped by the preprocessor, absent from rendered HTML); CNN optimizer selection "on test" (selects on training loss, adam.md:791-792); recipe-table undisclosed fields (dashes + "a dash means the report does not say" already there); QK-norm claim already scoped to input norms; "change one thing at a time" (we separately recommend joint sweeps + quasi-random search); parameter groups "by rank only" (split_lm already uses named roles); Adam bias correction (already scoped to stationarity); warmup single-cause (we give four mechanisms + Gotmare); AdaGrad growth (already "under persistent noise"); Newton claims (nonconvex failure + damping/trust-region already demonstrated); shuffling-vs-iid (sgd.md:300-310 already does it); "100× compute" (already scoped "per step"); "noise reduction at no cost" (we wrote "no extra gradient evaluations" and treat the effective-LR change); validation "not reported" for the decoupling grid (it is, adamw.md:306); "Adam smoothed ℓ∞" (already marked analogy with the exact limiting case); Muon 15-matmuls/1% (correct and cited); bf16 NS (capability claim, not a code claim).

**Policy conflicts (house rules govern):** all "≥3 seeds / error bars / CIs / fitted curves" demands; train/val/test three-way splits book-wide (schedule ranking and EMA eval on test follow the book convention; adamw.md already carves a val slice where generalization is the claim); NumPy-instead-of-tabs (and most gd.md cells are untagged/shared anyway, so the alleged duplication is minimal); step/token as primary axis (review itself concedes epochs for the classroom testbed).

**Design preferences (Alex's call, not correctness):** the four-question chapter reframe; marking Muon/muP as an optional advanced path; figure consolidations (17→4 schedule panels, phase-portrait SGD, contour saddle); "dated evidence map" table; Identity/Theorem/Heuristic/Report labeling scheme; roofline rewrite of the hardware argument. The review's "Hyperball (Kosson et al. 2026)" citation is unverified — treat with care if adopted.

## 5. Review's suggested additions, triaged

- **One-step scratch-vs-framework conformance tests** — good, cheap, matches the "code teaches" ethos; strongest candidates: Adam (adam.md) and Muon (muon.md, incl. tall/wide matrices). Worth doing.
- **Parameter-group diagram + exactly-one-group assertion** — muon.md already builds role-based groups in code; an assertion is one line; diagram optional.
- **Update-diagnostics dashboard** — partially exists (clip fraction + norms in practice.md, coordinate check in scaling.md); a consolidated dashboard is a nice-to-have, not a gap per se.
- **Experiment-protocol box** — adopt the *scoped* version only (declare budget/grid/selection metric up front; we largely do); the multi-seed half is policy-conflicted.
- **Time-scale conventions box** — small and useful; could live in minibatch-sgd or index.
- **Dated "as of mid-2026" stamps on frontier sections** — cheap, consistent with the precision policy.

## 6. If fixes are ordered: scope

- **Prose-only batch** (≈25 edits across 11 files): everything in §1 except rows 3/10, all of §2-prose, all of §3. No re-execution; re-render only.
- **Code batch** (re-execution + recapture of scaling / muon / lr-scheduler [+practice, adamw if opted]): muP bias, Muon nesterov pin, warmup seeding, branch-plot absolute axis, (optional) heatmap shared scale, JAX EMA filter, NS transpose.
- Not doing: everything in §4.
