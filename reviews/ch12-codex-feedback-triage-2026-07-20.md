# Triage: Codex review of ch. 12 "State Space Models"

*2026-07-20. Input: `ssm_feedback.md` (~773 lines, ~68 distinct checkable
claims + a style mandate + structural proposals). Every factual claim was
verified by two Opus passes against the source files, the committed
executed outputs (both frameworks), the figure generators, and the primary
papers (HiPPO, S4, Mamba-2, copy bounds, negative-eigenvalue parity,
test-time regression). Detailed per-claim tables with quotes and file:line:
`scratchpad/ssm-feedback-verify-{classical,modern}.md` (session scratchpad).
Verdicts here; fixes are a separate pass.*

## Headline

A high-quality review — the best-supported round Codex has produced on this
book. Its execution-output numbers are accurate without exception, and 12
of its claims are genuine defects, including two "Critical" rows that fully
stand (HiPPO's missing 1/t; the matrix-state weighted-least-squares
overstatement, which also contradicts our own ch. 12.6). Several other
"Critical" rows shrink on verification: the LSTM total-Jacobian charge and
the overwrite-conclusion charge are already mostly handled by the chapter's
own scoping and reduce to one-line qualifiers. Tallies:

| REAL | PARTIAL | FALSE | STYLE | POLICY |
|---|---|---|---|---|
| 13 | 32 | 7 | 16 (+ mandated sweep) | 2 (+ additions list) |

Style ruling from Alex (2026-07-20, supersedes the earlier keep-the-motif
judgment): the absolutes sweep and the **complete** "honest" purge (7
remaining occurrences across index/lstm/matrix-state/deltanet) are
mandated for the fix pass.

## Confirmed real defects (fix pass, priority order)

1. **HiPPO-LegS displayed as an LTI ODE** — the paper and our own figure
   generator integrate the time-varying `x' = (Ax+Bf)/t` system; the prose
   says the coefficients evolve by "exactly the form" of the LTI equation.
   Fix the theorem statement + discrete update, and add the short
   HiPPO→S4-initialization bridge so the real-diagonal S4D toy stops
   inheriting the Legendre-optimality theorem. (ssm.md:685-688 vs
   gen_mdl_modernrnn_figures.py:703-706.) The one genuinely substantive
   correction in the round.
2. **matrix-state calls the decayed outer-product state a weighted-LS
   *solution*** (:197-199) — it is the weighted cross-moment; the exact
   minimizer needs the inverse weighted key covariance, exactly as our own
   12.6 says ("covariance-deleted"). Cross-file contradiction; fix the
   matrix-state sentence + one displayed exact-solution formula.
3. **hybrids extrapolation exercise indexes past the position table** —
   the exercise pads to 96 tokens; `pos_emb` has 64 rows. Rewrite the
   exercise (adequate max_len or position-free variant) + an executable
   length assertion.
4. **Sequential-image readout inconsistency** — prose says the class is
   requested "at the end" and leans on final-state retention; the code
   mean-pools all 784 outputs. Fix prose (describe mean-pooled sequence
   classification) or add the final-state readout comparison; do not keep
   the retention conclusion as-is.
5. **mamba.md summary line "topped our capstone scoreboard"** — JAX's
   committed run has minGRU 85.1 < Mamba 88.9. The body is properly
   hedged; the summary is not. One line.
6. **Memory-table units** — SSM state priced at fp32 against fp16 KV rows
   in one table (matrix-state), and hybrids uses a different Mamba-2
   config/dtype (655,360 elts fp32 vs 1,048,576 elts fp16) with no
   reconciliation. One symbolic formula + consistent dtype + explicit
   config per instance.
7. **mLSTM overflow detector argmaxes a boolean** — reports index 0 when
   nothing overflowed. `any()` guard.
8. **Chunk-size-one sentence** (matrix-state) — diagonal blocks become 1×1,
   they do not vanish. One sentence.
9. **WY cost stated as O(C²)** omitting the d_k/d_v factors, and the code
   materializes an explicit inverse (solve vs identity, then multiply)
   instead of solving against the actual right-hand side. Fix cost line +
   use the direct triangular solve.
10. **JAX/PT normalization epsilon parity** in deltanet's read-out —
    identical numerical contract in both tabs.
11. **GLAMixer misnamed** (hybrids) — scalar-per-head gate = the Mamba-2/
    scalar rung of the chapter's own taxonomy, not GLA's per-coordinate
    gate. Rename (e.g. `GatedStateMixer`, "scalar-gated") or implement the
    per-coordinate gate.
12. **Zamba2 diagram says 54 layers** (2.7B variant) while table/config say
    81 for the 7B. Regenerate figure from the table's config.
13. **Five-verbs count** (index) — "five verbs" then six italicized verbs
    with *hybridize* framed as the truce. One-line rewording.

## Notable PARTIALs (kernel real; smaller fix than proposed)

LSTM/GRU Jacobian: chapter already frames diag(F) as the protected additive
path; add the direct-vs-total qualifier + math-appendix cross-ref (the
"false as written / Critical" framing is overstated). ZOH B̄: state the
invertibility assumption + expm1 note. S4 used bilinear: one sentence +
the discretization-table row (see policy). Capacity law: add the
random-independent-isotropic-keys assumptions to prose + figure caption
(+ cheap correlated-key curve — accepted, numpy-only). Parity: quantify the
theorem's assumptions; report JAX's tanh failure already at T=16 (output
confirmed 0.487/1.000/1.000); β=2σ(·) boundary note. Copy bound: state the
uniform-distribution + finite-state assumptions next to the formula; same
quantifier treatment for the transformer-side theorem. Hybrid sweep
summary: quote the JAX hybrid row (0.94-1.00) rather than
"indistinguishable across the sweep". TTR training semantics: the
"no gradient from any pretraining objective" sentence needs the
outer/inner distinction (a diagram is policy; the sentence fix is not).
TTR solver spectrum: align objectives (add the same ridge term to the GD
steps or compare to batch OLS) — JAX 30-pass 0.008 < ridge 0.010 confirms
the comparator mismatch; keep NW as a separate estimator family. Longhorn
"not designed, parameterized, or learned" vs the learned β_t — reword to
"form derived, coefficient learned". Titans "trained from scratch inside
the forward pass" — outer/inner phrasing. Drift-cell conclusions scoped.
Timing-plot provenance lines (device/dtype/warm-up/"teaching
implementation"). Param-count printout in the hybrids matched-models cell.
Memory plot axis → "persistent decode state". MOHAWK described as learned
approximation (not duality-guaranteed conversion). Recipe-table context
column split (train vs max vs eval). bpb over the full target stream
(currently one batch column). Δ-collapse narration → add the cheap
inference-only Δ-vs-token-role plot (accepted: zero-training evidence for
a narrated mechanism). Index pre-announcements softened to match scoped
section claims. Concise-cell init caveat. "Starts as S4D" → decay-timescale
only. Selective-copy "no time-invariant model" → designed-to-stress
phrasing (both index + mamba).

## False alerts (do NOT "fix")

1. **Duplicate "Sequential Image Classification" heading** — only one
   exists.
2. **Rerun benchmark framed as a KV-cache comparison** — the chapter
   already frames it as step-vs-rerun and handles the KV cache separately.
3. **"Growth is impossible" unscoped** — already scoped to the protected
   path at the cited location.
4. **Scoreboard caveats hide the param/optimizer asymmetry** — mamba.md
   discloses both (the caveats paragraph).
5. **Overwrite conclusion overbroad** — the restricted hypothesis class is
   defined before the experiment and the conclusion is scoped to it.
6. **Longhorn stability wording** — already "along the key" /
   nonexpansive-scoped.
7. **Titans presented as the full architecture** — the omissions are
   already stated.

## Style (mandated by Alex — fix pass, non-negotiable)

Absolutes sweep: every "proves / exactly / never / the whole family /
theory says must / without bound / no time-invariant model / as far as
anyone knows" replaced by the supported quantifier. Complete "honest"
purge (7 occurrences). Also accepted from the style section: the
evidence-vocabulary discipline (identity / proposition / diagnostic /
illustration / empirical finding) applied opportunistically during the
sweep, and dated fragile claims ("the ladder stops here").

## Policy items (Alex to decide)

1. **Core/Advanced reading tracks** within the chapter (Codex's split:
   core = lstm/ssm/mamba(+SSD); advanced = matrix-state/deltanet/ttr/
   hybrids). Cheap: an index paragraph + per-section tag line.
2. **Title** — Codex reprises the rename ("Modern Recurrent and
   State-Space Models") or a scope paragraph. Alex already decided: title
   stays; the scope paragraph is a cheap middle ground.
3. **Content additions**: classical-SSM foundations box;
   discretization/stability table (ZOH/bilinear/Euler); unified cost/state
   table across the eight architectures; evaluation taxonomy table;
   inner/outer-loop diagram + terminology box; stochastic-SSM scope note;
   "state" terminology table; prerequisites + "what this does/doesn't
   show" boxes per notebook; exercise tags (conceptual/short-code/
   extended). All are additive teaching aids, none blocking.
4. **Deployment-cost trims**: the 18 parity trainings (count confirmed)
   and Longhorn's 200-solve/GD verification loop — Codex wants them
   author-side. Counterpoint: total runtime is modest (~90 s parity,
   seconds for Longhorn) and the multi-seed parity variability IS the
   taught content (representational wall vs optimization horizon).
   Recommend: keep parity as-is but trim to 2 seeds or cite an authoring
   table; shrink Longhorn's check to a handful of deterministic residuals.
