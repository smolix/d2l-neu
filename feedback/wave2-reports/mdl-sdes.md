# Wave-2 report: chapter_mdl-dynamics/mdl-sdes.md

## Structure (1126 lines; was a 321-line stub)

4 top-level `##` + Summary + Exercises + Discussions tab + 6 slides; 11 code
cells (1 imports ×4 tabs + 10 untagged NumPy), 15 `:eqlabel:`s. All 9 stub
labels survive on matching headings: `sec_mdl-sdes` (anchor),
`sec_mdl-why-randomness`, `sec_mdl-wiener-process` (###s under ## Brownian
Motion), `sec_mdl-quadratic-variation`, `sec_mdl-ito-lemma` (under ## Itô
Calculus), `sec_mdl-sde-definition` (## SDEs and Euler–Maruyama),
`sec_mdl-euler-maruyama` (###), `sec_mdl-ornstein-uhlenbeck` (##),
`fig_mdl-dyn-sde-paths` (kept include). Propositions w/ proofs: BM covariance,
QV→t (L², 3 lines), Itô isometry (3 lines), Itô's lemma (Taylor heuristic,
proof cited to Øksendal), EM strong/weak orders (statements + Milstein-term
intuition), OU transition kernel (full derivation via Itô on e^{θt}X + isometry).

## Errata fixed (review lines 776–780)

- **EM order**: both rates stated — strong ½ general/multiplicative, strong 1
  additive g(t) ("every SDE in this chapter"), weak 1 always; text predicts
  slope ≈ 1 on OU and the code measures **1.02** (plus GBM vs its exact
  solution measuring **0.45** — both propositions confirmed side by side).
  Exercise 6 rewritten to match (replaces drafted ex. 4).
- **VP wording**: Var(X_t) = ᾱ_t·Var(X₀) + (1−ᾱ_t) = 1 for **all** t given
  unit-variance data (σ² = 2θ, ᾱ_t = e^{−2θt}); explicitly *not* the
  "drift balances diffusion at stationarity" claim. Demonstrated on Rademacher
  ±1 data: Var pinned at 1.000–1.014 while E[X⁴] glides 1 → 3 (= 3 − 2ᾱ_t²).
- **§6.x purged**; all cross-refs are `:numref:` (12 distinct targets, all
  verified to exist). Citations converted to `:cite:`/`:citet:` keys.
- **Caption nit**: envelope now "half-width $2\sqrt{(\sigma^2/2\theta)(1-e^{-2\theta t})}$".
- GBM exercise added (ex. 5: −σ²/2 via Itô on log X, ties to the strong-order cell).

## Figure SPECS (report-only; NOT created — text has no dangling refs)

- `mdl-dyn-brownian-paths`: 3 panels, the ±√Δt walk at Δt = 0.1 / 0.01 / 0.001
  (same seed, consistent coarsening) converging to a Brownian path, ±√t and
  ±2√t envelopes; shows the random-walk → Wiener limit and the √t spread.
- `mdl-dyn-qv-convergence`: running partial sums t ↦ Σ_{t_i≤t}(ΔW_i)² of one
  path at n = 16 / 256 / 4096 against the line y = t (converging band), with a
  gray companion curve Σ(Δx_i)² → 0 for the smooth x(t) = sin 2πt.
- `mdl-dyn-ou-mean-reversion`: OU drift arrows −θx in the (t, x) plane, 3–4
  sample paths from different starts relaxing to 0, saturating ±2σ_t band, and
  the stationary Gaussian drawn as a sideways density on the right edge.

## BibTeX (NEW keys used; d2l.bib NOT edited — for reconciliation pass)

```bibtex
@book{Sarkka.Solin.2019, title={Applied Stochastic Differential Equations},
  author={S{\"a}rkk{\"a}, Simo and Solin, Arno}, year={2019},
  publisher={Cambridge University Press}}
@book{Oksendal.2003, title={Stochastic Differential Equations: An Introduction
  with Applications}, author={{\O}ksendal, Bernt}, year={2003}, edition={6},
  publisher={Springer}}
@book{Kloeden.Platen.1992, title={Numerical Solution of Stochastic
  Differential Equations}, author={Kloeden, Peter E. and Platen, Eckhard},
  year={1992}, publisher={Springer}}
@article{Uhlenbeck.Ornstein.1930, title={On the Theory of the Brownian Motion},
  author={Uhlenbeck, George E. and Ornstein, Leonard S.}, year={1930},
  journal={Physical Review}, volume={36}, number={5}, pages={823--841}}
```

Existing keys used: `ho2020denoising`, `song2021score`.

## Verification log

- Cells extracted from the committed .md and exec'd cumulatively, in order:
  `ALL 11 CELLS OK` under `.venv-pytorch`, `.venv-tensorflow`, `.venv-jax`,
  `.venv-mxnet` (identical output; pure seeded NumPy). Slowest cell 0.23 s.
- Key measured numbers quoted in prose: Var(W_t) = 0.2486/0.4950/1.0014 at
  t = 0.25/0.5/1; max |Cov − min(s,t)| = 0.005; QV 0.9980 at n = 2¹⁸ vs sin-QV
  7.5e−5; Itô gap 1.0150 (T = 1), max |gap−t| = 0.024; strong slopes OU 1.02 /
  GBM 0.45; weak slopes 1.01/1.01; OU cloud mean 0.7336 vs 0.7358 at t = 1,
  ±2σ coverage 0.952 vs 0.954; stationary var 0.4095 vs 0.405.
- `tools/lint_source.py chapter_mdl-sdes.md`: exit 0, no warnings.
- Gotcha sweeps clean: no closing-$-then-digit, body `$$` fences alone on
  their lines with `:eqlabel:` attached, no `]` in the image caption, tab
  block balanced, all `:numref:`/`:eqref:` targets exist.

## Deviations

- 1126 lines vs the ~750–900 target — matches the finished Wave-2 peer
  (constrained-optimization-duality, 1145) and the review's ~800–1000 band;
  ~190 of those lines are Summary/Exercises/slides.
- "One imports cell": one cell ID (`#sdes-imports`) with four `#@tab` variants
  (`from d2l import <fw> as d2l` + numpy), exactly the peer's pattern — a
  truly untagged imports cell cannot load the per-framework d2l module.
- Strong-order cell measures GBM (multiplicative, exact solution) *alongside*
  OU so both proposition rates are confirmed in one plot — a deliberate small
  extension of the spec'd "slope on OU" demo.
- "Martingale" used once, defined inline in one sentence (fair-game property),
  per the dependency note.
