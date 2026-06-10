# Wave-2 report: chapter_mdl-dynamics/mdl-fokker-planck-probability-flow.md

## Structure (1190 lines; was 314-line stub)

5 top-level `##` + Summary/Exercises/Discussions + 7 slides; 8 logical code
cells (1 tabbed imports + 7 untagged pure-NumPy), 3 committed figures kept.

- `## From Paths to Densities` — Lagrangian/Eulerian; **### Three Identities
  from Vector Calculus** (divergence+flux/divergence theorem, Laplacian as
  neighborhood-average, integration by parts w/ 1-line proof — the part's
  dependency hole, closed self-contained); OU cloud-vs-analytic demo.
- `## The Fokker–Planck Equation` — test-function derivation (Itô + parts);
  **Prop: Gaussian solves OU-FP iff ṁ=−θm, v̇=σ²−2θv** (coefficient-matching
  proof ⇒ OU kernel solves FP by direct differentiation; stationary = fixed
  point); heat equation f=0; finite-difference residual check.
- `## The Continuity Equation and the Probability-Flow ODE` — flux-balance
  derivation; **SIGN FIXED**: ½g²Δp = ∇·(p·(+½g²∇log p)) (eq_mdl-dyn-
  diffusion-as-transport), minus appears only inside −∇·(pv), v = f−½g²∇log p;
  explicit sign-warning sentence; numerically verified (plus-sign residual
  3.2e-6 vs 0.79 scale; minus-sign misses by 1.59 = 2×); PF-ODE prop + proof
  (linear transport uniqueness); likelihood corollary → CNF; showcase
  `#fokker-planck-pf-ode-overlay`.
- `## The Score Function` — def, Gaussian + K-mixture (responsibility
  formula) + symmetric-pair tanh closed form; Z-invariance; mixture-score cell.
- `## Time Reversal` — Bayes-on-EM-kernel intuition; Anderson stated;
  **λ-family prop** b_λ = f − (1+λ²)/2 g²∇log p (proof = FP bookkeeping in the
  reversed clock; λ=1 Anderson, λ=0 PF-ODE) ⇒ factor-of-2 story; zero-training
  reverse-SDE sampler from N(0,1) recovers the bimodal p₀.

All 10 stub labels survive on matching headings (`sec_mdl-probability-flow-ode`
is now a `###` inside the continuity `##`). All §6.x pseudo-numbering purged.

## Figure specs (NOT created; decision per brief)

- **mdl-dyn-density-spreading — TRIMMED as redundant.** The
  `#fokker-planck-cloud-marginals` cell overlays evolving OU histograms +
  analytic densities at 3 times, and committed `mdl-dyn-forward-reverse.svg`
  already shows bimodal→Gaussian density slices. Recommend not creating.
- **mdl-dyn-sde-vs-pfode — optional; message already carried** by the showcase
  cell's left panel (8 jagged EM paths over 8 smooth Heun PF-ODE trajectories,
  same p₀, plus 3 matching marginal panels). If a static version is wanted for
  decks: 1×2 panels, house style via tools/gen_mdl_dynamics_figures.py; left:
  OU θ=1, σ=√2 from bimodal p₀, 6 EM paths (thin C0, dt=0.005) + 6 PF-ODE
  trajectories (thick C1, closed-form mixture score), t∈[0,3]; right: shared-y
  marginal density curves p_t at t∈{0.25,1,3} (analytic, one per color),
  annotated "same marginals, different paths".

## BibTeX (new keys cited in text; NOT added to d2l.bib)

```bibtex
@Article{Anderson.1982,
  title   = {Reverse-time diffusion equation models},
  author  = {Anderson, Brian D. O.},
  journal = {Stochastic Processes and their Applications},
  volume  = {12},
  number  = {3},
  pages   = {313--326},
  year    = {1982}
}
@Book{Risken.1996,
  title     = {The Fokker--Planck Equation: Methods of Solution and Applications},
  author    = {Risken, Hannes},
  edition   = {2},
  publisher = {Springer},
  year      = {1996}
}
```
Existing keys used: `song2021score`. Fence-id placeholders + 3 fig refs lint-clean.

## Verification log (2026-06-10)

- Cells extracted cumulatively and run end-to-end in **all four venvs**
  (.venv-pytorch/-tensorflow/-jax/-mxnet, MPLBACKEND=Agg): identical output,
  ~4–8 s total per venv (well under 60 s/cell). Seeds: 42/0/7/99.
- Key numbers (quoted in prose): cloud t=0.5 mean 1.2137/1.2131, std
  .7987/.7951; FP residual ≤1.3e-4 vs scales 4.8/0.54/0.067; identity
  +sign 3.18e-6 vs wrong-sign 1.588 (scale 0.794); ∫p_t=1.000000;
  KS(SDE,ODE) = .0065/.0086/.0081 (n=20k, crit ≈.014); tanh-vs-responsibility
  5.2e-14; Z-invariance 3.6e-13; score zeros [−2,0,2]; reverse SDE: split
  .500/.500, modes ±2.00, widths .260 (→.252 at half Δt, O(Δt) weak bias),
  KS .0172; max|p_T−N(0,1)| .0015.
- `tools/lint_source.py` (pytorch venv): exit 0, no warnings. Closing-$-digit
  scan: none. No `]` in captions. eqlabels (20, `eq_mdl-dyn-*`) corpus-unique.

## Deviations

- **Length 1190 > brief's ~700–850**: the mandated self-contained vector-
  calculus subsection, 7 verified code cells, 8 exercises and 7 slides did not
  fit the smaller envelope; matches the finished peer (constrained-optimization
  ≈1139). Happy to compress on request.
- Canonical-collision risk for wave-3 reconciliation: this file claims
  `eq_mdl-dyn-score-def` and `eq_mdl-dyn-reverse-sde`; 27.4 should reference,
  not redefine. Forward refs into 27.4 use only the file anchor
  `sec_mdl-score-matching-diffusion-flow` (no subsection labels), per risk note.
- Discussions block converted from stub's `:begin_tab:` form to the finished-
  peer `## Discussions` prose + link form.
