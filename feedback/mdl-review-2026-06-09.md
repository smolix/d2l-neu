# Mathematics for Deep Learning — Comprehensive Review (2026-06-09)

State reviewed: `main` @ `02cb3f7b` (post the 2026-05-31 expert-review fix pass).
Method: 16 parallel adversarial reviewers — one per written section (12), one per
stub chapter (3), one part-level architecture review — each grounded against
external benchmarks (Strang/Trefethen/Axler, Bubeck/Boyd/EECS127/EE364a,
Wasserman/Blitzstein/Bishop/Murphy/Wainwright–Jordan, Cover&Thomas/Polyanskiy–Wu/
Poole/McAllester–Stratos, Särkkä–Solin/Song/Lipman/MIT 6.S184, Garrett Thomas
math4ml/MML/CMU 10-606, CS229/CS231n/CS168/CS236), with numerical verification of
proofs, code cells, and committed outputs where checkable. Three of the boldest
findings were independently re-verified in the main thread.

---

## Executive summary

**No written section is broken, and none is finished.** All 12 written sections
land at **MINOR WORK** — the architecture, proofs-first style, and coverage are
genuinely at or above the Berkeley/Stanford bar (the part is a strict superset of
MML Part I, Garrett Thomas's CS189 companion, and CMU 10-606's math half), but
adversarial checking still surfaced ~40 must-fix correctness defects, almost all
surgically small: false claims in captions/slides, figure↔text mismatches, one
false exercise, and a handful of genuinely wrong mathematical sentences inside
otherwise-correct proofs. The three stub chapters have unusually good skeletons;
their outlines contain a small number of factual bugs that **must be fixed before
delegating writing** (writing agents transcribe stubs verbatim).

| Section | Verdict | Headline must-fix |
|---|---|---|
| 22.1 Geometry | MINOR WORK | det-figure prose describes a figure that doesn't exist; d=2 cosine caption false |
| 22.2 Eigendecomposition | MINOR WORK | norm-ratio→\|λ₁\| claim false for complex pairs; caption rate 0.382 vs true 0.146 |
| 22.3 SVD & Low Rank | MINOR WORK | Exercise 8 is mathematically false; inverted clause in tightness proof |
| 23.1 Single-Variable | MINOR WORK | subgradient "descent direction" claim false; promised exercise missing |
| 23.2 Multivariable | MINOR WORK | slide says PSD⇒min (false); Lagrange missing ∇g≠0 |
| 23.3 Matrix Calc & AD | MINOR WORK | tape example: code/figure/caption are three different graphs |
| 23.4 Integral | MINOR WORK | improper-integral cell contradicts its own printout; N^(−2/d) rate mismatch |
| 25.1 Random Variables | MINOR WORK | Chebyshev "equality case" false in the stated (closed-interval) form |
| 25.2 Distributions | MINOR WORK | max-entropy claim missing the base measure h; figure contradicts caption |
| 25.3 Maximum Likelihood | MINOR WORK | proof sentence false on its own coin example; broken slide figure |
| 25.4 Naive Bayes | MINOR WORK | "MAP under uniform prior" claim is wrong (it's the posterior mean / Beta(2,2) MAP) |
| 25.5 Statistics | MINOR WORK | ddof=1 code contradicts "exact identity" claim; no double-descent caveat |
| 24 Optimization | MAJOR (unwritten) | strong skeleton; outline errata: bfloat16 ε=2⁻⁷, garbled Welford, stale §3.x refs |
| 26 Information Theory | MAJOR (partial) | legacy 26.1: two numerically broken worked examples (joint sums to 1.1; MI off ~100×) |
| 27 Dynamics | MAJOR (unwritten) | strong skeleton; outline errata: FP sign error (×2), EM strong-order claim |

Part-level verdict: the 6-chapter spine (LA→Calc→Opt→Prob→IT→Dynamics) is sound;
one mild ordering inversion (24's SGD needs ch.25 expectations — mitigate with an
explicit prerequisite line), and one real hole: **ELBO / EM / variational
inference is taught nowhere in the part**, while two ch.26 stubs point at
"main-book VAE / self-supervised chapters" that do not exist.

---

## Priority roadmap

### P0 — Correctness (fix before anything ships)

**Mathematically false statements:**
1. **25.1 Chebyshev** (`mdl-random-variables.md:416–435` + figure caption + slide + generator panel title): stated over the *closed* interval, so at p=1/8 the boundary atoms are *inside* and the "equality case" has mass 0 outside, not 1/4. Restate in the standard form P(|X−μ|≥ασ) ≤ 1/α² throughout. *(Re-verified in main thread.)*
2. **25.4 Laplace smoothing** (`mdl-naive-bayes.md:121`, echoed at 216 and slide 276): "(n+1)/(n+2) is the MAP under a uniform prior" is false — Beta(1,1)-MAP is the unsmoothed MLE; add-one is the posterior *mean* (rule of succession), or MAP under Beta(2,2). Also the "+|V|" forward-ref conflates multinomial and Bernoulli event models (l.121 vs l.208). *(Re-verified.)*
3. **26.1 broken worked examples** (`mdl-information-theory.md:335–373, 418`): conditional-entropy joint `[[0.1,0.5],[0.2,0.3]]` sums to **1.1** and matches neither marginal; MI example feeds the wrong marginal (computed 0.719 bits vs true 0.0074). Every printed number in both cells is meaningless. *(Re-verified: joint sums to 1.1.)*
4. **22.2 power iteration** (`mdl-eigendecomposition.md:848–850, 909–911`): "norm ratio still converges to |λ₁|" for a dominant complex pair is false for non-normal matrices (counterexample: ratios oscillate 0.32–8.6 forever). And the figure caption + in-SVG annotation (`gen_mdl_figures.py:619–621`) attach rate |λ₂/λ₁|≈0.382 to the *norm-ratio* panel whose gap actually closes at (λ₂/λ₁)²≈0.146; 0.382 is the *direction* rate.
5. **22.3 Exercise 8** (`mdl-svd-low-rank.md:962–965`): "a rank-r update changes at most r singular values" is false (rank-1 update moving both σ's verified). Restate as Weyl |σᵢ(W+Δ)−σᵢ(W)| ≤ σ₁(Δ) or interlacing.
6. **23.1 subgradients** (`mdl-single-variable-calculus.md:622`): "any element of the subdifferential is a legitimate descent direction" is false (subgradient method is not a descent method), and chained per-kink subgradients need not be subgradients of the composite (torch returns slope 0 for relu(x)−relu(−x) ≡ x at 0). Weaken per Bolte–Pauwels.
7. **23.2 slides** (`mdl-multivariable-calculus.md:925, 935–936`): "PSD = local minimum" (must be PD/ND); "O(model size) *per parameter*" (the whole point is one sweep for all parameters).
8. **25.3 proof sentence** (`mdl-maximum-likelihood.md:345`): "each xᵢ carries empirical mass 1/n" is false with repeated outcomes (the coin has p̂(H)=9/13). Fix the justification, not the identity.
9. **25.5 bias-variance** (`mdl-statistics.md:138, 136–141`): "MSE, which is the expected test error" drops the irreducible-noise term; "literally the same picture" needs an overparameterization/double-descent caveat (the cited `sec_generalization_basics` itself warns the U-curve breaks).
10. **25.2 max-entropy** (`mdl-distributions.md:966–972`): solution p ∝ h·exp(ηᵀT) does not follow from plain Shannon entropy — must be entropy *relative to base measure h* (Wainwright–Jordan §3.1; geometric-vs-Poisson counterexample). Also l.908 "every distribution above" — the uniforms are *not* exponential-family; say so (classic counterexample).
11. **23.4 integral** (`mdl-integral-calculus.md`): l.79–80 false integrability claim (need bounded + piecewise continuous, not absolute integrability); l.331–361 improper-integral cell prints 1.000455→1.000500 then claims "limit = 1.000000" (left-rule bias — all 4 frameworks' committed outputs); l.719–745 N^(−2/d) is the *second-order*-rule rate while the chapter teaches the first-order left rule, and the d=4 grid line coincides exactly with the MC line so the prose "they cross near d=4" describes something the plot doesn't show.
12. **23.2 Lagrange** (`mdl-multivariable-calculus.md:372–378`): needs the ∇g(x*)≠0 constraint qualification — one clause.
13. **22.2 BPTT formula** (`mdl-eigendecomposition.md:1006–1011`): missing transposes in both layout conventions.
14. **22.3 tightness proof** (`mdl-svd-low-rank.md:745–749`): parenthetical asserts b=u₁ gives the *largest*-norm solution; it gives the smallest (which is why the ratio is maximized).
15. **23.3 tape example** (`mdl-matrix-calculus-autodiff.md:629–665` + `gen_mdl_calculus_figures.py:696`): code builds `(u*v+u)*(u*v+u)` (two duplicate subgraphs, 7 nodes), text/caption claim a shared-r diamond (5 nodes), figure shows a third graph (unary square). Fix: `r = u*v+u; y = r*r` everywhere — also the best demo of gradient `+=` accumulation.
16. **23.3 Exercise 7** (`:804–806`): the PyTorch recipe is reverse-over-reverse, mislabeled as the forward-over-reverse :eqref:`eq_mdl-hvp`; use `torch.func.jvp(torch.func.grad(L), …)`.
17. **22.1 figure↔text integrity** (`mdl-geometry-linear-algebraic-ops.md`): l.1130–1150 prose describes a determinant figure (grid + highlighted region, A=[[1,2],[−1,3]], "the specific case shown here") that does not exist — the actual figure shows three different matrices; l.841 linear-map figure silently uses a different matrix than the worked example; l.1167–1185 code uses the transpose of the text's A (works only because det(Aᵀ)=det(A), stated nowhere); l.478 caption "d=2 histogram is broad and flat" is false (arcsine-shaped, peaks at ±1; *flat* is d=3).
18. **23.1**: l.172 promises a quotient-rule exercise that doesn't exist; l.374 Rolle step claims "opposite signs" where the correct statement is opposite *weak* signs with equal limits.
19. **25.5 ddof**: `mdl-statistics.md:145` claims the decomposition "holds to numerical precision" but the demo (l.220–250) uses ddof=1, which visibly breaks exactness (5×10⁻⁵ gap); exercise 3 (l.530) asserts the false converse. Use ddof=0 in the demo; rewrite ex.3.
20. **25.3 broken slide + broken promise**: `:798` slide `@fig:mdl-mle-kl` resolves to nothing in `img/auto/` (figure silently dropped); `:472–474` cites the Cramér–Rao bound "of :numref:`sec_mdl-statistics`" — which never states it.
21. **Cross-chapter rule violation**: 26.1 re-derives the binary NLL (l.674–684) and the full multinoulli MLE⟺CE proof (l.791–831), both owned by `subsec_mdl-nll-crossentropy` in 25.3 — convert to callbacks.

**Stub-outline errata (fix BEFORE delegating chapter-writing):**
- **24 Optimization**: bfloat16 ε is 2⁻⁷ not 2⁻⁸ (`mdl-numerical-stability-conditioning.md:41`); the Welford update at l.92–94 is garbled (conflates mean and M2 recursions); heavy-ball √κ rate is quadratic-only (Lessard) — caveat at `mdl-gradient-based-optimization.md:70`; stale "§3.x" planning-doc numbering baked into callouts.
- **27 Dynamics**: sign error in `mdl-fokker-planck-probability-flow.md:127,137` — ½∇·(g²∇p) = ∇·(p·(+½g²∇log p)), the minus appears only inside −∇·(pv); "Euler–Maruyama strong order ½" (`mdl-sdes.md:224–232` + ex.4) is wrong for this chapter's additive-noise SDEs (order 1 — the planned exercise would refute the text); "variance-preserving" wording must say Var(x_t)=1 for all t, not "drift balances diffusion"; DDPM step = EM only to first order; "§6.x" refs in rendered prose will be wrong (chapter renders as 27); ~8 citations (vincent2011, chen2018, anderson1982, lipman2022, liu2022, …) have no d2l.bib keys.

### P1 — High-value improvements (per-section, the big ones)

- **22.1**: define span/basis/dimension/null space properly (used throughout, never defined — biggest LA gap vs 18.06/CS229/3B1B); finish the Fashion-MNIST demo with the w·x histogram figure + interpretation; state matrix-mult = composition; move/compress the einsum section (~200 lines off the geometric arc); add exercises covering the new material.
- **22.2**: close the distinct-eigenvalues proof gap (minimal-relation observation); one sentence of QR-algorithm honesty; unify w₁/v₁ + color between caption and SVG; trim dead imports; 3-line Cholesky remark (or explicit deferral).
- **22.3**: fix the vacuous LoRA/Eckart–Young claim (state the honest link; PiSSA makes it literal); **use the polar decomposition you derived — Muon** (one paragraph makes the section current); Gavish–Donoho/noise-threshold paragraph or exercise; convert plain-text citations to `:cite:`; replace exercises 2–3 (re-ask proofs printed in the text).
- **23.1**: state Newton's method (invoked 3×, never stated); add the descent-lemma sandwich figure; deduplicate the 3×4 plotting cells (or make one compute the Taylor error rate); one autograd cross-check cell; visualize the η-sweep regimes.
- **23.2**: fix the Clairaut "proof sketch" (symmetric second difference, one line); collapse the first 4-framework example to one untagged cell; TF backprop cell should use one `t.gradient(f, [w,x,y,z])` call; add a Lagrange exercise; differentiability caveat (partials → gradient needs C¹).
- **23.3**: make TF/MXNet tabs actually verify against their autograd (the section's stated premise); state the diagonal-Jacobian fact for elementwise activations (the most common Jacobian in practice — absent); add a Jacobian-as-linear-map figure; one vmap/batched-Jacobian sentence; exercise deriving ∇_A log det A = A⁻ᵀ (used by name in two other chapters, derived nowhere).
- **23.4**: add **integration by parts** (promised by `mdl-distributions.md:568` and needed by ch.26–27, never stated); add the missing (∫e^(−x²))² = ∬ Fubini bridge to √π; pre-generate the 3-D bell surface (only "code draws" violation in the chapter); fix the FTC-check off-by-one (compare against f(x[1:])).
- **25.1**: prove Chebyshev via Markov (and get Markov — the one standard omission vs Wasserman/B&H); add one compute cell to the 550-line Several Variables half (e.g. verify the log-normal change-of-variables by histogram); make the covariance-vs-correlation figure contrast real (currently invisible); state discrete LOTUS.
- **25.2**: hoist the six mid-cell scipy imports; add one computed teaching plot (Binomial→Poisson overlay); compress the two trivial uniform cells into a closing reference table (law/pmf/mean/var/DL role); close the over-dispersion loop (Gamma–Poisson *is* negative binomial); one-paragraph mixture-models hook (whole chapter never defines a GMM); seed the unseeded cells.
- **25.3**: add Beta-prior MAP on the coin (naive-Bayes already forward-promises it — and fixing it surfaces P0 #2); MLE equivariance sentence (contrast with MAP non-invariance already discussed); misspecification one-liner (KL-projection, White 1982); promote "Why ML Works" + Fisher into their own `##` section; consistency hypotheses honesty (identifiability + uniform convergence).
- **25.4**: add the generative-vs-discriminative figure (the prior review's request was only half-delivered); one paragraph on continuous features (Gaussian NB); exercises: NB→softmax bridge + empirical pseudocount sweep; fix overloaded n (classes vs examples).
- **25.5**: add one worked hypothesis test in code (the section promises A/B-test machinery and never computes a p-value); power-curve figure; one-sentence FDR/Benjamini–Hochberg; bootstrap-failure sub-question on the Unif(0,θ) max exercise; add `Efron.1979` to d2l.bib.

### P2 — Part-level structure (from the architecture review)

1. **Give ELBO/EM a home**: new 25.3 subsection "Latent Variables, EM, and the ELBO"; fixes the two dangling "main-book VAE/SSL chapters" pointers in 26.2/26.3 (those chapters don't exist) and gives ch.27's Luo-style diffusion-as-ELBO pointer a target.
2. **Execute the 26.1 migration** (its own l.222 note): joint/conditional entropy + MI → 26.3; make 25.3 canonical for CE=MLE; rewrite the legacy prose to the bar (8 `##` sections → 3–5; one-imports-cell; bits-vs-nats policy — recommend nats default).
3. **Replace all hard-coded §3.x/§5.x/§6.x stub refs with `:numref:`** — they render wrong as chapters 24/26/27.
4. **Two scores disambiguation**: parameter score ∇_θ log p (25.3, Fisher) vs data score ∇_x log p (26.2 canonical, 27.3 references) — add an explicit warning box; classic student confusion.
5. **κ/conditioning**: 22.3 canonical; 24.1/24.4 reference and add only what's new (κ(AᵀA)=κ², GD-rate link, ridge).
6. **Convexity vs main-book ch.12**: theory → 24.2, applied/code → ch.12, reciprocal numrefs. Today `chapter_optimization/` has **zero** back-references to MDL; `chapter_optimization/convexity.md:48–356` is a near-total duplicate of what 24.2 plans. Sequence the ch.12 slimming deliberately (touches captured outputs).
7. **Concentration**: add Markov→Chebyshev→Hoeffding (+ Gaussian-shell remark) in ch.25 — standard in every benchmark curriculum and needed for the generalization links.
8. **Unify stub conventions** (3 competing formats; optimization files have no `##` headings at all → empty ToCs) and end-matter (Discussions/slides markers).
9. **Naming**: retitle ch.25 "Probability and Statistics" (or earn "Statistical Learning"); reframe Naive Bayes as the Bayes-rule + log-space payoff (the one section a syllabus would raise an eyebrow at); normalize label hygiene (underscore/hyphen mix: `sec_mdl-maximum_likelihood` etc.) **before** the stub chapters accrete refs against today's labels.
10. **24.1 prerequisite line** for SGD's expectations (point to `sec_prob` + forward to 25.1) to neutralize the 24↔25 ordering inversion.
11. **Kernel-trick primer box** in 26.2 (MMD assumes RKHS vocabulary; a full kernels section is not warranted).

### P3 — Writing the three stub chapters

Detailed write plans are in the per-chapter sections below. Headlines:

- **24 Optimization** (~800–950 lines/section, 4-section split confirmed): make 24.1 self-contained on L-smooth + quadratics, move the convex/strongly-convex rate *proofs* into 24.2 as the payoff; add the smooth *non-convex* GD rate (the only guarantee that applies to deep nets — 3-line corollary of the descent lemma); define subgradients; add projected GD; add the Fenchel conjugate (needed by 26.2's Donsker–Varadhan); backward-vs-forward error in 24.4; one-paragraph edge-of-stability and implicit-bias remarks (no proofs). No cvxpy — verify KKT/duals with closed forms. Build the 5 missing figures.
- **26 Information Theory** (3-section split confirmed): rewrite 26.1 (4 `##` sections: Information/Entropy → CE+KL → coding view+perplexity → modern uses) with the Kraft/Shannon-code sketch grounding the pervasive "extra bits" language; 26.2 regroup 10 planned headings into 4, add Jensen–Shannon explicitly, a Sinkhorn paragraph, and prove 1-D W₁=∫|F_P−F_Q|; 26.3 add **data-processing inequality + conditional MI** (biggest gap — IB and all representation claims lean on it) and optionally Fano. The 26.1-trim and 26.3-write must land together or MI is duplicated.
- **27 Dynamics** (4-section split confirmed vs Särkkä–Solin/MIT 6.S184): add **Langevin dynamics** (zero mentions; the original score→sample bridge — a 40-line gift reusing 27.3), **classifier-free guidance math** (~30 lines; MIT's 2026 edition has a dedicated lecture), and **DDIM as a named result**; self-contained vector-calculus box (∇·, Δ, parts) in 27.3 — the part never teaches divergence; keep W₂/Benamou–Brenier self-contained (26.2's OT plan is W₁-only); budget exactly one real 4-framework training loop (CFM velocity field, Gaussian→two-moons); fix the time-axis convention clash (diffusion 0→T vs FM 0=noise→1=data) with an explicit callout.

---

# Part-level architecture review

**Verdict.** The 6-chapter architecture is sound and, when finished, would be
*more* than competitive with the benchmark curricula: it is a strict superset of
MML Part I (Deisenroth/Faisal/Ong), of Garrett Thomas's CS189 companion, and of
CMU 10-606's math half, and it retains all 11 sections of the original d2l
appendix while adding SVD/low-rank, matrix-calculus/AD, an
optimization-foundations chapter, divergences, MI, and a genuinely distinctive
Dynamics capstone no benchmark offers. The LA→Calculus→Optimization→Probability→
InfoTheory→Dynamics spine has one mild inversion (SGD's variance analysis needs
ch.25 expectations) and one real hole (variational inference/ELBO is taught
nowhere, yet two ch.26 stubs point to "main-book VAE chapters" that do not
exist). The biggest structural risks are not ordering but (a) the
half-modernized legacy ch.26.1 with a self-confessed internal duplication, (b)
three competing stub/Summary/Discussions conventions across the unwritten
chapters, and (c) hard-coded "§3.x/§5.x/§6.x" pseudo-numbering in stubs that
won't survive rendering as chapters 24/26/27.

### Ordering & dependency findings
- **Actual graph:** 22→{23,24,26}; 23→{24,25,27}; 24→{26 (Jensen/conjugates for f-GAN dual), 27}; 25→{26,27}; 26→27. Dynamics correctly declares all five predecessors in its index (`chapter_mdl-dynamics/index.md:10-14`). Order is topologically consistent except:
- **24↔25 inversion (mild):** ch.24's SGD outline (`mdl-gradient-based-optimization.md` §3.1.6) needs E[ĝ]=∇f, Var∝1/b, Robbins–Monro — ch.25 material. Mitigated because main-book `sec_prob` (preliminaries) exists; benchmarks split (MML puts Probability before Optimization; Thomas the reverse). Recommendation: keep the order (Calculus→Optimization continuity is worth more) but add an explicit prerequisite line in §3.1.6 pointing to `sec_prob` + forward to `sec_mdl-random_variables`.
- **ELBO/variational inference has no home — verified by grep.** Across `chapter_mdl-*` the only ELBO mentions are stub callouts: `mdl-divergences-distances.md:129-146` (MLE≡forward KL, "ELBO≡reverse" with cross-ref to "main-book VAE / variational-inference chapters") and `mdl-mutual-information.md` ("main-book self-supervised / contrastive chapters"). **Neither target exists** — the main book has GANs but no VAE/SSL chapter (grep over `chapter_*` confirms). Ch.27 deliberately derives DDPM via SDE rather than ELBO, but Barber–Agakov/VIB in 26.3 presuppose variational reasoning. Missing prerequisite content, not just a dangling link.
- 23.2 "Optimizing on a Constraint" (Lagrange teaser) precedes 24.3's full treatment — right direction, but the two must be explicitly linked teaser→canonical.
- Stubs encode dependencies as literal "§3.1.3 / §5.2.8 / §6.1" chapter-local numbers and bare backtick labels instead of `:numref:` — these numbers will be wrong on the rendered site (chapters render as 24/26/27). Conversion debt across all 11 stub files.

### Redundancy & canonical-home recommendations
- **Entropy/CE vs MLE:** ch.25 `mdl-maximum-likelihood.md` (written) already proves "NLL Is the Cross-Entropy to the Data"; legacy 26.1 re-derives "Cross-Entropy as An Objective Function of Multi-class Classification" and the IT index re-promises "minimizing cross-entropy *is* maximum likelihood". Canonical: **definitions (H, CE, KL) → 26.1; the CE=MLE equivalence → 25.3**, with 26 referencing back in one line.
- **26.1's confessed duplication:** `mdl-information-theory.md:222` says its joint/conditional-entropy + MI material "migrates to §5.3" but "is retained here for now". Execute the migration before polishing — currently MI is taught twice within ch.26.
- **Score function (three homes, two meanings):** parameter score ∇θ log p in 25.3 (written, Fisher information); data score ∇ₓ log p in 26.2 (Fisher divergence) and 27.3 (`## The Score Function`). Canonical: keep 25.3 for the Fisher/parameter score; make 26.2 the canonical *data-score/Fisher-divergence* definition; 27.3 references it. Add an explicit "two scores" disambiguation box — this is a classic student confusion the part currently does nothing to prevent.
- **Convexity vs main-book ch.12:** main book `chapter_optimization/convexity.md` (`sec_convexity`) already covers sets/functions/Jensen with code. The 24.2 stub plans the same catalog (three lenses, simplex, PSD cone) despite its own "do not duplicate" note. Canonical: **theory (equivalence proofs, strong convexity, conjugates, Jensen) → 24.2; keep `sec_convexity` as the applied/code treatment** with reciprocal numrefs. Jensen's canonical statement must live in 24.2 since 26 leans on it.
- **Condition number (three homes):** defined with the error bound in 22.3 (`mdl-svd-low-rank.md` §The Condition Number, written); 24.1 §3.1.3 reuses κ for GD rates (good); 24.4 §3.4.4 plans to *restate* κ=σmax/σmin and the forward-error bound. Canonical: **22.3 defines; 24 references and adds only what's new** (κ(AᵀA)=κ², GD-rate link, ridge-as-conditioning). The "one number, two consequences" framing in 24.4 is excellent — keep it, but as a synthesis, not a redefinition.

### Part-level coverage: add / cut (sources actually checked)
Checked: **Garrett Thomas, *Mathematics for Machine Learning* (gwthomas.github.io/docs/math4ml.pdf — full ToC extracted)**; **MML book ToC (mml-book.github.io)**; **CMU 10-606/10-607 F18 schedule (Gormley, cs.cmu.edu)**; **original d2l.ai appendix ToC (d2l.ai)**; CS229 review notes (cs229-linalg/cs229-prob) are already the cited refreshers in the LA and Prob index reading lists.
- **Nothing from the original appendix was dropped** — all 11 legacy sections have homes. Coverage ⊇ Thomas (incl. his 3.12–3.16 SVD/low-rank/pseudoinverse), ⊇ MML Part I, ⊇ 10-606; 10-607's discrete/computation half is rightly out of scope.
- **Add (in priority order):** (1) **ELBO / latent variables / EM** — one subsection in 25.3 after MAP ("Latent Variables, EM, and the ELBO") is the natural home; MML (ch.11 via GMM) and CS229 both teach EM; this also repairs the dangling pointers in 26.2/26.3 and gives Luo-style diffusion-as-ELBO a reference point. (2) **Concentration inequalities** — ch.25 stops at Chebyshev; add Markov→Chebyshev→Hoeffding (+ high-dim Gaussian shell) in 25.1 or 25.5; standard in every serious math-for-ML course and needed to make `sec_generalization_basics` links meaningful. (3) A **kernel-trick primer box** in 26.2 (MMD assumes RKHS vocabulary; full kernels/RKHS section not warranted — none of the four benchmarks has one). (4) MCMC: skip as a section; Langevin already lands in 27.4 — add one paragraph connecting it to sampling more broadly.
- **Don't add:** generalization theory (VC/Rademacher) — out of scope, keep the pointer to `sec_generalization_basics`; graphs/spectral — the PageRank/Perron–Frobenius aside in 22.2 is the right dosage; numerical LA/floating point — already a strength (24.4 + 22.3), distinctive vs all four benchmarks.
- **Cut/reframe:** **Naive Bayes (25.4)** is an ML model, not math — the one section a Stanford/Berkeley syllabus would raise an eyebrow at; reframe as the worked payoff of Bayes' rule + the log-space trick (which 24.4 already advertises as its payoff), or fold into 25.3. *(Section-level reviewer dissents: at 305 lines it earns its slot as the chapter's "probability becomes an algorithm" capstone — author's call; both agree it must not grow.)*

### Consistency findings (index pages, figures, cross-refs, summaries)
- **Index pages:** genuinely uniform and strong — one-paragraph opener stating the chapter's thesis, toc, "Resources and Further Reading" with Books/Courses/Tutorials. Comparable quality across all six; IT/Dynamics add "Foundational papers" (fine). Minor: optimization's list is the only one dominated by paywalled books; LA files LoRA under "Tutorials".
- **Figures:** SVG filenames perfectly uniform (`img/mdl-{la,cal,opt,prob,it,dyn}-*`; counts 17/21/4/19/3/5 — stubs thin, as expected, but figure inventories are pre-planned in stub callouts). **Figure *labels* are inconsistent**: `fig_mdl-prob-pdf-cdf` vs unprefixed `fig_mdl-secant-to-tangent`, `fig_mdl-bootstrap`, `fig_mdl-mutual_information`.
- **Labels mix underscores and hyphens:** legacy-derived `sec_mdl-maximum_likelihood`, `sec_mdl-random_variables`, `sec_mdl-information_theory`, etc. vs hyphenated everywhere else. Normalize before more cross-refs accrete.
- **Cross-refs: not silos.** Dense bidirectional `:numref:` traffic (calc→prob/dynamics, prob→IT variational bounds, integral-calc→`sec_mdl-continuous-normalizing-flows`, refs into main-book `sec_softmax`/`sec_gd`/`sec_prob`, and preliminaries chapters point forward into the part). The weak spot is stub-internal "§3.x" pseudo-refs.
- **Three competing stub conventions:** optimization = per-subsection `⟢ Planned` callouts, *no `##` headings at all* (those four pages render with an empty ToC); dynamics + 26.2/26.3 = top `callout-important` status banner + flat `##` runs of 8–11 sections (violating the house 3–5-section rule — needs regrouping under `###` when written); 26.1 = written legacy prose with embedded planning callouts.
- **End-matter drift:** all 23 files have Summary+Exercises, but `## Discussions` exists only in the 4 optimization files; dynamics/IT instead end with legacy `:begin_tab:` Discussions links; `<!-- slides -->` markers are missing from all 4 dynamics files and 26.2/26.3.
- **Titles:** part and chapter titles mostly catalog-ready. Two quibbles: "Probability and Statistical Learning" oversells — the chapter is classical probability + statistics, with zero learning theory; retitle "Probability and Statistics" (matching its directory name) or add the concentration/generalization material to earn it. "Dynamics: Differential Equations and Generative Flows" reads well but is the only colon-subtitled chapter of the six; acceptable for a capstone, just deliberate-ize it.

---

# Chapter 22 — Linear Algebra

## mdl-geometry-linear-algebraic-ops.md — verdict: MINOR WORK

The mathematics is solid: every proposition re-derived (Cauchy–Schwarz, triangle inequality, projection, near-orthogonality with Var = 1/d, the unifying det/dependence/invertibility theorem, det multiplicativity) and all are correct; every numerical claim spot-checked with numpy holds (rank(C)=2 with all 10 column-triples dependent, Av=[0,−5], det=5, all 8 exercises well-posed with the intended answers). The blockers are not math but **text↔figure integrity**: the determinant prose describes a figure that does not exist as described, the linear-map figure silently uses a different matrix than the worked example, and the high-dim-cosine caption makes a factually wrong claim about its own d=2 histogram (it is arcsine/U-shaped, not "broad and flat" — measured density deciles ~1.03 at ±1 vs ~0.32 at center; *flat* is d=3). Coverage-wise the section never properly defines span/basis/null space despite leaning on them, and the einsum section sits off the chapter's geometric arc.

### Scorecard
1. **Clarity/correctness of proofs: A−** — all proofs correct and intuition-first per house style; docked for the wrong cosine-histogram caption and the quietly circular C–S "one-picture" argument (uses the geometric dot formula it is meant to legitimize).
2. **Diagrams: B−** — generators (tools/gen_mdl_figures.py) are well-crafted, but two figures contradict their surrounding text and the strongest demo (Fashion-MNIST) has no figure of the thing it teaches.
3. **Teachability: B+** — lecture-ready propositions and a genuinely teaching mean-difference classifier, but the demo ends without interpreting its accuracy, and the 8 exercises cover none of the newly added material (C–S, projection, concentration, orthogonal matrices).
4. **Structure: A−** — five `##` sections with clean `###`/`####` nesting, intuition-before-formalism throughout; einsum (lines 1312–1508) breaks the geometric narrative the intro (lines 12–21) and index.md promise.
5. **Coverage: B** — vs MIT 18.06, CS229 notes, 3Blue1Brown (all checked): span/basis/subspace/null-space are name-dropped, never defined; matrix-multiplication-as-composition never stated; einsum wastes ~200 lines.

### Correctness issues (must-fix)
1. **Line 478 caption is false.** "The d=2 histogram is broad and flat" — the cosine of two random unit vectors in d=2 is arcsine-distributed, peaking at ±1 (verified numerically). Fix the caption, or change the figure's dims to {3, 10, 1000} (`fig_cosine_highd`, tools/gen_mdl_figures.py:436) where d=3 *is* flat.
2. **Lines 1130–1150 describe a nonexistent figure.** "Consider the grid image from before, but now with a highlighted region" and "in the specific case shown here of A=[[1,2],[−1,3]] … area is 5" — but `fig_determinant` (gen_mdl_figures.py:389–429) shows three *different* matrices (areas 2.06, 1.9, 0), no grid, and never the text's A. Rewrite the prose to match the 3-panel figure (or add A as panel (a)).
3. **Line 841 figure/text mismatch.** The worked example (lines 822–839) computes with A=[[1,2],[−1,3]] (Ae₁=[1,−1], Ae₂=[2,3]), but `fig_linear_map` (gen_mdl_figures.py:350) draws A=[[1.4,0.9],[0.3,1.2]]. A reader cannot check the computation against the picture. Regenerate with the text's matrix.
4. **Lines 1167–1185:** the det code cell uses [[1,−1],[2,3]] — the *transpose* of the text's A. It prints 5 only because det(Aᵀ)=det(A), a fact stated nowhere. Use the text's A verbatim.
5. **Line 894 dangling promise:** "as we will see in the next section their volume scaling is det Q = ±1" is never derived; it needs det(Qᵀ)=det(Q), which this section never states. Add the one-liner after multiplicativity (det(Q)² = det(QᵀQ) = 1, given det Mᵀ = det M) or drop the forward reference.

### High-value improvements (prioritized)
1. **Define span / basis / dimension / null space properly.** "Basis" appears once in passing (lines 818–820); column space + null space get one parenthetical (lines 997–1001). The eigen/SVD sections and the SVD's "four fundamental subspaces" payoff (mdl-svd-low-rank.md:26) presuppose these; one short `###` with a 2-D picture would close the gap.
2. **Finish the Fashion-MNIST demo.** After the accuracy cell (line 773): state the accuracy, interpret it, and add a figure of the 1-D histogram of w·x for both classes with the midpoint threshold — that figure teaches the hyperplane story far better than the two mean-image `imshow` cells (lines 668–713), which could be merged into one.
3. **State matrix multiplication = composition of maps** before line 1278 uses it implicitly in the multiplicativity proof.
4. **Add exercises on the new material:** C–S equality case, a projection computation, "sample pairwise cosines in d=10⁴ and compare std to 1/√d" (code), and "show orthogonal matrices preserve angles".
5. **Fix slide divergences** (lines 1630–1648): "In code (cont.)" claims the snippets cover "norms, determinants, and inverses" but @expressing-in-code-3/4 are einsum contractions; "Translate all of this into NumPy / PyTorch" is wrong framing in the TF/JAX framework views.

### Coverage: add / cut (sources checked: MIT 18.06 OCW calendar; Stanford CS229 "Linear Algebra Review and Reference" (Kolter/Do); 3Blue1Brown "Essence of Linear Algebra" chapter list)
- **Add (core):** span/basis/subspace/null space (18.06 lects 6–11; CS229 §3.6, §3.9; 3B1B chs. 2, 7) — used here but never defined; biggest gap.
- **Add (small):** one paragraph on nonsquare matrices as maps between dimensions (3B1B ch. 8); one sentence naming `linalg.solve` as the practical alternative in "Numerical Issues" (lines 1092–1125).
- **Defer-but-note:** Gram–Schmidt/QR (18.06 lects 14–17, CS229 §3.8) appears nowhere in the chapter; if eigen/SVD don't own it, the orthogonal-matrices subsection should at least say how orthonormal bases are constructed.
- **Cut/move:** the einsum section (lines 1312–1508) — pure API notation, absent from all three references' geometry treatments, and contradicts index.md's "geometry + two decompositions" framing; move to a tensor-mechanics home or compress to a half-page coda.

### Polish
1. Line 490: "In an $d$-dimensional vector space" → "a $d$-dimensional".
2. Line 834: transpose misplaced — "$(\mathbf{A}[0,1])^\top$" should be $\mathbf{A}[0,1]^\top$; also "+ −1(…)" reads badly.
3. Line 949: `\mathbf{v_i}` → `\mathbf{v}_i` (subscript outside the bold).
4. Line 980: "either column by itself is not linearly dependent" — double negative; say "each column alone is linearly independent".
5. Line 1424: "$\mathbf{A}\mathbf{v} = a_{ij}v_j$" conflates vector and component; write $(\mathbf{A}\mathbf{v})_i = a_{ij}v_j$.
6. Line 1456: entries set bold ($\mathbf{b}_{ijk}a_{il}$) — components should be non-bold; stray "." before "it can be implemented".
7. `fig_vectors` panel (b) draws the "same vector" in three different colors (gen_mdl_figures.py:170), visually undercutting the caption's point — use one color.
8. Lines 532–537 duplicate the figure caption's content nearly verbatim in prose; tighten one of the two.

## mdl-eigendecomposition.md — verdict: MINOR WORK

A strong section: the narrative arc (geometry → existence/diagonalizability → spectral theorem → PSD/Rayleigh → localization/computation → spectral radius and deep-net stability) is genuinely lecture-shaped, the proofs are short and intuition-first, and the four figures all earn their place. The prior fix to the power-iteration example holds at the headline level — figure, caption, and text all use B=[[3,1],[1,2]], λ₁=(5+√5)/2≈3.618, rate ≈0.382 (verified). But adversarial checking finds two real quantitative errors in exactly the showcase spots (must-fix 1–2 below). Everything else verified clean: Gershgorin example eigenvalues (0.9923, 2.9734, 4.9539, 9.0803), all four frameworks' committed power-iteration outputs match ratio = ρ to 10 decimals, exercise 3's Gershgorin bound, Gram-matrix PSD demo, and the spectral/Rayleigh/Gershgorin proofs re-derive correctly.

### Scorecard
1. Clarity & proof correctness: **B** — proofs re-derive cleanly, but two checkable quantitative errors (complex-pair ratio claim; r vs r² rate) plus a transposeless BPTT formula.
2. Diagrams: **B+** — all four figures earn their place and match the text's examples, but the wrong rate annotation is baked into `mdl-la-power-iter.svg` and the caption's w₁/red mismatches the figure's v₁/orange.
3. Teachability: **A−** — code cells verify claims rather than draw; 8 layered exercises; slides present; marred by dead imports and an mxnet tab that is 100% plain NumPy.
4. Structure: **A** — exemplary arc, intuition-first proofs, deferrals to SVD/optimization clearly signposted.
5. Coverage: **A−** — meets or exceeds Thomas math4ml §3.6–3.11 and MML ch.4; missing only a QR-algorithm reality line and (chapter-wide) Cholesky.

### Correctness issues (must-fix)
1. **Lines 848–850, 909–911** — "The *norm ratio* still converges to |λ₁|" for a dominant complex pair is **false** for non-normal matrices: counterexample S(1.5R₀.₇)S⁻¹ gives consecutive-norm ratios oscillating 0.317–8.611 forever (only ‖Aᵏv‖^{1/k} / the geometric mean → 1.5). Line 909's "*exactly*… not by coincidence" leans on this; the 4 committed seeds all happen to have real dominant eigenvalues. Fix: state that the ratio can oscillate persistently and that ‖Aᵏv‖^{1/k}→ρ(A), or note the demo's matrices have real dominant eigenvalues.
2. **Line 839 caption + `tools/gen_mdl_figures.py:619–621`** — "the ratio of consecutive norms converges… with the gap closing at rate |λ₂/λ₁|≈0.382" is wrong for the right panel: for symmetric B the eigenvalue-estimate gap closes at (λ₂/λ₁)²≈0.146 (verified empirically); 0.382 is the *direction* rate (left panel). Fix caption and the in-SVG annotation `gap ~ |λ2/λ1|^k`, then `make figures`.
3. **Lines 1006–1011** — BPTT formula ∂L/∂h₀=J_T⋯J₁ ∂L/∂h_T is wrong in both layout conventions: column-gradient form is J₁ᵀ⋯J_Tᵀ ∇_{h_T}L (row form puts the gradient on the left). Spectral conclusion unaffected; the formula as printed would be marked wrong in class.
4. **Lines 284–287** — the coefficient of λ^{n−1} in ∏ᵢ(λᵢ−λ) is (−1)^{n−1}∑ᵢλᵢ, not ∑ᵢλᵢ (false as written for even n); same sign factor on the determinant side. One-line fix: "both sides carry the common factor (−1)^{n−1}."

### High-value improvements (prioritized)
1. **Distinct-eigenvalues proof gap (lines 335–349):** the "shorter relation is nontrivial" step needs the observation that a minimal relation has ≥2 nonzero coefficients (a single-term relation c_m w_m=0 forces w_m=0). One sentence closes the hole.
2. **One sentence of algorithmic honesty** after the `eig` cell (line 198) or in the power-iteration intro: libraries compute spectra with the QR algorithm (Hessenberg reduction + shifts), and power iteration matters when only matvecs are affordable — grounded in Trefethen & Bau Part V.
3. **Figure/text notation:** caption says principal eigenvector "w₁ (red)"; the generator labels it `v₁` in ORANGE (`gen_mdl_figures.py:594–596`). Unify on w₁ (text's notation); slides at line 1124 also use VΛV⁻¹ vs the text's W.
4. **Dead imports (lines 22–58):** no code cell uses `d2l`, `display`, or matplotlib, yet all four imports cells load them and line 18–20 claims they're needed. Trim, or drop the prose claim.
5. **Slide "Eigenvectors govern long-run behavior" (lines 1170–1176):** "the gap closing at rate |λ₂/λ₁|" inherits ambiguity from must-fix #2 — say "the *direction* gap".

### Coverage: add / cut (sources checked: Strang 18.06 OCW lecture list; Garrett Thomas math4ml ToC §3.6–3.11; MML Deisenroth ch. 4; Trefethen & Bau attempted)
- **Covered and competitive:** everything in Thomas §3.6–3.11 and MML §4.1–4.2/4.4 is here, usually with better proofs; Strang's Markov-matrix lecture is covered by the PageRank/Perron aside; complex eigenvalues-as-rotations exceeds all three.
- **Add (small):** (a) the QR-algorithm sentence; (b) **Cholesky** appears nowhere in the chapter while MML gives it §4.3 — a 3-line remark after PSD (A≻0 ⟺ A=LLᵀ; how Gaussians are sampled) or an explicit deferral to the Probability chapter; (c) a half-line pointer that e^{At}/ODE stability lives in :numref: the Dynamics chapter.
- **Cut:** nothing. The Ginibre/Marchenko–Pastur passage is at the edge of scope but is load-bearing for the initialization thread and correctly stated (ρ∼√n vs σ_max∼2√n verified).

### Polish
1. Line 15–16: "as a **layerless** neural network does" — presumably "layered" (or "deep linear network"); as written it contradicts the iterated-map analogy.
2. Line 110: "the symmetric [[2,1],[1,2]] we analyze below" — it is never analyzed below, only assigned (exercise 1, line 1051).
3. Line 826: "the bracketed tail decays to c₁w₁" — the bracket *converges to* c₁w₁; the tail decays to zero.
4. Line 852: "run a few dozen iterations" vs `for _ in range(200)` in all four cells.
5. mxnet tab (lines 178–181, 721–728, 856–867) is plain NumPy throughout — legitimate (MXNet lacks `eig`), but worth a one-line `:begin_tab:` note so mxnet readers know why.
6. Line 298: "returns when we estimate log det Jacobians with the Hutchinson trace estimator" — dangling forward reference with no `:numref:`.
7. Lines 1110–1112: JAX discussion link duplicates TensorFlow's (`t/1087`).
8. Line 847–848: "(as happens for a real random matrix, whose eigenvalues come in conjugate pairs)" — overstated; real Ginibre matrices also have O(√n) real eigenvalues, and in all four demo seeds the dominant one *is* real.

## mdl-svd-low-rank.md — verdict: MINOR WORK

The mathematical core is strong and verified: the AᵀA existence proof, Eckart–Young (both norms), the pseudoinverse derivation, and the condition-number bound are all correct, short, and intuition-led; numerically confirmed the golden-ratio shear, the {2,1}-vs-√2 example, the lstsq/pinv agreement, cond(AᵀA)=cond(A)², and that the condition bound is exactly attained at b=u₁, δb=uₙ. Figures all exist and are generator-produced. What keeps it from READY: one exercise is mathematically **false**, one clause in the tightness proof asserts the opposite of the truth, a non-orthogonal matrix is called "the rotation matrix", a slide embeds the wrong code cell, and the "Modern Deep Learning" section — while accurate — reads 2021-era in a 2026 book (derives the polar decomposition, then never mentions Muon, the most prominent polar-decomposition application in current LLM training).

### Scorecard
1. **Clarity/proof correctness: A−** — all five major proofs check out (spot-checked numerically); marred by the inverted tightness clause (l.745–749) and "rotation matrix" misnomer (l.276).
2. **Diagrams: A** — all 5 figures exist (`img/mdl-la-svd-{action,subspaces}.svg`, `-eckart-young`, `-pca`, `-condition`), committed, generated in `tools/gen_mdl_figures.py:633–936`, captions self-contained; nothing missing.
3. **Teachability: B+** — code teaches and never draws; lecture-ready slides exist; but a slide/cell mismatch, a false exercise, and two exercises that re-ask proofs printed verbatim in the text.
4. **Structure: A** — 4 content `##` sections with clean `###` nesting; every `:numref:` target resolves (checked all 8 cross-chapter labels).
5. **Coverage: B+** — matches the Trefethen–Bau/CS168/Brunton–Kutz canon for the core; modern-DL picks are accurate but stale for 2026; no denoising/optimal-threshold view of truncation.

### Correctness issues (must-fix)
1. **Exercise 8 is false** (l.962–965). "A rank-r update changes at most r singular values" — counterexample (verified): W=diag(1,2), Δ=0.1·**11**ᵀ (rank 1) moves *both*: σ → {2.1099, 1.0901}. The true statements are Weyl perturbation |σᵢ(W+Δ)−σᵢ(W)| ≤ σ₁(Δ) and interlacing. Restate the exercise.
2. **Inverted clause in the tightness proof** (l.745–749): "b=u₁ (so x=σ₁⁻¹v₁ has the *largest* possible norm for its right-hand side)". Wrong direction: b=u₁ gives the **smallest**-norm solution — that is exactly why the ratio is maximized.
3. **"the rotation matrix"** (l.276–279, repeated in exercise 1, l.939): [[0,−2],[1,0]] is not a rotation (σ={2,1}, not orthogonal); call it "the scaled rotation".
4. **Slide embeds wrong cell** (l.1000–1012): slide "Where singular values come from" promises the defective shear with σ=φ,1/φ but embeds `@svd-verify` (the 3×2 reconstruction cell). Should be `@svd-defective-shear` (l.327).

### High-value improvements (prioritized)
1. **Fix the LoRA/Eckart–Young claim** (l.804–807). "Among all rank-r updates, the truncated SVD is the most expressive one" is vacuous — every rank-r matrix is expressible as BA; E–Y is about approximating a *known* matrix, while LoRA *learns* the update. State the honest link: if the full fine-tuning update has fast spectral decay, the best rank-r proxy misses only σ_{r+1}; PiSSA (arXiv:2404.02948) makes the link literal by initializing B,A from the truncated SVD of W.
2. **Use the polar decomposition you derived.** :eqref:`eq_mdl-polar` (l.141–160) is proved and then never applied. Muon — the optimizer behind 2025–26 frontier LLM training — replaces the momentum update by its polar factor UVᵀ via Newton–Schulz. One paragraph in §Modern-DL closes that loop and makes the section current.
3. **Citations as plain text** (l.409, 792, 813, 845) while the sibling section uses `:cite:`. None of Eckart–Young 1936 / Mirsky 1960 / Hu 2021 / Miyato 2018 / Halko 2011 have d2l.bib keys — add and convert.
4. **Exercises 2 and 3 re-ask proofs given verbatim in the text** (ex.2 ↔ l.425–431; ex.3 ↔ l.607–625). Replace — e.g., prove ‖A‖_F² = Σσᵢ² (asserted without proof at l.430–431 and used again at l.485), or an empirical noise-thresholding exercise.
5. **Modernize the attention remark** (l.825–827): "linear-attention approximations" is 2020-era framing; the prominent 2026 instance is DeepSeek MLA low-rank KV compression — one sentence.

### Coverage: add / cut (sources checked: Trefethen & Bau lectures 4–5; Stanford CS168 lecture 9; Brunton & Kutz ch. 1)
- **Add:** the *denoising* reading of truncation — Gavish–Donoho optimal hard threshold; one paragraph (or an exercise: add noise to a low-rank matrix, watch the spectrum split) would strengthen the "how to choose k" story, which currently has only the 95%-energy dial (l.491–503).
- **Add (one line):** matrix completion as a pointer application; state ‖A‖_F² = Σσᵢ² and |det A| = Πσᵢ as properties (used unproven).
- **Cut:** nothing — randomized SVD paragraph (l.844–856) is accurate and earns its place.

### Polish
1. l.785–786: `subsec_mdl-svd-modern-dl` label on a `##` section — rename to `sec_…`.
2. l.967–981: all four Discussions tabs point to the same placeholder `/t/svd`.
3. l.503–505: text credits fast decay to "natural images" but the figure uses a synthetic image (`gen_mdl_figures.py:807–811`); either reword or render a real photo.
4. l.568 caption: "drawn from the origin of the cloud" → "from the mean of the cloud".
5. l.467 + l.489: two $\blacksquare$ for one theorem; label the second "*Proof (Frobenius case).*"
6. l.920 + l.377–384: numerical-rank tolerance is σ₁·max(m,n)·ε in `matrix_rank`, not bare ε·σ₁.
7. l.836–841: `#svd-weight-spectrum` prints "10.5% of full" at rank 18 vs the 0.39% headline at l.803 — add a clause noting rank-for-95%-energy depends on decay rate.
8. l.255: deflation formula for σ_k without naming Courant–Fischer; a name + forward ref would help instructors.

---

# Chapter 23 — Calculus

## mdl-single-variable-calculus.md — verdict: MINOR WORK

The mathematical core is now solid: the gradient-descent treatment fully answers the prior review — "≈" is explicitly defined as first-order prediction (line 109), and the descent lemma (lines 269–290) carries an explicit L-Lipschitz-gradient hypothesis, a correct FTC-based proof, the η<2/L window and optimal η=1/L, matching Bubeck's β-smooth Lemma exactly; the x² worked example (L=2 ⟹ η<1) is verified correct, and every printed number reproduces (difference-quotient table incl. the 17.01/1.7e-7 error claims; the GD sweep +0.34868/0/+0.10737/+1.0/+6.19). MVT proof and the Taylor/Lagrange-remainder and smooth-not-analytic treatments are correct and honestly caveated. All 9 figures exist, are generator-backed, and stylistically uniform. What keeps it from READY: one mathematically false claim in "Why SGD Shrugs," a promised exercise that doesn't exist, ~12 near-identical 4-framework plotting cells that pad rather than teach, and Newton's method name-dropped three times but never stated.

### Scorecard
1. Clarity/correctness: **A−** — descent lemma, MVT, Taylor all verified correct; one false subgradient claim and two loose numerical/historical claims.
2. Diagrams: **A−** — 9 consistent generator-backed figures; missing the descent-lemma quadratic-upper-bound picture and any visual of the η-sweep oscillate/diverge regimes.
3. Teachability: **B+** — excellent narrative and exercises 2/5/6/8 are genuinely good, but 3 plot cells × 4 frameworks are duplication with zero framework insight, ex. 4 asks to "fill in" a proof printed in full, and no cell ever calls autograd to verify f'(4)=8.
4. Structure: **A** — five `##` sections with a clean arc; one imports cell per framework; eqlabels correctly attached.
5. Coverage: **A−** — meets/exceeds MML §5.1 and the 18.01 differentiation units, adds the right DL extensions (descent lemma, subgradients); misses inverse-function derivative and the Newton step.

### Correctness issues (must-fix)
1. **Line 622: "any element of the subdifferential is a legitimate descent direction" is false.** Subgradient steps are not descent steps in general (f(x,y)=|x|+2|y| at (1,0); cf. Bubeck §3.1), and the sentence sits in the multivariate network-training context. Adjacent overclaim, same line: "it only needs one valid subgradient per kink" — propagating per-piece subgradients through a composition does **not** yield a subgradient of the composite. Verified in the repo's pinned torch: `relu(x) - relu(-x) ≡ x`, yet autograd returns slope **0** at x=0. Weaken to: frameworks return one fixed element per kink; on the measure-zero kink set the chained value can even be wrong, and SGD is unbothered because such points are hit w.p. 0 (Bolte–Pauwels is the citable backing).
2. **Line 172 promises an exercise that doesn't exist.** "(The quotient rule … we leave it as an exercise.)" — no quotient-rule exercise appears in lines 634–642.
3. **Line 83: backprop "at the cost of essentially a single forward pass"** overstates; the cheap-gradient principle is a small constant (~2–3×) times one evaluation (Griewank–Walther, Baydin).
4. **Line 374 (Rolle step):** "the one-sided difference quotients have opposite signs yet must agree" — at an interior max they have opposite *weak* signs (≤0 / ≥0), and it is their *limits* that agree.

### High-value improvements (prioritized)
1. **State Newton's method.** It is invoked at lines 480, 565, and the recap slide (line 748) but the update x ← x − f'(x)/f''(x) never appears. Three lines after "The Best Quadratic" close the loop — especially since ch. 24 is still a stub.
2. **Add the descent-lemma figure**: f sandwiched below the parabola f(x)+f'(x)s+(L/2)s², with the step minimizing the parabola at η=1/L. It's *the* canonical picture and the lemma is the centerpiece of the section; `fig_mdl-gd-step` only shows the tangent step. Use the `mdl-figure` skill.
3. **Deduplicate the plot cells.** Lines 209–261, 391–447, 501–563 are three cells × 4 frameworks differing only in array namespace. Either merge linear+quadratic approximations into one cell, or make one of them compute something (e.g., max |f−P_n| vs. window width, demonstrating the (x−x₀)^{n+1} rate).
4. **One autograd cross-check cell.** The section hand-computes f'(4)=8 by finite differences but never verifies it with the framework's own derivative — would tie the chapter to :numref:`sec_autograd` and justify the framework tabs.
5. **Visualize the η-sweep** (line 314–326): the monotone/one-shot/oscillating/frozen/diverging regimes are only printed numbers; a small trajectory plot would make the 2/L threshold visceral.
6. **Fix ex. 4 (line 637):** "fill in the proof of the descent lemma" — the complete proof is printed at lines 278–290. Rephrase (reprove unaided, or extend to the vector case ‖∇f‖²).

### Coverage: add / cut (sources checked: MIT 18.01SC syllabus; Bubeck monograph; 3Blue1Brown Essence of Calculus lesson list; MML §5.1)
- **Add:** Newton step (above). Inverse-function derivative — one short paragraph; the DL-relevant instance is the sigmoid/logit pair and d/dx log x from e^x. Optionally one sentence on *why* e^x is its own derivative (e^ε ≈ 1+ε) since the table (line 123) asserts it while every rule gets a proof.
- **Rightly omitted** (don't add): implicit differentiation, related rates, L'Hôpital, curve sketching. The descent lemma + subgradient material *exceeds* MML §5.1 in exactly the DL-relevant direction — keep.
- **Cut:** nothing wholesale; trim the 4× plot-cell duplication and consider collapsing the three constant-curvature figures (lines 344–355) into one 3-panel figure.

### Polish
1. Line 45: `torch.pi = torch.acos(...)` — `torch.pi` exists natively in pinned torch 2.11 (verified); stale shim.
2. Lines 60–62 (jax imports): `import jax` and `import numpy as np` are unused in every jax cell of this section.
3. Line 276: "the bound is tightest at η = 1/L" — wrong word; the *guaranteed decrease is largest* there.
4. Line 128 heading "Three Rules from One Identity" introduces **four** rules (lines 132–135).
5. Line 622: "PyTorch, TensorFlow, and JAX all report ReLU′(0)=0" omits MXNet — say "all four frameworks."
6. Lines 658–660: jax discussion link duplicates tensorflow's (t/1089).
7. Line 723 (slide "Taylor series"): states f(x+ε)=Σ… as unconditional equality; the body (line 488) carefully caveats analyticity — slide contradicts text.
8. Line 8: "L(**w**) = f(x) with x ∈ ℝ" — say explicitly "freeze all weights but one," not an equality between an n-variable and 1-variable function.

## mdl-multivariable-calculus.md — verdict: MINOR WORK

The prose is strong and the math checks out: numerically verified the hand-computed backprop partials (all four equal −4096, matching torch.autograd), the gradient approximation example (1.081946 vs 1.082124), the sum-over-paths collapse to 1+4y, the Taylor-quadratic coefficients at (−1,0), the cubic-gap claim, and the critical-point values (−5, 0, −32). The steepest-descent, level-set, tangent-plane, and second-derivative-test arguments are correct, and the handoff to matrix calculus is clean. What keeps it from READY: two factual errors in the slides, one wrong cross-reference, a missing constraint qualification in the Lagrange teaser, a near-vacuous Clairaut "proof sketch," and several house-convention violations in the code cells.

### Scorecard
1. **Clarity/correctness of proofs: A−** — all derivations verified numerically and analytically; Clairaut sketch is the one weak proof.
2. **Diagrams: B+** — 5 figures all exist, one consistent style; missing a saddle figure at the second-derivative test (the bowl/trough/saddle surface lives in `img/mdl-la-psd.svg`, never pointed to from here).
3. **Teachability: B** — the two untagged pure-Python backprop cells (491–556) are the pedagogical core and genuinely teach; the first 4-framework example (97–177) sprawls; slides contain errors; 7 decent exercises but none exercises the new Lagrange subsection.
4. **Structure: A** — 5 top-level sections; the Bridge section defers Jacobian/layouts/AD cleanly and the next section picks up exactly there.
5. **Coverage: A−** — matches MIT 18.02 Unit 2, Berkeley math4ml §4, CS231n backprop notes. One real gap: no differentiability caveat.

### Correctness issues (must-fix)
1. **Slide "Hessians" is wrong** — line 925: "PSD Hessian = local minimum … NSD = maximum". PSD is *inconclusive* (the body says so at 792–794). Must say PD/ND. A student reading only slides learns a false test.
2. **Recap slide misstates backprop's complexity** — lines 935–936: "Backprop = reverse-mode chain rule, O(model size) *per parameter*". The entire point (body's punchline at 525–528) is O(model size) for *all* parameters in one sweep.
3. **Wrong cross-reference** — line 188: gradient descent "first introduced in :numref:`sec_autograd`". It's introduced in `sec_linear_regression`.
4. **Lagrange condition missing constraint qualification** — lines 372–378: silently requires ∇g(x*) ≠ 0 (and codimension 1). One clause fixes it; an instructor will be asked this in lecture.
5. **mxnet `np` shadowing** — line 747: the untagged hessians cell does `import numpy as np`; in the mxnet notebook this shadows `from mxnet import np` for all subsequent cells. Works today only because it's the last code cell — a latent trap and a one-imports-cell violation.

### High-value improvements (prioritized)
1. **Fix the Clairaut proof sketch** (lines 695–699): "perturbing in both orders produces the same net change" is trivially true of the shared endpoint — it proves nothing. State the real one-liner: both mixed partials are limits of the *same* symmetric second difference, symmetric in i,j by inspection.
2. **Collapse the first example (lines 97–177) to one untagged cell.** It is exp/log scalar arithmetic; the backprop cells already prove untagged plain Python is the house pattern for framework-free math.
3. **TF backprop cell contradicts the message** (lines 624–633): persistent tape + four separate `t.gradient` calls, right after prose saying one backward sweep yields all gradients at once. Use `t.gradient(f, [w, x, y, z])`.
4. **Add a saddle pointer or figure at the second-derivative test** (lines 786–794): either a small generated z=x²−y² figure or `:numref:` the figure at `mdl-eigendecomposition.md:566`.
5. **Add a Lagrange exercise** (e.g., maximize xy on x+y=1, or derive softmax as max-entropy on the simplex) — the new subsection is the only one with zero exercise coverage.
6. **One-sentence differentiability caveat** near lines 32–39: the coordinate-at-a-time derivation needs continuous partials.

### Coverage: add / cut (sources checked: MIT 18.02SC syllabus Unit 2; CS231n optimization-2; Garrett Thomas math4ml §4)
- **Add**: differentiability caveat (above); one sentence on why saddles dominate high-dimensional landscapes.
- **Right depth, keep**: Lagrange teaser — correct call to leave KKT/duality to ch.24, but needs the ∇g≠0 clause.
- **Cut**: nothing. Jacobian, layouts, AD, convexity are all correctly deferred.

### Polish
1. Line 593: pytorch comment "Initialize as ndarrays, then attach gradients" is mxnet copy-paste.
2. Lines 121/143/164: `import numpy as np` in pytorch/tf/jax import cells is unused by any tagged cell.
3. Label naming: `fig_mdl-cal-gradient-field` has the `cal-` infix; `fig_mdl-tangent-plane`, `fig_mdl-chain-1/2`, `fig_mdl-taylor-quadratic` don't. Pick one scheme.
4. Line 222: "the proposition's equality criterion" — ambiguous antecedent; say "the Cauchy–Schwarz equality criterion (:numref:…)".
5. Line 89: ∇f written as a row without ^⊤, two paragraphs after declaring the column convention.
6. Lines 872–874: jax Discussions link duplicates tensorflow's (t/1091).
7. Line 497: `print(f'    f at …')` magic 4-space alignment indent — fragile.
8. Level-set proof (lines 253–257) is definitional; one sentence defining tangent via a curve γ(t) in the level set with d/dt L(γ(t))=0 would make it a proof rather than a restatement.

## mdl-matrix-calculus-autodiff.md — verdict: MINOR WORK

A strong section, arguably the best-engineered in the calculus chapter: every derived identity checks out numerically (∇xᵀAx, the least-squares outer product, the softmax–CE collapse to p−y, the Dual/VDual JVP columns, the Var tape gradients against torch — all exact), the chain-rule and dual-number proofs are short and correct, the JVP/VJP cost model is stated crisply, and the Dual → VDual → tape progression is the right pedagogical ladder. But it ships with one genuine three-way inconsistency (tape code vs. figure vs. caption disagree about the computation graph), a mislabeled HVP recipe in Exercise 7, an editorial leftover, and the TF/MXNet tabs quietly break the section's own promise of per-framework autograd verification. Coverage gaps vs. the canonical sources are small but real: the diagonal Jacobian of elementwise activations — the single most common Jacobian in a real network — is never stated.

### Scorecard
1. **Clarity/correctness: A−** — all math and both AD engines verified correct; docked for the tape-example graph mismatch and Ex. 7 mislabel.
2. **Diagrams: B** — `mdl-cal-fwd-vs-rev.svg` and `mdl-cal-tape-dag.svg` exist and render the right ideas, but tape-dag contradicts the code and caption; no Jacobian-as-local-linear-map figure in §1.
3. **Teachability: A−** — ~20-line Dual, ~15-line VDual, ~35-line tape are the right sizes, sequenced correctly, each closed by a verification cell; 7 well-graded exercises; docked because TF/MXNet readers never see their framework verify anything.
4. **Structure: A** — 4 top-level `##` sections; one imports cell per framework; all cross-refs resolve; all 6 cite keys in `d2l.bib`.
5. **Coverage: B+** — JVP/VJP vocabulary, never-form-J, Pearlmutter HVP, checkpointing, cheap-gradient principle, layout wars: all present and correct. Missing: diagonal Jacobians, vmap/batching, ∇log det.

### Correctness issues (must-fix)
1. **Tape example: code, figure, and caption describe three different graphs.** The code (l.642,654,665) builds `y = (u*v+u)*(u*v+u)` — two duplicate subgraphs, 7 nodes; the text (629–634) and caption (636) claim a shared-r diamond (5 nodes); the figure (generator l.696) shows a third graph (unary z=q²). Fix: `r = u*v + u; y = r * r` in all three tabs (makes `r.grad` genuinely accumulate twice — the best possible demo of `+=`), and make the figure show z = q·q with a doubled edge. Gradients are correct either way (16, −16) — a consistency bug an instructor hits immediately at the whiteboard.
2. **Exercise 7 mislabels the PyTorch HVP recipe** (804–806): double-backward is **reverse-over-reverse**, not "the forward-over-reverse recipe of :eqref:`eq_mdl-hvp`" the sentence claims. The JAX half is genuinely forward-over-reverse. Use `torch.func.jvp(torch.func.grad(L), (x,), (v,))`, or present double-backward honestly as the alternative composition.
3. **Editorial leftover** at line 214: "the convention of the migrated derivations below" — rewrite-process vocabulary leaking from the legacy port.

### High-value improvements (prioritized)
1. **Make TF/MXNet tabs actually verify against their autograd** (349–357 softmax, 651–661 tape-check). The section's stated premise (19–21) and the lead-ins at 332 and 626 promise autograd verification, but those tabs do a pure-numpy shadow / closed-form check. `tf.GradientTape.jacobian` and `mx.autograd` both exist. This also fixes the per-framework slide divergence.
2. **State the diagonal-Jacobian fact for elementwise activations** — one short paragraph after :eqref:`eq_mdl-jacobian-product` (l.186): J of elementwise φ is diag(φ′(x)); ReLU's backward is a 0/1 mask; the depth-L product alternates dense Wᵢ with diagonal masks. Both Parr & Howard and CS231n treat this as core.
3. **Add a Jacobian-as-linear-map figure** to §"The Jacobian as the Best Linear Approximation" (l.63): unit circle/grid → ellipse under J at a point. The chapter already has the machinery (`mdl-cal-cov-jacobian` does exactly this in the integral section).
4. **One sentence on vmap/batched Jacobians** in "Never Form the Jacobian" (684–697): frameworks build full Jacobians by batching one-pass products (`jacrev`/`jacfwd`), and `jacfwd(jacrev(f))` gives small full Hessians.

### Coverage: add / cut (sources checked: Parr & Howard explained.ai; JAX autodiff cookbook; Baydin et al. JMLR 2018; CS231n optimization-2)
- **Add:** diagonal Jacobian (HV#2); vmap note (HV#4); an exercise deriving ∇_A log det A = A⁻ᵀ — log|det J| is used by name in `mdl-random-variables.md:717` (flows) and `mdl-eigendecomposition.md:297` (Hutchinson), and nothing anywhere derives it; optionally a one-line footnote that fwd/rev are the two extremes of Jacobian-accumulation orderings and the optimal ordering is NP-complete (Baydin §3.3).
- **Cut:** nothing — admirably free of Matrix-Cookbook table-dumping; the four identities chosen are the right four.

### Polish
1. Line 210: "the MIT matrix-calculus course" — name it (18.S096, Edelman–Johnson) or drop the appeal to authority.
2. Line 266: the step a_{kj}x_j + x_i a_{ik} = (a_{ki}+a_{ik})x_i renames j→i in the first term only; spell out the dummy-index renames.
3. Recap slide (892–893): "full gradient at the cost of one extra forward pass" contradicts the body's "small constant multiple (typically 2–4×)" (550) and collides with checkpointing's "one extra forward pass" (736). Align.
4. Line 722: "Exercise 4 already extends the dual numbers" — Ex. 4 adds `__pow__`/`log`, unrelated to the HVP construction; point to Ex. 7.
5. Lines 736–738: treeverse is the O(log L)-memory scheme; calling the √L scheme "the optimal trade-off going back to treeverse" conflates Chen et al.'s point with Griewank's. Attribute precisely.
6. Naming drift: text uses r, figure uses p/q/z, code prints y — unify once must-fix #1 lands.
7. Line 557: "Baur–Strassen" cited via the Griewank–Walther monograph; cite Baur & Strassen (1983) directly if a bib key exists.
8. `%matplotlib inline` in all four imports cells is dead weight — this section never plots.

## mdl-integral-calculus.md — verdict: MINOR WORK

Well-architected: the FTC proof is correct and intuition-first, the n-D change-of-variables theorem now carries the C¹-diffeomorphism + det≠0 hypotheses a prior review demanded (525–531, with the polar-seam caveat at 583–587), Fubini gets the absolute-integrability hypothesis plus the correct ±π/4 counterexample (verified analytically), all five figures exist in the house generator, and the Monte Carlo exercise is now backed by a real compute cell. But adversarial checking turns up four genuine correctness defects, each surgically small, plus a downstream chapter citing this section for integration by parts, which it never states.

### Scorecard
1. Clarity/correctness of proofs: **B−** — FTC, Fubini, Jacobian all sound, but four checkable defects survive in a chapter that is otherwise careful about hypotheses.
2. Diagrams: **A−** — all 5 referenced figures exist; only deviation is the inline 3-D wireframe (below).
3. Teachability: **B+** — every cell computes/verifies something; but the improper cell undermines its own lesson and the FTC check has an off-by-one.
4. Structure: **A−** — 4 top-level sections; "Integration Meets Probability" hands off density/expectation/MC exactly where `mdl-random-variables.md` picks them up.
5. Coverage: **B** — right-sized for DL vs 18.01/Owen, except integration by parts is promised downstream and absent.

### Correctness issues (must-fix)
1. **Line 79–80: false claim.** "For absolutely integrable f … the limit exists and does not depend on how the slices are chosen" — false (Dirichlet function). Correct hypothesis is bounded + (piecewise) continuous. One-clause fix.
2. **Lines 331–361 + 328–329: improper-integral cell contradicts itself.** The left-Riemann sums print 1.000455, then 1.000500 (identical in all 4 frameworks' committed outputs) — the partials *pass* 1 and settle at 1.0005 (the ε/2 left-rule bias), yet the next line prints "limit as b -> infinity = 1.000000" and the prose says "watch [it] approach its limit of 1". Print the exact partials 1−e^{−b}, or use the midpoint rule, or acknowledge the bias.
3. **Lines 719–723 + 741–745: rate claim mismatches the chapter's own rule.** N^{−2/d} is the rate for a *second-order* rule (Owen ch. 2), but the chapter only ever demonstrates the first-order left rule (error ∝ ε ⇒ N^{−1/d}, crossover d=2 not d=4). Also the plot is degenerate: `N**(-2/4)` ≡ `N**-0.5`, so the "grid d=4" curve lies *exactly on top of* the MC curve — the prose claim "the two cross near d=4" (751) is wrong about what is drawn. Use d = 2/6/10 or distinct constants.
4. **Lines 577–605 → 654–656: the √π bridge is missing.** The section computes ∬e^{−x²−y²} = π, then asserts "since ∫e^{−x²}dx = √π" with no connecting step. The factorization (∫e^{−x²}dx)² = ∬ — one line, itself a showcase of the just-proved Fubini — is never stated. Also 578–581 inverts the classical motivation (it's the 1-D integral that resists direct attack).
5. **Broken downstream promise: integration by parts.** `mdl-distributions.md:568–570` cites this section for it; ch.27's score-matching (Hyvärinen identity) also leans on it; this section mentions parts only in a passing clause (382). Add a 4–6 line statement + one example (∫₀^∞ u e^{−u}du = 1, exactly what the cite needs).

### High-value improvements (prioritized)
1. **Surface cell (431–482) violates "code teaches, does not draw":** four duplicated copies of raw-matplotlib 3-D wireframe plumbing drawing an *illustrative* bell surface. Pre-generate as `img/mdl-cal-bell-surface.svg` (mdl-figure skill) and keep only the 3-line Riemann-volume computation.
2. **'volume (exact = pi)' label (441):** the box-truncation justification exists but sits at 602–605, attached to the *later* Gaussian cell — the prior review's mislabel is only half-fixed. Move/duplicate the one-liner to the surface cell.
3. **FTC-check off-by-one (250–252 + 3 tabs):** `dFdx[i]` equals `f(x[i+1])` exactly, but is compared against `f(x[:-1])`, so the printed "max |dF/dx − f| = 0.000999" measures only the grid shift. Compare against `f(x[1:])` and the error collapses to float roundoff — far more striking for the same line count.
4. **MC error column floored by a biased reference (732–737):** errors measured against the left-rule quad (0.74686 vs true 0.746824), so at n=10⁵ the printed "error=0.00003" is mostly the reference's bias. Use a midpoint reference or note it.

### Coverage: add / cut (sources checked: MIT OCW 18.01SC syllabus; Owen, *Monte Carlo theory, methods and examples*, ch. 1–2)
- **Add: integration by parts** (must-fix #5).
- **Add (optional, one paragraph): differentiation under the integral sign** — swapping ∇_θ and ∫ underlies policy gradients/Fisher information later; not in any other MDL chapter (grepped).
- **Rightly omitted:** partial fractions, trig substitution, arc length, line integrals/Green's theorem.
- **Cut: nothing.** Owen's framing (dimension-free σ/√n vs n^{−k/d} grids) is faithfully represented once must-fix #3 lands.

### Polish
1. Line 811–813: JAX Discussions tab duplicates TensorFlow's thread t/1093.
2. Line 422: "hopeless by direct summation" overstates — soften to "opaque to rectangle-summing".
3. Line 750: "the grid lines fan out and flatten" — revise alongside must-fix #3.
4. Slide "Integration meets probability" (871–882): writes `∫e^{-x^2}=√π` without dx, and attaches `@!integral-density` rather than `@!integral-gaussian`.
5. Line 836 vs 868: placeholder styles `@!integral-…` and `@integral-surface` are mixed — confirm intentional.
6. FTC proof (201–206): squeeze is run for ε>0 only; a parenthetical "(the ε<0 sliver is symmetric)" closes it at zero cost.
7. Improper-integral prose (311): "infinite, in which case it diverges" — divergence-by-oscillation is silently excluded; "fails to settle" is more honest.

---

# Chapter 25 — Probability & Statistics

## mdl-random-variables.md — verdict: MINOR WORK

A strong, mostly correct section: the density thought-experiment is honest and well-hedged, every algebraic derivation checked by hand and with numpy is right (running 3-atom example: Var = 8p ✓; covariance grid: Cov = 4p−2, ρ = 2p−1 ✓; 4xy independence ✓; Cauchy divergence ✓; the Cauchy ∞−∞ argument is correctly stronger than "infinite mean"), the tower/Eve proof is clean, and change-of-variables correctly *cites* `eq_mdl-change_var_nd` rather than re-deriving det-as-volume. Two genuine defects keep it from READY: the Chebyshev sharpness claim is false for the inequality *as stated*, and the prose claims the Riemann-sum cell prints total mass 1 while the committed output prints 0.9954.

### Scorecard
1. **Clarity/proofs: B+** — proofs short and correct, but the section's deepest result (Chebyshev) is the only one *asserted without proof*, and its equality case is wrong as stated.
2. **Diagrams: A−** — all 7 figures wired, house style; MVN contours correctly live in distributions and are forward-referenced; but covariance/correlation figures are near-duplicates whose claimed contrast is invisible.
3. **Teachability: B+** — both code cells genuinely compute; 7 well-graded exercises; but the entire Several Variables half (≈550 lines) has zero code.
4. **Structure: A−** — clean 3-section hierarchy; change-of-variables is misfiled under "Several Variables".
5. **Coverage: A−** — matches Blitzstein–Hwang ch. 5/7/8/9/10, Wasserman ch. 2–4, MML §6.2/6.3/6.7; Markov's inequality is the one standard omission.

### Correctness issues (must-fix)
1. **Chebyshev equality case is false for the stated form.** Line 416 states P(X∉[μ−ασ, μ+ασ]) ≤ 1/α², i.e. strict |X−μ|>ασ. At p=1/8 the atoms a±2 sit **exactly on** the interval endpoints, hence inside the closed interval, so the mass outside is **0**, not 1/4. The claim "hitting the bound with equality" (429–430), the figure caption (435), the generator panel title "p=1/8 (tight)" (`tools/gen_mdl_probability_figures.py:417`), and the slide (1023–1027) are all wrong as written. Fix: state Chebyshev in the standard ≥ form throughout — then p=1/8 gives equality exactly as claimed. *(Re-verified in main thread.)*
2. **Prose contradicts the committed output.** Line 200: "The total mass comes out to 1" — but the cell integrates over [−5,5) and the committed store prints `total mass : 0.9954` (~0.0046 of the N(3,1) component lies above 5). Widen the grid (e.g. [−8,8] → 1.0000 to 4 d.p., then re-capture) or rewrite the prose.

### High-value improvements (prioritized)
1. **Prove Chebyshev** (410–421). House style is proofs-first, and the section proves trivialities while asserting its only deep inequality. Markov ⇒ Chebyshev is 3 lines and gives Markov for free.
2. **Add one compute cell to the Several Variables half.** Candidates: Monte-Carlo verification of Var(X+Y)=VarX+VarY+2Cov (800), or push X∼N(0,1) through e^X and compare a histogram against the log-normal density derived at 727–738 — making the change-of-variables formula *computed*, not just stated.
3. **Make the covariance-vs-correlation figure contrast real.** The text's whole point (875–880: covariance tracks units/scale, correlation doesn't) is invisible: the generator's covariance panels have nearly the same spread as the correlation panels. Rescale one panel dramatically (dollars→cents, cov ×100, same ρ) or merge into one two-row figure.
4. **State discrete LOTUS once.** E[g(X)]=Σ g(xᵢ)pᵢ is used silently at 361; the continuous version exists upstream but the discrete rule is never written.
5. **Re-house "Change of Variables for Densities"** (686–738): it concerns a single transformed variable yet interrupts the dependence storyline (conditional → tower → *[CoV]* → covariance → correlation).

### Coverage: add / cut (sources checked: Blitzstein & Hwang ToC; Wasserman All of Statistics ch. 1–4; MML ch. 6)
- **Add: Markov's inequality.** Wasserman makes ch. 4 *Inequalities* (Markov → Chebyshev → Hoeffding) and B&H ch. 10 likewise; Markov is the 2-line parent and is reused in info-theory/statistics. The one real gap against all three references.
- **Add (one sentence): quantile function** F⁻¹ named explicitly where inverse-transform sampling is teased (264–267).
- **Correctly deferred:** LLN/CLT (in distributions), named distributions, MGFs (rightly skipped for DL), MSE⟺Gaussian (owned by maximum-likelihood).
- **Cut: nothing.** Cauchy, Eve's law, and the normalizing-flows log-det pointer all earn their space.

### Polish
1. Label naming: `fig_mdl-marginal` (548, 557) breaks the `fig_mdl-prob-*` pattern of the other six figures.
2. `eq_mdl-cdf-deriv` (246) uses hyphens where every sibling label uses underscores.
3. JAX imports cell (49–56): `import jax` and `import numpy as np` both unused.
4. `tf.pi = float(tf.acos(0.) * 2)` (46) monkey-patches the tf module; a local constant is cleaner.
5. Log-form display (717) uses **x** without binding **x** = g⁻¹(**y**) inside the display.
6. PSD argument (905–909) invokes the *two*-variable `eq_mdl-var_sum` for an n-term variance; add "(applied inductively)".
7. Exercise 5 (951): "without using :eqref:`eq_mdl-exp_linear` on the deviation" is confusing — linearity is still needed to expand E[(aX+b)²]; reword.
8. Riemann cell prints `P(-2 < X <= 3) : 0.7725` vs exact 0.7731; a one-clause "(exact value 0.7731)" would model error-awareness.

## mdl-distributions.md — verdict: MINOR WORK

A strong section: every derivation checked (Bernoulli/binomial/uniform/exponential/Laplace moments, Poisson limit, Gaussian normalizer, MVN conditional/Schur, exp-family moment property, Beta posterior arithmetic) is analytically correct and verified numerically; all 10 `:numref:` targets resolve; all 4 figures exist with committed generators. The catalog-then-unify structure mirrors Bishop ch. 2 and earns its length. What keeps it from READY: a technically wrong maximum-entropy claim (the base measure h appears from nowhere), a figure whose caption contradicts its own rendering, an overclaim that "every distribution above" is exponential-family (the uniforms are not), and a cluster of house-convention violations (mid-cell scipy imports).

### Scorecard
1. **Correctness/proofs: A−** — all moment derivations, the Poisson limit, MVN properties, and conjugate arithmetic check out; docked for the max-ent base-measure slip and the exp-family membership overclaim.
2. **Diagrams: B+** — four pre-generated figures in one consistent style; but the family-tree caption contradicts the rendered SVG, and the section has zero computed teaching plots.
3. **Teachability: A−** — tight per-distribution template, lecture-able in ~2 sessions; the two uniform cells teach almost nothing, and a reference table (law, pmf/pdf, mean, var, DL role) is missing.
4. **Structure: A** — catalog → exponential family → conjugate priors is exactly Bishop's arc and the right one.
5. **Coverage: B+** — DL core complete; mixture models absent from the whole chapter; natural one-line hooks (negative binomial, Gumbel-max, heavy-tail pointer) left dangling.

### Correctness issues (must-fix)
1. **Max-entropy claim wrong as stated** (966–972): "maximizing H[p]=−∫p log p subject to E[T]=τ has … solution p ∝ h(x)exp(ηᵀT)". With plain Shannon entropy the solution has **no h**; counterexample: max-ent on {0,1,…} with fixed mean gives the *geometric*, not the Poisson (h(k)=1/k!). Wainwright & Jordan (§3.1) define the problem *relative to a base measure*. Fix: one clause — "maximizing entropy *relative to the base measure h*" (or set h≡1 and note the generalization).
2. **Figure caption contradicts the figure** (line 23 + `tools/gen_mdl_probability_figures.py:478–520`): caption says "Every node lies inside the exponential-family envelope", but in the rendered SVG the Beta/Gamma/Dirichlet boxes sit *outside* the dashed envelope. Since the text proves the conjugate priors are exponential families (1158–1161), enlarge the envelope (preferred — it reinforces the punchline) or restrict the caption. Also: Dirichlet drawn below the data tier while the caption says "above"; only 2 of the 4 claimed prior→likelihood arrows are drawn.
3. **"Every distribution above … is a special case"** (908): the list silently drops both uniforms — correctly, because U(a,b) with unknown endpoints is *not* exponential-family (support depends on the parameter). Say so explicitly; the classic counterexample an instructor will reach for.

### High-value improvements (prioritized)
1. **Mid-cell scipy imports violate the one-imports-cell rule** (322, 346–347, 356, 415, 437, 448). Hoist or drop — the binomial pmf has an elegant recursion `pmf[k+1]=pmf[k]·(n−k)/(k+1)·p/(1−p)` that teaches more.
2. **No computed teaching plot anywhere** despite `%matplotlib inline` in all four imports cells. A single short `d2l.plot` overlaying Binomial(n,λ/n) pmfs on the Poisson for growing n would *show* the section's central claim and is house-legal.
3. **Trivial cells**: the discrete-uniform (249–279) and continuous-uniform (502–532) cells only print plugged-in formulas. Compress, and add a closing **summary table** (law, pmf/pdf, mean, variance, DL role).
4. **Close the over-dispersion loop**: 409–411 names over-dispersion as the Poisson's failure mode, and 1147–1150 introduces Gamma–Poisson — but never says the Gamma–Poisson mixture *is* the negative binomial. One sentence.
5. **Seeding is inconsistent**: mxnet/categorical (194), discrete-uniform (252), binomial (328), Poisson (421) use unseeded global `np.random` while sibling cells use `default_rng(0)`/`PRNGKey(0)` — captured outputs irreproducible on re-execution.

### Coverage: add / cut (sources checked: Wainwright & Jordan 2008 §3.1–3.3; Murphy PML-1 ch. 2–3; Bishop PRML ch. 2)
- **Add (1 paragraph): mixture models.** Bishop §2.3.9 and Murphy §3.5 treat mixtures right here; the whole chapter never defines a GMM. The largest genuine gap. (Pairs with the part-level ELBO/EM recommendation.)
- **Add (2 sentences): what is *outside* the family.** Cauchy/Student-t live outside, which is why they lack conjugate priors and clean convex NLLs.
- **Add (optional, 1 sentence): Gumbel-max** — how one samples/differentiates through a categorical, given the softmax emphasis (159–171).
- **Cut: nothing.** Ten laws is the right size; compress the two uniforms rather than cutting.

### Polish
1. Line 396–397: "Writing X∼Poisson(λ)." — sentence fragment.
2. Lines 368–373: bus-stop motivation overloads p as a rate (possibly >1) inside "Bernoulli(p/n)"; use λ throughout.
3. Line 806 "positive-definite" vs. 816 "PSD" for the same Σ — the density (812) needs PD; pick one.
4. Line 464: "heavy-tailed-symmetric" — Laplace tails are exponential, not heavy-tailed in the standard sense; say "heavier-tailed than the Gaussian".
5. Prose calls α−1,β−1 the pseudo-counts (1117) but the slide (1362–1364) calls α,β the pseudo-counts; reconcile (mode vs. mean view).
6. The mxnet tabs are pure NumPy — fine for a math appendix but worth an explicit decision.
7. TF binomial cell (343–352) samples before defining the pmf and names the sample `m` inconsistently; reorder to match siblings.
8. TF and JAX discussion links are identical (1253, 1257: both t/1099).

## mdl-maximum-likelihood.md — verdict: MINOR WORK

The mathematical core is sound and verified: the coin MLE factorization (154–157), the NLL⟺cross-entropy chain (343–357), Gaussian-NLL⟺MSE (406–419), the Bernoulli Fisher information 1/(θ(1−θ)) (509–527), and the MAP λ=σ²/τ² arithmetic (584–599) are all correct; the gradient-descent cells numerically converge to 0.97131 in exactly 100 iterations at float32 as claimed, and the variance simulation reproduces 0.001039 vs the CR floor 0.00105. The loss⟺noise-model table a prior review asked for exists (428–434). What keeps it from READY: a proof sentence that is false on the section's own running example, a slide that silently drops its figure, a cross-reference that promises the Cramér–Rao bound where it is never stated, and consistency hypotheses that are not honest about identifiability — plus a missing Beta-prior MAP that a sibling section already advertises as living here.

### Scorecard
1. **Clarity/correctness: B+** — every headline derivation checks out, but one proof sentence is wrong under repeated outcomes and the consistency argument oversells a heuristic as "formal."
2. **Diagrams: B** — both figures on-style and legible, but the KL slide reference is broken and there is no likelihood-surface/curvature figure to ground the Fisher-as-curvature story.
3. **Teachability: A−** — GD-on-NLL cells genuinely teach, coin threaded end-to-end, 8 derivation exercises; no hands-on simulation exercise for asymptotic normality.
4. **Structure: A−** — clean 5-section arc, but estimator theory is mis-shelved under "Minimizing a Loss."
5. **Coverage: B+** — CS229/Murphy core all present; missing MLE equivariance, Beta-MAP on the coin, and a misspecification one-liner.

### Correctness issues (must-fix)
1. **:345** — "each xᵢ carries empirical mass p̂_data(xᵢ)=1/n" is false whenever values repeat; on the section's own coin, p̂(H)=9/13. The displayed identity is fine (regrouping), but the justification sentence in a labeled **Proof** is wrong. Fix: "each *observation* contributes mass 1/n, so outcome x receives total mass n_x/n."
2. **:798** — slide `@fig:mdl-mle-kl` cannot resolve: `gen_slides.py` looks up `img/auto/<id>.svg` and neither variant exists, so the "Why it works" slide renders with a hole. Fix both: rename to `@fig:mdl-prob-mle-kl` *and* copy the SVG into `img/auto/`.
3. **:472–474** — "the Cramér–Rao bound of :numref:`sec_mdl-statistics`" overpromises: mdl-statistics.md:91 explicitly says "We will not need the bound itself" and never defines Fisher information or states the bound. Either own the statement here or state it there.
4. **:444–451** — hypothesis honesty: "minimized *exactly* at the truth" needs **identifiability**, and "This is the formal content of consistency" passes off a pointwise-LLN heuristic as the proof (uniform convergence is the real ingredient — Wasserman Thm 9.13). One honest sentence fixes it; identifiability is DL-relevant (permutation symmetry makes neural nets non-identifiable), worth saying.

### High-value improvements (prioritized)
1. **Add Beta-prior MAP on the coin** to :numref:`subsec_mdl-map` (after 599): θ̂_MAP=(n_H+a−1)/(n+a+b−2) = pseudo-counts. Closes the running example, and `mdl-naive-bayes.md:121` already *forward-promises* it. Bonus: that naive-Bayes line mis-attributes add-one smoothing to a *uniform*-prior MAP — deriving it here surfaces and fixes that.
2. **MLE equivariance** (Wasserman §9.6): one sentence at 628–629 where MAP's *non*-invariance is already discussed — the contrast is free pedagogy and currently half-told.
3. **Misspecification one-liner** at the KL-projection paragraph (373–377): under misspecification the MLE converges to the KL-closest p_θ (White 1982).
4. **Promote "Why Maximum Likelihood Works" (439) + Fisher (483) to their own `##` section** — estimator theory, not part of the loss-equivalence story it currently nests under.
5. **Averaged-loss footnote** near 597–599: with the per-example *mean* loss, the correspondence is λ=σ²/(nτ²) — which quantifies "regularization matters most when data is scarce" (617–618).
6. **Optional figure**: a 2-parameter Gaussian (μ,σ) NLL contour with the curvature ellipse ∝ I⁻¹ — the picture the Fisher prose (503–505) is describing verbally.

### Coverage: add / cut (sources checked: Wasserman ch.9; Murphy PML ch.4; CS229 main notes)
**Add:** equivariance, Beta-MAP, misspecification sentence (above). **Rightly cut** for DL: method of moments, delta method, sufficiency, EM*, Newton/IRLS — no objection (*but see part-level recommendation: ELBO/EM may land here as a new subsection). Excess: none; length and ownership of the loss-equivalences are appropriate.

### Polish
1. :175–176 "no characteristic-polynomial to factor" — wrong term (that's eigenvalues); say "no polynomial to factor."
2. :490 Σᵢ s(θ)=0 — summand has no i; write s(xᵢ;θ).
3. :644–665 sample size switches from n to N for the Continuous Variables section only.
4. :80–82 "uninformative prior, e.g. … equally likely any value in [0,1]" — flat ≠ uninformative under reparameterization, the very issue raised at 628–629; hedge with "flat (uniform)".
5. :64 "most probable parameters" via P(θ|X) — for continuous θ this is a density mode; never remarked.
6. :443–447 "which the proposition above shows is minimized at the truth" — the minimization-at-truth comes from the CE=H+KL decomposition, not the proposition; reword.
7. Exercises (699–726) — all eight pencil-and-paper; add one simulation (histogram √n(θ̂−θ*) against N(0,θ*(1−θ*))).
8. :471 "halving it costs four times the data" — correct; consider noting per-flip information is *minimized* at θ=1/2: fair coins are the hardest to pin down.

## mdl-naive-bayes.md — verdict: MINOR WORK

A strong section: the Bayes-rule derivation, naive factorization, log-space argument, and affine/linear-classifier observation are all correct (the estimation/prediction code verified numerically — recovers true Bernoulli parameters and classifies perfectly on synthetic data from the model; the committed outputs record 0.8427 test accuracy in all four frameworks, matching the "around 84%" prose). The previously-missing figure now exists (`img/mdl-prob-naive-independence.svg` — clean two-panel star-graph schematic matching its caption). Two genuine technical errors remain — the Laplace-smoothing-as-MAP claim is wrong as worded and contradicts the MAP definition in the section it cites, and the text-classification smoothing remark conflates the Bernoulli and multinomial event models — plus the second figure the prior review asked for (generative-vs-discriminative) was never added.

### Scorecard
1. **Clarity/correctness: B+** — derivations and code verified correct; one real statistical error (smoothing≠MAP under uniform prior) and one event-model conflation.
2. **Diagrams: B** — the independence schematic landed and is good; the requested generative-vs-discriminative schematic is still absent — line 59's dense tradeoff paragraph carries it unaided.
3. **Teachability: A−** — MNIST is the *right* vehicle here: the `P_xy` template plot makes the naive assumption literally visible (180–187), and line 208 explicitly positions text/spam as the canonical home; exercises (3) are thin.
4. **Structure: A** — 3 top-level sections, ~240 content lines, earns its slot as the chapter's "probability becomes an algorithm" capstone; merging into MLE/statistics would bury it.
5. **Coverage: B+** — gen-vs-disc discussion present and accurate vs. Ng–Jordan/Mitchell/CS229; missing continuous-feature story and event-model distinction.

### Correctness issues (must-fix)
1. **Line 121: "This is no ad-hoc patch: it is the MAP estimate under a uniform Dirichlet (here Beta) prior"** — false. Under Beta(1,1) the posterior mode (MAP, exactly as `mdl-maximum-likelihood.md:556–565` defines it) is the *unsmoothed* MLE; (n_iy+1)/(n_y+2) is the posterior **mean** (Laplace's rule of succession), equivalently the MAP under Beta(2,2). Verified numerically (mode 0.3 vs mean 1/3 for 3/10). Fix: "posterior mean under a uniform prior" or "MAP with one phantom observation per outcome, i.e. a Beta(2,2) prior". Also echoed in the Summary (216) and the "Training is counting" slide (276). *(Re-verified in main thread.)*
2. **Lines 121 vs 208: event-model conflation.** The "+|V|" smoothing is for the **multinomial** event model, but the "below" at 208 describes the **Bernoulli** presence/absence model (binary features, +1/+2). Either name the multinomial model explicitly or drop the +|V| forward-reference.

### High-value improvements (prioritized)
1. **Add the generative-vs-discriminative figure**: two-panel "model p(x|y)p(y) + Bayes" vs "model p(y|x) directly", or Ng–Jordan crossing learning curves, in `tools/gen_mdl_probability_figures.py` — line 59 is the densest paragraph in the section and the natural anchor.
2. **One paragraph on continuous features.** Both references cover Gaussian NB / discretization; the section thresholds pixels by fiat (125) without ever saying per-feature Gaussians are the standard alternative.
3. **Strengthen exercises** (currently 3): add (a) derive p(y|x) from the affine scores and recognize the softmax — the NB↔logistic-regression bridge, completing line 113's "same hyperplanes" claim; (b) vary the pseudocount (0, 1, 10) and report MNIST accuracy.
4. **Fix the overloaded n**: line 47 classes, line 85 classes, line 119 training examples. Use K or C for class count.

### Coverage: add / cut (sources checked: Mitchell "Generative and Discriminative Classifiers" draft ch.; Stanford CS229 main notes ch. 4)
- **Add:** continuous-feature remark; multinomial vs Bernoulli event-model naming. Both one-paragraph additions.
- **Present and correct (keep):** parameter-count argument, chain-rule collapse, zero-count hazard, Ng–Jordan sample-efficiency asymmetry, NB-is-linear.
- **Cut:** nothing. GDA proper is rightly out of scope. *(Part-level reviewer suggests reframing the section as the Bayes-rule/log-space payoff or folding into 25.3 — author's call; this reviewer finds it earns its slot.)*

### Polish
1. Line 98 "underflows to a hard zero": in float64 the winning class's product can survive (~1e-300); hedge to "underflows for most classes, making the comparison meaningless".
2. Line 142: pytorch tab downloads MNIST to `root='./temp'` — repo convention is `data/`; `./temp` litters the chapter dir.
3. Lines 36–43: the JAX imports cell imports TF and numpy but no JAX — a JAX reader executes zero JAX in this section; add a one-line note that the model is framework-free counting.
4. Line 237: jax Discussions link duplicates tensorflow's; all three IDs point at threads for the retired appendix version.
5. Line 119: "a matrix P_xy" is a (10, 28, 28) array in the code.
6. Lines 89–94: the Proposition/Proof formalizes what line 55 already derived in-line; could be slimmed.
7. Right panel of the figure: the x₃–x_d dashed arc passes through the "⋯" ellipsis glyph; nudge in `fig_naive_independence()`.
8. Line 4 "the simplest thing that deserves the name 'learning'" — contestable (the chapter's own mean-template rule is simpler); consider softening.

## mdl-statistics.md — verdict: MINOR WORK

Well-structured; the proofs (bias-variance decomposition, n−1 unbiasedness) are correct and intuition-first, the t-vs-z treatment in the CI example is unusually careful, and the per-framework `ddof` claims all check out. All numeric claims verified: σ²/n=16/30≈0.53; power n≈8 vs ≈78,500; 1−0.95¹⁰≈0.40; exercise 7's t₍₁,₀.₇₅₎=1.0 exactly right. But three things keep it from READY: the in-code decomposition demo contradicts its own "holds to numerical precision" framing (ddof=1 leaves a visible 5×10⁻⁵ gap where ddof=0 is exact to machine epsilon, and exercise 3 asserts the false converse); the bias-variance→generalization bridge overstates ("MSE, which is the expected test error" drops the irreducible-noise term, and "literally the same picture" carries no double-descent caveat even though the cited generalization.md:442–446 warns larger models can generalize better); and the hypothesis-testing section — which promises "the framework behind A/B tests and benchmark comparisons" — contains zero code and never computes a p-value.

### Scorecard
1. Clarity/correctness of proofs: **B+** — both proofs correct and elegant; the ddof-exactness mismatch and test-error conflation are real but localized.
2. Diagrams: **A−** — five consistent generator-built figures, no inline drawing; missing only a power curve. Dartboard unneeded — the sampling-distribution figure covers it better.
3. Teachability: **B** — estimator/variance/CI/bootstrap code genuinely teaches; HT subsection is code-free; exercises strong except ex. 3's flawed claim.
4. Structure: **A** — four `##` sections; all `:numref:` targets resolve; outputs captured for all 4 frameworks.
5. Coverage: **B+** — matches Wasserman ch. 6–11 core and Stat 111 scope; Bonferroni present; FDR and a worked test absent.

### Correctness issues (must-fix)
1. **l.145 + l.220–250 + l.530 (ex. 3): exactness claim vs ddof=1 code.** With ddof=0 the empirical identity MSE = bias² + var is exact to 1e−16; with the code's ddof=1 the printed pair differs at the 5th significant digit. Exercise 3 claims "the identity holds regardless of which variance estimator you plug in" — backwards. Fix: ddof=0 in `#statistics-verify-decomposition` (the plug-in variance is the right object; the ddof=1 lesson already has its own subsection), rewrite ex. 3 to ask *which* choice makes it exact and why.
2. **l.138: "their sum---the MSE, which is the expected test error."** Expected test error = bias² + variance + irreducible noise σ². Equating them silently is wrong in the very reading ("θ̂ as a fitted model") the paragraph instructs.
3. **l.136–141: no overparameterization/double-descent caveat.** "literally the *same picture*" conflicts with the cited section's own warning. One caveat sentence + cite (Belkin et al. 2019) is mandatory for a DL book in 2026.
4. **l.476: misattributed citation.** "introduced by Bradley Efron in 1979 :cite:`Efron.Hastie.2016`" — add `Efron.1979` (Ann. Statist.) to d2l.bib and cite it here.

### High-value improvements (prioritized)
1. **Add one worked hypothesis test in code** (after 384). A ~10-line untagged permutation test (or paired test on per-example correctness) on two simulated model-accuracy vectors: computes an actual p-value, reuses the bootstrap cell's resampling idiom, needs no figure. The single largest gap vs Wasserman ch. 10 and CASI.
2. **Power curve figure.** l.364's striking numbers (8 vs ~80,000 samples) beg for a power-vs-n figure; would give "Significance and Power" its only visual.
3. **Name FDR / Benjamini–Hochberg in one sentence** at 383: model-selection sweeps at ML scale are exactly CASI's large-scale-testing use case; m=hundreds makes Bonferroni hopeless.
4. **Connect exercise 1 to bootstrap failure.** Ex. 1 already uses θ̂=max for Unif(0,θ) — the canonical statistic where the bootstrap *fails*. One added sub-question teaches the method's main limitation, currently unmentioned.

### Coverage: add / cut (sources checked: Wasserman ch. 6–11; Efron & Hastie CASI ToC; Harvard Stat 111 topics)
- **Add:** worked test in code; FDR mention; one sentence contrasting confidence vs Bayesian *credible* intervals (the chapter has MAP/priors, so the hook exists); optionally name Hoeffding at l.132 (the chapter never goes sub-Gaussian — see part-level concentration recommendation).
- **Cut:** nothing — at ~675 lines the section is lean; scope choices (no Cramér–Rao proof, no LRT, no delta method) are reasonable and explicitly flagged (l.91), though see 25.3 must-fix #3: if MLE keeps citing the CR bound here, it must be stated somewhere.

### Polish
1. l.339: "H_A is its negation" — too strong; one-sided/composite alternatives aren't negations.
2. l.364: "detecting that a mean-zero, variance-one Gaussian actually has mean near 1" — self-contradictory phrasing; say "testing H₀: μ=0 when the true mean is 1."
3. l.132: "by Chebyshev's inequality" — it's Markov applied to (θ̂−θ)² (Chebyshev centers at the mean, and θ̂ may be biased).
4. l.213 + l.462: JAX reuses `jax.random.PRNGKey(0)` in two cells — same stream; split keys.
5. l.551: JAX discussion link duplicates TensorFlow's (t/1103).
6. l.296/305: `print(f'true variance         = 4')` — f-string with no placeholder; padding misaligned.
7. l.376: rejection region described only via the figure's symmetric case; one clause noting it's the α=0.05 two-sided z region.
8. l.482: add practical B guidance (≈200 for SE, ≥1000–2000 for percentile CIs, per Efron & Tibshirani).

---

# Chapter 24 — Optimization (stub; write plan)

Verdict: MAJOR WORK — 0% prose written; but the outlines are unusually good (each
subsection callout fixes flow, key results, figures, worked examples, draft
exercises). A strong skeleton, not a blank page.

### What exists (per file, with line refs)
- `index.md` (45 lines): finished intro, toc, complete curated reading list with live links — Boyd, Nocedal&Wright, Nesterov, Wright&Recht, EE364a, CMU 10-725, Bottou/Curtis/Nocedal, Distill momentum. Publishable as-is.
- `mdl-gradient-based-optimization.md` (121): real intro prose; "Planned — outline only" banner; 7 subsection outlines §3.1.1–3.1.7; 1 figure live (`fig_mdl-opt-gd-bowl-vs-valley`); zero code cells.
- `mdl-convexity.md` (161): intro; 6 outlines; 3 figures live (`convex-vs-nonconvex-set`, `chord-above-graph`, `local-equals-global`). Zero code cells.
- `mdl-constrained-optimization-duality.md` (150): intro; 6 outlines; **0 of 3 planned figures built**.
- `mdl-numerical-stability-conditioning.md` (182): intro; 5 outlines; **0 of 2 planned figures built**.
- `tools/gen_mdl_optimization_figures.py` exists, generates the 4 existing `img/mdl-opt-*.svg`; 7+ planned figures missing. The outlines faithfully implement `feedback/math-appendix-detailed-toc.md:158–252`.
- Cross-refs sound; bib keys `Nesterov.2018`, `Polyak.1964` present.
- **Outline errata:** (a) `mdl-numerical-stability-conditioning.md:41` claims bfloat16 ε=2⁻⁸ — it is **2⁻⁷** (verified: `torch.finfo`); (b) the Welford update at 92–94 is garbled (conflates the mean and M2 recursions); (c) stale "§3.x" numbering baked into callout titles and cross-refs throughout (chapter is 24 now).

### Coverage gap analysis (sources checked: Stanford EE364a; Berkeley EECS127; Nocedal & Wright 2e ToC; Bubeck monograph; EPFL OptML CS-439 2026; Higham ASNA ch.1; edge-of-stability literature)
Planned coverage is solid on the EE364a/EECS127 core. Gaps a 2026 instructor would notice:
1. **Smooth *non-convex* GD rate missing** — min_k‖∇f(x_k)‖² ≤ 2L(f₀−f*)/k is a 3-line corollary of the descent lemma and the only guarantee that literally applies to deep nets. Must add.
2. **Subgradients never defined**, yet ℓ₁/hinge convexity is taught. One definition + picture suffices.
3. **Projected GD absent** — the intro promises "projected/clipped updates" but no algorithm appears. Add projection-is-nonexpansive + projected GD to the duality section.
4. **Fenchel conjugate absent** — needed downstream: ch.26's planned Donsker–Varadhan dual has no home without it. The lse ↔ negative-entropy pair unifies softmax, max-entropy, and the dual function. Add one subsection (Boyd §3.3).
5. **Backward vs forward error missing** from conditioning — Higham's central organizing idea. One proposition ("forward ≲ κ × backward").
6. **Edge of stability**: §3.1.4 already derives the 2/L ceiling; a one-paragraph remark that real nets train *at* sharpness ≈ 2/η (Cohen et al. 2021) is cheap, current, and distinguishes this from a 1999 textbook. Implicit bias (GD on least squares → min-norm) as a remark/exercise. PL condition already planned (good).
7. Honesty nuance: heavy-ball's √κ rate is quadratic-only (fails globally on general strongly-convex f — Lessard et al.); the outline states it without caveat.
Rightly pointer-level: interior-point mechanics, trust region, mirror descent, Frank–Wolfe, CG, SQP.

### Overlap map vs chapter_optimization (ch.12)
Ch.12 currently has **zero** back-references to MDL — the planned deferral edits are all outstanding.

| Topic | Ch.12 location | Owner | Deferral |
|---|---|---|---|
| Convex sets/functions/Jensen/local=global | `convexity.md:48–295` (near-total duplicate) | **MDL 24.2** | Slim ch.12 §12.2 to a 1-page primer + numref, or keep code demos only |
| Lagrangian, penalties, projections | `convexity.md:296–356` | **MDL 24.3** | Ch.12 keeps the teaser, numrefs MDL for KKT/duality |
| 1-D/2-D GD demos, learning rate | `gd.md:10–238` | **Ch.12** (how-to) | MDL shows no "watch GD run" pedagogy, only rate verification |
| Newton, convergence analysis, preconditioning, line search | `gd.md:239–366` | **MDL 24.1** (math) | Ch.12 keeps the demo; MDL owns the claims |
| SGD convex convergence proof | `sgd.md:192–277` | **MDL 24.1** | Ch.12 keeps dynamic-LR practice + experiments |
| Momentum quadratic/eigen analysis | `momentum.md:285–347` | **MDL 24.1** | Ch.12 keeps leaky-average view + implementations |
| Optimizer zoo, minibatch systems | ch.12 rest | **Ch.12** | MDL: one "Adam = diagonal preconditioner" pointer |

### Write plan (per section)
**Split verdict:** 4-section split is right; keep names. One structural fix: GD-before-convexity means §3.1.3's convex/strongly-convex rates can't be proven where stated. Recommend: keep file order; make 24.1 fully self-contained on **L-smooth + quadratics** (descent lemma, non-convex stationarity rate, eigenmode analysis); move the smooth-convex O(1/k) and strongly-convex linear-rate *proofs* into 24.2's "Why Convexity Matters" as the payoff. ~800–950 lines/section.

**24.1 Gradient-Based Optimization** — `##` Descent Directions / `##` GD and Smoothness (descent lemma; non-convex stationarity rate; 2/L ceiling + EoS remark; Armijo) / `##` The Quadratic Model and κ / `##` Momentum and Acceleration / `##` Stochastic Gradients / `##` Coda: Why Not Newton. Propositions (5): steepest descent via Cauchy–Schwarz; descent lemma; min‖∇f‖² = O(1/k); optimal-step contraction (κ−1)/(κ+1); minibatch unbiased + Var ∝ 1/b (heavy-ball √κ and Nesterov O(1/k²) *stated* with caveat). Code: per-mode contraction on diag(1,10) measured vs predicted; η-sweep table; backtracking; GD-vs-momentum-vs-Nesterov iterations-to-tol vs κ; SGD variance-vs-b + fixed-η plateau vs 1/k decay on a logistic toy; Newton one-step. Figures: existing bowl-vs-valley + new `mdl-opt-momentum-damping`, `mdl-opt-sgd-noise-ball`.

**24.2 Convex Sets and Convex Functions** — `##` Convex Sets / `##` Three Lenses (+ define subgradient, ~15 lines) / `##` Jensen / `##` Why Convexity Matters (local=global; **the two rate theorems proved here**; unique min) / `##` Recognizing Convexity (+ `###` The Convex Conjugate: lse ↔ neg-entropy) / `##` Reality Check (non-convexity, PL, implicit-bias paragraph). Propositions (6): intersection convex; three-lens equivalence; Jensen (⇒ KL≥0); local=global; the two rates; lse convex via Hessian = softmax covariance. Code: numeric three-lens check; Jensen demo; GD from many inits on convex vs double-well (basin histogram); lse Hessian PSD check; PL toy linear convergence. Figures: 3 exist; reuse.

**24.3 Constrained Optimization and Duality** — keep 6 subsections; insert `###` Projections and Projected GD. Propositions (6): ∇f ∥ ∇g from no-feasible-descent; projection nonexpansive; dual always concave; weak duality; Slater ⇒ strong duality (statement + geometric sketch, cite Boyd); λ* = −∂p*/∂b. Code: simplex projection via sort+threshold with KKT residuals checked numerically; SVM dual on 2-D toy solved by **projected gradient ascent on the dual** (avoids cvxpy — not in `pyproject.toml`); water-filling via bisection on μ; duality-gap demo on a tiny non-convex problem. Figures: build the 3 missing SVGs.

**24.4 Numerical Stability and Conditioning** — keep 5 subsections; add `###` Backward and Forward Error. Propositions (5): softmax shift-invariance; stable-lse exactness; ‖δx‖/‖x‖ ≤ κ‖δb‖/‖b‖ (via SVD, short); κ(AᵀA)=κ(A)²; ridge κ monotone-decreasing in λ. Code: finfo table fp16/bf16/fp32 (fix the 2⁻⁷); overflow → stable softmax; **from-logits vs from-probabilities CE — the one genuinely 4-framework-divergent cell**; Welford (corrected) vs naive variance; Hilbert-matrix digits-lost vs log₁₀κ; joint plot of κ and GD iterations vs λ. Figures: build the 2 missing SVGs.

### Risks / open decisions
1. **Rate-proof placement** (GD § vs convexity §) — decide before writing.
2. **Ch.12 deferral edits are a separate change to the main book** — touches frozen, output-captured notebooks; scope deliberately.
3. **No cvxpy** — verify KKT/duals with closed forms + scipy.
4. **Figure naming drift**: outlines plan `fig_mdl-momentum-damping` etc. without the `mdl-opt-` prefix; standardize before generating.
5. Outline errata (bfloat16 ε, Welford, §3.x, heavy-ball caveat) — fix while writing.
6. Modern-theory dosage (EoS, implicit bias): one remark each, no proofs.
7. Most code is framework-agnostic; only the CE-from-logits cell and finfo table genuinely exercise the 4 tabs — decide policy.

---

# Chapter 26 — Information Theory (partial; rewrite + write plan)

Chapter 26 = one written-but-legacy section + two high-quality planning stubs.
Figures generator `tools/gen_mdl_infotheory_figures.py` exists with 3 of ~14
needed figures done.

### mdl-information-theory.md assessment — verdict: rewrite candidate
The old d2l appendix with surgical modern patches (named Gibbs proof 543–547, Gaussian-KL closed form 557–668, differential-entropy caveat 142, semicolon note 379, removal of the old `abs()` KL demo) plus 5 "Planned" callouts. The patched math is right; the legacy skeleton, prose, and code around it are not at the LA/Calc/P&S bar.

**Correctness (spot-checked numerically):**
- **BUG — conditional-entropy example (335/347/360/373):** joint `[[0.1,0.5],[0.2,0.3]]` sums to **1.1** (not a distribution); given marginal `[0.2,0.8]` matches neither row sums `[0.6,0.5]` nor col sums `[0.3,0.8]`; broadcasting `p_xy/p_x` normalizes *columns*, and the resulting "conditionals" sum to 1.5/1.0. Every number in the cell is meaningless. *(Re-verified in main thread.)*
- **BUG — mutual-information example (418 etc.):** joint `[[0.1,0.5],[0.1,0.3]]` has row marginal `[0.6,0.4]`, but the code is fed `p_y=[[0.75,0.25]]`. Computed "MI" = 0.719 bits; true MI = **0.0074 bits** — off ~100×, and the inconsistency can even make the formula go negative.
- **Units clash:** l.38 promises base-2 "always", yet `gaussian_kl` is in nats (567) and `cross_entropy` (725) uses natural log. No bits/nats policy.
- **Dead code:** `kl_divergence` (506–536, 4 tabs) defined and never called; slide `@information-theory-definition-2` (995) shows a definition with no demo.
- **Verified correct:** card-deck bits (2 / 5.70 / 225.58); entropy example 1.685 bits; Gibbs-via-Jensen; CE = H + KL; CE example 0.9486 = `NLLLoss`; Gaussian-KL closed form matches `torch.distributions` on 4 parameter sets incl. asymmetry; the σ < 1/√(2πe) negative-differential-entropy claim.

**Bar gaps:** (1) **8** top-level `##` sections vs house 3–5. (2) One-imports-cell violated: `import math` inside 5 functions, `import optax` mid-file (871), unused imports. (3) **Cross-chapter violation:** re-derives the binary NLL (674–684) and the full multinoulli MLE⟺CE proof (791–831) — owned by `subsec_mdl-nll-crossentropy`; the planned "Modern Uses" callout would derive it a *third* time. All must become callbacks. (4) The ~270-line MI block (219–486) is flagged for migration to 26.3 but still present. (5) Legacy prose ("soul of information theory", dated "Amazon is on fire" example, typos 230/302/830); no Proposition/Proof structure except the patched Gibbs. (6) Planned `fig_mdl-self-info-curve` and `fig_mdl-bernoulli-entropy` not yet generated. (7) TF `nll_loss` "circular argument" comment (858–862) is sloppy; `from_logits=True` fed log-probs works only because softmax(log p)=p — unexplained. (8) Exercise 2 (913) now duplicates the in-body Gibbs proof.

### What exists in the two outline files
- **mdl-divergences-distances.md (368):** zero prose/code; 10 planned subsections in standard stub format (divergence axioms; f-family + generator table; Fenchel/f-GAN dual; forward-vs-reverse KL; TV+Pinsker; IPM/MMD; OT/Wasserstein+KR dual+WGAN; Fisher/score (ch.27 bridge); Stein coda; capstone divergence↔objective map). 9 figures planned, **none generated**. Plan quality is high.
- **mdl-mutual-information.md (272):** zero prose/code; 7 planned subsections (MI recap absorbing 26.1's block; nonlinear correlation; log N hardness; Barber–Agakov/DV-MINE/NWJ; InfoNCE; IB + Saxe caveat; limits/criticisms). **2 of 3 figures already generated**; explicit migration note.

### Coverage gap analysis (sources checked: Poole et al. 2019; McAllester & Stratos; Nowozin f-GAN; Oord CPC/InfoNCE; Peyré & Cuturi; Polyanskiy & Wu / MIT 6.441; Saxe et al. ICLR 2018)
The planned coverage is genuinely 2026-grade. **Missing:**
1. **Data-processing inequality** — core of P&W ch.2, in *no* outline; IB and all representation-learning claims lean on it. Biggest gap.
2. **Conditional MI / chain rule for MI** — absent; needed to state DPI honestly.
3. **Operational/coding meaning of entropy** — "lower bound on bits" is asserted (204) but never sketched (Kraft + Shannon code, à la Cover&Thomas ch.2/MacKay ch.4–5); the pervasive "extra bits" language is ungrounded without it.
4. **Jensen–Shannon defined explicitly** — the GAN row of the capstone table depends on it.
5. **Entropic OT / Sinkhorn** — Peyré–Cuturi's computational core; add a paragraph + tiny demo, or explicitly defer.
6. (Optional) **Fano's inequality** — one short proposition linking MI to achievable error would anchor 26.3.
**Excess:** none serious; rate–distortion rightly implicit via IB (say so in one line).

### Write/rewrite plan
- **26.1 rewrite (~700–900 lines, 4 `##`):** (1) *Information and Entropy* — card deck, self-info, axioms, properties; generate the 2 missing figures; (2) *Cross-Entropy and KL* — KL, **Proposition (Gibbs)**, Gaussian closed form + MC check (keep), CE, CE=H+KL; **replace 674–684 and 791–831 with callbacks to `subsec_mdl-nll-crossentropy`**; make `kl_divergence` *used* (verify CE−H=KL numerically); (3) *Coding view and Perplexity* — half-page Kraft/Shannon-code sketch, then PPL; (4) *Modern Uses* — label smoothing (derive optimal smoothed target), distillation T²-KL, MLE link as one-line callback. Move the MI block (219–486) to 26.3, fixing the two broken examples in flight; fix imports; pick a units policy; rewrite Summary; re-scope Exercise 2.
- **26.2 write (~900–1000):** content per stub (sound) but **regroup 10 `##` headings into 4**: *What is a Divergence + the f-family*; *Duality: f-GAN and Forward/Reverse KL*; *Metrics: TV/Pinsker, IPM/MMD, Optimal Transport* (+ Sinkhorn paragraph, JS defined); *Scores: Fisher + Stein + the Capstone Map*. Propositions: D_f≥0 (Jensen), f-GAN bound with tangent picture, Pinsker, KR-dual statement, 1-D W₁=∫|F_P−F_Q| (prove). Code: fwd-vs-rev KL Gaussian-fit to a mixture (two optima); unbiased MMD²; sorted-CDF W₁ vs tiny primal LP; numeric Pinsker check; χ²/Hellinger on categoricals. Trim figures 9→~6.
- **26.3 write (~800–1000, 4 `##`):** *MI fundamentals* (migrated material + Gaussian I=−½log(1−ρ²) anchor + nonlinear-correlation demo + **DPI and conditional MI as Propositions**); *Why MI is Hard* (log N, saturation simulation); *Variational Bounds and InfoNCE* (BA/DV/NWJ, InfoNCE = categorical CE callback, log N saturation experiment); *Information Bottleneck and Limits* (VIB β-sweep, Saxe caveat). Figures: 2 exist; add `infonce-pos-neg`, `ib-tradeoff`. Replace "Amazon is on fire" with a corpus-PMI mini-example.
- **3-section split: correct.** The defects are 26.1's internal state and 26.2's heading granularity, not the split.

### Risks / open decisions
1. **Bits vs nats policy** chapter-wide (recommend nats default to match DL practice; bits as a sidebar).
2. **26.1 trim + 26.3 write must land together** (one PR), or MI is duplicated / cross-refs dangle; slides referencing migrated cell IDs move too.
3. **4-framework training cells** for MINE/InfoNCE/VIB are the big authoring cost (especially MXNet); keep critics tiny and data Gaussian-pair-with-known-MI so cells stay CPU-cheap.
4. Where DPI lives (26.1 vs 26.3) — recommend 26.3, with the Markov-chain picture.
5. Discussion links are placeholders in both stub files; legacy thread IDs in 26.1.
6. The planned `fig_mdl-divergence-objective-map` is a table — make it a markdown table, not an SVG.

---

# Chapter 27 — Dynamics (stub; write plan)

**Status: 0% written prose.** All four sections are uniform planning stubs: a
real ~13-line rendered intro, then per-subsection `⟢ Planned — outline only`
callouts. Outline quality is high — most equations are already stated and mostly
correct.

### What exists (per file; correctness issues in existing math)
- `index.md` (57): **fully written** — intro, ToC, excellent curated reading list (Strogatz, Särkkä–Solin, Øksendal, MIT 6.S184, FM Guide, Luo, NCSN/DDPM/Score-SDE/FM/RF, Song/Weng/Dieleman/annotated-DDPM blogs).
- `mdl-odes-solvers.md` (383): 7 planned `##` subsections; 1 committed figure `img/mdl-dyn-ode-field.svg` (caption eigenvalues −0.5±i verified correct).
- `mdl-sdes.md` (321): 7 subsections; figure `mdl-dyn-sde-paths.svg`. Itô's lemma, QV→t, Itô isometry, OU mean/var/stationary: all **correct** (OU p_t solves FP verified by sympy).
- `mdl-fokker-planck-probability-flow.md` (314): 6 subsections; 3 figures. FP equation, PF velocity, Anderson reverse SDE: correct.
- `mdl-score-matching-diffusion-flow.md` (456): 9 subsections; no figures yet. Hyvärinen, DSM target −ε/σ, DDPM ᾱ-marginal, CFM theorem, RF, Benamou–Brenier: correct.
- Generator `tools/gen_mdl_dynamics_figures.py` produces all 5 committed SVGs (house style).

**Correctness bugs (verified with sympy/numpy):**
1. **Sign error, stated twice** — `mdl-fokker-planck-probability-flow.md:127,137`: claims ½∇·(g²∇p)=∇·(p·(−½g²∇log p)). Off by a sign; correct RHS has **+½g²∇log p** (the minus only appears after moving it into −∇·(pv)). The PF velocity at 167 is right, so a writer transcribing 127 derives a contradiction.
2. **Euler–Maruyama "strong order ½"** (`mdl-sdes.md:224–232` + ex.4): true for multiplicative noise, but every SDE in this chapter (OU/VP/VE, additive g(t)) has EM strong order **1** — numerically confirmed. The planned "measure strong order on OU" exercise would refute the text as written.
3. **"Variance-preserving" wording** (§6.2.7): must say Var(x_t)=ᾱ_t+(1−ᾱ_t)=1 for **all** t (unit-variance data), not "drift balances diffusion" (true of any stationary process).
4. Minor: DDPM forward step = EM on VP-SDE only to first order; sde-paths caption "width" of a ±2σ envelope; adjoint/CNF/Hutchinson statements all check out.
5. **§6.x numbering in rendered prose**: live intros say "§6.3 and §6.4" but the chapter renders as **27** — all such refs must become `:numref:` labels.
6. Plain-text citations lack bib keys: vincent2011, chen2018 (Neural ODE), grathwohl (FFJORD), anderson1982, lipman2022/2023, liu2022 (RF), tong2023, pooladian2023 — none in d2l.bib (only ho2020denoising, song2021score exist).

### Coverage gap analysis (sources checked: Yang Song's blog; MIT 6.S184 2025 schedule + 2026 lecture list; Stanford CS236 syllabus; Särkkä & Solin ToC; Karras EDM; Lipman FM Guide)
- **4-section split is right.** Särkkä–Solin's own sequence is exactly §1→§3; MIT 6.S184's lecture 2 is exactly the FP section. MIT teaches FM before diffusion; d2l's diffusion-first order is defensible. Keep.
- **Missing #1: Langevin dynamics.** Foundational in Song's blog (annealed Langevin, predictor–corrector) and historically the original score→sample bridge (NCSN, cited in the index). Zero mentions in any outline. A 40-line gift: FP steady state of dX=½∇log p dt+dW is p — reuses the FP section directly.
- **Missing #2: classifier(-free) guidance math.** MIT's 2026 edition added a dedicated CFG lecture; the Bayes identity ∇log p_t(x|y)=∇log p_t(x)+∇log p_t(y|x) plus the CFG extrapolation is ~30 lines of pure math. Currently deferred wholesale to the main book — under-scoped for 2026.
- **Missing #3: DDIM as a named result** — currently two parentheticals; deserves a short boxed derivation in the diffusion section.
- **ELBO route**: CS236 teaches MLE/ELBO/VAE before diffusion; this chapter takes the score-SDE-only route. Acceptable, but add one paragraph acknowledging the equivalent variational derivation (pointer: Luo 2022, already in the index).
- **Rightly out of scope**: consistency models, distillation (pointer), EDM as full topic (one remark: Heun/2nd-order, ~35 NFE), latent diffusion, architectures, discrete diffusion, stochastic interpolants (exercise-only). Nothing in the outline is excess; stiffness/implicit is the only compress-candidate.

### Dependency check (what it assumes that the part never teaches)
- **Divergence ∇·, Laplacian, divergence theorem**: the FP section's core language — zero coverage in `mdl-multivariable-calculus.md` (grep: no hits). Biggest dependency hole in the part.
- **Matrix exponential e^{At}**: §27.1's prereq block claims it comes from `sec_mdl-eigendecompositions` — that section never defines it. Fix the claim; define in 27.1 (outline already does, via the series).
- **Fisher divergence/score** and **OT**: labels exist but live in the ToC-only ch.26; worse, that OT stub plans only **W₁**/KR/WGAN while §27.4 needs **W₂ + Benamou–Brenier** — taught nowhere in the part.
- **ELBO/variational inference**: named only inside the info-theory stub; never taught. Any ELBO aside must be self-contained (see part-level action #1).
- **CLT** "callback": the prob chapter never states the CLT formally (passing mentions only). **Martingale** never defined — avoid the word or define inline. Solid: conditional expectation/tower, change-of-variables, reverse-mode AD/VJP, Gaussians — all genuinely taught.

### Write plan (per section)
**House-rule fix everywhere:** stubs use 7–9 top-level `##`; written sections must regroup into 3–5 `##` with `###` inside. Code: hand-rolled Euler/EM/RK4 + tiny MLPs only (no torchdiffeq/diffrax) so all 4 framework tabs work. Targets ~700–1000 lines each.
- **27.1 ODEs & Solvers (~900):** `##` Vector Fields, Trajectories, Well-Posedness (IVP, flow map; Picard–Lindelöf, contraction intuition; √|x|, x² counterexamples) · `##` Linear ODEs and Stability (e^{At} via series+eigendecomposition w/ 3-line proof; stability dictionary; Jacobian linearization; code: e^{At} eigen vs solver) · `##` Numerical Solvers (Euler+RK4 with order props, code: log–log slopes 1 & 4; stiffness/implicit, prop h<2/λ, code: stability sweep) · `##` Neural ODEs and the Adjoint (ResNet=Euler prop; adjoint theorem, intuition-first variational proof; code: 2-D ring→ring neural ODE trained through unrolled solver; verify adjoint grad = autograd on linear ODE) · `##` Continuous Normalizing Flows (instantaneous CoV via det(I+hJ)≈1+h·tr J, 5-line proof; Hutchinson; code: log p_t vs −t·tr A). Figures: keep ode-field; add phase-portrait gallery, uniqueness-fan, resnet-as-euler; error-vs-h as computed d2l.plot.
- **27.2 SDEs (~800):** `##` Brownian Motion (random-walk limit; code: Var=t, Cov=min) · `##` Itô Calculus (QV→t prop; Itô integral: left endpoint, zero mean, isometry 3-line proof; Itô's lemma via Taylor+(dW)²=dt; code: W² vs ∫2W dW gap = t) · `##` SDEs and Euler–Maruyama (**state additive-noise strong order = 1**, general ½; code: EM weak-order check) · `##` Ornstein–Uhlenbeck (solve via Itô on e^{θt}X; transition kernel; stationary; exact VP normalization; code: cloud vs analytic mean/band + stationary histogram). Figures: keep sde-paths; add brownian-paths, qv-convergence, ou-mean-reversion. Exercises: fix #4; add GBM −σ²/2.
- **27.3 Fokker–Planck & Probability Flow (~750):** `##` From Paths to Densities (+ self-contained "three vector-calculus identities" box: ∇·, Δ, integration by parts — closes the dependency hole locally) · `##` The Fokker–Planck Equation (derivation via Itô + parts; prop: OU Gaussian solves FP; heat-equation case) · `##` Continuity Equation and the PF-ODE (**fix the sign**; key prop via the one-line ∇p=p∇log p rewrite; likelihood corollary; code: OU SDE-cloud vs PF-ODE marginals overlay — the chapter's best demo) · `##` The Score Function (Gaussian/mixture scores; Z-invariance; code: mixture score field) · `##` Time Reversal (Anderson via Bayes-on-transition-kernels intuition; factor-of-2 remark; code: reverse SDE from N(0,·) recovers a bimodal p₀ using the closed-form mixture-under-OU score). Figures: keep 3; add density-spreading, sde-vs-pfode-marginals.
- **27.4 Score Matching, Diffusion, Flow Matching (~1000):** `##` Learning the Score (Hyvärinen w/ 1-D parts proof; DSM w/ Vincent proof — intuition: marginal score = posterior mean of conditional scores; code: DSM on 1-D mixture vs analytic score) · `##` Score-Based Diffusion (pipeline + VE/VP; DDPM: discretization prop, ᾱ-marginal, ε-loss=reweighted DSM; **NEW** Langevin & predictor–corrector; **NEW** DDIM/deterministic sampling; **NEW** guidance math, CFG identity) · `##` Flow Matching and Rectified Flow (CFM gradient-equality proof; linear path ⇒ constant velocity; reflow; code: **the** 4-framework training demo — MLP velocity field, Gaussian→two-moons, Euler sampling at 1/2/8/32 steps) · `##` Optimal Transport and Straightness (W₂+Benamou–Brenier stated **self-contained**, Jensen lower-bound proof sketch; minibatch OT-CFM remark) · `##` Sampling = Solving the Learned Dynamics (step-count vs quality experiment; EDM/Heun remark; **unifying table as a markdown table, not a figure**). Exercises: drafted 5 + Langevin stationarity + CFG-as-score-tilt.

### Risks / open decisions
1. **Fix outline bugs before delegating writing** (sign identity ×2, EM strong order, VP wording) — section-writing agents transcribe stubs verbatim.
2. **4-framework training cost**: budget exactly **one** real training loop (CFM two-moons) + one tiny 1-D score net; everything else closed-form/simulation. JAX needs `lax.scan`-style loops; decide whether MXNet tabs may simplify (cf. memory note on per-framework wording divergence).
3. **Upstream stubs land later**: §27.4 "callbacks" to Fisher-divergence/OT live in the unwritten ch.26, whose OT plan is W₁-only — either extend that plan to W₂/Benamou–Brenier or keep 27.4 self-contained (recommended). Same decision for ∇·/Laplacian: reopen the *finished* calculus chapter vs. the in-27.3 box (recommended).
4. **Numbering & refs**: purge "§6.x" from rendered prose; add the 8 missing bib keys; converge figure-plan names to the committed `mdl-dyn-*` naming.
5. **Time-axis convention clash** (diffusion 0→T noising vs FM 0=noise→1=data): pick per-section conventions and flag with one explicit callout in 27.4.
6. **Scope line to hold**: add Langevin/DDIM/CFG (small, math-only, 2026-expected); keep consistency models, EDM-as-topic, latent diffusion, discrete diffusion out (pointers only).
