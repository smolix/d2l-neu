# Wave-2 report: chapter_mdl-optimization/mdl-convexity.md (24.2)

## Structure (1166 lines, was 161-line stub)

- Intro + prerequisites + one untagged numpy imports cell (`#convexity-imports`).
- `## Convex Sets` — definition, catalog (half-spaces, norm balls, simplex, PSD
  cone), **Prop: intersections preserve convexity** (+ union counterexample;
  PSD cone as infinite intersection of half-spaces), affine maps, convex hull.
- `## Convex Functions: Three Lenses` — chord / first-order / second-order,
  epigraph dictionary, **Prop: three-lens equivalence** (4-part proof), strong
  convexity (ties κ = L/μ to 24.1), `### The Subgradient` (~20 lines, ℓ₁ +
  hinge, 0 ∈ ∂f ⟺ min), cell `#convexity-three-lenses`.
- `## Jensen's Inequality` :label:`subsec_mdl-jensen` — **Prop: Jensen**
  (2-line supporting-line proof, strict-equality case), **Corollary: KL ≥ 0
  with equality iff p = q** (explicit ch. 26 forward link), AM–GM, ELBO-gap
  remark, cell `#convexity-jensen-mc`.
- `## Why Convexity Matters` — **Prop: local minima are global** (+ convex
  minimizer set, stationary ⇒ global), **Prop: uniqueness** (strict/strong),
  cell `#convexity-basins`; then the payoff: descent lemma restated from 24.1,
  **Prop: O(1/k) smooth-convex rate** (complete-the-square + telescope; bonus
  monotone-distance remark), **Prop: linear rate under strong convexity**
  (proved via strong-convexity ⇒ PL, deliberately flagging that step (ii)
  uses only PL — sets up the reality check).
- `## Recognizing Convexity and Its Limits` — 4-rule calculus (certifies hinge,
  ℓ₁, logistic, ridge), **Prop: lse convex, Hessian = softmax covariance**
  (zero eigenvalue ↔ shift invariance ↔ stable softmax of 24.4), cell
  `#convexity-lse-hessian`; `### The Convex Conjugate`
  :label:`subsec_mdl-convex-conjugate` — definition, always-convex, Fenchel–
  Young, ½‖x‖² self-conjugate, **lse ↔ negative entropy** derived both ways
  (eq_mdl-opt-lse-entropy), links to 24.3 dual-as-conjugate and ch. 26 DV/f-GAN;
  `### Reality Check` — (ab−1)² non-convex via "minimizer set must be convex"
  contrapositive, permutation symmetry, PL condition + cell
  `#convexity-pl-rate`, one implicit-bias paragraph (min-norm + max-margin).
- `## Summary` (7 bullets), `## Exercises` (8, incl. min-norm implicit-bias
  Ex. 8 and PL Ex. 7), `## Discussions` + Discourse link, 7 slides.

## Coordination labels (both created, verified unique corpus-wide)

`subsec_mdl-jensen` (referenced by 24.3 + 26.1) and
`subsec_mdl-convex-conjugate` (referenced by 24.3) now resolve. All required
anchors survive: `sec_mdl-convexity`, the 3 figure labels (reused as-is, no new
figures, no figure specs needed). 12 new eqlabels, all `eq_mdl-opt-*`, all
unique; deliberately did NOT label the restated descent lemma to avoid
colliding with 24.1's in-flight eqlabel.

## New BibTeX (cited in text; NOT added to d2l.bib per scope rules)

```bibtex
@InProceedings{	  Karimi.Nutini.Schmidt.2016,
  title		= {Linear convergence of gradient and proximal-gradient
		  methods under the {P}olyak-{\L}ojasiewicz condition},
  author	= {Karimi, Hamed and Nutini, Julie and Schmidt, Mark},
  booktitle	= {Joint European Conference on Machine Learning and
		  Knowledge Discovery in Databases (ECML PKDD)},
  pages		= {795--811},
  year		= {2016},
  organization	= {Springer}
}

@Article{	  Soudry.Hoffer.Nacson.ea.2018,
  title		= {The implicit bias of gradient descent on separable data},
  author	= {Soudry, Daniel and Hoffer, Elad and Nacson, Mor Shpigel
		  and Gunasekar, Suriya and Srebro, Nathan},
  journal	= {Journal of Machine Learning Research},
  volume	= {19},
  number	= {70},
  pages		= {1--57},
  year		= {2018}
}
```

## Verification log

- `tools/lint_source.py chapter_mdl-optimization/mdl-convexity.md` → exit 0,
  no errors, no warnings.
- All 6 cells extracted in order and executed cumulatively in `.venv-pytorch`,
  `.venv-tensorflow`, `.venv-jax`, `.venv-mxnet`: exit 0 in all four, output
  **byte-identical** (md5 3b7c70c1…) across venvs; every number quoted in
  prose matches the printout. All cells pure seeded NumPy, < 2 s each.
- Dollar-before-digit, caption-bracket, "Planned" remnants: all clean.

## Deviations

- 1166 lines vs the ~800–950 guide (sibling 24.3 is 1145): the mandated content
  (6 proposition/proof blocks, 5 demo cells with quoted outputs, subgradient +
  conjugate subsections, 8 exercises, 7 slides) did not compress further
  without cutting required items.
- PL named for Polyak/Łojasiewicz in prose (1963, no bib keys exist);
  formal cite is Karimi.Nutini.Schmidt.2016 — kept new refs to the allowed 2.
- No plots: all five demos teach via printed numbers, so the imports cell is a
  single untagged `import numpy as np` (no per-framework d2l import needed).
