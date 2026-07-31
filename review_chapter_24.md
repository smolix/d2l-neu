# Review of Chapter 24: Linear Algebra

## Scope

Reviewed `chapter_mdl-linear-algebra/index.md`, `mdl-geometry-linear-algebraic-ops.md`, `mdl-eigendecomposition.md`, and `mdl-svd-low-rank.md`, including all prose, proofs, code cells, figures, summaries, exercises, and slide blocks.

## Executive assessment

This chapter is mathematically ambitious and substantially stronger than Chapter 23. It usually motivates a definition geometrically, proves the important result, and checks it in code. The main weaknesses are scale and editorial control: the geometry section is effectively a small book chapter, some prerequisites are repeated at length, several polished metaphors overstate what a computation establishes, and a few sentences contain clear grammatical or mathematical-summary defects. The slide decks cover the material but often use topic labels rather than claims.

Scores (0–10): **writing quality 7.5**, **explanation/pedagogy 8.0**, **technical/logical quality 8.0**.

## Architecture and logical order

Geometry → eigendecomposition → SVD is a sound progression. Within the geometry section, however, vectors, similarity in high dimensions, a classifier case study, full matrix theory, determinants, and Einstein notation create too many independent arcs. Split or visibly stage this material: vector geometry; subspaces and least squares; matrix maps/rank/determinants; tensors. The classifier example should be presented as an application after the relevant geometry, not as a long interruption before matrix maps.

## Detailed issues

| ID | Severity | Location | Problem and violated guide rule | Concrete revision direction |
|---|---|---|---|---|
| C24-01 | Moderate | `chapter_mdl-linear-algebra/index.md:4-10` | The index lists applications but does not give the reader a central question or dependency map. “Used repeatedly” is asserted rather than demonstrated. | Add a compact map from object to question: vectors/angles for similarity, eigenspectra for repeated square maps, SVD for arbitrary maps and approximation. |
| C24-02 | Major | `mdl-geometry-linear-algebraic-ops.md:1-13` and `:909-1691` | One section carries several chapters’ worth of material. The long return from the classifier application to matrices weakens continuity and makes later cross-references hard to navigate. | Split into two or three sections, or add explicit part-level signposts with short recaps and prerequisite statements. |
| C24-03 | Moderate | `mdl-geometry-linear-algebraic-ops.md:15-47` | Four framework import blocks arrive before the geometric idea is developed. This interrupts the opening and makes the section feel notebook-first. | Delay imports until the first executable example or hide them in setup; keep the opening conceptual. |
| C24-04 | Moderate | `mdl-geometry-linear-algebraic-ops.md:83-180` | The dot-product explanation is careful but overextended: two proofs, the law of cosines, Cauchy–Schwarz, domain caveat, and high-dimensional slogan appear before the reader uses the angle. | State the result and one geometric derivation in the main path; move the analytic proof or edge-case proof to a proposition box/exercise. |
| C24-05 | Moderate | `mdl-geometry-linear-algebraic-ops.md:506-908` | “Similarity in High Dimensions” combines concentration intuition, attention scaling, and a Fashion-MNIST nearest-centroid experiment. The application is useful, but the transition among these topics is abrupt and the experiment’s learning objective is not previewed. | Begin with the question the experiment answers, state expected observations, then interpret the result. Consider moving attention scaling to the attention chapter and retaining a cross-reference here. |
| C24-06 | Moderate | `mdl-geometry-linear-algebraic-ops.md:1766-1785` | The summary is an exhaustive inventory rather than a hierarchy. Several bullets contain multiple propositions, so the decisive distinctions—range vs. null space, orthogonal vs. invertible, determinant vs. singular values—are buried. | Reduce to 5–7 conceptual takeaways and point to a notation/reference table for the rest. |
| C24-07 | Major | `mdl-eigendecomposition.md:75-95` | The unit-circle description is limited to symmetric matrices but is easy to overgeneralize; input eigenvectors need not be ellipse axes for a nonsymmetric matrix. The qualification is present, yet the prose immediately calls it the picture of eigenvectors. | Explicitly contrast eigenvectors with right/left singular vectors at the first figure: the ellipse-axis picture is an SVD picture in general and an eigenvector picture only for normal/symmetric cases. |
| C24-08 | Moderate | `mdl-eigendecomposition.md:106-135` | “For … to happen, we see” and “by finding for what λ is …” are awkward. The determinant criterion is introduced procedurally without first stating that the null space must contain a nonzero vector. | Write the logical chain in one clean sequence: nonzero solution ⇔ singular matrix ⇔ zero determinant. |
| C24-09 | Moderate | `mdl-eigendecomposition.md:192-220` | The sentences defining `W` and `Λ` have comma-splice constructions (“Let W …, be”; “so we may …, we see”). This is exactly the sentence-level friction the guide asks reviewers to remove. | Rewrite as short declarative sentences and put matrix meaning immediately before each equation. |
| C24-10 | Moderate | `mdl-eigendecomposition.md:579-744` | The non-normal/transient-amplification material is important but arrives before the spectral theorem and positive definiteness, the standard conceptual payoff for most readers. It adds a sophisticated detour before consolidating the basic case. | Move symmetric/normal matrices and the spectral theorem before the defective/non-normal caution, then present transient growth as the boundary of eigenvalue intuition. |
| C24-11 | Moderate | `mdl-eigendecomposition.md:1286-1376` | The final deep-network/random-matrix application packs spectral radius, Jacobian products, initialization, and random spectra into a dense closing arc. Several claims are broad and the assumptions are easy to miss. | Separate theorem, heuristic, and empirical observation. State independence/isotropy assumptions next to random-matrix claims and end with a bounded conclusion. |
| C24-12 | Moderate | `mdl-svd-low-rank.md:285-318` | “The SVD has no such trouble” and “there is nothing defective about it” are lively but anthropomorphic; the conceptual reason—two orthonormal bases instead of one—is the useful part and appears only later. | Lead with the two-basis distinction and use the shear as evidence, not as a dramatic contest between decompositions. |
| C24-13 | Major | `mdl-svd-low-rank.md:514-528` | The denoising paragraph gives a sharp random-matrix threshold after only an informal iid-noise model. Rectangular aspect ratio, asymptotic regime, known noise, and normalization conventions matter; “signal values stand essentially where they were” is too loose. | State the precise setting or make the passage explicitly heuristic. Distinguish noise-edge intuition from the cited optimal hard threshold and give its dimensional assumptions. |
| C24-14 | Moderate | `mdl-svd-low-rank.md:537-550` | Calling an image “a visual proof of Eckart–Young” confuses illustration with proof; the proof was algebraic above. | Say “visual illustration” and state exactly what the panels verify empirically. |
| C24-15 | Major | `mdl-svd-low-rank.md:954-981` | The summary contains a grammatical corruption: “The energy ratio is the retained-energy ratio quantifies the approximation.” It also compresses many modern methods into one overloaded bullet. | Repair the sentence, distinguish approximation error from retained energy, and separate core theorem from optional applications. |
| C24-16 | Moderate | Slide blocks at `mdl-geometry-linear-algebraic-ops.md:1835-2331`, `mdl-eigendecomposition.md:1485-1937`, and `mdl-svd-low-rank.md:1055-1394` | Many titles are topic labels (“The eigendecomposition,” “The condition number,” “Summary”) rather than one-sentence conclusions. The decks are long enough that section-divider and recap slides add substantial bulk. | Convert titles into claims and prune setup/divider/duplicate recap slides. Keep one visual or derivation per slide. |

## Math and notation

The notation is largely sound and proofs are unusually complete. Preserve the explicit nonzero-eigenvector condition, the defective-matrix example, and the full Eckart–Young argument. Audit every use of “eigenvector direction” against the symmetric/normal qualification, state norms and matrix shapes consistently, and clarify the noise-model assumptions in low-rank denoising.

## Figures, captions, and slides

Most captions are self-contained and substantially better than the recommender chapter. The strongest captions identify objects and the mathematical inference. Remove claims that a numerical image is a “proof,” and align slide titles with the guide’s one-slide-one-claim rule.

## Code and experiment pedagogy

The numerical checks usually have an explicit mathematical target. Preserve that pattern. Move repeated imports out of the narrative, state dtype choices in prose only when they affect the mathematical conclusion, and add “expected result / interpretation” pairs around the longer classifier and random-matrix experiments.

## Recurring artifacts

- Excessive scope inside a single section.
- Repeated “everything,” “simply,” and dramatic reveal language.
- Applications introduced before the reader knows what question the code answers.
- Topic-label slide titles and repeated recap material.

## Strengths to preserve

- Geometry-first definitions followed by algebraic proofs.
- Clear counterexamples: defective shear, non-normal transient growth, and ill-conditioning.
- Self-contained figure captions and meaningful code checks.
- Explicit separation of exact rank and numerical rank.
- Full proofs of central claims rather than appeals to authority.

## Prioritized revision plan

1. Repair the SVD summary defect and qualify the denoising claims.
2. Reorder the eigendecomposition section so the standard symmetric case precedes advanced failure modes.
3. Split or visibly stage the geometry section.
4. Tighten repeated proofs and setup blocks while preserving conceptual derivations.
5. Convert the slide decks from topic sequences to claim sequences.
