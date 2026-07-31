# Chapter 1 style review: Preliminaries

## Scope

Reviewed every tracked Markdown source in `chapter_preliminaries`: `index.md`, `ndarray.md`, `pandas.md`, `linear-algebra.md`, `calculus.md`, `autograd.md`, `probability.md`, and `lookup-api.md`. The review covers prose, headings, mathematical exposition, captions, code and experiment explanations, exercises, and all 140 slide blocks. Line references below are to the current sources.

## Executive assessment

The chapter has a sound practical sequence: arrays and tabular data precede the mathematical tools that operate on them, automatic differentiation follows calculus, and API lookup closes with an immediately reusable workflow. Several sections already exhibit the desired style. In particular, the small dataframe pipeline, the calculus geometry, and the `discover -> inspect -> read -> verify` loop move cleanly from a concrete task to a reusable abstraction.

The chapter is not yet stylistically uniform. `probability.md` retains the older conversational, survey-like voice and delays its first experiment behind a long disciplinary preamble. `linear-algebra.md` is coherent locally but too broad for one section and uses “order” in the prose while the slides switch to “rank,” creating avoidable terminology conflict. Framework-tab repetition in `autograd.md` makes the explanation read like four lightly edited transcripts rather than one argument with framework-specific consequences. The slides are generally more compact than the prose, but some titles and takeaways use absolute slogans that are less precise than the corresponding main text.

## Scores

| Dimension | Score | Basis |
|---|---:|---|
| Writing quality | 7.4/10 | Mostly clear and professional, with localized second-person narration, filler transitions, inflated claims, and inconsistent density. |
| Explanation quality | 7.6/10 | Many good concrete demonstrations, but probability and linear algebra do not consistently maintain concrete -> conceptual -> formal -> operational order. |
| Technical quality | 8.2/10 | The mathematical core is strong; the main weaknesses are terminology, unqualified probability statements, and framework claims that need tighter scoping. |

## Chapter architecture and logical order

The top-level order is defensible and should be preserved. The opening, however, gives an inventory rather than a motivating problem (`chapter_preliminaries/index.md:4-13`). A short example—turning a small batch of records into a differentiable loss—could establish why arrays, preprocessing, linear algebra, calculus, probability, and documentation lookup belong in one chapter.

Within the chapter, the main structural problem is `probability.md`. The section begins with applications and a frequentist/Bayesian distinction (`chapter_preliminaries/probability.md:9-75`), while the coin experiment does not begin until line 115. Starting from the coin experiment would let probability, estimation, convergence, joint distributions, conditioning, and Bayes' rule arise as answers to concrete questions. The late “Discussion” introduces a second taxonomy of uncertainty after the working argument has already ended.

`linear-algebra.md` progresses sensibly from scalars to tensors, reductions, products, norms, and eigendecomposition, but it attempts to serve simultaneously as notation reference, code tutorial, and mathematical survey. Stronger signposting should distinguish the minimum operational path from optional geometric and spectral extensions.

## Section- and file-level issues

| ID | Severity | Evidence | Violated style-guide rule | Diagnosis | Concrete revision direction |
|---|---|---|---|---|---|
| C1-01 | Medium | `chapter_preliminaries/index.md:4-13`: “This chapter introduces the prerequisites...” | Begin with a problem or concrete instance before scope and roadmap. | The chapter opens as a syllabus. It names subjects but does not show the reader what failure these prerequisites prevent. | Add one compact end-to-end motivating example, then state the chapter's scope and dependency order in terms of that example. |
| C1-02 | High | `chapter_preliminaries/probability.md:9-75`; coin experiment only at `:115` | Prefer concrete -> conceptual -> formal -> operational order; each section should answer one recoverable question. | Applications, philosophy, and disciplinary definitions precede the first object the reader can reason about. The frequentist/Bayesian aside is too early and too long for its role. | Open with repeated coin tosses and the question “what can a finite sample tell us about an unknown rate?” Derive the vocabulary from the experiment; compress interpretive schools to a later note. |
| C1-03 | Medium | `chapter_preliminaries/probability.md:139-142`, `:515-521`, `:899-900`: “if you conduct... you might”; “You might already begin”; “Let's put our skills to the test.” | Avoid false intimacy, staged classroom narration, and reader-direction filler. | The narrator repeatedly predicts the reader's reactions or turns content transitions into conversation. These phrases add no dependency information. | Replace them with statements about the experiment or inference: what changes, what question follows, and why the next object is needed. |
| C1-04 | Medium | `chapter_preliminaries/probability.md:523-527`: `## The Formal Language` immediately followed by `### A More Formal Treatment` | Use descriptive headings; one section, one question. | The nested headings are synonymous and communicate neither the objects introduced nor the question answered. | Collapse them into a descriptive heading such as “From Counts to Random Variables and Distributions,” then state which informal claims will be formalized. |
| C1-05 | Medium | `chapter_preliminaries/probability.md:769`: “The joint distribution P(A,B) determines everything” | Captions must be self-contained and precise; avoid universal slogans. | The caption overclaims. A joint distribution determines marginals and conditionals only under the stated variable set, and conditioning requires a positive conditioning probability. | Say exactly what the diagram recovers: the displayed marginals and conditionals for rows with nonzero mass. Follow the figure in the main text with an interpretive sentence. |
| C1-06 | Medium | `chapter_preliminaries/probability.md:831-845`: independence is stated through `P(A | B)=P(A)` without support conditions | State assumptions before equations and qualify claim strength. | Conditional probability is not defined when the conditioning event has probability zero. The factorization criterion is cleaner and avoids this hidden condition. | Define independence first by `P(A,B)=P(A)P(B)` (or the event version), then present conditional invariance for conditioning values with positive probability. |
| C1-07 | Medium | `chapter_preliminaries/probability.md:1227-1245`: late “Discussion” introduces aleatoric and epistemic uncertainty | Preserve dependency and scope; conclusions should reconstruct the argument. | The section appears to conclude, then starts a new conceptual taxonomy with no operational use in this chapter. | Either introduce the distinction where prediction uncertainty first arises and apply it to the coin/HIV examples, or move it to the later uncertainty/generalization treatment. End here by reconstructing the probability workflow. |
| C1-08 | High | `chapter_preliminaries/linear-algebra.md:209-215` defines number of axes as “order”; slide title at `:1554` says “Rank n is just n axes” | Use notation and terminology consistently; avoid overloaded terms. | The prose deliberately avoids “dimension” ambiguity, but the slides introduce “rank” for tensor order even though matrix rank is a separate linear-algebra concept. This creates a technical collision inside the same section. | Use “order” or “number of axes” consistently in prose and slides. Reserve “rank” for the dimension of a matrix's row/column space and explicitly note framework APIs that use `ndim`. |
| C1-09 | Low | `chapter_preliminaries/linear-algebra.md:1357-1364`: “all the linear algebra that you will need...” | Match claim strength to evidence and scope. | The conclusion makes a broad promise that later chapters immediately complicate. It also addresses the reader directly. | State the bounded outcome: the section covered the tensor operations used in the next modeling chapters; name the exact topics deferred to the appendix. |
| C1-10 | Medium | `chapter_preliminaries/autograd.md:266-295`: three tabs repeat “Now let's calculate another function...” | Explain shared concepts once; use code to test claims, not to duplicate prose. | Framework-independent intent is repeated inside tabs, obscuring the only material distinction: accumulation versus replacement and explicit functional gradients. | Put the common experiment and expected derivative before the tabs. Restrict each tab to its state/accumulation semantics and interpret the observed result once after the code. |
| C1-11 | Low | `chapter_preliminaries/lookup-api.md:212-218`: “you will usually get a function and a working call in seconds” | Avoid promotional or temporally fragile claims; distinguish suggestion from evidence. | The timing and success claim is unsubstantiated and will age quickly. The verification advice is the durable idea. | Remove the speed promise. Describe coding-assistant output as an unverified candidate and make signature inspection plus a minimal test the explicit acceptance criterion. |
| C1-12 | Low | `chapter_preliminaries/ndarray.md:894-902`: “This minor inconvenience is actually quite important” | Lead with the causal explanation; remove throat-clearing. | The prose announces importance before explaining it, when the synchronization argument itself supplies the importance. | Begin with the distinct memory spaces and asynchronous execution, then state that copying avoids coordination. |

## Mathematics and notation

- The calculus section is the strongest mathematical model in the chapter. Its geometric motivation, equation sequence, and closing bridge to optimization and backpropagation (`calculus.md:457-482`) should be used as the template for other prerequisite sections.
- Probability should state domains and support conditions at the moment conditional probabilities and independence are introduced. Equations currently arrive with readable intuition, but not always with all assumptions visible.
- Linear algebra's scalar/vector/matrix/tensor progression is useful, but the “order”/“rank” split must be resolved. Shape, number of axes, vector dimensionality, and matrix rank should each have one stable term.
- Claims such as “One of the most fundamental operations” (`linear-algebra.md:672`) can usually be replaced by the downstream use: dot products compute weighted sums, similarity scores, and matrix products.

## Figures, captions, and slides

The main figures generally depict an actual relation rather than decorate the page. The calculus tangent construction and the dataframe pipeline are especially effective. Captions should preserve that precision. The probability joint-grid caption at `probability.md:769` is the clearest exception: it uses “everything” where a bounded enumeration is possible.

All 140 slide blocks were reviewed. Most slides have one visible claim, but several titles substitute compression for accuracy: “Rank n is just n axes” (`linear-algebra.md:1554`), “Everything else follows” (`probability.md:1499`), and “One table holds everything” (`probability.md:1543`). Rewrite these as claims with explicit scope. Divider slides are acceptable as navigation, but they should not fragment an already short dependency chain. Slide explanations should use the same terms as the main prose and should not introduce stronger claims than the chapter text.

## Code and experiment pedagogy

- `pandas.md` and `lookup-api.md` explain the purpose of each small run and interpret its output. Preserve this pattern.
- The coin simulation provides a useful known-process experiment, but its interpretation should distinguish observed convergence in one simulation from a probabilistic convergence result.
- In `autograd.md`, shared expectations should precede framework tabs. The tabs should expose only the semantic differences that the output tests.
- `lookup-api.md` has an excellent operational loop at lines 207-218. A one-sentence summary immediately before the exercises would make the section's close even more reconstructive.
- Framework outputs involving memory sharing, asynchronous execution, or gradient accumulation should name version/device conditions where behavior is not universal.

## Recurring artifacts

- Reader simulation: “you might,” “if you conduct,” “if you are eager.”
- Staged narration: “Let's see,” “Now let's calculate,” “put our skills to the test.”
- Importance announcements: “fundamentally important,” “minor inconvenience ... quite important.”
- Absolute slide slogans: “everything follows,” “holds everything,” “just n axes.”
- Broad conclusions that promise sufficiency rather than stating the next dependency.

## Positive patterns to preserve

- `calculus.md` starts from a geometric construction, formalizes the derivative, and returns to the operational consequence for optimization.
- `pandas.md` keeps the pipeline visible and uses a small dataset whose transformations can be checked by inspection.
- `lookup-api.md` turns documentation lookup into a memorable, testable procedure rather than a catalogue of commands.
- `probability.md`'s HIV example supplies concrete numbers and makes base-rate effects visible; it should remain, with a direct transition from conditional probability.
- Many slides now use descriptive declarative titles and small diagrams rather than generic topic labels.

## Prioritized revision plan

1. Rebuild the first third of `probability.md` around the coin experiment; move or compress the philosophy and uncertainty taxonomies.
2. Standardize linear-algebra terminology across prose, captions, code comments, and slides; reserve matrix rank unambiguously.
3. Consolidate framework-independent prose outside tabs in `autograd.md` and audit all framework behavior claims for scope.
4. Replace absolute slide slogans and second-person transitions throughout the chapter with bounded, dependency-bearing statements.
5. Strengthen the chapter introduction with one concrete motivating pipeline and revise section conclusions to reconstruct what can now be done.
6. Perform a final caption pass: each caption should identify the objects, relation, and takeaway without relying on the surrounding paragraph.

**Issue count: 12 total (2 high, 7 medium, 3 low).**
