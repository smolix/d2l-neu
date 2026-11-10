# Further Reading

Source: `chapter_mdl-calculus/further-reading.md`, about 1,400 words, no code,
no slides section.

## Verdict

A useful and mostly accurate reading guide, and a good model of the form. It
follows the order of the chapter, each topic starts from a specific result of
a specific section and says what breaks or what is left open, and nearly every
citation carries a clause saying what the work adds. A reader comes away
knowing which paper to open for which question. It stays inside the length
budget and nothing in the chapter depends on it. The faults are small: two
statements that are looser than the mathematics allows (quasi-Monte Carlo,
subgradient step sizes), three facts that the main sections already state in
the same words, and an opening paragraph that recites the house template to
the reader.

## Scores

| Dimension | Score |
|---|---|
| Clarity of explanation | 4 |
| Clarity of thought | 4 |
| Pedagogy | 4 |
| Language | 4 |
| Ease and enjoyment | 4 |

## What works

- **Each topic is anchored in the chapter.** "The one-level schedule of
  :numref:`sec_mdl-matrix-calculus-autodiff` stores $T/s+s$ values, at best
  $2\sqrt T$" and then what recursive schedules buy. I checked the anchors:
  the $\mathrm{ReLU}(x)-\mathrm{ReLU}(-x)$ example, the subgradient
  $\tfrac12$ at the minimum of $|x|$, the $2h/3$ cost factor, the $T/s+s$
  schedule, and the exercise "Second derivatives in one forward pass" all
  exist where the text says.
- **The circle example** for the failure of the mean value theorem ("returns
  to its start at $t=2\pi$, yet $\boldsymbol\gamma'(\xi)\neq\mathbf 0$ for
  every $\xi$") is the right counterexample in one sentence, and the closing
  remark on why the chapter's Taylor bounds did not need the vector version
  answers the question the reader would ask next.
- **Judgement is expressed.** "which is acceptable for small models and
  prohibitive for large ones"; "has no effect in 64-bit arithmetic, changes
  the backward pass in about half of their 32-bit experiments". This is the
  voice the guide asks for.
- **Lebesgue in five sentences** ("slices the range instead of the domain"),
  tied back to the two places the chapter needed it (the rationals, the ray
  $\theta=0$).

## Structural findings

- **Repetition with the main sections (moderate).** Three topics restate a
  sentence the chapter already contains: NP-completeness of optimal Jacobian
  accumulation with the same citation (`mdl-matrix-calculus.md:476`, "for a
  general computation graph finding the cheapest order is NP-complete");
  recursive checkpointing at $O(\log T)$ memory and $O(T\log T)$
  recomputation (`mdl-matrix-calculus-autodiff.md:705`); and the subgradient
  method converging "by shrinking its step sizes, not by descending at every
  step" (`mdl-single-variable-calculus.md:1109`). Under guide section 7.7 each
  fact has one home. Since the guide says this notebook "replaces the survey
  tails of individual sections", delete the sentences in the main sections
  and keep them here.
- **Forward gradients** is also an exercise in the automatic differentiation
  section (`mdl-matrix-calculus-autodiff.md:1328`). This topic gives the
  answer (unbiased, error $(n+1)\|\nabla\ell\|^2$). Either cross-reference the
  exercise, as the Taylor-mode topic does, or drop the result here.
- **Overlap with the chapter's resource list (minor).** The Matrix Cookbook
  and Mohamed et al. appear in both `index.md` and here, the second with
  nearly the same description. The guide says the two lists do not repeat
  each other.
- **Coverage.** The selection is sound and proportionate (four parts, eleven
  topics). The one gap a deep learning reader may notice is on the integral
  side: nothing on normalizing flows as the application of the change of
  variables, or on importance sampling, although the integral section ends by
  pointing to the latter in chapter 41. A pointer by label is enough.

## Passage-level findings

**FR-1** (minor, language). Opening paragraph. Anchor: "Each topic states the
problem it addresses and what it changes relative to".
This sentence is the house template (guide section 12.3) addressed to the
reader, who does not need to be told the rule the page follows. The same
holds, less strongly, for "Nothing else in the book depends on this section",
which is at least useful as permission to skip. Cut the first; keep the
second as "This section can be skipped without loss."

**FR-2** (minor, correctness; confident). Subgradient Methods. Anchor: "shrink,
with $\sum_t\eta_t=\infty$ and $\sum_t\eta_t^2<\infty$, and the".
The paragraph gives the step-size conditions for convergence of the iterates
and then the rate $RG/\sqrt T$ as if it followed from them. The rate belongs
to a different choice, the constant step $\eta=R/(G\sqrt T)$ (or
$\eta_t\propto1/\sqrt t$), which violates $\sum_t\eta_t^2<\infty$. Say so:
"With the constant step $R/(G\sqrt T)$, the best of $T$ iterates is within
$RG/\sqrt T$ of the minimum."

**FR-3** (minor, thought). Optimal Jacobian Accumulation. Anchor: "Frameworks
therefore offer only forward". The "therefore" claims that frameworks restrict
themselves to two modes because the general problem is NP-complete. That is
at most part of the reason (reverse mode is already optimal up to a constant
for scalar losses, which is what training needs). Write "Frameworks offer
forward mode, reverse mode and their compositions, and leave the choice to
the user."

**FR-4** (moderate, correctness; confident). Quasi-Monte Carlo. Anchor: "which
beats $N^{-1/2}$ once $N$ is large compared with".
The bound $N^{-1}(\log N)^d$ falls below $N^{-1/2}$ only when $N$ is
exponentially large in $d$: about $10^{4}$ at $d=2$, $10^{16}$ at $d=5$,
$10^{40}$ at $d=10$ (computed). "Large compared with the dimension" suggests
$N\gg d$. The honest statement is the interesting one: the bound is useless
in high dimension, yet in practice quasi-Monte Carlo often beats Monte Carlo
there, because many integrands depend mostly on a few coordinates (low
effective dimension); Owen discusses this. Rewrite along those lines.

**FR-5** (minor, explanation). Integration. Anchors: "The section quotes three
results" and "The section derives the score-function and pathwise
estimators". In a further-reading notebook, "the section" with no reference
reads as this section. Write ":numref:`sec_mdl-integral_calculus` quotes
..." in both places. The third topic has no `:numref:` at all, so a reader
who lands on it cannot tell which section derived the estimators.

**FR-6** (minor, explanation). Conservative Fields. Anchor: "For losses built
from the usual layers (definable". "Definable functions" is an undefined term
of art dropped in a parenthesis. Either gloss it in a clause ("functions
built from polynomials, exp, log and piecewise selections, which covers
standard networks") or leave the parenthesis out.

**FR-7** (minor, correctness; not checked). Conservative Fields. Anchor:
"changes the backward pass in about half of their 32-bit experiments". I
could not verify the three precision-dependent findings attributed to Bertoin
et al. offline. They match my recollection of the paper's abstract, but a
reviser should confirm against the source, since the paragraph quotes
specific proportions.

## What I verified

The Kronecker identity $d\operatorname{vec}\mathbf F=(\mathbf B^\top\otimes
\mathbf A)\,d\operatorname{vec}\mathbf X$; Rudin's Theorem 5.19 as the mean
value inequality; $O(K^3)$ for the matrix-chain dynamic program; the
checkpointing figures; the $2^k$ count for nested dual numbers and the
$O(k^2)$ product in Taylor mode; unbiasedness and the mean squared error
$(n+1)\|\nabla\ell\|^2$ of the forward gradient (from
$\mathbb E[(\mathbf g^\top\mathbf v)^2\|\mathbf v\|^2]=(n+2)\|\mathbf g\|^2$);
the dominated convergence statement; the Koksma-Hlawka rate. All citation keys
exist in `d2l.bib` and all `:numref:` targets exist. Not verified: FR-7, the
chapter numbers "15-17" in Owen, and the claim that `torch.utils.checkpoint`
and `jax.checkpoint` "apply one level per marked block" (both functions
exist).

## For the chapter review

- The notebook does what the guide asks and can serve as the template for
  the other chapters' further-reading notebooks. Its only systematic defect
  is that the main sections still carry the one-sentence versions of three of
  its topics.
- It leans on all five sections evenly, in order, which makes the chapter's
  arc visible: kinks, vector and matrix maps, automatic differentiation,
  integration.
- The division of labour with the resource list in `index.md` (resources for
  what the chapter teaches there, topics beyond it here) is clear and stated
  in both places; two entries are duplicated.
