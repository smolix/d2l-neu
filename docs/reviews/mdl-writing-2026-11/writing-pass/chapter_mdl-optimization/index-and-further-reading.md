# Writing pass: chapter 40 introduction and Further Reading

Files: `chapter_mdl-optimization/index.md`, `chapter_mdl-optimization/further-reading.md`.
Reviews followed: `40-optimization/index.md` (IX-1 to IX-6),
`40-optimization/further-reading.md` (S1 to S3, FR-1 to FR-8), `40-optimization.md`.
Read beforehand: the current openings and Summaries of the seven sections and
their writing-pass logs.

## Words (checker count, whole file, against 4deb96d6)

| File | Before | After |
|:--|--:|--:|
| `index.md` | 1180 | 1173 (introduction before the toc block: 405 to 385 by `wc -w`) |
| `further-reading.md` | 1500 | 1500 |

Checker: "OK: only prose changed" for both, no em-dash warning.
`tools/lint_source.py`: clean for both. The toc block, the notation table,
the resource list and all labels are untouched.

## Introduction (`index.md`)

- **IX-1.** No longer opens with chapter 9's "Every network in this book was
  trained". It opens on the loop and on the three questions chapter 9 left
  open. The three questions are kept verbatim (review: protect).
- **IX-2.** The Cauchy 1847 anecdote is removed (chapter 9 keeps it). No
  replacement history; the words went to the route and the relation
  paragraph.
- **IX-3.** "the running example of every section" is now "of the chapter",
  and the long sentence is split after "than in the other".
- **IX-4.** The seven "asks how" sentences are replaced by a route in three
  movements with a ranking: the first two sections "are the ones to read";
  three results are named (the ceiling, the step count with momentum's
  square root, the noise floor proportional to the step); Newton's method
  "and its relatives" can be skipped; the last three sections are "best read
  on demand".
- **Split of gradient descent.** The route names
  `sec_mdl-gradient-based-optimization` only for the ceiling, the step count
  and momentum, and mentions the curvature methods by name without a
  reference, so the sentence stays right when they become their own section.
  No section numbers and no count of sections other than "first two", "next
  two", "last three". After the split "The first two sections" should become
  "The first three" (or the Newton sentence gets its own `:numref:`).
- **Relation to chapter 9.** New four-sentence paragraph: chapter 9 measures
  and uses, this chapter proves; small results in both; long derivations
  only here; the main text can be read without them.
- **Where the main text uses the material** (verified by `git grep`):
  `chapter_linear-regression/weight-decay.md:193` cites
  `subsec_mdl-weight-decay-duality` (hence `sec_weight_decay`);
  `chapter_builders-guide/numerics.md:160` cites
  `sec_mdl-numerical-stability-conditioning` (hence `sec_numerics`);
  `chapter_optimization/gd.md:57` states the ceiling $2/L$ and cites this
  chapter.
- **IX-5.** The scope paragraph now carries
  `sec_mdl-multivariable_calculus` and `sec_mdl-eigendecompositions`.
- **IX-6.** Nothing to do here (the table is right).
- **Cut to stay within length:** the sentence "The same theory explains why
  momentum and Adam help, why clipping and weight decay act as constraints,
  and why a loss can turn into `NaN`." The route now says each of these
  where it names the section. The review asked to keep the theory paragraph
  verbatim; its central sentence ("The steepest curvature sets ...") is
  verbatim, this trailing one is the only loss.

Claims added or changed in the introduction: momentum "replaces the curvature
ratio by its square root" (the $\sqrt\kappa$ law, section Summary); "a fixed
step stalls at a noise floor proportional to the step" (stochastic Summary);
"Small results, such as the ceiling $2/L$ or the Adam update, appear in
both" (checked in `gd.md` and `adam.md`); `sec_weight_decay` "uses"
constraints and multipliers "to read weight decay as a norm budget" (it
cites the correspondence, proved in Duality for $\ell_2$ only, see DUA-11).

## Further Reading (`further-reading.md`)

**Opening (FR-1, S1).** The template paragraph, including "Each topic states
the problem it addresses and what it changes relative to the chapter", is
gone. The new opening ranks: five topics bear on training networks, the rest
is classical background "for when a problem calls for them". It separates
this file from the introduction's list (that list: textbooks and courses for
what the chapter teaches; this file: what lies past it).

**Order (S1).** Within each heading the deep learning topics now come first:
edge of stability before the two classical entries; Fisher before mirror
descent; saddle-point methods before interior-point, ADMM, Frank-Wolfe. No
entry added or removed (13).

**Loose statements, verified and fixed**

| ID | Was | Now | Check |
|:--|:--|:--|:--|
| FR-2 | refinement "of `sec_mdl-numerical-stability-conditioning` converges only when $\kappa u_{32}$ is well below one" | the method is described in one sentence; "the last exercise of [the section] finds how ill conditioned $\mathbf A$ may be before it fails" | The section has refinement only as its last exercise (no `:exlabel:`, so it is named by position); the old sentence gave away part (a). |
| FR-3 | error "$p_k(\mathbf A)\mathbf e_0$", "degree $k$", "In high dimension no method ... beats" | minimizer $\mathbf 0$ and spectrum in $[\mu, L]$ stated; "degree at most $k$"; "about $2\rho^k$, with the $\rho$ ... of heavy ball"; the lower bound holds "when the dimension $d$ is large compared with $k$" | Recomputed: gradients are $\mathbf A\mathbf x_j$, so $\mathbf x_k = p_k(\mathbf A)\mathbf x_0$, $p_k(0)=1$; $1/T_k((\kappa+1)/(\kappa-1)) = 2\rho^k/(1+\rho^{2k})$. Agrees with the lower-bound sentence of the gradient-descent section. |
| FR-3 | CG "picks the best polynomial for the actual eigenvalues" | CG "finds at every step the polynomial that minimizes $f$ for the actual $\mathbf A$ and $\mathbf x_0$" | CG minimizes the energy norm over the Krylov space, which is $f$ here. |
| FR-4 | $(1-\mu/L)^2$ quoted as a new number | "the section quotes the exact $(1-\mu/L)^2$, which halves the step count" | Convexity now states this itself (its "The proof gives away a square" paragraph); here it is one clause, per the chapter review ("tell the rates once"). |
| FR-5 | "K-FAC's is the Fisher" | "K-FAC ... approximates the Fisher matrix"; softmax/squared-loss equality stated first, then the general condition with `sec_mdl-exponential-family`; the empirical Fisher "is cheaper and is often what code calls the Fisher" | The adaptive section already states the equality for the two losses. |
| FR-6 | "constant-step SGD converges linearly, as gradient descent does" (no hypothesis) | "on a strongly convex loss constant-step SGD converges linearly (`sec_mdl-variance-reduction`)" | The stochastic section now teaches this with that hypothesis; the entry keeps only what goes beyond it (batch size, strong growth). |
| FR-7 | ADMM penalty $\rho$, clashing with the contraction factor $\rho$ in the same file | $\tau$ | Not done: the note that $s$ is Boyd and Vandenberghe's $t$ (no room). |
| FR-8 | GAN training without a label | `sec_basic_gan` (the label Duality uses) and exercise `ex_mdl-lagrangian-duality-descent-ascent` | **Disagreed in part:** I did not add $|1+i\eta|>1$, because that Duality exercise asks the reader to show exactly this. |
| S2 | undefined: Gram matrix, feedback loop, GMRES, nuclear-norm ball, natural parameter, regret | "(all inner products)"; "an unknown function inside a linear recursion"; "an iterative solver for general linear systems"; "(matrices whose singular values sum to at most a bound)"; reference to `sec_mdl-exponential-family`; "the regret defined in that subsection" | |
| works-2 | "every addition rounds the same way" (experiment not named) | "that subsection's sum of $10^7$ copies of $0.1$, where long runs of additions round the same way" | The section shows the direction changes between binades (too little below $2^{19}$, too much above), so "every" was too strong. |
| (own) | LU backward error "a modest multiple of $\rho_n u$" | "a low-degree polynomial in $n$ times $\rho_n u$" | Standard bound (Higham 2002, ch. 9). |
| (own) | edge of stability anchored at `subsec_mdl-quadratic-model` | `subsec_mdl-edge-of-stability` (label added by the gradient-descent pass) | The statement that the quadratic model does not explain the return of the sharpness is kept; the section now agrees. |

**Claims about papers softened or dropped (not verifiable here)**

- Drori and Teboulle: "Its value is the exact worst-case rate" is now "bound
  it by a small semidefinite program, whose dual solution is a proof". As
  far as I know their program is a relaxation and exactness is due to Taylor,
  Hendrickx and Glineur; the exact factor is attributed to the latter, as
  Convexity does.
- Lessard, Recht and Packard: "The method recovers Nesterov's rate" dropped.
  Kept: the fixed-size program and the heavy-ball counterexample (which the
  gradient-descent section also attributes to them).
- Ma, Bassily and Belkin: now "analyze this regime and find, for least
  squares, the batch size ..."; the unqualified linear-convergence claim is
  no longer attributed to them.
- Carson and Higham: "which lets half- and single-precision factorizations
  on GPUs deliver double-precision solutions" dropped; kept "extends it to
  far worse conditioned matrices".
- Dropped for length, not for doubt: "When to start averaging ... remain
  practical choices"; the second-order cone constraint formula; "so the
  iterates are combinations of few vertices"; "The model describes typical
  data".
- Not re-verified, left as they were: Polyak and Juditsky; Bach and Moulines;
  Vaswani, Bach and Schmidt; Nemirovski 2004 ($O(1/k)$ for averaged
  extragradient iterates); Daskalakis et al.; Higham and Mary; Damian,
  Nichani and Lee. All agree with what I know of the papers.

## Before and after (verbatim)

1. Before: "Every network in this book was trained by the same loop: compute
   the gradient of the loss, step against it, repeat."
   After: "A network is trained by one loop: compute the gradient of the
   loss, step against it, repeat."
2. Before: ":numref:`sec_mdl-gradient-based-optimization` asks how far a
   gradient step can go and how curvature sets the number of steps."
   After: "The first two sections assume exact gradients, and they are the
   ones to read."
3. Before: ":numref:`sec_mdl-stochastic-gradient-methods` asks what noisy
   gradients cost and how to remove that cost."
   After: ":numref:`sec_mdl-stochastic-gradient-methods` shows that a fixed
   step stalls at a noise floor proportional to the step, the third result
   to remember, ..."
4. Before: "We assume gradients, Hessians and eigenvalues."
   After: "We assume gradients and Hessians
   (:numref:`sec_mdl-multivariable_calculus`) and eigenvalues
   (:numref:`sec_mdl-eigendecompositions`)."
5. Before: "Each topic states the problem it addresses and what it changes
   relative to the chapter."
   After: "This section points past the chapter, and its topics are not of
   equal weight."
6. Before: "approximate a curvature matrix. K-FAC's is the Fisher"
   After: "K-FAC (:numref:`subsec_mdl-preconditioning-ladder`) approximates
   the Fisher matrix"

## Repetition

- Removed from the introduction: the Cauchy anecdote and the opener shared
  with chapter 9.
- Further Reading no longer re-teaches three things the sections now own:
  the squared factor (Convexity), the heavy-ball counterexample (gradient
  descent), linear convergence under interpolation (stochastic). Each is one
  clause with a reference.

## Deferred, and things to know

- S3 (add sharpness-aware minimization): not added; the file must not grow.
- After the split of gradient descent, adjust "The first two sections" in
  the introduction and check that `subsec_mdl-edge-of-stability` and
  `subsec_mdl-momentum-acceleration` stay in the first part.
- The refinement exercise of numerical stability has no `:exlabel:`; if one
  is added, replace "the last exercise of" here by an `:exref:`.
- The introduction's resource paragraph and list are unchanged (brief:
  resource lists stay). Their em dashes are list separators.
- Notation table: unchanged. Further Reading introduces $\tau$ for the ADMM
  penalty; nothing depends on it.
- Neither file has a slides section.
