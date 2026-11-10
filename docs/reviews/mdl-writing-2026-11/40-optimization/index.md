# Optimization (chapter introduction)

Source: `chapter_mdl-optimization/index.md`, about 400 words of introduction, then a notation table and a resource list (1,150 words in all).

## Verdict

A good chapter introduction, close to the template of the writing guide (§12.1): it starts from the training loop, poses three questions a reader of chapter 9 has actually met, states the central idea in one paragraph ("The steepest curvature sets the largest safe step, the ratio of the steepest to the flattest curvature sets the number of steps, and convexity decides whether the point where a run stops is the best one"), and closes with an honest scope paragraph. Two things keep it from being excellent. The route paragraph is seven consecutive sentences of one shape (":numref: asks how ..."), which is the defect the guide names. And the opening sentence and the Cauchy anecdote are the same as in the introduction of chapter 9 (`chapter_optimization/index.md`), so a reader who comes here from chapter 9, which is the intended path, reads the same two openers twice. The most important fix is the route: group the seven sections into three movements and say which a reader arriving from chapter 9 needs.

## Scores

| Clarity of explanation | Clarity of thought | Pedagogy | Language | Ease and enjoyment |
|---|---|---|---|---|
| 4 | 4 | 3 | 4 | 3 |

## What works

- **The three questions** in the first paragraph ("Why does the learning rate have a ceiling? Why can a step below it still make almost no progress? Why do noisy gradients keep a run from settling?"). Each is answered by a named section, and each is a thing the reader has seen happen.
- **The one-paragraph statement of the theory** (second paragraph). It tells the reader the three results before any definition, in plain words. Keep it verbatim.
- **The scope paragraph**: "Deep-network losses are neither, so for them the results are models, and each section says where its model stops." This is the candour the guide asks for (§10.2), and the two sections I read keep the promise (the Lessard counterexample for heavy ball, the PL subsection).
- **The notation table.** It is the right place to admit the overloads ($\lambda$ three ways, $\epsilon$ two ways, $f^\star$ against $f^*$), and it does. For a part that is read by lookup this table is worth its space.
- **The resource list** names the section each item accompanies and says what each adds.

## Structural findings

- **The route does not group.** Seven sections fall into three movements: unconstrained descent with exact gradients (gradient descent, convexity), noisy gradients (stochastic, adaptive), and constraints and arithmetic (constrained, Lagrangian, numerical stability). The introduction lists them one per sentence. A lookup reader also wants to know which sections stand alone; the first two do not (convexity proves rates that the first section states).
- **No reading path from chapter 9.** The first paragraph says chapter 9 "left the proofs to this chapter" but not which section proves what. One sentence would do: the ceiling and the $\kappa$ law are in the first section, the rates in the second, the noise ball in the third.
- **Division of labour with chapter 9.** The stated division (chapter 9 measures on real networks, this chapter proves) is sensible and mostly kept. It is broken in three places that the section reviews record: the edge-of-stability experiment is run in both chapters on the same 25-parameter network (GD-9), Newton's method is derived in both (GD structural), and the permutation-symmetry argument and its exercise appear in both (CV structural).

## Passage-level findings

**IX-1** (moderate, language). First paragraph.
Quote: "Every network in this book was trained by the same loop: compute the"
Chapter 9 opens "Every network in this book was trained with optimizer settings taken from a recipe". The two introductions that a reader is most likely to read back to back begin with the same seven words. The guide (§12.2) asks that two introductions not open with the same construction. Fix: open this one on what chapter 9 left unproved, for example: "Chapter 9 gave three rules of thumb and measured each: a learning rate has a ceiling, a rate below the ceiling can still crawl, and a noisy gradient keeps a run from settling. This chapter proves them."

**IX-2** (moderate, enjoyment). Second paragraph.
Quote: "In 1847 Augustin-Louis Cauchy, who wanted to compute the orbits of celestial"
The same anecdote, with the same year and the same orbits, is the history paragraph of `chapter_optimization/index.md` ("Augustin-Louis Cauchy proposed it in 1847 to solve the equations that determine the orbit of a celestial body"). Told once it is a nice fact; told twice it is a template. Fix: keep Cauchy in chapter 9, where gradient descent is introduced, and give this chapter its own origin: the theory is the subject here, so Polyak (1964, heavy ball), Nemirovski and Yudin (1983, the lower bound) or the 1963 PL inequality would each be the natural anchor and each is cited in the first two sections.

**IX-3** (minor, thought). Second paragraph.
Quote: "$\tfrac12(x_1^2 + 1000\,x_2^2)$, the running example of every section, is a"
In the first two sections the running example is quoted, not worked: `mdl-gradient-based-optimization.md` says "Most worked numbers below use the opening function" (a $\kappa = 10$ bowl), and convexity uses it only for two counts. The same sentence also runs long: the bowl, its definition, a relative clause, a consequence and a count in 50 words. Fix: "the running example of the chapter", and split the sentence after "than in the other".

**IX-4** (moderate, pedagogy). Third paragraph.
Quote: ":numref:`sec_mdl-gradient-based-optimization` asks how far a gradient step"
Seven sentences, each "X asks how/when/what ...". The guide (§12.1) records that readers skip such a paragraph. A draft in three movements:

> The first two sections assume exact gradients. :numref:`sec_mdl-gradient-based-optimization` finds the largest safe step and shows how curvature sets the number of steps; :numref:`sec_mdl-convexity` says when the point reached is the best one and proves the rates. The next two add noise: :numref:`sec_mdl-stochastic-gradient-methods` measures what a minibatch gradient costs, and :numref:`sec_mdl-adaptive-stochastic-methods` shows how an optimizer can learn one step size per coordinate, and when that fails. The last three leave the unconstrained setting. Weights confined to a set need projections and multipliers (:numref:`sec_mdl-constrained-optimization-duality`, :numref:`sec_mdl-lagrangian-duality`), and finite precision can decide the answer before the algorithm does (:numref:`sec_mdl-numerical-stability-conditioning`).

**IX-5** (minor, explanation). Scope paragraph.
Quote: "We assume gradients, Hessians and eigenvalues. The proofs assume smooth or"
"We assume gradients, Hessians and eigenvalues" does not say where a reader who lacks them should go. This is an appendix read by lookup; add the two labels (:numref:`sec_mdl-multivariable_calculus`, :numref:`sec_mdl-eigendecompositions`), which the first section cites anyway.

**IX-6** (minor, explanation). Notation table.
Quote: "| $\lambda$, $\nu$ | multipliers of inequality and equality constraints; $\lambda$ also weights the decay term"
The first section writes its ridge coefficient as $\nu$ ("\qquad \nu = 0.1," under Newton's Method), which this table reserves for equality multipliers, and the second section writes the same coefficient as $\lambda$. Fix in the sections, not here (GD-3); the table is right.

## Exercises

None, correctly.

## Still open from October

"Every opening lists questions instead of posing a problem" is fixed in the two sections I read and half fixed here: the first paragraph now poses the problem, but the route paragraph is still a list of seven questions.

## For the chapter review

- The introduction promises one running example for every section; the first two sections use four quadratics between them ($\kappa = 10$, $20$ in a figure, $100$ in the Newton cell, $1000$). Check whether later sections keep to the $\kappa = 1000$ bowl.
- The chapter and chapter 9 share an opening sentence, a history paragraph, one experiment, one derivation and one exercise. Decide per item which chapter owns it; guide §7.6 says the main text never hands a demonstration to the appendix, so duplicated demonstrations should be cut here, not there.
- Order of sections: gradient descent before convexity works (the reader meets rates before the property that proves them), at the price of one forward-stated table of rates in the first section (GD-10).
