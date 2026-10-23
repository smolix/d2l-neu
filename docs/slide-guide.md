# D2L Slide Guide

Scope: the `<!-- slides -->` decks of Basics, Advanced, and the Mathematics for
Deep Learning appendix. `slides.md` documents the slide grammar and build;
this guide sets the standard for content. The quality bar is the author's
MLSS 2026 deck "Efficiency in LLMs"
(<https://alex.smola.org/posts/45-mlss-efficiency/main.pdf>), for its content,
not its visual style. The review that motivated this guide is
`reviews/slide-review-2026-10.md`.

## 1. The Standard

A deck must teach on its own. A lecturer can teach the section from it without
the book, and a student who missed the lecture can learn from it afterward.
Everything on a slide is therefore defined on that slide or earlier in the
deck, and nothing depends on text the slide does not show.

## 2. Unit

- **Deck:** one per section by default, 15–25 slides.
- **Lecture:** one to three consecutive decks. Thin sections that exist for
  plumbing (data loaders, object-oriented scaffolding, concise
  reimplementations) are merged into the deck of the idea they serve.
- **Chapter opener:** a short deck from the chapter `index.md` with the map of
  the chapter, the running example, and the goals. Each later deck reuses the
  map as "you are here".

## 3. Opener

Three slides:

1. **The problem, with a number.** "The math is right; the loss is NaN."
   "A 1B-parameter model fills a 24 GB GPU before it sees a token."
2. **What you will be able to do,** as three to five things to compute,
   derive, estimate, or decide, plus the running example and the idea in one
   sentence.
3. **Scope and place:** what this deck skips, and where it sits on the
   chapter map.

## 4. Body

Each slide makes one claim.

- **The title states the claim:** "Batching Amortizes Weights but Not the KV
  Cache", "Hoeffding Certifies ±1% at 95% with 18,445 Examples". Claim titles
  are right for slides even though book headings are noun phrases. Labels
  ("Implementation", "Results"), wh-clauses, and riddles are not.
- **The body is the evidence:** a diagram, a plot, a small table, a
  back-of-the-envelope calculation with units, a derivation step, a worked
  example, or a counterexample.
- **One takeaway** in a callout, at body size or larger.
- **At most about 40 words** of prose. No paragraphs.
- **Symbols are defined on the slide** where they first appear.
- **Sources** for numbers and claims go in a small footer.

## 5. Running Examples

Each chapter carries one example with real numbers through all its decks, so
every new idea lands on something familiar. A deck that introduces a number
the running example will reuse says so. Suggested examples, from the review:

| Chapters | Running example |
|---|---|
| 1 Preliminaries | a Fashion-MNIST minibatch, (256, 1, 28, 28) float32 |
| 2 Linear Regression | the synthetic line $w^*=[2,-3.4]$, $b^*=4.2$ |
| 3–4, 6–7 | a Fashion-MNIST scoreboard: parameters, multiply-adds, accuracy, seconds per epoch |
| 5 Computation | an 8B-parameter model: does it fit? |
| 8, 12 | *The Time Machine* with BPE-1024 and a bits-per-byte scoreboard; for ch. 12 also a Mamba-2 layer against a GQA cache at 128K |
| 9 Optimization | the $\kappa=20$ valley, then TinyLM |
| 10–11 | one GPT-2-small head, then GPT-2 small (124M) |
| 13 | one RTX 4090 and a 1B-parameter GPT |
| 14–15 | FrozenLake 4×4; CartPole and Pendulum |
| 16–17 | one 2-D mixture; sprites; the three-mode mixture and Fashion-MNIST |
| 24–26 | t-shirts against trousers and one 2×2 matrix; log-sum-exp; the $\kappa=10^3$ quadratic |
| 27–29 | the coin (9 heads, 4 tails); $P=(0.6,0.3,0.1)$ against $Q=(0.2,0.5,0.3)$; an OU process on bimodal data |

## 6. Checks

Every six to eight slides, a question or an estimate the audience can work out
in a minute, with the answer revealed by a fragment ("How many examples for
±0.5% at 99%?"; "Which terms grow 4× when the hidden size doubles?").

## 7. Mathematics Decks

The grammar is the same; the evidence changes. A derivation runs over several
slides, one algebraic move per slide (at most five annotated lines), numbered
I, II, III, ending in a boxed result and one line on what it buys. A worked
example uses numbers small enough to check by hand. Rates appear as slopes on
a log–log plot, not as printed tables. Assumptions go in a footer line rather
than in hedging bullets.

## 8. Code

Code appears only when the code is the idea: the TD update, the scan combine,
the loss, the sampler, an `einsum`. At most ten lines of at most 70
characters, with no imports, seeds, setup, helpers, class boilerplate, or
plotting code. Output is trimmed to about five lines or shown as a table or
plot. Results that come from code are shown with output-only cells (`@!id`).

## 9. Closer

A check slide, then a recap that restates the deck's results with the
running example's numbers and introduces nothing new, then one line on what
comes next.

## 10. Layout

Nothing overflows the frame or is clipped. Figure text is at least 14 px at
slide size. Book directives (`:numref:`, `:eqref:`, `:cite:`), section numbers,
exercise numbers, and phrases like "the run below" never appear on a slide.

## 11. Checklist

1. Could a student learn this section from the deck alone?
2. Does the opener state the problem with a number, the goals, and the scope?
3. Is every title a claim, and is every claim backed by evidence on the slide?
4. Is the running example present, with its numbers?
5. Is there a check every six to eight slides?
6. Is every symbol defined where used?
7. Is code limited to the idea itself, at most ten lines?
8. Does the recap restate results without adding content?
9. Does anything overflow, clip, or render too small?
