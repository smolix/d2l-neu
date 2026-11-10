# Calculus

Source: `chapter_preliminaries/calculus.md`. About 4,900 words of prose (6,500 in the file to the slides marker; about 300 of the prose words and 90 lines of code belong to the plotting utilities), plus 800 words of exercises.

## Verdict

The best-taught section of the chapter and the one closest to the bar. A single tiny model (predict one number $w$ for labels 1, 1, 0; then a line through three points) carries the derivative, the linear approximation, the minimum, partial derivatives, the gradient, a gradient-descent step, the chain rule and backpropagation, and every number can be checked by hand: $L(1) = 1$, $L'(1) = 2$, the step to $0.8$ with loss $0.72$ against a predicted $0.6$, the gradient $(2, 4)$ obtained three ways. I checked all of them; they are right. This is what "solve the smallest case completely, with numbers" looks like. Two things hold it back. A block of matplotlib plumbing ("Plotting Utilities") sits between the closing paragraph and the summary, which is the single worst structural defect in the chapter. And the finite-difference experiment, good in itself, interrupts the path from the definition of the derivative to the rules for computing it.

## Scores

| Clarity of explanation | Clarity of thought | Pedagogy | Language | Ease and enjoyment |
|:-:|:-:|:-:|:-:|:-:|
| 5 | 4 | 4 | 4 | 4 |

## What works

- The opening poses a question with numbers and a reason why brute force fails: "Should the parameter go up or down, and how much will the loss change if the parameter moves by $0.01$?" The question is answered explicitly later ("This approximation answers the opening question").
- The derivative is met as a sequence of printed ratios $2.3, 2.03, 2.003$, then explained by algebra ($2 + 3h$), then defined. Experiment, explanation, definition: the right order.
- The paragraph on accuracy is the best motivation for surrogate losses I have read at this level: "A small change of a parameter rarely changes any predicted class, so the accuracy stays constant and its derivative is zero wherever it exists ... Neither tells us which way to move."
- "Linear Approximation and Minima" computes the error of the linear prediction and explains it: "The gap of $0.12$ is the term $3h^2$ ... which the tangent ignores. It grows with the square of the step, which is why steps must be small." The need for a learning rate is derived from arithmetic the reader has done.
- The minimum $w = 2/3$ is identified as "the mean of the three labels". The reader learns a fact about least squares on the side.
- "Rates of change multiply. If $u$ changes twice as fast as $x$, and $z$ changes three times as fast as $u$, then $z$ changes six times as fast as $x$." The chain rule in two sentences.
- Honesty about what is not proved: "The argument evaluates each partial derivative at a point that has already moved a little, so the approximation needs partial derivatives that change continuously". This is the right use of the appendix: the text gives the argument and says exactly where it is incomplete.
- Steepest descent is derived from the cosine formula, with three named directions (along, against, perpendicular to the gradient), and the perpendicular case gives the level sets. The figure matches the text, and the remark that the negative gradient "points $45^\circ$ away from" the minimum is checked and correct.
- The three-line code cell for the forward and backward pass, followed by the cost comparison with finite differences, sets up the next section precisely.

## Structural findings

- **"Plotting Utilities" does not belong here** (CAL-9). It is the last thing the reader meets before the summary of a section on calculus.
- **Placement of "Finite Differences".** The subsection is a self-contained lesson in floating-point error with a well-read plot, and it supplies one of the two reasons frameworks do not differentiate numerically. But it sits between the definition of the derivative and the rules of differentiation, so the rusty reader goes from limits to rounding error to truncation error to the best $h$ before being reminded that $\frac{d}{dx} x^n = n x^{n-1}$. It also needs `d2l.plot`, "which we define at the end of this section". Consider moving it to the end of "Partial Derivatives", after `numerical_gradient`, where both objections to finite differences (precision and cost) can be made together and where the phrase "they also show what the opening called too slow" already lives. The Derivatives part would then run: limit, rules, linear approximation, minimum.
- **Gradient descent is taught here, in a paragraph.** The update rule, the learning rate, a worked step and the zigzag are all introduced under "Steepest Descent". That is efficient and I would keep it, but chapter 2 should then build on it explicitly instead of introducing gradient descent afresh. (Not checked here; for the chapter 2 reviewer.)
- **Length.** Without the plotting block the section is about 4,600 words, appropriate for what it teaches. "Gradients of Common Functions" is the only stretch that drags: four identities in two paragraphs with no numbers until the last line.

## Passage-level findings

**CAL-1** (minor, pedagogy). After the imports.

> from torch.nn import functional as F

The imports cell loads `math`, `nn`, `F` and (in other tabs) `gluon`, `optax`, `flax.nnx`, none of which the section uses, and a second cell imports `ticker` and `backend_inline` for the plotting block. A reader meeting the book's code for the fourth time sees a wall of unexplained imports in a section whose code is three NumPy functions. Import only what the section uses; the plumbing imports leave with the plotting block.

**CAL-2** (minor, enjoyment). Under "The Derivative as a Limit".

> The idea behind the limit is much older than the derivative. Archimedes found the area of a circle of radius $r$ by squeezing it between inscribed and circumscribed regular polygons

One of the few places in the chapter with some colour, and worth keeping. It sits between the formal definition and its geometric meaning, so it delays the secant picture, and the figure shows only inscribed polygons while the text speaks of squeezing between both. Move the paragraph after the secant and tangent figure, and either mention only inscribed polygons or say that the figure shows half of the argument (the caption nearly does).

**CAL-3** (minor, language). Under "Finite Differences".

> The relative rounding error of each arithmetic operation is at most about $\epsilon$, where $\epsilon \approx 2 \times 10^{-16}$ for float64 and $\epsilon \approx 10^{-7}$ for float32.

Fine, but the next paragraph then speaks of "an error of up to about $\epsilon / h$ in the ratio" without saying that this uses $L(1) \approx 1$. One clause ("since the loss is about 1 here") keeps the reader from generalizing a formula that holds only for this scale.

**CAL-4** (minor, explanation). Under "Finite Differences", plot interpretation.

> On log–log axes both error curves form a V.

In the stored plot the float32 curve is a plateau at 2 for all $h$ below $10^{-8}$, a noisy descent, and a short rise; it reads as a ramp, not a V, and the horizontal axis runs from $10^{-16}$ at the left to $10^{-1}$ at the right, so "on the right, where $h$ is large" needs the reader to notice the direction. Say first what the plateau is (it is explained only in the last sentence of the paragraph: "Once $1 + h$ rounds to $1$ itself, the computed ratio is $0$ and the error is $2$"), then describe the two branches.

**CAL-5** (minor, language). Under "Rules of Differentiation".

> The logarithm requires $x > 0$, as does a power with a non-integer exponent, and a negative integer exponent requires $x \neq 0$.

Three domain conditions in one sentence, directly after the table of derivatives the reader is trying to recall. Correct, and nobody will read it. Move to a footnote-like parenthesis after the rules, or keep only "$\log x$ requires $x > 0$".

**CAL-6** (minor, pedagogy). Under "Rules of Differentiation".

> The product rule has a picture: $f(x) g(x)$ is the area of a rectangle with sides $f(x)$ and $g(x)$.

A good explanation delivered as prose about a picture that is not shown. The chapter has figures for much simpler things (a Venn diagram). Either add the small rectangle figure or accept the prose; do not cut the argument.

**CAL-7** (minor, thought). Under "Steepest Descent".

> Later optimizers modify the step with momentum, per-parameter step sizes, or curvature, yet each of them starts from the gradient.

Three undefined terms in a closing sentence. "Later chapters refine this step, but each refinement starts from the gradient" says what the reader can use.

**CAL-8** (moderate, explanation). Under "Gradients of Common Functions".

> For a symmetric matrix this is $2\mathbf{A}\mathbf{x}$, and with $\mathbf{A} = \mathbf{I}$ it gives $\nabla_{\mathbf{x}} \|\mathbf{x}\|^2 = 2\mathbf{x}$.

The identity matrix $\mathbf{I}$ has not been defined anywhere in the chapter (see LA structural findings). More generally this subsection is the one place where the section leaves its running example for several lines of index manipulation: the quadratic form is differentiated in general before any instance. The fix is cheap, because the instance is on the next line: lead with the line model ("we want the gradient of $\|\mathbf{X}\mathbf{w} - \mathbf{y}\|^2$ without writing out three squares"), state the two identities needed, and check them on $(2, 4)$. The Frobenius-norm identity is not used in this section and can go to the weight decay section.

**CAL-9** (major, pedagogy). Heading and the block under it.

> ## Plotting Utilities

After the closing paragraph ("What remains is the labor ...") the reader gets 90 lines of matplotlib: `use_svg_display`, `set_figsize`, a 17-line `set_log_ticks` with a nested formatter, `set_axes`, and `plot` with its input-normalizing logic. The text itself says "none of its code is needed to follow the calculus". Then comes the Summary of a section about derivatives. This is the textbook case of code that does not teach, placed where it does the most damage to the section's ending, and it is the reader's first sustained look at the book's library code. It also introduces the `#@save` convention in passing, in a section where no reader will look for it. Move the block out of the chapter: to the Tools part, or to a short "The `d2l` Package" unit that also explains `#@save` once. Here, one sentence is enough: "We plot with `d2l.plot`, a thin wrapper around matplotlib that is listed in ...". October recommended the same move; it is still open.

**CAL-10** (minor, language). Closing paragraph.

> What remains is the labor.

A four-word paragraph beat, the only stagey sentence in the section. It is not wrong and some readers will like it. If the house style is strict about such beats, fold it into the next sentence: "What remains is the labor of deriving the Jacobian of every layer by hand, which is slow and error-prone."

**CAL-11** (minor, thought). Under "Partial Derivatives".

> in running text we also write $\partial_{x_i} f$

Introduced and not used again in the section. Cut (guide: no notation the section does not use).

## Exercises

Eight exercises, graded and purposeful. "Mean and median" is outstanding: it uses the section's own constant model to show that squared error gives the mean and absolute error the median, and ends with an outlier ("Which of them changes when the label $10$ becomes $100$?"). "Derivative of $x^x$" (the correct answer is the sum of the two wrong ones) is the sort of problem readers remember. "Gradient checking" teaches a practice every practitioner needs and connects to the float32 result of the text.

- Exercise 1, part 2 (derive the derivatives of $e^x$ and $\log x$ from the limit) is hard for the intended reader without a definition of $e$; the hint "Identify the limit involving $e$" helps. Mark it as optional or give the limit.
- No exercise uses the tangent-line prediction on a new function. One short numerical problem (predict $\sqrt{4.1}$ or the change in a loss, then compare) would exercise the idea the section leans on most.

## Still open from October

- "A sentence does not parse" (`calculus.md:95`): fixed.
- "Eight notations are given for one partial derivative": fixed; two remain, one of them unused (CAL-11).
- "Steepest descent rests on two facts that are sent to the appendix": fixed; the cosine formula and Cauchy-Schwarz are now restated in the paragraph.
- "Cut: plotting utilities (move to Tools)": still open (CAL-9).
