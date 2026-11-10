# Linear Regression

Source: `chapter_linear-regression/linear-regression.md`, about 5,000 words of prose before the exercises (all framework tabs counted; the PyTorch reader sees a little less).

## Verdict

This is good textbook writing and in places better than that. Every abstraction has a number next to it (the 480,000 dollar house, the three-point normal equations, the corrupted label that pulls a slope from 2 to 23), and one example, the corrupted label, is carried through three subsections until maximum likelihood explains it. That thread is the kind of thing MacKay does. What keeps the section from the top mark is its load: it teaches the model, the loss, robustness, the normal equations, projection, SGD, the noise ball, vectorization, maximum likelihood, a menu of losses and the network view in one sitting, at 5,000 words, above the house ceiling. The single most important fix is to lighten the middle: move the vectorization timing out of "Minimizing the Loss", and either work the noise-ball argument in one dimension or reduce it to the picture.

All numbers in the prose were checked against the stored PyTorch outputs and by hand (22.88, 2.02, the normal-equation solution, 390x, $e^{-12.5}$) and are correct.

## Scores

| Clarity of explanation | Clarity of thought | Pedagogy | Language | Ease and enjoyment |
|---|---|---|---|---|
| 4 | 4 | 4 | 4 | 4 |

## What works

- **The worked price.** "the house of 150 square meters built 20 years ago is predicted to cost $3{,}000 \cdot 150 - 1{,}000 \cdot 20 + 50{,}000 = 480{,}000$ dollars". The weights have units (dollars per square meter, dollars per year) before they become $\mathbf{w}$.
- **The corrupted label** (under "The Squared Loss"). Twenty points on $y = 2x$, one label set to 10,000, two fits, one printed line. The example returns under "Maximum Likelihood" ("lies about $10^4$ standard deviations away") and again under the loss menu, each time explaining more. This is the best piece of teaching in the section and a model for the book: one small experiment, reused, instead of three.
- **The three-point normal equations.** The matrices are written out, solved by hand to $(7/6, -1/2)$, solved again by `lstsq`, and then the error vector is checked to be perpendicular to both columns with explicit arithmetic. The reader can verify every digit.
- **Small true remarks that reward attention**: "'normal' is an old word for perpendicular"; "Perpendicularity to the column of ones means that the errors sum to zero"; "This formula is for analysis, not for computation."
- **"A learning rate has a meaning only together with the loss it is used with."** A practical fact most texts never state.
- **The closing**, which names two open questions and so gives the next six sections a reason.
- Figures `mdl-linreg-projection.svg` and `mdl-linreg-gd-sgd.svg` both carry their explanations and are legible.

## Structural findings

- **Too much for one section.** Eleven topics in 5,000 words. Nothing is wrong, but the difficulty peaks twice (the uniqueness argument for the normal equations, the noise ball) on points that are secondary for a first reading. A reader who only needs "model, loss, SGD, why squared loss" has to pass both. Either split after "Minimizing the Loss" into two sections, or trim: see LR-6, LR-7, LR-8.
- **"Vectorization for Speed" is in the wrong place.** It sits inside "Minimizing the Loss", after the noise ball, and is about neither. Its result is needed earlier: the SGD subsection says a processor computes a matrix product "much faster than the same arithmetic one example at a time, as we measure below". Move the timing to the end of "The Linear Model", where the text first claims that a framework evaluates the predictions "with one call to a matrix routine", so that the SGD subsection can cite a result the reader has seen.
- **The house disappears.** The section opens with a house to value and its last mention is a subordinate clause under vectorization ("such as the house we set out to value"). No house is ever valued, here or anywhere in the chapter. One closing sentence that points to the real house-price data in the multilayer perceptron chapter would keep the promise honest.

## Passage-level findings

**LR-1** (moderate, explanation). Under "The Linear Model".
Quote: "$\mathbb{E}[Y \mid X = \mathbf{x}]$, the mean price of houses with features"
A capital $X$ and $Y$ appear as random variables one paragraph after the bold design matrix $\mathbf{X}$ was introduced, and are never used again except once under "The Normal Distribution". Fix: write "the average label of examples with features $\mathbf{x}$, which we write $\mathbb{E}[y \mid \mathbf{x}]$".

**LR-2** (minor, language). Under "The Squared Loss".
Quote: "costs a hundred times as much as an error of 1. The squared loss therefore"
The sentence before the colon, the example after it, and the next sentence ("penalizes large errors heavily") say the same thing three times. Keep the example: "An error of 10 costs a hundred times as much as an error of 1, so a single bad label can dominate the fit."

**LR-3** (moderate, pedagogy). Under "The Squared Loss".
Quote: "has no derivative, at zero, we use $0$; elsewhere its slope is"
To show that the absolute error is robust, the demonstration needs gradient descent (not yet reintroduced in this chapter), a hand-tuned learning rate of 0.002, 2,000 steps, and a new term, subgradient. It then prints 2.02 where the true minimizer is exactly 2, which the prose glosses as "stays near 2" and an exercise later has to explain. The machinery distracts from the point. Fix: find the minimizer of the mean absolute error by scanning a grid of slopes (`w_grid[np.argmin(...)]`), which needs no derivative and prints 2.00, and move the subgradient and the 2.02 puzzle wholly into exercise 3, where they are already treated.

**LR-4** (moderate, explanation). Under "The Squared Loss".
Quote: "know only when we generate the data ourselves. Since the training data are"
One paragraph introduces the hat convention, reserves $\mathbf{w}^*$ for a use four sections away, observes that $L$ depends on the parameters alone, links $L$ to the line model of the calculus section, and remarks on scaling a loss. Five jobs. Fix: keep the hats and the link to the earlier loss here; move the $\mathbf{w}^*$ convention to the synthetic data section, where it is first needed; the scaling remark is made properly under "Minibatch Stochastic Gradient Descent" and can go.

**LR-5** (moderate, explanation). Under "The Normal Equations".
Quote: "of linear equations. To write the gradient compactly, we first absorb the"
From this sentence on, $\mathbf{X}$ has $d+1$ columns and $\mathbf{w}$ contains the bias, through this subsection and the next. Under "Minibatch Stochastic Gradient Descent" both silently return to their old meaning and $b$ reappears. A first-time reader will not notice either switch and will be confused by the reappearance of $b$. Fix: say "For this subsection and the next, $\mathbf{X}$ and $\mathbf{w}$ denote the extended matrix and vector", and open the SGD subsection with "We return to writing the bias separately."

**LR-6** (minor, pedagogy). Under "The Normal Equations".
Quote: "With independent columns, the matrix $\mathbf{X}^\top \mathbf{X}$ is"
Three paragraphs establish that every solution is a minimum, when it is unique, and that $\mathbf{X}^\top\mathbf{X}$ is invertible, including a definition of the inverse. The Pythagoras expansion is elegant and is reused under projection, so keep it. The invertibility proof and the definition of an inverse could be one sentence for the stated reader (college linear algebra); the worked three-point example, which is what makes the idea land, would then arrive sooner.

**LR-7** (moderate, explanation). Under "Gradient Noise and the Learning Rate".
Quote: "adds noise of variance proportional to $\eta^2 / |\mathcal{B}|$, while the"
Three proportionality claims in three sentences, none derived and none with a number. "removes a fraction proportional to $\eta$ of the squared distance" is the step a reader cannot reconstruct. This is the densest paragraph of the section and it supports a secondary point. Fix, either of two. (a) Work the scalar case in three lines, in the manner of Strang: for $w \leftarrow (1-\eta)w + \eta\xi$ with noise of variance $s^2/|\mathcal{B}|$, the stationary variance $v$ satisfies $v = (1-\eta)^2 v + \eta^2 s^2/|\mathcal{B}|$, so $v \approx \eta s^2 / (2|\mathcal{B}|)$ for small $\eta$. (b) State only what the figure shows (a cloud whose width halves when $\eta$ is quartered) and leave the scaling to exercise 6, which measures it.

**LR-8** (minor, language). Under "Minibatch Stochastic Gradient Descent".
Quote: "(for draws with replacement; the loaders of this book draw without"
A five-line parenthesis with a forward reference to the optimization chapter, in the sentence that defines the central algorithm of the book. The reader cannot skip it without losing the sentence. Delete, or move to a footnote.

**LR-9** (moderate, pedagogy). Under "Vectorization for Speed".
Quote: "wait = lambda x: torch.cuda.synchronize() if torch.cuda.is_available() else None"
The first timing experiment of the book needs a per-framework `wait` helper and comments about asynchronous devices, a concept that is explained only in the next section. The lesson (one matrix product instead of a Python loop) is about the interpreter, and holds on a CPU with plain `time.perf_counter()`. Fix: time on the CPU without `wait`, or put the helper in `d2l` and call it without comment. Together with the relocation proposed above, the subsection shrinks to one cell and one paragraph.

**LR-10** (moderate, explanation). Under "Maximum Likelihood".
Quote: "data is the *likelihood*, the quantity $P(D \mid H)$ of Bayes' theorem in"
The step from "density of a label" to "function of the parameters" is the conceptual core of the subsection and gets one sentence and a pointer to chapter 1. No number accompanies it. Fix: add two lines with the three-point data already in hand. For example: "With $\sigma = 1$, the fitted line $(7/6, -1/2)$ gives the three labels a joint density of about 0.058, and the horizontal line through their mean a density of about 0.046. Maximum likelihood prefers the first." (I computed both values: $(2\pi)^{-3/2}e^{-1/12} \approx 0.058$ and $(2\pi)^{-3/2}e^{-1/3} \approx 0.046$.)

**LR-11** (minor, language). Under "Maximum Likelihood".
Quote: "the sum of squared errors divided by $2\sigma^2$. For a fixed $\sigma$,"
"For a fixed $\sigma$ ... whatever the value of $\sigma$" says it twice in one sentence. Draft: "The second is the sum of squared errors divided by $2\sigma^2$, so the minimizer over $\mathbf{w}$ and $b$ is the least-squares solution for every $\sigma$."

**LR-12** (moderate, explanation). Under "Choosing a Loss from a Noise Model".
Quote: "| counts (sales, arrivals) | Poisson |"
The table uses $\nu(\mathbf{x})$, which is defined only inside an exercise, and two of its four rows are supported by exercises alone. The symbol is undefined for a reader who does not do exercise 5. Fix: add "where $\nu(\mathbf{x})$ is the predicted rate" to the sentence that introduces the table, or write the loss in words.

**LR-13** (minor, thought). Under "Choosing a Loss from a Noise Model".
Quote: "corrupted. The same principle covers noise whose level depends on the"
A single clause about input-dependent noise ("by predicting $\sigma(\mathbf{x})$ alongside the mean") that the section neither explains nor uses. Cut.

**LR-14** (minor, language). Under "Linear Regression as a Neural Network".
Quote: "the feature dimensionality of the layer."
A term introduced in the last sentence of a paragraph and not used again in the chapter. Cut the sentence.

## Exercises

Eight problems, well graded, each tied to a passage of the text. Exercises 1 (constant model under three losses), 2 (centering), 5 (Poisson) and 6 (measure the noise ball) are excellent: each asks for a derivation or a measurement that the text prepared. Exercise 3(d), which asks why the demonstration prints 2.02, is a nice touch and should survive LR-3 (keep the gradient-descent version inside the exercise). Exercise 4(c), on prices quoted in whole cents, is vague ("Explain how this affects the log-price model") and could be cut. Exercise 8 has five parts and needs the singular value decomposition; it is marked hard, but parts (d) and (e) (noise injection equals ridge) are a different problem from (a) to (c) (minimum-norm solutions) and would be clearer as two exercises.

## Still open from October

The October items for this file are resolved: the scratch and concise learning rates are now reconciled (see the concise review for the opposite problem, over-explanation), and the passages it marked to keep (the house opening, projection) are intact.
