# Automatic Differentiation

Source: `chapter_preliminaries/autograd.md`. About 3,250 words of prose in the PyTorch view (6,400 in the file to the slides marker), plus 550 words of exercises.

## Verdict

A very good section with the clearest line of thought in the chapter. It continues the line model of the calculus section, lets the framework reproduce the gradient $(2, 4)$ the reader derived by hand, explains the result operation by operation, takes the gradient step the reader computed by hand and lands on the same $(0.8, -0.4)$ and loss $0.4$, and then states the idea most accounts of autograd miss: "Automatic differentiation differentiates the operations that ran, not the function that the program represents." All stored outputs match the prose. Its weakness is the last third. "Forward and Reverse Mode" is correct and well explained, but for a reader in the first chapter it stacks sweeps, vector-Jacobian and Jacobian-vector products, `torch.func.vjp` and `jvp`, memory costs and Hessian-vector products onto a section that had already made its point. The main fix is to shorten that part to its conclusion (one reverse sweep gives the whole gradient; forward mode would need one sweep per parameter) and move the Jacobian-filling demonstration to the appendix or mark it as optional.

## Scores

| Clarity of explanation | Clarity of thought | Pedagogy | Language | Ease and enjoyment |
|:-:|:-:|:-:|:-:|:-:|
| 4 | 5 | 4 | 4 | 4 |

## What works

- The opening starts from what the reader has just done ("we computed the gradient of a loss by hand, and it took some care even for a model with two parameters") and asks the natural question: "Can the computer do this work for us, given nothing more than the code that computes the loss?"
- The third paragraph of the opening explains the whole mechanism in plain words before any API: the framework "records each operation that it executes", knows the derivative of each, and "walks through the record backward". A reader who stops here already has the right mental model.
- "It is the forward pass only: no derivative appears anywhere in the code." A one-line observation that makes the reader look at the code differently.
- The explanation of the backward pass through the four operations ("The sum copies this gradient to each of its three inputs, the square multiplies it entry by entry by $2\mathbf{r}$, subtracting the constant labels leaves it unchanged, and the matrix–vector product multiplies it by $\mathbf{X}^\top$") with a figure that shows values and gradients at every node. It ends with a consequence the reader will meet in practice: "This is why training needs more memory than prediction".
- The detach example is minimal and exact: same values $(0, 1, 8, 27)$, different gradients $(0, 1, 4, 9)$ and $(0, 3, 12, 27)$. "Detaching changes the graph, not the values."
- The control-flow example is analysed, not just run: $k = 2^{12}$ and $100 \cdot 2^{11}$ are derived, and then the limit is shown: at the jumps autograd returns "a number with no meaning". Showing where a tool fails, with the cause named, is what the model texts do.
- "For a loss, the comparison is lopsided." The cost argument that follows ($n = 10^8$ parameters: one reverse sweep against $10^8$ forward sweeps) is the payoff of two sections.
- The closing paragraph makes the best transition in the chapter: "A gradient tells us how to lower the loss on the data at hand. Whether a lower loss reflects a pattern that will hold on new data, or only noise in the data we have, is a question of probability and statistics".

## Structural findings

- **The section has two levels.** Up to "Disabling Gradient Tracking" it is what every reader needs before chapter 2. "Higher-Order Derivatives" and "Forward and Reverse Mode" are what a reader needs before the optimization or performance chapters. The text does not tell the reader which is which. Either say so ("The rest of the section can be skipped on a first reading") or shorten: keep "The Cost of a Gradient" (it justifies the claim made in the opening), reduce "Rows and Columns of the Jacobian" to a paragraph and the figure without the code cell, and reduce higher-order derivatives to two sentences and a pointer.
- **Framework tabs carry most of the prose.** This is the most fragmented section in the chapter: in "The Forward Pass", "Gradient Accumulation", "Detaching Values", "Disabling Gradient Tracking" and both later parts, the shared text is one or two sentences and the explanation lives in the tab. In the PyTorch view this reads well. The shared sentence before "Gradient Accumulation" ("so it matters what happens to the gradient of the previous step") is followed by nothing shared after the cell; the PyTorch tab says the important thing ("A forgotten reset is a common bug") before the reader has seen the output $(4, 8)$. Moving that tab note below the cell would let the reader see the surprise first.
- **Vocabulary.** The backward computation is called the backward pass, a reverse sweep, reverse mode, a vector-Jacobian product and backpropagation. The text does connect each new name to the old one ("a reverse sweep is a backward pass", "Backpropagation is reverse mode applied to a neural network"), which is the right way to do it, but five names is a lot. "Sweep" could be dropped in favour of "pass" with no loss.
- **Repetition of the cost claim.** "at a small multiple of the cost of computing the loss, whatever the number of parameters" appears in the closing of the calculus section, in this section's opening, twice in "Forward and Reverse Mode" and again in the Summary. State it in the opening as a promise and in "The Cost of a Gradient" as a result.

## Passage-level findings

**AG-1** (minor, language). Opening, second paragraph.

> The line model of :numref:`sec_calculus` shows what the work involves.

The paragraph restates the calculus section's derivation (errors, Jacobian $\mathbf{X}$, backward pass, the value $(2, 4)$, the cost and precision of finite differences). For a front-to-back reader it is a recap of the previous page; for a reader arriving here directly it is too compressed to help. Cut it to two sentences: the model and its gradient $(2, 4)$.

**AG-2** (minor, explanation). Under "The Forward Pass", PyTorch tab.

> `requires_grad_(True)` marks `w`; the trailing underscore is PyTorch's convention for a method that changes its tensor in place.

Good. The cell that follows prints `tensor([1., 0.], requires_grad=True)` and the text moves on. Fine. But the next tab note says the gradient "will appear in the attribute `w.grad`, which is `None` until the first backward pass", and no cell shows it. Either show `w.grad` (as the MXNet tab does) or drop the clause.

**AG-3** (moderate, pedagogy). Under "Vector-Valued Functions".

> It starts from a vector $\mathbf{v}$ with one entry per output and returns

The reader's question at this point is practical: I have one loss per example; what do I call `backward` on? The text answers with the Jacobian, a three-part identity, and the chain rule, and only then says "For per-example losses this is what we want: the gradient of the total loss." Reverse the order: first the practical statement ("a backward pass differentiates one number; for a vector of losses we differentiate their sum, or more generally a weighted sum $\mathbf{v}^\top\mathbf{u}$"), then the cell, then the identity $\mathbf{J}^\top\mathbf{v}$ for readers who go on to "Forward and Reverse Mode".

**AG-4** (moderate, pedagogy). Under "Detaching Values".

> Reinforcement learning trains models this way, with targets built from the model's own predictions.

The motivation for detaching is an example from chapter 15, described abstractly ("a value that the output of a model should move toward, but that the model itself computes"). A newcomer cannot picture it, and the two sentences about what would go wrong ("the update could lower the loss by moving the target toward the prediction instead") are an argument about a setup they have not seen. The toy example $z = ux$ that follows is clear without it. Lead with the toy example and its two gradients, then say in one sentence where the reader will need this, with a label reference.

**AG-5** (minor, explanation). Under "Disabling Gradient Tracking", PyTorch tab.

> while it records, PyTorch does not allow an in-place change of a leaf tensor that requires gradients, such as `w`

"Leaf tensor" is used without definition (exercise 2 also asks the reader to inspect `w.is_leaf`). Define it in a clause ("a tensor that we created ourselves, as opposed to the result of a recorded operation") or avoid it ("of a tensor that we marked with `requires_grad_`").

**AG-6** (minor, pedagogy). Under "Control Flow".

> A language model reads sentences of different lengths, so its computation runs for a different number of steps on each input.

A good reason, followed by a function `f` that has nothing to do with it (doubling until the norm reaches 1000, then a branch on the sign of a sum). The function is the classic d2l example and it is analysed well afterwards. One connecting sentence would help: "We use a smaller program with the same two features, a loop whose length and a branch whose choice both depend on the input."

**AG-7** (minor, thought). Under "Higher-Order Derivatives".

> Some methods need the derivative of a derivative: the curvature of a loss, for instance, or the products of a vector with the matrix of second derivatives, the *Hessian*, which some optimizers use.

The motivation is "some methods" and "some optimizers", and the subsection ends "Both values match the formulas." Nothing here is used in the Basics part as far as the section says, and the MXNet tab has no example at all. This is the most dispensable subsection of the chapter. Reduce it to two sentences inside "Control Flow" ("the backward pass is itself a computation, so it can be recorded and differentiated again") or move it to where a second derivative is first needed.

**AG-8** (moderate, pedagogy). Under "Rows and Columns of the Jacobian".

> `torch.func.vjp(errors, w0)` returns the value of `errors` at `w0` and a function that maps $\mathbf{v}$ to $\mathbf{J}^\top \mathbf{v}$.

The demonstration fills a $3 \times 2$ Jacobian that the reader already knows to be $\mathbf{X}$, using two functional-API calls with tuple-in, tuple-out conventions (`(w0,), (d,)`, `[0]`, `[1]`). The API reading costs more than the result teaches, and the conclusion drawn ("forward mode needs fewer sweeps, because there are more outputs than inputs") is the opposite of the case that matters, which the next subsection has to turn around. The figure and the two paragraphs before it already make the point. Cut the cell from the main text (it would make a good exercise: exercise "Manual tracing" already asks for the same comparison by hand).

**AG-9** (minor, thought). Under "The Cost of a Gradient".

> the two modes combine well: forward mode applied to a gradient computed in reverse mode yields a Hessian–vector product at the cost of a few gradients

An advanced fact in the tail of a sentence about memory. It is the answer to the last exercise. Leave it to the exercise.

## Exercises

Seven exercises (the first two differ by framework). "Programs versus functions" is one of the best exercises in the chapter: $|x|$ against $\sqrt{x^2}$ at $0$, and "Write a program that returns $x^2$ for every input $x$ but whose automatic derivative at $x = 0$ is $1$". It tests exactly the idea the section wants to leave behind. "Manual parameter updates" (rebind `w` instead of updating in place and see what breaks) teaches the bug every beginner hits. "Manual tracing" is the standard backprop-by-hand drill and belongs here.

- "Cost of second derivatives" is reasonable only if "Forward and Reverse Mode" stays in the text; if that part is shortened, move the exercise with it.
- Exercise numbering restarts after the tabbed pair ("3." then "1."); cosmetic in source, check the rendered numbering.

## Still open from October

Nothing specific to this section was raised in October.
