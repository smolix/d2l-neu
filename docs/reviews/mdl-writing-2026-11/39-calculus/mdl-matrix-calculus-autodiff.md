# Automatic Differentiation

Source: `chapter_mdl-calculus/mdl-matrix-calculus-autodiff.md` (lines 1 to 1385, up to the slides marker). About 5,300 words of prose and mathematics outside code, plus 2,100 words of exercises; 14 code cells in the PyTorch view (10 of them framework-independent NumPy or pure Python), 4 figures.

## Verdict

This is the strongest of the sections I read, and parts of it meet the bar. The opening poses a problem with numbers a reader can feel (twelve days against 30 ms), the dual-number passage is a small piece of real mathematical exposition, the two engines are short enough to read in full, and the last third (Hessian–vector products, implicit differentiation, the unconverged solver) is economical and ends on an experiment whose result is not obvious. I found no error in a proof or derivation and recomputed every number in the prose and in the stored outputs; one stated visiting order differs from what the code does (AD-7), and one cited constant I could not confirm (see "Could not check"). The weakness is in the middle. "Two Orders of One Product" and "Parameter Gradients" teach again, at the same level and with new numbers, what the previous section taught twenty pages earlier, and the Hessian of log-sum-exp is derived here for the third time in the chapter. The reader who came for "how does a program do this" waits through a second account of "what the product is". The single most important fix is to cut those repeats down to a clause with a reference, which also brings the section's best material forward.

## Scores

| Dimension | Score |
|---|---|
| Clarity of explanation | 4 |
| Clarity of thought | 4 |
| Pedagogy | 4 |
| Language | 4 |
| Ease and enjoyment | 4 |

Explanation and enjoyment would be 5 for "Dual Numbers", "Vector Tangents and Jacobian Columns", "Hessian–Vector Products" and "Implicit Differentiation" taken alone.

## What works

- **The opening.** "At 10 ms per run, one gradient takes $10^6$ s, about 12 days ... Backpropagation returns all $10^8$ derivatives, correct up to rounding, for the cost of about three runs, roughly 30 ms. This section explains where the factor of $10^8$ goes." The section then pays this off: "which is the 30 ms of the opening".
- **Dual numbers.** $(3+\varepsilon)^2=9+6\varepsilon$ before any general statement; "No differentiation rule was applied; the derivatives came out of multiplication"; and the answer to the reader's objection, with $\varepsilon$ as a $2\times2$ matrix: "Dual numbers are as concrete as $2\times2$ matrices, just as complex numbers are as concrete as the rotations and scalings of the plane." This is the kind of sentence the voice guidance asks for. The care about primitives that are only differentiable (the display taken as a definition) and the structural induction are right and short.
- **Sentences that judge.** "Forward mode removes the step size but not the number of runs." "It can, but the saving is smaller than it looks ... Passes and work are different measures." "The test is not vacuous." Each tells the reader something about the material that a neutral account would leave out.
- **The hand trace** of $y=(uv+u)^2$, with a table, a figure of the diamond, and the consequence of the wrong implementation ("Overwriting instead of adding would give $\bar r=-4$"), which explains the `+=` in the code.
- **The count.** $80{,}200$ operations against $802$, measured by the two engines the reader has just read, with the formula $n(2n+1)$ against $2(2n+1)$ beside it.
- **The non-vacuous test** for the Hessian–vector product: a quadratic would let a wrong program pass, and the text says so and picks a function where it would not.
- **Implicit differentiation**, from $x^2=\theta$ to the general formula in half a page, and then its two limits shown, not asserted. The last cell (formula and unrolled derivative both wrong before convergence, by different amounts) is the best experiment in the two sections.
- **Honest tabs.** The MXNet notes say plainly what that framework cannot do and what the tab does instead.

## Structural findings

1. **The middle of the section repeats the previous section.** See AD-3 and AD-10. About 700 words ("Two Orders of One Product" up to the figure, and "Parameter Gradients" except its last paragraph) restate the Jacobian product, the two bracketings, a cost count, and the outer-product weight gradient, all of which `sec_mdl-matrix-calculus` derives. With them reduced to references, "Reverse Mode and the Tape" would open on the tape, which is its subject.
2. **Overlap with the main text.** `sec_autograd` already tells the reader, at the level of use, that reverse sweeps fill rows and forward sweeps fill columns, that a loss is one row, that $10^8$ parameters mean $10^8$ forward sweeps, and that reverse mode pays in memory; it also calls the two framework products on a small Jacobian. This section is deeper in the right ways (it builds both engines, proves the forward rule, measures the cost, and composes the modes), so the overlap is mostly healthy. Two places are at the same level as the main text and add nothing: the framework cell under "Jacobian–Vector and Vector–Jacobian Products" (AD-14) and "Parameter Gradients" against `sec_backprop` (AD-10).
3. **Two sections of the book are titled "Automatic Differentiation"** (this one and `sec_autograd`). In a table of contents or an index the reader cannot tell them apart. The file name suggests an older title. A title that says what this section adds would help the lookup reader, for example "Forward and Reverse Mode" or "How Automatic Differentiation Works".
4. **The running example is dropped where it matters most.** Log-sum-exp is announced as one of three recurring examples, and forward mode computes its gradient in three passes. Reverse mode, the mode the section argues for, is never run on it, because `Var` has no `exp` or `log`. See AD-8.
5. **Is it two sections?** No. Forward mode, reverse mode, cost and memory are one argument, and the three compositions are short. If length must fall further, "Gradient Checkpointing" is the most detachable part (it has no code in the body and an exercise that implements it), but it answers the memory question that the cost subsection raises, and I would keep it.
6. **What a reader might look for and not find.** Differentiation through control flow and at kinks, stopping gradients, and in-place operations are in `sec_autograd` or the Further Reading notebook, which is the right division. One sentence in the body that says so would stop the reader from searching (AD-1).

## Passage-level findings

**AD-1** · minor · thought
Anchor: "To differentiate a program, we therefore run it on $x+\varepsilon$ and read the" (Dual Numbers).
The Proposition is about an expression built from sums, products and primitives. The next sentence applies it to "a program". Programs have loops and branches, and the reader has seen in `sec_autograd` that the framework differentiates through them. The gap closes in one sentence, and it is worth closing because it is the reason operator overloading works at all.
Fix: add "A program with loops and branches executes, for a given input, one particular sequence of operations, which is an expression of this kind; the dual run differentiates that sequence (`sec_autograd` shows this for the frameworks)."

**AD-2** · minor · explanation (terms defined twice)
Anchors: "value $a$ and the derivative part $b$, which we call the *tangent*. It" (A Forward-Mode Engine); "the direction $\dot x$; forward mode computes tangents. The *adjoint*" (The Tape).
"Tangent" is defined under "A Forward-Mode Engine" and again, with new notation $\dot v$, under "The Tape", after forward mode is finished. "Adjoint" is defined here for the third time in the chapter (the multivariable section's `eq_mdl-adjoint`, the previous section's $\bar{\mathbf Y}$). The $\dot v$ notation is used only in this paragraph and in the figure.
Fix: introduce $\dot v$ where `Dual` is introduced ("the tangent, written $\dot v$"). Under "The Tape" keep the contrast, which is the valuable part ("A tangent is determined by the steps before $v$, an adjoint by the steps after it"), and cite the earlier definitions instead of restating them.

**AD-3** · major · thought (repetition across sections)
Anchor: "The derivative of a program is a product of small matrices, one per step. A" (Two Orders of One Product).
From "Consider a program of $K$ steps" to the figure, the subsection restates the previous section's "The Order of Multiplication": the product $\mathbf J_K\cdots\mathbf J_1$ (displayed again under a second label, `eq_mdl-ad-jacobian-product`, beside the existing `eq_mdl-jacobian-product`), the two bracketings, the rule that a seed keeps every step a matrix–vector product, and a cost count. The count uses a different network ("$1000$ inputs and two hidden layers of width $1000$ ... about $2\times10^6$ multiplications, while forming $\mathbf J_2\mathbf J_1$ first costs $10^9$") from the one the reader has just seen ($3\times10^6$ against $2.001\times10^9$), so the reader has to work out whether the two counts agree. They do, but nothing is learned from the second.
What is new here and should stay: the two displays of `eq_mdl-ad-two-orders` as programs with a seed, the names JVP and VJP with the reading "for $\mathbf u=\mathbf e_i$ it is row $i$", the cost rule in sweeps with the Baur–Strassen bound, and the figure of shapes.
Fix: open with one sentence, "The previous section showed that the derivative of a chain is the product `eq_mdl-jacobian-product` and that, with a row on the left, multiplying from the left costs $3\times10^6$ multiplications where the other order costs $2\times10^9$ (`fig_mdl-cal-jacobian-chain-order`)." Then go directly to `eq_mdl-ad-two-orders`. Delete the second display of the product, its label, and the second count. Later references to `eq_mdl-ad-jacobian-product` (in "Parameter Gradients" and in the exercise on mixed mode) change to `eq_mdl-jacobian-product`.

**AD-4** · minor · language
Anchor: "does not depend on $n$, but the cost of one sweep grows with the program. A" (Two Orders of One Product).
The "but" contrasts two things that do not oppose each other, and the second holds for both modes. I could not tell what the sentence is meant to warn against.
Fix: delete the sentence. The next one ("A scalar loss has a $1\times n$ Jacobian, a single row, so one reverse sweep seeded with $1$ returns the whole gradient") says what is needed.

**AD-5** · moderate · thought (three constants for one quantity)
Anchors: "derivatives, correct up to rounding, for the cost of about three runs," (opening); "partial derivatives together take at most five times as many operations as the" (Two Orders of One Product); "$802=2(2n+1)$: $401$ recorded on the forward pass and $401$ backward closures." (Cost of the Two Modes).
The cost of a gradient relative to one evaluation is given as about three (opening, and derived for a dense layer under "Parameter Gradients"), at most five (Baur and Strassen), and two (measured: $802$ against $401$). All three are right for what they count, but the text never says that they count different things, and the measured $2$ contradicts the promised $3$ on its face. The reason is that `Var.ops` counts each backward closure as one operation, although the closure of a product does two multiplications and two additions. Counted in arithmetic operations the engine's backward sweep costs more than its forward pass, in line with the bound.
Fix: one sentence after the count: "The counter treats a backward closure as one operation. The closure of a product performs two multiplications and two additions, so in arithmetic the sweep costs about twice the forward pass, which is the factor of three of the opening and within the bound of five." (The reviser should recount the exact arithmetic ratio for this $f$ before quoting it.)

**AD-6** · minor · thought (attribution)
Anchor: "backward. The record is called the *tape*, or Wengert list after" (The Tape).
The sentence can be read as crediting Wengert with the record that reverse mode replays, while `sec_autograd` says "Forward mode goes back to Wengert (1964), and reverse mode was developed by Linnainmaa (1970)". The two sections should tell the reader the same story.
Fix: "The record is called the *tape*, or Wengert list, after the list of elementary operations that Wengert (1964) introduced for forward mode; running it backward is reverse mode, introduced by Linnainmaa (1970)."

**AD-7** · minor · correctness (confidence high; I ran the code)
Anchor: "sweep visits the nodes in the order $y,r,t,u,v$. Each adjoint is the sum of the" (The Tape).
`backprop` as written visits $y,r,t,v,u$: the depth-first search finishes $u$ before $v$ (both are parents of $t$), so the reversed order ends with $v$, then $u$. Both orders are valid, and the table's adjoints are correct, since $\bar u$ is complete once $r$ and $t$ have run. A reader who adds a print statement to check the claim will see the other order.
Fix: write "in an order such as $y,r,t,u,v$, in which every node comes after all nodes that use it", or swap the last two rows of the table and the stated order to match the code.

**AD-8** · moderate · pedagogy (needs a small code change; deferred category)
Anchors: "Does the tape reproduce the table? We record the expression with `Var` and run" (The Tape); "$j=1,\ldots,200$, built from $+$ and $\times$ alone. Its gradient is" (Cost of the Two Modes).
The forward-mode demonstration asks "How many runs does forward mode need for this gradient?" on log-sum-exp and answers three. The matching demonstration, one backward sweep returning the same three softmax entries, never comes: `Var` implements only `+` and `*`, so the tape is shown on $(uv+u)^2$ and the cost comparison moves to $(\sum_jx_j^2)^2$, chosen because it is "built from $+$ and $\times$ alone". The section's announced running example is thus used for the mode that loses and not for the mode that wins.
Fix: give `Var` an `exp` and a `log` (six lines each, the mirror of the `Dual` methods), and after the hand trace run `lse` once on the tape at $(1,-0.5,2)$: one sweep, three partials $0.2537$, $0.0566$, $0.6897$. The cost comparison can then stay as it is or use log-sum-exp with $n=200$. If code changes wait, say in the prose why the example changes: "`Var` has only `+` and `*`, so we cannot run log-sum-exp on it; an exercise adds the missing primitives."

**AD-9** · moderate · explanation (unexplained jargon; tab prose repeats shared prose)
Anchors: "instead turns the program into an expression, linearizes it, and obtains the" (The Tape); "JAX traces: `jax.grad` turns the function into an expression, linearizes it," (JAX tab).
"Linearizes it, and obtains the VJP by transposing the JVP" is the only sentence in the section that a reader at the stated level cannot follow. "Linearize" has not been used for programs, and nothing so far says that a VJP is the transpose of a JVP, although the previous section's Proposition on backward rules says exactly that in other words. The sentence is then repeated verbatim in the JAX tab, and the three other tabs each repeat "keeps an eager tape" from the shared paragraph, which the guide's rule on tab prose (7.7) excludes.
Fix: in the shared prose, say it in the section's own terms: "A tracer runs the program once on symbolic inputs to obtain its sequence of operations, forms the forward-mode rule of each (a linear map of tangents), and reverses the sequence, replacing each linear map by its transpose. The result is the same backward sweep." Reduce each tab to the API fact that differs (which call records, which call replays), or drop the tabs.

**AD-10** · major · thought (repetition across sections and with the main text)
Anchor: "the weights, and they do not appear in that product. Where are they?" (Parameter Gradients).
The previous section raised the same question ("The weight matrices $\mathbf W_k$ appear in it as factors, never as variables"), answered it with the backward rule of a layer, derived $\bar{\mathbf W}=\bar{\mathbf y}\mathbf x^\top$ for a column and $\bar{\mathbf W}=\mathbf X^\top\bar{\mathbf Y}$ for a minibatch, and ran both on the opening network. This subsection poses the question as open, derives the outer product again by perturbing $\mathbf W_k$, works a second small numerical example, and then has to explain why its formula is the transpose of the one "quoted above". `sec_backprop` in the main text derives the same rule a third time by indices. The subsection's last paragraph is the only part that belongs to this section's argument: the outer product costs as much as the forward product, passing the adjoint down costs the same again, so the sweep costs two forward passes ("which is the 30 ms of the opening"), and forward mode would need $d_kd_{k-1}$ passes per layer.
Fix: replace everything before the last paragraph by two sentences: "The product `eq_mdl-jacobian-product` is the derivative with respect to the input. The weight gradient of layer $k$ is read off during the same sweep by the backward rule of `subsec_mdl-layer-vjp`: $\nabla_{\mathbf W_k}\ell=\bar{\mathbf h}_k\mathbf h_{k-1}^\top$, the outer product of the adjoint arriving from above and the activation arriving from below." Keep the sentence "a weight matters in proportion to how much its output matters to the loss and how large its input is", which is the best line of the subsection, and the cost paragraph. The subsection could then be merged into "Cost of the Two Modes" as its analytic half, before the measured half. The Summary's clause on the outer product goes with it.

**AD-11** · minor · language (pattern shared with the previous section)
Anchor: "The step that is not routine is to perturb $\mathbf W_k$ alone, with the input" (Parameter Gradients).
Third use of "the step that is not routine" in two sections. Here the step is routine: perturbing one variable with the others fixed is the definition of a partial derivative.
Fix: goes away with AD-10. If the passage stays, drop the phrase.

**AD-12** · minor · thought
Anchor: "is the transpose of the partial product $\mathbf J_K\cdots\mathbf J_{k+1}$," (Parameter Gradients).
The product was defined as the Jacobian of the network, a map with vector output. Here its partial product is "a row because the loss is a scalar", which is true only if the loss is counted as the last step $\mathbf f_K$. The previous section kept the two apart by writing the loss as a row $\mathbf u^\top$ in front of the product.
Fix: goes away with AD-10. Otherwise write "the transpose of $\mathbf u^\top\mathbf J_K\cdots\mathbf J_{k+1}$".

**AD-13** · minor · pedagogy (connection to practice)
Anchor: "The simplest remedy cuts the tape into segments of $s$ steps. The forward pass" (Gradient Checkpointing).
The subsection is clear and the arithmetic is right ($401,85,50,40,104,401$). It does not say where the reader will meet the technique. `sec_backprop` ends its memory count with the sentence "A framework can trade computation for memory by discarding some of the stored values and recomputing them", with no pointer here.
Fix: one sentence naming the use ("frameworks offer this for blocks of a deep network; a block is then run twice and its activations are held only while its own backward pass runs") and, in the chapter notes, a request that `sec_backprop` point to `subsec_mdl-checkpointing`.

**AD-14** · moderate · pedagogy (a cell at the level of the main text; a number shown five times)
Anchors: "and $\mathbf u^\top\mathbf J$. Everything a training loop needs is built by" (Composing Jacobian Products); "Do the framework's two products give the plane-map numbers? From" (Jacobian–Vector and Vector–Jacobian Products).
(a) "Everything a training loop needs is built by composing them" overstates: a training loop needs one VJP. What the part shows is what else can be built. (b) The cell calls the framework's `jvp`, `vjp` and Jacobian functions on the plane map. `sec_autograd` has the same three calls on another $2$-input function, with the same per-tab API notes. Within this section the matrix $\left(\begin{smallmatrix}1&0.8776\\0.5&1\end{smallmatrix}\right)$ has by now been displayed once and printed twice by `Dual`; this cell prints it a third time and the prose restates it. The paragraph after the cell is worth keeping (a dense Jacobian costs $\min(m,n)$ sweeps plus $mn$ numbers; form it when it is small or when its determinant is needed).
Fix: (a) "Curvature along a direction, derivatives of solutions, and full Jacobians when they are needed are all built by composing them." (b) Shrink the subsection to the paragraph after the cell, with the two expected products as a one-line hand computation and a reference to `sec_autograd` for the calls; or keep the cell and cut the earlier one-pass `Dual` printout of the same matrix. Either way the reader of one framework should see this Jacobian printed once in this section.

**AD-15** · moderate · thought (third derivation in the chapter)
Anchors: "whose one-line derivation we restate. With $S=\sum_ke^{x_k}$ and $\delta_{ij}$ equal to" (Hessian–Vector Products); "Two checks against intuition follow from :eqref:`eq_mdl-ad-lse-hessian`. First," (same).
The Hessian of log-sum-exp is derived by the quotient rule in the multivariable section, in the previous section, and here, and the text knows it ("we restate"). The paragraph "Two checks against intuition" then repeats, with the same opening words, the previous section's two remarks: $\mathbf H\mathbf 1=\mathbf 0$ by shift invariance, and positive semidefiniteness by a variance. The guide's rule is that a restatement "never re-derives" and "is never a paragraph". One thing here is new and good: $\mathbf H$ is the covariance matrix of a one-hot vector $\mathbf e_J$ with $J$ drawn from $\mathbf p$.
Fix: "Its Hessian is the softmax Jacobian $\operatorname{diag}(\mathbf p)-\mathbf p\mathbf p^\top$ of `eq_mdl-softmax-jacobian`, so $\mathbf H\mathbf v=\mathbf p\odot\mathbf v-\mathbf p\,(\mathbf p^\top\mathbf v)$", then the numbers and the "not vacuous" paragraph. Delete the "Two checks" paragraph and move the covariance sentence to the previous section, where the variance argument is made and where it gives that argument its name.

**AD-16** · moderate · enjoyment (pattern: 9 questions before 11 output cells)
Anchors: "Does one run on $1.3+\varepsilon$ return both $f(1.3)$ and $f'(1.3)$ for"; "Does the tape reproduce the table? We record the expression with `Var` and run"; "Does the framework's forward over reverse reproduce the formula?"; "Does :eqref:`eq_mdl-implicit-grad` agree with a finite difference of the".
Nearly every cell is led in by a question, and the two sections together do this fourteen times. Where the answer is not known in advance the device works very well: "How many runs does forward mode need for this gradient?", "Can one pass return the whole matrix?" (answer: yes, but), "What do the formula and the unrolled solver return when Newton's method ... stops after $k=1,2,3$ steps?". Where the question is "does the code agree with the formula just derived?" the answer is always yes, and by the fourth such question the reader stops treating them as questions. The cells themselves are fine; it is the uniform framing that flattens the section.
Fix: keep the question form for the cells whose outcome carries information (the three quoted above, and the cost count). For the confirmations, lead in with a plain statement and let the output stand: "One run on $1.3+\varepsilon$ returns both numbers:" and after it only what the output adds.

**AD-17** · moderate · language (closing formula)
Anchor: "We can now write both modes of automatic differentiation in a few lines, trace" (end of the body).
A recap paragraph beginning "We can now ..." stands directly before the Summary, as in the previous section, and lists what the Summary lists. The guide (12.2, item 3) asks for a closing that belongs to the argument and names this formula as one to avoid; two consecutive sections now close with it. The paragraph that follows ("One problem stays open ...") is a real closing.
Fix: delete the "We can now" paragraph. End the body on the unconverged solver ("The remedy is to converge first.") and the open problem. In the open-problem paragraph, cite the previous section's statement on $|\lambda_{\max}|^K$ in a clause and add the label of the main-text section that treats the remedies, since "belong to optimization and to the design of architectures" names no place.

## Exercises

Ten problems, about 2,100 words, 40% of the body. They are the most ambitious set I have seen in the book, and almost all of them are good: each opens a topic the body only touched (complex step, sparse Jacobians by seed colouring, hyperparameter gradients, forward over reverse built from `Var(Dual(...))`, checkpointing implemented, equilibrium layers, mixed mode through a bottleneck, forward gradients and their variance, truncated dual numbers for second derivatives), with numbers given so that a reader can tell whether they have succeeded. I checked what could be checked cheaply: the variance $(n+1)\|\nabla\ell\|^2$ and the $2\times10^4$ passes of the forward-gradient exercise; $f''(1.3)\approx-3.2806$; the gradient formulas of the hyperparameter and equilibrium exercises; and I ran the nesting of exercise 4, parts (b) and (c), on the section's own classes, where `Var(Dual(...))` and `Var(Var(...))` both work unchanged and return the right $\mathbf H\mathbf v$.

- **Trim.** "A tensor tape", part (a), asks again for the derivation of $\bar{\mathbf X}=\bar{\mathbf Y}\mathbf W^\top$, $\bar{\mathbf W}=\mathbf X^\top\bar{\mathbf Y}$, which the previous section derives in the text and asks for in its own exercise; part (d) asks for the count of forward against backward multiplications a third time. Keep the broadcast and sum rules of (a), which appear nowhere else, and drop (d).
- **Fix.** "Forward gradients" cites Baydin and coauthors by a title in prose instead of a `:cite:` key, and part (c) asks for a plot "with `d2l.plot`" although this section does not import `d2l`. "Checkpointing on the tape" is clear, and its warning about the recursion limit is the kind of care that saves a reader an hour.
- **Size.** "Equilibrium layer" is a small project (four parts, two plots, two papers). It is marked [extended] and ●●, which is honest. If the count must fall, it is the one to move to the Further Reading notebook as a pointer, but it is the best application of implicit differentiation in the chapter.
- **Add.** If AD-8 is not fixed in the body, an early exercise should add `exp` and `log` to `Var` and compute the softmax by one backward sweep.

## Could not check

- "Baur and Strassen showed that all partial derivatives together take at most five times as many operations as the program itself." The theorem and its order of magnitude are right; I could not confirm from memory whether the constant in the cited paper, for the operation count used here, is 5, 4, or 3. Worth one look at the source.
- The attribution "Problem 1" of the MIT problem set for the hyperparameter exercise (the previous section's linear-solve exercise carries the same problem number).
- The MXNet, TensorFlow and JAX stored outputs. I compared the prose with the PyTorch store only.

## Still open from October

"Backprop is explained three times" is still open, and this section holds the third account (AD-10). The October review's "keep" item, forward and reverse mode as two parenthesizations of one product, is still here and still good; the duplication with the previous section is the price of how the two were split.

## For the chapter review

- **Order and fit.** The position is right: the section is the answer to the chapter introduction's second question ("one call returns every slope for the cost of a few. How?"). It assumes the Jacobian, the chain rule as a product, and the layer rules of `sec_mdl-matrix-calculus`, and the use of the framework calls from `sec_autograd`. It sets up the adjoint method in `sec_mdl-odes-solvers` and the Hessian–vector products that the optimization chapter needs.
- **Boundary with the previous section.** The two sections were evidently one. The clean division is: the previous section owns the product, its cost by order, and the backward rule of a layer; this one owns seeds as programs, the two engines, measured cost, memory, and compositions. AD-3 and AD-10 implement that division from this side.
- **Repetition across the chapter.** The $10^8$-parameter argument appears in the chapter introduction, in `sec_autograd`, and in this opening. This section is its home; the chapter introduction might give the question without the numbers.
- **A wrong address in the main text.** `sec_backprop` sends readers here for the layer rules "for general tensor operations". They are in `subsec_mdl-layer-vjp` of the previous section; here they are only an exercise.
- **Level against Basics.** Deeper throughout except in the two places noted (AD-10, AD-14). The section is a good example of what the appendix should be relative to the main text: the main text shows how to call the tool, this section builds it in forty lines and proves why it works.
- **Suspected patterns for the part.** A question before every cell regardless of whether the answer is in doubt; a "We can now ..." paragraph before the Summary; and results derived afresh in each section that needs them instead of cited from the one that owns them.
