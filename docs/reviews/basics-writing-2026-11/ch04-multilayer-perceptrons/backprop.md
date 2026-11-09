# Backpropagation

Source: `chapter_multilayer-perceptrons/backprop.md`. About 3,300 words of prose, 970 words of exercises, three code cells, three generated figures.

## Verdict

Mechanically this is a sound and careful treatment: three local rules derived at the level of matrix entries, a worked example whose every number checks, a hand-written backward pass that agrees with autograd to sixteen digits, and a memory and cost count in real units. A reader who works through it can compute the gradient of a one-hidden-layer MLP. What the section does not yet give is the understanding around the mechanics. It asks in its first paragraph how one backward pass can produce 203,530 derivatives at once and never answers in its own text. It never says what the gradient arriving at a hidden unit *means*. It stops at one hidden layer, although the next section needs the recursion over many. And it starts over from a scalar graph that is simpler than the matrix graph the reader already worked through in `sec_autograd`, without mentioning that graph. The most important fix is to rebuild the first third on top of the autograd section instead of beside it, and spend the space saved on the answer to the opening question and on depth.

## Scores

| Clarity of explanation | Clarity of thought | Pedagogy | Language | Ease and enjoyment |
|---|---|---|---|---|
| 4 | 3 | 3 | 4 | 3 |

## What works

- **The matrix-product rule is derived, then checked three ways.** From $a_{ij} = \sum_r x_{ir} w_{rj}$ to $\mathbf{X}^\top\mathbf{G}$ by counting which entries $w_{rj}$ enters; then by shapes ("no other product of the two factors has the right shape"); then read as "a sum of $n$ outer products, one per example". This is the equation protocol done well, and the outer-product reading is the kind of sentence Cover and Thomas put after an identity.
- **The worked example.** Two inputs, two hidden units, one inactive. All numbers are right (I recomputed them), the zero column of $\partial L / \partial \mathbf{W}^{(1)}$ is tied to the dead unit, and the figure `mdl-mlp-backprop-graph` shows every value and gradient in the same black and red convention as the autograd chapter. The code cell then prints the same numbers.
- **The practical remark on fan-out.** "A hand-written backward pass that assigns the second contribution instead of adding it, `=` where `+=` belongs, computes a wrong gradient without any error message." Brief, concrete, true.
- **Memory in megabytes.** 268,800 stored values, 1.1 MB, against 0.8 MB of parameters, and the reason training runs out of memory before prediction does. Likewise the cost count: 205,824 against 203,264 multiply-adds, "about two forward passes for this network and about three for a network whose first layer is small".
- **The check is stated as a claim about the framework**: "for this network, :eqref:`eq_backprop-mlp` is the computation that automatic differentiation performs."
- The opening's three practical questions (memory, dead ReLU, vanishing gradients) give the reader reasons to care about a derivation they could otherwise skip.

## Structural findings

- **Fit with `sec_autograd` (chapter 1).** The autograd section already taught: the computational graph with values in black and gradients in red; the backward pass as local rules applied from the loss; the rule that a matrix-vector product passes back $\mathbf{X}^\top$ times the incoming gradient; that the rules read values saved in the forward pass, "This is why training needs more memory than prediction"; vector-Jacobian products; forward against reverse mode; and the cost argument ("one reverse sweep does the work of $10^8$ forward sweeps"). It ends by promising that `sec_backprop` "opens it for a network with one hidden layer". The two sections agree in terminology and figure style, which is good. But the backprop section behaves as if the reader had seen none of that. It reintroduces the graph on $e = (a + b)c$, a scalar example with no matrix and no value used twice, which is a step *down* from the graph the reader already traced. It restates the memory point as a new question. It mentions the earlier section twice, once in the opening and once in a subordinate clause ("special cases of the vector–Jacobian products of :numref:`sec_autograd`"). The division of labour should be explicit: chapter 1 gave the general mechanism and its cost; this section supplies what is new for a network, namely gradients with respect to weight *matrices*, the gradient passed *through* a layer to the one below, the elementwise activation rule, and what all of it costs for a real model.
- **The opening question is answered in chapter 1 and in the exercises, not here.** See BP-2.
- **Three different setups.** The general derivation uses cross-entropy plus weight decay on a minibatch (objective $J$). The worked example uses squared error, one output, one example, no weight decay (loss $L$). The numerical check returns to cross-entropy plus weight decay. Each choice is defensible, and each transition costs the reader a paragraph of adjustment (in the worked example, "only the first line of :eqref:`eq_backprop-mlp` changes"). Weight decay is in the derivation for one purpose, to produce a value that feeds two nodes, and the text does not say so.
- **One hidden layer only.** The section never writes the recursion for $K$ layers. Exercise 3 asks the reader to generalize it, and the next section opens with "the backward pass ... multiplies the gradient by one transposed weight matrix per layer" as known. A four-line statement of the general step and a six-line loop would make the closing sentence of this section, and the opening of the next, something the reader has seen.
- **No interpretation.** The name "backpropagation" is about errors propagating back. The section has the formula $\partial J / \partial \mathbf{H} = (\partial J / \partial \mathbf{O})\mathbf{W}^{(2)\top}$ and never reads it: each hidden unit receives the output errors $\hat{\mathbf{y}} - \mathbf{y}$, weighted by its own outgoing weights. That sentence is the difference between knowing the rules and understanding why the first layer learns anything.

## Passage-level findings

- **BP-1** (major, pedagogy: order and reuse)
  - Anchor (A Small Graph): "both for the smallest graph that contains an addition and a multiplication,"
  - The reader traced a four-operation matrix graph in `sec_autograd`, with a figure in the same style. Starting again from three scalars teaches nothing new except the two rules that an addition copies the incoming gradient and a multiplication scales it by the other input, and the example lacks the one feature that is new and error-prone here, a value used twice.
  - Fix: open "Computational Graphs" by recalling the autograd graph in two sentences and stating what it lacked (two layers, a nonlinearity, parameters that are matrices, a value that feeds two nodes). If a scalar warm-up is kept, make it $e = (a + b)\,b$ or similar, so that $b$ collects two contributions and the sum-over-paths rule is seen with numbers before it reappears as weight decay. October listed the $e = (a+b)c$ example as a keep; I would keep its style and change its content.

- **BP-2** (major, thought)
  - Anchor (section opening): "cost of the forward pass. How can one pass backward through a network"
  - The section's headline question. The body derives the rules and counts the cost, but never states the answer: every parameter's gradient reuses the same intermediate gradients ($\partial J / \partial \mathbf{O}$, $\partial J / \partial \mathbf{Z}$), each computed once; the alternative, nudging one parameter at a time, would take 203,530 forward passes. The contrast with forward mode appears only in exercise 5, and the path-sharing argument only in part (d) of exercise 7.
  - Fix: add to "Computation" a short paragraph, for example: "This answers the question we started with. Finite differences, or forward mode, would need one pass per parameter, 203,530 of them. The backward pass computes $\partial J / \partial \mathbf{O}$ and $\partial J / \partial \mathbf{Z}$ once, and every one of the 203,530 derivatives is read off from these two matrices by a single product. The work is shared, which is what :numref:`sec_autograd` called reverse mode."

- **BP-3** (moderate, pedagogy: abstraction before example)
  - Anchor (A Small Graph): "Before the backward pass can run, two questions must be answered for every"
  - The two questions (which forward values does a local derivative need, when is a gradient ready to be passed on) use "node", "local derivative" and "ready" before the reader has a graph in front of them. They are good questions and the section returns to them twice ("The graph answers the two questions", "answers the two questions for the MLP"), but as an opening they are abstract.
  - Fix: compute the small example first, then ask: "Two things in this computation will matter for a network. Which values did the backward pass need from the forward pass? And in which order did the gradients become available?"

- **BP-4** (moderate, explanation)
  - Anchor (The Graph of an MLP): "class $y_i$, and $L$ is the average of $l_i$ over the $n$ examples. We also add weight decay (:numref:`sec_weight_decay`),"
  - Weight decay, the Frobenius norm, the penalty $s$ and the objective $J$ arrive in three lines with no reason given. The reader has just been told the loss is cross-entropy and now has a second scalar and a third symbol for "what we minimize".
  - Fix: give the reason: "We also add weight decay, both because training usually includes it and because it makes each weight matrix feed two nodes, which shows how gradients from several paths combine." If BP-1 is adopted and fan-out is taught on the scalar graph, weight decay could instead be dropped from the derivation and left to an exercise, which would remove $s$, $J$ and one of the three setups.

- **BP-5** (moderate, explanation: a result left unread)
  - Anchor (The Backward Pass): "Read from top to bottom,"
  - The paragraph after the five-line display reads it for order and for stored values, which is useful, but not for meaning. No sentence says what $\partial J / \partial \mathbf{H}$ is.
  - Fix: add after the display: "The third line is where the name comes from. The error at the output, $\hat{\mathbf{Y}} - \mathbf{Y}$, is sent back through the same weights that carried the hidden activations forward: hidden unit $j$ receives the errors of the ten outputs, each weighted by the weight from $j$ to that output. A unit with large weights to outputs that are wrong receives a large gradient."

- **BP-6** (minor, correctness: text and figure disagree)
  - Anchor (The Backward Pass): "rules, from the top of :numref:`fig_forward` to the bottom:"
  - In the figure the objective $J$ is at the top right, the network runs left to right along the bottom row, and the backward pass runs right to left along that row. "From the top to the bottom" matches the display that follows, not the figure.
  - Fix: "from $J$ back to the first layer, right to left in :numref:`fig_forward`".

- **BP-7** (moderate, pedagogy: scope)
  - Anchor (closing paragraph): "backward pass multiplies the gradient by one transposed weight matrix and"
  - This sentence describes the general recursion, which the section has not written. The next section depends on it and has to rederive it in a different notation (per-example column vectors, $\nabla$, a matrix $\mathbf{M}^{(k)}$; see NS-3).
  - Fix: before "Memory and Computation", add a short subsection "Deeper Networks": the step $\partial J / \partial \mathbf{H}^{(k-1)} = \bigl(\partial J / \partial \mathbf{H}^{(k)} \odot \phi'(\mathbf{Z}^{(k)})\bigr)\mathbf{W}^{(k)\top}$, one sentence that it repeats lines 3 and 4 of the display, and a loop of about six lines over a list of weight matrices. The function `moments` in the next section already contains this loop; introducing it here lets that section reuse it.

- **BP-8** (minor, writing for the authors)
  - Anchor (after the verification cell): "column for the inactive unit. Some outputs show $-0$, a zero with a negative"
  - The stored PyTorch output shows no $-0$. The sentence narrates what some framework tab does. A PyTorch reader looks for a $-0$ that is not there.
  - Fix: move into a tab note for the frameworks that print it, or delete.

- **BP-9** (minor, language)
  - Anchor (Stored Values): "The forward pass and the backward pass depend on each other. The forward"
  - A paragraph of four sentences saying that training alternates forward pass, backward pass and update. The reader has known this since chapter 2. It delays the memory count, which is the content of the subsection.
  - Fix: delete the paragraph, keeping at most "The backward pass reads values that the forward pass computed, so the forward pass must store them."

- **BP-10** (minor, explanation: prose fits one framework)
  - Anchor (section opening): "In :numref:`sec_mlp-implementation` we trained an MLP by calling"
  - The reader did not call `backward` in that section; the `Trainer` did, and in the TensorFlow and JAX tabs nothing is named `backward` at all.
  - Fix: "we trained an MLP without writing a line of calculus: the framework supplied the gradients."

- **BP-11** (minor, notation)
  - The chapter uses $\partial J / \partial \mathbf{W}$ here, $\nabla_{\mathbf{h}} L$ in the next section, and $\mathbf{G}$ both as "the incoming gradient" in the rules and as a code variable. $L$ is the average loss here and the objective in the next section, where there is no penalty. None of this is wrong; a reader moving between the two sections has to translate. Settle on one gradient notation for both sections.

## Exercises

Seven problems, the strongest set in the chapter. "Jacobian of a layer" makes the reader count the 4-index object that backprop never forms. "Forward mode" and part (d) of "Autograd engine" contain the efficiency argument that the text lacks (BP-2); keep them, but the text should state the result they explore. "Gradient checking" teaches the trade-off between truncation and rounding error with a plot. "Hessian of a ReLU network" is short and has a clean answer (rank one). "Autograd engine" is the micrograd exercise and is marked extended, rightly.

- "Memory of a deeper network" part (a) asks the reader to generalize the backward pass to $K$ layers. If BP-7 is adopted, rephrase it to use the recursion from the text.
- No exercise is filler.

## Still open from October

October's two findings are closed: the weight layout now matches `mlp.md` (rows are inputs), and the undefined $\textrm{prod}$ operator is gone, replaced by explicit matrix rules. October's "keep" for the $e = (a+b)c$ example is honored; I suggest changing the example for the reason in BP-1.
