# Data Manipulation

Source: `chapter_preliminaries/ndarray.md`. About 3,500 words of prose in the PyTorch view (7,700 words in the file including code and the other three framework tabs), plus 800 words of exercises.

## Verdict

Solid, clear, well-organized textbook prose, a clear 4. The section has a real running example (a minibatch of 256 images, 200,704 numbers) that returns at every step, and its treatment of broadcasting is as good as any I know. It is not yet a page from MacKay because it is a little dutiful: it walks through the API in the order of the API, a few motivations are borrowed from chapters the reader has not seen (symmetry breaking, layers, offsets), and one promised demonstration (reshape returns a view) is never read out. The most important fix is to close that loop in "Writing Elements" and to stop the small variable names (`x`, `X`, `a`, `b`) from changing meaning under the reader.

## Scores

| Clarity of explanation | Clarity of thought | Pedagogy | Language | Ease and enjoyment |
|:-:|:-:|:-:|:-:|:-:|
| 4 | 4 | 4 | 4 | 3 |

## What works

- The opening earns the section with one computation: rescaling 200,704 pixels takes "three nested loops" in Python and "one expression, `images / 255`" with a tensor library. The reader feels the problem before any API appears.
- The running example is used, not just announced. `images` is created, reshaped to `(256, 784)`, indexed (`images[:, 0, 14, 14]`), rescaled, and used to test the broadcasting rule ("Adding a vector of 256 numbers, one per image, to the flattened minibatch of shape $(256, 784)$ fails"). This last paragraph is the best in the section: it applies the rule to predict one success and one failure and then repairs the failure.
- The broadcasting rule is stated once, completely, with its edge case ("a missing leading axis counts as $1$") and its failure mode ("raises an error rather than guessing"). Stored outputs agree with everything the prose says about them.
- "An integer index removes its axis ... whereas a slice keeps its axis" names the single most common source of shape bugs and shows it in a figure.
- The memory subsection gives a reason to care: "A model with a billion parameters, however, needs 4 GB to store them as `float32` numbers". It also makes the less obvious point that an in-place update "reaches every part of the program that refers to the tensor".
- The closing paragraph hands over to the next section with a real gap: "The numbers in this section were made up."

## Structural findings

- **Order follows the API, with one exception that hurts.** Whether `reshape` copies is raised in "Reshaping" ("writing an element of `X` below also changes the corresponding element of `x`"), but the write happens two subsections later and is not interpreted there (NDA-4). Views and copies are also the subject of "Allocation and In-Place Updates" at the end. The reader meets shared memory in three places (reshape, in-place update, NumPy conversion) without ever being told that these are one idea. Either gather the three into the "Memory and Conversion" part under one sentence ("two names can refer to the same memory; then a write through one is seen through the other"), or keep them where they are and add that sentence at the first occurrence and refer back to it.
- **Motivations borrowed from later chapters.** "Initial Values" explains symmetry breaking (NDA-2) and "the offsets that a layer adds to its outputs", and "Indexing" mentions "the score that a classifier gives to the correct category". The reader does not yet know what a layer, a weight, or an update is. The classifier remark is fine because it is one clause; the symmetry paragraph is a detour.
- **Framework tabs.** In the PyTorch view the section reads smoothly. The cost of four frameworks shows in two places: the many one-line tab notes of the form "In TensorFlow, this function is called `range`", and "Writing Elements" and "Allocation and In-Place Updates", where the common prose is reduced to almost nothing and each tab tells its own story. That is the right trade here, because mutability is where the frameworks differ, but it means the PyTorch reader never learns from the main text that immutability exists as a design choice until the Summary mentions "libraries with immutable tensors".
- **Proportion.** Memory, identity and conversion take about a fifth of the section and two of the seven exercises. For a first-contact section that is slightly heavy relative to indexing and broadcasting, which the reader will use on every page of the book. The NumPy-conversion material could lose the cell that writes `A[0, 0] = 100` with no loss.

## Passage-level findings

**NDA-1** (minor, explanation). Under "Shape".

> The number of axes is the *order* of a tensor, and the minibatch has order 4.

"Order" is introduced here; the figure file and many libraries say rank or `ndim`; the linear algebra section later adds "dimensionality" and discusses "dimension". Four words for adjacent ideas within two sections. Decide here: give `ndim` as the code name in the same sentence ("in code, `images.ndim`"), since that is what the reader will type.

**NDA-2** (moderate, pedagogy). Under "Initial Values".

> If all the weights of a layer started at the same value, all its outputs would be equal and would receive equal updates during training, so they could never come to differ.

This is an argument about hidden units, gradients and updates, given to a reader who has met none of them, in order to motivate a call to `randn`. It cannot be followed here and it is taught properly in the multilayer perceptron chapter. Replace with one sentence that needs no theory: "The weights of a model start at small random values (the reason is in the chapter on multilayer perceptrons), so we need tensors of random numbers." Drop the coined italic "*break this symmetry*".

**NDA-3** (minor, explanation). Under "Reshaping".

> Whether it does so matters only if elements can later be changed in place, which some libraries allow.

Clear to someone who already knows about views; a newcomer has not yet seen an element changed, so "changed in place" has no referent. Move this paragraph and its tab notes to "Writing Elements", where the write happens.

**NDA-4** (moderate, pedagogy). Under "Writing Elements" (PyTorch and MXNet tab).

> We display `x`, from which `X` was reshaped, as well:

The output shows that `x[6]` has become 17 although only `X[1, 2]` was assigned. This is the demonstration promised under "Reshaping", and the prose after the cell says nothing about it; the next sentence moves on to slices. A reader who has forgotten the remark three screens earlier sees two tensors and no point. Add after the output: "The vector `x` has changed too: its element 6 is now 17, because `X` is a view of the same memory."

**NDA-5** (minor, thought). Throughout "Operations".

> x = torch.tensor([1.0, 2, 4, 8])

The names `x`, `X`, `Y`, `a`, `b` are rebound several times: `x` is `arange(12)` and then `(1, 2, 4, 8)`; `X` is a view of the old `x`, is then overwritten with 12s, and is recreated in "Broadcasting" and again in "Concatenation"; `a` is a random vector, then a column, then the scalar 3.5. A reader running the notebook out of order, or flipping back to a figure, cannot tell which `X` is meant. Use distinct names for distinct objects (`u`, `v` for the elementwise pair), or say "we redefine `X`" where it happens.

**NDA-6** (minor, language). Under "Elementwise Operations".

> One formula, applied at every position, thus turns a function of two numbers into a function of two tensors of any shape.

The definition that precedes it ($c_i = f(u_i, v_i)$) is heavier than the idea, and this sentence then generalizes from vectors to "tensors of any shape" without saying how. The code that follows makes the idea plain at once. Consider showing `x + y` first and stating the definition in one sentence afterward.

**NDA-7** (minor, correctness). Under "Elementwise Operations".

> Division with `/` always returns floating-point numbers, here `float32`, even where a quotient happens to be an integer.

True for the four frameworks and this input. "Always" is safe; "here `float32`" is what the output shows. No change needed, but the exercise on `uint8` overflow shows that `+` does not promote. One clause here ("unlike `+`, which keeps `uint8`") would prepare that exercise and prevent the natural wrong inference that all arithmetic promotes.

**NDA-8** (minor, explanation). Under "Allocation and In-Place Updates".

> In-place updates have one restriction: a library that records operations to compute derivatives may still need an old value, so an in-place update must not overwrite it.

A correct warning with nothing the reader can attach it to yet. Either cut it (the autograd section's `no_grad` paragraph makes the point where it matters) or mark it as a forward pointer to :numref:`sec_autograd`.

**NDA-9** (minor, pedagogy). Under "Allocation and In-Place Updates", after the last cell.

> X += Y

The three `id` cells print `False`, `True`, `True` and none of the three outputs is read out in shared prose. The tab text before each cell predicts the result, which is acceptable, but the last cell ends the subsection on a bare `True`. One closing sentence would do: "`X` is still the same object, now holding the sum."

## Exercises

Seven exercises, all with a purpose, well above the usual standard for an API section. "Shifted slices" (inputs and targets of a language model from one tensor) and the second part of "Broadcasting shapes" (`y_hat - y` with shapes `(5, 1)` and `(5,)` silently giving a 5 by 5 matrix) teach bugs the reader will really write. "Limits of data types" is excellent.

- Exercise 2 uses "token" without definition. Add "(integers that stand for words)".
- Exercises 5 and 6 are both about identity and storage and differ in every tab. Exercise 5 in the PyTorch tab relies on `X.set_(...)`, an API the reader will never need. Keep one of the two (exercise 6 is the better problem) and replace the other with an indexing or masking problem, which the section currently does not exercise at all.
- Exercise 6, JAX tab, has a broken sentence: "Write `(A + B) * (-A / 2)` as a function `f(A, B)`, count the intermediate arrays that eager evaluation creates." Insert "and".

## Still open from October

October listed the broadcasting rule as a passage to keep; it is still the best passage. Nothing else from October concerned this section.
