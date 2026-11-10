# Documentation

Source: `chapter_preliminaries/lookup-api.md`. About 1,300 words of prose in the PyTorch view (3,700 in the file to the slides marker), plus 570 words of exercises in one tab.

## Verdict

A short section with a good idea and an uneven execution. The idea: do not list functions, teach a procedure (discover, inspect, read, verify) and insist that the last step be a prediction that could fail. The paragraph on coding assistants applies the same procedure to suggested code and is the most up-to-date advice in the chapter. The execution problem is that the section's one running puzzle, what `ones(2, 3)` does, has a real answer in only one of the four frameworks. In TensorFlow the call silently returns two `int32` ones, which is a fine detective story. In PyTorch, the reader's default, `torch.ones(2, 3)` returns exactly the $2 \times 3$ matrix it appears to ask for, so the shared prose ("a call can look harmless and still do something other than what we meant") describes something that does not happen on the page. A second thread, the search for a sampler, is opened and then handed to an exercise. The main fix is to choose a puzzle per framework that actually surprises, or one shared puzzle that surprises in all four.

## Scores

| Clarity of explanation | Clarity of thought | Pedagogy | Language | Ease and enjoyment |
|:-:|:-:|:-:|:-:|:-:|
| 4 | 3 | 3 | 4 | 3 |

## What works

- The opening is honest about scale: "the main module of a deep learning framework defines several hundred to a thousand public names, and new releases add names and deprecate others."
- The three conventions for reading a signature (default values, the lone `*`, `*name`) are exactly what a reader needs to read `help` output and are given in three short sentences.
- "A small example settles the question, provided that we predict its result before we run it, so that the run could prove us wrong. A run that merely completes without an error proves little." This is the section's real lesson, and the exercise "Tests that can fail" builds on it well.
- The coding-assistant paragraph ends on the right point: "a wrong suggestion need not fail: it can name a function that exists, fit its signature, and run, yet compute something else."
- The TensorFlow tab's resolution (`tf.as_dtype(3)` is `tf.int32`, found by reading the source) shows the full loop of the figure, including the return arrow.

## Structural findings

- **The shared prose fits one framework.** See DOC-1. In the PyTorch view the prediction is "a $2 \times 3$ matrix of ones", the result is a $2 \times 3$ matrix of ones, and the text says "The prediction holds". Nothing was at stake. The interesting PyTorch fact, that `torch.ones(shape=(2, 3))` fails because the signature has no `shape`, is mentioned in the opening tab and in the signature tab but is not the thing verified in the cell.
- **A dropped thread.** "Discovering Names" sets up a concrete need ("we need random samples from a distribution other than the normal distribution") and lists twenty names. The next part begins "We set the samplers aside, since the first exercise completes the search for a gamma sampler". The reader was following one search and is told to finish it as homework while the text starts another. Carry one example through all four steps. The gamma sampler is the better candidate: its scale-versus-rate parameterization is a real ambiguity in every framework, the signature does not settle it without reading, and a sample mean verifies it. That is the whole loop, and it would also use the probability section just finished.
- **Position and weight.** Last in the chapter, short, and practical: right. But the section is close to being a how-to page, and the chapter would lose little if it moved to the Tools part with a paragraph left here. If it stays, it should earn its place with one well-chosen investigation.

## Passage-level findings

**DOC-1** (major, pedagogy). Opening, second paragraph, and "Verifying Behavior".

> The call `ones(2, 3)` looks like a request for a $2 \times 3$ matrix of ones, but whether it is one depends on how the function reads its arguments

For PyTorch it is one. The suspense the shared text builds ("at the end of the section they explain the result above") resolves, for the default reader, into confirmation of the obvious. Options: use the gamma parameterization as the shared puzzle (see above); or, per framework, pick a call whose behaviour is surprising in that framework (for PyTorch, for example, whether `x.std()` divides by $n$ or $n - 1$, which the exercise "Standard-deviation conventions" already uses and which connects to the preprocessing and probability sections; or what `torch.arange(12)` returns as a type, which the first section had to work around).

**DOC-2** (moderate, thought). Under "Reading Documentation", first sentence.

> We set the samplers aside, since the first exercise completes the search for a gamma sampler, and inspect a function that we think we already know

See the structural note. As written, the sentence tells the reader that the example they just invested in was a prop.

**DOC-3** (minor, explanation). Under "Reading Documentation", PyTorch tab.

> The signature `ones(*size, *, out=None, dtype=None, ...)` collects all positional arguments into the shape `size`, and the data type follows the lone `*`

The signature PyTorch prints has `*size` followed by a lone `*`, which is not valid Python and will puzzle a reader who has just been taught the two conventions separately. One clause would help: "(PyTorch's built-in functions print both, although `*size` alone would already make the rest keyword-only)".

**DOC-4** (minor, pedagogy). Under "Reading Documentation".

> help(torch.ones)

The stored output is about 40 lines of docstring. The text says "The first lines of the output show the function's *signature*" and then discusses only the signature. Fine, but the reader scrolls past a long block to find the continuation. Consider printing the first few lines only, or use `inspect.signature` where it works, and tell the reader to run `help` in full themselves.

**DOC-5** (minor, language). Under "Verifying Behavior", last paragraph.

> They complete the tools of this chapter: tensors, tables, linear algebra, derivatives, probability, and a way to find what the book leaves out.

A good closing list for the chapter. It is doing the job of a chapter conclusion from inside the last section; keep it, since the chapter has no other ending.

## Exercises

Five exercises (the first three differ by framework). "Scale versus rate" and "Standard-deviation conventions" are well chosen: both are real ambiguities that only the documentation or a test resolves, and both connect to earlier sections. "Tests that can fail" (rounding rules; stability of `argsort`) is excellent and teaches the logic of testing. "Reading the source" on `d2l.plot` is reasonable; if the plotting utilities leave the calculus section (CAL-9), update its reference.

- "Checking a coding assistant's answer" depends on whatever assistant the reader has and on the day; part 2 asks for a function "no library function provides", which is a claim that may not stay true. Keep part 1 and rephrase part 2 around verifying any one suggestion.
- If the gamma sampler becomes the section's worked example, exercise 1 needs a replacement; the standard-deviation exercise can take its place and a new one on a different sampler can follow.

## Still open from October

Nothing specific to this section was raised in October.
