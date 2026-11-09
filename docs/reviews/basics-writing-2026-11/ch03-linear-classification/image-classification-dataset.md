# Image Classification Data

Source: `chapter_linear-classification/image-classification-dataset.md`. About 1,700 words of body prose (including four framework tabs), 690 words of exercises, 12 code cells in the PyTorch tab.

## Verdict

A competent utility section with a good opening and a good habit (look at the data before training). The opening does what the guide asks: it poses a real question ("Which dataset should we use?"), gives the two criteria, and shows MNIST and ImageNet each failing one of them before Fashion-MNIST is introduced. After that the section is a walk through a loader, and the prose spends its attention on details that do not matter to the reader (how many images land in the last minibatch, how a `tf.data` buffer shuffles) while passing quickly over the one thing that will matter for the rest of the chapter: what the ten classes look like and which of them resemble each other. The most important fix is to trade the loader arithmetic for a better look at the data.

## Scores

| Explanation | Thought | Pedagogy | Language | Enjoyment |
|:-:|:-:|:-:|:-:|:-:|
| 4 | 4 | 3 | 4 | 3 |

## What works

- The three-paragraph opening. "MNIST ... is small but too easy" and ImageNet "separates models well, but training a network on it takes hours to days" set up Fashion-MNIST as the answer to a problem.
- The candid remark that the test images serve as validation data: "A test set that we consult again and again while we compare models stops being an unbiased test of the model we finally choose". It plants the question that `generalization-classification.md` answers, in one sentence and without machinery.
- "A look catches errors that no loss curve reveals: labels attached to the wrong images, images flipped or transposed, a channel axis in the wrong place." This is advice from someone who has debugged data, and it gives the last subsection a purpose.
- The observation after the image grid ("the shirts and the coats above differ in details that $28 \times 28$ pixels barely resolve") predicts the confusion matrix of the next-but-one section. The stored output does show shirts and coats in the first 16 validation images.

## Structural findings

- **The look at the data is too thin.** Sixteen images in stored order show whatever classes come first. The opening claims that "a shirt can look like a T-shirt, a pullover, or a coat"; the section should show it. A grid with one row per class (or just the four tops side by side) would let the reader see the difficulty that the rest of the chapter measures, and would give the later confusion matrix a visual referent.
- **Loader detail crowds the prose.** See ICD-1 and ICD-2. The section's idea is "a dataset hard enough and small enough, delivered in minibatches of a known shape". The incomplete last minibatch, worker processes, and compiled-step retracing are implementation notes.
- **A promise without a follow-up.** The timing cell asks the reader to compare loader time with "the time per epoch" in the following sections, but no later section reports a time per epoch (ICD-4).
- **Imports.** The section has two import cells in a row, and the first imports `math`, `numpy`, `nn` and `F`, none of which the section uses. The same seven-line block opens every notebook of the chapter. This is a code-guide matter, but the reader sees it six times.

## Passage-level findings

**ICD-1** (moderate, pedagogy). Anchor: "With minibatches of 64, the 60,000 training images make 937 full", under "Reading Minibatches".
A full paragraph (about 110 words) on the 32 images of the last minibatch, including the alternative of dropping it "at the price of leaving 32 images out of each epoch". The same arithmetic returns in `classification.md` ("end in a minibatch of 16") and in `softmax-regression-scratch.md` ("$60{,}000 = 234 \times 256 + 96$"). No argument in the chapter depends on it. Reduce to one sentence: "The last minibatch of an epoch is smaller than the others (32 images here); both loaders keep it, so every epoch covers every image."

**ICD-2** (moderate, language). Anchor: "JAX has no loader for image datasets, so we build a pipeline with", JAX tab under "Reading Minibatches".
The TensorFlow and JAX tabs each contain a 60 to 110 word paragraph listing six pipeline stages in one sentence and then explaining that the `Trainer` "traces" or "compiles its training step once for the full minibatches and once more for the shorter last one". Tracing and compilation have not been introduced. A reader on these tabs gets denser and more technical prose than a reader on the PyTorch tab, for the same lesson. Move the stage list into code comments (the code is five lines and nearly self-explanatory) and delete the retracing sentences.

**ICD-3** (minor, pedagogy). Anchor: "of images takes $64 \times 784 \times 4$ bytes, about 200 KB, and the whole", under "Reading Minibatches".
The memory arithmetic is correct but has no stated purpose. Say why it is worth knowing: the whole dataset fits in memory many times over, so for Fashion-MNIST the loader is limited by per-image work, not by reading from disk. That also sets up the timing cell.

**ICD-4** (moderate, pedagogy). Anchor: "take. When we train models in the following sections, compare it with the", under "Reading Minibatches".
The reader is told to compare the loader's 2 seconds with the time per training epoch, but the following sections never report an epoch time, so the comparison cannot be made from the page. Either have `softmax-regression-scratch.md` report the time of its training run and make the comparison there in one sentence, or drop the instruction and keep only the statement that loader time is a lower bound on epoch time. The summary sentence "we check that loading an epoch takes less time than training on it" describes a check the section does not perform.

**ICD-5** (minor, language). Anchor: "Fashion-MNIST has 7,000 images in each of its 10 classes: t-shirt, trouser,", under "Loading the Dataset".
The first class is "t-shirt" here and in `text_labels`, "T-shirt" in `softmax-regression.md` and `softmax-regression-scratch.md`. Pick one. The list of ten classes is also given three times in the chapter (the opening of `softmax-regression.md`, here, and before the confusion matrix).

**ICD-6** (minor, explanation). Anchor: "An image is a three-dimensional tensor: a number of *channels* $c$, one for", under "Reading Minibatches".
The definition of channels and of the two layouts arrives after four tabs of loader code that already mention "a channel axis". Move this paragraph before the loader code, where the term is first needed.

## Exercises

Six problems. "Shuffling and validation" and "Reshaping versus transposing" are good: short, conceptual, and each catches a real bug. "Upsampling and linear models" is a nice idea (a linear model cannot gain from a linear resize), though its second part depends on the section after next.

- **Cut** "Loader or compute" (the extended heatmap). It is a two-dimensional sweep with a plotting task and no question that the two one-dimensional sweeps before it have not asked.
- **Merge** "Throughput versus batch size" and the worker or parallel-loading problem into one problem with two parts. Three of six problems on loader throughput is too many for a section whose subject is a dataset.
- **Add** a problem about the data: compute the mean image of each class and the pairwise distances between the means, and predict from them which classes a linear classifier will confuse. It prepares the confusion matrix and the weight templates of `softmax-regression-scratch.md`.

## Still open from October

Not applicable; October did not comment on this section.
