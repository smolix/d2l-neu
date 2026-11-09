# AlexNet and Learned Image Representations

Source: `chapter_convolutional-modern/alexnet.md`. About 2,800 words of prose before the exercises, 670 words of exercises.

## Verdict

The strongest of the four sections in this half, and the nearest to the bar. It asks a real question ("Why did learned features take so long to beat engineered ones, and what changed when they did?"), answers it with three causes, and backs two of them with arithmetic the reader can redo: the 2,048 crops per image and the back-of-envelope estimate of the training time on two GTX 580s. The layer-by-layer cost table then produces the fact that drives the next section. I checked every number in the table and in the compute estimate and they are right. The main defect is structural: a 300-word introduction of warmup and cosine decay sits inside "Training on Fashion-MNIST", although it is the shared training procedure of the whole chapter and has nothing to do with AlexNet. The second defect is that the experiment cannot show what the section claims, which the text admits honestly but late.

## Scores

| Dimension | Score |
|---|:-:|
| Clarity of explanation | 4 |
| Clarity of thought | 4 |
| Pedagogy | 3 |
| Language | 4 |
| Ease and enjoyment | 4 |

## What works

- The opening paragraph gives the two-stage pipeline a mechanism, not just names: "The training loss could improve the classifier but never the features, so progress depended on people inventing better features." That one sentence is why end-to-end learning matters.
- The hardware estimate under "Hardware" (about $6.5 \times 10^{17}$ operations for the paper's 90 epochs, then "The two GPUs need two to three days for that at their peak, and the actual run took five to six days. A CPU at its peak would have needed more than a month.") is the kind of calculation MacKay would do: order-of-magnitude, checkable, and it lands on the historical fact.
- "A crop can start at any of about $32 \times 32$ offsets and can be mirrored or not, so each image yields about $32 \cdot 32 \cdot 2 = 2{,}048$ different inputs" turns "data augmentation" into a count.
- :numref:`tab_alexnet_cost` and the paragraph after it ("The convolutional layers hold 8% of the 47 million parameters and perform 95% of the multiply-adds") give the reader a tool, the parameter and multiply-add ledger, that the rest of the chapter reuses. The closing "The next section asks whether a network needs that head at all" is a model section ending.
- The reading of the result is honest: "passes LeNet's accuracy in a third of the epochs, but only by a point or two, while it performs 2,250 times the arithmetic per image". The stored run (validation accuracy 0.91, training loss 0.22, validation loss 0.25) matches the prose.
- Exercises 1, 5 and 6 (memory ledger, arithmetic intensity, batch size against memory) are real problems that extend the section's ledger. Exercise 3 (compare your own first-layer filters with :numref:`fig_filters` and explain the differences from the input data) is excellent.

## Structural findings

- **The shared training procedure is defined in the wrong place.** Warmup, the cosine formula :eqref:`eq_cosine_warmup`, the saved function `cosine_warmup`, and the argument for both are introduced in the middle of "Training on Fashion-MNIST". Sections 7.2, 7.3 and 7.4 then restate the procedure in full each time (five restatements across the half-chapter) and redefine the same `configure_optimizers` cell four times. The writing guide's rule for shared objects (§7.5: define once, in a place of its own, under a heading that names it) applies. Give it a short `###` subsection of its own, for instance "The Training Procedure of This Chapter", state once that every network in the chapter uses it and why (comparisons need one procedure; the decay makes the last epoch a fair summary), and let later sections cite it in one sentence.
- **The experiment cannot support the thesis, and the reader learns this only afterwards.** The section argues that scale (data, compute) made learned features win. Fashion-MNIST upsampled eightfold has neither the data nor the detail, so AlexNet gains one or two points over LeNet. The text says so after the run. Say it before: "Fashion-MNIST cannot show what AlexNet's capacity is for; we train on it to check the implementation and to measure cost. Expect a small gain." Then the reader reads the plot with the right question.
- **Nothing in the section lets the reader see a learned feature of their own network.** :numref:`fig_filters` is the emotional center of the section (the network rediscovers oriented edge detectors), yet the only follow-up is exercise 3. A four-line cell that displays the 96 first-layer kernels of the model just trained, with one sentence on why they look blurrier than the ImageNet ones, would let the code teach the section's main idea. This passes the "code teaches; it does not draw" test because the picture is a computed result.

## Passage-level findings

**ALX-1** (minor, explanation). Section opening, second paragraph. Quote: "The winners of the two previous years, both built on engineered features, had top-5 errors of about 28% and 26%."
The paragraph carries six numbers in four sentences (1000, 28%, 26%, 15.3%, 26.2%, then "tens of millions"). The reader needs two: the runner-up's 26% and AlexNet's 15%. Drop the two earlier winners, or move them to one clause: "after two years in which the winning error had fallen by two points".

**ALX-2** (minor, explanation). Under "Learned Features, Data, and Hardware". Quote: "A representation is learned when the loss decides what the features compute."
A good definition, but it arrives as the first sentence of the body after the opening has already used "learned features" four times. Move it into the first paragraph of the section, next to the description of the two-stage pipeline, where the contrast is being set up.

**ALX-3** (minor, explanation). Under "Hardware". Quote: "for the forward pass and the two products of the backward pass".
"The two products" is unexplained. The reader who knows backpropagation from chapter 4 can reconstruct it (each layer computes one product for the gradient with respect to its input and one for the gradient with respect to its weights), but the estimate should not depend on that. Write: "a training step costs about three times as many: one pass forward, and on the way back each layer computes two products of the same size, one for the gradient of its weights and one for the gradient it hands to the layer below".

**ALX-4** (minor, language). Under "Hardware". Quote: "instead of a few fast cores with branch predictors, speculative execution, and large caches".
Three pieces of processor jargon that the book has not defined and does not use again. "Instead of a few fast, complicated cores" says what the argument needs.

**ALX-5** (minor, explanation). Under "The AlexNet Architecture". Quote: "AlexNet changes the size of every part and three of its components."
The three components are never listed as three. The subsections cover ReLU and dropout; max-pooling in place of LeNet's average pooling appears only in the figure caption and as a fact ("Max-pooling with $3 \times 3$ windows and stride 2 follows"), with no reason given, and the windows overlap (3 by 3 at stride 2), which is also new. Name the three in this sentence and give max-pooling its one sentence of justification, or say "two" and treat pooling as a detail.

**ALX-6** (moderate, thought). Under "ReLU Activations". Quote: "so the gradient passes the active units of a layer unchanged, however many layers lie in between".
This overstates. The gradient passes the ReLU unchanged on active units, but it is still multiplied by each layer's weights, and inactive units pass nothing. The next section on batch normalization and the ResNet section both exist because gradients in deep ReLU networks are still badly scaled. Write: "so a ReLU, unlike a sigmoid, does not shrink the gradient on its active units; what remains is the scaling by the weights of each layer, which the initialization below is chosen to control."

**ALX-7** (minor, thought). Under "ReLU Activations". Quote: "why LeNet needed a learning rate of 2.0".
The sentence before it gives a factor of at most 1/4 per sigmoid; four sigmoids give $4^{-4} \approx 1/256$, while the learning rate was raised twentyfold. The reader who multiplies will wonder about the gap. Either give the number and say the rate of 2.0 is a compromise between the first and the last layers, or soften to "This is one reason LeNet needed".

**ALX-8** (minor, explanation). Under "Implementation". Quote: "a ReLU sets about half of its inputs to zero and halves the second moment of what passes through it".
Correct and compact, but it is the whole justification for He initialization and it restates a result from chapter 4 in a clause. The guide's rule (restate the fact, then cite) is followed; the sentence that follows is a 50-word one with two definitions in it. Split after "doubling the variance."

**ALX-9** (moderate, pedagogy). Under "Training on Fashion-MNIST". Quote: "We also change the learning rate during training, as AlexNet's authors did: they divided it by 10 whenever the validation error stopped improving."
The sentence suggests continuity with the paper and then introduces a different schedule that dates from 2016 and 2017. The reader is given no reason to prefer warmup and cosine decay over the step schedule just described, other than "A common schedule handles both ends". See the first structural finding: move this to its own subsection and say plainly that it is the modern default, chosen for the whole chapter, and not part of AlexNet.

**ALX-10** (minor, explanation). Same passage. Quote: "we use 0.03".
The class default is `lr=0.1` and the run passes `lr=0.03`. No reason for 0.03 is given here; the reason appears two sections later for NiN ("0.03 is the largest at which NiN trains reliably") and four sections later for ResNet. One clause now ("the largest peak at which all the networks of the next section train reliably") spares the reader the puzzle.

**ALX-11** (minor, thought). Under "Training on Fashion-MNIST". Quote: "LeNet reaches about 88–90% after 30 epochs."
The comparison is between a sigmoid network at a constant rate of 2.0 for 30 epochs and a ReLU network on a decaying schedule for 10. That is acceptable for "the reader's previous model against this one", but the conclusion drawn ("passes LeNet's accuracy in a third of the epochs") credits the architecture for what the schedule, the activation and the initialization also contribute. One clause would do: "with a different activation, initialization, and schedule as well as a different size".

**ALX-12** (minor, language). Summary. Quote: "and two GPUs made a network of 60 million parameters trainable in days".
The summary is a faithful list of what was said, 120 words, seven numbers. It does not answer the opening question in a sentence. Lead with the answer: "Learned features overtook engineered ones when three things were in place at once: a dataset large enough to constrain tens of millions of parameters, GPUs that could compute their gradients in days, and ReLU, dropout and augmentation to keep such a network trainable." Then the ledger and the Fashion-MNIST result.

## Exercises

Six problems, graded, all tied to the section's ledger. The best set in this half. Exercise 2 (receptive fields) is a drill on a formula from chapter 6, acceptable because the answer ("where does a unit first see the whole image?") is interesting. Exercise 4 (dropout on a 6,000-image subset) is a good experiment with a predictable and instructive answer. No cuts. If the filter display moves into the text (third structural finding), exercise 3 keeps its second part.

## Still open from October

Nothing specific to this section. The October review's general complaint about training runs without a readout is fixed here.
