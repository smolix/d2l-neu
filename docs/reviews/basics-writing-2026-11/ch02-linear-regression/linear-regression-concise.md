# Concise Implementation

Source: `chapter_linear-regression/linear-regression-concise.md`, about 2,300 words of prose before the exercises (all tabs; about 1,500 for the PyTorch reader).

## Verdict

A short section with a clear job, done competently: replace the hand-written layer, loss and optimizer by the framework's, and observe that the trainer does not change. The opening motivates it well. But the section has little new to teach, and it fills the space by repeating itself and the previous section. One fact, that the framework loss is twice the scratch loss so the learning rate must be halved, is stated in the opening, under the loss, under training, after the plot, in the closing, in the summary and in exercise 1. The check of the recovered parameters repeats the scratch section's explanation almost word for word. The most important fix is to tell the factor of 2 once and to replace the repeated checks by a direct comparison with the scratch model.

Stored outputs agree with the prose (validation loss $1.10 \times 10^{-4}$, parameter errors $-5.0$, $6.6$ and $-7.5 \times 10^{-4}$).

## Scores

| Clarity of explanation | Clarity of thought | Pedagogy | Language | Ease and enjoyment |
|---|---|---|---|---|
| 4 | 3 | 3 | 4 | 3 |

## What works

- **The opening.** "The image classifier ResNet-50 ... has 25.6 million parameters in 161 tensors, and it is trained with update rules that keep a running average of past gradients for every parameter." One concrete model makes the case for frameworks, and the next sentences add the point that hand-written code fails silently: "The model still trains, only less well."
- **The meaning of the mean squared error**: "its square root is the typical size of an error, in the units of the label." A sentence many texts omit.
- **The PyTorch note on `nn.MSELoss`**: with labels of shape $(32,)$ against predictions of shape $(32, 1)$ "it warns, and then computes the mean over the $32 \times 32$ matrix of differences anyway; training proceeds on the wrong loss." It picks up the bug from the scratch section's opening and tells the reader what the tool they will actually use does about it.
- **The closing rule**: the book "writes a part by hand only to show how it works or when the framework does not provide it."
- **Exercise 5** (optimization error against estimation error across training-set sizes) is a real question with a quantitative answer.

## Structural findings

- **One fact, seven tellings (major).** See CON-1.
- **The verification is a rerun of the previous section's.** The paragraphs after the plot and after the parameter printout re-derive the factor of 7 per epoch, the noise floor, the standard error $\sigma/\sqrt{n}$ and the remaining bias error. The reader finished those derivations a page ago. See CON-2.
- **Tab fragmentation.** Every subsection has four per-framework paragraphs, five such groups in all, and the shared prose between them is thin. For the PyTorch reader this is acceptable; in the source it means more than a third of the prose is framework notes.

## Passage-level findings

**CON-1** (major, thought). Opening, "The Loss Function", "Training", closing, "Summary".
Quote: "twice our scratch loss, so the learning rate 0.015 reproduces the steps"
The same statement appears at "twice ours, which doubles the step that a given learning rate takes" (opening), "the framework loss and its gradient are twice ours. This" (loss), the quoted passage (training), "The framework gradient is twice as large and the learning rate half as" (after the plot), "The only adjustment was the learning rate," (closing), and "gradient is twice ours, and the learning rate 0.015 reproduces the steps of" (summary). `linear-regression.md` already made the general point under "Minibatch Stochastic Gradient Descent". The October review complained that the two learning rates disagreed without comment; the correction has overshot. Fix: keep the full explanation once, under "Training", where the number 0.015 appears. Reduce the opening to "one difference needs care, the scale of the loss", the loss subsection to the definition, and the summary to one clause. Consider also removing the mismatch at its source (see the chapter file): if the scratch loss were the plain mean squared error, or the concise loss returned half of `nn.MSELoss`, neither this section nor the weight decay section would need a conversion.

**CON-2** (moderate, thought). After the parameter printout under "Training".
Quote: "$4.2 \times 0.97^{320} \approx 2.5 \times 10^{-4}$ short of it after the"
This sentence and the two before it restate, with the same numbers, the paragraph under "Recovered Parameters" in the scratch section. Fix: replace by one sentence ("The errors have the size we found for the scratch model, for the same reasons") and, better, show the claim made at the start of the section. The text says that "From the same initial parameters and on the same minibatches, the two models compute the same updates" but never demonstrates it; exercise 1(b) does. Moving a small version of that exercise into the text (train both on the full batch from the same start and print the largest difference between their parameters) would give the section a result of its own.

**CON-3** (moderate, explanation). PyTorch tab under "The Fully Connected Layer" and under "Training".
Quote: "width shrinks with the number of inputs. The trainer of this chapter does"
The PyTorch model defines a method `weight_init` that nothing in this chapter calls, and the training cell then initializes the layer by hand with the same two `nn.init` lines. The prose explains: "The trainer of this chapter does not call `weight_init`; the trainer of :numref:`sec_use_gpu` does". A reader sees an unused method, duplicate code, and a sentence about a trainer three chapters ahead. Fix: either call the method in the cell (`model.net.apply(model.weight_init)` after the first forward pass), which removes the duplicate lines and the explanation, or define the method where it is first used.

**CON-4** (minor, explanation). After the plot under "Training".
Quote: "This is the behavior of the scratch model with every loss doubled."
True, but the reader must hold the scratch plot in memory to see it. A cell that draws the scratch and concise validation curves on one axis, or prints the ratio of the two final losses, would show it.

**CON-5** (minor, language). Under "Training".
Quote: "problem training would still converge, but it would no longer repeat the"
The paragraph ends by explaining what would happen at a learning rate nobody proposed (0.03 with the framework loss). It can be cut.

**CON-6** (minor, language). Summary.
Quote: "the estimation error of least squares on 1,000 noisy examples, and its"
The summary re-reports the parameter error, the standard error and the noise floor, which are results of the previous section. Three sentences would do: what the framework provides; that `Module`, `DataModule` and `Trainer` are unchanged; that a constant factor in the loss changes the effective learning rate.

## Exercises

Five problems, all quantitative. Exercise 1 (loss scaling, with the two models trained to agreement) and exercise 5 are the best. Exercise 3 (two outputs; the framework loss averages over all entries, so the gradient per output halves) extends the section's one lesson in a useful direction. Exercise 4 (feature scale and step size) is good but its preamble is a paragraph of theory that belongs in the text or in the scratch section's learning-rate subsection. Exercise 2 (time scratch against concise for 10, 100 and 1,000 epochs) is close to busywork: the answer is "nearly the same, since both run the same loop", and the three durations add nothing.

## Still open from October

"The scratch and concise sections disagree on the learning rate without comment." Resolved, and now over-explained (CON-1).
