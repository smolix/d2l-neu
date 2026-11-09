# Implementation from Scratch

Source: `chapter_linear-regression/linear-regression-scratch.md`, about 3,900 words of prose before the exercises (all tabs; about 2,900 for the PyTorch reader).

## Verdict

The second half of this section, "Checking the Implementation", is the best sustained piece of teaching in the chapter. Before training, the text derives what a correct program must produce: a loss floor of $\sigma^2/2 = 5 \times 10^{-5}$ and a fall by a factor of about 7 per epoch. Then it trains, and the plot shows a straight line on a log axis that levels off at $5 \times 10^{-5}$ after seven or eight epochs. Then it explains why the loss curve could not have revealed a parameter error of $5 \times 10^{-4}$. A reader learns a method here, not just a model. The first half is weaker for a structural reason: the section is called "from scratch", but its training loop is the book's production trainer, with a batch hook, an on-device averager, a gradient-clipping call and a scheduler step, and the simple four-operation step that the prose describes never appears as code on its own. The most important fix is to run the bare loop first and add the bookkeeping afterwards.

Stored outputs agree with the prose: gradient difference 0; validation losses $2.4$, $5.0 \times 10^{-5}$, $5.0 \times 10^{-5}$, `nan` at the four learning rates; parameter errors $-5.4 \times 10^{-4}$, $5.8 \times 10^{-5}$, $-2.2 \times 10^{-5}$. I checked the arithmetic of the predictions ($0.97^{64} \approx 0.14$, $16.6 \times 0.14^7 \approx 2 \times 10^{-5}$, $0.97^{320} \approx 6 \times 10^{-5}$, $16.6 \times 0.997^{640} \approx 2.4$); all correct.

## Scores

| Clarity of explanation | Clarity of thought | Pedagogy | Language | Ease and enjoyment |
|---|---|---|---|---|
| 4 | 4 | 4 | 4 | 4 |

## What works

- **The opening bug.** "Subtracting a label vector of shape $(32,)$ from predictions of shape $(32, 1)$, for example, raises no error, because broadcasting forms the $32 \times 32$ matrix of all pairwise differences. Training on the loss computed from it lowers that loss steadily while learning the wrong weights." A concrete, slightly alarming fact that motivates the whole section, and it returns in the loss cell and in exercise 2.
- **"the gradient is the error-weighted input"**, followed at once by a numerical comparison of the hand formula with automatic differentiation. Formula, plain sentence, check.
- **Prediction before measurement** (under "The Expected Loss"): the expected loss "is half the squared distance between the parameters and the true ones, plus a floor of $\sigma^2/2$". The reader is told what the plot will show and why, and the plot then shows it.
- **"The loss does not reveal errors of this size."** The paragraph computes that a parameter error of $5 \times 10^{-4}$ raises the loss by $4 \times 10^{-7}$, less than 1% of the floor, and concludes that parameters, not losses, expose implementation errors. It then fences the claim: the comparison works only when the minimizer is unique, which fails for deep networks. This is clarity of thought at the level of the model texts.
- **The learning-rate sweep.** One factor, $1 - \eta$, predicts all four outcomes, including `nan` at $\eta = 3$, and the remark on feature scale ("lowers the limit to about $2/s^2$, which is one reason why inputs are standardized") connects it to practice.
- **The exercises** (see below).

## Structural findings

- **The loop is not from scratch (major).** See SCR-1. The reader is promised "the four components of training ourselves, using nothing but tensors and automatic differentiation". What `fit_epoch` shows is fourteen lines of which six serve later chapters or bookkeeping.
- **"One Training Step" has no code.** The subsection describes four operations and two ordering rules in prose and per-framework notes. The code that performs them appears sixty lines later, inside `fit_epoch`. See SCR-2.
- **The first half is interleaved with library construction.** `prepare_batch`, `Averager` and the per-framework `fit_epoch`/`validate` are parts of `d2l.Trainer` that happen to be defined here. They would sit better with the `Trainer` class, leaving this section to the model, the loss, the update, one loop, and the check.

## Passage-level findings

**SCR-1** (major, pedagogy). PyTorch `fit_epoch` cell under "Training and Validation Passes".
Quote: "if self.gradient_clip_val > 0:  # To be discussed later"
The first training loop a reader meets calls `self.clip_gradients` (not defined anywhere yet), steps an optional scheduler, maps a `prepare_batch` hook over the loader, wraps the result in `as_metrics`, detaches a dictionary of metrics and feeds an `Averager`. The four operations of SGD are four of its fourteen lines. Fix: before this cell, train with the bare loop on the objects already in hand:

```python
optim = model.configure_optimizers()
for epoch in range(3):
    for X, y in data.train_dataloader():
        loss = model.loss(model(X), y)
        optim.zero_grad()
        loss.backward()
        with torch.no_grad():
            optim.step()
    print(epoch, float(loss))
```

Then say that `Trainer.fit_epoch` does the same and adds averaging, validation and hooks for later chapters, and show (or only cite) the full method. The reader then knows which lines are the algorithm.

**SCR-2** (moderate, pedagogy). Under "One Training Step".
Quote: "One step of minibatch SGD consists of four operations. It computes the"
A paragraph of prose about clearing gradients and not recording the update, with no cell. The minibatch `X, y` and a fresh `model` are already defined from the gradient check. Fix: one cell that performs a single step on that minibatch and prints the loss before and after; the per-framework notes then annotate visible code. With SCR-1 this cell and the bare loop could be the same thing.

**SCR-3** (moderate, explanation). Under "The Gradient".
Quote: "$\mathbf{r} = \mathbf{X}\mathbf{w} + b - \mathbf{y}$. In this section $L$"
$L$ was the loss over the whole training set two sections ago and is redefined here as the loss of a minibatch; $\mathbf{X}$, $\mathbf{y}$ and $\mathbf{r}$ change the same way without comment. The text is honest about $L$, but an announced clash is still a clash (writing guide, section 8.2). Fix: write $L_{\mathcal{B}}$ for the minibatch loss here and in the update equation.

**SCR-4** (moderate, thought). Under "Epochs and Averages".
Quote: "metric on the device and converts the sums into Python numbers once per"
The third telling of the host-synchronization story (twice in `oo-design.md`, once here), and the second of the rule that the loss is listed first and the other metrics alphabetically. Neither matters for three parameters on a CPU. Fix: remove here; keep one telling where GPUs are introduced. The sentence "It weights each minibatch by its size, so the shorter last minibatch counts correctly" is the part worth keeping.

**SCR-5** (moderate, explanation). Under "The Expected Loss".
Quote: "itself, so a step along this gradient with learning rate $\eta$ multiplies"
The paragraph is correct and valuable, but it runs nine numbers through six sentences ("0.97", "$0.97^2$", "32 steps", "$0.97^{64} \approx 0.14$", "about 7", "16.6", "$0.14^7$", "$2 \times 10^{-5}$") and the reader can check the result only by reading a log-scale plot. Fix: let the code carry the check. After training, print the validation loss minus the floor next to the prediction $16.6 \times 0.14^k$ for epochs $k = 1, \ldots, 6$. The prose can then state the rule (a factor $1 - \eta$ per step, hence about 7 per epoch) and point at the table.

**SCR-6** (moderate, thought). Under "Recovered Parameters".
Quote: "remains, which makes the error of the bias negative in about three runs out"
A claim about the distribution of a sign over repeated runs, which a reader with one run cannot check and has no reason to want. It ends a paragraph that already names three sources of error of similar size. The same computation is repeated in the concise section. Fix: cut the sentence, or turn it into an exercise ("Predict the sign of the error of the bias after ten epochs, and how often it comes out that way over 20 runs").

**SCR-7** (minor, language). Under "The Parameters".
Quote: "scale of the initial weights matters a great deal, which is why the standard"
The argument is named `sigma` and defaults to 0.01. The noise of the data is $\sigma = 0.01$. Two different quantities share a symbol and a value for the rest of the chapter, and in the weight decay section the text has to compute with both ("$0.01\sqrt{180}$" is the initialization; "$\sigma = 0.01$" is the noise). Fix: rename the argument (`init_std`), or at least say here that it is unrelated to the noise level.

**SCR-8** (minor, language). Summary.
Quote: "weighted by minibatch size, and reads them from the device once per epoch."
Nine sentences, several of them restating implementation detail and numbers. A summary of three to six sentences would keep: the four components; gradient equals error-weighted input, confirmed against autograd; the expected loss predicts floor, rate and the stable range of learning rates; parameters expose errors that the loss hides.

## Exercises

Eight problems and the strongest set in the chapter. Each one asks for a prediction and then a check, which is the section's own method: zero and huge initialization (1), the shape bug worked out to the weights it converges to (2), the Hessian by hand and by autograd (3), stability against minibatch size with a derivation for single examples (4), reshuffling (5), robust losses on the corrupted label of section 1 (6), uncleared gradients as a recurrence (7). They are graded, with the hard ones marked. Exercise 8 (Planck's law) is a different kind of problem (a nonlinear one-parameter fit) and is a pleasant change of scene; it would be better with one sentence saying why it is here (the same `Module` and `Trainer` fit any differentiable model). Three exercises that `oo-design.md` defers to this section (early stopping, test split, per-minibatch synchronization) should move here if they are kept.

## Still open from October

"Keep: 'the gradient is the error-weighted input'." Kept. No open item.
