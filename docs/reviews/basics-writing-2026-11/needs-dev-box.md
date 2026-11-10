# Changes that need the development machine

Recorded 2026-11-09. The writing pass is being done on a laptop with no GPU and
no framework environments, so nothing that changes a notebook code cell or
needs outputs re-captured was done. These items wait for the 4x4090 box.

## Requested by the author, blocked on code or re-capture

| Where | What | Why it is blocked |
|---|---|---|
| 1.6 Probability, "Simulated Coin Tosses" | The stored PyTorch and JAX samples of 100 tosses are both exactly [50, 50], right after the text says this is rare. The cell is unseeded, so this is luck (about an 8% chance per framework; TensorFlow has 43/57, MXNet 53/47). Re-execute the notebook and re-capture until the sample is not 50/50, or seed the cell with a value that gives an uneven split. | Needs execution and capture for pytorch and jax. |
| 2.6 Generalization, three-panel polynomial plot (GEN-5) | The degree is hidden in the x-axis label and the panels render small. Give each panel a title ("degree 1", "degree 3", "degree 19") and enlarge the figure. | Code cell (`xlabel=f'x (degree {d})'`, figure size). |
| 8.1 Sequence, rollout plot (SEQ-7) | The plot shows one-step predictions as a stand-in for the data. Plot the observed series (thin, grey) with the rollout on top and drop the one-step curve. | Code cell. |
| Ch. 3, 4, 6: a picture of a learned feature | No figure in the part shows what a model learned. Add: the ten weight templates of softmax regression (now exercise 5 of 3.4), a few first-layer units of the MLP (4.2), and the first-layer kernels or feature maps of the trained LeNet (6.6). Each is a short cell after training, `d2l.show_images` on the reshaped weights. | New code cells and outputs. |

## Accepted as they are (author, 2026-11-09)

- 3.4 and 5.3: prose that describes a range over repeated runs. This is a
  deliberate policy: the experiments were repeated to make sure the statements
  hold throughout.

## Slides to re-sync after the chapter 3 writing pass

The body of chapter 3 was revised; its slide sections were not. They still
carry a few statements the body no longer makes: "near 0.45 and 0.49" and
"7,850 parameters memorize little" and the threshold of 88 twice
(`softmax-regression-scratch.md`); the last-minibatch arithmetic and "trace
once more for the short shape" (`image-classification-dataset.md`); "In every
run", "40-65%", "slope about 1 to 2" (`environment-and-distribution-shift.md`);
the loss 2.27 for the dog (`softmax-regression.md`, now 2.3 in the body). None
is wrong enough to mislead, but the decks should follow the body.

## Found during the writing pass (2026-11-09)

- **4.7 house prices, JAX store.** The stored JAX run of the linear model has
  one validation fold near 0.28, which lifts its five-fold mean to about 0.17
  (the other frameworks give 0.144 to 0.148). The prose now says only that a
  single fold "now and then" does much worse. Re-capture JAX and check whether
  the outlier is a seed effect or a real difference; the slides still say 0.15.
- **4.6 dropout.** Body and slides say "which of them is ahead changes from
  run to run", but dropout is ahead in all four stored runs (by 0.6 to 2.1
  points, within epoch noise). Confirm on re-capture or reword.
