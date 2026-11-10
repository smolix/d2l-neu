# Writing pass: `chapter_builders-guide/numerics.md` (5.5 Numerical Formats)

Prose words (checker count, all tabs): **5706 before, 5620 after** (-86).
Checker: "OK: only prose changed". `tools/lint_source.py`: clean. No code
cell, label, figure, or slide was touched. The net change is small because
about 190 words of review-requested explanation (NUM-1, NUM-3, NUM-8, NUM-9,
NUM-10) went in while about 275 words of tab prose and repetition came out.

## Before/after pairs

1. Opening, second paragraph.
   - Before: "Halving the bits halves memory and memory traffic, but it makes
     the numbers coarser, and a training run can fail in two ways without an
     error message."
   - After: "Not without some care. Sixteen bits make the numbers coarser,
     and a training run can then fail in two ways, neither of which produces
     an error message."
2. Range and Precision.
   - Before: "90000, a finite result that is wrong in the fourth digit. Both
     formats spend 16 bits, but they divide them differently."
   - After: "90000, a finite result that is already wrong in the third digit.
     Both formats spend 16 bits. They merely divide them differently."
3. Subnormals.
   - Before: "`tiny`, the smallest *normal* value, below which the format
     switches to *subnormal* numbers, whose precision falls with every
     halving until they reach zero"
   - After: "What lies below `tiny`? There the exponent cannot shrink any
     further, and the format continues with *subnormal* numbers, which are
     evenly spaced all the way down to zero. In fp16 they are the multiples
     of $2^{-24}$, so a value near $3 \times 10^{-5}$ still keeps 9
     significant bits, while one near $10^{-7}$ keeps a single bit."
4. Promotion (mxnet/pytorch/jax tab).
   - Before: "Two float tensors promote to the wider format, so an fp16
     computation that meets one stray fp32 tensor continues in fp32 and needs
     twice the memory from then on."
   - After: "Two float tensors promote to the wider format. This is the one
     rule worth remembering: an fp16 computation that meets a single stray
     fp32 tensor continues in fp32, without a message, and needs twice the
     memory from then on."
5. End of "Casting a Model".
   - Before: "For *training*, the rounding experiment above has already shown
     what goes wrong: updates that are small relative to the weights are lost
     when the weights are stored in 16 bits."
   - After: "For *training* a cast model will not do. Its weights are exactly
     the 16-bit weights that lost their updates in the rounding experiment."
6. Opening of "Mixed-Precision Training".
   - Before: "*Mixed-precision training* :cite:`…` separates storage from
     computation."
   - After: "We seem to be stuck: the weights need 32 bits, yet the speed
     lies in 16. *Mixed-precision training* :cite:`…` gets out of this by
     separating storage from computation."
7. After the training experiment.
   - Before: "We did not time the runs. The speed of 16-bit arithmetic
     depends on …"
   - After: "What have we gained? On the evidence of this experiment, nothing
     yet, since we did not time the runs. The speed of 16-bit arithmetic
     depends on …"
8. Summary.
   - Before: "Exponentials and long sums are computed in stable forms and
     with fp32 accumulators."
   - After: "Compute exponentials in their shifted form and long sums in an
     fp32 accumulator."

## Repetition removed

- "The loss values stay on the device …" paragraph, identical in four tabs:
  now one shared sentence with a reference to `sec_use_gpu`; the JAX tab
  keeps only its own remark on `param_dtype`.
- "The parameters take half the bytes" at the head of four tabs: one shared
  sentence; the MXNet tab no longer repeats that Gluon refuses a mismatched
  input (said in the tab before the cell).
- "… all apply the shift" in three tabs: one shared sentence; each tab keeps
  one line naming its `logsumexp`. The re-explanation of why the shift works
  (owned by ch. 3, `subsec_softmax-implementation-revisited`) is one clause.
- Subnormal thresholds were given three times ($6.1 \times 10^{-5}$ in the
  format subsection, the loss-scaling opening, and after the measurement);
  the third is now "the subnormal range of fp16".
- Bytes per parameter (owned by 5.2 `sec_parameters`): still one clause with
  a reference; no count re-derived here.
- PyTorch `GradScaler` caveats (`unscale_`, gradient accumulation) cut.
- TF32, promotion, `finfo`, Keras policy, and Gluon `multi_precision` tab
  notes shortened; the "weak scalar" term dropped.

## Review findings

Applied: NUM-1 (third digit; hand check of 65504 and 90112 from the formula),
NUM-3 (subnormals defined with numbers), NUM-4 ("half the smallest
subnormal"), NUM-5 ("This is why …"), NUM-7, NUM-8 (prose half: why 100
steps suffice), NUM-9 (prose half: the limit of mixed precision is its own
paragraph), NUM-10 (prose option: the damage is slight for this network; the
MXNet tab no longer says the gradients "stay inside the range of fp16"),
NUM-11, NUM-14. Structural 3 and 4 in part: TF32 is announced as a detour
and its tabs trimmed; promotion prose reduced to the one consequence.

Already addressed in the file, left alone: NUM-12 (the accumulation
paragraph and its tab notes are unchanged).

Deferred:
- Structural 1 / NUM-8 (bf16-weights failing arm): needs a new training run.
- Structural 2 / NUM-9 (timing): needs a GPU cell.
- Structural 3 / NUM-6 (move or remove TF32): its cell cannot be removed or
  moved in this pass.
- Structural 5 / NUM-2 (drop tf32 and fp8 rows): `bg-float-formats.svg`
  draws both rows and its caption says it draws the table; needs the figure
  regenerated. The fp8 aside is shortened instead.
- NUM-10 (experiment in which scaling rescues a run): needs a new cell.
- NUM-13 (increment $2^{-10}$): needs a changed cell and outputs.

## Factual claims added or changed

- 90112 is wrong in the **third** digit (was "fourth").
- fp16: $e$ runs from $-14$ to 15; largest number
  $2^{15}(2 - 2^{-10}) = 65504$.
- bf16 numbers between $2^{16}$ and $2^{17}$ are 512 apart;
  $175 \cdot 512 = 89600$, $176 \cdot 512 = 90112$.
- fp16 subnormals are the multiples of $2^{-24}$; a value near
  $3 \times 10^{-5}$ keeps 9 significant bits, one near $10^{-7}$ a single
  bit.
- $3 \times 10^{-8}$ is half the smallest subnormal.
- After 100 steps the MLP gets "about two thirds" of the validation images
  right (stored: 0.652 to 0.662 across the four frameworks; seeds are random).
- "The 16 GB of the opening are what the 8-billion-parameter model occupies
  once it is cast for inference"; for training, mixed precision saves only
  activation memory and matmul time (already in the text, now emphasised).
- Loss scaling: "For this small network the damage is slight. It need not
  be: when a loss is averaged over many elements, or a gradient has passed
  through many layers, far more of the entries can fall below the threshold,
  and Micikevicius et al. (2018) report networks that train in fp16 only
  with the remedy that follows." **Please check this attribution**; it is
  from memory of the paper, not verified here.
- Shared sentence: "the log-softmax and cross-entropy functions of every
  framework apply it [the shift]" (TF tab adds: Keras losses only with
  `from_logits=True`).
- TensorFlow `finfo` tab no longer mentions `as_numpy_dtype`; JAX TF32 tab
  no longer mentions `jnp.dot(..., precision=...)`; MXNet TF32 tab no longer
  recommends `float64`.

## Slides

No statement in the slides section disagrees with the revised body. The
slides still carry the tf32 "three to four digits" lines and the 128 GB
figure, both consistent with the text.
