# Numerical Formats

Source: `chapter_builders-guide/numerics.md` (read to the slides marker, line 1451). Prose a PyTorch reader sees: about 3,000 words plus 550 words of exercises; across all four tabs the source holds about 4,600 words of prose in 54 tab blocks, and 20 code cells per framework.

## Verdict

This is the best-motivated section of the second half and close to the bar. It opens with a question the reader can hold ("Can we train in 16 bits and obtain the same model?"), states the two silent failures with numbers ($300^2$ is infinite in fp16, $1 + 0.001 = 1$ in bf16), and then earns each mechanism from a failure the reader has just watched. The weakness is that the experiments confirm only that nothing goes wrong. Mixed precision is never shown to be faster or smaller, the lost-update failure that master weights exist to prevent is never shown in training, and loss scaling is never shown to rescue anything. The reader is taught three remedies and sees zero diseases beyond one-line toys. The single most important fix is to add the failing arm to the training experiment (weights stored in bf16, no master copy) so that the curve stalls and the remedy is seen to work.

## Scores

| Dimension | Score |
|---|---|
| Clarity of explanation | 4 |
| Clarity of thought | 4 |
| Pedagogy | 3 |
| Language | 4 |
| Ease and enjoyment | 4 |

## What works

- The opening (first three paragraphs). It starts from what the reader has done ("Every number in our models so far has been stored in the 32-bit floating-point format, and we have not asked whether a model needs that many bits"), prices the question with the 8-billion-parameter model, and names both failure modes in one sentence each. Keep it as the model for the chapter's section openings.
- "Squaring 300 in the two 16-bit formats shows the two ways in which a format can fail." One cell, two outputs (`inf` and 90112), and the whole range-versus-precision trade is on the screen before any definition. This is the MacKay move.
- "Rounding of Small Updates". The $10^{-3}$ and $10^{-4}$ increments were chosen so that the three formats give three different outcomes, and the prose reads each line of the output against the spacing ($2^{-7}$, $2^{-10}$). The conclusion "A model whose weights are stored in 16 bits can therefore stop learning even though its gradients are correct" is exact and memorable.
- The log-sum-exp example: $e^{12} \approx 163{,}000$ against the fp16 maximum of 65504 makes a ch. 3 trick concrete in a new setting, in four sentences.
- The measurement before the loss-scaling definition ("About a third of the nonzero first-layer gradients lie below the smallest normal fp16 number"). The output (32.2%, 0.16%) matches the prose.
- The closing of "Numerical Failure Modes" ("These failures share one cause: a 16-bit format has too little range or too little precision for one intermediate value, even when the inputs and the result are representable"). One sentence that unifies three subsections.
- Exercises 1 and 3 ask for predictions from the section's own numbers.

## Structural findings

1. **The experiments never show a failure in training.** The line of argument is: 16-bit weights lose updates, so keep fp32 master weights; fp16 gradients underflow, so scale the loss. Both failures are shown only on a scalar. The training experiment compares fp32 with the finished recipe and reports that the curves coincide. A third curve (the cast model of "Casting a Model", trained directly in bf16 at a learning rate small enough that $\eta g$ falls below half the spacing) would stall, and then the master-weights curve would be the fix. That is the failure-then-remedy order the writing guide asks for in its section 3.2, and it would use the `net.bfloat16()` model that the section builds and then abandons.
2. **The benefit is never measured.** The opening promises that "recent GPUs multiply 16-bit matrices at a multiple of their 32-bit rate". The body says "We did not time the runs" and sends the reader to an exercise two sections ahead. The gpus section already runs GPU-guarded cells with a CPU fallback message; one such cell here (or in `gpus-devices-memory.md` main text, referenced from here) should show the speedup on a wide layer. As written, the reader pays the whole cost of the section and takes the benefit on trust.
3. **TF32 interrupts the argument.** "TF32 Matrix Multiplication" sits between the rounding experiment and dtype promotion. It concerns fp32 tensors, changes nothing the reader stores, and its cell prints `highest` and `high`. It also forces a `tf32` row into the format table before the reader can understand it. Move it after mixed precision as a short paragraph without a cell, or to the gpus section where the flag is actually set (`torch.set_float32_matmul_precision('high')` appears there in the imports).
4. **"Dtype Promotion and Model Casting" is documentation.** Promotion rules matter here only for one consequence (a stray fp32 tensor silently doubles memory), which is one sentence. The subsection spends a cell and four tab blocks on the rule table. "Casting a Model" is needed, but its job is to set up the failing arm proposed in finding 1; at present it ends in a forward-looking paragraph and the cast model is never trained.
5. **fp8 is still in the body.** The table row, a four-line aside, and an exercise. The October review listed fp8 as a cut. The exercise is good and can carry it alone; drop the table row and the aside.

## Passage-level findings

**NUM-1** (moderate, explanation). Under "Range and Precision": "90000, a finite result that is wrong in the fourth digit. Both formats".
90112 against 90000 differs in the third significant digit. More important, the display $x = (-1)^s \cdot 2^{e} \cdot (1 + f)$ that follows is never used to compute anything, and the range of $e$ is never stated, so the reader cannot derive either 65504 or 90112. Fix the digit and add the two-line check: "With 5 exponent bits, $e$ runs from $-14$ to 15, so the largest fp16 number is $2^{15}(2 - 2^{-10}) = 65504$. With 7 mantissa bits, bf16 numbers between $2^{16}$ and $2^{17}$ are $2^{9} = 512$ apart, and 90000 lies between $175 \cdot 512 = 89600$ and $176 \cdot 512 = 90112$; it rounds to the nearer." This turns the formula into the explanation of the cell the reader just ran.

**NUM-2** (minor, pedagogy). Under "Range and Precision": "| tf32   | 1    | 8        | 10       |" and "The formats in use are:".
Two of the five rows (tf32, fp8) are not storage formats the section uses, and the reader is told so only after the table ("The tf32 row is a mode of matrix multiplication rather than a storage format, and we return to it below"). Reduce the table and `fig_bg_float-formats` to fp32, bf16, fp16. The figure is clear and worth keeping.

**NUM-3** (moderate, explanation). Under "Range and Precision": "value, below which the format switches to *subnormal* numbers, whose".
Subnormal numbers carry the whole loss-scaling subsection, yet they are defined in a relative clause inside a list of three `finfo` fields ("whose precision falls with every halving until they reach zero"). A first-time reader does not come away knowing what a subnormal is. Give it two sentences of its own with numbers: below $2^{-14}$ the exponent cannot shrink further, so fp16 numbers are multiples of $2^{-24}$; a value near $3 \times 10^{-5}$ keeps 9 significant bits, one near $10^{-7}$ keeps 1.

**NUM-4** (minor, thought). Under "Range and Precision": "subnormal below about $6.1 \times 10^{-5}$, and has no positive number below", and under "Loss Scaling for fp16": "significant bits, and below about $3 \times 10^{-8}$ it rounds them to zero."
The smallest positive number is given as $6.0 \times 10^{-8}$ in one place and the round-to-zero threshold as $3 \times 10^{-8}$ in the other. Both are right (the second is half the first), but the reader is not told the relation. Add "(half the smallest subnormal, $2^{-24}$)".

**NUM-5** (minor, language). Under "Rounding of Small Updates": "first reason why training keeps the weights in fp32."
No second reason is ever named. Either name it where it arrives (the optimizer state and accumulations of exercise 4) or write "This is why training keeps the weights in fp32."

**NUM-6** (moderate, pedagogy). Under "TF32 Matrix Multiplication": "multiplication becomes faster and less precise, with results that typically".
See structural finding 3. The cell teaches an API getter and setter, and the MXNet tab that follows is two environment variables. Nothing later in the section depends on the subsection.

**NUM-7** (minor, language). Under "Training in fp32 and in 16 Bits": "The loss values stay on the device during training and are converted to".
This paragraph appears four times, once per tab, with identical wording in three of them, and it is a forward reference to `sec_use_gpu` in the middle of an experiment. It is implementation commentary. Make it a code comment, or one untabbed sentence.

**NUM-8** (major, pedagogy). Under "Training in fp32 and in 16 Bits": "Does 16-bit computation change what the model learns? We train the MLP for".
See structural finding 1. The stored output is `fp32 0.652, bf16 0.652` and two coincident curves. The experiment cannot fail, so it teaches only reassurance. Add the bf16-weights arm, and state in one sentence why 100 steps at 65% accuracy suffice for the comparison (the claim is about agreement, not about accuracy).

**NUM-9** (major, pedagogy). Under "Training in fp32 and in 16 Bits": "the same rate to the same place. We did not time the runs. The speed of".
See structural finding 2. Also, the sentence that follows carries the most useful fact in the subsection and deserves the emphasis: mixed precision leaves the 16 bytes per parameter untouched and saves only activation memory and matmul time. That limit should be a short paragraph of its own, since it corrects the expectation the opening set up (32 GB against 16 GB).

**NUM-10** (moderate, thought). End of "Training in fp32 and in 16 Bits", MXNet tab, and the PyTorch loop under "Loss Scaling for fp16": "scaling: with plain SGD on this well-scaled loss, the gradients stay inside".
In the next subsection the same MLP on the same minibatch has a third of its first-layer gradients below the smallest normal fp16 number. The two statements can be reconciled (subnormals lose bits but few entries reach zero), but nobody reconciles them, and the PyTorch loop that follows prints `loss 2.019, loss scale 65536`, which shows only that the scaler ran. Loss scaling is the one piece of the recipe that the reader must take on faith. Either build a case in which the unscaled fp16 run visibly degrades (a loss averaged over a large batch, or a deeper stack) and the scaled one does not, or say plainly: "For this small network the underflow costs little: 0.2% of the gradients vanish. In deep networks and with losses averaged over many elements the fraction is large enough to stop learning."

**NUM-11** (minor, language). Under "Loss Scaling for fp16", PyTorch tab: "Two rules apply when the loop does more. `scaler.unscale_(optimizer)` may be".
API caveats in the register of release notes, with "gradient accumulation", "micro-step" and "effective batch" undefined. Cut; they belong in the PyTorch documentation.

**NUM-12** (major, correctness and pedagogy). Under "Accumulation in fp32": "the terms: a left-to-right loop stalls at 4, where the spacing of fp16".
The paragraph explains outputs the default reader does not have. The stored PyTorch output is `9784` for the fp16 sum and `10004.0439` for the fp32 accumulator. Nothing stalls at 4, the fp32 total is not "about 2% low" (it is exact to eight digits), and the fp16 total is off by the 2% that the prose attributes to a left-to-right fp32 loop. The nine-sentence paragraph narrates summation orders of four libraries, which is the per-framework narration the writing guide forbids (section 4.5) and a sign that the experiment does not show what the prose wants. Replace the cell with one whose result is the same everywhere: a plain left-to-right loop over a few thousand fp16 additions of $10^{-3}$, which stalls at 4 in every framework, next to the same loop with an fp32 accumulator. Then the prose can be three sentences.

**NUM-13** (minor, explanation). Under "Accumulation in fp32": "The stored value of $10^{-3}$ in fp16 is slightly larger than $10^{-3}$, so".
The 10004.04 detail is forced on the text by the choice of increment. Use an increment that fp16 stores exactly (for example $2^{-10}$) and the distraction disappears.

**NUM-14** (minor, language). Summary: "underflow; bf16 does not. Exponentials and long sums are computed in stable".
"Are computed in stable forms and with fp32 accumulators" states a practice in the passive as if it were a fact about the world. Write "Compute exponentials in their shifted form and long sums in an fp32 accumulator."

## Exercises

Five per tab, well graded. Exercise 1 (rounded updates, on paper) and 3 (dynamic loss scaling, including the fraction of skipped steps as a function of $N$) are the kind of problem Cover and Thomas would set. Exercise 4 (RMSNorm in fp16) is good and reuses the previous section. Exercise 2 is long for its yield; part 2 ("At which batch size does the activation saving equal the persistent training state") is the valuable half and could stand alone. Exercise 5 (e4m3fn) is fine as the only home for fp8. Missing: an exercise that trains with weights stored in bf16 and finds the learning rate below which training stalls; if structural finding 1 is adopted in the body, this becomes its natural follow-up.

## Still open from October

The TF32 version history is gone. fp8 remains in the table and body (October: cut). Loss scaling, which October marked "keep", is kept, but see NUM-10.
