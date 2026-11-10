# Batch Size

Source: `chapter_optimization/batch-size.md`, lines 1 to 971 (slides ignored). About 4,400 words of body prose plus 700 words of exercises; 13 code cells per framework.

## Verdict

This is one of the better sections I have read in the book, and the closest of my three to the bar. It asks one question (when does a larger batch train faster?), turns it into a number the reader can measure (the gradient-noise scale), predicts a curve from that number, and then tries to break the prediction on two models. The numbers in the prose match both stored tabs. The authors are candid about what the fit does and does not test. What keeps it from a 5 is the second half: the protocol leans on a learning-rate rule that is justified only later and has to be patched for the CNN before the reader has seen it fail, the Adam test of the square-root rule cannot separate the rule from its neighbours, and the last two subsections ("Limits of the Rules", "Growing the Batch") turn from measurement to a recital of eight papers. The single most important fix: end the section with the decision it has earned, a short procedure for choosing a batch size for a new model, in place of the literature tour. The reader finishes knowing what the critical batch size is and not quite what to do with it.

## Scores

| Dimension | Score |
|---|---|
| Clarity of explanation | 4 |
| Clarity of thought and focus | 4 |
| Pedagogy | 4 |
| Language | 4 |
| Ease and enjoyment | 3 |

## What works

- **The opening** (lines 4 to 13). "Doubling the batch from 64 to 128 sequences doubles the work in every step. If the run then needs half as many steps to reach the same loss, nothing is lost ... If it needs as many steps as before, half of the work was spent for nothing." The whole section is in those three sentences, with numbers, before any symbol. Protect it.
- **Signal and noise with a number** (lines 87 to 90): "the gradient of a single example carries a hundred times more noise power than signal, and a batch of $b_{\textrm{noise}} = 100$ examples brings the two to parity." The same value 100 returns in the worked case under :eqref:`eq_steps-examples` (lines 336 to 339: $11\,S_{\min}$ steps against $1.1\,E_{\min}$ examples). This is the concrete-case-after-abstraction pattern done properly.
- **The estimator is derived in four lines and coded in four lines** (lines 105 to 127 against `noise_scale`). "On average a minibatch gradient is longer than the true gradient, and the excess squared length is its noise power" is a sentence a reader keeps.
- **A prediction that could fail**: "The noise scale makes a prediction about training speed that could fail, so we test it." The protocol paragraph explains why the target is evaluated on fixed held-out examples ("would let the small-batch runs, whose loss estimates fluctuate most, cross the target early by chance"). That is the kind of detail that matters in practice and is rarely written down.
- **Candour about the fit** (lines 578 to 582): "Part of this agreement is built in, because each model's two constants were fitted to its own four or five runs". And lines 552 to 555: "Closer agreement is more than :eqref:`eq_steps-examples` promises here". The section states the strongest claim the evidence supports and stops.
- **Ranges are used correctly throughout.** I checked every range against both stored tabs (noise scales 89/407 and 76/174 for the CNN, 2.2/60 and 2.5/78 for `TinyLM`; fitted $b_{\textrm{crit}}$ 90 and 87, 242 and 301; SGD shift 4 and 8; Adam shift 1.9 and 3.2). All fall inside the stated ranges.
- **Exercises 4, 6 and 7** build on each other (time-optimal batch $\sqrt{b_{\textrm{sat}} b_{\textrm{noise}}}$, then measure it). They are the best exercises of my three sections.

## Structural findings

1. **The experiment uses the learning-rate rule before the rule exists.** "Steps to a Target" moves $\eta$ by $\sqrt{b/b_0}$ (line 356), "which the next part of this section derives and tests". The CNN sweep then needs an exception (hold $\eta$ above 512), explained by a forward reference to `subsec_batch-size-limits` and a description of a failure that no cell shows. And "Learning-Rate Rules" in turn uses the $b_{\textrm{crit}}$ fitted in the earlier part to argue that both rules should hold. The dependency is circular and the reader carries two IOUs for 250 lines. There is no perfect order, but a better one exists: state both rules with their one-paragraph justification immediately after the noise scale (they are consequences of the same signal-and-noise split: SGD's noise ball depends on $\eta/b$, Adam's denominator is $\sigma/\sqrt{b}$), then run steps-to-target with the rule in hand, then run the two-batch-size sweep as the test, then the limits. The saturation $\eta_{\textrm{opt}}(b) = \eta_{\max}/(1+b_{\textrm{noise}}/b)$ belongs right next to :eqref:`eq_steps-examples`, because it is the same model and it is what explains the 512 cap.
2. **The section ends on a survey.** "Limits of the Rules" (lines 794 to 821) and "Growing the Batch" (lines 826 to 857) cite eight papers in about 450 words and run no cell. Three of the limits are sound and short. The fourth paragraph ("None of this is a universal law ...") and most of "Growing the Batch" narrate what papers found (GPT-3, Zhang et al., Merrill et al., Smith et al., Shah et al.). The one claim here that the section has itself measured, that the noise scale grows during training, is left to exercise 1. Either move the body of exercise 1 into the text as the closing experiment (noise scale against loss over 3,000 steps, and where a doubling rule would fire), or cut "Growing the Batch" to one paragraph that states the consequence and names one example.
3. **No decision procedure.** The part introduction promises that the chapter teaches "how to choose a learning rate, schedule and batch size". This section holds every ingredient and never assembles them. A reader with a new model wants something like: (a) the batch must at least saturate the device (:numref:`sec_sgd`) and fit in memory; (b) measure `noise_scale` at initialization and after a few hundred steps; (c) if compute is the scarce resource, stay at or below it; if time is scarce and you have devices, go to about the critical batch size and no further; (d) move $\eta$ by the rule and re-sweep a factor of two around it; (e) expect the answer to grow during training. Five sentences at the end of the body, in place of the survey, would make this the section people return to.
4. **Length.** 4,400 words is within budget and most of it is used. What could go with no loss: the description of `TinyLM` and `FashionCNN` (lines 136 to 141; the reader has met both three times), the Malladi time-constant detail (lines 643 to 648), the Keskar sharp-minima history, the Shah et al. Muon sentence, and half of "Growing the Batch". About 350 words.

## Passage-level findings

**BATCH-1** (minor, thought). Under the opening. Anchor: "an accelerator processes a larger batch in barely more time per step until it is fully occupied".
Presented as one of "Two facts from earlier sections". :numref:`sec_sgd` measured it on a single 4 MB matrix product, and its own text says the time "grows with $b$" once the arithmetic units are busy. The reader's device for `TinyLM` at $b = 64$ may already be past that point. Fix: say "up to the batch that saturates it (:numref:`sec_sgd`)" and note that the question of this section starts there: beyond saturation a larger batch costs proportionally more, so it must save steps to be worth it.

**BATCH-2** (minor, thought). Under "Signal and Noise". Anchor: "This is the $1/b$ law that :numref:`sec_sgd` checked on a small network."
The law is derived in :numref:`sec_sgd` with the same display and the same symbols. Here it is displayed again with its reason ("because the variance of an average of $b$ independent terms is $1/b$ times the variance of one"). By guide 7.7 a restatement is one clause with a reference. The definition of $\operatorname{tr}\boldsymbol{\Sigma}$ must stay, since the noise scale needs it; the second display and the reason can go: "A minibatch gradient $\hat{\mathbf{g}}_b$ of $b$ independent examples has noise power $\operatorname{tr}\boldsymbol{\Sigma}/b$ (:numref:`subsec_sgd-variance`)."

**BATCH-3** (moderate, explanation). Under "Signal and Noise". Anchor: "gives a markedly better direction, so each step can".
The step from "better direction" to "fewer steps" is the crux of the section and is asserted. Why does a less noisy gradient permit fewer steps? Because the optimizer can use a larger step without the noise dominating the progress; at a fixed step, halving the noise changes little. The learning rate is the missing actor, and it only appears 500 lines later as $\eta_{\textrm{opt}}(b)$. Fix: one sentence here. "A less noisy gradient tolerates a larger learning rate, and it is the larger step that saves steps; a larger batch at an unchanged learning rate mostly reduces the jitter." This also prepares the rules.

**BATCH-4** (minor, explanation). Under "A Two-Batch Estimator". Anchor: "and the large batch, nearly free of noise, pins down the signal".
For `TinyLM` after 500 steps the noise scale is 50 to 200 and the large probe batch is 256, so the large batch is not nearly free of noise; its squared norm is up to about 80% noise. The estimator still works because it solves two equations, not because the large batch is clean. Fix: "and subtracting that noise from the large-batch norm leaves the signal."

**BATCH-5** (moderate, correctness of an unsupported claim). Under "A Two-Batch Estimator". Anchor: "Measured again on the same trained".
"it changes by less than a third" is a number no cell prints, and the next sentence builds a conclusion on it ("so the spread between runs comes from the runs themselves"). The cells call `noise_scale` once per model state. Fix: call it twice on the trained CNN in cell `batch-size-a-two-batch-estimator-3` and print both values (deferred with the other experiment changes), or soften to what the reader can check and move the repeatability claim to an exercise.

**BATCH-6** (minor, explanation). Under "A Two-Batch Estimator". Anchor: "against several thousand bytes".
The section's unit for `TinyLM` is sequences (noise scale 50 to 200). The conversion to bytes (times 64) is left to the reader at the moment a comparison across four orders of magnitude is made. Write "against 50 to 200 sequences, several thousand bytes, here".

**BATCH-7** (moderate, explanation). Under "Steps to a Target". Anchor: "A model of noisy optimization on a quadratic".
:eqref:`eq_steps-examples` is the central formula of the section and arrives by citation alone. The reader cannot see why $S$ should be $S_{\min}(1 + b_{\textrm{noise}}/b)$ and not some other interpolation. A three-line argument is available and uses only what the section has: with the best step for a gradient of signal $\|\nabla f\|^2$ and noise $\operatorname{tr}\boldsymbol{\Sigma}/b$, the expected decrease per step is the noise-free decrease divided by $1 + b_{\textrm{noise}}/b$ (minimize $-\eta\|\nabla f\|^2 + \tfrac{L}{2}\eta^2(\|\nabla f\|^2 + \operatorname{tr}\boldsymbol{\Sigma}/b)$ over $\eta$). That gives both the step count and $\eta_{\textrm{opt}}(b)$ at once, removes the need to cite $\eta_{\textrm{opt}}$ separately at line 628, and is a small result, so it belongs in the main text.

**BATCH-8** (minor, explanation). Under "Steps to a Target". Anchor: "In the model, the noise scale weights each direction of parameter space by".
The paragraph break is missing before this sentence (it runs on from "most of the extra work is wasted."), and the sentence itself is too compressed to follow: which noise scale, weighted how? Fix: new paragraph. "The model's noise scale is not quite the ratio we measured. It weights each direction by its curvature, $\operatorname{tr}(\mathbf{H}\boldsymbol{\Sigma}) / \nabla f^\top \mathbf{H} \nabla f$. Our ratio :eqref:`eq_noise-scale` equals it when the curvature is the same in every direction and is a cheap substitute otherwise."

**BATCH-9** (moderate, pedagogy). Under "The CNN". Anchor: "Adam's first steps overshoot, the training loss jumps above its".
Six lines describe a failure at $b = 2048$ that the reader never sees: no cell runs the rule-scaled rate at 2,048. The description comes before the cell, as a justification of the `min(b, 512)` in the code, and it refers forward to a subsection the reader has not read. Fix under the reordering of structural finding 1: once $\eta_{\textrm{opt}}$ saturates at the noise scale, "we hold the rate fixed above 512 images, about twice the critical batch size we will find" needs one sentence. The overshoot description moves to "Limits of the Rules" or becomes the result of the exercise that already exists for `TinyLM` ("Past the elbow").

**BATCH-10** (minor, correctness). Under "The CNN" and "The Language Model". Anchor: "saves
at most about half of the steps".
The evaluation runs every ten steps, so the step counts at the top of the range (70 to 120 steps in the stored outputs) are resolved to about 10%, and the fitted $S_{\min}$ (66 to 81 steps for the CNN) inherits this. The prose claims survive, but the reader should be told once: "counts are resolved to ten steps". The exercise "The factor-of-two check" asks for a comparison that this resolution limits.

**BATCH-11** (minor, thought). Under "The Critical Batch Size". Anchor: "We write $b_{\textrm{crit}}$ for this fitted critical batch size,".
Good decision to separate the two symbols. But the term *critical batch size* was introduced at line 340 as the batch size at $b = b_{\textrm{noise}}$ in the model, and the summary says "a fitted critical batch size of the same order as their measured noise scales". State the relation once, plainly: in the model the two coincide; in measurement $b_{\textrm{noise}}$ is the gradient-statistics estimate and $b_{\textrm{crit}}$ the training-curve estimate of the same quantity.

**BATCH-12** (minor, language). Under "The Critical Batch Size". Anchor: "This is why large training runs pick batch sizes near".
"and why the exercises trace this tradeoff" is not a consequence of anything. End the paragraph on "accelerators or days", then: "Large runs choose batch sizes near their critical batch size for this reason."

**BATCH-13** (moderate, explanation). Under "Linear and Square-Root Scaling". Anchor: "and already makes the signal in each step $\sqrt{k}$ times larger".
The Adam argument is the hardest step of the section and goes by in two sentences. "Signal in each step" has not been defined for Adam, and the reader has to reconstruct that the update is $\eta\, m/\sqrt{v}$ with $m \approx \nabla f$ on average and $\sqrt{v} \approx \sigma/\sqrt{b}$. Write it: "the expected update is about $\eta\, \nabla f \sqrt{b}/\sigma$ per coordinate. A batch $k$ times larger makes each update $\sqrt{k}$ times longer at the same $\eta$. To cover $k$ steps' distance in one, $\eta$ must supply the other $\sqrt{k}$." Then the Malladi result can shrink to one sentence; the conditions on $\beta_i$ and $\epsilon$ ("for $k < 10$ at $\beta_1 = 0.9$") serve nobody here, since the section immediately says practice ignores them. Keep them only if an exercise uses them.

**BATCH-14** (minor, correctness). Under "Linear and Square-Root Scaling". Anchor: "follows from
how SGD behaves near a minimum".
The noise-ball argument is a fair derivation, but it is not the argument of the cited paper (Goyal et al. argue that $k$ small steps on nearly equal gradients equal one step $k$ times larger); the $\eta/b$ invariance is due to Smith and Le and to Smith et al., which the section cites later. Also the sweep that follows uses momentum 0.9, where the invariant is $\eta/((1-\beta)b)$; say so in a clause. Sure of both.

**BATCH-15** (moderate, pedagogy; experiment read for about what it shows, but it shows little). Under "Learning-Rate Sweeps at Two Batch Sizes". Anchor: "fitted optimum moves by a factor between about 1.5 and 5 across our runs".
The prose is honest, but look at what remains: a prediction of 2.8, a measured range of 1.5 to 5, and stored values of 1.9 (PyTorch) and 3.2 (JAX). The experiment excludes the linear rule (8) and barely excludes no scaling at all. The rendered Adam plots show why: at $b = 64$ the final loss is flat within a few hundredths from $10^{-3}$ to $2.6 \times 10^{-2}$, so a parabola through three points finds a vertex wherever the noise puts it. The "about 2.5 on average" is a number no cell prints. A teaching experiment that needs a parabola fit, a paragraph of protocol (lines 659 to 668) and a paragraph of caveats is, by guide section 9, a research experiment. Directions, for the experiment pass: a larger batch ratio (8 against 512 gives predicted shifts of 8 and 64, which no flat basin can blur), or measure the largest stable learning rate instead of the optimum (a sharp edge instead of a flat basin). For the writing pass now: cut "and by about 2.5 on average", and state the claim the data supports: "the shift is clearly smaller than the batch ratio, and consistent with the square-root rule within the resolution of a flat basin".

**BATCH-16** (minor, language). Under "Learning-Rate Sweeps at Two Batch Sizes". Anchor: "runs the grid, quietly, and `d2l.final_loss` summarizes".
"quietly" (for `verbose=False`) is on the guide's list. Delete. The same paragraph spends 110 words on protocol; the reader needs two facts (same example budget, same seed and data order). The definition of the final loss as the mean over the last quarter can be a code comment.

**BATCH-17** (minor, correctness). Under "Limits of the Rules". Anchor: "which
:citet:`Goyal.Dollar.Girshick.ea.2017` introduced for this reason".
Goyal et al. introduced *gradual* warmup for large-batch training; constant warmup at a low rate was used earlier (He et al., 2016, for the 110-layer ResNet). "used for this reason" or "introduced gradual warmup" is safe. Fairly sure. Check that :numref:`sec_scheduler` says the same.

**BATCH-18** (moderate, thought). Under "Limits of the Rules". Anchor: "None of this is a universal law.".
This paragraph is the survey inside the teaching section: Shallue on workloads, Keskar on sharp minima, Shallue again on retuning, in six sentences, none of them connected to an experiment here. The generalization question is a real one a reader will ask ("will a large batch hurt my test accuracy?"), and it deserves a direct two-sentence answer in the authors' voice, not a history: "A large batch does not by itself hurt generalization. The early reports of a gap compared runs whose learning rate and length had not been retuned for the larger batch :cite:`Keskar...,Shallue...`." The rest goes to further reading.

**BATCH-19** (moderate, focus). Under "Growing the Batch". Anchor: "Measurements during pretraining find that the".
From here to the end of the subsection the text retells Zhang et al., Merrill et al., Smith et al. and Shah et al. at one sentence each. The Shah sentence ("stays data-efficient up to larger batches than AdamW, which moves the critical batch size ... to the right") is a claim about Muon with no support in this book's experiments and belongs in :numref:`sec_muon` or further reading. Keep: the compromise argument (lines 830 to 833, which is the authors' own and good), GPT-3 as the one example, and the $\eta/b$ link to decay. See structural finding 2.

**BATCH-20** (minor, thought). Under "Growing the Batch". Anchor: "One question remains open.".
It is not open; it is postponed to :numref:`chap_performance`. And the section's opening already told the reader that the gain needs "twice as many accelerators". End on the last result, or on the pointer without the drum roll: "Fewer, larger steps save time only when each step is split across devices, the subject of :numref:`chap_performance`."

**BATCH-21** (minor, thought). Under "Summary". Anchor: "Above it, the step count levels off at a floor and the extra
examples are wasted.".
The body said, correctly, that above the critical batch size extra examples buy time at a poor rate and that this can be the right choice ("Which point is best depends on what is scarce, accelerators or days"). The summary flattens this to "wasted". Use the body's version.

## Code and experiments

- All prose ranges agree with the stored PyTorch and JAX outputs (listed under "What works").
- **Numbers no cell prints:** the repeatability "less than a third" (BATCH-5); the overshoot at $b = 2048$ under the unclipped square-root rule (BATCH-9); "about 2.5 on average" (BATCH-15).
- **`batch-size-the-bill-in-examples`** (the collapse onto the hyperbola): the prose is right that two fitted constants per model leave two or three degrees of freedom. This cell is a check that could fail only mildly, but it is the one picture that shows both models on one curve, and the prose does not overread it. Keep.
- **`batch-size-learning-rate-rules-3`** (Adam sweep): see BATCH-15. The effect is too small for the grid and the basin. `best_lr` is 5 lines of plumbing that exists only to rescue it.
- **`batch-size-learning-rate-rules-1`, JAX tab:** the `FixedOrder` data module (8 lines) exists so that both batch sizes see the same order. This is framework plumbing in the middle of the argument; it belongs in `d2l` or in a comment. The PyTorch tab has no equivalent and its prose claim ("see the same images in the same order") rests on `d2l.lr_sweep` taking a seed.
- **The CNN sweep's learning rate is not retuned per batch size.** The McCandlish model assumes the best rate at each batch. The section says so and gives the retuning to an exercise ("Retuning instead of scaling"), which is acceptable; but then the fitted $b_{\textrm{crit}}$ of the CNN partly measures the 512 cap. One clause in "The Critical Batch Size" should say this.
- Tabs fragment the prose nowhere. The shared prose is true of both tabs in every case I checked.

## Exercises

Seven problems, 700 words, graded, each with a question. Keep all. Notes:

- Exercise 1 (noise scale over training) is the experiment the section's last claim needs; consider promoting part 1 into the body (structural finding 2).
- Exercise 3 (retuning) is good but costs 15 training runs to target; say so.
- Exercise 4 has a clean closed form ($b^\star = \sqrt{b_{\textrm{sat}} b_{\textrm{noise}}}$, $E = E_{\min}(1 + \sqrt{b_{\textrm{sat}}/b_{\textrm{noise}}})$). I checked it.
- Exercise 5 (noise scale per parameter group) is the most original. Its part 2 is slightly leading ("Why does the same count for `pos_emb` always come out zero?") but fine.
- Missing: a pencil exercise on the estimator itself, for example its variance (how many small batches $m$ are needed for a 10% estimate when $b_{\textrm{s}} \ll b_{\textrm{noise}}$), which would also back BATCH-5.

## Still open from October

The October review counted the $1/b$ law four times in the chapter. It now has one owner (`sgd.md`, `subsec_sgd-variance`, derived and measured), one sentence in `optimization-intro.md` (line 412, a fair preview), and a second display with its reason here (BATCH-2). So: down to one redundant telling. The October "Cut: Large-Scale Use / The Current Frontier" items appear to have been cut or folded into "Growing the Batch", which is where the survey habit survives.

## For the chapter review

- Placement is right: after Adam and schedules (it needs both), before Scaling Up. It assumes the noise ball and the $1/b$ law from `sgd.md`, `TinyLM`, `FashionCNN`, `d2l.train_steps` and `d2l.lr_sweep` from `adam.md`, and warmup from `lr-scheduler.md`. Nothing from the appendix is needed; there are no `sec_mdl-*` references in the file.
- `TinyLM` is described again in full here and in `practice.md` ("two blocks, about 0.47 million parameters"); the chapter describes it four times. One clause with `subsec_tinylm` suffices after its home in `adam.md`.
- The $\eta/b$ invariance is used in `sgd.md`, here twice, and in `lr-scheduler.md`; it should be stated as a named fact once in `sgd.md` and cited.
- Pattern for the chapter and probably the part: sections do their own work for two thirds of their length and then close with a "what the field does" passage of five to eight citations at one sentence each. The first two thirds are at the bar; the closings are not.
- The forward pointer to `chap_performance` is not load-bearing. No reference to the Language Models part.
