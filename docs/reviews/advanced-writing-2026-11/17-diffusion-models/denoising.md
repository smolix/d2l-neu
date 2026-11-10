# Denoising and the Score

Source: `chapter_diffusion-models/denoising.md` (read to the slides marker). About 3,750 words of body prose, 1,850 words of exercises, 34 code cells (17 per framework).

## Verdict

A good section, close to the bar in its second half. It takes one small distribution on which everything is exact, predicts numbers before cells print them (0.18 to 0.13 for the denoiser), and reaches three results a reader will remember: a denoiser is a score, a score near the data cannot see the mode weights (Fisher divergence $10^{-5}$ against a KL of 0.07 nats), and a single noise level either keeps the wrong weights or samples a blur. I verified every formula and every number quoted in the prose against the stored PyTorch and JAX outputs and found no mathematical error. The most important thing to fix is focus in the first half. The section carries the chapter's shared code (about 140 lines of helpers per framework before the first idea is tested) and a 550-word detour on energy-based models and importance sampling that arrives after the score has already been trained, duplicates the appendix paragraph by paragraph, and ends in a cell that confirms a formula derived one sentence earlier.

## Scores

| Dimension | Score |
|---|---|
| Clarity of explanation | 4 |
| Clarity of thought and focus | 3 |
| Pedagogy | 4 |
| Language | 4 |
| Ease and enjoyment | 4 |

## What works

- **The opening.** It starts from what the reader holds (the conditional-mean fact that closed the GAN chapter), turns the defect into the starting point ("For the generators of :numref:`chap_gans` that average was a defect. Here it is the starting point"), and gives one concrete object, the camera denoiser fed static. Keep all three paragraphs.
- **Tweedie in one sentence.** "have targets related by the map $\mathbf{x}_0 \mapsto (\mathbf{x}_0 - \tilde{\mathbf{x}})/\sigma^2$, which is affine once $\tilde{\mathbf{x}}$ is given, so their minimizers, two conditional means, are related by the same map." This is the shortest correct derivation I know and it needs no appendix. Protect it.
- **Prediction before measurement.** The Gaussian case gives the shrinkage factor 0.74 and the error 0.13, and the cell then prints 0.1317. The bias formula gives 0.0004, 0.28, 2.6 and the cell prints 0.0004, 0.2826, 2.5955.
- **"Blindness to Mode Weights".** "An error of $0.2$ in the Fisher divergence sounds small, but small compared with what?" is the voice the guide asks for, and the answer (two mixtures 0.07 nats apart whose scores differ by $10^{-5}$) is the best experiment of the three sections. It teaches something no theorem in the text states.
- **The Langevin triple** (exact score of $p$, exact score of $p_\sigma$, trained network) sets up the next section's question exactly, and the 12-nat barrier with a crossing time "of the order of a million steps" is checked by an exercise. I recomputed both (12.0 nats from mode 1; about $2 \cdot 10^6$ steps from Kramers' formula).
- **The Wiener baseline for images.** A learned denoiser is compared with the best linear one, not just shown, and the closing pair "A denoiser returns an average, and a generator must return a draw." is the right last sentence for the body.
- Appendix use is correct in kind: results are stated with intuition, proofs are left to `chap_mdl-dynamics`, and every label resolves to a passage that holds the cited content (checked: `sec_mdl-denoising-score-matching`, `sec_mdl-score-matching`, `sec_mdl-langevin-sampling`, `sec_mdl-fisher-divergence`, `sec_mdl-bayes-importance`, `fig_mdl-dyn-tweedie`, `ex_mdl-divergences-distances-score-blindness`). Notation agrees with the appendix ($\tilde{\mathbf{x}}$, $\sigma$, $J_{\mathrm{DSM}}$, the Langevin step $h$).

## Structural findings

1. **The section pays the chapter's setup cost.** Before the reader tests one idea they read `GaussianMixture` (with a `velocity` method for a path defined in the next section), `nll`, `summary`, `scatter_panels`, `time_embedding`, `TimeMLP` (with `num_times`, used in the few-step section), and `train_regression` (cosine schedule, EMA, compilation, none needed for 2,000 steps at one noise level). That is eight of the first nine cells. The ideas of "The Optimal Denoiser" are interleaved with library code, and the first experiment arrives at line 459 of 870. See DEN-1.
2. **"The Normalizer Never Appears" is in the wrong place and too long.** It answers "why learn a score at all", which is motivation, yet it comes third, after the network is trained and Tweedie is checked. It retells the appendix subsection "A Normalizer-Free Objective" in the same order (energy-based model, $Z$ changes with every update, the score does not see $Z$, explicit score matching, Hyvärinen, the cost of the trace). See DEN-6.
3. **The reason that matters most for images is only in an exercise.** The text gives two limits of a small noise level (noisy targets, blindness to weights). The third, that data near a low-dimensional set has a score that blows up like $1/\sigma^2$ across the set, is :exref:`ex_denoising-data-on-a-line`. The image subsection then says "natural images lie near a low-dimensional set" without connecting it to the limits. See DEN-9.
4. **Order within the section is otherwise right**: exact example, denoiser and score, limits, images. It is one section, not two. With DEN-1 and DEN-6 applied it loses about 600 words and several cells and gains pace.

## Passage-level findings

**DEN-1** (moderate, pedagogy). Under "Noise and the Regression Target".
> `train_regression` is the one training loop of the chapter

The paragraph documents a library function (Adam, cosine decay, EMA with half-life, compiled training step, CUDA kernels) in the middle of the argument that leads to Tweedie's formula. The reader of this section needs "Adam for 2,000 steps on fresh noise". Fix: introduce the trainer in one sentence here and move the description of its options to where each is first used (cosine and EMA in `sec_diffusion-gaussian-paths`, which sets them). Move `GaussianMixture.velocity` to the next section with `d2l.add_to_class`, as `nll` and `summary` already are.

**DEN-2** (minor, language). Same paragraph.
> the networks of this chapter are small enough that launching their kernels one at a time would cost more than running them

Framework plumbing in the explanation of an algorithm (guide section 9). Make it a code comment. The unexplained cell `torch.set_float32_matmul_precision('high')` after the imports in all three sections is the same kind.

**DEN-3** (minor, explanation). Under "The Running Example".
> with its density, score and posterior mean in closed form at every noise level $\mathbf{x}_t = \alpha \mathbf{x}_0 + \sigma \boldsymbol\epsilon$

$\mathbf{x}_t$, $\alpha$ and "the velocity of the linear path" appear here, a section before they mean anything, and the section then writes $\tilde{\mathbf{x}}$ throughout. Say only that the class adds noise of level $\sigma$; let the next section extend it.

**DEN-4** (minor, correctness of reference). Under "Noise and the Regression Target".
> squared error is minimized by the conditional mean (:numref:`sec_mdl-denoising-score-matching`)

The label is right (the regression lemma is there), but the reader was given this fact in `sec_gan_beyond` and the opening paragraph has just recalled it. Cite the main-text owner; the appendix pointer makes a known fact look like new appendix material.

**DEN-5** (moderate, correctness of practice; confident of the numbers, effect on results not measured). Under "Noise and the Regression Target".
> an MLP on the point plus sinusoidal features of a noise level or time from `time_embedding`

`time_embedding` uses frequencies from 1 down to $10^{-4}$, which suit arguments up to 10,000. The chapter always passes $t \in [0, 1]$ (or one fixed $\sigma$). I computed the features on a grid of $t$ in $[0, 1]$ for `dim = 128`: only 10 of 128 features vary by more than 0.1, the centered feature matrix has two singular values above 1% of the largest (the features are in effect $t$ and $t^2$), and $t = 0.50$ and $0.51$ differ by 0.02 on a vector of norm 8. With $1000\,t$ the same test gives 106 features and rank 60. The networks still train, because the targets vary smoothly in $t$ and an MLP follows, but the code does not do what its description says, published code scales the time (DDPM uses integer steps to 1,000; flow-matching code multiplies by 1,000), and :exref:`ex_image-diffusion-broken-trainer` treats "$1000\,t$" as a bug. A reader who copies this function to a harder problem inherits a weak time signal. Fix: scale inside `time_embedding` (or embed the log-SNR) and re-capture, or keep the code and say in one sentence what the features are for $t \le 1$ and why that is enough here. This affects every network in the chapter.

**DEN-6** (moderate, thought and focus). Under "The Normalizer Never Appears".
> Why not fit a density and differentiate it?

and
> Estimating $Z$ directly fails in high dimensions.

(a) Placement: this is the motivation for scores and belongs before the first regression, or in the opening, which already says it in one sentence. (b) Size: 550 words and a cell on a method (estimating $Z$ by importance sampling from a fixed proposal) that the section neither uses nor needs to refute; the reader has not met importance sampling in the main text and is sent to the appendix for it. (c) The cell confirms the formula $\sqrt{(e^d - 1)/n}$ derived in the sentence before it (author decision 4). (d) The passage duplicates the appendix. Fix: cut to one paragraph of about 150 words (a density needs $Z$, the score does not, Hyvärinen removes the data score at the cost of a trace, denoising removes the trace, with the two appendix pointers), place it after "Tweedie's Formula" or fold it into the opening, and drop the importance-sampling paragraph and cell. :exref:`ex_denoising-log-normal-weight` already holds the mathematics and can keep the experiment. Rename the heading (it is a sentence; guide section 11): "Scores and Normalizers".

**DEN-7** (minor, correctness). Same subsection.
> an expectation under the model itself, which only a Markov chain can supply

Too strong, and the next paragraph estimates the same quantity without a chain. Write "which has to be estimated by sampling from the model, in practice with a Markov chain".

**DEN-8** (minor, thought). Under "The Loss Floor and the Noise Level".
> The denoising loss does not reach zero, because even the best field leaves the conditional variance of the target

The subsection is titled for the loss floor and opens on it, but the experiment measures something else (the error against the clean score) and never prints a floor. The floor is taught properly in the next section. Reduce this paragraph to the one consequence used here (the target is mostly noise at small $\sigma$) and title the subsection "Choosing the Noise Level" or similar.

**DEN-9** (moderate, pedagogy). Same subsection.
> At a small noise level the target $-\boldsymbol\epsilon/\sigma$ is large and mostly noise, and a network trained on a finite sample fits it poorly.

This is where the reader should also learn that for data near a curve or surface the score across that set grows like $1/\sigma^2$, so a small level is hard to represent as well as hard to estimate. Two sentences and a pointer to :exref:`ex_denoising-data-on-a-line` would do it, and "Denoising Images" could then refer back when it says images lie near a low-dimensional set.

**DEN-10** (moderate, correctness: prose against stored output). Under "Sampling with a Score".
> Its weights depend on the seed, staying near the start's in some runs and drifting in others

The stored runs do not show "staying near the start's". From starts of about a third each, the PyTorch output ends at weights 0.486, 0.244, 0.270 and the JAX output at 0.394, 0.295, 0.312: in both, mass moves into the heaviest mode, by 0.18 and 0.07. A reader who sees 0.486 next to the true 0.5, straight after being told the score is blind to the weights, will conclude the opposite of what the paragraph says. With the exact score no chain crosses a barrier of 9 to 12 nats in 2,000 steps, so the crossings come from the learned field's errors in the valleys, the cause the paragraph gives only for the strays. Fix: say that chains cross because the learned field is wrong between the modes, that where they end is an accident of extrapolation (state the range over seeds if the authors have it), and that the agreement of one weight with the truth in a given run is not evidence that the weights were learned. If repeated runs show a consistent drift toward the heavy mode, say so and explain it; I could not rerun (no PyTorch on this machine).

**DEN-11** (minor, explanation). Same subsection.
> But its samples have twice the spread

The printed "spread" is the mean squared distance of the points inside the three-standard-deviation disc, so it saturates: the variance of $p_\sigma$ at $\sigma = 1$ is five times that of $p$ (1.25 against 0.25 per coordinate), and the statistic reports 0.99 against 0.49. Say "five times the variance (the printed spread, which counts only points within the disc, doubles)".

**DEN-12** (minor, language: notation). Under "Noise and the Regression Target" and "Sampling with a Score".
> whose half-life $h$ is the number of updates after which a weight's contribution to the average halves

$h$ is the EMA half-life here and the Langevin step size 300 lines later ("with a small step size $h$"). $s$ is the component standard deviation, $\mathbf{s}$ the score, and `s` the code name for $\sigma$. Rename the half-life (the GAN chapter uses a decay $\beta$, `image-diffusion.md` a decay $\delta$) and settle one symbol across chapters 16 and 17.

**DEN-13** (minor, pedagogy). Under "Tweedie's Formula".
> With the exact score the two sides agree to single-precision rounding.

The first printed line compares two closed forms of the same class, so it tests the code, not the formula, and cannot fail. Keep the three squared errors, which do teach, and drop the gap line or move it to a comment.

## Recurring patterns

- **Section openers made of three questions** (2 here, 2 more in `gaussian-paths.md`): "What does a denoiser learn, how is that a score, and why is the score the object to learn?"; "How good is that score, what can it not see, and what does a sampler driven by it produce?" Each is a roadmap in question form. One clause of connection is enough; the subsection headings already say this.
- **Cells introduced by an inventory sentence** (about 11 of 17 cells): six paragraphs open with a backticked name as subject ("`GaussianMixture` holds", "`scatter_panels` draws", "`TimeMLP` is", "`fit_score` trains", "`langevin` runs", "`load_fashion_mnist_pm1` returns") and five with "The cell checks / compares / trains / computes / estimates". This is the question-then-cell formula in another form (author decision 6). The cells that are introduced by the argument ("An error of $0.2$ ... small compared with what?") read far better.
- "exact" or "exactly" 12 times in the body; most are needed here because the contrast with "learned" is the point. Leave them.

## Code and experiments

- `denoising-the-normalizer-never-appears`: confirms a derived formula; cut (DEN-6). If kept: "Up to $d = 8$ the estimate is close to one" is loose for the stored JAX value 0.860.
- `denoising-tweedie-s-formula`: first printed line cannot fail (DEN-13). The rest matches the prose (0.1782, 0.1317, 0.1329; JAX 0.1806, 0.1331, 0.1340).
- `denoising-the-loss-floor-and-the-noise-level`: seven networks; prose matches (U shape, within 0.1 of the bias from $\sigma = 0.5$). "The best single level lies between $\sigma = 0.1$ and $0.2$, at an error of about $0.15$ to $0.2$" holds for both stored runs (0.205 at 0.1; 0.152 at 0.2).
- `denoising-blindness-to-mode-weights`: matches ($1.5 \cdot 10^{-5}$, $1.4 \cdot 10^{-3}$, $9.5 \cdot 10^{-3}$; KL 0.069). Keep unchanged.
- `denoising-sampling-with-a-score-1`: third chain, see DEN-10; second chain, see DEN-11.
- `denoising-denoising-images-2`: matches (0.2493, 0.0447, 0.0310). I rendered the stored image: the "Lee" lettering survives in the network row and smears in the Wiener row, as the prose says.
- No TensorFlow or MXNet material. The two tabs carry no tab prose, so they do not fragment the text.

## Exercises

Eleven problems, the strongest set of the three sections: almost every one extends the argument (data on a line, two-point data, Laplace noise, step size, escaping a mode, the wrong noise level). They are hard for a main-text section, and at 1,850 words they are half the length of the body. Above the band of ten: move :exref:`ex_denoising-persistent-chains` (training an energy-based model, "the route that this section avoids") to the further-reading notebook. :exref:`ex_denoising-log-normal-weight` should absorb the importance-sampling cell if DEN-6 is applied. The label `ex_denoising-regression-identity` sits on the problem titled "Two-point data"; rename one. Attributions to Berkeley CS180 could not be verified here.

## Still open from October

The October review read a different version of this chapter (its files `langevin.md`, `limits.md` and `flow-matching.md` no longer exist). Its findings on time direction, lecture-course citations and late arrival of DDPM are resolved. The notation clash it reported is reduced, not gone (DEN-12).

## For the chapter review

- This section owns: the running mixture, the denoiser as posterior mean, Tweedie, the three limits of one noise level, Langevin sampling, and the chapter's shared helpers. It sets up `gaussian-paths.md` precisely (the last Summary sentence is the handoff).
- Overlap with the appendix section `mdl-score-matching-diffusion-flow.md` is large in topic (both have a subsection titled "Limits of a Single Noise Level") but of the permitted kind, except for "The Normalizer Never Appears" (DEN-6). The section reads without the appendix once that passage is cut.
- Pattern to check in the rest of the chapter: shared `#@save` code placed in the first section that could hold it, not the first that needs it; cells introduced by naming the function; three-question openers.
- The time-embedding finding (DEN-5) applies to every network of the chapter.
