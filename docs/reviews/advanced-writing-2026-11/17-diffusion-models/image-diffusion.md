# Diffusion for Images

Source: `chapter_diffusion-models/image-diffusion.md` (read to the slides marker). About 3,400 words of body prose, 2,100 words of exercises, 26 code cells.

## Verdict

The most readable of the three sections and the one closest to the bar. It has a clear list of four things that change for images (network, recipe, steering, measurement), builds each one, and treats measurement with a care that is rare in textbooks: a Fréchet distance in features, read against the distance between two sets of real images of the same size, with KID beside it for the bias, class agreement beside it for guidance, and ranges over training seeds for the moving average. To the reviewer's question: the model is trained at a scale where quality can be judged (the stored samples are recognizable garments with varied shapes, and the FD of about 1 is five to six times the real-against-real value of about 0.2, which leaves room for later sections to show gains), and quality is measured, not only looked at. The mathematics of guidance is correct and all stored numbers agree with the prose in both frameworks. The most important fix is to the network description: the time embedding does not do what the text says it does (IMG-1), and the reader is not told what was left out of the published U-Net or what the evaluation costs. Second, the section never shows the process it is about: there is no picture of an image being noised or of a sample forming.

## Scores

| Dimension | Score |
|---|---|
| Clarity of explanation | 4 |
| Clarity of thought and focus | 4 |
| Pedagogy | 4 |
| Language | 4 |
| Ease and enjoyment | 4 |

Pedagogy would be a 5 for "Measuring Sample Quality" alone; the ordering problem (IMG-2) and the missing pictures (IMG-12) hold it at 4.

## What works

- **The opening** starts from the reader ("some have moved a slider that makes the picture follow the prompt more closely"), names the four changes, and gives each an origin in one sentence. Keep it.
- **Why a U-Net**, as a problem before the mechanism: "at high noise the network must decide from the whole image whether a blurred blob is a shoe or a bag", then "the edge of a sleeve or the stripes on a sneaker" for what the bottleneck would lose. The remark on group normalization ("it does not mix the statistics of images at different noise levels") is a detail that matters and is explained.
- **"Reading the Number".** The real-against-real FD at 1,000 and 5,000 images (about 1 and about 0.2) teaches the bias of the score better than any formula, and the conclusion "An FD can therefore be compared only with FDs from the same number of images" is drawn from the reader's own output. The comparison of averaged and last weights ("the average scored an FD of about 1 every time, the last weights anywhere from 1 to 3.5") is how the book should report a random experiment.
- **The guidance sweep.** Six scales, two numbers each, read candidly: 90% agreement at $\gamma = 1$ is "about as high as this classifier can confirm", and above 99% is "more than the feature network achieves on real images", so "the guided samples are the class's prototypes, easier to classify than real ones". This is a sentence a reader can quote a week later.
- **The affine argument** for why the rule is the same for velocities as for scores ("the weights $1 - \gamma$ and $\gamma$ sum to one, so it passes through the combination unchanged") is short and complete.
- **"Guidance as a Tilt"** says what almost every tutorial omits: the tilted densities are not noised versions of any distribution, so the guided sampler does not sample the tilt either.
- Appendix references resolve and agree (`sec_mdl-ddim-guidance`, the two guidance exercises); the curvature bound is stated with its intuition and proved only in the appendix, which is the right division.

## Structural findings

1. **Guidance is implemented 350 lines before it is explained.** `cfg`, $\gamma$ and the formula $\mathbf{u}_\varnothing + \gamma(\mathbf{u}_c - \mathbf{u}_\varnothing)$ appear in "Conditioning on a Class" with the note "until the guidance experiment we use $\gamma = 1$". See IMG-2.
2. **No picture of the process.** See IMG-12.
3. **Cost of the experiments.** Training is "a few minutes on one GPU", but the section's evaluations need more network passes than its training. See IMG-6.
4. **The order of the four parts is right** and the section is one section. "The Moving Average" is in the right place (after the reader has sampled from `ema` and before the FD shows why).
5. **Guidance is taught twice in the book.** The appendix subsection "Guidance with Bayes' Rule" (in `mdl-score-matching-diffusion-flow.md`) has the same equations and the same sentences ("reads ... from right to left"). By the author's rule this material is specific to the main chapter and belongs here; the appendix should keep the curvature proposition and its one-dimensional experiment and point here for the rest. That is a finding for the appendix, not for this file.

## Passage-level findings

**IMG-1** (moderate, correctness of practice; confident of the numbers, effect on FD not measured). Under "Skips, Resolutions and the Time Embedding".
> geometrically spaced frequencies, like the positional encodings of

`UNet` calls `d2l.time_embedding(t, 128)` with $t \in [0, 1]$. The frequencies run from 1 down to $10^{-4}$, chosen for positions up to 10,000. For $t \le 1$ I computed: 10 of the 128 features vary by more than 0.1, the centered features have effective rank 2, and $t = 0.50$ and $t = 0.51$ differ by 0.02 on a vector of norm 8. The embedding is in effect $(t, t^2)$ plus constants, added to a label embedding drawn from $\mathcal{N}(0, 1)$ in 128 dimensions. The paragraph says the right prediction "depends strongly on the level, so the time enters every block", and then supplies a weak time signal. DDPM embeds integer steps up to 1,000 and continuous-time code multiplies $t$ by 1,000; :exref:`ex_image-diffusion-broken-trainer` lists passing $1000\,t$ as a bug. The model trains anyway. Fix: scale $t$ inside `time_embedding` (or embed the log-SNR), re-capture, and report whether the FD moves; or say in one sentence that for $t \in [0, 1]$ these features are nearly linear in $t$ and that this is enough at $28 \times 28$. Either way the exercise needs rewording. See DEN-5 in `denoising.md`.

**IMG-2** (moderate, pedagogy: order). Under "Conditioning on a Class".
> A sampler needs a field of $(\mathbf{x}, t)$ alone. `cfg` turns the

The reader meets a function named for classifier-free guidance, a scale $\gamma$ and an extrapolation formula before any of them has a meaning, and every sample until the guidance subsection is computed with a doubled batch at $\gamma = 1$, where the second half is discarded. Fix: here, define only what is needed, a wrapper that fixes the label ("`conditional(net, c)` returns the field $(\mathbf{x}, t) \mapsto \mathbf{u}_{\boldsymbol\theta}(\mathbf{x}, t, c)$"), and keep label dropout with the forward pointer it already has. Define `cfg` in "Classifier-Free Guidance" directly under :eqref:`eq_diffusion-cfg`, where "This is what `cfg` computes" then points at code the reader has just read. The cell id (`image-diffusion-classifier-free-guidance`) suggests it used to live there.

**IMG-3** (minor, correctness: dependency). Under "Skips, Resolutions and the Time Embedding".
> the fully convolutional networks of :numref:`sec_fcn` have the same encoder

`sec_fcn` is in the Image Models part, after this chapter, so the sentence presents later material as known. Write "the fully convolutional networks of :numref:`sec_fcn` will use the same encoder and decoder without them", or drop the clause.

**IMG-4** (minor, correctness). Same subsection.
> twice, from $28 \times 28$ to $14 \times 14$ to $7 \times 7$, doubling the

The channels double once (64 at $28 \times 28$, 128 at $14 \times 14$ and at $7 \times 7$), as the figure shows. Say "from 64 to 128 channels".

**IMG-5** (moderate, explanation: technical detail). After the shape check.
> The network has about 1.2 million parameters, a thirtieth of the 35.7

The comparison with the CIFAR-10 model of Ho et al. is the place to say what was left out, in one sentence: that model has four resolutions, two blocks per resolution, self-attention at $16 \times 16$ and dropout; ours has none of these. The reader should also learn two details that matter in practice and are currently hidden: (a) initialization (the JAX tab note says that with Flax's default initialization "this U-Net learns markedly more slowly", which is a fact about the model, not about JAX, and the PyTorch reader never sees it); (b) that published networks initialize the last convolution of each block and of the output to zero. A sentence of candour also belongs here on currency: since about 2023 large systems replace the U-Net by a transformer on latent patches, which :numref:`sec_diffusion-latent` builds. The opening ("Every image generator is built from these four parts") and the Summary ("An image diffusion model is the model of ... with a U-Net in place of the multilayer perceptron") read as if the U-Net were still the standard.

**IMG-6** (moderate, pedagogy: experiment scale). Under "One Trainer".
> for 15,000 updates, a few minutes on one GPU. Diffusion papers use a rate of

The stated cost covers training only. Counting network passes: training is $15{,}000 \times 128 \approx 1.9$ million forward and backward passes; each FD of 5,000 samples with 50 Euler steps and the doubled batch of `cfg` is 0.5 million forward passes, and the section computes eight of them; the reference chain with 1,000 steps is 10 million forward passes by itself. The evaluation therefore costs more than twice the training, and the single ancestral run more than the training. On a laptop without a GPU the section does not run in reasonable time, and the text does not say so. Fix: state the total time on the reference GPU and on a CPU; drop the doubled batch at $\gamma = 1$ (IMG-2), which halves most of it; and consider 250 ancestral steps on the same 5,000 samples (the sample count must stay, as the section itself teaches).

**IMG-7** (minor, correctness: unsupported claim). Same paragraph.
> too slowly, whichever target the network predicts.

No cell shows this. Either cut the clause or make it an exercise. (Ho et al. state the rate $2 \cdot 10^{-4}$; the warm-up is in their released code, as far as I recall. Unverified.)

**IMG-8** (minor, correctness: comparison within noise). Under "Reading the Number".
> The thousand-step chain scores a tenth to a quarter lower than fifty Euler

The stored values are 0.89 against 0.71 (PyTorch) and 0.99 against 0.90 (JAX). Four paragraphs later the text says that 0.89 against 1.02 is "less than the spread between training runs". The sampler comparison is paired (same network, same count), which is why it can be trusted more than a difference between runs, but the text does not say so, and the comparison changes the sampler, the number of steps and the grid at once. Say "slightly lower in every run we made" and that the comparison is on one network; the conclusion the chapter needs ("a modest gain for twenty times the evaluations") stands.

**IMG-9** (moderate, correctness: experiment read for more than it shows). Under "Classifier-Free Guidance".
> model: the unconditional prediction covers all ten classes and was trained

The FD of about 3 at $\gamma = 0$ is attributed to an under-trained unconditional prediction. The cell cannot separate that from a second cause: at $\gamma \ge 1$ the 5,000 samples hold exactly 500 of each class by construction, while at $\gamma = 0$ the class proportions are whatever the model produces, and the mean term of the FD is sensitive to them. Fix: print the class histogram of the $\gamma = 0$ samples (the feature network's head gives it in one line) and say which cause the numbers support, or soften to "two things can raise it" and point to :exref:`ex_image-diffusion-label-dropout`, which varies the dropout rate.

**IMG-10** (moderate, pedagogy: demonstration handed to the appendix). Under "Guidance as a Tilt".
> power $\gamma > 1$ can break this bound, and on the mixture of the appendix

The claim that guidance breaks the curvature bound is supported only by an appendix experiment on a different example (guide section 7.6: the main text "never hands a demonstration to the appendix"). The section's own :exref:`ex_image-diffusion-gaussian-guidance` contains a two-line case that would do it in the text: for a class $\mathcal{N}(\boldsymbol\mu_c, s^2\mathbf{I})$ inside all images $\mathcal{N}(\mathbf{0}, S^2\mathbf{I})$ the tilted density is Gaussian with variance $v = s^2S^2/(\gamma S^2 - (\gamma - 1)s^2) < s^2$ and mean $\kappa\boldsymbol\mu_c$ with $\kappa > 1$: narrower than the class and pushed away from the other images. That also gives the reader a formula for "trades variety for typicality". Keep the appendix pointer for the general statement.

**IMG-11** (minor, correctness of reference). Under "Bayes' Rule on Scores".
> meaningless gradients. The appendix treats the rule on a one-dimensional

`sec_mdl-ddim-guidance` labels the whole appendix part "Deterministic Sampling and Guidance", most of which is DDIM. The subsection "Guidance with Bayes' Rule" has no label of its own. Add one in the appendix and cite it here and in "Guidance as a Tilt".

**IMG-12** (moderate, enjoyment and pedagogy). Opening.
> integrate it and the ancestral chain apply unchanged to a $28 \times 28$

None of the three sections shows an image being noised, or a sample forming. The only pictures of the process are a graphical model and a coefficient plot. This is the gap between the section and an Illustrated-Transformer explanation, and two short cells close it, each of which computes something the prose discusses: (a) one test image at $t = 0, 0.25, 0.5, 0.75, 1$ on the linear path and on the VP path (`show_gray(path.sample(...))`), which makes "Halfway along in time the VP path is already mostly noise" visible; (b) for a few samples, $\mathbf{x}_t$ and the implied $\hat{\mathbf{x}}_0$ at five steps of the Euler sampler, which shows the coarse-to-fine order (class and outline first, texture last) that the two-dimensional ladder showed for mode weights and that the guidance-interval exercise relies on.

**IMG-13** (minor, thought: repetition). Under "The Moving Average".
> after every update, so that the weights of $h$ updates ago count half as

The moving average is now explained three times: in `sec_dcgan` (formula with decay $\beta$, half-life 500), in `sec_diffusion-denoising` (half-life $h$ defined in words), and here (formula with decay $\delta$). This subsection is the best of the three and has the evidence; make it the owner, reduce the other two to a clause, and use one symbol for the decay in both chapters.

**IMG-14** (minor, explanation). Code of `UNet`.
> """A small U-Net on time t, a label y (num_classes = null), and cond."""

The `cond` argument is never mentioned in the prose and "num_classes = null" does not parse. If `cond` serves a later section, say so in a comment ("an extra embedding, used in :numref:`sec_diffusion-few-step`") or add it there.

## Recurring patterns

- Language is plain and exact; no dash asides, no intensifiers ("exactly" does not occur). The file is hard-wrapped at 78 columns, unlike its two neighbours, which only matters for tools that grep quotes.
- Headings: "One Trainer", "Reading the Number", "Guidance as a Tilt" are tags, not subjects. Suggested: "Training and Sampling", "A Reference Value for the Fréchet Distance", "The Distribution a Guided Sampler Follows".
- "exercise :exref:" is lower-case in the body and "Exercise :exref:" in the exercises; pick one.

## Code and experiments

- Stored outputs against prose, both frameworks: parameters 1,198,913; test accuracy 0.926; FD 0.89 and 0.99 ("about 1"); real against real 1.01 and 0.17 (JAX 1.02, 0.18); KID 0.031 and $-0.0006$ (JAX 0.045, 0.0004); last weights 3.04 and 3.19; chain 0.71 and 0.90; the six guidance rows. All agree at the stated precision.
- `image-diffusion-classifier-free-guidance` (`cfg`): move (IMG-2).
- `image-diffusion-reading-the-number-2`: paired comparison, say so (IMG-8); cost (IMG-6).
- `image-diffusion-classifier-free-guidance-1`: the $\gamma = 0$ row is read for more than it shows (IMG-9).
- The range "anywhere from 1 to 3.5" for the last weights is a deliberate range over seeds; both stored runs sit near its upper end (3.04, 3.19), which is consistent with it.
- I rendered the stored sample grid, the two guidance grids and `fig_diffusion-unet`, `fig_diffusion-guidance`: the descriptions in the prose are accurate (recognizable classes, shirts and pullovers and coats confusable, more uniform columns at $\gamma = 4$). The guidance figure honestly shows the guided prediction landing outside the class contours.
- The loss in this section is a mean over pixels; `regression_loss` of the previous section sums over coordinates and is not reused. Harmless, but the floors of the two sections are then in different units; say "per pixel".
- JAX tab note on initialization: move its content into the shared prose (IMG-5). No TensorFlow or MXNet material.

## Exercises

Twelve problems, above the band of ten. Quality is high: "What the Fréchet distance sees", "The floor of the Fréchet distance" and "Diagnosing a broken trainer" are the kind of problem that teaches practice, and "Guidance on Gaussian classes" is the analytic companion the text needs (IMG-10). To come down to ten: :exref:`ex_image-diffusion-mixture-conditional` overlaps the appendix exercises it stands beside and could move there; :exref:`ex_image-diffusion-autoguidance` is an extended project on a method the text mentions in one clause and fits the further-reading notebook. Reword bug (ii) of the broken trainer after IMG-1. The interval $0.28 < \bar\sigma_t \le 2.9$ attributed to Kynkäänniemi et al. and the attributions to CMU 10-799 are unverified; the 99.5th percentile of dynamic thresholding matches my memory of the Imagen paper.

## Still open from October

Nothing specific to this section; the October review predates it. Its general request that chapter 17 state results and the appendix prove them is met here, with the duplication of guidance noted under structural finding 5.

## For the chapter review

- This section turns the two-dimensional model into an image model and supplies the measuring instruments (`generate`, `fashion_feature_cnn`, `fd_features`, the real-against-real reference) that `samplers.md`, `design-space.md`, `latent-diffusion.md` and `few-step.md` reuse. Its FD of about 1 at 5,000 samples is the baseline for those sections, so any change to the network (IMG-1) moves every later number and needs one coordinated re-capture.
- It assumes `sec_dcgan` for the EMA class, the FD derivation and KID, and `subsec_utils_feature_cnn` for the feature network; these dependencies are stated. It reads without the appendix except for one supporting claim (IMG-10).
- What a reader would look for and not find: pictures of the process (IMG-12), a sentence on how a text prompt conditions the network (a pointer to the cross-attention subsection of `latent-diffusion.md` would do), and what the small U-Net omits (IMG-5).
- Pattern worth checking across the chapter and part: the cost of evaluation cells is not reported where the cost of training is.
