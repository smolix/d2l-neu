# Chapter 17, Diffusion Models: writing review

Sources: the ten `.md` files of `chapter_diffusion-models/`. The eight teaching sections were read in full by three section reviewers (`denoising`, `gaussian-paths`, `image-diffusion`; `samplers`, `design-space`; `latent-diffusion`, `few-step`, `discrete-diffusion`). I read `index.md` and `further-reading.md` in full and reviewed them, read all eight section reviews, and checked the cross-section claims below against the sources, the stored outputs and the appendix review (`docs/reviews/mdl-writing-2026-11/43-dynamics.md`). By the reviewers' counts the eight sections hold about 29,000 words of body prose and 14,000 words of exercises (87 problems), and the further reading 1,700 words.

Per-file reviews:
[index](17-diffusion-models/index.md) ·
[denoising](17-diffusion-models/denoising.md) ·
[gaussian-paths](17-diffusion-models/gaussian-paths.md) ·
[image-diffusion](17-diffusion-models/image-diffusion.md) ·
[samplers](17-diffusion-models/samplers.md) ·
[design-space](17-diffusion-models/design-space.md) ·
[latent-diffusion](17-diffusion-models/latent-diffusion.md) ·
[few-step](17-diffusion-models/few-step.md) ·
[discrete-diffusion](17-diffusion-models/discrete-diffusion.md) ·
[further-reading](17-diffusion-models/further-reading.md)

## Verdict

This is the best chapter of the part's second half and, in its derivations, among the best treatments of the subject I know at textbook length. No reviewer found a wrong derivation in eight sections, every worked number was recomputed, and each section has a passage at the bar: Tweedie's formula in one sentence, the annealing experiment on exact scores, the Fréchet distance read against real images, the DDIM family from "training fixes the marginals and nothing else", noise levels as image scales, the flow map as "where it arrives", the two-token table in which the cost of parallel decoding is the mutual information divided by the number of steps. The method is right throughout: an exact small case, a number predicted before the cell prints it, a measured floor. The chapter is current for 2026 (a velocity on the linear path is the default, DDPM the second model) and, unlike the GAN chapter, it does not lean on one paper.

It is not yet at the bar, for four reasons that the section reviewers could not see whole. (1) One shared function, `time_embedding`, gives every network of the chapter a time signal that is close to a linear function of $t$; the choice is deliberate for one section and undeclared in the seven others, and every image number in the chapter depends on it. (2) The chapter explains how diffusion models work and stops before telling the reader what to choose: no section ends with a procedure for picking a sampler and a step count, a target and a weighting, or a backbone. (3) From the fourth section on, conclusions are drawn from differences in a Fréchet distance that are smaller than the difference between the two frameworks' stored runs of the same row, and two experiments (the recipe comparison, the backbone scaling) do not have the design their conclusions need. (4) The chapter says of itself that "the mathematics lives in" the appendix; three load-bearing steps are in fact only there, and the chapter's close, which is also the close of the Advanced part, is the second half of a paragraph about BERT. What matters most: settle the time embedding before anything is re-captured, since it moves every number; then give each of sections 17.4 to 17.6 the experiment and the closing recommendation its question asks for.

## Score table

Scores as given by the section reviewers (index and further reading are mine).

| Section | Explanation | Thought | Pedagogy | Language | Enjoyment |
|---|---|---|---|---|---|
| [index](17-diffusion-models/index.md) | 4 | 3 | 3 | 4 | 3 |
| [denoising](17-diffusion-models/denoising.md) | 4 | 3 | 4 | 4 | 4 |
| [gaussian-paths](17-diffusion-models/gaussian-paths.md) | 3 | 4 | 3 | 3 | 3 |
| [image-diffusion](17-diffusion-models/image-diffusion.md) | 4 | 4 | 4 | 4 | 4 |
| [samplers](17-diffusion-models/samplers.md) | 4 | 3 | 4 | 4 | 4 |
| [design-space](17-diffusion-models/design-space.md) | 3 | 3 | 3 | 4 | 3 |
| [latent-diffusion](17-diffusion-models/latent-diffusion.md) | 4 | 3 | 3 | 4 | 3 |
| [few-step](17-diffusion-models/few-step.md) | 4 | 3 | 4 | 4 | 4 |
| [discrete-diffusion](17-diffusion-models/discrete-diffusion.md) | 4 | 4 | 5 | 4 | 4 |
| [further-reading](17-diffusion-models/further-reading.md) | 4 | 3 | 3 | 4 | 3 |

The three reviewers calibrated alike, and the table discriminates more than the one for the GAN chapter. Its most informative column is "thought": five of eight sections get a 3 there with a 4 for explanation, and in each case the reason is the same, a correct exposition whose experiments or claims do not finish the argument. The strongest sections are `discrete-diffusion.md` (one result, two short proofs, one exact example that carries every conclusion; the only 5 in the table) and `image-diffusion.md` (the only section with no score below 4). The weakest are `design-space.md` (three topics, two experiments that cannot answer its question) and `gaussian-paths.md`, and the second matters more: it is the section the chapter stands on, its mathematics is right, and it is the densest and the least readable of the eight, with the one theorem it rests on handed to the appendix in a subordinate clause (GP-1).

## Technical correctness and currency

**No derivation was found wrong.** All three reviewers rederived the identities of their sections (the five targets and their conversions, the ancestral step, the DDIM family, the weight of the bound as a log-SNR step, all six closed forms of the weighting table, the EDM coefficients, the flow-map identities, both propositions on masked diffusion). The findings of substance are a defect in shared code, claims stronger than their support, and prose that contradicts stored output. Ordered by severity; confidence is the reviewer's unless marked.

**Code:**

- **DEN-5, IMG-1** (moderate as filed; I rate it the chapter's most serious finding; confirmed). `time_embedding` uses frequencies from 1 down to $10^{-4}$ and is always called with $t \in [0, 1]$. I recomputed it for 128 features on a grid of $t$: the second singular value of the centered feature matrix is a tenth of the first, the third 0.4%, the fourth 0.01%. The features are in effect $t$ and $t^2$. `image-diffusion.md` describes them as "like the positional encodings" and says the prediction "depends strongly on the level". See the section "The time embedding" below.

**Claims stronger than their support, and errors of statement:**

- **SMP-4** (major; reviewer sure). "Fresh noise corrects errors" is stated in the body and in the Summary of `samplers.md`; the experiment shows only the cost of noise. The deterministic sampler wins at every budget tested, and the two families are never compared at a large equal budget.
- **LAT-14** (major; confirmed in the store). "DiT-L beats it" rests on 1.41 against 1.50 in PyTorch (JAX: 1.19 against 1.62), inside the ranges the section quotes (1.2 to 1.5, and 1.5 to 1.6), at 2.7 times the compute. The Summary repeats it.
- **DSP-4** (moderate; reviewer sure, and I recomputed). The prose says the log-SNR falls by $\log(k/(k-1))$ per step at the first steps and by $\log 2$ from the first step to the second "whatever $T$ is". On the chapter's schedule the steps are 0.780 at $k = 2$ and 0.157 at $k = 10$ for $T = 1000$ (against 0.693 and 0.105), and 1.10 for $T = 100$. The cell above the sentence prints half of the first two numbers. The formula holds for constant $\beta$ only.
- **DSP-8** (moderate; reviewer about 85% on the normalization). Flow matching and $\mathbf{v}$-prediction are "likelihood training of slightly noisy data": the mixing density over noise levels implied by the weight puts most of its mass on heavy noise. Drop "slightly".
- **DSP-14** (moderate). "the clean image is the simpler target at every level" contradicts the EDM subsection of the same file, whose point is that the best target depends on the level.
- **FEW-10** (moderate; fairly sure). The reason given for the stop-gradient ("the target moves with every update") holds with the stop-gradient too. The reason is that the averaging argument needs the network to enter through the prediction alone. Exercise 6a of the section asks for the right answer.
- **FEW-1** (moderate; fairly sure). "inversion and editing" are promised for the flow map; the trained network jumps toward the data only.
- **FEW-16** (moderate; unverified numbers). "The remaining gap at one evaluation is a matter of budget" cites "the students in the papers" without a paper, compares per example where the reader will read per iteration, and is contradicted by its own last sentence.
- **SMP-3** (moderate; sure of the mathematics). "more noise than the posterior allows" treats the posterior variance as the correct one; exercise 8 of the same section shows the optimal variance lies between the two.
- **SMP-7** (moderate). "The one-dimensional result does not transfer": half of it does. The explanation offered is untested, two rival explanations are not mentioned (one of them the time embedding, see below), and "on larger images" is probably wrong for the paper cited (unverified).
- **SMP-8** (moderate; sure). An exponential integrator is not defined by where it puts its steps. `further-reading.md` states it correctly (FR-3).
- **SMP-13** (moderate). The slow decay of the inversion error is attributed to the learned field by assertion, and a code standard deviation of 0.55 to 0.80 is reported in a neutral voice.
- **LAT-10** (moderate; sure of the arithmetic). The factor 4,096 for latents against pixels compares two transformers nobody would run; exercise 2c of the section shows that patches of 16 on pixels give the same token count.
- **DISC-12** (moderate; from memory of the papers). The comparison with autoregressive models cites a paper for its premise and omits its finding. `further-reading.md` has both halves (FR-4).
- **IMG-9** (moderate). The FD of about 3 at guidance scale 0 is attributed to an under-trained unconditional prediction; the cell cannot separate that from class proportions that are no longer balanced by construction.
- Statements not tagged `correctness` that concern truth: DSP-6 (an experiment that cannot distinguish yes from no is read as a yes, and the Summary says "Diffusion training is maximum likelihood up to a weighting"), DSP-12 (see "Experiments"), LAT-15 (the transformer "keeps improving as it grows" where only the transformer was grown), LAT-8 ("cannot fall below" for a Fréchet distance), LAT-9 (the cause of the latent gap asserted), FEW-5 (reflow "keeps the data distribution" for an exact field only), FEW-18 (which teacher bounds which student), FEW-14 (a two-dimensional effect of one paper's loss weight shown without the caveat that it is two-dimensional), DSP-10 (`sigma_data = 0.5` unchecked against data whose root-mean-square value is probably about 0.8), DSP-13, SMP-14, LAT-12.
- Minor: DEN-4, DEN-7, GP-5, IMG-3, IMG-4, IMG-7, IMG-11, DSP-9, DSP-11, LAT-3, LAT-16, FEW-6, FEW-9, DISC-2, DISC-10, DISC-17, and from my two files IDX-2, IDX-9, FR-2, FR-5, FR-6.

**Stored outputs against prose:**

- **DEN-10** (moderate). "staying near the start's in some runs and drifting in others": both stored runs drift toward the heaviest mode (0.486 and 0.394 from a third), directly after the reader has been told the score cannot see the weights.
- **FEW-13** (moderate). "Both students reproduce the mixture in one evaluation": MeanFlow's spread is 0.33 against 0.48 for the data and stays at 0.37 to 0.39 with more jumps. The section named for the method ends its central experiment on a defect it does not explain.
- **SMP-15** (minor). The edit at $t_0 = 0.3$ is read as gaining "the seams, prints and edges"; its FD is about 30.
- **DSP-4** (above), **LAT-14** (above).
- **GP-9** (minor): $\bar\alpha = 0.079$ is quoted where the cell prints $\alpha = 0.28$. **DEN-11** (minor): "twice the spread" for a truncated statistic where the variance is five times. **IMG-8** (minor): "a tenth to a quarter lower" against 0.89 and 0.71, 0.99 and 0.90.
- Numbers no cell prints: the floor at 1,000 images in `latent-diffusion.md` (LAT-7; "about six times" is 4.6 to 5 by the next section's cell); "Training our student longer lowers its one-step FD only slowly" (FEW-16); "whichever target the network predicts" (IMG-7); the speckle of the fifty-step ancestral images (SMP-11); "about 9 nats" for the bound's integral (code notes of `design-space.md`); the cost of a training step with a Jacobian-vector product (FEW-12); the scoping-run figures in exercise 10c of `discrete-diffusion.md`; and the sentence in `few-step.md` that embedding $1000\,t$ "turns the target into noise" (my observation; attributed to a paper, shown by no cell).

**Unverified claims about papers** (a reviser should check each): IMG-7 (the warm-up of Ho et al.), SMP-7(c) (the evaluation counts and resolutions of Karras et al.), DSP-9 (Nichol and Dhariwal's two remedies), DSP-11 (Stable Diffusion's terminal signal), DSP-14 (Li and He), LAT-16 (dimensions per head in DiT), FEW-6 (later reflow results), FEW-16, DISC-12, IDX-2, IDX-9, FR-9 (six citations dated 2026), the interval attributed to Kynkäänniemi et al. in an exercise of `image-diffusion.md`, and the exercise attributions to three university courses.

**Currency.** The chapter is current, and its choices of default will age well because each method is derived from an identity, not narrated as a system (structural finding 5 of `few-step.md`). Four places describe practice that has moved without saying so: the U-Net is presented as the standard image network (IMG-5), while the same chapter's sixth section replaces it; multistep solvers, which are what practitioners run at 10 to 25 evaluations, get one imprecise sentence (structural finding 3 of `samplers.md`, SMP-8); "most large text-to-image systems since 2022 diffuse latents" omits the return of pixel-space transformers that the section's own exercise and the further reading describe (structural finding 4 of `latent-diffusion.md`); and the autoencoder paragraph has not followed the GAN chapter's account of the move away from the critic (LAT-6). The parts most likely to date are the MeanFlow loss-weight experiment (FEW-14) and the entries of 2026 in the further reading (FR-9).

## The time embedding

This was put to me as a confirmed finding to be traced. The facts: `time_embedding` is defined and saved in `denoising.md` with frequencies $10^{-4i/64}$, at most 1; every call in the chapter passes $t \in [0, 1]$ (or $t - s$, or one fixed $\sigma$). The 128 features then span, to within half a percent, the functions $1$, $t$ and $t^2$.

**The section reviews contradict each other here, and the source explains why.** DEN-5 and IMG-1 call this a weak time signal and propose to scale $t$ inside the function, as published code does. The review of `few-step.md` praises, under "What works", the paragraph of that section which says the low frequencies are needed: the flow-map losses differentiate the network with respect to $t$, the derivative "passes through the network's time features, which must vary slowly", and "Diffusion code often embeds $1000\,t$, which ... turns the target into noise". Both reviewers are right about what they read. The chapter made a choice for the sake of section 17.7 and applied it to every network from 17.1 on without saying so; the reader is told in 17.3 that the features are "like the positional encodings", is asked in an exercise of 17.3 to recognize "$1000\,t$" as a bug, and learns the reason five sections after the function is defined. No cell tests the reason.

**Every network and experiment that depends on it:**

| Network | Where the time enters | Sections and experiments |
|---|---|---|
| `TimeMLP` | `time_embedding`, concatenated per time | `gaussian-paths.md`: the velocity model and the noise model (floors, both samplers, the dictionary check). `design-space.md`: the noise model whose likelihood is computed. `few-step.md`: the mixture teacher and both students (two times each). In `denoising.md` the noise level is fixed, so the features are constant and nothing changes. |
| `UNet` | `time_embedding`, a two-layer MLP, one bias per channel in every block | `image-diffusion.md`: the baseline, the moving-average comparison, the guidance sweep. `samplers.md`: all seven sampler and grid combinations, inversion, interpolation, editing, inpainting. `design-space.md`: all four recipes, `EDM` included (it wraps the same network and passes the path's $t$). `latent-diffusion.md`: the latent and the pixel U-Net. `few-step.md`: the image teacher. `discrete-diffusion.md`: `MaskedUNet`. |
| `UNetFlowMap` | the U-Net's embedding plus a second `time_embedding` of $t - s$ | `few-step.md`: the image student. |
| `DiT` | `time_embedding`, an MLP, then adaptive layer norm in every block | `latent-diffusion.md`: DiT-S, M, L on pixels and DiT-M on latents. |

Not affected: every experiment on exact fields (Tweedie, blindness to mode weights, the Langevin chains with exact scores, annealing, the samplers on the exact predictor, the exact-field rows of the likelihood table), the autoencoder, the feature network behind the Fréchet distance, and `MaskedMLP`, which takes the raw $t$.

**Which conclusions might change after a fix.** This is a judgment, and none of it has been run.

- Safe: everything in two dimensions. The learned predictors sit within $10^{-3}$ of the exact ones (the code notes of `gaussian-paths.md`), so the mixture does not need more. Safe by argument: the bias of the Fréchet distance with sample count, the value of the moving average, the trade of guidance, the reason a noise predictor fails where $\alpha_1 \approx 0.007$.
- The baseline. The FD of about 1 at 5,000 samples, and the plateau of about 2 at 1,000 samples that SMP-9 reads as "the model's error", are properties of a network that resolves the noise level poorly. Both are the reference for five later sections.
- The step grid. The log-SNR on the linear path changes fastest near the two ends of $[0, 1]$, which is where features with slope at most 1 in $t$ distinguish levels least. The finding that the grid crowded toward low noise loses on this model, and that Heun's method gains nothing after about twenty evaluations (SMP-7, SMP-9), is the conclusion I would trust least. The same holds for the slow decay of the inversion error (SMP-13), which lives in the first low-noise steps.
- The recipe comparison (DSP-12). The four recipes differ in where their difficulty lies along the path, and the variance-preserving path compresses its hard levels into a short interval of $t$. A sharper time signal could change the order of the rows as easily as their values.
- The backbone comparison (LAT-14, LAT-15). The U-Net adds the embedding as a bias; the transformer lets it set the scale, shift and gate of every block. The two need not gain equally, so "the U-Net matches DiT-M with a sixth of the parameters" and the size of the last step in the scaling curve are both open.
- Few-step generation, in the other direction. If the frequencies are raised for all networks, the flow-map target gets noisier, by the section's own argument. MeanFlow's unexplained contraction (FEW-13) and the one-step gap on images (FEW-16) may be partly the price of a network that sees $(s, t)$ through near-linear features, or may get worse with sharper ones. This is the one place where the present choice has a stated reason.
- Probably little: masked diffusion on images, where the section itself shows the time input is redundant (DISC-10).

**What to do.** Do not adopt $1000\,t$ blindly. Put on the list for the development machine one small study on the image baseline: the present features; a frequency range matched to $[0, 1]$ (highest frequency of a few tens of radians per unit $t$); and features of the log-SNR or of $\log\bar\sigma$, which is what EDM passes and what DSP-10 notes the chapter's `EDM` class does not do. Report the FD at 5,000 samples in both tabs for a velocity model and the one-step FD for a flow-map student. Then choose once, re-capture the seven notebooks that use `UNet` or `DiT` together (the closing note of the `image-diffusion.md` review), and reword the three descriptions (17.1, 17.3, 17.7) and bug (ii) of :exref:`ex_image-diffusion-broken-trainer`. If the present features are kept, say in `denoising.md` what they are for $t \le 1$ and why, in one sentence, where the function is defined.

## Three more questions put to this review

**Does the chapter read without the appendix? Nearly, and the exceptions are few and cheap.** The reviewers checked every appendix label the chapter cites and found that each resolves to a passage that holds the cited content. Most citations are a parenthesis after a result stated with its reason, which is the right use. Five passages cannot be followed by a reader who has not opened the appendix: the continuity equation, on which the claim that the marginal velocity transports the density rests (GP-1, the most important); the reverse stochastic differential equation, written in a notation the main text never introduces (GP-3); the evidence lower bound, cited and not shown, on which the first third of `design-space.md` stands (DSP-1); the retelling of the appendix's one-dimensional solver experiment in one 60-word sentence (SMP-6); and the claim that guidance breaks the curvature bound, whose only demonstration is an appendix experiment (IMG-10). Each needs three or four sentences, and drafts are in the section files. Fourth-order Runge-Kutta is used in `design-space.md` without a main-text introduction (DSP-5). The sentence in the introduction that "the mathematics lives in" the appendix should be rewritten to the author's rule (IDX-5).

**Does the chapter end the part well? The material does; the writing does not.** The order is right for a last chapter. The part began with a recurrent language model that could not hold a sentence together, and it ends on the question "What is noise for a token?", with BERT and GPT placed as the two ends of one family of models; the next part is Language Models. No better last section could be chosen. But the reader is not told that anything is ending. The close of the chapter is the second half of a paragraph in a subsection titled "Masked Language Models as One Step" (DISC-16; I confirmed the passage), it repeats the introduction's promise in the past tense (IDX-6), one of its four claims is not supported by the chapter (LAT-15), and the forward pointer beside it has no label (DISC-18). The comparison with autoregressive models, which is the bridge to the next part, is spread over three places and rests on half of a citation (structural finding 1 of `discrete-diffusion.md`, DISC-12). Proposal: (a) collect the comparison into one subsection after the experiments, as that review lays out; (b) end the chapter with a short part of its own, not under BERT, that says what each of the four answers cost (a thousand evaluations, then fifty, then a second training run to reach four; a likelihood that the preferred weighting gives up; an autoencoder that bounds quality; revision that masked diffusion does not offer), with the table of model families `tab_diffusion_models` as its last element, moved from the Summary of `gaussian-paths.md` where the reader cannot yet read half its rows (GP-11); (c) one sentence that hands over to the next part by label once a label exists. Whether the Advanced part needs a closing page of its own is a question for the overview; at present this paragraph is all it has.

**What remains to align with the appendix chapter?** The appendix review found that `chapter_mdl-dynamics` was "written as if it were the book's only treatment of diffusion" and asked for one division of labour: the appendix keeps theorems, their smallest worked cases and the exact-field experiments; chapter 17 keeps what a practitioner chooses. Seen from this side, with the author's rule applied:

- *Homes confirmed here.* Guidance (the derivation, classifier-free guidance, the tilt) belongs to `image-diffusion.md`; the appendix keeps the curvature proposition (structural finding 5 of `image-diffusion.md`). The limits of one noise level and annealing belong here, where the mixture has unequal weights. Solver order and step grids belong to `samplers.md`. The table of model families belongs here (GP-11). The weight of the variational bound belongs to `design-space.md`.
- *To cut on this side.* "The Normalizer Never Appears" retells the appendix subsection paragraph by paragraph and ends in a cell that confirms a formula (DEN-6); reduce to one paragraph. The one-dimensional solver result (SMP-6); reduce to a pointer.
- *To add on this side.* GP-1, GP-3, DSP-1 and IMG-10, as above. The appendix review suggested that `gaussian-paths.md` shorten its reasons for the marginal velocity and the conditional flow matching identity to pointers. Under the author's later rule the opposite holds: these are small, they stay in both places, and the one missing reason is added.
- *References to repair.* The guidance subsection of the appendix needs a label of its own (IMG-11). Two facts are cited to the appendix that have a main-text owner the reader has read: the conditional-mean property of squared error (DEN-4; owner :numref:`sec_gan_beyond`) and the split of a stochastic step into a Langevin step and a flow step (SMP-16; owner :numref:`sec_diffusion-gaussian-paths`). The appendix review reported a wrong citation of `sec_mdl-ddim-guidance` in `gaussian-paths.md`; it is no longer in the file.
- *Labels that must stay stable.* Ten of the chapter's citations into the appendix point at exercises (mode merging three times, the variance of Hutchinson's estimator, the Bayes-risk floor, annealing, guided densities and levels, late decisions, target conditioning). The appendix revision should state those results in its body and must keep the labels. If the appendix cuts "Sampling with a Learned Field", the citation of `sec_mdl-sampling-learned-dynamics` in `samplers.md` has to be redirected.
- *The other direction.* The appendix says three times that "the previous section trained a network"; none is trained there. The networks are trained here, and the appendix can say so by label (`sec_diffusion-gaussian-paths`). Its further reading should drop the entries on the design space and few-step generation, which are full sections here.
- *Agreement.* Notation ($\tilde{\mathbf{x}}$, $\alpha_t$, $\sigma_t$, $\bar\alpha_t = \alpha_t^2$, data at $t = 0$) and the callout on the clock agree between the two chapters; no long derivation is written out in both.

## Arc and order

The chapter has one line of argument and the sections are in the right order: what a denoiser knows (17.1), a path through all noise levels and one regression (17.2), images and how to judge them (17.3), then four sections that each change one thing (the sampler, the objective, the representation and backbone, the number of steps), then tokens (17.8). Every hand-over sentence between sections was checked by a reviewer and works; several are singled out ("fifty evaluations per sample remain fifty; the next section asks how to get by with one or two"). Each section has one ending, with a single exception (GP-11). Neither holds for the GAN chapter or the appendix.

Problems of proportion and boundary:

- **`gaussian-paths.md` is two sections.** The seam is before "Sampling by Integration": what is trained (path, chain, annealing, regression, targets, floor), then how to sample (probability-flow ODE, Euler, ancestral chain, reverse SDE). Each half then has room for the sentences it lacks (GP-1, GP-2, GP-3). I recommend this split and no other. The reviewers of `design-space.md` (three topics) and `latent-diffusion.md` (two arguments) both name a seam and both stop short of recommending a split; applying all three would give eleven teaching sections. Nine is the ceiling, and the chapter should pay for the ninth with the cuts listed below.
- **The first section pays the chapter's setup cost.** Eight of the first nine cells of `denoising.md` are shared helpers, some for paths and networks defined sections later (DEN-1, DEN-3). The same holds on a smaller scale in `gaussian-paths.md` (`make_pred`, `time_at`, `ode_sample`; GP-7) and `image-diffusion.md` (`cfg` 350 lines before guidance is explained, IMG-2; an undocumented `cond` argument, IMG-14). Shared code sits in the first section that could hold it, not the first that needs it.
- **`design-space.md` is three topics and its proportions are inverted.** A quarter of the section computes a likelihood that the Summary reports in half a sentence; the weighting view, which a practitioner will use, has no cell (structural findings 1 and 3 there).
- **`latent-diffusion.md` promises three answers and gives two.** How a caption steers the model gets one paragraph of six systems (LAT-1, LAT-19). Cut the promise or teach it.
- **Three sections end their body with a survey**: "Cross-Attention and Joint Attention" and "Video" (LAT-19, LAT-20), "The Other Route: Matching Distributions" (structural finding 4 of `few-step.md`), "Parameterization at Finite Capacity" (DSP-14).
- **Boundaries that leak.** The loss floor is announced in `denoising.md` and taught in `gaussian-paths.md` (DEN-8). DDIM is named in code comments one section before it is defined (GP-12). The table of models sits in the Summary of 17.2 (GP-11). The evidence that stochastic sampling can help is in 17.5, the claim in 17.4, and only one cites the other (SMP-4). The reason for the time features is in 17.7. The one-step mean is taught in 17.2 and retold in 17.7 (structural finding 2 of `few-step.md`).
- **No picture of the process.** In eight sections nothing shows an image being noised or a sample forming (IMG-12). This is the largest gap between the chapter and an Illustrated-Transformer explanation, and two short cells close it.

## The most important findings

1. **Settle the time embedding, then re-capture once.** DEN-5, IMG-1, and the section above.
2. **Tell the reader what to choose.** The samplers section ends with a statement about one model, the design-space section without a recommendation, the latent section with judgments withheld. Drafts exist for the first two. SMP-9, DSP-15, LAT-18, and the closing notes of the `design-space.md` review ("the chapter teaches how diffusion models work and not how to make the choices").
3. **State the resolution of the Fréchet distance once and read every table by it.** At 1,000 samples the real-against-real value is 0.8 to 0.9 and moves by 0.05 to 0.08 between draws; the same row differs by 0.3 and up to 75% between the two stores. Three reviewers asked for this independently. SMP-9, SMP-7, IMG-8, DSP-12, LAT-14, LAT-15.
4. **Make the design-space experiments answer the section's question.** Train the bound-weighted model beside the simple-loss model, at a capacity where they can differ; replace the four-recipe table by comparisons that change one thing. Correct the log-SNR sentence. DSP-6, DSP-12, DSP-4.
5. **Make the main text carry its own theorem.** Why the average velocity moves the density (GP-1), the sampler coordinates every later section uses (GP-2), the notation of the reverse SDE (GP-3), one line of Jensen for the bound (DSP-1); and split `gaussian-paths.md`.
6. **Test or withdraw the claim that noise corrects errors**, and fix the statement about the posterior variance. One added row (both DDIM variants at 1,000 steps) decides it. SMP-4, SMP-3.
7. **Make the scaling experiment a comparison, or say whose claim it is.** LAT-15, LAT-14, LAT-10; and close the loop on why the latent model loses (LAT-5, LAT-9).
8. **Make the four passages around the flow-map derivation as exact as the derivation.** FEW-10, FEW-1, FEW-13, FEW-16, with the caveat of FEW-14.
9. **Give the chapter and the part a close, and the comparison with autoregressive models one home.** DISC-16, DISC-12, DISC-1, IDX-6.
10. **Move shared code to where it is needed and introduce cells by the argument.** DEN-1, GP-7, IMG-2, and the inventory pattern counted in `denoising.md`.

## Focus

- **Surveys inside teaching sections**: LAT-19 and LAT-20; the closing part of `few-step.md`; DSP-14; the last paragraph of "Inpainting and Posterior Sampling" (SMP-17).
- **Paper retellings**: the adaptive-layer-norm comparison (LAT-13); the scaling claim of Peebles and Xie, which the experiment is read as reproducing (LAT-15); "the students in the papers" (FEW-16); the finding of Li and He (DSP-14). Compared with the GAN chapter these are few and short.
- **Roadmaps by citation**: three section openings introduce the section through its papers, one sentence each (SMP-1, DSP-16, DISC-3).
- **Detail that serves nobody**: 550 words and a cell on estimating a normalizer by importance sampling (DEN-6); the description of the trainer's options in the middle of the argument for Tweedie's formula (DEN-1); a bisection inside every sampler call, explained in one compressed sentence (GP-7).
- **Further reading** keeps its rule of one idea per entry and does not rank (FR-1); its discrete-diffusion heading is a chapter in outline (structural finding 3 of that review).
- The rule proposed in the GAN review holds here with little cost: a system is named in a teaching section only if the section derives or runs the idea it stands for.

## Experiments

- **Cells that only confirm or cannot fail**: the importance-sampling cell (DEN-6) and the first printed line of the Tweedie check (DEN-13); three confirmations in a row in `gaussian-paths.md` (GP-6, structural finding 2 there); the DDIM-equals-Euler check (SMP-2); the exact field checked under both likelihood measures (code notes of `design-space.md`). By the author's rule one or two may stay; I would keep the dictionary check as one printed row and drop the others.
- **Read for more than they show**: the third Langevin chain (DEN-10); the guidance row at scale 0 (IMG-9); the sampler study, in three places (SMP-4, SMP-7, SMP-9); the inversion error (SMP-13); the edit at $t_0 = 0.3$ (SMP-15); the mixture likelihood (DSP-6); the latent gap (LAT-9); the scaling curve (LAT-15); the adaptive loss weight (FEW-14); the hundred-step speckle count (DISC-14).
- **Unfair or confounded comparisons**: the four recipes differ in two to four factors at a time, on one seed at a third of the usual budget, and "EDM" is not sampled on its own grid (DSP-12); the transformer is grown and the U-Net is not (LAT-15); the two latent models are compared at 15 against 185 GFLOP (LAT-18); the chain is compared at 1,000 steps with deterministic samplers that stop at 100 (SMP-4, IMG-8); independent pixels are the only yardstick for 0.17 bits per pixel (DISC-13).
- **Experiments that do not test the section's question**: "which choices matter" is asked of a model with one weighting (DSP-6); "the transformer scales better" of a plot with one U-Net point (LAT-15); "noise corrects errors" of budgets at which it only costs (SMP-4).
- **Scale and cost.** The brief requires experiments that run on a reader's machine. The planar and two-token experiments do, in seconds. From 17.3 on they need a GPU and no section says what the total is: evaluation in `image-diffusion.md` costs more than twice its training, and the single thousand-step chain more than the training (IMG-6); `latent-diffusion.md` trains seven networks for 20,000 updates, half of the compute on DiT-L (code notes there); `few-step.md` trains a teacher and a student with a Jacobian-vector product (code notes there); two planar networks train for 40,000 steps each with no stated time (`gaussian-paths.md`). The image model is at a scale where quality can be judged, which the reviewer of `image-diffusion.md` checked and confirmed; the scale is too small for the later sections' finer claims.
- **Two sample counts.** `image-diffusion.md` teaches that an FD "can therefore be compared only with FDs from the same number of images" and scores at 5,000; the later sections score at 1,000, where the floor is five times higher. The switch is legitimate and should be said once where it happens.
- **Done well and to be kept**: prediction before measurement in `denoising.md`; annealing on exact scores; the real-against-real reference and the ranges over seeds in `image-diffusion.md`; shared noise and counted evaluations in `samplers.md`; paired likelihood differences in `design-space.md`; the zero-initialization check that could fail in `latent-diffusion.md`; teacher and student on the same noise in `few-step.md`; the four reveal orders computed exactly, with no training, in `discrete-diffusion.md`. The last is the model the noisier sections could follow: an exact small case in place of a single-seed FD.

## Dependencies

- **From Basics**: squared error and conditional means, Gaussians, KL divergence, convolutions and normalization, cross-entropy. Nothing missing was reported.
- **From earlier Advanced chapters**: positional encodings (chapter 10), `ViTBlock` and `PatchEmbedding` from :numref:`sec_vision-transformer` and the masked-token objective (chapter 11), and from the GAN chapter the conditional-mean handover, the moving average, the Fréchet distance and KID, the Gaussian KL formula (used twice by equation label and never restated; structural finding 4 of `design-space.md`). All cited where used except the masked-token objective, which is cited to the retired chapter in the Attic instead of `subsec_mlm-loss-by-position` (DISC-17). The link to the GAN chapter is one-way: this chapter cites it in at least five files; it cites this one twice, by chapter label.
- **From the appendix**: see the second question above. My count of appendix citations before the slides marker, exercises and equations included: `gaussian-paths.md` 22, `denoising.md` 11, `design-space.md` 9, `discrete-diffusion.md` 8, `few-step.md` 6, `samplers.md` 5, `image-diffusion.md` 4, `latent-diffusion.md` 2. The density in `gaussian-paths.md` is itself a finding: one citation per 190 words of body prose makes a section read as a commentary on another text.
- **References that point to the wrong place**: none that fail to resolve. Three point to a place the reader should not be sent: :numref:`sec_fcn` is presented as known and lies in a later part (IMG-3); `sec_bert` is in the Attic (DISC-17); `sec_mdl-ddim-guidance` labels a whole appendix part when one subsection is meant (IMG-11).
- **Forward references to the Language Models part**: none by label (I searched the ten files for `chap_lm` and `sec_lm`). Three unlabelled promises: "a later part of this book" in the introduction and in `latent-diffusion.md` for text-to-image and video (IDX-7, LAT-2), which nothing in the tree stands behind, and "the next part of this book" in `discrete-diffusion.md` (DISC-18). None is load-bearing. The dependence runs the other way: the plan for the Language Models part restates the masked-diffusion bound from this chapter, so `subsec_diffusion-masked-ce` and `eq_diffusion-masked-loss` must stay stable.

## Repetition

Inside the chapter:

- The one-step Euler mean: derived, run and set as an exercise in `gaussian-paths.md`, then in the opening, a subsection, the Summary and an exercise of `few-step.md` (structural finding 2 there). Home: `gaussian-paths.md`; `few-step.md` keeps the cell for its two- and four-step rows.
- The moving average of weights: `sec_dcgan`, `denoising.md`, `image-diffusion.md`, with three symbols for the decay (IMG-13, DEN-12). Home: `image-diffusion.md`, which has the evidence.
- The time features: described in `denoising.md`, `image-diffusion.md` and `few-step.md`, in three ways that do not agree. Home: where the function is defined.
- "One object behind three routes": the introduction and the opening of `gaussian-paths.md` (GP-13). Home: the introduction.
- "Nothing used the number two": the end of `gaussian-paths.md` and the opening of `image-diffusion.md` (GP-11). Home: the opening.
- The four limits and their answers: the introduction and the last body paragraph (IDX-6, DISC-16).
- A stochastic step as a Langevin step plus a flow step: `gaussian-paths.md`, and again by appendix reference in `samplers.md` (SMP-16).
- Contraction of the mixture's "spread": explained for Euler strides in `samplers.md` (SMP-5), unexplained for MeanFlow in `few-step.md` (FEW-13), and measured by a truncated statistic that `denoising.md` does not describe as such (DEN-11). Home: define the statistic once in `denoising.md` and explain contraction once in `samplers.md`.
- The loss floor (DEN-8), the classifier-free guidance formula (LAT-19), "all noisy points look alike" with two thresholds (SMP-10), a sentence restating its predecessor (LAT-17), "where it arrives" in body and Summary of `few-step.md`.
- Sections against `further-reading.md`: latent systems and video (LAT-19, LAT-20), exponential integrators (FR-3), the autoregressive comparison (FR-4), the order-agnostic equivalence (FR-7). In two of these the further reading is the more accurate text.

With other Advanced chapters: the conditional-mean argument and the remedies for a one-step student are owned by :numref:`sec_gan_beyond`, the flow-map route by `few-step.md`; the division is sound, but the distribution-matching objective is displayed in neither chapter outside an exercise (structural finding 4 of `few-step.md`). The perceptual loss is described with a drift between the two chapters (LAT-6). The masked-token objective is taught in chapter 11 and should be cited from there (DISC-17).

With Basics: nothing reported.

With the appendix: see the third question above. By the author's rule the only passage to cut is DEN-6; guidance is duplicated but its home is here.

## Recurring patterns

- **Sections stop before the recommendation.** `samplers.md`, `design-space.md`, `latent-diffusion.md` (SMP-9, DSP-15, the voice pattern of the `latent-diffusion.md` review). The prose is exact and seldom ranks its own material.
- **A fact the body needs is in an exercise, or an exercise corrects the body.** The exercise asks for the right reason where the body gives another (FEW-10 and exercise 6a); the exercise shows the body's variance claim is too strong (SMP-3 and exercise 8); the exercise refutes the body's cost argument (LAT-10 and exercise 2c); the two experiments `design-space.md` needs are its exercises 3b and 4d (DSP-6, DSP-12); the cause of the latent gap is exercise 7 of `latent-diffusion.md`; the floor $H + I/T$ and the order-agnostic bound are exercises 7 and 8 of `discrete-diffusion.md` (four instances counted there); the third limit of a small noise level is an exercise of `denoising.md` (DEN-9); the Gaussian case that would demonstrate guidance in the text is an exercise of `image-diffusion.md` (IMG-10). The GAN review found the same habit. The exercises are more careful than the prose.
- **Exercises.** 87 problems and about 14,000 words, close to half the body, above the band of ten in five of eight sections. Every reviewer called a set the strongest part of a section. Several are afternoons of compute; the proposed moves (into the text, into the further-reading notebook) are in the section files.
- **Cells introduced by an inventory sentence.** About 11 of 17 cells in `denoising.md` open with a backticked name as subject or "The cell checks"; 4 of 15 in `samplers.md` (SMP-19); the compressed code sentences of `gaussian-paths.md` carry 22 semicolons against 9 in its neighbour. Cells introduced by the argument ("small compared with what?") read far better, and most of the later sections do it.
- **Openers and headings on a template.** Three-question roadmaps, 4 in the first two sections; slogan headings of the form "One ...", 5 (the pattern note of `gaussian-paths.md`); tag headings in `image-diffusion.md`, 3; one heading that is a sentence (DEN-6).
- **Costs of evaluation unreported** where the cost of training is given (the closing note of the `image-diffusion.md` review; also 17.2, 17.6, 17.7).
- **Plumbing in the prose or in saved code**: compilation and kernel launches (DEN-2), imports inside saved functions with a comment about the preamble, 2 (the pattern note of `few-step.md`), a stray import cell in `discrete-diffusion.md`, a fact about initialization that only the JAX tab states (IMG-5).
- **Notation.** $s$ has five meanings and $\sigma$ five decorations in `gaussian-paths.md`; the bar means a square in $\bar\alpha_t$ and a quotient in $\bar\sigma$ (GP-2); $h$ is a half-life and a step size (DEN-12); the last section turns $t$ into an integer, $u$ into a masking probability and $\bar\alpha_t$ into a survival probability (DISC-4); $\cosh^{-1}$ is used for $1/\cosh$, 5 times (DSP-7); $\beta(t)$ is used and never defined (GP-4); "oracle" and "dictionary" are coined and then used as proper nouns across three sections (GP-10). Against this, the two middle sections keep $\bar\sigma$, $\lambda$ and $\nu_k$ consistent with visible care. A five-line notation table in the introduction would remove most of the load (structural finding 5 of my `index.md` review).
- **Language** is clean in every section: no stock phrases, no double endings, few intensifiers. Voice is present in the openings (the camera denoiser, the slider, the cat) and thins in the middle of sections.
- **Framework tabs** do not fragment the prose anywhere; there is no TensorFlow or MXNet material.

## What to cut

About 3,000 words of body prose and 1,500 of exercises, which pays for the additions asked for above:

- `denoising.md`: "The Normalizer Never Appears" to one paragraph and no cell (DEN-6), the trainer's options (DEN-1), the gap line (DEN-13). About 600 words.
- `gaussian-paths.md`: two of the three closing confirmations (GP-6), the mechanism of `make_pred` (GP-7), the second ending (GP-11). About 500 words.
- `samplers.md`: the confirming cell (SMP-2), the retold appendix experiment (SMP-6), two of three opening attributions (SMP-1), the closing pointer (SMP-17). About 300 words.
- `design-space.md`: "Parameterization at Finite Capacity" merged or moved (DSP-14), one of the two exact-field likelihood checks, the roadmap by papers (DSP-16). About 300 words.
- `latent-diffusion.md`: text conditioning to two sentences unless it is taught (LAT-19), video (LAT-20), the restating sentence (LAT-17), and DiT-L to an exercise if U-Net sizes are added instead (LAT-15). About 400 words.
- `few-step.md`: the retelling of the one-step mean, the guidance-distillation paragraph, the loss-weight cell to an exercise (FEW-14). About 350 words.
- `discrete-diffusion.md`: the intermediate-states grid (structural finding 5 there), the renormalization remark (DISC-5), the four-limits sentence (DISC-16). About 200 words.
- `further-reading.md`: half of the discrete-diffusion heading.
- Exercises: bring the five sections above the band down to ten each, by the moves the section files name.

## What is missing

- One sentence, where `time_embedding` is defined, on what its features are for $t \le 1$; and the study that decides whether to keep them.
- A decision procedure for sampler and step count (SMP-9), a recommendation for target, weighting and schedule (DSP-15), and a plain verdict on when a latent pays (LAT-18).
- The resolution of the Fréchet distance at the sample counts the chapter uses, stated once.
- Why the average velocity transports the density (GP-1); the name "variance exploding" (GP-8); the two-line derivation of the sampler coordinates (GP-2).
- A picture of an image being noised and of a sample forming (IMG-12).
- What the small U-Net omits, and that it is no longer the standard (IMG-5).
- Multistep solvers, and one measurement of sampling under guidance (structural finding 3 of `samplers.md`).
- One line of Jensen's inequality (DSP-1); one-factor comparisons (DSP-12); a bound-weighted model (DSP-6).
- A second U-Net size, or the statement that the scaling claim is the paper's (LAT-15); two values of the KL weight (structural finding 3 of `latent-diffusion.md`).
- The distribution-matching gradient, displayed once (structural finding 4 of `few-step.md`); the cause of MeanFlow's contraction, or the statement that it is not known (FEW-13).
- The comparison with autoregressive models as one subsection; two sentences on the continuous-time Markov chain and on ratio scores, which are the first words a reader meets in the 2024 to 2026 literature (structural findings 1 and 3 of `discrete-diffusion.md`).
- A close for the chapter and the part.
- In the introduction: a route with a judgment, the chapter's default model, the running examples, the prerequisites, the cost, and the notation of the clock (IDX-3, IDX-4).
- In the further reading: a ranked start (FR-1), an entry on text conditioning, an entry on diffusion beyond images and text.

## What is best and should set the model

- **`denoising.md`, Tweedie's formula in one sentence**, and "Blindness to Mode Weights": "An error of $0.2$ in the Fisher divergence sounds small, but small compared with what?", answered by two mixtures 0.07 nats apart whose scores differ by $10^{-5}$.
- **`gaussian-paths.md`, the annealing experiment on exact scores** ("The proportions are decided at the coarse levels, where the smoothed density has no valleys; the fine levels only sharpen each mode"), the build-up from one data point, and the ancestral step split into a Langevin step and a flow step.
- **`image-diffusion.md`, "Reading the Number".** A score read against two sets of real images at two sample sizes, a second score for the bias, ranges over training seeds, and the sentence that guided samples are "the class's prototypes, easier to classify than real ones". The model for every scored experiment in the book, together with the instruments of `dcgan.md`.
- **`samplers.md`, the DDIM family from one observation**, the collapse of a Gaussian cloud in one stride, the Heun step worked by hand, and a negative result reported against the cited paper's recommendation.
- **`design-space.md`, the weight of the bound as a log-SNR step** ("The bound charges every step by how much it lowers the log-SNR"), "a weighting in disguise", and "Noise Levels Are Scales", a question, twelve lines of code on the reader's own data, and two consequences.
- **`latent-diffusion.md`, the reconstruction FD as a floor**, and adaptive layer norm with a zero gate stated as a testable fact and then tested.
- **`few-step.md`, the flow map.** "A velocity model says which way a trajectory points; a flow map says where it arrives", the one-dimensional Gaussian in which the two differ, three identities each with a physical reading and a cost, and four published methods as the cells of one table. A reader can place a paper of 2028 with it.
- **`discrete-diffusion.md`, the two-token table.** The independence error, the floor $H + I/T$, and confidence-ordered unmasking shrinking the smallest mode from 0.2 to 0.07 with no training at all. Also the chapter's best opening image: "Adding a small real number to the index of the word "cat" does not produce a slightly noisy cat."
- **The exercises**, in every section.

What these have in common is a case small enough to compute exactly, placed before the general statement, with the number predicted before it is printed and a floor to read it against. The chapter does this in two dimensions and on two tokens without exception. The revision should hold the image experiments of sections 17.4 to 17.6 to the same standard, or say where they cannot meet it.

## Contradictions between section reviews

- **The time embedding.** DEN-5 and IMG-1 treat the low frequencies as a defect and propose scaling by 1,000; the review of `few-step.md` lists the paragraph that defends them among the section's strengths. Resolved above: the choice is deliberate for one section, undeclared elsewhere, and untested.
- **How many sections.** One reviewer recommends splitting `gaussian-paths.md`; two others name seams in `design-space.md` and `latent-diffusion.md` without recommending a split. Resolved above: one split.
- **What counts as good.** `samplers.md` treats two to three times the real-against-real floor as the reference quality of the thousand-step chain; `design-space.md` calls about three times that floor "far from the data" in the source, which its reviewer noted (DSP-12(c)). The chapter needs one vocabulary for distance from the floor.
- **Appendix references.** The reviewer of `gaussian-paths.md` found all 18 labels correct; the appendix review reported one wrong citation in that file. The citation is no longer there.

No two reviews disagree about a mathematical statement or about the order of the sections.

## Could not check

- Rendered HTML and PDF; I read sources, stored text outputs and the review files, and did not re-render figures the section reviewers had inspected.
- Anything that needs a run. There is no PyTorch or JAX on this machine. The effect of the time embedding on any number, the claims over repeated runs (the store holds one run per framework), and every proposed experiment (the embedding study, DDIM at 1,000 steps, one-factor recipe comparisons, a bound-weighted model, U-Net sizes, the KL weight, a longer MeanFlow run, an exact $S$-step table for the pair model) are untested and belong on the list for the development machine.
- The claims about papers listed as unverified, in particular those dated 2026.
- `CLAUDE.md` still describes chapter 17 as a placeholder; the appendix review noted this and it remains.
