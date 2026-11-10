# Further Reading (Diffusion Models)

Source: `chapter_diffusion-models/further-reading.md`, read in full (the file has no slides section). About 1,700 words, 11 headings, 34 bold entries, about 90 citations. No code. The labels cited resolve (`sec_gan-further-reading`, `sec_mdl-dynamics-further-reading`, `sec_basic_gan`, `sec_rl_sequences`, `sec_regularized`, `ex_samplers-dpm-solver`). Claims about papers were checked against what I know of them; those I could not confirm are marked unverified.

## Verdict

A disciplined page that keeps its own rule: "Each entry states what a later work changes; nothing in the book depends on it." Almost every entry is one sentence that names the one idea of a paper and ties it to the section it extends, with few scores and no leaderboard. It is better in this respect than the further reading of the GAN chapter. Against the bar it lacks two things. It does not rank: 34 entries and about 90 papers stand at equal weight, the headings do not follow the order of the sections, and a reader with one evening is not told what to read. And in two places it is more accurate than the teaching section it serves (exponential integrators against `samplers.md`, the comparison with autoregressive models against `discrete-diffusion.md`), which means the body is carrying the looser statement of something this page gets right. The most important fix is a short ranked list at the top (FR-1), then reconciling the two passages with the body (FR-3, FR-4).

## Scores

| Dimension | Score | Note |
|---|---|---|
| Clarity of explanation | 4 | One idea per entry, stated in the chapter's notation |
| Clarity of thought and focus | 3 | No ranking; headings out of the chapter's order; one heading is a second chapter in outline |
| Pedagogy | 3 | Does not say what to read first, or for which purpose |
| Language | 4 | Compact and exact; a few loose comparatives |
| Ease and enjoyment | 3 | A reference page, used and not read; the entries on generalization are the exception |

## What works

- **The contract in the first paragraph**, and the cross-pointers that go with it: distribution matching and the pitfalls of FID are sent to :numref:`sec_gan-further-reading`, bridges and couplings to :numref:`sec_mdl-dynamics-further-reading`. Nothing is listed twice across the three pages that I could find.
- **Entries that say what changes relative to the chapter.** "DPM-Solver ... approximates the remaining integral of the predicted noise by a polynomial in $\log \bar\sigma$; its first-order version is DDIM". "for the absorbing kernel a ratio is a time factor times the clean conditional, what the masked denoiser learns". A reader of the chapter can place each paper at once.
- **"Evaluation, Memorization and Generalization."** The three entries on memorization and on why networks generalize are the only part of the page with an argument ("The exact optimum of the regression on a finite set reproduces training points, and ... the noisy target is not what prevents this"). This is the question a thoughtful reader has after `gaussian-paths.md`, whose exercise on memorization raises it, and the page answers with four papers in the right order.
- **Honest hedges where the evidence is a paper's**: "reported to converge several times faster", "it helps on some datasets and hurts on others".
- **"Post-Training by Reward and Preference"** connects the chapter to chapter 15 (:numref:`sec_rl_sequences`, :numref:`sec_regularized`) by label, the only place in the chapter that looks back past the GAN chapter into the part.

## Structural findings

1. **No ranking.** See FR-1.
2. **The headings do not follow the chapter.** Sections run denoising, paths, images, samplers, design space, latent, few-step, discrete. The headings run energy models (17.1), schedules and networks (17.5 and 17.3 mixed), samplers (17.4), guidance and editing (17.3 and 17.4), latent (17.6), few-step (17.7), likelihood (17.5 again), evaluation, video, discrete (17.8), post-training. "Likelihood" should sit with "Schedules, Weighting and Networks", and each heading should name its section once.
3. **"Discrete Diffusion and Diffusion Language Models" is a chapter in outline**: five entries, 22 citations, a quarter of the page, on the topic the chapter treats in its last section and the book's next part takes up. It is where the page drifts from "what a later work changes" to a survey of a field in motion (seven of its citations are dated 2025 or 2026). Halve it, keeping continuous time and ratios, revision, and the scaling comparison, and let the Language Models part own the systems (LLaDA, Dream, DiffuLLaMA, Fast-dLLM) when it exists.
4. **It will receive more.** Section reviewers propose to move here: the text-conditioning and video paragraphs of `latent-diffusion.md` (LAT-19, LAT-20), the guidance-distillation paragraph of `few-step.md`, the Chung et al. sentence of `samplers.md` (SMP-17), "Parameterization at Finite Capacity" from `design-space.md` (DSP-14), and three extended exercises (energy-model training, NCSN, autoguidance). The page already has entries for most of these. It has none for how a caption conditions a model (cross-attention, joint attention, the text encoders), which becomes the page's largest gap if LAT-19 is cut from the body.
5. **Missing for a 2026 reader.** (a) Diffusion outside images and text: molecules and proteins (one clause exists, under discrete flow matching), audio, and robot policies; one entry of three citations would tell the reader that the chapter's regression is used well beyond pictures. (b) How accurate the score must be for the sampler to be accurate (the sampling-error bounds), which the appendix review also found missing on its page; one of the two pages should carry it. (c) Guidance beyond the scale: guidance restricted to an interval of noise levels and guidance by a weaker model, both of which the exercises of `image-diffusion.md` use.

## Passage-level findings

**FR-1** (moderate, pedagogy). Opening paragraph.
Anchor: "Each entry states what a later work changes; nothing in the book depends on it."
Add a ranked start of five or six lines before the first heading, with a reason for each. A proposal, to be adjusted by the authors: for a second, longer derivation of the whole chapter, the monograph cited in the chapter introduction; for what to change first in a real training run, Karras et al. on the design space and its 2024 successor; for sampling in ten to twenty-five evaluations, DPM-Solver++; for what a current text-to-image system is, the rectified-flow transformer paper of Esser et al.; for few-step generation, the continuous-time consistency paper; for discrete data, MDLM. Everything else on the page is for the reader with a specific question, and the page should say so.

**FR-2** (minor, correctness: currency). Under "Schedules, Weighting and Networks".
Anchor: "The result, ADM, is the usual baseline."
It was the usual baseline until about 2023. Since then comparisons are made against transformer backbones (DiT and its flow-matching successor) and against the 2024 U-Net of Karras et al., which this page cites two entries earlier. "was the usual baseline until transformers replaced it (:numref:`sec_diffusion-latent`)".

**FR-3** (moderate, thought: body and further reading disagree). Under "Samplers".
Anchor: "Dividing by $\alpha$ removes the part of the probability-flow ODE that is linear in $\mathbf{x}$."
This entry states the idea of an exponential integrator correctly and connects it to the change of coordinates the chapter teaches. `samplers.md` says such solvers "take steps in $\log$" of the noise ratio, which confuses the method with its grid (SMP-8). The correct statement belongs in the body, where the coordinates are introduced, and this entry then shrinks to the two successors. The same entry is where the reader should be told the practical fact that the samplers section omits: the multistep members reach second order at one evaluation per step and are what is used at ten to twenty-five evaluations.

**FR-4** (moderate, thought: body and further reading disagree). Under "Discrete Diffusion and Diffusion Language Models", entry "Scale".
Anchor: "Masked diffusion follows autoregressive scaling laws with a constant-factor compute gap"
This sentence gives both halves of the comparison with autoregressive models (a constant-factor gap with fresh data, an advantage under repeated data), each with its source. `discrete-diffusion.md` cites the second paper for the first half only and concludes "so the case for diffusion is not a better likelihood" (DISC-12). Move this sentence into the comparison subsection that DISC-16 and structural finding 1 of that review ask for, and keep here only the systems. Both attributions are from memory of the papers and should be checked (unverified).

**FR-5** (minor, correctness). Same heading, entry "Uniform noise and revision".
Anchor: "relate them to a Gaussian process and distill them to few steps"
A Gaussian process is a distribution over functions, which is not what is meant. As I recall the paper, uniform-state discrete diffusion is obtained from a Gaussian diffusion by taking the argmax of the noised one-hot vector, which is what lets consistency distillation be carried over. Write "relate them to an underlying Gaussian diffusion". Fairly sure.

**FR-6** (minor, correctness; loose comparatives). Several entries.
Anchors: "shows that curating video data matters as much as the architecture"; "works best for rectified flow at every model size"; "turn any distillation scheme into training from scratch"; "and trains many times faster".
Four claims in a paper's own voice without the hedge the page uses elsewhere. "As much as" compares two things no experiment puts on one scale; "at every model size" and "any" are universal; "many times faster" is the paper's headline figure for one model on ImageNet. Use the page's own formula ("report that ...") or state the finding without the comparative: "show how much the curation of video data changes the result", "find a logit-normal distribution among the best in a large comparison", "train consistency-type models from scratch by self-distillation", "is reported to cut training time severalfold".

**FR-7** (minor, explanation). Under "Discrete Diffusion and Diffusion Language Models", entry "The time input".
Anchor: "show that sampling categories in low precision lowers the effective temperature and flatters reported generative perplexity"
The heading is "The time input" and this sentence is about numerical precision in a sampler. It is an important caution for anyone reading that literature's tables and deserves its own bold lead ("Evaluation") or a place under "Evaluation, Memorization and Generalization". The first sentence of the entry repeats what `discrete-diffusion.md` says in the body about the order-agnostic equivalence; one of the two is enough once the body states the result (the pattern note of that review).

**FR-8** (minor, language). Whole page.
Anchor: "ControlNet :cite:`Zhang.Rao.Agrawala.2023` instead trains a copy of the encoder on edges, depth or pose."
Several entries end with a system added by "instead" or a semicolon that belongs to a different idea (ControlNet is conditioning, not restoration; Stochastic-gradient Langevin dynamics is Bayesian inference for weights, not sampling from a score). Each is correct and each is in the wrong entry. Give ControlNet to "Guidance" and drop or move the other.

**FR-9** (minor, correctness; unverified). Whole page.
Six citations carry the year 2026 (`Zheng.Ma.Tong.ea.2026`, `Li.He.2026`, `Geng.Lu.Wu.ea.2026`, `Wu.Zhang.Xue.ea.2026`, `Massias.Bertrand.2026`, `Dieleman.2026b`). The descriptions of the first three match what I know of the preprints; I could not confirm venues or the last three. Check each before print, and consider whether a textbook page should rest a heading on work less than a year old: "Transformers on pixels" and "General flow maps" are the two entries most likely to read as dated by 2028.

## Code and experiments

None.

## Exercises

None, rightly. Three extended exercises from the sections are proposed for the further-reading notebook (:exref:`ex_denoising-persistent-chains`, :exref:`ex_gaussian-paths-ncsn`, :exref:`ex_image-diffusion-autoguidance`); the page has entries that match the first and could take pointers to all three.

## Still open from October

Not applicable; the page postdates that review.

## For the chapter review

- The page is the most accurate statement in the chapter of two things the body gets loosely (FR-3, FR-4). In both cases the fix is to move content from here into the body.
- It is the proposed destination for five passages cut from teaching sections (structural finding 4). After those moves it needs an entry on text conditioning and will be about 2,200 words; the ranking of FR-1 then matters more.
- It does not repeat the appendix's further reading, and the appendix page should drop its two entries on the design space and few-step generation, which this chapter teaches (FRD-1 in the appendix review).
- It gives the chapter its only backward link into the reinforcement learning chapters.
