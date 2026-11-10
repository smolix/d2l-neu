# Diffusion Models (chapter introduction)

Source: `chapter_diffusion-models/index.md`, read to the slides marker. About 330 words of introduction in four paragraphs and a resources list of ten annotated entries (about 300 words). No code, no figure. The labels cited resolve. The closing sentence was compared with the last body paragraph of `discrete-diffusion.md`, and the sentence about the appendix with the three passages the section reviewers found that cannot be read without it (GP-1, GP-3, DSP-1).

## Verdict

A good first paragraph, a compressed second, a list for a third, and a fourth that promises more than the chapter delivers. The opening is the best chapter opening of the part's last two chapters: it takes the reader from the four limits that closed the GAN chapter to one question, "Regression onto what, then, and how can fitting a target produce something new?", which is the question a reader who has only trained classifiers would ask. After that the introduction stops leading. The route is eight questions in a row, each followed by a label, with nothing on which sections carry the chapter, which can be skipped, what the chapter adopts as its default model, what its two running examples are, or what the experiments cost. The last paragraph tells the reader that "The mathematics lives in" the appendix, which is the opposite of the author's rule (the main text reads without the appendix, which is optional deeper reading), and it claims four answers of which one, "a transformer trained stably at scale", is not shown by anything the chapter runs. The most important fix is to replace the list by a route with a judgment in it (IDX-3) and to restate the contract with the appendix (IDX-5).

## Scores

| Dimension | Score | Note |
|---|---|---|
| Clarity of explanation | 4 | The question and the common object are clear; the physics sentence is too compressed to help |
| Clarity of thought and focus | 3 | The route is a list; one of four closing claims is not supported; the appendix contract is stated the wrong way round |
| Pedagogy | 3 | No prerequisites, no running examples, no default model, no compute note, nothing on what to skip |
| Language | 4 | Plain and exact; the third paragraph is eight interrogatives of one shape |
| Ease and enjoyment | 3 | The first paragraph invites; the third reads as a table of contents |

## What works

- **The handover.** "that chapter closed with four limits the game leaves in place: no density, two networks and a tuned penalty, no cost for a missing mode, and instability at scale." The reader arrives with a list of complaints and is told this chapter will answer them. Keep it, subject to IDX-6.
- **The question.** "A classifier regresses onto a given label; a generator has nothing to copy, and its image must come out of noise. Regression onto what, then, and how can fitting a target produce something new?" Two sentences, and the reader wants the chapter. This is the model for a chapter opening in this part.
- **One object behind three histories.** "All routes arrive at one network that predicts the clean point from a noisy one at every noise level, and a sampler that integrates its predictions." The reader is told before the first section that the three literatures are one model. `gaussian-paths.md` then repeats this as its own opening (GP-13); it should be said once, here.
- **The resources list** says what each entry is for and ties six of the ten to a section by label. "derives the samplers of :numref:`sec_diffusion-samplers` without stochastic calculus" tells a reader exactly when to open that paper.

## Structural findings

1. **The route is eight questions.** See IDX-3. The questions are the sections' real questions, which is better than eight "asks how" sentences, but eight interrogatives of equal weight are a table of contents. The chapter has a shape that the list hides: two sections that hold the whole model (17.1, 17.2), one that moves it to images and builds the measuring instruments (17.3), four that each change one thing and can be read in any order after 17.3 (samplers, objective, representation and backbone, number of steps), and one that leaves continuous data (17.8).
2. **The chapter's default is not stated.** The chapter teaches the velocity on the linear path as its first model and DDPM as its second, and shows that both hold the same posterior mean. That is an editorial decision of 2026 and a good one. The introduction gives the two histories at equal weight and never says which the chapter builds on. See IDX-4.
3. **The running examples and the cost are missing from the text and present in the slides.** The slide deck below the marker has "A Mixture Where Everything Is Exact, Then Garments" and "What the Chapter Measures". The prose has neither. A reader should learn here that the first two sections run on a planar mixture whose density, score and posterior mean are known exactly, and that from 17.3 on the notebooks need a GPU (section reviewers found no stated total cost in `image-diffusion.md`, `latent-diffusion.md` or `few-step.md`; IMG-6 counts more network passes in evaluation than in training).
4. **No prerequisites.** The chapter needs: squared error is minimized by a conditional mean (recalled from :numref:`sec_gan_beyond`), the moving average of weights, the Fréchet distance and KID from :numref:`sec_dcgan`, the feature network of the utilities chapter, `ViTBlock` and `PatchEmbedding` from :numref:`sec_vision-transformer`, positional encodings, the Gaussian KL formula from the GAN chapter, and the masked-token objective of chapter 11. One sentence.
5. **No notation.** Two section reviewers asked for a short notation table at the head of the chapter (the pattern note in `gaussian-paths.md`; DISC-4). The clock ($t = 0$ data, $t = 1$ noise), $\alpha_t$, $\sigma_t$, $\bar\alpha_t = \alpha_t^2$, the log-SNR $\lambda$, and the ratio $\bar\sigma = \sigma_t/\alpha_t$ are used in seven sections. Five lines here, or a pointer to the callout in :numref:`sec_diffusion-gaussian-paths`, would carry the reader. The slides have this too ("One Clock").
6. **Nothing on how the chapter ends, and the chapter ends the part.** See IDX-6.

## Passage-level findings

**IDX-1** (minor, explanation). Second paragraph.
Anchor: "structure such as a drop of ink in water is destroyed by many small random steps, and a small step's reversal has the same simple form"
The ink drop is a good image, and the clause after it is the idea that makes diffusion models possible, compressed past the point where a first-time reader can use it. "The same simple form" as what? Say it: "each small step adds a little Gaussian noise, and when the step is small enough the step back is again approximately Gaussian, so undoing it means learning only a mean". That one sentence prepares :numref:`sec_diffusion-gaussian-paths` and costs twenty words.

**IDX-2** (minor, correctness; unverified in detail). Second paragraph.
Anchor: "made the undoing a regression onto the added noise and matched the sample quality of adversarial training"
As I recall the paper, Ho et al. reported a CIFAR-10 FID of 3.17, better than most GANs of the time and behind the best conditional ones; that diffusion surpassed GANs on ImageNet is the claim of Dhariwal and Nichol a year later. "came close to the sample quality of the best GANs on small images" is safe. Fairly sure; check against the paper.

**IDX-3** (moderate, pedagogy). Third paragraph.
Anchor: "What does a network that removes noise know about the data (:numref:`sec_diffusion-denoising`)?"
Eight questions, eight labels, one paragraph. Replace by a route with a voice that says what matters and what can be skipped. Draft: "The first two sections hold the whole model. :numref:`sec_diffusion-denoising` shows on three Gaussians in the plane, where every quantity is known exactly, that a denoiser is a score and that one noise level is not enough. :numref:`sec_diffusion-gaussian-paths` connects all noise levels by a path, trains one network along it by one regression, and samples by integrating. A reader who stops there has understood diffusion models. :numref:`sec_diffusion-images` moves the model to images and builds the instruments (a score for sample quality, read against a reference) that every later section uses. The next four sections each change one thing and can be read in any order: the sampler (:numref:`sec_diffusion-samplers`), the objective (:numref:`sec_diffusion-design-space`), the representation and the network (:numref:`sec_diffusion-latent`), and the number of steps (:numref:`sec_diffusion-few-step`). :numref:`sec_diffusion-discrete` asks what remains of all this when the data are tokens. Of the four, the builder needs the first and the third; the second is for the reader who wants to know why this loss."

**IDX-4** (moderate, thought). Whole introduction.
Anchor: "All routes arrive at one network that predicts the clean point from a noisy one at every noise level"
Add, after this sentence, what the chapter chooses and why: "We train a velocity on a straight path between data and noise, the form in which most systems since 2024 are trained, and meet the 2020 noise predictor as a second target on a second path that holds the same information." Then say what the notebooks can and cannot show, as the GAN review asked of that chapter: the planar experiments are exact and their conclusions are safe; the image experiments are single runs of a 1.2-million-parameter network on $28 \times 28$ garments, scored by a Fréchet distance that resolves differences of a few tenths at best, and several claims of the later sections (recipes, backbones, grids) are about effects of that size.

**IDX-5** (moderate, thought). Fourth paragraph.
Anchor: "The mathematics lives in :numref:`chap_mdl-dynamics`, to which the chapter points instead of repeating it"
This tells the reader that the mathematics is elsewhere and that the appendix is needed to read the chapter. The author's rule is the reverse: small results appear in both places, long derivations go to the appendix, and the main text reads without it. The sentence is also not true of the chapter, to its credit: Tweedie's formula, the DDIM family, the variational bound, the weighting table and both propositions on masked diffusion are derived in the chapter. Rewrite: "The chapter states each result with the reason for it and derives what is specific to diffusion models. The longer proofs (the continuity equation, time reversal, the convergence of Langevin dynamics) are in :numref:`chap_mdl-dynamics`, which is optional reading." For this to be true, three passages need a few sentences each: GP-1 (why the average velocity transports the density), GP-3 (stochastic differential equation notation used without introduction) and DSP-1 (the evidence lower bound cited, not shown).

**IDX-6** (moderate, thought). Fourth paragraph.
Anchor: "a loss under which every missing mode costs, and a transformer trained stably at scale"
Three problems. (a) The fourth answer is not shown by the chapter. `latent-diffusion.md` trains a transformer at three sizes on $28 \times 28$ images, the last step of which is inside the run-to-run range (LAT-14, LAT-15); stability "at scale" is a claim about systems the chapter cites. Write "and networks, convolutional or transformer, that train by plain regression without the balancing a game needs", which the chapter does show in every section. (b) The fourth limit itself ("instability at scale") was found by the GAN review to rest on one 2026 preprint (ADV-14 there); if that close changes, this paragraph and the first must follow. (c) The same sentence, in the past tense, is the last body sentence of the chapter ("Over the chapter the four limits of the game of :numref:`chap_gans` found their answers", `discrete-diffusion.md`; DISC-16). A promise and its repetition are not a close. Keep the promise here, and let the end of the chapter say what the answers cost: a thousand evaluations, then fifty, then a second training run to get to four; a likelihood that the weighting gives up for sample quality; an autoencoder that bounds quality.

**IDX-7** (minor, thought; dependency). Fourth paragraph.
Anchor: "image generation at scale, text-to-image and video belong to a later part of this book"
An unlabelled forward reference. `latent-diffusion.md` repeats it ("Systems at full scale are the subject of a later part of this book", LAT-2). The Image Models part is at present the former computer vision chapter, and I found nothing in the tree that teaches text-to-image or video. Cite a label when one exists; until then say "are outside this chapter; :numref:`sec_diffusion-further-reading` gives entry points".

**IDX-8** (minor, pedagogy). Resources.
Anchor: "These resources accompany particular sections; :numref:`sec_diffusion-further-reading` turns to the methods the chapter does not teach."
The division of labour with the further-reading section is clear and should stay. The list of ten is unranked. Say which one to open first and for whom: for the reader who wants the same material at greater length with proofs, the monograph of Lai et al.; for the reader who wants to code along, the MIT course; for the reader who wants pictures, the interactive article of Gao et al. Three clauses.

**IDX-9** (minor, correctness; unverified). Resources.
Anchor: "and that the weighting matters more than the schedule"
This is the thesis of :numref:`sec_diffusion-design-space` attributed to a blog article. As I recall the article, it shows that for a fixed weighting the training noise schedule changes only the variance of the loss estimate; "matters more" is the chapter's gloss. The chapter's own experiment does not establish it either (DSP-6, DSP-12). Write "and that the training noise schedule acts only through the weighting of noise levels". The two 2026 entries of Dieleman's blog and the publication years in three citation keys (the cited arXiv numbers are from 2024 and 2025) should be checked before print.

## Code and experiments

None in this file.

## Exercises

None, rightly.

## Still open from October

The October review read an earlier chapter with different files. Its request that the chapter state results and the appendix prove them is what the fourth paragraph announces; see IDX-5 for the wording.

## For the chapter review

- The introduction and the last body paragraph of the chapter say the same sentence (IDX-6, DISC-16). The chapter has an opening promise and no close.
- The sentence on the appendix (IDX-5) sets the wrong expectation for the whole chapter and should be rewritten together with GP-1, GP-3 and DSP-1.
- The slide deck under the marker contains three things the prose lacks: the running examples, a table of what the chapter measures, and the notation of the clock. Each is two or three sentences of prose.
- The introduction is the place for the one statement about the resolution of the Fréchet distance that three section reviewers asked for (the "For the chapter review" notes of `samplers.md` and `design-space.md`; pattern 1 of `latent-diffusion.md`).
