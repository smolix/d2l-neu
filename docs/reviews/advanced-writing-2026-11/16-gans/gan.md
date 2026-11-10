# Generative Adversarial Networks

Source: `chapter_generative-adversarial-networks/gan.md` (lines 1 to 958, slides not read). About 4,200 words of body prose (300 of them in a notation table), 1,300 words of exercises, 35 code cells counting both tabs. Stored outputs checked for PyTorch and JAX (`gan.json`), all five plots and the sample grid rendered, the three figures rendered. Worked numbers recomputed with NumPy.

## Verdict

This is the best of the chapter's three opening sections and a good textbook section by any standard. It has one question (what does the game measure, and does the generator get a usable signal), a motivation the reader feels, a derivation with no skipped step, a figure that carries the saturation argument, and experiments that mostly show what the prose says. I found no mathematical error: the optimal critic, the Jensen-Shannon identity, the mutual-information reading, both gradient weights, the non-saturating value and all quoted numbers check. What keeps it from the bar is local. A 20-row notation table for the whole chapter interrupts the first section before any mathematics has happened. The passage on what the non-saturating generator minimizes hands its key fact to a two-dot exercise and then argues mode dropping from "the first term" of a difference. And the garment experiment is read as "not yet at equilibrium at this budget", which suggests that more epochs would close the gap; the section's own last exercise and the opening of :numref:`sec_gan_convergence` say otherwise. The single most important fix is the non-saturating passage (GAN-5, GAN-6): state in the body that the generator's divergence is $2 D_{\textrm{KL}}(m \,\|\, p)$, and argue from that.

## Scores

| Dimension | Score | Note |
|---|---|---|
| Clarity of explanation | 4 | Optimal critic, value and weights are derived step by step with a Gaussian worked by hand; "The Same Fixed Point" runs two different calculations in eight lines without saying why |
| Clarity of thought and focus | 4 | One question, answered; the mode-dropping argument is weaker than the fact it rests on; the garment reading overreaches |
| Pedagogy | 4 | Motivation, prediction and test are in the right order for the Gaussian; the notation table and the position of the garments blunt it |
| Language | 4 | Plain, exact, almost free of mannerisms; some sentences with voice |
| Ease and enjoyment | 3 | A pleasure up to mode dropping; after that 500 lines of two-tab code and three checks in a row |

## What works

- **The opening problem and the noise mixture.** "Nor is likelihood what we want from a sampler" followed by the 1%/99% mixture, the 4.6 nats, and "Judging a sampler requires looking at its samples." This is the memorable example the guide asks for (section 10.2, item 3). Protect it.
- **The link back to Basics.** The covariate-shift classifier of :numref:`subsec_covariate-shift-correction` is exactly the right prior knowledge, and the reference points to a passage that really does read a log ratio off a logit ($2x_1 - 2$ there).
- **The scalar reduction** under "The Optimal Critic": "depends on $D$ only through the number $D(x)$", then one concave problem in $u$. Followed at once by two unit Gaussians worked by hand, with the remark that a logistic regression contains the optimal critic.
- **The information reading** of :eqref:`eq_gan_entropy_gap`: "the number of nats that one sample provides about its source", and "A binary label carries at most $\log 2$ nats". This sentence makes the ceiling obvious and is reused well in the next two sections.
- **"The case is not exotic."** A short sentence that tells the reader the point masses are the typical case, not a curiosity.
- **:numref:`fig_gan_weights`** with $e^3 \approx 20$ and $e^5 \approx 148$ marked on it. The figure does the work of a paragraph.
- **Candour at the end of the saturation experiment**: "The non-saturating weight amplifies the factor $\nabla_\theta D$ but cannot create it." It says what the experiment did not show and sets up the next section in the argument, not in a recap.
- **The fit measured in nats.** A linear generator whose density is known lets the section measure what the loss cannot. The remark that "a critic loss of $\log 2$ shows only that this critic can no longer separate the samples, not that $q = p$" is the right warning at the right place.

## Structural findings

1. **The chapter's notation table sits in the middle of this section's motivation.** Twenty rows, fourteen of them for symbols of later sections ($\Phi$, $\mathcal{E}_{\mathrm{diag}}$, $\nu_{1,2}$, $\zeta_c$), with a "papers write" column, between the picture of the game and its first equation. It is material written for the authors (guide 4.5) and it contains errors (GAN-2). Move it to the chapter introduction or to the end of the chapter, and introduce this section's six symbols where they are used, which the text already does.
2. **Order of the two experiments.** The opening promises garments, and the garments come before the Gaussian that tests the analysis. The result is that the derived predictions (critic equals $\lambda$, saturation) are tested 450 to 600 lines after they are derived, and the section ends on three checks instead of on its most interesting picture. Consider: theory, then the Gaussian (each prediction tested while fresh), then the garments as the finale. The garments' unsettled losses are what the Summary's last sentence and the openings of :numref:`sec_gan_objectives` and :numref:`sec_gan_convergence` lean on, so ending on them makes the handover natural.
3. **Not two sections.** The length is right for the chapter's foundation. What can go: the notation table (300 words), the paragraph on `make_mlp` initialization, and the first of the two calculations in "The Same Fixed Point" (GAN-5).
4. **Appendix use follows the author's rule.** The Jensen-Shannon identity and the optimal critic are short, specific to this chapter, and derived here; the Gaussian KL formula is quoted with a pointer. Nothing here needs the appendix to be followed.

## Passage-level findings

**GAN-1** (minor, pedagogy). Section opening.
Anchor: "Suppose we want a network that draws a Fashion-MNIST garment from a hundred random numbers in a single forward pass."
The chapter introduction opens on the same image ("a network that turns a hundred random numbers into an image in one forward pass") and then offers the Bradley-Terry reward model as the thing "we have met before", where this section offers the covariate-shift classifier. A reader meets the same opening twice within a page with two different antecedents. Keep the covariate-shift link here (it is the exact antecedent) and let the introduction carry only the question.

**GAN-2** (major, pedagogy). Under "Notation".
Anchor: ":numref:`tab_gan_notation` collects the chapter's symbols, each with one meaning throughout; letters defined inside an exercise are local to it."
See structural finding 1. Errors in the table: $w_{\textrm{Rp}}$ and $d(p, q)$ are listed as first used in this section, but appear first in :numref:`sec_gan_relativistic` and :numref:`sec_gan_objectives`; the "papers write" entries "$\eta$ for a step size, $h$ here" and "$\delta$ for a separation of means" cannot be decoded by a reader. Fix: move the table out of the section and repair those rows.

**GAN-3** (minor, language). Under "A Classifier between Real and Generated Samples".
Anchor: "We call it the *critic*, the name used for its successors later in the chapter, which score samples without classifying them."
A defensible choice, stated once. Three loose ends: :numref:`fig_gan` labels the box "discriminator" (rendered and checked) while its caption says critic; the code says `net_D`, `update_D`; and the reader has just spent a chapter with the *critic* of actor-critic methods. Change the figure label, and add a clause that this critic is unrelated to a value function.

**GAN-4** (minor, explanation). Under "The Value of the Game".
Anchor: "(:numref:`sec_mdl-divergences-distances` proves both bounds)"
The pointer is to a section of 2,400 source lines, and the sentence it sits in has just proved the upper bound itself ("A binary label carries at most $\log 2$ nats"); nonnegativity is the nonnegativity of mutual information or of KL. Drop the pointer, or point at the Jensen-Shannon definition `eq_mdl-js-def`, where the appendix states the bounds.

**GAN-5** (moderate, explanation). Under "The Same Fixed Point".
Anchor: "Changing the weight does not change the population minimizer against an optimal critic."
The subsection runs two calculations. The first re-optimizes the critic for every $q$ and bounds the resulting loss (:eqref:`eq_gan_ns_value`). The second, three sentences later, holds the critic fixed and names the divergence the gradient descends. The reader is not told that these are different objects, why both are needed, or which one describes training. Training holds the critic fixed during the generator's step, so the second is the one that matters. Lead with it, and reduce the first to a clause or an exercise.

**GAN-6** (moderate, thought). Same subsection and "Mode Dropping".
Anchor: "an $f$-divergence whose only minimizer is again $q = p$ (Exercise :exref:`ex_gan-generator-weight-divergences`)"
A difference of two divergences, $D_{\textrm{KL}}(q \,\|\, p) - 2\,\mathrm{JS}(p, q)$, is not visibly nonnegative, and the text supports the claim only by a two-dot exercise. The exercise's part (d) holds the fact that settles it: the expression equals $2\, D_{\textrm{KL}}(m \,\|\, p)$ (I checked the identity and the weight-to-$f$ correspondence $\rho^2 f''(\rho) = 1/(1+\rho)$). That is a three-line computation and a small result; by the author's duplication rule it belongs in the body. With it, nonnegativity and the unique minimizer are immediate, and the next subsection no longer has to argue from the anchor "That divergence inherits the asymmetry of its first term", which is not a valid inference for a difference. Draft: "Collecting terms, $D_{\textrm{KL}}(q \,\|\, p) - 2\,\mathrm{JS}(p, q) = 2\, D_{\textrm{KL}}(m \,\|\, p)$. This is a divergence, zero only at $q = p$. It is infinite when $q$ puts mass where the data have none and stays finite when $q$ misses mass the data have: for the half-covered mixture below it is 0.26 nats, where the forward KL is infinite." (0.26 is $\log 2 - 2 \cdot 0.216$; the text's 0.69 and 0.22 are correct.)

**GAN-7** (minor, correctness). Under "Mode Dropping".
Anchor: "Gradient steps rarely leave such a state"
"Rarely" is an empirical claim with no evidence here. What the sentence then shows is that at the best response the gradient on the covered mode is exactly zero. Say that, and say that a real critic is not at its best response, which is the subject of the next paragraph.

**GAN-8** (moderate, thought). Under "Mode Dropping".
Anchor: "Later in the chapter, a critic that ranks a generated sample against a real one replaces this threshold with a comparison."
This promises that ranking addresses the "second blindness". :numref:`sec_gan_relativistic` then concludes that ranking alone leaves the collapse minima in place and that only a fixed one-to-one pairing removes them, which minibatch training does not have. Temper the promise so that the reader is not set up for a result the chapter withdraws: "asks whether a critic that ranks ... helps".

**GAN-9** (moderate, correctness; confident about the general fact, not verified for this model). Under "Garments from Noise".
Anchor: "so at this budget the game has not reached its equilibrium, where both losses would sit at $\log 2$"
Stored outputs: critic 0.52 and 0.53, generator 1.37 and 1.35, matching "about 0.5" and "about 1.4". The reading is the problem. "At this budget" tells the reader that training longer would bring both losses to $\log 2$. For image GANs in general it does not: the critic stays ahead, here with 530,000 parameters against 60,000 images it sees every epoch (the section's exercise on finite samples says this) and with supports that barely overlap (the section's own "Separated Supports"). The opening of :numref:`sec_gan_convergence` describes the same run as losses that "never settled". Say here what is known: the gap is not evidence of too few epochs, a critic loss below $\log 2$ can mean a generator that is still wrong or a critic that has memorized, and the next three sections take up why. No one has run this model for 100 epochs in the notebook, so do not claim either way about this run.

**GAN-10** (minor, language). Under "The Game on a Gaussian".
Anchor: "three predictions can be checked exactly"
A network trained by SGD that reaches $R^2 \approx 0.95$ is not an exact check. "can be checked against known answers".

**GAN-11** (minor, correctness). Under "A Linear Generator with Known Densities".
Anchor: "apart from an occasional epoch in which both losses jump above it and fall back"
The stored PyTorch plot is smooth after epoch ten; no jump is visible. If this records repeated runs, say "in some runs".

**GAN-12** (moderate, pedagogy). Under "The Learned Density Ratio".
Anchor: "A network trained only to classify has learned the log density ratio"
The reader saw this demonstrated in Basics, on Gaussians, with the same conclusion. Under the author's rule on confirming cells this one is the allowed "once", but its opening sentence should say what is new. Two things are: $\lambda$ is quadratic here, so a network is needed, and the fit fails in the tails. The last two sentences ("The objective constrains the critic only where the mixture $m$ has appreciable mass, so the estimate is least accurate where the ratio is most extreme") are the part the rest of the chapter uses; lead with the question they answer. Also, "a constant critic has $R^2 = 0$" holds only for the constant equal to the mean of $\lambda$; with this definition the zero critic scores below zero.

**GAN-13** (minor, explanation). Under "Saturating and Non-Saturating Losses Compared".
Anchor: "Two arms start from the same initialization, see the same minibatches and latent draws, and differ only in the generator loss."
The generator's learning rate is 0.01 here and was 0.005 in the previous run; no reason is given. One clause, or use one value.

**GAN-14** (minor, thought). Under "The Update Rules".
Anchor: "a shorter momentum lets each player respond sooner to the other's latest move"
Stated as a mechanism; it is a plausible reading of an empirical habit, and :numref:`sec_gan_convergence` later uses $\beta_1 = 0$ without relating the two. Say "the usual explanation is" and point forward.

## Code and experiments

- `gan-garments-from-noise-2`, `-3`, `-4`: outputs match the prose (see GAN-9 for the reading). The 30 samples are recognizable and grainy, as described.
- `gan-quantifying-the-fit`: 7.50 and 7.86 nats at initialization, 0.010 and 0.001 after training. The prose ("about eight", "a few hundredths of a nat or less") is at the right precision. Note that most of the eight nats is the log-determinant of a nearly degenerate initial $q$, not distance.
- `gan-verifying-the-optimal-critic-2`: correlation 0.981 and 0.985, $R^2$ 0.962 and 0.953; prose says above 0.97 and above 0.9. Correct. The hand-set $q$ is at 3.41 nats as stated (recomputed).
- `gan-saturating-versus-non-saturating-training-1`: 0.02 against 338 nats; prose correct. The comparison is honest about being constructed ("needs a start at which they differ").
- Numbers no cell prints: the Jensen-Shannon values 0.11, 0.34, 0.63 and the four-digit agreement at $\mu = 8$ (recomputed: 0.111, 0.337, 0.633, 0.69305 against 0.69315; the last agrees to four decimals only after rounding), the mode-dropping 0.69 and 0.22 (correct), "more than three hundred nats" (338 printed).
- `update_G` (PyTorch) backpropagates through `net_D` and leaves gradients in the critic's parameters, which the next `trainer_D.zero_grad()` clears. A one-line comment would save a reader who reorders the loop.
- `make_mlp`: the sentence on $\pm 1/\sqrt{n_{\mathrm{in}}}$ initialization is detail the argument never uses; a docstring is enough.
- No tab prose; the two tabs do not fragment the text. No TensorFlow or MXNet material.

## Exercises

Ten problems, graded, about 30% of the body. The set is strong: unequal priors (3), supports of measure zero (4, which needs Kantorovich-Rubinstein duality from the appendix, fine for an exercise), the dead critic (5) and the classifier two-sample test (7) all test understanding, not recall. I checked the algebra of 2, 3, 4, 7 and 9. Exercise 9 is the best problem of the section and the body depends on its part (d): move that identity into the text (GAN-6) and keep the rest. Its attribution to a named homework problem is unverified. Nothing to cut.

## Still open from October

"99 percent of the mixture's samples are noise" was marked keep and is kept. The October remark that the density-ratio idea is "buried under divergence algebra" does not apply to this section; it applies to the two that follow.

## For the chapter review

- This section does what the chapter needs from it and trains an image generator, so the reader does get a picture early. The next image appears four sections later; see the notes in `objectives.md` and `relativistic.md`.
- It assumes logistic regression, the covariate-shift classifier, KL and entropy from Basics, and nothing from the appendix. Its cross-references into the appendix all resolve to passages that hold the cited content.
- The unsettled garment losses are retold in the openings of 16.2 and 16.4. One clause each is enough once this section says clearly what they mean (GAN-9).
- The promise about ranking (GAN-8) and the chapter's later verdict on ranking should be made consistent.
- Pattern to check across the chapter: a fact needed by the body is placed in an exercise and cited from the text (here exercise 9d and 2b).
