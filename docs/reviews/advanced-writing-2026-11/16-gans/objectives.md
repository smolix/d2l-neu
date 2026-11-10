# Adversarial Objectives and Divergences

Source: `chapter_generative-adversarial-networks/objectives.md` (lines 1 to 707, slides not read). About 3,600 words of body prose, 1,200 words of exercises, 20 code cells counting both tabs. Stored outputs checked for PyTorch and JAX (`objectives.json`), the three plots rendered for both tabs where they differ, both figures rendered. The table, the coin, the Gaussian MMD formula and the exercise identities recomputed by hand or with NumPy.

## Verdict

The first half is very good and the second half does not answer the question the section asks. The coin worked under three losses, the Bayes-risk gap, the proposition that turns every classification loss into an f-divergence, and the experiment in which four critics trained on one problem report four different functions of the same $\lambda$ are original, correct and well taught. Then the section turns to its stated question, "does any of them keep a gradient when the supports separate?", and answers it with a closed-form plot and a schematic. The Wasserstein critic, which is the answer, gets about ninety words, its defining duality is used without being named, and no experiment trains one. The one training experiment starts from overlapping supports and reports a side effect, a hinge residual of a few thousandths of a nat, with an explanation the text asserts and the section's own exercise treats as open. The most important fix: make the experiment and the Wasserstein passage carry the section's question (OBJ-9, OBJ-12), and demote the hinge finding to what the run shows (OBJ-13).

## Scores

| Dimension | Score | Note |
|---|---|---|
| Clarity of explanation | 3 | Coin, gap and table are clear; seven new terms in 400 words; RKHS, conjugate and "witness" used undefined; the generator's losses appear only in code; $W_1$ is two definitions equated without a word |
| Clarity of thought and focus | 3 | The two-family frame is sound; the experiment answers a different question from the section's; an untested explanation is stated as fact |
| Pedagogy | 3 | Concrete case before abstraction in the first half; the second half is a list of metrics with cited rates |
| Language | 4 | Plain and exact; no mannerisms worth counting |
| Ease and enjoyment | 3 | The coin and the four links are a pleasure; the integral-probability-metric half reads as a survey |

## What works

- **The opening.** "Open almost any GAN codebase and its training script offers a menu of losses" and "Practitioners switch between them the way they switch optimizers, keeping whichever trains." A problem the reader has, in the reader's words.
- **"Two Outcomes by Hand".** One coin, three losses, three savings (0.131 nats, $\tfrac{1}{16}$, $\tfrac12$), each recognized as a known divergence. All numbers check. This is the concrete case before the abstraction that the model texts have, and the proposition that follows is then a statement the reader already believes.
- **The proof of the proposition**, in particular convexity through the tangent-line envelope. Short and complete.
- **The four-critics experiment** (`objectives-critic-links-1`). It is not a confirming cell: four networks trained on the same two mixtures produce the identity, a sigmoid, a step and a shifted line of the same $\lambda$, and the reader sees that "what the critic outputs" depends on the loss. The closing remark, that all four drift "because the objective constrains a critic only where samples land", is the right reading.
- **"Smoothness is the condition, not the family"**, with total variation as an integral probability metric that still saturates. A judgment that prevents the wrong generalization.
- **:numref:`fig_gan_critics_separated`.** The three critics on separated supports, with the slope at $q$ marked, carry the argument of "Optimal Critics on Separated Supports" in one look.
- **The last body paragraph's judgment**: "The choice among them matters less than two questions the losses leave open". This is the sentence the reader needs. It comes at the end; see structural finding 3.

## Structural findings

1. **The section's question is not tested.** The title of the last part is "Gradients Far from and Near the Data". "Far" is covered by a plot of three closed forms and a figure. "Near" gets the only training experiment. The sentence "A $W_1$ arm would need the Lipschitz penalty of :numref:`sec_gan_convergence`" concedes it, and :numref:`sec_gan_convergence` does not run a Wasserstein arm either (it discusses the one-centered penalty and leaves its dynamics to an exercise). So the chapter never trains the critic it presents as the answer to separated supports. Exercise 9 of this section ("Separated initializations") is the missing experiment, in one dimension and cheap. Put a reduced version in the body: a generator $\mathcal{N}(\mu, 1)$ started at $\mu_0 \in \{2, 10\}$ under the logistic game, a spectrally normalized hinge critic, and MMD at two bandwidths. This is a redesign, so it belongs on the list for the development machine; until then the prose should say that the claim rests on the closed forms.
2. **Two halves of unequal care.** The f-divergence half has a worked case, a theorem, a table and an experiment. The integral-probability-metric half states two definitions, cites three convergence rates and points to the appendix five times. The seam is before "## Integral Probability Metrics". Do not split the section; rebalance it by giving the Wasserstein critic the detail in OBJ-9 and cutting the estimation paragraph to what the argument needs (OBJ-10).
3. **The reader is not told what is used.** The chapter introduction says the adversarial term survives in distilled one-step generators, image autoencoders and vocoders. As far as I know those mostly train with the hinge loss (VQGAN-style autoencoders, adversarial diffusion distillation), least squares (HiFi-GAN vocoders) and the non-saturating logistic loss with a gradient penalty (the StyleGAN line); I am confident of these but have not rechecked them. The section's only statement about the hinge is that it "stalls". One paragraph of practice, with the section's closing judgment moved up beside it, would tell the reader what matters and what can be skipped.
4. **"Convexity in the Generator's Distribution" comes too early.** It sits before any example and is used again only in the Summary and exercise 1. Move it after the table, or reduce it to two sentences where the five-loss experiment needs it.
5. **Squared hinge exists only as a control.** It is introduced in the table so that the experiment can isolate the link. That is good experimental design, but it adds a fourth row and a fifth arm for a finding the section should not lean on (OBJ-13).
6. **Appendix use.** The f-GAN bound and $\mathrm{MMD}^2$ are stated with the result in the text and the proof in the appendix, which is the author's rule. The Bayes-risk proposition is specific to this chapter and rightly lives here. All five references resolve to the right passages. Two things do not follow the rule: the Wasserstein distance cannot be understood from the main text alone (OBJ-9), and the appendix says of Kantorovich-Rubinstein duality that ":numref:`sec_gan_objectives` trains such a critic", which this section does not do.

## Passage-level findings

**OBJ-1** (moderate, explanation). Under "The Template" and "The Bayes-Risk Gap".
Anchor: "Let $\ell$ be a payoff on critic outputs, so that $-\ell$ is a loss."
Within forty lines the same game is written with two scoring functions $\ell_p$, $\ell_q$ that are subtracted, then with one payoff $\ell$ applied to $D$ and $-D$ and added, and the Bayes risk is defined on $-\ell$. The table's first column is headed "loss". A reader tracking signs has three conventions. Work with losses in this half (the critic minimizes), or say once how $\ell_p$, $\ell_q$ and $\ell$ are related and keep one.

**OBJ-2** (moderate, language). Same two subsections.
Anchor: "is its *margin*, and $-\ell$ is a *margin loss*"
Seven terms are introduced in about 400 words: payoff, margin, margin loss, conditional Bayes risk, Bayes-risk gap, divergence function, link. "Margin loss" is not used again. "Link" is used before it is explained ("each loss reports it through its own *link*") and is borrowed from generalized linear models, which Basics does not name. Drop "margin loss"; define the link in a clause: "the function of $\lambda$ that the optimal critic outputs".

**OBJ-3** (moderate, explanation). Under "Divergences of Common Losses".
Anchor: "The square row scores a probability against zero--one targets."
The framework scores $D$ on real and $-D$ on generated samples. The reader cannot see how squared error on a probability fits it, and the code uses `(r - 1) ** 2 + f ** 2` on raw scores. The squared hinge row uses $\pm 1$ targets, so the two rows differ by a change of variable that the text mentions only through its consequence ("four times the square row's"). Give the map $t = 2D - 1$ once.

**OBJ-4** (minor, correctness). Proposition.
Anchor: "**Proposition.** *For any loss with concave Bayes risk $L$,*"
The text has just shown that $L$ is concave for every $\ell$, so the hypothesis excludes nothing. What the proof needs is that $L$ is finite on $[0, 1]$. State that.

**OBJ-5** (moderate, pedagogy). Under "Critic Links".
Anchor: "Fenchel duality builds a game for any convex divergence function $f$."
The subsection headed "Critic Links" opens with a different topic, the f-GAN bound, and reaches the links experiment a paragraph later. Give the bound its own short heading. The conjugate is given by its formula alone; a reader from Basics has not met it. One sentence of geometry (the tangent line of slope $t$ lies below $f$, and $-f^*(t)$ is its intercept) makes :eqref:`eq_gan_fgan_bound` a statement about tangents and keeps the main text readable without the appendix.

**OBJ-6** (moderate, explanation). Before `objectives-loss-pairs`.
Anchor: "Each loss is stored as a pair: the critic's loss on the scores of real and generated samples, and the generator's loss that"
The generator's losses appear only as lambdas: `softplus(-f)` for the logistic game, `(f - 1) ** 2` for least squares, `-f` for both hinges. The prose never says what the generator minimizes in the least-squares and hinge games, nor that none of the three is the minimax loss of the game whose divergence the table lists. The later sentence "Every divergence in :numref:`tab_gan_loss_divergences` vanishes at $q = p$, so with overlapping supports every loss should drive a generator to the data" depends on this. Add two sentences, or a column "generator loss used in practice".

**OBJ-7** (moderate, explanation). Under "Maximum Mean Discrepancy".
Anchor: "Taking $\mathcal{F}$ to be the unit ball of a reproducing kernel Hilbert space with kernel $k$ gives the maximum mean discrepancy (MMD)."
The reader of Basics has never met a reproducing kernel Hilbert space, and the sentence defines MMD through it. The usable content is the next display: within-sample similarity minus twice the cross-sample similarity. Start from that ("MMD asks whether points of one sample are as similar to the other sample as to their own"), then say in a clause which function class it is the metric of. "Witness" is used two subsections later and in :numref:`fig_gan_critics_separated` without a definition.

**OBJ-8** (minor, correctness). Under "Gradients Far from and Near the Data".
Anchor: "the divergence stays at its ceiling, and the generator's gradient vanishes"
For KL and $\chi^2$ the value on disjoint supports is infinite, so there is no ceiling and the gradient is undefined. "The divergence is the same constant, finite or infinite, at every separation."

**OBJ-9** (major, correctness and explanation). Under "Wasserstein-1".
Anchor: "Taking $\mathcal{F}$ to be the 1-Lipschitz functions gives the Wasserstein-1 distance $W_1$, the least average distance over which the mass of $p$ must be moved to turn it into $q$"
Three problems in the passage that carries the section's answer.
(a) The sentence equates a supremum over Lipschitz functions with an infimum over transport plans as if by definition. That equality is the Kantorovich-Rubinstein theorem. It is named in an exercise of the previous section and in the appendix, never here, and its hypotheses (finite first moments; distances measured in the norm on $\mathcal{X}$, which for images is pixel distance) are not stated. The pixel-distance point matters: it is why $W_1$ gives a direction and also why it is a crude notion of image similarity.
(b) Anchor: "enforces the constraint approximately". The reader is not told what each device enforces, and they are not interchangeable. Weight clipping bounds the Lipschitz constant by an unknown $K$ that depends on the architecture and the clip value, so the critic estimates at best $K \cdot W_1$ over a reduced function class. The gradient penalty of WGAN-GP pushes $\|\nabla_x T\|$ toward one, softly, and only at points on lines between real and generated samples. Spectral normalization bounds every layer's largest singular value by one, a sufficient condition that excludes most 1-Lipschitz functions. The appendix says the consequence in one sentence ("What an approximately Lipschitz critic reports is no longer exactly $W_1$"); the main text should say it too, since this is what a reader implementing a WGAN most needs to know. About 150 words.
(c) No shapes, no cost, no pitfall: nothing a reader could implement from. Either add the experiment of structural finding 1 or say that the chapter does not train a Wasserstein critic and why (the zero-centered penalties of :numref:`sec_gan_convergence` replaced it).

**OBJ-10** (moderate, thought). Under "Estimation from Samples".
Anchor: "so halving its error can take $2^{d_x}$ times as many samples"
Three rates are cited and none is shown; exercise 7 does the showing. As written, the paragraph tells the reader that $W_1$ cannot be estimated in high dimension, one page after presenting it as the remedy, and does not connect the two. Either connect them in a sentence (a WGAN critic is a small parametric family, so it does not estimate $W_1$ between the empirical measures; it finds a smooth direction) or cut the paragraph to the forward-KL variance remark, which the section uses.

**OBJ-11** (minor, correctness of reading). Under "Divergence against Separation".
Anchor: "peaks at a separation set by the bandwidth and then decays like a Gaussian in $\mu$"
True of :eqref:`eq_gan_mmd_gauss` (the peak is at $\sqrt{s^2 + 2} \approx 4.2$), but the plot stops at $\mu = 8$, where the MMD slope has only halved (0.27 to 0.14). The decay the sentence describes is not on the page. Plot to $\mu = 20$ or use $s = 1.5$. The Jensen-Shannon numbers match the stored output (slope 0.1987 and 0.0013, ratio 0.9990).

**OBJ-12** (major, pedagogy). Under "Five Losses on a Gaussian Mixture".
Anchor: "A $W_1$ arm would need the Lipschitz penalty of :numref:`sec_gan_convergence`."
See structural finding 1. The experiment that closes a section about separated supports starts every arm from overlapping supports ("each mean lies one to two and a half units from a different mode") and leaves out the one objective the section recommends for separation.

**OBJ-13** (major, correctness: experiment read for more than it shows). Same subsection.
Anchor: "The hinge stall comes from what its critic must track."
Stored medians: hinge 0.0070 (PyTorch) and 0.0059 (JAX); the others 0.0008 to 0.0013; MMD 0.0001. So "at about $0.005$ nats" should read "between 0.005 and 0.01", and "usually several times above the rest" is supported. What is not supported is the explanation. It is one seeded run per framework; the traces move by a factor of three to five from one record to the next (in the JAX plot the logistic and least-squares traces rise to about 0.003 near the end, inside the hinge's band). The lag account that follows is a hypothesis, and exercise 6b of this section asks the reader to find out whether "the outcome point[s] to minibatch noise or to lag". A text cannot assert in the body what it poses as open in the exercises. And the effect is six thousandths of a nat on a planar toy, for the loss that much of practice uses; the Summary nevertheless ends on it. Fix: either run the control in the body (five critic steps per generator step; if the residual falls, the lag account stands and the paragraph is earned) or report the residual as an observation with the lag account marked as the likely cause, and take it out of the Summary's last sentence.

**OBJ-14** (minor, thought). Same subsection.
Anchor: "The logistic and least-squares runs also stay on their initial plateau longer than the other three."
Visible in both stored plots, and left unexplained; exercise 6a holds the explanation (the non-saturating weight against the constant weight of $-D$). Give it in a clause or remove the observation.

**OBJ-15** (minor, pedagogy). End of body.
Anchor: "With overlapping supports every loss reaches the data"
The last body paragraph restates the result and the Summary restates it again two paragraphs later. Keep the judgment and the handover; let the Summary do the restating.

**OBJ-16** (minor, pedagogy). :numref:`fig_gan_template`.
Anchor: "Two choices define the classical adversarial objectives: the scoring functions and the critic class."
Rendered: a tilted plane with three dots and an annotation "off the plane". It restates the sentence above it and places nothing the reader can read off. A two-by-two table (scores: nonlinear or linear; critic: free or constrained; one example per cell, with the hinge on a constrained critic in the fourth) would say more, and would show that practice mixes the families (exercise 5).

## Code and experiments

- `objectives-js-by-quadrature`: 0.033 nats, "a few hundredths". Correct.
- `objectives-critic-links-1`: teaches (see "What works"). The hinge panel shows a wide scatter around the step, which the prose notes.
- `objectives-divergence-against-separation`: see OBJ-11.
- `objectives-five-losses-on-a-gaussian-mixture-1`: 0.529 nats, "about half a nat". Correct.
- `objectives-one-testbed-four-losses-6`: see OBJ-13; "about $0.005$" does not match 0.0070 and 0.0059.
- `objectives-one-testbed-four-losses-3` (PyTorch): `TRUNC_STD = 0.8796` and eight lines exist to reproduce the JAX default initialization. This is plumbing that serves the build, not the reader; hide it in a saved helper or accept different initializations.
- `mmd2_unbiased` is saved for :numref:`sec_dcgan`; fine.
- All five arms share learning rates and steps "with no separate tuning". For a comparison whose conclusion is a ranking of losses this is the untuned-baseline problem; it is a further reason not to build on the hinge result.
- No tab prose. No TensorFlow or MXNet material.

## Exercises

Nine problems, about a third of the body; the algebra of 2, 3, 4, 5 and 8 checks, and $D_{\textrm{KL}}(p \,\|\, q) \approx 0.16$ nats in 7c is right (0.163 by quadrature; the "85 to 95 percent" is unverified). Quality is high. Exercise 5 (the unconstrained hinge game is the integral probability metric of the sup-norm ball; with a Lipschitz critic it gives a gradient over a bounded range) is the bridge between the section's two families and explains the practice of hinge with spectral normalization. Its conclusion deserves two sentences in the body. Exercise 9 is the experiment the body lacks (OBJ-12). In exercise 2a, drop "Confirm the factor $\tfrac18$ on Dirichlet draws": it confirms by sampling what the first half of the problem proves. The claim in exercise 5 about scores under spectral normalization and the corollary number in exercise 8 are unverified citations.

## Still open from October

"Recall from `eq_mdl-f-div-def`" is gone: the f-divergence is now defined in the text. "Buried under divergence algebra" is half open: the first half of this section now earns its algebra with the coin; the second half is where a reader loses the thread.

## For the chapter review

- This is the first of three sections with no image and no neural generator. Its testbed (`train_toy`, the three-mode mixture, exact $\mathrm{JS}$ after every step) is a good instrument that the next section could reuse and does not.
- It hands :numref:`sec_gan_convergence` the Wasserstein critic without having trained one, and that section does not train one either. Decide at chapter level whether the book teaches WGAN or cites it; at present it is presented as the answer and then set aside.
- Overlap with the appendix is within the author's rule. The appendix's table `tab_mdl-divergence-objectives` and this section's table are complementary. Fix the appendix sentence that says this section trains a Lipschitz critic.
- Pattern, shared with 16.1 and 16.3: a result or a test that the body needs is placed in an exercise (here the hinge as a metric, the lag control, the separated start).
- "Convex in $q$, not in the parameters" is explained here and again in :numref:`sec_gan_relativistic`; one owner.
