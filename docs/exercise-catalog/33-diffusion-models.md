# Chapter Overview — chapter_diffusion-models

The chapter (rebuilt 2026-11) has eight sections with exercises, 80 in all, ten per section, in the named and tagged style with `:exlabel:` ids. The sets are already much stronger than anything in the external tradition. Most problems are derivations or measurements on the chapter's own three-mode mixture and Fashion-MNIST models, with explicit success criteria. All sections except latent-diffusion (28%) exceed the ~30% guide of `tools/exercise_budget.py`: denoising 44%, gaussian-paths 43%, image-diffusion 44%, samplers 41%, design-space 41%, few-step 36%, discrete 59%. New proposals are therefore few. Each fills a gap the course tradition exposes, and the rewrites mostly remove duplication.

**Where the course homework is.** The richest verified homework comes from four sources:
- CMU 10-799 *Diffusion & Flow Matching* (Kelly He, Spring 2026), HW1–4 PDFs read in full. Its strongest device is the "debug your classmate" problem, which diagnoses a broken trainer from its loss curve and samples. The chapter has no counterpart.
- Berkeley CS180/280A Project 5 (Fall 2025), parts A and B read in full. It covers a single-level denoiser tested at other noise levels, denoising pure noise to the mean image, SDEdit, class- and text-conditional translation, RePaint inpainting, visual anagrams and hybrid images.
- Berkeley CS294-158 HW4 (Spring 2024, notebook read): a 2-D toy, a pixel U-Net, and a latent DiT with a VAE scale factor and CFG.
- MIT 6.S184 Labs 1–3 (IAP 2026, notebooks read): Langevin, conditional paths and score from flow, CFG, and a DiT plus VAE built from scratch.

**Textbook problems are drills.** CMU 10-423 HW2/HW4 (Spring 2026) and Prince's *UDL* ch. 18 Problems 18.1–18.12 (PDF read) pose forward-marginal and ELBO-algebra drills that the chapter's prose already carries; only UDL 18.12, guidance compared with truncation, adds an idea. Bishop & Bishop ch. 20 has exercises, but they could not be read, so they are not cited. Nakkiran et al., Chan, Lai et al., the FM Guide, Weng's and Dieleman's posts and the EDM paper pose no exercises.

**No homework tradition at all** (a finding, not a gap):
- EDM preconditioning, log-SNR weighting and bits per dimension through the probability-flow ODE (design-space).
- Flow maps, MeanFlow and consistency training (few-step; only 10-799's open "Speed" research track).
- Masked and discrete diffusion (6.S184 and 10-799 lecture on it, with no lab or homework).

**Overlaps.** The math appendix (ch. 29) shares material in five places, all noted per entry: Langevin bias, the loss floor, target conditioning, Hutchinson variance and prior mismatch. One duplication lies inside the chapter: the covariance of the exact reverse transition is derived both in `ex_gaussian-paths-gaussian-reversal`(a) and in `ex_samplers-optimal-variance`(a). Totals: 80 existing, keep 71, rewrite 8, drop 1 (merged); 9 new proposals.

---

## chapter_diffusion-models/denoising.md — Denoising and the Score

**Topic:** Energy models and their normalizer, denoising score matching (Vincent), Tweedie's formula, the loss floor, a single noise level's blindness to mode weights, Langevin sampling and its failure to mix, and a convolutional denoiser on Fashion-MNIST against the Wiener filter.

**Current exercises:** 10; disposition: keep 10, rewrite 0, drop 0. This is a strong set. It mixes exact calculations (two-point data, Laplace noise, data on a line) with code that measures the section's own Langevin runs, and it ends in an extended EBM-by-MCMC problem. That problem runs against 2026 practice (EBM training has left every practitioner course, per the course survey), but it is the only place where the reader sees why the section avoids maximum likelihood, so it stays. `ex_denoising-langevin-step-size` uses the stationary-variance formula derived in `ex_mdl-score-matching-diffusion-flow-langevin-stationarity` and cites it rather than re-deriving it, which is correct. `ex_denoising-escaping-mode` (Kramers) complements the appendix's `ex_mdl-score-matching-diffusion-flow-annealing` and does not duplicate it. The section's image experiment, a denoiser trained at σ = 0.5, has no exercise; that is the gap the external tradition fills.

**External sources found:**
- Berkeley CS180/280A, Project 5B §1.2.2 "Out-of-Distribution Testing", Fall 2025: a single-level MNIST denoiser trained at σ = 0.5 is evaluated at σ ∈ {0, …, 1.0} — https://cal-cs180.github.io/fa25/hw/proj5/partb.html
- Berkeley CS180/280A, Project 5B §1.2.3 "Denoising Pure Noise", Fall 2025: the same denoiser is trained with pure-noise inputs. Students explain why the outputs look like a centroid of the training images (the MSE minimizer). — same URL
- Berkeley CS180/280A, Project 5A §1.2–1.3, Fall 2025: Gaussian-blur denoising is compared with one-step denoising by a pretrained U-Net at three noise levels. Our Wiener-filter comparison is the quantitative version. — https://cal-cs180.github.io/fa25/hw/proj5/parta.html
- MIT 6.S184, Lab 1 Q3.1–3.2, IAP 2026: implement Langevin dynamics, and show that the OU process is Langevin dynamics for a Gaussian target. — https://github.com/eje24/iap-diffusion-labs (branch `2026`, `labs/lab_one.ipynb`)
- Cornell CS 5788 PS3, Spring 2026: written problems on Gaussian convolution and the DDPM closed form. Reported by the 2026 course survey; not re-verified here, so no URL is cited.
- Prince, *UDL* ch. 18: no problem on denoising or the score. Its problems start at the forward process.

**Proposed problem set** (11 problems):
1. [conceptual] **Learning without normalization** (`ex_denoising-learning-without-normalization`). Keep. *Provenance:* book-original.
1. [conceptual] **The log-normal weight** (`ex_denoising-log-normal-weight`). Keep. *Provenance:* book-original.
1. [conceptual] **Data on a line** (`ex_denoising-data-on-a-line`). Keep. *Provenance:* book-original.
1. [conceptual] **Two-point data** (`ex_denoising-regression-identity`). Keep. Its floor part overlaps lightly with `ex_gaussian-paths-loss-floor` and the appendix's `ex_mdl-score-matching-diffusion-flow-bayes-risk-floor`. This one is the exact one-dimensional case, which the other two use. *Provenance:* book-original.
1. [conceptual] **Bias of the smoothed score** (`ex_denoising-smoothed-score-bias`). Keep. *Provenance:* book-original.
1. [conceptual] **Laplace noise** (`ex_denoising-laplace-noise`). Keep. It is the best "which property makes Tweedie work" problem in the chapter. *Provenance:* book-original.
1. [short-code] **The best linear denoiser** (`ex_denoising-linear-denoiser`). Keep. *Provenance:* book-original.
1. [short-code] **Step size** (`ex_denoising-langevin-step-size`). Keep. *Provenance:* book-original (uses the appendix formula with a citation).
1. [short-code] **Escaping a mode** (`ex_denoising-escaping-mode`). Keep. *Provenance:* book-original.
1. [extended] **Training an energy by persistent chains** (`ex_denoising-persistent-chains`). Keep. *Provenance:* book-original.
1. [short-code] **A denoiser at the wrong noise level** (new). The reader evaluates the section's σ = 0.5 noise predictor on test images noised at σ′ ∈ {0.1, 0.25, 0.5, 0.75, 1.0}, denoising with x̃ − σ′ε̂. They plot the per-pixel error against σ′ for four denoisers: the network, the Wiener filter built for σ = 0.5, the Wiener filter recomputed at each σ′, and the identity.
    1. Where does the network beat the matched Wiener filter, and on which side of 0.5 does it over-smooth or under-smooth? Explain with Tweedie's formula why the optimal denoiser depends on σ, and name the input that would let one network serve every level. (This motivates the next section.)
    1. Retrain at σ = 10, where the input is nearly pure noise. Show that the output approaches the mean training image. Check that the per-pixel error is within a few percent of tr(Σ)/784, the error of predicting the mean. Connect the result to the one-step failure of the few-step section.
   *Provenance:* adapted from Berkeley CS180/280A Project 5B §1.2.2–1.2.3, Fall 2025 (overlap med: same two experiments, made quantitative against the Wiener filter and the tr Σ prediction; cite on adoption).

---

## chapter_diffusion-models/gaussian-paths.md — Gaussian Paths: Diffusion and Flow Matching

**Topic:** One Gaussian path from data to noise on a log-SNR clock; the chain view and annealing; one regression with four interchangeable targets (noise, clean point, score, velocity) and its loss floor; sampling by the probability-flow ODE (Euler) and by the reverse SDE (ancestral chain); experiments on the mixture.

**Current exercises:** 10; disposition: keep 8, rewrite 2, drop 0. The set is excellent and goes well past the course tradition, which stops at implementing α_t, β_t and the conditional path (6.S184 Lab 2) or at forward-marginal drills (UDL 18.1–18.3, CMU 10-423 HW2 6.3–6.5). Two rewrites:
- `ex_gaussian-paths-prediction-targets`(b) repeats the error propagation of the appendix's `ex_mdl-flow-matching-transport-target-conditioning` on another path. Shorten it to a pointer.
- `ex_gaussian-paths-loss-floor` gains one part from CMU 10-799's debugging problem: losses of different targets are not comparable.

**External sources found:**
- MIT 6.S184, Lab 2 Problems 2.1–2.4 and 3.1–3.3, IAP 2026: implement α_t and β_t, the Gaussian conditional path, the conditional vector field and score, and the CFM and CSM trainers; derive the marginal score from the marginal flow. — https://github.com/eje24/iap-diffusion-labs (branch `2026`, `labs/lab_two.ipynb`)
- Berkeley CS294-158, HW4 Q1, Spring 2024: continuous-time ε-prediction on a 2-D toy with the cosine schedule, an MLP conditioned by concatenating t, and the DDPM sampler written in DDIM form. — https://github.com/rll/deepul/blob/master/homeworks/hw4/hw4.ipynb
- CMU 10-799, HW2 Q2 and Q3(a), Spring 2026: implement flow matching on CelebA-64. Q3(a) asks whether a converged flow-matching loss far above the DDPM loss of the same architecture is cause for concern. — https://kellyyutonghe.github.io/10799S26/assets/pdf/hw2.pdf
- CMU 10-423/623, HW2 6.3–6.5, Spring 2026: sample x_τ by iterating the chain and by the closed form, and compare the time complexity of the two (drill). — https://www.cs.cmu.edu/~mgormley/courses/10423/homework/hw2.zip
- Prince, *UDL* ch. 18, Problems 18.1–18.3: variance preservation of one step, sums of Gaussians, and the closed-form marginal after three steps (drills already in the prose). — https://github.com/udlbook/udlbook/releases (v5.0.3 PDF)
- Berkeley CS180/280A, Project 5B Part 2, Fall 2025: time-conditioned flow matching on MNIST with linear interpolation and the velocity target. — https://cal-cs180.github.io/fa25/hw/proj5/partb.html

**Proposed problem set** (10 problems):
1. [conceptual] **The marginal of the linear path** (`ex_gaussian-paths-linear-marginal`). Keep. *Provenance:* book-original.
1. [conceptual] **Velocity, noise and clean-point targets** (`ex_gaussian-paths-prediction-targets`). Rewrite (b). Keep the two conversions in (a) and the loss-weight identities in (c). Replace the per-coordinate error derivation in (b) with a one-line consequence on the linear path: which prediction is a poor source of the velocity near t → 0 and near t → 1. Cite `ex_mdl-flow-matching-transport-target-conditioning` for the general error propagation. *Provenance:* book-original; (b) overlaps the appendix exercise (overlap med), resolved by the pointer.
1. [short-code] **The loss floor** (`ex_gaussian-paths-loss-floor`). Rewrite: keep (a)–(c) and add (d). "A velocity model on the linear path and a noise model on the VP path, trained on the same data, print final losses that differ by a large factor. Is either model worse? Convert both to the implied clean-point error at t ∈ {0.1, 0.5, 0.9} with the weights of `ex_gaussian-paths-prediction-targets`(c), and compare each loss with its own floor. State which comparison is fair." Success criterion: the comparison against floors ranks the models; the raw losses do not. *Provenance:* (a)–(c) book-original; (d) adapted from CMU 10-799 HW2 Q3(a), Spring 2026 (overlap med; cite on adoption).
1. [conceptual] **The ancestral step to first order, for Gaussian data** (`ex_gaussian-paths-ancestral-first-order`). Keep. It is related to the appendix's `ex_mdl-fokker-planck-probability-flow-gaussian-reverse-step` (OU, exact conditioning), and the two are complementary: this one gives the per-step variance error of the discrete chain. *Provenance:* book-original.
1. [conceptual] **Step size and Gaussian reversal** (`ex_gaussian-paths-gaussian-reversal`). Keep. This is the home of the reverse-transition covariance τ²I + c²Cov[x₀|x_t]; `ex_samplers-optimal-variance` should cite it. *Provenance:* book-original.
1. [short-code] **Commitment time** (`ex_gaussian-paths-commitment-time`). Keep. It already points to the appendix's late-decisions and mode-merging exercises. *Provenance:* adapted from Biroli et al. 2024 (cited in the problem).
1. [conceptual] **Same ratio, same flow** (`ex_gaussian-paths-same-ratio-same-flow`). Keep. *Provenance:* book-original.
1. [conceptual] **Ladder spacing in high dimensions** (`ex_gaussian-paths-ladder-spacing`). Keep. *Provenance:* adapted from Song & Ermon 2020 (cited).
1. [short-code] **The exact optimum memorizes** (`ex_gaussian-paths-memorization`). Keep. This is the experiment the tutorial survey singled out as "an ideal exercise on the mixture". *Provenance:* adapted from Bertrand et al. 2025 (cited).
1. [extended] **A noise-conditional score network on the mixture** (`ex_gaussian-paths-ncsn`). Keep. *Provenance:* adapted from Song & Ermon 2019 (cited).

---

## chapter_diffusion-models/image-diffusion.md — Diffusion for Images

**Topic:** The denoising U-Net (skips, resolutions, time embedding), class conditioning, the training recipe and EMA, sample quality by Fréchet distance and KID, classifier and classifier-free guidance as a tilt.

**Current exercises:** 10; disposition: keep 8, rewrite 1, drop 1 (merged). `ex_image-diffusion-kid` repeats the KID-against-FD comparison of `ex_image-diffusion-fd-floor`(b). Its one new part, whether 500-sample scores separate γ = 1 from γ = 1.5, moves into fd-floor, and its ranking question adds little. The rest is strong: an exact equivalence of guidance rules, interval guidance and autoguidance (2024 results), thresholding, and the conditional-score calculation. The external tradition shows two gaps. First, no exercise teaches what the Fréchet distance responds to (10-799 HW1 Q2). Second, nothing asks the reader to diagnose a broken trainer, the most useful practical skill in 10-799. UDL 18.12 suggests a third: guidance compared with truncation.

**External sources found:**
- CMU 10-799, HW1 Q5 "Pseudo Debugging Scenarios" and HW2 Q3, Spring 2026. A classmate's DDPM shows a good loss curve but pure-noise samples, or a bumpy plateau, or NaN spikes under flow matching; students name the likely bugs and the fixes. — https://kellyyutonghe.github.io/10799S26/assets/pdf/hw1.pdf and …/hw2.pdf
- CMU 10-799, HW1 Q2(a) and Q4(c), Spring 2026. Would a model that is state of the art on full CelebA also score state-of-the-art FID/KID against a filtered subset? Report KID mean and std on 1k samples. — hw1.pdf, same site
- Berkeley CS294-158, HW4 Q2 and Q3(c), Spring 2024: a pixel U-Net on CIFAR-10 with x̂ clipped to [−1, 1] during sampling; CFG over a range of w, noting the U-shaped quality curve. — https://github.com/rll/deepul/blob/master/homeworks/hw4/hw4.ipynb
- MIT 6.S184, Lab 3 Q2.2–2.4, IAP 2026: a CFG flow trainer with a null label, a conditional vector field, and a sanity check on a Gaussian mixture. — https://github.com/eje24/iap-diffusion-labs (branch `2026`, `labs/lab_three.ipynb`)
- Berkeley CS180/280A, Project 5B §2.4–2.6, Fall 2025: class conditioning with 10% label dropout and CFG at γ = 5 on MNIST. — https://cal-cs180.github.io/fa25/hw/proj5/partb.html
- CMU 10-423/623, HW2 7.2, Spring 2026: compute FID every 100 steps over 1,000 training steps and plot it. — https://www.cs.cmu.edu/~mgormley/courses/10423/homework/hw2.zip
- Prince, *UDL* Problem 18.12: CFG produces more "canonical" images. Which methods in earlier chapters (decoders, GANs, Glow) also reduce variation? Is limiting the output inevitable? — UDL v5.0.3 PDF

**Proposed problem set** (12 problems):
1. [conceptual] **Equivalent guidance rules** (`ex_image-diffusion-equivalent-guidance`). Keep. *Provenance:* book-original; the Bayes-rule step overlaps 6.S184 Lab 3 Part 2 (overlap low).
1. [short-code] **The floor of the Fréchet distance** (`ex_image-diffusion-fd-floor`). Rewrite: keep (a)–(c) and add (d) from the dropped KID exercise. Score ten disjoint 500-sample subsets at γ = 1 and γ = 1.5 with both FD and KID, and decide whether either separates the scales. *Provenance:* book-original.
1. [short-code] **A clean classifier as guide** (`ex_image-diffusion-clean-classifier-guide`). Keep. *Provenance:* book-original.
1. [short-code] **Guidance in a limited interval** (`ex_image-diffusion-guidance-interval`). Keep. It is the image counterpart of the appendix's one-dimensional `ex_mdl-score-matching-diffusion-flow-guidance-levels`; add a cross-reference. *Provenance:* adapted from Kynkäänniemi et al. 2024 (cited).
1. [extended] **Autoguidance** (`ex_image-diffusion-autoguidance`). Keep. *Provenance:* adapted from Karras et al. 2024 (cited).
1. [short-code] **Skip connections** (`ex_image-diffusion-skip-connections`). Keep. *Provenance:* book-original.
1. [conceptual] **Conditional scores of the mixture** (`ex_image-diffusion-mixture-conditional`). Keep. *Provenance:* book-original.
1. [short-code] **The rate of label dropout** (`ex_image-diffusion-label-dropout`). Keep. It is a sweep, but it has a question and a mechanism (updates per prediction). *Provenance:* book-original.
1. ~~**KID against FD** (`ex_image-diffusion-kid`)~~. Drop; (b) is merged into fd-floor (d).
1. [short-code] **Thresholding the predicted image** (`ex_image-diffusion-thresholding`). Keep. *Provenance:* adapted from Saharia et al. 2022 (cited); clipping also appears as a hint in CS294-158 HW4 Q2.
1. [short-code] **What the Fréchet distance sees** (new). Corrupt 5,000 real test images in controlled ways and score each set with `fd_and_agreement` against the other test half:
    - drop one class and resample to 5,000;
    - double the weight of one class;
    - blur with σ = 0.5 and σ = 1 pixel;
    - add noise of standard deviation 0.05;
    - replace 10% of the images by their class mean.

   Tabulate FD, KID and agreement against the floor. Which corruption of these sizes produces the model's FD of about 1 at n = 5,000? Is FD more sensitive to a missing class or to blur? What does the answer imply for reading FD as a measure of diversity versus fidelity? Success criterion: every corrupted set lies above the floor, and the table ranks the corruptions.
   *Provenance:* adapted from CMU 10-799 HW1 Q2(a), Spring 2026 (overlap low: that question asks only about a mismatched reference set; the controlled-corruption table is original).
1. [short-code] **Diagnosing a broken trainer** (new). Before running anything, the reader predicts how each of four bugs shows up in (i) the training loss and (ii) the samples and their FD:
    - images scaled to [0, 1] instead of [−1, 1];
    - the time passed to the network in different units at training and at sampling;
    - the noise drawn once and reused for every minibatch;
    - label dropout that substitutes class 0 instead of the null label.

   Then inject each bug into `batch_fn`/`loss_fn`/`generate`, train 3,000 updates, and fill in the table: loss at the end, FD and agreement at γ = 1.5. Which bugs are invisible in the loss curve, and why can no training loss detect a sampler-side bug? Success criterion: the predicted symptom matches the measured one for at least three bugs, and the mismatches are explained.
   *Provenance:* adapted from CMU 10-799 HW1 Q5 and HW2 Q3, Spring 2026 (overlap med: the same "debug from symptoms" format, with bugs we inject and measure instead of figures we show; cite on adoption).
1. [conceptual] **Guidance on Gaussian classes** (new). Let a class be N(μ_c, s²I) and, for this sketch, model the unconditional data as N(0, S²I) with S² > s². Both stay Gaussian on the linear path.
    1. Show that the classifier-free score γ∇log p_t(x|c) + (1 − γ)∇log p_t(x) is the score of a Gaussian. At t = 0 its variance is s²S²/(γS² − (γ − 1)s²) < s², and its mean is μ_c scaled by a factor above one: guidance narrows the class and exaggerates it.
    1. The guided probability-flow ODE is linear here. Integrate it numerically from N(0, I) for γ ∈ {1, 2, 4}. Compare the end variance with (a), and explain the difference (the guided fields are not the velocities of one noised path; compare `ex_mdl-score-matching-diffusion-flow-guided-marginals`).
    1. Compare with the truncation trick of `ex_conditional-biggan-s-truncation-trick` and with sampling at a temperature below one. Which of the three shrinks the variance without moving the mean?

   *Provenance:* inspired by Prince *UDL* Problem 18.12 (overlap low: UDL asks for a qualitative comparison; the Gaussian calculation is original).

---

## chapter_diffusion-models/samplers.md — Samplers, Inversion and Editing

**Topic:** The DDIM family as Euler's method in (x̄, σ̄) with a noise-strength η; fewer steps, second-order steps and the step grid (EDM's ρ-grid, Heun); DDIM inversion, editing by partial noising (SDEdit), inpainting by replacement.

**Current exercises:** 10; disposition: keep 9, rewrite 1, drop 0. The set is excellent: the admissibility and Markov-member derivations, Gaussian data (DDIM always shrinks variance), the optimal reverse variance, RePaint, and the DPM-Solver-2 extended problem. One rewrite: `ex_samplers-optimal-variance`(a) re-derives the reverse-transition covariance already proved in `ex_gaussian-paths-gaussian-reversal`(a). The course tradition centres on editing (CS180's SDEdit, text-conditional translation, inpainting, anagrams, hybrids). The section edits only with an unconditional model, so one new problem steers an edit by class.

**External sources found:**
- CMU 10-799, HW2 Q4–Q5, Spring 2026: implement DDIM on the HW1 DDPM without retraining; report KID against the step count for DDIM and flow matching; find the step count at which each matches 1,000-step DDPM. — https://kellyyutonghe.github.io/10799S26/assets/pdf/hw2.pdf
- CMU 10-799, HW1 Q7, Spring 2026: KID of DDPM sampling at 100–1,000 steps. — hw1.pdf, same site
- Berkeley CS180/280A, Project 5A §1.4 "Iterative Denoising" with strided timesteps, §1.7 SDEdit at several starting indices, §1.7.2 inpainting by replacing known pixels with the forward-noised original, and §1.7.3 text-conditional image-to-image translation, Fall 2025. — https://cal-cs180.github.io/fa25/hw/proj5/parta.html
- Berkeley CS180/280A, Project 5A §1.8–1.9, Fall 2025: visual anagrams average the noise estimates of a flipped and an unflipped prompt; hybrid images take the low-pass of one prompt's estimate plus the high-pass of another's (factorized diffusion). — same URL
- MIT 6.S184, Lab 1 Q1.1, IAP 2026: implement the Euler and Euler–Maruyama step methods. — https://github.com/eje24/iap-diffusion-labs (branch `2026`, `labs/lab_one.ipynb`)
- Berkeley CS294-158, HW4 Q1, Spring 2024: write the DDPM sampler in the DDIM paper's form. — https://github.com/rll/deepul/blob/master/homeworks/hw4/hw4.ipynb

**Proposed problem set** (11 problems):
1. [conceptual] **Admissible noise scales** (`ex_samplers-admissible-noise`). Keep. *Provenance:* book-original.
1. [conceptual] **The forward step and the Markov member** (`ex_samplers-forward-step`). Keep. *Provenance:* book-original.
1. [conceptual] **Gaussian data** (`ex_samplers-gaussian-data`). Keep. It complements the appendix's `ex_mdl-score-matching-diffusion-flow-ddim-heun`, which does Heun in the same coordinates. *Provenance:* book-original.
1. [short-code] **The strength η and the code** (`ex_samplers-eta-strength`). Keep. *Provenance:* book-original; the step sweep parallels 10-799 HW2 Q5 (overlap low).
1. [short-code] **Inversion on both examples** (`ex_samplers-inversion-steps`). Keep. *Provenance:* book-original (compares with Song et al. 2020, Table 2).
1. [extended] **A second-order sampler in log-SNR** (`ex_samplers-dpm-solver`). Keep. *Provenance:* adapted from Lu et al. 2022 (cited).
1. [short-code] **Fewer ancestral steps** (`ex_samplers-ancestral-strides`). Keep. *Provenance:* book-original.
1. [conceptual] **The optimal reverse variance** (`ex_samplers-optimal-variance`). Rewrite (a). Cite `ex_gaussian-paths-gaussian-reversal`(a) for the covariance and ask only for the identification k = α_s − σ_sκα_t/σ_t. Keep (b)–(d). *Provenance:* adapted from Bao et al. 2022 (cited).
1. [short-code] **Inpainting with harmonization** (`ex_samplers-repaint`). Keep. *Provenance:* adapted from Lugmayr et al. 2022 (cited); inpainting by replacement is also CS180 Project 5A §1.7.2 (overlap low).
1. [short-code] **A rescaled schedule** (`ex_samplers-rescaled-schedule`). Keep. *Provenance:* adapted from Gao et al. 2025 (cited).
1. [short-code] **Editing toward another class** (new). Use the class-conditional U-Net of the image section.
    1. Take 100 test images of one class (trousers, say), noise them to t₀ ∈ {0.3, 0.5, 0.7, 0.9}, and integrate back with 25 Euler steps of `cfg(ema, c, 2.0)` for a different class c (dresses). Report the fraction that `cnn` assigns to c and the squared distance to the input, as in the SDEdit cell.
    1. From pure noise, sample with label a for t > t* and label b for t ≤ t*, for t* ∈ {0.2, 0.4, 0.6, 0.8}. Plot the fraction classified as b against t*, and compare the switch-over with the t₀ of (a) and with `ex_gaussian-paths-commitment-time`.
    1. Why must an edit start from a larger t₀ than the latest effective switch time t*? Answer in terms of what the input image fixes at every level.

   Success criterion: both curves rise monotonically, and the reader states the gap between them with its explanation.
   *Provenance:* adapted from Berkeley CS180/280A Project 5A §1.7.3 (text-conditional image-to-image translation), Fall 2025 (overlap med on (a); (b)–(c) original; cite on adoption).

---

## chapter_diffusion-models/design-space.md — Likelihood, Weighting and the Design Space

**Topic:** The chain as a latent-variable model and its bound, reduction to weighted regression, a likelihood for the mixture (bound and probability-flow ODE); every objective as a weighting in log-SNR, the time distribution as importance sampling; EDM's table, noise levels as scales, resolution and the schedule, a recipe ablation, parameterization at finite capacity.

**Current exercises:** 10; disposition: keep 8, rewrite 2, drop 0. This is the deepest set in the chapter. External homework reaches only the ELBO surgery: CMU 10-423 HW2 6.1–6.2, and UDL 18.4–18.11 (Gaussian product and KL identities, ᾱ algebra). Here those are a single worked exercise, `ex_design-space-forward-posterior`, which is the right proportion. No course poses problems on log-SNR weightings, EDM preconditioning, paired likelihood comparisons or bits per dimension through the ODE. Two rewrites remove overlap with the math appendix:
- `ex_design-space-hutchinson`(a) repeats the appendix's Hutchinson-variance derivation.
- `ex_design-space-prior-term`(c) is the ancestral half of the appendix's extended "Prior mismatch, ODE versus SDE".

**External sources found:**
- CMU 10-423/623, HW2 6.1–6.2, Spring 2026: ELBO surgery into reconstruction, per-step KL and a constant, then explain in words what L_t does to p_θ(x_t | x_{t+1}). — https://www.cs.cmu.edu/~mgormley/courses/10423/homework/hw2.zip
- Prince, *UDL* ch. 18, Problems 18.4–18.11: Gaussian change of variables and products (18.4–18.5), derivations of the posterior and the loss terms (18.6–18.8), and ᾱ identities (18.9–18.10). These are drills that match `ex_design-space-forward-posterior` in scope. — UDL v5.0.3 PDF
- CMU 10-799, HW1 Q6 (extra credit), Spring 2026: derive an alternative to ε-prediction, train it with the same configuration, and compare KID. This parallels `ex_design-space-target-and-weighting`(b). — https://kellyyutonghe.github.io/10799S26/assets/pdf/hw1.pdf
- Berkeley CS294-158, HW4 Q1–Q2, Spring 2024: the cosine schedule, and the advice to clip x̂ to [−1, 1] while sampling. This parallels `ex_design-space-cosine-clipping`. — https://github.com/rll/deepul/blob/master/homeworks/hw4/hw4.ipynb
- **No external tradition** for EDM preconditioning, log-SNR weighting, or likelihood in bits per dimension via the probability-flow ODE. The EDM paper, Kingma & Gao 2023 and Dieleman's posts pose no exercises.

**Proposed problem set** (10 problems):
1. [short-code] **The prior term** (`ex_design-space-prior-term`). Rewrite (c): keep it, and add a pointer to the extended "Prior mismatch, ODE versus SDE" in `sec_mdl-fokker-planck-probability-flow`, which compares how the ODE and the SDE carry the same mismatch. *Provenance:* book-original; (c) overlaps the appendix project (overlap med), resolved by the pointer.
1. [short-code] **The weights of the bound** (`ex_design-space-weights-of-the-bound`). Keep. Its importance-sampling part complements the appendix's `ex_mdl-score-matching-diffusion-flow-weightings`(c) (Song et al. 2021 versus Nichol & Dhariwal 2021). *Provenance:* book-original.
1. [short-code] **Target and weighting** (`ex_design-space-target-and-weighting`). Keep. *Provenance:* book-original; the comparison reproduces Ho et al. 2020, Table 2. Its parallel with 10-799 HW1 Q6 is overlap low.
1. [conceptual] **The implied weighting** (`ex_design-space-implied-weighting`). Keep. *Provenance:* book-original.
1. [short-code] **Likelihoods in many dimensions** (`ex_design-space-hutchinson`). Rewrite (a). Cite `ex_mdl-odes-solvers-hutchinson-variance` for the variance of one probe. Keep the unbiasedness of the log-density estimate and the measured standard deviation per point. Keep (b), bits per dimension on images. *Provenance:* book-original; (a) overlaps the appendix exercise (overlap high), resolved by the pointer.
1. [conceptual] **Resolution and the schedule** (`ex_design-space-resolution-schedule`). Keep. *Provenance:* book-original (compares with Hoogeboom et al. 2023).
1. [short-code] **Paired likelihood comparisons** (`ex_design-space-paired-likelihoods`). Keep. *Provenance:* book-original.
1. [conceptual] **The forward posterior** (`ex_design-space-forward-posterior`). Keep. Overlap med with UDL 18.5–18.6 and CMU 10-423 HW2 6.1; the T = 2 worked case goes beyond both. *Provenance:* book-original.
1. [short-code] **Clipping the predicted image** (`ex_design-space-cosine-clipping`). Keep. *Provenance:* book-original.
1. [conceptual] **EDM's preconditioning** (`ex_design-space-edm-preconditioning`). Keep. *Provenance:* adapted from Karras et al. 2022 and Kingma & Gao 2023 (cited).

---

## chapter_diffusion-models/latent-diffusion.md — Latent Diffusion and Transformers

**Topic:** The cost of pixels and a KL-regularized autoencoder; the same regression on scaled latents; the diffusion transformer (patches, adaptive layer norm with zero-initialized gates), scaling with compute; conditioning on text by cross-attention and joint attention (MMDiT), and video.

**Current exercises:** 10; disposition: keep 10, rewrite 0, drop 0. This is the only section under the guide (28%). The set covers the latent scale factor, token arithmetic, zero gates, patch size, bottleneck width, KL weight, caption dropout and scaling. It is close to CS294-158 HW4 Q3 and 6.S184 Lab 3, which build exactly this pipeline. There is room for two additions:
- The text-conditioning subsection has only one exercise, on cost. CMU 10-423 HW4 2.3 poses a sharp question about cross-attention.
- The design-space section states :citet:`Li.He.2026`'s finding that large-patch transformers need the clean-image target, but no exercise tests it. The DiT here is the natural place to do so.

**External sources found:**
- Berkeley CS294-158, HW4 Q3(a), Spring 2024: VAE reconstructions; the scale factor is the standard deviation of 1,000 encoded latents. Parallels `ex_latent-scale-factor`. — https://github.com/rll/deepul/blob/master/homeworks/hw4/hw4.ipynb
- Berkeley CS294-158, HW4 Q3(b)–(c), Spring 2024: a latent DiT with 2-D sin-cos positions, adaLN-modulated blocks and gates, a class embedding with CFG dropout, then CFG at several w. — same notebook
- MIT 6.S184, Lab 3 Parts 3–5, IAP 2026: a Fourier time encoder, a patchifier and depatchifier, a diffusion transformer, a VAE, and a latent CFG trainer, all from scratch on MNIST. — https://github.com/eje24/iap-diffusion-labs (branch `2026`, `labs/lab_three.ipynb`)
- CMU 10-423/623, HW4 2.1–2.3, Spring 2026: the latent size at f = 8 for a 512² image; why diffuse in a latent space; cross-attention with values computed from the image instead of the text, asking what goes wrong mathematically and intuitively. — https://www.cs.cmu.edu/~mgormley/courses/10423/homework/hw4.zip
- Stanford CS231n, Lecture 14 slides, 2025: latent diffusion with a VAE plus GAN decoder, DiT conditioning, FLUX and Movie Gen. These are slides only, with no exercise. — https://cs231n.stanford.edu/slides/2025/lecture_14.pdf (content from the course survey; URL verified)

**Proposed problem set** (12 problems):
1. [conceptual] **The scale of the latents** (`ex_latent-scale-factor`). Keep. *Provenance:* book-original; the scale-factor computation parallels CS294-158 HW4 Q3(a) (overlap low).
1. [conceptual] **Tokens at high resolution** (`ex_latent-compression-arithmetic`). Keep. *Provenance:* book-original; (a) overlaps CMU 10-423 HW4 2.1 (overlap low).
1. [short-code] **Why start at zero** (`ex_latent-zero-gate`). Keep. *Provenance:* book-original.
1. [short-code] **Patch size** (`ex_latent-patch-size`). Keep. *Provenance:* book-original.
1. [short-code] **Width of the bottleneck** (`ex_latent-bottleneck`). Keep. *Provenance:* book-original.
1. [short-code] **The KL weight** (`ex_latent-kl-weight`). Keep. *Provenance:* book-original.
1. [conceptual] **Captions and guidance** (`ex_latent-caption-dropout`). Keep. *Provenance:* book-original.
1. [short-code] **Scaling in the latent space** (`ex_latent-scaling-in-latents`). Keep. *Provenance:* book-original.
1. [short-code] **Compute per step and compute in total** (`ex_latent-total-compute`). Keep. *Provenance:* book-original (compares with Peebles & Xie 2023).
1. [extended] **Latent diffusion for sprites** (`ex_latent-sprites`). Keep. *Provenance:* book-original.
1. [conceptual] **Cross-attention as a lookup** (new). Image tokens Y ∈ ℝ^{m×d_y} query caption tokens X ∈ ℝ^{n×d_x}.
    1. Show that each image position receives a convex combination of the caption's value vectors. With the one-token empty caption, every position receives the same vector, so the unconditional branch of guidance sees cross-attention as a learned constant shift.
    1. Suppose the values are instead computed from the image, V = YW_v. What goes wrong with the shapes when m ≠ n, and what would such a layer compute even when m = n?
    1. In cross-attention the caption's keys and values do not depend on x_t. In MMDiT the caption stream is updated by the image tokens in every block. For a caption of 77 tokens, 1,024 image tokens and 50 guided steps, count the caption-side work that cross-attention can cache and MMDiT cannot.

   *Provenance:* (b) adapted from CMU 10-423/623 HW4 2.3, Spring 2026 (overlap med; cite on adoption); (a) and (c) original.
1. [short-code] **Large patches and the clean-image target** (new). Train `DiT` on pixels with `patch_size=7`, so each token carries 49 values, at widths 32 and 256, for 10,000 updates each. Two output parameterizations share the same velocity loss: the network outputs the velocity directly, or it outputs x̂₀ and the velocity is formed as (x_t − x̂₀)/t (with t clipped at 0.05).
    1. Report the FD of the four models.
    1. Is the clean-image output better at width 32, where a token has more values than the network has channels, and does the gap close at width 256?
    1. Explain the result with the argument of the design-space section: the clean image lies near a low-dimensional set and the noise fills every dimension.

   Success criterion: a 2×2 FD table that shows whether the predicted interaction appears.
   *Provenance:* inspired by Li & He 2026 (overlap low: the claim comes from the paper; this small-scale test is original).

---

## chapter_diffusion-models/few-step.md — Few-Step Generation

**Topic:** Why one Euler step lands on the mean; straighter paths (reflow) help but do not finish; flow maps, consistency models and the MeanFlow identity; experiments on the mixture and on images; distribution matching (adversarial and score-difference distillation) as the other route.

**Current exercises:** 10; disposition: keep 10, rewrite 0, drop 0. This is a strong, current set: reflow, progressive distillation, discrete versus continuous consistency distillation, the stop-gradient, the Lagrangian loss, a guided student, a cost curve, and shortcut models as the extended problem. There is no course homework on flow maps (a finding). The nearest is CMU 10-799's open "Speed" research track. One gap is the last subsection, "The Other Route: Matching Distributions". It explains score-difference distillation (Yin et al. 2024) in prose and has no exercise. A Gaussian calculation shows both why it works and that it seeks modes, which connects to the GAN chapter.

**External sources found:**
- CMU 10-799, HW2 Part V and HW3–4 "Speed" track, Spring 2026: reach a KID threshold in as few steps as possible by distillation, consistency models or better solvers; survey 4–6 methods, implement a baseline, propose an improvement. HW4 Q3(b) names MeanFlow's JVP as an example of a practical design choice. — https://kellyyutonghe.github.io/10799S26/assets/pdf/hw2.pdf, …/hw3.pdf, …/hw4.pdf
- Berkeley CS180/280A, Project 5B §1.2.3, Fall 2025: a denoiser trained on pure noise returns the centroid of the training images. This is the one-step failure in its simplest form, matching `ex_few-step-one-step-mean`(a). — https://cal-cs180.github.io/fa25/hw/proj5/partb.html
- Berkeley CS294-158, Lecture 6, Spring 2024: progressive distillation, DD-GAN and SDXL-Lightning are taught, but HW4 has no few-step problem (from the course survey; deck not re-fetched).
- **No external tradition** for consistency training, MeanFlow or shortcut models as problems. Lai et al. (*Principles of Diffusion Models*, ch. 10–11) and Dieleman's 2026 flow-map post pose none.

**Proposed problem set** (11 problems):
1. [conceptual] **One and two Euler steps** (`ex_few-step-one-step-mean`). Keep. It already points to the appendix's two-point flow. *Provenance:* book-original.
1. [short-code] **Reflow on the mixture** (`ex_few-step-reflow`). Keep. It is the empirical counterpart of the appendix's exact `ex_mdl-flow-matching-transport-reflow-two-points`. *Provenance:* adapted from Liu et al. 2022 (cited).
1. [short-code] **Progressive distillation** (`ex_few-step-progressive`). Keep. *Provenance:* adapted from Salimans & Ho 2022 (cited).
1. [short-code] **Which time pairs to train on** (`ex_few-step-time-pairs`). Keep. *Provenance:* book-original.
1. [short-code] **The derivative against a difference** (`ex_few-step-discrete-cd`). Keep. *Provenance:* adapted from Song et al. 2023 (cited).
1. [short-code] **The stop-gradient** (`ex_few-step-stop-gradient`). Keep. *Provenance:* book-original.
1. [conceptual] **The Lagrangian loss** (`ex_few-step-lagrangian`). Keep. *Provenance:* adapted from Boffi et al. 2025 (cited).
1. [short-code] **A guided student** (`ex_few-step-guided-student`). Keep. *Provenance:* adapted from Meng et al. 2023 (cited).
1. [short-code] **Quality against cost** (`ex_few-step-cost-curve`). Keep. *Provenance:* book-original.
1. [extended] **Shortcut models** (`ex_few-step-shortcut`). Keep. *Provenance:* adapted from Frans et al. 2025 (cited).
1. [conceptual] **Matching noised distributions** (new). A one-step generator x = g_θ(z) is trained along E[(∇log q_{θ,t}(x_t) − ∇log p_t(x_t))ᵀ ∂x_t/∂θ], with x_t = α_t g_θ(z) + σ_t ε: the score of its own noised samples minus the teacher's.
    1. Show that this is the gradient of KL(q_{θ,t} ‖ p_t). The term from θ's influence on q itself vanishes because E_q[∇_θ log q_θ] = 0.
    1. For the generator m + sz and a target N(μ, v), write KL(q_t ‖ p_t) in closed form on the linear path and show that m = μ, s² = v is its only stationary point at every t.
    1. For the target ½N(−a, v) + ½N(a, v) with a = 2, v = 0.25, evaluate KL(q_t ‖ p_t) by quadrature on a grid of (m, s) at t ∈ {0.1, 0.5, 0.9}. Where is the minimizer at each t? Relate the answer to mode dropping in the GAN chapter, and to the mode merging of `ex_mdl-fokker-planck-probability-flow-mode-merging`. Which weighting over t would let a Gaussian generator cover both modes, and why can no weighting make it fit them?

   Success criterion: the minimizer moves from one mode toward the centre as t grows.
   *Provenance:* original, inspired by Yin et al. 2024 (DMD) (overlap low).

---

## chapter_diffusion-models/discrete-diffusion.md — Discrete Diffusion

**Topic:** Transition matrices and the absorbing (mask) kernel; the masked bound as a weighted cross-entropy and its continuous-time limit; parallel unmasking and its independence error; diffusion language models in practice (confidence remasking); experiments on a quantized two-token mixture and on binary Fashion-MNIST; masked language models as one step.

**Current exercises:** 10; disposition: keep 8, rewrite 2, drop 0. The set is excellent and tightly interlocked: the uniform kernel, schedule weights, pairs revealed together, the exact floor (H + I·Σ Δ²), the order-agnostic bound, time-agnostic predictions and confidence remasking. Several exercises reference each other's parts, which makes the exact-network theory a short course of its own. At 59% it is the heaviest section in the chapter, and it has no external tradition to draw on, so no new problems are proposed. Both rewrites shorten:
- `ex_discrete-diffusion-two-forms`(d) asks for a sequence-level generalization that the other three parts already motivate.
- `ex_discrete-diffusion-language-model` spends many lines on API scaffolding (the nested `CausalAttention` override) that a code hint could carry.

**External sources found:**
- MIT 6.S184, Lecture 5 "Discrete diffusion" and lecture notes ch. 7 (CTMCs, factorized mixture and masking paths), IAP 2026: new in 2026, with no lab or problem attached. — https://diffusion.csail.mit.edu/2026/ (labs at https://github.com/eje24/iap-diffusion-labs, branch `2026`: three labs, none on discrete diffusion)
- CMU 10-799, Lectures 9–10 (D3PM, SEDD, MDLM, LLaDA; discrete flow matching, Edit Flows), Spring 2026: the homework (HW1–4, read) contains no discrete problem. — https://kellyyutonghe.github.io/10799S26/
- Stanford CS236, Lecture 18 "Diffusion for discrete data", Fall 2023: lecture only; homework not public (course survey).
- **No external tradition found.** The closest are the repositories cited in the chapter intro (MDLM) and the scaling studies the extended problem reproduces (Prabhudesai et al. 2025).

**Proposed problem set** (10 problems):
1. [conceptual] **The uniform kernel** (`ex_discrete-diffusion-uniform-kernel`). Keep. *Provenance:* book-original (after Austin et al. 2021).
1. [conceptual] **Weights of other schedules** (`ex_discrete-diffusion-schedule-weights`). Keep. *Provenance:* book-original.
1. [conceptual] **Pairs revealed together** (`ex_discrete-diffusion-pairs-revealed`). Keep. *Provenance:* book-original.
1. [short-code] **One token at a time** (`ex_discrete-diffusion-one-token`). Keep. *Provenance:* book-original.
1. [conceptual] **Two reverse parameterizations** (`ex_discrete-diffusion-two-forms`). Rewrite. Drop (d), the sequence-level factorized case, or reduce it to its final question, "which inputs does each form's network have to use at position i?", with the answer's setup given. Keep (a)–(c). *Provenance:* book-original (after Austin et al. 2021).
1. [short-code] **Time-agnostic predictions** (`ex_discrete-diffusion-time-agnostic`). Keep. *Provenance:* book-original (after Ou et al. 2025).
1. [short-code] **Floor of an exact network** (`ex_discrete-diffusion-revealed-together`). Keep. *Provenance:* book-original.
1. [conceptual] **The order-agnostic bound** (`ex_discrete-diffusion-order-agnostic`). Keep. *Provenance:* adapted from Ou et al. 2025 (cited).
1. [short-code] **Confidence-ordered unmasking** (`ex_discrete-diffusion-confidence-unmasking`). Keep. *Provenance:* book-original (rule of Nie et al. 2025).
1. [extended] **A masked diffusion language model** (`ex_discrete-diffusion-language-model`). Rewrite (length only). Move the subclassing recipe of (b) into a two-line hint: "override the nested `CausalAttention` and call fused attention without the mask". Keep the scoping-run expectation and all five parts. *Provenance:* adapted from Prabhudesai et al. 2025 (cited).
