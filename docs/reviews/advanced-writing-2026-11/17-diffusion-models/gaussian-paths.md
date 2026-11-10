# Gaussian Paths: Diffusion and Flow Matching

Source: `chapter_diffusion-models/gaussian-paths.md` (read to the slides marker). About 4,100 words of body prose, 1,900 words of exercises, 38 code cells (about 20 per framework).

## Verdict

This is the section the chapter stands on, and its mathematics is right: I rederived every identity (the VP schedule and its $\beta_k$, the score and velocity conversions for all five targets including the code in `x0_eps`, the $2/\pi$ factor for $\mathbf{v}$ on the cosine path, the ancestral step in score form and its first-order split, the reverse SDE and probability-flow ODE for the VP path, the posterior mean and variance in `ancestral_sample`) and checked every number in the prose against both stored outputs. No error. The design is also right for 2026: the default model is a velocity on the linear path, DDPM is the second model, and the two are shown to hold the same posterior mean. What keeps the section from the bar is density. In 4,100 words it introduces the path, the log-SNR, the DDPM chain, annealed Langevin dynamics, conditional flow matching, five targets, the loss floor, the probability-flow ODE, a change of variables to $\bar\sigma$, the ancestral chain and the reverse SDE, and then runs three verification experiments. The one claim the whole section rests on, that following the marginal velocity carries noise to data, is handed to the appendix in a subordinate clause. Fix that first.

## Scores

| Dimension | Score |
|---|---|
| Clarity of explanation | 3 |
| Clarity of thought and focus | 4 |
| Pedagogy | 3 |
| Language | 3 |
| Ease and enjoyment | 3 |

## What works

- **The annealing experiment with exact scores** ("Because the scores of the smoothed mixture are known exactly, we can test this before training any network"). It answers the previous section's dilemma with no learning in the way, and its reading is a sentence worth keeping: "The proportions are decided at the coarse levels, where the smoothed density has no valleys; the fine levels only sharpen each mode."
- **Conditional and marginal, said plainly.** "given $\mathbf{x}_0$, the best guess of $\mathbf{x}_0$ is $\mathbf{x}_0$ itself" and "Where only one clean point can have produced $\mathbf{x}$, it is that point's velocity; where several can, it is their average". The four-panel figure (`fig_diffusion-blueprint`) carries this; I rendered it and it is correct and legible.
- **The build-up from one data point** to two to a mixture of point masses, ending in the prediction that one Euler step sends every start to the data mean $(-0.5, -0.7)$, which the cell then prints as $(-0.52, -0.715)$. This is the Illustrated-Transformer move the rest of the section needs more of.
- **The ancestral step split into a Langevin step plus an Euler step of the flow.** A reader who followed the previous section sees at once why the chain is a corrected flow. I checked the algebra and the VP velocity $-\tfrac12\beta(t)(\mathbf{x} + \nabla\log p_t)$ against the general identity.
- **The loss floor.** "Losses of different targets cannot be compared; predictors trained on different targets can, through the dictionary." The printed table (floors 0.54 and 4.9, training losses 0.542 and 4.899) makes a point practitioners get wrong.
- **The clock callout** fixes the time direction once and tells the reader how to read flow-matching papers. It agrees word for word with the appendix callout, and all symbols ($\alpha_t$, $\sigma_t$, $\bar\alpha_t = \alpha_t^2$, data at $t = 0$) agree with `chap_mdl-dynamics`.
- All 18 distinct appendix labels cited here resolve to passages that hold the cited content (checked each one, including the exercises).

## Structural findings

1. **The section is two sections.** The seam is before "## Sampling by Integration". Part A (path, chain, annealing, the regression, the targets, the floor; about 2,100 words) answers "what is trained". Part B (probability-flow ODE, Euler, ancestral chain, reverse SDE, the comparison with the oracle; about 2,000 words) answers "how do we sample". Each half would then have room for the two or three sentences of explanation it lacks (GP-1, GP-2, GP-3). If it stays one file, mark the seam and move "Experiments on the Running Example" into the parts it checks.
2. **"Experiments on the Running Example" is three confirmations in a row** (author decision 4). The loss-floor cell has already shown that both networks sit at their floors; "The Learned Predictor against the Optimal One" shows it again per time step and adds a Pythagoras check; "The Dictionary Checked" confirms an identity. Only "Samples from Both Samplers" answers the section's opening question (do the samplers recover the weights that one noise level could not). See GP-6.
3. **The teaching sampler and the library sampler compete.** `euler` is five lines and teaches. `ode_sample`, `pred_bar`, `sigma_bars`, `make_pred` and `time_at` are about 70 lines that serve every path, target and later section (the cell id shows `ode_sample` came from `samplers.md`). Their one new idea, the $\bar\sigma$ coordinates, gets two sentences. See GP-2 and GP-7.
4. **Missing: the name "variance exploding".** The previous section and the ladder here use $\alpha = 1$ with a growing $\sigma$; the reader will meet "VE" in every paper and gets the term only in `design-space.md`. See GP-8.
5. **The Summary holds a table and is followed by a second ending.** See GP-11.

## Passage-level findings

**GP-1** (major, explanation). Under "The Probability-Flow ODE and Euler's Method".
> The pair $(p_t, \mathbf{u}_t)$ satisfies the continuity equation (:numref:`sec_mdl-flow-matching`)

This clause is the theorem of the section: the average of the per-example velocities moves the whole density along the path, so integrating it from noise yields data. The continuity equation has not appeared in the main text, so a reader without the appendix has only "therefore". The point-mass build-up that follows shows the conditional flows, not why their posterior average transports the marginal, which is the step that is not obvious (averaged trajectories do not follow averaged velocities in general). Fix in three or four sentences, no proof: say in words what the continuity equation states (the density at a point changes by the net inflow of mass); say why the marginal velocity satisfies it (the flow of mass through a point is the sum of the flows of the conditional clouds, and dividing that sum by the density gives exactly the posterior-weighted average, which is how $\mathbf{u}_t$ was defined); then point to the appendix for the proof. Draft: "Why should the average velocity move the cloud correctly? Each conditional cloud $\mathcal{N}(\alpha_t\mathbf{x}_0, \sigma_t^2\mathbf{I})$ is moved correctly by its own velocity. The mass crossing a point $\mathbf{x}$ per unit time is the sum of what each cloud carries across, and that sum divided by the density at $\mathbf{x}$ is the posterior average :eqref:`eq_diffusion-marginal-velocity`. So a field that moves mass at this rate moves the mixture of the clouds, which is $p_t$."

**GP-2** (moderate, explanation). Same subsection.
> The library's sampler takes the same steps in other coordinates

Two sentences introduce $\bar{\mathbf{x}} = \mathbf{x}/\alpha_t$ and $\bar\sigma = \sigma_t/\alpha_t$ and assert $d\bar{\mathbf{x}}/d\bar\sigma = \hat{\boldsymbol\epsilon}$; every later sampler of the chapter uses this form. The derivation is two lines and should be shown: divide the path by $\alpha_t$ to get $\bar{\mathbf{x}} = \mathbf{x}_0 + \bar\sigma\boldsymbol\epsilon$, differentiate, take the posterior mean. Then show the step in the form readers will meet, $\mathbf{x}_s = \alpha_s\hat{\mathbf{x}}_0 + \sigma_s\hat{\boldsymbol\epsilon}$, and say that on the linear path it coincides with the Euler step in $t$ (true only there; on the VP path Euler in $t$ is a different method). Also: the bar now means two things, a square in $\bar\alpha_t = \alpha_t^2$ and a quotient in $\bar\sigma$ and $\bar{\mathbf{x}}$, in one section. Rename the quotient (for instance $\rho = \sigma_t/\alpha_t$, or Karras' own convention stated as the variance-exploding coordinates).

**GP-3** (moderate, explanation). Under "The Reverse SDE and the Ancestral Chain".
> A forward process $d\mathbf{x} = \mathbf{f}\, dt + g\, d\mathbf{W}$ is reversed, from $t = 1$ down to $0$, by

The main text has never written a stochastic differential equation; $d\mathbf{W}$, drift, "a Brownian motion run backward" and "Euler--Maruyama" are appendix vocabulary. This paragraph cannot be followed without `mdl-sdes.md`, against the author's rule. Nothing in the section's code depends on it, so either gloss the notation in one sentence using the forward step the reader has just seen ("read $d\mathbf{x} = \mathbf{f}\,dt + g\,d\mathbf{W}$ as the limit of steps $\mathbf{x} \leftarrow \mathbf{x} + \mathbf{f}\,\Delta t + g\sqrt{\Delta t}\,\mathbf{z}$; :eqref:`eq_diffusion-forward` is such a step with $\mathbf{f} = -\tfrac12\beta\mathbf{x}$, $g^2 = \beta$"), or mark the paragraph as the continuous-time view that can be skipped.

**GP-4** (minor, explanation). Same subsection.
> since on the VP path $\mathbf{u}_t = -\tfrac12 \beta(t)\, (\mathbf{x} + \nabla \log p_t)$

$\beta(t)$ is used here and in the next paragraph and never defined. The section gives only $\log\bar\alpha_t$. Add "$\beta(t) = \beta_{\min} + (\beta_{\max} - \beta_{\min})\,t$, so that $\log\bar\alpha_t = -\int_0^t \beta(r)\,dr$" where the schedule is introduced. The section also has $\beta_k$ (chain) and a bare $\beta$ (one step); say once that they are the same quantity on a grid.

**GP-5** (minor, correctness). Under "Signal, Noise and the Log-SNR Clock".
> it is the Ornstein--Uhlenbeck process of :numref:`sec_mdl-ornstein-uhlenbeck`

The appendix process has a constant rate $\theta$; the path here has a time-varying $\beta(t)$. Write "an Ornstein--Uhlenbeck process run on a clock that speeds up".

**GP-6** (moderate, pedagogy). Under "The Learned Predictor against the Optimal One".
> The loss column equals the floor plus the gap, to within the Monte Carlo error.

This checks the orthogonality identity stated in the line above the cell; it cannot fail except by sampling noise. The gap column is the useful part and fits in the loss-floor table. Suggested end of the section: keep "Samples from Both Samplers" as the experiment that answers the opening question, add the gap column to `gaussian-paths-the-loss-floor`, and keep "The Dictionary Checked" only as its last sentence ("No score was learned, and no velocity appears in the noise model's loss") plus one printed row, or move it to an exercise. The check "largest difference to the loop" in `gaussian-paths-the-probability-flow-ode-and-euler-s-method-1` is a unit test of the same kind; one clause is enough for it.

**GP-7** (moderate, pedagogy: code). Under "Four Targets, One Oracle".
> `make_pred` wraps a trained network as the pair $(\hat{\mathbf{x}}_0, \hat{\boldsymbol\epsilon})$ at any $(\alpha, \sigma)$, by querying it at the time of its own path with the same log-SNR, which `time_at` finds by bisection

The sentence is accurate and too compressed to follow on a first reading, and it describes plumbing (a 50-iteration bisection inside every sampler call) whose purpose, running any sampler on any network, the reader does not need until the dictionary check. The code is correct (I checked the rescaling $\mathbf{x}\,\sigma'/\sigma$ and the invariance of $\hat{\mathbf{x}}_0$ and $\hat{\boldsymbol\epsilon}$), but it does not teach. Fix: introduce `make_pred` where it is first needed, state its contract in one plain sentence ("a path-independent predictor: give it a noisy point and its $\alpha$ and $\sigma$, get $\hat{\mathbf{x}}_0$ and $\hat{\boldsymbol\epsilon}$"), and leave the mechanism to the docstrings. In the split of structural finding 1 this belongs to the sampling half.

**GP-8** (moderate, explanation). Under "Signal, Noise and the Log-SNR Clock".
> with $(\alpha_0, \sigma_0) = (1, 0)$ and $(\alpha_1, \sigma_1) = (0, 1)$, or nearly so

This definition excludes the path the reader has used so far ($\alpha = 1$, $\sigma$ up to 3 on the ladder) and the coordinates the section's own sampler works in. Name the third family here: "the *variance-exploding* (VE) path keeps $\alpha_t = 1$ and lets $\sigma_t$ grow until the data are negligible; the previous section and the ladder below use it." Then the $\bar\sigma$ coordinates of GP-2 can be introduced as "every path rescaled to the VE one", which is the idea.

**GP-9** (minor, correctness: prose against output). Same subsection.
> with $\bar\alpha = 0.079$ and $\lambda \approx -2.5$ against $\lambda = 0$ on the linear path

The cell prints $\alpha = 0.28$, not $\bar\alpha$; the next clause ("less than two percent of the signal against ten percent") then compares $\alpha$ values. Quote $\alpha = 0.28$ against $0.5$, the printed numbers.

**GP-10** (minor, thought). Under "Four Targets, One Oracle".
> A fifth target, $\mathbf{v} = \alpha_t \boldsymbol\epsilon - \sigma_t \mathbf{x}_0$, was introduced by

The heading counts four and the paragraph five, the Summary lists five. Also "oracle" is coined here for the posterior mean and doubles as a function name, and "dictionary" for the conversions is then used as a proper noun across three sections. Both are tolerable; define each once in italics at first use (the first "dictionary" is undefined: "by the dictionary they should hold the same posterior mean") and retitle the subsection "Equivalent Targets".

**GP-11** (minor, pedagogy). After the Summary table.
> Nothing in this section used the fact that $\mathbf{x}$ had two coordinates.

This is a second ending after the Summary, and the next section opens with the same sentence ("nothing in it used the number two"). Keep it in one place. The table `tab_diffusion_models` is cited 480 lines before it appears and lists rows the reader cannot read yet ($\mathbf{v}$ on the cosine path, masked diffusion); put it where it is first cited, restricted to the rows taught so far, or in the chapter's last section.

**GP-12** (minor, explanation). Code of `ode_sample`.
> xb = x0 + r_s * eps                    # the Euler (= DDIM) step

DDIM is named only in code comments here and defined in `sec_diffusion-samplers`. Either drop the name from the comments or add one clause in the prose ("the step known as DDIM, :numref:`sec_diffusion-samplers`").

**GP-13** (minor, language). Opening.
> Underneath the three routes by which the literature reached the answer (:numref:`chap_diffusion`) is one object:

A 55-word inverted sentence that lists four things as "one object". Split: "The literature reached the answer by three routes (:numref:`chap_diffusion`). All three build the same thing: a Gaussian path from the data at $t = 0$ to noise at $t = 1$, and a network that predicts the clean point along it. One regression trains the network, and a sampler integrates the differential equation that its prediction defines."

## Recurring patterns

- **Slogan headings on one template** (5): "One Path from Data to Noise", "One Regression", "Four Targets, One Oracle", the callout "One clock", and "One Trainer" in the next section. Use noun phrases (guide section 11): "The Path", "The Regression Target", "Equivalent Targets". "The Dictionary Checked" is a participle tag.
- **Semicolon chains** (22 in the body, against 9 in `denoising.md`): for example "`target` gives the per-example regression target of a kind on any path. For `'velocity'` it is the conditional velocity, with $\dot\alpha_t$ and $\dot\sigma_t$ from forward-mode differentiation (`rates`); `x0_eps` turns a prediction of any kind back into the pair". These are the sentences where the code inventory is compressed into the prose.
- **The letter $s$** is the component standard deviation, the earlier time in $s < t$, the cosine offset `s=0.008`, the code name for $\sigma$, and (bold) the score. $\sigma$ appears as $\sigma$, $\sigma_t$, $\tilde\sigma_i$, $\bar\sigma$ and $\sigma_{t|s}$. A five-line notation table at the start of the chapter would remove most of the load.
- **Three-question roadmaps** open the section and "One Regression" (see `denoising.md`).

## Code and experiments

- All stored outputs agree with the prose at the stated precision: schedule ($\beta_1 = 1.1 \cdot 10^{-4}$, $\beta_T = 0.0198$), ladder (weights within 0.03, control a third each), floors and training losses, one-step mean and spread, 10 and 100 steps, the ancestral chain, the gap table, the five sampler rows, the dictionary table.
- `gaussian-paths-the-learned-predictor-against-the-optimal-one`: confirmation of an identity (GP-6).
- `gaussian-paths-the-dictionary-checked`: confirmation; keep small (GP-6).
- "largest difference to the loop": check that cannot fail by algebra (GP-6).
- The time input: both `TimeMLP` networks receive $t \in [0, 1]$ through `time_embedding`, whose features are then nearly linear in $t$ (see DEN-5 in `denoising.md`). The gaps of $10^{-3}$ to the oracle show that this suffices on the mixture.
- Two networks trained for 40,000 steps each on a two-dimensional example: the cost is not stated. Say what it is on a CPU, since this section should run on a laptop.
- Training draws fresh points from the mixture at every step, whereas the previous section trained on a fixed set of 2,000. The switch is silent; one clause would tell the reader that memorization is therefore not at issue here (:exref:`ex_gaussian-paths-memorization` takes it up).
- No TensorFlow or MXNet material; no tab prose.

## Exercises

Ten problems, all substantive and several excellent (the ancestral step for Gaussian data, Gaussian reversal and step size, commitment time, same ratio same flow, ladder spacing; I checked the algebra of the first two and the count of 232 levels). The set is steep: five carry a difficulty mark and none is a direct first exercise on the section's main skill. Add one easy problem that asks the reader to convert a prediction between two targets by hand at one $(\alpha, \sigma)$ and to write the DDIM step from it. :exref:`ex_gaussian-paths-ncsn` (train an NCSN) is extended work on a method the chapter treats as history; it could move to the further-reading notebook. Attributions to CMU 10-799 and the claims about Biroli et al. and Bertrand et al. match what I know of those papers but were not verified.

## Still open from October

October: "$\alpha$ has three meanings, $\sigma$ four, and $x_0$ flips from data to noise". The flip is gone and the clock is fixed. $\alpha_t$ and $\bar\alpha_t$ are now two symbols for one schedule, but $\sigma$ and $s$ are still overloaded (see the pattern above). "DDPM first appears in section 7" is resolved: it is here, in section 2. The split of ownership with the appendix that October asked for has been made, with the exceptions GP-1 and GP-3.

## For the chapter review

- This section owns the path, the regression and its targets, the loss floor, and both basic samplers. Everything after it in the chapter reuses `GaussianPath`, `make_pred`, `ode_sample` and `ancestral_sample`.
- Against the appendix: the conversions, the callout and the two-point example are small and may stay in both. Two passages do not read without the appendix (GP-1, GP-3); both are cheap to repair. No long derivation is written out in both places.
- Suggested split (structural finding 1) would give the chapter nine sections; the first half could take the title "Gaussian Paths" and the second "Sampling by Integration", which also lets `samplers.md` start from a section that already owns DDIM.
- Suspected for the whole chapter: general-purpose `#@save` machinery explained in compressed prose; verification cells in runs of three; one-template headings.
