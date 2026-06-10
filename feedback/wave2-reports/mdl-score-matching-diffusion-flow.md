# Wave-2 report: 27.4 Score Matching, Diffusion, and Flow Matching

## Structure (1546 lines; lint clean)
5 top-level `##` + Summary + Exercises (8) + 7 slides. All 10 stub labels survive:
`sec_mdl-score-matching-diffusion-flow` (anchor) · `-score-matching` (## Learning the Score: Z-free scores, Hyvärinen w/ 1-D parts proof + "implicit SM" cost remark; **regression-to-the-conditional-mean lemma** stated once and reused; `-denoising-score-matching` as ### w/ Vincent proof via the lemma + Tweedie + loss-floor corollaries; 1-D DSM code) · `-score-based-generative-modeling` (## Score-Based Diffusion: VE/VP, λ(t)-DSM, **two-clocks time-axis callout**; `-ddpm-discretized-sde` as ### w/ 3 props (EM-first-order, ᾱ-marginal by induction, ε-loss = DSM·(1−ᾱ)) + ELBO paragraph (Luo 2022, `sec_mdl-latent-em-elbo`) + chain-vs-formula cell; ### Langevin w/ FP stationarity prop + warm/cold-start cell + annealed/PC remark; ### DDIM boxed derivation + η-family + PF-ODE remark; ### Guidance: Bayes identity, classifier guidance as tilt, CFG extrapolation w/ normalizability caveat) · `-flow-matching` (## FM and RF: marginal-velocity-generates-marginal-path prop; **CFM theorem = the lemma again**, full proof; `-rectified-flow` as ### w/ crossing/curvature discussion + reflow; 4-tab CFM train + 4-tab Euler sampler + untagged 1/2/8/32-step panels) · `-ot-connection` (## OT and Straightness: W₂ self-contained, Benamou–Brenier w/ Jensen lower-bound proof sketch (`subsec_mdl-jensen`), straightness = Jensen equality, minibatch OT-CFM remark + honesty caveat) · `-sampling-learned-dynamics` (## Sampling: ODE-vs-SDE, untagged energy-distance-vs-steps cell, untagged Euler-vs-Heun on exact-score VP PF-ODE + EDM remark + out-of-scope pointers; `-unifying-table` as ### **markdown table**: DDPM/score-SDE/PF-ODE/DDIM/FM-RF × object/loss/sampler/stochasticity).
Code: 21 cells, 11 ids; exactly TWO training loops (untagged NumPy hand-backprop DSM net; 4-tab CFM). All else untagged NumPy, seeded, deterministic.

## Figure specs (do not exist yet; no includes added to the .md — insert at the marked spots when generated)
1. `img/mdl-dyn-noising-denoising.svg` — 2×3 snapshots of a 2-D two-moons cloud under VP noising (t = 0, T/4, T, left→right, top row) and reverse denoising (right→left, bottom row) with short score arrows on the bottom row. Place: §"From One Noise Level to All of Them", after the VE/VP bullets.
2. `img/mdl-dyn-fm-paths.svg` — straight conditional segments source→target with two visible crossings, overlaid/beside the curved non-crossing marginal-flow trajectories the posterior mean induces. Place: §"Rectified Flow and Straight Paths", at the crossings paragraph (ties to eq_mdl-marginal-velocity).
3. `img/mdl-dyn-time-conventions.svg` — two time axes: diffusion 0(data)→T(noise), sampling arrow leftward; FM 0(noise)→1(data), sampling arrow rightward; density glyphs at ends. Place: beside the "Two clocks" callout.
Sampling-quality-vs-steps is a computed d2l.plot (as mandated), not a figure.

## Verification (every cell, cumulative, per venv; untagged outputs byte-identical across frameworks)
- pytorch: all 10 cells pass; CFM train 1.8 s; whole notebook 9.5 s. Panels figure inspected (moons emerge 1→2→8→32 steps).
- tensorflow: pass; CFM train 1.7 s (tf.function); total 10.2 s.
- jax: pass; CFM train 1.6 s (jit + lax.scan, as mandated); total 10.9 s. Panels inspected.
- mxnet: pass; CFM train 3.3 s (gluon, same algorithm/hyperparams); total 13.4 s.
- DSM cell 2.1 s; prints loss 2.036 vs analytic floor 2.038, max score error 0.199 — quoted in prose.
- Heun-vs-Euler: order-1/order-2 slopes confirmed; Heun@20 steps (40 NFE) 0.0085 < Euler@40 0.0099.
- `tools/lint_source.py` exit 0; `d2l_preprocess.py` smoke test: all eqlabels attach (incl. inside the DDIM callout), unifying table converts intact (norms written `\lVert…\rVert` to avoid pipe collision).

## BibTeX to add (text uses existing `ho2020denoising`, `song2021score`; 16 new keys)
```bibtex
@article{Hyvarinen.2005, author={Hyv{\"a}rinen, Aapo}, title={Estimation of Non-Normalized Statistical Models by Score Matching}, journal={Journal of Machine Learning Research}, volume={6}, pages={695--709}, year={2005}}
@article{Vincent.2011, author={Vincent, Pascal}, title={A Connection Between Score Matching and Denoising Autoencoders}, journal={Neural Computation}, volume={23}, number={7}, pages={1661--1674}, year={2011}}
@inproceedings{Song.Ermon.2019, author={Song, Yang and Ermon, Stefano}, title={Generative Modeling by Estimating Gradients of the Data Distribution}, booktitle={Advances in Neural Information Processing Systems}, year={2019}}
@inproceedings{Song.Meng.Ermon.2020, author={Song, Jiaming and Meng, Chenlin and Ermon, Stefano}, title={Denoising Diffusion Implicit Models}, booktitle={International Conference on Learning Representations}, year={2021}}
@inproceedings{Dhariwal.Nichol.2021, author={Dhariwal, Prafulla and Nichol, Alexander}, title={Diffusion Models Beat {GANs} on Image Synthesis}, booktitle={Advances in Neural Information Processing Systems}, year={2021}}
@article{Ho.Salimans.2022, author={Ho, Jonathan and Salimans, Tim}, title={Classifier-Free Diffusion Guidance}, journal={arXiv preprint arXiv:2207.12598}, year={2022}}
@inproceedings{Lipman.Chen.BenHamu.ea.2022, author={Lipman, Yaron and Chen, Ricky T. Q. and Ben-Hamu, Heli and Nickel, Maximilian and Le, Matt}, title={Flow Matching for Generative Modeling}, booktitle={International Conference on Learning Representations}, year={2023}}
@inproceedings{Liu.Gong.Liu.2022, author={Liu, Xingchao and Gong, Chengyue and Liu, Qiang}, title={Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow}, booktitle={International Conference on Learning Representations}, year={2023}}
@article{Tong.Fatras.Malkin.ea.2023, author={Tong, Alexander and Fatras, Kilian and Malkin, Nikolay and Huguet, Guillaume and Zhang, Yanlei and Rector-Brooks, Jarrid and Wolf, Guy and Bengio, Yoshua}, title={Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport}, journal={Transactions on Machine Learning Research}, year={2024}}
@inproceedings{Pooladian.BenHamu.DomingoEnrich.ea.2023, author={Pooladian, Aram-Alexandre and Ben-Hamu, Heli and Domingo-Enrich, Carles and Amos, Brandon and Lipman, Yaron and Chen, Ricky T. Q.}, title={Multisample Flow Matching: Straightening Flows with Minibatch Couplings}, booktitle={International Conference on Machine Learning}, year={2023}}
@inproceedings{Karras.Aittala.Aila.ea.2022, author={Karras, Tero and Aittala, Miika and Aila, Timo and Laine, Samuli}, title={Elucidating the Design Space of Diffusion-Based Generative Models}, booktitle={Advances in Neural Information Processing Systems}, year={2022}}
@article{Benamou.Brenier.2000, author={Benamou, Jean-David and Brenier, Yann}, title={A Computational Fluid Mechanics Solution to the {Monge}--{Kantorovich} Mass Transfer Problem}, journal={Numerische Mathematik}, volume={84}, number={3}, pages={375--393}, year={2000}}
@article{Luo.2022, author={Luo, Calvin}, title={Understanding Diffusion Models: A Unified Perspective}, journal={arXiv preprint arXiv:2208.11970}, year={2022}}
@article{Anderson.1982, author={Anderson, Brian D. O.}, title={Reverse-Time Diffusion Equation Models}, journal={Stochastic Processes and their Applications}, volume={12}, number={3}, pages={313--326}, year={1982}}
@inproceedings{SohlDickstein.Weiss.Maheswaranathan.ea.2015, author={Sohl-Dickstein, Jascha and Weiss, Eric and Maheswaranathan, Niru and Ganguli, Surya}, title={Deep Unsupervised Learning Using Nonequilibrium Thermodynamics}, booktitle={International Conference on Machine Learning}, year={2015}}
@article{Albergo.Boffi.VandenEijnden.2023, author={Albergo, Michael S. and Boffi, Nicholas M. and Vanden-Eijnden, Eric}, title={Stochastic Interpolants: A Unifying Framework for Flows and Diffusions}, journal={arXiv preprint arXiv:2303.08797}, year={2023}}
```
Note: `Anderson.1982` may also arrive from the 27.3 writer — dedupe centrally.

## Deviations
- **Length 1546 vs ~950–1100 target.** The mandated additions (Langevin+PC, DDIM box, guidance), 8 framework-tab code blocks, and 7 full proof blocks don't fit 1100 at exemplar prose density; I prioritized completeness + the 5/5/5 bar. Trimmed where possible (sampler plotting deduped into one untagged cell; steps-quality is one untagged cell, not 4 tabs).
- **DSM training loop is untagged NumPy** (hand-written backprop + Adam, ~25 lines) rather than 4-tab: identical deterministic printout in all four notebooks, and it teaches backprop; the 4-tab budget is spent on the CFM demo as mandated. MXNet did **not** need to simplify — all four tabs run the same algorithm/hyperparameters (4000 Adam steps, batch 256, 3→64→64→2 tanh).
- No figure includes in the .md (no SVGs exist; specs above). Stub's planned figs (`fig_mdl-dsm-target`, `fig_mdl-forward-reverse-strip`, …) superseded by the 3 specs.
- Cross-refs into parallel work, per instructions: `subsec_mdl-jensen` (24.2), `sec_mdl-fisher-divergence` / `sec_mdl-optimal-transport` / `sec_mdl-ipm-mmd` (ch. 26 stubs), `sec_mdl-fokker-planck` etc. (27.3). §6.x pseudo-numbering fully purged; all plain-text cites converted to :cite: keys.
