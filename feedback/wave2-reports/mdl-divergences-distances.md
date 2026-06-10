# Wave-2 report: chapter_mdl-information-theory/mdl-divergences-distances.md

## Structure (1304 lines; stub 368 → full section)
4 content `##` + Summary + Exercises + 4-tab Discussions + 7 slides. All 11 stub labels survive:
1. **What Is a Divergence? The f-Divergence Family** (`sec_mdl-what-is-a-divergence`) — axioms/metrics/3-family taxonomy; f-template (`sec_mdl-f-divergences`) with Prop. D_f≥0 via Jensen (`subsec_mdl-jensen`); generator markdown table + JS defined explicitly (eq_mdl-js-def, ≤ log 2); f''(1) local-χ² remark.
2. **Duality: The Variational View** — Fenchel conjugate callback (`subsec_mdl-convex-conjugate`), Prop. f-GAN bound + proof, tangent picture (`sec_mdl-f-gan-dual`); GAN=JS, KL conjugate → DV pointer to 26.3; fwd-vs-rev KL (`sec_mdl-fwd-vs-rev-kl`) with Prop. forward-KL=moment matching + proof.
3. **Metrics: TV, MMD, Optimal Transport** — Prop. TV=½ℓ₁+optimal event (proof); Prop. Pinsker (binary case + log-sum reduction, DPI deferred to 26.3) (`sec_mdl-tv-pinsker`); IPM/Müller, MMD mean-embedding derivation + unbiased estimator (`sec_mdl-ipm-mmd`); OT primal, KR duality stated + price-schedule intuition + WGAN/GP/spectral-norm, Prop. 1-D W₁=∫|F_P−F_Q| FULL proof via KR+Fubini, Sinkhorn paragraph + demo kept, W₂ → pointer to `sec_mdl-score-matching-diffusion-flow` (`sec_mdl-optimal-transport`).
4. **Scores: Fisher, Stein, Objective Map** — canonical data-score home, param-score disambiguation, mixture score, normalizer-blindness, Fisher div + Gaussian closed form, Hyvärinen stated/deferred (`sec_mdl-fisher-divergence`); Prop. Stein identity + proof via `subsec_mdl-integration-by-parts`, KSD coda (`sec_mdl-stein-discrepancy`); capstone **markdown table** 8 rows (`sec_mdl-divergence-objective-map`).

Code: 10 cells, one 4-tab imports (d2l + numpy + scipy), 9 untagged numpy/scipy, all seeded, nats; key numbers quoted in prose. New eqlabels: eq_mdl-{f-div-def, js-def, fenchel-young, f-gan-bound, tv-def, pinsker, ipm-def, mmd2, w1-primal, kr-dual, w1-cdf, score-def, fisher-div-def, stein-identity}.

## FIGURE SPECS (6 SVGs to generate via tools/gen_mdl_infotheory_figures.py; do NOT exist yet)
1. `img/mdl-it-divergence-taxonomy.svg` (fig_mdl-divergence-taxonomy): 3-region Venn (f-divergences / IPMs / optimal transport). Place: KL, reverse KL, χ², Hellinger², JS in f-only; **TV in f∩IPM**; MMD in IPM-only; **W₁ in IPM∩OT**; W₂ in OT-only. Distinguish metrics (TV, H, MMD, W₁, W₂) with a marker/legend note.
2. `img/mdl-it-f-div-generators.svg` (fig_mdl-f-div-generators): curves on u∈[0.05,3]: u·ln u, −ln u, (u−1)², (√u−1)², ½|u−1|, (u/2)ln u−((u+1)/2)ln((u+1)/2); all touch (1,0) — mark it; legend matches table row names; clip y≈[−0.6,2.5].
3. `img/mdl-it-f-gan-tangent-bound.svg` (fig_mdl-f-gan-tangent-bound): f(u)=u·ln u on (0,3]; 3–4 tangent lines; one highlighted tangent of slope t with y-intercept annotated −f*(t); annotation "f = sup of its tangents (Fenchel–Young)".
4. `img/mdl-it-tv-area.svg` (fig_mdl-tv-area): two overlapping 1-D densities (suggest N(−1,1), N(1.2,0.8²)); shade |p−q| between curves; mark A* = {p>q} as interval bar on x-axis; annotate TV = ½ shaded area.
5. `img/mdl-it-ot-transport-plan.svg` (fig_mdl-ot-transport-plan): two panels. Left: the in-text 6-atom pair (atoms 0..5, p=(.30,.20,.25,.10,.10,.05), q=(.05,.10,.10,.25,.20,.30)) as mirrored bars with monotone-plan arrows + the two step CDFs with area between shaded (=1.7). Right: vs separation d — JS(d)=ln 2 flat line vs W₁(d)=d line; annotate "no gradient"/"gradient ±1".
6. `img/mdl-it-score-field.svg` (fig_mdl-score-field): 2-D Gaussian mixture (e.g. w=(.7,.3), μ=(−1.5,−1),(1.5,1)) density contours + quiver of ∇log p on a coarse grid (arrows uphill). Chapter 27 stubs reuse this label — keep id exactly `fig_mdl-score-field`.

## BibTeX — 11 NEW keys used via :cite: (NOT added to d2l.bib; wave-2 must add)
```bibtex
@Article{Csiszar.1967, author={Csisz{\'a}r, Imre}, title={Information-type measures of difference of probability distributions and indirect observations}, journal={Studia Scientiarum Mathematicarum Hungarica}, volume={2}, pages={299--318}, year={1967}}
@Article{Muller.1997, author={M{\"u}ller, Alfred}, title={Integral probability metrics and their generating classes of functions}, journal={Advances in Applied Probability}, volume={29}, number={2}, pages={429--443}, year={1997}}
@Article{Hyvarinen.2005, author={Hyv{\"a}rinen, Aapo}, title={Estimation of non-normalized statistical models by score matching}, journal={Journal of Machine Learning Research}, volume={6}, pages={695--709}, year={2005}}
@Article{Gretton.Borgwardt.Rasch.ea.2012, author={Gretton, Arthur and Borgwardt, Karsten M. and Rasch, Malte J. and Sch{\"o}lkopf, Bernhard and Smola, Alexander}, title={A kernel two-sample test}, journal={Journal of Machine Learning Research}, volume={13}, pages={723--773}, year={2012}}
@InProceedings{Cuturi.2013, author={Cuturi, Marco}, title={Sinkhorn distances: lightspeed computation of optimal transport}, booktitle={Advances in Neural Information Processing Systems}, volume={26}, year={2013}}
@InProceedings{Li.Swersky.Zemel.2015, author={Li, Yujia and Swersky, Kevin and Zemel, Richard}, title={Generative moment matching networks}, booktitle={Proceedings of the 32nd International Conference on Machine Learning}, year={2015}}
@InProceedings{Liu.Lee.Jordan.2016, author={Liu, Qiang and Lee, Jason D. and Jordan, Michael I.}, title={A kernelized {S}tein discrepancy for goodness-of-fit tests}, booktitle={Proceedings of the 33rd International Conference on Machine Learning}, year={2016}}
@InProceedings{Nowozin.Cseke.Tomioka.2016, author={Nowozin, Sebastian and Cseke, Botond and Tomioka, Ryota}, title={f-{GAN}: training generative neural samplers using variational divergence minimization}, booktitle={Advances in Neural Information Processing Systems}, volume={29}, year={2016}}
@InProceedings{Arjovsky.Chintala.Bottou.2017, author={Arjovsky, Martin and Chintala, Soumith and Bottou, L{\'e}on}, title={Wasserstein generative adversarial networks}, booktitle={Proceedings of the 34th International Conference on Machine Learning}, year={2017}}
@InProceedings{Gulrajani.Ahmed.Arjovsky.ea.2017, author={Gulrajani, Ishaan and Ahmed, Faruk and Arjovsky, Martin and Dumoulin, Vincent and Courville, Aaron}, title={Improved training of {W}asserstein {GAN}s}, booktitle={Advances in Neural Information Processing Systems}, volume={30}, year={2017}}
@Article{Peyre.Cuturi.2019, author={Peyr{\'e}, Gabriel and Cuturi, Marco}, title={Computational optimal transport}, journal={Foundations and Trends in Machine Learning}, volume={11}, number={5--6}, pages={355--607}, year={2019}}
```
Existing keys used: Goodfellow.Pouget-Abadie.Mirza.ea.2014, Miyato.Kataoka.Koyama.ea.2018; cross-file :eqref:`eq_mdl-gaussian_kl` (26.1).

## Verification log
- `tools/lint_source.py <file>` and `--corpus <file>`: **exit 0** (note: lint does not actually validate cite keys against d2l.bib, so the 11 new keys don't trip it).
- All cells extracted from the committed file, run **cumulatively in order** via ipython (MPLBACKEND=Agg) under `.venv-pytorch`, `.venv-tensorflow`, `.venv-jax`, `.venv-mxnet`: identical outputs, ~3 s/framework. Key numbers (quoted in prose): KL 0.3961/0.3653 (matches 26.1), χ² 1.0133/0.8000, JS 0.0911 both routes; f-GAN optimal critic = exact 1.0133, perturbed 0.9678/0.3661/0.9600 all below; fwd fit (−0.800, 1.929)=moments, rev (−1.998, 0.603) KL 0.356 ≈ ln(1/0.7), 2nd optimum (+1.995, 0.607) 1.202 ≈ ln(1/0.3); Pinsker max ratio 0.9926 over 10k Dirichlet pairs, coins → 0.999999; MMD² +0.00054 vs +0.05890; W₁ CDF=LP=1.7000000000 (10 dp); Sinkhorn 1.7700/1.7000/1.7000 at ε=1/0.1/0.02; mixture score gap 3.5e−4, rescale 1.8e−12, Fisher 0.500000=closed form; Stein +0.0020/+0.0005.
- Preprocessor smoke test to a temp dir: tables, eqlabels, figure crossrefs, and :numref: inside table cells all convert cleanly (repo .qmd untouched).

## Deviations
1. **1304 lines vs ~900–1000 target** (coincidentally equal to sibling 26.1): overage is the 4-tab imports, 7-slide deck, 8 exercises, and the two requested full proofs. Nothing padded; trimming would cut mandated content.
2. Sinkhorn **kept** (paragraph + 12-line cell), not deferred.
3. Figures trimmed 9 → 6; fwd-vs-rev-KL realized as the computed d2l.plot; `fig_mdl-mmd-embedding` and a separate objective-map figure dropped (map is a markdown table per review risk #6).
4. Discussions links are placeholders (`/t/divergences`, 4 tabs) — thread IDs TBD at publication.
5. Pinsker's merging step proved inline via the log-sum inequality; full DPI deferred to 26.3 as instructed.
6. `subsec_mdl-jensen` / `subsec_mdl-convex-conjugate` confirmed present (parallel writer landed them in mdl-convexity.md), so all numrefs resolve today.
