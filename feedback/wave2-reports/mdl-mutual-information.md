# Wave-2 report: chapter_mdl-information-theory/mdl-mutual-information.md

## Structure (1483 lines: 1014 prose, 384 code, 85 slides)
4 top-level `##` + Summary + Exercises (10) + Discussions + 7 slides. All 10 stub labels survive
(3 demoted to `###`: mi-nonlinear-correlation, infonce, mi-limits). Legacy eqlabels
`eq_mdl-{joint_ent,cond_ent,mut_ent,pmi}_def` reused; 9 new eqlabels; nats throughout.
1. **Definitions and Properties** (`sec_mdl-mi-recap`): joint/cond. entropy + chain-rule Proposition;
   MI as KL(joint‖product) + properties Proposition; Gaussian anchor I=−½log(1−ρ²) derived; PMI +
   corpus bigram demo (replaces "Amazon on fire"); nonlinear-correlation + invariance Proposition;
   conditional MI + chain-rule Proposition + free-side-information Corollary; **DPI Proposition**
   (two-way chain-rule proof, Markov picture, 3 readings) + numeric BSC-chain verification.
2. **Why Measuring MI Is Hard** (`sec_mdl-mi-hard`): binning curse; McAllester–Stratos log N theorem
   (informal + rare-event intuition); perfect-critic (true density ratio) saturation plot, N∈{16,128,1024}.
3. **Variational Bounds and InfoNCE**: BA, DV (Gibbs-distribution proof), NWJ Propositions; InfoNCE
   defined as categorical CE (callback `subsec_mdl-nll-crossentropy`); rigorous bound proof (uses the
   §1 corollary: negatives are free → BA on the bag); perfect-critic DV/NWJ/NCE table (variance
   explosion at I=6: DV 23±57, NWJ 1±12, NCE 4.61±0.05); 4-tab InfoNCE training cell.
4. **IB and the Limits of MI**: IB Lagrangian (DPI makes it well-posed); closed-form Gaussian-bottleneck
   information plane with β-sweep + collapse below β*=1/ρ²; Saxe/Shwartz-Ziv debate (honest); **Fano
   Proposition** (proof uses DPI; k=1000/5% → 6.36 nats → N>581 ties §2 and §4); `sec_mdl-mi-limits`
   guidance (Tschannen: bound tightness ≠ representation quality).
Legacy 26.1 bugs NOT transcribed: valid joint [[0.1,0.4],[0.2,0.3]], marginals computed FROM joint
(I=0.0242 nats, hand-verified); every printed number re-derived.

## Figure specs (NOT created; no dangling refs in text)
- **mdl-it-infonce-pos-neg** (`fig_mdl-infonce-pos-neg`, after eq_mdl-infonce_def): left, anchor x with
  highlighted positive y⁺ and N−1 gray negatives in a 2-D embedding sketch, critic arrows f(x,y_j);
  right, softmax-over-scores bar chart with the positive bar accented; annotate "I ≥ log N − L_NCE ≤ log N".
- **mdl-it-ib-tradeoff** (`fig_mdl-ib-tradeoff`, near eq_mdl-ib_lagrangian): information plane I(Y;Z) vs
  I(X;Z); concave frontier from origin saturating at dashed I(X;Y)=−½log(1−ρ²) ceiling (ρ=0.9); shade
  infeasible region above; β-tangent operating points incl. collapse for β ≤ 1/ρ². Curve = the closed
  form in cell `mutual-information-ib-plane`. Both via tools/gen_mdl_infotheory_figures.py house style.
- Existing `mdl-it-mi-overlap.svg`, `mdl-it-mi-variational-bounds.svg` kept with stub captions.

## BibTeX (14 new keys; radford2021learning already in d2l.bib — do not re-add)
```bibtex
@inproceedings{Tishby.Pereira.Bialek.1999, title={The information bottleneck method}, author={Tishby, Naftali and Pereira, Fernando C and Bialek, William}, booktitle={Proceedings of the 37th Allerton Conference on Communication, Control, and Computing}, pages={368--377}, year={1999}}
@inproceedings{Barber.Agakov.2003, title={The {IM} algorithm: a variational approach to information maximization}, author={Barber, David and Agakov, Felix}, booktitle={Advances in Neural Information Processing Systems}, volume={16}, year={2003}}
@article{Chechik.Globerson.Tishby.ea.2005, title={Information bottleneck for {G}aussian variables}, author={Chechik, Gal and Globerson, Amir and Tishby, Naftali and Weiss, Yair}, journal={Journal of Machine Learning Research}, volume={6}, pages={165--188}, year={2005}}
@article{Nguyen.Wainwright.Jordan.2010, title={Estimating divergence functionals and the likelihood ratio by convex risk minimization}, author={Nguyen, XuanLong and Wainwright, Martin J and Jordan, Michael I}, journal={IEEE Transactions on Information Theory}, volume={56}, number={11}, pages={5847--5861}, year={2010}}
@article{Donsker.Varadhan.1983, title={Asymptotic evaluation of certain {M}arkov process expectations for large time. {IV}}, author={Donsker, Monroe D and Varadhan, SR Srinivasa}, journal={Communications on Pure and Applied Mathematics}, volume={36}, number={2}, pages={183--212}, year={1983}}
@inproceedings{Belghazi.Baratin.Rajeshwar.ea.2018, title={Mutual information neural estimation}, author={Belghazi, Mohamed Ishmael and Baratin, Aristide and Rajeshwar, Sai and Ozair, Sherjil and Bengio, Yoshua and Courville, Aaron and Hjelm, Devon}, booktitle={International Conference on Machine Learning}, pages={531--540}, year={2018}, organization={PMLR}}
@article{Oord.Li.Vinyals.2018, title={Representation learning with contrastive predictive coding}, author={van den Oord, Aaron and Li, Yazhe and Vinyals, Oriol}, journal={ArXiv:1807.03748}, year={2018}, url={https://arxiv.org/abs/1807.03748}}
@inproceedings{Poole.Oord.Alemi.ea.2019, title={On variational bounds of mutual information}, author={Poole, Ben and Ozair, Sherjil and van den Oord, Aaron and Alemi, Alexander A and Tucker, George}, booktitle={International Conference on Machine Learning}, pages={5171--5180}, year={2019}, organization={PMLR}}
@inproceedings{McAllester.Stratos.2020, title={Formal limitations on the measurement of mutual information}, author={McAllester, David and Stratos, Karl}, booktitle={International Conference on Artificial Intelligence and Statistics}, pages={875--884}, year={2020}, organization={PMLR}}
@inproceedings{Alemi.Fischer.Dillon.ea.2017, title={Deep variational information bottleneck}, author={Alemi, Alexander A and Fischer, Ian and Dillon, Joshua V and Murphy, Kevin}, booktitle={International Conference on Learning Representations}, year={2017}}
@inproceedings{Saxe.Bansal.Dapello.ea.2018, title={On the information bottleneck theory of deep learning}, author={Saxe, Andrew M and Bansal, Yamini and Dapello, Joel and Advani, Madhu and Kolchinsky, Artemy and Tracey, Brendan D and Cox, David D}, booktitle={International Conference on Learning Representations}, year={2018}}
@article{Shwartz-Ziv.Tishby.2017, title={Opening the black box of deep neural networks via information}, author={Shwartz-Ziv, Ravid and Tishby, Naftali}, journal={ArXiv:1703.00810}, year={2017}, url={https://arxiv.org/abs/1703.00810}}
@article{Church.Hanks.1990, title={Word association norms, mutual information, and lexicography}, author={Church, Kenneth and Hanks, Patrick}, journal={Computational Linguistics}, volume={16}, number={1}, pages={22--29}, year={1990}}
@inproceedings{Tschannen.Djolonga.Rubenstein.ea.2020, title={On mutual information maximization for representation learning}, author={Tschannen, Michael and Djolonga, Josip and Rubenstein, Paul K and Gelly, Sylvain and Lucic, Mario}, booktitle={International Conference on Learning Representations}, year={2020}}
```

## Verification log
All 12 cell groups executed cumulatively, in document order, per venv (matplotlib Agg, magics stripped):
- pytorch: ALL OK; training cell 1.7 s (bounds 0.570/1.407/1.762/1.888 vs ln N caps, true 1.959).
- tensorflow: ALL OK; training 6.3 s (0.570/1.409/1.766/1.893). jax: ALL OK; 5.4 s (0.569/1.405/1.757/1.882).
- mxnet: ALL OK; training 14.9 s (0.570/1.408/1.764/1.891). Untagged-cell outputs byte-identical in all 4 venvs
  (shared seeded onp). Slowest untagged cell: log-n-ceiling ~4 s. Everything ≪ 60 s.
- `tools/lint_source.py` (file): exit 0, zero issues; `--corpus`: zero issues for this file.
- Prose-quoted numbers re-checked against executed outputs (binned-MI bias 1.78 vs 1.96; DPI 0.368→0.120;
  Fano 6.364/581; perfect-critic table; β-sweep points).

## Deviations
1. **Length 1483 > target 800–1000**: DPI/cond-MI/Fano additions + two complete 4-tab groups; sibling
   rewritten 26.1 is 1305 lines. Trimming would cut review-mandated content.
2. **VIB β-sweep → closed-form Gaussian-bottleneck demo** (explicitly allowed): exact, fast, verifiable;
   a second 4-tab training cell (reparam-encoder VIB in gluon et al.) judged not worth the budget. VIB
   covered in prose with Alemi cite.
3. **MXNet training tab = full InfoNCE, no simplified objective.** Divergence from other tabs: hand-rolled
   stable logsumexp + diagonal via `(f * eye).sum(axis=1)` (mx.np lacks those conveniences); ~15 s.
4. Stub's 7 planned `##` regrouped into 4 `##` (house 3–5 rule); all labels preserved.
5. Discussions block kept as stub placeholder (pytorch-only generic link) pending thread IDs.
6. Cell IDs use `mutual-information-` prefix (matches 26.1's `information-theory-` convention; the
   add_cell_ids derivation would say `mdl-mutual-information-` — accepted, lint emits no warning).
