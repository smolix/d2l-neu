# Report: 24.4 Numerical Stability and Conditioning (wave 2)

## Structure (1114 lines; stub was 182)
- `## Floating-Point Arithmetic` — A Number System with Gaps (fig fp-number-line; 4-tab finfo; untagged spacing/overflow) / Overflow, Underflow, and Mixed Precision (loss scaling, :cite:`Micikevicius.Narang.Alben.ea.2018`).
- `## Making Softmax and Cross-Entropy Safe` — shift-invariance Prop + naive-vs-stable demo / lse exactness + sandwich Prop + log-space demo / **Pass Logits, Not Probabilities**: the one genuinely 4-fw-divergent tabbed cell + per-framework `:begin_tab:` readings.
- `## Catastrophic Cancellation` — amplification factor, log1p demo / Welford **corrected** recursions (eq_mdl-opt-welford) + exactness Prop (induction) + 384-vs-1.000257 demo.
- `## Conditioning: One Number, Two Consequences` — **### Backward and Forward Error** (Higham; Prop fwd ≤ κ·bwd) / perturbation-bound Prop (leans on :numref:`subsec_mdl-condition-number`) + Hilbert digits-lost table with backward-error column / κ(AᵀA)=κ(A)² Prop + normal-eq-vs-SVD demo / ridge monotone-κ Prop (eq_mdl-opt-ridge-kappa) + joint κ/GD-iterations sweep (d2l.plot) + fig conditioning-ellipse.
- Summary, 8 Exercises, Discussions, 7 slides. 7 propositions, all with proofs.
- Errata fixed: bfloat16 ε = 2⁻⁷ printed and asserted `== 2.0**-7` in all 4 tabs; Welford recursions corrected per review; all §3.x purged. Anchor `sec_mdl-numerical-stability-conditioning` preserved.

## FIGURE SPECS (do not exist yet; includes already in the .md)
**img/mdl-opt-fp-number-line.svg** (`fig_mdl-opt-fp-number-line`), ~5.5×2.0:
single schematic number line. Ticks for representable floats across 4–5 binades
(e.g. [0.25,0.5), [0.5,1), [1,2), [2,4)), even spacing *within* a binade, gap
doubling at each power of two (annotate "gap doubles"). Arrow ε_mach between 1
and its right neighbor. Shaded left region below smallest normal: "subnormals →
underflow to 0". Axis break, then two overflow cliffs as dashed verticals:
fp16 at 65504, fp32 at 3.4e38, shaded "overflow → inf" beyond. House style.

**img/mdl-opt-conditioning-ellipse.svg** (`fig_mdl-opt-conditioning-ellipse`), ~5.5×2.6:
two panels, same axes. Left: contour ellipses of ½wᵀdiag(1, 1/16)w, title
"λ = 0:  κ = 16", elongated; optional GD zig-zag path (match mdl-la-condition
style). Right: ½wᵀ(diag(1,1/16)+λI)w with λ = 0.5, title "λ = 0.5:  κ ≈ 2.7",
near-circular; straighter GD path. Annotate axis lengths σ₁²+λ, σₙ²+λ.

## BibTeX (cited in text; NOT added to d2l.bib — wave-2 reconciliation)
```bibtex
@Book{Higham.2002, title={Accuracy and Stability of Numerical Algorithms},
  author={Higham, Nicholas J.}, edition={2nd}, publisher={SIAM}, year={2002}}
@Article{Goldberg.1991, title={What every computer scientist should know about
  floating-point arithmetic}, author={Goldberg, David}, journal={ACM Computing
  Surveys}, volume={23}, number={1}, pages={5--48}, year={1991}}
@InProceedings{Micikevicius.Narang.Alben.ea.2018, title={Mixed precision training},
  author={Micikevicius, Paulius and Narang, Sharan and Alben, Jonah and Diamos,
  Gregory and Elsen, Erich and Garcia, David and Ginsburg, Boris and Houston,
  Michael and Kuchaiev, Oleksii and Venkatesh, Ganesh and Wu, Hao},
  booktitle={International Conference on Learning Representations}, year={2018}}
@Article{Welford.1962, title={Note on a method for calculating corrected sums of
  squares and products}, author={Welford, B. P.}, journal={Technometrics},
  volume={4}, number={3}, pages={419--420}, year={1962}}
```
Existing keys used: Goodfellow.Bengio.Courville.2016.

## Verification log
- Cumulative in-order cell execution (untagged ⇒ all four): `ALL CELLS PASSED`
  in .venv-pytorch, .venv-tensorflow, .venv-jax, .venv-mxnet. Slowest cell 0.3 s.
- Cross-entropy tab outputs (quoted in `:begin_tab:` prose, all verified):
  pytorch/mxnet 103→103.2789, 104→inf; jax 103→inf (no subnormal stop);
  tensorflow all rows 16.1181 (Keras clips p to [1e-7, 1−1e-7] — silent).
- finfo tab prints bfloat16 eps 7.812e-03 and `== 2^-7: True` in all 4 tabs
  (mxnet via numpy bit-level bf16 round-to-nearest-even emulation).
- `tools/lint_source.py` → exit 0, no warnings. All 11 :numref: targets exist.
- Tripwire greps clean: no closing-$-then-digit, no ] in captions, $$/:eqlabel:
  placement correct, no "Planned", no stale §3.x.

## Deviations
1. 1114 lines vs ~750–950 target (sibling duality section is 1145); everything
   requested is in, nothing padded — flagging rather than cutting propositions.
2. 7 propositions vs ≈5: added lse sandwich + Welford exactness (both short).
3. Untagged Hilbert/normal-eq/Welford printouts vary in trailing digits across
   NumPy/LAPACK builds (mxnet venv has numpy 1.x: naive variance −256 vs 384).
   Prose hedged to be true in every tab; the sign-flip is taught explicitly.
4. Discussions link uses slug URL style of the sibling (topic must be created).
5. `import ml_dtypes` (TF tab) and `import optax` (JAX tab) added to imports
   cell — both already ship in the respective venvs.
