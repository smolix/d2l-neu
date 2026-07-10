# Numerics: Dtypes and Mixed Precision
:label:`sec_numerics_v2`

> **Role.** Entirely new. A tensor has a *dtype* as well as a shape and a
> device, and since ~2020 the dtype column has become as important as the
> device column: essentially all serious training now runs in bf16/fp16
> mixed precision, and the planned transformer/generative chapters need
> this vocabulary established once, here. The floating-point *format*
> basics (sign/exponent/mantissa, fp64/fp32) exist in the numerical-
> stability appendix (:numref:`sec_mdl-numerics`); this section is the
> practical builder's view.

## The Dtype Zoo **[NEW]**

*Topics.* fp32 as the historical default; **tf32** (what Ampere+ GPUs
silently do to fp32 matmuls, and the one-line switch controlling it);
**fp16** vs **bf16** — same 16 bits, opposite trade (mantissa vs exponent),
why bf16's fp32-sized exponent range made it the modern default; int8/fp8
as the inference/production frontier (two sentences, forward pointer only).
Range/precision table; what overflows where (fp16 max ≈ 65504 meets a loss
of 10⁵).

*Code (PyTorch).* Create the same tensor in each dtype; print
`torch.finfo` for each; demonstrate one honest failure — square a large
fp16 value → `inf`, same value fine in bf16. Check
`torch.backends.cuda.matmul.allow_tf32` and time an fp32 matmul with it on
vs off *(numbers land in the committed output store, captured on the GPU
box)*.

## Dtype Rules: Promotion, Parameters, and Casts **[NEW]**

*Topics.* Type-promotion rules in one paragraph (when mixing dtypes
upcasts, when it errors); `module.to(dtype)` vs `tensor.to(dtype)`; the
distinction between *casting the model* (all parameters become bf16 — fine
for inference, risky for training) and *mixed precision* (fp32 master
weights, low-precision compute) — the single most common confusion in
practice, stated as a rule.

*Code (PyTorch).* Cast the chapter's MLP to bf16, run inference, inspect
parameter dtypes; show the memory halving with the byte-accounting helper
from :numref:`sec_parameters_v2` — the two sections' arithmetic composes.

## Mixed-Precision Training **[NEW]**

*Topics.* The `autocast` pattern: forward + loss in low precision where
safe (matmuls) and fp32 where not (reductions, norms) — the framework
maintains the per-op allowlist so you don't have to. fp16's extra
requirement: **loss scaling** (`GradScaler`) and why bf16 does not need it
(exponent range). What stays fp32: master weights, optimizer state — connect
back to the memory multiplier of :numref:`sec_parameters_v2`. The complete
modern recipe is five lines around an existing training loop; show it on
the book's own `Trainer` loop (as a local demonstration — whether the d2l
`Trainer` itself grows an `amp=True` flag is a rewrite-time decision,
flagged for review).

*Code (PyTorch).*

```python
scaler = torch.amp.GradScaler()
for X, y in loader:
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss = F.cross_entropy(net(X), y)
    scaler.scale(loss).backward()
    scaler.step(opt); scaler.update(); opt.zero_grad()
```

Train the Fashion-MNIST MLP two ways (fp32 vs bf16 autocast); report
time/epoch and final accuracy — same accuracy, measurably faster, which is
the entire pitch.

## When Numerics Bite **[NEW, short]**

*Topics.* A field guide, three short entries: (i) loss goes `NaN` — check
for fp16 overflow before suspecting the learning rate (and the `logsumexp`
trick already met in softmax, now named as the general pattern);
(ii) accumulate in higher precision — long sums (means over large batches)
in low precision drift; (iii) reproducibility across dtypes is not expected
— tf32/bf16 results differ from fp32 in the last bits by design (forward
pointer to :numref:`sec_repro_v2`).

*Code (PyTorch).* Sum 10⁷ small fp16 values naively vs via fp32
accumulation; the drift is visible in the third digit — one cell, one
lesson.

## Summary and Exercises

*Exercises (sketch).* (1) Redo the memory arithmetic of
:numref:`sec_parameters_v2` for bf16 weights + fp32 Adam state; which
term now dominates? (2) Find the smallest model/batch where autocast is
*slower* than fp32 (overhead vs savings). (3) `torch.finfo(torch.float8_e4m3fn)`
— read the fields, compare with fp16/bf16, and explain the name. (4) Why do
normalization layers stay in fp32 under autocast? Test what happens to
RMSNorm (:numref:`sec_custom_layers_v2`) if you force it to fp16 on inputs
with std 10².

> **Downstream constraints.** None existing (new section). Establishes
> vocabulary (bf16, autocast, master weights) that planned transformer/
> generative chapters should *assume* rather than re-teach.

## Framework Coverage

*The most divergent section of the chapter — by design each tab teaches its
framework's honest mixed-precision story rather than a forced translation.*

- **JAX** — coverage via a *better* mechanism: `param_dtype=` (storage) vs
  `dtype=` (compute) as two always-visible constructor args (verified:
  fp32 params, bf16 outputs) — the "casting vs mixed precision" rule is
  *structural* here, no autocast context or invisible allowlist. tf32 via
  `jax.default_matmul_precision` (verified); fp8 dtypes present with
  working `finfo` (verified). **One skip: no `GradScaler` equivalent in
  optax** (verified absent) → the JAX tab teaches bf16-only and says so —
  honest 2026 practice anyway.
- **TensorFlow** — full coverage: `keras.mixed_precision.set_global_policy
  ('mixed_bfloat16')` verified (compute bf16 / variables fp32 — exactly
  the master-weights pattern), `LossScaleOptimizer` present for fp16.
  Bonus teaching angle: TF is dtype-*strict* — mixing fp32/fp16 in one op
  raises immediately (verified), a sharper hook for dtype discipline than
  promotion rules. **One trim: no native fp8 dtype** in TF 2.21 (only via
  `ml_dtypes`, unintegrated) → forward-pointer prose only.
- **MXNet — reduced variant (decision needed).** Lead with the verified
  low-ceremony recipe: `net.cast('float16')` + `Trainer(...,
  {'multi_precision': True})` (fp32 master weights — real, in source). A
  full `mxnet.amp` module exists in the pinned wheel (fp16 *and* bf16
  targets) but has **zero corpus evidence** of ever running under it —
  demote to a pointer unless the GPU box validates it. tf32: SKIP (no user
  switch exists). fp8: SKIP (dtype absent). bf16 dtype exists in source
  [UNVERIFIED on GPU]. The mxnet tab of this section is therefore
  ~"fp16 + master weights, everything else marked not-available" — honest,
  and consistent with the archived status.
