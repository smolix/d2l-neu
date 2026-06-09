# MXNet advanced-indexing strictness — introduced, then reverted (post-mortem)

**Status:** RESOLVED — the strictness was reverted upstream; d2l carries no
workaround. This note is kept for the *next wheel rebuild*: it records a
footgun that can resurface and the discipline that caught it.

**Timeline (all 2026-06-08):**

| Wheel | Behavior | d2l |
|---|---|---|
| `…20260529.3` (and earlier) | permissive: float index arrays coerced to int, cross-context index arrays auto-moved | plain float-index idioms work |
| `…20260607.1` | **strict**: float index → `IndexError`; cpu-index-on-gpu-tensor → context error | broke 10/131 notebooks → worked around in commit `80d0b12` (cast to int + `ctx=`) |
| `…20260608.1` | **permissive again** (PR #51 "Restore permissive float-index coercion") | workaround reverted; back to plain float-index idioms |

So the `80d0b12` workaround was live for exactly one wheel. The maintainer
restored the old behavior, so casting index arrays to int / pinning their
context is **no longer required**. (It's still *harmless* and arguably cleaner,
but we reverted it to keep the source matching upstream d2l-en.)

## The footgun (still latent — watch on every wheel bump)

mxnet's `np.array([...])` and `np.arange(...)` default to **float32** (mxnet's
global default dtype), *even for all-integer input* — unlike numpy/torch/jax
which infer an integer dtype. And `np.arange(...)` allocates on **CPU** by
default. So the idiomatic way to build an index array yields a *float, CPU*
array. As long as the wheel coerces (float→int) and auto-moves (cpu→gpu) during
advanced indexing, this is invisible. If a future wheel re-tightens either
behavior, the same ~10 notebooks break again. The two error signatures to
recognize:

```
IndexError: arrays used as indices must be of integer type
    # mxnet/ndarray/ndarray.py → _advanced_index_to_array (~line 1157)
MXNetError: ... require all inputs live on the same context  (2 vs. 1)
    # multi-axis index arrays get stack()ed; only bites the GPU path
```

The integer-dtype one trips first and *masks* the context one — the context
error only appears once indices are already int (which is why `bert`'s CPU demo
passed but `bert-pretraining`'s GPU training failed in the strict wheel).

If strictness ever returns, the fix idioms are:
`d2l.tensor(indices, dtype=d2l.int32)`, `x.astype('int32')`,
`np.arange(n, dtype='int32', ctx=X.ctx)`. Affected sites (for reference):
`cross_entropy` labels (mdl-information-theory), softmax-scratch demo labels,
`TokenEmbedding.__getitem__` (shared `#@save`), `SyntheticRegressionData`
batch indices, `seqrec` sort index, `MaskLM.forward` batch_idx.

## Diagnosis discipline (the durable lesson)

This is the part that generalizes beyond this one episode:

1. **Re-run the *full* mxnet notebook set on every wheel bump.** A wheel can
   silently change index/dtype/context behavior. `make venv-mxnet` then
   `make run-notebooks-mxnet`.
2. **Serial-re-check before blaming contention.** Under the *contended* batch
   run, the failures threw misleading secondary errors — `MXGetGPUCount` and
   `Build with USE_OPENCV=1 for image resize operator` (the documented
   ulimit/OpenCV symptom, see [mxnet-runtime-diagnostics.md](mxnet-runtime-diagnostics.md)).
   Re-running each failure **one at a time, no concurrency** showed all 10 were
   the *same deterministic* `IndexError`. Always serial-re-check first.

## Other wheel-bump checklist items (carried forward)

- **cuDNN match.** `…20260607.1` was compiled against cuDNN 9.23 while pip
  ships `nvidia-cudnn-cu13` 9.22 (ABI-compatible) → a `cuDNN lib mismatch`
  warning that gets captured into outputs. Masked with
  `MXNET_CUDNN_LIB_CHECKING=0` in `EXTRA_ENV_mxnet` (Makefile). `…20260608.1`
  showed no such warning in our run; the env var is harmless either way, kept.
- **Provenance.** Public version stays `2.0.0` (the `+cu13.bw.*` build tag is
  stripped for freshness keying), so a bump does **not** force a store
  re-capture — re-capture only what you re-run.
- **macOS asset.** `…20260608.1` shipped **linux x86_64 only** (no macOS
  arm64 asset), so `pyproject.toml`'s darwin mxnet line stays on `…20260529.3`.
  macOS is render-only (no notebook execution), so this is cosmetic.
