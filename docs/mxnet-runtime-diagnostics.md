# MXNet Runtime Diagnostics

## Latest status (2026-06-06 — wheel `2.0.0+cu13.bw.20260529.3`)

This supersedes the 2026-05-19 snapshot below (which was wheel `…20260517`). The
custom MXNet wheel has been rebuilt twice since; the current pin is
`2.0.0+cu13.bw.20260529.3` (GitHub release; see `docs/build-system.md` §6.5 for
the pin mechanics).

**The dominant `cudaErrorNoKernelImageForDevice` (error 209) failure is FIXED.**
The wheel now ships `sm_89` kernels, so the "Missing GPU kernels on RTX 4090"
group (§2 below — ~33 notebooks) and the GPU-scalar-to-host "could not execute a
primitive" group (§3 below — 6 notebooks) **all pass** on the 4×4090 box.
Confirmed in a full `make all` run:

- Standalone probe `(np.ones((1024,1024), ctx=mx.gpu(0)) @ …).sum()` passes.
- The previously-209 notebooks now run real GPU training to completion:
  `seq2seq`, `bahdanau-attention`, `recommender-systems/{mf,neumf,…}`,
  `recurrent-modern/{gru,lstm,deep-rnn}`, `rnn-scratch`, `minibatch-sgd`,
  `word2vec-pretraining`, etc.
- **Zero** error-209 occurrences across `logs/nb-errors/mxnet/` in the run.

So the §2/§3 "fix direction: rebuild with sm_89" below is **done**. The
self-contained `gpu-*` reproducers (§"Self-Contained Reproducers") now pass and
are useful as a regression check when bumping the wheel again.

**Remaining failure (new surface): `Build with USE_OPENCV=1` under heavy
parallelism.** In the full `make all` run, ~28 image-dataset notebooks
(everything that loads FashionMNIST/CIFAR/CV data — `lenet`, `alexnet`, the
`convolutional-modern/*` family, `computer-vision/*`, `softmax-regression-*`,
`dropout`, `mlp-implementation`, …) failed with:

```text
MXNetError: Build with USE_OPENCV=1 for image resize operator.   (or "… for image io")
```

This is **not** a wheel defect and **not** the §1 "missing OpenCV runtime libs"
(an `OSError` at import). Here import succeeds and OpenCV is genuinely present:

- `mxnet.runtime.Features().is_enabled('OPENCV')` → `True`.
- `libopencv_{core,imgproc,imgcodecs}.so.406` are bundled in
  `…/site-packages/mxnet/lib/` (resolved by `ldd libmxnet.so`) **and** installed
  system-wide; `tools/check_runtime_deps.py mxnet` exits 0.
- The image ops **succeed in isolation** — `mx.image.imresize(...)` and the full
  `d2l.FashionMNIST(...).get_dataloader()` (DataLoader worker subprocesses)
  return a batch cleanly, **including after a GPU context is initialized first**
  (ruling out a fork-after-CUDA-init hypothesis).

The ops fail *only* under the concurrent build. Root cause: **process/thread
exhaustion against `ulimit -u`** — even at the ~56 threads/proc the
`EXTRA_ENV_mxnet` env vars leave, MXNet image-dataset notebooks spawn Gluon
DataLoader worker subprocesses, and at the default `GPU_SLOTS` (~2/GPU → 8 here)
the combined count exceeds the host's `ulimit -u` (soft 4096 / hard 8192). A
starved worker then can't lazy-`dlopen` the bundled OpenCV and surfaces the
misleading "build without OpenCV" error.

**Fixed (2026-06-06, verified).** Confirmed by re-running the image notebooks
serially (`make -B …/lenet.executed`, etc.) — every one passes in isolation. The
durable fix is a per-framework concurrency cap, **`MXNET_GPU_SLOTS ?= 2`** in the
Makefile (injected as `D2L_MXNET_GPU_SLOTS`, honored by `run_one_notebook.py`'s
`acquire_fw_cap` on top of the global pool; pt/tf/jax keep using the freed
slots). Re-verified end-to-end: a full `make clean && make all` ran all 28
formerly-failing MXNet image notebooks (lenet, the `convolutional-modern/*`
family, `computer-vision/*`, `softmax-regression-*`, dropout, mlp-implementation,
…) to completion with **zero** OpenCV errors. Raise the cap only if the host's
`ulimit -u` is raised.

Also note one **content** bug surfaced under this wheel, unrelated to the runtime:
`chapter_mdl-linear-algebra/mdl-geometry-linear-algebraic-ops` used `np.einsum`'s
sublist/interleaved form, which MXNet's einsum does not support (it requires the
string-subscript form, like TensorFlow) — fixed in source by making the mxnet tab
a comment, matching the existing tf tab.

---

Date: 2026-05-19  *(historical snapshot — wheel `…20260517`; the error-209
analysis below is now resolved per the section above)*

This note summarizes the current MXNet notebook failures and gives
self-contained reproducers that do not require the D2L notebooks or the `d2l`
package.

## Clean Run

The clean MXNet run cleared old `.executed` stamps, cleared notebook outputs,
cleared `logs/nb-errors/mxnet`, and then ran:

```bash
make -k run-notebooks-mxnet NUM_GPUS=4 GPU_SLOTS=8
make -k run-notebooks-multigpu-mxnet NUM_GPUS=4 GPU_SLOTS=8
python3 tools/audit_notebook_results.py \
  --out docs/notebook-result-audit-2026-05-19.md
```

Run logs:

- Single-GPU/CPU queues: `logs/run-mxnet-20260519-070820.log`
- Explicit multi-GPU queue: `logs/run-mxnet-20260519-071521.log`
- Per-notebook failures: `logs/nb-errors/mxnet/**/*.log`

Current stamp state:

| Bucket | Passed | Failed |
|--------|--------|--------|
| All MXNet notebooks | 86 / 128 | 42 |
| CPU notebooks | 53 / 53 | 0 |
| GPU notebooks | 31 / 69 | 38 |
| Multi-GPU notebooks | 2 / 6 | 4 |

Failure log grouping:

| Primary symptom | Notebooks |
|-----------------|-----------|
| Missing CUDA kernel image (`cudaErrorNoKernelImageForDevice`, 209) | 33 |
| `MXNetError: could not execute a primitive` | 6 |
| Dead kernel with no Python traceback | 3 |
| Missing OpenCV runtime libraries | 0 in this clean run |

Important artifact-quality note: `chapter_builders-guide/use-gpu.ipynb` now
has a passing stamp, but its current output still contains two MXNet error
outputs when it displays GPU arrays. Treat it as a runtime-quality failure even
though Make marked the notebook target as OK.

Hardware and runtime facts:

- GPUs: 4x NVIDIA GeForce RTX 4090, compute capability 8.9.
- Clean run scheduling: `GPU_SLOTS=8`, i.e. two notebook slots per 24GB GPU.
- MXNet: `2.0.0+cu13.bw.20260517` in `.venv-mxnet`.
- The custom wheel imports after installing OpenCV 4.6 runtime packages:
  `libopencv-core406t64`, `libopencv-imgproc406t64`,
  `libopencv-imgcodecs406t64`.

## Self-Contained Reproducers

Use `tools/repro_mxnet_runtime.py` for reproduction. It imports only `mxnet`
and standard-library modules; it does not import `d2l` and does not read any
notebook files. It also re-execs with pip-installed NVIDIA library directories
on `LD_LIBRARY_PATH` when they are present under the active Python
environment. Run the GPU probes from a normal GPU-capable shell; the restricted
sandbox can fail earlier while querying CUDA.

Basic environment check:

```bash
.venv-mxnet/bin/python tools/repro_mxnet_runtime.py --case info
```

GPU kernel coverage checks:

```bash
CUDA_VISIBLE_DEVICES=0 \
  .venv-mxnet/bin/python tools/repro_mxnet_runtime.py --case gpu-sum

CUDA_VISIBLE_DEVICES=0 \
  .venv-mxnet/bin/python tools/repro_mxnet_runtime.py --case gpu-softmax

CUDA_VISIBLE_DEVICES=0 \
  .venv-mxnet/bin/python tools/repro_mxnet_runtime.py --case gpu-transpose

CUDA_VISIBLE_DEVICES=0 \
  .venv-mxnet/bin/python tools/repro_mxnet_runtime.py --case gpu-dense-loss
```

GPU scalar-to-host synchronization checks:

```bash
CUDA_VISIBLE_DEVICES=0 \
  .venv-mxnet/bin/python tools/repro_mxnet_runtime.py --case gpu-scalar-to-host

CUDA_VISIBLE_DEVICES=0 \
  .venv-mxnet/bin/python tools/repro_mxnet_runtime.py --case gpu-gru-scalar-to-host
```

Transformer decoder native-crash check:

```bash
CUDA_VISIBLE_DEVICES= \
  .venv-mxnet/bin/python tools/repro_mxnet_runtime.py \
  --case transformer-decoder-standalone
```

The current wheel fails these standalone probes on this machine:

- `gpu-sum` fails with `mxnet_generic_kernel ErrStr:no kernel image is available`.
- `gpu-scalar-to-host` and `gpu-gru-scalar-to-host` fail on the same
  GPU-to-host path, currently surfacing as error 209.
- `transformer-decoder-standalone` reaches `step: sync output` and then the
  Python process exits with signal 139.

## Passed Notebook Quality

The current audit report is `docs/notebook-result-audit-2026-05-19.md`.
Among stamped MXNet notebooks, the scoreable output is limited, but the
available training curves are qualitatively sane:

- `linear-regression-{scratch,concise}`, `softmax-regression-concise`,
  `dropout`, `mlp-implementation`, `kaggle-house-price`, and the optimizer
  demos show loss curves moving in the expected direction.
- `softmax-regression-concise`, `dropout`, and `mlp-implementation` show
  validation accuracy improving.
- `chapter_generative-adversarial-networks/gan.ipynb` reports
  `loss_D=0.693, loss_G=0.694`, which is a plausible toy-GAN endpoint.

The audit did not find all-zero accuracy, high RMSE, high perplexity, or
zero-BLEU convergence failures in stamped MXNet notebooks. The dominant problem
is that many training notebooks cannot run at all on this MXNet GPU runtime.

## Failure Groups

### 1. Missing OpenCV Runtime Libraries

Initial MXNet runs failed before notebook code executed because the custom
wheel links against system OpenCV 4.6 libraries that Python package metadata
does not install.

Symptom:

```text
OSError: libopencv_imgcodecs.so.406: cannot open shared object file:
No such file or directory
```

Minimal check:

```bash
python3 tools/check_runtime_deps.py mxnet
```

The same dependency can be inspected directly:

```bash
ldd .venv-mxnet/lib/python3.11/site-packages/mxnet/libmxnet.so | grep opencv
```

Ubuntu 24.04 fix:

```bash
sudo apt-get install -y libopencv-core406t64 \
  libopencv-imgproc406t64 libopencv-imgcodecs406t64
```

Status on this machine: fixed locally after installing those packages. Keep it
on the issue list because clean hosts will otherwise fail before reaching any
notebook-specific code.

### 2. Missing GPU Kernels On RTX 4090

The dominant clean-run failure is CUDA error 209,
`cudaErrorNoKernelImageForDevice`. The affected notebook errors are different
surface forms of the same problem:

- `mxnet_generic_kernel ErrStr:no kernel image is available`
- `mxnet_generic_kernel_ex ErrStr:no kernel image is available`
- `MapPlanKernel ErrStr:no kernel image is available`
- `TransposePseudo2D kernel failure: no kernel image is available`
- `softmax_stride1_compute_kernel ErrStr:no kernel image is available`

Clean-run failures in this group:

```text
chapter_attention-mechanisms-and-transformers/bahdanau-attention.ipynb
chapter_computational-performance/auto-parallelism.ipynb
chapter_computational-performance/multiple-gpus-concise.ipynb
chapter_computational-performance/multiple-gpus.ipynb
chapter_computer-vision/fcn.ipynb
chapter_computer-vision/fine-tuning.ipynb
chapter_computer-vision/image-augmentation.ipynb
chapter_computer-vision/kaggle-cifar10.ipynb
chapter_computer-vision/kaggle-dog.ipynb
chapter_computer-vision/neural-style.ipynb
chapter_computer-vision/ssd.ipynb
chapter_convolutional-modern/alexnet.ipynb
chapter_convolutional-modern/batch-norm.ipynb
chapter_convolutional-modern/cnn-design.ipynb
chapter_convolutional-modern/densenet.ipynb
chapter_convolutional-modern/googlenet.ipynb
chapter_convolutional-modern/nin.ipynb
chapter_convolutional-modern/resnet.ipynb
chapter_convolutional-modern/vgg.ipynb
chapter_convolutional-neural-networks/lenet.ipynb
chapter_generative-adversarial-networks/dcgan.ipynb
chapter_natural-language-processing-applications/natural-language-inference-attention.ipynb
chapter_natural-language-processing-applications/sentiment-analysis-cnn.ipynb
chapter_natural-language-processing-pretraining/bert-pretraining.ipynb
chapter_natural-language-processing-pretraining/word2vec-pretraining.ipynb
chapter_optimization/lr-scheduler.ipynb
chapter_recommender-systems/autorec.ipynb
chapter_recommender-systems/deepfm.ipynb
chapter_recommender-systems/fm.ipynb
chapter_recommender-systems/mf.ipynb
chapter_recommender-systems/neumf.ipynb
chapter_recommender-systems/seqrec.ipynb
chapter_recurrent-modern/seq2seq.ipynb
```

Minimal repro:

```bash
CUDA_VISIBLE_DEVICES=0 \
  .venv-mxnet/bin/python tools/repro_mxnet_runtime.py --case gpu-sum
```

Expected failure:

```text
mxnet_generic_kernel ErrStr:no kernel image is available for execution on the device
```

Equivalent minimal Python:

```python
import mxnet as mx
from mxnet import np, npx

npx.set_np()
x = np.ones((128,), ctx=mx.gpu(0))
print((x + 1).sum().asnumpy())
```

Additional probes:

```bash
CUDA_VISIBLE_DEVICES=0 \
  .venv-mxnet/bin/python tools/repro_mxnet_runtime.py --case gpu-softmax

CUDA_VISIBLE_DEVICES=0 \
  .venv-mxnet/bin/python tools/repro_mxnet_runtime.py --case gpu-transpose

CUDA_VISIBLE_DEVICES=0 \
  .venv-mxnet/bin/python tools/repro_mxnet_runtime.py --case gpu-dense-loss
```

Fix direction: rebuild the custom MXNet wheel with kernels for the local GPU
architecture (`sm_89` for RTX 4090) and preferably a PTX fallback. The current
wheel is Blackwell-oriented; on Ada it imports, but common GPU kernels are
missing. Verify a candidate wheel with the probes above before rerunning full
notebooks.

### 3. `could not execute a primitive`

Six clean-run failures reported `MXNetError: could not execute a primitive`:

```text
chapter_optimization/minibatch-sgd.ipynb
chapter_recurrent-modern/deep-rnn.ipynb
chapter_recurrent-modern/gru.ipynb
chapter_recurrent-modern/lstm.ipynb
chapter_recurrent-neural-networks/rnn-concise.ipynb
chapter_recurrent-neural-networks/rnn-scratch.ipynb
```

In the RNN notebooks, the failure happens when training tries to plot a metric:
`RNNLMScratch.training_step` computes the loss/perplexity, then
`Module.plot` calls `d2l.numpy(value)`, which calls `value.asnumpy()`. The
GPU-to-host synchronization then raises `could not execute a primitive`.

For `minibatch-sgd`, the failure happens during `evaluate_loss`, where
`Accumulator.add` converts an MXNet scalar to Python `float`, which internally
calls `ndarray.item()` and then `asnumpy()`.

These failures are probably another surface of the broken MXNet GPU runtime,
but keep them separate from error 209 until a rebuilt wheel is tested.

Self-contained probes for the same GPU scalar-to-host path:

```bash
CUDA_VISIBLE_DEVICES=0 \
  .venv-mxnet/bin/python tools/repro_mxnet_runtime.py --case gpu-scalar-to-host

CUDA_VISIBLE_DEVICES=0 \
  .venv-mxnet/bin/python tools/repro_mxnet_runtime.py --case gpu-gru-scalar-to-host
```

On this wheel the standalone probes currently fail earlier with error 209
(`no kernel image is available`). That still exercises the same scalar
conversion path shown in the notebook logs: compute a GPU scalar, then convert
it to host/Python.

```python
import mxnet as mx
from mxnet import np, npx

npx.set_np()
x = np.ones((32, 32), ctx=mx.gpu(0))
value = (x @ x).sum()
print(float(value))
```

Interpretation: fix and retest the error-209 probes first. If a rebuilt wheel
fixes those but the six notebook-derived primitive failures persist, the next
place to inspect is MXNet's GPU-to-host synchronization and primitive dispatch
path. The `gpu-gru-scalar-to-host` probe is the notebook-free stress case for
that path.

### 4. Dead Kernels

Three clean-run failures ended as `DeadKernelError` without a useful Python
exception:

```text
chapter_attention-mechanisms-and-transformers/transformer.ipynb
chapter_natural-language-processing-applications/natural-language-inference-bert.ipynb
chapter_natural-language-processing-applications/sentiment-analysis-rnn.ipynb
```

The transformer crash now has a notebook-free reproducer. It builds only the
minimal MXNet/Gluon attention, AddNorm, feed-forward, encoder-block, and
decoder-block classes needed for the failing operation.

Repro:

```bash
CUDA_VISIBLE_DEVICES= \
  .venv-mxnet/bin/python tools/repro_mxnet_runtime.py \
  --case transformer-decoder-standalone
```

Observed output before the process exits with signal 139:

```text
step: construct blocks
step: make inputs
step: compute encoder state
step: call decoder block
step: sync output
```

The same standalone probe still crashes in the current wheel with
`MXNET_ENGINE_TYPE=NaiveEngine`, so this is a native MXNet runtime issue and
not a notebook artifact.

For local narrowing against the generated notebooks, the older prefix runner is
still available, but it is not required to reproduce the runtime failure:

```bash
MPLCONFIGDIR=/tmp/matplotlib-d2l CUDA_VISIBLE_DEVICES= \
  .venv-mxnet/bin/python tools/repro_mxnet_failures.py \
  --case transformer-decoder-block
```

For the NLI-BERT and sentiment-RNN failures, the notebook-free GPU scalar/RNN
probes above cover the same GPU execution and host-sync class. If more exact
local narrowing is needed, use the optional notebook prefix runner:

```bash
MPLCONFIGDIR=/tmp/matplotlib-d2l CUDA_VISIBLE_DEVICES=0 \
  .venv-mxnet/bin/python tools/repro_mxnet_failures.py \
  --case notebook-prefix \
  --notebook chapter_natural-language-processing-applications/natural-language-inference-bert.ipynb \
  --stop-cell 18

MPLCONFIGDIR=/tmp/matplotlib-d2l CUDA_VISIBLE_DEVICES=0 \
  .venv-mxnet/bin/python tools/repro_mxnet_failures.py \
  --case notebook-prefix \
  --notebook chapter_natural-language-processing-applications/sentiment-analysis-rnn.ipynb \
  --stop-cell 15
```

Fix direction: first verify whether rebuilding the MXNet wheel for `sm_89`
eliminates these. If not, investigate the standalone transformer decoder crash
directly; in the current wheel, `MXNET_ENGINE_TYPE=NaiveEngine` does not prevent
the notebook-free reproducer from crashing.

### 5. Base BERT Notebook Passed

`chapter_natural-language-processing-pretraining/bert.ipynb` failed once in an
earlier parallel run, but it passed in the clean run and has a current stamp.
Keep the older failure in mind as possible parallel resource or engine
instability, but it is not deterministic with the current evidence.

### 6. Sandbox-Only CUDA Initialization Failure

Inside the restricted sandbox, importing `d2l.mxnet` can fail with:

```text
CUDA: Check failed: e == cudaSuccess (304 vs. 0) :
OS call failed or operation not supported on this OS
```

This is not the elevated notebook failure. It happens because some saved
functions query GPUs at import time via default arguments such as:

```python
def train_ch13(..., devices=d2l.try_all_gpus(), ...):
    ...
```

Recommended code fix:

```python
def try_all_gpus():
    devices = [gpu(i) for i in range(num_gpus())]
    return devices if devices else [cpu()]

def train_ch13(..., devices=None, split_f=d2l.split_batch):
    if devices is None:
        devices = d2l.try_all_gpus()
    ...
```

Apply the same lazy-default pattern to `train_recsys_rating`. This avoids CUDA
probing during import and makes CPU-only execution more robust.

## Non-Root-Cause Noise

- `Notebook JSON is invalid: Additional properties are not allowed ('id' was unexpected)`
  appears before many failures. nbconvert continues executing; this warning is
  not the runtime failure.
