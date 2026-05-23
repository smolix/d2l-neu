#!/usr/bin/env python3
"""Standalone MXNet runtime probes.

This file intentionally does not import d2l and does not read notebooks.  It is
meant to be copyable into a clean MXNet environment to reproduce the runtime
failures seen while executing the D2L MXNet notebooks.
"""

from __future__ import annotations

import argparse
import faulthandler
import glob
import os
import sys
import sysconfig
import traceback
from pathlib import Path


def ensure_nvidia_ld_path():
    """Re-exec with pip-installed NVIDIA library dirs on LD_LIBRARY_PATH.

    The custom MXNet wheel may rely on CUDA/cuDNN/NCCL libraries provided by
    pip packages under site-packages/nvidia/*/lib.  This keeps the reproducer
    copyable while still working inside this repository's `.venv-mxnet`.
    """
    if os.environ.get("D2L_MXNET_RUNTIME_REEXEC") == "1":
        return
    site_roots = {
        Path(sysconfig.get_paths().get("purelib", "")),
        Path(sys.executable).resolve().parent.parent / "lib/python3.12/site-packages",
    }
    lib_dirs: list[str] = []
    for site_root in site_roots:
        if not site_root:
            continue
        lib_dirs.extend(glob.glob(str(site_root / "nvidia/*/lib")))
    lib_dirs = sorted({str(Path(path).resolve()) for path in lib_dirs if Path(path).is_dir()})
    if not lib_dirs:
        return
    current = [item for item in os.environ.get("LD_LIBRARY_PATH", "").split(":") if item]
    missing = [item for item in lib_dirs if item not in current]
    if not missing:
        return
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(missing + current)
    env["D2L_MXNET_RUNTIME_REEXEC"] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


def import_mxnet():
    import mxnet as mx
    from mxnet import autograd, gluon, np, npx
    from mxnet.gluon import nn

    npx.set_np()
    return mx, np, npx, autograd, gluon, nn


def gpu_context(mx):
    count = mx.context.num_gpus()
    if count < 1:
        raise RuntimeError("MXNet reports no visible GPUs")
    return mx.gpu(0)


def run_case(name, fn):
    print(f"== {name} ==", flush=True)
    try:
        value = fn()
    except Exception:
        print(f"FAIL: {name}", flush=True)
        traceback.print_exc()
        return False
    print(f"OK: {name}: {value}", flush=True)
    return True


def sync(value):
    if hasattr(value, "asnumpy"):
        return value.asnumpy()
    return value


def case_info():
    mx, _, _, _, _, _ = import_mxnet()
    print(f"mxnet={mx.__version__}", flush=True)
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    print(f"num_gpus={mx.context.num_gpus()}", flush=True)


def case_gpu_sum():
    mx, np, _, _, _, _ = import_mxnet()
    ctx = gpu_context(mx)
    x = np.ones((128,), ctx=ctx)
    return (x + 1).sum().asnumpy()


def case_gpu_softmax():
    mx, np, npx, _, _, _ = import_mxnet()
    ctx = gpu_context(mx)
    x = np.ones((4, 10), ctx=ctx)
    return npx.softmax(x, axis=1).asnumpy()[0, :3]


def case_gpu_transpose():
    mx, np, _, _, _, _ = import_mxnet()
    ctx = gpu_context(mx)
    x = np.arange(2 * 8 * 16, ctx=ctx).reshape((2, 8, 16))
    return x.transpose((0, 2, 1)).asnumpy()[0, 0, :4]


def case_gpu_dense_loss():
    mx, np, _, autograd, gluon, nn = import_mxnet()
    ctx = gpu_context(mx)
    net = nn.Dense(10)
    net.initialize(ctx=ctx)
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    x = np.ones((8, 32), ctx=ctx)
    y = np.zeros((8,), ctx=ctx)
    with autograd.record():
        l = loss(net(x), y)
    l.backward()
    return float(l.sum())


def case_gpu_scalar_to_host():
    mx, np, _, _, _, _ = import_mxnet()
    ctx = gpu_context(mx)
    x = np.ones((32, 32), ctx=ctx)
    return float((x @ x).sum())


def case_gpu_gru_scalar_to_host():
    mx, np, npx, _, _, _ = import_mxnet()
    ctx = gpu_context(mx)
    batch_size, num_steps, num_inputs, num_hiddens = 32, 8, 64, 32
    indices = np.arange(batch_size * num_steps, ctx=ctx).reshape(
        batch_size, num_steps
    ) % num_inputs
    inputs = npx.one_hot(indices.astype("int32"), num_inputs).transpose((1, 0, 2))

    def weight(*shape):
        return np.random.normal(scale=0.01, size=shape, ctx=ctx)

    W_xz = weight(num_inputs, num_hiddens)
    W_hz = weight(num_hiddens, num_hiddens)
    b_z = np.zeros(num_hiddens, ctx=ctx)
    W_xr = weight(num_inputs, num_hiddens)
    W_hr = weight(num_hiddens, num_hiddens)
    b_r = np.zeros(num_hiddens, ctx=ctx)
    W_xh = weight(num_inputs, num_hiddens)
    W_hh = weight(num_hiddens, num_hiddens)
    b_h = np.zeros(num_hiddens, ctx=ctx)

    H = np.zeros((batch_size, num_hiddens), ctx=ctx)
    for X in inputs:
        Z = npx.sigmoid(X @ W_xz + H @ W_hz + b_z)
        R = npx.sigmoid(X @ W_xr + H @ W_hr + b_r)
        H_tilde = np.tanh(X @ W_xh + (R * H) @ W_hh + b_h)
        H = Z * H + (1 - Z) * H_tilde
    return float(H.sum())


def case_transformer_decoder_standalone():
    _, np, npx, autograd, _, nn = import_mxnet()

    def masked_softmax(X, valid_lens):
        if valid_lens is None:
            return npx.softmax(X)
        shape = X.shape
        if valid_lens.ndim == 1:
            valid_lens = valid_lens.repeat(shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        X = npx.sequence_mask(
            X.reshape(-1, shape[-1]),
            valid_lens,
            True,
            value=-1e6,
            axis=1,
        )
        return npx.softmax(X).reshape(shape)

    class DotProductAttention(nn.Block):
        def __init__(self, dropout):
            super().__init__()
            self.dropout = nn.Dropout(dropout)

        def forward(self, queries, keys, values, valid_lens=None):
            d = queries.shape[-1]
            scores = npx.batch_dot(queries, keys, transpose_b=True) / (d ** 0.5)
            weights = masked_softmax(scores, valid_lens)
            return npx.batch_dot(self.dropout(weights), values)

    class MultiHeadAttention(nn.Block):
        def __init__(self, num_hiddens, num_heads, dropout):
            super().__init__()
            self.num_heads = num_heads
            self.attention = DotProductAttention(dropout)
            self.W_q = nn.Dense(num_hiddens, use_bias=False, flatten=False)
            self.W_k = nn.Dense(num_hiddens, use_bias=False, flatten=False)
            self.W_v = nn.Dense(num_hiddens, use_bias=False, flatten=False)
            self.W_o = nn.Dense(num_hiddens, use_bias=False, flatten=False)

        def transpose_qkv(self, X):
            X = X.reshape(X.shape[0], X.shape[1], self.num_heads, -1)
            X = X.transpose(0, 2, 1, 3)
            return X.reshape(-1, X.shape[2], X.shape[3])

        def transpose_output(self, X):
            X = X.reshape(-1, self.num_heads, X.shape[1], X.shape[2])
            X = X.transpose(0, 2, 1, 3)
            return X.reshape(X.shape[0], X.shape[1], -1)

        def forward(self, queries, keys, values, valid_lens):
            queries = self.transpose_qkv(self.W_q(queries))
            keys = self.transpose_qkv(self.W_k(keys))
            values = self.transpose_qkv(self.W_v(values))
            if valid_lens is not None:
                valid_lens = valid_lens.repeat(self.num_heads, axis=0)
            output = self.attention(queries, keys, values, valid_lens)
            return self.W_o(self.transpose_output(output))

    class PositionWiseFFN(nn.Block):
        def __init__(self, ffn_num_hiddens, ffn_num_outputs):
            super().__init__()
            self.dense1 = nn.Dense(ffn_num_hiddens, flatten=False, activation="relu")
            self.dense2 = nn.Dense(ffn_num_outputs, flatten=False)

        def forward(self, X):
            return self.dense2(self.dense1(X))

    class AddNorm(nn.Block):
        def __init__(self, dropout):
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            self.ln = nn.LayerNorm()

        def forward(self, X, Y):
            return self.ln(self.dropout(Y) + X)

    class TransformerEncoderBlock(nn.Block):
        def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout):
            super().__init__()
            self.attention = MultiHeadAttention(num_hiddens, num_heads, dropout)
            self.addnorm1 = AddNorm(dropout)
            self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
            self.addnorm2 = AddNorm(dropout)

        def forward(self, X, valid_lens):
            Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
            return self.addnorm2(Y, self.ffn(Y))

    class TransformerDecoderBlock(nn.Block):
        def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout):
            super().__init__()
            self.attention1 = MultiHeadAttention(num_hiddens, num_heads, dropout)
            self.addnorm1 = AddNorm(dropout)
            self.attention2 = MultiHeadAttention(num_hiddens, num_heads, dropout)
            self.addnorm2 = AddNorm(dropout)
            self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
            self.addnorm3 = AddNorm(dropout)

        def forward(self, X, state):
            enc_outputs, enc_valid_lens, cached = state
            key_values = X if cached is None else np.concatenate((cached, X), axis=1)
            if autograd.is_training():
                batch_size, num_steps, _ = X.shape
                dec_valid_lens = np.tile(
                    np.arange(1, num_steps + 1, ctx=X.ctx), (batch_size, 1)
                )
            else:
                dec_valid_lens = None
            X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
            Y = self.addnorm1(X, X2)
            Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
            Z = self.addnorm2(Y, Y2)
            return self.addnorm3(Z, self.ffn(Z)), [enc_outputs, enc_valid_lens, key_values]

    print("step: construct blocks", flush=True)
    encoder_blk = TransformerEncoderBlock(24, 48, 8, 0.5)
    decoder_blk = TransformerDecoderBlock(24, 48, 8, 0.5)
    encoder_blk.initialize()
    decoder_blk.initialize()
    print("step: make inputs", flush=True)
    X = np.ones((2, 100, 24))
    valid_lens = np.array([3, 2])
    print("step: compute encoder state", flush=True)
    enc_state = encoder_blk(X, valid_lens)
    print("step: call decoder block", flush=True)
    out, _ = decoder_blk(X, [enc_state, valid_lens, None])
    print("step: sync output", flush=True)
    return sync(out[0, 0, :4])


def main():
    ensure_nvidia_ld_path()
    faulthandler.enable(all_threads=True)
    cases = {
        "info": case_info,
        "gpu-sum": case_gpu_sum,
        "gpu-softmax": case_gpu_softmax,
        "gpu-transpose": case_gpu_transpose,
        "gpu-dense-loss": case_gpu_dense_loss,
        "gpu-scalar-to-host": case_gpu_scalar_to_host,
        "gpu-gru-scalar-to-host": case_gpu_gru_scalar_to_host,
        "transformer-decoder-standalone": case_transformer_decoder_standalone,
    }
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=[*cases, "gpu-all"], default="gpu-all")
    args = parser.parse_args()

    selected = (
        [(name, cases[name]) for name in cases if name.startswith("gpu-")]
        if args.case == "gpu-all"
        else [(args.case, cases[args.case])]
    )
    ok = True
    for name, fn in selected:
        ok = run_case(name, fn) and ok
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
