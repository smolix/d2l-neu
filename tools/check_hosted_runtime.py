#!/usr/bin/env python3
"""Validate one hosted framework's lock, imports, APIs, and update semantics.

Run this with the matching framework environment, for example::

    .venv-jax/bin/python tools/check_hosted_runtime.py jax

The check is intentionally a small runtime contract rather than a notebook
test.  It covers the shared environment and framework paths used by many
notebooks, including the NNX view/optimizer behavior that hosted runtimes have
previously broken.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.util
import io
import math
import os
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from export_hosted_env import load_profile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "2")


def public_version(value: str) -> str:
    return value.split("+", 1)[0]


def check_versions(profile: dict, include_optional: bool = True,
                   provider_compatible: bool = False) -> None:
    failures = []
    packages = list(profile["packages"])
    if include_optional:
        packages.extend(profile["optional_packages"])
    if provider_compatible:
        packages = [
            package for package in packages
            if package.get("hosted_policy", "exact") == "exact"
        ]
    for package in packages:
        distribution = package["distribution"]
        try:
            installed = version(distribution)
        except PackageNotFoundError:
            if package in profile["packages"]:
                failures.append(f"{distribution}: missing")
            continue
        actual = (public_version(installed)
                  if package["match"] == "public" else installed)
        if actual != package["version"]:
            failures.append(
                f"{distribution}: installed {installed}, expected "
                f"{package['version']} ({package['match']})"
            )
    if failures:
        raise RuntimeError("runtime version contract failed:\n  " + "\n  ".join(failures))


def resolve_api(path: str):
    parts = path.split(".")
    for boundary in range(len(parts), 0, -1):
        module_name = ".".join(parts[:boundary])
        try:
            value = importlib.import_module(module_name)
        except ModuleNotFoundError as error:
            if not (error.name == module_name or
                    module_name.startswith(f"{error.name}.")):
                raise
            continue
        for part in parts[boundary:]:
            value = getattr(value, part)
        return value
    raise ModuleNotFoundError(path)


def check_apis(profile: dict) -> None:
    failures = []
    for path in profile["critical_apis"]:
        try:
            resolve_api(path)
        except Exception as error:
            failures.append(f"{path}: {type(error).__name__}: {error}")
    if failures:
        raise RuntimeError("critical API contract failed:\n  " + "\n  ".join(failures))


def check_installed_optional_imports(profile: dict) -> None:
    failures = []
    for package in profile["optional_packages"]:
        try:
            version(package["distribution"])
        except PackageNotFoundError:
            continue
        try:
            with contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                importlib.import_module(package["import"])
        except Exception as error:
            failures.append(
                f"{package['import']}: {type(error).__name__}: {error}"
            )
    if failures:
        raise RuntimeError(
            "optional dependency import contract failed:\n  "
            + "\n  ".join(failures)
        )


def check_tensorflow_metadata_if_present() -> None:
    """Exercise the exact generated-proto import that failed in Colab."""
    try:
        present = importlib.util.find_spec(
            "tensorflow_metadata.proto.v0.anomalies_pb2"
        )
    except (ImportError, AttributeError):
        present = None
    if present is not None:
        importlib.import_module("tensorflow_metadata.proto.v0.anomalies_pb2")


def check_pytorch(device: str) -> None:
    import torch
    from d2l import torch as d2l  # noqa: F401

    if device == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access the requested GPU")
    target = torch.device("cuda:0" if device == "gpu" else "cpu")
    torch.manual_seed(0)
    model = torch.nn.Linear(2, 1).to(target)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    x = torch.tensor([[1.0, -1.0], [0.5, 2.0]], device=target)
    y = torch.tensor([[2.0], [-1.0]], device=target)
    before = model.weight.detach().clone()
    loss = ((model(x) - y) ** 2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if model.weight.device.type != target.type:
        raise RuntimeError(f"PyTorch computation ran on {model.weight.device}")
    if not math.isfinite(loss.item()) or torch.equal(before, model.weight.detach()):
        raise RuntimeError("PyTorch optimizer contract did not update parameters")


def check_tensorflow(device: str) -> None:
    import tensorflow as tf

    if device == "gpu":
        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            raise RuntimeError("TensorFlow cannot access the requested GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    from d2l import tensorflow as d2l  # noqa: F401

    requested = "/GPU:0" if device == "gpu" else "/CPU:0"
    with tf.device(requested):
        weight = tf.Variable([[0.5], [-0.25]], dtype=tf.float32)
        x = tf.constant([[1.0, -1.0], [0.5, 2.0]])
        y = tf.constant([[2.0], [-1.0]])
        optimizer = tf.keras.optimizers.SGD(0.1)
        before = weight.numpy().copy()
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.square(tf.matmul(x, weight) - y))
        optimizer.apply_gradients([(tape.gradient(loss, weight), weight)])
    if device == "gpu" and "GPU:" not in loss.device.upper():
        raise RuntimeError(f"TensorFlow computation ran on {loss.device}")
    if not math.isfinite(float(loss.numpy())) or (before == weight.numpy()).all():
        raise RuntimeError("TensorFlow optimizer contract did not update parameters")
    check_tensorflow_metadata_if_present()


def check_jax(device: str) -> None:
    import jax
    import jax.numpy as jnp
    import optax
    from flax import nnx
    from d2l import jax as d2l  # noqa: F401

    try:
        target = jax.devices("gpu" if device == "gpu" else "cpu")[0]
    except (IndexError, RuntimeError) as error:
        raise RuntimeError(f"JAX cannot access the requested {device}") from error

    class TinyStatefulModel(nnx.Module):
        def __init__(self, rngs):
            self.batch_norm = nnx.BatchNorm(2, rngs=rngs)
            self.dropout = nnx.Dropout(0.1, rngs=rngs)
            self.linear = nnx.Linear(2, 1, rngs=rngs)

        def __call__(self, x):
            return self.linear(self.dropout(self.batch_norm(x)))

    with jax.default_device(target):
        model = TinyStatefulModel(nnx.Rngs(0))
        train_model = nnx.view(
            model, deterministic=False, use_running_average=False
        )
        eval_model = nnx.view(
            model, deterministic=True, use_running_average=True
        )
    if train_model.linear.kernel is not eval_model.linear.kernel:
        raise RuntimeError("NNX model views do not share parameter variables")

    with jax.default_device(target):
        optimizer = nnx.Optimizer(model, optax.sgd(0.05), wrt=nnx.Param)
        x = jnp.array([[1.0, -1.0], [0.5, 2.0], [-1.0, 0.5], [2.0, 1.0]])
        y = jnp.array([[2.0], [-1.0], [0.5], [1.5]])
    before = jax.device_get(model.linear.kernel[...]).copy()

    @nnx.jit
    def train_step(active_model, active_optimizer, features, labels):
        def loss_fn(candidate):
            return jnp.mean((candidate(features) - labels) ** 2)

        loss, grads = nnx.value_and_grad(loss_fn)(active_model)
        active_optimizer.update(active_model, grads)
        return loss

    with jax.default_device(target):
        loss = train_step(train_model, optimizer, x, y)
        prediction = eval_model(x)
    jax.block_until_ready(prediction)
    if target not in prediction.devices():
        raise RuntimeError(f"JAX computation ran on {prediction.devices()}")
    after = jax.device_get(model.linear.kernel[...])
    if (not math.isfinite(float(loss)) or not bool(jnp.all(jnp.isfinite(prediction)))
            or (before == after).all()):
        raise RuntimeError("JAX/NNX optimizer contract did not update shared state")
    check_tensorflow_metadata_if_present()


def check_d2l_training_path(framework: str) -> None:
    """Exercise the introductory notebook's shared OO training path.

    Import and one-layer optimizer probes do not cover the ``Trainer``
    implementation. Keep this deliberately tiny while following the same
    ``SyntheticRegressionData -> Trainer.fit`` path readers execute first.
    """
    helper_name = "torch" if framework == "pytorch" else framework
    module = importlib.import_module(f"d2l.{helper_name}")
    truth = module.tensor([2.0, -3.4])
    model = module.LinearRegressionScratch(2, lr=0.03)
    data = module.SyntheticRegressionData(
        w=truth, b=4.2, num_train=32, num_val=16, batch_size=16
    )
    trainer = module.Trainer(max_epochs=1)
    trainer.fit(model, data)


CHECKS = {
    "pytorch": check_pytorch,
    "tensorflow": check_tensorflow,
    "jax": check_jax,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("framework", choices=tuple(CHECKS))
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument(
        "--core-only", action="store_true",
        help="ignore optional packages not installed by the selected notebook",
    )
    parser.add_argument(
        "--provider-compatible", action="store_true",
        help="preserve provider-managed packages and enforce only hosted exact pins",
    )
    parser.add_argument(
        "--d2l-root", type=Path,
        help="prefer a revision-pinned downloaded d2l package over the worktree",
    )
    args = parser.parse_args()
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        os.environ["JAX_PLATFORMS"] = "cpu"
    if args.d2l_root:
        sys.path.insert(0, str(args.d2l_root.resolve()))
    profile = load_profile(args.framework)
    check_versions(
        profile, include_optional=not args.core_only,
        provider_compatible=args.provider_compatible,
    )
    check_apis(profile)
    if not args.core_only:
        check_installed_optional_imports(profile)
    CHECKS[args.framework](args.device)
    check_d2l_training_path(args.framework)
    print(
        f"Hosted runtime contract OK: {args.framework}/{args.device} "
        f"({profile['environment_sha256'][:12]})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
