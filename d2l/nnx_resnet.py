# Copyright 2026 The Dive into Deep Learning authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Small NNX ResNet implementation with Hugging Face weight loading.

The architecture and source-key mapping follow the Apache-2.0 JAX Bonsai
ResNet implementation, adapted here for the JAX/Flax versions pinned by D2L.
"""

from __future__ import annotations

import pathlib
import re

from flax import nnx
from huggingface_hub import snapshot_download
import jax.numpy as jnp
from safetensors.flax import load_file


DEFAULT_REPO_ID = "microsoft/resnet-50"
DEFAULT_REVISION = "34c2154c194f829b11125337b98c8f5f9965ff19"
DEFAULT_FILENAME = "model.safetensors"


class Bottleneck(nnx.Module):
    def __init__(self, in_channels, channels, stride=1, downsample=None,
                 *, rngs):
        self.conv0 = nnx.Conv(in_channels, channels, (1, 1), use_bias=False,
                              rngs=rngs)
        self.bn0 = nnx.BatchNorm(channels, use_running_average=True, rngs=rngs)
        self.conv1 = nnx.Conv(channels, channels, (3, 3), strides=stride,
                              padding=1, use_bias=False, rngs=rngs)
        self.bn1 = nnx.BatchNorm(channels, use_running_average=True, rngs=rngs)
        self.conv2 = nnx.Conv(channels, 4 * channels, (1, 1), use_bias=False,
                              rngs=rngs)
        self.bn2 = nnx.BatchNorm(4 * channels, use_running_average=True,
                                 rngs=rngs)
        self.downsample = downsample

    def __call__(self, X):
        identity = X
        X = nnx.relu(self.bn0(self.conv0(X)))
        X = nnx.relu(self.bn1(self.conv1(X)))
        X = self.bn2(self.conv2(X))
        if self.downsample is not None:
            identity = self.downsample(identity)
        return nnx.relu(X + identity)


class Downsample(nnx.Module):
    def __init__(self, in_channels, out_channels, stride, *, rngs):
        self.conv = nnx.Conv(in_channels, out_channels, (1, 1), strides=stride,
                             use_bias=False, rngs=rngs)
        self.bn = nnx.BatchNorm(out_channels, use_running_average=True,
                                rngs=rngs)

    def __call__(self, X):
        return self.bn(self.conv(X))


class BlockGroup(nnx.Module):
    def __init__(self, in_channels, channels, blocks, stride, *, rngs):
        downsample = None
        if stride != 1 or in_channels != 4 * channels:
            downsample = Downsample(in_channels, 4 * channels, stride,
                                    rngs=rngs)
        layers = [Bottleneck(in_channels, channels, stride, downsample,
                             rngs=rngs)]
        layers.extend(Bottleneck(4 * channels, channels, rngs=rngs)
                      for _ in range(1, blocks))
        self.blocks = nnx.List(layers)

    def __call__(self, X):
        for block in self.blocks:
            X = block(X)
        return X


class ResNet50(nnx.Module):
    """ImageNet ResNet-50 whose parameters can be loaded from Hugging Face."""
    def __init__(self, num_classes=1000, *, rngs):
        self.stem_conv = nnx.Conv(3, 64, (7, 7), strides=2, padding=3,
                                  use_bias=False, rngs=rngs)
        self.stem_bn = nnx.BatchNorm(64, use_running_average=True, rngs=rngs)
        self.stages = nnx.List([
            BlockGroup(64, 64, 3, 1, rngs=rngs),
            BlockGroup(256, 128, 4, 2, rngs=rngs),
            BlockGroup(512, 256, 6, 2, rngs=rngs),
            BlockGroup(1024, 512, 3, 2, rngs=rngs)])
        self.fc = nnx.Linear(2048, num_classes, rngs=rngs)

    def feature_map(self, X):
        """Return the final convolutional feature map before pooling."""
        X = nnx.relu(self.stem_bn(self.stem_conv(X)))
        X = nnx.max_pool(X, (3, 3), (2, 2), padding=((1, 1), (1, 1)))
        for stage in self.stages:
            X = stage(X)
        return X

    def features(self, X):
        return self.feature_map(X).mean(axis=(1, 2))

    def __call__(self, X):
        return self.fc(self.features(X))

    @classmethod
    def from_pretrained(cls, repo_id=DEFAULT_REPO_ID, *, revision=None,
                        filename=DEFAULT_FILENAME):
        if revision is None:
            if repo_id != DEFAULT_REPO_ID:
                raise ValueError(
                    "Custom repositories require an explicit immutable "
                    "revision (preferably a commit hash).")
            revision = DEFAULT_REVISION
        directory = snapshot_download(
            repo_id=repo_id, revision=revision, allow_patterns=filename)
        path = pathlib.Path(directory) / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"Checkpoint {filename!r} not found in {repo_id}@{revision}")
        tensors = load_file(path)
        model = cls(rngs=nnx.Rngs(0))
        _load_huggingface_weights(model, tensors)
        return model


def _assign(variable, tensor, kind="default", *, source_key):
    if kind == "conv":
        tensor = tensor.transpose(2, 3, 1, 0)
    elif kind == "linear":
        tensor = tensor.T
    if variable.shape != tensor.shape:
        raise ValueError(
            f"Shape mismatch for {source_key}: checkpoint {tensor.shape}, "
            f"model {variable.shape}")
    variable[...] = tensor


def _expected_huggingface_keys(model):
    keys = {
        "resnet.embedder.embedder.convolution.weight",
        "resnet.embedder.embedder.normalization.weight",
        "resnet.embedder.embedder.normalization.bias",
        "resnet.embedder.embedder.normalization.running_mean",
        "resnet.embedder.embedder.normalization.running_var",
        "classifier.1.weight",
        "classifier.1.bias",
    }
    for stage_index, stage in enumerate(model.stages):
        for block_index, block in enumerate(stage.blocks):
            prefix = (f"resnet.encoder.stages.{stage_index}.layers."
                      f"{block_index}")
            for layer_index in range(3):
                layer = f"{prefix}.layer.{layer_index}"
                keys.add(f"{layer}.convolution.weight")
                for field in ("weight", "bias", "running_mean", "running_var"):
                    keys.add(f"{layer}.normalization.{field}")
            if block.downsample is not None:
                keys.add(f"{prefix}.shortcut.convolution.weight")
                for field in ("weight", "bias", "running_mean", "running_var"):
                    keys.add(f"{prefix}.shortcut.normalization.{field}")
    return keys


def _load_huggingface_weights(model, tensors):
    stem = {
        "resnet.embedder.embedder.convolution.weight":
            (model.stem_conv.kernel, "conv"),
        "resnet.embedder.embedder.normalization.weight":
            (model.stem_bn.scale, "default"),
        "resnet.embedder.embedder.normalization.bias":
            (model.stem_bn.bias, "default"),
        "resnet.embedder.embedder.normalization.running_mean":
            (model.stem_bn.mean, "default"),
        "resnet.embedder.embedder.normalization.running_var":
            (model.stem_bn.var, "default"),
        "classifier.1.weight": (model.fc.kernel, "linear"),
        "classifier.1.bias": (model.fc.bias, "default"),
    }
    expected = _expected_huggingface_keys(model)
    supplied = set(tensors)
    ignored = {key for key in supplied
               if key.endswith(".normalization.num_batches_tracked")}
    missing = sorted(expected - supplied)
    unexpected = sorted(supplied - expected - ignored)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing keys: {missing}")
        if unexpected:
            details.append(f"unexpected keys: {unexpected}")
        raise ValueError("Incompatible ResNet-50 checkpoint (" +
                         "; ".join(details) + ")")

    assigned = set()
    for key, (variable, kind) in stem.items():
        _assign(variable, tensors[key], kind, source_key=key)
        assigned.add(key)

    layer_pattern = re.compile(
        r"resnet\.encoder\.stages\.(\d+)\.layers\.(\d+)\."
        r"layer\.(\d+)\.(convolution|normalization)\."
        r"(weight|bias|running_mean|running_var)")
    shortcut_pattern = re.compile(
        r"resnet\.encoder\.stages\.(\d+)\.layers\.(\d+)\."
        r"shortcut\.(convolution|normalization)\."
        r"(weight|bias|running_mean|running_var)")
    for key, tensor in tensors.items():
        match = layer_pattern.fullmatch(key)
        if match:
            stage, block, layer = map(int, match.group(1, 2, 3))
            kind, field = match.group(4, 5)
            module = getattr(model.stages[stage].blocks[block],
                             f"{'conv' if kind == 'convolution' else 'bn'}{layer}")
            name = ({"weight": "kernel"} if kind == "convolution" else
                    {"weight": "scale", "bias": "bias",
                     "running_mean": "mean", "running_var": "var"})[field]
            _assign(getattr(module, name), tensor,
                    "conv" if kind == "convolution" else "default",
                    source_key=key)
            assigned.add(key)
            continue
        match = shortcut_pattern.fullmatch(key)
        if match:
            stage, block = map(int, match.group(1, 2))
            kind, field = match.group(3, 4)
            down = model.stages[stage].blocks[block].downsample
            module = down.conv if kind == "convolution" else down.bn
            name = ({"weight": "kernel"} if kind == "convolution" else
                    {"weight": "scale", "bias": "bias",
                     "running_mean": "mean", "running_var": "var"})[field]
            _assign(getattr(module, name), tensor,
                    "conv" if kind == "convolution" else "default",
                    source_key=key)
            assigned.add(key)

    if assigned != expected:
        raise ValueError(
            "Internal ResNet-50 mapping did not assign keys: "
            f"{sorted(expected - assigned)}")
