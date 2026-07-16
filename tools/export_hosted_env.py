#!/usr/bin/env python3
"""Export reproducible hosted-notebook runtime profiles from ``uv.lock``.

The generated JSON files are consumed by the notebook setup-cell generator and
the runtime contract tests.  The matching constraint files are useful for
reproducing the same core stack outside the repository.  Never edit generated
files by hand; update ``pyproject.toml`` / ``uv.lock`` and regenerate them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "hosted"


@dataclass(frozen=True)
class PackageSpec:
    distribution: str
    import_name: str
    extra: str | None = None
    gpu_extra: str | None = None
    public_version: bool = False
    hosted_exact: bool = True


PROFILES = {
    "pytorch": (
        PackageSpec(
            "torch", "torch", public_version=True, hosted_exact=False
        ),
        PackageSpec(
            "torchvision", "torchvision", public_version=True,
            hosted_exact=False,
        ),
    ),
    "tensorflow": (
        PackageSpec("tensorflow", "tensorflow", hosted_exact=False),
        PackageSpec("keras", "keras", hosted_exact=False),
        PackageSpec("protobuf", "google.protobuf", hosted_exact=False),
        PackageSpec("ml-dtypes", "ml_dtypes", hosted_exact=False),
    ),
    "jax": (
        PackageSpec("jax", "jax", gpu_extra="cuda12"),
        PackageSpec("jaxlib", "jaxlib"),
        PackageSpec("flax", "flax"),
        PackageSpec("optax", "optax"),
        PackageSpec("orbax-checkpoint", "orbax.checkpoint"),
        PackageSpec("tensorflow", "tensorflow", hosted_exact=False),
        PackageSpec("protobuf", "google.protobuf", hosted_exact=False),
        PackageSpec("ml-dtypes", "ml_dtypes", hosted_exact=False),
    ),
}

OPTIONAL_PROFILES = {
    "pytorch": (
        PackageSpec("gpytorch", "gpytorch"),
        PackageSpec("gymnasium", "gymnasium", extra="toy-text"),
        PackageSpec("safetensors", "safetensors"),
        PackageSpec(
            "syne-tune", "syne_tune.optimizer.baselines", extra="gpsearchers"
        ),
        PackageSpec("tiktoken", "tiktoken"),
    ),
    "tensorflow": (
        PackageSpec("gymnasium", "gymnasium", extra="toy-text"),
        PackageSpec("safetensors", "safetensors"),
        PackageSpec("tensorflow-probability", "tensorflow_probability", extra="tf"),
        # Colab manages TensorFlow. Match tf-keras to that provider version
        # instead of replacing the provider's TensorFlow/protobuf stack with
        # the version selected by the local authoring lock.
        PackageSpec("tf-keras", "tf_keras", hosted_exact=False),
        PackageSpec("tiktoken", "tiktoken"),
    ),
    "jax": (
        PackageSpec("gymnasium", "gymnasium", extra="toy-text"),
        PackageSpec("safetensors", "safetensors"),
        PackageSpec("tiktoken", "tiktoken"),
    ),
}

CRITICAL_APIS = {
    "pytorch": (
        "torch.nn.Linear",
        "torch.optim.SGD",
        "torchvision.transforms.Compose",
    ),
    "tensorflow": (
        "tensorflow.GradientTape",
        "tensorflow.data.Dataset",
        "tensorflow.keras.optimizers.SGD",
    ),
    "jax": (
        "jax.jit",
        "flax.nnx.view",
        "flax.nnx.Optimizer",
        "flax.nnx.Param",
        "optax.sgd",
        "tensorflow.config.set_visible_devices",
    ),
}


def _canonical(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()


def _lock_entries(lock: dict, distribution: str) -> list[dict]:
    normalized = distribution.replace("_", "-").lower()
    return [
        package for package in lock.get("package", [])
        if package.get("name", "").replace("_", "-").lower() == normalized
    ]


def _linux_entry(lock: dict, distribution: str) -> dict:
    entries = _lock_entries(lock, distribution)
    if not entries:
        raise ValueError(f"{distribution!r} is absent from uv.lock")
    if len(entries) == 1:
        return entries[0]

    def score(entry: dict) -> int:
        markers = " ".join(entry.get("resolution-markers", []))
        value = 0
        if "sys_platform == 'linux'" in markers:
            value += 4
        if "platform_machine == 'x86_64'" in markers:
            value += 2
        if "sys_platform == 'darwin'" in markers:
            value -= 8
        source = entry.get("source", {}).get("registry", "")
        if "pytorch.org/whl" in source:
            value += 1
        return value

    ranked = sorted(entries, key=score, reverse=True)
    if score(ranked[0]) == score(ranked[1]):
        choices = ", ".join(e.get("version", "?") for e in ranked)
        raise ValueError(
            f"cannot choose the linux-x86_64 {distribution} entry: {choices}"
        )
    return ranked[0]


def build_profile(framework: str, lock_path: Path) -> dict:
    lock_bytes = lock_path.read_bytes()
    lock = tomllib.loads(lock_bytes.decode())
    def package_record(spec: PackageSpec) -> dict:
        entry = _linux_entry(lock, spec.distribution)
        locked = entry["version"]
        wanted = locked.split("+", 1)[0] if spec.public_version else locked
        base = spec.distribution
        install_name = f"{base}[{spec.extra}]" if spec.extra else base
        record = {
            "distribution": base,
            "import": spec.import_name,
            "version": wanted,
            "match": "public" if spec.public_version else "exact",
            "hosted_policy": "exact" if spec.hosted_exact else "preserve",
            "requirement": f"{install_name}=={wanted}",
            "constraint": f"{base}=={wanted}",
        }
        if spec.gpu_extra:
            record["gpu_requirement"] = (
                f"{base}[{spec.gpu_extra}]=={wanted}"
            )
        return record

    packages = [package_record(spec) for spec in PROFILES[framework]]
    optional_packages = [
        package_record(spec) for spec in OPTIONAL_PROFILES[framework]
    ]
    profile = {
        "schema_version": 2,
        "framework": framework,
        "python": "3.12",
        "platform": "linux-x86_64",
        "source_lock": lock_path.name,
        "source_lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
        "packages": packages,
        "optional_packages": optional_packages,
        "critical_apis": list(CRITICAL_APIS[framework]),
    }
    environment = {
        key: value for key, value in profile.items()
        if key not in {"source_lock", "source_lock_sha256"}
    }
    profile["environment_sha256"] = hashlib.sha256(
        _canonical(environment)
    ).hexdigest()
    return profile


def _json_text(profile: dict) -> str:
    return json.dumps(profile, indent=2, sort_keys=True) + "\n"


def _constraints_text(profile: dict) -> str:
    lines = [
        "# Generated by tools/export_hosted_env.py from uv.lock.",
        f"# environment-sha256: {profile['environment_sha256']}",
    ]
    lines.extend(
        package["constraint"]
        for package in profile["packages"] + profile["optional_packages"]
    )
    return "\n".join(lines) + "\n"


def write_profiles(output: Path, lock_path: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for framework in PROFILES:
        profile = build_profile(framework, lock_path)
        (output / f"hosted-lock-{framework}.json").write_text(
            _json_text(profile), encoding="utf-8"
        )
        (output / f"constraints-{framework}.txt").write_text(
            _constraints_text(profile), encoding="utf-8"
        )


def check_profiles(output: Path, lock_path: Path) -> int:
    with tempfile.TemporaryDirectory(prefix="d2l-hosted-env-") as tmp:
        candidate = Path(tmp)
        write_profiles(candidate, lock_path)
        expected = {p.name: p.read_bytes() for p in candidate.iterdir()}
    actual_paths = list(output.glob("hosted-lock-*.json")) + list(
        output.glob("constraints-*.txt")
    )
    actual = {path.name: path.read_bytes() for path in actual_paths}
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        changed = sorted(
            name for name in set(expected) & set(actual)
            if expected[name] != actual[name]
        )
        print("Hosted environment files are stale.")
        for name in missing:
            print(f"  missing: {output / name}")
        for name in unexpected:
            print(f"  unexpected: {output / name}")
        for name in changed:
            print(f"  changed: {output / name}")
        print("Run: python3 tools/export_hosted_env.py generate")
        return 1
    print("Hosted environment files match uv.lock")
    return 0


def load_profile(framework: str, root: Path = ROOT) -> dict:
    path = root / "hosted" / f"hosted-lock-{framework}.json"
    profile = json.loads(path.read_text(encoding="utf-8"))
    lock_path = root / profile["source_lock"]
    lock_sha = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    if profile["source_lock_sha256"] != lock_sha:
        raise RuntimeError(
            f"{path} is stale relative to {lock_path}; run "
            "python3 tools/export_hosted_env.py generate"
        )
    if profile.get("framework") != framework:
        raise RuntimeError(f"{path} contains the wrong framework profile")
    return profile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("generate", "check"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.root / "hosted"
    lock_path = args.root / "uv.lock"
    if args.command == "generate":
        write_profiles(output, lock_path)
        print(f"Wrote hosted environment files to {output}")
        return 0
    return check_profiles(output, lock_path)


if __name__ == "__main__":
    raise SystemExit(main())
