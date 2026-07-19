#!/usr/bin/env python3
"""Tests for lock-derived hosted runtime profiles."""

import tempfile
import tomllib
import unittest
from pathlib import Path

import export_hosted_env as hosted_env


def lock_data():
    return tomllib.loads((hosted_env.ROOT / "uv.lock").read_text())


class HostedEnvironmentTests(unittest.TestCase):
    def test_profiles_are_derived_from_current_lock(self):
        lock = hosted_env.ROOT / "uv.lock"
        for framework, specs in hosted_env.PROFILES.items():
            profile = hosted_env.build_profile(framework, lock)
            packages = {
                package["distribution"]: package
                for package in profile["packages"]
            }
            self.assertEqual(set(packages), {spec.distribution for spec in specs})
            for spec in specs:
                locked = hosted_env._linux_entry(lock_data(), spec.distribution)[
                    "version"
                ]
                expected = locked.split("+", 1)[0] if spec.public_version else locked
                self.assertEqual(packages[spec.distribution]["version"], expected)
                self.assertEqual(
                    packages[spec.distribution]["hosted_policy"],
                    "exact" if spec.hosted_exact else "preserve",
                )
            self.assertRegex(profile["environment_sha256"], r"^[0-9a-f]{64}$")
            optional = {
                package["distribution"]: package
                for package in profile["optional_packages"]
            }
            self.assertEqual(
                set(optional),
                {
                    spec.distribution
                    for spec in hosted_env.OPTIONAL_PROFILES[framework]
                },
            )

    def test_generation_is_deterministic_and_checkable(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            lock = hosted_env.ROOT / "uv.lock"
            hosted_env.write_profiles(output, lock)
            first = {
                path.name: path.read_bytes() for path in output.iterdir()
            }
            hosted_env.write_profiles(output, lock)
            second = {
                path.name: path.read_bytes() for path in output.iterdir()
            }
            self.assertEqual(first, second)
            self.assertEqual(hosted_env.check_profiles(output, lock), 0)

    def test_constraints_do_not_contain_extras(self):
        profile = hosted_env.build_profile("jax", hosted_env.ROOT / "uv.lock")
        with tempfile.TemporaryDirectory() as tmp:
            hosted_env.write_profiles(Path(tmp), hosted_env.ROOT / "uv.lock")
            constraints = (Path(tmp) / "constraints-jax.txt").read_text()
        jax = next(
            package for package in profile["packages"]
            if package["distribution"] == "jax"
        )
        self.assertIn(jax["constraint"], constraints)
        self.assertNotIn("jax[cuda12]", constraints)
        self.assertEqual(jax["requirement"], f"jax=={jax['version']}")
        self.assertEqual(
            jax["gpu_requirement"], f"jax[cuda12]=={jax['version']}"
        )

    def test_tracked_profiles_are_current(self):
        self.assertEqual(
            hosted_env.check_profiles(
                hosted_env.DEFAULT_OUTPUT, hosted_env.ROOT / "uv.lock"
            ),
            0,
        )

    def test_provider_stack_is_preserved_but_nnx_stack_is_exact(self):
        pytorch = self._packages("pytorch")
        tensorflow = self._packages("tensorflow")
        jax = self._packages("jax")
        self.assertEqual(pytorch["torch"]["hosted_policy"], "preserve")
        self.assertEqual(
            tensorflow["tensorflow"]["hosted_policy"], "preserve"
        )
        self.assertEqual(jax["tensorflow"]["hosted_policy"], "preserve")
        self.assertEqual(jax["protobuf"]["hosted_policy"], "preserve")
        self.assertEqual(jax["flax"]["hosted_policy"], "exact")

    @staticmethod
    def _packages(framework):
        profile = hosted_env.build_profile(
            framework, hosted_env.ROOT / "uv.lock"
        )
        return {
            package["distribution"]: package
            for package in profile["packages"]
        }


if __name__ == "__main__":
    unittest.main()
