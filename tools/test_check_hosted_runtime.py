#!/usr/bin/env python3
"""Focused tests for hosted runtime contract helpers."""

import json.decoder
import unittest
from unittest import mock

import check_hosted_runtime as contract


class HostedRuntimeContractTests(unittest.TestCase):
    def test_public_version_ignores_local_build_label(self):
        self.assertEqual(contract.public_version("2.11.0+cu128"), "2.11.0")
        self.assertEqual(contract.public_version("0.10.2"), "0.10.2")

    def test_resolve_api_imports_longest_module_prefix(self):
        self.assertIs(
            contract.resolve_api("json.decoder.JSONDecoder"),
            json.decoder.JSONDecoder,
        )

    def test_core_version_check_ignores_optional_packages(self):
        profile = {
            "packages": [],
            "optional_packages": [{
                "distribution": "not-installed-by-this-notebook",
                "version": "99",
                "match": "exact",
            }],
        }
        with mock.patch.object(contract, "version", return_value="1"):
            contract.check_versions(profile, include_optional=False)
            with self.assertRaisesRegex(RuntimeError, "expected 99"):
                contract.check_versions(profile, include_optional=True)

    def test_provider_compatible_version_check_skips_preserved_package(self):
        profile = {
            "packages": [{
                "distribution": "provider-package",
                "version": "99",
                "match": "exact",
                "hosted_policy": "preserve",
            }],
            "optional_packages": [],
        }
        with mock.patch.object(contract, "version", return_value="1"):
            contract.check_versions(profile, provider_compatible=True)
            with self.assertRaisesRegex(RuntimeError, "expected 99"):
                contract.check_versions(profile, provider_compatible=False)


if __name__ == "__main__":
    unittest.main()
