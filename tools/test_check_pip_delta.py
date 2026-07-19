#!/usr/bin/env python3
"""Tests for hosted provider ``pip check`` delta handling."""

import tempfile
import unittest
from pathlib import Path

import check_pip_delta


class PipDeltaTests(unittest.TestCase):
    def test_ignores_provider_baseline_conflicts(self):
        with tempfile.TemporaryDirectory() as tmp:
            before = Path(tmp) / "before"
            after = Path(tmp) / "after"
            before.write_text("existing conflict\n")
            after.write_text("existing conflict\n")
            self.assertEqual(check_pip_delta.new_conflicts(before, after), [])

    def test_reports_only_introduced_conflicts(self):
        with tempfile.TemporaryDirectory() as tmp:
            before = Path(tmp) / "before"
            after = Path(tmp) / "after"
            before.write_text("existing conflict\n")
            after.write_text("new conflict\nexisting conflict\n")
            self.assertEqual(
                check_pip_delta.new_conflicts(before, after), ["new conflict"]
            )


if __name__ == "__main__":
    unittest.main()
