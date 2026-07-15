#!/usr/bin/env python3
"""Focused tests for hosted-notebook classification and normalization."""

import json
import tempfile
import unittest
from pathlib import Path

import build_hosted_notebooks as hosted


class HostedNotebookTests(unittest.TestCase):
    def classify(self, text):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "page.md"
            path.write_text(text, encoding="utf-8")
            return hosted.classify_source(path)

    def test_framework_independent_code_is_numpy(self):
        self.assertEqual(
            self.classify("# Demo\n\n```{.python .input}\nimport numpy as np\n```\n"),
            ["numpy"],
        )

    def test_explicit_variants_preserve_only_pytorch_and_jax(self):
        text = """```{.python .input}\n#@tab pytorch\nimport torch\n```\n
```{.python .input}\n#@tab jax\nimport jax\n```\n
```{.python .input}\n#@tab tensorflow\nimport tensorflow\n```\n"""
        self.assertEqual(self.classify(text), ["pytorch", "jax"])

    def test_prose_only_page_has_no_notebook(self):
        self.assertEqual(self.classify("# Prose only\n"), [])

    def test_prose_tab_does_not_claim_an_unimplemented_variant(self):
        text = """```{.python .input}
#@tab pytorch
import torch
```

:begin_tab:`jax`
The JAX implementation is unavailable.
:end_tab:
"""
        self.assertEqual(self.classify(text), ["pytorch"])

    def test_normalization_removes_sync_headers_and_adds_setup(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "page.ipynb"
            path.write_text(json.dumps({
                "cells": [
                    {"cell_type": "markdown", "metadata": {},
                     "source": ["<!-- d2l:prose id=x-md-1 fw=all -->\n", "# Demo\n",
                                "![Figure](../img/example.svg)\n"]},
                    {"cell_type": "code", "metadata": {}, "source": ["from d2l import torch as d2l\n"],
                     "outputs": [], "execution_count": None},
                ],
                "metadata": {}, "nbformat": 4, "nbformat_minor": 5,
            }), encoding="utf-8")
            nb = hosted.normalize_notebook(path, "pytorch", "abc123")
            self.assertEqual(nb["cells"][0]["id"], "d2l-hosted-setup")
            markdown = "".join(nb["cells"][1]["source"])
            self.assertNotIn("d2l:prose", markdown)
            self.assertIn(
                "https://raw.githubusercontent.com/smolix/d2l-neu/notebooks/"
                "img/example.svg", markdown)
            self.assertEqual(nb["metadata"]["d2l"]["revision"], "abc123")
            self.assertEqual(nb["metadata"]["kernelspec"]["name"], "python3")


if __name__ == "__main__":
    unittest.main()
