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

    def test_explicit_variants_preserve_hosted_frameworks(self):
        text = """```{.python .input}\n#@tab pytorch\nimport torch\n```\n
```{.python .input}\n#@tab jax\nimport jax\n```\n
```{.python .input}\n#@tab tensorflow\nimport tensorflow\n```\n"""
        self.assertEqual(
            self.classify(text), ["pytorch", "tensorflow", "jax"])

    def test_prose_only_page_has_no_notebook(self):
        self.assertEqual(self.classify("# Prose only\n"), [])

    def test_source_files_exclude_library_only_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            regular = root / "chapter_demo" / "page.md"
            regular.parent.mkdir(parents=True)
            regular.write_text("# Demo\n", encoding="utf-8")
            lib_only = root / hosted.LIB_ONLY_FILES[0]
            lib_only.parent.mkdir(parents=True, exist_ok=True)
            lib_only.write_text("# Library extraction only\n", encoding="utf-8")
            self.assertEqual(hosted.source_files(root), [regular])

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
                    {"cell_type": "code", "metadata": {}, "id": "x" * 100,
                     "source": ["from d2l import torch as d2l\n"],
                     "outputs": [], "execution_count": None},
                ],
                "metadata": {}, "nbformat": 4, "nbformat_minor": 4,
            }), encoding="utf-8")
            nb = hosted.normalize_notebook(path, "pytorch", "abc123")
            self.assertEqual(nb["cells"][0]["id"], "d2l-hosted-setup")
            markdown = "".join(nb["cells"][1]["source"])
            self.assertNotIn("d2l:prose", markdown)
            self.assertIn(
                "https://raw.githubusercontent.com/smolix/d2l-neu/notebooks/"
                "img/example.svg", markdown)
            self.assertEqual(nb["metadata"]["d2l"]["revision"], "abc123")
            self.assertRegex(
                nb["metadata"]["d2l"]["environment_sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(nb["metadata"]["kernelspec"]["name"], "python3")
            self.assertEqual(nb["nbformat_minor"], 5)
            ids = [cell["id"] for cell in nb["cells"]]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertTrue(all(1 <= len(cell_id) <= 64 for cell_id in ids))

    def test_jax_setup_does_not_install_tensorflow_datasets(self):
        source = "".join(hosted._setup_cell("jax", "abc123")["source"])
        profile = hosted.load_profile("jax")
        for package in profile["packages"]:
            self.assertIn(package["requirement"], source)
        jax = next(
            package for package in profile["packages"]
            if package["distribution"] == "jax"
        )
        self.assertIn(jax["gpu_requirement"], source)
        self.assertIn('D2L_HOSTED_DEVICE', source)
        self.assertIn('CUDA_VISIBLE_DEVICES', source)
        self.assertIn('JAX_PLATFORMS', source)
        self.assertIn("actual != wanted", source)
        self.assertIn('Path(".d2l-hosted") / "abc123"', source)
        self.assertIn('os.environ["PYTHONPATH"]', source)
        self.assertNotIn("tensorflow-datasets", source)
        self.assertNotIn("tensorflow_datasets", source)
        self.assertIn("tensorflow-cpu", source)
        self.assertIn("--no-deps", source)

    def test_tensorflow_setup_uses_locked_compatibility_stack(self):
        source = "".join(hosted._setup_cell("tensorflow", "abc123")["source"])
        self.assertIn("'__init__.py'", source)
        self.assertIn("'tensorflow.py'", source)
        profile = hosted.load_profile("tensorflow")
        for package in profile["packages"]:
            self.assertIn(package["requirement"], source)
            self.assertIn(package["requirement"], source.split("fallbacks =", 1)[1])
        self.assertNotIn("tensorflow-datasets", source)
        self.assertNotIn("tensorflow_datasets", source)
        self.assertIn("tensorflow-cpu", source)

    def test_pytorch_setup_does_not_overlay_tensorflow_cpu(self):
        source = "".join(hosted._setup_cell("pytorch", "abc123")["source"])
        self.assertIn("'pytorch' in (\"tensorflow\", \"jax\")", source)

    def test_tensorflow_probability_matches_runtime_tf_keras(self):
        source = "".join(hosted._setup_cell(
            "tensorflow", "abc123", ("tensorflow-probability",))["source"])
        self.assertIn("tensorflow_probability", source)
        optional = {
            package["distribution"]: package
            for package in hosted.load_profile("tensorflow")["optional_packages"]
        }
        self.assertIn(optional["tensorflow-probability"]["requirement"], source)
        self.assertIn(optional["tf-keras"]["requirement"], source)
        self.assertIn("'tf-keras': 'tf_keras'", source)
        self.assertIn('fallbacks["tf-keras"] = f"tf-keras=={tensorflow_version}"',
                      source)

    def test_setup_fetches_referenced_d2l_submodules(self):
        source = "".join(hosted._setup_cell(
            "jax", "abc123", helper_modules=("nnx_resnet",))["source"])
        self.assertIn("'nnx_resnet.py'", source)

    def test_urls_use_kaggle_dynamic_importer(self):
        urls = hosted._urls("jax/chapter_demo/example.ipynb")
        self.assertEqual(
            urls["kaggle"],
            "https://www.kaggle.com/kernels/welcome?src="
            "https%3A%2F%2Fraw.githubusercontent.com%2Fsmolix%2Fd2l-neu%2F"
            "notebooks%2Fjax%2Fchapter_demo%2Fexample.ipynb",
        )

    def test_canonical_kaggle_url_overrides_dynamic_importer(self):
        canonical = "https://www.kaggle.com/code/d2l/example"
        self.assertEqual(
            hosted._urls("numpy/chapter_demo/example.ipynb", canonical)["kaggle"],
            canonical,
        )

    def test_normalization_adds_page_specific_dependencies(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "page.ipynb"
            path.write_text(json.dumps({
                "cells": [{"cell_type": "code", "metadata": {},
                           "source": ["from d2l import torch as d2l\n",
                                      "import syne_tune\n"],
                           "outputs": [], "execution_count": None}],
                "metadata": {}, "nbformat": 4, "nbformat_minor": 5,
            }), encoding="utf-8")
            nb = hosted.normalize_notebook(path, "pytorch", "abc123")
            setup = "".join(nb["cells"][0]["source"])
            self.assertIn("syne-tune", setup)
            self.assertIn("'syne-tune': 'syne_tune'", setup)


if __name__ == "__main__":
    unittest.main()
