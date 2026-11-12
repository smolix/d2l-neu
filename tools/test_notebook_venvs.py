#!/usr/bin/env python3
"""Tests for separate execution venvs (the Language Models part's .venv-lm).

docs/build-system.md §6.4.2. Notebooks under chapter_lm-* execute in .venv-lm
(PyTorch plus Transformers/OpenAI) while keeping framework "pytorch" for their
stamps, manifests and freshness key. This file checks that the pieces agree:

  * pyproject.toml: the lm extra pins the same torch/torchvision as the
    pytorch extra, carries the LM packages, sits in the uv conflict set, and
    the pytorch extra (the rest of the book) carries none of them;
  * tools/runtime_env.py routes exactly the chapter_lm-* notebooks to lm;
  * make/notebooks.mk has an EXEC_RULE for every NOTEBOOK_VENVS entry;
  * capture_outputs provenance: framework_version from the notebook's venv
    (falling back to the framework venv), the venv name and transformers
    version recorded for lm notebooks only, package drift reported but never
    a freshness key;
  * gen_notebooks writes the d2l-lm kernelspec into chapter_lm-* notebooks.

Stdlib only (no venv, no GPU): `python3 tools/test_notebook_venvs.py` or
`make test-trap`.
"""

import json
import re
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
sys.path.insert(0, str(TOOLS))
import capture_outputs as cap        # noqa: E402
import runtime_env                   # noqa: E402

LM_ONLY = ("transformers", "openai", "accelerate", "huggingface-hub")


def _requirements(extra):
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    return data["project"]["optional-dependencies"][extra]


def _pins(reqs, names):
    """{(name, marker): requirement} for the named distributions."""
    out = {}
    for req in reqs:
        spec, _, marker = req.partition(";")
        name = re.split(r"[\[=<>~! ]", spec.strip(), maxsplit=1)[0].lower()
        if name in names:
            out[(name, marker.strip())] = spec.strip()
    return out


class PyprojectTests(unittest.TestCase):
    def test_lm_extra_pins_the_pytorch_torch(self):
        names = {"torch", "torchvision"}
        pytorch = _pins(_requirements("pytorch"), names)
        lm = _pins(_requirements("lm"), names)
        self.assertTrue(pytorch)
        self.assertEqual(lm, pytorch)

    def test_lm_extra_carries_the_lm_packages(self):
        lm = _pins(_requirements("lm"), set(LM_ONLY))
        got = {name for name, _ in lm}
        self.assertEqual(got, set(LM_ONLY))
        specs = set(lm.values())
        self.assertIn("transformers[serving]==5.22.2", specs)
        self.assertIn("openai==3.49.0", specs)

    def test_rest_of_the_book_does_not_carry_them(self):
        lm_names = {"transformers", "openai", "accelerate"}
        for extra in ("pytorch", "run", "build"):
            self.assertEqual(_pins(_requirements(extra), lm_names), {},
                             f"{extra} extra installs an LM package")

    def test_lm_is_in_the_uv_conflict_set(self):
        data = tomllib.loads((ROOT / "pyproject.toml").read_text())
        groups = data["tool"]["uv"]["conflicts"]
        extras = [{c.get("extra") for c in g} for g in groups]
        self.assertTrue(any({"lm", "pytorch", "jax"} <= g for g in extras))


class RoutingTests(unittest.TestCase):
    def test_notebook_venv(self):
        nv = runtime_env.notebook_venv
        self.assertEqual(nv("pytorch", "chapter_lm-pretrained/hf-models.ipynb"),
                         "lm")
        self.assertEqual(nv("pytorch", "chapter_lm-agents"), "lm")
        self.assertEqual(nv("pytorch", "./chapter_lm-pretrained/x.md"), "lm")
        self.assertEqual(nv("pytorch", "chapter_transformers/gpt.ipynb"),
                         "pytorch")
        self.assertEqual(nv("pytorch", "chapter_language-models/x.ipynb"),
                         "pytorch")
        self.assertEqual(nv("jax", "chapter_lm-pretrained/x.ipynb"), "jax")

    def test_chapter_venv(self):
        self.assertEqual(runtime_env.chapter_venv("chapter_lm-serving"),
                         ("pytorch", "lm"))
        self.assertIsNone(runtime_env.chapter_venv("chapter_attention"))

    def test_make_has_a_rule_per_entry(self):
        mk = (ROOT / "make" / "notebooks.mk").read_text()
        rules = set(re.findall(
            r"^\$\(eval \$\(call EXEC_RULE,(\w+),(\w+),([\w-]+)\)\)",
            mk, re.M))
        want = {(fw, venv, prefix)
                for fw, entries in runtime_env.NOTEBOOK_VENVS.items()
                for prefix, venv in entries}
        self.assertEqual(rules, want)

    def test_kernels_cover_every_venv(self):
        mk = (ROOT / "make" / "venvs.mk").read_text()
        m = re.search(r"^KERNEL_VENVS := (.*)$", mk, re.M)
        self.assertIsNotNone(m)
        venvs = {v for e in runtime_env.NOTEBOOK_VENVS.values() for _, v in e}
        self.assertLessEqual(venvs, set(m.group(1).split()))


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self._orig = cap._installed_version
        cap._VERSION_CACHE.clear()
        self.calls = []

        def fake(py, pkg):
            self.calls.append((Path(py).parent.parent.name, pkg))
            return {"torch": "2.11.0+cu128",
                    "transformers": "5.22.2"}.get(pkg)
        cap._installed_version = fake
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        cap._installed_version = self._orig
        cap._VERSION_CACHE.clear()
        self.tmp.cleanup()

    def venv(self, name):
        py = self.root / f".venv-{name}" / "bin" / "python"
        py.parent.mkdir(parents=True)
        py.write_text("")

    def d(self, chapter):
        return self.root / "_notebooks" / "pytorch" / chapter / "x.d"

    def test_lm_notebook_reads_the_lm_venv(self):
        self.venv("lm")
        self.venv("pytorch")
        prov = cap.build_provenance(self.root, "pytorch",
                                    self.d("chapter_lm-pretrained"))
        self.assertEqual(prov["framework_version"], "torch==2.11.0")
        self.assertEqual(prov["venv"], "lm")
        self.assertEqual(prov["packages"],
                         {"transformers": "transformers==5.22.2"})
        self.assertIn((".venv-lm", "torch"), self.calls)
        self.assertNotIn((".venv-pytorch", "torch"), self.calls)

    def test_lm_notebook_falls_back_to_the_framework_venv(self):
        # A host with .venv-pytorch but no .venv-lm (the same torch pin) must
        # not see the part as stale; the package versions stay unknown.
        self.venv("pytorch")
        prov = cap.build_provenance(self.root, "pytorch",
                                    self.d("chapter_lm-pretrained"))
        self.assertEqual(prov["framework_version"], "torch==2.11.0")
        self.assertEqual(prov["packages"], {"transformers": "unknown"})

    def test_other_notebooks_are_unchanged(self):
        self.venv("lm")
        self.venv("pytorch")
        prov = cap.build_provenance(self.root, "pytorch",
                                    self.d("chapter_transformers"))
        self.assertEqual(set(prov), {"framework_version",
                                     "d2l_lib_fingerprint"})
        self.assertNotIn((".venv-lm", "torch"), self.calls)

    def test_package_drift_warns_only_on_known_changes(self):
        old = {"packages": {"transformers": "transformers==5.22.2"}}
        self.assertEqual(cap.package_drift(old, old), [])
        new = {"packages": {"transformers": "transformers==5.23.0"}}
        self.assertEqual(cap.package_drift(old, new),
                         ["transformers==5.22.2 → transformers==5.23.0"])
        unknown = {"packages": {"transformers": "unknown"}}
        self.assertEqual(cap.package_drift(old, unknown), [])
        self.assertEqual(cap.package_drift({}, new), [])


class AuditTests(unittest.TestCase):
    def test_package_drift_is_not_staleness(self):
        import audit_outputs
        orig = cap.build_provenance
        cur = {"framework_version": "torch==2.11.0",
               "d2l_lib_fingerprint": "sha256:x", "venv": "lm",
               "packages": {"transformers": "transformers==5.23.0"}}
        cap.build_provenance = lambda *a, **k: dict(cur)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                nb_dir = root / "_notebooks" / "pytorch" / "chapter_lm-x"
                nb_dir.mkdir(parents=True)
                code = {"cell_type": "code", "id": "c1",
                        "metadata": {"id": "c1"}, "source": "print(1)\n",
                        "outputs": []}
                (nb_dir / "p.ipynb").write_text(json.dumps(
                    {"cells": [code], "metadata": {}, "nbformat": 4,
                     "nbformat_minor": 5}))
                fps, _ = cap.prefix_fingerprints([code])
                man_dir = root / "outputs" / "pytorch" / "chapter_lm-x"
                man_dir.mkdir(parents=True)
                man = man_dir / "p.json"
                man.write_text(json.dumps({
                    "source": "chapter_lm-x/p.md",
                    "provenance": dict(
                        cur, packages={"transformers": "transformers==5.22.2"}),
                    "cells": {"c1": {"code_fingerprint": fps["c1"],
                                     "outputs": [{"type": "stream",
                                                  "text": "1\n"}]}},
                }))
                rec = audit_outputs.audit_manifest(
                    root, root / "_notebooks", man, "pytorch", None)
        finally:
            cap.build_provenance = orig
        self.assertFalse(rec["stale"])
        self.assertEqual(rec["package_drift"],
                         ["transformers==5.22.2 → transformers==5.23.0"])


class KernelspecTests(unittest.TestCase):
    def kernel(self, chapter):
        import gen_notebooks
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / chapter / "p.ipynb"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"cells": [], "metadata": {},
                                        "nbformat": 4, "nbformat_minor": 4}))
            gen_notebooks.patch_ipynb_ids(path, "pytorch")
            return json.loads(path.read_text())["metadata"]["kernelspec"]

    def test_lm_chapter_uses_the_lm_kernel(self):
        self.assertEqual(self.kernel("chapter_lm-pretrained")["name"],
                         "d2l-lm")

    def test_other_chapters_keep_the_framework_kernel(self):
        self.assertEqual(self.kernel("chapter_transformers")["name"],
                         "d2l-pytorch")


if __name__ == "__main__":
    unittest.main(verbosity=1)
