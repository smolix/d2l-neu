#!/usr/bin/env python3
"""Build provider-neutral notebooks for the public ``notebooks`` branch.

The authoring branch remains the source of truth.  This program takes the
single-framework notebooks produced by :mod:`tools.gen_notebooks`, removes
authoring-only metadata, adds a revision-pinned setup cell when the local
``d2l`` package is used, copies referenced images, and writes a deterministic
manifest consumed by the book site.

It deliberately publishes PyTorch, TensorFlow, JAX, and genuinely
framework-neutral NumPy notebooks.  MXNet may remain in the book, but it is
not a hosted-notebook target.

Examples
--------
Build a staging tree and the HTML data include::

    python3 tools/build_hosted_notebooks.py build

Verify that rebuilding would not change it::

    python3 tools/build_hosted_notebooks.py check

The generated tree is suitable for replacing the contents of an orphan Git
branch.  Publication is intentionally handled by ``publish-notebooks-branch``
in the Makefile so a normal book build never pushes Git state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "_notebooks"
DEFAULT_OUTPUT = ROOT / "_hosted_notebooks"
DEFAULT_DATA = ROOT / "_d2l-notebooks-data.html"
DEFAULT_KAGGLE_MAP = ROOT / "hosted_notebooks_kaggle.json"
REPOSITORY = "smolix/d2l-neu"
BRANCH = "notebooks"
HOSTED_FRAMEWORKS = ("pytorch", "tensorflow", "jax")
# Hosted JAX notebooks download d2l/jax.py from the build revision, so their
# runtime must match the NNX stack against which that helper was generated.
# TensorFlow is part of this stack because d2l/jax.py uses it for data loading;
# pinning it also gives TF and its Protobuf runtime a coherent version pair.
HOSTED_JAX_VERSIONS = {
    "jax": ("0.10.2", "jax[cuda12]==0.10.2"),
    "flax": ("0.12.7", "flax==0.12.7"),
    "optax": ("0.2.8", "optax==0.2.8"),
    "tensorflow": ("2.21.0", "tensorflow==2.21.0"),
}

_CODE_BLOCK_RE = re.compile(r"```\{\.python[^\n]*\}\n(.*?)```", re.DOTALL)
_TAB_RE = re.compile(r"^(?:%%tab|#@tab)\s+([^\n]+)$", re.MULTILINE)
_INTERACT_RE = re.compile(r"tab\.interact_select\(([^)]+)\)")
_FRAMEWORK_CODE_RE = re.compile(
    r"(?:\bimport\s+(?:torch|torchvision|jax|flax|optax|tensorflow|mxnet)\b|"
    r"\bfrom\s+(?:torch|torchvision|jax|flax|tensorflow|mxnet|d2l)\b)")
_PROSE_HEADER_RE = re.compile(
    r"^<!--\s*d2l:prose\s+id=[^\s]+\s+fw=[^\s]+\s*-->\s*\n?",
    re.MULTILINE,
)
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\((\.\./img/[^)\s]+)")
_IMPORT_RE = re.compile(
    r"^\s*(?:from\s+([A-Za-z_]\w*)|import\s+([A-Za-z_]\w*))",
    re.MULTILINE,
)
_CELL_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_OPTIONAL_PACKAGES = {
    "gpytorch": "gpytorch",
    "gymnasium": "gymnasium",
    "orbax": "orbax-checkpoint",
    "safetensors": "safetensors",
    "syne_tune": "syne-tune",
    "tensorflow_probability": "tensorflow-probability",
    "tiktoken": "tiktoken",
}
_PACKAGE_IMPORTS = {
    "orbax-checkpoint": "orbax",
    "pillow": "PIL",
    "syne-tune": "syne_tune",
    "tensorflow-probability": "tensorflow_probability",
}


def git_revision(root: Path = ROOT) -> str:
    """Return the immutable source revision recorded in generated notebooks."""
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def source_files(root: Path = ROOT) -> list[Path]:
    """Return code-bearing book sources and notebook-only supplements."""
    return sorted(root.glob("chapter_*/*.md"))


def classify_source(path: Path) -> list[str]:
    """Return hosted variants supported by *path*.

    Untagged scientific-Python notebooks become a single NumPy variant.  Any
    explicit framework branch or framework import opts the source out of that
    fallback; such a source receives only the PyTorch/TensorFlow/JAX variants
    it actually implements.
    """
    text = path.read_text(encoding="utf-8")
    code = "\n".join(_CODE_BLOCK_RE.findall(text))
    if not code.strip():
        return []

    explicit: set[str] = set()
    has_specific = False
    for match in _TAB_RE.finditer(text):
        names = {name.strip() for name in match.group(1).split(",")}
        specific = names - {"all"}
        explicit.update(specific & set(HOSTED_FRAMEWORKS))
        has_specific = has_specific or bool(specific)
    for match in _INTERACT_RE.finditer(text):
        names = set(re.findall(r"['\"]([\w-]+)['\"]", match.group(1)))
        explicit.update(names & set(HOSTED_FRAMEWORKS))
        has_specific = has_specific or bool(names - {"all"})

    if has_specific:
        return [variant for variant in HOSTED_FRAMEWORKS if variant in explicit]
    if _FRAMEWORK_CODE_RE.search(code):
        variants = []
        if re.search(r"\b(?:torch|torchvision)\b|from\s+d2l\s+import\s+torch", code):
            variants.append("pytorch")
        if re.search(
                r"\b(?:tensorflow|keras)\b|from\s+d2l\s+import\s+tensorflow",
                code):
            variants.append("tensorflow")
        if re.search(r"\b(?:jax|flax|optax)\b|from\s+d2l\s+import\s+jax", code):
            variants.append("jax")
        return variants
    return ["numpy"]


def _source_string(value) -> str:
    return "".join(value) if isinstance(value, list) else (value or "")


def _normalized_cell_id(cell: dict, index: int, used: set[str]) -> str:
    """Return a deterministic, unique nbformat-compatible cell ID."""
    raw = str(cell.get("id") or cell.get("metadata", {}).get("id") or "")
    if _CELL_ID_RE.fullmatch(raw) and raw not in used:
        used.add(raw)
        return raw

    stem = re.sub(r"[^A-Za-z0-9_-]+", "-", raw).strip("-_")
    if not stem:
        stem = "d2l-" + str(cell.get("cell_type", "cell"))
    basis = f"{raw}\0{index}\0{_source_string(cell.get('source'))}"
    nonce = 0
    while True:
        material = basis if nonce == 0 else f"{basis}\0{nonce}"
        suffix = hashlib.sha256(material.encode()).hexdigest()[:12]
        candidate = f"{stem[:51]}-{suffix}"
        if candidate not in used:
            used.add(candidate)
            return candidate
        nonce += 1


def _setup_cell(variant: str, revision: str,
                optional_packages: tuple[str, ...] = ()) -> dict:
    common = "numpy pandas matplotlib requests scipy pillow regex"
    if variant == "pytorch":
        module = "torch"
        packages = f"{common} torch torchvision".split()
    elif variant == "tensorflow":
        module = "tensorflow"
        packages = f"{common} tensorflow".split()
    else:
        module = "jax"
        packages = f"{common} jax flax optax tensorflow".split()
    pinned = HOSTED_JAX_VERSIONS if variant == "jax" else {}
    packages.extend(optional_packages)
    imports = {package: _PACKAGE_IMPORTS[package]
               for package in packages if package in _PACKAGE_IMPORTS}
    source = f'''# Hosted D2L setup: fetch the exact helper module used to build this notebook.
from pathlib import Path
from urllib.request import urlretrieve
from importlib.metadata import PackageNotFoundError, version
import importlib.util, subprocess, sys

required = {packages!r}
imports = {imports!r}
pinned = {pinned!r}
missing = []
for package in required:
    if package in pinned:
        wanted, requirement = pinned[package]
        try:
            installed = version(package)
        except PackageNotFoundError:
            installed = None
        if installed != wanted:
            missing.append(requirement)
    elif importlib.util.find_spec(imports.get(package, package)) is None:
        missing.append(package)
# TensorFlow Probability still uses the legacy TF-Keras package. If this page
# needs it, match TF-Keras to Colab's existing TensorFlow minor release instead
# of allowing pip to replace the provider's coherent TensorFlow/CUDA stack.
if "tensorflow-probability" in required and (
        importlib.util.find_spec("tensorflow_probability") is None or
        importlib.util.find_spec("tf_keras") is None):
    import tensorflow as _tf
    tf_minor = ".".join(_tf.__version__.split(".")[:2])
    missing = [p for p in missing if p != "tensorflow-probability"]
    missing.extend(["tensorflow-probability", f"tf-keras~={{tf_minor}}.0"])
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])

root = Path(".d2l-hosted") / "{revision}"
package = root / "d2l"
package.mkdir(parents=True, exist_ok=True)
base = "https://raw.githubusercontent.com/{REPOSITORY}/{revision}/d2l"
for name in ("__init__.py", "{module}.py"):
    target = package / name
    if not target.exists():
        urlretrieve(f"{{base}}/{{name}}", target)
if str(root.resolve()) not in sys.path:
    sys.path.insert(0, str(root.resolve()))
'''
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": "d2l-hosted-setup",
        "metadata": {"tags": ["d2l-hosted-setup"]},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def normalize_notebook(src: Path, variant: str, revision: str) -> dict:
    """Return the stable public representation of one generated notebook."""
    nb = json.loads(src.read_text(encoding="utf-8"))
    cells = []
    uses_d2l = False
    optional_packages = set()
    for cell in nb.get("cells", []):
        source = _source_string(cell.get("source"))
        if cell.get("cell_type") == "markdown":
            source = _PROSE_HEADER_RE.sub("", source)
            image_base = (
                f"https://raw.githubusercontent.com/{REPOSITORY}/{BRANCH}/img/"
            )
            source = re.sub(r"\.\./img/([^)\s]+)",
                            lambda match: image_base + match.group(1), source)
            cell["source"] = source.splitlines(keepends=True)
        elif cell.get("cell_type") == "code":
            uses_d2l = uses_d2l or bool(re.search(r"\bfrom\s+d2l\b|\bimport\s+d2l\b", source))
            modules = {match.group(1) or match.group(2)
                       for match in _IMPORT_RE.finditer(source)}
            optional_packages.update(
                _OPTIONAL_PACKAGES[module]
                for module in modules if module in _OPTIONAL_PACKAGES)
            cell.setdefault("outputs", [])
            cell.setdefault("execution_count", None)
        cells.append(cell)

    if uses_d2l and variant in set(HOSTED_FRAMEWORKS):
        cells.insert(0, _setup_cell(
            variant, revision, tuple(sorted(optional_packages))))

    used_ids: set[str] = set()
    for index, cell in enumerate(cells):
        cell["id"] = _normalized_cell_id(cell, index, used_ids)
    nb["cells"] = cells
    if nb.get("nbformat") == 4:
        nb["nbformat_minor"] = max(5, nb.get("nbformat_minor", 0))
    metadata = nb.setdefault("metadata", {})
    metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    metadata["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}
    metadata["d2l"] = {
        "repository": REPOSITORY,
        "revision": revision,
        "variant": variant,
        "generated": True,
    }
    return nb


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=1, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _kaggle_import_url(raw: str) -> str:
    """Return Kaggle's dynamic importer for a public notebook URL."""
    return "https://www.kaggle.com/kernels/welcome?src=" + quote(raw, safe="")


def _urls(path: str, kaggle: str | None = None) -> dict[str, str]:
    github = f"https://github.com/{REPOSITORY}/blob/{BRANCH}/{path}"
    raw = f"https://raw.githubusercontent.com/{REPOSITORY}/{BRANCH}/{path}"
    colab = f"https://colab.research.google.com/github/{REPOSITORY}/blob/{BRANCH}/{path}"
    return {
        "github": github,
        "raw": raw,
        "colab": colab,
        "kaggle": kaggle or _kaggle_import_url(raw),
    }


def load_kaggle_map(path: Path) -> dict:
    """Load optional canonical Kaggle URL overrides, or an empty mapping."""
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def manifest_from_sources(root: Path, revision: str, kaggle_map: dict) -> dict:
    """Build the fast site manifest without requiring generated notebooks."""
    pages = {}
    for source in source_files(root):
        variants = classify_source(source)
        if not variants:
            continue
        rel = source.relative_to(root)
        page_key = str(rel.with_suffix(""))
        entry = {"variants": {}}
        for variant in variants:
            public_rel = Path(variant) / rel.with_suffix(".ipynb")
            kaggle = kaggle_map.get(page_key, {}).get(variant)
            entry["variants"][variant] = {
                "path": public_rel.as_posix(),
                **_urls(public_rel.as_posix(), kaggle),
            }
        pages[page_key] = entry
    return {
        "schema_version": 1,
        "repository": REPOSITORY,
        "branch": BRANCH,
        "source_revision": revision,
        "pages": pages,
    }


def build_tree(root: Path, input_dir: Path, output_dir: Path, revision: str,
               kaggle_map: dict) -> dict:
    """Build *output_dir* and return its page-key manifest."""
    manifest: dict[str, dict] = {}
    referenced_images: set[Path] = set()

    for source in source_files(root):
        variants = classify_source(source)
        if not variants:
            continue
        rel = source.relative_to(root)
        page_key = str(rel.with_suffix(""))
        entry = {"variants": {}}
        for variant in variants:
            generated_variant = "pytorch" if variant == "numpy" else variant
            src_nb = input_dir / generated_variant / rel.with_suffix(".ipynb")
            if not src_nb.exists():
                continue
            public_rel = Path(variant) / rel.with_suffix(".ipynb")
            notebook = normalize_notebook(src_nb, variant, revision)
            _write_json(output_dir / public_rel, notebook)
            source_text = source.read_text(encoding="utf-8")
            for match in _IMAGE_RE.finditer(source_text):
                image = (source.parent / match.group(1)).resolve()
                try:
                    image.relative_to(root / "img")
                except ValueError:
                    continue
                if image.exists():
                    referenced_images.add(image)
            kaggle = kaggle_map.get(page_key, {}).get(variant)
            urls = _urls(public_rel.as_posix(), kaggle)
            entry["variants"][variant] = {"path": public_rel.as_posix(), **urls}
        if entry["variants"]:
            manifest[page_key] = entry

    for image in sorted(referenced_images):
        rel = image.relative_to(root / "img")
        target = output_dir / "img" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, target)

    payload = {
        "schema_version": 1,
        "repository": REPOSITORY,
        "branch": BRANCH,
        "source_revision": revision,
        "pages": manifest,
    }
    _write_json(output_dir / "manifest.json", payload)
    readme = (
        "# Dive into Deep Learning — generated notebooks\n\n"
        "This orphan branch is generated from the book's Markdown sources. "
        "Do not edit notebooks here; submit source changes to the main branch.\n\n"
        f"Source revision: `{revision}`\n"
    )
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    return payload


def write_data_include(payload: dict, path: Path) -> None:
    """Write the small JavaScript manifest included by every HTML page."""
    pages = payload.get("pages", {})
    encoded = json.dumps(pages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    path.write_text(
        "<script>window.D2L_NOTEBOOKS_MANIFEST=" + encoded + ";</script>\n",
        encoding="utf-8",
    )


def tree_digest(path: Path) -> str:
    digest = hashlib.sha256()
    if not path.exists():
        return "missing"
    for child in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(child.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(child.read_bytes())
    return digest.hexdigest()


def _replace_tree(staged: Path, destination: Path) -> None:
    """Replace a generated directory without exposing a partial new tree."""
    backup = destination.with_name(destination.name + ".previous")
    if backup.exists():
        shutil.rmtree(backup)
    if destination.exists():
        destination.rename(backup)
    try:
        staged.rename(destination)
    except Exception:
        if backup.exists() and not destination.exists():
            backup.rename(destination)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def build(args) -> int:
    revision = args.revision or git_revision(args.root)
    kaggle_map = load_kaggle_map(args.kaggle_map)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="d2l-hosted-build-",
                                     dir=args.output.parent) as tmp:
        staged = Path(tmp) / args.output.name
        payload = build_tree(args.root, args.input, staged, revision, kaggle_map)
        _replace_tree(staged, args.output)
    write_data_include(payload, args.data_file)
    variants = sum(len(p["variants"]) for p in payload["pages"].values())
    print(f"Hosted notebooks: {variants} variants for {len(payload['pages'])} pages")
    print(f"Staged in {args.output}")
    return 0


def check(args) -> int:
    if not args.output.exists():
        print(f"ERROR: {args.output} does not exist; run the build command first")
        return 1
    with tempfile.TemporaryDirectory(prefix="d2l-hosted-check-") as tmp:
        candidate = Path(tmp)
        revision = args.revision or git_revision(args.root)
        kaggle_map = load_kaggle_map(args.kaggle_map)
        payload = build_tree(args.root, args.input, candidate, revision, kaggle_map)
        if tree_digest(candidate) != tree_digest(args.output):
            print("ERROR: hosted notebook staging tree is stale")
            return 1
        expected = candidate / "data.html"
        write_data_include(payload, expected)
        if not args.data_file.exists() or expected.read_bytes() != args.data_file.read_bytes():
            print("ERROR: hosted notebook HTML manifest is stale")
            return 1
    print("Hosted notebook staging tree and HTML manifest are current")
    return 0


def manifest(args) -> int:
    revision = args.revision or git_revision(args.root)
    kaggle_map = load_kaggle_map(args.kaggle_map)
    payload = manifest_from_sources(args.root, revision, kaggle_map)
    write_data_include(payload, args.data_file)
    variants = sum(len(p["variants"]) for p in payload["pages"].values())
    print(f"Notebook site manifest: {variants} variants for {len(payload['pages'])} pages")
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "check", "manifest"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--kaggle-map", type=Path, default=DEFAULT_KAGGLE_MAP,
                        help="optional JSON page/variant canonical Kaggle URL overrides")
    parser.add_argument("--revision", help="source Git revision (defaults to HEAD)")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.command == "build":
        raise SystemExit(build(arguments))
    if arguments.command == "check":
        raise SystemExit(check(arguments))
    raise SystemExit(manifest(arguments))
