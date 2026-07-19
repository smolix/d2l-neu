#!/usr/bin/env python3
"""Preflight every published notebook without rerunning long training jobs.

The check executes each distinct hosted setup cell, compiles every code cell
with IPython's own transformer, imports every referenced module/name, and
resolves module API attribute chains found in notebook code.  It is intended
to catch environment and removed-API failures across the complete public
notebook inventory.  A separate small runtime contract exercises actual model
construction, optimization, devices, and the introductory ``Trainer.fit``
path.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import re
import sys
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from IPython.core.inputtransformer2 import TransformerManager


FRAMEWORKS = ("pytorch", "tensorflow", "jax", "numpy")
CELL_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def source_text(cell: dict) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else value


def transformed_ast(source: str, transformer: TransformerManager) -> ast.AST:
    return ast.parse(transformer.transform_cell(source))


def import_requests(tree: ast.AST) -> list[tuple[str, tuple[str, ...]]]:
    requests = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            requests.extend((item.name, ()) for item in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = tuple(item.name for item in node.names if item.name != "*")
            requests.append((node.module, names))
    return requests


def update_module_aliases(tree: ast.Module, aliases: dict[str, str]) -> None:
    """Update notebook-global import aliases from top-level statements."""
    for node in tree.body:
        if isinstance(node, ast.Import):
            for item in node.names:
                bound = item.asname or item.name.split(".", 1)[0]
                aliases[bound] = item.name if item.asname else bound
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            for item in node.names:
                if item.name != "*":
                    aliases[item.asname or item.name] = f"{node.module}.{item.name}"
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    aliases.pop(target.id, None)


def attribute_chain(node: ast.Attribute) -> tuple[str, list[str]] | None:
    parts = []
    value: ast.AST = node
    while isinstance(value, ast.Attribute):
        parts.append(value.attr)
        value = value.value
    if not isinstance(value, ast.Name):
        return None
    return value.id, list(reversed(parts))


def api_requests(tree: ast.AST, aliases: dict[str, str]) -> set[str]:
    paths = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        chain = attribute_chain(node)
        if not chain:
            continue
        root, parts = chain
        target = aliases.get(root)
        if not target or target == "d2l" or target.startswith("d2l."):
            continue
        paths.add(".".join([target, *parts]))
    return paths


def resolve_path(path: str):
    parts = path.split(".")
    last_error = None
    # Start from the module object the notebook imported. Frameworks such as
    # TensorFlow expose public lazy API objects (for example ``tf.distribute``)
    # that differ from importing the same dotted package directly.
    for boundary in range(1, len(parts) + 1):
        module_name = ".".join(parts[:boundary])
        try:
            value = importlib.import_module(module_name)
        except ModuleNotFoundError as error:
            if not (error.name == module_name or
                    module_name.startswith(f"{error.name}.")):
                raise
            last_error = error
            continue
        for part in parts[boundary:]:
            value = getattr(value, part)
        return value
    if last_error:
        raise last_error
    raise ModuleNotFoundError(path)


def check_import(module_name: str, names: tuple[str, ...]) -> None:
    module = importlib.import_module(module_name)
    for name in names:
        try:
            getattr(module, name)
        except AttributeError:
            importlib.import_module(f"{module_name}.{name}")


def execute_unique_setups(records: list[tuple[Path, dict]]) -> list[dict]:
    groups: dict[str, dict] = {}
    for path, notebook in records:
        for cell in notebook.get("cells", []):
            if "d2l-hosted-setup" not in cell.get("metadata", {}).get("tags", []):
                continue
            source = source_text(cell)
            digest = hashlib.sha256(source.encode()).hexdigest()
            group = groups.setdefault(digest, {"source": source, "notebooks": []})
            group["notebooks"].append(str(path))
    failures = []
    for digest, group in groups.items():
        try:
            exec(compile(group["source"], f"<hosted-setup-{digest[:12]}>", "exec"),
                 {"__name__": "__d2l_hosted_setup__"})
        except Exception as error:
            failures.append({
                "kind": "setup",
                "setup_sha256": digest,
                "notebooks": group["notebooks"],
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            })
    print(f"Executed {len(groups)} distinct setup cells", flush=True)
    return failures


def preflight(root: Path, framework: str, execute_setup: bool,
              d2l_root: Path | None) -> dict:
    notebook_root = root / framework
    paths = sorted(notebook_root.glob("chapter_*/*.ipynb"))
    records = []
    failures = []
    transformer = TransformerManager()
    imports: dict[tuple[str, tuple[str, ...]], set[str]] = defaultdict(set)
    apis: dict[str, set[str]] = defaultdict(set)

    for path in paths:
        rel = path.relative_to(notebook_root).as_posix()
        try:
            notebook = json.loads(path.read_text(encoding="utf-8"))
        except Exception as error:
            failures.append({"kind": "notebook-json", "notebook": rel,
                             "error": f"{type(error).__name__}: {error}"})
            continue
        records.append((path, notebook))
        cells = notebook.get("cells", [])
        ids = [str(cell.get("id", "")) for cell in cells]
        if len(ids) != len(set(ids)) or any(not CELL_ID_RE.fullmatch(i) for i in ids):
            failures.append({"kind": "cell-ids", "notebook": rel,
                             "error": "cell IDs are missing, invalid, or duplicated"})
        aliases: dict[str, str] = {}
        for index, cell in enumerate(cells):
            if cell.get("cell_type") != "code":
                continue
            if "d2l-hosted-setup" in cell.get("metadata", {}).get("tags", []):
                continue
            try:
                tree = transformed_ast(source_text(cell), transformer)
            except Exception as error:
                failures.append({
                    "kind": "compile", "notebook": rel, "cell": index,
                    "error": f"{type(error).__name__}: {error}",
                })
                continue
            for request in import_requests(tree):
                imports[request].add(rel)
            update_module_aliases(tree, aliases)
            for api in api_requests(tree, aliases):
                apis[api].add(rel)

    if execute_setup:
        failures.extend(execute_unique_setups(records))
    if d2l_root:
        sys.path.insert(0, str(d2l_root.resolve()))

    for (module_name, names), notebooks in sorted(imports.items()):
        try:
            check_import(module_name, names)
        except Exception as error:
            failures.append({
                "kind": "import", "module": module_name, "names": list(names),
                "notebooks": sorted(notebooks),
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            })
    for path, notebooks in sorted(apis.items()):
        try:
            resolve_path(path)
        except Exception as error:
            failures.append({
                "kind": "api", "api": path, "notebooks": sorted(notebooks),
                "error": f"{type(error).__name__}: {error}",
            })

    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "framework": framework,
        "notebooks": len(paths),
        "distinct_imports": len(imports),
        "distinct_api_paths": len(apis),
        "status": "pass" if not failures else "fail",
        "failures": failures,
    }
    print(
        f"Hosted notebook preflight {result['status'].upper()}: {framework}: "
        f"{len(paths)} notebooks, {len(imports)} imports, {len(apis)} API paths, "
        f"{len(failures)} failures",
        flush=True,
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("framework", choices=FRAMEWORKS)
    parser.add_argument("--root", type=Path, default=Path("_hosted_notebooks"))
    parser.add_argument("--execute-setup", action="store_true")
    parser.add_argument("--d2l-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = preflight(args.root, args.framework, args.execute_setup, args.d2l_root)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if result["failures"]:
        for failure in result["failures"]:
            location = failure.get("notebook") or ", ".join(
                failure.get("notebooks", [])[:3]
            )
            print(f"  {failure['kind']}: {location}: {failure['error']}")
    return int(result["status"] != "pass")


if __name__ == "__main__":
    raise SystemExit(main())
