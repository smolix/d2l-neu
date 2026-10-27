#!/usr/bin/env python3
"""Report mechanical risks in D2L figures and Markdown references.

This is a review aid, not an aesthetic score. It checks facts a script can
establish and leaves composition, teaching value, and technical meaning to a
human reviewer.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import struct
import sys
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parent.parent
IMAGE_RE = re.compile(r"!\[(.*?)\]\((\.\./img/[^)]+)\)")
LABEL_RE = re.compile(r"^\s*:label:`fig_[^`]+`\s*$")
SENTENCE_RE = re.compile(r"[.!?](?:\s|$)")
STYLE_SIZE_RE = re.compile(r"font-size\s*:\s*([0-9.]+)(px|pt)?")
SCALE_RE = re.compile(r"\bscale\(\s*([0-9.]+)(?:[ ,]+-?[0-9.]+)?\s*\)")
HEX_RE = re.compile(r"#[0-9a-fA-F]{6}\b")


@dataclass(frozen=True)
class Issue:
    severity: str
    path: Path
    message: str
    line: int = 1


def rel(path):
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def svg_number(value):
    match = re.match(r"\s*([0-9.]+)", value or "")
    return float(match.group(1)) if match else None


def svg_dimensions(root):
    viewbox = root.attrib.get("viewBox", "").replace(",", " ").split()
    if len(viewbox) == 4:
        return float(viewbox[2]), float(viewbox[3])
    return svg_number(root.attrib.get("width")), svg_number(root.attrib.get("height"))


def png_dimensions(path):
    with path.open("rb") as handle:
        signature = handle.read(24)
    if signature[:8] != b"\x89PNG\r\n\x1a\n" or signature[12:16] != b"IHDR":
        return None
    return struct.unpack(">II", signature[16:24])


def jpeg_dimensions(path):
    data = path.read_bytes()
    if data[:2] != b"\xff\xd8":
        return None
    offset = 2
    starts = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
              0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
    while offset + 9 < len(data):
        if data[offset] != 0xFF:
            offset += 1
            continue
        marker = data[offset + 1]
        offset += 2
        if marker in {0xD8, 0xD9}:
            continue
        length = int.from_bytes(data[offset:offset + 2], "big")
        if marker in starts:
            height = int.from_bytes(data[offset + 3:offset + 5], "big")
            width = int.from_bytes(data[offset + 5:offset + 7], "big")
            return width, height
        offset += max(length, 2)
    return None


def audit_svg(path):
    issues = []
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
        root = ET.fromstring(content)
    except (OSError, ET.ParseError) as error:
        return [Issue("error", path, f"cannot parse SVG: {error}")], "unreadable SVG"

    width, height = svg_dimensions(root)
    aspect = width / height if width and height else None
    summary = f"SVG {width:g}×{height:g}" if width and height else "SVG unknown size"
    if aspect:
        summary += f", aspect {aspect:.2f}:1"
        if aspect > 2.0:
            issues.append(Issue(
                "warning", path,
                f"wide labeled figure ({aspect:.2f}:1): inspect at 390 px; "
                "consider stacking, splitting, or a narrow variant"))
        elif aspect < 0.42:
            issues.append(Issue(
                "warning", path,
                f"very tall figure ({aspect:.2f}:1): inspect page breaks and slides"))

    namespace = "{http://www.w3.org/2000/svg}"
    title = root.find(namespace + "title")
    desc = root.find(namespace + "desc")
    if title is None or not (title.text or "").strip():
        issues.append(Issue("note", path, "SVG has no concise <title>"))
    if desc is None or not (desc.text or "").strip():
        issues.append(Issue("note", path, "SVG has no explanatory <desc>"))
    if "<image" in content:
        issues.append(Issue(
            "warning", path,
            "SVG embeds a raster image; verify effective resolution and necessity"))

    sizes = []
    for value, unit in STYLE_SIZE_RE.findall(content):
        size = float(value)
        sizes.append(size * 0.75 if unit == "px" else size)
    # Matplotlib path text appears in groups scaled by font_size / 100.
    for value in SCALE_RE.findall(content):
        scale = float(value)
        if 0.05 <= scale <= 0.40:
            sizes.append(scale * 100)
    if sizes:
        minimum = min(sizes)
        summary += f", inferred min text {minimum:.1f} pt"
        if minimum < 8.5:
            issues.append(Issue(
                "warning", path,
                f"inferred text as small as {minimum:.1f} pt before page scaling"))

    colors = {color.lower() for color in HEX_RE.findall(content)}
    neutral = {"#000000", "#ffffff", "#7f7f7f", "#cfcfcf", "#808080", "#d3d3d3",
               "#1b2533", "#5c6878", "#aab4c0", "#d9e0e8", "#f5f7fa"}
    if len(colors - neutral) > 7:
        issues.append(Issue(
            "note", path,
            f"uses {len(colors - neutral)} chromatic colors; verify each has a role"))
    return issues, summary


def audit_raster(path):
    issues = []
    if path.suffix.lower() == ".png":
        dimensions = png_dimensions(path)
    elif path.suffix.lower() in {".jpg", ".jpeg"}:
        dimensions = jpeg_dimensions(path)
    else:
        dimensions = None
    if not dimensions:
        return issues, f"{path.suffix[1:].upper()} (dimensions not inspected)"
    width, height = dimensions
    summary = f"{path.suffix[1:].upper()} {width}×{height}, aspect {width / height:.2f}:1"
    if width < 1200:
        issues.append(Issue(
            "warning", path,
            f"raster width is {width}px; full-width print may appear soft"))
    return issues, summary


def audit_asset(path):
    if not path.exists():
        return [Issue("error", path, "asset does not exist")], "missing"
    return audit_svg(path) if path.suffix.lower() == ".svg" else audit_raster(path)


def audit_markdown(path):
    issues, assets = [], []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for index, line in enumerate(lines, 1):
        match = IMAGE_RE.search(line)
        if not match:
            continue
        caption, target = match.groups()
        assets.append((path.parent / target).resolve())
        if not caption.strip():
            issues.append(Issue(
                "note", path,
                "uncaptioned image; acceptable for slide-only reuse, otherwise add a caption",
                index))
            continue
        words = len(caption.split())
        sentences = len(SENTENCE_RE.findall(caption))
        if words < 8:
            issues.append(Issue(
                "warning", path,
                f"caption has only {words} words; identify setup, encoding, and comparison",
                index))
        if words > 95:
            issues.append(Issue(
                "warning", path,
                f"caption has {words} words; move argument or caveats into body prose",
                index))
        if words > 45 and sentences < 2:
            issues.append(Issue(
                "warning", path,
                "long caption has fewer than two sentence boundaries", index))
        following = next((candidate.strip() for candidate in lines[index:]
                          if candidate.strip()), "")
        if not LABEL_RE.match(following):
            issues.append(Issue(
                "note", path,
                "image has no attached figure label; number it if later prose refers to it",
                index))
    return issues, assets


def expand_inputs(inputs):
    markdown, assets = [], []
    for raw in inputs:
        path = Path(raw)
        path = path if path.is_absolute() else ROOT / path
        if path.is_dir():
            markdown.extend(sorted(path.rglob("*.md")))
        elif path.suffix.lower() == ".md":
            markdown.append(path)
        else:
            assets.append(path)
    return list(dict.fromkeys(markdown)), list(dict.fromkeys(assets))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="chapter directories, Markdown, or images")
    parser.add_argument("--strict", action="store_true",
                        help="return failure for warnings as well as errors")
    args = parser.parse_args()
    markdown, assets = expand_inputs(args.paths)
    issues = []
    for source in markdown:
        source_issues, referenced = audit_markdown(source)
        issues.extend(source_issues)
        assets.extend(referenced)
    assets = list(dict.fromkeys(assets))

    print(f"visual audit: {len(markdown)} Markdown file(s), {len(assets)} asset(s)")
    for asset in assets:
        asset_issues, summary = audit_asset(asset)
        print(f"  {rel(asset)}: {summary}")
        issues.extend(asset_issues)

    order = {"error": 0, "warning": 1, "note": 2}
    issues.sort(key=lambda item: (order[item.severity], rel(item.path), item.line,
                                  item.message))
    for issue in issues:
        print(f"{rel(issue.path)}:{issue.line}: {issue.severity}: {issue.message}")
    counts = {name: sum(issue.severity == name for issue in issues)
              for name in order}
    print(f"visual audit findings: {counts['error']} error(s), "
          f"{counts['warning']} warning(s), {counts['note']} note(s)",
          file=sys.stderr)
    return 1 if counts["error"] or (args.strict and counts["warning"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
