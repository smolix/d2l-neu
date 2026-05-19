#!/usr/bin/env python3
"""Audit authored slide blocks for teachability and overflow risks."""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path


SLIDE_OPEN_RE = re.compile(r"^(:{3,4})\s*\{\.slide\b([^}]*)\}")
TITLE_RE = re.compile(r'title="([^"]*)"')
CODE_OPEN_RE = re.compile(r"^```\{\.python\b.*#([A-Za-z0-9][A-Za-z0-9_-]*)")
PLACEHOLDER_RE = re.compile(r"^@(!?)([A-Za-z0-9][A-Za-z0-9_-]*)(?:@[A-Za-z]+)?\s*$")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")
INLINE_MATH_RE = re.compile(r"(?<!\\)\$[^$\n]{2,160}(?<!\\)\$")


def markdown_files(root: Path) -> list[Path]:
    return sorted(root.glob("chapter_*/*.md"))


def collect_code_cells(paths: list[Path]) -> dict[str, dict[str, int]]:
    cells: dict[str, dict[str, int]] = defaultdict(lambda: {"max_lines": 0, "max_width": 0})
    for path in paths:
        lines = path.read_text(encoding="utf-8").splitlines()
        i = 0
        while i < len(lines):
            match = CODE_OPEN_RE.match(lines[i])
            if not match:
                i += 1
                continue
            cell_id = match.group(1)
            body: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                body.append(lines[i])
                i += 1
            nonblank = [line for line in body if line.strip()]
            cells[cell_id]["max_lines"] = max(cells[cell_id]["max_lines"], len(nonblank))
            cells[cell_id]["max_width"] = max(
                cells[cell_id]["max_width"],
                max((len(line) for line in body), default=0),
            )
            i += 1
    return cells


def parse_slides(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    slides: list[dict] = []
    try:
        i = next(i for i, line in enumerate(lines) if "<!-- slides -->" in line) + 1
    except StopIteration:
        return slides

    while i < len(lines):
        match = SLIDE_OPEN_RE.match(lines[i])
        if not match:
            i += 1
            continue
        fence, attrs = match.groups()
        title_match = TITLE_RE.search(attrs)
        title = title_match.group(1) if title_match else ""
        body: list[str] = []
        start_line = i + 1
        i += 1
        while i < len(lines) and lines[i].strip() != fence:
            body.append(lines[i])
            i += 1
        slides.append({"file": path.as_posix(), "line": start_line, "title": title, "body": body})
        i += 1
    return slides


def source_math_count(text: str) -> int:
    return (
        text.count("$$") // 2
        + text.count(r"\[")
        + len(INLINE_MATH_RE.findall(text))
    )


def enrich_slide(slide: dict, code_cells: dict[str, dict[str, int]]) -> None:
    placeholders: list[str] = []
    output_placeholders: list[str] = []
    prose: list[str] = []
    math_markers = 0
    bullets = 0
    for line in slide["body"]:
        stripped = line.strip()
        match = PLACEHOLDER_RE.match(stripped)
        if match:
            placeholders.append(match.group(2))
            if match.group(1):
                output_placeholders.append(match.group(2))
            continue
        if "$" in line or r"\[" in line or r"\(" in line:
            math_markers += line.count("$") + line.count(r"\[") + line.count(r"\(")
        if re.match(r"\s*[-*+]\s+", line):
            bullets += 1
        if stripped and not stripped.startswith((":::", "::::", "<!--")):
            prose.append(line)

    code_lines = sum(
        code_cells.get(cell_id, {}).get("max_lines", 0)
        for cell_id in placeholders
        if cell_id not in output_placeholders
    )
    max_cell_lines = max(
        (
            code_cells.get(cell_id, {}).get("max_lines", 0)
            for cell_id in placeholders
            if cell_id not in output_placeholders
        ),
        default=0,
    )
    max_cell_width = max(
        (
            code_cells.get(cell_id, {}).get("max_width", 0)
            for cell_id in placeholders
            if cell_id not in output_placeholders
        ),
        default=0,
    )

    slide.update(
        {
            "placeholders": placeholders,
            "output_placeholders": output_placeholders,
            "word_count": len(WORD_RE.findall(" ".join(prose))),
            "math_markers": math_markers,
            "bullets": bullets,
            "code_lines": code_lines,
            "max_cell_lines": max_cell_lines,
            "max_cell_width": max_cell_width,
        }
    )


def audit(root: Path) -> dict:
    paths = markdown_files(root)
    code_cells = collect_code_cells(paths)
    slides = []
    for path in paths:
        for slide in parse_slides(path):
            enrich_slide(slide, code_cells)
            slides.append(slide)

    by_file: dict[str, list[dict]] = defaultdict(list)
    for slide in slides:
        by_file[slide["file"]].append(slide)

    first_issues: list[dict] = []
    code_walls: list[dict] = []
    sparse: list[dict] = []
    output_sparse: list[dict] = []
    likely_overflow: list[dict] = []
    math_light: list[tuple[str, int, int, int]] = []

    for file, file_slides in by_file.items():
        first = file_slides[0]
        if not first["title"] and (
            first["word_count"] < 55
            or first["output_placeholders"]
            or first["code_lines"]
        ):
            first_issues.append(first)
        for slide in file_slides:
            if (
                slide["code_lines"] >= 42
                or slide["max_cell_lines"] >= 30
                or (len(slide["placeholders"]) >= 4 and slide["word_count"] < 45)
            ):
                code_walls.append(slide)
            if (
                slide["word_count"] < 18
                and slide["placeholders"]
                and not slide["title"].lower().startswith("recap")
            ):
                sparse.append(slide)
            if slide["output_placeholders"] and slide["word_count"] < 25:
                output_sparse.append(slide)
            overflow_score = (
                slide["code_lines"]
                + slide["word_count"] / 6
                + 8 * len(slide["output_placeholders"])
            )
            if overflow_score > 75:
                likely_overflow.append(slide)

        text = Path(file).read_text(encoding="utf-8")
        before, _, after = text.partition("<!-- slides -->")
        src_math = source_math_count(before)
        slide_math = source_math_count(after)
        if src_math >= 12 and slide_math <= max(2, src_math // 8):
            math_light.append((file, len(file_slides), src_math, slide_math))

    no_slide_files = [
        path.as_posix()
        for path in paths
        if "<!-- slides -->" not in path.read_text(encoding="utf-8")
    ]
    return {
        "decks": by_file,
        "slides": slides,
        "no_slide_files": no_slide_files,
        "first_issues": first_issues,
        "code_walls": code_walls,
        "sparse": sparse,
        "output_sparse": output_sparse,
        "likely_overflow": likely_overflow,
        "math_light": math_light,
    }


def slide_row(slide: dict) -> str:
    ids = ", ".join(slide["placeholders"][:5])
    return (
        f"| `{slide['file']}:{slide['line']}` | {slide['title'] or '(untitled)'} | "
        f"{slide['word_count']} | {len(slide['placeholders'])} | "
        f"{len(slide['output_placeholders'])} | {slide['code_lines']} | "
        f"{slide['max_cell_lines']} | {ids} |"
    )


def write_report(result: dict, out: Path | None) -> str:
    decks = result["decks"]
    slides = result["slides"]
    lines = [
        "# Slide Quality Audit",
        "",
        "This report audits authored `<!-- slides -->` blocks for likely teaching and layout issues.",
        "It does not treat framework-specific missing code variants as failures;",
        "those warnings require semantic review against the source code because",
        "different frameworks often teach the same point with different cells.",
        "",
        "## Summary",
        "",
        "| Metric | Count |",
        "|--------|-------|",
        f"| Markdown files with slides | {len(decks)} |",
        f"| Authored slides | {len(slides)} |",
        f"| Markdown files without slides | {len(result['no_slide_files'])} |",
        f"| Untitled or weak first slides | {len(result['first_issues'])} |",
        f"| Likely code-wall slides | {len(result['code_walls'])} |",
        f"| Sparse placeholder slides | {len(result['sparse'])} |",
        f"| Sparse output-only slides | {len(result['output_sparse'])} |",
        f"| Likely overflow slides | {len(result['likely_overflow'])} |",
        f"| Math-heavy decks with math-light slides | {len(result['math_light'])} |",
        "",
    ]

    sections = (
        ("Likely Overflow", result["likely_overflow"]),
        ("Code Walls", result["code_walls"]),
        ("Sparse Output Slides", result["output_sparse"]),
        ("Sparse Placeholder Slides", result["sparse"]),
        ("Weak First Slides", result["first_issues"]),
    )
    for title, rows in sections:
        lines += [
            f"## {title}",
            "",
            "| Slide | Title | Words | Placeholders | Outputs | Code Lines | Max Cell | IDs |",
            "|-------|-------|-------|--------------|---------|------------|----------|-----|",
        ]
        for slide in sorted(
            rows,
            key=lambda s: (
                s["code_lines"] + s["word_count"] / 6 + 8 * len(s["output_placeholders"]),
                s["file"],
                s["line"],
            ),
            reverse=True,
        )[:80]:
            lines.append(slide_row(slide))
        lines.append("")

    lines += [
        "## Math-Light Decks",
        "",
        "| File | Slides | Source Math Markers | Slide Math Markers |",
        "|------|--------|---------------------|--------------------|",
    ]
    for file, count, src_math, slide_math in sorted(
        result["math_light"], key=lambda row: (row[2] - row[3], row[0]), reverse=True
    )[:80]:
        lines.append(f"| `{file}` | {count} | {src_math} | {slide_math} |")
    lines.append("")

    text = "\n".join(lines)
    if out:
        out.write_text(text + "\n", encoding="utf-8")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    result = audit(args.root)
    text = write_report(result, args.out)
    if args.out:
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
