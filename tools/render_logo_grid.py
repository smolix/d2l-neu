#!/usr/bin/env python3
"""Render the university logo grid in index.md from tools/universities.json.

Replaces content between the markers
    <!-- @universities-begin ... -->
    <!-- @universities-end -->
in index.md (and index.qmd if present, though the latter is auto-generated).

Every university with a logo is shown. A logo links to a course page,
  <a href="<course_url>" title="..."><img src="..." alt="..." width=.. height=.. loading="lazy"></a>
when the entry has non-weak evidence, and is an unlinked <span> otherwise.
The block holds the confirmed count, a first screen of FIRST_SCREEN logos and a
<details> element with every shown university grouped by country. The
"500+ universities" headline above the block is hand-written in index.md: it
includes adoptions recorded outside universities.json.

The adoption map is shown as a compressed WebP (MAP_REL), made from the PNG
that tools/render_google_university_map.py writes (MAP_PNG) whenever the PNG
is newer, if Pillow or ImageMagick is available.

Usage:
    python tools/render_logo_grid.py
"""
from __future__ import annotations

import hashlib
import html
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
JSON_PATH = REPO / "tools" / "universities.json"
LOGO_REL_DIR = "static/landing/universities"

BEGIN = "<!-- @universities-begin"
END = "<!-- @universities-end -->"
INDENT = " " * 6
MAP_PNG = "static/landing/universities-map-google.png"
MAP_REL = "static/landing/universities-map.webp"


def versioned(rel: str) -> str:
    """``rel`` plus ``?v=<hash of the file>``.

    The site's CDN caches images for hours under their URL, so a logo that is
    replaced under the same file name keeps being served stale. A content hash
    in the query string gives every new version a new URL; unchanged files keep
    theirs, so their cached copies stay valid.
    """
    digest = hashlib.sha256((REPO / rel).read_bytes()).hexdigest()[:10]
    return f"{rel}?v={digest}"


def is_weak_evidence(url: str) -> bool:
    """Skip URLs that aren't independent course evidence."""
    if not url:
        return True
    u = url.lower().rstrip("/")
    # The d2l.ai homepage adopters list is self-attested, not independent.
    if u in {"https://d2l.ai", "https://www.d2l.ai", "http://d2l.ai", "http://www.d2l.ai"}:
        return True
    if u.startswith("https://classic.d2l.ai") or u.startswith("http://classic.d2l.ai"):
        return True
    if u.startswith("https://d2l.ai/#") or u.startswith("https://www.d2l.ai/#"):
        return True
    # Studocu/coursehero/scribd are paywalled aggregators, not the institution's own page.
    if "studocu.com" in u or "coursehero.com" in u or "scribd.com" in u:
        return True
    # Temporary redirect tokens from a search tool; they expire and lead nowhere.
    if "vertexaisearch.cloud.google.com" in u:
        return True
    return False


def pick_link(evidence: list) -> dict | None:
    """Choose the strongest evidence URL — first non-weak entry by year-desc order."""
    for ev in evidence:
        if not is_weak_evidence(ev.get("url", "")):
            return ev
    return None


FIRST_SCREEN = 96  # logos shown before the <details> with the full list


def png_size(path: Path) -> tuple[int, int] | None:
    """(width, height) from a PNG header, or None for other formats."""
    head = path.read_bytes()[:24]
    if head[:8] == b"\x89PNG\r\n\x1a\n" and head[12:16] == b"IHDR":
        return int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big")
    return None


def evidence_count(entry: dict) -> int:
    return len({ev["url"] for ev in entry.get("evidence") or []
                if not is_weak_evidence(ev.get("url", ""))})


def logo_html(e: dict) -> str:
    """One logo: linked to a course page when the entry has non-weak evidence,
    otherwise unlinked (and without hover styling)."""
    rel = f"{LOGO_REL_DIR}/{e['logo']}"
    src = versioned(rel)
    alt = html.escape(e["name"])
    # Logos that are white-on-transparent disappear against the white
    # landing page; universities.json carries an `invert` flag for those.
    img_class = ' class="d2l-uni-logo-invert"' if e.get("invert") else ""
    size = png_size(REPO / rel)
    if size:
        # Scale the intrinsic size into the 96px cell so width/height keep the
        # aspect ratio (the browser reserves space before the image loads).
        w, h = size
        k = 96 / max(w, h)
        dims = f' width="{max(1, round(w * k))}" height="{max(1, round(h * k))}"'
    else:
        dims = ' width="96" height="96"'
    img = f'<img src="{src}"{img_class} alt="{alt}"{dims} loading="lazy">'
    ev = pick_link(e.get("evidence") or [])
    if not ev:
        return f'<span class="d2l-uni-cell" title="{alt}">{img}</span>'
    title_parts = [e["name"]] + [p for p in (ev.get("course"), ev.get("instructor")) if p]
    title = html.escape(" — ".join(title_parts))
    href = html.escape(ev["url"], quote=True)
    return f'<a href="{href}" title="{title}" target="_blank" rel="noopener">{img}</a>'


def shown_entries(entries: list) -> list:
    """Universities shown on the page: every entry with a logo file.

    Entries without a public course page (e.g. "evidence_note": "link lost")
    are real adopters too; they are shown unlinked.
    """
    return [e for e in entries if e.get("logo") and (REPO / LOGO_REL_DIR / e["logo"]).exists()]


def linked_entries(entries: list) -> list:
    """Shown entries whose logo links to a public, non-weak course page."""
    return [e for e in shown_entries(entries) if pick_link(e.get("evidence") or [])]


def render_grid(entries: list) -> str:
    shown = shown_entries(entries)
    linked = linked_entries(entries)
    countries = sorted({e["country"] for e in shown})
    # First screen: the institutions with the most independent course pages,
    # a rough proxy for sustained use; ties alphabetically.
    first = sorted(shown, key=lambda e: (-evidence_count(e), e["name"].lower()))[:FIRST_SCREEN]
    lines = [f"{INDENT}<!-- @universities-begin (auto-generated by tools/render_logo_grid.py from tools/universities.json) -->"]
    lines.append(f'{INDENT}<p class="d2l-uni-confirmed">Confirmed with public evidence: '
                 f'{len(linked)} universities in {len({e["country"] for e in linked})} countries; '
                 f'their logos link to a course page that lists the book.</p>')
    lines.append(f'{INDENT}<div class="d2l-uni-logos d2l-uni-first">')
    lines += [f"{INDENT}  {logo_html(e)}" for e in first]
    lines.append(f"{INDENT}</div>")
    lines.append(f'{INDENT}<details class="d2l-uni-all">')
    lines.append(f"{INDENT}  <summary>All {len(shown)} universities, by country</summary>")
    for c in countries:
        group = sorted((e for e in shown if e["country"] == c), key=lambda e: e["name"].lower())
        lines.append(f'{INDENT}  <h3 class="d2l-uni-country">{html.escape(c)} '
                     f'<span>{len(group)}</span></h3>')
        lines.append(f'{INDENT}  <div class="d2l-uni-logos">')
        lines += [f"{INDENT}    {logo_html(e)}" for e in group]
        lines.append(f"{INDENT}  </div>")
    lines.append(f"{INDENT}</details>")
    lines.append(f"{INDENT}{END}")
    return "\n".join(lines)


def replace_block(text: str, new_block: str) -> str:
    """Replace the block between BEGIN-line and END marker (inclusive)."""
    # Match from a line containing BEGIN to a line containing END (inclusive of both lines).
    pattern = re.compile(
        rf"^[ \t]*{re.escape('<!-- @universities-begin')}.*?{re.escape(END)}[ \t]*$",
        re.M | re.S,
    )
    if not pattern.search(text):
        raise SystemExit(
            "could not find @universities-begin/@universities-end markers in target file"
        )
    return pattern.sub(new_block.strip(), text, count=1)


def refresh_map_webp() -> None:
    """Re-encode MAP_PNG as MAP_REL (about a fifteenth of the size) when stale."""
    png, webp = REPO / MAP_PNG, REPO / MAP_REL
    if not png.exists() or (webp.exists() and webp.stat().st_mtime >= png.stat().st_mtime):
        return
    try:
        from PIL import Image
        Image.open(png).convert("RGB").save(webp, "WEBP", quality=80, method=6)
    except ImportError:
        import shutil
        import subprocess
        if not shutil.which("convert"):
            print(f"warning: {MAP_REL} is older than {MAP_PNG}; install Pillow or ImageMagick")
            return
        subprocess.run(["convert", str(png), "-strip", "-quality", "80",
                        "-define", "webp:method=6", str(webp)], check=True)
    print(f"wrote {webp}")


def main():
    refresh_map_webp()
    entries = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    block = render_grid(entries)
    rendered_count = len(shown_entries(entries))

    targets = [REPO / "index.md"]
    # index.qmd is regenerated by d2l_preprocess.py, but if it has the markers we sync it too.
    qmd = REPO / "index.qmd"
    if qmd.exists() and "@universities-begin" in qmd.read_text(encoding="utf-8"):
        targets.append(qmd)

    map_src = versioned(MAP_REL)
    for path in targets:
        text = path.read_text(encoding="utf-8")
        new_text = replace_block(text, block)
        # the adoption map below the grid is regenerated under the same name too
        new_text = re.sub(rf'src="{re.escape(MAP_REL)}(\?v=[0-9a-f]+)?"',
                          f'src="{map_src}"', new_text)
        if new_text != text:
            path.write_text(new_text, encoding="utf-8")
            print(f"updated {path}  ({rendered_count} universities)")
        else:
            print(f"unchanged {path}")


if __name__ == "__main__":
    main()
