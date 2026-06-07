#!/usr/bin/env python3
"""Rebuild _book/search.json from the rendered HTML.

Quarto only regenerates the project-level search index during a FULL project
render; parallel subset renders (tools/render_html_parallel.sh) produce the page
HTML but leave search.json untouched. This reconstructs it from the rendered
pages, matching quarto's per-section entry shape so the on-site search keeps
working: one entry per page (lead-in) + one per H2 section, each with
{objectID, href, title, section, text, crumbs}. Run AFTER fix_crossref_numbers.py
so headings carry the logical numbering.

Usage: build_search_index.py [book_dir]   (default _book)
"""
import json
import sys
from pathlib import Path
from bs4 import BeautifulSoup

BOOK = Path(sys.argv[1] if len(sys.argv) > 1 else "_book")
SKIP_TOP = {"slides", "site_libs"}


def text_of(node):
    """Visible text of a node, newline-separated, matching quarto's index text
    (drop nav/footer/script/style; keep headings + prose + code)."""
    for bad in node.find_all(["script", "style", "nav", "noscript"]):
        bad.decompose()
    return node.get_text("\n", strip=True)


def crumbs_of(soup):
    bc = (soup.select_one(".quarto-title-breadcrumbs ol.breadcrumb")
          or soup.select_one("ol.breadcrumb"))
    if not bc:
        return []
    out = []
    for li in bc.find_all("li"):
        a = li.find("a")
        node = a if a else li
        # preserve inner markup (chapter-number / chapter-title spans)
        html = node.decode_contents().strip()
        out.append(html if html else node.get_text(" ", strip=True))
    return out


def page_entries(html_path):
    rel = html_path.relative_to(BOOK).as_posix()
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "html.parser")
    main = soup.find("main", id="quarto-document-content") or soup.find("main")
    if main is None:
        return []
    h1 = soup.find("h1", class_="title") or soup.find("h1")
    title = h1.get_text(" ", strip=True) if h1 else html_path.stem
    crumbs = crumbs_of(soup)

    # Top-level H2 sections; their content (incl. nested H3s) is one entry each.
    h2s = main.find_all("section", class_="level2")
    entries = []

    # Page-level entry: the lead-in (everything in <main> minus the H2 sections).
    lead = BeautifulSoup(str(main), "html.parser")
    for s in lead.find_all("section", class_="level2"):
        s.decompose()
    lead_text = text_of(lead)
    entries.append({"objectID": rel, "href": rel, "title": title,
                    "section": "", "text": lead_text, "crumbs": crumbs})

    for sec in h2s:
        sid = sec.get("id")
        if not sid:
            continue
        h = sec.find(["h2", "h3", "h4"])
        heading = h.get_text(" ", strip=True) if h else ""
        href = f"{rel}#{sid}"
        entries.append({"objectID": href, "href": href, "title": title,
                        "section": heading, "text": text_of(sec),
                        "crumbs": crumbs})
    return entries


def main():
    all_entries = []
    pages = 0
    for html in sorted(BOOK.rglob("*.html")):
        parts = html.relative_to(BOOK).parts
        if parts and parts[0] in SKIP_TOP:
            continue
        try:
            e = page_entries(html)
        except Exception as ex:
            print(f"  warn: {html}: {ex}", file=sys.stderr)
            e = []
        if e:
            pages += 1
            all_entries.extend(e)
    out = BOOK / "search.json"
    out.write_text(json.dumps(all_entries, ensure_ascii=False), encoding="utf-8")
    print(f"search.json: {len(all_entries)} entries across {pages} pages -> {out}")


if __name__ == "__main__":
    main()
