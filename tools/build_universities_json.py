#!/usr/bin/env python3
"""Build canonical tools/universities.json from existing static logos
and the consolidated course-evidence TSV.

Schema (per entry):
    {
      "slug": "Carnegie-Mellon-University",      # filename stem (no extension)
      "name": "Carnegie Mellon University",      # display name
      "country": "USA",                          # may be empty for legacy entries
      "logo": "Carnegie-Mellon-University.svg",  # filename in static/landing/universities/, or null if logo not yet downloaded
      "evidence": [                              # list of independent course-evidence URLs (most recent first)
        {
          "url": "https://...",
          "course": "11-785 — Introduction to Deep Learning (Spring 2025)",
          "instructor": "Bhiksha Raj and Rita Singh",
          "snippet": "...",
          "source_file": "north_america.md",
          "year": "2025"
        }
      ]
    }

Output is sorted by slug for stable diffs.
"""
import csv
import json
import re
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EVIDENCE = Path("/home/smola/d2l/data/uni_evidence")
LOGOS_DIR = REPO / "static" / "landing" / "universities"
OUT = REPO / "tools" / "universities.json"


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.strip()).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]", "", s)


def normalize_uni(s: str) -> str:
    """Normalize a university name for matching (strip course-suffix, parens, leading numbers)."""
    s = re.sub(r"\s*[—–]\s*.*$", "", s)
    s = re.sub(r"\s*\([^()]+\)\s*$", "", s)
    s = re.sub(r"^\s*\d+\.\s*", "", s)
    return norm(s.strip())


def slug_for_filename(filename: str) -> str:
    """Logo filename → slug stem (drops extension)."""
    return re.sub(r"\.[^.]+$", "", filename)


def display_from_slug(slug: str) -> str:
    """Heuristic: 'Carnegie-Mellon-University' → 'Carnegie Mellon University'.
    Hyphens in actual names (e.g. 'Bar-Ilan-University') survive because we replace
    every '-' with ' '; a few legacy filenames lose precision but the alt text is
    overridden by the canonical evidence entry when available."""
    return slug.replace("-", " ").replace("  ", " ").strip()


def detect_year(text: str) -> str:
    if not text:
        return ""
    m = re.search(r"\b(20\d{2})\b", text)
    return m.group(1) if m else ""


def load_existing_logos() -> dict:
    """slug → filename for every logo currently on disk."""
    out = {}
    for p in sorted(LOGOS_DIR.glob("*")):
        if p.is_file() and p.suffix.lower() in {".svg", ".png", ".jpg", ".jpeg", ".webp"}:
            out[slug_for_filename(p.name)] = p.name
    return out


def load_listed_names() -> dict:
    """normalized → display name (best-known) for current 258 listed universities."""
    listed_path = EVIDENCE / "_currently_listed.txt"
    out = {}
    with listed_path.open() as f:
        for line in f:
            n = line.strip()
            if n:
                out[norm(n)] = n
    return out


def load_evidence_rows() -> list:
    """Read consolidated TSV rows."""
    tsv = EVIDENCE / "UNIVERSITIES.tsv"
    rows = []
    with tsv.open() as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for row in r:
            uni, country, status, course, instructor, url, snippet, source = row
            rows.append({
                "uni_raw": uni,
                "country": country,
                "status": status,
                "course": course,
                "instructor": instructor,
                "url": url,
                "snippet": snippet,
                "source_file": source,
            })
    return rows


def load_existing_overrides() -> dict:
    """Read the on-disk universities.json (if any) and return slug → fields
    we want to preserve across regeneration: currently just `invert`
    (manual annotation for logos that are white-on-transparent and need
    a CSS filter to be readable on the white landing page)."""
    if not OUT.exists():
        return {}
    try:
        prev = json.loads(OUT.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out = {}
    for e in prev:
        if e.get("invert"):
            out[e["slug"]] = {"invert": bool(e["invert"])}
    return out


def main():
    existing_logos = load_existing_logos()  # slug → filename
    existing_overrides = load_existing_overrides()  # slug → {invert}
    listed_norm_to_display = load_listed_names()
    rows = load_evidence_rows()

    # Build map: normalized_name → list of evidence rows
    by_norm: dict = {}
    by_norm_meta: dict = {}  # store best (display, country)
    for r in rows:
        n = normalize_uni(r["uni_raw"])
        if not n:
            continue
        if not r["url"]:
            continue
        # prefer listed display name; otherwise raw cleaned of leading number/parens
        listed = listed_norm_to_display.get(n)
        if listed:
            display = listed
        else:
            display = re.sub(r"^\s*\d+\.\s*", "", r["uni_raw"])
            display = re.sub(r"\s*[—–]\s*.*$", "", display)
            display = re.sub(r"\s*\([^()]+\)\s*$", "", display).strip()
        by_norm.setdefault(n, []).append(r)
        # keep first-seen meta
        by_norm_meta.setdefault(n, (display, r["country"]))

    # Build slug → entry, starting with each existing logo
    entries = []
    seen_norm = set()
    for slug, fname in sorted(existing_logos.items()):
        # Try to match this slug to a listed name and to evidence
        display = display_from_slug(slug)
        n = norm(display)
        # Listed name takes precedence for display
        if n in listed_norm_to_display:
            display = listed_norm_to_display[n]
        seen_norm.add(n)
        evidence_list = []
        if n in by_norm:
            for r in by_norm[n]:
                evidence_list.append({
                    "url": r["url"],
                    "course": r["course"],
                    "instructor": r["instructor"],
                    "snippet": r["snippet"],
                    "source_file": r["source_file"],
                    "year": detect_year(r["course"]),
                })
            # Sort: most recent year first; entries without year go last
            evidence_list.sort(key=lambda e: (e["year"] or "0000"), reverse=True)
        country = by_norm_meta.get(n, ("", ""))[1]
        entry = {
            "slug": slug,
            "name": display,
            "country": country,
            "logo": fname,
            "evidence": evidence_list,
        }
        if existing_overrides.get(slug, {}).get("invert"):
            entry["invert"] = True
        entries.append(entry)

    # Now add NEW universities from evidence that don't already have a logo
    for n, ev_list in sorted(by_norm.items()):
        if n in seen_norm:
            continue
        display, country = by_norm_meta[n]
        # Make a slug from display name
        slug = re.sub(r"[^A-Za-z0-9]+", "-", display).strip("-")
        # avoid clashes with existing slugs
        i = 2
        base = slug
        while slug.lower() in {e["slug"].lower() for e in entries}:
            slug = f"{base}-{i}"
            i += 1
        evidence_list = []
        for r in ev_list:
            evidence_list.append({
                "url": r["url"],
                "course": r["course"],
                "instructor": r["instructor"],
                "snippet": r["snippet"],
                "source_file": r["source_file"],
                "year": detect_year(r["course"]),
            })
        evidence_list.sort(key=lambda e: (e["year"] or "0000"), reverse=True)
        entries.append({
            "slug": slug,
            "name": display,
            "country": country,
            "logo": None,
            "evidence": evidence_list,
        })

    # Sort by slug for stable diffs
    entries.sort(key=lambda e: e["slug"].lower())

    # Stats
    existing_count = sum(1 for e in entries if e["logo"])
    new_count = sum(1 for e in entries if not e["logo"])
    with_evidence = sum(1 for e in entries if e["evidence"])
    print(f"Total entries:       {len(entries)}")
    print(f"  with existing logo: {existing_count}")
    print(f"  needing logo:       {new_count}")
    print(f"  with course URL:    {with_evidence}")

    OUT.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
