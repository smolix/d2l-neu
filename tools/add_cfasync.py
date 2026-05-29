#!/usr/bin/env python3
"""Cloudflare Rocket Loader workarounds for the staging R2 bucket.

Two patches, both build-side so we don't need zone-level CF dashboard
changes:

1. Add `data-cfasync="false"` to every `<script>` tag in `_book/*.html`.
   Rocket Loader rewrites every plain `<script src=...>` into a deferred
   async loader (it changes `type` to `<token>-text/javascript` and
   replays loads through its own runtime), which breaks Quarto's bundled
   load order and stops the search autocomplete from initializing. RL
   honors `data-cfasync="false"` and skips marked scripts.

2. Make `_book/site_libs/quarto-search/quarto-search.js` idempotent.
   Even after step 1 keeps RL away from our scripts, the RL runtime
   still loads on the page (Cloudflare injects its bootstrap into the
   served HTML) and, at the end of its activation pass, calls
   `simulateStateAfterDeferScriptsActivation` which fires a synthetic
   `DOMContentLoaded`. That re-fires every `DOMContentLoaded` listener,
   so the autocomplete library is instantiated a second time — a second
   `aa-DetachedSearchButton` appears in `#quarto-search`, which on
   mobile reads as "two search boxes" in the navbar. A one-line guard
   at the top of the `DOMContentLoaded` handler turns the duplicate
   firing into a no-op.
"""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path


_SCRIPT_TAG_RE = re.compile(
    r'(?P<open><script\b)(?![^>]*\bdata-cfasync\b)(?P<attrs>[^>]*)>',
    re.IGNORECASE,
)


def patch(html: str) -> tuple[str, int]:
    """Return (patched_html, n_patched)."""
    n = 0

    def repl(m: re.Match) -> str:
        nonlocal n
        n += 1
        return f'{m.group("open")} data-cfasync="false"{m.group("attrs")}>'

    out = _SCRIPT_TAG_RE.sub(repl, html)
    return out, n


# quarto-search.js's DOMContentLoaded handler signature. We splat a
# tiny idempotency guard immediately after the opening brace so a second
# (synthetic-from-RL) DOMContentLoaded firing is a no-op.
_QS_INIT_RE = re.compile(
    r'(window\.document\.addEventListener\("DOMContentLoaded",\s*function\s*\([^)]*\)\s*\{)',
)
_QS_GUARD = (
    'if (window.__d2lQuartoSearchInit) return; '
    'window.__d2lQuartoSearchInit = true; '
)


def patch_quarto_search(root: Path) -> str | None:
    """Insert the idempotency guard. Returns the post-patch content hash
    (first 12 hex chars of md5) so callers can bust the cache for HTML
    references to this file. Returns None if the file is missing or
    the regex didn't match."""
    qs = root / "site_libs" / "quarto-search" / "quarto-search.js"
    if not qs.exists():
        return None
    text = qs.read_text(encoding="utf-8")
    if "__d2lQuartoSearchInit" not in text:
        new, n = _QS_INIT_RE.subn(
            lambda m: m.group(1) + " " + _QS_GUARD, text, count=1
        )
        if n == 0:
            return None
        text = new
        qs.write_text(text, encoding="utf-8")
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


# `src="site_libs/quarto-search/quarto-search.js"` or
# `src="../site_libs/.../quarto-search.js"` — any number of `../` prefixes
# because chapter HTMLs sit at different depths. We append `?v=<hash>` so
# the cache key changes whenever the file's content changes; without
# this, R2 / the edge in front of it serves stale content under the
# bare URL even after we upload a fresh file with a new ETag (observed
# on this bucket: `cf-cache-status: DYNAMIC` + correct ETag header,
# but old body bytes).
_QS_SRC_RE = re.compile(
    r'(<script[^>]*src=")((?:\.\./)*site_libs/quarto-search/quarto-search\.js)(\?v=[^"]*)?"',
)


def stamp_quarto_search_src(root: Path, version: str) -> int:
    """Rewrite every `<script src=".../quarto-search.js">` in `_book/*.html`
    to carry `?v=<version>`. Returns number of files modified."""
    n_files = 0
    for p in root.rglob("*.html"):
        text = p.read_text(encoding="utf-8")
        new, n = _QS_SRC_RE.subn(
            lambda m: f'{m.group(1)}{m.group(2)}?v={version}"', text,
        )
        if n and new != text:
            p.write_text(new, encoding="utf-8")
            n_files += 1
    return n_files


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("_book")
    if not root.is_dir():
        print(f"add_cfasync: {root} not found", file=sys.stderr)
        return 1
    files = 0
    tags = 0
    for p in root.rglob("*.html"):
        text = p.read_text(encoding="utf-8")
        patched, n = patch(text)
        if n:
            p.write_text(patched, encoding="utf-8")
            files += 1
            tags += n
    qs_version = patch_quarto_search(root)
    qs_stamped = (
        stamp_quarto_search_src(root, qs_version) if qs_version else 0
    )
    print(
        f"add_cfasync: tagged {tags} <script> tags across {files} files; "
        f"quarto-search.js guard: v={qs_version or 'n/a'}, "
        f"stamped in {qs_stamped} HTML files"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
