#!/usr/bin/env python3
"""Flag reveal.js slides whose content overflows the 720px slide box, and
content hidden inside a slide (scrolled output boxes, wrapped code lines).

The source-length heuristic in tools/audit_outputs.py is unreliable: it
reported 0 overflowing slides while decks visibly overflowed. This tool
loads the *rendered* deck in headless chromium, walks every leaf slide,
and measures the real laid-out content height.

Why the measurement is correct (scale-independent)
---------------------------------------------------
Reveal lays every slide ``<section>`` out in unscaled author coordinates
(the deck's ``width`` × ``height``, e.g. 1280×720) and then applies a
single uniform ``transform: scale(...)`` to the whole ``.slides``
container to fit the viewport. An individual section's own box therefore
always lives in 720-px space, independent of viewport/scale. When content
is taller than the slide, the section's ``scrollHeight`` grows past its
(clamped) ``offsetHeight`` of 720. So ``scrollHeight > height`` is a
robust overflow signal.

Calibration: the deployed §2 decks (the bar) top out at exactly 720px
with 0 flagged; weight-decay's bad slides measured 731–956px.

Usage
-----
Serve ``_slides`` as the web root first so ``../../libs`` and ``../img``
resolve, then point this at a deck URL (local, file://, or deployed)::

    python3 -m http.server 8027 -d _slides &
    .venv-build/bin/python tools/check_slide_overflow.py \\
        http://localhost:8027/pytorch/chapter_linear-regression/weight-decay.html

Options::

    --slack <px>   tolerance above the slide height (default 1)
    --json         emit machine-readable JSON

Exit code is 0 when no slide overflows, 1 when any does (so it can gate a
build / a CI check). Requires Playwright + chromium in the venv::

    VIRTUAL_ENV=.venv-build uv pip install playwright
    .venv-build/bin/python -m playwright install chromium
"""
import argparse
import json
import sys

from playwright.sync_api import sync_playwright

# JS that runs in the page: walk leaf slides, return each one's measured
# content height in the slide's own (unscaled) coordinate system.
_MEASURE_JS = r"""
() => {
  const H = Reveal.getConfig().height;
  // Kill transitions/fragments so each slide measures at full content
  // (a hidden fragment would otherwise under-report the height).
  Reveal.configure({transition: 'none', fragments: false});
  const leaves = [...document.querySelectorAll('.slides section')]
      .filter(s => !s.querySelector('section'));
  const slides = [];
  for (const el of leaves) {
    const idx = Reveal.getIndices(el);
    Reveal.slide(idx.h, idx.v);
    el.getBoundingClientRect();              // force layout
    slides.push({
      id: el.id || ('h' + idx.h + '.v' + (idx.v || 0)),
      h: idx.h, v: idx.v || 0,
      scrollHeight: el.scrollHeight,
      slideHeight: H,
      overflowPx: el.scrollHeight - H,
    });
  }
  return {slideHeight: H, total: slides.length, slides};
}
"""


# Content hidden inside a slide that fits: an output box that scrolls
# (``.cell-output`` has ``max-height: 44vh``) and code lines that wrap. The
# overflow measure above cannot see either. Runs after it, with every slide
# displayed so that each one has a layout.
_CLIP_JS = r"""
() => {
  const res = [];
  const slides = [...document.querySelectorAll('.reveal .slides section')]
     .filter(s => !s.classList.contains('stack'));
  slides.forEach((s, i) => {
    const h = s.querySelector('h2, h1');
    const title = h ? h.textContent.trim().slice(0, 60) : '(untitled)';
    s.querySelectorAll('.cell-output, pre, .cell, table, div.sourceCode').forEach(el => {
      const cs = getComputedStyle(el);
      if (el.scrollHeight > el.clientHeight + 2 && cs.overflowY !== 'visible')
        res.push({slide: i, title, kind: 'clipped', detail:
                  `${el.className} ${el.scrollHeight}px > ${el.clientHeight}px`});
      if (el.scrollWidth > el.clientWidth + 4 && cs.overflowX !== 'visible')
        res.push({slide: i, title, kind: 'clipped-wide', detail:
                  `${el.className} ${el.scrollWidth}px > ${el.clientWidth}px`});
    });
    s.querySelectorAll('pre code').forEach(c => {
      if (c.closest('.cell-output')) return;
      const lines = c.textContent.replace(/\n$/, '').split('\n').length;
      const lh = parseFloat(getComputedStyle(c).lineHeight) || 0;
      if (lh && c.getBoundingClientRect().height / lh > lines + 0.6)
        res.push({slide: i, title, kind: 'code-wrap', detail: `${lines} line(s)`});
    });
  });
  return res;
}
"""


def check_deck(url: str, slack: int = 1):
    """Render *url* and return (report, offenders, clipped)."""
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        page.goto(url, wait_until="load")
        # Reveal initializes asynchronously.
        page.wait_for_function(
            "() => typeof Reveal !== 'undefined' "
            "&& Reveal.isReady && Reveal.isReady()",
            timeout=15000,
        )
        report = page.evaluate(_MEASURE_JS)
        page.add_style_tag(
            content=".reveal .slides section{display:block !important;}")
        page.wait_for_timeout(300)
        clipped = page.evaluate(_CLIP_JS)
        browser.close()
    offenders = [s for s in report["slides"] if s["overflowPx"] > slack]
    return report, offenders, clipped


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("url", help="deck URL (serve _slides as web root first)")
    ap.add_argument("--slack", type=int, default=1,
                    help="tolerance in px above the slide height (default 1)")
    ap.add_argument("--json", action="store_true",
                    help="emit machine-readable JSON")
    args = ap.parse_args()

    report, offenders, clipped = check_deck(args.url, args.slack)

    if args.json:
        print(json.dumps({"url": args.url, **report, "offenders": offenders,
                          "clipped": clipped}, indent=2))
    else:
        print(f"deck: {args.url}")
        print(f"slide box: {report['slideHeight']}px · "
              f"leaf slides: {report['total']} · tolerance: {args.slack}px")
        if not offenders:
            tallest = max((s["scrollHeight"] for s in report["slides"]),
                          default=0)
            print(f"OK: no overflow "
                  f"(tallest slide {tallest}px <= {report['slideHeight']}px)")
        else:
            print(f"OVERFLOW: {len(offenders)} slide(s) exceed "
                  f"{report['slideHeight']}px:")
            for s in sorted(offenders, key=lambda s: -s["overflowPx"]):
                print(f"  #{s['id']}  {s['scrollHeight']}px  "
                      f"(+{s['overflowPx']}px over)")
        if clipped:
            print(f"CLIPPED: {len(clipped)} element(s) hide content:")
            for c in clipped:
                print(f"  slide {c['slide']} [{c['title']}] {c['kind']}: "
                      f"{c['detail']}")

    sys.exit(1 if offenders or clipped else 0)


if __name__ == "__main__":
    main()
