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

    --slack <px>          tolerance above the slide height (default 1)
    --json                emit machine-readable JSON
    --timeout <s>         watchdog for the whole deck (default 120)
    --slide-timeout <s>   watchdog for one slide (default 30)

Exit code is 0 when no slide overflows, 1 when any does (so it can gate a
build / a CI check), and 2 when the deck hangs Chromium or fails to load:
the check runs in a child process that a watchdog kills, so one hanging
deck cannot stall a loop over many decks. Requires Playwright + chromium in the venv::

    VIRTUAL_ENV=.venv-build uv pip install playwright
    .venv-build/bin/python -m playwright install chromium
"""
import argparse
import json
import multiprocessing as mp
import os
import queue
import signal
import sys
import time

from playwright.sync_api import sync_playwright

# JS that runs in the page: walk leaf slides, return each one's measured
# content height in the slide's own (unscaled) coordinate system.
_SETUP_JS = r"""
() => {
  // Kill transitions/fragments so each slide measures at full content
  // (a hidden fragment would otherwise under-report the height).
  Reveal.configure({transition: 'none', fragments: false});
  return {slideHeight: Reveal.getConfig().height,
          total: [...document.querySelectorAll('.slides section')]
                   .filter(s => !s.querySelector('section')).length};
}
"""

# Measured one slide per evaluate() call, so a slide that hangs the renderer
# can be named (the parent's watchdog reports the last slide it started).
_MEASURE_JS = r"""
(k) => {
  const H = Reveal.getConfig().height;
  const el = [...document.querySelectorAll('.slides section')]
      .filter(s => !s.querySelector('section'))[k];
  const idx = Reveal.getIndices(el);
  Reveal.slide(idx.h, idx.v);
  el.getBoundingClientRect();              // force layout
  return {
    id: el.id || ('h' + idx.h + '.v' + (idx.v || 0)),
    h: idx.h, v: idx.v || 0,
    scrollHeight: el.scrollHeight,
    slideHeight: H,
    overflowPx: el.scrollHeight - H,
  };
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


def check_deck(url: str, slack: int = 1, progress=None):
    """Render *url* and return (report, offenders, clipped).

    *progress*, if given, is a shared integer set to the index of the slide
    being measured (-1 while loading, -2 during the clip pass), so a watchdog
    can say where a hang happened.
    """
    def _at(k):
        if progress is not None:
            progress.value = k

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        _at(-1)
        page.goto(url, wait_until="load")
        # Reveal initializes asynchronously.
        page.wait_for_function(
            "() => typeof Reveal !== 'undefined' "
            "&& Reveal.isReady && Reveal.isReady()",
            timeout=15000,
        )
        report = page.evaluate(_SETUP_JS)
        report["slides"] = []
        for k in range(report["total"]):
            _at(k)
            report["slides"].append(page.evaluate(_MEASURE_JS, k))
        _at(-2)
        page.add_style_tag(
            content=".reveal .slides section{display:block !important;}")
        page.wait_for_timeout(300)
        clipped = page.evaluate(_CLIP_JS)
        browser.close()
    offenders = [s for s in report["slides"] if s["overflowPx"] > slack]
    return report, offenders, clipped


def _child(url, slack, progress, out):
    # Own process group, so the watchdog can kill the Playwright driver and
    # Chromium together with this process.
    os.setsid()
    try:
        out.put(("ok", check_deck(url, slack, progress)))
    except Exception as e:  # noqa: BLE001 - reported to the parent
        out.put(("error", f"{type(e).__name__}: {e}"))


def check_deck_with_timeout(url: str, slack: int = 1, timeout: float = 120,
                            slide_timeout: float = 30):
    """Run check_deck in a child process under a watchdog.

    A slide that hangs Chromium's layout (e.g. `text-wrap: pretty` on text
    with math) blocks page.evaluate() forever, and Playwright's timeouts do
    not cover evaluate(). The child is killed when the whole deck exceeds
    *timeout* seconds or one slide exceeds *slide_timeout* seconds.
    Returns ("ok", result), ("error", message) or ("hang", message).
    """
    ctx = mp.get_context("spawn")
    progress = ctx.Value("i", -3)
    out = ctx.Queue()
    proc = ctx.Process(target=_child, args=(url, slack, progress, out))
    proc.start()
    t0 = time.monotonic()
    last, t_last = progress.value, t0
    while True:
        try:
            return out.get(timeout=0.5)
        except queue.Empty:
            pass
        now, cur = time.monotonic(), progress.value
        if cur != last:
            last, t_last = cur, now
        per_slide = cur >= 0 and now - t_last > slide_timeout
        if now - t0 > timeout or per_slide or not proc.is_alive():
            break
    if not proc.is_alive():
        try:
            return out.get(timeout=1)
        except queue.Empty:
            return ("error", f"checker exited with code {proc.exitcode}")
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        proc.kill()
    proc.join(5)
    where = {-3: "while starting", -1: "while loading the deck",
             -2: "in the clip pass"}.get(last, f"at leaf slide {last}")
    limit = (f"{slide_timeout:g}s on one slide" if last >= 0 and
             now - t_last > slide_timeout else f"{timeout:g}s for the deck")
    return ("hang", f"HANG {where} (no response within {limit})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("url", help="deck URL (serve _slides as web root first)")
    ap.add_argument("--slack", type=int, default=1,
                    help="tolerance in px above the slide height (default 1)")
    ap.add_argument("--json", action="store_true",
                    help="emit machine-readable JSON")
    ap.add_argument("--timeout", type=float, default=120,
                    help="seconds allowed for the whole deck (default 120)")
    ap.add_argument("--slide-timeout", type=float, default=30,
                    help="seconds allowed for one slide (default 30)")
    args = ap.parse_args()

    status, res = check_deck_with_timeout(args.url, args.slack, args.timeout,
                                          args.slide_timeout)
    if status != "ok":
        if args.json:
            print(json.dumps({"url": args.url, "status": status,
                              "detail": res}, indent=2))
        else:
            print(f"deck: {args.url}")
            print(res if status == "hang" else f"ERROR: {res}")
        sys.exit(2)
    report, offenders, clipped = res

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
