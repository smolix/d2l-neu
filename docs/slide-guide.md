# D2L Slide Style & Authoring Guide

Scope: the `<!-- slides -->` decks across **Basics** (Chapters 1–8), **Advanced** (Chapters 9–17), and **Mathematics for Deep Learning** (Chapters 33–38). Pair this guide with [`docs/slides.md`](slides.md) for the technical Reveal.js / Quarto syntax and pipeline mechanics. Reference exemplars: Chapter 1 (`chapter_preliminaries/*.md`).

---

## 1. Pedagogical Standard: University-Grade Teaching Decks

A deck must teach on its own at the level of an upper-division or graduate course at a top university (Stanford, MIT, CMU, Berkeley). A lecturer can teach the section directly from the slides without reading from the textbook, and a student reviewing the deck after lecture can follow the complete logical story, mathematical derivations, and systems pitfalls.

### What to Avoid
1. **No "Claude staccato" fragments:** Never write telegraphic 3-word sentence fragments or disconnected keyword lists that leave out the *why* and *how*. Write complete, natural, information-dense clauses that connect cause, mechanism, and effect.
2. **No walls of prose or pasted book paragraphs:** Never copy-paste multi-sentence textbook paragraphs onto a slide. Synthesize prose into structured bullets, side-by-side comparisons, compact equations, or summary tables.
3. **No bureaucratic meta-slides:** Never include filler slides such as *"By the End of This Deck / Uses / Skips"* or *"Agenda / Outline"*. Start teaching immediately after the cover slide.
4. **No walls of boilerplate code:** Show only the core 2–8 lines of code that express the concept. Omit imports, plotting plumbing, or repetitive setup (`@-id` suppresses noisy output; `@!id` shows the resulting plot or output only).

---

## 2. Deck Structure & Sizing

- **Scale slide count to the section's conceptual depth:**
  - **Chapter opener (`chapter_*/index.md`):** **5–7 slides** framing the chapter's central problem, mathematical/architectural roadmap table, concrete running benchmark, and study guide.
  - **Standard section notebook:** **15–22 slides**.
  - **Long or mathematically dense section notebook (e.g., *Mathematics for Deep Learning* or multi-stage architectures):** **20–30+ slides**. Never cram two distinct concepts or a long derivation onto one crowded slide to save slide count—add slides so every slide breathes and fits the 14" MacBook viewport cleanly.
- **Narrative arc of every section deck:**
  1. **Cover slide (`::: {.cover}`):** Section kicker (`[Dive into Deep Learning · §N.M]{.kicker}`, automatically updated by `gen_slides.py`), 1-line descriptive subtitle, and bold dot-separated keyword bar.
  2. **Motivation & Core Mental Model (Slide 2):** Why does this concept exist? Anchor immediately with concrete tensor shapes, parameter counts, computational costs, or failure modes of naive approaches.
  3. **Core Concepts, Mechanics & Visual/Code Evidence (Body Slides):** Develop one concept per slide, pairing mathematical definitions or algorithms with geometric diagrams (`@fig:...`), chapter SVG figures (`![](../img/...)`), or live code/output cards (`@cell-id`).
  4. **Practical Systems / Statistical Pitfalls:** Highlight what goes wrong in real deep learning workflows (e.g., memory aliasing, silent broadcasting into `(B, B)`, data leakage from fitting preprocessing on the test set, numerical underflow/overflow, vanishing/exploding gradients).
  5. **Interactive Concept Check (Second-to-last slide):** 2–3 concrete design or calculation questions the class can work out in 60 seconds, with answers revealed in a `::: {.d2l-note .fragment .fade-up}` callout.
  6. **Summary / Key Takeaways Table (Final slide):** A structured markdown table or synthesis contrasting the section's operations, mathematical forms, complexity/scaling, and deep learning roles.

---

## 3. Slide Anatomy & Visual Vocabulary (`_d2l-slides.scss`)

Every slide uses the classes defined in `_d2l-slides.scss`:

```markdown
<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §1.1]{.kicker}

Tensors: The Core Data Structure of Deep Learning<br>**storage buffers · shapes & strides · views · broadcasting · memory semantics**
:::
:::

::: {.slide title="A Tensor Is a Storage Buffer Plus Metadata"}
[The Unifying Mental Model]{.kicker}

Every operation in this section—reshaping, slicing, broadcasting, and in-place mutation—follows from one systems distinction:

::: {.cols .vc}
::: {.col}
- **Contiguous 1D storage buffer:** A flat block of typed bytes (`float32`, `bfloat16`, `int64`) living in CPU RAM or GPU HBM.
- **Lightweight metadata header:** Small integers (`shape`, `strides`, `dtype`, `device`) describing how multi-axis indices map to byte offsets:
  $$\text{offset}(i, j) = \text{base} + i \cdot s_0 + j \cdot s_1$$
- **Zero-copy view vs. allocation:** Changing metadata (`reshape`, `T`, slice) is $O(1)$; allocating a new buffer costs $O(n)$ memory traffic.
:::

::: {.col .fig}
@fig:ndarray-rank-ladder
:::
:::
:::
```

### Mandatory Markup Rules
- **Single-line slide titles (`title="..."`):** Every content slide must have a descriptive `title="..."` attribute (≤ 58 characters) that fits on **one `H2` line** at `1280×720`, paired with a `[Category]{.kicker}` eyebrow on the first body line.
- **Two-column layout (`::: {.cols}` / `::: {.cols .vc}`):**
  - Always use `::: {.cols}` (top-aligned) or `::: {.cols .vc}` (vertically centered) with child `::: {.col}` divs.
  - Right-column width modifiers:
    - `::: {.col .fig}` (44% width) — **default** whenever the left column contains display math (`$$...$$`), code cells (`@id`), or detailed bullets.
    - `::: {.col .fig .big}` (54% width) — use **only** when the left column has short prose/bullets and no wide display equations.
    - `::: {.col .narrow}` (36% width) — for a narrow text/table column paired with a wider main column.
  - **NEVER** use Quarto's reserved `::: {.columns}` / `::: {.column}` (which lacks our flex sizing and causes vertical stacking or overflow).
- **Callout boxes (`::: {.d2l-note}`):**
  - Use `::: {.d2l-note}` (blue info/takeaway), `::: {.d2l-note .warn}` (amber pitfall/hazard), or `::: {.d2l-note .rule}` (purple mathematical rule/invariant).
  - **NEVER** use `::: {.callout}` (reserved by Quarto).
- **Fragments:**
  - Top-level `. . .` pauses work **only** at the top level of a `::: {.slide}`, **never inside a `::: {.col}`** (where Pandoc renders them as literal `"..."`). Inside columns or for concept-check answers, use `::: {.d2l-note .fragment .fade-up}`.
- **No book-only directives on slides:**
  - Never put `:numref:`, `:eqref:`, `:cite:`, or `:label:` inside `<!-- slides -->`. Write self-contained references.

---

## 4. Active Linebreak & 14" MacBook Viewport Management

All decks are rendered onto a `1280×720` Reveal.js canvas and reviewed on a 14" MacBook screen (`1512×860` viewport). Every slide must pass five strict layout invariants:

### Invariant 1 — Zero Vertical Overflow (`scrollHeight <= 720`)
- Never rely on vertical scrollbars. Keep each slide within a clear height budget:
  - **Full-width text/table slide:** 1 intro line + 5–7 table rows (or 4–5 bullets) + at most 1 `.d2l-note`.
  - **Full-width code slide:** 1 intro sentence + 1 `@cell-id` card + 1 `.d2l-note` (or 2 compact `@cell-id` cards without a note).
  - **Two-column slide:** 3–4 bullets on the left + 1 figure (`@fig:...` or `@!cell-id`) or 1 compact `@cell-id` on the right.

### Invariant 2 — Zero Horizontal Column / Equation Overlap (`scrollWidth <= clientWidth`)
- Inside a two-column `.col` (~520–620 px wide), display math (`$$...$$`) renders at `0.84em`.
- **Never** place two display equations side-by-side with `\quad` or `\qquad` inside a `.col`.
- **Never** write a 5-term horizontal equality chain in a single `$$...$$` block inside a `.col`. Keep display equations in columns to a single compact relation (`A = B`), and move secondary definitions inline or onto a second line.
- When a left column has display math, pair it with `::: {.col .fig}` (44%), not `::: {.col .fig .big}` (54%).

### Invariant 3 — Atomic Inline Math & Multi-Step Chains
- `_d2l-slides.scss` sets `.reveal .math.inline, .reveal .math.inline .katex { white-space: nowrap; }` so atomic inline formulas (`$B = 256$`, `$\approx 100$`, `$P(\mathcal{A} \cup \mathcal{B})$`) never snap in half across a line break.
- `tools/gen_slides.py` automatically wraps parenthesized inline math and code (`($p = 0.5$)`, `(` + `<code>...</code>` + `)`) in `<span class="nowrap">...</span>` so an opening parenthesis `(` never dangles at the end of a line.
- **Authoring rule for multi-step inline chains:** Because each `$ ... $` span is non-breaking, **never wrap a long multi-clause derivation in a single `$ ... $` block** (e.g., avoid `$u = x^2 = 4 \to y = 3u = 12 \to dy/dx = 12$`). Instead, split multi-step inline chains into separate `$ ... $` tokens at arrow or clause boundaries:
  - Good: `$u = x^2 = 4$ $\to$ $y = 3u = 12$ $\to$ $dy/dx = 12$`
  - This allows the browser to wrap cleanly *between* steps while keeping every individual equation intact on one line.

### Invariant 4 — Single-Line Slide Titles (`H2`)
- `_d2l-slides.scss` applies `text-wrap: balance` to headings, but every `title="..."` should be authored concisely (≤ 55–58 characters) so that it renders on a **single line**.

### Invariant 5 — Unwrapped Code Lines Inside Two-Column Cells
- In two-column slides, `.reveal .cols .cell pre` renders at `0.52em`, fitting up to ~58 characters per line. If a referenced code cell sits inside a `.col` and has long trailing comments, prefer wider `.col` pairing (`.col.fig` at 44%) or full-width layout so code lines do not soft-wrap mid-comment.

---

## 5. Automated Verification Workflow (Lint + Render + Playwright/KaTeX Audit)

Before committing or opening a PR for any slide deck, run the source linter, render the slides (`pytorch` and `jax`), and run the headless Chromium + local KaTeX audit script to verify **0 layout/linebreak issues**:

```bash
# 1. Lint markdown sources
python3 tools/lint_source.py chapter_<name>/*.md

# 2. Force-render the modified decks for pytorch and jax
rm -f _slides/*/.deckhashes.json
python3 tools/gen_slides.py . _slides --frameworks pytorch jax --render --files chapter_<name>/*.md
```

Then run the Playwright audit (routing KaTeX CDN URLs to local KaTeX assets so offline Chromium measures authentic rendered math boxes):

```bash
NODE_PATH=/Users/smola/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules \
/Users/smola/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node -e '
const { chromium } = require("playwright");
const path = require("path");
const fs = require("fs");

const katexDir = "/Applications/Antigravity IDE.app/Contents/Resources/app/node_modules/katex/dist";
const files = process.argv.slice(1);

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1512, height: 860 } });
  await page.route("**/*katex*", async (route) => {
    const rel = route.request().url().replace(/^.*\/dist\//, "");
    const localFile = path.join(katexDir, rel);
    if (fs.existsSync(localFile)) {
      const ext = path.extname(localFile);
      const ct = ext === ".css" ? "text/css" : ext === ".js" ? "application/javascript" : "font/woff2";
      await route.fulfill({ status: 200, contentType: ct, body: fs.readFileSync(localFile) });
    } else {
      await route.abort();
    }
  });

  let totalIssues = 0;
  for (const file of files) {
    await page.goto("file://" + path.resolve(file), { waitUntil: "load" });
    await page.waitForTimeout(300);
    const total = await page.evaluate(() => Reveal.getTotalSlides());
    const issues = [];
    for (let i = 0; i < total; i++) {
      await page.evaluate((idx) => Reveal.slide(idx), i);
      await page.waitForTimeout(20);
      const res = await page.evaluate((idx) => {
        const s = Reveal.getCurrentSlide();
        const title = s.querySelector("h2, h1")?.innerText || "(cover)";
        const vOver = Math.max(0, s.scrollHeight - 720);
        const problems = [];
        const h2 = s.querySelector("h2");
        if (h2) {
          const lh = parseFloat(getComputedStyle(h2).lineHeight) || 40;
          if (Math.round(h2.getBoundingClientRect().height / lh) > 1) problems.push("h2-wraps");
        }
        s.querySelectorAll(".cols > .col").forEach((c, ci) => {
          if (c.scrollWidth > c.clientWidth + 2) problems.push(`col[${ci}]hOver=${c.scrollWidth - c.clientWidth}px`);
          const cRect = c.getBoundingClientRect();
          c.querySelectorAll(".katex-display, .katex, pre, table, svg, img, .d2l-note").forEach(ch => {
            const r = ch.getBoundingClientRect();
            if (r.right > cRect.right + 4) problems.push(`col[${ci}]ChildOver=${Math.round(r.right - cRect.right)}px`);
          });
        });
        s.querySelectorAll(".math.inline").forEach(m => {
          const rects = Array.from(m.getClientRects()).filter(r => r.width > 1 && r.height > 1);
          const lines = new Set(rects.map(r => Math.round(r.top / 6)));
          if (lines.size > 1) problems.push(`math-split(${m.innerText.replace(/\s+/g, " ").slice(0, 20)})`);
        });
        if (vOver > 0 || problems.length > 0) return { slide: idx + 1, title, vOver, problems };
        return null;
      }, i);
      if (res) issues.push(res);
    }
    totalIssues += issues.length;
    console.log(`${file} (${total} slides): ${issues.length} issues`);
    for (const iss of issues) console.log("  ", JSON.stringify(iss));
  }
  await browser.close();
  process.exit(totalIssues > 0 ? 1 : 0);
})();
' _slides/pytorch/chapter_<name>/*.html _slides/jax/chapter_<name>/*.html
```
