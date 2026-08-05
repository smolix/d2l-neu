# Exercise Review Rubric (d2l-neu)

You are reviewing the `## Exercises` sections of book chapters in
`/Users/smola/Repositories/github/d2l-neu`. The `.md` files are the source of
truth (never look at `.qmd`). The exercises across the book are known to be
inconsistent in style; your job is to produce a precise, evidence-backed
inventory for ONE group of chapters so the results can be aggregated.

## Procedure

1. Enumerate your files: `grep -rln "^## Exercises" <chapter_dir> --include="*.md"`.
2. Read EVERY file's full `## Exercises` section (from the heading to end of
   file, typically followed by `:begin_tab:`-wrapped Discussions links). Do not
   sample. Use Read with offset/limit; find the heading line first with
   `grep -n "^## Exercises" <file>`.
3. For each file, record the profile below. Then write your group report.

## Known style variants (anchor examples — record which each file uses)

- **Bare list** (legacy d2l style): `1. Prove that ...` — repeated `1.`
  markdown auto-numbering, no names, no tags.
- **Named + tagged** (new style, e.g. chapter_reinforcement-learning):
  `1. [short-code] *Cost of a sweep.* Give the cost ...` — bracketed type tag,
  italic name ending in a period.
- **Named only**: `1. *Monte Carlo dropout.* At test time ...` — some files
  have names on only a few exercises.
- Possibly others: **bold** names, `**Name:**`, difficulty markers, etc.
  Record whatever you find verbatim.

## Per-file profile (use EXACTLY this YAML-ish shape so it can be aggregated)

```
file: chapter_x/foo.md
heading_line: <line number of "## Exercises">
n_exercises: <count of top-level items>
numbering: repeated-1 | sequential | mixed | other(<describe>)
names: all | some(<k>/<n>) | none
name_style: italic-period | bold | mixed(<describe>) | n/a
tags: all | some(<k>/<n>) | none
tag_vocab: [<every distinct bracketed tag string found>] | n/a
difficulty_markers: <verbatim, or none>
citations: <count of :cite:/:citet: inside exercises> (styles used)
crossrefs: <count of :numref:/:eqref: inside exercises>
subproblems: none | nested-list(<which exercise numbers>) | inline-letters(<which>) | inline-numbers(<which>) | mixed(<describe>)
discussions: tabbed(<n tabs>) | single-link | missing | other
defects: 
  - L<line>: <exact description of the formatting defect>
clarity:
  - ex <k>: <what makes it underspecified/ambiguous/poorly explained>
notable: <anything else — e.g. unusually long/short section, tone outliers, exercises that assume tools/data the section never introduced>
```

## What counts as a formatting DEFECT (report with line numbers)

- Broken or un-rendered markup: stray `*`/`_`, unclosed math `$`, `\(`/`\)`
  instead of `$`, literal `:eqref:`/`:numref:` with malformed keys, HTML tags,
  broken links, double punctuation, tab-block syntax errors.
- Inline subproblem lettering `a) ... b) ... c)` crammed into one paragraph
  (vs. clean nested list items).
- Indentation that breaks list nesting (sub-items not rendering as sub-lists:
  needs 4-space indent under the parent item in this pipeline; 1–3 spaces or
  inconsistent indents are defects).
- Numbering mixing literal `1. 2. 3.` with repeated `1.` in the same list.
- Missing blank line before the list or after the heading, if it would change
  rendering.
- Sentence fragments where a list item ends mid-thought, duplicated words,
  obvious typos.
- Discussions block anomalies (missing frameworks, dead pattern, placed before
  the last exercise, etc.).

## What counts as a CLARITY problem (report with exercise number)

Judge by: would a prepared reader know (a) exactly what to do, (b) what
artifact/answer to produce, and (c) how to tell they succeeded?

- Underspecified tasks: "experiment with different values and see what
  happens" with no metric, no range, no comparison.
- References to nonexistent context: "the model above", "our previous
  experiment" when the section contains no such thing; references to a
  framework/dataset/tool the section never used.
- Ambiguous asks: two readings of what to prove/implement.
- Missing success criteria for open-ended design exercises.
- Exercises that are really just reading prompts ("think about X").
- Tone violations per house style (docs/style-guide.md): theatrical phrasing,
  "Can you...?" filler questions, promotional adjectives. Flag only clear cases.

Do NOT flag as unclear: legitimately open research-flavored exercises that
state a concrete deliverable, or standard "prove/derive/implement" tasks.

## Output

1. Write the full report (all per-file profiles + a group-level summary of
   patterns) to the file named in your instructions, under
   `/private/tmp/claude-501/-Users-smola-Repositories-boson-easy-demos/e85eca00-c9ac-4621-94a9-143fd8325e73/scratchpad/exercise-review/`.
2. Your group-level summary (max ~30 lines) must cover: dominant style per
   chapter, total exercises, files with tags/names (counts), tag vocabulary,
   worst formatting defects, worst clarity offenders.
3. Return ONLY the group-level summary as your final message (the full report
   lives in the file).

Be precise. Line numbers must be real (verify with grep/Read). Counts must be
exact. Never edit any repo file — this is a read-only review.
