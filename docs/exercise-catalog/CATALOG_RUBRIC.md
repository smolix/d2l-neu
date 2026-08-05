# Problem-Catalog Rubric (d2l-neu exercise reform)

You are building the exercise catalog for a group of sections ("notebooks") of
the book in /Users/smola/Repositories/github/d2l-neu (the `.md` files are the
source; ignore `.qmd`). Your instructions name the chapter directory (or
directories) you own and suggest external sources. The goal: for EVERY section
in your chapter(s) that has a `## Exercises` heading, (1) survey how the
world's best courses and textbooks exercise that section's topic, and (2)
propose an improved problem set with full provenance.

## Procedure per section

1. Read the section file: its H1 title, its summary, what code/experiments it
   contains, and its existing `## Exercises`. (A prior style review of every
   exercise section exists in
   /private/tmp/claude-501/-Users-smola-Repositories-boson-easy-demos/e85eca00-c9ac-4621-94a9-143fd8325e73/scratchpad/exercise-review/
   — group files 01–12; consult the profile for your files instead of
   re-assessing style.)
2. Web-research (WebSearch + WebFetch) how the suggested sources — and any
   better ones you find — pose exercises/homework/quiz questions on this
   topic. Prefer primary pages (course assignment pages, textbook exercise
   lists). Paraphrase; quote at most one short phrase per item with
   attribution. Record enough to cite later: institution/author, course code
   or book title, assignment/exercise identifier, year if visible, URL.
3. Produce the catalog entry for the section (format below).

## Catalog entry format (per section, exactly this shape)

    ## <chapter_dir>/<file>.md — <Section Title>

    **Topic:** <one line>
    **Current exercises:** <n>; disposition: keep <k>, rewrite <r>, drop <d>
    — <one line justifying drops/rewrites; strong existing sets should be
    kept and said so>

    **External sources found:**
    - <Institution, course/book, assignment/ex. id, year> — <what they ask,
      1–2 lines, paraphrased> — <URL>
    (3–8 entries; note explicitly if the topic has NO good external exercise
    tradition — that is a finding, not a failure)

    **Proposed problem set** (5–8 problems, our reference format):
    1. [tag] **Title.** Sketch of the task in 2–4 sentences: what the reader
       does, what artifact they produce, how they know it worked. Subproblems
       as an indented `1.` list if needed.
       *Provenance:* original | adapted from <source> (overlap high|med|low;
       cite on adoption) | inspired by <source> (overlap low)
    1. ...

## Rules for proposed problems

- Tags: exactly one of [conceptual] (pencil-and-paper), [short-code] (small
  implementation/run against the section's own code), [extended]
  (project-scale; at most 1 per section).
- Where the section has code: at least one [conceptual] AND one [short-code].
- Every problem names a deliverable and, where checkable, a success criterion.
  No "Can you...?", no "vary X and see what happens" without a metric, no
  bare "read this paper".
- Problems must be solvable with the section's own tools + stated
  prerequisites. Do not assume datasets/frameworks the book has not
  introduced by that point.
- Keep the book's strongest existing exercises (mark disposition keep) —
  especially in chapters whose exercises are already excellent. External
  material then serves as additions or upgrades, not replacement.
- Provenance is mandatory on every proposed problem. "Adapted (overlap
  high/med)" means we will cite the source in the book when adopting.
- Titles: descriptive noun phrases, 2–6 words, no teasers.

## Output

Write ONE file: the path given in your instructions, containing every catalog
entry for your chapter(s) in book order, preceded by a 10-line chapter
overview (best sources, coverage gaps, how the existing sets fare against the
external tradition). Return a summary of ≤20 lines: sources that proved best,
sections with no external tradition, totals (sections, proposed problems,
keep/rewrite/drop). Never edit repo files. Never fabricate provenance — if
you did not verify a URL, do not attach it.
