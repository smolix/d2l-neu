# Standalone Motivation Review Rubric

These reviews assess whether each notebook works when opened directly while
remaining concise for readers who arrived from the preceding notebook. They
apply the structural and language requirements in `docs/attic/style-guide.md` and
`docs/attic/language-guide.md`.

For every chapter, the review must:

1. State the chapter's intended intellectual progression.
2. Assess every notebook, including the chapter index.
3. Identify concepts, equations, algorithms, code, or surveys introduced
   before the reader has a reason to need them.
4. Distinguish missing local context from context that is merely supplied by a
   preceding notebook or a forward reference.
5. Mark catalogue passages that name several methods without organizing them
   by a shared problem or comparison axis.
6. Preserve good openings and transitions instead of recommending generic
   repetition everywhere.
7. Give concrete revision instructions: the problem or example to introduce,
   the material to move or remove, and the smallest bridge needed for an
   in-order reader.
8. Cite source files with line references wherever possible.

Use three priority levels:

- **High:** a direct reader cannot tell what problem the notebook or major
  section solves before substantial detail begins.
- **Medium:** the main progression works, but an internal section depends on
  unstated prior context, arrives in the wrong place, or becomes a catalogue.
- **Low:** a short local bridge, definition, or reordering would remove
  avoidable friction.

Each chapter review ends with an implementation sequence ordered by dependency
and expected benefit. The review is an editorial specification, not replacement
prose: it should leave a revising agent enough information to make the changes
without repeating the diagnosis.
