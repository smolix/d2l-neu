# fable-fix: auditing and repairing AI-polished book sections

Instructions for finding and fixing the failure modes that AI-assisted "improvement"
passes left in d2l-neu chapters. Distilled from the full audit and repair of
`chapter_mdl-linear-algebra/` (see `linear-algebra-audit.md` for the finding taxonomy
in action and `linear-algebra-fix-plan.md` for what an executable fix spec looks
like). Style ground rules live in `writing-avoid.md`; this file adds the
book-specific failure modes, the detection recipes, the repair rules, and the build
workflow.

Use this on any `chapter_*/**.md` source file. Work on source `.md` only; `.qmd` is
generated and must never be edited.

---

## Part 1 — The audit (always run before fixing)

Audit first, fix second, and keep them as separate deliverables: a findings list with
file:line anchors lets a human diff their own reading against yours before any text
changes. Audit against nine categories. For each finding record an ID, location, the
offending text, and the category.

### A. Style violations (writing-avoid.md)

Read `writing-avoid.md` in full; it is the normative list. The patterns that actually
dominated in practice, with detection greps:

1. **Em-dash saturation.** AI prose uses the dash as its default connective, often
   several per paragraph, in two spellings.
   Detect: `grep -c '—' file.md` and prose `---` (in d2l sources `---` is also an
   em-dash; `--` joins names like Eckart--Young and is fine).
   Fix: replace with commas, parentheses, colons, or sentence breaks. Target zero in
   prose. Keep `--` name-joins. Normalize whatever survives to one form.
2. **Significance-inflation tics.** Recurring self-praise vocabulary that tells the
   reader something is important instead of showing it. The big five found in
   practice: `worth` ("worth noting/pausing/stating/a second look"), `clean(ly|est)`
   as pure approval, `honest(y)` as a performed trait, `beautiful/elegant` praising
   the book's own argument, plus a per-file pet word used 3+ times ("workhorse",
   "story", "thread"). Also one-off inflators: "remarkably", "striking",
   "one of the most X in all of Y", "the most important point to internalize",
   "will pay for itself".
   Detect: `grep -inE 'worth |clean|honest|beautiful|elegant|remarkabl|striking'`.
   Fix: delete the framing, keep the content. Never synonym-swap.
3. **Negative parallelism.** "not only X but also Y", "not just X; it is Y",
   "not merely X, it is Y", "X is not a curiosity, it is Z".
   Detect: `grep -inE 'not only|not just|not merely|is not a'`.
   Fix: state the positive claim.
4. **Hedge/meta/throat-clearing sentences.** "It can be informative to...",
   "While we do not have time to dive into...", "hopefully the point is clear",
   "it is good to get any intuitive grasp we can", "X is beyond the scope of this
   appendix" (especially when the text then delivers exactly that scope), and
   meta-commentary about the prose itself ("the hedge is doing real work in that
   sentence").
   Fix: delete, or replace with the factual statement the hedge was avoiding.
5. **Blocked vocabulary** from writing-avoid.md §1 (delve, pivotal, crucial, robust,
   leverage, landscape, testament, ...). Rare in already-reviewed chapters but cheap
   to check: grep the full list.

### B. Used-before-introduced / referenced-but-never-defined

Terms that carry weight before (or without) a definition: named theorems used but
never stated (Weyl, Chebyshev, Courant–Fischer), concepts from later chapters used
casually (Hessian, PCA, Lipschitz, momentum, convex relaxation), jargon glossed
nowhere (KV cache, unitarily invariant). For each: does the first load-bearing use
have either an inline definition/statement or a `:numref:` to where it is defined?
Fix rule: a one-clause inline gloss plus a pointer usually suffices; a named
inequality that a *proof* depends on must be stated as a cited lemma at the point of
use.

### C. Missing citations

Named algorithms, named theorems, empirical claims, and historical attributions all
need a `:cite:`. Common gaps: the famous-enough-to-forget (PageRank, Gershgorin,
Weyl, Vaswani attention), application claims (matrix completion, contrastive
learning), and **wrong attributions** where an adjacent-but-different paper is cited
(check: does the cited work actually prove the stated claim, or just introduce the
object?). Add missing keys to `d2l.bib` first, in its house format (tab-indented,
`Name.Name.Year` keys), then cite.

### D. Duplicate introductions

The same concept introduced fresh in two or more places (in the LA chapter:
condition number twice, numerical-stability advice twice, Gavish–Donoho twice, one
identity derived three times). Detect by listing each section's introduced concepts
and diffing. Fix rule: **every topic gets exactly one home**; every other site keeps
at most one sentence plus a `:numref:` to the home. Choose the home where the concept
is actually *used*, not where it is first mentioned.

### E. Incomplete definitions

An object used far beyond the special case in which it was defined (determinant
defined only 2×2 but used as an n×n polynomial; a norm used that was defined only in
a proof parenthetical). Also placeholder hypotheses ("a mild connectivity
condition"). Fix rule: give the real definition at the natural point, state what is
taken on faith explicitly (see the granted-facts ledger below), and name hypotheses.

### F. Dense concept-dump paragraphs

One paragraph carrying 5+ new named concepts, none defined, most never used again.
Typically appear as "applications" or "connections" paragraphs near section ends.
Fix rule: each dump is either (a) promoted to a titled subsection with definitions
and a code/exercise payoff, (b) demoted to one or two cited sentences, or (c) moved
to an exercise. Never left as-is.

### G. Uneven pacing

Two registers in one file: slow legacy prose ("a vector is a list of numbers", a
code cell that computes nothing) next to graduate-level material with no ramp.
Detect by skimming for the seams: padding sentences, teach-nothing code cells,
disclaimers of depth followed by depth. Fix rule: compress the slow half up to the
target register (this is a math appendix; readers can take it); do not water down
the dense half. Delete code cells that neither compute nor demonstrate anything
(CLAUDE.md: "code teaches"). Cutting beats splitting: only split a file if the user
asks.

### H. Flowery metaphors

Two failure shapes: metaphors that make no sense ("wears its eigenvalues on its
diagonal", "the same circle closes that the chapter opened", "made flesh") and
metaphor *families* worn out by repetition (finance: dividend / promissory note /
on credit / pay off the promise, seven uses in one chapter). Fix rule: at most one
survivor per family per file, chosen for the one that genuinely aids the reader;
delete mixed metaphors outright.

### I. Incomplete logical reasoning

The most valuable category; read every proof skeptically:
- **∎ inflation**: an argument labeled `**Proof.**` with `$\blacksquare$` that
  outsources its core step to an unproved, uncited fact, or argues by picture.
- **Silent hypotheses**: a derivation that assumes diagonalizability / FTA / real
  eigenvalues without saying so.
- **Heuristics presented as derivations**: a plausibility argument ending in an
  exact "= d".
- **Asserted "so"**: a nontrivial claim carried by "so"/"hence" with no argument.
- **Outright false color**: vivid claims added for flavor that are wrong at the
  boundary (check the numerics of any colorful dynamical claim).
Fix rules: adopt the proof-labeling policy (∎ only for complete arguments; otherwise
`**Why (sketch).**` or plain prose, or repair to a real proof); maintain a
**granted-facts ledger**: the facts a chapter takes on faith are stated once,
explicitly, at first use ("the two facts we take on faith in this chapter are...")
and cited; verify colorful numeric claims by computing them.

### Framework monologue (d2l-specific)

Prose that narrates the multi-framework build ("the worked-verification cells below
branch per framework", "in each framework the random draw...", "every framework
exposes/provides/will manufacture"). A reader of a rendered notebook sees exactly
one framework tab, so this text is meaningless to them.
Detect: `grep -inE 'framework|frameworks' file.md` and judge each hit.
Fix: delete, or reword framework-neutrally ("the library's `qr` routine"). Text
inside `:begin_tab:` blocks may name its own framework but must read self-contained
(not "the MXNet tab therefore...").

---

## Part 2 — The fix pass

### Process

1. Write the audit (Part 1) to `<chapter>-audit.md`. Let the user review it if they
   asked to; otherwise proceed.
2. Write a per-file fix plan in the style of `linear-algebra-fix-plan.md`: a global
   rules block, then ordered, anchored edit items per file, exact new
   `:label:`/`:eqlabel:` names declared up front (so parallel agents can
   cross-reference each other's not-yet-written labels), new bib keys listed, and an
   explicit out-of-scope list.
3. Add all new `d2l.bib` entries yourself before dispatching edits (avoids agents
   colliding on the shared file).
4. Dispatch one editing agent per source file, in parallel; files are disjoint, the
   plan is the contract. Each agent's prompt: read the plan's global rules + its
   file's section + `writing-avoid.md` + the audit, edit only its file, then run the
   self-checks below and report deviations.
5. Verify, rebuild, and report (Part 3).

### Editing invariants (every agent, every file)

- Source `.md` only; never `.qmd`; never figure SVGs or generators (figure changes go
  through the `mdl-figure` skill separately).
- Existing code-cell IDs (`#...` on the fence line) never change. New cells get IDs
  in the file's convention (`#mdl-<file-stem>-<slug>`); verify afterwards with
  `python3 tools/add_cell_ids.py --check` (must report 0 to assign).
- Respect each file's tab conventions for new cells: per-framework `#@tab` blocks in
  files whose cells branch, untagged plain-numpy cells only in files where every
  tab's imports provide `np`.
- Existing `:label:`/`:eqlabel:` names never change (other chapters reference them).
  If content moves, the label moves with it and must stay `:numref:`-able (labels
  need a heading or equation to anchor to).
- Slides (`<!-- slides -->` block at end of file) must stay consistent: every
  `@<id>`/`@!<id>`/`@-<id>` must resolve to a cell that still exists; deleted cells
  need their slide references removed or replaced; new headline content gets a slide
  only when the plan says so; style rules apply to slide text too.
- PDF tripwires (CLAUDE.md, docs/build-system.md §6.6): `$$` alone on its own opening
  and closing lines with `:eqlabel:` on the next line; never `]` in an image caption;
  never a digit immediately after a closing `$`; parenthesized small matrices via
  `\left(\begin{smallmatrix}...\end{smallmatrix}\right)`.
- Math correctness outranks style: never weaken a complete proof while restyling it;
  every new proof must be complete at the rigor level the plan specifies.
- Deleting a teach-nothing code cell is encouraged; deleting a teaching cell is not.
  A changed or added cell invalidates its committed outputs (see Part 3).

### Self-check greps before an agent reports done

```
grep -c '—' file.md                                    # 0
grep -nE '[a-z)`$] --- [a-z`$*(]' file.md              # prose --- : none
grep -inE 'worth |not only|not just|not merely' file.md # judged, usually 0
grep -inE 'branch per framework|every framework' file.md# 0
python3 tools/lint_source.py file.md                    # exit 0
# every slide @-ref resolves:
grep -oE '^@!?-?[a-z0-9-]+' file.md | sed 's/^@!*-*//' | sort -u   # vs
grep -oE '^```\{\.python .input #[a-z0-9-]+' file.md               # fence ids
```

---

## Part 3 — Verify and rebuild

After all agents finish (orchestrator's job):

1. **Cross-file gates**: every newly declared label defined exactly once across the
   chapter; every cross-file `:numref:`/`:eqref:` added by one agent resolves to a
   label created by another; `lint_source.py` clean on all files;
   `add_cell_ids.py --check` reports zero.
2. **Regenerate + execute** the edited notebooks for the framework(s) in scope:
   `gmake -B _notebooks/<fw>/<chapter>/<file>.executed` per file. Zero errors
   required. (CPU notebooks run on this laptop; GPU notebooks need the GPU box.)
3. **Capture, scoped** — this is the step that destroys data when done wrong:
   `python3 tools/capture_outputs.py --frameworks <fw> <chapter>/<file>.md ...`
   Always pass `--frameworks` when only some frameworks were re-executed; an
   unscoped capture blesses ALL frameworks from `_notebooks/` and wipes the
   committed outputs of frameworks you didn't run.
4. **Render** what the user asked for: `gmake pdf-<fw>` for a per-framework PDF
   (the PDF path does not run the freshness gate); `gmake html` runs
   `--verify-fresh`, which on a CPU host FAILS on stale CPU notebooks of
   not-re-executed frameworks, so either re-execute those frameworks first or defer
   HTML to the GPU box.
5. **Spot-check the rendered artifact** at the edited chapter: new subsections
   present, citations resolved (no `?` marks), no LaTeX errors in the log, equations
   with new `:eqlabel:`s numbered.
6. **Report honestly**: which frameworks' stores are now stale and need GPU-box
   re-execution; which findings were deliberately left (with IDs); build log paths.

## Scope discipline

Fix what the audit found; do not rewrite content that is working. When a structural
option (splitting files, renumbering sections, adding figures) trades reader benefit
against build/renumbering/store churn, present the trade and default to the cheaper
edit (cut, don't split) unless the user chooses otherwise. Keep the audit, the plan,
and the diff as three separate reviewable artifacts.
