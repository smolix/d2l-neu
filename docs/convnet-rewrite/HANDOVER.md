# Convnet chapters modernization — handover

**To:** the Claude Fable agent on the GPU box.
**From:** the Claude Fable session on Alex's Mac (chisel), 2026-07-10.
**Read first:** repo `CLAUDE.md` (build system, authoring rules, gotchas),
then this directory: `spec-ch7.md`, `spec-ch8.md`, `figure-style.md`,
`spec-leading-pages.md`.

## What this is

Alex reviewed chapters 7 (convnet basics) and 8 (modern convnets) with me and
approved a modernization: ch7 gets surgical additions, ch8 gets a significant
restructure so that the two chapters together teach the 2026 state of the art
in convolutional networks — important topics, **not encyclopedic**. The specs
in this directory are the reviewed plan; treat them as the source of intent.
You have latitude on wording and code details; you do not have latitude on
scope (don't add sections beyond the spec'd TOC, don't resurrect cut
material) without asking Alex.

The review itself was grounded in a full read of both chapters and a
2020–2026 literature sweep; the key citable numbers are embedded in
`spec-ch8.md` (RSB 76.1→80.4; ConvNeXt-T 82.0 vs Swin-T 81.3; NFNet/JFT
90.4%; RepVGG INT8 75.1→40.2; ConvNeXt V2-H 88.9%). If you need more context
than the specs carry, re-research rather than guessing.

## Decisions Alex has already made (do not re-litigate)

1. Two-chapter goal: teach SOTA convnets; ch7 = fundamentals, ch8 = modern.
2. Ch8 restructure per `spec-ch8.md` §1 (8 sections, three new files, four
   retired). Ch7 per `spec-ch7.md` (no restructure).
3. **Figure style**: architecture diagrams follow the new gallery style
   (`figure-style.md`), derived from Sebastian Raschka's LLM architecture
   gallery at Alex's direction. NOT the matplotlib math house style. There is
   a **mandatory pilot gate**: two pilot figures + contact sheet → Alex
   approves → only then the full figure fleet. Alex noted Claude has
   struggled to match figure styles before; take the render-and-inspect loop
   seriously.
4. Slide decks may be rewritten freely (explicit approval).
5. MXNet remains a co-equal tab (standing decision); reduced variants with an
   honest tab note are acceptable where the framework lacks a piece (ch6
   §6.5 precedent).
6. Avoid em-dashes in book prose (Alex's standing style rule); use commas,
   parentheses, colons, or reword.
7. Work happens on the GPU box (this machine); Alex's Mac has no GPU and
   will only pull to review.

## Suggested execution order

Phases; commit at each verified milestone (see conventions below).

- **Phase 0 — foundations** (parallelizable):
  a. `d2l.bib` additions (spec-ch8 §3; grep for duplicates first).
  a'. Leading pages: add the `Resources and Further Reading` sections to the
     ch6, ch7, and ch8 `index.md` files per `spec-leading-pages.md` (curated
     content provided there; verify every URL). The ch6 one
     (`chapter_builders-guide/index.md`) is a standalone single-file edit and
     can land immediately; ch7/ch8 land together with their index rewrites.
  b. `tools/arch_diagrams.py` + the two pilot figures + contact sheet →
     **stop and get Alex's style approval** (push, tell him where to look).
     Ch7 prose work may proceed during the wait; ch8 sections that need
     figures should not finalize their figure set until the gate clears.
- **Phase 1 — chapter 7** (`spec-ch7.md`): six files, independent edits;
  suitable for parallel writer agents (one per file) with a shared brief.
  Execute + capture + verify per the spec's battery. Commit.
- **Phase 2 — chapter 8 skeleton**: config first (CHAPTER_NUMBERING,
  _quarto.yml, file creations/retirements with label preservation per
  spec-ch8 §1/§4) so the tree is never half-renamed. Then per-file content
  in dependency order: 8.3 batch-norm and 8.1 alexnet and 8.2 blocks
  (independent) → 8.4 resnet → 8.5 recipes (pilot the recipe experiment
  early; it gates 8.6's training setup) → 8.6 convnext → 8.7 efficient →
  8.8 cnn-design. Writer agents should be Fable for composition; Sonnet is
  fine for mechanical transforms and research lookups.
- **Phase 3 — integration**: full execute/capture across all four
  frameworks, freshness gate, `make html`, `make pdfs`, slides; orphan
  cleanup (legacy SVGs, retired outputs); final battery.

Notebook execution goes through the scheduler (`make run-notebooks-<fw>` or
per-file `-B` stamps); never wrap it in an outer `-j`. Capture with explicit
comma-separated `--frameworks`. Both standing memories apply: verify the
staged set before every commit, and grep repo-wide before renaming/retiring
any file.

## Commit & coordination conventions

- Commit to `main` after each verified milestone and **push** (Alex pulls on
  his Mac to review; the pilot-figure gate depends on pushes being prompt).
- Don't `git add -A` while writer agents are mid-edit; stage explicit paths
  and read the staged list.
- Another agent may be working on chapters 2–5 and the math appendix
  (`chapter_mdl-*`). Confine changes to `chapter_convolutional-*`,
  `chapter_builders-guide/index.md` (leading page only), the new
  `tools/arch_diagrams.py` + `tools/gen_arch_*` generators, `img/arch-*`,
  `d2l.bib`, the two config files (surgical edits only), and — only if a
  `#@save` genuinely changes, which the spec does not plan — `d2l/`. Pull
  before pushing; expect occasional non-ff and resolve by rebase or ff-merge,
  never force-push.
- If a spec instruction conflicts with what you find in the tree (drift since
  2026-07-10), trust the tree, note the conflict in a commit message, and
  flag it to Alex if it changes scope.

## Acceptance criteria (the definition of done)

1. Both chapters render in all four framework tabs from a green committed
   store; `audit_outputs.py --verify-fresh` clean on the box.
2. All four PDFs build; slides build for every touched file.
3. Every label listed in spec-ch8 §1 still resolves book-wide (grep for
   dangling `:numref:`).
4. No orphaned files: retired `.md`/`.qmd`/outputs manifests deleted; legacy
   `img/*.svg` with zero remaining references deleted.
5. Figures: all architecture diagrams in rewritten/new sections are
   gallery-style, pilot-approved, byte-idempotent, and visually inspected
   (contact sheet committed under the scratch area is fine, don't commit
   review PNGs to the repo).
6. Prose passes the repo authoring rules (CLAUDE.md "Content authoring" +
   the math/PDF tripwires) and `tools/lint_source.py`.
6'. All three leading pages carry their Resources sections in the exact
   format of the existing ten (spec-leading-pages.md), every link verified
   live, `:numref:` targets resolving.
7. A short completion note appended to this file (what changed, what was
   deferred, open questions), committed with the final battery results.
