# Language Models part: plan

The plan for the book's Language Models part: 14 chapters, 97 sections and 4 Further Reading notebooks, each section 2,000 to 3,500 words. The chapters exist as skeletons (`chapter_lm-*/`, book chapters 18 to 31): each file holds its title, its label and its plan entry verbatim as a hidden author brief, to be replaced by the written section. Nothing in this folder is rendered.

| File | Contents |
|---|---|
| [`language_models_part_plan.md`](language_models_part_plan.md) | **The plan.** Front matter (how to use the plan, goal, size, teaching approach, course paths, assignments, threads and contract with the rest of the book, what the part does not re-teach, compute tiers, running examples, conventions, state of the art, chapter overview), then one entry per section with its teaching arc, derivation, notebook, figures, exercises, writer notes and verified sources. |
| [`blueprints/`](blueprints/README.md) | Per-chapter blueprints with further implementation detail (code budgets, fixtures, pitfalls). They use their own section numbering; [`blueprints/README.md`](blueprints/README.md) maps it to the plan, and the plan wins where they disagree. Start with [`00_conventions.md`](blueprints/00_conventions.md) and [`01_interfaces.md`](blueprints/01_interfaces.md). |

Earlier drafts of the plan, its review and the decision log are in the git history (last present in commit `42ddd3bb`).

Status (2026-11-12): chapter 1 (book chapter 18, `chapter_lm-pretrained/`) is in stage 2. Its specification and the author's rulings of that date override the plan and blueprint ch01 where they differ; the plan's chapter-1 entry lists them under "Stage-2 decisions", and [`01_interfaces.md`](blueprints/01_interfaces.md) §6 records the new exports and corrections.
