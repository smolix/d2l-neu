# Visual Design Guide

This guide defines the visual language for diagrams, plots, and explanatory
figures in *Dive into Deep Learning*. It complements [the writing
guide](writing-guide.md). The goal is
not to decorate the book. It is to make structure, mechanism, and evidence
visible with less effort from the reader.

The reference set is deliberately broad:

- Distill's [GNN introduction](https://distill.pub/2021/gnn-intro/) shows how
  coordinated views and interaction can turn an abstract representation into
  something a reader can manipulate.
- Distill's [momentum article](https://distill.pub/2017/momentum/) shows how a
  restrained simulation, a single salient trajectory, and a few controls can
  carry the conceptual argument.
- Anthropic's [global-workspace
  article](https://www.anthropic.com/research/global-workspace) shows editorial
  storyboarding, repeated visual grammar, and careful integration of prompts,
  interventions, and outcomes.
- Sebastian Raschka's [Kimi K3 architecture
  notes](https://sebastianraschka.com/blog/2026/kimi-k3-architecture-notes.html),
  [open-weight model
  notes](https://sebastianraschka.com/blog/2026/notable-open-weight-models-this-week.html),
  and [LLM Architecture
  Gallery](https://sebastianraschka.com/llm-architecture-gallery/) show a useful
  vocabulary for nested network architectures, repeated blocks, zoomed
  submodules, and compact configuration callouts.

These references are sources of principles, not templates. Do not copy their
layouts, illustrations, palettes, typography, or distinctive motifs. A D2L
figure must be an original explanation of the book's own argument. It must also
be checked against primary technical sources and executable code. A polished
reference diagram can still contain an error.

## 1. The standard

A successful figure lets a reader answer three questions quickly:

1. What am I looking at?
2. Where should I look first, and in what order?
3. What fact or mechanism should I carry into the next paragraph?

Visual polish is subordinate to these questions. A beautiful diagram that
obscures a tensor path is worse than a plain diagram that exposes it. A plot
with elegant colors but no readable comparison is not finished.

Every figure should have one primary job. Typical jobs include:

- defining the parts of an object;
- tracing a computation or causal path;
- comparing two mechanisms;
- showing how a state changes;
- exposing an invariant across representations;
- reporting an experiment and its uncertainty;
- locating a method in a design space.

If the intended job cannot be written as one sentence, split the figure or
reconsider the explanation.

## 2. What the references do well

### 2.1 Distill: make relationships operational

The GNN article repeatedly presents the same object in coordinated forms. An
image, its adjacency matrix, and its graph are aligned; selecting an element in
one view changes or highlights the corresponding elements in the others. This
is stronger than placing three illustrations side by side. The figure teaches
the mapping between representations.

Its architecture diagrams also distinguish three levels cleanly:

- the object before a transformation;
- the operation applied to it;
- the transformed object afterward.

Labels sit next to the objects they name. Color is reused to preserve identity,
not to fill empty space. Explanatory notes occupy the margin rather than
interrupting the main flow.

The momentum article uses a different but related strategy. A quiet contour
field establishes the geometry; one orange trajectory carries the action; the
starting point, solution, and optimum receive direct labels. Controls sit
immediately below the view they change. The initial figure therefore acts as a
compact model of the article's claim rather than as a generic animation.

The lesson for D2L is not “make everything interactive.” It is this:

> Let a reader change a variable only when observing the resulting state change
> teaches a relationship that is cumbersome to express with static panels.

### 2.2 Anthropic: storyboard an experiment

The global-workspace figures read as short visual arguments. Prompt,
intervention, internal representation, and answer have distinct shapes and
positions. The same grammar recurs across figures, so the reader learns the
notation once. Muted backgrounds separate panels without heavy borders. Blue
and orange distinguish two semantic roles consistently, while dark neutral text
carries everything else.

Several devices are especially effective:

- a zoom rectangle connects a large system to the small region under study;
- before-and-after values are aligned and changed values receive emphasis;
- struck-out text shows replacement without requiring a paragraph of prose;
- panel headings state the function being tested, not merely the data shown;
- examples use sentence-like microcopy instead of cryptic abbreviations.

The figures are dense, but their density is organized. Repetition creates
fluency rather than clutter.

### 2.3 Raschka: expose architecture at two scales

The architecture references use a stable macro-to-micro grammar:

- the principal tensor path runs vertically through the main model;
- nested tinted regions show containment;
- a brace or multiplier represents repeated layers without drawing every copy;
- dotted leaders connect a component to a larger inset;
- the inset reveals the internal path of the selected block;
- configuration facts sit outside the flow and point inward;
- residual paths remain visible and visually distinct from callout connectors.

This is useful for modern architectures whose full detail cannot be shown at
one scale. The danger is crowding. At book width, a diagram with several insets
and dozens of labels can become an illegible thumbnail. Use the grammar
selectively: one overview plus one or two detail panels is usually better than a
poster-sized composite.

### 2.4 What not to import

Do not imitate surface features merely because they look polished. In
particular:

- do not reproduce Anthropic's exact warm background, blue-orange palette, or
  thought-bubble motif;
- do not reproduce Distill's article layouts, bespoke interactions, or graphic
  forms without an independent instructional reason;
- do not reproduce Raschka's diagrams, handwritten accents, compositions, or
  callout wording;
- do not infer technical truth from visual confidence;
- do not add interaction, gradients, shadows, or icons to signal modernity.

The transferable qualities are hierarchy, semantic consistency, alignment,
restraint, and a tight connection between visual state and explanatory purpose.

## 3. The D2L baseline

D2L already has several valuable foundations:

- illustrative figures can be generated deterministically as SVG;
- mathematical geometry is usually computed rather than sketched;
- the generated figures share line, arrow, and math-text conventions;
- prose normally introduces a figure before it appears;
- captions are attached to numbered figures and can be cross-referenced;
- diagrams survive HTML and PDF without raster blur.

Figure 16.1.1 is a good baseline. Its principal flow is evident: latent draw,
generator, generated batch, discriminator, and score. The real and generated
lanes remain distinct. The dashed return path makes the generator update
visible, and the caption explains both networks' roles. Whitespace separates the
stages, and direct labels reduce the need for a legend.

The current book also exhibits recurring problems that a visual pass should
address:

- very wide labeled SVGs shrink to unreadable thumbnails on phones;
- conceptual diagrams and notebook-generated plots sometimes look as though
  they belong to different books;
- the default Matplotlib palette is used as a style rather than as a semantic
  system;
- some diagrams place too much prose inside the canvas;
- architecture figures can become collections of similar boxes without a
  dominant path;
- some plots depend on legends where direct labels would be faster to read;
- line widths, marker sizes, and embedded labels are not always judged at final
  page size;
- a figure may show all recorded quantities rather than the comparison needed
  for the argument;
- code-cell outputs are not always treated as authored figures with titles,
  captions, and a stated conclusion;
- desktop inspection can conceal failures in mobile HTML or PDF.

Figure 16.1.1 itself reveals the most important responsive risk. At desktop
width its labels are clear. At a 390-pixel viewport, the whole landscape canvas
is scaled to roughly a hundred pixels high and its embedded labels cease to be
readable. A redesign should preserve the same conceptual organization while
stacking the two inputs and the update path more compactly, or provide a narrow
variant where the publication system permits one.

The other GAN diagrams are clean vectors and useful starting points, but they
also illustrate broader opportunities. The objective template asks the reader
to parse a perspective plane and a long “off the plane” annotation at once. The
three-exits figure carries several full sentences inside and below boxes. These
can be improved by separating topology from consequences: let the canvas show
the alternatives, and let the caption or adjacent prose explain their costs.
Experimental scatter plots are technically legible but should receive the same
editorial treatment as conceptual figures: a visible comparison, a direct
reference-line label, and a caption that states what agreement and deviations
mean.

## 4. Begin with an explanatory specification

Before drawing, write a short figure specification with these fields:

| Field | Question |
|---|---|
| Learning objective | What should the reader understand after ten seconds? |
| Reader action | Trace, compare, locate, manipulate, or estimate what? |
| Primary path | In what order should the eye move? |
| Invariant | What identity, quantity, or role must remain stable? |
| Change | What varies across panels, steps, or controls? |
| Evidence | Which visible feature supports the intended conclusion? |
| Exclusions | Which true details are deliberately omitted? |
| Outputs | Does the design work in desktop HTML, mobile HTML, slides, and PDF? |

For Figure 16.1.1, an appropriate objective is: “Trace how real and generated
samples reach the discriminator, then trace how the two parameter updates use
that computation.” The primary path is left to right; the generator update is a
secondary return path. Details of the loss function belong in the next figure or
equation, not in the architecture canvas.

## 5. Information hierarchy

Use at most four visual levels.

1. **Primary structure.** The main path, dominant comparison, or principal
   result. It uses the strongest contrast and the clearest line.
2. **Secondary structure.** Residual connections, branches, intervals,
   construction lines, or uncertainty. It is visible but quieter.
3. **Annotation.** Names, values, and brief explanations attached to objects.
4. **Context.** Grids, enclosing regions, unselected examples, or historical
   baselines. It should recede.

If every line is dark, every box is saturated, and every label is bold, there is
no hierarchy. Conversely, contextual material must remain visible in print; do
not use near-white strokes that vanish from the PDF.

### 5.1 The squint test

Shrink or blur the figure until labels cannot be read. The primary path or
comparison should still be apparent. If the result becomes an undifferentiated
mass of boxes, the hierarchy is too weak.

### 5.2 The ten-second test

Show the figure without its caption for ten seconds. A technically prepared
reader should be able to state its subject, direction, and main contrast. This
test does not require the figure to be self-explanatory; it requires the visual
organization to be honest.

## 6. Layout and composition

### 6.1 Use a conventional reading path

Prefer one of these paths:

- left to right for transformations and timelines;
- top to bottom for network stacks and hierarchical computation;
- aligned columns for before/after comparisons;
- small multiples for a controlled sweep;
- overview to inset for architecture detail.

Avoid circular arrangements unless cyclicity is the concept. Avoid diagonal
paths unless geometry makes the direction meaningful.

### 6.2 Align before embellishing

Shared baselines, equal panel sizes, consistent padding, and repeated box widths
create more polish than shadows or gradients. Objects with the same role should
share dimensions. Inputs to the same operation should meet it symmetrically
unless asymmetry conveys a real distinction.

Use an explicit spacing unit. In the helper library, one unit corresponds to the
nominal label height. Recommended multiples are:

- `0.5` for label-to-object separation;
- `1` for padding inside a module;
- `1.5` between adjacent stages;
- `2` to `3` between panels or explanatory groups.

### 6.3 Preserve whitespace

Whitespace groups related objects and separates unrelated ones. It is not unused
canvas. If a label fits only by crossing an arrow, increasing the canvas or
removing detail is preferable to shrinking the font.

### 6.4 Keep arrows out of text

Route connectors through clear channels. Stop arrows at box boundaries. Put edge
labels beside straight segments rather than over arrowheads or bends. When
crossings cannot be removed, use a bridge, a visible junction dot, or separate
panels; never leave the topology ambiguous.

### 6.5 Use insets for a change of scale

An inset is appropriate when a reader needs both the full architecture and the
inside of one block. Use one overview rectangle, one highlighted source region,
and one or two dotted leaders to the inset. The inset should be larger than the
source region and use the same colors and names.

Do not use an inset merely to fit overflow. If the detail does not need the
overview simultaneously, make it a second figure.

## 7. Typography and labels

### 7.1 Treat final size as the real size

The font size in the drawing program is irrelevant if the entire SVG is later
scaled by half. Judge labels in the built page.

At the expected desktop book width:

- ordinary labels should appear at least as large as 13 CSS pixels;
- secondary annotations should appear at least as large as 12 CSS pixels;
- panel titles should appear around 15–17 CSS pixels and use semibold weight;
- mathematical symbols should match surrounding prose in apparent size;
- no necessary label may depend on zoom.

For print, ordinary labels should be at least 8.5 pt at the figure's placed
width, preferably 9–10 pt. Slides need substantially larger text. A figure that
works in the book may need a simplified slide variant.

### 7.2 Use sentence case

Use sentence case for panel titles, labels, and controls. Reserve all caps for
short conventional abbreviations such as MLP or KL. Avoid title-style
capitalization inside the canvas.

### 7.3 Prefer names to prose

Good internal labels are short nouns or noun phrases:

- “generated batch”;
- “residual stream”;
- “optimal critic”;
- “validation loss.”

A complete explanatory sentence usually belongs in the caption or adjacent
body text. A short verb phrase can label an operation: “pool neighbors,” “swap
representation,” or “update critic.”

### 7.4 Label directly

Place a series name at the end of its line when space permits. Label a region in
the region. Label an arrow beside the arrow. Use a legend only when direct labels
would collide or when many series must be read repeatedly.

### 7.5 Keep notation synchronized

The diagram, equation, prose, and code must use the same symbols and roles. Do
not switch between “critic” and “discriminator,” or between `x_fake`, $x'$, and
$G(z)$, without explicitly defining the relation.

## 8. Color

### 8.1 Assign meaning before choosing hues

Color should answer “what role does this object play?” A useful local mapping is:

- neutral ink: structure, axes, and shared computation;
- blue: observed data, reference values, or the primary forward path;
- orange: generated data, alternatives, or changed values;
- green: successful state, estimate, or validated result;
- purple: optimization signal, intervention, or feedback path;
- red: failure, constraint violation, or warning.

This is not a universal ontology. A chapter may choose a different mapping, but
it must document and preserve it. The same hue must not mean “real data” in one
figure and “generator” in the next unless the context makes the identity
unambiguous.

### 8.2 Use color sparingly

Most of a figure should be neutral. One or two accent hues are usually enough;
four is a practical maximum for semantic categories. Tints may extend a hue for
enclosing regions, but a tint does not create a new category.

### 8.3 Never encode a distinction by hue alone

Combine color with position, shape, line pattern, marker, or direct text. A
reader should still recover the comparison in grayscale. Use, for example, a
solid line and a dashed line in addition to two colors.

### 8.4 Contrast requirements

Text and essential thin strokes should meet a contrast ratio of at least 4.5:1
against their background. Large text and thick graphical objects should meet at
least 3:1. Do not place saturated text on a saturated fill. Use dark text on
light tints and white text only on sufficiently dark fills.

The companion helper, `tools/diagram_style.py`, provides a restrained,
color-vision-conscious palette. It is a starting point, not permission to use
every color in one figure.

## 9. Shapes, lines, and containers

Use shape to encode type only when the distinction recurs:

- rounded rectangle: learned or named processing module;
- plain rectangle: tensor, data collection, table, or state;
- enclosing tinted region: subsystem or repeated block;
- circle with `+` or `×`: algebraic merge;
- pill: short value, token, or configuration parameter;
- dotted outline: detail boundary, optional subsystem, or zoomed inset;
- bracket or brace: repetition, with an explicit count.

Keep corners, borders, and arrowheads consistent within a chapter. Recommended
placed widths are approximately:

- primary path: 1.8–2.2 pt;
- secondary path: 1.2–1.6 pt;
- construction line or grid: 0.6–1.0 pt;
- container boundary: 1.0–1.4 pt.

Use solid lines for data or computation. Use dashed lines for gradients,
interventions, optional paths, or before/after correspondences. Use dotted lines
for callout leaders or boundaries. Never choose a pattern solely for decoration.

## 10. Architecture diagrams

Architecture figures need a stricter grammar than general schematics.

### 10.1 Show the main tensor path first

Draw the input, repeated block, output normalization, and output before adding
detail. Use one dominant direction. Residual paths should be visibly secondary
but topologically exact.

### 10.2 Compress repetition honestly

Do not draw 48 nearly identical blocks. Draw one representative block, add a
brace or stacked silhouette, and label it “×48.” If the first or final block
differs, show those exceptions explicitly.

### 10.3 Separate overview, block detail, and configuration

An overview answers where information flows. A block detail answers what happens
inside one repeated unit. A configuration table answers how wide or deep the
model is. Combining all three is acceptable only when the built figure remains
readable. Otherwise use adjacent figures or a figure plus table.

### 10.4 Make residual topology exact

Residual lines must start and end at the correct tensors. Addition and
multiplication nodes must be explicit. A callout leader must not resemble a
tensor edge. Route expert branches and gates so that it is clear which values
are mixed and where.

### 10.5 Attach dimensions to interfaces

Put shapes on tensors or edges, not in disconnected notes. Use a compact form
such as `$B \times T \times d$`. Put head counts, expert counts, and active
counts near the module they configure. Move long parameter lists to a table.

### 10.6 Verify the diagram independently

For a published model, check the technical report, released configuration, and
reference implementation where available. Record unresolved discrepancies in
the source script. Do not inherit a third-party diagram's claim without this
check.

## 11. Statistical and experimental plots

### 11.1 Plot the comparison, not the log

A training script may record dozens of metrics. A teaching plot should show the
few needed for the claim. Prefer small multiples with shared axes over a single
panel with many overlapping series.

### 11.2 State axes completely

Every quantitative axis needs a quantity and, when applicable, a unit. State
normalization, logarithmic scale, smoothing, and aggregation in the label or
caption. A metric abbreviation is not a unit.

### 11.3 Show uncertainty and sample size

When repeated trials exist, show individual runs, an interval, or a distribution
rather than only the mean. State the number of seeds. Do not add an error band to
a single run.

### 11.4 Use meaningful reference lines

Label chance level, an analytic prediction, zero, or a target value directly.
A dashed diagonal should say what agreement with it means. Avoid unlabeled
horizontal and diagonal guides.

### 11.5 End-label sparse series

When curves remain separated near the right edge, label them there. This reduces
eye travel to a legend and makes screenshots easier to interpret. Keep a legend
for dense crossings or more than roughly four series.

### 11.6 Preserve truthful scale

Do not truncate a bar-chart baseline without an explicit break. Use log scales
only when multiplicative comparisons matter. Keep shared scales across panels
that invite magnitude comparison. If scales differ, make that difference
visually conspicuous.

## 12. Interaction and animation

Interaction must correspond to a pedagogical verb:

- **select** an element to see its correspondences;
- **vary** a parameter to see sensitivity or a phase change;
- **step** through a computation;
- **compare** synchronized states;
- **construct** an example and observe an invariant.

Avoid interaction whose only outcome is movement, hover decoration, or hidden
text. Controls should be visible, labeled, keyboard accessible, and placed next
to the view they change. The displayed state must update immediately and
predictably.

Every interactive figure needs:

- a useful initial state;
- a one-sentence instruction;
- a reset action when state can become confusing;
- keyboard and touch support;
- a static fallback for PDF, slides, reduced-motion users, and script failure;
- a caption whose principal claim remains meaningful without interaction.

Use animation for continuity across states, not as a transition effect. Respect
`prefers-reduced-motion` and never require a reader to time a visual event.

## 13. Captions and placement

The rules in the writing style guide still apply. Visual design adds several
placement requirements.

### 13.1 Introduce the reading task

The sentence before a figure should tell the reader what to compare or trace.
Do not merely announce that the figure exists.

### 13.2 Keep captions outside the canvas

The caption should identify the setup, the encoding, and the conclusion. It
should not be burned into a raster image. Inside the figure, use only labels
needed to read the marks.

A strong caption usually follows this order:

1. what is shown;
2. what visual encodings mean;
3. what comparison matters;
4. what conclusion is directly visible;
5. what caveat is necessary to avoid misreading.

### 13.3 Use panel references consistently

For two panels, “Left” and “Right” are often clearer than `(a)` and `(b)`. For
three or more panels, use bold `(a)`, `(b)`, and `(c)` markers in the upper-left
of each panel and cite them in order. Panel markers should not replace
descriptive panel headings when those headings carry meaning.

### 13.4 Keep figures near their first use

Place a figure after the paragraph that establishes its question and before the
paragraph that develops consequences from it. Avoid referring to an unnumbered
code output from several paragraphs away.

## 14. Responsive and multi-format design

D2L figures serve four distinct contexts: desktop HTML, mobile HTML, slides, and
PDF. A single asset is acceptable only if it remains legible in all contexts.

### 14.1 Design mobile-first constraints

At a 390-pixel viewport:

- the principal labels must be readable without pinch zoom;
- a necessary tap target must be at least 44 by 44 CSS pixels;
- no explanatory text should become a line of microscopic glyphs;
- a horizontal comparison may stack vertically if alignment is preserved;
- a wide architecture may be split into overview and detail figures.

As a practical warning threshold, any labeled static figure wider than 2:1
requires an explicit mobile check. Wider figures are not forbidden, but they
must contain few labels or have an alternate narrow composition.

### 14.2 Build slide variants deliberately

Slides are not enlarged book pages. Remove secondary annotations, increase
labels, and reveal stages across slides when helpful. Preserve names, colors,
and topology so the audience recognizes the book figure.

### 14.3 Test PDF in grayscale

Print at actual size or inspect at 100 percent. Check fine construction lines,
small math, transparent fills, and page breaks. A figure and its caption should
not be separated. Verify that links and cross-references do not collide with the
canvas.

## 15. Accessibility

Every figure must have a useful text alternative. For a numbered static figure,
the caption usually provides the primary alternative, but the SVG should also
contain a short `<title>` and `<desc>` when the generation path supports them.
The description should state structure and conclusion, not list every pixel.

Additional requirements:

- never encode meaning by color alone;
- use sufficient contrast;
- avoid flashing and unnecessary motion;
- expose interactive state to assistive technology;
- keep DOM focus order aligned with visual reading order;
- make control labels explicit;
- retain a textual statement of every technical conclusion in the prose.

Text rendered as SVG paths is portable in PDF but not selectable. The figure's
caption and `<desc>` therefore matter. Raw SVG text improves accessibility but
can render inconsistently if it depends on unavailable fonts. Choose the tradeoff
deliberately and test both HTML and PDF.

## 16. Originality and technical integrity

Visual quality cannot compensate for technical inaccuracy. Each figure source
should make its derivation inspectable:

- compute geometric and statistical quantities from code;
- seed random examples;
- record important constants near their definitions;
- cite the primary technical source in the chapter;
- distinguish measured results from schematic illustration;
- note deliberate simplifications in the caption or source docstring;
- keep source code and generated asset in the same commit;
- make regeneration byte-idempotent where practical.

When adapting an idea seen elsewhere, rebuild the explanation from the book's
notation and learning objective. Change more than colors and labels. The
composition, examples, and emphasis should follow D2L's argument.

## 17. Reusable tooling

The repository contains six relevant tools:

- `tools/gen_mdl_figures.py` is the established generator and compatibility
  layer for existing mathematical figures.
- `tools/diagram_style.py` provides the higher-level visual tokens and drawing
  primitives described here: semantic colors, modules, groups, arrows, insets,
  panel labels, and deterministic accessible SVG output.
- `tools/gen_visual_design_example.py` is a compact original example showing an
  overview, a repeated block, a residual path, and a detail inset.
- `tools/gen_visual_design_gan_revisions.py` generates restrained one-to-one
  revision proposals for three authored GAN figures without changing the book
  assets under `img/`. Figure 16.1.1 is intentionally retained rather than
  redesigned.
- `tools/build_visual_design_review.py` builds a downloadable comparison bundle
  under `visual-design-examples/`, including 1800-pixel PNGs, SVG originals, and
  direct current-versus-redesign sheets for the GAN figures. It does not copy
  artwork from the external references.
- `tools/audit_visuals.py` performs mechanical checks on Markdown image
  references and SVG/raster assets. It does not replace visual or technical
  review.

Generate the example with:

```bash
.venv-pytorch/bin/python tools/gen_visual_design_example.py \
    --output /tmp/d2l-visual-design-example.svg
```

Build the local PNG comparison bundle with:

```bash
.venv-pytorch/bin/python tools/build_visual_design_review.py
```

Audit a chapter with:

```bash
python3 tools/audit_visuals.py chapter_generative-adversarial-networks
```

Audit particular assets with:

```bash
python3 tools/audit_visuals.py img/mdl-gan-architecture.svg \
    img/mdl-gan-template.svg
```

The audit reports, among other things, aspect ratio, embedded raster content,
missing SVG descriptions, small raw SVG text, low-resolution raster images, and
captions that are likely too short or too long. Matplotlib often converts text
to paths, so font-size checks can be incomplete. Always inspect the built page.

## 18. Review workflow

### 18.1 First pass: instructional design

1. Read the surrounding section without looking at the existing figure.
2. Write the figure's learning objective and reader action.
3. List the minimum objects and relations needed.
4. Choose a reading path and a visual encoding.
5. Decide what belongs in prose, caption, table, or separate panel.

### 18.2 Second pass: technical verification

1. Trace every arrow and residual edge.
2. Check tensor shapes, axes, units, sample sizes, and parameter counts.
3. Compare model claims against primary sources and released configuration.
4. Run the computation that produced plotted values.
5. Confirm that the caption does not claim more than the visual evidence.

### 18.3 Third pass: graphic refinement

1. Establish alignment and spacing.
2. Reduce the design to neutral structure plus necessary accents.
3. Make the primary path survive the squint test.
4. Direct-label marks where possible.
5. Remove prose and decorative elements that do not teach.

### 18.4 Fourth pass: publication QA

Inspect all of the following:

- desktop HTML at the normal content width;
- mobile HTML at 390 pixels;
- PDF at 100 percent and in grayscale;
- slides from the back of a room, or at a comparably small preview;
- light and dark browser modes if the page supports both;
- the static fallback of any interactive figure.

Run the visual audit and figure lint, but do not approve a figure solely because
the scripts pass.

## 19. Acceptance checklist

### Explanation

- [ ] The figure has one stated learning objective.
- [ ] The reader knows whether to trace, compare, locate, or manipulate.
- [ ] The visible evidence supports the intended conclusion.
- [ ] Omitted detail is deliberate and does not make the figure false.

### Structure

- [ ] The primary path is visually dominant.
- [ ] Related objects align and share spacing.
- [ ] Arrows do not cross labels or leave topology ambiguous.
- [ ] Insets, panels, and repeated blocks have a clear purpose.

### Language

- [ ] Labels are short, parallel, and in sentence case.
- [ ] Symbols match prose, equations, and code.
- [ ] The caption is self-contained but not an essay.
- [ ] The preceding prose states how to read the figure.

### Style

- [ ] Color has declared meaning and is not the only encoding.
- [ ] Neutral structure occupies most of the canvas.
- [ ] Type, borders, arrowheads, and corner radii are consistent.
- [ ] The design is original and does not mimic a reference composition.

### Technical quality

- [ ] Values, dimensions, topology, and labels have been independently checked.
- [ ] Generated quantities come from deterministic code where possible.
- [ ] SVG is preferred for diagrams and plots; raster is used for image data.
- [ ] The source and generated asset are committed together.

### Publication

- [ ] Labels are readable on desktop, mobile, slides, and PDF.
- [ ] The figure works in grayscale and with common color-vision deficiencies.
- [ ] The caption remains attached and the figure does not overflow.
- [ ] Interactive figures have keyboard, touch, reduced-motion, and static
      fallbacks.
- [ ] A useful text alternative is available.

## 20. A practical redesign order for the book

Do not redraw every figure at once. Begin where visual design has the greatest
instructional return.

1. Fix unreadable or ambiguous figures, especially wide labeled architecture
   diagrams on mobile.
2. Standardize high-value recurring grammars: tensor paths, residual blocks,
   optimization trajectories, distributions, and experimental comparisons.
3. Redesign chapter-opening figures that establish a mental model, using Figure
   16.1.1 as a structural baseline.
4. Replace legends with direct labels and reduce unnecessary series in teaching
   plots.
5. Split poster-like architecture composites into overview and detail views.
6. Add interaction only to a small number of relationships for which changing
   state materially improves understanding.
7. Perform a final cross-chapter pass for color semantics, label size, captions,
   and mobile/PDF behavior.

The desired result is not a book in which every page calls attention to its
design. It is a book in which the right relationship becomes visible at the
right moment, and the reader can move from picture to prose to equation without
having to reconstruct the explanation.
