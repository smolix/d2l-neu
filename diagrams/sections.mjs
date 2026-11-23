// Section numbers for slide-card text, by label, so cards survive chapter
// renumbering:  `${sec('sec_rl-td')} Tabular TD`  → "§14.3 Tabular TD",
// `§§${num('sec_a')}–${num('sec_b')}` → "§§11.1–11.2".
//
// The numbers come from section-numbers.json, which tools/section_refs.py
// writes from CHAPTER_NUMBERING (the map the book, PDFs and slides use).
// `make figures` refreshes it, and a changed number re-renders every
// diagram; `make html` refuses to render while it is out of date.
// A literal "§14.3" in diagram code fails that same check.
// Labels are the :label: of a chapter's or section's H1.
import { readFileSync } from 'node:fs';

const NUMBERS = JSON.parse(
  readFileSync(new URL('./section-numbers.json', import.meta.url), 'utf8'));

export function num(label) {
  const n = NUMBERS[label];
  if (n === undefined) {
    throw new Error(`sections.mjs: \`${label}\` is not the label of a numbered `
      + 'chapter or section (diagrams/section-numbers.json)');
  }
  return n;
}

export const sec = (label) => `§${num(label)}`;
