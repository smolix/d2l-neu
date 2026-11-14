#!/usr/bin/env python3
"""Regression tests for the generated-LaTeX postprocessor."""

import unittest

from d2l_preprocess import CHAPTER_NUMBERING, PDF_CHAPTER_FILES
from pathlib import Path

from fix_latex import (BOOK_COVER, SRC_MARKER, add_book_cover,
                       add_frontmatter_page_numbering, fix_all)


class AddFrontmatterPageNumberingTest(unittest.TestCase):
    def test_ignores_document_command_mentioned_in_comment(self):
        source = (
            '% Do this before \\begin{document}. This text must stay commented.\n'
            '\\usepackage{example}\n'
            '\\begin{document}\n'
            '\\maketitle\n'
        )

        result = add_frontmatter_page_numbering(source)

        self.assertIn(
            '% Do this before \\begin{document}. This text must stay commented.',
            result,
        )
        self.assertIn(
            '\\begin{document}\n\\pagenumbering{roman}\n\\maketitle',
            result,
        )
        self.assertEqual(result.count('\\pagenumbering{roman}'), 1)

    def test_requires_a_standalone_document_command(self):
        with self.assertRaisesRegex(ValueError, 'standalone'):
            add_frontmatter_page_numbering(
                '% Only a comment mentioning \\begin{document}.\n'
            )


class ChapterNumberingTest(unittest.TestCase):
    """fix_all() on a synthetic book: one \\chapter per PDF file, in order."""

    @classmethod
    def setUpClass(cls):
        chapters = ['\\chapter{Landing}\\label{landing}']
        for i, _ in enumerate(PDF_CHAPTER_FILES):
            chapters.append(f'\\chapter{{Title {i}}}\\label{{lab-{i}}}')
        chapters.append('\\chapter{References}\\label{references}')
        source = ('\\begin{document}\n'
                  + '\n\nText.\n\n'.join(chapters)
                  + '\n\\end{document}\n')
        cls.result = fix_all(source)

    def test_no_appendix_lettering(self):
        self.assertNotIn('\\appendix', self.result)

    def test_every_chapter_keeps_its_logical_number(self):
        for rel in PDF_CHAPTER_FILES:
            logical = CHAPTER_NUMBERING[rel]
            if logical is None or len(logical) != 1:
                continue
            i = PDF_CHAPTER_FILES.index(rel)
            self.assertIn(
                f'{SRC_MARKER}{rel}\n\\setcounter{{chapter}}'
                f'{{{logical[0] - 1}}}\n\\chapter{{Title {i}}}',
                self.result, rel)

    def test_math_tools_and_attic_are_numbered(self):
        for rel, n in [('chapter_mdl-linear-algebra/index.md', 33),
                       ('chapter_appendix-tools-for-deep-learning/index.md', 39),
                       ('chapter_gaussian-processes/index.md', 40)]:
            self.assertEqual(CHAPTER_NUMBERING[rel], [n])
            self.assertIn(f'{SRC_MARKER}{rel}\n\\setcounter{{chapter}}'
                          f'{{{n - 1}}}\n', self.result)


class AddBookCoverTest(unittest.TestCase):
    DOC = ('\\documentclass{book}\n% \\begin{document} in a comment\n'
           '\\begin{document}\n\\pagenumbering{roman}\n\\frontmatter\n'
           '\\maketitle\n\n\\tableofcontents\n\\end{document}\n')

    def test_cover_replaces_title_page(self):
        tex_dir = BOOK_COVER.parent.parent.parent / '_pdf' / 'pytorch'
        out = add_book_cover(self.DOC, tex_dir)
        self.assertNotIn('\\maketitle', out)
        self.assertIn('\\includepdf[pages=1]{../../static/cover/'
                      'd2l-cover-D-inter-trim-7x10in.pdf}', out)
        # pdfpages is loaded before the real \\begin{document}, not the comment
        self.assertIn('\\usepackage{pdfpages}\n\\begin{document}', out)
        self.assertIn('% \\begin{document} in a comment', out)
        self.assertEqual(add_book_cover(out, tex_dir), out)  # idempotent

    def test_cover_file_exists(self):
        self.assertTrue(BOOK_COVER.exists(), BOOK_COVER)


if __name__ == '__main__':
    unittest.main()
