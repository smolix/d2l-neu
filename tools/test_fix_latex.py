#!/usr/bin/env python3
"""Regression tests for the generated-LaTeX postprocessor."""

import unittest

from fix_latex import add_frontmatter_page_numbering


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


if __name__ == '__main__':
    unittest.main()
