#!/usr/bin/env python3
"""Focused tests for PDF project generation."""

import tempfile
import unittest
from pathlib import Path

import gen_pdf


class PdfProjectConfigurationTest(unittest.TestCase):
    def test_links_use_visible_print_friendly_color(self):
        with tempfile.TemporaryDirectory() as tmp:
            dst = Path(tmp)
            gen_pdf._write_quarto_yml(dst, 'pytorch', [])
            config = (dst / '_quarto.yml').read_text(encoding='utf-8')

        for setting in ('linkcolor', 'citecolor', 'urlcolor'):
            self.assertIn(
                f'{setting}: "{gen_pdf.PDF_LINK_COLOR}"',
                config,
            )
        self.assertNotIn('color: black', config)


if __name__ == '__main__':
    unittest.main()
