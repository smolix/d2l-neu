#!/usr/bin/env python3
"""CPU tests for sec_lm_word_embeddings (chapter_lm-encoders/word-embeddings.md),
the shared part: `TokenEmbedding` on the fixture `vec.txt` (three words),
loaded through a stubbed `download_extract`. Until the retiring
`similarity-analogy.md` is deleted, the library keeps that section's
definition, which is the same code; the test covers whichever is in force."""
import os
import unittest

import torch
from d2l import torch as d2l

GLOVE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lm',
                     'fixtures', 'encoders', 'glove')


class TokenEmbeddingTest(unittest.TestCase):
    def test_fixture(self):
        saved = d2l.download_extract
        d2l.download_extract = lambda name, folder=None: GLOVE
        try:
            emb = d2l.TokenEmbedding('glove.fixture')
        finally:
            d2l.download_extract = saved
        self.assertEqual(len(emb), 4)
        self.assertEqual(emb.idx_to_token, ['<unk>', 'the', 'cat', 'dog'])
        self.assertEqual(emb.token_to_idx['cat'], 2)
        self.assertEqual(emb.unknown_idx, 0)
        torch.testing.assert_close(
            emb[['cat', 'nope']],
            torch.tensor([[1.0, 0.0, -1.0], [0.0, 0.0, 0.0]]))


if __name__ == '__main__':
    unittest.main()
