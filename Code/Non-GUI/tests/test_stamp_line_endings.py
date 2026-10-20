"""A checkpoint or corpus stamped on one checkout loads on another.

The v2 vocabulary hash and the card vocabulary hash are taken over files that
git checks out with CRLF on Windows (autocrlf) and LF elsewhere. Hashing the
raw bytes made the A7 checkpoint, trained on Windows, refuse to load on the
cluster ("vocab_v2_hash be0485ac9f00 != today's 8569510badb6"). The hashes now
normalize line endings, and stamps made before that are aliased to the same
value.
"""

import hashlib
import os
import unittest
from unittest import mock

from agent import spec_v2 as S
from agent import vocab


class TestLineEndings(unittest.TestCase):
    def _hash_with(self, convert):
        real_open = open

        def fake_open(p, mode="r", *a, **kw):
            f = real_open(p, mode, *a, **kw)
            if "b" not in mode or not str(p).startswith(vocab._DIR):
                return f
            data = convert(f.read())
            f.close()
            import io

            return io.BytesIO(data)

        with mock.patch("builtins.open", fake_open):
            return vocab.vocab_hash()

    def test_the_vocabulary_hash_ignores_line_endings(self):
        lf = self._hash_with(lambda b: b.replace(b"\r\n", b"\n"))
        crlf = self._hash_with(lambda b: b.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
        self.assertEqual(lf, crlf)
        self.assertEqual(lf, vocab.vocab_hash())

    def test_stamps_from_before_the_fix_are_today_s(self):
        now = S.stamp()
        for key, alias in S._LEGACY_HASHES.items():
            (old, new), = alias.items()
            if now[key] != new:
                self.skipTest(f"the {key} files changed since; the alias is history now")
            self.assertTrue(S.same_stamp(dict(now, **{key: old})))
        self.assertFalse(S.same_stamp(dict(now, vocab_v2_hash="0" * 64)))

    def test_the_card_vocabulary_hash_ignores_line_endings(self):
        from keyforge.cards import vocabulary as v

        with open(v.VOCAB_JSON_PATH, "rb") as f:
            raw = f.read()
        self.assertEqual(v.VOCAB_HASH, hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest())
        self.assertTrue(os.path.isfile(v.VOCAB_JSON_PATH))


if __name__ == "__main__":
    unittest.main()
