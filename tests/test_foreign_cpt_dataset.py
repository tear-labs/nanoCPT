"""Smoke tests for the foreign-distribution CPT data path.

Covers:
1. The configurable CPT text-field flag exists with the expected default.
2. A locally-generated synthesis parquet (if present) has the right row shape
   for the CPT loader to consume.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import main


ROOT = Path(__file__).resolve().parents[1]


class CptTextFieldDefaultTest(unittest.TestCase):
    def test_default_is_text(self) -> None:
        self.assertEqual(main.DEFAULT_CPT_TEXT_FIELD, "text")


class LocalConlangParquetTest(unittest.TestCase):
    """If a local synthesis parquet exists (from running
    `scripts/synthesize_conlang_cpt.py`), check its row schema."""

    def _find_parquet(self) -> Path | None:
        candidates = sorted((ROOT / "data" / "conlang_cpt").glob("*/train.parquet"))
        return candidates[0] if candidates else None

    def test_parquet_shape(self) -> None:
        parquet_path = self._find_parquet()
        if parquet_path is None:
            self.skipTest("no local conlang corpus yet; run scripts/synthesize_conlang_cpt.py")

        import pyarrow.parquet as pq

        pf = pq.ParquetFile(parquet_path)
        self.assertGreater(pf.metadata.num_rows, 0)

        columns = set(pf.schema.names)
        self.assertIn("text", columns)

        table = pf.read_row_group(0)
        first_text = table["text"][0].as_py()
        self.assertIsInstance(first_text, str)
        self.assertGreater(len(first_text), 50)


class SynthesisSummaryTest(unittest.TestCase):
    """If a local synthesis summary exists, sanity-check it."""

    def test_summary_fields(self) -> None:
        candidates = sorted((ROOT / "data" / "conlang_cpt").glob("*/synthesis_summary.json"))
        if not candidates:
            self.skipTest("no synthesis_summary.json yet")
        summary = json.loads(candidates[0].read_text(encoding="utf-8"))
        for key in ("language_id", "total_chars", "rows_parquet"):
            self.assertIn(key, summary)
        self.assertGreater(summary["total_chars"], 0)
        self.assertGreater(summary["rows_parquet"], 0)


if __name__ == "__main__":
    unittest.main()
