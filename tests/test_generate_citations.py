from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import generate_citations as citations  # noqa: E402


class GenerateCitationTests(unittest.TestCase):
    def test_record_key_is_stable(self) -> None:
        path = Path("records/track_1_30min/2026-05-25_sft_gralora_track1_candidate")
        self.assertEqual(
            citations.record_key(path),
            "mct-records-track-1-30min-2026-05-25-sft-gralora-track1-candidate",
        )

    def test_empty_contributors_use_project_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            record_dir = Path(tmp) / "records" / "track" / "2026-05-25_test"
            record_dir.mkdir(parents=True)
            summary = {
                "record_description": "test run",
                "record_contributors": "",
                "record_date": "2026-05-25",
                "track": "1",
                "track_name": "30min",
                "eval_loss_drop": 0.1,
            }
            summary_path = record_dir / "summary.json"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            old_root = citations.ROOT
            try:
                citations.ROOT = Path(tmp)
                bibtex = citations.record_bibtex(summary_path, {})
            finally:
                citations.ROOT = old_root
        self.assertIn("author = {{modded-continued-training contributors}}", bibtex)
        self.assertIn("contributor metadata was not recorded", bibtex.lower())

    def test_markdown_cite_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "README.md").write_text("Uses [@known] and [@missing].\n", encoding="utf-8")
            old_root = citations.ROOT
            try:
                citations.ROOT = root
                errors = citations.validate_markdown_cites({"known"})
            finally:
                citations.ROOT = old_root
        self.assertEqual(errors, ["README.md references missing cite key: missing"])

    def test_markdown_cites_can_reference_record_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "README.md").write_text("Uses [@record-key].\n", encoding="utf-8")
            old_root = citations.ROOT
            try:
                citations.ROOT = root
                errors = citations.validate_markdown_cites({"record-key"})
            finally:
                citations.ROOT = old_root
        self.assertEqual(errors, [])

    def test_discovers_doi_and_arxiv_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "README.md").write_text(
                "See https://doi.org/10.48550/arXiv.2407.05000 and https://arxiv.org/abs/2402.12354.\n",
                encoding="utf-8",
            )
            old_root = citations.ROOT
            try:
                citations.ROOT = root
                sources = citations.discover_reference_sources()
            finally:
                citations.ROOT = old_root
        self.assertEqual(sources, ["arxiv:2402.12354", "doi:10.48550/arxiv.2407.05000"])

    def test_record_keys_include_parent_track_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_root = citations.ROOT
            try:
                citations.ROOT = root
                self.assertNotEqual(
                    citations.record_key(root / "records/a/same"),
                    citations.record_key(root / "records/b/same"),
                )
            finally:
                citations.ROOT = old_root


if __name__ == "__main__":
    unittest.main()
