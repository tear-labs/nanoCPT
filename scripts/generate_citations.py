#!/usr/bin/env python3
"""Generate project, record, and Markdown reference citations."""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import subprocess
import sys
import textwrap
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
PROJECT_NAME = "modded-continued-training"
PROJECT_URL = "https://github.com/Lev-Stambler/modded-continued-training"
RECORDS_BIB = ROOT / "citations" / "records.bib"
DOI_OVERRIDES = ROOT / "citations" / "doi-overrides.json"
REFERENCE_CACHE = ROOT / "citations" / "reference-cache.json"
REFERENCES_BIB = ROOT / "references.bib"
REFERENCES_MD = ROOT / "REFERENCES.md"
CITATION_CFF = ROOT / "CITATION.cff"


DOI_RE = re.compile(r"https?://(?:dx\.)?doi\.org/(10\.\d{4,9}/[^\s<>)\]]+)", re.I)
ARXIV_RE = re.compile(r"https?://arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})(?:v\d+)?", re.I)
CITE_RE = re.compile(r"\[@([A-Za-z0-9_:.+-]+)\]")
TRAILING_PUNCT = ".,;:"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def normalize_source(source: str) -> str:
    return source.rstrip(TRAILING_PUNCT)


def bib_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\\",
        "{": r"\{",
        "}": r"\}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "record"


def record_key(record_dir: Path) -> str:
    try:
        key_path = record_dir.relative_to(ROOT)
    except ValueError:
        key_path = record_dir
    return f"mct-{slugify(key_path.as_posix())}"


def git_record_ref(path: Path | None = None) -> str:
    ref = os.environ.get("MCT_CITATION_REF", "").strip()
    if ref:
        return ref
    if path is not None:
        try:
            return subprocess.check_output(
                ["git", "log", "-n", "1", "--format=%H", "--", str(path.relative_to(ROOT))],
                cwd=ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError, ValueError):
            pass
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "main"


def parse_authors(contributors: str) -> tuple[str, bool]:
    pieces = [
        part.strip().lstrip("@")
        for part in re.split(r"\s*(?:,|;|\band\b)\s*", contributors)
        if part.strip()
    ]
    if not pieces:
        return "{modded-continued-training contributors}", True
    return " and ".join(bib_escape(piece) for piece in pieces), False


def summary_paths() -> list[Path]:
    return sorted((ROOT / "records").glob("**/summary.json"))


def load_doi_overrides() -> dict[str, str]:
    if not DOI_OVERRIDES.exists():
        return {}
    return json.loads(read_text(DOI_OVERRIDES))


def record_bibtex(path: Path, doi_overrides: dict[str, str]) -> str:
    summary = json.loads(read_text(path))
    record_dir = path.parent
    key = record_key(record_dir)
    description = summary.get("record_description") or record_dir.name
    track = summary.get("track", "")
    track_name = summary.get("track_name", "")
    date = str(summary.get("record_date") or "")[:10]
    year = date[:4] if len(date) >= 4 else ""
    title = f"{description}: {PROJECT_NAME} track {track} ({track_name}) record"
    url = f"{PROJECT_URL}/tree/{git_record_ref(path)}/{record_dir.relative_to(ROOT).as_posix()}"

    authors, used_author_fallback = parse_authors(str(summary.get("record_contributors") or ""))
    note = "Run artifact snapshot with source, configuration, metrics, and summary."
    if used_author_fallback:
        note += " Contributor metadata was not recorded for this run."

    fields: list[tuple[str, object]] = [
        ("author", authors),
        ("title", title),
        ("year", year),
        ("url", url),
        ("note", note),
        ("howpublished", "GitHub record artifact"),
    ]
    for source, target in (
        ("record_date", "date"),
        ("model_id", "model"),
        ("dataset_id", "dataset"),
        ("data_mode", "data_mode"),
        ("adapter_mode", "adapter_mode"),
        ("optimizer_name", "optimizer"),
        ("eval_loss_drop", "eval_loss_drop"),
        ("baseline_eval_loss", "baseline_eval_loss"),
        ("final_eval_loss", "final_eval_loss"),
        ("steps", "steps"),
        ("tokens", "tokens"),
    ):
        value = summary.get(source)
        if value is not None:
            fields.append((target, value))
    doi = doi_overrides.get(key)
    if doi:
        fields.append(("doi", doi))

    body = []
    raw_fields = {"url", "doi"}
    for field, value in fields:
        if field == "author" and str(value).startswith("{") and str(value).endswith("}"):
            body.append(f"  {field} = {{{value}}},")
        elif field in raw_fields:
            body.append(f"  {field} = {{{value}}},")
        else:
            body.append(f"  {field} = {{{bib_escape(value)}}},")
    return f"@misc{{{key},\n" + "\n".join(body) + "\n}"


def generate_records_bib() -> str:
    overrides = load_doi_overrides()
    paths = summary_paths()
    keys = [record_key(path.parent) for path in paths]
    duplicate_keys = sorted({key for key in keys if keys.count(key) > 1})
    if duplicate_keys:
        raise ValueError(f"duplicate record citation keys: {', '.join(duplicate_keys)}")
    entries = [record_bibtex(path, overrides) for path in paths]
    return generated_header("BibTeX record citations") + "\n\n".join(entries) + "\n"


def generated_header(name: str) -> str:
    return f"% Generated by scripts/generate_citations.py. Do not edit manually.\n% {name}.\n\n"


def generate_citation_cff() -> str:
    return textwrap.dedent(
        f"""\
        # Generated by scripts/generate_citations.py. Do not edit manually.
        cff-version: 1.2.0
        message: "If you use this project or one of its run records, please cite the relevant record artifact."
        title: "{PROJECT_NAME}"
        type: software
        abstract: "Competitive single-H100 fine-tuning speedrun on Modal."
        repository-code: "{PROJECT_URL}"
        url: "{PROJECT_URL}"
        license: Apache-2.0
        authors:
          - family-names: "Stambler"
            given-names: "Lev"
          - name: "modded-continued-training contributors"
        """
    )


def markdown_paths() -> list[Path]:
    skipped = {"REFERENCES.md"}
    skipped_parts = {".git", ".venv", "__pycache__"}
    return [
        path
        for path in sorted(ROOT.rglob("*.md"))
        if path.name not in skipped and not (set(path.relative_to(ROOT).parts) & skipped_parts)
    ]


def discover_reference_sources() -> list[str]:
    sources: set[str] = set()
    for path in markdown_paths():
        text = read_text(path)
        for match in DOI_RE.finditer(text):
            sources.add(f"doi:{normalize_source(match.group(1)).lower()}")
        for match in ARXIV_RE.finditer(text):
            sources.add(f"arxiv:{normalize_source(match.group(1))}")
    return sorted(sources)


def reference_key(source: str) -> str:
    kind, value = source.split(":", 1)
    return f"{kind}-{slugify(value)}"


def load_reference_cache() -> dict[str, str]:
    if not REFERENCE_CACHE.exists():
        return {}
    return json.loads(read_text(REFERENCE_CACHE))


def save_reference_cache(cache: dict[str, str]) -> None:
    write_text(REFERENCE_CACHE, json.dumps(dict(sorted(cache.items())), indent=2) + "\n")


def fetch_url(url: str, headers: dict[str, str] | None = None) -> bytes:
    request = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def fetch_doi_bibtex(source: str, key: str) -> str:
    doi = source.split(":", 1)[1]
    data = fetch_url(
        f"https://doi.org/{urllib.parse.quote(doi, safe='/')}",
        {"Accept": "application/x-bibtex"},
    ).decode("utf-8")
    return re.sub(r"@\w+\{[^,]+,", lambda m: m.group(0).split("{", 1)[0] + "{" + key + ",", data, count=1).strip()


def fetch_arxiv_bibtex(source: str, key: str) -> str:
    arxiv_id = source.split(":", 1)[1]
    query = urllib.parse.urlencode({"id_list": arxiv_id})
    xml = fetch_url(f"https://export.arxiv.org/api/query?{query}").decode("utf-8")
    root = ET.fromstring(xml)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    entry = root.find("atom:entry", ns)
    if entry is None:
        raise ValueError(f"arXiv API returned no entry for {arxiv_id}")
    title = " ".join((entry.findtext("atom:title", default="", namespaces=ns) or "").split())
    published = entry.findtext("atom:published", default="", namespaces=ns) or ""
    year = published[:4]
    authors = [
        " ".join((author.findtext("atom:name", default="", namespaces=ns) or "").split())
        for author in entry.findall("atom:author", ns)
    ]
    author_field = " and ".join(bib_escape(author) for author in authors if author) or "{unknown}"
    return "\n".join(
        [
            f"@misc{{{key},",
            f"  author = {{{author_field}}},",
            f"  title = {{{bib_escape(title)}}},",
            f"  year = {{{bib_escape(year)}}},",
            f"  eprint = {{{bib_escape(arxiv_id)}}},",
            "  archivePrefix = {arXiv},",
            f"  url = {{https://arxiv.org/abs/{bib_escape(arxiv_id)}}},",
            "}",
        ]
    )


def refresh_reference_cache(cache: dict[str, str], sources: Iterable[str]) -> dict[str, str]:
    updated = dict(cache)
    for source in sources:
        key = reference_key(source)
        try:
            if source.startswith("doi:"):
                updated[source] = fetch_doi_bibtex(source, key)
            elif source.startswith("arxiv:"):
                updated[source] = fetch_arxiv_bibtex(source, key)
        except (urllib.error.URLError, TimeoutError, ValueError, ET.ParseError) as exc:
            if source not in updated:
                raise RuntimeError(f"could not resolve {source}: {exc}") from exc
    return updated


def parse_bib_keys(content: str) -> set[str]:
    return set(re.findall(r"@\w+\{\s*([^,\s]+)", content))


def generate_references_bib(cache: dict[str, str], sources: list[str]) -> str:
    entries = [cache[source].strip() for source in sources if source in cache]
    return generated_header("External paper references") + "\n\n".join(entries) + ("\n" if entries else "")


def generate_references_md(references_bib: str) -> str:
    entries = []
    for match in re.finditer(r"@\w+\{\s*([^,\s]+),(.*?)(?=\n@\w+\{|\Z)", references_bib, re.S):
        key = match.group(1)
        body = match.group(2)
        title_match = re.search(r"\btitle\s*=\s*\{(.+?)\}", body, re.S)
        url_match = re.search(r"\burl\s*=\s*\{(.+?)\}", body, re.S)
        title = re.sub(r"\s+", " ", title_match.group(1)).strip() if title_match else key
        url = url_match.group(1).strip() if url_match else ""
        line = f"- `[{key}]` "
        line += f"[{title}]({url})" if url else title
        entries.append(line)
    body = "\n".join(entries) if entries else "_No DOI or arXiv references found in Markdown._"
    return (
        "<!-- Generated by scripts/generate_citations.py. Do not edit manually. -->\n"
        "# References\n\n"
        f"{body}\n"
    )


def validate_markdown_cites(bib_keys: set[str]) -> list[str]:
    errors: list[str] = []
    for path in markdown_paths():
        for key in CITE_RE.findall(read_text(path)):
            if key not in bib_keys:
                errors.append(f"{path.relative_to(ROOT)} references missing cite key: {key}")
    return errors


def expected_files(refresh_references: bool) -> dict[Path, str]:
    sources = discover_reference_sources()
    cache = load_reference_cache()
    if refresh_references:
        cache = refresh_reference_cache(cache, sources)
    references_bib = generate_references_bib(cache, sources)
    return {
        CITATION_CFF: generate_citation_cff(),
        RECORDS_BIB: generate_records_bib(),
        DOI_OVERRIDES: "{}\n" if not DOI_OVERRIDES.exists() else read_text(DOI_OVERRIDES),
        REFERENCE_CACHE: json.dumps(dict(sorted(cache.items())), indent=2) + "\n",
        REFERENCES_BIB: references_bib,
        REFERENCES_MD: generate_references_md(references_bib),
    }


def show_diff(path: Path, expected: str) -> str:
    current = read_text(path) if path.exists() else ""
    return "".join(
        difflib.unified_diff(
            current.splitlines(keepends=True),
            expected.splitlines(keepends=True),
            fromfile=str(path.relative_to(ROOT)),
            tofile=f"{path.relative_to(ROOT)} (expected)",
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="write generated citation files")
    mode.add_argument("--check", action="store_true", help="check generated citation files are current")
    parser.add_argument(
        "--refresh-references",
        action="store_true",
        help="resolve DOI/arXiv Markdown links over the network before generating references",
    )
    args = parser.parse_args(argv)

    files = expected_files(refresh_references=args.refresh_references)
    bib_keys = parse_bib_keys(files[REFERENCES_BIB]) | parse_bib_keys(files[RECORDS_BIB])
    errors = validate_markdown_cites(bib_keys)
    missing_sources = [
        source for source in discover_reference_sources()
        if source not in load_reference_cache() and source not in json.loads(files[REFERENCE_CACHE])
    ]
    if missing_sources:
        errors.append(
            "unresolved DOI/arXiv references: "
            + ", ".join(missing_sources)
            + " (run with --write --refresh-references)"
        )
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    if args.write:
        for path, content in files.items():
            write_text(path, content)
        return 0

    diffs = [show_diff(path, content) for path, content in files.items() if not path.exists() or read_text(path) != content]
    if diffs:
        print("\n".join(diffs), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
