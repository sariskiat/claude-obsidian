"""P4 full-paper retrieval tests — resolver/import, skip-bodyless, idempotent, gitignore.

Task T1: failing tests first (AC1, AC2, AC6, AC7).
Tasks T4, T5 will add tests for AC3/AC4/AC5/AC9 in later cycles.

AC1: Resolver + import writes one .full.md per Tier-A paper, byte-equal body +
     contract frontmatter (type:paper-fulltext, slug, arxiv_id, source_path, paper).
AC2: Bodyless papers (Tier-B/C: pdf / paper.json / URL / stale path / bare dir
     with no inner .md) are skipped — no .full.md written, no crash, reported.
AC6: Import + index build are idempotent — run twice with same input -> identical
     .full.md files and identical index.
AC7: .vault-meta/graph/chunks/ and .vault-meta/graph/bm25/ are gitignored;
     wiki/graph/papers/*.full.md are git-tracked (not ignored).
"""

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = PROJECT_ROOT / "scripts"
FULLTEXT_SCRIPT = SCRIPTS / "graph-fulltext.py"
GRAPH_EXPORT = PROJECT_ROOT / "wiki" / "graph" / "graph-export.json"
PAPERS_DIR = PROJECT_ROOT / "wiki" / "graph" / "papers"

# ============================================================================
# Helpers
# ============================================================================

FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n?", re.DOTALL)


def _parse_frontmatter(text):
    """Return dict of frontmatter key:value from yaml-ish frontmatter, or {}."""
    m = FRONTMATTER_RE.match(text)
    if not m:
        return {}
    fm = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            fm[k.strip()] = v.strip().strip("'\"")
    return fm


def _body_after_frontmatter(text):
    """Return text after the frontmatter block."""
    m = FRONTMATTER_RE.match(text)
    if not m:
        return text
    return text[m.end():]


def _run_sync(*extra_args, env=None, cwd=None):
    """Run graph-fulltext.py sync and return (returncode, stdout, stderr)."""
    cmd = [sys.executable, str(FULLTEXT_SCRIPT), "sync"] + list(extra_args)
    r = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=cwd or PROJECT_ROOT,
        env=env or os.environ.copy(),
        timeout=120,
    )
    return r.returncode, r.stdout, r.stderr


def _load_graph_export():
    if not GRAPH_EXPORT.is_file():
        return None
    return json.loads(GRAPH_EXPORT.read_text(encoding="utf-8"))


def _resolve_tier_a(graph_export_data):
    """Compute the Tier-A set (papers with a real, existing .md source).

    Mirrors the resolver logic in graph-fulltext.py sync:
      - Absolute .md path that exists -> Tier-A
      - Path ends paper.json -> bare dir -> single inner .md -> Tier-A
      - Bare dir (no extension) that is a dir -> single inner .md -> Tier-A
      - Everything else (PDF, URL, stale, multi-md) -> Tier-B/skip
    """
    papers = graph_export_data.get("papers", [])
    tier_a = []
    for p in papers:
        sp = p.get("source_path") or ""
        if not sp or sp.startswith("http"):
            continue
        sp_exp = Path(os.path.expanduser(sp))
        # direct .md
        if sp_exp.suffix == ".md" and sp_exp.is_file():
            tier_a.append({"slug": p["slug"], "resolved_path": sp_exp,
                            "arxiv_id": p.get("arxiv_id")})
            continue
        # paper.json -> bare dir
        if sp_exp.name == "paper.json":
            d = sp_exp.parent
            if d.is_dir():
                mds = [f for f in d.iterdir() if f.suffix == ".md"]
                if len(mds) == 1:
                    tier_a.append({"slug": p["slug"], "resolved_path": mds[0],
                                   "arxiv_id": p.get("arxiv_id")})
                    continue
        # bare dir (no extension)
        if not sp_exp.suffix and sp_exp.is_dir():
            mds = [f for f in sp_exp.iterdir() if f.suffix == ".md"]
            if len(mds) == 1:
                tier_a.append({"slug": p["slug"], "resolved_path": mds[0],
                                "arxiv_id": p.get("arxiv_id")})
    return tier_a


def _resolve_tier_b(graph_export_data):
    """All papers that should be skipped (non Tier-A)."""
    papers = graph_export_data.get("papers", [])
    tier_a_slugs = {e["slug"] for e in _resolve_tier_a(graph_export_data)}
    return [p["slug"] for p in papers if p["slug"] not in tier_a_slugs]


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def graph_export_data():
    data = _load_graph_export()
    if data is None:
        pytest.skip("wiki/graph/graph-export.json not found")
    return data


@pytest.fixture(scope="session")
def tier_a_papers(graph_export_data):
    return _resolve_tier_a(graph_export_data)


@pytest.fixture(scope="session")
def tier_b_papers(graph_export_data):
    return _resolve_tier_b(graph_export_data)


# ============================================================================
# AC1: import — byte-equal body + contract frontmatter
# ============================================================================

class TestImport:
    """AC1: Each Tier-A paper gets a .full.md with correct frontmatter + body."""

    def test_fulltext_script_exists(self):
        """graph-fulltext.py must exist before any sync can be tested."""
        assert FULLTEXT_SCRIPT.exists(), (
            f"scripts/graph-fulltext.py does not exist — implement it (T2)"
        )

    @pytest.mark.skipif(not GRAPH_EXPORT.exists(), reason="graph-export.json not found")
    def test_sync_exits_zero(self, tier_a_papers):
        """sync subcommand must exit 0."""
        rc, stdout, stderr = _run_sync()
        assert rc == 0, f"graph-fulltext.py sync exited {rc}\nstdout: {stdout}\nstderr: {stderr}"

    @pytest.mark.skipif(not GRAPH_EXPORT.exists(), reason="graph-export.json not found")
    def test_full_md_exists_for_every_tier_a_paper(self, tier_a_papers):
        """Every Tier-A paper must have a corresponding .full.md written."""
        assert len(tier_a_papers) > 0, "No Tier-A papers found — check graph-export.json"
        missing = []
        for entry in tier_a_papers:
            slug = entry["slug"]
            full_md = PAPERS_DIR / f"{slug}.full.md"
            if not full_md.exists():
                missing.append(slug)
        assert not missing, (
            f"{len(missing)} Tier-A .full.md missing:\n" + "\n".join(f"  {s}" for s in missing)
        )

    @pytest.mark.skipif(not GRAPH_EXPORT.exists(), reason="graph-export.json not found")
    def test_full_md_body_byte_equal_to_source(self, tier_a_papers):
        """Body of each .full.md must be byte-equal to its source .md content."""
        assert len(tier_a_papers) > 0
        mismatches = []
        for entry in tier_a_papers:
            slug = entry["slug"]
            full_md = PAPERS_DIR / f"{slug}.full.md"
            if not full_md.exists():
                mismatches.append(f"  {slug}: .full.md missing")
                continue
            src_bytes = entry["resolved_path"].read_bytes()
            full_str = full_md.read_text(encoding="utf-8", errors="replace")
            body_after = _body_after_frontmatter(full_str).encode("utf-8", errors="replace")
            if body_after != src_bytes:
                mismatches.append(
                    f"  {slug}: body not byte-equal "
                    f"(expected {len(src_bytes)} bytes, got {len(body_after)} bytes)"
                )
        assert not mismatches, "Byte-equal violations:\n" + "\n".join(mismatches)

    @pytest.mark.skipif(not GRAPH_EXPORT.exists(), reason="graph-export.json not found")
    def test_full_md_contract_frontmatter_keys(self, tier_a_papers):
        """Every .full.md must have all required frontmatter keys."""
        required_keys = {"type", "slug", "arxiv_id", "source_path", "paper"}
        assert len(tier_a_papers) > 0
        violations = []
        for entry in tier_a_papers:
            slug = entry["slug"]
            full_md = PAPERS_DIR / f"{slug}.full.md"
            if not full_md.exists():
                violations.append(f"  {slug}: .full.md missing")
                continue
            text = full_md.read_text(encoding="utf-8")
            fm = _parse_frontmatter(text)
            missing_keys = required_keys - set(fm.keys())
            if missing_keys:
                violations.append(f"  {slug}: missing keys {missing_keys}")
        assert not violations, "Frontmatter contract violations:\n" + "\n".join(violations)

    @pytest.mark.skipif(not GRAPH_EXPORT.exists(), reason="graph-export.json not found")
    def test_full_md_type_field_value(self, tier_a_papers):
        """type: field must be exactly 'paper-fulltext'."""
        assert len(tier_a_papers) > 0
        violations = []
        for entry in tier_a_papers:
            slug = entry["slug"]
            full_md = PAPERS_DIR / f"{slug}.full.md"
            if not full_md.exists():
                continue
            fm = _parse_frontmatter(full_md.read_text(encoding="utf-8"))
            if fm.get("type") != "paper-fulltext":
                violations.append(f"  {slug}: type={fm.get('type')!r}")
        assert not violations, "type field violations:\n" + "\n".join(violations)

    @pytest.mark.skipif(not GRAPH_EXPORT.exists(), reason="graph-export.json not found")
    def test_full_md_slug_matches_filename(self, tier_a_papers):
        """slug: field must match the filename stem."""
        assert len(tier_a_papers) > 0
        violations = []
        for entry in tier_a_papers:
            slug = entry["slug"]
            full_md = PAPERS_DIR / f"{slug}.full.md"
            if not full_md.exists():
                continue
            fm = _parse_frontmatter(full_md.read_text(encoding="utf-8"))
            if fm.get("slug") != slug:
                violations.append(f"  {slug}: fm slug={fm.get('slug')!r}")
        assert not violations, "Slug mismatch:\n" + "\n".join(violations)

    @pytest.mark.skipif(not GRAPH_EXPORT.exists(), reason="graph-export.json not found")
    def test_full_md_paper_field_is_wikilink(self, tier_a_papers):
        """paper: field must be the Obsidian wikilink [[<slug>]]."""
        assert len(tier_a_papers) > 0
        violations = []
        for entry in tier_a_papers:
            slug = entry["slug"]
            full_md = PAPERS_DIR / f"{slug}.full.md"
            if not full_md.exists():
                continue
            fm = _parse_frontmatter(full_md.read_text(encoding="utf-8"))
            expected = f"[[{slug}]]"
            if fm.get("paper") != expected:
                violations.append(f"  {slug}: paper={fm.get('paper')!r} expected={expected!r}")
        assert not violations, "paper wikilink violations:\n" + "\n".join(violations)


# ============================================================================
# AC2: skip_bodyless — no .full.md written for Tier-B/C papers
# ============================================================================

class TestSkipBodyless:
    """AC2: Bodyless / non-Tier-A papers are skipped — no .full.md, no crash."""

    def test_stale_source_path_does_not_crash(self, tmp_path):
        """A paper whose source_path points to a deleted file is skipped, exit 0."""
        assert FULLTEXT_SCRIPT.exists(), "graph-fulltext.py must be implemented first (T2)"

        src_file = tmp_path / "real_paper.md"
        src_file.write_text("# Real paper\n\nBody content.", encoding="utf-8")

        fake_export = {
            "papers": [
                {
                    "slug": "real-paper",
                    "source_path": str(src_file),
                    "arxiv_id": None,
                },
                {
                    "slug": "stale-paper",
                    "source_path": str(tmp_path / "nonexistent.md"),
                    "arxiv_id": None,
                },
            ],
            "entities": [],
            "claims": [],
            "sections": [],
            "paper_authors": [],
            "predicates": [],
            "entity_edges": [],
            "citation_links": [],
            "aliases": [],
        }
        export_path = tmp_path / "graph-export.json"
        export_path.write_text(json.dumps(fake_export), encoding="utf-8")
        papers_out = tmp_path / "papers"
        papers_out.mkdir()
        meta_out = tmp_path / ".vault-meta" / "graph"
        meta_out.mkdir(parents=True)

        rc, stdout, stderr = _run_sync(
            "--export", str(export_path),
            "--papers-dir", str(papers_out),
            "--chunks-dir", str(meta_out / "chunks"),
            "--bm25-dir", str(meta_out / "bm25"),
        )
        assert rc == 0, f"exit {rc} on stale source\nstdout: {stdout}\nstderr: {stderr}"
        assert (papers_out / "real-paper.full.md").exists(), "real-paper.full.md should exist"
        assert not (papers_out / "stale-paper.full.md").exists(), "stale-paper.full.md should NOT exist"

    def test_pdf_source_path_is_skipped(self, tmp_path):
        """A paper with a PDF source_path is skipped without writing .full.md."""
        assert FULLTEXT_SCRIPT.exists(), "graph-fulltext.py must be implemented first (T2)"

        pdf_file = tmp_path / "paper.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake content")

        fake_export = {
            "papers": [
                {
                    "slug": "pdf-paper",
                    "source_path": str(pdf_file),
                    "arxiv_id": "2501.99999",
                },
            ],
            "entities": [], "claims": [], "sections": [],
            "paper_authors": [], "predicates": [], "entity_edges": [],
            "citation_links": [], "aliases": [],
        }
        export_path = tmp_path / "graph-export.json"
        export_path.write_text(json.dumps(fake_export), encoding="utf-8")
        papers_out = tmp_path / "papers"
        papers_out.mkdir()
        meta_out = tmp_path / ".vault-meta" / "graph"
        meta_out.mkdir(parents=True)

        rc, stdout, stderr = _run_sync(
            "--export", str(export_path),
            "--papers-dir", str(papers_out),
            "--chunks-dir", str(meta_out / "chunks"),
            "--bm25-dir", str(meta_out / "bm25"),
        )
        assert rc == 0, f"exit {rc}\nstdout: {stdout}\nstderr: {stderr}"
        assert not (papers_out / "pdf-paper.full.md").exists(), (
            "PDF paper should be skipped — no .full.md"
        )

    def test_url_source_path_is_skipped(self, tmp_path):
        """A paper with a URL source_path is skipped without writing .full.md."""
        assert FULLTEXT_SCRIPT.exists(), "graph-fulltext.py must be implemented first (T2)"

        fake_export = {
            "papers": [
                {
                    "slug": "url-paper",
                    "source_path": "https://platform.claude.com/cookbook/example",
                    "arxiv_id": None,
                },
            ],
            "entities": [], "claims": [], "sections": [],
            "paper_authors": [], "predicates": [], "entity_edges": [],
            "citation_links": [], "aliases": [],
        }
        export_path = tmp_path / "graph-export.json"
        export_path.write_text(json.dumps(fake_export), encoding="utf-8")
        papers_out = tmp_path / "papers"
        papers_out.mkdir()
        meta_out = tmp_path / ".vault-meta" / "graph"
        meta_out.mkdir(parents=True)

        rc, stdout, stderr = _run_sync(
            "--export", str(export_path),
            "--papers-dir", str(papers_out),
            "--chunks-dir", str(meta_out / "chunks"),
            "--bm25-dir", str(meta_out / "bm25"),
        )
        assert rc == 0, f"exit {rc}\nstdout: {stdout}\nstderr: {stderr}"
        assert not (papers_out / "url-paper.full.md").exists(), (
            "URL paper should be skipped — no .full.md"
        )

    @pytest.mark.skipif(not GRAPH_EXPORT.exists(), reason="graph-export.json not found")
    def test_no_full_md_for_tier_b_papers(self, tier_b_papers):
        """No .full.md must exist for any Tier-B paper slug after sync."""
        wrongly_written = [
            s for s in tier_b_papers
            if (PAPERS_DIR / f"{s}.full.md").exists()
        ]
        assert not wrongly_written, (
            f"{len(wrongly_written)} Tier-B papers incorrectly got .full.md:\n"
            + "\n".join(f"  {s}" for s in wrongly_written)
        )

    @pytest.mark.skipif(not GRAPH_EXPORT.exists(), reason="graph-export.json not found")
    def test_sync_reports_skipped_count(self):
        """sync must log a skip count (stderr or stdout) when bodyless papers exist."""
        assert FULLTEXT_SCRIPT.exists(), "graph-fulltext.py must be implemented first (T2)"
        rc, stdout, stderr = _run_sync()
        combined = (stdout + stderr).lower()
        assert re.search(r"skip", combined), (
            "sync output should report skipped papers\n"
            f"stdout: {stdout[:500]}\nstderr: {stderr[:500]}"
        )


# ============================================================================
# AC6: idempotent — run twice -> byte-identical output
# ============================================================================

class TestIdempotent:
    """AC6: Two consecutive sync runs with no source change yield identical output."""

    def test_idempotent_full_md_content(self, tmp_path):
        """Running sync twice produces byte-identical .full.md files."""
        assert FULLTEXT_SCRIPT.exists(), "graph-fulltext.py must be implemented first (T2)"

        src_file = tmp_path / "paper_alpha.md"
        src_file.write_text("# Alpha\n\nBody text here. More content.", encoding="utf-8")

        fake_export = {
            "papers": [
                {"slug": "paper-alpha", "source_path": str(src_file), "arxiv_id": "2501.00001"},
            ],
            "entities": [], "claims": [], "sections": [],
            "paper_authors": [], "predicates": [], "entity_edges": [],
            "citation_links": [], "aliases": [],
        }
        export_path = tmp_path / "graph-export.json"
        export_path.write_text(json.dumps(fake_export), encoding="utf-8")
        papers_out = tmp_path / "papers"
        papers_out.mkdir()
        meta_out = tmp_path / ".vault-meta" / "graph"
        meta_out.mkdir(parents=True)

        common_args = [
            "--export", str(export_path),
            "--papers-dir", str(papers_out),
            "--chunks-dir", str(meta_out / "chunks"),
            "--bm25-dir", str(meta_out / "bm25"),
        ]

        rc1, _, stderr1 = _run_sync(*common_args)
        assert rc1 == 0, f"first run exit {rc1}: {stderr1}"
        full_md = papers_out / "paper-alpha.full.md"
        assert full_md.exists(), ".full.md not written on first run"
        content_run1 = full_md.read_bytes()

        rc2, _, stderr2 = _run_sync(*common_args)
        assert rc2 == 0, f"second run exit {rc2}: {stderr2}"
        content_run2 = full_md.read_bytes()

        assert content_run1 == content_run2, (
            "Idempotency violated: .full.md content changed between runs"
        )

    def test_idempotent_bm25_index(self, tmp_path):
        """Running sync twice produces identical BM25 index (excluding updated_at timestamp)."""
        assert FULLTEXT_SCRIPT.exists(), "graph-fulltext.py must be implemented first (T2)"

        src_file = tmp_path / "paper_beta.md"
        src_file.write_text(
            "# Beta\n\nThis paper discusses variational inference.\n\nMore analysis here.",
            encoding="utf-8",
        )
        fake_export = {
            "papers": [
                {"slug": "paper-beta", "source_path": str(src_file), "arxiv_id": None},
            ],
            "entities": [], "claims": [], "sections": [],
            "paper_authors": [], "predicates": [], "entity_edges": [],
            "citation_links": [], "aliases": [],
        }
        export_path = tmp_path / "graph-export.json"
        export_path.write_text(json.dumps(fake_export), encoding="utf-8")
        papers_out = tmp_path / "papers"
        papers_out.mkdir()
        chunks_dir = tmp_path / ".vault-meta" / "graph" / "chunks"
        bm25_dir = tmp_path / ".vault-meta" / "graph" / "bm25"
        chunks_dir.mkdir(parents=True)
        bm25_dir.mkdir(parents=True)

        common_args = [
            "--export", str(export_path),
            "--papers-dir", str(papers_out),
            "--chunks-dir", str(chunks_dir),
            "--bm25-dir", str(bm25_dir),
        ]

        rc1, _, _ = _run_sync(*common_args)
        assert rc1 == 0
        idx_path = bm25_dir / "index.json"
        assert idx_path.exists(), "BM25 index not written on first run"
        idx1 = json.loads(idx_path.read_text(encoding="utf-8"))
        idx1.pop("updated_at", None)

        rc2, _, _ = _run_sync(*common_args)
        assert rc2 == 0
        idx2 = json.loads(idx_path.read_text(encoding="utf-8"))
        idx2.pop("updated_at", None)

        assert idx1 == idx2, "BM25 index not idempotent across two runs"


# ============================================================================
# AC7: gitignore — chunks/bm25 ignored; .full.md tracked
# ============================================================================

class TestGitignore:
    """AC7: Retrieval index dirs gitignored; .full.md git-tracked."""

    def test_vault_meta_graph_chunks_is_gitignored(self):
        r = subprocess.run(
            ["git", "check-ignore", ".vault-meta/graph/chunks/any.json"],
            capture_output=True, text=True, cwd=PROJECT_ROOT,
        )
        assert r.returncode == 0, (
            ".vault-meta/graph/chunks/ should be gitignored\n"
            f"git check-ignore rc={r.returncode}, stdout={r.stdout!r}"
        )

    def test_vault_meta_graph_bm25_is_gitignored(self):
        r = subprocess.run(
            ["git", "check-ignore", ".vault-meta/graph/bm25/index.json"],
            capture_output=True, text=True, cwd=PROJECT_ROOT,
        )
        assert r.returncode == 0, (
            ".vault-meta/graph/bm25/ should be gitignored\n"
            f"git check-ignore rc={r.returncode}, stdout={r.stdout!r}"
        )

    def test_full_md_under_papers_is_not_gitignored(self):
        r = subprocess.run(
            ["git", "check-ignore", "wiki/graph/papers/sample.full.md"],
            capture_output=True, text=True, cwd=PROJECT_ROOT,
        )
        assert r.returncode != 0, (
            "wiki/graph/papers/sample.full.md should NOT be gitignored "
            "(it is the git-tracked source of truth)\n"
            f"git check-ignore rc={r.returncode}, stdout={r.stdout!r}"
        )
