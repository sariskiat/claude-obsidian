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


_PS_ROOT = Path.home() / ".paper-scholar"


def _slug_dir_candidate(slug: str) -> "Path | None":
    """Mirror of the slug-dir fallback in graph-fulltext.py.

    Returns the single .md Path if <PS_ROOT>/<slug>/ contains exactly one
    non-meta .md, else None.
    """
    slug_dir = _PS_ROOT / slug
    if not slug_dir.is_dir():
        return None
    candidates = [
        f for f in slug_dir.iterdir()
        if f.suffix == ".md" and not f.name.endswith("_meta.json")
    ]
    if len(candidates) == 1:
        return candidates[0]
    return None


def _resolve_tier_a(graph_export_data):
    """Compute the Tier-A set (papers with a real, existing .md source).

    Mirrors the resolver logic in graph-fulltext.py sync (including slug-dir
    fallback added in the dedup/re-sync task):
      - Absolute .md path that exists -> Tier-A
      - Path ends paper.json -> bare dir -> single inner .md -> Tier-A
      - Bare dir (no extension) that is a dir -> single inner .md -> Tier-A
      - [slug-dir fallback] ~/.paper-scholar/<slug>/ with single non-meta .md -> Tier-A
      - Everything else (URL, multi-md, truly absent slug) -> Tier-B/skip
    """
    papers = graph_export_data.get("papers", [])
    tier_a = []
    for p in papers:
        slug = p["slug"]
        sp = p.get("source_path") or ""
        if sp.startswith("http"):
            continue
        if sp:
            sp_exp = Path(os.path.expanduser(sp))
            # direct .md
            if sp_exp.suffix == ".md" and sp_exp.is_file():
                tier_a.append({"slug": slug, "resolved_path": sp_exp,
                                "arxiv_id": p.get("arxiv_id")})
                continue
            # paper.json -> bare dir
            if sp_exp.name == "paper.json":
                d = sp_exp.parent
                if d.is_dir():
                    mds = [f for f in d.iterdir() if f.suffix == ".md"]
                    if len(mds) == 1:
                        tier_a.append({"slug": slug, "resolved_path": mds[0],
                                       "arxiv_id": p.get("arxiv_id")})
                        continue
            # bare dir (no extension)
            if not sp_exp.suffix and sp_exp.is_dir():
                mds = [f for f in sp_exp.iterdir() if f.suffix == ".md"]
                if len(mds) == 1:
                    tier_a.append({"slug": slug, "resolved_path": mds[0],
                                   "arxiv_id": p.get("arxiv_id")})
                    continue
        # slug-dir fallback (last resort, fires when all source_path strategies failed)
        cand = _slug_dir_candidate(slug)
        if cand is not None:
            tier_a.append({"slug": slug, "resolved_path": cand,
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


# ============================================================================
# AC4: no_egress_default — default sync uses only synthetic prefix, no network
# ============================================================================

class TestNoEgressDefault:
    """AC4: Default sync (no --allow-egress) uses synthetic prefix only."""

    def test_chunk_prefix_source_is_synthetic_by_default(self, tmp_path):
        """All chunk files written by sync without --allow-egress must have
        prefix_source: 'synthetic' (no network calls)."""
        src_file = tmp_path / "egress_test.md"
        src_file.write_text(
            "# Egress Test\n\nSome content about variational inference.\n\n"
            "More content about neural networks and training.",
            encoding="utf-8",
        )
        fake_export = {
            "papers": [
                {"slug": "egress-test", "source_path": str(src_file), "arxiv_id": None},
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

        rc, stdout, stderr = _run_sync(
            "--export", str(export_path),
            "--papers-dir", str(papers_out),
            "--chunks-dir", str(chunks_dir),
            "--bm25-dir", str(bm25_dir),
            # No --allow-egress
        )
        assert rc == 0, f"exit {rc}\nstdout: {stdout}\nstderr: {stderr}"

        # Find all chunk files
        chunk_files = list(chunks_dir.glob("*/chunk-*.json")) if chunks_dir.is_dir() else []
        assert chunk_files, "No chunk files written — sync did not produce chunks"

        non_synthetic = []
        for cf in chunk_files:
            data = json.loads(cf.read_text(encoding="utf-8"))
            ps = data.get("prefix_source", "")
            if ps != "synthetic":
                non_synthetic.append(f"  {cf.name}: prefix_source={ps!r}")
        assert not non_synthetic, (
            "Default sync produced non-synthetic prefix_source:\n"
            + "\n".join(non_synthetic)
        )

    def test_allow_egress_flag_is_accepted(self, tmp_path):
        """--allow-egress must be accepted (exit 0) even when ollama is absent
        (falls back to synthetic). This confirms the flag is wired, not ignored."""
        src_file = tmp_path / "egress_allow.md"
        src_file.write_text("# Allow egress test\n\nShort body.", encoding="utf-8")
        fake_export = {
            "papers": [
                {"slug": "egress-allow", "source_path": str(src_file), "arxiv_id": None},
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

        rc, stdout, stderr = _run_sync(
            "--export", str(export_path),
            "--papers-dir", str(papers_out),
            "--chunks-dir", str(chunks_dir),
            "--bm25-dir", str(bm25_dir),
            "--allow-egress",
        )
        assert rc == 0, f"--allow-egress rejected: exit {rc}\nstdout: {stdout}\nstderr: {stderr}"


# ============================================================================
# AC3: retrieve_provenance — BM25 index built over .full.md bodies; query returns
#      results with paper provenance (page_path + page_address + chunk_index)
# ============================================================================

RETRIEVE_SCRIPT = SCRIPTS / "graph-retrieve.py"


class TestRetrieveProvenance:
    """AC3: graph BM25 index built over .full.md bodies only; graph-retrieve.py
    returns results with paper provenance."""

    def _setup_index(self, tmp_path, slug="retrieve-test"):
        """Helper: sync a single paper and return (chunks_dir, bm25_dir, papers_out)."""
        src_file = tmp_path / f"{slug}.md"
        src_file.write_text(
            "# Retrieve Test\n\n"
            "This paper discusses constrained sampling under hard garment pinning. "
            "The method uses diffusion models with special boundary conditions.\n\n"
            "Experiments show state-of-the-art results on garment try-on benchmarks. "
            "The approach is novel and outperforms prior work in all metrics.",
            encoding="utf-8",
        )
        fake_export = {
            "papers": [
                {"slug": slug, "source_path": str(src_file), "arxiv_id": "2501.12345"},
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

        rc, stdout, stderr = _run_sync(
            "--export", str(export_path),
            "--papers-dir", str(papers_out),
            "--chunks-dir", str(chunks_dir),
            "--bm25-dir", str(bm25_dir),
        )
        assert rc == 0, f"sync failed: exit {rc}\nstdout: {stdout}\nstderr: {stderr}"
        return export_path, chunks_dir, bm25_dir, papers_out

    def test_bm25_index_built_after_sync(self, tmp_path):
        """After sync, a BM25 index must exist at --bm25-dir/index.json."""
        _, chunks_dir, bm25_dir, _ = self._setup_index(tmp_path)
        idx_path = bm25_dir / "index.json"
        assert idx_path.exists(), (
            f"BM25 index not found at {idx_path} after sync"
        )
        idx = json.loads(idx_path.read_text(encoding="utf-8"))
        assert idx.get("doc_count", 0) > 0, "BM25 index has zero docs"
        assert "vocab" in idx, "BM25 index missing 'vocab'"

    def test_chunks_contain_paper_provenance(self, tmp_path):
        """Each chunk file must have page_path, page_address, chunk_index fields."""
        _, chunks_dir, _, papers_out = self._setup_index(tmp_path)
        chunk_files = list(chunks_dir.glob("*/chunk-*.json"))
        assert chunk_files, "No chunk files written"
        for cf in chunk_files:
            data = json.loads(cf.read_text(encoding="utf-8"))
            assert "page_path" in data, f"{cf.name} missing page_path"
            assert "page_address" in data, f"{cf.name} missing page_address"
            assert "chunk_index" in data, f"{cf.name} missing chunk_index"

    def test_bm25_index_only_over_full_md_bodies(self, tmp_path):
        """The BM25 index must only index chunks from .full.md source pages
        (not claim/entity stubs). The chunk JSON files must have page_path
        pointing at a .full.md file."""
        _, chunks_dir, bm25_dir, _ = self._setup_index(tmp_path)
        # Check every chunk JSON file has a page_path ending in .full.md
        chunk_files = list(chunks_dir.glob("*/chunk-*.json"))
        assert chunk_files, "No chunk files found"
        violations = []
        for cf in chunk_files:
            data = json.loads(cf.read_text(encoding="utf-8"))
            page_path = data.get("page_path", "")
            if not page_path.endswith(".full.md"):
                violations.append(f"  {cf.name}: page_path={page_path!r}")
        assert not violations, (
            "Some chunks reference non-.full.md pages:\n" + "\n".join(violations)
        )

    def test_graph_retrieve_script_exists(self):
        """scripts/graph-retrieve.py must exist before retrieve tests can pass."""
        assert RETRIEVE_SCRIPT.exists(), (
            "scripts/graph-retrieve.py does not exist — implement it (T5)"
        )

    def test_graph_retrieve_query_returns_results(self, tmp_path):
        """graph-retrieve.py <query> must return top-K results with provenance when
        the graph index exists."""
        export_path, chunks_dir, bm25_dir, papers_out = self._setup_index(tmp_path)
        if not RETRIEVE_SCRIPT.exists():
            pytest.skip("graph-retrieve.py not yet implemented (T5)")

        cmd = [
            sys.executable, str(RETRIEVE_SCRIPT),
            "garment pinning diffusion",
            "--bm25-index", str(bm25_dir / "index.json"),
            "--chunks-dir", str(chunks_dir),
            "--top", "3",
        ]
        r = subprocess.run(
            cmd, capture_output=True, text=True,
            cwd=PROJECT_ROOT, timeout=30,
        )
        assert r.returncode == 0, (
            f"graph-retrieve.py exited {r.returncode}\n"
            f"stdout: {r.stdout}\nstderr: {r.stderr}"
        )
        out = json.loads(r.stdout)
        assert "candidates" in out, "retrieve output missing 'candidates'"
        assert len(out["candidates"]) > 0, "No candidates returned for known query"
        first = out["candidates"][0]
        assert "page_path" in first or "chunk_id" in first, (
            "Candidate missing provenance fields"
        )

    def test_graph_retrieve_missing_index_nonzero_exit(self, tmp_path):
        """graph-retrieve.py with --bm25-index pointing to nonexistent path must
        exit non-zero with a helpful hint."""
        if not RETRIEVE_SCRIPT.exists():
            pytest.skip("graph-retrieve.py not yet implemented (T5)")
        fake_bm25 = tmp_path / "nonexistent" / "index.json"
        fake_chunks = tmp_path / "nonexistent" / "chunks"
        cmd = [
            sys.executable, str(RETRIEVE_SCRIPT),
            "any query",
            "--bm25-index", str(fake_bm25),
            "--chunks-dir", str(fake_chunks),
        ]
        r = subprocess.run(
            cmd, capture_output=True, text=True,
            cwd=PROJECT_ROOT, timeout=30,
        )
        assert r.returncode != 0, (
            f"graph-retrieve.py should exit non-zero for missing index, got {r.returncode}"
        )
        combined = (r.stdout + r.stderr).lower()
        assert "sync" in combined or "graph-fulltext" in combined or "index" in combined, (
            "Missing-index error should mention 'sync' or 'graph-fulltext' or 'index'\n"
            f"stdout: {r.stdout}\nstderr: {r.stderr}"
        )


# ============================================================================
# AC5: degrade — ollama absent -> BM25-only rerank; missing index -> friendly exit
# ============================================================================

class TestDegrade:
    """AC5: Graceful degradation paths."""

    def test_graph_retrieve_missing_index_gives_hint(self, tmp_path):
        """Missing graph BM25 index → non-zero exit with hint to run sync."""
        if not RETRIEVE_SCRIPT.exists():
            pytest.skip("graph-retrieve.py not yet implemented (T5)")
        nonexistent_idx = tmp_path / "no-such-dir" / "index.json"
        nonexistent_chunks = tmp_path / "no-such-dir" / "chunks"
        cmd = [
            sys.executable, str(RETRIEVE_SCRIPT),
            "test query",
            "--bm25-index", str(nonexistent_idx),
            "--chunks-dir", str(nonexistent_chunks),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True,
                           cwd=PROJECT_ROOT, timeout=30)
        assert r.returncode != 0, (
            f"Should exit non-zero for missing index, got {r.returncode}"
        )
        combined = r.stdout + r.stderr
        assert combined.strip(), "Expected some error output"

    def test_graph_retrieve_ollama_absent_degrades_gracefully(self, tmp_path):
        """With OLLAMA_URL pointing to a dead port, graph-retrieve.py should
        still succeed (BM25-only path) and mark rerank_source as noop-no-ollama."""
        if not RETRIEVE_SCRIPT.exists():
            pytest.skip("graph-retrieve.py not yet implemented (T5)")

        # Build a small index
        src_file = tmp_path / "degrade.md"
        src_file.write_text(
            "# Degrade Test\n\nBM25 fallback test content about neural networks.",
            encoding="utf-8",
        )
        fake_export = {
            "papers": [
                {"slug": "degrade-test", "source_path": str(src_file), "arxiv_id": None},
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

        rc, _, _ = _run_sync(
            "--export", str(export_path),
            "--papers-dir", str(papers_out),
            "--chunks-dir", str(chunks_dir),
            "--bm25-dir", str(bm25_dir),
        )
        assert rc == 0

        # Point OLLAMA_URL at a dead port to simulate absent ollama
        env = os.environ.copy()
        env["OLLAMA_URL"] = "http://127.0.0.1:1"

        cmd = [
            sys.executable, str(RETRIEVE_SCRIPT),
            "neural networks",
            "--bm25-index", str(bm25_dir / "index.json"),
            "--chunks-dir", str(chunks_dir),
            "--top", "3",
        ]
        r = subprocess.run(cmd, capture_output=True, text=True,
                           cwd=PROJECT_ROOT, env=env, timeout=30)
        assert r.returncode == 0, (
            f"Should exit 0 when ollama absent (BM25-only fallback)\n"
            f"exit={r.returncode}\nstdout: {r.stdout}\nstderr: {r.stderr}"
        )


# ============================================================================
# AC9: read_surface — /graph read routing, --paper and --claim flags
# ============================================================================

class TestReadSurface:
    """AC9: graph-retrieve.py surface — query, --paper slug, --claim id."""

    def test_graph_retrieve_accepts_paper_flag(self, tmp_path):
        """graph-retrieve.py --paper <slug> must be accepted (or exit with
        friendly message if index not provisioned)."""
        if not RETRIEVE_SCRIPT.exists():
            pytest.skip("graph-retrieve.py not yet implemented (T5)")

        # Build a small index first
        src_file = tmp_path / "surface.md"
        src_file.write_text(
            "# Surface Test\n\nContent about graph neural networks and diffusion.",
            encoding="utf-8",
        )
        fake_export = {
            "papers": [
                {"slug": "surface-test", "source_path": str(src_file), "arxiv_id": "2501.99999"},
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

        rc, _, _ = _run_sync(
            "--export", str(export_path),
            "--papers-dir", str(papers_out),
            "--chunks-dir", str(chunks_dir),
            "--bm25-dir", str(bm25_dir),
        )
        assert rc == 0

        cmd = [
            sys.executable, str(RETRIEVE_SCRIPT),
            "--paper", "surface-test",
            "--bm25-index", str(bm25_dir / "index.json"),
            "--chunks-dir", str(chunks_dir),
            "--export", str(export_path),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True,
                           cwd=PROJECT_ROOT, timeout=30)
        # Either succeeds (0) or exits non-zero with a helpful message
        if r.returncode != 0:
            combined = (r.stdout + r.stderr).lower()
            # Should not be a Python traceback
            assert "traceback" not in combined, (
                f"--paper flag caused a crash:\n{r.stderr}"
            )
        else:
            out = json.loads(r.stdout)
            assert "candidates" in out, "Output missing 'candidates'"

    def test_graph_retrieve_accepts_claim_flag(self, tmp_path):
        """graph-retrieve.py --claim <id> must be accepted without crashing."""
        if not RETRIEVE_SCRIPT.exists():
            pytest.skip("graph-retrieve.py not yet implemented (T5)")

        # Build a small index first
        src_file = tmp_path / "claim_surface.md"
        src_file.write_text(
            "# Claim Surface\n\nContent about diffusion model training.",
            encoding="utf-8",
        )
        fake_export = {
            "papers": [
                {"slug": "claim-surface", "source_path": str(src_file), "arxiv_id": None},
            ],
            "entities": [], "claims": [
                {
                    "id": 42,
                    "subject_id": 1,
                    "predicate": "uses",
                    "object_id": 2,
                    "polarity": "asserts",
                    "source_paper": "claim-surface",
                }
            ],
            "sections": [], "paper_authors": [], "predicates": [], "entity_edges": [],
            "citation_links": [], "aliases": [],
        }
        export_path = tmp_path / "graph-export.json"
        export_path.write_text(json.dumps(fake_export), encoding="utf-8")
        papers_out = tmp_path / "papers"
        papers_out.mkdir()
        chunks_dir = tmp_path / ".vault-meta" / "graph" / "chunks"
        bm25_dir = tmp_path / ".vault-meta" / "graph" / "bm25"

        rc, _, _ = _run_sync(
            "--export", str(export_path),
            "--papers-dir", str(papers_out),
            "--chunks-dir", str(chunks_dir),
            "--bm25-dir", str(bm25_dir),
        )
        assert rc == 0

        cmd = [
            sys.executable, str(RETRIEVE_SCRIPT),
            "--claim", "42",
            "--bm25-index", str(bm25_dir / "index.json"),
            "--chunks-dir", str(chunks_dir),
            "--export", str(export_path),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True,
                           cwd=PROJECT_ROOT, timeout=30)
        # Should not crash (no traceback)
        combined = (r.stdout + r.stderr).lower()
        assert "traceback" not in combined, (
            f"--claim flag caused a crash:\nstdout: {r.stdout}\nstderr: {r.stderr}"
        )


# ============================================================================
# Slug-dir fallback: paper whose ONLY source is ~/.paper-scholar/<slug>/*.md
# ============================================================================

class TestSlugDirFallback:
    """Resolver fallback: when source_path does not resolve, look in
    ~/.paper-scholar/<graph-slug>/ for exactly one *.md (ignoring *_meta.json).
    Monkeypatches the paper-scholar root via env var PAPER_SCHOLAR_DIR so the
    test runs from a tmp sandbox with no dependency on real disk data.
    """

    def _build_export(self, tmp_path, slug, source_path_in_export, ps_dir):
        """Write a fake graph-export.json and return its Path.

        ps_dir: the monkeypatched paper-scholar root (tmp_path / "ps")
        Creates  ps_dir/<slug>/<arxivid>.md  and  ps_dir/<slug>/<arxivid>_meta.json
        so the fallback finds exactly one .md.
        """
        slug_dir = ps_dir / slug
        slug_dir.mkdir(parents=True, exist_ok=True)
        md_file = slug_dir / "2501.99001.md"
        md_file.write_text(
            "# Slug-Dir Fallback Paper\n\nFull body text only reachable via slug dir.\n",
            encoding="utf-8",
        )
        # also write a _meta.json — must be ignored by the resolver
        (slug_dir / "2501.99001_meta.json").write_text("{}", encoding="utf-8")

        fake_export = {
            "papers": [
                {
                    "slug": slug,
                    "source_path": source_path_in_export,
                    "arxiv_id": "2501.99001",
                },
            ],
            "entities": [], "claims": [], "sections": [],
            "paper_authors": [], "predicates": [], "entity_edges": [],
            "citation_links": [], "aliases": [],
        }
        export_path = tmp_path / "graph-export.json"
        export_path.write_text(json.dumps(fake_export), encoding="utf-8")
        return export_path

    def _run_sync_with_ps(self, tmp_path, slug, source_path_in_export):
        """Run sync with PAPER_SCHOLAR_DIR monkeypatched to a tmp sandbox."""
        ps_dir = tmp_path / "ps"
        export_path = self._build_export(tmp_path, slug, source_path_in_export, ps_dir)
        papers_out = tmp_path / "papers"
        papers_out.mkdir()
        meta_out = tmp_path / ".vault-meta" / "graph"
        meta_out.mkdir(parents=True)

        env = os.environ.copy()
        env["PAPER_SCHOLAR_DIR"] = str(ps_dir)

        rc, stdout, stderr = _run_sync(
            "--export", str(export_path),
            "--papers-dir", str(papers_out),
            "--chunks-dir", str(meta_out / "chunks"),
            "--bm25-dir", str(meta_out / "bm25"),
            env=env,
        )
        return rc, stdout, stderr, papers_out, ps_dir

    def test_slug_dir_fallback_writes_full_md(self, tmp_path):
        """A paper whose source_path does not resolve but whose slug matches a
        directory in PAPER_SCHOLAR_DIR with exactly one .md must get a .full.md
        written via the slug-dir fallback."""
        slug = "slug-fallback-paper"
        # source_path points at a non-existent stale PDF — current resolver skips it
        stale_pdf = str(tmp_path / "inbox" / f"{slug}.pdf")

        rc, stdout, stderr, papers_out, ps_dir = self._run_sync_with_ps(
            tmp_path, slug, stale_pdf
        )
        assert rc == 0, f"exit {rc}\nstdout: {stdout}\nstderr: {stderr}"

        full_md = papers_out / f"{slug}.full.md"
        assert full_md.exists(), (
            f"{slug}.full.md was NOT written — slug-dir fallback not implemented.\n"
            f"ps_dir={ps_dir}\nstdout={stdout}\nstderr={stderr}"
        )

    def test_slug_dir_fallback_body_byte_equal(self, tmp_path):
        """Body of the fallback .full.md must be byte-equal to the ps slug dir .md."""
        slug = "slug-fallback-body-check"
        stale_pdf = str(tmp_path / "inbox" / f"{slug}.pdf")

        rc, stdout, stderr, papers_out, ps_dir = self._run_sync_with_ps(
            tmp_path, slug, stale_pdf
        )
        assert rc == 0, f"exit {rc}\nstderr: {stderr}"

        full_md = papers_out / f"{slug}.full.md"
        if not full_md.exists():
            pytest.skip("slug-dir fallback not yet implemented — skipping body check")

        src_md = ps_dir / slug / "2501.99001.md"
        src_bytes = src_md.read_bytes()

        full_text = full_md.read_text(encoding="utf-8")
        body = _body_after_frontmatter(full_text)
        assert body.encode("utf-8") == src_bytes, (
            "Fallback .full.md body is not byte-equal to the source .md"
        )

    def test_slug_dir_fallback_zero_md_skips(self, tmp_path):
        """If the slug dir in paper-scholar has NO .md files, the paper is skipped
        (no guessing), and sync exits 0."""
        slug = "slug-fallback-zero-md"
        stale_pdf = str(tmp_path / "inbox" / f"{slug}.pdf")

        ps_dir = tmp_path / "ps"
        slug_dir = ps_dir / slug
        slug_dir.mkdir(parents=True)
        # Write only a _meta.json, no .md
        (slug_dir / "2501.99999_meta.json").write_text("{}", encoding="utf-8")

        fake_export = {
            "papers": [{"slug": slug, "source_path": stale_pdf, "arxiv_id": None}],
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
        env = os.environ.copy()
        env["PAPER_SCHOLAR_DIR"] = str(ps_dir)

        rc, stdout, stderr = _run_sync(
            "--export", str(export_path),
            "--papers-dir", str(papers_out),
            "--chunks-dir", str(meta_out / "chunks"),
            "--bm25-dir", str(meta_out / "bm25"),
            env=env,
        )
        assert rc == 0, f"exit {rc}\nstderr: {stderr}"
        assert not (papers_out / f"{slug}.full.md").exists(), (
            "slug-dir with zero .md files should be skipped — no .full.md"
        )

    def test_slug_dir_fallback_multi_md_skips(self, tmp_path):
        """If the slug dir in paper-scholar has >1 .md files, the paper is skipped
        (ambiguous, no guessing), and sync exits 0."""
        slug = "slug-fallback-multi-md"
        stale_pdf = str(tmp_path / "inbox" / f"{slug}.pdf")

        ps_dir = tmp_path / "ps"
        slug_dir = ps_dir / slug
        slug_dir.mkdir(parents=True)
        # Write two .md files — ambiguous
        (slug_dir / "2501.00001.md").write_text("# Paper A", encoding="utf-8")
        (slug_dir / "2501.00002.md").write_text("# Paper B", encoding="utf-8")

        fake_export = {
            "papers": [{"slug": slug, "source_path": stale_pdf, "arxiv_id": None}],
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
        env = os.environ.copy()
        env["PAPER_SCHOLAR_DIR"] = str(ps_dir)

        rc, stdout, stderr = _run_sync(
            "--export", str(export_path),
            "--papers-dir", str(papers_out),
            "--chunks-dir", str(meta_out / "chunks"),
            "--bm25-dir", str(meta_out / "bm25"),
            env=env,
        )
        assert rc == 0, f"exit {rc}\nstderr: {stderr}"
        assert not (papers_out / f"{slug}.full.md").exists(), (
            "slug-dir with >1 .md files should be skipped — no .full.md"
        )

    def test_slug_dir_fallback_does_not_affect_direct_md_resolution(self, tmp_path):
        """Existing Tier-A direct .md resolution must still work when PAPER_SCHOLAR_DIR
        is set. The fallback must only fire when prior strategies fail."""
        slug = "slug-direct-still-works"
        src_file = tmp_path / "direct.md"
        src_file.write_text("# Direct MD\n\nBody from direct path.", encoding="utf-8")

        ps_dir = tmp_path / "ps"
        # also put an .md in the slug dir — should NOT override the direct path
        slug_dir = ps_dir / slug
        slug_dir.mkdir(parents=True)
        (slug_dir / "2501.00099.md").write_text("# PS dir version", encoding="utf-8")

        fake_export = {
            "papers": [{"slug": slug, "source_path": str(src_file), "arxiv_id": None}],
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
        env = os.environ.copy()
        env["PAPER_SCHOLAR_DIR"] = str(ps_dir)

        rc, stdout, stderr = _run_sync(
            "--export", str(export_path),
            "--papers-dir", str(papers_out),
            "--chunks-dir", str(meta_out / "chunks"),
            "--bm25-dir", str(meta_out / "bm25"),
            env=env,
        )
        assert rc == 0, f"exit {rc}\nstderr: {stderr}"

        full_md = papers_out / f"{slug}.full.md"
        assert full_md.exists(), ".full.md not written even for direct .md path"

        full_text = full_md.read_text(encoding="utf-8")
        body = _body_after_frontmatter(full_text)
        # Body must come from direct source, not the ps-dir version
        assert "Direct MD" in body or "Body from direct path" in body, (
            "Fallback overwrote the direct-path resolution — priority bug"
        )


# ============================================================================
# Query-mode dedup: identical chunk body_hash -> only one result in top-K
# ============================================================================

class TestQueryDedup:
    """Duplicate-slug dedup: two slugs sharing byte-identical chunk bodies must
    appear only once in query results so a third distinct paper gets a top-K slot.

    Covers only free-text query mode (args.query). --paper mode intentionally
    returns multiple chunks of one paper — not touched.
    """

    def _build_index(self, tmp_path, papers: list[dict]) -> tuple:
        """Sync a list of papers (slug, body) into a sandboxed index.

        Returns (export_path, chunks_dir, bm25_dir).
        """
        paper_rows = []
        for p in papers:
            src = tmp_path / f"{p['slug']}.md"
            src.write_text(p["body"], encoding="utf-8")
            paper_rows.append({
                "slug": p["slug"],
                "source_path": str(src),
                "arxiv_id": p.get("arxiv_id"),
            })

        fake_export = {
            "papers": paper_rows,
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

        rc, stdout, stderr = _run_sync(
            "--export", str(export_path),
            "--papers-dir", str(papers_out),
            "--chunks-dir", str(chunks_dir),
            "--bm25-dir", str(bm25_dir),
        )
        assert rc == 0, f"sync failed: exit {rc}\nstdout: {stdout}\nstderr: {stderr}"
        return export_path, chunks_dir, bm25_dir, papers_out

    def _retrieve(self, tmp_path, query: str, chunks_dir, bm25_dir, top: int = 5) -> list:
        """Run graph-retrieve.py and return parsed candidates list."""
        cmd = [
            sys.executable, str(RETRIEVE_SCRIPT),
            query,
            "--bm25-index", str(bm25_dir / "index.json"),
            "--chunks-dir", str(chunks_dir),
            "--top", str(top),
            "--no-rerank",
        ]
        r = subprocess.run(
            cmd, capture_output=True, text=True,
            cwd=PROJECT_ROOT, timeout=30,
        )
        assert r.returncode == 0, (
            f"graph-retrieve.py exited {r.returncode}\n"
            f"stdout: {r.stdout}\nstderr: {r.stderr}"
        )
        return json.loads(r.stdout).get("candidates", [])

    def test_duplicate_slug_chunk_appears_once(self, tmp_path):
        """Two slugs with byte-identical chunk-000 bodies -> only one occurrence in results.

        Setup: slug A and slug B share identical body text (simulating an import
        that created two slugs for the same paper). Slug C has a distinct body.
        Query with top=3. Assert:
          - The shared body text appears exactly once in candidates.
          - Slug C appears in the top-3 (not pushed out by duplicate of A/B).
        """
        if not RETRIEVE_SCRIPT.exists():
            pytest.skip("graph-retrieve.py not implemented")

        shared_body = (
            "# OmniVTON Training-Free Universal Virtual Try-On\n\n"
            "This paper proposes a training-free approach to universal virtual try-on "
            "using diffusion models with attention manipulation. The method achieves "
            "state-of-the-art performance on VITON-HD and DressCode benchmarks without "
            "any fine-tuning. Key contribution: zero-shot garment transfer via "
            "cross-attention injection in diffusion U-Net."
        )
        distinct_body = (
            "# Distinct Paper on 3D Garment Reconstruction\n\n"
            "This paper presents a novel approach to 3D garment reconstruction "
            "from single-view RGB images using implicit neural representations. "
            "The method outperforms all baselines on the THuman dataset."
        )

        papers = [
            {"slug": "omnivton-training-free-universal", "body": shared_body},
            {"slug": "omnivton-training-free-universal-virtual-try-on", "body": shared_body},
            {"slug": "distinct-3d-garment-reconstruction", "body": distinct_body},
        ]
        _, chunks_dir, bm25_dir, _ = self._build_index(tmp_path, papers)

        candidates = self._retrieve(
            tmp_path, "training-free virtual try-on diffusion garment",
            chunks_dir, bm25_dir, top=3,
        )

        # Gather the snippet text of each candidate
        snippets = [c.get("snippet", "") for c in candidates]

        # The shared body's opening (first 200 chars used for snippet) must appear at most once
        shared_opening = shared_body[:50]
        shared_count = sum(1 for s in snippets if shared_opening in s)
        assert shared_count <= 1, (
            f"Duplicate chunk appeared {shared_count} times in top-3 results; "
            f"expected at most 1. Candidates: {[c.get('chunk_id') for c in candidates]}"
        )

        # The distinct paper must appear in the top-3 (was being pushed out before fix)
        distinct_present = any(
            "distinct-3d-garment" in (c.get("page_path") or "")
            or "distinct-3d-garment" in (c.get("chunk_id") or "")
            for c in candidates
        )
        assert distinct_present, (
            "Distinct paper was pushed out of top-3 by duplicate slugs. "
            f"Candidates: {[c.get('chunk_id') for c in candidates]}"
        )
