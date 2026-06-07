"""P6 graph-propose tests — Semantic Bridge / Claude-Code Directions Report.

AC1: Dossier assembly is deterministic and grounded — every anchor entity/paper
     exists in graph.db; dossier carries the citable allow-list. (no egress)
AC2: Prompt builder injects RESEARCH_PROFILE.md facts + proposals.md exemplar +
     bridge dossier + explicit citable allow-list + the FR7 section contract
     anchors (## The bar, ## Decision matrix, ### N., Takedown, ## Ranking,
     ## Execution).
AC3: Grounding gate accepts a clean fake report (cites subset of allow-list) ->
     saves file with 'N/N citations verified'; retries dirty-then-clean and
     records retry count in footer.
AC4: Hard-fail: always-dirty fake -> after <=3 retries, non-zero exit + a
     .rejected.md artifact with unverified cites flagged, NO clean -directions.md.
AC5: Source safety + no-clobber: ~/Desktop/research/proposals.md never written;
     output under wiki/graph/proposals/; same-day rerun suffixes -2.
AC6: Graceful degradation: missing db -> exit 1 + hint; --claude-cmd absent ->
     non-zero + hint; missing RESEARCH_PROFILE.md -> non-zero + hint.
AC7: Gold anchor: VTON x diffusion-sampling bridge appears in candidate dossier
     by entity membership (not hardcoded id) — reuses build_proposals.
AC10: Wiring: commands/graph.md and skills/graph/SKILL.md reference graph-propose;
      make test-propose exists.

Fixture approach (mirrors test_graph_bridge.py):
  - Tiny self-contained fixture graph.db (built in-process, no live db needed).
  - Three faked --claude-cmd shell scripts under a tmp dir:
      (a) clean.sh — echoes a report citing ONLY allow-list slugs/entities.
      (b) dirty_then_clean.sh — first call emits a fabricated cite, second is
          clean (keyed off a retry-count file written next to the script).
      (c) always_dirty.sh — always emits a fabricated cite.

All tests are OFFLINE and deterministic — no real claude egress.
"""

import json
import os
import re
import sqlite3
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = PROJECT_ROOT / "scripts"
PROPOSE_SCRIPT = SCRIPTS / "graph-propose.py"
BRIDGE_SCRIPT = SCRIPTS / "graph-bridge.py"
PROFILE_PATH = PROJECT_ROOT / "wiki" / "graph" / "RESEARCH_PROFILE.md"

# The real proposals.md must NEVER be written by the script.
REAL_PROPOSALS_PATH = Path.home() / "Desktop" / "research" / "proposals.md"

# Output directory for generated reports.
PROPOSALS_DIR = PROJECT_ROOT / "wiki" / "graph" / "proposals"


# ---------------------------------------------------------------------------
# Fixture DB builder (mirror test_graph_bridge._build_fixture_db style)
# ---------------------------------------------------------------------------

def _build_fixture_db(path: Path) -> None:
    """Build a tiny self-contained graph db for unit testing.

    Three communities structured to produce white-space proposals:
      Community A (VTON-side): entities 1-5, "virtual try-on" themed
      Community B (Aek-side):  entities 6-10, "diffusion sampling" themed
      Community C (Unrelated): entities 11-15, "unrelated" themed

    Intra-community claim edges only (all pairs are white-space).
    Papers: vton-paper, aek-paper, other-paper.
    Allow-list from these is what the clean fake engine cites.
    """
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA foreign_keys = OFF")

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS papers (
            id INTEGER PRIMARY KEY,
            slug TEXT UNIQUE NOT NULL,
            title TEXT,
            authors TEXT,
            source_path TEXT
        );
        CREATE TABLE IF NOT EXISTS sections (
            id INTEGER PRIMARY KEY,
            paper_slug TEXT NOT NULL,
            heading TEXT,
            role TEXT,
            summary TEXT,
            order_index INTEGER
        );
        CREATE TABLE IF NOT EXISTS paper_authors (
            paper_id INTEGER NOT NULL,
            author_name TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS entities (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            super_type TEXT,
            sub_type TEXT,
            description TEXT,
            source_paper TEXT,
            is_canonical INTEGER DEFAULT 1,
            canonical_id INTEGER DEFAULT NULL,
            merge_confidence REAL DEFAULT 1.0,
            metadata TEXT
        );
        CREATE TABLE IF NOT EXISTS predicates (
            id INTEGER PRIMARY KEY,
            name TEXT UNIQUE,
            domain_super_type TEXT,
            range_super_type TEXT
        );
        CREATE TABLE IF NOT EXISTS claims (
            id INTEGER PRIMARY KEY,
            subject_entity_id INTEGER NOT NULL,
            predicate TEXT NOT NULL,
            object_entity_id INTEGER NOT NULL,
            text TEXT,
            verbatim_quote TEXT,
            claim_type TEXT DEFAULT 'result',
            polarity TEXT DEFAULT 'asserts',
            strength TEXT DEFAULT 'moderate',
            support INTEGER DEFAULT 1,
            generated_by TEXT,
            source_paper TEXT,
            section_id INTEGER,
            confidence REAL,
            status TEXT DEFAULT 'confirmed'
        );
        CREATE TABLE IF NOT EXISTS entity_edges (
            id INTEGER PRIMARY KEY,
            source_entity_id INTEGER NOT NULL,
            target_entity_id INTEGER NOT NULL,
            predicate TEXT,
            confidence REAL,
            source_paper TEXT
        );
        CREATE TABLE IF NOT EXISTS citation_links (
            id INTEGER PRIMARY KEY,
            claim_id INTEGER,
            source_paper TEXT
        );
        CREATE TABLE IF NOT EXISTS aliases (
            alias TEXT NOT NULL,
            canonical_id INTEGER NOT NULL
        );
    """)

    conn.executemany(
        "INSERT INTO papers (id, slug, title, authors, source_path) VALUES (?,?,?,?,?)",
        [
            (1, "vton-paper",  "Virtual Try-On Paper",  "Auth A", "/vton.md"),
            (2, "aek-paper",   "Aek Diffusion Paper",   "Auth B", "/aek.md"),
            (3, "other-paper", "Other Paper",           "Auth C", "/other.md"),
        ],
    )

    entities = [
        # Community A (VTON-side): 1-5
        (1,  "virtual try-on",    "method",  "vton-paper", None),
        (2,  "garment warping",   "method",  "vton-paper", None),
        (3,  "try-on network",    "model",   "vton-paper", None),
        (4,  "appearance flow",   "concept", "vton-paper", None),
        (5,  "cloth deformation", "method",  "vton-paper", None),
        # Community B (Aek-side): 6-10
        (6,  "diffusion sampling","method",  "aek-paper",  None),
        (7,  "noise optimization","concept", "aek-paper",  None),
        (8,  "distillation",      "method",  "aek-paper",  None),
        (9,  "score matching",    "method",  "aek-paper",  None),
        (10, "momentum solver",   "model",   "aek-paper",  None),
        # Community C (unrelated): 11-15
        (11, "alpha beta pruning","method",  "other-paper", None),
        (12, "tree search",       "concept", "other-paper", None),
        (13, "minimax strategy",  "method",  "other-paper", None),
        (14, "game theory",       "concept", "other-paper", None),
        (15, "utility function",  "concept", "other-paper", None),
    ]
    conn.executemany(
        "INSERT INTO entities (id, name, super_type, source_paper, canonical_id) "
        "VALUES (?,?,?,?,?)",
        entities,
    )

    # Intra-community claims only — zero cross-community claim edges.
    claims = [
        (1,  1, "enables",     2, "vton-paper",  "result"),
        (2,  2, "produces",    3, "vton-paper",  "result"),
        (3,  3, "uses",        4, "vton-paper",  "limitation"),
        (4,  4, "drives",      5, "vton-paper",  "open-question"),
        (5,  6, "uses",        7, "aek-paper",   "result"),
        (6,  7, "refines",     8, "aek-paper",   "result"),
        (7,  8, "achieves",    9, "aek-paper",   "limitation"),
        (8,  9, "accelerates",10, "aek-paper",   "open-question"),
        (9,  11,"requires",   12, "other-paper", "result"),
        (10, 12,"guides",     13, "other-paper", "result"),
        (11, 13,"applies",    14, "other-paper", "result"),
        (12, 14,"maximizes",  15, "other-paper", "result"),
    ]
    conn.executemany(
        "INSERT INTO claims (id, subject_entity_id, predicate, object_entity_id, "
        "text, verbatim_quote, claim_type, polarity, strength, support, source_paper) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        [(c[0], c[1], c[2], c[3],
          f"Claim {c[0]}", f"Quote {c[0]}", c[5], "asserts", "moderate", 1, c[4])
         for c in claims],
    )

    conn.executemany(
        "INSERT INTO entity_edges "
        "(source_entity_id, target_entity_id, predicate, confidence, source_paper) "
        "VALUES (?,?,?,?,?)",
        [
            (1, 6, "related-to", 0.8, "vton-paper"),
            (6, 1, "related-to", 0.8, "aek-paper"),
        ],
    )

    conn.commit()
    conn.close()


def _fixture_db_path(tmp_path: Path) -> Path:
    db = tmp_path / "fixture.db"
    _build_fixture_db(db)
    return db


# ---------------------------------------------------------------------------
# Fake engine shell scripts (for --claude-cmd)
# ---------------------------------------------------------------------------

def _write_clean_engine(tmp_path: Path, allow_list_slugs: list, name: str = "clean_engine.sh") -> Path:
    """Write a shell script that emits a clean (allow-list-only) report.

    The report contains the mandatory section contract and cites only
    allow-list slugs/entities so the grounding gate accepts it on first try.
    The optional name parameter lets callers write multiple distinct engines
    to the same tmp_path without collisions.
    """
    # Backtick-wrap slugs so the new grounding gate can extract and verify them.
    # The prompt now mandates backtick wrapping; the fixture must model that.
    slugs_bt = " ".join(f"`{s}`" for s in allow_list_slugs[:2]) if allow_list_slugs else "`vton-paper` `aek-paper`"
    script = tmp_path / name
    script.write_text(f"""#!/bin/bash
# Fake clean engine: cites only known allow-list slugs (backtick-wrapped).
cat <<'REPORT'
## The bar
This is the bar section.

## Decision matrix

| # | Direction | Ceiling | Odds |
|---|-----------|:-------:|:----:|
| 1 | Bridge virtual try-on with diffusion sampling | 5 | 3 |
| 2 | Rate-distortion bridge | 4 | 3 |

### 1. Virtual try-on meets diffusion sampling

**Thesis:** Bridge `vton-paper` with `aek-paper`.

**Takedown:** Hard to execute at 4h/week.

### 2. Constrained sampling for garment pinning

**Thesis:** Use `aek-paper` machinery for `vton-paper` constraints.

**Takedown:** Requires clean theorem derivation.

### 3. Rate-distortion ceiling analysis

**Thesis:** Use `other-paper` formalism.

**Takedown:** Risk of pure benchmark paper.

## Ranking
Direction 3 > 2 > 1 at solo 4h/week.

## Execution
Week 1: reproduce baseline from `aek-paper`.
Week 2: toy model using `vton-paper` setup.

---
*Citations: `vton-paper`, `aek-paper`, `other-paper`*
REPORT
""")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script


def _write_dirty_then_clean_engine(tmp_path: Path) -> Path:
    """Write a shell script that emits a dirty report on first call, clean on second.

    Tracks invocation count via a counter file next to the script.
    First call: includes FABRICATED-PAPER-9999 (not in allow-list) -> gate retries.
    Second call: cites only real slugs -> gate accepts.
    """
    counter_file = tmp_path / "dirty_counter.txt"
    script = tmp_path / "dirty_then_clean.sh"
    script.write_text(f"""#!/bin/bash
COUNTER_FILE="{counter_file}"
if [ -f "$COUNTER_FILE" ]; then
    COUNT=$(cat "$COUNTER_FILE")
else
    COUNT=0
fi
COUNT=$((COUNT + 1))
echo "$COUNT" > "$COUNTER_FILE"

if [ "$COUNT" -eq 1 ]; then
    # First call: emit dirty report (fabricated citation)
    cat <<'REPORT'
## The bar
This is the bar section.

## Decision matrix

| # | Direction | Ceiling |
|---|-----------|:-------:|
| 1 | Some direction | 5 |
| 2 | Another direction | 4 |

### 1. First direction

**Thesis:** Uses FABRICATED-PAPER-9999.

**Takedown:** Too theoretical.

### 2. Second direction

**Thesis:** Uses vton-paper and FABRICATED-ENTITY-XYZ approach.

**Takedown:** Application layer risk.

### 3. Third direction

**Thesis:** Uses aek-paper.

**Takedown:** Needs theory co-author.

## Ranking
Direction 2 > 3 > 1.

## Execution
Week 1: reproduce vton-paper baseline.
REPORT
else
    # Second call: clean report citing only real slugs (backtick-wrapped)
    cat <<'REPORT'
## The bar
This is the bar section.

## Decision matrix

| # | Direction | Ceiling |
|---|-----------|:-------:|
| 1 | Bridge | 5 |
| 2 | Rate-distortion approach | 4 |

### 1. Virtual try-on meets diffusion

**Thesis:** Bridge `vton-paper` with `aek-paper` approach.

**Takedown:** Hard at 4h/week.

### 2. Rate-distortion ceiling

**Thesis:** Formalize `vton-paper` encoder constraints.

**Takedown:** Risk of benchmark paper.

### 3. Constrained sampling

**Thesis:** Use `aek-paper` for garment pinning.

**Takedown:** Needs clean theorem.

## Ranking
Direction 3 > 2 > 1 solo.

## Execution
Week 1: reproduce `aek-paper` baseline.

---
*Citations: `vton-paper`, `aek-paper`*
REPORT
fi
""")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script


def _write_always_dirty_engine(tmp_path: Path) -> Path:
    """Write a shell script that always emits a report with a fabricated citation."""
    script = tmp_path / "always_dirty.sh"
    script.write_text("""#!/bin/bash
cat <<'REPORT'
## The bar
Always dirty.

## Decision matrix

| # | Direction | Ceiling |
|---|-----------|:-------:|
| 1 | First | 5 |
| 2 | Second | 4 |

### 1. First direction

**Thesis:** Uses FABRICATED-PAPER-9999 approach.

**Takedown:** Bad citation.

### 2. Second direction

**Thesis:** Uses ANOTHER-FAKE-XYZ framework.

**Takedown:** Also bad.

### 3. Third direction

**Thesis:** And HALLUCINATED-ENTITY-ABC.

**Takedown:** Still bad.

## Ranking
All bad.

## Execution
Cannot execute with fake papers.
REPORT
""")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script


def _run_propose(*args, env=None, timeout=60):
    """Run graph-propose.py and return (exit_code, stdout, stderr)."""
    r = subprocess.run(
        [sys.executable, str(PROPOSE_SCRIPT)] + list(args),
        capture_output=True, text=True, cwd=PROJECT_ROOT,
        timeout=timeout, env=env,
    )
    return r.returncode, r.stdout, r.stderr


# ---------------------------------------------------------------------------
# AC1: Dossier determinism and grounding
# ---------------------------------------------------------------------------

class TestDossier:
    """AC1: Dossier assembly is deterministic and grounded.

    Tests that the dossier (offline, no egress) lists only entities/papers
    that exist in the fixture db and produces a citable allow-list.
    """

    @pytest.fixture
    def fixture_db(self, tmp_path):
        return _fixture_db_path(tmp_path)

    @pytest.fixture
    def proposals_dir(self, tmp_path):
        d = tmp_path / "proposals"
        d.mkdir()
        return d

    @pytest.fixture
    def clean_engine(self, tmp_path):
        # allow-list slugs that the clean engine will cite
        return _write_clean_engine(tmp_path, ["vton-paper", "aek-paper"])

    def test_dossier_assembles_without_egress(self, fixture_db, proposals_dir, clean_engine, tmp_path):
        """graph-propose.py with --dry-run-dossier-only should assemble dossier offline."""
        rc, out, err = _run_propose(
            "--db", str(fixture_db),
            "--output-dir", str(proposals_dir),
            "--profile", str(PROFILE_PATH),
            "--claude-cmd", str(clean_engine),
            "--dry-run-dossier-only",
        )
        assert rc == 0, f"Expected exit 0, got {rc}. stderr: {err}"
        data = json.loads(out)
        assert "candidates" in data, f"No 'candidates' key in dossier output: {data}"
        assert len(data["candidates"]) > 0, "Dossier produced no candidates"

    def test_dossier_all_entities_in_db(self, fixture_db, proposals_dir, clean_engine, tmp_path):
        """Every anchor entity in the dossier must exist in fixture graph.db."""
        rc, out, err = _run_propose(
            "--db", str(fixture_db),
            "--output-dir", str(proposals_dir),
            "--profile", str(PROFILE_PATH),
            "--claude-cmd", str(clean_engine),
            "--dry-run-dossier-only",
        )
        assert rc == 0, f"Expected exit 0, got {rc}. stderr: {err}"
        data = json.loads(out)

        # Collect all entity names from dossier
        all_entity_names = set()
        for c in data["candidates"]:
            for ae in c.get("anchor_entities", []):
                all_entity_names.add(ae["name"])

        # Verify each against the fixture db
        conn = sqlite3.connect(str(fixture_db))
        db_entity_names = {row[0] for row in conn.execute("SELECT name FROM entities")}
        conn.close()

        for name in all_entity_names:
            assert name in db_entity_names, \
                f"Dossier references entity '{name}' not in graph.db"

    def test_dossier_all_papers_in_db(self, fixture_db, proposals_dir, clean_engine, tmp_path):
        """Every anchor paper slug in the dossier must exist in fixture graph.db."""
        rc, out, err = _run_propose(
            "--db", str(fixture_db),
            "--output-dir", str(proposals_dir),
            "--profile", str(PROFILE_PATH),
            "--claude-cmd", str(clean_engine),
            "--dry-run-dossier-only",
        )
        assert rc == 0, f"Expected exit 0, got {rc}. stderr: {err}"
        data = json.loads(out)

        all_paper_slugs = set()
        for c in data["candidates"]:
            for slug in c.get("anchor_papers", []):
                all_paper_slugs.add(slug)

        conn = sqlite3.connect(str(fixture_db))
        db_slugs = {row[0] for row in conn.execute("SELECT slug FROM papers")}
        conn.close()

        for slug in all_paper_slugs:
            assert slug in db_slugs, \
                f"Dossier references paper '{slug}' not in graph.db"

    def test_dossier_contains_allow_list(self, fixture_db, proposals_dir, clean_engine, tmp_path):
        """Dossier output must include an 'allow_list' key."""
        rc, out, err = _run_propose(
            "--db", str(fixture_db),
            "--output-dir", str(proposals_dir),
            "--profile", str(PROFILE_PATH),
            "--claude-cmd", str(clean_engine),
            "--dry-run-dossier-only",
        )
        assert rc == 0, f"Expected exit 0, got {rc}. stderr: {err}"
        data = json.loads(out)
        assert "allow_list" in data, "Dossier output missing 'allow_list' key"
        assert isinstance(data["allow_list"], list), "allow_list must be a list"
        assert len(data["allow_list"]) > 0, "allow_list is empty"

    def test_dossier_deterministic(self, fixture_db, proposals_dir, clean_engine, tmp_path):
        """Two identical dossier runs must produce identical JSON."""
        args = [
            "--db", str(fixture_db),
            "--output-dir", str(proposals_dir),
            "--profile", str(PROFILE_PATH),
            "--claude-cmd", str(clean_engine),
            "--dry-run-dossier-only",
        ]
        rc1, out1, _ = _run_propose(*args)
        rc2, out2, _ = _run_propose(*args)
        assert rc1 == 0 and rc2 == 0
        # Parse and compare (JSON key order may vary but content must match)
        d1 = json.loads(out1)
        d2 = json.loads(out2)
        assert d1 == d2, "Dossier is not deterministic across identical runs"


# ---------------------------------------------------------------------------
# AC2: Prompt builder
# ---------------------------------------------------------------------------

class TestPrompt:
    """AC2: Prompt builder injects profile + exemplar + dossier + allow-list +
    section contract anchors.
    """

    @pytest.fixture
    def fixture_db(self, tmp_path):
        return _fixture_db_path(tmp_path)

    @pytest.fixture
    def proposals_dir(self, tmp_path):
        d = tmp_path / "proposals"
        d.mkdir()
        return d

    @pytest.fixture
    def clean_engine(self, tmp_path):
        return _write_clean_engine(tmp_path, ["vton-paper", "aek-paper"])

    def test_prompt_contains_profile(self, fixture_db, proposals_dir, clean_engine, tmp_path):
        """Prompt must contain text from RESEARCH_PROFILE.md."""
        rc, out, err = _run_propose(
            "--db", str(fixture_db),
            "--output-dir", str(proposals_dir),
            "--profile", str(PROFILE_PATH),
            "--claude-cmd", str(clean_engine),
            "--dry-run-prompt",
        )
        assert rc == 0, f"Expected exit 0, got {rc}. stderr: {err}"
        assert "first-author" in out.lower() or "top venue" in out.lower() or "VTON" in out or "virtual try-on" in out.lower(), \
            "Prompt does not contain RESEARCH_PROFILE.md content"

    def test_prompt_contains_section_contract(self, fixture_db, proposals_dir, clean_engine, tmp_path):
        """Prompt must contain all required FR7 section contract anchors."""
        rc, out, err = _run_propose(
            "--db", str(fixture_db),
            "--output-dir", str(proposals_dir),
            "--profile", str(PROFILE_PATH),
            "--claude-cmd", str(clean_engine),
            "--dry-run-prompt",
        )
        assert rc == 0, f"Expected exit 0, got {rc}. stderr: {err}"
        required_anchors = ["## The bar", "## Decision matrix", "Takedown", "## Ranking", "## Execution"]
        for anchor in required_anchors:
            assert anchor in out, f"Prompt missing section contract anchor: '{anchor}'"

    def test_prompt_contains_allow_list(self, fixture_db, proposals_dir, clean_engine, tmp_path):
        """Prompt must include 'cite ONLY' instruction and the allow-list."""
        rc, out, err = _run_propose(
            "--db", str(fixture_db),
            "--output-dir", str(proposals_dir),
            "--profile", str(PROFILE_PATH),
            "--claude-cmd", str(clean_engine),
            "--dry-run-prompt",
        )
        assert rc == 0, f"Expected exit 0, got {rc}. stderr: {err}"
        assert "cite only" in out.lower() or "CITE ONLY" in out or "only cite" in out.lower(), \
            "Prompt must include 'cite ONLY these' instruction"
        # At least one of the known fixture slugs must appear in the prompt
        assert "vton-paper" in out or "aek-paper" in out, \
            "Prompt must include at least one fixture paper slug in allow-list"

    def test_prompt_contains_dossier(self, fixture_db, proposals_dir, clean_engine, tmp_path):
        """Prompt must include bridge dossier content."""
        rc, out, err = _run_propose(
            "--db", str(fixture_db),
            "--output-dir", str(proposals_dir),
            "--profile", str(PROFILE_PATH),
            "--claude-cmd", str(clean_engine),
            "--dry-run-prompt",
        )
        assert rc == 0, f"Expected exit 0, got {rc}. stderr: {err}"
        # Dossier should reference community/anchor info
        assert "community" in out.lower() or "anchor" in out.lower(), \
            "Prompt does not appear to contain dossier content"

    def test_prompt_has_proposals_exemplar_or_warn(self, fixture_db, proposals_dir, clean_engine, tmp_path, capsys):
        """When proposals.md exemplar is present, it must appear in prompt; when
        absent, the script must warn and proceed (not crash).
        """
        # Use a fake proposals.md path (absent) to test warn+proceed path.
        fake_exemplar = tmp_path / "missing-proposals.md"
        rc, out, err = _run_propose(
            "--db", str(fixture_db),
            "--output-dir", str(proposals_dir),
            "--profile", str(PROFILE_PATH),
            "--claude-cmd", str(clean_engine),
            "--exemplar", str(fake_exemplar),
            "--dry-run-prompt",
        )
        # Must not crash; exits 0 or proceeds to produce prompt
        assert rc == 0, f"Absent exemplar must not crash prompt builder; got rc={rc}. stderr: {err}"


# ---------------------------------------------------------------------------
# AC3: Grounding gate — clean accept and dirty-retry
# ---------------------------------------------------------------------------

class TestGrounding:
    """AC3: Grounding gate clean-accept and dirty-retry paths."""

    @pytest.fixture
    def fixture_db(self, tmp_path):
        return _fixture_db_path(tmp_path)

    @pytest.fixture
    def proposals_dir(self, tmp_path):
        d = tmp_path / "proposals"
        d.mkdir()
        return d

    def test_grounding_clean_accept(self, fixture_db, proposals_dir, tmp_path):
        """Clean engine (cites only allow-list slugs) -> exit 0 + saved -directions.md."""
        engine = _write_clean_engine(tmp_path, ["vton-paper", "aek-paper"])
        rc, out, err = _run_propose(
            "--db", str(fixture_db),
            "--output-dir", str(proposals_dir),
            "--profile", str(PROFILE_PATH),
            "--claude-cmd", str(engine),
        )
        assert rc == 0, f"Clean engine should produce exit 0, got {rc}. stderr: {err}"

        # A -directions.md file must exist in proposals_dir
        reports = list(proposals_dir.glob("*-directions.md"))
        assert len(reports) >= 1, \
            f"Expected at least one -directions.md in {proposals_dir}, found none"

    def test_grounding_clean_accept_footer(self, fixture_db, proposals_dir, tmp_path):
        """Saved report must contain 'N/N citations verified' footer."""
        engine = _write_clean_engine(tmp_path, ["vton-paper", "aek-paper"])
        rc, out, err = _run_propose(
            "--db", str(fixture_db),
            "--output-dir", str(proposals_dir),
            "--profile", str(PROFILE_PATH),
            "--claude-cmd", str(engine),
        )
        assert rc == 0, f"Clean engine should produce exit 0, got {rc}. stderr: {err}"

        reports = list(proposals_dir.glob("*-directions.md"))
        assert reports, "No -directions.md report found"
        content = reports[0].read_text(encoding="utf-8")
        assert re.search(r"\d+/\d+ citations verified", content), \
            f"Report missing 'N/N citations verified' footer. Content tail:\n{content[-500:]}"

    def test_grounding_dirty_retry_and_clean(self, fixture_db, proposals_dir, tmp_path):
        """Dirty-then-clean engine -> retries once -> exit 0 + saved clean report."""
        engine = _write_dirty_then_clean_engine(tmp_path)
        rc, out, err = _run_propose(
            "--db", str(fixture_db),
            "--output-dir", str(proposals_dir),
            "--profile", str(PROFILE_PATH),
            "--claude-cmd", str(engine),
        )
        assert rc == 0, \
            f"Dirty-then-clean engine should ultimately exit 0, got {rc}. stderr: {err}"

        reports = list(proposals_dir.glob("*-directions.md"))
        assert reports, "No clean report written after dirty-then-clean"

    def test_grounding_dirty_retry_records_retry_count(self, fixture_db, proposals_dir, tmp_path):
        """Footer in the clean report must record retry count > 0."""
        engine = _write_dirty_then_clean_engine(tmp_path)
        rc, out, err = _run_propose(
            "--db", str(fixture_db),
            "--output-dir", str(proposals_dir),
            "--profile", str(PROFILE_PATH),
            "--claude-cmd", str(engine),
        )
        assert rc == 0, f"Expected exit 0, got {rc}. stderr: {err}"

        reports = list(proposals_dir.glob("*-directions.md"))
        assert reports, "No clean report found"
        content = reports[0].read_text(encoding="utf-8")
        # Footer must include retry count (any non-zero value is fine)
        assert re.search(r"retr(y|ies)[:\s]+[1-9]", content, re.IGNORECASE) or \
               "attempt 2" in content.lower() or "retry" in content.lower(), \
            f"Footer should record retry count. Content tail:\n{content[-500:]}"


# ---------------------------------------------------------------------------
# AC4: Hard-fail + cap exhaustion
# ---------------------------------------------------------------------------

class TestCapExhausted:
    """AC4: Always-dirty engine -> non-zero exit + .rejected.md + no clean report."""

    @pytest.fixture
    def fixture_db(self, tmp_path):
        return _fixture_db_path(tmp_path)

    @pytest.fixture
    def proposals_dir(self, tmp_path):
        d = tmp_path / "proposals"
        d.mkdir()
        return d

    def test_cap_exhausted_exit_nonzero(self, fixture_db, proposals_dir, tmp_path):
        """Always-dirty engine with default retries -> non-zero exit."""
        engine = _write_always_dirty_engine(tmp_path)
        rc, out, err = _run_propose(
            "--db", str(fixture_db),
            "--output-dir", str(proposals_dir),
            "--profile", str(PROFILE_PATH),
            "--claude-cmd", str(engine),
            "--retries", "2",
        )
        assert rc != 0, f"Always-dirty engine must produce non-zero exit, got {rc}"

    def test_cap_exhausted_rejected_md_exists(self, fixture_db, proposals_dir, tmp_path):
        """Always-dirty engine -> a .rejected.md artifact must be written."""
        engine = _write_always_dirty_engine(tmp_path)
        rc, out, err = _run_propose(
            "--db", str(fixture_db),
            "--output-dir", str(proposals_dir),
            "--profile", str(PROFILE_PATH),
            "--claude-cmd", str(engine),
            "--retries", "2",
        )
        rejected = list(proposals_dir.glob("*.rejected.md"))
        assert len(rejected) >= 1, \
            f"Expected a .rejected.md artifact, found none in {proposals_dir}. stderr: {err}"

    def test_cap_exhausted_no_clean_report(self, fixture_db, proposals_dir, tmp_path):
        """Always-dirty engine -> NO clean -directions.md must be written."""
        engine = _write_always_dirty_engine(tmp_path)
        rc, out, err = _run_propose(
            "--db", str(fixture_db),
            "--output-dir", str(proposals_dir),
            "--profile", str(PROFILE_PATH),
            "--claude-cmd", str(engine),
            "--retries", "2",
        )
        clean_reports = list(proposals_dir.glob("*-directions.md"))
        assert len(clean_reports) == 0, \
            f"No clean -directions.md should exist after cap exhaustion, found: {clean_reports}"

    def test_cap_exhausted_flagged_cites_in_rejected(self, fixture_db, proposals_dir, tmp_path):
        """The .rejected.md must flag the unverified citations inline."""
        engine = _write_always_dirty_engine(tmp_path)
        rc, out, err = _run_propose(
            "--db", str(fixture_db),
            "--output-dir", str(proposals_dir),
            "--profile", str(PROFILE_PATH),
            "--claude-cmd", str(engine),
            "--retries", "2",
        )
        rejected = list(proposals_dir.glob("*.rejected.md"))
        assert rejected, "No .rejected.md found"
        content = rejected[0].read_text(encoding="utf-8")
        # The content must mention at least one "FABRICATED" or "unverified" tag
        assert "FABRICATED" in content or "unverified" in content.lower() or \
               "HALLUCINATED" in content or "not in graph" in content.lower(), \
            f".rejected.md must flag unverified cites. Content tail:\n{content[-500:]}"


# ---------------------------------------------------------------------------
# AC5: Source safety and no-clobber
# ---------------------------------------------------------------------------

class TestNoClobber:
    """AC5: proposals.md never written; output in wiki/graph/proposals/;
    same-day rerun suffixes -2.
    """

    @pytest.fixture
    def fixture_db(self, tmp_path):
        return _fixture_db_path(tmp_path)

    @pytest.fixture
    def proposals_dir(self, tmp_path):
        d = tmp_path / "proposals"
        d.mkdir()
        return d

    @pytest.fixture
    def clean_engine(self, tmp_path):
        return _write_clean_engine(tmp_path, ["vton-paper", "aek-paper"])

    def test_no_clobber_real_proposals_never_written(self, fixture_db, proposals_dir, clean_engine, tmp_path):
        """~/Desktop/research/proposals.md must never be written by the script."""
        before_mtime = REAL_PROPOSALS_PATH.stat().st_mtime if REAL_PROPOSALS_PATH.exists() else None
        rc, out, err = _run_propose(
            "--db", str(fixture_db),
            "--output-dir", str(proposals_dir),
            "--profile", str(PROFILE_PATH),
            "--claude-cmd", str(clean_engine),
        )
        after_mtime = REAL_PROPOSALS_PATH.stat().st_mtime if REAL_PROPOSALS_PATH.exists() else None
        assert before_mtime == after_mtime, \
            "~/Desktop/research/proposals.md was modified — this is a safety violation"

    def test_no_clobber_output_under_proposals_dir(self, fixture_db, proposals_dir, clean_engine, tmp_path):
        """Report must land under the --output-dir."""
        rc, out, err = _run_propose(
            "--db", str(fixture_db),
            "--output-dir", str(proposals_dir),
            "--profile", str(PROFILE_PATH),
            "--claude-cmd", str(clean_engine),
        )
        assert rc == 0, f"Expected exit 0, got {rc}. stderr: {err}"
        reports = list(proposals_dir.glob("*-directions.md"))
        assert len(reports) >= 1, f"No report in {proposals_dir}"

    def test_no_clobber_same_day_suffix(self, fixture_db, proposals_dir, tmp_path):
        """A second same-day run must produce a -2 suffixed file, not clobber."""
        engine1 = _write_clean_engine(tmp_path, ["vton-paper", "aek-paper"], "engine1.sh")
        engine2 = _write_clean_engine(tmp_path, ["vton-paper"], "engine2.sh")

        rc1, _, err1 = _run_propose(
            "--db", str(fixture_db),
            "--output-dir", str(proposals_dir),
            "--profile", str(PROFILE_PATH),
            "--claude-cmd", str(engine1),
        )
        assert rc1 == 0, f"First run failed: {err1}"

        rc2, out2, err2 = _run_propose(
            "--db", str(fixture_db),
            "--output-dir", str(proposals_dir),
            "--profile", str(PROFILE_PATH),
            "--claude-cmd", str(engine2),
        )
        assert rc2 == 0, f"Second run failed (rc={rc2}): {err2}"

        # Glob for all direction reports (including suffixed ones like -directions-2.md)
        reports = sorted(proposals_dir.glob("*directions*.md"))
        # Filter out .rejected.md
        reports = [r for r in reports if "rejected" not in r.name]
        assert len(reports) >= 2, \
            f"Expected >=2 reports after two runs, found {[r.name for r in reports]}"

        # The second report must have a -2 suffix (or similar non-clobber suffix)
        names = [r.name for r in reports]
        has_suffix = any("-2" in n for n in names)
        assert has_suffix, f"No -2 suffixed report found: {names}"


# ---------------------------------------------------------------------------
# AC6: Graceful degradation
# ---------------------------------------------------------------------------

class TestDegrade:
    """AC6: Missing db, missing --claude-cmd binary, missing RESEARCH_PROFILE.md."""

    @pytest.fixture
    def proposals_dir(self, tmp_path):
        d = tmp_path / "proposals"
        d.mkdir()
        return d

    @pytest.fixture
    def fixture_db(self, tmp_path):
        return _fixture_db_path(tmp_path)

    @pytest.fixture
    def clean_engine(self, tmp_path):
        return _write_clean_engine(tmp_path, ["vton-paper", "aek-paper"])

    def test_degrade_missing_db(self, proposals_dir, clean_engine, tmp_path):
        """Missing db -> exit 1 + build hint in stderr."""
        rc, out, err = _run_propose(
            "--db", str(tmp_path / "nonexistent.db"),
            "--output-dir", str(proposals_dir),
            "--profile", str(PROFILE_PATH),
            "--claude-cmd", str(clean_engine),
        )
        assert rc != 0, f"Missing db must produce non-zero exit, got {rc}"
        hint_words = ["graph-build", "build", "run"]
        assert any(w in err.lower() for w in hint_words), \
            f"Missing db must emit a build hint in stderr. Got: {err}"

    def test_degrade_absent_claude_cmd(self, fixture_db, proposals_dir, tmp_path):
        """Non-existent --claude-cmd binary -> non-zero exit, no file written."""
        rc, out, err = _run_propose(
            "--db", str(fixture_db),
            "--output-dir", str(proposals_dir),
            "--profile", str(PROFILE_PATH),
            "--claude-cmd", str(tmp_path / "nonexistent-claude"),
        )
        assert rc != 0, f"Absent --claude-cmd must produce non-zero exit, got {rc}"
        # No clean report should be written
        clean_reports = list(proposals_dir.glob("*-directions.md"))
        assert len(clean_reports) == 0, \
            f"No report should be written when --claude-cmd is absent: {[r.name for r in clean_reports]}"

    def test_degrade_missing_profile(self, fixture_db, proposals_dir, clean_engine, tmp_path):
        """Missing RESEARCH_PROFILE.md -> non-zero exit + create-it hint."""
        missing_profile = tmp_path / "missing-profile.md"
        rc, out, err = _run_propose(
            "--db", str(fixture_db),
            "--output-dir", str(proposals_dir),
            "--profile", str(missing_profile),
            "--claude-cmd", str(clean_engine),
        )
        assert rc != 0, f"Missing profile must produce non-zero exit, got {rc}"
        all_output = out + err
        assert "profile" in all_output.lower() or "RESEARCH_PROFILE" in all_output, \
            f"Missing profile must emit a hint. Got stderr: {err}"

    def test_degrade_missing_exemplar_warns_and_proceeds(self, fixture_db, proposals_dir, clean_engine, tmp_path):
        """Missing proposals.md exemplar -> warn + proceed (exit 0, report saved)."""
        missing_exemplar = tmp_path / "no-proposals.md"
        rc, out, err = _run_propose(
            "--db", str(fixture_db),
            "--output-dir", str(proposals_dir),
            "--profile", str(PROFILE_PATH),
            "--claude-cmd", str(clean_engine),
            "--exemplar", str(missing_exemplar),
        )
        assert rc == 0, \
            f"Missing exemplar should warn but proceed (exit 0), got {rc}. stderr: {err}"
        # A warning about missing exemplar must appear somewhere
        all_output = out + err
        assert "exemplar" in all_output.lower() or "proposals.md" in all_output.lower() or \
               "warn" in all_output.lower() or "missing" in all_output.lower(), \
            "Missing exemplar must emit a warning"


# ---------------------------------------------------------------------------
# AC7: Gold anchor — VTON x diffusion-sampling bridge in candidate dossier
# ---------------------------------------------------------------------------

class TestGoldAnchor:
    """AC7: The VTON x diffusion-sampling bridge appears in candidates
    by entity membership (not hardcoded id).
    """

    @pytest.fixture
    def fixture_db(self, tmp_path):
        return _fixture_db_path(tmp_path)

    @pytest.fixture
    def proposals_dir(self, tmp_path):
        d = tmp_path / "proposals"
        d.mkdir()
        return d

    @pytest.fixture
    def clean_engine(self, tmp_path):
        return _write_clean_engine(tmp_path, ["vton-paper", "aek-paper"])

    def test_gold_anchor_vton_diffusion_in_dossier(self, fixture_db, proposals_dir, clean_engine, tmp_path):
        """The fixture db has VTON and diffusion-sampling communities with zero
        claim-edges between them; build_proposals must surface a candidate where
        one side has a VTON-keyword entity and the other has a diffusion-keyword entity.
        """
        rc, out, err = _run_propose(
            "--db", str(fixture_db),
            "--output-dir", str(proposals_dir),
            "--profile", str(PROFILE_PATH),
            "--claude-cmd", str(clean_engine),
            "--dry-run-dossier-only",
        )
        assert rc == 0, f"Expected exit 0, got {rc}. stderr: {err}"
        data = json.loads(out)

        VTON_KEYWORDS = {"virtual try-on", "garment", "vton", "try-on", "cloth"}
        DIFFUSION_KEYWORDS = {"diffusion", "sampling", "noise", "distillation", "momentum"}

        found_gold = False
        for cand in data["candidates"]:
            entity_names = {ae["name"].lower() for ae in cand.get("anchor_entities", [])}
            comm_a_members = {m.lower() for m in cand.get("community_a", {}).get("members", [])}
            comm_b_members = {m.lower() for m in cand.get("community_b", {}).get("members", [])}
            all_names = entity_names | comm_a_members | comm_b_members

            has_vton = any(kw in name for kw in VTON_KEYWORDS for name in all_names)
            has_diffusion = any(kw in name for kw in DIFFUSION_KEYWORDS for name in all_names)
            if has_vton and has_diffusion:
                found_gold = True
                break

        assert found_gold, \
            "Gold anchor (VTON x diffusion-sampling) not found in candidate dossier " \
            f"by entity membership. Candidates: {json.dumps(data['candidates'], indent=2)[:500]}"


# ---------------------------------------------------------------------------
# AC10: Wiring — commands/graph.md, SKILL.md, Makefile test-propose
# ---------------------------------------------------------------------------

class TestWiring:
    """AC10: commands/graph.md and SKILL.md reference graph-propose.py;
    make test-propose target exists.
    """

    def test_wiring_commands_graph_md(self):
        """commands/graph.md must reference graph-propose.py or /graph propose."""
        p = PROJECT_ROOT / "commands" / "graph.md"
        assert p.exists(), f"commands/graph.md not found at {p}"
        content = p.read_text(encoding="utf-8")
        assert "graph-propose" in content, \
            "commands/graph.md must reference graph-propose"

    def test_wiring_skill_md(self):
        """skills/graph/SKILL.md must reference graph-propose.py."""
        p = PROJECT_ROOT / "skills" / "graph" / "SKILL.md"
        assert p.exists(), f"skills/graph/SKILL.md not found at {p}"
        content = p.read_text(encoding="utf-8")
        assert "graph-propose" in content, \
            "skills/graph/SKILL.md must reference graph-propose"

    def test_wiring_makefile_test_propose(self):
        """Makefile must contain a test-propose target."""
        p = PROJECT_ROOT / "Makefile"
        assert p.exists()
        content = p.read_text(encoding="utf-8")
        assert re.search(r"^test-propose:", content, re.MULTILINE), \
            "Makefile missing 'test-propose:' target"


# ---------------------------------------------------------------------------
# BR1 FIX: Vacuous-pass guard (fix #1) and prose-citation detection (fix #2)
# ---------------------------------------------------------------------------

def _write_zero_citation_engine(tmp_path: Path) -> Path:
    """Write a shell script that emits a structurally valid report with NO citation tokens.

    This simulates a model that discusses papers only in un-backticked, ordinary
    prose with no ALL-CAPS tokens either — so _extract_citations() returns empty.
    The gate must FAIL this (vacuous-pass guard), not emit '0/0 verified ✓'.
    """
    script = tmp_path / "zero_citation_engine.sh"
    script.write_text("""#!/bin/bash
cat <<'REPORT'
## The bar
Three uncomfortable truths: the bar is very high.

## Decision matrix

| # | Direction | Ceiling | Odds |
|---|-----------|:-------:|:----:|
| 1 | Garment Diffusion Transfer | 5 | 2 |
| 2 | Try-on network integration | 4 | 3 |

### 1. Garment Diffusion Transfer

**Thesis:** This direction builds on recent garment transfer work. The Garment
Diffusion Transfer paper showed promising results. Wang et al. 2024 extended this.

**Takedown:** Fundamentally hard without theory muscle.

### 2. Try-on network integration

**Thesis:** The phantom-tryon-net approach is interesting.

**Takedown:** Scoop risk is high.

### 3. Constrained sampling bridge

**Thesis:** Connect diffusion sampling with try-on constraints.

**Takedown:** Requires clean theorem derivation.

## Ranking
Direction 3 > 2 > 1 at solo 4h/week.

## Execution
Week 1: reproduce the baseline from the Garment Diffusion Transfer paper.
REPORT
""")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script


def _write_adversarial_engine(tmp_path: Path) -> Path:
    """Write an adversarial engine that cites one real backticked slug + three prose fakes.

    Real slug (backticked, in allow-list): `vton-paper`
    Fabrications:
      - 'the Garment Diffusion Transfer paper'  (Title-Case + 'paper')
      - 'Wang et al. 2024'                      (author-year)
      - bare prose 'phantom-tryon-net'           (without backticks, not in graph.db)

    The gate must flag the three fakes, retry, and on persistence exit non-zero
    with a .rejected.md.  This engine always emits the same adversarial output.
    """
    script = tmp_path / "adversarial_engine.sh"
    script.write_text("""#!/bin/bash
cat <<'REPORT'
## The bar
The bar is steep. Three truths apply here.

## Decision matrix

| # | Direction | Ceiling | Odds |
|---|-----------|:-------:|:----:|
| 1 | Diffusion bridge | 5 | 2 |
| 2 | Constrained sampling | 4 | 3 |

### 1. Garment Diffusion Transfer approach

**Thesis:** Based on `vton-paper` methodology, this extends the Garment Diffusion
Transfer paper. Wang et al. 2024 showed that phantom-tryon-net can be adapted for
this purpose.

**Takedown:** Too dependent on unavailable data.

### 2. Constrained diffusion sampling

**Thesis:** Extend `vton-paper` constraints.

**Takedown:** Needs theorem co-author.

### 3. Rate-distortion bridge

**Thesis:** Combine approaches from `vton-paper`.

**Takedown:** Risk of pure benchmark.

## Ranking
Direction 3 > 2 > 1 at solo 4h/week.

## Execution
Week 1: reproduce `vton-paper` baseline.
REPORT
""")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script


class TestVacuousPassGuard:
    """BR1 fix #1: A non-empty report with zero extracted citations must FAIL.

    The grounding gate must never emit '0/0 citations verified' as a clean pass
    for a non-empty report — that is theater, not a guard.
    """

    @pytest.fixture
    def fixture_db(self, tmp_path):
        return _fixture_db_path(tmp_path)

    @pytest.fixture
    def proposals_dir(self, tmp_path):
        d = tmp_path / "proposals"
        d.mkdir()
        return d

    def test_zero_citation_report_fails_nonzero_exit(self, fixture_db, proposals_dir, tmp_path):
        """A report with zero extractable citations (none backticked, none ALL-CAPS)
        must produce a non-zero exit — not a clean save — vacuous-pass guard.
        """
        engine = _write_zero_citation_engine(tmp_path)
        rc, out, err = _run_propose(
            "--db", str(fixture_db),
            "--output-dir", str(proposals_dir),
            "--profile", str(PROFILE_PATH),
            "--claude-cmd", str(engine),
            "--retries", "1",
        )
        assert rc != 0, (
            f"A zero-citation non-empty report must fail (non-zero exit), got rc={rc}. "
            f"This is the vacuous-pass guard: '0/0 verified' is NOT a clean pass. stderr: {err}"
        )

    def test_zero_citation_report_no_clean_save(self, fixture_db, proposals_dir, tmp_path):
        """Zero-citation non-empty report: no clean -directions.md should be written."""
        engine = _write_zero_citation_engine(tmp_path)
        _run_propose(
            "--db", str(fixture_db),
            "--output-dir", str(proposals_dir),
            "--profile", str(PROFILE_PATH),
            "--claude-cmd", str(engine),
            "--retries", "1",
        )
        clean_reports = list(proposals_dir.glob("*-directions.md"))
        assert len(clean_reports) == 0, (
            f"No clean report should be saved for a zero-citation report, found: "
            f"{[r.name for r in clean_reports]}"
        )

    def test_zero_citation_report_writes_rejected(self, fixture_db, proposals_dir, tmp_path):
        """Zero-citation non-empty report: a .rejected.md must be written."""
        engine = _write_zero_citation_engine(tmp_path)
        _run_propose(
            "--db", str(fixture_db),
            "--output-dir", str(proposals_dir),
            "--profile", str(PROFILE_PATH),
            "--claude-cmd", str(engine),
            "--retries", "1",
        )
        rejected = list(proposals_dir.glob("*.rejected.md"))
        assert len(rejected) >= 1, (
            f"Expected a .rejected.md for zero-citation report, found none in {proposals_dir}"
        )


class TestProseAdversarialCitations:
    """BR1 fix #2: Prose-shaped citations (author-year, Title-Case+paper) must be detected.

    An adversarial report that cites one real backticked slug PLUS three fabrications
    in ordinary prose must trigger retries and ultimately exit non-zero with .rejected.md.
    """

    @pytest.fixture
    def fixture_db(self, tmp_path):
        return _fixture_db_path(tmp_path)

    @pytest.fixture
    def proposals_dir(self, tmp_path):
        d = tmp_path / "proposals"
        d.mkdir()
        return d

    def test_adversarial_prose_cites_flagged_rejected(self, fixture_db, proposals_dir, tmp_path):
        """Adversarial report: one real backticked slug + 3 prose fakes -> non-zero exit
        and .rejected.md, NOT a clean save.
        """
        engine = _write_adversarial_engine(tmp_path)
        rc, out, err = _run_propose(
            "--db", str(fixture_db),
            "--output-dir", str(proposals_dir),
            "--profile", str(PROFILE_PATH),
            "--claude-cmd", str(engine),
            "--retries", "2",
        )
        assert rc != 0, (
            f"Adversarial engine with prose-hallucinations must fail (non-zero exit), "
            f"got rc={rc}. The prose-citation detector is required. stderr: {err}"
        )

    def test_adversarial_prose_no_clean_save(self, fixture_db, proposals_dir, tmp_path):
        """Adversarial report: no clean -directions.md must be written."""
        engine = _write_adversarial_engine(tmp_path)
        _run_propose(
            "--db", str(fixture_db),
            "--output-dir", str(proposals_dir),
            "--profile", str(PROFILE_PATH),
            "--claude-cmd", str(engine),
            "--retries", "2",
        )
        clean_reports = list(proposals_dir.glob("*-directions.md"))
        assert len(clean_reports) == 0, (
            f"No clean report should be saved for adversarial prose engine, found: "
            f"{[r.name for r in clean_reports]}"
        )

    def test_adversarial_prose_writes_rejected_md(self, fixture_db, proposals_dir, tmp_path):
        """Adversarial report: a .rejected.md artifact must be written."""
        engine = _write_adversarial_engine(tmp_path)
        _run_propose(
            "--db", str(fixture_db),
            "--output-dir", str(proposals_dir),
            "--profile", str(PROFILE_PATH),
            "--claude-cmd", str(engine),
            "--retries", "2",
        )
        rejected = list(proposals_dir.glob("*.rejected.md"))
        assert len(rejected) >= 1, (
            f"Expected a .rejected.md after adversarial prose engine, found none in {proposals_dir}"
        )

    def test_adversarial_author_year_detected(self, tmp_path):
        """Unit test: 'Wang et al. 2024' is extracted by _extract_citations."""
        import importlib.util as _ilu
        spec = _ilu.spec_from_file_location("graph_propose", str(PROPOSE_SCRIPT))
        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        text = "Based on Wang et al. 2024 findings, the method works well."
        cites = mod._extract_citations(text)
        assert any("wang" in c.lower() or "2024" in c for c in cites), (
            f"'Wang et al. 2024' must be extracted as a citation. Got: {cites}"
        )

    def test_adversarial_title_case_paper_detected(self, tmp_path):
        """Unit test: 'the Garment Diffusion Transfer paper' is extracted."""
        import importlib.util as _ilu
        spec = _ilu.spec_from_file_location("graph_propose", str(PROPOSE_SCRIPT))
        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        text = "Building on the Garment Diffusion Transfer paper approach."
        cites = mod._extract_citations(text)
        assert any("garment" in c.lower() or "diffusion transfer" in c.lower() or "garment diffusion" in c.lower() for c in cites), (
            f"'Garment Diffusion Transfer paper' must be extracted. Got: {cites}"
        )

    def test_section_headers_not_flagged(self, tmp_path):
        """Unit test: section headers like '## The bar', '## Decision matrix', '## Ranking',
        '## Execution' must NOT be extracted as citations — no false positives.
        """
        import importlib.util as _ilu
        spec = _ilu.spec_from_file_location("graph_propose", str(PROPOSE_SCRIPT))
        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        text = """## The bar
This is analysis.

## Decision matrix

| # | Direction |

## Ranking
Direction 3 > 2.

## Execution
Week 1.
"""
        cites = mod._extract_citations(text)
        section_false_positives = {c for c in cites if c.lower() in {
            "the bar", "decision matrix", "ranking", "execution",
            "the", "bar", "decision", "matrix",
        }}
        assert len(section_false_positives) == 0, (
            f"Section headers must not be extracted as citations. Got FPs: {section_false_positives}"
        )
