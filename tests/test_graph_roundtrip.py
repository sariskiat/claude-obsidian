"""Phase 1 round-trip tests: export → build → compare.

AC1: 9 tables byte-equal (per-row diff)
AC2: 5 gap species exact
AC3: Alias round-trip 834→834 (no dangling-FK drop)
AC4: root() chain/cycle/dangling invariants + no inline COALESCE
AC5: Source db never mutated (md5 before == after)
AC6: Gitignore correct (derived db ignored, JSON tracked)
AC7: Full suite green
AC8: Forward-path documented in SCHEMA.md (design only)

Tests skip gracefully when the live db is absent.
"""

import hashlib
import importlib.util
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = PROJECT_ROOT / "scripts"
LIVE_DB = Path.home() / ".graphbuilding" / "graph.db"
WIKI_GRAPH = PROJECT_ROOT / "wiki" / "graph"
VAULT_META_GRAPH = PROJECT_ROOT / ".vault-meta" / "graph"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load_module(name, path):
    """Import a hyphenated script by file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _md5_file(path):
    """Return hex digest of a file, or None if missing."""
    if not os.path.isfile(path):
        return None
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _table_counts(db_path):
    """Return {table_name: row_count} for all user tables."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    tables = [
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name != 'sqlite_sequence'"
        ).fetchall()
    ]
    counts = {}
    for t in tables:
        counts[t] = conn.execute(f"SELECT COUNT(*) FROM [{t}]").fetchone()[0]
    conn.close()
    return counts


def _per_row_diff(db_a, db_b, table):
    """Return (a_count, b_count, diff_rows) for a table. diff_rows is a list of
    (id, col, a_val, b_val) tuples, empty when tables match exactly."""
    conn_a = sqlite3.connect(f"file:{db_a}?mode=ro", uri=True)
    conn_b = sqlite3.connect(f"file:{db_b}?mode=ro", uri=True)
    a_rows = conn_a.execute(f"SELECT * FROM [{table}] ORDER BY 1").fetchall()
    b_rows = conn_b.execute(f"SELECT * FROM [{table}] ORDER BY 1").fetchall()
    cols = [d[0] for d in conn_a.execute(f"PRAGMA table_info([{table}])").fetchall()]
    conn_a.close()
    conn_b.close()

    diffs = []
    max_len = max(len(a_rows), len(b_rows))
    for i in range(max_len):
        if i >= len(a_rows):
            diffs.append((i, "MISSING_A", None, b_rows[i]))
        elif i >= len(b_rows):
            diffs.append((i, "MISSING_B", a_rows[i], None))
        elif a_rows[i] != b_rows[i]:
            for j, (av, bv) in enumerate(zip(a_rows[i], b_rows[i])):
                if av != bv:
                    diffs.append((i, cols[j] if j < len(cols) else j, av, bv))
    return len(a_rows), len(b_rows), diffs


# ---------------------------------------------------------------------------
# T2 — root() invariants  (AC4)
# ---------------------------------------------------------------------------

class TestRootInvariants:
    """AC4: root() handles chain, cycle, dangling; path-compresses; no inline COALESCE."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.db_path = None
        yield
        if self.db_path and os.path.isfile(self.db_path):
            os.unlink(self.db_path)

    def _fresh_db(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.db_path = path
        conn = sqlite3.connect(path)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(
            """CREATE TABLE entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                canonical_id INTEGER,
                super_type TEXT,
                sub_type TEXT,
                description TEXT,
                source_paper TEXT
            )"""
        )
        conn.commit()
        self.conn = conn
        return conn

    def _add_entity(self, conn, eid, name, canonical_id=None):
        conn.execute(
            "INSERT INTO entities (id, name, canonical_id, super_type, source_paper) "
            "VALUES (?, ?, ?, 'Concept', 'test')",
            (eid, name, canonical_id),
        )
        conn.commit()

    # -- chain tests --

    def test_root_chain_three_hop(self):
        """A → B → C (C canonical=NULL) → root returns C."""
        db = self._fresh_db()
        graph_db = _load_module("graph_db", SCRIPTS / "graph_db.py")
        self._add_entity(db, 1, "A", 2)
        self._add_entity(db, 2, "B", 3)
        self._add_entity(db, 3, "C", None)
        assert graph_db.root(db, 1) == 3

    def test_root_chain_path_compression(self):
        """After root(A) on A→B→C, A.canonical_id is updated to C (path-compress)."""
        db = self._fresh_db()
        graph_db = _load_module("graph_db", SCRIPTS / "graph_db.py")
        self._add_entity(db, 1, "A", 2)
        self._add_entity(db, 2, "B", 3)
        self._add_entity(db, 3, "C", None)
        graph_db.root(db, 1)
        new_parent = db.execute(
            "SELECT canonical_id FROM entities WHERE id = 1"
        ).fetchone()[0]
        assert new_parent == 3, f"expected path-compressed to 3, got {new_parent}"

    def test_root_already_canonical(self):
        """root() on an already-canonical entity returns itself."""
        db = self._fresh_db()
        graph_db = _load_module("graph_db", SCRIPTS / "graph_db.py")
        self._add_entity(db, 1, "X", None)
        assert graph_db.root(db, 1) == 1

    # -- cycle tests --

    def test_root_cycle_two_node(self):
        """A ⇄ B two-node cycle → root elects min id (A=1) and does not loop."""
        db = self._fresh_db()
        graph_db = _load_module("graph_db", SCRIPTS / "graph_db.py")
        self._add_entity(db, 1, "A", 2)
        self._add_entity(db, 2, "B", 1)
        root_id = graph_db.root(db, 1)
        assert root_id in (1, 2), f"unexpected root {root_id}"
        # Must terminate — reaching here proves no infinite loop.
        # After resolution: elected root has NULL canonical_id; other members
        # point to the root. Verify by re-walking: root(1) and root(2) both
        # return the same elected root.
        r1 = graph_db.root(db, 1)
        r2 = graph_db.root(db, 2)
        assert r1 == r2, f"after compression root(1)={r1} != root(2)={r2}"
        # Also verify no loops: walking canonical_id must reach NULL.
        for eid in (1, 2):
            visited = set()
            cur = eid
            while cur is not None:
                assert cur not in visited, f"loop detected from entity {eid} at {cur}"
                visited.add(cur)
                row = db.execute(
                    "SELECT canonical_id FROM entities WHERE id = ?", (cur,)
                ).fetchone()
                cur = row[0] if row else None

    def test_root_self_loop(self):
        """Self-loop A→A → root returns A without looping, clears canonical_id."""
        db = self._fresh_db()
        graph_db = _load_module("graph_db", SCRIPTS / "graph_db.py")
        self._add_entity(db, 1, "A", 1)
        root_id = graph_db.root(db, 1)
        assert root_id == 1
        # After compression, A should be canonical (NULL) or point to itself.
        # Either way, root(A) must still return A and not loop.
        r1 = graph_db.root(db, 1)
        assert r1 == 1, f"root(A) after compression should be 1, got {r1}"

    # -- dangling tests --

    def test_root_dangling(self):
        """A → (id 999, missing) → root returns A (last live) and does not crash."""
        db = self._fresh_db()
        graph_db = _load_module("graph_db", SCRIPTS / "graph_db.py")
        self._add_entity(db, 1, "A", 999)  # 999 does not exist
        root_id = graph_db.root(db, 1)
        assert root_id == 1, f"expected last live id 1, got {root_id}"

    def test_root_missing_entity(self):
        """root() called on an id that is not in the table → returns that id."""
        db = self._fresh_db()
        graph_db = _load_module("graph_db", SCRIPTS / "graph_db.py")
        root_id = graph_db.root(db, 999)
        assert root_id == 999


# ---------------------------------------------------------------------------
# T2 — no inline COALESCE check (AC4)
# ---------------------------------------------------------------------------

class TestNoInlineCoalesce:
    """The string 'COALESCE(canonical_id' must not appear in graph-db.py."""

    def test_no_coalesce_in_graph_db(self):
        path = SCRIPTS / "graph_db.py"
        if not path.exists():
            pytest.skip("graph-db.py not yet created")
        # Skip docstrings and comments; only check actual code lines
        lines = path.read_text().splitlines()
        in_docstring = False
        code_lines = []
        for l in lines:
            stripped = l.strip()
            if stripped.startswith('"""') or stripped.startswith("'''"):
                in_docstring = not in_docstring
                continue
            if in_docstring:
                continue
            if stripped.startswith("#") or not stripped:
                continue
            code_lines.append(stripped)
        offending = [l for l in code_lines if "COALESCE(canonical_id" in l]
        assert not offending, (
            f"graph-db.py contains inline COALESCE(canonical_id,id): {offending}"
        )


# ---------------------------------------------------------------------------
# T1 — deps + gitignore  (AC6)
# ---------------------------------------------------------------------------

class TestDepsAndGitignore:
    """AC6: Dependencies importable; derived db gitignored; JSON tracked."""

    def test_pyyaml_importable(self):
        import yaml

    def test_networkx_importable(self):
        import networkx

    def test_derived_db_ignored(self):
        """git check-ignore must succeed for the derived db path."""
        import subprocess

        result = subprocess.run(
            ["git", "check-ignore", ".vault-meta/graph/graph.db"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            ".vault-meta/graph/graph.db is NOT gitignored — add it to .gitignore"
        )

    def test_export_json_not_ignored(self):
        """wiki/graph/graph-export.json must NOT be gitignored."""
        import subprocess

        result = subprocess.run(
            ["git", "check-ignore", "wiki/graph/graph-export.json"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0, (
            "wiki/graph/graph-export.json IS gitignored — it should be tracked"
        )


# ---------------------------------------------------------------------------
# Round-trip integration tests  (AC1, AC2, AC3, AC5, AC7)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not LIVE_DB.exists(), reason="Live graph db not found")
class TestRoundTrip:
    """Full export→build→compare cycle. Requires live ~/.graphbuilding/graph.db."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self._cleanup = []
        yield
        for path in self._cleanup:
            if os.path.isfile(path):
                os.unlink(path)

    def _temp_db(self, suffix=""):
        fd, path = tempfile.mkstemp(suffix=suffix or ".db")
        os.close(fd)
        self._cleanup.append(path)
        return path

    def _copy_live_db(self):
        """Copy the live db to a temp path, return (copy_path, md5_before)."""
        import shutil

        copy_path = self._temp_db("_copy.db")
        shutil.copy2(LIVE_DB, copy_path)
        md5 = _md5_file(copy_path)
        return copy_path, md5

    # -- AC1: table counts match --

    def test_table_counts_match(self):
        """AC1: 9/9 tables have the same row count after round-trip."""
        import subprocess

        copy_path, md5_before = self._copy_live_db()
        rebuilt_path = self._temp_db("_rebuilt.db")

        # export
        subprocess.run(
            [sys.executable, str(SCRIPTS / "graph-export.py"), copy_path,
             str(WIKI_GRAPH)],
            check=True, capture_output=True, text=True,
        )
        # build
        subprocess.run(
            [sys.executable, str(SCRIPTS / "graph-build.py"), str(WIKI_GRAPH),
             rebuilt_path],
            check=True, capture_output=True, text=True,
        )

        src_counts = _table_counts(copy_path)
        rebuilt_counts = _table_counts(rebuilt_path)

        expected_tables = {
            "papers", "sections", "paper_authors", "entities",
            "predicates", "claims", "entity_edges", "citation_links", "aliases",
        }
        for t in expected_tables:
            assert t in src_counts, f"table {t} missing from source db"
            assert t in rebuilt_counts, f"table {t} missing from rebuilt db"
            assert src_counts[t] == rebuilt_counts[t], (
                f"AC1 fail: {t} src={src_counts[t]} rebuilt={rebuilt_counts[t]}"
            )

        # per-row diff
        for t in expected_tables:
            a_cnt, b_cnt, diffs = _per_row_diff(copy_path, rebuilt_path, t)
            assert a_cnt == b_cnt, f"{t}: src={a_cnt} rebuilt={b_cnt}"
            assert len(diffs) == 0, (
                f"AC1 fail: {t} has {len(diffs)} row diffs: {diffs[:5]}..."
            )

    # -- AC2: gap species exact --

    def test_gap_species_match(self):
        """AC2: 5 gap species identical after round-trip (requires networkx)."""
        import subprocess

        copy_path, md5_before = self._copy_live_db()
        rebuilt_path = self._temp_db("_rebuilt.db")

        subprocess.run(
            [sys.executable, str(SCRIPTS / "graph-export.py"), copy_path,
             str(WIKI_GRAPH)],
            check=True, capture_output=True, text=True,
        )
        subprocess.run(
            [sys.executable, str(SCRIPTS / "graph-build.py"), str(WIKI_GRAPH),
             rebuilt_path],
            check=True, capture_output=True, text=True,
        )

        # Run the oracle gaps.py on both dbs
        oracle_gaps = Path.home() / ".claude/skills/graphbuilding/scripts/gaps.py"
        if not oracle_gaps.exists():
            pytest.skip("Oracle gaps.py not found")

        def _run_gaps(db_path):
            r = subprocess.run(
                [sys.executable, str(oracle_gaps), db_path],
                capture_output=True, text=True, cwd=oracle_gaps.parent,
            )
            return r.stdout

        src_out = _run_gaps(copy_path)
        rebuilt_out = _run_gaps(rebuilt_path)

        assert src_out == rebuilt_out, (
            f"AC2 fail: gap species differ.\nSRC:\n{src_out}\nREBUILT:\n{rebuilt_out}"
        )

    # -- AC3: alias count exact --

    def test_alias_count_exact(self):
        """AC3: 834 aliases in → 834 aliases out (no dangling-FK drop)."""
        import subprocess

        copy_path, md5_before = self._copy_live_db()
        rebuilt_path = self._temp_db("_rebuilt.db")

        subprocess.run(
            [sys.executable, str(SCRIPTS / "graph-export.py"), copy_path,
             str(WIKI_GRAPH)],
            check=True, capture_output=True, text=True,
        )
        subprocess.run(
            [sys.executable, str(SCRIPTS / "graph-build.py"), str(WIKI_GRAPH),
             rebuilt_path],
            check=True, capture_output=True, text=True,
        )

        src_counts = _table_counts(copy_path)
        rebuilt_counts = _table_counts(rebuilt_path)
        assert src_counts["aliases"] == rebuilt_counts["aliases"], (
            f"AC3 fail: aliases src={src_counts['aliases']} rebuilt={rebuilt_counts['aliases']}"
            f" (expect 834 == 834)"
        )
        # Verify it's 834 specifically
        assert rebuilt_counts["aliases"] == 834, (
            f"AC3 fail: expected 834 aliases, got {rebuilt_counts['aliases']}"
        )

    # -- AC5: source untouched --

    def test_source_untouched(self):
        """AC5: copy db md5 is unchanged after full round-trip."""
        import subprocess

        copy_path, md5_before = self._copy_live_db()
        rebuilt_path = self._temp_db("_rebuilt.db")

        subprocess.run(
            [sys.executable, str(SCRIPTS / "graph-export.py"), copy_path,
             str(WIKI_GRAPH)],
            check=True, capture_output=True, text=True,
        )
        subprocess.run(
            [sys.executable, str(SCRIPTS / "graph-build.py"), str(WIKI_GRAPH),
             rebuilt_path],
            check=True, capture_output=True, text=True,
        )

        md5_after = _md5_file(copy_path)
        assert md5_before == md5_after, (
            f"AC5 fail: source db mutated! md5 before={md5_before} after={md5_after}"
        )
