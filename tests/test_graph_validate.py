"""Tests for scripts/graph-validate.py — self-healing integrity guard.

The design's central write-side invariant (graphbuilding-fusion-design.md §7):
the engine must not be able to commit drift. validate() reports the four drift
species the debt doc found; heal() fixes the pointer-drift class via the shared
root() helper and never throws.

Schema matches the real derived db: entities have no `is_canonical` column —
an entity is canonical iff `canonical_id IS NULL`. `orphan` = a claim whose
subject/object entity id does not exist (heal cannot fabricate the entity, so
orphans are report-only).
"""

import importlib.util
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = PROJECT_ROOT / "scripts"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def gv():
    return _load("graph_validate", SCRIPTS / "graph-validate.py")


@pytest.fixture
def db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = OFF")  # seed drift on purpose
    # entities: real schema — canonicality is canonical_id IS NULL (no is_canonical)
    conn.execute(
        """CREATE TABLE entities (
            id INTEGER PRIMARY KEY, name TEXT NOT NULL, super_type TEXT,
            sub_type TEXT, description TEXT, source_paper TEXT,
            canonical_id INTEGER, metadata TEXT, merge_confidence REAL
        )"""
    )
    conn.execute(
        """CREATE TABLE claims (
            id INTEGER PRIMARY KEY, subject_entity_id INTEGER, predicate TEXT,
            object_entity_id INTEGER
        )"""
    )
    conn.commit()
    yield conn, path
    conn.close()
    os.unlink(path)


def _add(conn, eid, name, canonical_id=None):
    conn.execute(
        "INSERT INTO entities (id, name, canonical_id) VALUES (?, ?, ?)",
        (eid, name, canonical_id),
    )
    conn.commit()


def _claim(conn, cid, subj, obj):
    conn.execute(
        "INSERT INTO claims (id, subject_entity_id, predicate, object_entity_id) "
        "VALUES (?, ?, 'rel', ?)",
        (cid, subj, obj),
    )
    conn.commit()


# --- clean db ---

def test_clean_db_reports_no_issues(gv, db):
    conn, _ = db
    _add(conn, 1, "A")                 # canonical (NULL)
    _add(conn, 2, "B", canonical_id=1)  # variant -> 1
    _claim(conn, 10, 1, 2)             # valid claim
    report = gv.validate(conn)
    assert report["total"] == 0, report


# --- each drift species detected ---

def test_detects_self_loop(gv, db):
    conn, _ = db
    _add(conn, 1, "A", canonical_id=1)  # A -> A
    assert gv.validate(conn)["self_loop"] >= 1


def test_detects_chain(gv, db):
    conn, _ = db
    _add(conn, 1, "A", canonical_id=2)
    _add(conn, 2, "B", canonical_id=3)  # multi-hop: parent 2 is itself a variant
    _add(conn, 3, "C")
    assert gv.validate(conn)["chain"] >= 1


def test_detects_dangling(gv, db):
    conn, _ = db
    _add(conn, 1, "A", canonical_id=999)  # 999 missing
    assert gv.validate(conn)["dangling"] >= 1


def test_detects_orphan_claim(gv, db):
    conn, _ = db
    _add(conn, 1, "A")
    _claim(conn, 10, 1, 999)  # object 999 does not exist
    assert gv.validate(conn)["orphan"] >= 1


# --- healing the pointer-drift class ---

def test_heal_fixes_pointer_drift_and_is_idempotent(gv, db):
    conn, _ = db
    _add(conn, 1, "self", canonical_id=1)     # self-loop
    _add(conn, 2, "chainA", canonical_id=3)   # chain head
    _add(conn, 3, "chainB", canonical_id=4)
    _add(conn, 4, "chainC")
    _add(conn, 5, "dangle", canonical_id=999)  # dangling

    assert gv.validate(conn)["total"] > 0
    gv.heal(conn)
    after = gv.validate(conn)
    assert after["total"] == 0, f"heal left drift: {after}"

    gv.heal(conn)  # idempotent
    assert gv.validate(conn)["total"] == 0


def test_chain_is_path_compressed_after_heal(gv, db):
    conn, _ = db
    _add(conn, 1, "A", canonical_id=2)
    _add(conn, 2, "B", canonical_id=3)
    _add(conn, 3, "C")
    gv.heal(conn)
    a_parent = conn.execute("SELECT canonical_id FROM entities WHERE id=1").fetchone()[0]
    assert a_parent == 3, f"expected path-compress A->C (3), got {a_parent}"


def test_heal_does_not_fabricate_orphan_fix(gv, db):
    """Orphan claims need human/ingest repair — heal must not silently 'fix' them."""
    conn, _ = db
    _add(conn, 1, "A")
    _claim(conn, 10, 1, 999)  # object missing
    gv.heal(conn)
    assert gv.validate(conn)["orphan"] >= 1  # still reported


# --- CLI contract ---

def test_missing_db_exits_1():
    import subprocess
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "graph-validate.py"), "--db", "/nonexistent.db"],
        capture_output=True, text=True,
    )
    assert r.returncode == 1
