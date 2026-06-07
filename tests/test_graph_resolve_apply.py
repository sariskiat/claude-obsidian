"""Stage A tests for graph-resolve-apply.py — sandbox fixtures, TDD-first.

Stage A ACs (no live vault writes):
  AC2  — winner selection by claim-degree + tie->min-id matches App-A
  AC10 — dry-run default writes nothing; --commit required
  AC12 — edge cases: missing db, missing merge-file, loser-not-in-db,
          already-merged-elsewhere, idempotent re-run, frozenset dedup,
          3-way CFG flat chain, winner==loser self-pair

All tests use SANDBOX sqlite dbs built in tmp dirs.
NEVER touch the live vault, live db, or live entity .md files.
"""

import json
import sqlite3
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = PROJECT_ROOT / "scripts"
APPLY_SCRIPT = SCRIPTS / "graph-resolve-apply.py"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_apply(*args, cwd=None):
    """Run graph-resolve-apply.py and return (exit_code, stdout, stderr)."""
    r = subprocess.run(
        [sys.executable, str(APPLY_SCRIPT)] + list(args),
        capture_output=True, text=True,
        cwd=str(cwd or PROJECT_ROOT),
        timeout=30,
    )
    return r.returncode, r.stdout, r.stderr


def _make_sandbox_db(tmp_path: Path, rows: list) -> Path:
    """Create a minimal sandbox derived db with an entities table and claims table.

    rows is a list of dicts with keys: id, name, super_type, canonical_id (opt),
    and optionally claim_count (to insert matching claims).
    """
    db_path = tmp_path / "graph.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS entities (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            super_type TEXT,
            sub_type TEXT,
            description TEXT,
            source_paper TEXT,
            is_canonical INTEGER DEFAULT 1,
            canonical_id INTEGER,
            merge_confidence REAL,
            metadata TEXT,
            aliases TEXT DEFAULT '[]'
        );
        CREATE TABLE IF NOT EXISTS claims (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject_entity_id INTEGER,
            object_entity_id INTEGER,
            predicate TEXT DEFAULT 'relates_to',
            claim_type TEXT,
            source_paper TEXT
        );
    """)
    for row in rows:
        conn.execute(
            "INSERT INTO entities (id, name, super_type, canonical_id) VALUES (?,?,?,?)",
            (row["id"], row["name"], row.get("super_type", "Method"), row.get("canonical_id")),
        )
        # Insert claims to give this entity a degree
        for i in range(row.get("claim_count", 0)):
            conn.execute(
                "INSERT INTO claims (subject_entity_id, object_entity_id, source_paper) VALUES (?,?,?)",
                (row["id"], row["id"], f"paper-{row['id']}-{i}"),
            )
    conn.commit()
    conn.close()
    return db_path


def _make_tsv(tmp_path: Path, rows: list) -> Path:
    """Write a merge TSV file from a list of (loser_id, winner_id, reason) tuples."""
    tsv_path = tmp_path / "merges.tsv"
    lines = ["# comment header\n"]
    for loser, winner, reason in rows:
        lines.append(f"{loser}\t{winner}\t{reason}\n")
    tsv_path.write_text("".join(lines))
    return tsv_path


def _make_entity_md(entities_dir: Path, entity_id: int, name: str,
                    is_canonical: bool = True, canonical_id=None,
                    merge_confidence=None, slug_override=None) -> Path:
    """Write a minimal entity .md file matching the vault frontmatter schema."""
    slug = slug_override or f"{name.lower().replace(' ', '-')}__e{entity_id}"
    md_path = entities_dir / f"{slug}.md"
    lines = ["---\n"]
    lines.append(f"type: entity\n")
    lines.append(f"id: {entity_id}\n")
    lines.append(f"name: {name}\n")
    lines.append(f"super_type: Method\n")
    lines.append(f"source_paper: test-paper\n")
    lines.append(f"is_canonical: {'true' if is_canonical else 'false'}\n")
    if canonical_id is not None:
        lines.append(f"canonical_id: {canonical_id}\n")
    else:
        lines.append("canonical_id: null\n")
    if merge_confidence is not None:
        lines.append(f"merge_confidence: {merge_confidence}\n")
    else:
        lines.append("merge_confidence: null\n")
    lines.append("metadata: null\n")
    lines.append("aliases: []\n")
    lines.append("---\n\n")
    lines.append(f"Description of {name}.\n")
    md_path.write_text("".join(lines))
    return md_path


# ---------------------------------------------------------------------------
# TestWinnerSelection — AC2: claim-degree + tie->min-id
# ---------------------------------------------------------------------------

class TestWinnerSelection:
    """AC2: Winner chosen by claim-degree; tie -> min id. Never trusts TSV order."""

    def test_higher_degree_wins(self, tmp_path):
        """Entity with higher claim-degree becomes the winner."""
        # entity 10 has degree 5; entity 20 has degree 1 -> 10 wins, 20 is loser
        db = _make_sandbox_db(tmp_path, [
            {"id": 10, "name": "Alpha", "claim_count": 5},
            {"id": 20, "name": "Alpha", "claim_count": 1},
        ])
        entities_dir = tmp_path / "entities"
        entities_dir.mkdir()
        _make_entity_md(entities_dir, 10, "Alpha", slug_override="alpha__e10")
        _make_entity_md(entities_dir, 20, "Alpha", slug_override="alpha__e20")
        tsv = _make_tsv(tmp_path, [(20, 10, "test")])  # TSV says 20 loser, 10 winner
        rc, out, err = _run_apply(
            "--db", str(db),
            "--entities-dir", str(entities_dir),
            "--merge-file", str(tsv),
            "--json",
        )
        assert rc == 0, f"exit {rc}: {err}"
        result = json.loads(out)
        plan = {int(p["loser"]): int(p["winner"]) for p in result["plan"]}
        assert plan[20] == 10, f"expected loser=20->winner=10, got {plan}"

    def test_tsv_field_order_not_trusted(self, tmp_path):
        """Winner computed from degree, not TSV column order (loser could appear first)."""
        # TSV lists (10, 20) but 20 has higher degree, so 10 is the loser
        db = _make_sandbox_db(tmp_path, [
            {"id": 10, "name": "Beta", "claim_count": 1},
            {"id": 20, "name": "Beta", "claim_count": 5},
        ])
        entities_dir = tmp_path / "entities"
        entities_dir.mkdir()
        _make_entity_md(entities_dir, 10, "Beta", slug_override="beta__e10")
        _make_entity_md(entities_dir, 20, "Beta", slug_override="beta__e20")
        tsv = _make_tsv(tmp_path, [(10, 20, "test")])
        rc, out, err = _run_apply(
            "--db", str(db),
            "--entities-dir", str(entities_dir),
            "--merge-file", str(tsv),
            "--json",
        )
        assert rc == 0, f"exit {rc}: {err}"
        result = json.loads(out)
        plan = {int(p["loser"]): int(p["winner"]) for p in result["plan"]}
        # degree 5 > degree 1 -> 20 wins, 10 is loser
        assert plan[10] == 20, f"expected loser=10->winner=20, got {plan}"

    def test_tie_breaks_by_min_id(self, tmp_path):
        """Equal degree -> lower id wins."""
        db = _make_sandbox_db(tmp_path, [
            {"id": 100, "name": "Gamma", "claim_count": 0},
            {"id": 200, "name": "Gamma", "claim_count": 0},
        ])
        entities_dir = tmp_path / "entities"
        entities_dir.mkdir()
        _make_entity_md(entities_dir, 100, "Gamma", slug_override="gamma__e100")
        _make_entity_md(entities_dir, 200, "Gamma", slug_override="gamma__e200")
        tsv = _make_tsv(tmp_path, [(200, 100, "tie test")])
        rc, out, err = _run_apply(
            "--db", str(db),
            "--entities-dir", str(entities_dir),
            "--merge-file", str(tsv),
            "--json",
        )
        assert rc == 0, f"exit {rc}: {err}"
        result = json.loads(out)
        plan = {int(p["loser"]): int(p["winner"]) for p in result["plan"]}
        # both degree 0 -> min-id 100 wins; 200 is loser
        assert plan[200] == 100, f"expected loser=200->winner=100, got {plan}"


# ---------------------------------------------------------------------------
# TestFrozensetDedup — AC12 dedup / CFG 3-way
# ---------------------------------------------------------------------------

class TestFrozensetDedup:
    """AC12: pairs deduped by frozenset; 3-way group flattens to one winner."""

    def test_duplicate_pair_processed_once(self, tmp_path):
        """Same pair listed twice (e.g. Tier-1 and Tier-2 same pair) -> applied once."""
        db = _make_sandbox_db(tmp_path, [
            {"id": 879, "name": "DINOv2", "claim_count": 1},
            {"id": 1367, "name": "DINOv2", "claim_count": 0},
        ])
        entities_dir = tmp_path / "entities"
        entities_dir.mkdir()
        _make_entity_md(entities_dir, 879, "DINOv2", slug_override="dinov2__e879")
        _make_entity_md(entities_dir, 1367, "DINOv2", slug_override="dinov2__e1367")
        # List the same pair twice (as Tier-1 and Tier-2 would)
        tsv = _make_tsv(tmp_path, [
            (1367, 879, "T1 exact"),
            (879, 1367, "T2 jaccard same pair reversed"),
        ])
        rc, out, err = _run_apply(
            "--db", str(db),
            "--entities-dir", str(entities_dir),
            "--merge-file", str(tsv),
            "--json",
        )
        assert rc == 0, f"exit {rc}: {err}"
        result = json.loads(out)
        plan = result["plan"]
        # Should appear exactly once in the plan
        assert len(plan) == 1, f"expected 1 plan entry after dedup, got {len(plan)}: {plan}"
        assert int(plan[0]["loser"]) == 1367
        assert int(plan[0]["winner"]) == 879

    def test_cfg_three_way_flat_chain(self, tmp_path):
        """3-way group (730/1354/1386) -> both losers point to 730 directly (flat)."""
        db = _make_sandbox_db(tmp_path, [
            {"id": 730, "name": "Classifier-Free Guidance", "claim_count": 6},
            {"id": 1354, "name": "Classifier-Free Guidance", "claim_count": 2},
            {"id": 1386, "name": "Classifier-Free Guidance (CFG)", "claim_count": 1},
        ])
        entities_dir = tmp_path / "entities"
        entities_dir.mkdir()
        _make_entity_md(entities_dir, 730, "Classifier-Free Guidance", slug_override="classifier-free-guidance__e730")
        _make_entity_md(entities_dir, 1354, "Classifier-Free Guidance", slug_override="classifier-free-guidance__e1354")
        _make_entity_md(entities_dir, 1386, "Classifier-Free Guidance (CFG)", slug_override="classifier-free-guidance-cfg__e1386")
        tsv = _make_tsv(tmp_path, [
            (1354, 730, "T1 exact"),
            (1386, 730, "T2 acronym gloss"),  # plan already says -> 730
        ])
        rc, out, err = _run_apply(
            "--db", str(db),
            "--entities-dir", str(entities_dir),
            "--merge-file", str(tsv),
            "--json",
        )
        assert rc == 0, f"exit {rc}: {err}"
        result = json.loads(out)
        plan = {int(p["loser"]): int(p["winner"]) for p in result["plan"]}
        # Both losers must point straight to 730 (flat chain, no loser->loser)
        assert plan[1354] == 730, f"1354 should -> 730, got {plan}"
        assert plan[1386] == 730, f"1386 should -> 730, got {plan}"


# ---------------------------------------------------------------------------
# TestDryRunDefault — AC10: dry-run writes nothing
# ---------------------------------------------------------------------------

class TestDryRunDefault:
    """AC10: dry-run is the default; --commit is required to write."""

    def test_dry_run_writes_no_files(self, tmp_path):
        """Running without --commit leaves .md files byte-unchanged."""
        db = _make_sandbox_db(tmp_path, [
            {"id": 1, "name": "Foo", "claim_count": 3},
            {"id": 2, "name": "Foo", "claim_count": 1},
        ])
        entities_dir = tmp_path / "entities"
        entities_dir.mkdir()
        f1 = _make_entity_md(entities_dir, 1, "Foo", slug_override="foo__e1")
        f2 = _make_entity_md(entities_dir, 2, "Foo", slug_override="foo__e2")
        original_content_f2 = f2.read_text()
        tsv = _make_tsv(tmp_path, [(2, 1, "test dry-run")])
        rc, out, err = _run_apply(
            "--db", str(db),
            "--entities-dir", str(entities_dir),
            "--merge-file", str(tsv),
        )
        assert rc == 0, f"exit {rc}: {err}"
        # loser file must be byte-unchanged
        assert f2.read_text() == original_content_f2, (
            "Dry-run must not write loser file; content changed!"
        )

    def test_dry_run_prints_plan(self, tmp_path):
        """Dry-run prints the loser->winner plan to stdout."""
        db = _make_sandbox_db(tmp_path, [
            {"id": 1, "name": "Bar", "claim_count": 5},
            {"id": 2, "name": "Bar", "claim_count": 0},
        ])
        entities_dir = tmp_path / "entities"
        entities_dir.mkdir()
        _make_entity_md(entities_dir, 1, "Bar", slug_override="bar__e1")
        _make_entity_md(entities_dir, 2, "Bar", slug_override="bar__e2")
        tsv = _make_tsv(tmp_path, [(2, 1, "test")])
        rc, out, err = _run_apply(
            "--db", str(db),
            "--entities-dir", str(entities_dir),
            "--merge-file", str(tsv),
        )
        assert rc == 0, f"exit {rc}: {err}"
        assert "2" in out and "1" in out, f"expected loser/winner ids in stdout: {out!r}"

    def test_commit_writes_loser_frontmatter(self, tmp_path):
        """--commit actually writes the loser .md with is_canonical:false + canonical_id."""
        db = _make_sandbox_db(tmp_path, [
            {"id": 10, "name": "Baz", "claim_count": 3},
            {"id": 20, "name": "Baz", "claim_count": 1},
        ])
        entities_dir = tmp_path / "entities"
        entities_dir.mkdir()
        _make_entity_md(entities_dir, 10, "Baz", slug_override="baz__e10")
        loser_md = _make_entity_md(entities_dir, 20, "Baz", slug_override="baz__e20")
        tsv = _make_tsv(tmp_path, [(20, 10, "test commit")])
        rc, out, err = _run_apply(
            "--db", str(db),
            "--entities-dir", str(entities_dir),
            "--merge-file", str(tsv),
            "--commit",
        )
        assert rc == 0, f"exit {rc}: {err}"
        content = loser_md.read_text()
        assert "is_canonical: false" in content, f"loser missing is_canonical: false\n{content}"
        assert "canonical_id: 10" in content, f"loser missing canonical_id: 10\n{content}"
        assert "[[baz__e10]]" in content, f"loser missing canonical wikilink\n{content}"

    def test_commit_sets_merge_confidence(self, tmp_path):
        """--commit writes merge_confidence matching TSV reason context."""
        db = _make_sandbox_db(tmp_path, [
            {"id": 1, "name": "Qux", "claim_count": 2},
            {"id": 2, "name": "Qux", "claim_count": 0},
        ])
        entities_dir = tmp_path / "entities"
        entities_dir.mkdir()
        _make_entity_md(entities_dir, 1, "Qux", slug_override="qux__e1")
        loser_md = _make_entity_md(entities_dir, 2, "Qux", slug_override="qux__e2")
        # T1 exact -> confidence 1.0
        tsv = _make_tsv(tmp_path, [(2, 1, "T1 exact")])
        rc, out, err = _run_apply(
            "--db", str(db),
            "--entities-dir", str(entities_dir),
            "--merge-file", str(tsv),
            "--commit",
        )
        assert rc == 0, f"exit {rc}: {err}"
        content = loser_md.read_text()
        assert "merge_confidence: 1.0" in content, f"T1 exact should set 1.0: {content}"


# ---------------------------------------------------------------------------
# TestEdgeCases — AC12: error conditions + idempotency
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """AC12: error conditions and idempotent re-run."""

    def test_missing_db_exit_1(self, tmp_path):
        """Missing derived db -> exit 1 with rebuild hint."""
        tsv = _make_tsv(tmp_path, [(2, 1, "test")])
        rc, out, err = _run_apply(
            "--db", str(tmp_path / "nonexistent.db"),
            "--entities-dir", str(tmp_path / "entities"),
            "--merge-file", str(tsv),
        )
        assert rc == 1, f"expected exit 1 for missing db, got {rc}"
        assert "graph-build" in err.lower() or "graph-build" in out.lower(), (
            "Missing-db error should mention graph-build.py"
        )

    def test_missing_merge_file_exit_1(self, tmp_path):
        """Missing merge file -> exit 1."""
        db = _make_sandbox_db(tmp_path, [{"id": 1, "name": "X", "claim_count": 0}])
        entities_dir = tmp_path / "entities"
        entities_dir.mkdir()
        rc, out, err = _run_apply(
            "--db", str(db),
            "--entities-dir", str(entities_dir),
            "--merge-file", str(tmp_path / "no_such.tsv"),
        )
        assert rc == 1, f"expected exit 1 for missing merge-file, got {rc}"

    def test_empty_merge_file_exit_1(self, tmp_path):
        """Merge file with only comments and no data rows -> exit 1."""
        db = _make_sandbox_db(tmp_path, [{"id": 1, "name": "X", "claim_count": 0}])
        entities_dir = tmp_path / "entities"
        entities_dir.mkdir()
        tsv = tmp_path / "empty.tsv"
        tsv.write_text("# just a comment\n# another comment\n")
        rc, out, err = _run_apply(
            "--db", str(db),
            "--entities-dir", str(entities_dir),
            "--merge-file", str(tsv),
        )
        assert rc == 1, f"expected exit 1 for empty merge-file, got {rc}"

    def test_loser_not_in_db_skip_continue(self, tmp_path):
        """If loser id is not in the db, skip that pair with a warning; do not crash."""
        db = _make_sandbox_db(tmp_path, [
            {"id": 1, "name": "Real", "claim_count": 3},
            {"id": 2, "name": "Real", "claim_count": 0},
        ])
        entities_dir = tmp_path / "entities"
        entities_dir.mkdir()
        _make_entity_md(entities_dir, 1, "Real", slug_override="real__e1")
        _make_entity_md(entities_dir, 2, "Real", slug_override="real__e2")
        tsv = _make_tsv(tmp_path, [
            (999, 1, "loser 999 not in db"),  # ghost pair
            (2, 1, "valid pair"),
        ])
        rc, out, err = _run_apply(
            "--db", str(db),
            "--entities-dir", str(entities_dir),
            "--merge-file", str(tsv),
            "--json",
        )
        # Should not crash; valid pair still in plan
        assert rc == 0, f"exit {rc}: {err}"
        result = json.loads(out)
        plan = {int(p["loser"]): int(p["winner"]) for p in result["plan"]}
        assert 2 in plan, f"valid pair should be in plan: {plan}"
        assert 999 not in plan, f"ghost loser 999 should be skipped: {plan}"

    def test_already_merged_elsewhere_skip(self, tmp_path):
        """Loser already has canonical_id pointing to a DIFFERENT winner -> skip, no re-point."""
        db = _make_sandbox_db(tmp_path, [
            {"id": 5, "name": "W1", "claim_count": 5},
            {"id": 6, "name": "W2", "claim_count": 4},
            {"id": 7, "name": "Dup", "claim_count": 0},
        ])
        entities_dir = tmp_path / "entities"
        entities_dir.mkdir()
        _make_entity_md(entities_dir, 5, "W1", slug_override="w1__e5")
        _make_entity_md(entities_dir, 6, "W2", slug_override="w2__e6")
        # loser 7 already merged into 5 (different from our proposed 6)
        loser_md = _make_entity_md(entities_dir, 7, "Dup",
                                   is_canonical=False, canonical_id=5,
                                   slug_override="dup__e7")
        original_content = loser_md.read_text()
        tsv = _make_tsv(tmp_path, [(7, 6, "would re-point to 6")])
        rc, out, err = _run_apply(
            "--db", str(db),
            "--entities-dir", str(entities_dir),
            "--merge-file", str(tsv),
            "--commit",
        )
        # Script should succeed (not crash) but skip the re-point
        assert rc == 0, f"exit {rc}: {err}"
        # loser file must be byte-unchanged (already merged elsewhere, skip)
        assert loser_md.read_text() == original_content, (
            "already-merged loser must not be re-pointed"
        )

    def test_idempotent_rerun(self, tmp_path):
        """Running --commit twice -> second run is all no-ops, exit 0."""
        db = _make_sandbox_db(tmp_path, [
            {"id": 100, "name": "Idem", "claim_count": 2},
            {"id": 200, "name": "Idem", "claim_count": 0},
        ])
        entities_dir = tmp_path / "entities"
        entities_dir.mkdir()
        _make_entity_md(entities_dir, 100, "Idem", slug_override="idem__e100")
        loser_md = _make_entity_md(entities_dir, 200, "Idem", slug_override="idem__e200")
        tsv = _make_tsv(tmp_path, [(200, 100, "idempotent test")])
        args = [
            "--db", str(db),
            "--entities-dir", str(entities_dir),
            "--merge-file", str(tsv),
            "--commit",
        ]
        # First run
        rc1, out1, err1 = _run_apply(*args)
        assert rc1 == 0, f"first run exit {rc1}: {err1}"
        content_after_first = loser_md.read_text()
        assert "is_canonical: false" in content_after_first

        # Second run: same args, expect exit 0 and file unchanged (no-op)
        rc2, out2, err2 = _run_apply(*args)
        assert rc2 == 0, f"second run exit {rc2}: {err2}"
        assert loser_md.read_text() == content_after_first, (
            "Idempotent re-run must not change the file again"
        )

    def test_winner_equals_loser_skip(self, tmp_path):
        """Self-pair (winner == loser after degree resolution) -> skip with warning."""
        db = _make_sandbox_db(tmp_path, [
            {"id": 42, "name": "Self", "claim_count": 5},
        ])
        entities_dir = tmp_path / "entities"
        entities_dir.mkdir()
        _make_entity_md(entities_dir, 42, "Self", slug_override="self__e42")
        tsv = _make_tsv(tmp_path, [(42, 42, "self-pair")])
        rc, out, err = _run_apply(
            "--db", str(db),
            "--entities-dir", str(entities_dir),
            "--merge-file", str(tsv),
            "--json",
        )
        # Should not crash; plan should be empty for this pair
        assert rc == 0, f"exit {rc}: {err}"
        result = json.loads(out)
        plan = result["plan"]
        self_entries = [p for p in plan if int(p["loser"]) == 42 and int(p["winner"]) == 42]
        assert len(self_entries) == 0, "self-pair must not appear in plan"


# ---------------------------------------------------------------------------
# TestJsonOutput — JSON schema validation
# ---------------------------------------------------------------------------

class TestJsonOutput:
    """--json output must contain a 'plan' list with required fields."""

    def test_json_schema(self, tmp_path):
        """--json output has top-level 'plan' key with loser, winner, reason fields."""
        db = _make_sandbox_db(tmp_path, [
            {"id": 1, "name": "Jschema", "claim_count": 4},
            {"id": 2, "name": "Jschema", "claim_count": 0},
        ])
        entities_dir = tmp_path / "entities"
        entities_dir.mkdir()
        _make_entity_md(entities_dir, 1, "Jschema", slug_override="jschema__e1")
        _make_entity_md(entities_dir, 2, "Jschema", slug_override="jschema__e2")
        tsv = _make_tsv(tmp_path, [(2, 1, "schema test")])
        rc, out, err = _run_apply(
            "--db", str(db),
            "--entities-dir", str(entities_dir),
            "--merge-file", str(tsv),
            "--json",
        )
        assert rc == 0, f"exit {rc}: {err}"
        result = json.loads(out)
        assert "plan" in result, f"missing 'plan' key in output: {result}"
        assert isinstance(result["plan"], list)
        for entry in result["plan"]:
            for field in ("loser", "winner", "deg_loser", "deg_winner", "reason"):
                assert field in entry, f"plan entry missing field '{field}': {entry}"

    def test_dry_run_flag_indicated(self, tmp_path):
        """--json output indicates whether this was a dry-run."""
        db = _make_sandbox_db(tmp_path, [
            {"id": 1, "name": "Flag", "claim_count": 2},
            {"id": 2, "name": "Flag", "claim_count": 0},
        ])
        entities_dir = tmp_path / "entities"
        entities_dir.mkdir()
        _make_entity_md(entities_dir, 1, "Flag", slug_override="flag__e1")
        _make_entity_md(entities_dir, 2, "Flag", slug_override="flag__e2")
        tsv = _make_tsv(tmp_path, [(2, 1, "flag test")])
        rc, out, err = _run_apply(
            "--db", str(db),
            "--entities-dir", str(entities_dir),
            "--merge-file", str(tsv),
            "--json",
        )
        assert rc == 0
        result = json.loads(out)
        assert result.get("dry_run") is True, (
            f"dry-run should be indicated in json output: {result}"
        )
