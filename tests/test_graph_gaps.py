"""P2 gap scanner tests — count parity with oracle, schema validation, edge cases.

AC1: Gap counts match oracle exactly (899 total)
AC2: Per-species counts exact (65/0/473/49/312)
AC3: No COALESCE in graph-gaps.py — uses graph_db.root()
AC4: Zero imports from oracle dotdir
AC5: --json output schema valid
AC6: P1 suite still green
AC7: Derived db missing -> exit 1
AC8: Full P2 suite green
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = PROJECT_ROOT / "scripts"
DERIVED_DB = PROJECT_ROOT / ".vault-meta" / "graph" / "graph.db"
GAPS_SCRIPT = SCRIPTS / "graph-gaps.py"

# Expected counts post graph-resolve-apply 9-merge (2026-06-07)
# was frontier=65, debate=0, replication=473, coverage=49, white-space=312, total=899 pre graph-resolve-apply 9-merge
EXPECTED_COUNTS = {
    "frontier": 65,  # was 65 pre graph-resolve-apply 9-merge (unchanged)
    "debate": 0,  # was 0 pre graph-resolve-apply 9-merge (unchanged)
    "replication": 473,  # was 473 pre graph-resolve-apply 9-merge (unchanged)
    "coverage": 45,  # was 49 pre graph-resolve-apply 9-merge; -4 from CFG/CrossAttn/DINOv2 merges
    "white-space": 328,  # was 312 pre graph-resolve-apply 9-merge; +16 from entity consolidation
}
EXPECTED_TOTAL = 911  # was 899 pre graph-resolve-apply 9-merge; net +12


def _run_gaps(*args):
    """Run graph-gaps.py and return (exit_code, stdout, stderr)."""
    r = subprocess.run(
        [sys.executable, str(GAPS_SCRIPT)] + list(args),
        capture_output=True, text=True, cwd=PROJECT_ROOT,
        timeout=60,
    )
    return r.returncode, r.stdout, r.stderr


# ---------------------------------------------------------------------------
# AC1 + AC2: count parity
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not DERIVED_DB.exists(), reason="Derived db not found — run graph-build.py first")
class TestGapCountParity:
    """AC1, AC2: All gap counts match oracle exactly."""

    def test_total_count(self):
        """AC1: 899 total gaps."""
        exit_code, stdout, stderr = _run_gaps("--json", "--top", "999")
        assert exit_code == 0, f"exit {exit_code}: {stderr}"
        gaps = json.loads(stdout)
        assert len(gaps) == EXPECTED_TOTAL, (
            f"AC1 fail: expected {EXPECTED_TOTAL} gaps, got {len(gaps)}"
        )

    def test_frontier_count(self):
        exit_code, stdout, stderr = _run_gaps("--json", "--top", "999")
        assert exit_code == 0, f"exit {exit_code}: {stderr}"
        gaps = json.loads(stdout)
        count = sum(1 for g in gaps if g["species"] == "frontier")
        assert count == EXPECTED_COUNTS["frontier"], (
            f"AC2 fail: expected {EXPECTED_COUNTS['frontier']} frontier, got {count}"
        )

    def test_debate_count(self):
        exit_code, stdout, stderr = _run_gaps("--json", "--top", "999")
        assert exit_code == 0, f"exit {exit_code}: {stderr}"
        gaps = json.loads(stdout)
        count = sum(1 for g in gaps if g["species"] == "debate")
        assert count == EXPECTED_COUNTS["debate"], (
            f"AC2 fail: expected {EXPECTED_COUNTS['debate']} debate, got {count}"
        )

    def test_replication_count(self):
        exit_code, stdout, stderr = _run_gaps("--json", "--top", "999")
        assert exit_code == 0, f"exit {exit_code}: {stderr}"
        gaps = json.loads(stdout)
        count = sum(1 for g in gaps if g["species"] == "replication")
        assert count == EXPECTED_COUNTS["replication"], (
            f"AC2 fail: expected {EXPECTED_COUNTS['replication']} replication, got {count}"
        )

    def test_coverage_count(self):
        exit_code, stdout, stderr = _run_gaps("--json", "--top", "999")
        assert exit_code == 0, f"exit {exit_code}: {stderr}"
        gaps = json.loads(stdout)
        count = sum(1 for g in gaps if g["species"] == "coverage")
        assert count == EXPECTED_COUNTS["coverage"], (
            f"AC2 fail: expected {EXPECTED_COUNTS['coverage']} coverage, got {count}"
        )

    def test_whitespace_count(self):
        exit_code, stdout, stderr = _run_gaps("--json", "--top", "999")
        assert exit_code == 0, f"exit {exit_code}: {stderr}"
        gaps = json.loads(stdout)
        count = sum(1 for g in gaps if g["species"] == "white-space")
        assert count == EXPECTED_COUNTS["white-space"], (
            f"AC2 fail: expected {EXPECTED_COUNTS['white-space']} white-space, got {count}"
        )


# ---------------------------------------------------------------------------
# AC3: no COALESCE
# ---------------------------------------------------------------------------

class TestNoCoalesce:
    """AC3: graph-gaps.py must not contain COALESCE(canonical_id, id)."""

    def test_no_coalesce_in_gaps(self):
        if not GAPS_SCRIPT.exists():
            pytest.skip("graph-gaps.py not yet created")
        content = GAPS_SCRIPT.read_text()
        # Only check non-docstring, non-comment lines
        lines = content.splitlines()
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
        offending = [l for l in code_lines if "COALESCE" in l]
        assert not offending, (
            f"graph-gaps.py contains COALESCE: {offending}"
        )


# ---------------------------------------------------------------------------
# AC4: zero oracle imports
# ---------------------------------------------------------------------------

class TestNoOracleImports:
    """AC4: Zero imports from ~/.claude/skills/graphbuilding/."""

    def test_no_oracle_imports(self):
        if not GAPS_SCRIPT.exists():
            pytest.skip("graph-gaps.py not yet created")
        content = GAPS_SCRIPT.read_text()
        assert "skills/graphbuilding" not in content, (
            "graph-gaps.py imports from oracle dotdir"
        )


# ---------------------------------------------------------------------------
# AC5: JSON schema
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not DERIVED_DB.exists(), reason="Derived db not found")
class TestJsonSchema:
    """AC5: --json output schema matches oracle."""

    REQUIRED_KEYS = {"species", "description", "entities", "claims",
                     "gap_confidence", "explanation"}

    def test_json_schema_valid(self):
        exit_code, stdout, stderr = _run_gaps("--json", "--top", "5")
        assert exit_code == 0, f"exit {exit_code}: {stderr}"
        gaps = json.loads(stdout)
        assert isinstance(gaps, list), "top-level must be a list"
        assert len(gaps) <= 5, f"--top 5 should return <=5 gaps, got {len(gaps)}"
        if gaps:
            g = gaps[0]
            missing = self.REQUIRED_KEYS - set(g.keys())
            assert not missing, f"JSON gap missing keys: {missing}"
            assert g["gap_confidence"] <= 1.0, "confidence must be <= 1.0"
            assert g["gap_confidence"] >= 0.0, "confidence must be >= 0.0"

    def test_json_ranked(self):
        """Gaps must be ranked by confidence descending."""
        exit_code, stdout, stderr = _run_gaps("--json", "--top", "20")
        assert exit_code == 0, f"exit {exit_code}: {stderr}"
        gaps = json.loads(stdout)
        confs = [g["gap_confidence"] for g in gaps]
        assert confs == sorted(confs, reverse=True), (
            "gaps not sorted by confidence descending"
        )


# ---------------------------------------------------------------------------
# AC6: P1 suite still green
# ---------------------------------------------------------------------------

class TestP1SuiteStillGreen:
    """AC6: Phase 1 round-trip tests must still pass."""

    def test_p1_suite(self):
        r = subprocess.run(
            [sys.executable, "-m", "pytest",
             str(PROJECT_ROOT / "tests" / "test_graph_roundtrip.py"), "-q"],
            capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=60,
        )
        assert r.returncode == 0, (
            f"P1 suite failed!\nSTDOUT: {r.stdout}\nSTDERR: {r.stderr}"
        )


# ---------------------------------------------------------------------------
# AC7: missing db exits 1
# ---------------------------------------------------------------------------

class TestMissingDb:
    """AC7: graph-gaps.py exits 1 when derived db is missing."""

    def test_missing_db_exits_1(self):
        exit_code, stdout, stderr = _run_gaps("--db", "/nonexistent/path/graph.db")
        assert exit_code == 1, (
            f"AC7 fail: expected exit 1 for missing db, got {exit_code}\n"
            f"stdout: {stdout}\nstderr: {stderr}"
        )
