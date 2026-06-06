"""P3 entity resolution tests — exact match, Jaccard fallback, schema, edge cases.

AC1: Tier 1 exact-name duplicates
AC2: Tier 2A embedding (skips if no ollama)
AC3: Tier 2B token Jaccard fallback
AC4: JSON schema valid
AC5: No COALESCE
AC6: Zero oracle imports
AC7: P1+P2 suites green
AC8: Full P3 suite green
AC9: Missing db exits 1
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
RESOLVE_SCRIPT = SCRIPTS / "graph-resolve.py"


def _run_resolve(*args):
    r = subprocess.run(
        [sys.executable, str(RESOLVE_SCRIPT)] + list(args),
        capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=60,
    )
    return r.returncode, r.stdout, r.stderr


# ---------------------------------------------------------------------------
# AC1: Tier 1 exact-name duplicates
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not DERIVED_DB.exists(), reason="Derived db not found")
class TestTier1Exact:
    """AC1: Tier 1 finds exact-name duplicates within same super_type."""

    def test_produces_valid_json(self):
        exit_code, stdout, stderr = _run_resolve("--json", "--tier", "1")
        assert exit_code == 0, f"exit {exit_code}: {stderr}"
        proposals = json.loads(stdout)
        assert isinstance(proposals, list)

    def test_all_tier1_are_exact_method(self):
        exit_code, stdout, stderr = _run_resolve("--json", "--tier", "1")
        assert exit_code == 0
        proposals = json.loads(stdout)
        for p in proposals:
            assert p["tier"] == 1, f"expected tier 1, got {p['tier']}"
            assert p["method"] == "exact", f"expected exact, got {p['method']}"
            assert p["confidence"] == 1.0, f"exact confidence must be 1.0, got {p['confidence']}"

    def test_exact_duplicates_have_same_name_lower(self):
        """Every Tier 1 proposal must have same lower(name)."""
        exit_code, stdout, stderr = _run_resolve("--json", "--tier", "1")
        assert exit_code == 0
        proposals = json.loads(stdout)
        for p in proposals:
            assert p["entity_a"]["name"].lower().strip() == p["entity_b"]["name"].lower().strip(), (
                f"Tier 1 mismatch: '{p['entity_a']['name']}' vs '{p['entity_b']['name']}'"
            )

    def test_exact_duplicates_same_super_type(self):
        """Every Tier 1 proposal must have same super_type."""
        exit_code, stdout, stderr = _run_resolve("--json", "--tier", "1")
        assert exit_code == 0
        proposals = json.loads(stdout)
        for p in proposals:
            assert p["entity_a"]["super_type"] == p["entity_b"]["super_type"], (
                f"Tier 1 super_type mismatch: {p['entity_a']['super_type']} vs {p['entity_b']['super_type']}"
            )


# ---------------------------------------------------------------------------
# AC3: Tier 2B token Jaccard fallback (always works, no ollama needed)
# ---------------------------------------------------------------------------

class TestTier2Jaccard:
    """AC3: Tier 2B token Jaccard fallback works without ollama."""

    def test_jaccard_similarity_basic(self):
        """Direct unit test of Jaccard on known strings."""
        # We test via the resolve module import
        if not RESOLVE_SCRIPT.exists():
            pytest.skip("graph-resolve.py not yet created")
        import importlib.util
        spec = importlib.util.spec_from_file_location("graph_resolve", RESOLVE_SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["graph_resolve"] = mod
        spec.loader.exec_module(mod)

        # Jaccard: intersection / union of word tokens
        assert mod._jaccard("self attention mechanism", "self attention") > 0.6
        assert mod._jaccard("transformer architecture", "neural network") < 0.5
        assert mod._jaccard("self-attention", "self attention") == 1.0  # same tokens
        assert mod._jaccard("a", "b") == 0.0  # no overlap

    def test_tier2_produces_json(self):
        if not DERIVED_DB.exists():
            pytest.skip("Derived db not found")
        exit_code, stdout, stderr = _run_resolve("--json", "--tier", "2")
        assert exit_code == 0, f"exit {exit_code}: {stderr}"
        proposals = json.loads(stdout)
        assert isinstance(proposals, list)
        for p in proposals:
            assert p["tier"] == 2
            assert p["method"] in ("embedding", "jaccard"), f"unknown method: {p['method']}"
            assert 0.0 <= p["confidence"] <= 1.0, f"confidence out of range: {p['confidence']}"


# ---------------------------------------------------------------------------
# AC4: JSON schema
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not DERIVED_DB.exists(), reason="Derived db not found")
class TestJsonSchema:
    """AC4: JSON output schema valid and consistent."""

    REQUIRED_KEYS = {"tier", "entity_a", "entity_b", "confidence", "method"}
    ENTITY_KEYS = {"id", "name", "super_type", "sub_type"}

    def test_all_proposals_have_required_keys(self):
        exit_code, stdout, stderr = _run_resolve("--json")
        assert exit_code == 0, f"exit {exit_code}: {stderr}"
        proposals = json.loads(stdout)
        for p in proposals:
            missing = self.REQUIRED_KEYS - set(p.keys())
            assert not missing, f"proposal missing keys: {missing}"
            for side in ("entity_a", "entity_b"):
                emissing = self.ENTITY_KEYS - set(p[side].keys())
                assert not emissing, f"{side} missing keys: {emissing}"


# ---------------------------------------------------------------------------
# AC5 + AC6: no COALESCE, no oracle imports
# ---------------------------------------------------------------------------

class TestCodeQuality:
    """AC5 + AC6."""

    def test_no_coalesce(self):
        if not RESOLVE_SCRIPT.exists():
            pytest.skip("graph-resolve.py not yet created")
        lines = RESOLVE_SCRIPT.read_text().splitlines()
        in_doc = False
        code = []
        for l in lines:
            s = l.strip()
            if s.startswith('"""') or s.startswith("'''"): in_doc = not in_doc; continue
            if in_doc: continue
            if s.startswith("#") or not s: continue
            code.append(s)
        assert not [l for l in code if "COALESCE" in l], "COALESCE found in code"

    def test_no_oracle_imports(self):
        if not RESOLVE_SCRIPT.exists():
            pytest.skip("graph-resolve.py not yet created")
        assert "skills/graphbuilding" not in RESOLVE_SCRIPT.read_text()


# ---------------------------------------------------------------------------
# AC7: P1 + P2 still green
# ---------------------------------------------------------------------------

class TestPriorSuitesGreen:
    """AC7: P1 + P2 suites must still pass."""

    def test_p1_p2_suites(self):
        r = subprocess.run(
            [sys.executable, "-m", "pytest",
             str(PROJECT_ROOT / "tests" / "test_graph_roundtrip.py"),
             str(PROJECT_ROOT / "tests" / "test_graph_gaps.py"),
             "-q"],
            capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=120,
        )
        assert r.returncode == 0, f"P1+P2 suites failed!\n{r.stdout}\n{r.stderr}"


# ---------------------------------------------------------------------------
# AC9: missing db
# ---------------------------------------------------------------------------

class TestMissingDb:
    """AC9: graph-resolve.py exits 1 when derived db is missing."""

    def test_missing_db_exits_1(self):
        exit_code, stdout, stderr = _run_resolve("--db", "/nonexistent")
        assert exit_code == 1, f"expected exit 1, got {exit_code}"
