#!/usr/bin/env python3
"""test_retrieve.py — hermetic tests for scripts/retrieve.py and scripts/rerank.py.

No network, no ollama, no LLM calls. Tests cover:
  - import_sibling resolves hyphenated module names
  - chunk_snippet truncation behavior
  - rerank.cosine math correctness
  - rerank.rerank() no-op behavior when ollama is unreachable
  - retrieve.py exit 10 (not provisioned) when chunks/index are missing
  - dedupe-by-page logic via integration smoke test on synthetic fixtures

Usage:
  python3 tests/test_retrieve.py
"""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest.mock
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RETRIEVE = ROOT / "scripts" / "retrieve.py"
RERANK = ROOT / "scripts" / "rerank.py"
BM25 = ROOT / "scripts" / "bm25-index.py"


def import_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


retrieve = import_script("retrieve", RETRIEVE)
rerank = import_script("rerank", RERANK)
bm25 = import_script("bm25", BM25)


class Fail(SystemExit):
    pass


def assert_eq(label, expected, actual):
    if expected != actual:
        raise Fail(f"FAIL {label}: expected {expected!r}, got {actual!r}")
    print(f"OK   {label}")


def assert_true(label, cond, hint=""):
    if not cond:
        raise Fail(f"FAIL {label}{(': ' + hint) if hint else ''}")
    print(f"OK   {label}")


def assert_close(label, expected, actual, eps=1e-6):
    if abs(expected - actual) > eps:
        raise Fail(f"FAIL {label}: expected ~{expected}, got {actual}")
    print(f"OK   {label}")


# ─── import_sibling ──────────────────────────────────────────────────────────
def test_import_sibling_resolves_hyphenated_names():
    """retrieve.import_sibling('bm25_index', 'bm25-index.py') must succeed."""
    mod = retrieve.import_sibling("bm25_index", "bm25-index.py")
    assert_true("import_sibling returns module", mod is not None)
    assert_true("module has tokenize()", callable(getattr(mod, "tokenize", None)))


# ─── chunk_snippet ───────────────────────────────────────────────────────────
def test_chunk_snippet_short():
    """Short chunks should pass through unchanged."""
    out = retrieve.chunk_snippet({"raw_text": "short text"}, max_chars=200)
    assert_eq("chunk_snippet short pass-through", "short text", out)


def test_chunk_snippet_truncates_with_ellipsis():
    """Long chunks should be truncated with an ellipsis."""
    long_text = "x" * 500
    out = retrieve.chunk_snippet({"raw_text": long_text}, max_chars=100)
    assert_true("snippet length under cap", len(out) <= 110, hint=f"len={len(out)}")
    assert_true("snippet ends with ellipsis", out.endswith("…"))


# ─── rerank.cosine() ─────────────────────────────────────────────────────────
def test_cosine_identical():
    assert_close("cosine identical vectors", 1.0, rerank.cosine([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]))


def test_cosine_orthogonal():
    assert_close("cosine orthogonal", 0.0, rerank.cosine([1.0, 0.0], [0.0, 1.0]))


def test_cosine_anti_parallel():
    assert_close("cosine anti-parallel", -1.0, rerank.cosine([1.0, 0.0], [-1.0, 0.0]))


def test_cosine_length_mismatch():
    """Mismatched vector lengths should return 0.0 (defensive, not crash)."""
    assert_close("cosine length mismatch", 0.0, rerank.cosine([1.0], [1.0, 2.0]))


def test_cosine_zero_vector():
    assert_close("cosine zero vector", 0.0, rerank.cosine([0.0, 0.0], [1.0, 2.0]))


# ─── rerank.rerank() no-op fallback ──────────────────────────────────────────
def test_rerank_noop_when_ollama_unreachable():
    """When ollama is not reachable, rerank should pass candidates through with
    rerank_source='noop-no-ollama'. We force this by patching ollama_alive."""
    with unittest.mock.patch.object(rerank, "ollama_alive", return_value=(False, [])):
        candidates = [
            {"chunk_id": "c-001:0", "score": 7.5, "path": "fake/p1.json"},
            {"chunk_id": "c-002:0", "score": 5.1, "path": "fake/p2.json"},
        ]
        out = rerank.rerank("query", candidates, top_k=5)
        assert_eq("rerank no-op preserves order", ["c-001:0", "c-002:0"],
                  [c["chunk_id"] for c in out])
        assert_true("rerank no-op tags source",
                    all(c.get("rerank_source") == "noop-no-ollama" for c in out))
        assert_true("rerank no-op copies score to rerank_score",
                    all(c["rerank_score"] == c["score"] for c in out))


def test_rerank_noop_when_model_missing():
    """When ollama is up but model isn't pulled, rerank should still no-op."""
    with unittest.mock.patch.object(rerank, "ollama_alive", return_value=(True, ["other-model"])):
        candidates = [{"chunk_id": "c-001:0", "score": 5.0, "path": "x"}]
        out = rerank.rerank("query", candidates, top_k=5)
        assert_eq("rerank no-op for missing model", "noop-no-model", out[0]["rerank_source"])


def test_rerank_truncates_to_top_k():
    with unittest.mock.patch.object(rerank, "ollama_alive", return_value=(False, [])):
        candidates = [{"chunk_id": f"c-{i:03}:0", "score": float(i), "path": "x"} for i in range(10)]
        out = rerank.rerank("query", candidates, top_k=3)
        assert_eq("rerank truncates to top_k", 3, len(out))


# ─── retrieve.py CLI: exit 10 when not provisioned ────────────────────────────
def test_retrieve_exits_10_without_index():
    """End-to-end CLI test: with no .vault-meta/bm25/index.json, retrieve.py must exit 10."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Build a minimal vault layout under tmpdir
        sandbox = Path(tmpdir)
        (sandbox / "scripts").mkdir()
        (sandbox / ".vault-meta").mkdir()
        # Copy retrieve.py and its dependencies into the sandbox
        import shutil
        for f in ["retrieve.py", "bm25-index.py", "rerank.py"]:
            shutil.copy(ROOT / "scripts" / f, sandbox / "scripts" / f)
            os.chmod(sandbox / "scripts" / f, 0o755)
        # Run retrieve.py — should exit 10 because no bm25 index exists
        result = subprocess.run(
            [sys.executable, str(sandbox / "scripts" / "retrieve.py"), "test query"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert_eq("retrieve.py exit 10 when not provisioned", 10, result.returncode)
        assert_true("retrieve.py prints friendly error",
                    "no BM25 index" in result.stderr,
                    hint=result.stderr[:200])


# ─── Integration smoke test: end-to-end with synthetic data ──────────────────
def test_end_to_end_with_synthetic_chunks():
    """Build a minimal vault with 2 chunks, index it, run retrieve, verify output."""
    import hashlib
    with tempfile.TemporaryDirectory() as tmpdir:
        sandbox = Path(tmpdir)
        (sandbox / "scripts").mkdir()
        meta = sandbox / ".vault-meta"
        chunks_dir = meta / "chunks"
        bm25_dir = meta / "bm25"
        chunks_dir.mkdir(parents=True)
        bm25_dir.mkdir(parents=True)
        # Copy scripts
        import shutil
        for f in ["retrieve.py", "bm25-index.py", "rerank.py"]:
            shutil.copy(ROOT / "scripts" / f, sandbox / "scripts" / f)
            os.chmod(sandbox / "scripts" / f, 0o755)
        # Write 2 synthetic chunks
        def chunk(addr, idx, text):
            return {
                "schema_version": 1,
                "page_path": f"wiki/fake/{addr}.md",
                "page_address": addr,
                "chunk_index": idx,
                "raw_text": text,
                "contextualized_text": text,
                "prefix": "",
                "prefix_source": "synthetic",
                "char_count": len(text),
                "body_hash": "sha256:" + hashlib.sha256(text.encode()).hexdigest(),
                "page_body_hash": "sha256:0",
                "created_at": "2026-05-17T00:00:00Z",
            }
        (chunks_dir / "c-000001").mkdir()
        (chunks_dir / "c-000002").mkdir()
        (chunks_dir / "c-000001" / "chunk-000.json").write_text(
            json.dumps(chunk("c-000001", 0, "compounding wiki vault pattern by karpathy")))
        (chunks_dir / "c-000002" / "chunk-000.json").write_text(
            json.dumps(chunk("c-000002", 0, "obsidian cli transport detection")))
        # Build index via subprocess (uses the sandbox's META_DIR? no — it uses the
        # script's hard-coded paths relative to its location. Since we copied the
        # script into sandbox/scripts/, VAULT_ROOT will compute to `sandbox`.)
        result = subprocess.run(
            [sys.executable, str(sandbox / "scripts" / "bm25-index.py"), "build"],
            capture_output=True, text=True, timeout=10)
        assert_eq("bm25 build rc=0", 0, result.returncode)
        # Run retrieve
        result = subprocess.run(
            [sys.executable, str(sandbox / "scripts" / "retrieve.py"),
             "karpathy wiki", "--top", "2", "--no-rerank"],
            capture_output=True, text=True, timeout=10)
        assert_eq("retrieve rc=0", 0, result.returncode)
        out = json.loads(result.stdout)
        assert_eq("retrieve.strategy is bm25-only", "bm25-only", out["strategy"])
        assert_true("retrieve returns at least 1 candidate", len(out["candidates"]) >= 1)
        # c-000001 should rank above c-000002 for "karpathy wiki"
        first = out["candidates"][0]
        assert_eq("top hit is c-000001", "c-000001", first["page_address"])


# ─── M8 closure: --explain and --no-rerank flag coverage ─────────────────────
def test_explain_flag_adds_diagnostics_block():
    """v1.7.2 / closes audit M8: --explain must include an 'explain' diagnostics block."""
    import hashlib
    with tempfile.TemporaryDirectory() as tmpdir:
        sandbox = Path(tmpdir)
        (sandbox / "scripts").mkdir()
        meta = sandbox / ".vault-meta"
        chunks_dir = meta / "chunks"
        bm25_dir = meta / "bm25"
        chunks_dir.mkdir(parents=True)
        bm25_dir.mkdir(parents=True)
        import shutil
        for f in ["retrieve.py", "bm25-index.py", "rerank.py"]:
            shutil.copy(ROOT / "scripts" / f, sandbox / "scripts" / f)
            os.chmod(sandbox / "scripts" / f, 0o755)
        # 2 synthetic chunks
        (chunks_dir / "c-000010").mkdir()
        (chunks_dir / "c-000010" / "chunk-000.json").write_text(json.dumps({
            "schema_version": 1, "page_path": "wiki/fake/c-000010.md",
            "page_address": "c-000010", "chunk_index": 0,
            "raw_text": "hybrid retrieval pipeline",
            "contextualized_text": "hybrid retrieval pipeline",
            "prefix": "", "prefix_source": "synthetic",
            "char_count": 25,
            "body_hash": "sha256:" + hashlib.sha256(b"hybrid retrieval pipeline").hexdigest(),
            "page_body_hash": "sha256:0",
            "created_at": "2026-05-17T00:00:00Z",
        }))
        # Build index
        subprocess.run([sys.executable, str(sandbox / "scripts" / "bm25-index.py"), "build"],
                       capture_output=True, timeout=10, check=True)
        # Run with --explain --no-rerank
        result = subprocess.run(
            [sys.executable, str(sandbox / "scripts" / "retrieve.py"),
             "hybrid", "--top", "1", "--no-rerank", "--explain"],
            capture_output=True, text=True, timeout=10)
        assert_eq("retrieve --explain --no-rerank rc=0", 0, result.returncode)
        out = json.loads(result.stdout)
        assert_true("--explain produces 'explain' key",
                    "explain" in out, hint=f"keys={list(out.keys())}")
        explain = out.get("explain", {})
        assert_true("--explain reports BM25 candidate count",
                    "bm25_candidates" in explain or "bm25" in str(explain).lower(),
                    hint=f"explain={explain}")


def test_no_rerank_flag_strategy_bm25_only():
    """v1.7.2 / closes audit M8: --no-rerank must produce strategy='bm25-only'."""
    import hashlib
    with tempfile.TemporaryDirectory() as tmpdir:
        sandbox = Path(tmpdir)
        (sandbox / "scripts").mkdir()
        meta = sandbox / ".vault-meta"
        chunks_dir = meta / "chunks"
        bm25_dir = meta / "bm25"
        chunks_dir.mkdir(parents=True)
        bm25_dir.mkdir(parents=True)
        import shutil
        for f in ["retrieve.py", "bm25-index.py", "rerank.py"]:
            shutil.copy(ROOT / "scripts" / f, sandbox / "scripts" / f)
            os.chmod(sandbox / "scripts" / f, 0o755)
        (chunks_dir / "c-000020").mkdir()
        (chunks_dir / "c-000020" / "chunk-000.json").write_text(json.dumps({
            "schema_version": 1, "page_path": "wiki/fake/c-000020.md",
            "page_address": "c-000020", "chunk_index": 0,
            "raw_text": "transport detection fallback chain",
            "contextualized_text": "transport detection fallback chain",
            "prefix": "", "prefix_source": "synthetic",
            "char_count": 35,
            "body_hash": "sha256:" + hashlib.sha256(b"transport detection fallback chain").hexdigest(),
            "page_body_hash": "sha256:0",
            "created_at": "2026-05-17T00:00:00Z",
        }))
        subprocess.run([sys.executable, str(sandbox / "scripts" / "bm25-index.py"), "build"],
                       capture_output=True, timeout=10, check=True)
        result = subprocess.run(
            [sys.executable, str(sandbox / "scripts" / "retrieve.py"),
             "transport", "--top", "1", "--no-rerank"],
            capture_output=True, text=True, timeout=10)
        assert_eq("retrieve --no-rerank rc=0", 0, result.returncode)
        out = json.loads(result.stdout)
        assert_eq("--no-rerank sets strategy='bm25-only'", "bm25-only", out.get("strategy"))
        # --no-rerank produces a consistent shape: rerank fields are populated
        # but rerank_source is "skipped" (so callers don't have to special-case).
        candidates = out.get("candidates", [])
        assert_true("--no-rerank still returns candidates", len(candidates) >= 1)
        first = candidates[0]
        assert_eq("--no-rerank candidate rerank_source='skipped'", "skipped",
                  first.get("rerank_source"))
        assert_eq("--no-rerank candidate rerank_score equals bm25_score",
                  first.get("bm25_score"), first.get("rerank_score"))


# ─── claude tier tests ────────────────────────────────────────────────────────

def _make_fake_claude_exe(tmp_dir: Path, response_json: str) -> Path:
    """Write a shell script that acts as a fake 'claude -p' command.

    The script reads stdin (the prompt) and writes response_json to stdout.
    Accepts any arguments so it mirrors the real `claude -p` invocation.
    """
    exe = tmp_dir / "fake_claude"
    exe.write_text(
        "#!/bin/sh\n"
        f"echo '{response_json}'\n"
    )
    exe.chmod(0o755)
    return exe


def _make_never_called_claude_exe(tmp_dir: Path) -> Path:
    """Write a shell script that exits 1 with an error if ever called."""
    exe = tmp_dir / "fake_claude_never"
    exe.write_text(
        "#!/bin/sh\n"
        "echo 'FORBIDDEN: claude called on default path' >&2\n"
        "exit 1\n"
    )
    exe.chmod(0o755)
    return exe


def test_claude_tier_reorders_by_given_ids():
    """AC1: fake claude returns permuted order of real ids; output order must match."""
    candidates = [
        {"chunk_id": "c-aaa:0", "score": 9.0, "path": "x"},
        {"chunk_id": "c-bbb:0", "score": 8.0, "path": "x"},
        {"chunk_id": "c-ccc:0", "score": 7.0, "path": "x"},
    ]
    # fake claude returns reversed order
    permuted = '["c-ccc:0","c-bbb:0","c-aaa:0"]'
    with tempfile.TemporaryDirectory() as tmpdir:
        fake = _make_fake_claude_exe(Path(tmpdir), permuted)
        out = rerank.rerank_claude("test query", candidates, top_k=3,
                                   claude_cmd=str(fake))
    ids = [c["chunk_id"] for c in out]
    assert_eq("claude tier reorders by given ids",
              ["c-ccc:0", "c-bbb:0", "c-aaa:0"], ids)


def test_claude_tier_drops_invented_ids():
    """AC2: out-of-set ids are silently dropped; remainder fills from BM25 order."""
    candidates = [
        {"chunk_id": "c-111:0", "score": 5.0, "path": "x"},
        {"chunk_id": "c-222:0", "score": 4.0, "path": "x"},
    ]
    # fake claude returns one real id + one invented
    resp = '["c-111:0","c-INVENTED:0"]'
    with tempfile.TemporaryDirectory() as tmpdir:
        fake = _make_fake_claude_exe(Path(tmpdir), resp)
        out = rerank.rerank_claude("test query", candidates, top_k=2,
                                   claude_cmd=str(fake))
    ids = [c["chunk_id"] for c in out]
    assert_true("claude tier drops invented ids: count == input",
                len(out) == 2, hint=f"ids={ids}")
    assert_true("claude tier drops invented ids: c-INVENTED absent",
                "c-INVENTED:0" not in ids, hint=f"ids={ids}")
    assert_true("claude tier drops invented ids: c-111 present",
                "c-111:0" in ids, hint=f"ids={ids}")
    print("OK   claude tier drops invented ids")


def test_default_path_no_claude_call():
    """AC3: rerank_tier='auto' must never call claude subprocess."""
    candidates = [
        {"chunk_id": "c-001:0", "score": 7.5, "path": "fake/p1.json"},
    ]
    import unittest.mock as _mock
    with unittest.mock.patch.object(rerank, "ollama_alive", return_value=(False, [])):
        with _mock.patch("subprocess.run") as mock_run:
            rerank.rerank("query", candidates, top_k=5, rerank_tier="auto")
            assert_true("default path no claude call",
                        not mock_run.called,
                        hint=f"subprocess.run called {mock_run.call_count} times")


def test_retrieve_cli_claude_tier_strategy():
    """AC4: retrieve.py --rerank-tier claude with fake claude; strategy contains 'claude'."""
    import hashlib
    with tempfile.TemporaryDirectory() as tmpdir:
        sandbox = Path(tmpdir)
        (sandbox / "scripts").mkdir()
        meta = sandbox / ".vault-meta"
        chunks_dir = meta / "chunks"
        bm25_dir = meta / "bm25"
        chunks_dir.mkdir(parents=True)
        bm25_dir.mkdir(parents=True)
        import shutil
        for f in ["retrieve.py", "bm25-index.py", "rerank.py"]:
            shutil.copy(ROOT / "scripts" / f, sandbox / "scripts" / f)
            os.chmod(sandbox / "scripts" / f, 0o755)
        # One synthetic chunk
        (chunks_dir / "c-cli01").mkdir()
        (chunks_dir / "c-cli01" / "chunk-000.json").write_text(json.dumps({
            "schema_version": 1, "page_path": "wiki/fake/c-cli01.md",
            "page_address": "c-cli01", "chunk_index": 0,
            "raw_text": "attention mechanism efficiency transformer",
            "contextualized_text": "attention mechanism efficiency transformer",
            "prefix": "", "prefix_source": "synthetic", "char_count": 41,
            "body_hash": "sha256:" + hashlib.sha256(
                b"attention mechanism efficiency transformer").hexdigest(),
            "page_body_hash": "sha256:0", "created_at": "2026-05-17T00:00:00Z",
        }))
        subprocess.run([sys.executable, str(sandbox / "scripts" / "bm25-index.py"), "build"],
                       capture_output=True, timeout=10, check=True)

        # fake claude that returns the single real id
        fake_exe = sandbox / "fake_claude"
        fake_exe.write_text(
            "#!/bin/sh\n"
            'echo \'["c-cli01:0"]\'\n'
        )
        fake_exe.chmod(0o755)

        result = subprocess.run(
            [sys.executable, str(sandbox / "scripts" / "retrieve.py"),
             "attention mechanism", "--top", "1",
             "--rerank-tier", "claude", "--claude-cmd", str(fake_exe)],
            capture_output=True, text=True, timeout=15,
        )
        assert_eq("retrieve CLI claude tier strategy: exit 0", 0, result.returncode)
        out = json.loads(result.stdout)
        assert_true("retrieve CLI claude tier strategy",
                    "claude" in out.get("strategy", ""),
                    hint=f"strategy={out.get('strategy')!r}")


def test_fallback_ladder():
    """AC5: auto+no-ollama -> noop-no-ollama; claude+fake-ok -> claude:...; claude+fail -> claude-error."""
    candidates = [
        {"chunk_id": "c-x:0", "score": 5.0, "path": "x"},
    ]
    # auto + no ollama -> noop-no-ollama
    with unittest.mock.patch.object(rerank, "ollama_alive", return_value=(False, [])):
        out = rerank.rerank("q", candidates[:], top_k=5, rerank_tier="auto")
        assert_eq("fallback ladder: auto+no-ollama -> noop-no-ollama",
                  "noop-no-ollama", out[0]["rerank_source"])

    # claude + fake-ok -> source contains 'claude'
    with tempfile.TemporaryDirectory() as tmpdir:
        fake = _make_fake_claude_exe(Path(tmpdir), '["c-x:0"]')
        out2 = rerank.rerank("q", [{"chunk_id": "c-x:0", "score": 5.0, "path": "x"}],
                             top_k=5, rerank_tier="claude", claude_cmd=str(fake))
        assert_true("fallback ladder: claude+fake-ok -> source starts with claude",
                    out2[0]["rerank_source"].startswith("claude"),
                    hint=f"source={out2[0]['rerank_source']!r}")

    # claude + failing cmd -> claude-error
    with tempfile.TemporaryDirectory() as tmpdir:
        bad = Path(tmpdir) / "bad_claude"
        bad.write_text("#!/bin/sh\nexit 1\n")
        bad.chmod(0o755)
        out3 = rerank.rerank("q", [{"chunk_id": "c-x:0", "score": 5.0, "path": "x"}],
                             top_k=5, rerank_tier="claude", claude_cmd=str(bad))
        assert_eq("fallback ladder", "claude-error", out3[0]["rerank_source"])


def test_no_rerank_beats_claude_tier():
    """AC10: --no-rerank wins over --rerank-tier claude; fake claude never called."""
    import hashlib
    with tempfile.TemporaryDirectory() as tmpdir:
        sandbox = Path(tmpdir)
        (sandbox / "scripts").mkdir()
        meta = sandbox / ".vault-meta"
        chunks_dir = meta / "chunks"
        bm25_dir = meta / "bm25"
        chunks_dir.mkdir(parents=True)
        bm25_dir.mkdir(parents=True)
        import shutil
        for f in ["retrieve.py", "bm25-index.py", "rerank.py"]:
            shutil.copy(ROOT / "scripts" / f, sandbox / "scripts" / f)
            os.chmod(sandbox / "scripts" / f, 0o755)
        # One synthetic chunk
        (chunks_dir / "c-nrb01").mkdir()
        (chunks_dir / "c-nrb01" / "chunk-000.json").write_text(json.dumps({
            "schema_version": 1, "page_path": "wiki/fake/c-nrb01.md",
            "page_address": "c-nrb01", "chunk_index": 0,
            "raw_text": "no rerank beats claude tier test",
            "contextualized_text": "no rerank beats claude tier test",
            "prefix": "", "prefix_source": "synthetic", "char_count": 32,
            "body_hash": "sha256:" + hashlib.sha256(
                b"no rerank beats claude tier test").hexdigest(),
            "page_body_hash": "sha256:0", "created_at": "2026-05-17T00:00:00Z",
        }))
        subprocess.run([sys.executable, str(sandbox / "scripts" / "bm25-index.py"), "build"],
                       capture_output=True, timeout=10, check=True)

        # fake claude that would fail if called
        never_exe = sandbox / "never_claude"
        never_exe.write_text(
            "#!/bin/sh\n"
            "echo 'FORBIDDEN: claude called when --no-rerank active' >&2\n"
            "exit 1\n"
        )
        never_exe.chmod(0o755)

        result = subprocess.run(
            [sys.executable, str(sandbox / "scripts" / "retrieve.py"),
             "no rerank", "--top", "1",
             "--no-rerank", "--rerank-tier", "claude",
             "--claude-cmd", str(never_exe)],
            capture_output=True, text=True, timeout=15,
        )
        assert_eq("no-rerank beats claude tier: exit 0", 0, result.returncode)
        out = json.loads(result.stdout)
        assert_eq("no-rerank beats claude tier",
                  "bm25-only", out.get("strategy"))


def main():
    print("=== test_retrieve.py ===")
    test_import_sibling_resolves_hyphenated_names()
    test_chunk_snippet_short()
    test_chunk_snippet_truncates_with_ellipsis()
    test_cosine_identical()
    test_cosine_orthogonal()
    test_cosine_anti_parallel()
    test_cosine_length_mismatch()
    test_cosine_zero_vector()
    test_rerank_noop_when_ollama_unreachable()
    test_rerank_noop_when_model_missing()
    test_rerank_truncates_to_top_k()
    test_retrieve_exits_10_without_index()
    test_end_to_end_with_synthetic_chunks()
    test_explain_flag_adds_diagnostics_block()
    test_no_rerank_flag_strategy_bm25_only()
    # claude-tier tests (T1 — should be RED before implementation)
    test_claude_tier_reorders_by_given_ids()
    test_claude_tier_drops_invented_ids()
    test_default_path_no_claude_call()
    test_retrieve_cli_claude_tier_strategy()
    test_fallback_ladder()
    test_no_rerank_beats_claude_tier()
    print("\nAll retrieve tests passed.")


if __name__ == "__main__":
    main()
