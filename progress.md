# Progress Log

### 2026-06-06
- Goal: Phase 1 — markdown-vault migration round-trip. Make wiki/graph/ the source of truth, sqlite a derived index, prove lossless.
- Completed:
  - T1: pyproject.toml (PyYAML+networkx+pytest), .gitignore (derived db ignored), wiki/graph/ tree
  - T2: scripts/graph_db.py — connect(FK on), root() (chain/cycle/dangling-safe, path-compress), typed inserts, no inline COALESCE
  - T3: scripts/graph-export.py — db→vault markdown + JSON snapshot, ALL 834 aliases preserved (AC3 fix)
  - T4: scripts/graph-build.py — vault→derived db, ids preserved, root() self-heals drift, aliases FK-free
  - T5: tests/test_graph_roundtrip.py — full round-trip test suite (16 tests)
  - T6: SCHEMA.md — frontmatter contract + forward-path doc for graph-ingest.py (P4)
- Verification run: 16/16 tests PASS (5.68s)
  - AC1: 9/9 tables byte-equal
  - AC2: 5/5 gap species exact (frontier 65, debate 0, replication 473, coverage 49, white-space 312)
  - AC3: aliases 834→834 (55 dangling-FK rows preserved)
  - AC4: root() chain/cycle/dangling/path-compress, no inline COALESCE
  - AC5: source db md5 unchanged
  - AC6: gitignore correct
  - AC7: full suite green
  - AC8: SCHEMA.md forward-path documented
- Evidence recorded: git note (grade: pass), feature_list.json updated (phase: passing, all tasks completed)
- Commits: 79b1bc5 feat(graph-vault-migration): T1-T4
- Known risks: Subagent spawning broken with deepseek-v4-pro model (requires session restart with model: sonnet)
- Next best action: Rebase onto develop, push, open MR for human review

### 2026-06-06 (graph-full-paper-retrieval slug-dir fallback)
- Goal: Add slug-dir fallback to resolver + re-sync 95 papers.
- Completed:
  - TDD: added TestSlugDirFallback (5 tests) — confirmed RED first, then GREEN after impl.
  - scripts/graph-fulltext.py: _slug_dir_fallback() + PAPER_SCHOLAR_DIR env var + _paper_scholar_root(); resolver fallback fires last (after all source_path strategies).
  - tests/test_graph_fulltext.py: _resolve_tier_a helper updated to mirror slug-dir fallback logic; TestSlugDirFallback class added.
  - Data re-sync: uv run python scripts/graph-fulltext.py sync → 95 Tier-A resolved, 3 skipped (URL non-papers); 63 new .full.md committed.
- Verification: make test-graph 44/44 passed, 4 skipped; make test-fulltext 35/35 passed; ls wiki/graph/papers/*.full.md | wc -l = 95.
- Commit: 01a6479 feat(graph): slug-dir fallback in resolver + 63 new .full.md imports (95/98 total)
