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

## 2026-06-06 — EVALUATOR (graph-full-paper-retrieval, P4) → grade: PASS

- Independent real-execution verdict on HEAD 6e9c0bd (branch feature/graph-full-paper-retrieval, working tree clean).
- AC1-AC9 all PASS by fresh OS-process execution:
  - tests/test_graph_fulltext.py: 40 passed (5.17s); make test-graph: 44 passed/4 skipped (baseline held, AC8); make test-fulltext: 40 passed.
  - Live sync: 95 Tier-A imported / 0 new-or-updated / 95 unchanged / 3 skipped (anthropic-kg-cookbook, multimodal-crop-tool = url; bridge-synthesis = unknown_type). Target was >=22 → exceeded.
  - Live /graph read --claim 310 → diffusion-sampling-with-momentum paper, ranked passages w/ paper+path provenance, rerank_source=skipped (ollama-absent degrade, exit 0).
  - --paper / free-query → correct provenance; missing index → exit 10 + "run sync first" hint; empty query / unknown claim → exit 2.
  - Idempotency: .full.md byte-identical across 3 re-runs (md5 a8e97c…); BM25 index identical except updated_at timestamp (docs=3297, vocab=36305 equal); git working tree clean after re-sync.
  - AC3: all 95 chunk page_paths are .full.md under wiki/graph/papers/, 0 stubs. AC4: all 3297 chunks prefix_source=synthetic (zero egress default). AC7 gitignore correct. BR1 source read-only (read_bytes only). BR4 graph-build/export/graph_db untouched fcb1c41..HEAD.
- Hardening validated (real defects only execution would catch): 665 live body_hashes are shared across duplicate slugs (same paper, 2 slugs) — query-mode body_hash dedup + --paper/--claim direct-load by sha1('graph:<slug>')[:6] both correct and tested (10 dedup/twin/fallback tests green).
- MINOR (non-blocking): unknown --paper <slug> exits 0 with 0 chunks; spec edge-case table wants non-zero. Untested.
- PROCESS (harness): feature_list.json AC verification -k filters use underscores (skip_bodyless etc.) that pytest -k does not match against CamelCase classes (TestSkipBodyless) → exit 5 "no tests ran". Behaviors fully pass under camelCase / full suite. Planner -k strings should match real node-ids.
- Git note grade: pass attached to 6e9c0bd. Board: phase → passing, all 9 ACs → done. No push (local only).

### 2026-06-06 (graph-full-paper-retrieval Evaluator fixes)
- Goal: Two post-PASS fixes from Evaluator review — unknown-paper non-zero exit (code) + AC -k filter hygiene (board).
- FIX 1 (code, TDD): unknown --paper <slug> with zero chunks now exits 2 (mirrors --claim behavior). Failing tests written first (TestUnknownPaperNonzeroExit, 2 RED), implementation added in load_chunks_for_slug path of graph-retrieve.py, all 4 tests GREEN. Known-paper and free-text-no-hits still exit 0 (confirmed by 2 guard tests).
- FIX 2 (board): AC2/AC3/AC4/AC9 verification strings in feature_list.json updated from underscore -k filters (skip_bodyless, retrieve_provenance, no_egress_default, read_surface — collected 0 tests, exit 5) to CamelCase class names (SkipBodyless, RetrieveProvenance, NoEgressDefault, ReadSurface). Each verified to exit 0 and collect tests.
- Verification: tests/test_graph_fulltext.py 44 passed (was 40, +4 new); make test-graph 44/4 unchanged.
- Files: scripts/graph-retrieve.py, tests/test_graph_fulltext.py, feature_list.json, progress.md.
