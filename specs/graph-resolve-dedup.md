# Spec: Entity Resolution + Embedding Dedup

**Feature ID:** `graph-resolve-dedup`
**Status:** `approved`
**Grilled on:** 2026-06-06
**Approved by:** `saris` (human) on 2026-06-06
**Problem classification:** `unique`

> Phase 3 of the graphbuilding-fusion epic (`docs/graphbuilding-fusion-design.md`).
> P1 proved the round-trip. P2 built native gaps. P3 detects duplicate entities.

---

## 1. Problem, Objective, JTBD

```
┌────────────────────────────┐     ┌────────────────────────────┐     ┌────────────────────────────┐
│ PROBLEM                    │     │ OBJECTIVE                  │     │ METRIC                     │
│                            │     │                            │     │                            │
│ 689 variant entities       │     │ scripts/graph-resolve.py   │     │ Merge proposals found:      │
│ exist in the graph.        │ ──▶ │ detects duplicate entities │ ──▶ │   exact: ? (auto)           │
│ Same concepts under        │     │ via exact-name matching +  │     │   fuzzy: ? (review queue)   │
│ different names fragment   │     │ embedding similarity,      │     │                            │
│ gap counts and weaken the  │     │ outputs ranked review queue│     │ Baseline: 0 proposals today │
│ knowledge graph.           │     │                            │     │ (oracle not run on vault)   │
└────────────────────────────┘     └────────────────────────────┘     └────────────────────────────┘
```

**JTBD.** When I want to clean up duplicate entities in my claim graph, I need a tool
that finds likely duplicates using exact matching and semantic similarity, ranks them
by confidence, and presents them for human review — because I trust my judgment on
whether to merge, but need help finding the candidates.

---

## 2. Two-tier resolution strategy

```
  ┌─────────────────────────────────────────────────────────────┐
  │ Tier 1 — EXACT MATCH  (deterministic, auto-confirmed)       │
  │                                                             │
  │ Same lower(name) + same super_type → merge candidate        │
  │ Confidence: 1.0                                             │
  │ Example: "self-attention" (Concept) vs "Self-Attention"     │
  │          (Concept) → same entity                              │
  └─────────────────────────────────────────────────────────────┘
                               │
                               ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ Tier 2 — FUZZY MATCH  (semantic, human review required)     │
  │                                                             │
  │ Strategy (auto-selected at runtime):                        │
  │   A. EMBEDDING (ollama + nomic-embed-text available)        │
  │      → embed entity name+description, cosine > 0.85        │
  │   B. TOKEN OVERLAP (fallback, no ollama)                    │
  │      → Jaccard similarity on tokenized names > 0.6         │
  │                                                             │
  │ Output: ranked review queue. NEVER auto-merged.             │
  └─────────────────────────────────────────────────────────────┘
```

---

## 3. Architecture

```
  .vault-meta/graph/graph.db  (derived)
       │
       │  graph-resolve.py  ◀── THIS PHASE
       │  ├─ Tier 1: exact-name match  (deterministic)
       │  ├─ Tier 2A: ollama embedding cosine  (>0.85 threshold)
       │  └─ Tier 2B: token Jaccard fallback  (>0.6 threshold)
       ▼
  Ranked merge proposals  (stdout / JSON)
       │
       │  Human reviews, hand-edits wiki/graph/entities/*.md
       │  Re-runs graph-build.py
       ▼
  Cleaner graph → tighter gap counts
```

- **Reads:** `.vault-meta/graph/graph.db` (derived)
- **Writes:** Nothing — outputs proposals only
- **Uses:** `graph_db.root()`, `rerank.py` embed helpers (when ollama available)
- **Zero imports from:** oracle dotdir

---

## 4. Happy path

```
$ uv run python scripts/graph-resolve.py
# Entity Resolution Report
## Tier 1 — Exact Matches (auto-confirmed)
| # | Confidence | Entity A | Entity B | Reason |
|---|------------|----------|----------|--------|
| 1 | 1.00 | Self-Attention | self-attention | exact name match (Concept) |
...

## Tier 2 — Fuzzy Matches (review required)
| # | Confidence | Entity A | Entity B | Reason |
|---|------------|----------|----------|--------|
| 1 | 0.92 | Scaled Dot-Product Attention | Self-Attention | embedding cosine=0.92 |
| 2 | 0.78 | Multi-Head Attention | Parallel Attention | token overlap=0.78 |
...

$ uv run python scripts/graph-resolve.py --json
[{"tier": 1, "entity_a": {...}, "entity_b": {...}, "confidence": 1.0, "method": "exact"}, ...]

$ uv run python scripts/graph-resolve.py --tier 2 --top 20 --json  # fuzzy only
```

---

## 5. Users and roles

| Actor | Can do | Cannot do |
|---|---|---|
| Owner (human) | Run resolve, review fuzzy candidates, hand-edit vault to merge | — |
| `graph-resolve.py` | Read derived db, output ranked proposals | Write derived db, modify vault |
| ollama (optional) | Provide embeddings for semantic similarity | — |

---

## 6. Functional requirements

| ID | Requirement | Notes |
|---|---|---|
| FR1 | `scripts/graph-resolve.py` — two-tier entity resolution | No oracle imports |
| FR2 | Tier 1: exact-name match within same super_type | lower(name) equality |
| FR3 | Tier 2A: embedding cosine similarity via ollama | Reuses `rerank.py` embed pattern, threshold 0.85 |
| FR4 | Tier 2B: token Jaccard similarity fallback | When ollama absent, threshold 0.6 |
| FR5 | Output: ranked JSON with entity pairs, confidence, method | Same schema regardless of strategy |
| FR6 | `--tier 1|2|all` flag to control output | Default: all |
| FR7 | `--top N` flag to limit results | Default: 50 |
| FR8 | Graceful degradation when ollama absent | Falls back to token overlap, exits 0 |
| FR9 | `tests/test_graph_resolve.py` — unit + integration | Tier 1, Tier 2B, JSON schema, edge cases |
| FR10 | P1+P2 suites still green | 16 + 12 = 28 tests |

---

## 7. Business rules

| Rule ID | Rule | Rationale |
|---|---|---|
| BR1 | Exact-name match → confidence 1.0, auto-confirmed | Deterministic, no ambiguity |
| BR2 | Embedding cosine > 0.85 → fuzzy candidate | Empirical threshold for "same concept" |
| BR3 | Token Jaccard > 0.6 → fuzzy candidate (fallback) | Lower bar without semantics; more false positives OK since human reviews |
| BR4 | Same-entity skip (id_a == id_b after root()) | root() prevents self-matches |
| BR5 | Only compare within same super_type for exact; cross-type for fuzzy | "Attention" (Concept) vs "Attention Layer" (Method) — fuzzy can catch conceptual overlap |
| BR6 | Never auto-merge fuzzy candidates | Human judgment required for semantic similarity |
| BR7 | Skips entities already marked as variants (canonical_id IS NOT NULL) | Only canonical entities are resolution targets |

---

## 8. Data and integrations

| Data | R/W | Failure behavior |
|---|---|---|
| `.vault-meta/graph/graph.db` | R | Missing → exit 1 |
| ollama (nomic-embed-text) | R (HTTP) | Unreachable → Tier 2B fallback, exit 0 |
| `graph_db.root()` | import | Missing → import error, exit 1 |
| `rerank.py` embed helpers | import (pattern only) | Missing functions → Tier 2B fallback |

---

## 9. Edge cases

| Scenario | Expected |
|---|---|
| Derived db missing | Exit 1 with message |
| 0 canonical entities | 0 proposals, exit 0 |
| All entities unique (no matches) | 0 proposals, exit 0 |
| ollama unreachable | Tier 1 runs, Tier 2 falls back to token overlap |
| ollama reachable but nomic-embed-text not pulled | Tier 2 falls back to token overlap |
| Entity names with special chars / Unicode | Tokenized safely; Jaccard handles any string |
| Single-word entity names | Jaccard degrades gracefully (1-word overlap = 1.0 if same, 0.0 if different) |
| Very large entity set (1444 entities) | Pairwise comparison = ~285K pairs; filter by super_type first to reduce |

---

## 10. Constraints

- **Must not break:** `graph_db.py`, `graph-gaps.py`, `graph-build.py`, `graph-export.py`, existing test suites
- **Must not import from:** `~/.claude/skills/graphbuilding/`
- **Must not write:** derived db, vault markdown, or JSON snapshot
- **Performance:** <30s for full pairwise scan on 755 canonical entities
- **Mutable surface:** `scripts/graph-resolve.py`, `tests/test_graph_resolve.py`
- **Read-only surface:** Everything else

---

## 11. Acceptance criteria

| ID | Criterion | Verification |
|---|---|---|
| AC1 | Tier 1 finds exact-name duplicates within same super_type | `uv run python -m pytest tests/test_graph_resolve.py -q -k tier1` |
| AC2 | Tier 2A uses embedding when ollama available | `uv run python -m pytest tests/test_graph_resolve.py -q -k embedding` (skips if no ollama) |
| AC3 | Tier 2B token Jaccard fallback works without ollama | `uv run python -m pytest tests/test_graph_resolve.py -q -k jaccard` |
| AC4 | JSON output schema valid and consistent across tiers | `uv run python scripts/graph-resolve.py --json` produces valid JSON with tier/entity_a/entity_b/confidence/method keys |
| AC5 | root() used for all entity resolution — no COALESCE | `grep` on code lines only, exit 1 |
| AC6 | Zero oracle imports | `grep -r "skills/graphbuilding" scripts/graph-resolve.py` exits 1 |
| AC7 | P1+P2 suites still green | `uv run python -m pytest tests/test_graph_roundtrip.py tests/test_graph_gaps.py -q` |
| AC8 | Full P3 suite green | `uv run python -m pytest tests/test_graph_resolve.py -q` |
| AC9 | Missing db → exit 1 | `uv run python scripts/graph-resolve.py --db /nonexistent` exits 1 |

---

## 12. CAN / CANNOT

- **CAN modify:** `scripts/graph-resolve.py`, `tests/test_graph_resolve.py`, `feature_list.json`, `progress.md`
- **CANNOT modify:** All P1 + P2 scripts, wiki/graph/, pyproject.toml, derived db
- **Needs human approval before changing:** Merge thresholds (0.85 / 0.6), ollama endpoint pattern
- **Deferred to P4:** Automated merge that updates entity .md files and rebuilds derived db

---

## 13. Out of scope

- Automated merge execution (P4 — skills/graph/*)
- Hybrid query: claim → source expansion (deferred, open question in P1 spec §15)
- Unifying prose `wiki/entities/` with structured `wiki/graph/entities/`
- Cross-encoder reranker for entity resolution
- Git automation for post-merge commits

---

## 14. Open questions

- P4: Should `graph-resolve.py --apply` auto-edit entity .md files and rebuild?
- P4: Should the review queue be filed as a wiki page for Obsidian visibility?
- Future: Cross-encoder BGE-base reranker for higher precision fuzzy matching?
- Future: Should resolve auto-detect newly ingested entities and run after `graph-build.py`?

---

## Approval Checklist

- Domain lead approved: pending
- Tech lead approved: pending
- Critic reviewed testability: self-grilled
- Metric validated against user value: number of real duplicates found — qualitative, not binary
