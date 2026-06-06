# Spec: Native Gap Scanner — Replace Oracle with Repo-Native Analytics

**Feature ID:** `graph-native-gaps`
**Status:** `approved`
**Grilled on:** 2026-06-06
**Approved by:** `saris` (human) on 2026-06-06
**Problem classification:** `unique`

> Phase 2 of the graphbuilding-fusion epic (`docs/graphbuilding-fusion-design.md`).
> Phase 1 proved the round-trip. Phase 2 builds native analytics on it.

---

## 1. Problem, Objective, JTBD

```
┌────────────────────────────┐     ┌────────────────────────────┐     ┌────────────────────────────┐
│ PROBLEM                    │     │ OBJECTIVE                  │     │ METRIC                     │
│                            │     │                            │     │                            │
│ Gap scanner lives in       │     │ Native scripts/graph-gaps  │     │ Exact count match:         │
│ oracle dotdir:             │ ──▶ │ runs on derived db,        │ ──▶ │   frontier   65            │
│ ~/.claude/skills/          │     │ uses graph_db.root(),      │     │   debate      0            │
│   graphbuilding/scripts/   │     │ zero oracle imports        │     │   replication 473          │
│   gaps.py                  │     │                            │     │   coverage    49           │
│                            │     │ Same 5 species, same       │     │   white-space 312          │
│ • uses COALESCE (broken)   │     │ ranked report, same        │     │   ─────────────────────    │
│ • not in this repo         │     │ confidence formulas        │     │   total      899           │
│ • can't evolve with vault  │     │                            │     │                            │
└────────────────────────────┘     └────────────────────────────┘     └────────────────────────────┘
```

**JTBD.** When I scan my claim graph for research gaps, I want a native gap scanner
in this repo that uses `root()` (not broken COALESCE), runs on the derived db, and
produces identical results to the oracle, so the analytics engine lives with the vault
and evolves with it.

---

## 2. The one critical fix: `root()` replaces COALESCE

The oracle's `_get_canonical_id` uses the broken single-hop pattern:

```python
# ORACLE — BROKEN (BR2)
def _get_canonical_id(conn, entity_id):
    row = conn.execute(
        "SELECT COALESCE(canonical_id, id) FROM entities WHERE id = ?", (entity_id,)
    ).fetchone()
    return row[0] if row else entity_id
```

This is single-hop: A→B→C reports B, not C. The oracle's gap counts happen to be
"close enough" for the baseline because most chains are short, but as the graph grows
and chains lengthen (P2-P4), the miscount worsens.

**Our version uses `graph_db.root()`** — the shared helper from Phase 1 that follows
chains to the true canonical root, path-compresses, and survives cycles/dangling.

---

## 3. Architecture

```
  wiki/graph/  (SoT)
       │
       │  graph-build.py
       ▼
  .vault-meta/graph/graph.db  (derived, gitignored)
       │
       │  graph-gaps.py  ◀── THIS PHASE
       ▼
  5-species ranked gap report  (stdout / JSON)
```

- **Reads:** `.vault-meta/graph/graph.db` (derived) — can run anytime, safe
- **Never reads:** `~/.graphbuilding/graph.db` (live) — that's the oracle's job
- **Uses:** `graph_db.root()` — the one shared rollup helper
- **Zero imports from:** `~/.claude/skills/graphbuilding/` — no oracle dependency

---

## 4. Five gap species (identical to oracle)

| # | Species | What it finds | Key formula |
|---|---|---|---|
| 1 | Coverage | Canonical entities with 0 or very few claims | claim_count < avg×0.3 |
| 2 | Frontier | Author-flagged open-question / limitation claims | strength_weight × merge_conf |
| 3 | Debate | Opposite-polarity claims on same ⟨s,p,o⟩ | polarity_balance × avg_strength × merge_conf |
| 4 | Replication | Result claims with support=1 | strength_weight × 1/(support+1) × merge_conf |
| 5 | White-space | Entity clusters with no bridging claims (Louvain, seed=42) | size_balance × avg_merge_conf |

Same parameters as oracle:
- `MIN_COMMUNITY_SIZE = 5`
- `SPECIES_PRIORITY = ["frontier", "debate", "replication", "coverage", "white-space"]`
- `SPECIES_MULTIPLIER` (tight range 0.92–1.00 for tie-breaking)
- `STRENGTH_WEIGHT = {"strong": 0.9, "moderate": 0.6, "tentative": 0.3}`

---

## 5. Happy path

```
$ uv run python scripts/graph-gaps.py
# Gap Scan Report
**899 gaps found**, ranked by gap-confidence (highest first).
| 1 | 0.900 | 🚩 frontier | [open-question] How to scale attention to 1M tokens? |
| 2 | 0.882 | ⚡ debate | 3 assert vs 2 refute that 'Dropout' --[uses]--> 'Transformer' |
...

$ uv run python scripts/graph-gaps.py --json --top 10
[{"species": "frontier", "description": "...", "gap_confidence": 0.9, ...}, ...]

$ uv run python scripts/graph-gaps.py --db /tmp/copy.db   # custom db path
```

---

## 6. Users and roles

| Actor | Can do | Cannot do |
|---|---|---|
| Owner (human) | Run gap reports, pipe to JSON, limit with --top | — |
| `graph-gaps.py` | Read derived db, output ranked report | Write the derived db; read the live db |
| `graph_db.root()` | Chain-resolve entity ids for gap grouping | — |

---

## 7. Functional requirements

| ID | Requirement | Notes |
|---|---|---|
| FR1 | `scripts/graph-gaps.py` — native 5-species scanner | No oracle imports |
| FR2 | Uses `graph_db.root()` for all entity id resolution | Kills the COALESCE bug at source |
| FR3 | Reads `.vault-meta/graph/graph.db` by default | Derived, safe |
| FR4 | `--db` flag for custom path | Backward compat with tests |
| FR5 | `--json` flag for machine output | Same schema as oracle |
| FR6 | `--top N` flag to limit results | Default 20 |
| FR7 | `tests/test_graph_gaps.py` — counts match oracle on derived db | 899 total, per-species exact |
| FR8 | Falls back gracefully when networkx absent | Single placeholder gap, exit 0 |
| FR9 | `uv.lock` already committed; `pyproject.toml` already has networkx | No dep changes |

---

## 8. Data and integrations

| Data | R/W | Failure behavior |
|---|---|---|
| `.vault-meta/graph/graph.db` | R | Missing → "run graph-build.py first", exit 1 |
| `networkx` | dep | Missing → white-space returns 1 placeholder gap, rest work |
| `graph_db.root()` | import | Missing → import error, exit 1 |

---

## 9. Edge cases

| Scenario | Expected |
|---|---|
| Derived db missing | Exit 1 with message |
| Derived db empty (no entities) | 0 gaps, exit 0 |
| networkx not installed | White-space returns placeholder; other 4 species work normally |
| Single Louvain community | 0 white-space gaps (no cross-community pairs) |
| Graph too small for clustering (<4 nodes or <2 edges) | White-space fallback |
| 0 debate gaps (all claims agree) | Empty debate list (normal — current state) |
| Empty citation_links table | No impact (gap scanner doesn't touch it) |

---

## 10. Constraints

- **Must not break:** `graph_db.py`, `graph-build.py`, `graph-export.py`, round-trip test suite
- **Must not import from:** `~/.claude/skills/graphbuilding/`
- **Must not read:** `~/.graphbuilding/graph.db` (live — copy-only rule)
- **Performance:** <3s for 899 gaps on ~1050 claims (same ballpark as oracle)
- **Mutable surface:** `scripts/graph-gaps.py`, `tests/test_graph_gaps.py`
- **Read-only surface:** Everything else

---

## 11. Acceptance criteria

| ID | Criterion | Verification |
|---|---|---|
| AC1 | Gap counts match oracle exactly on derived db | `uv run python -m pytest tests/test_graph_gaps.py -q -k counts` |
| AC2 | Per-species counts: frontier 65, debate 0, replication 473, coverage 49, white-space 312 | Same test |
| AC3 | Uses `graph_db.root()` — no COALESCE(canonical_id, id) anywhere in graph-gaps.py | `grep -c "COALESCE" scripts/graph-gaps.py` == 0 |
| AC4 | Zero imports from `~/.claude/skills/graphbuilding/` | `grep -r "skills/graphbuilding" scripts/graph-gaps.py` exits 1 |
| AC5 | --json output schema matches oracle | `uv run python scripts/graph-gaps.py --json --top 5` produces valid JSON with species/description/entities/claims/gap_confidence/explanation |
| AC6 | networkx absent → graceful degradation | `uv run python -c "import graph_gaps"` doesn't crash without networkx |
| AC7 | Derived db missing → exit 1 with message | `uv run python scripts/graph-gaps.py --db /nonexistent` exits 1 |
| AC8 | P1 test suite still green | `uv run python -m pytest tests/test_graph_roundtrip.py -q` — 16 passed |
| AC9 | Full P2 suite green | `uv run python -m pytest tests/test_graph_gaps.py -q` |

**Metric.** Gap count parity with oracle: 899 total. Binary pass/fail — every species count must match exactly.

---

## 12. CAN / CANNOT

- **CAN modify:** `scripts/graph-gaps.py`, `tests/test_graph_gaps.py`, `feature_list.json`, `progress.md`
- **CANNOT modify:** `scripts/graph_db.py`, `scripts/graph-export.py`, `scripts/graph-build.py`, `tests/test_graph_roundtrip.py`, `wiki/graph/`, `pyproject.toml`
- **Needs human approval before changing:** Oracle `~/.claude/skills/graphbuilding/` files (read-only reference)

---

## 13. Out of scope

- Graph resolution / embedding dedup (P3)
- `skills/graph/*` plugin skills (P4)
- Live migration automation (P2 just commits the already-exported vault)
- Replacing the oracle's `resolve.py`, `validate.py`, `traces.py`, `bridge.py` (P3+)
- Unifying prose `wiki/entities/` with structured `wiki/graph/entities/`

---

## 14. Open questions (deferred, non-blocking)

- P3: Should `rerank.py` embeddings be used to re-rank gap results by semantic relevance?
- P3: Embedding dedup degrades gracefully when ollama absent.
- Future: Should gap reports be filed into `wiki/graph/_graph/gaps.md` as a committed artifact?

---

## Approval Checklist

- Domain lead approved: pending
- Tech lead approved: pending
- Critic reviewed testability: self-grilled
- Metric validated against user value: 899/899 gap count parity — binary, provable
