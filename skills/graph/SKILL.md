---
name: graph
description: "Claim-centric knowledge graph for the Compound Vault (v1.10 Graphbuilding Fusion). The claim is the atom — a typed, signed, quoted proposition — so the vault can answer questions no page-model organizer can: 'Paper A asserts X, Paper B refutes X', replication counts, and the five research-gap species (frontier / debate / replication / coverage / white-space). Markdown under wiki/graph/ is the source of truth; a sqlite index is derived and throwaway. Migrated natively from the standalone graphbuilding skill — zero oracle dependency. Triggers on: knowledge graph, claim graph, research gaps, what should I study next, find connections across papers, cross-paper, ingest this paper into the graph, graph build, graph gaps, graph resolve, duplicate entities, dedup entities, gap scan, my graph, /graph."
allowed-tools: Read Bash
---

# graph: Claim-Centric Knowledge Graph

**Origin (v1.10 "Graphbuilding Fusion"):** this skill is the in-repo, native rebuild of the
standalone `graphbuilding` skill (formerly at `~/.claude/skills/graphbuilding`, backed by a
scattered `~/.graphbuilding/graph.db` dotdir). The engine now lives in `scripts/graph-*.py`
with **zero imports from the oracle** and reuses this repo's plumbing (pytest harness,
`rerank.py` embeddings, gitignore policy). See
[`docs/graphbuilding-fusion-design.md`](../../docs/graphbuilding-fusion-design.md) for the
design and [`docs/graphbuilding-fusion-migration-report.md`](../../docs/graphbuilding-fusion-migration-report.md)
for the verification evidence.

**Why a claim graph and not pages.** The `/wiki` family is an *organizer* — excellent at
prose pages, links, and retrieval. It cannot represent "Paper A asserts X, Paper B refutes
X," compute how many papers replicate a finding, or surface where the literature is silent.
That requires the **claim** as the atom: a typed `subject —predicate→ object` triple with
polarity (asserts/refutes), support count, and a verbatim quote. This skill carries that
layer; it does **not** replace the page wiki.

---

## Architecture — markdown is truth, sqlite is throwaway

```
 wiki/graph/   ← SOURCE OF TRUTH (structured markdown, git-tracked, Obsidian-readable)
   papers/<slug>.md       title / authors / sections(+summaries)
   entities/<name>__e<id>.md   super_type, sub_type, canonical pointer, aliases  (body = description)
   claims/c<id>.md        the queryable triple + provenance; body = statement + > [!quote] + [[links]]
   _graph/                predicates, entity_edges, citation_links (aux tables)
   SCHEMA.md              the frontmatter contract
   graph-export.json      portable full snapshot (tracked)
        │  scripts/graph-build.py   (frontmatter → rebuildable index)
        ▼
 .vault-meta/graph/graph.db   ← DERIVED, gitignored, deletable at any time
        │  scripts/graph-gaps.py · scripts/graph-resolve.py
        ▼
   five gap species · replication rollup · communities · entity-resolution proposals
```

`wiki/graph/entities/` (structured, typed) is deliberately **separate** from `wiki/entities/`
(prose pages). Merging them would flatten the claim layer back into an organizer — the one
thing the fusion exists to prevent.

---

## The four operations

All scripts run under `uv` (PyYAML + networkx are declared in `pyproject.toml`). The derived
DB defaults to `.vault-meta/graph/graph.db`; override with `--db`.

### 1. Build the index from markdown
Rebuild the throwaway sqlite index from the markdown source of truth. Safe to run any time;
the DB is disposable.
```bash
uv run python scripts/graph-build.py wiki/graph .vault-meta/graph/graph.db
```

### 2. Scan for research gaps
The five-species scanner (Louvain communities are `seed=42` deterministic). Returns ranked
gaps; `--json` for machine output, `--top N` to cap.
```bash
uv run python scripts/graph-gaps.py --top 20          # human report
uv run python scripts/graph-gaps.py --json --top 50   # JSON
```
Species: **frontier** (active edges of a community), **debate** (asserts vs refutes on the
same triple), **replication** (single-support results begging confirmation), **coverage**
(thin entities), **white-space** (disconnected community pairs).

### 3. Resolve duplicate entities
Two-tier dedup: Tier 1 exact-name, Tier 2A embedding similarity (reuses `rerank.py` /
ollama nomic-embed), Tier 2B token-Jaccard fallback when ollama is absent. **Proposes only —
a human confirms merges.** Uses `graph_db.root()`, never `COALESCE`.
```bash
uv run python scripts/graph-resolve.py --json     # ranked merge proposals
```

### 4. (Re-)export the live graph to markdown
One-time migration / re-derivation from a sqlite DB into `wiki/graph/` markdown + the JSON
snapshot. Reads a **copy** — never mutates the source.
```bash
uv run python scripts/graph-export.py <source.db> wiki/graph
```

---

## Write-side invariants (so drift cannot re-accumulate)

The original skill's reads were forced to repair-then-recompute because the *write* path
never held the invariants the read path assumed. The native engine bakes them in:

- **One `root()` helper** (`scripts/graph_db.py`) — path-compressing, cycle- and
  dangling-safe. Replaces every inline `COALESCE(canonical_id, id)` (which was single-hop and
  wrong on chains). Shared by every script.
- **Claims never roll up / never merge** — `subject_id`/`object_id` are the entity *as
  extracted*; resolution happens at query time via `root()`, not by rewriting claims.
- **FK-on everywhere**; the derived `aliases` table tolerates dangling-FK rows so the export
  is lossless (the 834-alias round-trip bug is fixed).
- **Integrity guard** — `scripts/graph-validate.py` reports the four drift species
  (dangling / chain / self-loop / orphan) and `--heal` fixes the pointer-drift class via
  `root()` (never throws). The pre-commit `verifier` agent runs it as a read-only gate so
  drift can never be committed.
```bash
uv run python scripts/graph-validate.py            # report (exit 1 if drift)
uv run python scripts/graph-validate.py --heal     # fix in place, then re-report
```

---

## Graceful degradation

- **No ollama** → resolver silently uses Tier 2B token-Jaccard; gaps/build/export are
  unaffected (no embedding dependency).
- **No derived DB** → `graph-gaps.py` / `graph-resolve.py` exit 1 with a clear message; run
  `graph-build.py` first.
- **networkx required** for gap communities; it ships in `pyproject.toml`.

---

## Tests

```bash
uv run python -m pytest tests/test_graph_roundtrip.py tests/test_graph_gaps.py tests/test_graph_resolve.py -q
# or: make test-graph
```
Round-trip (`test_graph_roundtrip.py`) is the acceptance oracle: copy a live DB → export →
build → per-row `SELECT *` diff of all 9 tables + 5-species gap diff (run through the
**independent** oracle scanner) + source-md5-untouched.

---

## How to think (10-principle map)

- **OBSERVE the structure, not the prose.** Before answering a cross-paper question, read the
  claim triples and polarity — not the entity descriptions. The answer to "is X contested?"
  is `debate`-species claims, not a paragraph.
- **LISTEN for the gap the user names.** "What should I study next?" → `frontier` +
  `white-space`. "What's unconfirmed?" → `replication`. Map the ask to the species; don't
  dump all 899.
- **CONNECT, don't merge.** When two entities look identical, *propose* a resolution and let
  the human confirm — a wrong merge silently corrupts every downstream gap count.
- **ACCEPT the extractor is noisy.** Markdown makes bad claims visible and hand-fixable; that
  is the point of markdown-as-truth. Surface the quote so the user can judge it.
- **GROW the index, never trust it as truth.** The sqlite DB is derived. If a number looks
  wrong, rebuild from markdown and re-check — don't patch the DB.
