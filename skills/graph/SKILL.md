---
name: graph
description: "Claim-centric knowledge graph for the Compound Vault (v1.10 Graphbuilding Fusion). The claim is the atom — a typed, signed, quoted proposition — so the vault can answer questions no page-model organizer can: 'Paper A asserts X, Paper B refutes X', replication counts, and the five research-gap species (frontier / debate / replication / coverage / white-space). Markdown under wiki/graph/ is the source of truth; a sqlite index is derived and throwaway. Full-paper retrieval via /graph read uses BM25+rerank over wiki/graph/papers/*.full.md. Bridge proposals via /graph bridge (graph-bridge.py) rank white-space community pairs into justified next-paper candidates grounded in the real graph. Migrated natively from the standalone graphbuilding skill — zero oracle dependency. Triggers on: knowledge graph, claim graph, research gaps, what should I study next, find connections across papers, cross-paper, bridge proposals, next paper, white-space bridge, ingest this paper into the graph, graph build, graph gaps, graph resolve, graph read, graph bridge, read paper, retrieve passage, full text, duplicate entities, dedup entities, gap scan, my graph, /graph."
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
   papers/<slug>.md        title / authors / sections(+summaries)
   papers/<slug>.full.md   verbatim full text (P4; written by graph-fulltext.py sync)
   entities/<name>__e<id>.md   super_type, sub_type, canonical pointer, aliases  (body = description)
   claims/c<id>.md         the queryable triple + provenance; body = statement + > [!quote] + [[links]]
   _graph/                 predicates, entity_edges, citation_links (aux tables)
   SCHEMA.md               the frontmatter contract (incl. .full.md contract + add/re-add path)
   graph-export.json       portable full snapshot (tracked)
        │  scripts/graph-build.py   (frontmatter → rebuildable index)
        ▼
 .vault-meta/graph/graph.db   ← DERIVED, gitignored, deletable at any time
        │  scripts/graph-gaps.py · scripts/graph-resolve.py
        ▼
   five gap species · replication rollup · communities · entity-resolution proposals

 .vault-meta/graph/chunks/  ← DERIVED, gitignored (graph BM25 chunk store)
 .vault-meta/graph/bm25/    ← DERIVED, gitignored (graph BM25 index)
        │  scripts/graph-fulltext.py sync  (builds from .full.md)
        │  scripts/graph-retrieve.py       (/graph read backend)
        ▼
   BM25 + rerank retrieval over full-paper text · provenance per passage
```

`wiki/graph/entities/` (structured, typed) is deliberately **separate** from `wiki/entities/`
(prose pages). Merging them would flatten the claim layer back into an organizer — the one
thing the fusion exists to prevent.

---

## The five operations

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

### 4. Read full-text passages from indexed papers
`/graph read` retrieves ranked full-text passages via BM25 + rerank over `wiki/graph/papers/*.full.md`.
Run `graph-fulltext.py sync` first to import Tier-A papers and build the index.
```bash
# Import Tier-A papers + build graph BM25 index (idempotent)
uv run python scripts/graph-fulltext.py sync

# Free-text query across all indexed papers
uv run python scripts/graph-retrieve.py "constrained sampling garment pinning" --top 5

# Retrieve passages from one paper's full text
uv run python scripts/graph-retrieve.py --paper <slug>

# Trace a claim to its source paper, retrieve passages
uv run python scripts/graph-retrieve.py --claim <id> --export wiki/graph/graph-export.json
```
Output JSON: `candidates` array with `page_path`, `page_address`, `chunk_index`,
`bm25_score`, `rerank_score`, `rerank_source`, `snippet`.

Tier-B papers (PDFs without an extracted `.md`) are skipped with a log line and listed as
the "ready to add later" backlog. Once extracted, register the path in `graph-export.json`
and re-run sync (idempotent).

### 5. (Re-)export the live graph to markdown (one-time / re-derivation)
One-time migration / re-derivation from a sqlite DB into `wiki/graph/` markdown + the JSON
snapshot. Reads a **copy** — never mutates the source.
```bash
uv run python scripts/graph-export.py <source.db> wiki/graph
```

### 6. Find bridge proposals — turn white-space into next-paper candidates (P5)
`/graph bridge` (scripts/graph-bridge.py) ranks the 312 white-space community pairs into
justified "connect A ↔ B because …" proposals grounded in the real graph entities and papers.

```bash
# Ranked proposals (markdown report, default top-10)
uv run python scripts/graph-bridge.py --top 10

# JSON output for scripting
uv run python scripts/graph-bridge.py --json --top 10

# Narrative justification via claude-CLI (opt-in egress only)
uv run python scripts/graph-bridge.py --json --top 10 --synthesize
```

**Output per proposal:** `id`, `community_a/b` (id, size, top members by degree),
`anchor_entities` (most-connected per side), `anchor_papers`, `score`,
`signal_breakdown` (gap_confidence, bridgeability, limitation_pull, richness,
direction_relevance), `passages` (full-text snippets from graph-retrieve, degrades
to "no full text" when index is absent), `already_proposed` (FR7: flags if a
known bridge entity is already in a community), optional `justification`.

**Gold anchor:** the VTON-community ↔ diffusion-sampling/distillation-community
bridge surfaces near the top because `direction_relevance=1.0` when one community
contains virtual try-on entities and the other contains Neon/DNO/distillation entities.
This encodes [[research-goal-phd-paper]] into the ranking.

**Weights** default to `{gap_confidence: 0.25, bridgeability: 0.20, limitation_pull: 0.15,
richness: 0.15, direction_relevance: 0.25}`; retune at runtime via `--w-<signal>`.

**Egress posture:** zero egress by default. `--synthesize` is the only opt-in path (uses
`contextual-prefix.pick_prefix_tier`; falls back to structured template if claude is absent).

**Tests:** `make test-bridge` or `uv run python -m pytest tests/test_graph_bridge.py -q`.

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
uv run python -m pytest tests/test_graph_roundtrip.py tests/test_graph_gaps.py tests/test_graph_resolve.py tests/test_graph_validate.py -q
# or: make test-graph  (48 tests: roundtrip + gaps + resolve + validate)

# P4 full-paper retrieval suite (AC1–AC9):
uv run python -m pytest tests/test_graph_fulltext.py -q
# or: make test-fulltext

# P5 bridge proposals suite (AC1–AC8 + 100-case stress):
uv run python -m pytest tests/test_graph_bridge.py -q
# or: make test-bridge
```
Round-trip (`test_graph_roundtrip.py`) is the acceptance oracle: copy a live DB → export →
build → per-row `SELECT *` diff of all 9 tables + 5-species gap diff (run through the
**independent** oracle scanner) + source-md5-untouched.

`test_graph_fulltext.py` covers: resolver byte-equal import, skip-bodyless, idempotent
re-run, gitignore policy, zero-egress default, BM25 index build, retrieve provenance,
ollama-absent degrade, and the --paper/--claim read surface.

`test_graph_bridge.py` covers: deterministic ranking, grounding integrity, gold anchor
(VTON ↔ Aek), zero-egress default, graceful degradation, JSON schema, wiring, and
a 100-case stress harness (>=95% well-formed + grounded + non-degenerate + no crash).

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
