# Spec: Full-Paper Retrieval for /graph

**Feature ID:** graph-full-paper-retrieval
**Status:** `approved`
**Grilled on:** 2026-06-06
**Approved by:** `saris` on 2026-06-06
**Problem classification:** `unique`

## Problem Statement

Graph traversal can surface a claim or gap but cannot open the related paper's full text, so "trace a claim/gap → read the paper → reason toward my next contribution" dead-ends at the section *summaries* already in `papers/<slug>.md`.

## Objective

Make the claim graph able to pull the **full text** of related papers, and do it as a **reusable, system-ready pipeline** — so the 38 papers that already have full-text markdown flow in now, and any paper added or re-added later (the 57 PDFs once extracted, or brand-new papers) drops into the same full-context tracing with no rework. This serves the research goal ([[research-goal-phd-paper]]): next-paper proposals and toy models grounded in the *actual* text of related work, not summaries.

## User Context

- **Primary user:** saris (single-tenant fork owner). No other actors.
- **Environment:** macOS, `uv`-run Python 3.12, Obsidian vault = this repo, `/graph` skill (v1.10 Graphbuilding Fusion).
- **Current workflow:** `/graph gaps` → a gap names entities/claims → user must leave the tool and hunt for the paper's text by hand (scattered across `~/.paper-scholar`, `~/Desktop/research/`).
- **Pain points:** full text is scattered across ≥4 disk locations; `papers/<slug>.md` carries only summaries; no way to rank passages across papers for a given gap.
- **Stakeholders consulted:** saris (grill, 2026-06-06).

## JTBD

When I trace a claim or research gap, I want to open the related paper's **full text** and reason with full context — and I want a repeatable system so any paper I add or re-add later flows into that same full-context tracing — so I can derive next-paper proposals and proof-of-concept toy models grounded in real text.

## Users and Roles

| Actor | Can do | Cannot do |
|---|---|---|
| saris (owner) | run import, build index, `/graph read`, re-run idempotently | n/a (single-tenant) |
| graph-build.py (system) | rebuild derived index from `graph-export.json` — **unaffected** by `.full.md` | read/ingest `.full.md` |
| graph-retrieve (system) | chunk + index `.full.md`, answer `/graph read` | mutate the markdown SoT or source dirs |

## Functional Requirements

| ID | Requirement | Notes |
|---|---|---|
| FR1 | A **resolver** maps each in-graph paper to its best full-text source via `source_path` (absolute paper-scholar `.md`; or relative `.paper-scholar/<dir>` → inner `.md`). | Tier A only this slice. Resolver is structured so future source types register without rewriting callers (system-ready). |
| FR2 | For each resolved (Tier-A) paper, write a sibling `wiki/graph/papers/<slug>.full.md` = minimal frontmatter + **verbatim** body (byte-equal copy of the source `.md`). | ~38 files. Idempotent: re-run overwrites deterministically. |
| FR3 | A **graph-scoped retrieval index** (separate chunk store + BM25) is built over the `.full.md` bodies only — not over claim/entity stubs or the structured `papers/*.md`. | Reuses `contextual-prefix.py` / `bm25-index.py` / `rerank.py` machinery pointed at graph paths. |
| FR4 | Contextual prefixes default to **synthetic (zero egress)**; richer prefixes are opt-in via `--allow-egress` using the **claude-CLI** tier (subscription, no API key). | Mirrors `setup-retrieve.sh` default-deny posture; closes any B1-class egress gap. |
| FR5 | `/graph read` answers three forms: `<natural-language query>`, `--paper <slug>`, `--claim <id>` → ranked full-text passages with paper+claim provenance. | Backend `scripts/graph-retrieve.py` (BM25 + local rerank); the design's `graph-query` surface. |
| FR6 | The 60 bodyless papers (57 Tier-B PDFs + 3 non-papers) are **skipped gracefully** — no body, no crash, no fabricated text — and documented as the backlog the system is ready to ingest later. | |
| FR7 | `wiki/graph/SCHEMA.md` documents the `.full.md` contract **and** the "add / re-add a paper" path, so the system-ready aim is captured, not just coded. | |

## Happy Path

1. `uv run python scripts/graph-fulltext.py sync` → resolver finds the ~38 Tier-A papers → writes `papers/<slug>.full.md` siblings → builds the graph chunk store + BM25 index. *(idempotent)*
2. `/graph gaps` surfaces a gap citing claim `c123` / entity X.
3. `/graph read --claim c123` → traces claim → source paper → returns top-K full-text passages from that paper (+ related papers) with provenance.
4. `/graph read "constrained sampling under hard garment pinning"` → ranked passages across all indexed papers.
5. Later: a new paper's `.full.md` is added (or a Tier-B PDF is extracted) → re-run step 1 → instantly retrievable. *(the system-ready payoff)*

## Business Rules

| Rule ID | Rule | Source/Rationale | Verification |
|---|---|---|---|
| BR1 | Source dirs (`~/.paper-scholar`, `~/Desktop/research/`) are **read-only**; never mutated or deleted. | Re-add-later aim needs the source intact. | grep: no write/unlink to those paths |
| BR2 | `.full.md` body is a **verbatim** copy of the source `.md` (no reflow, no truncation). | Faithful full text is the whole point. | byte-equal assert in test |
| BR3 | Import + index build are **idempotent** — re-running yields identical `.full.md` + index. | Enables "re-add later" / repeatable. | run twice, diff = empty |
| BR4 | `graph-build.py` is **not modified**; `.full.md` are inert to it (it reads `graph-export.json`). The 9-table round-trip stays byte-equal. | Protect the 48-test moat. | `make test-graph` green |
| BR5 | Papers are **never merged**: duplicate graph slugs pointing at one source each get their own `.full.md` by slug. | Consistent with "claims never merge." | per-slug file assert |
| BR6 | Retrieval index is **derived/throwaway** (gitignored, rebuildable); `.full.md` are git-tracked SoT. | Same philosophy as `graph.db`. | `git check-ignore` test |
| BR7 | Default run performs **zero network egress**; egress only with explicit `--allow-egress`. | Default-deny; B1 retrospective. | no-network assert in default path |

## Data and Integrations

| Data/API | Owner | Read/Write | Failure behavior |
|---|---|---|---|
| `~/.paper-scholar/<slug>/<arxivid>.md` (Tier-A source) | upstream | **Read-only** | stale/missing path → skip paper, log, continue |
| `wiki/graph/papers/<slug>.full.md` | this feature | Write (idempotent) | n/a |
| `.vault-meta/graph/chunks/`, `.vault-meta/graph/bm25/index.json` | this feature | Write (derived, gitignored) | missing on read → friendly exit (mirror retrieve.py exit 10) |
| `rerank.py` (ollama / nomic-embed-text) | existing | Read (reuse) | ollama absent → BM25-only, `rerank_source: noop-no-ollama`, exit 0 |
| `contextual-prefix.py` / `bm25-index.py` functions | existing | Reuse | reuse functions; do not break wiki-retrieve defaults |

`.full.md` frontmatter contract:
```yaml
type: paper-fulltext
slug: <slug>
arxiv_id: <id|null>
source_path: <resolved absolute path>
paper: "[[<slug>]]"
```

## Errors and Edge Cases

| Scenario | Expected behavior |
|---|---|
| `source_path` stale / file gone | Skip that paper, log one line, continue; counts reported at end |
| Tier-B (PDF) or Tier-C/none paper | Skipped (no `.full.md`); listed in the "ready to add later" report |
| ollama unreachable | rerank no-ops to BM25 order; exit 0 with note |
| graph index missing on `/graph read` | Non-zero exit + "run `graph-fulltext.py sync` first" hint |
| `--allow-egress` not passed | Synthetic prefixes only; no network calls |
| empty query / unknown slug / unknown claim-id | Friendly message, non-zero exit |
| duplicate slug → same source | Each slug gets its own `.full.md`; no merge |
| re-run after no source change | Byte-identical output (idempotent) |

## Constraints

- **Must not break:** `make test-graph` (48 tests) green; 9-table round-trip byte-equal; existing `retrieve.py` / wiki-retrieve behavior unchanged; `wiki/entities/` ↔ `wiki/graph/entities/` stay separate; markdown SoT + derived/throwaway index; `uv` for all Python; TDD failing-test-first; verifier before each commit.
- **Performance:** import + index of 38 papers completes in well under a minute on the synthetic (zero-egress) path.
- **Security/compliance:** zero egress by default; opt-in egress (claude-CLI tier) sends only public arXiv text, behind an explicit flag.
- **Mutable surface:** `scripts/graph-fulltext.py` (new), `scripts/graph-retrieve.py` (new), `tests/test_graph_fulltext.py` (new), `wiki/graph/papers/*.full.md` (generated), `commands/graph.md` (+`read`), `skills/graph/SKILL.md` (+`read` routing), `wiki/graph/SCHEMA.md` (+`.full.md` contract + add/re-add path), `.gitignore` (graph chunks/bm25), `Makefile` (optional `test-graph` extension). Minimal, backward-compatible reuse of `contextual-prefix.py` / `bm25-index.py` functions.
- **Read-only surface:** `~/.paper-scholar/**`, `~/Desktop/research/**` (never written/deleted); `graph_db.py`, `graph-build.py`, `graph-gaps.py`, `graph-resolve.py`, `graph-validate.py`, `retrieve.py`, `rerank.py` (reuse, don't regress).

## CAN / CANNOT

- **CAN modify:** the new scripts/tests, generated `.full.md`, `commands/graph.md`, `skills/graph/SKILL.md`, `wiki/graph/SCHEMA.md`, `.gitignore`, `Makefile`.
- **CANNOT modify:** `graph-build.py` / `graph-export.py` / `graph_db.py` / gap+resolve+validate scripts; the round-trip + gaps + resolve + validate tests' expectations; the source dirs; `retrieve.py` behavior/defaults.
- **Needs human approval before changing:** retiring/deleting any source dir (default this slice = **KEEP** as upstream).

## Acceptance Criteria

| ID | Criterion | Verification |
|---|---|---|
| AC1 | Resolver + import writes one `.full.md` per Tier-A paper (~38), each a byte-equal copy of its source body with the contract frontmatter. | `uv run python -m pytest tests/test_graph_fulltext.py -q -k import` |
| AC2 | The 60 bodyless papers are skipped (no `.full.md`, no crash) and reported as the add-later backlog. | `pytest -k skip_bodyless` |
| AC3 | Graph retrieval index builds over `.full.md` bodies only; a seeded query returns the correct source paper's passage in top-K with provenance. | `pytest -k retrieve_provenance` |
| AC4 | Default run does **zero network egress** (synthetic prefixes); `--allow-egress` is the only path that egresses. | `pytest -k no_egress_default` |
| AC5 | ollama absent → rerank degrades to BM25-only, exit 0; missing index → friendly non-zero exit. | `pytest -k degrade` |
| AC6 | Import + index build are idempotent (run twice → identical `.full.md` + index). | `pytest -k idempotent` |
| AC7 | Retrieval index is gitignored; `.full.md` are git-tracked. | `bash -c 'git check-ignore .vault-meta/graph/bm25/index.json && ! git check-ignore wiki/graph/papers/*.full.md'` |
| AC8 | **No regression:** `make test-graph` stays green (48 tests); 9-table round-trip byte-equal. | `make test-graph` |
| AC9 | `/graph read` answers `<query>`, `--paper <slug>`, `--claim <id>`; `SCHEMA.md` documents the `.full.md` contract + the add/re-add path. | `pytest -k read_surface` + `grep -qi 'full.md\|add a paper\|re-add' wiki/graph/SCHEMA.md` |

## Metric

- **Primary metric:** in-graph papers whose full text is retrievable via `/graph read` (with a passing provenance spot-check).
- **Baseline:** 0 (no full text indexed; only summaries exist).
- **Target:** 38/38 Tier-A papers imported + indexed + a seeded claim/query returns its source paper's passage in top-K; import+build idempotent.
- **Direction:** increase (binary pass at the target).
- **Why the user will notice:** today, gap/claim → full paper text is impossible; after, it's one `/graph read`, and adding the next paper makes it instantly traceable.

## Success Signal

- **Completion promise:** Slice 1 delivers a system-ready full-paper retrieval pipeline — 38 papers traceable to full text, with adding/re-adding papers a first-class documented path.
- **Required evidence:** `make test-graph` green (48); `tests/test_graph_fulltext.py` green; a live `/graph read --claim <id>` transcript showing full-text passages + provenance; an idempotent re-run diff (empty).
- **Expected final marker:** all ACs verified with fresh command output; verifier run clean before each commit.

## Out of Scope

- **Tier-B PDF extraction (57 PDFs incl. the VTON corpus)** — the very next slice; this slice makes the system *ready* for it (resolver + `.full.md` contract not paper-scholar-only) but does not build the pdf→markdown front-end.
- **Figures** (`_page_*.jpeg` → `_attachments/`) — text-only retrieval this slice; noted as future.
- **Retiring/deleting `~/.paper-scholar` or `~/Desktop/research/` PDFs** — kept as upstream source (the re-add-later aim needs them).
- **`graph-bridge.py` proposal synthesis** — separate queued feature (`docs/next-proposal-finder.md`).
- **Marker-quality LaTeX extraction** (~22 min/paper, infeasible at 57).
- **Merging `wiki/entities/` (prose) with `wiki/graph/entities/` (structured).**

## Approval Checklist

- Domain lead approved: saris (2026-06-06)
- Tech lead approved: saris (2026-06-06)
- Critic reviewed testability: pending (Planner/Evaluator)
- Metric validated against user value: yes (0 → 38 traceable; user notices gap→full-text)

## Open Questions

- None blocking. Three defaults set during grill, flagged for explicit veto at approval:
  1. **Keep (not retire)** `~/.paper-scholar` + research PDFs as upstream — overrides handoff Q5's "retire like the oracle." (Rationale: re-add-later aim.)
  2. **Skip figures** this slice (text-only).
  3. Graph retrieval index is **derived/gitignored**; `.full.md` git-tracked.
