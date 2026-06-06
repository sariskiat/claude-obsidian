# Next feature — full-paper retrieval for /graph (handoff)

**Status:** chosen, NOT yet grilled/specced/built. This is the foundation feature picked after the v1.10 Graphbuilding Fusion shipped. **No code until the spec is human-approved (grill first).**

## One-line goal
Make graph traversal able to read the *full text* of related papers — and consolidate the scattered `~/.paper-scholar` full-text into the vault — so "trace a claim/gap → read the related paper → answer" works, and the dotdir can finally be retired.

## Read these first (in order)
1. `docs/graphbuilding-fusion-migration-report.md` — what's done + verified (v1.10).
2. `docs/graph-workflow.html` (or describe it) — the current paper→graph pipeline + the named gaps.
3. `docs/graphbuilding-fusion-design.md` §6 — the "retrieval indexes claim bodies" synergy this feature realizes.
4. `wiki/graph/SCHEMA.md` — the papers/entities/claims frontmatter contract.
5. Memory: `research-goal-phd-paper`, `graphbuilding-fusion-status` (why this matters).

## What already exists (reuse, don't reinvent)
- Engine: `scripts/graph_db.py` (`root()`), `graph-build.py`, `graph-gaps.py`, `graph-resolve.py`, `graph-validate.py`.
- Retrieval stack ALREADY shipped: `scripts/retrieve.py`, `scripts/bm25-index.py`, `scripts/rerank.py` (BM25 + ollama cosine rerank), `scripts/contextual-prefix.py`.
- Source data: `~/.paper-scholar/<slug>/` holds per paper: full-text `<arxivid>.md` (~150 KB), `_meta.json`, `paper.json`, and `_page_*.jpeg` figures. 37 papers, ~56 MB. **No PDFs** (those are trashed).
- Today `wiki/graph/papers/<slug>.md` carries only per-section *summaries* in frontmatter — NOT full text.

## Grill must close these decisions (one question at a time)
1. **Storage** — full text as the *body* of `papers/<slug>.md`, or a sibling `papers/<slug>.full.md`? (graph-build parses paper frontmatter; a 150 KB body may want to be a sibling so the build parser stays clean.)
2. **Figures** — bring `_page_*.jpeg` into `_attachments/` (git-lfs? gitignore?) or skip (text-only)?
3. **Retrieval wiring** — extend the existing `wiki/` BM25/rerank index to cover `wiki/graph/papers/`, or a separate graph index?
4. **Query surface** — new `/graph read <claim|paper>` path, or fold into `wiki-retrieve`?
5. **Source-of-truth & retirement** — import from `~/.paper-scholar` once and retire it (like the oracle), or keep it as upstream?

## Hard constraints (from CLAUDE.md + design)
- Markdown is SoT; sqlite stays derived/throwaway. `wiki/graph/entities/` (typed) stays SEPARATE from `wiki/entities/` (prose) — never merge.
- TDD: failing test first. Run `uv` for everything. One vertical slice.
- Branch: currently `feature/graph-resolve-dedup` (clean). Start a fresh `feature/graph-full-paper-retrieval` branch.
- Verifier before commit; `make test-graph` must stay green (48 tests).
