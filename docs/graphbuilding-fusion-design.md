# Graphbuilding Fusion — Design Spec

**Status:** draft for approval. No code lands until this is approved.
**Author/owner:** saris (fork owner)
**Date:** 2026-06-06
**Decision basis:** `~/Desktop/research/` analysis set —
[`graph-skill-debt.md`](file:///Users/saris.kia.adm/Desktop/research/graph-skill-debt.md),
[`graph-skill-redesign.md`](file:///Users/saris.kia.adm/Desktop/research/graph-skill-redesign.md),
[`graph-vs-obsidian-graphify.md`](file:///Users/saris.kia.adm/Desktop/research/graph-vs-obsidian-graphify.md),
[`claude-obsidian-review.md`](file:///Users/saris.kia.adm/Desktop/research/claude-obsidian-review.md).

---

## 1. Purpose

Fuse the **claim-centric gap-finder** (the `graphbuilding` skill) into **this repo's
engineering** (`claude-obsidian`), so one plugin gives both:

- **The moat** (from graphbuilding): the *claim* is the atom — a typed, signed, quoted
  proposition — which is the only representation that can answer "Paper A asserts X, Paper
  B refutes X," compute replication counts, and surface the five gap species. No page-model
  organizer (Obsidian, Graphify, this repo's current `/wiki`) can represent that question.
- **The engineering** (from this repo): markdown source-of-truth, transport fallback,
  per-file locking, BM25 + embedding retrieval, the `hot.md`/`index.md` cache, methodology
  modes, the pre-commit `verifier` agent, a real test harness, and plugin packaging.

The decisive review finding (`claude-obsidian-review.md`): this repo is the more mature
**organizer**, but it is *not* a gap finder — "no claim layer… no cross-source
aggregation… polarity is not queryable." The fusion keeps the moat as a **separate
structured layer** and lets the existing engineering carry it.

**The failure mode to avoid:** letting claims decay into prose wiki pages. If the claim
layer becomes editorial prose like `wiki/entities/*.md`, we *become* the organizer and
lose gap-finding. The structured frontmatter triple is non-negotiable.

---

## 2. The two halves being fused

| From graphbuilding (keep — the moat) | From this repo (reuse — the engineering) |
|---|---|
| Claim atom: `subject —predicate→ object`, never merged | Markdown vault as SoT, git-native |
| 6 entity super-types (Concept/Method/Artifact/Task/Measure/Source) | `.vault-meta/transport.json` fallback (cli→mcp→fs) |
| Typed predicates with domain/range enforced at insert | `scripts/wiki-lock.sh` advisory per-file locks |
| 5 gap species (frontier/debate/replication/coverage/white-space) | `scripts/retrieve.py` = BM25 + ollama rerank |
| Two walls: quote-in-source, triple-in-quote | `hot.md`/`index.md` tiered cache |
| Polarity (asserts/refutes), support count, confirmed/proposed | `scripts/wiki-mode.py` methodology routing |
| `canonical_id` entity resolution | `agents/verifier.md` pre-commit review |
| sqlite analytics engine | pytest harness + `tests/` |

Neither half is discarded. The graph layer *gains* this repo's plumbing; the page wiki is
untouched.

---

## 3. Architecture — markdown SoT, sqlite as a derived index

```
 wiki/graph/   (SOURCE OF TRUTH — structured markdown, git-tracked, Obsidian-readable)
   papers/<slug>.md        frontmatter: title/authors/sections(+summaries)
   entities/<slug>.md      frontmatter: super_type, sub_type, canonical pointer, aliases
                           body: description   (STRUCTURED — not the prose wiki/entities/)
   claims/c<id>.md         frontmatter: the queryable triple + provenance metadata
                           body: statement + > [!quote] verbatim + [[wikilinks]] + LaTeX
   _graph/                 predicates.md, entity_edges.md, citation_links.md (aux tables)
   SCHEMA.md               the frontmatter contract
        |  graph-build.py  (read frontmatter → rebuildable index)
        v
 .vault-meta/graph/graph.db   (DERIVED, gitignored, throwaway)
        |  graph-gaps.py / graph-resolve.py
        v
   five gap species · support rollup · communities · entity resolution
```

**Why markdown is SoT and sqlite is throwaway** (from `graph-skill-redesign.md` and the
direction in `memory2.md`): claims from a weak extractor may be noisy, so the
human-readable markdown (verbatim quote + equations + example) is the trustworthy,
hand-fixable, re-extractable source. The sqlite index is rebuilt from it on demand and can
be deleted at any time. **Round-trip already proven** (see §9).

**Critical separation:** `wiki/graph/entities/` (structured claim-layer entities) is
distinct from the existing `wiki/entities/` (prose pages). They are different models and
must not be merged in this phase — that is the moat-protection decision.

---

## 4. wiki/graph/ frontmatter schemas (the contract)

Authoritative copy ships as `wiki/graph/SCHEMA.md`. Summary:

**`claims/c<id>.md`** — frontmatter is the queryable triple; body is readable evidence.
```yaml
type: claim
id: 42
subject_id: 10        # the entity AS EXTRACTED — claims never roll up / never merge
predicate: evaluates-on
object_id: 7
claim_type: result|definition|proposal|limitation|open-question|...
polarity: asserts|refutes      # queryable conflict signal — NOT a prose callout
strength: strong|moderate|tentative
support: 3            # # distinct papers asserting equivalent <s,p,o>  (replication signal)
status: confirmed|proposed
source_paper: <slug>
section_id: 88
generated_by: claude-opus-4-8@v1
subject: "[[foo__e10]]"
object: "[[bar__e7]]"
```
Body markers parsed on build: statement (text above the callout), then the verbatim quote
inside a `> [!quote]` callout (the provenance wall). LaTeX `$$…$$` and worked examples are
readable-only — to make a new relation queryable you must promote it into the typed
frontmatter.

**`entities/<name>__e<id>.md`** — typed identity + canonical pointer + aliases in
frontmatter; description in body. Filename: name leads (Obsidian-readable), `__e<id>`
guarantees uniqueness (the VITON-HD-×3 duplication problem means names are *not* unique).

**`papers/<slug>.md`** — title/authors/source_path/ingested_at + a `sections` list (each
with id/heading/role/order_index/summary; section ids preserved because claims reference
them).

---

## 5. Deep native rebuild — module map

Decision: **reimplement the engine natively on this repo's conventions**, using the
existing `~/.claude/skills/graphbuilding/scripts/*` as the *reference spec and test oracle*
(not as code to copy verbatim). New modules follow this repo's `scripts/` style and wire
into its plumbing.

| New module (this repo) | Responsibility | Reference oracle | Wires into |
|---|---|---|---|
| `scripts/graph-db.py` | `connect()`, **`root()`** (path-compress, cycle/dangling-safe), typed insert helpers, FK-on | `graphbuilding/scripts/db_helpers.py` | — |
| `scripts/graph-build.py` | vault → derived `graph.db`; preserves ids; self-healing on read | the proven `graph_import.py` | `wiki-lock`, transport |
| `scripts/graph-export.py` | `graph.db` → `wiki/graph/` markdown (one-time live migration + re-derivation) | the proven `graph_export.py` | mode-independent |
| `scripts/graph-gaps.py` | **native** five-species scanner over the derived index | `graphbuilding/scripts/gaps.py` (seed=42 Louvain) | `hot.md` cache |
| `scripts/graph-resolve.py` | entity resolution: invariant-safe merge, **embedding dedup** | `graphbuilding/scripts/resolve.py` + redesign Change 1 | **`rerank.py` embeddings** |
| `scripts/graph-validate.py` | self-healing integrity (never crash, report+heal drift) | `graphbuilding/scripts/validate.py` | `agents/verifier.md` |
| `skills/graph/SKILL.md` (+ `graph-ingest`, `graph-query`, `graph-gaps` sub-skills) | user surface, mirrors the `wiki-*` family | `graphbuilding/SKILL.md` | transport, lock, mode |

The "native rebuild" payoff over a straight port: each module is rewritten to **call this
repo's existing infrastructure** instead of reinventing it (locking, embeddings, cache,
transport, tests), which is the entire reason for fusing rather than symlinking.

---

## 6. Reuse synergies — why fuse instead of bridge

1. **Embedding dedup reuses the retrieval embedding stack.** `graph-skill-redesign.md`
   Change 1 (the #1 felt pain: VITON-HD re-entering every paper) needs local embeddings to
   merge duplicates *at ingest*. This repo already runs nomic-embed via ollama for
   `rerank.py`. **One embedding stack, two wins:** retrieval rerank + ingest-time entity
   dedup. Graceful degrade when ollama is absent (same posture as retrieval today).
2. **Materialized analytics reuse the `hot.md`/`index.md` cache pattern.** Redesign Change
   4 (stop recomputing Louvain/gaps every question) maps onto the tiered cache this repo
   already shipped: a `wiki/graph/hot.md` (recent gap signal) + a `graph-snapshot.json`
   (hash-gated, incremental refresh).
3. **Claim ingest reuses `wiki-lock.sh`.** Multi-writer safety for parallel claim writes
   comes free — the same advisory locks the page wiki uses.
4. **The round-trip test reuses the pytest harness.** Ports straight into `tests/`.
5. **The `verifier` agent extends to graph invariants.** It already does a pre-commit
   six-cut review; we add integrity checks (no dangling/chain/orphan — the exact bug class
   from `graph-skill-debt.md`) so drift can never be committed.

---

## 7. Write-side invariants (so the native engine cannot re-accumulate drift)

`graph-skill-debt.md` showed reads were forced to repair-then-recompute because the *write*
path never maintained the invariants the read path depends on (103 dangling, 134 chains, 13
self-loops, 8 orphans in one live session). The native rebuild bakes the redesign's
write-side guarantees in from day one:

- **One `root()` helper** (Change 2) — path-compressing, cycle- and dangling-safe. Replaces
  every inline single-hop `COALESCE(canonical_id, id)`. Already drafted and validated.
- **Invariant-safe merge** — after pointing loser→winner, path-compress dependents and
  refuse any write that would create a self-loop or cycle.
- **Self-healing validate** (Change 3) — reports and heals; never throws on an orphan.
- **FK-on everywhere + delete policy** — `ON DELETE SET NULL` promotes children instead of
  orphaning; no raw delete bypasses `connect()`.

Result (debt doc diagram B): reads become pure reads; the model is invoked **only for
genuinely new ambiguity**, never to re-judge old drift.

---

## 8. What we deliberately do NOT copy

- **From this repo:** `wiki-ingest` / `wiki-query` *page-model brains* (editorial-judgment
  prose extraction). They would flatten the claim layer. The page wiki keeps them; the graph
  layer has its own structured, typed ingest.
- **From graphbuilding:** nothing is dropped — but the scattered `~/.graphbuilding` +
  `~/.paper-scholar` setup is consolidated into this repo (§ side task).
- **Native graph view (Obsidian):** untyped/cosmetic, hairballs at scale. Keep
  graphbuilding's typed `graph.html` viz for analysis; use Obsidian for reading/editing.

---

## 9. Round-trip contract & current status

The export↔import round-trip is the acceptance oracle for the native rebuild:
`gaps(rebuild-from-vault) == gaps(reference)`.

**Already achieved** on a copy of the live `~/.graphbuilding/graph.db` (98 papers, 755
canonical entities, 1052 claims):
- ✅ **Five gap species reproduce exactly** — 899 gaps (frontier 65, debate 0, replication
  473, coverage 49, white-space 312), identical before/after. This is the real contract.
- ✅ 6/7 table counts match (papers, sections, entities, canonical, claims, proposed).
- ⚠️ One known cosmetic bug: aliases 834→779 (55 lost to an over-eager dedup on import) —
  fix when porting `graph-build.py`.
- ✅ Live graph confirmed clean (0 chains/loops/dangling) so `root()` path-compression does
  not perturb the baseline; Louvain is `seed=42` deterministic; ids preserved on import so
  community partitions are stable.

---

## 10. Migration & git consolidation (side task)

One-time, then repeatable:
1. `graph-export.py` dumps the live `~/.graphbuilding/graph.db` → `wiki/graph/` markdown +
   `wiki/graph/graph-export.json` (portable snapshot). **Committed.**
2. `graph.db` becomes derived → `.vault-meta/graph/graph.db`, **gitignored**, rebuilt by
   `graph-build.py`.
3. Raw PDFs / `~/.paper-scholar/*` → gitignored or git-lfs; only the faithful markdown +
   the json snapshot live in git.

This de-risks the scattered dotdir ecosystem and answers the "ecosystem too split" worry.

---

## 11. Phased plan (WIP=1, vertical slices)

- **Phase 0 — this design doc.** ← *current; awaiting approval.*
- **Phase 1 — migration landed in-repo.** `graph-db.py` (connect/`root()`/invariants) +
  `graph-build.py` importer + `wiki/graph/SCHEMA.md` + round-trip test green in `tests/`,
  vault pointed at `wiki/graph/`. Fix the alias bug. *(Feature branch.)*
- **Phase 2 — native gap engine + live migration.** `graph-gaps.py` (5 species) on the
  derived index; `graph-export.py` migrates the live graph in; commit markdown + snapshot.
- **Phase 3 — resolution + invariants.** `graph-resolve.py` with embedding dedup reusing
  `rerank.py`; wire `wiki-lock`; `graph-validate.py` self-healing.
- **Phase 4 — skill surface + cache + verifier.** `skills/graph/*`; retrieval indexes claim
  bodies; `verifier` integrity extension; `wiki/graph/hot.md` materialized signal.
- **Phase 5 — consolidation + packaging.** git-lfs/gitignore policy; docs; `plugin.json`
  version bump + CHANGELOG.

Each phase is one reviewable slice; the round-trip test guards every one.

---

## 12. Open questions / risks

1. **sqlite in git or not?** Recommend derived + gitignored (markdown + json are the
   tracked truth). The side task earlier floated committing `graph.db` — flag for decision.
2. **Mode-awareness of the graph layer.** The claim layer is organizationally
   mode-independent (it has its own typed structure). Recommend `graph/` is *not* routed by
   `wiki-mode.py`. Confirm.
3. **Eventually unify `wiki/entities/` prose with `graph/entities/`?** Deferred — kept
   separate per the location decision. Possible future bridge: prose page `[[links]]` to its
   structured twin.
4. **Embedding dependency.** ollama is optional in retrieval; ingest dedup must degrade to
   name/normalized-name matching when it's absent (no hard dep).
5. **Extraction quality.** Markdown makes bad claims *visible and fixable*; it does not fix
   them. Real quality gains still need a stronger extractor + the two walls rejecting bad
   quotes (out of scope here, noted).

---

## 13. Acceptance for this doc

Approve to proceed to **Phase 1**. On approval I will: create a feature branch, port the
schema + `graph-db.py` + `graph-build.py` into this repo's `scripts/`, point the vault at
`wiki/graph/`, land the round-trip test in `tests/`, and fix the alias bug — one vertical
slice, verifier-reviewed before commit.
