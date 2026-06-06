# Spec: Graphbuilding Fusion — Phase 1: Markdown-Vault Migration Round-Trip

**Feature ID:** `graph-vault-migration`
**Status:** `approved`
**Grilled on:** 2026-06-06
**Approved on:** 2026-06-06
**Approved by:** `saris` (human)
**Problem classification:** `unique`

> Epic: [`docs/graphbuilding-fusion-design.md`](../docs/graphbuilding-fusion-design.md).
> Phase 1 only — the smallest provable slice.
> Research: `~/Desktop/research/graph-skill-debt.md`, `graph-skill-redesign.md`.

---

## 1. Problem → Objective → Metric

```
┌────────────────────────────┐     ┌────────────────────────────┐     ┌────────────────────────────┐
│ PROBLEM                    │     │ OBJECTIVE                  │     │ METRIC  (Phase 1)          │
│                            │     │                            │     │                            │
│ graph lives ONLY as sqlite │     │ markdown vault =           │     │ round-trip fidelity        │
│ in a dotdir:               │ ──▶ │   SOURCE OF TRUTH          │ ──▶ │ (binary pass / fail)       │
│   ~/.graphbuilding/graph.db│     │                            │     │                            │
│                            │     │ sqlite =                   │     │   ✓ 9/9 tables byte-equal  │
│ • not hand-editable        │     │   DERIVED, rebuildable,    │     │   ✓ 5/5 gap species exact  │
│ • not git-tracked          │     │   throwaway index          │     │                            │
│ • reads FORCE repair +     │     │                            │     │ baseline: 6/7 tables, 5/5  │
│   full recompute (drift:   │     │ hand-fix a claim → rebuild │     │ target:   9/9 tables, 5/5  │
│   103 dangling/134 chains) │     │ → same 5 gap species       │     │                            │
└────────────────────────────┘     └────────────────────────────┘     └────────────────────────────┘
```

**Problem Statement.** The claim graph lives only as a sqlite db in a scattered dotdir; it is
not hand-editable, not git-tracked, and reads force repair + full recompute because the write
path never guaranteed the read invariants.

**Objective.** Make a structured markdown vault the source of truth, sqlite a derived index,
and prove the derivation is lossless. Phase 1 builds and verifies that seam only.

**JTBD.** When I want to trust and hand-fix my claim graph, I want it stored as structured
markdown I can edit and version, so I can rebuild a queryable index from it losslessly and
stop re-mining on every read.

---

## 2. The two halves being fused (neither discarded)

```
        FROM graphbuilding  (keep — the moat)              FROM this repo  (reuse — the engineering)
       ┌──────────────────────────────────────┐          ┌──────────────────────────────────────┐
       │ claim = typed signed proposition      │          │ markdown SoT + git                    │
       │   (NEVER merged)                      │          │ per-file advisory locking             │
       │ 6 entity super-types                  │          │ BM25 + embedding retrieval            │
       │ typed predicates (domain/range)       │   ───▶   │ hot.md / index.md tiered cache        │
       │ 5 gap species                         │  fuse    │ transport fallback (cli→mcp→fs)       │
       │ polarity (asserts/refutes)            │          │ verifier agent · pytest harness       │
       │ support count · two provenance walls  │          │ methodology modes · plugin packaging  │
       └──────────────────────────────────────┘          └──────────────────────────────────────┘
        what Obsidian/Graphify CANNOT do                   the plumbing that carries the claim layer
```

---

## 3. Architecture — markdown SoT, sqlite as a derived index

```
 ╔═══════════════════════════════════════════════════════════════════════════════╗
 ║  wiki/graph/      SOURCE OF TRUTH      (git-tracked · hand-editable · Obsidian) ║
 ║                                                                                ║
 ║    papers/      98 .md   frontmatter: title, authors, sections+summaries       ║
 ║    entities/  1444 .md   frontmatter: typed identity, canonical ptr, aliases   ║
 ║    claims/    1052 .md   frontmatter: TRIPLE + polarity + support + provenance  ║
 ║    _graph/         .md   predicates · entity_edges · citation_links            ║
 ║    SCHEMA.md             the frontmatter contract                              ║
 ║    graph-export.json     full machine snapshot (committed)                     ║
 ╚═══════════════════════════════════════════════════════════════════════════════╝
                    │
                    │   graph-build.py   (reads YAML frontmatter → rebuilds index, ids preserved)
                    ▼
        ┌─────────────────────────────────────────────────────────────┐
        │  .vault-meta/graph/graph.db                                 │
        │  DERIVED · gitignored · throwaway                           │
        │  ids preserved → Louvain(seed=42) stable → gaps identical   │
        └─────────────────────────────────────────────────────────────┘
                    │
                    │   graph-gaps.py / graph-resolve.py   (analytics engine)
                    ▼
        ┌─────────────────────────────────────────────────────────────┐
        │  5 gap species · support rollup · Louvain communities       │
        │  the questions NO page-level organizer can even represent   │
        └─────────────────────────────────────────────────────────────┘

        ▲                                                             │
        └──────────────  graph-export.py  (re-derives the vault)  ◀───┘
                         ^^ this loop is exactly what Phase 1's round-trip proves
```

> **Is this a migration?** Yes — one-time AND permanent. Today the graph is sqlite-only. After Phase 1 the markdown vault is the source of truth forever. The **round-trip test** proves the migration is lossless: export existing 98 papers / 1444 entities / 1052 claims to markdown → rebuild sqlite from markdown → compare byte-for-byte. After Phase 1 the direction reverses: markdown is authored first, sqlite is rebuilt from it.

---

## 4. The frontmatter contract — what makes a claim queryable

```
 entities/attention-mechanism__e17.md
 ┌──────────────────────────── frontmatter (QUERYABLE) ──────────────────┐
 │ type: entity                                                          │
 │ id: 17                                                                │
 │ name: "Attention Mechanism"                                           │
 │ super_type: Concept        sub_type: "neural-network-component"       │
 │ is_canonical: true         canonical_id: null                         │
 │ merge_confidence: 0.95     source_paper: vaswani2017                   │
 │ aliases: ["self-attention", "Scaled Dot-Product Attn"]                │
 ├──────────────────────────── body (READABLE ONLY) ─────────────────────┤
 │ A mechanism that lets each position attend to all positions. Core     │
 │ building block of the Transformer architecture.                       │
 └───────────────────────────────────────────────────────────────────────┘

 claims/c42.md
 ┌──────────────────────────── frontmatter (QUERYABLE) ──────────────────┐
 │ type: claim                                                           │
 │ id: 42                                                                │
 │ subject_id: 17        ◀── entity AS EXTRACTED (claims never roll up)   │
 │ predicate: part-of                                                    │
 │ object_id: 3                                                          │
 │ claim_type: definition     polarity: asserts     strength: strong     │
 │ support: 3            ◀── # distinct papers asserting same ⟨s,p,o⟩     │
 │ status: confirmed          source_paper: vaswani2017                  │
 │ section_id: 88             generated_by: claude-opus-4-8@v1           │
 ├──────────────────────────── body (READABLE evidence) ─────────────────┤
 │ Attention is a core building block of the Transformer architecture.   │
 │                                                                       │
 │ > [!quote]                          ◀── THE PROVENANCE WALL           │
 │ > "We propose a new simple network architecture, the Transformer,     │
 │ >  based solely on attention mechanisms..."                           │
 └───────────────────────────────────────────────────────────────────────┘

 ┌───────────────────────────────────────────────────────────────────────┐
 │  FRONTMATTER = QUERYABLE   → powers the 5 gap species. importer reads. │
 │  BODY        = READABLE    → statement + quote callout + equations.    │
 │  to make a new relation queryable → promote it INTO the frontmatter.   │
 └───────────────────────────────────────────────────────────────────────┘

 ⚠  NOT the same as wiki/entities/ prose pages. Those are editorial prose;
    these are structured-claim entities. Prose drift = losing gap-finding.
```

---

## 5. Happy path — the round-trip pipeline

```
  ┌──────────┐    ┌───────────┐    ┌──────────┐    ┌───────────┐    ┌──────────┐
  │ ① COPY   │    │ ② EXPORT  │    │ ③ BUILD  │    │ ④ COMPARE │    │ ⑤ PASS   │
  │          │──▶ │           │──▶ │          │──▶ │           │──▶ │          │
  │ live.db  │    │ copy.db   │    │ vault →  │    │ copy.db   │    │ 9/9      │
  │  → copy  │    │  → vault  │    │ rebuilt  │    │   vs      │    │ tables   │
  │          │    │ writes    │    │ .db      │    │ rebuilt   │    │ 5/5      │
  │ never    │    │ all .md   │    │ ids kept │    │ per-row   │    │ species  │
  │ mutate   │    │ + .json   │    │ root()   │    │ diff, 9   │    │   ✓      │
  │ live(BR4)│    │           │    │ self-heal│    │ tables    │    │          │
  └──────────┘    └───────────┘    └──────────┘    └───────────┘    └──────────┘

  ⚡ The live ~/.graphbuilding/graph.db is NEVER mutated. The test copies it →
     exports → builds → diffs → discards everything in a temp dir (BR4).
```

1. Copy live `graph.db` → `copy.db` (temp).
2. `graph-export.py copy.db → wiki/graph/` writes all markdown + `graph-export.json`.
3. `graph-build.py wiki/graph/ → rebuilt.db`, ids preserved, `root()` self-heals.
4. Diff `copy.db` vs `rebuilt.db`: 9 tables byte-equivalent.
5. Run gaps on both → 5 species identical. Green = lossless SoT.

---

## 6. Business rule: the `root()` invariant helper

```
  Every current script does  COALESCE(canonical_id, id)  — single-hop and WRONG on chains.
  root() is the ONE shared rollup helper that replaces all of them.

  CASE 1 — CHAIN              CASE 2 — CYCLE / SELF-LOOP     CASE 3 — DANGLING
  ─────────────────          ──────────────────────────    ────────────────────
   A ──▶ B ──▶ C              A ──▶ B                        A ──▶ (id 999 = gone)
              (C canon)       ▲      │
                              └──────┘                       db:
  db:                                                         A.canonical_id → 999
   A.canonical_id → B         db:                              (no row 999 exists)
   B.canonical_id → C          A.canonical_id → B
   C.canonical_id = NULL       B.canonical_id → A            naive follow:
                                                              CRASHES ✗
  single-hop COALESCE:        naive follow:                   (crashed validate.py
   A → B  ✗ stops at B         loops forever ✗                 check 6, 103 found)
   (B is NOT canonical)
                              root():                        root():
  root():                      visited set → sees B again     parent missing → stop
   A→B→C  root = C  ✓          → elect min(A,B) = A  ✓         → return last live (A) ✓
   compress A→C, B→C           refuse the loop                no crash · reports it
                               compress B→A

  ────────────────────────────────────────────────────────────────────────────────
   root(conn, id) = always a live id · always terminates · path-compress on read ·
   cycle-safe · dangling-safe · serves EVERY hub/gap/community/validate query
  ────────────────────────────────────────────────────────────────────────────────
```

- **BR2** (redesign Change 2): `root()` follows the chain to the NULL root, path-compresses,
  refuses cycles/self-loops, survives dangling. Kills the chain-miscount bug at the source.
- **BR3** (redesign Change 3): `graph-build.py` calls `root()` on import → self-heals drift,
  reports it, never crashes. Drift can no longer accumulate.

---

## 7. Business rule: ids must be preserved (BR1)

```
   IDs PRESERVED  (correct)                 IDs RENUMBERED  (broken)
  ┌─────────────────────────────┐          ┌─────────────────────────────┐
  │ entity ids 10,17,42,88       │          │ auto-increment: 10→1, 17→2,  │
  │ → SAME ids in rebuilt db     │          │   42→3, 88→4                 │
  │                              │          │                              │
  │ Louvain(seed=42) partitions  │          │ Louvain re-partitions on the │
  │ on node labels → SAME        │          │ new labels → partition       │
  │ partition                    │          │ SHIFTS                       │
  │                              │          │                              │
  │ white-space gaps: 312  ✓     │          │ white-space gaps: drift  ✗   │
  │ per-row diff possible        │          │ no per-row diff → bugs hide  │
  └─────────────────────────────┘          └─────────────────────────────┘

  Louvain(seed=42) is deterministic but NOT label-invariant — the partition
  depends on entity ids. Preserve ids → round-trip is exact.
```

---

## 8. The alias bug (AC3 — the one thing broken today)

> **Root cause (empirically verified against the live db, NOT a case-fold collision).**
> `distinct lower(alias) == 834 == total` → there are **zero** case collisions. The 55 lost
> aliases are exactly the 55 rows whose `canonical_id` points at a **deleted/missing entity**
> (dangling FK). Export drops them while grouping aliases by `canonical_id` (no live entity to
> attach to), and the derived `aliases` table enforces `REFERENCES entities(id)`, so they could
> not be re-inserted under FK-on even if exported. 55 dangling rows = the 55 lost. Exact match.

```
  BEFORE (broken):  834 in live db  →  779 in rebuilt   (the 55 DANGLING-FK rows lost)
  ──────────────────────────────────────────────────────────────────────────────────
   aliases table:  alias UNIQUE,  canonical_id NOT NULL REFERENCES entities(id)

   55 rows:  alias="…",  canonical_id → (entity id 411–424, 940+ … NO SUCH ROW)
        │
        ├─ export: groups aliases by canonical_id → no live entity → row never written ✗
        └─ build : even if written, FK REFERENCES entities(id) rejects the insert ✗

  AFTER (fixed):    834 in live db  →  834 in rebuilt   ✓
  ──────────────────────────────────────────────────────────────────────────────────
   • export writes ALL 834 alias rows verbatim, including the 55 with a dangling
     canonical_id (carry the id as-is; the vault is the truth, dangling and all)
   • derived aliases table does NOT enforce the entities(id) FK → dangling rows re-insert
   • no normalize · no case-fold · no silent dedup · no FK rejection
   verify: AC1 per-row diff shows aliases 834 = 834  ✓
```


---

## 9. Constraints — the change surface

```
 ┌─ EXISTING REPO — READ-ONLY · CANNOT MODIFY ──────────────────────────────────┐
 │  skills/wiki-*  (15 skills)                                                   │
 │  scripts/{retrieve, bm25-index, rerank, contextual-prefix, wiki-mode,        │
 │           wiki-lock.sh, detect-transport.sh}                                 │
 │  wiki/entities/ prose pages · agents/verifier.md · hooks/ · bin/ · docs/     │
 │  plugin.json (version bump deferred to Phase 5) · _templates/ · .obsidian/   │
 │                                                                              │
 │   ┌─ NEW FILES — CAN CREATE / MODIFY  (THE BLAST RADIUS) ─────────────────┐  │
 │   │  scripts/graph-db.py        scripts/graph-export.py                   │  │
 │   │  scripts/graph-build.py     tests/test_graph_roundtrip.py             │  │
 │   │  wiki/graph/SCHEMA.md       wiki/graph/{papers,entities,claims,_graph}/│  │
 │   │  wiki/graph/graph-export.json (committed)                             │  │
 │   │  .gitignore (+1 entry)      pyproject.toml (+PyYAML +networkx deps)   │  │
 │   └────────────────────────────────────────────────────────────────────────┘  │
 └──────────────────────────────────────────────────────────────────────────────┘

   ~/.graphbuilding/graph.db  =  OUTSIDE the repo · COPY only · NEVER written
```

- **Must not break:** existing `skills/wiki-*`, retrieval scripts, `wiki/entities/` prose, transport/lock infra.
- **Performance:** round-trip over ~98 papers / ~1050 claims in seconds (prototype ~3.4s).
- **Security:** single-tenant; no external calls; reads sources only via the copied db.
- **Needs human approval before changing:** anything outside the mutable surface; native gap engine (Phase 2); committing real migrated claim data.

---

## 10. Acceptance criteria

```
  [ ]  AC1   9 tables byte-equal          test_table_counts_match + per-row diff
  [ ]  AC2   5 gap species exact          test_gap_species_match  (65 / 0 / 473 / 49 / 312)
  [ ]  AC3   alias bug fixed  779 → 834   part of AC1 per-row diff
  [ ]  AC4   root() chain/cycle/dangling  test_root_invariants
  [ ]  AC5   source db untouched          test_source_untouched   (hash before == after)
  [ ]  AC6   gitignore correct            git check-ignore  (.db ignored, .json tracked)
  [ ]  AC7   full suite green             uv run python -m pytest tests/test_graph_roundtrip.py -q
  [ ]  AC8   forward path documented      SCHEMA.md defines the contract; hand-authored
             (design only, not automated) test paper/entity/claim through graph-build.py
  ─────────────────────────────────────────────────────────────────────────────────────────
        all 7 automated ACs pass + AC8 documented  →  Evaluator verdict: PASS
```

**AC8 — New paper ingestion: the forward path (target design after Phase 4):**

```
  ┌──────────┐    ┌───────────┐    ┌────────────────┐    ┌──────────┐    ┌──────────┐
  │ SOURCE   │    │ WIKI      │    │ GRAPH INGEST   │    │ REBUILD  │    │ GAPS     │
  │ .raw/    │──▶ │ INGEST    │──▶ │ (Phase 4)      │──▶ │          │──▶ │          │
  │ PDF/text │    │ unchanged │    │ graph-         │    │ graph-   │    │ new gaps │
  │          │    │ → prose   │    │ ingest.py      │    │ build.py │    │ surface  │
  │          │    │ wiki/     │    │ writes to      │    │ ids kept │    │ in 5     │
  │          │    │ entities/ │    │ wiki/graph/:   │    │ root()   │    │ species  │
  │          │    │ (no chg)  │    │  papers/*.md   │    │ heals    │    │          │
  │          │    │           │    │  entities/*.md │    │          │    │          │
  │          │    │           │    │  claims/*.md   │    │          │    │          │
  └──────────┘    └───────────┘    └────────────────┘    └──────────┘    └──────────┘
```

- Phase 1 delivers: `SCHEMA.md` — the frontmatter contract `graph-ingest.py` must write to.
- Phase 1 does NOT implement: `graph-ingest.py`, `skills/graph/*`, git automation for commits.
- `wiki-ingest` prose side runs unchanged in parallel, writes to `wiki/entities/` as today.
- Claims always store `subject_id`/`object_id` as extracted — never rolled up at write time.
- `graph-build.py` is idempotent: safe to run after every ingest.

**Metric.** Round-trip fidelity, binary. Baseline 6/7 tables + 5/5 species → target 9/9 + 5/5.
**Why the user notices:** a green round-trip means the vault is hand-editable and the index
rebuilds with zero loss — the claim graph becomes a trusted, fixable, git-tracked artifact
instead of an opaque dotdir db.

---

## 11. Users and roles

| Actor | Can do | Cannot do |
|---|---|---|
| Owner (human) | run export/build, hand-edit `wiki/graph/*.md`, approve spec | — |
| `graph-build.py` | read `wiki/graph/` → write derived `.vault-meta/graph/graph.db` | write the live db; edit markdown |
| `graph-export.py` | read a sqlite db (a COPY) → write markdown vault | mutate the source db |
| round-trip test | copy live db, export, build, diff | run against the live db in place |

## 12. Functional requirements

| ID | Requirement | Notes |
|---|---|---|
| FR1 | `wiki/graph/SCHEMA.md` defines the frontmatter contract | authority split |
| FR2 | `graph-db.py`: `connect()` (FK on), `root()`, typed inserts | no inline COALESCE |
| FR3 | `graph-export.py` dumps db → vault + JSON snapshot | runs on COPY; ids preserved |
| FR4 | `graph-build.py` rebuilds derived db from vault | ids preserved; root() self-heals |
| FR5 | `tests/test_graph_roundtrip.py` proves AC1–AC7 | pytest harness |
| FR6 | Fix alias round-trip bug (834→834) | exact-string preservation |
| FR7 | Derived db gitignored; JSON tracked | derived is throwaway |
| FR8 | `wiki/graph/` paths fixed, never routed through `wiki-mode.py` | mode-independent |

## 13. Data, integrations, edge cases

| Data | R/W | Failure behavior |
|---|---|---|
| `~/.graphbuilding/graph.db` (live) | R only (copied) | absent → test skip |
| `.vault-meta/graph/graph.db` (derived) | W (rebuilt) | atomic overwrite; gitignored |
| `wiki/graph/**.md` | W export / R build | bad frontmatter → build raises with path |
| `wiki/graph/graph-export.json` | W | committed snapshot |
| PyYAML | dep | declared; `uv` install |
| networkx | dep (REQUIRED) | white-space species needs Louvain; absent → species collapses 312→1 (AC2 silently wrong) |

| Edge case | Expected |
|---|---|
| Live graph clean (current) | root() no-op; round-trip exact; build still self-heals |
| Drift present | root() compresses + reports; never crashes; orphans surfaced |
| citation_links empty | export/import empty table cleanly |
| Duplicate section headings | referenced by preserved id, not heading text |
| Live db missing | test skips with message |

---

## 14. Phase roadmap (this slice is P1)

```
  P0 ✓ ───▶  ● P1 ◀ THIS ───▶  P2 ───▶  P3 ───▶  P4 ───▶  P5
  design     import +          native   resolve  skills   consolidate
  doc        round-trip        gaps +   + embed  + cache   + package
                               migrate  dedup    + verifier
```

**Out of scope:** native `graph-gaps.py` & live migration (P2) · embedding dedup reusing
`rerank.py` (P3) · `wiki-lock`/retrieval/`verifier`/`hot.md` & `skills/graph/*` (P4) ·
git-lfs/PDF consolidation & `plugin.json` bump (P5) · unifying prose `wiki/entities/` with
`wiki/graph/entities/` (deferred indefinitely).

## 15. Open questions (deferred, non-blocking)

- **P3:** embedding dedup degrades gracefully when ollama absent.
- **Future:** prose `wiki/entities/` linking to its structured `wiki/graph/entities/` twin.
- **P2:** does the live migration commit `graph-export.json` only, or also a markdown snapshot.
- **P4 — full-document context for extraction:** `graph-ingest.py` sees the *whole* paper (full
  coreference/context) when extracting claims, not chunks. This is an extractor-*input* quality
  knob — it does NOT store the document in the graph and does NOT touch the Phase 1 schema or
  round-trip. The graph's value is the distillation; full text never becomes a graph node.
- **P3 — hybrid query spike (claim → source expansion):** test whether semantic query improves by
  joining the two layers at *query* time instead of merging them at storage time:
  `question → claim hit → follow section_id → pull surrounding passage from prose/.raw → rerank`.
  The keys already exist (claims carry `section_id` + `source_paper`; `papers/*.md` hold sections).
  A/B claim-only vs claim+expanded-section over a fixed question set; win criterion TBD.
  Rejected alternative: storing full document text inside `wiki/graph/` — dilutes the moat (gaps
  run on triples, not prose), duplicates the existing prose+`retrieve.py` layer (the "become the
  organizer" trap), and wrecks the byte-equal round-trip. The provenance-wall quote is already the
  correct, surgical amount of source text in the graph.

## Approval Checklist

- Domain lead approved: ✅ 2026-06-06 (saris)
- Tech lead approved: ✅ 2026-06-06 (saris)
- Critic reviewed testability: ✅ binary round-trip + AC8 forward-path documented
- Metric validated against user value: ✅ lossless, hand-editable, git-tracked claim graph

---

## 16. Persisted Task Breakdown (Planner — Phase 1)

Recon-verified facts (2026-06-06, branch `feature/graph-vault-migration`):
- Live DB `~/.graphbuilding/graph.db` exists (1.75MB). Counts confirmed: papers 98, sections 789,
  paper_authors 97, entities 1444 (755 canonical, `canonical_id IS NULL`), predicates 46,
  claims 1052 (64 `proposed`), entity_edges 543, citation_links 0, **aliases 834**.
- **AC3 root cause (verified):** 55 of the 834 alias rows have a `canonical_id` pointing at a
  non-existent entity id (dangling FK; ids 411-424, 940+, …). The oracle `graph_export.py` groups
  aliases by `canonical_id` and only emits ones attaching to a live entity → 779 reach frontmatter →
  779 imported. Fix = preserve ALL 834 alias rows verbatim through export AND import (dangling
  included). The derived `aliases` table must NOT enforce the `REFERENCES entities(id)` FK on these
  rows, else re-insert is rejected under `PRAGMA foreign_keys = ON`.
- **AC2 gap species (verified):** with `networkx` present the 5 species are exactly
  frontier 65 / debate 0 / replication 473 / coverage 49 / white-space 312 (total 899), identical
  src vs rebuilt. WITHOUT networkx the Louvain white-space path hits a fallback (white-space→1,
  total 588). **`networkx` is therefore a required dependency** in `pyproject.toml`, not optional.
- **AC5 (verified):** the oracle `gaps.py` and `export()` do NOT mutate their source db (md5 stable).
  The native scripts must keep this property; the test runs on a temp copy.
- Native script filenames are hyphenated (`graph-db.py`) → not importable as modules. The test loads
  them via `importlib.util.spec_from_file_location` or drives them by subprocess CLI.
- Missing today: `pyproject.toml`, `wiki/graph/`, all `scripts/graph-*.py`, `tests/test_graph_roundtrip.py`.
- Oracle (reference only, do NOT copy): `~/.claude/skills/graphbuilding/scripts/{db_helpers,graph_export,graph_import,init_db,gaps}.py`.

See `feature_list.json` for the machine-readable task graph and runnable verification per AC.
