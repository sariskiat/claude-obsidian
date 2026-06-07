# Vault schema — the contract between markdown and the derived index

This vault is the **source of truth**. `.vault-meta/graph/graph.db` is a derived
index rebuilt from here by `graph-build.py`. Edit markdown; rebuild the index.

## Authority split
- **Frontmatter = queryable structure.** ids, types, predicate, polarity, strength,
  support, status, canonical pointer. These power the five gap species. To make a new
  relation queryable you must promote it into this typed frontmatter — loose `[[links]]`
  in the body are readable only.
- **Body = readable text.** Entity description; claim statement + verbatim quote +
  equations + worked example. The importer parses only the marked fields below.

## entities/<name>__e<id>.md
```yaml
type: entity
id: <int>              # stable; preserved on rebuild
name: <str>
super_type: Concept|Method|Artifact|Task|Measure|Source
sub_type: <str|null>
source_paper: <slug|null>
is_canonical: <bool>   # true => this row IS the canonical entity
canonical_id: <int|null>   # root entity id (path-compressed); null if canonical
merge_confidence: <float|null>
metadata: <json-str|null>
aliases: [<str>, ...]  # surface forms attaching to this id
# canonical: "[[root-slug]]"  # readability wikilink, present on variants only
```
Body: the entity description (free text).

## claims/c<id>.md
```yaml
type: claim
id: <int>
subject_id: <int>      # the entity as extracted (NOT rolled up — claims never merge)
predicate: <str>
object_id: <int>
claim_type: result|definition|proposal|limitation|open-question|...
polarity: asserts|refutes
strength: strong|moderate|tentative
support: <int>         # # distinct papers asserting equivalent <s,p,o>
status: confirmed|proposed
source_paper: <slug>
section_id: <int|null>
generated_by: <str>    # extractor name@version
confidence: <float|null>
subject: "[[slug]]"
object: "[[slug]]"
```
Body markers (parsed on import):
- Statement: the text from the end of frontmatter up to the quote callout.
- Verbatim quote: the lines inside the `> [!quote]` callout (the provenance wall).
- Equations: LaTeX in `$$...$$` blocks (readable only).
- Example: worked-example prose (readable only).

## papers/<slug>.md
Frontmatter carries title/authors/source_path/ingested_at, an `authors_list`, and a
`sections` list (each with id/heading/role/order_index/summary — section ids are
preserved because claims reference them). Body is a human render.

## _graph/*.md
`predicates.md`, `entity_edges.md`, `citation_links.md` — auxiliary tables stored as
frontmatter `rows:` lists. Navigation/citation edges are not part of the claim moat.

## papers/<slug>.full.md — full-text contract (Phase 4, P4)

`wiki/graph/papers/<slug>.full.md` is the **git-tracked, verbatim full text** of a Tier-A
paper. It is written (and idempotently re-written) by `graph-fulltext.py sync`.

```yaml
type: paper-fulltext
slug: <slug>                      # matches the filename stem and papers/<slug>.md
arxiv_id: <id|null>               # arxiv id if available
source_path: <absolute path>      # the resolved source .md (read-only upstream)
paper: "[[<slug>]]"               # Obsidian wikilink back to the papers/<slug>.md node
```
Body: verbatim byte-equal copy of the source `.md` content (no reflow, no truncation).

The `.full.md` files power `/graph read` — a BM25+rerank retrieval layer built by
`graph-fulltext.py sync` over these bodies only (not over claim/entity stubs or the
structured `papers/<slug>.md`). The retrieval index lives under
`.vault-meta/graph/{chunks,bm25}/` which is **derived and gitignored** — deletable and
rebuildable at any time.

### Add / re-add a paper

**First time (Tier-A: source .md exists at source_path):**

1. Ensure the paper is in `graph-export.json` with a resolvable `source_path` pointing to
   a real `.md` file (or a bare `~/.paper-scholar/<dir>` containing exactly one `.md`).
2. Run `uv run python scripts/graph-fulltext.py sync`
   → writes `wiki/graph/papers/<slug>.full.md` (verbatim body + contract frontmatter)
   → builds `.vault-meta/graph/chunks/` and `.vault-meta/graph/bm25/index.json`
3. Verify: `uv run python scripts/graph-retrieve.py "a phrase from the paper"` returns
   a passage with `page_path: wiki/graph/papers/<slug>.full.md`.

**Re-add after source update (idempotent):**

Re-run step 2; if the source `.md` has not changed, the `.full.md` and index are
byte-identical. If it changed, the `.full.md` is overwritten and the index rebuilt.

**Tier-B paper (PDF or no .md yet):**

`graph-fulltext.py sync` skips it with one log line and adds it to the "ready to add
later" backlog. Once the PDF is extracted to `.md`, register the new path in
`graph-export.json` as `source_path` and re-run sync.

---

## Forward path — graph-ingest.py (Phase 4)

New papers enter the graph through `graph-ingest.py` (Phase 4), which writes
fresh `papers/*.md`, `entities/*.md`, and `claims/*.md` into this vault. The
contract `graph-ingest.py` must follow:

1. **papers/<slug>.md** — frontmatter with `type: paper`, `slug`, `title`,
   `authors`, `source_path`, `ingested_at`, `authors_list`, `sections` (list of
   {id, heading, role, order_index, summary}). Body renders the paper structure.

2. **entities/<name>__e<id>.md** — frontmatter with `type: entity`, `id` (assign
   new unique id), `name`, `super_type`, `sub_type`, `source_paper`,
   `is_canonical: true`, `canonical_id: null`, `aliases`.

3. **claims/c<id>.md** — frontmatter with `type: claim`, `id` (new unique id),
   `subject_id`, `predicate`, `object_id`, `claim_type`, `polarity`, `strength`,
   `support`, `status`, `source_paper`, `section_id`, `generated_by`.

After ingest, run `graph-build.py` to rebuild the derived sqlite index. The build
is idempotent and lossless — suitable for running after every ingest.

### Example: hand-authoring a new paper/entity/claim through graph-build.py

1. Create `wiki/graph/papers/vaswani2017.md` with frontmatter per the contract above.
2. Create `wiki/graph/entities/attention-mechanism__e17.md` with `type: entity`.
3. Create `wiki/graph/claims/c42.md` with `type: claim`, `subject_id: 17`,
   `predicate: part-of`, `object_id: 3`.
4. Run `uv run python scripts/graph-build.py wiki/graph/ .vault-meta/graph/graph.db`
   — the derived index is rebuilt with the new paper/entity/claim.

---

## Semantic Bridge — directions report output (P6, graph-propose.py)

`/graph propose` (scripts/graph-propose.py) generates a `proposals.md`-grade research
directions report grounded in the real graph. The output lands under:

```
wiki/graph/proposals/YYYY-MM-DD-directions.md   ← clean accepted report
wiki/graph/proposals/YYYY-MM-DD-directions-2.md ← same-day rerun (never clobbers)
wiki/graph/proposals/YYYY-MM-DD-directions.rejected.md ← cap-exhausted artifact
```

These files are **git-tracked** (not gitignored). They are the evidence of a grounded
LLM analysis run, so they belong in version control.

### Required report sections (FR7 section contract)

The grounding gate and AC9 grep check for these headers in every accepted report:

- `## The bar` — harsh honest assessment of what it takes to earn the PhD ticket
- `## Decision matrix` — a markdown table with ceiling/odds/theory/scoop columns
- `### N.` — one block per direction (minimum 3), each containing `**Takedown:**`
- `## Ranking` — ordered ranking with rationale + fork rule
- `## Execution` — first-week concrete proof-of-concept probe with gate condition

### Grounding audit footer

Every accepted report ends with:

```
*Grounding audit: N/N citations verified ✓ | retries: R | model: M | bridge candidates: K*
```

N = total citations extracted, all verified against `graph.db`. The gate rejects any
report where any citation is not in `papers.slug` or `entities.name`.

### RESEARCH_PROFILE.md

`wiki/graph/RESEARCH_PROFILE.md` is the user-owned profile injected into every prompt.
It encodes: goal (top-venue first-author paper → PhD ticket), strength/gap profile,
the 5 ranked directions, the fork rule, the Aek phase rule (no email until 4 deep-reads).
Seeded from the `research-goal-phd-paper` memory note at feature build time.
Edit it directly to update the research context.

### Source exemplar

`~/Desktop/research/proposals.md` is the style exemplar — the hand-written proposals.md
that establishes the genre and tone. It is **read-only**; graph-propose.py never writes to it.
