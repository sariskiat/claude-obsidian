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
