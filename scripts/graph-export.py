"""Export sqlite claim-graph → markdown vault + JSON snapshot.

The vault under ``wiki/graph/`` is the source of truth; sqlite is a derived index.
This script reads a COPY of the live db (never writes to it — BR4).

Layout:
    papers/<slug>.md       frontmatter: title/authors/sections+summaries
    entities/<slug>.md     frontmatter: typed identity + canonical pointer + aliases
    claims/c<id>.md        frontmatter: queryable triple + provenance
    _graph/predicates.md    predicate vocabulary (domain/range)
    _graph/entity_edges.md  navigation edges
    _graph/citation_links.md  CiTO links
    SCHEMA.md              frontmatter conventions (the contract)
    graph-export.json      full machine dump (portable snapshot)

CRITICAL — alias fix (AC3): Export ALL alias rows verbatim, including those with
dangling canonical_id (FK pointing to a non-existent entity). The oracle grouped
aliases by canonical_id and only emitted ones attaching to live entities — dropping
55 rows. We preserve all 834 rows by exporting the aliases table directly.

Usage:
    uv run python scripts/graph-export.py <db_path> <vault_dir>
"""

import json
import re
import sys
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from graph_db import connect, root  # noqa: E402

QUOTE_CALLOUT = "> [!quote]"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def slugify(name: str) -> str:
    s = re.sub(r"[^\w\s-]", "", str(name).lower()).strip()
    s = re.sub(r"[\s_]+", "-", s)
    return s or "x"


def entity_slug(eid: int, name: str) -> str:
    return f"{slugify(name)}__e{eid}"


def _yaml(fm: dict) -> str:
    return yaml.safe_dump(fm, sort_keys=False, allow_unicode=True,
                          default_flow_style=False, width=10_000).strip()


def write_md(path: Path, fm: dict, body: str) -> None:
    path.write_text(f"---\n{_yaml(fm)}\n---\n\n{body.rstrip()}\n", encoding="utf-8")


def _all_rows(conn, table: str) -> list[dict]:
    cur = conn.execute(f"SELECT * FROM {table}")
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------

def export(db_path: str, vault_dir: str) -> dict:
    """Export a sqlite db to a markdown vault. Returns per-table counts."""
    db_path = Path(db_path)
    vault_dir = Path(vault_dir)
    conn = connect(db_path)

    # Fresh vault tree
    for sub in ("papers", "entities", "claims", "_graph"):
        d = vault_dir / sub
        if d.exists():
            for f in d.glob("*.md"):
                f.unlink()
        d.mkdir(parents=True, exist_ok=True)

    # Load all tables
    ents = _all_rows(conn, "entities")
    slug_of = {e["id"]: entity_slug(e["id"], e["name"]) for e in ents}

    # ---- aliases — export ALL rows verbatim (AC3 fix) ----
    # The oracle grouped by canonical_id (live entities only), dropping 55
    # dangling-FK rows. We write every alias row as-is into the JSON snapshot;
    # the per-entity frontmatter still lists aliases for live canonical_ids only
    # (readability), but graph-export.json carries the full 834 rows → import
    # re-inserts them all. Dangling rows are just not shown in entity frontmatter.
    all_aliases = _all_rows(conn, "aliases")

    # Group by canonical_id for frontmatter readability (only live entities)
    alias_by_entity: dict[int, list[str]] = {}
    for a in all_aliases:
        cid = a["canonical_id"]
        alias_by_entity.setdefault(cid, []).append(a["alias"])

    # ---- entities ----
    for e in ents:
        eid = e["id"]
        is_canon = e["canonical_id"] is None
        root_id = None if is_canon else root(conn, eid, compress=False)
        fm = {
            "type": "entity",
            "id": eid,
            "name": e["name"],
            "super_type": e["super_type"],
            "sub_type": e["sub_type"],
            "source_paper": e["source_paper"],
            "is_canonical": is_canon,
            "canonical_id": root_id,
            "merge_confidence": e["merge_confidence"],
            "metadata": e["metadata"],
            "aliases": sorted(alias_by_entity.get(eid, [])),
        }
        if not is_canon:
            fm["canonical"] = f"[[{slug_of[root_id]}]]"
        body = (e.get("description") or "").strip() or "*(no description)*"
        write_md(vault_dir / "entities" / f"{slug_of[eid]}.md", fm, body)

    # ---- claims ----
    for c in _all_rows(conn, "claims"):
        s_slug = slug_of.get(c["subject_entity_id"], f"entity__e{c['subject_entity_id']}")
        o_slug = slug_of.get(c["object_entity_id"], f"entity__e{c['object_entity_id']}")
        fm = {
            "type": "claim",
            "id": c["id"],
            "subject_id": c["subject_entity_id"],
            "predicate": c["predicate"],
            "object_id": c["object_entity_id"],
            "claim_type": c["claim_type"],
            "polarity": c["polarity"],
            "strength": c["strength"],
            "support": c["support"],
            "status": c["status"],
            "source_paper": c["source_paper"],
            "section_id": c["section_id"],
            "generated_by": c["generated_by"],
            "confidence": c["confidence"],
            "subject": f"[[{s_slug}]]",
            "object": f"[[{o_slug}]]",
        }
        quote_lines = "\n".join(
            f"> {ln}" for ln in (c.get("verbatim_quote") or "").splitlines() or [""]
        )
        body = (
            f"{(c.get('text') or '').strip()}\n\n"
            f"{QUOTE_CALLOUT}\n{quote_lines}\n\n"
            f"**Relation:** [[{s_slug}]] —`{c['predicate']}`→ [[{o_slug}]]\n\n"
            f"<!-- equations: add LaTeX in $$...$$ blocks below this line -->\n"
            f"<!-- example: add a worked example below this line -->"
        )
        write_md(vault_dir / "claims" / f"c{c['id']}.md", fm, body)

    # ---- papers (+ sections, authors) ----
    sections_by_paper: dict[str, list[dict]] = {}
    for s in _all_rows(conn, "sections"):
        sections_by_paper.setdefault(s["paper_slug"], []).append(s)

    authors_by_paper: dict[str, list[str]] = {}
    for a in _all_rows(conn, "paper_authors"):
        slug = a.get("paper_slug", a.get("slug", ""))
        authors_by_paper.setdefault(slug, []).append(
            a.get("name", a.get("author_name", ""))
        )

    for p in _all_rows(conn, "papers"):
        slug = p.get("slug", "")
        secs = sorted(
            sections_by_paper.get(slug, []),
            key=lambda s: s.get("order_index", 0),
        )
        fm = {
            "type": "paper",
            "slug": slug,
            "title": p.get("title", ""),
            "authors": p.get("authors", ""),
            "source_path": p.get("source_path", ""),
            "ingested_at": p.get("ingested_at", ""),
            "authors_list": authors_by_paper.get(slug, []),
            "sections": [
                {
                    "id": s["id"],
                    "heading": s.get("heading", ""),
                    "role": s.get("role", ""),
                    "order_index": s.get("order_index", 0),
                    "summary": s.get("summary", ""),
                }
                for s in secs
            ],
        }
        body_parts = [f"# {p.get('title', '')}", ""]
        for s in secs:
            role = f"[{s.get('role', '')}] " if s.get("role") else ""
            body_parts.append(f"## {role}{s.get('heading', '')}")
            summary = (s.get("summary") or "").strip()
            body_parts.append(summary)
            body_parts.append("")
        write_md(vault_dir / "papers" / f"{slug}.md", fm, "\n".join(body_parts))

    # ---- _graph aux tables ----
    write_md(
        vault_dir / "_graph" / "predicates.md",
        {"type": "predicates", "rows": _all_rows(conn, "predicates")},
        "Predicate vocabulary (domain/range). Authoritative in frontmatter.",
    )
    write_md(
        vault_dir / "_graph" / "entity_edges.md",
        {"type": "entity_edges", "rows": _all_rows(conn, "entity_edges")},
        "Navigation edges. Ignored by the gap scanner.",
    )
    write_md(
        vault_dir / "_graph" / "citation_links.md",
        {"type": "citation_links", "rows": _all_rows(conn, "citation_links")},
        "CiTO links between in-corpus papers.",
    )

    # ---- full machine snapshot (ALL rows, including dangling aliases) ----
    dump = {
        t: _all_rows(conn, t)
        for t in (
            "papers", "sections", "paper_authors", "entities", "predicates",
            "claims", "entity_edges", "citation_links", "aliases",
        )
    }
    (vault_dir / "graph-export.json").write_text(
        json.dumps(dump, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    _write_schema(vault_dir)
    conn.close()

    counts = {t: len(dump[t]) for t in dump}
    return counts


def _write_schema(vault_dir: Path) -> None:
    if (vault_dir / "SCHEMA.md").exists():
        return  # don't overwrite hand-edited schema
    (vault_dir / "SCHEMA.md").write_text(SCHEMA_DOC, encoding="utf-8")


SCHEMA_DOC = """# Vault schema — the contract between markdown and the derived index

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
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <db_path> <vault_dir>", file=sys.stderr)
        sys.exit(2)
    db_path = sys.argv[1]
    vault_dir = sys.argv[2]
    counts = export(db_path, vault_dir)
    print(f"Exported {db_path} -> {vault_dir}")
    for t, n in sorted(counts.items()):
        print(f"  {t:16} {n}")


if __name__ == "__main__":
    main()
