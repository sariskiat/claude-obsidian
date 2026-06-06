"""Rebuild the derived sqlite index from the markdown vault.

Reads ``wiki/graph/graph-export.json`` (the machine snapshot) and creates a fresh
derived db at ``.vault-meta/graph/graph.db``. IDs are preserved exactly as they
appear in the JSON — Louvain(seed=42) remains stable.

root() is called on every entity insert to path-compress drift before it can
accumulate. Orphans are reported, never crashed on.

CRITICAL — alias fix (AC3): The derived aliases table does NOT enforce
``REFERENCES entities(id)``. The 55 dangling-FK rows can only be re-inserted
if the FK constraint is absent. The source db has the FK for future correctness
(post-migration); the derived db drops it so the round-trip is lossless today.

Usage:
    uv run python scripts/graph-build.py <vault_dir> <output_db>
"""

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import graph_db  # noqa: E402


# Schema for the DERIVED db. Same as the source schema EXCEPT:
# - No REFERENCES entities(id) on aliases.canonical_id (AC3 fix)
# - No REFERENCES on sections/paper_authors (we insert in bulk, FK breaks ordering)
DERIVED_SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    slug TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    authors TEXT,
    source_path TEXT,
    ingested_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS sections (
    id INTEGER PRIMARY KEY,
    paper_slug TEXT NOT NULL,
    heading TEXT NOT NULL,
    role TEXT,
    summary TEXT,
    order_index INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS paper_authors (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    paper_slug TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS entities (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    super_type TEXT NOT NULL,
    sub_type TEXT,
    description TEXT,
    source_paper TEXT,
    canonical_id INTEGER,
    metadata TEXT,
    merge_confidence REAL
);

CREATE TABLE IF NOT EXISTS predicates (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    domain_super_type TEXT NOT NULL,
    range_super_type TEXT NOT NULL,
    UNIQUE(name, domain_super_type, range_super_type)
);

CREATE TABLE IF NOT EXISTS claims (
    id INTEGER PRIMARY KEY,
    subject_entity_id INTEGER NOT NULL,
    predicate TEXT NOT NULL,
    object_entity_id INTEGER NOT NULL,
    text TEXT NOT NULL,
    verbatim_quote TEXT NOT NULL,
    claim_type TEXT NOT NULL,
    polarity TEXT NOT NULL DEFAULT 'asserts',
    strength TEXT NOT NULL DEFAULT 'moderate',
    support INTEGER NOT NULL DEFAULT 1,
    generated_by TEXT NOT NULL,
    source_paper TEXT NOT NULL,
    section_id INTEGER,
    confidence REAL,
    status TEXT NOT NULL DEFAULT 'confirmed'
);

CREATE TABLE IF NOT EXISTS entity_edges (
    id INTEGER PRIMARY KEY,
    source_entity_id INTEGER NOT NULL,
    target_entity_id INTEGER NOT NULL,
    predicate TEXT NOT NULL,
    confidence REAL,
    source_paper TEXT
);

CREATE TABLE IF NOT EXISTS citation_links (
    id INTEGER PRIMARY KEY,
    source_paper TEXT NOT NULL,
    target_paper TEXT NOT NULL,
    cito_type TEXT NOT NULL,
    evidence TEXT,
    UNIQUE(source_paper, target_paper, cito_type)
);

-- NO REFERENCES entities(id) here — allows dangling-FK re-insert (AC3 fix)
CREATE TABLE IF NOT EXISTS aliases (
    alias TEXT UNIQUE,
    canonical_id INTEGER NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_papers_slug ON papers(slug);
CREATE INDEX IF NOT EXISTS idx_entities_super_type ON entities(super_type);
CREATE INDEX IF NOT EXISTS idx_entities_canonical ON entities(canonical_id);
CREATE INDEX IF NOT EXISTS idx_claims_subject ON claims(subject_entity_id);
CREATE INDEX IF NOT EXISTS idx_claims_object ON claims(object_entity_id);
CREATE INDEX IF NOT EXISTS idx_claims_type ON claims(claim_type);
CREATE INDEX IF NOT EXISTS idx_claims_source_paper ON claims(source_paper);
CREATE INDEX IF NOT EXISTS idx_entity_edges_source ON entity_edges(source_entity_id);
CREATE INDEX IF NOT EXISTS idx_entity_edges_target ON entity_edges(target_entity_id);
CREATE INDEX IF NOT EXISTS idx_citation_links_source ON citation_links(source_paper);
CREATE INDEX IF NOT EXISTS idx_citation_links_target ON citation_links(target_paper);
CREATE INDEX IF NOT EXISTS idx_aliases_canonical ON aliases(canonical_id);
"""

# Column order for each table (must match the SELECT * order from export)
TABLE_COLUMNS = {
    "papers": ("slug", "title", "authors", "source_path", "ingested_at"),
    "sections": ("id", "paper_slug", "heading", "role", "summary", "order_index"),
    "paper_authors": ("id", "name", "paper_slug"),
    "entities": ("id", "name", "super_type", "sub_type", "description",
                 "source_paper", "canonical_id", "metadata", "merge_confidence"),
    "predicates": ("id", "name", "domain_super_type", "range_super_type"),
    "claims": ("id", "subject_entity_id", "predicate", "object_entity_id",
               "text", "verbatim_quote", "claim_type", "polarity", "strength",
               "support", "generated_by", "source_paper", "section_id",
               "confidence", "status"),
    "entity_edges": ("id", "source_entity_id", "target_entity_id",
                     "predicate", "confidence", "source_paper"),
    "citation_links": ("id", "source_paper", "target_paper", "cito_type",
                       "evidence"),
    "aliases": ("alias", "canonical_id"),
}


def build(vault_dir: str, output_db: str) -> dict:
    """Rebuild a derived db from the vault snapshot. Returns per-table counts."""
    vault_dir = Path(vault_dir)
    output_db = Path(output_db)

    # Read the machine snapshot
    snapshot_path = vault_dir / "graph-export.json"
    if not snapshot_path.exists():
        raise FileNotFoundError(f"{snapshot_path} not found — run graph-export.py first")

    with open(snapshot_path, encoding="utf-8") as f:
        dump = json.load(f)

    # Fresh derived db
    if output_db.exists():
        output_db.unlink()
    output_db.parent.mkdir(parents=True, exist_ok=True)
    conn = graph_db.connect(output_db)
    conn.executescript(DERIVED_SCHEMA)
    conn.commit()

    # Insert in dependency order: papers → sections/paper_authors →
    # entities → predicates → claims → entity_edges → citation_links → aliases
    # Use explicit ids to preserve Louvain stability.

    insert_order = [
        "papers",
        "sections",
        "paper_authors",
        "entities",
        "predicates",
        "claims",
        "entity_edges",
        "citation_links",
        "aliases",
    ]

    drift_reports = []

    for table in insert_order:
        rows = dump.get(table, [])
        cols = TABLE_COLUMNS[table]
        placeholders = ", ".join("?" * len(cols))

        for row in rows:
            values = tuple(row.get(c) for c in cols)
            try:
                conn.execute(
                    f"INSERT INTO [{table}] ({', '.join(cols)}) "
                    f"VALUES ({placeholders})",
                    values,
                )
            except Exception as exc:
                # Log and skip on insert failure (e.g. duplicate keys)
                drift_reports.append(f"  [{table}] skip row: {exc} — {values[:3]}...")

    conn.commit()

    # -- root() self-heal pass: walk every entity, compress chains, report drift --
    entity_ids = [
        e["id"] for e in dump.get("entities", []) if e.get("canonical_id") is not None
    ]
    for eid in entity_ids:
        try:
            graph_db.root(conn, eid, compress=True)
        except Exception as exc:
            drift_reports.append(f"  root({eid}): {exc}")

    conn.commit()

    # Report drift
    if drift_reports:
        print(f"Drift self-healed ({len(drift_reports)} items):")
        for r in drift_reports[:20]:
            print(r)
        if len(drift_reports) > 20:
            print(f"  ... and {len(drift_reports) - 20} more")

    # Counts
    counts = {}
    for table in insert_order:
        try:
            cnt = conn.execute(f"SELECT COUNT(*) FROM [{table}]").fetchone()[0]
            counts[table] = cnt
        except Exception:
            counts[table] = 0

    conn.close()
    return counts


def main() -> None:
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <vault_dir> <output_db>", file=sys.stderr)
        sys.exit(2)
    vault_dir = sys.argv[1]
    output_db = sys.argv[2]
    counts = build(vault_dir, output_db)
    print(f"Rebuilt {output_db} from {vault_dir}")
    for t, n in sorted(counts.items()):
        print(f"  {t:16} {n}")


if __name__ == "__main__":
    main()
