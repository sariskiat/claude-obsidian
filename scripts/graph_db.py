"""Graph database helpers — the single shared module for connect, root(), and inserts.

Every script that touches the graph db imports from here. No inline
COALESCE(canonical_id, id) anywhere — use root() instead.

    from graph_db import connect, root, insert_entity, insert_claim

Usage:
    conn = connect(db_path)                          # FK on
    eid = insert_entity(conn, name, super_type, ...)  # typed insert
    cid = insert_claim(conn, subj_id, pred, obj_id, ...)
    canonical = root(conn, entity_id)                 # chain-resolve
"""

import sqlite3
from pathlib import Path


# ---------------------------------------------------------------------------
# connect
# ---------------------------------------------------------------------------

def connect(db_path):
    """Open a sqlite connection with foreign keys enabled.

    Creates parent directories if they don't exist.
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# ---------------------------------------------------------------------------
# root — the single shared rollup helper
# ---------------------------------------------------------------------------

def root(conn, entity_id, compress=True):
    """Follow canonical_id to the ultimate canonical root.

    Replaces all inline ``COALESCE(canonical_id, id)`` — which is single-hop
    and silently wrong on chains (A→B→C reports B, not C).

    Semantics:
      - Chain: A→B→C (C.canonical_id IS NULL) → returns C.
      - Cycle: A⇄B or A→A → elects min id on the cycle, refuses the loop.
      - Dangling: canonical_id points to a missing row → stops at last live id.
      - Path-compression: when *compress* is True, rewrites every pointer
        visited to point straight at the root (union-find style).
    """
    seen = []
    seen_set = set()
    cur = entity_id

    while True:
        row = conn.execute(
            "SELECT canonical_id FROM entities WHERE id = ?", (cur,)
        ).fetchone()

        if row is None:
            # Dangling — cur is not a live row. Back off to last live id.
            cur = seen[-1] if seen else entity_id
            break

        parent = row[0]
        if parent is None:
            # cur IS canonical — found the root.
            break

        if parent in seen_set or parent == cur:
            # Cycle (including self-loop). Elect min id as root.
            # Also record cur so it gets path-compressed along with the chain.
            seen.append(cur)
            cur = min(seen_set | {cur})
            break

        seen.append(cur)
        seen_set.add(cur)
        cur = parent

    if compress and seen:
        conn.executemany(
            "UPDATE entities SET canonical_id = ? WHERE id = ? AND id != ?",
            [(cur, sid, cur) for sid in seen],
        )
        # If we resolved a cycle, the elected root may still point into the
        # cycle. Clear its canonical_id so it is truly canonical.
        conn.execute(
            "UPDATE entities SET canonical_id = NULL WHERE id = ?", (cur,)
        )

    return cur


# ---------------------------------------------------------------------------
# typed insert helpers
# ---------------------------------------------------------------------------

def insert_entity(conn, eid, name, super_type, sub_type, description,
                  source_paper, is_canonical=True, canonical_id=None,
                  merge_confidence=None, aliases=None, metadata=None):
    """Insert an entity with an explicit id (ids are preserved across round-trip).

    Does NOT auto-create an alias row — alias handling is the caller's
    responsibility (export/build preserves all alias rows verbatim).
    """
    import json as _json
    conn.execute(
        """INSERT INTO entities
           (id, name, super_type, sub_type, description, source_paper,
            is_canonical, canonical_id, merge_confidence, metadata)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (eid, name, super_type, sub_type, description, source_paper,
         int(is_canonical), canonical_id, merge_confidence,
         _json.dumps(metadata) if metadata else None),
    )


def insert_predicate(conn, name, domain_super_type, range_super_type):
    """Insert a predicate. Returns row id."""
    cur = conn.execute(
        "INSERT INTO predicates (name, domain_super_type, range_super_type) "
        "VALUES (?, ?, ?)",
        (name, domain_super_type, range_super_type),
    )
    return cur.lastrowid


def insert_claim(conn, claim_id, subject_entity_id, predicate, object_entity_id,
                 text, verbatim_quote, claim_type, polarity, strength,
                 source_paper, section_id=None, confidence=None,
                 generated_by="claude-opus-4-8@v1", status="confirmed",
                 support=1):
    """Insert a claim with an explicit id (ids preserved across round-trip)."""
    conn.execute(
        """INSERT INTO claims
           (id, subject_entity_id, predicate, object_entity_id,
            text, verbatim_quote, claim_type, polarity, strength,
            support, generated_by, source_paper, section_id, confidence, status)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (claim_id, subject_entity_id, predicate, object_entity_id,
         text, verbatim_quote, claim_type, polarity, strength,
         support, generated_by, source_paper, section_id, confidence, status),
    )


def insert_section(conn, section_id, paper_slug, heading, role, summary,
                   order_index):
    """Insert a section with explicit id."""
    conn.execute(
        "INSERT INTO sections (id, paper_slug, heading, role, summary, order_index) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (section_id, paper_slug, heading, role, summary, order_index),
    )


def insert_paper(conn, paper_id, slug, title, authors, source_path):
    """Insert a paper with explicit id."""
    conn.execute(
        "INSERT INTO papers (id, slug, title, authors, source_path) "
        "VALUES (?, ?, ?, ?, ?)",
        (paper_id, slug, title, authors, source_path),
    )


def insert_paper_author(conn, paper_id, author_name):
    """Insert a paper-author row."""
    conn.execute(
        "INSERT INTO paper_authors (paper_id, author_name) VALUES (?, ?)",
        (paper_id, author_name),
    )


def insert_alias(conn, alias, canonical_id):
    """Insert an alias row verbatim — no case-fold, no normalize, no FK check.

    The derived aliases table must NOT enforce REFERENCES entities(id) on
    these rows, otherwise the 55 dangling-FK alias rows can't be re-inserted.
    """
    conn.execute(
        "INSERT INTO aliases (alias, canonical_id) VALUES (?, ?)",
        (alias, canonical_id),
    )


def insert_entity_edge(conn, source_entity_id, target_entity_id, predicate,
                       confidence, source_paper):
    """Insert a lightweight navigation edge."""
    conn.execute(
        "INSERT INTO entity_edges (source_entity_id, target_entity_id, "
        "predicate, confidence, source_paper) VALUES (?, ?, ?, ?, ?)",
        (source_entity_id, target_entity_id, predicate, confidence, source_paper),
    )


def insert_citation_link(conn, claim_id, source_paper):
    """Insert a citation link row."""
    conn.execute(
        "INSERT INTO citation_links (claim_id, source_paper) VALUES (?, ?)",
        (claim_id, source_paper),
    )
