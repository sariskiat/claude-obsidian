"""Native five-species gap scanner for the claim-centric knowledge graph.

Runs on the DERIVED db (``.vault-meta/graph/graph.db``), never the live db.
Uses ``graph_db.root()`` for entity resolution — no broken COALESCE.

Gap species:
  1. Coverage     — canonical entities with zero (or very few) claims
  2. Frontier     — author-flagged open-question / limitation claims
  3. Debate       — conflicting polarity on same <s,p,o>
  4. Replication  — result claims with support=1 (only one paper)
  5. White-space  — entity clusters with no bridging claims (Louvain, seed=42)

Usage:
    uv run python scripts/graph-gaps.py
    uv run python scripts/graph-gaps.py --json --top 10
    uv run python scripts/graph-gaps.py --db /path/to/graph.db
"""

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from graph_db import connect, root  # noqa: E402

DEFAULT_DB = Path(__file__).resolve().parent.parent / ".vault-meta" / "graph" / "graph.db"

MIN_COMMUNITY_SIZE = 5
SPECIES_PRIORITY = ["frontier", "debate", "replication", "coverage", "white-space"]
SPECIES_MULTIPLIER = {
    "frontier": 1.00,
    "debate": 0.98,
    "replication": 0.96,
    "coverage": 0.94,
    "white-space": 0.92,
}
STRENGTH_WEIGHT = {"strong": 0.9, "moderate": 0.6, "tentative": 0.3}


# ---------------------------------------------------------------------------
# data class
# ---------------------------------------------------------------------------

@dataclass
class Gap:
    species: str
    description: str
    entities: list[str]
    claims: list[int]
    gap_confidence: float
    explanation: str


# ---------------------------------------------------------------------------
# helpers — all entity resolution goes through root(), never COALESCE
# ---------------------------------------------------------------------------

def _canonical_id(conn, entity_id: int) -> int:
    """Resolve to the true canonical root. Uses root(), not COALESCE."""
    try:
        return root(conn, entity_id, compress=False)
    except Exception:
        return entity_id


def _entity_name(conn, entity_id: int) -> str:
    row = conn.execute("SELECT name FROM entities WHERE id = ?", (entity_id,)).fetchone()
    return row[0] if row else f"<entity:{entity_id}>"


def _merge_confidence(conn, entity_id: int) -> float:
    row = conn.execute(
        "SELECT merge_confidence FROM entities WHERE id = ?", (entity_id,)
    ).fetchone()
    if row and row[0] is not None:
        return row[0]
    return 1.0


def _min_merge_confidence(conn, entity_ids: list[int]) -> float:
    if not entity_ids:
        return 1.0
    return min(_merge_confidence(conn, eid) for eid in entity_ids)


# ---------------------------------------------------------------------------
# species 1: coverage gaps
# ---------------------------------------------------------------------------

def _scan_coverage_gaps(conn) -> list[Gap]:
    entity_claim_count = defaultdict(int)
    for row in conn.execute(
        "SELECT subject_entity_id, object_entity_id FROM claims"
    ):
        entity_claim_count[_canonical_id(conn, row[0])] += 1
        entity_claim_count[_canonical_id(conn, row[1])] += 1

    canonicals = conn.execute(
        "SELECT id, name, super_type FROM entities WHERE canonical_id IS NULL"
    ).fetchall()

    if not canonicals:
        return []

    avg_claims = sum(entity_claim_count.values()) / len(canonicals) if canonicals else 0

    gaps = []
    for eid, name, stype in canonicals:
        count = entity_claim_count.get(eid, 0)
        if count == 0:
            conf = 1.0 * _merge_confidence(conn, eid)
            gaps.append(Gap(
                species="coverage",
                description=f"No claims involve '{name}' ({stype}) — "
                            f"this entity is in the graph but never asserted, "
                            f"compared, or evaluated",
                entities=[name],
                claims=[],
                gap_confidence=round(conf, 3),
                explanation=f"Zero claims; entity merge confidence={_merge_confidence(conn, eid):.2f}; "
                            f"avg claims/entity={avg_claims:.1f}",
            ))
        elif count < max(1, avg_claims * 0.3):
            density_factor = 1.0 - (count / max(1, avg_claims))
            conf = round(density_factor * _merge_confidence(conn, eid), 3)
            gaps.append(Gap(
                species="coverage",
                description=f"'{name}' ({stype}) has only {count} claim(s) "
                            f"(avg={avg_claims:.1f}) — lightly explored",
                entities=[name],
                claims=[],
                gap_confidence=conf,
                explanation=f"Claim density {count}/{avg_claims:.1f} avg; "
                            f"entity merge confidence={_merge_confidence(conn, eid):.2f}",
            ))

    gaps.sort(key=lambda g: g.gap_confidence, reverse=True)
    return gaps


# ---------------------------------------------------------------------------
# species 2: frontier gaps
# ---------------------------------------------------------------------------

def _scan_frontier_gaps(conn) -> list[Gap]:
    rows = conn.execute("""
        SELECT c.id, c.text, c.claim_type, c.strength, c.verbatim_quote,
               c.subject_entity_id, c.object_entity_id, c.source_paper,
               p.title
        FROM claims c
        JOIN papers p ON c.source_paper = p.slug
        WHERE c.claim_type IN ('open-question', 'limitation')
        ORDER BY
            CASE c.strength
                WHEN 'strong' THEN 1
                WHEN 'moderate' THEN 2
                WHEN 'tentative' THEN 3
            END
    """).fetchall()

    gaps = []
    for row in rows:
        cid, text, ctype, strength, quote, subj_id, obj_id, paper_slug, paper_title = row
        subj_name = _entity_name(conn, subj_id)
        obj_name = _entity_name(conn, obj_id)
        strength_w = STRENGTH_WEIGHT.get(strength, 0.5)
        merge_conf = _min_merge_confidence(conn, [subj_id, obj_id])
        conf = round(strength_w * merge_conf, 3)

        gaps.append(Gap(
            species="frontier",
            description=f"[{ctype}] {text}",
            entities=[subj_name, obj_name],
            claims=[cid],
            gap_confidence=conf,
            explanation=f"Author-flagged as {ctype} with {strength} confidence "
                        f"(from '{paper_title}'); "
                        f"min merge-confidence={merge_conf:.2f}; "
                        f"quote: \"{quote[:120]}…\"",
        ))

    gaps.sort(key=lambda g: g.gap_confidence, reverse=True)
    return gaps


# ---------------------------------------------------------------------------
# species 3: debate gaps
# ---------------------------------------------------------------------------

def _scan_debate_gaps(conn) -> list[Gap]:
    groups = defaultdict(list)
    for row in conn.execute(
        "SELECT c.id, c.subject_entity_id, c.predicate, c.object_entity_id, "
        "c.polarity, c.text, c.source_paper, c.strength FROM claims c"
    ):
        cid, subj_id, pred, obj_id, polarity, text, paper, strength = row
        subj_canon = _canonical_id(conn, subj_id)
        obj_canon = _canonical_id(conn, obj_id)
        key = (subj_canon, pred, obj_canon)
        groups[key].append({
            "id": cid, "polarity": polarity, "text": text,
            "paper": paper, "strength": strength,
            "subj_id": subj_canon, "obj_id": obj_canon,
        })

    gaps = []
    for (subj_canon, pred, obj_canon), claim_list in groups.items():
        pols = {c["polarity"] for c in claim_list}
        if len(pols) < 2:
            continue

        subj_name = _entity_name(conn, subj_canon)
        obj_name = _entity_name(conn, obj_canon)
        merge_conf = _min_merge_confidence(conn, [subj_canon, obj_canon])

        asserts = sum(1 for c in claim_list if c["polarity"] == "asserts")
        refutes = len(claim_list) - asserts
        polarity_factor = 1.0 - abs(asserts - refutes) / len(claim_list)
        strength_vals = [STRENGTH_WEIGHT.get(c["strength"], 0.5) for c in claim_list]
        avg_strength = sum(strength_vals) / len(strength_vals)

        conf = round(polarity_factor * avg_strength * merge_conf, 3)

        gaps.append(Gap(
            species="debate",
            description=f"Unresolved contradiction: {asserts} paper(s) assert vs "
                        f"{refutes} refute that '{subj_name}' --[{pred}]--> '{obj_name}'",
            entities=[subj_name, obj_name],
            claims=[c["id"] for c in claim_list],
            gap_confidence=conf,
            explanation=f"Polarity conflict ({asserts}:{refutes}); "
                        f"polarity-balance={polarity_factor:.2f}; "
                        f"avg claim strength={avg_strength:.2f}; "
                        f"min merge-confidence={merge_conf:.2f}",
        ))

    gaps.sort(key=lambda g: g.gap_confidence, reverse=True)
    return gaps


# ---------------------------------------------------------------------------
# species 4: replication gaps
# ---------------------------------------------------------------------------

def _scan_replication_gaps(conn) -> list[Gap]:
    rows = conn.execute("""
        SELECT c.id, c.text, c.support, c.strength, c.claim_type,
               c.subject_entity_id, c.object_entity_id,
               c.source_paper, p.title
        FROM claims c
        JOIN papers p ON c.source_paper = p.slug
        WHERE c.support = 1
          AND c.claim_type = 'result'
        ORDER BY
            CASE c.strength
                WHEN 'strong' THEN 1
                WHEN 'moderate' THEN 2
                WHEN 'tentative' THEN 3
            END
    """).fetchall()

    gaps = []
    for row in rows:
        cid, text, support, strength, ctype, subj_id, obj_id, paper_slug, paper_title = row
        subj_name = _entity_name(conn, subj_id)
        obj_name = _entity_name(conn, obj_id)
        strength_w = STRENGTH_WEIGHT.get(strength, 0.5)
        merge_conf = _min_merge_confidence(conn, [subj_id, obj_id])
        rep_factor = 1.0 / (support + 1)
        conf = round(strength_w * rep_factor * merge_conf, 3)

        gaps.append(Gap(
            species="replication",
            description=f"Unreplicated result: {text}",
            entities=[subj_name, obj_name],
            claims=[cid],
            gap_confidence=conf,
            explanation=f"Support={support} (only from '{paper_title}'); "
                        f"claim strength={strength}; "
                        f"min merge-confidence={merge_conf:.2f}; "
                        f"→ needs independent verification",
        ))

    gaps.sort(key=lambda g: g.gap_confidence, reverse=True)
    return gaps


# ---------------------------------------------------------------------------
# species 5: white-space (cross-topic) — Louvain clustering
# ---------------------------------------------------------------------------

def _scan_whitespace_gaps(conn) -> list[Gap]:
    try:
        import networkx as nx
    except ImportError:
        return [_whitespace_fallback(conn)]

    G = nx.Graph()
    for row in conn.execute(
        "SELECT id, name, super_type FROM entities WHERE canonical_id IS NULL"
    ):
        G.add_node(row[0], name=row[1], super_type=row[2])

    for row in conn.execute(
        "SELECT subject_entity_id, object_entity_id FROM claims"
    ):
        subj_canon = _canonical_id(conn, row[0])
        obj_canon = _canonical_id(conn, row[1])
        if subj_canon != obj_canon and subj_canon in G and obj_canon in G:
            if G.has_edge(subj_canon, obj_canon):
                G[subj_canon][obj_canon]["weight"] += 1
            else:
                G.add_edge(subj_canon, obj_canon, weight=1)

    if G.number_of_nodes() < 4 or G.number_of_edges() < 2:
        return [_whitespace_fallback(conn, "Graph too small for clustering")]

    try:
        communities = nx.community.louvain_communities(G, weight="weight", seed=42)
    except Exception:
        communities = list(nx.connected_components(G))

    if len(communities) < 2:
        return []

    node_to_community = {}
    for i, comm in enumerate(communities):
        for node in comm:
            node_to_community[node] = i

    cross_edges = defaultdict(int)
    for row in conn.execute(
        "SELECT subject_entity_id, object_entity_id FROM claims"
    ):
        subj_canon = _canonical_id(conn, row[0])
        obj_canon = _canonical_id(conn, row[1])
        if subj_canon in node_to_community and obj_canon in node_to_community:
            c_subj = node_to_community[subj_canon]
            c_obj = node_to_community[obj_canon]
            if c_subj != c_obj:
                pair = tuple(sorted([c_subj, c_obj]))
                cross_edges[pair] += 1

    all_pairs = set()
    for i in range(len(communities)):
        for j in range(i + 1, len(communities)):
            all_pairs.add((i, j))

    missing_pairs = all_pairs - set(cross_edges.keys())

    gaps = []
    for ci, cj in sorted(missing_pairs):
        comm_i = communities[ci]
        comm_j = communities[cj]

        if min(len(comm_i), len(comm_j)) < MIN_COMMUNITY_SIZE:
            continue

        rep_i = sorted([G.nodes[n]["name"] for n in comm_i])[:5]
        rep_j = sorted([G.nodes[n]["name"] for n in comm_j])[:5]

        types_i = {G.nodes[n]["super_type"] for n in comm_i}
        types_j = {G.nodes[n]["super_type"] for n in comm_j}

        size_factor = min(len(comm_i), len(comm_j)) / max(len(comm_i), len(comm_j))
        merge_confs_i = [_merge_confidence(conn, n) for n in comm_i]
        merge_confs_j = [_merge_confidence(conn, n) for n in comm_j]
        avg_merge_conf = (
            sum(merge_confs_i) / len(merge_confs_i) +
            sum(merge_confs_j) / len(merge_confs_j)
        ) / 2

        conf = round(size_factor * avg_merge_conf, 3)

        gaps.append(Gap(
            species="white-space",
            description=f"No claims bridge community {ci} ({', '.join(sorted(types_i))}) "
                        f"and community {cj} ({', '.join(sorted(types_j))}) — "
                        f"{len(comm_i)} + {len(comm_j)} entities with no connection",
            entities=rep_i + ["…"] + rep_j,
            claims=[],
            gap_confidence=conf,
            explanation=f"Community {ci}: {len(comm_i)} entities, "
                        f"types={types_i}, "
                        f"avg merge-conf={sum(merge_confs_i)/len(merge_confs_i):.2f}; "
                        f"Community {cj}: {len(comm_j)} entities, "
                        f"types={types_j}, "
                        f"avg merge-conf={sum(merge_confs_j)/len(merge_confs_j):.2f}; "
                        f"size-balance={size_factor:.2f}",
        ))

    gaps.sort(key=lambda g: g.gap_confidence, reverse=True)
    return gaps


def _whitespace_fallback(conn=None, reason: str = "NetworkX not available") -> Gap:
    return Gap(
        species="white-space",
        description=f"Cross-topic white-space scan skipped: {reason}",
        entities=[],
        claims=[],
        gap_confidence=0.0,
        explanation=f"White-space requires graph clustering (NetworkX Louvain). {reason}.",
    )


# ---------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------

def scan_all_gaps(conn, top_n: int = 20) -> list[Gap]:
    all_gaps: list[Gap] = []
    all_gaps.extend(_scan_coverage_gaps(conn))
    all_gaps.extend(_scan_frontier_gaps(conn))
    all_gaps.extend(_scan_debate_gaps(conn))
    all_gaps.extend(_scan_replication_gaps(conn))
    all_gaps.extend(_scan_whitespace_gaps(conn))

    for g in all_gaps:
        multiplier = SPECIES_MULTIPLIER.get(g.species, 1.0)
        g.gap_confidence = round(g.gap_confidence * multiplier, 4)

    all_gaps.sort(key=lambda g: g.gap_confidence, reverse=True)
    return all_gaps[:top_n]


def format_report(gaps: list[Gap]) -> str:
    species_emoji = {
        "coverage": "📭",
        "frontier": "🚩",
        "debate": "⚡",
        "replication": "🔁",
        "white-space": "🌌",
    }

    if not gaps:
        return "No gaps found. The graph may be too small or fully saturated."

    lines = [
        "# Gap Scan Report",
        "",
        f"**{len(gaps)} gaps found**, ranked by gap-confidence (highest first).",
        "",
        "| # | Confidence | Species | Description |",
        "|---|------------|---------|-------------|",
    ]

    for i, g in enumerate(gaps, 1):
        emoji = species_emoji.get(g.species, "❓")
        desc = g.description[:120] + ("…" if len(g.description) > 120 else "")
        lines.append(
            f"| {i} | {g.gap_confidence:.3f} | {emoji} {g.species} | {desc} |"
        )

    lines.append("")
    lines.append("---")
    lines.append("")

    for i, g in enumerate(gaps, 1):
        emoji = species_emoji.get(g.species, "❓")
        lines.append(f"### {i}. {emoji} [{g.species}] {g.description[:150]}")
        lines.append(f"**Confidence:** {g.gap_confidence:.3f}")
        if g.entities:
            lines.append(f"**Entities:** {', '.join(g.entities[:10])}")
        if g.claims:
            lines.append(f"**Claims:** {', '.join(f'#{c}' for c in g.claims[:10])}")
        lines.append(f"**Why:** {g.explanation}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Five-species gap scanner")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--top", type=int, default=20, help="Max gaps to return")
    parser.add_argument("--db", type=str, default=str(DEFAULT_DB),
                        help="Path to graph.db (default: derived db)")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Error: database not found at {db_path}", file=sys.stderr)
        print("Run: uv run python scripts/graph-build.py wiki/graph/ .vault-meta/graph/graph.db",
              file=sys.stderr)
        sys.exit(1)

    conn = connect(db_path)
    # Verify schema has claims table
    has_claims = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='claims'"
    ).fetchone()
    if not has_claims:
        print("Error: database missing claims table.", file=sys.stderr)
        conn.close()
        sys.exit(1)

    gaps = scan_all_gaps(conn, top_n=args.top)

    if args.json:
        output = [
            {
                "species": g.species,
                "description": g.description,
                "entities": g.entities,
                "claims": g.claims,
                "gap_confidence": g.gap_confidence,
                "explanation": g.explanation,
            }
            for g in gaps
        ]
        print(json.dumps(output, indent=2))
    else:
        print(format_report(gaps))

    conn.close()


if __name__ == "__main__":
    main()
