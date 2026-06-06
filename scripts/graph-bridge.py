#!/usr/bin/env python3
"""graph-bridge.py — Proposal Finder for white-space research gaps (P5).

Turns white-space community pairs (zero cross-community claim-edges, per the
seed=42 Louvain clustering identical to graph-gaps.py) into ranked, grounded
next-paper proposals.

Each proposal:
  - id                : "{comm_a_id}-{comm_b_id}"
  - community_a/b     : {id, size, members (up to 5), super_types}
  - anchor_entities   : most-connected canonical entities per side (list of {id, name, degree})
  - anchor_papers     : papers contributing anchor entities/claims
  - score             : weighted heuristic in [0, 1]
  - signal_breakdown  : {gap_confidence, bridgeability, limitation_pull, richness,
                         direction_relevance}
  - passages          : list of {paper, snippet} from graph-retrieve (degrade gracefully)
  - already_proposed  : true if either community contains a known bridge entity (FR7)
  - justification     : optional narrative (only with --synthesize)

Ranking is deterministic (seed=42, stable tie-breaks on (score DESC, id ASC)).

Usage:
    uv run python scripts/graph-bridge.py
    uv run python scripts/graph-bridge.py --json --top 10
    uv run python scripts/graph-bridge.py --json --top 10 --synthesize
    uv run python scripts/graph-bridge.py --db /path/to/graph.db

Exit codes:
  0  — success (proposals list may be empty)
  1  — database missing or unreadable (non-zero, build hint emitted)
"""

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from graph_db import connect, root  # noqa: E402

VAULT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = VAULT_ROOT / ".vault-meta" / "graph" / "graph.db"
DEFAULT_BM25_INDEX = VAULT_ROOT / ".vault-meta" / "graph" / "bm25" / "index.json"
DEFAULT_CHUNKS_DIR = VAULT_ROOT / ".vault-meta" / "graph" / "chunks"

# ---------------------------------------------------------------------------
# Clustering parameters — must match graph-gaps.py exactly
# ---------------------------------------------------------------------------
MIN_COMMUNITY_SIZE = 5

# ---------------------------------------------------------------------------
# Priority areas for direction_relevance (FR3 / AC3)
# Entity names touching BOTH sets get the boost.
# Set A: virtual try-on domain
# Set B: Aek / diffusion-sampling domain
# ---------------------------------------------------------------------------
PRIORITY_KEYWORDS_A = frozenset([
    "virtual try-on", "virtual tryon", "try-on", "vton", "try on",
])
PRIORITY_KEYWORDS_B = frozenset([
    "diffusion sampling", "diffusion noise", "noise optimization",
    "distillation", "heavy ball", "momentum", "dno", "neon",
    "negative extrapolation", "score identity", "universal inverse",
])

# Known bridge entities for FR7 dedup flag
# Matched case-insensitively by name substring
KNOWN_BRIDGE_ENTITY_NAMES = frozenset([
    "extrapolation-based iterate correction",
])

# ---------------------------------------------------------------------------
# Default signal weights (CLI-tunable, FR2)
# Weights are normalised to [0,1] via a weighted sum where each term is
# already bounded to [0,1].  Total weight sum should equal 1.0.
# ---------------------------------------------------------------------------
DEFAULT_WEIGHTS = {
    "gap_confidence": 0.25,
    "bridgeability": 0.20,
    "limitation_pull": 0.15,
    "richness": 0.15,
    "direction_relevance": 0.25,
}

# Number of anchor entities to surface per community side
ANCHOR_ENTITIES_PER_SIDE = 3
# Number of passages to retrieve per proposal
PASSAGES_PER_PROPOSAL = 2
# Maximum members listed in community_a/community_b (mirrors gap entity truncation)
MEMBERS_TRUNCATE = 5


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Proposal:
    id: str
    community_a: dict
    community_b: dict
    anchor_entities: list
    anchor_papers: list
    score: float
    signal_breakdown: dict
    passages: list
    already_proposed: bool = False
    justification: str = None


# ---------------------------------------------------------------------------
# Entity resolution (delegate to graph_db.root)
# ---------------------------------------------------------------------------

def _canonical_id(conn, entity_id: int) -> int:
    """Resolve via graph_db.root() — never COALESCE."""
    try:
        return root(conn, entity_id, compress=False)
    except Exception:
        return entity_id


def _entity_name(conn, entity_id: int) -> str:
    row = conn.execute("SELECT name FROM entities WHERE id = ?",
                       (entity_id,)).fetchone()
    return row[0] if row else f"<entity:{entity_id}>"


def _merge_confidence(conn, entity_id: int) -> float:
    row = conn.execute(
        "SELECT merge_confidence FROM entities WHERE id = ?", (entity_id,)
    ).fetchone()
    return row[0] if row and row[0] is not None else 1.0


# ---------------------------------------------------------------------------
# Build the Louvain graph — IDENTICAL construction to graph-gaps.py
# ---------------------------------------------------------------------------

def _build_louvain_graph(conn):
    """Build networkx graph identical to graph-gaps.py _scan_whitespace_gaps.

    Canonical nodes via graph_db.root(); weighted claim edges.
    Returns (G, communities) where communities is a list of frozensets.
    community_i may be None if graph too small for clustering.
    """
    try:
        import networkx as nx
    except ImportError:
        return None, None

    G = nx.Graph()
    for row in conn.execute(
        "SELECT id, name, super_type FROM entities WHERE canonical_id IS NULL"
    ):
        G.add_node(row[0], name=row[1], super_type=row[2])

    for row in conn.execute(
        "SELECT subject_entity_id, object_entity_id FROM claims"
    ):
        sc = _canonical_id(conn, row[0])
        oc = _canonical_id(conn, row[1])
        if sc != oc and sc in G and oc in G:
            if G.has_edge(sc, oc):
                G[sc][oc]["weight"] += 1
            else:
                G.add_edge(sc, oc, weight=1)

    if G.number_of_nodes() < 4 or G.number_of_edges() < 2:
        return G, None

    try:
        communities = nx.community.louvain_communities(G, weight="weight", seed=42)
    except Exception:
        communities = list(nx.connected_components(G))

    return G, communities


# ---------------------------------------------------------------------------
# Compute signals
# ---------------------------------------------------------------------------

def _compute_gap_confidence(comm_i, comm_j, conn) -> float:
    """Mirrors graph-gaps.py gap_confidence for white-space: size-balance * avg merge-conf."""
    merge_confs_i = [_merge_confidence(conn, n) for n in comm_i]
    merge_confs_j = [_merge_confidence(conn, n) for n in comm_j]
    size_factor = min(len(comm_i), len(comm_j)) / max(len(comm_i), len(comm_j))
    avg_mc = (
        sum(merge_confs_i) / len(merge_confs_i)
        + sum(merge_confs_j) / len(merge_confs_j)
    ) / 2
    return round(size_factor * avg_mc, 4)


def _compute_bridgeability(comm_i_nodes, comm_j_nodes, conn) -> float:
    """Shared-neighbor bridgeability via entity_edges.

    Counts entity_edges that cross the pair (source in comm_i, target in comm_j
    or vice versa). Also counts shared canonical neighbors via entity_edges.
    citation_links is empty (0 rows) — degrades gracefully to 0 contribution.

    Result is normalised to [0, 1] by sigmoid-like saturation: min(count/10, 1.0).
    """
    set_i = set(comm_i_nodes)
    set_j = set(comm_j_nodes)
    cross = 0
    shared_neighbors = set()

    # entity_edges crossing the pair
    for row in conn.execute(
        "SELECT source_entity_id, target_entity_id FROM entity_edges"
    ):
        sc = _canonical_id(conn, row[0])
        tc = _canonical_id(conn, row[1])
        if (sc in set_i and tc in set_j) or (sc in set_j and tc in set_i):
            cross += 1

    # Shared canonical neighbors: entities connected to both communities via claims
    neighbors_i: set = set()
    neighbors_j: set = set()
    for row in conn.execute(
        "SELECT subject_entity_id, object_entity_id FROM claims"
    ):
        sc = _canonical_id(conn, row[0])
        oc = _canonical_id(conn, row[1])
        if sc in set_i or oc in set_i:
            other = oc if sc in set_i else sc
            if other not in set_i:
                neighbors_i.add(other)
        if sc in set_j or oc in set_j:
            other = oc if sc in set_j else sc
            if other not in set_j:
                neighbors_j.add(other)
    shared_neighbors = neighbors_i & neighbors_j

    total = cross + len(shared_neighbors)
    return round(min(total / 10.0, 1.0), 4)


def _compute_limitation_pull(comm_i_nodes, comm_j_nodes, conn) -> float:
    """Count of limitation/open-question claims touching either community.

    Normalised to [0, 1] by saturation at 20 claims.
    """
    set_all = set(comm_i_nodes) | set(comm_j_nodes)
    count = 0
    for row in conn.execute(
        "SELECT subject_entity_id, object_entity_id FROM claims "
        "WHERE claim_type IN ('limitation', 'open-question')"
    ):
        sc = _canonical_id(conn, row[0])
        oc = _canonical_id(conn, row[1])
        if sc in set_all or oc in set_all:
            count += 1
    return round(min(count / 20.0, 1.0), 4)


def _compute_richness(comm_i, comm_j, conn) -> float:
    """min(entities/claims per side) normalised.

    Uses min of the two sides to penalise lopsided pairs.
    """
    set_i = set(comm_i)
    set_j = set(comm_j)

    def _claim_count(nodes):
        c = 0
        for row in conn.execute(
            "SELECT subject_entity_id, object_entity_id FROM claims"
        ):
            sc = _canonical_id(conn, row[0])
            oc = _canonical_id(conn, row[1])
            if sc in nodes or oc in nodes:
                c += 1
        return c

    entities_i = len(comm_i)
    entities_j = len(comm_j)
    claims_i = _claim_count(set_i)
    claims_j = _claim_count(set_j)

    # Score: (min entities * min claims) normalised; saturate at 20 entities and 20 claims
    min_e = min(entities_i, entities_j)
    min_c = min(claims_i, claims_j)
    return round(min(min_e / 20.0, 1.0) * min(min_c / 20.0, 1.0), 4)


def _compute_direction_relevance(comm_i, comm_j, G) -> float:
    """Boost when one community has VTON entities AND the other has Aek entities.

    Checks entity names (case-insensitive substring) against PRIORITY_KEYWORDS_A
    and PRIORITY_KEYWORDS_B.  Returns 1.0 for the gold anchor pair, 0.0 otherwise.
    """
    def _has_keyword(comm_nodes, keywords):
        for n in comm_nodes:
            name = G.nodes[n].get("name", "").lower()
            if any(kw in name for kw in keywords):
                return True
        return False

    i_has_a = _has_keyword(comm_i, PRIORITY_KEYWORDS_A)
    j_has_a = _has_keyword(comm_j, PRIORITY_KEYWORDS_A)
    i_has_b = _has_keyword(comm_i, PRIORITY_KEYWORDS_B)
    j_has_b = _has_keyword(comm_j, PRIORITY_KEYWORDS_B)

    if (i_has_a and j_has_b) or (j_has_a and i_has_b):
        return 1.0
    # Partial boost: one community touches either domain
    partial = 0.0
    if i_has_a or j_has_a:
        partial += 0.3
    if i_has_b or j_has_b:
        partial += 0.3
    return round(min(partial, 1.0), 4)


def _weighted_score(signals: dict, weights: dict) -> float:
    """Compute weighted heuristic score in [0, 1]."""
    s = sum(weights[k] * signals[k] for k in weights)
    return round(max(0.0, min(1.0, s)), 6)


# ---------------------------------------------------------------------------
# Anchor entities + papers
# ---------------------------------------------------------------------------

def _anchor_entities_for_community(comm_nodes, G, conn, top_n=ANCHOR_ENTITIES_PER_SIDE):
    """Return top_n most-connected (in-graph degree) canonical entities."""
    scored = []
    for n in comm_nodes:
        deg = G.degree(n, weight="weight") if G.has_node(n) else 0
        scored.append((deg, n))
    scored.sort(reverse=True)
    result = []
    for deg, n in scored[:top_n]:
        result.append({
            "id": n,
            "name": _entity_name(conn, n),
            "degree": deg,
        })
    return result


def _anchor_papers_for_community(comm_nodes, conn) -> list:
    """Papers contributing claims whose entities are in this community."""
    set_nodes = set(comm_nodes)
    paper_slugs = set()
    for row in conn.execute(
        "SELECT subject_entity_id, object_entity_id, source_paper FROM claims"
    ):
        sc = _canonical_id(conn, row[0])
        oc = _canonical_id(conn, row[1])
        if sc in set_nodes or oc in set_nodes:
            paper_slugs.add(row[2])
    return sorted(paper_slugs)


# ---------------------------------------------------------------------------
# Passage retrieval via graph-retrieve.py (degrade gracefully)
# ---------------------------------------------------------------------------

def _retrieve_passages_for_paper(paper_slug: str, top_n: int = 1) -> list:
    """Retrieve up to top_n passage snippets from the indexed full text.

    Delegates to graph-retrieve.py as a subprocess to avoid import side effects.
    Degrades gracefully: if the index is absent, returns a note string.
    """
    retrieve_script = SCRIPT_DIR / "graph-retrieve.py"
    if not retrieve_script.exists():
        return [{"paper": paper_slug, "snippet": "(graph-retrieve.py not found)"}]

    import subprocess as _sp
    try:
        r = _sp.run(
            [sys.executable, str(retrieve_script), "--paper", paper_slug,
             "--top", str(top_n), "--no-rerank"],
            capture_output=True, text=True, timeout=15, cwd=str(VAULT_ROOT),
        )
        if r.returncode != 0:
            return [{"paper": paper_slug, "snippet": "(no full text indexed)"}]
        data = json.loads(r.stdout)
        snippets = []
        for c in data.get("candidates", [])[:top_n]:
            snippets.append({
                "paper": paper_slug,
                "snippet": c.get("snippet", ""),
                "page_path": c.get("page_path", ""),
                "chunk_index": c.get("chunk_index", 0),
            })
        return snippets if snippets else [{"paper": paper_slug, "snippet": "(no full text indexed)"}]
    except Exception:
        return [{"paper": paper_slug, "snippet": "(no full text indexed)"}]


def _gather_passages(anchor_papers: list, n_per_paper: int = 1) -> list:
    """Retrieve passages for each anchor paper, up to PASSAGES_PER_PROPOSAL total."""
    passages = []
    for slug in anchor_papers[:PASSAGES_PER_PROPOSAL]:
        passages.extend(_retrieve_passages_for_paper(slug, top_n=n_per_paper))
    return passages


# ---------------------------------------------------------------------------
# FR7: Already-proposed flag
# ---------------------------------------------------------------------------

def _check_already_proposed(comm_i_nodes, comm_j_nodes, conn) -> bool:
    """Return True if either community contains a known bridge entity by name."""
    set_all = set(comm_i_nodes) | set(comm_j_nodes)
    for n in set_all:
        name = _entity_name(conn, n).lower()
        if any(kw in name for kw in KNOWN_BRIDGE_ENTITY_NAMES):
            return True
    return False


# ---------------------------------------------------------------------------
# Community info dict
# ---------------------------------------------------------------------------

def _community_info(comm_nodes, G, comm_idx: int) -> dict:
    """Build the community_a/community_b dict for a proposal."""
    names = sorted([G.nodes[n].get("name", f"e{n}") for n in comm_nodes])
    super_types = list({G.nodes[n].get("super_type", "unknown") for n in comm_nodes})
    return {
        "id": comm_idx,
        "size": len(comm_nodes),
        "members": names[:MEMBERS_TRUNCATE],
        "super_types": sorted(super_types),
    }


# ---------------------------------------------------------------------------
# Main ranking engine
# ---------------------------------------------------------------------------

def build_proposals(conn, top_n: int = 10, weights: dict = None,
                    retrieve_passages: bool = True) -> list:
    """Build ranked bridge proposals from the graph db.

    Returns a list of Proposal dataclass instances.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    G, communities = _build_louvain_graph(conn)
    if communities is None or len(communities) < 2:
        return []

    # Build node-to-community map
    node_to_community = {}
    for i, comm in enumerate(communities):
        for n in comm:
            node_to_community[n] = i

    # Count cross-community claim edges
    cross_edges: dict = defaultdict(int)
    for row in conn.execute(
        "SELECT subject_entity_id, object_entity_id FROM claims"
    ):
        sc = _canonical_id(conn, row[0])
        oc = _canonical_id(conn, row[1])
        if sc in node_to_community and oc in node_to_community:
            ca = node_to_community[sc]
            cb = node_to_community[oc]
            if ca != cb:
                cross_edges[tuple(sorted([ca, cb]))] += 1

    # Enumerate zero-claim-edge pairs with sufficient size (BR1)
    white_space_pairs = []
    for i in range(len(communities)):
        for j in range(i + 1, len(communities)):
            comm_i = communities[i]
            comm_j = communities[j]
            if min(len(comm_i), len(comm_j)) < MIN_COMMUNITY_SIZE:
                continue
            if cross_edges.get((i, j), 0) > 0:
                continue
            white_space_pairs.append((i, j))

    if not white_space_pairs:
        return []

    proposals = []
    for (ci, cj) in white_space_pairs:
        comm_i = communities[ci]
        comm_j = communities[cj]

        # Compute signals
        gap_conf = _compute_gap_confidence(comm_i, comm_j, conn)
        bridgeab = _compute_bridgeability(comm_i, comm_j, conn)
        lim_pull = _compute_limitation_pull(comm_i, comm_j, conn)
        richness = _compute_richness(comm_i, comm_j, conn)
        dir_rel = _compute_direction_relevance(comm_i, comm_j, G)

        signals = {
            "gap_confidence": gap_conf,
            "bridgeability": bridgeab,
            "limitation_pull": lim_pull,
            "richness": richness,
            "direction_relevance": dir_rel,
        }
        score = _weighted_score(signals, weights)

        # Anchor entities and papers
        anchor_ents_a = _anchor_entities_for_community(comm_i, G, conn)
        anchor_ents_b = _anchor_entities_for_community(comm_j, G, conn)
        anchor_entities = anchor_ents_a + anchor_ents_b

        papers_a = _anchor_papers_for_community(comm_i, conn)
        papers_b = _anchor_papers_for_community(comm_j, conn)
        anchor_papers = sorted(set(papers_a + papers_b))

        # FR7: already_proposed
        already = _check_already_proposed(comm_i, comm_j, conn)

        proposal = Proposal(
            id=f"{ci}-{cj}",
            community_a=_community_info(comm_i, G, ci),
            community_b=_community_info(comm_j, G, cj),
            anchor_entities=anchor_entities,
            anchor_papers=anchor_papers,
            score=score,
            signal_breakdown=signals,
            passages=[],  # filled after ranking to avoid retrieving for all 300+ pairs
            already_proposed=already,
            justification=None,
        )
        proposals.append(proposal)

    # Sort: score DESC, id ASC (stable tie-break for determinism, BR2)
    proposals.sort(key=lambda p: (-p.score, p.id))

    # Keep top_n
    proposals = proposals[:top_n]

    # Retrieve passages only for top proposals
    if retrieve_passages:
        for p in proposals:
            p.passages = _gather_passages(p.anchor_papers[:2],
                                          n_per_paper=1)

    return proposals


# ---------------------------------------------------------------------------
# --synthesize: optional narrative via contextual-prefix tier selection
# ---------------------------------------------------------------------------

def _synthesize_justification(proposal: Proposal, allow_egress: bool = True) -> str:
    """Generate a narrative justification via the contextual-prefix egress tier.

    Uses contextual-prefix.pick_prefix_tier to honour the egress posture:
    - Without --synthesize: this function is never called.
    - With --synthesize: picks tier (anthropic-api / claude-cli / synthetic);
      if claude is absent, falls back to a structured template, exit 0 (AC5).
    """
    cp_path = SCRIPT_DIR / "contextual-prefix.py"
    if not cp_path.exists():
        return _structured_justification(proposal)

    try:
        spec = importlib.util.spec_from_file_location("contextual_prefix", cp_path)
        cp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cp)

        tier = cp.pick_prefix_tier(force_synthetic=False, allow_egress=allow_egress)

        if tier == "synthetic":
            return _structured_justification(proposal)

        comm_a_names = ", ".join(proposal.community_a.get("members", [])[:3])
        comm_b_names = ", ".join(proposal.community_b.get("members", [])[:3])
        anchor_names = ", ".join(e["name"] for e in proposal.anchor_entities[:4])

        chunk_text = (
            f"Bridge community A ({comm_a_names}) with community B ({comm_b_names}). "
            f"Key entities: {anchor_names}. "
            f"Signal scores: gap_confidence={proposal.signal_breakdown['gap_confidence']:.3f}, "
            f"direction_relevance={proposal.signal_breakdown['direction_relevance']:.3f}."
        )
        fm = {"title": "Research Gap Bridge Proposal"}
        body = chunk_text

        prefix, _ = cp.generate_prefix(tier, fm, body, chunk_text)
        return prefix if prefix else _structured_justification(proposal)
    except Exception:
        return _structured_justification(proposal)


def _structured_justification(proposal: Proposal) -> str:
    """Deterministic structured fallback when no LLM is available."""
    comm_a_names = ", ".join(proposal.community_a.get("members", [])[:3])
    comm_b_names = ", ".join(proposal.community_b.get("members", [])[:3])
    anchor_names = ", ".join(e["name"] for e in proposal.anchor_entities[:3])
    dr = proposal.signal_breakdown.get("direction_relevance", 0.0)
    bridge_motive = (
        "This bridge targets a priority research direction (virtual try-on + diffusion)."
        if dr > 0.5 else
        "This bridge connects two unexplored community clusters."
    )
    return (
        f"Connect '{comm_a_names}' (community {proposal.community_a['id']}) with "
        f"'{comm_b_names}' (community {proposal.community_b['id']}) via "
        f"anchor concepts: {anchor_names}. "
        f"{bridge_motive} "
        f"Bridgeability={proposal.signal_breakdown['bridgeability']:.3f}, "
        f"limitation_pull={proposal.signal_breakdown['limitation_pull']:.3f}."
    )


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_report(proposals: list) -> str:
    """Render proposals as a markdown report."""
    if not proposals:
        return (
            "# Bridge Proposals\n\n"
            "No bridge proposals found. All community pairs are already connected "
            "by claim-edges, or communities are too small (< 5 entities each).\n"
        )

    lines = [
        "# Bridge Proposals — Ranked White-Space Gap Bridges",
        "",
        f"**{len(proposals)} proposal(s)** ranked by weighted heuristic score.",
        "",
        "| # | Score | Proposal | Direction Rel. | Already Proposed |",
        "|---|-------|----------|----------------|-----------------|",
    ]
    for i, p in enumerate(proposals, 1):
        comm_a = p.community_a
        comm_b = p.community_b
        label = (
            f"Comm {comm_a['id']} ({comm_a['members'][0] if comm_a['members'] else '?'}) "
            f"↔ Comm {comm_b['id']} ({comm_b['members'][0] if comm_b['members'] else '?'})"
        )
        dr = p.signal_breakdown.get("direction_relevance", 0.0)
        already = "YES" if p.already_proposed else ""
        lines.append(
            f"| {i} | {p.score:.4f} | {label[:80]} | {dr:.2f} | {already} |"
        )

    lines.append("")
    lines.append("---")
    lines.append("")

    for i, p in enumerate(proposals, 1):
        comm_a = p.community_a
        comm_b = p.community_b
        lines.append(
            f"### {i}. [{p.id}] Community {comm_a['id']} ↔ Community {comm_b['id']}"
        )
        lines.append(f"**Score:** {p.score:.4f}")
        lines.append(f"**Already proposed:** {p.already_proposed}")
        lines.append("")
        lines.append(f"**Community A** (size {comm_a['size']}): "
                     f"{', '.join(comm_a['members'])} ...")
        lines.append(f"**Community B** (size {comm_b['size']}): "
                     f"{', '.join(comm_b['members'])} ...")
        lines.append("")
        lines.append("**Signal breakdown:**")
        for k, v in p.signal_breakdown.items():
            lines.append(f"  - {k}: {v:.4f}")
        lines.append("")
        if p.anchor_entities:
            lines.append("**Anchor entities:**")
            for ae in p.anchor_entities:
                lines.append(f"  - {ae['name']} (degree {ae['degree']})")
            lines.append("")
        if p.anchor_papers:
            lines.append("**Anchor papers:** " + ", ".join(p.anchor_papers[:5]))
            lines.append("")
        if p.passages:
            lines.append("**Passages:**")
            for pas in p.passages:
                snippet = pas.get("snippet", "")[:200]
                lines.append(f"  - [{pas.get('paper', '?')}] {snippet}")
            lines.append("")
        if p.justification:
            lines.append(f"**Justification:** {p.justification}")
            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def proposal_to_dict(p: Proposal) -> dict:
    return {
        "id": p.id,
        "community_a": p.community_a,
        "community_b": p.community_b,
        "anchor_entities": p.anchor_entities,
        "anchor_papers": p.anchor_papers,
        "score": p.score,
        "signal_breakdown": p.signal_breakdown,
        "passages": p.passages,
        "already_proposed": p.already_proposed,
        "justification": p.justification,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="graph-bridge.py — Proposal Finder for white-space gaps"
    )
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON")
    parser.add_argument("--top", type=int, default=10,
                        help="Max proposals to return (default: 10)")
    parser.add_argument("--db", type=str, default=str(DEFAULT_DB),
                        help="Path to graph.db (default: derived db)")
    parser.add_argument("--synthesize", action="store_true",
                        help="Generate narrative justifications (opt-in egress; "
                             "falls back to structured text if claude absent)")
    # Weight overrides (FR2: CLI-tunable weights)
    parser.add_argument("--w-gap-confidence", type=float, default=None,
                        metavar="W",
                        help="Weight for gap_confidence signal (default: 0.25)")
    parser.add_argument("--w-bridgeability", type=float, default=None,
                        metavar="W",
                        help="Weight for bridgeability signal (default: 0.20)")
    parser.add_argument("--w-limitation-pull", type=float, default=None,
                        metavar="W",
                        help="Weight for limitation_pull signal (default: 0.15)")
    parser.add_argument("--w-richness", type=float, default=None,
                        metavar="W",
                        help="Weight for richness signal (default: 0.15)")
    parser.add_argument("--w-direction-relevance", type=float, default=None,
                        metavar="W",
                        help="Weight for direction_relevance signal (default: 0.25)")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Error: database not found at {db_path}", file=sys.stderr)
        print("Run: uv run python scripts/graph-build.py wiki/graph "
              ".vault-meta/graph/graph.db", file=sys.stderr)
        return 1

    # Build weights
    weights = dict(DEFAULT_WEIGHTS)
    if args.w_gap_confidence is not None:
        weights["gap_confidence"] = args.w_gap_confidence
    if args.w_bridgeability is not None:
        weights["bridgeability"] = args.w_bridgeability
    if args.w_limitation_pull is not None:
        weights["limitation_pull"] = args.w_limitation_pull
    if args.w_richness is not None:
        weights["richness"] = args.w_richness
    if args.w_direction_relevance is not None:
        weights["direction_relevance"] = args.w_direction_relevance

    conn = connect(db_path)

    # Verify schema
    has_claims = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='claims'"
    ).fetchone()
    if not has_claims:
        print("Error: database missing claims table.", file=sys.stderr)
        conn.close()
        return 1

    proposals = build_proposals(
        conn, top_n=args.top, weights=weights,
        retrieve_passages=True,
    )

    # Synthesise justifications if requested (AC4: egress opt-in only)
    if args.synthesize and proposals:
        for p in proposals:
            p.justification = _synthesize_justification(p, allow_egress=True)

    conn.close()

    if args.json:
        print(json.dumps([proposal_to_dict(p) for p in proposals],
                         indent=2, ensure_ascii=False))
    else:
        print(format_report(proposals))

    return 0


if __name__ == "__main__":
    sys.exit(main())
