"""Entity resolution — detect duplicate entities via exact matching and semantic similarity.

Two-tier strategy:
  Tier 1 — EXACT MATCH (deterministic): same lower(name) + same super_type → confidence 1.0
  Tier 2 — FUZZY MATCH (human review required):
    A. Embedding cosine via ollama + nomic-embed-text (threshold 0.85), or
    B. Token Jaccard similarity fallback (threshold 0.6)

Reads the derived db. Outputs ranked proposals. NEVER auto-modifies the vault.

Usage:
    uv run python scripts/graph-resolve.py
    uv run python scripts/graph-resolve.py --json --tier 2 --top 20
"""

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from graph_db import connect, root  # noqa: E402

DEFAULT_DB = Path(__file__).resolve().parent.parent / ".vault-meta" / "graph" / "graph.db"

# Thresholds
EMBED_THRESHOLD = 0.85   # cosine — high bar, same concept
JACCARD_THRESHOLD = 0.6  # token overlap — lower bar, more false positives OK

# ollama
OLLAMA_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _canonical_id(conn, entity_id: int) -> int:
    try:
        return root(conn, entity_id, compress=False)
    except Exception:
        return entity_id


def _load_canonical_entities(conn):
    """Return all canonical entities with their aliases."""
    rows = conn.execute("""
        SELECT id, name, super_type, sub_type, description
        FROM entities WHERE canonical_id IS NULL
        ORDER BY LOWER(name), super_type
    """).fetchall()

    entities = []
    for row in rows:
        eid, name, stype, ssub, desc = row
        entities.append({
            "id": eid, "name": name, "super_type": stype,
            "sub_type": ssub or "", "description": desc or "",
        })
    return entities


# ---------------------------------------------------------------------------
# Tier 1 — exact-name match
# ---------------------------------------------------------------------------

def _tier1_exact(entities: list[dict]) -> list[dict]:
    """Find exact-name duplicates within same super_type."""
    by_key = {}
    for e in entities:
        key = (e["name"].lower().strip(), e["super_type"])
        by_key.setdefault(key, []).append(e)

    proposals = []
    for (name_lower, stype), group in by_key.items():
        if len(group) < 2:
            continue
        # Pair first with each subsequent
        canonical = group[0]
        for loser in group[1:]:
            proposals.append({
                "tier": 1,
                "entity_a": {
                    "id": canonical["id"], "name": canonical["name"],
                    "super_type": canonical["super_type"],
                    "sub_type": canonical["sub_type"],
                },
                "entity_b": {
                    "id": loser["id"], "name": loser["name"],
                    "super_type": loser["super_type"],
                    "sub_type": loser["sub_type"],
                },
                "confidence": 1.0,
                "method": "exact",
                "reason": f"Exact name match within {stype}: "
                          f"'{canonical['name']}' == '{loser['name']}'",
            })
    return sorted(proposals, key=lambda p: p["entity_a"]["name"])


# ---------------------------------------------------------------------------
# Tier 2A — embedding cosine (ollama)
# ---------------------------------------------------------------------------

def _ollama_alive() -> bool:
    try:
        urllib.request.urlopen(
            urllib.request.Request("http://localhost:11434/api/tags"),
            timeout=2,
        )
        return True
    except Exception:
        return False


def _embed(text: str) -> Optional[list[float]]:
    """Get embedding vector from ollama. Returns None on failure."""
    try:
        payload = json.dumps({"model": EMBED_MODEL, "prompt": text}).encode()
        req = urllib.request.Request(OLLAMA_URL, data=payload)
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read()).get("embedding")
    except Exception:
        return None


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Tier 2B — token Jaccard fallback
# ---------------------------------------------------------------------------

def _jaccard(a: str, b: str) -> float:
    """Jaccard similarity on word tokens (case-insensitive, deduped, split on non-alnum)."""
    import re
    tokens_a = set(re.split(r"[^a-z0-9]+", a.lower().strip()))
    tokens_b = set(re.split(r"[^a-z0-9]+", b.lower().strip()))
    tokens_a.discard("")
    tokens_b.discard("")
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


# ---------------------------------------------------------------------------
# Tier 2 fuzzy scan
# ---------------------------------------------------------------------------

def _tier2_fuzzy(entities: list[dict]) -> list[dict]:
    """Find fuzzy duplicate candidates using embedding or Jaccard fallback."""
    use_embedding = _ollama_alive()
    method = "embedding" if use_embedding else "jaccard"
    threshold = EMBED_THRESHOLD if use_embedding else JACCARD_THRESHOLD

    proposals = []

    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            a, b = entities[i], entities[j]

            # Skip same-entity (root() resolves to same id)
            if a["id"] == b["id"]:
                continue

            if use_embedding:
                # Embed the combined name + description for semantic comparison
                text_a = f"{a['name']}: {a['description']}"[:512]
                text_b = f"{b['name']}: {b['description']}"[:512]
                vec_a = _embed(text_a)
                vec_b = _embed(text_b)
                if vec_a is None or vec_b is None:
                    continue  # skip this pair if embedding fails
                score = _cosine(vec_a, vec_b)
                if score > threshold:
                    proposals.append({
                        "tier": 2,
                        "entity_a": {
                            "id": a["id"], "name": a["name"],
                            "super_type": a["super_type"],
                            "sub_type": a["sub_type"],
                        },
                        "entity_b": {
                            "id": b["id"], "name": b["name"],
                            "super_type": b["super_type"],
                            "sub_type": b["sub_type"],
                        },
                        "confidence": round(score, 3),
                        "method": "embedding",
                        "reason": f"Embedding cosine={score:.3f} between "
                                  f"'{a['name']}' and '{b['name']}'",
                    })
            else:
                # Jaccard on the combined name+sub_type text
                text_a = f"{a['name']} {a['sub_type']}"
                text_b = f"{b['name']} {b['sub_type']}"
                score = _jaccard(text_a, text_b)
                if score > threshold:
                    proposals.append({
                        "tier": 2,
                        "entity_a": {
                            "id": a["id"], "name": a["name"],
                            "super_type": a["super_type"],
                            "sub_type": a["sub_type"],
                        },
                        "entity_b": {
                            "id": b["id"], "name": b["name"],
                            "super_type": b["super_type"],
                            "sub_type": b["sub_type"],
                        },
                        "confidence": round(score, 3),
                        "method": "jaccard",
                        "reason": f"Token Jaccard={score:.3f} between "
                                  f"'{a['name']}' and '{b['name']}'",
                    })

    proposals.sort(key=lambda p: p["confidence"], reverse=True)
    return proposals


# ---------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------

def resolve(conn, tier: str = "all", top_n: int = 50) -> list[dict]:
    entities = _load_canonical_entities(conn)
    proposals = []

    if tier in ("1", "all"):
        proposals.extend(_tier1_exact(entities))

    if tier in ("2", "all"):
        proposals.extend(_tier2_fuzzy(entities))

    # Tier 1 first, then Tier 2, both sorted by confidence
    proposals.sort(key=lambda p: (-p["tier"], -p["confidence"]))
    return proposals[:top_n]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Entity resolution — detect duplicates")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--tier", type=str, default="all",
                        choices=["1", "2", "all"], help="Which tier to run")
    parser.add_argument("--top", type=int, default=50, help="Max proposals")
    parser.add_argument("--db", type=str, default=str(DEFAULT_DB),
                        help="Path to graph.db")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Error: database not found at {db_path}", file=sys.stderr)
        print("Run: uv run python scripts/graph-build.py wiki/graph/ .vault-meta/graph/graph.db",
              file=sys.stderr)
        sys.exit(1)

    conn = connect(db_path)
    has_entities = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='entities'"
    ).fetchone()
    if not has_entities:
        print("Error: database missing entities table.", file=sys.stderr)
        conn.close()
        sys.exit(1)

    proposals = resolve(conn, tier=args.tier, top_n=args.top)

    if args.json:
        print(json.dumps(proposals, indent=2))
    else:
        if not proposals:
            print("No duplicate entity proposals found.")
        else:
            tier1 = [p for p in proposals if p["tier"] == 1]
            tier2 = [p for p in proposals if p["tier"] == 2]
            print(f"# Entity Resolution Report")
            print(f"**{len(proposals)} proposals** "
                  f"({len(tier1)} Tier 1 exact, {len(tier2)} Tier 2 fuzzy)")
            if tier1:
                print(f"\n## Tier 1 — Exact Matches")
                for i, p in enumerate(tier1, 1):
                    print(f"{i}. [{p['confidence']:.2f}] {p['entity_a']['name']} "
                          f"← {p['entity_b']['name']} ({p['reason']})")
            if tier2:
                print(f"\n## Tier 2 — Fuzzy Matches (review required)")
                for i, p in enumerate(tier2, 1):
                    print(f"{i}. [{p['confidence']:.3f}] {p['entity_a']['name']} "
                          f"↔ {p['entity_b']['name']} [{p['method']}]")
                    print(f"   {p['reason']}")

    conn.close()


if __name__ == "__main__":
    main()
