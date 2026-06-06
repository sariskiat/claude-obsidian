#!/usr/bin/env python3
"""graph-retrieve.py — /graph read backend for full-paper retrieval (P4).

Answers three query forms:
  graph-retrieve.py "natural language query"         # free-text BM25 over all indexed papers
  graph-retrieve.py --paper <slug>                   # all top chunks from one paper's full text
  graph-retrieve.py --claim <id>                     # trace claim -> source paper -> top chunks

Pipeline:
  query -> BM25 (graph index) -> rerank (ollama/noop) -> ranked candidates with provenance

Reuses bm25-index.query and rerank.rerank by importing and pointing their module-level
INDEX_PATH / CHUNKS_DIR globals at the graph-scoped paths under .vault-meta/graph/.

Options:
  --bm25-index PATH    BM25 index file (default: .vault-meta/graph/bm25/index.json)
  --chunks-dir PATH    Chunks store dir (default: .vault-meta/graph/chunks/)
  --export PATH        graph-export.json (needed for --claim lookups;
                       default: wiki/graph/graph-export.json)
  --top N              Number of results (default: 5)
  --bm25-top N         BM25 candidate count pre-rerank (default: 20)
  --no-rerank          Skip rerank; return BM25 order

Output (JSON to stdout):
{
  "query": "...",
  "strategy": "bm25+rerank:..." | "bm25+noop-rerank",
  "top_k": 5,
  "candidates": [
    {
      "chunk_id": "gph-abc123:0",
      "page_address": "gph-abc123",
      "page_path": "wiki/graph/papers/<slug>.full.md",
      "chunk_index": 0,
      "bm25_score": 7.12,
      "rerank_score": 0.81,
      "rerank_source": "cosine:nomic-embed-text" | "noop-no-ollama",
      "snippet": "... first 200 chars ..."
    },
    ...
  ]
}

Exit codes:
  0  — success
  2  — usage error (empty query, unknown flag)
  10 — graph index not provisioned (run graph-fulltext.py sync first)
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

VAULT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = VAULT_ROOT / "scripts"
META_DIR = VAULT_ROOT / ".vault-meta"

DEFAULT_BM25_INDEX = META_DIR / "graph" / "bm25" / "index.json"
DEFAULT_CHUNKS_DIR = META_DIR / "graph" / "chunks"
DEFAULT_EXPORT = VAULT_ROOT / "wiki" / "graph" / "graph-export.json"

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_NOT_PROVISIONED = 10


def log(msg):
    print(msg, file=sys.stderr)


def import_sibling(name, filename):
    """Import a hyphenated sibling .py file as a Python module."""
    target = SCRIPTS_DIR / filename
    if not target.is_file():
        log(f"ERR: sibling helper {filename} not found at {target}")
        sys.exit(EXIT_NOT_PROVISIONED)
    try:
        spec = importlib.util.spec_from_file_location(name, target)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except (ImportError, SyntaxError, AttributeError) as e:
        log(f"ERR: failed to import {filename}: {type(e).__name__}: {e}")
        sys.exit(EXIT_NOT_PROVISIONED)


def chunk_snippet(chunk_data, max_chars=200):
    text = chunk_data.get("raw_text", "")
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"


def load_export(export_path: Path):
    """Load graph-export.json; return None on missing or corrupt."""
    if not export_path.is_file():
        return None
    try:
        return json.loads(export_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def resolve_claim_to_slug(claim_id: int, export_data: dict) -> str | None:
    """Look up a claim by numeric id in the export data; return its source_paper slug."""
    for c in export_data.get("claims", []):
        if c.get("id") == claim_id:
            return c.get("source_paper")
    return None


def paper_slug_to_query(slug: str) -> str:
    """Derive a free-text query from a slug (for --paper mode: all chunks from that paper).

    We can't literally filter BM25 by slug, so we use the slug itself as a
    discriminating query token. The page_path of the paper's chunks ends with
    '<slug>.full.md', so we filter the BM25 results to that page_path afterward.
    """
    # Replace hyphens with spaces so tokenization works
    return slug.replace("-", " ").replace("_", " ")


def bm25_query_graph(bm25_mod, query_text: str, top_k: int, bm25_index_path: Path):
    """Call bm25-index.query() with the graph index path patched in."""
    # Repoint module-level INDEX_PATH to the graph-scoped index
    bm25_mod.INDEX_PATH = bm25_index_path
    return bm25_mod.query(query_text, top_k=top_k)


def load_chunk_file(vault_root: Path, rel_path: str, chunks_dir: Path):
    """Load a chunk JSON file.

    rel_path is relative to vault_root (from the BM25 index docs dict).
    However, when we monkeypatched CHUNKS_DIR in bm25-index, the rel_path
    may be relative to chunks_dir.parent.parent (tmp sandbox).
    We try both locations.
    """
    # Try absolute via vault_root (production)
    p1 = vault_root / rel_path
    if p1.is_file():
        try:
            return json.loads(p1.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    # Try chunks_dir parent.parent (sandboxed test)
    p2 = chunks_dir.parent.parent / rel_path
    if p2.is_file():
        try:
            return json.loads(p2.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return None


def build_candidates(bm25_hits, vault_root: Path, chunks_dir: Path):
    """Hydrate BM25 hits into full candidate dicts with provenance."""
    candidates = []
    for h in bm25_hits:
        chunk = load_chunk_file(vault_root, h["path"], chunks_dir)
        if not chunk:
            continue
        candidates.append({
            "chunk_id": h["chunk_id"],
            "page_address": chunk.get("page_address"),
            "page_path": chunk.get("page_path"),
            "chunk_index": chunk.get("chunk_index"),
            "bm25_score": h["score"],
            "path": h["path"],
            "snippet": chunk_snippet(chunk),
        })
    return candidates


def main():
    parser = argparse.ArgumentParser(
        description="graph-retrieve.py — /graph read backend (full-paper BM25+rerank)"
    )
    # Mutually exclusive query modes
    query_group = parser.add_mutually_exclusive_group()
    query_group.add_argument("query", nargs="?", default=None,
                             help="Natural-language query")
    query_group.add_argument("--paper", metavar="SLUG",
                             help="Return top chunks from this paper's full text")
    query_group.add_argument("--claim", metavar="ID", type=int,
                             help="Trace claim -> source paper -> top chunks")

    parser.add_argument("--bm25-index", metavar="PATH",
                        help="BM25 index file (default: .vault-meta/graph/bm25/index.json)")
    parser.add_argument("--chunks-dir", metavar="PATH",
                        help="Chunks store dir (default: .vault-meta/graph/chunks/)")
    parser.add_argument("--export", metavar="PATH",
                        help="graph-export.json (default: wiki/graph/graph-export.json)")
    parser.add_argument("--top", type=int, default=5,
                        help="Final result count (post-rerank)")
    parser.add_argument("--bm25-top", type=int, default=20,
                        help="BM25 candidate count (pre-rerank)")
    parser.add_argument("--no-rerank", action="store_true",
                        help="Skip rerank; return BM25 order")

    args = parser.parse_args()

    # Resolve paths
    bm25_index_path = Path(args.bm25_index) if args.bm25_index else DEFAULT_BM25_INDEX
    chunks_dir = Path(args.chunks_dir) if args.chunks_dir else DEFAULT_CHUNKS_DIR
    export_path = Path(args.export) if args.export else DEFAULT_EXPORT

    # Check index provisioned
    if not bm25_index_path.is_file():
        log(f"ERR: graph BM25 index not found at {bm25_index_path}")
        log("  Run `uv run python scripts/graph-fulltext.py sync` first to build the index.")
        return EXIT_NOT_PROVISIONED

    if not chunks_dir.is_dir():
        log(f"ERR: graph chunks directory not found at {chunks_dir}")
        log("  Run `uv run python scripts/graph-fulltext.py sync` first to build the index.")
        return EXIT_NOT_PROVISIONED

    # Determine query text
    effective_query = None
    paper_slug_filter = None

    if args.paper:
        paper_slug_filter = args.paper
        effective_query = paper_slug_to_query(args.paper)
    elif args.claim is not None:
        export_data = load_export(export_path)
        if export_data is None:
            log(f"ERR: graph-export.json not found at {export_path} (needed for --claim lookup)")
            return EXIT_USAGE
        slug = resolve_claim_to_slug(args.claim, export_data)
        if slug is None:
            log(f"ERR: claim id {args.claim} not found in {export_path}")
            return EXIT_USAGE
        paper_slug_filter = slug
        effective_query = paper_slug_to_query(slug)
    elif args.query:
        effective_query = args.query
    else:
        log("ERR: provide a query, --paper <slug>, or --claim <id>")
        return EXIT_USAGE

    if not effective_query.strip():
        log("ERR: empty query")
        return EXIT_USAGE

    # Import BM25 module and point it at graph index
    bm25 = import_sibling("bm25_index", "bm25-index.py")
    # Monkeypatch module-level globals to graph-scoped paths
    bm25.INDEX_PATH = bm25_index_path
    bm25.CHUNKS_DIR = chunks_dir

    # BM25 query
    try:
        bm25_hits = bm25_query_graph(bm25, effective_query, top_k=args.bm25_top,
                                     bm25_index_path=bm25_index_path)
    except SystemExit as e:
        log(f"ERR: BM25 query failed (index missing or corrupt). "
            f"Run graph-fulltext.py sync to rebuild.")
        return EXIT_NOT_PROVISIONED

    log(f"bm25: {len(bm25_hits)} hits")

    # Hydrate candidates
    candidates = build_candidates(bm25_hits, VAULT_ROOT, chunks_dir)

    # Filter by paper slug if --paper or --claim mode
    if paper_slug_filter:
        slug_suffix = f"{paper_slug_filter}.full.md"
        candidates = [c for c in candidates
                      if (c.get("page_path") or "").endswith(slug_suffix)]

    # Rerank
    if args.no_rerank or not candidates:
        final = candidates[:args.top]
        strategy = "bm25-only"
        for c in final:
            c["rerank_score"] = c.get("bm25_score", 0.0)
            c["rerank_source"] = "skipped"
    else:
        reranker = import_sibling("rerank", "rerank.py")
        try:
            final = reranker.rerank(
                effective_query, candidates, top_k=args.top,
                allow_remote=False,
            )
        except Exception as e:
            log(f"rerank failed ({e}); falling back to BM25 order")
            final = candidates[:args.top]
            for c in final:
                c["rerank_score"] = c.get("bm25_score", 0.0)
                c["rerank_source"] = "noop-rerank-error"

        first_src = (final[0].get("rerank_source") if final else "unknown")
        strategy = f"bm25+rerank:{first_src}"

    # Derive query label for output
    if args.paper:
        query_label = f"--paper {args.paper}"
    elif args.claim is not None:
        query_label = f"--claim {args.claim}"
    else:
        query_label = effective_query

    out = {
        "query": query_label,
        "strategy": strategy,
        "top_k": args.top,
        "candidates": final[:args.top],
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
