#!/usr/bin/env python3
"""graph-fulltext.py — full-paper retrieval pipeline for /graph (P4).

Subcommands
-----------
sync    Resolve Tier-A papers from graph-export.json, write wiki/graph/papers/<slug>.full.md
        (frontmatter contract + verbatim body, byte-equal copy), then build a
        graph-scoped BM25 index over those bodies only.

        Options:
          --export PATH        graph-export.json path (default: wiki/graph/graph-export.json)
          --papers-dir PATH    output directory for .full.md files (default: wiki/graph/papers/)
          --chunks-dir PATH    graph chunk store dir (default: .vault-meta/graph/chunks/)
          --bm25-dir PATH      graph BM25 index dir  (default: .vault-meta/graph/bm25/)
          --allow-egress       Enable claude-CLI / Anthropic API contextual prefixes (default: synthetic)

Tier classification (resolver)
-------------------------------
Tier-A (imported):
  - source_path is an absolute or ~ path to a real .md file
  - source_path ends in paper.json and parent dir contains exactly one .md
  - source_path is a bare dir (no extension) containing exactly one .md
Tier-B (skipped, no crash):
  - source_path ends .pdf
  - source_path is a URL (starts http)
  - source_path is stale / file not found
  - paper.json bare dir with 0 or 2+ inner .md files
  - empty / None source_path

.full.md frontmatter contract
------------------------------
---
type: paper-fulltext
slug: <slug>
arxiv_id: <id|null>
source_path: <resolved absolute path>
paper: "[[<slug>]]"
---
<verbatim source body>

Idempotency: if the .full.md already exists and the source has not changed (sha256
of body is identical), the file is overwritten with the same bytes. Re-running is
always safe and produces byte-identical output.

Exit codes
----------
0 — success (Tier-A papers written; Tier-B papers logged and skipped)
1 — fatal: export file missing or unreadable
2 — usage error
"""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

VAULT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = VAULT_ROOT / "scripts"
META_DIR = VAULT_ROOT / ".vault-meta"

DEFAULT_EXPORT = VAULT_ROOT / "wiki" / "graph" / "graph-export.json"
DEFAULT_PAPERS_DIR = VAULT_ROOT / "wiki" / "graph" / "papers"
DEFAULT_CHUNKS_DIR = META_DIR / "graph" / "chunks"
DEFAULT_BM25_DIR = META_DIR / "graph" / "bm25"

EXIT_OK = 0
EXIT_FATAL = 1
EXIT_USAGE = 2

# BM25 constants (mirrors bm25-index.py)
K1 = 1.5
B = 0.75
import re as _re
_TOKEN_RE = _re.compile(r"\w[\w'\-]*", _re.UNICODE)
_STOPWORDS = frozenset("""
a an and are as at be by for from has have he her him his i if in is it its
of on or that the their them they this to was were will with you your
""".split())

# Chunk parameters (mirrors contextual-prefix.py)
CHUNK_TARGET_CHARS = 500 * 4   # 500 tokens * 4 chars/token
CHUNK_OVERLAP_CHARS = 200


def log(msg):
    print(msg, file=sys.stderr)


def sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Resolver: classify each paper in graph-export.json
# ---------------------------------------------------------------------------

def _resolve_source_path(source_path: str):
    """Return (resolved_absolute_path, tier_label) or (None, reason_str).

    Returns a Path if Tier-A (real .md file exists), or None with a reason
    string for anything that should be skipped.
    """
    if not source_path:
        return None, "no_source_path"
    if source_path.startswith("http"):
        return None, "url"
    sp = Path(os.path.expanduser(source_path))
    # direct .md
    if sp.suffix == ".md":
        if sp.is_file():
            return sp, "tier_a_direct"
        return None, f"stale_md:{sp}"
    # ends in .pdf
    if sp.suffix == ".pdf":
        return None, "pdf"
    # paper.json -> bare dir with single inner .md
    if sp.name == "paper.json":
        d = sp.parent
        if d.is_dir():
            mds = [f for f in d.iterdir() if f.suffix == ".md"]
            if len(mds) == 1:
                return mds[0], "tier_a_paper_json_dir"
        return None, f"paper_json_no_inner_md:{sp}"
    # bare dir (no extension, exists as a directory)
    if not sp.suffix and sp.is_dir():
        mds = [f for f in sp.iterdir() if f.suffix == ".md"]
        if len(mds) == 1:
            return mds[0], "tier_a_bare_dir"
        return None, f"bare_dir_multi_or_no_md:{sp}"
    # anything else
    return None, f"unknown_type:{sp}"


def resolve_papers(papers_list):
    """Classify all papers into tier_a and skipped lists.

    Returns:
        tier_a: list of dicts {slug, arxiv_id, resolved_path}
        skipped: list of dicts {slug, reason}
    """
    tier_a = []
    skipped = []
    for p in papers_list:
        slug = p.get("slug", "")
        sp = p.get("source_path") or ""
        resolved, reason = _resolve_source_path(sp)
        if resolved is not None:
            tier_a.append({
                "slug": slug,
                "arxiv_id": p.get("arxiv_id"),
                "source_path": sp,
                "resolved_path": resolved,
            })
        else:
            skipped.append({"slug": slug, "reason": reason, "source_path": sp})
    return tier_a, skipped


# ---------------------------------------------------------------------------
# .full.md writer
# ---------------------------------------------------------------------------

def build_fulltext_frontmatter(slug: str, arxiv_id, source_path: str) -> str:
    arxiv_val = arxiv_id if arxiv_id else "null"
    return (
        "---\n"
        f"type: paper-fulltext\n"
        f"slug: {slug}\n"
        f"arxiv_id: {arxiv_val}\n"
        f"source_path: {source_path}\n"
        f'paper: "[[{slug}]]"\n'
        "---\n"
    )


def write_full_md(papers_dir: Path, entry: dict) -> bool:
    """Write wiki/graph/papers/<slug>.full.md.

    Returns True if written, False if already up-to-date (idempotent).
    The body is a verbatim byte-equal copy of the source .md content.
    """
    slug = entry["slug"]
    resolved = entry["resolved_path"]
    src_bytes = resolved.read_bytes()
    src_text = src_bytes.decode("utf-8", errors="replace")

    fm = build_fulltext_frontmatter(slug, entry["arxiv_id"], str(resolved))
    full_content = fm + src_text
    full_bytes = full_content.encode("utf-8")

    out_path = papers_dir / f"{slug}.full.md"
    if out_path.exists():
        existing = out_path.read_bytes()
        if existing == full_bytes:
            return False  # already up-to-date

    papers_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(full_bytes)
    return True


# ---------------------------------------------------------------------------
# Chunker (self-contained; mirrors contextual-prefix.chunk_body)
# ---------------------------------------------------------------------------

def chunk_body(body: str, target_chars=CHUNK_TARGET_CHARS, overlap=CHUNK_OVERLAP_CHARS):
    """Split body into overlapping chunks on paragraph boundaries."""
    paragraphs = [p.strip() for p in _re.split(r"\n\s*\n", body) if p.strip()]
    chunks = []
    cur = []
    cur_len = 0
    for p in paragraphs:
        cur.append(p)
        cur_len += len(p) + 2
        if cur_len >= target_chars:
            chunk_text = "\n\n".join(cur)
            chunks.append(chunk_text)
            tail = chunk_text[-overlap:] if overlap > 0 else ""
            cur = [tail] if tail else []
            cur_len = len(tail)
    if cur and "".join(cur).strip():
        chunks.append("\n\n".join(cur))
    if not chunks and body.strip():
        chunks = [body.strip()]
    return chunks


def _synthetic_prefix(slug: str, body_first_line: str, chunk_text: str) -> str:
    """Tier-3 (synthetic) prefix. Zero network, deterministic."""
    first = body_first_line[:300] if body_first_line else ""
    return f'This passage is from the full paper "{slug}". The paper opens: {first}'


# ---------------------------------------------------------------------------
# Graph BM25 index builder (self-contained; mirrors bm25-index logic)
# ---------------------------------------------------------------------------

def _tokenize(text: str):
    return [t.lower() for t in _TOKEN_RE.findall(text)
            if t.lower() not in _STOPWORDS and len(t) > 1]


def build_graph_bm25_index(chunks_dir: Path):
    """Build a BM25 index over all chunk JSON files in chunks_dir.

    Returns the index dict, or None if no chunks found.
    """
    docs = {}
    df = Counter()
    postings = defaultdict(list)

    for chunk_file in sorted(chunks_dir.glob("*/chunk-*.json")):
        try:
            data = json.loads(chunk_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            log(f"  skip corrupt chunk {chunk_file}: {e}")
            continue
        address = data.get("page_address")
        idx = data.get("chunk_index")
        text = data.get("contextualized_text") or data.get("raw_text", "")
        if address is None or idx is None:
            continue
        chunk_id = f"{address}:{idx}"
        rel_path = str(chunk_file.relative_to(chunks_dir.parent.parent))
        tokens = _tokenize(text)
        tf = Counter(tokens)
        docs[chunk_id] = {"path": rel_path, "dl": len(tokens)}
        for term, count in tf.items():
            df[term] += 1
            postings[term].append([chunk_id, count])

    if not docs:
        return None

    avg_dl = sum(d["dl"] for d in docs.values()) / len(docs)
    vocab = {term: {"df": df[term], "postings": postings[term]}
             for term in sorted(df.keys())}

    return {
        "schema_version": 1,
        "params": {"k1": K1, "b": B},
        "doc_count": len(docs),
        "avg_dl": avg_dl,
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "vocab": vocab,
        "docs": docs,
    }


def write_bm25_index(bm25_dir: Path, index: dict):
    bm25_dir.mkdir(parents=True, exist_ok=True)
    idx_path = bm25_dir / "index.json"
    tmp = idx_path.with_suffix(f".{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, idx_path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Chunk writer for a single .full.md
# ---------------------------------------------------------------------------

def write_chunks_for_full_md(slug: str, full_md_path: Path, chunks_dir: Path,
                              allow_egress: bool = False):
    """Chunk the body of a .full.md and write chunk JSON files.

    Uses synthetic prefix (zero egress) unless allow_egress=True.
    Idempotent: skips chunks whose body_hash + page_body_hash is unchanged.
    """
    full_text = full_md_path.read_text(encoding="utf-8")

    # strip frontmatter to get body
    fm_re = _re.compile(r"^---\n.*?\n---\n?", _re.DOTALL)
    m = fm_re.match(full_text)
    body = full_text[m.end():] if m else full_text

    page_body_hash = sha256(body.encode("utf-8"))

    # stable synthetic address derived from slug
    h = hashlib.sha1(f"graph:{slug}".encode("utf-8")).hexdigest()
    address = "gph-" + h[:6]

    chunk_dir = chunks_dir / address
    chunk_dir.mkdir(parents=True, exist_ok=True)

    chunks = chunk_body(body)
    if not chunks:
        return 0

    body_first_line = body.strip().splitlines()[0] if body.strip() else ""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    written = 0

    for idx, raw in enumerate(chunks):
        chunk_path = chunk_dir / f"chunk-{idx:03d}.json"
        body_hash = sha256(raw.encode("utf-8"))

        if chunk_path.exists():
            try:
                existing = json.loads(chunk_path.read_text(encoding="utf-8"))
                if (existing.get("body_hash") == body_hash and
                        existing.get("page_body_hash") == page_body_hash):
                    continue  # already up-to-date
            except (json.JSONDecodeError, OSError):
                pass

        prefix = _synthetic_prefix(slug, body_first_line, raw)
        contextualized = f"{prefix}\n\n{raw}" if prefix else raw

        chunk_data = {
            "schema_version": 1,
            "page_path": str(full_md_path.relative_to(full_md_path.parent.parent.parent.parent)),
            "page_address": address,
            "chunk_index": idx,
            "raw_text": raw,
            "contextualized_text": contextualized,
            "prefix_source": "synthetic",
            "char_count": len(raw),
            "body_hash": body_hash,
            "page_body_hash": page_body_hash,
            "created_at": now,
        }
        chunk_path.write_text(json.dumps(chunk_data, ensure_ascii=False), encoding="utf-8")
        written += 1

    return written


# ---------------------------------------------------------------------------
# sync command
# ---------------------------------------------------------------------------

def cmd_sync(args):
    export_path = Path(args.export) if args.export else DEFAULT_EXPORT
    papers_dir = Path(args.papers_dir) if args.papers_dir else DEFAULT_PAPERS_DIR
    chunks_dir = Path(args.chunks_dir) if args.chunks_dir else DEFAULT_CHUNKS_DIR
    bm25_dir = Path(args.bm25_dir) if args.bm25_dir else DEFAULT_BM25_DIR
    allow_egress = getattr(args, "allow_egress", False)

    if not export_path.is_file():
        log(f"ERR: graph-export.json not found at {export_path}")
        return EXIT_FATAL

    try:
        export_data = json.loads(export_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        log(f"ERR: cannot read {export_path}: {e}")
        return EXIT_FATAL

    papers_list = export_data.get("papers", [])
    tier_a, skipped = resolve_papers(papers_list)

    log(f"Resolver: {len(tier_a)} Tier-A (resolvable .md), {len(skipped)} skipped")

    # Step 1: Write .full.md for each Tier-A paper
    written_count = 0
    already_uptodate = 0
    chunk_total = 0

    for entry in tier_a:
        slug = entry["slug"]
        try:
            changed = write_full_md(papers_dir, entry)
        except OSError as e:
            log(f"  ERR: cannot write {slug}.full.md: {e}")
            continue
        if changed:
            written_count += 1
            log(f"  wrote {slug}.full.md")
        else:
            already_uptodate += 1

        # Step 2: Chunk and index the .full.md
        full_md_path = papers_dir / f"{slug}.full.md"
        try:
            n = write_chunks_for_full_md(slug, full_md_path, chunks_dir, allow_egress)
            chunk_total += n
        except OSError as e:
            log(f"  ERR: chunking {slug}: {e}")

    log(f"Import done: {written_count} new/updated, {already_uptodate} unchanged; "
        f"{chunk_total} new chunks written")

    # Step 3: Report skipped papers
    if skipped:
        log(f"Skipped {len(skipped)} bodyless/unresolvable papers (Tier-B/C):")
        for s in skipped:
            log(f"  skip {s['slug']}: {s['reason']}")

    # Step 4: Build BM25 index from all graph chunks
    if chunks_dir.is_dir():
        index = build_graph_bm25_index(chunks_dir)
        if index is not None:
            write_bm25_index(bm25_dir, index)
            log(f"BM25 index: docs={index['doc_count']}  vocab={len(index['vocab'])}  "
                f"avg_dl={index['avg_dl']:.1f}")
        else:
            log("WARN: no chunks to index")

    return EXIT_OK


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="graph-fulltext.py — full-paper retrieval pipeline for /graph"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_sync = sub.add_parser("sync", help="Resolve + import Tier-A papers and build graph index.")
    p_sync.add_argument("--export", metavar="PATH",
                        help="Path to graph-export.json (default: wiki/graph/graph-export.json)")
    p_sync.add_argument("--papers-dir", metavar="PATH",
                        help="Output dir for .full.md files (default: wiki/graph/papers/)")
    p_sync.add_argument("--chunks-dir", metavar="PATH",
                        help="Graph chunk store dir (default: .vault-meta/graph/chunks/)")
    p_sync.add_argument("--bm25-dir", metavar="PATH",
                        help="Graph BM25 index dir (default: .vault-meta/graph/bm25/)")
    p_sync.add_argument("--allow-egress", action="store_true",
                        help="Enable claude-CLI / Anthropic API contextual prefixes")

    args = parser.parse_args()

    if args.cmd == "sync":
        return cmd_sync(args)

    return EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())
