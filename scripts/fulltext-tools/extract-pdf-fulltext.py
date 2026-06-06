#!/usr/bin/env python3
"""AFK fast PDF -> paper-scholar-format markdown extractor.

Writes ~/.paper-scholar/<graph-slug>/<arxivid>.md (+ _meta.json) so the graph
resolver's slug-dir fallback can pick it up. Fast (pymupdf4llm, ~seconds/paper),
NOT marker. Idempotent: skips a slug that already has its <id>.md unless --force.

Usage:
  uv run --with pymupdf4llm --with pymupdf python /tmp/afk_extract.py [--force] [--limit N] [--only SLUG]
"""
import json, os, re, sys, hashlib, time

INV = "/tmp/afk_paper_inventory.json"
PS = os.path.expanduser("~/.paper-scholar")

def md_from_pdf(pdf_path):
    """Return (markdown, n_pages). Prefer pymupdf4llm; fall back to plain text."""
    try:
        import pymupdf4llm
        md = pymupdf4llm.to_markdown(pdf_path, show_progress=False)
        import fitz
        n = fitz.open(pdf_path).page_count
        return md, n
    except Exception:
        import fitz
        doc = fitz.open(pdf_path)
        parts = []
        for pg in doc:
            parts.append(pg.get_text("text"))
        return "\n\n".join(parts), doc.page_count

def main():
    force = "--force" in sys.argv
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
    only = None
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1]

    rows = json.load(open(INV))
    todo = [r for r in rows if (not r["has_md"]) and r["pdf"]]
    if only:
        todo = [r for r in todo if r["slug"] == only]
    if limit:
        todo = todo[:limit]

    # cache extraction per unique pdf (duplicate slugs -> same pdf)
    cache = {}
    done = skipped = failed = 0
    for r in todo:
        slug, pdf = r["slug"], r["pdf"]
        aid = r["aid"] or hashlib.sha1(pdf.encode()).hexdigest()[:8]
        outdir = os.path.join(PS, slug)
        outmd = os.path.join(outdir, f"{aid}.md")
        if os.path.isfile(outmd) and not force:
            skipped += 1
            continue
        try:
            if pdf not in cache:
                t0 = time.time()
                cache[pdf] = md_from_pdf(pdf)
                dt = time.time() - t0
            md, npages = cache[pdf]
            if not md or len(md) < 500:
                print(f"  WARN tiny extract ({len(md)}b) {slug[:40]}")
            os.makedirs(outdir, exist_ok=True)
            # prepend title H1 from the graph slug if md doesn't start with a heading
            header = f"<!-- extracted by afk_extract from {os.path.basename(pdf)} ({npages}p) -->\n\n"
            open(outmd, "w", encoding="utf-8").write(header + md)
            meta = {"source_pdf": pdf, "arxiv_id": r["aid"], "n_pages": npages,
                    "chars": len(md), "extractor": "pymupdf4llm", "slug": slug}
            json.dump(meta, open(os.path.join(outdir, f"{aid}_meta.json"), "w"), indent=2)
            done += 1
            print(f"  OK  {slug[:46]:46s} {len(md):>7}b {npages:>2}p")
        except Exception as e:
            failed += 1
            print(f"  FAIL {slug[:44]}: {e}")
    print(f"\nextracted={done} skipped(existing)={skipped} failed={failed} unique_pdfs={len(cache)}")

if __name__ == "__main__":
    main()
