#!/usr/bin/env python3
"""Fetch the 2 straggler papers from arXiv by title, extract to paper-scholar format.
Usage: uv run --with pymupdf4llm --with pymupdf python /tmp/afk_stragglers.py
"""
import os, sys, json, urllib.request, urllib.parse, re, glob

PS = os.path.expanduser("~/.paper-scholar")
sys.path.insert(0, "/tmp")
from afk_extract import md_from_pdf

STRAGGLERS = [
    ("span-id-page-0-0-span-defog-discrete-flow-matching-for-graph-generation",
     "DeFoG Discrete Flow Matching for Graph Generation"),
    ("span-id-page-0-0-span-temporal-difference-flows",
     "Temporal Difference Flows"),
]
def arxiv_lookup(title):
    q = urllib.parse.urlencode({"search_query": f'ti:"{title}"', "max_results": "1"})
    url = f"http://export.arxiv.org/api/query?{q}"
    xml = urllib.request.urlopen(url, timeout=30).read().decode()
    m_id = re.search(r"<id>http://arxiv.org/abs/([^<]+)</id>", xml)
    m_ti = re.search(r"<title>([^<]+)</title>", xml[xml.find("<entry>"):]) if "<entry>" in xml else None
    if not m_id:
        return None, None
    raw = m_id.group(1)               # e.g. 2410.04263v3
    aid = re.sub(r"v\d+$", "", raw)
    return aid, (m_ti.group(1).strip() if m_ti else "")

def main():
    for slug, title in STRAGGLERS:
        outdir = os.path.join(PS, slug)
        if glob.glob(os.path.join(outdir, "*.md")):
            print(f"SKIP (already present): {slug[:40]}")
            continue
        aid, found = arxiv_lookup(title)
        if not aid:
            print(f"NOT FOUND on arxiv: {title}")
            continue
        print(f"  {title[:40]:40s} -> arXiv:{aid}  ({found[:40]})")
        pdf_url = f"https://arxiv.org/pdf/{aid}.pdf"
        tmp = f"/tmp/{aid}.pdf"
        try:
            urllib.request.urlretrieve(pdf_url, tmp)
        except Exception as e:
            print(f"    download FAIL: {e}"); continue
        md, npages = md_from_pdf(tmp)
        os.makedirs(outdir, exist_ok=True)
        open(os.path.join(outdir, f"{aid}.md"), "w", encoding="utf-8").write(
            f"<!-- downloaded from arXiv:{aid}, extracted by afk ({npages}p) -->\n\n" + md)
        json.dump({"source": pdf_url, "arxiv_id": aid, "n_pages": npages, "chars": len(md),
                   "extractor": "pymupdf4llm", "slug": slug, "matched_title": found},
                  open(os.path.join(outdir, f"{aid}_meta.json"), "w"), indent=2)
        print(f"    OK {len(md)}b {npages}p -> ~/.paper-scholar/{slug}/{aid}.md")

if __name__ == "__main__":
    main()
