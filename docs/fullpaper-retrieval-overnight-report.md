# Full-Paper Retrieval — Overnight Build Report

**For:** saris
**Date:** 2026-06-06 (overnight, autonomous)
**Branch:** `feature/graph-full-paper-retrieval` (off `feature/graph-resolve-dedup`)
**State:** built, hardened, Evaluator PASS. **Local only — nothing pushed.** Ready for your review and manual push.

---

## What you asked for, and what happened

You approved the spec, told me to build it, then before sleeping asked me to: get full text for every paper that lacked it (transcribing the PDFs myself, fast, not via the slow paper-scholar skill), re-import, and pen-test the system 100 cases until it does the job. All four are done.

| Ask | Result |
|---|---|
| Build the full-paper retrieval system | Done via the TILDI spine (Planner → Generator → Evaluator). `/graph read` works. |
| Get full text for the papers that lacked it | 57 PDFs extracted + 6 already on disk → **95/98 papers now have full text**. |
| Transcribe myself, fast, same format/dir as paper-scholar | pymupdf4llm → `~/.paper-scholar/<slug>/<arxivid>.md`, ~seconds/paper. |
| Pen-test 100 cases until it works | **99/100**; deterministic claim-trace **40/40**. Two real bugs found and fixed. |

---

## Coverage: 95/98 papers

- **Tier A (32):** already had full-text markdown via the original `source_path`.
- **Tier B (57):** PDFs under `~/Desktop/research/` (the VTON corpus + iclr2026) — extracted with pymupdf4llm into `~/.paper-scholar/<slug>/<arxivid>.md`, then imported.
- **Already on disk (6):** bare `.paper-scholar/<slug>` dirs that already held text (incl. DeFoG, Temporal Difference Flows — fetched earlier).
- **The 3 not covered are not papers:** `anthropic-kg-cookbook` and `multimodal-crop-tool` (Claude cookbook URLs), `bridge-synthesis` (a dead `~/.graphbuilding` reference). They stay bodyless by design.

Correction to the original handoff: it claimed the PDFs were trashed. They were not — 43 PDFs were under `~/Desktop/research/`, and the source paths were relative to that directory. That is why coverage reached 95, not the 38 the spec estimated.

## How it works (the system that's now in place)

- **Import** (`scripts/graph-fulltext.py sync`): resolves each paper's best full-text source (now including a **slug-dir fallback** that finds `~/.paper-scholar/<slug>/*.md`), writes a byte-equal `wiki/graph/papers/<slug>.full.md`, then chunks + builds a graph-scoped BM25 index. Idempotent.
- **Read** (`scripts/graph-retrieve.py`, surfaced as `/graph read`): `--claim <id>` and `--paper <slug>` are **deterministic chunk-address lookups**; a free-text `<query>` is BM25 + (optional) rerank. Returns passages with paper provenance.
- **Egress:** synthetic prefixes by default — nothing leaves the machine. `--allow-egress` (claude-CLI tier) is opt-in.
- **Build untouched:** `graph-build.py` reads `graph-export.json`, so `.full.md` files are inert to it; the 9-table round-trip stays byte-equal (`make test-graph` 44/4).

## Pen-test (100 cases, BM25-only — ollama not installed)

```
claim_text     59/60   98.3%      paper_direct   12/12  100%
quote          15/15  100%        edge           13/13  100%
Recall@1       70/75   93.3%      OVERALL        99/100  99%
--claim trace (deterministic)     40/40  100%
```

The single miss (case 43) is a degenerate ground-truth: the claim text is literally "Evaluated on VITON-HD." — too generic for any method to pin one paper. The real way to trace that claim, `--claim <id>`, returns the correct paper (the trace path is exact).

## Two bugs the pen-test caught, both fixed (TDD)

1. **Duplicate-slug results** — some papers exist under two slugs (same body). Query results showed the paper twice and pushed distinct papers out of top-K. Fixed: dedup by chunk body-hash in query mode (`c955fed`).
2. **`--paper`/`--claim` returned empty for some papers** (e.g. ted-viton) — they were using BM25 *ranking* of the slug's words, which lost to other papers; the filter then matched nothing. Fixed: `--paper`/`--claim` now compute the chunk address from the slug and read chunks directly — a lookup, not a search (`6e9c0bd`). This is the core "trace a claim → paper" path.

Plus two minor close-outs (`ea3ed05`): unknown `--paper` now exits non-zero with a message; the board's AC `-k` filter strings were corrected to actually run.

## On rerank / OpenAI

ollama isn't installed, so rerank degrades to **BM25-only** (the designed fallback). I did not install ollama (heavyweight, unauthorized). You asked about an OpenAI key — I checked this repo's `.env` and the research repo's `.env` (narrowly, not a disk scan); **no OpenAI key is present**. It isn't needed: the one pen-test miss is a degenerate query that rerank cannot fix either. If you install ollama (`ollama serve` + `ollama pull nomic-embed-text`), rerank turns on automatically and would lift quality on paraphrased queries.

## Reproduce

```bash
make test-graph        # 44 passed, 4 skipped
make test-fulltext     # 44 passed
uv run python scripts/graph-fulltext.py sync          # idempotent re-import + index
uv run python scripts/graph-retrieve.py --claim 310   # trace a claim -> paper passages
# pen-test harness (persisted under scripts/fulltext-tools/):
uv run python scripts/fulltext-tools/afk_pentest_gen.py   # regenerate 100 cases
uv run python scripts/fulltext-tools/afk_pentest_run.py --top 5
```

## To finish (your call)

- Review the 9 commits on `feature/graph-full-paper-retrieval`, then push to your fork when ready (I did not push — a push to `fork` was blocked mid-run and I kept everything local per your egress posture).
- Decide whether the ~14 MB of committed `.full.md` should stay git-tracked (current, per the approved spec) or become gitignored/derived now that `~/.paper-scholar` is kept as upstream.
- Next slice (already specced as a follow-on): nothing required — the system is "ready" for re-adds. Drop a paper's text in `~/.paper-scholar/<slug>/` and re-run sync.
