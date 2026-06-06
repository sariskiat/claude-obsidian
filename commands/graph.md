---
description: Claim-centric knowledge graph — build the index, scan for research gaps, or resolve duplicate entities. Reads the graph skill.
---

Read the `graph` skill (`skills/graph/SKILL.md`). Then route on `$ARGUMENTS`:

- **(no args)** or **status** — Report graph state: does `.vault-meta/graph/graph.db` exist?
  Counts of papers / entities / claims (from the derived DB or `wiki/graph/graph-export.json`).
  Then offer the three actions below.

- **build** — Rebuild the derived index from the markdown source of truth:
  `uv run python scripts/graph-build.py wiki/graph .vault-meta/graph/graph.db`
  The DB is derived and gitignored; safe to run any time.

- **gaps** — Scan for research gaps (build first if the DB is missing):
  `uv run python scripts/graph-gaps.py --top 20`
  Summarize by species (frontier / debate / replication / coverage / white-space). If the
  user named a need ("what should I study next" → frontier + white-space; "what's
  unconfirmed" → replication), filter to that species rather than dumping all of them.

- **resolve** — Propose duplicate-entity merges for human review:
  `uv run python scripts/graph-resolve.py --json`
  Present the ranked proposals (exact tier-1 first, fuzzy tier-2 after). **Never auto-merge** —
  the user confirms each one.

- **read** `<query>` | **read** `--paper <slug>` | **read** `--claim <id>` — Retrieve full-text
  passages from indexed papers. Requires `graph-fulltext.py sync` to have been run first
  (builds the graph BM25 index). Uses BM25 + local rerank (ollama/nomic-embed-text when
  available; degrades to BM25-only if not). Outputs ranked passages with paper+claim
  provenance (page_path, snippet).
  ```bash
  # free-text query across all indexed papers
  uv run python scripts/graph-retrieve.py "constrained sampling garment pinning"
  # all top passages from one paper
  uv run python scripts/graph-retrieve.py --paper <slug>
  # trace a claim to its source paper then return passages
  uv run python scripts/graph-retrieve.py --claim <id> --export wiki/graph/graph-export.json
  # build / refresh the graph full-text index first if not already done
  uv run python scripts/graph-fulltext.py sync
  ```
  If the index is missing, exit 10 with a "run graph-fulltext.py sync first" hint.

- **export** `<source.db>` — Re-export a sqlite graph into `wiki/graph/` markdown + snapshot
  (reads a copy, never mutates the source):
  `uv run python scripts/graph-export.py <source.db> wiki/graph`

Always run scripts under `uv run`. The claim layer (`wiki/graph/`) is separate from the prose
page wiki (`wiki/entities/`) — do not merge them.
