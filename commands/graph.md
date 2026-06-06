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

- **export** `<source.db>` — Re-export a sqlite graph into `wiki/graph/` markdown + snapshot
  (reads a copy, never mutates the source):
  `uv run python scripts/graph-export.py <source.db> wiki/graph`

Always run scripts under `uv run`. The claim layer (`wiki/graph/`) is separate from the prose
page wiki (`wiki/entities/`) — do not merge them.
