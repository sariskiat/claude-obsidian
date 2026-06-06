# Next feature (after full-paper retrieval) — proposal finder / graph-bridge.py (handoff)

**Status:** queued, spec-gated, NOT started. Build AFTER `full-paper-retrieval` (proposals are far stronger when grounded in real paper math). **Grill → approved spec → TDD. No code until approved.**

## One-line goal
Turn the graph's white-space gaps into **ranked, justified next-paper proposals** — "connect idea X (community A) to idea Y (community B) because…, grounded in papers P, Q" — to drive Saris's research plan (see memory `research-goal-phd-paper`).

## Important: no oracle to match
The original `bridge.py` (oracle reference) was in `~/.claude/skills/graphbuilding`, which has been **retired and emptied from Trash — it is gone from disk.** That's fine: a proposal engine is **generative/heuristic**, not a deterministic migration, so there is no byte-parity test (unlike `graph-gaps.py`). Build it **spec-first** from the user's judgment of what a good proposal looks like. If the old `bridge.py` is ever wanted as inspiration, recover from a Time Machine/backup — it is NOT required.

## Raw material (all intact, native)
- `scripts/graph-gaps.py` → white-space species (312 gaps): pairs of graph regions with **zero claim-edges** between them. This is the proposal substrate.
- Louvain communities (seed=42), entities (typed), claims (typed triples) in the derived db / `wiki/graph/`.
- After full-paper-retrieval ships: full paper text for grounding the proposed bridge's math.

## Worked example to satisfy (acceptance anchor)
The engine should be able to surface: *VTON community (35) has zero claim-edges to Aek's communities (diffusion sampling / noise optimization / distillation)* → propose the **"Extrapolation-Based Iterate Correction"** bridge (Heavy-Ball-Momentum / GHVB ↔ Neon negative extrapolation), with the papers that anchor each side. (This bridge already exists hand-written in the user's research notes — use it as the gold-standard shape.)

## Grill must close
1. **Unit** — propose bridges between entity-pairs, or community-pairs, or both?
2. **Ranking heuristic** — what makes a bridge "good"? (e.g. community distance, shared neighbors, each side closes a stated limitation, citation proximity). Saris must define the signal.
3. **Grounding** — pure-graph proposal, or pull related papers' full text (via full-paper-retrieval) and have Claude synthesize the bridge + justification?
4. **Output** — JSON schema + a human report; how many proposals; dedup against already-written bridges in the research notes?
5. **Surface** — `/graph bridge` (or `/graph propose`) command + `scripts/graph-bridge.py`.

## Constraints
- Native, uses `graph_db.root()`, no oracle import (there is none). Markdown SoT; derived db throwaway.
- TDD: failing test first (test the heuristic + schema + edge cases on a fixture graph, not parity).
- New branch `feature/graph-bridge`. `make test-graph` stays green. Verifier before commit.

## To start (after full-paper-retrieval is merged)
> Read docs/next-proposal-finder.md and grill me one question at a time on the proposal-finder (graph-bridge.py) until we have an approved spec. No code until I approve.
