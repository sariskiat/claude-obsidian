# Spec: Proposal Finder — graph-bridge.py

**Feature ID:** graph-bridge
**Status:** `approved` (autonomous draft — pre-authorized by saris: "skipp grill … keep stress testing 100 time until it able to do this", 2026-06-06. Grill waived by the owner. Built local-only and reversible; **awaiting human ratification on wake** — nothing pushed or merged.)
**Grilled on:** n/a (owner waived grill)
**Approved by:** `saris (delegated, pre-authorized) 2026-06-06` — ratify on wake
**Problem classification:** `unique`

## Problem Statement

The graph can name white-space gaps (community pairs with zero claim-edges) but cannot turn them into ranked, justified next-paper proposals, so the analysis stops short of the research output it exists to serve.

## Objective

Turn white-space gaps into ranked, grounded "connect idea X (community A) to idea Y (community B) because …, anchored in papers P, Q" proposals — to drive Saris's first-author-paper plan ([[research-goal-phd-paper]]): novel-but-defensible bridges, no application layer.

## User Context

- **Primary user:** saris (single-tenant fork owner), asleep — this was built autonomously per an explicit "skip grill + keep stress testing until it works" directive.
- **Environment:** macOS, `uv` Python 3.12, `/graph` skill, derived db at `.vault-meta/graph/graph.db`.
- **Current workflow:** `/graph gaps` lists 116 white-space community-pair gaps; the user then reasons by hand toward a bridge (e.g. the hand-written "Extrapolation-Based Iterate Correction" linking Heavy-Ball-Momentum/GHVB ↔ Neon negative extrapolation).
- **Pain:** no engine ranks which gaps are worth bridging or grounds them in the papers' math.

## JTBD

When I see white-space in my claim graph, I want a ranked shortlist of justified bridge proposals grounded in the real papers, so I can pick my next paper's contribution instead of eyeballing 116 gaps.

## Users and Roles

| Actor | Can do | Cannot do |
|---|---|---|
| saris | run `/graph bridge`, tune signal weights, opt into `--synthesize` | n/a (single-tenant) |
| graph-bridge (system) | read derived db + graph-gaps white-space + graph-retrieve passages; emit JSON + report | mutate the markdown SoT or the db |

## Functional Requirements

| ID | Requirement | Notes |
|---|---|---|
| FR1 | Consume `graph-gaps.py` white-space gaps (community-pairs, zero claim-edges) as the proposal substrate — reuse, do not recompute Louvain. | Communities are runtime Louvain seed=42 inside graph-gaps. |
| FR2 | Rank candidates by a **transparent weighted heuristic** with a per-proposal **signal breakdown**. Weights are documented and CLI-tunable. | The owner retunes on wake — transparency is the risk control. |
| FR3 | Signals (all computable from the graph): `gap_confidence` (from white-space), `bridgeability` (shared-neighbor entities / citation_links / entity_edges spanning the pair — latent path despite zero claim-edge), `limitation_pull` (count of limitation/open-question claims each side), `richness` (min entities/claims per side), `direction_relevance` (boost pairs touching the user's priority areas: virtual-try-on AND diffusion-sampling/noise-optimization/distillation). | direction_relevance encodes [[research-goal-phd-paper]] so the gold-standard VTON↔Aek bridge ranks high. |
| FR4 | Ground each top proposal: anchor entities (most-connected per side), anchor papers (papers contributing those entities/claims), and top full-text passages from each side via `graph-retrieve.py`. | Reuses the full-paper retrieval shipped this cycle. |
| FR5 | Optional `--synthesize`: produce a narrative "connect X to Y because …" justification via the claude-CLI tier (no API key). **Off by default** (zero egress); gated through `contextual-prefix.pick_prefix_tier`. | Mirrors the egress posture; default output is graph-grounded structure, no LLM call. |
| FR6 | Output: JSON (list of proposals: id, community_a, community_b, anchor_entities, anchor_papers, score, signal_breakdown, passages, [justification]) + a markdown report. Default top-10. | |
| FR7 | Light dedup: flag a proposal whose endpoints already correspond to a hand-written bridge entity in the graph (e.g. "Extrapolation-Based Iterate Correction") as `already_proposed: true` rather than dropping it. | Full dedup against external research notes is out of scope. |
| FR8 | Surface as `scripts/graph-bridge.py` + `/graph bridge` subcommand. | Mirrors status/build/gaps/resolve/export/read. |

## Happy Path

1. `uv run python scripts/graph-bridge.py --top 10` → ranked proposals (JSON or report).
2. Each proposal: community A ↔ community B, why (signal breakdown), anchor entities + papers, and a few grounding passages from each side.
3. The VTON ↔ diffusion-sampling/distillation bridge appears near the top (the gold-standard anchor).
4. Optional: `--synthesize` writes the prose justification (opt-in, claude-CLI).

## Business Rules

| Rule ID | Rule | Rationale | Verification |
|---|---|---|---|
| BR1 | Only propose pairs with **zero claim-edges** between them (true white-space). | A "bridge" over an existing claim-edge isn't new. | unit test on fixture |
| BR2 | Deterministic given seed=42 (same Louvain, same ranking) — repeat runs identical. | Reproducible research. | run twice, diff empty |
| BR3 | Every proposal cites only entities/papers that exist in the graph (no fabrication). | Grounding integrity. | schema + existence test |
| BR4 | Default run does **zero egress**; `--synthesize` is the only egress path, opt-in. | Egress posture. | no-network assert |
| BR5 | Reuse `graph_db.root()`, `graph-gaps` white-space, `graph-retrieve`; no oracle import (the bridge.py oracle is gone). | Native, DRY. | grep no oracle path |

## Data and Integrations

| Data/API | Owner | R/W | Failure behavior |
|---|---|---|---|
| `.vault-meta/graph/graph.db` (derived) | existing | Read | missing → friendly non-zero exit, "run graph-build" |
| `graph-gaps.py` white-space | existing | Reuse | reuse function/CLI |
| `graph-retrieve.py` passages | this cycle | Reuse | ollama absent → BM25-only (already degrades) |
| `contextual-prefix.pick_prefix_tier` | existing | Reuse (synth gate) | default synthetic, no egress |

## Errors and Edge Cases

| Scenario | Expected |
|---|---|
| derived db missing | non-zero exit + build hint |
| no white-space gaps | empty proposal list, exit 0, friendly note |
| a community with 0 papers / no full text | proposal still emitted from graph structure; passages section notes "no full text" |
| `--synthesize` without claude on PATH | fall back to structured justification, note it, exit 0 |
| `--top` larger than candidate count | return all, exit 0 |
| duplicate-slug papers in anchors | dedup by content (reuse the retrieval dedup notion) |

## Constraints

- **Must not break:** `make test-graph` (44/4) + `make test-fulltext` (44) stay green; markdown SoT; derived db throwaway; `uv` for all; TDD failing-test-first; **local only — no push** (owner pushes on wake).
- **Performance:** ranking 116 gaps + grounding top-10 completes in well under a minute (synthetic/zero-egress path).
- **Security:** zero egress by default; `--synthesize` opt-in claude-CLI only.
- **Mutable surface:** `scripts/graph-bridge.py` (new), `tests/test_graph_bridge.py` (new), `commands/graph.md` (+`bridge`), `skills/graph/SKILL.md` (+bridge routing), `Makefile` (optional `test-bridge`), `wiki/graph/SCHEMA.md` (optional bridge note).
- **Read-only:** `graph_db.py`, `graph-build.py`, `graph-export.py`, `graph-gaps.py`, `graph-resolve.py`, `graph-validate.py`, `graph-retrieve.py`, `graph-fulltext.py`, `rerank.py`, `bm25-index.py`, `contextual-prefix.py`, and all existing tests.

## CAN / CANNOT

- **CAN modify:** the new script/tests, `commands/graph.md`, `skills/graph/SKILL.md`, `Makefile`, `wiki/graph/SCHEMA.md`.
- **CANNOT modify:** the existing graph engine scripts + their tests; the derived db schema; source dirs.
- **Needs human ratification:** the ranking weights and whether community-pair (vs entity-pair) is the right unit — flagged, built conservative + tunable.

## Acceptance Criteria

| ID | Criterion | Verification |
|---|---|---|
| AC1 | Ranks white-space gaps into proposals with a per-proposal signal breakdown; deterministic (seed=42). | `pytest tests/test_graph_bridge.py -k rank_deterministic` |
| AC2 | Every proposal cites only entities/papers present in the graph (no fabrication); only zero-claim-edge pairs. | `pytest -k grounding_integrity` |
| AC3 | **Gold anchor:** a VTON-containing community ↔ a diffusion-sampling/distillation-containing community bridge appears in the top-N (identified by entity membership, not a hardcoded community number). | `pytest -k gold_anchor_vton_aek` |
| AC4 | Default run does zero network egress; `--synthesize` is the only egress path. | `pytest -k no_egress_default` |
| AC5 | Graceful: missing db → non-zero; no gaps → empty+exit0; `--synthesize` without claude → structured fallback. | `pytest -k degrade` |
| AC6 | JSON schema valid + markdown report renders; `--top` honored. | `pytest -k schema` |
| AC7 | No regression: `make test-graph` 44/4 + `make test-fulltext` 44 green. | `make test-graph && make test-fulltext` |
| AC8 | `/graph bridge` wired (commands + SKILL). | `grep -q "graph-bridge\|graph bridge" commands/graph.md skills/graph/SKILL.md` |

## Metric

- **Primary:** does the gold-standard VTON↔Aek bridge surface in the top-N, and are proposals well-formed + grounded across a 100-case stress test?
- **Baseline:** 0 (no engine).
- **Target:** gold anchor in top-N; 100-case stress test ≥ 95% well-formed + grounded + non-degenerate + no crash; deterministic.
- **Direction:** binary pass.
- **Why the user notices:** 116 raw gaps become a ranked, justified shortlist that names the next paper.

## Success Signal

- **Completion promise:** `/graph bridge` produces ranked, grounded next-paper proposals; the known VTON↔Aek bridge is among them.
- **Required evidence:** `make test-graph`/`test-fulltext` green; `tests/test_graph_bridge.py` green; a live `/graph bridge` report showing the gold anchor + signal breakdown; a 100-case stress-test scorecard; idempotent re-run.
- **Expected final marker:** all ACs verified by fresh command output; Evaluator PASS; local only.

## Out of Scope

- LLM-judged ranking (kept heuristic + transparent); full dedup against external research notes; entity-pair bridges (community-pair this slice); writing proposals back into the graph as claims.

## Approval Checklist

- Domain lead approved: **pending human ratification on wake** (built under delegated pre-authorization)
- Tech lead approved: pending ratification
- Critic reviewed testability: Planner/Evaluator
- Metric validated against user value: yes (ranked grounded proposals from raw gaps)

## Open Questions (flagged for ratification — built conservative defaults)

1. **Ranking weights** — chose a documented default weighting; CLI-tunable. Retune on wake.
2. **Unit** — community-pair (matches the worked example + white-space substrate). Entity-pair is a possible future mode.
3. **Synthesis** — off by default (zero egress); opt-in claude-CLI. Confirm you want LLM narratives at all.
