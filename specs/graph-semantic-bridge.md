# Spec: Semantic Bridge — Claude-Code Directions Report from the Graph

**Feature ID:** `graph-semantic-bridge`
**Status:** approved
**Grilled on:** 2026-06-07
**Approved by:** saris on 2026-06-07
**Problem classification:** `unique`

## Problem Statement

The heuristic `graph-bridge.py` ranks *where* the white-space bridges are, but it cannot reason about *whether they are any good*, so there is no `proposals.md`-grade research-directions analysis generated from the graph.

## Objective

Turn the graph's ranked white-space bridges + the user's research situation into a harsh, opinionated, fully-grounded directions report — the genre of `~/Desktop/research/proposals.md` — written by Claude Code, so the user can pick a next-paper direction with the same taste a good advisor would bring, every citation traceable to a real graph node.

## User Context

- **Primary user:** saris (single-tenant fork owner). Goal: a first-author top-venue paper strong enough to be a top-tier PhD ticket. VTON + applied diffusion strength; theory muscle building; ~4h/week.
- **Environment:** macOS, `uv` Python 3.12, `/graph` skill v1.10. `claude` IS on PATH. No API key in repo `.env`.
- **Current workflow:** `/graph bridge` returns ranked pairs + a thin `--synthesize` line ("Connect A with B via X. dr=1.0"). To get a real analysis the user hand-writes it (that is what `proposals.md` is).
- **Pain points:** the gap between the heuristic's shallow narrative and a `proposals.md`-grade document is entirely manual; a fresh `claude -p` starts blank and will hallucinate citations if uncontrolled.
- **Stakeholders consulted:** saris (grill, 2026-06-07).

## JTBD

When I see ranked white-space bridges in my claim graph, I want Claude Code to read the grounded evidence plus my research situation and write a `proposals.md`-grade directions report (decision matrix, per-direction harsh takedown, execution probe), with every cited paper/entity verified against the real graph, so I can choose my next paper instead of hand-writing the analysis myself.

## Users and Roles

| Actor | Can do | Cannot do |
|---|---|---|
| saris | run `/graph propose`, edit `RESEARCH_PROFILE.md`, tune flags, read/keep/discard generated reports | n/a (single tenant) |
| `claude -p` subprocess | write the report prose from the assembled dossier | invent citations that survive the grounding gate; write anywhere on disk |
| the script | assemble dossier, prompt, run claude, verify grounding, save report | overwrite the source `proposals.md`; mutate the graph db / `graph-build.py` |

## Functional Requirements

| ID | Requirement | Notes |
|---|---|---|
| FR1 | New `scripts/graph-propose.py`, surfaced as `/graph propose`. | Verb = "propose directions". |
| FR2 | Reuse `graph-bridge.build_proposals(conn, top_n=K)` for candidates — import, do not fork. | `--bridges K` (default 12). |
| FR3 | Assemble a deterministic **dossier** per candidate: anchor entities, anchor papers, limitation/open-question claims, and full-text passages via `graph-retrieve.py`. | Dossier is JSON-serializable + testable with no egress. |
| FR4 | Build an explicit **citable allow-list** = the exact set of paper slugs + entity names present in the dossier; inject it into the prompt with a "cite ONLY these" instruction. | Allow-list is the grounding contract's source of truth. |
| FR5 | Inject `wiki/graph/RESEARCH_PROFILE.md` (facts) + `~/Desktop/research/proposals.md` (style/taste exemplar, read-only) into the prompt. | Profile seeded at build-time from the `research-goal-phd-paper` memory note + the proposals.md header. |
| FR6 | Engine = **headless `claude -p`** every run (egress accepted, by design). `--claude-cmd` (default `claude`) overrides the binary so tests inject a fake. | No in-session / synthetic path. |
| FR7 | The prompt requires a fixed section contract so output is checkable: `## The bar` , `## Decision matrix` (a table), `### N.` per-direction blocks each containing a **Takedown** subsection, `## Ranking`, `## Execution` (the probe). | Mirrors `proposals.md`. |
| FR8 | **Grounding gate (hard-fail + retry).** After each `claude -p` run, extract every cited slug/entity and verify against `graph.db`. Any unverified citation → re-prompt (emphasizing the offending citations) up to `--retries` (default 3). | 100%-grounded is the only accept condition. |
| FR9 | On clean pass: write `wiki/graph/proposals/<YYYY-MM-DD>-directions.md` (suffix `-2`,`-3`… if same-day file exists) with a grounding-audit footer (`N/N citations verified ✓`, retry count, model, bridge count). | Timestamped, never clobbers. |
| FR10 | On cap exhaustion: write `wiki/graph/proposals/<...>-directions.rejected.md` with the unverified citations flagged inline, emit the failing list to stderr, exit non-zero. No clean report is saved. | Hard-fail is visible, work not lost. |
| FR11 | Wire `/graph propose` into `commands/graph.md` and `skills/graph/SKILL.md`; add `make test-propose`. | Discoverable. |

## Happy Path

1. `uv run python scripts/graph-propose.py` (or `/graph propose`) → load derived db.
2. Reuse `graph-bridge` → top-K candidate bridges (gold anchor VTON×diffusion among them).
3. For each: gather anchor entities/papers, limitation claims, full-text passages → **dossier** + citable allow-list.
4. Build prompt = system(harsh advisor + section contract + "cite ONLY allow-list") + `RESEARCH_PROFILE.md` + `proposals.md` exemplar + dossier.
5. `claude -p` writes the report.
6. Grounding gate: extract citations → verify ⊆ allow-list/db. Dirty → retry (≤3). Clean → continue.
7. Save `wiki/graph/proposals/2026-06-07-directions.md` + audit footer. Exit 0.

## Business Rules

| Rule ID | Rule | Source/Rationale | Verification |
|---|---|---|---|
| BR1 | Every citation in an accepted report resolves to a real `papers.slug` or `entities.name` in `graph.db`. | "verify, don't trust self-tests" (memory). | grounding-gate test + live audit footer |
| BR2 | Hard-fail + retry ≤3; cap exhaustion → non-zero exit + `.rejected` artifact, no clean save. | user choice (grill Q4). | cap-exhaustion test |
| BR3 | Engine egresses every run (headless `claude -p`); this feature is explicitly the egress-on path. LLM prose is **non-deterministic and exempt** from determinism checks; the *dossier* and *grounding gate* are deterministic and tested. | user choice (grill Q2). | dossier-determinism test |
| BR4 | `~/Desktop/research/proposals.md` is read-only input; the system never writes there. | source-doc safety (grill Q5). | no-clobber test + grep |
| BR5 | Generated reports are timestamped under `wiki/graph/proposals/`; same-day reruns suffix `-2/-3`, never overwrite. | history + no clobber. | rerun-suffix test |
| BR6 | `graph-build.py` and the 9-table round-trip are untouched; `graph-bridge.py`/`graph-retrieve.py` reused, not forked. | protect the verified core. | `make test-graph` + git diff |

## Data and Integrations

| Data/API | Owner | Read/Write | Failure behavior |
|---|---|---|---|
| `.vault-meta/graph/graph.db` | derived | read | missing → exit 1 + build hint |
| `graph-bridge.build_proposals` | repo | import (read) | reused for candidates |
| `graph-retrieve.py` | repo | subprocess (read) | index absent → passage note, not crash |
| `wiki/graph/RESEARCH_PROFILE.md` | saris | read (created at build) | missing → friendly non-zero + "create it" hint |
| `~/Desktop/research/proposals.md` | saris | **read-only** | missing → warn, run without exemplar |
| `claude -p` (via `--claude-cmd`) | system | subprocess (egress) | binary absent → friendly non-zero + hint |
| `wiki/graph/proposals/<date>-directions.md` | system | **write** | clean pass only |

## Errors and Edge Cases

| Scenario | Expected behavior |
|---|---|
| db missing | exit 1, build hint |
| `claude`/`--claude-cmd` absent | non-zero, "install/point --claude-cmd" hint, no file written |
| `RESEARCH_PROFILE.md` missing | non-zero + hint to create (it's seeded at build, so normally present) |
| `proposals.md` exemplar missing | warn, proceed without exemplar (degraded taste, still grounded) |
| no white-space bridges | empty-state report note, exit 0 |
| graph index absent (passages) | dossier still built from db; passages show "(no full text indexed)" |
| dirty citations after 3 retries | `.rejected.md` + flagged cites + non-zero exit |
| same-day file exists | suffix `-2`/`-3`; never overwrite |

## Constraints

- **Must not break:** `make test-graph` (44 passed / 4 skipped), `make test-fulltext` (44), `make test-bridge`; the 9-table round-trip byte-equal.
- **Performance:** dossier + retrieval well under a minute; `claude -p` latency is the dominant term (acceptable, AFK tool).
- **Security/compliance:** egress is opt-in-by-design and only to `claude -p`; no API keys read; no other network calls.
- **Mutable surface:** `scripts/graph-propose.py`, `tests/test_graph_propose.py`, `wiki/graph/RESEARCH_PROFILE.md` (new), `wiki/graph/proposals/` (new, generated), `commands/graph.md`, `skills/graph/SKILL.md`, `Makefile`, `wiki/graph/SCHEMA.md`, `.gitignore`.
- **Read-only surface:** `scripts/graph-bridge.py`, `scripts/graph-retrieve.py`, `scripts/graph-fulltext.py`, `scripts/graph_db.py`, `scripts/graph-build.py`, `scripts/graph-export.py`, `scripts/graph-gaps.py`, `scripts/graph-resolve.py`, `scripts/contextual-prefix.py`, all existing `tests/test_graph_*.py`, `~/Desktop/research/**`, `.vault-meta/graph/graph.db`.

## CAN / CANNOT

- **CAN modify:** the mutable surface above.
- **CANNOT modify:** `graph-build.py`, `graph-bridge.py`, `graph-retrieve.py`, `graph_db.py`, the round-trip tests, `~/Desktop/research/proposals.md`.
- **Needs human approval before changing:** the ranking weights inside `graph-bridge.py` (out of scope here); the section contract (FR7) if `proposals.md` genre changes.

## Acceptance Criteria

| ID | Criterion | Verification |
|---|---|---|
| AC1 | Dossier assembly is deterministic and grounded: every candidate's anchor entities/papers exist in `graph.db`; dossier carries the citable allow-list. (no egress) | `uv run python -m pytest tests/test_graph_propose.py -q -k dossier` |
| AC2 | Prompt builder injects `RESEARCH_PROFILE.md` facts + `proposals.md` exemplar + bridge dossier + explicit allow-list + section contract. | `pytest -k prompt` |
| AC3 | Grounding gate accepts a clean fake report (cites ⊆ allow-list) → saves file with `N/N verified ✓`; retries a dirty-then-clean fake and records retry count. | `pytest -k grounding` (fake `--claude-cmd`) |
| AC4 | Hard-fail: always-dirty fake → after ≤3 retries, non-zero exit + `.rejected.md` with flagged cites, and NO `-directions.md` saved. | `pytest -k cap_exhausted` |
| AC5 | Source safety + no clobber: `~/Desktop/research/proposals.md` never written; output under `wiki/graph/proposals/`; same-day rerun suffixes `-2`. | `pytest -k no_clobber` |
| AC6 | Degradation: missing db → exit 1 + hint; `--claude-cmd` absent → non-zero + hint; missing profile → hint. | `pytest -k degrade` |
| AC7 | Gold anchor available: the VTON×diffusion-sampling bridge appears in the candidate dossier by entity membership (not a hardcoded id). | `pytest -k gold_anchor` |
| AC8 | No regression: `make test-graph` (44/4), `make test-fulltext` (44), `make test-bridge` stay green; `graph-build.py` unmodified. | `make test-graph && make test-fulltext && make test-bridge` |
| AC9 | **Live evidence:** one real `claude -p` run over the live graph yields a structurally-complete report (section contract present: bar / decision-matrix table / ≥3 direction blocks each with a Takedown / ranking / execution probe) with a `N/N citations verified ✓` footer and 100% grounded cites; transcript saved to the overnight report. | live run + `grep -E` on the saved report |
| AC10 | Wiring: `commands/graph.md` and `skills/graph/SKILL.md` both reference `/graph propose` → `graph-propose.py`; `make test-propose` exists. | `grep -q 'graph-propose' commands/graph.md skills/graph/SKILL.md` |

## Metric

- **Primary metric:** binary pass — a live `/graph propose` run produces a 100%-grounded, structurally-complete `proposals.md`-grade directions report (AC9), with AC1–AC8 + AC10 green.
- **Baseline:** 0 (no semantic report generator; only the thin heuristic `--synthesize` line).
- **Target:** 1 grounded structurally-complete report saved + grounding gate proven (clean-accept, dirty-retry, cap-reject) by tests with a faked engine.
- **Direction:** binary.
- **Why the user will notice:** today the analysis is hand-written; after, it is one AFK command that returns a harsh, grounded directions doc — and the user can trust the citations because the gate rejects hallucinations.

## Success Signal

- **Completion promise:** `/graph propose` writes a grounded, structurally-complete directions report or fails loudly; the grounding gate is proven in tests with a faked engine; no regression.
- **Required evidence:** green `tests/test_graph_propose.py`; `make test-graph`/`test-fulltext`/`test-bridge` green; one live transcript with the audit footer; an idempotent-safe no-clobber demo.
- **Expected final marker:** a saved `wiki/graph/proposals/<date>-directions.md` whose footer reads `N/N citations verified ✓`.

## Out of Scope

- In-session (non-headless) generation; Anthropic API / Agent SDK path.
- Re-tuning `graph-bridge.py` ranking weights.
- Per-bridge deep-dive mode and interactive `/graph propose <focus>` (the whole-graph report is the chosen shape; these are possible later modes).
- Auto-emailing Aek or any outward action (memory phase-rule: no Aek contact until the 4 deep-reads are done).

## Approval Checklist

- Domain lead approved: saris (2026-06-07)
- Tech lead approved: saris (2026-06-07)
- Critic reviewed testability: yes — deterministic dossier + faked-engine grounding tests keep the suite offline
- Metric validated against user value: yes — one AFK command returns a grounded, harsh directions doc

## Open Questions

- None (output location delegated to me → vault + timestamped + source read-only; retry cap defaulted to 3, CLI-tunable).
