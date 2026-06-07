# Spec: Claude -p Reranker Tier

**Feature ID:** `wiki-rerank-claude`
**Status:** approved
**Grilled on:** 2026-06-07 (self-grilled; delegated by conductor)
**Approved by:** saris (delegated) on 2026-06-07
**Problem classification:** `enhancement`

---

## 1. Problem Statement

```
┌────────────────────────────┐     ┌────────────────────────────┐
│ PROBLEM                    │     │ TODAY'S REALITY             │
│                            │     │                             │
│ ollama is not installed     │     │ cosine tier is inert        │
│ on this machine.           │ ──▶ │ retrieval falls through to  │
│                            │     │ BM25-only order — no        │
│ cosine reranker requires   │     │ semantic reranking at all   │
│ nomic-embed-text via       │     │                             │
│ ollama to produce a        │     │ BM25 alone is keyword-only; │
│ ranked list.               │     │ it misses semantic matches  │
└────────────────────────────┘     └────────────────────────────┘
```

The vault has `claude` on PATH. `ollama` is not installed. The cosine tier in
`rerank.py` falls through to the no-op BM25 path on every query, so the rerank
stage adds nothing today. A `claude -p` tier placed between cosine and BM25-only
gives this single-tenant vault real semantic reranking without requiring any
additional infrastructure.

---

## 2. Objective

Add a `claude` reranker tier to `scripts/rerank.py`. The new chain is:

```
  cosine(ollama)   <-- unchanged; still preferred when available
        |
        | ollama unreachable OR model missing
        v
  claude -p rerank  <-- NEW; reorders BM25 candidates by semantic relevance
        |
        | claude not enabled (default) OR claude call fails
        v
  BM25-only (noop)  <-- unchanged fallback floor
```

The tier is opt-in and off by default. Default behavior is identical to today
(zero egress, no new prompts, no new subprocess calls).

---

## 3. User Context

Single-tenant research vault. Owner: saris. The vault is used for daily research
reading and query. `claude` CLI is on PATH. `ollama` is not installed (confirmed
in feature brief). Python 3.12 via `uv run`. No external API keys beyond claude.

---

## 4. JTBD

When I run a retrieval query against my wiki or graph, I want the top-K results
to be ordered by genuine semantic relevance to my query — not just keyword
overlap — without having to install ollama. Enabling `--rerank claude` should
give me semantically ordered results using the claude CLI I already have.

---

## 5. Users and Roles

| Role | Interaction |
|------|-------------|
| saris (vault owner) | opts in via `--rerank claude` flag or `RERANK_TIER=claude` env var when running retrieve or graph-retrieve |
| automated scripts / CI | never opt in; default path stays zero-egress |
| test suite | injects a fake `claude` via `--claude-cmd` to run offline |

---

## 6. Functional Requirements

| ID | Requirement |
|----|-------------|
| FR1 | `rerank.py` gains a `rerank_claude(query, candidates, top_k, claude_cmd)` function that calls `claude -p`, passes a structured prompt via stdin, and returns the candidates list reordered by the returned ranking |
| FR2 | The public `rerank(query, candidates, top_k, allow_remote, rerank_tier, claude_cmd)` function gains two new keyword args: `rerank_tier` (default `"auto"`) and `claude_cmd` (default `"claude"`) |
| FR3 | When `rerank_tier == "claude"`: skip the ollama cosine tier entirely; call `rerank_claude` on the BM25 candidate shortlist; fall through to BM25-only if the call fails |
| FR4 | When `rerank_tier == "auto"` (default): behavior is identical to today — try cosine, fall through to BM25-only; claude tier is NEVER called |
| FR5 | The claude prompt instructs the model to return a JSON array of candidate ids (the `chunk_id` values) in ranked order. The prompt provides ONLY the chunk_ids and their snippet text — no external content is added |
| FR6 | After receiving the response, `rerank_claude` validates: every returned id must be present in the input candidate set. Any id not in the input set is silently dropped. If the surviving ids cover fewer than the available candidates, the unranked candidates are appended in original BM25 order to fill top_k |
| FR7 | `retrieve.py` gains `--rerank claude` as a valid value for a new `--rerank-tier` flag (default `"auto"`); it also gains `--claude-cmd CMD` (default `"claude"`) and forwards both to `reranker.rerank()` |
| FR8 | `graph-retrieve.py` gains the same two flags (`--rerank-tier`, `--claude-cmd`) with the same defaults and forwarding behavior |
| FR9 | When `RERANK_TIER=claude` is set in the environment, both consumers treat it as `--rerank-tier claude` unless the flag overrides it |
| FR10 | `rerank.py --peek` output gains a `"rerank_tier"` field showing the active tier and the resolved `claude_cmd` path |
| FR11 | The existing `--no-rerank` flag in both consumers continues to short-circuit everything, including the claude tier |

---

## 7. Happy Path

```
# User has claude on PATH; opts in explicitly
$ uv run python scripts/retrieve.py "attention mechanism efficiency" \
    --rerank-tier claude --top 5

# rerank.py:
#   1. ollama check skipped (rerank_tier=claude bypasses it)
#   2. BM25 returns 20 candidates
#   3. claude -p called with prompt:
#        SYSTEM: reorder these chunk ids by relevance to the query.
#                return ONLY a JSON array of the exact chunk_id strings.
#                do NOT invent new ids. any id you return that is not in
#                the input list will be ignored.
#        USER:   query: "attention mechanism efficiency"
#                candidates:
#                  c-001abc:0 | "...transformer self-attention..."
#                  c-002def:3 | "...linear attention approximation..."
#                  ...
#   4. claude returns: ["c-002def:3", "c-001abc:0", ...]
#   5. rerank_claude validates ids against input set (all ok here)
#   6. returns reordered list with rerank_score = 1/(rank+1) and
#      rerank_source = "claude:<resolved_cmd>"
# Output JSON includes: "strategy": "bm25+rerank:claude:claude"
{
  "query": "attention mechanism efficiency",
  "strategy": "bm25+rerank:claude:claude",
  "top_k": 5,
  "candidates": [...]
}
```

---

## 8. Business Rules

| ID | Rule |
|----|------|
| BR1 | **Reorder only.** The claude tier may ONLY reorder the provided candidates by their chunk_id. It must never invent, add, or fabricate a candidate. The prompt explicitly instructs this. |
| BR2 | **Out-of-set id rejection.** Any id returned by claude that is not in the input candidate set is silently ignored. If all returned ids are out-of-set, fall back to BM25 input order (noop). |
| BR3 | **Opt-in, default off.** The claude tier is never called unless `rerank_tier="claude"` is explicitly passed (via flag or `RERANK_TIER` env var). The default `rerank_tier="auto"` never touches claude. This is the repo's zero-egress default posture. |
| BR4 | **No egress on default path.** With default flags, running retrieve.py or graph-retrieve.py must never spawn a claude subprocess. Tests must prove this with a fake claude that asserts it is never called. |
| BR5 | **`--claude-cmd` test injection.** Both consumers and `rerank.py` accept `--claude-cmd` to replace the `claude` binary for tests. The fake command must be a real executable that echoes a valid ranking response or asserts it was not called. |
| BR6 | **Graceful fallback on failure.** If the claude subprocess exits non-zero, times out (>60s), or returns unparseable JSON, `rerank_claude` logs a warning to stderr and returns candidates in BM25 input order with `rerank_source="claude-error"`. The caller never raises. |
| BR7 | **Determinism exemption.** The claude tier produces non-deterministic ordering. No test asserts a specific order from a real claude call. Deterministic tests cover: id validation (BR2), fallback ladder (BR3/BR4/BR6), and opt-in behavior (BR3/BR4) using a fake engine only. |
| BR8 | **Rerank score assignment.** Candidates returned by claude get `rerank_score = 1.0 / (rank + 1)` (rank is 0-indexed). Appended un-ranked candidates (BR6 fill) get `rerank_score = 0.0`. `rerank_source` is set to `"claude:<claude_cmd>"`. |
| BR9 | **Prompt grounding contract.** The prompt sent to claude includes only: the query string, and for each candidate: its chunk_id and its snippet (<=200 chars). No page paths, no absolute paths, no vault metadata. |
| BR10 | **`RERANK_TIER` env var.** `rerank.py` reads `os.environ.get("RERANK_TIER", "auto")` as the default for `rerank_tier`. The explicit kwarg always wins over the env var. |

---

## 9. Data and Integrations

```
  scripts/rerank.py          <-- MUTABLE: gains rerank_claude() + new kwargs
  scripts/retrieve.py        <-- MUTABLE: gains --rerank-tier + --claude-cmd flags
  scripts/graph-retrieve.py  <-- MUTABLE: gains --rerank-tier + --claude-cmd flags
  tests/test_retrieve.py     <-- MUTABLE: extend with claude-tier tests
                               (or add tests/test_rerank_claude.py — see Constraints)
  scripts/bm25-index.py      <-- READ-ONLY: no changes
  scripts/rerank.py cosine   <-- READ-ONLY: existing cosine path unchanged
```

`claude` subprocess is called via `subprocess.run([claude_cmd, "-p"], input=prompt, ...)`.
Prompt is passed via stdin (mirrors graph-propose.py's `_run_claude`).
Response is read from stdout and parsed as JSON.

No new files written. No new dependencies. No new vault-meta paths.

---

## 10. Errors and Edge Cases

| Case | Behavior |
|------|----------|
| `rerank_tier="claude"` but `claude` not on PATH | log warning to stderr: "claude not found: {claude_cmd}"; return BM25 order with `rerank_source="claude-not-found"` |
| `claude -p` times out (>60s) | catch `subprocess.TimeoutExpired`; return BM25 order with `rerank_source="claude-timeout"` |
| `claude -p` returns valid JSON but empty array `[]` | treated as "no ordering provided"; return all candidates in BM25 order |
| `claude -p` returns JSON array where ALL ids are out-of-set | return BM25 order (BR2); log how many ids were rejected |
| `claude -p` returns JSON array where SOME ids are out-of-set | drop bad ids; append unranked remainder in BM25 order to fill top_k |
| `claude -p` returns non-JSON response | log parse error; return BM25 order with `rerank_source="claude-parse-error"` |
| `rerank_tier="claude"` + `--no-rerank` | `--no-rerank` wins; claude not called |
| `rerank_tier="claude"` + `candidates=[]` | return `[]` immediately; no subprocess |
| `RERANK_TIER=claude` set + `--rerank-tier auto` flag passed | explicit flag wins; auto behavior, claude not called |
| Fake `claude` in test injected via `--claude-cmd` but default path invoked | fake `claude` must never be called; test asserts subprocess mock was not invoked |

---

## 11. Constraints

**Mutable surface (the only files this feature touches):**

```
scripts/rerank.py            -- add rerank_claude() + new kwargs to rerank()
scripts/retrieve.py          -- add --rerank-tier + --claude-cmd flags
scripts/graph-retrieve.py    -- add --rerank-tier + --claude-cmd flags
tests/test_retrieve.py       -- extend with new test functions
  (alternative: tests/test_rerank_claude.py — either is acceptable)
```

**Read-only (must not be modified):**

```
scripts/bm25-index.py        -- BM25 logic is settled; no changes allowed
scripts/rerank.py cosine     -- existing cosine + noop paths must be preserved exactly
scripts/graph-propose.py     -- inspiration only; not modified
```

**Behavior constraint:** `make test-retrieve` and `make test-graph` must both pass
green after the change. No regression to any existing test.

**No new packages.** `subprocess`, `json`, `os`, `shutil.which` are all stdlib.
The implementation must not add entries to `pyproject.toml` or `requirements*.txt`.

---

## 12. CAN / CANNOT

```
CAN
  - Add rerank_claude() to rerank.py
  - Add --rerank-tier and --claude-cmd flags to retrieve.py and graph-retrieve.py
  - Add RERANK_TIER env var support to rerank.py
  - Extend tests/test_retrieve.py (or add tests/test_rerank_claude.py)
  - Change strategy string in output JSON to reflect "claude" tier
  - Use subprocess.run to call claude (same pattern as graph-propose.py)

CANNOT
  - Call claude on the default/auto path (zero-egress guarantee)
  - Modify bm25-index.py
  - Break the existing cosine or noop code paths in rerank.py
  - Invent candidate ids (prompt constraint + validation)
  - Add new PyPI dependencies
  - Write to vault-meta or any wiki page
  - Modify graph-propose.py
```

---

## 13. Acceptance Criteria

| ID | Description | Verification command |
|----|-------------|----------------------|
| AC1 | Claude tier reorders candidates by returned index only; invented ids are dropped | `python3 tests/test_retrieve.py` (or `test_rerank_claude.py`) — test `test_claude_tier_reorders_by_given_ids`: fake claude returns a permuted order of real chunk_ids; assert output order matches permutation |
| AC2 | Out-of-set ids are silently ignored; remainder fills from BM25 order | `python3 tests/test_retrieve.py` — test `test_claude_tier_drops_invented_ids`: fake claude returns 2 real ids + 1 invented id; assert invented id absent from output; assert total count == min(top_k, input candidates) |
| AC3 | Default path (`--rerank-tier auto`) never calls claude | `python3 tests/test_retrieve.py` — test `test_default_path_no_claude_call`: monkeypatch subprocess.run to assert_never; call `rerank.rerank("q", candidates, rerank_tier="auto")`; assert subprocess.run not called |
| AC4 | `--rerank-tier claude` on retrieve.py CLI triggers claude path | `python3 tests/test_retrieve.py` — end-to-end sandbox test: fake claude exe that returns valid JSON ranking; run `retrieve.py "q" --rerank-tier claude --claude-cmd ./fake_claude`; assert `"strategy"` contains `"claude"` |
| AC5 | Cosine -> claude -> BM25 fallback ladder is correct | `python3 tests/test_retrieve.py` — test `test_fallback_ladder`: with `rerank_tier="auto"` + ollama unreachable: assert source is `"noop-no-ollama"` (not claude). With `rerank_tier="claude"` + fake claude: assert source is `"claude:..."`. With `rerank_tier="claude"` + claude fails: assert source is `"claude-error"` |
| AC6 | Mocked-engine tests stay fully offline (no real egress, no real claude) | `python3 tests/test_retrieve.py` runs to completion in <10s in CI with no network; verified by `RERANK_TIER=claude python3 tests/test_retrieve.py` with a fake `--claude-cmd` that writes to a tmp file if called — assert file was written exactly once per opted-in test, never in default-path tests |
| AC7 | `make test-retrieve` passes green | `make test-retrieve` exits 0 |
| AC8 | `make test-graph` passes green | `make test-graph` exits 0 |
| AC9 | `rerank.py --peek` shows `rerank_tier` field | `uv run python scripts/rerank.py --peek "test"` outputs JSON with `"rerank_tier": "auto"` (or `"claude"` if env var set) |
| AC10 | `--no-rerank` wins over `--rerank-tier claude` | `uv run python scripts/retrieve.py "q" --no-rerank --rerank-tier claude` with fake claude installed: assert fake claude never called; `"strategy": "bm25-only"` in output |

---

## 14. Metric

**Definition of done:** AC1 through AC10 all pass. `make test-retrieve` green.
`make test-graph` green. The two consumers each have their two new flags documented
in `--help` output. No existing tests regress.

**Stretch signal (not a gate):** Running `retrieve.py "attention mechanism"
--rerank-tier claude` on the live vault returns a top-5 list visibly reordered
relative to BM25 order — confirming the tier is active and the prompt works.

---

## 15. Out of Scope

- Cross-encoder rerankers (BGE, Cohere, Voyage) — mentioned in rerank.py's docstring as future paths; not touched here
- Caching claude responses — non-deterministic output makes caching dangerous; out of scope
- Streaming claude output — not needed for reranking (response is small)
- Retry loop on claude grounding failures — unlike graph-propose, reranking has no "hallucinated entity" risk (ids are validated); single call with fallback is sufficient
- `--claude-cmd` flag on `rerank.py` CLI (standalone mode) — the flag is only needed on the two consumers; standalone `rerank.py` is rarely called directly by users; add only if the implementer finds a clean reason to
- Changing the default `rerank_tier` to `"claude"` — zero-egress default is a hard repo posture; never flip this without an explicit feature
- Modifying `wiki-ingest` or `autoresearch` skills — they do not call retrieve.py directly; out of scope

---

## 16. Open Questions (resolved)

| Question | Decision |
|----------|----------|
| Where does the new tier live — shared `rerank.py` or per-consumer? | **Shared `rerank.py`.** Both consumers import it by module already. Adding the tier there keeps the fallback logic in one place. No reason found against this. |
| Should `rerank_tier="claude"` skip the ollama check entirely? | **Yes.** When the user explicitly selects the claude tier they are opting out of cosine; running an ollama liveness check first adds latency with no benefit. The chain is: if `rerank_tier="claude"` go straight to claude tier, not cosine first. |
| How is the ranking score assigned for the claude ordering? | **`1.0 / (rank + 1)` (0-indexed rank).** This matches "first returned = highest score" and gives a numeric value compatible with the existing `rerank_score` field that downstream dedup and sort logic uses. |
| What timeout for the claude subprocess? | **60 seconds**, matching the approximate bound used for graph-propose retries. Configurable only via the `--claude-cmd` fake-engine pattern in tests; not exposed as a user flag (premature). |
| Should `RERANK_TIER` env var apply to the `rerank()` function or only the consumers' CLI? | **Both.** `rerank.py` reads `os.environ.get("RERANK_TIER", "auto")` as the default for the `rerank_tier` kwarg. The consumers' `--rerank-tier` flag always wins over the env var when explicitly passed. This mirrors the `OLLAMA_URL` pattern already in the file. |
| Does `--rerank-tier` replace `--no-rerank`? | **No.** `--no-rerank` is a hard short-circuit that predates this feature and some callers (e.g., graph-propose's `_retrieve_passages_for_paper`) pass it explicitly. It continues to win over everything. `--rerank-tier` is additive. |
| Should the standalone `rerank.py` CLI gain `--rerank-tier` and `--claude-cmd` flags? | **Deferred.** The CLI entry point (`rerank.py main()`) is used for testing and debugging; most production calls go through retrieve.py/graph-retrieve.py. Adding the flags to `main()` is trivial and may be done by the implementer if it aids debugging, but it is not a gate for AC. |
