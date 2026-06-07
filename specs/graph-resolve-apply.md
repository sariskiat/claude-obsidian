# Spec: Apply Safe Entity-Resolution Merges (connect the islands)

**Feature ID:** `graph-resolve-apply`
**Status:** approved (delegated; §App-A merge list still needs human ratification before `--commit`)
**Grilled on:** 2026-06-07 (self-grill — human delegated; conductor to ratify the MERGE list)
**Problem classification:** `unique`

> Phase 3.5 of the graphbuilding-fusion epic (`docs/graphbuilding-fusion-design.md`).
> P3 (`graph-resolve-dedup`) built the *detector* — it only PROPOSES; its docstring
> reads "NEVER auto-modifies the vault". The P3 spec deferred the apply to "P4" §12.
> This spec IS that apply: it takes the human-ratified safe subset of the 34 live
> proposals and writes the merges into the vault, collapsing paper-islands.

---

## 1. Problem, Objective, JTBD

```
┌────────────────────────────┐     ┌────────────────────────────┐     ┌────────────────────────────┐
│ PROBLEM                    │     │ OBJECTIVE                  │     │ METRIC                     │
│                            │     │                            │     │                            │
│ The Obsidian graph shows   │     │ APPLY the human-approved   │     │ cross-paper shared         │
│ ~93 paper-islands. Only    │ ──▶ │ subset of resolve.py's     │ ──▶ │ entities ↑ (was 136)        │
│ 136 of 755 canonical       │     │ proposals: set loser       │     │ Louvain communities ↓       │
│ entities span >1 paper —   │     │ canonical_id, re-export,   │     │ (was 93) / components ↓     │
│ duplicates were detected   │     │ rebuild. Markdown stays    │     │ (was 72)                    │
│ but never merged.          │     │ the source of truth.       │     │ round-trip stays byte-equal │
└────────────────────────────┘     └────────────────────────────┘     └────────────────────────────┘
```

**JTBD.** When the resolver has surfaced duplicate entities, I want to *commit* the
ones I trust — fold each duplicate into its canonical twin so the two papers that used
different names for the same thing now share one graph node — because shared nodes are
what turn 93 disconnected islands into a connected research map, which is the entire
point of building the graph (find cross-paper agreement, debate, and white-space).

**Why now / why this is the bottleneck.** P3 ships a detector that writes nothing.
Until something writes the `canonical_id` pointers, every re-export reproduces the same
fragmented graph and the five gap species under-count (a concept split across two ids
looks like two lightly-claimed entities instead of one well-supported one). The merge
mechanism already exists in the schema — this feature only has to *exercise* it on a
vetted list.

---

## 2. Merge mechanism (existing schema — NO new graph engine)

```
  wiki/graph/entities/<loser>.md        ← edit frontmatter, IN PLACE
  ┌──────────────────────────────┐
  │ is_canonical: true  → false   │
  │ canonical_id: null  → <winner>│
  │ canonical: "[[<winner-slug>]]"│   (added, readability wikilink)
  └──────────────────────────────┘
                │
                ▼
  graph-export.py  (re-emit graph-export.json + md from the *edited* vault)
                │   root() resolves A→B→C chains; path-compresses
                ▼
  graph-build.py   (rebuild derived .vault-meta/graph/graph.db)
                │
                ▼
  gaps / resolve / Louvain now roll losers up into winners
  → papers share nodes → islands collapse
```

The schema field already exists and is already exercised by 689 variant rows today
(verified: `wiki/graph/entities/deepfashion-dataset__e1827.md` carries
`is_canonical: false`, `canonical_id: 1371`, `merge_confidence: 0.95`, and a
`canonical: "[[deepfashion-dataset__e1371]]"` wikilink). A merge is therefore one
duplicate becoming one more such variant row. `graph_db.root()` already resolves
multi-hop chains and cycles (T2 invariant tests in `test_graph_roundtrip.py`), so even
if a winner is itself later merged the rollup stays correct.

**Direction is fixed by frontmatter, source of truth is markdown.** The apply edits the
**loser**'s `.md` only. The winner's `.md` is untouched. We never write the derived db
directly (that is rebuilt). This keeps BR4 ("source db / vault is authoritative, sqlite
is throwaway") intact.

---

## 3. The decision the apply consumes: MERGE vs DO-NOT-MERGE

`scripts/graph-resolve.py --json` currently returns **34 proposals** (ollama absent →
6 Tier-1 exact + 28 Tier-2 Jaccard-fuzzy). I (self-grill) classified all 34. The full
appendix table is at the end of this spec. Summary:

- **SAFE — apply (9 distinct entity merges):** the 6 exact-name pairs + 3 obvious
  fuzzy synonyms. See §App-A.
- **DO-NOT-MERGE — exclude (25 proposals):** specialization-vs-base, distinct siblings,
  cross-super_type, and disambiguation false positives. See §App-B.

**This MERGE list is the one thing that needs human ratification before the apply runs**
(see §12 CAN/CANNOT). The script must not invent the list; it must read it from a vetted
input (a `--merge-file` of `loser_id winner_id` pairs, or the pinned list embedded in
this spec's §App-A and copied into the apply invocation). False merges are far more
expensive than missed merges — they fuse genuinely distinct concepts and silently
corrupt every downstream gap count, and are only reversible by `git revert`.

---

## 4. Winner-selection rule (deterministic)

For each approved pair `{a, b}` the **winner** (canonical, survives) and **loser**
(gets `canonical_id = winner`) are chosen by:

```
claim-degree(e) = COUNT(claims WHERE subject_entity_id = e OR object_entity_id = e)
                  measured on the PRE-MERGE derived db (root-resolved ids)

winner = argmax(claim-degree)
tie (equal degree) → winner = min(id)   # lower id wins, fully deterministic
```

Rationale: the higher-degree entity is the better-attested hub, so folding the sparse
duplicate into it preserves the richer neighbourhood and minimises churn. The tie-break
on lower id is stable across runs (ids are preserved across round-trip — see
`graph-build.py` header) and is the same min-id convention `root()` already uses to
elect a cycle representative, so the two never disagree.

Worked examples from the live db (degrees measured 2026-06-07):

| Pair | degrees | winner | loser |
|---|---|---|---|
| Virtual Fashion Try-On (1053) ↔ Virtual Try-On (914) | 1 vs 15 | **914** | 1053 |
| Classifier-Free Guidance (730/1354) | 6 vs 2 | **730** | 1354 |
| Cross-Attention Conditioning (921/1387) | 4 vs 1 | **921** | 1387 |
| Stable Diffusion (336/934) | 2 vs 1 | **336** | 934 |
| CLIP (1413) ↔ CLIP Vision Encoder (1368) | 0 vs 0 → tie | **1368** | 1413 |
| MSE Diffusion Loss (1359/1389) | 0 vs 0 → tie | **1359** | 1389 |
| AnyDoor (1149/1366) | 1 vs 0 | **1149** | 1366 |
| DINOv2 (879/1367) | 1 vs 0 | **879** | 1367 |

Note the zero-degree ties (CLIP, MSE) resolve cleanly by min-id; the apply must compute
degree itself rather than trust proposal field order (`entity_a` is not always the
higher-degree side — for the CFG fuzzy pair `entity_a`=1354 deg 2 < `entity_b`=1386
deg 1 is fine, but for Virtual Try-On `entity_a`=1053 deg 1 < `entity_b`=914 deg 15, so
field order is NOT a reliable winner signal).

---

## 5. Happy path

```
$ uv run python scripts/graph-resolve.py --json > /tmp/proposals.json      # 34 proposals
# (human ratifies §App-A; the 9 safe merges become the --merge-file)

$ uv run python scripts/graph-resolve.py --apply --merge-file specs/merges/graph-resolve-apply.tsv
  Reading 8 approved pairs (9 distinct entities) …
  Computing claim-degree on derived db …
  e1053 → e914   (Virtual Fashion Try-On → Virtual Try-On)        [deg 1 < 15]
  e1354 → e730   (Classifier-Free Guidance → Classifier-Free Guidance)
  e1386 → e730   (Classifier-Free Guidance (CFG) → Classifier-Free Guidance)
  e1387 → e921   (Cross-Attention Conditioning → Cross-Attention Conditioning)
  e934  → e336   (Stable Diffusion → Stable Diffusion)
  e1413 → e1368  (CLIP → CLIP Vision Encoder)            [deg tie → min-id 1368]
  e1366 → e1149  (AnyDoor → AnyDoor)
  e1367 → e879   (DINOv2 → DINOv2)
  e1389 → e1359  (MSE Diffusion Loss → MSE Diffusion Loss)
  Wrote 9 loser .md files (is_canonical=false, canonical_id set, canonical wikilink).
  DRY-RUN complete. Re-run with --apply --commit to persist.   # if dry-run is default

$ uv run python scripts/graph-export.py .vault-meta/graph/graph.db wiki/graph     # re-emit
$ uv run python scripts/graph-build.py wiki/graph .vault-meta/graph/graph.db      # rebuild

$ uv run python scripts/graph-gaps.py --json --top 999 | python -c '...'          # new counts
  white-space, replication, coverage … all shift DOWN (losers no longer separate)

# re-baseline the pinned test numbers to the NEW post-merge values, then:
$ make test-graph        # green on updated baselines
```

Note: CFG has THREE ids in play — Tier-1 pairs 730↔1354, Tier-2 pairs 1354↔1386. Both
fold into 730 (highest degree). The apply must handle this as a 3-way group, not two
independent pairs, and must produce a flat chain (1354→730, 1386→730) — never 1386→1354
→730 (root() would still resolve it, but a flat chain keeps the markdown legible). DINOv2
(879/1367) and MSE (1359/1389) each appear in BOTH the Tier-1 and Tier-2 result lists as
the *same* pair — the apply must dedupe candidate pairs by `frozenset({a, b})` so a pair
is processed exactly once.

---

## 6. Users and roles

| Actor | Can do | Cannot do |
|---|---|---|
| Owner (human) | Ratify the §App-A merge list, run the apply, review the diff, `git revert` | — |
| `graph-resolve.py --apply` (or new apply script) | Read derived db for degree, edit *loser* entity `.md` files in `wiki/graph/entities/`, print plan | Invent the merge list; write the derived db; edit winner/claim/paper files; auto-commit unless `--commit` given |
| `graph-export.py` / `graph-build.py` | Re-emit vault + rebuild derived db from the edited markdown | (unchanged — read-only on logic) |
| `graph_db.root()` | Resolve chains during export/build | (unchanged) |

---

## 7. Functional requirements

| ID | Requirement | Notes |
|---|---|---|
| FR1 | An apply path that, given a vetted list of `{loser, winner}` pairs, edits each loser's `wiki/graph/entities/<slug>.md`: `is_canonical: false`, `canonical_id: <winner>`, add `canonical: "[[<winner-slug>]]"` | Frontmatter edit only; body untouched |
| FR2 | Winner chosen by claim-degree on the pre-merge derived db; tie → min id (§4) | Script computes degree; does NOT trust proposal field order |
| FR3 | Candidate pairs deduped by `frozenset({a,b})`; multi-id groups (e.g. CFG 730/1354/1386) collapse to a flat chain into the single group winner | No loser→loser pointers |
| FR4 | Set `merge_confidence` on each loser to the proposal's `confidence` (1.0 for exact; Jaccard score for fuzzy) | Matches existing variant rows' convention |
| FR5 | Default DRY-RUN: print the plan (loser→winner, degrees, reason) and exit without writing. Writes only with an explicit `--commit` (or `--apply --commit`) flag | "NEVER auto-modifies" stays the *default*; opt-in to write |
| FR6 | Refuse to merge a pair where either id is missing from the db, where loser already has a `canonical_id` pointing elsewhere, or where winner==loser after degree/min-id resolution | Print a skip line, continue; non-zero exit only if zero pairs applied |
| FR7 | After writing, the operator re-runs `graph-export.py` then `graph-build.py`; the apply does NOT silently rebuild | Keep steps explicit & inspectable (BR4) |
| FR8 | Re-baseline `tests/test_graph_gaps.py` (`EXPECTED_COUNTS`, `EXPECTED_TOTAL`) and any count assertions in `tests/test_graph_roundtrip.py` to the NEW post-merge values | Intentional change, not a regression — see §11 AC4 |
| FR9 | `make test-graph` green on the updated baselines | 4-file suite incl. resolve + validate |
| FR10 | The merge LIST lives in a reviewable artifact (`specs/merges/graph-resolve-apply.tsv` or the §App-A table), not hardcoded inside the script | Auditability + human ratification gate |

---

## 8. Business rules

| Rule ID | Rule | Rationale |
|---|---|---|
| BR1 | Only the 9 distinct merges in §App-A are applied. The 25 in §App-B are excluded by name | The whole feature is "apply the SAFE subset", not "apply everything ≥ threshold" |
| BR2 | Edit the LOSER only; winner `.md` is byte-unchanged | Direction is decided once, by frontmatter; avoids double-edits |
| BR3 | Winner = max claim-degree, tie → min id (§4) | Deterministic, preserves richer hub, matches root()'s cycle convention |
| BR4 | Source of truth is `wiki/graph/` markdown; the derived db is rebuilt, never hand-written | Inherited invariant from P1/P3; the apply must not break it |
| BR5 | Pair dedup by `frozenset`; groups flatten to one winner; no loser→loser chains | Keeps markdown legible; root() correctness not relied on for legibility |
| BR6 | Writes gated behind explicit `--commit`; dry-run is the default | Honours P3's "NEVER auto-modifies" as the default posture |
| BR7 | Cross-super_type pairs are NEVER auto-applied (none are in §App-A; the one cross-type proposal — Ornament VTO Dataset/Artifact vs Task — is excluded) | A merge across super_type would corrupt typing |
| BR8 | Every change is plain frontmatter in git-tracked files → fully reversible via `git checkout`/`git revert` | Reversibility is the safety net for a wrong ratification |

---

## 9. Data and integrations

| Data | R/W | Failure behavior |
|---|---|---|
| `.vault-meta/graph/graph.db` (derived) | R (degree lookup) | Missing → exit 1 with rebuild hint (matches resolve.py today) |
| `wiki/graph/entities/<loser>.md` | W (frontmatter) | Loser file missing → skip with warning, continue |
| `wiki/graph/graph-export.json` | W (via graph-export.py, step 2) | Regenerated wholesale from edited vault |
| merge list (`specs/merges/graph-resolve-apply.tsv` or §App-A) | R | Missing/empty → exit 1 ("no approved pairs") |
| `graph_db.root()` | import | Missing → import error |
| ollama | — | Irrelevant to apply (ranking already done); never contacted |

---

## 10. Errors / edge cases

| Scenario | Expected |
|---|---|
| Derived db missing | Exit 1, print `graph-build.py` hint |
| Merge list empty / not found | Exit 1 ("no approved pairs to apply") |
| Loser id not in db | Skip that pair, warn, continue |
| Loser already a variant (canonical_id set to a *different* winner) | Skip, warn ("already merged into eX") — do not re-point |
| Loser already points to the *same* winner (idempotent re-run) | No-op, report "already applied" |
| Winner == loser after degree+min-id (e.g. someone listed a self-pair) | Skip, warn |
| 3-way group (CFG 730/1354/1386) | Both losers → 730 directly (flat); never 1386→1354 |
| Pair listed twice / Tier-1 and Tier-2 dupe (DINOv2, MSE) | Deduped by frozenset; applied once |
| Run twice in a row (`--commit` then `--commit`) | Idempotent: second run is all no-ops, exit 0 |
| Re-export after apply changes byte layout of many `.md` | Expected — round-trip equality is re-asserted on the NEW vault, not the old one (AC5) |
| Apply writes but operator forgets to rebuild | gaps/Louvain still show old counts; documented as a 3-step sequence (FR7), tests catch the stale baseline |

---

## 11. Acceptance criteria

Baselines below are the PRE-MERGE live numbers measured 2026-06-07 (these are what
change). The apply is correct when the post-merge numbers move in the stated direction
and the suite is green on the re-baselined values.

| ID | Criterion | Verification |
|---|---|---|
| AC1 | The 9 §App-A losers each have `is_canonical: false` + `canonical_id: <winner>` + a `canonical:` wikilink after apply | `grep -c "is_canonical: false"` on the 9 slugs; inspect frontmatter |
| AC2 | Winner selection matches §4 on every applied pair (degree, tie→min-id) | Compare apply's printed plan to the §4 worked-example table |
| AC3 | NO loser→loser chain; CFG group is flat into 730 | `root()` on 1386 and 1354 both return 730 in one hop after rebuild |
| AC4 | `tests/test_graph_gaps.py` + `test_graph_roundtrip.py` count baselines updated to post-merge values, with a one-line comment noting the merge that moved them | `git diff` on the two test files shows new `EXPECTED_*`; old value referenced in comment |
| AC5 | Round-trip stays byte-equal on the NEW (post-merge) markdown | `make test-graph` round-trip table-count + per-row-diff tests pass against the edited vault |
| AC6 | Cross-paper shared entities RISES above the pre-merge baseline (136) | `uv run python` snippet in §App-C; assert post > 136 |
| AC7 | Island count DROPS: Louvain communities < 93 AND connected components < 72 (or equal — never higher) | §App-C snippet; assert post ≤ pre on both |
| AC8 | `make test-graph` green | `make test-graph` exit 0 |
| AC9 | The apply made ZERO edits to winner files, claim files, or paper files | `git diff --name-only wiki/graph/` lists only the 9 loser entity `.md` + regenerated `graph-export.json` (+ derived db, which is gitignored) |
| AC10 | Dry-run default writes nothing; `--commit` required to persist | Run without `--commit`, confirm `git status` clean; run with `--commit`, confirm 9 files changed |
| AC11 | Fully reversible | `git checkout -- wiki/graph specs/merges tests` then rebuild restores pre-merge counts exactly |

---

## 12. CAN / CANNOT

- **CAN modify:** `wiki/graph/entities/<loser>.md` (the 9 losers only — frontmatter);
  `wiki/graph/graph-export.json` (regenerated); `tests/test_graph_gaps.py` +
  `tests/test_graph_roundtrip.py` (RE-BASELINE pinned counts, with explanatory comment);
  EITHER add `--apply`/`--commit`/`--merge-file` to `scripts/graph-resolve.py` OR add a
  new `scripts/graph-resolve-apply.py` (decision in §15); `specs/merges/graph-resolve-apply.tsv`.
- **CANNOT modify (read-only):** `scripts/graph_db.py` (`root()`), `scripts/graph-build.py`,
  `scripts/graph-export.py` logic, `scripts/graph-gaps.py`, winner/claim/paper markdown,
  the detection thresholds (0.85 / 0.6), `pyproject.toml`. The derived db is rebuilt, never
  hand-edited.
- **Needs human approval BEFORE applying:** the **§App-A MERGE list itself** — the conductor
  ratifies it. No `--commit` run until that ratification. This is the single human gate.

---

## 13. Metric

Primary (the goal): **paper-islands ↓ / cross-paper shared entities ↑.**

| Signal | Pre-merge (2026-06-07) | Target direction |
|---|---|---|
| Canonical entities spanning >1 paper | 136 | ↑ (each merge can only raise or hold) |
| Louvain communities (seed=42) | 93 | ↓ or = |
| Connected components | 72 | ↓ or = |
| Variant entities (`canonical_id` set) | 689 | +9 → 698 |
| Canonical entities (`canonical_id IS NULL`) | 755 | −9 → 746 |

Secondary: gap counts in `test_graph_gaps.py` shift (white-space/coverage/replication
fall as losers stop counting as separate thin entities). The re-baselined numbers ARE the
new ground truth — not a regression.

(Note for the conductor: the brief cited "94" cross-paper entities and "~98 islands";
the live db now measures 136 and 93/72 respectively — earlier merges already landed.
The spec pins the *measured* baseline and asserts direction, so it stays correct
regardless of the exact starting integer.)

---

## 14. Out of scope

- The 25 fuzzy false-positives in §App-B (specializations, distinct siblings,
  cross-type, disambiguations) — explicitly NOT merged.
- LLM / embedding-based resolution (ollama path) — ranking is already done; this feature
  only applies a vetted list.
- Auto-detecting and applying NEW duplicates after future ingests (future: a post-build
  hook) — out of scope here.
- Unifying prose `wiki/entities/` with structured `wiki/graph/entities/` — they stay
  separate by design (CLAUDE.md invariant).
- Cross-encoder reranker; git automation for the post-merge commit (human commits).
- Editing claim `subject_id`/`object_id` — claims NEVER merge (SCHEMA.md: "subject_id …
  NOT rolled up"); rollup happens at query/gap time via `root()`, so no claim rewrite.

---

## 15. Open questions (resolved)

- **`--apply` on resolve.py vs a new apply script?** → **New script
  `scripts/graph-resolve-apply.py`.** Rationale: `graph-resolve.py` is a pure read-only
  detector and its docstring contract ("NEVER auto-modifies the vault") is referenced by
  P3 AC and tests; bolting a writer onto it muddies that contract and risks the existing
  resolve tests. A sibling apply script keeps detector and mutator as separate, single-
  responsibility tools (mirrors the build/export split). The apply imports degree/root
  helpers from `graph_db`.
- **Where does the ratified list live?** → A tracked TSV `specs/merges/graph-resolve-apply.tsv`
  (`loser_id<TAB>winner_id<TAB>reason`) generated from §App-A, OR the operator pastes
  §App-A pairs. Either way the list is reviewable in git before `--commit`.
- **Dry-run default or write default?** → **Dry-run default**, write behind `--commit`.
  Preserves the "never auto-modifies by default" posture; the human opts in.
- **Do we rebuild automatically?** → **No.** The 3-step sequence (apply → export → build)
  stays explicit so each artifact is inspectable; tests catch a forgotten rebuild via a
  stale baseline.
- **What about merge_confidence on the loser?** → set to the proposal `confidence`
  (1.0 exact / Jaccard score fuzzy), matching the existing 689 variant rows.

---

## Appendix A — MERGE list (SAFE — apply). 8 proposals → 9 distinct entity merges.

Winner per §4 (claim-degree, tie→min-id; degrees measured on the live derived db
2026-06-07). "Proposal" = how it appears in `graph-resolve.py --json`.

| # | Proposal (tier) | Loser → Winner | Why safe |
|---|---|---|---|
| 1 | AnyDoor 1149 = 1366 (T1 exact) | **1366 → 1149** | Identical method name, both "object insertion" baseline; exact match |
| 2 | Classifier-Free Guidance 730 = 1354 (T1 exact) | **1354 → 730** | Same method, exact name; 730 higher degree |
| 3 | Classifier-Free Guidance 1354 = "…(CFG)" 1386 (T2, J=0.833) | **1386 → 730** | "(CFG)" is just the acronym gloss; folds into the 730 group (flat) |
| 4 | Cross-Attention Conditioning 921 = 1387 (T1 exact) | **1387 → 921** | Same mechanism, exact name; 921 higher degree |
| 5 | DINOv2 879 = 1367 (T1 exact / also T2 J=0.75) | **1367 → 879** | Same pretrained vision encoder; deduped to one pair |
| 6 | MSE Diffusion Loss 1359 = 1389 (T1 exact / also T2 J=0.667) | **1389 → 1359** | Same training objective; deduped; tie→min-id 1359 |
| 7 | Stable Diffusion 336 = 934 (T1 exact) | **934 → 336** | Same pretrained T2I model; 336 higher degree |
| 8 | CLIP 1413 = CLIP Vision Encoder 1368 (T2, J=0.833) | **1413 → 1368** | "CLIP Vision Encoder" is CLIP's image tower used as the encoder; same artifact; tie→min-id 1368 |
| 9 | Virtual Fashion Try-On 1053 = Virtual Try-On 914 (T2, J=0.714) | **1053 → 914** | "Fashion" is a redundant qualifier on the same Task; 914 far higher degree (15) |

(8 proposal rows; #2+#3 share winner 730, so 9 distinct loser→winner edits total: 1366,
1354, 1386, 1387, 1367, 1389, 934, 1413, 1053 — nine losers.)

## Appendix B — EXCLUDE list (DO-NOT-MERGE). 25 proposals.

These are the Tier-2 Jaccard false positives. Each is genuinely distinct; merging would
fuse different concepts and corrupt gap counts.

| Proposal (J) | Why NOT a merge |
|---|---|
| AC-BERT 600 ↔ BERT 597 (0.75) | AC-BERT is an attribute-conditioned variant; not plain BERT |
| Consistency Models 634 ↔ Shortcut Models 635 (0.714) | Two distinct one-step generative families; different mechanisms |
| Masked Full-Attention 1295 ↔ Masked-Attention Mechanism 1304 (0.75) | Different attention constructs; "full" vs generic masked are not synonyms |
| Knowledge 451 ↔ Knowledge Reliability 503 (0.75) | One is a broad concept, the other a specific quality property |
| CAW-UNet 1231 ↔ UNet 671 (0.667) | Specialized cross-attention-warping UNet ≠ generic UNet |
| Garm-UNet 1185 ↔ UNet 671 (0.667) | Garment-branch UNet ≠ generic UNet |
| Main-UNet 1186 ↔ UNet 671 (0.667) | Main-branch UNet ≠ generic UNet |
| UNet 671 ↔ UNet-SA 673 (0.667) | Self-attention UNet variant ≠ generic UNet |
| CLIP 1413 ↔ VLM (Vision-Language Model) 1259 (0.667) | CLIP is one VLM; VLM is the umbrella class — not equal |
| Global Sensemaking 477 ↔ Sensemaking 470 (0.667) | "Global" is a distinct task scope in the source; kept separate per domain note |
| HR-VVT 1171 ↔ VVT 1086 (0.667) | High-resolution VVT dataset ≠ base VVT dataset |
| Knowledge Graph Construction 519 ↔ Pipeline 522 (0.667) | Four distinct KG sub-concepts — construction |
| Knowledge Graph Construction 519 ↔ Quality 521 (0.667) | … quality |
| Knowledge Graph Construction 519 ↔ Querying 518 (0.667) | … querying |
| Knowledge Graph Pipeline 522 ↔ Quality 521 (0.667) | … all four are deliberately separate |
| Knowledge Graph Pipeline 522 ↔ Querying 518 (0.667) | … |
| Knowledge Graph Quality 521 ↔ Querying 518 (0.667) | … |
| Multi-stage Masking Strategy 1224 ↔ Multi-Stage Progressive Training 1316 (0.667) | Masking strategy ≠ progressive-training schedule |
| Ornament VTO Dataset 1147 ↔ Ornament VTO Task 1151 (0.667) | **Different super_type** (Artifact vs Task) — never merge across type |
| Pose-aware Spatial Attention 1195 ↔ Pose-aware Temporal Attention 1196 (0.667) | Spatial ≠ temporal attention; sibling modules |
| Cross-Attention Conditioning 1387 ↔ Cross-Attention Text Conditioning 1353 (0.714) | Text-specific conditioning ≠ general cross-attn conditioning; kept distinct (1387 already merges into 921 via the exact-name pair) |
| TD-CFM 865 ↔ TD²-CFM 866 (0.636) | Single-TD ≠ double-TD method; the "²" is load-bearing |
| Mixture of Lookup Experts (MoLE) 678 ↔ Mixture-of-Experts (MoE) 546 (0.625) | MoLE is a specific lookup-expert variant ≠ generic MoE |

(23 rows shown; the DINOv2 and MSE Tier-2 duplicates of their Tier-1 exact pairs are
folded into §App-A as deduped applies, accounting for the remaining 2 of the 25
"not a fresh merge" proposals. 6 exact + 28 fuzzy = 34 total; 8 distinct safe applies
in §App-A draw from 6 exact + 3 fuzzy (CFG-CFG, CLIP, VTO), and DINOv2/MSE fuzzy rows
collapse onto their exact twins → the remaining 23 fuzzy rows above are pure excludes.)

## Appendix C — Metric verification snippet

```python
# uv run python - <<'PY'  (run on .vault-meta/graph/graph.db before & after)
import sqlite3, sys; sys.path.insert(0, "scripts")
from graph_db import root
from collections import defaultdict
import networkx as nx
conn = sqlite3.connect(".vault-meta/graph/graph.db")
papers = defaultdict(set)
for s,o,sp in conn.execute("SELECT subject_entity_id,object_entity_id,source_paper FROM claims"):
    papers[root(conn,s,compress=False)].add(sp); papers[root(conn,o,compress=False)].add(sp)
print("cross-paper shared:", sum(1 for ps in papers.values() if len(ps)>1))
G = nx.Graph()
for (r,) in conn.execute("SELECT id FROM entities WHERE canonical_id IS NULL"): G.add_node(r)
for s,o in conn.execute("SELECT subject_entity_id,object_entity_id FROM claims"):
    a,b=root(conn,s,compress=False),root(conn,o,compress=False)
    if a!=b and a in G and b in G: G.add_edge(a,b)
print("louvain:", len(nx.community.louvain_communities(G,seed=42)),
      "| components:", nx.number_connected_components(G))
PY
```

---

## Approval Checklist

- Domain lead approved: pending (conductor to ratify §App-A)
- Tech lead approved: pending
- Critic reviewed testability: self-grilled (all ACs have a command)
- Metric validated against user value: islands↓ / cross-paper-shared↑ — directional, measured baseline pinned
