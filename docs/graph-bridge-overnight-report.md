# Proposal Finder (graph-bridge) — Overnight Build Report

**For:** saris
**Date:** 2026-06-07 (overnight, autonomous)
**Branch:** `feature/graph-bridge` (off `feature/graph-full-paper-retrieval`)
**State:** built, stress-tested, Evaluator PASS. **Local only — nothing pushed. Awaiting your ratification.**

## How I read your instruction

You said "keep stress testing the plugin 100 times until it's able to do this, **skip grill**," then pasted the proposal-finder brief (which itself says "grill me / no code until I approve"). Since you were asleep, those can't both hold, so I read the intent as: don't interview you, draft the spec myself from the handoff, build it, stress-test until it works, keep everything local and reversible for your review. That's what I did. If you'd rather I had only drafted the spec and waited, say so and I'll treat the branch as throwaway.

## What it does

`/graph bridge` turns the graph's 312 white-space gaps (community pairs with zero claim-edges) into a ranked, grounded shortlist of next-paper proposals: "connect community A to community B because…, anchored in papers P, Q."

- **Substrate:** reconstructs the same seed=42 Louvain clustering `graph-gaps` uses, enumerates community pairs with zero claim-edges.
- **Ranking — transparent, not a black box.** Every proposal carries a `signal_breakdown`: `gap_confidence`, `bridgeability` (latent path via `entity_edges` — `citation_links` is empty so it contributes nothing and degrades cleanly), `limitation_pull` (limitation/open-question claims each side), `richness`, and `direction_relevance` (a boost for pairs touching your priority areas: virtual-try-on and diffusion-sampling/noise-optimization/distillation). Weights are CLI-tunable — retune them and the ranking changes, nothing is hidden.
- **Grounding:** anchor entities (highest-degree per side), anchor papers, and full-text passages from each side via the `graph-retrieve` you now have.
- **Synthesis:** `--synthesize` writes a narrative justification via claude-CLI. **Off by default — zero egress.** (Note: `claude` is on your PATH, so `--synthesize` would use it.)

## The gold anchor works

The VTON↔Aek bridge you care about ranks **#1**: `DiffSDA ↔ CatVTON`, `direction_relevance` 1.0. Your hand-written "Extrapolation-Based Iterate Correction" entity is detected and flagged `already_proposed` (not dropped). This was the acceptance anchor, verified by entity membership, not a hardcoded community number.

## Verification

- **Evaluator: PASS** — 8/8 acceptance criteria by real execution; `make test-graph` 44/4, `make test-fulltext` 44, bridge suite 26.
- **My independent stress: 740/740 invariant checks** — valid schema, score in [0,1], distinct communities, grounding integrity (zero fabricated entities/papers), determinism across 5 weight configs, anchor-level zero-claim-edge. Harness: `scripts/fulltext-tools/bridge_stress.py`.
- Gold anchor #1 (membership-checked), deterministic across runs, missing-db → exit 1, default path proven egress-free (the Evaluator put a sabotaged `claude` shim on PATH — it never fired).

## Flagged for your ratification (built conservative defaults)

1. **Ranking weights** — documented defaults, CLI-tunable. This is the main thing to tune to your taste.
2. **Unit = community-pair** (matches the worked example). Entity-pair could be a future mode.
3. **Synthesis off by default.** Decide if you want LLM narratives at all, and via which tier.

## Reproduce

```bash
uv run python scripts/graph-bridge.py --top 10              # ranked proposals (JSON)
uv run python scripts/graph-bridge.py --top 10 --synthesize # + narrative (opt-in, claude-CLI)
uv run python /tmp/.. -> scripts/fulltext-tools/bridge_stress.py   # 740-check stress
make test-graph && make test-fulltext && make test-bridge
```

## To finish (your call)

- Review the 5 commits on `feature/graph-bridge`, ratify or redirect, then push when ready (nothing pushed; the earlier `fork` block taught me to keep it local).
- Tune the ranking weights to your judgment of a "good" bridge.
- This sits on top of `feature/graph-full-paper-retrieval`, which is also unpushed — both branches are waiting for you.
