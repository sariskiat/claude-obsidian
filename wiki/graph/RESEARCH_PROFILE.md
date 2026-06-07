---
type: research-profile
updated: 2026-06-07
---

# Research Profile — Saris

## Goal

First-author paper at a top venue (NeurIPS, ICML, ICLR, or CVPR) strong enough to be a top-tier PhD ticket (Ivy or top industrial research lab).

Approximately 4 hours per week available. VTON and applied diffusion are the strength areas. Theory muscle is still being built.

## The bar

Three uncomfortable truths:

1. A top-tier paper is necessary but not sufficient, and not all of them count. Admissions committees can tell the difference between "competent engineer who applied a known method to a new dataset" and "person who had an idea." A paper must signal taste and independence.

2. To most theory-leaning committees, VTON is an application area. A pure try-on paper lands at CVPR/ICCV/WACV and reads as applied vision. The only way VTON helps the stated goal is if the contribution is a general insight — try-on is the testbed, not the point.

3. The letter matters as much as the paper. A top PhD admit usually rides on a recommender the committee trusts saying the ideas were yours.

## Swap test (hard rule — no application layer)

If a researcher who knows nothing about try-on could write the paper by swapping the dataset, it is application layer and will not earn the PhD ticket. The contribution must come from try-on's structural properties.

## Five directions (ranked by expected value)

Rank order: 1 > 3 > 2 > 5 > 4.

**Direction 1 — Selective/spatially-varying memorization for try-on (highest ceiling, highest risk)**
Garment region needs memorization regime; body and background need generation regime. Reframes memorization from "bug" to controllable resource. Fails the swap test cleanly. Requires a real theory co-author and more than 4h/week to have a real chance.

**Direction 2 — Rate-distortion ceiling on garment-detail transfer**
Formalize why three independent papers hit the same encoder bottleneck. Derive the ceiling, show where current encoders sit. Lower scoop risk. Risk: degrades to a benchmark paper without a defensible estimator and real lower bound.

**Direction 3 — Constrained-sampling theory under hard garment-pinning (most defensible, best risk-adjusted)**
Pins part of the output (garment) while sampling the rest. Generic sampler theory assumes free sampling. Characterize how a hard-constraint inpainting changes sampling dynamics and divergence. Aek's momentum machinery is the tool. Best chance of a clean, finishable, defensible result at 4h/week solo.

**Direction 4 — Video-VTON appearance vs motion objective tradeoff (skip)**
CVPR-level ceiling, weak admissions signal. Skip.

**Direction 5 — Memorization-as-a-feature framing (umbrella)**
Reframes memorization from failure to resource. Works as a wrapping framing but does not stand alone.

## Fork rule

- Theory co-author secured AND more than 4h/week: swing for Direction 1.
- Mostly solo at 4h/week: ship Direction 3.
- 4-week falsification probe (Direction 3 → Direction 1): GATE 1 (reproduce baseline by week 2 or pivot), GATE 2 (escalate/ship/fall-back).

## Aek collaboration — phase rule

In the graph, the VTON community has ZERO claim-edges to Aek's communities (diffusion sampling / noise optimization / distillation) — total white space. The existing hand-written bridge: "Extrapolation-Based Iterate Correction" linking Heavy Ball Momentum / GHVB (Aek) to Neon negative extrapolation.

**Phase rule: do NOT email Aek until all 4 Aek deep-reads are complete.**

Aek is a tool for Direction 3 — his momentum and boundary-locus machinery is the testbed, not the thesis.

## Strength summary

- Strong: VTON end-to-end pipelines, applied diffusion models, garment-conditioned generation.
- Building: diffusion theory (sampling dynamics, stability analysis, score functions).
- Gap: pure information-theoretic arguments, ergodic/stochastic analysis.
