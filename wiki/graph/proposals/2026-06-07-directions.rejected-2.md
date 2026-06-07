# Directions Report — REJECTED (grounding cap exhausted)

**Unverified citations:** ton-textured-3d-guided-consistent-video-try-on-via-diffusion-models

---

## The bar

Three truths first, because picking the wrong direction here costs you a year you do not have at 4 hours a week.

**Truth 1 — a top venue paper is necessary, not sufficient, and not all of them count.** Committees at the labs you are aiming at can tell a competent engineer who moved a known method to a new dataset apart from a person who had an idea. A clean Virtual Try-On benchmark win reads as the first kind. It confirms you can ship. It does not buy a PhD ticket. The contribution has to signal taste and independence or it is wasted motion.

**Truth 2 — to a theory-leaning committee, Virtual Try-On is an application area.** A pure try-on paper lands at CVPR/ICCV/WACV and reads as applied vision. The only path where try-on serves the stated goal is when the contribution is a general claim about diffusion, and try-on is the testbed that happens to expose it. Apply the swap test ruthlessly: if a researcher who has never heard of try-on could write your paper by swapping the dataset, it is application layer and it is dead on arrival for the goal.

**Truth 3 — the letter carries as much weight as the paper.** A top admit usually rides on a recommender the committee already trusts saying the ideas were yours. That is the real reason a theory co-author matters. Not the extra hands. The letter. This is also why the Aek phase rule exists: you do not get a strong letter by emailing too early with half a thought. Finish the four deep-reads first.

Now the menu.

## Decision matrix

Ratings 1–5, higher is better for the goal. Scoop risk of 5 means a powerhouse lab can and might do it first.

| # | Direction | Ceiling | Clean-exec odds @4h/wk solo | Theory load | Scoop risk | Builds on strength | Admissions signal |
|---|-----------|:------:|:------:|:------:|:------:|:------:|:------:|
| 1 | Selective / spatially varying memorization for try-on | 5 | 2 | 5 | 5 | 4 | 5 |
| 2 | Rate–distortion ceiling on garment-detail transfer | 4 | 3 | 4 | 3 | 5 | 4 |
| 3 | Constrained-sampling theory under hard garment pinning | 4 | 3 | 4 | 2 | 5 | 4 |
| 4 | Video try-on appearance versus motion objective tradeoff | 3 | 4 | 2 | 4 | 4 | 2 |
| 5 | Memorization as a feature (umbrella framing) | 4 | 2 | 4 | 4 | 3 | 4 |

### 1. Selective / spatially varying memorization for try-on

The thesis is the one genuinely PhD-shaped idea on this list. Try-on demands two regimes inside one image: the garment region must be copied close to verbatim (memorization), while body and background must be synthesized (creativity). `an-analytic-theory-of-creativity-in-convolutional-diffusion-models` (Kamb and Ganguli) gives you the analytic handle — the balance between copying and creating is governed by locality and equivariance, and the Equivariant Local Score (ELS) Machine is an interpretable, predictive object you can actually compute against. `why-diffusion-models-don-t-memorize-the-role-of-implicit-dynamical-regularization-in-training` supplies the other half of the dial. Reframing memorization from "bug to suppress" to "spatially controllable resource" passes the swap test cleanly: the structure comes from try-on having a hard copy region next to a free synthesis region, which generic image generation does not. Bridges 4, 8, and 10 in the dossier all anchor on the creativity theory crossing into the try-on and compression communities, so the graph agrees this is where the white space is.

**Takedown:** This is a five-ceiling idea with a two on clean execution, and the gap between those numbers is exactly your problem. The creativity theory is a convolutional-score result; pushing it to a conditioned latent try-on model is real theory work — equivariance arguments, locality scale estimation, the works — and your own profile lists ergodic and information-theoretic argument as the gap, not the strength. Scoop risk is a flat 5: Ganguli's group, or any well-staffed lab that read the same paper, can do this faster than you can solo. At 4 hours a week with no theory co-author this is not a project, it is a hobby that gets scooped in month three. The honest read: this is only live if you secure a theory co-author and break past 4h/week. Otherwise it is the trap — the most exciting idea on the page and the one most likely to leave you with nothing.

### 2. Rate–distortion ceiling on garment-detail transfer

Three independent try-on papers hit the same wall, and the dossier names it for you. `art-viton-measurement-guided-latent-diffusion-for-artifact-free-virtual-try-on` flags that DINOv2 and CLIP encoders are not optimized for detail preservation. Bridge 7's passage states it outright: "CLIP-based textual inversion fails to preserve fine garment details due to CLIP encoder bottleneck." `texture-preserving-diffusion-models-for-high-fidelity-virtual-try-on`, `difffit-disentangled-garment-warping-and-texture-refinement-for-virtual-try-on`, and `incorporating-visual-correspondence-into-diffusion-model-for-virtual-try-on` are all fighting the same loss of high-frequency garment information through a conditioning encoder. The move is to formalize it: pose garment-detail transfer as a rate–distortion problem, derive where the achievable ceiling sits, and show empirically how far current encoders are from it. That is a general claim (conditioning channels have a capacity; here is how to measure it) with try-on as the clean instance.

**Takedown:** This degrades into a benchmark paper the moment your lower bound is hand-wavy. A rate–distortion "ceiling" with no defensible estimator and no real lower bound is just a plot of FID versus encoder size with a Greek letter taped on, and a committee will read it as exactly that. The hard part — a genuine information-theoretic lower bound on recoverable garment detail through a fixed encoder — sits squarely in your stated gap area. Scoop risk is the highest on the board (5): the bottleneck is common knowledge, so whoever produces the first rigorous bound wins, and a stronger theory group will produce it faster. Medium clean-exec odds only because the empirical half is squarely in your wheelhouse — but the empirical half alone is the benchmark paper you are trying not to write.

### 3. Constrained-sampling theory under hard garment pinning

This is the best risk-adjusted bet and the dossier's strongest cluster of bridges. Try-on with inpainting pins part of the output — the garment, or the unmasked person region — exactly, while sampling the rest. Generic sampler theory assumes free sampling; it does not describe what a hard pin does to the trajectory. `diffusion-sampling-with-momentum-for-mitigating-divergence-artifacts` and Heavy Ball Momentum are the analysis tools (this is Aek's machinery — his momentum and boundary-locus work), and the bridge passage hands you the open question directly: "Countless unexplored ways exist to expand numerical method stability regions beyond HB and GHVB." `adjoint-schr-dinger-bridge-sampler` and the GLASS Flows result give you the constrained-transport framing, and `catvton-concatenation-is-all-you-need-for-virtual-try-on-with-diffusion-models` plus `art-viton-measurement-guided-latent-diffusion-for-artifact-free-virtual-try-on` are the concrete pinned-sampling try-on systems to test against. Bridges 1, 3, 5, and 6 all light up here at direction_relevance 1.00. The claim — "a hard inpainting constraint changes sampling divergence in this characterizable way" — is general, finishable, and provable at small scale.

**Takedown:** The ceiling is a 4, not a 5, and you should make peace with that now: this earns the ticket by being clean and defensible, not by being spectacular. The real risk is scope drift — "characterize constrained sampling dynamics" can balloon into a six-month theory project if you do not nail the toy model in week one. The Aek dependency is also a discipline test: the phase rule says no email until all four deep-reads are done, and using his momentum machinery as a testbed does not mean leaning on him as a crutch for the thesis. The other hazard is the swap test sneaking up on you — if your final result is just "momentum samplers behave under hard constraints," that is sampler theory with no try-on content and it fails the test from the other direction. The garment pin has to be load-bearing in the proof, not decoration.

### 4. Video try-on appearance versus motion objective tradeoff

The dossier surfaces this (Bridge 2 and Bridge 12, both direction_relevance 1.00): `3dv-ton-textured-3d-guided-consistent-video-try-on-via-diffusion-models [UNVERIFIED — not in graph.db]`, ChronoTailor via `chronotailor-harnessing-attention-guidance-for-fine-grained-video-virtual-try-on`, and DPIDM via `dpidm-temporal-consistent-video` all wrestle a frame-detail versus temporal-coherence tension. There is a real objective tradeoff to characterize.

**Takedown:** Skip it. Ceiling is a 3 and admissions signal is a 2 — this is the most "applied vision" item on the page and it screams CVPR-competent-engineer, which is precisely the read you are trying to escape per Truth 1. The theory load to make it general is high, the clean-exec odds collapse because video adds compute and data cost you cannot absorb at 4h/week, and even a clean win lands as "better video try-on." It does not move the goal. Do not spend a single one of your weekly hours here until something above it has died.

### 5. Memorization as a feature (umbrella framing)

This is the wrapper, not a standalone. `neon-negative-extrapolation-from-self-training-improves-image-generation`, `an-analytic-theory-of-creativity-in-convolutional-diffusion-models`, and `why-diffusion-models-don-t-memorize-the-role-of-implicit-dynamical-regularization-in-training` together support a reframing — memorization is a resource to be allocated, not a failure to be suppressed. Bridge 4 connects Neon's regime directly to the creativity theory.

**Takedown:** It does not stand alone and you should not try to make it. As a paper, "memorization is actually a feature" with no concrete mechanism is a position piece, and position pieces from a first-author with a building theory muscle do not get into NeurIPS. Its only legitimate use is as the framing that gives Direction 1 (or a constrained-sampling result that touches the copy region) a bigger story. Treat it as the abstract's first paragraph, never as the contribution.

## Ranking

1. **Direction 1 (selective memorization)** — highest ceiling and best admissions signal, but only reachable with a real theory co-author and more than 4h/week.
2. **Direction 3 (constrained sampling under garment pinning)** — best risk-adjusted bet; the one result a solo author at 4h/week can actually finish and defend.
3. **Direction 2 (rate–distortion ceiling)** — strong if you land a defensible bound, but the bound sits in your weakest area and scoop risk is maximal.
4. **Direction 5 (memorization as a feature)** — useful framing, not a paper; fold it into 1 or 3.
5. **Direction 4 (video try-on tradeoff)** — skip; CVPR-competent ceiling, weak signal for the stated goal.

**Fork rule.** Theory co-author secured AND more than 4h/week → swing for Direction 1. Mostly solo at 4h/week → ship Direction 3. Run the 4-week falsification probe from Direction 3 toward Direction 1: GATE 1, reproduce the baseline by week 2 or pivot; GATE 2, escalate to Direction 1, ship Direction 3, or fall back.

## Execution

First-week probe targets the falsification path, starting where a solo author can actually get traction.

```
 WEEK 1        WEEK 2 (GATE 1)         WEEK 3-4 (GATE 2)
 reproduce  →  toy model green?  →  yes: probe Direction 1 copy-region claim
 baseline      no: pivot now          no: ship Direction 3 as the paper
```

- **Paper to reproduce (week 1):** `diffusion-sampling-with-momentum-for-mitigating-divergence-artifacts`. Stand up the Heavy Ball / GHVB momentum sampler and confirm you reproduce its divergence-artifact mitigation on a small unconditional model. This is Aek's machinery and the tool for Direction 3 — but the phase rule holds: do not email him, just reproduce from the paper.
- **Toy model to build (weeks 1–2):** a 1-D or small 2-D diffusion where you can impose a *hard pin* on part of the state (fix coordinates to a target, sample the rest) and measure the trajectory's divergence behavior against the free-sampling baseline. The single question: does the pin change the stability region in a way the momentum analysis predicts? Cross-check the constrained-transport view against `adjoint-schr-dinger-bridge-sampler`. The concrete try-on anchor to graduate toward is `catvton-concatenation-is-all-you-need-for-virtual-try-on-with-diffusion-models`.
- **GATE 1 (end of week 2):** if you cannot reproduce the momentum sampler's baseline result AND show one clean, measurable effect of the hard pin on the toy, you do not have a theory project. Pivot — drop to the empirical half of Direction 2 and reassess, rather than burning month two chasing a proof you cannot reach solo.
- **GATE 2 (end of week 4):** if the toy gives a clean, predictable pin effect and a theory co-author has materialized, escalate toward Direction 1 and probe whether the copy region of `an-analytic-theory-of-creativity-in-convolutional-diffusion-models` behaves as the constrained analysis predicts. If you are still solo, stop escalating and write Direction 3 as the paper — a clean, finishable result beats a spectacular unfinished one every time.
