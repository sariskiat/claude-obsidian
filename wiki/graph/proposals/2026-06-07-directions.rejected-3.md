# Directions Report — REJECTED (grounding cap exhausted)

**Unverified citations:** ton-textured-3d-guided-consistent-video-try-on-via-diffusion-models

---

## The bar

Three uncomfortable truths, stated plainly because the whole plan dies if you flinch from them.

1. **A top venue paper is necessary but not sufficient, and not all of them count.** Committees at Ivy and top industrial labs can tell a competent engineer who applied a known method to a new dataset apart from a person who had an idea. A second application paper, even at CVPR, mostly confirms the first read. The paper must signal taste and independence or it does not buy the ticket.

2. **To theory leaning committees, try on is an application area.** A pure try on paper lands at CVPR, ICCV, or WACV and reads as applied vision. The only way it serves your stated goal is if the contribution is a general insight that try on happens to be the testbed for, not "better try on." Apply the swap test ruthlessly: if a researcher who knows nothing about garments could write the paper by swapping the dataset, it is application layer and it does not earn the ticket.

3. **The letter matters as much as the paper.** A top admit rides on a recommender the committee trusts saying the ideas were yours. That is the real reason a collaborator matters: not the extra hands, the letter. Choose partly by who it puts in your corner. With your stated profile (strong VTON, applied diffusion; theory muscle still building; about 4 hours per week) the binding constraint is not ambition, it is finishability of a defensible theoretical claim solo.

## Decision matrix

| # | Direction | Ceiling | Clean-exec odds @4h/wk solo | Theory load | Scoop risk | Builds on strength | Admissions signal |
|---|-----------|:--:|:--:|:--:|:--:|:--:|:--:|
| 1 | Selective spatially varying memorization for try on | 5 | 2 | 5 | 5 | 4 | 5 |
| 2 | Rate distortion ceiling on garment detail transfer | 4 | 3 | 4 | 3 | 5 | 4 |
| 3 | Constrained sampling theory under hard garment pinning | 4 | 3 | 4 | 2 | 5 | 4 |
| 4 | Video VTON appearance versus motion objective tradeoff | 3 | 4 | 2 | 4 | 4 | 2 |
| 5 | Memorization as a feature framing (umbrella) | 4 | 2 | 4 | 4 | 3 | 4 |

Scoop risk of 5 means a powerhouse lab can and might do it first. Higher is better for the goal on every column except where the column names a cost (theory load and scoop risk, where 5 means more load or more risk against you).

### 1. Selective spatially varying memorization for try on

**Thesis.** The garment region must sit in the memorization regime (copy exact patches: logos, weave, seams) while body and background must sit in the generation regime (synthesize plausibly). Reframe memorization from a bug into a spatially controllable resource. The theoretical spine is `an-analytic-theory-of-creativity-in-convolutional-diffusion-models`, where the balance between copying and creating is governed by locality scale and equivariance; the failure side is `why-diffusion-models-don-t-memorize-the-role-of-implicit-dynamical-regularization-in-training`. Try on is the one task that *demands* both regimes inside a single frame at known spatial coordinates, which is exactly why it passes the swap test: the structure comes from the garment mask, not the dataset.

The grounding for the gap is real. The detail preservation pain shows up independently in `art-viton-measurement-guided-latent-diffusion-for-artifact-free-virtual-try-on`, `texture-preserving-diffusion-models-for-high-fidelity-virtual-try-on`, and `improving-virtual-try-on-with-garment-focused-diffusion-models`. Bridge 4 in the dossier (Community 2 to Community 26) is the live connection between the creativity theory and the try on cluster through `DiffusionTrend` and the `Equivariant Local Score (ELS) Machine`.

**Takedown.** This is the highest ceiling and the one most likely to eat a year and leave you with nothing. The theory load is a 5: you would be extending an analytic creativity theory to a spatially conditioned, masked setting, which is real applied math, not a tuning study. Scoop risk is a 5 because Ganguli's group and any well staffed theory lab can reach the same reframe from the pure side and beat you to it without ever touching a garment. At 4 hours per week solo you will not reproduce the analytic machinery, extend it, and validate it on `VITON-HD Dataset` inside one cycle. This direction is only rational if you have secured a real theory co author and more than 4 hours per week. Without the letter writer, the upside collapses: even a good result reads as "engineer got lucky near a famous theory" rather than "this person had the idea."

### 2. Rate distortion ceiling on garment detail transfer

**Thesis.** Formalize *why* multiple independent try on systems hit the same encoder bottleneck and prove a ceiling. The recurring limitation across the corpus is that pretrained image encoders (DINOv2, CLIP) are not optimized for detail preservation, called out directly in `art-viton-measurement-guided-latent-diffusion-for-artifact-free-virtual-try-on` and visible in the CLIP bottleneck noted around the Community 52 to Community 60 bridge. Cast garment detail transfer as a rate distortion problem, derive the achievable distortion floor given encoder channel capacity, and show empirically where systems like `catvton-concatenation-is-all-you-need-for-virtual-try-on-with-diffusion-models`, `ootdiffusion-outfitting-fusion-based-latent-diffusion-for-controllable-virtual-try-on`, and `incorporating-visual-correspondence-into-diffusion-model-for-virtual-try-on` sit relative to that floor.

**Takedown.** The failure mode is sharp and common: without a defensible estimator and a real lower bound, this degrades into a benchmark paper that says "encoders lose detail, here are FID numbers." Everyone already believes encoders lose detail; measuring it again is not a contribution. The information theoretic argument is precisely your stated gap (pure information theoretic and rate distortion bounds are not your trained muscle), so the one part that makes this venue worthy is the part you are weakest at. If you cannot produce a non vacuous lower bound, kill it. Scoop risk is moderate rather than high because the framing is less fashionable than memorization, which is the only reason it ranks above Direction 1 on risk adjusted terms but not on ceiling.

### 3. Constrained sampling theory under hard garment pinning

**Thesis.** Try on inpainting hard pins part of the output (the garment region is fixed to a measured target) while the sampler runs free on the rest. Generic sampler stability theory assumes free sampling everywhere. Characterize how a hard constraint at known coordinates changes sampling dynamics, divergence behavior, and convergence order. The tool is the momentum and stability machinery in `diffusion-sampling-with-momentum-for-mitigating-divergence-artifacts` (the `Heavy Ball Momentum` and GHVB line, where the corpus already records that for β below 1 the method drops to first order convergence and image quality falls). The constrained sampler neighbors are `adjoint-schr-dinger-bridge-sampler` and `glass-flows-transition-sampling-for-alignment-of-flow-and-diffusion-models`. Dossier Bridge 1 (Community 0 to Community 60) and Bridge 3 are the exact white space the profile flagged: the VTON cluster has zero claim edges to the sampling and momentum communities, so this is genuinely unclaimed ground rather than a crowded race.

**Takedown.** The honest risk is that "constrained sampling" already exists as a literature and your contribution shrinks to "I applied a known constrained sampler to garments," which fails the swap test and dies. You avoid that only if the *structure of the try on constraint* (a hard region pin with a soft boundary seam, at known coordinates) provably changes the divergence or convergence analysis in a way free sampling does not predict. That is the whole paper; if the constraint turns out to be analytically boring, there is no result. Also note the phase rule on your own profile: this leans on Aek's machinery, and you do not email Aek until all four Aek deep reads are done. He is the testbed here, not the thesis. The upside is that this is the most finishable of the five solo at 4 hours per week, the theory load is the lowest of the serious options, and scoop risk is genuinely low because the bridge is empty.

### 4. Video VTON appearance versus motion objective tradeoff

**Thesis.** Video try on optimizes appearance fidelity and temporal motion coherence with objectives that pull against each other. Characterize the tradeoff frontier across `3dv-ton-textured-3d-guided-consistent-video-try-on-via-diffusion-models [UNVERIFIED — not in graph.db]`, `chronotailor-harnessing-attention-guidance-for-fine-grained-video-virtual-try-on`, and `catv2ton-taming-diffusion-transformers-for-vision-based-virtual-try-on-with-temporal-concatenation`.

**Takedown.** Skip it. The ceiling is CVPR applied vision and the admissions signal is a 2: this is the most "competent engineer" shaped of the five and the swap test is shaky because most of the tension is generic video diffusion temporal consistency, not a try on specific structure. It is the easiest to execute, which is exactly the trap, because easy plus low signal is the worst quadrant for your stated goal. Do not spend a cycle here.

### 5. Memorization as a feature framing (umbrella)

**Thesis.** Reframe memorization from failure to deliberately exploited resource, wrapping Direction 1 and parts of Direction 2. Same theoretical anchors: `an-analytic-theory-of-creativity-in-convolutional-diffusion-models` and `why-diffusion-models-don-t-memorize-the-role-of-implicit-dynamical-regularization-in-training`, with `neon-negative-extrapolation-from-self-training-improves-image-generation` and `universal-inverse-distillation-for-matching-models-with-real-data-supervision-no-gans` as the training side levers (Bridges 2, 4, 5).

**Takedown.** This does not stand alone as a paper. It is a framing, and a framing without a theorem or a measured ceiling is a blog post. Its only legitimate use is as the narrative wrapper that makes Direction 1 read as an idea rather than a trick. Treat it as packaging for 1, never as a standalone submission.

## Ranking

1. **Direction 1** (selective spatially varying memorization) — highest ceiling and best admissions signal, but only if a theory co author and more than 4 hours per week are real; otherwise it is a trap.
2. **Direction 3** (constrained sampling under hard garment pinning) — most defensible and most finishable solo, built on genuine graph white space, lowest theory load of the serious options.
3. **Direction 2** (rate distortion ceiling) — strong if you can land a real lower bound, but it leans hardest on your weakest muscle.
4. **Direction 5** (memorization as a feature) — valuable only as the wrapper for Direction 1, not as a standalone paper.
5. **Direction 4** (video appearance versus motion) — skip; low admissions signal and weak swap test.

**Fork rule.** Theory co author secured and more than 4 hours per week available: swing for Direction 1, wrapped in the Direction 5 framing. Mostly solo at 4 hours per week: ship Direction 3.

## Execution

Top ranked for the realistic solo case is **Direction 3**. First week proof of concept probe:

- **Reproduce.** Stand up the momentum sampler baseline from `diffusion-sampling-with-momentum-for-mitigating-divergence-artifacts` on a small unconditional diffusion model and confirm the recorded behavior: convergence order degradation when the momentum coefficient drops below 1. This is the cheap, known result you must be able to reproduce before trusting anything downstream.
- **Toy model.** Build a 2D synthetic sampler where part of the state is hard pinned to a fixed target (a stand in for the garment region) while the rest samples freely. Measure divergence and effective convergence order with the pin on versus off, sweeping the momentum coefficient. The question you are answering: does the hard pin change the stability region predicted by the free sampling analysis? Use a `VITON-HD Dataset` garment mask only to set realistic pin geometry later, not in week one.
- **Gate condition.** If by end of week two the pinned toy shows a *measurable* shift in the divergence or convergence behavior versus the free sampler, continue and escalate toward the masked latent setting using `catvton-concatenation-is-all-you-need-for-virtual-try-on-with-diffusion-models` as the try on backbone. If the pin is analytically inert (no shift beyond noise), the paper has no spine: pivot to Direction 2 and spend the next probe attempting a non vacuous rate distortion lower bound on the encoder bottleneck documented in `art-viton-measurement-guided-latent-diffusion-for-artifact-free-virtual-try-on`. Do not email Aek before all four Aek deep reads are complete, regardless of how the probe goes.
