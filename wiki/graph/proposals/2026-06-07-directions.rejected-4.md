# Directions Report — REJECTED (grounding cap exhausted)

**Unverified citations:** first-order-convergence

---

## The bar

Three things are true whether or not you want them to be, and the whole plan dies if you flinch on any of them.

**One: a top-tier paper is necessary, not sufficient, and not all of them count.** Committees at the places you are aiming read the difference between "competent engineer applied a known method to a new dataset" and "person who had an idea." Your existing strength, end-to-end VTON pipelines, is exactly the first kind. Another paper in that mold, even at CVPR, confirms the first read. The contribution has to signal taste and independence or it does not buy the ticket.

**Two: to a theory-leaning committee, VTON is an application area.** A pure try-on paper lands at CVPR, ICCV, or WACV and reads as applied vision. That is a fine outcome for an industry team and a weak one for an Ivy ML-theory group. The only way try-on helps the stated goal is if the contribution is a general insight that try-on is the testbed for, and would survive the swap test: if a researcher who knows nothing about garments could reproduce the paper by swapping in another dataset, it is application layer and it does not earn the ticket.

**Three: the letter matters as much as the paper.** A top admit usually rides on a recommender the committee trusts saying the ideas were yours. That is the real reason a collaborator matters, and it is also the trap. If Aek's name and machinery carry the paper, the letter says the ideas were Aek's. The phase rule in your own profile (no email until all four Aek deep-reads are done) exists for exactly this reason. His momentum and boundary-locus tools are the testbed for Direction 3, not the thesis.

The honest resource picture: 4h/week, solo, strong on VTON and applied diffusion, still building sampling-dynamics and stability theory, with a real gap on information-theoretic and ergodic arguments. That picture rules things in and out before any bridge score does.

## Decision matrix

| # | Direction | Ceiling | Clean-exec odds @4h/wk solo | Theory load | Scoop risk | Builds on strength | Admissions signal |
|---|-----------|:------:|:---:|:---:|:---:|:---:|:---:|
| 1 | Selective / spatially-varying memorization for try-on | 5 | 2 | 5 | 5 | 4 | 5 |
| 2 | Rate-distortion ceiling on garment-detail transfer | 4 | 3 | 4 | 3 | 5 | 4 |
| 3 | Constrained-sampling theory under hard garment-pinning | 4 | 3 | 4 | 2 | 5 | 4 |
| 4 | Video-VTON appearance-vs-motion objective tradeoff | 3 | 4 | 2 | 4 | 4 | 2 |
| 5 | Memorization-as-a-feature framing (umbrella) | 4 | 2 | 4 | 4 | 3 | 4 |

Scoop risk is scored so 5 is bad: a powerhouse lab can and might do it first. Clean-exec odds are conditioned on your real constraint, 4h/week and solo.

### 1. Selective / spatially-varying memorization for try-on

The thesis. Try-on is the one generation task where memorization and creativity are spatially separable by construction. The garment region must be copied close to verbatim, exact texture, print, seams, while the body and background must be synthesized to fit a new pose. `an-analytic-theory-of-creativity-in-convolutional-diffusion-models` gives an analytic handle on when a convolutional diffusion model copies versus invents, governed by locality and equivariance, with the `Equivariant Local Score (ELS) Machine` as the predictive object. `why-diffusion-models-don-t-memorize-the-role-of-implicit-dynamical-regularization-in-training` supplies the other half: memorization is something training dynamics suppress, so it can in principle be re-enabled selectively. Reframing memorization from bug to a spatially-controllable resource is a general insight; try-on is the cleanest place in vision to demonstrate it. This passes the swap test outright, because the spatial mask comes from try-on's structure, not from the data.

Bridge support: this is the direction behind Bridge 4 (Community 2 ↔ Community 26, anchors `Neon (Negative Extrapolation)`, `DiffusionTrend`, papers `an-analytic-theory-of-creativity-in-convolutional-diffusion-models`, `diffusion-sampling-with-momentum-for-mitigating-divergence-artifacts`) and Bridge 8 (Community 2 ↔ Community 24, anchor `Auto-Compressing Networks (ACNs)`). The flagged limitation from the creativity theory, "ideal score machine can only memorize training data," is precisely the knob you would turn into a feature.

**Takedown:** This is the highest ceiling and the one most likely to eat a year and produce nothing submittable solo at 4h/week. The theory load is a 5 because the result only counts if you can predict where the model crosses from copy to create as a function of locality scale, and prove your spatial control changes that crossing. That is real score-machine analysis in the regime your profile lists as your gap. Scoop risk is a 5: Ganguli-adjacent groups own this theory and can extend it to masked or conditional generation faster than you can climb the math. Without a theory co-author and more than 4h/week, the realistic outcome is a qualitative paper showing nicer garment fidelity, which collapses straight back into the application-layer paper you are trying to escape, and the swap test stops protecting you the moment the contribution becomes "it looks better."

### 2. Rate-distortion ceiling on garment-detail transfer

The thesis. Multiple independent try-on systems hit the same wall, and they name it the same way: the conditioning encoder cannot carry fine garment detail. The dossier states it twice, "Pre-trained image encoders like DINOv2 and CLIP are not optimized for detail preservation, critical for VTON" (Bridge 1, `art-viton-measurement-guided-latent-diffusion-for-artifact-free-virtual-try-on`) and "CLIP-based textual inversion fails to preserve fine garment details due to CLIP encoder bottleneck" (Bridge 7, `texture-preserving-diffusion-models-for-high-fidelity-virtual-try-on` lineage). The move is to formalize that wall as a rate-distortion ceiling: given an encoder of finite capacity, derive the minimum achievable garment-detail distortion, show where current systems sit relative to it, and say whether the field is fighting an architecture problem or an information-theoretic one. `incorporating-visual-correspondence-into-diffusion-model-for-virtual-try-on` and `difffit-disentangled-garment-warping-and-texture-refinement-for-virtual-try-on` give you the correspondence and warping baselines that sidestep the encoder, which is exactly the comparison a ceiling argument needs.

Bridge support: Bridge 1 and Bridge 5 (Community 2 ↔ Community 60, anchors `Neon (Negative Extrapolation)`, `CatVTON`), both at direction_relevance 1.00.

**Takedown:** The failure mode is a benchmark paper wearing a tuxedo. A rate-distortion "ceiling" is only a contribution if the bound is defensible, meaning a real estimator of the achievable rate and a real lower bound, not an FID curve you relabel as a frontier. Your profile flags pure information-theoretic argument as the gap, and this direction is information theory with a try-on coat of paint. If the estimator is hand-wavy, a reviewer with one mutual-information course will dismantle it, and you will not see the hole coming. There is also a swap-test exposure: rate-distortion on a conditioning encoder is not specific to garments, so unless the distortion measure is built from try-on's structure (warp-consistent detail, not pixel MSE) a referee can correctly say this is a generic encoder-capacity result that used clothes as a demo.

### 3. Constrained-sampling theory under hard garment-pinning

The thesis. Try-on at inference is not free sampling. The garment region is pinned to ground truth (inpainting / hard constraint) while the rest is sampled. Generic sampler theory and the stability machinery in Aek's community assume free sampling: `diffusion-sampling-with-momentum-for-mitigating-divergence-artifacts` analyzes divergence artifacts under `Heavy Ball Momentum`, `GLASS Flows` (`glass-flows-transition-sampling-for-alignment-of-flow-and-diffusion-models`) and `adjoint-schr-dinger-bridge-sampler` characterize transition and bridge sampling, all without a hard pin. The contribution: characterize how a hard inpainting constraint changes the sampling dynamics and the divergence/stability regime, and derive when the momentum correction that stabilizes free sampling helps or actively hurts at the constraint boundary. That is a general sampling-theory result; try-on supplies the constraint geometry.

Bridge support: this is the spine of Bridge 1 and Bridge 3 (Community 0 ↔ 60 and 0 ↔ 52, anchors `DiffSDA`, `Heavy Ball Momentum`, `GLASS Flows`), both direction_relevance 1.00, and the named open question is yours to take: "Countless unexplored ways exist to expand numerical method stability regions beyond HB and GHVB." The recurring limitation "When β<1, HB drops to first-order convergence, significantly decreasing image quality" is a concrete boundary-behavior claim you can test against a pinned region.

**Takedown:** The danger here is not that it is too hard, it is that it is too comfortable and quietly becomes Aek's paper. The momentum and boundary-locus tools are his, and if the result is "GHVB behaves like X near an inpainting boundary," the letter writes itself in the wrong name and your independence signal evaporates, which by truth three is fatal. To own it you must define the constrained-sampling object yourself before any email goes out, and the phase rule in your profile is non-negotiable for exactly this reason. Second risk: theory load is scored low (2) because the toolkit exists, but "characterize the dynamics" can degrade into an empirical ablation if you cannot state and prove one clean proposition about the constrained divergence regime. An ablation that confirms inpainting changes convergence is not a NeurIPS contribution, it is a workshop note.

### 4. Video-VTON appearance-vs-motion objective tradeoff

The thesis. Video try-on must hold garment appearance fixed while synthesizing motion, and the two objectives fight. `3dv-ton-textured-3d-guided-consistent-video-try-on-via-diffusion-models`, `chronotailor-harnessing-attention-guidance-for-fine-grained-video-virtual-try-on`, `dpidm-temporal-consistent-video`, `dreamvvt-mastering-realistic-video-virtual-try-on-in-the-wild-via-a-stage-wise-diffusion-transformer-framework`, and `vivid-video-virtual-try-on-using-diffusion-models` all wrestle the same appearance/temporal coherence tension. Formalize the tradeoff, show a frontier, propose a scheduling that dominates it.

Bridge support: Bridge 2 (Community 2 ↔ 52) and Bridge 12 (Community 1 ↔ 79), both direction_relevance 1.00.

**Takedown:** Skip it. Your own ranking already buries this and the matrix says why: ceiling 3, admissions signal 2. This is a CVPR-shaped applied-vision result, and per truth two that is the wrong venue signature for the stated goal. It fails the swap test the moment you state it, because "appearance versus motion" is a generic video-generation tradeoff that any video diffusion researcher could write with a different dataset; nothing in it is forced by garments. It is also the most resource-hungry option (theory load 2 only because there is barely any, exec cost high because video pipelines are heavy), so it spends your scarcest asset, time, on the lowest-signal outcome. The one defensible reason to touch it would be a fast applied win for an industry application, which is not the goal on record.

### 5. Memorization-as-a-feature framing (umbrella)

The thesis. A reframing wrapper over Directions 1 and 2: memorization is not a failure to suppress but a resource to allocate, and try-on is the task that makes the allocation legible. It gives a paper its narrative spine and connects `why-diffusion-models-don-t-memorize-the-role-of-implicit-dynamical-regularization-in-training` to the practical fidelity pressure in `art-viton-measurement-guided-latent-diffusion-for-artifact-free-virtual-try-on` and `texture-preserving-diffusion-models-for-high-fidelity-virtual-try-on`.

**Takedown:** It does not stand alone, and treating it as a direction is how you end up with a position paper no committee rewards. A framing with no theorem and no estimator is an essay; admissions signal collapses without a hard result underneath. Use it as the abstract for whichever of 1 or 2 you actually prove, never as the contribution itself. If you find yourself writing this paper directly, you have talked yourself out of doing the hard part.

## Ranking

1 > 3 > 2 > 5 > 4.

- **Direction 1** — highest ceiling and the cleanest swap-test pass, but only realistic with a theory co-author and real hours; as a swing it is the one that buys the ticket outright.
- **Direction 3** — the most defensible, best risk-adjusted, finishable solo at 4h/week because the tools exist and the constraint object is yours to define.
- **Direction 2** — strong fit to your strength and lower scoop risk, but one weak estimator from becoming a benchmark paper.
- **Direction 5** — real as a framing, empty as a standalone; ride it on top of 1 or 2.
- **Direction 4** — skip; CVPR-level ceiling, weak admissions signal, fails the swap test.

**Fork rule.** Theory co-author secured and more than 4h/week available: swing for Direction 1. Mostly solo at 4h/week: ship Direction 3. Run Direction 3 as a 4-week falsification probe toward Direction 1, with Gate 1 at week 2 (reproduce the baseline or pivot) and Gate 2 at week 4 (escalate to 1, ship 3, or fall back to 2). Hold the Aek phase rule throughout: no email until all four deep-reads are done, because the contribution must read as yours.

## Execution

Top-ranked finishable bet: Direction 3, constrained-sampling theory under hard garment-pinning.

**Week-1 probe.**

1. Reproduce, do not extend. Stand up the momentum sampler from `diffusion-sampling-with-momentum-for-mitigating-divergence-artifacts` on a pretrained latent diffusion model and confirm you can reproduce its divergence-artifact behavior under free sampling, including the reported β<1 first-order-convergence [UNVERIFIED — not in graph.db] drop. This is the baseline you must own before any theory.

2. Build the toy model. Take a 2D or low-dimensional Gaussian-mixture diffusion where the score is closed-form, then impose a hard pin on a subset of coordinates (the toy analog of a clamped garment region) and sample the rest. Measure how the divergence/stability behavior near the pinned boundary differs from free sampling, with and without the momentum correction. The whole point is a controlled setting where you can see the constrained dynamics analytically, not a try-on pipeline.

3. State one proposition. Write down a single falsifiable claim, for example: "the momentum correction that reduces divergence in free sampling changes sign in its effect within k steps of a hard constraint boundary." If you cannot state it cleanly by end of week 1, that is signal, not failure.

**Gate condition (pivot trigger).** By end of week 2 you must have (a) the reproduced baseline and (b) the toy model showing a measurable, repeatable difference in boundary behavior between pinned and free sampling. If the baseline will not reproduce, or the toy model shows no constraint-dependent effect, pivot to Direction 2 and reframe the week's work as the encoder-capacity ceiling, reusing `art-viton-measurement-guided-latent-diffusion-for-artifact-free-virtual-try-on` and `incorporating-visual-correspondence-into-diffusion-model-for-virtual-try-on` as the systems that sit against the bound. Do not contact Aek at either gate; the phase rule holds until all four deep-reads are complete.
