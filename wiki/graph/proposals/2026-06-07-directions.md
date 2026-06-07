## The bar

Three things have to be true at once, and all three are uncomfortable.

**First: a top-tier paper is necessary, not sufficient, and not every top-tier paper counts.** Committees at the labs you are aiming at have read a thousand "competent engineer ported a known method to a new dataset" papers. They can smell it. Your prior work, by your own description, reads that way. A second paper of the same shape, even accepted, confirms the first read instead of overturning it. The paper has to carry a fingerprint that says *this person had an idea and could defend it*, or it does not buy the ticket.

**Second: to a theory-leaning committee, try-on is an application area, full stop.** A pure try-on result lands at the flagship vision conference and gets filed under applied vision. That is fine for an industry team and weak signal for a NeurIPS-class machine-learning group. Try-on only helps your stated goal if the contribution is a *general* claim about diffusion that try-on happens to be the cleanest testbed for. The swap test is the whole game: if a researcher who has never heard of garments can reproduce your paper by swapping the dataset, you built an application and the ticket does not print. Every direction below lives or dies on that test.

**Third: the letter matters as much as the paper.** A top admit usually rides on a recommender the committee already trusts saying, in plain words, *the ideas were his*. That is the real reason a theory co-author matters for Direction 1, and it is also the reason you should not email Aek one day early. A co-author who joins before you have done the reading writes a letter about a student who needed scaffolding. A co-author who joins after you walk in with a worked toy model writes a letter about a peer.

Now the menu.

## Decision matrix

Ratings are 1–5, higher is better *for the PhD goal*, except **Scoop risk** where 5 means a powerhouse lab can and might do it first.

| # | Direction | Ceiling | Clean-exec odds @4h/wk solo | Theory load | Scoop risk (5=bad) | Builds on strength | Admissions signal |
|---|-----------|:--:|:--:|:--:|:--:|:--:|:--:|
| 1 | Spatially-varying / selective memorization for try-on | 5 | 2 | 5 | 5 | 4 | 5 |
| 2 | Rate–distortion ceiling on garment-detail transfer | 4 | 3 | 4 | 3 | 5 | 4 |
| 3 | Constrained-sampling theory under hard garment-pinning | 4 | 4 | 3 | 2 | 5 | 4 |
| 4 | Video try-on appearance-vs-motion objective tradeoff | 3 | 4 | 2 | 4 | 4 | 2 |
| 5 | Memorization-as-a-feature (umbrella framing) | 4 | 2 | 4 | 4 | 3 | 4 |

The graph also surfaced lower-relevance bridges that I am scoring out before they waste a week: the Knowledge-Graph / GraphRAG crossings (`agentic-graphrag-guide`, `anthropic-kg-cookbook`), the Auto-Compressing Networks crossing (`auto-compressing-networks`), and the DiffusionLight crossing (`diffusionlight-light-probes-for-free-by-painting-a-chrome-ball`) all came back at direction-relevance 0.30. They are tooling and adjacency, not a thesis. Ignore them.

### 1. Spatially-varying / selective memorization for try-on

Thesis: try-on is the rare task that needs *two different generative regimes inside one image*. The garment region must be in the memorization regime, where the model copies exact patches (logos, weave, seam topology). The body and background must stay in the creativity regime, where it synthesizes. The graph's strongest theory-to-try-on bridge here is Community 2 (Neon, Universal Inverse Distillation, Regularized Parameter Scaling) crossing into Community 26, anchored by `an-analytic-theory-of-creativity-in-convolutional-diffusion-models` and `diffusion-sampling-with-momentum-for-mitigating-divergence-artifacts` (see `bridge-synthesis`). The creativity theory's own stated limit — its ideal score machine *can only* memorize training data — is exactly the knob. If locality scale and equivariance govern the memorize/create balance, then a spatially-varying score should let you *place* the boundary on purpose. Try-on hands you a ground-truth segmentation of where each regime belongs. That is the structural property no other task gives you for free, and it passes the swap test cleanly.

**Takedown:** This is a theory paper wearing a try-on coat, and you do not yet have the theory muscle to carry it solo. The contribution is a claim about score-function locality, which is precisely your stated gap (ergodic / information-theoretic argument). At 4h/week without a real theory co-author, the most likely outcome is a beautiful framing with a hand-wavy core that a reviewer at a NeurIPS-class venue dismantles in one paragraph. Scoop risk is maximal: the `an-analytic-theory-of-creativity-in-convolutional-diffusion-models` group and anyone within one hop of it can do the spatially-varying extension faster than you, and they will if it is obvious — and it is becoming obvious. This is the swing-for-the-fence option, not the default. It only becomes rational once a co-author is secured *and* the hours go up.

### 2. Rate–distortion ceiling on garment-detail transfer

Thesis: at least three try-on papers independently hit the *same* wall and name it the same way. The recurring limitation across the bridges is blunt — pretrained image encoders are not optimized for fine garment detail, and inversion through them loses the weave. You see it in `art-viton-measurement-guided-latent-diffusion-for-artifact-free-virtual-try-on`, in `texture-preserving-diffusion-models-for-high-fidelity-virtual-try-on`, and in the temporal-concatenation line at `catv2ton-taming-diffusion-transformers-for-vision-based-virtual-try-on-with-temporal-concatenation`. When three independent groups hit one wall, the wall is probably a ceiling, not an engineering miss. The paper formalizes it: derive the rate–distortion bound on how much garment information can survive the encoder bottleneck, then show empirically where current encoders sit relative to that bound. Theory anchor for the generalization side is `on-the-closed-form-of-flow-matching-generalization-does-not-arise-from-target-stochasticity`.

**Takedown:** The failure mode is a benchmark paper in disguise. Without a *defensible estimator* of mutual information at these dimensions and a *real* lower bound — not a variational surrogate you wave at — this collapses into "we measured how bad the encoders are," which is a workshop result. Mutual-information estimation in high dimension is a knife that has cut a lot of people, and it sits squarely in your stated gap. The ceiling has to be tight enough that a reviewer believes no encoder can beat it, otherwise the response is "train a better encoder," and your theory evaporates. Builds beautifully on your strength, but the make-or-break step is the one part you are weakest at.

### 3. Constrained-sampling theory under hard garment-pinning

Thesis: every generic sampler theory assumes you are sampling *freely* from the model. Try-on does not do that. It pins part of the output — the garment region is effectively clamped — and samples the rest conditioned on that clamp. That is a hard constraint imposed mid-trajectory, and nobody has characterized what it does to sampling dynamics and divergence. The tooling is already in the graph: Community 0 (DiffSDA, Heavy Ball Momentum, sampling dynamics) bridges directly into the image try-on community, anchored by `diffusion-sampling-with-momentum-for-mitigating-divergence-artifacts`, with `glass-flows-transition-sampling-for-alignment-of-flow-and-diffusion-models` and `adjoint-schr-dinger-bridge-sampler` as the sampler-theory neighbors. The try-on instantiations are `catvton-concatenation-is-all-you-need-for-virtual-try-on-with-diffusion-models` and `art-viton-measurement-guided-latent-diffusion-for-artifact-free-virtual-try-on`. Bridge 12 adds the noise-optimization angle via `optimizing-diffusion-noise-can-serve-as-universal-motion-priors`. Aek's momentum and boundary-locus machinery is the *tool* here, not the thesis. The claim is general: hard constraints change divergence behavior in a way the free-sampling analysis misses, and the momentum order interacts with the constraint.

**Takedown:** The honest risk is that the result is *small*. You may prove a clean, correct statement about how a clamped coordinate shifts the momentum sampler's stability region and find it is a modest correction nobody outside sampling-theory cares about. That is a solid CVPR-adjacent paper and a weak admissions signal if you cannot tie it to something a theory committee finds surprising. The recurring caveat in the source material is a warning shot: when the momentum coefficient drops below one, the heavy-ball scheme degrades to first-order and quality collapses — which means your "interesting" regime may be a thin band that is hard to hit and hard to sell. The mitigation is that this is the *only* direction here that a solo author at 4h/week can actually finish with a defensible core, because the theory load is bounded and the testbed is your home turf.

### 4. Video try-on appearance-vs-motion objective tradeoff

Thesis: video try-on optimizes appearance fidelity and motion/temporal consistency against each other, and the bridge from Community 2 into Community 52 (3DV-TON, ChronoTailor, DPIDM) suggests the tradeoff is structural. Sources: `3dv-ton-textured-3d-guided-consistent-video-try-on-via-diffusion-models`, `chronotailor-harnessing-attention-guidance-for-fine-grained-video-virtual-try-on`, `dpidm-temporal-consistent-video`, and `pursuing-temporal-consistent-video-virtual-try-on-via-dynamic-pose-interaction`.

**Takedown:** Skip it. The ceiling is a strong applied-vision paper and the admissions signal is the weakest on the board (2). It fails the swap test the hardest: any video-generation researcher writes this by swapping the dataset, because the appearance-vs-temporal tradeoff is not a property of *garments*, it is a property of video. You would be spending your scarcest resource — hours — on the direction least likely to read as "had an idea." It earns a row only so the matrix is honest about why it loses.

### 5. Memorization-as-a-feature (umbrella framing)

Thesis: reframe memorization from a bug to be suppressed into a *resource to be allocated*. The literature treats it as failure — `why-diffusion-models-don-t-memorize-the-role-of-implicit-dynamical-regularization-in-training` studies how to *avoid* it, and `neon-negative-extrapolation-from-self-training-improves-image-generation` treats over-fit signal as something to extrapolate *away* from. The umbrella inverts that: in try-on, exact reproduction of the source garment is the *goal*, so memorization is the feature. The creativity theory `an-analytic-theory-of-creativity-in-convolutional-diffusion-models` supplies the regime language.

**Takedown:** This does not stand alone as a paper — it is a *framing*, and framings get desk-rejected when submitted as contributions. Its honest role is the wrapper around Direction 1: it is the abstract's first paragraph, not the result. Treat it as the narrative spine if Direction 1 fires, and as nothing if it does not. Do not let it seduce you into a position paper.

## Ranking

Order: **1 > 3 > 2 > 5 > 4.**

1. **Direction 1** — highest ceiling and best admissions signal because it is a general claim about score-function locality that try-on merely instantiates; this is the one that actually buys the ticket *if* it lands.
2. **Direction 3** — best risk-adjusted bet and the only one a solo author at 4h/week can finish with a defensible core, using `diffusion-sampling-with-momentum-for-mitigating-divergence-artifacts` as the tool.
3. **Direction 2** — strong and on-strength, but gated entirely on a tight lower bound you are not yet equipped to derive.
4. **Direction 5** — real as a framing, worthless as a standalone submission; it is the wrapper for #1.
5. **Direction 4** — highest finish odds, lowest payoff; fails the swap test, so it does not serve the goal.

**Fork rule.** Theory co-author secured *and* more than 4h/week → swing for Direction 1, with Direction 5 as its framing. Mostly solo at 4h/week → ship Direction 3 and treat it as the falsification beachhead toward Direction 1. Do not email Aek until all four Aek deep-reads are done; he is the tool for Direction 3, not the thesis.

## Execution

First-week probe runs Direction 3 as the falsification beachhead for Direction 1 — the cheapest experiment that can kill the top-ranked idea before you spend a quarter on it.

**Paper to reproduce:** `diffusion-sampling-with-momentum-for-mitigating-divergence-artifacts`. Reproduce its divergence-artifact baseline first, because every later claim rests on your sampler behaving exactly as that paper reports.

**Toy model to build:** a 2-D synthetic diffusion where one coordinate is *hard-pinned* (the garment analog) and the other is sampled freely (the body analog), driven by the heavy-ball momentum sampler from that paper.

```
   free coord (body)
        ^
        |    .  trajectory under free sampling
        |   /
        |  /        <- measure: divergence rate,
        | /            stability-region shift
        |/_____________ pinned coord (garment) = clamped
        garment held fixed mid-trajectory
```

Measure how the momentum sampler's stability region and divergence rate change when the pinned coordinate is clamped versus free. If clamping shifts the stability boundary in a way the free-sampling analysis does not predict, you have the seed of a general result and a reason to reach for the memorization-regime escalation via `an-analytic-theory-of-creativity-in-convolutional-diffusion-models`. Then, and only then, instantiate on real try-on with `catvton-concatenation-is-all-you-need-for-virtual-try-on-with-diffusion-models` and `art-viton-measurement-guided-latent-diffusion-for-artifact-free-virtual-try-on`.

**Gate one (end of week 2):** reproduce the momentum-divergence baseline to within the paper's reported numbers. If you cannot, the tooling is not under your control and no theory built on it will survive review — pivot to Direction 2 (rate–distortion), which leans on your encoder strengths instead of sampler internals.

**Gate two (end of week 4):** if the clamped toy shows a *measurable, predictable* shift in the stability region, escalate toward Direction 1 and start the Aek deep-reads in earnest. If the shift is absent or a negligible correction, ship Direction 3 as the bounded, defensible result and write the smaller honest paper rather than chasing a ceiling that is not there.

---

*Grounding audit: 19/19 citations verified ✓ | retries: 2 | model: 2.1.168 (Claude Code) | bridge candidates: 12*
