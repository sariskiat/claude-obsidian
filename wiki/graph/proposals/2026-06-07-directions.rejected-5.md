# Directions Report — REJECTED (grounding cap exhausted)

**Unverified citations:** memorize-vs-create, swap-test-failing, try-on-specific, vton-hd-dataset

---

## The bar

You are not trying to write a good paper. You are trying to write a paper that makes an admissions committee at an Ivy ML-theory group or a top industrial lab believe you had an idea. Those are different targets, and most VTON work clears the first and dies on the second. Three uncomfortable truths, stated plainly because flinching here costs you a year.

**Truth 1 — a top-tier paper is necessary, not sufficient, and not all of them count.** A committee can smell the difference between "competent engineer ported a known method to a new dataset" and "this person saw something." A second paper of the first kind, even at CVPR, just confirms the first read. The contribution has to signal taste and independence or it does not buy the ticket. Every VTON bridge in your dossier — `catv2ton-taming-diffusion-transformers-for-vision-based-virtual-try-on-with-temporal-concatenation`, `art-viton-measurement-guided-latent-diffusion-for-artifact-free-virtual-try-on`, `ootdiffusion-outfitting-fusion-based-latent-diffusion-for-controllable-virtual-try-on` — is the first kind unless you extract a general claim from it.

**Truth 2 — to a theory committee, Virtual Try-On is an application area.** A pure try-on result lands at CVPR/ICCV/WACV and reads as applied vision. The only way it helps your stated goal is if the contribution is a general insight about diffusion that try-on happens to be the cleanest testbed for. The swap test is the whole game: if a researcher who has never heard of garments could reproduce your paper by swapping the dataset, you built application-layer plumbing. `difffit-disentangled-garment-warping-and-texture-refinement-for-virtual-try-on` and `texture-preserving-diffusion-models-for-high-fidelity-virtual-try-on` are excellent engineering and they fail this test by construction.

**Truth 3 — the letter weighs as much as the paper.** A top admit usually rides on a recommender the committee already trusts vouching that the ideas were yours. That is the real reason a collaborator matters — not the extra hands, the letter. This is also why the Aek phase rule is not optional: a half-baked cold email burns the one bridge that could produce that letter. Do not email Aek until all four deep-reads are done. His momentum machinery in `diffusion-sampling-with-momentum-for-mitigating-divergence-artifacts` is a tool for your thesis, not the thesis.

Now the menu. I am scoring against the goal, not against "is this a publishable paper."

## Decision matrix

| # | Direction | Ceiling | Clean-exec odds @4h/wk solo | Theory load | Scoop risk | Builds on strength | Admissions signal |
|---|-----------|:------:|:----:|:----:|:----:|:----:|:----:|
| 1 | Selective / spatially-varying memorization for try-on | 5 | 2 | 5 | 5 | 4 | 5 |
| 2 | Rate-distortion ceiling on garment-detail transfer | 4 | 3 | 4 | 3 | 5 | 4 |
| 3 | Constrained-sampling theory under hard garment-pinning | 4 | 3 | 4 | 2 | 5 | 4 |
| 4 | Video-VTON appearance-vs-motion objective tradeoff | 3 | 4 | 2 | 4 | 4 | 2 |
| 5 | Memorization-as-a-feature framing (umbrella) | 4 | 2 | 4 | 4 | 3 | 4 |

(Scoop risk 5 = a powerhouse lab can and might ship it first. Higher is better on every column except where the column names a cost — read theory load and scoop risk as "how much this hurts.")

### 1. Selective / spatially-varying memorization for try-on

**Thesis.** Try-on is the one generation task where the ground truth *demands two regimes in one frame*: the garment region must be in the memorization regime (copy the exact texture, logo, weave) while body and background must be in the generation regime (synthesize plausibly, never copy the source identity). `an-analytic-theory-of-creativity-in-convolutional-diffusion-models` derives the memorize-vs-create [UNVERIFIED — not in graph.db] balance from locality and equivariance; `why-diffusion-models-don-t-memorize-the-role-of-implicit-dynamical-regularization-in-training` derives *when* memorization onsets dynamically. Neither asks what happens when you *want* memorization in a known sub-region and forbid it elsewhere. Try-on hands you a spatial mask that makes the regime boundary an input, not an emergent accident. That reframes memorization from a bug to a controllable resource — and it passes the swap test cleanly, because the structural fact (a hard spatial split of regimes) is intrinsic to try-on, not bolted on.

**Takedown.** This is the highest ceiling and the one most likely to eat your year with nothing to show. The theory load is a 5 — you would be extending an analytic creativity theory and an implicit-regularization argument simultaneously, and your own profile flags pure information-theoretic and ergodic analysis as the gap, not the strength. Scoop risk is also a 5: Ganguli's group, or anyone who already owns `an-analytic-theory-of-creativity-in-convolutional-diffusion-models`, can formalize spatially-varying memorization faster than you can, and they will recognize it as the obvious next move. At 4h/week solo you do not have the throughput to win a race against the lab that wrote the theory you are extending. This direction is only rational with a real theory co-author *and* more than 4h/week. Absent both, attempting it is how you produce a vague workshop paper that reads as derivative — the exact opposite of the "had an idea" signal.

### 2. Rate-distortion ceiling on garment-detail transfer

**Thesis.** Three independent VTON lines hit the *same* wall and each treats it as an engineering nuisance: `art-viton-measurement-guided-latent-diffusion-for-artifact-free-virtual-try-on`, `texture-preserving-diffusion-models-for-high-fidelity-virtual-try-on`, and `difffit-disentangled-garment-warping-and-texture-refinement-for-virtual-try-on` all fight to push fine garment detail through a pretrained encoder that was never optimized to preserve it. The dossier limitation is explicit — DINOv2/CLIP encoders are "not optimized for detail preservation, critical for VTON," and CLIP textual inversion "fails to preserve fine garment details due to CLIP encoder bottleneck." The paper is: stop patching the encoder, *formalize the ceiling*. Derive the rate-distortion bound on how much garment high-frequency detail survives a fixed-capacity latent, show analytically where each of these methods sits relative to the bound, and predict which fixes can and cannot help. That converts three engineering papers into one general statement about conditioning through bottlenecked latents.

**Takedown.** The failure mode is sharp and common: without a *defensible estimator* of the mutual information and a *real lower bound*, this degrades into a benchmark paper with a rate-distortion sticker on it. A committee reads "we measured the encoder bottleneck on VITON-HD and DressCode" as measurement, not theory — and measurement of a known bottleneck on `vton-hd-dataset [UNVERIFIED — not in graph.db]`-class data is application layer. The swap-test trap is subtle here: an information bound on lossy conditioning is *not* try-on-specific [UNVERIFIED — not in graph.db] by default, which cuts both ways — it helps the generality argument but means the try-on framing is decorative unless the garment structure (spatially concentrated high-frequency detail under a known mask) is what makes the bound tight or estimable. If you cannot show the try-on structure is load-bearing in the *proof*, you have written a generic information-theory note dressed in garments. Scoop risk is moderate, not low: the bottleneck is named out loud in the abstracts above, so others see it too.

### 3. Constrained-sampling theory under hard garment-pinning

**Thesis.** Standard sampler theory assumes free sampling — every coordinate evolves under the learned dynamics. Try-on violates this: inpainting-style methods *pin* the garment region as a hard constraint and sample only the complement. Nobody has characterized how a hard per-step projection changes the sampling dynamics, the divergence behavior, and the stability region. This is where Aek's machinery is the right tool, not a decoration: `diffusion-sampling-with-momentum-for-mitigating-divergence-artifacts` gives you Heavy Ball Momentum and GHVB with explicit stability regions, and the dossier hands you the precise crack — "when β<1, HB drops to first-order convergence, significantly decreasing image quality," and "countless unexplored ways exist to expand numerical method stability regions beyond HB and GHVB." A hard garment constraint *is* a perturbation to that stability region. `adjoint-schr-dinger-bridge-sampler` and `glass-flows-transition-sampling-for-alignment-of-flow-and-diffusion-models` give you the constrained-bridge and transition-sampling vocabulary to state the result cleanly. The claim: characterize how hard-constraint pinning shifts the divergence-artifact boundary, and derive the corrected momentum schedule that restores stability under the constraint.

**Takedown.** This is the most defensible and the most likely to *finish*, which is exactly why it is also the least flashy — a committee will not gasp. Theory load is a 2 because you are perturbing an existing, well-posed stability analysis rather than building one from scratch, but that same fact caps the ceiling at 4: "we extended GHVB's stability region to the constrained case" is a clean, real, finishable result, not a field-defining one. The real risk is that the constraint turns out to perturb the dynamics *trivially* — if hard pinning just restricts the sampler to a lower-dimensional manifold with the same stability constants, there is no paper, and you will not know that until you have reproduced the momentum baseline and actually injected a mask. The Aek dependency is also a scheduling hazard: the phase rule forbids contacting him until the four deep-reads finish, so for now you must carry his machinery solo from the papers alone. Do not let "Aek is the testbed" drift into "Aek is the thesis" — the thesis is the constrained-sampling characterization; his momentum is the instrument.

### 4. Video-VTON appearance-vs-motion objective tradeoff

**Thesis.** Video try-on optimizes two objectives that pull against each other — per-frame garment appearance fidelity and temporal/motion coherence. The community treats this as a loss-weighting knob: `3dv-ton-textured-3d-guided-consistent-video-try-on-via-diffusion-models`, `chronotailor-harnessing-attention-guidance-for-fine-grained-video-virtual-try-on`, `dpidm-temporal-consistent-video`, `dreamvvt-mastering-realistic-video-virtual-try-on-in-the-wild-via-a-stage-wise-diffusion-transformer-framework`, and `vivid-video-virtual-try-on-using-diffusion-models` each pick a point on the curve. The proposed paper formalizes the tradeoff frontier.

**Takedown.** Skip it. Ceiling is CVPR-level applied vision and the admissions signal is a 2 — this is the most swap-test-failing [UNVERIFIED — not in graph.db] direction on the menu, because "appearance vs temporal-coherence tradeoff" is a generic video-generation tension that try-on does not make special. Worse, it is the highest-cost direction to even enter: video pipelines are heavy, your 4h/week evaporates on data and compute plumbing before you reach a single theoretical claim, and the result still reads as "better video try-on." It builds on your strength, yes, and that is the trap — it is comfortable and it does not move you toward the stated goal. Listed only so the matrix is complete.

### 5. Memorization-as-a-feature framing (umbrella)

**Thesis.** A wrapping narrative: reframe memorization across diffusion from "failure to avoid" to "resource to allocate," with `neon-negative-extrapolation-from-self-training-improves-image-generation` (push *away* from self-generated samples), the creativity theory in `an-analytic-theory-of-creativity-in-convolutional-diffusion-models`, and the dynamical-regularization view in `why-diffusion-models-don-t-memorize-the-role-of-implicit-dynamical-regularization-in-training` as the three legs. Try-on is the cleanest place this framing bites because it is the task that *wants* memorization locally.

**Takedown.** It does not stand alone. An umbrella framing with no new theorem and no new estimator is a position paper, and position papers from a not-yet-PhD applicant read as ambition without a result — the worst possible signal for the "had an idea and executed it" bar. Its only legitimate use is as the *abstract framing* over Direction 1, where it earns its keep by giving the spatially-varying result a reason to exist. On its own, theory load 4 with nothing forced to close, clean-exec 2 because there is no crisp finish line. Fold it into 1; do not ship it.

## Ranking

**1 > 3 > 2 > 5 > 4.**

1. **Direction 1 (selective memorization)** — highest ceiling and best admissions signal, but only rational with a theory co-author and more than 4h/week; solo it is a year-killer.
2. **Direction 3 (constrained-sampling under garment-pinning)** — best risk-adjusted bet; defensible, finishable at 4h/week solo, and the one where Aek's `diffusion-sampling-with-momentum-for-mitigating-divergence-artifacts` machinery is genuinely the right instrument.
3. **Direction 2 (rate-distortion ceiling)** — real generality if the bound is defensible, but degrades to a benchmark paper the moment the estimator is hand-wavy.
4. **Direction 5 (memorization-as-a-feature)** — valuable only as the framing layer over Direction 1; cannot stand alone.
5. **Direction 4 (video tradeoff)** — CVPR ceiling, weakest admissions signal, fails the swap test; skip.

**Fork rule.** Theory co-author secured **and** more than 4h/week → swing for Direction 1, using Direction 5 as its framing. Mostly solo at 4h/week → ship Direction 3. Either way, run the 4-week falsification probe below before committing the year.

## Execution

Top-ranked finishable bet is **Direction 3**. First-week proof-of-concept, concrete:

**Reproduce.** Stand up the momentum sampler from `diffusion-sampling-with-momentum-for-mitigating-divergence-artifacts` — Heavy Ball plus GHVB — on a small pretrained latent diffusion checkpoint. You are not training anything. Goal: reproduce the divergence-artifact behavior and confirm the stated stability cliff, that β<1 collapses HB to first-order convergence with visible quality loss. If you cannot see that cliff by end of week 1, the instrument is not under your control and nothing downstream is trustworthy.

**Toy model.** Build the smallest possible constrained sampler: a 2D or low-dimensional synthetic diffusion where you can *pin* one coordinate block (the stand-in for the garment region) to a fixed target at every step and sample the complement freely. Measure the divergence boundary and the effective convergence order **with** and **without** the pin. The whole thesis lives or dies on one question this toy answers: *does the hard constraint shift the stability region nontrivially, or does it leave the constants unchanged?*

**Gate condition (GATE 1, week 2).** If by end of week 2 you have (a) reproduced the momentum baseline's stability cliff and (b) shown the pin moves the divergence boundary measurably in the toy model — proceed; you have a real perturbation to characterize, and Direction 3 is live. If the pin perturbs the dynamics trivially (same stability constants, garment block just restricts to a sub-manifold), **pivot**. The pivot target is Direction 2 (rate-distortion ceiling), where the encoder bottleneck named in `art-viton-measurement-guided-latent-diffusion-for-artifact-free-virtual-try-on` and `texture-preserving-diffusion-models-for-high-fidelity-virtual-try-on` gives you an independent, still-defensible problem that does not depend on the constraint being dynamically interesting.

**GATE 2 (week 4).** If GATE 1 passed, decide: escalate toward Direction 1 only if a theory co-author has materialized; otherwise lock Direction 3 and start writing the constrained-stability result. Aek stays dark until all four deep-reads are done — the machinery is yours to carry from the papers until then.
