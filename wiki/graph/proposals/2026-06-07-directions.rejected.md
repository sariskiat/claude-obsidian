# Directions Report — REJECTED (grounding cap exhausted)

**Unverified citations:** ton-textured-3d-guided-consistent-video-try-on-via-diffusion-models

---

## The bar

You want a first author paper that buys a ticket into a top tier PhD. So clear the bar the committee actually uses, not the one that feels good. Three uncomfortable truths, stated plainly because picking wrong here costs you a year.

**Truth 1 — a top venue paper is necessary, not sufficient, and not all of them count.** Committees at the places you are aiming at can tell apart "competent engineer who applied a known method to a new dataset" and "person who had an idea." Your fork rule already concedes the last paper was closer to the first kind. Another competent application, even at CVPR, mostly re confirms that read. The paper has to signal taste and independence or it does not buy anything.

**Truth 2 — to a theory leaning committee, VTON is an application area.** A pure try on paper lands at CVPR / ICCV / WACV and reads as applied vision. The only way VTON serves your stated goal is if the contribution is a general insight that try on happens to be the cleanest testbed for. Apply the swap test ruthlessly: if a researcher who has never heard of garments could write your paper by swapping the dataset, it is application layer and it does not earn the ticket. Every direction below lives or dies on that test.

**Truth 3 — the letter matters as much as the paper.** A top admit usually rides on a recommender the committee trusts saying the ideas were yours. That is the real reason a theory collaborator matters: not the extra hands, the letter. It is also why your Aek phase rule exists — do not spend that relationship on a half result. Note up front: the claim graph shows the VTON communities have zero claim edges into Aek's sampling / momentum / distillation communities. That white space is the opportunity and the warning. White space means nobody has bridged it; it does not mean the bridge is easy or that it is yours by default.

## Decision matrix

Ratings 1 to 5. For every column except Theory load and Scoop risk, higher is better for the goal. For Theory load, higher means a heavier lift you currently cannot fully carry solo. For Scoop risk, higher means a powerhouse lab can and might do it first.

| # | Direction | Ceiling | Clean exec odds @4h/wk solo | Theory load | Scoop risk | Builds on strength | Admissions signal |
|---|-----------|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | Selective / spatially varying memorization for VTON | 5 | 2 | 5 | 5 | 4 | 5 |
| 2 | Rate distortion ceiling on garment detail transfer | 4 | 3 | 4 | 3 | 5 | 4 |
| 3 | Constrained sampling theory under hard garment pinning | 4 | 3 | 4 | 2 | 5 | 4 |
| 4 | Video VTON appearance vs motion objective tradeoff | 3 | 4 | 2 | 4 | 4 | 2 |
| 5 | Memorization as a feature framing (umbrella) | 4 | 2 | 4 | 4 | 3 | 4 |

### 1. Selective / spatially varying memorization for VTON

**Thesis.** Try on is the rare task where the correct generative regime is different in different image regions. The garment region must be in a memorization regime — copy the exact weave, print, and seam — while body and background must be in a synthesis regime. `an-analytic-theory-of-creativity-in-convolutional-diffusion-models` gives you the handle: it shows the memorize versus create balance is governed by locality scale and equivariance, and its Equivariant Local Score (ELS) Machine is an analytic object you can actually compute against, not a vibe. The contribution is to turn memorization from a bug into a spatially controllable resource, with garment masks as the control field. This passes the swap test cleanly: the structural fact that one output is a hard copy region embedded in a free region is intrinsic to try on, not borrowed.

**Takedown.** This is the highest ceiling and the one most likely to eat your year. The theory load is a 5 and you have marked information theoretic and ergodic argument as your gap, not your strength. The ELS construction in `an-analytic-theory-of-creativity-in-convolutional-diffusion-models` is exactly the machinery you have not yet built muscle for, and a referee will demand you extend it, not cite it. Scoop risk is also a 5: Ganguli style labs are sitting on this theory and can formalize spatially varying regimes faster than you can solo at 4h/week. Worst case you spend six months re deriving their setup, produce a result that looks like a corollary, and a committee reads it as "applied someone else's theory to garments" — which is the exact failure mode Truth 1 warns about. This direction is only honest to attempt with a real theory collaborator and more than 4h/week. Without both, do not start it; start Direction 3 and let the gate escalate you here.

### 2. Rate distortion ceiling on garment detail transfer

**Thesis.** Three independent VTON papers hit the same wall from different sides: pretrained encoders are not built to preserve fine garment detail. `art-viton-measurement-guided-latent-diffusion-for-artifact-free-virtual-try-on` and `catv2ton-taming-diffusion-transformers-for-vision-based-virtual-try-on-with-temporal-concatenation` both surface it, and CatVTON / OOTDiffusion live inside the same bottleneck — CLIP and DINOv2 latents simply do not carry the bits. The thesis is to formalize this as a rate distortion ceiling: derive how many bits of garment texture survive a given latent channel, show where each current encoder sits relative to that ceiling, and turn an empirical complaint into a bound. That is a general insight about conditioning bottlenecks in latent generative models; garments are just the cleanest stress test.

**Takedown.** This is the most seductive trap on the menu because it sounds rigorous while letting you stay in your comfort zone. The moment you cannot produce a defensible estimator and a real lower bound, it degrades into a benchmark paper: "we measured detail loss across five encoders." That is application layer with a theory costume and a committee will see through it. The hard part is the bound, not the measurement, and the bound is genuinely information theoretic — your stated gap again. Scoop risk is moderate rather than high only because nobody finds it glamorous, which is also a signal that the upside is capped. If you take this, the success condition is a clean ceiling with a matching estimator on day one; if week 3 has you tuning evaluation metrics instead of proving a bound, you have already lost.

### 3. Constrained sampling theory under hard garment pinning

**Thesis.** Generic sampler theory assumes free sampling — every coordinate of the output is generated. Try on does not work that way: the garment region is hard pinned to a measured reference while the rest is sampled. That changes the sampling dynamics, the stability region, and the divergence behavior, and nobody has characterized it. This is where Aek's machinery is the right tool. `diffusion-sampling-with-momentum-for-mitigating-divergence-artifacts` gives the Heavy Ball Momentum and GHVB analysis whose stated limitation — when β drops below 1, the scheme falls to first order convergence and image quality degrades — is exactly the regime a hard constraint perturbs. `adjoint-schr-dinger-bridge-sampler` and `bridge-synthesis` give the boundary locus framing. The contribution: characterize how a hard pinning constraint reshapes the sampler's stability and divergence, with a constrained variant of the momentum analysis. Tested on a pinned VTON inpainting setup over CatVTON, but the result is about constrained samplers in general.

**Takedown.** This is the most defensible and best risk adjusted option, and it is still not safe — do not let the lower scoop risk lull you. Theory load is a 2 relative to the others only because it is stability and dynamics analysis, the exact muscle your profile says you are actively building, not the ergodic theory you lack. The failure mode here is subtler than the others: it is shipping a result that is true but reads as incremental — "we added a constraint to a known momentum sampler." To clear the bar, the constrained analysis must reveal something the free sampling theory cannot predict (a new instability, a shifted stability boundary, a divergence the unpinned sampler never sees), not just re run the existing bound with a mask. If your draft's main theorem is "the same as Heavy Ball Momentum but with a projection step," kill it. Also: respect your own phase rule. Aek's momentum and boundary locus machinery is the testbed, not the thesis, and you do not email him until all four Aek deep reads are done. A premature ask spends the letter on a sketch.

### 4. Video VTON appearance vs motion objective tradeoff

**Thesis.** Video try on couples a per frame appearance objective against a temporal motion objective, and those two pull apart. `3dv-ton-textured-3d-guided-consistent-video-try-on-via-diffusion-models [UNVERIFIED — not in graph.db]`, `chronotailor-harnessing-attention-guidance-for-fine-grained-video-virtual-try-on`, and `dpidm-temporal-consistent-video` (3DV-TON, ChronoTailor, DPIDM) each negotiate this tradeoff with a different hack. The thesis would formalize the tradeoff as a two objective frontier.

**Takedown.** Skip it. Ceiling is a 3 and the admissions signal is a 2 — the worst on the board for your stated goal. A video try on paper lands at CVPR and reads as applied vision no matter how you dress the framing, which means it confirms Truth 2 instead of beating it. The tradeoff framing is real but it is also obvious to everyone in that subfield, so you are competing on engineering polish at 4h/week against labs with GPU farms and full time students. This is the direction that feels productive and buys you nothing toward the ticket. The only world where it makes sense is if your goal quietly shifts from a theory leaning PhD to an industry vision team — and that is not the goal on record.

### 5. Memorization as a feature framing (umbrella)

**Thesis.** Reframe memorization across the board from failure to controllable resource. `neon-negative-extrapolation-from-self-training-improves-image-generation` (Neon (Negative Extrapolation)) and `universal-inverse-distillation-for-matching-models-with-real-data-supervision-no-gans` (Universal Inverse Distillation (UID)) show the field already treating the training versus data relationship as something to push on deliberately; `an-analytic-theory-of-creativity-in-convolutional-diffusion-models` supplies the memorize versus create axis. The umbrella ties Direction 1 into a larger story.

**Takedown.** It does not stand alone and you should not pretend it can. As a standalone paper it is a position piece, and position pieces from someone without a track record read as overreach — exactly the wrong signal. Its honest job is to be the framing wrapper around Direction 1 once Direction 1 has a hard result. Used that way it raises the ceiling of the whole program; used on its own it is hand waving. Do not write this first.

## Ranking

Ranked by expected value toward the ticket: **1 > 3 > 2 > 5 > 4.**

1. **Direction 1** — highest ceiling and best signal, but only real with a theory collaborator and more than 4h/week; otherwise it is a year sink.
2. **Direction 3** — the most defensible, best risk adjusted, finishable solo at 4h/week, and it builds the exact theory muscle you are growing.
3. **Direction 2** — strong in principle, but collapses to a benchmark paper without a real bound, and the bound is your weakest area.
4. **Direction 5** — valuable only as the framing wrapper around Direction 1, never standalone.
5. **Direction 4** — skip; CVPR ceiling, weak admissions signal, wrong target for the stated goal.

**Fork rule.** Theory collaborator secured AND more than 4h/week → swing for Direction 1, with Direction 5 as the framing. Mostly solo at 4h/week → ship Direction 3, full stop. The 4 week falsification probe runs Direction 3 first and lets the gate decide whether you have earned the right to escalate to Direction 1.

## Execution

Top ranked for the realistic solo case is **Direction 3**. First week proof of concept, structured to falsify fast.

**Reproduce first.** Stand up the momentum sampler from `diffusion-sampling-with-momentum-for-mitigating-divergence-artifacts` — the Heavy Ball Momentum / GHVB scheme — at its published settings and confirm you can see its divergence mitigation on an unconstrained diffusion sampling task. You are not innovating yet; you are proving you can drive the tool. If you cannot reproduce the baseline behavior, nothing downstream is trustworthy.

**Toy model to build.** A low dimensional constrained sampling testbed: a 2D (or small nD) diffusion where one coordinate is hard pinned to a fixed value at every step (the stand in for the pinned garment region) and the rest is sampled freely. Run the momentum sampler with and without the pin. Measure the stability boundary in β and step size, and the divergence behavior, pinned versus free. The single question: does the hard pin move the stability boundary or create a divergence the free sampler never shows? A yes — even a small, clean yes — is the seed of a real theorem. A no means the constraint is benign and the direction is dead in this form.

**Gates.**
- **Gate 1 (end of week 2).** Baseline reproduced AND the toy model runs cleanly. Miss either and pivot — do not push into week 3 on a sampler you cannot drive.
- **Gate 2 (end of week 4).** If the pinned toy shows a genuine, characterizable departure from free sampling theory: escalate — this is now a real constrained sampling result, and if a theory collaborator is in reach it can climb toward Direction 1. If the departure is real but modest: ship Direction 3 as scoped, solo. If the pin is provably benign: fall back to Direction 2 and attack the encoder bottleneck bound in `art-viton-measurement-guided-latent-diffusion-for-artifact-free-virtual-try-on` instead.

Hold the line on the phase rule throughout: Aek's machinery is the testbed here, and no email goes out until the four deep reads are done and the toy model has cleared Gate 1.
