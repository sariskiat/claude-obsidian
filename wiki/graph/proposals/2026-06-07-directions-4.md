## The bar

You are not being asked to write a good paper. You are being asked to write a paper that makes an admissions committee believe the idea was yours. Those are different targets, and most of the bridge candidates in the dossier optimize the first while quietly failing the second. Three truths, stated without cushioning.

**One. A top-tier paper is necessary, not sufficient, and most of them do not buy the ticket.** A committee at the places you are aiming can separate "competent engineer applied a known method to a new dataset" from "person who had an idea." Six of the twelve bridges in your dossier are the first kind wearing a theory costume. Bridge 7 (`3dv-ton-textured-3d-video` ↔ `catvton-concatenation-is-all-you-need-for-virtual-try-on-with-diffusion-models`) and Bridge 2 (`Neon (Negative Extrapolation)` ↔ `DPIDM`) scored 0.57–0.70 on cosine proximity, but proximity in an embedding graph is not a research idea. It is two literatures that use similar words. The graph cannot tell the difference. You have to.

**Two. To a theory committee, `Virtual Try-On` is an application.** A pure try-on result lands at CVPR/ICCV/WACV and reads as applied vision no matter how clean it is. The dossier is heavily weighted toward bridges whose VTON side is the *point* (Bridges 1, 5, 7, 12). Every one of those, if executed as written, is an application paper. The only bridges that survive the swap test are the ones where try-on is the *testbed for a general claim about sampling or memorization* — Bridges 3, 4, 6, and the constrained-sampling reading of Bridge 1. If a researcher who has never heard of garments could write your paper by swapping in CIFAR, you wrote an application paper.

**Three. The letter outweighs the paper.** A top admit rides on a recommender the committee already trusts saying the ideas were yours. This is the real reason the Aek collaboration matters, and the real reason the phase rule exists: `Heavy Ball Momentum` / GHVB is a *tool you borrow*, not a co-authorship you cash early. Bridge 9 and Bridge 11 (the `GraphRAG` / `Knowledge Graph` bridges) are not directions at all — they are your own tooling showing up in its own output. Ignore them as research; they are the lab notebook, not the experiment.

Now the menu. Harsh on purpose, because picking the flashiest row costs you a year.

## Decision matrix

Ratings 1–5. For **Theory load** and **Scoop risk**, higher = heavier / worse. For every other column, higher = better for the goal.

| # | Direction | Ceiling | Clean-exec odds @4h/wk solo | Theory load | Scoop risk | Builds on strength | Admissions signal |
|---|-----------|:------:|:----:|:----:|:----:|:----:|:----:|
| 1 | Selective / spatially-varying memorization for try-on | 5 | 2 | 5 | 5 | 4 | 5 |
| 2 | Rate-distortion ceiling on garment-detail transfer | 4 | 3 | 4 | 3 | 5 | 4 |
| 3 | Constrained-sampling theory under hard garment-pinning | 4 | 3 | 4 | 2 | 5 | 4 |
| 4 | Video-VTON appearance-vs-motion objective tradeoff | 3 | 4 | 2 | 4 | 4 | 2 |
| 5 | Memorization-as-a-feature framing (umbrella) | 4 | 2 | 4 | 4 | 3 | 4 |

A scoop risk of 5 means a lab with a real theorist and 40 hours a week can do it first, and will.

### 1. Selective / spatially-varying memorization for try-on

The thesis is the best one on the table and the one most likely to eat your year. Try-on is the one image task where the *correct* behavior is regime-split by region: the garment must be copied (memorization regime), the body and background must be synthesized (generation regime). The dossier hands you the theory side cleanly — Bridge 4 connects `Neon (Negative Extrapolation)` and `Regularized Parameter Scaling` to the `Equivariant Local Score (ELS) Machine` and `DiffusionTrend` community, and the load-bearing limitation lifted from `an-analytic-theory-of-creativity-in-convolutional-diffusion-models` is exactly the hinge: *"ideal score machine can only memorize training data."* That paper gives you a locality-and-equivariance account of *when* a convolutional diffusion model copies versus creates. Try-on is a natural experiment where you know the ground-truth regime map per pixel. That is a real idea. It passes the swap test outright — the contribution is "memorization is a spatially controllable resource," and garments are merely the cleanest place to measure it.

**Takedown:** You cannot finish this solo at 4h/week and you should not pretend otherwise. The theory load is a genuine 5 — you need to extend a locality-scale creativity theory into a *spatially-varying* regime, which means a real estimator for the local memorization/generation boundary, not a hand-wave. That is ergodic/measure-theoretic muscle the profile explicitly lists as your gap. Scoop risk is also a 5: the `an-analytic-theory-of-creativity-in-convolutional-diffusion-models` authors, or any lab adjacent to them, can connect this to conditional generation in a single sprint and will not need your garments to do it. If you start here without a theory co-author and more than 4h/week, the realistic outcome is a half-formalized intuition and an empirical demo that reviewers read as Direction 5 with extra steps. This is the swing, not the default.

### 2. Rate-distortion ceiling on garment-detail transfer

Three independent VTON papers hit the *same wall* and the dossier caught it. Bridge 7's extracted limitation is blunt: *"CLIP-based textual inversion fails to preserve fine garment details due to CLIP encoder bottleneck,"* and Bridge 1 independently surfaces *"pre-trained image encoders like DINOv2 and CLIP are not optimized for detail preservation — critical for VTON."* When `catv2ton-taming-diffusion-transformers-for-vision-based-virtual-try-on-with-temporal-concatenation`, `art-viton-measurement-guided-latent-diffusion-for-artifact-free-virtual-try-on`, and the `OOTDiffusion`/`CatVTON` line all bottleneck at the same encoder, that is not three engineering failures — it is a candidate information-theoretic ceiling. The paper: formalize the rate–distortion bound on how much garment detail a fixed pretrained encoder can pass to the denoiser, derive where current encoders sit relative to it, and show the gap is structural rather than a tuning problem. Bridge 8 gives you the compression vocabulary for free — `Auto-Compressing Networks (ACNs)` and `Conditional Flow Matching (CFM)` sit one hop away.

**Takedown:** This degrades into a benchmark paper the moment your lower bound is not defensible. A rate–distortion *ceiling* with no real estimator and no achievability argument is a plot of FID-vs-encoder-size with a Greek letter taped on, and reviewers at your target venues have seen that trick. The hard part is not noticing the bottleneck — every VTON author already noticed it, which is *why it is in the limitations sections you extracted*. The hard part is a bound that is tight enough to be falsifiable and general enough to not be "CLIP is lossy, news at eleven." Builds on your strength the most (5) precisely because it is closest to engineering, which is also the warning: the path of least resistance here is the application-layer version, and you will feel productive the whole way down it.

### 3. Constrained-sampling theory under hard garment-pinning

This is the one to ship if you are solo. Generic sampler theory assumes *free* sampling — every coordinate evolves under the same dynamics. Try-on inpainting violates that assumption: the garment region is hard-pinned while the rest samples freely. Nobody has characterized what a hard Dirichlet-style constraint does to sampling stability and divergence, and you have the exact tool in-house. Bridge 1 and Bridge 3 connect `Heavy Ball Momentum` / `GLASS Flows` / `DiffSDA` to the VTON communities, and the recurring extracted limitation — *"When β<1, HB drops to first-order convergence, significantly decreasing image quality"* — is precisely a statement about momentum-sampler convergence order that *changes* under a pinned boundary. Bridge 6 adds `adjoint-schr-dinger-bridge-sampler`, which is the right formalism for "sample one marginal while fixing part of the state." The contribution is general: *how a hard inpainting constraint reshapes the divergence-artifact regime of a momentum sampler*, with `diffusion-sampling-with-momentum-for-mitigating-divergence-artifacts` as your baseline and try-on as the testbed. Swap test: passes — the claim is about constrained samplers, garments are the boundary condition.

**Takedown:** The danger is the opposite of Direction 1 — not too hard, too *modest*. If you state it small ("inpainting changes the effective step size, here is a correction term"), you get a workshop paper, not a NeurIPS ticket. Scoop risk is genuinely low (2) because the intersection of momentum-sampler theory and try-on inpainting is white space — the profile confirms zero claim-edges between the VTON communities and Aek's. But low scoop risk often means low excitement; you have to make the constrained-divergence result *surprising*, not merely correct. And the phase rule is load-bearing: you do **not** email Aek until all four deep-reads are done, because if you borrow the momentum machinery before you can derive the β<1 convergence drop yourself, the letter says "ran someone else's sampler," which is the worst possible read for an independence signal.

### 4. Video-VTON appearance-vs-motion objective tradeoff

Bridge 2 (`Neon (Negative Extrapolation)` ↔ `3DV-TON`/`ChronoTailor`/`DPIDM`) and Bridge 12 (`MDM (Motion Diffusion Model)` / `Diffusion Noise Optimization (DNO)` ↔ `Virtual Try-On`) point at a real tension: video try-on must hold garment *appearance* fixed across frames while obeying *motion* dynamics, and those objectives fight. Bridge 10 even hands you the motion-side theory via `deep-compositional-phase-diffusion-for-long-motion-sequence-generation`.

**Takedown:** Skip it. CVPR-ceiling, weak admissions signal (2) — this is applied video vision wearing a tradeoff hat, and the swap test fails because the contribution lives in try-on's specific appearance/motion split rather than a general principle. The extracted limitations are all deployment complaints (*"deploying high-res VTON on terminal devices... remains unexplored"*), which is a tell: this community's open problems are engineering, not theory. High clean-exec odds (4) is a trap — easy to finish, easy to publish, does not move the ticket. Do not let "I could actually ship this" override "it does not signal taste."

### 5. Memorization-as-a-feature framing (umbrella)

The reframe — memorization is a controllable resource, not a failure mode — is the right *wrapper* for Direction 1 and gives Direction 3 a narrative home. `Neon (Negative Extrapolation)` and `Universal Inverse Distillation (UID)` (Community 2) supply the "memorization is sometimes the goal" prior.

**Takedown:** It does not stand alone and you know it. A framing paper with no theorem is a position piece, and position pieces do not get a first-author into a top theory group — they read as taste without proof, which is exactly the half of the signal a committee discounts. Ceiling 4 is generous and assumes it is *attached* to a real result. On its own the clean-exec odds (2) reflect that there is nothing to execute. Use it as the intro paragraph of #1 or #3, never as the paper.

## Ranking

**3 > 1 > 2 > 5 > 4**, against the goal at your current resources.

1. **#3 — Constrained-sampling under garment-pinning.** Best risk-adjusted ticket: white-space scoop risk, your in-house momentum tool, a general claim that survives the swap test, finishable solo at 4h/week.
2. **#1 — Selective memorization.** Highest ceiling and best signal, but theory load 5 and scoop risk 5 make it a co-author-only swing.
3. **#2 — Rate-distortion ceiling.** Defensible and on-strength, but degrades to a benchmark paper without a real lower bound and achievability argument.
4. **#5 — Memorization-as-feature.** A wrapper, not a paper. Bolt it onto #1 or #3.
5. **#4 — Video appearance/motion.** Skip. CVPR-ceiling, application-layer, fails the swap test.

**Fork rule.** Theory co-author secured *and* more than 4h/week → swing for **#1**, with **#5** as its framing. Mostly solo at 4h/week → ship **#3**, and hold the Aek email until all four deep-reads are done (the phase rule — borrow the machinery only after you can re-derive the β<1 convergence drop yourself).

## Execution

Run a four-week falsification probe on **#3**, structured as Direction 3 → Direction 1 so a clean failure escalates instead of stranding you.

**Week 1 — reproduce the baseline.** Stand up the momentum sampler from `diffusion-sampling-with-momentum-for-mitigating-divergence-artifacts` and reproduce its divergence-artifact mitigation on the paper's own free-sampling setting. You must independently recover the reported effect — including the *"when β<1, HB drops to first-order convergence"* behavior — with no co-author input. This is the independence rehearsal for the letter.

**Toy model.** Build a 2-D constrained sampler: a Gaussian-mixture target where a subset of coordinates is *hard-pinned* (Dirichlet boundary) while the rest evolve under the momentum sampler. Measure empirical convergence order and divergence-artifact rate of the *free* coordinates as a function of momentum β and the pinned-fraction. Hypothesis to falsify: pinning changes the effective convergence order of the free region in a way the free-sampling theory does not predict. Then lift the same pinning operator onto a real inpainting mask from `art-viton-measurement-guided-latent-diffusion-for-artifact-free-virtual-try-on` or `catvton-concatenation-is-all-you-need-for-virtual-try-on-with-diffusion-models`, using the garment mask as the boundary, and check the toy prediction holds on the image sampler.

**GATE 1 (end of week 2).** Baseline reproduced *and* the toy shows a measurable, repeatable gap between pinned and free convergence behavior → continue to formalize the correction term. If the baseline is not reproduced by week 2, or the constrained/free gap is within noise, **pivot** — the constrained-sampling story has no teeth.

**GATE 2 (end of week 4).** If the gap is real but the correction term is shallow (a one-line effective-step-size rescale), *escalate* to **#1**: re-task the same pinned-region apparatus to measure the spatial memorization/generation boundary using the `an-analytic-theory-of-creativity-in-convolutional-diffusion-models` locality framework, with the garment mask now as a known regime map rather than a sampling constraint. If the correction term is deep and surprising, ship **#3** and only then consider the Aek email — deep-reads complete first.

---

*Grounding audit: 7/7 citations verified ✓ | retries: 0 | engine: 2.1.168 (Claude Code) | bridge candidates: 12*
