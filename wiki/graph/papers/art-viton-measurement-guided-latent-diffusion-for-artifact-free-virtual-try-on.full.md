---
type: paper-fulltext
slug: art-viton-measurement-guided-latent-diffusion-for-artifact-free-virtual-try-on
arxiv_id: null
source_path: /Users/saris.kia.adm/.paper-scholar/art-viton-measurement-guided-latent-diffusion-for-artifact-free-virtual-try-on/2509.25749.md
paper: "[[art-viton-measurement-guided-latent-diffusion-for-artifact-free-virtual-try-on]]"
---
<!-- extracted by afk_extract from 2509.25749.pdf (22p) -->

## - ART-VITON: MEASUREMENT-GUIDED LATENT DIF FUSION FOR ARTIFACT-FREE VIRTUAL TRY-ON 

Junseo Park and Hyeryung Jang 

Department of Computer Science & Artificial Intelligence, Dongguk University 

## ABSTRACT 

Virtual try-on (VITON) aims to generate realistic images of a person wearing a target garment, requiring precise garment alignment in try-on regions and faithful preservation of identity and background in non-try-on regions. While latent diffusion models (LDMs) have advanced alignment and detail synthesis, preserving non-try-on regions remains challenging. A common post-hoc strategy directly replaces these regions with original content, but abrupt transitions often produce boundary artifacts. To overcome this, we reformulate VITON as a linear inverse problem and adopt trajectory-aligned solvers that progressively enforce measurement consistency, reducing abrupt changes in non-try-on regions. However, existing solvers still suffer from semantic drift during generation, leading to artifacts. We propose ART-VITON, a measurement-guided diffusion framework that ensures measurement adherence while maintaining artifact-free synthesis. Our method integrates residual prior-based initialization to mitigate training-inference mismatch and artifact-free measurement-guided sampling that combines data consistency, frequency-level correction, and periodic standard denoising. Experiments on VITON-HD, DressCode, and SHHQ-1.0 demonstrate that ART-VITON effectively preserves identity and background, eliminates boundary artifacts, and consistently improves visual fidelity and robustness over state-of-the-art baselines. 

## 1 INTRODUCTION 

Virtual try-on (VITON) aims to synthesize photorealistic images of a person wearing a desired garment, enabling personalized and immersive online shopping experiences. Given a person image and clothing item, the system must align the garment to the body (try-on regions) while preserving identity (e.g., face, hair) and background (non-try-on regions). Despite progress in generative models, this task remains challenging due to two requirements: precise garment alignment and faithful preservation of non-try-on regions. Various approaches have been proposed to address these challenges (Han et al., 2018; Yu et al., 2019; Yang et al., 2020; Ge et al., 2021; Choi et al., 2021b; Xie et al., 2023; Morelli et al., 2023; Gou et al., 2023; Wang et al., 2024; Kim et al., 2024a; Choi et al., 2024), yet they have primarily focused on garment alignment, leaving the preservation of non-try-on regions largely underexplored. 

Early VITON methods (Han et al., 2018; Yu et al., 2019; Yang et al., 2020; Ge et al., 2021) relied on GAN-based two-stage pipelines with garment warping and synthesis networks, which improved alignment but suffered from sensitivity to warping accuracy, instability, and poor generalization due to limited garment-person diversity in existing datasets (Han et al., 2018; Choi et al., 2021b; Morelli et al., 2022). Recent diffusion models (DMs) (Ramesh et al., 2021; Rombach et al., 2022; Podell et al., 2024) address these issues with stable training, broader coverage, and flexible conditioning, achieving higher fidelity and stability. Two-stage approaches (Morelli et al., 2023; Wan et al., 2024) still rely on garment warping, while one-stage approaches (Kim et al., 2024a; Choi et al., 2024) eliminate warping by conditioning on garment features (via LoRA Hu et al. (2022), DreamBooth Ruiz et al. (2023)) or structural signals (via ControlNet Zhang et al. (2023), IP-Adapter Ye et al. (2023)). These advances largely resolve alignment challenges and enable more reliable, detailed synthesis. 

Despite significant progress in garment alignment, preserving non-try-on regions has been largely overlooked. Even when models are directly conditioned on such regions, they fail to fully preserve 

> Correspondence to: Hyeryung Jang 

1 

**==> picture [397 x 93] intentionally omitted <==**

Figure 1: Comparison of boundary artifacts across methods. StableVITON generates artifact-free outputs (A) but violates measurements (M). Post-hoc replacement enforces M but introduces seams A. Inverse solvers maintain M but accumulate semantic drift A. ART-VITON satisfies measurement constraints while remaining artifact-free. Green: success (measurement adherence or artifact-free); red: violations or artifacts. Solid/Dashed boxes show final/intermediate () outputs. 

non-try-on areas, resulting in distorted facial features, altered backgrounds, and reduced realism (see Fig. 1, second column; also Appendix Fig. 6). A common strategy (Yang et al., 2020; Xie et al., 2023; Gou et al., 2023) for preserving identity is based on post-hoc replacement, where the generated output is projected onto predefined masks or clothing-agnostic maps (Fig. 1, leftmost column) so that non-try-on regions are directly overwritten with original pixels. In this work, we refer to these masks as measurements. While intuitive, this approach often introduces boundary artifacts at region interfaces, manifesting as color mismatches, lighting inconsistencies, or broken textures (Fig. 1). The root cause is a spatial discontinuity: the generative model evolves freely during inference, unaware of the hard replacement that will occur afterward, resulting in abrupt transition once replacement is applied. 

To address the issue of images being generated without completely reflecting measurements, we formulate VITON as a linear inverse problem and integrate existing trajectory-aligned inverse solvers (Chung et al., 2024; Kim et al., 2025) into the latent diffusion model (LDM) sampling process. Compared to post-hoc methods, these solvers progressively guide the latent denoising trajectory, better adhering to measurements and enabling smooth transitions instead of abrupt region replacements. Nevertheless, these solvers can induce semantic inconsistencies between try-on and non-try-on regions during generation, potentially accumulating into boundary artifacts (Fig. 1, fourth column). This limitation highlights the need for a more robust solver that can maintain semantic coherence while satisfying measurements throughout the generation process. 

To mitigate semantic drift and enhance visual quality, we propose ART-VITON, a novel latent diffusion inverse solver that enforces measurement consistency during generation, yielding artifact-free synthesis. Our solver incorporates three key components: (i) data consistency, preserving semantic coherence and reducing drift, (ii) frequency-level correction, restoring high-frequency details lost during pixel-to-latent transition, and (iii) periodic standard denoising, leveraging prior knowledge to provide temporal alignment across regions. To avoid instability from direct trajectory manipulation and mitigate training-inference mismatch Lin et al. (2024), a residual prior is injected at initialization to maintain both stability and generative diversity. Operating externally without modifying the LDM, our framework is model-agnostic and applicable to diverse VITON pipelines (Fig. 2). Consequently, ART-VITON preserves non-try-on regions, improves garment alignment, eliminates boundary artifacts (Fig. 1), and demonstrates improved results on three benchmark VITON datasets. 

## 2 RELATED WORK 

## 2.1 IMAGE-BASED VITON METHODS 

Early VITON approaches primarily relied on GAN-based two-stage pipelines, where garmets were warped to align with target poses and then integrated into the person image. Pioneering works (Han et al., 2018; Yang et al., 2020) used geometric matching or thin-plate spline transformations, while later methods, including VITON-HD Choi et al. (2021b), HR-VITON Lee et al. (2022), and GPVTON Xie et al. (2023), extended this framework to high-resolution settings, improving detail preservation. Despite progress, these pipelines remained highly sensitive to warping errors, un- 

2 

stable during training, and limited in generalization, while still depending on post-hoc replacement for preserving identity, which introduced boundary artifacts. 

Latent diffusion models (LDMs) brought more stable training, better garment fidelity, and controllable synthesis. Two-stage pipelines (e.g., LaDI-VTON Morelli et al. (2023), DCI-VTON Gou et al. (2023), FLDM-VTON Wang et al. (2024), GarDiff Wan et al. (2024)) retain warping modules before diffusion, while one-stage methods bypass warping by encoding garment semantics (e.g., LoRA Hu et al. (2022), Textual Inversion Gal et al. (2023)) or injecting spatial cues through adapters (Zhang et al., 2023; Ye et al., 2023; Hu, 2024; Kingma & Welling, 2022). StableVITON Kim et al. (2024a) strengthens garment–human interaction via a zero cross-attention block in ControlNet Zhang et al. (2023), while Boow-VTON Zhang et al. (2025b) encodes garments with a Parallel U-Net Hu (2024) and integrates them into self-attention to enhance structural representation. DreamPaint Seyfioglu et al. (2023) binds garments to custom tokens using DreamBooth Ruiz et al. (2023). Yet, even with these advances, most LDM-based approaches still rely on post-hoc replacement for non-try-on regions, leaving spatial discontinuity at boundaries unresolved. 

## 2.2 DIFFUSION INVERSE SOLVERS 

Diffusion inverse solvers aim to integrate measurement constraints into the denoising process. Instead of conditioning on measurements alone, inverse solvers modify the sampling trajectory to align outputs with observations. Early works such as RePaint Lugmayr et al. (2022b) and ILVR Choi et al. (2021a) applied hard projection strategies on pixel-space, while Diffusion Posterior Sampling (DPS) Chung et al. (2023) adjusted sampling trajectories with measurement gradients and Measurement-Constrained Gradient (MCG) Chung et al. (2022) enforced projection onto measurement subspaces. Although these methods improve measurement adherence, they often distort denoising trajectories at high noise levels and accumulate semantic mismatches, producing boundary artifacts. Recent extensions to LDMs attempt to mitigate this. PSLD Rout et al. (2023) extends DPS into the latent domain, Resample Song et al. (2024) reintroduces noise after replacement in an MCG-manner, and TReg Kim et al. (2025) or DreamSampler Kim et al. (2024b) alternate between pixel- and latent-space refinements for stability. While effective in reducing abrupt post-hoc inconsistencies when inverse solvers are applied to VITON, these approaches still fail to maintain smooth semantic coherence between try-on and non-try-on regions, motivating the need for a solver tailored to artifact-free try-on synthesis. 

## 3 PRELIMINARIES 

## 3.1 LATENT DIFFUSION MODELS 

Latent Diffusion Models (LDMs) Rombach et al. (2022) perform the diffusion process in a compressed latent space, improving efficiency while preserving semantics. An input image x is encoded into a latent code via a pre-trained encoder , which is progressively perturbed into at timestep by adding Gaussian noise. At each step, a denoising network[; t;][ c][predicts] the noise added, conditioned on auxiliary inputs (e.g., garments, measurements, or text). Using Tweedie’s formula, the posterior latent estimate is: 

**==> picture [282 x 14] intentionally omitted <==**

where[is][the][cumulative][noise][scale.][Based][on][this,][the][DDIM][Lugmayr][et][al.][(2022a)][sampler] provides a deterministic update: 

**==> picture [276 x 13] intentionally omitted <==**

These iterative refinements produce high-quality samples while allowing for controllable conditioning. 

## 3.2 LINEAR INVERSE PROBLEMS 

Many imaging tasks, such as inpainting, super-resolution, and deblurring, can be cast as linear inverse problems, where the observed measurement y is a partial or degraded version of the 

3 

underlying image x 2 R . This is generally expressed as: 

**==> picture [254 x 12] intentionally omitted <==**

where is a linear operator and denotes additive Gaussian noise. The objective is to recover that both satisfies the measurements and remains consistent with the natural image distribution. Classical approaches impose explicit priors, while diffusion-based inverse solvers incorporate measurement constraints directly into the denoising process. 

## 4 METHOD 

## 4.1 REFORMULATING VITON AS AN INVERSE PROBLEM 

Virtual try-on requires generating a new garment in try-on regions while preserving identity and background in non-try-on regions. Let x be the target person image and y the observed non-try-on regions defined by a clothing-agnostic map (see Fig. 1). This forms a linear inverse problem Eq. 3, where A is a masking operator. The objective is to reconstruct x such that (i) measurements y are faithfully preserved, attributes of the reference garment are retained, and overall visual coherence is achieved. Since is provided to the model as a noise-free conditioning input, it is assumed noise-free, i.e., no noise n in Eq. 3. 

This perspective enables direct incorporation of measurement consistency into the sampling trajectory of LDMs, avoiding reliance on post-hoc replacement. Assuming a well-trained autoencoder D, the target image x is reconstructed from the latent vector z via x and clean latent estimate ^ in Eq. 1. The conditional distribution then factorizes as: 

**==> picture [203 x 14] intentionally omitted <==**

where the first term encourages semantic plausibility (garment fidelity and visual coherence), while the second enforces measurement preservation (non-try-on regions). Standard LDM inference does not explicitly enforce this balance: non-try-on regions evolve freely and are often corrected posthoc, introducing boundary seams. Existing inverse solvers enforce measurements y during sampling but often too rigidly, leading to semantic drift and boundary artifacts. We therefore introduce ARTVITON, which directly embeds measurement consistency into the sampling trajectory through two innovations: (a) prior-based initialization and (b) artifact-free measurement-guided sampling. 

## 4.2 PRIOR-BASED INITIALIZATION 

Diffusion models suffer from a rain-es isatch (Choi et al., 2022; Lin et al., 2024): during training, the noisiest latents z contain residual signals, while at inference, sampling often begins from pure Gaussian noise. This discrepancy degrades generation quality. Prior works attempted to mitigate this mismatch by mixing external guidance with noise to provide residual-based initialization - e.g., low-quality inputs in PASD (Yang et al., 2023) and SeeSR Wu et al. (2024) or warped predictions in DCI-VTON (Gou et al., 2023). However, even DDIM Lugmayr et al. (2022a) and VITON baselines (e.g., (Wan et al., 2024; Kim et al., 2024a)) commonly start from reduced timesteps (e.g., ) instead of the training setting ( ), further aggravating the gap, see Sec. 5.1. 

To address this, we propose a residual prior-based initialization z that reintroduces residual structure without extra modules or preprocessing. Specifically, we start from Gaussian noise and apply a single DDPM Ho et al. (2020) denoising step to obtain (see Fig. 2 (A)). This simple step injects subtle structural cues consistent with training dynamics while preserving stochasticity. By using as the initialization , inference trajectories align more closely with the model’s learned distribution, stabilizing sampling when measurement constraints are applied. 

## 4.3 ARTIFACT-FREE MEASUREMENT-GUIDED SAMPLING 

Naively enforcing measurements during denoising can preserve non-try-on regions but often introduces boundary artifacts, since rigid constraints disrupt semantic continuity. To balance measurement fidelity with artifact-free semantic plausibility, ART-VITON iteratively refines samples to converge toward a latent code ^ that satisfies the measurement constraint, by integrating following complementary techniques, as shown in Fig. 2. 

4 

**==> picture [397 x 189] intentionally omitted <==**

Figure 2: ART-VITON pipeline. (A) Residual prior-based initialization mitigates train-test mismatch. (B) Artifact-free measurement-guided inverse solver enforces measurements while preserving semantics: 1 Tweedie estimation retains clothing details but lacks fidelity in non-try-on regions. 2 Hard measurement constraints in pixel space correct preserved regions. High-frequency losses during 3 VAE encoding are compensated by 4 Data consistency and 5 Frequency correction (shown in (B-1)). (C) Periodic standard denoising realigns trajectories with data manifolds M[for] smooth blending. (B-2) visualizes this sampling trajectory. 

2 Hard measurement constraint. At each step, non-try-on regions (in pixel-space) are replaced with ground-truth measurements, directly enforcing p in Eq. 4 and ensuring faithful identity preservation: 

**==> picture [151 x 10] intentionally omitted <==**

where M is a binary mask ( for measurements) and z is initialized as . The updated image is then re-encoded to[,][which][aligns][the][latent][with][measurement][constraints][but][may] cause information loss, moving ^ away from the semantic trajectory (red line in Fig. 2 (B-2)). 

4 Data consistency. Hard measurement constraint in 2 is insufficient to preserve reference (garment) image attributes, leading to semantic inconsistencies across regions. Thus, focusing on y in Eq. 4, z is initialized with and optimized via TReg Kim et al. (2025), i.e., ^ is interpolated toward the reference-informed latent ^ in Eq. 1: 

**==> picture [309 x 28] intentionally omitted <==**

where E[denotes encoder reconstruction noise and] [0 1] controls the interpolation strength. 

5 High-frequency correction. While 3 resembles the true latent , it loses detail (e.g., textures and blurs) through VAE compression, which is usually fixed via retraining (Zhang et al., 2025a; Novitskiy et al., 2025; Almog et al., 2025). To tackle this, we construct a corrected latent[′][by injecting high-frequency components from the reference-informed latent][ ^] into ^ via perchannel Fourier transform. In non-try-on regions, this corrected latent replaces blurred details, while try-on regions directly retain ^ : 

**==> picture [331 x 15] intentionally omitted <==**

This selective refinement shaprpens preserved areas without disturbint garment synthesis, improving overall visual coherence. 

(C) Standard denoising. To avoid instability from repeated measurement-guided corrections, every steps we apply standard denoising steps, leveraging the diffusion model’s inherent ability to harmonize inter-region inconsistencies. This realigns trajectories with the LDM manifold and prevents 

5 

**==> picture [397 x 136] intentionally omitted <==**

Figure 3: Comparison of StableVITON baseline and inverse solvers on VITON-HD. (a) Highfrequency loss leads to texture degradation. (b) Boundary artifacts show inconsistencies at region interfaces. Hard-constraint methods (RePaint, MCG) produce sharp transitions; progressive updates (DPS, FIG) show incomplete convergence; and hybrid stochastic methods (DreamSampler, TReg) degrade texture fidelity. Our method preserves both texture fidelity and seamless boundaries. 

over-constrained solution, e.g., noisy latent z is guided to be positioned on the subsequent noisy manifolds (in Fig. 2 (B-2)) Overall, the complete pipeline alternates between measurement-guided updates (A)(B) and standard denoising (C), following the sequence: (A)(B)(C)(B)(C) : : : , ensuring both measurement consistency and visual fidelity throughout generation. 

## 5 EXPERIMENTS 

Dataset. We evaluate our method on three datasets: VITON-HD (Choi et al., 2021b), DressCode (Morelli et al., 2022), and SHHQ-1.0 (Fu et al., 2022). VITON-HD contains 647 training and 2 032 test pairs of frontal-view female upper-body images ( ). DressCode includes full-body images with upper/lower/dress items, totaling 15 , 8 951, and 2 947 pairs, with 1 8 test pairs per category ( ); we conduct experiments only on upper-body items. SHHQ-1.0 provides K high-quality full-body images ( ); for evaluation, we use the first 032 images, applying VITON-HD preprocessing to generate input conditions. 

Baselines. We compare against representative GAN-based (HR-VITON Lee et al. (2022), GPVTON Xie et al. (2023)) and recent LDM-based VITON models (LaDI-VTON Morelli et al. (2023), DCI-VTON Gou et al. (2023), GarDiff Wan et al. (2024), StableVITON Kim et al. (2024a)). We also benchmark state-of-the-art inverse solvers, categorized as: hard constraint (RePaint Lugmayr et al. (2022b), MCG Chung et al. (2022)), progressive update (DPS Chung et al. (2023), FIG Yan et al. (2025)), and hybrid stochastic (DreamSampler Kim et al. (2024b), TReg Kim et al. (2025)). Unless otherwise noted, all comparisons use post-hoc replacement, which is also required for hard constraint and progressive update solvers as they fail to fully preserve measurements. See Appendices A.2 and A.3 for details of VITON and inverse solvers. 

Evaluation Metric. We evaluate performance under two settings: paired, where the model reconstructs the original clothing, and unpaired, where the clothing is replaced. In the paired setting, we report PSNR and SSIM for pixel fidelity and structural consistency, and LPIPS for perceptual similarity. In the unpaired setting, we adopt FID to measure visual realism and global distributional coherence, and KID to assess sample diversity. 

## 5.1 IMPACT OF PRIOR-BASED INITIALIZATION 

Our prior-based initialization mitigates the train-test mismatch and consistently improves performance across all architectures (Table 1). By default, all baselines start denoising at : DCI-VTON overlays warped garments from its module, 

||Model|SSIM <br>PSNR <br>LPIPS <br>FID <br>KID|SSIM <br>PSNR <br>LPIPS <br>FID <br>KID|SSIM <br>PSNR <br>LPIPS <br>FID <br>KID|
|---|---|---|---|---|
||DCI-VTON Gou et al. (2023)<br><br><br><br><br><br>+ Prior @ <br>0.8880<br>24.1447<br>0.0782||11.4713|0.0011|
||GarDiff Wan et al. (2024)<br><br><br><br><br><br>+ Prior @ <br>0.8448<br>21.8611<br>0.0864<br>StableVITON Kim et al. (2024a)<br><br><br><br><br><br>+ Prior @ <br>0.8552<br>23.1475<br>0.0833||10.5322<br>10.4362|0.0034<br>0.0014|



Table 1: Effect of prior-based initialization at T across baseline models on VITON-HD. Our method consistently improves all metrics regardless of architecture.6 

|Method|SSIM <br>PSNR <br>LPIPS <br>FID <br>KID|SSIM <br>PSNR <br>LPIPS <br>FID <br>KID|SSIM <br>PSNR <br>LPIPS <br>FID <br>KID|SSIM <br>PSNR <br>LPIPS <br>FID <br>KID|SSIM <br>PSNR <br>LPIPS <br>FID <br>KID|SSIM <br>PSNR <br>LPIPS <br>FID <br>KID|
|---|---|---|---|---|---|---|
|StableVITON (baseline)|||||||
|RePaint Lugmayr et al. (2022b)<br><br><br><br><br><br>MCG Chung et al. (2022)<br><br><br><br><br><br>DPS Chung et al. (2023)<br><br><br><br><br><br>DreamSampler Kim et al. (2024b)<br><br>23.8984<br><br><br><br>FIG Yan et al. (2025)<br><br><br><br><br><br>TReg Kim et al. (2025)<br>0.8909<br><br><br><br>|||||||
|Ours||0.0746|9.7669||0.0009||



Table 2: Comparison of StableVITON with existing inverse solvers on VITON-HD, evaluated with identical (A) initialization and (C) denoising step in Fig. 2; only measurement-guided sampling step (B) differs. Red cells: performance degradation compared to the baseline; bold indicates the best, and underline the second-best. 

GarDiff initializes with pure Gaussian noise, and StableVITON uses noisy real images. Since StableVITON’s initialization is tailored for unpaired settings, we replaced z with pure noise for fair paired comparisons. Adjusting starting timestep alone already boosts performance, particularly for StableVITON (paired) and DCI-VTON. In unpaired settings, our residual prior-based initialization better fills masked regions with plausible structure, yielding sharper and more consistent garments, especially for StableVITON. GarDiff also shows notable gains, demonstrating the broad utility across architectures of our approach. 

## 5.2 COMPARISON WITH EXISTING INVERSE SOLVERS 

Our method achieves balanced improvements across all metrics without the trade-offs inherent in existing inverse solver approaches (Table 2). Unlike prior methods that boost one metric at the expense of another, ART-VITON consistently enhances both reconstruction fidelity and perceptual quality. As shown in Fig. 3, hard-constraint methods (RePaint Lugmayr et al. (2022b), MCG Chung et al. (2022)) tightly enforce measurements in latent space, which induce semantic drift and boundary seams between try-on and non-try-on regions. Despite this, measurements are not fully reflected, and post-hoc replacement cannot resolve the resulting inconsistencies. Progressive update methods (DPS Chung et al. (2023), FIG Yan et al. (2025)) provide smoother optimization but fail to fully satisfy measurements. Post-hoc correction is applied, and although smoother optimization mitigates its abrupt changes, spatial discontinuities and artifacts persist. 

Hybrid stochastic solvers (DreamSampler Kim et al. (2024b), TReg Kim et al. (2025)) inject stochastic noise to soften transitions, which artificially inflates structural scores (SSIM, PSNR) but disrupts deterministic sampling. This leads to degraded unpaired performance (FID, KID), inconsistencies such as missing buttons, and blurred textures (LPIPS) due to latent-to-pixel transitions (see Fig. 3, top row). In contrast, our approach maintains semantic alignment and fine-grained details throughout generation. As shown in Fig. 3a, our method preserves fine garment (high-frequency) details, achieving both measurement satisfaction and artifact-free synthesis in Fig. 3b. 

## 5.3 COMPARISON WITH VITON BASELINES 

VITON-HD results. As shown in Fig. 4a, baseline models exhibit boundary artifacts in gradient heatmaps around necklines, sleeves, and waistlines, where try-on and non-try-on regions meet. Our method removes these discontinuities while preserving fine garment details, such as patterns, textures, and high-frequency elements (logos and text). Results of baseline models without our refinement are provided in Fig. 9. 

Cross-Domain Generalization. We further test models trained on VITON-HD in a cross-domain setting using SHHQ-1.0 (Table 3, right columns). The large domain gap between studio-quality and in-the-wild images challenges two-stage pipeline models. HR-VITON, LaDI-VTON, and DCIVTON, which depend on independent warping modules, often produce misaligned clothing in the try-on region (Fig. 4b). In contrast, applying our approach enables both DCI-VTON and StableVITON to generate artifact-free results across diverse poses, lighting conditions, and clothing styles. GP-VTON and GarDiff are excluded from SHHQ evaluation due to dataset-specific preprocessing. 

7 

**==> picture [396 x 101] intentionally omitted <==**

**==> picture [103 x 9] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a) VITON-HD/VITON-HD<br>**----- End of picture text -----**<br>


**==> picture [397 x 128] intentionally omitted <==**

**==> picture [98 x 10] intentionally omitted <==**

**----- Start of picture text -----**<br>
(b) VITON-HD/SHHQ-1.0<br>**----- End of picture text -----**<br>


Figure 4: Comparison of baseline models with and without our method across datasets. (a) On VITON-HD, our method removes boundary artifacts while preserving garment details in DCIVTON, GarDiff, and StableVITON. Heatmaps visualize gradient magnitudes at boundaries. (b) On HSSQ-1.0, cross-domain evaluation (trained on VITON-HD) shows our approach maintains artifact-free results and natural boundary transitions, demonstrating strong generalizability across clothing types and poses. 

|Dataset(train/test)<br>Method|VITON-HD / VITON-HD<br>SSIM <br>PSNR <br>LPIPS <br>FID <br>KID <br>FID|VITON-HD / VITON-HD<br>SSIM <br>PSNR <br>LPIPS <br>FID <br>KID <br>FID|VITON-HD / VITON-HD<br>SSIM <br>PSNR <br>LPIPS <br>FID <br>KID <br>FID|VITON-HD / VITON-HD<br>SSIM <br>PSNR <br>LPIPS <br>FID <br>KID <br>FID|VITON-HD / VITON-HD<br>SSIM <br>PSNR <br>LPIPS <br>FID <br>KID <br>FID|VITON-HD / SHHQ<br> <br>KID|VITON-HD / SHHQ<br> <br>KID|
|---|---|---|---|---|---|---|---|
|||||||<br>KID||
|HR-VITON Lee et al. (2022)<br><br><br><br><br><br><br><br>GP-VTON Xie et al. (2023)<br><br><br><br><br><br>LaDI-VTON Morelli et al. (2023)<br><br><br><br><br>0.0004<br><br>||||||||
|DCI-VTON Gou et al. (2023)||||||||
|DCI-VTON (Ours)|0.8946|24.6903|0.0722|||21.1485|0.0040|
|GarDiff Wan et al. (2024)||||||||
|GarDiff (Ours)||||||||
|StableVITON Kim et al. (2024a)||||||||
|StableVITON (Ours)||||9.7669|||0.0040|



Table 3: Quantitative comparison on VITON-HD and cross-domain evaluation on SHHQ-1.0. Left columns show same-domain results (VITON-HD/VITON-HD), right columns show generalization capability (VITON-HD/SHHQ-1.0). Our method, applied without architectural modifications, consistently improves all baseline models across both in-domain and cross-domain settings. 

|Method||SSIM <br>PSNR <br>LPIPS <br>FID <br>KID|
|---|---|---|
|GP-VTON<br><br><br><br><br><br>LaDI-VTON<br><br><br><br><br><br>StableVITON<br><br>26.5536<br><br><br><br>StableVITON (Ours)<br>0.9377<br>26.7143<br>0.0361<br>13.0083<br>0.0009|||



DressCode results. On DressCode GP-VTON upper-body, our method consistently LaDI-VTON improves performance and eliminates StableVITON 26.5536 StableVITON (Ours) 0.9377 26.7143 0.0361 13.0083 0.0009 boundary artifacts observed in prior approaches (Table 4, Fig. 11). ExTable 4: Quantitative evaluation on DressCode upper-body. isting methods struggle with comOur method consistently improves all metrics, showing roplex poses and long garments: GPbust performance in full-body scenarios. VTON produces severe distortions, 

LaDI-VTON suffers from texture degradation, and baseline StableVITON exhibits boundary seams. In contrast, StableVITON enhanced with our solver generates artifact-free results across challenging cases. 

8 

|@ <br>SSIM <br>PSNR <br>LPIPS <br>FID <br>|KID <br><br><br>0.0012<br><br><br><br><br>Method<br>SSIM <br>PSNR <br>LPIPS <br>FID <br>KID <br>Pure<br><br><br><br><br><br>+ (A) Prior-based<br><br><br><br><br><br>+ 2<br>Hard measure.<br><br><br><br><br><br>+ 4<br>Data consist.<br><br><br><br><br><br>+ 5<br>Freq-Corr.<br><br><br><br><br><br>+ (C) Std. denoising<br>0.8859<br>23.7027<br>0.0746<br>9.7669<br>0.0009|
|---|---|
|Pure<br><br><br><br><br>Pure (51 step)<br><br><br><br>Unmasked<br>0.8566<br>23.2551<br><br>Offset noise<br><br><br><br>10.3335<br><br>Prior (DDIM)<br><br><br><br><br><br>Prior (DDPM)<br><br><br>0.0833||



Table 5: Quantitative comparison of z configurations at T on StableVITON (VITONHD). Prior (DDPM) achieves a good balance, showing strong performance across all metrics. 

Table 6: Ablation study on StableVITON (VITONHD). Incrementally adding each component of our method leads to consistent improvements, confirming their complementary roles. 

**==> picture [318 x 150] intentionally omitted <==**

Figure 5: Ablation study of pipeline components. Direct measurement enforcement increases artifacts, while subsequent additions (data consistency, frequency correction, and periodic denoising) progressively reduce them, yielding artifact-free and coherent results. 

## 5.4 ABLATION STUDY 

Initialization strategy analysis. Our Prior (DDPM) initialization achieves balanced gains across both paired and unpaired metrics (Table 5). Injecting data into boosts paired metrics (SSIM, PSNR, LPIPS) by preserving structure, while semantic alignment benefits unpaired metrics (FID, KID). Alternative strategies reveal clear trade-offs: Pure lacks real data, lowering paired metrics; Unmasked replaces measurement regions with noisy observations, misaligning semantics and degrading FID/KID; Oset noise adds global correlated noise to expand brightness range, which preserves semantic alignment and improves FID/KID but lacks real data, leading to poor paired metrics; reduces diversity due to deterministic sampling. In contrast, injects 

minimal semantic structure into initialization, aligning masked and measured regions while retaining diversity, yielding the most balanced performance at T . 

Component Contribution. We further assess each module’s role. (A) Prior-based initialization stabilizes trajectories and improves overall quality (Table 6). 2 Direct measurement enforcement guarantees constraint satisfaction but introduces severe boundary artifacts, showing the need for semantic alignment (Fig. 5). 4 Data consistency mitigates residual artifacts but only partially. 5 Frequency correction recovers high-frequency details lost in VAE encoding, improving semantic alignment across regions. (C) Periodic standard denoising leverages LDM priors for harmonization, stabilizing trajectories, and enhancing coherence. Together, these results confirm that each component is complementary, and their integration is essential for artifact-free, coherent synthesis. 

## 6 CONCLUSION 

We propose ART-VITON, a model-agnostic framework that addresses boundary artifacts in virtual try-on. By reformulating VITON as a linear inverse problem and using measurement-guided diffusion sampling, it preserves non-try-on regions and maintains garment alignment. Key innovations include prior-based initialization to reduce training-inference mismatch and artifact-free sampling via data consistency, frequency-level correction, and standard denoising. Experiments show im- 

9 

proved boundary coherence and high-frequency detail. ART-VITON delivers accurate, artifact-free virtual try-on, providing users with a realistic and trustworthy preview of fit and style. 

10 

## REFERENCES 

- Gal Almog, Ariel Shamir, and Ohad Fried. Reed-vae: Re-encode decode training for iterative image editing with diffusion models. In Computer Graphics Forum, pp. e70020. Wiley Online Library, 2025. 

- Jooyoung Choi, Sungwon Kim, Yonghyun Jeong, Youngjune Gwon, and Sungroh Yoon. Ilvr: Conditioning method for denoising diffusion probabilistic models. In Procdings of the I iol Corc on Computer V, pp. 14367–14376, 2021a. 

- Jooyoung Choi, Jungbeom Lee, Chaehun Shin, Sungwon Kim, Hyunwoo Kim, and Sungroh Yoon. Perception prioritized training of diffusion models. In Procdings of the I/CVF on Computer Vson and Paern Rec, pp. 11472–11481, 2022. 

- Seunghwan Choi, Sunghyun Park, Minsoo Lee, and Jaegul Choo. Viton-hd: High-resolution virtual try-on via misalignment-aware normalization. In Proc r on computer vson and paern rec, pp. 14131–14140, 2021b. 

- Yisol Choi, Sangkyung Kwak, Kyungmin Lee, Hyungwon Choi, and Jinwoo Shin. Improving diffusion models for authentic virtual try-on in the wild. In European Corc on Computer V, pp. 206–235. Springer, 2024. 

- Hyungjin Chung, Byeongsu Sim, Dohoon Ryu, and Jong Chul Ye. Improving diffusion models for inverse problems using manifold constraints. Neural Proce , 35:25683–25696, 2022. 

- Hyungjin Chung, Jeongsol Kim, Michael Thompson Mccann, Marc Louis Klasky, and Jong Chul Ye. Diffusion posterior sampling for general noisy inverse problems. In tional Corc on Learig R, 2023. 

- Hyungjin Chung, Jong Chul Ye, Peyman Milanfar, and Mauricio Delbracio. Prompt-tuning latent diffusion models for inverse problems. In Proc rc Machine Lear, pp. 8941–8967, 2024. 

- Jianglin Fu, Shikai Li, Yuming Jiang, Kwan-Yee Lin, Chen Qian, Chen-Change Loy, Wayne Wu, and Ziwei Liu. Stylegan-human: A data-centric odyssey of human generation. , arXiv:2204.11823, 2022. 

Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit Haim Bermano, Gal Chechik, and Daniel Cohen-or. An image is worth one word: Personalizing text-to-image generation using textual inversion. In The Elvnth Iiol Corc on Learig R, 2023. 

Yuying Ge, Yibing Song, Ruimao Zhang, Chongjian Ge, Wei Liu, and Ping Luo. Parser-free virtual try-on via distilling appearance flows. In Procdings of the I/CVF r on computer son and paern rec, pp. 8485–8493, 2021. 

- Junhong Gou, Siyu Sun, Jianfu Zhang, Jianlou Si, Chen Qian, and Liqing Zhang. Taming the power of diffusion models for high-quality virtual try-on with appearance flow. In Proc 31st ACM Iiol Corc on Mult, pp. 7599–7607, 2023. 

Xintong Han, Zuxuan Wu, Zhe Wu, Ruichi Yu, and Larry S Davis. Viton: An image-based virtual try-on network. In Procdings of the I r on computer vson and paern recogni, pp. 7543–7552, 2018. 

Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. , 2022. 

- Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural rmat proceing , 33:6840–6851, 2020. 

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. , 1(2):3, 2022. 

11 

Li Hu. Animate anyone: Consistent and controllable image-to-video synthesis for character animation. In Procdings of the I/CVF Corc on Computer Vson and Paern Rec, pp. 8153–8163, 2024. 

Jeongho Kim, Guojung Gu, Minho Park, Sunghyun Park, and Jaegul Choo. Stableviton: Learning semantic correspondence with latent diffusion model for virtual try-on. In Proc /CVF r on computer vson and paern rec, pp. 8176–8185, 2024a. 

- Jeongsol Kim, Geon Yeong Park, and Jong Chul Ye. Dreamsampler: Unifying diffusion sampling and score distillation for image manipulation. In European Corc on Computer V, pp. 398–414. Springer, 2024b. 

- Jeongsol Kim, Geon Yeong Park, Hyungjin Chung, and Jong Chul Ye. Regularization by texts for latent diffusion inverse solvers. In The Tn Iiol Corc on Learig Rpr, 2025. 

Diederik P Kingma and Max Welling. Auto-encoding variational bayes, 2022. URL . 

- Sangyun Lee, Gyojung Gu, Sunghyun Park, Seunghwan Choi, and Jaegul Choo. High-resolution virtual try-on with misalignment and occlusion-handled conditions. In European Corc on Computer V, pp. 204–219. Springer, 2022. 

Shanchuan Lin, Bingchen Liu, Jiashi Li, and Xiao Yang. Common diffusion noise schedules and sample steps are flawed. In Procdings of the I/CVF winter r on ctons of computer v, pp. 5404–5411, 2024. 

- Andreas Lugmayr, Martin Danelljan, Andres Romero, Fisher Yu, Radu Timofte, and Luc Van Gool. Repaint: Inpainting using denoising diffusion probabilistic models, 2022a. URL . 

- Andreas Lugmayr, Martin Danelljan, Andres Romero, Fisher Yu, Radu Timofte, and Luc Van Gool. Repaint: Inpainting using denoising diffusion probabilistic models. In Proc /CVF r on computer vson and paern rec, pp. 11461–11471, 2022b. 

- Davide Morelli, Matteo Fincato, Marcella Cornia, Federico Landi, Fabio Cesari, and Rita Cucchiara. Dress code: High-resolution multi-category virtual try-on. In Procdings of the I/CVF conrnc on computer vson and paern rec, pp. 2231–2235, 2022. 

- Davide Morelli, Alberto Baldrati, Giuseppe Cartella, Marcella Cornia, Marco Bertini, and Rita Cucchiara. Ladi-vton: Latent diffusion textual-inversion enhanced virtual try-on. In Procdings of the 31st ACM ol r on , pp. 8580–8589, 2023. 

Lev Novitskiy, Viacheslav Vasilev, Maria Kovaleva, Vladimir Arkhipkin, and Denis Dimitrov. Vivat: Virtuous improving vae training through artifact mitigation. , 2025. 

OpenAI. Chatgpt (gpt-5). , 2025. Large language model. 

Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. In The Twelfth Iiol Corc on Learig R, 2024. 

- Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In Iiol r on machine , pp. 8821–8831. Pmlr, 2021. 

- Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. Highresolution image synthesis with latent diffusion models. In Procdings of the I/CVF confernc on computer vson and paern rec, pp. 10684–10695, 2022. 

Litu Rout, Negin Raoof, Giannis Daras, Constantine Caramanis, Alex Dimakis, and Sanjay Shakkottai. Solving linear inverse problems provably via posterior sampling with latent diffusion models. Advances in Neural Irmati Proceing Sy, 36:49960–49990, 2023. 

12 

Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Pror on ,rec pp. 22500– 22510, 2023. 

- Mehmet Saygin Seyfioglu, Karim Bouyarmane, Suren Kumar, Amir Tavanaei, and Ismail B Tutar. Dreampaint: Few-shot inpainting of e-commerce items for virtual try-on without 3d modeling. arXiv eint arXiv:, 2023. 

Bowen Song, Soo Min Kwon, Zecheng Zhang, Xinyu Hu, Qing Qu, and Liyue Shen. Solving inverse problems with latent diffusion models via hard data consistency, 2024. URL . 

- Siqi Wan, Yehao Li, Jingwen Chen, Yingwei Pan, Ting Yao, Yang Cao, and Tao Mei. Improving virtual try-on with garment-focused diffusion models. In European rc , pp. 184–199. Springer, 2024. 

- Chenhui Wang, Tao Chen, Zhihao Chen, Zhizhong Huang, Taoran Jiang, Qi Wang, and Hongming Shan. Fldm-vton: Faithful latent diffusion model for virtual try-on. In , 2024. 

- Rongyuan Wu, Tao Yang, Lingchen Sun, Zhengqiang Zhang, Shuai Li, and Lei Zhang. Seesr: Towards semantics-aware real-world image super-resolution. In Proc r on computer vson and paern rec, pp. 25456–25467, 2024. 

Zhenyu Xie, Zaiyu Huang, Xin Dong, Fuwei Zhao, Haoye Dong, Xijin Zhang, Feida Zhu, and Xiaodan Liang. Gp-vton: Towards general purpose virtual try-on via collaborative local-flow global-parsing learning. In Proc ern rec, pp. 23550–23559, 2023. 

Yici Yan, Yichi Zhang, Xiangming Meng, and Zhizhen Zhao. Fig: Flow with interpolant guidance for linear inverse problems. In The Tn I **te** iol Corc on Learig Rprsnta, 2025. 

- Han Yang, Ruimao Zhang, Xiaobao Guo, Wei Liu, Wangmeng Zuo, and Ping Luo. Towards photorealistic virtual try-on by adaptively generating-preserving image content. In Procdings of the /CVF r on computer vson and paern rec, pp. 7850–7859, 2020. 

- Tao Yang, Rongyuan Wu, Peiran Ren, Xuansong Xie, and Lei Zhang. Pixel-aware stable diffusion for realistic image super-resolution and personalized stylization. In The European Corc on Computer Vson (EV) , 2023. 

- Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models. arXiv eint arXiv:, 2023. 

- Ruiyun Yu, Xiaoqi Wang, and Xiaohui Xie. Vtnfp: An image-based virtual try-on network with body and clothing feature preservation. In Procdings of the I/CVF ol r on computer v, pp. 10511–10520, 2019. 

- Jinjin Zhang, Qiuyu Huang, Junjie Liu, Xiefan Guo, and Di Huang. Diffusion-4k: Ultra-highresolution image synthesis with latent diffusion models. In Procdings of the Computer V and Paern Rect Corc, pp. 23464–23473, 2025a. 

Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Procdings of the I/CVF ol r on computer v, pp. 3836–3847, 2023. 

- Xuanpu Zhang, Dan Song, Pengxin Zhan, Tianyu Chang, Jianhao Zeng, Qingguo Chen, Weihua Luo, and An-An Liu. Boow-vton: Boosting in-the-wild virtual try-on via mask-free pseudo data training. In Procdings of the Computer Vson and Paern Rect Corc, pp. 26399– 26408, 2025b. 

13 

## A APPENDIX 

## A.1 USE OF LARGE LANGUAGE MODELS 

We used a large language model OpenAI (2025) solely to improve the clarity and readability of the manuscript (e.g., grammar and phrasing). The model did not contribute to research ideation, methodology, or analysis, and the authors take full responsibility for all contents. 

## A.2 IMPLEMENTATION DETAILS OF BASELINES 

Pretrained checkpoints are used where available; StableVITON is retrained on DressCode upperbody items for consistency. All models use DDIM Lugmayr et al. (2022a) with steps and classifier-free guidance (CFG) Ho & Salimans (2022) with scale (except LaDI-VTON, scale ). For inverse solvers, all methods are adapted to the latent diffusion framework, sharing the same (A) initialization and (C) standard denoising steps ( = 2), differing only in the (B) measurementguided sampling component. 

## A.3 INVERSE SOLVER FORMULATION 

We classify inverse solvers into three types: hard constraints (RePaint Lugmayr et al. (2022b), MCG Chung et al. (2022)), progressive updates (DPS Chung et al. (2023), FIG Yan et al. (2025)), and hybrid stochastic methods (DreamSampler Kim et al. (2024b), TReg Kim et al. (2025)). Hard constraints induce semantic drift between regions due to strong measurement enforcement, directly causing boundary artifacts. Progressive updates maintain stable optimization and produce minimal artifacts. However, both hard constraints and progressive updates operate in latent space, failing to fully satisfy measurements (Fig. 7). To address this, we apply post-hoc replacement, which can still cause boundary artifacts due to semantic mismatch and spatial discontinuities. Hybrid stochastic methods enforce measurement constraints in pixel space and inject stochastic noise to harmonize regions, reducing artifacts. Nevertheless, persistent semantic drift still leads to artifact formation. 

We formulate virtual try-on as an inverse problem and integrate various solver strategies into the latent diffusion sampling process. This section presents the mathematical foundations and implementation details of each approach. We denote the measurement mask as M and the target measurement as y. The bar notation indicates resizing to match the latent code resolution. Specifically, denotes the measurement mask with value 1 in the resized measurement region, and represents the resized target measurement. A comparison with the inverse solvers is shown in Fig. 8. 

DDIM sampling Lugmayr et al. (2022a). The deterministic DDIM sampling forms the basis for all inverse solvers. Given a noisy latent at timestep , we first estimate the clean latent using Tweedie’s formula: 

**==> picture [282 x 13] intentionally omitted <==**

The denoising step then updates the latent to timestep t : 

**==> picture [269 x 13] intentionally omitted <==**

## A.3.1 HARD MEASUREMENT METHODS 

These methods enforce measurement consistency through direct projection or replacement in the latent space. 

RePaint Lugmayr et al. (2022b). This approach replaces the measurement region with noisy observations at each denoising step. We omit the resampling strategy proposed in Repaint as it is too time-consuming: 

**==> picture [275 x 27] intentionally omitted <==**

14 

MCG (Manifold-Constrained Gradient) Chung et al. (2022). This method combines gradientbased optimization with hard projection: 

**==> picture [276 x 29] intentionally omitted <==**

where is the gradient step size, which we set to 1. 

## A.3.2 PROGRESSIVE UPDATE METHODS 

These methods guide the sampling trajectory iteratively through gradient updates without relying on hard measurement constraints. 

DPS (Diffusion Posterior Sampling) Chung et al. (2023). DPS adjusts the sampling trajectory via measurement consistency gradients computed in the Tweedie space: 

**==> picture [276 x 14] intentionally omitted <==**

where we set = 1. 

FIG (Flow with Interpolant Guidance) Yan et al. (2025). By operating directly on the noisy latent, FIG performs gradient updates along the diffusion trajectory, preserving stability and sample diversity, whereas Tweedie-space optimization is more precise but incurs higher computational cost and reduces diversity. 

**==> picture [289 x 13] intentionally omitted <==**

with = 1. 

## A.3.3 HYBRID STOCHASTIC METHODS 

These approaches combine deterministic updates with stochastic noise injection, where the degree of stochasticity is controlled through[,][to][balance][measurement][consistency][and][generation][di-] versity. 

**==> picture [267 x 24] intentionally omitted <==**

where controls the noise level and[is the noise schedule.][The pixel-space optimization is per-] formed via gradient updates with a learning rate of , a regularization coefficient of , and 1 iterations: 

DreamSampler Kim et al. (2024b). DreamSampler integrates pixel-space and latent-space optimization to guide the diffusion sampling trajectory while maintaining measurement consistency. Let denote a null embedding, as introduced in the classifier-free guidance (CFG) framework, used to perform latent optimization without conditioning information. In the final latent update, the stochastic noise term is set by[,][controlling the amount of injected noise] to balance diversity and trajectory stability. 

**==> picture [359 x 100] intentionally omitted <==**

where E and D denote encoder and decoder, and balances data fidelity. 

TReg Kim et al. (2025). TReg performs the hybrid approach by performing optimization directly in pixel space with latent regularization. It solves a regularized inverse problem where 

15 

the measurement operator enforces constraints, while the regularization term maintains semantic coherence via the diffusion prior. In the stochastic update, the noise parameter is set as 

> [, following a noise schedule distinct from DreamSampler.] 

**==> picture [337 x 55] intentionally omitted <==**

A.4 ADDITIONAL RESULTS 

16 

**==> picture [397 x 608] intentionally omitted <==**

Figure 6: Qualitative results of baseline models on the SHHQ-1.0 dataset. Our observations show that generated images fail to preserve content in non-try-on regions: bags, skirts, cars, text, and human features (green boxes). Orange boxes indicate areas where facial details are not properly preserved. 

17 

**==> picture [397 x 390] intentionally omitted <==**

Figure 7: StableVITON on VITON-HD with inverse solvers applied without post-hoc replacement. Red indicates face zoom-in, and orange and green indicate artifact map zoom-ins. Hard constraint solvers (RePaint, MCG) and progressive update solvers (DPS, FIG) fail to fully satisfy measurements, highlighting the need for post-hoc replacement. Hard constraints generate artifacts due to semantic inconsistencies across regions, whereas progressive updates produce minimal artifacts, as each update induces only small changes. 

18 

**==> picture [396 x 434] intentionally omitted <==**

Figure 8: Comparison on the VITON-HD dataset with baseline (StableVITON) and existing inverse solvers. Red circles highlight texture degradation, particularly in hybrid stochastic methods (DreamSampler, TReg), while our approach preserves fine garment details and patterns. Orange boxes indicate artifacts present in other methods, which are absent in our results. 

19 

Figure 9: Additional qualitative results on the VITON-HD comparing baseline methods with our approach. (a) Comparison of baselines and their versions enhanced with our method: our approach consistently removes boundary artifacts while preserving high-frequency garment details such as logos, text, and complex patterns. (b) Results of the remaining models without our enhancement: in 2-stage pipeline models, warping results show garment distortions and color inconsistencies. 

20 

Figure 10: Extended comparison demonstrating robustness across domains on the SHHQ-1.0 dataset. (a) Comparison of baselines and their versions enhanced with our method: even in crossdomain scenarios, our approach effectively removes artifacts, demonstrating robustness. (b) Other VITON methods show boundary artifacts and garment distortion, whereas our approach preserves boundaries and garment details. 

21 

Figure 11: Qualitative comparison of baseline VITON methods on DressCode dataset. Traditional methods (GP-VTON, LaDI-VTON) exhibit misalignment and texture distortion, while StableVITON shows boundary artifacts despite better garment alignment. Our method applied to StableVITON (rightmost) eliminates boundary inconsistencies while preserving both garment details and identity features. 

22 

