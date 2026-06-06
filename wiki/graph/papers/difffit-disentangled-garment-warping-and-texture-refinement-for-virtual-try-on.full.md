---
type: paper-fulltext
slug: difffit-disentangled-garment-warping-and-texture-refinement-for-virtual-try-on
arxiv_id: null
source_path: /Users/saris.kia.adm/.paper-scholar/difffit-disentangled-garment-warping-and-texture-refinement-for-virtual-try-on/2506.23295.md
paper: "[[difffit-disentangled-garment-warping-and-texture-refinement-for-virtual-try-on]]"
---
<!-- extracted by afk_extract from 2506.23295.pdf (16p) -->

# DiffFit: Disentangled Garment Warping and Texture Refinement for Virtual Try-On 

Xiang Xu[1*] 

> 1*China Jiliang University. 

Corresponding author(s). E-mail(s): xu@cjlu.edu.cn; 

## **Abstract** 

Virtual try-on (VTON) aims to synthesize realistic images of a person wearing a target garment, with broad applications in e-commerce and digital fashion. While recent advances in latent diffusion models have substantially improved visual quality, existing approaches still struggle with preserving fine-grained garment details, achieving precise garment-body alignment, maintaining inference efficiency, and generalizing to diverse poses and clothing styles. To address these challenges, we propose DiffFit, a novel two-stage latent diffusion framework for high-fidelity virtual try-on. DiffFit adopts a progressive generation strategy: the first stage performs geometry-aware garment warping, aligning the garment with the target body through fine-grained deformation and pose adaptation. The second stage refines texture fidelity via a cross-modal conditional diffusion model that integrates the warped garment, the original garment appearance, and the target person image for high-quality rendering. By decoupling geometric alignment and appearance refinement, DiffFit effectively reduces task complexity and enhances both generation stability and visual realism. It excels in preserving garment-specific attributes such as textures, wrinkles, and lighting, while ensuring accurate alignment with the human body. Extensive experiments on large-scale VTON benchmarks demonstrate that DiffFit achieves superior performance over existing state-of-the-art methods in both quantitative metrics and perceptual evaluations. 

**Keywords:** Virtual Try-On, Latent Diffusion Models, High-Fidelity Synthesis, Cross-Modal Fusion 

1 

## **1 Introduction** 

As the e-commerce industry continues to expand rapidly, enhancing user experience through interactive and personalized content has become crucial for improving customer engagement and conversion rates. In this context, virtual try-on (VTON)[1, 2] has emerged as a transformative technology, allowing users to visualize themselves wearing selected garments directly from product images. By transferring a clothing item onto a target person image, VTON enables personalized virtual fitting, thereby bridging the gap between physical and digital shopping. 

While VTON and virtual dressing (VD) [3, 4] both focus on clothing-body integration, their goals and application scenarios differ. VTON primarily targets end-user personalization, supporting garment fit visualization on specific individuals. In contrast, VD is designed for merchant-facing scenarios, enabling large-scale clothing display across diverse body types, poses, and styles. VTON emphasizes static but personalized try-on experiences, whereas VD focuses on scalable, dynamic garment generation across varied conditions. Recent works in layout- and attribute-controllable generation [5] have further highlighted the potential of structured conditioning in improving visual fidelity and controllability in garment synthesis. 

Despite recent progress, achieving photorealistic and semantically aligned try-on results remains challenging. Precise garment deformation and fitting require accurate modeling of pose-dependent geometry, preservation of fine-grained garment details such as textures and wrinkles, and robustness to occlusions and diverse body shapes. Existing methods often struggle with realistic fabric draping and local detail preservation under complex pose conditions. The problem is further exacerbated by the difficulty of maintaining semantic coherence between the original garment and the synthesized try-on image. Early VTON approaches [6–9] primarily relied on generative adversarial networks (GANs), employing warping modules to align garment features and generators to synthesize final images. While effective to some extent, GAN-based methods suffer from optimization instability and limited capacity to preserve highfrequency textures. This leads to blurry outputs and poor garment-body alignment, especially in the presence of occlusions or pose variations. These limitations have motivated the shift toward more stable and expressive generative paradigms. 

Recently, latent diffusion models (LDMs) [10] have demonstrated remarkable capabilities in controlled image synthesis tasks, including pose-guided person generation [11] and long-term motion-consistent generation [12]. Their iterative denoising process allows for better integration of semantic priors and more effective texture preservation. In the VTON domain, diffusion-based methods enable multi-step refinement of garment fitting and detail rendering, overcoming many limitations of adversarial training. 

In this work, we propose DiffFit, a two-stage latent diffusion framework for highfidelity virtual try-on. Our method adopts a progressive generation strategy that decouples geometric alignment and texture refinement into two stages. In the first stage, we introduce a pose-adaptive diffusion network that implicitly learns semantic correspondences between garments and body shapes in the latent space, generating pose-aligned warped garments without relying on explicit keypoint annotations. This 

2 

design is inspired by prior work on geometry-aware garment modeling [13]. In the second stage, we present a cross-modal conditional diffusion network, which integrates the warped garment with VAE-encoded representations of the original garment and the target person. This enables faithful texture reconstruction, occlusion handling, and photorealistic rendering. We evaluate DiffFit on three widely-used VTON benchmarks: VITON-HD [14], Dress Code [9], and IGPair [3]. Extensive experiments demonstrate that our approach achieves state-of-the-art performance in both quantitative metrics and perceptual quality. Moreover, cross-domain evaluation shows strong generalization to unseen poses and garment types. Visual comparisons highlight DiffFit’s ability to preserve texture details, generate realistic fabric deformations, and maintain coherent garment-body alignment. Our main contributions are summarized as follows: 

- We propose DiffFit, a novel two-stage latent diffusion framework that progressively models geometric alignment and texture refinement for high-fidelity virtual try-on. 

- We design a pose-adaptive diffusion network that establishes garment-body correspondence in the latent space, enabling precise and natural garment fitting across diverse poses. 

- We introduce a cross-modal fusion module for conditional rendering that preserves texture consistency, handles occlusions, and synthesizes photorealistic results with fine structural details. 

## **2 Related Work** 

## **2.1 Virtual Try-on** 

Virtual try-on (VTON) has gained significant attention in computer vision due to its wide applicability in digital fashion and personalized e-commerce. Early approaches predominantly leveraged generative adversarial networks (GANs) [15] as their generative backbone. These methods typically adopt a two-stage pipeline [6, 7, 16, 17]: a geometric warping stage that deforms garments to align with the target human body, followed by a synthesis stage that fuses the warped garment with the person image. VITON-HD [14] proposes a synchronized optimization strategy to jointly refine garment deformation and human segmentation, addressing spatial misalignment and occlusion issues. GPVTON [16] introduces a dual-path network to separately handle local garment regions and global body structure. While effective, traditional GAN-based methods often struggle with preserving high-frequency garment details, especially in complex poses or backgrounds, and are limited in their ability to generalize to unseen poses or body shapes. 

With the emergence of diffusion models, several recent VTON methods have transitioned to more stable and expressive generative paradigms. DCI-VTON [18] integrates diffusion modules with conventional garment warping to improve texture preservation. LaDI-VTON [19] introduces enhanced skip connections to mitigate information loss during autoencoding. StableVITON [20] removes the need for explicit warping by learning garment-to-body correspondence directly in the latent space of a pre-trained diffusion model. However, these approaches still face challenges related to semantic alignment, fine-grained detail preservation, and inference efficiency. To address these 

3 

limitations, multi-branch and modular frameworks have been explored. TryOnDiffusion [21] employs dual UNets to separately model garment and person features. OOTDiffusion [22] incorporates attention-based fusion to enhance semantic coherence. IMAGDressing-v1 [3] introduces a garment-specific UNet alongside a denoising UNet with hybrid attention, effectively balancing garment appearance and conditional controls. Our work builds upon these insights by decoupling pose-guided garment warping and texture-aware rendering within a unified latent diffusion framework, achieving better alignment and visual fidelity across diverse VTON scenarios. 

## **2.2 Latent Diffusion Models** 

Latent diffusion models (LDMs) [10] have revolutionized image generation by combining variational autoencoding with multi-step denoising in a compressed latent space. This design drastically reduces computational cost while retaining rich semantic representations. Specifically, an input image _x_ is encoded into a latent variable _z_ via a variational autoencoder (VAE) encoder _E_ , and the reverse process reconstructs ˆ _x_ = _D_ ( _z_ ). The diffusion process gradually corrupts _z_ by injecting Gaussian noise over _T_ steps: 

**==> picture [275 x 12] intentionally omitted <==**

and the model is trained to predict the noise: 

**==> picture [256 x 14] intentionally omitted <==**

LDMs have shown remarkable generative quality and controllability in diverse vision tasks. In VTON, they enable flexible feature composition and high-fidelity rendering. For instance, TryOnDiffusion [21] and StableVITON [20] demonstrate that pose and garment can be integrated in the latent space without explicit spatial warping. 

Beyond VTON, recent works have extended LDMs to more complex generative tasks, such as pose-guided human synthesis [23], story-consistent image generation [24], and stylized character generation [25]. These methods explore progressive conditioning, cross-modal fusion, and motion-guided priors to achieve enhanced coherence and structure preservation. Notably, the IMAGDressing [3] framework exemplifies how hybrid attention and garment-specific modules can enhance controllability in fashion generation. Similarly, warping-guided latent diffusion approaches [26] demonstrate that integrating explicit geometric priors into the diffusion process improves fitting accuracy and detail alignment. Moreover, methods like Ensembling Diffusion [27] further show that adaptive feature aggregation across multiple diffusion models can boost consistency and robustness—a principle that aligns with our progressive architecture design. Inspired by these advances, our proposed **DiffFit** model leverages pose-aware latent guidance and cross-modal fusion to generate photorealistic and structurally aligned try-on results across a wide range of poses and garments. 

4 

**==> picture [371 x 71] intentionally omitted <==**

**----- Start of picture text -----**<br>
First Stage Mask Image Second Stage Warped Cloth Image<br>Person Image Cloth Image<br>Cross Attention  Conditional  Lattent<br>Warping UNet Diffusion UNet<br>Nosiy Image Warped Cloth Image Person Image Cloth Image Nosiy Image Target Image<br>Progressive<br>Aligned<br>**----- End of picture text -----**<br>


**Fig. 1** Overview of the proposed two-stage DiffFit framework. Stage I warps the garment to fit the target body using cross-attention-based diffusion, and Stage II fuses the warped garment with the person image via a conditional latent diffusion model to generate the final high-fidelity try-on result. 

## **3 Proposed Method** 

## **3.1 Overview** 

To address the challenges of semantic misalignment and detail preservation in virtual try-on, we propose a novel two-stage latent diffusion framework. An overview of our proposed model is illustrated in Figure 1. Our pipeline is designed to explicitly disentangle garment-body correspondence from high-fidelity image synthesis, thereby improving both geometric fitting and texture realism. 

Given a person image _xp ∈_ R _[H][×][W][ ×]_[3] , a clothing image _xc ∈_ R _[H][×][W][ ×]_[3] , and a binary mask _xm ∈_ R _[H][×][W][ ×]_[1] indicating the body region to be replaced, our goal is to generate a photorealistic try-on image _x_ ˆ _∈_ R _[H][×][W][ ×]_[3] in which the garment is realistically worn by the target person. 

Our framework consists of the following two stages: 

- **Stage I: Garment Warping.** We employ a cross-attention-based diffusion model to generate a warped garment image _x_ ˜ _c_ that is semantically aligned with the target pose and body shape, while preserving garment texture and structure. 

- **Stage II: Try-on Synthesis.** The warped garment _x_ ˜ _c_ , original person image _xp_ , and reference garment _xc_ are fused via a conditional latent diffusion model to synthesize the final try-on result _x_ ˆ, ensuring seamless integration and high-fidelity detail. 

By directly utilizing image inputs for both garment and target person, our method circumvents the limitations of text-guided editing, which often fails to capture subtle appearance cues. The two-stage design also allows for explicit control over garment deformation and appearance consistency, as motivated by our observations in Section 2. 

## **3.2 Cross Attention Warping UNet (CAW-UNet)** 

A key challenge for virtual try-on is to accurately align the garment with the target body while preserving the garment’s fine-grained features. To this end, we introduce the cross attention warping UNet (CAW-UNet), which leverages multi-source feature extraction and cross-attention fusion, as shown in Figure 2. 

Given _xp_ , _xc_ , and _xm_ , we extract their visual features using fixed image encoders _Ep_ , _Ec_ , and _Em_ , producing feature maps _fp_ = _Ep_ ( _xp_ ), _fc_ = _Ec_ ( _xc_ ), and _fm_ = 

5 

**==> picture [368 x 63] intentionally omitted <==**

**----- Start of picture text -----**<br>
Cloth Image Image Encoder Projection<br>Mask Image Image Encoder Projection ModulesFrozen Cloth Image Warped Cloth Image Image Encoder Projection C Concatenation<br>Person Image Image Encoder Projection<br>TrainableModules<br>C VAE Encoder VAE Decoder<br>VAE  Encoder VAE  Decoder Person Image Denoised Image<br>Noisy Image Warped Cloth Image Denoising U-Net<br>Denoising U-Net Nosiy Image<br>**----- End of picture text -----**<br>


**Fig. 2** Cross attention warping UNet (CAW- **Fig. 3** Overview of the second stage of our virUNet) architecture. Tokens of person, cloth, and tual try-on framework. The person image, the mask images are extracted via a frozen image original garment reference image and the warped encoder and a trainable projection layer, and subgarment image generated from the first stage are sequently injected into the UNet through crossjointly fed into a trainable U-Net. By effectively attention mechanism. aggregating multi-source features, the network synthesizes the final try-on result. 

_Em_ ( _xm_ ). These feature maps are subsequently projected into a unified embedding space via a trainable Q-Former [28], resulting in token sequences _tp_ = _Q_ ( _fp_ ), _tc_ = _Q_ ( _fc_ ), and _tm_ = _Q_ ( _fm_ ), where _Q_ ( _·_ ) denotes the Q-Former projection. 

The concatenated token sequence [ _tp_ ; _tc_ ; _tm_ ] is injected into the denoising UNet via cross-attention layers, enabling the model to dynamically attend to garment, person, and mask features during the diffusion process. This design allows for flexible and accurate warping of the garment according to the target body pose, while maintaining texture fidelity. 

The output of the first stage is a warped garment image _x_ ˜ _c_ , which serves as an intermediate representation for the subsequent synthesis stage. 

## **3.3 Conditional Input and Cross-Modal Fusion in UNet** 

In the second stage, our objective is to seamlessly integrate the warped garment with the person image, while preserving both garment details and human identity. To achieve this, we design a conditional latent diffusion UNet with cross-modal fusion, as shown in Figure 3. 

Let _zp_ = _E_ ( _xp_ ), _zc_ = _E_ ( _xc_ ), and _zc_ ˜ = _E_ (˜ _xc_ ) denote the latent representations of the person image, original garment, and warped garment, respectively, where _E_ is a frozen VAE encoder. These latents are concatenated along the channel dimension and fed into the U-Net backbone. To accommodate the increased input dimensionality, the first convolutional layer is modified accordingly. 

Additionally, the feature embedding of the warped garment, extracted via a fixed image encoder and a trainable projection layer, is injected into the intermediate layers of the U-Net through cross-attention. This mechanism enables the model to adaptively fuse garment appearance, warping information, and human context, thereby enhancing the realism and consistency of the generated try-on image. 

The inclusion of both original and warped garment features provides strong constraints for preserving garment patterns and textures, while the person image ensures the retention of identity and pose. 

## **3.4 Loss Function** 

To optimize the proposed framework, we adopt a two-stage training objective that follows the standard diffusion model paradigm, conditioning on multi-modal inputs in a unified manner. 

6 

Let _tp_ , _tc_ , and _tm_ denote the learned conditioning token embeddings for the person, clothing, and mask images, respectively. In the first stage (garment warping), the model is trained to align the garment with the target body, using the following objective: 

**==> picture [282 x 15] intentionally omitted <==**

where **z** _t_ denotes the noisy latent variable at diffusion timestep _t_ , _ϵ_ is Gaussian noise sampled from _N_ ( **0** _,_ **I** ), and _ϵθ_ is the denoising network parameterized by _θ_ . 

In the second (synthesis) stage, the goal is to seamlessly integrate the warped garment with the person image for realistic try-on results. Let _twc_ denote the embedding of the warped garment. The loss function for this stage is defined as: 

**==> picture [284 x 15] intentionally omitted <==**

where all variables retain the same meanings as above. 

This two-stage training strategy ensures that the model can learn both precise garment-body alignment and high-fidelity image synthesis, ultimately producing photorealistic and semantically aligned virtual try-on results. 

## **4 Experiment and Analysis** 

We evaluate the effectiveness of our proposed method by conducting comprehensive comparisons with several state-of-the-art virtual try-on approaches, including LaDIVTON [19], DCI-VTON [18], and IMAGEDressing [3]. For all comparative methods, we use official pre-trained weights when available; otherwise, we retrain the models using the released code and recommended settings. 

## **4.1 Datasets** 

We conduct experiments on three widely used virtual try-on datasets: DressCode [29], VITON-HD [14], and IGPair [3]. 

The DressCode dataset [29] contains a total of 53,795 high-quality image pairs, covering three apparel categories: upper-body clothing (15,366 pairs), lower-body clothing (8,951 pairs), and dresses (29,478 pairs). All images are standardized to a resolution of 1024 _×_ 768 pixels. The dataset is split into 48,395 pairs for training and 5,400 pairs for testing, with the test set uniformly sampled across all categories to support fine-grained performance evaluation. 

The VITON-HD dataset [14] is tailored to address challenges of complex human poses and garment alignment in virtual try-on scenarios. It comprises 13,679 image pairs, each consisting of a frontal-view female portrait (1024 _×_ 768 pixels) and the corresponding upper-body garment image. The dataset is divided into 11,647 training pairs and 2,032 testing pairs. 

The IGPair dataset [3] includes 324,857 high-quality image pairs, each formed by a garment item and one or more corresponding images of human models wearing that item. Compared with existing benchmarks, IGPair is significantly larger in scale and 

7 

features images collected at ultra-high resolution, enabling precise preservation and modeling of subtle garment characteristics. 

## **4.2 Evaluation Metrics** 

To ensure a fair comparison with baseline methods, all experiments are conducted at a fixed resolution. For quantitative evaluation, we adopt the Structural Similarity Index (SSIM) [30] and Learned Perceptual Image Patch Similarity (LPIPS) [31] in the paired setting to measure reconstruction quality. To assess the realism and distributional consistency of generated images, the Fr´echet Inception Distance (FID) [32] and Kernel Inception Distance (KID) [33] are computed in both paired (FID _p_ , KID _p_ ) and unpaired (FID _u_ , KID _u_ ) settings. 

## **4.3 Implementation Details** 

All experiments were conducted using a single NVIDIA RTX 4090 GPU. Both stages of our model share the same optimization and training strategy. Specifically, we employ the AdamW optimizer with a fixed learning rate of 5 _×_ 10 _[−]_[5] . The model is trained for 50,000 steps with a batch size of 6. Input images are uniformly resized to 640 _×_ 512 pixels. 

For inference, image generation is performed using the UniPC sampler, with the total number of sampling steps set to 50 and a guidance scale parameter _w_ of 7.5. All other hyperparameters are kept at their default values unless otherwise specified. 

## **4.4 Comparison with State-of-the-Art Methods** 

To thoroughly evaluate the effectiveness of our approach, we compare it with several state-of-the-art virtual try-on methods, including GP-VTON [16], VITON-HD [14], DCI-VTON [18], and LaDI-VTON [19]. Quantitative results for IMAGEDressing are not included, as its relevant evaluation metrics have not been publicly released. 

Table 1 summarizes the quantitative results on the DressCode dataset. Our method achieves the best FID _p_ (4.06), KID _p_ (1.51), FID _u_ (6.28), and the highest SSIM (0.986), indicating superior image fidelity and structural similarity. For KID _u_ , our score (1.71) is nearly identical to that of DCI-VTON (1.70), reflecting competitive performance in unpaired metric scenarios. Although DCI-VTON attains the lowest LPIPS (0.0301), our LPIPS (0.040) remains highly competitive. Meanwhile, LaDI-VTON reports a higher FID _p_ (4.85) and lower SSIM (0.906) compared to our method, indicating less realism and weaker structural retention on DressCode. 

Table 2 presents the quantitative comparisons on the VITON-HD dataset. Our approach consistently outperforms previous methods, achieving the lowest LPIPS (0.041), the highest SSIM (0.916), as well as the best FID _p_ (6.62), KID _p_ (1.08, tying with LaDI-VTON), and FID _u_ (8.21). For KID _u_ , DCI-VTON obtains the best result (0.49), while our method achieves a closely comparable score (1.36). Notably, LaDIVTON achieves a KID _p_ of 1.08, matching our method, but is outperformed by our approach in LPIPS and SSIM, underscoring our superior perceptual and structural consistency. 

8 

**Table 1** Quantitative comparisons on the DressCode dataset. Best results are highlighted in bold. 

|Method|LPIPS_↓_|SSIM_↑_|FID_p ↓_|KID_p ↓_|FID_u ↓_|KID_u ↓_|
|---|---|---|---|---|---|---|
|GP-VTON [16]|0.0359|0.9479|–|–|11.98|–|
|DCI-VTON [18]|**0.0301**|0.948|–|–|6.48|**1.70**|
|LaDI-VTON [19]|0.064|0.906|4.85|1.61|7.50|2.83|
|**Ours**|0.040|**0.986**|**4.06**|**1.51**|**6.28**|1.71|



Lower values ( _↓_ ) are better for LPIPS, FID, and KID; higher values ( _↑_ ) are better for SSIM. Subscript _p_ : paired; _u_ : unpaired; –: unavailable. 

**Table 2** Quantitative comparisons on the VITON-HD dataset. Best results are highlighted in bold. 

|Method|LPIPS_↓_|SSIM_↑_|FID_p ↓_|KID_p ↓_|FID_u ↓_|KID_u ↓_|
|---|---|---|---|---|---|---|
|GP-VTON [16]|0.0799|0.8939|–|–|9.197|–|
|VITON-HD [14]|0.116|0.863|11.01|3.71|12.96|4.09|
|DCI-VTON [18]|0.043|0.896|8.09|2.80|8.23|**0.49**|
|LaDI-VTON [19]|0.091|0.876|6.66|1.08|9.41|1.60|
|**Ours**|**0.041**|**0.916**|**6.62**|**1.08**|**8.21**|1.36|



Lower values ( _↓_ ) are better for LPIPS, FID, and KID; higher values ( _↑_ ) are better for SSIM. Subscript _p_ : paired; _u_ : unpaired; –: unavailable. 

On VITON-HD, our method achieves substantially higher SSIM and lower LPIPS than DCI-VTON, with consistently superior or competitive FID and KID scores, indicating improved structural fidelity, perceptual similarity, and image realism. LaDIVTON attains competitive FID _p_ and KID _p_ values, but is outperformed by our approach in SSIM and LPIPS. On DressCode, our method also ranks best in SSIM, FID _p_ , KID _p_ , and FID _u_ , and delivers comparable LPIPS to DCI-VTON, highlighting its robustness across datasets. Overall, our results confirm the effectiveness of our approach in both perceptual quality and structural preservation compared to state-of-the-art baselines. 

For qualitative comparison, we further evaluate the image synthesis quality of our method against three representative approaches, as illustrated in Figure 4. Our method demonstrates notable advantages in several key aspects. First, as observed in the first row, DCI-VTON produces artifacts around the clothing neckline, and when the clothing logo is partially occluded by hair, both LaDI-VTON and DCI-VTON fail to accurately reconstruct the textual details. In contrast, our method successfully restores the logo. Additionally, IMAGEDressing fails to generate hair that matches the model’s appearance. Second, in the second row, DCI-VTON results in garment deformation near the neckline, while IMAGEDressing is unable to preserve the original body pose. By comparison, our method maintains both the correct pose and garment structure. Third, as shown in the third row, DCI-VTON fails to retain the intended clothing style, LaDI-VTON produces inconsistent garment positioning, and IMAGEDressing 

9 

**==> picture [372 x 271] intentionally omitted <==**

**----- Start of picture text -----**<br>
Person Clothing DCI-VTON LaDI-VTON IMAGEDressing Ours<br>**----- End of picture text -----**<br>


**Fig. 4** Qualitative comparison of our method with state-of-the-art virtual try-on approaches (DCIVTON, LaDI-VTON, IMAGEDressing) on challenging cases from the DressCode and VITON-HD datasets. 

introduces prominent white edges around the vest. In contrast, our results remain visually realistic and structurally coherent across all examples. These improvements highlight the enhanced capacity of our model to handle complex textures, occlusions, and pose variations in virtual try-on applications. 

## **4.5 User Study** 

To further assess the perceptual effectiveness and practical applicability of our approach, we conducted a comprehensive user study involving comparisons with two state-of-the-art baselines, DCI-VTON and LaDI-VTON, on the DressCode and VITON-HD datasets. 

The evaluation was structured around three principal criteria: (1) overall image realism, (2) garment detail preservation, and (3) consistency with human body pose. In each trial, participants were presented with a pair of images: one produced by our method and the other by a baseline. Both the order of images and the choice of baseline were randomized to minimize presentation and expectation biases. Participants were instructed to select the image that appeared more realistic and visually satisfactory according to the specified criterion. 

10 

**Table 3** User study preference (%) comparing DCI-VTON, LaDI-VTON, and our method on VITON-HD and DressCode datasets for image realism, garment detail, and body consistency. Higher is better. 

|**Dataset**|**Criterion**|**DCI-VTON**|**LaDI-VTON**|**Ours**|
|---|---|---|---|---|
||Image Realism|18.4|21.2|60.4|
|VITON-HD [14]|Garment Detail|17.1|18.6|64.3|
||Body Consistency|18.3|18.6|63.1|
||Image Realism|18.6|19.3|62.1|
|DressCode [29]|Garment Detail|20.1|18.8|61.1|
||Body Consistency|20.1|19.2|60.7|



We collected a total of 1,000 valid responses from over 50 volunteers with diverse backgrounds, ensuring a broad and representative sample. To further reduce subjective bias, each image pair was evaluated independently by at least five different participants. 

The results shown in Table 3 demonstrate a significant preference for our method, selected as superior in more than 60% of trials. 

Our approach consistently outperformed both DCI-VTON and LaDI-VTON across all criteria and datasets. These results indicate that, beyond improvements in objective metrics, our method delivers virtual try-on results that are more convincing and appealing to human observers. 

## **4.6 Ablation Studies and Analysis** 

To thoroughly evaluate the contribution of each component within the DiffFit framework, we conduct a series of ablation studies by systematically removing or modifying key modules. These experiments rigorously assess the impact of architectural design and feature extraction strategies on overall model performance. 

**Feature Extraction Mechanism.** We investigate the effect of alternative feature extraction strategies in both stages of our framework. Specifically, we compare our proposed projection-based approach with a baseline that utilizes only a fixed image encoder, omitting the projection layer: 

- Projection-based: This configuration combines a fixed image encoder ( _E_ ) with a learnable projection layer ( _Q_ ) to map clothing image features into a unified embedding space, producing 768-dimensional semantic tokens ( _tc_ ). These are injected via cross-attention in both the garment warping and the cross-modal fusion UNet modules. 

- w/o Projection-based: This variant uses only the fixed image encoder ( _E_ ), directly injecting the extracted features into the corresponding UNet modules via crossattention, without the projection layer. 

As reported in Table 4, the projection-based method consistently outperforms the baseline across all evaluation metrics. Specifically, our method achieves a lower LPIPS of 0.041, a higher SSIM of 0.916, and improved FID _p_ and KID _p_ scores (6.62 and 

11 

**Table 4** Ablation study on VITON-HD: quantitative results for different settings. 

|Ablation Setting|LPIPS_↓_|SSIM_↑_|FID_p ↓_|KID_p ↓_|FID_u ↓_|KID_u ↓_|
|---|---|---|---|---|---|---|
|w/o Projection-based|0.138|0.651|13.01|4.12|15.12|5.21|
|w/o Garment Condition|0.067|0.787|8.98|2.67|9.51|1.68|
|Ours|0.041|0.916|6.62|1.56|8.21|1.36|



**==> picture [353 x 295] intentionally omitted <==**

**----- Start of picture text -----**<br>
Person Clothing w/o Projection-based w/o Garment Condition Ours<br>**----- End of picture text -----**<br>


**Fig. 5** Qualitative comparison of ablation settings on virtual try-on. From left to right: input person, reference clothing, w/o projection-based, w/o garment condition, and our full method. 

1.56, respectively) compared to the w/o projection-based baseline (0.138, 0.651, 13.01, and 4.12, respectively). Similar improvements are observed in the unpaired garment metrics (FID _u_ /KID _u_ of 8.21/1.36 against 15.12/5.21). 

Figure 5 qualitatively corroborates these findings: in the absence of the projection layer, the generated garments exhibit structural distortions and loss of fine-grained details, while the projection-based approach yields results with superior structural integrity and semantic consistency. 

**Conditional Input Analysis.** To further investigate the significance of conditional inputs in the second-stage synthesis module, we conduct an ablation in which 

12 

the original garment image condition is removed, retaining only the person and warped garment representations ( _w/o Garment Condition_ ). Specifically, the garment image latent ( _zc_ ) is omitted, whereas the person ( _zp_ ) and warped garment ( _zc_ ˜) latents are retained. 

Quantitative results presented in Table 4 demonstrate that removing the garment condition leads to a clear degradation in performance: SSIM decreases from 0.916 (full model) to 0.787, and FID _p_ increases from 6.62 to 8.98. LPIPS also increases from 0.041 to 0.067, and KID _p_ rises from 1.56 to 2.67. Similarly, the unpaired garment FID/KID worsen from 8.21/1.36 to 9.51/1.68. These results collectively indicate that the garment condition is crucial for faithful preservation of garment texture and stylistic consistency in the generated images. 

Qualitative comparisons in Figure 5 further corroborate these findings. Without the original garment condition, the synthesized outputs frequently exhibit structural inconsistencies and loss of detail. For instance, in the first example, the garment appears to float above the body, resulting in unrealistic spatial alignment. In the second example, the clothing is unnaturally suspended at the intersection of the arms, leading to ambiguous boundaries between the garment and body parts. In the third example, visual artifacts such as inaccurate lace patterns and misaligned garment straps are observed. By contrast, the full model consistently demonstrates superior structural integrity and semantic coherence, accurately preserving garment localization and intricate details across diverse cases. 

## **5 Conclusion** 

In this paper, we presented DiffFit, a two-stage latent diffusion framework designed to realize photorealistic virtual try-on (VTON). In the first stage, we established an implicit correspondence between the garment and the target body in the diffusion latent space, enabling pose-adaptive garment deformation while preserving structural and texture details. In the second stage, we synthesized the final try-on image by integrating multi-scale features from the warped garment, the original garment appearance, and the person representation using a hybrid UNet architecture. 

Experimental results demonstrated that DiffFit advanced the VTON paradigm by combining geometry-aware garment warping with effective multi-conditional fusion, achieving superior geometric alignment and visual realism compared to existing baselines. 

Despite these promising results, several limitations remain. For example, the current model is primarily designed for static images and may not generalize well to diverse poses or complex garment types, such as loose or reflective clothing. 

In future work, we plan to explore extensions of DiffFit for dynamic video-based try-on, automatic garment segmentation, and robust performance under real-world variations. Furthermore, we aim to incorporate interactive user control and garment customization, broadening the scope of VTON in practical e-commerce platforms. Through these directions, we hope to further enhance the flexibility, usability, and realism of virtual try-on systems. 

13 

## **References** 

- [1] Wang, C., Zhang, L., Lu, H., Yang, J.: Toward characteristic-preserving imagebased virtual try-on network. In: Proc. Eur. Conf. Comput. Vis. (2018) 

- [2] Han, X., Wu, Z., Wu, Z., Yu, R., Davis, L.S.: Viton: An image-based virtual try-on network. In: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (2018) 

- [3] Shen, F., Jiang, X., He, X., Ye, H., Wang, C., Du, X., Li, Z., Tang, J.: Imagdressing-v1: Customizable virtual dressing. In: Proc. AAAI Conf. Artif. Intell. (2025) 

- [4] Chen, W., Gu, T., Xu, Y., Chen, C.: Magic Clothing: Controllable GarmentDriven Image Synthesis. Preprint at urlhttps://arxiv.org/abs/2404.09512 (2024) 

- [5] Shen, F., Du, X., Gao, Y., Yu, J., Cao, Y., Lei, X., Tang, J.: IMAGHarmony: Controllable Image Editing with Consistent Object Quantity and Layout. Preprint at urlhttps://arxiv.org/abs/2506.01949 (2025) 

- [6] Bai, S., Zhou, H., Li, Z., Zhou, C., Yang, H.: Single stage virtual try-on via deformable attention flows. In: Proc. Eur. Conf. Comput. Vis. (2022) 

- [7] Ge, Y., Song, Y., Zhang, R., Ge, C., Liu, W., Luo, P.: Parser-free virtual try-on via distilling appearance flows. In: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (2021) 

- [8] Lee, S., Gu, G., Park, S., Choi, S., Choo, J.: High-resolution virtual try-on with misalignment and occlusion-handled conditions. In: Proc. Eur. Conf. Comput. Vis. (2022) 

- [9] Morelli, D., Cornia, M., Cucchiara, R.: Semantic-aware virtual try-on network. In: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. Workshops (2022) 

- [10] Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent diffusion models. In: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (2022) 

- [11] Shen, F., Tang, J.: Imagpose: A unified conditional framework for pose-guided person generation. Adv. Neural Inf. Process. Syst. **37** , 6246–6266 (2024) 

- [12] Shen, F., Wang, C., Gao, J., Guo, Q., Dang, J., Tang, J., Chua, T.-S.: Long-Term TalkingFace Generation via Motion-Prior Conditional Diffusion Model. Preprint at urlhttps://arxiv.org/abs/2502.09533 (2025) 

- [13] Shen, F., Yu, J., Wang, C., Jiang, X., Du, X., Tang, J.: IMAGGarment-1: FineGrained Garment Generation for Controllable Fashion Design. Preprint at 

14 

urlhttps://arxiv.org/abs/2504.13176 (2025) 

- [14] Choi, S., Park, S., Lee, M., Choo, J.: Viton-hd: High-resolution virtual try-on via misalignment-aware normalization. In: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (2021) 

- [15] Creswell, A., White, T., Dumoulin, V., Arulkumaran, K., Sengupta, B., Bharath, A.A.: Generative adversarial networks: An overview. In: IEEE Signal Process. Mag. (2018) 

- [16] Xie, Z., Huang, Z., Dong, X., Zhao, F., Dong, H., Zhang, X., Zhu, F., Liang, X.: Gp-vton: Towards general purpose virtual try-on via collaborative localflow global parsing learning. In: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (2023) 

- [17] Yang, H., Zhang, R., Guo, X., Liu, W., Zuo, W., Luo, P.: Towards photorealistic virtual try-on by adaptively generating-preserving image content. In: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., pp. 7850–7859 (2020) 

- [18] Gou, J., Sun, S., Zhang, J., Si, J., Qian, C., Zhang, L.: Taming the power of diffusion models for high-quality virtual try-on with appearance flow. In: Proc. 31st ACM Int. Conf. Multimedia (2023) 

- [19] Morelli, D., Baldrati, A., Cartella, G., Cornia, M., Bertini, M., Cucchiara, R.: Ladi-vton: Latent diffusion textual-inversion enhanced virtual try-on. In: Proc. 31st ACM Int. Conf. Multimedia (2023) 

- [20] Kim, J., Gu, G., Park, M., Park, S., Choo, J.: Stableviton: Learning semantic correspondence with latent diffusion model for virtual try-on. In: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., pp. 8176–8185 (2024) 

- [21] Zhu, L., Yang, D., Zhu, T., Reda, F., Chan, W., Saharia, C., Norouzi, M., Kemelmacher-Shlizerman, I.: Tryondiffusion: A tale of two unets. In: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (2023) 

- [22] Xu, Y., Gu, T., Chen, W., Chen, C.: OOTDiffusion: Outfitting Fusion based Latent Diffusion for Controllable Virtual Try-On. Preprint at urlhttps://arxiv.org/abs/2403.01779 (2024) 

- [23] Shen, F., Ye, H., Zhang, J., Wang, C., Han, X., Yang, W.: Advancing pose-guided image synthesis with progressive conditional diffusion models. Preprint at urlhttps://arxiv.org/abs/2310.06313 (2023) 

- [24] Shen, F., Ye, H., Liu, S., Zhang, J., Wang, C., Han, X., Wei, Y.: Boosting consistency in story visualization with rich-contextual conditional diffusion models. In: Proc. AAAI Conf. Artif. Intell., vol. 39, pp. 6785–6794 (2025) 

15 

- [25] Gao, J., Sun, Y., Shen, F., Jiang, X., Xing, Z., Chen, K., Zhao, C.: Faceshot: Bring any character into life. Preprint at urlhttps://arxiv.org/abs/2503.00740 (2025) 

- [26] Gao, B., Ren, J., Shen, F., Wei, M., Huang, Z.: Exploring warping-guided features via adaptive latent diffusion model for virtual try-on. In: 2024 IEEE Int. Conf. Multimedia Expo (ICME), pp. 1–6 (2024) 

- [27] Wang, C., Tian, K., Guan, Y., Zhang, J., Jiang, Z., Shen, F., Han, X., Gu, Q., Yang, W.: Ensembling Diffusion Models via Adaptive Feature Aggregation. Preprint at 

   - urlhttps://arxiv.org/abs/2405.17082 (2024) 

- [28] Li, D., Li, J., Hoi, S.C.H.: Blip-diffusion: Pre-trained subject representation for controllable text-to-image generation and editing. In: Adv. Neural Inf. Process. Syst. (2023) 

- [29] Morelli, D., Fincato, M., Cornia, M., Landi, F., Cesari, F., Cucchiara, R.: Dress code: High-resolution multi-category virtual try-on. In: Proc. Eur. Conf. Comput. Vis. (ECCV) (2022) 

- [30] Wang, Z., Bovik, A.C., Sheikh, H.R., Simoncelli, E.P.: Image quality assessment: From error visibility to structural similarity. IEEE Transactions on Image Processing **13** (4), 600–612 (2004) https://doi.org/10.1109/TIP.2003.819861 

- [31] Zhang, R., Isola, P., Efros, A.A., Shechtman, E., Wang, O.: The unreasonable effectiveness of deep features as a perceptual metric. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 586–595 (2018). https://doi.org/10.1109/CVPR.2018.00068 

- [32] Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., Hochreiter, S.: Gans trained by a two time-scale update rule converge to a local nash equilibrium. In: Advances in Neural Information Processing Systems (NeurIPS), vol. 30, pp. 6626–6637 (2017) 

- [33] Bi´nkowski, M., Sutherland, D.J., Arbel, M., Gretton, A.: Demystifying mmd Gans. In: International Conference on Learning Representations (ICLR) (2018) 

16 

