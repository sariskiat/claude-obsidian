---
type: paper-fulltext
slug: omnivton-training-free-universal
arxiv_id: null
source_path: /Users/saris.kia.adm/.paper-scholar/omnivton-training-free-universal/2507.15037.md
paper: "[[omnivton-training-free-universal]]"
---
<!-- extracted by afk_extract from 2507.15037.pdf (19p) -->

## **OmniVTON: Training-Free Universal Virtual Try-On** 

Zhaotong Yang[1] , Yuhui Li[1] , Shengfeng He[2] , Xinzhe Li[1] , Yangyang Xu[3] , Junyu Dong[1] , Yong Du[1][*] 1Ocean University of China, 2Singapore Management University, 3Harbin Institute of Technology (Shenzhen) 

**==> picture [56 x 14] intentionally omitted <==**

**----- Start of picture text -----**<br>
      In  Shop<br>**----- End of picture text -----**<br>


**==> picture [73 x 96] intentionally omitted <==**

**==> picture [37 x 49] intentionally omitted <==**

**==> picture [37 x 48] intentionally omitted <==**

_Model-to-Model_ 

**==> picture [37 x 49] intentionally omitted <==**

**==> picture [73 x 96] intentionally omitted <==**

**==> picture [37 x 48] intentionally omitted <==**

_Upper-to-Model_ 

**==> picture [36 x 49] intentionally omitted <==**

**==> picture [73 x 96] intentionally omitted <==**

**==> picture [72 x 96] intentionally omitted <==**

**==> picture [37 x 49] intentionally omitted <==**

**==> picture [37 x 48] intentionally omitted <==**

**==> picture [36 x 49] intentionally omitted <==**

_Lower-to-Model Dress-to-Model_ 

**==> picture [68 x 8] intentionally omitted <==**

**----- Start of picture text -----**<br>
        In the Wild<br>**----- End of picture text -----**<br>


**==> picture [73 x 96] intentionally omitted <==**

**==> picture [37 x 49] intentionally omitted <==**

**==> picture [73 x 96] intentionally omitted <==**

**==> picture [37 x 49] intentionally omitted <==**

**==> picture [37 x 49] intentionally omitted <==**

**==> picture [37 x 49] intentionally omitted <==**

**==> picture [167 x 10] intentionally omitted <==**

**----- Start of picture text -----**<br>
Model-to-Street Shop-to-Street<br>**----- End of picture text -----**<br>


**==> picture [73 x 96] intentionally omitted <==**

**==> picture [36 x 49] intentionally omitted <==**

**==> picture [37 x 49] intentionally omitted <==**

**==> picture [72 x 96] intentionally omitted <==**

**==> picture [36 x 48] intentionally omitted <==**

**==> picture [37 x 49] intentionally omitted <==**

**==> picture [167 x 8] intentionally omitted <==**

**----- Start of picture text -----**<br>
Street-to-Street Multi-Human<br>**----- End of picture text -----**<br>


Figure 1. We propose OmniVTON, a training-free universal virtual try-on framework that unifies both in-shop and in-the-wild scenarios while preserving garment details and ensuring pose consistency. 

## **Abstract** 

_Image-based Virtual Try-On (VTON) techniques rely on either supervised in-shop approaches, which ensure high fidelity but struggle with cross-domain generalization, or unsupervised in-the-wild methods, which improve adaptability but remain constrained by data biases and limited universality. A unified, training-free solution that works across both scenarios remains an open challenge. We propose OmniVTON, the first training-free universal VTON framework that decouples garment and pose conditioning to achieve both texture fidelity and pose consistency across diverse settings. To preserve garment details, we introduce a garment prior generation mechanism that aligns clothing with the body, followed by continuous boundary stitching technique to achieve fine-grained texture retention. For precise pose alignment, we utilize DDIM inversion to capture structural_ 

*Corresponding author (csyongdu@ouc.edu.cn). 

_cues while suppressing texture interference, ensuring accurate body alignment independent of the original image textures. By disentangling garment and pose constraints, OmniVTON eliminates the bias inherent in diffusion models when handling multiple conditions simultaneously. Experimental results demonstrate that OmniVTON achieves superior performance across diverse datasets, garment types, and application scenarios. Notably, it is the first framework capable of multi-human VTON, enabling realistic garment transfer across multiple individuals in a single scene. Code is available at https://github.com/JeromeYoung/OmniVTON ._ 

## **1. Introduction** 

Image-based virtual try-on (VTON) transforms online shopping by seamlessly integrating garment images with target human bodies to generate natural-looking results that 

conform to body poses while preserving texture consistency. It enhances the shopping experience by reducing uncertainty and minimizing return rates. 

Existing VTON methods are designed for either in-shop or in-the-wild scenarios. Supervised approaches [8, 17, 26, 41, 43] dominate in-shop settings, achieving high-fidelity synthesis using paired training data but struggling with cross-domain/-scenario generalization. Conversely, in-thewild methods [11] leverage unsupervised learning to improve adaptability across diverse input sources ( _e.g_ ., Shopto-Street, Model-to-Model, _etc_ .) but remain constrained by data distribution biases and limited universality. Both paradigms rely on dedicated models trained for specific conditions, making large-scale dataset construction across all garment categories, styles, and human poses highly impractical. This fragmentation underscores the need for a unified VTON framework that can generalize across domains without requiring additional training. 

To enable a training-free VTON framework, two critical challenges must be addressed: 

i) _Fine-grained Texture Consistency_ : Without a dedicated training phase, it is difficult to establish garmentbody alignment while preserving intricate texture details. Conventional methods rely on learned deformation priors, which are unavailable in a training-free setting. 

ii) _Human Pose Alignment_ : Existing methods condition on keypoints [5] or DensePose maps [18], requiring retraining for cross-modal feature fusion. Without explicit pose supervision, training-free approaches would struggle with pose consistency, especially for garments with ambiguous structures like sleeveless vests. 

To tackle these issues, we propose _OmniVTON_ , a training-free virtual try-on framework that leverages offthe-shelf diffusion models through a progressive garment adaptation mechanism. For texture preservation, we introduce Structured Garment Morphing (SGM), which ensures seamless garment-body integration while maintaining fine-grained texture details. First, a pseudo-person image is generated via semantic-guided garment completion. Then, multi-part semantic correspondence between the pseudoperson and the source person is established using a predicted segmentation map and skeleton data. Finally, localized transformations dynamically adjust different garment regions to achieve an anatomically accurate alignment, producing a structurally coherent adaptation result. To address inconsistencies along garment boundaries, we propose Continuous Boundary Stitching (CBS), which refines the transitions between segmented regions to ensure seamless integration. By leveraging semantic interactions between the latent features of the original garment image and the garmentinfused image, CBS eliminates harsh edges and discontinuities, preserving the visual realism of the final synthesis. 

For pose information injection, a naive approach is to 

directly apply DDIM Inversion [35], which preserves structural information by replacing the initial random noise with inversion noise from the source person. However, this also introduces unwanted texture contamination. To address this, we propose Spectral Pose Injection (SPI), which selectively integrates pose cues while suppressing texture interference. By leveraging spectral analysis in the latent space, SPI retains low-frequency inversion noise for structural consistency while replacing high-frequency components with random noise to enhance generative flexibility. This frequency-aware modulation maintains pose fidelity while preventing residual texture artifacts. 

Comprehensive experiments demonstrate that OmniVTON surpasses existing methods across multiple benchmarks, both qualitatively and quantitatively, producing high-fidelity try-on results while offering new insights into virtual try-on. Additionally, it showcases strong generalizability across various scenarios, datasets, and clothing types. The key contributions of this work include: 

- We propose OmniVTON, a training-free universal VTON framework that unifies in-shop and in-the-wild scenarios, significantly advancing the state of the art. 

- We introduce Structured Garment Morphing, ensuring fine-grained texture preservation and seamless garment adaptation across diverse clothing types and scenarios. 

- We develop Spectral Pose Injection and Continuous Boundary Stitching to effectively integrate pose information and refine textures, producing pose-consistent and texture-coherent try-on results. 

- Our method achieves state-of-the-art results across multiple evaluation metrics, demonstrating superior quality, generalizability, and scalability. Notably, it is the first to enable multi-human VTON, facilitating realistic garment transfer across multiple individuals. 

## **2. Related work** 

**Garment Warping.** Garment warping plays a fundamental role in virtual try-on by ensuring precise human-body alignment and texture preservation. Early approaches [8, 19] relied on Thin Plate Spline [4] (TPS) deformation with sparse control points, but their low-dimensional parametric representation struggled to accommodate complex pose variations. Later works [20, 22, 40, 43] improved deformation quality by predicting dense optical flow [47] for pixellevel semantic alignment, though they remained reliant on paired training data. To mitigate data constraints, PastaGAN++ [39] introduced a patch-routed disentanglement module for unpaired training. However, existing methods are often tailored to specific scenarios, limiting their adaptability across diverse input sources. In contrast, our OmniVTON enables training-free, universal garment adaptation by leveraging skeletal guidance. While StreetTryOn [11] also supports cross-scenario applications, its dense warping 

**==> picture [494 x 204] intentionally omitted <==**

**----- Start of picture text -----**<br>
Pseudo-Person Image Generation Multi-Part Semantic Correspondence Spectral Pose Injection Continuous Boundary Stitching<br>× T Semantic Parts × T<br>Denoising<br>Denoising U-Net<br>U-Net<br>[  ,  , ]<br>[  ,  , ]<br>Semantic Parts 1-<br>σ ( × ,  ) ×<br>σ (                   ) × , × [     ,    ]<br>↓<br>Attention Modulation σ ( × ,  ) × [   ,  ]<br>Homography<br>× T Estimation<br>Try-on Result<br>× T<br>Denoising<br>U-Net Denoising<br>Homography  Matrix U-Net<br>[  ,  , ] Text Prompt<br>Structured Garment Morphing Garment-Infused Image Inpainting<br>TAPPS DDIM Inversion<br>FFT & Shift<br>IShift & IFFT<br>Localized  ransformation<br>OpenPose T<br>**----- End of picture text -----**<br>


Figure 2. Overview of OmniVTON. It consists of two main steps: 1) Utilize pseudo-person image _Io_ to achieve multi-part deformation, generating adapted clothing. 2) Integrate this prior with clothing-agnostic image _Im_ to create garment-infused image _Ip[′]_[, which is concate-] nated with pose-encoded noise ˆ _zT_ as input, thereby obtaining refined try-on result through the Continuous Boundary Stitching mechanism. 

mechanism struggles with lower garments in in-shop scenarios and fails to preserve garment integrity. Our approach overcomes these limitations, achieving superior versatility and fidelity across a wide range of virtual try-on tasks. 

**Image-Based Virtual Try-On.** Implicit warping-based virtual try-on methods [6, 9, 24, 30, 41, 44] have recently gained attention for their ability to jointly model garment deformation and human-body synthesis using diffusion models’ powerful semantic correspondence capabilities. Ladi-VTON [30], for instance, employs textual inversion [13] to map garment textures to text-based conditions, but its reliance on textual ambiguity results in insufficient control over garment details. To improve garment-body interactions, IDM-VTON [9] and StableVITON [24] incorporate advanced attention mechanisms, yet the absence of explicit deformation constraints often leads to geometric misalignment and texture inconsistencies, particularly in open-domain scenarios. Unlike these methods, OmniVTON provides direct texture guidance through structured garment priors during the inpainting stage, ensuring precise alignment and preserving fine-grained garment details. Its training-free paradigm enables universal applicability across diverse datasets, scenarios, and garment categories. 

**Exemplar-Based Image Inpainting.** Both exemplar-based image inpainting [7, 27, 42] and virtual try-on require accurate feature transfer from reference images to target regions. PBE [42] trains an image encoder to align visual and textual semantics, while AnyDoor [7] enhances texture representation by injecting multi-level high-frequency features into U-Net [34]. However, excessive high-frequency retention often causes style inconsistencies in the generated outputs. In contrast, text-driven inpainting methods [23, 48] suffer 

from limited information granularity, leading to texture distortion, whereas personalized approaches [3, 13] generate more identity-preserving text embeddings but require finetuning. Our proposed Spectral Pose Injection (SPI) offers a novel alternative by abandoning strict high-frequency constraints while leveraging OmniVTON to maintain garment identity consistency. By integrating structured pose-aware noise modulation, SPI ensures that try-on results conform precisely to the target person’s pose without sacrificing texture fidelity. 

## **3. Approach** 

Given a garment-contained[1] image _Ic_ and a target person image _Ip_ , our goal is to seamlessly transfer the indicated garment onto the corresponding semantic region of _Ip_ without any training. To this end, we tackle two key challenges: 

- Warping the given garment in a training-free manner. 

- Preserving the original person’s pose while inpainting the cloth-agnostic image, also without training. 

As illustrated in Fig. 2, OmniVTON follows a two-step workflow. First, it morphs the target garment to create a garment prior aligned with the human body. Then, using this prior and pose-encoded noise, it progressively refines the boundary of the garment and completes the garment-infused image, ensuring a coherent and pose-matching result. 

## **3.1. Structured Garment Morphing** 

We propose Structured Garment Morphing (SGM) to accurately deform the target garment. Unlike TPS and Flowbased methods [8, 40], which require retraining for different 

> 1A garment-contained image _Ic_ refers to either a standalone garment or a person wearing it in diverse backgrounds. 

domains, SGM leverages skeletal information and parsing maps to constrain garment morphing. It establishes a oneto-one mapping between the target garment and the original worn person images using correspondences between the target and source person. While this approach is naturally applicable in Non-Shop-to-X settings, the Shop-to-X setting faces challenges when only the garment image is available, leading to failures in keypoint detection and parsing due to the lack of parseable human body structure. To ensure universality, SGM’s first task is to generate a pseudo-person image from the garment to extract reliable information. 

**Pesudo-Person Image Generation.** We empirically found that text-driven image generation paradigm often fails to produce the desired pseudo-human image _Io_ , likely due to weak controllability or the need for carefully crafted prompts. Instead, we propose modulating attention outputs for generation. Specifically, we first relocate the target garment to the semantically relevant region of the source person’s body, establishing an initial spatial correspondence. To inject the person’s semantic features, we parallel-denoise boththe garmentthe garment-conditionedimage _Ic_ and thenoiseinverted _zt_ , concatenatedcloth mask with _M_ ˜ _c_ , which indicates the region to be generated. We also apply the same noise conditioned on the worn person’s information, concatenated with the cloth-agnostic image _Im_ and the agnostic mask _Mp_ , integrating the latter’s key and value into the self-attention layers of the former: 

**==> picture [211 x 26] intentionally omitted <==**

where _Kp_ and _Vp_ are the key and value matrices of the worn person image, _Qc_ , _Kc_ , and _Vc_ represent the query, key, and value of the garment image, and _∥_ denotes tensor concatenation along the spatial dimension. Since the key and value encode an image’s spatial layout and content information [36], this mechanism effectively incorporates human body semantics into the pseudo-person generation process while preserving the target garment’s texture. 

**Multi-Part Semantic Correspondence.** After obtaining the target person image, we use skeleton and parsing to establish multi-part semantic correspondence between the target garment and the original worn person image. First, we define a set of _N_ human semantic regions and use OpenPose [5] for semantic disentanglement on both images. 

Taking an upper garment as an example, we define five semantic regions: torso, left and right upper arms, and left and right lower arms. Using the 25 keypoints predicted by OpenPose, we construct bounding boxes _{Bo[i][}]_[5] _i_ =1[to][en-] compass all keypoints of each region in _Io_ . Likewise, we obtain the corresponding bounding boxes _{Bp[i][}] i_[5] =1[for] _[ I][p]_[.] 

To avoid interference from overlapping parts, we then apply a human part segmentation map _Po_ , generated by TAPPS [12], to isolate pixels corresponding to each region: 

**==> picture [224 x 25] intentionally omitted <==**

where I( _·_ ) indicates the indicator function, and ( _xo, yo_ ) represents the pixel coordinates of _Io_ . Note that although the generated pseudo-person image may not always capture the entire human body, our relocate operation ensures that at least the outer body, including the garment, is covered. This is sufficient to establish the multi-part correspondence needed for the subsequent localized transformation. 

**Localized Transformations.** For the corner points of each bounding box pair _{Bo[i][, B] p[i][}]_[, we optimize the homography] matrix _Ho[i] →p[∈]_[R][3] _[×]_[3][using][the][Levenberg-Marquardt][al-] gorithm [14]. Then, a piecewise perspective transformation is applied to _Io_ to align it with the source human geometry: 

**==> picture [199 x 37] intentionally omitted <==**

where ( _x[′] o[, y] o[′]_[)][represents][the][pixel][coordinates][at][(] _[x][o][, y][o]_[)] after morphing. In this way, we obtain a coarse deformed garment _Iω_ , which, through multi-region stitching, serves as an effective prior for the subsequent step of garment-infused image inpainting. 

The one-to-one mapping characteristic of SGM eliminates the need for training; however, this multi-region stitching approach results in discontinuities along the boundaries of the morphed garment. We will discuss methods for boundary refinement in Sec. 3.3. 

## **3.2. Spectral Pose Injection** 

The VTON task requires high human pose fidelity, as relying solely on the local deformation from garment morphing is insufficient for full-body pose alignment. This issue is especially noticeable with garments of ambiguous structures, like sleeveless vests or shorts. While skeleton-based conditioning in diffusion models can improve pose controllability, combining it with other conditions, such as text prompts, may cause the model to overfit certain conditions while neglecting others [21]. Alternatively, we apply DDIM Inversion [35] to reverse-map _Ip_ into latent space, obtaining noise _zT[inv]_ that preserves source human body structure. However, _zT[inv]_ also retains the source garment’s texture, which may conflict with the texture generation of the target garment during image inpainting. 

To address this, we propose Spectral Pose Injection (SPI), inspired by our spectral analysis in latent space. As shown in Fig. 3, the human latent, when decomposed via the Fast Fourier Transform (FFT) into low- and high-frequency components, exhibits distinct characteristics in reconstructing the original image. The low-frequency component captures only the coarse human silhouette, which adequately 

**==> picture [207 x 113] intentionally omitted <==**

**----- Start of picture text -----**<br>
Gaussian Mask:<br>(1-     )<br>High-Frequency<br>Component<br>Fourier<br>Input Spectrum<br>Low-Frequency<br>Component<br>IShift & IFFT VAE Decoder<br>VAE Encoder FFT & Shift<br>IShift & IFFT VAE Decoder<br>**----- End of picture text -----**<br>


Figure 3. Visualization of distinct spectral bands in latent space. 

preserves pose information, while the high-frequency component retains both pose and fine garment textures. 

The core idea of SPI is to retain the low-frequency structural information from the inverted noise while leveraging the high-frequency components of random noise to enhance generative flexibility. Specifically, we first apply FFT and centralization to both _zT[inv]_ and a random noise _zT_ : 

**==> picture [213 x 12] intentionally omitted <==**

where _F_ ( _·_ ) denotes the Fourier transform, and Shift( _·_ ) shifts the low-frequency components to the center, facilitating spectral decoupling. 

Next, we perform frequency-domain weighted fusion using a Gaussian low-pass mask _Gτ_ , where _τ_ controls the cutoff frequency: 

**==> picture [191 x 13] intentionally omitted <==**

where _⊙_ denotes element-wise multiplication. The mask _Gτ_ ensures that the low-frequency pose information from _zT[inv]_ is preserved while injecting the high-frequency randomness from _zT_ to eliminate texture residuals. 

Finally, we apply the inverse Fourier transform to the fused spectrum _f_[ˆ] _T_ to obtain the mixed initial noise: 

**==> picture [170 x 13] intentionally omitted <==**

## **3.3. Continuous Boundary Stitching** 

During the inpainting stage, the diffusion model takes the concatenation of mixed noise _z_ ˆ _T_ , the garment-infused image _Ip[′]_[, and the cloth-agnostic mask as input to generate the] final try-on image. Since _Iω_ is assembled by morphing and stitching garment regions, its boundaries may exhibit texture discontinuities. These artifacts can be misinterpreted by the diffusion model as inherent garment details, resulting in unrealistic seams or misaligned patterns in the output. 

To address this, we propose the Continuous Boundary Stitching (CBS) mechanism, which leverages bidirectional semantic context information between _Ic_ and _Ip[′]_[to improve] boundary continuity during the inpainting process. Similar to attention modulation in pseudo-person image generation, CBS operates by manipulating the self-attention outputs. The key difference is that CBS enables dual-path feature exchange, where the interaction from the _Ic_ -path to the 

_I[′] p_[-path is defined as follows:] 

**==> picture [228 x 27] intentionally omitted <==**

where _↓_ represents downsampling _Mc_ to match the dimension of _Vc_ , aiming to suppress interference from the background information of _Ic_ . The operation in Eq. (7) allows the query _Q[′] p_[to][match][the][target][garment][texture,][thereby] bridging discontinuities caused by multi-region stitching. In addition, we also adjust the self-attention output of _Ic_ by using the key from the _Ip[′]_[-path:] 

_Ac_ = Softmax _Qc ·_ [ _Kc ∥ Kp[′]_[]] _[⊤] , fc_ = _Ac_ [: _,_ 1 : _n_ ] _· Vc,_ � ~~_√_~~ _d_ � (8) where _Ac ∈_ R _[n][×]_[2] _[n]_ denotes the self-attention map. This operation enhances the similarity between the attention maps of _Ic_ and _Ip[′]_[,][while suppressing dissimilar values.][As a re-] sult, _Ac_ retains its continuous boundary and adjusts to align with the layout of _Ip[′]_[.][This][optimization][further][improves] the information flow from the _Ic_ -path to the _Ip[′]_[-path,][aid-] ing boundary refinement in the subsequent time step. Note that we exclude _Vp[′]_[to prevent][texture interference][from its] discontinuities, so only the first _n_ columns of the attention map are used in the output calculation. 

## **4. Experiments** 

## **4.1. Experimental Setup** 

**Datasets.** We evaluate OmniVTON on two in-shop datasets (VITON-HD [8], DressCode [29]) and one in-the-wild dataset (DeepFashion2 [15]). VITON-HD provides 2,032 upper garment-model test pairs, while DressCode spans three subcategories (upper, lower, and dresses) with a total of 5,400 test samples. For DeepFashion2, following the StreetTryOn benchmark [11], we constructed a 2,089-image test set covering four try-on scenarios: Shopto-Street, Model-to-Model, Model-to-Street, and Streetto-Street. Input resolution was dynamically adjusted based on the target person source: 512 _×_ 384 for VITONHD/DressCode and 512 _×_ 320 for DeepFashion2. 

**Baselines and Metrics.** We compare OmniVTON against two baseline categories: exemplar-based image editing methods (PBE [42], AnyDoor [7], TIGIC [27], CrossImage [1]) and traditional virtual try-on models (PWS [2], PastaGAN++ [39], GP-VTON [40], CAT-DM [44], D[4] - VTON [43], IDM-VTON [9], and StreetTryOn [11]). Among image editing methods, PBE and AnyDoor leverage large-scale pretraining for image inpainting, while TIGIC and Cross-Image utilize cross-image attention for localized editing. Traditional VTON models, except StreetTryOn, are scenario-specific: PWS and PastaGAN++ are trained on Model-to-Model datasets (DeepFashion [28] and UPT [38], 

|Method|Year|FID_u↓_FID_p↓_SSIM_p↑_LPIPS_p↓_|
|---|---|---|
|PBE [42]<br>AnyDoor [7]<br>TIGIC [27]<br>Cross-Image [1]|2023 (CVPR)<br>2024 (CVPR)<br>2024 (ECCV)<br>2024 (SIGGRAPH)|19.230 17.649<br>0.784<br>0.227<br>14.830 9.922<br>0.796<br>0.164<br>90.338 88.900<br>0.613<br>0.422<br>62.614 57.286<br>0.760<br>0.256|
|GP-VTON [40]<br>CAT-DM [44]<br>D4-VTON [43]<br>IDM-VTON [9]|2023 (CVPR)<br>2024 (CVPR)<br>2024 (ECCV)<br>2024 (ECCV)|51.566 49.196<br>0.810<br>0.249<br>28.869 26.339<br>0.775<br>0.229<br>25.299 23.914<br>0.790<br>0.250<br>23.035 20.460<br>0.812<br>0.147|
|Ours|-|**9.621**<br>**7.758**<br>**0.832**<br>**0.145**|



Table 1. Quantitative comparisons on the VITON-HD dataset [8], where the subscript _u_ and _p_ indicates the unpaired and paired settings, respectively. 

respectively), whereas GP-VTON, CAT-DM, and others focus on Shop-to-Model settings. 

Evaluation follows standard protocols, using Fr´echet Inception Distance (FID) [31] to measure the similarity between generated try-on results and real image distributions. For VITON-HD and DressCode, which contain ground-truth images, we also employ Structural Similarity (SSIM) [37] and Learned Perceptual Image Patch Similarity (LPIPS) [46] to assess structural integrity and texture fidelity. 

## **4.2. Comparison with State-of-the-Art Methods** 

**Quantitative Evaluation.** Tab. 1 presents the quantitative evaluation of OmniVTON on the VITON-HD dataset. To assess _**cross-dataset**_ generalization, all VTON methods were tested using official checkpoints pre-trained on DressCode. OmniVTON outperforms the best-performing baseline by 0.020 in SSIM and 0.002 in LPIPS, confirming its superiority in pose preservation and appearance fidelity. More notably, our method reduces the FID metric by 5.209 in the unpaired setting, demonstrating exceptional crossdomain adaptability. Notably, while the second-best performer, AnyDoor, benefits from training on a dataset that includes VITON-HD samples, leading to favorable FID _u_ and FID _p_ scores, its structural accuracy remains constrained by the absence of geometric garment guidance. 

To evaluate _**cross-type**_ adaptability, we tested VTON methods using VITON-HD pre-trained models (which contain only upper garments) on the DressCode dataset, which includes diverse clothing types. As shown in Tab. 2, OmniVTON achieves substantial improvements across all metrics, outperforming both exemplar-based editing and VTON baselines with at least a 33.4% relative enhancement in FID _u_ . This performance gain stems from the synergistic effects of Structured Garment Morphing (SGM) and Continuous Boundary Stitching (CBS), which collectively enhance robustness across varied garment styles. 

|Method|Year|FID_u↓_FID_p↓_SSIM_p↑_LPIPS_p↓_|
|---|---|---|
|PBE [42]<br>AnyDoor [7]<br>TIGIC [27]<br>Cross-Image [1]|2023 (CVPR)<br>2024 (CVPR)<br>2024 (ECCV)<br>2024 (SIGGRAPH)<br>2023 (CVPR)<br>2024 (CVPR)<br>2024 (ECCV)<br>2024 (ECCV)<br>-|14.851 13.677<br>0.846<br>0.155<br>14.562 14.411<br>0.798<br>0.202<br>64.117 63.531<br>0.749<br>0.319<br>38.438 34.917<br>0.841<br>0.161|
|GP-VTON [40]<br>CAT-DM [44]<br>D4-VTON [43]<br>IDM-VTON [9]||44.753 44.469<br>0.843<br>0.218<br>13.678 12.028<br>0.858<br>0.125<br>22.390 21.435<br>0.841<br>0.152<br>9.685<br>8.377<br>0.842<br>0.138|
|Ours||**6.450**<br>**5.335**<br>**0.865**<br>**0.119**|



Table 2. Quantitative comparisons on the DressCode dataset [29], where the subscript _u_ and _p_ indicates the unpaired and paired settings, respectively. 

||Shop-to-Street|Model-to-Model|Model-to-Street|Street-to-Street|
|---|---|---|---|---|
||FID_↓_|FID_↓_|FID_↓_|FID_↓_|
|PBE [42]<br>AnyDoor [7]<br>TIGIC [27]<br>Cross-Image [1]|81.538<br>50.893<br>100.177<br>69.444|20.181<br>24.235<br>114.151<br>52.310|62.664<br>51.861<br>130.836<br>66.755|36.556<br>35.139<br>121.520<br>57.753|
|CAT-DM [44]<br>D4-VTON [43]<br>IDM-VTON [9]<br>PWS [2]<br>PastaGAN++ [39]<br>StreetTryOn [11]|37.484<br>35.003<br>42.282<br>-<br>-<br>34.054|-<br>-<br>-<br>34.858<br>13.848<br>12.185|-<br>-<br>-<br>77.274<br>71.090<br>34.191|-<br>-<br>-<br>84.990<br>67.016<br>33.039|
|Ours|**33.919**|**8.983**|**33.450**|**23.470**|



Table 3. Quantitative comparisons on the StreetTryOn benchmark [11]. Virtual try-on methods use publicly available models trained on VITON-HD [8], while PWS [2] and PastaGAN++ [39] are trained on DeepFashion [28] and UPT [38], respectively. StreetTryOn results are taken from its original paper. 

compares _**cross-scenario**_ try-on performance. Missing entries (‘-’) denote scenario-specific limitations of certain methods[1] . In Shop-to-Street tasks, D[4] -VTON and StreetTryOn, benefiting from warping priors, outperform priorfree methods, yet OmniVTON surpasses them in body reconstruction through Spectral Pose Injection (SPI). In Model-to-Model, Model-to-Street, and Street-to-Street settings, our training-free framework maintains a significant advantage, even outperforming StreetTryOn despite it being trained on in-domain data. Moreover, StreetTryOn struggles with lower-body garments and dresses in Shop-toStreet and Shop-to-Model tasks, as garment DensePose [10] fails to provide reliable predictions for these clothing categories. In contrast, SGM successfully generates pseudoperson images, ensuring comprehensive cross-scenario applicability. 

**Qualitative Evaluation.** We present qualitative results on the VITON-HD and DressCode datasets in Fig. 4. TIGIC and Cross-Image fail to generate realistic human images due to their lack of task-specific designs. While inpaint- 

Beyond Shop-to-Model scenarios, Tab. 3 systematically 

1We exclude GP-VTON due to unavailable parsing models. 

**==> picture [496 x 303] intentionally omitted <==**

**----- Start of picture text -----**<br>
Input Ours PBE AnyDoor TIGIC Cross-Image GP-VTON CAT-DM D [4] -VTON IDM-VTON<br>(A) VITON-HD<br>(B) DressCode<br>**----- End of picture text -----**<br>


Figure 4. Qualitative results across multiple datasets and clothing types. We provide upper garment try-on results on the VITON-HD dataset [8] (top) and lower garment/dresses visual comparisons on the DressCode dataset [29] (bottom). 

**==> picture [237 x 255] intentionally omitted <==**

**----- Start of picture text -----**<br>
Input Ours PBE AnyDoor CAT-DM D [4] -VTON<br>PWS PASTAGAN++<br>**----- End of picture text -----**<br>


Figure 5. Qualitative results on the StreetTryOn benchmark [11] under different scenarios (from top to bottom): Shop-to-Street, Model-to-Model, Model-to-Street, and Street-to-Street. 

ing models (PBE, AnyDoor) and traditional VTON methods improve human-body generation, they fail to transfer garment textures effectively and introduce noticeable artifacts, especially in cross-domain scenarios. As a GAN-based [16] method, GP-VTON exhibits poor human-body completion due to its limited generalization capability. In contrast, OmniVTON achieves high-fidelity try-on results while preserving garment textures with precision. 

Since StreetTryOn has not released its code, we compare six alternative methods on this benchmark. Among them, CAT-DM and D[4] -VTON are restricted to in-shop garment inputs, while PWS and PastaGAN++ extract target garments from model images. As shown in Fig. 5, although generic inpainting models (PBE, AnyDoor) demonstrate adaptability to different scenarios, they fail to maintain pose and texture consistency between pre- and post-tryon images. Scenario-specific methods like PWS and PastaGAN++, trained on constrained backgrounds, struggle with real-world complexities. OmniVTON consistently achieves accurate garment texture transfer and pose alignment across diverse settings, demonstrating superior generalizability. 

## **4.3. Ablation Analysis** 

We conduct an ablation analysis to assess the contributions of OmniVTON’s core modules through four model vari- 

|Method|SGM CBS SPI|FID_u↓_FID_p↓_SSIM_p↑_LPIPS_p↓_|
|---|---|---|
|Base<br>(A)<br>(B)<br>(C)|✓<br>✓<br>✓<br>✓|18.445 16.878<br>0.773<br>0.222<br>13.303 11.475<br>0.809<br>0.177<br>9.799<br>7.993<br>0.824<br>0.158<br>13.148 10.767<br>0.813<br>0.180|
|OmniVTON|✓<br>✓<br>✓|**9.621**<br>**7.758**<br>**0.832**<br>**0.145**|



Table 4. Ablation study of the Structured Garment Morphing (SGM), Continuous Boundary Stitching (CBS), and Spectual Pose Injection (SPI) on the VITON-HD dataset [8]. 

**==> picture [209 x 12] intentionally omitted <==**

**----- Start of picture text -----**<br>
w/ SGM<br>Input Base w/ SGM & CBS w/ SPI Ours<br>**----- End of picture text -----**<br>


**==> picture [40 x 53] intentionally omitted <==**

**==> picture [41 x 53] intentionally omitted <==**

**==> picture [40 x 53] intentionally omitted <==**

**==> picture [40 x 53] intentionally omitted <==**

**==> picture [41 x 53] intentionally omitted <==**

**==> picture [40 x 53] intentionally omitted <==**

**==> picture [40 x 54] intentionally omitted <==**

**==> picture [41 x 54] intentionally omitted <==**

**==> picture [40 x 54] intentionally omitted <==**

**==> picture [40 x 54] intentionally omitted <==**

**==> picture [41 x 54] intentionally omitted <==**

**==> picture [40 x 54] intentionally omitted <==**

Figure 6. Qualitative ablation study on different variants. 

ants. Starting from a baseline model (Base) using only text prompts, we incrementally integrate: (A) Structured Garment Morphing (SGM) for garment priors, (B) Continuous Boundary Stitching (CBS) for boundary refinement, and (C) Spectral Pose Injection (SPI) for pose-aware noise control. OmniVTON combines all components. 

**Effectiveness of SGM.** As shown in Tab. 4, (A) significantly outperforms Base across all metrics, confirming that SGM effectively aligns garments without end-to-end training. Fig. 6 further illustrates its ability to preserve garment textures via fine-grained geometric cues. 

**Effectiveness of CBS.** CBS eliminates boundary artifacts (red box, Fig. 6) and refines textures (blue box). Variant (B) improves LPIPS by 0.019 over (A), validating its role in enhancing perceptual quality. Since CBS primarily refines SGM-derived priors, we focus on their combined effect. **Effectiveness of SPI.** As seen in Tab. 4, (C) and OmniVTON show notable SSIM and FID _u_ gains, confirming SPI’s ability to suppress noise contamination while preserving structural consistency. Fig. 6 (green circle) highlights improved body part alignment, reducing pose misalignment. Integrating all components, OmniVTON achieves state-ofthe-art texture fidelity and pose consistency. 

## **4.4. Multi-Human Virtual Try-On** 

Beyond single-human virtual try-on, our method extends to multi-human interactive group try-on (Fig. 7). This capability arises from the innovative design of SGM, which enables seamless adaptation for multiple users. Given multiple garments, we concatenate them along spatial dimensions to generate multiple pseudo-person images simulta- 

**==> picture [119 x 119] intentionally omitted <==**

**==> picture [119 x 119] intentionally omitted <==**

**==> picture [45 x 59] intentionally omitted <==**

**==> picture [90 x 119] intentionally omitted <==**

**==> picture [90 x 119] intentionally omitted <==**

**==> picture [45 x 59] intentionally omitted <==**

Figure 7. Multi-human virtual try-on. Top row shows Model-toModel, while bottom row depicts Shop-to-Street. 

neously. By leveraging positional and semantic cues from skeleton and parsing maps, our approach allows for the effortless application of identical or distinct garments to multiple humans. Multi-human try-on broadens the scope of virtual try-on tasks, further demonstrating the universality of our method. This extension opens new directions for group-centric fashion experiences, such as coordinated family outfits and uniform design. 

## **5. Conclusions, Limitations, and Future Work** 

In this paper, we present OmniVTON, a training-free universal framework that ensures both texture fidelity and pose consistency across diverse settings. Structured Garment Morphing enables anatomical garment-body alignment, while Continuous Boundary Stitching ensures seamless texture transitions, achieving fine-grained texture consistency without domain-specific training. Spectral Pose Injection further enhances pose alignment through frequencyaware modulation of inversion noise, preserving structural integrity while eliminating texture contamination. Extensive experiments demonstrate OmniVTON’s superiority in flexibility and generalization, particularly in its pioneering capability for multi-human VTON. 

Despite its effectiveness, OmniVTON encounters challenges in extreme cases, such as high-density crowds or minimal target body regions, leading to garment misalignment. Visual results illustrating these limitations are provided in the supplementary materials. Future work will focus on developing more robust multi-human try-on frameworks to address these edge cases. 

## **Acknowledgements** 

This work is supported by the National Natural Science Foundation of China (No. 62102381, 41927805); Shandong Natural Science Foundation (No. ZR2021QF035); the National Key R&D Program of China (No. 2022ZD0117201); the Guangdong Natural Science Funds for Distinguished Young Scholar (No.2023B1515020097); the National Research Foundation, Singapore under its AI Singapore Programme (No.AISG3-GV-2023-011); the Singapore Ministry of Education AcRF Tier 1 Grant (No. MSS25C004); and the Lee Kong Chian Fellowships. 

## **References** 

- [1] Yuval Alaluf, Daniel Garibi, Or Patashnik, Hadar AverbuchElor, and Daniel Cohen-Or. Cross-image attention for zeroshot appearance transfer. In _ACM SIGGRAPH_ , pages 1–12, 2024. 5, 6 

- [2] Badour AlBahar, Jingwan Lu, Jimei Yang, Zhixin Shu, Eli Shechtman, and Jia-Bin Huang. Pose with style: Detailpreserving pose-guided image synthesis with conditional stylegan. _ACM TOG_ , 40(6):1–11, 2021. 5, 6 

- [3] Moab Arar, Rinon Gal, Yuval Atzmon, Gal Chechik, Daniel Cohen-Or, Ariel Shamir, and Amit H. Bermano. Domainagnostic tuning-encoder for fast personalization of text-toimage models. In _SIGGRAPH Asia_ , pages 1–10, 2023. 3 

- [4] Fred L. Bookstein. Principal warps: Thin-plate splines and the decomposition of deformations. _IEEE TPAMI_ , 11(6): 567–585, 1989. 2 

- [5] Zhe Cao, Tomas Simon, Shih-En Wei, and Yaser Sheikh. Realtime multi-person 2d pose estimation using part affinity fields. In _CVPR_ , pages 7291–7299, 2017. 2, 4 

- [6] Mengting Chen, Xi Chen, Zhonghua Zhai, Chen Ju, Xuewen Hong, Jinsong Lan, and Shuai Xiao. Wear-any-way: Manipulable virtual try-on via sparse correspondence alignment. In _ECCV_ , pages 124–142. Springer, 2024. 3 

- [7] Xi Chen, Lianghua Huang, Yu Liu, Yujun Shen, Deli Zhao, and Hengshuang Zhao. Anydoor: Zero-shot object-level image customization. In _CVPR_ , pages 6593–6602, 2024. 3, 5, 6 

- [8] Seunghwan Choi, Sunghyun Park, Minsoo Lee, and Jaegul Choo. Viton-hd: High-resolution virtual try-on via misalignment-aware normalization. In _CVPR_ , pages 14131– 14140, 2021. 2, 3, 5, 6, 7, 8, 12 

- [9] Yisol Choi, Sangkyung Kwak, Kyungmin Lee, Hyungwon Choi, and Jinwoo Shin. Improving diffusion models for authentic virtual try-on in the wild. In _ECCV_ , pages 206–235. Springer, 2024. 3, 5, 6 

- [10] Aiyu Cui, Sen He, Tao Xiang, and Antoine Toisoul. Learning garment densepose for robust warping in virtual try-on. _arXiv preprint arXiv:2303.17688_ , 2023. 6 

- [11] Aiyu Cui, Jay Mahajan, Viraj Shah, Preeti Gomathinayagam, Chang Liu, and Svetlana Lazebnik. Street tryon: Learning in-the-wild virtual try-on from unpaired person images. In _WACV_ , pages 8235–8239, 2025. 2, 5, 6, 7, 12 

- [12] Daan De Geus and Gijs Dubbelman. Task-aligned part-aware panoptic segmentation through joint object-part representations. In _CVPR_ , pages 3174–3183, 2024. 4 

- [13] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel CohenOr. An image is worth one word: Personalizing text-toimage generation using textual inversion. _arXiv preprint arXiv:2208.01618_ , 2022. 3 

- [14] Henri P Gavin. The levenberg-marquardt algorithm for nonlinear least squares curve-fitting problems. _Department of Civil and Environmental Engineering Duke University August_ , 3:1–23, 2019. 4 

- [15] Yuying Ge, Ruimao Zhang, Xiaogang Wang, Xiaoou Tang, and Ping Luo. Deepfashion2: A versatile benchmark for detection, pose estimation, segmentation and re-identification of clothing images. In _CVPR_ , pages 5337–5345, 2019. 5 

- [16] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In _NeurIPS_ , 2014. 7 

- [17] Junhong Gou, Siyu Sun, Jianfu Zhang, Jianlou Si, Chen Qian, and Liqing Zhang. Taming the power of diffusion models for high-quality virtual try-on with appearance flow. In _ACM MM_ , pages 7599–7607, 2023. 2 

- [18] Rıza Alp G¨uler, Natalia Neverova, and Iasonas Kokkinos. Densepose: Dense human pose estimation in the wild. In _CVPR_ , pages 7297–7306, 2018. 2 

- [19] Xintong Han, Zuxuan Wu, Zhe Wu, Ruichi Yu, and Larry S Davis. Viton: An image-based virtual try-on network. In _CVPR_ , pages 7543–7552, 2018. 2 

- [20] Xintong Han, Xiaojun Hu, Weilin Huang, and Matthew R Scott. Clothflow: A flow-based model for clothed person generation. In _ICCV_ , pages 10471–10480, 2019. 2 

- [21] Yucheng Han, Rui Wang, Chi Zhang, Juntao Hu, Pei Cheng, Bin Fu, and Hanwang Zhang. Emma: Your text-to-image diffusion model can secretly accept multi-modal prompts. _arXiv preprint arXiv:2406.09162_ , 2024. 4 

- [22] Sen He, Yi-Zhe Song, and Tao Xiang. Style-based global appearance flow for virtual try-on. In _CVPR_ , pages 3470– 3479, 2022. 2 

- [23] Xuan Ju, Xian Liu, Xintao Wang, Yuxuan Bian, Ying Shan, and Qiang Xu. Brushnet: A plug-and-play image inpainting model with decomposed dual-branch diffusion. In _ECCV_ , pages 150–168. Springer, 2024. 3 

- [24] Jeongho Kim, Guojung Gu, Minho Park, Sunghyun Park, and Jaegul Choo. Stableviton: Learning semantic correspondence with latent diffusion model for virtual try-on. In _CVPR_ , pages 8176–8185, 2024. 3 

- [25] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In _ICCV_ , pages 4015–4026, 2023. 11 

- [26] Sangyun Lee, Gyojung Gu, Sunghyun Park, Seunghwan Choi, and Jaegul Choo. High-resolution virtual try-on with misalignment and occlusion-handled conditions. In _ECCV_ , pages 204–219. Springer, 2022. 2 

- [27] Pengzhi Li, Qiang Nie, Ying Chen, Xi Jiang, Kai Wu, Yuhuan Lin, Yong Liu, Jinlong Peng, Chengjie Wang, and Feng Zheng. Tuning-free image customization with image and text guidance. In _ECCV_ , pages 233–250. Springer, 2024. 3, 5, 6 

- [28] Ziwei Liu, Ping Luo, Shi Qiu, Xiaogang Wang, and Xiaoou Tang. Deepfashion: Powering robust clothes recognition and retrieval with rich annotations. In _CVPR_ , pages 1096–1104, 2016. 5, 6 

- [29] Davide Morelli, Matteo Fincato, Marcella Cornia, Federico Landi, Fabio Cesari, and Rita Cucchiara. Dress code: Highresolution multi-category virtual try-on. In _CVPR_ , pages 2231–2235, 2022. 5, 6, 7, 12 

- [30] Davide Morelli, Alberto Baldrati, Giuseppe Cartella, Marcella Cornia, Marco Bertini, and Rita Cucchiara. Ladi-vton: Latent diffusion textual-inversion enhanced virtual try-on. In _ACM MM_ , pages 8580–8589, 2023. 3 

- [31] Gaurav Parmar, Richard Zhang, and Jun-Yan Zhu. On aliased resizing and surprising subtleties in gan evaluation. In _CVPR_ , pages 11410–11420, 2022. 6 

- [32] pharmapsychotic. Clip-interrogator. https://github. com / pharmapsychotic / clip - interrogator, 2023. 11 

- [33] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In _CVPR_ , pages 10684– 10695, 2022. 11 

   - [42] Binxin Yang, Shuyang Gu, Bo Zhang, Ting Zhang, Xuejin Chen, Xiaoyan Sun, Dong Chen, and Fang Wen. Paint by example: Exemplar-based image editing with diffusion models. In _CVPR_ , pages 18381–18391, 2023. 3, 5, 6 

   - [43] Zhaotong Yang, Zicheng Jiang, Xinzhe Li, Huiyu Zhou, Junyu Dong, Huaidong Zhang, and Yong Du. D[4] -vton: Dynamic semantics disentangling for differential diffusion based virtual try-on. In _ECCV_ , pages 36–52. Springer, 2024. 2, 5, 6 

   - [44] Jianhao Zeng, Dan Song, Weizhi Nie, Hongshuo Tian, Tongtong Wang, and An-An Liu. Cat-dm: Controllable accelerated virtual try-on with diffusion model. In _CVPR_ , pages 8372–8382, 2024. 3, 5, 6 

   - [45] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In _ICCV_ , pages 3836–3847, 2023. 12 

   - [46] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In _CVPR_ , pages 586–595, 2018. 6 

   - [47] Tinghui Zhou, Shubham Tulsiani, Weilun Sun, Jitendra Malik, and Alexei A Efros. View synthesis by appearance flow. In _ECCV_ , pages 286–301. Springer, 2016. 2 

   - [48] Junhao Zhuang, Yanhong Zeng, Wenran Liu, Chun Yuan, and Kai Chen. A task is worth one word: Learning with task prompts for high-quality versatile image inpainting. In _ECCV_ , pages 195–211. Springer, 2024. 3 

- [34] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In _MICCAI_ , pages 234–241. Springer, 2015. 3 

- [35] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. _ICLR_ , 2020. 2, 4 

- [36] Yoad Tewel, Rinon Gal, Gal Chechik, and Yuval Atzmon. Key-locked rank one editing for text-to-image personalization. In _ACM SIGGRAPH_ , pages 1–11, 2023. 4 

- [37] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. _IEEE TIP_ , 13(4):600–612, 2004. 6 

- [38] Zhenyu Xie, Zaiyu Huang, Fuwei Zhao, Haoye Dong, Michael Kampffmeyer, and Xiaodan Liang. Towards scalable unpaired virtual try-on via patch-routed spatiallyadaptive gan. _NeurIPS_ , 34:2598–2610, 2021. 5, 6 

- [39] Zhenyu Xie, Zaiyu Huang, Fuwei Zhao, Haoye Dong, Michael Kampffmeyer, Xin Dong, Feida Zhu, and Xiaodan Liang. Pasta-gan++: A versatile framework for high-resolution unpaired virtual try-on. _arXiv preprint arXiv:2207.13475_ , 2022. 2, 5, 6 

- [40] Zhenyu Xie, Zaiyu Huang, Xin Dong, Fuwei Zhao, Haoye Dong, Xijin Zhang, Feida Zhu, and Xiaodan Liang. Gpvton: Towards general purpose virtual try-on via collaborative local-flow global-parsing learning. In _CVPR_ , pages 23550–23559, 2023. 2, 3, 5, 6 

- [41] Yuhao Xu, Tao Gu, Weifeng Chen, and Chengcai Chen. Ootdiffusion: Outfitting fusion based latent diffusion for controllable virtual try-on. _arXiv preprint arXiv:2403.01779_ , 2024. 2, 3 

## **OmniVTON: Training-Free Universal Virtual Try-On** 

## Supplementary Material 

## **A. Details and Discussion** 

## **A.1. Implementation Details** 

All experiments were conducted using PyTorch 2.1.1 on a NVIDIA GeForce RTX 3090 GPU. We adopted Stable Diffusion v2 [33] as the base model, retaining default hyperparameter configurations. For both Pseudo-Person Image Generation and Garment-Infused Image Inpainting, we employed the standard DDIM sampler for deterministic inference with 50 time steps. For Spetral Pose Injection, we set the standard deviation _τ_ of the Gaussian mask to 0.1. 

During the garment morphing stage, we implemented distinct region segmentation strategies for different garment categories: 1) Upper garments underwent five-region processing (left and right upper arms, left and right lower arms, and torso regions); 2) Lower garments were similarly decoupled into five regions (left and right upper legs, left and right lower legs, and hip-above regions); 3) Dresses were segmented into upper and lower garment sections for separate processing. The agnostic and clothing masks are provided by the dataset. In practical applications, SAM [25] can be used to obtain the mask corresponding to the user input image. 

## **A.2. Text Prompts Acquisition** 

Here, we describe the process of acquiring text prompts and examine their impact. Specifically, we convert images into text using the CLIP Interrogator [32], where the generated descriptions consist of a core caption and auxiliary modifier terms. The core caption directly describes the image content, while the auxiliary terms are selected based on cosine similarity between garment features and text embeddings from four predefined datasets: artists, mediums, movements, and flavors. 

To verify the importance of text prompts in virtual try-on tasks, we conducted a controlled analysis using a generic prompt (“a person wearing an upper garment”). As shown in Fig. 8, more detailed text prompts lead to try-on results with enhanced identity consistency, highlighting the crucial role of precise textual descriptions in controlling the quality of generation. 

## **A.3. Additional Ablation Analysis** 

To demonstrate the rationale behind our component design, we conducted additional ablation experiments. First, for SGM, the role of semantic parsing is to perform pixellevel segmentation on skeleton-divided semantic regions, enabling multi-part decoupling. As shown in the upper part of Fig. 9, relying solely on bounding box-based segmenta- 

**==> picture [199 x 11] intentionally omitted <==**

**----- Start of picture text -----**<br>
Input Generic Prompt Detailed Prompt<br>**----- End of picture text -----**<br>


**==> picture [80 x 106] intentionally omitted <==**

**==> picture [79 x 106] intentionally omitted <==**

**==> picture [80 x 106] intentionally omitted <==**

**==> picture [80 x 106] intentionally omitted <==**

**==> picture [79 x 106] intentionally omitted <==**

**==> picture [80 x 106] intentionally omitted <==**

Figure 8. Influence of different text prompts. 

**==> picture [237 x 274] intentionally omitted <==**

**----- Start of picture text -----**<br>
w/o semantic<br>Input Ours Warped Result parsing Warped Result<br>Input Ours w/o high-frequency noise w/ average noisew/o SPI +  w/ ControlNetw/o SPI +<br>**----- End of picture text -----**<br>


Figure 9. Qualitative results of additional ablation analysis. 

tions, without semantic parsing, for localized transformations leads to erroneous morphing and part overlap, signifi- 

|Method|FID_u↓_FID_p↓_SSIM_p↑_LPIPS_p↓_|
|---|---|
|OmniVTON<br>w/o semantic parsing<br>w/o_Ic_-path attention modulation|**9.621**<br>**7.758**<br>0.832<br>**0.145**<br>13.705 11.930<br>0.817<br>0.170<br>9.808<br>7.939<br>0.831<br>0.149|
|w/o high-frequency noise<br>w/o SPI + w/ average noise<br>w/o SPI + w/ ControlNet|15.817 14.558<br>0.836<br>0.182<br>12.402 10.650<br>**0.849**<br>0.151<br>10.873 9.016<br>0.818<br>0.168|



|_τ_<br>0.01<br>0.05<br>**0.1**<br>0.3<br>0.5|FID_u↓_<br>FID_p↓_<br>SSIM_p↑_<br>LPIPS_p↓_|
|---|---|
||9.941<br>8.185<br>0.823<br>0.160<br>9.620<br>8.033<br>0.829<br>0.153<br>9.621<br>7.758<br>0.832<br>0.145<br>10.056<br>8.150<br>0.842<br>0.140<br>11.330<br>9.422<br>0.852<br>0.138|



Table 6. Sensitivity analysis of cutoff frequency _τ_ on VITON-HD. 

Table 5. More ablation studies of different components. 

cantly degrading the quality of the try-on results. The quantitative comparison of the “w/o semantic parsing” setting in Tab. 5 strongly reinforces the necessity of this component. Secondly, the “w/o _Ic_ -path attention modulation” setting involves replacing the attention modulation in Eq. (8) of the main paper with the original self-attention mechanism, resulting in noticeable degradation across all evaluation metrics, thus validating the effectiveness of bidirectional semantic context interaction. 

For SPI, the lower part of Tab. 5 and Fig. 9 present both quantitative and qualitative results for different variants. The “w/o high-frequency noise” variant retains only the low-frequency components of inversion noise, yet the absence of high-frequency noise leads to overly smoothed results. The “w/o SPI + w/ average noise” variant averages random noise and inversion noise as the initial noise. Compared with the “w/o high-frequency noise” variant, the introduction of random noise significantly improves perceptual quality. However, due to the lack of frequency-domain decoupling, this variant enhances performance in paired settings but fails to suppress source garment texture interference from inversion noise in unpaired settings, causing performance degradation. Furthermore, comparative experiments with ControlNet [45]-based skeleton-conditioned injection demonstrate that OmniVTON effectively overcomes the inherent bias of diffusion models in handling multiple conditions by decoupling garment and pose constraints, leading to improved try-on results. 

In Tab. 6, we provide additional analysis on the sensitivity of the cutoff frequency _τ_ . When _τ_ is too small, it suppresses low-frequency pose information, limiting SSIM. As _τ_ increases, metrics generally improve; however, if _τ_ becomes too large, it preserves excessive high-frequency details, which harms realism and worsens FID. Setting _τ_ = 0 _._ 1 balances pose consistency and visual fidelity. 

## **A.4. Inference Cost** 

As shown in the upper part of Tab. 7, we compare the inference costs of OmniVTON with three state-of-the-art methods. The results show that OmniVTON achieves the lowest memory consumption, outperforms Cross-Image in inference speed, and performs comparably to TIGIC and IDMVTON, all while maintaining optimal performance. The 

**==> picture [113 x 68] intentionally omitted <==**

**==> picture [113 x 68] intentionally omitted <==**

**==> picture [225 x 80] intentionally omitted <==**

**----- Start of picture text -----**<br>
(A) VITON-HD (B) DressCode<br>(C) Shop-to-Street (D) Other Scenarios<br>**----- End of picture text -----**<br>


Figure 10. User study on the VITON-HD dataset [8], DressCode dataset [29] and StreetTryOn benchmark [11]. 

lower part of the table further presents a module-wise breakdown of inference times. Notably, under the Non-Shopto-X setting, removing the pseudo-person generation step leads to a sharp reduction in the runtime of the SGM module, from 6.61s to just 0.14s, thereby reducing the overall inference time to 9.82 seconds and further highlighting OmniVTON’s strong potential for real-world deployment. 

## **A.5. User Study** 

We validate the effectiveness of our method through a rigorously designed user study, establishing a systematic evaluation framework across three benchmark datasets: VITONHD [8], DressCode [29], and StreetTryOn [11]. The experiment involved 100 volunteers, each participating in a visual evaluation questionnaire containing 100 comparative sample groups. Specifically, the VITON-HD dataset includes 20 test sample groups, the DressCode dataset covers 40 sample groups across three garment categories (upper, lower, dresses), and the StreetTryOn benchmark allocates the remaining 40 sample groups with a scenario-balanced distribution. Each task in the questionnaire asks, “Which method generates more realistic and accurate images?” with randomized option ordering to ensure unbiased results. As shown in Fig. 10, our method demonstrates significant superiority across all benchmarks. 

|Time / Memory||Training-free|Cross-Image<br>41.49s<br>15,748MB|Training|
|---|---|---|---|---|
||OmniVTON<br>16.29s<br>11,542MB|TIGIC<br>13.87s<br>23,578MB||IDM-VTON|
|||||11.87s<br>17,936MB|
||||||
|OmniVTON|FID_u↓_<br>9.621|SGM Time (s)<br>6.61s|SPI Time (s)<br>3.60s|CBS Time (s)|
|||||6.08s|



Table 7. Runtime and memory comparison on VITON-HD. 

**==> picture [186 x 9] intentionally omitted <==**

**----- Start of picture text -----**<br>
Person Garment Result<br>**----- End of picture text -----**<br>


**==> picture [80 x 106] intentionally omitted <==**

**==> picture [80 x 106] intentionally omitted <==**

**==> picture [79 x 106] intentionally omitted <==**

**==> picture [80 x 106] intentionally omitted <==**

**==> picture [80 x 106] intentionally omitted <==**

**==> picture [79 x 106] intentionally omitted <==**

in the StreetTryOn benchmark. 

## **B.2. More Try-on Results** 

As shown in Fig. 18, we further showcase various garmentmodel combinations, including virtual try-on results for lower-body garments and dresses under the Shop-to-Street scenario. This highlights OmniVTON’s ability to overcome the technical barriers that previously limited the performance of StreetTryOn in this task. 

Figure 11. Failure cases of our method. 

## **A.6. Failure Case Visualizations** 

We present several failure cases of OmniVTON in Fig. 11. As discussed in the main paper, our method encounters challenges in handling high-density crowds and targets with minimal visible body regions. These limitations primarily stem from OmniVTON’s partial reliance on pre-trained modules such as OpenPose and TAPPS, whose predictions can be unreliable under such extreme conditions. Such observations point to a promising direction for future work towards more robust and adaptable universal virtual try-on systems. 

## **B. Additional Visual Results** 

## **B.1. Visual Comparisons with SOTAs** 

Fig. 12 and Fig. 13 present supplementary visual comparisons between OmniVTON and baseline methods on the VITON-HD and DressCode datasets, respectively. While Fig. 14, Fig. 15, Fig. 16, and Fig. 17 showcase detailed visualized results of different methods across four scenarios 

**==> picture [496 x 585] intentionally omitted <==**

**----- Start of picture text -----**<br>
Input Ours PBE AnyDoor TIGIC Cross-Image GP-VTON CAT-DM D [4] -VTON IDM-VTON<br>Figure 12. Qualitative comparison on the VITON-HD dataset.<br>Input Ours PBE AnyDoor TIGIC Cross-Image GP-VTON CAT-DM D [4] -VTON IDM-VTON<br>**----- End of picture text -----**<br>


Figure 13. Qualitative comparison on the DressCode dataset. 

**==> picture [496 x 355] intentionally omitted <==**

**----- Start of picture text -----**<br>
Input Ours PBE AnyDoor TIGIC Cross-Image CAT-DM D [4] -VTON IDM-VTON<br>**----- End of picture text -----**<br>


Figure 14. Qualitative comparison for Shop-to-Street scenario on the StreetTryOn benchmark. 

**==> picture [496 x 401] intentionally omitted <==**

**----- Start of picture text -----**<br>
Input Ours PBE AnyDoor TIGIC Cross-Image PWS PASTAGAN++<br>**----- End of picture text -----**<br>


Figure 15. Qualitative comparison for Model-to-Model scenario on the StreetTryOn benchmark. 

**==> picture [496 x 401] intentionally omitted <==**

**----- Start of picture text -----**<br>
Input Ours PBE AnyDoor TIGIC Cross-Image PWS PASTAGAN++<br>**----- End of picture text -----**<br>


Figure 16. Qualitative comparison for Model-to-Street scenario on the StreetTryOn benchmark. 

**==> picture [496 x 401] intentionally omitted <==**

**----- Start of picture text -----**<br>
Input Ours PBE AnyDoor TIGIC Cross-Image PWS PASTAGAN++<br>**----- End of picture text -----**<br>


Figure 17. Qualitative comparison for Street-to-Street scenario on the StreetTryOn benchmark. 

**==> picture [61 x 79] intentionally omitted <==**

**----- Start of picture text -----**<br>
Garment<br>Human<br>**----- End of picture text -----**<br>


**==> picture [65 x 105] intentionally omitted <==**

**==> picture [65 x 105] intentionally omitted <==**

**==> picture [65 x 105] intentionally omitted <==**

**==> picture [65 x 105] intentionally omitted <==**

**==> picture [66 x 87] intentionally omitted <==**

**==> picture [66 x 105] intentionally omitted <==**

**==> picture [66 x 105] intentionally omitted <==**

**==> picture [66 x 105] intentionally omitted <==**

**==> picture [66 x 105] intentionally omitted <==**

**==> picture [66 x 87] intentionally omitted <==**

**==> picture [66 x 105] intentionally omitted <==**

**==> picture [66 x 105] intentionally omitted <==**

**==> picture [66 x 105] intentionally omitted <==**

**==> picture [66 x 105] intentionally omitted <==**

**==> picture [66 x 87] intentionally omitted <==**

**==> picture [66 x 105] intentionally omitted <==**

**==> picture [66 x 105] intentionally omitted <==**

**==> picture [66 x 105] intentionally omitted <==**

**==> picture [66 x 105] intentionally omitted <==**

**==> picture [66 x 87] intentionally omitted <==**

**==> picture [66 x 105] intentionally omitted <==**

**==> picture [66 x 105] intentionally omitted <==**

**==> picture [66 x 105] intentionally omitted <==**

**==> picture [66 x 105] intentionally omitted <==**

Figure 18. More try-on results of OmniVTON across various clothing types and scenarios. 

