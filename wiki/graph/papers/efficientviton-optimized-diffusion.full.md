---
type: paper-fulltext
slug: efficientviton-optimized-diffusion
arxiv_id: null
source_path: /Users/saris.kia.adm/.paper-scholar/efficientviton-optimized-diffusion/2501.11776.md
paper: "[[efficientviton-optimized-diffusion]]"
---
<!-- extracted by afk_extract from 2501.11776.pdf (7p) -->

1 

# EfficientVITON: An Efficient Virtual Try-On Model using Optimized Diffusion Process 

Mostafa Atef _[∗]_ , Mariam Ayman _[∗]_ , Ahmed Rashed _[∗]_ , Ashrakat Saeed _[∗]_ , Abdelrahman Saeed _[∗]_ , Ahmed Fares _[∗] ∗_ Department of Computer Science and Engineering, Egypt-Japan University of Science and Technology, Alexandria, Egypt. 

_**Abstract**_ **—Wouldn’t it be much more convenient for everybody to try on clothes by only looking into a mirror? The answer to that problem is “virtual try-on,” enabling users to digitally experiment with outfits. The core challenge lies in realistic imageto-image translation, where clothing must fit diverse human forms, poses, and figures. Early methods, which used 2D transformations, offered speed, but image quality was often disappointing and lacked the nuance of deep learning. Though GAN-based techniques enhanced realism, their dependence on paired data proved limiting. More adaptable methods offered great visuals but demanded significant computing power and time. Recent advances in diffusion models have shown promise for highfidelity translation, yet the current crop of virtual try-on tools still struggle with detail loss and warping issues. To tackle these challenges, this paper proposes EfficientVITON: a new virtual try-on system leveraging the impressive pre-trained Stable Diffusion model for better images and deployment feasibility. The system includes a spatial encoder to maintain clothing’s finer details and zero cross-attention blocks to capture the subtleties of how clothes fit a human body. Input images are carefully prepared, and the diffusion process has been tweaked to significantly cut generation time without image quality loss. The training process involves two distinct stages of fine-tuning, carefully incorporating a balance of loss functions to ensure both accurate try-on results and high-quality visuals. Rigorous testing on the VITON-HD dataset, supplemented with real-world examples, has demonstrated that EfficientVITON achieves stateof-the-art results.** 

_**Index Terms**_ **—Virtual Try-On, Diffusion Models, Deep Learning, Computer Vision, Image Synthesis, E-commerce.** 

## I. INTRODUCTION 

The rapid advancement of artificial intelligence (AI) and computer vision has revolutionized various industries, notably fashion and retail, with the emergence of virtual tryon technology. This technology allows users to digitally ”try on” garments using their own images, offering a personalized and engaging shopping experience [1], [2], [3]. Virtual tryon addresses key challenges in the fashion industry, including high return rates due to fit uncertainty, sustainability concerns related to excessive inventory and returns, and the need for enhanced customer engagement. 

Despite the transformative potential of virtual try-on, creating effective systems presents significant technical hurdles. Traditional methods, relying on paired datasets and external warping networks, often lack generalizability to diverse body poses, complex backgrounds, and varying clothing styles [4]. Moreover, preserving intricate garment details and achieving real-time performance remain challenging due to computational overhead. 

Recent progress in pre-trained diffusion models like Stable Diffusion offers promising avenues for high-quality image synthesis [5]. However, adapting these models for virtual try-on requires addressing challenges in semantic alignment between clothing and body features, as well as improving processing speed without sacrificing image quality [1], [6], [7]. 

EfficientVITON tackles these challenges by combining the strengths of diffusion models with innovative architectural enhancements. We introduce an optimized diffusion process using non-uniform timesteps for improved computational efficiency. Furthermore, a spatial encoder and zero cross-attention blocks are incorporated to maintain garment details and achieve accurate alignment [8]. These advancements allow EfficientVITON to handle diverse poses, backgrounds, and clothing styles while preserving textures and intricate features, leading to more realistic and efficient virtual try-on experiences. 

The versatility of virtual try-on extends its applicability to various domains, including e-commerce integration for reduced return rates and improved customer satisfaction, physical retail stores for enhanced shopping experiences, personalized styling and fashion exploration, sustainable fashion practices, and entertainment and media applications. EfficientVITON’s focus on speed and accuracy makes it particularly wellsuited for real-time applications, furthering its potential for widespread adoption. 

The sections of the paper are organized as follows: 

- 1) **Introduction:** Presents the motivation, problem statement, applications, and contributions of the work. 

- 2) **Related Work:** Reviews the state-of-the-art virtual try-on systems and highlights their limitations. 

- 3) **Methodology:** Describes the architecture of EfficientVITON, including the use of Stable Diffusion, zero crossattention blocks, and non-uniform diffusion steps. 

- 4) **Experimental Results:** Presents quantitative and qualitative evaluations, comparing EfficientVITON with existing methods. 

- 5) **Conclusion** Summarizes the contributions and outlines potential directions for future research. 

## II. RELATED WORK 

Virtual try-on systems have become a cornerstone of contemporary e-commerce, allowing users to see how clothing items would look on them without the need for physical trials. 

2 

Early approaches relied on 2D image warping and stitching techniques, which often fell short when handling complex poses, textures, or lighting variations. The emergence of deep learning, particularly generative models, has transformed this field by enabling more realistic and scalable solutions. In recent years, advances in diffusion models have further pushed the limits of image synthesis, delivering exceptional quality and stability in virtual try-on applications [9][3]. 

Also, the evolution of Generative Models in Virtual TryOn Generative models, particularly Generative Adversarial Networks (GANs), have become a cornerstone of virtual tryon systems due to their ability to generate highly realistic images. Despite their popularity, GANs face several challenges, including mode collapse, training instability, and difficulties in preserving fine details in high-resolution outputs. As an alternative, diffusion models have gained traction by employing iterative denoising processes to produce high-quality images with greater stability. Recent advancements have showcased the potential of diffusion models in addressing fashion-specific challenges, including their application in virtual try-on tasks [10][11]. 

Over the past five years, virtual try-on systems have advanced significantly as researchers explore new models and techniques to overcome critical challenges, such as pose alignment, texture preservation, and image realism. Below is an overview of the key approaches, their contributions, and their limitations: 

- 1) **Geometric Warping-Based Models** 

   - Early virtual try-on systems heavily relied on geometric warping techniques to align clothing items with the target body pose. These methods typically employed 2D image transformations, such as Thin Plate Spline (TPS), to warp garments onto the user’s image. For instance, VITON [12] introduced a TPS-based warping approach that enhanced alignment but struggled to preserve fine details and textures. While computationally efficient, these methods often produced unrealistic results when dealing with complex poses, occlusions, or fabric deformations. **Strengths:** Low computational cost and straightforward implementation. 

   - **Weaknesses:** Poor handling of complex poses, textures, and overall realism. 

- 2) **GAN-Based Models** In addition, the advent of GANs revolutionized virtual try-on systems by enabling the realistic synthesis of garments on target poses. These methods generally operate in two stages: first, a warping module aligns the clothing with the body, followed by a refinement module that generates the final output. For example, VITON [12] and CP-VTON [13] are GAN-based frameworks that combined pose estimation with texture preservation to achieve state-of-the-art results. However, GANs often face issues such as training instability, visible artifacts, and challenges in preserving fine details. **Strengths:** High-quality realism and the ability to handle intricate textures. 

   - **Weaknesses:** Unstable training, artifact generation, and difficulty preserving details. 

## 3) **Attention-Based Models** 

To address GANs’ limitations, attention mechanisms were introduced to enhance alignment and texture preservation. These models use attention maps to focus on relevant regions of the clothing and the body, enabling more precise alignment and detailed output. For example, CP-VTON+ [14] developed an attention-based try-on network that excelled at managing complex patterns and textures. However, attention-based models tend to be computationally intensive and require large datasets for training. 

**Strengths:** Better alignment and improved texture detail. **Weaknesses:** High computational cost and heavy data requirements. 

- 4) **Diffusion Models** Recently, diffusion models have emerged as a powerful alternative to GANs in virtual tryon applications. Unlike GANs, these models iteratively denoise images, which enhances stability and output quality. They have proven highly effective at managing complex textur [9] and Stable-VITON [3] are diffusionbased models that demonstrate exceptional image quality and stability. However, their iterative nature makes them computationally expensive, which poses challenges for real-time applications. 

   - **Strengths:** Exceptional image quality, stability, and detail preservation. 

   - **Weaknesses:** High computational demands and slow inference speeds. 

- 5) **Hybrid Models** 

   - Hybrid models combine the strengths of multiple approaches, such as geometric warping, GANs, and attention mechanisms, to achieve better alignment, realism, and efficiency. For instance, HR-VITON [4] proposed a hybrid framework that integrates geometric warping for initial alignment with GAN-based refinement for final synthesis. While effective, hybrid models often require complex architectures and significant tuning. 

   - **Strengths:** Balanced performance by leveraging multiple techniques. 

   - **Weaknesses:** Complex architecture and intensive tuning requirements. 

- 6) **Multi-Modal Models** 

   - Advancements in multi-modal inputs, such as text descriptions and sketches, have enhanced user interaction and personalization in virtual try-on systems. For example, Text2Cloth [15] developed a system allowing users to describe clothing in natural language, which the model then synthesizes onto their image. Similarly, Sketch2TryOn [16] introduced a sketch-based interface for designing and visualizing custom clothing in realtime. While engaging, these methods require additional preprocessing and are computationally expensive. 

   - **Strengths:** Greater personalization and user engagement. **Weaknesses:** High computational cost and preprocessing demands. 

Diffusion-based methods currently lead the field, generating high-resolution, photorealistic images that preserve intricate clothing details. Recent advancements include integrating pose 

3 

estimation, attention mechanisms for alignment, and multimodal inputs for personalization. However, challenges such as adapting to diverse body shapes, managing occlusions, and improving real-time performance remain unresolved. 

Despite significant progress, several gaps persist in virtual try-on research. Current methods often focus solely on static images, underutilizing temporal data for video-based try-on systems [17]. Personalization, such as accommodating user preferences or diverse body types, remains limited. The computational cost of diffusion models also restricts their use in real-time applications. Furthermore, there is a lack of standardized evaluation metrics tailored to virtual try-on systems, complicating objective comparisons between methods [14]. 

While existing systems have improved realism and usability, many struggle to generalize across diverse datasets and realworld scenarios. For instance, GAN-based methods often generate artifacts in complex scenarios, while diffusion models, despite their stability, remain computationally expensive [9][3]. Additionally, most systems rely on paired datasets, limiting scalability. 

The research proposes a diffusion model framework to improve virtual try-on experiences, utilizing non-uniform time steps, temporal coherence, and personalization to reduce computational demands and enhance image quality. 

## III. METHODOLOGY 

This section details the EfficientVITON framework for virtual try-on, leveraging a pre-trained diffusion model and incorporating optimizations for efficiency and enhanced clothing detail preservation. 

## _A. Data & Preprocessing_ 

EfficientVITON utilizes a preprocessed dataset derived from VITON-HD [1], applying the following steps (Fig. 9): 

- **1) Pose Estimation:** OpenPose (Fig. 1) extracts 25 keypoints to define body regions [18], [19], [20], [21]. 

- **2) Human Parsing:** LIP Parsing [22] segments the person image into 20 semantic parts (Fig. 2). 

- **3) Agnostic Image:** A grey mask, based on pose and parsing, covers the original clothing area (Fig. 3). 

- **4) Agnostic Mask:** A binary mask isolates the original clothing region (Fig. 4). 

- **5) Parse Agnostic:** The agnostic mask area is removed from the parsed semantic map (Fig. 5). 

- **6) Ground Truth Warp Mask:** A mask of the worn garment is extracted for training (Fig. 6). 

- **7) Dense Human Pose Estimation:** DensePose [17] generates a UV map (Fig. 7), subsequently encoded into a latent representation, providing detailed 3D body surface information. 

- **8) Cloth Mask:** A binary mask of the garment image is extracted (Fig. 8). 

**==> picture [127 x 126] intentionally omitted <==**

Fig. 1: OpenPose Output. 

**==> picture [127 x 126] intentionally omitted <==**

Fig. 2: LIP Parsing Output. 

## _B. Stable Diffusion Architecture_ 

EfficientVITON builds upon the Stable Diffusion model [5], leveraging its high-fidelity generation, latent space efficiency, and pre-trained knowledge of human and clothing features. Stable Diffusion consists of a Variational Autoencoder (VAE) for latent space compression and reconstruction, a U-Net for denoising, and a diffusion process operating in the latent space. 

While the quality of the output of Stable Diffusion is relatively high, it is not efficient enough. It uses a lot of memory resources and needs a lot of time to complete the diffusion process. Jiang et al. addressed the problem of an efficient diffusion process in their research [23], in which we took a similar approach to their solution to this problem. The main solution to this problem is to modify the timesteps required for the model to do the denoising process. Instead of a large number of uniformly distributed timesteps, we apply a non-uniform distribution of a small number of timesteps in the denoising process. 

Instead of sampling _**n**_ steps uniformly from all possible timesteps, we sample from a smaller set of strategically chosen timesteps. This significantly reduces the time needed for the diffusion process without sacrificing the quality of the output image, as the model will now learn the most significant timesteps in the denoising process and try to take more impulsive steps towards the final goal. The non-uniform distribution allows the model to take different amounts of denoising steps according to the position of the timestep. Fig. 12 shows a description of the non-uniform denoising steps which allows the model to be more efficient. 

4 

**==> picture [127 x 126] intentionally omitted <==**

Fig. 3: Agnostic Image Output. 

**==> picture [127 x 95] intentionally omitted <==**

Fig. 4: Agnostic Mask Output. 

## _C. EfficientVITON Architecture_ 

EfficientVITON Architecture (Fig. 10) integrates a spatial encoder and zero cross-attention blocks with the Stable Diffusion core to perform virtual try-on. 

**Inputs:** The model receives the preprocessed clothing image ( _xc_ ), agnostic map ( _xa_ ), agnostic mask ( _xma_ ), and latent dense pose ( _xp_ ). 

**Spatial Encoder:** This encoder, initialized with U-Net weights, processes the clothing image ( _xc_ ) and extracts multiresolution feature maps, capturing fine-grained clothing details. These features are then used as key (K) and value (V) inputs to the zero cross-attention blocks. 

**Zero Cross-Attention Blocks** (Fig. 11): Integrated within the U-Net decoder, these blocks learn the semantic correspondence between the clothing and the human body. The blocks use the spatial encoder’s clothing features (K, V) and U-Net decoder features (Q) to perform patch-wise warping in the latent space. A linear layer with zero-initialized weights helps to reduce noise. 

## _D. Efficient Diffusion Process_ 

EfficientVITON optimizes the diffusion process by employing non-uniform timestep sampling [23] (Fig. 12). This concentrates denoising steps at critical timesteps, reducing computation while maintaining output quality. 

_1) Loss Functions:_ EfficientVITON combines the Stable Diffusion loss ( _LLDM_ ) with an Attention Total Variation Loss ( _LAT V_ ) to refine attention maps: 

**==> picture [225 x 12] intentionally omitted <==**

**==> picture [223 x 11] intentionally omitted <==**

**==> picture [222 x 10] intentionally omitted <==**

**==> picture [127 x 126] intentionally omitted <==**

Fig. 5: Parse Agnostic Image Output. 

**==> picture [127 x 127] intentionally omitted <==**

Fig. 6: Ground Truth Warp Mask Output. 

_2) Training and Inference:_ Training is a two-stage process: (1) learning semantic correspondence with augmented inputs, and (2) refining attention maps with _LAT V_ . Inference uses the PLMS sampler and, for paired evaluations, the RePaint [24] approach. 

EfficientVITON’s key contributions are: (1) end-to-end virtual try-on with a pre-trained diffusion model, (2) latent space semantic correspondence learning via zero cross-attention, (3) attention total variation loss and augmentation, and (4) an efficient diffusion process through non-uniform timestep sampling. 

## IV. RESULTS 

This section presents the qualitative, quantitative, and efficiency evaluation of EfficientVITON, demonstrating its superior performance in virtual try-on. 

## _A. Qualitative Results_ 

EfficientVITON generates visually realistic and accurate virtual try-on images (Fig. 13), effectively handling complex clothing patterns, text, logos, and varying body types and poses. The model preserves intricate clothing details, maintains natural folds and shadows, and adapts seamlessly to different garment coverage levels, producing results visually indistinguishable from real photographs. 

## _B. Quantitative Results_ 

Quantitative evaluation using FID [25] and LPIPS [26] (Table I) demonstrates EfficientVITON’s superior performance. 

5 

**==> picture [127 x 126] intentionally omitted <==**

Fig. 7: DensePose Output. 

**==> picture [127 x 95] intentionally omitted <==**

Fig. 8: Cloth Mask Output. 

The model achieves a lower FID score than state-of-theart methods, indicating higher realism. While LPIPS scores are competitive, the RePaint [24] refinement further improves performance in paired settings by addressing potential reconstruction errors related to the agnostic map. EfficientVITON’s consistent performance across different experimental setups highlights its robustness and generalizability. 

|Method|LPIPS|FID|
|---|---|---|
|VITON-HD [1]|0.117|12.117|
|HR-VITON [4]|0.1045|11.265|
|LADI-VTON [11]<br>Paint-by-Example [27]<br>DCI-VTON [10]|0.0964<br>0.1428<br>0.0804|9.480<br>11.939<br>8.754|
|GP-VTON [28]|0.088|9.072|
|Ours|0.0842|8.703|
|Ours (RePaint)|0.0762|8.433|



TABLE I: Quantitative Comparison (Lower is Better). 

## _C. Efficiency Results_ 

The non-uniform timestep sampling strategy significantly improves efficiency (Table II), reducing both training and inference times. The inference time is reduced by 72.4%, enabling near real-time performance, while training time is reduced by 45.3%, facilitating faster model development and experimentation. These improvements enhance practicality for real-world applications and resource-constrained environments. 

## V. CONCLUSION 

This work introduces EfficientVITON, a novel virtual tryon framework that addresses the challenges of realism and efficiency. By incorporating non-uniform timestep sampling 

**==> picture [127 x 214] intentionally omitted <==**

Fig. 9: A sample from VITON-HD dataset. a) Normal Person Image. b) Human Parsing. c) OpenPose Pose Estimation. d) Agnostic Image. e) Parse Agnostic Image. f) DensePose Image. g) Unworn Cloth Image. h) Cloth Mask. 

||Method|Training time|Inference Time|
|---|---|---|---|
||Before|1570 h|58 s|
||After|859 h|16 s|



TABLE II: Comparison of Training time and Inference time for our model before and After using our new approach of applying the nonuniform timestep distribution. 

[23] into a pre-trained diffusion model, EfficientVITON significantly reduces computational overhead (45.3% reduction in training time and 72.4% in inference time) without sacrificing visual fidelity. The inclusion of a spatial encoder and zero cross-attention blocks [3] further enhances realism by preserving fine clothing details and ensuring accurate alignment with the target body. 

Quantitative results using FID [25] and LPIPS [26] demonstrate EfficientVITON’s superior performance compared to state-of-the-art GAN-based and diffusion-based virtual try-on methods. Qualitative results further confirm the model’s ability to generate realistic and visually appealing try-on images across diverse poses, skin tones, and clothing styles. 

EfficientVITON offers several key contributions: (1) an end-to-end virtual try-on framework based on a pre-trained diffusion model, eliminating the need for separate warping modules; (2) learning semantic correspondence directly in the latent space; (3) attention total variation loss and data augmentation for improved attention maps; and (4) enhanced efficiency through non-uniform timestep sampling. 

This work has broader implications beyond virtual try-on. The optimized diffusion process using non-uniform timesteps can be applied to other image synthesis tasks. Moreover, EfficientVITON’s efficiency makes it suitable for real-world applications in e-commerce, offering a personalized and engaging online shopping experience, potentially reducing return rates and promoting sustainability. EfficientVITON represents a sig- 

6 

**==> picture [412 x 248] intentionally omitted <==**

Fig. 10: EfficientVITON Architecture. 

**==> picture [127 x 233] intentionally omitted <==**

Fig. 11: Zero Cross-Attention Block. 

nificant advancement in virtual try-on technology, bridging the gap between high-quality image synthesis and computational efficiency, and opening up new avenues for future research and development in generative modeling. 

## REFERENCES 

- [1] S. Choi, S. Park, M. Lee, and J. Choo, “Viton-hd: High-resolution virtual try-on via misalignment-aware normalization,” in _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , 2021, pp. 14 131–14 140. 

**==> picture [253 x 154] intentionally omitted <==**

Fig. 12: Efficient Diffusion vs. Standard Diffusion. 

- [2] C. Schuhmann, R. Beaumont, R. Vencu, C. Gordon, R. Wightman, M. Cherti, T. Coombes, A. Katta, C. Mullis, M. Wortsman, P. Schramowski, S. Kundurthy, K. Crowson, L. Schmidt, R. Kaczmarczyk, and J. Jitsev, “Laion-5b: an open large-scale dataset for training next generation image-text models,” in _Proceedings of the 36th International Conference on Neural Information Processing Systems_ , ser. NIPS ’22. Red Hook, NY, USA: Curran Associates Inc., 2024. 

- [3] J. Kim, G. Gu, M. Park, S. Park, and J. Choo, “Stableviton: Learning semantic correspondence with latent diffusion model for virtual try-on,” in _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , 2024, pp. 8176–8185. 

- [4] Q. Lyu, Q. Wang, and K. Huang, “High-resolution virtual try-on network with coarse-to-fine strategy,” _Journal of Physics: Conference Series_ , vol. 1880, p. 012009, 04 2021. 

- [5] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “High-resolution image synthesis with latent diffusion models,” _2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , pp. 10 674–10 685, 2021. [Online]. Available: https://api.semanticscholar.org/CorpusID:245335280 

- [6] Y. Ge, Y. Song, R. Zhang, C. Ge, W. Liu, and P. Luo, “Parser-free virtual try-on via distilling appearance flows,” in _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2021, pp. 8485–8493. 

7 

**==> picture [438 x 292] intentionally omitted <==**

Fig. 13: Qualitative Results on VITON-HD. 

- [7] S. Lee, G. Gu, S. Park, S. Choi, and J. Choo, “High-resolution virtual tryon with misalignment and occlusion-handled conditions,” in _Proceedings of the European Conference on Computer Vision (ECCV)_ . Springer, 2022, pp. 204–219. 

- [8] L. Zhang, A. Rao, and M. Agrawala, “Adding conditional control to text-to-image diffusion models,” in _Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)_ , 2023, pp. 3836– 3847. 

- [9] L. Zhu, D. Yang, T. Zhu, F. Reda, W. Chan, C. Saharia, M. Norouzi, and I. Kemelmacher-Shlizerman, “Tryondiffusion: A tale of two unets,” in _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , June 2023, pp. 4606–4615. 

- [10] J. Gou, S. Sun, J. Zhang, J. Si, C. Qian, and L. Zhang, “Taming the power of diffusion models for high-quality virtual try-on with appearance flow,” _arXiv preprint arXiv:2308.06101_ , 2023. 

- [11] D. Morelli, A. Baldrati, G. Cartella, M. Cornia, M. Bertini, and R. Cucchiara, “Ladi-vton: Latent diffusion textual-inversion enhanced virtual try-on,” _arXiv preprint arXiv:2305.13501_ , 2023. 

- [12] X. Han, Z. Wu, Z. Wu _et al._ , “Viton: An image-based virtual try-on network,” in _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2018, pp. 7543–7552. 

- [13] B. Wang, H. Zheng, X. Liang _et al._ , “Toward characteristic-preserving image-based virtual try-on network,” in _Proceedings of the European Conference on Computer Vision (ECCV)_ , 2018, pp. 589–604. 

- [14] T. Issenhuth, J. Mary, and C. Calauzenes, “Do not mask what you do not need to mask: A parser-free virtual try-on,” in _Proceedings of the European Conference on Computer Vision (ECCV)_ , 2020, pp. 619–635. 

- [15] B. Ren, H. Tang, F. Meng _et al._ , “Cloth interactive transformer for virtual try-on,” _arXiv preprint arXiv:2104.05519_ , 2021. 

- [16] Z. Xie, Z. Huang, F. Zhao _et al._ , “Towards scalable unpaired virtual try-on via patch-routed spatially-adaptive gan,” in _Advances in Neural Information Processing Systems (NeurIPS)_ , 2021, pp. 2598–2610. 

- [17] R. A. G¨uler, N. Neverova, and I. Kokkinos, “Densepose: Dense human pose estimation in the wild,” in _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , 2018, pp. 7297–7306. 

- [18] Z. Cao, G. Hidalgo Martinez, T. Simon, S. Wei, and Y. A. Sheikh, “Openpose: Realtime multi-person 2d pose estimation using part affinity fields,” _IEEE Transactions on Pattern Analysis and Machine Intelligence_ , 2019. 

- [19] T. Simon, H. Joo, I. Matthews, and Y. Sheikh, “Hand keypoint detection in single images using multiview bootstrapping,” in _CVPR_ , 2017. 

- [20] Z. Cao, T. Simon, S.-E. Wei, and Y. Sheikh, “Realtime multi-person 2d pose estimation using part affinity fields,” in _CVPR_ , 2017. 

- [21] S.-E. Wei, V. Ramakrishna, T. Kanade, and Y. Sheikh, “Convolutional pose machines,” in _CVPR_ , 2016. 

- [22] K. Gong, X. Liang, D. Zhang, X. Shen, and L. Lin, “Look into person: Self-supervised structure-sensitive learning and a new benchmark for human parsing,” in _2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2017, pp. 6757–6765. 

- [23] H. Jiang, M. Imran, L. Ma, T. Zhang, Y. Zhou, M. Liang, K. Gong, and W. Shao, “Fast-ddpm: Fast denoising diffusion probabilistic models for medical image-to-image generation,” _arXiv preprint arXiv:2405.14802_ , 2024. 

- [24] A. Lugmayr, M. Danelljan, A. Romero, F. Yu, R. Timofte, and L. V. Gool, “Repaint: Inpainting using denoising diffusion probabilistic models,” _2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , pp. 11 451–11 461, 2022. [Online]. Available: https://api.semanticscholar.org/CorpusID:246240274 

- [25] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, G. Klambauer, and S. Hochreiter, “Gans trained by a two time-scale update rule converge to a nash equilibrium,” _ArXiv_ , vol. abs/1706.08500, 2017. [Online]. Available: https://api.semanticscholar.org/CorpusID:231697514 

- [26] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, “The unreasonable effectiveness of deep features as a perceptual metric,” _2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 586–595, 2018. [Online]. Available: https: //api.semanticscholar.org/CorpusID:4766599 

- [27] B. Yang, S. Gu, B. Zhang, T. Zhang, X. Chen, X. Sun, D. Chen, and F. Wen, “Paint by example: Exemplar-based image editing with diffusion models,” _2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , pp. 18 381–18 391, 2022. [Online]. Available: https://api.semanticscholar.org/CorpusID:253802085 

- [28] Z. Xie, Z. Huang, X. Dong, F. Zhao, H. Dong, X. Zhang, F. Zhu, and X. Liang, “Gp-vton: Towards general purpose virtual try-on via collaborative local-flow global-parsing learning,” _2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , pp. 23 550–23 559, 2023. [Online]. Available: https://api.semanticscholar. org/CorpusID:257757040 

