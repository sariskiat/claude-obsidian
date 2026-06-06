---
type: paper-fulltext
slug: jco-mvton-jointly-controllable-multi-modal-diffusion-transformer-for-mask-free-virtual-try-on
arxiv_id: null
source_path: /Users/saris.kia.adm/.paper-scholar/jco-mvton-jointly-controllable-multi-modal-diffusion-transformer-for-mask-free-virtual-try-on/2508.17614.md
paper: "[[jco-mvton-jointly-controllable-multi-modal-diffusion-transformer-for-mask-free-virtual-try-on]]"
---
<!-- extracted by afk_extract from 2508.17614.pdf (20p) -->

**==> picture [103 x 24] intentionally omitted <==**

## **JCo-MVTON: Jointly Controllable Multi-Modal Diffusion Transformer for Mask-Free Virtual Try-on** 

**Aowen Wang** _[∗]_[1] _[,]_[3] , **Wei Li** _[∗]_[1] , **Hao Luo** _[†]_[1] _[,]_[2] , **Mengxing Ao**[1] , **Chenyu Zhu**[3] , **Xinyang Li**[1] _[,]_[3] , **Fan Wang**[1] 

> 1DAMO Academy, Alibaba Group 

> 2Hupan Lab 

> 3Zhejiang University 

> _∗_ Equal contribution, _†_ Corresponding author 

Virtual try-on systems have long struggled with rigid dependencies on human body masks, limited fine-grained control over garment attributes, and poor generalization to in-the-wild scenarios. In this paper, we propose **JCo-MVTON** ( **J** ointly **C** ontrollable Multi-Modal Diffusion Transformer for **M** askFree **V** irtual **T** ry- **O** n), a novel framework that simultaneously addresses these challenges by unifying diffusion-based generation with multi-modal condition fusion. Our architecture leverages a Multi-Modal Diffusion Transformer (MM-DiT) backbone to integrate diverse control signals—including reference image and garment image—directly into the denoising process. Our key architectural innovation involves conditioning the MM-DiT by fusing reference and garment information into its self-attention layers through dedicated conditional pathways. This integration is augmented by critical refinements to the positional encodings and attention masks, which specialize the multi-conditional MM-DiT for the nuanced requirements of the virtual try-on task and yield superior results. On the data curation front, we introduce a bi-directional generation strategy to construct our training set. This approach leverages two complementary pathways: first, a mask-based model is employed to generate a substantial volume of reference images; second, a Try-Off model, sharing an identical architecture and trained via self-supervision, synthesizes the corresponding garment data. Furthermore, the entire generated corpus undergoes a meticulous manual screening process, allowing us to iteratively improve the quality and fidelity of the training data. Our method achieves state-of-the-art performance on public benchmarks (e.g., DressCode) and outperforms commercial systems in real-world scenarios. The online API demo is available at Alibaba Cloud Marketplace. 

**Project page and code** : https://github.com/damo-cv/JCo-MVTON 

**Date:** August 26, 2025 

## **1 Introduction** 

Virtual try-on (VTON) systems aim to synthesize a realistic image of a person wearing a new garment while preserving the person’s original pose, body shape, and appearance as faithfully as possible. This technology has become increasingly important for online fashion and retail: by enabling users to “virtually try on” clothing and accessories prior to purchasing them, VTON addresses common uncertainties about fit, proportion, and style compatibility, thereby enhancing customer satisfaction, increasing user engagement, and reducing return rates in e-commerce (Han et al., 2018a; Islam and Patel, 2023). As online shopping continues to grow, the lack of physical interaction with garments remains a key limitation; VTON bridges this gap by providing an immersive and personalized shopping experience that mimics in-store fitting rooms. 

Early VTON methods generally followed a multi-stage pipeline based on generative adversarial networks (GANs) and warping modules. These approaches warp the garment onto the target body (e.g., via thin-plate splines or learned flow fields) and then use a GAN-based inpainting network to blend it with the person (Han et al., 2018a; Chen et al., 2024a). However, GAN-based models often struggle to produce high-quality results 

**==> picture [60 x 12] intentionally omitted <==**

**----- Start of picture text -----**<br>
!" #$% &!’(<br>**----- End of picture text -----**<br>


**==> picture [423 x 300] intentionally omitted <==**

**----- Start of picture text -----**<br>
.+/#" ’()"& $% !"#$%%& !%*+,"- !"#$%%& $% ’()"&<br>+,,%*<br>’)&%*<br>(*%--<br>**----- End of picture text -----**<br>


**Figure 1** Our JCo-MVTON model for virtual try-on, a Multi-Modal Diffusion Transformer (MM-DiT) framework featuring an efficient data construction method and a specialized attention mechanism to achieve state-of-the-art results. 

under challenging poses and complex backgrounds. Recent surveys report that diffusion models have begun to surpass GANs in both fidelity and diversity for image synthesis tasks, including VTON (Lee and Kim, 2023; Chen et al., 2024a). 

Broadly, diffusion-based VTON methods fall into two camps: mask-based and mask-free approaches (Liu et al., 2024; Wang et al., 2024b). In mask-based pipelines, one first generates a binary human-parsing mask 

to occlude the subject’s original apparel region, then warps the target garment to align with that mask, and finally applies a diffusion (or GAN) model to inpaint the occluded area (Han et al., 2018a; Wang et al., 2024b). Although this formulation can yield finely detailed alignments, it entails two key drawbacks. First, the multi-stage nature of segmentation, warping, and inpainting makes the pipeline brittle: errors in mask prediction or garment alignment tend to cascade and compromise the final output. Second, reliance on hard masks limits flexibility, since occluding regions may discard critical context (e.g., hands, background elements) and demands highly accurate human parsing—a requirement that is often difficult to satisfy in real-world settings (Wang et al., 2024b). 

The inpainting-based paradigm benefits from a well-defined, modular workflow—human parsing yields a precise clothing mask, which guides texture-to-body correspondence and simplifies subsequent inpainting stages (Han et al., 2018b; Wan et al., 2025). Nevertheless, it is prone to cascading failures: parsing errors or complex occlusions can produce visible artifacts (e.g., misplaced hair or arms, clothing “leakage”), while masking disrupts the original image’s spatial and lighting cues, degrading garment detail and background coherence (Jiang et al., 2024; Atef et al., 2025; Choi et al., 2024). By contrast, mask-free methods dispense with explicit segmentation, thereby avoiding mask-induced artifacts and preserving the scene’s innate geometry and illumination. During inference, only the person and garment images are required—no separate parsing, warping, or fusion steps—resulting in a more streamlined and robust pipeline (Zhang et al., 2024; Niu et al., 2024; Chang et al., 2025). Consequently, mask-free diffusion models have emerged as the dominant paradigm for high-fidelity virtual try-on. 

However, the mask-free paradigm(Guo et al., 2025; Yang et al., 2024) also suffers from several drawbacks. First, mask-free methods entirely forgo the use of an explicit mask input for the target region, relying solely on the garment and model images; as a result, the network must infer all aspects of the clothing rendering automatically, which prevents precise, localized refinement along garment boundaries or in fine-detail regions. A second challenge lies in the construction of triplet datasets(Sun et al., 2025): training in a mask-free manner requires simultaneous access to three aligned images: the original model, the isolated garment, and the post-try-on result. Compared to traditional mask-based approaches, this greatly complicates dataset assembly. Indeed, mainstream benchmarks (e.g., VITON(Han et al., 2018a), DressCode(Morelli et al., 2022)) are generally released only after mask-based preprocessing and provide merely segmentation masks, without furnishing complete, pixel-level ground truth for all three elements. 

To address the limitations inherent in mask-free approaches, we propose the Jointly Controllable Multi-Modal Diffusion Transformer for Mask-Free Virtual Try-On (JCo-MVTON). First, we introduce a cyclic datapreparation pipeline to overcome the scarcity of mask-free triplet datasets: by leveraging Try-Off (Velioglu et al., 2024), mask-based Try-on, and IC-LoRA(Huang et al., 2024) techniques, we generate a large pool of candidate triplets, which are then manually filtered to yield high-quality training samples. Through successive iterations of model refinement and data augmentation, this pipeline becomes self-sustaining. Moreover, our model employs a multi-condition MM-DiT mechanism (Esser et al., 2024) to achieve more stable try-on results; compared to the previous state-of-the-art ReferenceNet(Chen et al., 2024b), MM-DiT significantly raises both the upper bound of generation quality and the consistency with the original garment. Finally, we further investigate the integration of positional encoding and a masked-attention mechanism to mitigate inter-branch interference, thereby enhancing the fidelity and alignment of generated outputs with the source model images. 

Overall, our method achieves significant success on the virtual try-on task. We establish a sustainable, novel paradigm for generating mask-free training data that co-evolves with our model through continuous self-iteration and optimization, thereby cyclically improving both model performance and data quality. Furthermore, our model itself not only attains state-of-the-art results on benchmarks such as VITON-HD, but also demonstrates exceptional performance across diverse scenarios: varying garment types (tops and bottoms), styles (anime, real-world apparel, avant-garde costumes), and application contexts (B2B and B2C). This versatility evidences its high stability and image fidelity. In our human evaluation against existing commercial solutions, our approach markedly outperforms competitors on all assessed metrics—including Image Authenticity, Image Clarity, Silhouette Consistency, Detail Consistency, and Overall Harmony—underscoring its superiority in practical deployment. 

## **2 Human Evaluation** 

In order to evaluate the performance of different models in virtual try-on tasks, we designed a human evaluation framework based on five key metrics: Image Authenticity , Image Clarity , Silhouette Consistency , Detail Consistency , and Overall Harmony . The definitions of these metrics are as follows: 

**==> picture [424 x 312] intentionally omitted <==**

**----- Start of picture text -----**<br>
!"#$%&80+.%1+*3*+, !"#$%&80+.%1+*3*+,<br>!"#$%& 56%)#((& !"#$%& 56%)#((&<br>’(#)*+, 7#)"/1, ’(#)*+, 7#)"/1,<br>-*(./0%++%& 4%+#*(& -*(./0%++%& 4%+#*(&<br>’/12*2+%13, ’/12*2+%13, ’/12*2+%13, ’/12*2+%13,<br>!"#$%&80+.%1+*3*+,<br>!"#$%&80+.%1+*3*+,<br>!"#$%& 56%)#((&<br>!"#$%& 56%)#((& ’(#)*+, 7#)"/1,<br>’(#)*+, 7#)"/1,<br>-*(./0%++%& 4%+#*(&<br>-*(./0%++%& 4%+#*(& ’/12*2+%13, ’/12*2+%13,<br>’/12*2+%13, ’/12*2+%13,<br>**----- End of picture text -----**<br>


**Figure 2** Radar-chart comparison of three models (Kling, OutfitAnyone, and Ours JCo-MVTON) across four virtual try-on scenarios: (a) Overall: JCo-MVTON achieves the highest scores on all five metrics; (b) Upper: JCo-MVTON leads on every metric; (c) Lower: JCo-MVTON tops all metrics; (d) Dress: (c) Lower: JCo-MVTON tops all metrics; attains the best performance on all metrics except a slight lag behind Kling in Detail Consistency. 

- **Image Authenticity** : Assesses how visually similar the generated image is to a real image, considering color, lighting, and background realism. 

- **Image Clarity** : Evaluates the clarity of the generated image, checking for blurriness, noise, and other unwanted visual artifacts. 

- **Silhouette Consistency** : Measures the consistency of the garment’s silhouette with the body shape, ensuring that the virtual clothing fits naturally. 

- **Detail Consistency** : Assesses the accuracy of clothing details, such as fabric texture, wrinkles, and patterns, and whether they appear natural. 

- **Overall Harmony** : Evaluates the overall visual harmony, considering how well the clothing and body integrate, and whether the outfit looks aesthetically pleasing. 

We compared three models: **Kling** , **OutfitAnyone** , and **Ours JCo-MVTON** . These models represent different virtual try-on implementations, and we performed a human assessment for each model across various try-on scenarios. The specific evaluation scenarios are: 

- **Overall** : A holistic evaluation considering all metrics combined. 

- **Upper** : Focused on the performance of upper-body clothing, such as shirts, blouses, and jackets. 

- **Lower** : Focused on lower-body clothing, such as pants, skirts, and shorts. 

- **Dress** : Focused on full-body clothing, especially dresses. 

To facilitate the analysis and comparison of the results, we present radar charts for each scenario. The radar charts visually depict the scores of each model across the five evaluation metrics, providing a clear comparison of their performances in different scenarios. 

!"#$%&’ ()*%+ !"#$%#&’()’* +,%’./)0123!4 !"#$%&’ ()*%+ !"#$%#&’()’* +,%’./)0123!4 

**Figure 3** The manual evaluation results were derived from a sampled set of outcomes across upper-body, lower-body, and dress (full-body) tasks. The results, presented from left to right, reflect the performance of OutfitAnyone, Kling, and JCo-MVTON, respectively. According to our evaluation metrics, JCo-MVTON demonstrates consistently superior performance. 

## **2.1 Radar Chart Analysis** 

- **Overall** : Our model ( JCo-MVTON ) outperforms Kling and OutfitAnyone, particularly in Image Clarity and Image Authenticity . 

- **Upper** : In the upper-body clothing scenario, JCo-MVTON slightly outperforms Kling in Detail Consistency, but achieving much higher scores in other metrics. 

- **Lower** : In the lower-body clothing scenario, JCo-MVTON excels again, much better on all five metric . 

- **Dress** : For full-body clothing, JCo-MVTON shows the strongest Overall Harmony although it falls slightly short of Kling in Detail Consistency . 

## **2.2 Discussion and Conclusion** 

Based on the radar chart analysis, our model ( Ours ) consistently outperforms the other models across all scenarios, particularly in terms of Image Authenticity , Detail Consistency , and Overall Harmony . This indicates that Ours is better at handling the fusion of clothing and the human body, producing more natural and harmonious virtual try-on results. 

Although Kling and OutfitAnyone exhibit some advantages in specific scenarios, their performance in detail preservation and overall harmony remains limited. Therefore, based on these evaluation results, we conclude that Ours demonstrates superior practicality and reliability in virtual try-on tasks. 

## **3 Related Works** 

## **3.1 Image-based Conditional Generation** 

Recent advances in image-conditioned generation have significantly improved both the fidelity and controllability of diffusion-based models.(Zhang et al., 2025)proposed EasyControl, introducing a lightweight plug-and-play adapter for Diffusion Transformers (DiT)(Peebles and Xie, 2023) that injects image conditions via a parallel Low-Rank Adaptation (LoRA)(Hu et al., 2021) branch. By processing conditional inputs in isolation and employing separable convolutions with multi-scale feature aggregation, EasyControl achieves high-resolution outputs with flexible spatial and subject control, supporting zero-shot combination of controls at inference time. Shortly thereafter, Ominicontrol and Ominicontrol 2(Xie et al., 2024; Tan et al., 2025), which reuses the DiT’s own VAE encoder and multi-modal attention to embed image conditions with negligible extra parameters. OmniControl constructs a unified token sequence containing noisy image latents, text, and condition tokens, enabling the pretrained DiT to directly “wear” the control and handle both spatially-aligned inputs (e.g., edges, depth maps) and non-aligned subject images within a single model. Building on these, (Wang et al., 2025a)introduced UniCombine, a DiT-based architecture for multi-conditional generation. UniCombine incorporates Condition-MMDiT Attention and multiple LoRA modules to jointly fuse text prompts, spatial maps, and reference images, offering both training-free and fine-tuned variants, and demonstrating state-of-the-art alignment across all input constraints on the new SubjectSpatial200K dataset. 

## **3.2 Mask-based and Mask-free Virtual Try-On** 

In the past two years, image-based virtual try-on methods have bifurcated into mask-based and mask-free paradigms, each offering distinct advantages. StableGarment(Wang et al., 2024a) pioneered the integration of a stability regularization term into the try-on pipeline, employing a two-stage framework that first predicts a coarse garment warped to the target shape and then refines details via a stability-aware generator to reduce artifacts along garment edges. Building on multi-view consistency, MV-VTON(Wang et al., 2025b) incorporates multiple camera perspectives at training time: by enforcing cross-view feature alignment and view-consistent synthesis, it produces more accurate 3D-aware garment draping, particularly under challenging poses. LaDI-VTON(Morelli et al., 2023) introduces a layered deformation initiative, decomposing garment transfer into a local alignment stage—where a dense correspondence field aligns fine-grained garment parts—and a global deformation network that adjusts the warped clothing to the body silhouette; this layered design significantly improves local detail preservation around collars and cuffs. DCI-VTON(Gou et al., 2023) further enhances local detail by proposing a Dual-Channel Integration module that fuses semantic segmentation and dense flow channels, enabling the network to exploit both structural layout and pixel-level correspondence cues; experiments demonstrate marked gains in rendering fidelity for complex patterns and textures. CatVTON(Chong et al., 2025) proposes a category-aware transfer mechanism, in which garments are first classified into semantic categories (e.g., tops, bottoms, outerwear) and then passed through categoryspecific deformation subnets, resulting in improved adaptability across diverse garment types. More recently, OOTDiffusion(Xu et al., 2024) applies diffusion-based generative modeling to the try-on task, formulating garment transfer as a conditional denoising process that iteratively refines a noisy composite of the person and target clothing; this approach excels at generating high-frequency textures and natural shading transitions but can incur greater computational cost due to the multi-step diffusion schedule. BooW-VTON(Zhang et al., 2024) is a mask-free virtual try-on method that uses augmented data instead of segmentation masks. It combines a U-Net with pre-trained models to encode clothing features and uses cross-attention to apply them. A special loss helps focus on the clothing area, reducing errors. This improves alignment and texture, even with difficult poses and lighting. MF-VITON(Wan et al., 2025) removes the need for user-provided masks with a two-stage, mask-free approach. First, it uses a mask-based model to create a large, diverse dataset with different backgrounds. Then, it fine-tunes the model to work without masks. This allows garment transfer using just one person image and a target garment, while keeping texture and shape details and achieving top performance. Nevertheless, while current approaches remain limited by either architectural complexity or insufficient control over virtual try-on synthesis, they collectively inform pathways toward more efficient and streamlined implementations. 

## **4 Jointly Conditional MM-DiT Model for Mask-Free Vitual Try-on** 

In this section, we provide an overview of the Jointly Controllable MMDiT, which achieves high-quality, stable, mask-free virtual try-on by jointly injecting multiple image features (e.g., reference image and garment image) into the self-attention mechanism. 

## **4.1 Preliminary** 

**Multimodal Diffusion Transformers** Multimodal Diffusion Transformers (MM-DiT)(Esser et al., 2024) is a text-driven image generation architecture, commonly used in text-to-image models such as FLUX(Labs, 2024) and SD3(Esser et al., 2024). It can effectively integrate textual and visual features into the attention architecture, making it especially well-suited for multi-image–controlled virtual try-on (VTON) tasks. The model leverages diffusion processes to generate visual content while employing transformer-based structures for cross-modal information fusion, enabling effective alignment between clothing and the human body, thereby enhancing the quality and stability of virtual try-on results. In MM-DiT, the diffusion model generates the target image based on a conditional diffusion process. Let x 0 represent the initial state of the generated image (i.e., starting from noise), and the true image is gradually restored through the diffusion process. The training objective of the model is to minimize the following loss function: 

**==> picture [321 x 14] intentionally omitted <==**

where x _t_ represents the image representation at step t in the diffusion process, f _θ_ is the transformer-based generative network, and c is the conditional information (such as body features, clothing features, etc.). Through this process, the model can generate high-quality virtual try-on images based on different conditional information. 

**Rectified Flow for Diffusion-Based Generation** Rectified Flow (RF) (Liu et al., 2022) is a reformulation of the traditional diffusion framework, designed to simplify the learning dynamics of generative models. Instead of modeling a complex score function or simulating stochastic processes, RF directly learns a deterministic vector field that transports noisy samples back to data points in a continuous and efficient manner. This deterministic perspective enables the generation process to be framed as solving an ordinary differential equation (ODE), thus reducing the complexity of sampling. 

In standard score-based diffusion models, a sample x _t_ is generated by gradually adding noise to the data x 0, and then denoised via the learned score function ∇ _x_ t log p _t_ (x _t_ ) . In contrast, Rectified Flow directly models the velocity field v _θ_ (x _t_ , t) that maps x _t_ toward x 0 for t ∈ [0, 1] , where t is a continuous time variable. 

The training objective of Rectified Flow is to minimize the velocity matching loss : 

**==> picture [363 x 32] intentionally omitted <==**

where x _t_ is constructed as a linear interpolation between data and noise: 

**==> picture [278 x 11] intentionally omitted <==**

This formulation allows x _t_ to smoothly interpolate between clean data x 0 and Gaussian noise , while the model learns to output a velocity vector that correctly points from x _t_ to x 0, scaled by the temporal factor (1 t) _[−]_[1] . 

At inference time, samples are generated by solving the following ODE: 

**==> picture [266 x 22] intentionally omitted <==**

**==> picture [472 x 246] intentionally omitted <==**

**----- Start of picture text -----**<br>
!-493-.$%&&’<br>/01+244<br>567"3$89’:-093; @=+A&!"<br>!"#$%&&’<br>(")*+,")-.$/01+23<br><=&+(>/2?<br>!"#$%&’ !"#$%&’’<br>!"#$% &’(!$)* &’(!$)* !"#$% !"#$"#<br>!"#$%#&$&%’()*$+#,&-.#+#/<br>0&1234%&*$+#,/&%4#&-$5#&(6%72%&<br>’("+&89&$&.(+72:#+%&5(:#,&;<br>0&<#7%&*$+#,/&%4#&$=$+%)3$":#&<br>3$"5#+%&$,(+#&(+&$&*,$2+&;<br>!"#$%&&’()*+, (-./$0-.+*’!"#$%1’()*+, +("!+* 9:$;)<- 234-1’56,7+"618<br>**----- End of picture text -----**<br>


**Figure 4** Overview of the two-stage data pipeline used to build our mask-free virtual try-on dataset. Stage I bootstraps a raw pool by alternately running a Try-Off model, which generates garment images, and a mask-based try-on model, which produces initial reference images. Stage II iteratively refines and enlarges the dataset: human-in-the-loop filtering cleans the seed pool, JCo-MVTON regenerates sharper triplets, and ICLoRA injects new styles to widen the domain. The cycle repeats until the refined pool reaches the desired quality and diversity. 

Compared to conventional stochastic diffusion processes, Rectified Flow enables a more efficient and stable sampling procedure. This property makes it especially suitable for high-fidelity generation tasks such as virtual try-on (VTON), where stability, sample quality, and alignment between modalities are crucial. 

## **4.2 Data Preparation** 

The objective of our pipeline is to construct a large-scale, high–fidelity mask-free virtual try-on dataset composed of triplets {G, P, R} , where G is a garment image, P is a source person image, and R is the reference (post-try-on) person image. We adopt a two–stage strategy that progressively improves both data quality and domain diversity. 

## **4.2.1 Stage I – Data Collection** 

**I.a Data Collection.** We start from two public benchmarks—VITON (Han et al., 2018a) and DressCode (Morelli et al., 2022)—and crawl additional Internet images. The raw pool contains isolated person images {P } as well as paired garment–person examples {G, P } . Isolated images significantly enrich pose, identity, and lighting variation, but an explicit garment representation is missing. 

**I.b Try-Off garment recovery** To infer G from a sole person image P , we train a Try-Off diffusion model that mirrors the architecture of our JCo-MVTON but swaps the generation target to the garment image. Training pairs are obtained from the paired {G, P } subset, where the garment foreground is extracted with BiRefNet(Zheng et al., 2024). Because Try-Off task is mask-free and leverages the strong generative prior of FLUX, it generalizes well to challenging poses, occlusions, and lighting conditions. 

**I.c Mask-based Try-On.** The recovered pairs {G, P } pairs are fed to a FLUX-Fill model that performs mask-guided try-on. For each person image P we randomly sample a garment G _[′]_ and obtain corresponding R _[′]_ . This produces the first coarse triplet set that bootstraps our mask-free model. 

## **4.2.2 Stage II: Quality and Domain Refinement** 

**II.a Human–in–the–loop Filtering.** Triplets are scored along three axes: (i) **Garment Consistency** ( G vs. P in garment region), (ii) **Person Consistency** ( P vs. R in pose, identity, body shape), and (iii) **Photorealism** of R . We employ professional annotators and an interactive GUI to retain only high-quality samples. 

**II.b ICLoRA-based Domain Expansion.** To broaden style coverage, we employ In-Context LoRA (ICLoRA) (Huang et al., 2024) to fine-tune the frozen FLUX backbone with low-rank adapters on the filtered {G, P } pairs Prompt engineering with an LLM then drives the model to synthesize novel style-aware pairs, ranging from anime style to cyber-punk outfits, largely enriching the domain distribution. 

**II.c Iterative Mask-free Bootstrapping.** The curated triplets train our first–round JCo-MVTON model. Owing to its flexible conditioning (no segmentation mask), JCo-MVTON can now regenerate higher-fidelity R images, especially for failure cases discovered in Stage I.c. The regenerated triplets are re-filtered via II.a, yielding an expanded corpus with both better quality and wider coverage. We iterate II.a–II.c until performance saturates (three rounds in practice, resulting in 120 K triplets). 

## **4.3 Multi-Conditional MMDiT Attention on Virtual Try-on** 

As shown in Figure 5, JCo-MVTON builds upon the FLUX(Labs, 2024) architecture. FLUX injects noisy image embeddings alongside textual prompts into its MM-DIT self-attention layers, leveraging cross-modal interactions to iteratively guide and refine the synthesis process, thereby generating images that faithfully reflect the input text. Inspired by this approach, we propose augmenting the conditioning signal by directly injecting VAE-encoded image latent representations into the self-attention layers. By injecting diverse information as multi-conditional inputs into the self-attention layers, we enable seamless incorporation of various control signals, facilitating high-fidelity, controllable image generation. 

**Conditional Self-Attention for Try-on.** In the mask-free virtual try-on framework, we first encode the text prompt, random noise, reference image, and garment image using a text encoder and a variational auto-encoder (VAE), respectively, to obtain their corresponding intermediate feature maps. These feature maps are then projected into a shared embedding space through an embedding network, yielding embedding vectors T , X , C 1, and C 2. Subsequently, the embeddings are concatenated along the channel dimension to form a unified embedding sequence: 

**==> picture [278 x 11] intentionally omitted <==**

which serves as the conditional input for the subsequent self-attention mechanism to enable coherent garment synthesis and pose transfer. 

In order to preserve the fundamental capabilities of FLUX, the newly injected conditional inputs, such as the reference image, garment image, are processed through the same VAE and embedding layers as the original data. The only modification introduced is the use of two separate query, key, and value (QKV) projection branches during the self-attention operation. These branches are initialized with the weights from the original text & noise projection branch and are updated during training. 

During self-attention computation, the sequences of prompt and noise features are first projected through the primary branch to produce (Q 1 , K 1 , V 1 ) . Similarly, the reference and garment feature sequences are projected via the conditional branch to produce (Q 2 , K 2 , V 2 ) and (Q 3 , K 3 , V 3 ) , respectively. Formally, this projection step can be written as 

**==> picture [367 x 11] intentionally omitted <==**

## **Masked Attention Mechanism for Try-on.** 

Masked self-attention enforces a causality constraint within each individual QKV head before concatenation, ensuring that the multi-head attention mechanism, as a whole, does not utilize information from future tokens in the sequence. In the context of the try-on task, particularly when dealing with multiple conditional inputs, there is often no direct relationship between the features of the reference image and the garment image. When 

!"#$%$"& ’&(")$&* 

**==> picture [469 x 286] intentionally omitted <==**

**----- Start of picture text -----**<br>
./(0%*<br>FG6HIBBFG6JIBBBBBBFG6KIM FL6HIBBBBBBBBBFL6JIBBBBBBBBBBBBBBFL6KIM +"!’ ’/#1*#23%<br>M M 4(*51-1(*&6/#*78<br>FL6HIBFL6JIBBBBBBFL6KIM FLNK6HIBFLNK6JIBBBBBBBFLNK6KIM ’%,-9:(1+% 6/#*78<br>L ; K L ; L<br>!"#$%&’()%*+<br>O#P O#P O#P<br>!" !"<br>(%’%(%)*%<br>>8.<8.<br>!"#$% &%’%(%)*% +,(-%).<br>4,(-%). !"#$%&’’()#*&+,"-./<br>!"#$%$"& ’&(")$&*<br>GBBHBBJBBBBBBBBBBBBB2%)4.0!<br>*".3$/ .%;.BEB)"#$% (%’%(%)*% 4,(-%).<br>)"#$%<br>’%,-&’()%*+ ?@AB ?@AB ?@AB<br>/ ’,$0#") -"1%2 3%,(#)4 $.52#$0 C("D C("D C("D<br>*2".0#)46 0#407(%$"28.#") 9:6 #-4 #1$ E .%;. #1$<br>1%.,#2%1 .%;.8(%$6 (%,2#$.#* ’;&<*7(5%/<br>2#40.#)46 ’,$0#") <0"."4(,<05<br>$.52%= <("-<. 012 .%;.BEB)"#$% (%’%(%)*% 4,(-%).<br>C("-<.<br>M M M M M M<br>!"#<br>!"#!"#<br>**----- End of picture text -----**<br>


**!"#$% &&’(#) *"+ )+,’"$** 

**Figure 5** The overall framework of JCo-MVTON. Given a fixed prompt, reference image, and garment image as inputs, the Joint MM-DiT architecture fuses multi-conditional features to synthesize the try-on image. Within each Joint MM-DiT block, noise and conditional features are processed through three parallel branches and fused via self-attention. 

multi-head attention is applied without considering this, it can lead to suboptimal performance by introducing unwanted interactions between the two input sources. To address this, it is essential to mask the interactions between features from different conditions, thereby preventing them from attending to each other during the attention process. This masking mechanism allows for independent attention among text, noise, and individual conditions, while ensuring that the conditions themselves do not cross-attend. The resulting model not only enhances the generative performance for the try-on task but also helps reduce computational time complexity, as it limits unnecessary interactions during attention computation. 

Let M be the binary mask that controls the allowable attention between conditional branches. We define 

**==> picture [401 x 11] intentionally omitted <==**

where Q _t_ & _n_ , K _t_ & _n_ , V _t_ & _n_ correspond to the text & noise branch, Q _c_ 1 , K _c_ 1 , V _c_ 1 correspond to the reference-image branch, and Q _c_ 2 , K _c_ 2 , V _c_ 2 correspond to the garment-image branch. 

For each position i , we compute raw attention scores against each position j : 

**==> picture [278 x 26] intentionally omitted <==**

We then apply the mask M ∈ R _[n][×][n]_ defined by 

**==> picture [343 x 34] intentionally omitted <==**

Adding M to the score matrix S effectively sets any S _ij_ with (i ∈ c 1 , j ∈ c 2 ) ∨ (i ∈ c 2 , j ∈ c 1 ) to −∞ , so that after the softmax they receive zero weight. 

**==> picture [333 x 25] intentionally omitted <==**

In multi-condition try-on tasks, a mutually exclusive attention mask is introduced to constrain self-attention so that each condition branch attends only to itself or to explicitly permitted branches, thereby preventing interference and noise propagation. This mechanism not only preserves each branch’s internal feature modeling capacity—enhancing the accuracy and robustness of the generated results—but also reduces computational and memory overhead by eliminating invalid attention computations. By flexibly defining the mask matrix, the approach can be extended to additional condition branches or more complex combinations, striking a balance between branch-specific modeling and necessary cross-condition fusion. 

## **4.4 Controllable Positional Encoding for Try-on** 

Transformers fundamentally rely on positional encodings to distinguish among tokens or patches, yet conventional fixed or learned absolute embeddings often fail to generalize across varying sequence lengths or spatial resolutions. In contrast, the FLUX framework adopts rotary positional embeddings (RoPE) (Su et al., 2023) to introduce a natural relative-position bias: by applying a rotation in the query/key vector space, RoPE inherently encodes inter-token relationships and thus seamlessly supports inputs of arbitrary length or resolution. 

**Image Positional Encoding** In the first step of our try-on model, the input image of spatial dimensions H W is divided into a grid of non-overlapping patches, each of size P P , yielding a total of 

**==> picture [65 x 22] intentionally omitted <==**

patches. Each patch is then flattened into a vector of length P[2] C , where C is the number of input channels, and subsequently projected through a learnable linear layer to produce a fixed-dimensional embedding of size d . 

Formally, if we denote the set of flattened patches by 

**==> picture [36 x 12] intentionally omitted <==**

the patch embedding operation can be written as 

**==> picture [178 x 10] intentionally omitted <==**

where 

**==> picture [150 x 14] intentionally omitted <==**

are parameters learned during training. This yields a sequence of patch embeddings {x _i_ } _[N] i_ =1[, each of dimension] d , which serves as the input token sequence for the subsequent Transformer layers. 

## **Try-on based Jonit Positional Encoding.** 

In mask-free virtual try-on tasks, the choice of positional encoding is critical. We compare two schemes: 

- **Scheme I:** Adopt a unified positional space for the prompt, noise, reference image, and garment image. 

- **Scheme II:** Recognizing that in try-on scenarios the noise and reference image share identical spatial dimensions and backgrounds, assign them a common positional space. Inspired by CatVTON(Chong et al., 2025), we then incorporate the garment image into this space by concatenation, thereby preventing interference with the background of the generated output. 

In our scheme, we assume the noise and reference images have spatial dimensions (H, W ) , while the garment image has dimensions (H, H) . Their positional encodings in sequence are: 

**==> picture [106 x 10] intentionally omitted <==**

for the noise and reference images, and 

**==> picture [156 x 10] intentionally omitted <==**

for the garment image. 

Our experiments demonstrate that this concatenation-based positional encoding strategy is better suited to virtual try-on tasks, yielding stronger background preservation and higher-quality garment swapping results. 

## **4.5 Training Strategy** 

**Coarse-to-fine training.** For complex tasks such as virtual try-on, it is difficult for the FLUX(Labs, 2024) base model to produce a high-resolution result in a single training pass. Inspired by the approaches of (Karras et al., 2018; Ho et al., 2021) et al, we therefore first train a low-resolution (512×512) try-on model, and subsequently fine-tune it to obtain a more stable high-resolution (1024×1024) variant. To minimize disruption of the FLUX model’s original capabilities, we freeze all of its pre-existing parameters and train only an auxiliary QKV branch. Two strategies are explored for integrating this branch: one based on LoRA and the other via full-parameter fine-tuning. Our comparative experiments demonstrate that full-parameter training yields superior performance. 

**Fine-Tuning with Augmented Data.** To address the try-on model’s inadequate performance in certain task-specific scenarios, we conducted targeted post-training fine-tuning on the pretrained model, thereby ensuring its strong generalization across multiple contexts. Using the IC-LoRa technique, we generated augmented data that encompassed rare scenarios and uncommon garment types. Experimental results show that training with only 500–1,000 sets of such targeted augmented samples is sufficient to achieve strikingly improved performance. 

## **5 Experiments** 

**Implementation details.** The model was initialized from the original FLUX.1-dev(Labs, 2024) weights with two additional QKV(Vaswani et al., 2023) projection branches and trained on eight NVIDIA H20 GPUs using the Prodigy(Mishchenko and Defazio, 2024) optimizer at a learning rate of 1 in two stages—first at 512×384 resolution with a batch size of 16, then at 1024×768 resolution with a batch size of 4. 

**Datasets.** We have cumulatively collected tens of millions of e-commerce data related to clothing, and then performed basic data preprocessing such as cleaning, deduplication, and filtering. Finally, the training dataset comprises a total of 141,734 high-quality triplets (each consisting of a model, a reference image and a garment), spanning three scenarios: upper-body, lower-body and full-body. Specifically, it contains 69,261 upper-body triplets, 33,838 lower-body triplets and 38,635 dress triplets. All samples were initially generated by IC-LoRA(Huang et al., 2024) and mask-based try-on networks, and subsequently subjected to manual filtering to ensure quality. 

**Metrics.** In paired try-on scenarios where test sets include ground-truth images, we assess the similarity between generated and real images using four standard metrics—Structural Similarity Index (SSIM) (Wang et al., 2004), Learned Perceptual Image Patch Similarity (LPIPS) (Zhang et al., 2018), Frechet Inception Distance (FID)(Seitzer, 2020) and Kernel Inception Distance (KID)(Bińkowski et al., 2021). For unpaired settings, we instead compare the distributions of generated and real samples using FID and KID. 

**Table 1** Quantitative comparison with other methods on VITON-HD(Choi et al., 2021) dataset. The best results are demonstrated in **bold** . 

|nstrated in **bold**.|||
|---|---|---|
|**Methods**|**Paired**<br>SSIM ↑<br>FID ↓<br>KID ↓<br>LPIPS ↓|**Unpaired**<br>FID ↓<br>KID ↓|
|StableGarment (Wang et al., 2024a)<br>MV-VTON (Wang et al., 2025b)<br>LaDI-VTON (Morelli et al., 2023)<br>DCI-VTON (Gou et al., 2023)<br>OOTDifusion (Xu et al., 2024)<br>GP-VTON (Xie et al., 2023)<br>JCo-MVTON (Ours)|0.8029<br>15.567<br>8.519<br>0.1042<br>0.8083<br>15.442<br>7.501<br>0.1171<br>0.8603<br>11.386<br>7.248<br>0.0733<br>0.8620<br>9.408<br>4.547<br>0.0606<br>0.8187<br>9.305<br>4.086<br>0.0876<br>**0.8701**<br>8.726<br>3.944<br>**0.0585**<br>0.8601<br>**8.103**<br>**2.003**<br>0.0891|17.115<br>8.851<br>17.900<br>3.861<br>14.648<br>8.754<br>12.531<br>5.251<br>12.408<br>4.689<br>11.844<br>4.310<br>**9.561**<br>**2.700**|



## **5.1 Comparison with State-of-the-Art Try-on Models** 

To quantitatively evaluate the performance of our try-on model on garment-exchange tasks, we conducted experiments on two datasets—VITON-HD(Choi et al., 2021) and DressCode(Morelli et al., 2022). For each dataset, we trained a dedicated version of the try-on model and carried out independent tests at a resolution of 1024 × 768. The VITON-HD dataset comprises only upper-body garments, whereas the DressCode dataset includes upper-body, lower-body, and dress clothing. Moreover, both datasets provide paired and unpaired configurations for evaluation. 

As shown in Table 1, we compare the performance of JCo-MVTON on the VITON-HD dataset. Owing to its mask-free design, our model substantially outperforms all competing approaches under the unpaired evaluation protocol and achieves state-of-the-art or near–state-of-the-art results under the paired evaluation protocol, particularly surpassing all baselines on both FID and KID metrics. Table 2 reports results on the DressCode dataset, which comprises three distinct garment categories—upper, lower, and dress. Again, JCo-MVTON consistently exceeds the performance of all compared methods, delivering outstanding results across every task category. 

**Table 2** Quantitative comparison on the DressCode (Morelli et al., 2022)dataset for the upper-body, lower-body, and dress tasks. The best results are shown in bold. 

|**Model**|**LPIPS**↓|**SSIM**↑|**FID**_p_ ↓|**KID**_p_ ↓|**FID**_u_ ↓|**KID**_u_ ↓|
|---|---|---|---|---|---|---|
|**Upper**|||||||
|LaDI-VITON|0.1091|0.9044|18.14|3.703|16.43|4.829|
|OOTD|0.0855|0.8997|16.20|5.862|13.20|**1.860**|
|IDM-VTON|0.0761|0.9125|16.25|7.352|13.60|2.952|
|**Ours**|**0.0695**|**0.9123**|**10.92**|**3.022**|**11.53**|2.574|
|**Lower**|||||||
|LaDI-VITON|0.1314|0.8855|14.98|3.920|13.95|**2.564**|
|OOTD|0.1168|0.8706|15.56|3.797|21.50|8.248|
|IDM-VTON|0.1103|0.8869|18.20|8.123|15.97|5.386|
|**Ours**|**0.0721**|**0.8913**|**11.08**|**2.569**|**13.72**|3.83|
|**Dress**|||||||
|LaDI-VITON|0.1753|0.8424|24.00|13.70|16.86|5.005|
|OOTD|0.1490|0.8440|25.75|15.29|20.95|8.149|
|IDM-VTON|0.1381|0.8627|25.96|17.67|12.82|9.739|
|**Ours**|**0.0732**|**0.9032**|**11.82**|**2.942**|**12.54**|**3.576**|



## **5.2 Ablation Study** 

**Condition LoRA Branch** To efficiently and conveniently incorporate conditional variables (i.e. reference and garment images), we propose extending MMDiT by branching its Q, K, and V projections. In our 

**Table 3** Detailed ablation study on the VITON-HD dataset: comparison of three configurations—without positional encoding, without updating all parameters, and our full model —evaluated on paired and unpaired settings. 

|**Method**|**Paired**<br>SSIM ↑<br>FID ↓<br>KID ↓<br>LPIPS ↓|**Unpaired**<br>FID ↓<br>KID ↓|
|---|---|---|
|w/o Pos Encoding<br>w/o Full-Params|0.85<br>9.04<br>3.18<br>0.09<br>0.84<br>10.54<br>4.08<br>0.06|10.20<br>2.77<br>11.49<br>4.76|
|Ours|**0.86**<br>**8.10**<br>**2.00**<br>**0.09**|**9.56**<br>**2.70**|



first scheme, we duplicate the original QKV projection branch, freeze all backbone weights, and train only the newly added branch. In the alternative scheme, we further augment the duplicated branch with a LoRA(Hu et al., 2021) adapter and optimize exclusively its low-rank parameters—leaving both the backbone and duplicated projections intact. By restricting updates to a small subset of parameters, the LoRA-based approach substantially reduces training complexity, accelerates convergence, and better preserves the integrity of the pretrained model. 

**Jonit Positional Encoding.** The FLUX model employs positional encodings to represent the relative spatial locations of pixels. Notably, while the pixels of the target and reference images are spatially aligned with respect to the background, the garment exhibits little inherent spatial correspondence. To reinforce background consistency in the try-on task, we propose to synchronize the positional encodings of the noise input with those of the reference image, whereas the garment image’s positional encoding is constructed via horizontal concatenation. Experimental results demonstrate that this joint positional encoding scheme significantly enhances both the synthesis quality and background preservation capabilities of the model in the try-on task. In Table 3 we can see that once we replace the original sinusoidal positional encoding with one that is learned to exactly match the characteristics of our generated outputs.Intuitively, because the reference model’s positions and the generated clothing-selection positions now share the same embedding space, our simple "concatenate-then-decode" fusion no longer suffers from misaligned spatial priors; this alignment effectively suppresses the cross-interference that previously blurred fine details around sleeve hems and neckline contours. 

By contrast, the version that routes garment-position signals through a LoRA branch underperforms dramatically. We attribute this deficit to LoRA’s inherently low-rank adaptation, which—while parameterefficient—struggles to encode precise, per-pixel spatial offsets. In practice, the branch’s limited capacity to generate sharply localized shifts results in "floating" artifacts around the boundary of the overlaid garment, suggesting that a more expressive injection mechanism (e.g., full dense fusion or higher-rank adapters) would be necessary to regain fine-grained control in the try-on pipeline. 

## **5.3 Visualization** 

To more intuitively and vividly demonstrate our virtual try-on model’s performance across diverse scenarios, we provide visual comparisons between our method and four state-of-the-art research models—CatVTON, IDM, StableGarment, and OOTD—on the VITON-HD and DressCode datasets in Figure 6. 

These comparisons reveal a pronounced advantage in garment consistency, subject clarity, and overall harmony. To further substantiate our results, we conducted rigorous evaluations across multiple try-on tasks under varying conditions. Finally, we benchmarked our model against leading commercial systems (Kling, OutfitAnyone, and GPT-4o) on wild images. The sample outputs in Figure 7 clearly show that our approach achieves more stable performance, adapts effectively to different styles, and consistently yields high-resolution, highly consistent results.Finally, to more effectively demonstrate our model’s performance in real-world tests, we present in Figure 8a a selection of representative results for intuitive illustration. 

**==> picture [424 x 420] intentionally omitted <==**

**Figure 6** The figure presents high-resolution results of our try-on model across various scene-specific tasks, demonstrating its strong generalization capability and the consistently high quality of its outputs. 

## **6 Discussion and Conclusion** 

In this work, we present JCo-MVTON, a novel mask-free virtual try-on model built upon the Multi-Modal Diffusion Transformer (MMDiT) architecture. Inspired by the success of MMDiT in multi-modal generation, we adapt this powerful framework to the multi-conditional nature of virtual try-on, where the generation process must simultaneously respect the reference garment, the wearer’s pose and body structure, and the original image context. Unlike conventional approaches that rely on explicit segmentation masks or dense pose annotations, JCo-MVTON operates directly on RGB inputs, enabling end-to-end training without auxiliary supervision and improving robustness to real-world variations in clothing style and human posture. 

To address the scarcity of high-quality, diverse try-on data, we introduce a dual-stream data construction pipeline that leverages generative models to synthesize large-scale, realistic training samples. One stream generates coherent garment-person compositions through deformation and texture transfer, while the other enhances local fidelity and background consistency via diffusion-based inpainting. We further refine the MMDiT backbone by designing an adaptive attention masking mechanism that reduces interference between conditional inputs, along with a pose-aware positional encoding strategy that improves spatial alignment. 

!"#$%&’( 

’)*+,*-./#.0 

12,.3 

45&$6# 

**Figure 7** Comparison of our model and other commercial systems on wild-image scenarios. From left to right: our result, OutfitAnyone, Kling, and GPT-4o. The samples encompass real-person, anime, and classic styles. Across all style categories, our model delivers superior performance. 

These architectural enhancements lead to more accurate garment draping and stronger preservation of person-specific details and background content. 

**==> picture [424 x 448] intentionally omitted <==**

**Figure 8** Our model’s results on wild images for upper-body, lower-body, and dress try-on scenarios. 

Extensive experiments on both consumer-grade and in-the-wild images demonstrate that JCo-MVTON outperforms existing commercial and academic models in visual realism, structural coherence, and identity preservation. While the current implementation shows promising results, we observe limitations in handling fine-grained details—such as hand gestures and small accessories—and occasional instability in challenging poses. Future work will focus on refining detail reconstruction through higher-fidelity data curation and exploring more robust architectural designs for practical deployment in real-world fashion applications. 

## **References** 

Mostafa Atef, Mariam Ayman, Ahmed Rashed, Ashrakat Saeed, Abdelrahman Saeed, and Ahmed Fares. Efficientviton: An efficient virtual try-on model using optimized diffusion process, 2025. ogb250 . 

Mikołaj Bińkowski, Danica J. Sutherland, Michael Arbel, and Arthur Gretton. Demystifying mmd gans, 2021. ogb . 

|Tianyu Chang, Xiaohao Chen, Zhichao Wei, Xuanpu Zhang, Qing-Guo Chen, Weihua Luo, Peipei Song, and Xun|Yang.|
|---|---|
|Pemf-vto: Point-enhanced video virtual try-on via mask-free paradigm, 2025.|.|
|Lei Chen, Ming Zhao, and Yi Sun. Difusion-based approaches for virtual try-on: A comprehensive study. <br>   <br>, 132(5):1123–1140, 2024a.||
|Mengting Chen, Xi Chen, Zhonghua Zhai, Chen Ju, Xuewen Hong, Jinsong Lan, and Shuai Xiao. Wear-any-way:||
|Manipulable virtual try-on via sparse correspondence alignment, 2024b.|.|
|Seunghwan Choi, Sunghyun Park, Minsoo Lee, and Jaegul Choo.<br>Viton-hd: High-resolution virtual try-on via||
|misalignment-aware normalization, 2021. <br>.||
|Yisol Choi, Sangkyung Kwak, Kyungmin Lee, Hyungwon Choi, and Jinwoo Shin. Improving difusion models for||
|authentic virtual try-on in the wild, 2024. <br>.||
|Zheng Chong, Xiao Dong, Haoxiang Li, Shiyue Zhang, Wenqing Zhang, Xujie Zhang, Hanqing Zhao, Dongmei||
|Jiang, and Xiaodan Liang. Catvton: Concatenation is all you need for virtual try-on with difusion models,|2025.|
|.||
|Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik||
|Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, Kyle Lacey, Alex Goodwin,||
|Yannik Marek, and Robin Rombach. Scaling rectifed fow transformers for high-resolution image synthesis,|2024.|
|.||
|Junhong Gou, Siyu Sun, Jianfu Zhang, Jianlou Si, Chen Qian, and Liqing Zhang. Taming the power of difusion||
|models for high-quality virtual try-on with appearance fow.<br>In      <br>  <br>, MM ’23, page 7599–7607. ACM, October 2023. doi: 10.1145/3581783.3612255. <br><br>.||
|Hailong Guo, Bohan Zeng, Yiren Song, Wentao Zhang, Chuang Zhang, and Jiaming Liu. Any2anytryon: Leveraging||
|adaptive position embeddings for versatile virtual clothing tasks, 2025.|.|
|Xingjie Han, Zhe Wu, and Yunhong Zhang. Viton: An image-based virtual try-on network. In||
|<br>, 2018a.||
|Xintong Han, Zuxuan Wu, Zhe Wu, Ruichi Yu, and Larry S. Davis. Viton: An image-based virtual try-on network,||
|2018b. <br>.||
|Jonathan Ho, Chitwan Saharia, William Chan, David J. Fleet, Mohammad Norouzi, and Tim Salimans. Cascaded||
|difusion models for high fdelity image generation, 2021. <br>.||
|Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen.||
|Lora: Low-rank adaptation of large language models, 2021. <br>.||
|Lianghua Huang, Wei Wang, Zhi-Fan Wu, Yupeng Shi, Huanzhang Dou, Chen Liang, Yutong Feng, Yu Liu, and||
|Jingren Zhou. In-context lora for difusion transformers.   <br>, 2024.||
|Farah Islam and Rohan Patel. Enhancing e-commerce with virtual try-on technologies.    <br>, 7||
|(2):45–58, 2023.||
|Boyuan Jiang, Xiaobin Hu, Donghao Luo, Qingdong He, Chengming Xu, Jinlong Peng, Jiangning Zhang, Chengjie||
|Wang, Yunsheng Wu, and Yanwei Fu. Fitdit: Advancing the authentic garment details for high-fdelity virtual||
|try-on, 2024. <br>.||
|Tero Karras, Timo Aila, Samuli Laine, and Jaakko Lehtinen. Progressive growing of gans for improved quality, stability,||
|and variation, 2018. <br>.||
|Black Forest Labs. Flux. <br>, 2024.||
|Sangho Lee and Jisoo Kim. A survey of difusion models in generative virtual try-on.||
|<br>, 45(11):3201–3217, 2023.||
|Xiao Liu, Antonio Perez, and Miguel Torres. Mask-free virtual try-on with implicit warping. In   <br>    <br>, 2024.||
|Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data|with|
|rectifed fow, 2022. <br>.||



- Konstantin Mishchenko and Aaron Defazio. Prodigy: An expeditiously adaptive parameter-free learner, 2024. ogb23 . 

- Davide Morelli, Matteo Fincato, Marcella Cornia, Federico Landi, Fabio Cesari, and Rita Cucchiara. Dress code: High-resolution multi-category virtual try-on, 2022. ogb . 

- Davide Morelli, Alberto Baldrati, Giuseppe Cartella, Marcella Cornia, Marco Bertini, and Rita Cucchiara. Ladi-vton: Latent diffusion textual-inversion enhanced virtual try-on, 2023. ogb2 . 

- Xiaoyu Niu, Qiang Sun, and Jun Liu. Pfdm: Pseudo-image guided diffusion models for mask-free virtual try-on. In European rce (EV) , 2024. 

- William Peebles and Saining Xie. Scalable diffusion models with transformers, 2023. ogb 

   - . 

- Maximilian Seitzer. pytorch-fid: FID Score for PyTorch. **come** y-fd , August 2020. Version 0.3.0. 

- Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding, 2023. ogb21 . 

- Xianbing Sun, Yan Hong, Jiahui Zhan, Jun Lan, Huijia Zhu, Weiqiang Wang, Liqing Zhang, and Jianfu Zhang. Ds-vton: High-quality virtual try-on via disentangled dual-scale generation, 2025. ogb25 . 

- Zhenxiong Tan, Qiaochu Xue, Xingyi Yang, Songhua Liu, and Xinchao Wang. Ominicontrol2: Efficient conditioning for diffusion transformers, 2025. ogb . 

- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need, 2023. ogb1 . 

- Riza Velioglu, Petra Bevandic, Robin Chan, and Barbara Hammer. Tryoffdiff: Virtual-try-off via high-fidelity garment reconstruction using diffusion models, 2024. ogb24 . 

- Zhenchen Wan, Yanwu xu, Dongting Hu, Weilun Cheng, Tianxi Chen, Zhaoqing Wang, Feng Liu, Tongliang Liu, and Mingming Gong. Mf-viton: High-fidelity mask-free virtual try-on with minimal input, 2025. xiv.og . 

- Haoxuan Wang, Jinlong Peng, Qingdong He, Hao Yang, Ying Jin, Jiafu Wu, Xiaobin Hu, Yanjie Pan, Zhenye Gan, Mingmin Chi, Bo Peng, and Yabiao Wang. Unicombine: Unified multi-conditional combination with diffusion transformer, 2025a. ogb . 

- Haoyu Wang, Zhilu Zhang, Donglin Di, Shiliang Zhang, and Wangmeng Zuo. Mv-vton: Multi-view virtual try-on with diffusion models, 2025b. ogb2 . 

- Rui Wang, Hailong Guo, Jiaming Liu, Huaxia Li, Haibo Zhao, Xu Tang, Yao Hu, Hao Tang, and Peipei Li. Stablegarment: Garment-centric generation via stable diffusion, 2024a. ogb24 . 

- Xi Wang, Yifan Li, and Lei Zhang. Boow-vton: Body-oriented wearable virtual try-on with diffusion models. In Proc rce Rec , 2024b. 

- Zhou Wang, A.C. Bovik, H.R. Sheikh, and E.P. Simoncelli. Image quality assessment: from error visibility to structural similarity. Tr Proce , 13(4):600–612, 2004. doi: 10.1109/TIP.2003.819861. 

- Yiming Xie, Varun Jampani, Lei Zhong, Deqing Sun, and Huaizu Jiang. Omnicontrol: Control any joint at any time for human motion generation, 2024. ogb231 . 

- Zhenyu Xie, Zaiyu Huang, Xin Dong, Fuwei Zhao, Haoye Dong, Xijin Zhang, Feida Zhu, and Xiaodan Liang. Gpvton: Towards general purpose virtual try-on via collaborative local-flow global-parsing learning, 2023. 01756 . 

- Yuhao Xu, Tao Gu, Weifeng Chen, and Chengcai Chen. Ootdiffusion: Outfitting fusion based latent diffusion for controllable virtual try-on, 2024. ogb24 . 

- Zhaotong Yang, Zicheng Jiang, Xinzhe Li, Huiyu Zhou, Junyu Dong, Huaidong Zhang, and Yong Du. D -vton: Dynamic semantics disentangling for differential diffusion based virtual try-on, 2024. ogb2407 . 

- Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric, 2018. ogb . 

- Xuanpu Zhang, Dan Song, Pengxin Zhan, Tianyu Chang, Jianhao Zeng, Qingguo Chen, Weihua Luo, and Anan Liu. Boow-vton: Boosting in-the-wild virtual try-on via mask-free pseudo data training, 2024. xiv.ogb . 

- Yuxuan Zhang, Yirui Yuan, Yiren Song, Haofan Wang, and Jiaming Liu. Easycontrol: Adding efficient and flexible control for diffusion transformer, 2025. ogb . 

- Peng Zheng, Dehong Gao, Deng-Ping Fan, Li Liu, Jorma Laaksonen, Wanli Ouyang, and Nicu Sebe. Bilateral reference for high-resolution dichotomous image segmentation. 1.37 , 2024. 

