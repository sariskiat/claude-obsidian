---
type: paper-fulltext
slug: ted-viton-transformer-empowered-diffusion
arxiv_id: null
source_path: /Users/saris.kia.adm/.paper-scholar/ted-viton-transformer-empowered-diffusion/2411.17017.md
paper: "[[ted-viton-transformer-empowered-diffusion]]"
---
<!-- extracted by afk_extract from 2411.17017.pdf (10p) -->

## **TED-VITON: Transformer-Empowered Diffusion Models for Virtual Try-On** 

Zhenchen Wan[1] Yanwu Xu Zhaoqing Wang[2] Feng Liu[1] Tongliang Liu[2] Mingming Gong[1] _[,]_[3] 1University of Melbourne, Melbourne, Australia 2The University of Sydney, Sydney, Australia 

3Mohamed bin Zayed University of Artificial Intelligence, Abu Dhabi, UAE 

zhenchenw@student.unimelb.edu.au, zwan6779@uni.sydney.edu.au, fengliu.ml@gmail.com tongliang.liu@sydney.edu.au, mingming.gong@unimelb.edu.au 

**==> picture [496 x 258] intentionally omitted <==**

Figure 1. We propose an implementation of Virtual Try-On using the Diffusion Transformer (DiT) architecture, which demonstrates stateof-the-art visual quality by preserving fine garment details and text clarity, even under challenging conditions involving complex human poses and diverse lighting environments. 

## **Abstract** 

_Recent advancements in Virtual Try-On (VTO) have demonstrated exceptional efficacy in generating realistic images and preserving garment details, largely attributed to the robust generative capabilities of text-to-image (T2I) diffusion backbones. However, the T2I models that underpin these methods have become outdated, thereby limiting the potential for further improvement in VTO. Additionally, current methods face notable challenges in accurately rendering text on garments without distortion and preserving fine-grained details, such as textures and material fidelity. The emergence of Diffusion Transformer (DiT) based T2I_ 

_models has showcased impressive performance and offers a promising opportunity for advancing VTO. Directly applying existing VTO techniques to transformer-based T2I models is ineffective due to substantial architectural differences, which hinder their ability to fully leverage the models’ advanced capabilities for improved text generation. To address these challenges and unlock the full potential of DiT-based T2I models for VTO, we propose TEDVITON, a novel framework that integrates a Garment Semantic (GS) Adapter for enhancing garment-specific features, a Text Preservation Loss to ensure accurate and distortion-free text rendering, and a constraint mechanism to generate prompts by optimizing Large Language Model_ 

1 

_(LLM). These innovations enable state-of-the-art (SOTA) performance in visual quality and text fidelity, establishing a new benchmark for VTO task. Project page: https: //zhenchenwan.github.io/TED-VITON/_ 

## **1. Introduction** 

Using input images of an individual and a selected garment, image-based VTO technology generates realistic images of the individual wearing the selected garment. By bypassing the necessity for physical fitting, VTO offers a transformative solution for applications in e-commerce, fashion cataloging, and the burgeoning metaverse. The primary challenges in VTO are threefold: (1) human body alignment, whereby the generated try-on image must accurately reflect the person’s body shape and pose; (2) garment fidelity, which preserves fine garment details, such as texture, color, and logo clarity, is essential to ensure authenticity; and (3) image quality, which pertains to the final output’s resolution and the absence of artifacts. 

While early VTO methods based on Generative Adversarial Networks (GANs) [10] addressed these challenges to some extent [2, 4, 7, 13, 28, 40, 44, 46], they often struggled with garment misalignment, visible artifacts, and limited generalizability. To address these limitations, diffusion models [16] have emerged as a promising alternation in VTO research, leveraging a progressive noise-reversal process that enhances control over image generation and significantly improves texture and detail preservation. Recent UNet-based methods [5, 11, 19, 23, 30, 31, 39] utilize the generative strength of pretrained text-to-image (T2I) diffusion models [34, 36] to capture detailed garment semantics and enhance the realism of try-on images. These approaches achieve high image fidelity by encoding garment semantics through simple description [5, 30] or using explicit warping networks [11, 39] to align garment structure with human poses. However, despite their advancements, these models still face challenges in preserving finegrained garment details, such as logos, text, and intricate textures, and often struggle with accurately representing natural lighting and adapting to complex body poses. 

To overcome these limitations, we explore the use of transformers in diffusion models, specifically building upon the DiT architecture [8]. Unlike UNet-based architectures, transformers offer enhanced scalability, long-range dependency modeling, and the ability to handle diverse visual contexts. However, directly migrating existing VTO approaches to the Transformer-based diffusion model does not guarantee performance improvements, as traditional UNetbased methods fail to fully exploit the potential of the transformer architecture. This observation aligns with our initial experiments, where a naive application of prior VTO techniques on DiT yielded suboptimal results. 

To harness the capabilities of DiT, this paper proposes _Transformer-Empowered Diffusion Models for Virtual TryOn_ (TED-VITON), a framework designed to overcome key challenges in VTO by leveraging Transformer-based diffusion architectures. TED-VITON integrates several novel components to address limitations in garment detail preservation, model generalization, and text fidelity. Our contributions can be summarized as follows: 

- **Successful Migration of VTO to DiT-based Architecture:** We demonstrate the successful adaptation of Virtual Try-On technology to a DiT-based architecture. This paves the way for subsequent enhancements in the preservation of garment detail, semantic alignment, and visual fidelity. 

- **Enhanced Garment Semantics with GS-Adapter:** Integrating the GS-Adapter, TED-VITON precisely aligns high-order semantic features from the image encoder with the DiT. This integration allows the model to more accurately capture occlusions, wrinkles, and material properties, maintaining realism across varied poses. 

- **Text and Logo Clarity through Text Preservation Loss:** We introduce a Text Preservation Loss to address common challenges in text and logo fidelity. This loss function effectively enhances clarity and mitigates distortion, ensuring high-quality, distortion-free renderings of logos and text, critical for garments with complex designs. 

- **Optimized Prompt Generation through Constraint Mechanism for LLMs:** To optimize DiT training, we introduce a constraint mechanism that tailors LLM prompts to garment-specific semantics. This mechanism improves training input quality, facilitating effective learning and generating outputs with superior visual fidelity. 

## **2. Related Works** 

**Pose-Guided Person Image Synthesis (PPIS).** VTO technology originated with Pose-Guided Person Image Synthesis (PPIS). Initial PPIS approaches aimed to generate person images conditioned on specific body poses, laying the groundwork for generating visually convincing images of people in various postures. Pioneering works in this domain [1, 9, 22, 24, 26, 27, 49, 51] concentrated on aligning human poses with target clothing images, addressing key challenges in pose transfer and adapting to individual body shapes. 

**GAN-based VTO.** Following PPIS advancements, VTO progressed to the application of Generative Adversarial Networks (GANs) for 2D VTO. GAN-based VTO approaches [1, 7, 9, 17, 20, 21, 24, 27, 33, 35, 38, 43] typically involve two stages: deforming the garment to match the target person’s body shape, followed by fusing this deformed garment with the person’s image. Methods for improving garment deformation include using dense flow maps to cre- 

2 

ate a seamless fit, while normalization and distillation techniques help to minimize misalignment. However, GANbased VTO models face generalization limitations, especially in complex backgrounds and varied poses, limiting their applicability in dynamic real-world environments. **Diffusion-based VTO.** Diffusion models have opened new avenues in VTO, enabling enhanced fidelity and detail preservation. Recent diffusion-based VTO methods [3, 5, 6, 11, 19, 23, 30, 31, 39, 50] extend beyond standard Stable Diffusion (SD), often employing customized architectures to boost performance. For instance, StableVITON [19] builds on SD1.4 [36] and incorporates ControlNet [47] to enhance control over garment and body alignment, while IDM-VTON [5] leverages SDXL [34] with IP-Adapter [45] to refine garment-body fit through additional image-based control signals. These approaches effectively address key limitations of GAN-based methods, particularly in garment fidelity and preservation of fine details, establishing diffusion-based models as suitable for complex VTO applications. However, preserving intricate elements like garment text, logos, and texture under diverse poses and lighting conditions remains a challenge. TED-VITON aims to bridge these gaps, advancing VTO with a DiT architecture that integrates the GS-Adapter for semantic alignment, a DINOv2 encoder for capturing fine-grained garment details, and a Text Preservation Loss that ensures clarity in logos and text. 

## **3. Methodology** 

## **3.1. Background on Controlling Diffusion Models** 

**Stable Diffusion (SD) 3 Model.** The SD 3 model [8] represents a significant breakthrough as the first diffusion model to utilize a Transformer-based architecture. Building upon Latent Diffusion Models (LDMs) [36], SD3 introduces the rectified flow approach [25], which connects data points in the latent space via straight linear paths, replacing the traditional curved trajectories. This straight-line trajectory minimizes noise accumulation and allows for efficient, highquality image synthesis. 

In SD3, the input image _x_ is encoded into a latent representation _z_ 0 = _E_ ( _x_ ) by a pre-trained encoder _E_ . The rectified flow formulation defines a forward diffusion process with a variance schedule _βt_ expressed as follows: 

**==> picture [194 x 11] intentionally omitted <==**

where _t ∈{_ 1 _, . . . , T }_ indicates diffusion steps, _αt_ := 1 _− βt_ , and _α_ ¯ _t_ :=[�] _[t] s_ =1 _[α][s]_[.][SD3][leverages][the][Conditional] Flow Matching (CFM) loss to guide rectified flow during training: 

**==> picture [226 x 18] intentionally omitted <==**

where _ut_ ( _z|ϵ_ ) denotes the rectified vector field for direct, linear alignment. Unlike prior diffusion models, SD3 employs a Transformer backbone (Multimodal DiT) that facilitates bidirectional information flow between text and image tokens. This multimodal structure enhances text comprehension and visual quality, making SD3 highly suitable for text-guided image generation with improved fidelity and detail retention. 

**ControlNet for Conditional Image Generation.** ControlNet [47] extends diffusion models by enabling conditional image generation with additional guidance inputs, such as edge maps, segmentation masks, or pose annotations. ControlNet operates by branching from the base model’s intermediate features _F_ . The conditional input _C_ is processed with learnable weights _Wc_ , resulting in conditioned features _F_ ctrl = ControlNet( _C_ ; _Wc_ ). These conditioned features are merged back into the main pipeline as _F_ combined = _F_ + _λF_ ctrl, where _λ_ regulates the influence of the conditional input. During training, ControlNet minimizes a composite loss: 

**==> picture [169 x 11] intentionally omitted <==**

where _L_ diff is the base diffusion model’s loss, _L_ cond ensures alignment with the guidance input, and _γ_ balances their contributions. This framework enables precise control over image generation, making ControlNet highly effective for tasks requiring fine-grained customization. 

## **3.2. TED-VITON** 

Figure 2 (a) illustrates the TED-VITON framework, comprising DiT-GarmentNet, the Garment Semantic (GS) Adapter and DiT-TryOnNet. The following section provides a comprehensive description of each module and the training procedure. 

**DiT-GarmentNet.** DiT-GarmentNet is designed to extract fine-grained garment features, including textures, patterns, fabric structures, logos, and other subtle design elements essential for realistic VTO results. By preserving the garment’s true visual characteristics, this module ensures high fidelity, particularly in applications requiring precise appearance rendering. 

DiT-GarmentNet processes the latent representation of the garment image _E_ ( _Xg_ ), extracted via a pre-trained VAE encoder _E_ , along with the conditioned text prompt _τθ_ ( _D_ ) generated by a multi-modal text encoders. These representations flow through multiple transformer layers, refining and retaining intricate garment details. The transformer architecture, inspired by Esser et al. [8], captures long-range dependencies, ensuring consistent textures, accurate logo placement. 

DiT-GarmentNet processes the garment image _Xg_ alongside the conditioned text prompt _D_ , is defined as: 

**==> picture [210 x 13] intentionally omitted <==**

3 

**==> picture [496 x 343] intentionally omitted <==**

Figure 2. **Overview of TED-VITON:** We present the architecture of the proposed model along with details of its block modules. **(a)** Our model consists of 1) DiT-GarmentNet that encodes fine-grained features of _Xg_ , 2) GS-Adapter [45] that captures higher-order semantics of garment image _Xg_ , and 3) DiT-TryOnNet, the main Transformer for processing person images. The Transformer input is formed by concatenating the noised latents _Xt_ with the segmentation mask _m_ , masked image _E_ ( _X_ model), and Densepose [12] _E_ ( _x_ pose). Additionally, a detailed description of the garment (e.g., “[D]: The clothing item is a black T-shirt...”) is generated through an LLM and fed as input to both the DiT-GarmentNet and DiT-TryOnNet. The model aims to preserve garment-specific details through a text preservation loss, which ensures that key textual features are retained. **(b)** Intermediate features from DiT-TryOnNet and DiT-GarmentNet are concatenated. These are then refined through joint-attention and cross-attention layers, with the GS-Adapter further contributing to the refinement process. In this architecture, the DiT-TryOnNet and GS-Adapter modules are fine-tuned, while other components remain frozen. 

where _F_ garment _[i]_[denotes][the][fine-grained][features][extracted] from the _i_ -th transformer layer of DiT-GarmentNet. 

In this way, DiT-GarmentNet ensures high visual fidelity by combining garment-specific details with broader model context, enabling the VTO system to accurately render complex designs on various body shapes and poses. 

**Garment Semantic Adapter (GS-Adapter).** The GSAdapter [45] is a key module that enhances generalization, making the model less sensitive to variations in body poses, garment deformations, and conditions like lighting or camera angles. By focusing on low-frequency features, it captures essential garment attributes, enabling consistent performance across diverse scenarios. 

Unlike DiT-GarmentNet, which extracts high-frequency 

details like textures and logos, the GS-Adapter uses the DINOv2 encoder [32] to distill semantic garment information, including structure, style, and material. These highorder semantics, _H_ semantic, encapsulate broader contextual attributes while maintaining adaptability. 

The GS-Adapter employs a decoupled cross-attention mechanism to independently process joint and image embeddings. Let **Q** _∈_ R _[N][×][d]_ represent the query matrix, and **K** _j,_ **V** _j_ and **K** _i,_ **V** _i_ denote key-value pairs for joint and image embeddings, respectively. The combined output is: 

**==> picture [237 x 22] intentionally omitted <==**

4 

where _λ_ balances image and joint feature contributions. This design allows the GS-Adapter to generalize effectively across diverse poses, complex garments, and varying environmental conditions, enhancing model robustness and ensuring realistic outputs. 

**DiT-TryOnNet.** DiT-TryOnNet builds upon the DiT architecture, leveraging its powerful Transformer-based diffusion capabilities within the latent space of a pre-trained VAE. By integrating DiT, our model benefits from the scalability and long-range dependency modeling of Transformers, enabling precise alignment and realistic rendering in virtual try-on scenarios. For DiT-TryOnNet, we construct a combined input _ζ_ = [ _E_ ( _X_ model); _m_ ; _E_ ( _X_ mask); _E_ ( _X_ pose)] to provide a comprehensive context. This input consists of: the person’s latent image representation _E_ ( _X_ model) as the primary structural guide; a dynamically resized mask _m_ to isolate the garment area and focus the model’s attention; the masked person’s image _X_ mask = (1 _− m_ ) _⊙ X_ model for garment reconstruction; and the DensePose embedding _E_ ( _X_ pose) to align with the person’s pose. 

Within the **MM-DiT-Block** (Fig. 2(b)), fine-grained garment details _F_ garment _[i]_[extracted][from][the] _[i]_[-th][transformer] layer of DiT-GarmentNet, merge with the feature representation _F_ tryon _[i]_[from][the][corresponding] _[i]_[-th][layer][DiT-] TryOnNet to form _F_ image _[i]_[,][which][serves][as][the][primary][in-] put for attention processing. Descriptive text embeddings _τθ_ ( _D_ ), generated by multimodal text encoders, are concatenated with _F_ image _[i]_[within][the][query,][key,][and][value][com-] ponents of the joint attention mechanism (i.e., _Q_ joint = Concat( _Q[i]_ image _[, Q][τ] θ_[(] _[D]_[)][)][,] _[K]_[joint][=][Concat][(] _[K]_ image _[i][, K][τ] θ_[(] _[D]_[)][)][,] _V_ joint = Concat( _V_ image _[i][, V][τ] θ_[(] _[D]_[)][)][).][This][results][in][a][hidden] state _H_ joint _[i]_[that unifies visual and textual modalities.][Subse-] quently, this hidden state is further enriched by incorporating high-order semantic features, _H_ semantic provided by the GS-Adapter, as described in Eq. 5. 

To produce the final VTO output _X_[ˆ] , DiT-TryOnNet leverages the combined input _ζ_ and garment description embedding _τθ_ ( _D_ ): 

**==> picture [182 x 13] intentionally omitted <==**

**Prior Preservation for Text Generation.** To retain the model’s ability to generate accurate and clear text, such as logos and labels, we introduce a prior preservation mechanism inspired by DreamBooth [37]. This mechanism incorporates a text preservation loss to ensure text clarity and fidelity, preventing the model from losing this capability while fine-tuning for VTO tasks. As the final component of our framework, prior preservation complements the GSAdapter and DiT-TryOnNet. Together, they form a comprehensive training objective, achieving a balance between high-fidelity garment rendering and robust text generation for realistic VTO outputs. 

As shown in Fig. 2(a), the total loss function combines two main components: (1) the CFM loss _L_ CFM defined in Eq. 2, which ensures high-quality VTO outputs by aligning generated images with the desired garment and pose, and (2) the text preservation loss _L_ pres, which maintains clarity in text details. The CFM loss guides the model in generating the VTO result _X_[ˆ] leveraging DiT-GarmentNet for detail retention and DiT-TryOnNet for fit adjustments based on pose and body type. The text preservation loss _L_ pres is computed as _L_ pres = MSE( _X,_[ˆ] _X_[ˆ] _[′]_ ), where _X_[ˆ] _[′]_ is the baseline latent representation from the original model, helping to retain text fidelity in the fine-tuned output. The final loss function is given by: 

**==> picture [176 x 12] intentionally omitted <==**

where _λ_ pres controls the balance between VTO adaptation and text retention. This approach enables high-quality garment realism while preserving essential text rendering for realistic try-on images. 

**GPT-4o Generated Garment Descriptions.** Our approach uses GPT-4o to generate detailed garment descriptions that capture both basic and nuanced features. These descriptions provide rich semantic context, enhancing the model’s ability to faithfully represent garment details. For **DiTGarmentNet** , descriptions help preserve intricate details like texture and logos. Meanwhile, for **DiT-TryOnNet** , the text prompt is tailored to emphasize how the garment appears when worn, focusing on fit and interaction with the body. This adjustment improves realism in the generated images. This dual-conditioning approach enables more accurate garment representation, as shown in Fig. 2(a). 

## **4. Experiment** 

To thoroughly evaluate TED-VITON, we conduct a comprehensive study that includes quantitative and qualitative analyses, ablation studies to assess the contributions of individual components, and a user study to gauge human preferences. For the quantitative analysis, we measure standard metrics to evaluate the generated images’ alignment with ground truth and overall visual quality. In the qualitative analysis, we compare TED-VITON’s outputs with those of baseline models to examine its ability to reproduce fine garment features, such as textures, logos, and material details. We also perform ablation studies by systematically removing key components to assess their impact on performance and image quality. Since human preference is a critical measure of success in generative tasks, we conduct a user study to gather feedback on the perceived realism and aesthetic appeal of the fitting images. The results highlight TEDVITON’s ability to produce visually compelling and realistic outputs that surpass existing methods. 

5 

**==> picture [496 x 276] intentionally omitted <==**

Figure 3. Qualitative comparison with baseline methods. This figure demonstrates the superior performance of our model compared to various SOTA approaches. Zooming in reveals finer details. 

|**Dataset**<br>**Method**|**VITON-HD**<br>LPIPS↓<br>SSIM↑<br>CLIP-I↑<br>FID↓UN<br>KID↓UN|**VITON-HD**<br>LPIPS↓<br>SSIM↑<br>CLIP-I↑<br>FID↓UN<br>KID↓UN|**VITON-HD**<br>LPIPS↓<br>SSIM↑<br>CLIP-I↑<br>FID↓UN<br>KID↓UN|**DressCode Upper-body**|
|---|---|---|---|---|
|||||LPIPS↓<br>SSIM↑<br>CLIP-I↑<br>FID↓UN<br>KID↓UN|
|||**GAN-based methods**|||
|**HR-VITON [21]**<br>**SD-VITON[38]**||0.115<br>0.877<br>0.800<br>12.238<br>3.757<br>0.104<br>**0.896**<br>0.831<br>9.857<br>1.450||0.118<br>0.910<br>0.749<br>29.383<br>3.104<br>-<br>-<br>-<br>-<br>-|
|||**Diffusion-based methods**|||
|**LaDI-VTON [30]**<br>**DCI-VTON [11]**<br>**StableVITON [19]**<br>**IDM–VTON [5]**<br>**TED-VITON (Ours)**||0.166<br>0.873<br>0.819<br>0.197<br>0.863<br>0.823<br>0.142<br>0.875<br>0.838<br>0.102<br>0.868<br>0.875<br>**0.095**<br>0.881<br>**0.878**|9.386<br>1.590<br>9.775<br>1.762<br>9.371<br>1.990<br>9.156<br>1.242<br>**8.848**<br>**0.858**|0.157<br>0.905<br>0.789<br>22.689<br>2.580<br>0.171<br>0.893<br>0.756<br>24.184<br>2.379<br>0.113<br>0.910<br>0.844<br>19.712<br>2.149<br>0.065<br>0.920<br>0.870<br>11.852<br>**1.181**<br>**0.050**<br>**0.934**<br>**0.875**<br>**11.451**<br>1.393|



Table 1. Quantitative comparison on training models on VITON-HD and evaluate them on both VITON-HD and DressCode upper-body datasets. **Bold** and underline denote the best and the second best result, respectively. “UN” indicated the unpaired setting. KID score is multiplied by 100. 

## **4.1. Experiment Setup** 

**Baselines.** We evaluate our method against both GANbased and diffusion-based VTO approaches. The GANbased baselines include HR-VITON [21] and SD-VTON [38]. Both methods employ a separate warping module to fit the garment onto the target person, followed by GAN-based generation with the fitted garment as input. Among the diffusion-based methods, we compare with LaDI-VTON [30], DCI-VTON [11], StableVITON [19], and IDM-VTON [5]. All of these models leverage pretrained SD models, though with different conditioning 

techniques. LaDI-VTON and DCI-VTON incorporate distinct warping modules for garment conditioning, while StableVITON directly uses the SD1.4 encoder for conditioning. IDM-VTON, by contrast, utilizes the SDXL inpainting model checkpoints from official repositories. Our approach similarly builds on SD3 original checkpoints from official sources. For a fair comparison, we generate images at a resolution of 1024 _×_ 768 when available; otherwise, we generate images at 512 _×_ 384 and upscale them to 1024 _×_ 768 using interpolation or super-resolution techniques [41], reporting the highest-quality results achieved. 

**Evaluation datasets.** We evaluate the effectiveness of 

6 

**==> picture [460 x 244] intentionally omitted <==**

Figure 4. An ablation study on the key components of TED-VITON. 

|**Component**<br>**w/o DINOv2**<br>**w/o GS-Adapter**|**LPIPS↓**<br>0.120<br>0.111|**SSIM↑**<br>0.870<br>0.842|**CLIP-I↑**<br>0.852<br>0.829|**FID↓UN**<br>9.655<br>9.674|**KID↓UN**<br>1.734<br>1.680|
|---|---|---|---|---|---|
|**w/o DiT-GarmentNet**<br>**w/o Text Preservation Loss**|0.113<br>0.098|0.850<br>0.877|0.817<br>0.864|9.931<br>9.438|1.693<br>1.487|
|**Full Model**|**0.095**|**0.881**|**0.878**|**8.848**|**0.858**|



Table 2. Quantitative comparison of models trained on the VITON-HD dataset with and without each component. “UN” indicated the unpaired setting. KID score is multiplied by 100. 

TED-VITON on two widely-used VTO datasets, VITONHD [4] and DressCode [29]. The VITON-HD dataset consists of 13,679 pairs of frontal-view images of women and corresponding upper garments. Following the standard dataset practices of previous works [5, 11, 19, 30, 39], we divide VITON-HD into a training set of 11,647 pairs and a test set of 2,032 pairs. The DressCode dataset contains 15,366 image pairs focused specifically on upper-body garments. Consistent with the original dataset splits, we use 1,800 upper-body image pairs from DressCode as the test set. All experiments on both VITON-HD and DressCode are conducted at a resolution of 1024 _×_ 768. 

**Evaluation metrics.** We evaluate TED-VITON in both paired and unpaired settings, following established practices in VTO literature. In the paired setting, the input garment matches the one originally shown in the person image. To assess performance, we use three key metrics: Structural Similarity Index (SSIM) [42], Learned Perceptual Image Patch Similarity (LPIPS) [48] and the CLIP image similarity score (CLIP-I) [14] to measure similarity between the generated image and the ground truth. Additionally, in the unpaired setting, where the garment in the person im- 

age is replaced with a different one and no ground truth is available, we assess TED-VITON’s performance in terms of image quality and realism using Fr´echet Inception Distance (FID) [15] and Kernel Inception Distance (KID) [18] scores. 

## **4.2. Qualitative Results** 

Fig. 3 provides a qualitative comparison of VTO models alongside the input person image and selected garments. TED-VITON stands out as the only model capable of accurately reproducing text details on garments, such as the large “1969” and “Wrangler” logos, as well as finer text like “Vans”. In terms of color and texture fidelity, TEDVITON precisely aligns the four colors in “1969” across the text rows, maintaining the original garment’s design. Unlike other models, which often exhibit text distortion or color misalignment, TED-VITON preserves text clarity and color accuracy. This is achieved through the integration of a Text Preservation Loss and enhanced prompt conditioning, which together ensure that fine-grained text and color details are retained in the generated VTO images. 

## **4.3. Quantitative Results** 

**VITON-HD.** We evaluate TED-VITON on the VITON-HD dataset and compare it with SOTA VTO methods, including GAN-based approaches (HR-VITON [21] and SD-VITON [38]) and diffusion-based methods (LaDI-VTON [30], DCIVTON [11], StableVITON [19], and IDM-VTON [5]). Table 1 presents the quantitative results, where TED-VITON achieves top scores in LPIPS, CLIP-I, FID, and KID, in- 

7 

**==> picture [237 x 96] intentionally omitted <==**

Figure 5. User study results based on 10 selected pairs from the VITON-HD [4] dataset: 5 pairs assessed text and logo preferences, and the other 5 focused on pattern and texture preferences. 

dicating superior perceptual quality and realism. It ranks second in SSIM, highlighting its strong structural similarity preservation and alignment with perceptual semantics. **DressCode Upper-body.** To evaluate TED-VITON’s generalization across diverse garment styles, we test it on the DressCode upper-body dataset. As shown in Table 1, TED-VITON outperforms other models across most metrics, achieving top scores in LPIPS, SSIM, CLIP-I and FID scores, which indicate strong alignment with the perceptual features of the in-shop garment. TED-VITON outperforms diffusion-based models like IDM-VTON and StableVITON by consistently capturing finer patterns and more accurate garment textures. In contrast, GAN-based methods struggle with complex patterns, resulting in lower-quality outputs on this dataset. 

## **4.4. User Study** 

To complement objective metrics, we conducted a user study to evaluate the visual appeal of our model. The study used 10 image pairs from the VITON-HD dataset, divided into two groups: 5 pairs focused on text and logo clarity, and 5 pairs evaluated pattern and texture fidelity. As shown in Fig. 5, with 50 valid responses, TED-VITON was the preferred model, demonstrating strong user preference for text clarity and pattern accuracy. 

## **4.5. Ablation Study** 

**Effect of key components of TED-VITON.** To analyze the contributions of each key component in TED-VITON, we conduct an ablation study by systematically removing individual components, such as DINOv2, the GS-Adapter, DiTGarmentNet, or Text Preservation Loss, and evaluate their impact on the results. In Fig. 4(a), replacing DINOv2 with a standard CLIP encoder results in blurry and distorted text. This highlights DINOv2’s role, in conjunction with the GSAdapter, in enhancing text clarity and garment alignment by capturing fine semantic and garment details. Fig. 4(b) demonstrates the impact of removing the GS-Adapter, leading to misaligned garment features and reinforcing its importance for detailed garment representation. As shown in Fig. 4(c), removing DiT-GarmentNet compromises fine gar- 

**==> picture [237 x 166] intentionally omitted <==**

Figure 6. Detailed GPT-generated captions enhance the accuracy of garment-specific details, including color, text, and texture. 

|**Detailed Captions**|**LPIPS↓**|**SSIM↑**|**CLIP-I↑**|**FID↓UN**|**KID↓UN**|
|---|---|---|---|---|---|
|✗|0.113|0.872|0.829|9.881|1.706|
|✓|**0.095**|**0.881**|**0.878**|**8.848**|**0.858**|



Table 3. Quantitative comparison of models on VITON-HD with and without GPT-generated captions. “UN” indicated the unpaired setting. KID score is multiplied by 100. 

ment details, like textures and logo placement, indicating its role in preserving intricate design elements. In Fig. 4(d), without Text Preservation Loss, text appears slightly distorted, emphasizing this loss function’s role in maintaining text fidelity. As shown in Fig. 4(e), the full model achieves optimal performance by incorporating all components, accurately capturing both structural and stylistic details. Quantitative evaluation in Table 2 further supports these observations. 

**Effect of using GPT-generated captions.** We conducted an ablation study to evaluate the effect of detailed GPTgenerated captions on TED-VITON’s performance. As shown in Fig. 6, utilizing GPT-generated captions significantly improves the model’s ability to render multi-line text and garment details like color accuracy and texture. Using a brief description, the model accurately renders the first line, “SUN”, but fails to capture “SAND” and “SURF” due to insufficient contextual guidance. Using a detailed caption provides the necessary guidance for accurately rendering all lines and maintaining consistent color and texture. Such quantitative improvements across all metrics are demonstrated in Table 3. 

## **5. Conclusion** 

We presented TED-VITON, a novel VTO framework built on the DiT architecture to tackle critical challenges in garment detail fidelity and text clarity. By incorporating a GS-Adapter and a Text Preservation Loss, TED-VITON significantly improves garment-specific feature representa- 

8 

tion and ensures distortion-free rendering of logos and text. Additionally, a constraint mechanism for LLM-generated prompts enhances training inputs, leading to superior performance. Comprehensive evaluations on the VITON-HD [4] and DressCode [29] datasets showcase state-of-the-art results in visual quality, garment alignment, and text fidelity, establishing TED-VITON as a scalable and highquality solution for next-generation VTO applications. 

## **References** 

- [1] Badour Albahar, Jingwan Lu, Jimei Yang, Zhixin Shu, Eli Shechtman, and Jia-Bin Huang. Pose with style: detailpreserving pose-guided image synthesis with conditional StyleGAN. _ACM Transactions on Graphics_ , 40(6):218:1– 218:11, 2021. 

- [2] Shuai Bai, Huiling Zhou, Zhikang Li, Chang Zhou, and Hongxia Yang. Single Stage Virtual Try-on via Deformable Attention Flows, 2022. arXiv:2207.09161 [cs]. 

- [3] Bharat Lal Bhatnagar, Garvita Tiwari, Christian Theobalt, and Gerard Pons-Moll. Multi-Garment Net: Learning to Dress 3D People From Images. pages 5420–5430, 2019. 

- [4] Seunghwan Choi, Sunghyun Park, Minsoo Lee, and Jaegul Choo. VITON-HD: High-Resolution Virtual Try-On via Misalignment-Aware Normalization. pages 14131–14140, 2021. 

- [5] Yisol Choi, Sangkyung Kwak, Kyungmin Lee, Hyungwon Choi, and Jinwoo Shin. Improving Diffusion Models for Virtual Try-on, 2024. arXiv:2403.05139 [cs]. 

- [6] Aiyu Cui, Jay Mahajan, Viraj Shah, Preeti Gomathinayagam, Chang Liu, and Svetlana Lazebnik. Street TryOn: Learning In-the-Wild Virtual Try-On from Unpaired Person Images. pages 8235–8239, 2024. 

- [7] Haoye Dong, Xiaodan Liang, Xiaohui Shen, Bowen Wu, Bing-Cheng Chen, and Jian Yin. FW-GAN: Flow-Navigated Warping GAN for Video Virtual Try-On. pages 1161–1170, 2019. 

- [8] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, and Robin Rombach. Scaling Rectified Flow Transformers for High-Resolution Image Synthesis. 2024. 

- [9] Anna Fr¨uhst¨uck, Krishna Kumar Singh, Eli Shechtman, Niloy J. Mitra, Peter Wonka, and Jingwan Lu. InsetGAN for Full-Body Image Generation. pages 7723–7732, 2022. 

- [10] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative Adversarial Nets. In _Advances in Neural Information Processing Systems_ . Curran Associates, Inc., 2014. 

- [11] Junhong Gou, Siyu Sun, Jianfu Zhang, Jianlou Si, Chen Qian, and Liqing Zhang. Taming the Power of Diffusion Models for High-Quality Virtual Try-On with Appearance Flow. In _Proceedings of the 31st ACM International Conference on Multimedia_ , pages 7599–7607, New York, NY, USA, 2023. Association for Computing Machinery. 

- [12] Rıza Alp G¨uler, Natalia Neverova, and Iasonas Kokkinos. DensePose: Dense Human Pose Estimation in the Wild. pages 7297–7306, 2018. 

- [13] Xintong Han, Zuxuan Wu, Zhe Wu, Ruichi Yu, and Larry S. Davis. VITON: An Image-based Virtual Try-on Network, 2018. arXiv:1711.08447 [cs]. 

- [14] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. CLIPScore: A Reference-free Evaluation Metric for Image Captioning. In _Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing_ , pages 7514–7528, Online and Punta Cana, Dominican Republic, 2021. Association for Computational Linguistics. 

- [15] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium. In _Advances in Neural Information Processing Systems_ . Curran Associates, Inc., 2017. 

- [16] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising Diffusion Probabilistic Models. In _Advances in Neural Information Processing Systems_ , pages 6840–6851. Curran Associates, Inc., 2020. 

- [17] Shion Honda. VITON-GAN: Virtual Try-on Image Generator Trained with Adversarial Loss, 2019. Publication Title: arXiv e-prints ADS Bibcode: 2019arXiv191107926H. 

- [18] Junho Kim, Minjae Kim, Hyeonwoo Kang, and Kwanghee Lee. U-GAT-IT: Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation, 2019. Publication Title: arXiv e-prints ADS Bibcode: 2019arXiv190710830K. 

- [19] Jeongho Kim, Gyojung Gu, Minho Park, Sunghyun Park, and Jaegul Choo. StableVITON: Learning Semantic Correspondence with Latent Diffusion Model for Virtual Try-On, 2023. arXiv:2312.01725 [cs]. 

- [20] Robin Kips, Pietro Gori, Matthieu Perrot, and Isabelle Bloch. CA-GAN: Weakly Supervised Color Aware GAN for Controllable Makeup Transfer. In _Computer Vision – ECCV 2020 Workshops_ , pages 280–296, Cham, 2020. Springer International Publishing. 

- [21] Sangyun Lee, Gyojung Gu, Sunghyun Park, Seunghwan Choi, and Jaegul Choo. High-Resolution Virtual TryOn with Misalignment and Occlusion-Handled Conditions, 2022. arXiv:2206.14180 [cs]. 

- [22] Kun Li, Jinsong Zhang, Yebin Liu, Yu-Kun Lai, and Qionghai Dai. PoNA: Pose-Guided Non-Local Attention for Human Pose Transfer. _IEEE Transactions on Image Processing_ , 29:9584–9599, 2020. Conference Name: IEEE Transactions on Image Processing. 

- [23] Nannan Li, Qing Liu, Krishna Kumar Singh, Yilin Wang, Jianming Zhang, Bryan A. Plummer, and Zhe Lin. UniHuman: A Unified Model for Editing Human Images in the Wild, 2023. arXiv:2312.14985 [cs]. 

- [24] Wen Liu, Zhixin Piao, Jie Min, Wenhan Luo, Lin Ma, and Shenghua Gao. Liquid Warping GAN: A Unified Framework for Human Motion Imitation, Appearance Transfer and Novel View Synthesis. pages 5904–5913, 2019. 

9 

- [25] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow, 2022. arXiv:2209.03003. 

- [26] Liqian Ma, Xu Jia, Qianru Sun, Bernt Schiele, Tinne Tuytelaars, and Luc Van Gool. Pose Guided Person Image Generation. In _Advances in Neural Information Processing Systems_ . Curran Associates, Inc., 2017. 

- [27] Yifang Men, Yiming Mao, Yuning Jiang, Wei-Ying Ma, and Zhouhui Lian. Controllable Person Image Synthesis With Attribute-Decomposed GAN. pages 5084–5093, 2020. 

- [28] Matiur Rahman Minar, T. Tuan, Heejune Ahn, Paul L. Rosin, and Yu-Kun Lai. CP-VTON+: Clothing Shape and Texture Preserving Image-Based Virtual Try-On. 2020. 

- [29] Davide Morelli, Matteo Fincato, Marcella Cornia, Federico Landi, Fabio Cesari, and Rita Cucchiara. Dress Code: HighResolution Multi-Category Virtual Try-On. pages 2231– 2235, 2022. 

- [30] Davide Morelli, Alberto Baldrati, Giuseppe Cartella, Marcella Cornia, Marco Bertini, and Rita Cucchiara. LaDIVTON: Latent Diffusion Textual-Inversion Enhanced Virtual Try-On. In _Proceedings of the 31st ACM International Conference on Multimedia_ , pages 8580–8589, New York, NY, USA, 2023. Association for Computing Machinery. 

- [31] Shuliang Ning, Duomin Wang, Yipeng Qin, Zirong Jin, Baoyuan Wang, and Xiaoguang Han. PICTURE: PhotorealistIC virtual Try-on from UnconstRained dEsigns. pages 6976–6985, 2024. 

- [32] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Herv´e Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. Dinov2: Learning robust visual features without supervision, 2023. 

- [33] Sonia Pecenakova, Nour Karessli, and Reza Shirvany. FitGAN: Fit- and Shape-Realistic Generative Adversarial Networks for Fashion. In _2022 26th International Conference on Pattern Recognition (ICPR)_ , pages 3097–3104, 2022. ISSN: 2831-7475. 

- [34] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis, 2023. arXiv:2307.01952 [cs]. 

- [35] Amir Hossein Raffiee and Michael Sollami. GarmentGAN: Photo-realistic Adversarial Fashion Transfer. In _2020 25th International Conference on Pattern Recognition (ICPR)_ , pages 3923–3930, 2021. ISSN: 1051-4651. 

- [36] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-Resolution Image Synthesis With Latent Diffusion Models. pages 10684– 10695, 2022. 

- [37] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation, 2023. arXiv:2208.12242. 

- [38] Sang-Heon Shim, Jiwoo Chung, and Jae-Pil Heo. Towards Squeezing-Averse Virtual Try-On via Sequential Deformation. _Proceedings of the AAAI Conference on Artificial Intelligence_ , 38(5):4856–4863, 2024. Number: 5. 

- [39] Siqi Wan, Yehao Li, Jingwen Chen, Yingwei Pan, Ting Yao, Yang Cao, and Tao Mei. Improving Virtual Try-On with Garment-focused Diffusion Models, 2024. 

- [40] Bochao Wang, Huabin Zheng, Xiaodan Liang, Yimin Chen, Liang Lin, and Meng Yang. Toward CharacteristicPreserving Image-based Virtual Try-On Network. pages 589–604, 2018. 

- [41] Xintao Wang, Liangbin Xie, Chao Dong, and Ying Shan. Real-ESRGAN: Training Real-World Blind SuperResolution With Pure Synthetic Data. pages 1905–1914, 2021. 

- [42] Zhou Wang, A.C. Bovik, H.R. Sheikh, and E.P. Simoncelli. Image quality assessment: from error visibility to structural similarity. _IEEE Transactions on Image Processing_ , 13(4): 600–612, 2004. Conference Name: IEEE Transactions on Image Processing. 

- [43] Zhenyu Xie, Zaiyu Huang, Xin Dong, Fuwei Zhao, Haoye Dong, Xijin Zhang, Feida Zhu, and Xiaodan Liang. GPVTON: Towards General Purpose Virtual Try-On via Collaborative Local-Flow Global-Parsing Learning. pages 23550– 23559, 2023. 

- [44] Han Yang, Ruimao Zhang, Xiaobao Guo, Wei Liu, Wangmeng Zuo, and Ping Luo. Towards PhotoRealistic Virtual Try-On by Adaptively Generating$ _\_ leftrightarrow$Preserving Image Content, 2020. arXiv:2003.05863 [cs, eess]. 

- [45] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. IPAdapter: Text Compatible Image Prompt Adapter for Textto-Image Diffusion Models, 2023. arXiv:2308.06721 [cs]. 

- [46] Ruiyun Yu, Xiaoqi Wang, and Xiaohui Xie. VTNFP: An Image-Based Virtual Try-On Network With Body and Clothing Feature Preservation. pages 10511–10520, 2019. 

- [47] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding Conditional Control to Text-to-Image Diffusion Models. pages 3836–3847, 2023. 

- [48] Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. pages 586–595, 2018. 

- [49] Xinyue Zhou, Mingyu Yin, Xinyuan Chen, Li Sun, Changxin Gao, and Qingli Li. Cross Attention Based Style Distribution for Controllable Person Image Synthesis. In _Computer Vision – ECCV 2022_ , pages 161–178, Cham, 2022. Springer Nature Switzerland. 

- [50] Luyang Zhu, Dawei Yang, Tyler Zhu, Fitsum Reda, William Chan, Chitwan Saharia, Mohammad Norouzi, and Ira Kemelmacher-Shlizerman. TryOnDiffusion: A Tale of Two UNets. pages 4606–4615, 2023. 

- [51] Zhen Zhu, Tengteng Huang, Baoguang Shi, Miao Yu, Bofei Wang, and Xiang Bai. Progressive Pose Attention Transfer for Person Image Generation. pages 2347–2356, 2019. 

10 

