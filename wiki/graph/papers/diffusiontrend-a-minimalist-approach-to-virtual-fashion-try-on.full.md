---
type: paper-fulltext
slug: diffusiontrend-a-minimalist-approach-to-virtual-fashion-try-on
arxiv_id: null
source_path: /Users/saris.kia.adm/.paper-scholar/diffusiontrend-a-minimalist-approach-to-virtual-fashion-try-on/2412.14465.md
paper: "[[diffusiontrend-a-minimalist-approach-to-virtual-fashion-try-on]]"
---
<!-- extracted by afk_extract from 2412.14465.pdf (15p) -->

JOURNAL OF L[A] TEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 

1 

## DiffusionTrend: A Minimalist Approach to Virtual Fashion Try-On 

Wengyi Zhan, Mingbao Lin, Shuicheng Yan, _Fellow, IEEE_ , Rongrong Ji, _Senior Member, IEEE_ 

_**Abstract**_ **—We introduce DiffusionTrend for virtual fashion try-on, which forgoes the need for retraining diffusion models. Using advanced diffusion models, DiffusionTrend harnesses latent information rich in prior information to capture the nuances of garment details. Throughout the diffusion denoising process, these details are seamlessly integrated into the model image generation, expertly directed by a precise garment mask crafted by a lightweight and compact CNN. Although our DiffusionTrend model initially demonstrates suboptimal metric performance, our exploratory approach offers some important advantages: (1) It circumvents resource-intensive retraining of diffusion models on large datasets. (2) It eliminates the necessity for various complex and user-unfriendly model inputs. (3) It delivers a visually compelling try-on experience, underscoring the potential of training-free diffusion model. This initial foray into the application of untrained diffusion models in virtual try-on technology potentially paves the way for further exploration and refinement in this industrially and academically valuable field.** 

_**Index Terms**_ **—Diffusion Model, Virtual Try-On, Image Editing** 

## I. INTRODUCTION 

IRTUAL try-on technology [1], [2], which digitally su- **V** perimposes images of models wearing various outfits, represents a pivotal innovation in the fashion sector. This advancement offers consumers an immersive and interactive experience with garments, allowing them to preview how clothing might look on them without physically visiting stores. For retailers, virtual try-on technology streamlines their operations by obviating the need to employ live models for merchandise display. It also circumvents the financial burdens associated with traditional product photography, leading to a marked improvement in operational efficiency. For commercial platforms, virtual try-on technology serves as a magnet for attracting a larger user base and fostering user loyalty. Collectively, virtual try-on is reshaping the landscape of the apparel retail industry, propelling the fashion sector toward a future characterized by heightened efficiency and customization. 

Traditional virtual try-on solutions [11]–[16] are predicated on a two-stage pipeline utilizing Generative Adversarial 

Manuscript received Feb. XX, 2025. (Corresponding author: Rongrong Ji) This work was supported by National Science and Technology Major Project (No. 2022ZD0118202), the National Science Fund for Distinguished Young Scholars (No.62025603), the National Natural Science Foundation of China (No. U21B2037, No. U22B2051, No. U23A20383, No. 62176222, No. 62176223, No. 62176226, No. 62072386, No. 62072387, No. 62072389, No. 62002305 and No. 62272401), and the Natural Science Foundation of Fujian Province of China (No. 2021J06003, No.2022J06001). 

W. Zhan and R. Ji are with the Key Laboratory of Multimedia Trusted Perception and Efficient Computing, Ministry of Education of China, Xiamen University, China. E-mail: zhanwy@stu.xmu.edu.cn, rrji@xmu.edu.cn 

M. Lin and S. Yan are with the Skywork AI, Singapore 118222. E-mail: linmb001@outlook.com, Shuicheng.yan@gmail.com 

TABLE I 

COMPARISON OF INPUT REQUIREMENTS FOR PREVIOUS VIRTUAL TRY-ON MODELS. A CHECKMARK (�) INDICATES THAT THE INPUT MODALITY IS REQUIRED, WHILE A DASH (-) INDICATES THAT IT IS NOT OR NOT 

MENTIONED. 

|Method|Clothes Mask<br>Densepose<br>Segment Map<br>Clothes-Agnostic<br>Keypoint|
|---|---|
|TryOnDiffusion [3]<br>DCI-VTON [4]<br>LaDI-VTON [5]<br>WarpDiffusion [6]<br>OOTDiffusion [7]<br>StableVITON [8]<br>IDM-VTON [9]<br>Wear-Any-Way [10]<br>DiffusionTrend (Ours)|-<br>-<br>�<br>�<br>�<br>�<br>�<br>�<br>�<br>-<br>�<br>�<br>-<br>�<br>-<br>�<br>-<br>-<br>�<br>-<br>�<br>-<br>�<br>-<br>�<br>�<br>�<br>-<br>�<br>-<br>�<br>�<br>-<br>�<br>-<br>�<br>-<br>-<br>�<br>�<br>�<br>-<br>-<br>-<br>-|



Networks (GANs) [17]. The initial stage in this framework involves the application of an explicit warp module [18]– [22] to deform the clothing to the desired area on the body. The subsequent stage integrates the deformed clothing using a GAN-based try-on generator. To attain accurate clothing deformation, earlier methodologies [12]–[14], [23]–[26] have employed a trainable network designed to estimate a dense flow map [27]–[29], thereby facilitating the mapping of the clothing onto the human form. In parallel, various approaches [11]– [14], [30]–[33], have been proposed to address the misalignment issues between the warped clothing and the human body. Techniques such as normalization [11] and distillation [13], [31] have been implemented to mitigate these discrepancies. Recent advancements in diffusion models [34] have led to a notable enhancement in the quality of image synthesis tasks, with a particular emphasis on the domain of virtual try-on. In this context, contemporary research endeavors have leveraged pre-trained text-to-image diffusion models to produce highfidelity results. The TryOnDiffusion model [3], for instance, employs a dual U-Net architecture to perform the try-on task. LADI-VTON [5] and DCI-VTON [4] either conceptualize clothing items as pseudo-words or integrate garments through the use of warping networks into pre-trained diffusion models. 

Despite these advancements, a review of existing studies reveals that the high precision and realism achieved by current methods require training on extensive try-on datasets [35], particularly in diffusion-based approaches [3]–[5]. Diffusion models simulate the data generation process by iteratively modeling the diffusion and reverse diffusion steps. Each iteration involves complex probability distribution calculations and requires substantial computational resources, leading to considerable overall training costs. In the words, current methods rely on resource-intensive network training. While intensive training is well-suited for generating complex poses 

JOURNAL OF L[A] TEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 

2 

**==> picture [504 x 108] intentionally omitted <==**

Fig. 1. Images generated by the proposed DiffusionTrend model, given an input target model and a try-on clothing item both from DressCode dataset [35]. 

and capturing fine image details, the substantial computational costs can make it an unattractive option. Besides, as shown in Table I, these methods often require multiple auxiliary modules or APIs as model inputs, such as densepose [36], segmentation maps [37], clothes-agnostic images/masks/representations [38] and keypoints [39]. The integration of these tools not only increases inference overhead but also suffers from insufficient encapsulation in some cases, leaving non-professional users to deal with complex configurations and unstandardized interfaces, which significantly hampers accessibility and user experience. There is a pressing need for the community to explore more accessible and less resource-intensive methods, with the effort of less compromise on the quality of image generation— a domain that, to our knowledge, remains uncharted to date. 

In marked contrast, our proposed DiffusionTrend offers a streamlined, lightweight training approach that circumvents the need to retrain a diffusion model, thus liberating it from the reliance on expensive and resource-intensive computational infrastructure. This technique reduces inference overhead and simplifies the workflow by dispensing with the requirement for intricate segmentation, pose extraction, and other preliminary processing steps for the input images, leading to a more accessible and economical solution. Specifically, a lightweight, compact CNN is utilized to outline the clothing in both the model and garment images. It harmonizes image and textual features and conducts clustering at the feature level to produce effective masks. Subsequently, we make full use of the latents derived from DDIM inversion [40], which are replete with prior information and act as superior conduits for the detailed features of the garment. During the early stages of the diffusion denoising process, the seamless integration of the target garment into the model’s image reconstruction is achieved by blending the latent representations of both the model and the garment. In the subsequent stages of denoising, leveraging the self-repairing capabilities of the pre-trained latent diffusion model, we maintain the model’s identity and background coherence by selectively replacing the latents in the background areas. This approach ensures that the garment is harmoniously merged into the overall image. To further enhance robustness under diverse generation conditions, we apply an adaptive sampling termination strategy based on perceptual color difference to automatically determine the optimal number of sampling steps during inference. Fig. 1 shows several examples of our proposed DiffusionTrend. 

Despite its suboptimal metric performance, our initial ex- 

ploration with DiffusionTrend has yielded several valuable contributions that could indeed pave the way for further research and refinement in the realm of untrained diffusion models for virtual try-on technology: 

- It eliminates the necessity for resource-intensive training of diffusion models on extensive datasets, thereby reducing the computational demands. 

- It removes the need for cumbersome and user-unfriendly model inputs, reducing computational costs and streaming the pipeline of virtual try-on. 

- It delivers a visually appealing virtual try-on experience, highlighting the potential of diffusion models that do not require training for future research. 

These contributions underscore the significance of our DiffusionTrend model as a foundation for future advancements in virtual try-on technology, emphasizing the potential of exploring a training-free approach. 

## II. RELATED WORK 

## _A. Image Editing through Diffusion Processes_ 

The practice of integrating specific content into a base image to produce realistic composites is prevalent in image editing leveraging diffusion processes. Initially, the field was dominated by text-based models for image editing [41], [42]. InstructPix2Pix [41] employs paired data to train diffusion models capable of generating an edited image from an input image and a textual instruction. Conversely, Imagic [42] harnesses a pre-trained text-to-image diffusion model to generate text embeddings that align with both the input image and the target text. The abstract nature of text poses a limitation in accurately delineating the subtleties of objects, therefore, image conditioning was introduced to offer more concrete and precise descriptions. DCFF [43] pioneers the use of pyramid filters for image composition, which was subsequently advanced by Paint by Example [44], employing CLIP embeddings of the reference image to condition the diffusion model. Most contemporary methodologies, such as Dreambooth [45] (all model parameters), Textual Inversion [46] (a word vector for novel concepts), and Custom-Diffusion [47] (cross-attention parameters), rely heavily on fine-tuning techniques. In contrast, a handful of approaches [48], [49] adopts a trainingfree paradigm. Prompt-to-prompt [48] modifies the input text prompt to steer the cross-attention mechanism for nuanced image editing, while MasaCtrl [49] employs a mask-guided 

JOURNAL OF L[A] TEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 

3 

differential equation [50] through incremental steps, ensuring a controlled transition from initial state _z_ 0 to final noise _zT_ : 

mutual attention strategy for non-rigid image synthesis and editing. These training-free methods offer a cost-effective alternative, eliminating extensive training while still delivering commendable generative outcomes. 

**==> picture [243 x 46] intentionally omitted <==**

## _B. Virtual Try-on with Diffusion Models_ 

Diffusion models have demonstrated remarkable efficacy in the domain of image editing, with image-based virtual try-on representing a specialized subset of these tasks, contingent upon a specific garment image. Adapting text-toimage diffusion models to accommodate images as conditions, is straightforward, but the spatial discrepancies between the garment and the subject’s pose challenge the fidelity of texture details in the virtual try-on outcomes [4]–[6]. Methodologies such as WarpDiffusion [6], DCI-VTON [4], and LADIVTON [5] conceptualize clothing as pseudo-words, employing warping techniques through CNN networks to adjust clothing to various poses, yielding satisfactory results. TryOnDiffusion [3] employs a dual U-Net architecture for the virtual try-on task, implicitly conducting garment warping through the interplay between cross-attention layers. This approach effectively resolves the issue of texture misalignment without the need for a dedicated warp module. Similarly, the StableVITON [8] incorporates zero cross-attention blocks to condition the intermediate feature maps of a spatial encoder, thereby circumventing the requirement for a warp module. The Wear-Any-Way [10] enhances the process of virtual garment alteration, providing more adaptable control over the manner in which clothing is depicted. The IDM-VTON [9] enhances the virtual try-on process by integrating attention mechanisms and high-level semantic encoding into the diffusion model. 

We start with the noisy latent state _zT[∗]_[and][proceed][with] denoising as outlined in Eq. (2). This method approximates _z_ 0 _[∗]_[,] which closely resembles the original latent representation _z_ 0. Our goal is to incorporate information from a garment image _I[g]_ into the reconstruction of the model image _I[m]_ , depicting the model wearing the garment from _I[g]_ . 

To address the challenge of extensive training overhead and various model inputs, we choose not to alter any weights or structures of the pre-trained diffusion model. Instead, we introduce a lite-training visual try-on method called “DiffusionTrend”. Our methodology consists of three stages. 1) A lightweight and compact CNN accurately delineates the apparel in both the model and garment images. 2) At an appropriate point in the process, garment details are integrated into the reconstruction phase of the model image. 3) To ensure the coherence of the generated background with the model, we use a latent substitution technique to restore the background, leveraging the diffusion model’s restorative properties to blend it seamlessly with the newly rendered apparel. A comprehensive discussion of the first stage is in Sec. III-B, while the latter two stages are discussed in Sec. III-C. 

## _B. Precise Apparel Localization_ 

In conventional GAN-based [12]–[14] or current Diffusionbased [3]–[5] try-on methods, a precise clothing mask is crucial. This mask ensures correct apparel placement on the model and accurate extraction of garment features while avoiding interference from non-garment regions. 

## III. METHODOLOGY 

## _A. Preliminaries_ 

**Latent Diffusion Models.** Latent diffusion models (LDMs) use an encoder _E_ to convert an input image _x_ 0 _∈_ R _[H][×][W][ ×]_[3] into a lower-dimensional _z_ 0 = _E_ ( _x_ 0) _∈_ R _[h][×][w][×][c]_ . Here, the downsampling ratio is _f_ = _H/h_ = _W/w_ , and _c_ is the channel number. The forward diffusion process is: 

Traditional segmentation models [51], [52] excel in mask generation tasks. However, after thorough evaluation, we have chosen not to employ pre-built models like Segment Anything Model (SAM) [51]. Instead, we propose a lightweight CNN tailored for our application. This decision is informed by the following considerations: 

**==> picture [210 x 12] intentionally omitted <==**

where _{αt}[T] t_ =1[denotes][variance][schedules,][with] _α_ ¯ _t_ = � _ti_ =1 _[α][i]_[.][A][U-Net] _[ϵ][θ]_[refines][noise][estimation.][This][is][crucial] for reconstructing the latent representation _z_ 0 from the initial noisy state _zT ∼N_ (0 _,_ **I** ): 

**==> picture [259 x 135] intentionally omitted <==**

**----- Start of picture text -----**<br>
Parameters (M) Computational Memory Inference<br>Cost (GFLOPs) Usage (GB) Speed (ms/img)<br>372 2.7 1383<br>93.74<br>685<br>1.41 2 0.15<br>SAM Ours SAM Ours SAM Ours SAM Ours<br>**----- End of picture text -----**<br>


**==> picture [262 x 40] intentionally omitted <==**

The text encoder _τθ_ ( _P_ ) converts text prompt _P_ into an embedding that is integrated with the U-Net’s intermediate noise representation using cross-attention mechanisms. At time step _t_ = 0, the decoder _D_ transforms the latent space representation _z_ 0 back into the original image domain _x_ 0 = _D_ ( _z_ 0). 

**DDIM Inversion.** DDIM inversion uses DDIM sampling [40] to ensure deterministic sampling by setting the variance in Eq. (2). It assumes the reversibility of the ordinary 

Fig. 2. Inference cost comparison: Segment Anything Model (SAM) [51] _vs_ . our Apparel Localization Network. 

JOURNAL OF L[A] TEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 

4 

**==> picture [516 x 164] intentionally omitted <==**

**----- Start of picture text -----**<br>
Masks from  Apparel Localization Network  Reference  DiffusionTrend Reference  DiffusionTrend<br>Attention-Map (Ours) U-Net (Ours) U-Net (Ours)<br>(a) (b)<br>**----- End of picture text -----**<br>


Fig. 3. Visual comparison of mask generation and try-on results. (a) Precision of apparel localization masks generated by our network compared to masks from attention maps. (b) Quality comparison of generation results: Reference U-Net method _vs_ . DiffusionTrend. 

**Resource Efficiency in Constrained Environment:** While models like SAM offer outstanding performance, they come with a significant computational overhead, particularly in terms of memory usage and processing time. As shown in Fig. 2, SAM requires 2.7 GB of GPU memory and 372 GFLOPs to process an image of size 768 _×_ 1024. In contrast our lightweight CNN requires a mere 0.15 GB of memory and 2 GFLOPs, making a substantial reduction in resource usage. In practical virtual try-on applications, especially in online multi-user settings, servers are tasked with managing a high volume of concurrent requests. In such scenarios, efficient resource use is more critical than slight gains in segmentation precision. Our lightweight CNN not only minimizes operational expenses but also significantly improves response time. 

**Task-Specific Optimization Over General-Purpose Segmentation:** Off-the-shelf models like SAM are engineered for a wide range of semantic segmentation tasks and lack customization for the nuances of virtual try-on applications. Our proposed lightweight CNN, on the other hand, incorporates task-specific textual cues ( _e.g._ , “top”, “skirt”) to enable precise segmentation of clothing areas, sidestepping irrelevant regions. We do not adopt the CLIP [53] image encoder, as it is primarily optimized for global image-text alignment and lacks the spatial resolution required for pixel-accurate apparel segmentation. Moreover, it introduces additional inference cost, which contradicts our lightweight design principle. 

**Balancing Computational Load Across the Pipeline:** The diffusion model at the core of our system is inherently resource-intensive. Incorporating a high-cost segmentation model like SAM would further strain computational resources, contracting our aim of creating a lightweight and cost-effective virtual try-on solution. Our lightweight CNN not only harmonizes with diffusion, but also supports our overarching goal of developing an economical segmentation network for this specific domain. Using a generic segmentation model would compromise this objective and diminish the distinctive value of our research. 

In addition, we have attempted to extract masks using the intermediate attention map of a diffusion U-Net. During the inversion process, we utilize the concept of “clothes” 

to interact with image features via cross-attention. However, the extracted masks, we find, fail to meet the precision requirements for virtual try-on tasks. As shown in Fig. 3(a), the attention map roughly identifies the garment region but often includes extraneous parts. 

Therefore, in the consideration of balancing accuracy and saving computational resources, we have engineered a compact CNN in Fig. 4(a) for precise apparel localization. By combining textual and visual features, we minimize manual intervention. Users only need to specify the target category ( _e.g._ , upper garments, lower garments, or dresses), and the model automatically generates precise mask outputs. Our network accepts a model image _I_ 0 as input and processes it through three 3 _×_ 3 _Convi_[3] _[×]_[3] layers with _ReLU_ activation, culminating in the image features _I_ 3, as: 

**==> picture [217 x 14] intentionally omitted <==**

An apparel-related prompt _P_ is transformed into a text embedding _T_ 0 = _Clip_ ( _P_ ) by a _Clip_ [53] text encoder with fixed parameters. The text embedding _T_ 0 is then advanced through two _FCi_ layers with a _ReLU_ function in between, to produce the text-derived features _T_ 2 as: 

**==> picture [192 x 19] intentionally omitted <==**

Next, we amalgamate the image feature _I_ 3 with the text feature _T_ 2 using a 1 _×_ 1 _Conv_[1] _[×]_[1] layer, followed by _Sigmoid_ and _Upsample_ functions as: 

**==> picture [229 x 19] intentionally omitted <==**

For the training, we compute the _ℓ_ 1 norm between _M_ and ground-truth mask _MGT_ , employing this as the loss function to refine the network’s parameters. Note that the predicted _M_ serves as a mask for all clothing items on the model, which may encompass both upper and lower garments. 

To delineate between the upper and lower garments and extract their respective masks, we perform K-means clustering upon the masked features _M · I_ 3. During the clustering process, the cluster with the smallest mean feature value is typically associated with the background and is excluded from 

5 

JOURNAL OF L[A] TEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 

**==> picture [490 x 287] intentionally omitted <==**

**----- Start of picture text -----**<br>
⊗<br>P<br>Cluster<br>Precise Apparel<br>Localization<br>(a) Training (b) Inference<br>𝑖𝑛𝑣𝑒𝑟𝑠𝑖𝑜𝑛<br>z<br>Garment Latent  z<br>z<br>Infusion<br>Reshape<br>Reshape<br>𝑖𝑛𝑣𝑒𝑟𝑠𝑖𝑜𝑛 Sampling<br>Termination<br>× (𝑇−𝑡1) × (𝑡1 −𝑡2) × 𝑡2<br>𝑡= 𝑇 𝑡= 𝑡1 𝑡= 𝑡2<br>(c) Vitrual Try-On Image Reconstruction Background Restoration<br>11 ×<br>+ReLU1  +ReLU2  +ReLU3<br>Conv Sigmoid Upsample<br>Conv Conv Conv<br>CLIP +ReLU1 FC2<br>Precise Apparel Localization<br>FC<br>**----- End of picture text -----**<br>


Fig. 4. DiffusionTrend framework. (a) A lightweight apparel localization network to predict precise garment masks. (b) The network inference. (c) The reconstruction of the virtual try-on image. 

further consideration. The remaining clusters are then analyzed based on the vertical positions of their centroids, enabling effective separation of top and bottom garment masks: 

**==> picture [198 x 12] intentionally omitted <==**

It should be noted that for models adorned in addresses or when processing a garment image, _M_ is utilized directly as the mask, thereby obviating the need for clustering. 

To further enhance the precision and boundary regularity of the K-means-generated masks, we apply a classic active contour-based [54] refinement method as a post-processing step. This technique iteratively optimizes the mask boundary by minimizing an energy function that balances edge attraction and contour smoothness, producing more natural and connected garment regions. A comparative study of garment mask refinement strategies is presented in Sec. IV-C. 

Our apparel localization network is notably lightweight, making the training process highly cost-effective when compared to methods [4], [6], [8] that require training a diffusion model. It takes us only 20 hours to process the entire DressCode dataset [35] using two RTX 3090 GPUs. As shown in Fig. 2, our proposed localization network operates fully automatically with only 2.00 GFLOPs and 0.15G of memory, and its accuracy analysis can be found in Sec. IV-C. 

## _C. Virtual Try-On Image Reconstruction_ 

Herein, we delve into the process of integrating garment details into the reconstruction phase of the model image. As illustrated in Fig. 4(c), our virtual try-on image reconstruction 

process encompasses two critical stages: the infusion of garment latents and the restoration of the background. 

**Garment Latent Infusion.** Recent studies [7], [9] indicate that employing U-Net for feature extraction necessitates treating it as a high-parameter module, which entails considerable training expenses. This approach conflicts with our research goal of achieving a training-free diffusion model solution, as such high computational costs undermine the feasibility of lightweight virtual try-on applications. 

We first experiment with cross-attention-based MasaCtrl [49], which involves a reference U-Net extracting garment features and performing key-value exchanges with the main U-Net during the attention stage. Although this method avoids additional training, it heavily relies on dual-branch U- Nets and attempts to guide attention interactions with masks derived from our apparel localization network. As shown in Fig. 3(b), the results fall short in accurately rendering fine-grained garment details. We surmise that direct feature injection through cross-attention, while effective at generating semantically coherent content, lacks the ability to capture the intricate details essential for high-quality virtual try-on results. 

Given these challenges, we have been iteratively refining our methodology. After extensive exploration, we develop the approach illustrated in Fig. 4(c), which achieves a precise and efficient fusion of garment details with model images while maintaining the lightweight and training-free objectives. 

Given a model image _I[m]_ and a garment image _I[g]_ , we utilize our apparel localization network to acquire the masks _M[m]_ and _M[g]_ . A natural thought is to leverage an explicit 

6 

JOURNAL OF L[A] TEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 

**==> picture [516 x 116] intentionally omitted <==**

**----- Start of picture text -----**<br>
w/o  w/o Background w/o  w/o Background<br>Restoration Restoration<br>**----- End of picture text -----**<br>


Fig. 5. Ablation studies on _M[bg]_ and background restoration are presented. Omitting _M[bg]_ results in inaccuracies such as the reconstruction of long skirts as short skirts and long sleeves as short sleeves. Without background restoration, various issues arise, including altered background colors, distorted facial features, and unintended changes to other parts of the model’s attire. 

warp module to align the in-store garment _I[g]_ with the pose, position, and content of the clothing in _I[m]_ . Initially, we adopted three representative methods, including the warping modules from DCI-VTON [4], SCW-VTON [55], and PLVTON [56], to preprocess garment images. However, the results were unsatisfactory. In Fig. 10, the warped garment images introduce severe distortions into the inversion and reconstruction phases. This, in turn, leads to highly distorted virtual try-on results, significantly degrading the overall quality of the pipeline. 

We attribute this to incompatibility between the independent warp module and the untrained U-Net in our training-free paradigm, with detailed analysis provided in Sec. IV-C. Consequently, we opt for perspective transformations to address the alignment problem. By computing the bounding boxes _B[m]_ and _B[g]_ for _M[m]_ and _M[g]_ , respectively, we reshape the in-store garment image _I[g]_ to align with the position and size of the clothing in the model image _I[m]_ . 

The reshaping process starts by resizing the bounding box _B[g]_ to match _B[m]_ , achieving initial positional alignment. However, this direct resizing may lead to distortions in the garment’s proportions, affecting its style and fit ( _e.g._ , changes in pant length or skirt length). To address this, we analyze the inherent features of the original garment, such as shoulder width, waistband, or hemline position. These features help estimate appropriate scaling factors for different garment types, ensuring that key attributes ( _e.g._ , shirt length, pant length, or dress silhouette) are preserved. Using these scaling factors, we apply proportional stretching adjustments to restore the garment’s original design and fit. 

Following the reshaping step, we apply appropriate perspective transformations to simulate simple rotational and deformational effects. For instance, the left side of the garment can be rotated inward to mimic the perspective of a model turning to the left. These transformations simulate basic pose variations, addressing alignment and deformation challenges. 

Once aligned, the model image _I[m]_ and the adjusted garment image _I[g]_ are transformed into their latent representations _z_ 0 _[m]_[=] _[E]_[(] _[I][m]_[)][and] _[z]_ 0 _[g]_[=] _[E]_[(] _[I][g]_[)][,][where] _[E]_[denotes][the][encoder.] These latents are then subjected to the DDIM Inversion process in Eq. (3), yielding noisy latent sets _{zt[∗][m] }[T] t_ =0[and] _[{][z] t[∗][g][}][T] t_ =0[.] Starting from _zT[∗][m]_[,][we][reconstruct][the][model][image][using] Eq. (2) where the latent at the _t_ -th time step is _zt[m]_[.][Our][goal] 

is to decode an image of the model wearing the garment from image _I[g]_ , thereby achieving a harmonious fusion of the model’s appearance and the desired attire. A literature review [57] has shown that latents derived from the inversion process retains rich prior information. This property makes it ideal for capturing the intricate features of the target garment. An effective strategy is to integrate the garment latent _zt[∗]_ 1 _[g]_[,] preserved during the inversion stage, into the masked region _M[g]_ at time step _t_ 1. This occurs early in the denoising process as: 

**==> picture [197 x 12] intentionally omitted <==**

The denoising continues with the infused latent. It ensures the seamless information transferred from the garment image to the model image, and achieves a precise attire on the model. We choose latent-space fusion over image-space masking, as it enables progressive integration of garment details and avoids visual artifacts caused by hard region cuts. 

While perspective transformations cannot simulate the complex deformations in real try-on scenarios, they offer a simple and lightweight solution for basic alignment, which aligns with our goal of a training-free virtual try-on framework and lays the groundwork for future advancements in garment alignment. 

**Background Restoration.** While the garment latent infusion yields significant results, it negatively impacts the generation of background content in subsequent denoising steps. In Fig. 5, issues such as altered background color, distorted facial features, and unintended changes to other parts of the model’s attire occur. We speculate that this may be due to the prompt “a model wearing clothes” used during the generation process, which guides the reconstruction to focus on maintaining the garment’s structure, while the background, lacking specific guidance, becomes more susceptible to distortion. To address this, we must implement strategies to restore the background and preserve the model’s identity and background information. 

Motivated by this, we inject the model latent _zt[m]_ 2[into][the] regions outside the model clothing mask _M[m]_ at time step _t_ 2, a later stage in the diffusion denoising process focused on generating detailed information. Leveraging the diffusion model’s inherent repair capability, a few subsequent denoising steps integrate the background latents with the target garment. 

JOURNAL OF L[A] TEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 

7 

TABLE II 

QUANTITATIVE RESULTS ON THE VITON-HD AND DRESSCODE DATASETS. 

|Method|VITON-HD|DressCode|
|---|---|---|
||LPIPS_↓_<br>SSIM_↑_<br>FID_↓_<br>KID_↓_|LPIPS_↓_<br>SSIM_↑_<br>FID_↓_<br>KID_↓_|
||||
|TryOnDiffusion [3]<br>DCI-VTON [4]<br>LaDI-VTON [5]<br>WarpDiffusion [6]<br>OOTDiffusion [7]<br>StableVITON [8]<br>IDM-VTON [9]<br>Wear-Any-Way [10]<br>DiffusionTrend (Ours)|-<br>-<br>13.447<br>6.964<br>0.0530<br>0.8920<br>9.130<br>0.870<br>0.0910<br>0.8760<br>9.410<br>0.160<br>0.0880<br>0.9850<br>8.610<br>-<br>0.0710<br>0.8780<br>8.810<br>0.820<br>0.0732<br>0.8880<br>8.233<br>0.490<br>0.1020<br>0.8700<br>6.290<br>-<br>0.0780<br>0.8770<br>8.155<br>0.780<br>0.0918<br>0.8592<br>10.433<br>0.540|-<br>-<br>-<br>-<br>0.0443<br>-<br>11.800<br>-<br>0.0640<br>0.9060<br>6.480<br>0.220<br>0.0890<br>0.9010<br>9.187<br>-<br>0.0450<br>0.9270<br>4.200<br>0.370<br>0.0388<br>0.9370<br>9.940<br>0.120<br>0.0620<br>0.9200<br>8.640<br>-<br>0.0409<br>0.9340<br>11.720<br>0.330<br>0.0720<br>0.9172<br>9.704<br>0.431|



This process can be formalized as: 

**==> picture [201 x 12] intentionally omitted <==**

After extensive experiments, we found that using _M[m]_ to differentiate between the foreground and background is not optimal. As shown in Fig. 5, if the original model is wearing a short-sleeved garment and the target garment is long-sleeved, the sleeve in the generated image is incorrectly marked as background and replaced with the arm from the original model image _I[m]_ , causing a style mismatch. To solve this, we use the union of the model clothing mask _M[m]_ and the garment mask _M[g]_ , and the complement as the background mask _M[bg]_ : 

**==> picture [181 x 12] intentionally omitted <==**

**==> picture [202 x 10] intentionally omitted <==**

**==> picture [202 x 13] intentionally omitted <==**

The subsequent denoising steps proceed on _zt[∗]_ 2[until][the][la-] tent _z_ 0 is reached according to Eq. (2). Ultimately, by decoding _z_ 0, we obtain the generated try-on result image _I[r]_ . 

## _D. Adaptive Sampling Termination_ 

While DiffusionTrend achieves training-free virtual try-on by injecting garment and background latents at predefined timesteps, we observe that the quality of the generated results is sensitive to the total number of sampling steps _T_ , especially in terms of color consistency and structural coherence. To mitigate this issue and enhance robustness across diverse generation conditions, we introduce a sampling termination strategy based on the perceptual color difference metric, CIEDE2000 [58] (∆ _E_ 00). This strategy allows the model to dynamically determine the optimal termination step during sampling. 

Specifically, we compute the average ∆ _E_ 00 between the generated image and the target garment image in the garmentmasked region after each sampling step. Once the color deviation increases relative to the previous step, it indicates that the generative result has started to deviate from the target appearance. At this point, we halt the sampling process and roll back to the previous intermediate result with the 

lowest ∆ _E_ 00 value. This approach requires no additional training or parameter tuning and effectively balances quality and efficiency. 

## IV. EXPERIMENTATION 

## _A. Experimental Setup_ 

**Dataset.** We conduct extensive experiments on two highresolution datasets from the VITON benchmark: VITONHD [11] and DressCode [35]. The VITON-HD dataset includes 13,679 pairs of images, each featuring a front-view upper-body shot of women alongside corresponding in-store garments, split into 11,647 training pairs and 2,032 testing pairs. The DressCode dataset is larger, with 48,392 training pairs and 5,400 testing pairs, featuring front-view full-body images of individuals with corresponding in-store garments, categorized into upper-body, lower-body, and dresses. 

To train the apparel localization network, we use model images from all DressCode. We extract relevant clothing portions from the DressCode training set label maps as our ground truth, _MGT_ . We also integrate in-store garment images and their masks from the VITON-HD training set into _MGT_ . For evaluation, we use the test sets from DressCode and VITON-HD to assess our method. 

**Evaluation Metrics.** We assess both paired and unpaired scenarios. In the paired scenario, the target human image and its corresponding garment image are used for reconstruction. In the unpaired scenario, different garment images are used for the virtual try-on experience. To evaluate the quality of images generated in the paired scenario, we use LPIPS [60] and SSIM [61] metrics to measure resemblance to the original image. In the unpaired scenario, we use FID [62] and KID [63] to gauge the realism and fidelity of the synthesized images. 

**Implementation Details.** We use the Adam optimizer [64] for the apparel localization network with a learning rate of 1e-4, halving it every 10 epochs. The network is trained for 35 epochs on two RTX 3090 GPUs. During inference, the number of clusters _K_ is set to 5. We set the apparel-related prompt to “clothes” and apply our method to Stable Diffusion XL (SDXL) [65]. For the inversion phase, we use an empty prompt, and for model image generation, the prompt is “model 

JOURNAL OF L[A] TEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 

8 

**==> picture [490 x 672] intentionally omitted <==**

**----- Start of picture text -----**<br>
DCI-VTON LaDI-VTON IDM-VTON StableVITON OOTDiffusion DiffusionTrend (Ours)<br>**----- End of picture text -----**<br>


Fig. 6. Qualitative comparisons on VITON-HD [11](1st _∼_ 2nd row) and DressCode [35] (3rd _∼_ 8th row). 

JOURNAL OF L[A] TEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 

9 

**==> picture [438 x 185] intentionally omitted <==**

Fig. 7. Qualitative results on SHHQ-1.0 [59] model images with garments from DressCode [35] and VITON-HD [11]. 

wearing clothes.” We conduct DDIM sampling [40] with 50 steps and set the classifier-free guidance to 7.5. Garment latent infusion occurs at time step _t_ 1 = 40, and background restoration at _t_ 2 = 15. The entire generation process is carried out on a single RTX 3090 GPU. 

## _B. Experimental Results_ 

**Quantitative Results.** Table II shows the quantitative comparisons between DiffusionTrend and other methods on VITON-HD [11] and DressCode [35] test datasets. The results on DressCode of DCI-VTON [4] and StableVITON [8] contain our implementation because no usable codes are given. Despite relying solely on pre-trained models for tasks like garment latent infusion and background restoration, our performance lags behind SOTA models. Past methods incur high training costs to ensure generated try-on images remain realistic and natural under various complex poses, as confirmed by evaluation metrics. In contrast, DiffusionTrend aims to provide a resource-efficient, user-friendly tool for quickly confirming purchase intentions under simple try-on poses. Without training the diffusion model on large datasets, our basic operations fall short in handling complex poses, leading to suboptimal performance. As the first visual try-on model without training on diffusion models, our approach differs from traditional methods in motivation and technical approaches, making traditional try-on dataset evaluations insufficient for comprehensively measuring our method’s effectiveness. Thus, quantitative experiments serve only as a reference, while qualitative experiments will further demonstrate our superiority. 

**Qualitative Results.** Fig. 6 provides the qualitative comparison of DiffusionTrend with the state-of-the-art baselines on the VITON-HD [11] and DressCode [35] datasets. The results indicate that our DiffusionTrend performs as well as baseline methods under simple poses. 

First, most baseline methods fail to generate realistic wrinkles after applying warp techniques, simply transferring wrinkles from in-store garment images. 

Second, our approach extracts richer detail features from noise latent, allowing for more accurate detail reconstruction 

in complex garment patterns. For instance, in the first row of Fig. 6, the cartoon pattern on the garments reconstructed by our method more closely resembles the original image compared to other baseline methods. In the fourth row, while most methods erroneously reconstruct the garment as a short skirt, our approach captures the appropriate skirt length and details the metallic embellishments at the waist. 

To further evaluate the generalizability of our method across datasets, we use person images from SHHQ-1.0 [59] as the model image and garments from DressCode [35] and VITONHD [11] for try-on. As shown in Fig. 7, our method produces visually coherent and naturally blended results, demonstrating stable performance and generalizability across datasets. 

**Accuracy of the Precise Apparel Localization Network.** To verify the effectiveness of our proposed precise apparel localization network, we conduct experiments on the test sets of the DressCode and VITON-HD datasets. For the DressCode dataset, we utilize label maps to extract ground truth masks, while for the VITON-HD dataset, we use imageparse-v3. In both cases, we extract specific colored pixels and convert them into binary masks. We then use our apparel localization network to extract clothing masks and evaluate the performance using IoU (Intersection over Union) and Dice (Dice Coefficient) metrics. Generally, a model with an IoU above 0.5 and a Dice score above 0.7 is considered applicable in the research field. 

Our results, as presented in Table III, demonstrate that our compact CNN achieves high accuracy in mask extraction 

## TABLE III 

QUANTITATIVE METRICS ON VIRTUAL TRY-ON DATASETS FOR VALIDATING MASK PREDICTION ACCURACY. 

|Metrics|DressCode|DressCode|DressCode|VITON-HD|
|---|---|---|---|---|
||upper<br>body|lower<br>body|dresses||
|IoU|0.7213<br>0.7138<br>0.7969<br>0.7357||||
|Dice|0.8162|0.8038|0.8538|0.8194|



JOURNAL OF L[A] TEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 

10 

**==> picture [438 x 186] intentionally omitted <==**

**----- Start of picture text -----**<br>
𝑡1 = 50 𝑡1 = 45 𝑡1 = 40(Ours) 𝑡1 = 35 𝑡1 = 30 𝑡1 = 25<br>𝑡2 = 25 𝑡2 = 20 𝑡2 = 15(Ours) 𝑡2 = 10 𝑡2 = 5 𝑡2 = 0<br>**----- End of picture text -----**<br>


Fig. 8. Visual ablations for _t_ 1 in Garment Latent Infusion and _t_ 2 in Background Restoration. 

TABLE IV 

QUANTITATIVE ABLATIONS FOR _t_ 1 IN GARMENT LATENT INFUSION AND _t_ 2 IN BACKGROUND RESTORATION. **BOLD** NUMBERS INDICATE THE BEST PERFORMANCE. 

|Timesteps|Timesteps|Metrics|Metrics|Metrics|Metrics|
|---|---|---|---|---|---|
|_t_1<br>_t_2<br>LPIPS_↓_<br>SSIM_↑_<br>FID_↓_<br>KID_↓_||||||
|||||||
|50<br>45<br>40 (Ours)<br>35<br>30<br>25|15|0.0791<br>0.0756<br>0.0720<br>0.0712<br>0.0698<br>**0.0691**|0.9142<br>0.9153<br>0.9172<br>0.9177<br>0.9183<br>**0.9187**|9.8943<br>9.7351<br>**9.7040**<br>9.8118<br>10.0197<br>10.1865|**0.43**<br>**0.43**<br>**0.43**<br>0.45<br>0.47<br>0.47|
|||||||
|40|25<br>20<br>15 (Ours)<br>10<br>5<br>0|0.0754<br>0.0737<br>**0.0720**<br>0.0724<br>0.0727<br>0.1676|0.9144<br>0.9160<br>0.9172<br>**0.9173**<br>**0.9173**<br>0.8527|**9.2217**<br>9.4248<br>9.7040<br>10.0531<br>10.6983<br>15.9559|**0.40**<br>0.42<br>0.43<br>0.46<br>0.50<br>0.73|



tasks, with IoU scores reaching 0.7 and Dice scores exceeding 0.8. While its accuracy may lag behind SOTA segmentation models, it is more than sufficient for handling virtual try-on tasks. These results validate the effectiveness and practicality of our approach. 

## _C. Ablation Study_ 

In this section, we examine the optimality of our method’s components and experimental settings, including background restoration and the timestep for latent infusion or replacement (Sec. III-C), garment mask refinement (Sec. III-B), garment warping and adaptive sampling termination (Sec. III-C, Sec. III-D). 

_t_ 1 **for Garment Latent Infusion.** The upper part of Table IV shows that incorporating clothing information too early tends to lower the LPIPS, SSIM, and FID scores. Specifically, 

as observed in the top row of Fig. 8, prematurely incorporating garment information results in a deficiency of integration between the model and the garment, effectively reconstructing their respective latent representations in disparate areas without any interaction. This issue is clearly visible, as there are pronounced boundaries between the clothing and the human figure, creating an impression of disjunction rather than a cohesive unity. On the other hand, introducing the information at a later stage impedes the development and enhancement of the garment’s finer details. This is observable in the figure, where the cartoon patterns are prone to becoming indistinct, thereby compromising the overall quality of the image. 

_t_ 2 **for Background Restoration.** Although the quantitative results in Table IV show that the FID and KID scores are higher at _t_ 2 = 25 and 20, it can be observed from the second row of Fig. 8 that, performing background restoration too early 

JOURNAL OF L[A] TEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 

11 

can negatively impact the generation of garment details. For instance, at _t_ 2 = 25, the waist’s metallic embellishments are lost during the reconstruction process; at _t_ 2 = 20, the situation improves slightly, but the metallic dots on either side are still missing. In contrast, performing it too late creates a distinct boundary between the background and the foreground, resulting in unnatural outcomes, and leads to the over-rendering of details. In the figure at _t_ 2 = 15, there are small red artifacts visible around the metallic dots. 

**Garment Mask Refinement Strategy.** Initially, to determine an effective approach for separating upper and lower garments from the coarse masks, we evaluate several structural decomposition methods. Region growing tends to over-expand into background areas due to intensity fluctuations, while graph cut often includes the entire human silhouette based on global contrast, deviating from our goal of localized garment segmentation. We observe that a basic K-means clustering can achieve a reasonably effective separation between upper and lower garments. To further improve the segmentation precision and boundary quality, we explore two post-processing techniques: guided filtering and active contour. The former smooths edges but lacks structural modeling ability, while the latter explicitly optimizes contour continuity through energy minimization. Considering both performance and efficiency, we adopt the combination of K-means clustering and active contour refinement as our final strategy. This setup achieves more natural garment boundaries with minimal additional overhead. Qualitative comparisons are presented in Fig. 9. 

**==> picture [247 x 286] intentionally omitted <==**

**----- Start of picture text -----**<br>
Model  Rough  Region  Graph  K-Means K-Means+ K-Means+<br>Image Mask Growing Cut Guided  Active<br>Filter Contour<br>(Ours)<br>**----- End of picture text -----**<br>


Fig. 9. Visual comparison of different garment structure decomposition strategies. 

**Garment Warping Strategy.** To assess the feasibility of garment warping strategies, we conduct a comparative study 

on several representative explicit warping modules and perspective transformations. Specifically, we extract the warping modules from: (1) DCI-VTON [4], which represents a dense correspondence-based appearance flow approach; (2) SCW-VTON [55], which introduces shape constraints into the appearance flow to improve contour alignment; (3) PLVTON [56], which combines affine transformation with dense flow. These approaches represent typical warping modules in diffusion-based and CNN-based virtual try-on pipelines. All of them require dedicated training and rely on structured annotations such as parsing maps and keypoints. To ensure a fair comparison, we integrate their warped garment outputs into the DiffusionTrend pipeline and compared the try-on results with those using perspective transformation. The visual comparison is presented in Fig. 10. 

Both visual results and technical analysis reveal several shared drawbacks among these warping methods: (1) Unstable results with distorted edges: Outputs vary considerably across samples, often showing irregular deformations or missing garment parts after warping. (2) Poor generalization and weak robustness: These modules, trained on specific datasets (e.g., VITON [66] or VITON-HD [11]), but exhibit clear structural failures on the DressCode [35] dataset, as shown in our visual examples. (3) Heavy reliance on structural inputs: Most methods require annotations such as keypoints, parsing maps, or densepose, which increases system complexity and reduces accessibility. (4) Strong coupling with downstream try-on models: Although the warping modules are trained independently, their outputs are typically used together with try-on networks during training, leading to joint optimization where the try-on network compensates for warping errors to some extent. However, since we do not train the diffusion model, these imperfect warped results cannot be rectified by the diffusion process itself, leading to failure cases. 

Beyond trained warping modules, we also explore nonlearning heuristics such as region-based pasting using parsing masks or relative keypoint-based garment stretching. While theoretically simple and training-free, these approaches also rely on structured annotations and showed inferior visual consistency. 

In light of these conclusions, we retain perspective transformation as our final garment alignment strategy. Despite its simplicity, it offers consistent and controllable warping results without requiring structural supervision or additional training, making it well-suited to the training-free paradigm of DiffusionTrend. 

**Adaptive Sampling Termination.** We conduct an experiment in which the number of sampling steps is gradually reduced from 50 to 44, in order to evaluate how this parameter affects the generation quality of DiffusionTrend and to verify the effectiveness of our adaptive sampling termination strategy. As shown in Fig. 11, varying the sampling steps leads to three major types of degradation: (1) Color deviation, where higher sampling steps (e.g., 50) produce oversaturated garments and backgrounds with noticeable reddish or overly bright tones; (2) Loss of detail, where steps below a threshold (e.g., _<_ 45) result in blurry textures and contours; and (3) Structural distortion, where overly few steps lead to background blending failures 

JOURNAL OF L[A] TEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 

12 

**==> picture [490 x 156] intentionally omitted <==**

**----- Start of picture text -----**<br>
Warping Module of DCI-VTON Warping Module of SCW-VTON Warping Module of PL-VTON Perspective Transformations (Ours)<br>**----- End of picture text -----**<br>


Fig. 10. Comparison of results using different explicit warping modules and perspective transformations. Left: garment with warped/perspective transformation; Right: results generated using the processed garment. 

**==> picture [490 x 180] intentionally omitted <==**

**----- Start of picture text -----**<br>
steps = 50 steps = 49 steps = 48 steps = 47 steps = 46 (Ours) steps = 45 steps = 44 garment<br>steps = 50 steps = 49 steps = 48 steps = 47 (Ours) steps = 46 steps = 45 steps = 44 garment<br>DDIM Inversion<br>EasyInv<br>**----- End of picture text -----**<br>


Fig. 11. Effectiveness of our adaptive sampling termination across different inversion methods. 

and facial deformations. 

We attribute this sensitivity to two key factors. First, the training-free nature of DiffusionTrend leaves sampling steps as the only adjustable variable to balance quality and efficiency. Second, a larger number of sampling steps tends to amplify prompt influence, which can cause unwanted deviation from the appearance of the reference garment. 

Our adaptive sampling termination strategy mitigates this by terminating sampling at the point where visual fidelity begins to decline. To ensure early-stage stability, this termination strategy is only activated after step _t_ = 45 in our implementation. As shown in Fig. 11, under the standard SDXL [65] + DDIM Inversion [40] pipeline, our adaptive sampling termination strategy consistently terminates at step 46, aligning well with our empirical observation. We further applied the same strategy to the SDXL + EasyInv [67] setup, where the model automatically selected step 47 as the optimal stopping point. The generated results are visually consistent and stable, demonstrating the robustness and reliability of our strategy across different settings. 

## V. LIMITATIONS AND FUTURE WORK 

Our DiffusionTrend model encounters several challenges: **Reliance on Inversion Quality.** Our proposed DiffusionTrend’s performance is indeed influenced by the quality of 

DDIM inversion results. However, we find that the evolution of more powerful diffusion models is likely to yield better results even with the same inversion method, As demonstrated in Fig. 12. Moreover, ongoing advancements in inversion methods, such as NULL-Text Inversion [68], ReNoise [69], FixedPoint Iteration [70], and EasyInv [67], provide promising directions for further improvements. 

Furthermore, our framework is inherently compatible with ongoing inversion advances and can directly benefit from them without any architectural changes. We thus view current inversion limitations not as constraints, but as evolving components of an improving ecosystem. We believe that the emergence of these improved diffusion models and inversion techniques will significantly expand the application space of our training-free try-on paradigm. 

**Challenges in Adapting to Complex Poses.** It encounters difficulties in generating complex poses, especially in rendering body parts that are not visible in the original model image, due to limitations inherent in the pre-trained model. For instance, if the original model is depicted wearing a long-sleeved shirt and the target garment is a short-sleeved one, the model is unable to convincingly render the exposed arms—a challenge commonly faced by most virtual try-on models. These limitations suggest that future research should 

JOURNAL OF L[A] TEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 

13 

**==> picture [233 x 151] intentionally omitted <==**

**----- Start of picture text -----**<br>
DDIM Inversion Fixed-Point Iteration EasyInv<br>SD-V1.4<br>SD-XL<br>**----- End of picture text -----**<br>


Fig. 12. Improved inversion results achieved by SD-XL and advanced inversion methods. 

prioritize better recovery of fine-grained details in clothing and strengthen the model’s ability to generate unseen body parts. 

Moreover, due to the limitations of perspective transformations, DiffusionTrend encounters difficulties when dealing with complex model poses, particularly in accurately aligning garments with intricate body movements or occluded regions. While regular poses are generally sufficient for consumers to make purchase decisions by providing a clear view of garment fit and style, further exploration into addressing these challenges and refining the quality of background restoration are areas that merit deeper investigation. 

**Inference Overhead and Optimization Prospects.** DiffusionTrend requires additional time during inference due to the DDIM inversion process, which involves a similar number of steps as DDIM sampling. This time overhead mainly occurs during the inference process and is a potential drawback. However, it can be mitigated through quantization ( _e.g._ , Q-Diffusion [71], NDTC [72]) and pruning ( _e.g._ , LDPruner [73]). While this paper focuses on reducing training overhead, combining these techniques to optimize inference speed will be an important direction for future research. 

Despite these shortcomings, DiffusionTrend offers a lowcost, lightweight paradigm for the virtual try-on field that circumvents extensive diffusion model training. Optimism remains high that this approach will continue to evolve as the field advances. We respectfully ask the academic community to recognize the value of this exploratory work and extend the necessary patience and support for further development. 

## VI. CONCLUSION 

In this paper, we have introduced DiffusionTrend, a novel try-on methodology that forgoes the need for training diffusion models, thereby offering straightforward, conventional pose virtual try-on services with minimal computational demands. Capitalizing on sophisticated diffusion models, DiffusionTrend harnesses latents brimming with prior information to encapsulate the nuances of garment details. Throughout the diffusion denoising process, these details are effortlessly merged into the model image generation, expertly directed by a precise garment mask generated by a lightweight and compact CNN. Differing from other approaches, DiffusionTrend sidesteps 

the necessity for labor-intensive training of diffusion models on extensive datasets. It also dispenses with the need for various types of user-unfriendly model inputs. Our experiments demonstrate that, despite lower metric performance, DiffusionTrend delivers a visually convincing virtual try-on experience, all while maintaining the quality and richness of fashion presentation. 

## REFERENCES 

- [1] D. Song, X. Zhang, J. Zhou, W. Nie, R. Tong, and A.-A. Liu, “Imagebased virtual try-on: A survey,” _arXiv preprint arXiv:2311.04811_ , 2023. 

- [2] T. Islam, A. Miron, X. Liu, and Y. Li, “Deep learning in virtual try-on: A comprehensive survey,” _IEEE Access_ , 2024. 

- [3] L. Zhu, D. Yang, T. Zhu, F. Reda, W. Chan, C. Saharia, M. Norouzi, and I. Kemelmacher-Shlizerman, “Tryondiffusion: A tale of two unets,” in _Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)_ , 2023, pp. 4606–4615. 

- [4] J. Gou, S. Sun, J. Zhang, J. Si, C. Qian, and L. Zhang, “Taming the power of diffusion models for high-quality virtual try-on with appearance flow,” in _ACM Int. Conf. Multimedia_ , 2023, pp. 7599–7607. 

- [5] D. Morelli, A. Baldrati, G. Cartella, M. Cornia, M. Bertini, and R. Cucchiara, “Ladi-vton: latent diffusion textual-inversion enhanced virtual try-on,” in _ACM Int. Conf. Multimedia_ , 2023, pp. 8580–8589. 

- [6] X. Li, M. Kampffmeyer, X. Dong, Z. Xie, F. Zhu, H. Dong, X. Liang _et al._ , “Warpdiffusion: Efficient diffusion model for high-fidelity virtual try-on,” _arXiv preprint arXiv:2312.03667_ , 2023. 

- [7] Y. Xu, T. Gu, W. Chen, and C. Chen, “Ootdiffusion: Outfitting fusion based latent diffusion for controllable virtual try-on,” _arXiv preprint arXiv:2403.01779_ , 2024. 

- [8] J. Kim, G. Gu, M. Park, S. Park, and J. Choo, “Stableviton: Learning semantic correspondence with latent diffusion model for virtual try-on,” _arXiv preprint arXiv:2312.01725_ , 2023. 

- [9] Y. Choi, S. Kwak, K. Lee, H. Choi, and J. Shin, “Improving diffusion models for virtual try-on,” _arXiv preprint arXiv:2403.05139_ , 2024. 

- [10] M. Chen, X. Chen, Z. Zhai, C. Ju, X. Hong, J. Lan, and S. Xiao, “Wear-any-way: Manipulable virtual try-on via sparse correspondence alignment,” _arXiv preprint arXiv:2403.12965_ , 2024. 

- [11] S. Choi, S. Park, M. Lee, and J. Choo, “Viton-hd: High-resolution virtual try-on via misalignment-aware normalization,” in _Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)_ , 2021, pp. 14 131–14 140. 

- [12] S. Lee, G. Gu, S. Park, S. Choi, and J. Choo, “High-resolution virtual tryon with misalignment and occlusion-handled conditions,” in _Eur. Conf. Comput. Vis._ , 2022, pp. 204–219. 

- [13] Y. Ge, Y. Song, R. Zhang, C. Ge, W. Liu, and P. Luo, “Parser-free virtual try-on via distilling appearance flows,” in _Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)_ , 2021, pp. 8485–8493. 

- [14] Z. Xie, Z. Huang, X. Dong, F. Zhao, H. Dong, X. Zhang, F. Zhu, and X. Liang, “Gp-vton: Towards general purpose virtual try-on via collaborative local-flow global-parsing learning,” in _Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)_ , 2023, pp. 23 550–23 559. 

- [15] N. Fang, L. Qiu, S. Zhang, Z. Wang, and K. Hu, “Pg-vton: A novel image-based virtual try-on method via progressive inference paradigm,” _IEEE Trans. Multimedia_ , 2024. 

- [16] K. Li, J. Zhang, and D. Forsyth, “Povnet: Image-based virtual tryon through accurate warping and residual,” _IEEE Trans. Pattern Anal. Mach. Intell._ , vol. 45, no. 10, pp. 12 222–12 235, 2023. 

- [17] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio, “Generative adversarial nets,” _Proc. Adv. Neural Inform. Process. Syst._ , vol. 27, 2014. 

- [18] S. Zhang, M. Ni, S. Chen, L. Wang, W. Ding, and Y. Liu, “A two-stage personalized virtual try-on framework with shape control and texture guidance,” _IEEE Trans. Multimedia_ , 2024. 

- [19] C.-L. Chou, C.-Y. Chen, C.-W. Hsieh, H.-H. Shuai, J. Liu, and W.H. Cheng, “Template-free try-on image synthesis via semantic-guided optimization,” _IEEE Trans. Neural Netw. Learn. Syst._ , vol. 33, no. 9, pp. 4584–4597, 2021. 

- [20] J. Xu, Y. Pu, R. Nie, D. Xu, Z. Zhao, and W. Qian, “Virtual try-on network with attribute transformation and local rendering,” _IEEE Trans. Multimedia_ , vol. 23, pp. 2222–2234, 2021. 

- [21] S. Zhang, X. Han, W. Zhang, X. Lan, H. Yao, and Q. Huang, “Limbaware virtual try-on network with progressive clothing warping,” _IEEE Trans. Multimedia_ , vol. 26, pp. 1731–1746, 2023. 

JOURNAL OF L[A] TEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 

14 

- [22] B. Hu, P. Liu, Z. Zheng, and M. Ren, “Spg-vton: Semantic prediction guidance for multi-pose virtual try-on,” _IEEE Trans. Multimedia_ , vol. 24, pp. 1233–1246, 2022. 

- [23] S. Bai, H. Zhou, Z. Li, C. Zhou, and H. Yang, “Single stage virtual tryon via deformable attention flows,” in _Eur. Conf. Comput. Vis._ , 2022, pp. 409–425. 

- [24] X. Han, X. Hu, W. Huang, and M. R. Scott, “Clothflow: A flow-based model for clothed person generation,” in _Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV)_ , 2019, pp. 10 471–10 480. 

- [25] Z. Xing, Y. Wu, S. Liu, S. Di, and H. Ma, “Virtual try-on with garment self-occlusion conditions,” _IEEE Trans. Multimedia_ , vol. 25, pp. 7323– 7336, 2022. 

- [26] D. Song, J.-H. Zeng, M. Liu, X.-Y. Li, and A.-A. Liu, “Fashion customization: Image generation based on editing clue,” _IEEE Trans. Circuits Syst. Video Technol._ , 2023. 

- [27] T. Zhou, S. Tulsiani, W. Sun, J. Malik, and A. A. Efros, “View synthesis by appearance flow,” in _Eur. Conf. Comput. Vis._ , 2016, pp. 286–301. 

- [28] C. Du, F. Yu, M. Jiang, A. Hua, X. Wei, T. Peng, and X. Hu, “Vtonscfa: A virtual try-on network based on the semantic constraints and flow alignment,” _IEEE Trans. Multimedia_ , vol. 25, pp. 777–791, 2022. 

- [29] F. Yu, A. Hua, C. Du, M. Jiang, X. Wei, T. Peng, L. Xu, and X. Hu, “Vton-mp: Multi-pose virtual try-on via appearance flow and feature filtering,” _IEEE Trans. Consum. Electron._ , 2023. 

- [30] H. Yang, R. Zhang, X. Guo, W. Liu, W. Zuo, and P. Luo, “Towards photo-realistic virtual try-on by adaptively generating-preserving image content,” in _Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)_ , 2020, pp. 7850–7859. 

- [31] T. Issenhuth, J. Mary, and C. Calauzenes, “Do not mask what you do not need to mask: a parser-free virtual try-on,” in _Eur. Conf. Comput. Vis._ , 2020, pp. 619–635. 

- [32] T. Liu, J. Zhang, X. Nie, Y. Wei, S. Wei, Y. Zhao, and J. Feng, “Spatial-aware texture transformer for high-fidelity garment transfer,” _IEEE Trans. Image Process._ , vol. 30, pp. 7499–7510, 2021. 

- [33] S. Song, W. Zhang, J. Liu, Z. Guo, and T. Mei, “Unpaired person image generation with semantic parsing transformation,” _IEEE Trans. Pattern Anal. Mach. Intell._ , vol. 43, no. 11, pp. 4161–4176, 2020. 

- [34] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” _Proc. Adv. Neural Inform. Process. Syst._ , vol. 33, pp. 6840–6851, 2020. 

- [35] D. Morelli, M. Fincato, M. Cornia, F. Landi, F. Cesari, and R. Cucchiara, “Dress code: high-resolution multi-category virtual try-on,” in _Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)_ , 2022. 

- [36] R. A. G¨uler, N. Neverova, and I. Kokkinos, “Densepose: Dense human pose estimation in the wild,” in _Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)_ , 2018, pp. 7297–7306. 

- [37] K. Gong, X. Liang, D. Zhang, X. Shen, and L. Lin, “Look into person: Self-supervised structure-sensitive learning and a new benchmark for human parsing,” in _Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)_ , 2017, pp. 932–940. 

- [38] X. Han, Z. Wu, Z. Wu, R. Yu, and L. S. Davis, “Viton: An imagebased virtual try-on network,” in _Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)_ , 2018, pp. 7543–7552. 

- [39] Z. Cao, T. Simon, S.-E. Wei, and Y. Sheikh, “Realtime multi-person 2d pose estimation using part affinity fields,” in _Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)_ , 2017, pp. 7291–7299. 

- [40] J. Song, C. Meng, and S. Ermon, “Denoising diffusion implicit models,” _arXiv preprint arXiv:2010.02502_ , 2020. 

- [41] T. Brooks, A. Holynski, and A. A. Efros, “Instructpix2pix: Learning to follow image editing instructions,” in _Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)_ , 2023, pp. 18 392–18 402. 

- [42] B. Kawar, S. Zada, O. Lang, O. Tov, H. Chang, T. Dekel, I. Mosseri, and M. Irani, “Imagic: Text-based real image editing with diffusion models,” in _Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)_ , 2023, pp. 6007–6017. 

- [43] B. Xue, S. Ran, Q. Chen, R. Jia, B. Zhao, and X. Tang, “Dccf: Deep comprehensible color filter learning framework for high-resolution image harmonization,” in _Eur. Conf. Comput. Vis._ , 2022, pp. 300–316. 

- [44] B. Yang, S. Gu, B. Zhang, T. Zhang, X. Chen, X. Sun, D. Chen, and F. Wen, “Paint by example: Exemplar-based image editing with diffusion models,” in _Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)_ , 2023, pp. 18 381–18 391. 

- [45] N. Ruiz, Y. Li, V. Jampani, Y. Pritch, M. Rubinstein, and K. Aberman, “Dreambooth: Fine tuning text-to-image diffusion models for subjectdriven generation,” in _Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)_ , 2023, pp. 22 500–22 510. 

- [46] R. Gal, Y. Alaluf, Y. Atzmon, O. Patashnik, A. H. Bermano, G. Chechik, and D. Cohen-Or, “An image is worth one word: Person- 

   - alizing text-to-image generation using textual inversion,” _arXiv preprint arXiv:2208.01618_ , 2022. 

- [47] N. Kumari, B. Zhang, R. Zhang, E. Shechtman, and J.-Y. Zhu, “Multiconcept customization of text-to-image diffusion,” in _Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)_ , 2023, pp. 1931–1941. 

- [48] A. Hertz, R. Mokady, J. Tenenbaum, K. Aberman, Y. Pritch, and D. Cohen-Or, “Prompt-to-prompt image editing with cross attention control,” _arXiv preprint arXiv:2208.01626_ , 2022. 

- [49] M. Cao, X. Wang, Z. Qi, Y. Shan, X. Qie, and Y. Zheng, “Masactrl: Tuning-free mutual self-attention control for consistent image synthesis and editing,” in _Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV)_ , 2023, pp. 22 560–22 570. 

- [50] R. T. Chen, Y. Rubanova, J. Bettencourt, and D. K. Duvenaud, “Neural ordinary differential equations,” _Proc. Adv. Neural Inform. Process. Syst._ , vol. 31, 2018. 

- [51] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo _et al._ , “Segment anything,” in _Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV)_ , 2023, pp. 4015– 4026. 

- [52] K. He, G. Gkioxari, P. Doll´ar, and R. Girshick, “Mask r-cnn,” in _Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV)_ , 2017, pp. 2961–2969. 

- [53] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark _et al._ , “Learning transferable visual models from natural language supervision,” in _Int. Conf. Mach. Learn._ , 2021, pp. 8748–8763. 

- [54] M. Kass, A. Witkin, and D. Terzopoulos, “Snakes: Active contour models,” _Int. J. Comput. Vis. (IJCV)_ , vol. 1, no. 4, pp. 321–331, 1988. 

- [55] X. Han, S. Zheng, Z. Li, C. Wang, X. Sun, and Q. Meng, “Shape-guided clothing warping for virtual try-on,” in _ACM Int. Conf. Multimedia_ , 2024, pp. 2593–2602. 

- [56] S. Zhang, X. Han, W. Zhang, X. Lan, H. Yao, and Q. Huang, “Limbaware virtual try-on network with progressive clothing warping,” _IEEE Trans. Multimedia_ , vol. 26, pp. 1731–1746, 2023. 

- [57] T. Wu, C. Si, Y. Jiang, Z. Huang, and Z. Liu, “Freeinit: Bridging initialization gap in video diffusion models,” _arXiv preprint arXiv:2312.07537_ , 2023. 

- [58] G. Sharma, W. Wu, and E. N. Dalal, “The ciede2000 color-difference formula: Implementation notes, supplementary test data, and mathematical observations,” _Color Research and Application_ , vol. 30, no. 1, pp. 21–30, 2005. 

- [59] J. Fu, S. Li, Y. Jiang, K.-Y. Lin, C. Qian, C. C. Loy, W. Wu, and Z. Liu, “Stylegan-human: A data-centric odyssey of human generation,” in _Eur. Conf. Comput. Vis._ Springer, 2022, pp. 1–19. 

- [60] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, “The unreasonable effectiveness of deep features as a perceptual metric,” in _Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)_ , 2018, pp. 586–595. 

- [61] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image quality assessment: from error visibility to structural similarity,” _IEEE Trans. Image Process._ , vol. 13, no. 4, pp. 600–612, 2004. 

- [62] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter, “Gans trained by a two time-scale update rule converge to a local nash equilibrium,” _Proc. Adv. Neural Inform. Process. Syst._ , vol. 30, 2017. 

- [63] M. Bi´nkowski, D. J. Sutherland, M. Arbel, and A. Gretton, “Demystifying mmd gans,” _arXiv preprint arXiv:1801.01401_ , 2018. 

- [64] D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” _arXiv preprint arXiv:1412.6980_ , 2014. 

- [65] D. Podell, Z. English, K. Lacey, A. Blattmann, T. Dockhorn, J. M¨uller, J. Penna, and R. Rombach, “Sdxl: Improving latent diffusion models for high-resolution image synthesis,” _arXiv preprint arXiv:2307.01952_ , 2023. 

- [66] X. Han, Z. Wu, Z. Wu, R. Yu, and L. S. Davis, “Viton: An imagebased virtual try-on network,” in _Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)_ , 2018. 

- [67] Z. Zhang, M. Lin, S. Yan, and R. Ji, “Easyinv: Toward fast and better ddim inversion,” _arXiv preprint arXiv:2408.05159_ , 2024. 

- [68] R. Mokady, A. Hertz, K. Aberman, Y. Pritch, and D. Cohen-Or, “Nulltext inversion for editing real images using guided diffusion models,” in _Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)_ , 2023, pp. 6038–6047. 

- [69] D. Garibi, O. Patashnik, A. Voynov, H. Averbuch-Elor, and D. CohenOr, “Renoise: Real image inversion through iterative noising,” _arXiv preprint arXiv:2403.14602_ , 2024. 

- [70] Z. Pan, R. Gherardi, X. Xie, and S. Huang, “Effective real image editing with accelerated iterative diffusion inversion,” in _Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV)_ , 2023, pp. 15 912–15 921. 

15 

JOURNAL OF L[A] TEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 

- [71] X. Li, Y. Liu, L. Lian, H. Yang, Z. Dong, D. Kang, S. Zhang, and K. Keutzer, “Q-diffusion: Quantizing diffusion models,” in _Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV)_ , 2023, pp. 17 535–17 545. 

- [72] Y. Shang, Z. Yuan, B. Xie, B. Wu, and Y. Yan, “Post-training quantization on diffusion models,” in _Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)_ , 2023, pp. 1972–1981. 

- [73] T. Castells, H.-K. Song, B.-K. Kim, and S. Choi, “Ld-pruner: Efficient pruning of latent diffusion models using task-agnostic insights,” in _Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)_ , 2024, pp. 821–830. 

**Wengyi Zhan** completed her undergraduate studies and obtained the Bachelor’s degree in Intelligence Science and Technology from Xiamen University, Xiamen, China, in 2023. 

**==> picture [66 x 91] intentionally omitted <==**

She is currently a master’s student at the School of Informatics, Xiamen University, specializing in Intelligence Science and Technology. Her current research interest focuses on diffusion models for consistent video/image generation. 

**Mingbao Lin** finished his M.S.-Ph.D. study and obtained the Ph.D. degree in intelligence science and technology from Xiamen University, Xiamen, China, in 2022. Earlier, he received the B.S. degree from Fuzhou University, Fuzhou, China, in 2016. 

**==> picture [66 x 91] intentionally omitted <==**

He is currently a research scientist with the Skywork AI, Singapore, and also an adjunct industry supervisor with Xiamen University. His current research interest is to develop low-latency multimodal interaction system, such as text, audio, image, video, _etc_ . 

**Rongrong Ji** (Senior Member, IEEE) is a Nanqiang Distinguished Professor at Xiamen University, the Director of the Office of Science and Technology at Xiamen University, and the Director of Media Analytics and Computing Lab. He was awarded as the National Science Foundation for Excellent Young Scholars (2014), the National Ten Thousand Plan for Young Top Talents (2017), and the National Science Foundation for Distinguished Young Scholars (2020). His research falls in the field of computer vision, multimedia analysis, and machine learning. He has published 50+ papers in ACM/IEEE Transactions, including TPAMI and IJCV, and 100+ full papers on top-tier conferences, such as CVPR and NeurIPS. His publications have got over 20K citations in Google Scholar. He was the recipient of the Best Paper Award of ACM Multimedia 2011. He has served as Area Chairs in top-tier conferences such as CVPR and ACM Multimedia. He is also an Advisory Member for Artificial Intelligence Construction in the Electronic Information Education Committee of the National Ministry of Education. 

**Shuicheng Yan** (Fellow, IEEE) is currently the Managing Director of Kunlun 2050 Research and Chief Scientist of Kunlun Tech & Skywork AI, and the former Group Chief Scientist of Sea. Prof. Yan Shuicheng is a Fellow of Singapore’s Academy of Engineering, AAAI, ACM, IEEE, and IAPR. His research areas include computer vision, machine learning, and multimedia analysis. Till now, Prof Yan has published over 800 papers at top international journals and conferences, with an H-index of 140+. He has also been named among the annual World’s Highly Cited Researchers nine times. Prof. Yan’s team received ten-time winners or honorable-mention prizes at two core competitions, Pascal VOC and ImageNet (ILSVRC), deemed the “World Cup” in the computer vision community. Besides, his team won more than ten best papers and best student paper awards, particularly a grand slam at the ACM Multimedia, the toptiered conference in multimedia, including the Best Paper Awards thrice, Best Student Paper Awards twice, and Best Demo Award once. 

