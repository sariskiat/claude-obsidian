---
type: paper-fulltext
slug: voost-unified-dit-bidirectional
arxiv_id: null
source_path: /Users/saris.kia.adm/.paper-scholar/voost-unified-dit-bidirectional/2508.04825.md
paper: "[[voost-unified-dit-bidirectional]]"
---
<!-- extracted by afk_extract from 2508.04825.pdf (22p) -->

## Voost: A Unified and Scalable Diffusion Transformer for Bidirectional Virtual Try-On and Try-Off 

Seungyong Lee[1*] Jeong-gi Kwak[1,2*†] 1 NXN Labs 2 University of British Columbia https://nxnai.github.io/Voost 

**==> picture [496 x 282] intentionally omitted <==**

Figure 1. Teaser – We present a single diffusion transformer that jointly addresses virtual try-on and try-off, producing high-quality results that remain robust across diverse poses, garment types, backgrounds, lighting conditions, and image compositions. 

## Abstract 

Virtual try-on aims to synthesize a realistic image of a person wearing a target garment, but accurately modeling garment–body correspondence remains a persistent challenge, especially under pose and appearance variation. In this paper, we propose Voost—a unified and scalable framework that jointly learns virtual try-on and try-off with a single diffusion transformer. By modeling both tasks jointly, Voost enables each garment-person pair to supervise both directions and supports flexible conditioning over gen- 

eration direction and garment category—enhancing garment–body relational reasoning without task-specific networks, auxiliary losses, or additional labels. In addition, we introduce two inference-time techniques: attention temperature scaling for robustness to resolution or mask variation, and self-corrective sampling that leverages bidirectional consistency between tasks. Extensive experiments demonstrate that Voost achieves state-of-the-art results on both try-on and try-off benchmarks, consistently outperforming strong baselines in alignment accuracy, visual fidelity, and generalization. 

> *Equal contribution 

> †Corresponding author 

1 

## 1. Introduction 

Virtual try-on (VTON) is an emerging generative task that enables realistic garment transfer onto a person image, offering a new paradigm for fashion in e-commerce and AR/VR. Despite its promise, VTON remains challenging due to the need for precise alignment, detail preservation, and robustness to occlusion and pose variation. 

Early approaches [13, 21, 27, 85] have adopted warpingbased strategies, often combined with generative modeling, but have struggled to maintain garment fidelity and structural consistency. 

Effectively adapting T2I diffusion models [18, 66] to virtual try-on requires strong image-based conditioning and accurate modeling of garment–person correspondence. To this end, various efforts have been made to improve spatial alignment and relational understanding. These include incorporating auxiliary objectives [41, 91], leveraging reference networks [14, 84, 92], or injecting conditioning signals through spatial concatenation [15]. Despite these efforts, existing methods often fail to establish precise garment– person correspondence. As illustrated in Fig. 2, the dispersed attention patterns reflect limited relational understanding, often resulting in lower visual quality and failure to faithfully reconstruct garment details. 

To address these challenges, we introduce Voost, a unified framework that jointly learns virtual try-on and its inverse task, virtual try-off, which aims to reconstruct the original appearance of the worn garment from a person image, within a single diffusion transformer. Voost adopts a token-level concatenation structure, where spatially aligned garment and person images are fed into a shared embedding space. This design enables the model to reason bidirectionally across try-on and try-off scenarios using a common conditioning layout. By jointly training both directions, Voost strengthens garment–person correspondence while requiring no task-specific networks, auxiliary losses, or additional annotations, as each garment-person pair naturally provides supervision for the reverse process. 

Our framework leverages token-level concatenation and the architectural flexibility of diffusion transformers to handle diverse poses, aspect ratios, and spatial layouts. A task token encodes both generation direction and garment category, enabling scalable learning without task-specific models or fixed resolutions. This unified setup not only supports multitask learning but also mitigates architectural, task-specific, and category-specific inductive biases by exposing the model to broader structural variation. 

In addition, we propose two simple inference-time techniques to improve robustness. Attention temperature scaling mitigates resolution or mask size mismatch between training and test time. We also introduce self-corrective sampling, where try-on and try-off predictions iteratively refine each other within the unified model. 

Remarkably, our unified diffusion transformer achieves state-of-the-art performance on both VTON and VTOFF benchmarks, surpassing strong task-specific baselines [14, 15, 41, 78, 82, 91] optimized individually for each task. We provide extensive qualitative and quantitative analyses that validate the effectiveness of our unified approach and its ability to model garment–person interaction across diverse scenarios. 

## 2. Related work 

Image-conditioned diffusion models. Diffusion models have rapidly evolved to support image-conditioned generation beyond the text-to-image setting [62, 66]. Early methods such as Textual Inversion [19] and DreamBooth [67] enabled concept-specific synthesis via embedding injection or fine-tuning, but incurred high computational costs. Later approaches improved efficiency by leveraging pre-trained encoders like CLIP [63] and DINO [9, 59], as seen in IPAdapter [86] and AnyDoor [12], which use image embeddings to guide generation. To improve spatial consistency, MasaCtrl [6] introduced mutual self-attention to preserve object identity. This was extended by reference-based methods [11, 14, 34, 76, 84] that inject structured visual features through dedicated networks. More recently, concatenationbased methods [15, 35, 75] have emerged as lightweight alternatives that directly feed conditioning signals without auxiliary modules. Additionally, image-conditioned diffusion has expanded beyond image-to-image generation to video [5, 26, 34, 71] and 3D synthesis [45, 49, 79], broadening its applicability. 

Virtual try-on using generative models. Early virtual try-on systems primarily relied on warping-based methods [20, 28, 80], where the main challenge was to deform and align the garment image to match the target person’s pose and body shape. With the emergence of Generative Adversarial Networks (GANs) [22, 38, 39], a series of GAN-based approaches [1, 13, 21, 27, 36, 53, 83, 85, 90] achieved more realistic image synthesis through adversarial learning. However, these models still struggled to preserve garment details and maintain structural consistency. 

With the advancement of large-scale text-to-image diffusion models [4, 18, 62, 64, 66, 68], recent works [2, 14, 15, 24, 56, 69, 84, 87, 91, 92] have adopted pre-trained diffusion backbones as strong generative priors for virtual tryon. A key challenge remains: effectively guiding the model to learn the complex relationship between the garment and the target person. To address this, various techniques have been proposed to improve spatial correspondence and visual fidelity—such as auxiliary loss functions [41, 91], dual or reference networks [10, 14, 92], warping [24, 56, 90, 91], external image encoders [14, 41, 84], and spatial concatenation [15]. However, many of these approaches rely on weak 

2 

**==> picture [496 x 113] intentionally omitted <==**

Figure 2. Attention map comparison— CatVTON [15] shows dispersed attention unrelated to the query point, indicating weak spatial grounding. In contrast, our model produces sharply localized maps that align well with the corresponding garment regions, demonstrating stronger relational understanding. 

garment–person coupling or introduce additional components that may degrade generation quality. 

While most existing work focuses on virtual try-on, a few studies [78, 82] have explored virtual try-off, the inverse of virtual try-on, which aims to predict a clean garment image from a person wearing it. This task is inherently challenging due to occlusion, wrinkles, and deformation, and similarly requires strong relational modeling. We propose a unified DiT-based framework that jointly learns virtual try-on and try-off using a shared spatial conditioning layout. This design enables robust bidirectional garment–person modeling and remains scalable across diverse fashion imagery. 

## 3. Method 

In this section, we introduce our framework for virtual tryon and try-off using a unified diffusion transformer. We first present the preliminaries of our approach (Sec. 3.1), and then describe how we design a diffusion architecture that supports both tasks within a shared model (Sec. 3.2). Finally, we introduce inference-time techniques that further enhance output quality (Sec. 3.3). 

## 3.1. Preliminary 

We begin by reviewing conditional diffusion models in the context of virtual try-on, and motivating the need for strong garment–target correspondence to enable faithful synthesis. 

Diffusion models for virtual try-on. Diffusion models [32, 66, 73] generate images by denoising a sequence of latent representations corrupted over time. Given a latent , the network is trained to recover the clean signal or the noise , conditioned on external input . In general, the denoising step follows: 

**==> picture [149 x 10] intentionally omitted <==**

where �( denotes an integration step, such as DDIM [72], DPM-Solver [52], or deterministic alternatives like flow 

matching [48]. In virtual try-on, y typically includes a garment image and a masked person image, optionally with pose or parsing maps [8, 25]. We adopt a unified DiT [60] that handles both try-on and try-off tasks. Details follow in Sec. 3.2. 

Garment–target correspondence analysis. A key challenge in virtual try-on is aligning the garment structure with the target body while preserving visual fidelity. We assess this correspondence by analyzing transformer attention maps at specific spatial queries. As shown in Fig.2, CatVTON [15] exhibits dispersed attention across unrelated areas, indicating weak spatial grounding, which can reduce structural fidelity and impair detail preservation. In contrast, attention maps produced by our model are sharply localized and semantically consistent, suggesting stronger garment–target alignment. This motivates our bidirectional training strategy, where jointly learning try-on and try-off within a single model improves spatial alignment. 

## 3.2. Try-On & Off via Unified Transformer 

As shown in Fig. 3, we use a horizontally concatenated layout that places the garment image and the person image side by side, allowing the model to process both conditioning and generation regions within a unified input. While this setup enables joint processing of conditioning and generation regions, existing methods often reconstruct the conditioning region without masking, turning it into a trivial task. To address this, we extend the setup for bidirectional learning: by selectively masking the garment region while showing the dressed person, we reinterpret the task as virtual try-off, allowing the model to infer the garment from contextual cues. 

Pipeline. Let denote the standalone garment image, and the target person image. We construct the input by horizontally concatenating the two, forming a combined image . Task-specific inpainting regions are defined by a binary mask 1 , applied in the im- 

3 

**==> picture [496 x 194] intentionally omitted <==**

Figure 3. Overview of pipeline. Voost enables bidirectional virtual try-on and try-off with a unified diffusion transformer for scalable learning. 

age space prior to encoding. For the try-on task, the mask is defined as on[,][where] on[masks][out] the garment area in the person image while preserving the garment image[.][For][the][try-off][task,][we][use] = [ j 0, masking the entire garment image and leaving the person image unmasked. The masked input is obtained as X masked[ X][, where][denotes element-wise] multiplication. Our architecture follows a latent diffusion framework [4, 18], where denoising is performed in the latent space. 

The full and masked images are encoded by a frozen encoder into latent representations and c masked[.] The mask is downsampled via a pixelunshuffle [70] operation to match the latent resolution, producing c[.][These][three][components][ z] c[,][and] c[are] concatenated along the channel dimension, passed through a token embedding layer, and then used as input to the DiT backbone. We define the task token mode category as the concatenation of a mode token mode on off, which specifies the generation direction, and a category token category upper lower full, which encodes the garment type. The combined token is passed to the transformer as additional conditioning. 

Dynamic layout. Our framework supports dynamic input layouts by leveraging the token-based representation of vision transformers [16, 17, 60], which enables the processing of images with varying spatial resolutions and aspect ratios without requiring fixed dimensions. In our pipeline, the concatenated garment–person image is tokenized and flattened, allowing the model to accommodate inputs of arbitrary shape. Consequently, the aforementioned and need not remain fixed throughout the pipeline. To support batch training with variable-length sequences, we in- 

sert padding tokens as needed and standardize the sequence length to a fixed maximum N . Additionally, we also use Rotary Position Embedding (RoPE) [74] to robustly handle inputs with diverse aspect ratios and spatial configurations. This design provides the foundation for scalable virtual tryon, enabling Voost to handle diverse poses, aspect ratios, and compositions within a single batch. By jointly modeling task direction and garment category, it supports flexible multitask learning and mitigates inductive biases introduced by the rigid preprocessing of fixed-resolution pipelines. 

Training strategy. We adopt a flow matching formulation [48], where the model learns a time-dependent velocity field that transports samples from the data distribution to noise along a continuous path. Let be a data latent and �N I be a sampled noise latent. We define a trajectory z between them and train the denoising model to predict the velocity at each intermediate point. The unified training objective is given by: 

**==> picture [216 x 29] intentionally omitted <==**

In our case, we adopt the rectified flow formulation [18, 50], where the trajectory is defined as a straight line between and z : 

**==> picture [168 x 10] intentionally omitted <==**

Note that denotes continuous time during training, but serves as a discrete timestep index during inference. This reduces the training target to a constant displacement vector, simplifying optimization and aligning with the rectified flow formulation. To retain the strong generative prior of 

4 

**==> picture [237 x 144] intentionally omitted <==**

Figure 4. Impact of temperature scaling. Adaptive temperature scaling enhances visual detail by adjusting attention behavior under varying spatial proportions of mask and garment regions. 

the pretrained DiT backbone while adapting to the try-on and try-off setting, we finetune only the attention modules within each transformer block, freezing the rest. Adapting only attention layers enables precise garment–person reasoning in try-on and try-off, without sacrificing the generative prior learned from large-scale diffusion. A detailed analysis of this design choice is provided in the supplementary material. 

## 3.3. Inference-time Refinement 

Token-length and mask-aware temperature scaling. While our framework flexibly supports varying spatial aspect ratios during training, real-world inference often involves inputs with resolutions or mask proportions that differ from those seen during training. Such shifts can lead to suboptimal attention behavior, especially when inputs fall outside the training distribution. To address this, we introduce a dynamic temperature scaling scheme that adapts attention sharpness to both absolute token length changes and relative spatial imbalance between the masked and conditioning regions. We define the adjusted attention temperature as: 

**==> picture [210 x 51] intentionally omitted <==**

Here, is the key, query, value vector dimension used in scaled dot-product attention [77]; infer[and] train[are] the total token counts at inference and training; mask[and] garment[denote][the][masked][and][garment][region][tokens.] The hyperparameters and control the influence of the global and relative scaling terms, respectively, while is a fixed positive constant that stabilizes the computation when mask[is][small.][The][global][term][stabilizes][attention][across] 

**==> picture [238 x 229] intentionally omitted <==**

token lengths, while the relative term adapts to spatial imbalance between masked and garment regions. This improves robustness to layout shifts and maintains consistent attention, particularly when the mask covers a small proportion relative to the garment (Fig. 4). 

Self-correction with unified transformer. Our unified model enables both try-on and try-off generation under a shared concatenated layout. We leverage this dual capability at inference time through a self-correction mechanism (Alg. 1) that promotes consistency between the generated output and the conditioning garment. The core intuition is that a faithful try-on result should implicitly preserve sufficient information to recover the original garment via a reverse (try-off) process. At a designated timestep corr during denoising, the model predicts the dressed person image[on] from the current latent , using the try-on task token on (i.e., the mode component of the full task token mode category[).][This intermediate prediction is then] used as a conditioning input to perform a reverse try-off pass, yielding a reconstructed garment[off] . We compute reconstruction error between[off] and z c, using its gradient to update via backpropagation. This refinement is repeated times to gradually align the generation with the conditioning signal. 

## 4. Experiments 

## 4.1. Datasets and experimental setup 

We evaluate our method on two standard benchmarks: DressCode [55] and VITON-HD [13] datasets, using both qualitative and quantitative metrics. Each dataset con- 

5 

**==> picture [496 x 389] intentionally omitted <==**

Figure 5. Qualitative comparison of try-on results with existing try-on methods [14, 15, 41, 91]. Best viewed in color and under zoom. 

tains high-resolution image pairs of in-shop garments and corresponding person images. We use the official train/validation/test splits provided by both datasets. All outputs are generated at a resolution of . Our model is trained on NVIDIA H100 GPUs and evaluated on a single NVIDIA A100 GPU. To assess generalization, we also present qualitative results on in-the-wild images. 

## 4.2. Qualitative comparison 

Fig. 5 and Fig. 6 show qualitative comparisons between our method and state-of-the-art approaches [14, 15, 41, 91] on the VTON task. Given the increasing interest in practical applications of virtual try-on, we also include comparisons with commercial models [23, 43, 57, 58, 61] in Fig. 9. 

Importantly, our unified model handles try-on and tryoff with a single set of parameters, without task-specific re- 

training. Try-off results are shown in Fig. 7 and Fig. 11, where we compare our outputs to those of existing methods [78, 82]. Across both tasks, our method produces more coherent and photorealistic results, benefiting from shared garment-body reasoning and joint training. 

In particular, our model demonstrates strong robustness on in-the-wild images with diverse poses, backgrounds, and lighting conditions, consistently outperforming existing methods in these challenging scenarios. We provide additional qualitative results and comparisons in the supplementary material. 

## 4.3. Quantitative results 

We evaluate visual fidelity and structural consistency using standard metrics. For realism, we report Fr´echet Inception Distance (FID) [31] and Kernel Inception Distance (KID) [3]. To assess structural consistency, we use 

6 

**==> picture [496 x 597] intentionally omitted <==**

Figure 6. Additional qualitative comparison of try-on results with other methods. Best viewed in color and under zoom. 

7 

|Methods|VITON-HD [13]<br>Paired<br>SSIM<br>LPIPS<br>FID<br>KID<br>FID|Unpaired<br>KID<br>SSIM|DressCode [55]<br>Paired<br>Unpaired<br>LPIPS<br>FID<br>KID<br>FID<br>KID|
|---|---|---|---|
|StableVITON [41]<br>OOTDiffusion [84]<br>IDM-VTON [14]<br>CatVTON [15]<br>Leffa [91]|0.867<br>0.084<br>6.851<br>1.255<br>0.851<br>0.096<br>6.520<br>0.896<br>0.881<br>0.079<br>6.343<br>1.322<br>0.869<br>0.097<br>6.141<br>0.964<br>0.872<br>0.081<br>6.310<br>1.208|9.591<br>1.451<br>9.672<br>1.206<br>9.613<br>1.639<br>9.141<br>1.267<br>9.442<br>1.249|0.905<br>0.107<br>4.482<br>1.530<br>6.728<br>1.742<br>0.898<br>0.073<br>3.953<br>0.720<br>6.704<br>1.863<br>0.923<br>0.048<br>3.801<br>1.201<br>5.621<br>1.554<br>0.901<br>0.071<br>3.283<br>0.670<br>5.424<br>1.549<br>0.911<br>0.060<br>3.651<br>0.709<br>5.462<br>1.528|
|Ours (VTON-only)|0.868<br>0.079<br>5.804<br>0.618|9.112<br>1.051|0.910<br>0.052<br>3.043<br>0.565<br>5.298<br>1.132|
|Ours (w/o scaling)<br>Ours|0.885<br>0.072<br>5.242<br>0.460<br>0.898<br>0.056<br>5.269<br>0.404|8.991<br>0.912<br>8.982<br>0.899|0.925<br>0.046<br>2.774<br>0.390<br>5.030<br>0.808<br>0.933<br>0.044<br>2.787<br>0.377<br>5.081<br>0.787|



Table 1. Quantitative results on VITON-HD [13] and DressCode [55] for the try-on task. We report both paired and unpaired evaluation results across benchmarks. Our unified dual-task model (Voost) consistently outperforms all baselines, including the single-task (VTONonly) model. Bold and underline indicate the best and second-best scores, respectively. 

**==> picture [237 x 260] intentionally omitted <==**

Figure 7. Qualitative comparison of try-off results with other methods. Best viewed in color and under zoom. 

**==> picture [233 x 222] intentionally omitted <==**

Figure 8. Effectiveness of the self-correction mechanism. These examples demonstrate improved garment–body alignment and reduction of visual artifacts when the self-correction module is applied. 

LPIPS [89] and SSIM [81]. As shown in Table. 1 and Table. 2, our model outperforms existing methods across all metrics. 

## 4.4. User study 

While virtual try-on allows multiple plausible outputs, preserving garment characteristics and structure remains crucial for realism. However, standard metrics only partially capture key aspects of try-on quality, including garment realism and alignment consistency. To better evaluate perceptual fidelity, we conducted a user study comparing our 

method with existing approaches. Participants evaluated 50 samples randomly selected from DressCode, VITON-HD, and in-the-wild images, judging three aspects: photorealism, garment detail, and garment structure. For each sample, they selected the most compelling result in each category. Full details on the criteria, survey questions, and interface are provided in the supplementary material. As shown in Fig. 10, our method was consistently preferred across all evaluation criteria, demonstrating its superiority in visual realism and garment preservation. 

8 

**==> picture [496 x 341] intentionally omitted <==**

Figure 9. Qualitative comparison of try-on results with other commercial models [23, 43, 57, 58, 61]. Best viewed in color and under zoom. 

||TryOffDiff [78]|TryOffAnyOne [82]|Ours (VTOFF-only)|Ours|
|---|---|---|---|---|
|FID|28.25|25.20|12.88|10.06|
|KID|11.42|6.98|3.54|2.48|



Table 2. Quantitative results on VITON-HD for the try-off task. 

## 4.5. Ablation study 

Effect of dual-task training. We examine the benefit of jointly training a single model for both try-on and try-off, compared to training separate models for each task. As shown in Table. 1 and Table. 2, our dual-task model consistently outperforms its single-task counterparts in all metrics. This indicates that joint training fosters a more generalizable garment–person correspondence prior, improving performance in both directions. As illustrated in Fig. 12, the attention maps of the joint model show sharper and more accurate garment-to-person correspondences, with regions in the person attending more precisely to their counterparts in the garment, highlighting improved spatial alignment. Effect of inference-time refinement. We evaluate the two 

**==> picture [237 x 105] intentionally omitted <==**

Figure 10. User study results. Comparing our method (Voost) with other baselines on three criteria: photorealism, garment detail, and garment structure. Across all categories, Voost received the highest top-1 selection rates, indicating clear user preference for its visual realism and garment fidelity. 

inference-time strategies introduced in Sec. 3.3: temperature scaling and self-correction. As shown in Table. 1, temperature scaling provides consistent improvements on benchmarks like VITON-HD and DressCode, which feature relatively uniform image compositions and garment place- 

9 

**==> picture [237 x 311] intentionally omitted <==**

Figure 11. Comparison results on in-the-wild images (Row 1–3) for the virtual try-off task. Unlike existing methods [78, 82], our method supports diverse garment types including upper, lower, and dresses (Last row). 

ments. Its effectiveness becomes even more evident in challenging real-world scenarios. As illustrated in Fig. 4, it helps preserve garment fidelity when the masked region is small or spatially imbalanced. The self-correction mechanism, while not included in the main quantitative benchmarks due to its selective and user-invoked nature, offers a practical enhancement in difficult cases. As shown in Fig. 8, it can successfully recover garment structure and detail when initial generations fall short, improving perceptual realism without additional model retraining. 

Effect of trainable parameters. We compare training strategies with different subsets of trainable weights. As shown in Table 3, our attention-only tuning achieves the best overall performance, outperforming full fine-tuning, single-block training, and LoRA [33]. It effectively captures garment–person correspondence while significantly reducing training cost. Fig. 13 further highlights the qualitative advantage: attention-only tuning yields sharper, more coherent outputs with fewer artifacts. The supplementary material provides a detailed analysis of this design choice, highlighting sensitivity studies that justify our focus on at- 

**==> picture [237 x 93] intentionally omitted <==**

Figure 12. Attention map comparison of single-task and dual-task models. Voost (dual-task) attends more precisely to the relevant garment region based on the query point. 

**==> picture [237 x 134] intentionally omitted <==**

Figure 13. Qualitative comparison of training strategies. 

||Training strategy<br># params.<br>SSIM<br>LPIPS<br>FID<br>KID<br>Full<br>11.9B<br>0.875<br>0.081<br>6.351<br>0.886<br>Single DiT Blocks<br>5.38B<br>0.872<br>0.078<br>5.975<br>0.634|
|---|---|
||Attention-only (ours)<br>2.69B<br>0.899<br>0.056<br>5.269<br>0.404<br> <br>|
||LoRA (r=128)<br>359M<br>0.843<br>0.108<br>6.668<br>0.906|



Table 3. Quantitative comparison of training strategies with different updated layers on VITON-HD [13]. 

tention layers. 

## 5. Conclusion 

In this work, we proposed Voost, a unified and scalable framework that jointly models virtual try-on and try-off within a single diffusion transformer. By formulating the two tasks as bidirectional counterparts, Voost enables mutual supervision without relying on task-specific architectures, auxiliary losses, or additional labels. To further improve robustness and consistency, we introduced two inference-time techniques: attention temperature scaling and self-corrective sampling. Comprehensive experiments demonstrate that Voost consistently surpasses strong baselines on both try-on and try-off benchmarks, achieving state-of-the-art performance in alignment, visual fidelity, and generalization. These results highlight the effectiveness of unified diffusion modeling for fashion synthesis and suggest a promising direction for integrated human–garment 

10 

understanding. 

Limitations and future work. While our model is capable of producing photorealistic results by jointly learning try-on and try-off through bidirectional attention, precise control over the garment’s fit remains somewhat ambiguous due to the lack of explicit structural or sizing information. 

In future work, we plan to incorporate additional cues such as body measurements or garment metadata to improve controllability. More broadly, the strong image-level foundation of Voost makes it well-suited for downstream extensions such as video-based [29, 37] or 3D-based [7, 30, 88] synthesis, where consistent and faithful garment–person interaction remains essential yet challenging. 

## References 

- [1] Shuai Bai, Huiling Zhou, Zhikang Li, Chang Zhou, and Hongxia Yang. Single stage virtual try-on via deformable attention flows. In European Conference on Computer Vision (ECCV), 2022. 2 

- [2] Alberto Baldrati, Davide Morelli, Giuseppe Cartella, Marcella Cornia, Marco Bertini, and Rita Cucchiara. Multimodal garment designer: Human-centric latent diffusion models for fashion image editing. In Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 2 

- [3] Mikołaj Bi´nkowski, Danica J Sutherland, Michael Arbel, and Arthur Gretton. Demystifying mmd gans. arXiv preprint arXiv:1801.01401, 2018. 6 

- [4] Black Forest Labs. Flux. https://github.com/ black-forest-labs/flux, 2024. 2, 4 

- [5] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 2 

- [6] Mingdeng Cao, Xintao Wang, Zhongang Qi, Ying Shan, Xiaohu Qie, and Yinqiang Zheng. MasaCtrl: Tuning-free mutual self-attention control for consistent image synthesis and editing. In Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 2 

- [7] Yukang Cao, Masoud Hadi, Liang Pan, and Ziwei Liu. GSVTON: Controllable 3d virtual try-on with gaussian splatting. arXiv, 2024. 11 

- [8] Zhe Cao, Tomas Simon, Shih-En Wei, and Yaser Sheikh. Realtime multi-person 2d pose estimation using part affinity fields. In Conference on Computer Vision and Pattern Recognition (CVPR), 2017. 3 

- [9] Mathilde Caron, Hugo Touvron, Ishan Misra, Herv´e J´egou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In International Conference on Computer Vision (ICCV), 2021. 2 

- [10] Mengting Chen, Xi Chen, Zhonghua Zhai, Chen Ju, Xuewen Hong, Jinsong Lan, and Shuai Xiao. Wear-any-way: Manipulable virtual try-on via sparse correspondence alignment. In European Conference on Computer Vision (ECCV), 2024. 2 

- [11] Xi Chen, Yutong Feng, Mengting Chen, Yiyang Wang, Shilong Zhang, Yu Liu, Yujun Shen, and Hengshuang Zhao. Zero-shot image editing with reference imitation. In Advances in Neural Information Processing Systems (NeurIPS), 2024. 2 

- [12] Xi Chen, Lianghua Huang, Yu Liu, Yujun Shen, Deli Zhao, and Hengshuang Zhao. Anydoor: Zero-shot object-level image customization. In Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 2 

- [13] Seunghwan Choi, Sunghyun Park, Minsoo Lee, and Jaegul Choo. VITON-HD: High-resolution virtual try-on via misalignment-aware normalization. In Conference on Computer Vision and Pattern Recognition (CVPR), 2021. 2, 5, 8, 10, 15, 17 

- [14] Yisol Choi, Sangkyung Kwak, Kyungmin Lee, Hyungwon Choi, and Jinwoo Shin. Improving diffusion models for authentic virtual try-on in the wild. In European Conference on Computer Vision (ECCV), 2024. 2, 6, 8 

- [15] Zheng Chong, Xiao Dong, Haoxiang Li, Shiyue Zhang, Wenqing Zhang, Xujie Zhang, Hanqing Zhao, and Xiaodan Liang. CatVTON: Concatenation is all you need for virtual try-on with diffusion models. In International Conference on Learning Representations (ICLR), 2025. 2, 3, 6, 8 

- [16] Mostafa Dehghani, Basil Mustafa, Josip Djolonga, Jonathan Heek, Matthias Minderer, Mathilde Caron, Andreas Steiner, Joan Puigcerver, Robert Geirhos, Ibrahim M Alabdulmohsin, et al. Patch n’pack: Navit, a vision transformer for any aspect ratio and resolution. In Advances in Neural Information Processing Systems (NeurIPS), 2023. 4 

- [17] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations (ICLR), 2021. 4 

- [18] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, Kyle Lacey, Alex Goodwin, Yannik Marek, and Robin Rombach. Scaling rectified flow transformers for high-resolution image synthesis. In International Conference on Machine Learning (ICML), 2024. 2, 4, 15 

- [19] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit Haim Bermano, Gal Chechik, and Daniel Cohen-or. An image is worth one word: Personalizing text-to-image generation using textual inversion. In International Conference on Learning Representations (ICLR), 2023. 2 

- [20] Yuying Ge, Yibing Song, Ruimao Zhang, Chongjian Ge, Wei Liu, and Ping Luo. Parser-free virtual try-on via distilling appearance flows. In Conference on Computer Vision and Pattern Recognition (CVPR), 2021. 2 

- [21] Yuying Ge, Yibing Song, Ruimao Zhang, Chongjian Ge, Wei Liu, and Ping Luo. Parser-free virtual try-on via distilling appearance flows. In European Conference on Computer Vision (ECCV), 2021. 2 

- [22] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and 

11 

Yoshua Bengio. Generative adversarial networks. In Advances in Neural Information Processing Systems (NeurIPS), 2014. 2 

- [23] Google DeepMind. Gemini 2.0 flash - multimodal ai model. https://gemini.google.com/, 2025. 6, 9 

- [24] Junhong Gou, Siyu Sun, Jianfu Zhang, Jianlou Si, Chen Qian, and Liqing Zhang. Taming the power of diffusion models for high-quality virtual try-on with appearance flow. In ACM International Conference on Multimedia (ACMMM), 2023. 2 

- [25] Rıza Alp G¨uler, Natalia Neverova, and Iasonas Kokkinos. Densepose: Dense human pose estimation in the wild. In Conference on Computer Vision and Pattern Recognition (CVPR), 2018. 3, 15 

- [26] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized textto-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725, 2023. 2 

- [27] Xintong Han, Zuxuan Wu, Zhe Wu, Ruichi Yu, and Larry S. Davis. VITON: An image-based virtual try-on network. In Conference on Computer Vision and Pattern Recognition (CVPR), 2018. 2 

- [28] Xintong Han, Xiaojun Hu, Weilin Huang, and Matthew R Scott. Clothflow: A flow-based model for clothed person generation. In Conference on Computer Vision and Pattern Recognition (CVPR), 2019. 2 

- [29] Zijian He, Peixin Chen, Guangrun Wang, Guanbin Li, Philip HS Torr, and Liang Lin. Wildvidfit: Video virtual try-on in the wild via image-based controlled diffusion models. In European Conference on Computer Vision (ECCV), 2024. 11 

- [30] Zijian He, Yuwei Ning, Yipeng Qin, Guangrun Wang, Sibei Yang, Liang Lin, and Guanbin Li. VTON 360: High-fidelity virtual try-on from any viewing direction. In Conference on Computer Vision and Pattern Recognition (CVPR), 2025. 11 

- [31] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. GANs trained by a two time-scale update rule converge to a local nash equilibrium. In Advances in Neural Information Processing Systems (NeurIPS), 2017. 6 

- [32] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems (NeurIPS), 2020. 3 

- [33] Edward J Hu, yelong shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations (ICLR), 2022. 10 

- [34] Li Hu. Animate anyone: Consistent and controllable imageto-video synthesis for character animation. In Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 2 

- [35] Lianghua Huang, Wei Wang, Zhi-Fan Wu, Yupeng Shi, Huanzhang Dou, Chen Liang, Yutong Feng, Yu Liu, and Jingren Zhou. In-context lora for diffusion transformers. arXiv, 2024. 2 

- [36] Thibaut Issenhuth, J´er´emie Mary, and Cl´ement Calauzenes. Do not mask what you do not need to mask: a parser-free 

virtual try-on. In European Conference on Computer Vision (ECCV), 2020. 2 

- [37] Johanna Karras, Yingwei Li, Nan Liu, Luyang Zhu, Innfarn Yoo, Andreas Lugmayr, Chris Lee, and Ira KemelmacherShlizerman. Fashion-VDM: Video diffusion model for virtual try-on. In In ACM SIGGRAPH Asia, 2024. 11 

- [38] Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In Conference on Computer Vision and Pattern Recognition (CVPR), 2019. 2 

- [39] Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Analyzing and improving the image quality of stylegan. In Conference on Computer Vision and Pattern Recognition (CVPR), 2020. 2 

- [40] Rawal Khirodkar, Timur Bagautdinov, Julieta Martinez, Su Zhaoen, Austin James, Peter Selednik, Stuart Anderson, and Shunsuke Saito. Sapiens: Foundation for human vision models. arXiv preprint arXiv:2408.12569, 2024. 15 

- [41] Jeongho Kim, Guojung Gu, Minho Park, Sunghyun Park, and Jaegul Choo. StableVITON: Learning semantic correspondence with latent diffusion model for virtual try-on. In Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 2, 6, 8 

- [42] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Doll´ar, and Ross Girshick. Segment anything. arXiv:2304.02643, 2023. 15 

- [43] Kolors-Team. Kolors: Effective training of diffusion model for photorealistic text-to-image synthesis. arXiv preprint, 2024. 6, 9 

- [44] Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Multi-concept customization of text-to-image diffusion. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1931–1941, 2023. 16 

- [45] Jeong-gi Kwak, Erqun Dong, Yuhe Jin, Hanseok Ko, Shweta Mahajan, and Kwang Moo Yi. Vivid-1-to-3: Novel view synthesis with video diffusion models. In Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 2 

- [46] Peike Li, Yunqiu Xu, Yunchao Wei, and Yi Yang. Selfcorrection for human parsing. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(6):3260–3271, 2020. 15 

- [47] Yijun Li, Richard Zhang, Jingwan Lu, and Eli Shechtman. Few-shot image generation with elastic weight consolidation. arXiv preprint arXiv:2012.02780, 2020. 16 

- [48] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv, 2023. 3, 4 

- [49] Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, and Carl Vondrick. Zero-1-to-3: Zero-shot one image to 3d object. In Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 2 

- [50] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv, 2022. 4 

12 

- [51] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations (ICLR), 2019. 15 

- [52] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. DPM-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. In NeurIPS, 2022. 3 

- [53] Yifang Men, Yiming Mao, Yuning Jiang, Wei-Ying Ma, and Zhouhui Lian. Controllable person image synthesis with attribute-decomposed gan. In Conference on Computer Vision and Pattern Recognition (CVPR), 2020. 2 

- [54] Maxwell Meyer and Jack Spruyt. Ben: Using confidenceguided matting for dichotomous image segmentation. arXiv preprint arXiv:2501.06230, 2025. 15 

- [55] Davide Morelli, Matteo Fincato, Marcella Cornia, Federico Landi, Fabio Cesari, and Rita Cucchiara. Dress code: Highresolution multi-category virtual try-on. In Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 5, 8, 15, 17 

- [56] Davide Morelli, Alberto Baldrati, Giuseppe Cartella, Marcella Cornia, Marco Bertini, and Rita Cucchiara. LaDIVTON: Latent Diffusion Textual-Inversion Enhanced Virtual Try-On. In ACM International Conference on Multimedia (ACMMM), 2023. 2 

- [57] OmniousAI. Vella 1.0. https://vellaml.com/, 2025. 6, 9 

- [58] OpenAI. Chatgpt-4o: Multimodal ai model. https:// openai.com/index/introducing- 4o- imagegeneration/, 2025. 6, 9 

- [59] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel HAZIZA, Francisco Massa, Alaaeldin El-Nouby, Mido Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Herve Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. DINOv2: Learning robust visual features without supervision. Transactions on Machine Learning Research, 2024. 2 

- [60] William Peebles and Saining Xie. Scalable diffusion models with transformers. In International Conference on Computer Vision (ICCV), 2023. 3, 4 

- [61] Pic Copilot. Pic copilot - virtual try-on tool. https:// www.piccopilot.com/, 2024. 6, 9 

- [62] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. SDXL: Improving latent diffusion models for high-resolution image synthesis. In International Conference on Learning Representations (ICLR), 2024. 2 

- [63] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning (ICML), 2021. 2 

- [64] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. 

Zero-shot text-to-image generation. In International Conference on Machine Learning (ICML), 2021. 2 

- [65] Jeff Rasley, Samyam Rajbhandari, Olatunji Ruwase, and Yuxiong He. Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters. In Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery & data mining, 2020. 15 

- [66] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 2, 3 

- [67] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023. 2 

- [68] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. In Advances in Neural Information Processing Systems (NeurIPS), 2022. 2 

- [69] Fei Shen, Xin Jiang, Xin He, Hu Ye, Cong Wang, Xiaoyu Du, Zechao Li, and Jinhui Tang. Imagdressing-v1: Customizable virtual dressing. In AAAI Conference on Artificial Intelligence (AAAI), 2025. 2 

- [70] Wenzhe Shi, Jose Caballero, Ferenc Husz´ar, Johannes Totz, Andrew P Aitken, Rob Bishop, Daniel Rueckert, and Zehan Wang. Real-time single image and video super-resolution using an efficient sub-pixel convolutional neural network. In Conference on Computer Vision and Pattern Recognition (CVPR), 2016. 4 

- [71] Xiaoyu Shi, Zhaoyang Huang, Fu-Yun Wang, Weikang Bian, Dasong Li, Yi Zhang, Manyuan Zhang, Ka Chun Cheung, Simon See, Hongwei Qin, et al. Motion-i2v: Consistent and controllable image-to-video generation with explicit motion modeling. In In ACM SIGGRAPH, 2024. 2 

- [72] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In International Conference on Learning Representations (ICLR), 2021. 3 

- [73] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations (ICLR), 2021. 3 

- [74] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024. 4 

- [75] Zhenxiong Tan, Songhua Liu, Xingyi Yang, Qiaochu Xue, and Xinchao Wang. Ominicontrol: Minimal and universal control for diffusion transformer. arXiv preprint arXiv:2411.15098, 2024. 2 

- [76] Linrui Tian, Qi Wang, Bang Zhang, and Liefeng Bo. Emo: Emote portrait alive generating expressive portrait videos with audio2video diffusion model under weak conditions. In European Conference on Computer Vision (ECCV), 2024. 2 

13 

- [77] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems (NeurIPS), 2017. 5 

- [78] Riza Velioglu, Petra Bevandic, Robin Chan, and Barbara Hammer. TryOffDiff: Virtual-try-off via high-fidelity garment reconstruction using diffusion models. arXiv, 2024. 2, 3, 6, 9, 10 

- [79] Vikram Voleti, Chun-Han Yao, Mark Boss, Adam Letts, David Pankratz, Dmitry Tochilkin, Christian Laforte, Robin Rombach, and Varun Jampani. Sv3d: Novel multi-view synthesis and 3d generation from a single image using latent video diffusion. In European Conference on Computer Vision (ECCV), 2024. 2 

      - VTON: Towards general purpose virtual try-on via collaborative local-flow global-parsing learning. In Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 2 

   - [91] Zijian Zhou, Shikun Liu, Xiao Han, Haozhe Liu, Kam Woh Ng, Tian Xie, Yuren Cong, Hang Li, Mengmeng Xu, JuanManuel P´erez-R´ua, Aditya Patel, Tao Xiang, Miaojing Shi, and Sen He. Learning flow fields in attention for controllable person image generation. In Conference on Computer Vision and Pattern Recognition (CVPR), 2025. 2, 6, 8 

   - [92] Luyang Zhu, Dawei Yang, Tyler Zhu, Fitsum Reda, William Chan, Chitwan Saharia, Mohammad Norouzi, and Ira Kemelmacher-Shlizerman. TryOnDiffusion: A tale of two unets. In Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 2 

- [80] Bochao Wang, Huabin Zheng, Xiaodan Liang, Yimin Chen, Liang Lin, and Meng Yang. Toward characteristicpreserving image-based virtual try-on network. In European Conference on Computer Vision (ECCV), 2018. 2 

- [81] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600–612, 2004. 8 

- [82] Ioannis Xarchakos and Theodoros Koukopoulos. TryOffAnyone: Tiled cloth generation from a dressed person. arXiv, 2025. 2, 3, 6, 9, 10 

- [83] Zhenyu Xie, Zaiyu Huang, Fuwei Zhao, Haoye Dong, Michael Kampffmeyer, and Xiaodan Liang. Towards scalable unpaired virtual try-on via patch-routed spatiallyadaptive gan. In Advances in Neural Information Processing Systems (NeurIPS), 2021. 2 

- [84] Yuhao Xu, Tao Gu, Weifeng Chen, and Chengcai Chen. OOTDiffusion: Outfitting fusion based latent diffusion for controllable virtual try-on. In AAAI Conference on Artificial Intelligence (AAAI), 2025. 2, 8 

- [85] Han Yang, Ruimao Zhang, Xiaobao Guo, Wei Liu, Wangmeng Zuo, and Ping Luo. Towards photo-realistic virtual try-on by adaptively generating-preserving image content. In Conference on Computer Vision and Pattern Recognition (CVPR), 2020. 2 

- [86] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. IPAdapter: Text compatible image prompt adapter for text-toimage diffusion models. 2023. 2 

- [87] Jianhao Zeng, Dan Song, Weizhi Nie, Hongshuo Tian, Tongtong Wang, and An-An Liu. CAT-DM: Controllable accelerated virtual try-on with diffusion model. In Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 2 

- [88] Nannan Zhang, Yijiang Li, Dong Du, Zheng Chong, Zhengwentai Sun, Jianhao Zeng, Yusheng Dai, Zhengyu Xie, Hairui Zhu, and Xiaoguang Han. Robust-mvton: Learning cross-pose feature alignment and fusion for robust multiview virtual try-on. In Conference on Computer Vision and Pattern Recognition (CVPR), 2025. 11 

- [89] Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. arXiv, 2018. 8 

- [90] Xie Zhenyu, Huang Zaiyu, Dong Xin, Zhao Fuwei, Dong Haoye, Zhang Xijin, Zhu Feida, and Liang Xiaodan. GP- 

14 

## Voost: A Unified and Scalable Diffusion Transformer for Bidirectional Virtual Try-On and Try-Off 

## Supplementary Material 

## 6. Implementation Details 

## 6.1. Dataset composition 

We train our model on a combination of VITON-HD [13], DressCode [55], and an additional curated set of real-world images. The latter consists of (1) high-resolution photographs captured in-house under controlled conditions and (2) in-the-wild fashion images from online sources. All external data were manually filtered to ensure high garment visibility, realistic lighting, and diverse poses. 

Our dataset maintains a reasonably diverse distribution across clothing types and subject compositions. Among all garment inputs, 52.1% are tops, 24.7% are bottoms, and 23.2% are full-body outfits. In terms of person image composition, 61.4% depict women and 38.6% depict men. Based on framing, 45.3% are upper-body shots, 31.8% fullbody, and 22.9% lower-body. 

Unlike previous works that operate on fixed-size inputs, our model supports variable aspect ratios during both training and inference (see Section 3.2). Accordingly, the dataset includes a diverse range of aspect ratios. The most common format is 3:4 (width:height), followed by 1:1, 2:3, 1:2, and 1:3, with a smaller portion falling outside these standard categories. This diversity enables our model to generalize well to in-the-wild scenarios with arbitrary image dimensions. 

## 6.2. Mask augmentation 

To prevent overfitting and improve generalization, we apply targeted augmentations to the agnostic masks, using distinct strategies for try-on and try-off tasks. 

For try-on, we ensure that all garment regions in the person image are fully masked, while explicitly preserving facial and hand regions. To obtain these masks, we leverage state-of-the-art human parsing and pose estimation models [25, 40, 42, 46]. During training, we generate diverse masks of varying shapes and sizes that perfectly occlude the clothing area but retain key human features. This encourages the model to rely entirely on the conditioning garment image rather than residual cues in the person input, significantly improving silhouette realism and garment fit, especially for longer garments. 

For try-off, we use a dichotomous segmentation model [54] to isolate the garment region, and apply masks of varying sizes—from tight garment-specific masks to nearly full-image masks. This approach forces the model to infer plausible underlying appearances across a range 

of masking levels, enhancing robustness under challenging conditions. 

## 6.3. Optimization 

We use the AdamW [51] optimizer with a learning rate of and a weight decay of 0.001. The model is trained with a batch size of 128 using the DeepSpeed ZeRO-2 [65] optimizer state partitioning strategy for approximately 48 single-H100 GPU days. 

## 6.4. Inference 

At test time, we employ a flow-matching-based Euler scheduler introduced in Stable Diffusion 3 [18], using exactly 28 sampling steps. All qualitative results shown in the supplementary figures were generated using this consistent inference setup. 

## 6.5. Inference Technique Details 

Temperature Scaling. We use and in 

our temperature scaling formula. The value of was empirically chosen based on garment-to-mask ratios aggregated over a large number of training samples, capturing typical spatial layouts seen during training. 

Self-Correction. For self-corrective sampling, we apply updates at two timesteps, and , based on a total denoising schedule of . These correspond to early and mid-to-late stages of the denoising process, chosen to target complementary aspects: = 5 primarily influences coarse shape, while t contributes to fine detail refinement. At the shape correction step ( ), we use the same mask as the standard try-off setting, M = [ j 0, encouraging the model to recover overall garment length and structure from the try-on result. For detail correction ( ), we instead use a garment-specific mask obtained via segmentation models [40, 42, 54] of the input garment image. This focuses the gradient signal on texture and detail consistency, while avoiding penalization for plausible geometric differences. We set the number of refinement iterations to R = 5 for each correction step. 

## 7. Rationale for Attention-Only Training 

To adapt a pretrained DiT for virtual try-on and try-off under our concatenation-based setup, we fine-tune only the attention modules while freezing all other parameters—a choice grounded in both empirical evidence and task-specific insight. 

15 

**==> picture [496 x 437] intentionally omitted <==**

Figure 14. User study interface for evaluating photorealism, design detail, and garment structure across five try-on results. 

**==> picture [213 x 76] intentionally omitted <==**

Figure 15. Layer-wise parameter update magnitudes during full parameter tuning. 

Following [44, 47], we analyze parameter sensitivity by fully fine-tuning the model on concatenated inputs[ and computing the normalized weight update for each] 

layer as 

**==> picture [77 x 10] intentionally omitted <==**

where[and][denote the pretrained and updated weights,] respectively, and is a general vector norm. As shown in Fig. 15, attention layers exhibit the most significant changes, underscoring their key role in garment-to-person alignment. 

In contrast, full-model fine-tuning often leads to overfitting, introducing spurious accessories or artifacts unrelated to the garment. This suggests that unconstrained updates may distort the conditioning signal and reduce generation fidelity. 

By limiting updates to attention layers, we preserve the 

16 

pretrained diffusion prior while enabling controlled, localized adaptation—achieving a favorable trade-off between flexibility and stability. 

## 8. Details on Human Evaluation 

This section provides details on the user study protocol described in Sec. 4.4. Participants were shown a reference garment and person image, along with five generated try-on results from our model and four from other state-of-the-art baselines. 

Each result was evaluated using three questions: (1) Which image looks the most photo-realistic? (2) Which image best preserves the garment’s design details?—Focus on the fidelity of the color, design details (button, zipper, pocket, logo, etc.), and material (3) Which image best preserves the garment’s structure?—Assess how well the physical shape and form of the garment are maintained on the model. 

We evaluated 50 different garment-person pairs, yielding 150 unique queries. Each query was answered by 30 independent users, resulting in a total of 4,500 responses. To ensure fairness, we provided specific assessment guidelines for every 150 query—for example, whether text or graphics (e.g., “PAKU”, a dog graphic, or “CREAMSODA” logo) appeared clearly and without distortion. 

A screenshot of the evaluation interface is shown in Fig. 14, including instructions and visual examples. 

## 9. Failure Cases 

While our method demonstrates superior and robust performance across various conditions, there are certain failure cases worth noting (Fig. 16). 

One common issue occurs when the input mask provided by the user or generated automatically does not fully cover the original garment. In such cases, the model leverages the exposed regions as strong inpainting priors, leading to undesired continuation of the original garment instead of fully replacing it with the target garment. 

We also observe cases where, even with a correctly defined mask, subtle cues such as shadows or shading patterns from the original clothing bias the generation process, resulting in incomplete or distorted reconstruction of the intended garment. 

**==> picture [237 x 187] intentionally omitted <==**

Figure 16. Failure cases of our method. (Top) When the input mask, either user-provided or automatically generated, does not fully cover the original garment, the model may use the exposed regions as inpainting priors, resulting in undesired continuation of the original clothing. (Bottom) Even with a correct mask, residual cues such as shadows from the original garment can bias the generation process, leading to incomplete or distorted reconstruction of the target garment. 

The samples include diverse garment types applied to people with varying poses, viewpoints, and body shapes. These results highlight the robustness of our unified model in producing perceptually convincing outputs under challenging real-world conditions. 

## 11. More Qualitative Comparisons 

We present additional qualitative comparisons with state-ofthe-art baselines on VITON-HD [13], DressCode [55], and a set of in-the-wild examples gathered from online sources. 

Fig. 18 shows try-on results on VITON-HD, highlighting our model’s ability to maintain structural consistency and garment fidelity. Fig. 19 presents results on DressCode, demonstrating improved alignment and design preservation across various clothing types. To further evaluate generalization, Fig. 20 compares in-the-wild images, where our model consistently generates realistic and artifact-free outputs under diverse poses, lighting, and backgrounds. 

These cases highlight the sensitivity of diffusion-based inpainting to residual visual cues and motivate future work on more robust garment masking and conditioning strategies. 

## 10. Extra Qualitative Results 

We further showcase the versatility of our model in Fig. 17, 22 which contains additional try-on and try-off results across a wide range of garments and human appearances. 

17 

**==> picture [496 x 566] intentionally omitted <==**

Figure 17. Qualitative comparison of try-on results in the wild, showcasing various clothing styles, poses, and viewpoints. 

18 

**==> picture [496 x 576] intentionally omitted <==**

Figure 18. Qualitative comparison of virtual try-on results on the VITON-HD dataset. 

19 

**==> picture [496 x 576] intentionally omitted <==**

Figure 19. Qualitative comparison of virtual try-on results on the DressCode dataset. 

20 

**==> picture [496 x 576] intentionally omitted <==**

Figure 20. Qualitative comparison of virtual try-on results on the in-the-wild images. 

21 

**==> picture [472 x 437] intentionally omitted <==**

Figure 21. Qualitative comparison of try-off results on the VITON-HD and in-the-wild images. For comparison with other methods, the evaluation is conducted using upper clothing only. 

**==> picture [456 x 90] intentionally omitted <==**

Figure 22. Qualitative results of the try-off task on the in-the-wild images, demonstrating performance across various clothing types including lower and full-body clothing. 

22 

