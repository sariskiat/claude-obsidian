---
type: paper-fulltext
slug: chronotailor-harnessing-attention-guidance-for-fine-grained-video-virtual-try-on
arxiv_id: null
source_path: /Users/saris.kia.adm/.paper-scholar/chronotailor-harnessing-attention-guidance-for-fine-grained-video-virtual-try-on/2506.05858.md
paper: "[[chronotailor-harnessing-attention-guidance-for-fine-grained-video-virtual-try-on]]"
---
<!-- extracted by afk_extract from 2506.05858.pdf (21p) -->

# **ChronoTailor: Harnessing Attention Guidance for Fine-Grained Video Virtual Try-On** 

Jinjuan Wang[1] _[,][∗]_ , Wenzhang Sun[2] _[,][∗]_ , Ming Li[1] , Yun Zheng[1] , Fanyao Li[1] , Zhulin Tao[1] _[,][†]_ , Donglin Di[2] , Hao Li[2] , Wei Chen[2] , Xianglin Huang[1] 

1Communication University of China, 2Li Auto 

## **Abstract** 

Video virtual try-on aims to seamlessly replace the clothing of a person in a source video with a target garment. Despite significant progress in this field, existing approaches still struggle to maintain continuity and reproduce garment details. In this paper, we introduce ChronoTailor, a diffusion-based framework that generates temporally consistent videos while preserving fine-grained garment details. By employing a precise spatio-temporal attention mechanism to guide the integration of fine-grained garment features, ChronoTailor achieves robust try-on performance. First, ChronoTailor leverages region-aware spatial guidance to steer the evolution of spatial attention and employs an attention-driven temporal feature fusion mechanism to generate more continuous temporal features. This dual approach not only enables fine-grained local editing but also effectively mitigates artifacts arising from video dynamics. Second, ChronoTailor integrates multi-scale garment features to preserve low-level visual details and incorporates a garmentpose feature alignment to ensure temporal continuity during dynamic motion. Additionally, we collect StyleDress, a new dataset featuring intricate garments, varied environments, and diverse poses, offering advantages over existing public datasets, and will be publicly available for research. Extensive experiments show that ChronoTailor maintains spatio-temporal continuity and preserves garment details during motion, significantly outperforming previous methods. 

## **1 Introduction** 

Driven by e-commerce expansion and demands for immersive digital experiences, video virtual try-on technology [36, 12] enables seamless garment replacement in dynamic videos, allowing users to visualize wearing target clothing without physical trials. By enhancing immersion and interactivity, it bridges the gap between static product images and real-world wearing scenarios. A key challenge is preserving fine details like textures, patterns, and fabric folds while ensuring consistent visuals across frames—especially when the body moves in complex poses. 

Previous methods use flow-driven [41, 10, 46] warping modules and neural generators to align garments via shape deformation. However, their reliance on warping operations causes inherent limitations in temporal coherence. Recent approaches leverage pre-trained diffusion models [53, 4, 35, 15, 28], such as dual-UNet architectures [39] and Diffusion Transformer-based frameworks [25, 48], to eliminate explicit warping and integrate temporal attention layers for modeling motion dynamics. While these methods enhance generative scalability and partially address coherence issues, key challenges persist: (1) Spatial-temporal inconsistency of injected garment features during dynamic motion, leading to misalignment or flickering artifacts; (2) Degradation of fine-grained 

> _∗_ Equal contribution 

> _†_ Corresponding author: `taozhulin@gmail.com` 

Preprint. Under review. 

**==> picture [396 x 151] intentionally omitted <==**

Figure 1: ChronoTailor excels at generating high-quality virtual try-ons for diverse garments, effectively maintaining visual coherence and preserving background details. 

details due to over-smoothing in generative processes and insufficient high-fidelity training data. We claim that virtual video try-on requires accurate guidance to steer the injection of garment information, and such information should stably preserve fine-grained garment details during motion. 

In this paper, we present ChronoTailor, a diffusion-based [21]framework designed to provide stable spatial-temporal attention guidance and pose-aligned garment features with fine-grained details for precise information injection. ChronoTailor uses segmentation results to guide attention to target regions and incorporates an additional attention regularization term to balance spatial consistency between edited and unedited areas. To enhance temporal stability, we deploy asymmetric crossattention to model temporal dependencies in video sequences. Moreover, the framework extracts multilevel garment features to preserve intricate details and aligns these features with pose representations via cross-attention modules with shared weights, ensuring coherent interaction between clothing and human dynamics. 

Specifically, ChronoTailor is composed of two key components: (1) _Spatial-Temporal Attention Guidance_ , which provides a stable spatial-temporal guidance for garment injection. First, we employ segmentation masks [44] to guide the evolution of the spatial attention map. With the aim of attaining an equilibrium of information within and outside the edited region, we further incorporate a regularization term specifically for the attention map. This incorporation empowers the information injection process to place additional emphasis on unedited regions. In addition, we employ an asymmetric cross-space attention mechanism, using original features as queries and features obtained by randomly fusing the current frame with other frames within the video clip as key-values, to effectively integrate temporal information. (2) _Multi-scale Garment-Pose Feature Alignment_ , which enables the acquisition of more precise garment information during human movement. We extract multi-level garment features and fuse them with learnable weights to enhance low-level garment details. Additionally, to align garments with poses, the cross-attention modules in the reference and denoising UNets integrate pose and garment features respectively and align them via weight sharing. Additionally, we collect _StyleDress_ , a new dataset featuring intricate garments, varied environments, and diverse human poses, comprising 14,258 high-quality images and 12,500 videos, offering advantages over existing public datasets. StyleDress will be publicly available for research. In conclusion, the main contributions can be summarized as follows: 

- We introduce ChronoTailor, a diffusion-based framework that leverages spatial-temporal attention guidance and fine-grained pose-aligned garment features to effectively generate temporally coherent videos while preserving intricate garment details, as shown in Figure 1. 

- We introduce Spatial-Temporal Attention Guidance and Multi-scale Garment-Pose Feature Alignment, enabling the model to focus more on edited regions and achieve better preservation of garment textures during motion. 

- We present StyleDress, a comprehensive dataset featuring intricate garments, diverse environments, and varied human poses. Outperforming existing datasets in diversity and quality, it will be publicly released to advance research. 

2 

## **2 Related Work** 

## **2.1 Image Virtual Try-On** 

Traditional virtual try-on methods, predominantly Generative Adversarial Networks [14, 10, 18, 6, 14] (GANs), synthesize images of individuals wearing desired clothing by deforming garments and compositing them onto a person’s image while aiming to preserve identity and consistency. Early works like VITON [18] used coarse-to-fine strategies for garment deformation, while CP-VTON [13] improved alignment via geometric matching. VITON-HD [6] introduced the ALIAS mechanism to mitigate misalignment issues. However, these GAN-based approaches struggle with generalization to complex poses and diverse backgrounds [22]. The emergence of diffusion models has revolutionized the artificial intelligence generated content [40, 49, 33]field and inspired innovative virtual tryon techniques [7, 11, 28, 53, 32, 1]. StableVITON [28] employs zero-cross attention to improve texture fidelity but lacks explicit multi-scale modeling of garment features, resulting in suboptimal adaptive fusion of details across different resolutions. The parallel UNet architecture proposed in TryOnDiffusion [53] captures multi-scale information in an implicit manner via shared parameters. This design lacks the capability to dynamically adjust feature weights in response to the complexity of input garments. ChronoTailor pioneers a learnable cross-scale weighting mechanism to capture both local textures and global semantics simultaneously, enabling adaptive feature fusion and addressing the detail loss inherent in traditional static feature aggregation. 

## **2.2 Video Virtual Try-On** 

Current video virtual try-on methods exhibit relatively primitive applications of attention mechanisms[9, 3, 29, 41, 26, 43, 5, 24]. For example, the visual Transformer in ClothFormer [26] models garment-human relationships within single frames exclusively via self-attention, lacking explicit semantic mask guidance. This leads to attention weights diffusing into non-editing regions, causing boundary artifacts. Although Tunnel TryOn [47] pioneered the use of diffusion models for video try-on, the generated boundaries between garment and non-garment regions remain blurred, with noticeable feature leakage. ChronoTailor mitigates such artifacts by balancing contextual information across regions and introduces an attention-driven temporal feature fusion mechanism to effectively alleviate error accumulation in long videos. 

Additionally, existing methods inadequately modeled the correlation between reference dynamics and human poses[27, 23, 54, 43]. FashionMirror [3] and MV-TON [52] propagated pose-agnostic global features solely through optical flow or memory modules, failing to explicitly disentangle garment deformation from joint movements. This resulted in garment tearing during dynamic actions (e.g., sleeve texture fractures when swinging arms). While ViViD [12] introduced pose information as conditional inputs, it fused them via simplistic fusion mechanisms, unable to establish spatial correspondences between garment features and pose coordinates. ChronoTailor achieves cross-level alignment for the first time by integrating global-local pose information with fine-grained garment features through a dual cross-attention mechanism, guiding accurate deformation of garment features. 

## **3 Method** 

**Overall Architecture.** The architecture, illustrated in Figure 2, consists of two main components: (1) Spatial-Temporal Attention Guidance, which includes Region-Aware Spatial Guidance (RASG) that uses segmentation masks to guide spatial attention map evolution, and Attention-Driven Temporal Feature Fusion (ATFF), which employs an asymmetric cross-space attention mechanism using original features as queries and randomly fused other frame features as key-values to integrate temporal information effectively. (2) Multi-scale Garment-Pose Feature Alignment, which first applies Adaptive Multi-scale Feature Extraction (AMFE) to enhance low-level garment details, and then aligns garment features with pose features using Garment-Pose Feature Alignment (GPFA). More details are available in A.1. 

## **3.1 Spatial-Temporal Attention Guidance** 

In the process of feature integration, the guidance of attention is critical. The goal of spatial-temporal attention guidance is to adjust the regions of attention from both spatial and temporal dimensions, providing more accurate guidance for the feature integration process. 

3 

**==> picture [397 x 207] intentionally omitted <==**

**----- Start of picture text -----**<br>
Multi-scale Garment-Pose Feature Alignment<br>Reference Unet Multi-scale Features Feature Aggregation<br>Q<br>… K<br>V<br>Local Branch<br>Noise<br>Q<br>Global<br>Branch<br>Denosing Unet Spatial-Temporal Attention Guidance zi<br>… …Randomsample zj ( wj zj ) Q<br>Pose Sequence … QK ( V ) zi K<br>Cross- … … V<br>Attention RandomSample zk ( wk zk )<br>Masked Video Masks Attention GuidanceSpatial-Temporal  Attention Map &Masks SequencesLatent Weighted Latents<br>Region-Aware Spatial Guidance Attention-Driven Temporal Feature Fusion<br>Spatial Attention Cross Attention Attention-Driven Temporal Feature Fusion Temporal Attention Region-Aware Spatial Guidance Multi-scale Garment-Pose Feature Alignment<br>VAE Conv SiLU GroupNorm Reference Cross-Attn<br>Clip<br>Multi-scale Garment-Pose  Feature Alignment Pose  Enc Spatial Attn Denosing Cross-Attn<br>Temporal Attention<br>Constrain<br>**----- End of picture text -----**<br>


Figure 2: **Architecture of the ChronoTailor.** Spatial-Temporal Attention Guidance enables the acquisition of stable guidance for garment feature injection. Multi-scale Garment-Pose Feature Alignment, meanwhile, facilitates the capture of more precise garment information during motion. 

## **3.1.1 Region-Aware Spatial Guidance** 

Conventional cross-attention mechanisms typically lack explicit semantic guidance, resulting in diffuse attention across replacement regions. To address these limitations, we introduce Region-Aware Spatial Guidance (RASG), which leverages segmentation masks to explicitly guide attention toward edit regions. Furthermore, we incorporate a regularization term to balance attention distribution, preventing overemphasis on edited regions and ensuring that critical contextual details beyond the replacement area are preserved for spatial consistency. RASG is formulated as: 

**==> picture [324 x 25] intentionally omitted <==**

where _N_ denotes the video sequence length, _M_ represents the segmentation mask, _A_[(] _i[a]_[)] is the attention probability for location _a_ in frame _i_ , and _λN_ balances the contributions of positive and negative attention regularization. The first term enforces alignment between attention maps and the target region, while the second term regularizes attention in non-target regions to maintain global context. The final loss function integrates RASG with the base diffusion model: 

_L_ ChronoTailor = _L_ LDM + _λ_ R _L_ RASG _,_ (2) 

where _L_ LDM is the loss of the latent diffusion model and _λ_ R tunes the weight of RASG. This dualobjective strategy enables the model to focus precisely on target garment regions using segmentation guidance; preserve global coherence by leveraging contextual information from unedited areas. 

## **3.1.2 Attention-Driven Temporal Feature Fusion** 

Attention-Driven Temporal Feature Fusion (ATFF) employs an asymmetric cross-space attention mechanism, using original features as queries and features obtained by randomly fusing the current frame with other frames within the video clip as key-values, to effectively integrate temporal information. Given garment features _{z_ 1 _, z_ 2 _, . . . , zN }_ from a video clip, for each current frame feature _zi_ , we compute attention scores _wj,k_ by randomly selecting frames from the entire clip instead of strictly adjacent frames. Specifically: 

**==> picture [271 x 31] intentionally omitted <==**

where _d_ = _−_ 1 denotes the dimensional index for feature aggregation along the last dimension. These scores weight the randomly selected features _zj_ and _zk_ for fusion with _zi_ , yielding: 

_z_ = _GroupNorm_ ( _wj · zj_ + _zi_ + _wk · zk_ ) _._ (4) 

4 

We employ a cross-attention layer where _z_ is used as keys/values and _zi_ as queries. Through linear projections with shared weights, the corresponding keys _K_ , values _V_ , and queries _Q_ are computed. The final output of ATFF is calculated as: 

**==> picture [256 x 31] intentionally omitted <==**

where _dk_ is the dimensionality of the key vectors. This design thereby enhances temporal continuity. 

## **3.2 Multi-scale Garment-Pose Feature Alignment** 

Precise garment detail preservation and accurate pose alignment are critical challenges in virtual try-on. We propose Adaptive Multi-scale Feature Extraction (AMFE) for optimized garment features and Garment-Pose Feature Alignment (GPFA) for precise alignment, enhancing virtual try-on quality. 

## **3.3 Adaptive Multi-scale Feature Extraction** 

Adaptive Multi-scale Feature Extraction (AMFE) addresses a critical limitation of conventional methods—their exclusive reliance on single-level feature representations, which fails to capture garment details. AMFE extracts hierarchical garment features by processing feature maps _fl_ from the VAE’s last 3 downsampling layers using a structured adaptive fusion. Initially, a standardized transformation _T_ is applied to each feature map _fl_ : 

¯ _fl_ = _T_ ( _fl_ ) = _S ◦N ◦W_ 2 _d ◦P_ ( _fl_ ) _,_ (6) 

where _l ∈{_ 1 _,_ 2 _,_ 3 _}_ indexes the VAE downsampling layers, _P_ aligns resolution via pixel-shuffling, _W_ 2 _d_ mixes channels with 1 _×_ 1 convolution, _N_ performs group normalization, and _S_ is the SiLU activation. This cascade normalizes features for consistent fusion. Subsequently, AMFE employs a dynamic fusion strategy with learnable weights _⃗α_ = [ _α_ 1 _, α_ 2 _, α_ 3]: 

**==> picture [283 x 13] intentionally omitted <==**

where _F_ 1 uses 3 _×_ 3 convolutions to capture local details, _F_ 2 employs 7 _×_ 7 convolutions for global semantics, and _f_[¯] 3 represents the original output of the VAE encoder. The weights _⃗α_ are optimized during training to dynamically balance contributions from different scales, ensuring retention of critical details. 

## **3.4 Garment-Pose Feature Alignment** 

Leveraging the texture representations from AMFE, we introduce the Garment-Pose Feature Alignment (GPFA) module to address the challenge of aligning garments with human poses. GPFA integrates local and global pose information to establish precise correspondences between garment features and body configurations, ensuring semantic consistency across dynamic movements. 

First, we extract dense pose sequences _xp_ using DensePose and derive global pose features _fp_ via a pre-trained CLIP image encoder. For local pose modeling, _xp_ is processed through a four-layer learnable convolutional network Conv _θ_ , then fused with a noise latent feature _z_ 0 to form the input _z_ for the denoising U-Net: 

_z_ = _z_ 0 + Conv _θ_ ( _xp_ ) _,_ (8) 

where Conv _θ_ denotes a convolutional layer with learnable parameters _θ_ . To establish cross-modal dependencies, we design cross-attention modules in both reference and denoising UNet architectures with a weight-sharing mechanism _W_ . The reference cross-attention’s query _QR_ , key _KR_ , and value _VR_ are defined as: 

_QR_ = _zRWQ, KR_ = _fgWK, VR_ = _fgWV ,_ (9) where _WQ, WK, WV_ are shared attention weight matrices; _fg_ is the CLIP-encoded garment feature; and _zR_ is the latent feature from the reference UNet. Similarly, the denoising cross-attention operates as: 

_QD_ = _zDWQ, KD_ = _fpWK, VD_ = _fpWV ,_ (10) with _fp_ denoting the CLIP-encoded pose feature and _zD_ the latent feature from the denoising UNet. By sharing weights, GPFA enhances cross-modal generalization, thus maintaining garment stability during dynamic motion. 

5 

**==> picture [397 x 300] intentionally omitted <==**

**----- Start of picture text -----**<br>
Table 1: Comparison of StyleDress with public<br>datasets.<br>Dataset Pairs Media Gender Body Pose Background<br>VVT 791 V F Avg. Catwalk Indoor<br>ViViD 9700 V F Avg. Catwalk Indoor<br>Figure 3: Sample pairs from the StyleDress. StyleDress 26758 I-V F-M Diverse Diverse Diverse<br>Input<br>Anydoor<br>CatVTON<br> OOTDiffusion<br> ViViD<br>CP-VTON<br> Ours<br> StableVITON<br>**----- End of picture text -----**<br>


Figure 4: Qualitative comparison on the VVT dataset. The dashed box highlights an area with barely perceptible artifacts. Our method demonstrates superior temporal consistency (reduced artifacts) and generates finer garment details. 

## **4 StyleDress** 

Public image/video virtual try-on datasets face challenges: VITON-HD [30] and DressCode [34] offer high-quality visuals but suffer from simple backgrounds and limited poses. VVT [10](videobased) has monotonous actions, uniform backgrounds, and low resolution (256 _×_ 192), restricting real-world use. ViViD [12]provides higher resolution (832 _×_ 624) and more clothing items but exhibits gender imbalance, singular poses, and limited body shape diversity (predominantly slender females). Real-world scenarios demand diverse backgrounds, camera angles, and lighting conditions, necessitating robust datasets. To bridge this gap, we present StyleDress, a high-resolution video dataset for virtual try-on featuring: complex backgrounds, varied body shapes/poses, and balanced gender representation (Figure 3) and Enhanced practical applicability. Table 1 compares dataset diversity, with additional visualizations in Appendix B.2. 

## **5 Experiments** 

**Datasets.** We evaluate our method on the video datasets VVT [10] and ViViD [12]. For a fair comparison with baseline methods, we utilize models trained at a resolution of 512×384 pixels. To evaluate the virtual video try-on performance of our approach, we evaluate it in the VVT data set with a consistent resolution of 512×384 pixels. 

**Metrics and Baselines.** We adhere to the VTON video generation evaluation paradigm employing SSIM [42], LPIPS [50], _V FIDI_ 3 _D_ [2] and _V FIDRes_ [20] scores. Given the limited public availability of most comparative methods, our evaluation relies on reported results and accessible generated videos. We start our comparison with CP-VTON[41], a GAN-based method, followed by comparisons with state-of-the-art diffusion-based methods: OOTDiffusion [45], Anydoor [4], StableVITON[28] , CatVTON[8], and ViViD[12]. 

6 

**==> picture [396 x 302] intentionally omitted <==**

**----- Start of picture text -----**<br>
Input CP-VTON OOTDiffusion Anydoor StableVITON CatVTON ViViD Ours<br>Image-Based Video-Based<br>**----- End of picture text -----**<br>


Figure 5: Qualitative comparison on the ViViD dataset. We demonstrate the results for lower garments, where ChronoTailor better preserves the patterns and shapes of trousers. 

||||||Input|ViViD|Ours||
|---|---|---|---|---|---|---|---|---|
|Table 2: Comparison on|||the VVT|dataset: _↑_|||||
|denotes higher|is better, while_↓_indicates lower||||||||
|is better.|||||||||
|Method|SSIM_↑_|LPIPS_↓_|_V FIDI_3_D↓_|_V FIDResNeXt↓_|||||
|CP-VTON [41]|0.459|0.535|6.361|12.100|||||
|FW-GAN [10]|0.675|0.283|8.019|12.150|||||
|PBAFN [13]|0.870|0.157|4.516|8.690|||||
|MVTON [52]|0.853|0.068|8.367|9.702|||||
|Tunnel Try-on [46]|0.913|0.054|**3.345**|4.614|||||
|ClothFormer [26]|0.921|0.081|3.967|5.048|||||
|AnyDoor [4]|0.800|0.127|4.535|5.990|||||
|StableVITON [28]|0.760|0.184|17.068|11.254|||||
|CatVTON [8]|0.887|0.099|8.311|6.013|||||
|ViViD [12]|0.949|0.068|3.405|5.074|||||
|**Ours**|**0.978**|**0.049**|3.721|**4.065**|||||



Table 2: Comparison on the VVT dataset: _↑_ denotes higher is better, while _↓_ indicates lower is better. 

Figure 6: ChronoTailor effectively preserves information in non - edited regions. 

**Implementation Details.** The experiments are conducted on two NVIDIA A800 GPUs. During the training process, all data is adjusted to a uniform resolution of 512×384. We set the batch size to 8 and the learning rate to 1 _e[−]_[5] . More details are provided in Appendix A.2. 

## **5.1 Qualitative Analysis** 

Figure 4 presents a visual comparison between our proposed method and baseline approaches using the VVT dataset. Further evidence of ChronoTailor’s consistently superior quality and robustness in virtual try-on results is provided by the visualization results on the ViViD dataset, as shown in Figure 5. Specifically, our method demonstrates enhanced preservation of garment texture details while maintaining inter-frame consistency, a significant challenge frequently encountered by other approaches that often struggle with the fidelity of clothing try-ons. In contrast, our method exhibits 

7 

Table 3: User study for the preference rate on the ViViD test dataset. 

|Method|Fidelity|Background|Consistency|Overall|
|---|---|---|---|---|
|OOTDiffusion [45]|15.36%|10.34%|0.00%|15.52%|
|StableVITON[28]|2.51%|6.58%|0.00%|2.82%|
|CatVTON[8]|6.90%|9.09%|0.00%|6.58%|
|ViViD[12]|19.91%|21.79%|39.00%|20.22%|
|**ChronoTailor**|**55.33**%|**52.19**%|**61.00**%|**54.86**%|



Table 4: Ablation study of Spatial-Temporal Attention Guidance 

|Method|PSNR_↑_|SSIM_↑_|LPIPS_↓_|_V FIDI_3_D↓_|_V FIDRES↓_|
|---|---|---|---|---|---|
|Baseline|24.62|0.850|0.086|10.000|0.630|
|w/ RASG|25.86|0.894|0.087|9.478|0.535|
|w/ ATFF|26.52|0.899|0.084|9.834|0.612|
|w/ RASG + ATFF|**26.64**|**0.901**|**0.083**|**9.630**|**0.560**|



**==> picture [191 x 129] intentionally omitted <==**

**----- Start of picture text -----**<br>
Image Try-on Video Try-on<br>StyleDress<br>Wild Data<br>**----- End of picture text -----**<br>


Figure 7: Qualitative results of our method on the StyleDress dataset. 

remarkable robustness, which can be attributed to our novel Spatial-Temporal Attention Guidance and Multi-scale Garment-Pose Feature Alignment mechanisms. In contrast to the ViViD method, our approach exhibits enhanced preservation of information beyond the edited regions, as depicted in Figure 6. Furthermore, Figure 7 showcases the effectiveness of ChronoTailor when evaluated on our collected StyleDress dataset and unseen in-the-wild data, thereby further validating its generalization. Appendix B.1 presents additional visual comparison results between our method and other baselines. 

## **5.2 Quantitative Comparisons** 

**Metric Evaluation.** Table 2 presents a comparative analysis of our proposed method against baseline approaches on the VVT dataset. Our method demonstrates superior performance in terms of the SSIM and LPIPS scores, indicating improved visual fidelity. Furthermore, we achieve strong results in the _V FIDRes_ metric and exhibit competitive performance in _V FIDI_ 3 _D_ , collectively suggesting improved visual quality and temporal consistency. This robust performance effectively showcases the reliability of ChronoTailor for the task of virtual garment try-on in videos. Notably, while image-based methods can generate accurate single-frame results, they often suffer from flickering and inconsistencies due to limitations in temporal coherence and inter-frame motion handling. 

**User Study.** We conduct a user study to evaluate the generation quality of our model. We utilize all test garments from the ViViD dataset and randomly present participants with 13 video virtual try-on results generated by both baseline methods and our approach. Each participant is asked to select their most preferred result based on four criteria: garment fidelity, background preservation, temporal consistency, and overall evaluation. We receive valid responses from 40 users, and the collected preferences are reported in Table 3. Across the four evaluation criteria, our proposed method is preferred by the majority of participants, achieving preference rates of 55 _._ 33%, 52 _._ 19%, 61 _._ 00%, and 54 _._ 86%, respectively. Furthermore, Appendix B.3 presents further visual comparisons of the user study. 

## **5.3 Ablation Study** 

**Effectiveness of RASG&ATFF.** To validate the efficacy of our spatial-temporal attention guidance, we first conduct an ablation study excluding the region-aware spatial guidance (RASG). As depicted in Figure 8, the integration of RASG demonstrably reduces the leakage of garment semantic information compared to the absence of RASG, indicating its effectiveness in constraining garment features to the appropriate regions. In Figure 9, "w/o ATFF" indicates the absence of adaptive attention-driven temporal feature fusion (ATFF), leading to blurriness. Conversely, our ATFF recovers sharp textures by leveraging weighted temporal features to enhance the current frame’s reconstruction, thereby improving temporal consistency and reducing video blur. The comparative results presented in Table 4 further substantiate the effectiveness of each proposed module. Appendix B.4.1 further presents a comprehensive validation of the overall effectiveness of the spatial-temporal attention guidance. 

8 

**==> picture [191 x 106] intentionally omitted <==**

**----- Start of picture text -----**<br>
Input w/o RASG w/ RASG<br>**----- End of picture text -----**<br>


Figure 8: Ablation study of Region-Aware Spatial Guidance. Our method balances attention between editing and non - editing regions. 

**==> picture [195 x 115] intentionally omitted <==**

**----- Start of picture text -----**<br>
Input w/o AMFE w/ AMFE<br>**----- End of picture text -----**<br>


Figure 10: Ablation study of Adaptive Multi-scale Feature Extraction. The upper garment try-on fully recovers two floral patterns, demonstrating ChronoTailor’s capability to preserve intricate textures and semantic details. 

**==> picture [191 x 104] intentionally omitted <==**

**----- Start of picture text -----**<br>
w/o ATFF<br>Input<br>w/ ATFF<br>**----- End of picture text -----**<br>


Figure 9: Ablation study of Attention-Driven Temporal Feature Fusion. 

Table 5: Ablation study of Multi-scale GarmentPose Feature Alignment. 

**==> picture [195 x 107] intentionally omitted <==**

**----- Start of picture text -----**<br>
Method PSNR ↑ SSIM ↑ LPIPS ↓ V FIDI 3 D↓ V FIDRES↓<br>Baseline 24.62 0.850 0.086 10.000 0.630<br>w/ AMFE 25.61 0.860 0.085 9.830 0.562<br>w/ GPFA 26.23 0.904 0.085 9.855 0.574<br>w/ AMFE+GPFA 26.26 0.904 0.084 9.739 0.553<br>Input w/o Pose w/ Pose w/ Pose+AMFE<br>**----- End of picture text -----**<br>


Figure 11: Ablation study of Garment-Pose Feature Alignment. GPFA aligns garment features to poses, restoring occluded neck regions. 

**Effectiveness of AMFE&GPFA.** To validate our multi-scale garment-pose feature alignment, we performed ablation experiments using ReferenceNet-encoded garment features as the baseline. Figure 10 shows that integrating the adaptive multi-scale feature extraction (AMFE) module significantly improves garment detail preservation and semantic reconstruction. Figure 11 demonstrates the effectiveness of our garment-pose feature alignment. Incorporating Global-Local Pose Constraints (w/ Pose) enhances virtual try-on fidelity. Aligning AMFE-extracted multi-scale details with pose information (w/ Pose+AMFE) enables more precise pose-garment alignment, increasing try-on accuracy(see Appendix B.4.2). Quantitative results in Table 5 further confirm the substantial visual quality improvement of our method. 

## **6 Conclusion and Limitations** 

This paper presents ChronoTailor, a diffusion-based video virtual try-on framework enabling precise garment manipulation and fine-grained detail preservation via Spatial-Temporal Attention Guidance (segmentation masks, regularization, and cross-space attention for temporal coherence) and Multi-scale Garment-Pose Feature Alignment (hierarchical feature extraction and UNet-based pose alignment). We also introduce StyleDress, a high-quality dataset with diverse garments, environments, and poses, to be open-sourced post-publication. Extensive experiments show that ChronoTailor outperforms previous methods, generating temporally continuous videos while preserving garment details during motion. 

**Limitations.** Existing generative video virtual try-on methods heavily depend on high-fidelity segmentation and pose estimation, which degrade significantly under occlusion, low resolution, or high-dynamic motion. This often leads to attentional misalignment or spatial misregistration. While our method achieves precise garment alignment via spatial-temporal attention guidance, enhancing its generalization robustness in these complex scenarios remains a critical open challenge. 

9 

## **References** 

- [1] Alberto Baldrati, Davide Morelli, Giuseppe Cartella, Marcella Cornia, Marco Bertini, and Rita Cucchiara. Multimodal garment designer: Human-centric latent diffusion models for fashion image editing. In _Proceedings of the IEEE/CVF international conference on computer vision_ , pages 23393–23402, 2023. 

- [2] Joao Carreira and Andrew Zisserman. Quo vadis, action recognition? a new model and the kinetics dataset. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 6299–6308, 2017. 

- [3] Chieh-Yun Chen, Ling Lo, Pin-Jui Huang, Hong-Han Shuai, and Wen-Huang Cheng. Fashionmirror: Co-attention feature-remapping virtual try-on with sequential template poses. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 13809–13818, 2021. 

- [4] Xi Chen, Lianghua Huang, Yu Liu, Yujun Shen, Deli Zhao, and Hengshuang Zhao. Anydoor: Zero-shot object-level image customization. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 6593–6602, 2024. 

- [5] Xi Chen, Zhiheng Liu, Mengting Chen, Yutong Feng, Yu Liu, Yujun Shen, and Hengshuang Zhao. Livephoto: Real image animation with text-guided motion control. In _European Conference on Computer Vision_ , pages 475–491. Springer, 2024. 

- [6] Seunghwan Choi, Sunghyun Park, Minsoo Lee, and Jaegul Choo. Viton-hd: High-resolution virtual try-on via misalignment-aware normalization. In _Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , Jun 2021. 

- [7] Yisol Choi, Sangkyung Kwak, Kyungmin Lee, Hyungwon Choi, and Jinwoo Shin. Improving diffusion models for virtual try-on. Mar 2024. 

- [8] Zheng Chong, Xiao Dong, Haoxiang Li, Shiyue Zhang, Wenqing Zhang, Xujie Zhang, Hanqing Zhao, Dongmei Jiang, and Xiaodan Liang. Catvton: Concatenation is all you need for virtual try-on with diffusion models. _arXiv preprint arXiv:2407.15886_ , 2024. 

- [9] Aiyu Cui, Daniel McKee, and Svetlana Lazebnik. Dressing in order: Recurrent person image generation for pose transfer, virtual try-on and outfit editing. In _Proceedings of the IEEE/CVF international conference on computer vision_ , pages 14638–14647, 2021. 

- [10] Haoye Dong, Xiaodan Liang, Xiaohui Shen, Bowen Wu, Bing-Cheng Chen, and Jian Yin. Fw-gan: Flow-navigated warping gan for video virtual try-on. In _2019 IEEE/CVF International Conference on Computer Vision (ICCV)_ , Oct 2019. 

- [11] Patrick Esser, Johnathan Chiu, Parmida Atighehchian, Jonathan Granskog, and Anastasis Germanidis. Structure and content-guided video synthesis with diffusion models. In _Proceedings of the IEEE/CVF international conference on computer vision_ , pages 7346–7356, 2023. 

- [12] Zixun Fang, Wei Zhai, Aimin Su, Hongliang Song, Kai Zhu, Mao Wang, Yu Chen, Zhiheng Liu, Yang Cao, and Zheng-Jun Zha. Vivid: Video virtual try-on using diffusion models. _arXiv preprint arXiv:2405.11794_ , 2024. 

- [13] Yuying Ge, Yibing Song, Ruimao Zhang, Chongjian Ge, Wei Liu, and Ping Luo. Parser-free virtual try-on via distilling appearance flows. In _Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , Jun 2021. 

- [14] Ian J Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. _Advances in neural information processing systems_ , 27, 2014. 

- [15] Junhong Gou, Siyu Sun, Jianfu Zhang, Jianlou Si, Chen Qian, and Liqing Zhang. Taming the power of diffusion models for high-quality virtual try-on with appearance flow. In _Proceedings of the 31st ACM International Conference on Multimedia_ , pages 7599–7607, 2023. 

- [16] Rıza Alp Güler, Natalia Neverova, and Iasonas Kokkinos. Densepose: Dense human pose estimation in the wild. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 7297–7306, 2018. 

- [17] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. _arXiv preprint arXiv:2307.04725_ , 2023. 

10 

- [18] Xintong Han, Zuxuan Wu, Zhe Wu, Ruichi Yu, and Larry S. Davis. Viton: An image-based virtual try-on network. In _Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , Jun 2018. 

- [19] Tiankai Hang, Shuyang Gu, Chen Li, Jianmin Bao, Dong Chen, Han Hu, Xin Geng, and Baining Guo. Efficient diffusion training via min-snr weighting strategy. In _Proceedings of the IEEE/CVF international conference on computer vision_ , pages 7441–7451, 2023. 

- [20] Kensho Hara, Hirokatsu Kataoka, and Yutaka Satoh. Can spatiotemporal 3d cnns retrace the history of 2d cnns and imagenet? In _Proceedings of the IEEE/CVF conference on Computer Vision and Pattern Recognition_ , pages 6546–6555, 2018. 

- [21] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. _Advances in neural information processing systems_ , 33:6840–6851, 2020. 

- [22] Shion Honda. Viton-gan: Virtual try-on image generator trained with adversarial loss. _arXiv preprint arXiv:1911.07926_ , 2019. 

- [23] Li Hu. Animate anyone: Consistent and controllable image-to-video synthesis for character animation. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 8153–8163, 2024. 

- [24] Yaosi Hu, Chong Luo, and Zhenzhong Chen. Make it move: controllable image-to-video generation with text descriptions. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 18219–18228, 2022. 

- [25] Boyuan Jiang, Xiaobin Hu, Donghao Luo, Qingdong He, Chengming Xu, Jinlong Peng, Jiangning Zhang, Chengjie Wang, Yunsheng Wu, and Yanwei Fu. Fitdit: Advancing the authentic garment details for high-fidelity virtual try-on. _arXiv preprint arXiv:2411.10499_ , 2024. 

- [26] Jianbin Jiang, Tan Wang, He Yan, Junhui Liu, and Bigo Bigo. Clothformer: Taming video virtual try-on in all module. 

- [27] Johanna Karras, Aleksander Holynski, Ting-Chun Wang, and Ira Kemelmacher-Shlizerman. Dreampose: Fashion image-to-video synthesis via stable diffusion. In _2023 IEEE/CVF International Conference on Computer Vision (ICCV)_ , pages 22623–22633. IEEE, 2023. 

- [28] Jeongho Kim, Gyojung Gu, Minho Park, Sunghyun Park, and Jaegul Choo. Stableviton: Learning semantic correspondence with latent diffusion model for virtual try-on. Dec 2023. 

- [29] Gaurav Kuppa, Andrew Jong, Xin Liu, Ziwei Liu, and Teng-Sheng Moh. Shineon: Illuminating design choices for practical video-based virtual clothing try-on. In _Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision_ , pages 191–200, 2021. 

- [30] Sangyun Lee, Gyojung Gu, Sunghyun Park, Seunghwan Choi, and Jaegul Choo. High-resolution virtual try-on with misalignment and occlusion-handled conditions. In _European Conference on Computer Vision_ , pages 204–219. Springer, 2022. 

- [31] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In _International conference on machine learning_ , pages 19730–19742. PMLR, 2023. 

- [32] Yifei Li, Tao Du, Kui Wu, Jie Xu, and Wojciech Matusik. Diffcloth: Differentiable cloth simulation with dry frictional contact. _ACM Transactions on Graphics (TOG)_ , 42(1):1–20, 2022. 

- [33] Huaize Liu, Wenzhang Sun, Donglin Di, Shibo Sun, Jiahui Yang, Changqing Zou, and Hujun Bao. Moee: Mixture of emotion experts for audio-driven portrait animation. _arXiv preprint arXiv:2501.01808_ , 2025. 

- [34] Davide Morelli, Matteo Fincato, Marcella Cornia, Federico Landi, Fabio Cesari, and Rita Cucchiara. Dress code: High-resolution multi-category virtual try-on. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 2231–2235, 2022. 

- [35] Davide Morelli, Alberto Baldrati, Giuseppe Cartella, Marcella Cornia, Marco Bertini, and Rita Cucchiara. Ladi-vton: Latent diffusion textual-inversion enhanced virtual try-on. In _Proceedings of the 31st ACM international conference on multimedia_ , pages 8580–8589, 2023. 

- [36] Hung Nguyen, Quang Qui-Vinh Nguyen, Khoi Nguyen, and Rang Nguyen. Swifttry: Fast and consistent video virtual try-on with diffusion models. _arXiv preprint arXiv:2412.10178_ , 2024. 

11 

- [37] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, et al. Sam 2: Segment anything in images and videos. _arXiv preprint arXiv:2408.00714_ , 2024. 

- [38] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 10684–10695, 2022. 

- [39] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In _Medical image computing and computer-assisted intervention–MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18_ , pages 234–241. Springer, 2015. 

- [40] Wenzhang Sun, Xiang Li, Donglin Di, Zhuding Liang, Qiyuan Zhang, Hao Li, Wei Chen, and Jianxun Cui. Uniavatar: Taming lifelike audio-driven talking head generation with comprehensive motion and lighting control. _arXiv preprint arXiv:2412.19860_ , 2024. 

- [41] Bochao Wang, Huabin Zheng, Xiaodan Liang, Yimin Chen, Liang Lin, and Meng Yang. Toward characteristic-preserving image-based virtual try-on network. In _Proceedings of the European conference on computer vision (ECCV)_ , pages 589–604, 2018. 

- [42] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. _IEEE transactions on image processing_ , 13(4):600–612, 2004. 

- [43] Min Wei, Chaohui Yu, Jingkai Zhou, and Fan Wang. 3dv-ton: Textured 3d-guided consistent video try-on via diffusion models. _arXiv preprint arXiv:2504.17414_ , 2025. 

- [44] Xinyu Xiong, Zihuang Wu, Shuangyi Tan, Wenxue Li, Feilong Tang, Ying Chen, Siying Li, Jie Ma, and Guanbin Li. Sam2-unet: Segment anything 2 makes strong encoder for natural and medical image segmentation. _arXiv preprint arXiv:2408.08870_ , 2024. 

- [45] Yuhao Xu, Tao Gu, Weifeng Chen, and Arlene Chen. Ootdiffusion: Outfitting fusion based latent diffusion for controllable virtual try-on. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 39, pages 8996–9004, 2025. 

- [46] Zhengze Xu, Mengting Chen, Zhao Wang, Linyu Xing, Zhonghua Zhai, Nong Sang, Jinsong Lan, Shuai Xiao, and Changxin Gao. Tunnel try-on: Excavating spatial-temporal tunnels for high-quality virtual try-on in videos. In _Proceedings of the 32nd ACM International Conference on Multimedia_ , pages 3199–3208, 2024. 

- [47] Zhongcong Xu, Jianfeng Zhang, JunHao Liew, Hanshu Yan, Jia-Wei Liu, Chenxu Zhang, Jiashi Feng, and MikeZheng Shou. Magicanimate: Temporally consistent human image animation using diffusion model. 

- [48] Jianhao Zeng, Dan Song, Weizhi Nie, Hongshuo Tian, Tongtong Wang, and An-An Liu. Cat-dm: Controllable accelerated virtual try-on with diffusion model. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 8372–8382, 2024. 

- [49] Qiyuan Zhang, Chenyu Wu, Wenzhang Sun, Huaize Liu, Donglin Di, Wei Chen, and Changqing Zou. Semantic latent motion for portrait video generation, 2025. 

- [50] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 586–595, 2018. 

- [51] Zijun Zhang. Improved adam optimizer for deep neural networks. In _2018 IEEE/ACM 26th international symposium on quality of service (IWQoS)_ , pages 1–2. Ieee, 2018. 

- [52] Xiaojing Zhong, Zhonghua Wu, Taizhe Tan, Guosheng Lin, and Qingyao Wu. Mv-ton: Memory-based video virtual try-on network. In _Proceedings of the 29th ACM International Conference on Multimedia_ , Oct 2021. 

- [53] Luyang Zhu, Dawei Yang, Tyler Zhu, Fitsum Reda, William Chan, Chitwan Saharia, Mohammad Norouzi, and Ira Kemelmacher-Shlizerman. Tryondiffusion: A tale of two unets. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 4606–4615, 2023. 

- [54] Shenhao Zhu, Junming Leo Chen, Zuozhuo Dai, Zilong Dong, Yinghui Xu, Xun Cao, Yao Yao, Hao Zhu, and Siyu Zhu. Champ: Controllable and consistent human image animation with 3d parametric guidance. In _European Conference on Computer Vision_ , pages 145–162. Springer, 2024. 

12 

## **A APPENDIX** 

## **A.1 Preliminary** 

The Diffusion Model represents a category of generative models that synthesize data via an iterative denoising process from a foundation of random noise. In order to enhance the proficiency of training and inference, Stable Diffusion (SD) [38] functions within the compact latent space of a pre-trained autoencoder, denoted as _D_[¯] ( _E_[¯] ( _·_ )), where _E_[¯] and _D_[¯] symbolize the encoder and decoder, respectively. Considering an input image _I_ 0 along with its corresponding latent feature _x_ 0 = _E_[¯] ( _I_ 0), the diffusion forward process is delineated as follows: 

**==> picture [284 x 12] intentionally omitted <==**

here, _α_ ¯ _t_ =[�] _[t] t_ 0=1[(1] _[ −][β]_[¯] _[t]_[0][)][ and] _[β]_[¯] _[t]_[0][represents the pre-specified variance schedule at timestep] _[ t]_[0][.] 

During the diffusion backward process, referred to as _LLDM_ and conditioned on the textual context _c_ ¯, the model is engineered to anticipate the noise _ϵθ_ ( **x** _t,_ ¯ _c, t_ ), which can be summarized as: 

**==> picture [272 x 12] intentionally omitted <==**

Where _ϵθ_ stands for a U-Net model that is trained to anticipate the noise within the diffusion process. 

## **A.2 Training and Inference Details** 

The backbone network initializes the UNet and clothing encoder with Stable Diffusion-1.5 weights, while keeping the VAE encoder and CLIP image encoder weights unchanged. The temporal module uses weights from the AnimeDiff [17] motion module for initialization. Our model is optimized using the Adam optimizer [51] with _β_ 1 = 0 _._ 9, _β_ 2 = 0 _._ 999, and a fixed learning rate applied consistently across image pretraining and video fine-tuning stages. We integrate the Min-SNR Weighting Strategy [19] with _γ_ = 5 _._ 0 to enhance training stability, and employ mixed-precision FP16 throughout training to optimize computational efficiency. 

The training pipeline consists of two phases: an initial image pretraining of 60,000 steps on image datasets (240 GPU hours) focused on spatial feature learning, followed by video fine-tuning for an additional 30,000 steps on video data (120 GPU hours), where only the temporal module is trained to optimize temporal coherence while preserving spatial accuracy, utilizing a continuous 24-frame sequence sampled for video training. 

For inference, tests are conducted on an NVIDIA GeForce RTX 4090. The model processes 48 frames per video at 512 _×_ 384 resolution to balance efficiency and fidelity. 

## **A.3 Video Generation Model Details** 

## **A.3.1 Region-Aware Spatial Guidance** 

Notably, RASG is exclusively applied during the training phase and introduces no additional computational overhead at inference. This design ensures that our approach maintains high efficiency for real-time applications while significantly enhancing the visual realism and semantic consistency of virtual try-on results during training. By decoupling guidance mechanism usage from inference, RASG achieves a practical balance between training-time optimization and deployment efficiency. 

## **B Additional Analysis for ChronoTailor** 

## **B.1 Additional Qualitative Results** 

## **B.1.1 Qualitative results on ViViD and VVT Datasets** 

**Qualitative results on the ViViD dataset.** In Figure 12, we present the video virtual try-on results of models on the VIVID dataset [12]. The GAN-based model [41]exhibits color discrepancies in skin tones and clothing textures. OOTDiffusion [45] incorrectly generates ultra-shorts into mid-length shorts, while AnyDoor [4] struggles to achieve precise garment replacement. Both StableVITON [28] and CatVTON [8] produce clothing category errors—for example, transforming shorts into long 

13 

pants or skirts. ViViD not only shows color deviations but also introduces spurious patterns absent from the input. 

**Qualitative results on the VVT dataset.** The experimental results on the VVT dataset[10] demonstrate our model’s capability to restore texture details while maintaining excellent pose alignment, as shown in Figure 13. 

## **B.1.2 Qualitative results on the StyleDress dataset** 

Figure 14 showcases additional results of our video virtual try-on method on the StyleDress dataset. By leveraging the dataset’s multimodal data (images and videos), our approach demonstrates superior performance compared to methods trained on the VVT dataset. Specifically, our method ensures consistent garment texture quality across frames, enabled by StyleDress’ rich information on clothing types, backgrounds, and poses, highlighting the dataset’s unique value for robust, high-fidelity virtual try-on systems. 

## **B.1.3 Qualitative results on the VITON-HD Dataset** 

Our approach is also effective for image virtual try-on tasks. As shown in Figure 15, the results on the VITON-HD [30] dataset demonstrate that our method achieves competitive performance in image-based garment transfer, balancing visual realism and clothing detail preservation. 

## **B.2 StyleDress** 

Our **StyleDress multimodal dataset** addresses key limitations in existing virtual try-on datasets, particularly concerning gender bias, limited pose variations, and restricted body shape representation. We developed this high-resolution, video-centric dataset, augmented with an image dataset, to provide comprehensive training materials for realistic virtual try-on in diverse, real-world scenarios. 

The video dataset features multi-action indoor scenes (frame rate _≥_ 24fps) capturing complex dynamics like walking and dancing, alongside a wide array of clothing categories. This design specifically enhances the model’s ability to learn spatio-temporal consistency. Complementing this, the image dataset, compiled from e-commerce platforms and fashion social networks, boasts diverser material categories and design elements, emphasizing high-resolution texture details for increased data diversity. StyleDress significantly expands diversity by including a full gender spectrum, varied body shapes (BMI 16-28), a wide range of human postures, and complex indoor and outdoor backgrounds. This broad coverage of real-world complexities, such as dynamic lighting, multi-angle camera views, and articulated human motions, highlights the dataset’s robust advantages over traditional collections. 

The dataset employs a five-tuple annotation system and a rigorous processing pipeline. Videos are preprocessed and cropped to 832 _×_ 624 pixels, utilizing DensePose [16] for 3D pose and surface semantics and OOTDiffusion for agnostic masks. Clothing images are precisely segmented using SAM2 [37], followed by manual refinement after BLIP-2 [31] classification, which also added 7 new subcategories like jumpsuits. To further enhance semantic retention in non-edited areas, over 5000 accessory samples from Net-A-Porter are included with annotated retention masks in occluded regions. A hierarchical clothing attribute label system, constructed through BLIP-2 and manual annotation, further enriches the dataset. Figure 16 showcases typical samples from this dataset, which will be open-sourced for public research. 

## **B.3 User Study Details** 

In this section, we conduct a comprehensive user study through qualitative evaluation to compare ChronoTailor with existing methods (OOTDiffusion [45], Anydoor [4], StableVITON [28], CatVTON [8], and ViViD [12]), as illustrated in Figure 17. Our user study utilizes 13 reference garment images from the ViViD test set and their corresponding virtual try-on video results, with exemplars visualized in Figure 10. A total of 50 evaluation samples are collected. All methods are anonymized as Method _⃝_ 1 , _⃝_ 2 , _⃝_ 3 , _⃝_ 4 and _⃝_ 5 , and the presentation order of try-on video results is randomized in each comparative group to eliminate order bias. The evaluation dimensions and their operational definitions are as follows: 

- **Fidelity** : The faithfulness of garment appearance reproduction in try-on results. 

14 

- **Background** : The retention of environmental details, accessories, and unedited regions (e.g., hands). 

- **Consistency** : The smoothness and coherence of video sequences across frames. 

- **Overall** : A comprehensive assessment of videos’ holistic generation quality, integrating visual fidelity, temporal coherence, and background preservation to evaluate realism and technical integrity. 

This randomized and anonymized evaluation framework ensures objective and unbiased qualitative comparisons, enabling rigorous assessment of ChronoTailor’s performance against state-of-the-art alternatives in terms of visual fidelity, background preservation (of unedited regions), temporal consistency, and overall visual quality. 

## **B.4 More Ablation Study** 

## **B.4.1 Effectiveness of Spatial-Temporal Attention Guidance** 

In Figure 18, we present more visual results to validate the effectiveness of Spatial-Temporal Attention Guidance (STAG). "w/o STAG" denotes the case without adopting Spatial-Temporal Attention Guidance, while "w/ STAG" denotes the case with adopting it. "w/o STAG" encounters severe color difference issues, resulting in significant loss of semantic information in clothing colors. In contrast, "w/ STAG" remarkably reduces color differences and improves clothing consistency. This is attributed to the fact that our designed Spatial-Temporal Attention Guidance adjusts the attention regions from both spatial and temporal dimensions, thereby providing more precise guidance for the feature integration process. 

## **B.4.2 Effectiveness of Multi-scale Garment-Pose Feature Alignment** 

Figure 19 shows additional visual results validate the effectiveness of the Multi-Scale Garment Position Alignment (MG-PFA). The baseline model without MG-PFA (w/o MG-PFA, Base model) exhibits significant geometric distortion and semantic information loss in garments, manifested as contour warping and texture fragmentation. In contrast, the model with MG-PFA (w/ MGPFA) achieves precise alignment between garment semantics and human poses through cross-layer interaction of multi-scale features, significantly enhancing geometric consistency and detail fidelity. Experimental results demonstrate that MG-PFA mitigates deformation defects caused by traditional single-scale alignment via multi-resolution hierarchical alignment, effectively suppressing garment distortion under complex poses. 

15 

**==> picture [396 x 558] intentionally omitted <==**

**----- Start of picture text -----**<br>
Input CP-VTON  OOTDiffusion  Anydoor StableVITONN CatVTON ViViD Ours<br>Image-Based Video-Based<br>**----- End of picture text -----**<br>


Figure 12: Qualitative comparison on the ViViD dataset. Our method can generate more coloraccurate fitting results. 

16 

**==> picture [397 x 212] intentionally omitted <==**

**----- Start of picture text -----**<br>
 Input<br> Anydoor<br> CatVTON<br>OOTDiffusion<br> ViViD<br>CP-VTON<br>Ours<br> StableVITON<br>**----- End of picture text -----**<br>


Figure 13: Qualitative comparison on the VVT dataset. Our method generates clearer texture details in the editing region. 

17 

**==> picture [396 x 360] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a) Image  Try - on (b) Video  Try - on<br>**----- End of picture text -----**<br>


Figure 14: Additional video try-on results on the StyleDress dataset are shown. Left column image try-ons for complex textures demonstrate precise pattern mapping via dataset-enhanced textures. Right column video try-ons show: 1) semantic alignment for small body types (first two rows); 2) spatiotemporal texture consistency during walking/turning motions (third row); 3) background integrity preservation during clothing replacement (fourth row). 

18 

**==> picture [397 x 570] intentionally omitted <==**

**----- Start of picture text -----**<br>
Input CP-VTON OOTDiffusion Anydoor StableVITON CatVTON Ours<br>Figure 15: Qualitative comparison on the VITON-HD dataset.<br>Image Video<br>Hats Scarves<br>Bags 375 249<br>2443 Upper<br>5500<br>Shoes<br>4691<br>Business Low<br>1000 4500<br>Suit<br>1000 Onesie Dress<br>1000 5000<br>**----- End of picture text -----**<br>


Figure 16: Our StyleDress dataset offers additional examples with diverse clothing, challenging poses and environments, and balanced representation of body shapes, genders, and ages. 

19 

**==> picture [396 x 261] intentionally omitted <==**

**----- Start of picture text -----**<br>
Input<br> OOTDiffusion<br> StableVITON<br> Catvton<br>ViViD<br>Ours<br>**----- End of picture text -----**<br>


Figure 17: ChronoTailor best preserves the texture details and appearance information of garments across different poses. 

**==> picture [199 x 255] intentionally omitted <==**

**----- Start of picture text -----**<br>
Input<br>w/o STAG<br>w/ STAG<br>**----- End of picture text -----**<br>


Figure 18: Ablation study of Spatial-Temporal Attention Guidance. Our method not only achieves superior semantic information preservation but also further reduces color discrepancy. 

20 

**==> picture [198 x 103] intentionally omitted <==**

**----- Start of picture text -----**<br>
Input w/o MG-PFA w/ MG-PFA<br>**----- End of picture text -----**<br>


Figure 19: Ablation study of Multi-scale Garment-Pose Feature Alignment. ChronoTailor achieves precise retention of garment details through deep semantic modeling, significantly maintaining the hierarchical texture of upper garment lace and the visual integrity of text. 

21 

