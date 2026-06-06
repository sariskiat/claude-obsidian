---
type: paper-fulltext
slug: catv2ton-taming-diffusion-transformers-for-vision-based-virtual-try-on-with-temporal-concatenation
arxiv_id: null
source_path: /Users/saris.kia.adm/.paper-scholar/catv2ton-taming-diffusion-transformers-for-vision-based-virtual-try-on-with-temporal-concatenation/2501.11325.md
paper: "[[catv2ton-taming-diffusion-transformers-for-vision-based-virtual-try-on-with-temporal-concatenation]]"
---
<!-- extracted by afk_extract from 2501.11325.pdf (11p) -->

# **CatV**[2] **TON: Taming Diffusion Transformers for Vision-Based Virtual Try-On with Temporal Concatenation** 

**Zheng Chong**[1,4*] , **Wenqing Zhang**[2*] , **Shiyue Zhang**[1] , **Jun Zheng**[1] , **Xiao Dong**[1] , **Haoxiang Li**[3] , **Yiling Wu**[4] , **Dongmei Jiang**[4] , **Xiaodan Liang**[1,4] _[†]_ 

Sun Yat-Sen University[1] , National University of Singapore[2] , Pixocial Technology[3] , Pengcheng Laboratory[4] 

*[Equal][Contribution,] _[†]_[Corresponding][Author] 

https://github.com/Zheng-Chong/CatV[2] TON 

**==> picture [41 x 54] intentionally omitted <==**

**==> picture [80 x 107] intentionally omitted <==**

**==> picture [41 x 54] intentionally omitted <==**

**==> picture [241 x 118] intentionally omitted <==**

**----- Start of picture text -----**<br>
Image-Based Try-On<br>**----- End of picture text -----**<br>


**==> picture [38 x 54] intentionally omitted <==**

**==> picture [81 x 107] intentionally omitted <==**

**==> picture [38 x 54] intentionally omitted <==**

~~Video-Based Try-On~~ 

**==> picture [98 x 121] intentionally omitted <==**

**==> picture [348 x 121] intentionally omitted <==**

Figure 1. Examples of CatV[2] TON’s unified virtual try-on capabilities, demonstrating high-quality garment consistency across both imagebased and video-based try-on tasks, including dynamic long-video scenarios. 

## **Abstract** 

_Virtual try-on (VTON) technology has gained attention due to its potential to transform online retail by enabling realistic clothing visualization of images and videos. However, most existing methods struggle to achieve highquality results across image and video try-on tasks, especially in long video scenarios. In this work, we introduce CatV_[2] _TON, a simple and effective vision-based virtual tryon (V_[2] _TON) method that supports both image and video_ 

_try-on tasks with a single diffusion transformer model. By temporally concatenating garment and person inputs and training on a mix of image and video datasets, CatV_[2] _TON achieves robust try-on performance across static and dynamic settings. For efficient long-video generation, we propose an overlapping clip-based inference strategy that uses sequential frame guidance and Adaptive Clip Normalization (AdaCN) to maintain temporal consistency with reduced resource demands. We also present ViViD-S, a refined video try-on dataset, achieved by filtering back-facing_ 

1 

_frames and applying 3D mask smoothing for enhanced temporal consistency. Comprehensive experiments demonstrate that CatV_[2] _TON outperforms existing methods in both image and video try-on tasks, offering a versatile and reliable solution for realistic virtual try-ons across diverse scenarios._ 

## **1. Introduction** 

The rapid evolution of image and video synthesis techniques has driven significant advancements in downstream tasks, including vision-based virtual try-on, which can be categorized into image-based and video-based approaches. Image-based virtual try-on methods [7, 8, 25, 40, 44, 47, 48, 51, 57] have been extensively explored, achieving high levels of garment realism and detail on static images. Meanwhile, video-based virtual try-on [11, 13, 16, 23, 52], bolstered by recent advancements in video generation, has garnered increasing research interest for dynamic applications. 

However, current methods for image-based and videobased try-ons often rely on separate model designs and frameworks. For instance, warping-based methods [14, 25, 47, 48] are tailored specifically for image try-on, adjusting garment shapes to match each pose but unable to maintain temporal consistency across frames. Methods that utilize specialized networks like ReferenceNet or GarmentNet [37, 40, 44, 51, 57] achieve highly realistic try-on effects, but the added encoding networks increase computational overhead. While video-based try-on methods [13, 16, 52] have made progress and can perform image try-on tasks, their performance still lags behind models designed exclusively for image try-on. Based on these observations, we introduce CatV[2] TON, a streamlined vision-based virtual tryon diffusion transformer framework. By temporally concatenating garment and person inputs and training on a combination of image and video datasets, CatV[2] TON addresses both static and dynamic try-on scenarios within a single, cohesive model. With only 20% of the backbone parameters allocated to trainable components and no additional modules, it offers a flexible and efficient framework for diverse try-on applications. 

Besides, a critical challenge in video generation is producing long, temporally consistent sequences, which is often resource-intensive and susceptible to quality degradation over time. To generate high-quality long videos, many methods [4, 49] typically involve pre-training on short videos followed by fine-tuning on a limited set of long video data. However, the high hardware requirements and extended inference times needed to produce high-definition long videos significantly restrict their practical application. To address this, we propose an overlapping clip-based inference strategy that leverages preceding frames as temporal guidance and integrates Adaptive Clip Normalization 

(AdaCN) to ensure consistency while reducing resource costs. Specifically, during training, we use only short video segments with a probability of exposing preceding frames to the model, enabling frame-guided generation. At inference, long video sequences are generated in segments, with the final frames of the previous segment serving as guidance for the next. However, this sequential approach may introduce flickering and color mismatches between segments. To counteract these issues, Adaptive Clip Normalization (AdaCN) is introduced to rectify segments based on guiding frames, thereby maintaining coherence in segmented long video generation. 

Additionally, we identified several issues with current video try-on datasets. On the one hand, video try-on datasets include human videos with turning or rotating actions but usually contain only garment images in front view. Due to the lack of back-view garment information, generating realistic try-on results when the person is in a rearfacing pose becomes unfeasible. This discrepancy is particularly noticeable for garments with logos, text, or intricate designs, and Using these back-facing frames as ground truth in training will slow model convergence and reduce garment consistency. To address this, we trained a specialized recognition model to detect the person’s orientation in each frame, filtering out back-facing frames to create a cleaner dataset. On the other hand, while large quantities of continuous person videos are available as video try-on training data, the indispensable clothing-agnostic masks still rely on frame-by-frame processing using imagebased parsing models, which introduces temporal discontinuities. To reduce flickering and edge inconsistency caused by this frame-by-frame processing, we propose a 3D Mask Smoothing operation. This involves applying spatial and temporal average pooling to the clothing-agnostic masks followed by re-binarization, effectively reducing flicker and leakage of edge information across frames. 

In summary, the contributions of this work include: 

- We introduce CatV[2] TON, a simple yet efficient visionbased virtual try-on diffusion transformer that seamlessly handles both static and dynamic try-on scenarios. It is trained on a mixed image-video dataset with more than 80% parameters of the backbone frozen by temporally concatenating garment and person inputs. 

- We propose an overlapping clip-based inference strategy for long try-on video generation, utilizing preceding frames as guidance and applying Adaptive Clip Normalization (AdaCN) to reduce resource demands while maintaining temporal consistency. 

- We present ViViD-S, a refined video try-on dataset that has undergone noise data reduction and quality enhancement through specially trained recognition models and 3D mask smoothing, providing high-quality samples suitable for video try-on tasks. 

2 

- Extensive qualitative and quantitative evaluations on both image and video try-on datasets demonstrate that our approach outperforms existing baseline methods in both quantitative and qualitative analysis, as well as in in-thewild scenarios. 

## **2. Related Work** 

## **2.1. Video Synthesis and Generation** 

Significant advancements have been made in video synthesis and generation, especially in generating coherent and high-quality video sequences from text descriptions. Models like StableVideo [4] and Dreamix [29] leverage diffusion models to capture both content and style with temporal consistency, while ImagenVideo [20] and Make-AVideo [38] focus on high-resolution outputs with fine details across frames. Methods such as FateZero [34] and PVDM [55] emphasize inter-frame coherence, crucial for naturallooking animations. For portrait-relevant editing, maintaining identity and expression consistency across frames remains a key challenge. Approaches like [12] balance structural and content control to preserve facial identity, while MagicVideo [59] utilizes latent diffusion for smooth and temporally stable animations. Methods like Tune-AVideo [46] and Follow-Your-Pose [28] bring innovations in one-shot tuning and pose-guided generation, respectively, for realistic human animations in videos. Additionally, image animation methods like EasyAnimate [49], MagicAnimate [53], and AnimateAnyone [21] use transformerbased and reference-guided techniques to enhance temporal coherence and identity preservation for character-specific animations. However, despite these remarkable achievements, high-quality video synthesis still faces challenges in maintaining fine-grained detail consistency, particularly in subject-driven video generation. 

## **2.2. Vision-based Virtual Try-On** 

Vision-based virtual try-on includes both image-based and video-based approaches. Image-based try-on generates realistic garment fittings on a target person’s photo from a garment image. Methods such as OOTDiffusion [51], IDM-VTON [7], StableGarment [44], and OutfitAnyone [40], utilize diffusion models and dual-stream networks to achieve high-fidelity garment rendering, addressing challenges like pose variation and complex backgrounds. Methods like GP-VTON [48], DCI-VTON [14], WarpDiffusion [25], and GarDiff [42] use pre-warped garments as guidance to enhance realism and detail. Approaches like MMTryon [57], IMAGDressing-v1 [37], and Wear-AnyWay [5] allow for user-controlled try-ons with multimodal conditioning. Lightweight solutions, such as CatVTON [8], provide efficient methods by spatially concatenating garment and person images. Video-based try-on techniques 

aim to address temporal consistency and garment deformation across frames. Flow-based methods such as FWGAN [10] and ClothFormer [23] employ warping modules to manage garment alignment and occlusions over various poses and backgrounds. Diffusion-based models like ViViD [13], WildVidFit [16], and VITON-DiT [58] integrate garment encoding and pose tracking to ensure consistent fitting across diverse body movements. Meanwhile, approaches like Tunnel Try-On [52] enhance motion stability using Kalman filtering for commercially viable, smooth, and detailed garment displays. Despite these advancements, current methods are typically limited to single-domain applications, focusing solely on either image or video-based tryons. A unified vision-based try-on model, capable of seamless virtual garment fitting across both images and videos, remains an area for further exploration. 

## **3. Method** 

Our approach aims to develop a streamlined, efficient vision-based virtual try-on network that addresses high resource demands and continuity challenges in generating extended try-on videos. To this end, we propose a refined diffusion transformer model based on a pre-trained video generation model [49], achieving task adaptation with less than 20% of the parameters required for full training (see Section 3.1). Furthermore, we introduce a novel Overlapping Clip-Based Inference strategy along with Adaptive Clip Normalization (AdaCN), facilitating the segmented generation of long-form videos (see Section 3.2). 

## **3.1. Vision-based Try-On Diffusion Transformer** 

## **3.1.1. Input Conditions** 

As shown in Figure 2, CatV[2] TON takes as input images or videos of persons, clothing-agnostic masks, pose representations, and target garment images. These inputs are encoded by the video VAE encoder and projected into the latent space. The main backbone, DiT [32], generates the try-on result through multiple denoising steps, which is then decoded into a video by the video VAE decoder. 

**Person-Related Conditions.** We apply the mask to occlude the input person video or image, resulting in a masked person representation. This masked person is concatenated with the mask along the channel dimension as conditioning information. 

**Garment-Related Conditions.** We use an all-zero mask concatenated along the channel dimension to ensure alignment with the person input. The garment conditions are then concatenated with the person conditions along the temporal dimension. Pose guidance is crucial for maintaining motion continuity in dynamic video generation. 

**Pose Conditions.** We use DensePose [15] as the pose representation, which provides more detailed information com- 

3 

**==> picture [496 x 166] intentionally omitted <==**

**----- Start of picture text -----**<br>
frame ids RoPE<br>DiT Block<br>DensePose<br>weight init<br>Garment Condi1on<br>DiT Block 0<br>Try-on Video<br>DiT Block 1~N<br>Mask<br>Channel Concatena;on Temporal Concatena;on Element-wise Add<br>Masked Video<br>Norm Norm FFN<br>Self A&n<br>Video VAE Encoder Temporal Embed Video VAE Decoder<br>**----- End of picture text -----**<br>


Figure 2. Overview of the CatV[2] TON architecture. CatV[2] TON uses DiT [32] as the backbone, with the first DiT block duplicated as the Pose Encoder. The person and garment conditions are concatenated temporally as try-on conditions. The entire trainable portion consists only of the self-attention layers and Pose Encoder, accounting for less than 1/5 of the total parameters. 

pared to skeleton-based methods like OpenPose [2] and MMPose [9]. 

## **3.1.2. Network Structure** 

State-of-the-art video generation models [2, 49] commonly use Diffusion Transformers (DiTs) [32] as the backbone. To leverage pre-trained video generation models and speed up training, we adopt DiT as our backbone and initialize our weights from EasyAnimateV4 [49], removing the crossattention layers. This modification is made because our task does not require CLIP [35] or text conditions. Our network consists of _N_ stacked DiT blocks, each with temporal and positional embeddings (using RoPE [39]), and includes only self-attention, feedforward, and normalization layers. For the Pose Encoder, inspired by ControlNeXt [33], we encode DensePose [15] sequences by duplicating the first DiT block and unlocking training, then injecting it into the network using element-wise addition after the first backbone block. This approach is lightweight compared to ControlNeXt [33], as our pose encoder is integrated into the backbone, eliminating the need for feature normalization before element-wise addition. 

## **3.1.3. Training Strategy** 

Due to the use of pre-trained weights, and inspired by [8], we unlock only the self-attention layers that involve interactions. The Pose Encoder is fully unlocked, with only 89.90 M trainable parameters, so the entire trainable portion accounts for less than 20% of the backbone network’s total parameters. During training, we apply a 10% chance of dropping the garment condition, which enhances generation quality during inference by enabling classifier-free guidance [18]. To facilitate the generation of long videos in segments, we do not mask the first _k_ frames of the person video during training, setting their masks to all-zero. This 

allows the model to learn the ability to generate subsequent frames based on the preceding frames. In our experiments, this probability is set to 20%. 

## **3.2. Overlapping Clip-Based Inference** 

To address the high computational resource requirements of generating long video try-ons, we propose the Overlapping Clip-Based Inference strategy. Specifically, during training, we actively expose earlier frames as prompt frames to enable the model to learn the ability to continue generation (as described in Section 3.1.3). This allows the model to use the results from previous inference steps as prompts to continue generating during inference. 

In detail, as shown in Figure 3a, for a long video that requires try-on generation, we first divide it into _n_ clips, each containing repeated frames. For each clip, all frames are masked during inference, and after generating the results, the last _k_ frames of the clip are used as prompt frames for generating the next clip. This process repeats for the entire video. 

However, this inference method presents a challenge: when generating sequentially, the prompt frames, after denoising, may undergo feature shifts, leading to color discrepancies and motion misalignment. This results in relative discontinuities between clips in the final generated video. 

To address this issue, inspired by AdaIN [22], we extend it to the level of video clips and propose Adaptive Clip Normalization (AdaCN). As shown in Figure 3b, specifically, for a clip _x_ with prompt frames, we first compute the mean _µy_ and standard deviation _σy_ of the prompt frame features _y_ , and the mean _µx_ and standard deviation _σx_ of the denoised prompt frames _x_ 0. We then normalize the entire clip _x_ using these statistics: 

4 

**==> picture [170 x 24] intentionally omitted <==**

Finally, we decode the concatenated feature sequences from multiple generated clips, after removing the redundant parts, into a coherent video. 

## **4. Experiments** 

stage, each sample contained 32 frames, the batch size was reduced to 1. This progressive strategy allowed the model to first learn at lower resolutions before refining its performance at higher resolutions. For all stages, we applied the AdamW optimizer [27] with a constant learning rate of 1e-5 and gradient clipping set to 1.0. The entire training process was carried out on 4 NVIDIA A100 GPUs. Additionally, all model variants used in the ablation study were trained under identical hyperparameter settings to ensure fair comparison. 

## **4.1. Datasets** 

**Image Datasets.** We utilized two publicly available imagebased try-on datasets—VITON-HD [6] and DressCode [30]—comprising 11,647 and 48,392 paired training samples, respectively, to construct our image-based training dataset. Comparative experiments for image-based virtual try-on were conducted on the test sets of VITON-HD [6] and DressCode [30] datasets. 

**Video Datasets.** We constructed the ViViD-S dataset based on the ViViD [13] dataset. Specifically, we trained a human orientation classifier based on EfficientNet [41] to detect sequences of frontal frames from the videos, filtering for videos that contain more than 24 consecutive frontal frames. Consequently, we selected 6,064 videos with a total of 513,896 frontal frames from the 7,759 videos in the ViViD [13] training dataset to serve as our training set. Due to the impracticality of testing on thousands of videos from a time and cost perspective, we randomly selected 180 videos (60 for dresses, 60 for uppers, and 60 for bottoms) from the ViViD [13] test set, each containing 64 consecutive frontal frames, to form the test set. Additionally, we utilized the VVT [11] dataset, which is a standard video virtual try-on dataset comprising 791 paired person videos and clothing images, with a resolution of 192×256. The comparative experiments are also conducted on the VVT test set which comprises 130 videos. 

## **4.2. Implementation Details** 

We initialized the DiT backbone using the pre-trained weights from EasyAnimateV4 [50], which was fine-tuned based on HunyuanDiT [26], and employed the pre-trained MagViT [54] as the video VAE to handle video data. We employed a progressive training strategy across three stages. In the first stage, we trained the model at a resolution of 256×192 using four datasets—DressCode, VITON-HD, VIViD, and VVT. Each training video sample consisted of 72 frames, with a batch size of 16, and the training ran for 128K steps. In the second stage, we increased the resolution to 512×384 and focused on three datasets—DressCode, VITON-HD, and VIViD. The number of frames per training sample was reduced to 48, with a batch size of 8, and the training continued for 64K steps. Finally, in the third stage, we used a high resolution of 832×624, training on the DressCode, VITON-HD, and VIViD-S datasets. In this 

## **4.3. Metrics** 

For image-based try-on settings, we employ four widely used metrics: SSIM [45], LPIPS [56], FID [36], and KID [1]. SSIM and LPIPS are used to measure the similarity between two images, while FID and KID assess the similarity between two sets of images. In paired try-on tests, we use all four metrics, whereas in unpaired scenarios, we only use FID and KID. For video try-on scenarios, we use SSIM [45], LPIPS [56], and VFID with I3D [3] and ResNext as metrics to evaluate video quality. 

## **4.4. Qualitative Comparison** 

Figure 4, Figure 5 and Figure 6 present qualitative comparisons of our method with StableVITON [24], OOTDiffusion [51], and ViViD [13] on the ViViD-S test set for unpaired visual try-on with dress, lower and upper clothing. Our approach demonstrates superior performance in generating try-on videos with improved temporal coherence and garment consistency compared to other baseline methods. Figure 7 presents the results of our method applied to the same person in a video with different types of clothing changes. This demonstrates the capability of our approach to maintain texture consistency, clothing shape, and temporal coherence. Additional comparison results are provided in the supplementary materials. 

## **4.5. Quantitative Comparison** 

**Image-based Virtual Try-On.** We conducted quantitative comparisons with advanced image-based try-on methods [14, 24, 31, 43, 44, 48, 51] under both paired and unpaired settings on the VITON-HD [6] and DressCode [30] datasets. During inference for image try-on, we used the DDPM [19] noise schedule for 20 steps, with a CFG [17] strength set to 3.0. As shown in Table 1 and Table 2, although our approach is designed for unified visual tryon, it outperforms traditional image-based try-on methods across various metrics for generated image quality, particularly in the unpaired scenario. This demonstrates that our method exhibits strong generalization performance even when trained on a single dataset. 

**Video-based Virtual Try-On.** Due to the limited availability of open-source video try-on methods, we selected two relatively high-performing image-based virtual try-on 

5 

**==> picture [496 x 185] intentionally omitted <==**

**----- Start of picture text -----**<br>
Clip 0 Clip 1 Clip 2 Clip n 3 3 4 5<br>0 1 2 3 0 3 4 5 0 5 6 7 … 1<br>2<br>DiT 3 μy σy μx σx<br>4<br>Normaliza)on<br>1 2 3 3 4 5 5 6 7 …<br>5 !𝑥= 𝜎! "#$%! ! +  𝜇!<br>AdaCN AdaCN …<br>N<br>0 Garment Latent x Video Input Latent<br>x Result Latent x Normalized Latent 3 4 5 5 6 7 … 3 4 5<br>(a) Overlapping Clip-based Inference (b) AdaCN<br>…<br>**----- End of picture text -----**<br>


Figure 3. Illustration of the Overlapping Clip-Based Inference strategy. (a) A long video is divided into _n_ overlapping clips, with each clip consisting of repeated frames. The last _k_ frames of each clip are used as prompt frames for generating the next clip. (b) Adaptive Clip Normalization (AdaCN) is applied to normalize the entire clip based on the mean and standard deviation of the prompt frame features and the denoised prompt frames, ensuring smooth continuity across clips in the generated video. 

**==> picture [239 x 235] intentionally omitted <==**

**----- Start of picture text -----**<br>
Source Garment Mask Stable OOTD ViViD Ours<br>**----- End of picture text -----**<br>


Figure 4. Qualitative comparison on the ViViD [13] dataset for dresses. We use Stable and OOTD as the short for StableVITON [24] and OOTDiffusion [51]. Additional comparison results are provided in the supplementary materials. Please zoom in for more details. 

**==> picture [238 x 234] intentionally omitted <==**

**----- Start of picture text -----**<br>
Source Garment Mask Stable OOTD ViViD Ours<br>**----- End of picture text -----**<br>


Figure 5. Qualitative comparison on the ViViD [13] dataset for lower. We use Stable and OOTD as the short for StableVITON [24] and OOTDiffusion [51]. Additional comparison results are provided in the supplementary materials. Please zoom in for more details. 

methods [24, 51] for comparison. These methods were evaluated using frame-by-frame inference and by applying MagicAnimate [53] to generate videos from the first frame. Additionally, we compared our approach with the videobased method ViViD [13]. The comparison results on the 

ViViD-S test set are presented in Table 4, where our method outperforms others in both paired and unpaired scenarios across all evaluation metrics. We also compared our approach with other video try-on methods on the VVT [11] dataset, as shown in Table Table 3. Due to the limited avail- 

6 

**==> picture [239 x 236] intentionally omitted <==**

**----- Start of picture text -----**<br>
Source Garment Mask Stable OOTD ViViD Ours<br>**----- End of picture text -----**<br>


Figure 6. Qualitative comparison on the ViViD [13] dataset for upper. We use Stable and OOTD as the short for StableVITON [24] and OOTDiffusion [51]. Additional comparison results are provided in the supplementary materials. Please zoom in for more details. 

|Methods|Paired<br>|Paired<br>|Paired<br>|Unpaired<br>|
|---|---|---|---|---|
||SSIM_↑_FID_↓_KID_↓_LPIPS_↓_|||FID_↓_KID_↓_|
|StableGarment [44]<br>MV-VTON [43]<br>LaDI-VTON [31]<br>DCI-VTON [14]<br>OOTDiffusion [51]<br>GP-VTON [48]<br>StableVTON [24]|0.8029 <br>0.8083 <br>0.8603 <br>0.8620<br>0.8187<br>0.8701<br>0.8543|15.567 8.519<br> 15.442 7.501<br> 11.386 7.248<br>9.408<br>4.547<br>9.305<br>4.086<br>8.726<br>3.944<br>**6.439**<br>**0.942**|0.1042<br>0.1171<br>0.0733<br>0.0606<br>0.0876<br>0.0585<br>0.0905|17.115 8.851<br>17.900 3.861<br>14.648 8.754<br>12.531 5.251<br>12.408 4.689<br>11.844 4.310<br>**11.054** 3.914|
|CatV2TON (Ours)|**0.8902**|8.095<br>2.245|**0.0570**|11.222<br>**2.986**|



Table 1. Quantitative comparison with other methods on VITONHD [6] dataset. The best and second-best results are demonstrated in **bold** and underlined, respectively. 

ability of open-source implementations, we reproduced the results of ViViD [13] using its code, while for other methods, we directly adopted the results reported in their papers. The results demonstrate that our method outperforms the others. During inference for video try-on, we used the DDPM [19] noise schedule for 15 steps, with a CFG [17] strength set to 2.5. 

## **4.6. Ablation Studies** 

To validate the contribution of different components or strategies to the final performance, we conducted ablation experiments on PoseNet, training data (ViViD [13] dataset or ViViD-S dataset), and AdaCN. 

|Methods|Paired<br>SSIM_↑_FID_↓_KID_↓_LPIPS_↓_|Paired<br>SSIM_↑_FID_↓_KID_↓_LPIPS_↓_|Unpaired|
|---|---|---|---|
||||FID_↓_KID_↓_|
|GP-VTON [48]<br>LaDI-VTON [31]<br>IDM-VTON [7]<br>OOTDiffusion [51|0.7711 9.927 4.610<br>0.7656 9.555 4.683<br>0.8797 6.821 2.924<br>]<br>0.8854<br>**4.610 0.955**|0.1801<br>0.2366<br>0.0563<br>0.0533|12.791 6.627<br>10.676 5.787<br>9.546<br>4.320<br>12.567 6.627|
|CatV2TON (Ours)|<br>**0.9222**<br>5.722<br>2.338|**0.0367**|**8.627**<br>**3.838**|



Table 2. Quantitative comparison with other methods on DressCode [30] dataset. The best and second-best results are demonstrated in **bold** and underlined, respectively. 

|Methods<br>|Paired|Paired|Unpaired|
|---|---|---|---|
||VFID_I ↓_VFID_R ↓_SSIM_↑_LPIPS_↓_||VFID_I ↓_VFID_R ↓_|
|FW-GAN [11]<br>MV-TON [7]<br>ClothFormer [23]<br>WildVidFit [16]<br>ViViD [13]|8.019<br>8.367<br>3.967<br>4.202<br>3.793|0.1215<br>0.675<br>0.283<br>0.0972<br>0.853<br>0.233<br>0.0505<br>**0.921**<br>0.081<br>-<br>-<br>-<br>0.0348<br>0.822<br>0.107|-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>3.994<br>0.0416|
|CatV2TON(Ours)|**1.778**|**0.0103**<br>0.900<br>**0.0385**|**1.902**<br>**0.0141**|



Table 3. Quantitative comparison with other methods on VVT [11] dataset. The best and second-best results are demonstrated in **bold** and underlined, respectively. 

|Methods|Paired|Unpaired|
|---|---|---|
||VFID_I ↓_VFID_R ↓_SSIM_↑_LPIPS_↓_|VFID_I ↓_VFID_R ↓_|
|StableVITON [24]<br>OOTDiffusion [51]<br>IDM-VTON [7]|34.2446<br>0.7735<br>0.8019<br>0.1338<br>29.5253<br>3.9372<br>0.8087<br>0.1232<br>20.0812<br>0.3674<br>0.8227<br>0.1163|36.8985<br>0.9064<br>35.3170<br>5.7078<br>25.4972<br>0.7167|
|StableVITON+AM<br>OOTDiffusion+AM<br>IDM-VTON+AM|19.9239<br>0.7586<br>0.8207<br>0.1291<br>22.0262<br>0.8283<br>19.3173<br>0.9382<br>0.8154<br>0.1244<br>23.3938<br>1.1485<br>18.2048<br>0.4481<br>0.8252<br>0.1212<br>22.5881<br>0.5397||
|ViViD [13]|17.2924<br>0.6209<br>0.8029<br>0.1221<br>21.8032<br>0.8212||
|CatV2TON(Ours)|**13.5962**<br>**0.2963**<br>**0.8727**<br>**0.0639**|**19.5131**<br>**0.5283**|



Table 4. Quantitative comparison with other methods on ViViD dataset. The best and second-best results are demonstrated in **bold** and underlined, respectively. 

**PoseEncoder.** We trained a model without PoseEncoder, using the same hyperparameters as mentioned in Section 4.2, to assess the impact of adding PoseEncoder on performance. As shown in Table 5 and Figure 8, the performance decreased without pose information, and it exhibited defects in handling complex poses. 

**AdaCN.** We introduced Adaptive Clip Normalization (AdaCN) for long video stitching inference in Section 3.2. To validate its effectiveness, we conducted ablation experiments to assess alignment from both metric and visual perspectives. As shown in Figure 8, without AdaCN, the inference results showed noticeable color mismatch accumulation, and the transitions between segments were not smooth. Table 5 further indicates that the use of AdaCN improves the quality of generated videos. 

7 

**==> picture [496 x 299] intentionally omitted <==**

Figure 7. Results of video try-on with CatV[2] TON. CatV[2] TON can perform video try-on with various types of garments, achieving high consistency in garment texture and shape. Additional comparison results are provided in the supplementary materials. Please zoom in for more details. 

**==> picture [239 x 189] intentionally omitted <==**

**----- Start of picture text -----**<br>
w AdaCN<br>w/o AdaCN<br>**----- End of picture text -----**<br>


Figure 8. Ablation visual results about AdaCN. When AdaCN is not utilized for inference, the clothing parts in the try-on results will exhibit color difference issues, which typically intensify with the increase in video length. 

|Variations|Paired|Paired|Paired|Paired|Unpaired|Unpaired|
|---|---|---|---|---|---|---|
||VFID_I ↓_VFID_R ↓_SSIM_↑_LPIPS_↓_||||VFID_I ↓_VFID_R ↓_||
|w/o Pose<br>w/o AdaCN||4.6398<br>4.5125|0.0562<br>0.8280<br>**0.0438**<br>**0.8507**|0.0990<br>**0.0842**|4.9894<br>4.4474|0.0598<br>0.0522|
|CatV2TON||**4.3657**|0.0491<br>0.8461|0.0983|**4.4324**|**0.0506**|



Table 5. Ablation results of AdaCN, PoseEncoder. The best and second-best results are demonstrated in **bold** and underlined, respectively. 

## **5. Limitations** 

**Resolution and Clarity.** In comparison to image data, video data, despite having a similar resolution, cannot achieve the same level of clarity due to the dynamic changes it exhibits. Even though the current resolution of 832 _×_ 624 is sufficient in terms of pixel count, it still falls short of meeting application requirements in terms of clarity. A higher-quality, higher-resolution video try-on dataset is crucial to addressing this issue. 

**Physical Laws in Video.** The key difference between video and image data lies in the dynamic nature of video, which must adhere strictly to physical laws. Otherwise, unrealistic artifacts may emerge, particularly in try-on tasks where 

8 

clothing movement during different actions is critical. Currently, there is a lack of foundational video generation models that can accurately simulate these physical behaviors. Achieving this would likely require larger-scale models and a higher-quality, larger-volume video dataset. 

## **6. Conclusion** 

In this work, we proposed CatV[2] TON, a simple and efficient diffusion transformer framework for both image and video virtual try-on tasks. By temporally concatenating garment and person inputs and training with a mixed imagevideo dataset, our model achieves high-quality results with only 20% of the backbone parameters as trainable components. To support long, temporally consistent try-on video generation, we introduced an overlapping clip-based inference strategy with Adaptive Clip Normalization (AdaCN), reducing resource demands while maintaining temporal continuity. Additionally, we propose a curated video try-on dataset, ViViD-S, created by filtering out back-view frames and applying 3D Mask Smoothing to enhance the temporal consistency of masks. Extensive experiments demonstrate that CatV[2] TON outperforms baseline methods in both quantitative and qualitative evaluations, marking a significant step forward for unified models in vision-based virtual try-on research. 

## **References** 

- [1] Mikołaj Bi´nkowski, Danica J. Sutherland, Michael Arbel, and Arthur Gretton. Demystifying mmd gans, 2021. 5 

- [2] Z. Cao, G. Hidalgo Martinez, T. Simon, S. Wei, and Y. A. Sheikh. Openpose: Realtime multi-person 2d pose estimation using part affinity fields. _IEEE Transactions on Pattern Analysis and Machine Intelligence_ , 2019. 4 

- [3] Joao Carreira and Andrew Zisserman. Quo vadis, action recognition? a new model and the kinetics dataset, 2018. 5 

- [4] Wenhao Chai, Xun Guo, Gaoang Wang, and Yan Lu. Stablevideo: Text-driven consistency-aware diffusion video editing. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 23040–23050, 2023. 2, 3 

- [5] Mengting Chen, Xi Chen, Zhonghua Zhai, Chen Ju, Xuewen Hong, Jinsong Lan, and Shuai Xiao. Wear-any-way: Manipulable virtual try-on via sparse correspondence alignment. _arXiv preprint arXiv:2403.12965_ , 2024. 3 

- [6] Seunghwan Choi, Sunghyun Park, Minsoo Lee, and Jaegul Choo. Viton-hd: High-resolution virtual try-on via misalignment-aware normalization. In _Proc. of the IEEE conference on computer vision and pattern recognition (CVPR)_ , 2021. 5, 7 

- [7] Yisol Choi, Sangkyung Kwak, Kyungmin Lee, Hyungwon Choi, and Jinwoo Shin. Improving diffusion models for virtual try-on. _arXiv preprint arXiv:2403.05139_ , 2024. 2, 3, 7 

- [8] Zheng Chong, Xiao Dong, Haoxiang Li, Shiyue Zhang, Wenqing Zhang, Xujie Zhang, Hanqing Zhao, and Xiaodan Liang. Catvton: Concatenation is all you need for virtual tryon with diffusion models. _arXiv preprint arXiv:2407.15886_ , 2024. 2, 3, 4 

- [9] MMPose Contributors. Openmmlab pose estimation toolbox and benchmark. https://github.com/openmmlab/mmpose, 2020. 4 

- [10] Haoye Dong, Xiaodan Liang, Xiaohui Shen, Bowen Wu, Bing-Cheng Chen, and Jian Yin. Fw-gan: Flow-navigated warping gan for video virtual try-on. In _Proceedings of the IEEE/CVF international conference on computer vision_ , pages 1161–1170, 2019. 3 

- [11] Haoye Dong, Xiaodan Liang, Xiaohui Shen, Bowen Wu, Bing-Cheng Chen, and Jian Yin. Fw-gan: Flow-navigated warping gan for video virtual try-on. In _2019 IEEE/CVF International Conference on Computer Vision (ICCV)_ , pages 1161–1170, 2019. 2, 5, 6, 7 

- [12] Patrick Esser, Johnathan Chiu, Parmida Atighehchian, Jonathan Granskog, and Anastasis Germanidis. Structure and content-guided video synthesis with diffusion models. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 7346–7356, 2023. 3 

- [13] Zixun Fang, Wei Zhai, Aimin Su, Hongliang Song, Kai Zhu, Mao Wang, Yu Chen, Zhiheng Liu, Yang Cao, and ZhengJun Zha. Vivid: Video virtual try-on using diffusion models, 2024. 2, 3, 5, 6, 7 

- [14] Junhong Gou, Siyu Sun, Jianfu Zhang, Jianlou Si, Chen Qian, and Liqing Zhang. Taming the power of diffusion models for high-quality virtual try-on with appearance flow. In _Proceedings of the 31st ACM International Conference on Multimedia_ , 2023. 2, 3, 5, 7 

- [15] Rıza Alp G¨uler, Natalia Neverova, and Iasonas Kokkinos. Densepose: Dense human pose estimation in the wild, 2018. 3, 4 

- [16] Zijian He, Peixin Chen, Guangrun Wang, Guanbin Li, Philip HS Torr, and Liang Lin. Wildvidfit: Video virtual tryon in the wild via image-based controlled diffusion models. _arXiv preprint arXiv:2407.10625_ , 2024. 2, 3, 7 

- [17] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance, 2022. 5, 7 

- [18] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance, 2022. 4 

- [19] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models, 2020. 5, 7 

- [20] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. _arXiv preprint arXiv:2210.02303_ , 2022. 3 

- [21] Li Hu. Animate anyone: Consistent and controllable imageto-video synthesis for character animation. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 8153–8163, 2024. 3 

- [22] Xun Huang and Serge Belongie. Arbitrary style transfer in real-time with adaptive instance normalization, 2017. 4 

9 

- [23] Jianbin Jiang, Tan Wang, He Yan, and Junhui Liu. Clothformer: Taming video virtual try-on in all module. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 10799–10808, 2022. 2, 3, 7 

- [24] Jeongho Kim, Gyojung Gu, Minho Park, Sunghyun Park, and Jaegul Choo. Stableviton: Learning semantic correspondence with latent diffusion model for virtual try-on, 2023. 5, 6, 7 

- [25] Xiu Li, Michael Kampffmeyer, Xin Dong, Zhenyu Xie, Feida Zhu, Haoye Dong, Xiaodan Liang, et al. Warpdiffusion: Efficient diffusion model for high-fidelity virtual tryon. _arXiv preprint arXiv:2312.03667_ , 2023. 2, 3 

- [26] Zhimin Li, Jianwei Zhang, Qin Lin, Jiangfeng Xiong, Yanxin Long, Xinchi Deng, Yingfang Zhang, Xingchao Liu, Minbin Huang, Zedong Xiao, Dayou Chen, Jiajun He, Jiahao Li, Wenyue Li, Chen Zhang, Rongwei Quan, Jianxiang Lu, Jiabin Huang, Xiaoyan Yuan, Xiaoxiao Zheng, Yixuan Li, Jihong Zhang, Chao Zhang, Meng Chen, Jie Liu, Zheng Fang, Weiyan Wang, Jinbao Xue, Yangyu Tao, Jianchen Zhu, Kai Liu, Sihuan Lin, Yifu Sun, Yun Li, Dongdong Wang, Mingtao Chen, Zhichao Hu, Xiao Xiao, Yan Chen, Yuhong Liu, Wei Liu, Di Wang, Yong Yang, Jie Jiang, and Qinglin Lu. Hunyuan-dit: A powerful multi-resolution diffusion transformer with fine-grained chinese understanding, 2024. 5 

- [27] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization, 2019. 5 

- [28] Yue Ma, Yingqing He, Xiaodong Cun, Xintao Wang, Siran Chen, Xiu Li, and Qifeng Chen. Follow your pose: Poseguided text-to-video generation using pose-free videos. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , pages 4117–4125, 2024. 3 

- [29] Eyal Molad, Eliahu Horwitz, Dani Valevski, Alex Rav Acha, Yossi Matias, Yael Pritch, Yaniv Leviathan, and Yedid Hoshen. Dreamix: Video diffusion models are general video editors. _arXiv preprint arXiv:2302.01329_ , 2023. 3 

- [30] Davide Morelli, Matteo Fincato, Marcella Cornia, Federico Landi, Fabio Cesari, and Rita Cucchiara. Dress code: Highresolution multi-category virtual try-on, 2022. 5, 7 

- [31] Davide Morelli, Alberto Baldrati, Giuseppe Cartella, Marcella Cornia, Marco Bertini, and Rita Cucchiara. LaDIVTON: Latent Diffusion Textual-Inversion Enhanced Virtual Try-On. In _Proceedings of the ACM International Conference on Multimedia_ , 2023. 5, 7 

- [32] William Peebles and Saining Xie. Scalable diffusion models with transformers. _arXiv preprint arXiv:2212.09748_ , 2022. 3, 4 

- [33] Bohao Peng, Jian Wang, Yuechen Zhang, Wenbo Li, MingChang Yang, and Jiaya Jia. Controlnext: Powerful and efficient control for image and video generation, 2024. 4 

- [34] Chenyang Qi, Xiaodong Cun, Yong Zhang, Chenyang Lei, Xintao Wang, Ying Shan, and Qifeng Chen. Fatezero: Fusing attentions for zero-shot text-based video editing. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 15932–15942, 2023. 3 

- [35] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen 

   - Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision, 2021. 4 

- [36] Maximilian Seitzer. pytorch-fid: FID Score for PyTorch. https://github.com/mseitzer/pytorch-fid, 2020. Version 0.3.0. 5 

- [37] Fei Shen, Xin Jiang, Xin He, Hu Ye, Cong Wang, Xiaoyu Du, Zechao Li, and Jinghui Tang. Imagdressing-v1: Customizable virtual dressing. _arXiv preprint arXiv:2407.12705_ , 2024. 2, 3 

- [38] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. Make-a-video: Text-to-video generation without text-video data. _arXiv preprint arXiv:2209.14792_ , 2022. 3 

- [39] Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding, 2023. 4 

- [40] Ke Sun, Jian Cao, Qi Wang, Linrui Tian, Xindi Zhang, Lian Zhuo, Bang Zhang, Liefeng Bo, Wenbo Zhou, Weiming Zhang, et al. Outfitanyone: Ultra-high quality virtual try-on for any clothing and any person. _arXiv preprint arXiv:2407.16224_ , 2024. 2, 3 

- [41] Mingxing Tan and Quoc V. Le. Efficientnet: Rethinking model scaling for convolutional neural networks, 2020. 5 

- [42] Siqi Wan, Yehao Li, Jingwen Chen, Yingwei Pan, Ting Yao, Yang Cao, and Tao Mei. Improving virtual try-on with garment-focused diffusion models. _arXiv preprint arXiv:2409.08258_ , 2024. 3 

- [43] Haoyu Wang, Zhilu Zhang, Donglin Di, Shiliang Zhang, and Wangmeng Zuo. Mv-vton: Multi-view virtual try-on with diffusion models. _arXiv preprint arXiv:2404.17364_ , 2024. 5, 7 

- [44] Rui Wang, Hailong Guo, Jiaming Liu, Huaxia Li, Haibo Zhao, Xu Tang, Yao Hu, Hao Tang, and Peipei Li. Stablegarment: Garment-centric generation via stable diffusion. _arXiv preprint arXiv:2403.10783_ , 2024. 2, 3, 5, 7 

- [45] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. _IEEE transactions on image processing_ , 13(4):600–612, 2004. 5 

- [46] Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Stan Weixian Lei, Yuchao Gu, Yufei Shi, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 7623–7633, 2023. 3 

- [47] Zhenyu Xie, Zaiyu Huang, Fuwei Zhao, Haoye Dong, Michael Kampffmeyer, Xin Dong, Feida Zhu, and Xiaodan Liang. Pasta-gan++: A versatile framework for highresolution unpaired virtual try-on, 2022. 2 

- [48] Zhenyu Xie, Zaiyu Huang, Xin Dong, Fuwei Zhao, Haoye Dong, Xijin Zhang, Feida Zhu, and Xiaodan Liang. Gpvton: Towards general purpose virtual try-on via collaborative local-flow global-parsing learning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 23550–23559, 2023. 2, 3, 5, 7 

- [49] Jiaqi Xu, Xinyi Zou, Kunzhe Huang, Yunkuo Chen, Bo Liu, MengLi Cheng, Xing Shi, and Jun Huang. Easyanimate: 

10 

A high-performance long video generation method based on transformer architecture. _arXiv preprint arXiv:2405.18991_ , 2024. 2, 3, 4 

- [50] Jiaqi Xu, Xinyi Zou, Kunzhe Huang, Yunkuo Chen, Bo Liu, MengLi Cheng, Xing Shi, and Jun Huang. Easyanimate: A high-performance long video generation method based on transformer architecture, 2024. 5 

- [51] Yuhao Xu, Tao Gu, Weifeng Chen, and Chengcai Chen. Ootdiffusion: Outfitting fusion based latent diffusion for controllable virtual try-on. _arXiv preprint arXiv:2403.01779_ , 2024. 2, 3, 5, 6, 7 

- [52] Zhengze Xu, Mengting Chen, Zhao Wang, Linyu Xing, Zhonghua Zhai, Nong Sang, Jinsong Lan, Shuai Xiao, and Changxin Gao. Tunnel try-on: Excavating spatial-temporal tunnels for high-quality virtual try-on in videos. _arXiv preprint arXiv:2404.17571_ , 2024. 2, 3 

- [53] Zhongcong Xu, Jianfeng Zhang, Jun Hao Liew, Hanshu Yan, Jia-Wei Liu, Chenxu Zhang, Jiashi Feng, and Mike Zheng Shou. Magicanimate: Temporally consistent human image animation using diffusion model. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 1481–1490, 2024. 3, 6 

- [54] Lijun Yu, Yong Cheng, Kihyuk Sohn, Jos´e Lezama, Han Zhang, Huiwen Chang, Alexander G Hauptmann, MingHsuan Yang, Yuan Hao, Irfan Essa, and Lu Jiang. MAGVIT: Masked generative video transformer. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , 2023. 5 

- [55] Sihyun Yu, Kihyuk Sohn, Subin Kim, and Jinwoo Shin. Video probabilistic diffusion models in projected latent space. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 18456–18466, 2023. 3 

- [56] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In _CVPR_ , 2018. 5 

- [57] Xujie Zhang, Ente Lin, Xiu Li, Yuxuan Luo, Michael Kampffmeyer, Xin Dong, and Xiaodan Liang. Mmtryon: Multi-modal multi-reference control for high-quality fashion generation. _arXiv preprint arXiv:2405.00448_ , 2024. 2, 3 

- [58] Jun Zheng, Fuwei Zhao, Youjiang Xu, Xin Dong, and Xiaodan Liang. Viton-dit: Learning in-the-wild video try-on from human dance videos via diffusion transformers. _arXiv preprint arXiv:2405.18326_ , 2024. 3 

- [59] Daquan Zhou, Weimin Wang, Hanshu Yan, Weiwei Lv, Yizhe Zhu, and Jiashi Feng. Magicvideo: Efficient video generation with latent diffusion models. _arXiv preprint arXiv:2211.11018_ , 2022. 3 

11 

