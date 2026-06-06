---
type: paper-fulltext
slug: dpidm-temporal-consistent-video
arxiv_id: null
source_path: /Users/saris.kia.adm/.paper-scholar/dpidm-temporal-consistent-video/2505.16980.md
paper: "[[dpidm-temporal-consistent-video]]"
---
<!-- extracted by afk_extract from 2505.16980.pdf (10p) -->

# **Pursuing Temporal-Consistent Video Virtual Try-On via Dynamic Pose Interaction** 

Dong Li[1] , Wenqi Zhong[2] _[∗]_ , Wei Yu[1] , Yingwei Pan[1] , Dingwen Zhang[2] , Ting Yao[1] , Junwei Han[2] , and Tao Mei[1] 

> 1 HiDream.ai Inc. 2 Northwest Polytechnical University, Xi’an, China 

lidong@hidream.ai, wenqizhong@mail.nwpu.edu.cn, _{_ yuwei, pandy _}_ @hidream.ai zhangdingwen2006yyy@gmail.com, tiyao@hidream.ai, junweihan2010@gmail.com 

tmei@hidream.ai 

## **Abstract** 

_Video virtual try-on aims to seamlessly dress a subject in a video with a specific garment. The primary challenge involves preserving the visual authenticity of the garment while dynamically adapting to the pose and physique of the subject. While existing methods have predominantly focused on image-based virtual try-on, extending these techniques directly to videos often results in temporal inconsistencies. Most current video virtual try-on approaches alleviate this challenge by incorporating temporal modules, yet still overlook the critical spatiotemporal pose interactions between human and garment. Effective pose interactions in videos should not only consider spatial alignment between human and garment poses in each frame but also account for the temporal dynamics of human poses throughout the entire video. With such motivation, we propose a new framework, namely_ _**D** ynamic_ _**P** ose_ _**I** nteraction_ _**D** iffusion_ _**M** odels (DPIDM), to leverage diffusion models to delve into dynamic pose interactions for video virtual try-on. Technically, DPIDM introduces a skeleton-based pose adapter to integrate synchronized human and garment poses into the denoising network. A hierarchical attention module is then exquisitely designed to model intraframe human-garment pose interactions and long-term human pose dynamics across frames through pose-aware spatial and temporal attention mechanisms. Moreover, DPIDM capitalizes on a temporal regularized attention loss between consecutive frames to enhance temporal consistency. Extensive experiments conducted on VITON-HD, VVT and ViViD datasets demonstrate the superiority of our DPIDM against the baseline methods. Notably, DPIDM achieves VFID score of 0.506 on VVT dataset, leading to 60.5% improvement over the state-of-the-art GPD-VVTO approach._ 

> _∗_ Equal contribution. This work was performed at HiDream.ai. 

**==> picture [46 x 67] intentionally omitted <==**

**==> picture [39 x 51] intentionally omitted <==**

**==> picture [40 x 52] intentionally omitted <==**

**==> picture [30 x 39] intentionally omitted <==**

**==> picture [32 x 41] intentionally omitted <==**

**==> picture [40 x 54] intentionally omitted <==**

**==> picture [41 x 54] intentionally omitted <==**

**==> picture [31 x 41] intentionally omitted <==**

Figure 1. Given a garment and a person video, DPIDM generates a lifelike video that preserves the visual authenticity of the garment while adapting to the individual’s pose and physique. 

## **1. Introduction** 

Generative models have made tremendous progress in recent years, with wide-ranging applications across various fields, including computer vision [9, 36, 59], computer graphics [6, 13], aesthetic design [35], and medical research [18]. Virtual try-on is a quintessential generative task that aims to synthesize a lifelike image/video of the specific person wearing the provided garment. By harnessing the powerful generative capacities of diffusion models, this field has emerged as a promising domain with versatile applications in immersive e-commerce and short-form video platforms. 

Prior research mainly focused on image-based virtual try-on. The earlier approaches typically build on Generative Adversarial Networks (GANs) [7, 19, 39, 45], incorporating a warping module alongside a try-on generator. The warping module deforms clothing to align with the human body, and then the warped garment is fused with the person image through the try-on generator. However, in light of the recent emergence of UNet-based Latent Diffusion Models (LDMs) [5, 8, 50, 56, 62], researchers have progressively directed their focus towards these innovative models. A diffusionbased try-on network combines the warping and blending 

1 

processes into a unified operation of cross-attention without explicit segregation. By employing pre-trained text-toimage weights, these diffusion approaches demonstrate superior fidelity compared to their GAN-based counterparts. 

In recent years, researchers have endeavored to replicate the success of image try-on within the realm of video. One direct approach involves applying image try-on techniques to process videos frame by frame. However, this method often results in notable inter-frame inconsistencies. To address this challenge, various specialized designs have been explored for video virtual try-on [10, 26, 28, 61]. These approaches commonly integrate optical flow prediction modules to warp frames produced by the try-on generator. Despite reasonable performance, GAN-based methods encounter difficulties in handling garment-person misalignment, particularly when faced with inaccurate warping flow estimations. To harness the capabilities of pretrained image-based diffusion models, TunnelTry-on [51] and ViViD [12] utilize a pre-trained inpainting U-Net as the primary branch and integrate a reference U-Net to capture detailed clothing features. They enhance temporal consistency by incorporating standard temporal attention after each stage of the main U-Net. GPD-VVTO [46] takes a step further by integrating garment features into temporal attention, enabling the model to better preserve fidelity to the garment during temporal modeling. Despite showing promise, these methods struggle to maintain both visual integrity and motion consistency simultaneously, particularly when there is a significant stylistic gap between the garment being tried on and the original one. To alleviate these issues, we argue that two key ingredients should be taken into account. One is spatial alignment between human and garment poses within each frame. Such alignment is essential for virtual try-on as it ensures that clothing adapts appropriately to factors like coverage and wrinkles based on the individual’s posture. The other is the temporal dynamics of human poses across the entire video. A comprehensive understanding of this long-term interaction is paramount for upholding temporal coherence, given that human movements naturally progress as a continuum of interconnected poses. 

By consolidating the idea of modeling both spatial and temporal pose interactions in a video, we novelly present **D** ynamic **P** ose **I** nteraction **D** iffusion **M** odels (DPIDM) for boosting video virtual try-on. Specifically, DPIDM employs a dual-branch architecture, where the Main U-Net conducts denoising processes on the video, while the Garment U-Net extracts intricate details from the garment image and integrates them into the Main U-Net. This architecture facilitates a holistic understanding of the relationship between humans and garments within a unified feature space. To effectively leverage the spatiotemporal pose interactions in videos, we introduce a skeleton-based pose adapter that seamlessly integrates synchronized human and 

garment poses into the Main U-Net. With the adapter, we present a hierarchical attention module comprising four key components: a pose-aware spatial attention block, a temporal-shift attention block, a cross-attention block, and a pose-aware temporal attention block. The pose-aware spatial attention block plays a crucial role in capturing intricate intra-frame human-garment pose interactions, while the pose-aware temporal attention block models the longterm dynamics of human poses across frames. Moreover, a temporal regularized attention loss is defined on consecutive frames to ensure temporal coherence. As a result, our DPIDM excels in generating videos with meticulously aligned garment details and enhanced temporal stability. 

The main contribution of this work is the proposal of Dynamic Pose Interaction Diffusion Models that facilitate the video virtual try-on (VVTON) task. This also leads to the elegant view of how a diffusion model should be designed for excavating the pose-focused prior knowledge (e.g., spatial pose alignment and temporal pose dynamics) tailored to VVTON, and how to improve diffusion process with these amplified pose-focused guidance. Through an extensive set of experiments on VITON-HD, VVT, and ViViD datasets, DPIPM consistently achieves superior results over state-ofthe-art methods in both image and video virtual try-on tasks. 

## **2. Related Works** 

## **2.1. Image-based Virtual Try-on** 

The development of image-based virtual try-on is largely inspired by the success of GANs [15], leading to pioneering works such as [7, 11, 19, 31, 33, 45]. A standard GAN-based virtual try-on framework typically adheres to a two-stage paradigm. The initial stage involves an image warping process, such as Thin Plate Spline (TPS) warping [19], which aligns the garment with the target individual’s body. In the second stage, a GAN-based try-on generator merges the warped attire with the target individual’s clothing-agnostic representation to produce authentic tryon outcomes. However, these methods heavily rely on the warping module to preserve garment details and often suffer from performance degradation due to inaccurate warping. 

Recently, with the powerful generative capabilities of Latent Diffusion Models (LDMs), many diffusion-based models have been introduced to generate more realistic virtual try-on results [5, 44, 52, 54, 56, 62]. TryOnDiffusion [62] was the first to propose a dual U-Net architecture that simultaneously preserves garment details and implicitly warps the garment to match the target pose. StableVITON [56] introduced a zero cross-attention block to learn the semantic correspondence between the garment and the target person. Wear-any-way [5] utilized a sparse point control method to align person and garment features in feature space. MMTryon [57] presented a multi-modal, multi- 

2 

**==> picture [467 x 200] intentionally omitted <==**

**----- Start of picture text -----**<br>
Human Pose Garment Pose Human Diffusion Garment Diffusion<br>Feature Feature Feature Feature<br>Garment U-Net<br>... ... ...<br>× t × t<br>C C<br>EncoderCLIP Adapter (Zero Init)<br>十<br>t<br>EstimatorPose EncoderPose h 2w<br>Self-Attention<br>t<br>h<br>w<br>+ noise C ... ... ... Temporal-Shift Attention<br>t<br>Main U-Net Adapter (Zero Init) h w EncoderCLIP<br>Cross Attention<br>十<br>Pose-Aware Spatial Attention Temporal-Shift Attention Cross-Attention<br>Pose-Aware Temporal Attention C Concatenate 十 Element-wise Add Temporal Attention<br>Trainable Module Freeze Module Pose-Aligned Feature<br>masks masked images (a) Overall framework of DPIDM (b) Attention blocks in main U-Net<br>**----- End of picture text -----**<br>


Figure 2. (a) Overall architecture of DPIDM. DPIDM emplys a dual-branch architecture. The main U-Net processes a concatenated input comprising the noisy latent of the video, the latent of the cloth-agnostic video, and the cloth-agnostic mask sequence. The garment U-Net extracts fine-grained garment features, which are subsequently integrated into the main U-Net. The pose estimator is utilized to extract aligned human and garment poses, which are then fed into the attention modules of the main U-Net to guide the diffusion process. The VAE is not shown for clarity. (b) Detailed illustration of the proposed pose-aware attention module within the main U-Net. The module comprises pose-aware spatial attention, temporal-shift attention, cross-attention, and pose-aware temporal attention. The pose embeddings are seamlessly integrated into the attention module through a specialized pose adapter. (better viewed in color) 

garment virtual try-on approach that combines text and multiple reference images to create a multimodal embedding as a condition to control the diffusion model. IDM [8] proposed a RefNet-style architecture to align the feature space between garment and person. Although these diffusionbased methods have achieved high-fidelity results in singleimage inference, their application to video virtual try-on reveals a crucial limitation: the oversight of inter-frame relationships. This deficiency results in notable inter-frame inconsistencies, ultimately yielding unsatisfactory outcomes. 

## **2.2. Video-based Virtual Try-on** 

In recent studies, researchers have aimed to expand the capabilities of image-based try-on approaches into video applications. A key issue in video generation is inter-frame inconsistency. ClothFormer [26] leverages optical flow in the warping and blending processes to achieve temporal smoothing. Recently, diffusion-based video virtual try-on methods have been introduced to generate more realistic results [12, 21, 46, 51, 60]. For instance, TunnelTry-on [51] proposed a focus-tunnel method that crops the human image according to the pose map, aligning the try-on image to accommodate scenarios where the person is off-center in the frame. VTON-DiT [60] introduced a DiT architecture compatible with OpenSora, where the clothing-agnostic human image is provided via ControlNet. WildVidFit [21] proposed an image-based diffusion model and used video MAE to update the intermediate representations of the diffusion process to improve frame consistency. ViViD [12] 

introduced a large-scale video try-on dataset and proposed an additional reference-based U-Net. GPD-VVTON [46] went a step further by integrating garment features into temporal attention to generate spatio-temporally consistent results. While these explorations of video try-on make steady advancements, they currently struggle to accurately capture the dynamic interaction between clothing and the human body in real-world scenarios. Our method employs a hierarchical attention module to capture the critical spatiotemporal pose interactions between the human and the garment. 

## **3. Methodology** 

In this paper, we present **D** ynamic **P** ose **I** nteraction **D** iffusion **M** odels (DPIDM), a pioneering framework based on latent diffusion models designed to enhance the VVTON task. The complete architecture is depicted in Figure 2 (a). This section begins with a brief introduction to Stable Diffusion [39] in Section 3.1 to lay the groundwork for subsequent discussions. Subsequently, we offer a comprehensive overview of DPIDM in Section 3.2, and delve into the specific design details in Section 3.3-3.5. Finally, we encapsulate our training and inference strategies in Section 3.6. 

## **3.1. Preliminary** 

Stable Diffusion (SD) stands out as one of the most prevalent latent diffusion models within the community. It leverages a variational autoencoder (VAE) architecture, comprising an encoder _E_ and a decoder _D_ , to facilitate image rep- 

3 

resentations in the latent space. Additionally, a U-Net [40] _ϵθ_ is trained to denoise a Gaussian noise _ϵ_ with a conditioning input encoded by a CLIP text encoder [37] _τθ_ . Given an image **x** and a text prompt **y** , the denoising U-Net _ϵθ_ undergoes training by minimizing the loss function: 

**==> picture [228 x 24] intentionally omitted <==**

where _t_ denotes the time step of the forward diffusion process, and **z** _t_ is the noisy latent constructed by adding Gaussian noise _ϵ ∼N_ (0 _,_ 1) to the encoded image _E_ ( **x** ). During the inference stage, **z** _t_ is randomly sampled from a Gaussian distribution and iteratively denoised to derive **z** 0 following a predefined sampling schedule [23, 41]. Finally, the latent decoder _D_ decodes **z** 0 back into the image space. 

## **3.2. Overview** 

The overall architecture of DPIDM is depicted in Fig. 2. DPIDM takes a reference garment image _**G** ∈_ R[3] _[×][H][×][W]_ and a source human video _**I** S ∈_ R _[T][ ×]_[3] _[×][H][×][W]_ as input, where _H_ and _W_ represent the height and width of each frame, and _T_ signifies the video length. Its objective is to generate a realistic video _**I**_[ˆ] _∈_ R _[T][ ×]_[3] _[×][H][×][W]_ showing the individual in _**I** S_ adorned in the garment _**G**_ . 

We approach the video virtual try-on as a video inpainting challenge, aimed at integrating the garment onto clothing-agnostic areas. To achieve this, DPIDM employs a dual-branch architecture. The main U-Net is the inpainting model initialized with the pre-trained weights from SD. It takes in a 9-channel tensor comprising 4 channels for the noisy latent of the video, 4 channels for the clothingagnostic video latent, and 1 channel for the binary agnostic mask sequence. In contrast to the original SD model, which utilizes text embeddings to guide the diffusion process, we have substituted these embeddings with image embeddings derived from a CLIP image encoder representing the garment image. While the CLIP image embedding effectively captures the overall colors and textures of the garment, it may not retain finer details. To address this, we also use a Garment U-Net to extract detailed garment features. The Garment U-Net follows a standard text-to-image diffusion model with a 4-channel input. This dual-branch architecture has proven to be effective by previous works [12, 46, 51]. 

To further enhance the generation, we incorporate human and garment poses as supplementary guidance to refine the diffusion process. We develop a light-weight pose encoder to extract features of the pose maps and a pose adapter to inject these features into the attention modules of the main U-Net. The human/garment pose maps are derived from the input video/image with pose estimator. 

## **3.3. Pose Estimator** 

We directly employ DW-Pose [53] to estimate human poses _**P** h_ from video frames. However, there is a lack of an open- 

**==> picture [39 x 52] intentionally omitted <==**

**==> picture [39 x 52] intentionally omitted <==**

**==> picture [39 x 52] intentionally omitted <==**

**==> picture [39 x 52] intentionally omitted <==**

**==> picture [38 x 52] intentionally omitted <==**

**==> picture [38 x 52] intentionally omitted <==**

Figure 3. Visualization of predicted human and garment poses. 

source garment pose estimator currently. To establish alignment between human and garment poses, we train a garment pose estimator to predict a set of landmarks _**P** g_ that correspond to various parts of the human pose _**P** h_ . The ground truth of _**P** g_ is labeled manually, with the number of landmarks varying based on the garment types. For instance, in upper clothing, _|_ _**P** g|_ =9, encompassing landmarks such as the neck, shoulders, elbows, wrists, and hips. These landmarks serve as explicit indicators of the alignment between human and garment poses, and therefore can be used as guidance to improve the diffusion process. The visualization of human and garment poses is shown in Fig. 3. 

## **3.4. Dynamic Pose Interaction** 

To facilitate the dynamic pose interaction, we propose a hierarchical attention module consisting of a pose-aware spatial attention block, a temporal-shift attention block, a crossattention block, and a pose-aware temporal attention block. 

**Pose-aware Spatial Attention (PASA)** . To capture intricate intra-frame human-garment pose interactions, we inject the features of human and garment poses into the spatial attention layer. In our baseline, the human feature map **f** _h_ from the main U-Net is concatenated with the corresponding garment feature map **f** _g_ from the garment U-Net to jointly compute self-attention, which is computed as: 

**==> picture [182 x 11] intentionally omitted <==**

where **f** = [ **f** _h,_ **f** _g_ ] is the concatenation of human feature map and garment feature map in the spatial dimension, and _ψq, ψk, ψv_ are linear projections. Afterward, only the half feature from the main U-Net undergoes further computation. This attention layer enables garment features extracted by the garment U-Net to be integrated into the main U-Net. 

To enable the correspondence control with pose guidance, we modify this attention layer by adding the pose embeddings of the human and garment. Formally, given the human pose embedding **p** _h_ and garment pose embedding **p** _g_ extracted from the pose encoder, we first concatenate them in spatial dimension **p** = [ **p** _h,_ **p** _g_ ]. Then we utilize a pose adapter to project **p** into the diffusion feature space. The pose adapter employs two fully connected (FC) layers with an intermediate activation layer. The first FC layer maps the input to a lower-dimensional space, while the second FC layer maps it back to the original dimensional. The pose adapter can be written as: 

**==> picture [198 x 11] intentionally omitted <==**

4 

where **W** _up_ and **W** _down_ are the learnable weight matrix. To preserve the original feature space of diffusion models, we initialize **W** _up_ with zeros. Subsequently, we add the diffusion feature **f** and the adapter embedding _Adpt_ ( **p** ) together and feed them into the self-attention mechanism, leading to the revised form of Eq. 2: 

**==> picture [233 x 9] intentionally omitted <==**

In this way, when integrating the garment feature into the main U-Net, the feature aggregation process takes into account the alignment of human and garment poses. This implicit consideration results in the garment being deformed to better fit the natural pose of the person. 

Existing PoseGuider-style methods [24, 46, 51] typically achieve pose control by directly adding processed human pose images and noisy latents as inputs to the first layer of the diffusion network, without incorporating pose information into the attention modules. In contrast, our PASA introduces pose information into each layer’s attention modules through a pose adapter, enabling more precise and varying degrees of pose control. Moreover, besides human poses, we further inject garment poses into PASA, thus achieving intricate intra-frame human-garment pose alignment. 

**Temporal-shift Attention (TSA)** . In the original SD model, the attention block of the U-Net solely focuses on self-attention within individual frames, overlooking valuable information across frames. Current VVTON techniques predominantly integrate a temporal attention layer at each stage to capture temporal dependencies among video frames. Nevertheless, this approach can introduce a covariate shift in the feature space, potentially undermining the model’s generative capacity [58]. While a direct remedy could involve joint space-time 3D attention, this method escalates the complexity of attention calculations quadratically. Drawing inspiration from prior works [1, 32, 49], a novel approach involves employing temporal shift operations that utilize a 2D module to amalgamate spatial and temporal information by incorporating neighboring frame data into the current frame. Specifically, in addition to examining tokens within the present frame, a patch-level shifting operation transpires along the temporal dimension, transferring tokens from the preceding _L_ frames to the current frame, thereby constructing a novel latent feature frame. We concatenate the latent feature of the current frame **h** with the temporally shifted latent feature **h** _shift_ in the spatial dimension, thus establishing keys and values for subsequent self-attention mechanisms. The temporal-shift attention can be formally written as: 

**==> picture [224 x 14] intentionally omitted <==**

It notably reduces the computational load when juxtaposed with 3D attention. Furthermore, it empowers the model to discern short-term relationships between adjacent frames, enhancing temporal coherence during video generation. 

**Cross-Attention (CA)** . In contrast to the original SD model, which employs text embeddings to steer the diffusion process with cross-attention, we have replaced these embeddings with garment image embeddings generated from a CLIP image encoder. These embeddings encapsulate comprehensive global information like color and style, which serve as a valuable supplement to the detailed appearance characteristics provided by the Garment U-Net. 

**Pose-aware Temporal Attention (PATA)** . Although TSA can effectively capture short-term relationships between adjacent frames, it fails to consider the long-term temporal dynamics across the entire video. Previous studies [12, 17, 24] have introduced a plug-and-play motion module with temporal attention to enhance video smoothness. However, these approaches overlook the dynamic changes in human poses, which are crucial for the VVTON task where garment appearance must adapt to individual poses. To address this limitation, we propose a pose-aware temporal attention. Similar to PASA, our method involves feeding the human pose embedding **p** _h_ into a pose adapter. This adapted embedding is then element-wise added to the output feature of the cross-attention block, followed by standard temporal attention. Our experiments have demonstrated that this straightforward strategy significantly improves video continuity and enhances the realism of clothing dynamics in motion, as shown in Fig. 5. 

## **3.5. Temporal Regularized Attention Loss** 

Generally, SD is merely optimized with the mean-squared loss defined in Eq. 1, which treats all regions of the synthesized video equally without emphasizing temporal consistency. Inspired by the idea that self-attention maps within diffusion models capture the structural essence of generated content [3, 25, 29, 42], we employ the _temporal regularized attention loss_ . This loss function is designed to minimize variations in self-attention maps across successive frames: 

**==> picture [190 x 31] intentionally omitted <==**

where _A_[(] _i[j]_[)] represents the self-attention map of the PASA block in the _i_ -th layer on _j_ -th frame. Specifically, We calculate _LT RA_ only on the last two decoder layers in the main U-Net, with _γi_ = 0 _._ 5 in our context. 

## **3.6. Training and Inference** 

**Training Strategy** . Different from previous multi-stage training approaches [46, 51], we employ a single-stage training strategy. The loss function is defined as follows: 

**==> picture [166 x 10] intentionally omitted <==**

where _λ_ is the hyper-parameter used to balance the MSE loss and the proposed TRA loss. When training with image datasets, the TSA and PATA blocks are omitted. As for 

5 

video datasets, we implement an image-video joint training strategy. Initially, we randomly pick _B_ videos from the dataset and extract a single frame randomly from each video for training. At this stage, solely the PASA and CA modules undergo training. Subsequently, in the following phase, another set of _B_ videos is randomly chosen and a sequence of _T_ consecutive frames is extracted from each video for training purposes. Here, the training focuses exclusively on the TSA and PATA modules. This process repeats iteratively until the training concludes. Here, _B_ represents the batch size, and _T_ symbolizes the length of video clips. This training methodology effectively reduces the GPU memory requirements and enhances the speed of convergence. 

To tackle discontinuous pose sequences, we also implement a condition dropping strategy during training. Specifically, each pose keypoint has a 0.05 probability of being dropped. When a keypoint of the current frame is dropped, the model is forced to refer to the poses of adjacent frames to infer the correct pose of the current frame. This strategy not only enhances the model’s robustness to pose estimation methods but also improves temporal consistency. 

**Inference Strategy** . To improve temporal coherence and smoothness in long videos, we follow [12, 46] and implement a sliding window approach during inference. It involves dividing the long video into overlapping segments of a specified length _T_ and conducting inference on each segment. For overlapping frames, the ultimate output is derived by averaging the results obtained from each inference. 

## **4. Experiments** 

## **4.1. Datasets** 

**Video Virtual Try-On** . We empirically verify and analyze the effectiveness of our DPIDM on two popular video virtual try-on datasets, VVT [10] and ViViD [12]. The VVT dataset contains 791 video clips with a resolution of 256 _×_ 192. Following the official settings[10, 51], we partition it into a training set of 661 clips and a test set of 130 clips. However, the VVT dataset exhibits limited diversity in terms of garment varieties and human actions. ViViD, a recent and challenging dataset, includes 9,700 video-image pairs at a resolution of 832 _×_ 624. It classifies garments into three categories: upper-body, lower-body, and dresses. The dataset is divided into 7,759 videos for training and 1,941 videos for testing. It is worth noting that due to inaccuracies in the provided cloth-agnostic masks for individuals in a back-facing posture, we excluded video segments depicting such scenarios from the original footage. Consequently, only the remaining segments were utilized for testing. 

**Image Virtual Try-On** . To further validate the effectiveness of our method, we carried out additional experiments on the widely recognized image-based virtual tryon dataset, VITON-HD [7]. This dataset comprises 13,679 

Table 1. Comparison results on the VVT dataset. VFID _I_ and VFID _R_ represent VFID _I_ 3 _D_ and VFID _ResNeXt_ respectively. _↑_ denotes higher is better, while _↓_ indicates lower is better. 

|**Method**|**SSIM**_↑_<br>**LPIPS**_↓_<br>**VFID**_I↓_<br>**VFID**_R↓_|**SSIM**_↑_<br>**LPIPS**_↓_<br>**VFID**_I↓_<br>**VFID**_R↓_|**SSIM**_↑_<br>**LPIPS**_↓_<br>**VFID**_I↓_<br>**VFID**_R↓_|**SSIM**_↑_<br>**LPIPS**_↓_<br>**VFID**_I↓_<br>**VFID**_R↓_|**SSIM**_↑_<br>**LPIPS**_↓_<br>**VFID**_I↓_<br>**VFID**_R↓_|
|---|---|---|---|---|---|
|||||||
|CP-VTON [45]<br>FW-GAN [10]<br>PBAFN [14]<br>ClothFormer [26]||0.459<br>0.675<br>0.870<br>0.921|0.535<br>0.283<br>0.157<br>0.081|6.361<br>8.019<br>4.516<br>3.967|12.10<br>12.15<br>8.690<br>5.048|
|||||||
|LaDI-VTON [34]<br>StableVITON [56]<br>TunnelTry-on [51]<br>ViViD [12]<br>VITON-DiT [60]<br>GPD-VVTO [46]<br>**DPIDM (Ours)**||0.878<br>0.876<br>0.913<br>**0.949**<br>0.896<br>0.928<br>0.930|0.190<br>0.076<br>0.054<br>0.068<br>0.080<br>0.056<br>**0.041**|5.880<br>4.021<br>3.345<br>3.405<br>2.498<br>1.280<br>**0.506**|-<br>5.076<br>4.614<br>5.074<br>0.187<br>-<br>**0.047**|



pairs of upper-body model and garment images, out of which 2,032 pairs were specifically reserved for testing. 

**Evaluation Metrics** . For image datasets, we use Structural Similarity (SSIM) [47] and Learned Perceptual Image Patch Similarity (LPIPS) [55] to measure the similarity between the generated image and the ground-truth. Additionally, Fr´echet Inception Distance(FID) [22] and Kernel Inception Distance(KID) [2] are employed to measure the quality and realism of the generated images. For video datasets, we further utilize Video Fr´echet Inception Distance (VFID) [43] to evaluate spatiotemporal consistency. Two CNN backbones, I3D [4] and 3D-ResNeXt101 [20], are adopted as feature extractors for VFID. 

## **4.2. Implementation Details** 

We initialize the main U-Net with the pre-trained SD v1.5 inpainting model and the garment U-Net with SD v1.5. The pose estimator is trained offline using YOLOv8 [38]. The pose encoder incorporates four convolutional layers to align the pose image with the noise latent at the same resolution. During training, all data is resized to a uniform resolution of 512 × 384. To improve model resilience, we employ data augmentation by randomly flipping images horizontally with a 50% probability. We utilize the Adam optimizer [27] with a consistent learning rate of 2 _×_ 10 _[−]_[5] . All the experiments are conducted using 16 NVIDIA A100 GPUs. Sequences of 24 frames are sampled, and the batch size _B_ is set to 32. The model is trained for 80,000 iterations. The hyper-parameter _λ_ is set to 10 _[−]_[3] for video datasets and 0 for image datasets. During inference, we utilize the DDIM [41] sampler and set the classifier-free guidance scale to 1.5. 

## **4.3. Quantitative Results** 

**Video Virtual Try-On** . We compare our DPIDM with a series of state-of-the-art virtual try-on methods, which can be grouped into two directions: GAN-based methods 

6 

Table 2. Comparison results on the ViViD dataset. 

|**Method**|**SSIM**_↑_<br>**LPIPS**_↓_<br>**VFID**_I↓_<br>**VFID**_R↓_|**SSIM**_↑_<br>**LPIPS**_↓_<br>**VFID**_I↓_<br>**VFID**_R↓_|**SSIM**_↑_<br>**LPIPS**_↓_<br>**VFID**_I↓_<br>**VFID**_R↓_|**SSIM**_↑_<br>**LPIPS**_↓_<br>**VFID**_I↓_<br>**VFID**_R↓_|
|---|---|---|---|---|
||||||
|LaDI-VTON [34]<br>CAT-DM [54]<br>ViViD [12]<br>**DPIDM (Ours)**||0.824<br>0.826<br>0.846<br>**0.883**|0.164<br>0.162<br>0.118<br>**0.081**|8.283<br>1.185<br>9.354<br>2.139<br>1.894<br>0.870<br>**0.488**<br>**0.090**|



Table 3. Quantitative results of image-based virtual try-on task on the VITON-HD dataset. FID _u_ /KID _u_ stands for the FID/KID score in unpaired setting. Note that the KID score is multiplied by 1000. 

|**Method**|**SSIM**_↑_<br>**LPIPS**_↓_<br>**FID**_u↓_<br>**KID**_u↓_|**SSIM**_↑_<br>**LPIPS**_↓_<br>**FID**_u↓_<br>**KID**_u↓_|
|---|---|---|
||||
|VITON-HD [7]<br>HR-VITON [30]<br>GP-VTON [48]||0.862<br>0.117<br>12.12<br>3.23<br>0.876<br>0.096<br>12.31<br>3.81<br>0.890<br>0.085<br>9.82<br>1.42|
||||
|LaDI-VTON [34]<br>WearAnyWay [5]<br>DCI-VTON [16]<br>StableVITON [56]<br>CAT-DM [54]<br>IDM [8]<br>GPD-VVTO [46]<br>**DPIDM (Ours)**||0.875<br>0.091<br>9.32<br>1.55<br>0.877<br>0.078<br>8.16<br>0.78<br>0.890<br>0.072<br>8.77<br>0.89<br>0.878<br>0.075<br>9.43<br>1.54<br>0.877<br>0.080<br>8.93<br>1.37<br>0.881<br>0.078<br>8.60<br>0.55<br>0.891<br>0.070<br>8.57<br>0.78<br>**0.893**<br>**0.067**<br>**8.15**<br>**0.32**|



[10, 14, 26, 45] and Diffusion-based methods [12, 34, 46, 51, 54, 56, 60]. Table 1 summarizes the performance comparisons on the VVT dataset. Notably, our DPIDM consistently outperforms other state-of-the-art methods across LPIPS and VFID metrics. Although ViViD [12] attains a higher SSIM score, its advantage is expected due to the utilization of a significantly larger volume of high-resolution videos during training, approximately six times more than the VVT dataset. Despite this, our DPIDM notably surpasses ViViD in the VFID metric, showcasing substantial enhancements in video coherence while preserving garment visual authenticity. ClothFormer improves FW-GAN by introducing an appearance-flow tracking module to ensure temporal consistency in garment warping. TunnelTry-on and ViViD further boost the performances by leveraging the capabilities of pre-trained diffusion models. Additionally, GPD-VVTO introduces garment-aware temporal attention to enhance temporal consistency, leading to a clear VFID score boost. However, existing approaches often neglect crucial spatiotemporal pose interactions between individuals and garments. In contrast, our DPIDM effectively models both intra-frame human-garment pose interactions and long-term human pose dynamics across frames through pose-aware attention mechanisms, leading to superior results in the VVTON task. Specifically, our DPIDM achieves a VFID _I_ score of 0.506, marking a substantial 60.5% relative improvement over the top competitor GPD-VVTO. Note that GPD-VVTO employs SD v2.1 for model initialization, whereas we choose SD v1.5 for fewer parameters. 

**==> picture [25 x 33] intentionally omitted <==**

**==> picture [52 x 69] intentionally omitted <==**

**==> picture [52 x 69] intentionally omitted <==**

**==> picture [52 x 69] intentionally omitted <==**

**==> picture [52 x 69] intentionally omitted <==**

**==> picture [25 x 37] intentionally omitted <==**

**==> picture [25 x 33] intentionally omitted <==**

**==> picture [52 x 69] intentionally omitted <==**

**==> picture [52 x 69] intentionally omitted <==**

**==> picture [52 x 69] intentionally omitted <==**

**==> picture [52 x 69] intentionally omitted <==**

**==> picture [25 x 37] intentionally omitted <==**

**==> picture [52 x 69] intentionally omitted <==**

**==> picture [25 x 33] intentionally omitted <==**

**==> picture [52 x 68] intentionally omitted <==**

**==> picture [52 x 69] intentionally omitted <==**

**==> picture [52 x 69] intentionally omitted <==**

**==> picture [25 x 37] intentionally omitted <==**

**==> picture [217 x 6] intentionally omitted <==**

**----- Start of picture text -----**<br>
Input LaDI-VTON CAT-DM ViViD DPIDM (Ours)<br>**----- End of picture text -----**<br>


Figure 4. Qualitative comparison on the ViViD dataset. Our DPIDM excels in seamlessly integrating the garment with the wearer and maintaining the visual integrity of the garment. 

Table 2 presents performance comparisons on the ViViD dataset. As observed with VVT, DPIDM consistently outperforms other VVTON methods across all metrics. which again evinces the pivotal merit of the pose-aware guidance for preserving spatiotemporal consistency in the generated videos. Particularly, DPIDM demonstrates a substantial relative improvement of 74.2% over ViViD in VFID _I_ . 

**Image Virtual Try-On** . Table 3 displays the quantitative results on the VITON-HD [7]. Following the official settings, we utilize SSIM and LPIPS metrics for the paired setting, and FID and KID for the unpaired setting. Our DPIDM consistently outperforms other methods across all metrics, showcasing its ability to produce authentic and lifelike images while maintaining the integrity of the initial garment structure. It is noteworthy that diffusion-based approaches exhibit lower FID and KID scores compared to GAN-based methods, suggesting that diffusion models excel in generating images with heightened fidelity. 

## **4.4. Qualitative Results** 

We conduct qualitative analysis from two aspects: visual authenticity of the garment and temporal consistency. When evaluating visual authenticity, we compare results from several virtual try-on methods using individual video frames, as illustrated in Figure 4. In the first row, LaDI-VTON and CAT-DM change the style of the target garment and introduce color discrepancies. While ViViD achieves a natural fit, it struggles to maintain the original patterns, leading to their displacement on the garment. In contrast, our method excels in seamlessly blending the garment with the wearer while preserving intricate fabric patterns. Moving 

7 

**==> picture [472 x 211] intentionally omitted <==**

**----- Start of picture text -----**<br>
Input ViViD DPIDM (Ours)<br>**----- End of picture text -----**<br>


Figure 5. Qualitative comparison on the ViViD dataset. Our DPIDM maintains temporal consistency even during substantial movements. 

Table 4. Performance contribution of each component in DPIDM. 

|_id_|PAA<br>TSA<br>TRA|**SSIM**_↑_<br>**LPIPS**_↓_<br>**VFID**_I↓_<br>**VFID**_R↓_|
|---|---|---|
||||
|(a)<br>(b)<br>(c)<br>(d)|✗<br>✗<br>✗<br>✓<br>✗<br>✗<br>✓<br>✓<br>✗<br>✓<br>✓<br>✓|0.893<br>0.084<br>3.451<br>2.435<br>0.925<br>0.050<br>1.068<br>0.153<br>0.929<br>0.043<br>0.721<br>0.075<br>**0.930**<br>**0.041**<br>**0.506**<br>**0.047**|



to the second row with a pair of striped shorts, competitors maintain the garment’s style but exhibit noticeable artifacts around the legs. Conversely, our method adeptly addresses these issues by naturally supplementing missing skin tones while upholding the garment’s visual integrity. This trend continues in the subsequent example with a dress in the third row, highlighting the effectiveness of DPIDM. To further evaluate temporal consistency, we extract four frames at equal intervals from the generated video and present them in Fig. 5. While ViViD achieves a natural fit in individual frames, it struggles to maintain garment details and temporal consistency across frames, especially during person movements. In contrast, our DPIDM consistently delivers high-quality results that preserve garment details and ensure temporal consistency without introducing artifacts, even during actions involving significant movements. 

## **4.5. Ablation Studies** 

In this study, we analyze the impact of each design element within DPIDM on the overall performance using the VVT dataset. As shown in Table 4, the proposed elements are denoted as follows: _PAA_ refers to pose-aware attention, _TSA_ to temporal-shift attention, and _TRA_ to temporal regularized attention loss. We start with a basic model (a) that directly inserts standard temporal attention after each stage of the main U-Net. Subsequently, in (b), we substitute the 

standard spatial/temporal attention with our proposed poseaware spatial/temporal attention. Notably, the introduction of _PAA_ yields a significant performance enhancement over (a), surpassing even the state-of-the-art methods detailed in Table 1. This not only validates the efficacy of our proposed pose-aware spatial/temporal attention but also underscores the critical role of integrating spatiotemporal pose interactions into the VVTON task. Furthermore, we introduce the temporal-shift attention and the temporal regularized attention loss in (c) and (d) respectively. While _TSA_ and _TRA_ did not result in significant enhancements in the SSIM and LPIPS metrics, they notably improved the VFID metric, highlighting their effectiveness in enhancing video consistency. Collectively, the innovative modules introduced in DPIDM enhance the visual integrity of the garment and improve the temporal consistency of the generated videos. 

## **5. Conclusion** 

In this work, we have presented Dynamic Pose Interaction Diffusion Models (DPIDM), a novel approach that capitalizes on spatial and temporal pose interactions to enhance video virtual try-on. Particularly, we study the problem from the viewpoint of employing intra-frame humangarment pose interactions and long-term human pose dynamics across frames. To verify our claim, we develop a hierarchical pose-aware attention module to integrate human and garment pose features into spatial and temporal attention, enhancing the temporal coherence of the generated videos while preserving the visual fidelity of the garments. Additionally, a new temporal regularized attention loss is devised to bolster consistency across successive frames. Through extensive experiments, we demonstrate that our DPIDM surpasses existing state-of-the-art methods on image-based and video-based virtual try-on datasets. 

8 

## **Acknowledgments** 

This work was supported in part by the Beijing Municipal Science and Technology Project No. Z241100001324002 and Beijing Nova Program No. 20240484681. 

## **References** 

- [1] Jie An, Songyang Zhang, Harry Yang, Sonal Gupta, Jia-Bin Huang, Jiebo Luo, and Xi Yin. Latent-shift: Latent diffusion with temporal shift for efficient text-to-video generation. _arXiv preprint arXiv:2304.08477_ , 2023. 5 

- [2] Mikołaj Bi´nkowski, Danica J Sutherland, Michael Arbel, and Arthur Gretton. Demystifying mmd gans. In _ICLR_ , 2018. 6 

- [3] Mingdeng Cao, Xintao Wang, Zhongang Qi, Ying Shan, Xiaohu Qie, and Yinqiang Zheng. Masactrl: Tuning-free mutual self-attention control for consistent image synthesis and editing. In _CVPR_ , 2023. 5 

- [4] Joao Carreira and Andrew Zisserman. Quo vadis, action recognition? a new model and the kinetics dataset. In _CVPR_ , 2017. 6 

- [5] Mengting Chen, Xi Chen, Zhonghua Zhai, Chen Ju, Xuewen Hong, Jinsong Lan, and Shuai Xiao. Wear-any-way: Manipulable virtual try-on via sparse correspondence alignment. In _ECCV_ , 2024. 1, 2, 7 

- [6] Zilong Chen, Feng Wang, Yikai Wang, and Huaping Liu. Text-to-3d using gaussian splatting. In _CVPR_ , 2024. 1 

- [7] Seunghwan Choi, Sunghyun Park, Minsoo Lee, and Jaegul Choo. Viton-hd: High-resolution virtual try-on via misalignment-aware normalization. In _CVPR_ , 2021. 1, 2, 6, 7 

- [8] Yisol Choi, Sangkyung Kwak, Kyungmin Lee, Hyungwon Choi, and Jinwoo Shin. Improving diffusion models for authentic virtual try-on in the wild. In _ECCV_ , 2024. 1, 3, 7 

- [9] Florinel-Alin Croitoru, Vlad Hondru, Radu Tudor Ionescu, and Mubarak Shah. Diffusion models in vision: A survey. _IEEE Transactions on Pattern Analysis and Machine Intelligence_ , 45(9):10850–10869, 2023. 1 

- [10] Haoye Dong, Xiaodan Liang, Xiaohui Shen, Bowen Wu, Bing-Cheng Chen, and Jian Yin. Fw-gan: Flow-navigated warping gan for video virtual try-on. In _CVPR_ , 2019. 2, 6, 7 

- [11] Haoye Dong, Xiaodan Liang, Yixuan Zhang, Xujie Zhang, Xiaohui Shen, Zhenyu Xie, Bowen Wu, and Jian Yin. Fashion editing with adversarial parsing learning. In _CVPR_ , 2020. 2 

- [12] Zixun Fang, Wei Zhai, Aimin Su, Hongliang Song, Kai Zhu, Mao Wang, Yu Chen, Zhiheng Liu, Yang Cao, and ZhengJun Zha. Vivid: Video virtual try-on using diffusion models. _arXiv preprint arXiv:2405.11794_ , 2024. 2, 3, 4, 5, 6, 7 

- [13] Xuehao Gao, Yang Yang, Zhenyu Xie, Shaoyi Du, Zhongqian Sun, and Yang Wu. Guess: Gradually enriching synthesis for text-driven human motion generation. _IEEE Transactions on Visualization and Computer Graphics_ , 30 (12):7518–7530, 2024. 1 

- [14] Yuying Ge, Yibing Song, Ruimao Zhang, Chongjian Ge, Wei Liu, and Ping Luo. Parser-free virtual try-on via distilling appearance flows. In _CVPR_ , 2021. 6, 7 

- [15] Ian J Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. _Advances in neural information processing systems_ , 27, 2014. 2 

- [16] Junhong Gou, Siyu Sun, Jianfu Zhang, Jianlou Si, Chen Qian, and Liqing Zhang. Taming the power of diffusion models for high-quality virtual try-on with appearance flow. In _ACM MM_ , 2023. 7 

- [17] Yuwei Guo, Ceyuan Yang, Anyi Rao, Yaohui Wang, Yu Qiao, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. In _ICLR_ , 2024. 5 

- [18] Ibrahim Ethem Hamamci, Sezgin Er, Anjany Sekuboyina, Enis Simsar, Alperen Tezcan, Ayse Gulnihan Simsek, Sevval Nil Esirgun, Furkan Almas, Irem Do˘gan, Muhammed Furkan Dasdelen, et al. Generatect: textconditional generation of 3d chest ct volumes. In _ECCV_ , 2025. 1 

- [19] Xintong Han, Zuxuan Wu, Zhe Wu, Ruichi Yu, and Larry S Davis. Viton: An image-based virtual try-on network. In _CVPR_ , 2018. 1, 2 

- [20] Kensho Hara, Hirokatsu Kataoka, and Yutaka Satoh. Can spatiotemporal 3d cnns retrace the history of 2d cnns and imagenet? In _CVPR_ , 2018. 6 

- [21] Zijian He, Peixin Chen, Guangrun Wang, Guanbin Li, Philip HS Torr, and Liang Lin. Wildvidfit: Video virtual tryon in the wild via image-based controlled diffusion models. In _ECCV_ , 2024. 3 

- [22] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. _Advances in neural information processing systems_ , 30, 2017. 6 

- [23] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. _Advances in neural information processing systems_ , 33:6840–6851, 2020. 4 

- [24] Li Hu. Animate anyone: Consistent and controllable imageto-video synthesis for character animation. In _CVPR_ , 2024. 5 

- [25] Jaeseok Jeong, Mingi Kwon, and Youngjung Uh. Trainingfree content injection using h-space in diffusion models. In _WACV_ , 2024. 5 

- [26] Jianbin Jiang, Tan Wang, He Yan, and Junhui Liu. Clothformer: Taming video virtual try-on in all module. In _CVPR_ , 2022. 2, 3, 6, 7 

- [27] Diederik P Kingma. Adam: A method for stochastic optimization. _arXiv preprint arXiv:1412.6980_ , 2014. 6 

- [28] Gaurav Kuppa, Andrew Jong, Xin Liu, Ziwei Liu, and TengSheng Moh. Shineon: Illuminating design choices for practical video-based virtual clothing try-on. In _WACV_ , 2021. 2 

- [29] Mingi Kwon, Seoung Wug Oh, Yang Zhou, Difan Liu, Joon-Young Lee, Haoran Cai, Baqiao Liu, Feng Liu, and Youngjung Uh. Harivo: Harnessing text-to-image models for video generation. In _ECCV_ , 2024. 5 

- [30] Sangyun Lee, Gyojung Gu, Sunghyun Park, Seunghwan Choi, and Jaegul Choo. High-resolution virtual try-on with 

9 

misalignment and occlusion-handled conditions. In _ECCV_ , 2022. 7 

- [31] Zhi Li, Pengfei Wei, Xiang Yin, Zejun Ma, and Alex C Kot. Virtual try-on with pose-garment keypoints guided inpainting. In _CVPR_ , 2023. 2 

- [32] Ji Lin, Chuang Gan, and Song Han. Tsm: Temporal shift module for efficient video understanding. In _CVPR_ , 2019. 5 

- [33] Davide Morelli, Matteo Fincato, Marcella Cornia, Federico Landi, Fabio Cesari, and Rita Cucchiara. Dress code: Highresolution multi-category virtual try-on. In _CVPR_ , 2022. 2 

- [34] Davide Morelli, Alberto Baldrati, Giuseppe Cartella, Marcella Cornia, Marco Bertini, and Rita Cucchiara. Ladi-vton: Latent diffusion textual-inversion enhanced virtual try-on. In _ACM MM_ , 2023. 6, 7 

- [35] Sangeun Oh, Yongsu Jung, Seongsin Kim, Ikjin Lee, and Namwoo Kang. Deep generative design: integration of topology optimization and generative models. _Journal of Mechanical Design_ , 141(11):111405, 2019. 1 

- [36] Yingwei Pan, Zhaofan Qiu, Ting Yao, Houqiang Li, and Tao Mei. To create what you tell: Generating videos from captions. In _ACM MM_ , 2017. 1 

- [37] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In _ICML_ , 2021. 4 

- [38] J Redmon. You only look once: Unified, real-time object detection. In _CVPR_ , 2016. 6 

- [39] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In _CVPR_ , 2022. 1, 3 

- [40] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In _MICCAI_ , 2015. 4 

- [41] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. _arXiv preprint arXiv:2010.02502_ , 2020. 4, 6 

- [42] Narek Tumanyan, Michal Geyer, Shai Bagon, and Tali Dekel. Plug-and-play diffusion features for text-driven image-to-image translation. In _CVPR_ , 2023. 5 

- [43] Thomas Unterthiner, Sjoerd Van Steenkiste, Karol Kurach, Raphael Marinier, Marcin Michalski, and Sylvain Gelly. Towards accurate generative models of video: A new metric & challenges. _arXiv preprint arXiv:1812.01717_ , 2018. 6 

- [44] Siqi Wan, Yehao Li, Jingwen Chen, Yingwei Pan, Ting Yao, Yang Cao, and Tao Mei. Improving virtual try-on with garment-focused diffusion models. In _ECCV_ , 2024. 2 

- [45] Bochao Wang, Huabin Zheng, Xiaodan Liang, Yimin Chen, Liang Lin, and Meng Yang. Toward characteristicpreserving image-based virtual try-on network. In _ECCV_ , 2018. 1, 2, 6, 7 

- [46] Yuanbin Wang, Weilun Dai, Long Chan, Huanyu Zhou, Aixi Zhang, and Si Liu. Gpd-vvto: Preserving garment details in video virtual try-on. In _ACM MM_ , 2024. 2, 3, 4, 5, 6, 7 

- [47] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. _IEEE transactions on image processing_ , 13(4):600–612, 2004. 6 

- [48] Zhenyu Xie, Zaiyu Huang, Xin Dong, Fuwei Zhao, Haoye Dong, Xijin Zhang, Feida Zhu, and Xiaodan Liang. Gpvton: Towards general purpose virtual try-on via collaborative local-flow global-parsing learning. In _CVPR_ , 2023. 7 

- [49] Zhen Xing, Qi Dai, Han Hu, Zuxuan Wu, and Yu-Gang Jiang. Simda: Simple diffusion adapter for efficient video generation. In _CVPR_ , 2024. 5 

- [50] Yuhao Xu, Tao Gu, Weifeng Chen, and Chengcai Chen. Ootdiffusion: Outfitting fusion based latent diffusion for controllable virtual try-on. _arXiv preprint arXiv:2403.01779_ , 2024. 1 

- [51] Zhengze Xu, Mengting Chen, Zhao Wang, Linyu Xing, Zhonghua Zhai, Nong Sang, Jinsong Lan, Shuai Xiao, and Changxin Gao. Tunnel try-on: Excavating spatial-temporal tunnels for high-quality virtual try-on in videos. In _ACM MM_ , 2024. 2, 3, 4, 5, 6, 7 

- [52] Xu Yang, Changxing Ding, Zhibin Hong, Junhao Huang, Jin Tao, and Xiangmin Xu. Texture-preserving diffusion models for high-fidelity virtual try-on. In _CVPR_ , 2024. 2 

- [53] Zhendong Yang, Ailing Zeng, Chun Yuan, and Yu Li. Effective whole-body pose estimation with two-stages distillation. In _CVPR_ , 2023. 4 

- [54] Jianhao Zeng, Dan Song, Weizhi Nie, Hongshuo Tian, Tongtong Wang, and An-An Liu. Cat-dm: Controllable accelerated virtual try-on with diffusion model. In _CVPR_ , 2024. 2, 7 

- [55] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In _CVPR_ , 2018. 6 

- [56] Xuanpu Zhang, Dan Song, Pengxin Zhan, Qingguo Chen, Kuilong Liu, and Anan Liu. Stableviton: Learning semantic correspondence with latent diffusion model for virtual tryon. In _CVPR_ , 2023. 1, 2, 6, 7 

- [57] Xujie Zhang, Ente Lin, Xiu Li, Yuxuan Luo, Michael Kampffmeyer, Xin Dong, and Xiaodan Liang. Mmtryon: Multi-modal multi-reference control for high-quality fashion generation. _arXiv preprint arXiv:2405.00448_ , 2024. 2 

- [58] Zicheng Zhang, Bonan Li, Xuecheng Nie, Congying Han, Tiande Guo, and Luoqi Liu. Towards consistent video editing with text-to-image diffusion models. _Advances in Neural Information Processing Systems_ , 36:58508–58519, 2023. 5 

- [59] Zhongwei Zhang, Fuchen Long, Yingwei Pan, Zhaofan Qiu, Ting Yao, Yang Cao, and Tao Mei. Trip: Temporal residual learning with image noise prior for image-to-video diffusion models. In _CVPR_ , 2024. 1 

- [60] Jun Zheng, Fuwei Zhao, Youjiang Xu, Xin Dong, and Xiaodan Liang. Viton-dit: Learning in-the-wild video try-on from human dance videos via diffusion transformers. _arXiv preprint arXiv:2405.18326_ , 2024. 3, 6, 7 

- [61] Xiaojing Zhong, Zhonghua Wu, Taizhe Tan, Guosheng Lin, and Qingyao Wu. Mv-ton: Memory-based video virtual tryon network. In _ACM MM_ , 2021. 2 

- [62] Luyang Zhu, Dawei Yang, Tyler Zhu, Fitsum Reda, William Chan, Chitwan Saharia, Mohammad Norouzi, and Ira Kemelmacher-Shlizerman. Tryondiffusion: A tale of two unets. In _CVPR_ , 2023. 1, 2 

10 

