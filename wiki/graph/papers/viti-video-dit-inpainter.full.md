---
type: paper-fulltext
slug: viti-video-dit-inpainter
arxiv_id: null
source_path: /Users/saris.kia.adm/.paper-scholar/viti-video-dit-inpainter/2506.21270.md
paper: "[[viti-video-dit-inpainter]]"
---
<!-- extracted by afk_extract from 2506.21270.pdf (10p) -->

# **Video Virtual Try-on with Conditional Diffusion Transformer Inpainter** 

Cheng Zou[*] , Senlin Cheng _[∗]_ , Bolei Xu, Dandan Zheng, Xiaobo Li, Jingdong Chen, Ming Yang Ant Group 

_{_ wuyou.zc,senlin.csl,shishi.yl,yuandan.zdd,xiaobo.lixb 

jingdongchen.cjd,m.yang _}_ @antgroup.com 

## **Abstract** 

_Video virtual try-on aims to naturally fit a garment to a target person in consecutive video frames. It is a challenging task, on the one hand, the output video should be in good spatial-temporal consistency, on the other hand, the details of the given garment need to be preserved well in all the frames. Naively using image-based try-on methods frame by frame can get poor results due to severe inconsistency. Recent diffusion-based video try-on methods, though very few, happen to coincide with a similar solution: inserting temporal attention into image-based try-on model to adapt it for video try-on task, which have shown improvements but there still exist inconsistency problems. In this paper, we propose ViTI (Video Try-on Inpainter), formulate and implement video virtual try-on as a conditional video inpainting task, which is different from previous methods. In this way, we start with a video generation problem instead of an image-based try-on problem, which from the beginning has a better spatial-temporal consistency. Specifically, at first we build a video inpainting framework based on Diffusion Transformer with full 3D spatial-temporal attention, and then we progressively adapt it for video garment inpainting, with a collection of masking strategies and multistage training. After these steps, the model can inpaint the masked garment area with appropriate garment pixels according to the prompt with good spatial-temporal consistency. Finally, as other try-on methods, garment condition is added to the model to make sure the inpainted garment appearance and details are as expected. Both quantitative and qualitative experimental results show that ViTI is superior to previous works._ 

## **1. Introduction** 

Virtual try-on provides a practical commercial scene in online shopping that the customers can try on some preferred 

- *Equal contributions. 

clothes without touching or physically trying on them before purchasing, which can reduce the probabilities of refunds, exchanges and returns, and in the end improve the shopping experience and stimulate future shoppings. Also, it provides a low cost trail and error solution for designers, which can improve the efficiency of costume design. 

Intuitively, video virtual try-on is much more userfriendly than image virtual try-on, because it can bring more impressive information for the customer, i.e., the figure from different angles, the dynamic movements of the clothes when trying on his/her body. However, previous works mainly focus on image virtual try-on [1, 5, 10, 12, 17, 21, 24, 27, 29, 30, 36, 45], very few studies have attempted video virtual try-on [8, 22, 39], so the current ability is far from realistic applications. 

On the way to powerful video virtual try-on system, there are two main obstacles facing us, one is to precisely preserve the details of the garment, the other is to ensure the spatial-temporal consistency of the generated video. Recent state-of-the-art diffusion-based video virtual try-on methods [8, 39] happen to coincide with a similar solution, that is, training an image-based try-on model at first to acquire the basic try-on ability, and then insert temporal module to improve temporal consistency. In this first-image-thenvideo paradigm, the image part is usually a main U-Net pretrained on a large number of image data, while the video part is just temporal attention layer after the spatial attention in the main U-Net. As is known, the temporal attention layer is designed to conduct self-attention on features of the same spatial position across different frames, which is more like a temporal smoother, though efficient but with poor ability for spatial-temporal modeling. So these methods still suffer from inconsistency problems. The main difference between video and image virtual try-on lies in the demand of spatial-temporal consistency, which is also the fundamental of natural videos. The above mentioned firstimage-then-video methods, can be interpreted as imagebased methods decorated with limited temporal filters or smoothers, naturally have inconsistency problems. 

**==> picture [70 x 94] intentionally omitted <==**

**==> picture [70 x 93] intentionally omitted <==**

**==> picture [70 x 94] intentionally omitted <==**

**==> picture [70 x 93] intentionally omitted <==**

**==> picture [71 x 94] intentionally omitted <==**

**==> picture [71 x 93] intentionally omitted <==**

**==> picture [71 x 94] intentionally omitted <==**

**==> picture [71 x 93] intentionally omitted <==**

**==> picture [70 x 94] intentionally omitted <==**

**==> picture [70 x 93] intentionally omitted <==**

**==> picture [70 x 94] intentionally omitted <==**

**==> picture [70 x 93] intentionally omitted <==**

**==> picture [70 x 94] intentionally omitted <==**

**==> picture [70 x 93] intentionally omitted <==**

**==> picture [70 x 94] intentionally omitted <==**

**==> picture [70 x 93] intentionally omitted <==**

**==> picture [71 x 94] intentionally omitted <==**

**==> picture [71 x 93] intentionally omitted <==**

**==> picture [71 x 94] intentionally omitted <==**

**==> picture [71 x 93] intentionally omitted <==**

**==> picture [70 x 94] intentionally omitted <==**

**==> picture [70 x 93] intentionally omitted <==**

**==> picture [70 x 94] intentionally omitted <==**

**==> picture [70 x 93] intentionally omitted <==**

Figure 1. Results of ViTI (Video Try-on Inpainter). The first two columns provide the input. ViTI performs well on garment detail preservation and spatial-temporal consistency, and also supports try-on for different clothing types, such as tops, pants, and skirts. 

In this paper, we propose ViTI (Video Try-on Inpainter) and formulate video virtual try-on as a conditional video inpainting task, which is different from previous methods. In this way, we start with a video generation problem instead of an image-based try-on problem, which from the beginning will have a better spatial-temporal consistency. Specifically, we first build a video inpainting framework based on Diffusion Transformer [26] with full 3D spatialtemporal attention. To train this video inpainting model, we design a collection of masking strategies, such as timeinvariant box mask, time-variant box mask, instance-level mask, etc., to make the model robust when given a variety of mask shapes. As far as we know, we are the first to implement video inpainting with full 3D attention Diffusion Transformer. We choose full 3D attention because it has a strong spatial-temporal modeling ability for video generation, let alone video inpainting and video virtual try-on. When the video inpainting framework is ready, we progres- 

sively adapt it for video garment inpainting task with multistage training. And to train the model, we further collect a human-centric dataset with 51,278 video clips for video tryon pretraining, whose clothes are segmented and masked with pretrained segmentation model [23], which is the first large scale dataset for video try-on pretraining. After these steps, the model can inpaint the masked garment area with appropriate garment pixels according to the prompt with good spatial-temporal consistency. Finally, garment condition is added to make sure the inpainted garment appearance and details are as expected. During training, a temporal consistency loss for diffusion model is proposed to explicitly constrain the difference between consecutive latent frames to get better temporal consistency. 

## **2. Related Work** 

## **2.1. Image Virtual Try-on** 

Though GAN-based image virtual try-on methods [5, 10, 17, 21, 27, 29, 36] are pioneers, we pay more attention to the recent diffusion-based methods [1, 12, 24, 30, 45]. The key problem in image virtual try-on lies in the garment detail preservation and warping. DCI-VTON [12] proposes an exemplar-based approach that leverages a warping module to guide the diffusion model, which helps to preserve the local details. LaDI-VITON [30] treats garment image features as a set of pseudo-word token embeddings to maintain the texture and details of the garment. StableVITON [24] proposes zero cross-attention blocks, which not only preserve the garment details by learning the semantic correspondence but also generate high-fidelity images by utilizing the inherent knowledge of the pretrained model in the warping process. OOTDiffusion [38] proposes outfitting fusion in the self-attention layers of the denoising UNet to precisely align the garment features with the target human body. TryOnDiffusion [45] employs parallel-UNet to preserve garment details and warp the garment for significant pose and body change in a single network. These methods have made great progress in image try-on to generate highfidelity images, but when applied to video try-on frame by frame there will be poor results due to severe inter-frame inconsistency. 

absence of modern video generation improvements. In this paper, we formulate video virtual try-on as a conditional video inpainting task and implement it with a modern video generation paradigm, which obviously has a better spatialtemporal consistency. 

## **2.3. Video Inpainting** 

Video inpainting is the task of filling masked regions in a video, which is challenging due to its requirements of both spatial and temporal consistencies. AVID [43] is based on a text-to-image inpainting model and finetunes the motion module for temporal consistency, which works in a similar way as AnimateDiff [15]. FGDVI [14] utilizes a decoupled flow completion module to predict the flow from masked frames, and leverages optical flow as a guidance for diffusion model to improve inpainting quality and temporal consistency. In [13], they reframe video inpainting as a conditional generative modeling problem via a 4DUNet and devise a method for conditioning on the known pixels in incomplete frames. Infusion [4] also proposes a diffusion-based method, it adopts an internal learning approach, which leverages only the information contained in the current video. To the best of our knowledge, all the state-of-the-art diffusion-based video inpainting methods so far allow video input by inflating the U-Net with a temporal dimension. In this paper, we implement video inpainting with a full 3D attention diffusion transformer for better spatial-temporal consistency. 

## **2.2. Video Virtual Try-on** 

One of the most important things for video virtual try-on is to ensure spatial-temporal consistency. FW-GAN [7] uses a flow-guided fusion module to warp the past frames to generate coherent video, a warping net to warp clothes, and a parsing constraint loss to alleviate the misalignment problem, which leading to good temporal consistency and visual quality. ClothFormer [22] proposes an appearanceflow tracking module that utilizes ridge regression and optical flow correction to smooth the dense flow sequence to generate a temporally consistent warped garment sequence. MV-TON [44] generates a coarse try-on result at first, and then adopts a memory refinement module to save the previously generated frames for the following frame generation. Tunnel Try-on [39] leverages the Kalman filter to construct smoother crops in the focus tunnel and inject position embedding of the tunnel to improve continuity, also it zooms in on the focus tunnel region to better preserve the fine details of the garment. ViViD [8] proposes a diffusionbased framework and insert temporal modules into the textto-image stable diffusion model for coherent video synthesis. The recent state-of-the-art diffusion-based video virtual try-on methods still suffer from inconsistency problems. We humbly hypothesize that this is mainly due to the limitation of the 1D temporal attention module and the 

## **3. Method** 

We formulate video virtual try-on as a conditional video inpainting task and we propose ViTI (Video Try-on Inpainter), a video virtual try-on framework built upon full 3D attention diffusion transformer, so in Sec. 3.1 we give a brief introduction to diffuion model and diffusion transformer. Then in Sec. 3.2 we introduce the video garment inpainting model, which is the base of ViTI. In Sec. 3.3 we add conditions to the video garment inpainting model to build ViTI. 

## **3.1. Preliminaries** 

**Diffusion model** . Latent diffusion model (LDM) conducts the diffusion process in the latent space instead of pixel space for efficiency. Given an input _x_ 0, a task-specific encoder _E_ is used to map it into the latent space _z_ 0 = _E_ ( _x_ 0). During training, the diffusion process is applied to the latent _z_ 0 in _t_ timesteps to produce a noisy latent _zt_ . And then the denoising diffusion model _ϵθ_ is trained to predict the added noise _ϵ_ with the following loss function, 

**==> picture [202 x 13] intentionally omitted <==**

where _c_ denotes the conditional inputs, such as text prompt. 

**==> picture [397 x 204] intentionally omitted <==**

Figure 2. Overall architecture. The main framework of ViTI (Video Try-on Inpainter) consists of two parts, the video inpainting part, and the try-on condition part. For video inpainting, the mask video through reshaper, the agnostic video through video VAE encoder, are added with the diffused noise in latent space to feed into Diffusion Transformer (DiT) as input. After a _T_ -step denoising, the output tokens of DiT is decoded to pixel space through a video VAE decoder. The inpainting process is only guided by the text prompt at this time. When the lower part, including garment encoder and pose encoder, is added to the framework, the inpainting process can also be guided by the garment image. This figure shows that video virtual try-on is a conditional video inpainting task with the garment image as condition. 

**Diffusion Transformer for Video Generation** . Different 

from U-Net, diffusion transformer (DiT) is based on vision transformer and it takes the token sequences instead of feature maps as input. In video generation, the input video is first compressed by a pretrained video VAE encoder to get latent feature maps, and then the feature maps are patchified and flatten into token sequences. Patchify module is essential in DiT because it is used to convert the spatial input into token sequences. In inference, the token sequence is first unpachified to spatial and then decoded with the video VAE decoder to pixel space. As for architecture, full 3D spatial-temporal attention directly calculates attention on the whole sequence, which has a strong spatial-temporal modeling ability but with a huge computation cost. Another popular architecture design follows Latte [28], which uses stacked spatial attention block and temporal attention block to approximate full attention, which is efficient but with poor spatial-temporal modeling ability. 

## **3.2. Video Garment Inpainting** 

**Overview** . In this section, a video inpainting framework with full 3D attention diffusion transformer is built at first, and then we progressively adapt it for video garment inpainting with the proposed masking strategies and multistage training. 

**Video Inpainting with DiT** . Video inpainting is the task of filling masked regions in a video naturally according to the known regions and text guidance. Formally, given an 

_N_ -frame video _x_ 0 = _{x[i]_ 0 _[}][N] i_ =1[,][a][corresponding] _[N]_[-frame] binary mask _m_ 0 = _{m[i]_ 0 _[}][N] i_ =1[, and text prompt, the training] objective for video inpainting is to minimize the reconstruction loss of the masked area, 

**==> picture [222 x 27] intentionally omitted <==**

where _x[i] p_[is][the][prediction][for][frame] _[i]_[,][the][values][in] _[m][i]_ 0 equal to 1 if the pixels are to be modified, I[ _cond_ ] equals to 1 if _cond_ is true otherwise 0. 

In our implementation with DiT, there are slight differences because: 1) we use diffusion model so the loss is calculated on noises instead of pixel values; 2) we use a video VAE encoder _E_ whose compression rate in three axis are all larger than 1, so the operations are done in latent space with a smaller spatial and temporal size; 3) the original mask _m_ 0 should be transformed to latent space with a reshape operator _R_ to match token shapes. Let _z_ 0 = _E_ ( _x_ 0) = _{z_ 0 _[i][}][T] i_ =1[,] _mz_ = _R_ ( _m_ 0) = _{m[i] z[}][T] i_ =1[,][the][diffusion][transformer][for] video inpainting is trained with the following loss, 

**==> picture [232 x 21] intentionally omitted <==**

where _zt_ is a noisy latent which is diffused from _z_ 0 with _t_ timesteps. 

**Network Architecture** . The proposed DiT based video inpainting framework is illustrated in Figure. 2. It consists 

**==> picture [205 x 159] intentionally omitted <==**

Figure 3. Full 3D attention transformer layer. The garment embeddings interact with the block through an additional cross-attention, which lies in parallel to the text embedding cross-attention, with a scale factor to control the guidance strength. 

of four parts, a text encoder, a mask reshaper, a video VAE and a diffusion transformer. The text encoder is used to extract text embeddings from user input prompt, which acts as the generation guidance for video inpainting, we choose T5 [33] as the text encoder because it offers good prompt following ability in generation tasks. The reshaper projects the original mask from pixel space to latent space, which is mainly used for token shape conversion, and it is implemented by interpolation operator. The video VAE encoder is used to compress video, which can significantly reduce the token numbers, and video VAE decoder recovers the video from latent space to pixel space, we use causal video VAE [26] because it provides a temporal compression, which can further reduce the length of token sequence. 

As the core of the framework, the diffusion transformer is composed of stacked full 3D attention transformer layers. As illustrated in Figure. 3, the full 3D attention block consists of pre-normalization, self-attention, cross-attention, residual connection, post-normalization and FFN. For video inpainting, only the cross-attention with text embedding is essential. The input hidden states comes from the patchified and flattened token sequence, and it is feed into full 3D spatial-temporal attention to update itself. 

It is noteworthy that the input noise to DiT is the addition of three parts: reshaped mask _mz_ , diffused latent _zt_ , and masked video latent _E_ ( _x_ 0 �(1 _− m_ 0)), where � is pixelwise multiplication. 

**Training strategy** . To train the video garment inpainting model, we carefully design a collection of masking strategies, and then we propose multi-stage training to progressively adapt a pretrained text-to-video model to a video garment inpainting model. The core reason for us to design such multi-stage training with different masking strategies is the lack of high quality data, because we have too little 

**==> picture [151 x 132] intentionally omitted <==**

Figure 4. Masking strategies for garment inpainting model. These are time-invariant box mask, time-variant box mask, instance-level mask and garment mask. 

videos with precise garment masks to train the garment inpainting model all at once. 

As illustrated in Figure. 4, we design four kind of masking strategies, time-invariant box mask, time-variant box mask, instance-level mask and garment mask. In timeinvariant box mask, the position and size of the mask is random, but all the frames share the same mask, while in time-variant box mask, the position and size of the mask can be different between frames. These two kind of masks are general and can be applied to any videos, so they are used in the early training stage with a large video dataset [6, 40]. And during training, there is a probability to inverse these masks to improve the diversity. As for instance-level mask, the mask in each frame is a segmentation mask of the content of interest, and similar in garment mask, the mask in each frame is a segmentation mask of the garment. These two kind of masks are used in the later training stage, because only few videos have precise instance-level mask and garment mask due to the high acquisition cost of such data. 

Then we train the video garment inpainting model in multiple stages with different dataset. Specifically, in the first stage (Stage 1), the model is trained on a large video dataset [6, 40], with time-invariant box mask and timevariant box mask strategies. The text prompt for each video is the original caption from the video dataset, though not semantically aligned with the mask area, still works in practice. After this stage, the model has a basic video inpainting ability given coarse masks. In the second stage (Stage 2), the model is trained on VOS (Video Object Segmentation) dataset [2, 20, 32], with instance-level mask. The text prompt for each video comes from the ‘motion expression’ (indeed also image captions) of the target object, which has a good alignment with the mask area. In the third stage (Stage 3), the model is trained on our collected humancentric dataset VTP, with garment mask, and the text prompt is associated with the garment mask area. The collection of the dataset VTP is detailed in experiments part. 

**==> picture [231 x 78] intentionally omitted <==**

Figure 5. Visualization of an example of the proposed dataset VTP for video try-on pretraining. From left to right, these are garment image, original video, garment-agnostic video, densepose video and garment mask video. It is noteworthy that there is not a standard garment image for each Internet-sourced video, so the shown garment image (leftmost) is cropped from one of the original video frames via its corresponding garment mask. 

After these training stages, the model can inpaint the masked garment area with garment pixels according to the text prompt with good spatial-temporal consistency. In the next section, we introduce conditions to this model to fullfill the task of video virtual try-on, which can control the appearance and details of the garment by reference image. 

cross-attention for garment embedding lies in parallel to the text embedding cross-attention, with a scale factor to control the guidance strength. 

**Temporal consistency loss** . Although full 3D attention diffusion transformer has a strong spatial-temporal modeling ability, there still exist flickers in the generated videos because of the limited high quality training data. Thus we seek to explicitly improve the temporal consistency. We use the following loss to constrain the difference between consecutive latent frames, 

**==> picture [222 x 30] intentionally omitted <==**

where _ϵθ_ ( _∗_ ) _[i]_ is the sub-latent in the _i_ -th frame. So the total loss for training ViTI is as follows, 

**==> picture [175 x 11] intentionally omitted <==**

where _α_ = 0 _._ 1 is a hyperparameter. Initialized with the weights of the pretrained video garment inpainting model, ViTI is optimized with _Ltotal_ on dataset Vivid [8]. 

## **3.3. Video Try-on Inpainter (ViTI)** 

**Overview** . Based on the above video inpainting framework, conditions are added in this section to make video try-on a conditional video inpainting task. As illustrated in the lower part of Figure. 2, a garment encoder on the right is added to extract garment embedding as condition to the DiT block through cross-attention. A pose encoder on the left is added to extract pose latent as content prior. Also, a loss to constrain the difference between consecutive frames is explicitly applied to further improve the temporal consistency. **Garment encoder** . One of the key problems for virtual try-on is garment detail preservation. We design garment encoder to extract the visual feature of the reference garment image. It mainly consists of two parts, an image VAE encoder [46] and a DINOv2 [31]. The image VAE encoded feature and the DINOv2 extracted feature, are first transformed by linear layers, and then concatenated together to send into an MLP layer to produce the output garment embedding. DINOv2 is used because its visual features are robust and perform well across domains, which meets the need of extracting robust feature for the garment image. 

**Pose encoder** . Pose encoder is composed of a DensePose [16] encoder followed by an MLP layer, the output of which has the same tensor shape with latent noise, and it is fused with other latents as input to the DiT model through addition in latent space. DensePose is used because we hypothesize that the depth information in it can provide consistent spatial content prior. 

**Garment adapter** . The garment embedding is injected into full 3D attention transformer layer by cross-attention, a similar way as in IPAdapter [41]. As illustrated in Figure. 3, the 

## **4. Experiments** 

In this section, we first give an introduction to the dataset, then we compare the proposed ViTI with other state-of-theart methods both qualitatively and quantitatively, and finally we conduct ablation studies to illustrate some of the key design choices behind ViTI. 

## **4.1. Implementation Details** 

**Dataset preparation** . For video virtual try-on task, there are only two publicly available dataset, VVT [22] and Vivid [8]. VVT is widely used for benchmarking, it includes 791 videos with corresponding garment images at a resolution of 256 × 192. Vivid comprises 9,700 clothingvideo pairs with a resolution of 832 × 624, categorized into three clothing types, upper-body, lower-body, and dresses. These datasets with high quality are more suitable for finetuning, which are usually used in the last training stage. However, as we know, data plays a key role in two aspects, the quality and the amount. Vivid provides a small amount of high quality dataset, while no large amount dataset for video try-on pretraining. So we construct a human-centric video dataset for Video Try-on Pretraining (VTP) with a variety of clothing styles, body movements and backgrounds, which is specifically designed for early stage pretraining. Using a collection of tools, such as human detection, parsing, blur evaluation and aesthetic evaluation, we obtain a total of 51,278 video clips in the end, each of which contains a single human with enough upper body or lower body ratio, Figure. 5 shows an example. For more information about VTP please refer to the supplementary materials. 

|**Method**|**SSIM**_↑_|**LPIPS**_↓_|**VFID(I3D)**_↓_|
|---|---|---|---|
|FW<br>GAN [7]|0.675|0.283|8.019|
|CP-VTON [35]|0.459|0.535|6.361|
|PBAFN [11]|0.870|0.157|4.516|
|WildVidFit [19]|-|-|4.202|
|OOTDiffusion [38]|0.882|0.070|4.167|
|StableVITON [24]|0.876|0.076|4.021|
|ClothFormer [22]|0.921|0.081|3.967|
|VIVID [8]|**0.949**|0.068|3.405|
|Tunnel Try-on [39]|0.913|0.054|3.345|
|ViTI (Ours)|0.938|**0.042**|**2.121**|



Table 1. Quantitative comparisons on VVT dataset. The best results are highlighted in bold. As can be seen, ViTI achieves significant improvements on VFID. Result of WildVidFit [19] is partially reported due to the lack of open-sourced model. A whole table with VFID (Resnext3d) please refer to the supplementaries. 

**Experimental Settings** . The model is trained across 8 Nvidia A100 GPUs using the AdamW optimizer with a learning rate of 1e-5. The resolution of the video clip for training is set to 29 x 512 × 384. 

## **4.2. Qualitative Results** 

We conduct a qualitative comparison with several recent state-of-the-art image-based try-on methods (OOTDiffusion [38] and StableVTON [25]) and diffusion-based video try-on method (Vivid [8]). Figure. 6 shows the visual comparison result, as can be seen, our method ViTI obviously outperforms other methods. The image-based try-on methods, on the one hand, struggle with handling side-view scenes effectively, and the detail of the garment is also not satisfactory. On the other hand, due to the lack of temporal information, they fail to achieve temporal consistency. The diffusion-based state-of-the-art method Vivid, which incorporates a temporal module, shows notable improvement in temporal consistency, but still lacks of sufficient preservation of garment details. Figure.1 showcases a range of results generated by ViTI, encompassing various scenes and garment types. Our method consistently achieves high-detail preservation and maintains temporal consistency across the generated try-on videos. For more visualization results, please refer to the supplementaries. 

## **4.3. Quantitative Results** 

**Evaluation Metrics** . For video try-on evaluation, Video Frechet Inception Distance (VFID) [34] is used to measure both visual quality and temporal consistency of the generated results. There are two kind of 3D convolution feature extractor used for VFID, one is I3D [3] and the other is 3D-ResNeXt101 [18]. For the single-frame evaluation, we rely on two primary metrics: Structural Similarity Index 

**==> picture [199 x 346] intentionally omitted <==**

Figure 6. Qualitative comparisons with other methods on the public dataset. ViTI achieves high-detail preservation and good temporal consistency, and performs better than any other methods. 

(SSIM) [37] and Learned Perceptual Image Patch Similarity (LPIPS) [42]. These metrics are used to assess the quality of individual frames in a paired comparison setting. A higher SSIM value or a lower LPIPS score indicates better visual similarity between the generated frame and the original one. Although SSIM and LPIPS are not sufficient to evaluate the quality of videos, we still use them for video frame quality evaluation as a supplement. 

We perform a quantitative evaluation on the dataset VVT [22] with existing visual try-on methods, including GAN based methods like FW-GAN [7], PBAFN [9] and Cloth Former [22], and diffusion-based methods like Vivid [8]. Some image-based try-on algorithms are also evaluated, including CP-VTON [36] and StableVITON [25]. As shown in Table. 1, the proposed method ViTI achieves state-of-the-art performance on most metrics, especially on VFID with a significant improvement compared to previous methods. 

|**Method**|**SSIM**_↑_|**LPIPS**_↓_|**VFID(I3D)**_↓_|
|---|---|---|---|
|VAE|0.889|0.072|3.421|
|VAE+CLIP|0.921|0.059|2.356|
|VAE+DINOv2|**0.938**|**0.042**|**2.121**|



Table 2. Ablation study on the garment encoder. 

|**PoseE**|**TCLoss**|**SSIM**_↑_|**LPIPS**_↓_|**VFID(I3D)**_↓_|
|---|---|---|---|---|
|||0.911|0.062|2.446|
|✓||0.929|0.049|2.223|
||✓|0.920|0.054|2.359|
|✓|✓|**0.938**|**0.042**|**2.121**|



Table 3. Ablation study on the pose encoder (PoseE) and the temporal consistency loss (TCLoss). 

|**Method**|**SSIM**_↑_|**LPIPS**_↓_|**VFID(I3D)**_↓_|
|---|---|---|---|
|ViTI (w/o VTP)|0.922|0.055|2.935|
|ViTI|**0.938**|**0.042**|**2.121**|



Table 4. Ablation study on the VTP dataset. It’s noteworthy that even without VTP dataset, ViTI achieves best VFID(I3D) compared to other methods as shown in Table 1. 

## **4.4. Ablation Study** 

In this section, we conduct ablation experiments to illustrate some key design choices behind ViTI. 

**Garment encoder** . Detail preservation is important to tryon task, so image encoders those can capture detail information are preferred. VAE is trained to compress the image and then recover it from the compressed latent, so VAE is a first choice because its latent contains much detail information used for reconstruction. However using image VAE encoder alone yields poor results, mainly due to the lack of semantic information. So we introduce CLIP vision encoder for high-level features and combine the VAE encoder and CLIP vision encoder together, which achieves better result. To put it further, we replace CLIP vision encoder with DINOv2, a more robust vision encoder, which focuses on self-supervised pre-training with extensive collections of clustered images, can produce features with strong semantic as well as much details. Table. 2 reports the experiment results of different garment encoder choices. 

**Pose encoder** . Densepose information is considered to be a valuable conditional input for capturing depth information, which is often critical for non-rigid objects, so DensePose is introduced as pose encoder to deal with those non-rigid objects (garment and human body) in try-on task for better spatial-temporal consistency, in the hypothesis that depth information is continuous in the spatial-temporal space. Ta- 

|**Method**|**Stage 1**|**Stage 2**|**Stage 3**|
|---|---|---|---|
|Imaging Quality_↑_|0.3940|0.4271|**0.4303**|
|Subject Consistency_↑_|0.4111|0.4129|**0.4171**|
|Temporal Flickering_↑_|0.9801|0.9814|**0.9821**|
|Motion Smoothness_↑_|0.9930|**0.9937**|0.9936|
|Overall Consistency_↑_|0.1387|0.1659|**0.2096**|



Table 5. Quantitative analysis of different stages. Because there is no benchmark to evaluate the garment inpainting task, so we use metrics from VBench to evaluate the video quality. 

ble. 3 reports the experiment results of using or not using pose encoder. It can be seen that using pose encoder shows better performance. 

**Temporal consistency loss** . Similarly, we evaluate the effectiveness of the temporal consistency loss by using it or not using it during training. Table. 3 reports the experiment results, it can be seen that there is a performance improvement by using temporal consistency loss. So we can explicitly constrain the difference between consecutive latent frames to get better temporal consistency, especially when the amount of high quality training data is small. **Multi-stage training** . As mentioned in Sec. 3.2, the core reason to design multi-stage training with different masking strategies is the lack of high quality data, because we have too little videos with precise garment masks to train the garment inpainting model all at once. So we have to progressively make it. It’s noteworthy that after Stage 1/2/3 training, we only get a video garment inpainting model, which can not be evaluated on VVT, and also there is no specific benchmark to evaluate this kind of task, so we use metrics from VBench to evaluate the video quality as a reference. Table. 5 reports the quantitative results after different stage training. As can be seen, the performance improves steady via progressive training. Typical visualization results of different stages please refer to supplementary materials. 

**Video try-on pretraining (VTP) dataset** . With only public video try-on dataset (such as Vivid) we find that although the model has the ability to try-on clothes in most cases, it shows poor robustness when doing out-of-distribution test, and it often has unexpected flickering, implying that the model is not sufficiently trained. So we collect VTP, a dataset for video try-on pretraining, which is larger than the previous video try-on dataset. With VTP, ViTI shows not only better but also robust performance. Table. 4 shows the effectiveness of the VTP dataset. 

## **5. Conclusions** 

We propose ViTI and formulate video try-on as a conditional video inpainting task. Specifically, we first build a video inpainting framework based on diffsion transformer with full 3D spatial-temporal attention. And then we pro- 

gressively adapt it for video garment inpainting with a collection of masking strategies and multi-stage training. And to train the model, we collect a human-centric dataset with 51,278 video clips for video try-on pretraining. Finally, garment condition is added to make sure the inpainted garment appearance and details are as expected. ViTI utilizes a completely different framework to deal with video try-on task compared to previous methods, which shows significant improvements. We hope our method can bring new insight to the community. 

## **References** 

- [1] Alberto Baldrati, Davide Morelli, Giuseppe Cartella, Marcella Cornia, Marco Bertini, and Rita Cucchiara. Multimodal garment designer: Human-centric latent diffusion models for fashion image editing. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 23393– 23402, 2023. 1, 3 

- [2] Sergi Caelles, Jordi Pont-Tuset, Federico Perazzi, Alberto Montes, Kevis-Kokitsi Maninis, and Luc Van Gool. The 2019 davis challenge on vos: Unsupervised multi-object segmentation. _arXiv:1905.00737_ , 2019. 5 

- [3] Joao Carreira and Andrew Zisserman. Quo vadis, action recognition? a new model and the kinetics dataset, 2018. 7 

- [4] Nicolas Cherel, Andr´es Almansa, Yann Gousseau, and Alasdair Newson. Infusion: Internal diffusion for video inpainting. _arXiv preprint arXiv:2311.01090_ , 2023. 3 

- [5] Seunghwan Choi, Sunghyun Park, Minsoo Lee, and Jaegul Choo. Viton-hd: High-resolution virtual try-on via misalignment-aware normalization. In _Proc. of the IEEE conference on computer vision and pattern recognition (CVPR)_ , 2021. 1, 3 

- [6] Henghui Ding, Chang Liu, Shuting He, Xudong Jiang, and Chen Change Loy. Mevis: A large-scale benchmark for video segmentation with motion expressions, 2023. 5 

- [7] Haoye Dong, Xiaodan Liang, Xiaohui Shen, Bowen Wu, Bing-Cheng Chen, and Jian Yin. Fw-gan: Flow-navigated warping gan for video virtual try-on. In _Proceedings of the IEEE/CVF international conference on computer vision_ , pages 1161–1170, 2019. 3, 7 

- [8] Zixun Fang, Wei Zhai, Aimin Su, Hongliang Song, Kai Zhu, Mao Wang, Yu Chen, Zhiheng Liu, Yang Cao, and ZhengJun Zha. Vivid: Video virtual try-on using diffusion models. 2024. 1, 3, 6, 7 

- [9] Yuying Ge, Yibing Song, Ruimao Zhang, Chongjian Ge, Wei Liu, and Ping Luo. Parser-free virtual try-on via distilling appearance flows. In _In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , page 8485–8493, 2021. 7 

- [10] Yuying Ge, Yibing Song, Ruimao Zhang, Chongjian Ge, Wei Liu, and Ping Luo. Parser-free virtual try-on via distilling appearance flows. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 8485–8493, 2021. 1, 3 

- [11] Yuying Ge, Yibing Song, Ruimao Zhang, Chongjian Ge, Wei Liu, and Ping Luo. Parser-free virtual try-on via distilling appearance flows, 2021. 7 

- [12] Junhong Gou, Siyu Sun, Jianfu Zhang, Jianlou Si, Chen Qian, and Liqing Zhang. Taming the power of diffusion models for high-quality virtual try-on with appearance flow. _arXiv preprint arXiv:2308.06101_ , 2023. 1, 3 

- [13] Dylan Green, William Harvey, Saeid Naderiparizi, Matthew Niedoba, Yunpeng Liu, Xiaoxuan Liang, Jonathan Lavington, Ke Zhang, Vasileios Lioutas, Setareh Dabiri, et al. Semantically consistent video inpainting with conditional diffusion models. _arXiv preprint arXiv:2405.00251_ , 2024. 3 

- [14] Bohai Gu, Yongsheng Yu, Heng Fan, and Libo Zhang. Flow-guided diffusion for video inpainting. _arXiv preprint arXiv:2311.15368_ , 2023. 3 

- [15] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized text-toimage diffusion models without specific tuning. _International Conference on Learning Representations_ , 2024. 3 

- [16] Rıza Alp G¨uler, Natalia Neverova, and Iasonas Kokkinos. Densepose: Dense human pose estimation in the wild, 2018. 6 

- [17] Xintong Han, Zuxuan Wu, Zhe Wu, Ruichi Yu, and Larry S Davis. Viton: An image-based virtual try-on network. In _CVPR_ , 2018. 1, 3 

- [18] Kensho Hara, Hirokatsu Kataoka, and Yutaka Satoh. Can spatiotemporal 3d cnns retrace the history of 2d cnns and imagenet?, 2018. 7 

- [19] Zijian He, Peixin Chen, Guangrun Wang, Guanbin Li, Philip HS Torr, and Liang Lin. Wildvidfit: Video virtual tryon in the wild via image-based controlled diffusion models. _arXiv preprint arXiv:2407.10625_ , 2024. 7 

- [20] Lingyi Hong, Zhongying Liu, Wenchao Chen, Chenzhi Tan, Yuang Feng, Xinyu Zhou, Pinxue Guo, Jinglun Li, Zhaoyu Chen, Shuyong Gao, Wei Zhang, and Wenqiang Zhang. Lvos: A benchmark for large-scale long-term video object segmentation, 2024. 5 

- [21] Thibaut Issenhuth, J´er´emie Mary, and Cl´ement Calauzenes. Do not mask what you do not need to mask: a parser-free virtual try-on. In _Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XX 16_ , pages 619–635. Springer, 2020. 1, 3 

- [22] Jianbin Jiang, Tan Wang, He Yan, and Junhui Liu. Clothformer: Taming video virtual try-on in all module. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2022. 1, 3, 6, 7 

- [23] Rawal Khirodkar, Timur Bagautdinov, Julieta Martinez, Su Zhaoen, Austin James, Peter Selednik, Stuart Anderson, and Shunsuke Saito. Sapiens: Foundation for human vision models. _arXiv preprint arXiv:2408.12569_ , 2024. 2 

- [24] Jeongho Kim, Gyojung Gu, Minho Park, Sunghyun Park, and Jaegul Choo. Stableviton: Learning semantic correspondence with latent diffusion model for virtual try-on. _arXiv preprint arXiv:2312.01725_ , 2023. 1, 3, 7 

- [25] Jeongho Kim, Guojung Gu, Minho Park, Sunghyun Park, and Jaegul Choo. Stableviton: Learning semantic correspon- 

   - dence with latent diffusion model for virtual try-on. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 8176–8185, 2024. 7 

- [26] PKU-Yuan Lab and Tuzhan AI etc. Open-sora-plan, 2024. 2, 5 

- [27] Sangyun Lee, Gyojung Gu, Sunghyun Park, Seunghwan Choi, and Jaegul Choo. High-resolution virtual try-on with misalignment and occlusion-handled conditions. In _European Conference on Computer Vision_ , pages 204–219. Springer, 2022. 1, 3 

- [28] Xin Ma, Yaohui Wang, Gengyun Jia, Xinyuan Chen, Ziwei Liu, Yuan-Fang Li, Cunjian Chen, and Yu Qiao. Latte: Latent diffusion transformer for video generation. 2024. 4 

- [29] Davide Morelli, Matteo Fincato, Marcella Cornia, Federico Landi, Fabio Cesari, and Rita Cucchiara. Dress code: highresolution multi-category virtual try-on. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 2231–2235, 2022. 1, 3 

- [30] Davide Morelli, Alberto Baldrati, Giuseppe Cartella, Marcella Cornia, Marco Bertini, and Rita Cucchiara. LaDIVTON: Latent Diffusion Textual-Inversion Enhanced Virtual Try-On. In _Proceedings of the ACM International Conference on Multimedia_ , 2023. 1, 3 

- [31] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Herv´e Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. Dinov2: Learning robust visual features without supervision, 2024. 6 

- [32] Jordi Pont-Tuset, Federico Perazzi, Sergi Caelles, Pablo Arbel´aez, Alex Sorkine-Hornung, and Luc Van Gool. The 2017 davis challenge on video object segmentation, 2018. 5 

lable virtual try-on. _arXiv preprint arXiv:2403.01779_ , 2024. 3, 7 

   - [39] Zhengze Xu, Mengting Chen, Zhao Wang, Linyu Xing, Zhonghua Zhai, Nong Sang, Jinsong Lan, Shuai Xiao, and Changxin Gao. Tunnel try-on: Excavating spatial-temporal tunnels for high-quality virtual try-on in videos. _arXiv preprint_ , 2024. 1, 3, 7 

   - [40] Linjie Yang, Yuchen Fan, and Ning Xu. Video instance segmentation, 2019. 5 

   - [41] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ipadapter: Text compatible image prompt adapter for text-toimage diffusion models. In _arXiv preprint arxiv:2308.06721_ , 2023. 6 

   - [42] Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric, 2018. 7 

   - [43] Zhixing Zhang, Bichen Wu, Xiaoyan Wang, Yaqiao Luo, Luxin Zhang, Yinan Zhao, Peter Vajda, Dimitris Metaxas, and Licheng Yu. Avid: Any-length video inpainting with diffusion model. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 7162– 7172, 2024. 3 

   - [44] Xiaojing Zhong, Zhonghua Wu, Taizhe Tan, Guosheng Lin, and Qingyao Wu. Mv-ton: Memory-based video virtual tryon network. In _Proceedings of the 29th ACM International Conference on Multimedia_ , pages 908–916, 2021. 3 

   - [45] Luyang Zhu, Dawei Yang, Tyler Zhu, Fitsum Reda, William Chan, Chitwan Saharia, Mohammad Norouzi, and Ira Kemelmacher-Shlizerman. Tryondiffusion: A tale of two unets. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 4606–4615, 2023. 1, 3 

   - [46] Zixin Zhu, Xuelu Feng, Dongdong Chen, Jianmin Bao, Le Wang, Yinpeng Chen, Lu Yuan, and Gang Hua. Designing a better asymmetric vqgan for stablediffusion, 2023. 6 

- [33] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer, 2023. 5 

- [34] Thomas Unterthiner, Sjoerd van Steenkiste, Karol Kurach, Raphael Marinier, Marcin Michalski, and Sylvain Gelly. Towards accurate generative models of video: A new metric & challenges. _arXiv preprint arXiv:1812.01717_ , 2018. 7 

- [35] Bochao Wang, Huabin Zheng, Xiaodan Liang, Yimin Chen, Liang Lin, and Meng Yang. Toward characteristicpreserving image-based virtual try-on network, 2018. 7 

- [36] Bochao Wang, Huabin Zheng, Xiaodan Liang, Yimin Chen, Liang Lin, and Meng Yang. Toward characteristicpreserving image-based virtual try-on network. In _Proceedings of the European conference on computer vision (ECCV)_ , pages 589–604, 2018. 1, 3, 7 

- [37] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. _IEEE transactions on image processing_ , 13(4):600–612, 2004. 7 

- [38] Yuhao Xu, Tao Gu, Weifeng Chen, and Chengcai Chen. Ootdiffusion: Outfitting fusion based latent diffusion for control- 

