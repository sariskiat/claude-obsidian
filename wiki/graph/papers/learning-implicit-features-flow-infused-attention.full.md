---
type: paper-fulltext
slug: learning-implicit-features-flow-infused-attention
arxiv_id: null
source_path: /Users/saris.kia.adm/.paper-scholar/learning-implicit-features-flow-infused-attention/2412.11435.md
paper: "[[learning-implicit-features-flow-infused-attention]]"
---
<!-- extracted by afk_extract from 2412.11435.pdf (10p) -->

# **Learning Implicit Features with Flow Infused Attention for Realistic Virtual Try-On** 

Delong Zhang[1] _[∗†]_ Qiwei Huang[2] _[†]_ Yuanliu Liu[2] Yang Sun[2] Wei-Shi Zheng[1] Pengfei Xiong[2] _[‡]_ Wei Zhang[2] 1Sun Yat-sen University 2Shopee 

**==> picture [481 x 290] intentionally omitted <==**

**----- Start of picture text -----**<br>
Target Person Garment Warped Garment GP-VTON StableVITON D4-VTON Ours<br>**----- End of picture text -----**<br>


Figure 1. Warp-based method (e.g. GP-VTON) is prone to severe visual artifacts and distortions (see the green dashed box) affected by the warp module. Learning-based methods (e.g. StableVTON and _D_[4] -VTON) are difficult to reconstruct complex detailed textures (see the red dashed box). Our FIA-VTON designs a Flow Infused Attention module, utilizing the warping flow as an implicit guide to reconstruct complex detailed textures while maintaining the consistency of the garment. 

## **Abstract** 

_Image-based virtual try-on is challenging since the generated image should fit the garment to model images in various poses and keep the characteristics and details of the garment simultaneously. A popular research stream warps the garment image firstly to reduce the burden of the generation stage, which relies highly on the performance of the warping module. Other methods without explicit warping often lack sufficient guidance to fit the garment to the model images. In this paper, we propose FIA-VTON, which leverages the implicit warp feature by adopting a Flow Infused_ 

_Attention module on virtual try-on. The dense warp flow map is projected as indirect guidance attention to enhance the feature map warping in the generation process implicitly, which is less sensitive to the warping estimation accuracy than an explicit warp of the garment image. To further enhance implicit warp guidance, we incorporate high-level spatial attention to complement the dense warp. Experimental results on the VTON-HD and DressCode dataset significantly outperform state-of-the-art methods, demonstrating that FIA-VTON is effective and robust for virtual try-on._ 

> _∗ Work done during internship at Shopee Inc._ 

> _† Equal Contribution._ 

> _‡ Corresponding Author._ 

## **1. Introduction** 

Virtual try-on technology generates images or videos of fitting a garment image on a model image, which is widely used in online shopping, special effects entertainment, AR, and other scenarios. Virtual try-on is challenging since the garment image needs to be warped properly to fit the pose of the model, keeping the garment consistent before and after try-on. At the same time, the generated image should be clear enough to maintain the garment details[31]. 

Along with the rapid progress of image generation techniques, the backbone of virtual try-on has evolved from GAN-based methods to diffusion models. However, how to warp the garment properly remains under investigation. Previous methods can be roughly categorized into two research streams. Warping-based methods warp the garment explicitly to fit the model’s skeleton and then take the warped garment as input for the following generation process [12, 26, 28]. In particular, many methods in the stream put the warped garment onto the cloth-agnostic model image, so the generation process becomes a typical inpainting problem [10, 29]. However, these methods heavily rely on the performance of the warping module, which is highly challenging itself, as can be seen in the green dashed box of Figure 1. On the other hand, Learning-based methods take the garment image as a control condition[6, 17, 31]. Both the warping and inpainting effects are produced simultaneously by a single-generation model. While these methods do not rely on a separate warping module, they lack direct guidance to the warping process. The try-on results will be degraded for model images and lack of complex detailed textures, which is depicted in the red dashed box of Figure 1. 

To overcome these limitations, we propose to leverage the implicit warp feature to guide the generation process, which adopts a Flow Infused Attention module in the diffusion model (FIA-VTON). Firstly, a dense flow map is predicted by a warp network from the garment image and the model pose. Then the dense flow is infused into the diffusion model by cross-attention. Unlike previous models which warp the garment explicitly by the flow map and input the warped garment to the generation model, flow attention takes the flow map itself to guide the implicit warp. In this way, the flow guidance is straight and clear. At the same time, the flow guidance is not as rigid as a warped garment, so the generation process can correct inaccurate flow estimations according to other sources of guidance such as the model pose. 

While the dense optical flow captures the deformation pattern for try-on, the static garment details are captured by the local garment feature from a Garment Net, following Tryondiffusion[31]. To further capture high-level characteristics of the garment, we incorporate extra spatial attention into the flow attention. All the guidance, including the 

dense flow map, the local garment feature, and the highlevel spatial feature are infused into the denoising UNet of the diffusion model by decoupled cross attention. As a result, the try-on image exhibits accurate deformations while preserving both high-level characteristics and fine details. 

We conduct extensive experiments on the VITONHD [4] and Dress-Code [22] dataset. The results demonstrate that FIA-VTON achieves state-of-the-art performance consistently. Qualitative results show that our model fitted the garment accurately to the model pose and maintained fine details as well. 

The main contributions of this paper are as follows: 

- We propose to leverage Flow Infused Attention to guide the diffusion model, which acts as a straight and clear guide to the warping process of try-on. 

- We infuse both the dense flow map, the local garment feature, and the high-level spatial feature by decoupled cross-attention, which captures the dynamic warp pattern, the garment details, and the high-level characteristics in a uniform way. 

- We achieve state-of-the-art performance on widely used benchmarks, which validates the advantage of flow guidance over explicit guidance of warped garments and implicit warp without flow estimation. 

## **2. Related Work** 

**Learning-based Virtual Try-on.** Recently, diffusion models [7, 11, 15, 16] start to dominate in natural image generation due to its superior ability in generating high-fidelity realistic images compared to GAN-based models. Inspired by this, a series of learning-based virtual try-on models, which leverage diffusion framework to learn garment fitting and transformation, began to emerge. TryOnDiffusion [31] unifies two UNets to preserve garment details and warp the garment in a single network. LaDI-VTON [23] introduces the textual inversion component that maps visual features of reference garment to CLIP token embedding space as a condition of diffusion model. DCI-VTON [10] further use warping network to warp reference garment, which is fed into diffusion model as additional guidance. StableVITON [17] discarded independent warping and proposed a zero cross-attention block to learn semantic correlation between the clothes and human body. Although promising results are attained, these learning-based VTON approaches still fail to completely retain every detail of the reference garment. In contrast, our method introduces the Flow-infused Layer, which combines with implicit garment features to provide explicit guidance for fine-grained texture reconstruction. 

**Wrap Guidance.** Most virtual try-on methods have a prewarping stage, which warps flat garments to align with the corresponding positions on the human body. DCI-VTON [10] and _D_[4] -VTON [29] paste the warped garments onto 

**==> picture [491 x 204] intentionally omitted <==**

**----- Start of picture text -----**<br>
Garment Net<br>Flow Spatial<br>VAE …… Projector Projector<br>Encoder<br>Flow<br>Guider Garment Model<br>Spatial<br>Guider<br>𝐹𝑉 𝐹𝐾 𝐹𝑄<br>Denoise UNet Flow Attention<br>VAEEncoder …… VAEDecoder 𝑆𝑄 𝑆𝐾 𝑆𝑉<br>Spatial Attention<br>(a) FIA-VTON (b) Flow Infused Attention<br>Self Attn Cross Attn Self Attn Cross Attn Self Attn Cross Attn Self Attn Cross Attn<br>E FIA FIA FIA FIA E<br>Self Attn Self Attn Self Attn Self Attn<br>**----- End of picture text -----**<br>


Figure 2. (a) Overview of FIA-VTON, illustrating the main components: a pair of VAE Encoder and Decoder, a Garment Net, and a Denoising UNet. The model takes a garment image, a target person image, and mask and pose based control images as inputs. We adopt a Flow Guider to generate dense flow map, and a Spatial Guider to extract global garment characteristics. These features are processed in a Flow Infused Attention module, interactions with Garment Net and Denoising UNet to generate the final try-on output. (b) Illustration of the Flow Infused Attention module. The flow is projected into garment and model feature space, and then fused using a cross-attention mechanism to produce flow-guided features. Then the high-level spatial feature is further integrated by decoupled cross-attention, which captures the garment consistency, details, and high-level characteristics uniformly. 

the cloth-agnostic model image, adds noise, and then inputs into a denoising diffusion model. GP-VTON [28] concatenates the warped garment with cloth agnostic model image and other guide images spatially, and inputs them into a UNet generator. GarDiff[26] concatenates the warped garment with the noised image and feeds them into the self-attention layer of a denoising diffusion model. In contrast, our model injects the dense optical flow instead of the warped garment into the diffusion model through cross attention, which is softer and thus less sensitive to the warp estimation precision. GarDiff also combines the Embeddings of warped Garment and raw garment image. ALDM[8] align the features of the raw garment and the warped garment by an Adaptive Alignment Module, and then input the aligned feature into the diffusion model by cross attention. These combined features cannot reflect the warping pattern directly. In comparison, our model uses the flow map instead, which can give clear and direct guidance to warping. 

**Flow Estimation.** Early try-on methods [19, 27] employ Thin Plate Spline (TPS) to warp flat garments to align with the corresponding positions on the human body. To better handle complex deformations, appearance flow-based methods [9, 13, 14, 29] have been introduced to predict appearance flow to warp the flat garment to achieve pixel-level alignment with the target person’s body. _D_[4] -VTON [29] proposed a Dynamic Semantics Disentangling Module to predict flows separately for different semantic layers. Our paper follows _D_[4] -VTON to extract dense warp flows. 

## **3. Our method** 

In this work, we propose a Flow Infused Attention module for diffusion-based virtual try-on (FIA-VTON). Figure 2 demonstrates an overview of the FIA-VTON model. FIAVTON is built on an inpainting version of Stable Diffusion (SD) [25]. SD first encodes the noised image and the masks by a VAE encoder, then generates a latent image by a Denoising Unet, and decodes into a real image by a VAE decoder. In FIA-VTON, all the cross-attention modules of the Denoising UNet are replaced by the Flow Infused Attention (FIA) modules. A FIA module fuses the dense warp flow, the local garment feature, and the coarse spatial feature into the model feature from the upstream UNet layers. Here the local garment feature is extracted by a Garment Net, and we adopt the typical structure of a VAE Encoder and a UNet. The dense flow is estimated by a Flow Guider as described in Section 3.2. The coarse spatial feature is provided by a Spatial Guider, detailed in Section 3.3. 

## **3.1. FIA module** 

Previous learning-based VTON approaches struggle to preserve the details of garments, such as high-frequency textures in complex patterns. In addition, warp-based methods are prone to serve visual artifacts and distortions affected by the garment warp module. To solve these problems, we leverage the appearance flow and spatial characteristics, combining with the local garment feature to implicitly guide the diffusion model, in order to integrate the two previously 

mentioned approaches. 

The FIA module takes four inputs. The first one is the dense warp flow _F ∈_ R _[h][×][w][×]_[2] between the original garment image and the garment in the target try-on image, where _h_ and _w_ are the height and width of the garment image, respectively. The second input is the garment feature _G_ extracted by the Garment Net. The third input is the model feature _P_ from the upstream layer of the Denoising Unet. The last input is the spatial embeddings _S_ generated by the Spatial Guider. The detail architecture is depicted in Figure 2. 

Firstly, both the model feature _P_ and garment feature _G_ are combined with the projection of the warp flow _F_ by element-wise summation, as follows: 

**==> picture [148 x 23] intentionally omitted <==**

where _FP_ is the flow feature produced by an MLP projector that aligns the optical flow _F_ with the model feature _P_ . Similarly, _FG_ is the flow features aligned with the garment feature _G_ . Then, the combined features _Pf_ and _Gf_ are infused into the Denoising Unet by joint cross attention. 

**==> picture [215 x 27] intentionally omitted <==**

where the matrices _FQ_ , _FK_ , and _FV_ are standard crossattention weights for query, key, and value, respectively. The denominator ~~�~~ _df_ is the standard scale factor. The output is a model feature _Pe_ modulated by the dense optical flow and the garment feature. The flow _F_ acts as an offset of the intermediate features of the Denoising UNet that turns _P → Pe_ , which brings spatial deformation guidance for implicit warping and inpainting. 

Secondly, the modulated feature _Pe_ is further modulated by the spatial embeddings _S_ . Specifically, given the garment image **I** _g_ , the spatial embeddings _S_ is calculated as: 

**==> picture [150 x 11] intentionally omitted <==**

where **S** _v_ is the Spatial Guider encoder. Different from the garment feature _G_ , the embedding _S_ focuses on high-level spatial information at different granularity. Formally, the spatial embedding _S_ is integrated with the enhanced model feature _Pe_ by a cross-attention layer, as follows: 

**==> picture [222 x 25] intentionally omitted <==**

where the matrices _SQ_ , _SK_ , and _SV_ are standard crossattention weights for query, key, and value respectively. The denominator _[√] ds_ is the standard scale factor. 

We replace all cross-attention layers of the Denoising Unet with the FIA modules. In doing so, it supplements 

the latent space of the diffusion model with positional information of texture details, ensuring that all critical details are more accurately reconstructed. 

**Discussion.** There several possible ways to combine the warp flow and the garment feature and the model feature, such as pixel-wise multiplication, concatenation, and so on. In practice the warp flow often concentrates in local areas with large motion, which is sparse. So we choose feature summation for numeric stability and simplicity. This implicit fusion of flow and diffusion features not only corrects potential deformation errors but also ensures that the final try-on results maintain both accurate alignment and detailed texture preservation, demonstrating superior experimental performance. Detailed ablation study are given in Section 4.3. 

## **3.2. Flow Guider** 

The Flow Guider estimates the dense warp flow _F ∈_ R _[h][×][w][×]_[2] from the original garment image to the target garment in try-on image. In this paper we adopt the Dynamic Semantics Disentangling Module (DSDM) from _D_[4] -VTON [29]. DSDM takes as input the garment image and the condition triplet _T_ including the human pose, the DensePose pose, and the preserve region mask. Two feature pyramid networks (FPN) are used to extract multi-scale features, and DSDM generates local flows for different semantic regions from coarse to fine. In practice we utilize the final flow as an input to FIA, and keep the DSDM freezed during training FIA-VTON. 

## **3.3. Spatial Guider** 

The Spatial Guider aims to extract high-level spatial features from the garment image. In comparison, the garment feature _G_ extracted by a UNet focus on low-level image features. These two features are complementary, so they can fully excavate the information of the garment. 

We adopt FashionCLIP as the Spatial Guider in practice. FashionCLIP is fine-tuned on millions of image-text pairs of fashion clothes, so it achieves better understanding on the patterns of the clothes textures and complex texts than vanilla CLIP. Since the Spatial Guider needs to extract high-level appearance information while keeping spatial information, we select the highest level feature map without visual projection. 

## **3.4. FIA-VTON** 

The FIA-VTON model can be considered as an inpainting version of Stable Diffusion with control signals infused by the FIA modules in Section 3.1. Specifically, given a model image **I** _p ∈_ R _[H][×][W][ ×]_[3] and garment image **I** _g ∈_ R _[H][×][W][ ×]_[3] , our goal is to synthesize them into a realistic image **[ˆ] I** _∈_ R _[H][×][W][ ×]_[3] , which has the same person attributes as in **I** _p_ while retaining the garment from **I** _g_ . 

The input to the Denoising Unet is a 12-channel tensor composed of 4 channels of latent noise _ϵ_ , 4-channel masked person latent _xm_ , 1-channel mask **m** , and 3-channel of the skeleton **s** . The _xm ∈_ R _[h][×][w][×]_[3] is generated from **I** _p_ using OpenPose [3] and HumanParsing [20], and go through a VAE encoder to transform it into the latent space, where _h_ = _[H]_ 8[and] _[ w]_[=] _[W]_ 8[. On the other side, we feed the garment] images **I** _g_ to the Garment Net to obtain the local garment feature in a single time step. The Garment Net is a network similar to a denoising UNet. During training, we initialize the Garment Net and the denoising UNet by the same weights to ensure the consistency between the garment’s local texture features and the denoised image features. 

Along with the aforementioned auxiliary conditioning input, the Garment Net and the denoising UNet are jointly trained by minimizing the following loss function: 

**==> picture [226 x 13] intentionally omitted <==**

where **x** _g_ is the latent feature of the garment image from the VAE encoder, **x** _t_ is the noised latent feature of the tryon image at time step _t_ with noise _ϵ_ , _θ_ is the parameters of denoising UNet model and the garment model, _S_ is the high-level feature of Fashion-CLIP, and _F_ is the predicted flow of the warping module. 

## **4. Experiments** 

In this section, we present an evaluation of our proposed FIA-VTON method on two widely studied datasets [4, 22], and compare it to various state-of-the-art approaches, including Warp-based (VITON-HD [4], HR-VTON [18], GP-VTON [28], DCI-VTON [10], _D_[4] -VTON [29]), and Learning-based methods (LADI-VTON [23], StableVITON [17], IDM-VTON [5], GarDiff [26]). Additionally, we investigate the effect of implicit attention modules in FIA to demonstrate the robustness of our method. 

## **4.1. Experiments Setup** 

**Datasets.** There are two commonly used benchmarks for evaluating high-resolution virtual try-on: VITON-HD [4] and Dress-Code [22]. These two datasets exhibit significant differences in the poses and styles of the models. VITON-HD consists of 13,679 image pairs of front-view upper-body model and upper-body garment images, which are further split into 11,647 training pairs and 2,032 testing. DressCode contains 53792 image pairs of front-view fullbody person and in-store garment images, which are composed of three subsets with different category pairs, that is upper, lower, and dresses. It has 48,392 training set and 5,400 testing set. We follow the experimental setup of previous methods [17, 23] and conduct experiments under the image resolution of 512 × 384. 

**Evaluation metrics.** We conduct evaluations with two testing settings: paired and unpaired. In the paired setting, a 

Table 1. Quantitative results on VITON-HD dataset. We compare with Warp-based and Learning-based virtual try-on methods. FID _p_ /FID _u_ stands for the FID score in the paired/unpaired setting. We follow the data processing approach of GP-VTON and _D_[4] -VTON, using bilinear interpolation for downsampling. An asterisk ( ***** ) indicates cases where cubic downsampling was applied. **Bold** denotes the best score for each metric and Underlining indicates the second best result. 

|Method|SSIM_↑_PSNR_↑_LPIPS_↓_**FIDp** _↓_**FIDu** _↓_|SSIM_↑_PSNR_↑_LPIPS_↓_**FIDp** _↓_**FIDu** _↓_|SSIM_↑_PSNR_↑_LPIPS_↓_**FIDp** _↓_**FIDu** _↓_|SSIM_↑_PSNR_↑_LPIPS_↓_**FIDp** _↓_**FIDu** _↓_|
|---|---|---|---|---|
||Warp-based methods||||
|VITON-HD<br>HR-VTON<br>GP-VTON<br>DCI-VTON<br>_D_4-VTON|0.862<br>-<br>0.117<br>0.878<br>21.61<br>0.105<br>0.884<br>23.41<br>0.081<br>0.880<br>24.01<br>0.080<br>0.892<br>24.71<br>0.065|||-<br>12.12<br>11.383<br>11.27<br>6.031<br>9.072<br>5.521<br>8.754<br>4.845<br>8.53|
||Learning-based methods||||
|LaDI-VTON<br>StableVITON<br>IDM-VTON<br>GarDiff|0.864<br>0.888<br>0.870<br>0.912|22.49<br>-<br>-<br>-|0.096<br>0.073<br>0.102<br>**0.036**|6.602<br>9.48<br>-<br>8.233<br>-<br>8.64<br>6.021<br>7.89|
|**Ours***<br>**Ours**|0.906<br>**0.913**|27.355<br>**27.827**|0.047<br>0.047|5.124<br>**7.869**<br>**4.686**<br>7.914|



garment image is used to reconstruct the image of the person originally wearing it, while in the unpaired setting, the garment worn by the person is replaced with a different one. For both evaluation settings, we adopt a widely used metric: Frechet Inception Distance (FID) [24]. Moreover, in the paired setting where ground truth is available for comparison, we employ Structural Similarity (SSIM) [2], Peak Signal-to-Noise Ratio (PSNR) and Learned Perceptual Image Patch Similarity (LPIPS) [30] to thoroughly evaluate the performance of the try-on results. Note that we find different downsampling methods have a significant impact on both paired and unpaired FID scores. Unless otherwise specified, our experiments use the bilinear interpolation setup from GP-VTON and _D_[4] -VTON as the default. Additionally, we present qualitative results, providing a comparison with previous methods to further demonstrate the effectiveness of our approach. 

**Implementation Details.** We train our models based on the inpainting version’s pre-trained StableDiffusion v2.1[1]. For a fair comparison, we train two separate models for the VITON-HD and DressCode datasets, evaluating them on their respective test sets as done in previous methods. All models are trained with identical hyperparameters: the AdamW [21] optimizer is used with a batch size of 64 and a constant learning rate of 1 _e −_ 5 for 100,000 steps at a resolution of 512 _×_ 384. All experiments are conducted on 4 NVIDIA H100 GPUs, with each model requiring approximately 80 hours of training. 

**==> picture [445 x 7] intentionally omitted <==**

**----- Start of picture text -----**<br>
Target Person Garment VITON-HD HR-VITON GP-VTON DCI-VTON LaDI-VTON StableVITON D4-VTON Ours<br>**----- End of picture text -----**<br>


**==> picture [466 x 62] intentionally omitted <==**

**==> picture [466 x 63] intentionally omitted <==**

**==> picture [466 x 63] intentionally omitted <==**

**==> picture [466 x 63] intentionally omitted <==**

Figure 3. Qualitative comparison on VITON-HD dataset [4]. Examples generated by VITON-HD, HR-VTON, GP-VTON, LaDI-VTON, DCI-VTON, StableVITON, _D_[4] -VTON and our model. Zoom in for a better view. 

Table 2. Quantitative results on DressCode datasets. FID _p_ /FID _u_ stands for the FID score in the paired/unpaired setting. **Bold** denotes the best score for each metric and Underlining indicates the second best result. 

|Method|DressCode-Dresses|DressCode-Lower|DressCode-Upper|
|---|---|---|---|
||SSIM_↑_<br>LPIPS_↓_<br>**FIDu** _↓_<br>**FIDp** _↓_|SSIM_↑_<br>LPIPS_↓_<br>**FIDu** _↓_<br>**FIDp** _↓_|SSIM_↑_<br>LPIPS_↓_<br>**FIDu** _↓_<br>**FIDp** _↓_|
||Warp-based methods|||
|HR-VITON<br>GP-VTON<br>DCI-VTON<br>_D_4-VTON|0.865<br>0.113<br>18.81<br>16.82<br>0.881<br>0.073<br>12.64<br>7.44<br>0.887<br>0.070<br>12.35<br>8.48<br>0.890<br>0.061<br>12.29<br>8.28|0.937<br>0.045<br>16.39<br>11.41<br>0.941<br>0.042<br>16.07<br>7.73<br>0.939<br>0.045<br>15.45<br>7.97<br>0.946<br>0.033<br>14.86<br>6.67|0.916<br>0.071<br>16.82<br>15.37<br>0.947<br>0.036<br>11.98<br>7.38<br>0.942<br>0.041<br>11.64<br>7.47<br>0.942<br>0.041<br>11.00<br>6.55|
||Learning-based methods|||
|LaDI-VTON|0.868<br>0.089<br>13.40<br>-|0.922<br>0.051<br>14.80<br>-|0.928<br>0.049<br>13.26<br>-|
|StableVITON<br>GarDiff|-<br>0.068<br>12.25<br>-<br>0.891<br>0.065<br>12.05<br>8.77|-<br>0.035<br>12.34<br>-<br>0.939<br>0.035<br>12.29<br>8.01|0.937<br>0.039<br>**9.94**<br>-<br>**0.952**<br>0.030<br>11.32<br>8.69|
|Ours|**0.901**<br>**0.049**<br>**10.45**<br>**6.85**<br>**0.952**<br>**0.027**<br>**10.58**<br>**6.02**<br>**0.952**<br>**0.027**<br>10.76<br>**6.22**|||



## **4.2. Comparison with the state-of-the-arts** 

As presented in Table Table 1 and Table 2, our proposed FIA-VTON demonstrates superior performance compared to previous state-of-the-art methods, with a significant margin across all two datasets and in various scenarios. 

**VITON-HD.** We compare the state-of-the-art Warp-based and Learning-based virtual try-on methods on the VITONHD dataset. The quantitative comparison is presented in Table 1. Our method achieves an SSIM/PSNR score of 0.913/27.827, significantly outperforming other methods and demonstrating superior consistency between the generated garment and the model. Moreover, our method achieves outstanding performance in the FID _p_ metric, with a best score of 7.869, indicating a reduced perceptual distance between the generated images and real ones. DCI- 

VTON and _D_[4] -VTON explicitly combine the warping module and diffusion model, which is the most relative to our method. However, they are strongly dependent on the performance of the warping results. This dependency leads to incorrect garment regions when dealing with substantial variations or significant transformations in garment style. DCI-VTON and _D_[4] -VTON perform worse than our method by 0.885 and 0.661 on the **FIDu** metric, respectively. Also when compared to Learning-based methods, our method is still able to achieve more accurate fine-grained garment deformation, enabling better recovery of garment details. That our SSIM score exceeds LdDI-VTON and StableVITON by 0.049 and 0.025. 

Furthermore, we depict the qualitative comparison with other methods on the VITON-HD in Figure 3. As can be 

Table 3. Ablation study on each module of FIA-Diff on VITONHD. The table compares our model’s performance with and without the Flow Guider or Spatial Guider. Removing either module leads to degradation in all metrics. 

|Model|**FIDu** _↓_<br>**FIDp** _↓_<br>SSIM_↑_<br>PSNR_↑_<br>LPIPS_↓_|
|---|---|
|w/o Flow Guider<br>w/o Spatial Guider<br>w/ openclip|8.530<br>5.438<br>0.902<br>26.545<br>0.055<br>8.645<br>6.029<br>0.888<br>24.330<br>0.076<br>8.032<br>4.693<br>**0.914**<br>**27.925**<br>**0.047**|
|Concat Input<br>Cross-Attention|8.204<br>4.950<br>0.912<br>27.604<br>0.047<br>8.31<br>5.28<br>0.904<br>26.945<br>0.051|
|Ours|**7.914**<br>**4.686**<br>0.913<br>27.827<br>**0.047**|



seen, Warp-based methods like HR-VITON and GP-VTON often struggle to generate realistic human body parts, such as abdomens or arms (row 1), resulting in noticeable “copy and pasts” artifacts. The underlying reason is that Warpbased approaches rely heavily on the effectiveness of the warp model and cannot handle overly vigorous deformations. Among Learning-based methods, LADI-VTON, and StableVITON fail to capture accurate garment texture (rows 2,3) due to a lack of guidance of texture information, leading to texture details that differ significantly from the target garment. DCI-VTON and _D_[4] -VTON incorporate warping module and diffusion model to generate natural dressing results. However, they still use explicit warping as a priori information and are dependent on the performance of the warping module. When confronted with drastic deformation of clothing styles, wrong clothing artifacts are usually generated (row 4). In contrast, our FIA-VTON applies the flow generated by the warping module as an implicit guide, skillfully integrating it into the attention layer. It effectively prompts the diffusion model to reconstruct texture details while adeptly handling significant garment deformations. **DressCode.** Table 2 summarizes the performance comparisons on the DressCode dataset. FIA-VTON surpasses other competing methods across all evaluation metrics on “DressCode-Dresses” and “DressCode-Lower” test sets, emphasizing its ability to generate natural fitting results and preserve fine-grained garment textures. Figure 4 illustrates the comparison for different garment types (including upper, lower, and dress) on full-body person images from the DressCode dataset. For uppers, akin to the results in Figure 4, our approach can generate results more consistent with the garment textures, devoid of artifacts. Regarding lowers and dresses, our method can accurately recognize the type and texture of the garments (row 3-5), and perform better in rendering semi-transparent materials (row 6). 

## **4.3. Ablation Study** 

Table 3 and Figure 5 present the ablation study results on the FIA-VTON model, analyzing the contributions of two key components: Flow Guider and Spatial Guider. The ablation study includes three settings: (1) removing the Flow Encoder, (2) removing the Spatil Guider, (3) replac- 

**==> picture [225 x 301] intentionally omitted <==**

Figure 4. Qualitative comparison on the DressCode [22] dataset. FIA-VTON demonstrates a distinct advantage in handling complex textures and Drastic deformation. Please zoom in for more details. 

**==> picture [233 x 186] intentionally omitted <==**

**----- Start of picture text -----**<br>
Target Person Garment w/o Spatial Guider w/o Flow Guider Ours<br>**----- End of picture text -----**<br>


Figure 5. Ablation study on our FIA-VTON. 

ing Fashion-CLIP with OpenCLIP, and (4) the full model. **Effectiveness of the Flow Guider module.** As shown in Table 3, compared to the model without the Flow guider module, the full model (Ours) achieves better performance on all metrics, particularly on FID _p_ . The FID _p_ is reduced from 5.438 to 4.686. In the 1st row of Figure 5 we observe that the lack of the Flow Guider makes the textual informa- 

**==> picture [228 x 156] intentionally omitted <==**

**----- Start of picture text -----**<br>
Garment Concatenation Pixel-wise multiplication FIA<br>**----- End of picture text -----**<br>


Figure 6. Ablation study on Flow Infused Attention (FIA). 

tion in the generated results illegible. This highlights the significant role of the Flow Guider in capturing texture semantics to accurately reconstruct garment details. 

**Effectiveness of the Spatial-Guider module.** From “w/o Spatial Guider” of Table 3, we can observe that removing the Spatial Guider leads to increased **FIDu** (from 7.914 to 8.645) and decreased SSIM (from 0.913 to 0.888), indicating degraded image quality and alignment. Qualitative results in Figure 5 further confirm that without the Spatial Guider, generated garments exhibit misalignment and less coherent blending with the target person, particularly for garments with complex textures or patterns. 

**Variants of Flow Infused Attention.** To demonstrate the superiority of our elaborated Flow Guider, we provide variant experiments in Figure 6 and Table 3. We additionally designed two modules that combine the warping module with diffusion. The ”Concatenation” directly concatenates the warped garment to the input. The ”Pixel-wise multiplication” warps the output features from the reference network and inputs them into the main UNet via a crossattention mechanism. Both methods represent explicit uses of warp flow. In Figure 6, we observed that explicitly invoking warp flow leads to two main issues: First, generated results may lack the fine details of the original garment, especially if there is a complex pattern on the garment. There is an obvious lack of fit between the garment and the human body, especially at the shoulders and sleeves, where the garment does not appear to be able to fully adapt to the shape of the target person’s body, resulting in an incompatible fitting effect. We speculate that such explicit guidance would rely heavily on the effectiveness of the warping module and be limited by the target mask. In contrast, our Flow Guider module provides implicit guidance that helps mitigate these issues, resulting in more natural and visually coherent tryon results. 

## **4.4. In-the-wild senarios** 

Furthermore, we evaluate the wild scenarios to test the robustness and applicability of FIA-VTON in real-world con- 

**==> picture [225 x 267] intentionally omitted <==**

**----- Start of picture text -----**<br>
OOTDiffusion StablVITON Ours<br>**----- End of picture text -----**<br>


Figure 7. Qualitative comparison in the wild scenarios. Compared with state-of-the-art methods (OOTDiffusion, CAT-VTON). Our method generates more natural images that seamlessly combine background, person, and garment in complex scenarios. Zoom in for more details. 

ditions. As shown in Figure 7, FIA-VTON accurately recognizes and integrates the shape of complex garments, such as off-shoulder designs, with the person. It can generate interlaced parts for complex poses, such as sitting. Additionally, it effectively completes and integrates the background with the garment in complex in-the-wild scenarios. 

## **5. Conclusions** 

In this paper, we presented FIA-VTON, a novel approach for virtual try-on that integrates flow-guided attention and high-level semantic features to achieve realistic garment fitting and texture detail preservation. By addressing the shortcomings of previous Warp-based and Learning-based methods, FIA-VTON leverages the Flow Infused Attention module to enhance garment alignment while reducing visual artifacts. In IFA, a Flow Attention is designed to take the dense flow map itself to guide the implicit warp, and extra Spatial Attention to extract high-level semantic information, ensuring consistent garment and model integration. Experimental results demonstrate that our method outperforms existing approaches in both global alignment and fine-grained texture reconstruction. Future work may focus on extending FIA-VTON to handle more diverse clothing types and complex poses, as well as optimizing efficiency for real-time applications. 

## **References** 

- [1] Stability AI. Stable diffusion 2.1. https : / / huggingface . co / stabilityai / stable - diffusion-2-1, 2023. Accessed: 2023-11-15. 5 

- [2] Mikołaj Bi´nkowski, Danica J Sutherland, Michael Arbel, and Arthur Gretton. Demystifying mmd gans. arXiv preprint arXiv:1801.01401, 2018. 5 

- [3] Zhe Cao, Tomas Simon, Shih-En Wei, and Yaser Sheikh. Realtime multi-person 2d pose estimation using part affinity fields. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 7291–7299, 2017. 5 

- [4] Seunghwan Choi, Sunghyun Park, Minsoo Lee, and Jaegul Choo. Viton-hd: High-resolution virtual try-on via misalignment-aware normalization. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 14131–14140, 2021. 2, 5, 6 

- [5] Yisol Choi, Sangkyung Kwak, Kyungmin Lee, Hyungwon Choi, and Jinwoo Shin. Improving diffusion models for virtual try-on. arXiv preprint arXiv:2403.05139, 2024. 5 

- [6] Zheng Chong, Xiao Dong, Haoxiang Li, Shiyue Zhang, Wenqing Zhang, Xujie Zhang, Hanqing Zhao, and Xiaodan Liang. Catvton: Concatenation is all you need for virtual try-on with diffusion models, 2024. 2 

- [7] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021. 2 

- [8] Bo Gao, Junchi Ren, Fei Shen, Mengwan Wei, and Zijun Huang. Exploring Warping-Guided Features via Adaptive Latent Diffusion Model for Virtual try-on . In 2024 IEEE International Conference on Multimedia and Expo (ICME), pages 1–6, Los Alamitos, CA, USA, 2024. IEEE Computer Society. 3 

- [9] Yuying Ge, Yibing Song, Ruimao Zhang, Chongjian Ge, Wei Liu, and Ping Luo. Parser-free virtual try-on via distilling appearance flows. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8485– 8493, 2021. 3 

- [10] Junhong Gou, Siyu Sun, Jianfu Zhang, Jianlou Si, Chen Qian, and Liqing Zhang. Taming the power of diffusion models for high-quality virtual try-on with appearance flow. In Proceedings of the 31st ACM International Conference on Multimedia, pages 7599–7607, 2023. 2, 5 

- [11] Shuyang Gu, Dong Chen, Jianmin Bao, Fang Wen, Bo Zhang, Dongdong Chen, Lu Yuan, and Baining Guo. Vector quantized diffusion model for text-to-image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10696–10706, 2022. 2 

- [12] Xintong Han, Zuxuan Wu, Zhe Wu, Ruichi Yu, and Larry S Davis. Viton: An image-based virtual try-on network. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 7543–7552, 2018. 2 

- [13] Xintong Han, Xiaojun Hu, Weilin Huang, and Matthew R Scott. Clothflow: A flow-based model for clothed person generation. In Proceedings of the IEEE/CVF international conference on computer vision, pages 10471–10480, 2019. 3 

- [14] Sen He, Yi-Zhe Song, and Tao Xiang. Style-based global appearance flow for virtual try-on. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3470–3479, 2022. 3 

- [15] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 2 

- [16] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 2 

- [17] Jeongho Kim, Guojung Gu, Minho Park, Sunghyun Park, and Jaegul Choo. Stableviton: Learning semantic correspondence with latent diffusion model for virtual try-on. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8176–8185, 2024. 2, 5 

- [18] Sangyun Lee, Gyojung Gu, Sunghyun Park, Seunghwan Choi, and Jaegul Choo. High-resolution virtual try-on with misalignment and occlusion-handled conditions. In European Conference on Computer Vision, pages 204–219. Springer, 2022. 5 

- [19] Kedan Li, Min Jin Chong, Jeffrey Zhang, and Jingen Liu. Toward accurate and realistic outfits visualization with attention to details. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 15546– 15555, 2021. 3 

- [20] Peike Li, Yunqiu Xu, Yunchao Wei, and Yi Yang. Selfcorrection for human parsing. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(6):3260–3271, 2020. 5 

- [21] Ilya Loshchilov, Frank Hutter, et al. Fixing weight decay regularization in adam. arXiv preprint arXiv:1711.05101, 5, 2017. 5 

- [22] Davide Morelli, Matteo Fincato, Marcella Cornia, Federico Landi, Fabio Cesari, and Rita Cucchiara. Dress code: Highresolution multi-category virtual try-on. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2231–2235, 2022. 2, 5, 7 

- [23] Davide Morelli, Alberto Baldrati, Giuseppe Cartella, Marcella Cornia, Marco Bertini, and Rita Cucchiara. Ladi-vton: Latent diffusion textual-inversion enhanced virtual try-on. In Proceedings of the 31st ACM International Conference on Multimedia, pages 8580–8589, 2023. 2, 5 

- [24] Gaurav Parmar, Richard Zhang, and Jun-Yan Zhu. On aliased resizing and surprising subtleties in gan evaluation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11410–11420, 2022. 5 

- [25] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 3 

- [26] Siqi Wan, Yehao Li, Jingwen Chen, Yingwei Pan, Ting Yao, Yang Cao, and Tao Mei. Improving virtual tryon with garment-focused diffusion models. arXiv preprint arXiv:2409.08258, 2024. 2, 3, 5 

- [27] Bochao Wang, Huabin Zheng, Xiaodan Liang, Yimin Chen, Liang Lin, and Meng Yang. Toward characteristicpreserving image-based virtual try-on network. In Proceedings of the European conference on computer vision (ECCV), pages 589–604, 2018. 3 

- [28] Zhenyu Xie, Zaiyu Huang, Xin Dong, Fuwei Zhao, Haoye Dong, Xijin Zhang, Feida Zhu, and Xiaodan Liang. Gpvton: Towards general purpose virtual try-on via collaborative local-flow global-parsing learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 23550–23559, 2023. 2, 3, 5 

- [29] Zhaotong Yang, Zicheng Jiang, Xinzhe Li, Huiyu Zhou, Junyu Dong, Huaidong Zhang, and Yong Du. D4-vton: Dynamic semantics disentangling for differential diffusion based virtual try-on. arXiv preprint arXiv:2407.15111, 2024. 2, 3, 4, 5 

- [30] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018. 5 

- [31] Luyang Zhu, Dawei Yang, Tyler Zhu, Fitsum Reda, William Chan, Chitwan Saharia, Mohammad Norouzi, and Ira Kemelmacher-Shlizerman. Tryondiffusion: A tale of two unets. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4606– 4615, 2023. 2 

