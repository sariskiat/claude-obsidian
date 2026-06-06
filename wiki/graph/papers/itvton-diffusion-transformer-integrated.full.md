---
type: paper-fulltext
slug: itvton-diffusion-transformer-integrated
arxiv_id: null
source_path: /Users/saris.kia.adm/.paper-scholar/itvton-diffusion-transformer-integrated/2501.16757.md
paper: "[[itvton-diffusion-transformer-integrated]]"
---
<!-- extracted by afk_extract from 2501.16757.pdf (15p) -->

## **ITVTON: Virtual Try-On Diffusion Transformer Based on Integrated Image and Text** 

## Haifeng Ni[(][�][)] and Ming Xu 

School of Software Engineering, East China Normal University, Shanghai, China `71255902129@stu.ecnu.edu.cn, mxu@cs.ecnu.edu.cn` 

**==> picture [347 x 161] intentionally omitted <==**

**Fig. 1.** The ITVTON model is utilized to create virtual try-on images derived from the VITON-HD [4] dataset (row 1) and the filtered IGPair [23] dataset (row 2). For optimal visual evaluation, it is recommended to examine the images in enlarged form. 

**Abstract.** Virtual try-on, which aims to seamlessly fit garments onto person images, has recently seen significant progress with diffusion-based models. However, existing methods commonly resort to duplicated backbones or additional image encoders to extract garment features, which increases computational overhead and network complexity. In this paper, we propose ITVTON, an efficient framework that leverages the Diffusion Transformer (DiT) as its single generator to improve image fidelity. By concatenating garment and person images along the width dimension and incorporating textual descriptions from both, ITVTON effectively captures garment-person interactions while preserving realism. To further reduce computational cost, we restrict training to the attention parameters within a single Diffusion Transformer (Single-DiT) block. Extensive experiments demonstrate that ITVTON surpasses baseline methods both qualitatively and quantitatively, setting a new standard for virtual tryon. Moreover, experiments on 10,257 image pairs from IGPair confirm its robustness in real-world scenarios. 

**Keywords:** virtual try-on · diffusion transformer · parameter training 

2 Haifeng Ni, Ming Xu 

## **1 Introduction** 

Virtual try-on technology [3,7,9] has emerged as an indispensable tool in e- commerce and personal fashion, significantly enhancing consumers’ shopping experiences by generating realistic try-on images and offering novel marketing avenues for brands and retailers. Traditional methods often rely on Generative Adversarial Networks (GANs) [10] in a two-stage process: first warping the garment according to the person’s pose, then using a generator to merge the warped garment with the person image. Although these approaches improve image realism, they heavily depend on the warping module, leading to potential overfitting, unstable generation quality, limited generalization, and difficulties in handling complex backgrounds or dynamic poses. 

Recently, diffusion models [25,24] have emerged as a compelling alternative for virtual try-on tasks, owing to their superior generative capabilities and enhanced flexibility. Unlike GAN-based pipelines, diffusion models [26] tend to generate more stable, high-quality, and diverse outputs. They also possess strong control and optimization properties during training, making them well-suited for addressing challenges in virtual try-on. Current diffusion-based virtual tryon methods primarily adopt parallel U-shaped UNet structures [22], typically introducing a garment UNet alongside a generative UNet, with attention mechanisms bridging garment features and the person image. While this approach substantially advances virtual try-on, it also entails additional network modules and higher computational costs at both training and inference stages. 

Building upon the progress in diffusion transformers (DiT) [18], our goal is to harness their scalability and generative performance to improve virtual try-on image quality. To streamline the network design and jointly capture person and garment information, we avoid auxiliary image encoders by instead concatenating the garment and person images along the width dimension. As the DiT model capitalizes on textual cues, we further enhance realism by introducing textual descriptions of both the garment and person. Within the transformer backbone, image and text signals are fused directly through attention mechanisms, which proves critical for improving garment-person interactions. To reduce computational overhead, we focus training solely on the attention parameters within a single DiT block (Single-DiT), thereby striking a balance between performance and efficiency. 

The contributions of the present paper are summarized as: 

- We propose _ITVTON_ , a DiT-based virtual try-on model that leverages the strong scaling and generative capabilities of DiT to preserve fine-grained garment features and enhance interactions with person images. 

- We align try-on results by concatenating garment and person images along the width dimension, integrating image-text descriptions for improved generation quality. This design eliminates the need for an extra image encoder, resulting in a compact network structure. 

- We introduce a parameter-efficient training scheme that targets only the attention parameters within a Single-DiT block, reducing training time and computational overhead while enhancing stability and output fidelity. 

ITVTON 

3 

## **2 Related Work** 

The virtual try-on task [9] involves generating a realistic photograph of a model wearing a specified outfit while maintaining consistency in other regions. Scholars initially explored generative adversarial networks (GANs)[10] for this task, employing a two-stage process: aligning the garment with the model and fusing them via a GAN-based network. VITON-HD[4] introduces ALIgnment-Aware Segment (ALIAS) normalization to address misalignment by segmenting and fitting garments. GP-VTON [30] proposes a Local-Flow Global-Parsing (LFGP) warping module and a dynamic gradient truncation strategy to simulate garment deformation more effectively. However, GAN-based models suffer from unstable training and struggle to preserve garment texture details. 

Recently, diffusion models have demonstrated superior stability and quality in virtual try-on. LADI-VTON [17] and DCI-VTON [11] align garments with the body before using diffusion models for fusion. TryOnDiffusion [34] employs a parallel UNet to unify garment deformation and fusion. OOTDiffusion [31] captures detailed clothing features in a single step, while IDM-VTON [5] incorporates the IP-Adapter and pose guidance for enhanced control. Paint-by-Example [32] reframes try-on as an inpainting problem, fine-tuning diffusion models on virtual try-on datasets to enable fine-grained image control. 

Much work has been inspired by the PCDMs [26] second-stage inpainting conditional diffusion model, which uses the concatenation of source conditions and target images and is gradually gaining popularity in virtual try-on tasks. For example, CatVTON [6] similarly fine-tunes the restoration diffusion model but eliminates the text encoder and cross-attention block, training only the selfattention block to create a lightweight network structure. IC-LoRA [14] posits that the text-to-image diffusion transformer model [18] is inherently contextually generative, enabling high-quality image generation through integrated captioning of multiple images using small datasets (e.g., 20–100 samples) with taskspecific LoRA [13] tuning. In this paper, we adopt the DiT model as a prior and achieve a high-quality, high-fidelity virtual try-on task through a simple network architecture and an efficient training strategy, which involves splicing garment and person images along the width dimension and using integrated image-text descriptions as inputs. Additionally, our approach further improves generation performance and training efficiency while maintaining a lightweight network structure. 

## **3 Proposed Method** 

## **3.1 Background on Diffusion Model** 

Stable Diffusion. The stable diffusion [21] is a prominent example of the latent diffusion model. It comprises a variational autoencoder (VAE) _Eθ_ , a CLIP [19] text encoder _τθ_ , and a denoising UNet _ϵθ_ . The key advantage is that image information is compressed into a low-dimensional latent space, which significantly reduces the computational cost associated with training and inference. The input 

4 Haifeng Ni, Ming Xu 

**==> picture [347 x 157] intentionally omitted <==**

**----- Start of picture text -----**<br>
 Train: MM-DiT Block  Concat in the Width Dim<br> guidance_scale = 3.5                       Single-DiT Block            Concat in Channel Dim<br> Inference:                       MM-DiT Block Attention           Frozen Module<br>VAE Encoder  guidance_scale = 30                       Single-DiT Block Attention           Trainable Module<br>... ...<br>“[TRY-ON] In this image, a  ... ...<br>model has changed into the<br>clothing on the left,[IMAGE-1]<br>Comprehensive display of<br>clothing,[IMAGE-2] A model<br>wears the same clothing in the living environment.” Transformer<br>VAE Decoder<br>Text Encoder<br>**----- End of picture text -----**<br>


**Fig. 2.** Overview of ITVTON. Our approach achieves high-quality virtual try-on by concatenating the garment image with the target person image along the width dimension and incorporating integrated image-text representations(as illustrated in the figure). Only the attention parameters in the Single-DiT module remain learnable during training, ensuring a streamlined and efficient try-on network. 

consists of an image **x** along with textual cues **y** , and the following loss function is minimized during training: 

**==> picture [286 x 13] intentionally omitted <==**

where **t** _∈{_ 1 _, . . . , T }_ represents the time step of forward noise addition in the diffusion model, and **Zt** denotes the VAE-encoded image with added random noise _ϵ ∼N_ (0 _,_ 1). 

FLUX. FLUX is a type of diffusion model that employs the DiT architecture and is trained using a parameterize rectified flow model (RF) [8]. The training process of FLUX is defined as a rectified flow that links the data distribution to the noise distribution via a straight line. A key characteristic of the forward process is that **zt** is obtained by linearly interpolating the data **x** 0 and the noise _ϵ_ , i.e.: 

**==> picture [243 x 11] intentionally omitted <==**

The core concept of the rectified flow technique employed in FLUX is to streamline the training and inference processes by introducing a flow-matching method that enhances the efficiency of image generation. The loss function is defined as 

**==> picture [284 x 21] intentionally omitted <==**

where _λ_ **t** corresponds to a signal-to-noise ratio and **wt** is a time dependentweighting factor. _ϵΘ_ ( **zt** _,_ **t** ) denotes the network parameters of the transformer module. 

ITVTON 

5 

## **3.2 ITVTON** 

The ITVTON architecture, illustrated in Fig. 2, aims to generate high-quality and high-fidelity virtual try-on images using a simple network structure and minimal training cost. This section outlines the components of our network architecture and the underlying architectural rationale, as well as explores efficient parameter training strategies. 

**Network Module** Previous approaches primarily focus on achieving detailed alignment between garments and persons. For instance, LADI-VTON [17] first warps the garments to match the model’s pose, then uses a diffusion model to blend the warped garments with the model. IDM-VTON [5] employs a parallel UNet to extract garment features and incorporates an attention mechanism into the generative UNet to maintain alignment during virtual try-on. However, these methods either lack sufficient accuracy or necessitate the introduction of additional network modules for their implementation, failing to meet our expectations. 

We found that the mature generative model exhibits a strong ability to generate masked regions; for example, when the mouth of a face is masked, the model is likely to generate an intact face upon repair. Based on this, if the clothing area of the model is masked and subsequently fed into the model along with a garment image and a mask image, what would happen during repair? The model would likely generate content for the masked area based on the image information outside the mask. However, to generate a high-quality and high-fidelity image of the garment change, the model must also be trained and fine-tuned. Furthermore, because the attention mechanism of FLUX’s transformer module processes image information in conjunction with textual data, it is beneficial to leverage the model’s textual input. 

Consequently, we directly incorporate the three modules: 

**VAE** : The VAE encoder transforms pixel-level images into a latent representation to optimize computational efficiency. Compared to traditional VAEs, the FLUX series VAE outputs latent features. Before these features are fed into the diffusion model, a patching operation is performed, stacking 2 _×_ 2 pixel blocks directly along the channel dimension. This method preserves the original resolution of each pixel block and prevents the loss of critical image feature information. Consequently, we concatenate the garment and person features along the width dimension, leveraging the robust capabilities of the VAE while ensuring alignment between inputs. 

**Text Encoder** : Two text encoders (CLIP ViT-L and T5-XXL [20]) process textual prompts as input, enabling the generation of more controllable, highquality images. Inspired by IC-LoRA [14], multi-image integrated text inputs have a positive impact on virtual try-on tasks. Therefore, we utilize multi-image integrated text (text format shown in Fig. 2) as input, enhancing the controllability and quality of the generated images. 

**Transformer** : The transformer synthesizes the final try-on image by integrating features from the latent space. It accepts concatenated garment and 

6 Haifeng Ni, Ming Xu 

person features, along with noise and masks as image inputs, and receives multiimage integrated titles as text inputs. By seamlessly integrating all this information, the transformer facilitates effective learning and synthesis of the final try-on image. 

Our network architecture does not rely on additional modules, and the tryon task can be efficiently performed using only the inherent functionality of the model. 

**Architectural Inference** In this work, we utilize stitched garment and person images as input and employ integrated image-text pairs as the textual input. This approach ensures a streamlined network architecture and simplifies the preprocessing steps prior to inference. 

Specifically, consider a garment image **Ig** _∈_ R[3] _[×][H][×][W]_ and a person image **Ip** _∈_ R[3] _[×][H][×][W]_ , which are concatenated along the width dimension to generate **Cgp** _∈_ R[3] _[×][H][×]_[2] _[W]_ . The corresponding binary, clothing-agnostic mask map of the person image, denoted **mp** _∈_ R _[H][×][W]_ , is concatenated with an all-zero mask map of the same dimensions to produce **mop** _∈_ R _[H][×]_[2] _[W]_ . 

**==> picture [209 x 26] intentionally omitted <==**

where © denotes concatenation along the width dimension, **O** represents an all-zero mask image. **Cgp** is multiplied with the processed elements of **mop** to generate the clothing-agnostic person and garment-spliced image **Cmasked** _∈_ R[3] _[×][H][×]_[2] _[W]_ . **Cmasked** is then encoded using the VAE encoder _E_ to obtain **Vmasked** _∈_ R[16] _[×][H]_ 8 _[×]_[2] _[W]_ 8 , which has a channel value of 16. Finally, **mop** is interpolated to produce **Xom** _∈_ R[64] _[×][H]_ 8 _[×]_[2] _[W]_ 8 with a channel value of 64. 

**==> picture [238 x 26] intentionally omitted <==**

where _⊗_ denotes element-wise multiplication. **Vmasked** and **Xom** undergo a packing operation to generate **Pmasked** _∈_ R[64] _[×]_[(] 8 _H×_ 2 _[×]_ 8[2] _×[W]_ 2[)] and **Pom** _∈_ R[256] _[×]_[(] 8 _H×_ 2 _[×]_ 8[2] _×[W]_ 2[)] , respectively, before being fed into the transformer module.The channel values are scaled by a factor of 4, while the width and height are each halved, then multiplied and compressed into a single-dimensional value. 

**==> picture [234 x 25] intentionally omitted <==**

The text encoder _τ_ processes the integrated image-text pair, producing _τtext_ . Prior to noise reduction, **Pmasked** and **Pom** are concatenated along the channel dimension with random noise **Zt** of the same size as **Pmasked** , which is then fed into the transformer module to predict **Zt** _−_ **1** alongside _τtext_ . 

**==> picture [286 x 11] intentionally omitted <==**

ITVTON 7 

where _⊙_ denotes the splicing operation along the width dimension. **Z0** _∈_ R[64] _[×]_[(] 8 _H×_ 2 _[×]_ 8[2] _×[W]_ 2[)] is obtained after t-steps of cyclic noise reduction, followed by **Vcon** _∈_ R[16] _[×][H]_ 8 _[×]_[2] _[W]_ 8 , which is generated after the unpacking operation. Next, **Icon** _∈_ R[3] _[×][H][×]_[2] _[W]_ is derived through VAE decoding. The final image **Iresult** _∈_ R[3] _[×][H][×][W]_ is produced after cropping. 

**Parameter-Efficient Training** The pre-trained DiT model demonstrates strong capability and robustness in generating masked regions; however, fine-tuning is required to ensure accurate interaction between garment and person features, thereby generating efficient images. On the other hand, these pre-trained models already possess a wealth of prior knowledge, and an excessive number of trainable parameters not only increases the training burden—requiring more GPU memory and a longer training duration—but may also degrade the model’s existing performance. 

The Transformer in FLUX comprises the MM-DiT block structures and the Single-DiT block structures. The MM-DiT block facilitates the fusion of the two modal information streams, which are subsequently fed into the Single-DiT block to deepen the model’s architecture and enhance its learning capability. Additionally, a parallel attention mechanism is incorporated into the SingleDiT blocks to further optimize model performance. The attention within these blocks fuses both image and text information for processing. We hypothesized that the attention mechanism is crucial for achieving high-quality results and conducted experiments to identify the most relevant blocks. Specifically, we configured the trainable components as the all attention layer, the attention layer in the MM-DiT block, and the attention layer in the Single-DiT block. The experimental results indicate that training all attention layers, the attention layer in the MM-DiT block, and the attention layer in the Single-DiT block can generate satisfactory trial-wear effects; however, the metrics obtained by training the attention layer in the Single-DiT block are optimal, with the lowest number of training parameters. Therefore, we employ a parameter-efficient training strategy to fine-tune the attention layer within the Single-DiT block, which contains 1076.2M parameters. Additionally, we apply a 10% probability to discard multiimage integrated text cues during training to improve the model’s generalization ability. 

## **4 Experimentation and Analysis** 

## **4.1 Datasets** 

Our experiments utilized two publicly available fashion datasets: VITON-HD [4] and IGPair [23]. The VITON-HD [4] dataset comprises 13,679 image pairs, including 11,647 pairs in the training set and 2,032 pairs in the test set. Each pair consists of a frontal upper-body image of a person paired with an upper-body clothing image, both at a resolution of 768 _×_ 1024. The IGPair [23] dataset contains 324,857 image pairs with clothing classified into 18 categories encompassing 

## 8 Haifeng Ni, Ming Xu 

**==> picture [277 x 205] intentionally omitted <==**

**----- Start of picture text -----**<br>
  DCI-VTON         GP-VTON      OOTDiffusion    StableVITON       IDM-VTON         CatVTON              Ours<br>**----- End of picture text -----**<br>


**Fig. 3.** Qualitative comparison on the VITON-HD [4] Dataset. ITVTON exhibits significant advantages in processing complex patterns and text. We recommend zooming in for a detailed inspection. 

a broad spectrum of styles, such as casual, formal, sports, fashion, and vintage. For each apparel item, 2 to 5 model images taken from various angles are provided, with a resolution of 1024 _×_ 1280. From these, we meticulously selected 10,381 image pairs featuring the same model in multiple poses while wearing the same attire. Of these pairs, 10,257 are allocated for training purposes, with the remainder reserved for testing. We employ OpenPose [2,27,29] and HumanParsing [16] to generate garment-independent masks and uniformly adjust the resolution of all images to 768 _×_ 1024 pixels. 

## **4.2 Implementation Details** 

We utilize the pre-trained FLUX.1-Fill-dev model for training. We trained the model on the VITON-HD [4] dataset and evaluated its performance against other approaches using a designated test set. Additionally, we trained the model on the IGPair [23] dataset to assess its generative capability in realistic scenarios. Each model was trained with identical hyperparameters: 36,000 steps at a resolution of 384 _×_ 512 using the AdamW 8-bit optimizer, a batch size of 4, and a constant learning rate of 1e-5. All experiments were conducted on a single NVIDIA A100 GPU. 

## **4.3 Qualitative Comparison** 

We evaluated the visual quality using the VITON-HD [4] dataset. As shown in Fig. 3, it demonstrates the try-on effects of various styles and patterns from 

ITVTON 

9 

the dataset, and compares the try-on variations across different methods. While other methods encounter issues such as color discrepancies and inconsistent details during virtual try-ons, ITVTON delivers more realistic and detailed outcomes owing to its simplicity and efficiency. 

Additionally, we assess ITVTON on selected IGPair [23] datasets and benchmark it against other diffusion-based VTON methods. As shown in Fig. 4, ITVTON demonstrates its capability to accurately recognize and model the shape of complex garments, such as a sheath dress, and seamlessly integrate them with human figures. Notably, it can generate high-quality images even in complex poses (e.g., sitting on the chair). In intricate field scenes, ITVTON effectively handles the integration of backgrounds with clothing, maintaining consistency and realism across diverse and challenging environments. 

## **4.4 Quantitative Comparison** 

For the inference results of the test pair dataset, we use two metrics, frechet inception distance (FID) [12] and kernel inception distance (KID) [1], to compare the gap between real and generated images, and two metrics, structural similarity index (SSIM) [28] and learned perceptual image patch similarity (LPIPS) [33] which are two metrics to measure the overall structural and perceptual similarity between images. 

We conducted a quantitative evaluation of several state-of-the-art open-source virtual try-on methods using the VITON-HD [4] dataset. Comparisons were made in paired settings to assess the similarity between the synthesized results and ground truth, as well as the model’s generalization performance. The results, presented in Table 1, show that our method outperforms all others across all metrics. This demonstrates the efficacy of our simplified network architecture and parameter fine-tuning in virtual try-on tasks. Additionally, the data indicates that both CatVTON [6] and IDM-VTON [5] demonstrated strong performance in virtual try-on tasks. 

## **4.5 Ablation Studies** 

We conducted an ablation study to investigate the effects of several factors: 

1. different trainable modules, 

2. the inclusion or exclusion of multi-image integrated text during both training and inference, 

3. the impact of the guidance scale on the try-on results. 

For a fair comparison, we trained different versions of the model using the VITON-HD [4] dataset. 

**Trainable Modules** : We evaluated three trainable modules in Transformer: 1) all attention parameters. 2) attention parameters of MM-DiT blocks. 3) attention parameters of Single-DiT blocks, with progressively fewer trainable parameters. Table 2 presents the evaluation results on the VITON-HD [4] dataset. We 

10 Haifeng Ni, Ming Xu 

**Table 1.** Quantitative Comparison with Other Methods. We compare metric performance in paired settings on the VITON-HD [4] dataset. The best and second-best results are bolded and underlined, respectively. 

|Methods|_SSIM ↑FID ↓KID ↓LPIPS ↓_|_SSIM ↑FID ↓KID ↓LPIPS ↓_|
|---|---|---|
|DCI-VTON [11]<br>StableVITON [15]<br>GP-VTON [30]<br>OOTDifusion [31]<br>IDM-VTON [5]<br>CatVTON [6]|0.8620<br>9.408<br>4.547<br>0.8543<br>6.439<br>0.942<br>0.8701<br>8.726<br>3.944<br>0.8187<br>9.305<br>4.086<br>0.8499<br>5.762<br>0.732<br>0.8704<br>5.425<br>0.411|0.0606<br>0.0905<br>0.0585<br>0.0876<br>0.0603<br>0.0565|
|ITVTON (Ours)|**0.8734**<br>**5.064**<br>**0.264**|**0.0530**|



**Table 2.** Ablation results of different trainable module on VITON-HD [4] dataset.Taken together, the results indicate that training the attention parameters within the Single-DiT block achieves the optimal performance.The best results are shown in bold. 

|Trainable Module|_SSIM ↑FID ↓KID ↓LPIPS ↓_|Trainable Params(M)|
|---|---|---|
|All Attention<br>MM-DiT Block Attention|0.8702<br>5.155<br>**0.250**<br>0.0557<br>0.8641<br>5.438<br>0.332<br>0.0578|1611.23<br>1434.93|
|Single-DiT Block Attention|**0.8734**<br>**5.064**<br>0.264<br>**0.0530**|1076.2|



observe that the model trained with attention parameters within the Single-DiT block performs optimally across three metrics. The other metric, KID, shows only a slight decrease compared to the model that trains all attention parameters, while the Single-DiT model uses only 1076.2M attention parameters. Fewer training weights both reduce memory requirements and speed up training, and do not pull down performance. This demonstrates that training the attention parameters within the Single-DiT block is optimal. 

**Multi-Image Integrated Text** : Ordinary text, such as “A model is wearing a top.”. Multi-image integrated text is shown in Fig. 2. Both types of text inputs were employed during training and inference. As shown in Table 3, two metrics (SSIM and LPIPS) perform optimally when only multi-image integrated text is used during inference, although the improvement is marginal compared to the model using multi-image integrated text for both training and inference. Additionally, the other two metrics (FID and KID) are optimal for the model that utilizes multi-image integrated text for both training and inference, demonstrating that this approach is the most effective. 

ITVTON 11 

**==> picture [347 x 205] intentionally omitted <==**

**----- Start of picture text -----**<br>
OOTDiffusion      IDM-VTON         CatVTON              Ours<br>Condition<br>2<br>3.5<br>30<br>Fig. 5. Comparison of the effects of<br>setting different guidance scales during<br>training. The model with the guidance<br>Train: Guidance_Scale<br>**----- End of picture text -----**<br>


**Fig. 5.** Comparison of the effects of setting different guidance scales during training. The model with the guidance scale set to 30 during inference and the guidance scale set to 2 or 30 during training does not perform well when generating garments with text or complex patterns. 

**Fig. 4.** Qualitative comparison in field scenarios demonstrates that our method generates more natural try-on effects, even in complex scenes and with varied postures. 

**==> picture [243 x 166] intentionally omitted <==**

**----- Start of picture text -----**<br>
Guidance_Scale<br>Condition<br>        0                   3.5                 15                  30                  45                  70<br>**----- End of picture text -----**<br>


**Fig. 6.** Comparison of the effects of setting different guidance scales during inference. 

**Guidance Scale** : The guidance scale is crucial in both the training and inference processes of the FLUX model. First, adjust the guidance scale setting to 30 for inference, as shown in Table 4. The metrics FID, KID, and LPIPS 

12 Haifeng Ni, Ming Xu 

**Table 3.** Ablation results from post-training inference on the VITON-HD [4] dataset using different text inputs are presented. The best results are shown in bold. Overall, training with integrated text and reasoning with integrated text yield the best performance. 

|Train|Inference|_SSIM ↑_|_FID ↓_|_KID ↓_|_LPIPS ↓_|
|---|---|---|---|---|---|
|Ordinary Text Integrated Text|Ordinary Text Integrated Text|||||
|✓<br>-<br>✓<br>-<br>-<br>✓|✓<br>-<br>-<br>✓<br>✓<br>-|0.8727<br>5.178<br>0.377<br>0.0534<br>**0.8737**<br>5.169<br>0.379<br>**0.0529**<br>0.8723<br>5.098<br>0.298<br>0.0537||||
|-<br>✓|-<br>✓|0.8734<br>**5.064**<br>**0.264**<br>0.0530||||



**Table 4.** Ablation results for various guidance scale settings during training. The best results are shown in bold. Setting the guidance scale to 30 during inference and to 2, 3.5, and 30 during training results in very similar values for the four metrics of the VITON-HD [4] test set. However, collectively, a guidance scale of 3.5 emerges as the optimal setting. 

|Guidance<br>~~S~~cale|_SSIM ↑FID ↓KID ↓LPIPS ↓_|
|---|---|
|2|0.8727<br>5.246<br>0.394<br>0.0532|
|3.5|0.8734<br>**5.064**<br>**0.264**<br>**0.0530**|
|30|**0.8748**<br>5.068<br>0.267<br>0.0540|



achieve optimal results with a guidance scale setting of 3.5 during training, while the SSIM metric is slightly lower compared to the model with a guidance scale of 30 during training. This suggests that setting the guidance scale to 3.5 during training yields better performance. As shown in Fig. 5, models with the guidance scale set to 2 or 30 during training do not perform well when generating garments with repeated or intricate patterns, while models with the guidance scale set to 3.5 can accurately reproduce repeated patterns or images. Next, set the guidance scale to 3.5 for training and evaluate the effects of different guidance scale settings during inference. As shown in Fig. 6, setting the guidance scale to 30 or 45 during inference results in the best restoration of color and pattern. For example, setting the guidance scale to 0 during inference results in color discrepancies and broken patterns. In summary, setting the guidance scale to 3.5 for training and 30 for inference yields the best inference results. 

## **5 Conclusion** 

In this paper, we introduced ITVTON, a simple yet effective virtual try-on diffusion transformer model with 1,076.2M trainable parameters. ITVTON spliced garment and person images along the width dimension, taken integrated imagetext descriptions as input, and required only fine-tuning to generate high-quality, high-fidelity virtual try-on images. 

Extensive experiments have demonstrated that ITVTON could achieve superior qualitative and quantitative results compared to state-of-the-art methods, 

ITVTON 

13 

while maintaining a simple and efficient network architecture. These findings also suggested that ITVTON would hold significant potential for widespread application in virtual try-on technology. 

## **References** 

1. Bi´nkowski, M., Sutherland, D.J., Arbel, M., Gretton, A.: Demystifying mmd gans (2021), https://arxiv.org/abs/1801.01401 

2. Cao, Z., Simon, T., Wei, S.E., Sheikh, Y.: Realtime multi-person 2d pose estimation using part affinity fields. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (July 2017) 

3. Chen, C.Y., Chen, Y.C., Shuai, H.H., Cheng, W.H.: Size does matter: Size-aware virtual try-on via clothing-oriented transformation try-on network. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 7513– 7522 (October 2023) 

4. Choi, S., Park, S., Lee, M., Choo, J.: Viton-hd: High-resolution virtual try-on via misalignment-aware normalization. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 14131–14140 (June 2021) 

5. Choi, Y., Kwak, S., Lee, K., Choi, H., Shin, J.: Improving diffusion models for authentic virtual try-on in the wild (2024), https://arxiv.org/abs/2403.05139 

6. Chong, Z., Dong, X., Li, H., Zhang, S., Zhang, W., Zhang, X., Zhao, H., Liang, X.: Catvton: Concatenation is all you need for virtual try-on with diffusion models (2024), https://arxiv.org/abs/2407.15886 

7. Dong, H., Liang, X., Shen, X., Wang, B., Lai, H., Zhu, J., Hu, Z., Yin, J.: Towards multi-pose guided virtual try-on network. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) (October 2019) 

8. Esser, P., Kulal, S., Blattmann, A., Entezari, R., M¨uller, J., Saini, H., Levi, Y., Lorenz, D., Sauer, A., Boesel, F., Podell, D., Dockhorn, T., English, Z., Rombach, R.: Scaling rectified flow transformers for high-resolution image synthesis. In: Fortyfirst International Conference on Machine Learning (2024), https://openreview. net/forum?id=FPnUhsQJ5B 

9. Gao, B., Ren, J., Shen, F., Wei, M., Huang, Z.: Exploring warping-guided features via adaptive latent diffusion model for virtual try-on. In: 2024 IEEE International Conference on Multimedia and Expo (ICME). pp. 1–6. IEEE (2024) 

10. Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., Bengio, Y.: Generative adversarial networks. Commun. ACM **63** (11), 139–144 (Oct 2020). https://doi.org/10.1145/3422622, https://doi.org/10. 1145/3422622 

11. Gou, J., Sun, S., Zhang, J., Si, J., Qian, C., Zhang, L.: Taming the power of diffusion models for high-quality virtual try-on with appearance flow. In: Proceedings of the 31st ACM International Conference on Multimedia. p. 7599–7607. MM ’23, Association for Computing Machinery, New York, NY, USA (2023). https://doi.org/10.1145/3581783.3612255, https://doi.org/10.1145/ 3581783.3612255 

12. Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., Hochreiter, S.: Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems **30** (2017) 

## 14 Haifeng Ni, Ming Xu 

13. Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W.: Lora: Low-rank adaptation of large language models (2021), https://arxiv. org/abs/2106.09685 

14. Huang, L., Wang, W., Wu, Z.F., Shi, Y., Dou, H., Liang, C., Feng, Y., Liu, Y., Zhou, J.: In-context lora for diffusion transformers (2024), https://arxiv.org/abs/ 2410.23775 

15. Kim, J., Gu, G., Park, M., Park, S., Choo, J.: Stableviton: Learning semantic correspondence with latent diffusion model for virtual try-on. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 8176–8185 (June 2024) 

16. Li, P., Xu, Y., Wei, Y., Yang, Y.: Self-correction for human parsing. IEEE Transactions on Pattern Analysis and Machine Intelligence **44** (6), 3260–3271 (2022). https://doi.org/10.1109/TPAMI.2020.3048039 

17. Morelli, D., Baldrati, A., Cartella, G., Cornia, M., Bertini, M., Cucchiara, R.: Ladi-vton: Latent diffusion textual-inversion enhanced virtual try-on. In: Proceedings of the 31st ACM International Conference on Multimedia. p. 8580–8589. MM ’23, Association for Computing Machinery, New York, NY, USA (2023). https://doi.org/10.1145/3581783.3612137, https://doi.org/10.1145/ 3581783.3612137 

18. Peebles, W., Xie, S.: Scalable diffusion models with transformers. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 4195– 4205 (October 2023) 

19. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., Sutskever, I.: Learning transferable visual models from natural language supervision. In: Meila, M., Zhang, T. (eds.) Proceedings of the 38th International Conference on Machine Learning. Proceedings of Machine Learning Research, vol. 139, pp. 8748–8763. PMLR (18–24 Jul 2021), https://proceedings.mlr.press/v139/radford21a.html 

20. Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., Liu, P.J.: Exploring the limits of transfer learning with a unified textto-text transformer. Journal of Machine Learning Research **21** (140), 1–67 (2020), http://jmlr.org/papers/v21/20-074.html 

21. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent diffusion models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 10684– 10695 (June 2022) 

22. Ronneberger, O., Fischer, P., Brox, T.: U-net: Convolutional networks for biomedical image segmentation. In: Navab, N., Hornegger, J., Wells, W.M., Frangi, A.F. (eds.) Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015. pp. 234–241. Springer International Publishing, Cham (2015) 

23. Shen, F., Jiang, X., He, X., Ye, H., Wang, C., Du, X., Li, Z., Tang, J.: Imagdressingv1: Customizable virtual dressing (2024), https://arxiv.org/abs/2407.12705 

24. Shen, F., Wang, C., Gao, J., Guo, Q., Dang, J., Tang, J., Chua, T.S.: Long-term talkingface generation via motion-prior conditional diffusion model. arXiv preprint arXiv:2502.09533 (2025) 

25. Shen, F., Ye, H., Liu, S., Zhang, J., Wang, C., Han, X., Yang, W.: Boosting consistency in story visualization with rich-contextual conditional diffusion models. arXiv preprint arXiv:2407.02482 (2024) 

26. Shen, F., Ye, H., Zhang, J., Wang, C., Han, X., Yang, W.: Advancing pose-guided image synthesis with progressive conditional diffusion models. arXiv preprint arXiv:2310.06313 (2023) 

ITVTON 15 

27. Simon, T., Joo, H., Matthews, I., Sheikh, Y.: Hand keypoint detection in single images using multiview bootstrapping. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (July 2017) 

28. Wang, Z., Bovik, A., Sheikh, H., Simoncelli, E.: Image quality assessment: from error visibility to structural similarity. IEEE Transactions on Image Processing **13** (4), 600–612 (2004). https://doi.org/10.1109/TIP.2003.819861 

29. Wei, S.E., Ramakrishna, V., Kanade, T., Sheikh, Y.: Convolutional pose machines. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (June 2016) 

30. Xie, Z., Huang, Z., Dong, X., Zhao, F., Dong, H., Zhang, X., Zhu, F., Liang, X.: Gp-vton: Towards general purpose virtual try-on via collaborative local-flow global-parsing learning (2023), https://arxiv.org/abs/2303.13756 

31. Xu, Y., Gu, T., Chen, W., Chen, C.: Ootdiffusion: Outfitting fusion based latent diffusion for controllable virtual try-on (2024), https://arxiv.org/abs/2403.01779 

32. Yang, B., Gu, S., Zhang, B., Zhang, T., Chen, X., Sun, X., Chen, D., Wen, F.: Paint by example: Exemplar-based image editing with diffusion models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 18381–18391 (June 2023) 

33. Zhang, R., Isola, P., Efros, A.A., Shechtman, E., Wang, O.: The unreasonable effectiveness of deep features as a perceptual metric. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (June 2018) 

34. Zhu, L., Yang, D., Zhu, T., Reda, F., Chan, W., Saharia, C., Norouzi, M., Kemelmacher-Shlizerman, I.: Tryondiffusion: A tale of two unets. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 4606–4615 (June 2023) 

