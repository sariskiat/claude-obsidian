---
type: paper-fulltext
slug: improving-virtual-try-on-with-garment-focused-diffusion-models
arxiv_id: null
source_path: /Users/saris.kia.adm/.paper-scholar/improving-virtual-try-on-with-garment-focused-diffusion-models/2409.08258.md
paper: "[[improving-virtual-try-on-with-garment-focused-diffusion-models]]"
---
<!-- extracted by afk_extract from 2409.08258.pdf (16p) -->

# **Improving Virtual Try-On with Garment-focused Diffusion Models** 

Siqi Wan[1] _[∗]_ , Yehao Li[2] , Jingwen Chen[2] , Yingwei Pan[2] , Ting Yao[2] , Yang Cao[1] , and Tao Mei[2] 

> 1 University of Science and Technology of China 

> 2 HiDream.ai Inc. 

```
wansiqi4789@mail.ustc.edu.cn,{liyehao,chenjingwen,pandy,tiyao}@hidream.ai
forrest@ustc.edu.cn,tmei@hidream.ai
```

**Abstract.** Diffusion models have led to the revolutionizing of generative modeling in numerous image synthesis tasks. Nevertheless, it is not trivial to directly apply diffusion models for synthesizing an image of a target person wearing a given in-shop garment, i.e., image-based virtual try-on (VTON) task. The difficulty originates from the aspect that the diffusion process should not only produce holistically high-fidelity photorealistic image of the target person, but also locally preserve every appearance and texture detail of the given garment. To address this, we shape a new Diffusion model, namely GarDiff, which triggers the garment-focused diffusion process with amplified guidance of both basic visual appearance and detailed textures (i.e., high-frequency details) derived from the given garment. GarDiff first remoulds a pre-trained latent diffusion model with additional appearance priors derived from the CLIP and VAE encodings of the reference garment. Meanwhile, a novel garment-focused adapter is integrated into the UNet of diffusion model, pursuing local fine-grained alignment with the visual appearance of reference garment and human pose. We specifically design an appearance loss over the synthesized garment to enhance the crucial, high-frequency details. Extensive experiments on VITON-HD and DressCode datasets demonstrate the superiority of our GarDiff when compared to state-of-the-art VTON approaches. Code is publicly available at: https://github.com/siqi0905/GarDiff/tree/master. 

**Keywords:** Virtual Try-on · Diffusion Model · Appearance Prior 

## **1 Introduction** 

Image-based Virtual Try-ON (VTON), a prominent research topic in computer vision field, aims to synthesize an image of a specific person wearing a desired in-shop garment. Such automatic generation of person images sidesteps the requirement of physical fitting and thus has ushered in a new era of creativity for e-commerce and metaverse. Practical VTON systems have a tremendous potential impact on real-world applications, e.g., online shopping, fashion catalog 

> _∗_ This work was performed at HiDream.ai. 

2 Siqi Wan et al. 

creation, etc. The objective of VTON task is three-fold: 1) human body alignment: the synthesized person image should conform to the human body/pose of given specific person; 2) garment fidelity: the synthesized person image should preserve every appearance and texture detail of the in-shop garment; 3) quality: the synthesized person image should be of high-quality with few artifacts. 

**==> picture [315 x 117] intentionally omitted <==**

**----- Start of picture text -----**<br>
Target Person    In-shop Garment            VITON-HD              HR-VTON                 GP-VTON               LaDI-VTON              DCI-VTON               GarDiff<br> (a)                             (b)                            (c)                           (d)                            (e)                              (f)<br>**----- End of picture text -----**<br>


**Fig. 1:** Existing GAN-based VTON methods (e.g., VITON-HD [6], HR-VTON [21] and GP-VTON [36]) and Diffusion-based VTON techniques (e.g., LaDI-VTON [25] and DCI-VTON [12]) often fail to perfectly retain every appearance/texture detail of the given garment (e.g., the complex patterns or texts). Instead, our GarDiff exploits garment-focused diffusion process to preserve most of fine-grained details of the given garment, pursuing more controllable person image generation. 

To tackle VTON task, prior works [1, 6, 8, 9, 14, 24, 34, 34, 38, 39] commonly hinge on an explicit warping process to directly deform the appearance of in-shop garment conditioned on the pose of target person. However, this way is often prone to suffer from distortions and artifacts over warped garments, due to the misalignment between warped garment and target person’s body. To alleviate this misalignment issue, several subsequent works [10, 15, 21, 22, 36] further upgrade warping process with an additional generative process. They capitalize on the typical Generative Adversarial Networks [11] to synthesize the final person image based on conditions like warped garment and human body. While effective, the synthesized person images still tend to be unsatisfactory in challenging cases (see the unrealistic artifacts in Figure 1 (a), (b) and (c)). This might be attributed to the limited capacity of the underlying generative model (GAN) to synthesize person image with complex patterned garment and variable human pose. 

Recently, Diffusion models [4,5,17,27,29,31,33,41,43] emerge as a new trend of generative modeling in numerous image synthesis tasks, demonstrating better scalability and easier/stable training than GAN-based solutions. Motivated by this, recent advances [12, 25] have been dedicated to remould pre-trained Latent Diffusion Model [29] of text-to-image synthesis by leveraging warped garment as additional condition during diffusion process. Although promising results are attained, these diffusion-based VTON approaches (e.g., LaDI-VTON [25] and DCI-VTON [12]) still fail to completely retain every detail of in-shop garment, especially for high-frequency texture details of complex patterns/texts (see Fig- 

Improving Virtual Try-On with Garment-focused Diffusion Models 

3 

ure 1 (d) and (e)). We speculate that degenerated results might be caused by holistic diffusion process over latent space where compressed latent code is not capable of memorizing every intricate garment detail and providing sufficient guidance for VTON task. 

In an effort to mitigate this problem, our work shapes a new way to upgrade the latent diffusion model with amplified guidance of both visual appearance and high-frequency texture details for VTON task. Technically, we propose a novel Garment-focused Diffusion model (GarDiff) to progressively excavate more prior knowledge about fine-grained garment details. Such prior knowledge acts as amplified garment-focused guidance to improve virtual try-on results. Specifically, CLIP [28] and Variational Auto-encoder (VAE) [20] are employed to encode the reference garment into appearance priors, which is regarded as additional conditions to guide diffusion process. To effectively leverage these priors, a new garment-focused adapter is introduced to the UNet of latent diffusion model. This design triggers the local fine-grained appearance alignment between the synthetic person image and reference garment & human pose. Moreover, a novel appearance loss is defined on the warped reference garment to achieve the guidance of high-frequency prior, which is utilized to supervise the synthesis of high-frequency details in synthesized garment, pursuing better preservation of high-frequency texture details in VTON. Eventually, our GarDiff faithfully produces person images with better-aligned garment details (see Figure 1 (f)). 

The main contribution of this work is the proposal of garment-focused diffusion model that facilitates virtual try-on tasks. This also leads to the elegant view of how a diffusion model should be designed for excavating the garmentfocused prior knowledge (e.g., visual appearance and high-frequency texture details) tailored to VTON, and how to improve diffusion process with these amplified garment-focused guidance. Through an extensive set of experiments on VITON-HD and DressCode datasets, our GarDiff consistently achieves competitive performances against state-of-the-art VTON methods. 

## **2 Related Work** 

**GAN-based Virtual Try-on.** To tackle VTON task, prior works [1,6,8–10,14, 15, 21, 22, 24, 34, 34, 36, 39] capitalize on the GAN to synthesize the final person image based on the conditions like warped garment and human body. VITON [14] is a pioneering work that employs a refinement network to composite warped garments generated through TPS [3] with the target person. CP-VTON [24] introduces an upgraded learnable TPS transformation for achieving more robust alignment between the target person and the garment. VITON-HD [6] designed an alignment-aware segment generator to fill the misaligned regions with the garment texture through multi-scale refinement. HR-VTON [21] proposes a novel try-on condition generator that unifies the warping and segmentation generation modules for handling the misalignment and occlusion. GP-VTON [36] presents an advanced LFGP warping module for creating deformed garments, which is optimized with a new DGT training strategy. 

4 Siqi Wan et al. 

**==> picture [307 x 181] intentionally omitted <==**

**----- Start of picture text -----**<br>
high-freq<br>CLIP Vision Embeddings appearance<br>CLIP Vision Encoder MLP f clip<br>Edgegt Edgepred<br>I c<br>spatial<br>VAE Embeddings<br>VAE  f vae<br>Encoder MLP<br>I w<br>...<br>x t x w x a<br>UNet ^ I<br>High-pass<br>Self Self<br>Attention GF Adapter Attention GF Adapter<br>Filters<br>**----- End of picture text -----**<br>


**Fig. 2:** An overview of our GarDiff. The cross-attention layer is substituted with the garment-focused vision adapter in each Transformer block. First, we extract the CLIP visual embeddings **f** _clip_ and VAE embeddings **f** _vae_ of the target garment **I** _c_ and warped garment **I** _w_ , respectively. Then the two embeddings are fed into the garment-focused adapter as keys and values via a decoupled cross-attention to guide the diffusion process for pursuing local fine-grained alignment with the appearance of target garment. Meanwhile, we employ a novel appearance loss _Lappearance_ comprised of spatial perceptual loss _Lspatial_ and high-frequency promoted loss _Lhigh_ - _freq_ over the generated garment to enhance the proficiency of GarDiff in generating high-frequency details. 

**Diffusion-based Virtual Try-on.** Recently, diffusion models [13, 17, 31] start to dominate in natural image generation due to its superior ability in generating high-fidelity realistic images compared to GAN-based models. Inspired by this, a series of diffusion-based virtual try-on models begin to emerge. TryOnDiffusion [42] unifies two UNets to preserve garment details and warp the garment in a single network. LaDI-VTON [25] introduces the textual inversion component that maps visual features of reference garment to CLIP token embedding space as condition of diffusion model. DCI-VTON [12] further use warping network to warp reference garment, which is fed into diffusion model as an additional guidance. Although promising results are attained, these diffusion-based VTON approaches still fail to completely retain every detail of the reference garment. 

## **3 METHOD** 

In this work, we propose a Garment-focused Diffusion model (GarDiff), an upgraded latent diffusion model with amplified guidance of both visual appearance and high-frequency texture details for VTON task. In this section, we will first 

Improving Virtual Try-On with Garment-focused Diffusion Models 

5 

**==> picture [302 x 93] intentionally omitted <==**

**----- Start of picture text -----**<br>
Garment Area Other Areas<br>VAE  V<br>Embeddings<br>K<br>I w<br>CLIP Vision  Self<br>Embeddings Attention Q<br>Feature<br>I c Mattn<br>**----- End of picture text -----**<br>


**Fig. 3:** Implementation details of our garment-focused adapter. For the given target garment **I** _c_ and warped garment **I** _w_ , the CLIP visual embeddings **f** _clip_ and VAE embeddings **f** _vae_ are extracted and fed into the garment-focused adapter as the keys and values through a decoupled cross-attention. **M** _attn_ is used to suppress the weights unrelated to garment area in the attention map for generating garment-focused features. 

provide a concise overview of our GarDiff, followed by the details of the proposed garment-focused vision adapter and appearance loss. 

## **3.1 Overview** 

Generally, given a person image **I** _p ∈_ R _[H][×][W][ ×]_[3] and in-shop garment **I** _c ∈_ R _[H][′][×][W][ ′][×]_[3] , our GarDiff is optimized to synthesize a high-quality realistic image **I** _∈_ R _[H][×][W][ ×]_[3] , where the person wears the in-shop garment **I** _c_ . To effectively leverage the appearance guidance of the given garment for high-fidelity person image generation, the original cross-attention layers in the UNet of diffusion model are substituted with our proposed garment-focused vision adapter modules. An overview of our GarDiff is illustrated in Figure 2. 

In the forward diffusion process, similar to vanilla diffusion model, we gradually add noise to the target image **I** according to the Markov chain. Specifically, we first utilize the VAE encoder _E_ ( _·_ ) to map the target image **I** to the latent space: **x** 0 = _E_ ( **I** ). Then we add noise _ϵ_ to **x** 0 at an arbitrary timestep _t ∈_ [1 _,_ 1000] as follows: 

**==> picture [227 x 12] intentionally omitted <==**

where _αt_ =[�] _[t] s_ =1[(1] _[−][β][s]_[)][,] _[ ϵ][ ∼N]_[(0] _[,]_[ 1)][, and] _[ β][s]_[ is the pre-defined variance schedule] at timestep _s_ . 

In the reverse diffusion process, we first warp the in-shop garment **I** _c_ similar to [36] to achieve the maximum conformity of the target garment with the body’s posture, obtaining the warped garment **I** _w ∈_ R _[H][′][×][W][ ′][×]_[3] and warped mask **m** _w ∈ {_ 0 _,_ 1 _}[H][×][W]_ . The warped mask **m** _w_ indicates the area of the warped garment. To preserve the area unrelated to the garment, a garment-agnostic image **I** _a_ with the region intended for garment placement fully masked from the person image **I** _p_ is also extracted. Then, the noisy image latent **x** _t_ , the latent warped garment 

6 Siqi Wan et al. 

map **x** _w_ = _E_ ( **I** _w_ ) and the latent agnostic map **x** _a_ = _E_ ( **I** _a_ ) are concatenated along the channel dimension, leading to the input of the UNet _ϵθ_ [30] of the diffusion model: 

**==> picture [225 x 11] intentionally omitted <==**

During denoising, the UNet is trained to predict the added noise _ϵ_ conditioned on the target garment **I** _c_ and warped garment **I** _w_ . The objective function is formed as the mean-squared loss: 

**==> picture [238 x 13] intentionally omitted <==**

## **3.2 Garment-Focused Adapter** 

To retain the appearance details of the given garment, previous works [12,25,42] merely steer the diffusion process with the garment appearance captured by CLIP vision encoder. However, it is difficult for CLIP vision encoder to perceive the fine-grained details in the target garment, since CLIP is optimized for imagetext alignment at a coarse level. In our experiments, we found that VAE, serving as the reconstruction module for stable diffusion, exhibits much stronger capabilities in preserving texture details in images than CLIP. Therefore, it would be beneficial to involve VAE as additional appearance prior into stable diffusion. 

Technically, we replace the cross-attention layer with a vision adapter in each Transformer block. The vision adapter takes two appearance priors are inputs: 1) the VAE embeddings of the warpped rendition **I** _w_ of the target garment **I** _c_ for recovering the complex patterns in the synthesized image, 2) the CLIP visual embeddings of **I** _c_ for generating the holistic structure regardless of the imperfection of the warpped result **I** _w_ . Specifically, given the target garment **I** _c_ and the warped garment **I** _w_ , the CLIP visual embeddings **f** _clip_ and VAE embeddings **f** _vae_ are calculated as: 

**==> picture [238 x 28] intentionally omitted <==**

where **CLIP** _v_ , **MLP** _[clip]_ , **MLP** _[vae]_ are the CLIP vision encoder, multi-layer perceptrons for CLIP visual embeddings and VAE embeddings, respectively. Considering the two embeddings focus on different granularity levels of the target garment, a decoupled cross-attention mechanism is devised in the vision adapter to separate the cross-attention layers for the CLIP visual embeddings and VAE embeddings. Given the features **f** _q_ from the last self-attention layer, the vision adapter operates as follows: 

**==> picture [317 x 57] intentionally omitted <==**

Improving Virtual Try-On with Garment-focused Diffusion Models 

7 

where **W** _[q]_ , **W** _∗[k]_[,] **[W]** _∗[v]_[are][the][trainable][projection][matrices][in][the][decoupled] cross-attention of the vision adapter for queries, keys and values, respectively. 

It is relatively easier for the model to achieve favorable results on the regions devoid of garments, such as the torso skin, since those regions usually exhibit simpler patterns or textures. Therefore, improving virtual try-on results hinges critically upon the restoration of the fine-grained details in the given garment. In light of this, we propose to amplify the garment-focused guidance in the diffusion process. Specifically, the vision adapter is further upgraded with a novel garment-focused attention, leading to a garment-focused (GF) adapter that aims to pursue local fine-grained alignment with the visual appearance of the reference garment and human pose. As illustrated in Figure 3, the warped mask **m** _w_ is downsampled to an attention mask **M** _attn_ that matches the resolution of the corresponding attention layer in the GF adapter, which is leveraged to suppress the attention weights unrelated to garment area in the attention map. Hence, Equation (5) can be reformulated for the garment-focused adapter as follows: 

**==> picture [283 x 54] intentionally omitted <==**

## **3.3 Appearance Loss** 

Generally, diffusion model is merely optimized with the mean-squared loss defined in Equation (3), which treats all the regions of the synthesized image equally without emphasizing the texture details in the garment area, failing to generate the accurate garment patterns. As complicated details in the inshop garment typically manifest as high-frequency components (i.e., edges), a novel appearance loss is proposed to enforce the synthesized garment to be geometrically consistent with the high-frequency details of the reference garment, achieving improved fidelity and fine-grained textures. The appearance loss, as a composite adaptation loss, can be decomposed into two components: a spatial perceptual loss _Lspatial_ and a high-frequency promoted loss _Lhigh_ - _freq_ . 

Specifically, we estimate the latent ˆ **x** 0 given the noisy one **x** _t_ and the predicted noise _ϵt_ from the UNet at timestep _t_ by reversing the process in Equation (1): 

**==> picture [221 x 26] intentionally omitted <==**

The latent **x** ˆ0 is further converted back to the pixel space by the VAE decoder _D_ , leading to the predicted image[ˆ] **I** . 

**Spatial Perceptual Loss.** Inspired by the Deep Image Structure and Texture Similarity (DISTS) metric [7], the spatial perceptual loss is designed to capture both the structural and textural disparities between the predicted and groundtruth images in a perceptual feature space beyond pixels: 

**==> picture [254 x 13] intentionally omitted <==**

8 Siqi Wan et al. 

Note that we use the warped mask **m** _w_ to emphasize the garment area. **High-Frequency Promoted Loss.** To substantively enhance the model’s proficiency in generating high-frequency details, we employ edge detection to extract high-frequency information. Technically, the horizontal and vertical Sobel kernels are adopted as the high-pass filters to extract the edge maps[ˆ] **I** _h_ / **I** _h_ from the predicted/target image, respectively. Formally, the high-frequency promoted loss is defined as: 

**==> picture [255 x 13] intentionally omitted <==**

Finally, the UNet is optimized by the following improved objective function: 

**==> picture [250 x 26] intentionally omitted <==**

where _λ_ is the hyper-parameter used to balance the mean-squared loss and the proposed appearance loss. 

## **4 Experiments** 

## **4.1 Experimental Settings** 

**Dadasets.** We empirically verify and analyze the effectiveness of our GarDiff on two popular virtual try-on datasets, VITON-HD [6] and DressCode [26]. The VITON-HD dataset [6] comprises 13,679 frontal-view woman and upper garment image pairs. In line with the general practices of the previous works [12, 25], the dataset is divided into two disjoint subsets: a training set with 11,647 pairs and a test set with 2,032 pairs. The DressCode dataset consists of 53,795 image pairs, which are categorized into three macro-categories: 15,366 for upper-body clothes, 8,951 pairs lower-body clothes and 29,478 for dresses. Following the original splits, 1,800 image pairs from each category are reserved for test while the remaining image pairs are utilized for training. The experiments on DressCode and VITON-HD are conducted at the resolution of 512 _×_ 384. 

**Evaluation Metrics.** We evaluate our GarDiff in both paired and unpaired settings following the virtual try-on literature. In the paired setting, the input garment corresponds to the one originally depicted in the person image. Following the standard evaluation setup, Structural Similarity (SSIM) [35] and Learned Perceptual Image Patch Similarity (LPIPS) [40] are adopted to measure the similarity between the generated image and the ground-truth one. Additionally, the Fréchet Inception Distance(FID) [16] and Kernel Inception Distance(KID) [2] are employed to measure the quality and realism of the generated images. In the unpaired setting, where the garment of the person image is changed to a different one and the ground truth is unavailable, we report the performances of GarDiff in terms of FID and KID. 

**Implementation Details.** Our GarDiff is initialized from the pre-trained Stable Diffusion 2.1 and finetuned on the virtual try-on datasets. AdamW [23] 

Improving Virtual Try-On with Garment-focused Diffusion Models 

9 

**Table 1:** Quantitative performance comparisons on VITON-HD dataset. **FIDp** / **KIDp** stands for the **FID** / **KID** score in paired setting, while **FIDu** / **KIDu** stands for the **FID** / **KID** score in unpaired setting. Note that the KID score is multiplied by 100. 

|**Model**|**SSIM** _↑_**LPIPS**|**SSIM** _↑_**LPIPS**|_↓_**FIDp**|_↓_**KIDp** _↓_**FIDu**|_↓_**KIDp** _↓_**FIDu**|_↓_**KIDu** _↓_|
|---|---|---|---|---|---|---|
|VITON-HD [6]|0.843|0.076|-|-|11.64|0.300|
|PF-AFN [10]|0.858|0.082|-|-|11.30|0.283|
|HR-VTON [21]|0.878|0.061|-|-|9.90|0.188|
|GP-VTON [36]|0.894|0.080|-|-|9.20|-|
|Paint-by-Example [37]|0.843|0.087|-|-|10.15|0.204|
|LaDI-VTON [25]|0.879|0.059|6.66|0.108|9.41|0.167|
|DCI-VTON [12]|0.896|0.043|-|-|8.09|0.028|
|**GarDif**|**0.912**|**0.036**|**6.02**|**0.019**|**7.89**|**0.027**|



( _β_ 1 = 0 _._ 9, _β_ 2 = 0 _._ 999) is employed to optimize the model for 200k steps. The learning rate is set to 0.00005 with linear warmup of 500 iterations. The hyperparameter _λ_ in Equation (10) and the weight decay are set to 0.001 and 0.01, respectively. OpenCLIP ViT-H/14 [19] is utilized to extract the CLIP visual embeddings of the target garment. To enable classifier-free guidance [18], the embeddings of the garment are randomly dropped with a probability of 0.05. We train GarDiff on 4 NVidia RTX4090 GPUs for about four days and the model size is 5.15 GB. During inference, the image is progressively generated over 100 steps with a DDIM [32] sampler, and the scale of classifier-free guidance is set to 7.5 by default. 

## **4.2 Quantitative Results** 

**VITON-HD.** We compare our GarDiff with a series of state-of-the-art virtual try-on methods including GAN-based methods (VITON-HD [6], PF-AFN [10], HR-VTON [21], GP-VTON [36]) and Diffusion-based methods (DCI-VTON [12], Paint-by-Example [37] and LaDI-VTON [25]). Table 1 summarizes the performance comparisons on the VITON-HD dataset. It can be easily observed that our proposed GarDiff consistently demonstrates superior performances compared to the other methods across all the evaluation metrics. In particular, GP-VITON improves VITON-HD by introducing the try-on condition generator that serves as a unified module in warping and segmentation generation stages. Paint-by-Example further boosts the performances by framing the VTON task as exemplar-based image inpainting and filling the target region of source image with the garment in the reference image. By exploiting the textual inversion to maintain the details of the in-shop garment, LaDI-VTON exhibits better performance than Paint-by-Example. Moreover, DCI-VTON leverages a warping network to guide the image generation of diffusion model, leading to clear performance boosts. However, these existing approaches merely steer the generation process at a coarse level, and thus fail to perfectly retain the details of the given garment. On the contrary, our GarDiff facilitates the preservation of the garment’s appearance by excavating the garment-focused prior knowledge and strengthening diffusion process with these amplified garment-focused guidance, 

10 Siqi Wan et al. 

**Table 2:** Quantitative performance comparisons on DressCode dataset. **FIDp** / **KIDp** stands for the **FID** / **KID** score in paired setting, while **FIDu** / **KIDu** stands for the **FID** / **KID** score in unpaired setting. Note that the KID score is multiplied by 100. 

||**Dataset**<br>**Model**<br>|**DC_upper-body**<br>**SSIM** _↑_**LPIPS** _↓_**FIDp** _↓_**KIDp** _↓_**FIDu** _↓_**KIDu** _↓_|
|---|---|---|
|CP-VTON+ [24]<br>PF-AFN [10]<br>PSAD [26]<br>GP-VTON [36]||0.918<br>0.078<br>19.70<br>1.16<br>22.18<br>12.09<br>0.918<br>-<br>-<br>-<br>14.32<br>-<br>0.938<br>0.049<br>13.87<br>0.640<br>17.51<br>7.15<br>0.947<br>0.036<br>-<br>-<br>11.98<br>-|
|LaDI-VTON [25]<br>DCI-VTON [12]<br>**GarDif**||0.928<br>0.049<br>9.53<br>0.198<br>13.26<br>2.67<br>-<br>**0.030**<br>-<br>-<br>**10.82**<br>-<br>**0.952**<br>**0.030**<br>**8.69**<br>**0.144**<br>11.32<br>**2.34**|
||**Dataset**<br>**Model**<br>|**DC_lower-body**<br>**SSIM** _↑_**LPIPS** _↓_**FIDp** _↓_**KIDp** _↓_**FIDu** _↓_**KIDu** _↓_|
|CP-VTON+ [24]<br>PF-AFN [10]<br>PSAD [26]<br>GP-VTON [36]||0.913<br>0.083<br>-<br>-<br>18.85<br>10.24<br>0.907<br>-<br>-<br>-<br>18.32<br>-<br>0.932<br>0.051<br>13.14<br>0.559<br>19.68<br>8.90<br>**0.941**<br>0.042<br>-<br>-<br>16.07<br>-|
|LaDI-VTON [25]<br>DCI-VTON [12]<br>**GarDif**||0.922<br>0.051<br>8.52<br>0.104<br>14.80<br>3.13<br>-<br>**0.035**<br>-<br>-<br>12.34<br>-<br>0.939<br>**0.035**<br>**8.01**<br>**0.090**<br>**12.29**<br>**2.88**|
||**Dataset**<br>**Model**<br>|**DC_dresses**<br>**SSIM** _↑_**LPIPS** _↓_**FIDp** _↓_**KIDp** _↓_**FIDu** _↓_**KIDu** _↓_|
|CP-VTON+ [24]<br>PF-AFN [10]<br>PSAD [26]<br>GP-VTON [36]||0.863<br>0.123<br>18.75<br>1.11<br>21.83<br>12.31<br>0.869<br>-<br>-<br>-<br>13.59<br>-<br>0.885<br>0.074<br>12.38<br>0.468<br>17.07<br>6.66<br>0.886<br>0.072<br>-<br>-<br>12.26<br>-|
|LaDI-VTON [25]<br>DCI-VTON [12]<br>**GarDif**||0.868<br>0.089<br>9.07<br>0.112<br>13.40<br>2.50<br>-<br>0.068<br>-<br>-<br>12.25<br>-<br>**0.891**<br>**0.065**<br>**8.77**<br>**0.096**<br>**12.05**<br>**2.09**|



achieving much better results in VTON task. Specifically, our GarDiff achieves 0.912 on SSIM score and makes a relative improvement of 1.78% against the best competitor DCI-VTON. 

**DressCode.** Table 2 summarizes the performance comparisons on the DressCode dataset. Similar to the observations on VITON-HD, our proposed GarDiff surpasses the performances of the other competing methods across all the three macro-categories, which again evinces the pivotal merit of the garment-focused appearance guidance for preserving fine-grained garment attributes in the generated images. Particularly, our GarDiff leads to the relative improvements over LaDI-VTON by 2.58% on SSIM for upper-body settings. 

## **4.3 Qualitative Results** 

Figure 4 showcases several virtual try-on results of different methods, coupled with the input person image and in-shop garment. As evidenced by the presented exemplar results, all the approaches demonstrate a certain degree of proficiency in resembling the appearance and texture details of the input in-shop 

Improving Virtual Try-On with Garment-focused Diffusion Models 

11 

**==> picture [315 x 6] intentionally omitted <==**

**----- Start of picture text -----**<br>
Target Person    In-shop Garment            VITON-HD              HR-VTON                 GP-VTON               LaDI-VTON              DCI-VTON               GarDiff<br>**----- End of picture text -----**<br>


**==> picture [42 x 56] intentionally omitted <==**

**==> picture [43 x 56] intentionally omitted <==**

**==> picture [42 x 56] intentionally omitted <==**

**==> picture [43 x 56] intentionally omitted <==**

**==> picture [42 x 57] intentionally omitted <==**

**==> picture [43 x 57] intentionally omitted <==**

**==> picture [42 x 56] intentionally omitted <==**

**==> picture [43 x 56] intentionally omitted <==**

**==> picture [42 x 56] intentionally omitted <==**

**==> picture [43 x 56] intentionally omitted <==**

**==> picture [42 x 56] intentionally omitted <==**

**==> picture [42 x 56] intentionally omitted <==**

**==> picture [43 x 56] intentionally omitted <==**

**==> picture [42 x 56] intentionally omitted <==**

**==> picture [43 x 56] intentionally omitted <==**

**==> picture [43 x 56] intentionally omitted <==**

**==> picture [42 x 56] intentionally omitted <==**

**==> picture [42 x 56] intentionally omitted <==**

**==> picture [43 x 56] intentionally omitted <==**

**==> picture [42 x 56] intentionally omitted <==**

**==> picture [43 x 56] intentionally omitted <==**

**==> picture [43 x 56] intentionally omitted <==**

**==> picture [42 x 57] intentionally omitted <==**

**==> picture [42 x 57] intentionally omitted <==**

**==> picture [43 x 57] intentionally omitted <==**

**==> picture [42 x 57] intentionally omitted <==**

**==> picture [43 x 57] intentionally omitted <==**

**==> picture [43 x 57] intentionally omitted <==**

**==> picture [42 x 56] intentionally omitted <==**

**==> picture [42 x 56] intentionally omitted <==**

**==> picture [43 x 56] intentionally omitted <==**

**==> picture [42 x 56] intentionally omitted <==**

**==> picture [43 x 56] intentionally omitted <==**

**==> picture [43 x 56] intentionally omitted <==**

**==> picture [42 x 56] intentionally omitted <==**

**==> picture [42 x 56] intentionally omitted <==**

**==> picture [43 x 56] intentionally omitted <==**

**==> picture [42 x 56] intentionally omitted <==**

**==> picture [43 x 56] intentionally omitted <==**

**==> picture [43 x 56] intentionally omitted <==**

**Fig. 4:** Examples generated by VITON-HD, HR-VTON, GP-VTON, LaDI-VTON, DCI-VTON and our GarDiff. 

garments. Specifically, Diffusion-based methods (LaDI-VTON, DCI-VTON and our GarDiff) consistently outperform the GAN-based approaches (VITON-HD, HR-VTON and GP-VTON) by leveraging the strong capability of diffusion models in generating high-fidelity images. Despite both DCI-VTON and our GarDiff capitalizing on diffusion models, the former approach achieves inferior results to the latter. The underlying rationale lies in the fact that DCI-VTON guides the diffusion process with CLIP visual embeddings of the garment only, while our GarDiff additionally leverages the appearance prior from the VAE encoder through a vision adapter to better preserve the fine-grained details. Moreover, GarDiff is further upgraded with garment-focused attention machanism and optimized with a new appearance loss to steer the diffusion process with amplified garment-focused guidance, yielding images with enhanced local alignment to the appearance of the garment. For example, our GarDiff better restores the text “wrangler” on the garment than DCI-VTON in the fourth row. 

Furthermore, we conduct human study to compare our GarDiff against five strong baselines (HR-VTON, VITON-HD, GP-VTON, Ladi-VTON, DCI-VTON) 

12 Siqi Wan et al. 

**==> picture [265 x 69] intentionally omitted <==**

**----- Start of picture text -----**<br>
26.1 73.9 12.1 87.9 VITON-HD<br>30.8 69.2 23.5 76.5 HR-VTON<br>GP-VTON<br>41.5 58.5 44.6 55.4<br>LaDI-VTON<br>37.2 62.8 31.8 68.2<br>DCI-VTON<br>43.4 56.6 39.7 60.3 GarDiff<br>Garment Detail Human Pose<br>**----- End of picture text -----**<br>


**Fig. 5:** User study on 100 garment-person pairs randomly sampled from VITON-HD. 

**Table 3:** Ablation study of our proposed GarDiff on VITON-HD dataset. **Base** : based model; **GFA** : garment-focused adapter; **AL** : appearance loss. 

|**Model**|**SSIM** _↑_**LPIPS**|**SSIM** _↑_**LPIPS**|_↓_**FIDp**|_↓_**KIDp** _↓_**FIDu**|_↓_**KIDp** _↓_**FIDu**|_↓_**KIDu** _↓_|
|---|---|---|---|---|---|---|
|**Base**|0.809|0.117|10.03|0.190|10.96|0.473|
|**Base+AL**|0.829|0.090|8.72|0.103|10.22|0.300|
|**Base+GFA**|0.898|0.052|6.61|0.047|8.21|0.069|
|**Base+GFA+AL**|**0.912**|**0.036**|**6.02**|**0.019**|**7.89**|**0.027**|



over 100 randomly sampled garment-person pairs in VITON-HD (unpaired setting). 10 evaluators from diverse education background are invited to rank the best VTON result between our GarDiff and the five competing methods based on two criteria: (1) garment detail preservation, (2) human pose alignment. Figure 5 shows the percentages of top-1 ranking for each method, and our GarDiff achieves the best results in both evaluated dimensions. 

## **4.4 Analysis and Discussions** 

**Ablation Study on GarDiff.** We conduct an ablation study to investigate how each design in our GarDiff influences the overall performances on the VITON-HD dataset. Table 3 details the performance comparisons among different ablated runs of our GarDiff. We start from the stable diffusion inpainting as our base model ( **Base** ), which replaces the original CLIP text encoder with the CLIP vision encoder. **Base+GFA** further boosts **Base** ) by concurrently leveraging the CLIP visual embeddings and VAE embeddings to guide the diffusion process, which verifies the merit of applying the appearance prior from VAE. It is not surprising that **Base+AL** outperforms **Base** by supervising the training with the novel appearance loss that facilitates the preservation of garment detail. Finally, our full GarDiff ( **Base+GFA+AL** ) achieves the best performances through the synergetic integration of the two proposed components. 

To illustrate the impact of these components more intuitively, we visualize these ablated runs in Figure 6. Compared with other ablated runs, our final model **Base+GFA+AL** (i.e., GarDiff) can preserve most of the fine-grained details of the given garments. Take the first case as an example, compared to the base model **Base** , **Base+GFA** effectively retains texture details of the letters. 

Improving Virtual Try-On with Garment-focused Diffusion Models 13 

**==> picture [308 x 7] intentionally omitted <==**

**----- Start of picture text -----**<br>
Target Person    In-shop Garment           Base Base+AL Base+GFA Base+GFA+AL<br>**----- End of picture text -----**<br>


**==> picture [53 x 71] intentionally omitted <==**

**==> picture [53 x 71] intentionally omitted <==**

**==> picture [53 x 71] intentionally omitted <==**

**==> picture [53 x 71] intentionally omitted <==**

**==> picture [53 x 71] intentionally omitted <==**

**==> picture [54 x 71] intentionally omitted <==**

**==> picture [53 x 71] intentionally omitted <==**

**==> picture [54 x 71] intentionally omitted <==**

**==> picture [53 x 71] intentionally omitted <==**

**==> picture [54 x 71] intentionally omitted <==**

**==> picture [53 x 71] intentionally omitted <==**

**==> picture [54 x 71] intentionally omitted <==**

**Fig. 6:** Ablation study on the pivotal components of GarDiff. 

Similarly, when the garment-focused vision adapter is additionally integrated into the **Base** model, the generated results are further improved by **Base+AL** . 

**Effect of Unwarpped Garment.** In contrast to the Diffusion-based DCIVTON that merely captializes on the warpped garment derived from an warping networks for VTON, our GarDiff additionally incorporates the unwarpped garment to faithfully retain the appearance and shape details of the reference garment. In this way, high-quality VTON results can be achieved even when the warpped garments predicted from the warping networks are defective. Some exampels are shown in Figure 7(a). Regarding the long-sleeve T-shirt on th first row, the warpped garment fails to accurately restore the shape of the input garment with one of the sleeves missing. As a result, less favorable outcomes are obtained by solely leveraging the erroneously warped garment. Our GarDiff, employing both the appearance priors over the warpped and unwarpped garments, generates high-fidelity images. 

**Preservation of Fine-grained Details.** With the assistance of the proposed garment-focused attention mechanism and appearance loss, our GarDiff is capable of accurately aligning the visual appearance of the garment in the generated samples with the reference one. Figure 7(b) showcases the images synthesized by Diffusion-based competing methods (LaDI-VTON and DCI-VTON) and our proposed GarDiff, involving the cases with fine-grained details. Compared to LaDIVTON and DCI-VTON which fail to achieve satisfactory results, our GarDiff successfully restores the small letters in the garment on the first row. 

14 Siqi Wan et al. 

**==> picture [339 x 167] intentionally omitted <==**

**----- Start of picture text -----**<br>
    In-shop Garment    Warped Garment w/ Warped Garment   Unwarped Garment w/ Warped and         In-shop Garment     Target Person   LaDI-VTON      DCI-VTON            GarDiff<br>   (a)    (b)<br>**----- End of picture text -----**<br>


**Fig. 7:** (a) Examples generated by our GarDiff with or without unwarpped garment. (b) Comparisons between Diffusion-based baselines (LaDI-VTON and DCI-VTON) and our GarDiff regarding the preservation of fine-grained details. 

## **5 Conclusion** 

In this work, we have presented the Garment-focused Diffusion model (GarDiff) that is capable of preserving the fine-grained details of the target garment in the virtual try-on task. Specifically, GarDiff remoulds the pre-trained latent diffusion model with appearance priors from the CLIP vision encoder and the VAE encoder for the reference garment and then integrates these priors into UNet through a garment-focused vision adapter. In this way, the diffusion process is effectively strengthened with the amplified appearance guidance from the given garment. A novel appearance loss is further devised to enforce the synthesized garment to be consistent with the high-frequency details and the geometric shape of target garment. Extensive experiments conducted on VITON-HD and DressCode datasets demonstrate the superiority of our GarDiff. More remarkably, we achieve new state-of-the-art performances on the two virtual try-on datasets. 

**Broader Impact.** Recent advances in generative modeling offer new possibilities for creating and manipulating digital media but also pose risks of generating deceptive content. Our proposed GarDiff might be nefariously used to “undress” individuals by substituting their original attire with undergarments for pornographic applications, and we emphatically denounce any such activities. 

Improving Virtual Try-On with Garment-focused Diffusion Models 15 

## **References** 

1. Bai, S., Zhou, H., Li, Z., Zhou, C., Yang, H.: Single stage virtual try-on via deformable attention flows. In: ECCV (2022) 2, 3 

2. Bińkowski, M., Sutherland, D.J., Arbel, M., Gretton, A.: Demystifying mmd gans. In: ICLR (2018) 8 

3. Bookstein, F.L.: Principal warps: Thin-plate splines and the decomposition of deformations. IEEE TPAMI **11** (6), 567–585 (1989) 3 

4. Chen, J., Pan, Y., Yao, T., Mei, T.: Controlstyle: Text-driven stylized image generation using diffusion priors. In: ACM MM (2023) 2 

5. Chen, Y., Pan, Y., Li, Y., Yao, T., Mei, T.: Control3d: Towards controllable textto-3d generation. In: ACM MM (2023) 2 

6. Choi, S., Park, S., Lee, M., Choo, J.: Viton-hd: High-resolution virtual try-on via misalignment-aware normalization. In: CVPR (2021) 2, 3, 8, 9 

7. Ding, K., Ma, K., Wang, S., Simoncelli, E.P.: Image quality assessment: Unifying structure and texture similarity. IEEE TPAMI **44** (5), 2567–2581 (2020) 7 

8. Dong, H., Liang, X., Shen, X., Wu, B., Chen, B.C., Yin, J.: Fw-gan: Flow-navigated warping gan for video virtual try-on. In: ICCV (2019) 2, 3 

9. Fenocchi, E., Morelli, D., Cornia, M., Baraldi, L., Cesari, F., Cucchiara, R.: Dualbranch collaborative transformer for virtual try-on. In: CVPR Workshops (2022) 2, 3 

10. Ge, Y., Song, Y., Zhang, R., Ge, C., Liu, W., Luo, P.: Parser-free virtual try-on via distilling appearance flows. In: CVPR (2021) 2, 3, 9, 10 

11. Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., Bengio, Y.: Generative adversarial nets. In: NeurIPS (2014) 2 

12. Gou, J., Sun, S., Zhang, J., Si, J., Qian, C., Zhang, L.: Taming the power of diffusion models for high-quality virtual try-on with appearance flow. In: ACM MM (2023) 2, 4, 6, 8, 9, 10 

13. Gu, S., Chen, D., Bao, J., Wen, F., Zhang, B., Chen, D., Yuan, L., Guo, B.: Vector quantized diffusion model for text-to-image synthesis. In: CVPR (2022) 4 

14. Han, X., Wu, Z., Wu, Z., Yu, R., Davis, L.S.: Viton: An image-based virtual try-on network. In: CVPR (2018) 2, 3 

15. He, S., Song, Y.Z., Xiang, T.: Style-based global appearance flow for virtual try-on. In: CVPR (2022) 2, 3 

16. Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., Hochreiter, S.: Gans trained by a two time-scale update rule converge to a local nash equilibrium. In: NeurIPS (2017) 8 

17. Ho, J., Jain, A., Abbeel, P.: Denoising diffusion probabilistic models. In: NeurIPS (2020) 2, 4 

18. Ho, J., Salimans, T.: Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598 (2022) 9 

19. Ilharco, G., Wortsman, M., Wightman, R., Gordon, C., Carlini, N., Taori, R., Dave, A., Shankar, V., Namkoong, H., Miller, J., et al.: Openclip. Zenodo **4** , 5 (2021) 9 

20. Kingma, D.P., Welling, M.: Auto-encoding variational bayes. In: ICLR (2014) 3 

21. Lee, S., Gu, G., Park, S., Choi, S., Choo, J.: High-resolution virtual try-on with misalignment and occlusion-handled conditions. In: ECCV (2022) 2, 3, 9 

22. Li, K., Chong, M.J., Zhang, J., Liu, J.: Toward accurate and realistic outfits visualization with attention to details. In: CVPR (2021) 2, 3 

23. Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. In: ICLR (2019) 8 

16 Siqi Wan et al. 

24. Minar, M.R., Tuan, T.T., Ahn, H., Rosin, P., Lai, Y.K.: Cp-vton+: Clothing shape and texture preserving image-based virtual try-on. In: CVPR Workshops (2020) 2, 3, 10 

25. Morelli, D., Baldrati, A., Cartella, G., Cornia, M., Bertini, M., Cucchiara, R.: Ladi-vton: Latent diffusion textual-inversion enhanced virtual try-on. In: ACM MM (2023) 2, 4, 6, 8, 9, 10 

26. Morelli, D., Fincato, M., Cornia, M., Landi, F., Cesari, F., Cucchiara, R.: Dress code: High-resolution multi-category virtual try-on. In: CVPR Workshops (2022) 8, 10 

27. Qian, Y., Cai, Q., Pan, Y., Li, Y., Yao, T., Sun, Q., Mei, T.: Boosting diffusion models with moving average sampling in frequency domain. In: CVPR (2024) 2 

28. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: ICML (2021) 3 

29. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer: High-resolution image synthesis with latent diffusion models. In: CVPR (2022) 2 

30. Ronneberger, O., Fischer, P., Brox, T.: U-net: Convolutional networks for biomedical image segmentation. In: MICCAI (2015) 6 

31. Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., Ganguli, S.: Deep unsupervised learning using nonequilibrium thermodynamics. In: ICML (2015) 2, 4 

32. Song, J., Meng, C., Ermon, S.: Denoising diffusion implicit models. In: ICLR (2021) 9 

33. Song, Y., Ermon, S.: Generative modeling by estimating gradients of the data distribution. In: NeurIPS (2019) 2 

34. Wang, B., Zheng, H., Liang, X., Chen, Y., Lin, L., Yang, M.: Toward characteristicpreserving image-based virtual try-on network. In: ECCV (2018) 2, 3 

35. Wang, Z., Bovik, A.C., Sheikh, H.R., Simoncelli, E.P.: Image quality assessment: from error visibility to structural similarity. IEEE TIP (2004) 8 

36. Xie, Z., Huang, Z., Dong, X., Zhao, F., Dong, H., Zhang, X., Zhu, F., Liang, X.: Gp-vton: Towards general purpose virtual try-on via collaborative local-flow global-parsing learning. In: CVPR (2023) 2, 3, 5, 9, 10 

37. Yang, B., Gu, S., Zhang, B., Zhang, T., Chen, X., Sun, X., Chen, D., Wen, F.: Paint by example: Exemplar-based image editing with diffusion models. In: CVPR (2023) 9 

38. Yang, H., Zhang, R., Guo, X., Liu, W., Zuo, W., Luo, P.: Towards photo-realistic virtual try-on by adaptively generating-preserving image content. In: CVPR (2020) 2 

39. Yu, R., Wang, X., Xie, X.: Vtnfp: An image-based virtual try-on network with body and clothing feature preservation. In: ICCV (2019) 2, 3 

40. Zhang, R., Isola, P., Efros, A.A., Shechtman, E., Wang, O.: The unreasonable effectiveness of deep features as a perceptual metric. In: CVPR (2018) 8 

41. Zhang, Z., Long, F., Pan, Y., Qiu, Z., Yao, T., Cao, Y., Mei, T.: Trip: Temporal residual learning with image noise prior for image-to-video diffusion models. In: CVPR (2024) 2 

42. Zhu, L., Yang, D., Zhu, T., Reda, F., Chan, W., Saharia, C., Norouzi, M., Kemelmacher-Shlizerman, I.: Tryondiffusion: A tale of two unets. In: CVPR (2023) 4, 6 

43. Zhu, R., Pan, Y., Li, Y., Yao, T., Sun, Z., Mei, T., Chen, C.W.: Sd-dit: Unleashing the power of self-supervised discrimination in diffusion transformer. In: CVPR (2024) 2 

