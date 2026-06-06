---
type: paper-fulltext
slug: shining-yourself-ornaments-vton
arxiv_id: null
source_path: /Users/saris.kia.adm/.paper-scholar/shining-yourself-ornaments-vton/2503.16065.md
paper: "[[shining-yourself-ornaments-vton]]"
---
<!-- extracted by afk_extract from 2503.16065.pdf (14p) -->

## **Shining Yourself: High-Fidelity Ornaments Virtual Try-on with Diffusion Model** 

Yingmao Miao[1] _[,]_[2][*] , Zhanpeng Huang[2] , Rui Han[†][2] , Zibin Wang[2] , Chenhao Lin[1][†] , Chao Shen[1] 1Xi’an Jiaotong University, 2 SenseTime Research 

mym2017@stu.xjtu.edu.cn, _{_ linchenhao, chaoshen _}_ @xjtu.edu.cn, 

_{_ huangzhanpeng, hanrui, wangzibin _}_ @sensetime.com 

**==> picture [496 x 330] intentionally omitted <==**

Figure 1. **Shining Yourself.** We propose the virtual try-on task for ornaments including bracelets, rings, earrings, and necklaces for the first time. Our method achieves realistic virtual try-on results and high-fidelity identity preservation of ornament using pose-aware mask prediction and mask-guided attention. Project Page: https://shiningyourself.github.io/ 

## **Abstract** 

_While virtual try-on for clothes and shoes with diffusion models has gained attraction, virtual try-on for ornaments, such as bracelets, rings, earrings, and necklaces, remains_ 

> *Completed during the internship at SenseTime Research 

> †Corresponding authors 

_largely unexplored. Due to the intricate tiny patterns and repeated geometric sub-structures in most ornaments, it is much more difficult to guarantee identity and appearance consistency under large pose and scale variances between ornaments and models. This paper proposes the task of virtual try-on for ornaments and presents a method to improve the geometric and appearance preservation of orna-_ 

_ment virtual try-ons. Specifically, we estimate an accurate wearing mask to improve the alignments between ornaments and models in an iterative scheme alongside the denoising process. To preserve structure details, we further regularize attention layers to map the reference ornament mask to the wearing mask in an implicit way. Experimental results demonstrate that our method successfully wears ornaments from reference images onto target models, handling substantial differences in scale and pose while preserving identity and achieving realistic visual effects._ 

## **1. Introduction** 

As diffusion model [9, 15, 16, 22, 24, 26, 40] becomes the de facto standard in image generation, it’s also widely adopted in the field of virtual try-on [7, 17, 32, 35, 39, 42]. Given a reference image of an item and a target image of a model, the task is to get a preview of the fitting effect using image generation methods. Since no in-person wearing or physical fitting room is required, it has great potential for massive advertising materials generation in various applications of retail, e-commerce, and advertisement. 

The main challenge of virtual try-on is how to generate a realistic fitting effect while preserving the fidelity of the garment. Massive efforts [7, 17, 32, 35, 39] have been dedicated to solving the problem in the field of garments. The methods usually employ a network module (e.g., CLIP image encoder [21] or ReferenceNet [11, 16]) to extract garment features, which are injected into the process of diffusion denoising to preserve the identity and details of the garment. It has made great progress to be widely adopted for commercial use. 

However, few works focus on virtual try-ons of ornaments such as bracelets, rings, earrings, and necklaces, despite significant practical demand. Virtual try-on for ornaments presents unique challenges that existing methods struggle to address effectively. 1) Most ornaments often feature intricate, small-scale geometric structures, such as rings and holes, that are difficult to preserve in virtual tryons. In contrast, garments typically have sparser and/or repeated textures and lack complex geometric details. 2) For garment try-ons, generative visual artifacts can blend with natural cloth deformations and wrinkles, reducing their visibility. However, most ornaments are rigid or consist of rigid components, making any distortion or artifacts immediately noticeable. 3) Most garment virtual try-on methods require silhouette, skeleton, and semantic maps as additional inputs, while even the coarse silhouette mask of the ornament is not easy to depict due to its diverse structures and posedepended occlusion. 

To tackle the problems, we propose a method to predict an accurate wearing mask to align the poses and scales between ornaments and models without additional inputs such 

as the silhouette, skeleton, or semantic maps. Our method also preserves structures and detailed features using a maskguided attention of ornaments and models to preserve geometric structures. Specifically, to obtain a precise poseaware wearing mask without explicit extraction of various maps of ornaments and models, we refine the mask from an input mask as coarse as a bounding box. A refined mask is then estimated using the intermediate features in the generation, which is further used as input to predict a more accurate one iteratively. As the mask indicates structure information, especially for multiple and tiny geometric structures, we formulate the attention layers to implicitly learn a mapping between the reference mask and the ground truth mask to preserve the structure patterns in ornaments. In summary, our contributions are as follows: 

- To the best of our knowledge, we are the first to use diffusion models for virtual try-ons of various ornaments including bracelets, rings, earrings, and necklaces. Our method can generate realistic virtual try-on results as well as high-fidelity preservation of ornament identity. 

- We propose an iterative scheme to estimate a pose-aware wearing mask to significantly improve the pose and scale alignments between ornaments and models. It also facilitates virtual try-on applications without the requirements of various additional inputs such as silhouette, semantic, and skeleton maps. 

- By constraining attention to learning a mapping between reference and wearing masks, our method improves geometric feature preservation, especially for tiny items with complicated geometric structures, such as ornaments. 

## **2. Related Works** 

**Personalized image generation** To address the challenge of text prompts not fully capturing user intent, personalized image generation, and editing have garnered significant attention from researchers. Inversion-based methods, such as Textual Inversion [10] focus on optimizing a special prompt word to represent a target concept. Meanwhile, finetuning approaches [12, 20, 23, 25] adjust pre-trained diffusion models using a small set of images of the target concept to create personalized models. Although these methods can generate high-fidelity images, they do not allow users to control the generation area for the target, and the additional optimization and fine-tuning required during inference can hinder large-scale applications. On the other hand, some zero-shot and training-free personalized image generation methods [30, 36], can produce images of target concepts. However, they excel in stylized generation by representing high-level semantic features in the lack of preserving details and identity. 

**Local image editing** Our virtual ornament-wearing task closely resembles local image editing techniques. However, most previous local image editing methods relied on 

**==> picture [492 x 216] intentionally omitted <==**

Figure 2. **The overview of our method.** a) In training, given reference ornament and model images and masks, our method concatenates ornament and masked model images as input to the ReferenceNet branch, which extracts features to predict wearing mask in an iterative way. The extracted features are also injected into the denoising U-Net to improve details generation. b) We enforce the attention layers to preserve structure details by formulating the layers to map the reference ornament mask to the ground truth wearing mask in an implicit way rather than directly imposing the mask onto attention maps. 

text prompts, such as Blended Diffusion [1] and Blended Latent Diffusion [2], which employed multi-step semantic blending during denoising to produce harmonious images containing target semantic information within a defined mask. Inpainting Anything [37] replaces any object in the input image with a target described by text prompts. However, for virtual ornament wearing, it is crucial to ensure alignment with the intricate details of the target. The traditional image composition pipeline [5, 8] involves cutting and pasting foreground images onto background images, followed by the harmonization techniques. Recently, numerous diffusion-based methods [6, 27, 31, 34] have emerged in this field, significantly enhancing the quality and coherence of generated images. For instance, Paint by Example [34] employs a CLIP image encoder [21] to convert reference images into embeddings for guidance, generating objects that are semantically aligned with the reference image. ObjectStitch [27] similarly utilizes CLIP to align text and images, guiding the generation of the diffusion model. ObjectDrop [31] first trains an object removal network, assembles a large dataset, and subsequently conducts object insertion training. AnyDoor [6] leverages ControlNet and a DINO [3] encoder to extract detailed semantic information, improving its ability to maintain object identities. These methods focus on general object insertion into the background smoothly without pose alignment, which is normally required in virtual try-on tasks, especially for ornament wearing that needs precise pose alignment between 

## the ornament and wrist, finger, or neck. 

**Virtual try-on** Virtual try-on [7, 17, 32, 35, 39, 42] takes a model image and an item image to generate an image of the model wearing the item. Early virtual try-on methods were based on Generative Adversarial Networks (GANs). Recently, with the significant success of diffusion models, researchers have explored their application in the field of virtual try-on. Most works focus on virtual try-on of garments, such as TryonDiffusion [42], OOTDiffusion [32], and IDMVTON [7]. These methods utilize two parallel U-Nets for garment feature extraction, integrating them through self-attention and achieving impressive results. StableVITON [17] introduced a zero-initialization cross-attention module to inject garment features into the denoising network. A few works [4, 33] explore virtual try-on of shoes and earrings. In their settings, either the shoe pose is fixed to be aligned by the model or the earring has an almost vertical pose as it hangs down from the ear. Ornament virtual try-on requires pose alignment with different body parts at various poses and scales. Methods such as OOTDiffusion and IDMVTON use additional inputs such as skeleton and semantic maps to guide wearing pose. In contrast, it’s much more difficult to depict ornament wearing mask due to complicated tiny geometry structures and pose-related occlusion. 

## **3. Methodology** 

We propose a zero-shot method for ornament virtual tryon with a reference ornament image, a target model image, 

and a coarse bounding box. The bounding box coarsely indicates the wearing location as ornament wearing is userspecific (e.g., a ring has its finger symbolism). We can generate realistic and high-fidelity fitting effects without additional inputs such as pose and semantic maps. The model comprises two vital components: 1) an iterative pose-aware wearing mask prediction and refinement module from the bounding box, which improves pose alignments between the ornament and the model; 2) a mask-guided attention module to improve identity and detail preservation. The framework of our method is illustrated in Fig. 2a. 

## **3.1. Diffusion model and ReferenceNet** 

**Diffusion model** Our method is built upon the latent diffusion model (LDM) and ReferenceNet module, which has been widely adopted for condition generation and virtual try-on tasks. A typical LDM implementation [22] comprises an encoder-decoder module and a denoising network. The encoder embeds the input image into a low-dimension latent code to reduce computational overhead, which is diffused and then denoised by the denoising network to recover from a random noise. The denoised latent code is then decoded to generate an RGB image. The training process is formalized as follows: 

**==> picture [187 x 13] intentionally omitted <==**

where _zt_ represents the latent feature at time step _t_ , which can be obtained through _zt_ = _[√] α_ ¯ _tz_ 0 + _[√]_ 1 _− α_ ¯ _tϵ, ϵ ∈ N_ (0 _, I_ ). _c_ is the condition embedding from a text prompt or reference image with a text or image encoder, and injected into the cross-attention layers to guide the generation. Our model adopts the widely used CLIP image encoder to extract features of the reference ornament image. 

**ReferenceNet module** The module is widely used in virtual try-ons to improve detail and structure preservation. It is designed to be similar and parallelized with the denoising U-Net. The module extracts hierarchical latent features of the reference image which are injected into related layers in the denoising network. Specifically, latent features in the ReferenceNet are concatenated onto their counterparts in the denoising network for attention calculation. 

## **3.2. Pose-aware Mask Refinement** 

We conducted several experiments to explore how the pose and scale impact the generative results, from which we have several key findings: 1) The diffusion-based model has the capability to fit ornaments to various model poses even without finetuning on ornament try-on datasets, which is attributed to the image prior from the pre-trained diffusion base model. 2) Poses and scales have significant influences on the fitting effects. In general, using an accurate wearing mask will significantly improve the pose alignment between ornaments and models even with large poses and 

scale variances. However, the wearing mask is not equivalent to the semantic mask, which is predicted from an existing image while wearing mask is hallucinated from two irrelevant ornament and model images. It’s difficult or even infeasible to obtain accurate wearing mask in inference. In addition, ornaments usually show close-up views, which require a much more accurate wearing mask than the coarse silhouette mask in garment virtual try-on. 

Previous works [13, 29] have shown that intermediate results (e.g., latent features and attention maps) in early generative phases contain the semantic structure of the generated images. It might be possible to extract a wearing mask from these intermediate maps. However, extracted masks are too coarse to be used. To solve the problem, we proposed to estimate a more accurate wearing mask. We add an additional linear layer to predict the wearing mask from the intermediate maps. The predicted wearing mask is further used as input to guide the generation. The iterative refinement converges to an accurate pose-aware mask aligned with the model in the final generative image. Specifically, triplet images of an ornament, a model, and a coarse bounding box _Mb_ are fed into the ReferenceNet. The ornament latent features _fo[t]_[and the model latent features] _[ f] m[ t]_[are injected] into the denoising counterpart similar to most networks in garment virtual try-ons. The latent features are further concatenated and linearly projected to predict the wearing mask _M_ ˆ _[t]_[The predicted mask and the bounding box are blended] _p_[.] as a new wearing mask input, which is updated as follows: 

**==> picture [198 x 14] intentionally omitted <==**

**==> picture [181 x 14] intentionally omitted <==**

where _αt ∈_ [0 _,_ 1] is a hyperparameter in terms of training step _t_ . In the early training, the predicted wearing mask is coarse, and _αt_ is set to be small. As the mask gets more accurate in the late stage, _αt_ approximates to 1.0. 

As mask prediction and image generation are entangled, we employ an ornament try-on dataset with wearing masks to regularize mask prediction with a _L_ 2 loss: 

**==> picture [163 x 15] intentionally omitted <==**

where _Mo[gt]_ is the ground-truth wearing mask. The regularization is important to prevent the dual degeneration of both results due to mutual dependence. In inference, only a bounding box is required to indicate the user-specific wearing location. 

## **3.3. Mask-guided Attention** 

The precise wearing mask improves the alignment between ornaments and models with various poses and scales. It also has a positive effect on detail generation. However, most ornaments comprise complicated tiny geometric components such as repeated shapes and/or ring structures in 

**==> picture [496 x 439] intentionally omitted <==**

Figure 3. Visual comparison between previous methods and ours. No existing method could keep appearance and structure consistent, especially geometric details and numbers of components in ornaments. Our method preserves both details and identity and achieves highquality and high-fidelity fitting results. 

pearl necklaces and beaded bracelets. Our early attempts found that the model had difficulty preserving the topology and/or the number of components, especially in repeated geometric patterns. To take a few examples, it may ignore small parts interleaved with other large components, or fill the hole of a ring structure. We suspect that existing generative networks can capture appearance and spatial details rather than geometric structures, as geometric shapes require hard constraints on local primitive structures of edges and contours. 

Attention maps retain shape details with affinities be- 

tween the spatial features [13]. A possible solution is to impose a geometric structure constraint in attention maps. The semantic segmentation mask contains rich geometric structure information, but the mask is difficult to extract due to massive tiny complicated sub-components in ornaments. As the binary mask is also full of geometric structures of edges and contours and easy to obtain (e.g., with SAM [19]), we propose to employ the reference ornament mask to inject geometric structure into the generation. 

However, directly blending attention maps with the mask may mask out too much information to degrade gener- 

ated results. We introduce an indirect way to restrict geometric structure changes of ornaments in reference and generated images. Specifically, we obtain attention maps _{Ma[i] ∈_ R _[d][i][×][d][i] }_ 1 _[N]_[of][latent][features][and][ornament][em-] beddings from various layers in denoising U-Net, where _N_ is the number of extracted attention maps and _di_ is the dimension of _i_ -th attention map. The ornament mask _Mo_ in the reference image is down-sampled and flattened as onedimensional masks _{Mo[i][∈]_[R] _[d][i][}][N]_ 1[.][We then apply] _[ M][ i] o_[to] mask out the attention map _Ma[i]_[along][one][dimension][and] margin it along the other dimension. The result is then reshaped and up-sampled to _M_[˜] _o[i]_[as][the][same][dimension][of] _Mo_ . All result masks _{M_[˜] _o[i][}]_[are][then][averaged][as][the][final] mask _M_[˜] _o_ , which can be formulated as: 

**==> picture [213 x 55] intentionally omitted <==**

**==> picture [231 x 34] intentionally omitted <==**

**==> picture [154 x 30] intentionally omitted <==**

where **T[d1] ops** _[→]_ **[d2]** is the operator with **ops** as operation type and **d1** _→_ **d2** as dimension mapping from **d1** to **d2** . Sequential operators are defined to execute from right to left. The masking operation enforces latent features to attend to ornament regions in the reference image, while the margining operation diffuses ornament features to the wearing region in the generated image. The reference mask _Mo_ is mapped to the wearing masks _M_[˜] _o_ via the attention map. Inversely, in order to enable the attention map to learn the mapping, we require the transformed wearing masks _M_[˜] _o_ to be consistent with the ground truth wearing ornament mask _Mo[gt]_ with an _L_ 2 loss as below: 

**==> picture [161 x 13] intentionally omitted <==**

The process is shown in Fig.2b. Down-sampling and upsampling operations are not displayed for concise illustration. 

Our dataset does not require pose alignment between the ornament and the model, which is easy to collect, and also prevents the model from learning a simple copy-andpaste strategy. In total, we collect about 64k image triplets, roughly evenly distributed over four categories of bracelets, rings, earrings, and necklaces. Each image triplet also contains a reference mask and a wearing mask of the ornament. **Training loss** Our training loss comprises the aforementioned three items : 

**==> picture [181 x 10] intentionally omitted <==**

where _λ_ 1 and _λ_ 2 are loss weights. The two weights decay as the training step increases, which forces the model to learn the wearing mask in the early stages. As the mask becomes accurate, the model focuses on the generation of appearance details. The scheme follows common observation [28] that image layout and structure are sketched in the early stage and details are generated in the late stage. 

## **4. Experiments** 

## **4.1. Implement details** 

Our model adopts the Stable Diffusion V1.5 as the network backbone. The ornament region is cropped and resized to 512 _×_ 512, and Adam [18] optimizer is chosen with an initial learning rate of 1 _e[−]_[5] . We employ a simple linear decay for the _α_ ( _t_ ) and select the self-attention maps with the highest resolution from the encoder and decoder for masked-guided attention. We found these simple settings are enough to obtain compelling results. We follow a similar scheme as AnyDoor [6] in handling inputs and composing final results. Specifically, the ground truth mask is resized to be square, and the cropped image is scaled by a factor of 1.5. The generated result is pasted back to the original masked region to compose the final result. Our model takes about 10 hours to train on 8 A100 GPUs with 10 epochs. 

For quantitative comparison, we adopt FID [14] and LPIPS [41] to evaluate image quality, while the CLIP image similarity score and DINO-based feature similarity score are used to measure the identity consistency of the ornaments. All results are calculated and averaged on a test image set split from our dataset. 

## **4.2. Comparisons** 

## **3.4. Training** 

**Dataset** Inspired by the common practice in garment virtual try-ons, we collect image pairs of ornaments and models wearing the ornaments. We mask out the ornament in the model image to obtain the target image and ground-truth wearing mask. The reference ornament image, the model image with masked-out ornament, and the original model image are combined as a training triplet image. We also label the masks in ornament images as reference masks. 

As we are the first to focus on ornament virtual try-ons, we select several works that are most related to ours in the broad field of image edits. These works include Paintby-Example(CVPR’23) [34], AnyDoor(CVPR’24) [6], and IDM-VTON(ECCV’24) [7]. The first two works are designed to insert reference objects into target images, while the latter is dedicated to garment virtual try-ons. Similar to ours, these methods require a reference image of the item and a target image as well as masks to define local edit 

**==> picture [496 x 343] intentionally omitted <==**

Figure 4. Virtual try-on results on other categories including bracelets, rings, necklaces, and earrings. 

Table 1. Quantitative evaluations between our and other methods 

||Compared againstground truth|Compared against reference ornament|
|---|---|---|
|Method|FID_↓_<br>LPIPS_↓_<br>CLIP Score_↑_<br>DINO Score_↑_|CLIP Score_↑_<br>DINO Score_↑_|
|Paint-by-Example [34]<br>AnyDoor [6]<br>IDM-VTON [7]<br>Ours|23.49<br>0.0789<br>85.6<br>64.8<br>28.28<br>0.1029<br>85.1<br>67.2<br>22.99<br>0.0709<br>85.9<br>65.0<br>**19.00**<br>**0.0593**<br>**88.7**<br>**74.5**|57.2<br>35.4<br>54.8<br>35.9<br>55.9<br>35.2<br>**57.3**<br>**38.7**|
|Ground Truth|-<br>-<br>-<br>-|59.0<br>43.3|



regions. All methods are trained or fine-tuned with our dataset. Limited to the page length requirement, we take the bracelet category as an example to illustrate all visual results. Please refer to the Appendix for more results of other ornament categories. 

**Qualitative results** Fig. 3 qualitatively compares fitting results on ornaments of various structures and poses. Paintby-Example could hardly preserve geometric structures and appearance in most cases. AnyDoor struggles to preserve the scales of the whole structure and/or major parts. IDMVTON could preserve the scale to a certain extent, but it has 

problems maintaining structure layouts, especially for complicated ornaments with multiple parts. None of the previous methods could hold the number of sub-parts in ornaments or recover repeated geometric patterns. Our method has the most visual appearance and structure similarities to both reference ornaments and ground truth images, which indicates its ability to preserve both appearance and local and global structures, and tiny surface geometric patterns (the last row). Surprisingly, our results seem to be biased towards reference ornaments with less specular reflections than ground truth. It’s partially because model images do 

Table 2. Quantitatively comparisons of our models with different module configurations. 

|Method|CLIP Score_↑_<br>DINO Score_↑_|
|---|---|
|Baseline<br>w/o mask refnement<br>w/o mask-guided atten.<br>Ours|86.9<br>71.9<br>88.5<br>73.3<br>88.0<br>73.9<br>**88.7**<br>**74.5**|



not have enough hints of environmental illumination that our model has difficulty in learning the exact light effect as the ground truth. 

**Quantitative results** Table 1 illustrates a quantitative comparison between ours and previous methods. The two ornaments in reference and ground-truth images are usually captured with different conditions such as views and illumination. Therefore, we compare the results generated by all methods against both the reference ornaments and the ground truth, and calculate the corresponding consistency metrics. Our approach achieves the best results in all metrics, demonstrating its capability to generate more realistic and high-fidelity virtual results. 

## **4.3. Ablation Study** 

We conduct a comprehensive ablation study to evaluate the effectiveness of our proposed components. The experiments are designed by adding a component from the basic models. The baseline is adapted from ReferenceNet and Stable Diffusion. The results are evaluated from both qualitative and quantitative aspects. Fig. 5 shows the visual comparisons with various component configurations. The basic model has noticeable defects in details and geometric structures. If we do not integrate the mask prediction, the results lack appearance details and specular lights. It may also lose structure consistency to some extent (e.g., flow structure missing in the first ornament). Without mask-guided attention, both the local and global structures are destroyed in form of adding or missing components as well as changing scales. On the other hand, the full model preserves both appearance and geometric details as well as global structures. The quantitative results in Table 2 also indicate the importance of the two proposed modules to improve the final virtual try-on results. More results are in the Appendix. 

## **4.4. More Results** 

We use our method to wear various types of ornaments including bracelets, necklaces, earrings, and rings. Figure 4 lists the results. Other configurations including a model wearing different ornaments and different models wearing the same ornaments are also illustrated in Figure 1 (the last row). All results demonstrate that our method can handle various ornament structures of local and global rigid and 

**==> picture [237 x 126] intentionally omitted <==**

Figure 5. The visual comparisons of our models with different module configurations. The full model archives the best results with the proposed two modules. 

**==> picture [237 x 79] intentionally omitted <==**

Figure 6. Our model is robust to achieve consistent results with different poses and scales. 

non-rigid components. More results are in the Appendix. 

To evaluate the robustness of our model under conditions of different poses and scales. We also conduct experiences by randomly rotating and scaling the reference ornament, which is then used to wear on the same model. As Fig. 6 shows, the results are consistent in details and geometric structures with different configurations. The experiment results show our model is robust with large pose differences between ornaments and models. 

## **5. Conclusion** 

We propose the virtual try-on ornament task for the first time. To tackle the more challenging problems of intricate geometric structures in ornaments, we devise two modules of mask prediction and mask-guided attention to obtain accurate wearing masks and impose geometric structures, which preserve both appearance details and geometric structures to achieve identity consistency. Currently, our method is biased toward reference images rather than ground truth images, which lack specular reflections to a certain extent. Inspired by the work [38], we would like to add more fine-grained lighting control in our future work. Besides, inaccurate control over wearing orientations (e.g., rotated along a wrist) leading to featured components being hidden behind the wrist occasionally. Secondary masks and local feature injection into the diffusion process may fix the problem. 

## **Acknowledge** 

This research is supported by Sensetime, the National Key Research and Development Program of China (2023YFB3107401), the National Natural Science Foundation of China (T2341003, 62376210, 62161 160337, 62132011, U21B2018, U24B20185, 62206217), the Shaanxi Province Key Industry Innovation Program (2023-ZDLGY-38). Special thanks to Jessie Geng for the coordination of computing resources. 

## **References** 

- [1] Omri Avrahami, Dani Lischinski, and Ohad Fried. Blended diffusion for text-driven editing of natural images. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 18208–18218, 2022. 3 

- [2] Omri Avrahami, Ohad Fried, and Dani Lischinski. Blended latent diffusion. _ACM transactions on graphics (TOG)_ , 42 (4):1–11, 2023. 3 

- [3] Mathilde Caron, Hugo Touvron, Ishan Misra, Herv´e J´egou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In _Proceedings of the IEEE/CVF international conference on computer vision_ , pages 9650–9660, 2021. 3 

- [4] Binghui Chen, Wenyu Li, Yifeng Geng, Xuansong Xie, and Wangmeng Zuo. Shoemodel: Learning to wear on the user-specified shoes via diffusion model. _arXiv preprint arXiv:2404.04833_ , 2024. 3 

- [5] Bor-Chun Chen and Andrew Kae. Toward realistic image compositing with adversarial learning. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 8415–8424, 2019. 3 

- [6] Xi Chen, Lianghua Huang, Yu Liu, Yujun Shen, Deli Zhao, and Hengshuang Zhao. Anydoor: Zero-shot object-level image customization. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 6593–6602, 2024. 3, 6, 7, 1 

- [7] Yisol Choi, Sangkyung Kwak, Kyungmin Lee, Hyungwon Choi, and Jinwoo Shin. Improving diffusion models for authentic virtual try-on in the wild. In _ECCV_ , pages 206–235, 2024. 2, 3, 6, 7, 1 

- [8] Wenyan Cong, Jianfu Zhang, Li Niu, Liu Liu, Zhixin Ling, Weiyuan Li, and Liqing Zhang. Dovenet: Deep image harmonization via domain verification. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 8394–8403, 2020. 3 

- [9] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. In _Advances in Neural Information Processing Systems_ , pages 8780–8794. Curran Associates, Inc., 2021. 2 

- [10] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit Haim Bermano, Gal Chechik, and Daniel Cohen-or. An image is worth one word: Personalizing text-to-image generation using textual inversion. In _The Eleventh International Conference on Learning Representations_ . 2 

- [11] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized textto-image diffusion models without specific tuning. In _The Twelfth International Conference on Learning Representations_ . 2 

- [12] Ligong Han, Yinxiao Li, Han Zhang, Peyman Milanfar, Dimitris Metaxas, and Feng Yang. Svdiff: Compact parameter space for diffusion fine-tuning. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 7323–7334, 2023. 2 

- [13] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-or. Prompt-to-prompt image editing with cross-attention control. In _The Eleventh International Conference on Learning Representations_ . 4, 5 

- [14] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. _Advances in neural information processing systems_ , 30, 2017. 6 

- [15] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. _Advances in neural information processing systems_ , 33:6840–6851, 2020. 2 

- [16] Li Hu. Animate anyone: Consistent and controllable imageto-video synthesis for character animation. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 8153–8163, 2024. 2 

- [17] Jeongho Kim, Guojung Gu, Minho Park, Sunghyun Park, and Jaegul Choo. Stableviton: Learning semantic correspondence with latent diffusion model for virtual try-on. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 8176–8185, 2024. 2, 3 

- [18] Diederik P Kingma. Adam: A method for stochastic optimization. _arXiv preprint arXiv:1412.6980_ , 2014. 6 

- [19] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 4015–4026, 2023. 5 

- [20] Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Multi-concept customization of text-to-image diffusion. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 1931–1941, 2023. 2 

- [21] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In _International conference on machine learning_ , pages 8748–8763. PMLR, 2021. 2, 3 

- [22] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 10684–10695, 2022. 2, 4 

- [23] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine 

tuning text-to-image diffusion models for subject-driven generation. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 22500– 22510, 2023. 2 

- [24] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. _Advances in neural information processing systems_ , 35:36479–36494, 2022. 2 

- [25] Jing Shi, Wei Xiong, Zhe Lin, and Hyun Joon Jung. Instantbooth: Personalized text-to-image generation without test-time finetuning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 8543–8552, 2024. 2 

- [26] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In _International Conference on Learning Representations_ . 2 

- [27] Yizhi Song, Zhifei Zhang, Zhe Lin, Scott Cohen, Brian Price, Jianming Zhang, Soo Ye Kim, and Daniel Aliaga. Objectstitch: Object compositing with diffusion model. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 18310–18319, 2023. 3 

- [28] Yoad Tewel, Omri Kaduri, Rinon Gal, Yoni Kasten, Lior Wolf, Gal Chechik, and Yuval Atzmon. Training-free consistent text-to-image generation. _ACM Transactions on Graphics (TOG)_ , 43(4):1–18, 2024. 6 

- [29] Narek Tumanyan, Michal Geyer, Shai Bagon, and Tali Dekel. Plug-and-play diffusion features for text-driven image-to-image translation. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , pages 1921–1930, 2023. 4 

- [30] Qixun Wang, Xu Bai, Haofan Wang, Zekui Qin, Anthony Chen, Huaxia Li, Xu Tang, and Yao Hu. Instantid: Zero-shot identity-preserving generation in seconds. _arXiv preprint arXiv:2401.07519_ , 2024. 2 

_the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 7017–7026, 2024. 2, 3 

   - [36] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ipadapter: Text compatible image prompt adapter for text-toimage diffusion models. 2023. 2 

   - [37] Tao Yu, Runseng Feng, Ruoyu Feng, Jinming Liu, Xin Jin, Wenjun Zeng, and Zhibo Chen. Inpaint anything: Segment anything meets image inpainting. _arXiv preprint arXiv:2304.06790_ , 2023. 3 

   - [38] Chong Zeng, Yue Dong, Pieter Peers, Youkang Kong, Hongzhi Wu, and Xin Tong. Dilightnet: Fine-grained lighting control for diffusion-based image generation. In _ACM SIGGRAPH 2024 Conference Papers_ , 2024. 8 

   - [39] Jianhao Zeng, Dan Song, Weizhi Nie, Hongshuo Tian, Tongtong Wang, and An-An Liu. Cat-dm: Controllable accelerated virtual try-on with diffusion model. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 8372–8382, 2024. 2, 3 

   - [40] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 3836–3847, 2023. 2 

   - [41] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 586–595, 2018. 6 

   - [42] Luyang Zhu, Dawei Yang, Tyler Zhu, Fitsum Reda, William Chan, Chitwan Saharia, Mohammad Norouzi, and Ira Kemelmacher-Shlizerman. Tryondiffusion: A tale of two unets. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 4606–4615, 2023. 2, 3 

- [31] Daniel Winter, Matan Cohen, Shlomi Fruchter, Yael Pritch, Alex Rav-Acha, and Yedid Hoshen. Objectdrop: Bootstrapping counterfactuals for photorealistic object removal and insertion. _arXiv preprint arXiv:2403.18818_ , 2024. 3 

- [32] Yuhao Xu, Tao Gu, Weifeng Chen, and Chengcai Chen. Ootdiffusion: Outfitting fusion based latent diffusion for controllable virtual try-on. _arXiv preprint arXiv:2403.01779_ , 2024. 2, 3 

- [33] Youze Xue, Binghui Chen, Yifeng Geng, Xuansong Xie, Jiansheng Chen, and Hongbing Ma. Strictly-id-preserved and controllable accessory advertising image generation. _arXiv preprint arXiv:2404.04828_ , 2024. 3 

- [34] Binxin Yang, Shuyang Gu, Bo Zhang, Ting Zhang, Xuejin Chen, Xiaoyan Sun, Dong Chen, and Fang Wen. Paint by example: Exemplar-based image editing with diffusion models. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 18381–18391, 2023. 3, 6, 7, 1 

- [35] Xu Yang, Changxing Ding, Zhibin Hong, Junhao Huang, Jin Tao, and Xiangmin Xu. Texture-preserving diffusion models for high-fidelity virtual try-on. In _Proceedings of_ 

## **Shining Yourself: High-Fidelity Ornaments Virtual Try-on with Diffusion Model** Supplementary Material 

## **6. Additional Experiments** 

## **6.1. Comparison on Necklaces, Earrings, and Rings** 

In addition to the comparison with previous methods using the bracelet dataset, as discussed in the main text, we extended the evaluation to other categories of ornaments. The results are shown in Fig. 9. Consistent with the qualitative comparisons in the main text, Paint-by-Example [34] faces challenges in preserving geometric or ID information, retaining only partial semantic features. AnyDoor [6] has difficulty capturing correct wearing patterns and natural region inpainting. IDM-VTON [7] shows some improvement over other methods but still struggles with maintaining fine details and spatial relationships. 

In contrast, our method demonstrates superior performance across all categories. Necklaces, due to their small wearing area and the challenge of invisible chain parts, are the most difficult among all categories. Nevertheless, our method achieves good results. Notably, in the first row of examples, where some structures are not visible in the reference image, our method faithfully preserves the reference ornament’s ID information. Rings share a similar wearing pattern to bracelets but include more subtle structural details. As seen in the 5th and 6th rows, our method generates high-fidelity try-on images that preserve these fine details. For earrings, the results in the 8th row demonstrate that our method effectively handles fine linear decorations. 

## **6.2. Additional Results** 

We conducted further experiments across all categories, and Fig. 10 presents the results of our virtual try-on. Our method demonstrates stable, high-fidelity virtual try-on for ornaments. Additionally, we explored cross-domain ornament virtual try-on and found that our method is capable of virtual try-on for certain animated characters. This showcases the robustness of our approach and indicates that, with sufficient data, our method can be extended to images from other domains. 

## **6.3. Additional Ablation Study** 

**Experiments to Validate Our Motivation.** To validate the motivation for mask refinement, we conducted experiments on different masks using the baseline based on ReferenceNet and Stable Diffusion. The results are shown in Fig. 7. We trained the model with various input masks, including bounding boxes, oriented bounding boxes (OBB), convex hulls, and ground truth masks. The results show that as mask refinement increases, the fidelity and ID consistency of the generated images improve. For instance, us- 

**==> picture [237 x 326] intentionally omitted <==**

Figure 7. Experiments to validate our motivation. 

ing a bounding box as the mask increases the probability of errors in the wearing pattern. The OBB, which adds limited pose information, shows some improvement but still fails to produce satisfactory results. The convex hull adds extra shape information, and some geometric structures, which earlier methods struggled to preserve, are retained in the output. The ground truth mask yields the best results, providing precise shape and location information. These results highlight the importance of pose and accurate masks for generation quality, which is a key idea in our work. However, as mentioned in Section 3.2 of the main text, a fine-grained wearing mask cannot be obtained during the inference process. Our method uses only the bounding box as input to predict the wearing mask and then refines the original mask. The results demonstrate that our method achieves performance close to that of the ground truth mask. 

## **6.4. Results on Garments Virtual Try-on** 

Our task extends the field of garments virtual try-on. As analyzed in the introduction, the proposed ornaments virtual try-on presents more significant challenges compared to garments virtual try-on. Therefore, we have developed a series of customized improvement strategies. To demonstrate the performance of our method in garments virtual try-on, we conducted qualitative experiments on the VITON-HD dataset, and the comparative results with IDM-VTON are shown in Fig.8. 

It should be noted that our method is not specifically designed for garments virtual try-on, and thus many inputs for garments virtual try-on are not considered. We only utilize the model image, garments image, and mask. Despite this limitation, our method achieves comparable results to stateof-the-art (SOTA) methods. Given that garments virtual tryon has been extensively studied and previous methods have achieved remarkable results, our method faces certain limitations in further enhancing garments try-on performance. 

Through multiple experimental validations presented in this paper, we can conclude that our method exhibits excellent versatility. It maintains the performance of garments virtual try-on while achieving outstanding results in ornaments virtual try-on. This characteristic makes our method highly promising for a wide range of applications in the field of virtual try-on. 

**==> picture [237 x 184] intentionally omitted <==**

Figure 8. Visual comparison on garments virtual try-on. 

**==> picture [472 x 618] intentionally omitted <==**

Figure 9. Visual comparison on necklaces, rings, and earrings. 

**==> picture [472 x 618] intentionally omitted <==**

Figure 10. More results. 

