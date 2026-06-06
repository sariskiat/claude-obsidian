---
type: paper-fulltext
slug: vton-360-high-fidelity-any-view
arxiv_id: null
source_path: /Users/saris.kia.adm/.paper-scholar/vton-360-high-fidelity-any-view/2503.12165.md
paper: "[[vton-360-high-fidelity-any-view]]"
---
<!-- extracted by afk_extract from 2503.12165.pdf (14p) -->

# **VTON 360: High-Fidelity Virtual Try-On from Any Viewing Direction** 

Zijian He[1] Yuwei Ning[1] Yipeng Qin[2] Guangrun Wang[1] Sibei Yang[3] Liang Lin[1,4,5] Guanbin Li[1,4,5] _[∗]_ 1Sun Yat-sen University 2Cardiff University 3ShanghaiTech University 4Guangdong Key Laboratory of Big Data Analysis and Processing 5Peng Cheng Laboratory hezj39@mail2.sysu.edu.cn, yuwei ~~n~~ ing@hust.edu.cn, _{_ qinyipeng1991,wanggrun _}_ @gmail.com, yangsb@shanghaitech.edu.cn linliang@ieee.org, liguanbin@mail.sysu.edu.cn 

**==> picture [496 x 175] intentionally omitted <==**

Figure 1. **Results of VTON 360.** Our VTON 360 enables high-fidelity 3D Virtual Try-On (VTON) by seamlessly adapting E-commerce garments onto a clothed 3D human model, supporting full 360 _[◦]_ view rendering. The highlighted bounding boxes (dashed line) demonstrate our method’s ability to preserve intricate clothing details and patterns ( _e.g_ ., collar accessories, horizontal line patterns, logos, texts, numbers) across diverse garment types. 

## **Abstract** 

_Virtual Try-On (VTON) is a transformative technology in e-commerce and fashion design, enabling realistic digital visualization of clothing on individuals. In this work, we propose VTON 360, a novel 3D VTON method that addresses the open challenge of achieving high-fidelity VTON that supports any-view rendering. Specifically, we leverage the equivalence between a 3D model and its rendered multiview 2D images, and reformulate 3D VTON as an extension of 2D VTON that ensures 3D consistent results across multiple views. To achieve this, we extend 2D VTON models to include multi-view garments and clothing-agnostic human body images as input, and propose several novel techniques to enhance them, including: i) a pseudo-3D pose representation using normal maps derived from the SMPL-X 3D human model, ii) a multi-view spatial attention mechanism that models the correlations between features from different viewing angles, and iii) a multi-view CLIP embedding that enhances the garment CLIP features used in 2D VTON with camera information. Extensive experiments on largescale real datasets and clothing images from e-commerce_ 

*Corresponding author is Guanbin Li. 

_platforms demonstrate the effectiveness of our approach. Project page: https://scnuhealthy.github.io/ VTON360._ 

## **1. Introduction** 

Virtual Try-On (VTON) enables realistic digital visualization of clothing on individuals and has emerged as a transformative technology in e-commerce and fashion design. While significant research efforts have been made on 2D VTON solutions [8, 12, 19, 33, 39], these approaches are inherently limited in their representation of view-related features. To overcome this limitation and enable high-fidelity any-view rendering, 3D VTON methods were introduced. 

3D VTON requires accurate garment transfer onto a 3D human body while ensuring realistic garment fitting, texture preservation, and 3D consistency. The two primary aims of 3D VTON are i) achieving _high-fidelity_ and ii) supporting _any-view rendering_ . Leveraging the inherent capability of 3D models for _any-view rendering_ , early 3D VTON methods [13, 15, 28] make clothing simulation on synthetic human bodies. Specifically, these methods utilized 3D scanners to capture clothing meshes, followed by the development of specialized dressing algorithms. Although effec- 

tive, these methods rely on costly 3D scanning equipment and the physical presence of the human body/clothing ( _i.e_ ., not fully virtual), restricting their practicality in real-world applications. As a byproduct, most early methods focused on developing geometrically correct dressing algorithms using standard templates of human body and clothing models. Addressing this limitation, researchers extended 3D VTON by introducing algorithms that reconstruct 3D clothing models from input images, enabling the use of imagebased clothing inputs [3, 32, 41, 42]. However, since input clothing images (usually frontal) are inherently 2D and lack multi-view information, this approach struggles to reconstruct high-fidelity clothing models that can be rendered well from all viewing directions. 

To complement this missing information, DreamVTON [50] introduces a novel approach that leverages Textto-Image (T2I) diffusion models to reconstruct both the human body and clothing from input images. Its key insight is that T2I models learned view-agnostic “concepts” of both bodies and garments during their training, and that the corresponding concepts for the input body and clothing images can be obtained using LoRA [21]. By utilizing Score Distillation Sampling (SDS) [37], DreamVTON can generate visual-pleasing 3D VTON results by ensuring consistency between renderings from arbitrary viewpoints and the concepts. Nonetheless, DreamVTON’s high flexibility comes at the cost of low fidelity. This limitation stems from the fact that the concepts learned by T2I models are semantic in nature, thus lacking 3D geometric consistency and pixellevel accuracy with respect to the input body and clothing images. Recently, a concurrent work, namely GaussianVTON [6], partially addressed this limitation by formulating 3D VTON as a 3D scene editing task, where a given 3D human model is edited using multi-view images generated by 2D VTON methods. While it significantly enhances the fidelity of the human body, the fidelity and 3D consistency of clothing remain problematic, as there are no 2D VTON methods that can generate multi-view images with 3D consistency. Therefore, to the best of our knowledge, achieving high-fidelity 3D VTON that supports any-view rendering remains an open challenge. 

In this work, we address the above-mentioned challenge via proposing VTON 360, a novel 3D VTON method that achieves high-fidelity VTON from arbitrary viewing directions. Similar to GaussianVTON [6], our method edits a given 3D human model by inpainting the rendered images using a latent diffusion model. However, we set ourselves apart through our novel garment fidelity preservation strategy that can generate high-fidelity on-body garments in all viewing directions. Specifically, we first extend both the garment and clothing-agnostic human body inputs to typical 2D VTON models to leverage multi-view information, including paired front and back view garment images as well 

as a set of multi-view clothing-agnostic human body images sampled from random azimuth angles. Then, we propose several novel enhancements to bridge the gap between typical 2D VTON methods and our multi-view 3D consistency requirements: i) We propose a pseudo-3D pose representation using normal maps derived from the SMPL-X 3D human model, which captures fine-grained surface orientation details and provides more consistent geometry across views compared to the 2D pose representations (semantic segmentation maps) used in 2D VTON models. ii) We design a Multi-view Spatial Attention mechanism that models the correlations between features from different viewing angles, featuring a novel “correlation” matrix modeling the relationships among different input views. iii) We propose a multi-view CLIP embedding that enhances the garment CLIP embedding used in 2D VTON methods with camera information, thereby facilitating network learning of features relevant to a particular view. Together, these innovations enable our 2D VTON model to generate highquality, multi-view and 3D-consistent virtual try-on results. Extensive experiments on Thuman2.0 [55] and MVHumanNet [51] datasets demonstrate that our method achieves high fidelity 3D VTON which supports any-view rendering. In addition, we show the effectiveness and generalizability of our methodology by testing it using garments from e- commerce platforms. Our conclusions include: 

- We propose a novel 3D Virtual Try-On (VTON) method, namely _VTON 360_ , which achieves high-fidelity VTON from arbitrary viewing directions. 

- Leveraging the _equivalence_ between a 3D model and its rendered multi-view 2D images, we reformulate 3D VTON as an extension of 2D VTON that ensures consistent results across multiple views. Specifically, we introduce several novel techniques, including: (i) pseudo3D pose representation; (ii) multi-view spatial attention; and (iii) multi-view CLIP embedding. These innovations enhance traditional 2D VTON models to generate multiview and 3D-consistent results. 

- Extensive experimental results on two large real datasets as well as real clothing images from e-commerce platforms demonstrate the effectiveness of our approach. 

## **2. Related Work** 

**2D Virtual Try-On.** 2D Virtual Try-On (VTON) aims to transfer a desired garment to the corresponding region of a target human image while preserving the human pose and identity. Early methods [2, 8, 10, 11, 16, 18, 29, 31, 38, 54, 56] use Generative Adversarial Networks (GANs) to deform the garments to match the target body shape, which a critical step for achieving realistic VTON. However, accurately adapting to diverse real-world conditions remains a significant challenge. Addressing this issue, recent methods [12, 19, 33, 60] reframe 2D VTON as a conditioned in- 

painting task, leveraging the strong priors provided by diffusion models [20, 43, 45] to achieve promising results. This strategy is further improved by [9, 26, 53], which introduce a ReferenceNet to extract hierarchical garment features and apply attention mechanisms to condition the Main UNet. 

**3D Virutal Try-On.** For 3D Virtual Try-On (VTON), traditional methods [4, 13, 15, 28, 36] rely on 3D scanning or cloth simulation to generate highly precise body and garment geometry. These methods were then extended by learning-based methods [3, 32] that employ differentiable rendering to dress the SMPL [30] model with a desired garment mesh. Despite their effectiveness, such methods rely on costly 3D scanning and the physical presence of human body/clothing, limiting their application in the real world. Addressing this limitation, M3D-VTON [59] proposes a depth-based 3D VTON framework to reconstruct 3D clothed human models from 2D human and garment images, but the results often suffer from explicit warping artifacts. To improve 3D VTON results, recent methods [23, 24, 50, 62] resort to text-to-image (T2I) diffusion models and employ the Score Distillation Sampling (SDS) loss [40] to ensure consistency among different viewing directions. Specifically, TeCH [24] adapts the generative priors of T2I diffusion model to the specific person and clothes by training descriptive text prompts with DreamBooth [40]. DreamWaltz [23] leverages Pose ControlNet [57] to attain clothed human body models. DreamVTON [50] introduces a multi-concept LoRA [21] to personalize the T2I diffusion model, and uses a template-based optimization mechanism that combines with SDS loss to better preserve patterns on the garment. Although effective, these methods often produce results lacking in fidelity, as the concepts learned by T2I models are semantic rather than at the pixel level. Concurrent to our work, GaussianVTON [6] proposes an alternative approach by combining Gaussian Splatting [25] with pre-trained 2D VTON models and formulate it as an editing task. However, since there are no 2D VTON methods that can generate multi-view images with 3D consistency, the fidelity and 3D consistency of the clothing generated remain problematic. 

**Radiance Field-based 3D Human or Scene Editing.** Recently, radiance field-based editing has attracted significant interest due to its efficient differentiable rendering capabilities, sparking substantial advancements in text-driven 3D editing. For example, InstructN2N [17] employ an imagebased diffusion model InstructP2P [5] to modify the rendered image by the user’s text description with a variant of the score distillation sampling (SDS) [37] loss. GaussianEditor [7] applies Gaussian Splatting [25] as 3D representation instead of NeRF, adopting Gaussian semantic tracking to track target Gaussian values, significantly improving editing speed and controllability. To enable accu- 

rate location and appearance control, subsequent works [47, 61] specify the editing region using the attention score or with a segmentation model. TIP-Editor [63] proposes a content personalization step dedicated to the reference image based on LoRA, achieving the editing following a hybrid text-image prompt. GaussCtrl [48] leverage depth conditions and attention-based latent code alignment to achieve 3D-aware multi-view consistent editing instead of iteratively editing single views using SDS loss. However, these works primarily focus on global appearance modifications within a text-driven pipeline, while our approach emphasizes preserving fine textural details from different viewing directions throughout the editing process. 

## **3. Preliminary** 

**Latent Diffusion Model** . Latent Diffusion Model [39] is a variant of diffusion models that performs denoising within the latent space of a Variational Autoencoder (VAE) [27]. During training, given a fixed encoder _E_ , an input image _x_ is transformed into its latent representation _z_ 0 = _E_ ( _x_ ). A conditional diffusion model ˆ _**ϵ** θ_ , typically implemented with a UNet architecture, is then trained using a weighted denoising score matching objective: 

**==> picture [194 x 13] intentionally omitted <==**

where **z** _t_ := _αt_ **x** + _σt_ _**ϵ**_ denotes the forward diffusion process at timestep _t_ ; _αt, σt_ are time-dependent functions defined by the diffusion model formulation; **c** denotes the conditional input and _**ϵ** ∼N_ ( **0** _,_ **1** ) is Gaussian noise. During inference, data samples are generated by initiating from Gaussian noise **z** _T ∼N_ ( **0** _,_ **1** ) and iteratively refining it using a DDIM [44] sampler. 

## **4. Method** 

Our method leverages the _equivalence_ between a 3D model and its rendered multi-view 2D images to achieve highfidelity, any-view 3D VTON. Specifically, as Fig. 2 shows, given an input 3D human model and a garment image, our method 1) renders the 3D model into multi-view 2D images and 2) formulates 3D VTON as a consistent, unified 2D VTON process across these rendered views; 3) By reconstructing the edited images into a 3D model using existing 3D reconstruction methods, we ensure visual coherence and precise garment alignment from any viewing angle. Among them, the second step is crucial as existing 2D VTON methods [9, 26, 53] lack 3D knowledge, preventing them from generating multi-view images with 3D consistency. 

To address this challenge, we propose several novel techniques (Sec. 4.2) that equip a typical 2D VTON network (Sec. 4.1), which is built on a latent diffusion model [39], with the capability to generate 3D-consistent results. We use Gaussian Splatting [25] as our 3D representation. 

**==> picture [496 x 270] intentionally omitted <==**

Figure 2. **Overview of VTON 360.** Given an input 3D human model **G** src and a pair of garment images ( _gf_ , _gb_ ), our method 1) renders **G** src into multi-view 2D images (left) and 2) edits the rendered multi-view 2D images (middle); 3) reconstructs the edited images into a 3D model **G** VTON (right). In the crucial step 2), we propose three novel techniques to equip a typical 2D VTON network with the capability to generate 3D-consistent results: 1) Pseudo-3D Pose Input, 2) Multi-view Spatial Attention, and 3) Multi-view CLIP Embedding. 

**==> picture [237 x 140] intentionally omitted <==**

Figure 3. **DensePose (2D) vs. SMPL-X normal map (pseudo3D) representations.** DensePose applies uniform labels per body part, lacking 3D consistency across views and causing artifacts and temporal inconsistencies (highlighted with red boxes). In contrast, SMPL-X normal maps capture fine surface details, ensuring geometric coherence and stable, realistic shading across views. 

## **4.1. Recap of 2D VTON Framework** 

Following previous works [12, 26, 53], we formulate 2D VTON as an exemplar-based inpainting problem, aiming to fill an input clothing-agnostic image **A** with a given garment image _g_ , where **A** is obtained following the method used in [53]. As illustrated in Fig. 2 (middle), the network architecture is based on the latent diffusion model [39] with 

an encoder _E_ and comprises two components: 

- A GarmentNet [9, 53] that extracts features from _E_ ( _g_ ). 

- A Main UNet that inpaints **A** according to i) detailed garment features extracted by the GarmentNet; ii) the 2D pose of **A** represented by semantic labels using DensePose [14]; iii) CLIP embeddings of input garment _g_ . Among them, i) and ii) together with noise are input to the self-attention layers, while iii) is input to the crossattention layers of the Main UNet. 

- Both the GarmentNet and the Main UNet share the same network architecture. 

## **4.2. Multi-view 2D VTON with 3D Consistency** 

To enable the aforementioned 2D VTON model to generate multi-view and 3D-consistent results, we propose the following novel enhancements to its design: 

**Multi-view Inputs.** We extend both inputs to the model: 

- _Multi-view Garment Inputs:_ We extend the input garment representation from a single image _g_ to paired front and back view images _gf_ , _gb_ , providing comprehensive garment information across all viewing angles. Accordingly, we use the encoder _E_ to encode _gf , gb_ into their latent representations _E_ ( _gf_ ) _, E_ ( _gb_ ) and feed them into GarmentNet to obtain their multi-layer features _Ff[l]_[and] _[F][ l] b_[at][layer] _[l]_[,] respectively. 

- _Multi-view Clothing-agnostic Image Inputs:_ Based on 

the _equivalence_ between a 3D human model and its rendered multi-view 2D images, we extend the input human body representation from a single, clothing-agnostic image, **A** , to a set of _m_ multi-view images, denoted as **A1** _,_ **A2** _, ...,_ **Am** . These images are sampled from random azimuth angles, allowing the 2D VTON model to access comprehensive, multi-view information from the input 3D human model. 

**Pseudo-3D Pose Input.** As shown in Fig. 3, the 2D DensePose representations [14] commonly used in state-of-the-art 2D VTON methods [9, 26] assign a uniform semantic label to all pixels within each body part ( _e.g_ ., thigh), inherently lack 3D geometric consistency across multiple views, and often introduce artifacts and temporal inconsistencies. To address these limitations, we propose a novel pseudo-3D pose representation: the normal maps **N** derived from the SMPL-X [35] model of the input body. These normal maps capture fine-grained surface orientation details, providing a more consistent representation across views by preserving geometric structure in the 3D space. Furthermore, they enable smoother, temporally stable transitions and realistic shading effects across multi-view scenarios. In practice, we employ a lightweight PoseEncoder _E[′]_ [22] and feed _E[′]_ ( **N** ) into the Main UNet. We obtain the SMPL-X model from the multi-view images of the input body using EasyMoCap [1]. 

Accordingly, we concatenate three components as the enhanced input to the Main UNet: i) a noise latent _zt_ ; ii) the encoded pseudo-3D poses _E[′]_ ( **N1** ) _, E[′]_ ( **N2** ) _, ..., E[′]_ ( **Nm** ); and iii) the encoded multi-view clothing-agnostic images _E_ ( **A1** ) _, E_ ( **A2** ) _, ..., E_ ( **Am** ). Let _F_ 1 _[l][, F]_ 2 _[ l][, ..., F] m[ l]_[be][the][fea-] ture representations at layer _l_ of the Main UNet, and recall the garment features _Ff[l]_[and] _[ F][ l] b_[defined above, we enhance] the self-attention layers of the Main UNet as: 

**Multi-view Spatial Attention.** To cope with the aforementioned multi-view input features and ensure their consistency, we draw insights from the _temporal_ attention layer commonly used in video generation and editing [49, 58] and extend it to our multi-view _spatial_ attention layer, denoted as MVAttention. The key distinction of our MVAttention is that its input multi-view features _F_ 1 _[l][, F]_ 2 _[ l][, ..., F] m[ l]_[are from] images captured from non-uniform spatial intervals, with the viewing angles varying randomly. Consequently, features from similar views exhibit a higher correlation, while those from distinct views are largely independent. To model this relationship, we construct a “correlation” matrix _C_ based on the angular disparity obtained from camera rotation matrices of the input multi-view images, and define our MVAttention as follows: 

**==> picture [223 x 59] intentionally omitted <==**

where _i ∈{_ 1 _,_ 2 _, ..., m}_ denotes _i_ -th view; the Query _Q_ comes directly from **F[l]** and the concatenation of [ **F[l]** _, Ff[l][, F] b[ l]_[]][serves][as][the][key] _[K]_[and][the][value] _[V]_[ ;] _[⊕]_[indi-] cates matrix concatenation along the token axis; _d_ denotes the dimension; _W[Q]_ , _W[K]_ , _W[V]_ represent the linear transformation matrices; we omitted the _l_ of the attention matrices and parameters for simplicity; _C ∈_ R _[m][×][m]_ , _Ci_ represents _i_ -th row in _C_ , and its “correlation” value between _i_ -th and _j_ -th features is _Cij_ : 

**==> picture [194 x 13] intentionally omitted <==**

where _Ri_ and _Rj_ are the extrinsic rotation matrices of the corresponding camera views, (trace( _Ri[T][R][j]_[)] _[ −]_[1)] _[/]_[2][is][the] cosine value of the angle between these camera views. **Multi-view CLIP Embedding.** Camera viewpoints can serve as an effective condition signal to enhance 3D consistency in video content generation [52]. Building on this insight, we incorporate camera condition within our try-on network by encoding camera parameters as an additional token, enabling the generation of more consistent multi-view images. Specifically, we define a world coordinate system in which the camera faces the subject directly. For each input image (view) **Ai** , 1 _≤ i ≤ m_ , we extract the rotation matrix from the camera’s corresponding extrinsic matrix. This rotation matrix is then reshaped into a 9-dimensional tensor **ri** , which undergoes positional encoding to effectively integrate the camera parameters into the feature representation _Fi[c]_[.] 

**==> picture [187 x 29] intentionally omitted <==**

where _L_ is the length of positional embedding. We then project _Fi[c]_[to match the dimensionality of the garment CLIP] image embedding _F[g]_ via an MLP and concatenate them along the token axis to form _Yi_ . This combined representation, _Yi_ , is subsequently used in the key _K_ x and value _V_ x of the cross-attention layers of the Main UNet: 

**==> picture [208 x 57] intentionally omitted <==**

where **H[l]** is the output of the MVAttention of the _l_ -th layer; we omitted the _l_ of the cross attention matrices and parameters for simplicity. 

**Training.** Our enhanced multi-view 2D VTON network can be trained by minimizing the following latent diffusion model loss function: 

**==> picture [212 x 13] intentionally omitted <==**

**==> picture [237 x 159] intentionally omitted <==**

Figure 4. **Illustration of the proposed Multi-view Spatial Attention.** Query (Q): multi-view features **F[l]** ; Key (K) and Value (V): concatenation of **F[l]** and garment features _Ff[l]_[and] _[ F][ l] b_[.][The at-] tention score between viewpoints _i_ and _j_ is modulated by a weight _Cij_ , determined by the cosine of the angle between them. 

where _η_ = [ _E_ ( _gf_ ); _E_ ( _gb_ ); _E_ ( **Ni** ) _i[m]_ =1[]][represents][the][input] latent garment images and latent normal maps; _ζ_ = [ _E[′]_ ( **Ai** ) _i[m]_ =1[]][denotes][the][input][latent][clothing-agnoistic][im-] ages; _ψ_ = **Y** is the proposed multi-view CLIP embedding. 

## **5. Experiments** 

## **5.1. Experimental Setup** 

**Datasets.** We conduct experiments on two public datasets: Thuman2.0 [55] and MVHumanNet [51]. Thuman2.0 comprises 526 reconstructed clothed human scans, from which we render multi-view input images. Of these samples, 426 are used for training, while the remaining 100 are set aside for testing. To further evaluate the effectiveness and robustness of our method, we also perform experiments on MVHumanNet, a large-scale dataset of multi-view human images that encompasses a diverse range of subjects, daily outfits and motion sequences. The images in MVHumanNet are captured using a multi-view system with either 48 or 24 cameras. We use 4,990 subjects from this dataset, allocating 4,790 to training and 200 for tests. For each subject, we randomly select two frames of multi-view images from its entire motion sequence. While MVHumanNet provides multi-view images directly for editing and reconstruction, we render uniformly distributed views around each human subject in Thuman2.0 to ensure consistent input. 

**Baselines.** We primarily compare our method with three existing methods: DreamWaltz [23], GaussCtrl [48], and TIPEditor [63]. DreamWaltz is a method designed for directly generating 3D human bodies based on textual descriptions, while GaussCtrl and TIP-Editor are two radiance-based editing methods. GaussCtrl is based on Stable Diffusion, using a description-like prompt to edit the scene. TIP-Editor accepts both text and image prompts. We configure it by 

specifying the human body as the editing region and the desired garment as the image prompt. We use ChatGPT to generate the text prompts corresponding to the clothing images. 

**Evaluation Metrics.** For quantitative evaluation, we assess garment-to-person alignment between the edited person and the reference image. Following [63], we calculate the average DINO similarity [34] between the reference image and the rendered multi-view images of the edited 3D scene. Additionally, to evaluate multi-view consistency, we compute the CLIP Directional Consistency Score as outlined in [17]. Given the large scale of experiments (repeated 3DGS reconstruction), we selected a subset of examples from the dataset for metric evaluation. Specifically, from the test sets of Thuman and MVHumanNet, we randomly sampled 10 human scans each, performing virtual try-on with 6 randomly chosen garments per human scan. 

We further conducted a user study involving 50 participants who rated the results of our method and three baseline methods based on two criteria: overall “Quality” and “Alignment” with the reference image. Each evaluation consisted of two questions: (1) Which method produces the highest quality of the edited 3D human? and (2) Which method achieves the most consistent alignment with the target clothing? Participants viewed the VTON results as rotating randomized video sequences. 

**Implementation Details.** During pre-processing, we crop the multi-view images to the bounding box around the person and resize them to a resolution of 768 _×_ 576. The front view and the back view of garment images are obtained from the corresponding clothed human images. After editing, we pad the images back to their original size. The data processing pipeline is the same for both Thuman2.0 and MVHumanNet datasets. 

The Main UNet and the GarmentNet are initialized by the Stable Diffusion V1.5 [39]. The training process is divided into two stages. In the first stage, each view is trained independently, during which we establish the feature extraction capabilities of both the PoseEncoder and GarmentNet, as well as the generative capability of the Main UNet. The second stage involves multi-view training, where we randomly select _M_ views for each human subject. This stage is focused on training the proposed MVAttention module to enhance the network’s ability to maintain consistency across views. Due to memory constraints, we set _M_ = 8 for the training phase. During the testing phase, we uniformly sampled 32 views from a 360-degree rotation around the subject. The editing of these 32 views is conducted in two batches, with each batch processing _M_ = 16 views. 

## **5.2. Comparisons with State-of-the-Art Methods** 

**Qualitative Evaluation.** Fig. 5 shows visual comparisons between our method and the baselines. DreamWaltz [23] 

|Method<br>DreamWaltz [23]<br>TIP-Editor [63]<br>GaussCtrl [48]|Thuman2.0[55]<br>CLIP_cons ↑_<br>DINO_sim ↑_<br>Vote_quality_<br>Vote_align_<br>0.887<br>0.556<br>0.46%<br>1.54%<br>**0.939**<br>0.569<br>0.92%<br>0.62%<br>0.931<br>0.577<br>1.08%<br>1.38%|MVHumanNet[51]|
|---|---|---|
|||CLIP_cons ↑_<br>DINO_sim ↑_<br>Vote_quality_<br>Vote_align_|
|||0.935<br>0.495<br>0.46%<br>0.46%<br>**0.948**<br>0.512<br>2.15%<br>1.38%<br>0.938<br>0.521<br>1.69%<br>1.23%|
|Ours|0.923<br>**0.633**<br>**97.54**%<br>**96.46**%|0.933<br>**0.623**<br>**95.69**%<br>**96.92**%|



Table 1. **Quantitative comparisons.** CLIP _cons_ denotes the CLIP Direction Consistency Score. DINO _sim_ is the DINO similarity. 

**==> picture [446 x 325] intentionally omitted <==**

Figure 5. **Qualitative comparison.** The first two rows show the results on Thuman2.0 dataset while the last two rows show the results on MVHumaNet dataset. Our method achieves good texture preservation (highlighted by the blue boxes), while three baseline methods mostly fail. 

regenerates 3D clothed humans from text prompts but struggles to accurately retain both body and clothing characteristics. GaussCtrl [48], lacking support for image prompts, fails to maintain detailed clothing textures. While TipEditor [63] leverages Lora [21] for personalization, it encounters difficulties in consistently mapping clothing inputs from two views into the 3D human because the personalized concept are semantic in 2D space. In contrast, our method effectively preserves intricate clothing details, such as text, stripes, and logos. 

**Quantitative Evaluation.** Tab. 1 shows the results for the CLIP Directional Consistency Score and DINO similarity on Thuman2.0 and MVHumanNet datasets. Our approach surpasses other methodes on DINO _sim_ , clearly illustrating 

the superiority of our method in terms of garment texture preservation. While our results on CLIP _cons_ are comparable to those of other methods, it is important to note that these methods incorporate the SDS loss, which to some extent smooths the representation of humans in 3D space. Additionally, the ”flatter” textures of other methods could also result in artificially higher consistency scores. Furthermore, user studies have shown that our method significantly exceeds baselines in terms of edited 3D human quality and the alignment of clothing details. 

## **5.3. Visual Results using E-commerce Garment** 

Fig. 6 showcases VTON results using garments from the MVG dataset [46], whose images are from e-commerce 

**==> picture [237 x 200] intentionally omitted <==**

**----- Start of picture text -----**<br>
Original 3DGS<br>Garment Novel views rendered from edited 3DGS<br>**----- End of picture text -----**<br>


Figure 6. **Generalization to e-commerce garments (the MVG dataset).** Our method, trained on the THuman2.0 dataset, shows strong generalizability when applied to e-commerce garments. For clarity in visualization, we display garment images on human models; however, in the actual VTON process, the garments are segmented from the models using parse maps. 

platforms like YOOX NET-A-PORTER, Taobao, and TikTok[*] , and a model trained on the Thuman2.0 dataset [55]. The results demonstrate that our method effectively preserves intricate garment details and textures. For instance, it accurately maintains the stripe patterns in the first row, the cute tie in the second row, and the buttons in the third row, highlighting the robustness of our approach in handling diverse and realistic clothing items. 

## **5.4. Ablation Study** 

We conduct an ablation study on Thuman2.0 dataset in Tab. 2 and Fig. 7 to evaluate the impact of our three proposed modules in enhancing a typical 2D VTON network with 3D-consistent generation capabilities. Starting with the 2D VTON baseline [53] using DensePose, we progressively replace DensePose with our pseudo-3D pose, incorporate multi-view CLIP embeddings, and ultimately integrate MVAttention in the final configuration. Results in Tab. 2 indicate that each module contributes to metric improvements. Fig. 7 visualizes an example of multi-view image editing. The incorporation of pseudo-3D pose substantially improves limb generation compared to the 2D VTON baseline. Comparing rows 4 and 5, prior to the integration of multi-view CLIP embedding, the model captures limited spatial information, resulting in detail loss at specific angles (columns 3, 4, and 6). Finally, the proposed MVAttention achieves a more coherent generation across views. 

**==> picture [214 x 255] intentionally omitted <==**

**----- Start of picture text -----**<br>
Original<br>images<br>Garment<br>Baseline:<br>2D VTON<br>+ Pseudo -3D<br>Pose<br>+  Multi-view<br>CLIP Embedding<br>+  Multi-view<br>Spatial Attention<br>**----- End of picture text -----**<br>


Figure 7. **Visualization of the impact of the three proposed techniques on multi-view consistent editing.** The red boxes highlight the artifacts. Starting from the 2D VTON baseline, the pseudo-3D pose improves limb generation quality, multi-view CLIP embedding enhances detail across different viewing directions, and finally, MVAttention further strengthens consistency in the generated images. 

|Methods|CLIP_cons ↑_|DINO_sim ↑_|
|---|---|---|
||||
|2D-VTON<br>+ Pseudo-3D Pose<br>+ Multi-view CLIP Embedding<br>+ Multi-view Spatial Attention|0.892<br>0.910<br>0.913<br>**0.923**|0.609<br>0.626<br>0.631<br>**0.633**|



Table 2. **Ablation studies.** We ablate the impact of the three proposed techniques on Thuman2.0 dataset. 

## **6. Conclusions** 

In this work, we proposed VTON 360, a novel 3D Virtual Try-On (VTON) method that achieves high-fidelity VTON with the ability to render clothing from arbitrary viewing directions. Our method features a novel formulation of 3D VTON as an extension of 2D VTON that ensures 3D consistent results across multiple views. To bridge the gap between 2D VTON models and 3D consistency requirements, we introduce several key innovations, including multi-view inputs, pseudo-3D pose representation, multi-view spatial attention, and multi-view CLIP embedding. Extensive experiments demonstrate the effectiveness of our approach, significantly outperforming prior 3D VTON techniques in both fidelity and any-view rendering. 

*https://net-a-porter.com, www.taobao.com, www.douyin.com 

## **Acknowledgement** 

This work is supported in part by the National Key R&D Program of China under Grant No.2024YFB3908503, in part by the National Natural Science Foundation of China under Grant NO. 62322608 and in part by the CCFKuaishou Large Model Explorer Fund (NO. CCF-KuaiShou 2024007). 

## **References** 

- [1] Easymocap - make human motion capture easier. Github, 2021. 5 

- [2] Shuai Bai, Huiling Zhou, Zhikang Li, Chang Zhou, and Hongxia Yang. Single stage virtual try-on via deformable attention flows. In _European Conference on Computer Vision_ , pages 409–425. Springer, 2022. 2 

- [3] Bharat Lal Bhatnagar, Garvita Tiwari, Christian Theobalt, and Gerard Pons-Moll. Multi-garment net: Learning to dress 3d people from images. In _Proceedings of the IEEE/CVF international conference on computer vision_ , pages 5420– 5430, 2019. 2, 3 

- [4] Robert Bridson, Ronald Fedkiw, and John Anderson. Robust treatment of collisions, contact and friction for cloth animation. In _Proceedings of the 29th annual conference on Computer graphics and interactive techniques_ , pages 594–603, 2002. 3 

- [5] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 18392–18402, 2023. 3 

- [6] Haodong Chen, Yongle Huang, Haojian Huang, Xiangsheng Ge, and Dian Shao. Gaussianvton: 3d human virtual tryon via multi-stage gaussian splatting editing with image prompting. _arXiv preprint arXiv:2405.07472_ , 2024. 2, 3, 1 

- [7] Yiwen Chen, Zilong Chen, Chi Zhang, Feng Wang, Xiaofeng Yang, Yikai Wang, Zhongang Cai, Lei Yang, Huaping Liu, and Guosheng Lin. Gaussianeditor: Swift and controllable 3d editing with gaussian splatting. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 21476–21485, 2024. 3 

- [8] Seunghwan Choi, Sunghyun Park, Minsoo Lee, and Jaegul Choo. Viton-hd: High-resolution virtual try-on via misalignment-aware normalization. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 14131–14140, 2021. 1, 2 

- [9] Yisol Choi, Sangkyung Kwak, Kyungmin Lee, Hyungwon Choi, and Jinwoo Shin. Improving diffusion models for virtual try-on. _arXiv preprint arXiv:2403.05139_ , 2024. 3, 4, 5 

- [10] Xin Dong, Fuwei Zhao, Zhenyu Xie, Xijin Zhang, Daniel K Du, Min Zheng, Xiang Long, Xiaodan Liang, and Jianchao Yang. Dressing in the wild by watching dance videos. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 3480–3489, 2022. 2 

- [11] Yuying Ge, Yibing Song, Ruimao Zhang, Chongjian Ge, Wei Liu, and Ping Luo. Parser-free virtual try-on via distilling appearance flows. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 8485–8493, 2021. 2 

- [12] Junhong Gou, Siyu Sun, Jianfu Zhang, Jianlou Si, Chen Qian, and Liqing Zhang. Taming the power of diffusion models for high-quality virtual try-on with appearance flow. _arXiv preprint arXiv:2308.06101_ , 2023. 1, 2, 4 

- [13] Peng Guan, Loretta Reiss, David A Hirshberg, Alexander Weiss, and Michael J Black. Drape: Dressing any person. _ACM Transactions on Graphics (ToG)_ , 31(4):1–10, 2012. 1, 3 

- [14] Rıza Alp G¨uler, Natalia Neverova, and Iasonas Kokkinos. Densepose: Dense human pose estimation in the wild. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 7297–7306, 2018. 4, 5 

- [15] Fabian Hahn, Bernhard Thomaszewski, Stelian Coros, Robert W Sumner, Forrester Cole, Mark Meyer, Tony DeRose, and Markus Gross. Subspace clothing simulation using adaptive bases. _ACM Transactions on Graphics (TOG)_ , 33(4):1–9, 2014. 1, 3 

- [16] Xintong Han, Zuxuan Wu, Zhe Wu, Ruichi Yu, and Larry S Davis. Viton: An image-based virtual try-on network. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 7543–7552, 2018. 2 

- [17] Ayaan Haque, Matthew Tancik, Alexei A Efros, Aleksander Holynski, and Angjoo Kanazawa. Instruct-nerf2nerf: Editing 3d scenes with instructions. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 19740–19750, 2023. 3, 6, 1 

- [18] Sen He, Yi-Zhe Song, and Tao Xiang. Style-based global appearance flow for virtual try-on. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , pages 3470–3479, 2022. 2 

- [19] Zijian He, Peixin Chen, Guangrun Wang, Guanbin Li, Philip HS Torr, and Liang Lin. Wildvidfit: Video virtual tryon in the wild via image-based controlled diffusion models. In _European Conference on Computer Vision_ , pages 123– 139. Springer, 2024. 1, 2 

- [20] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. _Advances in Neural Information Processing Systems_ , 33:6840–6851, 2020. 3 

- [21] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. _arXiv preprint arXiv:2106.09685_ , 2021. 2, 3, 7 

- [22] Li Hu, Xin Gao, Peng Zhang, Ke Sun, Bang Zhang, and Liefeng Bo. Animate anyone: Consistent and controllable image-to-video synthesis for character animation. _arXiv preprint arXiv:2311.17117_ , 2023. 5 

- [23] Yukun Huang, Jianan Wang, Ailing Zeng, He Cao, Xianbiao Qi, Yukai Shi, Zheng-Jun Zha, and Lei Zhang. Dreamwaltz: Make a scene with complex 3d animatable avatars. _Advances in Neural Information Processing Systems_ , 36, 2024. 3, 6, 7 

- [24] Yangyi Huang, Hongwei Yi, Yuliang Xiu, Tingting Liao, Jiaxiang Tang, Deng Cai, and Justus Thies. Tech: Text-guided 

reconstruction of lifelike clothed humans. In _2024 International Conference on 3D Vision (3DV)_ , pages 1531–1542. IEEE, 2024. 3 

- [25] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. _ACM Trans. Graph._ , 42(4):139–1, 2023. 3, 1 

- [26] Jeongho Kim, Gyojung Gu, Minho Park, Sunghyun Park, and Jaegul Choo. Stableviton: Learning semantic correspondence with latent diffusion model for virtual try-on. _arXiv preprint arXiv:2312.01725_ , 2023. 3, 4, 5 

- [27] Diederik P Kingma. Auto-encoding variational bayes. _arXiv preprint arXiv:1312.6114_ , 2013. 3 

- [28] Zorah Lahner, Daniel Cremers, and Tony Tung. Deepwrinkles: Accurate and realistic clothing modeling. In _Proceedings of the European conference on computer vision (ECCV)_ , pages 667–684, 2018. 1, 3 

- [29] Sangyun Lee, Gyojung Gu, Sunghyun Park, Seunghwan Choi, and Jaegul Choo. High-resolution virtual try-on with misalignment and occlusion-handled conditions. In _Proceedings of the European conference on computer vision (ECCV)_ , 2022. 2 

- [30] Matthew Loper, Naureen Mahmood, Javier Romero, Gerard Pons-Moll, and Michael J Black. Smpl: A skinned multiperson linear model. In _Seminal Graphics Papers: Pushing the Boundaries, Volume 2_ , pages 851–866. 2023. 3 

- [31] Yifang Men, Yiming Mao, Yuning Jiang, Wei-Ying Ma, and Zhouhui Lian. Controllable person image synthesis with attribute-decomposed gan. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 5084–5093, 2020. 2 

- [32] Aymen Mir, Thiemo Alldieck, and Gerard Pons-Moll. Learning to transfer texture from clothing images to 3d humans. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 7023–7034, 2020. 2, 3 

- [33] Davide Morelli, Alberto Baldrati, Giuseppe Cartella, Marcella Cornia, Marco Bertini, and Rita Cucchiara. Ladi-vton: Latent diffusion textual-inversion enhanced virtual try-on. _arXiv preprint arXiv:2305.13501_ , 2023. 1, 2 

- [34] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. _arXiv preprint arXiv:2304.07193_ , 2023. 6 

- [35] Georgios Pavlakos, Vasileios Choutas, Nima Ghorbani, Timo Bolkart, Ahmed AA Osman, Dimitrios Tzionas, and Michael J Black. Expressive body capture: 3d hands, face, and body from a single image. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 10975–10985, 2019. 5 

- [36] Gerard Pons-Moll, Sergi Pujades, Sonny Hu, and Michael J Black. Clothcap: Seamless 4d clothing capture and retargeting. _ACM Transactions on Graphics (ToG)_ , 36(4):1–15, 2017. 3 

- [37] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. _arXiv preprint arXiv:2209.14988_ , 2022. 2, 3 

- [38] Yurui Ren, Xiaoqing Fan, Ge Li, Shan Liu, and Thomas H Li. Neural texture extraction and distribution for controllable person image synthesis. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 13535–13544, 2022. 2 

- [39] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 10684–10695, 2022. 1, 3, 4, 6 

- [40] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. _arXiv preprint arXiv:2208.12242_ , 2022. 3 

- [41] Igor Santesteban, Nils Thuerey, Miguel A Otaduy, and Dan Casas. Self-supervised collision handling via generative 3d garment models for virtual try-on. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 11763–11773, 2021. 2 

- [42] Igor Santesteban, Miguel Otaduy, Nils Thuerey, and Dan Casas. Ulnef: Untangled layered neural fields for mix-andmatch virtual try-on. _Advances in Neural Information Processing Systems_ , 35:12110–12125, 2022. 2 

- [43] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In _International Conference on Machine Learning_ , pages 2256–2265. PMLR, 2015. 3 

- [44] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. _arXiv preprint arXiv:2010.02502_ , 2020. 3 

- [45] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. _Advances in Neural Information Processing Systems_ , 32, 2019. 3 

- [46] Haoyu Wang, Zhilu Zhang, Donglin Di, Shiliang Zhang, and Wangmeng Zuo. Mv-vton: Multi-view virtual try-on with diffusion models. _arXiv preprint arXiv:2404.17364_ , 2024. 7 

- [47] Junjie Wang, Jiemin Fang, Xiaopeng Zhang, Lingxi Xie, and Qi Tian. Gaussianeditor: Editing 3d gaussians delicately with text instructions. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 20902–20911, 2024. 3 

- [48] Jing Wu, Jia-Wang Bian, Xinghui Li, Guangrun Wang, Ian Reid, Philip Torr, and Victor Adrian Prisacariu. Gaussctrl: multi-view consistent text-driven 3d gaussian splatting editing. _arXiv preprint arXiv:2403.08733_ , 2024. 3, 6, 7 

- [49] Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Stan Weixian Lei, Yuchao Gu, Yufei Shi, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 7623–7633, 2023. 5 

- [50] Zhenyu Xie, Haoye Dong, Yufei Gao, Zehua Ma, and Xiaodan Liang. Dreamvton: Customizing 3d virtual tryon with personalized diffusion models. _arXiv preprint arXiv:2407.16511_ , 2024. 2, 3 

- [51] Zhangyang Xiong, Chenghong Li, Kenkun Liu, Hongjie Liao, Jianqiao Hu, Junyi Zhu, Shuliang Ning, Lingteng Qiu, 

   - Chongjie Wang, Shijie Wang, et al. Mvhumannet: A largescale dataset of multi-view daily dressing human captures. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 19801–19811, 2024. 2, 6, 7 

- [52] Dejia Xu, Weili Nie, Chao Liu, Sifei Liu, Jan Kautz, Zhangyang Wang, and Arash Vahdat. Camco: Cameracontrollable 3d-consistent image-to-video generation. _arXiv preprint arXiv:2406.02509_ , 2024. 5 

- [53] Yuhao Xu, Tao Gu, Weifeng Chen, and Chengcai Chen. Ootdiffusion: Outfitting fusion based latent diffusion for controllable virtual try-on. _arXiv preprint arXiv:2403.01779_ , 2024. 3, 4, 8 

- [54] Han Yang, Xinrui Yu, and Ziwei Liu. Full-range virtual try-on with recurrent tri-level transform. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , pages 3460–3469, 2022. 2 

- [55] Tao Yu, Zerong Zheng, Kaiwen Guo, Pengpeng Liu, Qionghai Dai, and Yebin Liu. Function4d: Real-time human volumetric capture from very sparse consumer rgbd sensors. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 5746–5756, 2021. 2, 6, 7, 8 

- [56] Jinsong Zhang, Kun Li, Yu-Kun Lai, and Jingyu Yang. Pise: Person image synthesis and editing with decoupled gan. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 7982–7990, 2021. 2 

- [57] Lvmin Zhang and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. _arXiv preprint arXiv:2302.05543_ , 2023. 3 

- [58] Yabo Zhang, Yuxiang Wei, Dongsheng Jiang, Xiaopeng Zhang, Wangmeng Zuo, and Qi Tian. Controlvideo: Training-free controllable text-to-video generation. _arXiv preprint arXiv:2305.13077_ , 2023. 5 

- [59] Fuwei Zhao, Zhenyu Xie, Michael Kampffmeyer, Haoye Dong, Songfang Han, Tianxiang Zheng, Tao Zhang, and Xiaodan Liang. M3d-vton: A monocular-to-3d virtual tryon network. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 13239–13249, 2021. 3 

- [60] Luyang Zhu, Dawei Yang, Tyler Zhu, Fitsum Reda, William Chan, Chitwan Saharia, Mohammad Norouzi, and Ira Kemelmacher-Shlizerman. Tryondiffusion: A tale of two unets. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 4606–4615, 2023. 2 

- [61] Jingyu Zhuang, Chen Wang, Liang Lin, Lingjie Liu, and Guanbin Li. Dreameditor: Text-driven 3d scene editing with neural fields. In _SIGGRAPH Asia 2023 Conference Papers_ , pages 1–10, 2023. 3 

- [62] Jingyu Zhuang, Di Kang, Linchao Bao, Liang Lin, and Guanbin Li. Dagsm: Disentangled avatar generation with gs-enhanced mesh. _arXiv preprint arXiv:2411.15205_ , 2024. 3 

- [63] Jingyu Zhuang, Di Kang, Yan-Pei Cao, Guanbin Li, Liang Lin, and Ying Shan. Tip-editor: An accurate 3d editor following both text-prompts and image-prompts. _arXiv preprint arXiv:2401.14828_ , 2024. 3, 6, 7, 1 

## **VTON 360: High-Fidelity Virtual Try-On from Any Viewing Direction** Supplementary Material 

Appendix A introduces the preliminaries of 3DGS. The detailed formulations of the two quantitative metrics are presented in Appendix B. Additionally, Appendix C outlines the post-processing techniques applied to ensure the preservation of human characteristics in image editing. Appendix D elaborates on the failure cases and proposes a mitigation strategy to address it. Finally, Appendix E showcases additional VTON results, including those from a real 3D scene used in GaussianVTON [6]. 

## **A. 3D Representation: Gaussian Splatting** 

3D Gaussian Splatting (3DGS) [25] has emerged as a prominent technique in 3D reconstruction due to its ability to render high-quality scenes in real-time. Unlike traditional point cloud based methods, which directly represent scenes as discrete points, 3DGS models each point as a continuous Gaussian function _gi_ : 

**==> picture [193 x 14] intentionally omitted <==**

where _x_ is the position vector of _gi_ , _µi ∈_ R[3] and _Σi ∈_ R[3] _[×]_[3] are _gi_ ’s mean and covariance matrix, respectively. Then, _gi_ is projected onto a 2D image plane to facilitate rendering. This projection yields a new mean vector _µi[′] ∈_ R[2] and an updated covariance matrix _Σi[′][∈]_[R][2] _[×]_[2][ defined as:] 

**==> picture [200 x 12] intentionally omitted <==**

where _J_ is the Jacobian matrix derived from the affine approximation of the perspective projection, _T_ and _K_ denote the extrinsic and intrinsic matrices, respectively. Given the color _ci_ and opacity _αi_ at the Gaussian center point, the rendered color at a 2D pixel _p_ is calculated as follows: 

**==> picture [185 x 67] intentionally omitted <==**

where _Ti_ denotes the cumulative transmission along the ray. 

## **B. Metrics** 

In the quantitative evaluation, we employ two metrics: 

- Average DINO Similarity [63], which measures the alignment between the garment image and the edited 3D human. 

- CLIP Directional Consistency Score [17], which evaluates multi-view consistency. 

Specifically, given an edited 3D human (after VTON), 120 views are uniformly projected around its central axis. These views are divided into three categories based on orientation: _Sf_ , _Sb_ , and _Ss_ , corresponding to 40 front views, 40 back views, and 40 side views, respectively. Let _D_ ( _·_ ) represent the normalized DINO embedding and _C_ ( _·_ ) denote the normalized CLIP embedding. Using these, we formally define the two metrics as follows: 

**==> picture [233 x 63] intentionally omitted <==**

where _ei_ , _ei_ +1 and _oi_ , _oi_ +1 denotes the two consecutive novel views from the edited 3DGS and the original 3DGS, respectively. 

## **C. Post-processing** 

The clothing-agnostic maps **A** often mask parts of the face and hair, particularly for females. Due to the inherent properties of the diffusion model, it is unable to fully restore the intricate details of these masked regions. To ensure high-fidelity preservation of human characteristics, we apply a post-processing step where, after editing the rendered views, we “copy” the face and hair from the original image _o_ onto the edited image _e_ . Specifically, let _m_ represent the region corresponding to the face and hair, which can be extracted from the parsed map during pre-processing, we implement post-processing as: 

**==> picture [167 x 11] intentionally omitted <==**

**==> picture [237 x 67] intentionally omitted <==**

Figure 8. Our multi-view editing may fail in certain views with complex poses (red box in pink background) but these views can be automatically discarded to mitigate their impact on 3D VTON (blue background). 

## **D. Limitations** 

As shown in Fig. 8, our method may fail in certain views with complex postures. To address this, we use Z-Score 

Normalization to automatically identify and discard problematic views based on the view reconstruction loss during the process of lifting multiple views to 3D space, mitigating their adverse impact. 

## **E. Additional Visualization Results** 

Fig. 9 illustrates additional VTON results. The first two rows showcase results from the THuman2.0 dataset; the middle two rows showcase results from the MVHumanNet dataset. To further demonstrate the effectiveness of our method, we apply it on a real 3D scene used in GaussianVTON [6]. The last two rows in Fig. 9 illustrate these VTON results with the model trained on Thuman2.0 dataset. Despite the data gap, including w/wo background and unseen camera poses, our method exhibits robust performance and preserves the details of the clothing well. 

**==> picture [496 x 624] intentionally omitted <==**

Figure 9. **Additional visualization results.** The first, middle, and last two rows show results on Thuman2.0, MVHumanNet, and a real 3D scene used in GaussianVTON, respectively. 

