---
type: paper-fulltext
slug: dreamvvt-mastering-realistic-video-virtual-try-on-in-the-wild-via-a-stage-wise-diffusion-transformer-framework
arxiv_id: null
source_path: /Users/saris.kia.adm/.paper-scholar/dreamvvt-mastering-realistic-video-virtual-try-on-in-the-wild-via-a-stage-wise-diffusion-transformer-framework/2508.02807.md
paper: "[[dreamvvt-mastering-realistic-video-virtual-try-on-in-the-wild-via-a-stage-wise-diffusion-transformer-framework]]"
---
<!-- extracted by afk_extract from 2508.02807.pdf (18p) -->

# **DreamVVT: Mastering Realistic Video Virtual Try-On in the Wild via a Stage-Wise Diffusion Transformer Framework** 

Tongchun Zuo[1] _[⋆]_ Zaiyu Huang[1] _[⋆]_ Shuliang Ning[1] Ente Lin[2] Chao Liang[1] Zerong Zheng[1] Jianwen Jiang[1] Yuan Zhang[1] Mingyuan Gao[1] Xin Dong[1] _[†]_ 1ByteDance Intelligent Creation, 2Shenzhen International Graduate School, Tsinghua University ztcustc@mail.ustc.edu.cn, huangzaiyu,liangchao,dongxin.1016@bytedance.com, linet22@mails.tsinghua.edu.cn, shuliangning@link.cuhk.edu.cn, zrzheng1995@foxmail.com, jianwen.alan,zhang.yuan09,gaomingyuan001@gmail.com 

**==> picture [496 x 287] intentionally omitted <==**

**----- Start of picture text -----**<br>
Input Video & Garment Synthetic Video Input Video & Garment Synthetic Video<br>**----- End of picture text -----**<br>


Figure 1. DreamVVT can generate high-fidelity and temporally coherent virtual try-on videos for diverse garments and in unconstrained scenarios. Specifically, **the first row** shows its ability to handle complex human motions like runway walks and 360-degree rotations; **the second row** illustrates robustness to complex backgrounds and challenging camera movements; **the third row** highlights visually coherent try-on results for cartoon characters with real garments. 

## **Abstract** 

_Video virtual try-on (VVT) technology has garnered considerable academic interest owing to its promising applications in e-commerce advertising and entertainment. However, most existing end-to-end methods rely heavily on scarce paired garment-centric datasets and fail to effectively leverage priors of advanced visual models and test-_ 

> _⋆_ Equal contribution. 

> _†_ Project leader. 

_time inputs, making it challenging to accurately preserve fine-grained garment details and maintain temporal consistency in unconstrained scenarios. To address these challenges, we propose_ _**DreamVVT** , a carefully designed twostage framework built upon Diffusion Transformers (DiTs), which is inherently capable of leveraging diverse unpaired human-centric data to enhance adaptability in real-world scenarios. To further leverage prior knowledge from pretrained models and test-time inputs, in the first stage, we sample representative frames from the input video and uti-_ 

1 

_lize a multi-frame try-on model integrated with a visionlanguage model (VLM), to synthesize high-fidelity and semantically consistent keyframe try-on images. These images serve as complementary appearance guidance for subsequent video generation._ _**In the second stage** , skeleton maps together with fine-grained motion and appearance descriptions are extracted from the input content, and these along with the keyframe try-on images are then fed into a pretrained video generation model enhanced with LoRA adapters. This ensures long-term temporal coherence for unseen regions and enables highly plausible dynamic motions. Extensive quantitative and qualitative experiments demonstrate that DreamVVT surpasses existing methods in preserving detailed garment content and temporal stability in real-world scenarios. Our project page_ https: //virtu-lab.github.io/ 

## **1. Introduction** 

With the scaling up of training data and model parameters, Diffusion Transformer (DiT)-based models have achieved remarkable progress in various visual generation tasks, including text-to-image [10, 24] and image-to-video generation [29, 43]. As a prominent downstream task, video virtual try-on (VVT) aims to faithfully render arbitrary garments onto characters within video sequences, as shown in Figure 1. Recently, it has attracted considerable attention from the research community [7, 21, 26, 56] owing to its broad application potential in promising domains such as e-commerce and entertainment. 

Despite considerable efforts, existing methods[3, 7, 11, 25, 26, 28, 51, 56] still struggle to accurately preserve finegrained garment details and maintain temporal consistency in unconstrained scenarios, such as complex subject or camera motion, dynamic scenes, and diverse character styles. We posit that these limitations stem primarily from the reliance on an end-to-end training paradigm, which inherently constrains the effective exploitation of unpaired data, priors of advanced visual models, and additional information at inference stage. **First** , these methods exhibit a strong reliance on insufficient paired clothing-video data [9, 11], most of which are collected in homogeneous indoor environments. This often results in reduced garment visual fidelity and increased temporal instability, particularly for arbitrary garments and complex video inputs. Furthermore, collecting large-scale paired clothing–video datasets across diverse real-world scenarios remains extremely challenging. **Second** , these methods[7, 26, 56] adapt a pretrained text-tovideo generation model to deform spatially misaligned garment images onto the person frame by frame. However, this approach disrupts the pretrained model’s inherent capacity for smooth spatiotemporal modeling, making model convergence more challenging. Additionally, performing full fine-tuning of all parameters [3, 26, 56] in a pretrained 

model with a limited amount of data is prone to disrupting the pretrained priors, which in turn degrades the quality and temporal stability of the generated videos. Even when a substantial portion of model parameters are trained on large-scale datasets and diverse video tasks, unified video creation and editing methods [18, 21, 52] still struggle to accurately preserve garment details and maintain temporal coherence, primarily due to the lack of task-specific design for virtual try-on. **Third** , at the inference stage, providing only front-view garment images to guide the virtual tryon process often leads to implausible outcomes for invisible regions when the person turns or the camera viewpoint changes significantly. 

To tackle these issues, we introduce **DreamVVT** , an improved stage-wise framework built upon Diffusion Transformers (DiTs), which is inherently capable of leveraging unpaired human-centric data from diverse sources to improve generalization in real-world scenarios. To further exploit prior information from pretrained models and the inference process, in the first stage, we first sample keyframes with significant motion changes from input video, a visionlanguage model (VLM) is then employed to generate textual descriptions that map the input garment to each keyframe. These descriptions, along with garment images and other relevant conditions, are provided to a multi-frame try-on model equipped with LoRA [15] adapters, resulting in highfidelity and semantically consistent try-on images for each keyframe. These images serve as complementary appearance guidance for subsequent video generation. In the second stage, we employ a temporally smooth pose guider for skeletal feature encoding and utilize an advanced video large language model (video LLM) to extract fine-grained action descriptions and other high-level visual information from input content. These features, together with the spatially aligned keyframe try-on images, are subsequently provided as inputs to a pretrained video generation model enhanced with LoRA adapters. By leveraging the pretrained priors of large-scale video generation models, this model is endowed with enhanced generalization capability in in-thewild scenarios. Moreover, incorporating multiple keyframe try-on images and precise motion guidance enables the generation of long-term virtual try-on videos that exhibit strong temporal consistency and highly plausible dynamic motions. Additionally, a multi-task learning strategy is introduced to maintain the controllability of conditions from different modalities. 

In summary, the contributions of this paper are threefold: 

- We present a carefully designed stage-wise framework based on DiTs, which is capable of leveraging unpaired data, priors from advanced visual models and test-time inputs to enhance virtual try-on performance in real-world scenarios. 

- We propose integrating keyframe try-on with video LLM 

2 

reasoning, which supplies abundant appearance and motion information for video generation and ensures both garment detail preservation and temporal consistency. 

- Extensive experiments demonstrate that DreamVVT outperforms current existing methods in preserving highfidelity garment details and ensuring temporal stability across diverse scenarios. 

## **2. Related work** 

## **2.1. Image-based Virtual Try-on** 

A series of works[13, 14, 23, 30, 36, 44, 50, 54] that build on powerful SDXL[38] and Flux[24], achieve a significant improvement in realism of the synthesis results compared to previous GAN-based methods[5, 49]. However, these methods often failed to preserve the fine-grained detail of the garment image because they heavily relies on the garment feature extracted by a pre-trained image clip encoder. To address these problems, some other pioneering approaches[41, 50] proposed to introduce a parallel reference U-net architecture to effectively extract the reference tokens and then merge with the original tokens through the attention process. Dreamfit[30] introduces a plug-andplay fine-tuning mechanism by leveraging LoRA[15] and parallel architecture. While CatVTON[6] presents a lightweighted approach, which replaces the parallel architecture with a shared-weighted single one and directly injects the conditional image through input concatenation. Although these methods demonstrate a superior performance in garment preservation, most of them are limited in conventional single-view virtual try-on settings without fully exploring the multi-images generation ability of large diffusion model. Our method takes a step forward focus on instruction-guided key frame try-on image generation, which expands the border of try-on applications and further facilitates the usage of generated results to support downstream tasks like video try-on. 

## **2.2. Video-based Virtual Try-on** 

In recent works, [11, 25, 51] employ a pretrained inpainting U-Net as the primary branch, complemented by a reference U-Net to capture fine-grained garment features. Temporal consistency is further enhanced by introducing standard temporal attention modules after each stage of the main U-Net. Owing to their intrinsically decoupled spatiotemporal architecture, these approaches persistently exhibit pronounced temporal instability in try-on videos featuring complex garments. To alleviate these issues, [28] encompasses a Clothing & Temporal Consistency strategy based on Stable Video Diffusion [3]. [7, 26, 56] adopt the Diffusion Transformer [37] backbone with full attention mechanisms to enhance temporal consistency, and further propose more efficient training strategies. Despite their promise, these single-stage methods are limited by their 

inability to leverage unpaired videos or images across a wide range of scenarios and their tendency to disrupt pretrained priors. Consequently, they struggle to simultaneously maintain visual fidelity and motion consistency, particularly when the input videos involve camera movements, dynamic backgrounds, or complex human actions. In this work, we divide the VVT task into two sequential stages, enabling efficient use of unpaired videos and images from diverse sources and fully exploiting the priors of pretrained models, thereby achieving significant improvements in unconstrained scenarios. 

## **2.3. Pose guided Human Video generation** 

GAN-based methods[33, 39, 40] and diffusion-based approaches[16, 17, 34, 45, 55] form the main-stream implementation paradigm of human video generation in recent years, which mostly contains an individual reference encoder responsible for extracting image features and a pose guider for injecting motion signal. In particular, AnimateAnyone[16] first introduces the popular parallel u-net architecture based on the stable diffusion model. Mimo[34] subsequently proposes to decouple the background and foreground signal by adopting an expert encoder for different conditional inputs. AnimateAnyone2[17] focus on character-environment integration and pose robustness through a new environment and pose formulation strategy. We observe that methods using an entire image as a reference frequently degrade garment details, especially when the subject or camera undergoes large rotations. Therefore, to achieve high-fidelity video try-on, we develop a video editing model guided by multiple try-on keyframes, incorporating human tracking-based cropping and multitask learning. 

## **3. Methodology** 

Our DreamVVT adopts a stage-wise framework based on large-scale Diffusion Transformers to achieve high-fidelity virtual try-on video generation in unconstrained scenarios. As illustrated in Figure 2, it consists of two sequential stages. In the first stage, we sample frames with notable motion variations from a input video as keyframes, and then a multi-frame try-on model is developed to fit garment images onto these keyframes while maintaining content consistency and preserving fine details. In the second stage, we present a modified video generation model, which synthesizes a plausible try-on video conditioned on keyframe try-on images, pose features, and textual descriptions. We further elaborate on each specific module within our framework, emphasizing its importance and role in the overall framework. 

3 

**==> picture [496 x 299] intentionally omitted <==**

**----- Start of picture text -----**<br>
2)34,5%+67,*)0, "K)<(1"J,.+%)1"<br>6%$8,#/09,:#++5+:,<br>:6+)$:8%($<br>;(/0$,5%+67,,*)0,<br>6%$8,#/09,:#++5+:,<br>:6+)$:8%($,6%$8,<br>68%$+,;#/6+(:<, >/?@<br>:"0,1+:%90<=<br>0*$1#2,34+1536+7#8+*93:%9;)&3<)%3=#9<%$/#>3<br>A%1+/,A@B,B03/1+(<br>!""#$%$&’# !"#$%"&’()*"+,)"-%,./"-$01%"23’-%./4"<br>&%33’-"5’3’.%6"1%71"/-%,1/$0.14",)6"8 ,C)$38%;.<br>()*+)&,  9%"0)010,33&"/1,)6/"2,50)*"1$%"5,+%.,"<br>.%3,7%6:"9%"-,3;/"1’-,.6/4"/1’</4"1(.)/"/30*$13&4"8 2<br>-&.+%)&/#&* ! "#$%"=06%’"0/"/%1"0)","/<,50’(/".’’+4":: ?1%71 ?!"#<br>! #%7L>C" #%7??N" !<br>K+*L>C K+*??N<br>?$!%<br>G’MD G’MD<br>! C06L>C C06??N<br>OOF0# P N<br>,/&01’,!*-).,, 0*$1#?,3(@8*+;/)7$83A@+7#73B+%*@$83:%9;)&3B+7#)3A#&#%$*+)&33 $%&’()’*+,!*-).,,<br>!"#<br>>%&"?.,+%/" @,+<30)* !"#$%&’()*+ ,-(.&/0,!/1+#<br>GGO<br>""C06%’"<br>H’/%" "J(06%.<br>B03/1+(<br>A%1+/,A@B, "H,15$02&<br>$)3456’’)&’*.&<br>#.,5;0)* ,)6 I.’<<0)*<br>A)<,15$02&""B"CDE"F%5’6%." G,<3,50,)"H&.,+06""?(/0’)<br>**----- End of picture text -----**<br>


Figure 2. Overview of DreamVVT. The framework comprises two sequential stages: the first stage selects frames with significant motion changes and generates try-on images for these keyframes, while the second stage synthesizes the final virtual try-on video using finegrained motion guidance and complementary appearance cues. 

## **3.1. Input conditions** 

**Pose Conditions.** To enable virtual try-on for both real and cartoon characters, we adopt RTMPose [20] as a robust and efficient pose representation. Given that the character in the raw video may occupies only a limited spatial area, directly feeding the pose sequence at its original resolution into the try-on model can result in significant garment detail loss in the generated video, primarily due to spatial downsampling. To address this issue, We begin by cropping each frame with tracking bounding boxes of uniform width and height, thereby isolating the character region in each frame. **Agnostic Masks.** Most previous methods [7, 26, 51] generate agnostic masks by directly dilating the segmented clothing regions in the video, which can easily cause leakage of the original garment style cues. To mitigate this problem, we employ the human bounding box together with the dilated pose skeleton to generate clothing-agnostic masks that effectively prevent information leakage while preserving as much of the original background as possible. 

**Agnostic Images.** By applying the agnostic masks, we occlude the garment regions in the input character video or image, thereby generating agnostic images. 

**Garment Images.** For garment image input, we firstly employ a saliency segmentation detection model[57] to extract 

the foreground area and then remove the background region by filling it with white pixels. To further promote the preservation of garment details, we calculate a tight bounding box according to the extracted segmentation and subsequently crop the region of interest. Finally, the cropped image is resized to a specific resolution before being fed into the network. 

## **3.2. Stage1: High Fidelity Try-on for Keyframes** 

## **3.2.1. Keyframe Sampling** 

We select frames with significant motion changes to provide more comprehensive guidance for video generation. Initially, given that the majority of input garment images are captured from a frontal perspective, we predefine a frontalview person image in an A-pose as the anchor frame. Subsequently, we compute the motion similarity between each video frame and the anchor frame by measuring the cosine distance between their respective skeletal joint direction vectors. This similarity is further weighted by the area ratio of the subject to the entire frame, to produce the final score. Lastly, frames are sorted in descending order by their final scores, and a reverse-order search constrained by a minimum score interval is performed to obtain a set of key images with minimal informational redundancy. The details 

4 

**==> picture [496 x 281] intentionally omitted <==**

**----- Start of picture text -----**<br>
VidGmn e  [�6] TON Magic yon OVidGs S or mn C [�6] TON Magic yon Ous<br>**----- End of picture text -----**<br>


Figure 3. Qualitative comparison on the ViViD dataset.Please zoom in for more details. 

are provided in the Supplementary Material. 

## **3.3. Stage2: Multi-modal Guided Virtual Try-on Video Generation** 

## **3.2.2. Multi-frame Try-on Model** 

Given selected keyframes, we leverage a diffusion transformer G featuring a minimal set of learnable parameters on a pre-trained Seedream[12] model G to generate the final multi-frames try-on results. We modify each MMDiT blocks[10] within G by integrating the attention modules with pluggable LoRA[15] layers, and introduce an additional parameter-sharing network branch to process reference image input following implementations of [30]. Notably, G takes as input the multi-keyframes imagecondition pairs alongside an elaborately designed consistent image instruction, which clarifies different input components and helps guide the model towards synthesizing desirable outcomes. Specifically, we first tokenize each conditional input via a parallel network architecture to align different modalities, then jointly aggregate information across keyframes through Q , K , V catenation during the attention process. This mechanism ensures robust information interaction between each conditional input and keyframes intermediate features, thereby enabling the coherent multiframes try-on results with consistent details. For text inputs, we resort to [42] for detailed descriptions, including garment category, material, and patterns for each keyframe. Subsequently, a text alignment procedure is introduced by asking VLM to rewrite and gather all the text results, which further reinforces the consistency of keyframe descriptions. 

Our virtual video try-on model is based on a pretrained image-to-video generation framework [32] that employs stacked sequential MMDiT blocks [10], each of which integrates text and video streams. To accurately reconstruct body movements in the input video, we extract the corresponding 2D skeleton sequences. After cropping, a tailored pose guider with temporal attention transforms the framewise skeleton maps into temporally smoothed pose latents that match the resolution of the noise latents. Similarly, the cropped agnostic images are fed into the video VAE encoder to obtain agnostic latents, and the cropped agnostic mask is resized to the same resolution as the agnostic latents. Subsequently, the agnostic latents, resized agnostic masks, noise latents, and pose latents are concatenated along the channel dimension and patchified into video tokens, denoted as F vid ∈ R[l][v][c] (where lv = t h w , and t = T[=] H[=] W[T, H, W][is][the][shape] 4[, h] 16[, w] 16[,] of input video) Furthermore, since pose skeletons capture only coarse-grained body motion and cannot fully represent fine-grained garment interactions, we employ Qwen2.5VL[2] to extract an attribute-disentangled textual description, which contains detailed motion descriptions and highlevel visual information(During inference, the appearancerelated descriptions are replaced with those corresponding to the target garment). These textual descriptions are then 

5 

!"#$%&’()*&"’’’’’’ +,$-&./’’’’’’’’’’0,/![!] 123’’’’’’4,5)%61$7".’’’’’’8"9(:0;’’’’’’’’’’’’2#$< 

!"#$%&’()*&"’’’’’’ +,$-&./’’’’’’’’’’0,/![!] 123’’’’’’4,5)%61$7".’’’’’’8"9(:0;’’’’’’’’’’’’2#$< 

Figure 4. Qualitative results on our Wild-TryOn Benchmark. Please zoom in for more detail. Additional comparison results are provided in the supplementary material. 

processed by a Qwen LLM [1] into text tokens, denoted as F text ∈ R[l][t][c][t] . For the appearance branch, keyframe tryon images are first processed frame by frame by the video VAE encoder to extract image latents, which are then transformed into image tokens, denoted as F img ∈ R[l][i][c] (where li = k h w and k is the number of keyframes). To preserve the spatiotemporal modeling and prompt adherence capabilities of the model, we freeze the parameters of the text streams. Lightweight LoRA adapters, comprising only 10% of the trainable parameters, are inserted into the frozen video streams and image streams that are directly duplicated from the video streams, with shared memory. As the channel dimensions of the video and image tokens have increased, the input projection layers for the video and image streams are set to be trainable. Finally, all these token sets are processed by their respective QKV projection layers and then concatenated along the l dimension. The resulting sequence is fed into a full self-attention module, which enables the model to adaptively align visual content with textual descriptions across both spatial and temporal dimensions. Following the self-attention operation, the joint tokens are demultiplexed by index into text, image, and video tokens, which are subsequently processed by the following DiT blocks. After several denoising iterations within the DiT backbone, the network generates the try-on video to- 

||Dataset<br>ViViD<br>Method<br>VFID_p_<br>_I_ ↓VFID_p_<br>_R_ ↓VFID_u_<br>_I_ ↓VFID_u_<br>_R_ ↓SSIM↑LPIPS↓|
|---|---|
||ViViD<br>17.2924 0.6209 21.8032 0.8212 0.8029 0.1221<br>CatV2TON 13.5962 0.2963 19.5131 0.5283 0.8727 0.0639<br>MagicTryON 12.1988 **0.2346** 17.5710 0.5073 **0.8841**0.0815<br>DreamVVT **11.0180** 0.2549 **16.9468 0.4285** 0.8737**0.0619**|



Table 1. Quantitative comparisons on ViViD dataset. The best results are demonstrated on **bold** . 

kens, which are then decoded into video sequences by the Video VAE decoder. An efficient laplacian pyramid fusion method is subsequently applied to seamlessly blend the generated try-on video into the corresponding regions of the original video. During training, we introduce a multi-task learning strategy similar to [19, 31], wherein one task (e.g., text to video, pose with text and keyframes to video) is randomly selected based on a predefined probabilistic schedule, in order to fully exploit the complementary advantages of various modalities. 

6 

**==> picture [496 x 198] intentionally omitted <==**

**----- Start of picture text -----**<br>
!"#$%3&’()*3+3,-*.#% !"#$%&’ /0"%1)%’234’()* !"#$% &’()* + ,-*.#% !"#$%&’ /0"%1)%’234’()*<br>!"#$%&’ :+$,-+./"0&+6,5;0+.#"6,"3+.,8/-2+&)*+8,$*",4+&)*-+&1"-0+,"#1-3+,-3+6,5;0+,6,<+6/&)+)/0+=,8;+&#+&)*+8,$*",9 !"#$%&’ ()*+$,-+./"0&+&122*3+,&+&)*+)*$+#.+)/0+0)/"&4+&)*-+%155*3+#-+)/0+06*,&%,-&0+&#+3*$#-0&",&*+&)*/"+*78*55*-&+0&"*&8)/-*009<br>!"#$%&’ :+$,-+./"0&+6,5;0+.#"6,"3+ !"#$%&’ ()*+$,-+./"0&+&122*3+,&+&)*+<br>.,8/-2+&)*+8,$*",4+&)*-+&1"-0+,"#1-3+ )*$+#.+)/0+0)/"&4+&)*-+%155*3+#-+)/0+<br>,-3+6,5;0+,6,<+6/&)+)/0+=,8;+&#+&)*+ 06*,&%,-&0+&#+3*$#-0&",&*+&)*/"+<br>8,$*",9 *78*55*-&+0&"*&8)/-*009<br>3"%4’%3/’5$67+(,$- !"#$%&#$’’./01’(2(*%$+-<br>3"%4 /#$ 5$67+(,$<br>!"#$%&#$’())’*(+(,$%$+-<br>**----- End of picture text -----**<br>


Figure 5. **Left** : Ablation studies with different keyframes. **Right** : Ablation studies with finetune all parameters or LoRA adapters 

|Method<br>CatV2TON[7]<br>MagicTryON[26]<br>GPT4o+VACE[21]|GP↑<br>1.30<br>1.19<br>2.67|PR↑<br>1.04<br>1.81<br>3.51|TC↑<br>1.08<br>1.88<br>2.61|
|---|---|---|---|
|DreamVVT|**3.41**|**3.69**|**3.32**|



Table 2. Quantitative comparisons on the Wild-TryOnBench dataset. We use GP, PR, TC as the short for Garment Preservation, Physical Realism and Temporal Consistency. 

## **4. Experiments** 

the standard generation loss of rectified flow [10]. During inference, we employ the Euler scheduler [22] with 50 sampling steps, set the classifier-free guidance scale to 2.5, and fix the random seed at 42. For keyframe try-on, each result is generated three times, and the best outcome is selected. In contrast, for video generation, each result is produced in a single run. Please refer to the supplementary material for details regarding the video and image caption, as well as the implementation of training and inference procedures for long videos. 

## **4.2.1. Evaluation** 

## **4.1. Datasets** 

We curated a high-quality human-centric video dataset comprising 69,643 samples, characterized by unrestricted subject and camera movements as well as dynamic scenes. Additionally, over one million pairs of multi-view images of the same individual were gathered from public websites. We performed mixed training using the collected unpaired data in combination with three publicly available try-on datasets: VITON-HD [5], DressCode [35], and ViViD [11]. During testing, we conducted indoor scenario evaluations on the ViViD-S dataset, which contains 180 samples, following the methodology of [7]. Additionally, we created an in-the-wild benchmark, named Wild-TryOnBench, consisting of 81 samples that encompass rich variations in subject or camera movement, scene changes, diverse forms of garment input and character styles. 

## **4.2. Implementation Details** 

We used the AdamW optimizer with a constant learning rate of 2e-5, weight decay set to 0.01, and gradient clipping set to 1.0. The entire training process was conducted on 8 NVIDIA H20(96G) GPUs for about 10 days. The LoRA rank of the model is set to 256 and the model is trained using 

The evaluation metrics follow those adopted in the previous works, whereas VFID with I3D[4]and ResNext[48] is introduced to evaluate video quality in unpaired scenarios, while SSIM[46], LPIPS[53] is added to measure the similarity between synthesized results and ground-truth in paired try-on test. Note that the VFID metric is mainly proposed to measure the quality of the video. We further introduce human evaluation on our Wild-TryOn benchmark, which incorporates three evaluation criteria: garment detail preservation, physical realism, and temporal consistency. Each criterion is scored on a scale of 0 to 5, with 0 indicating the worst performance and 5 the best. 

## **4.3. Qualitative Comparison** 

For qualitative evaluation, we compare our method with advanced video virtual try-on methods including CatV[2] TON, MagicTryOn and GPT4o+VACE. As shown in Fig.3 and Fig.4, our method demonstrates a strong superiority in synthesizing realistic and spatiotemporally smooth virtual try-on results in both the ViViD-S dataset and our WildTryOn evaluation benchmark. In contrast, CatV[2] TON[7] and MagicTryOn[26] show limited scalability in handling in-the-wild scenarios due to scarce garment-video pairs for 

7 

|Method|GP↑|PR↑|TC↑|
|---|---|---|---|
|DreamVVT-K1-w/o LoRA|3.10|3.43|3.26|
|DreamVVT-K1-w LoRA|3.16|3.62|3.29|
|DreamVVT-K2-w LoRA|**3.41**|**3.69**|**3.32**|



Table 3. Quantitative results for ablation studies on WildTryOn benchmark. K1 and K2 denote using one and two keyframes, respectively, while ”w/o LoRA” refers to full-parameter fine-tuning. 

training, which results in drastic performance degradation in out-of-domain test cases in our Wild-TryOn benchmark. Note that CatV[2] TON and MagicTryOn are prone to generate blurry results when encountering complex try-on scenarios such as 360-degree rotation, which often involves severe self-occlusion. GPT4o+VACE struggle to preserve the person’s identity and reproduce cloth details from given images. Our method, however, outperforms previous methods in generalizabilty with robust synthesis quality on arbitrary resolution, frame rate, indoor and outdoor scenarios. 

In contrast, model with two keyframes produces much more plausible results with the clear patterns. The quantitative results further support this observation. The variant with two keyframes achieves an obvious improvement compared to one keyframe setting on detail preservation metric. Besides, with sufficient garment and motion information provided by two keyframes, the score of physical realism and temporal consistency also have a slight increase. 

## **4.5.2. LoRA adapters for video generation model** 

As shown in the right panel of Fig. 5, when provided with the same textual description, the model trained with the LoRA adapter better preserves the pretrained model’s text control capability compared to full-parameter fine-tuning, thereby generating physically realistic clothing-interaction videos. The quantitative results in 3 further demonstrate that this approach significantly improves the score of physical realism without compromising garment detail preservation or temporal consistency. 

## **4.4. Quantitative Comparison** 

For quantitative evaluation, as reported in Tab.1, our method achieves state-of-the-art performance among compared baselines in both ViViD-S and our Wild-TryOn benchmark. Specifically, in the ViViD-S dataset, our method exceeds other baselines by a considerably large margin in unpaired try-on setting, with the lowest VFID score with I3D and ResNext. For paired try-on setting, our method achieves competitive results compared to existing video virtual try-on approaches, with the best performance on VID I and LPIPS metrics, and the second-best performance on VID R and SSIM metrics. Tab.2 shows similar observations, where our method outperforms existing baseline approaches, achieving the best performance on all three metrics. 

## **4.5. Ablation Study** 

To validate the contribution of the proposed modifications to the final improvement, we further conduct ablation experiments in our Wild-TryOn benchmark. Specifically, we implement three variants of the model by training model with and without LoRA, and testing model with different numbers of keyframes. The human evaluation in Tab.3 is introduced to evaluate the performance of model with different components. 

## **4.5.1. Key frame number** 

In the left panel of Fig.5 show qualitative results of our DreamVVT with different numbers of keyframes. As demonstrated in the figure, integrating multiple number of keyframes helps enhance the model performance under complicated scenarios like turn around. Model with only one keyframe input is prone to produce blurry results or generate artifacts due to lack of garment detail information. 

## **5. Discussions** 

In this work, we present DreamVVT, a stagewise framework based on Diffusion Transformers (DiTs) that effectively leverages unpaired human-centric data, pretrained model priors, and test-time inputs by integrating keyframes try-on and multi-modal guided virtual try-on video generation. Extensive experiments demonstrate that DreamVVT surpasses state-of-the-art methods in preserving garment details and temporal consistency under unrestricted scenarios, and effectively handles diverse garments, highlighting its potential for e-commerce and entertainment applications. 

**Limitations.** DreamVVT has several limitations. To accommodate arbitrary garment styles, the currently precomputed agnostic masks tend to cover large regions, which may compromise the integrity of both foreground objects and complex scenes. These challenges will be addressed in future research by employing mask-free video try-on techniques, for which DreamVVT will be utilized to construct the corresponding dataset. In addition, the method still struggles to achieve a high success rate in handling complex garment interaction motions, primarily due to limitations in the generative capabilities of pretrained models and the captioning of fine-grained actions. We plan to address and optimize this limitation in future work. 

## **6. Acknowledgement** 

We extend our sincere gratitude to Lu Jiang, Jiaqi Yang, Yanbo Zheng, Haoteng He, Yue Liu, Juan Li for their invaluable contributions and supports to this research work. 

8 

## **References** 

- [1] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji Lin, Dayiheng Liu, Gao Liu, Chengqiang Lu, Keming Lu, Jianxin Ma, Rui Men, Xingzhang Ren, Xuancheng Ren, Chuanqi Tan, Sinan Tan, Jianhong Tu, Peng Wang, Shijie Wang, Wei Wang, Shengguang Wu, Benfeng Xu, Jin Xu, An Yang, Hao Yang, Jian Yang, Shusheng Yang, Yang Yao, Bowen Yu, Hongyi Yuan, Zheng Yuan, Jianwei Zhang, Xingxuan Zhang, Yichang Zhang, Zhenru Zhang, Chang Zhou, Jingren Zhou, Xiaohuan Zhou, and Tianhang Zhu. Qwen technical report. _arXiv preprint arXiv:2309.16609_ , 2023. 6 

- [2] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. _arXiv preprint arXiv:2502.13923_ , 2025. 5, 12 

- [3] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. _arXiv preprint arXiv:2311.15127_ , 2023. 2, 3 

- [4] Joao Carreira and Andrew Zisserman. Quo vadis, action recognition? a new model and the kinetics dataset. In _proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pages 6299–6308, 2017. 7 

- [5] Seunghwan Choi, Sunghyun Park, Minsoo Lee, and Jaegul Choo. Viton-hd: High-resolution virtual try-on via misalignment-aware normalization. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 14131–14140, 2021. 3, 7 

- [6] Zheng Chong, Xiao Dong, Haoxiang Li, Shiyue Zhang, Wenqing Zhang, Xujie Zhang, Hanqing Zhao, Dongmei Jiang, and Xiaodan Liang. Catvton: Concatenation is all you need for virtual try-on with diffusion models. _arXiv preprint arXiv:2407.15886_ , 2024. 3 

- [7] Zheng Chong, Wenqing Zhang, Shiyue Zhang, Jun Zheng, Xiao Dong, Haoxiang Li, Yiling Wu, Dongmei Jiang, and Xiaodan Liang. Catv2ton: Taming diffusion transformers for vision-based virtual try-on with temporal concatenation. _arXiv preprint arXiv:2501.11325_ , 2025. 2, 3, 4, 7 

- [8] Zuozhuo Dai, Zhenghao Zhang, Yao Yao, Bingxue Qiu, Siyu Zhu, Long Qin, and Weizhi Wang. Animateanything: Finegrained open domain image animation with motion guidance, 2023. 12 

- [9] Haoye Dong, Xiaodan Liang, Xiaohui Shen, Bowen Wu, Bing-Cheng Chen, and Jian Yin. Fw-gan: Flow-navigated warping gan for video virtual try-on. In _Proceedings of the IEEE/CVF international conference on computer vision_ , pages 1161–1170, 2019. 2 

- [10] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling recti- 

fied flow transformers for high-resolution image synthesis. In _Forty-first international conference on machine learning_ , 2024. 2, 5, 7 

- [11] Zixun Fang, Wei Zhai, Aimin Su, Hongliang Song, Kai Zhu, Mao Wang, Yu Chen, Zhiheng Liu, Yang Cao, and ZhengJun Zha. Vivid: Video virtual try-on using diffusion models. _arXiv preprint arXiv:2405.11794_ , 2024. 2, 3, 7 

- [12] Yu Gao, Lixue Gong, Qiushan Guo, Xiaoxia Hou, Zhichao Lai, Fanshi Li, Liang Li, Xiaochen Lian, Chao Liao, Liyang Liu, et al. Seedream 3.0 technical report. _arXiv preprint arXiv:2504.11346_ , 2025. 5 

- [13] Junhong Gou, Siyu Sun, Jianfu Zhang, Jianlou Si, Chen Qian, and Liqing Zhang. Taming the power of diffusion models for high-quality virtual try-on with appearance flow. In _Proceedings of the 31st ACM International Conference on Multimedia_ , pages 7599–7607, 2023. 3 

- [14] Hailong Guo, Bohan Zeng, Yiren Song, Wentao Zhang, Chuang Zhang, and Jiaming Liu. Any2anytryon: Leveraging adaptive position embeddings for versatile virtual clothing tasks. _arXiv preprint arXiv:2501.15891_ , 2025. 3 

- [15] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. _ICLR_ , 1(2):3, 2022. 2, 3, 5 

- [16] Li Hu. Animate anyone: Consistent and controllable imageto-video synthesis for character animation. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 8153–8163, 2024. 3 

- [17] Li Hu, Guangyuan Wang, Zhen Shen, Xin Gao, Dechao Meng, Lian Zhuo, Peng Zhang, Bang Zhang, and Liefeng Bo. Animate anyone 2: High-fidelity character image animation with environment affordance. _arXiv preprint arXiv:2502.06145_ , 2025. 3 

- [18] Teng Hu, Zhentao Yu, Zhengguang Zhou, Sen Liang, Yuan Zhou, Qin Lin, and Qinglin Lu. Hunyuancustom: A multimodal-driven architecture for customized video generation. _arXiv preprint arXiv:2505.04512_ , 2025. 2 

- [19] Jianwen Jiang, Gaojie Lin, Zhengkun Rong, Chao Liang, Yongming Zhu, Jiaqi Yang, and Tianyun Zhong. Mobileportrait: Real-time one-shot neural head avatars on mobile devices. In _Proceedings of the Computer Vision and Pattern Recognition Conference_ , pages 15920–15929, 2025. 6 

- [20] Tao Jiang, Peng Lu, Li Zhang, Ningsheng Ma, Rui Han, Chengqi Lyu, Yining Li, and Kai Chen. Rtmpose: Realtime multi-person pose estimation based on mmpose. _arXiv preprint arXiv:2303.07399_ , 2023. 4, 12 

- [21] Zeyinzi Jiang, Zhen Han, Chaojie Mao, Jingfeng Zhang, Yulin Pan, and Yu Liu. Vace: All-in-one video creation and editing. _arXiv preprint arXiv:2503.07598_ , 2025. 2, 7 

- [22] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. 2022. 7 

- [23] Jeongho Kim, Guojung Gu, Minho Park, Sunghyun Park, and Jaegul Choo. Stableviton: Learning semantic correspondence with latent diffusion model for virtual try-on. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 8176–8185, 2024. 3 

9 

- [24] Black Forest Labs. Flux. https://github.com/ black-forest-labs/flux , 2024. 2, 3 

- [25] Dong Li, Wenqi Zhong, Wei Yu, Yingwei Pan, Dingwen Zhang, Ting Yao, Junwei Han, and Tao Mei. Pursuing temporal-consistent video virtual try-on via dynamic pose interaction. _arXiv preprint arXiv:2505.16980_ , 2025. 2, 3 

- [26] Guangyuan Li, Siming Zheng, Hao Zhang, Jinwei Chen, Junsheng Luan, Binkai Ou, Lei Zhao, Bo Li, and Peng-Tao Jiang. Magictryon: Harnessing diffusion transformer for garment-preserving video virtual try-on, 2025. 2, 3, 4, 7 

- [27] Hui Li, Mingwang Xu, Yun Zhan, Shan Mu, Jiaye Li, Kaihui Cheng, Yuxuan Chen, Tan Chen, Mao Ye, Jingdong Wang, and Siyu Zhu. Openhumanvid: A large-scale high-quality dataset for enhancing human-centric video generation, 2024. 12 

- [28] Siqi Li, Zhengkai Jiang, Jiawei Zhou, Zhihong Liu, Xiaowei Chi, and Haoqian Wang. Realvvt: Towards photorealistic video virtual try-on via spatio-temporal consistency. _arXiv preprint arXiv:2501.08682_ , 2025. 2, 3 

- [29] Zhimin Li, Jianwei Zhang, Qin Lin, Jiangfeng Xiong, Yanxin Long, Xinchi Deng, Yingfang Zhang, Xingchao Liu, Minbin Huang, Zedong Xiao, Dayou Chen, Jiajun He, Jiahao Li, Wenyue Li, Chen Zhang, Rongwei Quan, Jianxiang Lu, Jiabin Huang, Xiaoyan Yuan, Xiaoxiao Zheng, Yixuan Li, Jihong Zhang, Chao Zhang, Meng Chen, Jie Liu, Zheng Fang, Weiyan Wang, Jinbao Xue, Yangyu Tao, Jianchen Zhu, Kai Liu, Sihuan Lin, Yifu Sun, Yun Li, Dongdong Wang, Mingtao Chen, Zhichao Hu, Xiao Xiao, Yan Chen, Yuhong Liu, Wei Liu, Di Wang, Yong Yang, Jie Jiang, and Qinglin Lu. Hunyuan-dit: A powerful multi-resolution diffusion transformer with fine-grained chinese understanding, 2024. 2 

- [30] Ente Lin, Xujie Zhang, Fuwei Zhao, Yuxuan Luo, Xin Dong, Long Zeng, and Xiaodan Liang. Dreamfit: Garment-centric human generation via a lightweight anything-dressing encoder. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , pages 5218–5226, 2025. 3, 5 

- [31] Gaojie Lin, Jianwen Jiang, Jiaqi Yang, Zerong Zheng, and Chao Liang. Omnihuman-1: Rethinking the scaling-up of one-stage conditioned human animation models. _arXiv preprint arXiv:2502.01061_ , 2025. 6 

- [32] Shanchuan Lin, Xin Xia, Yuxi Ren, Ceyuan Yang, Xuefeng Xiao, and Lu Jiang. Diffusion adversarial post-training for one-step video generation. _arXiv preprint arXiv:2501.08316_ , 2025. 5 

- [33] Wen Liu, Zhixin Piao, Jie Min, Wenhan Luo, Lin Ma, and Shenghua Gao. Liquid warping gan: A unified framework for human motion imitation, appearance transfer and novel view synthesis. In _Proceedings of the IEEE/CVF international conference on computer vision_ , pages 5904–5913, 2019. 3 

- [34] Yifang Men, Yuan Yao, Miaomiao Cui, and Liefeng Bo. Mimo: Controllable character video synthesis with spatial decomposed modeling. _arXiv preprint arXiv:2409.16160_ , 2024. 3 

- [35] Davide Morelli, Matteo Fincato, Marcella Cornia, Federico Landi, Fabio Cesari, and Rita Cucchiara. Dress code: Highresolution multi-category virtual try-on. In _Proceedings of_ 

   - _the IEEE/CVF conference on computer vision and pattern recognition_ , pages 2231–2235, 2022. 7 

- [36] Davide Morelli, Alberto Baldrati, Giuseppe Cartella, Marcella Cornia, Marco Bertini, and Rita Cucchiara. Ladi-vton: Latent diffusion textual-inversion enhanced virtual try-on. In _Proceedings of the 31st ACM international conference on multimedia_ , pages 8580–8589, 2023. 3 

- [37] William Peebles and Saining Xie. Scalable diffusion models with transformers. In _Proceedings of the IEEE/CVF international conference on computer vision_ , pages 4195–4205, 2023. 3 

- [38] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. _arXiv preprint arXiv:2307.01952_ , 2023. 3 

- [39] Yurui Ren, Xiaoming Yu, Junming Chen, Thomas H Li, and Ge Li. Deep image spatial transformation for person image generation. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 7690–7699, 2020. 3 

- [40] Yurui Ren, Xiaoqing Fan, Ge Li, Shan Liu, and Thomas H Li. Neural texture extraction and distribution for controllable person image synthesis. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 13535–13544, 2022. 3 

- [41] Ke Sun, Jian Cao, Qi Wang, Linrui Tian, Xindi Zhang, Lian Zhuo, Bang Zhang, Liefeng Bo, Wenbo Zhou, Weiming Zhang, et al. Outfitanyone: Ultra-high quality virtual try-on for any clothing and any person. _arXiv preprint arXiv:2407.16224_ , 2024. 3 

- [42] ByteDance Seed Team. Seed1.5-vl technical report. _arXiv preprint arXiv:2505.07062_ , 2025. 5 

- [43] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao, Keyu Yan, Lianghua Huang, Mengyang Feng, Ningyi Zhang, Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng, Shiwei Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang, Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu, Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, Yangyu Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yitong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, Yun Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi Jiang, Zhen Han, Zhi-Fan Wu, and Ziyu Liu. Wan: Open and advanced large-scale video generative models. _arXiv preprint arXiv:2503.20314_ , 2025. 2 

- [44] Haoyu Wang, Zhilu Zhang, Donglin Di, Shiliang Zhang, and Wangmeng Zuo. Mv-vton: Multi-view virtual try-on with diffusion models. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , pages 7682–7690, 2025. 3 

- [45] Xiang Wang, Shiwei Zhang, Changxin Gao, Jiayu Wang, Xiaoqiang Zhou, Yingya Zhang, Luxin Yan, and Nong Sang. Unianimate: Taming unified video diffusion models for consistent human image animation. _arXiv preprint arXiv:2406.01188_ , 2024. 3 

10 

- [46] Zhou Wang, Alan Conrad Bovik, Hamid Rahim Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. _IEEE transactions on image processing_ , 13(4):600–612, 2004. 7 

- [47] Haoning Wu, Erli Zhang, Liang Liao, Chaofeng Chen, Jingwen Hou Hou, Annan Wang, Wenxiu Sun Sun, Qiong Yan, and Weisi Lin. Exploring video quality assessment on user generated contents from aesthetic and technical perspectives. In _International Conference on Computer Vision (ICCV)_ , 2023. 12 

- [48] Saining Xie, Ross Girshick, Piotr Doll´ar, Zhuowen Tu, and Kaiming He. Aggregated residual transformations for deep neural networks. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 1492–1500, 2017. 7 

- [49] Zhenyu Xie, Zaiyu Huang, Xin Dong, Fuwei Zhao, Haoye Dong, Xijin Zhang, Feida Zhu, and Xiaodan Liang. Gpvton: Towards general purpose virtual try-on via collaborative local-flow global-parsing learning. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 23550–23559, 2023. 3 

- [50] Yuhao Xu, Tao Gu, Weifeng Chen, and Arlene Chen. Ootdiffusion: Outfitting fusion based latent diffusion for controllable virtual try-on. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , pages 8996–9004, 2025. 3 

- [51] Zhengze Xu, Mengting Chen, Zhao Wang, Linyu Xing, Zhonghua Zhai, Nong Sang, Jinsong Lan, Shuai Xiao, and Changxin Gao. Tunnel try-on: Excavating spatial-temporal tunnels for high-quality virtual try-on in videos. In _Proceedings of the 32nd ACM International Conference on Multimedia_ , pages 3199–3208, 2024. 2, 3, 4 

- [52] Zixuan Ye, Xuanhua He, Quande Liu, Qiulin Wang, Xintao Wang, Pengfei Wan, Di Zhang, Kun Gai, Qifeng Chen, and Wenhan Luo. Unic: Unified in-context video editing. _arXiv preprint arXiv:2506.04216_ , 2025. 2 

- [53] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 586–595, 2018. 7 

- [54] Xuanpu Zhang, Dan Song, Pengxin Zhan, Tianyu Chang, Jianhao Zeng, Qingguo Chen, Weihua Luo, and Anan Liu. Boow-vton: Boosting in-the-wild virtual try-on via maskfree pseudo data training. _arXiv preprint arXiv:2408.06047_ , 2024. 3 

- [55] Yuang Zhang, Jiaxi Gu, Li-Wen Wang, Han Wang, Junqi Cheng, Yuefeng Zhu, and Fangyuan Zou. Mimicmotion: High-quality human motion video generation with confidence-aware pose guidance. _arXiv preprint arXiv:2406.19680_ , 2024. 3 

- [56] Jun Zheng, Jing Wang, Fuwei Zhao, Xujie Zhang, and Xiaodan Liang. Dynamic try-on: Taming video virtual try-on with dynamic attention mechanism. _arXiv preprint arXiv:2412.09822_ , 2024. 2, 3 

- [57] Peng Zheng, Dehong Gao, Deng-Ping Fan, Li Liu, Jorma Laaksonen, Wanli Ouyang, and Nicu Sebe. Bilateral reference for high-resolution dichotomous image segmentation. _CAAI Artificial Intelligence Research_ , 3:9150038, 2024. 4 

11 

## **7. Supplementary Material** 

**==> picture [496 x 150] intentionally omitted <==**

**----- Start of picture text -----**<br>
!"#$% &"’($)*<br>!*+,"# -’.#*/ !"#$ !"#$# !%&’((’$) !+,$ !"#$% &’()"*+ ,%-.$ /0123 01$ !"#$% +,,%(-("%,*<br>!"# $%&’#(#)#*#’*+ ?@ A%,$B C4DE%,$<br>6-02 07#( "0,-’# %&’()* ),-.*/0( 1-02 3" *0 45" -.&/* 4$5* %. $66$-*, %--)’,"%7 801219 %&’()* !"#$% -(A*"%7B FG$7?29H!I<br>8.*."#*" 0- /(*#-(#* :’;(7 <<%5 (.$( /0=353><br>!2#$ /(3*)4 5) /(3*)4 ./-0$ &"’($)* !"!$ /(3*)4 5) /(3*)4<br>!*+,"# 012)#/ )3 6738(45 5) /(3*)4 J;(K$ &’()"*+ ,%-.$ /0123 )3 6738(45 5) /(3*)4 ./-0$ +,,%(-("%,*<br>4$5* %. $66$-*, %--)’,"%7 801219 ?@ A%,$B C4DE%,$<br>6-02 07#( "0,-’# &876(* &876(*<br>:’;(7 <<%5 (.$( /0=353> J;(K$ -(A*"%7B @%’<(%M29 !I<br>8.*."#*" 0- /(*#-(#*<br>LAA$(.(7-$ ()"K7;$7* 0 4.’$<br>**----- End of picture text -----**<br>


Figure 6. Our curated and annotated pipeline for constructing human-centric video and paired image datasets. 

## **7.1. Dataset Details** 

## **7.1.1. Video Dataset Construction** 

To obtain high-quality video data for diverse scenarios, we first collected approximately 102K videos from public datasets [27] and online sources. We then used PySceneDetect to segment these videos into shots, resulting in 187K video clips, each with a duration limited to between 3 and 20 seconds. Subsequently, we applied a series of filtering operators to remove invalid clips. Specifically, we filtered out low-quality videos based on VQA scores [47], discarded nearly static videos according to motion strength [8], and used video OCR (tesserocr) to detect and exclude clips with severe occlusion by identifying extensive text regions. For annotation, we adopted RTMPose [20] as a robust and efficient pose representation. To extract textual descriptions from the raw videos, we employed the Qwen2.5-VL model [2] using a predefined template shown as Listing 1 that covers three aspects: environment, character appearance, and motion. In particular, the motion descriptions not only capture the overall movement of the person, but also provide detailed information about interactions with clothing and the environment. During training, we randomly drop the appearance and environment descriptions to encourage the model to focus more on the motion descriptions. When generating try-on videos, the character appearance from the original video is replaced with that from the try-on image. 

   - Listing 1. Video caption prompt 

- 1 prompt = f"""Describe input video in the following json format: 

- 2 {{"ENVIRONMENT": "Sentences describing the environment, video source and tags", 

- 3 "APPERANCE": "Sentences describing the character’s apperance", 

- 4 "MOTION": "Sentences describing the character’ s actions, expressions and interaction with the environment. Do not add any action descriptions that are not present in the original description."}} 

- 5 Only return in this format. Make sure that the output text can be directly parsed by a JSON 

- parser. Do not add objects or character actions that do not exist in the original video description. Make sure that the output text should match the tone of video caption generation, therefore, phrases like "as described" should not appear.""" 

## **7.1.2. Image Dataset Construction** 

To construct a multi-frame consistent dataset for training multi-frame try-on models, in addition to sampling keyframes from videos, we also fully leveraged high-quality paired images with diverse viewpoints and poses collected from public websites and datasets. Specifically, after applying a series of filtering steps such as image quality assessment, appearance consistency check, and person size verification to the collected 1.3 million images, we obtained a clean set of 1.01 million images. For image caption, since our model targets at synthesizing multi-frame virtual try-on results, we prepare elaborately designed prompt for every frame generation, which features detailed orientation description like front, back, side, and common identity and environment descriptions including clothing details as well as scenario background. Specifically, we caption every sampled frame to get appearance-unrelated information such as poseture and orientations, and then merge it with input garment information to get the frame-independent prompt. Subsequently, to ensure the consistency of prompt description, we also employ a VLM rewrite procedure to eliminate potential conflicts introduced by the captioning model. 

12 

Algorithm 1: Keyframe Sampling **Input** : Input video: Vin Anchor frame: fanchor **Parameter** : Total number of frames in the input video: N Number of selected key frames: K = 2 Weight of the area ratio: = 0.3 weight of score interval: = 0.2 **Output** : Key frame images: fkey 

1: Obtain skeletal joint direction vectors of anchor frame Danchor = compute ~~j~~ oint ~~d~~ irection(fanchor) 

2: Let i = 0 

3: **while** i < N **do** 

4: Obtain skeletal joint direction vectors of video frame Dv[i] = compute ~~j~~ oint ~~d~~ irection(Vin[i]) 5: Compute motion similarity Sm[i] =[P] Danchor · Dv[i] 6: Compute area ratio Sr[i] = Asubject/Aframe 7: Update final score Sfinal.update(index = i, score = Sm[i]+ Sr[i]) 

of errors over time, which results in noticeable degradation in the quality of the generated video as the sequence length increases. Therefore, we remove the decoder and encoder processes, and instead directly use the latent representation of the last frame from the previous video segment as the initial frame for the subsequent segment. By leveraging the pretrained model’s image-to-video generation capability for continuation, this approach significantly extends the duration before noticeable degradation occurs. 

## **7.3. More Results** 

Fig.7 and Fig.8 show additional visual comparison results for our proposed DreamVVT network and the baseline methods on the ViViD datasets. While Fig.9 and Fig.10 show additional comparison results on the WildTryOn dataset. Fig. 11 presents an additional ablation result, demonstrating that training LoRA adapters enables garment interaction more effectively than full parameter finetuning. 

8: **end while** 

- 9: Calculate minimum score interval 

Ts ~~m~~ in[=][get] ~~s~~ cores(Sfinal).mean 10: Sfinal = sort(Sfinal, score : descending) 11: Initial key frame index Idxkey[0] = get ~~i~~ ndexes(Sfinal)[0] 12: Let i = N 1 

13: **while** i >= 0 **do** 14: cur idx = get ~~i~~ ndexes(Sfinal)[i] 15: **if** All(|(Sfinal[cur ~~i~~ dx] Sfinal[Idxkey[:]]| >= Ts ~~m~~ in[)] **[ then]** 16: Idxkey.append(get ~~i~~ ndexes(Sfinal)[i]) 17: **end if** 18: **end while** 

19: fkey[: K] = Vin[Idxkey[: K]] 

20: **return** fkey 

## **7.2. Implementation Details Supplement** 

## **7.2.1. Keyframe Sampling Algorithm** 

To facilitate reproducibility, we provide the pseudocode for the Keyframe Sampling strategy, as shown Algorithm 1. 

## **7.2.2. Long-term Video Generation** 

A common strategy for long video generation is to iteratively synthesize video segments such that the initial frames of the subsequent segment overlap with the final frames of the preceding segment. However, we observe that repeatedly passing the generated results of the previous segment through the decoder and encoder leads to the accumulation 

13 

!"#$%&’()*&"’’’’’’’’’’+,$-&./’’’’’’’’’’’0,/![!] 123’’’’’’’’4,5)%61$7".’’’’’’’’’’’’2#$8’ 

Figure 7. Additional qualitative comparison results on the ViViD dataset.Please zoom in for more details. 

14 

!"#$%&’()*&"’’’’’’’’’’+,$-&./’’’’’’’’’’’0,/![!] 123’’’’’’’’4,5)%61$7".’’’’’’’’’’’’2#$8’’’’’’’’’’’’ 

> Figure 8. Additional qualitative comparison results on the ViViD dataset.Please zoom in for more details. 

15 

!"#$%&’()*&"’’’’’’’+,$-&./’’’’’’’’’’0,/![!] 123’’’’’’’4,5)%61$7".’’’’’’8"9(:0;’’’’’’’’’’’’2#$< 

Figure 9. Additional qualitative comparison results on the WildTryOn dataset.Please zoom in for more details. 

16 

!"#$%&’()*&"’’’’’’’+,$-&./’’’’’’’’’’0,/![!] 123’’’’’’’4,5)%61$7".’’’’’’8"9(:0;’’’’’’’’’’’’2#$< 

Figure 10. Additional qualitative comparison results on the WildTryOn dataset.Please zoom in for more details. 

17 

**==> picture [170 x 274] intentionally omitted <==**

**----- Start of picture text -----**<br>
!"#$%&’()* !"#$%&’()*#+$,’<br>!"#$%&’$(’$)"#$*(+#,$(-$+#%,’-).&)(’/$<br>)"#$#0&-)(1()2$,3$)"#$!4-"(.)5$6#$3(.-)$<br>7800-$,’$)"#$1,00&.9$)"#’$)8.’-$-(+#:&2-$<br>&’+$-).#)1"#-$)"#$3.,’)$,3$)"#$-"(.)5<br>!"#$%&’$(’$)"#$*(+#,$(-$+#%,’-).&)(’/$<br>)"#$#0&-)(1()2$,3$)"#$!4-"(.)5$6#$3(.-)$<br>7800-$,’$)"#$1,00&.9$)"#’$)8.’-$-(+#:&2-$<br>&’+$-).#)1"#-$)"#$3.,’)$,3$)"#$-"(.)5<br>’(2(*%$+-<br>’’./01<br>!"#$%&#$<br>’())’*(+(,$%$+-<br>!"#$%&#$<br>**----- End of picture text -----**<br>


**==> picture [40 x 6] intentionally omitted <==**

**----- Start of picture text -----**<br>
+,"%-)%’.&’()*<br>**----- End of picture text -----**<br>


Figure 11. Additional ablation results: finetuning LoRA adapters vs. full model parameters. 

18 

