---
type: paper
slug: deep-compositional-phase-diffusion-for-long-motion-sequence-generation
title: Deep Compositional Phase Diffusion for Long Motion Sequence Generation
authors: Ho Yin Au, Junkun Jiang, Jie Chen, Jingyu Xiang
source_path: .paper-scholar/deep-compositional-phase-diffusion-for-long-motion-sequence-generation
ingested_at: '2026-06-05 05:49:09'
authors_list: []
sections:
- id: 491
  heading: Ho Yin Au
  role: other
  order_index: 0
  summary: Hong Kong Baptist University cshyau@comp.hkbu.edu.hk
- id: 492
  heading: Junkun Jiang
  role: other
  order_index: 1
  summary: Hong Kong Baptist University csjkjiang@comp.hkbu.edu.hk
- id: 493
  heading: Jie Chen\*
  role: other
  order_index: 2
  summary: Hong Kong Baptist University chenjie@comp.hkbu.edu.hk
- id: 494
  heading: Jingvu Xiang
  role: other
  order_index: 3
  summary: Hong Kong Baptist University csjyxiang@comp.hkbu.edu.hk
- id: 495
  heading: Abstract
  role: abstract
  order_index: 4
  summary: Recent research on motion generation has shown significant progress in generating semantically aligned motion with singular semantics. However, when employing these models to create composite sequences containing multiple semantically generated motion clips, they often struggle to preserve the continuity of motion dynamics at the transition boundaries between clips, resulting in awkward transitions and abrupt artifacts.
- id: 496
  heading: 1 Introduction
  role: introduction
  order_index: 5
  summary: Deep learning-based human motion generation holds significant potential for creating virtual humanoid animations and enhancing robotics applications. With more advanced modeling techniques [\[1,](#page-9-0) [2,](#page-9-1) [3\]](#page-9-2) and more motion data being captured [\[4,](#page-9-3) [5,](#page-9-4) [6,](#page-9-5) [7\]](#page-9-6), motion generation models are evolving rapidly and can be adapted to a variety of multimodal generation tasks.
- id: 497
  heading: 2 Related work
  role: background
  order_index: 6
  summary: '**Motion Phase Modeling.** Pioneering approaches [16, 17, 18] incorporate explicit phase inputs, such as foot contact during walking, to achieve smooth motion extrapolation and transition. DeepPhase [15] further extends this concept by developing a Periodic Autoencoder (PAE) that encodes motion segments into phase latent parameters, i.e., frequency (**F**), amplitude (**A**), offset (**B**), and phase shift (**S**).'
- id: 498
  heading: 3 Compositional Phase Diffusion
  role: other
  order_index: 7
  summary: 'We propose three key components for the framework: the Action-Centric Periodic Autoencoder (ACT-PAE), the Transitional Phase Diffusion Module (TPDM), and the Semantic Phase Diffusion Module (SPDM). ACT-PAE creates a motion latent manifold that captures important semantic and transition-aware phase information for each motion segment $\mathbf{X} \in \mathbb{R}^{N \times E}$ and represent them as a set of latent variables $\mathbf{P} = [\mathbf{F}, \mathbf{A}, \mathbf{B}, \mathbf{S}]$ .'
- id: 499
  heading: <span id="page-3-0"></span>3.1 Key Components
  role: method
  order_index: 8
  summary: ''
- id: 500
  heading: '<span id="page-3-3"></span>3.1.1 ACT-PAE: Action-Centric Periodic Autoencoder'
  role: other
  order_index: 9
  summary: Our ACT-PAE builds upon the transformer-based motion autoencoder architecture from ACTOR [37]. ACT-PAE encoder first processes input motion $\mathbf{X} \in \mathbb{R}^{N \times E}$ of N frames into four phase parameters $\mathbf{F}, \mathbf{A}, \mathbf{B}, \mathbf{S} \in \mathbb{R}^Q$ .
- id: 501
  heading: '<span id="page-3-4"></span>3.1.2 SPDM: Semantic Phase Diffusion Module'
  role: other
  order_index: 10
  summary: SPDM is designed to denoise phase parameters so that the corresponding decoded motion segment is aligned to the semantic condition. In text-to-motion settings, SPDM employs the pre-trained *CLIP-ViT-B/32* [23] to encode the input text conditions into embedding vector $C_{\mathbf{p}}$ , as shown in Fig.
- id: 502
  heading: '<span id="page-4-4"></span>3.1.3 TPDM: Transitional Phase Diffusion Module'
  role: other
  order_index: 11
  summary: TDPM is designed to denoise phase parameters such that the resulting decoded motions are transitionally aligned with adjacent motions. Depending on the specific application scenario, these adjacent motions may come from either the forward or backward direction, which will be explained in Sec.
- id: 503
  heading: <span id="page-4-0"></span>3.2 Applications
  role: method
  order_index: 12
  summary: ''
- id: 504
  heading: <span id="page-4-3"></span>3.2.1 Compositional Motion Pair Generation
  role: other
  order_index: 13
  summary: The compositional motion pair generation task focuses on creating two sequentially connected motion segments, $\mathbf{X_p}$ and $\mathbf{X_s}$ . To ensure a smooth transition while maintaining semantic alignment, we develop a compositional motion diffusion pipeline that progressively incorporates the **semantic information** and the **phase dynamics information from adjacent segments** in the diffusion process.
- id: 505
  heading: 3.2.2 Motion Inbetweening
  role: other
  order_index: 14
  summary: The motion inbetweening task aims to generate an inbetweening motion $X_i$ , which is of a specified length to bridge the gap between two separated motions $[X_p, X_s]$ . The pipeline for the task is illustrated in Fig.
- id: 506
  heading: 3.2.3 Long-term Motion Generation
  role: other
  order_index: 15
  summary: Long-term motion sequence generation extends beyond short-term compositional motion pair generation by producing much longer continuous motion, composed of hundreds or thousands of motion segments. While short-term tasks focus on semantics and transitions within a few segments, long-term generation involves monitoring kinetic dynamics, which can impact motion over extended sequences and potentially disrupt motion realism and physical plausibility.
- id: 507
  heading: 4 Experiments
  role: experiment
  order_index: 16
  summary: ''
- id: 508
  heading: 4.1 Implementation and Evaluation Details
  role: experiment
  order_index: 17
  summary: ''
- id: 509
  heading: 4.1.1 Training and Evaluation Dataset
  role: experiment
  order_index: 18
  summary: We use the BABEL-TEACH dataset [4, 12] for training and evaluation, as it provides annotated subsequence pairs essential for long-term motion generation [12, 13, 14], facilitating the learning of transitions between subsequences. These annotated pairs are derived from decomposing fine-grained text subsequence annotations from BABEL [4].
- id: 510
  heading: 4.1.2 Evaluation Metrics
  role: experiment
  order_index: 19
  summary: 'We assess the results of **compositional motion pair generation**, **long-term motion generation**, and **conditional motion inbetweening** based on two key aspects: *Fréchet Inception Distance* (FID) for Motion Realism and *Multimodal Distance* (MMD) for Text Alignment, following the T2M [6]'
- id: 511
  heading: 4.2 Compositional Motion Generation Performance Evaluation
  role: experiment
  order_index: 20
  summary: ''
- id: 512
  heading: 4.2.1 Compositional Motion Pair Generation
  role: other
  order_index: 21
  summary: The compositional motion pair experiment follows the setup illustrated on the left in Fig. 3, with the objective of generating motions $\mathbf{X_p}$ , $\mathbf{X_t}$ , and $\mathbf{X_s}$ based on the corresponding text condition pairs $(C_p, C_s)$ .
- id: 513
  heading: 4.2.2 Long-term Motion Generation
  role: other
  order_index: 22
  summary: To assess the long-term motion generation performance, we combine all text conditions from the testing dataset into a single extended text sequence of 3,164 texts, and apply comparison models to
- id: 514
  heading: 4.3 Motion Inbetweening Performance Evaluation
  role: experiment
  order_index: 23
  summary: The unconditional motion inbetweening (UMIB) experiment follows a setup similar to Fig. 4, where a specific number of frames around the transition boundary of testing motion pairs are masked to evaluate various methods for reconstructing the masked motion content.
- id: 515
  heading: 4.4 Ablation Studies and User Studies
  role: other
  order_index: 24
  summary: We assess the effects of our proposed modules and recommended hyperparameters on compositional motion generation and motion inbetweening tasks. Firstly, the integration of *frame-level tokens* within SPDM and TPDM significantly enhances their performance in denoising *param-level tokens*.
- id: 516
  heading: 5 Conclusion and Future Work
  role: conclusion
  order_index: 25
  summary: We present the Transitional Phase Diffusion Module (TPDM) and the Semantic Phase Diffusion Module (SPDM), which operate within the periodic latent space generated by the Action-Centric Periodic Autoencoder. These modules inject semantic guidance and neighbouring phase information into the motion denoising process, enabling the generation of semantically meaningful motion clips with smooth transitions.
- id: 517
  heading: References
  role: other
  order_index: 26
  summary: '- <span id="page-9-0"></span>[1] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need.'
- id: 518
  heading: A Technical Details of Compositional Phase Diffusion
  role: other
  order_index: 27
  summary: ''
- id: 519
  heading: A.1 Compositional Motion Generation Algorithm
  role: other
  order_index: 28
  summary: Algorithm implementation details of the phase diffusion pipeline for the compositional motion generation task in Sec. 3.2.1.
- id: 520
  heading: A.2 Adjustment to T and PE
  role: other
  order_index: 29
  summary: As discussed in Sec. 3.1.1, we have refined the sinusoidal positional embedding PE and the time window T to support motion autoencoding with variable lengths.
- id: 521
  heading: A.3 Details of SPDM and TPDM
  role: other
  order_index: 30
  summary: As discussed in Sec. 3.1.2 and Sec.
- id: 522
  heading: A.4 Implementation Details
  role: other
  order_index: 31
  summary: We apply the emphasis projection with c=15, as demonstrated in GMD [25], to incorporate root trajectory information into the motion representation. Also, our models are designed based on phase latent size Q=512, which serves as both the latent dimension for all diffusion modules and the number of periodic signals in ACT-PAE.
- id: 523
  heading: B Conditional Motion Inbetweening Evaluation
  role: experiment
  order_index: 32
  summary: 'Figure 7: Visualization of the **CMIB** with 120 transition boundary frames conditioned with *bend* arms up: preceding motion in blue, transitioning motion in green, and succeeding motion in yellow.'
- id: 524
  heading: C Impact Statements
  role: other
  order_index: 33
  summary: The exploration and application of phase latent spaces in this work contribute to the advancement of deep learning by offering new methodologies for signal processing and multimedia generation. It has no negative impact on society as the focus is on technological improvement rather than datasets that could be sensitive or have privacy implications.
---

# Deep Compositional Phase Diffusion for Long Motion Sequence Generation

## [other] Ho Yin Au
Hong Kong Baptist University cshyau@comp.hkbu.edu.hk

## [other] Junkun Jiang
Hong Kong Baptist University csjkjiang@comp.hkbu.edu.hk

## [other] Jie Chen\*
Hong Kong Baptist University chenjie@comp.hkbu.edu.hk

## [other] Jingvu Xiang
Hong Kong Baptist University csjyxiang@comp.hkbu.edu.hk

## [abstract] Abstract
Recent research on motion generation has shown significant progress in generating semantically aligned motion with singular semantics. However, when employing these models to create composite sequences containing multiple semantically generated motion clips, they often struggle to preserve the continuity of motion dynamics at the transition boundaries between clips, resulting in awkward transitions and abrupt artifacts.

## [introduction] 1 Introduction
Deep learning-based human motion generation holds significant potential for creating virtual humanoid animations and enhancing robotics applications. With more advanced modeling techniques [\[1,](#page-9-0) [2,](#page-9-1) [3\]](#page-9-2) and more motion data being captured [\[4,](#page-9-3) [5,](#page-9-4) [6,](#page-9-5) [7\]](#page-9-6), motion generation models are evolving rapidly and can be adapted to a variety of multimodal generation tasks.

## [background] 2 Related work
**Motion Phase Modeling.** Pioneering approaches [16, 17, 18] incorporate explicit phase inputs, such as foot contact during walking, to achieve smooth motion extrapolation and transition. DeepPhase [15] further extends this concept by developing a Periodic Autoencoder (PAE) that encodes motion segments into phase latent parameters, i.e., frequency (**F**), amplitude (**A**), offset (**B**), and phase shift (**S**).

## [other] 3 Compositional Phase Diffusion
We propose three key components for the framework: the Action-Centric Periodic Autoencoder (ACT-PAE), the Transitional Phase Diffusion Module (TPDM), and the Semantic Phase Diffusion Module (SPDM). ACT-PAE creates a motion latent manifold that captures important semantic and transition-aware phase information for each motion segment $\mathbf{X} \in \mathbb{R}^{N \times E}$ and represent them as a set of latent variables $\mathbf{P} = [\mathbf{F}, \mathbf{A}, \mathbf{B}, \mathbf{S}]$ .

## [method] <span id="page-3-0"></span>3.1 Key Components


## [other] <span id="page-3-3"></span>3.1.1 ACT-PAE: Action-Centric Periodic Autoencoder
Our ACT-PAE builds upon the transformer-based motion autoencoder architecture from ACTOR [37]. ACT-PAE encoder first processes input motion $\mathbf{X} \in \mathbb{R}^{N \times E}$ of N frames into four phase parameters $\mathbf{F}, \mathbf{A}, \mathbf{B}, \mathbf{S} \in \mathbb{R}^Q$ .

## [other] <span id="page-3-4"></span>3.1.2 SPDM: Semantic Phase Diffusion Module
SPDM is designed to denoise phase parameters so that the corresponding decoded motion segment is aligned to the semantic condition. In text-to-motion settings, SPDM employs the pre-trained *CLIP-ViT-B/32* [23] to encode the input text conditions into embedding vector $C_{\mathbf{p}}$ , as shown in Fig.

## [other] <span id="page-4-4"></span>3.1.3 TPDM: Transitional Phase Diffusion Module
TDPM is designed to denoise phase parameters such that the resulting decoded motions are transitionally aligned with adjacent motions. Depending on the specific application scenario, these adjacent motions may come from either the forward or backward direction, which will be explained in Sec.

## [method] <span id="page-4-0"></span>3.2 Applications


## [other] <span id="page-4-3"></span>3.2.1 Compositional Motion Pair Generation
The compositional motion pair generation task focuses on creating two sequentially connected motion segments, $\mathbf{X_p}$ and $\mathbf{X_s}$ . To ensure a smooth transition while maintaining semantic alignment, we develop a compositional motion diffusion pipeline that progressively incorporates the **semantic information** and the **phase dynamics information from adjacent segments** in the diffusion process.

## [other] 3.2.2 Motion Inbetweening
The motion inbetweening task aims to generate an inbetweening motion $X_i$ , which is of a specified length to bridge the gap between two separated motions $[X_p, X_s]$ . The pipeline for the task is illustrated in Fig.

## [other] 3.2.3 Long-term Motion Generation
Long-term motion sequence generation extends beyond short-term compositional motion pair generation by producing much longer continuous motion, composed of hundreds or thousands of motion segments. While short-term tasks focus on semantics and transitions within a few segments, long-term generation involves monitoring kinetic dynamics, which can impact motion over extended sequences and potentially disrupt motion realism and physical plausibility.

## [experiment] 4 Experiments


## [experiment] 4.1 Implementation and Evaluation Details


## [experiment] 4.1.1 Training and Evaluation Dataset
We use the BABEL-TEACH dataset [4, 12] for training and evaluation, as it provides annotated subsequence pairs essential for long-term motion generation [12, 13, 14], facilitating the learning of transitions between subsequences. These annotated pairs are derived from decomposing fine-grained text subsequence annotations from BABEL [4].

## [experiment] 4.1.2 Evaluation Metrics
We assess the results of **compositional motion pair generation**, **long-term motion generation**, and **conditional motion inbetweening** based on two key aspects: *Fréchet Inception Distance* (FID) for Motion Realism and *Multimodal Distance* (MMD) for Text Alignment, following the T2M [6]

## [experiment] 4.2 Compositional Motion Generation Performance Evaluation


## [other] 4.2.1 Compositional Motion Pair Generation
The compositional motion pair experiment follows the setup illustrated on the left in Fig. 3, with the objective of generating motions $\mathbf{X_p}$ , $\mathbf{X_t}$ , and $\mathbf{X_s}$ based on the corresponding text condition pairs $(C_p, C_s)$ .

## [other] 4.2.2 Long-term Motion Generation
To assess the long-term motion generation performance, we combine all text conditions from the testing dataset into a single extended text sequence of 3,164 texts, and apply comparison models to

## [experiment] 4.3 Motion Inbetweening Performance Evaluation
The unconditional motion inbetweening (UMIB) experiment follows a setup similar to Fig. 4, where a specific number of frames around the transition boundary of testing motion pairs are masked to evaluate various methods for reconstructing the masked motion content.

## [other] 4.4 Ablation Studies and User Studies
We assess the effects of our proposed modules and recommended hyperparameters on compositional motion generation and motion inbetweening tasks. Firstly, the integration of *frame-level tokens* within SPDM and TPDM significantly enhances their performance in denoising *param-level tokens*.

## [conclusion] 5 Conclusion and Future Work
We present the Transitional Phase Diffusion Module (TPDM) and the Semantic Phase Diffusion Module (SPDM), which operate within the periodic latent space generated by the Action-Centric Periodic Autoencoder. These modules inject semantic guidance and neighbouring phase information into the motion denoising process, enabling the generation of semantically meaningful motion clips with smooth transitions.

## [other] References
- <span id="page-9-0"></span>[1] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need.

## [other] A Technical Details of Compositional Phase Diffusion


## [other] A.1 Compositional Motion Generation Algorithm
Algorithm implementation details of the phase diffusion pipeline for the compositional motion generation task in Sec. 3.2.1.

## [other] A.2 Adjustment to T and PE
As discussed in Sec. 3.1.1, we have refined the sinusoidal positional embedding PE and the time window T to support motion autoencoding with variable lengths.

## [other] A.3 Details of SPDM and TPDM
As discussed in Sec. 3.1.2 and Sec.

## [other] A.4 Implementation Details
We apply the emphasis projection with c=15, as demonstrated in GMD [25], to incorporate root trajectory information into the motion representation. Also, our models are designed based on phase latent size Q=512, which serves as both the latent dimension for all diffusion modules and the number of periodic signals in ACT-PAE.

## [experiment] B Conditional Motion Inbetweening Evaluation
Figure 7: Visualization of the **CMIB** with 120 transition boundary frames conditioned with *bend* arms up: preceding motion in blue, transitioning motion in green, and succeeding motion in yellow.

## [other] C Impact Statements
The exploration and application of phase latent spaces in this work contribute to the advancement of deep learning by offering new methodologies for signal processing and multimedia generation. It has no negative impact on society as the focus is on technological improvement rather than datasets that could be sensitive or have privacy implications.
