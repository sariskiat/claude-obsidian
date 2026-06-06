---
type: paper
slug: auto-compressing-networks
title: Auto-Compressing Networks
authors: Vaggelis Dorovatas, Georgios Paraskevopoulos, Alexandros Potamianos
source_path: /Users/saris.kia.adm/.paper-scholar/auto-compressing-networks/2506.09714.md
ingested_at: '2026-06-05 05:46:36'
authors_list: []
sections:
- id: 333
  heading: Abstract
  role: section
  order_index: 0
  summary: Deep neural networks with short residual connections have demonstrated remarkable success across domains, but increasing depth often introduces computational redundancy without corresponding improvements in representation quality. We introduce Auto-Compressing Networks (ACNs), an architectural variant where additive long feedforward connections from each layer to the output replace traditional short residual connections.
- id: 334
  heading: 1 Introduction
  role: section
  order_index: 1
  summary: Deep learning has achieved significant breakthroughs across diverse tasks and domains [\[28,](#page-11-0) [30,](#page-11-1) [6\]](#page-9-0); however, it still lacks the flexibility, robustness, and efficiency of biological networks. Modern models rely on deep architectures with billions of parameters, leading to high computational, storage, and energy costs.
- id: 335
  heading: 2 Auto-Compressing Networks
  role: section
  order_index: 2
  summary: 'In ACNs $^1$ , residual short connections are replaced with long feedforward connections, as described in Eq. 1 for a network of depth L, and shown $^2$ in Table 1:'
- id: 336
  heading: 2.1 Gradient Propagation Across Network Architectures
  role: section
  order_index: 3
  summary: 'In this section, we examine and compare the forward and backward pass (gradient flow) dynamics of three architectures: traditional feedforward networks (FFN), residual networks (ResNet), and the proposed auto-compressing networks (ACN), based on the equations of Table 1. See Appendix C for a detailed derivation of the gradient equations for 1D linear neural networks.'
- id: 337
  heading: 2.2 Emergent Gradient Paths
  role: section
  order_index: 4
  summary: 'The forward and backward components: As shown in Table 1, each gradient $w_i$ (see $\frac{\partial y_*}{\partial w_i}$ in the 3rd column of the respective table) decomposes into forward and backward terms. The forward term determines gradient and forward propagation stability (whether the signal vanishes or explodes), while the backward term influences learning.'
- id: 338
  heading: '3 ACNs in Practice: Information Compression and Gradient Flow'
  role: section
  order_index: 5
  summary: Next, we implement auto-compressing networks on top of state-of-the-art neural architectures across diverse tasks and datasets. We implement ACNs using variants of the Transformer [52] for language and vision tasks and MLP-Mixer [50] for vision tasks.
- id: 339
  heading: 4 ACNs compress more
  role: section
  order_index: 6
  summary: ''
- id: 340
  heading: 4.2 Auto-Compressing Encoder Architectures for Language Modeling
  role: section
  order_index: 8
  summary: In this section, we conduct a preliminary study on the effectiveness of the ACN architecture in general pre-training (masked language modeling with a BERT architecture) followed by fine-tuning. The
- id: 341
  heading: 4.2.1 Masked Language Modeling and Transfer Learning with ACNs
  role: section
  order_index: 9
  summary: We compare the ACN and residual architectures in the standard BERT pre-training and fine-tuning paradigm. Using the original BERT pretraining corpus (BooksCorpus [64] and English Wikipedia), we train both architectures to equivalent loss values; the AC-BERT variant requires two epochs vs one epoch for the residual baseline.
- id: 342
  heading: 5 ACNs generalize better
  role: section
  order_index: 10
  summary: 'While ACNs demonstrate effective parameter reduction through architectural compression, a key question remains: do these compressed representations offer additional benefits beyond parameter efficiency? In this section, we investigate whether the concentrated information in ACNs'' early layers leads to improved generalization capabilities compared to traditional residual architectures.'
- id: 343
  heading: 5.1 Robustness to Input Noise
  role: section
  order_index: 11
  summary: Next, we present results assessing the robustness of ACNs versus residual transformer architectures to input noise. The experiments are performed with the AC-ViT and residual ViT architectures trained on ImageNet-1K.
- id: 344
  heading: 5.2 Robustness to Data Sparsity
  role: section
  order_index: 12
  summary: Next, we experimentally compare the performance of residual and long connections architectures in low-data scenarios. For this purpose, we create a random subset of CIFAR-10 [27] by retaining only 100 samples per class, resulting in a total of 1000 examples.
- id: 345
  heading: 6 ACNs forget less
  role: section
  order_index: 13
  summary: Continual learning involves training models on a sequence of tasks without access to past data, aiming to retain performance on previous tasks while learning new ones [8, 55]. A central challenge in CL is catastrophic forgetting—the tendency of neural networks to overwrite old knowledge when updated with new data.
- id: 346
  heading: 8 Conclusion
  role: section
  order_index: 15
  summary: In this work, we introduced Auto-Compressing Networks (ACNs), an architectural design that organically compresses information into early layers of a neural network during training via long skip connections from each layer to the output, a property we coined as *auto-compression*. Unlike residual networks, ACNs do not require explicit compression objectives or regularization; instead, they leverage architectural design and gradient-based optimization to induce implicit layer-wise training dynamics that drive auto-compression.
- id: 347
  heading: 9 Limitations and Broader Impact
  role: section
  order_index: 16
  summary: Due to resource constraints, our evaluation of ACNs was limited to relatively small-scale tasks, though the architecture consistently performed well across modalities, datasets, and state-of-the-art baselines. Broader validation—including large-scale, self-supervised, and multi-task settings (e.g., language or multimodal models)—is essential to fully understand its capabilities and boundaries.
- id: 348
  heading: 10 Acknowledgments and funding disclosure
  role: section
  order_index: 17
  summary: This work has been partially supported by project MIS-5154714 of the National Recovery and Resilience Plan Greece 2.0 funded by the EU under the NextGenerationEU Program. We acknowledge EuroHPC JU project ID EHPC-AI-2024-A04-051 for use of the supercomputer LEONARDO@ CINECA, Italy.
- id: 349
  heading: References
  role: section
  order_index: 18
  summary: '- <span id="page-9-2"></span>[1] Guillaume Alain and Yoshua Bengio. Understanding intermediate layers using linear classifier probes.'
- id: 350
  heading: Technical Appendices and Supplementary Material
  role: section
  order_index: 19
  summary: ''
- id: 351
  heading: ACN forward pass
  role: section
  order_index: 23
  summary: $$y_A = x_0 + \sum_{i=1}^{L} x_i = x_0 + \sum_{i=1}^{L} \prod_{j=1}^{i} w_j x_0$$ (10)
- id: 352
  heading: F Summary of Results
  role: section
  order_index: 26
  summary: Table [6](#page-18-0) presents the main results comparing residual (Res-) and auto-compressing (AC-) variants across both vision and language tasks. We observe that ACN variants consistently achieve comparable or superior accuracy while requiring significantly fewer parameters, lower computational cost at inference, and reduced storage requirements.
---

# Auto-Compressing Networks

## [section] Abstract
Deep neural networks with short residual connections have demonstrated remarkable success across domains, but increasing depth often introduces computational redundancy without corresponding improvements in representation quality. We introduce Auto-Compressing Networks (ACNs), an architectural variant where additive long feedforward connections from each layer to the output replace traditional short residual connections.

## [section] 1 Introduction
Deep learning has achieved significant breakthroughs across diverse tasks and domains [\[28,](#page-11-0) [30,](#page-11-1) [6\]](#page-9-0); however, it still lacks the flexibility, robustness, and efficiency of biological networks. Modern models rely on deep architectures with billions of parameters, leading to high computational, storage, and energy costs.

## [section] 2 Auto-Compressing Networks
In ACNs $^1$ , residual short connections are replaced with long feedforward connections, as described in Eq. 1 for a network of depth L, and shown $^2$ in Table 1:

## [section] 2.1 Gradient Propagation Across Network Architectures
In this section, we examine and compare the forward and backward pass (gradient flow) dynamics of three architectures: traditional feedforward networks (FFN), residual networks (ResNet), and the proposed auto-compressing networks (ACN), based on the equations of Table 1. See Appendix C for a detailed derivation of the gradient equations for 1D linear neural networks.

## [section] 2.2 Emergent Gradient Paths
The forward and backward components: As shown in Table 1, each gradient $w_i$ (see $\frac{\partial y_*}{\partial w_i}$ in the 3rd column of the respective table) decomposes into forward and backward terms. The forward term determines gradient and forward propagation stability (whether the signal vanishes or explodes), while the backward term influences learning.

## [section] 3 ACNs in Practice: Information Compression and Gradient Flow
Next, we implement auto-compressing networks on top of state-of-the-art neural architectures across diverse tasks and datasets. We implement ACNs using variants of the Transformer [52] for language and vision tasks and MLP-Mixer [50] for vision tasks.

## [section] 4 ACNs compress more


## [section] 4.2 Auto-Compressing Encoder Architectures for Language Modeling
In this section, we conduct a preliminary study on the effectiveness of the ACN architecture in general pre-training (masked language modeling with a BERT architecture) followed by fine-tuning. The

## [section] 4.2.1 Masked Language Modeling and Transfer Learning with ACNs
We compare the ACN and residual architectures in the standard BERT pre-training and fine-tuning paradigm. Using the original BERT pretraining corpus (BooksCorpus [64] and English Wikipedia), we train both architectures to equivalent loss values; the AC-BERT variant requires two epochs vs one epoch for the residual baseline.

## [section] 5 ACNs generalize better
While ACNs demonstrate effective parameter reduction through architectural compression, a key question remains: do these compressed representations offer additional benefits beyond parameter efficiency? In this section, we investigate whether the concentrated information in ACNs' early layers leads to improved generalization capabilities compared to traditional residual architectures.

## [section] 5.1 Robustness to Input Noise
Next, we present results assessing the robustness of ACNs versus residual transformer architectures to input noise. The experiments are performed with the AC-ViT and residual ViT architectures trained on ImageNet-1K.

## [section] 5.2 Robustness to Data Sparsity
Next, we experimentally compare the performance of residual and long connections architectures in low-data scenarios. For this purpose, we create a random subset of CIFAR-10 [27] by retaining only 100 samples per class, resulting in a total of 1000 examples.

## [section] 6 ACNs forget less
Continual learning involves training models on a sequence of tasks without access to past data, aiming to retain performance on previous tasks while learning new ones [8, 55]. A central challenge in CL is catastrophic forgetting—the tendency of neural networks to overwrite old knowledge when updated with new data.

## [section] 8 Conclusion
In this work, we introduced Auto-Compressing Networks (ACNs), an architectural design that organically compresses information into early layers of a neural network during training via long skip connections from each layer to the output, a property we coined as *auto-compression*. Unlike residual networks, ACNs do not require explicit compression objectives or regularization; instead, they leverage architectural design and gradient-based optimization to induce implicit layer-wise training dynamics that drive auto-compression.

## [section] 9 Limitations and Broader Impact
Due to resource constraints, our evaluation of ACNs was limited to relatively small-scale tasks, though the architecture consistently performed well across modalities, datasets, and state-of-the-art baselines. Broader validation—including large-scale, self-supervised, and multi-task settings (e.g., language or multimodal models)—is essential to fully understand its capabilities and boundaries.

## [section] 10 Acknowledgments and funding disclosure
This work has been partially supported by project MIS-5154714 of the National Recovery and Resilience Plan Greece 2.0 funded by the EU under the NextGenerationEU Program. We acknowledge EuroHPC JU project ID EHPC-AI-2024-A04-051 for use of the supercomputer LEONARDO@ CINECA, Italy.

## [section] References
- <span id="page-9-2"></span>[1] Guillaume Alain and Yoshua Bengio. Understanding intermediate layers using linear classifier probes.

## [section] Technical Appendices and Supplementary Material


## [section] ACN forward pass
$$y_A = x_0 + \sum_{i=1}^{L} x_i = x_0 + \sum_{i=1}^{L} \prod_{j=1}^{i} w_j x_0$$ (10)

## [section] F Summary of Results
Table [6](#page-18-0) presents the main results comparing residual (Res-) and auto-compressing (AC-) variants across both vision and language tasks. We observe that ACN variants consistently achieve comparable or superior accuracy while requiring significantly fewer parameters, lower computational cost at inference, and reduced storage requirements.
