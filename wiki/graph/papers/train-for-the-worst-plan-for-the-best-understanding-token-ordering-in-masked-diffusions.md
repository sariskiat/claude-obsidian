---
type: paper
slug: train-for-the-worst-plan-for-the-best-understanding-token-ordering-in-masked-diffusions
title: 'Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions'
authors: Jaeyeon Kim, Kulin Shah, Vasilis Kontonis, Sham Kakade, Sitan Chen
source_path: /Users/saris.kia.adm/.paper-scholar/train-for-the-worst-plan-for-the-best-understanding-token-ordering-in-masked-diffusions/2502.06768.md
ingested_at: '2026-06-05 05:49:34'
authors_list: []
sections:
- id: 537
  heading: Abstract
  role: section
  order_index: 0
  summary: In recent years, masked diffusion models (MDMs) have emerged as a promising alternative approach for generative modeling over discrete domains. Compared to autoregressive models (ARMs), MDMs trade off complexity at training time with flexibility at inference time.
- id: 538
  heading: 1. Introduction
  role: section
  order_index: 1
  summary: 'While diffusion models [\(Ho et al.,](#page-10-0) [2020;](#page-10-0) [Song et al.,](#page-11-0) [2021\)](#page-11-0) are now the dominant approach for generative modeling in continuous domains like image, video, and audio, efforts to extend this methodology to discrete domains like text and proteins [\(Austin et al.,](#page-9-0) [2021;](#page-9-0) [Lou et al.,](#page-10-1) [2024;](#page-10-1) [Hoogeboom](#page-10-2) [et al.,](#page-10-2) [2021b\)](#page-10-2) remain nascent. Among numerous proposals, masked diffusion models (MDMs) [\(Lou et al.,](#page-10-1) [2024;](#page-10-1) [Sahoo](#page-11-1) [et al.,](#page-11-1) [2025;](#page-11-1) [Shi et al.,](#page-11-2) [2024\)](#page-11-2) have emerged as a leading variant, distinguished by a simple and principled objective: to generate samples, learn to reverse a noise process which independently and randomly masks tokens.'
- id: 539
  heading: MDM training t=0 t=1
  role: section
  order_index: 3
  summary: ''
- id: 540
  heading: MDM inferences (Vanilla vs. Adaptive)
  role: section
  order_index: 4
  summary: а М с
- id: 541
  heading: 2.1. Reformulating the training and inference of MDMs
  role: section
  order_index: 5
  summary: In this section, we first discuss training of MDMs and compare it with "left-to-right" order training of autoregressive models in Section 2.1.1. Then, we reformulate vanilla MDM inference in Section 2.1.2 to set the stage for the upcoming discussion.
- id: 542
  heading: Adaptive MDM inference
  role: section
  order_index: 14
  summary: '- (a) Sample a set of masked tokens $S = \mathcal{F}(\theta, x_t) \subseteq \{i \mid x_t^i = 0\}.$ - (b) For each $i \in \mathcal{S}$ , sample $x_s^i \sim p_\theta(x^i|x_t)$ .'
- id: 543
  heading: 4.3. Eliciting sequence-dependent reasoning paths using adaptive MDM inference in logic puzzles
  role: section
  order_index: 17
  summary: In this section, we study the effectiveness of adaptive MDM inference in finding the right reasoning/generation order for tasks where every sequence has a different "natural" order. To do so, we will compare the performance of adaptive MDM inference to that of ARM on Sudoku and Zebra puzzles.
- id: 544
  heading: 4.4. Adaptive MDM inference on natural language tasks
  role: section
  order_index: 18
  summary: 'To examine the effect of different inference strategies on text benchmarks, we adapted LLaDA, the 8B MDM model from [\(Nie et al.,](#page-10-12) [2025\)](#page-10-12). We compare three inference strategies: vanilla, top probability, and top probability margin.'
- id: 545
  heading: 4.5. Easy to hard generalization
  role: section
  order_index: 19
  summary: In the previous section we showed that when the training and inference sequences come from the same distribution, order-agnostic training of MDMs combined with adaptive inference can perform very well on logic puzzles. To evaluate if the model has learned the correct way of solving the puzzles and test the robustness of adaptive inference, we also test the MDMs on harder puzzles than the ones from training, for Sudoku.
- id: 546
  heading: 5. Conclusion
  role: section
  order_index: 20
  summary: In this work, we examined the impact of token generation order on training and inference in MDMs. We provided theoretical and experimental evidence that MDMs train on hard masking problems.
- id: 547
  heading: Impact statement
  role: section
  order_index: 21
  summary: This paper advances the understanding of discrete diffusion models, contributing to the broader field of Machine Learning. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here.
- id: 548
  heading: References
  role: section
  order_index: 22
  summary: '- <span id="page-9-8"></span>Alaoui, A. E.'
- id: 549
  heading: A. Related works
  role: section
  order_index: 23
  summary: Discrete diffusion models. (Continuous) diffusion models were originally built on continuous-space Markov chains with Gaussian transition kernels [\(Sohl-Dickstein et al.,](#page-11-13) [2015;](#page-11-13) [Ho et al.,](#page-10-0) [2020\)](#page-10-0).
- id: 550
  heading: B. Technical details from Section [3](#page-3-0)
  role: section
  order_index: 24
  summary: Notations. Throughout this section, we use x i to denote the i-th coordinate of the vector x and z(j) to denote the j-th example.
- id: 551
  heading: C. Experimental details in Section 3
  role: section
  order_index: 29
  summary: ''
- id: 552
  heading: C.2.1. EXPERIMENT ON L&O-NAE-SAT DISTRIBUTION
  role: section
  order_index: 32
  summary: We consider the L&O-NAE-SAT distribution with (N,P)=(20,280). For each example sequence from L&O-NAE-SAT, we pad the last 212 tokens with an additional token value of 2.
- id: 553
  heading: C.2.2. EXPERIMENT ON TEXT DATA
  role: section
  order_index: 33
  summary: We take a 170M MDM pretrained with text data for a baseline model. To measure the performance imbalance between likelihood modeling tasks
- id: 554
  heading: D. Experimental details in Section 4
  role: section
  order_index: 34
  summary: ''
- id: 555
  heading: D.1.1. EXPERIMENT ON L&O-NAE-SAT DISTRIBUTION
  role: section
  order_index: 36
  summary: 'We consider five instances of L&O-NAE-SAT: (N, P) = (25, 275), (30, 270), (40, 260), (50, 250), (100, 200). For each distribution, we train a 19M MDM and measure the accuracy difference between vanilla inference and adaptive inference using top probability margin.'
- id: 556
  heading: D.1.2. EXPERIMENT ON TEXT DATA
  role: section
  order_index: 37
  summary: '**Top probability margin sampler with temperature.** To modify our inference for text data modeling, which does not have a determined answer, we found that adding a certain level of temperature to the oracle is useful. This is because the top probability margin or the top probability often leads to greedy sampling, which harms the diversity (entropy) of the generated samples.'
- id: 557
  heading: D.2. Experimental details on Sudoku and Zebra puzzles
  role: section
  order_index: 38
  summary: '**Dataset.** For both Sudoku and Zebra puzzles, we use the dataset provided in Shah et al. (2024) to train our model.'
---

# Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions

## [section] Abstract
In recent years, masked diffusion models (MDMs) have emerged as a promising alternative approach for generative modeling over discrete domains. Compared to autoregressive models (ARMs), MDMs trade off complexity at training time with flexibility at inference time.

## [section] 1. Introduction
While diffusion models [\(Ho et al.,](#page-10-0) [2020;](#page-10-0) [Song et al.,](#page-11-0) [2021\)](#page-11-0) are now the dominant approach for generative modeling in continuous domains like image, video, and audio, efforts to extend this methodology to discrete domains like text and proteins [\(Austin et al.,](#page-9-0) [2021;](#page-9-0) [Lou et al.,](#page-10-1) [2024;](#page-10-1) [Hoogeboom](#page-10-2) [et al.,](#page-10-2) [2021b\)](#page-10-2) remain nascent. Among numerous proposals, masked diffusion models (MDMs) [\(Lou et al.,](#page-10-1) [2024;](#page-10-1) [Sahoo](#page-11-1) [et al.,](#page-11-1) [2025;](#page-11-1) [Shi et al.,](#page-11-2) [2024\)](#page-11-2) have emerged as a leading variant, distinguished by a simple and principled objective: to generate samples, learn to reverse a noise process which independently and randomly masks tokens.

## [section] MDM training t=0 t=1


## [section] MDM inferences (Vanilla vs. Adaptive)
а М с

## [section] 2.1. Reformulating the training and inference of MDMs
In this section, we first discuss training of MDMs and compare it with "left-to-right" order training of autoregressive models in Section 2.1.1. Then, we reformulate vanilla MDM inference in Section 2.1.2 to set the stage for the upcoming discussion.

## [section] Adaptive MDM inference
- (a) Sample a set of masked tokens $S = \mathcal{F}(\theta, x_t) \subseteq \{i \mid x_t^i = 0\}.$ - (b) For each $i \in \mathcal{S}$ , sample $x_s^i \sim p_\theta(x^i|x_t)$ .

## [section] 4.3. Eliciting sequence-dependent reasoning paths using adaptive MDM inference in logic puzzles
In this section, we study the effectiveness of adaptive MDM inference in finding the right reasoning/generation order for tasks where every sequence has a different "natural" order. To do so, we will compare the performance of adaptive MDM inference to that of ARM on Sudoku and Zebra puzzles.

## [section] 4.4. Adaptive MDM inference on natural language tasks
To examine the effect of different inference strategies on text benchmarks, we adapted LLaDA, the 8B MDM model from [\(Nie et al.,](#page-10-12) [2025\)](#page-10-12). We compare three inference strategies: vanilla, top probability, and top probability margin.

## [section] 4.5. Easy to hard generalization
In the previous section we showed that when the training and inference sequences come from the same distribution, order-agnostic training of MDMs combined with adaptive inference can perform very well on logic puzzles. To evaluate if the model has learned the correct way of solving the puzzles and test the robustness of adaptive inference, we also test the MDMs on harder puzzles than the ones from training, for Sudoku.

## [section] 5. Conclusion
In this work, we examined the impact of token generation order on training and inference in MDMs. We provided theoretical and experimental evidence that MDMs train on hard masking problems.

## [section] Impact statement
This paper advances the understanding of discrete diffusion models, contributing to the broader field of Machine Learning. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here.

## [section] References
- <span id="page-9-8"></span>Alaoui, A. E.

## [section] A. Related works
Discrete diffusion models. (Continuous) diffusion models were originally built on continuous-space Markov chains with Gaussian transition kernels [\(Sohl-Dickstein et al.,](#page-11-13) [2015;](#page-11-13) [Ho et al.,](#page-10-0) [2020\)](#page-10-0).

## [section] B. Technical details from Section [3](#page-3-0)
Notations. Throughout this section, we use x i to denote the i-th coordinate of the vector x and z(j) to denote the j-th example.

## [section] C. Experimental details in Section 3


## [section] C.2.1. EXPERIMENT ON L&O-NAE-SAT DISTRIBUTION
We consider the L&O-NAE-SAT distribution with (N,P)=(20,280). For each example sequence from L&O-NAE-SAT, we pad the last 212 tokens with an additional token value of 2.

## [section] C.2.2. EXPERIMENT ON TEXT DATA
We take a 170M MDM pretrained with text data for a baseline model. To measure the performance imbalance between likelihood modeling tasks

## [section] D. Experimental details in Section 4


## [section] D.1.1. EXPERIMENT ON L&O-NAE-SAT DISTRIBUTION
We consider five instances of L&O-NAE-SAT: (N, P) = (25, 275), (30, 270), (40, 260), (50, 250), (100, 200). For each distribution, we train a 19M MDM and measure the accuracy difference between vanilla inference and adaptive inference using top probability margin.

## [section] D.1.2. EXPERIMENT ON TEXT DATA
**Top probability margin sampler with temperature.** To modify our inference for text data modeling, which does not have a determined answer, we found that adding a certain level of temperature to the oracle is useful. This is because the top probability margin or the top probability often leads to greedy sampling, which harms the diversity (entropy) of the generated samples.

## [section] D.2. Experimental details on Sudoku and Zebra puzzles
**Dataset.** For both Sudoku and Zebra puzzles, we use the dataset provided in Shah et al. (2024) to train our model.
