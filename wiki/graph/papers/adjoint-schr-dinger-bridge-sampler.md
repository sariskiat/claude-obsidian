---
type: paper
slug: adjoint-schr-dinger-bridge-sampler
title: Adjoint Schrödinger Bridge Sampler
authors: Guan-Horng Liu, Jaemoo Choi, Yongxin Chen, Benjamin Kurt Miller, Ricky T. Q. Chen
source_path: /Users/saris.kia.adm/.paper-scholar/adjoint-schr-dinger-bridge-sampler/2506.22565.md
ingested_at: '2026-06-05 05:50:16'
authors_list: []
sections:
- id: 616
  heading: 1 Introduction
  role: section
  order_index: 0
  summary: 'Sampling from Boltzmann distributions is a fundamental problem in computational science, with widespread applications in Bayesian inference, statistical physics, and chemistry (Box and Tiao, 2011; Binder et al., 1992; Tuckerman, 2023). Mathematically, we aim to sample from a target distribution $\nu(x)$ known up to a unnormalized, often differentiable, energy function $E(x): \mathcal{X} \subseteq \mathbb{R}^d \to \mathbb{R}$ ,'
- id: 617
  heading: 2 Preliminary
  role: section
  order_index: 1
  summary: We revisit the memoryless condition introduced by Domingo-Enrich et al. (2025) and examine its impact on the constructions of SOC-based diffusion samplers (Zhang and Chen, 2022; Havens et al., 2025), which are closely related to our ASBS.
- id: 618
  heading: 3 Adjoint Schrödinger Bridge Sampler
  role: section
  order_index: 2
  summary: We introduce a new diffusion sampler by solving the SB problem (3), where the target distribution $\nu(x)$ is given by its energy function E(x) rather than explicit samples. All proofs are left in Section B.
- id: 619
  heading: 3.1 SOC Characteristics of the SB Problem
  role: section
  order_index: 3
  summary: 'The SB problem (3)—as an optimization problem with distribution constraints—is widely explored in optimal transport, stochastic control, and recently machine learning (Léonard, 2012; Chen et al., 2021; De Bortoli et al., 2021). Its kinetic-optimal drift $u^*$ satisfies the following optimality equations:'
- id: 620
  heading: 3.3 Alternating Optimization with Adjoint and Corrector Matching
  role: section
  order_index: 6
  summary: 'Building upon the theoretical characterization in [Section 3.2,](#page-4-7) we aim to design a learning algorithm that finds a diffusion sampler satisfying [\(12\)](#page-4-3) and [\(13\)](#page-4-6), which correspond to two simple matching-based objectives. However, these matching objectives cannot be naively implemented due to their interdependency: Solving [\(12\)](#page-4-3) for the kinetic-optimal drift u ⋆ requires knowing ∇ log ˆφ1.'
- id: 621
  heading: 4 Theoretical Analysis
  role: section
  order_index: 7
  summary: We provide the proof of Theorem 3.2 and highlight theoretical insights throughout. While ASBS is specialized to a degenerate base drift $f_t := 0$ , all theoretical results here apply to general $f_t$ .
- id: 622
  heading: 5 Related Works
  role: section
  order_index: 8
  summary: '**Data-driven Schrödinger Bridges** The SB problem has attracted notable interests in machine learning due to its connection to diffusion-based generative models (Wang et al., 2021). Earlier methods implemented classical IPF algorithms (De Bortoli et al., 2021; Vargas et al., 2021; Chen et al., 2022), with scalability later enhanced by bridge matching-based methods (Shi et al., 2023; Liu et al., 2024).'
- id: 623
  heading: 6 Experiments
  role: section
  order_index: 9
  summary: Benchmarks We evaluate our ASBS on three classes of multi-particle energy functions E(x).
- id: 624
  heading: 7 Conclusion and Limitation
  role: section
  order_index: 10
  summary: We introduced Adjoint Schrödinger Bridge Sampler (ASBS), a new diffusion sampler for Boltzmann distributions that solves general SB problems given only target energy functions. ASBS is based on a scalable matching framework, converges theoretically to the global solution, and performs superiorly across various benchmarks.
- id: 625
  heading: Acknowledgements
  role: section
  order_index: 11
  summary: The authors would like to thank Aaron Havens, Juno Nam, Xiang Fu, Bing Yan, Brandon Amos, and Brian Karrer for the helpful discussions and comments.
- id: 626
  heading: References
  role: section
  order_index: 12
  summary: '- <span id="page-11-4"></span>Julius Berner, Lorenz Richter, and Karen Ullrich. An optimal control perspective on diffusion-based generative modeling.'
- id: 627
  heading: Contents
  role: section
  order_index: 13
  summary: ''
- id: 628
  heading: A.2 Schrödinger Bridge (SB)
  role: section
  order_index: 16
  summary: 'In this subsection, we provide additional clarification on SB and specifically the derivation of [\(13\)](#page-4-6). Recall the optimality equations of SB in [\(8\)](#page-3-2):'
- id: 629
  heading: B Proofs
  role: section
  order_index: 17
  summary: ''
- id: 630
  heading: D.1.1 Energy functions
  role: section
  order_index: 24
  summary: 'In this section, we provide the exact setup for our synthetic energy experiments in Table 2. We consider four synthetic energy functions that have been widely used in recent literature to benchmark sampling and generative algorithms: MW-5, DW-4, LJ-13, and LJ-55.'
- id: 631
  heading: D.1.2 Baselines
  role: section
  order_index: 25
  summary: Here, we outline the procedure used to obtain the values reported in Table 2 for the baseline methods.
- id: 632
  heading: D.1.3 Evaluation Metrics
  role: section
  order_index: 26
  summary: 'In this subsection, we outline the evaluation criteria used to quantitatively assess the quality of samples generated from synthetic energy functions. We employ three primary metrics: Sinkhorn distance, geometric $W_2$ , and energy $W_2$ , each designed to capture different aspects of distributional similarity between generated and ground truth samples.'
- id: 633
  heading: Amortized conformer generation
  role: section
  order_index: 28
  summary: In this subsection, we provide some context for the experimental results found in Table 4 regarding the generation of conformers.
- id: 634
  heading: D.4 Additional Experiments and Discussions
  role: section
  order_index: 29
  summary: Ablation study between AS and ASBS using the same EGNN For the amortized conformer generation task in Table 4, we use an EGNN architecture with 20 layers, whereas AS employs the same architecture with 12 layers. In Table 6, we report the results of ASBS using the same 12-layer EGNN as AS.
---

# Adjoint Schrödinger Bridge Sampler

## [section] 1 Introduction
Sampling from Boltzmann distributions is a fundamental problem in computational science, with widespread applications in Bayesian inference, statistical physics, and chemistry (Box and Tiao, 2011; Binder et al., 1992; Tuckerman, 2023). Mathematically, we aim to sample from a target distribution $\nu(x)$ known up to a unnormalized, often differentiable, energy function $E(x): \mathcal{X} \subseteq \mathbb{R}^d \to \mathbb{R}$ ,

## [section] 2 Preliminary
We revisit the memoryless condition introduced by Domingo-Enrich et al. (2025) and examine its impact on the constructions of SOC-based diffusion samplers (Zhang and Chen, 2022; Havens et al., 2025), which are closely related to our ASBS.

## [section] 3 Adjoint Schrödinger Bridge Sampler
We introduce a new diffusion sampler by solving the SB problem (3), where the target distribution $\nu(x)$ is given by its energy function E(x) rather than explicit samples. All proofs are left in Section B.

## [section] 3.1 SOC Characteristics of the SB Problem
The SB problem (3)—as an optimization problem with distribution constraints—is widely explored in optimal transport, stochastic control, and recently machine learning (Léonard, 2012; Chen et al., 2021; De Bortoli et al., 2021). Its kinetic-optimal drift $u^*$ satisfies the following optimality equations:

## [section] 3.3 Alternating Optimization with Adjoint and Corrector Matching
Building upon the theoretical characterization in [Section 3.2,](#page-4-7) we aim to design a learning algorithm that finds a diffusion sampler satisfying [\(12\)](#page-4-3) and [\(13\)](#page-4-6), which correspond to two simple matching-based objectives. However, these matching objectives cannot be naively implemented due to their interdependency: Solving [\(12\)](#page-4-3) for the kinetic-optimal drift u ⋆ requires knowing ∇ log ˆφ1.

## [section] 4 Theoretical Analysis
We provide the proof of Theorem 3.2 and highlight theoretical insights throughout. While ASBS is specialized to a degenerate base drift $f_t := 0$ , all theoretical results here apply to general $f_t$ .

## [section] 5 Related Works
**Data-driven Schrödinger Bridges** The SB problem has attracted notable interests in machine learning due to its connection to diffusion-based generative models (Wang et al., 2021). Earlier methods implemented classical IPF algorithms (De Bortoli et al., 2021; Vargas et al., 2021; Chen et al., 2022), with scalability later enhanced by bridge matching-based methods (Shi et al., 2023; Liu et al., 2024).

## [section] 6 Experiments
Benchmarks We evaluate our ASBS on three classes of multi-particle energy functions E(x).

## [section] 7 Conclusion and Limitation
We introduced Adjoint Schrödinger Bridge Sampler (ASBS), a new diffusion sampler for Boltzmann distributions that solves general SB problems given only target energy functions. ASBS is based on a scalable matching framework, converges theoretically to the global solution, and performs superiorly across various benchmarks.

## [section] Acknowledgements
The authors would like to thank Aaron Havens, Juno Nam, Xiang Fu, Bing Yan, Brandon Amos, and Brian Karrer for the helpful discussions and comments.

## [section] References
- <span id="page-11-4"></span>Julius Berner, Lorenz Richter, and Karen Ullrich. An optimal control perspective on diffusion-based generative modeling.

## [section] Contents


## [section] A.2 Schrödinger Bridge (SB)
In this subsection, we provide additional clarification on SB and specifically the derivation of [\(13\)](#page-4-6). Recall the optimality equations of SB in [\(8\)](#page-3-2):

## [section] B Proofs


## [section] D.1.1 Energy functions
In this section, we provide the exact setup for our synthetic energy experiments in Table 2. We consider four synthetic energy functions that have been widely used in recent literature to benchmark sampling and generative algorithms: MW-5, DW-4, LJ-13, and LJ-55.

## [section] D.1.2 Baselines
Here, we outline the procedure used to obtain the values reported in Table 2 for the baseline methods.

## [section] D.1.3 Evaluation Metrics
In this subsection, we outline the evaluation criteria used to quantitatively assess the quality of samples generated from synthetic energy functions. We employ three primary metrics: Sinkhorn distance, geometric $W_2$ , and energy $W_2$ , each designed to capture different aspects of distributional similarity between generated and ground truth samples.

## [section] Amortized conformer generation
In this subsection, we provide some context for the experimental results found in Table 4 regarding the generation of conformers.

## [section] D.4 Additional Experiments and Discussions
Ablation study between AS and ASBS using the same EGNN For the amortized conformer generation task in Table 4, we use an EGNN architecture with 20 layers, whereas AS employs the same architecture with 12 layers. In Table 6, we report the results of ASBS using the same 12-layer EGNN as AS.
