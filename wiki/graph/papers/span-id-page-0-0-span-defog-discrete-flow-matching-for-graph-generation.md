---
type: paper
slug: span-id-page-0-0-span-defog-discrete-flow-matching-for-graph-generation
title: 'DeFoG: Discrete Flow Matching for Graph Generation'
authors: Yiming Qin, Manuel Madeira, Dorina Thanou, Pascal Frossard
source_path: .paper-scholar/span-id-page-0-0-span-defog-discrete-flow-matching-for-graph-generation
ingested_at: '2026-06-05 05:51:53'
authors_list: []
sections:
- id: 787
  heading: Abstract
  role: abstract
  order_index: 0
  summary: Graph generative models are essential across diverse scientific domains by capturing complex distributions over relational data. Among them, graph diffusion models achieve superior performance but face inefficient sampling and limited flexibility due to the tight coupling between training and sampling stages.
- id: 788
  heading: 1. Introduction
  role: introduction
  order_index: 1
  summary: Graph generation has become a fundamental task across diverse fields, from molecular chemistry to social network analysis, due to graphs' capacity to represent complex relationships and generate realistic structured data. Diffusion-based graph generative models [\(Niu et al.,](#page-10-0) [2020;](#page-10-0) [Jo et al.,](#page-9-0) [2022\)](#page-9-0), particularly those tailored for discrete data [\(Vignac et al.,](#page-11-0) [2022\)](#page-11-0), have emerged as compelling ap-
- id: 789
  heading: <span id="page-1-2"></span>2. Background
  role: other
  order_index: 2
  summary: In generative modeling, the primary goal is to generate new data samples from the underlying distribution that produced the original data, $p_{\rm data}$ . An effective approach is to learn a mapping between a simpler distribution $p_{\epsilon}$ that can be easily sampled, and $p_{\rm data}$ .
- id: 790
  heading: <span id="page-2-3"></span>3. DeFoG Framework
  role: method
  order_index: 3
  summary: In this section, we present DeFoG (Discrete Flow Matching on Graphs), a novel iterative refinement framework for graph generation that leverages the decoupling of training and sampling stages. We begin by describing its noising and denoising processes, highlighting how they enable this disentanglement, as illustrated in Figure 1b.
- id: 791
  heading: 3.1. Learning Discrete Flows over Graphs
  role: other
  order_index: 4
  summary: We instantiate undirected graphs with N nodes as $G=(x^{1:n:N},e^{1:i< j:N})$ , where $x^{1:n:N}=(x^{(n)})_{1\leq n\leq N}$ and $e^{1:i< j:N}=(e^{(ij)})_{1\leq i< j\leq N}$ denote the node and edge sets, respectively, with $x^{(n)}\in\mathcal{X}=\{1,\ldots,X\}$ and $e^{(ij)}\in\mathcal{E}=\{1,\ldots,E\}$ . Following standard practice in the field (Vignac et al., 2022; Xu et al., 2024; Siraudin et al., 2024), we consider an edge between every pair of nodes, where one of the edge categories explicitly represents the absence of an edge (i.e., a "non-existing" edge).)
- id: 792
  heading: Algorithm 1 DeFoG Training
  role: other
  order_index: 5
  summary: '1: Input: Graph dataset $$\mathcal{D} = \{G^1, \dots, G^M\}$$'
- id: 793
  heading: Algorithm 2 DeFoG Sampling
  role: other
  order_index: 6
  summary: '``` 1: Input: # graphs to sample S'
- id: 794
  heading: <span id="page-3-0"></span>3.2. Design Space of DeFoG
  role: method
  order_index: 7
  summary: As described in the previous section, DeFoG benefits from greater flexibility due to its training-sampling decoupling.
- id: 795
  heading: 3.3. Permutation Invariance Guarantees
  role: other
  order_index: 8
  summary: Lemma 3 (Node Permutation Equivariance and Invariance Properties of DeFoG). *For any permutation equivariant denoising neural network, the loss function of DeFoG is permutation invariant, and its sampling probability is permutation invariant.*
- id: 796
  heading: 4. Related Work
  role: background
  order_index: 9
  summary: Graph Generative Models Graph generation has applications across various domains, including molecular generation [\(Mercado et al.,](#page-10-5) [2021\)](#page-10-5), combinatorial optimization [\(Sun & Yang,](#page-11-1) [2024\)](#page-11-1), and inverse protein folding [\(Yi](#page-11-3) [et al.,](#page-11-3) [2024\)](#page-11-3). Existing methods for this task generally fall into two main categories.
- id: 797
  heading: <span id="page-5-0"></span>5. Experiments
  role: experiment
  order_index: 10
  summary: This section highlights DeFoG's SOTA performance, enabled by its highly decoupled framework and effective sampling methods. We present DeFoG's performance in generating graphs with diverse topological structures and in molecular datasets with rich prior information.
- id: 798
  heading: <span id="page-5-1"></span>5.1. Graph Generation Performance
  role: other
  order_index: 11
  summary: Synthetic Graph Generation We evaluate DeFoG using the widely adopted *Planar*, *SBM* [\(Martinkus et al.,](#page-10-9) [2022\)](#page-10-9), and *Tree* datasets [\(Bergmeister et al.,](#page-8-8) [2023\)](#page-8-8), along with the associated evaluation methodology. In Tab.
- id: 799
  heading: 5.2. Efficiency Improvement
  role: experiment
  order_index: 12
  summary: We now show that DeFoG enhances both training and sampling efficiency significantly across diverse datasets.
- id: 800
  heading: 5.3. Ablations
  role: other
  order_index: 13
  summary: Here, we focus on evaluating the impact of different sampling approaches. We start from the vanilla sampling setup and sweep over sample distortion, target guidance, or stochasticity independently.
- id: 801
  heading: 6. Conclusion
  role: conclusion
  order_index: 14
  summary: We introduce DeFoG, a novel discrete flow matching framework for graphs. This formulation enables training-sampling decoupling, which we ground theoretically to ensure faithful graph distribution modeling.
- id: 802
  heading: Acknowledgements
  role: other
  order_index: 15
  summary: We would like to thank Clement Vignac and Andrew ´ Campbell for the useful discussions and suggestions.
- id: 803
  heading: Impact Statement
  role: other
  order_index: 16
  summary: The primary objective of this paper is to advance graph generation under a more flexible framework, with applications spanning general graph generation, molecular design, and digital pathology. The ability to generate graphs with discrete labels can have broad-reaching implications for fields such as drug discovery and diagnostic technologies.
- id: 804
  heading: References
  role: other
  order_index: 17
  summary: '- <span id="page-8-0"></span>Asthana, R., Conrad, J., Dawoud, Y., Ortmanns, M., and Belagiannis, V. Multi-conditioned graph diffusion for neural architecture search.'
- id: 805
  heading: Appendix Overview
  role: other
  order_index: 18
  summary: 'In the Appendix, we provide additional details organized as follows:'
- id: 806
  heading: <span id="page-13-0"></span>A. Contextualizing Related Research
  role: background
  order_index: 19
  summary: In this section, we further contextualize DeFoG within the scope of related work. We begin by introducing the methods used for comparison with DeFoG in Appendix [A.1.](#page-13-1) Subsequently, we outline the key distinctions between DeFoG and existing diffusion-based graph generative models in Appendix [A.2.](#page-13-2)
- id: 807
  heading: <span id="page-13-1"></span>A.1. Overview of Compared Methods
  role: method
  order_index: 20
  summary: 'In Sec. [5,](#page-5-0) we evaluate DeFoG against a diverse set of graph generative models, which we introduce below:'
- id: 808
  heading: <span id="page-13-2"></span>A.2. DeFoG and Graph Diffusion Models
  role: other
  order_index: 21
  summary: In this section, we contextualize DeFoG in relation to existing graph diffusion models.
- id: 809
  heading: A.2.1. FROM CONTINUOUS TO DISCRETE STATE-SPACES
  role: other
  order_index: 22
  summary: Early diffusion-based graph generative models extended continuous diffusion and score-based methods from image generation to graphs by relaxing adjacency matrices into continuous state-spaces [\(Niu et al.,](#page-10-0) [2020;](#page-10-0) [Jo et al.,](#page-9-0) [2022\)](#page-9-0). However, this approach overlooks the inherent discreteness of graph-structured data, resulting in topologically uninformed noising processes.
- id: 810
  heading: A.2.2. From Discrete to Continuous time
  role: other
  order_index: 23
  summary: The initial discrete-time diffusion frameworks for graph generation (Vignac et al., 2022; Haefeli et al., 2022) were built upon Discrete Denoising Diffusion Probabilistic Models (D3PMs) (Austin et al., 2021), which operate with a fixed partitioning of time. This discretization constrains the model to denoise at specific time points and ties the sampling process to the same fixed time steps used during training, leading to a rigid coupling between the training and sampling stages.
- id: 811
  heading: A.2.3. From Continuous-Time Discrete Diffusion to Discrete Flow Matching
  role: other
  order_index: 24
  summary: While both continuous-time discrete diffusion and discrete flow matching (DFM) share the CTMC formulation for the denoising process, they differ fundamentally in the formulation of the noising process. Continuous-time discrete diffusion-based graph generative models (Xu et al., 2024; Siraudin et al., 2024) define the noising process as a CTMC, akin to the denoising process.
- id: 812
  heading: A.2.4. MIXED INTEGRATION OF CONTINUOUS AND DISCRETE STATE-SPACES
  role: other
  order_index: 25
  summary: Integrating continuous and categorical data within graph generative models is an important challenge, as many real-world applications involve heterogeneous data types (e.g., molecular graphs containing atomic coordinates alongside categorical atom and bond types). A recent example addressing this challenge is GBD (Liu et al., 2024), which incorporates beta diffusion to jointly model both continuous and discrete variables.
- id: 813
  heading: <span id="page-15-1"></span>B. Sample Optimization
  role: other
  order_index: 26
  summary: In this section, we explore the proposed sampling optimization in more detail. We start by analysing the different time distortion functions in Appendix B.1.
- id: 814
  heading: <span id="page-15-0"></span>B.1. Time Distortion Functions
  role: other
  order_index: 27
  summary: In Sec. 3, we explore the utilization of different *distortion functions*, i.e., functions that are used to transform time.
- id: 815
  heading: <span id="page-16-1"></span>B.2. Target Guidance
  role: other
  order_index: 28
  summary: In this section, we demonstrate that the proposed *target guidance* design for the rate matrices violates the Kolmogorov equation with an error that is linear in $\omega$ . This result indicates that a small guidance factor effectively helps fit the distribution, whereas a larger guidance factor, as shown in Figure 9, while enhancing topological properties such as planarity, increases the distance between generated and training data on synthetic datasets according to the metrics of average ratio.
- id: 816
  heading: <span id="page-18-0"></span>B.3. Detailed Balance, Prior Incorporation, and Stochasticity
  role: other
  order_index: 29
  summary: Campbell et al. (2024) show that although their $z_1$ -conditional formulation of $R_t^*$ generates $p_{t|1}$ , it does not span the full space of valid rate matrices — those that satisfy the conditional Kolmogorov equation (Eq.
- id: 817
  heading: <span id="page-19-0"></span>B.4. Hyperparameter Optimization Pipeline
  role: other
  order_index: 30
  summary: A significant advantage of flow matching methods is their inherently greater flexibility in the sampling process compared to diffusion models, as they are more disentangled from the training stage. Each of the proposed optimization strategies exposed in Sec.
- id: 818
  heading: <span id="page-21-0"></span>B.5. Understanding the Sampling Dynamics of $R_t^*$
  role: other
  order_index: 31
  summary: In this section, we aim to provide deeper intuition into the sampling dynamics imposed by the design of $R_t^*$ , as proposed by (Campbell et al., 2024). The explicit formulation of $R_t^*$ can be found in Eq.
- id: 819
  heading: <span id="page-22-0"></span>B.6. Performance Improvement for Undertrained Models
  role: other
  order_index: 32
  summary: In this section, we present the performance of a model trained on the QM9 dataset and the Planar dataset using only 30% of the epochs compared to the final model being reported. We employ the same hyperparameters as in Tab.
- id: 820
  heading: <span id="page-27-1"></span>C. Train Optimization
  role: other
  order_index: 33
  summary: In this section, we provide a more detailed analysis of the influence of the various training optimization strategies introduced in Sec. 3.
- id: 821
  heading: <span id="page-27-0"></span>C.1. Initial Distributions
  role: other
  order_index: 34
  summary: Under DeFoG's framework, the noising process for each dimension is modeled as a linear interpolation between the clean data distribution (the one-hot representation of the current state) and an initial distribution, $p_0$ . As such, it is intuitive that different initial distributions result in varying performances, depending on the denoising dynamics they induce.
- id: 822
  heading: <span id="page-28-0"></span>C.2. Interaction between Sample and Train Distortions
  role: other
  order_index: 35
  summary: From Appendix B.4, we observe that time distortions applied during the sampling stage can significantly affect performance. This suggests that graph discrete flow models do not behave evenly across time and are more sensitive to specific time intervals, where generative performance benefits from finer updates achieved by using smaller time steps.
- id: 823
  heading: <span id="page-30-1"></span>D. Theoretical Results
  role: experiment
  order_index: 36
  summary: In this section, we provide the proofs of the different theoretical results of the paper. First, we provide results that are domain agnostic, i.e., that hold for general discrete data, and then instantiate them for the specific case of graphs, yielding Theorems 1 and 2.
- id: 824
  heading: <span id="page-30-0"></span>D.1. Theoretical Results on the Generative Framework
  role: method
  order_index: 37
  summary: In Appendix D.1.1, we provide a proof of the bounded estimation error of the rate matrix for general discrete data, and instantiate it for graphs, yielding Theorem 1. Then, in Appendix D.1.2, we prove the bounded deviation of the generated distribution for general discrete data as well, and instantiate it for graphs, yielding Theorem 2.
- id: 825
  heading: <span id="page-30-2"></span>D.1.1. BOUNDED ESTIMATION ERROR OF UNCONDITIONAL MULTIVARIATE RATE MATRIX
  role: other
  order_index: 38
  summary: We start by introducing two important concepts, which will reveal useful for the proof of the intended result.
- id: 826
  heading: <span id="page-34-0"></span>D.1.2. BOUNDED DEVIATION OF THE GENERATED DISTRIBUTION
  role: other
  order_index: 39
  summary: As in Appendix D.1.1, we start by introducing the necessary concepts that will reveal useful for the proof of the intended result.
- id: 827
  heading: D.1.3. CRITICAL ANALYSIS AND POSITIONING OF THEOREM 1 AND THEOREM 2
  role: other
  order_index: 40
  summary: Theorem 1 establishes that minimizing the cross-entropy (CE) loss directly corresponds to minimizing an upper bound on the rate matrix estimation error. This result provides a direct and principled justification for using the CE loss, as it promotes accurate sampling from the underlying CTMC.
- id: 828
  heading: D.2. Theoretical Results on Architectural Expressivity
  role: experiment
  order_index: 41
  summary: We now proceed to the graph specific theoretical results.
- id: 829
  heading: <span id="page-41-0"></span>D.2.1. NODE PERMUTATION EQUIVARIANCE AND INVARIANCE PROPERTIES
  role: other
  order_index: 42
  summary: The different components of a graph generative model have to respect different graph symmetries. For example, the permutation equivariance of the model architecture ensures the output changes consistently with any reordering of input nodes, while permutation-invariant loss evaluates the model's performance consistently across isomorphic graphs, regardless of node order.
- id: 830
  heading: D.2.2. ADDITIONAL FEATURES EXPRESSIVITY
  role: other
  order_index: 43
  summary: This section explains the expressivity of the RRWP features used in DeFoG. We summarize the findings of Ma et al.
- id: 831
  heading: <span id="page-45-0"></span>E. Conditional Generation
  role: other
  order_index: 44
  summary: In this section, we describe how to seamlessly integrate DeFoG with existing methods for CTMC-based conditioning mechanisms. In this setting, all the examples are assumed to have a label.
- id: 832
  heading: <span id="page-46-1"></span>F. Experimental Details
  role: experiment
  order_index: 45
  summary: This section provides further details on the experimental settings used in the paper.
- id: 833
  heading: <span id="page-46-0"></span>F.1. Architecture and Additional Features
  role: other
  order_index: 46
  summary: '**Denoising Neural Architecture** DeFoG''s denoising neural network takes a noisy graph $G_t$ as input and predicts the clean marginal probability for each node $x^{(n)}$ via $p_{1|t}^{\theta,(n)}(\cdot|G_t)$ and for each edge $e^{(ij)}$ via $p_{1|t}^{\theta,(ij)}(\cdot|G_t)$ . This formulation boils down the graph generative task to a graph-to-graph mapping.'
- id: 834
  heading: <span id="page-46-2"></span>F.2. Dataset Details
  role: other
  order_index: 47
  summary: ''
- id: 835
  heading: F.2.1. SYNTHETIC DATASETS
  role: other
  order_index: 48
  summary: Here, we describe the datasets employed in our experiments and outline the specific metrics used to evaluate model performance on each dataset. Additional visualizations of example graphs from each dataset, along with generated graphs, are provided in Figures 16 to 18.
- id: 836
  heading: <span id="page-47-0"></span>F.2.2. MOLECULAR DATASETS
  role: other
  order_index: 49
  summary: '**Description** Molecular generation is a key real-world application of graph generation. It poses a challenging task to current graph generation models to their rich chemistry-specific information, involving several nodes and edges classes and leaning how to generate them jointly, and more complex evaluation pipelines.'
- id: 837
  heading: F.2.3. DIGITAL PATHOLOGY DATASETS
  role: other
  order_index: 50
  summary: '**Description** Graphs, with their natural ability to represent relational data, are widely used to capture spatial biological dependencies in tissue images. This approach has proven successful in digital pathology tasks such as microenvironment classification (Wu et al., 2022), cancer classification (Pati et al., 2022), and decision explainability (Jaume et al., 2020).'
- id: 838
  heading: F.3. Resources
  role: other
  order_index: 51
  summary: The training and sampling times for the different datasets explored in this paper are provided in Tab. [4.](#page-48-1) All the experiments in this work were run on a single NVIDIA A100-SXM4-80GB GPU.
- id: 839
  heading: F.4. Hyperparameter Tuning
  role: other
  order_index: 52
  summary: The default hyperparameters for training and sampling for each dataset can be found in the provided code repository. In Tab.
- id: 840
  heading: <span id="page-52-1"></span>G. Additional Results
  role: experiment
  order_index: 53
  summary: First, we discuss conditional generation for real-world digital pathology graph generation. Next, we provide comprehensive tables detailing results on synthetic datasets in Appendix G.2.
- id: 841
  heading: <span id="page-52-0"></span>G.1. Conditional Generation
  role: other
  order_index: 54
  summary: 'Table 6: TLS conditional generation results.'
- id: 842
  heading: <span id="page-52-2"></span>G.2. Synthetic Graph Generation
  role: other
  order_index: 55
  summary: 'In Tab. 7, we present the full results for DeFoG for the three different datasets: planar, tree, and SBM.'
- id: 843
  heading: <span id="page-54-1"></span>G.3. Molecular Graph Generation
  role: other
  order_index: 56
  summary: For the molecular generation tasks, we begin by examining the results for QM9, considering both implicit and explicit hydrogens (Vignac et al., 2022). In the implicit case, hydrogen atoms are inferred to complete the valencies, while in the explicit case, hydrogens must be explicitly modeled, making it an inherently more challenging task.
- id: 844
  heading: <span id="page-56-0"></span>G.4. Impact of Additional Features
  role: other
  order_index: 57
  summary: In graph diffusion methods, the task of graph generation is decomposed into a mapping of a graph to a set of marginal probabilities for each node and edge. This problem is typically addressed using a Graph Transformer architecture, which is augmented with additional features to capture structural aspects that the base architecture might struggle to model effectively (Vignac et al., 2022; Xu et al., 2024; Siraudin et al., 2024) otherwise.
---

# DeFoG: Discrete Flow Matching for Graph Generation

## [abstract] Abstract
Graph generative models are essential across diverse scientific domains by capturing complex distributions over relational data. Among them, graph diffusion models achieve superior performance but face inefficient sampling and limited flexibility due to the tight coupling between training and sampling stages.

## [introduction] 1. Introduction
Graph generation has become a fundamental task across diverse fields, from molecular chemistry to social network analysis, due to graphs' capacity to represent complex relationships and generate realistic structured data. Diffusion-based graph generative models [\(Niu et al.,](#page-10-0) [2020;](#page-10-0) [Jo et al.,](#page-9-0) [2022\)](#page-9-0), particularly those tailored for discrete data [\(Vignac et al.,](#page-11-0) [2022\)](#page-11-0), have emerged as compelling ap-

## [other] <span id="page-1-2"></span>2. Background
In generative modeling, the primary goal is to generate new data samples from the underlying distribution that produced the original data, $p_{\rm data}$ . An effective approach is to learn a mapping between a simpler distribution $p_{\epsilon}$ that can be easily sampled, and $p_{\rm data}$ .

## [method] <span id="page-2-3"></span>3. DeFoG Framework
In this section, we present DeFoG (Discrete Flow Matching on Graphs), a novel iterative refinement framework for graph generation that leverages the decoupling of training and sampling stages. We begin by describing its noising and denoising processes, highlighting how they enable this disentanglement, as illustrated in Figure 1b.

## [other] 3.1. Learning Discrete Flows over Graphs
We instantiate undirected graphs with N nodes as $G=(x^{1:n:N},e^{1:i< j:N})$ , where $x^{1:n:N}=(x^{(n)})_{1\leq n\leq N}$ and $e^{1:i< j:N}=(e^{(ij)})_{1\leq i< j\leq N}$ denote the node and edge sets, respectively, with $x^{(n)}\in\mathcal{X}=\{1,\ldots,X\}$ and $e^{(ij)}\in\mathcal{E}=\{1,\ldots,E\}$ . Following standard practice in the field (Vignac et al., 2022; Xu et al., 2024; Siraudin et al., 2024), we consider an edge between every pair of nodes, where one of the edge categories explicitly represents the absence of an edge (i.e., a "non-existing" edge).)

## [other] Algorithm 1 DeFoG Training
1: Input: Graph dataset $$\mathcal{D} = \{G^1, \dots, G^M\}$$

## [other] Algorithm 2 DeFoG Sampling
``` 1: Input: # graphs to sample S

## [method] <span id="page-3-0"></span>3.2. Design Space of DeFoG
As described in the previous section, DeFoG benefits from greater flexibility due to its training-sampling decoupling.

## [other] 3.3. Permutation Invariance Guarantees
Lemma 3 (Node Permutation Equivariance and Invariance Properties of DeFoG). *For any permutation equivariant denoising neural network, the loss function of DeFoG is permutation invariant, and its sampling probability is permutation invariant.*

## [background] 4. Related Work
Graph Generative Models Graph generation has applications across various domains, including molecular generation [\(Mercado et al.,](#page-10-5) [2021\)](#page-10-5), combinatorial optimization [\(Sun & Yang,](#page-11-1) [2024\)](#page-11-1), and inverse protein folding [\(Yi](#page-11-3) [et al.,](#page-11-3) [2024\)](#page-11-3). Existing methods for this task generally fall into two main categories.

## [experiment] <span id="page-5-0"></span>5. Experiments
This section highlights DeFoG's SOTA performance, enabled by its highly decoupled framework and effective sampling methods. We present DeFoG's performance in generating graphs with diverse topological structures and in molecular datasets with rich prior information.

## [other] <span id="page-5-1"></span>5.1. Graph Generation Performance
Synthetic Graph Generation We evaluate DeFoG using the widely adopted *Planar*, *SBM* [\(Martinkus et al.,](#page-10-9) [2022\)](#page-10-9), and *Tree* datasets [\(Bergmeister et al.,](#page-8-8) [2023\)](#page-8-8), along with the associated evaluation methodology. In Tab.

## [experiment] 5.2. Efficiency Improvement
We now show that DeFoG enhances both training and sampling efficiency significantly across diverse datasets.

## [other] 5.3. Ablations
Here, we focus on evaluating the impact of different sampling approaches. We start from the vanilla sampling setup and sweep over sample distortion, target guidance, or stochasticity independently.

## [conclusion] 6. Conclusion
We introduce DeFoG, a novel discrete flow matching framework for graphs. This formulation enables training-sampling decoupling, which we ground theoretically to ensure faithful graph distribution modeling.

## [other] Acknowledgements
We would like to thank Clement Vignac and Andrew ´ Campbell for the useful discussions and suggestions.

## [other] Impact Statement
The primary objective of this paper is to advance graph generation under a more flexible framework, with applications spanning general graph generation, molecular design, and digital pathology. The ability to generate graphs with discrete labels can have broad-reaching implications for fields such as drug discovery and diagnostic technologies.

## [other] References
- <span id="page-8-0"></span>Asthana, R., Conrad, J., Dawoud, Y., Ortmanns, M., and Belagiannis, V. Multi-conditioned graph diffusion for neural architecture search.

## [other] Appendix Overview
In the Appendix, we provide additional details organized as follows:

## [background] <span id="page-13-0"></span>A. Contextualizing Related Research
In this section, we further contextualize DeFoG within the scope of related work. We begin by introducing the methods used for comparison with DeFoG in Appendix [A.1.](#page-13-1) Subsequently, we outline the key distinctions between DeFoG and existing diffusion-based graph generative models in Appendix [A.2.](#page-13-2)

## [method] <span id="page-13-1"></span>A.1. Overview of Compared Methods
In Sec. [5,](#page-5-0) we evaluate DeFoG against a diverse set of graph generative models, which we introduce below:

## [other] <span id="page-13-2"></span>A.2. DeFoG and Graph Diffusion Models
In this section, we contextualize DeFoG in relation to existing graph diffusion models.

## [other] A.2.1. FROM CONTINUOUS TO DISCRETE STATE-SPACES
Early diffusion-based graph generative models extended continuous diffusion and score-based methods from image generation to graphs by relaxing adjacency matrices into continuous state-spaces [\(Niu et al.,](#page-10-0) [2020;](#page-10-0) [Jo et al.,](#page-9-0) [2022\)](#page-9-0). However, this approach overlooks the inherent discreteness of graph-structured data, resulting in topologically uninformed noising processes.

## [other] A.2.2. From Discrete to Continuous time
The initial discrete-time diffusion frameworks for graph generation (Vignac et al., 2022; Haefeli et al., 2022) were built upon Discrete Denoising Diffusion Probabilistic Models (D3PMs) (Austin et al., 2021), which operate with a fixed partitioning of time. This discretization constrains the model to denoise at specific time points and ties the sampling process to the same fixed time steps used during training, leading to a rigid coupling between the training and sampling stages.

## [other] A.2.3. From Continuous-Time Discrete Diffusion to Discrete Flow Matching
While both continuous-time discrete diffusion and discrete flow matching (DFM) share the CTMC formulation for the denoising process, they differ fundamentally in the formulation of the noising process. Continuous-time discrete diffusion-based graph generative models (Xu et al., 2024; Siraudin et al., 2024) define the noising process as a CTMC, akin to the denoising process.

## [other] A.2.4. MIXED INTEGRATION OF CONTINUOUS AND DISCRETE STATE-SPACES
Integrating continuous and categorical data within graph generative models is an important challenge, as many real-world applications involve heterogeneous data types (e.g., molecular graphs containing atomic coordinates alongside categorical atom and bond types). A recent example addressing this challenge is GBD (Liu et al., 2024), which incorporates beta diffusion to jointly model both continuous and discrete variables.

## [other] <span id="page-15-1"></span>B. Sample Optimization
In this section, we explore the proposed sampling optimization in more detail. We start by analysing the different time distortion functions in Appendix B.1.

## [other] <span id="page-15-0"></span>B.1. Time Distortion Functions
In Sec. 3, we explore the utilization of different *distortion functions*, i.e., functions that are used to transform time.

## [other] <span id="page-16-1"></span>B.2. Target Guidance
In this section, we demonstrate that the proposed *target guidance* design for the rate matrices violates the Kolmogorov equation with an error that is linear in $\omega$ . This result indicates that a small guidance factor effectively helps fit the distribution, whereas a larger guidance factor, as shown in Figure 9, while enhancing topological properties such as planarity, increases the distance between generated and training data on synthetic datasets according to the metrics of average ratio.

## [other] <span id="page-18-0"></span>B.3. Detailed Balance, Prior Incorporation, and Stochasticity
Campbell et al. (2024) show that although their $z_1$ -conditional formulation of $R_t^*$ generates $p_{t|1}$ , it does not span the full space of valid rate matrices — those that satisfy the conditional Kolmogorov equation (Eq.

## [other] <span id="page-19-0"></span>B.4. Hyperparameter Optimization Pipeline
A significant advantage of flow matching methods is their inherently greater flexibility in the sampling process compared to diffusion models, as they are more disentangled from the training stage. Each of the proposed optimization strategies exposed in Sec.

## [other] <span id="page-21-0"></span>B.5. Understanding the Sampling Dynamics of $R_t^*$
In this section, we aim to provide deeper intuition into the sampling dynamics imposed by the design of $R_t^*$ , as proposed by (Campbell et al., 2024). The explicit formulation of $R_t^*$ can be found in Eq.

## [other] <span id="page-22-0"></span>B.6. Performance Improvement for Undertrained Models
In this section, we present the performance of a model trained on the QM9 dataset and the Planar dataset using only 30% of the epochs compared to the final model being reported. We employ the same hyperparameters as in Tab.

## [other] <span id="page-27-1"></span>C. Train Optimization
In this section, we provide a more detailed analysis of the influence of the various training optimization strategies introduced in Sec. 3.

## [other] <span id="page-27-0"></span>C.1. Initial Distributions
Under DeFoG's framework, the noising process for each dimension is modeled as a linear interpolation between the clean data distribution (the one-hot representation of the current state) and an initial distribution, $p_0$ . As such, it is intuitive that different initial distributions result in varying performances, depending on the denoising dynamics they induce.

## [other] <span id="page-28-0"></span>C.2. Interaction between Sample and Train Distortions
From Appendix B.4, we observe that time distortions applied during the sampling stage can significantly affect performance. This suggests that graph discrete flow models do not behave evenly across time and are more sensitive to specific time intervals, where generative performance benefits from finer updates achieved by using smaller time steps.

## [experiment] <span id="page-30-1"></span>D. Theoretical Results
In this section, we provide the proofs of the different theoretical results of the paper. First, we provide results that are domain agnostic, i.e., that hold for general discrete data, and then instantiate them for the specific case of graphs, yielding Theorems 1 and 2.

## [method] <span id="page-30-0"></span>D.1. Theoretical Results on the Generative Framework
In Appendix D.1.1, we provide a proof of the bounded estimation error of the rate matrix for general discrete data, and instantiate it for graphs, yielding Theorem 1. Then, in Appendix D.1.2, we prove the bounded deviation of the generated distribution for general discrete data as well, and instantiate it for graphs, yielding Theorem 2.

## [other] <span id="page-30-2"></span>D.1.1. BOUNDED ESTIMATION ERROR OF UNCONDITIONAL MULTIVARIATE RATE MATRIX
We start by introducing two important concepts, which will reveal useful for the proof of the intended result.

## [other] <span id="page-34-0"></span>D.1.2. BOUNDED DEVIATION OF THE GENERATED DISTRIBUTION
As in Appendix D.1.1, we start by introducing the necessary concepts that will reveal useful for the proof of the intended result.

## [other] D.1.3. CRITICAL ANALYSIS AND POSITIONING OF THEOREM 1 AND THEOREM 2
Theorem 1 establishes that minimizing the cross-entropy (CE) loss directly corresponds to minimizing an upper bound on the rate matrix estimation error. This result provides a direct and principled justification for using the CE loss, as it promotes accurate sampling from the underlying CTMC.

## [experiment] D.2. Theoretical Results on Architectural Expressivity
We now proceed to the graph specific theoretical results.

## [other] <span id="page-41-0"></span>D.2.1. NODE PERMUTATION EQUIVARIANCE AND INVARIANCE PROPERTIES
The different components of a graph generative model have to respect different graph symmetries. For example, the permutation equivariance of the model architecture ensures the output changes consistently with any reordering of input nodes, while permutation-invariant loss evaluates the model's performance consistently across isomorphic graphs, regardless of node order.

## [other] D.2.2. ADDITIONAL FEATURES EXPRESSIVITY
This section explains the expressivity of the RRWP features used in DeFoG. We summarize the findings of Ma et al.

## [other] <span id="page-45-0"></span>E. Conditional Generation
In this section, we describe how to seamlessly integrate DeFoG with existing methods for CTMC-based conditioning mechanisms. In this setting, all the examples are assumed to have a label.

## [experiment] <span id="page-46-1"></span>F. Experimental Details
This section provides further details on the experimental settings used in the paper.

## [other] <span id="page-46-0"></span>F.1. Architecture and Additional Features
**Denoising Neural Architecture** DeFoG's denoising neural network takes a noisy graph $G_t$ as input and predicts the clean marginal probability for each node $x^{(n)}$ via $p_{1|t}^{\theta,(n)}(\cdot|G_t)$ and for each edge $e^{(ij)}$ via $p_{1|t}^{\theta,(ij)}(\cdot|G_t)$ . This formulation boils down the graph generative task to a graph-to-graph mapping.

## [other] <span id="page-46-2"></span>F.2. Dataset Details


## [other] F.2.1. SYNTHETIC DATASETS
Here, we describe the datasets employed in our experiments and outline the specific metrics used to evaluate model performance on each dataset. Additional visualizations of example graphs from each dataset, along with generated graphs, are provided in Figures 16 to 18.

## [other] <span id="page-47-0"></span>F.2.2. MOLECULAR DATASETS
**Description** Molecular generation is a key real-world application of graph generation. It poses a challenging task to current graph generation models to their rich chemistry-specific information, involving several nodes and edges classes and leaning how to generate them jointly, and more complex evaluation pipelines.

## [other] F.2.3. DIGITAL PATHOLOGY DATASETS
**Description** Graphs, with their natural ability to represent relational data, are widely used to capture spatial biological dependencies in tissue images. This approach has proven successful in digital pathology tasks such as microenvironment classification (Wu et al., 2022), cancer classification (Pati et al., 2022), and decision explainability (Jaume et al., 2020).

## [other] F.3. Resources
The training and sampling times for the different datasets explored in this paper are provided in Tab. [4.](#page-48-1) All the experiments in this work were run on a single NVIDIA A100-SXM4-80GB GPU.

## [other] F.4. Hyperparameter Tuning
The default hyperparameters for training and sampling for each dataset can be found in the provided code repository. In Tab.

## [experiment] <span id="page-52-1"></span>G. Additional Results
First, we discuss conditional generation for real-world digital pathology graph generation. Next, we provide comprehensive tables detailing results on synthetic datasets in Appendix G.2.

## [other] <span id="page-52-0"></span>G.1. Conditional Generation
Table 6: TLS conditional generation results.

## [other] <span id="page-52-2"></span>G.2. Synthetic Graph Generation
In Tab. 7, we present the full results for DeFoG for the three different datasets: planar, tree, and SBM.

## [other] <span id="page-54-1"></span>G.3. Molecular Graph Generation
For the molecular generation tasks, we begin by examining the results for QM9, considering both implicit and explicit hydrogens (Vignac et al., 2022). In the implicit case, hydrogen atoms are inferred to complete the valencies, while in the explicit case, hydrogens must be explicitly modeled, making it an inherently more challenging task.

## [other] <span id="page-56-0"></span>G.4. Impact of Additional Features
In graph diffusion methods, the task of graph generation is decomposed into a mapping of a graph to a set of marginal probabilities for each node and edge. This problem is typically addressed using a Graph Transformer architecture, which is augmented with additional features to capture structural aspects that the base architecture might struggle to model effectively (Vignac et al., 2022; Xu et al., 2024; Siraudin et al., 2024) otherwise.
