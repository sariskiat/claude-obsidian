---
type: paper
slug: advancing-expert-specialization-for-better-moe
title: Advancing Expert Specialization for Better MoE
authors: Hongcan Guo, Haolang Lu, Guoshun Nan, Bolun Chu, Jialin Zhuang, Yuan Yang, Wenhao Che, Xinye Cao, Sicong Leng, Qimei Cui, Xudong Jiang
source_path: .paper-scholar/advancing-expert-specialization-for-better-moe
ingested_at: '2026-06-05 05:50:41'
authors_list: []
sections:
- id: 650
  heading: Abstract
  role: abstract
  order_index: 0
  summary: Mixture-of-Experts (MoE) models enable efficient scaling of large language models (LLMs) by activating only a subset of experts per input. However, we observe that the commonly used auxiliary load balancing loss often leads to expert overlap and overly uniform routing, which hinders expert specialization and degrades overall performance during post-training.
- id: 651
  heading: 1 Introduction
  role: introduction
  order_index: 1
  summary: Large language models (LLMs) [\[67,](#page-13-0) [65,](#page-13-1) [62,](#page-13-2) [6\]](#page-10-0) have demonstrated remarkable generalization capabilities [\[52,](#page-13-3) [69,](#page-14-0) [74,](#page-14-1) [73\]](#page-14-2) across a wide range of tasks [\[53,](#page-13-4) [24\]](#page-11-0), but their inference cost [\[15,](#page-10-1) [57\]](#page-13-5) grows rapidly with scale, hindering practical deployment and efficiency. Mixture-of-Experts (MoE) [\[9,](#page-10-2) [3,](#page-9-0) [37\]](#page-11-1) architectures alleviate this problem by activating only a subset of experts per input [\[19\]](#page-10-3), thus enabling greater model capacity without a commensurate increase in computational overhead [\[22,](#page-10-4) [49,](#page-12-0) [33\]](#page-11-2).
- id: 652
  heading: 2 Motivation
  role: other
  order_index: 2
  summary: ''
- id: 653
  heading: 2.1 Preliminaries of MoE
  role: other
  order_index: 3
  summary: 'In a typical MoE layer, let there be n experts, and a sequence of input tokens represented by $X = \{x_1, x_2, \cdots, x_N\}$ , where N is the total number of tokens in the sequence. The routing score matrix after applying the top-k mechanism is denoted as:'
- id: 654
  heading: 2.2 Observations
  role: other
  order_index: 4
  summary: 'Obs I (Expert Overlap): Introduction of the auxiliary loss function leads to a more homogenized distribution of tokens across experts, which may reduce the distinctiveness of each expert.'
- id: 655
  heading: 3 Method
  role: method
  order_index: 5
  summary: 'Based on the observations above, we propose the following design to mitigate *expert overlap* and *routing uniformity*, the overall loss function $\mathcal{L}$ is defined as follows:'
- id: 656
  heading: <span id="page-3-0"></span>3.1 Implementations of Losses $\mathcal{L}_o$ and $\mathcal{L}_v$
  role: method
  order_index: 6
  summary: In this section, we introduce two critical loss functions $\mathcal{L}_o$ and $\mathcal{L}_v$ that act on the expert and router components, respectively.
- id: 657
  heading: <span id="page-3-1"></span>3.2 Compatibility of Multi-Objective Optimization
  role: other
  order_index: 7
  summary: In this section, we analyze how each component influences the optimization dynamics of expert parameters $\theta_{E_j}$ and routing parameters $\theta_R$ during training. Meanwhile, we will focus on the optimization and compatibility of the two losses $L_o$ and $L_v$ with respect to load balancing and expert specificity.
- id: 658
  heading: 4 Experiments
  role: experiment
  order_index: 8
  summary: 'In this section, we conduct experiments to address the following research questions:'
- id: 659
  heading: 4.1 Experimental Setup
  role: experiment
  order_index: 9
  summary: '**Environment.** All experiments are performed on a CentOS Linux 7 server with PyTorch 2.3. The hardware specifications consist of 240GB of RAM, a 16-core Intel Xeon CPU, and two NVIDIA A800 GPUs, each having 80GB of memory.'
- id: 660
  heading: 4.2 Performance in Downstream Tasks (RQ1)
  role: other
  order_index: 10
  summary: 'To verify that our $\mathcal{L}_{balance}$ enhances model performance in downstream task scenarios through expert orthogonality and routing output diversification, as shown in Table 1, we design downstream task scenarios on 11 well-known benchmarks and validate our method against four baseline methods with distinct loss designs on three widely used MoE models. We make the following observations:'
- id: 661
  heading: <span id="page-6-0"></span>4.3 Load Balancing (RQ2)
  role: other
  order_index: 11
  summary: To verify that our newly added losses $\mathcal{L}_v$ and $\mathcal{L}_o$ do not affect the load balancing effect, we conduct statistical measurements on the load balancing of all combinations of $\mathcal{L}_{aux}$ , $\mathcal{L}_v$ , and $\mathcal{L}_o$ across various models during training.
- id: 662
  heading: 4.4 Behaviors of Experts and Routing (RQ3)
  role: other
  order_index: 12
  summary: To verify that $\mathcal{L}_v$ and $\mathcal{L}_o$ can jointly promote expert orthogonality and routing score diversification, following the method setup in Section 4.3, we will conduct evaluations of expert orthogonality and measurements of routing score diversification for different loss combinations.
- id: 663
  heading: 4.5 Ablation among Losses (RQ4)
  role: method
  order_index: 13
  summary: To demonstrate that both $\mathcal{L}_o$ and $\mathcal{L}_v$ have positive effects on the model's performance in downstream task scenarios, and their combination synergistically enhances each other's efficacy, we design ablation experiments for these two losses on three models.
- id: 664
  heading: 5 Related Work
  role: background
  order_index: 14
  summary: '**Auxiliary Losses in MoE Training.** Auxiliary losses [39, 85] are commonly used to prevent expert collapse by encouraging balanced expert utilization [14]. Early approaches focus on suppressing routing imbalance, while later works [81] introduce capacity constraints or multi-level objectives to separate routing stability from load balancing [65, 39, 20].'
- id: 665
  heading: 6 Limitation & Future Discussion
  role: other
  order_index: 15
  summary: While Lbalance balances load and enhances performance in downstream tasks, its potential in other domains remains unexplored. Specifically, it could be extended to visual models, as suggested in recent work [\[26\]](#page-11-13), and multimodal or full-modal settings [\[8\]](#page-10-12), offering opportunities for crossdomain applications.
- id: 666
  heading: 7 Conclusion
  role: conclusion
  order_index: 16
  summary: In this work, we present a theoretically grounded framework that resolves the inherent conflict between expert specialization and routing uniformity in MoE training. By introducing orthogonality and variance-based objectives, our method significantly improves downstream performance without any architectural changes.
- id: 667
  heading: 8 Acknowledgements
  role: other
  order_index: 17
  summary: This work was supported in part by the National Key Research and Development Program of China under Grant 2022YFB2902200; in part by the National Natural Science Foundation of China under Grant 62471064; in part by the Fundamental Research Funds for the Beijing University of Posts and Telecommunications under Grant 2025AI4S02.
- id: 668
  heading: References
  role: other
  order_index: 18
  summary: '- <span id="page-9-3"></span>[1] Eneko Agirre, Llu''is M''arquez, and Richard Wicentowski, editors. *Proceedings of the Fourth International Workshop on Semantic Evaluations (SemEval-2007)*.'
- id: 669
  heading: A Notations
  role: other
  order_index: 19
  summary: 'Table 2: Notations and Definitions'
- id: 670
  heading: B Motivation
  role: other
  order_index: 20
  summary: ''
- id: 671
  heading: B.1 MoE Layer Structure
  role: other
  order_index: 21
  summary: A Mixture of Experts (MoE) layer enhances the capacity of a neural network model by conditionally activating different specialized sub-networks, known as "experts," for different input tokens. This architecture allows the model to scale its parameter count significantly while maintaining a relatively constant computational cost per token during inference.
- id: 672
  heading: B.2 Observation
  role: other
  order_index: 22
  summary: '**Obs I(Expert Overlap)**: Introduction of the auxiliary loss function leads to a more homogenized distribution of tokens across experts, which may reduce the distinctiveness of each expert.'
- id: 673
  heading: C Method
  role: method
  order_index: 23
  summary: ''
- id: 674
  heading: C.1 Specialized Losses $\mathcal{L}_o$ and $\mathcal{L}_v$
  role: method
  order_index: 24
  summary: 'In this section, we introduce two critical loss functions: the orthogonality loss $\mathcal{L}_o$ , which acts on the expert representations, and the variance loss $\mathcal{L}_v$ , which acts on the routing scores. These losses are designed to encourage expert specialization and routing diversity, respectively.'
- id: 675
  heading: C.2 Compatibility of Multi-Objective Optimization
  role: other
  order_index: 25
  summary: In this section, we conduct a detailed analysis of how each loss component, namely $\mathcal{L}_h$ , $\mathcal{L}_{aux}$ , $\mathcal{L}_o$ , $\mathcal{L}_v$ , influences the optimization dynamics of expert parameters $\theta_{E_j}$ (for $j=1,\ldots,n$ experts) and routing parameters $\theta_R$ during the training process. Our primary focus is to demonstrate the theoretical compatibility and synergistic interplay between the specialized losses $\mathcal{L}_o$ (promoting expert orthogonality) and $\mathcal{L}_v$ (promoting routing score diversification) in conjunction with the load balancing loss $\mathcal{L}_{aux}$ and the primary task loss $\mathcal{L}_h$ .
- id: 676
  heading: Mutually Compatible
  role: other
  order_index: 26
  summary: We elaborate on the compatibility of $\mathcal{L}_o$ and $\mathcal{L}_v$ by examining their respective gradient contributions to expert parameters and routing parameters. The total loss function is $\mathcal{L} = \mathcal{L}_h + \alpha \mathcal{L}_{aux} + \beta \mathcal{L}_o + \gamma \mathcal{L}_v$ .
- id: 677
  heading: Mutually Reinforcing
  role: other
  order_index: 27
  summary: Beyond mere compatibility, $\mathcal{L}_o$ and $\mathcal{L}_v$ can create a synergistic effect, where improvements in one facilitate the optimization of the other.
- id: 678
  heading: C.3 Proof of Lemmas
  role: other
  order_index: 28
  summary: '**Lemma 1** Let $S \in \mathbb{R}^{N \times n}$ be a matrix that satisfies following conditions: each row sums to 1, each row contains k non-zero elements and n-k zero elements. Then, there always exists a state in which the following two objectives are simultaneously optimized: 1.'
- id: 679
  heading: proof C.1 1. Preliminaries and Assumptions
  role: other
  order_index: 29
  summary: The lemma implicitly requires $k \ge 2$ . If k = 1, each row i has a single non-zero element $s_{i,j_i} = 1$ .
- id: 680
  heading: 2. Construction of an Initial State $S^{(0)}$ Optimizing Objective 1
  role: other
  order_index: 30
  summary: To optimize Objective 1, we select a support matrix $\mathcal{P}$ such that its column sums (degrees of column nodes in the associated bipartite graph), $d_j = \sum_{i=1}^N p_{ij}$ , are as uniform as possible. That is, each $d_j \in \{\lfloor Nk/n \rfloor, \lceil Nk/n \rceil\}$ .
- id: 681
  heading: 3. Perturbation via a Cycle in the Support Graph $G_{\mathcal{P}}$
  role: other
  order_index: 31
  summary: Let $G_{\mathcal{P}} = (U \cup V, E_{\mathcal{P}})$ be the bipartite graph associated with $\mathcal{P}$ , where $U = \{r_1, \dots, r_N\}$ represents rows, $V = \{c_1, \dots, c_n\}$ represents columns, and an edge $(r_i, c_j) \in E_{\mathcal{P}}$ if and only if $p_{ij} = 1$ .
- id: 682
  heading: 4. Existence of the Desired State and Conclusion
  role: conclusion
  order_index: 32
  summary: The construction of S' from $S^{(0)}$ demonstrates that if $k \geq 2$ and the support graph $G_{\mathcal{P}}$ (chosen to optimize Objective 1) contains a cycle, then a state S' exists satisfying the lemma's conditions. Objective 1 remains optimized, and Objective 2 is achieved because the variance of non-zero elements in rows participating in the cycle is strictly increased from zero.
- id: 683
  heading: C.4 Computational Overhead of Lo
  role: other
  order_index: 33
  summary: While $L_o$ has quadratic complexity in theory, the actual overhead is negligible in practice due to the small number of activated experts (k) and efficient batched implementations. It does not present a bottleneck in our setup.
- id: 684
  heading: <span id="page-27-0"></span>D Datasets
  role: other
  order_index: 34
  summary: '**GSM8K** [12] is a benchmark designed to evaluate mathematical reasoning through 8,000 elementary and middle school word problems across arithmetic, algebra, geometry, and other topics. Each'
- id: 685
  heading: <span id="page-29-0"></span>E Metrics
  role: other
  order_index: 35
  summary: MaxVioglobal [\[68\]](#page-14-10) is a metric introduced to quantify load imbalance in Mixture-of-Experts (MoE) models.A lower value indicates more balanced expert utilization, while a higher value reflects severe imbalance. It evaluates global load balance across the entire validation set, reflecting long-term efficiency and fairness in expert usage.
- id: 686
  heading: <span id="page-30-0"></span>F Implementation Details
  role: other
  order_index: 36
  summary: '**DeepSeek-Moe-16B**[14] DeepSeekMoE-16B is a Mixture-of-Experts (MoE) language model with 16.4B parameters. It employs an innovative MoE architecture, which involves two principal strategies: fine-grained expert segmentation and shared experts isolation.'
- id: 687
  heading: <span id="page-31-0"></span>G Baselines
  role: other
  order_index: 37
  summary: GShard[\[39\]](#page-12-7) GShard is a pioneering Mixture-of-Experts (MoE) architecture developed by Google Research, designed for massively parallelized training across thousands of devices. It introduces automatic tensor sharding to scale model parameters and data efficiently, achieving dynamic load balancing during distributed computation.
- id: 688
  heading: H Experiments Details
  role: experiment
  order_index: 38
  summary: ''
- id: 689
  heading: <span id="page-31-1"></span>H.1 Hyperparameter Sensitivity
  role: other
  order_index: 39
  summary: To address the importance of hyperparameter sensitivity, we conducted experiments varying the values of the loss weights α (for Laux), β (for Lo), and γ (for Lv) across different magnitudes.
- id: 690
  heading: <span id="page-32-0"></span>H.2 Configurations and Base Model Performance
  role: other
  order_index: 40
  summary: A discrepancy between our reported results and the original model figures from public citations (e.g., Moonlight, DeepSeek) was observed. This disparity primarily arises from differences in model versions, prompting strategies, and inference settings.
- id: 691
  heading: H.3 Performance Under Larger and More Diverse Training Data
  role: other
  order_index: 41
  summary: We conducted an experiment to evaluate the impact of training data size and diversity on the effectiveness of our method.
- id: 692
  heading: H.3.1 Motivation from Single-Task Settings
  role: other
  order_index: 42
  summary: As noted in the introduction, our method is motivated by the observation that in post-training scenarios, the training data is often domain-specific and less diverse. This results in highly skewed token distributions, which intensifies the conflict between load balancing (which encourages even token-toexpert allocation) and expert specialization (which encourages domain-specific token routing).
- id: 693
  heading: H.3.2 Performance on Mixed and Richer Datasets
  role: other
  order_index: 43
  summary: To test whether our method still performs well with more diverse training data, we constructed a mixed dataset combining Numina (math), GPQA (science), and HumanEval (coding), totaling 18k examples. We fine-tuned the Moonlight (Kimi) model for 3 epochs on this combined dataset.
- id: 694
  heading: <span id="page-33-0"></span>I More Baselines and MoE Architectures
  role: other
  order_index: 44
  summary: ''
- id: 695
  heading: I.1 Comparison with Additional Baselines
  role: other
  order_index: 45
  summary: To provide a more comprehensive evaluation, we expanded our set of comparison methods to include two additional state-of-the-art baselines. We re-evaluated all methods on the most comprehensive subsets of our benchmark suite.
- id: 696
  heading: I.2 Performance on Diverse MoE Architectures
  role: other
  order_index: 46
  summary: To further validate the generality of our method, we extended our evaluation to more diverse MoE architectures. Our initial experiments focused on DeepSeek and Moonlight models due to their strong open-source performance and recent community adoption.
- id: 697
  heading: <span id="page-35-0"></span>J Training Overhead
  role: other
  order_index: 47
  summary: While our method introduces some additional computation due to the proposed regularization losses, the training time remains within a practical range and compares favorably with existing baselines. We report the average step time (in seconds per iteration) on the DeepSeek V2 Lite model using a batch size of 32.
- id: 698
  heading: K Visualization
  role: other
  order_index: 48
  summary: Figures [5](#page-35-2) present the PCA projection of token embeddings assigned to the top 3 most active experts from baseline models. The significant overlap among different colors suggests that the token representations routed to different experts are not well separated.
---

# Advancing Expert Specialization for Better MoE

## [abstract] Abstract
Mixture-of-Experts (MoE) models enable efficient scaling of large language models (LLMs) by activating only a subset of experts per input. However, we observe that the commonly used auxiliary load balancing loss often leads to expert overlap and overly uniform routing, which hinders expert specialization and degrades overall performance during post-training.

## [introduction] 1 Introduction
Large language models (LLMs) [\[67,](#page-13-0) [65,](#page-13-1) [62,](#page-13-2) [6\]](#page-10-0) have demonstrated remarkable generalization capabilities [\[52,](#page-13-3) [69,](#page-14-0) [74,](#page-14-1) [73\]](#page-14-2) across a wide range of tasks [\[53,](#page-13-4) [24\]](#page-11-0), but their inference cost [\[15,](#page-10-1) [57\]](#page-13-5) grows rapidly with scale, hindering practical deployment and efficiency. Mixture-of-Experts (MoE) [\[9,](#page-10-2) [3,](#page-9-0) [37\]](#page-11-1) architectures alleviate this problem by activating only a subset of experts per input [\[19\]](#page-10-3), thus enabling greater model capacity without a commensurate increase in computational overhead [\[22,](#page-10-4) [49,](#page-12-0) [33\]](#page-11-2).

## [other] 2 Motivation


## [other] 2.1 Preliminaries of MoE
In a typical MoE layer, let there be n experts, and a sequence of input tokens represented by $X = \{x_1, x_2, \cdots, x_N\}$ , where N is the total number of tokens in the sequence. The routing score matrix after applying the top-k mechanism is denoted as:

## [other] 2.2 Observations
Obs I (Expert Overlap): Introduction of the auxiliary loss function leads to a more homogenized distribution of tokens across experts, which may reduce the distinctiveness of each expert.

## [method] 3 Method
Based on the observations above, we propose the following design to mitigate *expert overlap* and *routing uniformity*, the overall loss function $\mathcal{L}$ is defined as follows:

## [method] <span id="page-3-0"></span>3.1 Implementations of Losses $\mathcal{L}_o$ and $\mathcal{L}_v$
In this section, we introduce two critical loss functions $\mathcal{L}_o$ and $\mathcal{L}_v$ that act on the expert and router components, respectively.

## [other] <span id="page-3-1"></span>3.2 Compatibility of Multi-Objective Optimization
In this section, we analyze how each component influences the optimization dynamics of expert parameters $\theta_{E_j}$ and routing parameters $\theta_R$ during training. Meanwhile, we will focus on the optimization and compatibility of the two losses $L_o$ and $L_v$ with respect to load balancing and expert specificity.

## [experiment] 4 Experiments
In this section, we conduct experiments to address the following research questions:

## [experiment] 4.1 Experimental Setup
**Environment.** All experiments are performed on a CentOS Linux 7 server with PyTorch 2.3. The hardware specifications consist of 240GB of RAM, a 16-core Intel Xeon CPU, and two NVIDIA A800 GPUs, each having 80GB of memory.

## [other] 4.2 Performance in Downstream Tasks (RQ1)
To verify that our $\mathcal{L}_{balance}$ enhances model performance in downstream task scenarios through expert orthogonality and routing output diversification, as shown in Table 1, we design downstream task scenarios on 11 well-known benchmarks and validate our method against four baseline methods with distinct loss designs on three widely used MoE models. We make the following observations:

## [other] <span id="page-6-0"></span>4.3 Load Balancing (RQ2)
To verify that our newly added losses $\mathcal{L}_v$ and $\mathcal{L}_o$ do not affect the load balancing effect, we conduct statistical measurements on the load balancing of all combinations of $\mathcal{L}_{aux}$ , $\mathcal{L}_v$ , and $\mathcal{L}_o$ across various models during training.

## [other] 4.4 Behaviors of Experts and Routing (RQ3)
To verify that $\mathcal{L}_v$ and $\mathcal{L}_o$ can jointly promote expert orthogonality and routing score diversification, following the method setup in Section 4.3, we will conduct evaluations of expert orthogonality and measurements of routing score diversification for different loss combinations.

## [method] 4.5 Ablation among Losses (RQ4)
To demonstrate that both $\mathcal{L}_o$ and $\mathcal{L}_v$ have positive effects on the model's performance in downstream task scenarios, and their combination synergistically enhances each other's efficacy, we design ablation experiments for these two losses on three models.

## [background] 5 Related Work
**Auxiliary Losses in MoE Training.** Auxiliary losses [39, 85] are commonly used to prevent expert collapse by encouraging balanced expert utilization [14]. Early approaches focus on suppressing routing imbalance, while later works [81] introduce capacity constraints or multi-level objectives to separate routing stability from load balancing [65, 39, 20].

## [other] 6 Limitation & Future Discussion
While Lbalance balances load and enhances performance in downstream tasks, its potential in other domains remains unexplored. Specifically, it could be extended to visual models, as suggested in recent work [\[26\]](#page-11-13), and multimodal or full-modal settings [\[8\]](#page-10-12), offering opportunities for crossdomain applications.

## [conclusion] 7 Conclusion
In this work, we present a theoretically grounded framework that resolves the inherent conflict between expert specialization and routing uniformity in MoE training. By introducing orthogonality and variance-based objectives, our method significantly improves downstream performance without any architectural changes.

## [other] 8 Acknowledgements
This work was supported in part by the National Key Research and Development Program of China under Grant 2022YFB2902200; in part by the National Natural Science Foundation of China under Grant 62471064; in part by the Fundamental Research Funds for the Beijing University of Posts and Telecommunications under Grant 2025AI4S02.

## [other] References
- <span id="page-9-3"></span>[1] Eneko Agirre, Llu'is M'arquez, and Richard Wicentowski, editors. *Proceedings of the Fourth International Workshop on Semantic Evaluations (SemEval-2007)*.

## [other] A Notations
Table 2: Notations and Definitions

## [other] B Motivation


## [other] B.1 MoE Layer Structure
A Mixture of Experts (MoE) layer enhances the capacity of a neural network model by conditionally activating different specialized sub-networks, known as "experts," for different input tokens. This architecture allows the model to scale its parameter count significantly while maintaining a relatively constant computational cost per token during inference.

## [other] B.2 Observation
**Obs I(Expert Overlap)**: Introduction of the auxiliary loss function leads to a more homogenized distribution of tokens across experts, which may reduce the distinctiveness of each expert.

## [method] C Method


## [method] C.1 Specialized Losses $\mathcal{L}_o$ and $\mathcal{L}_v$
In this section, we introduce two critical loss functions: the orthogonality loss $\mathcal{L}_o$ , which acts on the expert representations, and the variance loss $\mathcal{L}_v$ , which acts on the routing scores. These losses are designed to encourage expert specialization and routing diversity, respectively.

## [other] C.2 Compatibility of Multi-Objective Optimization
In this section, we conduct a detailed analysis of how each loss component, namely $\mathcal{L}_h$ , $\mathcal{L}_{aux}$ , $\mathcal{L}_o$ , $\mathcal{L}_v$ , influences the optimization dynamics of expert parameters $\theta_{E_j}$ (for $j=1,\ldots,n$ experts) and routing parameters $\theta_R$ during the training process. Our primary focus is to demonstrate the theoretical compatibility and synergistic interplay between the specialized losses $\mathcal{L}_o$ (promoting expert orthogonality) and $\mathcal{L}_v$ (promoting routing score diversification) in conjunction with the load balancing loss $\mathcal{L}_{aux}$ and the primary task loss $\mathcal{L}_h$ .

## [other] Mutually Compatible
We elaborate on the compatibility of $\mathcal{L}_o$ and $\mathcal{L}_v$ by examining their respective gradient contributions to expert parameters and routing parameters. The total loss function is $\mathcal{L} = \mathcal{L}_h + \alpha \mathcal{L}_{aux} + \beta \mathcal{L}_o + \gamma \mathcal{L}_v$ .

## [other] Mutually Reinforcing
Beyond mere compatibility, $\mathcal{L}_o$ and $\mathcal{L}_v$ can create a synergistic effect, where improvements in one facilitate the optimization of the other.

## [other] C.3 Proof of Lemmas
**Lemma 1** Let $S \in \mathbb{R}^{N \times n}$ be a matrix that satisfies following conditions: each row sums to 1, each row contains k non-zero elements and n-k zero elements. Then, there always exists a state in which the following two objectives are simultaneously optimized: 1.

## [other] proof C.1 1. Preliminaries and Assumptions
The lemma implicitly requires $k \ge 2$ . If k = 1, each row i has a single non-zero element $s_{i,j_i} = 1$ .

## [other] 2. Construction of an Initial State $S^{(0)}$ Optimizing Objective 1
To optimize Objective 1, we select a support matrix $\mathcal{P}$ such that its column sums (degrees of column nodes in the associated bipartite graph), $d_j = \sum_{i=1}^N p_{ij}$ , are as uniform as possible. That is, each $d_j \in \{\lfloor Nk/n \rfloor, \lceil Nk/n \rceil\}$ .

## [other] 3. Perturbation via a Cycle in the Support Graph $G_{\mathcal{P}}$
Let $G_{\mathcal{P}} = (U \cup V, E_{\mathcal{P}})$ be the bipartite graph associated with $\mathcal{P}$ , where $U = \{r_1, \dots, r_N\}$ represents rows, $V = \{c_1, \dots, c_n\}$ represents columns, and an edge $(r_i, c_j) \in E_{\mathcal{P}}$ if and only if $p_{ij} = 1$ .

## [conclusion] 4. Existence of the Desired State and Conclusion
The construction of S' from $S^{(0)}$ demonstrates that if $k \geq 2$ and the support graph $G_{\mathcal{P}}$ (chosen to optimize Objective 1) contains a cycle, then a state S' exists satisfying the lemma's conditions. Objective 1 remains optimized, and Objective 2 is achieved because the variance of non-zero elements in rows participating in the cycle is strictly increased from zero.

## [other] C.4 Computational Overhead of Lo
While $L_o$ has quadratic complexity in theory, the actual overhead is negligible in practice due to the small number of activated experts (k) and efficient batched implementations. It does not present a bottleneck in our setup.

## [other] <span id="page-27-0"></span>D Datasets
**GSM8K** [12] is a benchmark designed to evaluate mathematical reasoning through 8,000 elementary and middle school word problems across arithmetic, algebra, geometry, and other topics. Each

## [other] <span id="page-29-0"></span>E Metrics
MaxVioglobal [\[68\]](#page-14-10) is a metric introduced to quantify load imbalance in Mixture-of-Experts (MoE) models.A lower value indicates more balanced expert utilization, while a higher value reflects severe imbalance. It evaluates global load balance across the entire validation set, reflecting long-term efficiency and fairness in expert usage.

## [other] <span id="page-30-0"></span>F Implementation Details
**DeepSeek-Moe-16B**[14] DeepSeekMoE-16B is a Mixture-of-Experts (MoE) language model with 16.4B parameters. It employs an innovative MoE architecture, which involves two principal strategies: fine-grained expert segmentation and shared experts isolation.

## [other] <span id="page-31-0"></span>G Baselines
GShard[\[39\]](#page-12-7) GShard is a pioneering Mixture-of-Experts (MoE) architecture developed by Google Research, designed for massively parallelized training across thousands of devices. It introduces automatic tensor sharding to scale model parameters and data efficiently, achieving dynamic load balancing during distributed computation.

## [experiment] H Experiments Details


## [other] <span id="page-31-1"></span>H.1 Hyperparameter Sensitivity
To address the importance of hyperparameter sensitivity, we conducted experiments varying the values of the loss weights α (for Laux), β (for Lo), and γ (for Lv) across different magnitudes.

## [other] <span id="page-32-0"></span>H.2 Configurations and Base Model Performance
A discrepancy between our reported results and the original model figures from public citations (e.g., Moonlight, DeepSeek) was observed. This disparity primarily arises from differences in model versions, prompting strategies, and inference settings.

## [other] H.3 Performance Under Larger and More Diverse Training Data
We conducted an experiment to evaluate the impact of training data size and diversity on the effectiveness of our method.

## [other] H.3.1 Motivation from Single-Task Settings
As noted in the introduction, our method is motivated by the observation that in post-training scenarios, the training data is often domain-specific and less diverse. This results in highly skewed token distributions, which intensifies the conflict between load balancing (which encourages even token-toexpert allocation) and expert specialization (which encourages domain-specific token routing).

## [other] H.3.2 Performance on Mixed and Richer Datasets
To test whether our method still performs well with more diverse training data, we constructed a mixed dataset combining Numina (math), GPQA (science), and HumanEval (coding), totaling 18k examples. We fine-tuned the Moonlight (Kimi) model for 3 epochs on this combined dataset.

## [other] <span id="page-33-0"></span>I More Baselines and MoE Architectures


## [other] I.1 Comparison with Additional Baselines
To provide a more comprehensive evaluation, we expanded our set of comparison methods to include two additional state-of-the-art baselines. We re-evaluated all methods on the most comprehensive subsets of our benchmark suite.

## [other] I.2 Performance on Diverse MoE Architectures
To further validate the generality of our method, we extended our evaluation to more diverse MoE architectures. Our initial experiments focused on DeepSeek and Moonlight models due to their strong open-source performance and recent community adoption.

## [other] <span id="page-35-0"></span>J Training Overhead
While our method introduces some additional computation due to the proposed regularization losses, the training time remains within a practical range and compares favorably with existing baselines. We report the average step time (in seconds per iteration) on the DeepSeek V2 Lite model using a batch size of 32.

## [other] K Visualization
Figures [5](#page-35-2) present the PCA projection of token embeddings assigned to the top 3 most active experts from baseline models. The significant overlap among different colors suggests that the token representations routed to different experts are not well separated.
