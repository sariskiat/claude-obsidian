---
type: paper
slug: span-id-page-0-0-span-temporal-difference-flows
title: Temporal Difference Flows
authors: '(from paper: Temporal Difference Flows)'
source_path: .paper-scholar/span-id-page-0-0-span-temporal-difference-flows
ingested_at: '2026-06-05 05:51:17'
authors_list: []
sections:
- id: 738
  heading: 1 Introduction
  role: introduction
  order_index: 0
  summary: Predictive modeling lies at the heart of intelligent decision-making, enabling agents to reason and plan in complex environments. In Reinforcement Learning (RL), this predictive capability has traditionally been achieved through world models that capture the transition structure of the environment.
- id: 739
  heading: 2 Background
  role: background
  order_index: 1
  summary: In the following, we use capital letters to denote random variables, sans-serif fonts for sets (e.g., A), and $\mathscr{P}(A)$ to denote the space of probability measures over a measurable set A.
- id: 740
  heading: <span id="page-2-3"></span>3 Temporal Difference Flows
  role: other
  order_index: 2
  summary: 'Flow Matching (FM; Lipman et al., 2023, 2024; Liu et al., 2023; Albergo and Vanden-Eijnden, 2023) constructs a time-dependent probability path $m_t: S \times A \to \mathscr{P}(S)$ for $t \in [0,1]$ that evolves smoothly from the source distribution $m_0 = p_0 \in \mathscr{P}(S)$ to the target distribution $m_1 \approx m^{\pi}$ . This evolution is governed by a vector field $v_t: S \times S \times A \to S$ , which dictates the instantaneous movement of samples along $m_t$ .'
- id: 741
  heading: <span id="page-4-5"></span>3.1 Extension to Diffusion Models
  role: other
  order_index: 3
  summary: Denoising Diffusion models (Sohl-Dickstein et al., 2015; Ho et al., 2020) build a diffusion process starting from a data sample $X_0 \sim q_0 = m^{\pi} (\cdot \mid S, A)^2$ and corrupting it via a stochastic differential equation (SDE),
- id: 742
  heading: 4 Theoretical Analysis
  role: method
  order_index: 4
  summary: We now study the learning dynamics of an idealized version of the TD-Flow methods, assuming that the flow-matching loss is minimized exactly at each iteration. Under this assumption, at each iteration we compute a probability path $m_t^{(n)}$ such that $m_1^{(n)} = \mathcal{T}^{\pi} m_1^{(n-1)}$ , which immediately implies that $m_1^{(n)} \to m^{\pi}$ by the contraction property of $\mathcal{T}^{\pi}$ .
- id: 743
  heading: 5 Experiments
  role: experiment
  order_index: 5
  summary: We now present a series of experiments to assess the efficacy of our TD-based flow and diffusion approaches with baselines employing Generative Adversarial Networks (Goodfellow et al., 2014) and $\beta$ -Variational Auto-Encoders (Higgins et al., 2017). Following the methodology from Touati et al.
- id: 744
  heading: <span id="page-7-1"></span>5.1 Empirical Evaluation of Geometric Horizon Models
  role: other
  order_index: 6
  summary: Before benchmarking, we must first obtain a policy to evaluate. We follow the approach taken in Thakoor et al.
- id: 745
  heading: Planning via Generalized Policy Improvement
  role: other
  order_index: 7
  summary: Figure 3 Performance improvement over the zero-shot Forward Backward (FB; Touati and Ollivier, 2021) policies when planning with Generalized Policy Improvement (GPI; Barreto et al., 2017). FB-GPI performs GPI over the FB value-function $Q^{\pi_w}$ .
- id: 746
  heading: 5.2 Planning via Generalized Policy Improvement
  role: other
  order_index: 8
  summary: We now turn our attention towards training policy-conditioned GHMs which can be utilized for test-time planning. To accomplish this, we first pre-train a Forward Backward (FB; Touati and Ollivier, 2021; Touati et al., 2023) representation using the same dataset of 10M transitions as described in §5.1.
- id: 747
  heading: 6 Discussion
  role: conclusion
  order_index: 9
  summary: In this paper, we introduced temporal difference flows, a novel generative modeling approach that significantly advances long-horizon predictive models of state. By leveraging the successor measure's temporal difference structure both in its sampling procedure and learning objective, TD<sup>2</sup>-CFM and TD<sup>2</sup>-DD effectively address challenges associated with modeling long-range state dynamics.
- id: 748
  heading: Acknowledgements
  role: other
  order_index: 10
  summary: We thank Yaron Lipman for informative discussions on flow matching and for providing valuable feedback on our work. Additionally, we thank Yann Ollivier for his keen insights and thoughtful suggestions throughout the development of this work.
- id: 749
  heading: References
  role: other
  order_index: 11
  summary: '- <span id="page-11-16"></span>Laurent Dinh, David Krueger, and Yoshua Bengio. NICE: Non-linear independent components estimation.'
- id: 750
  heading: Appendices
  role: other
  order_index: 12
  summary: ''
- id: 751
  heading: <span id="page-15-0"></span>A Related Work
  role: other
  order_index: 13
  summary: The Successor Representation [\(Dayan,](#page-10-1) [1993\)](#page-10-1) was originally proposed for tabular MDPs and was later generalized to continuous state spaces with the Successor Measure [\(Blier et al.,](#page-10-0) [2021\)](#page-10-0). Successor Features [\(Barreto et al.,](#page-10-5) [2017,](#page-10-5) [2020\)](#page-10-7) extends these ideas by instead modeling the evolution of multi-dimensional features assuming rewards decompose linearly over these features.
- id: 752
  heading: <span id="page-16-2"></span>B Extension to Score Matching and Diffusion Models
  role: other
  order_index: 14
  summary: This section extends our framework to score matching and denoising diffusion models. We leverage the unification of these methods under stochastic differential equations (Song et al., 2021b) introducing an analogous class of Temporal Difference Diffusion methods.
- id: 753
  heading: <span id="page-16-3"></span>B.1 Background
  role: background
  order_index: 15
  summary: Both score-based generative modeling (Song and Ermon, 2019) and diffusion probabilistic modeling (Sohl-Dickstein et al., 2015; Ho et al., 2020) can be unified under the framework of stochastic differential equations (SDE) introduced in Song et al. (2021b).
- id: 754
  heading: B.2 Temporal Difference Diffusion
  role: other
  order_index: 16
  summary: 'To learn a predictive model of $m^{\pi}$ using diffusion from an offline dataset, we follow a similar approach to what we presented in §3 and we define an iterative process starting from initial weights $\theta^{(0)}$ and at each iteration minimizing the Temporal-Difference Denoising Diffusion (TD-DD) loss:'
- id: 755
  heading: <span id="page-19-0"></span>C Experimental Details
  role: experiment
  order_index: 17
  summary: ''
- id: 756
  heading: <span id="page-19-1"></span>C.1 Evaluation
  role: other
  order_index: 18
  summary: Evaluating a GHM can be challenging, TD-based losses employing bootstrapping do not provide a good signal as to the quality of the learned model. Instead, we opt to measure 1) the likelihood of a trajectory coming from the true discounted occupancy of a given policy, 2) the Earth Mover's Distance (EMD; Rubner et al., 2000) between samples from the true occupancy and our GHM which provides an estimate of the distance between these two probability distributions, and 3) the value-function approximation error.
- id: 757
  heading: C.2 Environments
  role: other
  order_index: 19
  summary: Experiments in this paper were conducted with a subset of domains from the DeepMind Control Suite (Tunyasuvunakool et al., 2020) highlighted in Figure 4.
- id: 758
  heading: <span id="page-21-1"></span>C.3 Geometric Horizon Models
  role: other
  order_index: 20
  summary: ''
- id: 759
  heading: C.3.1 Flow Matching
  role: other
  order_index: 21
  summary: ''
- id: 760
  heading: <span id="page-21-2"></span>Algorithm 1 Template for TD-Flow algorithms
  role: other
  order_index: 22
  summary: '1: **Inputs**: offline dataset $\mathcal{D}$ , policy $\pi$ , batch size n, Polyak coefficient $\zeta$ , weight decay $\lambda$ , randomly initialized weights $\theta$ , discount factor $\gamma$ , learning rate $\eta$ , one-step conditional path $\vec{p}_{t|1}$ and conditional vector-field $\vec{u}_{t|1}$ , bootstrap path $\hat{p}_t$ and vector-field $\hat{v}_t$ .'
- id: 761
  heading: Compute loss
  role: other
  order_index: 23
  summary: '11: \ell(\theta) = \frac{1}{K} \sum_{k=1}^{K} (1 - \gamma) \vec{\ell}_k(\theta) + \gamma \hat{\ell}_k(\theta)'
- id: 762
  heading: Perform gradient step
  role: other
  order_index: 24
  summary: '12: 13: \theta \leftarrow \theta - \eta \nabla_{\theta} \left( \ell(\theta) + \lambda ||\theta||^2 \right) 14:'
- id: 763
  heading: Update parameters of target vector field
  role: other
  order_index: 25
  summary: '15: \bar{\theta} \leftarrow \zeta \bar{\theta} + (1 - \zeta)\theta 16: 17: end for ```'
- id: 764
  heading: C.3.2 Denoising Diffusion
  role: other
  order_index: 26
  summary: We train a Denoising Diffusion Probabilistic Model (DDPM; Ho et al., 2020) using the same architecture as our flow matching model above, with the output now being interpreted as a prediction of the noise seed $\epsilon_0$ that began the diffusion process. We discretize the diffusion process using 1,000 steps with $\beta_{\min} = 0.1$ and $\beta_{\max} = 20$ .
- id: 765
  heading: C.3.3 Generative Adversarial Network
  role: other
  order_index: 27
  summary: We implement a modern Generative Adversarial Network (GAN; Goodfellow et al., 2014) baseline based on the recommendations in Huang et al. (2024).
- id: 766
  heading: C.3.4 Variational Auto-Encoder
  role: other
  order_index: 28
  summary: We implement a β-Variational Auto-Encoder [\(Kingma and Welling,](#page-12-21) [2014;](#page-12-21) [Higgins et al.,](#page-11-6) [2017\)](#page-11-6) following the best practices outlined in [Thakoor et al.](#page-14-1) [\(2022\)](#page-14-1). That is, we train our VAE to minimize the following loss,
- id: 767
  heading: <span id="page-24-0"></span>C.4 Hyperparameters
  role: other
  order_index: 29
  summary: We report the hyper-parameters for training the GHM models used in the single and multi-policy experiments. [Table 5](#page-24-1) shows the parameters for Flow Matching and Denoising Diffusion.
- id: 768
  heading: <span id="page-26-0"></span>D Additional Experimental Results
  role: experiment
  order_index: 30
  summary: In this section, we report additional results about the experiments.
- id: 769
  heading: <span id="page-33-0"></span>E Theoretical Results
  role: other
  order_index: 31
  summary: ''
- id: 770
  heading: <span id="page-33-1"></span>E.1 Proofs of Main Results
  role: other
  order_index: 32
  summary: '**Lemma 1.** Let $\vec{p}_t$ be a probability path for P generated by vector field $\vec{v}_t$ and $\widehat{p}_t^{(n)}$ be a probability path for $P^{\pi}m_1^{(n)}$ generated by $\widehat{v}_t^{(n)}$ such that $\vec{p}_0 = \widehat{p}_0^{(n)} = m_0$ . For any $t \in [0, 1]$ and (s, a) let <sup>4</sup>'
- id: 771
  heading: <span id="page-34-0"></span>E.2 General Results
  role: other
  order_index: 33
  summary: $$v_t := \frac{(1-\gamma)p_t^1 v_t^1 + \gamma p_t^2 v_t^2}{(1-\gamma)p_t^1 + \gamma p_t^2}.$$ (26)
- id: 772
  heading: <span id="page-35-0"></span>E.3 Analysis of TD<sup>2</sup>-CFM
  role: method
  order_index: 34
  summary: We study the learning dynamics of an idealized variant of TD<sup>2</sup>-CFM which minimizes the flow-matching loss exactly. Starting from an arbitrary vector field $v_t^{(0)}$ , at each iteration $n \ge 0$ we compute
- id: 773
  heading: E.4 Analysis of TD-CFM
  role: method
  order_index: 35
  summary: We study the learning dynamics of an idealized variant of TD-CFM which minimizes the flow-matching loss exactly. Starting from an arbitrary vector field $v_t^{(0)}$ , at each iteration $n \ge 0$ we compute
- id: 774
  heading: <span id="page-38-0"></span>E.5 Analysis of TD-CFM(C)
  role: method
  order_index: 36
  summary: The idealized update of td-cfm(c) is, for any n ≥ 0,
- id: 775
  heading: <span id="page-40-0"></span>E.6 Variance Analysis
  role: method
  order_index: 37
  summary: $$\begin{split} g_{_{TD^{2}-CFM}}(t,s,a,s',\vec{X_{t}},X_{t}^{(n)}) &:= (1-\gamma)\nabla_{\theta}v_{t}(\vec{X_{t}}|s,a;\theta)^{\top} \left(v_{t}(\vec{X_{t}}|s,a;\theta) - u_{t|1}(\vec{X_{t}}|s')\right) \\ &+ \gamma\nabla_{\theta}v_{t}(X_{t}^{(n)}|s,a;\theta)^{\top} \left(v_{t}(X_{t}^{(n)}|s,a;\theta) - v_{t}^{(n)}(X_{t}^{(n)}|s',\pi(s'))\right) \\ g_{_{TD^{-}CFM}}(t,s,a,s',\vec{X_{t}},X_{1},X_{t}) &:= (1-\gamma)\nabla_{\theta}v_{t}(\vec{X_{t}}|s,a;\theta)^{\top} \left(v_{t}(\vec{X_{t}}|s,a;\theta) - u_{t|1}(\vec{X_{t}}|s')\right) \\ &+ \gamma\nabla_{\theta}v_{t}(X_{t}|s,a;\theta)^{\top} \left(v_{t}(X_{t}|s,a;\theta) - u_{t|1}(X_{t}|X_{1})\right) \end{split}$$
- id: 776
  heading: <span id="page-43-0"></span>E.7 Transport Cost Analysis
  role: method
  order_index: 38
  summary: '**Theorem 9.** Assume that $m_t^{(n)}(x \mid s, a) = \int p_{t|1}(x \mid x_1)m_1^{(n)}(x_1 \mid s, a)\mathrm{d}x_1$ , where $p_{t|1}(\cdot \mid x_1) = \mathcal{N}(tx_1, (1-t)^2I)$ is a Gaussian path. Then, the conditional paths built by TD-CFM(C) and TD<sup>2</sup>-CFM to generate $m_1^{(n+1)} = \mathcal{T}^{\pi}m_1^{(n)}$ induce a smaller transport cost than those built by TD-CFM.'
---

# Temporal Difference Flows

## [introduction] 1 Introduction
Predictive modeling lies at the heart of intelligent decision-making, enabling agents to reason and plan in complex environments. In Reinforcement Learning (RL), this predictive capability has traditionally been achieved through world models that capture the transition structure of the environment.

## [background] 2 Background
In the following, we use capital letters to denote random variables, sans-serif fonts for sets (e.g., A), and $\mathscr{P}(A)$ to denote the space of probability measures over a measurable set A.

## [other] <span id="page-2-3"></span>3 Temporal Difference Flows
Flow Matching (FM; Lipman et al., 2023, 2024; Liu et al., 2023; Albergo and Vanden-Eijnden, 2023) constructs a time-dependent probability path $m_t: S \times A \to \mathscr{P}(S)$ for $t \in [0,1]$ that evolves smoothly from the source distribution $m_0 = p_0 \in \mathscr{P}(S)$ to the target distribution $m_1 \approx m^{\pi}$ . This evolution is governed by a vector field $v_t: S \times S \times A \to S$ , which dictates the instantaneous movement of samples along $m_t$ .

## [other] <span id="page-4-5"></span>3.1 Extension to Diffusion Models
Denoising Diffusion models (Sohl-Dickstein et al., 2015; Ho et al., 2020) build a diffusion process starting from a data sample $X_0 \sim q_0 = m^{\pi} (\cdot \mid S, A)^2$ and corrupting it via a stochastic differential equation (SDE),

## [method] 4 Theoretical Analysis
We now study the learning dynamics of an idealized version of the TD-Flow methods, assuming that the flow-matching loss is minimized exactly at each iteration. Under this assumption, at each iteration we compute a probability path $m_t^{(n)}$ such that $m_1^{(n)} = \mathcal{T}^{\pi} m_1^{(n-1)}$ , which immediately implies that $m_1^{(n)} \to m^{\pi}$ by the contraction property of $\mathcal{T}^{\pi}$ .

## [experiment] 5 Experiments
We now present a series of experiments to assess the efficacy of our TD-based flow and diffusion approaches with baselines employing Generative Adversarial Networks (Goodfellow et al., 2014) and $\beta$ -Variational Auto-Encoders (Higgins et al., 2017). Following the methodology from Touati et al.

## [other] <span id="page-7-1"></span>5.1 Empirical Evaluation of Geometric Horizon Models
Before benchmarking, we must first obtain a policy to evaluate. We follow the approach taken in Thakoor et al.

## [other] Planning via Generalized Policy Improvement
Figure 3 Performance improvement over the zero-shot Forward Backward (FB; Touati and Ollivier, 2021) policies when planning with Generalized Policy Improvement (GPI; Barreto et al., 2017). FB-GPI performs GPI over the FB value-function $Q^{\pi_w}$ .

## [other] 5.2 Planning via Generalized Policy Improvement
We now turn our attention towards training policy-conditioned GHMs which can be utilized for test-time planning. To accomplish this, we first pre-train a Forward Backward (FB; Touati and Ollivier, 2021; Touati et al., 2023) representation using the same dataset of 10M transitions as described in §5.1.

## [conclusion] 6 Discussion
In this paper, we introduced temporal difference flows, a novel generative modeling approach that significantly advances long-horizon predictive models of state. By leveraging the successor measure's temporal difference structure both in its sampling procedure and learning objective, TD<sup>2</sup>-CFM and TD<sup>2</sup>-DD effectively address challenges associated with modeling long-range state dynamics.

## [other] Acknowledgements
We thank Yaron Lipman for informative discussions on flow matching and for providing valuable feedback on our work. Additionally, we thank Yann Ollivier for his keen insights and thoughtful suggestions throughout the development of this work.

## [other] References
- <span id="page-11-16"></span>Laurent Dinh, David Krueger, and Yoshua Bengio. NICE: Non-linear independent components estimation.

## [other] Appendices


## [other] <span id="page-15-0"></span>A Related Work
The Successor Representation [\(Dayan,](#page-10-1) [1993\)](#page-10-1) was originally proposed for tabular MDPs and was later generalized to continuous state spaces with the Successor Measure [\(Blier et al.,](#page-10-0) [2021\)](#page-10-0). Successor Features [\(Barreto et al.,](#page-10-5) [2017,](#page-10-5) [2020\)](#page-10-7) extends these ideas by instead modeling the evolution of multi-dimensional features assuming rewards decompose linearly over these features.

## [other] <span id="page-16-2"></span>B Extension to Score Matching and Diffusion Models
This section extends our framework to score matching and denoising diffusion models. We leverage the unification of these methods under stochastic differential equations (Song et al., 2021b) introducing an analogous class of Temporal Difference Diffusion methods.

## [background] <span id="page-16-3"></span>B.1 Background
Both score-based generative modeling (Song and Ermon, 2019) and diffusion probabilistic modeling (Sohl-Dickstein et al., 2015; Ho et al., 2020) can be unified under the framework of stochastic differential equations (SDE) introduced in Song et al. (2021b).

## [other] B.2 Temporal Difference Diffusion
To learn a predictive model of $m^{\pi}$ using diffusion from an offline dataset, we follow a similar approach to what we presented in §3 and we define an iterative process starting from initial weights $\theta^{(0)}$ and at each iteration minimizing the Temporal-Difference Denoising Diffusion (TD-DD) loss:

## [experiment] <span id="page-19-0"></span>C Experimental Details


## [other] <span id="page-19-1"></span>C.1 Evaluation
Evaluating a GHM can be challenging, TD-based losses employing bootstrapping do not provide a good signal as to the quality of the learned model. Instead, we opt to measure 1) the likelihood of a trajectory coming from the true discounted occupancy of a given policy, 2) the Earth Mover's Distance (EMD; Rubner et al., 2000) between samples from the true occupancy and our GHM which provides an estimate of the distance between these two probability distributions, and 3) the value-function approximation error.

## [other] C.2 Environments
Experiments in this paper were conducted with a subset of domains from the DeepMind Control Suite (Tunyasuvunakool et al., 2020) highlighted in Figure 4.

## [other] <span id="page-21-1"></span>C.3 Geometric Horizon Models


## [other] C.3.1 Flow Matching


## [other] <span id="page-21-2"></span>Algorithm 1 Template for TD-Flow algorithms
1: **Inputs**: offline dataset $\mathcal{D}$ , policy $\pi$ , batch size n, Polyak coefficient $\zeta$ , weight decay $\lambda$ , randomly initialized weights $\theta$ , discount factor $\gamma$ , learning rate $\eta$ , one-step conditional path $\vec{p}_{t|1}$ and conditional vector-field $\vec{u}_{t|1}$ , bootstrap path $\hat{p}_t$ and vector-field $\hat{v}_t$ .

## [other] Compute loss
11: \ell(\theta) = \frac{1}{K} \sum_{k=1}^{K} (1 - \gamma) \vec{\ell}_k(\theta) + \gamma \hat{\ell}_k(\theta)

## [other] Perform gradient step
12: 13: \theta \leftarrow \theta - \eta \nabla_{\theta} \left( \ell(\theta) + \lambda ||\theta||^2 \right) 14:

## [other] Update parameters of target vector field
15: \bar{\theta} \leftarrow \zeta \bar{\theta} + (1 - \zeta)\theta 16: 17: end for ```

## [other] C.3.2 Denoising Diffusion
We train a Denoising Diffusion Probabilistic Model (DDPM; Ho et al., 2020) using the same architecture as our flow matching model above, with the output now being interpreted as a prediction of the noise seed $\epsilon_0$ that began the diffusion process. We discretize the diffusion process using 1,000 steps with $\beta_{\min} = 0.1$ and $\beta_{\max} = 20$ .

## [other] C.3.3 Generative Adversarial Network
We implement a modern Generative Adversarial Network (GAN; Goodfellow et al., 2014) baseline based on the recommendations in Huang et al. (2024).

## [other] C.3.4 Variational Auto-Encoder
We implement a β-Variational Auto-Encoder [\(Kingma and Welling,](#page-12-21) [2014;](#page-12-21) [Higgins et al.,](#page-11-6) [2017\)](#page-11-6) following the best practices outlined in [Thakoor et al.](#page-14-1) [\(2022\)](#page-14-1). That is, we train our VAE to minimize the following loss,

## [other] <span id="page-24-0"></span>C.4 Hyperparameters
We report the hyper-parameters for training the GHM models used in the single and multi-policy experiments. [Table 5](#page-24-1) shows the parameters for Flow Matching and Denoising Diffusion.

## [experiment] <span id="page-26-0"></span>D Additional Experimental Results
In this section, we report additional results about the experiments.

## [other] <span id="page-33-0"></span>E Theoretical Results


## [other] <span id="page-33-1"></span>E.1 Proofs of Main Results
**Lemma 1.** Let $\vec{p}_t$ be a probability path for P generated by vector field $\vec{v}_t$ and $\widehat{p}_t^{(n)}$ be a probability path for $P^{\pi}m_1^{(n)}$ generated by $\widehat{v}_t^{(n)}$ such that $\vec{p}_0 = \widehat{p}_0^{(n)} = m_0$ . For any $t \in [0, 1]$ and (s, a) let <sup>4</sup>

## [other] <span id="page-34-0"></span>E.2 General Results
$$v_t := \frac{(1-\gamma)p_t^1 v_t^1 + \gamma p_t^2 v_t^2}{(1-\gamma)p_t^1 + \gamma p_t^2}.$$ (26)

## [method] <span id="page-35-0"></span>E.3 Analysis of TD<sup>2</sup>-CFM
We study the learning dynamics of an idealized variant of TD<sup>2</sup>-CFM which minimizes the flow-matching loss exactly. Starting from an arbitrary vector field $v_t^{(0)}$ , at each iteration $n \ge 0$ we compute

## [method] E.4 Analysis of TD-CFM
We study the learning dynamics of an idealized variant of TD-CFM which minimizes the flow-matching loss exactly. Starting from an arbitrary vector field $v_t^{(0)}$ , at each iteration $n \ge 0$ we compute

## [method] <span id="page-38-0"></span>E.5 Analysis of TD-CFM(C)
The idealized update of td-cfm(c) is, for any n ≥ 0,

## [method] <span id="page-40-0"></span>E.6 Variance Analysis
$$\begin{split} g_{_{TD^{2}-CFM}}(t,s,a,s',\vec{X_{t}},X_{t}^{(n)}) &:= (1-\gamma)\nabla_{\theta}v_{t}(\vec{X_{t}}|s,a;\theta)^{\top} \left(v_{t}(\vec{X_{t}}|s,a;\theta) - u_{t|1}(\vec{X_{t}}|s')\right) \\ &+ \gamma\nabla_{\theta}v_{t}(X_{t}^{(n)}|s,a;\theta)^{\top} \left(v_{t}(X_{t}^{(n)}|s,a;\theta) - v_{t}^{(n)}(X_{t}^{(n)}|s',\pi(s'))\right) \\ g_{_{TD^{-}CFM}}(t,s,a,s',\vec{X_{t}},X_{1},X_{t}) &:= (1-\gamma)\nabla_{\theta}v_{t}(\vec{X_{t}}|s,a;\theta)^{\top} \left(v_{t}(\vec{X_{t}}|s,a;\theta) - u_{t|1}(\vec{X_{t}}|s')\right) \\ &+ \gamma\nabla_{\theta}v_{t}(X_{t}|s,a;\theta)^{\top} \left(v_{t}(X_{t}|s,a;\theta) - u_{t|1}(X_{t}|X_{1})\right) \end{split}$$

## [method] <span id="page-43-0"></span>E.7 Transport Cost Analysis
**Theorem 9.** Assume that $m_t^{(n)}(x \mid s, a) = \int p_{t|1}(x \mid x_1)m_1^{(n)}(x_1 \mid s, a)\mathrm{d}x_1$ , where $p_{t|1}(\cdot \mid x_1) = \mathcal{N}(tx_1, (1-t)^2I)$ is a Gaussian path. Then, the conditional paths built by TD-CFM(C) and TD<sup>2</sup>-CFM to generate $m_1^{(n+1)} = \mathcal{T}^{\pi}m_1^{(n)}$ induce a smaller transport cost than those built by TD-CFM.
