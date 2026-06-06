---
type: paper-fulltext
slug: adjoint-schr-dinger-bridge-sampler
arxiv_id: null
source_path: /Users/saris.kia.adm/.paper-scholar/adjoint-schr-dinger-bridge-sampler/2506.22565.md
paper: "[[adjoint-schr-dinger-bridge-sampler]]"
---
# Adjoint Schrödinger Bridge Sampler

 $\textbf{Guan-Horng Liu}^{1,*}, \ \textbf{Jaemoo Choi}^{2,*}, \ \textbf{Yongxin Chen}^{2}, \ \textbf{Benjamin Kurt Miller}^{1}, \ \textbf{Ricky T. Q. Chen}^{1,*}$ 

<sup>1</sup>FAIR at Meta, <sup>2</sup>Georgia Institute of Technology

Computational methods for learning to sample from the Boltzmann distribution—where the target distribution is known only up to an unnormalized energy function—have advanced significantly recently. Due to the lack of explicit target samples, however, prior diffusion-based methods, known as diffusion samplers, often require importance-weighted estimation or complicated learning processes. Both trade off scalability with extensive evaluations of the energy and model, thereby limiting their practical usage. In this work, we propose Adjoint Schrödinger Bridge Sampler (ASBS), a new diffusion sampler that employs simple and scalable matching-based objectives yet without the need to estimate target samples during training. ASBS is grounded on a mathematical model—the Schrödinger Bridge—which enhances sampling efficiency via kinetic-optimal transportation. Through a new lens of stochastic optimal control theory, we demonstrate how SB-based diffusion samplers can be learned at scale via Adjoint Matching and prove convergence to the global solution. Notably, ASBS generalizes the recent Adjoint Sampling (Havens et al., 2025) to arbitrary source distributions by relaxing the so-called memoryless condition that largely restricts the design space. Through extensive experiments, we demonstrate the effectiveness of ASBS on sampling from classical energy functions, amortized conformer generation, and molecular Boltzmann distributions.

Date: November 26, 2025

Correspondence: ghliu@meta.com

Code: https://github.com/facebookresearch/adjoint\_samplers

<span id="page-0-1"></span><span id="page-0-0"></span>Meta

### 1 Introduction

Sampling from Boltzmann distributions is a fundamental problem in computational science, with widespread applications in Bayesian inference, statistical physics, and chemistry (Box and Tiao, 2011; Binder et al., 1992; Tuckerman, 2023). Mathematically, we aim to sample from a target distribution  $\nu(x)$  known up to a unnormalized, often differentiable, energy function  $E(x): \mathcal{X} \subseteq \mathbb{R}^d \to \mathbb{R}$ ,

$$\nu(x) := \frac{e^{-E(x)}}{Z}, \quad \text{where } Z := \int_{\mathcal{X}} e^{-E(x)} dx$$
 (1)

is an intractable normalization constant. For instance, the energy function E(x) of a molecular system quantifies the stability of a chemical structure based on the 3D positions of particles. A lower energy indicates a more stable structure and hence a higher likelihood of its occurrence, i.e.,  $\nu(x) \propto e^{-E(x)}$ .

Classical methods that generate samples from  $\nu(x)$  rely on Markov Chain Monte Carlo algorithms, which run a Markov chain whose stationary distribution is  $\nu(x)$  (Metropolis et al., 1953; Neal, 2001; Del Moral et al., 2006). These methods, however, tend to suffer from slow mixing time and require extensive evaluations of energy function, limiting their practical usages due to prohibitive complexity.

To improve sampling efficiency, modern samplers focus on learning better proposal distributions (Noé et al., 2019; Midgley et al., 2023). Among those, recent advances in diffusion-based generative models (Song et al., 2021; Ho et al., 2020) have given rise to a family of *Diffusion Samplers*, which consider stochastic differential equations (SDEs) of the following form:

$$dX_t = \left[ f_t(X_t) + \sigma_t u_t^{\theta}(X_t) \right] dt + \sigma_t dW_t, \qquad X_0 \sim \mu(X_0), \tag{2}$$

<sup>\*</sup>Core contributors

<span id="page-1-1"></span>**Table 1** Compared to prior diffusion samplers, **Adjoint Schrödinger Bridge Sampler** (**ASBS**) offers the most flexible design for diffusion samplers (2), while learning the drift  $u_t^{\theta}$  via scalable matching objectives that do not rely on computation of importance weights (IWs).

|                                                                 | Design cond    | ition for $(2)$ | Learning method for $u_t^{\theta}$ |                    |  |
|-----------------------------------------------------------------|----------------|-----------------|------------------------------------|--------------------|--|
| Method                                                          | Non-memoryless | Arbitrary prior | Matching objective <sup>1</sup>    | No reliance on IWs |  |
| PIS (Zhang and Chen, 2022) DDS (Vargas et al., 2023)            | ×              | ×               | Х                                  | ✓                  |  |
| LV-PIS & LV-DDS (Richter and Berner, 2024)                      | X              | ×               | X                                  | X                  |  |
| PDDS (Phillips et al., 2024) iDEM (Akhound-Sadegh et al., 2024) | X              | ×               | ✓                                  | X                  |  |
| AS (Havens et al., 2025)                                        | X              | ×               | ✓                                  | ✓                  |  |
| Sequential SB (Bernton et al., 2019)                            | ✓              | ✓               | ×                                  | ×                  |  |
| Adjoint Schrödinger Bridge Sampler (Ours)                       | ✓              | ✓               | ✓                                  | ✓                  |  |

where  $f_t(x):[0,1]\times\mathcal{X}\to\mathcal{X}$  the base drift,  $\sigma_t:[0,1]\to\mathbb{R}_{>0}$  the noise schedule, and  $\mu(x)$  the initial source distribution. Given  $(f_t,\sigma_t,\mu)$ , the diffusion sampler learns a parametrized drift  $u_t^{\theta}(x)$  transporting samples to the target distribution  $\nu(x)$  at the terminal time t=1.

Computational methods for learning diffusion samplers have grown significantly recently (Zhang and Chen, 2022; Vargas et al., 2023; Berner et al., 2024; Chen et al., 2025). Due to the distinct problem setup in (1), the target distribution is defined exclusively by its energy E(x), rather than by explicit target samples. This characteristic renders modern generative modeling techniques for scalability—particularly the score matching objectives<sup>1</sup>—less applicable. As such, prior matching-based diffusion samplers (Phillips et al., 2024; Akhound-Sadegh et al., 2024; De Bortoli et al., 2024) often require computationally intensive estimation of target samples via importance weights (IWs).

Recently, Havens et al. (2025) introduced Adjoint Sampling (AS), a new class of diffusion samplers whose matching objectives rely only on on-policy samples, thereby greatly enhancing scalability. By incorporating stochastic optimal control (SOC) theory (Kappen, 2005; Todorov, 2007), AS facilitates the use of Adjoint Matching (Domingo-Enrich et al., 2025), a novel matching objective that imposes self-consistency in generated samples, effectively eliminating the needs for target samples.

The efficiency of AS, however, is achieved through a specific instantiation of the SDE (2) to satisfy the so-called memoryless condition. This condition—formally discussed in Section 2—restricts its source distribution to be Dirac delta  $\mu(x) := \delta$ , precluding the use of common priors such as Gaussian or domain-specific priors such as the harmonic oscillators in molecular systems (Jing et al., 2023). Notably, the memoryless condition underlies all previous matching-based diffusion samplers, restricting the design space of (2) from other choices known to enhance transportation efficiency (Shaul et al., 2023). While the condition has been relaxed in non-matching-based methods at extensive computational complexity (Richter and Berner, 2024; Bernton et al., 2019), no existing diffusion sampler—to our best understanding—has successfully combined matching objectives with non-memoryless condition. Table 1 summarizes the comparison between prior diffusion samplers.

<span id="page-1-3"></span>In this work, we propose **Adjoint Schrödinger Bridge Sampler (ASBS)**, a new adjoint-matching-based diffusion sampler that eliminates the requirement for memoryless condition entirely. Formally, ASBS recasts learning diffusion sampler as a distributionally constrained optimization, known as the Schrödinger Bridge (SB) problem (Schrödinger, 1931, 1932; Léonard, 2013; Chen et al., 2016):

<span id="page-1-2"></span>
$$\min_{u} D_{\mathrm{KL}}(p^{u}||p^{\mathrm{base}}) = \mathbb{E}_{X \sim p^{u}} \left[ \int_{0}^{1} \frac{1}{2} ||u_{t}^{\theta}(X_{t})||^{2} dt \right], \tag{3a}$$

s.t. 
$$dX_t = [f_t(X_t) + \sigma_t u_t^{\theta}(X_t)] dt + \sigma_t dW_t, \qquad X_0 \sim \mu(X_0), \qquad X_1 \sim \nu(X_1).$$
 (3b)

Here,  $p^u$  denotes the path distribution induced by the SDE in (3b), whereas  $p^{\text{base}} := p^{u:=0}$  denotes the path distribution induced by the "base" SDE when  $u_t := 0$ . By minimizing their KL divergence, the SB problem (3) seeks the kinetic-optimal drift  $u_t^*$ —an optimality structure well correlated with sampling efficiency in

<span id="page-1-0"></span><sup>&</sup>lt;sup>1</sup>The matching objective is a simple regression loss,  $\mathbb{E}[|u_t^{\theta}(X_t) - v_t(X_t, X_1)|]^2$ , w.r.t. some tractable  $v_t$ .

generative modeling (Finlay et al., 2020; Liu et al., 2023). Since the SOC problem in AS corresponds to a specific case of the SB problem with  $(f_t, \mu) := (0, \delta)$ , ASBS extends AS to handle non-memoryless conditions by solving more general SB problems (see Theorem 3.1). Computationally, ASBS retains all scalability advantages from AS by utilizing an adjoint-matching objective that removes the need for estimating target samples. It also introduces a corrector-matching objective to correct nontrivial biases arising from non-memoryless conditions. We prove that alternating optimization between the two matching objectives is equivalent to executing the Iterative Proportional Fitting algorithm (Kullback, 1968), ensuring global convergence of ASBS to  $u_t^*$  (see Theorem 3.2). Though extensive experiments, we show superior performance of ASBS over prior diffusion samplers across various benchmarks on sampling multi-particle energy functions.

In summary, we present the following contributions:

- We introduce ASBS, an SB-based diffusion sampler capable of sampling target distributions using only unnormalized energy functions, by solving general SB problems with arbitrary priors.
- We base ASBS on a new SOC framework that removes the restrictive memoryless condition, develop a scalable matching-based algorithm, and prove theoretical convergence to global solution.
- <span id="page-2-0"></span>• We show ASBS's superior performance over prior methods on sampling Boltzmann distributions of classical energy functions, alanine dipeptide molecule and amortized conformer generation.

### 2 Preliminary

We revisit the memoryless condition introduced by Domingo-Enrich et al. (2025) and examine its impact on the constructions of SOC-based diffusion samplers (Zhang and Chen, 2022; Havens et al., 2025), which are closely related to our ASBS. Additional review can be found in Section A.

**Stochastic Optimal Control (SOC)** The SOC problem (4) studies an optimization problem:

<span id="page-2-1"></span>
$$\min_{u} \mathbb{E}_{X \sim p^{u}} \left[ \int_{0}^{1} \frac{1}{2} \|u_{t}(X_{t})\|^{2} dt + g(X_{1}) \right] \quad \text{s.t. (2)},$$

which, unlike the SB problem (3), includes an additional terminal cost  $g(x): \mathcal{X} \to \mathbb{R}$  at the terminal time t = 1 and considers the SDE without the terminal constraint  $X_1 \sim \nu$ . The primary reason for studying this specific optimization problem is that the optimal distribution is known analytically by<sup>2</sup>

<span id="page-2-4"></span>
$$p^{\star}(X_0, X_1) = p^{\text{base}}(X_0, X_1)e^{-g(X_1) + V_0(X_0)}, \quad \text{where} \quad V_0(x) = -\log \int p_{1|0}^{\text{base}}(y|x)e^{-g(y)} dy$$
 (5)

is the initial value function. That is, the optimal distribution  $p^*$  is an exponentially tilted version of the base distribution,  $p^{\text{base}} := p^{u:=0}$ . Specifically,  $p^{\text{base}}$  is tilted by the terminal cost " $-g(X_1)$ " and the initial value function  $V_0(X_0)$ , which is intractable. Consequently, to ensure its marginal  $p^*(X_1)$  follows the target distribution  $\nu(X_1)$ , we must eliminate the *initial value function bias* from  $V_0(X_0)$ .

Memoryless condition & SOC-based diffusion sampler A common approach to eliminate the aforementioned initial value function bias, adopted by most diffusion samplers, is to restrict the class of base processes to be *memoryless*. Formally, the memoryless condition assumes statistical independency between  $X_0$  and  $X_1$  in the base distribution:

<span id="page-2-3"></span>
$$p^{\text{base}}(X_0, X_1) \stackrel{\text{memoryless}}{:=} p^{\text{base}}(X_0) p^{\text{base}}(X_1). \tag{6}$$

This memoryless condition (6) simplifies the optimal distribution at the terminal time t=1 and, upon choosing a proper terminal cost g(x), recovers the target distribution  $\nu$ ,

$$p^{\star}(X_1) \stackrel{\text{memoryless}}{=} \int p^{\text{base}}(X_0) p^{\text{base}}(X_1) e^{-g(X_1) + V_0(X_0)} \mathrm{d}X_0 \propto p^{\text{base}}(X_1) e^{-g(X_1)} = \nu(X_1),$$

<span id="page-2-2"></span><sup>&</sup>lt;sup>2</sup>Equation (5) can be obtained by rewriting (4) as  $D_{\text{KL}}(p^u||p^{\text{base}}) + \mathbb{E}_{p_1^u}[g(X_1)]$  and then computing the analytic solution  $p^*(X_1|X_0) \propto p^{\text{base}}(X_1|X_0)e^{-g(X_1)}$  and normalization  $\int p^{\text{base}}(X_1|X_0)e^{-g(X_1)}dX_1 = e^{-V_0(X_0)}$ . See Section A.1 for details.

<span id="page-3-0"></span>![](_page_3_Figure_0.jpeg)

Figure 1 Effect of the memoryless condition on learning SOC-based diffusion samplers. We consider Gaussian prior  $\mu(x) := \mathcal{N}(x; 0, 1)$  with  $(f_t, \sigma_t)$  set to VP-SDE for the first plot and (0, 0.2) for the rest; see Section A.1 for details. The memoryless condition injects significant noise (left) to correct the otherwise biased optimization (middle), whereas ASBS can successfully debias any non-memoryless processes (right).

where the last equality is due to setting the terminal cost to  $g(x) := \log \frac{p_t^{\operatorname{base}(x)}}{\nu(x)}$ . Typically, the memoryless condition (6) is enforced by a careful design of the base distribution  $p^{\operatorname{base}}$  or, equivalently, the parameters  $(f_t, \sigma_t, \mu)$  in (2). For instance, the variance-preserving process (VP; Song et al., 2021) considers a linear base drift  $f_t$ , a noise schedule  $\sigma_t$  that grows significantly with time, and a Gaussian prior  $\mu$ ; see Figure 1. Alternatively, one could implement (6) with Dirac delta prior  $\mu(x) := \delta_0(x)$  and  $f_t := 0$ , leading to the following SOC problem (Zhang and Chen, 2022):

<span id="page-3-1"></span>
$$\min_{u} \mathbb{E}_{X \sim p^{u}} \left[ \int_{0}^{1} \frac{1}{2} \|u_{t}(X_{t})\|^{2} dt + \log \frac{p_{1}^{\text{base}}(X_{1})}{\nu(X_{1})} \right] \quad \text{s.t. } dX_{t} = \sigma_{t} u_{t}(X_{t}) dt + \sigma_{t} dW_{t}, \quad X_{0} = 0.$$
(7)

Based on the aforementioned reasoning, solving (7) results in a diffusion sampler that transports samples to the target distribution at t=1, with Adjoint Sampling (Havens et al., 2025) as the only scalable method of this class. Despite encouraging, the SOC problem in (7) is nevertheless limited by its trivial source, precluding potentially more effective options for sampling Boltzmann distributions.

# 3 Adjoint Schrödinger Bridge Sampler

We introduce a new diffusion sampler by solving the SB problem (3), where the target distribution  $\nu(x)$  is given by its energy function E(x) rather than explicit samples. All proofs are left in Section B.

#### 3.1 SOC Characteristics of the SB Problem

The SB problem (3)—as an optimization problem with distribution constraints—is widely explored in optimal transport, stochastic control, and recently machine learning (Léonard, 2012; Chen et al., 2021; De Bortoli et al., 2021). Its kinetic-optimal drift  $u^*$  satisfies the following optimality equations:

<span id="page-3-4"></span><span id="page-3-3"></span><span id="page-3-2"></span>
$$u_t^{\star}(x) = \sigma_t \nabla \log \varphi_t(x), \quad \text{where} \begin{cases} \varphi_t(x) = \int p_{1|t}^{\text{base}}(y|x)\varphi_1(y) dy, & \varphi_0(x)\hat{\varphi}_0(x) = \mu(x) \\ \hat{\varphi}_t(x) = \int p_{t|0}^{\text{base}}(x|y)\hat{\varphi}_0(y) dy, & \varphi_1(x)\hat{\varphi}_1(x) = \nu(x) \end{cases}$$
(8a)

and  $p_{t|s}^{\text{base}}(y|x) := p^{\text{base}}(X_t = y|X_s = x)$  is the transition kernel of the base process for observing y at time t given x at time s. The SB potentials  $\varphi_t(x), \hat{\varphi}_t(x) \in C^{1,2}([0,1], \mathbb{R}^d)$  are then defined (up to some multiplicative constant) as solutions to forward and backward time integrations w.r.t.  $p_{t|s}^{\text{base}}$ .

Equation (8) are computationally challenging to solve—even when  $p_{t|s}^{\text{base}}$  has an analytical solution—due to the intractable integration and coupled boundaries at t=0 and 1. Our key observation is that the first equation (8a) resembles the optimality condition of the SOC problem (4) (see Section A.1). This implies that the optimality conditions of SB hints an SOC reinterpretation, which, as we will demonstrate, is more tractable than solving (8) directly. We formalize our finding below.

<span id="page-4-0"></span>**Theorem 3.1** (SOC characteristics of SB). The kinetic-optimal drift  $u_{\star}^{\star}$  in (8) solves an SOC problem

<span id="page-4-1"></span>
$$\min_{u} \mathbb{E}_{X \sim p^{u}} \left[ \int_{0}^{1} \frac{1}{2} \|u_{t}(X_{t})\|^{2} dt + \log \frac{\hat{\varphi}_{1}(X_{1})}{\nu(X_{1})} \right] \quad s.t. \quad (2).$$

Theorem 3.1 suggests that every SB problem (3) can be solved like an SOC problem (4) with the terminal cost  $g(x) := \log \frac{\hat{\varphi}_1(x)}{\nu(x)}$ . Comparing to the formulation in Adjoint Sampling (Havens et al., 2025), the two SOC problems, namely (7) and (9), differ in their terminal costs—where  $p_1^{\text{base}}$  is replaced by  $\hat{\varphi}_1$ —and the relaxation of the source distribution from Dirac delta  $X_0 = 0$  to general source  $\mu(X_0)$ .

How  $\hat{\varphi}_1(\cdot)$  debiases non-memoryless SOC problems Taking a closer look at the effect of  $\hat{\varphi}_1$ , notice that the optimal distribution of the SB problem—according to Theorem 3.1 and (5)—follows

<span id="page-4-2"></span>
$$p^{\star}(X_0, X_1) = p^{\text{base}}(X_0, X_1) \exp\left(-\log\frac{\hat{\varphi}_1(X_1)}{\nu(X_1)} - \log\varphi_0(X_0)\right), \tag{10}$$

where " $-\log \varphi_0$ " is the equivalent initial value function. One can verify that the marginal at the terminal time t=1 indeed satisfies the target distribution,

$$p^{\star}(X_{1}) = \int p^{\star}(X_{0}, X_{1}) dX_{0} \stackrel{\text{(10)}}{=} \frac{\nu(X_{1})}{\hat{\varphi}_{1}(X_{1})} \int p^{\text{base}}(X_{0}, X_{1}) \frac{1}{\varphi_{0}(X_{0})} dX_{0}$$

$$\stackrel{\text{(8a)}}{=} \frac{\nu(X_{1})}{\hat{\varphi}_{1}(X_{1})} \int p^{\text{base}}(X_{1}|X_{0}) \hat{\varphi}_{0}(X_{0}) dX_{0} \stackrel{\text{(8b)}}{=} \nu(X_{1}).$$
(11)

<span id="page-4-4"></span>That is, the optimality equations in (8), in their essence, construct a specific function  $\hat{\varphi}_1(\cdot)$  that eliminates the initial value function bias associated with any non-memoryless processes, thereby ensuring that the optimal distribution satisfies the target  $\nu$  at t=1.

### <span id="page-4-7"></span>3.2 Adjoint Sampling with General Source Distribution

We now specialize Theorem 3.1 to sampling Boltzmann distributions (1), where  $\nu(x) \propto e^{-E(x)}$ , and hence the terminal cost of the new SOC problem in (9) becomes  $\log \frac{\hat{\varphi}_1(x)}{\nu(x)} = E(x) + \log \hat{\varphi}_1(x)$ . To encourage minimal transportation cost (Chen and Georgiou, 2015; Peyré and Cuturi, 2019), we consider the Brownian-motion base process with a degenerate base drift  $f_t := 0$ . Applying Adjoint Matching (AM; Domingo-Enrich et al., 2025) to the resulting SOC problem leads to

<span id="page-4-3"></span>
$$u^{\star} = \arg\min_{u} \mathbb{E}_{p_{t|0,1}^{\text{base}} p_{0,1}^{\bar{u}}} \left[ \|u_t(X_t) + \sigma_t \left( \nabla E + \nabla \log \hat{\varphi}_1 \right) (X_1) \|^2 \right], \quad \bar{u} = \text{stopgrad}(u). \tag{12}$$

Note that the AM objective in (12) functions as a self-consistency loss—in that both the regression and its expectation depend on the optimization variable u. This makes (12) particularly suitable for learning SB-based diffusion samplers, unlike previous matching-based SB methods (Shi et al., 2023; Liu et al., 2024), which all require ground-truth target samples from  $X_1 \sim \nu$ .

Computing the AM objective in (12) requires knowing  $\nabla \log \hat{\varphi}_1(x)$ , which, as we discussed in (11), serves as a *corrector* that debiases the optimization toward the desired target. Notably, this corrector function  $\nabla \log \hat{\varphi}_1(x)$  also admits a variational form (Peluchetti, 2022, 2023; Shi et al., 2023):

<span id="page-4-6"></span>
$$\nabla \log \hat{\varphi}_1 = \arg \min_{h} \mathbb{E}_{p_{0,1}^{u^*}} \left[ \|h(X_1) - \nabla_{x_1} \log p^{\text{base}}(X_1 | X_0) \|^2 \right]. \tag{13}$$

To summarize, Equations (12) and (13) characterize two distinct matching objectives that any kinetic-optimal drift  $u_t^{\star}$  of SBs must satisfy. When the source distribution degenerates to Dirac delta  $X_0 := 0$ , (13) is minimized at  $\nabla \log p_1^{\text{base}}$ , and (12) simply recovers the objective used in Adjoint Sampling (Havens et al., 2025). In other words, (12) and (13) should be understood as a generalization of Adjoint Sampling to handle arbitrary—including non-memoryless—source distributions.

<span id="page-4-5"></span><sup>&</sup>lt;sup>3</sup>Formally,  $\nabla \log \hat{\varphi}_t(x)$  is the kinetic-optimal drift along the reversed time coordinate s := 1 - t, and (13) is its variational formulation, *i.e.*, the Markovian projection at s = 0; see Section A.2 for details.

#### <span id="page-5-2"></span>Algorithm 1 Adjoint Schrödinger Bridge Sampler (ASBS)

Require: Sample-able source X<sup>0</sup> ∼ µ, differentiable energy E(x), parametrized uθ(t, x) and hϕ(x)

- 1: Initialize h (0) ϕ := 0
- 2: for stage k in 1, 2, . . . do
- 3: Update drift u (k) θ by solving [\(14\)](#page-5-0) ▷ adjoint matching
  - by solving [\(15\)](#page-5-1) ▷ corrector matching
- 4: Update corrector h (k) ϕ
- 5: end for

![](_page_5_Figure_8.jpeg)

Figure 2 Illustration of ASBS on a 2D example. By alternatively minimizing the Adjoint Matching (AM) objective [\(14\)](#page-5-0) and the Corrector Matching (CM) objective [\(15\)](#page-5-1), ASBS progressively learns a better corrector h (k) ϕ that debiases the SOC problem for the control u (k) . Note that since the corrector is initialized with h ϕ := 0, the first AM stage simply regresses u θ to the energy gradient ∇E.

### 3.3 Alternating Optimization with Adjoint and Corrector Matching

Building upon the theoretical characterization in [Section 3.2,](#page-4-7) we aim to design a learning algorithm that finds a diffusion sampler satisfying [\(12\)](#page-4-3) and [\(13\)](#page-4-6), which correspond to two simple matching-based objectives. However, these matching objectives cannot be naively implemented due to their interdependency: Solving [\(12\)](#page-4-3) for the kinetic-optimal drift u ⋆ requires knowing ∇ log ˆφ1. Likewise, solving [\(13\)](#page-4-6) for the corrector function ∇ log ˆφ<sup>1</sup> requires samples from u ⋆ . We relax the interdependency with an alternating optimization scheme. Specifically, given an approximation of ∇ log ˆφ<sup>1</sup> ≈ h (k−1) from the previous stage k − 1, we first update the drift u (k) with the AM objective:

<span id="page-5-0"></span>
$$u^{(k)} := \arg\min_{u} \mathbb{E}_{p_{t|0,1}^{\text{base}} p_{0,1}^{\bar{u}}} \left[ \| u_t(X_t) + \sigma_t(\nabla E + h^{(k-1)})(X_1) \|^2 \right], \quad \bar{u} = \text{stopgrad}(u). \tag{14}$$

Then, we use the resulting drift u (k) to update h (k) by minimizing the following matching objective, which—in light of the corrector role of ∇ log ˆφ1—we refer to as the Corrector Matching objective:

<span id="page-5-1"></span>
$$h^{(k)} := \arg\min_{h} \mathbb{E}_{p_{0,1}^{u(k)}} \left[ \|h(X_1) - \nabla_{x_1} \log p^{\text{base}}(X_1 | X_0) \|^2 \right]. \tag{15}$$

[Equation \(15\)](#page-5-1) should be distinguish from the bridge-matching objectives in data-driven SB methods [\(Shi](#page-14-5) [et al.,](#page-14-5) [2023;](#page-14-5) [Somnath et al.,](#page-14-6) [2023\)](#page-14-6), where X<sup>1</sup> must be drawn from the target distribution ν. In contrast, the matching objectives in [\(14\)](#page-5-0) and [\(15\)](#page-5-1) depend only on model samples at the current stage X<sup>1</sup> ∼ p u (k) <sup>θ</sup> (X1|X0), hence can be used to learn SB-based diffusion samplers at scale.

The alternating optimization between [\(14\)](#page-5-0) and [\(15\)](#page-5-1) creates a sequence of updates, (u (0), h(0)) → · · ·(u (k) , h(k) ) → · · · , that may be thought of as running coordinate descent between the control u and the corrector h. Intuitively, at each stage k, we first find the control u (k) that best aligns with the corrector from previous stage, h (k−1), then update the corrector h (k) accordingly to reflect the "memorylessness" of the current control u (k) .

We summarize our method, Adjoint Schrödinger Bridge Sampler (ASBS), in Algorithm 1, while leaving the full details with additional components, such as replay buffers, in Section C. Finally, we prove that this alternating optimization indeed converges to the kinetic-optimal drift  $u^*$  in (8).

<span id="page-6-0"></span>**Theorem 3.2** (Global convergence of ASBS). Algorithm 1 converges to the Schrödinger bridge solution of (3), provided all matching stages achieve their critical points, i.e.,

$$\lim_{k \to \infty} u^{(k)} = u^*.$$

### 4 Theoretical Analysis

We provide the proof of Theorem 3.2 and highlight theoretical insights throughout. While ASBS is specialized to a degenerate base drift  $f_t := 0$ , all theoretical results here apply to general  $f_t$ . To simplify notation, we omit the parameters  $\theta$ ,  $\phi$  and reparametrize the corrector by  $h^{(k)} = \nabla \log \bar{h}^{(k)}$ . All proofs are left in Section B.

Our first result presents a variational characteristic to the solution of the AM objective in (14).

**Theorem 4.1** (Adjoint Matching solves a forward half bridge). Let  $p^{u^{(k)}}$  be the path distribution induced by the drift  $u^{(k)}$  in (14) at stage k. Then,  $p^{u^{(k)}}$  solves the following variational problem:

<span id="page-6-2"></span>
$$p^{u^{(k)}} = \arg\min_{p} \left\{ D_{\text{KL}}(p||q^{\bar{h}^{(k-1)}}) : p_0 = \mu \right\}, \tag{16}$$

where  $q^{\bar{h}^{(k-1)}}$  is the path distribution induced by a "backward" SDE on the reversed time coordinate s:=1-t, defined by the corrector from the previous stage  $\bar{h}^{(k-1)}$ :

$$dY_s = \left[ -f_s(Y_s) + \sigma_s^2 \nabla \log \phi_s(Y_s) \right] ds + \sigma_s dW_s, \quad \phi_s(y) = \int p_{1-s|0}^{\text{base}}(y|z) \phi_1(z) dz, \tag{17}$$

<span id="page-6-1"></span>with the boundary conditions  $Y_0 \sim \nu$  and  $\phi_0(y) = \bar{h}^{(k-1)}(y)$ .

Theorem 4.1 suggests that any SOC problems with the terminal cost  $g(x) := \log \frac{\bar{h}^{(k)}(x)}{\nu(x)}$  can be reinterpreted as KL minimization w.r.t. a specific backward SDE (17) that is fully characterized by  $\nu$ —which serves as its source distribution—and  $\bar{h}^{(k)}$ —which defines its drift through the function  $\phi_s(y)$ . The objective in (16) differs from the one in the original SB problem (3) by disregarding the target boundary constraint,  $X_1 \sim \nu$ . Consequently, (16) only solves a forward half bridge.

Next, we show that the CM objective (15) admits a similar variational form, except backward in time.

**Theorem 4.2** (Corrector Matching solves a backward half bridge). Let  $\bar{h}^{(k)}$  be the corrector in (15) at stage k. Then, the path distribution  $q^{\bar{h}^{(k)}}$  solves the following variational problem:

<span id="page-6-3"></span>
$$q^{\bar{h}^{(k)}} = \underset{q}{\operatorname{arg\,min}} \left\{ D_{\mathrm{KL}}(p^{u^{(k)}}||q) : q_1 = \nu \right\}$$
 (18)

Unlike (16), the objective in (18) disregards the source boundary constraint  $\mu$  instead, thereby solving a backward half bridge. Theorems 4.1 and 4.2 imply that our ASBS in Algorithm 1 implicitly employs an optimization scheme that alternates between solving forward and backward half bridges, thereby instantiating the celebrated Iterative Proportional Fitting algorithm (IPF; Fortet, 1940; Kullback, 1968). Combining with the analysis by (De Bortoli et al., 2021) leads to our final result in Theorem 3.2.

### 5 Related Works

**Data-driven Schrödinger Bridges** The SB problem has attracted notable interests in machine learning due to its connection to diffusion-based generative models (Wang et al., 2021). Earlier methods implemented classical IPF algorithms (De Bortoli et al., 2021; Vargas et al., 2021; Chen et al., 2022), with scalability later enhanced by bridge matching-based methods (Shi et al., 2023; Liu et al., 2024). Unlike ASBS, all of them

focus on generative modeling and assume access to extensive target samples during training, making them unsuitable for sampling from Boltzmann distributions.

SB-inspired Diffusion Samplers Notably, in the context of diffusion samplers, the SB formulation has been constantly emphasized as a mathematically appealing framework for both theoretical analysis and method motivation [\(Zhang and Chen,](#page-14-2) [2022;](#page-14-2) [Vargas et al.,](#page-14-9) [2024;](#page-14-9) [Richter and Berner,](#page-13-4) [2024;](#page-13-4) [Havens et al.,](#page-12-0) [2025\)](#page-12-0). None of the prior methods, however, offers general solutions to learning SB-based diffusion samplers, instead specializing to either the memoryless condition or non-matching-based objectives, which largely complicate the learning process (see [Table 1\)](#page-1-1). Conceptually, our ASBS stands closest to SSB [\(Bernton et al.,](#page-11-3) [2019\)](#page-11-3) by learning general SB samplers. However, the two methods differ fundamentally in scalability: SSB is a Sequential Monte Carlo-based method [\(Chopin,](#page-11-15) [2002\)](#page-11-15) augmented with learned transition kernels using Gaussian-approximated SB potentials. As with many MCMC-augmented samplers [\(Gabrié et al.,](#page-12-8) [2022;](#page-12-8) [Matthews et al.,](#page-13-13) [2022\)](#page-13-13), SSB requires extensive evaluations on the energy E(x), in contrast to ASBS, which is much more energy-efficient.

Learning-augmented MCMC This class of methods can be thought of as extension of classical sampling methods—such as MCMC [\(Metropolis et al.,](#page-13-0) [1953;](#page-13-0) [HASTINGS,](#page-12-9) [1970\)](#page-12-9), Sequential Monte Carlo (SMC; [Del Moral et al.,](#page-11-2) [2006\)](#page-11-2) and Annealed Importance Sampling (AIS; [Neal,](#page-13-1) [2001\)](#page-13-1)—where traditional proposal distributions are replaced with modern machine learning models. For instance, [Arbel et al.](#page-10-1) [\(2021\)](#page-10-1) and [Gabrié](#page-12-8) [et al.](#page-12-8) [\(2022\)](#page-12-8) use normalizing flows [\(Chen et al.,](#page-11-16) [2018\)](#page-11-16) as learned proposal distributions, whereas [Matthews](#page-13-13) [et al.](#page-13-13) [\(2022\)](#page-13-13) employ stochastic normalizing flow [\(Wu et al.,](#page-14-10) [2020\)](#page-14-10). More recently, [Chen et al.](#page-11-5) [\(2025\)](#page-11-5) have explored the use of diffusion models [\(Song et al.,](#page-14-1) [2021;](#page-14-1) [Ho et al.,](#page-12-1) [2020\)](#page-12-1). However, training these models typically requires computing importance weights, which necessitates a large number of energy evaluations.

MCMC-augmented Diffusion Samplers Alternatively, methods of this class adopt modern generative models to sampling Boltzmann distributions and incorporate MCMC techniques to mitigate the lack of explicit target samples. For example, [Phillips et al.](#page-13-5) [\(2024\)](#page-13-5), [\(De Bortoli et al.,](#page-11-6) [2024\)](#page-11-6) and [\(Akhound-Sadegh et al.,](#page-10-0) [2024\)](#page-10-0) employ score matching objective from score-based diffusion models [\(Song et al.,](#page-14-1) [2021;](#page-14-1) [Ho et al.,](#page-12-1) [2020\)](#page-12-1). In contrast, [Albergo and Vanden-Eijnden](#page-10-2) [\(2025\)](#page-10-2) base their method on action matching objectives [\(Neklyudov](#page-13-14) [et al.,](#page-13-14) [2023\)](#page-13-14). However, estimating target samples requires computing importance weights, which makes these methods computationally expensive in terms of energy function evaluations.

# 6 Experiments

Benchmarks We evaluate our ASBS on three classes of multi-particle energy functions E(x).

- Synthetic energy functions These are classical potentials based on pair-wise distances of an n-particle system, where E(x) is known analytically. Following [\(Akhound-Sadegh et al.,](#page-10-0) [2024;](#page-10-0) [Chen et al.,](#page-11-5) [2025\)](#page-11-5), we consider a 2D 4-particle Double-Well potential (DW-4), a 1D 5-particle Many-Well potential (MW-5), a 3D 13-particle Lennard-Jones potential (LJ-13) and a 3D 55-particle Lennard-Jones potential (LJ-55). For the ground-truth samples, we sample analytically from MW-5 and use the MCMC samples from [\(Klein](#page-12-10) [et al.,](#page-12-10) [2023\)](#page-12-10) for the rest of three potentials.
- Alanine dipeptide This is a molecule consisting of 22 atoms in 3D. Specifically, we consider the alanine dipeptide in an implicit solvent and aim to sample from its Boltzmann distribution at a temperature 300K. Following prior methods [\(Zhang and Chen,](#page-14-2) [2022;](#page-14-2) [Wu et al.,](#page-14-10) [2020\)](#page-14-10), we use the energy function E(x) from the OpenMM library [\(Eastman et al.,](#page-11-17) [2017\)](#page-11-17) and consider a more structural internal coordinate with the dimension d = 60. The ground-truth samples contain 10<sup>7</sup> configurations, simulated from Molecular Dynamics [\(Midgley et al.,](#page-13-3) [2023\)](#page-13-3).
- Amortized conformer generation Finally, we consider a new benchmark proposed in [\(Havens et al.,](#page-12-0) [2025\)](#page-12-0) for large-scale conformer generation. Conformers are locally stable configurations located at the local minima of the molecule's potential energy surface [\(Hawkins,](#page-12-11) [2017\)](#page-12-11). Sampling conformers is essentially a conditional generation task, targeting a Boltzmann distribution ν(x|g) ∝ e − <sup>1</sup> <sup>τ</sup> <sup>E</sup>(x|g) at a low temperature τ ≪ 1, conditioned on the molecular topology g ∈ G. The training set Gtrain contains 24,477 molecular topologies from SPICE [\(Eastman et al.,](#page-11-18) [2023\)](#page-11-18), represented by the SMILES strings [\(Weininger,](#page-14-11) [1988\)](#page-14-11), whereas the test set Gtest contains 80 topologies from SPICE and another 80 from GEOM-DRUGS [\(Axelrod](#page-10-3) [and Gomez-Bombarelli,](#page-10-3) [2022\)](#page-10-3). As with [\(Havens et al.,](#page-12-0) [2025\)](#page-12-0), we consider E(x|g) a foundation model

<span id="page-8-0"></span>**Table 2** Results on the synthetic energy functions for n-particle bodies with their corresponding dimensions d. Following (Chen et al., 2025; Havens et al., 2025), we report Sinkhorn for MW-5 and the Wasserstein-2 distances w.r.t samples,  $W_2$ , and energies,  $E(\cdot)W_2$ , for the rest. All values are averaged over three random trials. Best results are highlighted.

|                                    | MW-5 $(d=5)$                  | DW-4 $(d = 8)$                |                                     | LJ-13 $(d = 39)$              |                                     | LJ-55                         | (d=165)                                        |
|------------------------------------|-------------------------------|-------------------------------|-------------------------------------|-------------------------------|-------------------------------------|-------------------------------|------------------------------------------------|
| Method                             | Sinkhorn ↓                    | $W_2\downarrow$               | $E(\cdot) \mathcal{W}_2 \downarrow$ | $W_2\downarrow$               | $E(\cdot) \mathcal{W}_2 \downarrow$ | $W_2\downarrow$               | $E(\cdot) \mathcal{W}_2 \downarrow$            |
| PDDS (Phillips et al., 2024)       |                               | $0.92$ $\pm 0.08$             | $0.58 \scriptstyle{\pm 0.25}$       | 4.66±0.87                     | $56.01{\scriptstyle\pm10.80}$       | _                             |                                                |
| SCLD (Chen et al., 2025)           | $0.44{\scriptstyle \pm 0.06}$ | $1.30 \scriptstyle{\pm 0.64}$ | $0.40 \scriptstyle{\pm 0.19}$       | $2.93{\scriptstyle \pm 0.19}$ | $27.98 \scriptstyle{\pm}~1.26$      |                               |                                                |
| PIS (Zhang and Chen, 2022)         | $0.65{\scriptstyle\pm0.25}$   | $0.68 \scriptstyle{\pm 0.28}$ | $0.65 \scriptstyle{\pm 0.25}$       | $1.93{\scriptstyle \pm 0.07}$ | $18.02 \scriptstyle{\pm}~1.12$      | $4.79 \scriptstyle{\pm 0.45}$ | $228.70 \scriptstyle{\pm 131.27}$              |
| DDS (Vargas et al., 2023)          | $0.63{\scriptstyle\pm0.24}$   | $0.92 \scriptstyle{\pm 0.11}$ | $0.90 \scriptstyle{\pm 0.37}$       | $1.99 \scriptstyle{\pm 0.13}$ | $24.61 \scriptstyle{\pm}~8.99$      | $4.60 \scriptstyle{\pm 0.09}$ | $173.09 \scriptstyle{\pm} \scriptstyle{18.01}$ |
| LV-PIS (Richter and Berner, 2024)  | _                             | $1.04 \scriptstyle{\pm 0.29}$ | $1.89{\scriptstyle \pm 0.89}$       | _                             |                                     | _                             |                                                |
| iDEM (Akhound-Sadegh et al., 2024) |                               | $0.70 \scriptstyle{\pm 0.06}$ | $0.55 \scriptstyle{\pm 0.14}$       | $1.61 \scriptstyle{\pm 0.01}$ | $30.78 \scriptstyle{\pm 24.46}$     | $4.69{\scriptstyle\pm1.52}$   | $93.53 \scriptstyle{\pm}16.31$                 |
| AS (Havens et al., 2025)           | $0.32 \scriptstyle{\pm 0.06}$ | $0.62 \scriptstyle{\pm 0.06}$ | $0.55 \scriptstyle{\pm 0.12}$       | $1.67 \scriptstyle{\pm 0.01}$ | $2.40{\scriptstyle \pm\ 1.25}$      | $4.04 \scriptstyle{\pm 0.05}$ | $30.83 \pm 8.19$                               |
| ASBS (Ours)                        | $0.15{\scriptstyle \pm 0.02}$ | $0.43 \scriptstyle{\pm 0.05}$ | $0.20_{\pm 0.11}$                   | $1.59{\scriptstyle \pm 0.03}$ | $1.99 \scriptstyle{\pm~1.01}$       | 4.00±0.03                     | 28.10± 8.15                                    |

<span id="page-8-1"></span>![](_page_8_Figure_2.jpeg)

<span id="page-8-2"></span>![](_page_8_Figure_3.jpeg)

**Figure 3** The energy histograms of DW-4 and LJ-13 from Table 2. ASBS generates samples whose energy profiles closely match those of the ground-truth samples.

Figure 4 Complexity w.r.t. the number of function evaluation (NFE) on LJ-13 potential.

eSEN from (Fu et al., 2025), which predicts energy with density-functional-theory accuracy at a much lower computational cost. We use CREST conformers (Pracht et al., 2024) as the ground-truth samples.

Baselines and evaluation We compare ASBS with a wide range of diffusion samplers, including PIS (Zhang and Chen, 2022), DDS (Vargas et al., 2023), PDDS (Phillips et al., 2024), SCLD (Chen et al., 2025), LV (Richter and Berner, 2024), iDEM (Akhound-Sadegh et al., 2024) and finally Adjoint Sampling (AS; Havens et al., 2025). For the conformer generation task, we include additionally a domain-specific baseline, RDKit ETKDG (Riniker and Landrum, 2015), which relies on chemistry-based heuristics. The evaluation pipelines are consistent with prior methods, where we adopt the SCLD setup for MW-5, the PIS setup for alanine dipeptide, and the AS setup for all the rest; see Section D for details.

**ASBS models** For all tasks, we consider a degenerate base drift  $f_t := 0$ , as discussed in Section 3.2, and set  $\sigma_t$  a geometric noise schedule. For energy functions that directly take particle systems as inputs—such as DW, LJ, and eSEN—we parametrize the models  $u_{\theta}$ ,  $h_{\phi}$  with two Equivariant Graph Neural Networks (Satorras et al., 2021) and consider a domain-specific source distribution—the harmonic prior (Jing et al., 2023). Formally, for an n-particle system  $x = \{x_i\}_{i=0}^n$ , the harmonic prior  $\mu_{\text{harmonic}}(x)$  is a quadratic potential that can be sampled analytically from an anisotropic Gaussian:

<span id="page-8-3"></span>
$$\mu_{\text{harmonic}}(x) \propto \exp(-\frac{\alpha}{2} \sum_{i,j} \|x_i - x_j\|^2). \tag{19}$$

For other energy functions, we use standard fully-connected neural networks and consider Gaussian priors. All models are trained with Adam (Kingma and Ba, 2015) and, following standard practices (Havens et al., 2025; Akhound-Sadegh et al., 2024), utilize replay buffers; see Section C for details.

**Results** Table 2 presents the results on synthetic energy functions. Notably, ASBS consistently outperforms prior diffusion samplers across all energy functions. In Figure 3, we compare the energy histograms of DW-4 and LJ-13 potentials between the ground-truth MCMC samples and those from ASBS. It is evident that ASBS generates samples that closely resemble the target Boltzmann distribution  $\nu(x) \propto e^{-E(x)}$ , resulting in energy profiles E(x) that are almost indistinguishable from the ground truth. Computationally, Figure 4

<span id="page-9-1"></span>Table 3 Comparison between diffusion samplers on sampling the molecular Boltzmann distribution of the alanine dipeptide. We report the KL divergence DKL for the 1D marginal across five torsion angles and the Wasserstein-2 W<sup>2</sup> on jointly (ϕ, ψ), known as Ramachandran plots (see [Figure 5\)](#page-9-0). Best results are highlighted.

|                            |           | on each torsion's marginal ↓<br>DKL |           |           |           |           |  |  |
|----------------------------|-----------|-------------------------------------|-----------|-----------|-----------|-----------|--|--|
| Method                     | ϕ         | ψ                                   | γ1        | γ2        | γ3        | (ϕ, ψ)    |  |  |
| PIS (Zhang and Chen, 2022) | 0.05±0.03 | 0.38±0.49                           | 5.61±1.24 | 4.49±0.03 | 4.60±0.03 | 1.27±1.19 |  |  |
| DDS (Vargas et al., 2023)  | 0.03±0.01 | 0.16±0.07                           | 2.44±0.96 | 0.03±0.00 | 0.03±0.00 | 0.68±0.09 |  |  |
| AS (Havens et al., 2025)   | 0.09±0.09 | 0.04±0.04                           | 0.17±0.17 | 0.56±0.09 | 0.51±0.06 | 0.65±0.52 |  |  |
| ASBS (Ours)                | 0.02±0.00 | 0.01±0.00                           | 0.03±0.01 | 0.02±0.00 | 0.02±0.00 | 0.25±0.01 |  |  |

<span id="page-9-2"></span>Table 4 Results on large-scale amortized conformer generation, evaluated on two test sets, SPICE and GEOM-DRUGS, both with and without post-processing relaxation. We report the coverage (%) and Absolute Mean RMSD (AMR) of the recall at the threshold 1.0Å. Note that "+RDKit warmup" refers to warm-starting the model u<sup>θ</sup> using RDKit conformers; see [Section D](#page-24-0) for details. Best results without and with RDKit warm-up are highlighted separately.

|                                         | without relaxation |           |             |           | with relaxation |           |             |           |
|-----------------------------------------|--------------------|-----------|-------------|-----------|-----------------|-----------|-------------|-----------|
|                                         | SPICE              |           | GEOM-DRUGS  |           | SPICE           |           | GEOM-DRUGS  |           |
| Method                                  | Coverage ↑         | AMR ↓     | Coverage ↑  | AMR ↓     | Coverage ↑      | AMR ↓     | Coverage ↑  | AMR ↓     |
| RDKit ETKDG (Riniker and Landrum, 2015) | 56.94±35.82        | 1.04±0.52 | 50.81±34.69 | 1.15±0.61 | 70.21±31.70     | 0.79±0.44 | 62.55±31.67 | 0.93±0.53 |
| AS (Havens et al., 2025)                | 56.75±38.15        | 0.96±0.26 | 36.23±33.42 | 1.20±0.43 | 82.41±25.85     | 0.68±0.28 | 64.26±34.57 | 0.89±0.45 |
| ASBS w/ Gaussian prior (Ours)           | 73.04±31.95        | 0.83±0.24 | 50.23±35.98 | 1.05±0.43 | 88.26±20.57     | 0.60±0.24 | 72.32±29.68 | 0.77±0.35 |
| ASBS w/ harmonic prior (Ours)           | 74.05±31.61        | 0.82±0.23 | 53.14±35.69 | 1.03±0.42 | 88.71±18.63     | 0.59±0.24 | 72.77±29.94 | 0.78±0.35 |
| AS +RDKit warmup (Havens et al., 2025)  | 72.21±30.22        | 0.84±0.24 | 52.19±35.20 | 1.02±0.34 | 87.84±19.20     | 0.60±0.23 | 73.88±28.63 | 0.76±0.34 |
| ASBS +RDKit warmup (Ours)               | 77.84±28.37        | 0.79±0.23 | 57.19±35.14 | 0.98±0.40 | 88.08±18.84     | 0.58±0.24 | 73.18±30.09 | 0.76±0.37 |

<span id="page-9-0"></span>![](_page_9_Figure_4.jpeg)

Figure 5 Ramachandran plots for the alanine dipeptide between ground-truth and ASBS samples.

<span id="page-9-3"></span>![](_page_9_Figure_6.jpeg)

Figure 6 Example of ASBS generative process on amortized conformer generation. Given an unseen molecular topology g ∈ Gtest from the test set—COCSc1sc2ccccc2[n+]1[O-] in this case—ASBS transports samples from the harmonic prior X<sup>0</sup> ∼ µharmonic to generate conformers X1.

shows the average number of evaluation required on the energy E(x) and the model uθ(t, x) for each gradient update. ASBS is much more efficient than most diffusion samplers, with a slight overhead compared to AS due to the additional network hϕ(x).

[Table 3](#page-9-1) summarizes the results for alanine dipeptide. Following standard pipeline [\(Zhang and Chen,](#page-14-2) [2022\)](#page-14-2), we generate model samples X<sup>1</sup> ∈ R <sup>60</sup> and extract five torsion angles—including the backbone angles ϕ, ψ and methyl rotation angles γ1, γ2, γ3—all of them exhibit multi-modal distributions. Notably, ASBS achieves lowest KL divergence to the ground-truth marginals across all five torsions. [Figure 5](#page-9-0) further compares the joint distributions of (ϕ, ψ), known as the Ramachandran plots [\(Spencer et al.,](#page-14-12) [2019\)](#page-14-12), between ground-truth and ASBS. While ASBS identifies all high-density modes in the region ϕ ∈ [−π, 0], it misses few low-density modes. This mode-seeking behavior, inherit in all SOC-based diffusion samplers, could be improved with important weighting. We provide further discussions in [Section D.4.](#page-28-0)

[Table 4](#page-9-2) presents the recall for amortized conformer generation compared to ground-truth samples. For prior diffusion samplers, we primarily compare to AS [\(Havens et al.,](#page-12-0) [2025\)](#page-12-0) due to the benchmark's scale. Following AS, we ablate a warm-start stage using RDKit conformers, which are close but not identical to ground-truth

<span id="page-10-4"></span>![](_page_10_Figure_0.jpeg)

Figure 7 Recall coverage curves on amortized conformer generation on the SPICE and GEOM-DRUGS test sets without RDKit warm-start. Note that [Table 4](#page-9-2) reports the recall coverages at the threshold 1.0Å.

samples, and include results with relaxation for post-generation optimization. Since AS is a specific instance of ASBS with a Dirac delta prior—as discussed in [Section 3.2—](#page-4-7)any performance improvements from AS to ASBS highlight the added capability to handle arbitrary priors and, consequently, non-memoryless processes. Remarkably, without any warm-start, ASBS with the harmonic prior [\(19\)](#page-8-3) already matches and, in many cases, surpasses the RDKit-warm-up AS. With warm-start, ASBS achieves best performance across most metrics. This highlights the significance of domain-specific priors, aiding exploration as effectively as warm-start with additional data, which may not always be available. Finally, we visualize the generation process of ASBS with harmonic prior [\(19\)](#page-8-3) in [Figure 6](#page-9-3) and report the recall curves in [Figure 7.](#page-10-4) In practice, we observe that ASBS achieves slightly better results with a harmonic prior compared to a Gaussian prior, with both significantly outperforming AS [\(Havens et al.,](#page-12-0) [2025\)](#page-12-0). See [Section D.4](#page-28-0) for further ablation studies.

# 7 Conclusion and Limitation

We introduced Adjoint Schrödinger Bridge Sampler (ASBS), a new diffusion sampler for Boltzmann distributions that solves general SB problems given only target energy functions. ASBS is based on a scalable matching framework, converges theoretically to the global solution, and performs superiorly across various benchmarks. Despite these encouraging results, further enhancement with importance sampling techniques is worth investigating to mitigate the mode collapse inherent in SOC-inspired diffusion samplers. Exploring its effectiveness in sampling amortized Boltzmann distributions would also be valuable.

# Acknowledgements

The authors would like to thank Aaron Havens, Juno Nam, Xiang Fu, Bing Yan, Brandon Amos, and Brian Karrer for the helpful discussions and comments.

# References

<span id="page-10-0"></span>Tara Akhound-Sadegh, Jarrid Rector-Brooks, Avishek Joey Bose, Sarthak Mittal, Pablo Lemos, Cheng-Hao Liu, Marcin Sendera, Siamak Ravanbakhsh, Gauthier Gidel, Yoshua Bengio, Nikolay Malkin, and Alexander Tong. Iterated denoising energy matching for sampling from Boltzmann densities. In International Conference on Machine Learning (ICML), 2024.

<span id="page-10-2"></span>Michael S Albergo and Eric Vanden-Eijnden. NETS: A non-equilibrium transport sampler. In International Conference on Machine Learning (ICML), 2025.

<span id="page-10-1"></span>Michael Arbel, Alex Matthews, and Arnaud Doucet. Annealed flow transport monte carlo. In International Conference on Machine Learning (ICML), 2021.

<span id="page-10-3"></span>Simon Axelrod and Rafael Gomez-Bombarelli. GEOM: energy-annotated molecular conformations for property prediction and molecular generation. Scientific Data, 9(1):185, 2022.

<span id="page-10-5"></span>Richard Bellman. The theory of dynamic programming. Technical report, Rand corp santa monica ca, 1954.

- <span id="page-11-4"></span>Julius Berner, Lorenz Richter, and Karen Ullrich. An optimal control perspective on diffusion-based generative modeling. Transactions on Machine Learning Research (TMLR), 2024.
- <span id="page-11-3"></span>Espen Bernton, Jeremy Heng, Arnaud Doucet, and Pierre E Jacob. Schrödinger bridge samplers. arXiv preprint arXiv:1912.13170, 2019.
- <span id="page-11-1"></span>Kurt Binder, Dieter W Heermann, and K Binder. Monte Carlo simulation in statistical physics, volume 8. Springer, 1992.
- <span id="page-11-20"></span>Denis Blessing, Xiaogang Jia, Johannes Esslinger, Francisco Vargas, and Gerhard Neumann. Beyond ELBOs: a large-scale evaluation of variational methods for sampling. In International Conference on Machine Learning (ICML), 2024.
- <span id="page-11-0"></span>George EP Box and George C Tiao. Bayesian inference in statistical analysis. John Wiley & Sons, 2011.
- <span id="page-11-19"></span>James Bradbury, Roy Frostig, Peter Hawkins, Matthew James Johnson, Chris Leary, Dougal Maclaurin, George Necula, Adam Paszke, Jake VanderPlas, Skye Wanderman-Milne, and Qiao Zhang. JAX: composable transformations of Python+NumPy programs, 2018. <http://github.com/google/jax>.
- <span id="page-11-5"></span>Junhua Chen, Lorenz Richter, Julius Berner, Denis Blessing, Gerhard Neumann, and Anima Anandkumar. Sequential controlled langevin diffusions. In International Conference on Learning Representations (ICLR), 2025.
- <span id="page-11-16"></span>Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, and David K Duvenaud. Neural ordinary differential equations. In Advances in Neural Information Processing Systems (NeurIPS), 2018.
- <span id="page-11-14"></span>Tianrong Chen, Guan-Horng Liu, and Evangelos A Theodorou. Likelihood training of Schrödinger bridge using forward-backward SDEs theory. In International Conference on Learning Representations (ICLR), 2022.
- <span id="page-11-12"></span>Yongxin Chen and Tryphon Georgiou. Stochastic bridges of linear systems. IEEE Transactions on Automatic Control, 61(2):526–531, 2015.
- <span id="page-11-8"></span>Yongxin Chen, Tryphon T Georgiou, and Michele Pavon. On the relation between optimal transport and Schrödinger bridges: A stochastic control viewpoint. Journal of Optimization Theory and Applications, 169:671–691, 2016.
- <span id="page-11-10"></span>Yongxin Chen, Tryphon T Georgiou, and Michele Pavon. Stochastic control liaisons: Richard sinkhorn meets gaspard monge on a schrödinger bridge. SIAM Review, 63(2):249–313, 2021.
- <span id="page-11-15"></span>Nicolas Chopin. A sequential particle filter method for static models. Biometrika, 89(3):539–552, 2002.
- <span id="page-11-11"></span>Valentin De Bortoli, James Thornton, Jeremy Heng, and Arnaud Doucet. Diffusion Schrödinger bridge with applications to score-based generative modeling. In Advances in Neural Information Processing Systems (NeurIPS), 2021.
- <span id="page-11-6"></span>Valentin De Bortoli, Michael Hutchinson, Peter Wirnsberger, and Arnaud Doucet. Target score matching. arXiv preprint arXiv:2402.08667, 2024.
- <span id="page-11-2"></span>Pierre Del Moral, Arnaud Doucet, and Ajay Jasra. Sequential monte carlo samplers. Journal of the Royal Statistical Society Series B: Statistical Methodology, 68(3):411–436, 2006.
- <span id="page-11-7"></span>Carles Domingo-Enrich, Michal Drozdzal, Brian Karrer, and Ricky T. Q. Chen. Adjoint Matching: Fine-tuning flow and diffusion generative models with memoryless stochastic optimal control. In International Conference on Learning Representations (ICLR), 2025.
- <span id="page-11-17"></span>Peter Eastman, Jason Swails, John D Chodera, Robert T McGibbon, Yutong Zhao, Kyle A Beauchamp, Lee-Ping Wang, Andrew C Simmonett, Matthew P Harrigan, Chaya D Stern, Rafal P. Wiewiora, Bernard R. Brooks, and Vijay S. Pande. OpenMM 7: Rapid development of high performance algorithms for molecular dynamics. PLoS computational biology, 13(7):e1005659, 2017.
- <span id="page-11-18"></span>Peter Eastman, Pavan Kumar Behara, David L Dotson, Raimondas Galvelis, John E Herr, Josh T Horton, Yuezhi Mao, John D Chodera, Benjamin P Pritchard, Yuanqing Wang, Gianni De Fabritiis, and Thomas E. Markland. SPICE, a dataset of drug-like molecules and peptides for training machine learning potentials. Scientific Data, 10(1):11, 2023.
- <span id="page-11-9"></span>Chris Finlay, Jörn-Henrik Jacobsen, Levon Nurbekyan, and Adam Oberman. How to train your neural ODE: The world of jacobian and kinetic regularization. In International Conference on Machine Learning (ICML), 2020.
- <span id="page-11-13"></span>Robert Fortet. Résolution d'un système d'équations de M. Schrödinger. Journal de Mathématiques Pures et Appliquées, 19(1-4):83–105, 1940.

- <span id="page-12-12"></span>Xiang Fu, Brandon M Wood, Luis Barroso-Luque, Daniel S Levine, Meng Gao, Misko Dzamba, and C Lawrence Zitnick. Learning smooth and expressive interatomic potentials for physical property prediction. In International Conference on Machine Learning (ICML), 2025.
- <span id="page-12-8"></span>Marylou Gabrié, Grant M Rotskoff, and Eric Vanden-Eijnden. Adaptive monte carlo augmented with normalizing flows. Proceedings of the National Academy of Sciences, 119(10):e2109420119, 2022.
- <span id="page-12-9"></span>WK HASTINGS. Monte carlo sampling methods using markov chains and their applications. Biometrika, 57(1): 97–109, 1970.
- <span id="page-12-0"></span>Aaron Havens, Benjamin Kurt Miller, Bing Yan, Carles Domingo-Enrich, Anuroop Sriram, Brandon Wood, Daniel Levine, Bin Hu, Brandon Amos, Brian Karrer, Xiang Fu, Guan-Horng Liu, and Ricky T. Q. Chen. Adjoint Sampling: Highly scalable diffusion samplers via Adjoint Matching. In International Conference on Machine Learning (ICML), 2025.
- <span id="page-12-11"></span>Paul CD Hawkins. Conformation generation: The state of the art. Journal of chemical information and modeling, 57 (8):1747–1756, 2017.
- <span id="page-12-1"></span>Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems (NeurIPS), 2020.
- <span id="page-12-15"></span>Kiyosi Itô. On stochastic differential equations, volume 4. American Mathematical Soc., 1951.
- <span id="page-12-3"></span>Bowen Jing, Ezra Erives, Peter Pao-Huang, Gabriele Corso, Bonnie Berger, and Tommi S Jaakkola. EigenFold: Generative protein structure prediction with diffusion models. In International Conference on Learning Representations (ICLR), Workshop Track, 2023.
- <span id="page-12-2"></span>Hilbert J Kappen. Path integrals and symmetry breaking for optimal control theory. Journal of Statistical Mechanics: Theory and Experiment, 2005(11):P11011, 2005.
- <span id="page-12-17"></span>Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. In Advances in Neural Information Processing Systems (NeurIPS), 2022.
- <span id="page-12-13"></span>Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations (ICLR), 2015.
- <span id="page-12-10"></span>Leon Klein, Andrew Foong, Tor Fjelde, Bruno Mlodozeniec, Marc Brockschmidt, Sebastian Nowozin, Frank Noé, and Ryota Tomioka. Timewarp: Transferable acceleration of molecular dynamics by learning time-coarsened dynamics. In Advances in Neural Information Processing Systems (NeurIPS), 2023.
- <span id="page-12-18"></span>Jonas Köhler, Leon Klein, and Frank Noé. Equivariant Flows: Exact likelihood generative learning for symmetric densities. In International Conference on Machine Learning (ICML), 2020.
- <span id="page-12-6"></span>Solomon Kullback. Probability densities with given marginals. The Annals of Mathematical Statistics, 39(4):1236–1243, 1968.
- <span id="page-12-20"></span>Greg Landrum. Rdkit: Open-source cheminformatics. <https://www.rdkit.org>, 2006.
- <span id="page-12-16"></span>Jean-François Le Gall. Brownian motion, martingales, and stochastic calculus. Springer, 2016.
- <span id="page-12-7"></span>Christian Léonard. From the Schrödinger problem to the Monge–Kantorovich problem. Journal of Functional Analysis, 262(4):1879–1920, 2012.
- <span id="page-12-4"></span>Christian Léonard. A survey of the Schrödinger problem and some of its connections with optimal transport. Discrete and Continuous Dynamical Systems, 2013.
- <span id="page-12-14"></span>Christian Léonard, Sylvie Rœlly, and Jean-Claude Zambrini. Reciprocal processes. A measure-theoretical point of view. Probability Surveys, 2014.
- <span id="page-12-19"></span>Daniel S Levine, Muhammed Shuaibi, Evan Walter Clark Spotte-Smith, Michael G Taylor, Muhammad R Hasyim, Kyle Michel, Ilyes Batatia, Gábor Csányi, Misko Dzamba, Peter Eastman, Nathan C. Frey, Xiang Fu, Vahe Gharakhanyan, Aditi S. Krishnapriyan, Joshua A. Rackers, Sanjeev Raja, Ammar Rizvi, Andrew S. Rosen, Zachary Ulissi, Santiago Vargas, C. Lawrence Zitnick, Samuel M. Blau, and Brandon M. Wood. The Open Molecules 2025 (OMol25) dataset, evaluations, and models. arXiv preprint arXiv:2505.08762, 2025.
- <span id="page-12-5"></span>Guan-Horng Liu, Arash Vahdat, De-An Huang, Evangelos A Theodorou, Weili Nie, and Anima Anandkumar. I<sup>2</sup>SB: Image-to-Image Schrödinger bridge. In International Conference on Machine Learning (ICML), 2023.

- <span id="page-13-10"></span>Guan-Horng Liu, Yaron Lipman, Maximilian Nickel, Brian Karrer, Evangelos A Theodorou, and Ricky T. Q. Chen. Generalized Schrödinger bridge matching. In International Conference on Learning Representations (ICLR), 2024.
- <span id="page-13-13"></span>Alex Matthews, Michael Arbel, Danilo Jimenez Rezende, and Arnaud Doucet. Continual repeated annealed flow transport monte carlo. In International Conference on Machine Learning (ICML), 2022.
- <span id="page-13-0"></span>Nicholas Metropolis, Arianna W Rosenbluth, Marshall N Rosenbluth, Augusta H Teller, and Edward Teller. Equation of state calculations by fast computing machines. The journal of chemical physics, 21(6):1087–1092, 1953.
- <span id="page-13-3"></span>Laurence Illing Midgley, Vincent Stimper, Gregor NC Simm, Bernhard Schölkopf, and José Miguel Hernández-Lobato. Flow annealed importance sampling bootstrap. In International Conference on Learning Representations (ICLR), 2023.
- <span id="page-13-1"></span>Radford M Neal. Annealed importance sampling. Statistics and computing, 11:125–139, 2001.
- <span id="page-13-22"></span>F. Neese. The orca program system. WIRES Comput. Molec. Sci., 2(1):73–78, 2012. doi: 10.1002/wcms.81.
- <span id="page-13-14"></span>Kirill Neklyudov, Daniel Severo, and Alireza Makhzani. Action matching: A variational method for learning stochastic dynamics from samples. In International Conference on Machine Learning (ICML), 2023.
- <span id="page-13-18"></span>Edward Nelson. Dynamical theories of Brownian motion, volume 106. Princeton university press, 2020.
- <span id="page-13-2"></span>Frank Noé, Simon Olsson, Jonas Köhler, and Hao Wu. Boltzmann generators: Sampling equilibrium states of many-body systems with deep learning. Science, 365(6457):eaaw1147, 2019.
- <span id="page-13-19"></span>Bernt Øksendal. Stochastic differential equations. In Stochastic Differential Equations, pages 65–84. Springer, 2003.
- <span id="page-13-21"></span>Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Kopf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. Pytorch: An imperative style, high-performance deep learning library. In Advances in Neural Information Processing Systems (NeurIPS), 2019.
- <span id="page-13-11"></span>Stefano Peluchetti. Non-Denoising forward-time diffusions, 2022. <https://openreview.net/forum?id=oVfIKuhqfC>.
- <span id="page-13-12"></span>Stefano Peluchetti. Diffusion bridge mixture transports, Schrödinger bridge problems and generative modeling. arXiv preprint arXiv:2304.00917, 2023.
- <span id="page-13-9"></span>Gabriel Peyré and Marco Cuturi. Computational optimal transport: With applications to data science. Foundations and Trends® in Machine Learning, 11(5-6):355–607, 2019.
- <span id="page-13-5"></span>Angus Phillips, Hai-Dang Dau, Michael John Hutchinson, Valentin De Bortoli, George Deligiannidis, and Arnaud Doucet. Particle denoising diffusion sampler. In International Conference on Machine Learning (ICML), 2024.
- <span id="page-13-15"></span>Philipp Pracht, Stefan Grimme, Christoph Bannwarth, Fabian Bohle, Sebastian Ehlert, Gereon Feldmann, Johannes Gorges, Marcel Müller, Tim Neudecker, Christoph Plett, Sebastian Spicher, Pit Steinbach, Patryk A. Wesołowski, and Felix Zeller. CREST—A program for the exploration of low-energy molecular chemical space. The Journal of Chemical Physics, 160(11), 2024.
- <span id="page-13-4"></span>Lorenz Richter and Julius Berner. Improved sampling via learned diffusions. In International Conference on Learning Representations (ICLR), 2024.
- <span id="page-13-16"></span>Sereina Riniker and Gregory A Landrum. Better informed distance geometry: using what we know to improve conformation generation. Journal of chemical information and modeling, 55(12):2562–2574, 2015.
- <span id="page-13-20"></span>Simo Särkkä and Arno Solin. Applied stochastic differential equations, volume 10. Cambridge University Press, 2019.
- <span id="page-13-17"></span>Vıctor Garcia Satorras, Emiel Hoogeboom, and Max Welling. E(n) equivariant graph neural networks. In International Conference on Machine Learning (ICML), 2021.
- <span id="page-13-7"></span>Erwin Schrödinger. Über die Umkehrung der Naturgesetze, volume IX. Sitzungsberichte der Preuss Akad. Wissen. Phys. Math. Klasse, Sonderausgabe, 1931.
- <span id="page-13-8"></span>Erwin Schrödinger. Sur la théorie relativiste de l'électron et l'interprétation de la mécanique quantique. In Annales de l'institut Henri Poincaré, 1932.
- <span id="page-13-6"></span>Neta Shaul, Ricky T. Q. Chen, Maximilian Nickel, Matthew Le, and Yaron Lipman. On kinetic optimal probability paths for generative models. In International Conference on Machine Learning (ICML), 2023.

- <span id="page-14-5"></span>Yuyang Shi, Valentin De Bortoli, Andrew Campbell, and Arnaud Doucet. Diffusion Schrödinger bridge matching. In Advances in Neural Information Processing Systems (NeurIPS), 2023.
- <span id="page-14-6"></span>Vignesh Ram Somnath, Matteo Pariset, Ya-Ping Hsieh, Maria Rodriguez Martinez, Andreas Krause, and Charlotte Bunne. Aligned diffusion Schrödinger bridges. In Conference on Uncertainty in Artificial Intelligence (UAI), 2023.
- <span id="page-14-1"></span>Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Scorebased generative modeling through stochastic differential equations. In International Conference on Learning Representations (ICLR), 2021.
- <span id="page-14-12"></span>Ryan K Spencer, Glenn L Butterfoss, John R Edison, James R Eastwood, Stephen Whitelam, Kent Kirshenbaum, and Ronald N Zuckermann. Stereochemistry of polypeptoid chain configurations. Biopolymers, 110(6):e23266, 2019.
- <span id="page-14-13"></span>Vincent Stimper, Bernhard Schölkopf, and José Miguel Hernández-Lobato. Resampling base distributions of normalizing flows. In International Conference on Artificial Intelligence and Statistics (AISTATS), 2022.
- <span id="page-14-4"></span>Emanuel Todorov. Linearly-solvable Markov decision problems. In Advances in Neural Information Processing Systems (NeurIPS), 2007.
- <span id="page-14-0"></span>Mark E Tuckerman. Statistical mechanics: theory and molecular simulation. Oxford university press, 2023.
- <span id="page-14-8"></span>Francisco Vargas, Pierre Thodoroff, Neil D Lawrence, and Austen Lamacraft. Solving Schrödinger bridges via maximum likelihood. Entropy, 2021.
- <span id="page-14-3"></span>Francisco Vargas, Will Grathwohl, and Arnaud Doucet. Denoising diffusion samplers. In International Conference on Learning Representations (ICLR), 2023.
- <span id="page-14-9"></span>Francisco Vargas, Shreyas Padhy, Denis Blessing, and Nikolas Nüsken. Transport meets variational inference: Controlled monte carlo diffusions. In International Conference on Learning Representations (ICLR), 2024.
- <span id="page-14-7"></span>Gefei Wang, Yuling Jiao, Qian Xu, Yang Wang, and Can Yang. Deep generative learning via Schrödinger bridge. In International Conference on Machine Learning (ICML), 2021.
- <span id="page-14-11"></span>David Weininger. Smiles, a chemical language and information system. 1. introduction to methodology and encoding rules. Journal of chemical information and computer sciences, 28(1):31–36, 1988.
- <span id="page-14-10"></span>Hao Wu, Jonas Köhler, and Frank Noé. Stochastic normalizing flows. In Advances in Neural Information Processing Systems (NeurIPS), 2020.
- <span id="page-14-2"></span>Qinsheng Zhang and Yongxin Chen. Path integral sampler: A stochastic control approach for sampling. In International Conference on Learning Representations (ICLR), 2022.

# Contents

|   | A Additional Preliminary                                          | 16             |
|---|-------------------------------------------------------------------|----------------|
|   | A.1<br>Stochastic Optimal Control (SOC)<br>                       | 16             |
|   | A.2<br>Schrödinger Bridge (SB)                                    | 18             |
| B | Proofs                                                            | 19             |
|   | B.1<br>Preliminary and Additional Theoretical Results             | 19             |
|   | B.2<br>Missing Proofs in Main Paper                               | 20             |
| C | Practical Implementation of ASBS                                  | 22             |
|   |                                                                   |                |
|   |                                                                   |                |
| D | Experiment Details                                                | 25             |
|   | D.1<br>Synthetic Energy Functions<br><br>D.2<br>Alanine dipeptide |                |
|   | D.3<br>Amortized conformer generation<br>                         | 25<br>27<br>28 |

# <span id="page-15-0"></span>A Additional Preliminary

### <span id="page-15-1"></span>A.1 Stochastic Optimal Control (SOC)

In this subsection, we expand [Section 2](#page-2-0) with details. Recall the SOC problem in [\(4\)](#page-2-1):

<span id="page-15-4"></span><span id="page-15-3"></span><span id="page-15-2"></span>
$$\min_{u} \mathbb{E}_{X \sim p^{u}} \left[ \int \frac{1}{2} \|u_{t}(X_{t})\|^{2} dt + g(X_{1}) \right]$$
 (20a)

s.t. 
$$dX_t = [f_t(X_t) + \sigma_t u_t(X_t)] dt + \sigma_t dW_t, \quad X_0 \sim \mu.$$
 (20b)

Similar to [\(8\)](#page-3-2), the optimal control to [\(20\)](#page-15-2) can be characterized through an optimality equation:

<span id="page-15-5"></span>
$$u_t^{\star}(x) = -\sigma_t \nabla V_t(x), \quad \text{where} \quad V_t(x) = -\log \int p_{1|t}^{\text{base}}(y|x)e^{-V_1(y)} dy, \quad V_1(x) = g(x)$$
 (21)

is the value function known to satisfy the Hamilton–Jacobi–Bellman (HJB) equation [\(Bellman,](#page-10-5) [1954\)](#page-10-5). We provide further characterization below.

Optimal distribution The optimization problem in [\(20\)](#page-15-2) is known analytically. Specifically, notice that the entropy-regularized objective in [\(20\)](#page-15-2) can be reformulated as:

$$D_{KL}(p(X)||p^{\text{base}}(X)) + \mathbb{E}_{p(X)}[g(X_1)]$$

$$= D_{KL}(p(X_0)||p^{\text{base}}(X_0)) + \mathbb{E}_{p(X_0)}[D_{KL}(p(X|X_0)||p^{\text{base}}(X|X_0)) + \mathbb{E}_{p(X|X_0)}[g(X_1)]]$$

$$= D_{KL}(p(X_0)||p^{\text{base}}(X_0)) + \mathbb{E}_{p(X_0)}[D_{KL}(p(X|X_0)||p^{\text{base}}(X|X_0)e^{-g(X_1)})]$$
(22)

where we shorthand X ≡ X[0,1] and denote p base the base distribution induced by [\(20b\)](#page-15-3) with u := 0, i.e., the uncontrolled distribution. Minimizing [\(22\)](#page-15-4) w.r.t. p yields

<span id="page-15-6"></span>
$$p^{\star}(X|X_0) = \frac{1}{Z(X_0)} p^{\text{base}}(X|X_0) e^{-g(X_1)}, \qquad p^{\star}(X_0) = p^{\text{base}}(X_0)$$
(23)

where Z(X0) is the normalization term defined by

<span id="page-15-7"></span>
$$Z(X_0) := \int p^{\text{base}}(X|X_0)e^{-g(X_1)}dX = \int p^{\text{base}}(X_1|X_0)e^{-g(X_1)}dX_1$$
(24)

which is exactly e <sup>−</sup><sup>V</sup> (X0) due to [\(21\)](#page-15-5). Combing [\(23\)](#page-15-6) and [\(24\)](#page-15-7) leads to the the optimal distribution in [\(5\)](#page-2-4), which we restate below for completeness:

<span id="page-15-8"></span>
$$p^{\star}(X) = p^{\text{base}}(X)e^{-g(X_1) + V_0(X_0)} \implies p^{\star}(X_0, X_1) = p^{\text{base}}(X_0, X_1)e^{-g(X_1) + V_0(X_0)}$$
(25)

Adjoint Matching (AM) Scalable computational methods for solving (20) have been challenging, as naively back-propagating through (20) induces prohibitively high computational cost. Instead, Adjoint Matching (Domingo-Enrich et al., 2025) employs a matching-based objective, named Adjoint Matching (AM):

$$u^* = \arg\min_{u} \mathbb{E}_{X \sim p^{\bar{u}}} \left[ \|u_t(X_t) + \sigma_t a_t\|^2 \right], \qquad \bar{u} = \text{stopgrad}(u), \tag{26a}$$

where 
$$-da_t = a_t \cdot \nabla f_t(X_t)dt$$
,  $a_1 = \nabla g(X_1)$  (26b)

is the backward dynamics of the (lean) adjoint state  $a_t \equiv a(t; X_{[t,1]})$ . It has been proven that the unique critical point of (26) is the optimal control  $u^*$ , implying a new characteristics of the optimal control  $u^*$  using the adjoint state:

<span id="page-16-6"></span><span id="page-16-5"></span><span id="page-16-2"></span><span id="page-16-0"></span>
$$u_t^{\star}(x) = -\sigma_t \mathbb{E}_{p^{\star}}[a_t | X_t = x]. \tag{27}$$

Adjoint Sampling (AS) Recently, Havens et al. (2025) introduced an adaptation of AM tailored to sampling Boltzmann distribution  $\nu(x) \propto e^{-E(x)}$  by considering

<span id="page-16-3"></span>
$$f_t := 0, \qquad \mu(x) := \delta_0(x), \qquad g(x) := \log \frac{p_1^{\text{base}}(x)}{\nu(x)}.$$
 (28)

That is, AS considers the following SOC problem with a degenerate base drift, a Dirac delta prior, and a specific instantiation of the terminal cost  $g(x) := \log \frac{p_1^{\text{base}}(x)}{\nu(x)}$ :

<span id="page-16-1"></span>
$$\min_{u} \mathbb{E}_{X \sim p^{u}} \left[ \int_{0}^{1} \frac{1}{2} \|u_{t}(X_{t})\|^{2} dt + \log \frac{p_{1}^{\text{base}}(X_{1})}{\nu(X_{1})} \right] \quad \text{s.t. } dX_{t} = \sigma_{t} u_{t}(X_{t}) dt + \sigma_{t} dW_{t}, \quad X_{0} = 0.$$
 (29)

Notably, this SOC problem (29) admits a simplified adjoint state  $a_t$  and a degenerate initial value function  $V_0(x)$ :

<span id="page-16-4"></span>
$$a_t \stackrel{\text{(26b)}}{=} \nabla g(X_1) \stackrel{\text{(28)}}{=} \nabla \log p_1^{\text{base}}(X_1) + \nabla E(X_1) \qquad \forall t \in [0, 1]$$

$$(30)$$

$$V_0(x) \stackrel{\text{(28)}}{=} -\log \int p_1^{\text{base}}(y) \frac{\nu(y)}{p_1^{\text{base}}(y)} dy = -\log 1 = 0, \tag{31}$$

which further implies that the optimal distribution  $p^*$  is a reciprocal process (Léonard et al., 2014):

<span id="page-16-7"></span>
$$p^{\star}(X) \stackrel{\text{(31)}}{=} p^{\text{base}}(X)e^{-V_1(X_1)} \stackrel{\text{(28)}}{=} p^{\text{base}}(X)\frac{\nu(X_1)}{p_1^{\text{base}}(X_1)} = p^{\text{base}}(X|X_1)p^{\star}(X_1). \tag{32}$$

Combining the adjoint characteristics of the optimal control (27) with the simplified adjoint state  $a_t$  in (30) and optimal distribution  $p^*$  (32) motivates the following *Reciprocal Adjoint Matching (RAM)* objective used in AS, where the unique critical point remains to be the optimal control  $u^*$  in (21).

$$u^{\star} = \arg\min_{u} \mathbb{E}_{p_{t|1}^{\text{base}} p_1^{\bar{u}}} \left[ \| u_t(X_t) + \sigma_t \left( \nabla E + \nabla \log p_1^{\text{base}} \right) (X_1) \|^2 \right], \quad \bar{u} = \text{stopgrad}(u). \tag{33}$$

Remark on reciprocal representation The reciprocal representation of the optimal-controlled distribution  $p^*$  in (32) extends to general SOC problems (20) with non-trivial base drifts and source distributions. Specifically, any optimal-controlled distribution that solves (20) can be factorized by

$$p^{\star}(X) = p^{\text{base}}(X|X_0, X_1)p^{\star}(X_0, X_1). \tag{34}$$

We leave a formal statement in Theorem B.3 and Corollary B.4.

AS with linear base drift and Gaussian prior (Figure 1) Here, we discuss an alternative instantiation of AM for sampling with linear base drift and Gaussian prior, which reproduces the leftmost plot in Figure 1. Consider

<span id="page-16-8"></span>
$$f_t(x) := -\frac{1}{2}\beta_t x, \qquad \mu(x) := \mathcal{N}(x; 0, I), \qquad \sigma_t := \sqrt{\beta_t}, \qquad g(x) := \log \frac{p_1^{\text{base}}(x)}{\nu(x)}.$$
 (35)

where β<sup>t</sup> is chosen such that (ft, µ, σt) fulfill the memoryless condition. For instance, [Figure 1](#page-3-0) adopts the VPSDE [\(Song et al.,](#page-14-1) [2021\)](#page-14-1) setup:

<span id="page-17-1"></span>
$$\beta_t = (1 - t)\beta_{\text{max}} + t\beta_{\text{min}}, \qquad \beta_{\text{max}} = 20, \qquad \beta_{\text{min}} = 0.1.$$
 (36)

Similar to [\(30\)](#page-16-6), the resulting SOC problem admits a simplified adjoint state at:

<span id="page-17-2"></span>
$$a_t \stackrel{\text{(26b)}}{=} \kappa_t \cdot \nabla g(X_1) \stackrel{\text{(35)}}{=} \kappa_t \cdot (\nabla \log p_1^{\text{base}}(X_1) + \nabla E(X_1)), \qquad \kappa_t := e^{-\frac{1}{2} \int_t^1 \beta_\tau d\tau} \stackrel{\text{(36)}}{=} e^{-\frac{1}{4}(1-t)(\beta_t + \beta_1)}$$
(37)

and the RAM objective becomes

$$u^{\star} = \underset{u}{\operatorname{arg\,min}} \, \mathbb{E}_{p_{t|0,1}^{\operatorname{base}} p_{0,1}^{\bar{u}}} \left[ \| u_t(X_t) + \sigma_t \kappa_t \left( \nabla E + \nabla \log p_1^{\operatorname{base}} \right) (X_1) \|^2 \right], \quad \bar{u} = \operatorname{stopgrad}(u). \tag{38}$$

Note that p base t|0,1 can be sampled analytically:

$$p_{t|0,1}^{\text{base}}(X_t|X_0, X_1) \stackrel{\text{(35)}}{=} \mathcal{N}(X_t; \frac{\bar{\kappa}_t(1 - \kappa_t^2)}{1 - \bar{\kappa}_1^2} X_0 + \frac{\kappa_t(1 - \bar{\kappa}_t^2)}{1 - \bar{\kappa}_1^2} X_1, \frac{(1 - \kappa_t^2)(1 - \bar{\kappa}_t^2)}{1 - \bar{\kappa}_1^2} I), \tag{39}$$

<span id="page-17-0"></span>where κ<sup>t</sup> is defined in [\(37\)](#page-17-2) and κ¯<sup>t</sup> := e − <sup>1</sup> 2 R <sup>t</sup> β<sup>τ</sup> dτ [\(36\)](#page-17-1) = e − <sup>1</sup> 4 t(βt+β0) .

### A.2 Schrödinger Bridge (SB)

In this subsection, we provide additional clarification on SB and specifically the derivation of [\(13\)](#page-4-6). Recall the optimality equations of SB in [\(8\)](#page-3-2):

<span id="page-17-5"></span>
$$u_t^{\star}(x) = \sigma_t \nabla \log \varphi_t(x), \quad \text{where} \begin{cases} \varphi_t(x) = \int p_{1|t}^{\text{base}}(y|x)\varphi_1(y)\mathrm{d}y, & \varphi_0(x)\hat{\varphi}_0(x) = \mu(x) \\ \hat{\varphi}_t(x) = \int p_{t|0}^{\text{base}}(x|y)\hat{\varphi}_0(y)\mathrm{d}y, & \varphi_1(x)\hat{\varphi}_1(x) = \nu(x) \end{cases}$$
(40a)

Just like how the value function of an SOC problem fully characterizes the optimal control and its corresponding optimal distribution, so does the SB potential φt(x):

<span id="page-17-6"></span><span id="page-17-4"></span><span id="page-17-3"></span>
$$p^{\star}(X) = p^{\text{base}}(X) \frac{\varphi_1(X_1)}{\varphi_0(X_0)} = p^{\text{base}}(X|X_0)\varphi_1(X_1)\hat{\varphi}_0(X_0), \tag{41}$$

where the last equality is due to p base(X) = p base(X|X0)µ(X0) and then invoking [\(40a\)](#page-17-3). Note that [\(41\)](#page-17-4) recovers [\(10\)](#page-4-2) by marginalizing over t ∈ (0, 1). Due to the construction of φt(x) and φˆt(x) in [\(40\)](#page-17-5), the marginal optimal distribution admits a strikingly simple factorization:

$$p_t^{\star}(x) = \int p^{\text{base}}(X, X_t = x | X_0) \varphi_1(X_1) \hat{\varphi}_0(X_0) dX$$

$$= \int \int p^{\text{base}}(X_1 | X_t = x) p^{\text{base}}(X_t = x | X_0) \varphi_1(X_1) \hat{\varphi}_0(X_0) dX_0 dX_1$$

$$= \left( \int p^{\text{base}}(X_t = x | X_0) \hat{\varphi}_0(X_0) dX_0 \right) \left( \int p^{\text{base}}(X_1 | X_t = x) \varphi_1(X_1) dX_1 \right)$$

$$= \hat{\varphi}_t(x) \varphi_t(x), \tag{42}$$

or, more generally,

<span id="page-17-9"></span><span id="page-17-8"></span><span id="page-17-7"></span>
$$p_{s,t}^{\star}(y,x) = p_{t|s}^{\text{base}}(x|y)\hat{\varphi}_s(y)\varphi_t(x), \qquad s \le t.$$
(43)

Derivation of [\(13\)](#page-4-6) We now provide a simpler derivation of [\(13\)](#page-4-6) compared to its original derivation based on path measure theory [\(Shi et al.,](#page-14-5) [2023\)](#page-14-5):

$$\nabla \log \hat{\varphi}_{t}(x) \stackrel{\text{(40b)}}{=} \frac{1}{\hat{\varphi}_{t}(x)} \nabla_{x} \int p_{t|0}^{\text{base}}(x|y) \hat{\varphi}_{0}(y) dy$$

$$= \frac{1}{\hat{\varphi}_{t}(x)} \int \nabla_{x} \log p_{t|0}^{\text{base}}(x|y) p_{t|0}^{\text{base}}(x|y) \hat{\varphi}_{0}(y) dy$$

$$= \int \nabla_{x} \log p_{t|0}^{\text{base}}(x|y) p_{0|t}^{\star}(y|x) dy, \tag{44}$$

where the last equality follows by

$$p_{0|t}^{\star}(y|x) \stackrel{\text{(42)}}{=} \frac{p_{0,t}^{\star}(y,x)}{\hat{\varphi}_{t}(x)\varphi_{t}(x)} \stackrel{\text{(43)}}{=} \frac{p_{t|0}^{\text{base}}(x|y)\hat{\varphi}_{0}(y)\varphi_{t}(x)}{\hat{\varphi}_{t}(x)\varphi_{t}(x)} = \frac{p_{t|0}^{\text{base}}(x|y)\hat{\varphi}_{0}(y)}{\hat{\varphi}_{t}(x)}.$$

[Equation \(44\)](#page-17-9) implies a matching-based variational formulation of ∇ log ˆφt(·)—also known as the bridge matching objective in data-driven SB [\(Shi et al.,](#page-14-5) [2023;](#page-14-5) [Liu et al.,](#page-12-5) [2023\)](#page-12-5).

<span id="page-18-3"></span>
$$\nabla \log \hat{\varphi}_t = \operatorname*{arg\,min}_h \mathbb{E}_{p_{0,t}^{\star}} \left[ \|h_t(X_t) - \nabla_{x_t} \log p^{\mathrm{base}}(X_t | X_0) \|^2 \right]. \tag{45}$$

<span id="page-18-0"></span>[Equation \(45\)](#page-18-3) recovers [\(13\)](#page-4-6) at t = 1.

### B Proofs

### <span id="page-18-1"></span>B.1 Preliminary and Additional Theoretical Results

Lemma B.1 (Itô lemma [\(Itô,](#page-12-15) [1951\)](#page-12-15)). Let X<sup>t</sup> be the solution to the Itô SDE:

$$dX_t = f_t(X_t)dt + \sigma_t dW_t.$$

Then, the stochastic process vt(Xt), where v ∈ C 1,2 ([0, 1], R d ), is also an Itô process:

<span id="page-18-7"></span>
$$dv_t(X_t) = \left[ \partial_t v_t(X_t) + \nabla v_t(X_t) \cdot f + \frac{1}{2} \sigma_t^2 \Delta v_t(X_t) \right] dt + \sigma_t \nabla v_t(X_t) \cdot dW_t.$$
 (46)

Lemma B.2 (Laplacian trick). For any twice-differentiable function π such that π(x) ̸= 0, it holds that

<span id="page-18-6"></span>
$$\frac{1}{\pi(x)}\Delta\pi(x) = \|\nabla\log\pi(x)\|^2 + \Delta\log\pi(x) \tag{47}$$

Proof.

$$\Delta \pi(x) = \nabla \cdot \nabla \pi(x)$$

$$= \nabla \cdot (\pi(x)\nabla \log \pi(x))$$

$$= \nabla \pi(x) \cdot \nabla \log \pi(x) + \pi(x)\Delta \log \pi(x)$$

$$= \pi(x) \left( \|\nabla \log \pi(x)\|^2 + \Delta \log \pi(x) \right)$$

<span id="page-18-2"></span>Theorem B.3 (SB characteristics of SOC). The optimal distribution p <sup>⋆</sup> of the SOC problem in [\(20\)](#page-15-2) is also the solution to the following SB problem:

<span id="page-18-5"></span>
$$\underset{p}{\operatorname{arg\,min}} \left\{ D_{\mathrm{KL}}(p||p^{\mathrm{base}}) : p_0 = \mu, \quad p_1 = p_1^{\star} \right\}. \tag{48}$$

Proof. We aim to show that there exist a transform such that the SOC's optimality equation [\(21\)](#page-15-5) can be reinterpreted as the ones for SB [\(40\)](#page-17-5). To this end, consider

<span id="page-18-4"></span>
$$\varphi_t(x) := e^{-V_t(x)}, \qquad \hat{\varphi}_t(x) := e^{V_t(x)} p_t^{\star}(x).$$
 (49)

One can verify that the value function Vt(x) defined in [\(21\)](#page-15-5) can be rewritten as

$$\varphi_t(x) = \int p_{1|t}^{\text{base}}(y|x)\varphi_1(y)\mathrm{d}y.$$

On the other hand, we can expand  $\hat{\varphi}_t(x)$  by

$$\hat{\varphi}_{t}(x) = e^{V_{t}(x)} \int p^{*}(X|X_{t} = x) dX$$

$$= e^{V_{t}(x)} \int p^{\text{base}}(X_{1}|X_{t} = x) p^{\text{base}}(X_{t} = x, X_{0}) e^{-V_{1}(X_{1}) + V_{0}(X_{0})} dX_{1} dX_{0} \qquad \text{by (25)}$$

$$= e^{V_{t}(x)} \int p^{\text{base}}(X_{t} = x, X_{0}) e^{-V_{t}(x) + V_{0}(X_{0})} dX_{0} \qquad \text{by (21)}$$

$$= \int p^{\text{base}}(X_{t} = x|X_{0}) \mu(X_{0}) e^{V_{0}(X_{0})} dX_{0}$$

$$= \int p^{\text{base}}(x|y) \hat{\varphi}_{0}(y) dy. \qquad \text{by (49)}$$

Combined, the optimality equation (21) for the SOC problem can be rewritten equivalently as

$$u_t^{\star}(x) = \sigma_t \nabla \log \varphi_t(x), \quad \text{where} \begin{cases} \varphi_t(x) = \int p_{1|t}^{\text{base}}(y|x)\varphi_1(y)\mathrm{d}y, & \varphi_0(x)\hat{\varphi}_0(x) = \mu(x), \\ \hat{\varphi}_t(x) = \int p_{t|0}^{\text{base}}(x|y)\hat{\varphi}_0(y)\mathrm{d}y, & \varphi_1(x)\hat{\varphi}_1(x) = p_1^{\star}(x). \end{cases}$$

We conclude that  $p^*$  indeed solves (48).

<span id="page-19-1"></span>Corollary B.4 (Reciprocal process of the SOC problem). The optimal distribution  $p^*$  of the SOC problem in (20) is a reciprocal process, i.e.,

$$p^{\star}(X) = p^{\text{base}}(X|X_0, X_1)p^{\star}(X_0, X_1). \tag{51}$$

### <span id="page-19-0"></span>**B.2** Missing Proofs in Main Paper

**Proof of Theorem 3.1** Comparing (8a) to (21), we can reinterpret  $\varphi_t(x)$  as an value function  $V_t(x)$  by reinterpreting

$$V_t(x) := -\log \varphi_t(x), \quad g(x) := -\log \varphi_1(x) \stackrel{\text{(8b)}}{=} \log \frac{\hat{\varphi}_1(x)}{\nu(x)}.$$

That is, the kinetic-optimal drift of SB solves an SOC problem (4) with a terminal cost  $g(x) := \frac{\hat{\varphi}_1(x)}{\nu(x)}$ .

**Proof of Theorem 4.1** For notational simplicity, we will denote  $q \equiv q^{\bar{h}^{(k-1)}}$  throughout the proof. We first rewrite the backward SDE (17) in the forward direction (Nelson, 2020):

$$dX_t = \left[ f_t - \sigma_t^2 \nabla \log \phi_t + \sigma_t^2 \nabla \log q_t \right] dt + \sigma_t dW_t, \quad X_1 \sim \nu,$$

where we rewrite  $\phi_t(x)$  w.r.t. the forward time coordinate:

<span id="page-19-2"></span>
$$\phi_t(x) = \int p_{t|0}^{\text{base}}(x|y)\phi_0(y)dy, \qquad \phi_1(x) = \bar{h}^{(k-1)}(x).$$
 (52)

Note that (52) admits an equivalent PDE form by invoking Feynman-Kac formula (Le Gall, 2016):

<span id="page-19-3"></span>
$$\partial_t \phi_t(x) = -\nabla \cdot (f_t \phi_t) + \frac{\sigma_t^2}{2} \Delta \phi_t(x), \quad \phi_1(x) = \bar{h}^{(k-1)}(x). \tag{53}$$

On the other hand, the dynamics of  $\partial_t q$  follows the Fokker Plank equation ( $\emptyset$ ksendal, 2003):

$$\partial_t q_t = -\nabla \cdot \left( \left( f_t - \sigma_t^2 \nabla \log \phi_t + \sigma_t^2 \nabla \log q_t \right) q_t \right) + \frac{1}{2} \sigma_t^2 \Delta q_t$$
$$= \nabla \cdot \left( \left( \sigma_t^2 \nabla \log \phi_t - f_t \right) q_t \right) - \frac{1}{2} \sigma_t^2 \Delta q_t,$$

<span id="page-20-1"></span>and straightforward calculation yields

$$\partial_t \log q_t = \sigma_t^2 \Delta \log \phi_t - \nabla \cdot f_t + \left(\sigma_t^2 \nabla \log \phi_t - f_t\right) \cdot \nabla \log q_t - \frac{1}{2} \sigma_t^2 \|\nabla \log q_t\|^2 - \frac{1}{2} \sigma_t^2 \Delta \log q_t, \tag{54}$$

where we apply the Laplacian trick [\(47\)](#page-18-6) to <sup>1</sup> <sup>q</sup> ∆q = ∥∇ log qt∥ <sup>2</sup> + ∆ log qt.

Now, recall that p is the path distribution induced by the following SDE:

<span id="page-20-3"></span><span id="page-20-0"></span>
$$dX_t = [f_t(X_t) + \sigma_t u_t(X_t)] dt + \sigma_t dW_t, \qquad X_0 \sim \mu.$$
(55)

Invoke Ito Lemma [\(46\)](#page-18-7) to log qt(Xt), where X<sup>t</sup> follows [\(55\)](#page-20-0):

$$d\log q_{t} = \left[\frac{\partial_{t}\log q_{t} + \nabla \log q_{t} \cdot (f_{t} + \sigma_{t}u_{t}) + \frac{1}{2}\sigma_{t}^{2}\Delta \log q_{t}}{\det + \sigma_{t}\nabla \log q_{t} \cdot dW_{t}}\right] dt + \sigma_{t}\nabla \log q_{t} \cdot dW_{t}$$

$$\stackrel{(54)}{=} \left[\sigma_{t}^{2}\Delta \log \phi_{t} - \nabla \cdot f_{t} + \sigma_{t}^{2}\nabla \log \phi_{t} \cdot \nabla \log q_{t} - \frac{1}{2}\sigma_{t}^{2}\|\nabla \log q_{t}\|^{2} + \nabla \log q_{t} \cdot (\sigma_{t}u_{t})\right] dt$$

$$+ \sigma_{t}\nabla \log q_{t} \cdot dW_{t}$$

$$(56)$$

Likewise, invoke Ito Lemma [\(46\)](#page-18-7) to log ϕt(Xt), where X<sup>t</sup> follows [\(55\)](#page-20-0):

d log ϕ<sup>t</sup>

$$\begin{aligned}
&= \left[ \partial_{t} \log \phi_{t} + \nabla \log \phi_{t} \cdot (f_{t} + \sigma_{t} u_{t}) + \frac{1}{2} \sigma_{t}^{2} \Delta \log \phi_{t} \right] dt + \sigma_{t} \nabla \log \phi_{t} \cdot dW_{t} \\
&\stackrel{(53)}{=} \left[ -\nabla \cdot f_{t} + \frac{\sigma_{t}^{2}}{2} \frac{\Delta \phi_{t}}{\phi_{t}} + \nabla \log \phi_{t} \cdot (\sigma_{t} u_{t}) + \frac{1}{2} \sigma_{t}^{2} \Delta \log \phi_{t} \right] dt + \sigma_{t} \nabla \log \phi_{t} \cdot dW_{t} \\
&\stackrel{(47)}{=} \left[ -\nabla \cdot f_{t} + \frac{\sigma_{t}^{2}}{2} \left( \|\nabla \log \phi_{t}\|^{2} + \Delta \log \phi_{t} \right) + \nabla \log \phi_{t} \cdot (\sigma_{t} u_{t}) + \frac{1}{2} \sigma_{t}^{2} \Delta \log \phi_{t} \right] dt + \sigma_{t} \nabla \log \phi_{t} \cdot dW_{t} \\
&= \left[ -\nabla \cdot f_{t} + \frac{\sigma_{t}^{2}}{2} \|\nabla \log \phi_{t}\|^{2} + \nabla \log \phi_{t} \cdot (\sigma_{t} u_{t}) + \sigma_{t}^{2} \Delta \log \phi_{t} \right] dt + \sigma_{t} \nabla \log \phi_{t} \cdot dW_{t} 
\end{aligned} \tag{57}$$

Subtracting [\(57\)](#page-20-2) from [\(56\)](#page-20-3) leads to

<span id="page-20-4"></span>
$$d\log\phi_t - d\log q_t = \left[\frac{1}{2}\|u_t + \sigma_t\nabla\log\phi_t - \sigma_t\nabla\log q_t\|^2 - \frac{1}{2}\|u_t\|^2\right]dt + \sigma_t\nabla\log\frac{\phi_t}{q_t} \cdot dW_t.$$
 (58)

Finally, we are ready to compute the variational objective in [\(16\)](#page-6-2):

<span id="page-20-2"></span>
$$D_{\mathrm{KL}}(p||q^{\bar{h}^{(k-1)}}) = \mathbb{E}_{X \sim p^{u}} \left[ \int_{0}^{1} \frac{1}{2} \|u_{t}(X_{t}) + \sigma_{t} \nabla \log \phi_{t}(X_{t}) - \sigma_{t} \nabla \log q_{t}(X_{t}) \|^{2} dt \right]$$

$$\stackrel{(58)}{=} \mathbb{E}_{X \sim p^{u}} \left[ \int_{0}^{1} \left( \frac{1}{2} \|u_{t}(X_{t})\|^{2} + d \log \phi_{t}(X_{t}) - d \log q_{t}(X_{t}) \right) dt \right]$$

$$= \mathbb{E}_{X \sim p^{u}} \left[ \int_{0}^{1} \frac{1}{2} \|u_{t}(X_{t})\|^{2} dt + \log \frac{\phi_{1}(X_{1})}{q_{1}(X_{1})} - \log \frac{\phi_{0}(X_{0})}{q_{0}(X_{0})} \right]$$

$$\propto \mathbb{E}_{X \sim p^{u}} \left[ \int_{0}^{1} \frac{1}{2} \|u_{t}(X_{t})\|^{2} dt + \log \frac{\bar{h}^{(k-1)}(X_{1})}{\nu(X_{1})} \right].$$

$$(60)$$

That is, we have shown that the variational objective DKL(p||q <sup>h</sup>¯(k−1) ) is equivalent (up to an additive constant) to an SOC problem [\(60\)](#page-20-5). Applying Reciprocal Adjoint Matching [\(Havens et al.,](#page-12-0) [2025\)](#page-12-0) with the reciprocal process from [Corollary B.4](#page-19-1) conclude that DKL(p||q <sup>h</sup>¯(k−1) ) is minimized by p u (k) . □

Proof of [Theorem 4.2](#page-6-0) For notational simplicity, we will denote p (k) ≡ p u (k) throughout the proof. Let q be the path distribution induced by a backward SDE, propagating along the time coordinate s := 1 − t:

<span id="page-20-6"></span><span id="page-20-5"></span>
$$dY_s = [-f_s(Y_s) + \sigma_s v_s(Y_s)] ds + \sigma_s dW_s, \qquad Y_0 \sim \nu.$$

Next, rewrite the forward SDE  $p^{(k)}$  in the backward direction:

$$dY_s = \left[ -f_s - \sigma_s u_s^{(k)} + \sigma_s^2 \nabla \log p_s^{(k)} \right] ds + \sigma_s dW_s, \quad Y_0 \sim p_t^{(k)}|_{t=1}.$$

By Theorem B.3, we know that  $p^{(k)}$  is the SB solution, thereby satisfying

$$u_t^{(k)}(x) = \sigma_t \nabla \log \varphi_t(x), \text{ where } \begin{cases} \varphi_t(x) = \int p_{1|t}^{\text{base}}(y|x)\varphi_1(y)\mathrm{d}y, & \varphi_0(x)\hat{\varphi}_0(x) = \mu(x) \\ \hat{\varphi}_t(x) = \int p_{t|0}^{\text{base}}(x|y)\hat{\varphi}_0(y)\mathrm{d}y, & \varphi_1(x)\hat{\varphi}_1(x) = p_1^{(k)}(x) \end{cases}$$
(61a)

Since we are working with the backward time coordinate s, it is convenience to define  $\phi_s := \hat{\varphi}_{1-t}$  and rewrite (61b) by

<span id="page-21-3"></span><span id="page-21-1"></span>
$$\phi_s(y) = \int p_{1-s|0}^{\text{base}}(y|z)\phi_1(z)dz, \quad \phi_0(y) = \frac{p_1^{(k)}(y)}{\varphi_1(y)}. \tag{62}$$

Now, expanding the variational objective with Girsanov Theorem yields (Särkkä and Solin, 2019)

<span id="page-21-2"></span>
$$D_{\mathrm{KL}}(p^{(k)}||q) = \mathbb{E}_{Y \sim p^{(k)}} \left[ \int_0^1 \frac{1}{2} \| -\sigma_s \nabla \log \varphi_s(Y_s) + \sigma_s \nabla \log p_s^{(k)}(Y_s) - v_s(Y_s) \|^2 ds \right], \tag{63}$$

which is minimized point-wise at

$$v_s^{\star}(y) = \sigma_s \nabla \log \frac{p_s^{(k)}(y)}{\varphi_s(y)} \stackrel{(42)}{=} \sigma_s \nabla \log \hat{\varphi}_s(y).$$

In other words, the backward SDE that minimizes (63) must obey

$$dY_s = \left[ -f_s(Y_s) + \sigma_s^2 \nabla \log \phi_s(Y_s) \right] ds + \sigma_s dW_s, \quad Y_0 \sim \nu,$$

with  $\phi_s$  defined in (62). That is, we have concluded so far that

$$q^{p_1^{(k)}/\varphi_1} = \arg\min_{q} \left\{ D_{\text{KL}}(p^{(k)}||q) : q_1 = \nu \right\}.$$
 (64)

Hence, it remains to be shown that the minimizer  $\bar{h}_1^{(k)}$  of the CM objective at stage k equals  $\frac{p_1^{(k)}}{\varphi_1}$ . This is indeed the case since  $p^{(k)}$  is the SB solution:

$$\nabla \log \bar{h}^{(k)} \stackrel{\text{(15)}}{:=} \arg \min_{h} \mathbb{E}_{p_{0,1}^{(k)}} \left[ \|h(X_1) - \nabla_{x_1} \log p^{\text{base}}(X_1 | X_0) \|^2 \right] \stackrel{\text{(45)}}{=} \nabla \log \hat{\varphi}_1 \stackrel{\text{(42)}}{=} \nabla \log \frac{p_1^{(k)}}{\varphi_1}.$$

# <span id="page-21-0"></span>C Practical Implementation of ASBS

Algorithm 2 summarizes the practical implementation of ASBS, where we expand the adjoint and corrector matching steps (*i.e.*, lines 3 and 4 in Algorithm 1) to full details. Table 5 provides the hyper-parameters for each task. We break down each component as follows:

**Harmonic prior**  $\mu_{\text{harmonic}}$  Recall the harmonic prior in (19):

<span id="page-21-4"></span>
$$\mu_{\text{harmonic}}(x) \propto \exp(-\frac{1}{2} \sum_{i,j} ||x_i - x_j||^2).$$
 (65)

In practice, we set  $\alpha = 1$  and implement (65) as an anisotropic Gaussian. For instance, for a 2-particle system in 3D, *i.e.*,  $x = [x_1; x_2] \in \mathbb{R}^6$ , we can rewrite (65) as a quadratic potential,

$$\exp(-\frac{1}{2}||x_1 - x_2||^2) = \exp(x^{\top}Rx), \quad \text{where } R = \begin{bmatrix} 1 & 0 & 0 & -\frac{1}{2} & 0 & 0\\ 0 & 1 & 0 & 0 & -\frac{1}{2} & 0\\ 0 & 0 & 1 & 0 & 0 & -\frac{1}{2}\\ -\frac{1}{2} & 0 & 0 & 1 & 0 & 0\\ 0 & -\frac{1}{2} & 0 & 0 & 1 & 0\\ 0 & 0 & -\frac{1}{2} & 0 & 0 & 1 \end{bmatrix}, \tag{66}$$

#### <span id="page-22-0"></span>Algorithm 2 Adjoint Schrödinger Bridge Sampler (ASBS)

**Require:** Sample-able source  $X_0 \sim \mu$ , differentiable energy E(x), parametrized drift  $u_{\theta}(t,x)$  and corrector  $h_{\phi}(x)$ , replay buffers  $\mathcal{B}_{\mathrm{adj}}$  and  $\mathcal{B}_{\mathrm{crt}}$ , number of stages K, numbers of AM and CM epochs  $M_{\mathrm{adi}}$  and  $M_{\mathrm{crt}}$ , number of resamples N, number of gradient steps L, time scaling  $\lambda_t$ , maximum energy gradient norm

1: Initialize  $h_{\phi}^{(0)} := 0$ 

▷ adjoint matching

- for stage k in  $1, 2, \ldots, K$  do
- for epoch in  $1, 2, \ldots, M_{\text{adj}}$  do
- Sample from model  $\{(X_0^{(i)}, X_1^{(i)})\}_{i=1}^N \sim p^{\bar{u}^{(k)}}$ , where  $\bar{u}^{(k)} = \text{stopgrad}(u_{\theta}^{(k)})$ 4:
- Compute adjoint target  $a_t^{(i)} := \operatorname{stopgrad}\left(\operatorname{clip}\left(\nabla E(X_1^{(i)}), \alpha_{\max}\right) + h_{\phi}^{(k)}(X_1^{(i)})\right)$ 5:
- Update replay buffer  $\mathcal{B}_{\text{adj}} \leftarrow \mathcal{B}_{\text{adj}} \cup \{(X_0^{(i)}, X_1^{(i)}, a_t^{(i)})\}_{i=1}^N$ 6:
- Take L gradient steps  $\nabla_{\theta} \mathcal{L}_{AM}$  w.r.t. the AM objective: 7:

$$\mathcal{L}_{\mathrm{AM}}(\theta) := \mathbb{E}_{t \sim \mathcal{U}[0,1], (X_0, X_1, a_t) \sim \mathcal{B}_{\mathrm{adj}}, X_t \sim p^{\mathrm{base}}(\cdot | X_0, X_1)} \left[ \lambda_t \| u_{\theta}^{(k)}(t, X_t) + \sigma_t a_t \|^2 \right]$$

end for 8:

- for epoch in  $1, 2, \ldots, M_{\rm crt}$  do ▷ corrector matching 9: Sample from model  $\{(X_0^{(i)}, X_1^{(i)})\}_{i=1}^N \sim p^{\bar{u}^{(k)}}, \text{ where } \bar{u}^{(k)} = \text{stopgrad}(u_{\theta}^{(k)})$ 10:
- Update replay buffer  $\mathcal{B}_{crt} \leftarrow \mathcal{B}_{crt} \cup \{(X_0^{(i)}, X_1^{(i)})\}_{i=1}^N$ 11:
- Take L gradient steps  $\nabla_{\phi} \mathcal{L}_{\text{CM}}$  w.r.t. the CM objective: 12:

$$\mathcal{L}_{\text{CM}}(\phi) := \mathbb{E}_{(X_0, X_1) \sim \mathcal{B}_{\text{crt}}} \left[ \| h_{\phi}^{(k)}(X_1) - \nabla_{x_1} \log p^{\text{base}}(X_1 | X_0) \|^2 \right]$$

- 13: end for
- 14: end for
- 15: **return** Kinetic-optimal drift  $u^* \approx u_\theta(t, x)$

and then sample x from the Gaussian  $\mathcal{N}(x; 0, (R + \epsilon I)^{-1})$ , where we set  $\epsilon = 10^{-4}$ .

We consider two types of noise schedule. Noise schedule  $\sigma_t$ 

• The geometric noise schedule (Song et al., 2021; Karras et al., 2022) monotonically decays from t=0 to 1 according to some prescribed  $\beta_{\min}$  and  $\beta_{\max}$ :

$$\sigma_t \stackrel{\text{geometric}}{:=} \beta_{\min} \left( \frac{\beta_{\max}}{\beta_{\min}} \right)^{1-t} \sqrt{2 \log \frac{\beta_{\max}}{\beta_{\min}}}. \tag{67}$$

It is convenience to further define

<span id="page-22-2"></span>
$$\kappa_{t|s} := \int_{s}^{t} \sigma_{\tau}^{2} d\tau \stackrel{\text{geometric}}{=} \beta_{\text{max}}^{2} \cdot \left( \left( \frac{\beta_{\text{min}}}{\beta_{\text{max}}} \right)^{2s} - \left( \frac{\beta_{\text{min}}}{\beta_{\text{max}}} \right)^{2t} \right), \quad \bar{\beta}^{2} := \beta_{\text{max}}^{2} - \beta_{\text{min}}^{2}, \quad \gamma_{t} := \frac{\kappa_{t|0}}{\bar{\beta}^{2}}.$$
 (68)

With them, the conditional base distribution when f := 0 can be represented compactly by

$$p^{\text{base}}(X_t|X_0) = \mathcal{N}(X_t; X_0, \kappa_{t|0}I) \tag{69a}$$

$$p^{\text{base}}(X_t|X_0, X_1) = \mathcal{N}(X_t; (1 - \gamma_t)X_0 + \gamma_t X_1, \bar{\beta}^2 \gamma_t (1 - \gamma_t)I)$$
(69b)

• The constant noise schedule simply sets

<span id="page-22-1"></span>
$$\sigma_t \stackrel{\text{constant}}{:=} \sigma. \tag{70}$$

When f := 0, the base SDE is effectively a standard Brownian motion whose conditional distributions

$$p^{\text{base}}(X_t|X_0) = \mathcal{N}(X_t; X_0, \sigma^2 t I)$$
(71a)

$$p^{\text{base}}(X_t|X_0, X_1) = \mathcal{N}(X_t; (1-t)X_0 + tX_1, \sigma^2 t(1-t)I)$$
(71b)

Replay buffers  $\mathcal{B}_{\text{adj}}$  and  $\mathcal{B}_{\text{crt}}$  Similar to many previous diffusion samplers (Havens et al., 2025; Akhound-Sadegh et al., 2024; Chen et al., 2025), we employ replay buffers  $\mathcal{B}$  in computation of both adjoint (14) and corrector (15) matching objectives. Specifically, we rebase the expectation over model samples  $p^{u^{(k)}}$  onto a replay buffer  $\mathcal{B}$ , which stores the most latest  $|\mathcal{B}|$  samples. We update the buffer with N new samples every L gradient steps. Note that the use of replay buffers effectively render ASBS a hybrid method between on-policy and off-policy.

**Parametrization of**  $u_{\theta}$  and  $h_{\phi}$  For each energy function, we parametrize the drift  $u_{\theta}(t, x)$  and the corrector  $h_{\phi}(x)$  with two neural networks,  $v_{\theta}(t, x)$  and  $v_{\phi}(t, x)$ , of the same architecture.

Specifically, we parametrize the drift as  $u_{\theta}(t,x) := \sigma_t v_{\theta}(t,x)$ , which effectively eliminates the noise schedule " $\sigma_t$ " in matching target (see (14)), making it time-invariant for each sampled trajectory. The only exception is the conformer generation task, where we keep the original parametrization  $u_{\theta}(t,x) := v_{\theta}(t,x)$ , which empirically yields better results. On the other hand, since  $h_{\phi}(x)$  is independent of time, we simply set a fixed time input t = 1, i.e.,  $h_{\phi}(x) := v_{\phi}(1,x)$ .

The specific parametrization v(t,x) employed for each task are detailed below.

• MW-5: We consider v(t,x) a standard fully-connected network with 4 layers with 64 hidden features of the following form:

$$output = layer_n \circ \cdots \circ layer_1 \circ (x_embed(x) + t_embed(t))$$

- DW-4, LJ-13, LJ-55: We consider v(t,x) a Equivariant Graph Neural Network (EGNN; Satorras et al., 2021) with 5 layers and 128 hidden features. The architecture of EGNN is aligned with prior methods (Akhound-Sadegh et al., 2024; Havens et al., 2025).
- Alanine dipeptide: We use the same architecture as in MW-5, except with 8 layers with 256 hidden features.
- Conformer generation: We consider v(t, x) a similar EGNN used in Adjoint Sampling (Havens et al., 2025), except with 20 layers. Ablation study on the same EGNN architecture can be found in Section D.4.

Clipping  $\alpha_{\text{max}}$  We clip the energy gradient to prevent its maximum norm from exceeding  $\alpha_{\text{max}}$ .

Time scaling  $\lambda_t$  Following standard practices for AM objective, we employ a time scaling  $\lambda_t$  to improve numerical stability. Note that this does not affect the minimizer of the AM objective. We set  $\lambda_t := \frac{1}{\sigma_t^2}$  for all tasks.

**Translation invariance** For DW-4, LJ-13, LJ-55, and conformer generation tasks, we follow prior methods (Akhound-Sadegh et al., 2024; Havens et al., 2025) by restricting the state space to a zero center-of-mass (ZCOM) subspace and thereby enforcing translation invariance.

For a *n*-particle *k*-dimensional system, *i.e.*,  $x = [x_1; \dots; x_n]$  where  $x_i \in \mathbb{R}^k$ , the ZCOM subspace is defined as  $\mathcal{X}^{\text{ZCOM}} = \{x \in \mathbb{R}^{nk} : \sum_{i=1}^n x_i = 0\}$ . Practically, this is achieved by projecting the initial sample  $X_0 \sim \mu$ , the SDE's noise  $dW_t$ , and the energy gradient  $\nabla E(\cdot)$  onto  $\mathcal{X}^{\text{ZCOM}}$ . Note that the output of EGNN is by construction ZCOM.

Formally, the adaption is equivalent to augmenting the SDE with a projection matrix  $A \in \mathbb{R}^{nk \times nk}$ :

$$dX_t = \sigma_t A u_t(X_t) dt + \sigma_t A dW_t, \quad X_0 = AY_0, \quad Y_0 \sim \mu, \quad A = \left(I_n - \frac{1}{n} \mathbf{1}_n \mathbf{1}_n^\top\right) \otimes I_k, \tag{72}$$

where  $\otimes$  is the Kronecker product,  $I_n \in \mathbb{R}^{n \times n}$  is an identity matrix, and  $\mathbf{1}_n \in \mathbb{R}^n$  is a vector of ones.

Initialization and alternate procedure As ASBS is an instantiation of the IPF algorithm (see Theorem 3.2), it must adhere to the IPF initialization protocol to ensure theoretical convergence to the global solution. Specifically, the IPF initialization can be implemented in two ways

• Initialize with  $h_{\phi}^{(0)}:=0$  and run AM, CM, ... until convergence. We adopt this setup for all tasks.

• Initialize with  $u_{\theta}^{(0)} := 0$  and run CM, AM, ... until convergence. Since  $p^{u^{(0)}} = p^{\text{base}}$  in this setup, the optimal corrector at the first CM stage is known analytically:

<span id="page-24-2"></span>
$$h^{(1)}(x) \stackrel{\text{(15)}}{=} \int p_{0|1}^{\text{base}}(y|x) \nabla_x \log p_{1|0}^{\text{base}}(x|y) dy$$

$$= \int \frac{p_{0|1}^{\text{base}}(y|x)}{p_{1|0}^{\text{base}}(x|y)} \nabla_x p_{1|0}^{\text{base}}(x|y) dy$$

$$= \frac{1}{p_1^{\text{base}}(x)} \nabla_x \int p_0^{\text{base}}(y) p_{1|0}^{\text{base}}(x|y) dy$$

$$= \nabla \log p_1^{\text{base}}(x)$$
(73)

In practice, we find that the two setups yield similar performance.

**RDKit warm-start** This warm-starts the drift  $u_{\theta}$  using RDKit samples. The procedure is inspired by the fact that (Shi et al., 2023; Liu et al., 2023):

$$u_{t}^{\star} = \sigma_{t} \nabla \log \varphi_{t}$$

$$= \underset{u_{t}}{\operatorname{arg \, min}} \mathbb{E}_{p_{t,1}^{\star}} \left[ \|u_{t}(X_{t}) - \sigma_{t} \nabla_{x_{t}} \log p^{\operatorname{base}}(X_{1}|X_{t})\|^{2} \right]$$

$$= \underset{u_{t}}{\operatorname{arg \, min}} \mathbb{E}_{(X_{0},X_{1}) \sim p_{0,1}^{\star}, X_{t} \sim p^{\operatorname{base}}(\cdot|X_{0},X_{1})} \left[ \|u_{t}(X_{t}) - \sigma_{t} \nabla_{x_{t}} \log p^{\operatorname{base}}(X_{1}|X_{t})\|^{2} \right]. \tag{74}$$

where the last equality is due to

$$p_{0,t,1}^{\star}(x,y,z) \stackrel{\text{(41)}}{=} p_{t,1|0}^{\text{base}}(y,z|x)\hat{\varphi}_{0}(x)\varphi_{1}(z)$$

$$= p_{t|0,1}^{\text{base}}(y|x,z)p_{1|t}^{\text{base}}(z|y)\hat{\varphi}_{0}(x)\varphi_{1}(z) \qquad \text{by Markov property}$$

$$\stackrel{\text{(43)}}{=} p_{t|0,1}^{\text{base}}(y|x,z)p_{0,1}^{\star}(x,z). \qquad (75)$$

Equation (74) can be understood as an analogy of (45) for another SB potential  $\varphi_t$ . In practice, given RDKit samples  $X_1 \sim q^{\text{RDKit}}$ , we warm-start ASBS by minimizing w.r.t. the following objective:

$$\mathcal{L}_{\text{warmup}}(\theta) = \mathbb{E}_{t \sim \mathcal{U}[0,1], X_0 \sim \mu, X_1 \sim q^{\text{RDKit}}, X_t \sim p^{\text{base}}(\cdot | X_0, X_1)} \left[ \tilde{\lambda}_t \| u_t(X_t) - \sigma_t \nabla_{x_t} \log p^{\text{base}}(X_1 | X_t) \|^2 \right]$$

$$\stackrel{(69a)}{=} \mathbb{E}_{t \sim \mathcal{U}[0,1], X_0 \sim \mu, X_1 \sim q^{\text{RDKit}}, X_t \sim p^{\text{base}}(\cdot | X_0, X_1)} \left[ \tilde{\lambda}_t \| u_t(X_t) - \frac{\sigma_t}{\kappa_{1|t}} (X_1 - X_t) \|^2 \right],$$

$$(76)$$

where  $\kappa_{1|t}$  is defined in (68) for the geometric noise schedule. We set the time scaling  $\tilde{\lambda}_t := \sqrt{\frac{\sigma_t}{\kappa_{1|t}}}$ . Note that, unlike AS, the minimizer of (76) does not equal  $u^*$ , since  $(X_0, X_1) \sim \mu \otimes q^{\text{RDKit}} \neq p_{0,1}^*$  are sampled independently.

### <span id="page-24-0"></span>D Experiment Details

#### <span id="page-24-1"></span>D.1 Synthetic Energy Functions

#### D.1.1 Energy functions

In this section, we provide the exact setup for our synthetic energy experiments in Table 2. We consider four synthetic energy functions that have been widely used in recent literature to benchmark sampling and generative algorithms: MW-5, DW-4, LJ-13, and LJ-55.

**MW-5** The MW-5 (Many-Well in 5D) energy is a 5-particle 1D system adopted from Chen et al. (2025), where  $x = [x_1; \dots; x_5] \in \mathbb{R}^5$  with  $x_i \in \mathbb{R}$ , . The energy function is defined as follows:

<span id="page-24-3"></span>
$$E(x) = \sum_{i=1}^{5} (x_i^2 - \delta)^2$$
 (77)

**Table 5** Hyperparameters of ASBS for the each task.

<span id="page-25-0"></span>

|                    | S                      | ynthetic               | energy fu              | ınctions               | Alanine                | Conformer              |  |
|--------------------|------------------------|------------------------|------------------------|------------------------|------------------------|------------------------|--|
|                    | MW-5                   | 5 DW-4 LJ-13           |                        | LJ-55                  | dipeptide              | generation             |  |
| $\mu$              | $\mathcal{N}(0,1)$     | $\mu_{\rm harmon}$     | ic in (19)             | with $\alpha=2,2,1$    | $\mathcal{N}(0, 0.25)$ | $\mu_{\rm harmonic}$   |  |
| $\beta_{\min}$     | _                      | 0.001                  | 0.001                  | 0.001                  | 0.001                  | 0.001                  |  |
| $\beta_{\rm max}$  | _                      | 1                      | 1                      | 2                      | 0.5                    | 1                      |  |
| $\sigma$           | 0.2                    | _                      | _                      |                        | _                      |                        |  |
| K                  | 5                      | 20                     | 15                     | 15                     | 15                     | 3                      |  |
| $M_{\rm adj}$      | 100                    | 200                    | 300                    | 300                    | 4000                   | 2500                   |  |
| $M_{\rm crt}$      | 20                     | 20                     | 20                     | 20                     | 2000                   | 2000                   |  |
| N                  | 1000                   | 1000                   | 1000                   | 1000                   | 1000                   | 128                    |  |
| L                  | 200                    | 100                    | 100                    | 100                    | 100                    | 100                    |  |
| $ \mathcal{B} $    | $10^{4}$               | $10^{4}$               | $10^{4}$               | $10^{4}$               | $10^{4}$               | $6.4 \times 10^4$      |  |
| $\alpha_{\rm max}$ | _                      | 100                    | 100                    | 100                    | 100                    | 150                    |  |
| $\lambda_t$        | $\frac{1}{\sigma_t^2}$ | $\frac{1}{\sigma_t^2}$ | $\frac{1}{\sigma_t^2}$ | $\frac{1}{\sigma_t^2}$ | $\frac{1}{\sigma_t^2}$ | $\frac{1}{\sigma_t^2}$ |  |

where we set  $\delta = 4$ . This creates distinct modes centered at combinations of  $\pm \sqrt{\delta}$  in each of the d dimensions.

**DW-4** The DW-4 (Double-Well for 4 particles in 2D) energy is a physically motivated pairwise potential originally proposed in Köhler et al. (2020) and subsequently used in Akhound-Sadegh et al. (2024); Havens et al. (2025). It defines a system of four particles, each living in  $\mathbb{R}^2$ , leading to an 8D state vector  $x = [x_1; x_2; x_3; x_4] \in \mathbb{R}^8$  with  $x_i \in \mathbb{R}^2$ . The energy function reads

$$E(x) = \exp\left[\frac{1}{2\tau} \sum_{i < j} \left( a(d_{ij} - d_0) + b(d_{ij} - d_0)^2 + c(d_{ij} - d_0)^4 \right) \right], \tag{78}$$

where  $d_{ij} = ||x_i - x_j||_2$  is the Euclidean distance between particles i and j. We follow the standard configuration with a = 0, b = -4, c = 0.9,  $d_0 = 1$ , and temperature  $\tau = 1$ .

**LJ-13 and LJ-55** The Lennard-Jones (LJ) potentials are classical intermolecular potentials commonly used in physics to model atomic interactions. These are defined for a system of n particles in 3D space, with  $x = [x_1; \ldots; x_n] \in \mathbb{R}^{3n}$  and  $x_i \in \mathbb{R}^3$ . The index following "LJ-" indicates the number of particles (e.g., 13 or 55). The unnormalized energy function takes the form:

$$E(x) = \frac{\epsilon}{2\tau} \sum_{i < j} \left[ \left( \frac{r_m}{d_{ij}} \right)^6 - \left( \frac{r_m}{d_{ij}} \right)^{12} \right] + \frac{c}{2} \sum_i \|x_i - C(x)\|^2,$$
 (79)

where  $d_{ij} = ||x_i - x_j||_2$  is the pairwise distance and C(x) denotes the center of mass of the particles. We use the parameter values  $r_m = 1$ ,  $\epsilon = 1$ , c = 0.5, and  $\tau = 1$ , following prior work. The LJ-13 and LJ-55 systems correspond to 39D and 165D, respectively.

#### D.1.2 Baselines

Here, we outline the procedure used to obtain the values reported in Table 2 for the baseline methods.

For PIS (Zhang and Chen, 2022), DDS (Vargas et al., 2023), and LV-PIS (Richter and Berner, 2024), iDEM (Akhound-Sadegh et al., 2024), and AS (Havens et al., 2025), we reuse the values reported in AS (Havens et al., 2025, Table 1) for DW-4, LJ-13, and LJ-55 energy functions. As for MW-5, which is not included in AS, we run iDEM using their official implementation and the rest of baseline methods using our own implementation in PyTorch (Paszke et al., 2019). We were unable to obtain reportable results for LV-PIS and iDEM on this energy function.

For PDDS (Phillips et al., 2024) and SCLD (Chen et al., 2025), we run their official implementations in JAX (Bradbury et al., 2018) using the default hyperparameter settings specified for the Log-Gaussian Cox Process experiment in their respective papers. To enhance stability and convergence on synthetic energy functions, we tune the gradient clipping parameters. For PDDS, we apply clipping to the gradient of the energy function. For SCLD, we clip both the energy gradient and the Langevin norm. In both cases, the clipping magnitude is selected from the set  $\{1, 10, 100, 1000\}$  based on the best validation performance. Training is performed for 100,000 iterations across all runs. For SCLD, we use subtrajectory splitting with the default value of 4, so that it does not degenerate to CMCD (Vargas et al., 2024). In practice, we find that using subtrajectories yields better results.

#### D.1.3 Evaluation Metrics

In this subsection, we outline the evaluation criteria used to quantitatively assess the quality of samples generated from synthetic energy functions. We employ three primary metrics: Sinkhorn distance, geometric  $W_2$ , and energy  $W_2$ , each designed to capture different aspects of distributional similarity between generated and ground truth samples.

Sinkhorn distance To evaluate the similarity between the empirical distributions of generated and reference samples, we compute the Sinkhorn distance using the entropy-regularized optimal transport formulation (Peyré and Cuturi, 2019), following the implementation of Blessing et al. (2024) and Chen et al. (2025). The Sinkhorn regularization coefficient is set to  $10^{-3}$  throughout. We use 2,000 samples from both the generated and ground truth distributions to compute the metric.

**Geometric**  $W_2$  For DW and LJ tasks, the potential energy functions—and consequently, the sample distributions—exhibit invariance to both particle permutations and rigid transformations such as rotations and reflections. To appropriately account for these symmetries, we employ the geometric  $W_2$  distance as defined by Akhound-Sadegh et al. (2024) and Havens et al. (2025). Formally, the 2-Wasserstein distance is computed as:

<span id="page-26-1"></span>
$$W_2^2(\hat{\nu}, \nu) = \inf_{\pi \in \Pi(\hat{\nu}, \nu)} \int D(x, y)^2 \, \pi(x, y) \, dx dy, \tag{80}$$

where  $\Pi(\hat{\nu}, \nu)$  denotes the set of joint couplings with prescribed marginals  $\hat{\nu}$  (generated) and  $\nu$  (ground truth), and D(x, y) is a symmetry-aware distance between samples defined as:

$$D(x,y) = \min_{R \in O(s), P \in S(n)} \|x - (R \otimes P)y\|_{2}.$$
 (81)

Here, O(s) denotes the group of orthogonal transformations in s spatial dimensions (rotations and reflections), and S(n) represents the symmetric group over n particles. As exact minimization over these symmetry groups is computationally infeasible, we adopt the approximation scheme of Köhler et al. (2020). We use 2000 samples from each generated and ground truth distribution to compute the metric.

**Energy**  $W_2$  To evaluate fidelity with respect to the target energy landscape, we also compute the 2-Wasserstein distance between the energy values of generated samples and those of ground truth samples. For each target distribution, we generate 2,000 samples from both the model and the reference, and compare their respective energy histograms. This scalar-based Wasserstein metric serves as a proxy for how well the generative model captures the energy histogram of the target distribution.

#### <span id="page-26-0"></span>D.2 Alanine dipeptide

Benchmark description We adopt the experiment setup primarily from (Midgley et al., 2023). Given a configuration of alanine dipeptide, which consists of 22 particles in 3D, i.e.,  $x = [x_1; \dots; x_{22}] \in \mathbb{R}^{66}$  where  $x_i \in \mathbb{R}^3$ , we apply the same coordinate transform  $\mathcal{T}$  proposed by Midgley et al. (2023). This coordinate transform maps the Cartesian coordinates to internal coordinates,  $\mathcal{T}(x) =: z \in \mathbb{R}^{60}$ , which include bond lengths, bond angles, and dihedral angles (Stimper et al., 2022). This process effectively removes six degrees of freedom—three for translation and three for rotation—thereby enforcing structural invariance. Non-angular coordinates are further normalized using samples with minimal energies. We refer readers to (Midgley et al.,

2023, Appendix F.1) for further details. Note that the internal coordinate transformation is bijective. Hence, we can compute the energy via

$$E(x) = E(\mathcal{T}^{-1}(z)) \tag{82}$$

For each sample  $x = \mathcal{T}^{-1}(z) \in \mathbb{R}^{66}$ , we extract five torsion angles, including the backbone angles  $\phi$ ,  $\psi$  and methyl rotation angles  $\gamma_1$ ,  $\gamma_2$ ,  $\gamma_3$ . We report two divergence metrics with respect to the ground-truth distribution, which contains 10<sup>7</sup> samples simulated by Molecular Dynamics. We implement the baseline methods, including PIS (Zhang and Chen, 2022), DDS (Vargas et al., 2023), AS (Havens et al., 2025), using PvTorch (Paszke et al., 2019).

For the KL divergences, we adopt setup from (Wu et al., 2020) and compute the divergence of the ground-truth marginal to model marginal for each torsion angle:

$$D_{\mathrm{KL}}(p^{\star}(\cdot)||p^{u_{\theta}}(\cdot)) \approx \sum P^{\star}(\cdot)\log\frac{P^{\star}(\cdot) + \epsilon}{P^{u_{\theta}}(\cdot) + \epsilon}, \qquad \epsilon = 10^{-5},$$
(83)

where  $P^*$  and  $P^{u_{\theta}}$  are histograms of 10<sup>7</sup> samples, discretized between  $[-\pi, \pi]$  with 200 intervals.

For the Wasserstein-2 distance, we use the Geometric  $W_2$  in (80), where each sample is now in 2D, x = $[\phi,\psi] \in \mathbb{R}^2$ . Due to the high computational cost, we compute the value using a subset of  $10^4$  samples from the test set ground-truth samples, which is fixed for all methods.

<span id="page-27-0"></span>Finally, both Ramachandran plots in Figure 5 are generated using 10<sup>7</sup> samples.

### Amortized conformer generation

In this subsection, we provide some context for the experimental results found in Table 4 regarding the generation of conformers.

Benchmark description Conformers are atomic representations of molecules in cartesian space with their constituent atoms arranged into local minima on the potential energy surface. Molecules are defined to be a graph of atoms (nodes) connected by bonds (edges); conformers are geometric realizations of that molecule. Torsion angles, or rotatable bonds, are particularly important degrees of freedom for defining conformations since bond lengths and bond angles are typically much more stable due to a high sensitivity to perturbations. It is common to consider bond lengths and bond angles fixed, while the torsional degrees of freedom define the conformer.

The task in this benchmark is to take a representation of the molecular graph, usually a SMILES string (Weininger, 1988), and comprehensively sample the conformational configuration space. In flexible molecules, there can be a large number of conformers with many separated modes in a 3n-6 dimensional space. (Where n represents the number of atoms and 6 comes from the irrelevance of rotation and translation of the conformer.) We quantify the notion of comprehensively sampling the space by comparing generated structures to a set of conformers sampled using expensive, standard search techniques (Pracht et al., 2024) that were further relaxed using extremely precise density function theory-based, quantum chemistry methods (Neese, 2012; Levine et al., 2025). A detailed description of this benchmark can be found in its source (Havens et al., 2025, Appendix F.).

**Evaluation and baselines** The method of comparison between proposed structure and reference conformer is to use RDKit's (Landrum, 2006) implementation of Root Mean Squared Displacement (RMSD), a measure of distance between atomic structures that is invariant to translation and rotation. We set a threshold RMSD for two structures to match and computed the Recall Coverage and Recall Average Minimum RMSD (AMR). The experiment was performed with both generated structures and with generated structures after a so-called relaxation, i.e. geometry optimization of energy, using eSEN (Fu et al., 2025). The equations for computing these metrics are:

$$COV-R(\delta) := \frac{1}{L} |\{l \in \{1, \dots, L\} : \exists k \in \{1, \dots, K\}, \quad RMSD(C_k, C_l^*) < \delta\}|$$
(84)

COV-R(
$$\delta$$
) :=  $\frac{1}{L} |\{l \in \{1, ..., L\} : \exists k \in \{1, ..., K\}, \quad \text{RMSD}(C_k, C_l^*) < \delta\}|$  (84)  
AMR-R :=  $\frac{1}{L} \sum_{l \in \{1, ..., L\}} \min_{k \in \{1, ..., K\}} \text{RMSD}(C_k, C_l^*)$  (85)

<span id="page-28-1"></span>**Table 6** Ablation study on amortized conformer generation using the same EGNN architecture as in AS (Havens et al., 2025). We report the recall at the thresholds **1.0Å** and **1.25Å**, where the latter was reported in AS.

|                 |                                                             | without relaxation              |                               |                               |                               | with relaxation                 |                               |                                 |                               |
|-----------------|-------------------------------------------------------------|---------------------------------|-------------------------------|-------------------------------|-------------------------------|---------------------------------|-------------------------------|---------------------------------|-------------------------------|
|                 |                                                             | SPICE                           |                               | GEOM-DRUGS                    |                               | SPICE                           |                               | GEOM-DRUGS                      |                               |
|                 | Method                                                      | Coverage ↑                      | $\mathrm{AMR}\downarrow$      | Coverage ↑                    | AMR ↓                         | Coverage ↑                      | AMR ↓                         | Coverage ↑                      | AMR ↓                         |
| Threshold 1.0Å  | RDKit ETKDG (Riniker and Landrum, 2015)                     | $56.94{\scriptstyle\pm35.82}$   | $1.04{\scriptstyle \pm 0.52}$ | $50.81{\scriptstyle\pm34.69}$ | $1.15{\scriptstyle \pm 0.61}$ | $70.21{\scriptstyle\pm31.70}$   | $0.79 \scriptstyle{\pm 0.44}$ | $62.55{\scriptstyle\pm31.67}$   | $0.93 \scriptstyle{\pm 0.53}$ |
|                 | AS (Havens et al., 2025)                                    | $56.75{\scriptstyle\pm38.15}$   | $0.96 \scriptstyle{\pm 0.26}$ | $36.23{\scriptstyle\pm33.42}$ | $1.20{\scriptstyle \pm 0.43}$ | $82.41{\scriptstyle\pm25.85}$   | $0.68 \scriptstyle{\pm 0.28}$ | $64.26{\scriptstyle\pm34.57}$   | $0.89 \scriptstyle{\pm 0.45}$ |
|                 | ASBS w/ Gaussian prior (Ours)                               | $68.61{\scriptstyle\pm33.48}$   | $0.88 \scriptstyle{\pm 0.25}$ | $46.03{\scriptstyle\pm35.99}$ | ightharpoonup 1.08 $\pm$ 0.36 | $84.77{\scriptstyle\pm22.65}$   | $0.64 \scriptstyle{\pm 0.25}$ | $68.83{\scriptstyle\pm31.53}$   | $0.80 \pm 0.37$               |
|                 | ${\rm ASBS}\ w/\ {\rm Harmonic\ prior\ }(\textbf{Ours})$    | $70.70{\scriptstyle\pm33.21}$   | $0.86 \scriptstyle{\pm 0.24}$ | $52.19{\scriptstyle\pm35.93}$ | $1.05{\scriptstyle\pm0.41}$   | $86.79 \scriptstyle{\pm 22.86}$ | $0.61 \scriptstyle{\pm 0.24}$ | $70.08{\scriptstyle\pm31.60}$   | $0.80 \pm 0.37$               |
|                 | AS +RDKit warmup (Havens et al., 2025)                      | $72.21{\scriptstyle\pm30.22}$   | $0.84 \scriptstyle{\pm 0.24}$ | $52.19{\scriptstyle\pm35.20}$ | 1.02±0.34                     | 87.84±19.20                     | $0.60 \scriptstyle{\pm 0.23}$ | $73.88{\scriptstyle\pm28.63}$   | 0.76±0.34                     |
|                 | ${\rm ASBS} + {\rm RDKit} \ {\rm warmup} \ (\textbf{Ours})$ | $74.29{\scriptstyle\pm31.25}$   | $0.82 \scriptstyle{\pm 0.24}$ | $55.88{\scriptstyle\pm36.51}$ | $0.98 \scriptstyle{\pm 0.34}$ | $87.25{\scriptstyle\pm20.77}$   | $0.60{\scriptstyle \pm 0.24}$ | $74.11{\scriptstyle\pm30.16}$   | $0.75{\scriptstyle \pm 0.34}$ |
| Threshold 1.25Å | RDKit ETKDG (Riniker and Landrum, 2015)                     | $72.74{\scriptstyle\pm33.18}$   | $1.04{\scriptstyle \pm 0.52}$ | $63.51{\scriptstyle\pm34.74}$ | 1.15±0.61                     | 81.61±27.58                     | $0.79 \scriptstyle{\pm 0.44}$ | $71.72{\scriptstyle\pm29.73}$   | $0.93 \scriptstyle{\pm 0.53}$ |
|                 | AS (Havens et al., 2025)                                    | $82.22{\scriptstyle\pm25.72}$   | $0.96 \scriptstyle{\pm 0.26}$ | $60.93{\scriptstyle\pm35.15}$ | $1.20{\scriptstyle \pm 0.43}$ | $94.10{\scriptstyle\pm15.67}$   | $0.68 \scriptstyle{\pm 0.28}$ | $79.08{\scriptstyle\pm29.44}$   | $0.89 \scriptstyle{\pm 0.45}$ |
|                 | ASBS w/ Gaussian prior (Ours)                               | $87.20{\scriptstyle \pm 21.88}$ | $0.88 \scriptstyle{\pm 0.25}$ | $70.86{\scriptstyle\pm31.98}$ | ighthalpoonup 1.08 $\pm$ 0.36 | $95.19{\scriptstyle\pm10.29}$   | $0.64 \scriptstyle{\pm 0.25}$ | $84.66{\scriptstyle\pm25.03}$   | $0.80 \pm 0.37$               |
|                 | ${\rm ASBS}\ w/\ {\rm Harmonic\ prior\ }(\textbf{Ours})$    | $89.66{\scriptstyle\pm19.42}$   | $0.86 \scriptstyle{\pm 0.24}$ | $74.50{\scriptstyle\pm32.32}$ | $1.05{\scriptstyle\pm0.41}$   | $96.64 \scriptstyle{\pm 10.15}$ | $0.61 \scriptstyle{\pm 0.24}$ | $83.76 \scriptstyle{\pm 24.77}$ | $0.80 \pm 0.37$               |
|                 | AS +RDKit warmup (Havens et al., 2025)                      | 89.42±17.48                     | $0.84$ $_{\pm0.24}$           | $72.98{\scriptstyle\pm30.82}$ | 1.02±0.34                     | $96.65{\scriptstyle\pm7.51}$    | 0.60±0.23                     | 87.01±22.79                     | 0.76±0.34                     |
|                 | $ASBS + RDKit \ warmup \ (\textbf{Ours})$                   | $90.85{\scriptstyle\pm17.74}$   | $0.82 \scriptstyle{\pm 0.24}$ | $77.86{\scriptstyle\pm30.37}$ | $0.98 \scriptstyle{\pm 0.34}$ | $97.28{\scriptstyle\pm6.55}$    | $0.60{\scriptstyle \pm 0.24}$ | $87.81{\scriptstyle\pm22.75}$   | $0.75{\scriptstyle \pm 0.34}$ |

<span id="page-28-2"></span>![](_page_28_Figure_2.jpeg)

Figure 8 Ablation study on full recall coverage curves (without RDKit warm-start) using the same EGNN architecture as in AS (Havens et al., 2025). Note that Table 6 reports the values at the thresholds 1.0Å and 1.25Å.

where  $\delta = 0.75$  Å is the coverage threshold,  $L = \max(L', 128)$ , where L' is the number of reference conformers, K = 2L, and let  $\{C_l^*\}_{l \in [1,L]}$  and  $\{C_k\}_{k \in [1,K]}$  be the sets of ground truth and generated conformers respectively. We capped the reference conformers per molecule at 512 in COV-R.

<span id="page-28-0"></span>The values for the baselines are adopted from AS (Havens et al., 2025).

#### D.4 Additional Experiments and Discussions

Ablation study between AS and ASBS using the same EGNN For the amortized conformer generation task in Table 4, we use an EGNN architecture with 20 layers, whereas AS employs the same architecture with 12 layers. In Table 6, we report the results of ASBS using the same 12-layer EGNN as AS. Notably, our ASBS consistently outperforms AS on all metrics across all setups, except the coverage for GEOM-DRUGS with relaxation and RDKit warm-start, where ASBS falls slightly behind AS by only 1.0%. Finally, Figure 8 reports the full recall coverage curves that reproduce Table 4.

Ability of ASBS in finding modes We conduct additional experiments on the 40-mode GMM in 2D. Specifically, we instantiate ASBS with a uni-variance Gaussian source distribution centered at zero, effectively assuming no prior knowledge of the target modes, as the initial distribution does not coincide with any target modes. We also run a vanilla Langevin baseline for 1 million steps, starting from the same source distribution.

Figure 9 represents the quantitative results. Notably, ASBS is able to identify almost all modes. In contrast, the vanilla Langevin baseline appears to suffer from a slow mixing rate, recovering less than half of the total

<span id="page-29-0"></span>![](_page_29_Figure_0.jpeg)

Figure 9 Compared to vanilla Langevin baseline, our ASBS—instantiated with a standard uni-variance Gaussian—is able to identify almost all modes without any prior knowledge of where the target modes were located.

modes even after 1 million steps. We highlight this distinction as an advantage of constructing diffusion samplers from the stochastic control and Schrödinger Bridge frameworks, which allows theoretical convergence to target distribution within a finite horizon. Finally, we believe that with proper tuning of the ASBS noise schedule, its performance can be further enhanced.

**Discussion on important weights** Finally, we discuss the potential integration of ASBS with importance weights, emphasizing that our theoretical and algorithmic frameworks do not preclude the use of importance weights to further enhance performance or robustness.

Formally, the importance weights over model path  $X \sim p^u$  admit the following representation:

<span id="page-29-1"></span>
$$w(X) := \frac{\mathrm{d}p^{\star}(X)}{\mathrm{d}p^{u}(X)} = \exp\left(\int_{0}^{1} -\frac{1}{2}||u_{t}(X_{t})||^{2} \mathrm{d}t - \int_{0}^{1} u_{t}(X_{t}) \cdot \mathrm{d}W_{t} - \log\frac{\hat{\varphi}_{1}(X_{1})}{\nu(X_{1})} + \log\frac{\hat{\varphi}_{0}(X_{0})}{\mu(X_{0})}\right), \tag{86}$$

which can be obtained from (59) by setting  $\bar{h} := \hat{\varphi}_1$  so that  $q^{\bar{h}} = p^*$  is the optimal distribution of SB.

Note that when the source distribution degenerates to the Dirac delta  $\mu(X_0) = \delta_0(X_0)$ , the last term  $\log \frac{\hat{\varphi}_0(X_0)}{\mu(X_0)}$  becomes a constant and—as discussed in Section 3.2— $\hat{\varphi}_1 = p_1^{\text{base}}$ , thereby recovering the weights used in prior SOC-based methods (Zhang and Chen, 2022; Havens et al., 2025).

Equation (86) is also a more concise representation than the one derived in (Richter and Berner, 2024), by recognizing the following relation through the application of Ito Lemma (46) to  $\log \hat{\varphi}_t(X_t)$ :

$$\frac{\log \hat{\varphi}_1(X_1)}{\log \hat{\varphi}_0(X_0)} = \int_0^1 \left[ \frac{1}{2} \|v_t(X_t)\|^2 + (u_t \cdot v_t)(X_t) + \nabla \cdot (\sigma_t v_t(X_t) - f_t(X_t)) \right] dt + \int_0^1 v_t(X_t) \cdot dW_t, \quad (87)$$

where we shorthand  $v_t(x) := \sigma_t \nabla \log \hat{\varphi}_t(x)$ .

Estimating the weight in (86) requires knowing the ratios  $\frac{\hat{\varphi}_1(x)}{\nu(x)}$  and  $\frac{\hat{\varphi}_0(x)}{\mu(x)}$ , which are not immediately available with the current parametrization,  $u_{\theta}(t,x) \approx \sigma_t \nabla \log \varphi_t(x)$  and  $h_{\phi}(x) \approx \nabla \log \hat{\varphi}_1(x)$ . One accommodation is to reparametrize the functions with potential network  $v(t,x): [0,1] \times \mathcal{X} \to \mathbb{R}$ ,

$$u_{\theta}(t,x) := \sigma_t \nabla v_{\theta}(t,x), \qquad h_{\phi}(x) := \nabla v_{\phi}(1,x) \tag{88}$$

and then regress their gradients onto the adjoint and corrector targets. With that, the logarithmic ratios can be easily estimated:

$$\log \frac{\hat{\varphi}_1(x)}{\nu(x)} = v_{\phi}(1, x) + E(x), \qquad \log \frac{\hat{\varphi}_0(x)}{\mu(x)} \stackrel{(42)}{=} -\log \varphi_0(x) = v_{\theta}(0, x). \tag{89}$$

A more detailed investigation of this importance sampling scheme is left for future work.