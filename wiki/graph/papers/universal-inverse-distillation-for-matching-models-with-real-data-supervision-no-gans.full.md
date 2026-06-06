---
type: paper-fulltext
slug: universal-inverse-distillation-for-matching-models-with-real-data-supervision-no-gans
arxiv_id: null
source_path: /Users/saris.kia.adm/.paper-scholar/universal-inverse-distillation-for-matching-models-with-real-data-supervision-no-gans/universal_inverse_distillation.md
paper: "[[universal-inverse-distillation-for-matching-models-with-real-data-supervision-no-gans]]"
---
# UNIVERSAL INVERSE DISTILLATION FOR MATCHING MODELS WITH REAL-DATA SUPERVISION (NO GANS)

### Nikita Kornilov<sup>∗</sup>

Applied AI Institute, Moscow, Russia MIRAI, Moscow, Russia BRAIn Lab, Moscow, Russia jhomanik14@gmail.com

### Tikhon Mavrin<sup>∗</sup>

Applied AI Institute, Moscow, Russia tixonmavrin@gmail.com

### Nikita Gushchin

Applied AI Institute, Moscow, Russia AXXX, Moscow, Russia

### Iaroslav Koshelev

AI Foundation lab, Moscow, Russia

### David Li<sup>∗</sup>

AI Foundation lab, Moscow, Russia MBZUAI, Abu Dhabi, UAE David.Li@mbzuai.ac.ae

### Aleksei Leonov

AI Foundation lab, Moscow, Russia MIRAI, Moscow, Russia

### Evgeny Burnaev

Applied AI Institute, Moscow, Russia AXXX, Moscow, Russia

### Alexander Korotin

Applied AI Institute, Moscow, Russia AXXX, Moscow, Russia iamalexkorotin@gmail.com

# ABSTRACT

While achieving exceptional generative quality, modern diffusion, flow, and other matching models suffer from slow inference, as they require many steps of iterative generation. Recent distillation methods address this problem by training efficient one-step generators under the guidance of a pre-trained teacher model. However, these methods are often constrained to only one specific framework, e.g., only to diffusion or only to flow models. Furthermore, these methods are originally datafree, and to benefit from the usage of real data, it is required to use an additional complex adversarial training with an extra discriminator model. In this paper, we present RealUID, a universal distillation framework for all matching models that seamlessly incorporates real data into the distillation procedure without GANs. Our RealUID approach offers a simple theoretical foundation that covers previous distillation methods for Flow Matching and Diffusion models, and can be also extended to their modifications, such as Bridge Matching and Stochastic Interpolants. The code can be found in <https://github.com/David-cripto/RealUID>.

# <span id="page-0-0"></span>1 INTRODUCTION

In generative modeling, the goal is to learn to sample from complex data distributions (e.g., images), and two powerful paradigms for it are the diffusion models (DM) and the flow matching (FM) models. While they share common principles and are even equivalent under certain conditions [\(Holderrieth et al.,](#page-10-0) [2024;](#page-10-0) [Gao et al.,](#page-10-1) [2025\)](#page-10-1), they are typically studied separately. Diffusion models [\(Sohl-Dickstein et al.,](#page-12-0) [2015;](#page-12-0) [Ho et al.,](#page-10-2) [2020;](#page-10-2) [Song et al.,](#page-12-1) [2021\)](#page-12-1) transform data into noise through a forward process and then learn a reverse-time stochastic differential equation (SDE) to recover the data distribution. Training minimizes score-matching objectives, yielding unbiased estimates of intermediate scores. Sampling requires simulating the reverse dynamics, which is computationally heavy but delivers high-quality and diverse results. Flow Matching [\(Lipman et al.,](#page-11-0) [2023;](#page-11-0) [Liu,](#page-11-1) [2022\)](#page-11-1)

<sup>\*</sup>Equal contribution

instead interpolates between source and target distributions by learning the vector field of an ordinary differential equation (ODE). The field is estimated through unbiased conditional objectives, but the resulting ODE often has curved trajectories, making sampling costly due to expensive integration. Beyond these, **Bridge Matching** (Peluchetti, 2023; Liu et al., 2022) and **Stochastic Interpolants** (Albergo et al., 2023) generalize the framework and naturally support *data couplings*, which are crucial for data-to-data translation. Since all of the above optimize *conditional matching* objectives to recover an ODE/SDE for generation, we refer to them collectively as *matching models*.

Despite their success, matching models share a major drawback: sampling is slow, as generation requires integrating many steps of an SDE or ODE. To address this problem, a range of distillation techniques have been proposed to compress multi-step dynamics into efficient one-step or few-step generators. Although matching models follow a similar mathematical framework, many distillation works consider only one particular framework, e.g., only diffusion models (Zhou et al., 2024a;b), Flow Matching (Huang et al., 2024), or Bridge Matching (Gushchin et al., 2025). Furthermore, these distillation methods are data-free by construction and cannot benefit from the utilization of real data without using additional GAN-based losses. *Thus, the following issues remain:* 

- 1. Similar distillation techniques developed separately for similar matching models frameworks.
- 2. Absence of a natural way to incorporate real data in distillation procedures (without GANs).

**Contributions.** In this paper, we address these issues and present the following **main contributions**:

- 1. We present the *Universal Inverse Distillation with real data* (*RealUID*) framework for matching models, including diffusion and flow matching models (§3) as well as Bridge Matching and Stochastic Interpolants (Appendix C). It unifies previously introduced Flow Generator Matching (FGM), Score Identity Distillation (SiD) and Inverse Bridge Matching Distillation (IBMD) methods (§3.2) for flow, score and bridge matching models respectively, provides simple yet rigorous theoretical explanations based on a linearization technique, and reveals the connections between these methods and inverse optimization (§3.3).
- 2. Our RealUID introduces a novel and natural way to incorporate real data directly into the distillation loss, eliminating the need for extra adversarial losses which require additional discriminator networks used in GANs from the previous works (§3.4).

### <span id="page-1-2"></span>2 BACKGROUNDS ON TRAINING AND DISTILLING MATCHING MODELS

Here, we describe diffusion, flow (§2.1) and other types of matching models (§2.2) along with the distillation methods for them (§2.3). Then, we discuss how GANs can be used to add real data into the distillation procedure (§2.4).

**Preliminaries.** We work on the D-dimensional Euclidean space  $\mathbb{R}^D$ . This space is equipped with the standard scalar product  $\langle x,y\rangle=\sum_{d=1}^D x_dy_d$ , the  $\ell_2$ -norm  $\|x\|=\sqrt{\langle x,x\rangle}$  and the  $\ell_2$ -distance  $\|x-y\|, \forall x,y\in\mathbb{R}^D$ . We consider probability distributions from the set  $\mathcal{P}(\mathbb{R}^D)$  of absolutely continuous distributions with finite variance and support on the whole  $\mathbb{R}^D$ .

### <span id="page-1-0"></span>2.1 DIFFUSION AND FLOW MODELS

**Diffusion models** (Sohl-Dickstein et al., 2015; Ho et al., 2020; Song et al., 2021) consider a forward noising process  $p_t$  that gradually transforms clean data  $p_0$  into a noise  $p_T$  on the time interval [0, T]:

<span id="page-1-1"></span>
$$dx_t = f_t \cdot x_t \cdot dt + g_t \cdot d\mathbf{w}_t, \quad x_0 \sim p_0,$$

where  $f_t$  and  $g_t$  are time-dependent scalars and  $\mathbf{w}_t$  is a standard Wiener process. This process defines a conditional distributions  $p_t(x_t|x_0) = \mathcal{N}(\alpha_t x_0|\sigma_t^2 \mathbf{I})$ , where

$$\alpha_t = \exp\left(\int_0^t f_s \, ds\right), \quad \sigma_t = \left(\int_0^t g_s^2 \exp\left(-2\int_0^s f_u \, du\right) ds\right)^{1/2}.$$

Each conditional distribution admits a conditional score function, describing it:

$$s_t(x_t|x_0) := \nabla_{x_t} \log p_t(x_t|x_0) = -(x_t - \alpha_t x_0)/\sigma_t^2$$
.

The reverse dynamics from the noise distribution  $p_T$  to the data distribution  $p_0$  is provided by the following *reverse-time* SDE with a reverse-time Wiener process  $\bar{\mathbf{w}}_t$ :

$$dx_t = (f_t \cdot x_t - g_t^2 \cdot s_t(x_t))dt + g_t d\bar{\mathbf{w}}_t,$$

where  $s_t(x_t) = \mathbb{E}_{x_0 \sim p_0(\cdot|x_t)}[s_t(x_t|x_0)]$  is the score function of  $p_t(x_t) = \int p(x_t|x_0)p(x_0)dx_0$ . This unconditional score function is learned via minimizing the denoising score matching (DSM) loss:

$$\mathcal{L}_{\text{DSM}}(s', p_0) = \mathbb{E}_{t \sim [0, T], x_0 \sim p_0, x_t \sim p_t(\cdot | x_0)} \left[ \gamma_t \| s'_t(x_t) - s_t(x_t | x_0) \|_2^2 \right], \tag{1}$$

where  $\gamma_t$  are some positive weights. The reverse dynamics admits a probability flow ODE (PF-ODE):

$$dx_t = u_t(x_t)dt$$
,  $u_t(x_t) := (f_t \cdot x_t - g_t^2 \cdot s_t(x_t)/2)$ ,

which provides faster inference than the SDE formulation.

Flow Matching framework (Lipman et al., 2023; Liu et al., 2023) constructs the flow directly by learning the drift  $u_t(x_t)$ . Specifically, for each data point  $x_0 \sim p_0$ , one defines a conditional flow  $p_t(x_t|x_0)$  with the corresponding conditional vector field  $u_t(x_t|x_0)$  generating it via ODE:

$$dx_t = u_t(x_t|x_0)dt.$$

Then, to construct the flow between the data  $p_0$  and noise  $p_T$ , one needs to compute the unconditional vector field  $u_t(x_t) = \mathbb{E}_{x_0 \sim p_0(\cdot|x_t)}[u_t(x_t|x_0)]$  which generates the flow  $p_t(x_t) = \int p(x_t|x_0)p(x_0)dx_0$ . It can be done by minimizing the following Conditional Flow Matching (CFM) loss:

<span id="page-2-2"></span>
$$\mathcal{L}_{CFM}(v, p_0) = \mathbb{E}_{t \sim [0, T], x_0 \sim p_0, x_t \sim p_t(\cdot | x_0)} \left[ \|v_t(x_t) - u_t(x_t | x_0)\|_2^2 \right]. \tag{2}$$

In practice, the most popular choice is the Gaussian conditional flows  $p_t(x_t|x_0) = \mathcal{N}(\alpha_t x_0, \sigma_t^2 \mathbf{I})$ . For this conditional flow samples can be obtained as  $x_t = \alpha_t x_0 + \sigma_t \epsilon$ ,  $\epsilon \sim \mathcal{N}(0, \mathbf{I})$  and the conditional drift can be calculated as  $u_t(x_t|x_0) = \dot{\alpha}_t x_0 + \dot{\sigma}_t \epsilon$ .

We recall <u>data-to-data</u> models working with data couplings, such as **Bridge Matching** and **Stochastic Interpolants**, in Appendices C.1 and C.2, respectively.

### <span id="page-2-0"></span>2.2 Universal loss for matching models

From a mathematical point of view, it was shown in (Holderrieth et al., 2024; Gao et al., 2025) that flow and diffusion models basically share the same loss structure. We recall this structure, but use our own notation. We call diffusion and flow models and their extensions as *matching models*.

A matching model constructs a probability path  $p_t$  on the time interval [0,T], transforming the desired data  $p_0 \in \mathcal{P}(\mathbb{R}^D)$  to the noise  $p_T \in \mathcal{P}(\mathbb{R}^D)$ . This path is built as a mixture of simple conditional paths  $p_t(\cdot|x_0)$  conditioned on samples  $x_0 \sim p_0$ , i.e.,  $p_t(x_t) = \int_{\mathbb{R}^D} p_t(x_t|x_0)p_0(x_0)dx_0, \forall x_t \in \mathbb{R}^D$ . The path  $p_t$  determines the function  $f^{p_0}: [0,T] \times \mathbb{R}^D \to \mathbb{R}^D$  which recovers it (e.g., score function or drift). The conditional paths also determine their own simple conditional functions  $f^{p_0}(\cdot|x_0)$  so that they express  $f_t^{p_0}(x_t) = \mathbb{E}_{x_0 \sim p_0(\cdot|x_t)}[f_t^{p_0}(x_t|x_0)]$ , where  $p_0(\cdot|x_t)$  is a data distribution  $p_0$  conditioned on sample  $x_t$  at time t. Since the function  $f^{p_0}$  cannot be computed directly, it is approximated by a trainable function  $f:[0,T] \times \mathbb{R}^D \to \mathbb{R}^D$  at each time t from [0,T] and point  $x_t \sim p_t$  using known analytically conditional functions:

$$\|f_t(x_t) - f_t^{p_0}(x_t)\|^2 = \|f_t(x_t) - \mathbb{E}_{x_0 \sim p_0(\cdot|x_t)}[f_t^{p_0}(x_t|x_0)]\|^2 \propto \mathbb{E}_{x_0 \sim p_0(\cdot|x_t)}[\|f_t(x_t) - f_t^{p_0}(x_t|x_0)\|^2].$$

We also change the sampling order  $x_t \sim p_t, x_0 \sim p_0(\cdot|x_t)$  to more natural  $x_0 \sim p_0, x_t \sim p_t(\cdot|x_0)$ .

<span id="page-2-3"></span>**Definition 1.** We define Universal Matching (UM) loss  $\mathcal{L}_{UM}(f, p_0)$  that takes trainable function f and distribution  $p_0 \in \mathcal{P}(\mathbb{R}^D)$  as arguments and upon minimization over f returns the function  $f^{p_0}$ :

<span id="page-2-4"></span>
$$\mathcal{L}_{UM}(f, p_0) := \mathbb{E}_{t \sim [0, T]} \mathbb{E}_{x_0 \sim p_0, x_t \sim p_t(\cdot | x_0)} [\|f_t(x_t) - f_t^{p_0}(x_t | x_0)\|^2], f^{p_0} := \arg\min_{f} \mathcal{L}_{UM}(f, p_0).$$
(3)

The notation  $t \sim [0,T]$  hides time sampling and loss weighting inherent to the given matching model.

### <span id="page-2-1"></span>2.3 DISTILLATION OF MATCHING-BASED MODELS

To solve the long inference problem of matching models, a line of distillation approaches sharing similar principles was introduced: **Score Identity Distillation** (Zhou et al., 2024b;a, **SiD**), **Flow Generator Matching** (Huang et al., 2024, **FGM**), and **Inverse Bridge Matching Distillation** (Gushchin et al., 2025, **IBMD**), for diffusion, flow, and bridge matching models, respectively.

The **SiD** approach trains a *student generator*  $G_{\theta}: \mathcal{Z} \to \mathbb{R}^D$  (parameterized by  $\theta$ ) that produces a distribution  $p_0^{\theta}$  from a latent distribution  $p^{\mathcal{Z}}$  on  $\mathcal{Z}$ . This approach minimizes the squared  $\ell_2$ -distance

between the known teacher score function  $s^* := \arg\min_{s'} \mathcal{L}_{DSM}(s', p_0^*)$  on real data  $p_0^*$  and the unknown student score function  $s^{\theta}$ :

<span id="page-3-2"></span>
$$\min_{\theta} \mathbb{E}_{t \sim [0,T]} \mathbb{E}_{x_t^{\theta} \sim p_t^{\theta}} [\|s_t^{\theta}(x_t^{\theta}) - s_t^*(x_t^{\theta})\|^2], \quad \text{s.t. } s^{\theta} = \arg\min_{s'} \mathcal{L}_{\text{DSM}}(s', p_0^{\theta}), \tag{4}$$

where  $p_t^{\theta}$  is the forward noising process for the generated data  $p_0^{\theta}$ . The authors propose the tractable loss with parameter  $\alpha_{\text{SiD}}$  to approximate the real gradients of (4):

<span id="page-3-3"></span>
$$\mathcal{L}_{SiD}(\theta) := \mathbb{E}_{t \sim [0,T]} \mathbb{E}_{z \sim p^{\mathcal{Z}}, x_{0}^{\theta} = G_{\theta}(z), x_{t}^{\theta} \sim p_{t}^{\theta}} [-2\omega_{t} \cdot \alpha_{SiD} \| s_{t}^{*}(x_{t}^{\theta}) - s_{t}^{sg[\theta]}(x_{t}^{\theta}) \|^{2}$$

$$+ 2\omega_{t} \langle s_{t}^{*}(x_{t}^{\theta}) - s_{t}^{sg[\theta]}(x_{t}^{\theta}), s_{t}^{*}(x_{t}^{\theta}) - s_{t}^{\theta}(x_{t}^{\theta} | x_{0}^{\theta}) \rangle], s^{\theta} = \arg\min_{s'} \mathcal{L}_{DSM}(s', p_{0}^{\theta}), (5)$$

where  $w_t$  are normalizing weights and gradients w.r.t.  $\theta$  are not calculated for the variables under stop-gradient  $sg[\cdot]$  operator. The SiD pipeline is two alternating steps: first, refine the *fake score*  $s^{sg[\theta]}$  by minimizing DSM loss (1) on new  $p_0^{\theta}$  from the previous step. Then, update the generator  $G_{\theta}$  using the gradient of (5) with frozen  $s^{sg[\theta]}$ . The  $\alpha_{\text{SiD}}$  parameter is chosen from the range [0.5, 1.2], although theoretically only the value  $\alpha_{\text{SiD}} = 0.5$  restores true gradient as we show in our paper.

The authors of **FGM** propose a similar approach, but for the flow matching models. Specifically, they also use a generator  $G_{\theta}$  to produce a distribution  $p_0^{\theta}$ , but instead of DSM loss (1), consider CFM loss (2). The method minimizes the squared  $\ell_2$ -distance between the student and teacher drifts:

<span id="page-3-4"></span>
$$\min_{\theta} \mathbb{E}_{t \sim [0,T]} \mathbb{E}_{x_t \sim p_t^{\theta}} [\|u_t^{\theta}(x_t) - u_t^*(x_t)\|^2], \quad \text{s.t. } u^{\theta} := \arg\min_{v} \mathcal{L}_{CFM}(v, p_0^{\theta}), \tag{6}$$

where the interpolation path  $p_t^{\theta}$  is constructed between the noise  $p_T$  and generator  $p_0^{\theta}$  distributions. To avoid the same problem of differentiating through  $\arg \min$  operator as in SiD, the authors derive a tractable loss whose gradients match those of (6):

<span id="page-3-5"></span>
$$\mathcal{L}_{FGM}(\theta) := \mathbb{E}_{t \sim [0,T]} \mathbb{E}_{z \sim p^{Z}, x_{0}^{\theta} = G_{\theta}(z), x_{t}^{\theta} \sim p_{t}^{\theta}} [-\|u_{t}^{*}(x_{t}^{\theta}) - u_{t}^{sg[\theta]}(x_{t}^{\theta})\|^{2}$$

$$+ 2\langle u_{t}^{*}(x_{t}^{\theta}) - u_{t}^{sg[\theta]}(x_{t}^{\theta}), u_{t}^{*}(x_{t}^{\theta}) - u_{t}^{\theta}(x_{t}^{\theta}|x_{0}^{\theta})\rangle], \text{ s.t. } u^{\theta} = \arg\min_{v} \mathcal{L}_{CFM}(v, p_{0}^{\theta}).$$

$$(7)$$

For data-to-data bridge matching models, the **IBMD** method applies the same idea of minimizing the difference between student and teacher drifts using a similar loss. Notably, all these approaches (SiD, FGM, IBMD) are *data-free*, i.e., they do not use any real data from  $p_0^*$  to train a generator.

#### <span id="page-3-1"></span>2.4 GANS FOR REAL DATA INCORPORATION

Although FGM and SiD methods exhibit strong performance in one-step generation, the generator in these methods is trained under the guidance of the teacher model alone. It means that the generator cannot get more information about the real data that the teacher has learned. For example, it is not expected to correct the teacher's errors. To address this problem, recent works (Yin et al., 2024a; Zhou et al., 2024a) propose adding real data via GANs (Goodfellow et al., 2014). In such approaches, the encoder of fake model is typically augmented with an extra discriminator head D that distinguishes between the generated and real data noising processes via the following adversarial loss:

<span id="page-3-6"></span>
$$\mathcal{L}_{\text{adv}} = \mathbb{E}_{t \sim [0, T]} \left[ \mathbb{E}_{x_t^* \sim p_t^*} \left[ \ln D_t(x_t^*) \right] + \mathbb{E}_{x_t^\theta \sim p_t^\theta} \left[ \ln \left[ 1 - D_t(x_t^\theta) \right] \right] \right]. \tag{8}$$

The overall objective in such hybrid frameworks (Zhou et al., 2024a) consists of generator loss:

$$\mathcal{L}_{G_{\theta}} = \mathcal{L}_{\text{FGM/SiD}}(\theta) + \lambda_{\text{adv}}^{G_{\theta}} \mathcal{L}_{\text{adv}}(\theta),$$

And fake model/discriminator loss:

$$\mathcal{L}_D = \mathcal{L}_{\text{CFM/DSM}} + \lambda_{\text{adv}}^D \mathcal{L}_{\text{adv}}.$$

Here,  $\lambda_{\rm adv}^{G_{\theta}}$  and  $\lambda_{\rm adv}^{D}$  are weighting coefficients for the adversarial components. Despite empirical gains, the GAN augmentation entails nontrivial costs: it necessitates architectural modifications, such as an auxiliary discriminator head, and inherits the well-known optimization problems of adversarial training, such as non-stationary objectives, mode collapse, and sensitivity to training dynamics.

### <span id="page-3-0"></span>3 Universal distillation of matching models with real data

In this section, we present our novel RealUID approach for matching models enhanced by real data. First, we show that the previous data-free distillation methods can be unified under the single UID framework (§3.1). Then, we describe how this framework is connected to prior works (§3.2) and inverse optimization (§3.3). Using this intuition, we propose and discuss the real data modified UID framework (RealUID) with a natural way to incorporate real data without GANs (§3.4).

#### <span id="page-4-0"></span>3.1 Universal Inverse Distillation

To learn a complex real data distribution  $p_0^*$ , one usually trains a teacher function  $f^* := \arg\min_f \mathcal{L}_{\text{UM}}(f,p_0^*)$  that is then used in a multi-step sampling procedure (Def. 1). To avoid time-consuming sampling, one can train a simple student generator  $G_\theta : \mathcal{Z} \to \mathbb{R}^D$  with parameters  $\theta$  to reproduce the real data  $p_0^*$  from the distribution  $p^\mathcal{Z}$  on the latent space  $\mathcal{Z}$ . The teacher function serves as a guide that shows how close the student distribution  $p_0^\theta$  and the real data  $p_0^*$  are. FGM and SiD methods (§2.3) train such generator via minimizing the squared  $\ell_2$ -distance between the known teacher function  $f^*$  and an unknown student function  $f^\theta := \arg\min_f \mathcal{L}_{\text{UM}}(f,p_0^\theta)$ :

<span id="page-4-1"></span>
$$\mathbb{E}_{t \sim [0,T]} \mathbb{E}_{x_{t}^{\theta} \sim p_{t}^{\theta}} [\|f_{t}^{*}(x_{t}^{\theta}) - f_{t}^{\theta}(x_{t}^{\theta})\|^{2}] = \mathbb{E}_{t \sim [0,T]} \mathbb{E}_{x_{t}^{\theta} \sim p_{t}^{\theta}} [\|f_{t}^{*}(x_{t}^{\theta}) - \mathbb{E}_{x_{0}^{\theta} \sim p_{0}^{\theta}(\cdot|x_{t}^{\theta})} [f_{t}^{\theta}(x_{t}^{\theta}|x_{0}^{\theta})]\|^{2}]$$

$$= \mathbb{E}_{t \sim [0,T]} \mathbb{E}_{x_{t}^{\theta} \sim p_{t}^{\theta}} [\|f_{t}^{*}(x_{t}^{\theta})\|^{2}] - 2 \cdot \mathbb{E}_{t \sim [0,T]} \mathbb{E}_{x_{t}^{\theta} \sim p_{t}^{\theta}, x_{0}^{\theta} \sim p_{0}^{\theta}(\cdot|x_{t}^{\theta})} [\langle f_{t}^{*}(x_{t}^{\theta}), f_{t}^{\theta}(x_{t}^{\theta}|x_{0}^{\theta})\rangle]$$

$$+ \mathbb{E}_{t \sim [0,T]} \mathbb{E}_{x_{t}^{\theta} \sim p_{t}^{\theta}} [\|\mathbb{E}_{x_{0}^{\theta} \sim p_{0}^{\theta}(\cdot|x_{t}^{\theta})} [f_{t}^{\theta}(x_{t}^{\theta}|x_{0}^{\theta})]\|^{2}], \tag{9}$$

where  $p_t^{\theta}$  is the probability path constructed between generated data  $p_0^{\theta}$  and noise  $p_T$ . The problem is that the final term (9) cannot be calculated directly, since it involves the math expectation inside the squared norm, unlike the other terms which are linear in the expectations. It means that a simple estimate of  $\|f_t^{\theta}(x_t^{\theta}|x_0^{\theta})\|^2$  using samples  $x_0^{\theta}$  and  $x_t^{\theta}$  will be biased. Moreover, to differentiate through the math expectation inside the norm, an explicit dependence of  $p_0^{\theta}$  on  $\theta$  is required, while, in practice, usually only dependence of samples  $x_0^{\theta}$  on  $\theta$  is known.

**Making loss tractable via linearization.** To resolve this problem, we use the identity  $||a||^2 = \max_{b \in \mathbb{R}^D} \{-||b||^2 + 2\langle b, a \rangle\}, \forall a \in \mathbb{R}^D$ . For a fixed time t and point  $x_t^\theta$ , we reformulate the squared norm (9) as this identity and parametrize vector b via an auxiliary function  $\delta: [0, T] \times \mathbb{R}^D \to \mathbb{R}^D$ :

<span id="page-4-2"></span>
$$\mathbb{E}_{t \sim [0,T], [\|f_t^*(x_t^{\theta}) - f_t^{\theta}(x_t^{\theta})\|^2]} = \max_{\delta_t(x_t^{\theta})} \mathbb{E}_{t \sim [0,T], [-\|\delta_t(x_t^{\theta})\|^2 + 2\langle \delta_t(x_t^{\theta}), f_t^*(x_t^{\theta}) - f_t^{\theta}(x_t^{\theta})\rangle]} \\ = \max_{\delta_t(x_t^{\theta})} \mathbb{E}_{t \sim [0,T], [-\|\delta_t(x_t^{\theta})\|^2 + 2\langle \delta_t(x_t^{\theta}), f_t^*(x_t^{\theta})\rangle - 2\langle \delta_t(x_t^{\theta}), \mathbb{E}_{x_0^{\theta} \sim p_0^{\theta}(\cdot|x_t^{\theta})}[f_t^{\theta}(x_t^{\theta}|x_0^{\theta})]\rangle]} . (10)$$

Now, all loss terms are linear and can be sampled. The parameterization  $\delta = f^* - f$  with a fake function  $f: [0,T] \times \mathbb{R}^D \to \mathbb{R}^D$  allows us to get an elegant form:

$$(10) = \max_{f_{t}(x_{t}^{\theta})} \mathbb{E}_{t \sim [0,T], x_{0}^{\theta} \sim p_{0}^{\theta}, \left\{-\|f_{t}^{*}(x_{t}^{\theta}) - f_{t}(x_{t}^{\theta})\|^{2} + 2\langle f_{t}^{*}(x_{t}^{\theta}) - f_{t}(x_{t}^{\theta}), f_{t}^{*}(x_{t}^{\theta}) - f_{t}^{\theta}(x_{t}^{\theta}|x_{0}^{\theta})\rangle\right\}}$$

$$= \max_{f_{t}(x_{t}^{\theta})} \left\{ \mathbb{E}_{t \sim [0,T], x_{0}^{\theta} \sim p_{0}^{\theta}, \left[\|f_{t}^{*}(x_{t}^{\theta}) - f_{t}^{\theta}(x_{t}^{\theta}|x_{0}^{\theta})\|^{2}\right] - \mathbb{E}_{t \sim [0,T], x_{0}^{\theta} \sim p_{0}^{\theta}, \left[\|f_{t}(x_{t}^{\theta}) - f_{t}^{\theta}(x_{t}^{\theta}|x_{0}^{\theta})\|^{2}\right]}\right\}.$$

$$= \mathcal{L}_{\text{IM}}(f, x_{0}^{\theta})$$

$$= \mathcal{L}_{\text{IM}}(f, x_{0}^{\theta})$$

$$= \mathcal{L}_{\text{IM}}(f, x_{0}^{\theta})$$

**Summary.** We build a universal distillation framework as a single min-max optimization (13), implicitly minimizing squared  $\ell_2$ -distance between teacher and student functions. When real and generated probability paths match, these functions match as well, and the distance attains its minimum.

<span id="page-4-8"></span>**Theorem 1** (Real data generator minimizes UID loss). Let teacher  $f^* := \arg\min_f \mathcal{L}_{UM}(f, p_0^*)$  be the minimizer of UM loss (Def. 1) on real data  $p_0^* \in \mathcal{P}(\mathbb{R}^D)$ . Then, real data generator  $G_{\theta^*}$  s.t.  $p_0^{\theta^*} = p_0^*$  is a solution to the min-max optimization of Universal Inverse Distillation (UID) loss  $\mathcal{L}_{UID}(f, p_0^{\theta})$  over fake function f and generator distribution  $p_0^{\theta}$ :

<span id="page-4-6"></span><span id="page-4-5"></span><span id="page-4-3"></span>
$$\min_{\theta} \max_{f} \left\{ \mathcal{L}_{UID}(f, p_0^{\theta}) := \mathcal{L}_{UM}(f^*, p_0^{\theta}) - \mathcal{L}_{UM}(f, p_0^{\theta}) \right\}. \tag{13}$$

<span id="page-4-7"></span>**Lemma 1** (UID loss minimizes squared  $\ell_2$ -distance). Maximization of UID loss (13) over fake function f represents the squared  $\ell_2$ -distance between the student function  $f^{\theta} := \arg\min_f \mathcal{L}_{UM}(f, p_0^{\theta})$  and the teacher  $f^* := \arg\min_f \mathcal{L}_{UM}(f, p_0^*)$ :

<span id="page-4-4"></span>
$$\max_{f} \mathcal{L}_{UID}(f, p_0^{\theta}) = \mathbb{E}_{t \sim [0, T]} \mathbb{E}_{x_t^{\theta} \sim p_t^{\theta}} [\|f_t^*(x_t^{\theta}) - f_t^{\theta}(x_t^{\theta})\|^2].$$
 (14)

In UID framework, the trained fake model simply learns the current student function  $f^{\theta}$  by minimizing UM loss  $\mathcal{L}_{\text{UM}}(f,p_0^{\theta})$ . Note that for points  $x_t^{\theta}$  out of the generator's domain s.t.  $p_t^{\theta}(x_t^{\theta}) \approx 0$ , the distance (14) vanishes, and the generator cannot receive feedback from the uncovered real data. Moreover, if the teacher function is inaccurate, the generator will learn it with all inaccuracies.

![](_page_5_Figure_1.jpeg)

Figure 1: Pipeline of **our RealUID distillation framework** (§3) with the direct incorporation of real data  $p_0^*$  adjusted by parameters  $\alpha, \beta \in (0, 1]$ . The figure depicts flow matching models predicting denoised samples. It distills a costly frozen teacher model  $f^*$  (blue) into a one-step generator  $G_\theta$  (red) upon min-max optimization of  $\mathcal{L}_{R\text{-UID}}^{\alpha,\beta}(f,p_0^\theta)$  loss over fake model f (green) and generator distribution  $p_0^\theta$  with parameters  $\theta$ . It updates the fake model several times per one generator update for stability. Algorithm's pseudocode is located in Appendix B.

#### <span id="page-5-0"></span>3.2 RELATION TO PRIOR DISTILLATION WORKS

FGM and SiD approaches formulate distillation as a constraint minimization of generator loss subject to the optimal fake model. For generator updates, the explicit UID loss (11) matches FGM loss (7) and SiD loss (5) with  $\alpha_{\text{SiD}} = 0.5$ . For a fake model, it also minimizes the UM loss on the generated data. The work (Gushchin et al., 2025) was the first to formulate the distillation of bridge matching models in their IBMD framework as a min-max optimization of the single loss (12).

Although previous works derive the same losses, we give a new, simple explanation using a linearization technique. This technique is more powerful and general for handling intractable terms than complex proofs for concrete models from FGM, SiD or IBMD. It allows us to build other distillations, e.g., a loss for minimizing the  $\ell_2$ -distance instead of the squared one (see Appendix A.4).

#### <span id="page-5-1"></span>3.3 CONNECTION WITH INVERSE OPTIMIZATION

We derive UID loss (13) by minimizing the squared  $\ell_2$ -distance between teacher and student functions. However, this loss admits another interpretation: its structure is typical for inverse optimization (Chan et al., 2025). In this framework, one considers a parametric family of optimization problems  $\min_f \mathcal{L}(f,\theta)$  with objective loss  $\mathcal{L}(f,\theta)$  depending on argument f and parameters  $\theta$ . The goal is to find the parameters  $\theta^*$  that yield a known, desired solution  $f^* = \arg\min_f \mathcal{L}(f,\theta^*)$ . One standard way to recover the required parameters is to solve the same min-max problem as (13):

<span id="page-5-3"></span>
$$\min_{\theta} \max_{f} \left\{ \mathcal{L}(f^*, \theta) - \mathcal{L}(f, \theta) \right\} \sim \min_{\theta} \left\{ \mathcal{L}(f^*, \theta) - \min_{f} \left\{ \mathcal{L}(f, \theta) \right\} \right\}. \tag{15}$$

The inverse problem (15) always has minimum 0 which is attained when  $\theta = \theta^*$ .

Although the inverse optimization can handle arbitrary losses  $\mathcal{L}$ , it does not describe the properties of the optimized functions or how to find solutions. In our case, we show that all losses are tractable and minimize the distances between teacher and student functions (Lemmas 1 and 2).

#### <span id="page-5-2"></span>3.4 REALUID: NATURAL APPROACH FOR REAL DATA INCORPORATION

Previous distillation methods add real data during training only via GANs with extra discriminator and adversarial loss. We propose a simpler, more natural way that requires no extra models or losses.

<span id="page-5-4"></span>Based on intuition from inverse optimization (§3.3), we see that the min-max inverse problem (15) is compatible with other losses. It allows us to redesign the UM loss (3) to incorporate real data into it. A key constraint is that the loss must still yield the same teacher upon minimization on the real data. Thus, we derive a novel Unified Matching loss with real data - a weighted sum of two UM-like losses on generated and real data parameterized by  $\alpha, \beta \in (0, 1]$  which control the weights.

**Definition 2.** We define Universal Matching loss with real data (RealUM)  $\mathcal{L}_{R\text{-}UM}^{\alpha,\beta}(f,p_0^{\theta})$  that is parametrized by  $\alpha,\beta \in (0,1]$  and takes trainable function f and generated data  $p_0^{\theta}$  as arguments:

<span id="page-6-1"></span>
$$\mathcal{L}_{R\text{-}UM}^{\alpha,\beta}(f,p_{0}^{\theta}) = \underbrace{\alpha \cdot \mathbb{E}_{t \sim [0,T]} \mathbb{E}_{x_{0}^{\theta} \sim p_{0}^{\theta}, x_{t}^{\theta} \sim p_{t}^{\theta}(\cdot | x_{0}^{\theta})} \left[ \|f_{t}(x_{t}^{\theta}) - \frac{\beta}{\alpha} f^{\theta}(x_{t}^{\theta} | x_{0}^{\theta}) \|^{2} \right]}_{generated data \ p_{0}^{\theta} \ term} + \underbrace{(1-\alpha) \cdot \mathbb{E}_{t \sim [0,T]} \mathbb{E}_{x_{0}^{*} \sim p_{0}^{*}, x_{t}^{*} \sim p_{t}^{*}(\cdot | x_{0}^{*})} \left[ \|f_{t}(x_{t}^{*}) - \frac{1-\beta}{1-\alpha} f_{t}^{*}(x_{t}^{*} | x_{0}^{*}) \|^{2} \right]}_{real \ data \ p_{0}^{*} \ term}.$$

$$(1-\alpha) \cdot \mathbb{E}_{t \sim [0,T]} \mathbb{E}_{x_{0}^{*} \sim p_{0}^{*}, x_{t}^{*} \sim p_{t}^{*}(\cdot | x_{0}^{*})} \left[ \|f_{t}(x_{t}^{*}) - \frac{1-\beta}{1-\alpha} f_{t}^{*}(x_{t}^{*} | x_{0}^{*}) \|^{2} \right].$$

$$(16)$$

For  $\alpha = 1$ , we consider only  $\beta = 1$ , i.e., the pure generated data term.

RealUM loss (16) for all  $\alpha, \beta$  and UM loss (3) yield the same teacher when input is real data  $p_0^*$ , since if we consider only the f-dependent terms in the losses, we have:

$$\mathcal{L}_{\text{R-UM}}^{\alpha,\beta}(f,p_0^*) \propto \mathbb{E}_{t,x_0^*,x_t^*} \underbrace{\left[\underbrace{[\alpha+(1-\alpha)]} \cdot \left\langle f_t(x_t^*), f_t(x_t^*) \right\rangle + 2\underbrace{\left[\alpha\cdot\frac{\beta}{\alpha}+(1-\alpha)\cdot\frac{1-\beta}{1-\alpha}\right]}_{=1} \left| \left\langle f_t(x_t^*), f_t^*(x_t^*|x_0^*) \right\rangle \right]}_{=1} \\ \propto \mathbb{E}_{t,x_0^*,x_t^*} \big[ \|f_t(x_t^*) - f_t^*(x_t^*|x_0^*)\|^2 \big] \Longrightarrow \arg\min_{f} \mathcal{L}_{\text{R-UM}}^{\alpha,\beta}(f,p_0^*) = \arg\min_{f} \mathcal{L}_{\text{UM}}(f,p_0^*) = f^*.$$

Hence, the min-max inverse scheme (15) with RealUM loss and the old teacher  $f^*$  still has a real data generator as a solution, but now real data is incorporated via the real data terms of  $\mathcal{L}_{\text{R-UM}}^{\alpha,\beta}(f,p_0^\theta)$ :

$$\min{}_{\theta}\{\underbrace{\mathcal{L}_{\text{R-UM}}^{\alpha,\beta}(f^*,p_0^{\theta}) - \min{}_{f}\{\mathcal{L}_{\text{R-UM}}^{\alpha,\beta}(f,p_0^{\theta})\}}_{\geq 0}\} \stackrel{p_0^{\theta}=p_0^*}{=} \mathcal{L}_{\text{R-UM}}^{\alpha,\beta}(f^*,p_0^*) - \underbrace{\min{}_{f}\{\mathcal{L}_{\text{R-UM}}^{\alpha,\beta}(f,p_0^*)\}}_{=\mathcal{L}_{\text{R-UM}}^{\alpha,\beta}(f^*,p_0^*)} = 0.$$

<span id="page-6-3"></span>**Theorem 2** (Real data generator minimizes RealUID loss). Let teacher  $f^* := \arg\min_f \mathcal{L}_{UM}(f, p_0^*)$  be the minimizer of UM loss (Def. 1) on real data  $p_0^* \in \mathcal{P}(\mathbb{R}^D)$ . Then, real data generator  $G_{\theta^*}$  s.t.  $p_0^{\theta^*} = p_0^*$  is a solution to the min-max optimization of Universal Inverse Distillation loss with real data (RealUID)  $\mathcal{L}_{R-UID}^{\alpha,\beta}(f,p_0^{\theta})$  over fake function f and generator distribution  $p_0^{\theta}$  (see Def. 2):

<span id="page-6-4"></span>
$$\min_{\theta} \max_{f} \left\{ \mathcal{L}_{R\text{-}U\!I\!D}^{\alpha,\beta}(f,p_0^{\theta}) := \mathcal{L}_{R\text{-}U\!M}^{\alpha,\beta}(f^*,p_0^{\theta}) - \mathcal{L}_{R\text{-}U\!M}^{\alpha,\beta}(f,p_0^{\theta}) \right\}. \tag{17}$$

We provide analysis of RealUID in Appendix A.1, below we highlight the most important findings.

**Role of coefficients**  $\alpha, \beta$ . Our RealUID uses real data only to minimize  $\mathcal{L}^{\alpha,\beta}_{\text{R-UM}}(f,p^{\theta}_0)$  loss over fake function f. Thus, the trained fake function memorizes both the real data and the generator's current state. In turn, the generator is influenced by the real data indirectly, only via this fake function. As shown in Lemma 2, RealUID implicitly minimizes the rescaled distance (18) between the teacher and generator functions. This distance is still minimal when  $p^{\theta}_0 = p^*_0$ , alternatively proving Theorem 2.

<span id="page-6-0"></span>**Lemma 2** (Distance minimized by RealUID loss). Maximization of RealUID loss (16) over fake function f represents the weighted squared  $\ell_2$ -distance between the student function  $f^{\theta} := \arg\min_f \mathcal{L}_{UM}(f, p_0^{\theta})$  and the teacher  $f^* := \arg\min_f \mathcal{L}_{UM}(f, p_0^{\pi})$ :

<span id="page-6-2"></span>
$$\max_{f} \mathcal{L}_{R\text{-}UID}^{\alpha,\beta}(f,p_{0}^{\theta}) = \mathbb{E}_{t \sim [0,T], \\ x_{t}^{*} \sim p_{t}^{*}} \left[ \frac{\left\| \frac{\beta}{\alpha} \cdot [p_{t}^{*}(x_{t}^{*}) f_{t}^{*}(x_{t}^{*}) - p_{t}^{\theta}(x_{t}^{*})] + (p_{t}^{\theta}(x_{t}^{*}) - p_{t}^{*}(x_{t}^{*})) \cdot f_{t}^{*}(x_{t}^{*})}{p_{t}^{*}(x_{t}^{*})((1-\alpha)p_{t}^{*}(x_{t}^{*}) + \alpha p_{t}^{\theta}(x_{t}^{*}))/\alpha^{2}} \right]. \tag{18}$$

The proof of Lemma 2 is located in Appendix A.1.2. With the help of real data, our RealUID loss now provides the generator with the feedback on the real data domain it needs to cover, i.e., the distance (18) does not vanish for points  $x_t$  s.t.  $p^{\theta}(x_t) \approx 0$ ,  $p^*(x_t) \gg 0$  (see Appendix A.1.3). Moreover, if teacher function is inaccurate, RealUID can now provably fix teacher's errors (see Appendix A.1.4).

Choice of coefficients  $\alpha$ ,  $\beta$ . Lemma 2 shows that, instead of values  $\alpha$  and  $\beta$ , actually the values  $\alpha$  and  $\beta/\alpha$  determine the balance between real and generated data in the minimized distance (18). Furthermore, coefficient  $\alpha$  only sets the general scaling of the distance, while  $\beta/\alpha$  plays the most important role, as it determines the relation between  $f_t^{\theta}$  and  $f_t^*$  inside the distance.

The value  $\beta/\alpha = 1$  yields the distance identical to the data-free distance (14) up to scaling. Thus, even when  $\alpha = \beta < 1$  and real data is formally added, it may have no effect on the generator. Excessively

low α and β diminish the effect of the generated data terms in the trained fake function, leading to vanishing gradients. The same issue occurs with <sup>β</sup>/<sup>α</sup> ≪ 1 in [\(18\)](#page-6-2), while <sup>β</sup>/<sup>α</sup> ≫ 1 diminish the effect of real data in the right term of [\(18\)](#page-6-2). See complete distance analysis in Appendix [A.1.3.](#page-16-0) Moreover, if teacher function is inaccurate, only the choice <sup>β</sup>/<sup>α</sup> ̸= 1 can fix teacher's errors (see Appendix [A.1.4\)](#page-18-0).

Hence, good coefficients α, β ∈ (0, 1] can be chosen by first finding good <sup>β</sup>/<sup>α</sup> ̸= 1, as it has the largest impact, and then adjusting α < 1. Both <sup>β</sup>/<sup>α</sup> and α should be close to 1.

Comparison with GAN-based methods. Unlike SiD and FGM with GANs [\(8\)](#page-3-6), we do not use extra adversarial losses and discriminator to incorporate real data. We only modify UM loss, preserving its core structure and fake model architecture. While general adversarial loss is unrelated to the main distillation loss and has uninterpretable scaling hyperparameters, our RealUID loss and weighting coefficients α, β ∈ (0, 1] come naturally from the data-free UID loss. The original UID loss [\(13\)](#page-4-3), equivalent to FGM [\(7\)](#page-3-5) and SiD [\(5\)](#page-3-3) with αSiD = 0.5, is obtained when α = β = 1.

Alternative loss form. Our RealUID is implicitly related to the linearization scheme used to obtain data-free UID ([§3.1\)](#page-4-0). *The loss* [\(17\)](#page-6-4) *can be derived by splitting each term in the linearized UID loss* [\(10\)](#page-4-2) *between real and generated data according to proportions* α *and* β (see Appendix [A.1.1\)](#page-14-1). This form helps to prove RealUID's properties and extend it beyond the inversion scheme ([§5\)](#page-9-0).

Extension for Bridge Matching and Stochastic Interpolants frameworks. In Appendix [C.3,](#page-29-0) we demonstrate that our framework can be easily extended to data-to-data matching models by parameterizing the generated data coupling π θ (x0, x<sup>T</sup> ) instead of the data distribution p θ 0 .

# <span id="page-7-2"></span>4 EXPERIMENTS

All our PyTorch implementations and the latest checkpoints are publicly available in

<https://github.com/David-cripto/RealUID>.

This section provides an ablation study and evaluation of our RealUID, assessing both its performance and computational efficiency. We begin in ([§4.1\)](#page-7-0) by detailing the experimental setup based on flow matching models. In ([§4.2\)](#page-7-1), we show that our incorporation of real data via coefficients α, β improves performance, speeds up convergence, and enables effective fine-tuning. In ([§4.3\)](#page-9-1), we assess the benchmark performance and computational demands of RealUID relative to SOTA methods. Additional experimental details and results are provided in Appendix [D.](#page-30-0)

# <span id="page-7-0"></span>4.1 EXPERIMENTAL SETUP

Datasets and Evaluation Protocol. The experiments were conducted on the CIFAR-10 dataset with 32 × 32 resolution [\(Krizhevsky et al.,](#page-11-6) [2009\)](#page-11-6) and on the CelebA dataset with 64 × 64 resolution [\(Liu et al.,](#page-11-7) [2015\)](#page-11-7), see Appendix [D.3.](#page-31-0) In line with the prior works [\(Karras et al.,](#page-11-8) [2019;](#page-11-8) [2022\)](#page-11-9), we report test FID scores [\(Heusel et al.,](#page-10-7) [2017\)](#page-10-7), computed using 50k generated samples.

Implementation Details. We implement our RealUID framework for flow matching models from Appendix [B.](#page-27-0) In contrast to prior studies [\(Zhou et al.,](#page-12-3) [2024b](#page-12-3)[;a;](#page-12-2) [Huang et al.,](#page-11-4) [2024\)](#page-11-4) which employ the computationally demanding EDM architecture [\(Karras et al.,](#page-11-9) [2022\)](#page-11-9) our work adopts a more lightweight alternative [\(Tong et al.,](#page-12-5) [2024\)](#page-12-5). We also train our own flow matching teacher models using CFM loss [\(2\)](#page-2-2). Further implementation details and efficiency analysis are provided in Appendix [D.1.](#page-30-1)

### <span id="page-7-1"></span>4.2 BENCHMARKING METHODS UNDER A UNIFIED EXPERIMENTAL CONFIGURATION

We evaluate RealUID under a unified experimental protocol (fixed architecture and implementation). We begin by conducting an ablation over α, β to assess the influence of real-data incorporation. We then compare RealUID to a GAN-based alternative, showing that RealUID achieves comparable or superior accuracy. Furthermore, we analyze convergence, indicating that RealUID variants with real data train substantially faster than baselines without real-data. Finally, we explore a fine-tuning stage initialized from strong RealUID checkpoints, showing further performance gains.

Ablation study of coefficients α, β. We restrict the search for optimal parameters α and β to values near 1, specifically α, β ∈ [0.85, 1.0] with increments of 0.02. Setting these parameters too low leads to noisy generated samples. Following the analysis in (§3.4), we perform a grid search over the values  $\alpha$  and  $\beta/\alpha$  instead of the original  $\alpha$  and  $\beta$ . The results are reported in Table 1. As a baseline, we highlight the UID model without data incorporation, i.e., our RealUID with  $\alpha = 1.0$ ,  $\beta = 1.0$ .

As shown in the table, the ratio  $\beta/\alpha$  has the largest impact on the final metrics, while  $\alpha$  only adjusts them. Using real data with  $\beta/\alpha=1$  or with large values outside the range [0.98, 1.02] consistently degrades performance. In contrast, values  $\beta/\alpha=0.98$  or  $\beta/\alpha=1.02$  outperform the baseline for a majority of  $\alpha$ . Note that these practical results match the theoretical description in (§3.4).

<span id="page-8-0"></span>

| Generation    | $\alpha \setminus \frac{\beta}{\alpha}$ | 0.96 | 0.98 | 1.00 | 1.02 | 1.04 |
|---------------|-----------------------------------------|------|------|------|------|------|
|               | 0.90                                    | 2.66 | 2.44 | 2.66 | 2.25 | 2.55 |
| Unconditional | 0.92                                    | 2.73 | 2.36 | 2.65 | 2.23 | 2.66 |
|               | 0.94                                    | 2.79 | 2.35 | 2.65 | 2.28 | 2.58 |
|               | 0.96                                    | 2.85 | 2.37 | 2.58 | 2.29 | 2.65 |
|               | 0.98                                    | 2.97 | 2.33 | 2.62 | 2.38 | -    |
|               | 1.0                                     | -    | -    | 2.58 | -    | -    |
| Conditional   | 0.90                                    | 2.34 | 2.16 | 2.38 | 2.19 | 2.26 |
|               | 0.92                                    | 2.28 | 2.12 | 2.35 | 2.21 | 2.23 |
|               | 0.94                                    | 2.29 | 2.13 | 2.35 | 2.19 | 2.25 |
|               | 0.96                                    | 2.36 | 2.09 | 2.32 | 2.13 | 2.27 |
|               | 0.98                                    | 2.34 | 2.02 | 2.26 | 2.05 | -    |
|               | 1.0                                     | -    | -    | 2.21 | -    | -    |

| Generation    | $\lambda_{\rm adv}^{G_{\theta}}$ | $\lambda_{\rm adv}^D$ | FID (↓) |  |
|---------------|----------------------------------|-----------------------|---------|--|
| Unconditional | 0.1                              | 0.3                   | 2.42    |  |
|               | 0.3                              | 1                     | 2.29    |  |
|               | 1                                | 3                     | 2.39    |  |
|               | 5                                | 15                    | 2.54    |  |
| Conditional   | 0.1                              | 0.3                   | 2.22    |  |
|               | 0.3                              | 1                     | 2.12    |  |
|               | 1                                | 3                     | 2.15    |  |
|               | 5                                | 15                    | 2.40    |  |

Table 1: Ablation studies of our  $(\alpha, \frac{\beta}{\alpha})$  parameters in the left table and adversarial weighting parameters  $(\lambda_{\text{adv}}^{G_{\theta}}, \lambda_{\text{adv}}^{D})$  in the right table for CIFAR-10. The baseline RealUID  $(\alpha = 1.0, \beta = 1.0)$  does not use real data. Configurations that sligtly and substantially outperform the baseline are highlighted. All values report FID  $\downarrow$ , where lower is better. The best configuration in each case is **bolded**. The mark "–" denotes infeasible parameters.

Comparison with GAN-based method. We integrate the GAN-based approach (8) proposed by (Zhou et al., 2024a) as an alternative method for incorporating real data, enabling a direct comparison with our RealUID formulation. We combine the GAN loss with the UID baseline. As shown in Table 1, the best-performing configurations are achieved with GAN losses ( $\lambda_{\rm adv}^{G_{\theta}}=0.3,\,\lambda_{\rm adv}^{D}=1$ ). While this setup performs comparably to RealUID ( $\alpha=0.92,\,\beta=0.94$ ) in the unconditional setting, it remains clearly inferior to RealUID ( $\alpha=0.98,\,\beta=0.96$ ) in the conditional case.

<span id="page-8-2"></span>Table 2: This table presents the results of ablation study of our RealUID framework, evaluated using the FID metric under both unconditional and conditional generation setups. The Teacher Flow model with 100 NFE is reported as a reference. The performance of the UID (FGM) baseline without real-data incorporation is indicated in *italic*. For emphasis, we <u>underline</u> the two counterparts that incorporate real data: the GAN-based and our RealUID methods. The best-performing configurations, obtained via an additional fine-tuning stage, are highlighted in **bold**. Qualitative results are presented in Appendix D.5.1.

| Model                                                                                                                              | FID (↓) |
|------------------------------------------------------------------------------------------------------------------------------------|---------|
| Teacher Flow (NFE=100)                                                                                                             | 3.57    |
| UID (FGM)                                                                                                                          | 2.58    |
| UID + GAN ( $\lambda_{odv}^{G_{\theta}} = 0.3, \lambda_{odv}^{D} = 1 \mid \lambda_{FT}^{G_{\theta}} = 25, \lambda_{FT}^{D} = 75$ ) | 2.10    |
| RealUID ( $\alpha = 0.92, \beta = 0.94 \mid \alpha_{FT} = 0.92, \beta_{FT} = 0.86$ ) (Ours)                                        | 1.98    |

| Model                                                                                                                                                                                                                                                                                          | FID (↓)      |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------|
| Teacher Flow (NFE=100)                                                                                                                                                                                                                                                                         | 5.56         |
| UID (FGM)                                                                                                                                                                                                                                                                                      | 2.21         |
| UID + GAN ( $\lambda_{\text{adv}}^{G_{\theta}} = 0.3$ , $\lambda_{\text{adv}}^{D} = 1$   $\lambda_{\text{FT}}^{G_{\theta}} = 25$ , $\lambda_{\text{FT}}^{D} = 75$ )<br>RealUID ( $\alpha = 0.98$ , $\beta = 0.96$   $\alpha_{\text{FT}} = 0.96$ , $\beta_{\text{FT}} = 0.92$ ) ( <b>Ours</b> ) | 1.88<br>1.87 |

<span id="page-8-1"></span>![](_page_8_Figure_10.jpeg)

![](_page_8_Figure_11.jpeg)

Figure 2: Evolution of FID during CIFAR-10 distillation for (i) the UID (FGM) baseline, (ii) the best-performing RealUID configurations, and (iii) subsequent fine-tuning, evaluated in both unconditional and conditional settings. The performances of Teacher Flow and UID+GAN are indicated by horizontal lines in their respective colors.

**Convergence Speed.** Our RealUID $(\alpha, \beta)$  with parameters which are highlighted in Table 1 achieves faster convergence than the UID baseline. For clarity, we present qualitative comparisons in Figure 2. The best RealUID configurations reach the saturated performance level of the baseline after  $\sim 100$ k iterations, whereas the baseline requires  $\sim 300$ k iterations to achieve comparable metrics.

**Fine-tuning stage.** We observe that RealUID and GAN frameworks offer substantial flexibility for fine-tuning. In this procedure, the generator is initialized from the best-performing checkpoint obtained during training from scratch of the corresponding framework, while the fake model is initialized from the teacher. Fine-tuning then proceeds with new values  $\alpha_{\rm FT}$  and  $\beta_{\rm FT}$  for our RealUID and  $\lambda_{\rm FT}^D$  and  $\lambda_{\rm FT}^D$  for GANs. We present the best-found fine-tuning configurations for both methods in Table 2. Ablation study analyzing the effect of loss coefficients is provided in Appendix D.2.

Scaling to larger datasets. In Appendix D.3, we provide the results of the same ablation study on the CelebA dataset with  $64 \times 64$  resolution. Notably, our RealUID performance and the optimal values  $\frac{\beta}{\alpha}$  remain the same across datasets.

### <span id="page-9-1"></span>4.3 BASELINE COMPARISON

As shown in Tables 3 and 4, our RealUID after fine-tuning consistently outperforms all prior flow-based models on CIFAR-10, significantly surpassing the strongest flow distillation baseline, FGM. Despite its compact and lightweight architecture (§4.1) with nearly  $2\times$  faster inference, it achieves performance comparable to leading diffusion distillation methods SiD ( $\alpha_{\text{SiD}}=1.0\setminus1.2$ ), while falling short of adversarially enhanced models such as SiD<sup>2</sup>A. We hypothesize that this performance gap is attributed to architectural and teacher capacity differences rather than the lack of adversarial loss.

### Our latest checkpoints and metrics (Appendix D.4) are available in our repository.

<span id="page-9-2"></span>Table 3: Comparison of *unconditional* generation on CIFAR-10. The best method under the FID metric in each section is highlighted with **bold**.

Table 4: Comparison of *conditional* generation on CIFAR-10. The best method under the FID metric in each section is highlighted with **bold**.

| Family          | Model                                                                                                                                                                                                                                      | NFE  | $FID(\downarrow)$                    | each section is highlighted with <b>bold</b> . |                                                                                                              |      |         |
|-----------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------|--------------------------------------|------------------------------------------------|--------------------------------------------------------------------------------------------------------------|------|---------|
|                 | DDPM (Ho et al., 2020)                                                                                                                                                                                                                     | 1000 | 3.17                                 |                                                | is inginigated with bold.                                                                                    |      |         |
|                 | VP-EDM (Karras et al., 2022)                                                                                                                                                                                                               | 35   | 1.97                                 | Family                                         | Model                                                                                                        | NFE  | FID (↓) |
|                 | StyleGAN2+ADA+Tune (Karras et al., 2020)                                                                                                                                                                                                   | 1    | 2.92                                 | 1<br>4<br>1                                    | VP-EDM (Karras et al., 2022)                                                                                 | 35   | 1.79    |
|                 | StyleGAN2+ADA+Tune+DI (Luo et al., 2023)                                                                                                                                                                                                   | 1    | 2.71<br>2.54                         |                                                | GET-Base (Geng et al., 2023)                                                                                 | 1    | 6.25    |
|                 | Diffusion ProjectedGAN (Wang et al., 2022)                                                                                                                                                                                                 | 1    |                                      |                                                | BigGAN (Brock et al., 2018)                                                                                  | i    | 14.73   |
|                 | iCT-deep (Song & Dhariwal, 2023)                                                                                                                                                                                                           | 1    | 2.51                                 |                                                | BigGAN+Tune (Brock et al., 2018)                                                                             | 1    | 8.47    |
| Diffusion & GAN | Diff-Instruct (Luo et al., 2023)                                                                                                                                                                                                           | 1    | 4.53                                 |                                                | StyleGAN2+ADA (Karras et al., 2020)                                                                          | 1    | 3.49    |
| Diffusion & GAN | DMD (Yin et al., 2024b)                                                                                                                                                                                                                    | 1    |                                      | StyleGAN2+ADA+Tune (Karras et al., 2020)       | 1                                                                                                            | 2.42 |         |
|                 | CTM (Kim et al., 2023)                                                                                                                                                                                                                     | 1    | 1.98                                 |                                                | StyleGAN2+ADA+Tune+DI (Luo et al., 2023)<br>StyleGAN-XL (Sauer et al., 2022)                                 | 1    | 2.27    |
|                 | sCD (Lu & Song, 2024)                                                                                                                                                                                                                      | 1    | 3.66                                 |                                                |                                                                                                              | 1    | 1.85    |
|                 | sCT (Lu & Song, 2024)                                                                                                                                                                                                                      | 1    | 2.85                                 |                                                | StyleSAN-XL (Takida et al., 2023)                                                                            | 1    | 1.36    |
|                 |                                                                                                                                                                                                                                            | 1    | 2.03 Diffusion & GAN<br>1.92<br>1.52 | Diff-Instruct (Luo et al., 2023)               | 1                                                                                                            | 4.19 |         |
|                 |                                                                                                                                                                                                                                            | 1    |                                      | 1.92<br>1.52<br>1.52                           |                                                                                                              | 1    | 2.66    |
|                 |                                                                                                                                                                                                                                            | 1    |                                      |                                                |                                                                                                              | 1    | 3.82    |
|                 |                                                                                                                                                                                                                                            | 1    |                                      |                                                |                                                                                                              | 1    | 5.58    |
|                 |                                                                                                                                                                                                                                            | 1    |                                      |                                                |                                                                                                              | 1    | 1.44    |
|                 | SiD, $\alpha_{SD} = 1.0$ (Zhou et al., 2024b) 1 1.52 Diffusion & GAN DMD (Yin et al., 2024b) SiDA, $\alpha_{SD} = 1.0$ (Zhou et al., 2024a) 1 1.52 DMD ( $\omega_0$ . KL.) (Yin et al., 2024b) DMD ( $\omega_0$ . RL.) (Yin et al., 2024b) | 1    | 1.73                                 |                                                |                                                                                                              |      |         |
|                 | CFM (Yang et al., 2024)                                                                                                                                                                                                                    | 2    | 5.34                                 |                                                |                                                                                                              | 1    | 1.93    |
| Flow-based      |                                                                                                                                                                                                                                            | 1    |                                      |                                                |                                                                                                              | 1    | 1.71    |
|                 | MeanFlow (Geng et al., 2025)                                                                                                                                                                                                               | 1    | 1 2.92<br>1 2.69                     |                                                | 1                                                                                                            | 1.44 |         |
|                 | FACM (Peng et al., 2025)                                                                                                                                                                                                                   | 1    |                                      |                                                | $SiD^2A$ , $\alpha_{SiD} = 1.0$ (Zhou et al., 2024a)<br>$SiD^2A$ , $\alpha_{SiD} = 1.2$ (Zhou et al., 2024a) | 1    | 1.40    |
|                 | 1-ReFlow (+Distill) (Liu et al., 2023)                                                                                                                                                                                                     | 1    | 6.18                                 |                                                | SID A, $\alpha_{SiD} = 1.2$ (Zhou et al., 2024a)                                                             | 1    | 1.39    |
|                 | 2-ReFlow (+Distill) (Liu et al., 2023)                                                                                                                                                                                                     | 1    | 1 4.85                               | Flow-based                                     | FGM (Huang et al., 2024)                                                                                     | 1    | 2.58    |
|                 | 3-ReFlow (+Distill) (Liu et al., 2023)                                                                                                                                                                                                     | 1    | 5.21                                 |                                                | RealUID + FT (Ours)                                                                                          |      | 1.87    |
|                 |                                                                                                                                                                                                                                            | 1    |                                      |                                                |                                                                                                              |      |         |
|                 | FGM (Huang et al., 2024)                                                                                                                                                                                                                   | 1    | 3.08                                 |                                                |                                                                                                              |      |         |
|                 | RealUID + FT (Ours)                                                                                                                                                                                                                        | - 1  | 1.98                                 |                                                |                                                                                                              |      |         |

### <span id="page-9-0"></span>5 DISCUSSION AND EXTENSIONS

**Extensions.** Our RealUID framework (§3.4 and Appendix C) can distill Flow/Bridge Matching, diffusion models, and Stochastic Interpolants enhanced by a novel natural way to incorporate real data. In Appendix A, we provide three extensions of our RealUID: more flexible General RealUID (Appendix A.2), General SiD framework for all matching models with real data and  $\alpha_{\text{SiD}} \neq 1/2$  (Appendix A.3) and Normalized RealUID for minimizing non-squared  $\ell_2$ -distance (Appendix A.4).

**Relation to DMD.** Instead of minimizing the squared  $\ell_2$ -distance between the score functions, *Distribution Matching Distillation* (Luo et al., 2023; Wang et al., 2023; Yin et al., 2024b;a, **DMD**) approach minimizes the KL divergence between the real and generated data. Its gradients are computed using the generator and teacher score functions, leading to the similar alternating updates. The DMD is a special case of our UID framework with the special KL loss (different from UM loss) and only one parameter  $\alpha = \beta$  (Appendix A.5).

# <span id="page-10-11"></span>6 BROADER IMPACT

This paper presents work whose goal is to advance the field of Artificial Intelligence, Machine Learning and Generative Modeling. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

Acknowledgements. The work was supported by the grant for research centers in the field of AI provided by the Ministry of Economic Development of the Russian Federation in accordance with the agreement 000000C313925P4F0002 and the agreement №139-10-2025-033.

# <span id="page-10-12"></span>7 LLM USAGE

Large Language Models (LLMs) were used only to assist with rephrasing sentences and improving the clarity of the text. All scientific content, results, and interpretations in this paper were developed solely by the authors.

# REFERENCES

- <span id="page-10-3"></span>Michael S Albergo, Nicholas M Boffi, and Eric Vanden-Eijnden. Stochastic interpolants: A unifying framework for flows and diffusions. *arXiv preprint arXiv:2303.08797*, 2023.
- <span id="page-10-10"></span>Andrew Brock, Jeff Donahue, and Karen Simonyan. Large scale gan training for high fidelity natural image synthesis. *arXiv preprint arXiv:1809.11096*, 2018.
- <span id="page-10-6"></span>Timothy CY Chan, Rafid Mahmood, and Ian Yihang Zhu. Inverse optimization: Theory and applications. *Operations Research*, 73(2):1046–1074, 2025.
- <span id="page-10-13"></span>Valentin De Bortoli, Guan-Horng Liu, Tianrong Chen, Evangelos A Theodorou, and Weilie Nie. Augmented bridge matching. *arXiv preprint arXiv:2311.06978*, 2023.
- <span id="page-10-1"></span>Ruiqi Gao, Emiel Hoogeboom, Jonathan Heek, Valentin De Bortoli, Kevin Patrick Murphy, and Tim Salimans. Diffusion models and gaussian flow matching: Two sides of the same coin. In *The Fourth Blogpost Track at ICLR 2025*, 2025.
- <span id="page-10-9"></span>Zhengyang Geng, Ashwini Pokle, and J Zico Kolter. One-step diffusion distillation via deep equilibrium models. *Advances in Neural Information Processing Systems*, 36:41914–41931, 2023.
- <span id="page-10-8"></span>Zhengyang Geng, Mingyang Deng, Xingjian Bai, J Zico Kolter, and Kaiming He. Mean flows for one-step generative modeling. *arXiv preprint arXiv:2505.13447*, 2025.
- <span id="page-10-5"></span>Ian J Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. *Advances in neural information processing systems*, 27, 2014.
- <span id="page-10-14"></span>Nikita Gushchin, Alexander Kolesov, Alexander Korotin, Dmitry P Vetrov, and Evgeny Burnaev. Entropic neural optimal transport via diffusion processes. *Advances in Neural Information Processing Systems*, 36, 2024.
- <span id="page-10-4"></span>Nikita Gushchin, David Li, Daniil Selikhanovych, Evgeny Burnaev, Dmitry Baranchuk, and Alexander Korotin. Inverse bridge matching distillation. 2025.
- <span id="page-10-7"></span>Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. *Advances in neural information processing systems*, 30, 2017.
- <span id="page-10-2"></span>Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. *Advances in neural information processing systems*, 33:6840–6851, 2020.
- <span id="page-10-0"></span>Peter Holderrieth, Marton Havasi, Jason Yim, Neta Shaul, Itai Gat, Tommi Jaakkola, Brian Karrer, Ricky TQ Chen, and Yaron Lipman. Generator matching: Generative modeling with arbitrary markov processes. *arXiv preprint arXiv:2410.20587*, 2024.

- <span id="page-11-4"></span>Zemin Huang, Zhengyang Geng, Weijian Luo, and Guo-jun Qi. Flow generator matching. *arXiv preprint arXiv:2410.19310*, 2024.
- <span id="page-11-18"></span>J Stuart Hunter. The exponentially weighted moving average. *Journal of quality technology*, 18(4): 203–210, 1986.
- <span id="page-11-8"></span>Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pp. 4401–4410, 2019.
- <span id="page-11-10"></span>Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Analyzing and improving the image quality of stylegan. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pp. 8110–8119, 2020.
- <span id="page-11-9"></span>Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusionbased generative models. *Advances in neural information processing systems*, 35:26565–26577, 2022.
- <span id="page-11-12"></span>Dongjun Kim, Chieh-Hsin Lai, Wei-Hsiang Liao, Naoki Murata, Yuhta Takida, Toshimitsu Uesaka, Yutong He, Yuki Mitsufuji, and Stefano Ermon. Consistency trajectory models: Learning probability flow ode trajectory of diffusion. *arXiv preprint arXiv:2310.02279*, 2023.
- <span id="page-11-17"></span>Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. *arXiv preprint arXiv:1412.6980*, 2014.
- <span id="page-11-6"></span>Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.
- <span id="page-11-0"></span>Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In *The Eleventh International Conference on Learning Representations*, 2023. URL <https://openreview.net/forum?id=PqvMRDCJT9t>.
- <span id="page-11-1"></span>Qiang Liu. Rectified flow: A marginal preserving approach to optimal transport. *arXiv preprint arXiv:2209.14577*, 2022.
- <span id="page-11-3"></span>Xingchao Liu, Lemeng Wu, Mao Ye, et al. Let us build bridges: Understanding and extending diffusion generative models. In *NeurIPS 2022 Workshop on Score-Based Methods*, 2022.
- <span id="page-11-5"></span>Xingchao Liu, Chengyue Gong, and qiang liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In *The Eleventh International Conference on Learning Representations*, 2023. URL <https://openreview.net/forum?id=XVjTT1nw5z>.
- <span id="page-11-7"></span>Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Deep learning face attributes in the wild. In *Proceedings of International Conference on Computer Vision (ICCV)*, December 2015.
- <span id="page-11-13"></span>Cheng Lu and Yang Song. Simplifying, stabilizing and scaling continuous-time consistency models. *arXiv preprint arXiv:2410.11081*, 2024.
- <span id="page-11-11"></span>Weijian Luo, Tianyang Hu, Shifeng Zhang, Jiacheng Sun, Zhenguo Li, and Zhihua Zhang. Diffinstruct: A universal approach for transferring knowledge from pre-trained diffusion models. *Advances in Neural Information Processing Systems*, 36:76525–76546, 2023.
- <span id="page-11-2"></span>Stefano Peluchetti. Non-denoising forward-time diffusions. *arXiv preprint arXiv:2312.14589*, 2023.
- <span id="page-11-14"></span>Yansong Peng, Kai Zhu, Yu Liu, Pingyu Wu, Hebei Li, Xiaoyan Sun, and Feng Wu. Flow-anchored consistency models. *arXiv preprint arXiv:2507.03738*, 2025.
- <span id="page-11-15"></span>A Sauer, K Schwarz, and A StyleGAN-XL Geiger. scaling stylegan to large diverse datasets. In *Proceedings of the SIGGRAPH Conference. ACM*, pp. 1–10, 2022.
- <span id="page-11-16"></span>Daniil Selikhanovych, David Li, Aleksei Leonov, Nikita Gushchin, Sergei Kushneriuk, Alexander Filippov, Evgeny Burnaev, Iaroslav Koshelev, and Alexander Korotin. One-step residual shifting diffusion for image super-resolution via distillation. *arXiv preprint arXiv:2503.13358*, 2025.

- <span id="page-12-0"></span>Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In *International conference on machine learning*, pp. 2256–2265. pmlr, 2015.
- <span id="page-12-7"></span>Yang Song and Prafulla Dhariwal. Improved techniques for training consistency models. *arXiv preprint arXiv:2310.14189*, 2023.
- <span id="page-12-1"></span>Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In *International Conference on Learning Representations*, 2021. URL [https://openreview.net/forum?](https://openreview.net/forum?id=PxTIG12RRHS) [id=PxTIG12RRHS](https://openreview.net/forum?id=PxTIG12RRHS).
- <span id="page-12-11"></span>Yuhta Takida, Masaaki Imaizumi, Takashi Shibuya, Chieh-Hsin Lai, Toshimitsu Uesaka, Naoki Murata, and Yuki Mitsufuji. San: Inducing metrizability of gan with discriminative normalized linear layer. *arXiv preprint arXiv:2301.12811*, 2023.
- <span id="page-12-5"></span>Alexander Tong, Kilian FATRAS, Nikolay Malkin, Guillaume Huguet, Yanlei Zhang, Jarrid Rector-Brooks, Guy Wolf, and Yoshua Bengio. Improving and generalizing flow-based generative models with minibatch optimal transport. *Transactions on Machine Learning Research*, 2024. ISSN 2835- 8856. URL <https://openreview.net/forum?id=CD9Snc73AW>. Expert Certification.
- <span id="page-12-6"></span>Zhendong Wang, Huangjie Zheng, Pengcheng He, Weizhu Chen, and Mingyuan Zhou. Diffusion-gan: Training gans with diffusion. *arXiv preprint arXiv:2206.02262*, 2022.
- <span id="page-12-13"></span>Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation. *Advances in neural information processing systems*, 36:8406–8441, 2023.
- <span id="page-12-9"></span>Ling Yang, Zixiang Zhang, Zhilong Zhang, Xingchao Liu, Minkai Xu, Wentao Zhang, Chenlin Meng, Stefano Ermon, and Bin Cui. Consistency flow matching: Defining straight flows with velocity consistency. *arXiv preprint arXiv:2407.02398*, 2024.
- <span id="page-12-4"></span>Tianwei Yin, Michael Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Fredo Durand, and Bill ¨ Freeman. Improved distribution matching distillation for fast image synthesis. *Advances in neural information processing systems*, 37:47455–47487, 2024a.
- <span id="page-12-8"></span>Tianwei Yin, Michael Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T Freeman, ¨ and Taesung Park. One-step diffusion with distribution matching distillation. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pp. 6613–6623, 2024b.
- <span id="page-12-12"></span>Kaiwen Zheng, Guande He, Jianfei Chen, Fan Bao, and Jun Zhu. Diffusion bridge implicit models. *arXiv preprint arXiv:2405.15885*, 2024.
- <span id="page-12-10"></span>Linqi Zhou, Stefano Ermon, and Jiaming Song. Inductive moment matching. *arXiv preprint arXiv:2503.07565*, 2025.
- <span id="page-12-2"></span>Mingyuan Zhou, Huangjie Zheng, Yi Gu, Zhendong Wang, and Hai Huang. Adversarial score identity distillation: Rapidly surpassing the teacher in one step. *arXiv preprint arXiv:2410.14919*, 2024a.
- <span id="page-12-3"></span>Mingyuan Zhou, Huangjie Zheng, Zhendong Wang, Mingzhang Yin, and Hai Huang. Score identity distillation: Exponentially fast distillation of pretrained diffusion models for one-step generation. In *Forty-first International Conference on Machine Learning*, 2024b.

# CONTENTS

| 1 |     | Introduction                                                                                          | 1  |  |  |  |  |
|---|-----|-------------------------------------------------------------------------------------------------------|----|--|--|--|--|
| 2 |     | Backgrounds on training and distilling matching models                                                | 2  |  |  |  |  |
|   | 2.1 | Diffusion and flow models<br>                                                                         | 2  |  |  |  |  |
|   | 2.2 | Universal loss for matching models                                                                    | 3  |  |  |  |  |
|   | 2.3 | Distillation of matching-based models<br>                                                             | 3  |  |  |  |  |
|   | 2.4 | GANs for real data incorporation                                                                      | 4  |  |  |  |  |
| 3 |     | Universal distillation of matching models with real data                                              | 4  |  |  |  |  |
|   | 3.1 | Universal Inverse Distillation                                                                        | 5  |  |  |  |  |
|   | 3.2 | Relation to prior distillation works<br>                                                              | 6  |  |  |  |  |
|   | 3.3 | Connection with Inverse Optimization<br>                                                              | 6  |  |  |  |  |
|   | 3.4 | RealUID: natural approach for real data incorporation                                                 | 6  |  |  |  |  |
| 4 |     | Experiments                                                                                           | 8  |  |  |  |  |
|   | 4.1 | Experimental setup<br>                                                                                | 8  |  |  |  |  |
|   | 4.2 | Benchmarking methods under a unified experimental configuration                                       | 8  |  |  |  |  |
|   | 4.3 | Baseline comparison<br>                                                                               | 10 |  |  |  |  |
| 5 |     | Discussion and extensions                                                                             | 10 |  |  |  |  |
| 6 |     | Broader impact                                                                                        | 11 |  |  |  |  |
| 7 |     | LLM Usage                                                                                             | 11 |  |  |  |  |
| A |     | Theoretical proofs and extensions                                                                     | 15 |  |  |  |  |
|   | A.1 | RealUID theoretical properties<br>                                                                    | 15 |  |  |  |  |
|   |     | A.1.1<br>Alternative RealUID split form                                                               | 15 |  |  |  |  |
|   |     | A.1.2<br>Proof of RealUID Distance Lemma 2<br>                                                        | 16 |  |  |  |  |
|   |     | Explanation of the choice of coefficients α and β<br>A.1.3<br>                                        | 17 |  |  |  |  |
|   |     | A.1.4<br>Correction of teacher's errors                                                               | 19 |  |  |  |  |
|   | A.2 | General RealUID loss                                                                                  | 20 |  |  |  |  |
|   | A.3 | General SiD with real data<br>                                                                        | 22 |  |  |  |  |
|   | A.4 | Normalized UID and RealUID losses for minimizing ℓ2-distance                                          | 24 |  |  |  |  |
|   | A.5 | DMD and Inverse Optimization                                                                          | 25 |  |  |  |  |
| B |     | RealUID Algorithm for flow matching models                                                            | 28 |  |  |  |  |
| C |     | Unified Inverse Distillation with real data for Bridge Matching and Stochastic Inter<br>polants<br>29 |    |  |  |  |  |
|   | C.1 | Bridge Matching<br>                                                                                   | 29 |  |  |  |  |

|   | C.2 | Stochastic Interpolants                  | 29         |
|---|-----|------------------------------------------|------------|
|   | C.3 | Objective for general data coupling      | 30         |
| D | Exp | erimental details and additional results | 31         |
|   | D.1 | CIFAR-10 distillation from scratch       | 31         |
|   | D.2 | CIFAR-10 distillation fine-tuning        | 32         |
|   | D.3 | CelebA distillation                      | 32         |
|   | D.4 | Further hyperparameters gridsearch       | 34         |
|   | D.5 | Example of samples for various methods   | 34         |
|   |     | D.5.1 CIFAR-10 generated images          | 35         |
|   |     | D.5.2 CelebA generated images            | <b>ļ</b> 1 |

### <span id="page-14-2"></span>A THEORETICAL PROOFS AND EXTENSIONS

In this appendix, we discuss our RealUID framework (Appendix A.1) in theoretical details and provide three extensions of it: *General RealUID* framework with 3 degrees of freedom (Appendix A.2), *General SiD framework with real data* (Appendix A.3) and *Normalized RealUID* framework for minimizing  $\ell_2$ -distance between teacher and student functions instead of the squared one (Appendix A.4). All proofs are based on the linearization technique and splitting terms in linearized decomposition between real and generated data. We also show that DMD approach is a special case of our UID framework and we can similarly incorporate real data into it (Appendix A.5).

### <span id="page-14-0"></span>A.1 REALUID THEORETICAL PROPERTIES

In this section, we discuss our RealUID loss in detail. We begin by presenting its alternative form and how it connects linearization technique and real data incorporation (Appendix A.1.1). We then demonstrate that the loss minimizes a squared  $\ell_2$ -distance between the rescaled teacher and student functions (Appendix A.1.2). Finally, we provide the motivation of the best choice of coefficients  $\alpha \neq \beta$  from the perspectives of the better distance (Appendix A.1.3) and the correction of the teacher's errors (Appendix A.1.4).

### <span id="page-14-1"></span>A.1.1 ALTERNATIVE REALUID SPLIT FORM

Let us recall the linearization trick that we apply to make the minimized squared norm between the student function  $f^{\theta}$  and the teacher  $f^*$  tractable. For each time t and generated point  $x_t^{\theta}$ , we restate this squared norm as the identity  $||a||^2 = \max_{b \in \mathbb{R}^D} \{-||b||^2 + 2\langle b, a \rangle\}$ ,  $\forall a \in \mathbb{R}$  and use an auxiliary function  $\delta_t(x_t)$  to parametrize a vector b. In the end, we substitute the student function  $f_t^{\theta}(x_t^{\theta})$  with its conditional and differentiable estimate  $f_t^{\theta}(x_t^{\theta}|x_0^{\theta})$ :

<span id="page-14-3"></span>
$$\mathbb{E}_{t \sim [0,T], [\|f_t^*(x_t^{\theta}) - f_t^{\theta}(x_t^{\theta})\|^2]} = \max_{\delta_t(x_t^{\theta})} \mathbb{E}_{t \sim [0,T], [-\|\delta_t(x_t^{\theta})\|^2 + 2\langle \delta_t(x_t^{\theta}), f_t^*(x_t^{\theta}) - f_t^{\theta}(x_t^{\theta})\rangle]} \\
= \mathbb{E}_{t \sim [0,T], x_0^{\theta} \sim p_0^{\theta}, [-\|\delta_t(x_t^{\theta})\|^2 + 2\langle \delta_t(x_t^{\theta}), f_t^*(x_t^{\theta})\rangle - 2\langle \delta_t(x_t^{\theta}), f_t^{\theta}(x_t^{\theta}|x_0^{\theta})\rangle]}.$$
(19)
$$x_t^{\theta} \sim p_t^{\theta} (\cdot |x_0^{\theta})$$

In addition, we use parameterization  $\delta = f^* - f$  with a fake model f to obtain our UID loss which matches the previous distillation losses.

Originally, we derived our RealUID loss (17) from the idea of *splitting each term in the linearized* form of data-free UID (19) between the generated and real data in proportions defined by coefficients  $\alpha$  and  $(1 - \alpha)$ ,  $\alpha$  and  $(1 - \alpha)$  and  $\beta$  and  $(1 - \beta)$ . We present the split form of RealUID loss in Lemma 3, and this form completely matches the inverse optimization form defined in Theorem 2.

<span id="page-15-1"></span>Lemma 3 (RealUID split form). The RealUID loss (17) can be restated as

$$\mathcal{L}_{R\text{-}UID}^{\alpha,\beta}(f,p_0^{\theta}) = \mathbb{E}_{t \sim [0,T], x_0^{\theta} \sim p_0^{\theta},} [-\alpha \|\delta_t(x_t^{\theta})\|^2 + 2\alpha \langle \delta_t(x_t^{\theta}), f_t^*(x_t^{\theta}) \rangle - 2\beta \langle \delta_t(x_t^{\theta}), f_t^{\theta}(x_t^{\theta}|x_0^{\theta}) \rangle]$$

$$+ \mathbb{E}_{t \sim [0,T], x_0^* \sim p_0^*,} [-(1-\alpha) \|\delta_t(x_t^*)\|^2 + 2(1-\alpha) \langle \delta_t(x_t^*), f_t^*(x_t^*) \rangle - 2(1-\beta) \langle \delta_t(x_t^*), f_t^*(x_t^*|x_0^*) \rangle],$$

$$+ \mathbb{E}_{t \sim [0,T], x_0^* \sim p_0^*,} [-(1-\alpha) \|\delta_t(x_t^*)\|^2 + 2(1-\alpha) \langle \delta_t(x_t^*), f_t^*(x_t^*) \rangle - 2(1-\beta) \langle \delta_t(x_t^*), f_t^*(x_t^*|x_0^*) \rangle],$$

with the parameterization  $\delta = f^* - f$ .

The idea of splitting coefficients between two data types helps to prove properties of RealUID, and extend our real data incorporation technique to general form (Appendix A.2), SiD framework with  $\alpha_{\text{SiD}} \neq \frac{1}{2}$  (Appendix A.3) and new distances (Appendix A.4).

*Proof.* Putting explicit values for RealUM loss (16) in RealUID loss (17), we get:

$$\begin{split} &\mathcal{L}_{\text{R-UID}}^{\alpha,\beta}(f,p_0^{\theta}) = \mathcal{L}_{\text{R-UM}}^{\alpha,\beta}(f^*,p_0^{\theta}) - \mathcal{L}_{\text{R-UM}}^{\alpha,\beta}(f,p_0^{\theta}) \\ &= \alpha \cdot \mathbb{E}_{t,x_0^{\theta},x_t^{\theta}} \left[ \|f_t^*(x_t^{\theta}) - \frac{\beta}{\alpha} f^{\theta}(x_t^{\theta}|x_0^{\theta})\|^2 \right] + (1-\alpha) \cdot \mathbb{E}_{t,x_0^*,x_t^*} \left[ \|f_t^*(x_t^*) - \frac{1-\beta}{1-\alpha} f_t^*(x_t^*|x_0^*)\|^2 \right] \\ &- \alpha \cdot \mathbb{E}_{t,x_0^{\theta},x_t^{\theta}} \left[ \|f_t(x_t^{\theta}) - \frac{\beta}{\alpha} f^{\theta}(x_t^{\theta}|x_0^{\theta})\|^2 \right] - (1-\alpha) \cdot \mathbb{E}_{t,x_0^*,x_t^*} \left[ \|f_t(x_t^*) - \frac{1-\beta}{1-\alpha} f_t^*(x_t^*|x_0^*)\|^2 \right]. \end{split}$$

Then, we group the factors with the same data type and multipliers:

$$\begin{split} &\mathcal{L}_{\text{R-UID}}^{\alpha,\beta}(f,p_{0}^{\theta}) = \mathcal{L}_{\text{R-UM}}^{\alpha,\beta}(f^{*},p_{0}^{\theta}) - \mathcal{L}_{\text{R-UM}}^{\alpha,\beta}(f,p_{0}^{\theta}) \\ &= \mathbb{E}_{t,x_{0}^{\theta},x_{t}^{\theta}} \left[ \alpha \cdot \|f_{t}^{*}(x_{t}^{\theta}) - \frac{\beta}{\alpha} f^{\theta}(x_{t}^{\theta}|x_{0}^{\theta})\|^{2} - \alpha \cdot \|f_{t}(x_{t}^{\theta}) - \frac{\beta}{\alpha} f^{\theta}(x_{t}^{\theta}|x_{0}^{\theta})\|^{2} \right] \\ &+ \mathbb{E}_{t,x_{0}^{*},x_{t}^{*}} \left[ (1-\alpha) \cdot \|f_{t}^{*}(x_{t}^{*}) - \frac{1-\beta}{1-\alpha} f_{t}^{*}(x_{t}^{*}|x_{0}^{*})\|^{2} - (1-\alpha) \cdot \|f_{t}(x_{t}^{*}) - \frac{1-\beta}{1-\alpha} f_{t}^{*}(x_{t}^{*}|x_{0}^{*})\|^{2} \right] \\ &= \mathbb{E}_{t,x_{0}^{\theta},x_{t}^{\theta}} \left[ \alpha \cdot \|f_{t}^{*}(x_{t}^{\theta})\|^{2} - 2\beta \cdot \langle f_{t}^{*}(x_{t}^{\theta}), f^{\theta}(x_{t}^{\theta}|x_{0}^{\theta}) \rangle - \alpha \cdot \|f_{t}(x_{t}^{\theta})\|^{2} + 2\beta \cdot \langle f_{t}(x_{t}^{\theta}), f^{\theta}(x_{t}^{\theta}|x_{0}^{\theta}) \rangle \right] \\ &+ \mathbb{E}_{t,x_{0}^{\theta},x_{t}^{\theta}} \left[ (1-\alpha) \cdot \|f_{t}^{*}(x_{t}^{*})\|^{2} - 2(1-\beta) \cdot \langle f_{t}^{*}(x_{t}^{*}) - f_{t}(x_{t}^{*}), f^{*}(x_{t}^{*}|x_{0}^{*}) \rangle - (1-\alpha) \cdot \|f_{t}(x_{t}^{*})\|^{2} \rangle \right] \\ &= \mathbb{E}_{t,x_{0}^{\theta},x_{t}^{\theta}} \left[ \alpha \cdot (\|f_{t}^{*}(x_{t}^{\theta})\|^{2} - \|f_{t}(x_{t}^{\theta})\|^{2}) - 2\beta \cdot \langle f_{t}^{*}(x_{t}^{\theta}) - f_{t}(x_{t}^{\theta}), f^{\theta}(x_{t}^{\theta}|x_{0}^{\theta}) \rangle \right] \\ &+ \mathbb{E}_{t,x_{0}^{\theta},x_{t}^{\theta}} \left[ \alpha \cdot (\|f_{t}^{*}(x_{t}^{*})\|^{2} - \|f_{t}(x_{t}^{*})\|^{2}) - 2(1-\beta) \cdot \langle f_{t}^{*}(x_{t}^{*}) - f_{t}(x_{t}^{*}), f^{*}(x_{t}^{*}|x_{0}^{*}) \rangle \right] \\ &= \mathbb{E}_{t,x_{0}^{\theta},x_{t}^{\theta}} \left[ \alpha \cdot (-\|f_{t}^{*}(x_{t}^{\theta}) - f_{t}(x_{t}^{\theta})\|^{2} + 2\langle f_{t}^{*}(x_{t}^{\theta}) - f_{t}(x_{t}^{\theta}), f_{t}^{*}(x_{t}^{\theta}) \rangle \right] \\ &+ \mathbb{E}_{t,x_{0}^{\theta},x_{t}^{\theta}} \left[ \alpha \cdot (-\|f_{t}^{*}(x_{t}^{\theta}) - f_{t}(x_{t}^{\theta}), f^{\theta}(x_{t}^{\theta}|x_{0}^{\theta}) \rangle \right] \\ &+ \mathbb{E}_{t,x_{0}^{\theta},x_{t}^{\theta}} \left[ \alpha \cdot \langle f_{t}^{*}(x_{t}^{\theta}) - f_{t}(x_{t}^{\theta}), f^{\theta}(x_{t}^{\theta}|x_{0}^{\theta}) \rangle \right] \\ &+ \mathbb{E}_{t,x_{0}^{\theta},x_{t}^{\theta}} \left[ (1-\alpha) \cdot (-\|f_{t}^{*}(x_{t}^{\theta}) - f_{t}(x_{t}^{\theta}), f^{\theta}(x_{t}^{\theta}|x_{0}^{\theta}) \rangle \right] \\ &+ \mathbb{E}_{t,x_{0}^{\theta},x_{t}^{\theta}} \left[ (1-\beta) \cdot \langle f_{t}^{*}(x_{t}^{\theta}) - f_{t}(x_{t}^{\theta}), f^{\theta}(x_{t}^{\theta}|x_{0}^{\theta}) \rangle \right] \\ &+ \mathbb{E}_{t,x_{0}^{\theta},x_{t}^{\theta}} \left[ (1-\beta) \cdot \langle f_{t}^{*}(x_{t}^{\theta}) - f_{t}(x_{t}^{\theta}), f^{\theta}(x_{t}^{\theta}|x_{0}^{\theta}) \rangle \right] \\ &+ \mathbb{E}_{t,x_{0}^{\theta},x_{t}^{\theta}} \left[$$

Finally, denoting parameterization  $\delta = f^* - f$ , we obtain the required form:

$$\mathcal{L}_{\text{R-UID}}^{\alpha,\beta}(f, p_0^{\theta}) = \mathbb{E}_{t, x_0^{\theta}, x_t^{\theta}} [-\alpha \|\delta_t(x_t^{\theta})\|^2 + 2\alpha \langle \delta_t(x_t^{\theta}), f_t^*(x_t^{\theta}) \rangle - 2\beta \langle \delta_t(x_t^{\theta}), f_t^{\theta}(x_t^{\theta}|x_0^{\theta}) \rangle]$$

$$+ \mathbb{E}_{t, x_0^*, x_t^*} [-(1-\alpha) \|\delta_t(x_t^*)\|^2 + 2(1-\alpha) \langle \delta_t(x_t^*), f_t^*(x_t^*) \rangle - 2(1-\beta) \langle \delta_t(x_t^*), f_t^*(x_t^*|x_0^*) \rangle].$$

### <span id="page-15-0"></span>A.1.2 Proof of RealUID DISTANCE LEMMA 2

*Proof of Lemma 2.* In this proof, we use the split form of our ReaLUID loss from Lemma 3. First, we take math expectation over data points  $x_0^*$ . Since the expectation can be taken in a reverse order, i.e.,  $\mathbb{E}_{x_0^* \sim p_0^*, x_t^* \sim p_t^*(\cdot|x_0^*)} = \mathbb{E}_{x_t^* \sim p_t^*, x_0^* \sim p_0^*(\cdot|x_t^*)}$ , we see that

$$\mathbb{E}_{x_0^* \sim p_0^*, x_t^* \sim p_t^*(\cdot | x_0^*)} [\langle \delta_t(x_t^*), f_t^*(x_t^* | x_0^*) \rangle] = \mathbb{E}_{x_t^* \sim p_t^*} [\langle \delta_t(x_t^*), \mathbb{E}_{x_0^* \sim p_0^*(\cdot | x_t^*)} [f_t^*(x_t^* | x_0^*)] \rangle] \\
= \mathbb{E}_{x_t^* \sim p_t^*} [\langle \delta_t(x_t^*), f_t^*(x_t^*) \rangle]. \tag{20}$$

For the generated data term  $\mathbb{E}_{x_0^{\theta} \sim p_0^{\theta}, x_t^{\theta} \sim p_t^{\theta}(\cdot|x_0^{\theta})}[\langle \delta_t(x_t^{\theta}), f_t^{\theta}(x_t^{\theta}|x_0^{\theta}) \rangle] = \mathbb{E}_{x_t^{\theta} \sim p_t^{\theta}}[\langle \delta_t(x_t^{\theta}), f_t^{\theta}(x_t^{\theta}) \rangle],$  the reasoning is similar. Thus, we can write down RealUID loss in an explicit form with  $\delta_t = f_t^* - f_t$ :

<span id="page-15-2"></span>
$$\mathcal{L}_{\text{R-UID}}^{\alpha,\beta}(\delta,p_0^{\theta}) = \mathbb{E}_{t \sim [0,T]} \mathbb{E}_{x_t^{\theta} \sim p_t^{\theta}} [-\alpha \|\delta_t(x_t^{\theta})\|^2 + 2\alpha \langle \delta_t(x_t^{\theta}), f_t^*(x_t^{\theta}) \rangle - 2\beta \langle \delta_t(x_t^{\theta}), f_t^{\theta}(x_t^{\theta}) \rangle]$$

$$+\mathbb{E}_{t\sim[0,T]}\mathbb{E}_{x_{t}^{*}\sim p_{t}^{*}}[-(1-\alpha)\|\delta_{t}(x_{t}^{*})\|^{2}+2(1-\alpha)\langle\delta_{t}(x_{t}^{*}),f_{t}^{*}(x_{t}^{*})\rangle-2(1-\beta)\langle\delta_{t}(x_{t}^{*}),f_{t}^{*}(x_{t}^{*})\rangle]. \tag{21}$$

Then, we rescale the generated data terms in RealUID loss (21) using the equality  $p_t^{\theta}(x_t) = \frac{p_t^{\theta}(x_t)}{p_t^*(x_t)}p_t^*(x_t)$  for  $x_t \in \mathbb{R}^D$  (we assume  $p_t^*(x_t) > 0, \forall x_t, t$ ) leaving only math expectation w.r.t. the real data, i.e,

$$\begin{split} \mathcal{L}_{\text{R-UID}}^{\alpha,\beta}(\delta,p_{0}^{\theta}) &= \mathbb{E}_{\substack{t \sim [0,T] \\ x_{t}^{*} \sim p_{t}^{*}}} \left[ -[(1-\alpha) + \alpha \frac{p_{t}^{\theta}(x_{t}^{*})}{p_{t}^{*}(x_{t}^{*})}] \|\delta_{t}(x_{t}^{*})\|^{2} \right] \\ &- \mathbb{E}_{\substack{t \sim [0,T] \\ x_{t}^{*} \sim p_{t}^{*}}} \left[ 2\beta \frac{p_{t}^{\theta}(x_{t}^{*})}{p_{t}^{*}(x_{t}^{*})} \langle \delta_{t}(x_{t}^{*}), f_{t}^{\theta}(x_{t}^{*}) \rangle + 2[(\beta-\alpha) + \alpha \frac{p_{t}^{\theta}(x_{t}^{*})}{p_{t}^{*}(x_{t}^{*})}] \langle \delta_{t}(x_{t}^{*}), f_{t}^{*}(x_{t}^{*}) \rangle \right]. \end{split}$$

Finally, we maximize the loss w.r.t.  $\delta_t(x_t^*)$  for each  $x_t^*$  and t as a quadratic function. The maximum is achieved when

$$\delta_t(x_t^*) = \frac{[(\beta - \alpha) + \alpha \frac{p_t^{\theta}(x_t^*)}{p_t^*(x_t^*)}] f_t^*(x_t^*) - \beta \frac{p_t^{\theta}(x_t^*)}{p_t^*(x_t^*)} f_t^{\theta}(x_t^*)}{[(1 - \alpha) + \alpha \frac{p_t^{\theta}(x_t^*)}{p_t^*(x_t^*)}]}$$

or in terms of the fake model  $f = f^* - \delta$ 

$$\left(\arg\max_{f} \mathcal{L}_{\text{R-UID}}^{\alpha,\beta}(f, p_0^{\theta})\right)(t, x_t) = \frac{f_t^*(x_t) \cdot (1 - \beta) + f_t^{\theta}(x_t) \cdot \beta \frac{p_t^{\theta}(x_t)}{p_t^*(x_t)}}{(1 - \alpha) + \alpha \frac{p_t^{\theta}(x_t)}{p_t^*(x_t)}}.$$
(22)

The maximum itself equals to

$$\max_{f} \mathcal{L}_{\text{R-UID}}^{\alpha,\beta}(f, p_0^{\theta}) = \mathbb{E}_{t \sim [0,T]} \mathbb{E}_{x_t^* \sim p_t^*} \left[ \frac{\|f_t^*(x_t^*) \cdot ((\beta - \alpha) + \alpha \frac{p_t^{\theta}(x_t^*)}{p_t^*(x_t^*)}) - f_t^{\theta}(x_t^*) \cdot \beta \frac{p_t^{\theta}(x_t^*)}{p_t^*(x_t^*)}}{(1 - \alpha) + \alpha \frac{p_t^{\theta}(x_t^*)}{p_t^*(x_t^*)}} \right]^2 \right]$$

It is easy to see that when  $p_0^{\theta} = p_0^*$  and  $f^{\theta} = f^*$  this distance achieves its minimal value 0. Moreover, optimal fake model in this case matches the teacher  $f^*$ , i.e.,

$$\left(\arg\max_{f} \mathcal{L}_{\text{R-UID}}^{\alpha,\beta}(f, p_0^*)\right)(t, x_t) = \frac{f_t^*(x_t) \cdot (1 - \beta) + f_t^*(x_t) \cdot \beta \frac{p_t^*(x_t)}{p_t^*(x_t)}}{(1 - \alpha) + \alpha \frac{p_t^*(x_t)}{p_t^*(x_t)}} = f_t^*(x_t).$$

### <span id="page-16-0"></span>A.1.3 Explanation of the choice of coefficients $\alpha$ and $\beta$

Here we show that the best way to incorporate real data during generator training is to set  $\beta/\alpha \neq 1$ .

Following Lemma 2, we know exactly what distance our RealUID loss implicitly minimizes. Below we examine it for various  $\alpha, \beta \in (0, 1]$ :

$$\max_{f} \mathcal{L}_{\text{R-UID}}^{\alpha,\beta}(f,p_0^{\theta}) = \int_{x_t} l_t(x_t,\beta,\alpha) dx_t,$$

$$l_t(x_t,\beta,\alpha) := \frac{\alpha^2 \|(p_t^*(x_t)(\frac{\beta}{\alpha}-1) + p_t^{\theta}(x_t)) \cdot f_t^*(x_t) - \frac{\beta}{\alpha} \cdot p_t^{\theta}(x_t) \cdot f_t^{\theta}(x_t)\|^2}{(1-\alpha)p_t^*(x_t) + \alpha p_t^{\theta}(x_t)},$$

where  $l_t(x_t, \beta, \alpha)$  denotes the distance for the particular point  $x_t$ .

The total distance mostly sums up from the two groups of points: incorrectly generated points from the generator's main domain, i.e.,  $p_t^{\theta}(x_t) \gg 0$ ,  $p^*(x_t) \approx 0$ , and real data points which are not covered by the generator, i.e.,  $p_t^{\theta}(x_t) \approx 0$ ,  $p^*(x_t) \gg 0$ . For the points out of both domains  $p_t^{\theta}(x_t) \approx 0$ ,  $p_t^*(x_t) \approx 0$ , the distance tends to 0, as well as for matching points  $p_t^{\theta}(x_t) \approx p_t^*(x_t)$ .

<span id="page-17-0"></span>![](_page_17_Figure_1.jpeg)

![](_page_17_Figure_2.jpeg)

Figure 3: RealUID loss for 1D-Gaussians under various coefficients  $(\alpha, \beta)$ .

Choice of coefficients  $\alpha, \beta$ . Next, we consider various coefficients  $\alpha, \beta \in (0, 1]$  and how they affect two main groups of points.

• All configurations affect the incorrectly generated points  $x_t: p_t^*(x_t) \approx 0, p^{\theta}(x_t) \gg 0$ :

$$l_t(x_t, \beta, \alpha) \approx \frac{\|\alpha p_t^{\theta}(x_t) \cdot f_t^*(x_t) - \beta p_t^{\theta}(x_t) \cdot f_t^{\theta}(x_t)\|^2}{\alpha p_t^{\theta}(x_t)} \approx \frac{\beta^2 \|f_t^{\theta}(x_t)\|^2}{\alpha} p_t^{\theta}(x_t) \gg 0.$$
 (23)

Note that increasing  $\beta/\alpha > 1$  will diminish the weight of the distance in comparison with  $\alpha = \beta = 1$ , while decreasing otherwise will lift the weight up.

• Configuration  $\beta < \alpha = 1$  is unstable for uncovered real data points  $x_t : p_t^{\theta}(x_t) \approx 0, p^*(x_t) \gg 0$ :

$$l_t(x_t, \beta, \alpha) \approx \frac{\|p_t^*(x_t)(\beta - 1) \cdot f_t^*(x_t) - \beta p_t^{\theta}(x_t) \cdot f_t^{\theta}(x_t)\|^2}{p_t^{\theta}(x_t)} \approx \infty.$$

• Configuration  $\beta = \alpha = 1$  (UID loss) does not affect uncovered real data points  $x_t : p_t^{\theta}(x_t) \approx 0, p^*(x_t) \gg 0$ :

$$l_t(x_t, \beta, \alpha) \approx \frac{\|p_t^{\theta}(x_t) \cdot f_t^*(x_t) - p_t^{\theta}(x_t) \cdot f_t^{\theta}(x_t)\|^2}{p_t^{\theta}(x_t)} = \|f_t^*(x_t) - f_t^{\theta}(x_t)\|^2 p_t^{\theta}(x_t) \approx 0.$$

• Configuration  $\beta = \alpha < 1$  does not affect uncovered real data points  $x_t : p_t^{\theta}(x_t) \approx 0, p^*(x_t) \gg 0$ :

$$l_t(x_t, \beta, \alpha) \approx \frac{\|\alpha p_t^{\theta}(x_t) f_t^*(x_t) - \beta p_t^{\theta}(x_t) f_t^{\theta}(x_t)\|^2}{(1 - \alpha) p_t^*(x_t)} = \frac{\|\alpha f_t^*(x_t) - \beta f_t^{\theta}(x_t)\|^2}{(1 - \alpha)} \frac{(p_t^{\theta}(x_t))^2}{p_t^*(x_t)} \approx 0.$$

Notably, in this configuration, the distance drops even faster than when  $\alpha = \beta = 1$ , what makes it even less preferable.

• Only configuration  $\beta/\alpha \neq 1$  affects the uncovered real data points  $x_t : p_t^{\theta}(x_t) \approx 0, p^*(x_t) \gg 0$ :

$$l_t(x_t, \beta, \alpha) \approx \frac{\|p_t^*(x_t)(\beta - \alpha) \cdot f_t^*(x_t) - \beta p_t^{\theta}(x_t) \cdot f_t^{\theta}(x_t)\|^2}{(1 - \alpha)p_t^*(x_t)} \gg 0.$$

**Visual illustration.** We analytically calculate the loss surface  $l_t(x_t, \alpha, \beta)$  between the FM models transforming one-dimensional real data Gaussian  $\mathcal{N}(\mu^*, 1)$  and generated Gaussian  $\mathcal{N}(\mu^\theta, 1)$  to noise  $\mathcal{N}(0, 1)$  on the time interval [0, 1]. In this case, the generated and real data interpolations are  $p_t^\theta(x_t) = \mathcal{N}(x_t|\mu^\theta(1-t), t^2+(1-t)^2)$  and  $p_t^*(x_t) = \mathcal{N}(x_t|\mu^*(1-t), t^2+(1-t)^2)$ . The unconditional vector field u=f between  $\mathcal{N}(\mu, 1)$  and  $\mathcal{N}(0, 1)$  can be calculated as

$$u_{t}(x_{t}) = \mathbb{E}_{x_{0} \sim p_{0}(\cdot|x_{t})} \left[ \frac{x_{t} - x_{0}}{t} \right] = \int_{x_{0}} \left( \frac{x_{t} - x_{0}}{t} \right) \cdot \mathcal{N} \left( \frac{x_{t} - x_{0}(1 - t)}{t} | 0, 1 \right) \cdot \mathcal{N}(x_{0} | \mu, 1) dx_{0}$$

$$= \frac{a(2t^{2} - 2t) - bt^{2}}{\sqrt{2\pi}(1 - 2t + 2t^{2})^{\frac{3}{2}}} \exp\left( -\frac{(x_{t} - \mu(1 - t))^{2}}{2(1 - 2t + 2t^{2})^{2}} \right). \tag{24}$$

In Figure 3, we depict the loss surfaces for the fixed time t=1/3, real data  $\mu^*=2$ , generated data  $\mu^\theta=-2$  and various pairs of  $(\alpha,\beta)$ . We can see that configurations  $\beta/\alpha=1$  do not detect the real data sample, even when  $\alpha=\beta<1$  and real data is formally used. while  $\beta/\alpha\neq1$  actually spots both domains, increasing the weight of generator domain when  $\beta/\alpha>1$  and decreasing it otherwise.

#### <span id="page-18-0"></span>A.1.4 CORRECTION OF TEACHER'S ERRORS

In this chapter, we assume that instead of accurate teacher  $f^* = \arg\min_f \mathcal{L}_{UM}(f, p_0^*)$  we have access only to the arbitrary corrupted teacher  $\tilde{f}^*$ . We will show that adding real data via our approach with  $\alpha \neq \beta$  provably mitigates the teacher's errors in the final generator.

**Minimized distance.** With the corrupted teacher  $\tilde{f}^*$  and  $\tilde{\delta} = \tilde{f}^* - f$ , our corrupted RealUID loss takes the split form from Lemma 3

$$\mathcal{L}_{\text{R-UID}}^{\alpha,\beta}(\tilde{\delta},p_0^{\theta}) = \mathbb{E}_{t \sim [0,T],x_0^{\theta} \sim p_0^{\theta},} [-\alpha \|\tilde{\delta}_t(x_t^{\theta})\|^2 + 2\alpha \langle \tilde{\delta}_t(x_t^{\theta}), \tilde{f}_t^*(x_t^{\theta}) \rangle - 2\beta \langle \tilde{\delta}_t(x_t^{\theta}), f_t^{\theta}(x_t^{\theta}|x_0^{\theta}) \rangle ]$$

$$+\mathbb{E}_{t \sim [0,T],x_0^* \sim p_0^*, [-(1-\alpha)\|\tilde{\delta}_t(x_t^*)\|^2 + 2(1-\alpha)\langle\tilde{\delta}_t(x_t^*), \tilde{f}_t^*(x_t^*)\rangle - 2(1-\beta)\langle\tilde{\delta}_t(x_t^*), f_t^*(x_t^*|x_0^*)\rangle].$$

Note that sampled terms  $f_t^*(x_t^*|x_0^*)$  and  $f_t^{\theta}(x_t^{\theta}|x_0^{\theta})$  are not affected by the corruption and give the accurate functions  $f_t^*(x_t^*) = \mathbb{E}_{x_0^* \sim p_t^*(\cdot|x_t^*)}[f_t^*(x_t^*|x_0^*)]$  and  $f_t^{\theta}(x_t^{\theta}) = \mathbb{E}_{x_0^{\theta} \sim p_t^{\theta}(\cdot|x_0^{\theta})}[f_t^{\theta}(x_t^{\theta}|x_0^{\theta})]$ :

<span id="page-18-1"></span>
$$\mathcal{L}_{\text{R-UID}}^{\alpha,\beta}(\tilde{\delta}, p_0^{\theta}) = \mathbb{E}_{t \sim [0,T]} \mathbb{E}_{x_t^{\theta} \sim p_t^{\theta}} [-\alpha \|\tilde{\delta}_t(x_t^{\theta})\|^2 + 2\alpha \langle \tilde{\delta}_t(x_t^{\theta}), \tilde{f}_t^*(x_t^{\theta}) \rangle - 2\beta \langle \delta_t(x_t^{\theta}), f_t^{\theta}(x_t^{\theta}) \rangle] \\ + \mathbb{E}_{t \sim [0,T]} \mathbb{E}_{x_t^* \sim p_t^*} [-(1-\alpha) \|\tilde{\delta}_t(x_t^*)\|^2 + 2(1-\alpha) \langle \tilde{\delta}_t(x_t^*), \tilde{f}_t^*(x_t^*) \rangle - 2(1-\beta) \langle \tilde{\delta}_t(x_t^*), f_t^*(x_t^*) \rangle].$$

Then, we rescale the generated data terms using the equality  $p_t^{\theta}(x_t) = \frac{p_t^{\theta}(x_t)}{p_t^*(x_t)} p_t^*(x_t)$  for  $x_t \in \mathbb{R}^D$  (we assume  $p_t^*(x_t) > 0, \forall x_t, t$ ) leaving only math expectation w.r.t. the real data, i.e,

$$\mathcal{L}_{\text{R-UID}}^{\alpha,\beta}(\tilde{\delta}, p_0^{\theta}) = \mathbb{E}_{t \sim [0,T], x_t^* \sim p_t^*} \left[ -[(1-\alpha) + \alpha \frac{p_t^{\theta}(x_t^*)}{p_t^*(x_t^*)}] (\|\tilde{\delta}_t(x_t^*)\|^2 + 2\langle \tilde{\delta}_t(x_t^*), \tilde{f}_t^*(x_t^*) \rangle) \right] \\ - \mathbb{E}_{t \sim [0,T],} \left[ 2\langle \tilde{\delta}_t(x_t^*), (1-\beta) f_t^*(x_t^*) + \beta \frac{p_t^{\theta}(x_t^*)}{p_t^*(x_t^*)} f_t^{\theta}(x_t^*) \rangle \right].$$

Finally, we maximize the loss w.r.t.  $\tilde{\delta}_t(x_t^*)$  for each  $x_t^*$  and t as a quadratic function  $\max_{\tilde{\delta}} \mathcal{L}_{\text{R-UID}}^{\alpha,\beta}(\tilde{\delta},p_0^{\theta}) =$ 

$$\mathbb{E}_{t \sim [0,T]} \mathbb{E}_{x_t^* \sim p_t^*} \left[ \frac{\|\tilde{f}_t^*(x_t^*) \cdot ((1-\alpha) + \alpha \frac{p_t^{\theta}(x_t^*)}{p_t^*(x_t^*)}) - (1-\beta) f_t^*(x_t^*) - \beta \frac{p_t^{\theta}(x_t^*)}{p_t^*(x_t^*)} f_t^{\theta}(x_t^*) \|^2}{(1-\alpha) + \alpha \frac{p_t^{\theta}(x_t^*)}{p_t^*(x_t^*)}} \right]. \quad (25)$$

Hence, max-min optimization of the corrupted RealUID loss implicitly minimizes expected distance (25). However, due to arbitrary function  $\tilde{f}$ , we now cannot guarantee that minimum is achived when the relation inside the norm equals 0. Previously, we could use the solution  $p^{\theta} = p^*$  which obviously achieved a minimum of 0. Now, due to the implicit and complex relationship between  $f^{\theta}$  and  $p^{\theta}$ , we can neither find an explicit form for the optimal  $p^{\theta}$  nor guarantee the minimum of 0.

Choice of coefficients  $\alpha, \beta$ . Here we give an intuition on why coefficients  $\beta/\alpha \neq 1$  can fix the teacher's errors, while  $\beta/\alpha = 1$  cannot. For simplicity, we assume that the minimized distance (25) actually attains minimum of 0 when

<span id="page-18-2"></span>
$$((1 - \alpha)p_t^*(x_t) + \alpha p_t^{\theta}(x_t)) \cdot \tilde{f}_t^*(x_t) - (1 - \beta)p_t^*(x_t) \cdot f_t^*(x_t) - \beta p_t^{\theta}(x_t) \cdot f_t^{\theta}(x_t^*) = 0.$$
 (26)

- In case of  $\alpha = \beta = 1$ , we have  $\tilde{f}_t^* = f_t^{\theta}$ , i.e., the generator learns the corrupted function.
- In case of  $\alpha = \beta < 1$ , we have

$$\tilde{f}_t^*(x_t) = \frac{(1-\alpha)p_t^*(x_t)}{(1-\alpha)p_t^*(x_t) + \alpha p_t^{\theta}(x_t)} \cdot f_t^*(x_t) + \frac{\alpha p_t^{\theta}(x_t)}{(1-\alpha)p_t^*(x_t) + \alpha p_t^{\theta}(x_t)} \cdot f_t^{\theta}(x_t^*).$$

In this convex combination, the corrupted function  $\tilde{f}^*$  is always between the true teacher function  $f^*$  and the optimal generator function  $f^{\theta}$ , i.e., the generator learns even worse function.

• In case of  $\beta/\alpha \neq 1$ , there exist intervals of  $\alpha, \beta$  which can give better generator function than the corrupted teacher. For example, coefficients  $\alpha \neq \beta$  close to 1 allow to neglect the terms  $(1-\alpha)p_t^*(x_t)\cdot \tilde{f}_t^*(x_t)$  and  $(1-\beta)p_t^*(x_t)\cdot f_t^*(x_t)$  in (26) to get  $f_t^\theta(x_t)\approx \frac{\alpha}{\beta}\tilde{f}_t^*(x_t)$ . Hence, we can steer  $f^\theta$  towards the true teacher picking  $\beta/\alpha < 1$  or  $\beta/\alpha > 1$  depending on the corrupted and clean teacher's values. However, we cannot find all these intervals analytically due to complex distributions and functions.

Note that we derive the same recommendation  $\beta/\alpha \neq 1$  from the perspective of correcting the teacher's errors and from the perspective of the minimized distance surface from Appendix A.1.3.

**Visual illustration.** For visual demonstration, we consider the FM models transforming one-dimensional real data Gaussian  $\mathcal{N}(\mu^*,1)$  and generated Gaussian  $\mathcal{N}(\mu^\theta,1)$  to noise  $\mathcal{N}(0,1)$  on the time interval [0,1]. In this case, the generated and real data interpolations are  $p_t^\theta(x_t) = \mathcal{N}(x_t|\mu^\theta(1-t),t^2+(1-t)^2)$  and  $p_t^*(x_t) = \mathcal{N}(x_t|\mu^*(1-t),t^2+(1-t)^2)$ . The unconditional vector field u=f between  $\mathcal{N}(\mu,1)$  and  $\mathcal{N}(0,1)$  can be calculated as

$$u_{t}(x_{t}) = \mathbb{E}_{x_{0} \sim p_{0}(\cdot|x_{t})} \left[ \frac{x_{t} - x_{0}}{t} \right] = \int_{x_{0}} \left( \frac{x_{t} - x_{0}}{t} \right) \cdot \mathcal{N} \left( \frac{x_{t} - x_{0}(1 - t)}{t} | 0, 1 \right) \cdot \mathcal{N}(x_{0} | \mu, 1) dx_{0}$$

$$= \frac{a(2t^{2} - 2t) - bt^{2}}{\sqrt{2\pi}(1 - 2t + 2t^{2})^{\frac{3}{2}}} \exp\left( -\frac{(x_{t} - \mu(1 - t))^{2}}{2(1 - 2t + 2t^{2})^{2}} \right). \tag{27}$$

In Figure 4, we depict the optimal generator mean  $\mu^{\theta}$  and vector field  $u^{\theta}$  satisfying (26) for various deviations  $\tilde{u}^* - u^*$  and fixed time t = 1/3, real data  $\mu^* = -2$  and point  $x_t = -1$ .

We can see that with  $\alpha=\beta=1$ , the generator learns the corrupted vector field, and with  $\alpha=\beta<1$ , the learned field and means are often even worse. In contrast, with  $\beta/\alpha\neq 1$ , the generator can learn vector fields and means which are closer to the real data. Although the generator cannot satisfy relation (26) under large deviations, it still produces better results with the real data.

<span id="page-19-1"></span>![](_page_19_Figure_7.jpeg)

Figure 4: Learned generators for RealUID loss between 1D-Gaussians with corrupted teachers.

#### <span id="page-19-0"></span>A.2 GENERAL REALUID LOSS

**Extending our real data incorporation.** We recall that UID loss (Theorem 1) can be restated via linearization technique with  $\delta = f^* - f$  as:

$$\mathcal{L}_{\text{UID}}(\delta, p_0^{\theta}) = \mathbb{E}_{t \sim [0, T], x_0^{\theta} \sim p_0^{\theta}, \left\{ -\|\delta_t(x_t^{\theta})\|^2 + 2\langle \delta_t(x_t^{\theta}), f_t^*(x_t^{\theta}) \rangle - 2\langle \delta_t(x_t^{\theta}), f_t^{\theta}(x_t^{\theta}|x_0^{\theta}) \rangle \right\}.$$

Following alternative definition of RealUID loss from Lemma 3, one can incorporate real data into data-free loss by splitting each term in the linearized form between generated and real data as:

$$\begin{split} \mathcal{L}_{\text{R-UID}}^{\alpha,\beta}(\delta,p_{0}^{\theta}) &= \mathbb{E}_{t \sim [0,T],x_{0}^{\theta} \sim p_{0}^{\theta},} [-\alpha \|\delta_{t}(x_{t}^{\theta})\|^{2} + 2\alpha \langle \delta_{t}(x_{t}^{\theta}), f_{t}^{*}(x_{t}^{\theta}) \rangle - 2\beta \langle \delta_{t}(x_{t}^{\theta}), f_{t}^{\theta}(x_{t}^{\theta}|x_{0}^{\theta}) \rangle ] \\ & \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad$$

In RealUID loss (17), its three terms are split with proportions  $\alpha$  and  $1-\alpha$ ,  $\alpha$  and  $1-\alpha$  and  $\beta$  and  $1-\beta$ , respectively. We can go even further and split the first quadratic coefficient  $-\|\delta_t(\cdot)\|^2$  using a new parameter  $\gamma$  to create one more degree of freedom. Moreover, we can use other parameterization of  $\delta$ , since its form does not change the proof of distance lemma.

<span id="page-20-1"></span>**Definition 3.** We introduce General RealUID loss  $\mathcal{L}_{R\text{-}UID}^{\alpha,\beta,\gamma}(\delta,p_0^{\theta})$  on generated data  $p_0^{\theta} \in \mathcal{P}(\mathbb{R}^D)$  with coefficients  $\alpha,\beta,\gamma$ :

$$\mathcal{L}_{R\text{-}U\!I\!D}^{\alpha,\beta,\gamma}(\delta,p_0^{\theta}) := \mathbb{E}_{t \sim [0,T],x_0^{\theta} \sim p_0^{\theta}} [-\gamma \|\delta_t(x_t^{\theta})\|^2 + 2\alpha \langle \delta_t(x_t^{\theta}), f_t^*(x_t^{\theta}) \rangle - 2\beta \langle \delta_t(x_t^{\theta}), f_t^{\theta}(x_t^{\theta}|x_0^{\theta}) \rangle ] \\ + \mathbb{E}_{t \sim [0,T],x_0^{\theta} \sim p_0^{\theta}} [-(1-\gamma) \|\delta_t(x_t^*)\|^2 + 2(1-\alpha) \langle \delta_t(x_t^*), f_t^*(x_t^*) \rangle - 2(1-\beta) \langle \delta_t(x_t^*), f_t^*(x_t^*|x_0^*) \rangle ].$$

Optionally, one can change default parameterization  $\delta = f^* - f$  (e.g., with  $\delta = \beta(f^* - f)$ ), and substitute sampled real data term  $f_t^*(x_t^*|x_0^*)$  with the unconditional teacher  $f_t^*(x_t^*)$  and vice versa.

Theoretical properties. In case of  $\delta=f^*-f$  and  $\gamma\neq\alpha$ , the General RealUID loss cannot be expressed as inverse min-max problem (15) for simple losses, since some scalar products do not eliminate each other. Nevertheless, min-max optimization of  $\mathcal{L}_{\text{R-UID}}^{\alpha,\beta,\gamma}$  still minimizes the squared  $\ell_2$ -distance between the weighted teacher and generator functions, attaining minimum when  $p_0^\theta=p_0^*$ . Lemma 4 (Distance minimized by General RealUID loss). Maximization of General RealUID loss  $\mathcal{L}_{R\text{-UID}}^{\alpha,\beta,\gamma}$  over  $\delta$  represents the weighted squared  $\ell_2$ -distance between the student function  $f^\theta:=\arg\min_f \mathcal{L}_{\text{UM}}(f,p_0^\theta)$  and the teacher  $f^*:=\arg\min_f \mathcal{L}_{\text{UM}}(f,p_0^*)$ :

<span id="page-20-3"></span><span id="page-20-0"></span>
$$\max_{\delta} \mathcal{L}_{R\text{-}UID}^{\alpha,\beta,\gamma}(\delta, p_0^{\theta}) = \mathbb{E}_{t \sim [0,T], \atop x_t^* \sim p_t^*} \left[ \frac{\|\frac{\beta}{\alpha} [p_t^*(x_t^*) f_t^*(x_t^*) - p_t^{\theta}(x_t^*) f_t^{\theta}(x_t^*)] + (p_t^{\theta}(x_t^*) - p_t^*(x_t^*)) f_t^*(x_t^*) \|^2}{p_t^*(x_t^*) \cdot \max\{0, (1-\gamma) p_t^*(x_t^*) + \gamma p_t^{\theta}(x_t^*)\} / \alpha^2} \right].$$
(28)

The distances being minimized for RealUID (Lemma 2) and General RealUID (Lemma 4) are almost identical except the scale factor. Thus, we keep the same recommendations for choosing coefficients  $\alpha$ ,  $\beta$  as we discuss in Section 3.4. The factor  $\beta/\alpha$  still has the largest impact within the distance, while  $\alpha$  and  $\gamma$  set the scaling. Values  $\beta/\alpha$  and  $\gamma$  should be chosen close to 1, but not exactly 1.

*Proof.* First, we take math expectation over data points  $x_0^*$ . Since the expectation can be taken in a reverse order, i.e.,  $\mathbb{E}_{x_0^* \sim p_0^*, x_t^* \sim p_t^*(\cdot|x_0^*)} = \mathbb{E}_{x_t^* \sim p_t^*, x_0^* \sim p_0^*(\cdot|x_t^*)}$ , we see that

$$\mathbb{E}_{x_0^* \sim p_0^*, x_t^* \sim p_t^*(\cdot | x_0^*)} [\langle \delta_t(x_t^*), f_t^*(x_t^* | x_0^*) \rangle] = \mathbb{E}_{x_t^* \sim p_t^*} \langle \delta_t(x_t^*), \mathbb{E}_{x_0^* \sim p_0^*(\cdot | x_t^*)} [f_t^*(x_t^* | x_0^*)] \rangle \\
= \mathbb{E}_{x_t^* \sim p_t^*} [\langle \delta_t(x_t^*), f_t^*(x_t^*) \rangle]. \tag{29}$$

For the term  $\mathbb{E}_{x_0^{\theta} \sim p_0^{\theta}, x_t^{\theta} \sim p_t^{\theta}(\cdot|x_0^{\theta})}[\langle \delta_t(x_t^{\theta}), f_t^{\theta}(x_t^{\theta}|x_0^{\theta}) \rangle] = \mathbb{E}_{x_t^{\theta} \sim p_t^{\theta}}[\langle \delta_t(x_t^{\theta}), f_t^{\theta}(x_t^{\theta}) \rangle]$ , the reasoning is similar. Thus, we write down General RealUID loss (Def. 3) in an explicit form with  $\delta_t = f_t^* - f_t$ 

<span id="page-20-2"></span>
$$\mathcal{L}_{\text{R-UID}}^{\alpha,\beta}(\delta, p_0^{\theta}) = \mathbb{E}_{t \sim [0,T]} \mathbb{E}_{x_t^{\theta} \sim p_t^{\theta}} [-\gamma \|\delta_t(x_t^{\theta})\|^2 + 2\alpha \langle \delta_t(x_t^{\theta}), f_t^*(x_t^{\theta}) \rangle - 2\beta \langle \delta_t(x_t^{\theta}), f_t^{\theta}(x_t^{\theta}) \rangle] + \mathbb{E}_{t \sim [0,T]} \mathbb{E}_{x_t^* \sim p_t^*} [-(1-\gamma) \|\delta_t(x_t^*)\|^2 + 2(1-\alpha) \langle \delta_t(x_t^*), f_t^*(x_t^*) \rangle - 2(1-\beta) \langle \delta_t(x_t^*), f_t^*(x_t^*) \rangle].$$

Then, we rescale the generated data terms in the General RealUID loss using the equality  $p_t^{\theta}(x_t) = \frac{p_t^{\theta}(x_t)}{p_t^*(x_t)}p_t^*(x_t)$  for  $x_t \in \mathbb{R}^D$  (we assume  $p_t^*(x_t) > 0, \forall x_t, t$ ) leaving only math expectation w.r.t. the real data, i.e.

$$\mathcal{L}_{\text{R-UID}}^{\alpha,\beta,\gamma}(\delta, p_0^{\theta}) = \mathbb{E}_{\substack{t \sim [0,T], \\ x_t^* \sim p_t^*}} \left[ -[(1-\gamma) + \gamma \frac{p_t^{\theta}(x_t^*)}{p_t^*(x_t^*)}] \|\delta_t(x_t^*)\|^2 \right]$$

+ 
$$\mathbb{E}_{\substack{t \sim [0,T], \\ x_t^* \sim p_t^*}} \left[ 2[(\beta - \alpha) + \alpha \frac{p_t^{\theta}(x_t^*)}{p_t^*(x_t^*)}] \langle \delta_t(x_t^*), f_t^*(x_t^*) \rangle - 2\beta \frac{p_t^{\theta}(x_t^*)}{p_t^*(x_t^*)} \langle \delta_t(x_t^*), f_t^{\theta}(x_t^*) \rangle \right].$$

Next we maximize the loss w.r.t.  $\delta_t(x_t^*)$  for each  $x_t^*$  and t as a quadratic function. If  $(1-\gamma) \cdot p_t^*(x_t^*) + \gamma \cdot p_t^{\theta}(x_t^*) \leq 0$ , then the maximum tends to  $+\infty$ . Otherwise, the maximum is achieved when

$$\delta_t(x_t^*) = \frac{\left[ (\beta - \alpha) + \alpha \frac{p_t^{\theta}(x_t^*)}{p_t^*(x_t^*)} \right] f_t^*(x_t^*) - \beta \frac{p_t^{\theta}(x_t^*)}{p_t^*(x_t^*)} f_t^{\theta}(x_t^*)}{\left[ (1 - \gamma) + \gamma \frac{p_t^{\theta}(x_t^*)}{p_t^*(x_t^*)} \right]}.$$
(30)

The maximum itself equals to

$$\max_{\delta} \mathcal{L}_{\text{R-UID}}^{\alpha,\beta,\gamma}(\delta,p_0^{\theta}) = \mathbb{E}_{t \sim [0,T]} \mathbb{E}_{x_t^* \sim p_t^*} \left[ \frac{\|f_t^*(x_t^*) \cdot ((\beta - \alpha) + \alpha \frac{p_t^{\theta}(x_t^*)}{p_t^*(x_t^*)}) - f_t^{\theta}(x_t^*) \cdot \beta \frac{p_t^{\theta}(x_t^*)}{p_t^*(x_t^*)}\|^2}{(1 - \gamma) + \gamma \frac{p_t^{\theta}(x_t^*)}{p_t^*(x_t^*)}} \right].$$

**Alternative parameterization.** In the proximity of the solution, when generated data approaches real one, i.e.,  $p_t^{\theta} \approx p_t^*$ , the optimal  $\delta_t$  (30) approaches

$$\delta_t(x_t^*) \approx \frac{[(\beta - \alpha) + \alpha \cdot 1]f_t^*(x_t^*) - \beta \cdot 1 \cdot f_t^{\theta}(x_t^*)}{[(1 - \gamma) + \gamma \cdot 1]} \approx \beta(f_t^*(x_t^*) - f_t^{\theta}(x_t^*)).$$

Thus, the parameterization  $\delta_t = \beta(f_t^* - f_t)$  may naturally help reach the solution without making the fake model learn extra information about the teacher near the optimum.

In experiments in Tables 1 and 7, this parameterization with the corresponding coefficients  $\gamma = \alpha$  and  $\beta$  yields slightly better metrics from +0.02 to +0.04.

Extra ranges for coefficients  $\alpha, \beta, \gamma$  New perspective on our RealUID loss allows us to expand the range of feasible configurations for the parameters  $\alpha, \beta$ , and  $\gamma$ . Specifically, it is now possible to set  $\alpha = 1$  for any  $\beta$ , whereas in the original loss (16) this configuration is unavailable due to division by zero in the real data term. Additionally, one can now use values  $\alpha, \beta, \gamma > 1$ .

However, we observe that in the experiments reported in Tables 1 and 7, these extra configurations are highly unstable and lead to degraded results. This degradation occurs due to out-of-domain generated samples and negative quadratic summands leading to infinite losses and metric (28). Hence, we stick to the original ranges  $\alpha, \beta, \gamma \in (0, 1]$ .

### <span id="page-21-0"></span>A.3 GENERAL SID WITH REAL DATA

Our real data incorporation. We recall that data-free UID loss (Theorem 1) can be restated via linearization technique with  $\delta = f - f^*$  as:

<span id="page-21-1"></span>
$$\mathcal{L}_{\text{UID}}(\delta, p_0^{\theta}) = \mathbb{E}_{t \sim [0, T], x_0^{\theta} \sim p_0^{\theta}, \left[ -\|\delta_t(x_t^{\theta})\|^2 + 2\langle \delta_t(x_t^{\theta}), f_t^*(x_t^{\theta}) \rangle - 2\langle \delta_t(x_t^{\theta}), f_t^{\theta}(x_t^{\theta}|x_0^{\theta}) \rangle \right]. \tag{31}$$

Following alternative definition of our RealUID loss from Lemma 3, one can incorporate real data into data-free loss by splitting each term in the linearized form between generated and real data as:

$$\mathcal{L}_{\text{R-UID}}^{\alpha,\beta}(\delta, p_{0}^{\theta}) = \mathbb{E}_{t \sim [0,T], x_{0}^{\theta} \sim p_{0}^{\theta}, [-\alpha \|\delta_{t}(x_{t}^{\theta})\|^{2} + 2\alpha \langle \delta_{t}(x_{t}^{\theta}), f_{t}^{*}(x_{t}^{\theta}) \rangle - 2\beta \langle \delta_{t}(x_{t}^{\theta}), f_{t}^{\theta}(x_{t}^{\theta}|x_{0}^{\theta}) \rangle] 
+ \mathbb{E}_{t \sim [0,T], x_{0}^{*} \sim p_{0}^{*}, [-(1-\alpha) \|\delta_{t}(x_{t}^{*})\|^{2} + 2(1-\alpha) \langle \delta_{t}(x_{t}^{*}), f_{t}^{*}(x_{t}^{*}) \rangle - 2(1-\beta) \langle \delta_{t}(x_{t}^{*}), f_{t}^{*}(x_{t}^{*}|x_{0}^{*}) \rangle].$$
(32)

<span id="page-21-2"></span>**General data-free SiD.** The authors of the SiD framework (Zhou et al., 2024a;b) for diffusion models empirically notice that scaling the first coefficient  $-\|\delta_t(x_t^\theta)\|^2$  by the factor  $2\alpha_{\text{SiD}}$  in the UID loss (31) for generator updates yields better performance. Hence, we generalize the SiD loss to

other matching models. Namely, the **General SiD loss** for the generator is the following loss with  $\delta = f - f^*$  and parameter  $\alpha_{SiD} \in [0.5, 1.2]$ :

<span id="page-22-0"></span>
$$\mathcal{L}_{\text{SiD}}(\theta) := \mathbb{E}_{t \sim [0,T], x_0^{\theta} \sim p_0^{\theta}, \left[ -2\alpha_{\text{SiD}} \|\delta_t(x_t^{\theta})\|^2 + 2\langle \delta_t(x_t^{\theta}), f_t^*(x_t^{\theta}) \rangle - 2\langle \delta_t(x_t^{\theta}), f^{\theta}(x_t^{\theta}|x_0^{\theta}) \rangle \right],$$

$$(32)$$

while the UM loss (Def. 1) for the fake model remains intact. The same positive effect is observed in experiments with flow matching models in FGM (Huang et al., 2024), where the authors do not calculate the gradient through some loss terms and obtain the General SiD loss (33) with  $\alpha_{\text{SiD}} = 1$ , achieving better performance.

General SiD with real data. Following the structure of the General SiD loss (33), we propose to scale the first coefficient in our RealUID loss (32) during generator updates. The whole **General SiD pipeline with real data (RealSiD)**, defined by coefficients  $\alpha, \beta \in (0, 1], \alpha_{\text{SiD}} \in [0.5, 1.2]$  and teacher  $f^*$ , is two alternating steps:

1. Make one or several fake model f update steps, minimizing UM loss with real data  $\mathcal{L}_{R-UM}^{\alpha,\beta}(f,p_0^{\theta})$ :

<span id="page-22-1"></span>
$$L_{\text{R-UM}}^{\alpha,\beta}(f,p_0^{\theta}) \ := \ \underbrace{\alpha \cdot \mathbb{E}_{t \sim [0,T]} \mathbb{E}_{x_0^{\theta} \sim p_0^{\theta}, x_t^{\theta} \sim p_t^{\theta}(\cdot|x_0^{\theta})} \left[ \|f_t(x_t^{\theta}) - \frac{\beta}{\alpha} f_t^{\theta}(x_t^{\theta}|x_0^{\theta})\|^2 \right]}_{\text{generated data } p_0^{\theta} \text{ term}} \\ + \ \underbrace{(1-\alpha) \cdot \mathbb{E}_{t \sim [0,T]} \mathbb{E}_{x_0^* \sim p_0^*, x_t^* \sim p_t^*(\cdot|x_0^*)} \left[ \|f_t(x_t^*) - \frac{1-\beta}{1-\alpha} f_t^*(x_t^*|x_0^*)\|^2 \right]}_{\text{real data } p_0^* \text{ term}}.$$

2. Make a generator update step, minimizing the loss  $\mathcal{L}_{R-SiD}^{\alpha,\beta}(\theta) :=$ 

$$\mathbb{E}_{t \sim [0,T], x_0^{\theta} \sim p_0^{\theta}, \left[-2\alpha_{\text{SiD}} \cdot \alpha \|\delta_t(x_t^{\theta})\|^2 + 2\alpha \langle \delta_t(x_t^{\theta}), f_t^*(x_t^{\theta}) \rangle - 2\beta \langle \delta_t(x_t^{\theta}), f_t^{\theta}(x_t^{\theta}|x_0^{\theta}) \rangle \right], \quad (34)}$$

where  $\delta_t = f_t - f_t^*$ .

In the SiD framework for diffusion models, the data-free generator SiD loss (33) is additionally normalized, and the SiD loss with real data (34) should be normalized the same way. For more details on normalization, time sampling, weighting, etc., refer to the original articles (Zhou et al., 2024a;b).

**Experimental validation.** We modify the data-free SiD loss in the official SiD implementation with real data and conduct a short ablation study on unconditional CIFAR-10. The SiD codebase for diffusion models can be found in

We compare the data-free SiD loss (33) and our RealSiD loss (34) with the best coefficients  $\alpha$ ,  $\beta$  from Table 1 for both the theoretically justified  $\alpha_{\text{SiD}} = 0.5$  and the best practical heuristic  $\alpha_{\text{SiD}} = 1.2$ . We do not change anything else and use the default hyperparameters and training pipeline as described in (Zhou et al., 2024b). The results are presented in Figure 5.

For the accurate  $\alpha_{SiD}=0.5$ , the RealSiD results for diffusion models are similar to those for flow models. Configurations with  $\beta/\alpha=1.02$  boost convergence compared to the data-free baseline ( $\alpha=\beta=1$ ), whereas in the case of  $\alpha=\beta\neq 1$ , the convergence speed remains close to the baseline.

However, for the heuristic  $\alpha_{SiD}=1.2$ , our best configurations with  $\beta/\alpha\neq 1.0$  either degrade performance compared to the baseline or become unstable. Thus, the heuristical SiD may require a different approach to incorporate real data, or a more careful tuning of the coefficients  $\alpha, \beta$  and other hyperparameters, due to differing architectures and training pipelines.

We would like to highlight that all our analyses and recommendations were justified only for  $\alpha_{SiD} = 0.5$ . For other  $\alpha_{SiD}$  values, this justification may not hold true.

<span id="page-23-1"></span>![](_page_23_Figure_1.jpeg)

<span id="page-23-3"></span>![](_page_23_Figure_2.jpeg)

Figure 5: Evolution of FID during unconditional CIFAR-10 distillation for the data-free SiD loss ( $\alpha = \beta = 1.0$ ) and our RealSiD loss for  $\alpha_{\text{SiD}} = 0.5$  (left) and  $\alpha_{\text{SiD}} = 1.2$  (right).

### <span id="page-23-0"></span>A.4 Normalized UID and RealUID losses for minimizing $\ell_2$ -distance

Using the linearization technique from (§3.1), we can estimate the non-squared  $\ell_2$ -distance between the teacher  $f^* := \arg\min_f \mathcal{L}_{\text{UM}}(f, p_0^*)$  and student  $f^\theta := \arg\min_f \mathcal{L}_{\text{UM}}(f, p_0^\theta)$  functions. In this case, the connection with the inverse optimization disappears.

For a fixed point  $x_t^{\theta} \sim p_t^{\theta}$  and time  $t \sim [0, T]$ , we derive:

$$\begin{aligned} \|f_t^*(x_t^{\theta}) - f_t^{\theta}(x_t^{\theta})\| &= \max_{\delta_t(x_t^{\theta})} \left\{ \langle \frac{\delta_t(x_t^{\theta})}{\|\delta_t(x_t^{\theta})\|}, f_t^*(x_t^{\theta}) - f_t^{\theta}(x_t^{\theta}) \rangle \right\} \\ &= \max_{\delta_t(x_t^{\theta})} \mathbb{E}_{x_0^{\theta} \sim p_0^{\theta}(\cdot|x_t^{\theta})} \left[ \langle \frac{\delta_t(x_t^{\theta})}{\|\delta_t(x_t^{\theta})\|}, f_t^*(x_t^{\theta}) \rangle - \langle \frac{\delta_t(x_t^{\theta})}{\|\delta_t(x_t^{\theta})\|}, f_t^{\theta}(x_t^{\theta}|x_0^{\theta}) \rangle \right]. \end{aligned} (35)$$

With the parameterization  $\delta_t = f_t^* - f_t$ , the **Normalized UID loss**  $\hat{\mathcal{L}}_{\text{UID}}(f, p_0^{\theta})$  for solving  $\min_{\theta} \mathbb{E}_{t \sim [0,T]} \mathbb{E}_{x_t^{\theta} \sim p_t^{\theta}}[||f_t^*(x_t^{\theta}) - f_t^{\theta}(x_t^{\theta})||]$  is

<span id="page-23-2"></span>
$$\min_{\theta} \max_{f} \left\{ \hat{\mathcal{L}}_{\text{UID}}(f, p_{0}^{\theta}) := \mathbb{E}_{\substack{t \sim [0, T], x_{0}^{\theta} \sim p_{0}^{\theta}, \\ x_{t}^{\theta} \sim p_{t}^{\theta}(\cdot | x_{0}^{\theta})}} \left[ \left\langle \frac{f_{t}^{*}(x_{t}^{\theta}) - f_{t}(x_{t}^{\theta})}{\|f_{t}^{*}(x_{t}^{\theta}) - f_{t}(x_{t}^{\theta})\|}, f_{t}^{*}(x_{t}^{\theta}) - f_{t}^{\theta}(x_{t}^{\theta} | x_{0}^{\theta}) \right\rangle \right] \right\}. (36)$$

**Adding real data.** Following alternative definition of RealUID loss from Lemma 3, we can incorporate real data in Normalized UID loss (36) as well. We need to split two terms in the linearized form (35) into generated and real data parts with weights  $\alpha$ ,  $(1 - \alpha)$  and  $\beta$ ,  $(1 - \beta)$ .

**Definition 4.** We introduce Normalized RealUID loss  $\hat{\mathcal{L}}_{R\text{-UID}}^{\alpha,\beta}(f,p_0^{\theta})$  on generated data  $p_0^{\theta} \in \mathcal{P}(\mathbb{R}^D)$  with coefficients  $\alpha,\beta \in (0,1]$ :

<span id="page-23-4"></span>
$$\begin{split} \hat{\mathcal{L}}_{R\text{-}UID}^{\alpha,\beta}(f,p_0^{\theta}) &:= \mathbb{E}_{t \sim [0,T], x_0^{\theta} \sim p_0^{\theta},} \left[ \langle \frac{f_t^*(x_t^{\theta}) - f_t(x_t^{\theta})}{\|f_t^*(x_t^{\theta}) - f_t(x_t^{\theta})\|}, \alpha \cdot f_t^*(x_t^{\theta}) - \beta \cdot f_t^{\theta}(x_t^{\theta}|x_0^{\theta}) \rangle \right] \\ &+ \mathbb{E}_{t \sim [0,T], x_0^{\theta} \sim p_0^{\theta},} \left[ \langle \frac{f_t^*(x_t^*) - f_t(x_t^*)}{\|f_t^*(x_t^*) - f_t(x_t^*)\|}, (1 - \alpha) \cdot f_t^*(x_t^*) - (1 - \beta) \cdot f_t^*(x_t^*|x_0^*) \rangle \right]. \end{split}$$

Similar to the proof of RealUID distance Lemma 2, we can show that min-max optimization of Normalized RealUID loss minimizes the non-squared  $\ell_2$ -norm between the similar weighted student  $f^{\theta}$  and teacher  $f^*$  functions:

$$\max_{f} \hat{\mathcal{L}}_{\text{R-UID}}^{\alpha,\beta}(f, p_0^{\theta}) = \mathbb{E}_{t \sim [0,T]} \mathbb{E}_{x_t^* \sim p_t^*} \left[ \| ((\beta - \alpha) + \alpha \frac{p_t^{\theta}(x_t^*)}{p_t^*(x_t^*)}) \cdot f_t^*(x_t^*) - \beta \frac{p_t^{\theta}(x_t^*)}{p_t^*(x_t^*)} \cdot f_t^{\theta}(x_t^*) \| \right].$$

This distance attains minimum when  $p_0^{\theta} = p_0^*$ , justifying the procedure.

### <span id="page-24-0"></span>A.5 DMD AND INVERSE OPTIMIZATION

**Distribution Matching Distillation** (Luo et al., 2023; Wang et al., 2023; Yin et al., 2024b;a, **DMD**) approach distills Gaussian diffusion models with forward process  $x_t = x_0 + \sigma_t \epsilon$ ,  $\epsilon \sim \mathcal{N}(0, I)$ .

This approach minimizes KL divergence  $\mathbb{E}_{t \sim [0,T]}[D_{\mathrm{KL}}(p_t^{\theta}||p_t^*)] = \mathbb{E}_{t \sim [0,T]}\mathbb{E}_{x_t^{\theta} \sim p_t^{\theta}}\left[\log\left(\frac{p_t^{\theta}(x_t^{\theta})}{p_t^*(x_t^{\theta})}\right)\right]$ 

between the generated data  $p_t^{\theta}$  and the real data  $p_t^*$ . The authors meticulously derive that the true gradient of  $\mathbb{E}_{t \sim [0,T]}[D_{\mathrm{KL}}(p_t^{\theta}||p_t^*)]$  w.r.t.  $\theta$  can be computed via the score functions:

$$\mathbb{E}_{t \sim [0,T]} \left[ \frac{dD_{\mathrm{KL}}(p_t^{\theta} || p_t^*)}{d\theta} \right] = \mathbb{E}_{\substack{t \sim [0,T], z \sim p^{\mathcal{Z}}, \\ x_\theta^{\theta} = G(z), x_t^{\theta} \sim p_t^{\theta}}} \left[ \left( \nabla_{x_t^{\theta}} \log p_t^{sg[\theta]}(x_t^{\theta}) - \nabla_{x_t^{\theta}} \log p_t^*(x_t^{\theta}) \right) \frac{dG_{\theta}(z)}{d\theta} \right].$$

Then, this true gradient is estimated with the teacher score function  $s^* := \arg\min_s \mathcal{L}_{\text{DSM}}(s, p_0^*)$  and student score  $s^\theta = \arg\min_s \mathcal{L}_{\text{DSM}}(s, p_0^\theta)$  obtained via minimizing DSM loss (1):

$$\mathbb{E}_{t \sim [0,T]} \left[ \frac{dD_{\mathrm{KL}}(p_t^{\theta} || p_t^*)}{d\theta} \right] = \mathbb{E}_{\substack{t \sim [0,T], z \sim p^{\mathcal{Z}}, \\ x_{\theta}^{\theta} = G(z), x_{t}^{\theta} \sim p_{t}^{\theta}}} \left[ \left( s_t^{\theta}(x_t^{\theta}) - s_t^*(x_t^{\theta}) \right) \frac{dG_{\theta}}{d\theta} \right].$$

The final algorithm alternates updates for the fake model and the generator similar to SiD approach.

Below we show that DMD fits our UID framework, but with another loss, different from the UM loss. In the case of diffusions, the UM loss is the  $\mathcal{L}_{DSM}(s,p_0^\theta)$  loss, and with this loss, the resulting UID loss becomes exactly the SiD loss, not the DMD.

**Inverse optimization view.** We decompose KL divergence as the difference of two KL losses:

$$\mathbb{E}_{t \sim [0,T]} \left[ D_{\mathrm{KL}}(p_t^{\theta} || p_t^*) \right] = \mathbb{E}_{t \sim [0,T], \left[ \log \left( p_t^{\theta}(x_t^{\theta}) \right) \right]} - \mathbb{E}_{t \sim [0,T], \left[ \log p_t^*(x_t^{\theta}) \right]}$$

$$= \mathcal{L}_{\mathrm{KL}}(p_t^{\theta}, p_t^{\theta}) - \mathcal{L}_{\mathrm{KL}}(p_t^*, p_t^{\theta}),$$

where  $\mathcal{L}_{\mathrm{KL}}(q_t, p_t) := \mathbb{E}_{t \sim [0, T], x_t \sim p_t} [\log q_t(x_t)]$ . We can differentiate through the generated samples  $x_t^{\theta} \sim p_t^{\theta}$  from the second arguments of  $\mathcal{L}_{\mathrm{KL}}$  losses; however, the term  $\mathcal{L}_{\mathrm{KL}}(p_t^{\theta}, p_t^{\theta}) = \mathbb{E}_{t \sim [0, T], x_t^{\theta} \sim p_t^{\theta}} \left[\log \left(p_t^{\theta}(x_t^{\theta})\right)\right]$  remains intractable as it involves explicit generated data density  $\log(p_t^{\theta}(\cdot))$  from the first argument. To make this term tractable, we apply the *linearization trick*. Due to non-negativity of KL divergence, we have the relation:

$$\mathbb{E}_{t \sim [0,T]}\left[D_{\mathrm{KL}}(p_t||q_t)\right] = \mathcal{L}_{\mathrm{KL}}(p_t,p_t) - \mathcal{L}_{\mathrm{KL}}(q_t,p_t) \geq 0, \forall q_t \Rightarrow \mathcal{L}_{\mathrm{KL}}(p_t,p_t) = \max_{q_t} \mathcal{L}_{\mathrm{KL}}(q_t,p_t),$$

where maximum is attained when  $q_t = p_t$ . Thus, we substitute the intractable term  $\mathcal{L}_{KL}(p_t^{\theta}, p_t^{\theta})$  with the maximization problem parametrized by the *fake* distribution  $q_t$ :

$$\min_{\theta} \mathbb{E}_{t \sim [0,T]} \left[ D_{\text{KL}}(p_t^{\theta} || p_t^*) \right] = \min_{\theta} \{ \mathcal{L}_{\text{KL}}(p_t^{\theta}, p_t^{\theta}) - \mathcal{L}_{\text{KL}}(p_t^*, p_t^{\theta}) \} 
= \min_{\theta} \{ \underbrace{\max_{q_t} \{ \mathcal{L}_{\text{KL}}(q_t, p_t^{\theta}) \} - \mathcal{L}_{\text{KL}}(p_t^*, p_t^{\theta}) \}}_{>0} \} = \min_{\theta} \max_{q_t} \{ \mathcal{L}_{\text{KL}}(q_t, p_t^{\theta}) - \mathcal{L}_{\text{KL}}(p_t^*, p_t^{\theta}) \}.$$
(37)

Our min-max formulation of DMD (37) is the special case of the inverse optimization scheme (15) with the KL loss and teacher  $p_t^* = \arg\max_{q_t} \mathcal{L}_{\mathrm{KL}}(q_t, p_t^*)$ . The generated data density  $p_t^\theta$  appears only during sampling and we can efficiently backpropagate through it. Next, we easily calculate the gradient w.r.t. generator parameters, since we do not need to differentiate through independent  $q_t$  which is equal to  $p_t^{sg[\theta]}$  in the optimum:

$$\begin{split} \mathbb{E}_{t \sim [0,T]} \left[ \frac{dD_{\text{KL}}(p_t^{\theta} || p_t^*)}{d\theta} \right] &= \frac{d}{d\theta} [\mathcal{L}_{\text{KL}}(p_t^{sg[\theta]}, p_t^{\theta})] - \frac{d}{d\theta} [\mathcal{L}_{\text{KL}}(p_t^*, p_t^{\theta})] \\ &= \mathbb{E}_{\substack{t \sim [0,T], z \sim p^{\mathcal{Z}}, \\ x_0^{\theta} = G(z), x_t^{\theta} \sim p_t^{\theta}}} \left[ \frac{d}{d\theta} [\log p_t^{sg[\theta]}(x_t^{\theta})] - \frac{d}{d\theta} [\log p_t^*(x_t^{\theta})] \right] \\ &= \mathbb{E}_{\substack{t \sim [0,T], z \sim p^{\mathcal{Z}}, \\ x_0^{\theta} = G(z), x_t^{\theta} \sim p_t^{\theta}}} \left[ \underbrace{\left( \nabla_{x_t^{\theta}} \log p_t^{sg[\theta]}(x_t^{\theta}) - \nabla_{x_t^{\theta}} \log p_t^*(x_t^{\theta}) \right)}_{=s_t^*(x_t^{\theta})} \frac{dG_{\theta}(z)}{d\theta} \right], \end{split}$$

where teacher score  $s^* := \arg\min_s \mathcal{L}_{DSM}(s, p_0^*)$  and fake score  $s^\theta = \arg\min_s \mathcal{L}_{DSM}(s, p_0^\theta)$  are obtained via minimizing DSM loss (1) instead of maximizing the KL loss directly.

**Adding real data.** Following the logic from §3.4, we can modify the data-free KL loss  $\mathcal{L}_{\text{KL}}(q_t, p_t) = \mathbb{E}_{t \sim [0,T], x_t \sim p_t} [\log q_t(x_t)]$  with the real data and put it back into the inverse optimization scheme (37). We propose the following modified loss with a single parameter  $\alpha \in (0,1]$ :

$$\mathcal{L}_{\text{R-KL}}^{\alpha}(q_{t}, p_{t}^{\theta}) := \alpha \cdot \mathcal{L}_{\text{KL}}(q_{t}, p_{t}^{\theta}) + (1 - \alpha) \cdot \mathcal{L}_{\text{KL}}(q_{t}, p_{t}^{*})$$

$$= \underbrace{\alpha \cdot \mathbb{E}_{t \sim [0, T], x_{t}^{\theta} \sim p_{t}^{\theta}} \left[ \log q_{t}(x_{t}^{\theta}) \right]}_{\text{generated data } p_{t}^{\theta} \text{ term}} + \underbrace{(1 - \alpha) \cdot \mathbb{E}_{t \sim [0, T], x_{t}^{*} \sim p_{t}^{*}} \left[ \log q_{t}(x_{t}^{*}) \right]}_{\text{real data } p_{t}^{*} \text{ term}}$$

$$= \mathcal{L}_{\text{KL}}(q_{t}, \alpha \cdot p_{t}^{\theta} + (1 - \alpha) \cdot p_{t}^{*}). \tag{38}$$

With this loss structure, the teacher distribution is preserved:  $p_t^* = \arg\max_{q_t} \mathcal{L}_{R-KL}^{\alpha}(q_t, p_t^*) = \arg\max_{q_t} \mathcal{L}_{KL}(q_t, p_t^*)$ . The KL loss  $\mathcal{L}_{KL}(q_t, p_t^{\theta})$  is linear in  $p_t^{\theta}$ , thus, adding extra  $\beta$  coefficient as we did in UM loss (16) is not working:

$$\alpha \cdot \mathcal{L}_{\mathrm{KL}}(q_t, \frac{\beta}{\alpha} p_t^{\theta}) + (1 - \alpha) \cdot \mathcal{L}_{\mathrm{KL}}(q_t, \frac{1 - \beta}{1 - \alpha} p_t^*) = \beta \cdot \mathcal{L}_{\mathrm{KL}}(q_t, p_t^{\theta}) + (1 - \beta) \cdot \mathcal{L}_{\mathrm{KL}}(q_t, p_t^*).$$

Putting the  $\mathcal{L}_{R-KL}$  loss back into the inverse scheme (37), we explicitly obtain the KL divergence which minimizes the difference between the real data, and the mix of real and generated data:

<span id="page-25-0"></span>
$$\min_{\theta} \max_{q_t} \{ \mathcal{L}_{R-KL}^{\alpha}(q_t, p_t^{\theta}) - \mathcal{L}_{R-KL}^{\alpha}(p_t^*, p_t^{\theta}) \} 
= \min_{\theta} \max_{q_t} \{ \mathcal{L}_{KL}(q_t, \alpha \cdot p_t^{\theta} + (1 - \alpha) \cdot p_t^*) - \mathcal{L}_{KL}(p_t^*, \alpha \cdot p_t^{\theta} + (1 - \alpha) \cdot p_t^*) \} 
= \min_{\theta} \{ \mathcal{L}_{KL}(\alpha \cdot p_t^{\theta} + (1 - \alpha) \cdot p_t^*, \alpha \cdot p_t^{\theta} + (1 - \alpha) \cdot p_t^*) - \mathcal{L}_{KL}(p_t^*, \alpha \cdot p_t^{\theta} + (1 - \alpha) \cdot p_t^*) \} 
= \min_{\theta} \mathbb{E}_{t \sim [0, T]} \left[ D_{KL}(\alpha \cdot p_t^{\theta} + (1 - \alpha) \cdot p_t^*) | p_t^*) \right].$$
(39)

Similarly, we calculate the gradient over generator parameters of our modified DMD (39):

$$\begin{split} & \mathbb{E}_{t \sim [0,T]} \left[ \frac{dD_{\text{KL}}(\alpha \cdot p_t^{\theta} + (1 - \alpha) \cdot p_t^* || p_t^*)}{d\theta} \right] \\ & = \frac{d}{d\theta} [\mathcal{L}_{\text{KL}}(\alpha \cdot p_t^{sg[\theta]} + (1 - \alpha) \cdot p_t^*, \alpha \cdot p_t^{\theta} + (1 - \alpha) \cdot p_t^*)] - \frac{d}{d\theta} [\mathcal{L}_{\text{KL}}(p_t^*, \alpha \cdot p_t^{\theta} + (1 - \alpha) \cdot p_t^*)] \\ & = \alpha \cdot \mathbb{E}_{\substack{t \sim [0,T], z \sim p^Z, \\ x_0^{\theta} = G(z), x_t^{\theta} \sim p_t^{\theta}}} \left[ \frac{d}{d\theta} [\log(\alpha \cdot p_t^{sg[\theta]}(x_t^{\theta}) + (1 - \alpha) \cdot p_t^*(x_t^{\theta}))] - \frac{d}{d\theta} [\log p_t^*(x_t^{\theta})] \right] \\ & = \alpha \cdot \mathbb{E}_{\substack{t \sim [0,T], z \sim p^Z, \\ x_0^{\theta} = G(z), x_t^{\theta} \sim p_t^{\theta}}} \left[ \underbrace{(\nabla_{x_t^{\theta}} \log(\alpha \cdot p_t^{sg[\theta]}(x_t^{\theta}) + (1 - \alpha) \cdot p_t^*(x_t^{\theta}))}_{=s_t^{\theta} \cap \alpha} - \underbrace{\nabla_{x_t^{\theta}} \log p_t^*(x_t^{\theta})}_{=s_t^{\theta} \cap \alpha} \underbrace{\frac{dG_{\theta}(z)}{d\theta}} \right]. \end{split}$$

The score  $s_t^{\theta,\alpha}$  of the mixed data  $\alpha \cdot p_t^{\theta} + (1-\alpha) \cdot p_t^*$  can be found using the DSM loss modified with the real data:

<span id="page-25-1"></span>
$$\mathcal{L}_{\text{R-DSM}}^{\alpha}(s, p_0^{\theta}) \quad := \quad \underbrace{\alpha \cdot \mathbb{E}_{t \sim [0, T]} \mathbb{E}_{x_0^{\theta} \sim p_0^{\theta}, x_t^{\theta} \sim p_t^{\theta}(\cdot|x_0)} \left[ \gamma_t \| s_t(x_t^{\theta}) - s^{\theta}(x_t^{\theta}|x_0^{\theta}) \|^2 \right] }_{\text{generated data } p_0^{\theta} \text{ term}} \\ + \quad \underbrace{\left( 1 - \alpha \right) \cdot \mathbb{E}_{t \sim [0, T]} \mathbb{E}_{x_0^* \sim p_0^*, x_t^* \sim p_t^*(\cdot|x_0^*)} \left[ \gamma_t \| s_t(x_t^*) - s_t^*(x_t^*|x_0^*) \|^2 \right] }_{\text{real data } p_0^* \text{ term}} .$$

We sum up all conclusions in the lemma below.

**Lemma 5** (DMD with real data). Consider real data  $p_0^* \in \mathcal{P}(\mathbb{R}^D)$  and generated data  $p_0^\theta \in \mathcal{P}(\mathbb{R}^D)$ . Then, KL divergence between mixed and real data for  $\alpha \in (0,1]$  has the following gradient with modified student score  $s_t^{\theta,\alpha} := \arg\min_s \mathcal{L}_{R\text{-DSM}}^{\alpha}(s,p_0^{\theta})$  and teacher score  $s_t^* := \arg\min_s \mathcal{L}_{DSM}(s,p_0^*)$ :

$$\mathbb{E}_{t \sim [0,T]} \left[ \frac{dD_{\mathit{KL}}(\alpha \cdot p_t^{\theta} + (1-\alpha) \cdot p_t^* || p_t^*)}{d\theta} \right] = \mathbb{E}_{\substack{t \sim [0,T], z \sim p^{\mathcal{Z}}, \\ x_\theta^{\theta} = G_{\theta}(z), x_t^{\theta} \sim p_t^{\theta}}} \left[ \alpha(s_t^{\theta,\alpha}(x_t^{\theta}) - s_t^*(x_t^{\theta})) \frac{dG_{\theta}}{d\theta} \right].$$

Even though this approach is theoretically justified, it requires coefficients  $\alpha = \beta$  which work poorly for our RealUID; see Table 1.

**Extension to Bregman divergences.** KL divergence belongs to a large family of distance functions called Bregman divergences. A divergence  $D_{\Psi}(p_t^{\theta}, p_t^*)$  is determined by a convex function  $\Psi$ :

$$D_{\Psi}(p_t^{\theta}, p_t^*) := \Psi(p_t^{\theta}) - \Psi(p_t^*) - \langle \nabla \Psi(p_t^*), p_t^{\theta} - p_t^* \rangle \ge 0.$$
 (40)

For KL divergence, we have the function  $\Psi(p)=\int \log(p(x))p(x)dx$ . In (40), the only non-linear and intractable term w.r.t.  $p_t^\theta$  is the function  $\Psi(p_t^\theta)$ , but we can similarly make it tractable via the linearization trick:

$$\Psi(p_t^{\theta}) = \max_{q_t} \{ \Psi(q_t) + \langle \nabla \Psi(q_t), p_t^{\theta} - q_t \rangle \},$$

and build a general inverse distillation scheme:

$$\min_{\theta} \mathbb{E}_{t}[D_{\Psi}(p_{t}^{\theta}, p_{t}^{*})] = \min_{\theta} \max_{q_{t}} \{\underbrace{\mathbb{E}_{t}[\Psi(q_{t}) + \langle \nabla \Psi(q_{t}), p_{t}^{\theta} - q_{t} \rangle]}_{=:\mathcal{L}_{\Psi}(q_{t}, p_{t}^{\theta})} - \underbrace{\mathbb{E}_{t}[\Psi(p_{t}^{*}) + \langle \nabla \Psi(p_{t}^{*}), p_{t}^{\theta} - p_{t}^{*} \rangle]}_{=:\mathcal{L}_{\Psi}(p_{t}^{*}, p_{t}^{\theta})} \}.$$

### <span id="page-27-0"></span>REALUID ALGORITHM FOR FLOW MATCHING MODELS

We provide a practical implementation of our RealUID approach for flow matching models in Algorithm 1. In the loss functions, we retain only the terms dependent on the target parameters. For the fake model, we reformulate the maximization objective as a minimization. We use alternating optimization, updating the fake model K times per one student update for stability.

### <span id="page-27-1"></span>Algorithm 1 Real data modified Unified Inversion Distillation (RealUID) for Flow Matching

**Input:** teacher drift  $u^*$ , student generator  $G_\theta$ , fake drift  $u_\psi$ , real data  $p_0^*$ , coefficients  $\alpha, \beta \in (0, 1]$ , generator update steps K, number of iterations N, batch size B, fake drift minimizer  $Opt_{st}$ , generator minimizer  $Opt_{gen}$ , latent distribution  $p^{\mathcal{Z}}$ , noise distribution  $p_1$ .

- 1: **for** n = 0, ..., N-1 **do**
- Sample generated batch  $\{x_{0,i}^{\theta}=G_{\theta}(z_i)\}_{i=1}^{B}, z_i \sim p^{\mathcal{Z}} \text{ and noise batch } \{x_{1,i}\}_{i=1}^{B} \sim p_1;$
- Sample time batch  $\{t_i\}_{i=1}^B \sim Uniform[0,1]$  and calculate  $x_{t_i,i}^\theta = (1-t_i)x_{0,i}^\theta + t_ix_{1,i}$ ; 3:
- if student step  $(n\%(K+1) \neq 0)$  then
- Sample real data batch  $\{x_{0,i}^*\}_{i=1}^B \sim p_0^*$  and calculate  $x_{t_i,i}^* = (1-t_i)x_{0,i}^* + t_i x_{1,i};$  Update fake drift parameters  $\psi$  via minimizer  $Opt_{st}$  step with gradients of
- 6:

$$\frac{1}{B} \sum_{i=1}^{B} \left[ \alpha \|u_{\psi}(t_{i}, x_{t_{i}, i}^{sg[\theta]}) - \frac{\beta}{\alpha} (x_{1, i} - x_{0, i}^{sg[\theta]}) \|^{2} + (1 - \alpha) \|u_{\psi}(t_{i}, x_{t_{i}, i}^{*}) - \frac{1 - \beta}{1 - \alpha} (x_{1, i} - x_{0, i}^{*}) \|^{2} \right];$$

- 7:
- 8: Update generator parameters  $\theta$  via minimizer  $Opt_{gen}$  step with gradients of

$$\frac{1}{B} \sum_{i=1}^{B} \left[ \alpha \| u^*(t_i, x_{t_i, i}^{\theta}) - \frac{\beta}{\alpha} (x_{1,i} - x_{0, i}^{\theta}) \|^2 - \alpha \| u_{sg[\psi]}(t_i, x_{t_i, i}^{\theta}) - \frac{\beta}{\alpha} (x_{1,i} - x_{0, i}^{\theta}) \|^2 \right];$$

- end if
- 10: **end for**

# <span id="page-28-0"></span>C UNIFIED INVERSE DISTILLATION WITH REAL DATA FOR BRIDGE MATCHING AND STOCHASTIC INTERPOLANTS

### <span id="page-28-1"></span>C.1 BRIDGE MATCHING

Bridge Matching [\(Liu et al.,](#page-11-3) [2022;](#page-11-3) [Peluchetti,](#page-11-2) [2023\)](#page-11-2) is an extension of diffusion models specifically design to solve data-to-data, e.g., image-to-image problems. Typically, the distribution p<sup>T</sup> is the distribution of "corrupted data" and p<sup>0</sup> is the distribution of clean data, furthermore, there is some coupling of clean and corrupted data π(x0, x<sup>T</sup> ) with marginals p0(x0) and p<sup>T</sup> (x<sup>T</sup> ). To construct the diffusion which recovers clean data given a corrupted data, one first needs to build prior process (which often is the same forward process used in diffusions):

$$dx_t = f_t(x_t)dt + g_t d\mathbf{w}_t,$$

where ft(·) is a drift function, g<sup>t</sup> is a time-dependent scalar noise scheduler and w<sup>t</sup> is a standard Wiener process. This prior process defines conditional density pt(xt|x0) and the posterior density pt(xt|x0, x<sup>T</sup> ) called "diffusion bridge". To recover p<sup>0</sup> from p<sup>T</sup> , one can use reverse-time SDE with a reverse-time Wiener process w¯t:

$$dx_t = \left( f_t(x_t) - g_t^2 \cdot u_t(x_t) \right) dt + g_t d\bar{\mathbf{w}}_t,$$

where the drift ut(xt) is learned via solving of the bridge matching problem:

$$\mathcal{L}_{BM}(v,\pi) = \mathbb{E}_{t \sim [0,T],(x_0,x_T) \sim \pi, x_t \sim p_t(\cdot|x_0,x_T)} \left[ \|v_t(x_t) - \nabla_{x_t} \log p_t(x_t|x_0)\|^2 \right]. \tag{41}$$

.

However, this reverse-time diffusion in general does not guarantee that the produced samples come from the same coupling π(x0, x<sup>T</sup> ) used for training. It happens only if π(x0, x<sup>T</sup> ) solves entropic optimal transport between p<sup>0</sup> and p<sup>T</sup> . To guarantee the preservance of the coupling π(x0, x<sup>T</sup> ), there exists another version of Bridge Matching called either Augmented Bridge Matching or Conditional Bridge Matching [\(De Bortoli et al.,](#page-10-13) [2023\)](#page-10-13), which differs only by addition of a condition on x<sup>T</sup> to the trainable drift vt(xt, x<sup>T</sup> ):

$$\mathcal{L}_{\text{ABM}}(v, \pi) = \mathbb{E}_{t \sim [0, T], (x_0, x_T) \sim \pi, x_t \sim p(\cdot | x_0, x_T)} \left[ \|v_t(x_t, x_T) - \nabla_{x_t} \log p_t(x_t | x_0)\|_2^2 \right]$$

The learned conditional drift u(xt, x<sup>T</sup> ) is then used for sampling via the reverse-time SDE starting from a given x<sup>T</sup> ∼ p<sup>T</sup> :

$$dx_t = \left(f_t(x_t) - g_t^2 \cdot u_t(x_t, x_T)\right) dt + g_t d\bar{\mathbf{w}}_t.$$

### <span id="page-28-2"></span>C.2 STOCHASTIC INTERPOLANTS

The Stochastic Interpolants framework generalizes Flow Matching and diffusion models, constructing a diffusion or flow between two given distributions p<sup>0</sup> and p<sup>T</sup> . To do so, one needs to consider the interpolation between any pair of points (x0, x<sup>T</sup> ) which are sampled from the coupling π(x0, x<sup>T</sup> ) with marginals p<sup>0</sup> and p<sup>T</sup> . The interpolation itself is given by formula

$$x_t = I(t, x_0, x_T) + \gamma_t \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I}), \quad t \in [0, T],$$

where I(0, x0, x<sup>T</sup> ) = x0, I(T, x0, x<sup>T</sup> ) = x<sup>T</sup> , γ<sup>0</sup> = γ<sup>T</sup> = 0 and γ<sup>t</sup> > 0 for all t ∈ (0, T). This interpolant defines a conditional Gaussian path pt(xt|x0, x<sup>T</sup> ). Note that in the original paper [\(Albergo](#page-10-3) [et al.,](#page-10-3) [2023\)](#page-10-3), the authors consider the time interval [0, 1], but those two intervals are interchangeable by using a change of variable t ′ = T t . Thus, the ODE interpolation between p<sup>0</sup> and p<sup>T</sup> is given by:

$$dx_t = u_t(x_t)dt, \quad x_0 \sim p_0,$$

where ut(x, x<sup>T</sup> ) := E[ ˙xt|x<sup>t</sup> = x] = E[∂tI(t, x0, x<sup>T</sup> ) + ˙γtϵ|x<sup>t</sup> = x] is the unique minimizer of the quadratic objective:

$$\mathcal{L}_{SI}(v,\pi) = \mathbb{E}_{\substack{t \sim [0,T], (x_0, x_T) \sim \pi, \\ (x_t, \epsilon) \sim p(\cdot | x_0, x_T)}} \left[ \|v_t(x_t, x_T) - (\partial_t I(t, x_0, x_T) + \dot{\gamma}_t \epsilon)\|^2 \right]. \tag{42}$$

The authors also provide a way of matching the score and the SDE drift of the reverse process by solving similar MSE matching problems.

### <span id="page-29-0"></span>C.3 OBJECTIVE FOR GENERAL DATA COUPLING

The essential difference of Bridge Matching and Stochastic Interpolants from diffusion models and Flow Matching with a Gaussian path is that they additionally introduce coupling  $\pi(x_0, x_T)$  used to sample  $x_t$  and can work with conditional drifts.

This difference can be easily incorporated to our RealUID distillation framework just by parametrizing the generator  $G_{\theta}$  to output not the samples from the initial distribution  $p_{0}^{\theta}$ , but from the coupling  $\pi^{\theta}$ . One can do it by setting  $\pi^{\theta}(x_{0}, x_{T}) = p_{T}(x_{T})\pi_{0}^{\theta}(x_{0}|x_{T})$ , where conditional data distribution  $\pi_{0}^{\theta}(x_{0}|x_{T})$  is parametrized by the *student generator*  $G_{\theta}: \mathcal{Z} \times \mathbb{R}^{D} \to \mathbb{R}^{D}$  conditioned on a sample  $x_{T} \sim p_{T}$ . This approach is specifically used in Inverse Bridge Matching Distillation (IBMD) (Gushchin et al., 2024). Hence, our Universal Inverse Distillation objective can be written just by substituting student distribution  $p_{0}^{\theta}$  by student coupling  $\pi^{\theta}$ , substituting real data  $p_{0}^{*}$  by real data coupling  $\pi^{*}$  and adding extra conditions.

**Definition 5.** We define Universal Matching loss with real data for general coupling on generated data coupling  $\pi^{\theta} \in \mathcal{P}(\mathbb{R}^D \times \mathbb{R}^D)$  with  $\alpha, \beta \in (0, 1]$ :

$$\mathcal{L}_{R\text{-}UM\text{-}coup}^{\alpha,\beta}(f,\pi^{\theta}) = \underbrace{\alpha \cdot \mathbb{E}_{t \sim [0,T]} \mathbb{E}_{x_{T} \sim p_{T}, x_{0}^{\theta} \sim \pi_{0}^{\theta}(\cdot|x_{T}),} \left[ \|f_{t}(x_{t}^{\theta}, x_{T}) - \frac{\beta}{\alpha} f^{\theta}(x_{t}^{\theta}|x_{0}^{\theta}, x_{T}) \|^{2} \right]}_{generated \ data \ \pi^{\theta} \ term} + \underbrace{(1-\alpha) \cdot \mathbb{E}_{t \sim [0,T]} \mathbb{E}_{x_{T} \sim p_{T}, x_{0}^{*} \sim \pi_{0}^{*}(\cdot|x_{T}),} \left[ \|f_{t}(x_{t}^{*}, x_{T}) - \frac{1-\beta}{1-\alpha} f_{t}^{*}(x_{t}^{*}|x_{0}^{*}, x_{T}) \|^{2} \right]}_{real \ data \ \pi^{*} \ term}.$$

And the corresponding Universal Inverse Distillation loss with real data for general coupling is:

$$\min_{\theta} \max_{f} \{\mathcal{L}_{\textit{R-UID-coup}}^{\alpha,\beta}(f,\pi^{\theta}) := \mathcal{L}_{\textit{R-UM-coup}}^{\alpha,\beta}(f^*,\pi^{\theta}) - \mathcal{L}_{\textit{R-UM-coup}}^{\alpha,\beta}(f,\pi^{\theta})\}.$$

In case of coupling match  $\pi^{\theta} = \pi^*$ , the RealUID loss for couplings attains its minimum, i.e.,

$$\begin{split} \min_{\theta} \max_{f} \mathcal{L}_{\text{R-UID-coup}}^{\alpha,\beta}(f,\pi^{\theta}) &= \min_{\theta} \{ \underbrace{\mathcal{L}_{\text{R-UM-coup}}^{\alpha,\beta}(f^{*},\pi^{\theta}) - \min_{f} \{\mathcal{L}_{\text{R-UM-coup}}^{\alpha,\beta}(f,\pi^{\theta})\} \}}_{\geq 0} \\ &= \mathcal{L}_{\text{R-UM-coup}}^{\alpha,\beta}(f^{*},\pi^{*}) - \underbrace{\min_{f} \{\mathcal{L}_{\text{R-UM-coup}}^{\alpha,\beta}(f,\pi^{*})\}}_{=\mathcal{L}_{\text{R-UM-coup}}^{\alpha,\beta}(f^{*},\pi^{*})} = 0. \end{split}$$

It is worth noting that the inverse distillation from the IBMD framework can be extended to discrete-time Bridge Matching models, as is done in **Residual Shifting Distillation** (Selikhanovych et al., 2025, **RSD**). Likewise, our RealUID objective can be extended to discrete-time models as well.

### <span id="page-30-0"></span>D EXPERIMENTAL DETAILS AND ADDITIONAL RESULTS

#### <span id="page-30-1"></span>D.1 CIFAR-10 DISTILLATION FROM SCRATCH

Codebase, dataset and teachers. Building on the reference codebase and network architectures of (Tong et al., 2024), we implement the training algorithm described in our Algorithm 1. We evaluate the resulting approach on CIFAR-10 (32×32), under both conditional and unconditional settings, benchmarking against established baselines. The codebase implementation is publicly available in

Note that in this codebase, the <u>time flow is reversed</u>, i.e., the time t=0 corresponds to the pure noise, while the time t=1 is the real data. As an unconditional teacher, we use already trained Conditional Flow Matching checkpoints from the above repository. For conditional setup, we slightly modify the original code and train our own teacher. Our trained checkpoints, along with the code, are located in

**Training hyperparameters.** We train both our models with Adam (Kingma & Ba, 2014), using the same momentums  $(\beta_1, \beta_2) = (0, 0.999)$ , learning rate  $3 \times 10^{-5}$  and a 500-step linear warm-up. Similar to SiD framework (Zhou et al., 2024a), we do not recommend setting momentum  $\beta_1 \neq 0$  as it is crucial for a successful convergence in our min-max optimization.

To regulate adaptation between the generator and the fake model, the generator is updated once for every K=5 updates of the fake model, following DMD2 (Yin et al., 2024a). While the SiD framework leverages an EDM architecture (Karras et al., 2022) and updates the generator after a single update of the fake model (K=1), our RealUID approach becomes unstable for values K<3 due to the different (Tong et al., 2024) architecture.

We do not use dropout in generator and fake models. We set a batch size of 256 and maintain an EMA of the generator parameters with decay 0.999 (Hunter, 1986). Additionally, at each optimization step we apply  $\ell_2$  gradient-norm clipping with threshold 1.0 to both the generator and the fake model.

**Training time.** All distillation experiments were trained for 500,000 gradient updates, corresponding to approximately 5 days. The experiments were executed on a <u>single</u> Ascend910B NPU with 65 GB of VRAM memory.

Generator parameterization and models initialization. We parameterize generator  $G_{\theta}(\cdot)$  using a time-dependent U-Net  $g_{\theta}(0,\cdot)$  with a fixed time input t=0 and a one-step integration scheme:

$$G_{\theta}(z) = z + g_{\theta}(0, z).$$

We initialize the model  $g_{\theta}$  with a teacher model, and the fake model with <u>random weights</u>. Empirically, we observe that this initialization strategy lead to improved performance on the considered datasets.

**GAN details.** We integrate a GAN loss into our framework in line with SiD²A and DMD2 (Zhou et al., 2024a; Yin et al., 2024a). In the original setup of (Zhou et al., 2024a), the adversarial loss employs a coefficient ratio of  $\lambda_{\rm adv}^D/\lambda_{\rm adv}^{G_\theta}=10^2$  (see Table 6 in Zhou et al. (2024a)), a choice that poses practical difficulties due to the extreme imbalance between the generator and discriminator losses. To mitigate this issue, we adopt the formulation of (Yin et al., 2024a), where the ratio is  $\approx 3$ , and evaluate different coefficient scales (see the results in Table 1). Additionally, we can select the range of times within which adversarial loss is applied between noised generated and real data samples. We found that the best choice is not to take only clear real data or the whole interval [0,1], but rather to take the range of not severely corrupted data, namely times from 0.8 to 1.

**Evaluation protocol.** We evaluate image quality using the Fréchet Inception Distance (FID; Heusel et al., 2017), computed from 50,000 generated samples following (Karras et al., 2022; 2020; 2019). In line with SiD (Zhou et al., 2024b), we periodically compute FID during distillation and select the checkpoint achieving the minimum value. To ensure statistical reliability, we repeat the evaluation over 3 independent runs, rather than 10 as in SiD, because the empirical variance of FID in our experiments was below 0.01.

**Efficiency comparison.** In terms of efficiency, RealUID leverages a lightweight architecture based on (Tong et al., 2024). Therefore, as summarized in Table 5, it achieves nearly  $2 \times$  faster inference, lower memory usage, and reduced model size compared to recent distillation approaches (Zhou et al., 2024b;a; Huang et al., 2024).

<span id="page-31-2"></span>

| Methods Infer                                                            | rence Time (ms)      | # Total Param (M)    | Max GPU Mem Alloc (MB) | Max GPU Mem Reserved (MB) |  |
|--------------------------------------------------------------------------|----------------------|----------------------|------------------------|---------------------------|--|
| RealUID (Ours)<br>FGM (Huang et al., 2024)<br>SiD (Zhou et al., 2024b:a) | <b>18.636</b> 30.745 | <b>36.784</b> 55.734 | <b>165</b><br>242      | 172<br>276                |  |

Table 5: Inference complexity on an Ascend 910B3 (65 GB) NPU. All methods require only 1 NFE. For each method, we report (i) the mean inference time per image (bs=1, fp32), averaged over 10,000 iterations; (ii) the total number of parameters (Millions); and (iii) peak NPU memory usage (maximum allocated and reserved, in MB). Best values are **bolded**.

#### <span id="page-31-1"></span>D.2 CIFAR-10 DISTILLATION FINE-TUNING

This section presents an ablation study of the fine-tuning stage over the loss-balancing coefficients for GANs and our RealUID on CIFAR-10. In this stage, the generator is initialized from the best-performing checkpoint obtained during training from scratch of the corresponding framework, while the fake model is initialized from the teacher model. In the unconditional setup, the best configuration are RealUID with ( $\alpha=0.92, \beta=0.94$ ) and FID 2.22, and GAN with ( $\lambda_{\rm adv}^{G_{\theta}}=0.3, \lambda_{\rm adv}^{D}=1$ ) and FID 2.29. In the unconditional setup, it is RealUID with ( $\alpha=0.98, \beta=0.96$ ) and FID 2.02, and GAN with ( $\lambda_{\rm adv}^{G_{\theta}}=0.3, \lambda_{\rm adv}^{D}=1$ ) and FID 2.12. Fine-tuning then proceeds with new values  $\alpha_{\rm FT}$  and  $\beta_{\rm FT}$  for our RealUID and  $\lambda_{\rm FT}^{G_{\theta}}$  and  $\lambda_{\rm FT}^{D}$  for GANs. The results are summarized in Table 6.

<span id="page-31-3"></span>Table 6: Ablation of the fine-tuning parameters  $(\alpha_{FT}, \frac{\beta_{FT}}{\alpha_{FT}})$  for our RealUID and fine-tuning scales  $(\lambda_{FT}^{G_{\theta}}, \lambda_{FT}^{D})$  for GANs for unconditional (left) and conditional (right) generation. All values report FID  $\downarrow$ , where lower is better. The mark "–" indicates that configuration is infeasible, and the mark "–" shows that the method did not converge. Best results for each method are **bolded**.

| $\begin{array}{c ccccccccccccccccccccccccccccccccccc$                                                               |                                                                    |      |      |      |      |     |      |      |      |      |
|---------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|------|------|------|------|-----|------|------|------|------|
| $\begin{array}{cccccccccccccccccccccccccccccccccccc$                                                                | $\alpha_{\rm FT} \setminus \frac{\beta_{\rm FT}}{\alpha_{\rm FT}}$ | 0.92 | 0.94 | 0.96 | 0.98 | 1.0 | 1.02 | 1.04 | 1.06 | 1.08 |
| $\begin{array}{cccccccccccccccccccccccccccccccccccc$                                                                | 0.92                                                               | 1.99 | 1.98 | 2.02 | -    | -   | -    | 2.04 | 2.04 | 2.02 |
| $\lambda_{\text{FT}}^{G_{\theta}}$ 0.1 0.3 1.0 5.0 25.0 100.0 $\lambda_{\text{FT}}^{D}$ 0.3 1.0 3.0 15.0 75.0 300.0 | 0.94                                                               | 2.02 | 2.02 | 2.04 | -    | _   | _    | 2.07 | 2.06 | -    |
| $\lambda_{\text{FT}}^{G_{\theta}}$ 0.1 0.3 1.0 5.0 25.0 100.0 $\lambda_{\text{FT}}^{D}$ 0.3 1.0 3.0 15.0 75.0 300.0 | 0.96                                                               | 2.06 | 2.04 | 2.09 | -    | -   | -    | 2.08 | -    | -    |
| $\lambda_{\text{FT}}^D$ 0.3 1.0 3.0 15.0 75.0 300.0                                                                 | 0.98                                                               | 2.07 | 2.05 | 2.07 | -    | -   | -    | -    | -    | -    |
|                                                                                                                     | $\lambda_{\text{FT}}^{G_{\theta}}$                                 | 0.1  | . 0  | .3   | 1.0  | 5.  | 0    | 25.0 | 10   | 0.0  |
| FID 2.25 <b>2.10</b> 2.12                                                                                           | $\lambda_{\text{FT}}^D$                                            | 0.3  | 3 1  | .0   | 3.0  | 15. | 0    | 75.0 | 30   | 0.00 |
|                                                                                                                     | FID↓                                                               | -    |      | -    | -    | 2.2 | 5    | 2.10 | 2    | 2.12 |

| $\alpha_{\mathrm{FT}} \setminus \frac{\beta_{\mathrm{FT}}}{\alpha_{\mathrm{FT}}}$ | 0.92 | 0.94 | 0.96 | 0.98 | 1.0 | 1.02 | 1.04 | 1.06 | 1.08 |
|-----------------------------------------------------------------------------------|------|------|------|------|-----|------|------|------|------|
| 0.92                                                                              | 1.92 | 1.91 | 1.99 | -    | -   | -    | 1.96 | 1.94 | 1.92 |
| 0.94                                                                              | 1.92 | 1.90 | 1.88 | =    | -   | _    | 1.96 | 1.91 | -    |
| 0.96                                                                              | 1.93 | 1.94 | 1.87 | -    | -   | _    | 1.96 | -    | -    |
| 0.98                                                                              | 1.91 | 1.95 | 1.95 | -    | -   | _    | -    | -    | -    |
| $\lambda_{\text{FT}}^{G_{\theta}}$                                                | 0.1  | 0    | .3   | 1.0  | 5.  | 0    | 25.0 | 100  | 0.0  |
| $\lambda_{\text{FT}}^D$                                                           | 0.3  | 3 1  | .0   | 3.0  | 15. | 0    | 75.0 | 30   | 0.0  |
| FID↓                                                                              | _    |      | -    | _    | 1.9 | 4    | 1.88 | 2    | .04  |

We observe that fine-tuning is highly sensitive to the choice of factor  $\frac{\beta_{\rm FT}}{\alpha_{\rm FT}}$  which still brings the main impact. The best factors  $\frac{\beta_{\rm FT}}{\alpha_{\rm FT}}=0.94$  or  $\frac{\beta_{\rm FT}}{\alpha_{\rm FT}}=1.06$  are much farther from 1.0 compared to training from scratch (Table 1), i.e., fine-tuning relies more on information from real data rather than on guidance from a teacher. Meanwhile, configurations closer to 1.0 are unstable, underscoring the crucial role of real data. In the case of GANs, small adversarial losses similarly fail to converge, and only high scales which particularly emphasize real data achieve improvement.

**Training details.** We run fine-tuning with a smaller learning rate  $1 \times 10^{-5}$  and without warm-up. All other details remain the same as described in Appendix D.1 for training from scratch.

**Training time.** All fine-tuning experiments were conducted for 100,000 gradient updates, which took a little more than 1 day, starting from the best distillation checkpoints. The experiments were executed on a single Ascend910B NPU with 65 GB of VRAM memory.

### <span id="page-31-0"></span>D.3 CELEBA DISTILLATION

In this section, we present the results of the same ablation study from ( $\S4.2$ ) on the CelebA dataset with higher  $64 \times 64$  resolution (Liu et al., 2015). The results are summarized in Table 7. Similar to

Table 1 for CIFAR10, the same pairs of coefficients with  $\beta/\alpha = 1.02$  or  $\beta/\alpha = 0.98$  yield a significant improvement in quality over the baseline ( $\alpha = 1.0, \beta = 1.0$ ), reaching a level comparable to GANs.

<span id="page-32-0"></span>

| $\alpha \diagdown \frac{\beta}{\alpha}$ | 0.96 | 0.98 | 1.00 | 1.02 | 1.04 |
|-----------------------------------------|------|------|------|------|------|
| 0.88                                    | 1.03 | 1.08 | 1.36 | 1.14 | 1.45 |
| 0.90                                    | 1.06 | 1.03 | 1.38 | 1.06 | 1.48 |
| 0.92                                    | 1.12 | 1.04 | 1.28 | 1.10 | 1.69 |
| 0.94                                    | 1.13 | 1.03 | 1.18 | 1.10 | 1.64 |
| 0.96                                    | 1.24 | 1.11 | 1.25 | 1.07 | 1.69 |
| 0.98                                    | 1.65 | 1.26 | 1.22 | 1.29 | -    |
| 1.0                                     | -    | -    | 1.20 | -    | -    |

| $\lambda_{\rm adv}^{G_\theta}$ | $\lambda_{\rm adv}^D$ | FID (↓) |
|--------------------------------|-----------------------|---------|
| 0.1                            | 0.3                   | 1.14    |
| 0.3                            | 1                     | 1.18    |
| 1                              | 3                     | 1.10    |
| 5                              | 15                    | 1.04    |
| 25                             | 75                    | 3.31    |

Table 7: Ablation studies of our  $(\alpha, \frac{\beta}{\alpha})$  parameters in the left table and adversarial weighting parameters  $(\lambda_{\rm adv}^{G_{\theta}}, \lambda_{\rm adv}^{D})$  in the right table for CelebA, 800,000 iterations. The baseline RealUID  $(\alpha = 1.0, \beta = 1.0)$  does not use real data. Configurations that substantially outperform the baseline are highlighted. All values report FID  $\downarrow$ , where lower is better. The best configuration is **bolded**. The mark "–" denotes infeasible configurations.

Training hyperparameters and details. We take the same architecture (Tong et al., 2024) as for CIFAR-10, but adapt it to a larger resolution. We train both models with Adam (Kingma & Ba, 2014) for 800,000 iterations, using the same momentums  $(\beta_1, \beta_2) = (0, 0.999)$ , learning rate  $5 \times 10^{-6}$  and a 500-step linear warm-up. Similar to SiD framework (Zhou et al., 2024a), we do not recommend setting momentum  $\beta_1 \neq 0$  as it is crucial for a successful convergence in our min-max optimization.

To regulate adaptation between the generator and the fake model, the generator is updated once for every K=5 updates of the fake model, following DMD2 (Yin et al., 2024a). While the SiD framework leverages an EDM architecture (Karras et al., 2022) and updates the generator after a single update of the fake model (K=1), our RealUID approach becomes unstable for values K<3 due to the different (Tong et al., 2024) architecture.

We do not use dropout in generator and fake models. We set a batch size of 64 and maintain an EMA of the generator parameters with decay 0.999 (Hunter, 1986). Additionally, at each optimization step we apply  $\ell_2$  gradient-norm clipping with threshold 1.0 to both the generator and the fake model.

All other details remain the same as described in Appendix D.1 for CIFAR-10.

**Teacher training.** For CelebA, we train our own teacher model based on the official implementation of the conditional flow matching procedure from (Tong et al., 2024). We use the same pipeline, architectures, and hyperparameters, but with larger networks and a different dataset. The adapted code for teacher training and final checkpoints for distillation can be found in our repository:

https://github.com/David-cripto/RealUID.

**Fine-tuning.** For fine-tuning, we hold the data-free UID baseline (our RealUID with  $\alpha = 1.0, \beta = 1.0$ ) and all highlighted GAN and RealUID setups from Table 7 for twice as long, i.e., for 1,600,000 iterations. The best-found configurations and results are reported in Table 8 and Figure 6. According to it, our RealUID still outperforms data-free UID baseline, reaching the same performance as GANs.

**Training time.** All experiments were executed on a <u>single</u> Ascend910B NPU with 65 GB of VRAM memory. Regular 800,000 gradient updates took approximately 5 days, while longer fine-tuning with 1,600,000 iterations took 10 days.

<span id="page-33-2"></span>Table 8: This table presents the results of ablation study of our RealUID framework, evaluated using the FID metric on CelebA dataset, 1,600,000 iterations. The Teacher Flow model with 100 NFE is reported as a reference. The performance of the UID (FGM) baseline without real-data incorporation is indicated in *italic*. For emphasis, we <u>underline</u> the two counterparts that incorporate real data: the GAN-based and our RealUID methods. The best-performing configuration is highlighted in **bold**. Qualitative results are presented in Appendix D.5.2.

| Model                                                                                   | FID (↓)     |
|-----------------------------------------------------------------------------------------|-------------|
| Teacher Flow (NFE=100)                                                                  | 2.46        |
| UID (FGM)                                                                               | 0.96        |
| UID + GAN ( $\lambda_{\text{adv}}^{G_{\theta}} = 1.0, \lambda_{\text{adv}}^{D} = 3.0$ ) | <u>0.87</u> |
| RealUID ( $\alpha = 0.88, \beta = 0.90$ ) ( <b>Ours</b> )                               | 0.89        |

<span id="page-33-3"></span>![](_page_33_Figure_3.jpeg)

Figure 6: Evolution of FID during CelebA distillation for the data-free UID baseline and the best-performing RealUID configuration. The performances of Teacher Flow and UID+GAN are indicated by horizontal lines in their respective colors.

### <span id="page-33-0"></span>D.4 FURTHER HYPERPARAMETERS GRIDSEARCH

The primary goal across all experiments in this paper was to study RealUID framework, focusing on the effects of the coefficients  $\alpha$  and  $\beta$ , and provide a fair comparison with GANs. For this reason, we kept all other hyperparameters fixed at their standard values. Now that we have identified the optimal settings for RealUID, we can explore other hyperparameters. Below, we provide a list of useful findings, while the latest hyperparameters sets and training pipelines are described in our repository

**EMA decays.** One can track not only a single EMA decay but a range of values, e.g., [0.999, 0.9996, 0.9999], during a single training run. In long-distance training, larger EMA decays can lead to more stable convergence dynamics and better metrics, whether training from scratch or fine-tuning.

#### <span id="page-33-1"></span>D.5 EXAMPLE OF SAMPLES FOR VARIOUS METHODS

This section presents representative sample outputs from various studies conducted within the RealUID framework.

# <span id="page-34-0"></span>D.5.1 CIFAR-10 GENERATED IMAGES

![](_page_34_Figure_2.jpeg)

Figure 7: Uncurated samples for *unconditional* generation by the one-step data-free baseline UID trained on CIFAR-10. Quantitative results are reported in Table [2.](#page-8-2)

![](_page_35_Figure_1.jpeg)

Figure 8: Uncurated samples for *unconditional* generation by the one-step UID + GAN ( $\lambda_{\rm adv}^{G_{\theta}} = 0.3, \lambda_{\rm adv}^{D} = 1 | \lambda_{\rm FT}^{G_{\theta}} = 25, \lambda_{\rm FT}^{D} = 75$ ) trained on CIFAR-10. Quantitative results are reported in Table 2.

![](_page_36_Figure_1.jpeg)

Figure 9: Uncurated samples for *unconditional* generation by our one-step RealUID (α = 0.92, β = 0.94 | αFT = 0.92, βFT = 0.86) trained on CIFAR-10. Quantitative results are reported in Table [2.](#page-8-2)

![](_page_37_Figure_1.jpeg)

Figure 10: Uncurated samples for *conditional* generation by the one-step data-free baseline UID trained on CIFAR-10. Quantitative results are reported in Table [2.](#page-8-2)

![](_page_38_Figure_1.jpeg)

Figure 11: Uncurated samples for *conditional* generation by the one-step UID + GAN ( $\lambda_{\rm adv}^{G_{\theta}} = 0.3, \lambda_{\rm adv}^D = 1 | \lambda_{\rm FT}^{G_{\theta}} = 25, \lambda_{\rm FT}^D = 75$ ) trained on CIFAR-10. Quantitative results are reported in Table 2.

![](_page_39_Figure_1.jpeg)

Figure 12: Uncurated samples for *conditional* generation by our one-step RealUID (α = 0.98, β = 0.96 | αFT = 0.96, βFT = 0.92) trained on CIFAR-10. Quantitative results are reported in Table [2.](#page-8-2)

# <span id="page-40-0"></span>D.5.2 CELEBA GENERATED IMAGES

![](_page_40_Figure_2.jpeg)

Figure 13: Uncurated samples by the one-step data-free baseline UID trained on CelebA. Quantitative results are reported in Table [7.](#page-32-0)

![](_page_41_Figure_1.jpeg)

Figure 14: Uncurated samples by the one-step UID + GAN ( $\lambda_{\rm adv}^{G_{\theta}}=1.0, \lambda_{\rm adv}^{D}=3.0$ ) trained on CelebA. Quantitative results are reported in Table 7.

![](_page_42_Figure_1.jpeg)

Figure 15: Uncurated samples by our one-step RealUID (α = 0.88, β = 0.9) trained on CelebA. Quantitative results are reported in Table [7.](#page-32-0)