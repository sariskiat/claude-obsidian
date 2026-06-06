---
type: paper-fulltext
slug: optimizing-diffusion-noise-can-serve-as-universal-motion-priors
arxiv_id: null
source_path: /Users/saris.kia.adm/.paper-scholar/optimizing-diffusion-noise-can-serve-as-universal-motion-priors/02-optimizing-diffusion-noise-cvpr2024.md
paper: "[[optimizing-diffusion-noise-can-serve-as-universal-motion-priors]]"
---
# <span id="page-0-0"></span>Optimizing Diffusion Noise Can Serve As Universal Motion Priors

Korrawe Karunratanakul<sup>1</sup> Konpat Preechakul<sup>2</sup> Emre Aksan<sup>4</sup> Thabo Beeler<sup>4</sup> Supasorn Suwajanakorn<sup>3</sup> Siyu Tang<sup>1</sup>

> <sup>1</sup>ETH Zurich, Switzerland ¨ <sup>2</sup>UC Berkeley <sup>3</sup>VISTEC, Thailand <sup>4</sup>Google

<https://korrawe.github.io/dno-project/>

![](_page_0_Figure_5.jpeg)

Figure 1. Our proposed Diffusion Noise Optimization (DNO) can leverage the existing human motion diffusion models as universal motion priors. We demonstrate its capability in the motion editing tasks where DNO can preserve the content of the original model and accommodates a diverse range of editing modes, including changing trajectory, pose, joint location, and avoiding newly added obstacles.

## Abstract

*We propose Diffusion Noise Optimization (DNO), a new method that effectively leverages existing motion diffusion models as motion priors for a wide range of motion-related tasks. Instead of training a task-specific diffusion model for each new task, DNO operates by optimizing the diffusion latent noise of an existing pre-trained text-to-motion model. Given the corresponding latent noise of a human motion, it propagates the gradient from the target criteria defined on the motion space through the whole denoising process to update the diffusion latent noise. As a result, DNO supports any use cases where criteria can be defined as a function of motion. In particular, we show that, for mo-* *tion editing and control, DNO outperforms existing methods in both achieving the objective and preserving the motion content. DNO accommodates a diverse range of editing modes, including changing trajectory, pose, joint locations, or avoiding newly added obstacles. In addition, DNO is effective in motion denoising and completion, producing smooth and realistic motion from noisy and partial inputs. DNO achieves these results at inference time without the need for model retraining, offering great versatility for any defined reward or loss function on the motion representation.*

## <span id="page-1-0"></span>1. Introduction

Many applications of great interest to the motion modeling community can be framed as "finding a plausible motion that fulfills a set of criteria", typically formulated as minimizing a cost function addressing the given constraints. These include but are not limited to generating motions that follow a trajectory, target locations for joints, keyframes, or avoiding obstacles; denoising, completing missing parts of a motion or editing an existing one. Though they may seem diverse, a unified framework should be able to address these diverse yet relevant tasks, which typically follow a task-agnostic motion prior that ensures plausible motions and task-specific cost functions. Such high-quality motion priors are highly sought after, often with efforts that focus on improving architectures. Yet, a truly versatile framework utilizing the underlying motion prior expressively is still lacking.

Among many human motion priors, the diffusion-based models [\[14,](#page-8-0) [33,](#page-9-0) [41,](#page-9-1) [46,](#page-9-2) [66,](#page-10-0) [80\]](#page-10-1) have become the most prominent ones by achieving impressive performance on certain motion modeling tasks. Avatar Grow Legs [\[14\]](#page-8-0) and EgoEgo [\[41\]](#page-9-1) can generate full-body motions from limited joint specifications, such as head and hand poses, but their specific scope precludes their use as general motion priors. PriorMDM [\[66\]](#page-10-0) introduces a more flexible setting; however, the underlying root-relative motion representation requires dense condition signals for individual joints or keyframes, limiting its use to a small set of applications. OmniControl [\[80\]](#page-10-1) addresses this limitation by augmenting the motion diffusion model with another network to encode input conditions. While this improves the fidelity and quantity of control signals, it is not always straightforward to support many other constraints. For instance, generating locomotion in a scene with obstacles requires a perception of the physical scene, which is unclear how to cover arbitrary scenes effectively. It is only exacerbated when the scene dynamically changes over time, such as a scene populated by other human agents. GMD [\[33\]](#page-9-0) presents a more versatile approach by limiting the conditioning to task-based objectives, which theoretically supports arbitrary use cases. Yet there is a trade-off between this flexibility and the fidelity of controls and quality of motions.

To this end, we propose a simple yet effective approach to utilize a motion diffusion model as a motion prior. By treating the denoising process as a black box, we can frame motion-related tasks, such as motion editing and refinement, as optimization problems on the latent manifold of the diffusion model, similar to other classes of generative models such as GANs [\[50\]](#page-9-3) and VAEs [\[59\]](#page-10-2) where optimization is seen as iteratively updating the solution on the learned motion manifold (Fig. [2a](#page-3-0)). In this work, we demonstrate that it is possible and feasible to back-propagate gradients through the full-chain diffusion process, and then optimize the noise vector based on user-provided criteria in the motion space. This simple approach, dubbed Diffusion Noise Optimization (DNO), which we show to be effective at editing and preserving contents, is a unified and versatile framework that supports a wide variety of applications without the need for fine-tuning the underlying motion model for each specific application. By changing the optimization objective, defined as any differentiable loss function computed on the output motion, DNO enables the diffusion model to effectively serve as a motion prior.

Our experiments show that our unified framework DNO, without any model fine-tuning, produces high-quality motions for motion editing, outperforming existing methods in both preserving the motion content and fulfilling a wide range of objectives including changing trajectory, pose, joint location, and avoiding obstacles. In addition, we demonstrate that it can produce smooth and realistic motion from noisy and partial inputs. Lastly, we provide extensive studies to validate our design choices, which can serve as a basis to effectively extend our framework to other motionrelated tasks.

## 2. Related Works

Motion synthesis, editing, and completion. The human motion generation task aims to generate motions either conditionally or unconditionally [\[51,](#page-9-4) [62,](#page-10-3) [81,](#page-11-0) [89,](#page-11-1) [90\]](#page-11-2). Various conditioning signals have been explored such as partial poses [\[15,](#page-8-1) [20\]](#page-8-2), trajectories [\[34,](#page-9-5) [66,](#page-10-0) [77,](#page-10-4) [88\]](#page-11-3), images [\[7,](#page-8-3) [59\]](#page-10-2), music [\[38,](#page-9-6) [39,](#page-9-7) [42\]](#page-9-8), objects [\[79\]](#page-10-5), action labels [\[16,](#page-8-4) [52\]](#page-9-9), scene [\[27\]](#page-8-5), or text [\[1,](#page-8-6) [18,](#page-8-7) [19,](#page-8-8) [35,](#page-9-10) [53\]](#page-9-11). Recently, the focus has been on generating motion based on natural-language descriptions using diffusion-based models [\[8,](#page-8-9) [73\]](#page-10-6). These models, utilizing the CLIP model [\[57\]](#page-10-7), have shown significant improvements in text-to-motion generation [\[6,](#page-8-10) [84,](#page-11-4) [87\]](#page-11-5) and support conditioning on partial motions or music [\[2,](#page-8-11) [74\]](#page-10-8). Alternatively, the motion can be treated as a new language and embedded into the language model framework [\[30,](#page-9-12) [85\]](#page-11-6) Nevertheless, they lack the ability to handle spatial conditioning signals, such as keyframe locations or trajectories. GMD [\[33\]](#page-9-0) handles this problem with classifier-based guidance at inference time to steer the denoising process toward the target conditions. OmniControl [\[80\]](#page-10-1) combined GMD with ControlNet [\[86\]](#page-11-7) to improve realism but limits the conditioning signals to text and partial motion observations. However, motion editing under the motion synthesis framework remains unexplored due to the lack of an explicit mechanism to retain the original motion content.

In parallel, the generation of full-body poses from sparse tracking signals of body joints has gained considerable interest within the community. Previous work such as EgoEgo [\[41\]](#page-9-1), AGRoL [\[14\]](#page-8-0), and AvatarPoser [\[31\]](#page-9-13) while showing impressive results, tend to be specialized models trained ex<span id="page-2-1"></span>plicitly for the motion completion or denoising task. Therefore, it is unclear how we can leverage the motion priors in these trained models for solving other tasks. In this work, we focus on using the trained motion diffusion model to tackle various motion-related tasks under a unified framework including editing, completion, and refinement.

Diffusion models and guidance. Diffusion-based probabilistic generative models (DPMs), a class of generative models based on learning to progressively denoising the input data, [\[24,](#page-8-12) [67,](#page-10-9) [70,](#page-10-10) [71\]](#page-10-11) have gained significant attention across multiple fields of research. They have been successfully applied to tasks such as image generation [\[13\]](#page-8-13), image super-resolution [\[40,](#page-9-14) [63\]](#page-10-12), speech synthesis [\[37,](#page-9-15) [55\]](#page-9-16), video generation [\[25,](#page-8-14) [26\]](#page-8-15), 3D shape generation [\[54,](#page-9-17) [78\]](#page-10-13), and reinforcement learning [\[29\]](#page-9-18). The growing interest in the diffusion-based model stems from their superior results and impressive generation controllability, for example, in textconditioned generation [\[58,](#page-10-14) [61,](#page-10-15) [64\]](#page-10-16) and image-conditioned editing [\[3,](#page-8-16) [4,](#page-8-17) [9,](#page-8-18) [22,](#page-8-19) [48\]](#page-9-19). In terms of conditioning, there are various methods for the diffusion-based models such as imputation and inpainting [\[9,](#page-8-18) [10,](#page-8-20) [48\]](#page-9-19), classifier-based guidance [\[10,](#page-8-20) [13\]](#page-8-13), and classifier-free guidance [\[23,](#page-8-21) [58,](#page-10-14) [61,](#page-10-15) [64\]](#page-10-16). For refinement, SDEdit [\[48\]](#page-9-19) enables the repetition of the denoising process to gradually improve output quality but lacks the ability to provide editing guidance. Recently, DOODL [\[75\]](#page-10-17) demonstrates a direct latent optimization approach for image editing with the help of an invertible ODE [\[76\]](#page-10-18). Being inspired by that work, we propose a related method with an improved optimization algorithm that speeds up optimization. DNO can effectively be used in various motion-related tasks making it suitable as a versatile human motion priors. Additionally, we discover that an invertible ODE is not required which makes DNO simpler and convenient to use with minimal effort.

## 3. Background

#### 3.1. Motion generation with diffusion model

A diffusion probabilistic model is a denoising model that learns to invert a diffusion process. A diffusion process is defined as q(xt|x0) = N ( √ αtx0,(1 − αt)I) where x<sup>0</sup> is a clean motion and x<sup>t</sup> is a noisy motion at the level of t defined by noise schedule αt. With the diffusion process, we can infer an inverse denoising process q(xt−1|xt, x0). Now, we can train the diffusion model by predicting q(xt−1|xt, x0) with a learned pθ(xt−1|xt, c), which is parameterized by a function dθ(xt, c) where c is additional conditions, e.g. text prompts. Conditioning is particularly useful for specifying what kind of motion activity we want from the model.

While diffusion models are stochastic, there exist deterministic sampling processes that share the same marginal distribution. These processes include those defined by probability flow ODE [\[71\]](#page-10-11) or by reformulating the diffusion process to be non-Markovian as in DDIM [\[68\]](#page-10-19). To obtain a sample based on deterministic sampling, we can solve the associated ODE using an ODE solver starting from x<sup>T</sup> ∼ N (0, I).

#### 3.2. Diffusion model inversion

Diffusion inversion is a process that retrieves the corresponding noise map x<sup>T</sup> given an input x0. Under deterministic sampling, the associated ODE not only describes the denoising process from x<sup>T</sup> to x<sup>0</sup> but also its reverse from x<sup>0</sup> to x<sup>T</sup> , which is achievable by solving the ODE backward. However, the fidelity of this inversion relies on the smoothness assumption dθ(xt) ≈ dθ(xt−1), which is unlikely to hold true when solving the ODEs with just a few discretization steps. For tasks requiring multiple back-andforth evaluations between x<sup>T</sup> and x0, an alternative inversion method is available [\[76\]](#page-10-18).

#### 3.3. Motion representation

The relative-root representation [\[17\]](#page-8-22) has been widely adopted for text-to-motion diffusion models [\[73\]](#page-10-6). The representation is a matrix of human joint features over the motion frames with shape D × M, where D = 263 and M are the representation size and the number of motion frames, respectively. Each motion frame represents root relative rotation and velocity, root height, joint locations, velocities, rotations, and foot contact labels. As the relative representation abstracts away the absolute root location, it can improve the generalization of motion models but makes the controllable generation more challenging [\[33,](#page-9-0) [73\]](#page-10-6).

## 4. Diffusion Noise Optimization

A straightforward way to obtain a motion that fulfills a criterion L(x) is by optimizing x <sup>∗</sup> = arg min<sup>x</sup> L(x) via iterative optimization such as gradient descent. However, optimization in this representation space often yields implausible results as most motion samples x ∈ R <sup>D</sup>×<sup>M</sup> do not encode plausible motions. This motivates performing optimization on an expressive *latent* space z which provides valid motion samples when decoded. The new optimization task becomes

<span id="page-2-0"></span>
$$\mathbf{z}^* = \arg\min_{\mathbf{z}} \mathcal{L}(f(\mathbf{z})) \tag{1}$$

Such optimization has led to the success in image editing of GANs [\[21,](#page-8-23) [32\]](#page-9-20), enabled by its smooth latent space z whose mapping is parameterized by a powerful generative model f(z) and learned from a large dataset. In this work, we show that the latent optimization extends well to f, parameterized as pretrained diffusion models in the motion domain.

For diffusion models, our choice of the latent variable is the diffusion's noise at time T, z = x<sup>T</sup> . While x<sup>T</sup> is not

<span id="page-3-2"></span><span id="page-3-0"></span>![](_page_3_Figure_0.jpeg)

(a) At each optimization step, DNO maintains the output motion equality by making a step in the latent space  $\mathbf{x}_T$ , which is decodable to a realistic motion almost everywhere.

![](_page_3_Figure_2.jpeg)

(b) DNO's step direction is obtained from the gradient of a task-specific criterion  $\mathcal L$  via backpropagation through an ODE solver,  $f(\mathbf x_T) = \text{ODESolver}(d(\cdot),\mathbf x_T)$ . At convergence, the optimized  $x_T$  is ultimately decoded via the ODE solver to get the prediction  $x_0$ .

Figure 2. Diffusion Noise Optimization (DNO).

perfectly smooth [56, 68],  $\mathbf{x}_T$  offers the highest-level abstraction readily available. Now, getting the output  $\mathbf{x}$  from a diffusion model  $d(\cdot)$  requires solving an ordinary differential equation (ODE) using an ODE solver [71]. We now see Equation 1 as

$$\mathbf{x}_T^* = \operatorname*{arg\,min}_{\mathbf{x}_T} \mathcal{L}(\text{ODESolver}(d(\cdot), \mathbf{x}_T))$$
 (2)

This allows us to approach many motion tasks simply by adjusting the task-specific criterion  $\mathcal{L}$  while keeping the motion model, d, intact. In this work, we use the DDIM-ODE [68] and its solver.

This optimization is iteratively solved using gradient descent. Starting from a certain noise  $\mathbf{x}_T$ , we solve the ODE, arrive at a prediction  $\mathbf{x}$ , and evaluate the criterion function  $\mathcal{L}(\mathbf{x})$ . Then, we obtain the gradient  $\nabla_{\mathbf{x}_T} \mathcal{L}(\mathbf{x})$  by backpropagating through the ODE solver. An optimizer updates

 $\mathbf{x}_T$  based on the gradient, possibly with a small random perturbation to encourage exploration [32, 75]. We repeat this until convergence. The final output is the motion obtained by solving the ODE one last time with the optimized  $\mathbf{x}_T$ . We call the above algorithm Diffusion Noise Optimization (DNO) and summarize it in Algo. 1.

Maintaining the intermediate activations for solving the ODE during backpropagation can be memory-intensive. This issue can be addressed with gradient checkpointing [11] or an invertible ODE [75, 76], at the cost of more computation or model complexity. In the motion domain, impressive results can be achieved from an ODE solver with a minimal number of function evaluations and feasible memory overhead. Given the ongoing efforts to reduce diffusion sampling steps, as evidenced in distillation [43, 65, 72] and high-order solvers studies [44, 45, 91], our simple design, despite requiring back-propagation through all steps, can become increasingly relevant and applicable.

Optimization through ODE solver. Empirically, gradients via backpropagation through the ODE solver have norms spanning multiple orders of magnitude, making the optimization unstable and slow. We instead propose to normalize the gradient to have a unit norm, which optimizes faster in practice. Normalized gradient methods are also found to help escape saddle-point in the loss landscape faster [12, 49]. When combined with momentum, the update becomes a moving average of directions, which is robust to gradient norm outliers.

#### <span id="page-3-1"></span>Algorithm 1 Diffusion Noise Optimization

**Require:**  $\mathbf{x}_T$ , motion model d, ODESolver, Optimizer, criterion  $\mathcal{L}$ , learning rate  $\eta$ , perturbation  $\gamma$  (default 0)

- 1: while not converged do
- 2:  $\mathbf{x} \leftarrow \text{ODESolver}(d(\cdot), \mathbf{x}_T)$
- 3:  $\nabla \leftarrow \nabla_{\mathbf{x}_T} \mathcal{L}(\mathbf{x})$   $\triangleright$  Task-specific
- 4:  $\mathbf{x}_T \leftarrow \text{Optimizer}(\mathbf{x}_T, \nabla / \|\nabla\|, \eta) + \gamma \mathcal{N}(\mathbf{0}, \mathbf{1})$
- 5: end while
- 6: **return** ODESolver( $\mathbf{x}_T, f$ )

**DNO fundamentally differs from a guided motion diffusion method** [33] in how the criterion  $\mathcal{L}(\cdot)$  is computed at each iteration. For guided diffusion methods, the criterion is computed on an expected  $\hat{x} = \mathbb{E}[\mathbf{x}_0|\mathbf{x}_t]$  at a denoising step t [69, 83] that is  $\mathcal{L}(\mathbf{x}_t) = \mathbb{E}_{p(\mathbf{x}_0|\mathbf{x}_t)}\mathcal{L}(\mathbf{x}_0) \approx \mathcal{L}(\hat{\mathbf{x}})$ , resulting in severe error when  $\mathrm{Var}[\mathbf{x}_0|\mathbf{x}_t]$  is large. For DNO, the criterion is exactly computed on  $\mathbf{x}$  after the full-chain denoising which eliminates the approximation error. We further discuss these differences in the supp. mat.

#### 5. Applications

We demonstrate the versatility of DNO on a wide range of conditional motion synthesis tasks. These tasks can be <span id="page-4-1"></span>solved by designing a task-specific criterion  $\mathcal{L}$ , along with any additional auxiliary loss functions. The method for initializing the noise  $\mathbf{x}_T$  can vary depending on the task. In the following sections,  $\mathbf{x}$  refers to an output from an ODE solver which is a function of  $\mathbf{x}_T$ .

#### 5.1. Motion editing and control

The goal of motion editing tasks is to modify a given reference motion  $\mathbf{x}_{ref}$  to satisfy certain objectives. These objectives may include following a specific trajectory, conforming to specified poses in all or some keyframes, and avoiding static or moving obstacles, all while preserving the key characteristics of the input motion.

Editing to follow trajectories and poses. Editing a motion to follow a specific trajectory or to match specific poses can be easily achieved by minimizing the average distance between each generated joint location and its corresponding target location. Target locations can be specified for any subset of joints in any subset of motion frames.

Specifically, let  $\mathbf{c}_j^k \in \mathbb{R}^3$  be the target location for joint j at keyframe k, and  $\hat{\mathbf{c}}_j^k$  be its generated location from the current motion  $\mathbf{x}$ . The loss function can be defined as:

$$\mathcal{L}_{\text{pose}}(\mathbf{x}, O) := \frac{1}{|O|} \sum_{(j,k) \in O} \left\| \hat{\mathbf{c}}_j^k(\mathbf{x}) - \mathbf{c}_j^k \right\|_1, \quad (3)$$

where O is the *observed* set containing (j, k) pairs denoting the target joints and keyframes determined by the task.

Editing to avoid obstacles. Obstacles can be represented as a signed distance function (SDF), whose gradient field defines the repelling direction away from the obstacles. The loss function can incorporate a safe distance threshold  $\tau$ , beyond which the gradient becomes zero such that

$$\mathcal{L}_{\text{obs}}(\mathbf{x}) := \sum_{j,k} -\min \left[ \text{SDF}^k(\hat{\mathbf{c}}_j^k(\mathbf{x})), \tau \right], \tag{4}$$

where  $SDF^k$  is the signed distance function for obstacles in frame k, which may vary across frames in the case of moving obstacles.

**Preserving original characteristics.** When modifying, for example, a jumping motion to align with a certain trajectory, it is crucial to retain key aspects of the jump like its rhythm, height, and overall body coordination. This can be achieved using two means. First, we invert the reference motion  $\mathbf{x}_{\text{ref}}$  using diffusion inversion to obtain the corresponding noise sample  $\mathbf{x}_{T\text{ref}} = \text{ODESolver}^{-1}(d(\cdot), \mathbf{x}_{\text{ref}})$  and use  $\mathbf{x}_{T\text{ref}}$  as the intial value for  $\mathbf{x}_{T}$  we are optimizing. Second, we penalize the distance between  $\mathbf{x}_{T\text{ref}}$  and  $\mathbf{x}_{T}$  to ensure that it remains close to the reference motion during optimization:

$$\mathcal{L}_{\text{cont}}(\mathbf{x}_T) := \|\mathbf{x}_{T\,\text{ref}} - \mathbf{x}_T\|_2. \tag{5}$$

Many motion editing tasks can be solved using a combination of these loss functions with a set of balancing weights

$$\mathcal{L}(\cdot) = \mathcal{L}_{pose}(\mathbf{x}) + \lambda_{obs}\mathcal{L}_{obs}(\mathbf{x}, O) + \lambda_{cont}\mathcal{L}_{cont}(\mathbf{x}_T)$$
 (6)

#### 5.2. Motion refinement and completion

The tasks in this category is to reconstruct a motion from noisy and/or incomplete input. This class of tasks includes completing a motion with missing frames or joints, seamlessly blending motions together, denoising a noisy motion, or any combination of these tasks.

Motion refinement. Given a noisy input motion, motion refinement seeks to enhance the input so that it becomes more realistic. We can solve this problem simply by starting the optimization from a random  $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ , setting the observed set O to include all joints from the input noisy motion, and optimizing  $\mathcal{L}_{pose}(\mathbf{x}, O)$ . While using  $\mathcal{L}_{pose}$  here may seem counterintuitive, as it seeks to match the predicted motion to the original noisy input, DNO is able to generate a plausible motion by eliminating the noise components from the motion.

Since this optimization begins with a random noise  $\mathbf{x}_T$ , the initial prediction can be far from the desired predicted motion, requiring significant changes on  $\mathbf{x}_T$  with several optimization steps, which tend to correlate neighboring noises and reduce the representation capacity. We empirically find that regularizing the noise to decorrelate each latent dimension alleviates the foot skating problem. Inspired by StyleGAN2 [32], we introduce a latent decorrelation loss across motion frames:

$$\mathcal{L}_{\text{decorr}}^{m}(\mathbf{x}_{T}) = \frac{1}{mD} \sum_{i=1}^{m} \mathbf{x}_{T,m}(i)^{\top} \mathbf{x}_{T,m}(i+1). \quad (7)$$

We apply this loss at several scales of temporal resolutions  $m \in \{M, M/2, M/4, \dots, 2\}$ . Specifically, starting with the original length M, we downsample the sequence's temporal resolution by half via average pooling two consecutive frames successively.

This loss, summed over resolutions, encourages more plausible motions and can be used together with  $\mathcal{L}_{pose}$  for motion refinement:

<span id="page-4-0"></span>
$$\mathcal{L}(\cdot) = \mathcal{L}_{pose}(\mathbf{x}, O) + \lambda_{decorr} \sum_{m} \mathcal{L}_{decorr}^{m}(\mathbf{x}_{T})$$
 (8)

**Motion completion.** Unlike motion refinement, where its input contains complete joint information for all frames, this task involves taking an incomplete motion, that may be noisy, as input and seek to fill in the missing information. We begin the optimization with  $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$  and apply the same loss in Equation 8, but with O containing the existing joints in the input motion for  $\mathcal{L}_{pose}(\mathbf{x}, O)$  term.

## <span id="page-5-1"></span>6. Experiments

Datasets. When applicable, we evaluate generated motions on the HumanML3D [\[18\]](#page-8-7) dataset, which contains 44,970 motion annotations of 14,646 motion sequences from AMASS [\[47\]](#page-9-25) and HumanAct12 [\[16\]](#page-8-4) datasets.

Motion diffusion model. As DNO is plug-and-play and model-agnostic, it works with any trained motion diffusion model. When paired with MDM [\[73\]](#page-10-6), which is trained on the HumanML3D dataset, we name the combination DNO-MDM. We retrained MDM with Exponential Moving Averaging [\[24\]](#page-8-12) with no further modifications to help stabilize the model, which was previously found to produce inconsistent output between checkpoints.

DNO implementation details. We use Adam optimizer [\[36\]](#page-9-26) with a learning rate of 0.05, a linear warm-up for the first 50 steps, and a cosine learning rate schedule to zero for the entire optimization. We use a unit-normalized gradient. We set the coefficients λcont = 0.01, λdecorr = 10<sup>3</sup> , λobs = 1.0, and the perturbation amount γ = 0. This design choice is explored in Section [6.3.](#page-7-0)

For editing tasks, we obtain the initial x<sup>T</sup> from DDIM-100 inversion on MDM [\[73\]](#page-10-6) without text conditions. The subsequent optimization is run for 300 steps, which takes approximately 3 minutes on an Nvidia 3090 GPU. For refinement tasks, the optimization is run for 500 steps. DNO optimization with DDIM-10 and a batch size of 16 requires 18 GB of GPU memory. Optimization in all tasks is also done without text conditions to the diffusion model. Gradient checkpointing [\[5,](#page-8-26) [11\]](#page-8-24) can be used with DNO to reduce memory usage further for a low-memory GPU at the cost of more computation.

## 6.1. Motion Editing

Given an input motion, we generate 16 edited motions using DNO, each with a single randomly chosen keyframe between frames 60 and 90 (i.e., 2-3 seconds after the start). The objective is to change the pelvis position to a random target location within an area ranging from -2m to 2m with respect to the initial position. We conducted this experiment on 6 input motions, resulting in a total of 96 edited motions. To generate the input motions, we use the prompt "*a person is [action]*" with a predetermined *[action]*. During editing, DNO does not leverage the text prompt or the action class.

Evaluation metrics. We focus on three main aspects: (1) is the output motion realistic?, (2) can it fulfill the editing objective?, and (3) how much are the original motion characteristics preserved after editing?

To evaluate the realism of the output, we measure *Jitter* and *Foot skating ratio*, following prior work [\[18,](#page-8-7) [33,](#page-9-0) [82\]](#page-11-10). *Jitter* is a proxy for motion smoothness, measuring the mean changes in acceleration of all joints over time in

<span id="page-5-0"></span>Table 1. Motion editing evaluation on specific actions generated from MDM given the text prompts. We focus on actions that can be distinctly classified per frame basis. The Content Preservation scores are computed against the inputs.

| Action                     | Content ↑<br>Preserve | Objective ↓<br>Error (m) | Foot ↓<br>skating ratio | Jitter ↓ |
|----------------------------|-----------------------|--------------------------|-------------------------|----------|
| "jumping"                  |                       |                          |                         |          |
| Input                      | 1.00                  | 1.62                     | 0.01                    | 1.42     |
| GMD [33]                   | 0.64                  | 0.22                     | 0.12                    | 3.07     |
| DNO-MDM Edit               | 0.95                  | 0.00                     | 0.05                    | 1.31     |
| "doing a long jump"        |                       |                          |                         |          |
| Input                      | 1.00                  | 3.03                     | 0.01                    | 1.20     |
| GMD                        | 0.59                  | 0.22                     | 0.15                    | 4.10     |
| DNO-MDM Edit               | 0.92                  | 0.00                     | 0.07                    | 1.34     |
| "walking with raised hand" |                       |                          |                         |          |
| Input                      | 1.00                  | 2.76                     | 0.01                    | 0.24     |
| GMD                        | 0.65                  | 0.16                     | 0.11                    | 0.51     |
| DNO-MDM Edit               | 0.92                  | 0.00                     | 0.05                    | 0.32     |
| "crawling"                 |                       |                          |                         |          |
| Input                      | 1.00                  | 1.83                     | 0.06                    | 0.47     |
| GMD                        | 0.79                  | 0.15                     | 0.04                    | 2.76     |
| DNO-MDM Edit               | 0.94                  | 0.00                     | 0.08                    | 0.48     |

102m/s<sup>3</sup> . *Foot skating ratio* is a proxy for the incoherence between the trajectory and human motion, which measures the proportion of frames in which a foot skates for more than a certain distance (2.5 cm) while maintaining the contact with the ground (foot height < 5 cm).

To evaluate the success in the editing task, we use *Objective error*, which measures the distance between the target location of the edited joint at a specific frame and the corresponding joint position in the output motion. When the editing is performed on the ground plane, this metric measures the 2D distance.

While metrics leveraging action classifiers can provide a coarse evaluation of whether the action class is preserved or not after the editing, we require a more precise approach to quantify the content similarity. We report the ratio of frames in which both motions, before editing and after, perform the same action throughout the entire sequence, which we call *Content preservation ratio*. The criteria to determine if two actions are the same are action-dependent. The ratio will be 1 if both motions perform the same action in all frames. Since there are no established methods for per-frame content similarity, in this work, we consider only distinct actions for which the boundaries of the actions can be clearly defined: For "jumping", the action is defined as having both feet more than 5cm above the ground. For "raised hands", both hands need to be above the head. For the "crawling" motion, the head needs to be below 150 cm.

Results. As shown in Table [1,](#page-5-0) our proposed DNO outperforms GMD [\[33\]](#page-9-0), the state of the art in spatially-conditioned motion generation, in all actions and metrics, except for *Foot skating ratio* on "crawling." Overall, our method

<span id="page-6-1"></span>![](_page_6_Figure_0.jpeg)

Figure 3. Qualitative results from motion editing task. Each line indicates the starting and target location of the selected joint at a specific keyframe.

achieves significantly lower *Jitter*. Furthermore, our higher *Content preservation ratio* suggests that our approach retains the original content if editing is not necessary while successfully achieving the editing objectives, as indicated by the low *Objective error*s.

We present qualitative results of the edited motions across various tasks in the supplementary video. To follow a desired trajectory, we can use a set of target locations for the pelvis. Similarly, our DNO can edit motions to avoid obstacles without requiring explicit target keyframes. Furthermore, our method enables fine-grained editing by targeting individual joints such as the hand joint or an entire pose at a specific keyframe. Please refer to the supp. mat. for additional qualitative results.

## 6.2. Motion Refinement

We evaluate motion refinement performance on a random subset of size 300 from the test set of the HumanML3D dataset in two setups. First, we assess if DNO can recover a clean and plausible motion from a noisy input motion by adding Gaussian noise with a standard deviation of 5 cm to all axes of all joints. We then evaluate the motion refinement performance for an input motion with incomplete and noisy joint information in three scenarios: (1) *six joints*: head + two hands + two legs + pelvis, (2) *eight joints*: six joints + two shoulders, (3) *ten joints*: eight joints + two knees.

Evaluation metrics. As in the motion editing tasks, we report *Foot skating ratio* and *Jitter* as well as the established *Mean per-joint pose error* (MPJPE) between the groundtruth and predicted motions, and *FID* which measures the distance between the ground-truth and synthetic data distributions using a pretrained motion encoder [\[16\]](#page-8-4).

Results. As shown in Table [2,](#page-6-0) we observe that, in every experiment, DNO successfully improves the signal by reducing MPJPE beyond the input level and producing smooth and realistic motions. We demonstrate that DNO's performance scales with the base model by pairing it with

<span id="page-6-0"></span>Table 2. Noisy motion refinement results (noise std. 5 cm) on a subset of HumanML3D [\[18\]](#page-8-7) dataset. All experiments were run with N = 300. FIDs are computed against *Real* except *Real*'s FIDs which are computed against a holdout set from the dataset. HuMoR\* means we exclude the sequence when its optimization fails. DNO-MLD\* runs with 1,000 optimization steps.

|              | MPJPE ↓<br>observed (cm) | FID ↓ | Foot ↓<br>skating ratio | Jitter ↓ |
|--------------|--------------------------|-------|-------------------------|----------|
| Real         | 0.0                      | 0.50  | 0.08                    | 0.50     |
| All joints   |                          |       |                         |          |
| Noisy        | 11.1                     | 58.76 | 0.66                    | 28.60    |
| HuMoR* [59]  | 7.2                      | 0.87  | 0.13                    | 0.33     |
| GMD          | 25.7                     | 6.91  | 0.08                    | 0.81     |
| DNO-MDM      | 9.1                      | 0.69  | 0.07                    | 0.36     |
| DNO-MLD*     | 10.41                    | 0.27  | 0.11                    | 1.37     |
| DNO-GMD      | 7.0                      | 0.10  | 0.08                    | 0.89     |
| Six joints   |                          |       |                         |          |
| Noisy        | 11.6                     | 58.73 | 0.66                    | 28.61    |
| HuMoR*       | 8.8                      | 1.40  | 0.10                    | 0.20     |
| GMD          | 31.2                     | 7.07  | 0.08                    | 0.80     |
| DNO-MDM      | 8.8                      | 1.15  | 0.07                    | 0.38     |
| DNO-MLD*     | 11.4                     | 0.56  | 0.10                    | 1.35     |
| DNO-GMD      | 7.5                      | 0.36  | 0.07                    | 0.97     |
| Eight joints |                          |       |                         |          |
| Noisy        | 11.4                     | 58.73 | 0.66                    | 28.61    |
| HuMoR*       | 8.5                      | 1.19  | 0.14                    | 0.20     |
| GMD          | 29.9                     | 6.99  | 0.08                    | 0.80     |
| DNO-MDM      | 9.3                      | 0.85  | 0.07                    | 0.36     |
| DNO-MLD*     | 11.6                     | 0.42  | 0.11                    | 1.36     |
| DNO-GMD      | 7.2                      | 0.13  | 0.08                    | 0.96     |
| Ten joints   |                          |       |                         |          |
| Noisy        | 11.3                     | 58.73 | 0.66                    | 28.61    |
| HuMoR*       | 9.3                      | 1.06  | 0.13                    | 0.21     |
| GMD          | 28.4                     | 6.88  | 0.09                    | 0.80     |
| DNO-MDM      | 9.0                      | 0.66  | 0.08                    | 0.39     |
| DNO-MLD*     | 11.0                     | 0.43  | 0.11                    | 1.35     |
| DNO-GMD      | 7.1                      | 0.12  | 0.07                    | 0.96     |

three motion diffusion models (ordered by FID from highto-low on motion generation): MDM [\[73\]](#page-10-6), MLD [\[6\]](#page-8-10), and GMD [\[33\]](#page-9-0). The DNO-GMD combination outperforms the weaker combinations, DNO-MLD and DNO-MDM, and the optimization-based motion prior HuMoR [\[60\]](#page-10-24). And any DNO combination outperforms the guided diffusion method, GMD, in the motion refinement task. For the GMD baseline, we use the spatially-conditioned generation as in the original paper. While it also successfully denoises the motion (low Jitter 0.8), it struggles to satisfy fine pose signals and can only follow trajectory guidance leading to high MPJPE > 25 cm. We also tried SDEdit [\[48\]](#page-9-19) on motion refinement, however, it cannot deal with such a level of noise while being able to retain the original motion. Additional details and tasks such as motion completion, blending, and in-betweening, are discussed in the supplementary. Note that, unlike the task-specific methods such as [\[14,](#page-8-0) [31\]](#page-9-13), our method is never trained specifically for these tasks.

<span id="page-7-2"></span><span id="page-7-1"></span>Table 3. Ablation study on the noisy motion refinement task (std. 1 cm) results on a subset of HumanML3D [\[18\]](#page-8-7). All experiments were run with N = 300. FIDs are computed against *Real* except *Real*'s FIDs which are computed against a holdout set from the dataset.

|                              | MPJPE ↓<br>all (cm) |      | FID ↓ Jitter ↓ | Foot ↓<br>skating ratio |
|------------------------------|---------------------|------|----------------|-------------------------|
| Real                         | 0.0                 | 0.50 | 0.50           | 0.08                    |
| Noisy                        | 6.4                 | 9.91 | 5.87           | 0.15                    |
| DNO                          | 8.7                 | 0.66 | 0.33           | 0.07                    |
| − Normalized grad.           | 30.2                | 4.13 | 0.34           | 0.06                    |
| − Norm grad. & Ldecorr       | 25.5                | 4.02 | 0.34           | 0.06                    |
| − Ldecorr                    | 6.8                 | 0.65 | 0.40           | 0.10                    |
| − LR scheduler & warmup      | 8.4                 | 0.58 | 0.37           | 0.07                    |
| Perturb γ = 0 (DNO)          | 8.7                 | 0.66 | 0.33           | 0.07                    |
| Perturb γ = 2 × 10−4         | 8.5                 | 0.75 | 0.34           | 0.07                    |
| Perturb γ = 5 × 10−4         | 8.4                 | 0.73 | 0.36           | 0.07                    |
| Perturb γ = 10−3             | 9.1                 | 0.91 | 0.34           | 0.07                    |
| Optimize for 300 steps       | 12.1                | 1.48 | 0.31           | 0.07                    |
| Optimize for 500 steps (DNO) | 8.7                 | 0.66 | 0.33           | 0.07                    |
| Optimize for 700 steps       | 7.2                 | 0.54 | 0.36           | 0.07                    |
| DDIM 5 steps                 | 9.8                 | 0.90 | 0.36           | 0.09                    |
| DDIM 10 steps (DNO)          | 8.7                 | 0.66 | 0.33           | 0.07                    |
| DDIM 20 steps                | 7.9                 | 0.92 | 0.34           | 0.07                    |

### <span id="page-7-0"></span>6.3. Ablation studies

To motivate and justify DNO's design choices, we conducted experiments on a surrogate motion refinement task, where we add noise with a standard deviation of 1 cm to an input motion and seek to recover the original motion.

Normalized gradients. As ODESolver(·) involves iterative calls to the diffusion model, the gradients can become highly unstable. In our experiments, we observe that the gradient norms can span multiple orders of magnitude, leading to slow or poor convergence. Table [3](#page-7-1) demonstrates that our choice of using normalized gradients has improved the solutions by reducing MPJPE (which corresponds to the lower Lpose value) from 30.2 to 8.7 cm.

Decorrelating noise. DNO's main motivation is to identify a latent sample that is capable of generating a plausible motion while fulfilling the task-driven constraints. However, in practice, not all latent samples yield a plausible motion sample. One of the factors accounting for this is the uncorrelated random noise samples that the diffusion models are trained with. Correlated latent samples often results in poor motion samples. For example, the latent sample with all zeros has the highest likelihood, yet often generates lowquality motions. Our Ldecorr, motivated by this observation, discourages correlations across the time axis in x<sup>T</sup> . Table [3](#page-7-1) demonstrates that this loss directly contributes to the motion quality. It significantly improves the *Foot skating ratio* from 0.10 to 0.07 and the *Jitter* from 0.40 to 0.33, though at the cost of an increased optimization difficulty as indicated by the rise in MPJPE from 6.8 to 8.7. However, while it is rather straightforward to improve the MPJPE by increasing optimization steps, it is more challenging to fix the artifacts in the generated motion.

Random perturbation. As suggested in multiple studies [\[32,](#page-9-20) [75\]](#page-10-17), random perturbation is theoretically motivated as a facilitator of exploration, which helps optimization escape from local minima. To ensure that γ converges to zero, we tie γ with the learning rate scheduler, which has a warm-up period and a cosine decay. We found that small γ's may decrease the optimization errors only marginally, while a larger γ = 10−<sup>3</sup> has a negative impact on the optimization, yet they all produced motions with worse FIDs. While the results are not exhaustive, they serve as a piece of evidence to support our choice of γ = 0.

ODE solver steps. We experimented with varying the number of DDIM sampling steps from 5, 10, 20, which produce MPJPE scores of 9.8, 8.7, and 7.9, respectively. This suggests that higher DDIM steps may be able to capture finer motion detail. We chose 10-step sampling as a balanced choice between result quality and resource usage.

## 7. Discussion and Limitations

In this work, we proposed DNO a simple and versatile method to use a pretrained motion diffusion model as universal motion priors. We show that DNO can leverage the motion priors to achieve precise and fine-grained control for motion editing. Apart from editing, we demonstrate that our formulation can be extended to a wide range of motion tasks. Our method is plug-and-play and does not require training a new model for every new task. DNO is, however, not a perfect motion priors. It works better when the observation has good coverage of the human body. While the decorrelation loss does help, there are situations where DNO cannot easily project to a realistic motion. Ultimately, the effectiveness of our method is limited by the performance of the underlying diffusion model; nevertheless, we expect the performance of the base model to increase as the community is searching for better model designs and collecting more training data. As the inference speed of the diffusion model keeps increasing, we hope to overcome the speed limitation of our optimization framework to achieve interactive motion editing. In summary, our extensive studies on diffusion noise optimization effectiveness can serve as a useful basis for leveraging existing diffusion models to solve a wider range of tasks.

Acknowledgement. This work was supported by the SNSF project grant 200021 204840. Konpat Preechakul is funded by DARPA MCS. We sincerely thank Bram Wallace for an insightful discussion on EDICT and DOODL which inspired this project.

## References

- <span id="page-8-6"></span>[1] Chaitanya Ahuja and Louis-Philippe Morency. Language2pose: Natural language grounded pose forecasting. In *2019 International Conference on 3D Vision (3DV)*, pages 719–728. IEEE, 2019. [2](#page-1-0)
- <span id="page-8-11"></span>[2] Simon Alexanderson, Rajmund Nagy, Jonas Beskow, and Gustav Eje Henter. Listen, denoise, action! audio-driven motion synthesis with diffusion models. *arXiv preprint arXiv:2211.09707*, 2022. [2](#page-1-0)
- <span id="page-8-16"></span>[3] Yogesh Balaji, Seungjun Nah, Xun Huang, Arash Vahdat, Jiaming Song, Karsten Kreis, Miika Aittala, Timo Aila, Samuli Laine, Bryan Catanzaro, Tero Karras, and Ming-Yu Liu. eDiff-I: Text-to-Image diffusion models with an ensemble of expert denoisers. 2022. [3](#page-2-1)
- <span id="page-8-17"></span>[4] T Brooks, A Holynski, and A A Efros. InstructPix2Pix: Learning to follow image editing instructions. In *CVPR*. arxiv.org, 2023. [3](#page-2-1)
- <span id="page-8-26"></span>[5] Tianqi Chen, Bing Xu, Chiyuan Zhang, and Carlos Guestrin. Training deep nets with sublinear memory cost. *Proceedings of the 33rd International Conference on Machine Learning*, 2016. [6](#page-5-1)
- <span id="page-8-10"></span>[6] Xin Chen, Biao Jiang, Wen Liu, Zilong Huang, Bin Fu, Tao Chen, Jingyi Yu, and Gang Yu. Executing your commands via motion diffusion in latent space. *arXiv preprint arXiv:2212.04048*, 2022. [2,](#page-1-0) [7](#page-6-1)
- <span id="page-8-3"></span>[7] Xin Chen, Zhuo Su, Lingbo Yang, Pei Cheng, Lan Xu, Bin Fu, and Gang Yu. Learning variational motion prior for video-based motion capture. *arXiv preprint arXiv:2210.15134*, 2022. [2](#page-1-0)
- <span id="page-8-9"></span>[8] Xin Chen, Biao Jiang, Wen Liu, Zilong Huang, Bin Fu, Tao Chen, Jingyi Yu, and Gang Yu. Executing your commands via motion diffusion in latent space. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 18000–18010, 2023. [2](#page-1-0)
- <span id="page-8-18"></span>[9] Jooyoung Choi, Sungwon Kim, Yonghyun Jeong, Youngjune Gwon, and Sungroh Yoon. ILVR: Conditioning method for denoising diffusion probabilistic models. In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, pages 14367–14376, 2021. [3](#page-2-1)
- <span id="page-8-20"></span>[10] Hyungjin Chung, Byeongsu Sim, Dohoon Ryu, and Jong Chul Ye. Improving diffusion models for inverse problems using manifold constraints. In *Advances in Neural Information Processing Systems*, 2022. [3](#page-2-1)
- <span id="page-8-24"></span>[11] Kevin Clark, Paul Vicol, Kevin Swersky, and David J Fleet. Directly Fine-Tuning diffusion models on differentiable rewards. In *The Twelfth International Conference on Learning Representations*, 2024. [4,](#page-3-2) [6](#page-5-1)
- <span id="page-8-25"></span>[12] Jorge Cortes. Finite-time convergent gradient flows with ap- ´ plications to network consensus. *Automatica*, 42(11):1993– 2000, 2006. [4](#page-3-2)
- <span id="page-8-13"></span>[13] Prafulla Dhariwal and Alex Nichol. Diffusion models beat GANs on image synthesis. In *Advances in Neural Information Processing Systems*, pages 8780–8794, 2021. [3,](#page-2-1) [2](#page-1-0)
- <span id="page-8-0"></span>[14] Yuming Du, Robin Kips, Albert Pumarola, Sebastian Starke, Ali Thabet, and Artsiom Sanakoyeu. Avatars grow legs: Generating smooth human motion from sparse tracking inputs with diffusion model. In *Proceedings of the IEEE/CVF*

- *Conference on Computer Vision and Pattern Recognition*, pages 481–490, 2023. [2,](#page-1-0) [7](#page-6-1)
- <span id="page-8-1"></span>[15] Yinglin Duan, Tianyang Shi, Zhengxia Zou, Yenan Lin, Zhehui Qian, Bohan Zhang, and Yi Yuan. Singleshot motion completion with transformer. *arXiv preprint arXiv:2103.00776*, 2021. [2](#page-1-0)
- <span id="page-8-4"></span>[16] Chuan Guo, Xinxin Zuo, Sen Wang, Shihao Zou, Qingyao Sun, Annan Deng, Minglun Gong, and Li Cheng. Action2motion: Conditioned generation of 3d human motions. In *Proceedings of the 28th ACM International Conference on Multimedia*, pages 2021–2029, 2020. [2,](#page-1-0) [6,](#page-5-1) [7,](#page-6-1) [1](#page-0-0)
- <span id="page-8-22"></span>[17] Chuan Guo, Shihao Zou, Xinxin Zuo, Sen Wang, Wei Ji, Xingyu Li, and Li Cheng. Generating diverse and natural 3D human motions from text. In *2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 5152–5161. IEEE, 2022. [3](#page-2-1)
- <span id="page-8-7"></span>[18] Chuan Guo, Shihao Zou, Xinxin Zuo, Sen Wang, Wei Ji, Xingyu Li, and Li Cheng. Generating diverse and natural 3d human motions from text. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 5152–5161, 2022. [2,](#page-1-0) [6,](#page-5-1) [7,](#page-6-1) [8,](#page-7-2) [1](#page-0-0)
- <span id="page-8-8"></span>[19] Chuan Guo, Xinxin Zuo, Sen Wang, and Li Cheng. Tm2t: Stochastic and tokenized modeling for the reciprocal generation of 3d human motions and texts. In *Computer Vision– ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XXXV*, pages 580– 597. Springer, 2022. [2](#page-1-0)
- <span id="page-8-2"></span>[20] Felix G Harvey, Mike Yurick, Derek Nowrouzezahrai, and ´ Christopher Pal. Robust motion in-betweening. *ACM Transactions on Graphics (TOG)*, 39(4):60–1, 2020. [2](#page-1-0)
- <span id="page-8-23"></span>[21] Zhenliang He, Wangmeng Zuo, Meina Kan, Shiguang Shan, and Xilin Chen. AttGAN: Facial attribute editing by only changing what you want. *IEEE Trans. Image Process.*, 28 (11):5464–5478, 2019. [3](#page-2-1)
- <span id="page-8-19"></span>[22] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-Prompt image editing with cross attention control. 2022. [3](#page-2-1)
- <span id="page-8-21"></span>[23] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. In *NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications*. [3](#page-2-1)
- <span id="page-8-12"></span>[24] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In *Advances in Neural Information Processing Systems*, pages 6840–6851, 2020. [3,](#page-2-1) [6](#page-5-1)
- <span id="page-8-14"></span>[25] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, and Tim Salimans. Imagen video: High definition video generation with diffusion models. 2022. [3](#page-2-1)
- <span id="page-8-15"></span>[26] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. In *International Conference on Learning Representations*, 2022. [3](#page-2-1)
- <span id="page-8-5"></span>[27] Siyuan Huang, Zan Wang, Puhao Li, Baoxiong Jia, Tengyu Liu, Yixin Zhu, Wei Liang, and Song-Chun Zhu. Diffusionbased generation, optimization, and planning in 3d scenes. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 16750–16761, 2023. [2](#page-1-0)

- <span id="page-9-27"></span>[28] Inbar Huberman-Spiegelglas, Vladimir Kulikov, and Tomer Michaeli. An edit friendly DDPM noise space: Inversion and manipulations. 2023. [2](#page-1-0)
- <span id="page-9-18"></span>[29] Michael Janner, Yilun Du, Joshua B Tenenbaum, and Sergey Levine. Planning with diffusion for flexible behavior synthesis. In *International Conference on Machine Learning*, 2022. [3](#page-2-1)
- <span id="page-9-12"></span>[30] Biao Jiang, Xin Chen, Wen Liu, Jingyi Yu, Gang Yu, and Tao Chen. Motiongpt: Human motion as a foreign language. *arXiv preprint arXiv:2306.14795*, 2023. [2](#page-1-0)
- <span id="page-9-13"></span>[31] Jiaxi Jiang, Paul Streli, Huajian Qiu, Andreas Fender, Larissa Laich, Patrick Snape, and Christian Holz. Avatarposer: Articulated full-body pose tracking from sparse motion sensing. In *European Conference on Computer Vision*, pages 443–460. Springer, 2022. [2,](#page-1-0) [7](#page-6-1)
- <span id="page-9-20"></span>[32] Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Analyzing and improving the image quality of StyleGAN. In *2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 8107–8116, 2019. [3,](#page-2-1) [4,](#page-3-2) [5,](#page-4-1) [8](#page-7-2)
- <span id="page-9-0"></span>[33] Korrawe Karunratanakul, Konpat Preechakul, Supasorn Suwajanakorn, and Siyu Tang. Guided motion diffusion for controllable human motion synthesis. In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, pages 2151–2162, 2023. [2,](#page-1-0) [3,](#page-2-1) [4,](#page-3-2) [6,](#page-5-1) [7,](#page-6-1) [1](#page-0-0)
- <span id="page-9-5"></span>[34] Manuel Kaufmann, Emre Aksan, Jie Song, Fabrizio Pece, Remo Ziegler, and Otmar Hilliges. Convolutional autoencoders for human motion infilling. In *2020 International Conference on 3D Vision (3DV)*, pages 918–927. IEEE, 2020. [2](#page-1-0)
- <span id="page-9-10"></span>[35] Jihoon Kim, Jiseob Kim, and Sungjoon Choi. Flame: Freeform language-based motion synthesis & editing. *arXiv preprint arXiv:2209.00349*, 2022. [2](#page-1-0)
- <span id="page-9-26"></span>[36] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. *arXiv preprint arXiv:1412.6980*, 2014. [6](#page-5-1)
- <span id="page-9-15"></span>[37] Zhifeng Kong, Wei Ping, Jiaji Huang, Kexin Zhao, and Bryan Catanzaro. DiffWave: A versatile diffusion model for audio synthesis. In *International Conference on Learning Representations*, 2021. [3](#page-2-1)
- <span id="page-9-6"></span>[38] Hsin-Ying Lee, Xiaodong Yang, Ming-Yu Liu, Ting-Chun Wang, Yu-Ding Lu, Ming-Hsuan Yang, and Jan Kautz. Dancing to music. *Advances in neural information processing systems*, 32, 2019. [2](#page-1-0)
- <span id="page-9-7"></span>[39] Buyu Li, Yongchi Zhao, Shi Zhelun, and Lu Sheng. Danceformer: Music conditioned 3d dance generation with parametric motion transformer. In *Proceedings of the AAAI Conference on Artificial Intelligence*, pages 1272–1279, 2022. [2](#page-1-0)
- <span id="page-9-14"></span>[40] Haoying Li, Yifan Yang, Meng Chang, Huajun Feng, Zhihai Xu, Qi Li, and Yueting Chen. SRDiff: Single image Super-Resolution with diffusion probabilistic models. *Neurocomputing*, 479:47–59, 2022. [3](#page-2-1)
- <span id="page-9-1"></span>[41] Jiaman Li, Karen Liu, and Jiajun Wu. Ego-Body pose estimation via Ego-Head pose estimation. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 17142–17151, 2023. [2](#page-1-0)
- <span id="page-9-8"></span>[42] Ruilong Li, Shan Yang, David A Ross, and Angjoo Kanazawa. Ai choreographer: Music conditioned 3d dance

- generation with aist++. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 13401– 13412, 2021. [2](#page-1-0)
- <span id="page-9-21"></span>[43] Xingchao Liu, Xiwen Zhang, Jianzhu Ma, Jian Peng, and Qiang Liu. InstaFlow: One step is enough for High-Quality Diffusion-Based Text-to-Image generation. In *The Twelfth International Conference on Learning Representations*, 2024. [4](#page-3-2)
- <span id="page-9-22"></span>[44] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. DPM-Solver++: Fast solver for guided sampling of diffusion probabilistic models. 2022. [4](#page-3-2)
- <span id="page-9-23"></span>[45] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. DPM-Solver: A fast ODE solver for diffusion probabilistic model sampling in around 10 steps. In *Advances in Neural Information Processing Systems*, pages 5775–5787. Curran Associates, Inc., 2022. [4](#page-3-2)
- <span id="page-9-2"></span>[46] Jianxin Ma, Shuai Bai, and Chang Zhou. Pretrained diffusion models for unified human motion synthesis. *arXiv preprint arXiv:2212.02837*, 2022. [2](#page-1-0)
- <span id="page-9-25"></span>[47] Naureen Mahmood, Nima Ghorbani, Nikolaus F Troje, Gerard Pons-Moll, and Michael J Black. Amass: Archive of motion capture as surface shapes. In *Proceedings of the IEEE/CVF international conference on computer vision*, pages 5442–5451, 2019. [6,](#page-5-1) [2](#page-1-0)
- <span id="page-9-19"></span>[48] Chenlin Meng, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. SDEdit: Image synthesis and editing with stochastic differential equations. In *International Conference on Learning Representations (ICLR)*, 2022. [3,](#page-2-1) [7,](#page-6-1) [2](#page-1-0)
- <span id="page-9-24"></span>[49] Ryan Murray, Brian Swenson, and Soummya Kar. Revisiting normalized gradient descent: Fast evasion of saddle points. *IEEE Trans. Automat. Contr.*, 64(11):4818–4824, 2019. [4](#page-3-2)
- <span id="page-9-3"></span>[50] Xingang Pan, Ayush Tewari, Thomas Leimkuhler, Lingjie ¨ Liu, Abhimitra Meka, and Christian Theobalt. Drag your gan: Interactive point-based manipulation on the generative image manifold. In *ACM SIGGRAPH 2023 Conference Proceedings*, pages 1–11, 2023. [2](#page-1-0)
- <span id="page-9-4"></span>[51] Georgios Pavlakos, Vasileios Choutas, Nima Ghorbani, Timo Bolkart, Ahmed AA Osman, Dimitrios Tzionas, and Michael J Black. Expressive body capture: 3d hands, face, and body from a single image. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 10975–10985, 2019. [2](#page-1-0)
- <span id="page-9-9"></span>[52] Mathis Petrovich, Michael J Black, and Gul Varol. Action- ¨ conditioned 3d human motion synthesis with transformer vae. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 10985–10995, 2021. [2](#page-1-0)
- <span id="page-9-11"></span>[53] Mathis Petrovich, Michael J Black, and Gul Varol. Temos: ¨ Generating diverse human motions from textual descriptions. In *Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XXII*, pages 480–497. Springer, 2022. [2](#page-1-0)
- <span id="page-9-17"></span>[54] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. DreamFusion: Text-to-3D using 2D diffusion. 2022. [3](#page-2-1)
- <span id="page-9-16"></span>[55] Vadim Popov, Ivan Vovk, Vladimir Gogoryan, Tasnima Sadekova, and Mikhail Kudinov. Grad-TTS: A diffusion

- probabilistic model for Text-to-Speech. In *Proceedings of the 38th International Conference on Machine Learning*, pages 8671–8682, 2021. [3](#page-2-1)
- <span id="page-10-20"></span>[56] Konpat Preechakul, Nattanat Chatthee, Suttisak Wizadwongsa, and Supasorn Suwajanakorn. Diffusion autoencoders: Toward a meaningful and decodable representation. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 10619–10629, 2022. [4](#page-3-2)
- <span id="page-10-7"></span>[57] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In *International conference on machine learning*, pages 8748–8763. PMLR, 2021. [2](#page-1-0)
- <span id="page-10-14"></span>[58] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with CLIP latents. *ArXiv*, 2022. [3](#page-2-1)
- <span id="page-10-2"></span>[59] Davis Rempe, Tolga Birdal, Aaron Hertzmann, Jimei Yang, Srinath Sridhar, and Leonidas J Guibas. HuMoR: 3D human motion model for robust pose estimation. In *2021 IEEE/CVF International Conference on Computer Vision (ICCV)*. IEEE, 2021. [2,](#page-1-0) [7](#page-6-1)
- <span id="page-10-24"></span>[60] Davis Rempe, Tolga Birdal, Aaron Hertzmann, Jimei Yang, Srinath Sridhar, and Leonidas J Guibas. Humor: 3d human motion model for robust pose estimation. In *Proceedings of the IEEE/CVF international conference on computer vision*, pages 11488–11499, 2021. [7,](#page-6-1) [2](#page-1-0)
- <span id="page-10-15"></span>[61] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High-Resolution image ¨ synthesis with latent diffusion models. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 10684–10695, 2022. [3](#page-2-1)
- <span id="page-10-3"></span>[62] Alla Safonova and Jessica K. Hodgins. Construction and optimal search of interpolated motion graphs. *ACM Trans. Graph.*, 26(3):106–es, 2007. [2](#page-1-0)
- <span id="page-10-12"></span>[63] Chitwan Saharia, Jonathan Ho, William Chan, Tim Salimans, David J Fleet, and Mohammad Norouzi. Image Super-Resolution via iterative refinement. *IEEE Trans. Pattern Anal. Mach. Intell.*, 45:4713–4726, 2021. [3](#page-2-1)
- <span id="page-10-16"></span>[64] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour, Burcu Karagol Ayan, S Sara Mahdavi, Rapha Gontijo Lopes, Tim Salimans, Jonathan Ho, David J Fleet, and Mohammad Norouzi. Photorealistic Text-to-Image diffusion models with deep language understanding. In *Advances in Neural Information Processing Systems*, 2022. [3](#page-2-1)
- <span id="page-10-21"></span>[65] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. In *International Conference on Learning Representations*, 2022. [4](#page-3-2)
- <span id="page-10-0"></span>[66] Yonatan Shafir, Guy Tevet, Roy Kapon, and Amit H Bermano. Human motion diffusion as a generative prior. In *The Twelfth International Conference on Learning Representations*, 2024. [2](#page-1-0)
- <span id="page-10-9"></span>[67] Jascha Sohl-Dickstein, Eric A Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised

- learning using nonequilibrium thermodynamics. In *Proceedings of the 32nd International Conference on Machine Learning*, pages 2256–2265, 2015. [3](#page-2-1)
- <span id="page-10-19"></span>[68] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In *International Conference on Learning Representations (ICLR)*, 2021. [3,](#page-2-1) [4](#page-3-2)
- <span id="page-10-23"></span>[69] Jiaming Song, Qinsheng Zhang, Hongxu Yin, Morteza Mardani, Ming-Yu Liu, Jan Kautz, Yongxin Chen, and Arash Vahdat. Loss-guided diffusion models for plug-and-play controllable generation. In *International Conference on Machine Learning*, pages 32483–32498. PMLR, 2023. [4,](#page-3-2) [2](#page-1-0)
- <span id="page-10-10"></span>[70] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. In *Advances in Neural Information Processing Systems 32*, pages 11895– 11907, 2019. [3](#page-2-1)
- <span id="page-10-11"></span>[71] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-Based generative modeling through stochastic differential equations. In *International Conference on Learning Representations*, 2021. [3,](#page-2-1) [4](#page-3-2)
- <span id="page-10-22"></span>[72] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. 2023. [4](#page-3-2)
- <span id="page-10-6"></span>[73] Guy Tevet, Sigal Raab, Brian Gordon, Yonatan Shafir, Daniel Cohen-Or, and Amit H Bermano. Human motion diffusion model. In *The Eleventh International Conference on Learning Representations*, 2023. [2,](#page-1-0) [3,](#page-2-1) [6,](#page-5-1) [7](#page-6-1)
- <span id="page-10-8"></span>[74] Jonathan Tseng, Rodrigo Castellon, and C Karen Liu. Edge: Editable dance generation from music. *arXiv preprint arXiv:2211.10658*, 2022. [2](#page-1-0)
- <span id="page-10-17"></span>[75] Bram Wallace, Akash Gokul, Stefano Ermon, and Nikhil Naik. End-to-End diffusion latent optimization improves classifier guidance. In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, pages 7280–7290, 2023. [3,](#page-2-1) [4,](#page-3-2) [8](#page-7-2)
- <span id="page-10-18"></span>[76] Bram Wallace, Akash Gokul, and Nikhil Naik. EDICT: Exact diffusion inversion via coupled transformations. In *2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*. IEEE, 2023. [3,](#page-2-1) [4,](#page-3-2) [2](#page-1-0)
- <span id="page-10-4"></span>[77] Jiashun Wang, Huazhe Xu, Jingwei Xu, Sifei Liu, and Xiaolong Wang. Synthesizing long-term 3d human motion and interaction in 3d scenes. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 9401–9411, 2021. [2](#page-1-0)
- <span id="page-10-13"></span>[78] Daniel Watson, William Chan, Ricardo Martin-Brualla, Jonathan Ho, Andrea Tagliasacchi, and Mohammad Norouzi. Novel view synthesis with diffusion models. 2022. [3](#page-2-1)
- <span id="page-10-5"></span>[79] Yan Wu, Jiahao Wang, Yan Zhang, Siwei Zhang, Otmar Hilliges, Fisher Yu, and Siyu Tang. Saga: Stochastic wholebody grasping with contact. In *Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part VI*, pages 257–274. Springer, 2022. [2](#page-1-0)
- <span id="page-10-1"></span>[80] Yiming Xie, Varun Jampani, Lei Zhong, Deqing Sun, and Huaizu Jiang. OmniControl: Control any joint at any time for human motion generation. In *The Twelfth International Conference on Learning Representations*, 2024. [2](#page-1-0)

- <span id="page-11-0"></span>[81] Sijie Yan, Zhizhong Li, Yuanjun Xiong, Huahan Yan, and Dahua Lin. Convolutional sequence generation for skeletonbased action synthesis. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 4394– 4402, 2019. [2](#page-1-0)
- <span id="page-11-10"></span>[82] Xinyu Yi, Yuxiao Zhou, Marc Habermann, Soshi Shimada, Vladislav Golyanik, Christian Theobalt, and Feng Xu. Physical inertial poser (pip): Physics-aware real-time human motion tracking from sparse inertial sensors. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 13167–13178, 2022. [6](#page-5-1)
- <span id="page-11-9"></span>[83] Jiwen Yu, Yinhuai Wang, Chen Zhao, Bernard Ghanem, and Jian Zhang. Freedom: Training-free energy-guided conditional diffusion model. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 23174– 23184, 2023. [4,](#page-3-2) [2](#page-1-0)
- <span id="page-11-4"></span>[84] Ye Yuan, Jiaming Song, Umar Iqbal, Arash Vahdat, and Jan Kautz. Physdiff: Physics-guided human motion diffusion model. *arXiv preprint arXiv:2212.02500*, 2022. [2](#page-1-0)
- <span id="page-11-6"></span>[85] Jianrong Zhang, Yangsong Zhang, Xiaodong Cun, Shaoli Huang, Yong Zhang, Hongwei Zhao, Hongtao Lu, and Xi Shen. T2m-gpt: Generating human motion from textual descriptions with discrete representations. *arXiv preprint arXiv:2301.06052*, 2023. [2](#page-1-0)
- <span id="page-11-7"></span>[86] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models, 2023. [2](#page-1-0)
- <span id="page-11-5"></span>[87] Mingyuan Zhang, Zhongang Cai, Liang Pan, Fangzhou Hong, Xinying Guo, Lei Yang, and Ziwei Liu. Motiondiffuse: Text-driven human motion generation with diffusion model. *arXiv preprint arXiv:2208.15001*, 2022. [2](#page-1-0)
- <span id="page-11-3"></span>[88] Siwei Zhang, Yan Zhang, Federica Bogo, Marc Pollefeys, and Siyu Tang. Learning motion priors for 4d human body capture in 3d scenes. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 11343– 11353, 2021. [2](#page-1-0)
- <span id="page-11-1"></span>[89] Yan Zhang, Michael J Black, and Siyu Tang. Perpetual motion: Generating unbounded human motion. *arXiv preprint arXiv:2007.13886*, 2020. [2](#page-1-0)
- <span id="page-11-2"></span>[90] Rui Zhao, Hui Su, and Qiang Ji. Bayesian adversarial human motion synthesis. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 6225–6234, 2020. [2](#page-1-0)
- <span id="page-11-8"></span>[91] Kaiwen Zheng, Cheng Lu, Jianfei Chen, and Jun Zhu. DPM-Solver-v3: Improved diffusion ODE solver with empirical model statistics. In *Thirty-seventh Conference on Neural Information Processing Systems*, 2023. [4](#page-3-2)

# Optimizing Diffusion Noise Can Serve As Universal Motion Priors \*\*Appendix\*\*

## A. Motion completion

Table [A.1](#page-12-0) shows the results of the motion completion task. The task is evaluated under the same setting as the motion refinement task in the main paper, except that the ground truth joint locations are given without added noise. The goal of the task is to generate the full-body motion given partial joint observations.

The results are consistent with the motion denoising experiment where DNO's performance scale with base model, DNO-GMD outperforms other baselines regarding MPJPE and FID, while HuMoR tends to produce smoother motions.

<span id="page-12-0"></span>Table A.1. Motion completion results on a subset of HumanML3D [\[18\]](#page-8-7) dataset. All experiments were run with N = 300 except DNO-MLD\* which runs with 1,000 optimization steps. FIDs are computed against *Real*. The *Real*'s FIDs are computed against a holdout set from the dataset. HuMoR\* means we exclude the sequence when its optimization fails.

| MPJPE ↓<br>observed (cm)<br>0.0 | FID ↓<br>0.50                                                                                               | Foot ↓<br>skating ratio<br>0.08                              | Jitter ↓<br>0.50                                             |
|---------------------------------|-------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------|--------------------------------------------------------------|
|                                 |                                                                                                             |                                                              |                                                              |
|                                 |                                                                                                             |                                                              |                                                              |
|                                 |                                                                                                             |                                                              |                                                              |
|                                 |                                                                                                             |                                                              |                                                              |
|                                 |                                                                                                             |                                                              | 0.17                                                         |
|                                 | 7.08                                                                                                        | 0.08                                                         | 0.79                                                         |
|                                 | 1.31                                                                                                        | 0.07                                                         | 0.33                                                         |
|                                 | 0.67                                                                                                        | 0.10                                                         | 1.29                                                         |
|                                 | 0.30                                                                                                        | 0.07                                                         | 0.92                                                         |
|                                 |                                                                                                             |                                                              |                                                              |
|                                 |                                                                                                             |                                                              | 0.17                                                         |
|                                 |                                                                                                             |                                                              | 0.79                                                         |
|                                 | 0.98                                                                                                        | 0.07                                                         | 0.34                                                         |
|                                 | 0.51                                                                                                        | 0.11                                                         | 1.30                                                         |
|                                 | 0.12                                                                                                        | 0.08                                                         | 0.93                                                         |
|                                 |                                                                                                             |                                                              |                                                              |
|                                 |                                                                                                             |                                                              | 0.18                                                         |
|                                 |                                                                                                             |                                                              | 0.79                                                         |
|                                 |                                                                                                             |                                                              | 0.36                                                         |
|                                 |                                                                                                             |                                                              | 1.31                                                         |
|                                 |                                                                                                             |                                                              | 0.93                                                         |
|                                 | 8.7<br>31.1<br>8.5<br>11.0<br>6.6<br>8.4<br>29.8<br>8.7<br>11.3<br>6.6<br>8.3<br>28.4<br>8.6<br>11.3<br>6.5 | 1.53<br>1.22<br>7.06<br>1.06<br>6.88<br>0.80<br>0.49<br>0.11 | 0.13<br>0.13<br>0.08<br>0.12<br>0.08<br>0.07<br>0.11<br>0.07 |

## B. Additional motion-related tasks

Under the DNO framework, the same method presented in Algorithm 1 can be adapted to many motion-related tasks without retraining the model. In this section, we present different settings of DNO for motion blending and motion in-betweening tasks. The qualitative results are presented in our supplementary video.

## B.1. Motion Blending.

For motion blending, the goal is to smoothly transition from one distinct action to another. The inputs are two motion sequences and the expected output is a long motion that combines the two input motions together. With DNO, the problem can be formulated in the same manner as the motion refinement and completion task (Sec. 5.2), where the joint locations of the concatenated input motions are used as targets and the optimization is initialized from a random x<sup>T</sup> ∼ N (0, I). To facilitate a smooth blending between motions, we define a 10-frame window around the concatenated frame as a transition period where we drop all target joints. Consequentially, the model needs to fill in this transition according to its motion prior. We set the content criterion λcont = 0.0, λdecorr = 10<sup>3</sup> , the perturbation amount γ = 0, and the optimization step to 1000 for this task.

### B.2. Motion In-betweening.

For motion in-betweening, the inputs are the starting pose and the ending pose, given by the location of each joint. The goal is to generate the in-between motion according to those two poses. Similar to motion blending and motion completion, this task can be formulated as an optimization with partial observation as targets. We use the same setting as in the motion blending task with the only difference being the number of target joints.

# C. Why we do not report FID for motion editing.

As the motion Frechet inception distance (FID) is a mea- ´ surement *between two data distributions*, it requires a large number of samples in both datasets [\[16\]](#page-8-4). For the motion editing task, only one motion sequence exists before editing and only a few sequences exist after editing, thus there are not enough data points to measure a meaningful FID.

## D. GMD implementation details

To compare with GMD [\[33\]](#page-9-0), we use the released model with Emphasis projection and Dense gradient propagation for all tasks. The trajectory model is not used. When conditioned on the ground locations, we use the provided point-to-point imputing method until t = 20 as suggested in their experiments. The guidance is provided using the same criterion terms used in our method for all tasks. As GMD does not support editing while preserving the content, in the editing task, we instead provided the text prompt together with the target condition as inputs for the motion editing task. The observed joints are used without text conditioning for noisy motion refinement and motion completion.

## E. HuMoR implementation details

We use the officially released version of HuMoR [\[60\]](#page-10-24) which uses both the pose prior and motion prior for evaluations. We note that the released model is trained on a subset of the AMASS dataset [\[47\]](#page-9-25) at 30 FPS which does not entirely overlap with the 20 FPS sequences in the HumanML3D dataset [\[18\]](#page-8-7). The HuMoR code accepts the FPS number and does its interpolation to match the input with its learned motion prior. We also noticed that HuMoR optimization fails on some sequences in the test set, resulting in NaN error. We removed those sequences when computing the metrics for HuMoR.

# F. SDEdit on motion refinement

We include SDEdit [\[48\]](#page-9-19) results on the motion refinement task in Tab. [F.1.](#page-13-0) We tried all possible hyperparameters t in 100 increments from 100-1000. Except for the very extreme values of t = 1000, SDEdit exhibits unrealistic motions affected by the presence of noise in the original motion representation with high FID and Jitter. At t = 1000, SDEdit becomes a normal DDPM generative process, and no original content is preserved. In all cases, SDEdit fails to preserve the original content suggested by very high MPJPE. Note that the high FID of 29.73 for t = 1000 comes from the fact that the motions generated from MDM without any text prompts are heavily biased toward simple motions, e.g. standing, which do not capture the wide range of possible motions in the HumanML3D dataset. We conclude that SDEdit is not an effective motion refinement method.

<span id="page-13-0"></span>Table F.1. SDEdit [\[48\]](#page-9-19) results on the motion refinement task (noise std. = 5 cm.). We used the default number of repetitions k = 3 in all of the following experiments.

|                 | MPJPE ↓<br>observed (cm) | FID ↓ | Foot ↓<br>skating ratio | Jitter ↓ |
|-----------------|--------------------------|-------|-------------------------|----------|
| All joints      |                          |       |                         |          |
| Real            | 0.0                      | 0.48  | 0.08                    | 0.50     |
| Noisy           | 11.4                     | 58.82 | 0.66                    | 28.61    |
| SDEdit (t=100)  | 346.3                    | 36.10 | 0.12                    | 3.12     |
| SDEdit (t=200)  | 313.6                    | 33.92 | 0.14                    | 1.61     |
| SDEdit (t=300)  | 288.7                    | 32.47 | 0.14                    | 1.11     |
| SDEdit (t=400)  | 259.4                    | 31.24 | 0.13                    | 1.01     |
| SDEdit (t=500)  | 226.3                    | 30.53 | 0.12                    | 0.86     |
| SDEdit (t=600)  | 187.2                    | 28.99 | 0.10                    | 0.93     |
| SDEdit (t=700)  | 150.2                    | 27.76 | 0.09                    | 1.48     |
| SDEdit (t=800)  | 122.7                    | 30.83 | 0.08                    | 2.35     |
| SDEdit (t=900)  | 79.2                     | 19.09 | 0.06                    | 1.33     |
| SDEdit (t=1000) | 68.2                     | 29.73 | 0.00                    | 0.04     |

## F.1. Qualitative Results

Please check our supplementary video for qualitative results from DNO in all tasks including motion editing, refinement, blending, and in-betweening.

## G. Differences from guided diffusion method

While both DNO and loss-guided or classifier-guidance diffusion methods [\[13,](#page-8-13) [33,](#page-9-0) [69,](#page-10-23) [83\]](#page-11-9) can be used to produce motion samples with specific guidance objectives, these processes are completely different.

The loss-guided or classifier-guidance diffusion method (LGD) is a *sampling technique* that uses the gradient of a loss function to steer the trajectory of the diffusion sampling. The process is done in one full-chain sampling and outputs a *sample* that follows the guidance.

In contrast, DNO is a *latent optimization technique* where each optimization step involves a full-chain diffusion sampling. The output is a *latent code* whose decoded sample follows the guidance.

The differences between DNO and LGD also have the following practical implications:

1) Latent optimization (DNO) does not have approximation error during guidance because it operates on the exact output x<sup>0</sup> from solving the full-chain diffusion process via an ODE solver, while in LGD, the loss criterion L(·) is *approximately* computed on an expected xˆ = E[x0|xt] as explained in Eq. 7, 8 of [\[69\]](#page-10-23) and [\[83\]](#page-11-9) as follows:

$$\mathcal{L}(\mathbf{x}_t) = \mathbb{E}_{p(\mathbf{x}_0|\mathbf{x}_t)} \mathcal{L}(\mathbf{x}_0)$$

$$\approx \mathcal{L}(\hat{\mathbf{x}})$$

The approximation error is severe when Var[x0|xt] is large, particularly near the beginning where T ∼ 1000. This means the guidance is only effective near the end of the denoising process. Empirically, we observe that GMD [\[33\]](#page-9-0) does not reach the targets as well compared to DNO (25.7 vs 9.1 MPJPE, Table 2).

- 2) The latent space can serve as universal priors for valid motions. DNO can answer the question "What is the closest valid motion to the input x?" by optimizing latent x<sup>T</sup> to produce a valid motion x<sup>0</sup> that best matches the input x. GMD, an LGD method, is ineffective at generating valid motions from noisy inputs as shown in the refinement task in Table 2.
- 3) LGD cannot easily preserve content (Tab. 1). As editing in LGD is equivalent to new conditional sampling with the input motion, it is not obvious how to specify what aspects of the input motion are to be preserved and how to preserve them with LGD.

Most recent developments in diffusion image editing operate on the latent noise space with the help of the conditional inversion process [\[28,](#page-9-27) [76\]](#page-10-18). This direction further bolsters the merits of DNO as a latent approach for content preservation. The latent space naturally provides smooth transitions between valid motions; samples that are close in latent space x<sup>T</sup> are also likely to be close in motion space x0. DNO enables content-preserving editing through minimal updates on the latent space and results in a minimal change in the input motion to fulfill the objectives.

As shown in the experiments, DNO enables a wide range of tasks that require precise control, motion prior, or content preservation, which cannot be effectively solved with LGD.