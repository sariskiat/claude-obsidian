---
type: paper
slug: on-the-closed-form-of-flow-matching-generalization-does-not-arise-from-target-stochasticity
title: 'On the Closed-Form of Flow Matching: Generalization Does Not Arise from Target Stochasticity'
authors: Quentin Bertrand, Anne Gagneux, Mathurin Massias, Remi Emonet
source_path: .paper-scholar/on-the-closed-form-of-flow-matching-generalization-does-not-arise-from-target-stochasticity
ingested_at: '2026-06-05 05:47:20'
authors_list: []
sections:
- id: 353
  heading: Abstract
  role: abstract
  order_index: 0
  summary: Modern deep generative models can now produce high-quality synthetic samples that are often indistinguishable from real training data. A growing body of research aims to understand why recent methods, such as diffusion and flow matching techniques, generalize so effectively.
- id: 354
  heading: 1 Introduction
  role: introduction
  order_index: 1
  summary: Recent deep generative models, such as diffusion (Sohl-Dickstein et al., 2015; Ho et al., 2020; Song et al., 2021) and flow matching models (Lipman et al., 2023; Albergo and Vanden-Eijnden, 2023; Liu et al., 2023), have achieved remarkable success in synthesizing realistic data across a wide range of domains. State-of-the-art diffusion and flow matching methods are now capable of producing multi-modal outputs that are virtually indistinguishable from human-generated content, including images (Stability AI, 2023), audio (Borsos et al., 2023), video (Villegas et al., 2022; Brooks et al., 2024), and text (Gong et al., 2023; Xu et al., 2025).
- id: 355
  heading: Contributions.
  role: other
  order_index: 2
  summary: '- We challenge the prevailing belief that generalization in flow matching stems from an inherently noisy loss (Section 3.1). This assumption, largely supported by studies in low-dimensional settings, fails to hold in realistic high-dimensional data regimes.'
- id: 356
  heading: <span id="page-1-0"></span>2 Recalls on conditional flow matching
  role: other
  order_index: 3
  summary: Let $p_0 = \mathcal{N}(0, \mathrm{Id})$ be the source distribution<sup>2</sup> and $p_{\mathrm{data}}$ the data distribution. We are given n data points $x^{(1)}, \ldots, x^{(n)} \sim p_{\mathrm{data}}, x^{(i)} \in \mathbb{R}^d$ .
- id: 357
  heading: <span id="page-2-0"></span>3 Investigating the key sources of generalization
  role: other
  order_index: 4
  summary: In this section, we investigate the key sources of flow matching generalization using the closed-form formula of its velocity field. First in Section 3.1 we challenge the claim that generalization stems from the stochastic approximation $u^{\text{cond}}$ of the optimal velocity field $\hat{u}^*$ .
- id: 358
  heading: <span id="page-3-0"></span>3.1 Target stochasticity is not what you need
  role: other
  order_index: 5
  summary: One recent hypothesis is that generalization arises from the fact that the regression target $u^{\mathrm{cond}}$ of conditional flow matching is only a stochastic estimate of $\hat{u}^*$ . The fact that the target regression objective only equals the true objective on average is referred to by Vastola (2025) as "generalization through variance".
- id: 359
  heading: <span id="page-4-0"></span>3.2 Failure to learn the optimal velocity field
  role: other
  order_index: 6
  summary: This subsection investigates how well the learned velocity field $u_{\theta}$ approximates the optimal/ideal velocity field $\hat{u}^{\star}$ , and how the quality of this approximation correlates with generalization. To do so, we propose the following experiment.
- id: 360
  heading: <span id="page-5-0"></span>3.3 When does generalization arise?
  role: other
  order_index: 7
  summary: To investigate whether the failure to approximate $\hat{u}^*$ matters the most at small or large values of t, we carry out the following experiment.
- id: 361
  heading: <span id="page-6-0"></span>4 Learning with the closed-form formula
  role: other
  order_index: 8
  summary: In this section, in order to discard the impact of stochastic target on the generalization, we propose to directly regress against the closed-form formula in Equation (6).
- id: 362
  heading: 4.1 Empirical flow matching
  role: other
  order_index: 9
  summary: Regressing against the closed-form $\hat{u}^{\star}$ , defined in Equation (6), at a point $(x_t, t)$ requires computing a weighted sum of the conditional velocity fields over *all* the n training points $x^{(i)}$ . For a dataset of n samples of size d, and a batch of size $|\mathcal{B}|$ , computing the weights of the exact closed-form formula $\hat{u}^{\star}(x,t)$ of flow matching requires $\mathcal{O}(n \times |\mathcal{B}| \times d)$ .
- id: 363
  heading: <span id="page-7-2"></span>Algorithm 1 Vanilla Flow Matching
  role: method
  order_index: 10
  summary: ''
- id: 364
  heading: for k in $1, \ldots, n_{\text{iter}}$ do $\begin{vmatrix} t \sim \mathcal{U}([0,1]) \\ x_0 \sim \mathcal{N}(0, \text{Id}), x_1 \sim \hat{p}_{\text{data}}, \\ x_t = (1-t)x_0 + tx_1 \\ u^{\text{cond}}(x_t, t) = \frac{x_1 - x_t}{1-t} = x_1 - x_0 \\ \mathcal{L}(\theta) = \left\| u_{\theta}(x_t, t) - u^{\text{cond}}(x_t, t) \right\|^2 \\ \text{Compute } \nabla \mathcal{L}(\theta) \text{ and update } \theta$ $\mathbf{return} \ u_{\theta}$
  role: other
  order_index: 11
  summary: ''
- id: 365
  heading: <span id="page-7-1"></span>Algorithm 2 Empirical Flow Matching
  role: method
  order_index: 12
  summary: '$\begin{aligned} & \operatorname{param}: M \text{ } /\!\!/ \operatorname{Number} \text{ of samples in the empirical mean} \\ & \operatorname{for} k \text{ } in 1, \ldots, n_{\mathrm{iter}} \operatorname{do} \\ & x_0 \sim \mathcal{N}(0, \mathrm{Id}), x_1 \sim \hat{p}_{\mathrm{data}}, t \sim \mathcal{U}([0, 1]) \\ & x_t = (1 - t)x_0 + tx_1 \\ & b^{(1)} = x_1 \\ & \forall j \in [\![2, M]\!], b^{(j)} \sim \hat{p}_{\mathrm{data}} \text{ } /\!\!/ \operatorname{Samples} \operatorname{from} \hat{p}_{\mathrm{data}} \\ & \hat{u}_M^\star(x_t, t) = \sum_{j=1}^M \frac{b^{(j)} - x_t}{1 - t} \cdot \left[\operatorname{softmax} \left(-\frac{\|x_t - t \cdot b\|^2}{2(1 - t)^2}\right)\right]_j \\ & \mathcal{L}(\theta) = \|u_\theta(x_t, t) - \hat{u}_M^\star(x_t, t)\|^2 \\ & \operatorname{Compute} \nabla \mathcal{L}(\theta) \text{ and update } \theta \end{aligned}$'
- id: 366
  heading: <span id="page-7-0"></span>4.2 Experiments
  role: experiment
  order_index: 13
  summary: We now learn with empirical flow matching (EFM, Equation (7) and Algorithm 2) in practical high-dimensional settings. Our goal with this empirical investigation is first to observe if regressing against a more deterministic target leads to performance improvement/degradation.
- id: 367
  heading: <span id="page-8-0"></span>5 Related work
  role: other
  order_index: 14
  summary: 'The existing literature related to our study can be roughly divided into three approaches: leveraging the closed-form, studies on the memorization vs generalization, and characterization of the different phases of the generating dynamics.'
- id: 368
  heading: 6 Conclusion, limitations and broader impact
  role: conclusion
  order_index: 15
  summary: Conclusion. By challenging the assumption that stochasticity in the loss function is a key driver of generalization, our findings help clarify the role of approximation of the exact velocity field in flow matching models.
- id: 369
  heading: 7 Acknowledgments
  role: other
  order_index: 16
  summary: The authors thank the Blaise Pascal Center for its computational support, using the SIDUS [\(Quemener](#page-11-18) [and Corvellec,](#page-11-18) [2013\)](#page-11-18) solution.
- id: 370
  heading: References
  role: other
  order_index: 17
  summary: '- <span id="page-10-0"></span>M. S.'
- id: 371
  heading: NeurIPS Paper Checklist
  role: other
  order_index: 18
  summary: 'The checklist is designed to encourage best practices for responsible machine learning research, addressing issues of reproducibility, transparency, research ethics, and societal impact. Do not remove the checklist: The papers not including the checklist will be desk rejected.'
- id: 372
  heading: 'IMPORTANT, please:'
  role: other
  order_index: 19
  summary: '- Delete this instruction block, but keep the section heading "NeurIPS Paper Checklist", - Keep the checklist subsection headings, questions/answers and guidelines below. - Do not modify the questions and only use the provided macros for your answers.'
- id: 373
  heading: 1. Claims
  role: other
  order_index: 20
  summary: 'Question: Do the main claims made in the abstract and introduction accurately reflect the paper''s contributions and scope?'
- id: 374
  heading: 'Guidelines:'
  role: other
  order_index: 21
  summary: '- The answer NA means that the abstract and introduction do not include the claims made in the paper. - The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations.'
- id: 375
  heading: 2. Limitations
  role: other
  order_index: 22
  summary: 'Question: Does the paper discuss the limitations of the work performed by the authors?'
- id: 376
  heading: 3. Theory assumptions and proofs
  role: other
  order_index: 23
  summary: 'Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?'
- id: 377
  heading: 'Guidelines:'
  role: other
  order_index: 24
  summary: '- The answer NA means that the paper does not include theoretical results. - All the theorems, formulas, and proofs in the paper should be numbered and cross-referenced.'
- id: 378
  heading: 4. Experimental result reproducibility
  role: experiment
  order_index: 25
  summary: 'Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?'
- id: 379
  heading: 5. Open access to data and code
  role: other
  order_index: 26
  summary: 'Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?'
- id: 380
  heading: 'Guidelines:'
  role: other
  order_index: 27
  summary: '- The answer NA means that paper does not include experiments requiring code. - Please see the NeurIPS code and data submission guidelines ([https://nips.cc/public/](https://nips.cc/public/guides/CodeSubmissionPolicy) [guides/CodeSubmissionPolicy](https://nips.cc/public/guides/CodeSubmissionPolicy)) for more details.'
- id: 381
  heading: 6. Experimental setting/details
  role: experiment
  order_index: 28
  summary: 'Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?'
- id: 382
  heading: 'Guidelines:'
  role: other
  order_index: 29
  summary: • The answer NA means that the paper does not include experiments.
- id: 383
  heading: 7. Experiment statistical significance
  role: experiment
  order_index: 30
  summary: 'Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?'
- id: 384
  heading: 'Guidelines:'
  role: other
  order_index: 31
  summary: '- The answer NA means that the paper does not include experiments. - The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.'
- id: 385
  heading: 8. Experiments compute resources
  role: experiment
  order_index: 32
  summary: 'Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?'
- id: 386
  heading: 'Guidelines:'
  role: other
  order_index: 33
  summary: '- The answer NA means that the paper does not include experiments. - The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.'
- id: 387
  heading: 9. Code of ethics
  role: other
  order_index: 34
  summary: 'Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics <https://neurips.cc/public/EthicsGuidelines>?'
- id: 388
  heading: 10. Broader impacts
  role: other
  order_index: 35
  summary: 'Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?'
- id: 389
  heading: 'Guidelines:'
  role: other
  order_index: 36
  summary: '- The answer NA means that there is no societal impact of the work performed. - If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.'
- id: 390
  heading: 11. Safeguards
  role: other
  order_index: 37
  summary: 'Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?'
- id: 391
  heading: 'Guidelines:'
  role: other
  order_index: 38
  summary: '- The answer NA means that the paper poses no such risks. - Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.'
- id: 392
  heading: 12. Licenses for existing assets
  role: other
  order_index: 39
  summary: 'Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?'
- id: 393
  heading: 13. New assets
  role: other
  order_index: 40
  summary: 'Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?'
- id: 394
  heading: 14. Crowdsourcing and research with human subjects
  role: other
  order_index: 41
  summary: 'Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?'
- id: 395
  heading: 15. Institutional review board (IRB) approvals or equivalent for research with human subjects
  role: other
  order_index: 42
  summary: 'Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?'
- id: 396
  heading: 16. Declaration of LLM usage
  role: other
  order_index: 43
  summary: 'Question: Does the paper describe the usage of LLMs if it is an important, original, or nonstandard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.'
- id: 397
  heading: A Proofs of Section 2
  role: other
  order_index: 44
  summary: $$\hat{u}^{\star}(x,t) = \sum_{i=1}^{n} u^{\text{cond}}(x, z = x^{(i)}, t) \cdot \frac{p(x|z = x^{(i)}, t)}{\sum_{i'=1}^{n} p(x|z = x^{(i')}, t)} . \tag{12}$$
- id: 398
  heading: <span id="page-20-0"></span>A.1 Proof of Proposition 1
  role: other
  order_index: 45
  summary: '*Proof.* • In the case where $z \sim \hat{p}_{\text{data}}$ , conditional probability writes'
- id: 399
  heading: <span id="page-21-0"></span>B Additional details and comments on empirical flow matching
  role: other
  order_index: 46
  summary: First, recalls on the optimal velocity (Equation (6)) and the empirical flow matching loss (Equations (7) and (8)) are provided in Section B.1. The unbiasedness of the estimator is presented in Section B.2, and its proof is in Section B.3.
- id: 400
  heading: <span id="page-21-3"></span>B.1 Recalls
  role: other
  order_index: 47
  summary: 'The closed-form formula of the "optimal" velocity field is:'
- id: 401
  heading: <span id="page-21-1"></span>B.2 Theoretical properties of the proposed estimator
  role: other
  order_index: 48
  summary: First, we discuss below the relation between Proposition 2 and the sampling literature.
- id: 402
  heading: <span id="page-22-0"></span>B.3 Proof of Proposition [2](#page-6-4)
  role: other
  order_index: 49
  summary: We first recall Section [B,](#page-21-0) which we prove in this section.
- id: 403
  heading: C Additional experiments
  role: experiment
  order_index: 50
  summary: 'We present below the results for the MNIST dataset. The conclusions atre the same as for the CIFAR-10 and CelebA 64 × 64: regressing against a more deterministic velocity field does not hurt generalization.'
- id: 404
  heading: <span id="page-26-0"></span>D Experiments details
  role: experiment
  order_index: 51
  summary: For all the experiment we used all the same learning hyperparameters, the default ones form Tong et al. (2024).
- id: 405
  heading: D.1 Compute time
  role: other
  order_index: 52
  summary: Given that regressing against an estimate of the closed-form, EFM, seems to improve on CFM, one may wonder what is the additional cost induced by EFN. To alleviate the non-linearity of GPU computing (parallelism may cause some discontinuities in terms of costs), we ran an exhaustive set of timing experiments, varying the batch size and the EFM sample size.
- id: 406
  heading: <span id="page-26-2"></span>D.2 Figures 1a and 1c
  role: other
  order_index: 53
  summary: 'For Figure 1a no deep learning is involved: the datasets 2-moons and CIFAR-10 are loaded. Then, 256 points from $p_0 \times \hat{p}_{\text{data}}$ are drawn, and one computes the mean of the cosine similarities between $\hat{u}^*((1-t)x_0+tx_1,t)$ and $u^{\text{cond}}((1-t)x_0+tx_1,z=x_1,t)=x_1-x_0$ , for each value of $t \in \{0,1/100,2/100,\ldots,99/100\}$ .'
- id: 407
  heading: D.3 Figure 2
  role: other
  order_index: 54
  summary: In Figure 2, networks are trained with a vanilla conditional flow matching, with the standard 34 million parameters U-Net for diffusion by Nichol and Dhariwal (2021), with default settings from the torchfm codebase <sup>4</sup> (Tong et al., 2024). Training uses the CFM loss.
- id: 408
  heading: D.4 Figure [3](#page-5-1)
  role: other
  order_index: 55
  summary: In Figure [3,](#page-5-1) for each dataset (CIFAR-10 and CelebA 64 × 64), one network is trained using a vanilla conditional flow matching with the default parameters of [Tong et al.](#page-12-14) [\(2024\)](#page-12-14) (the most important ones are recalled in Table [3\)](#page-26-1). Then images are generated first following the closed-form formula of the optimal velocity field uˆ ⋆ from 0 to τ .
- id: 409
  heading: <span id="page-27-0"></span>D.5 Figure [4](#page-8-1)
  role: other
  order_index: 56
  summary: For experiments involving training on CIFAR-10 (Figures [2](#page-4-1) and [3\)](#page-5-1), we rely on the standard 34 million parameters U-Net for diffusion by [Nichol and Dhariwal](#page-11-19) [\(2021\)](#page-11-19), with default settings from the torchfm codebase [\(Tong et al.,](#page-12-14) [2024\)](#page-12-14). For each algorithm, the networks are trained for 500k iterations with batch size 128, *i.e.,* 1280 epochs.
---

# On the Closed-Form of Flow Matching: Generalization Does Not Arise from Target Stochasticity

## [abstract] Abstract
Modern deep generative models can now produce high-quality synthetic samples that are often indistinguishable from real training data. A growing body of research aims to understand why recent methods, such as diffusion and flow matching techniques, generalize so effectively.

## [introduction] 1 Introduction
Recent deep generative models, such as diffusion (Sohl-Dickstein et al., 2015; Ho et al., 2020; Song et al., 2021) and flow matching models (Lipman et al., 2023; Albergo and Vanden-Eijnden, 2023; Liu et al., 2023), have achieved remarkable success in synthesizing realistic data across a wide range of domains. State-of-the-art diffusion and flow matching methods are now capable of producing multi-modal outputs that are virtually indistinguishable from human-generated content, including images (Stability AI, 2023), audio (Borsos et al., 2023), video (Villegas et al., 2022; Brooks et al., 2024), and text (Gong et al., 2023; Xu et al., 2025).

## [other] Contributions.
- We challenge the prevailing belief that generalization in flow matching stems from an inherently noisy loss (Section 3.1). This assumption, largely supported by studies in low-dimensional settings, fails to hold in realistic high-dimensional data regimes.

## [other] <span id="page-1-0"></span>2 Recalls on conditional flow matching
Let $p_0 = \mathcal{N}(0, \mathrm{Id})$ be the source distribution<sup>2</sup> and $p_{\mathrm{data}}$ the data distribution. We are given n data points $x^{(1)}, \ldots, x^{(n)} \sim p_{\mathrm{data}}, x^{(i)} \in \mathbb{R}^d$ .

## [other] <span id="page-2-0"></span>3 Investigating the key sources of generalization
In this section, we investigate the key sources of flow matching generalization using the closed-form formula of its velocity field. First in Section 3.1 we challenge the claim that generalization stems from the stochastic approximation $u^{\text{cond}}$ of the optimal velocity field $\hat{u}^*$ .

## [other] <span id="page-3-0"></span>3.1 Target stochasticity is not what you need
One recent hypothesis is that generalization arises from the fact that the regression target $u^{\mathrm{cond}}$ of conditional flow matching is only a stochastic estimate of $\hat{u}^*$ . The fact that the target regression objective only equals the true objective on average is referred to by Vastola (2025) as "generalization through variance".

## [other] <span id="page-4-0"></span>3.2 Failure to learn the optimal velocity field
This subsection investigates how well the learned velocity field $u_{\theta}$ approximates the optimal/ideal velocity field $\hat{u}^{\star}$ , and how the quality of this approximation correlates with generalization. To do so, we propose the following experiment.

## [other] <span id="page-5-0"></span>3.3 When does generalization arise?
To investigate whether the failure to approximate $\hat{u}^*$ matters the most at small or large values of t, we carry out the following experiment.

## [other] <span id="page-6-0"></span>4 Learning with the closed-form formula
In this section, in order to discard the impact of stochastic target on the generalization, we propose to directly regress against the closed-form formula in Equation (6).

## [other] 4.1 Empirical flow matching
Regressing against the closed-form $\hat{u}^{\star}$ , defined in Equation (6), at a point $(x_t, t)$ requires computing a weighted sum of the conditional velocity fields over *all* the n training points $x^{(i)}$ . For a dataset of n samples of size d, and a batch of size $|\mathcal{B}|$ , computing the weights of the exact closed-form formula $\hat{u}^{\star}(x,t)$ of flow matching requires $\mathcal{O}(n \times |\mathcal{B}| \times d)$ .

## [method] <span id="page-7-2"></span>Algorithm 1 Vanilla Flow Matching


## [other] for k in $1, \ldots, n_{\text{iter}}$ do $\begin{vmatrix} t \sim \mathcal{U}([0,1]) \\ x_0 \sim \mathcal{N}(0, \text{Id}), x_1 \sim \hat{p}_{\text{data}}, \\ x_t = (1-t)x_0 + tx_1 \\ u^{\text{cond}}(x_t, t) = \frac{x_1 - x_t}{1-t} = x_1 - x_0 \\ \mathcal{L}(\theta) = \left\| u_{\theta}(x_t, t) - u^{\text{cond}}(x_t, t) \right\|^2 \\ \text{Compute } \nabla \mathcal{L}(\theta) \text{ and update } \theta$ $\mathbf{return} \ u_{\theta}$


## [method] <span id="page-7-1"></span>Algorithm 2 Empirical Flow Matching
$\begin{aligned} & \operatorname{param}: M \text{ } /\!\!/ \operatorname{Number} \text{ of samples in the empirical mean} \\ & \operatorname{for} k \text{ } in 1, \ldots, n_{\mathrm{iter}} \operatorname{do} \\ & x_0 \sim \mathcal{N}(0, \mathrm{Id}), x_1 \sim \hat{p}_{\mathrm{data}}, t \sim \mathcal{U}([0, 1]) \\ & x_t = (1 - t)x_0 + tx_1 \\ & b^{(1)} = x_1 \\ & \forall j \in [\![2, M]\!], b^{(j)} \sim \hat{p}_{\mathrm{data}} \text{ } /\!\!/ \operatorname{Samples} \operatorname{from} \hat{p}_{\mathrm{data}} \\ & \hat{u}_M^\star(x_t, t) = \sum_{j=1}^M \frac{b^{(j)} - x_t}{1 - t} \cdot \left[\operatorname{softmax} \left(-\frac{\|x_t - t \cdot b\|^2}{2(1 - t)^2}\right)\right]_j \\ & \mathcal{L}(\theta) = \|u_\theta(x_t, t) - \hat{u}_M^\star(x_t, t)\|^2 \\ & \operatorname{Compute} \nabla \mathcal{L}(\theta) \text{ and update } \theta \end{aligned}$

## [experiment] <span id="page-7-0"></span>4.2 Experiments
We now learn with empirical flow matching (EFM, Equation (7) and Algorithm 2) in practical high-dimensional settings. Our goal with this empirical investigation is first to observe if regressing against a more deterministic target leads to performance improvement/degradation.

## [other] <span id="page-8-0"></span>5 Related work
The existing literature related to our study can be roughly divided into three approaches: leveraging the closed-form, studies on the memorization vs generalization, and characterization of the different phases of the generating dynamics.

## [conclusion] 6 Conclusion, limitations and broader impact
Conclusion. By challenging the assumption that stochasticity in the loss function is a key driver of generalization, our findings help clarify the role of approximation of the exact velocity field in flow matching models.

## [other] 7 Acknowledgments
The authors thank the Blaise Pascal Center for its computational support, using the SIDUS [\(Quemener](#page-11-18) [and Corvellec,](#page-11-18) [2013\)](#page-11-18) solution.

## [other] References
- <span id="page-10-0"></span>M. S.

## [other] NeurIPS Paper Checklist
The checklist is designed to encourage best practices for responsible machine learning research, addressing issues of reproducibility, transparency, research ethics, and societal impact. Do not remove the checklist: The papers not including the checklist will be desk rejected.

## [other] IMPORTANT, please:
- Delete this instruction block, but keep the section heading "NeurIPS Paper Checklist", - Keep the checklist subsection headings, questions/answers and guidelines below. - Do not modify the questions and only use the provided macros for your answers.

## [other] 1. Claims
Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

## [other] Guidelines:
- The answer NA means that the abstract and introduction do not include the claims made in the paper. - The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations.

## [other] 2. Limitations
Question: Does the paper discuss the limitations of the work performed by the authors?

## [other] 3. Theory assumptions and proofs
Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

## [other] Guidelines:
- The answer NA means that the paper does not include theoretical results. - All the theorems, formulas, and proofs in the paper should be numbered and cross-referenced.

## [experiment] 4. Experimental result reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

## [other] 5. Open access to data and code
Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

## [other] Guidelines:
- The answer NA means that paper does not include experiments requiring code. - Please see the NeurIPS code and data submission guidelines ([https://nips.cc/public/](https://nips.cc/public/guides/CodeSubmissionPolicy) [guides/CodeSubmissionPolicy](https://nips.cc/public/guides/CodeSubmissionPolicy)) for more details.

## [experiment] 6. Experimental setting/details
Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

## [other] Guidelines:
• The answer NA means that the paper does not include experiments.

## [experiment] 7. Experiment statistical significance
Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

## [other] Guidelines:
- The answer NA means that the paper does not include experiments. - The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.

## [experiment] 8. Experiments compute resources
Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

## [other] Guidelines:
- The answer NA means that the paper does not include experiments. - The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

## [other] 9. Code of ethics
Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics <https://neurips.cc/public/EthicsGuidelines>?

## [other] 10. Broader impacts
Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

## [other] Guidelines:
- The answer NA means that there is no societal impact of the work performed. - If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.

## [other] 11. Safeguards
Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

## [other] Guidelines:
- The answer NA means that the paper poses no such risks. - Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.

## [other] 12. Licenses for existing assets
Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

## [other] 13. New assets
Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

## [other] 14. Crowdsourcing and research with human subjects
Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

## [other] 15. Institutional review board (IRB) approvals or equivalent for research with human subjects
Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

## [other] 16. Declaration of LLM usage
Question: Does the paper describe the usage of LLMs if it is an important, original, or nonstandard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

## [other] A Proofs of Section 2
$$\hat{u}^{\star}(x,t) = \sum_{i=1}^{n} u^{\text{cond}}(x, z = x^{(i)}, t) \cdot \frac{p(x|z = x^{(i)}, t)}{\sum_{i'=1}^{n} p(x|z = x^{(i')}, t)} . \tag{12}$$

## [other] <span id="page-20-0"></span>A.1 Proof of Proposition 1
*Proof.* • In the case where $z \sim \hat{p}_{\text{data}}$ , conditional probability writes

## [other] <span id="page-21-0"></span>B Additional details and comments on empirical flow matching
First, recalls on the optimal velocity (Equation (6)) and the empirical flow matching loss (Equations (7) and (8)) are provided in Section B.1. The unbiasedness of the estimator is presented in Section B.2, and its proof is in Section B.3.

## [other] <span id="page-21-3"></span>B.1 Recalls
The closed-form formula of the "optimal" velocity field is:

## [other] <span id="page-21-1"></span>B.2 Theoretical properties of the proposed estimator
First, we discuss below the relation between Proposition 2 and the sampling literature.

## [other] <span id="page-22-0"></span>B.3 Proof of Proposition [2](#page-6-4)
We first recall Section [B,](#page-21-0) which we prove in this section.

## [experiment] C Additional experiments
We present below the results for the MNIST dataset. The conclusions atre the same as for the CIFAR-10 and CelebA 64 × 64: regressing against a more deterministic velocity field does not hurt generalization.

## [experiment] <span id="page-26-0"></span>D Experiments details
For all the experiment we used all the same learning hyperparameters, the default ones form Tong et al. (2024).

## [other] D.1 Compute time
Given that regressing against an estimate of the closed-form, EFM, seems to improve on CFM, one may wonder what is the additional cost induced by EFN. To alleviate the non-linearity of GPU computing (parallelism may cause some discontinuities in terms of costs), we ran an exhaustive set of timing experiments, varying the batch size and the EFM sample size.

## [other] <span id="page-26-2"></span>D.2 Figures 1a and 1c
For Figure 1a no deep learning is involved: the datasets 2-moons and CIFAR-10 are loaded. Then, 256 points from $p_0 \times \hat{p}_{\text{data}}$ are drawn, and one computes the mean of the cosine similarities between $\hat{u}^*((1-t)x_0+tx_1,t)$ and $u^{\text{cond}}((1-t)x_0+tx_1,z=x_1,t)=x_1-x_0$ , for each value of $t \in \{0,1/100,2/100,\ldots,99/100\}$ .

## [other] D.3 Figure 2
In Figure 2, networks are trained with a vanilla conditional flow matching, with the standard 34 million parameters U-Net for diffusion by Nichol and Dhariwal (2021), with default settings from the torchfm codebase <sup>4</sup> (Tong et al., 2024). Training uses the CFM loss.

## [other] D.4 Figure [3](#page-5-1)
In Figure [3,](#page-5-1) for each dataset (CIFAR-10 and CelebA 64 × 64), one network is trained using a vanilla conditional flow matching with the default parameters of [Tong et al.](#page-12-14) [\(2024\)](#page-12-14) (the most important ones are recalled in Table [3\)](#page-26-1). Then images are generated first following the closed-form formula of the optimal velocity field uˆ ⋆ from 0 to τ .

## [other] <span id="page-27-0"></span>D.5 Figure [4](#page-8-1)
For experiments involving training on CIFAR-10 (Figures [2](#page-4-1) and [3\)](#page-5-1), we rely on the standard 34 million parameters U-Net for diffusion by [Nichol and Dhariwal](#page-11-19) [\(2021\)](#page-11-19), with default settings from the torchfm codebase [\(Tong et al.,](#page-12-14) [2024\)](#page-12-14). For each algorithm, the networks are trained for 500k iterations with batch size 128, *i.e.,* 1280 epochs.
