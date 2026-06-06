---
type: paper
slug: why-diffusion-models-don-t-memorize-the-role-of-implicit-dynamical-regularization-in-training
title: 'Why Diffusion Models Don''t Memorize: The Role of Implicit Dynamical Regularization in Training'
authors: Tony Bonnaire, Giulio Biroli, Raphael Urfin, Marc Mezard
source_path: /Users/saris.kia.adm/.paper-scholar/why-diffusion-models-don-t-memorize-the-role-of-implicit-dynamical-regularization-in-training/2505.17638.md
ingested_at: '2026-06-05 05:50:56'
authors_list: []
sections:
- id: 699
  heading: Tony Bonnaire† LPENS
  role: section
  order_index: 0
  summary: Université PSL, Paris tony.bonnaire@phys.ens.fr
- id: 700
  heading: Giulio Biroli
  role: section
  order_index: 1
  summary: LPENS Université PSL, Paris giulio.biroli@phys.ens.fr
- id: 701
  heading: Raphaël Urfin† LPENS
  role: section
  order_index: 2
  summary: Université PSL, Paris raphael.urfin@phys.ens.fr
- id: 702
  heading: Marc Mézard
  role: section
  order_index: 3
  summary: Department of Computing Sciences Bocconi University, Milano marc.mezard@unibocconi.it
- id: 703
  heading: Abstract
  role: section
  order_index: 4
  summary: Diffusion models have achieved remarkable success across a wide range of generative tasks. A key challenge is understanding the mechanisms that prevent their memorization of training data and allow generalization.
- id: 704
  heading: 1 Introduction
  role: section
  order_index: 5
  summary: 'Diffusion Models [DMs, [49,](#page-12-0) [20,](#page-11-0) [54,](#page-13-0) [55\]](#page-13-1) achieve state-of-the-art performance in a wide variety of AI tasks such as the generation of images [\[45\]](#page-12-1), audios [\[64\]](#page-13-2), videos [\[33\]](#page-11-1), and scientific data [\[31,](#page-11-2) [40\]](#page-12-2). This class of generative models, inspired by out-of-equilibrium thermodynamics [\[49\]](#page-12-0), corresponds to a two-stage process: the first one, called *forward*, gradually adds noise to a data, whereas the second one, called *backward*, generates new data by denoising Gaussian white noise samples.'
- id: 705
  heading: Related works.
  role: section
  order_index: 6
  summary: '- The memorization transition in DMs has been the subject of several recent empirical investigations [\[9,](#page-10-2) [50,](#page-12-6) [51\]](#page-12-7) which have demonstrated that state-of-the-art image DMs – including Stable Diffusion and DALL·E – can reproduce a non-negligible portion of their training data, indicating a form of memorization. Several additional works [\[18,](#page-11-7) [63\]](#page-13-4) examined how this phenomenon is influenced by factors such as data distribution, model configuration, and training procedure, and provide a strong basis for the numerical part of our work.'
- id: 706
  heading: 4 Conclusions
  role: section
  order_index: 9
  summary: 'We have shown that the training dynamics of neural network-based score functions display a form of implicit regularization that prevents memorization even in highly overparameterized diffusion models. Specifically, we have identified two well-separated timescales in the learning: $\tau_{\rm gen}$ , at which models begins to generate high-quality, novel samples, and $\tau_{\rm mem}$ , beyond which they start to memorize the training data.'
- id: 707
  heading: Limitations and future works.
  role: section
  order_index: 10
  summary: '- While we derived our results under SGD optimization, most DMs are trained in practice with Adam [28]. In SM Sects.'
- id: 708
  heading: Acknowledgments and Disclosure of Funding
  role: section
  order_index: 11
  summary: The authors thank Valentin De Bortoli for initial motivating discussions on memorization– generalization transitions. RU thanks Beatrice Achilli, Jérome Garnier-Brun, Carlo Lucibello and Enrico Ventura for insightful discussions.
- id: 709
  heading: References
  role: section
  order_index: 12
  summary: '- <span id="page-10-3"></span>[1] Achilli, B., Ventura, E., Silvestri, G., Pham, B., Raya, G., Krotov, D., Lucibello, C., and Ambrogioni, L. (2024).'
- id: 710
  heading: 'Why Diffusion Models Don''t Memorize: The Role of Implicit Dynamical Regularization in Training Supplementary Material (SM)'
  role: section
  order_index: 13
  summary: Tony Bonnaire<sup>†</sup>, Raphaël Urfin<sup>†</sup>, Giulio Biroli, Marc Mézard
- id: 711
  heading: A.1 Details on the numerical setup
  role: section
  order_index: 15
  summary: '**Dataset.** All numerical experiments in Sect. 2 of the MT use the CelebA face dataset [34].'
- id: 712
  heading: 'A.2 Batch-size effect: repetition vs. memorization'
  role: section
  order_index: 16
  summary: All the experiments in the MT use a fixed batch size B = 512, and in Sect [2](#page-3-3) we emphasize that the observed O(n) scaling of τmem cannot be explained by repetition over training samples. To validate this statement, the left panel of Fig.
- id: 713
  heading: A.3 What about Adam?
  role: section
  order_index: 17
  summary: We conclude this section by repeating our analysis at fixed W = 64 using the Adam optimizer [\[28\]](#page-11-11) instead of SGD with momentum. The learning rate is η = 1 × 10<sup>−</sup><sup>4</sup> , gradient averages take values (β1, β2) = (0.9, 0.999), and batch size B = min(512, n).
- id: 714
  heading: B.1 Settings
  role: section
  order_index: 19
  summary: Data distribution. We focus on data iid sampled from a d-dimensional Gaussian Mixture Model (GMM) made of two balanced Gaussians centered on ±µ with unit covariance, i.e.,
- id: 715
  heading: B.2 Scaling of $\tau_{\rm mem}$ and $\tau_{\rm gen}$ with n and W
  role: section
  order_index: 20
  summary: In Fig. 9, the left panel shows how the KL divergence and memorization fraction evolve with training time $\tau$ for different training set sizes n at fixed width W=128, while the right panel fixes n=2048 and varies W.
- id: 716
  heading: B.3 Discussion on conditional diffusion models
  role: section
  order_index: 21
  summary: Conditional generation aims to sample from distributions of the form $p(\mathbf{x}|\mathbf{y})$ , where $\mathbf{y}$ denotes a conditioning variable such as a class label, a text embedding, or any other contextual information. DMs can naturally be extended to this setting using for instance classifier-free guidance [21].
- id: 717
  heading: Closed form of the learning dynamics
  role: section
  order_index: 24
  summary: $$\frac{\mathbf{A}(\tau)}{\sqrt{p}} = -\frac{1}{\sqrt{\Delta_t}} \mathbf{V}^T \mathbf{U}^{-1} + (\frac{1}{\sqrt{\Delta_t}} \mathbf{V}^T \mathbf{U}^{-1} + \frac{\mathbf{A}_0}{\sqrt{p}}) e^{-\frac{2\Delta_t}{\psi_p} \mathbf{U}\tau}$$ (53)
- id: 718
  heading: C.5 Proof of Theorem 3.2
  role: section
  order_index: 27
  summary: We recall Theorem 3.2 of the MT.
- id: 719
  heading: C.6 Dynamics on the fast timescales
  role: section
  order_index: 28
  summary: In the following we denote for a matrix $\mathbf{A} \in \mathbb{R}^{p \times p}$ ,
- id: 720
  heading: Second term.
  role: section
  order_index: 29
  summary: $$\|(\frac{\mathbf{\Omega}\mathbf{\Omega}^T}{n} - \mathbf{I}_p)\|_{\text{op}}.$$ (195)
---

# Why Diffusion Models Don't Memorize: The Role of Implicit Dynamical Regularization in Training

## [section] Tony Bonnaire† LPENS
Université PSL, Paris tony.bonnaire@phys.ens.fr

## [section] Giulio Biroli
LPENS Université PSL, Paris giulio.biroli@phys.ens.fr

## [section] Raphaël Urfin† LPENS
Université PSL, Paris raphael.urfin@phys.ens.fr

## [section] Marc Mézard
Department of Computing Sciences Bocconi University, Milano marc.mezard@unibocconi.it

## [section] Abstract
Diffusion models have achieved remarkable success across a wide range of generative tasks. A key challenge is understanding the mechanisms that prevent their memorization of training data and allow generalization.

## [section] 1 Introduction
Diffusion Models [DMs, [49,](#page-12-0) [20,](#page-11-0) [54,](#page-13-0) [55\]](#page-13-1) achieve state-of-the-art performance in a wide variety of AI tasks such as the generation of images [\[45\]](#page-12-1), audios [\[64\]](#page-13-2), videos [\[33\]](#page-11-1), and scientific data [\[31,](#page-11-2) [40\]](#page-12-2). This class of generative models, inspired by out-of-equilibrium thermodynamics [\[49\]](#page-12-0), corresponds to a two-stage process: the first one, called *forward*, gradually adds noise to a data, whereas the second one, called *backward*, generates new data by denoising Gaussian white noise samples.

## [section] Related works.
- The memorization transition in DMs has been the subject of several recent empirical investigations [\[9,](#page-10-2) [50,](#page-12-6) [51\]](#page-12-7) which have demonstrated that state-of-the-art image DMs – including Stable Diffusion and DALL·E – can reproduce a non-negligible portion of their training data, indicating a form of memorization. Several additional works [\[18,](#page-11-7) [63\]](#page-13-4) examined how this phenomenon is influenced by factors such as data distribution, model configuration, and training procedure, and provide a strong basis for the numerical part of our work.

## [section] 4 Conclusions
We have shown that the training dynamics of neural network-based score functions display a form of implicit regularization that prevents memorization even in highly overparameterized diffusion models. Specifically, we have identified two well-separated timescales in the learning: $\tau_{\rm gen}$ , at which models begins to generate high-quality, novel samples, and $\tau_{\rm mem}$ , beyond which they start to memorize the training data.

## [section] Limitations and future works.
- While we derived our results under SGD optimization, most DMs are trained in practice with Adam [28]. In SM Sects.

## [section] Acknowledgments and Disclosure of Funding
The authors thank Valentin De Bortoli for initial motivating discussions on memorization– generalization transitions. RU thanks Beatrice Achilli, Jérome Garnier-Brun, Carlo Lucibello and Enrico Ventura for insightful discussions.

## [section] References
- <span id="page-10-3"></span>[1] Achilli, B., Ventura, E., Silvestri, G., Pham, B., Raya, G., Krotov, D., Lucibello, C., and Ambrogioni, L. (2024).

## [section] Why Diffusion Models Don't Memorize: The Role of Implicit Dynamical Regularization in Training Supplementary Material (SM)
Tony Bonnaire<sup>†</sup>, Raphaël Urfin<sup>†</sup>, Giulio Biroli, Marc Mézard

## [section] A.1 Details on the numerical setup
**Dataset.** All numerical experiments in Sect. 2 of the MT use the CelebA face dataset [34].

## [section] A.2 Batch-size effect: repetition vs. memorization
All the experiments in the MT use a fixed batch size B = 512, and in Sect [2](#page-3-3) we emphasize that the observed O(n) scaling of τmem cannot be explained by repetition over training samples. To validate this statement, the left panel of Fig.

## [section] A.3 What about Adam?
We conclude this section by repeating our analysis at fixed W = 64 using the Adam optimizer [\[28\]](#page-11-11) instead of SGD with momentum. The learning rate is η = 1 × 10<sup>−</sup><sup>4</sup> , gradient averages take values (β1, β2) = (0.9, 0.999), and batch size B = min(512, n).

## [section] B.1 Settings
Data distribution. We focus on data iid sampled from a d-dimensional Gaussian Mixture Model (GMM) made of two balanced Gaussians centered on ±µ with unit covariance, i.e.,

## [section] B.2 Scaling of $\tau_{\rm mem}$ and $\tau_{\rm gen}$ with n and W
In Fig. 9, the left panel shows how the KL divergence and memorization fraction evolve with training time $\tau$ for different training set sizes n at fixed width W=128, while the right panel fixes n=2048 and varies W.

## [section] B.3 Discussion on conditional diffusion models
Conditional generation aims to sample from distributions of the form $p(\mathbf{x}|\mathbf{y})$ , where $\mathbf{y}$ denotes a conditioning variable such as a class label, a text embedding, or any other contextual information. DMs can naturally be extended to this setting using for instance classifier-free guidance [21].

## [section] Closed form of the learning dynamics
$$\frac{\mathbf{A}(\tau)}{\sqrt{p}} = -\frac{1}{\sqrt{\Delta_t}} \mathbf{V}^T \mathbf{U}^{-1} + (\frac{1}{\sqrt{\Delta_t}} \mathbf{V}^T \mathbf{U}^{-1} + \frac{\mathbf{A}_0}{\sqrt{p}}) e^{-\frac{2\Delta_t}{\psi_p} \mathbf{U}\tau}$$ (53)

## [section] C.5 Proof of Theorem 3.2
We recall Theorem 3.2 of the MT.

## [section] C.6 Dynamics on the fast timescales
In the following we denote for a matrix $\mathbf{A} \in \mathbb{R}^{p \times p}$ ,

## [section] Second term.
$$\|(\frac{\mathbf{\Omega}\mathbf{\Omega}^T}{n} - \mathbf{I}_p)\|_{\text{op}}.$$ (195)
