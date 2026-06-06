---
type: paper
slug: scaling-collapse-reveals-universal-dynamics-in-compute-optimally-trained-neural-networks
title: Scaling Collapse Reveals Universal Dynamics in Compute-Optimally Trained Neural Networks
authors: Shikai Qiu, Lechao Xiao, Andrew Gordon Wilson, Jeffrey Pennington, Atish Agarwala
source_path: .paper-scholar/scaling-collapse-reveals-universal-dynamics-in-compute-optimally-trained-neural-networks
ingested_at: '2026-06-05 05:50:07'
authors_list: []
sections:
- id: 585
  heading: Abstract
  role: abstract
  order_index: 0
  summary: What scaling limits govern neural network training dynamics when model size and training time grow in tandem? We show that despite the complex interactions between architecture, training algorithms, and data, compute-optimally trained models exhibit a remarkably precise universality.
- id: 586
  heading: 1. Introduction
  role: introduction
  order_index: 1
  summary: As machine learning systems grow in scale, accurate predictive models of their training dynamics become increasingly valuable, both for interpreting costly experiments and for designing robust, efficient training pipelines [\(Wortsman et al.,](#page-11-0)
- id: 587
  heading: 2. Empirical Observations
  role: experiment
  order_index: 2
  summary: We demonstrate our main empirical findings in this section, independently on multiple tasks and architectures which can be studied even in academic settings.
- id: 588
  heading: 2.1. Experiment Setup
  role: experiment
  order_index: 3
  summary: In each task, we train a sequence of models with increasing compute, scaling hyperparameters such as data, initialization, and learning rate with the model. We refer to a sequence of training configurations as a scaling ladder.
- id: 589
  heading: 2.2. Estimating Compute-Optimal Scaling Laws
  role: other
  order_index: 4
  summary: Let $L(t, p, \omega)$ be the test loss after t tokens (proportional to steps) for a model with p parameters trained with random seed $\omega$ . We estimate the compute-optimal training horizon in tokens for a p-parameter model as $t^*(p) = (p/p_0)^{\gamma}$ , where $\gamma$ is the data exponent, by extracting the Pareto frontier of expected loss (estimated using 5 seeds) vs.
- id: 590
  heading: 2.3. Scaling Collapse of Compute-Optimal Loss Curves
  role: other
  order_index: 5
  summary: The loss curves for different model sizes cover varying ranges of compute and loss values, but appear to follow a consistent shape, which motivates us to affinely rescale them to the *normalized loss curve* $\ell$ given by
- id: 591
  heading: <span id="page-2-3"></span>2.4. Quantifying the Collapse Quality
  role: other
  order_index: 6
  summary: 'We quantify the quality of collapse using the *collapse deviation* $\Delta$ , defined as:'
- id: 592
  heading: '2.5. Supercollapse: Consistency Below the Noise Floor'
  role: other
  order_index: 7
  summary: Remarkably, with learning rate decay, we find that the collapse deviation is less than the noise floor for a significant fraction of training; that is, $\Delta(x) < \sigma(x,p)$ for $x>1-\delta$ for some moderate $\delta$ as large as 0.5 (Figure 1c). We refer to this stronger form of collapse as *supercollapse* (in contrast to the collapse in Figure 3).
- id: 593
  heading: 2.6. Suboptimal Scaling Breaks Supercollapse
  role: other
  order_index: 8
  summary: Supercollapse provides a practical method for comparing inherently noisy training loss curves across model scales with precision that exceeds naive noise floor estimates, without the need for expensive multi-seed experiments typically required to obtain equally clean signals. This comparison can provide valuable diagnostic information about scaling where the ability to distinguish small signal from noise is often crucial (Xiao, 2024), which we now demonstrate.
- id: 594
  heading: 3. Explaining Loss Curve Scaling Collapse
  role: other
  order_index: 9
  summary: 'In this section, we investigate theoretical explanations for the scaling collapse of compute-optimal loss curves and supercollapse. Our analysis starts with a simple observation: the numerator of the collapse deviation $\Delta(x)$ can be decomposed as:'
- id: 595
  heading: <span id="page-4-0"></span>3.1. Scaling Collapse from Power-Law Scaling
  role: other
  order_index: 10
  summary: In this section, we consider deterministic models of the loss curves and assume all randomness has been averaged out.
- id: 596
  heading: <span id="page-5-0"></span>3.2. Universality of Learning Rate Schedules
  role: other
  order_index: 11
  summary: To understand why scaling collapse is robust across learning rate schedules, we develop a quantitative model for how learning rate schedules affect the loss curves. While an exact theoretical model seems out of reach for the realistic training setup, we show that a simple model based on quadratic loss analysis proves surprisingly effective.
- id: 597
  heading: <span id="page-5-1"></span>3.2.1. A SIMPLE MODEL FOR LR SCHEDULES
  role: other
  order_index: 12
  summary: Let w(t) and L(w(t)) denote the parameters and loss at step t, we can model the dynamics of full-batch gradient descent under a small learning rate $\eta(t)$ with a gradient flow $\frac{dw}{dt} = -\eta(t)\nabla L(w(t))$ . To model stochastic effects, a noise term is added to the gradient, leading to the SDE $\frac{dw}{dt} = -\eta(t)\left(\nabla L(w) + \Sigma^{1/2}(w)\xi(t)\right)$ (Li et al., 2017; Malladi et al., 2022), where the mini-batch gradient noise $\Sigma^{1/2}(w)\xi(t)$ satisfies $\mathbb{E}[\xi(t)\xi(t')] = \delta(t-t')I$ , and we allow its covariance (which depends on batch size) $\Sigma(w)$ to be a function of the parameters.
- id: 598
  heading: <span id="page-6-3"></span>3.2.2. PREDICTING LOSS CURVES ACROSS SCHEDULES
  role: other
  order_index: 13
  summary: We apply this simple model to predict empirical loss curves in the CIFAR-5M experiments. We measure the trace of the preconditioned gradient covariance on a fixed set of 2M tokens (see Appendix [A](#page-12-0) for experiment details).
- id: 599
  heading: 3.2.3. Universal Scaling of Gradient Noise
  role: other
  order_index: 14
  summary: For typical loss functions, the gradient covariance can be related to the loss itself. For example, in noiseless high-dimensional linear regression with Gaussian features drawn from $\mathcal{N}(0,K)$ , we have $\mathrm{Tr}(\Sigma)\approx 2\mathcal{L}\,\mathrm{Tr}(K)$ (Paquette et al., 2021), an intuitive result since the gradient scales with both the prediction error and the input.
- id: 600
  heading: <span id="page-7-0"></span>3.3. Supercollapse as Variance Reduction
  role: other
  order_index: 15
  summary: 'Lastly, we turn to understanding the "super" in supercollapse: why does learning rate decay significantly improve the collapse, to the extent that the collapse deviation $\Delta(x)$ drops below the per-model noise floor $\sigma(x,p)$ for a substantial fraction of training? Again, the simple quadratic model provides quantitative insights into this phenomenon.'
- id: 601
  heading: 4. Discussion
  role: conclusion
  order_index: 16
  summary: Scale has enabled remarkable progress in machine learning, but a thorough scientific understanding of scaling remains elusive. Key open questions include identifying robust principles that guide general hyperparameter transfer and characterizing scaling limits under realistic scaling ladders.
- id: 602
  heading: Acknowledgements
  role: other
  order_index: 17
  summary: 'We thank Courtney Paquette and Zixi Chen for helpful comments on an earlier version of this paper. SQ was supported by Google''s TPU Research Cloud (TRC) program: <https://sites.research.google/trc/>.'
- id: 603
  heading: Contribution Statement
  role: other
  order_index: 18
  summary: SQ designed and conducted the majority of experiments, led the theory development, and wrote the paper. LX initially observed supercollapse, contributed to theory, experimental design, and writing the paper.
- id: 604
  heading: Impact Statement
  role: other
  order_index: 19
  summary: This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.
- id: 605
  heading: References
  role: other
  order_index: 20
  summary: '- <span id="page-9-0"></span>Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F. L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., et al.'
- id: 606
  heading: <span id="page-12-0"></span>A. Experiment Details
  role: experiment
  order_index: 21
  summary: '**Transformer Architecture.** We use GeLU activations (Hendrycks & Gimpel, 2016), RMSNorm (Zhang & Sennrich, 2019), and learned positional embeddings. We untie the embedding matrix from the output head and do not use bias anywhere.'
- id: 607
  heading: <span id="page-13-0"></span>B. Scaling Collapse Across Transformer Depths
  role: other
  order_index: 22
  summary: 'Figure 9: Depthwise scaling collapse for transformers trained on chess.'
- id: 608
  heading: <span id="page-13-1"></span>C. Estimating Compute-Optimal Training Horizon
  role: other
  order_index: 23
  summary: 'To estimate the optimal compute for training each model, we perform the following steps in each experiment:'
- id: 609
  heading: <span id="page-14-0"></span>D. Universality and Scaling Collapse in Other Sciences
  role: other
  order_index: 24
  summary: The simplest versions of collapse come from statistics and probability, where entire *distributions* of random variables show universal behavior between systems of different types and scales. The most well known is the central limit theorem which predicts a universal Gaussian form for the sums of random variables with appropriately bounded moments (and Levy distributions for heavy tailed distributions).
- id: 610
  heading: <span id="page-14-1"></span>E. Power-Law Pareto Frontier is Necessary for Collapse
  role: other
  order_index: 25
  summary: Recall $t^{\star}(p)$ is the optimal training horizon for model size p, i.e. $L(t^{\star}(p),p) = \min_{t',p':t'p'=t^{\star}(p)p} L(t',p')$ .
- id: 611
  heading: <span id="page-16-0"></span>F. Collapse for General Sum-of-Power-Laws Loss Curves
  role: other
  order_index: 26
  summary: '**Theorem F.1.** Suppose the loss curve is given by'
- id: 612
  heading: Remarks.
  role: other
  order_index: 27
  summary: '- 1. Compute-optimal data exponent implies asymptotic collapse, but the converse is not necessarily true when $m \geq 3$ , since there can be multiple choices of $\gamma$ that lead to balanced dominant power laws, which imply collapse, but only one of them can be compute-optimal.'
- id: 613
  heading: <span id="page-18-1"></span>G. A Perturbative Model of Learning Rate Schedules
  role: other
  order_index: 28
  summary: Let w' denote the parameter trajectory under the influence of gradient noise. The dynamics of stochastic gradient descent in gradient flow time are given by
- id: 614
  heading: <span id="page-21-0"></span>H. Computing $\tilde{\Delta}$ to Leading Order
  role: other
  order_index: 29
  summary: Let $\psi(\tau) = \frac{\mathcal{L}(\tau) - \bar{\mathcal{L}}(\tau)}{\bar{\mathcal{L}}(\tau)}$ and define
- id: 615
  heading: <span id="page-22-0"></span>I. Additional Results on Learning Rate Schedules
  role: experiment
  order_index: 30
  summary: '**MLP fits.** Figure 12 shows our predictions for MLP loss curves. With a single $\alpha = 0.26$ (very close to 1/4), we obtain excellent fits across schedules, model sizes, and training horizons.'
---

# Scaling Collapse Reveals Universal Dynamics in Compute-Optimally Trained Neural Networks

## [abstract] Abstract
What scaling limits govern neural network training dynamics when model size and training time grow in tandem? We show that despite the complex interactions between architecture, training algorithms, and data, compute-optimally trained models exhibit a remarkably precise universality.

## [introduction] 1. Introduction
As machine learning systems grow in scale, accurate predictive models of their training dynamics become increasingly valuable, both for interpreting costly experiments and for designing robust, efficient training pipelines [\(Wortsman et al.,](#page-11-0)

## [experiment] 2. Empirical Observations
We demonstrate our main empirical findings in this section, independently on multiple tasks and architectures which can be studied even in academic settings.

## [experiment] 2.1. Experiment Setup
In each task, we train a sequence of models with increasing compute, scaling hyperparameters such as data, initialization, and learning rate with the model. We refer to a sequence of training configurations as a scaling ladder.

## [other] 2.2. Estimating Compute-Optimal Scaling Laws
Let $L(t, p, \omega)$ be the test loss after t tokens (proportional to steps) for a model with p parameters trained with random seed $\omega$ . We estimate the compute-optimal training horizon in tokens for a p-parameter model as $t^*(p) = (p/p_0)^{\gamma}$ , where $\gamma$ is the data exponent, by extracting the Pareto frontier of expected loss (estimated using 5 seeds) vs.

## [other] 2.3. Scaling Collapse of Compute-Optimal Loss Curves
The loss curves for different model sizes cover varying ranges of compute and loss values, but appear to follow a consistent shape, which motivates us to affinely rescale them to the *normalized loss curve* $\ell$ given by

## [other] <span id="page-2-3"></span>2.4. Quantifying the Collapse Quality
We quantify the quality of collapse using the *collapse deviation* $\Delta$ , defined as:

## [other] 2.5. Supercollapse: Consistency Below the Noise Floor
Remarkably, with learning rate decay, we find that the collapse deviation is less than the noise floor for a significant fraction of training; that is, $\Delta(x) < \sigma(x,p)$ for $x>1-\delta$ for some moderate $\delta$ as large as 0.5 (Figure 1c). We refer to this stronger form of collapse as *supercollapse* (in contrast to the collapse in Figure 3).

## [other] 2.6. Suboptimal Scaling Breaks Supercollapse
Supercollapse provides a practical method for comparing inherently noisy training loss curves across model scales with precision that exceeds naive noise floor estimates, without the need for expensive multi-seed experiments typically required to obtain equally clean signals. This comparison can provide valuable diagnostic information about scaling where the ability to distinguish small signal from noise is often crucial (Xiao, 2024), which we now demonstrate.

## [other] 3. Explaining Loss Curve Scaling Collapse
In this section, we investigate theoretical explanations for the scaling collapse of compute-optimal loss curves and supercollapse. Our analysis starts with a simple observation: the numerator of the collapse deviation $\Delta(x)$ can be decomposed as:

## [other] <span id="page-4-0"></span>3.1. Scaling Collapse from Power-Law Scaling
In this section, we consider deterministic models of the loss curves and assume all randomness has been averaged out.

## [other] <span id="page-5-0"></span>3.2. Universality of Learning Rate Schedules
To understand why scaling collapse is robust across learning rate schedules, we develop a quantitative model for how learning rate schedules affect the loss curves. While an exact theoretical model seems out of reach for the realistic training setup, we show that a simple model based on quadratic loss analysis proves surprisingly effective.

## [other] <span id="page-5-1"></span>3.2.1. A SIMPLE MODEL FOR LR SCHEDULES
Let w(t) and L(w(t)) denote the parameters and loss at step t, we can model the dynamics of full-batch gradient descent under a small learning rate $\eta(t)$ with a gradient flow $\frac{dw}{dt} = -\eta(t)\nabla L(w(t))$ . To model stochastic effects, a noise term is added to the gradient, leading to the SDE $\frac{dw}{dt} = -\eta(t)\left(\nabla L(w) + \Sigma^{1/2}(w)\xi(t)\right)$ (Li et al., 2017; Malladi et al., 2022), where the mini-batch gradient noise $\Sigma^{1/2}(w)\xi(t)$ satisfies $\mathbb{E}[\xi(t)\xi(t')] = \delta(t-t')I$ , and we allow its covariance (which depends on batch size) $\Sigma(w)$ to be a function of the parameters.

## [other] <span id="page-6-3"></span>3.2.2. PREDICTING LOSS CURVES ACROSS SCHEDULES
We apply this simple model to predict empirical loss curves in the CIFAR-5M experiments. We measure the trace of the preconditioned gradient covariance on a fixed set of 2M tokens (see Appendix [A](#page-12-0) for experiment details).

## [other] 3.2.3. Universal Scaling of Gradient Noise
For typical loss functions, the gradient covariance can be related to the loss itself. For example, in noiseless high-dimensional linear regression with Gaussian features drawn from $\mathcal{N}(0,K)$ , we have $\mathrm{Tr}(\Sigma)\approx 2\mathcal{L}\,\mathrm{Tr}(K)$ (Paquette et al., 2021), an intuitive result since the gradient scales with both the prediction error and the input.

## [other] <span id="page-7-0"></span>3.3. Supercollapse as Variance Reduction
Lastly, we turn to understanding the "super" in supercollapse: why does learning rate decay significantly improve the collapse, to the extent that the collapse deviation $\Delta(x)$ drops below the per-model noise floor $\sigma(x,p)$ for a substantial fraction of training? Again, the simple quadratic model provides quantitative insights into this phenomenon.

## [conclusion] 4. Discussion
Scale has enabled remarkable progress in machine learning, but a thorough scientific understanding of scaling remains elusive. Key open questions include identifying robust principles that guide general hyperparameter transfer and characterizing scaling limits under realistic scaling ladders.

## [other] Acknowledgements
We thank Courtney Paquette and Zixi Chen for helpful comments on an earlier version of this paper. SQ was supported by Google's TPU Research Cloud (TRC) program: <https://sites.research.google/trc/>.

## [other] Contribution Statement
SQ designed and conducted the majority of experiments, led the theory development, and wrote the paper. LX initially observed supercollapse, contributed to theory, experimental design, and writing the paper.

## [other] Impact Statement
This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

## [other] References
- <span id="page-9-0"></span>Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F. L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., et al.

## [experiment] <span id="page-12-0"></span>A. Experiment Details
**Transformer Architecture.** We use GeLU activations (Hendrycks & Gimpel, 2016), RMSNorm (Zhang & Sennrich, 2019), and learned positional embeddings. We untie the embedding matrix from the output head and do not use bias anywhere.

## [other] <span id="page-13-0"></span>B. Scaling Collapse Across Transformer Depths
Figure 9: Depthwise scaling collapse for transformers trained on chess.

## [other] <span id="page-13-1"></span>C. Estimating Compute-Optimal Training Horizon
To estimate the optimal compute for training each model, we perform the following steps in each experiment:

## [other] <span id="page-14-0"></span>D. Universality and Scaling Collapse in Other Sciences
The simplest versions of collapse come from statistics and probability, where entire *distributions* of random variables show universal behavior between systems of different types and scales. The most well known is the central limit theorem which predicts a universal Gaussian form for the sums of random variables with appropriately bounded moments (and Levy distributions for heavy tailed distributions).

## [other] <span id="page-14-1"></span>E. Power-Law Pareto Frontier is Necessary for Collapse
Recall $t^{\star}(p)$ is the optimal training horizon for model size p, i.e. $L(t^{\star}(p),p) = \min_{t',p':t'p'=t^{\star}(p)p} L(t',p')$ .

## [other] <span id="page-16-0"></span>F. Collapse for General Sum-of-Power-Laws Loss Curves
**Theorem F.1.** Suppose the loss curve is given by

## [other] Remarks.
- 1. Compute-optimal data exponent implies asymptotic collapse, but the converse is not necessarily true when $m \geq 3$ , since there can be multiple choices of $\gamma$ that lead to balanced dominant power laws, which imply collapse, but only one of them can be compute-optimal.

## [other] <span id="page-18-1"></span>G. A Perturbative Model of Learning Rate Schedules
Let w' denote the parameter trajectory under the influence of gradient noise. The dynamics of stochastic gradient descent in gradient flow time are given by

## [other] <span id="page-21-0"></span>H. Computing $\tilde{\Delta}$ to Leading Order
Let $\psi(\tau) = \frac{\mathcal{L}(\tau) - \bar{\mathcal{L}}(\tau)}{\bar{\mathcal{L}}(\tau)}$ and define

## [experiment] <span id="page-22-0"></span>I. Additional Results on Learning Rate Schedules
**MLP fits.** Figure 12 shows our predictions for MLP loss curves. With a single $\alpha = 0.26$ (very close to 1/4), we obtain excellent fits across schedules, model sizes, and training horizons.
