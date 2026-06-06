---
type: paper-fulltext
slug: span-id-page-0-0-span-blink-of-an-eye-a-simple-theory-for-feature-localization-in-generative-models
arxiv_id: null
source_path: /Users/saris.kia.adm/.paper-scholar/span-id-page-0-0-span-blink-of-an-eye-a-simple-theory-for-feature-localization-in-generative-models/2502.00921.md
paper: "[[span-id-page-0-0-span-blink-of-an-eye-a-simple-theory-for-feature-localization-in-generative-models]]"
---
# <span id="page-0-0"></span>Blink of an eye: a simple theory for feature localization in generative models

Marvin Li\* Harvard College

Aayush Karan† Harvard SEAS

Sitan Chen‡ Harvard SEAS

June 10, 2025

## **Abstract**

Large language models can exhibit unexpected behavior in the blink of an eye. In a recent computer use demo, a language model switched from coding to Googling pictures of Yellowstone, and these sudden shifts in behavior have also been observed in reasoning patterns and jailbreaks. This phenomenon is not unique to autoregressive models: in diffusion models, key features of the final output are decided in narrow "critical windows" of the generation process. In this work we develop a simple, unifying theory to explain this phenomenon using the formalism of stochastic localization samplers. We show that it emerges generically as the generation process localizes to a sub-population of the distribution it models.

While critical windows have been studied at length in diffusion models, existing theory heavily relies on strong distributional assumptions and the particulars of Gaussian diffusion. In contrast to existing work our theory (1) applies to autoregressive and diffusion models; (2) makes no distributional assumptions; (3) quantitatively improves previous bounds even when specialized to diffusions; and (4) requires basic tools and no stochastic calculus or statistical-physics-based machinery. We also identify an intriguing connection to the all-or-nothing phenomenon from statistical inference. Finally, we validate our predictions empirically for LLMs and find that critical windows often coincide with failures in problem solving for various math and reasoning benchmarks.

<sup>\*</sup>Email: marvinli@college.harvard.edu

<sup>†</sup>Email: akaran1@g.harvard.edu, supported in part by the PD Soros fellowship

<sup>‡</sup>Email: sitan@seas.harvard.edu, supported in part by NSF Award 2331831

## **Contents**

| 1 | Introduction                                    | 3  |  |  |  |  |
|---|-------------------------------------------------|----|--|--|--|--|
|   | 1.1<br>Our contributions<br>                    | 4  |  |  |  |  |
|   | 1.2<br>Related work<br>                         | 5  |  |  |  |  |
| 2 | Technical preliminaries                         | 6  |  |  |  |  |
|   | 2.1<br>Stochastic localization samplers<br>     | 6  |  |  |  |  |
|   | 2.2<br>Forward-reverse experiment<br>           | 7  |  |  |  |  |
|   | 2.2.1<br>Instantiation for LLMs<br>             | 8  |  |  |  |  |
| 3 | Characterization of critical windows            | 8  |  |  |  |  |
|   | 3.1<br>Main result<br>                          | 9  |  |  |  |  |
|   | 3.2<br>Proof of Theorem<br>2<br>                | 11 |  |  |  |  |
| 4 | Examples of critical windows                    |    |  |  |  |  |
|   | 4.1<br>Diffusion<br>                            | 12 |  |  |  |  |
|   | 4.2<br>Autoregression<br>                       | 13 |  |  |  |  |
|   | 4.2.1<br>In-context learning<br>                | 14 |  |  |  |  |
|   | 4.2.2<br>All-or-nothing phenomenon<br>          | 14 |  |  |  |  |
| 5 | Hierarchies in stochastic localization samplers | 16 |  |  |  |  |
| 6 | Experiments                                     | 17 |  |  |  |  |
|   | 6.1<br>Structured output experiments<br>        | 17 |  |  |  |  |
|   | 6.2<br>Chain of thought experiments<br>         | 18 |  |  |  |  |
| 7 | Discussion                                      | 20 |  |  |  |  |
| A | Additional related work                         | 27 |  |  |  |  |
| B | Examples of stochastic localization samplers    | 27 |  |  |  |  |
| C | Proof details                                   | 28 |  |  |  |  |
| D | Experiments                                     | 37 |  |  |  |  |

## <span id="page-2-0"></span>**1 Introduction**

In large language models (LLMs), undesirable behavior can often emerge very suddenly. For example,

- Claude transitioned from coding to browsing pictures of Yellowstone while using a computer [[Ant24](#page-20-0)].
- OpenAI's frontier reasoning model chose to hack a reward evaluation after saying that the task seems difficult [[Ope25](#page-24-0)].
- The Phi-4 team reported that the probability of correctly answering a math problem can plummet with a single token [[AAB](#page-20-1)+24, [LLX](#page-23-0)+24].
- Gemini abruptly threatened a student who was using it to study [[Gem24](#page-22-0)].
- Llama models can be jailbroken by manipulating the first few generated tokens [[QPL](#page-24-1)+24, [Hai24b](#page-22-1)].

These abrupt shifts are not unique to autoregressive models. In diffusion models, it has been observed that certain properties like the presence of an object in the background or the image class emerge in narrow time intervals, sometimes called *critical windows*, of the generation process [[HJA20,](#page-22-2) [MHS](#page-23-1)+22, [CLS](#page-21-0)+22, [RA23,](#page-24-2) [GVS](#page-22-3)+23, [SFLW24,](#page-24-3) [SFW25,](#page-24-4) [BBdBM24,](#page-20-2) [LC24](#page-22-4)].

<span id="page-2-1"></span>![](_page_2_Figure_8.jpeg)

Figure 1: Examples of critical windows for different data modalities and samplers, including reasoning [[AAB](#page-20-1)+24, [QPL](#page-24-1)+24] and certain jailbreaks [[Hai24b](#page-22-1)] for language models and image class for diffusion models [[GVS](#page-22-3)+23].

Critical windows, broadly characterizable as *a few steps of the sampling procedure* during which features of the final output appear, arise in many contexts for different generative models and data modalities (Figure [1\)](#page-2-1). They are extremely useful from an interpretability perspective as they represent the steps of the sampler responsible for a given property of the output [[GVS](#page-22-3)+23, [QPL](#page-24-1)+24], and have also been used to provide richer stepwise rewards for preference optimization and finetuning [[AAB](#page-20-1)+24, [LLX](#page-23-0)+24, [QPL](#page-24-1)+24]. As the applications of generative models proliferate, it is crucial from interpretability, safety, and capability perspectives to understand how and why these critical windows emerge.

Recently, this phenomenon has received significant attention within the theoretical literature on diffusion models [[RA23,](#page-24-2) [SFLW24,](#page-24-3) [SFW25,](#page-24-4) [BBdBM24,](#page-20-2) [LC24](#page-22-4)]. While existing works do offer predictive theory in the diffusions setting, they either (A) make strong distributional assumptions or (B) rely heavily on the particulars of diffusion, which do not straightforwardly extend to autoregressive models. Works in the former category carry out non-rigorous statistical physics calculations tailored to specific toy models of data like mixtures of Gaussians or context-free grammars with random production

rules [[SFW25,](#page-24-4) [SFLW24,](#page-24-3) [BBdBM24,](#page-20-2) [RA23](#page-24-2)]. Works in the latter category derive rigorous bounds in settings without explicit parametric structure, e.g. mixtures of strongly log-concave distributions [[LC24](#page-22-4)], but they rely on tools like Girsanov's theorem which are specific to Gaussian diffusion. Additionally, the bounds in the latter are generally cruder, losing dimension-dependent factors. We ask:

*Is there a simple, general theory that can explain critical windows across all generative modeling paradigms and data modalities?*

## <span id="page-3-0"></span>**1.1 Our contributions**

In this work, we develop a simple theoretical framework that characterizes critical windows in both diffusion models and autoregressive models. Our theory is fully rigorous and show that such windows arise generically when the model localizes from a larger sub-population to a smaller one, distilling the phenomenon of critical windows to very general facts about the data distribution. In particular, we show that the critical window is parameterized by the last step of the generative model at which the larger and smaller sub-populations are indistinguishable and the first step of the generative model at which that smaller and larger sub-populations have negligible overlap between each other. We will formalize these notions in Section [3.](#page-7-1)

The key insight of our theory is to apply to the formalism of *stochastic localization samplers* (see Section [2.1](#page-5-1) for a formal description) to understand critical windows [[Mon23a,](#page-23-2) [CE22](#page-21-1)]. Roughly speaking, a stochastic localization scheme is any generative model given by a time-reversal of a Markovian degradation process which takes a sample from the target distribution and generates progressively less informative "observations" of it. In diffusion models, the degradation is a convolution of the original sample with larger and larger amounts of Gaussian noise. In autoregressive models, the degradation is the masking of entries from right to left. Importantly, our theory does not use anything about the specific structure of the sampler beyond the Markovianity of the observation process. Below we highlight our main contributions:

- 1. **Generality**: In comparison to existing work, our theory (Theorem [2\)](#page-8-1) makes very few distributional assumptions and requires no statistical physics or stochastic calculus machinery, relying only on simple mathematical tools. The simplicity and generality of our theory yield concrete improvements to rigorously characterizing the location of critical windows for well-studied models of data. For example, in contrast to [[LC24](#page-22-4)], who could only identify an analogue to Theorem [2](#page-8-1) with bounds that grow at a polynomial factor with dimension, our improved theory allows us to obviate this dimensional-dependence and replace it with a constant.
- 2. **Diverse instantiations**: Another blessing of the generality of our framework is that we can broadly characterize the location of critical windows for many different models of data and generative models, while previous works were restricted to one particular form of data or model. To illustrate the flexibility of our bounds, we explicitly compute the locations and widths of these windows for different generative models and data modalities (Section [4\)](#page-11-0). One such example we provide elucidates a new connection between critical windows for in-context learning and the all-or-nothing phenomenon in statistical inference.
- 3. **Insights into hierarchical data**: We instantiate our bounds for hierarchically structured models of data, significantly generalizing results of [[LC24](#page-22-4)] which only applied to diffusions and Gaussian mixtures (Section [5\)](#page-15-0). This allows us to show that the hierarchy for a generative model may resemble the hierarchy of the true data generating process if both come from the same kind of sampler, but in general may differ. We also use our theory to argue that autoregressive models can support deeper hierarchies than diffusion models.

4. **Experimental results**: Finally, we show that our theory makes accurate predictions for the location of critical windows for LLMs in a toy synthetic setting. We also empirically demonstrate critical windows for generations from LLAMA-3.1-8B-Instruct, Phi-3-7B-Instruct, and Qwen-2.5-7B-Instruct on 7 different math and reasoning benchmarks. Concurrently with [[AAB](#page-20-1)+24, [LLX](#page-23-0)+24], we observe that critical windows occur during important mistakes in the reasoning patterns of LLMs. [1](#page-4-1)

Our theory provides valuable insights for practitioners. For instance, in Example [4](#page-12-1) we provide a model for critical windows in jailbreaks and the Yellowstone example [[Ant24,](#page-20-0) [QPL](#page-24-1)+24], and argue that training on corrections from critical windows can enable models to recover from these 'bad' modes of behavior. This provides rigorous theoretical justification for [[QPL](#page-24-1)+24]'s approach for deepening safety alignment through finetuning.

## <span id="page-4-0"></span>**1.2 Related work**

We briefly overview some related work here and defer our discussion of other relevant literature to Appendix [A.](#page-26-0)

**Theory of critical windows in diffusion.** Several recent works have studied critical windows in the context of diffusion models, using either statistical physics methods [[RA23,](#page-24-2) [SFLW24,](#page-24-3) [SFW25,](#page-24-4) [BBdBM24](#page-20-2)] or Girsanov's theorem [[LC24](#page-22-4)]. The statistical physics papers assume an explicit functional form for the data and use accurate and non-rigorous statistical physics methods to compute critical windows. For instance, [[BBdBM24](#page-20-2)] computes the critical time at which the reverse process specializes to one component for a mixture of two spherical Gaussians using a Landau-type perturbative calculation, and [[SFW25,](#page-24-4) [SFLW24](#page-24-3)] passed through a mean-field approximation to compute the critical windows for a *random hierarchy model* [[PCT](#page-24-5)+23], a multi-level context-free grammar with random production rules. Our work is most similar to [[LC24](#page-22-4)], which derives rigorous, non-asymptotic bounds analogous to our Theorem [2](#page-8-1) for mixtures of log-concave distributions with Girsanov's theorem [[CCL](#page-21-2)+23].

In contrast to existing work, our theory applies to all localization-based samplers, including diffusion and autoregressive language models, and imposes no functional form or log-concavity assumptions on the distribution. We also improve upon the main theorem of [[LC24](#page-22-4)] by obtaining dimension-independent error bounds. Using our improved theorem, we can extend the definition of hierarchy of critical windows from [[LC24](#page-22-4)] to all localization-based samplers and, for continuous diffusions, to distributions beyond mixtures of Gaussians.

**Forward-reverse experiment.** Here we study the forward-reverse experiment, where we *noise and denoise* samples with a given attribute to understand critical windows. This was also explored in [[LC24,](#page-22-4) [SFW25,](#page-24-4) [SFLW24](#page-24-3)]. This approach is very similar to the framework in which one imagines rerunning the reverse process at an intermediate point *Y<sup>t</sup>* [[GVS](#page-22-3)+23, [BBdBM24,](#page-20-2) [RA23](#page-24-2)]. Both perspectives provide rigorous frameworks to understand critical windows, and in the case where the forward process is deterministic, i.e. autoregressive language models, these frameworks are equivalent.

**Stochastic localization.** [[EAMS22,](#page-21-3) [MW23,](#page-24-6) [AMS23,](#page-20-3) [Mon23b,](#page-24-7) [HMP24](#page-22-5)] applied Eldan's stochastic localization method [[Eld13,](#page-21-4) [Eld20](#page-21-5)] to develop new sampling algorithms for distributions inspired by statistical physics. Our work applies the stochastic localization framework [[Mon23a](#page-23-2)] to understand an empirical phenomenon appearing among different localization-based samplers widely used in practice.

<span id="page-4-1"></span><sup>1</sup>The code to reproduce the experiments can be found at [https://github.com/marvinli-harvard/](https://github.com/marvinli-harvard/critical-windows-lm) [critical-windows-lm](https://github.com/marvinli-harvard/critical-windows-lm).

## <span id="page-5-0"></span>2 Technical preliminaries

**Probability notation.** Given distributions P,Q defined on  $(\Omega, \mathcal{F})$  with a base measure  $\mu$ , the *total variation distance* is defined as  $TV(P,Q) \triangleq \frac{1}{2} \int |dP - dQ| d\mu$ . For random variables X,Y, we will also use TV(X,Y) as shorthand to denote the TV of the measures of X,Y. Let  $supp(P) = \{x \in \Omega | dP(x) > 0\}$  denote the support. We will also use the following well-known relationship.

<span id="page-5-5"></span>**Lemma 1.** For probability measures 
$$P, Q, \mathbb{E}_{x \sim P} \left[ \frac{dQ}{dP + dQ} \right] \leq \frac{1}{2} \sqrt{1 - TV^2(P, Q)}$$
.

To study feature localization in diffusion and autoregressive models, we consider a *forward-reverse* experiment. A forward-reverse experiment considers the amount of "noise" one would need to add to a generation so that running the generative model starting from the noised generation would still yield a sample with the same feature. For a diffusion model, this could mean taking an image of a cat, adding Gaussian noise, and resampling to see if the result is still a cat. For a language model, it could mean truncating a story about a cat and resampling to check if the story remains about a cat. Now, we will use the language of stochastic localization to place these analogous experiments for diffusion and language models within the same framework.

## <span id="page-5-1"></span>2.1 Stochastic localization samplers

We formally define the framework for stochastic localization samplers, following [Mon23a]. Let  $X \sim p$  be a random variable over  $\mathbb{R}^d$ . We consider a sequence of random variables  $(Y_t)_{t \in I}$  with a compact index set  $I \subset [0, \infty) \cup \{\infty\}$ . As t increases,  $Y_t$  becomes *less informative* and *degrades* the original information about X (Definition 1). As in [Mon23a], we will only consider *complete* observation processes, where information about the path  $(Y_t)_{t \in I}$  uniquely identifies X: for any measurable set  $A \subset \mathbb{R}^n$ , we require  $P(X \in A | (Y_t)_{t \in I}) \in \{0,1\}$ . For the sake of simplicity, we will assume  $0, \infty \in I$  and  $Y_\infty$  is totally uninformative about X.

<span id="page-5-3"></span>**Definition 1.**  $(Y_t)_{t \in \mathbf{I}}$  is an observation process with respect to X if for any positive integer k and sequence  $t_1 < t_2 < \cdots < t_k \in \mathbf{I}$ , the sequence  $X \to Y_{t_1} \to Y_{t_2} \to \cdots \to Y_{t_k}$  forms a Markov chain.

Because  $X \to Y_{t_1} \to \cdots \to Y_{t_k}$  is a Markov chain, its reverse  $Y_{t_k} \to \cdots \to Y_{t_1} \to X$  is also a Markov chain. To any such observation process one can thus associate a generative model as follows:

**Definition 2.** Given observation process  $(Y_t)_{t\in I}$  and times  $t_1 < \cdots < t_m = \infty$  in I, the associated stochastic localization sampler is the algorithm that generates a sample for X by first sampling  $Y_{t_m}$  and then, for  $k = m-1, m-2, \ldots, 0$ , sampling from the posterior on  $Y_{t_k}$  conditioned on  $Y_{t_{k+1}}$  by taking one step in the reverse Markov chain above, and finally sampling X conditioned on  $Y_{t_0}$ .

In Appendix B, we formally verify that diffusion and autoregressive models are special cases of this framework. In practice, one does not have access to the true posteriors of the data distribution and must learn approximations to the posterior from data. This issue of learning the true distribution is orthogonal to our work, and thus we define  $X \sim p$  to be the sampler's distribution. Furthermore, it is more natural to study the sampler's distribution for applications such as interpretability or jailbreaks.

<span id="page-5-4"></span><span id="page-5-2"></span><sup>&</sup>lt;sup>2</sup>These definitions are easily carried over to the setting where *X* lives in a discrete space.

<sup>&</sup>lt;sup>3</sup>Note that our formulation of stochastic localization differs slightly in several minor ways. First, in that work the index set **I** is not necessarily compact; while we assume compactness of **I**, this still encapsulates most applications of generative models, in which the sample X is realized in finitely many steps. Secondly, our indexing of time is the reverse of that of in [Mon23a]; in that work, the Y,'s become *more* informative about X as t increases. We make this choice purely for cosmetic reasons.

**Features, mixtures, and sub-mixtures.** To capture the notion of a feature of the generation, we assume that the distribution  $X \sim p$  is a *mixture model*. Consider a discrete set  $\Theta = \{\theta_1, \dots, \theta_K\}$  with non-negative weights  $w_1, \dots, w_K$  summing to 1. Each  $\theta_i \in \Theta$  is associated with a probability density function  $p^{\theta_i} : \mathbb{R}^n \to \mathbb{R}^{\geq 0}$ . To generate a sample  $X \sim p$ , we first draw  $\theta \sim \text{Cat}(\Theta, \{w_i\}_{i=1}^K)$  and return  $X \sim p^{\theta}$ . This yields an overall density of  $p \triangleq \sum_{\theta \in \Theta} w_{\theta} p^{\theta}$ . For any non-empty  $S \subset \Theta$ , we also define the sub-mixture  $p^S$  by  $p^S \triangleq \sum_{\theta \in S} \frac{w_{\theta}}{\sum_{\phi \in S} w_{\phi}} p^{\theta}$ .

**Remark 1.** Note that the definition of  $\Theta$  is extremely flexible and can be tailored to the particular data modality or task. For example,  $\Theta$  could be  $\{cat, dog\}$  for image diffusion models;  $\{right, wrong\}$  for math and reasoning tasks;  $\{unsafe, safe\}$  for jailbreaks.

Here we study a *family of observation processes* corresponding to observation processes for different initial distributions of  $X \sim p^S$  for  $S \subset \Theta$ . To ensure that we can meaningfully compare the observation processes within this family, we will assume that the *degradation procedure is fixed*. To formalize this intuition, we borrow the language from diffusion models of a forward process, which degrades X, and a reverse process, which takes a degraded  $Y_t$  and produces X.

## <span id="page-6-0"></span>2.2 Forward-reverse experiment

Now we describe the general formalism under which we will study critical windows. Fixing some  $t \in \mathbf{I}$  and  $S \subset \Theta$ , we start with some  $X \sim p^S$ , sample  $Y_t|X$  from the observation process conditioning on X, and finally take  $X'|Y_t$  from the stochastic localization sampler conditioning on  $Y_t$ . The can be understood as a generalization of the *forward-reverse* experiment in diffusions, originally studied in [SFW25, SFLW24, LC24], to arbitrary stochastic localization samplers.

Forward process. For any  $t \in \mathbf{I}$ , define the forward Markov transition kernel  $P_t^{\neg}(A|X) = P(Y_t \in A|X)$ . Note the forward Markov transition kernel does not depend on the distribution of X. The fact that the forward process is agnostic to the specifics of the original distribution is shared by the most widely used stochastic localization samplers. For example, in diffusion and flow-matching models, the forward transition is a convolution of X with a Gaussian; in autoregressive language models, it is masking of the last remaining token in the sequence. For any  $t \in \mathbf{I}$  and  $S \subset \Theta$ , we let  $P_t^S$  denote the law of  $Y_t^S$ , where we sample  $X^S \sim P^S$  and then sample  $Y_t^S \sim P_t^{\neg}(\cdot|X^S)$ . We omit the  $\Theta$  in  $P_t^\Theta$ .

**Reverse process.** For any  $t \in \mathbf{I}$  and initial distribution  $X \sim p$ , we define the posterior of X given  $Y_t$  by  $P^-(A|Y_t) = P_{X \sim p}(X \in A|Y_t)$ , that is, the distribution of X given by starting the sampling process at  $t \in \mathbf{I}$  and  $Y_t$  instead of  $\infty$  and  $Y_{\infty}$ . We will also use this notation for the probability density.

Now, we are ready to describe the main forward-reverse experiment that we will study.

**Definition 3** (Forward-reverse experiment [SFW25, SFLW24, LC24]). For nonempty  $S \subset \Theta$  and  $\widehat{T} \in I$ , let  $p^{S,\widehat{T}}$  be the distribution of  $X^{S,\widehat{T}}$  defined by the following procedure:<sup>4</sup>

- 1. Sample  $Y_{\widehat{T}}^S \sim p_{\widehat{T}}^S$  i.e. run the forward process for time  $\widehat{T}$  starting at the sub-mixture  $p^S$ .
- 2. Sample  $X^{S,\widehat{T}} \sim P^{\leftarrow}(\cdot|Y_{\widehat{T}}^S)$  i.e. run the reverse process starting at  $Y_{\widehat{T}}^S$  to sample from the posterior on X.

We emphasize that in the second step, we run the reverse process with the prior on X given by the *entire distribution* p rather than the sub-mixture  $p^S$ . If we did the latter, the marginal distribution of the

<span id="page-6-1"></span><sup>&</sup>lt;sup>4</sup>Note that this equips  $2^{\Theta}$  with the structure of a poset, i.e.  $A \leq B$  if and only if there exists some  $t \in I$  such that running the forward-reverse experiment up to t from  $p^A$  yields  $p^B$ .

result would simply be  $p^S$ . Instead, the marginal distribution of  $X^{S,\widehat{T}}$  is some distribution whose relation to p and sub-mixtures thereof is a priori unclear. Intuitively, as  $\widehat{T} \to 0$ , this distribution converges to  $p^S$ , and as  $\widehat{T} \to \infty$ , this distribution converges to p. The essence of our work is to understand the transition between these two regimes as one varies  $\widehat{T}$ .

**Remark 2.** Another way to think of  $p^{S,\hat{T}}$  to consider a partition of  $\Theta$  into  $\bigsqcup_{i=1}^n S_i$  and observe that you can write  $p = \sum_{i=1}^n \left(\sum_{j \in S_i} w_j\right) p^{S_i,\hat{T}}$ . The forward-reverse experiment defines an alternative decomposition of the mixture p, where the classes in  $\Theta$  are "smudged" in this sense.

#### <span id="page-7-0"></span>2.2.1 Instantiation for LLMs

For intuition about what these definitions actually mean, consider an autoregressive language model, which produces stories of cats or dogs. We have a discrete set of tokens  $\mathcal{A}$  and  $X \in \mathcal{A}^T$  representing length-T sequences. Letting  $\mathbf{I} = \{0,1,2\ldots,T\}$ , the observation process is defined with  $Y_i \in \mathcal{A}^{T-i}$ ,  $Y_0 \triangleq X$ , and  $Y_i$  being the first T-i elements of  $Y_{i-1}$  for  $i=1,\ldots,T$ . It is easy to see that this is Markovian and the samples become less informative about the original X as  $t \to \infty$ . In the associated stochastic localization sampler, the posterior  $Y_t|Y_{t+1}$  for adjacent  $t,t+1\in \mathbf{I}$  is the conditional distribution of T-t-length sequences given a prefix of length T-(t+1), exactly the task of next-token prediction. Finally, we study the forward-reverse experiment applied to a story of a cat. For LLMs, this means masking the last  $\widehat{T}$  tokens of a sample and then resampling with the same model. If  $\widehat{T}$  is small, the story will likely still mention a cat and resampling will yield a story about a cat. If  $\widehat{T}$  is large, then the first appearance of cat may be truncated, so resampling could produce a story about a dog as well.

## <span id="page-7-1"></span>3 Characterization of critical windows

<span id="page-7-2"></span>For expositional clarity, we will introduce our theory along with a toy setting to provide some useful intuition: consider an image diffusion model that outputs pictures of orange cats, brown cats, and dogs and a hypothetical critical window in which we transition from sampling from both cats and dogs to cats with certainty (Figure 2). In this case, the critical window simply refers to the dramatic increase in the probability of sampling a cat between time interval  $[T_{\text{after}}^C, T_{\text{before}}^C]$ . Note that the intuition heretofore will apply to all localization-based samplers with a forward and reverse process indexed by a time  $t \in \mathbb{R}^+$ .

![](_page_7_Figure_6.jpeg)

Figure 2: Illustration of a critical window for a cat feature with an image diffusion model.

Importantly, the critical window is situated between and in some sense defined by the two intervals which surround it: the left interval indicates that we have committed to only sampling from the

distribution of cats indexed by  $S_{\rm after}$  and have a high probability of producing a cat as the final image, and the right interval, indexed by  $S_{\rm before}$ , indicates we can sample images of cats or dogs. Thus critical windows simply represent the *speciation* from sampling from a larger sub-population of the distribution given by  $S_{\rm before}$  to a smaller sub-population given by  $S_{\rm after}$ . In other words, we want to understand at what interval  $t \in [B_1, B_2]$  is some *feature* (herein the feature of being a cat or dog) determined by the diffusion process and then at what interval  $t \in [A_1, A_2]$  the more *specific version of the feature* (herein the feature of being a cat) is determined by the diffusion, yielding the critical window as the transition period  $[A_2, B_1]$  in which we specialize to the more specific version of the feature. We can characterize both of these intervals  $[A_1, A_2], [B_1, B_2]$  via the forward-reverse experiment.

More generally, let  $S_{\text{init}} \subset \Theta$  denote some sub-mixture, corresponding to a sub-population of p that possesses a certain property. Let  $S_{\text{target}} \supset S_{\text{init}}$  denote some sub-mixture containing  $S_{\text{init}}$ . For instance,  $S_{\text{target}}$  might correspond to all possible responses to the math question, including incorrect ones. We are interested in the following question: if we run the forward-reverse experiment for time  $\widehat{T}$  starting from  $p^{S_{\text{init}}}$ , is there some range of times for which the resulting distribution is close to  $p^{S_{\text{target}}}$ ? That is, can we characterize the  $\widehat{T}$  for which

$$\mathsf{TV}(p^{S_{\mathsf{init}},\widehat{T}},p^{S_{\mathsf{target}}})$$

is small?

Suppose one could prove that the range of  $\widehat{T}$  for which this is the case is some interval  $[T_0, T_1]$ . This would mean that if the stochastic localization sampler runs for time T and ends up at a sample from  $p^{S_{\text{init}}}$ , then from time  $T-T_1$  to time  $T-T_0$  of the generation process, the sampler has not yet localized the features that distinguish  $p^{S_{\text{init}}}$  from the larger sub-mixture  $p^{S_{\text{target}}}$ . However, the sampler has localized the features that distinguish  $p^{S_{\text{target}}}$  from  $p^{\Theta-S_{\text{target}}}$ . When there is a shift from localizing the features  $S_{\text{target}}$  to the features  $S_{\text{init}}$ , we say there is a critical window. We now formally state and prove our main result.

#### <span id="page-8-0"></span>3.1 Main result

We will parameterize the interval by the total variation distance between sub-mixtures  $S_{\text{init}}$  and  $S_{\text{target}}$  along the forward process. For an error parameter  $0 < \epsilon < 1$ , define

$$\begin{split} & T_{\text{start}}^{S}(\epsilon) \in \sup\{t \in \mathbf{I} : \text{TV}(p_{t}^{S_{\text{target}}}, p_{t}^{\Theta - S_{\text{target}}}) \geq 1 - \epsilon^{2}\} \\ & T_{\text{end}}^{S}(\epsilon) \in \inf\{t \in \mathbf{I} : \text{TV}(p_{t}^{S_{\text{init}}}, p_{t}^{S_{\text{target}}}) \leq \epsilon\} \,. \end{split}$$

When the value of  $\epsilon$  is understood, we abbreviate the above with  $T_{\text{start}}^S$  and  $T_{\text{end}}^S$ . Our main result is that in  $\widehat{T} \in \mathbf{I} \cap [T_{\text{end}}^S, T_{\text{start}}^S]$ , the distance  $\text{TV}(p^{S_{\text{init}}, \widehat{T}}, p^{S_{\text{target}}})$  is small:

<span id="page-8-1"></span>**Theorem 2.** Let 
$$S_{\text{init}} \subset S_{\text{target}} \subset \Theta$$
 and  $W = \frac{\sum_{\theta \in \Theta - S_{\text{target}}} w_{\theta}}{\sum_{\theta \in S_{\text{target}}} w_{\theta}}$ . For  $\epsilon > 0$ , if  $\widehat{T} \in \mathbf{I} \cap [T_{\text{end}}^S, T_{\text{start}}^S]$ , then  $\text{TV}(p^{S_{\text{init}}, \widehat{T}}, p^{S_{\text{target}}}) \leq \epsilon \cdot (1 + \max(1, W) / \sqrt{2})$ .

Intuitively,  $T_{\text{start}}^S$  represents the largest t for which there is still separation between  $S_{\text{target}}$  and  $\Theta - S_{\text{target}}$ , and  $T_{\text{end}}^S$  represents the smallest t for which samples from  $S_{\text{init}}$ ,  $S_{\text{target}}$  are indistinguishable. Thus, running it for  $\widehat{T} \in \mathbf{I} \cap [T_{\text{end}}^S, T_{\text{start}}^S]$  erases the differences between samples from  $S_{\text{init}}$  and  $S_{\text{target}}$  but preserves the difference between  $S_{\text{target}}$  and  $\Theta - S_{\text{target}}$ , yielding samples looking like  $p^{S_{\text{target}}}$ .

![](_page_9_Picture_0.jpeg)

Figure 3: Intuition with the forward-reverse experiment with  $S_{\rm init} = \{ {\rm orange\ cat} \}$  and  $S_{\rm target} = \{ {\rm orange\ cat}, {\rm brown\ cat} \}$ . In the low-noise regime, the forward-reverse experiment does not modify the image non-trivially, meaning we still sample from  $S_{\rm init}$ ; in the high-noise regime, the forward-reverse experiment destroys all information in the image and thus could yield a dog, lying outside of  $S_{\rm target}$ . Thus there is a "sweet spot" where the differences between orange and brown cats collapse but those between cats and dogs persist.

A priori it is not clear why  $T_{\rm end}^S$  should be smaller than  $T_{\rm start}^S$ , and indeed in general it need not be and our bound would be vacuous. In Section 4 however, we show that in many natural settings for diffusion models and autoregressive models, the relation does hold.

**Remark 3.** Note a similar bound was shown in the context of diffusions by [LC24] (see Theorem 7 therein). Our result is a strict improvement of that bound along several important axes. First, our results apply to all stochastic localization samplers, not just diffusions. Secondly, [LC24] needed to assume that the components of p were strongly log-concave and that the score, i.e. gradient of the log-density, of  $p_t$  was Lipschitz and moment-bounded for all t. Thirdly, their final bound includes a polynomial dependence on the moments of the score, which scale with the dimension d; in contrast, our final bound is independent of d.

With Theorem 2 in place, we are ready to formally define *critical windows*. These capture the moments where we transition from sampling from a sub-mixture to a subset of that sub-mixture.

**Definition 4.** Define  $S_{\text{after}} \subset S_{\text{before}} \subset \Theta$ . For  $S_{\text{before}}$ , we define  $T_{\text{before}}^C = \inf\{t \in \mathbf{I} : \text{TV}(p_t^{S_{\text{after}}}, p_t^{S_{\text{before}}}) \leq \epsilon$  and  $\text{TV}(p_t^{S_{\text{before}}}, p_t^{\Theta - S_{\text{before}}}) \geq 1 - \epsilon^2\}$  ( $S_{\text{init}} \triangleq S_{\text{after}}; S_{\text{target}} \triangleq S_{\text{before}}$ ). For  $S_{\text{after}}$ , consider  $T_{\text{after}}^C = \sup\{t \in \mathbf{I} : \text{TV}(p_t^{S_{\text{after}}}, p_t^{\Theta - S_{\text{after}}}) \geq 1 - \epsilon^2\}$  ( $S_{\text{init}}, S_{\text{target}} \triangleq S_{\text{after}}$ ). A **critical window** is the interval  $[T_{\text{after}}^C, T_{\text{before}}^C]$ , where there is a transition from sampling from  $S_{\text{before}}$  to the smaller subset  $S_{\text{after}}$ .

Note that this definition does not directly explain why or when a critical window is sharp, which depends on the properties of the model of the data and forward process. In Section 4, we analytically compute critical windows in different toy settings and demonstrate that they are sharp in a natural sense. These examples also provide general intuition where critical windows are sharp, i.e. for autoregressive LLMs, critical windows may occur when a few tokens are very unlikely under one mode than other in Example 4. In general, we intuit that sharp critical windows occur when it only takes a few steps from the forward process to erase the differences between  $S_{\text{before}}$  and  $S_{\text{after}}$ . This could happen if the data has

a multi-scale hierarchical structure, where a feature is decided in a narrow intermediate band of the tree (Section 5).

## <span id="page-10-0"></span>3.2 Proof of Theorem 2

Crucially, our proof relies in several places on the Markov property of stochastic localization samplers, together with the data processing inequality. We also define some helpful notation for our proof. For nonempty  $S \subset \Theta$  and  $t \in \mathbf{I}$ , we define  $P^{\leftarrow}(\cdot|Y_t,S)$  to be the posterior of X with the prior  $X \sim p^S$ . We similarly define  $P^{\leftarrow}_{t\to\Theta}(\cdot|Y_t)$  and  $P^{\leftarrow}_{t\to\Theta}(\cdot|Y_t,S)$  to be the posterior of  $\theta$  conditioning on  $Y_t$  with  $X \sim p$  or  $X \sim p^S$ , respectively. When  $S = \{i\}$ , we exclude the braces.

Proof of Theorem 2. By the triangle inequality, we can write

$$\mathrm{TV}(p^{S_{\mathrm{init}},\widehat{T}},p^{S_{\mathrm{target}}}) \leq \underbrace{\mathrm{TV}(p^{S_{\mathrm{init}},\widehat{T}},p^{S_{\mathrm{target}},\widehat{T}})}_{(\mathrm{I})} + \underbrace{\mathrm{TV}(p^{S_{\mathrm{target}},\widehat{T}},p^{S_{\mathrm{target}}})}_{(\mathrm{II})}.$$

 $p^{S_{\mathrm{init}},\widehat{T}}$  and  $p^{S_{\mathrm{target}},\widehat{T}}$  are the laws of the posterior  $P^-(\cdot|\cdot)$  but applied to  $Y_{\widehat{T}}$  with distributions  $p_{\widehat{T}}^{S_{\mathrm{init}}}$  and  $p_{\widehat{T}}^{S_{\mathrm{target}}}$ . Using the Markov property of localization-based samplers (Definition 1), we apply the data processing inequality twice and the definition of  $T_{\mathrm{end}}^S$  to bound (I) via

$$\mathsf{TV}(p^{S_{\mathsf{init}},\widehat{T}},p^{S_{\mathsf{target}},\widehat{T}}) \leq \mathsf{TV}(p_{\widehat{T}}^{S_{\mathsf{init}}},p_{\widehat{T}}^{S_{\mathsf{target}}}) \leq \mathsf{TV}(p_{T_{\mathsf{end}}}^{S_{\mathsf{init}}},p_{T_{\mathsf{end}}}^{S_{\mathsf{target}}}) \leq \epsilon.$$

To bound (II), we use the definition of TV and a coupling argument. Observe that the observation processes associated with  $p^{S_{\text{target}},\widehat{T}}$  and  $p^{S_{\text{target}}}$  have the same distribution at index  $\widehat{T}$ . Thus, taking  $Y_{\widehat{T}} \sim p_{\widehat{T}}^{S_{\text{target}}}$ , we can express by the law of total probability,

$$p^{S_{\text{target}}, \widehat{T}}(x) = \mathbb{E}[P^{\leftarrow}(x|Y_{\widehat{T}})]$$
$$p^{S_{\text{target}}}(x) = \mathbb{E}[P^{\leftarrow}(x|Y_{\widehat{T}}, S_{\text{target}})].$$

as these observation processes have the same distribution at index  $\hat{T}$ . Thus,

$$\text{TV}(p^{S_{\text{target}},\widehat{T}},p^{S_{\text{target}}}) = \frac{1}{2} \int \left| p^{S_{\text{target}},\widehat{T}}(x) - p^{S_{\text{target}}}(x) \right| dx = \frac{1}{2} \int \left| \mathbb{E}[P^{\leftarrow}(x|Y_{\widehat{T}})] - \mathbb{E}[P^{\leftarrow}(x|Y_{\widehat{T}},S_{\text{target}})] \right| dx.$$

By Jensen's inequality and Fubini's theorem, we bring the expectation outside the integral,

$$\mathrm{TV}(p^{S_{\mathrm{target}},\widehat{T}},p^{S_{\mathrm{target}}}) \leq \frac{1}{2} \int \mathbb{E}\left[\left|P^{\leftarrow}(x|Y_{\widehat{T}}) - P^{\leftarrow}(x|Y_{\widehat{T}},S_{\mathrm{target}})\right|\right] dx = \frac{1}{2} \mathbb{E}\left[\int \left|P^{\leftarrow}(x|Y_{\widehat{T}}) - P^{\leftarrow}(x|Y_{\widehat{T}},S_{\mathrm{target}})\right| dx\right].$$

To simplify the above expression, we use the following two lemmas, proved in App. C.1.

<span id="page-10-1"></span>**Lemma 3.** By applying the law of total probability and Bayes' rule, we can show for  $Y_{\widehat{T}} \in \text{supp}(p_{\widehat{T}}^{S_{\text{target}}})$ ,

$$\int \left| P^{\leftarrow}(x|Y_{\widehat{T}}) - P^{\leftarrow}(x|Y_{\widehat{T}}, S_{\text{target}}) \right| dx \le 2 \sum_{\theta \in \Theta - S_{\text{target}}} P_{t \to \Theta}^{\leftarrow}(\theta|Y_{\widehat{T}}).$$

<span id="page-11-2"></span>**Lemma 4.** By Bayes's rule, we can derive for  $Y_{\widehat{T}} \in \text{supp}(p_{\widehat{T}})$ ,

$$\sum_{\theta \in \Theta - S_{\text{target}}} P_{t \to \Theta}^{\leftarrow}(\theta | Y_{\widehat{T}}) \leq \max(1, W) \frac{p_{\widehat{T}}^{\Theta - S_{\text{target}}}(Y_{\widehat{T}})}{p_{\widehat{T}}^{\Theta - S_{\text{target}}}(Y_{\widehat{T}}) + p_{\widehat{T}}^{S_{\text{target}}}(Y_{\widehat{T}})}$$

Combining Lemmas 3 and 4, we find

$$\text{TV}(p^{S_{\text{target}},\widehat{T}}, p^{S_{\text{target}}}) \leq \max(1, W) \mathbb{E}\left[\frac{p_{\widehat{T}}^{\Theta - S_{\text{target}}}(Y_{\widehat{T}})}{p_{\widehat{T}}^{\Theta - S_{\text{target}}}(Y_{\widehat{T}}) + p_{\widehat{T}}^{S_{\text{target}}}(Y_{\widehat{T}})}\right].$$

Then, finally applying Lemma 1, we are able to bound the total variation in terms of  $\epsilon$ ,

$$\text{TV}(p^{S_{\text{target}},\widehat{T}}, p^{S_{\text{target}}}) \leq \frac{1}{2} \max(1, W) \sqrt{1 - \text{TV}^2(p_{\widehat{T}}^{\Theta - S_{\text{target}}}, p_{\widehat{T}}^{S_{\text{target}}})} \leq \frac{\sqrt{2}}{2} \max(1, W) \epsilon.$$

Combining our bounds on (I) and (II) achieves the desired result.

## <span id="page-11-0"></span>4 Examples of critical windows

In this section, we analytically compute  $T_{\mathrm{before}}^C$ ,  $T_{\mathrm{after}}^C$  for diverse stochastic localization samplers and models of data, including diffusion and autoregression processes. In these natural settings, the critical window is small in the sense of having a size which shrinks or does not depend on the dimension or context length. We shall also connect our framework to in-context learning and the all-or-nothing phenomenon. <sup>6</sup>

## <span id="page-11-1"></span>4.1 Diffusion

We first consider two examples of Gaussian Mixture Models and a diffusion model. We show that with two isotropic Gaussians, the critical window appears around a single point,  $\ln \|\mu\|$ , with width independent of the dimension.

**Example 1** (Two Isotropic Gaussians). Let  $\Theta = \{\pm 1\}$ ,  $p^{+1} = \mathcal{N}(\mu, \mathrm{Id})$ ,  $p^{-1} = \mathcal{N}(-\mu, \mathrm{Id})$ . Then, we have a critical window transitioning from sampling from both components to the component +1 between  $T_{\mathrm{before}}^C = \ln \|\mu\| + \ln 2 + \ln 1/\epsilon$  and  $T_{\mathrm{after}}^C = \ln \|\mu\| - \ln \ln \frac{1}{2\epsilon^2}$ . When  $\widehat{T} \leq T_{\mathrm{after}}^C$ , then  $\mathrm{TV}(p^{+1,\widehat{T}},p^{+1}) \lesssim \epsilon$ . When  $\widehat{T} \geq T_{\mathrm{before}}^C$ ,  $\mathrm{TV}(p^{+1,\widehat{T}},p) \lesssim \epsilon$ .

For an isotropic Gaussian mixture model with randomly selected means, the critical window between sampling from one component to the entire mixture is also narrow. Note that we derive dimension-free widths in Example 2, an improvement over [LC24] who had a  $\ln \ln d$  dependence on dimension for isotropic Gaussians.

<span id="page-11-4"></span>**Example 2** (Random mean spherical Gaussians). We first sample  $\mu_i \sim \mathcal{N}(0, \mathrm{Id})$  for  $i \in [K]$  i.i.d. and let  $\Theta = \{\mathcal{N}(\mu_i, \mathrm{Id})\}_{i \in [K]}$ . We let  $S_{\mathrm{before}} = \Theta$  and  $S_{\mathrm{after}} = \{\mu_i\}$ . Then, we can compute  $T_{\mathrm{before}}^C = \max_{j \in [K]} \ln \|\mu_i - \mu_j\| + \ln(1/\epsilon)$  and  $T_{\mathrm{after}}^C = \min_{j \in [K], i \neq j} \ln \|\mu_i - \mu_j\| - \frac{1}{2} \ln 8 \ln \frac{K}{\epsilon}$ . Furthermore, with high probability over the selection of the means,  $T_{\mathrm{before}}^C - T_{\mathrm{after}}^C = O_{K,\epsilon}(1)$  as  $d \to \infty$ .

<span id="page-11-3"></span><sup>&</sup>lt;sup>6</sup>Proofs are deferred to Appendix C.2.

We also compute the critical windows of a discrete diffusion model. As the context length T goes infinity, we show that the length of the critical window goes to 0.

**Example 3** (Two Dirac delta functions with a random masking procedure). Let  $p \in \{\pm 1\}^T$ , and consider a forward process with index set  $\mathbf{I} = [0,1]$ ,  $Y_0 = X$ , and  $Y_t \in \{\pm 1, [\text{MASKED}]\}^T$ . For  $t \in \mathbf{I}$ , we let all the value at index  $i \in [T]$  be set to [MASKED] with probability t independently. For a mixture of two Dirac delta functions, we can express the critical window in terms of the Hamming distance between the corresponding strings. Let  $\Theta = \{\theta_{\pm 1}\}, \ell_{\pm 1} \in \{\pm 1\}^T, p^{\theta_{\pm 1}} \sim \delta_{\ell_{\pm 1}}, w_{\pm 1} = \frac{1}{2}$ . Then, on component 1 we have the critical window

$$T_{\text{before}}^{C} = \exp\left[\frac{\ln(1-\epsilon)}{d_{H}(\delta_{\ell_{1}}, \delta_{\ell_{-1}})}\right], T_{\text{after}}^{C} = \exp\left[\frac{\ln\epsilon^{2}}{d_{H}(\delta_{\ell_{1}}, \delta_{\ell_{-1}})}\right]$$

For sufficiently large  $d_H(\delta_{\ell_1}, \delta_{\ell_{-1}})$ , the window size  $T^C_{\text{before}} - T^C_{\text{after}} = O\left(\frac{1}{d_H(\delta_{\ell_1}, \delta_{\ell_{-1}})}\right)$ . If  $d_H(\delta_{\ell_1}, \delta_{\ell_{-1}})$  increases with T, then the width of the critical window negligible.

## <span id="page-12-0"></span>4.2 Autoregression

We first present a theoretical model for important critical windows in LLMs, e.g., jailbreaks that occur over the first few tokens in the generation and the Yellowstone example [Ant24, QPL<sup>+</sup>24].

<span id="page-12-1"></span>**Example 4** ("Critical Tokens" for Jailbreaks and Yellowstone [QPL+24, Ant24]). Again consider an autoregressive language model, with  $\mathcal{A}$  denoting the vocabulary,  $p \in \mathcal{A}^T$ , a forward process indexed by  $\mathbf{I} = \{0,1,2,\ldots,T\}$ , and  $Y_t$  to be the first T-t tokens of X. Let  $\Theta = \{\theta_{harmful}, \theta_{safe}\}$  (or  $\{\theta_{Googling\ Yellowstone}, \theta_{coding}\}$ ). We assume that these two modes do not differ until some  $T-T' \in \mathbf{I}$ . Between T-T' and T-T'-k, the distributions become nearly disjoint,  $P_{x \sim p_{T-T'-k}}^{\theta_{harmful}} \left(x \in \operatorname{supp}(p_{T-T'-k}^{\theta_{safe}})\right) \leq \epsilon$ . In the jailbreaking example, T' = 0 and they are disjoint because the first tokens generated in the safe mode is always some form of refusal. In the Yellowstone example, they are disjoint the first time the agent decides to Google Yellowstone pictures. Then, on component  $\theta_{harmful}$  we have the critical window  $T_{before}^C = T - T'$  and  $T_{after}^C = T - T' - k$ .

Notice that we can actually mitigate the effect of these critical windows by finetuning on examples of corrections to increase  $P_{x \sim p_{T-T'-k}^{\theta_{\text{harmful}}}} \left( x \in \text{supp}(p_{T-T'-k}^{\theta_{\text{safe}}}) \right)$ . This explains the effectiveness of finetuning on corrections in [QPL+24]. Furthermore, the quantity that measures probability of mode-switching,  $p^{\theta_{\text{harmful}}}/p$ , suggests using a likelihood ratio to distinguish between harmful and benign prompts. In App. D.1.2, we test a class of likelihood ratio methods that obtain recall 5-10× the false positive rate for 5 different types of jailbreaks (Table 2). We can also identify a critical window for a stylized model of solving a math problem as a random walk.

**Example 5** (Math problem-solving as a random walk). We model solving a math problem as taking a random walk on  $\mathbb{Z}$  with stepsize 1 of length T. If the random walk hits +A, then it has 'solved' the problem; if the random walk hits -A, then it has obtained an incorrect solution. Assume that we have two modes: a strong problem solving mode (denoted +1), which takes a+1 step with probability  $0.5+\Delta$ , and a weak problem solving mode (denoted -1), which takes a+1 step with probability  $0.5-\Delta$ . Assuming that  $\frac{\ln(2/\epsilon^2)}{2\Delta^2} < A$  and  $\epsilon^2 < 10^{-3}(0.5-\Delta)(0.5+\Delta)$ , there is a critical window for the strong problem solving window of  $T_{\text{before}}^C = T - \frac{\epsilon^2}{\Delta^2} + 2$  and  $T_{\text{after}}^C = T - \frac{\ln(2/\epsilon^2)}{2\Delta^2}$ . Note the critical window has width  $\Theta(1/\Delta^2)$  independent of T.

We defer an example of a critical window for an autoregressive model which expresses the outputs as emissions from a random walk of an underlying concept variable, akin to the model in [ALL<sup>+</sup>19], to Appendix C.2.3.

### <span id="page-13-0"></span>4.2.1 In-context learning

Autoregressive critical windows can also be applied to describe in-context learning. In particular, we can capture the idea that with sufficiently many in-context examples, we learn the  $\theta^* \in \Theta$  that generated the transitions for in-context examples, with a sample complexity in terms of  $T_{\text{after}}^C$ .

**Example 6** (Informal, see Example 11). Consider an in-context learning setup, where the context

$$[x_1, y_1, o, \dots, x_{T+1}, y_{T+1}, o]$$

consists of question-answer pairs  $(x_i, y_i)$ , delimiters o, and  $x_i \to y_i$  sampled from  $p^{\theta^*}$  for some  $\theta^* \in \Theta$ . In the forward-reverse experiment, we truncate it to  $[x_1, y_1, o, ..., x_{T+1}]$ , and then resample with p to produce  $[x_1, ..., x_{T+1}, \tilde{y}_{T+1}, o]$ . The total variation between the sequences  $[x_1, y_1, o, ..., x_{T+1}]$  and  $[x_1, ..., x_{T+1}, \tilde{y}_{T+1}, o]$  can be viewed as the average-case error of the in-context learner and can be bounded within our critical windows framework. We have  $T_{\text{after}}^C = 3T + 3 - O_{\epsilon}(1)$ , with  $O_{\epsilon}(1)$  independent of  $O_{\epsilon}(1)$  independent of  $O_{\epsilon}(1)$  independent of  $O_{\epsilon}(1)$  independent of  $O_{\epsilon}(1)$  independent of  $O_{\epsilon}(1)$  independent of  $O_{\epsilon}(1)$  independent of  $O_{\epsilon}(1)$  independent of  $O_{\epsilon}(1)$  independent of  $O_{\epsilon}(1)$  independent of  $O_{\epsilon}(1)$  independent of  $O_{\epsilon}(1)$  independent of  $O_{\epsilon}(1)$  independent of  $O_{\epsilon}(1)$  independent of  $O_{\epsilon}(1)$  independent of  $O_{\epsilon}(1)$  independent of  $O_{\epsilon}(1)$  independent of  $O_{\epsilon}(1)$  independent of  $O_{\epsilon}(1)$  independent of  $O_{\epsilon}(1)$  independent of  $O_{\epsilon}(1)$  independent of  $O_{\epsilon}(1)$  independent of  $O_{\epsilon}(1)$  independent of  $O_{\epsilon}(1)$  independent of  $O_{\epsilon}(1)$  independent of  $O_{\epsilon}(1)$  independent of  $O_{\epsilon}(1)$  independent of  $O_{\epsilon}(1)$  independent of  $O_{\epsilon}(1)$  independent of  $O_{\epsilon}(1)$  independent of  $O_{\epsilon}(1)$  independent of  $O_{\epsilon}(1)$  independent of  $O_{\epsilon}(1)$  independent of  $O_{\epsilon}(1)$  independent of  $O_{\epsilon}(1)$  independent of  $O_{\epsilon}(1)$  independent of  $O_{\epsilon}(1)$  independent of  $O_{\epsilon}(1)$  independent of  $O_{\epsilon}(1)$  independent of  $O_{\epsilon}(1)$  independent of  $O_{\epsilon}(1)$  independent of  $O_{\epsilon}(1)$  independent of  $O_{\epsilon}(1)$  independent of  $O_{\epsilon}(1)$  independent of  $O_{\epsilon}(1)$  independent of  $O_{\epsilon}(1)$  independent of  $O_{\epsilon}(1)$  independent of  $O_{\epsilon}(1)$  independent of  $O_{\epsilon}(1)$  independent of  $O_{\epsilon}(1)$  independent of  $O_{\epsilon}(1)$  independent of  $O_{\epsilon}(1)$  independent of  $O_{\epsilon}(1)$  independent of  $O_{\epsilon}(1)$  independent of  $O_{\epsilon}(1)$  independent of  $O_{\epsilon}(1)$  independent of  $O_{\epsilon}(1)$  indepe

One might ask if there is a  $T_{\mathrm{before}}^{C}$  for in-context learning, a threshold such that it is impossible to distinguish between  $S_{\mathrm{after}}$ ,  $S_{\mathrm{before}}$  with that many samples. In the next section, we will provide an example of a  $T_{\mathrm{before}}^{C}$  for in-context learning with the all-or-nothing phase transition.

#### <span id="page-13-1"></span>4.2.2 All-or-nothing phenomenon

Here we elucidate a formal connection between the critical windows phenomenon in in-context learning and the *all-or-nothing* phenomenon. To begin, we first define the notions of strong and weak detection:

**Definition 5.** Let  $(N_s)$  be an increasing sequence of integers. Given sequences of distributions  $(p_s), (q_s)$  over  $z \in \mathbb{R}^{N_s}$ , a sequence of test statistics  $(A_s : \mathbb{R}^{N_s} \to \mathbb{R})$  with threshold  $(\tau_s)$  achieves:

- strong detection if  $\limsup_{s\to\infty} \{ \Pr_{z\sim p_s} [A_s(z) < \tau_s] + \Pr_{z\sim q_s} [A_s(z) \ge \tau_s] \} = 0.$
- weak detection if  $\limsup_{s\to\infty} \{ \Pr_{z\sim p_s} [A_s(z) < \tau_s] + \Pr_{z\sim q_s} [A_s(z) \ge \tau_s] \} < 1.$

By the operational characterization of TV distance, strong detection is (information-theoretically) possible if and only if  $\liminf_{s\to\infty} \text{TV}(p_s,q_s)=1$ , and weak detection is (information-theoretically) possible if and only if  $\liminf_{s\to\infty} \text{TV}(p_s,q_s)>0$ .

Now we consider the following Bayesian inference problem, given by a joint distribution  $\pi$  over  $(\theta, z) \in \mathbb{R}^n \times \mathbb{R}^m$ . Nature samples unknown signal  $\theta \in \mathbb{R}^n$  from  $\pi_{\theta}$ . Given sample size N, we receive observations  $\{z_i\}_{i=1}^N$  drawn i.i.d. from  $\pi_{z|\theta}$ ; the goal is to infer  $\theta$  from these observations. Let  $\pi^{(N)}$  denote the distribution over  $\{z_i\}_{i=1}^N$ , the mixture of product measures parametrized by  $\theta$ .

**Definition 6.** Let  $(\pi_s)$  be a sequence of inference tasks over  $\mathbb{R}^{n_s} \times \mathbb{R}^{m_s}$  and  $(\pi^{\text{null}_s})$  be a sequence of distributions over  $\mathbb{R}^{m_s}$ .  $(\pi_s)$  exhibits an all-or-nothing phase transition at threshold  $(N_s)$  with respect to null models  $(\pi^{\text{null}_s})$  if:

- For any  $\beta < 1$ : weak detection between  $(\pi^{(\beta N_s)})$  and  $((\pi^{\text{null}_s})^{\otimes \beta N_s})$  is information-theoretically impossible
- For any  $\beta > 1$ : strong detection between the planted model  $(\pi^{(\beta N_s)})$  and the null model  $((\pi^{\text{null}_s})^{\otimes \beta N_s})$  is information-theoretically possible

All-or-nothing phase transitions have been established for a number of natural inference tasks like sparse linear regression [RXZ19, GZ19], sparse PCA [NWZ20], generalized linear models [BMR20], group testing [TAS21, COGHK<sup>+</sup>22], linear and phase retrieval models [SC16, TS20], planted subgraphs [MNWS<sup>+</sup>23], and planted Gaussian perceptron [NWZ23]. Here is an example for sparse linear regression:

**Theorem 5** ([RXZ19]). Let  $\pi_s$  be the distribution over  $\mathbb{R}^{n_s} \times \mathbb{R}^{m_s}$  for  $n_s = s$  and  $m_s = s+1$  where the marginal over  $\theta$  is given by the uniform distribution over  $k_s$ -sparse vectors in  $\{0,1\}^s$ , and the conditional distribution  $\pi_{z|\theta}$  is given by sampling  $x \sim \mathcal{N}(0, \mathrm{Id}_s)$ , taking  $y = \langle \theta, x \rangle + \xi$  for  $\xi \sim \mathcal{N}(0, \sigma_s^2)$ , and outputting observation z = (x, y). The null model  $\pi_s^{\mathrm{null}}$  is given by sampling  $x \sim \mathcal{N}(0, \mathrm{Id}_s)$  and outputting  $y = \mathcal{N}(0, k_s + \sigma_s^2)$ . If  $\sigma_s^2 \ll k_s \leq s^{0.499}$ , then  $(\pi_s)$  exhibits an all-or-nothing phase transition at threshold  $(N_s^*)$  with respect to null models  $(\pi_s^{\mathrm{null}})$  for  $N_s^* \triangleq \frac{2k_s \log(s/k_s)}{\log(1+k_s/\sigma_s^2)}$ .

Having defined the all-or-nothing phenomenon, we rigorously instantiate it as a critical window for in-context learning. We first define a mixture model  $p_{(N)}^{\Theta}$  for sequence lengths N onto which we will identify a critical window.

**Definition 7.** To any inference task  $\pi$ , null model  $\pi^{\text{null}}$ , and sequence length N, we can associate the following in-context learning task. Let  $\Theta = \Theta_{\text{signal}} \sqcup \{\text{NULL}\}$  where  $\Theta_{\text{signal}} \triangleq \text{supp}(\pi_{\theta})$ . Given  $\theta \in \text{supp}(\pi_{\theta})$ , let  $p_{(N)}^{\theta}$  denote the distribution over sequences  $(z_1, \ldots, z_N, ?, \theta)$  where  $z_1, \ldots, z_N$  are i.i.d. samples from  $p_{z|\theta}$ . Let  $p_{(N)}^{\text{null}}$  denote the distribution over observations  $(z_1, \ldots, z_N, ?, \text{NULL})$  where  $z_1, \ldots, z_N$  are i.i.d. samples from  $\pi^{\text{null}}$ . We then take  $p_{(N)}^{\Theta} \triangleq \mathbb{E}_{\theta \sim \frac{1}{2}\pi_{\theta} + \frac{1}{2}\delta_{\text{NULL}}} p_{(N)}^{\theta}$ .

Under this model of data, we have the following theorem expressing the all-or-nothing phase transition in terms of  $T_{\text{before}}^C$ ,  $T_{\text{after}}^C$ .

<span id="page-14-0"></span>**Theorem 6.** Suppose  $(\pi_s)$  is a sequence of inference tasks that exhibits an all-or-nothing phase transition at threshold  $(N_s^*)$  with respect to null models  $(\pi_s)$ . Given  $N_s \geq N_s^*$ , let  $(p_{(N_s)}^{s;\theta})_{\theta \in \Theta_s}$  denote the sequence of in-context learning tasks. For any constant  $0 < \epsilon < 1$ , there exist constants  $\delta,\underline{s}$  such that for all  $s \geq \underline{s}$ , next-token prediction for  $(p_{(N_s)}^{s;\theta})_{\theta \in \Theta_s}$  exhibits a critical window over  $[N_s + 2 - (1 + \delta)N_s^*, N_s + 2 - (1 - \delta)N_s^*]$  in which we transition from sampling a distribution  $O(\epsilon)$ -close in TV to  $S_{\text{before}} = \Theta_s$ ; signal, to sampling from a distribution  $O(\epsilon)$ -close in TV to  $S_{\text{after}} = \Theta_s$ .

a distribution  $O(\epsilon)$ -close in TV to  $S_{\text{after}} = \Theta_s$ . In other words, we have  $T_{\text{before}}^C \triangleq N + 2 - (1 - \delta)N_s^*$  and  $T_{\text{after}}^C \triangleq N + 2 - (1 + \delta)N_s^*$ .

The proof of Theorem 6 is essentially immediate from Theorem 2 and the definition of the all-ornothing phase transition:

*Proof.* Let us first apply Theorem 2 to  $S_{\text{init}} = S_{\text{targ}} = \Theta_{s;\text{signal}}$ . By the definition of  $D_{\Theta}$ , the parameter W therein is 1. Furthermore, we trivially have that  $T_{\text{end}}(\epsilon) = 0$ . Finally, because strong detection is possible provided there are  $N \geq \beta N_s^*$  in-context examples for  $\beta > 1$ , there exists  $\delta_1$  depending only on  $\epsilon$  for which  $\text{TV}(p_t^{S_{\text{targ}}}, p_t^{\Theta_s - S_{\text{targ}}}) \geq 1 - \epsilon^2$  for  $t = N_s + 2 - (1 + \delta_1)N_s^*$ . By Theorem 2 we conclude that  $\text{TV}(p_{\text{init}}, N_s + 2 - (1 + \delta_1)N_s^*) \lesssim \epsilon$ . Next, let us apply Theorem 2 to  $S_{\text{init}} = \Theta_{s;\text{signal}}$  and  $S_{\text{targ}} = \Theta_s$ . The parameter W therein is now 0. Furthermore, we trivially have that  $T_{\text{end}}(\epsilon) = N_s + 2$ . Finally, because weak detection is impossible provided there are  $N \leq \beta N_s^*$  in-context examples for  $\beta < 1$ , there exists  $\delta_2$  depending only on  $\epsilon$  for which  $\text{TV}(p_t^{S_{\text{init}}}, p_t^{S_{\text{targ}}}) \leq \epsilon$  for  $t = N_s + 2 - (1 - \delta_2)N_s^*$ . By Theorem 2 we conclude that  $\text{TV}(p_t^{S_{\text{init}}}, N_s + 2 - (1 - \delta_2)N_s^*) \leq \epsilon$ . Taking  $\delta = \max(\delta_1, \delta_2)$  concludes the proof.

## <span id="page-15-0"></span>5 Hierarchies in stochastic localization samplers

Herein we propose a theory of hierarchical sampling within our critical windows framework. It is motivated by the observation that a single trajectory can contain multiple critical windows (Figure 4), each splitting a sub-population into smaller sub-populations. This hierarchy is naturally represented as a tree: the root signifies that all sub-populations are indistinguishable under enough noise, while the leafs represent distinct modes in p. A path from the root to a leaf captures the progressive refinement of the original distribution p into increasingly specific components. To formalize this, we introduce the concept of an  $\epsilon$ -mixture tree, which decomposes p into a hierarchical structure.

<span id="page-15-1"></span>**Definition 8.** For an error term  $\epsilon > 0$  and mixture model p, an  $\epsilon$ -mixture tree is a tuple

$$(T, \{P^{\rightarrow}(\cdot|\cdot)\}, \mathbf{I}, \Theta, \{p^{\theta}\}_{\theta \in \Theta}, Subset, NoiseAmount).$$

The tree T=(V,E) is associated with a function Subset:  $V\to 2^\Theta\backslash\{\emptyset\}$ , which maps vertices to sub-mixtures. We require Subset satisfies the following two properties: (1) Subset(root) =  $\Theta$ ; (2) If u is a parent of v, Subset(v)  $\subset$  Subset(u). We consider a NoiseAmount:  $V\to\mathbb{R}^{\geq 0}$ , which characterizes the noise levels that result in the aggregations of mixture components described by vertices in the mixture tree. NoiseAmount(u) is defined such that all  $p_{NoiseAmount(u)}^{\theta}$  for  $\theta \in Subset(u)$  overlap greatly and for  $p_{NoiseAmount(u)}^{Subset(u)}$ ,  $p_{NoiseAmount(u)}^{\Theta-Subset(u)}$  have negligible overlap. Thus we require that NoiseAmount satisfy three properties: (1) For distinct  $\theta_i, \theta_j \in \Theta$  with leaf nodes w, v such that  $\theta_i \in Subset(w)$ ,  $\theta_j \in Subset(v)$ , if u is the lowest common ancestor of w, v, then we require  $TV(p_{NoiseAmount(u)}^{\theta_i}, p_{NoiseAmount(u)}^{\theta_j}) \leq \varepsilon$ ; (2) For  $u \in V$ , we have statistical separation between Subset(u) and  $\Theta - Subset(u)$  in terms of TV,  $TV(p_{NoiseAmount(u)}^{Subset(u)}, p_{NoiseAmount(u)}^{\Theta-Subset(u)}) \geq 1 - \varepsilon^2$ ; and (3) If  $v \in V$  is a parent of u, we have NoiseAmount(u) < NoiseAmount(v). Property 1 establishes bounds on  $T_{end}^S$ , and properties 2 and 3 establishes bounds on  $T_{start}^S$ .

We emphasize that this framework is highly general, solely defined with the *initial distribution* p and the forward process. It strictly expands the definition in [LC24], which focused on hierarchies of isotropic Gaussians, to all localization-based samplers and mixture models. We can also relate it to the sequences of critical windows we observe in Figure 4, capturing the idea that each critical window represents the refinement into smaller subpopulations of p.

**Corollary 1.** Consider an  $\epsilon$ -mixture tree. For  $\theta_i \in \Theta$ , consider the path  $u_1, u_2, u_3, \ldots, u_{H'} \in V$  where  $u_1$  is the leaf node with  $\theta_i \in Subset(u_1)$  and  $u_{H'}$  is the root. There is a sequence of times  $T_1 < T_2 < \cdots < T_{H'}$  with  $TV(p^{\{i\},T_\ell},p^{Subset(u_\ell)}) \lesssim_w \epsilon$ .

*Proof.* For  $\ell \in [H']$ , we let  $T_\ell = \text{NoiseAmount}(u_\ell)$ . We apply Theorem 2 with  $S_{\text{init}} = \{i\}$  and  $S_{\text{target}} = \text{Subset}(u_\ell)$ . We know  $\text{TV}(p_{T_\ell}^{S_{\text{target}}}, p_{T_\ell}^{\Theta - S_{\text{target}}}) \geq 1 - \epsilon^2$  by Condition 2 in Definition 8. By Lemma 15 of [LC24], we know  $\text{TV}(p_{T_\ell}^{\{i\}}, p_{T_\ell}^{S_{\text{init}}}) \leq \max_{j \in S_{\text{init}}} \text{TV}(p_{T_\ell}^{\{i\}}, p_{T_\ell}^{\{j\}})$ . This is ≤  $\epsilon$  for all  $j \in S_{\text{init}}$  by Condition 3 on NoiseAmount and the data processing inequality. □

We first observe that the hierarchy of two samplers with the same forward process are identical if the samplers agree on sub-populations. Assume we have  $\{p^{\theta}\}_{\theta \in \Theta}$  (e.g. the true distribution) and  $\{q^{\theta}\}_{\theta \in \Theta}$  (e.g. a generative model), where  $q^{\theta} \approx p^{\theta}$  across all  $\theta \in \Theta$  with the same  $\{w_{\theta}\}_{\theta \in \Theta}$ .

**Corollary 2.** Consider an  $\epsilon$ -mixture tree  $(T, \{P^{\neg}(\cdot|\cdot)\}, \mathbf{I}, \Theta, \{p^{\theta}\}_{\theta \in \Theta}, Subset, NoiseAmount)$ . Suppose we have another distribution  $\{q^{\theta}\}_{\theta \in \Theta}$  such that  $\mathrm{TV}(p^{\theta}, q^{\theta}) \leq \delta/2$  for all  $\theta \in \Theta$ . Then we have  $\epsilon + \sqrt{\delta}$ -mixture tree given by  $(T, \{P^{\neg}(\cdot|\cdot)\}, \mathbf{I}, \Theta, \{q^{\theta}\}_{\theta \in \Theta}, Subset, NoiseAmount)$ .

*Proof.* We need only check the first and second properties of NoiseAmount with parameter  $\epsilon + \sqrt{\delta}$ . To do this, it suffices to show  $\mathrm{TV}(q_{\mathrm{NoiseAmount}(u)}^{\theta_i}, q_{\mathrm{NoiseAmount}(v)}^{\theta_j}) \leq \epsilon + \delta$  and  $\mathrm{TV}(q_{\mathrm{NoiseAmount}(u)}^{\mathrm{Subset}(u)}, q_{\mathrm{NoiseAmount}(u)}^{\Theta-\mathrm{Subset}(u)}) \geq 1 - \epsilon^2 - \delta$ . By the data processing inequality, we just need to show this at t = 0, and we prove the stronger statement that for  $S_1 \subset \Theta$ ,  $\mathrm{TV}(p^{S_1}, q^{S_1}) \leq \delta/2$ . This follows from Lemma 15 of [LC24] and  $\mathrm{TV}(p^{\theta}, q^{\theta}) \leq \delta/2$  for all  $\theta \in \Theta$ .

This similarity does not hold generally, if the generative model does not have the same forward process the data generating procedure. In fact, we can define arbitrary hierarchies by designing an appropriate forward process.

<span id="page-16-2"></span>**Example 7.** Consider a set of alphabets  $\{A_i\}_{i=1}^d$  and define  $\Theta = \{(a_i)_{i=1}^d : \forall i \in [d], a_i \in A_i\}$  and  $p^{\theta_i} = \delta_{\theta_i}$ . Let  $\mathbf{I} = [0, 1, 2, \ldots, d]$ . and for any permutation  $i_1, i_2, \ldots, i_d$  of [d], define a forward process such that at  $t \in \mathbf{I}$ , we mask all  $i_d, i_{d-1}, \ldots, i_{d-t}$ . This constructs a hierarchy where the values for  $i_1, i_2, \ldots, i_d$  are decided in that order.

*Proof.* We construct the following 0-mixture tree as follows. We let the leaf nodes be the set  $\Theta$ . We let two leaf nodes u, v have the same parent if and only if they share the same values on the alphabet at  $i_1, i_2, \ldots, i_{d-1}$ ; we also define the parent as the union of all of its children. We now treat the parents we constructed as the roots, and let them have the same parent if and only if they share the same values on the tuple  $i_1, i_2, \ldots, i_{d-2}$ . We continue to do this until we are left with one root node. We let Subset map each node to the corresponding set and NoiseAmount map each node to its distance from a leaf node.

By the construction of T, it is clear that Subset satisfies the desired properties. For distinct  $\theta_i, \theta_j \in \Theta$ , the lowest common ancestor of  $\theta_i, \theta_j$  represents the largest k such that indices  $i_1, \ldots, i_k$  are the same for  $\theta_i, \theta_j$ . Because  $p_{\text{NoiseAmount}(u)}^{\theta_i}$  is just the tuple of the values of  $\theta_i, \theta_j$  at  $i_1, \ldots, i_k$ , we know  $\text{TV}(p_{\text{NoiseAmount}(u)}^{\theta_i}, p_{\text{NoiseAmount}(u)}^{\theta_j}) = 0$ . For any  $u \in V$  representing the values at index  $(i_\ell)_{\ell=1}^k$ , all  $\theta \notin \text{Subset}(u)$  does not share the same values at these indices by definition, so we also know

$$\mathrm{TV}(p_{\mathrm{NoiseAmount}(u)}^{\mathrm{Subset}(u)}, p_{\mathrm{NoiseAmount}(u)}^{\Theta-\mathrm{Subset}(u)}) = 1.$$

Finally, we note that hierarchies of diffusions are generally shallower than hierarchies for autoregressive models. The hierarchy for a mixture of Gaussians cannot grow linearly with the dimension d, e.g. it is O(1) in Example 2 for mixtures of Gaussians with randomly selected means or  $O(\sqrt{\log d})$  in the hierarchy of Gaussians in [LC24]. This is because the forward process simultaneously contracts all distances with the same dependence on d together at the same time. However, in contrast, depth can scale linearly with the context length for autoregressive models, (Example 7 or Figure 4). We speculate that this could mean autoregressive models can learn more complex feature hierarchies than diffusions.

## <span id="page-16-0"></span>6 Experiments

As many authors [HJA20, MHS<sup>+</sup>22, CLS<sup>+</sup>22, RA23, GVS<sup>+</sup>23, SFLW24, SFW25, BBdBM24, LC24] have already empirically studied critical windows in the context of diffusion, we focus on experiments on critical windows for LLMs. In Section 6.1, we validate our theory on outputs with a hierarchical structure, showing strong agreement with Section 5. In Section 6.2, we probe critical windows for LLAMA-3.1-8B-Instruct, Phi-3-7B-Instruct, and Qwen-2.5-7B-Instruct in real-world reasoning benchmarks.

## <span id="page-16-1"></span>6.1 Structured output experiments

To verify our theory for  $T_{\mathrm{before}}^{C}$ ,  $T_{\mathrm{after}}^{C}$ , we have to compute the total variation between truncated responses from an LLM. This usually would take a large number of samples, so to circumvent this issue, we restrict

the diversity of the LLM's generations and force the LLM to generate tokens in a structured format. In particular, we have LLAMA-3.1-8B-Instruct<sup>7</sup> respond to following prompt, which asks it to answer a series of fill-in-the-blank questions in a structured format. We also prefill the model's generations with \n\n 1. to ensure that the outputs comport to this format. To compute  $T_{\rm before}^C$ ,  $T_{\rm after}^C$ , we look at when the generations diverge based on the first occurrence of the identifying information. For example, the  $T_{\rm before}^C$  of the first critical window is 1. The , because the first answer has not appeared in the generation, and the  $T_{\rm after}^C$  of the first critical window is 1. The P or 1. The N, because that uniquely identifies the answer. Figure 4 plots the probability of obtaining the same answers as the original generation after truncating different amounts from the generation in the forward-reverse experiments, computed with 10,000 generations. Our theory predicts that jumps in the probability will occur at  $T_{\rm before}^C$ ,  $T_{\rm after}^C$  which represent when the model has committed to a particular answer to a question in the generation. These predictions are validated with our experiments, as the jumps in probability, representing the model localizing to a more specific set of answers, occur exactly at  $T_{\rm before}^C$ ,  $T_{\rm after}^C$ 

## **Structured Output Prompt**

Complete the following by choosing only one option for each blank. The options are provided in parentheses, and your response must match the exact case and meaning of the chosen option. Respond with only the completed sentence, no explanations or additional text.

- 1. The (Pirate/Ninja) jumped across the ship.
- 2. She adopted a (Dog/Cat) from the shelter.
- 3. The (River/Bridge) sparkled under the sun.
- 4. A (Dragon/Knight) guarded the castle gates.
- 5. He ordered (Pizza/Sushi) for dinner.

<span id="page-17-1"></span>![](_page_17_Figure_8.jpeg)

![](_page_17_Figure_9.jpeg)

Figure 4: Structured output plots for LLAMA-3.1-8B-Instruct.

## <span id="page-17-0"></span>6.2 Chain of thought experiments

We then identify critical windows for LLAMA-3.1-8B-Instruct, Phi-3-7B-Instruct, and Qwen-2.5-7B-Instruct on 7 different math and reasoning benchmarks on which performance is known to improve with chain of thought reasoning [LCR<sup>+</sup>23]: ARC Challenge and Easy [CCE<sup>+</sup>18], AQua [LYDB17], LogiQA [LCL<sup>+</sup>20], MMLU [HBB<sup>+</sup>21], and TruthfulQA [LHE22] multiple-choice benchmarks and the MATH benchmark from [HBK<sup>+</sup>21].<sup>8</sup> In the forward-reverse experiments, we take the original generation, truncate a fraction of tokens, and check if resampling yields the same answer, using a direct text comparison for the multiple choice benchmarks and the prm800k grader for MATH [LKB<sup>+</sup>23]. We do this for 400 questions from each dataset and resample at each truncation fraction 100 times. Critical windows,

<span id="page-17-3"></span><span id="page-17-2"></span><sup>&</sup>lt;sup>7</sup>Default sampling parameters of temperature of 0.6 and top-p sampling of 0.9

<sup>&</sup>lt;sup>8</sup>See Appendix D.2 for more results across models and datasets and a discussion on the effect of temperature on critical windows.

defined as a *>* 0.5 jump in probability of obtaining the same answer in consecutive truncation fractions, appear prominently across all models and benchmarks that we tested (Figures [5,](#page-18-0) [9,](#page-39-0) and [10\)](#page-40-0); for MATH, they occur in 42.2%, 16.6%, 30.2% of generations from LLAMA-3.1-8B-Instruct, Qwen-2.5-7B-Instruct, and Phi-3-7B-Instruct.

<span id="page-18-0"></span>![](_page_18_Figure_1.jpeg)

Figure 5: Probability that the answer is the same as a function of the percentage of the generation remaining for different math and reasoning benchmarks for LLAMA-3.1-8B-Instruct. Highlighted are generations with a 50% increase and no 30% decrease in the probability over one step.

These jumps occur during important steps of reasoning: in Figure [6,](#page-19-1) the correct formula is first expressed in the critical window. Furthermore, we find that generations with critical windows are significantly *less accurate* than generations without critical windows across all datasets and models. For LLAMA-3.1-8B-Instruct, critical windows result in up to **73% lower accuracy**, and incorrect responses contain **11%-33% more critical windows** (Table [1\)](#page-18-1).

<span id="page-18-1"></span>Table 1: Differences between Accuracy (Acc) without versus with critical windows and frequency of critical windows (CW) when the original generation is wrong versus right.

| Dataset       | LLAMA-3.1-8B-Instruct |      | Phi-3-7B-Instruct |      | Qwen-2.5-7B-Instruct |      |
|---------------|-----------------------|------|-------------------|------|----------------------|------|
|               | ∆Acc                  | ∆CW  | ∆Acc              | ∆CW  | ∆Acc                 | ∆CW  |
| AQUA-RAT      | 0.42                  | 0.20 | 0.36              | 0.16 | 0.03                 | 0.01 |
| ARC Challenge | 0.53                  | 0.22 | 0.50              | 0.24 | 0.38                 | 0.11 |
| ARC Easy      | 0.73                  | 0.26 | 0.28              | 0.13 | 0.40                 | 0.07 |
| LogiQA        | 0.15                  | 0.11 | 0.21              | 0.19 | 0.23                 | 0.11 |
| MATH          | 0.41                  | 0.33 | 0.36              | 0.33 | 0.46                 | 0.29 |
| MMLU          | 0.45                  | 0.24 | 0.38              | 0.21 | 0.26                 | 0.11 |
| TruthfulQA    | 0.44                  | 0.20 | 0.42              | 0.23 | 0.55                 | 0.23 |

#### <span id="page-19-1"></span>MATH (Phi-3-7B-Instruct)

**Q**: If the first term of an arithmetic sequence equals the third minus the second, and the fourth term is 8, what is the first term?

A: 1. We know that the arithmetic sequence has a common difference, which we will denote as "d". 2. Let's denote the terms in the sequence as  $T_1$ ,  $T_2$ ,  $T_3$ , and  $T_4$  for the first, second, third, and fourth terms, respectively. 3. We're given the information that  $T_1 = T_3 - T_2$ . Using the general formula for an arithmetic sequence, we can write this **as**  $T_1 = T_2 + 2d - T_2$ , **which simplifies to**  $T_1 = 2d$ . **4. We're also given that**  $T_4 = 8$ . **The general formula for an arithmetic sequence tells us that**  $T_4 = T_1 + 3d$ . **So we can write**  $8 = T_1 + 3d$ . 5. Now we have a system of two equations: 1)  $T_1 = 2d$  2)  $8 = T_1 + 3d$  6. We can substitute equation (1) into equation (2), which gives us 8 = 2d + 3d = 5d. 7. Solving for d, we get d = 8/5. 8. Now that we know d, we can find  $T_1$ . Using the equation from step 1, we get  $T_1 = 2d = 2*(8/5) = 16/5$ . Therefore, the first term is  $T_1 = 16/5$ .

![](_page_19_Figure_3.jpeg)

Figure 6: Example of critical window for Phi-3-7B-Instruct (bolded).

## <span id="page-19-0"></span>7 Discussion

In this work, we developed a simple yet general theory for critical windows for stochastic localization samplers like diffusion and autoregressive models. Already, practitioners have applied critical windows to make LLMs safer [QPL+24] and reason better [AAB+24, LLX+24]. Our theory significantly streamlines our understanding of critical windows and provides concrete insights for practitioners. We pair our theory with extensive experiments, and demonstrate its usefulness in monitoring for jailbreaks and understanding reasoning failures.

**Limitations.** The theory applies to the most prominent and empirically successful generative models (autoregressive language models, continuous diffusion and flow matching models). However, some less widely-used generative models do not belong to the family of localization-based samplers. Generative adversarial networks [GPAM+14] and consistency models [SDCS23] both use a singular evaluation of a neural network to map noise into an image. However, we argue the restriction to the localization-based samplers is extremely minor because these models are either not widely used in practice or based on localization-based samplers.

**Future Work.** While we instantiate our theory in many settings (Section 4), an important future direction is compute the location of critical windows in other settings and connect it to statistical physics predictions for other models of data, e.g., the random hierarchy model [SFLW24].

**Impact statement.** Though this paper is largely theoretical in nature, it does describe a theory for jailbreaks which could impact model safety in the future. We hope the insights in this manuscript about jailbreaks lead to better alignment strategies and training methods.

## **Acknowledgments**

ML would like to thank Cynthia Dwork for regular meetings and kind mentorship which were of great help throughout this research project, Cem Anil, Michael Li, and Eric Zelikman for insightful conversations which inspired some of the experiments, and Max Simchowitz, Michael Albergo, Seth Neel, and Sahil Kuchlous for thoughtful discussions regarding the framing of this work.

## **Code**

The code to reproduce the experiments can be found in [https://github.com/marvinli-harvard/](https://github.com/marvinli-harvard/critical-windows-lm) [critical-windows-lm](https://github.com/marvinli-harvard/critical-windows-lm).

## **References**

- <span id="page-20-1"></span>[AAB+24] Marah Abdin, Jyoti Aneja, Harkirat Behl, Sébastien Bubeck, Ronen Eldan, Suriya Gunasekar, Michael Harrison, Russell J. Hewett, Mojan Javaheripi, Piero Kauffmann, James R. Lee, Yin Tat Lee, Yuanzhi Li, Weishung Liu, Caio C. T. Mendes, Anh Nguyen, Eric Price, Gustavo de Rosa, Olli Saarikivi, Adil Salim, Shital Shah, Xin Wang, Rachel Ward, Yue Wu, Dingli Yu, Cyril Zhang, and Yi Zhang. Phi-4 technical report, 2024.
- <span id="page-20-8"></span>[ADS+24] Cem Anil, Esin Durmus, Mrinank Sharma, Joe Benton, Sandipan Kundu, Joshua Batson, Nina Rimsky, Meg Tong, Jesse Mu, Daniel Ford, Francesco Mosconi, Rajashree Agrawal, Rylan Schaeffer, Naomi Bashkansky, Samuel Svenningsen, Mike Lambert, Ansh Radhakrishnan, Carson Denison, Evan J Hubinger, Yuntao Bai, Trenton Bricken, Timothy Maxwell, Nicholas Schiefer, Jamie Sully, Alex Tamkin, Tamera Lanham, Karina Nguyen, Tomasz Korbak, Jared Kaplan, Deep Ganguli, Samuel R. Bowman, Ethan Perez, Roger Grosse, and David Duvenaud. Many-shot jailbreaking. In *Advances in the Thirty-Eighth Annual Conference on Neural Information Processing Systems*, 2024.
- <span id="page-20-7"></span><span id="page-20-6"></span>[AJPG24] Aryaman Arora, Dan Jurafsky, Christopher Potts, and Noah D. Goodman. Bayesian scaling laws for in-context learning, 2024.
  - [AK23] Gabriel Alon and Michael Kamfonas. Detecting language model attacks with perplexity, 2023.
- <span id="page-20-4"></span>[ALL+19] Sanjeev Arora, Yuanzhi Li, Yingyu Liang, Tengyu Ma, and Andrej Risteski. A latent variable model approach to pmi-based word embeddings, 2019.
- <span id="page-20-3"></span>[AMS23] Ahmed El Alaoui, Andrea Montanari, and Mark Sellke. Sampling from mean-field gibbs measures via diffusion processes. *arXiv preprint arXiv:2310.08912*, 2023.
- <span id="page-20-0"></span>[Ant24] Anthropic. Developing a computer use model. [https://www.anthropic.com/news/](https://www.anthropic.com/news/developing-computer-use) [developing-computer-use](https://www.anthropic.com/news/developing-computer-use), 2024.
- <span id="page-20-5"></span>[ASA+23] Ekin Akyürek, Dale Schuurmans, Jacob Andreas, Tengyu Ma, and Denny Zhou. What learning algorithm is in-context learning? investigations with linear models, 2023.
- <span id="page-20-2"></span>[BBdBM24] Giulio Biroli, Tony Bonnaire, Valentin de Bortoli, and Marc Mézard. Dynamical regimes of diffusion models, 2024.

- <span id="page-21-6"></span>[BMR20] Jean Barbier, Nicolas Macris, and Cynthia Rush. All-or-nothing statistical and computational phase transitions in sparse spiked matrix estimation, 2020.
- <span id="page-21-10"></span>[BSS+24] Luke Bailey, Alex Serrano, Abhay Sheshadri, Mikhail Seleznyov, Jordan Taylor, Erik Jenner, Jacob Hilton, Stephen Casper, Carlos Guestrin, and Scott Emmons. Obfuscated activations bypass llm latent-space defenses, 2024.
- <span id="page-21-8"></span>[CCE+18] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint 1803.05457, 2018.
- <span id="page-21-2"></span>[CCL+23] Sitan Chen, Sinho Chewi, Jerry Li, Yuanzhi Li, Adil Salim, and Anru Zhang. Sampling is as easy as learning the score: theory for diffusion models with minimal data assumptions. In *The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023*. OpenReview.net, 2023.
  - [CE22] Yuansi Chen and Ronen Eldan. Localization schemes: A framework for proving mixing bounds for markov chains. In *2022 IEEE 63rd Annual Symposium on Foundations of Computer Science (FOCS)*, pages 110–122. IEEE, 2022.
- <span id="page-21-1"></span><span id="page-21-0"></span>[CLS+22] Jooyoung Choi, Jungbeom Lee, Chaehun Shin, Sungwon Kim, Hyunwoo Kim, and Sungroh Yoon. Perception prioritized training of diffusion models. In *2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 11462–11471, 2022.
- <span id="page-21-12"></span><span id="page-21-11"></span><span id="page-21-9"></span><span id="page-21-7"></span><span id="page-21-5"></span><span id="page-21-4"></span><span id="page-21-3"></span>[COGHK+22] Amin Coja-Oghlan, Oliver Gebhard, Max Hahn-Klimroth, Alexander S Wein, and Ilias Zadik. Statistical and computational phase transitions in group testing. In Po-Ling Loh and Maxim Raginsky, editors, *Proceedings of Thirty Fifth Conference on Learning Theory*, volume 178 of *Proceedings of Machine Learning Research*, pages 4764–4781. PMLR, 02–05 Jul 2022.
  - [CRD+23] Patrick Chao, Alexander Robey, Edgar Dobriban, Hamed Hassani, George J. Pappas, and Eric Wong. Jailbreaking black box large language models in twenty queries, 2023.
  - [DCX+23] Ning Ding, Yulin Chen, Bokai Xu, Yujia Qin, Zhi Zheng, Shengding Hu, Zhiyuan Liu, Maosong Sun, and Bowen Zhou. Enhancing chat language models by scaling high-quality instructional conversations, 2023.
  - [DLD+24] Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Jingyuan Ma, Rui Li, Heming Xia, Jingjing Xu, Zhiyong Wu, Tianyu Liu, Baobao Chang, Xu Sun, Lei Li, and Zhifang Sui. A survey on in-context learning, 2024.
  - [EAMS22] Ahmed El Alaoui, Andrea Montanari, and Mark Sellke. Sampling from the sherringtonkirkpatrick gibbs measure via algorithmic stochastic localization. In *2022 IEEE 63rd Annual Symposium on Foundations of Computer Science (FOCS)*, pages 323–334. IEEE, 2022.
    - [Eld13] Ronen Eldan. Thin shell implies spectral gap up to polylog via a stochastic localization scheme. *Geometric and Functional Analysis*, 23(2):532–569, 2013.
    - [Eld20] Ronen Eldan. Taming correlations through entropy-efficient measure decompositions with applications to mean-field approximation. *Probability Theory and Related Fields*, 176(3-4):737–755, 2020.

- <span id="page-22-0"></span>[Gem24] Gemini. Challenges and solutions for aging adults. [https://gemini.google.com/](https://gemini.google.com/share/6d141b742a13) [share/6d141b742a13](https://gemini.google.com/share/6d141b742a13), 2024.
- <span id="page-22-14"></span><span id="page-22-13"></span><span id="page-22-12"></span><span id="page-22-11"></span><span id="page-22-10"></span><span id="page-22-9"></span><span id="page-22-8"></span><span id="page-22-7"></span><span id="page-22-6"></span><span id="page-22-5"></span><span id="page-22-4"></span><span id="page-22-3"></span><span id="page-22-2"></span><span id="page-22-1"></span>[GPAM+14] Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial networks, 2014.
  - [gri24] grimjim. Jailbroken llama-3.1-8b-instruct via lora. [https://huggingface.co/](https://huggingface.co/grimjim/Llama-3.1-8B-Instruct-abliterated_via_adapter) [grimjim/Llama-3.1-8B-Instruct-abliterated\\_via\\_adapter](https://huggingface.co/grimjim/Llama-3.1-8B-Instruct-abliterated_via_adapter), 2024.
  - [GTLV23] Shivam Garg, Dimitris Tsipras, Percy Liang, and Gregory Valiant. What can transformers learn in-context? a case study of simple function classes, 2023.
  - [GVS+23] Kristian Georgiev, Joshua Vendrow, Hadi Salman, Sung Min Park, and Aleksander Madry. The journey, not the destination: How data guides diffusion models. *arXiv preprint arXiv:2312.06205*, 2023.
    - [GZ19] David Gamarnik and Ilias Zadik. High-dimensional regression with binary coefficients. estimating squared error and a phase transition, 2019.
  - [Hai24a] Haize Labs. Automated multi-turn red-teaming with cascade. [https://blog.](https://blog.haizelabs.com/posts/cascade) [haizelabs.com/posts/cascade](https://blog.haizelabs.com/posts/cascade), 2024.
  - [Hai24b] Haize Labs. A trivial jailbreak against llama 3. [https://github.com/haizelabs/](https://github.com/haizelabs/llama3-jailbreak) [llama3-jailbreak](https://github.com/haizelabs/llama3-jailbreak), 2024.
  - [HBB+21] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. In *International Conference on Learning Representations*, 2021.
  - [HBK+21] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset, 2021.
    - [HJA20] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. *Advances in Neural Information Processing Systems*, 33:6840–6851, 2020.
  - [HMP24] Brice Huang, Andrea Montanari, and Huy Tuan Pham. Sampling from spherical spin glasses in total variation via algorithmic stochastic localization. *arXiv preprint arXiv:2404.15651*, 2024.
  - [HXH24] Luxi He, Mengzhou Xia, and Peter Henderson. What is in your safe data? identifying benign data that breaks safety, 2024.
    - [LC24] Marvin Li and Sitan Chen. Critical windows: non-asymptotic theory for feature emergence in diffusion models, 2024.
  - [LCL+20] Jian Liu, Leyang Cui, Hanmeng Liu, Dandan Huang, Yile Wang, and Yue Zhang. Logiqa: A challenge dataset for machine reading comprehension with logical reasoning. In Christian Bessiere, editor, *Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence, IJCAI-20*, pages 3622–3628. International Joint Conferences on Artificial Intelligence Organization, 7 2020. Main track.

- <span id="page-23-4"></span>[LCR+23] Tamera Lanham, Anna Chen, Ansh Radhakrishnan, Benoit Steiner, Carson Denison, Danny Hernandez, Dustin Li, Esin Durmus, Evan Hubinger, Jackson Kernion, Kamil˙e Lukošiut ¯ ˙e, Karina Nguyen, Newton Cheng, Nicholas Joseph, Nicholas Schiefer, Oliver Rausch, Robin Larson, Sam McCandlish, Sandipan Kundu, Saurav Kadavath, Shannon Yang, Thomas Henighan, Timothy Maxwell, Timothy Telleen-Lawton, Tristan Hume, Zac Hatfield-Dodds, Jared Kaplan, Jan Brauner, Samuel R. Bowman, and Ethan Perez. Measuring faithfulness in chain-of-thought reasoning, 2023.
- <span id="page-23-6"></span>[LHE22] Stephanie Lin, Jacob Hilton, and Owain Evans. TruthfulQA: Measuring how models mimic human falsehoods. In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 3214–3252, Dublin, Ireland, May 2022. Association for Computational Linguistics.
- <span id="page-23-11"></span>[LHS+24] Nathaniel Li, Ziwen Han, Ian Steneker, Willow Primack, Riley Goodside, Hugh Zhang, Zifan Wang, Cristina Menghini, and Summer Yue. Llm defenses are not robust to multi-turn human jailbreaks yet, 2024.
- <span id="page-23-7"></span>[LKB+23] Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let's verify step by step. *arXiv preprint arXiv:2305.20050*, 2023.
- <span id="page-23-0"></span>[LLX+24] Zicheng Lin, Tian Liang, Jiahao Xu, Xing Wang, Ruilin Luo, Chufan Shi, Siheng Li, Yujiu Yang, and Zhaopeng Tu. Critical tokens matter: Token-level contrastive estimation enhances llm's reasoning capability, 2024.
- <span id="page-23-9"></span>[LME24] Aaron Lou, Chenlin Meng, and Stefano Ermon. Discrete diffusion modeling by estimating the ratios of the data distribution, 2024.
- <span id="page-23-8"></span>[LRL+23] Bill Yuchen Lin, Abhilasha Ravichander, Ximing Lu, Nouha Dziri, Melanie Sclar, Khyathi Chandu, Chandra Bhagavatula, and Yejin Choi. The unlocking spell on base llms: Rethinking alignment via in-context learning, 2023.
- <span id="page-23-10"></span>[LXCX24] Xiaogeng Liu, Nan Xu, Muhao Chen, and Chaowei Xiao. Autodan: Generating stealthy jailbreak prompts on aligned large language models. In *The Twelfth International Conference on Learning Representations*, 2024.
- <span id="page-23-5"></span>[LYDB17] Wang Ling, Dani Yogatama, Chris Dyer, and Phil Blunsom. Program induction by rationale generation: Learning to solve and explain algebraic word problems. In *Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 158–167, Vancouver, Canada, July 2017. Association for Computational Linguistics.
- <span id="page-23-1"></span>[MHS+22] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. SDEdit: Guided image synthesis and editing with stochastic differential equations. In *International Conference on Learning Representations*, 2022.
- <span id="page-23-3"></span><span id="page-23-2"></span>[MNWS+23] Elchanan Mossel, Jonathan Niles-Weed, Youngtak Sohn, Nike Sun, and Ilias Zadik. Sharp thresholds in inference of planted subgraphs. In Gergely Neu and Lorenzo Rosasco, editors, *Proceedings of Thirty Sixth Conference on Learning Theory*, volume 195 of *Proceedings of Machine Learning Research*, pages 5573–5577. PMLR, 12–15 Jul 2023.
  - [Mon23a] Andrea Montanari. Sampling, diffusions, and stochastic localization, 2023.

- <span id="page-24-7"></span><span id="page-24-6"></span>[Mon23b] Andrea Montanari. Sampling, diffusions, and stochastic localization. *arXiv preprint arXiv:2305.10690*, 2023.
  - [MW23] Andrea Montanari and Yuchen Wu. Posterior sampling from the spiked models via diffusion processes. *arXiv preprint arXiv:2304.11449*, 2023.
- <span id="page-24-14"></span>[NCH+23] Milad Nasr, Nicholas Carlini, Jonathan Hayase, Matthew Jagielski, A. Feder Cooper, Daphne Ippolito, Christopher A. Choquette-Choo, Eric Wallace, Florian Tramèr, and Katherine Lee. Scalable extraction of training data from (production) language models, 2023.
- <span id="page-24-9"></span>[NWZ20] Jonathan Niles-Weed and Ilias Zadik. The all-or-nothing phenomenon in sparse tensor pca. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, *Advances in Neural Information Processing Systems*, volume 33, pages 17674–17684. Curran Associates, Inc., 2020.
- <span id="page-24-11"></span>[NWZ23] Jonathan Niles-Weed and Ilias Zadik. It was "all" for "nothing": sharp phase transitions for noiseless discrete channels, 2023.
- <span id="page-24-0"></span>[Ope25] OpenAI. Detecting misbehavior in frontier reasoning models. *OpenAI Research*, 2025.
- <span id="page-24-5"></span>[PCT+23] Leonardo Petrini, Francesco Cagnetta, Umberto M Tomasini, Alessandro Favero, and Matthieu Wyart. How deep neural networks learn compositional data: The random hierarchy model. *arXiv preprint arXiv:2307.02129*, 2023.
- <span id="page-24-2"></span><span id="page-24-1"></span>[QPL+24] Xiangyu Qi, Ashwinee Panda, Kaifeng Lyu, Xiao Ma, Subhrajit Roy, Ahmad Beirami, Prateek Mittal, and Peter Henderson. Safety alignment should be made more than just a few tokens deep, 2024.
  - [RA23] Gabriel Raya and Luca Ambrogioni. Spontaneous symmetry breaking in generative diffusion models. In *Thirty-seventh Conference on Neural Information Processing Systems*, 2023.
- <span id="page-24-15"></span>[RKV+24] Paul Röttger, Hannah Rose Kirk, Bertie Vidgen, Giuseppe Attanasio, Federico Bianchi, and Dirk Hovy. Xstest: A test suite for identifying exaggerated safety behaviours in large language models, 2024.
  - [Roo01] B. Roos. Binomial approximation to the poisson binomial distribution: The krawtchouk expansion. *Theory of Probability & Its Applications*, 45(2):258–272, 2001.
- <span id="page-24-13"></span><span id="page-24-10"></span><span id="page-24-8"></span>[RXZ19] Galen Reeves, Jiaming Xu, and Ilias Zadik. The all-or-nothing phenomenon in sparse linear regression, 2019.
  - [SC16] Jonathan Scarlett and Volkan Cevher. Limits on support recovery with probabilistic models: An information-theoretic framework, 2016.
- <span id="page-24-12"></span>[SDCS23] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models, 2023.
- <span id="page-24-3"></span>[SFLW24] Antonio Sclocchi, Alessandro Favero, Noam Itzhak Levi, and Matthieu Wyart. Probing the latent hierarchical structure of data via diffusion models, 2024.
- <span id="page-24-4"></span>[SFW25] Antonio Sclocchi, Alessandro Favero, and Matthieu Wyart. A phase transition in diffusion models reveals the hierarchical nature of data. *Proceedings of the National Academy of Sciences*, 122(1):e2408799121, 2025.

- <span id="page-25-8"></span>[SLB+24] Alexandra Souly, Qingyuan Lu, Dillon Bowen, Tu Trinh, Elvis Hsieh, Sana Pandey, Pieter Abbeel, Justin Svegliato, Scott Emmons, Olivia Watkins, and Sam Toyer. A strongreject for empty jailbreaks, 2024.
- <span id="page-25-0"></span>[TAS21] Lan V. Truong, Matthew Aldridge, and Jonathan Scarlett. On the all-or-nothing behavior of bernoulli group testing, 2021.
- <span id="page-25-6"></span><span id="page-25-1"></span>[TS20] Lan V. Truong and Jonathan Scarlett. Support recovery in the phase retrieval model: Information-theoretic fundamental limits, 2020.
  - [Ver] Roman Vershynin. *High-Dimensional Probability: An Introduction with Applications in Data Science*. Cambridge Series in Statistical and Probabilistic Mathematics. Cambridge University Press.
- <span id="page-25-7"></span>[vH16] Ramon van Handel. Probability in high dimension. 2016.
- <span id="page-25-11"></span>[WHS23] Alexander Wei, Nika Haghtalab, and Jacob Steinhardt. Jailbroken: How does llm safety training fail?, 2023.
- <span id="page-25-2"></span>[XRLM22] Sang Michael Xie, Aditi Raghunathan, Percy Liang, and Tengyu Ma. An explanation of in-context learning as implicit bayesian inference, 2022.
- <span id="page-25-9"></span>[ZPW+24] Andy Zou, Long Phan, Justin Wang, Derek Duenas, Maxwell Lin, Maksym Andriushchenko, Rowan Wang, Zico Kolter, Matt Fredrikson, and Dan Hendrycks. Improving alignment and robustness with circuit breakers, 2024.
  - [ZW24] Xiao Zhang and Ji Wu. Dissecting learning and forgetting in language model finetuning. In *The Twelfth International Conference on Learning Representations*, 2024.
- <span id="page-25-10"></span><span id="page-25-4"></span>[ZWC+23] Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr, J. Zico Kolter, and Matt Fredrikson. Universal and transferable adversarial attacks on aligned language models, 2023.
- <span id="page-25-5"></span>[ZZX+24] Zhengyue Zhao, Xiaoyun Zhang, Kaidi Xu, Xing Hu, Rui Zhang, Zidong Du, Qi Guo, and Yunji Chen. Adversarial contrastive decoding: Boosting safety alignment of large language models via opposite prompt optimization, 2024.
- <span id="page-25-3"></span>[ZZYW23] Yufeng Zhang, Fengzhuo Zhang, Zhuoran Yang, and Zhaoran Wang. What and how does incontext learning learn? bayesian model averaging, parameterization, and generalization, 2023.

## <span id="page-26-0"></span>**A Additional related work**

**Theory of in-context learning for language models.** With respect to theory for language models, our results are most closely related to the Bayesian framework for in-context learning [[XRLM22,](#page-25-2) [ASA](#page-20-5)+23, [GTLV23,](#page-22-11) [ZZYW23,](#page-25-3) [AJPG24](#page-20-6)]. For example, [[XRLM22](#page-25-2)] also considered a mixture model of topics and showed that language models can learn the underlying class despite in-context learning and training distribution mismatch. We view this manuscript as connecting the Bayesian framework for in-context learning to other empirical phenomena observed in language models and diffusion models and the all-or-nothing phenomenon.

**Chain of thought.** [[LLX](#page-23-0)+24, [AAB](#page-20-1)+24] also observed that the presence of critical windows in the chain of thought of math and reasoning tasks and their significance in leading the model to incorrect outputs, concurrent with our results in Figure [6.](#page-19-1) They then used them to provide rewards or data for a preference optimization algorithm to improve reasoning performance. [[LLX](#page-23-0)+24] called them *critical tokens* and utilized a contrastive estimation algorithm to identify critical windows and provide token-level rewards. The Phi-4 Technical report called them *pivotal tokens*, developed a binary-search based algorithm to identify the location of critical windows, and used them to produce contrasting pairs for preference optimization [[AAB](#page-20-1)+24]. Using our broad theoretical perspective, we provide new insight into critical windows of these kinds and view our work as corroborating and extending these empirical works.

**Jailbreaks.** Existing work on jailbreaks has studied the appearance of critical windows in the first few generated tokens [[QPL](#page-24-1)+24, [ZW24,](#page-25-4) [HXH24,](#page-22-12) [LRL](#page-23-8)+23]. Our theory provides a simple explanation for when jailbreaks occur: when the unaligned component assigns a much higher probability to the current text than the aligned component, then the model is jailbroken. This generalizes the explanation from [[QPL](#page-24-1)+24] (see Example [4](#page-12-1) for our particular formalism of their insights). It also explains the success of perplexity-based monitors for jailbreaks [[AK23](#page-20-7)], which monitor for a low probability of the context and generation. We view our work as providing a rigorous mathematical framework for jailbreaks, as well as highlighting the important role off-distribution contexts play in eliciting harmful behaviors; we also develop a novel jailbreak from our framework (Section [D.1.2](#page-37-0) ) similar to the adversarial contrast decoding method proposed by [[ZZX](#page-25-5)+24], which also uses a likelihood ratio between an unaligned and an aligned model. However, we use a jailbroken and non-jailbroken pair of models instead of two versions of the model with different prompts.

## <span id="page-26-1"></span>**B Examples of stochastic localization samplers**

In this section, we present several kinds of generative models within the stochastic localization framework and their forward and reverse processes.

**Example 8** (Continuous Diffusion Models [[LC24](#page-22-4)])**.** *For continuous diffusion models, the forward process progressively degrades samples X* ∼ *p into pure Gaussian noise through scaling and convolution with Gaussian noise. It is the Ornstein-Uhlenbeck process, a stochastic process* (*X<sup>t</sup>* )*t*≥<sup>0</sup> *given by the following stochastic differential equation (SDE),*

$$dX_t = -X_t dt + \sqrt{2}dB_t, \qquad X_0 \sim p,$$

*where* (*B<sup>t</sup>* )*t*≥0 *is a standard Brownian motion. Let q<sup>t</sup>* ≜ law(*X<sup>t</sup>* ) *for t* ≥ 0*, and observe that as t* → ∞*, q<sup>t</sup> converges exponentially quickly to the standard Gaussian distribution* N (0, Id)*. Assume we end the forward* process at time  $T \ge 0$ . For the reverse process  $(X_t^-)_{t \in [0,T]}$ , we employ the reversal of the Ornstein-Uhlenbeck SDE, given by

$$\mathrm{d}X_t^{\leftarrow} = \{X_t^{\leftarrow} + 2\nabla \ln q_{T-t}(X_t^{\leftarrow})\}\,\mathrm{d}t + \sqrt{2}\,\mathrm{d}B_t\,, \qquad X_T^{\leftarrow} \sim q_T$$

where here  $(B_t)_{t\geq 0}$  is also a Brownian motion. Defining  $(Y_t)_{t\in \mathbf{I}}=(X_t)_{t\in \mathbf{I}}$ , we see that the forward process satisfies the Markov property in Definition 1, and the information from the original sample  $X_0$  is degraded by more steps in the SDE. Furthermore, the reverse SDE with parameterized by the score function  $\nabla \ln q_{T-t}(X_t^{\leftarrow})$  can be viewed as successively sampling from the posteriors via Tweedie's formula.

**Example 9** (Discrete Diffusion Models [LME24]). Consider a set A denoting the vocabulary and let  $p \in A^T$ , and consider a forward process with index set  $I = [0, K] \cup {\infty}$ ,  $Y_0 = X$ , and  $Y_t \in A^T$  defined in the limit as follows,

$$p(Y_{t+\Delta t} = a \mid Y_t = b) = \delta_{ab} + Q_t(b, a)\Delta t + O(\Delta t^2),$$

where  $Q_t \in \mathbb{R}^{n \times n}$  are diffusion matrices with nonnegative non-diagonal entries and columns which sum to 0.  $(Y_t)_{t \in I}$  is also a Markov chain and as  $t \to \infty$ ,  $Y_t$  is degraded until it is eventually uninformative about the original sample  $Y_0$ .

**Example 10** (Autoregressive Language Models). Consider a set A denoting the vocabulary and let  $p \in A^T$ , and consider a forward process with index set  $I = \{0, 1, 2, ..., T\}$ ,  $Y_0 = X$ , and  $Y_t \in A^{T-t}$ . For  $t \in I$ , we let  $Y_t$  equal the last first T - t tokens of X. Clearly this is a Markov Chain, and the reverse process is equivalent to next-token prediction.

## <span id="page-27-0"></span>C Proof details

#### <span id="page-27-1"></span>C.1 Deferred details from Section 3

For Theorem 2, we employ the following two helper lemmas.

**Lemma 3.** By applying the law of total probability and Bayes' rule, we can show for  $Y_{\widehat{T}} \in \text{supp}(p_{\widehat{T}}^{S_{\text{target}}})$ ,

$$\int \left| P^{\leftarrow}(x|Y_{\widehat{T}}) - P^{\leftarrow}(x|Y_{\widehat{T}}, S_{\text{target}}) \right| dx \le 2 \sum_{\theta \in \Theta - S_{\text{target}}} P_{t \to \Theta}^{\leftarrow}(\theta|Y_{\widehat{T}}).$$

*Proof.* We can rewrite  $P^-(x|Y_{\widehat{T}}), P^-(x|Y_{\widehat{T}}, S_{\text{target}})$  using the law of total probability and Bayes' rule.

$$\begin{split} P^{\leftarrow}(x|Y_{\widehat{T}}) &= \sum_{\theta \in \Theta} P^{\leftarrow}_{t \to \Theta}(\theta|Y_{\widehat{T}}) P^{\leftarrow}(x|Y_{\widehat{T}}, \theta) \\ P^{\leftarrow}(x|Y_{\widehat{T}}, S_{\text{target}}) &= \sum_{\theta \in S_{\text{target}}} P^{\leftarrow}_{t \to \Theta}(\theta|Y_{\widehat{T}}, S_{\text{target}}) P^{\leftarrow}(x|Y_{\widehat{T}}, \theta) = \frac{\sum_{\theta \in S_{\text{target}}} P^{\leftarrow}_{t \to \Theta}(\theta|Y_{\widehat{T}}) P^{\leftarrow}(x|Y_{\widehat{T}}, \theta)}{\sum_{\theta \in S_{\text{target}}} P^{\leftarrow}_{t \to \Theta}(\theta|Y_{\widehat{T}})}. \end{split}$$

Note that the second equality on the second line follows from the fact that for all  $\theta \in S_{\text{target}}$ , the posteriors  $P_{t \to \Theta}^{\leftarrow}(\cdot|Y_t) \propto P_{t \to \Theta}^{\leftarrow}(\cdot|Y_t, S_{\text{target}})$  by the same normalization constant. Therefore the difference can be

written as

$$\begin{split} &\int \left| P^{\leftarrow}(x|Y_{\widehat{T}}) - P^{\leftarrow}(x|Y_{\widehat{T}}, S_{\text{target}}) \right| dx \\ &= \int \left| \sum_{\theta \in \Theta} P^{\leftarrow}_{t \to \Theta}(\theta|Y_{\widehat{T}}) P^{\leftarrow}(x|Y_{\widehat{T}}, \theta) - \frac{\sum_{\theta \in S_{\text{target}}} P^{\leftarrow}_{t \to \Theta}(\theta|Y_{\widehat{T}}) P^{\leftarrow}(x|Y_{\widehat{T}}, \theta)}{\sum_{\theta \in S_{\text{target}}} P^{\leftarrow}_{t \to \Theta}(\theta|Y_{\widehat{T}})} \right| dx \\ &= \int \left| \left( 1 - \frac{1}{\sum_{\theta \in S_{\text{target}}} P^{\leftarrow}_{t \to \Theta}(\theta|Y_{\widehat{T}})} \right) \sum_{\theta \in S_{\text{target}}} P^{\leftarrow}_{t \to \Theta}(\theta|Y_{\widehat{T}}) P^{\leftarrow}(x|Y_{\widehat{T}}, \theta) + \sum_{\theta \in \Theta - S_{\text{target}}} P^{\leftarrow}_{t \to \Theta}(\theta|Y_{\widehat{T}}) P^{\leftarrow}(x|Y_{\widehat{T}}, \theta) \right| dx. \end{split}$$

If  $\sum_{\theta \in \Theta - S_{\text{target}}} P_{t \to \Theta}^{\leftarrow}(\theta | Y_{\widehat{T}}) = 0$ , then the above is equal to 0 and we are done. If it is non-zero, we can factor out  $\sum_{\theta \in \Theta - S_{\text{target}}} P_{t \to \Theta}^{\leftarrow}(\theta | Y_{\widehat{T}})$  term, which allows us to write everything in terms of posteriors with respect to  $\Theta - S_{\text{target}}$  and  $S_{\text{target}}$ ,

$$\begin{split} &\int \left| \left( 1 - \frac{1}{\sum_{\theta \in S_{\text{target}}} P_{t \to \Theta}^{\leftarrow}(\theta | Y_{\widehat{T}})} \right) \sum_{\theta \in S_{\text{target}}} P_{t \to \Theta}^{\leftarrow}(\theta | Y_{\widehat{T}}) P^{\leftarrow}(x | Y_{\widehat{T}}, \theta) + \sum_{\theta \in \Theta - S_{\text{target}}} P_{t \to \Theta}^{\leftarrow}(\theta | Y_{\widehat{T}}) P^{\leftarrow}(x | Y_{\widehat{T}}, \theta) \right| dx \\ &= \sum_{\theta \in \Theta - S_{\text{target}}} P_{t \to \Theta}^{\leftarrow}(\theta | Y_{\widehat{T}}) \int \left| P^{\leftarrow}(x | Y_{\widehat{T}}, S_{\text{target}}) - P^{\leftarrow}(x | Y_{\widehat{T}}, \Theta - S_{\text{target}}) \right| dx \end{split}$$

Employing the trivial observation that

$$\int \left| P^{\leftarrow}(x|Y_{\widehat{T}}, S_{\text{target}}) - P^{\leftarrow}(x|Y_{\widehat{T}}, \Theta - S_{\text{target}}) \right| dx$$

$$\leq \int P^{\leftarrow}(x|Y_{\widehat{T}}, S_{\text{target}}) + P^{\leftarrow}(x|Y_{\widehat{T}}, \Theta - S_{\text{target}}) dx \leq 2,$$

we have

$$\int \left| P^{\leftarrow}(x|Y_{\widehat{T}}) - P^{\leftarrow}(x|Y_{\widehat{T}}, S_{\text{target}}) \right| dx \leq 2 \sum_{\theta \in \Theta - S_{\text{target}}} P_{t \to \Theta}^{\leftarrow}(\theta|Y_{\widehat{T}}).$$

**Lemma 4.** By Bayes's rule, we can derive for  $Y_{\widehat{T}} \in \text{supp}(p_{\widehat{T}})$ ,

$$\sum_{\theta \in \Theta - S_{\text{target}}} P_{t \to \Theta}^{\leftarrow}(\theta | Y_{\widehat{T}}) \leq \max(1, W) \frac{p_{\widehat{T}}^{\Theta - S_{\text{target}}}(Y_{\widehat{T}})}{p_{\widehat{T}}^{\Theta - S_{\text{target}}}(Y_{\widehat{T}}) + p_{\widehat{T}}^{S_{\text{target}}}(Y_{\widehat{T}})}$$

Proof. We obtain through Bayes' rule,

$$\sum_{\theta \in \Theta - S_{\text{target}}} P_{t \to \Theta}^{\leftarrow}(\theta | Y_{\widehat{T}}) = \frac{\sum_{\theta \in \Theta - S_{\text{target}}} w_{\theta} p_{\widehat{T}}^{\theta}(Y_{\widehat{T}})}{\sum_{\theta \in \Theta} w_{\theta} p_{\widehat{T}}^{\theta}(Y_{\widehat{T}})}.$$

We divide by the same normalizing constant  $\sum_{\theta \in \Theta - S_{\mathrm{target}}} w_{\theta}$  to obtain

$$\begin{split} \frac{\sum_{\theta \in \Theta - S_{\text{target}}} w_{\theta} p_{\widehat{T}}^{\theta}(Y_{\widehat{T}})}{\sum_{\theta \in \Theta} w_{\theta} p_{\widehat{T}}^{\theta}(Y_{\widehat{T}})} &= \frac{\frac{\sum_{\theta \in \Theta - S_{\text{target}}} w_{\theta} p_{\widehat{T}}^{\theta}(Y_{\widehat{T}})}{\sum_{\theta \in \Theta - S_{\text{target}}} w_{\theta}}}{\frac{\sum_{\theta \in \Theta - S_{\text{target}}} w_{\theta} p_{\widehat{T}}^{\theta}(Y_{\widehat{T}})}{\sum_{\theta \in \Theta - S_{\text{target}}} w_{\theta}} + \frac{\sum_{\theta \in S_{\text{target}}} w_{\theta} p_{\widehat{T}}^{\theta}(Y_{\widehat{T}})}{\sum_{\theta \in \Theta - S_{\text{target}}} w_{\theta}} \cdot \frac{\sum_{\theta \in S_{\text{target}}} w_{\theta}}{\sum_{\theta \in \Theta - S_{\text{target}}} w_{\theta}}} \\ &\leq \max \left(1, \frac{\sum_{\theta \in \Theta - S_{\text{target}}} w_{\theta}}{\sum_{\theta \in S_{\text{target}}} w_{\theta}}\right) \frac{p_{\widehat{T}}^{\Theta - S_{\text{target}}}(Y_{\widehat{T}})}{p_{\widehat{T}}^{\Theta - S_{\text{target}}}(Y_{\widehat{T}}) + p_{\widehat{T}}^{S_{\text{target}}}(Y_{\widehat{T}})}. \end{split}$$

#### <span id="page-29-0"></span>C.2 Proofs for Section 4

#### C.2.1 Diffusions

Here we use an alternative f-divergence to characterize the critical windows, the squared *Hellinger distance*, defined as  $H^2(P,Q) \triangleq \int (\sqrt{dP} - \sqrt{dQ})^2 d\mu$ , because there are explicit computations for the Hellinger distance for mixtures of Gaussians. We similarly exploit the following ratio inequality akin to Lemma 1,

**Lemma 7.** For probability measures P,Q,

$$\mathbb{E}_{x \sim P} \left[ \frac{\mathrm{d}Q}{\mathrm{d}P + \mathrm{d}Q} \right] \le \frac{1}{2} (1 - \frac{1}{2} \mathrm{H}^2(P, Q)).$$

We apply the following well-known formula for the Hellinger distance between two Gaussians.

<span id="page-29-1"></span>Lemma 8. We have

$$1 - \frac{1}{2}H^{2}(\mathcal{N}(\mu_{P}, \Sigma_{P}), \mathcal{N}(\mu_{Q}, \Sigma_{Q})) = \frac{|\Sigma_{P}|^{1/4}|\Sigma_{Q}|^{1/4}}{\left|\frac{\Sigma_{P} + \Sigma_{Q}}{2}\right|^{1/2}} \exp\left\{-\frac{1}{8}(\mu_{P} - \mu_{Q})^{\top}\left[\frac{\Sigma_{P} + \Sigma_{Q}}{2}\right]^{-1}(\mu_{P} - \mu_{Q})\right\}.$$

**Example 1** (Two Isotropic Gaussians). Let  $\Theta = \{\pm 1\}$ ,  $p^{+1} = \mathcal{N}(\mu, \mathrm{Id})$ ,  $p^{-1} = \mathcal{N}(-\mu, \mathrm{Id})$ . Then, we have a critical window transitioning from sampling from both components to the component +1 between  $T_{\mathrm{before}}^C = \ln \|\mu\| + \ln 2 + \ln 1/\epsilon$  and  $T_{\mathrm{after}}^C = \ln \|\mu\| - \ln \ln \frac{1}{2\epsilon^2}$ . When  $\widehat{T} \leq T_{\mathrm{after}}^C$ , then  $\mathrm{TV}(p^{+1,\widehat{T}},p^{+1}) \lesssim \epsilon$ . When  $\widehat{T} \geq T_{\mathrm{before}}^C$ ,  $\mathrm{TV}(p^{+1,\widehat{T}},p) \lesssim \epsilon$ .

*Proof.* The proof for  $T_{\text{before}}^C$ , a simple application of Pinsker's inequality, can be found in Appendix B.1 of [LC24]. Directly applying the new Master Theorem 2 to  $T_{\text{after}}^C$ , we need only show that

$$\frac{\sqrt{2}}{2} \left( 1 - \frac{1}{2} \mathbf{H}^2(p_{T_{\mathrm{after}}^c}^{+1}, p_{T_{\mathrm{after}}^c}^{-1}) \right) \leq \frac{\sqrt{2}}{2} \exp(-\frac{1}{2} \|\mu\|^2 e^{-2t}) \leq \epsilon.$$

**Example 2** (Random mean spherical Gaussians). We first sample  $\mu_i \sim \mathcal{N}(0, \mathrm{Id})$  for  $i \in [K]$  i.i.d. and let  $\Theta = \{\mathcal{N}(\mu_i, \mathrm{Id})\}_{i \in [K]}$ . We let  $S_{\mathrm{before}} = \Theta$  and  $S_{\mathrm{after}} = \{\mu_i\}$ . Then, we can compute  $T_{\mathrm{before}}^C = \max_{j \in [K]} \ln \|\mu_i - \mu_j\| + \ln(1/\epsilon)$  and  $T_{\mathrm{after}}^C = \min_{j \in [K], i \neq j} \ln \|\mu_i - \mu_j\| - \frac{1}{2} \ln 8 \ln \frac{K}{\epsilon}$ . Furthermore, with high probability over the selection of the means,  $T_{\mathrm{before}}^C - T_{\mathrm{after}}^C = O_{K,\epsilon}(1)$  as  $d \to \infty$ .

*Proof.* The proof for  $T_{\mathrm{before}}^{C}$  can be found in Section 5.2 of [LC24]. We need to slightly modify the proof of Theorem 2 so that we can write the desired bound for  $T_{\mathrm{after}}^{C}$  in terms of the Hellinger distance of individual components. We use the same notation. By convexity, we can bound

$$\begin{split} \mathbb{E}_{Y_{\widehat{T}} \sim p_{\widehat{T}}^{S_{\text{target}}}} \left[ \frac{\sum_{\theta \in \Theta - S_{\text{target}}} w_{\theta} p_{\widehat{T}}^{\theta}(Y_{\widehat{T}})}{\sum_{\theta \in \Theta} w_{\theta} p_{\widehat{T}}^{\theta}(Y_{\widehat{T}})} \right] &\leq \sum_{\theta \in \Theta - S_{\text{target}}} w_{\theta} \sum_{\phi \in S_{\text{target}}} w_{\phi} \mathbb{E}_{Y_{\widehat{T}} \sim p_{\widehat{T}}^{\phi}} \left[ \frac{p_{\widehat{T}}^{\theta}(Y_{\widehat{T}})}{w_{\theta} p_{\widehat{T}}^{\theta}(Y_{\widehat{T}}) + w_{\phi} p_{\widehat{T}}^{\phi}(Y_{\widehat{T}})} \right] \\ &\leq K \max_{\theta \in \Theta - S_{\text{target}}, \phi \in S_{\text{target}}} \left( 1 - \frac{1}{2} H^{2}(p_{\widehat{T}}^{\theta}, p_{\widehat{T}}^{\phi}) \right) \leq \epsilon, \end{split}$$

when  $\widehat{T} \leq T_{\text{after}}^C$ . To conclude the second part of the theorem, observe that by concentration of measure (e.g., Theorem 3.1.1 from [Ver]) and a union bound, there exists a constant T independent of d such that  $\|\mu_j\| \in [\sqrt{d} - T, \sqrt{d} + T]$  for all  $j \in [K]$  with high probability. Furthermore, by known Gaussian Suprema inequalities, we can also assume that there exists a constant T' independent of d such that  $|\langle \mu_i, \mu_i \rangle| \leq T' \|\mu_i\|$  (Lemma 5.1 from [vH16]). Thus, we can conclude that

$$\begin{split} \max_{j \in [K]} \|\mu_i - \mu_j\|^2 & \leq 2d + 4T\sqrt{d} + 2T^2 + 2T'(\sqrt{d} + T) = O(d). \\ \max_{j \in [K]} \|\mu_i - \mu_j\|^2 & \geq 2d - 4T\sqrt{d} + 2T^2 - 2T'(\sqrt{d} + T) = \Omega(d). \end{split}$$

The difference in log scale is thus constant,

$$\frac{1}{2} \left( \ln \max_{j \in [K]} \|\mu_i - \mu_j\|^2 - \ln \min_{j \in [K], i \neq j} \|\mu_i - \mu_j\|^2 \right) = O_{K, \epsilon}(1)$$

**Example 3** (Two Dirac delta functions with a random masking procedure). Let  $p \in \{\pm 1\}^T$ , and consider a forward process with index set  $\mathbf{I} = [0,1]$ ,  $Y_0 = X$ , and  $Y_t \in \{\pm 1, [\mathsf{MASKED}]\}^T$ . For  $t \in \mathbf{I}$ , we let all the value at index  $i \in [T]$  be set to  $[\mathsf{MASKED}]$  with probability t independently. For a mixture of two Dirac delta functions, we can express the critical window in terms of the Hamming distance between the corresponding strings. Let  $\Theta = \{\theta_{\pm 1}\}, \ell_{\pm 1} \in \{\pm 1\}^T, p^{\theta_{\pm 1}} \sim \delta_{\ell_{\pm 1}}, w_{\pm 1} = \frac{1}{2}$ . Then, on component 1 we have the critical window

$$T_{\text{before}}^{C} = \exp\left[\frac{\ln(1-\epsilon)}{d_{H}(\delta_{\ell_{1}}, \delta_{\ell_{-1}})}\right], T_{\text{after}}^{C} = \exp\left[\frac{\ln\epsilon^{2}}{d_{H}(\delta_{\ell_{1}}, \delta_{\ell_{-1}})}\right]$$

For sufficiently large  $d_H(\delta_{\ell_1}, \delta_{\ell_{-1}})$ , the window size  $T_{\text{before}}^C - T_{\text{after}}^C = O\left(\frac{1}{d_H(\delta_{\ell_1}, \delta_{\ell_{-1}})}\right)$ . If  $d_H(\delta_{\ell_1}, \delta_{\ell_{-1}})$  increases with T, then the width of the critical window negligible.

*Proof.* To prove  $\mathrm{TV}(p^{1,\widehat{T}},p^1)\lesssim \epsilon$  when  $\widehat{T}\leq T^C_{\mathrm{after}}$ , observe that when  $\widehat{T}\leq T^C_{\mathrm{after}}$ , the probability that all the differing elements between  $\ell_1,\ell_{-1}$  are masked is exactly  $\widehat{T}^{d_H(\delta_{\ell_1},\delta_{\ell_{-1}})}\leq \epsilon^2$ . That means that there exists a set A with  $p_{\widehat{T}}^{+1}(A)\geq 1-\epsilon^2$  and  $p_{\widehat{T}}^{-1}(A)=0$ , so by the definition of total variation,  $\mathrm{TV}(p_{\widehat{T}}^{+1},p_{\widehat{T}}^{-1})\geq 1-\epsilon^2$ . Obviously,  $\mathrm{TV}(p_{\widehat{T}}^{+1},p_{\widehat{T}}^{+1})=0$  as well, so by Theorem 2, we obtain  $\mathrm{TV}(p^{1,\widehat{T}},p^1)\lesssim \epsilon$ . To prove that  $\mathrm{TV}(p^{1,\widehat{T}},p^{\{\pm 1\}})\leq \epsilon$  when  $\widehat{T}\geq T^C_{\mathrm{before}}$ , we need only show that  $\mathrm{TV}(p^{+1}_{\widehat{T}},p_{\widehat{T}})\leq \epsilon$ . By Lemma 15 of [LC24], it suffices to show that  $\mathrm{TV}(p^{+1}_{\widehat{T}},p^{-1}_{\widehat{T}})\leq \epsilon$  by a simple triangle inequality argument. Consider the set  $A\subset \{\pm 1,[\mathrm{MASKED}]\}^T$  such that  $\mathrm{TV}(p^{+1}_{\widehat{T}},p^{-1}_{\widehat{T}})=p^{+1}_{\widehat{T}}(A)-p^{-1}_{\widehat{T}}(A)$ . Consider the set  $B=\mathrm{supp}(p^{+1}_{\widehat{T}})\cap \mathrm{supp}(p^{-1}_{\widehat{T}})$ . For any  $x\in B$ , we know  $p^{+1}_{\widehat{T}}(x)=p^{-1}_{\widehat{T}}(x)$  because the same number of tokens need to be masked from  $\ell_{\pm 1}$ . This means we have  $p^{+1}_{\widehat{T}}(B)=p^{-1}_{\widehat{T}}(B)\geq \widehat{T}^{d_H(\ell_{+1},\ell_{-1})}\geq 1-\epsilon$ . Because  $p^{+1}_{\widehat{T}}(A)-p^{-1}_{\widehat{T}}(A)=p^{+1}_{\widehat{T}}(A-B)-p^{-1}_{\widehat{T}}(A-B)$ , we have  $p^{+1}_{\widehat{T}}(A)-p^{-1}_{\widehat{T}}(A)\leq p^{+1}_{\widehat{T}}(\{\pm 1,[\mathrm{MASKED}]\}^T-B)\leq \epsilon$ . □

### C.2.2 Autoregression

**Example 5** (Math problem-solving as a random walk). We model solving a math problem as taking a random walk on  $\mathbb{Z}$  with stepsize 1 of length T. If the random walk hits +A, then it has 'solved' the problem; if the random walk hits -A, then it has obtained an incorrect solution. Assume that we have two modes: a strong problem solving mode (denoted +1), which takes a+1 step with probability  $0.5+\Delta$ , and a weak problem solving mode (denoted -1), which takes a+1 step with probability  $0.5-\Delta$ . Assuming that  $\frac{\ln(2/\epsilon^2)}{2\Delta^2} < A$  and  $\epsilon^2 < 10^{-3}(0.5-\Delta)(0.5+\Delta)$ , there is a critical window for the strong problem solving window of  $T_{\text{before}}^C = T - \frac{\epsilon^2}{\Delta^2} + 2$  and  $T_{\text{after}}^C = T - \frac{\ln(2/\epsilon^2)}{2\Delta^2}$ . Note the critical window has width  $\Theta(1/\Delta^2)$  independent of T.

*Proof.* Because only the direction of steps matter, we can model the critical window for this random walk as observing a sequence of  $\pm 1$  with an autoregressive language model. Let  $p \in \{\pm 1\}^T$ , and consider a forward process with index set  $\mathbf{I} = \{0, 1, 2, \dots, T\}$ ,  $Y_0 = X$ , and  $Y_t \in \{\pm 1, [\text{MASKED}]\}^T$ . For  $t \in \mathbf{I}$ , we let the last t tokens of  $Y_t$  be deterministically set to [MASKED]. We generate data as a mixture of biased coins with separation  $2\Delta < 0.01$ . For a mixture of two biased coins, with probabilities of  $0.5 \pm \Delta$  ( $\theta_{\pm 1}$  respectively) of yielding 1, we can compute the critical window and show that it tightly clusters around  $\Theta(1/\Delta^2)$ . Let  $\Theta = \{\theta_{\pm 1}\}, p^{\theta_{\pm 1}} \sim (\text{Bern}(\theta_{\pm 1}))^{\otimes T}, w_{\pm 1} = \frac{1}{2}$ . We also assume  $\epsilon^2 < 10^{-3}(0.5 - \Delta)(0.5 + \Delta)$ . Then, on component 1 we have the critical window  $T_{\text{before}}^C = T - \frac{\epsilon^2}{\Delta^2} + 2$  and  $T_{\text{after}}^C = T - \frac{\ln(2/\epsilon^2)}{2\Delta^2}$ . When  $\widehat{T} \leq T_{\text{after}}^C$ , then  $\text{TV}(p^{1,\widehat{T}},p^1) \lesssim \epsilon$ . When  $\widehat{T} \geq T_{\text{before}}^C$ ,  $\text{TV}(p^{1,\widehat{T}},p) \lesssim \epsilon$ .

Note that the number of +1 is sufficient for disambiguating  $\theta_{\pm 1}$ . To prove the bounds  $T_{\text{before}}^C$ , we show that with only  $\frac{\epsilon^2}{\Delta^2} - 2$  samples the total variation between  $0.5 - \Delta$  and  $0.5 + \Delta$  is negligible. Using [Roo01], we find

$$\text{TV}(p_{T_{\text{after}}^{C}}^{+1}, p_{T_{\text{after}}^{C}}^{-1}) \leq \frac{2\Delta\sqrt{\frac{T - T_{\text{after}}^{C}}{2(0.5 - \Delta)(0.5 + \Delta)}}}{\left(1 - 2\Delta\sqrt{\frac{T - T_{\text{after}}^{C}}{2(0.5 - \Delta)(0.5 + \Delta)}}\right)^{2}} \lesssim 3\epsilon.$$

For  $T_{\text{after}}^C$ , we compute how many samples it takes for  $p^{\pm 1}$  to have only  $\epsilon$  overlap in total variation using Hoeffding's inequality. If we have n samples, the mean  $\overline{X}$  of the n samples of  $\pm 1$  for  $p=0.5+\Delta$  satisfies the concentration inequality  $P(|\overline{X}-2\Delta|>\Delta)\leq 2\exp(-2nt^2)$  (furthermore we can ignore the stopping condition by our requirement that  $\frac{\ln(2/\epsilon^2)}{2\Delta^2}< A$ ). We find  $P(|\overline{X}-2\Delta|>\Delta)\leq \epsilon^2$  for  $T-T_{\text{after}}^C$  samples, proving that the total variation is at least  $1-\epsilon^2$ .

## <span id="page-31-0"></span>C.2.3 Autoregressive model with a Gaussian mixture model as the underlying concept distribution

We consider a model for autoregressive data similar to the one presented in [ALL<sup>+</sup>19]. Each word is a vector  $w \in \mathbb{R}^d$  and the context length is  $T \in \mathbb{Z}^{>0}$ . The original samples are  $x \in \mathbb{R}^{T \times d}$ . Let  $\Theta = \{u, v\}$ , where  $u, v \in \mathcal{S}^{d-1}$ . We define the distribution  $p^{\theta}$  for  $\theta \in \Theta$  as follows. We generate the path of a discourse vector  $\left(C_t^{\theta}\right)_{t \in [0,\infty]} \in \mathbb{R}^d$  with the reverse SDE Orstein-Uhlenbeck process such that  $C_{\infty}^{\theta} \sim \mathcal{N}(0, \mathrm{Id})$  and  $C_{0}^{\theta} \sim \mathcal{N}(0, \mathrm{Id} + \alpha\theta\theta^{\top})$  for some  $\alpha > 0$ . We let  $q_t^{\theta}$  be the law of  $C_t^{\theta}$  for  $t \geq 0$ . We let  $\mathbf{I} = \{0, 1, \dots, T\}$ , and for  $t \in \mathbf{I}$ , we draw samples  $w_t^{\theta} \in \mathbb{R}^d$  where we impose a normal Gaussian prior and have  $w_t^{\theta} \mid C_t^{\theta} \propto \exp(\langle C_t^{\theta}, \cdot \rangle)$ . Then we return the corpus  $\{w_t^{\theta}\}_{t \in \mathbf{I}}$  as an output.

**Theorem 9** (Autoregressive with a mixture of two Gaussians as the concept distribution ). We assume that  $\exp(-T)\sqrt{\alpha - \log(1+\alpha)} \le \epsilon$ . Let  $\phi(x) = \frac{2+x}{(2+x(1+\langle u,v\rangle)/2)(2+x(1-\langle u,v\rangle)/2)}$ . Then, on component u we

have

$$T_{\mathrm{after}}^{C} = \frac{1}{2} \ln \left[ \frac{\alpha}{\phi^{-1}(\epsilon^2)} \right], T_{\mathrm{before}}^{C} = \frac{1}{2} \ln \left[ \frac{\alpha \sqrt{1 + \alpha^2} \sqrt{1 - \langle u, v \rangle^2}}{\epsilon} \right].$$

When  $\widehat{T} \leq T_{\text{after}}^C$ , then  $\text{TV}(p^{\{u\},\widehat{T}}, p^{\{u\}}) \lesssim \epsilon$ . When  $\widehat{T} \geq T_{\text{before}}^C$ ,  $\text{TV}(p^{\{u\},\widehat{T}}, p) \lesssim \epsilon$ .

This proof will require Theorem 10 and Lemmas 8 and 11, which is stated below:

<span id="page-32-0"></span>**Theorem 10** (Section 5.2 of [CCL<sup>+</sup>23]). Let  $(Y_t)_{t \in [0,T]}$  and  $(Y'_t)_{t \in [0,T]}$  denote the solutions to

$$\begin{split} \mathrm{d}Y_t &= b_t(Y_t)\,\mathrm{d}t + \sqrt{2}\mathrm{d}B_t\,, \qquad Y_0 \sim q \\ \mathrm{d}Y_t' &= b_t'(Y_t')\,\mathrm{d}t + \sqrt{2}\mathrm{d}B_t\,, \qquad Y_0' \sim q\,. \end{split}$$

Let q and q' denote the laws of  $Y_T$  and  $Y_T'$  respectively. If  $b_t$ ,  $b_t'$  satisfy that  $\int_0^T \mathbb{E}_Q \|b_t(Y_t) - b_t'(Y_t)\|^2 dt < \infty$ , then  $\mathrm{KL}(q\|q') \leq \int_0^T \mathbb{E}_Q \|b_t(Y_t) - b_t'(Y_t)\|^2 dt$ .

<span id="page-32-1"></span>**Lemma 11.** Let  $u, v \in S^{d-1}$ . Then

$$||uu^{\top} - vv^{\top}||_{\text{op}} \le \sqrt{1 - \langle u, v \rangle^{2}}$$
$$\lambda(uu^{\top} + vv^{\top}) = \{1 \pm \langle u, v \rangle\}.$$

*Proof.* There exists  $r \in S^{d-1}$  such that  $v = \langle u, v \rangle u + \sqrt{1 - \langle u, v \rangle^2} r$  and  $u \perp r$ . We find that

$$uu^\top - vv^\top = (1 - \langle u, v \rangle^2)uu^\top - (1 - \langle u, v \rangle^2)rr^\top - \langle u, v \rangle \sqrt{1 - \langle u, v \rangle^2}[ur^\top + ru^\top].$$

We can explicitly compute the eigenvalues of  $uu^{\top} - vv^{\top}$  using the discriminant and find that they are equal to  $\pm \sqrt{1 - \langle u, v \rangle^2}$ . By a similar derivation, we can write

$$uu^{\top} + vv^{\top} = (1 + \langle u, v \rangle^2)uu^{\top} + (1 - \langle u, v \rangle^2)rr^{\top} + \langle u, v \rangle \sqrt{1 - \langle u, v \rangle^2}[ur^{\top} + ru^{\top}].$$

which gives us eigenvalues for  $uu^{\top} + vv^{\top}$  of  $1 \pm \langle u, v \rangle$ .

*Proof.* To compute the  $T^C_{\text{after}}$  bounds, we compare the difference in Hellinger distance of the distribution of words words at generated at index  $\widehat{T}$ ,  $w^u_{\widehat{T}}$ . By the data processing inequality  $1-\frac{1}{2}H^2(p^u_{\widehat{T}},p^v_{\widehat{T}}) \leq 1-\frac{1}{2}H^2(w^u_{\widehat{T}},w^v_{\widehat{T}})$ , so it suffices to show  $1-\frac{1}{2}H^2(w^u_{\widehat{T}},w^v_{\widehat{T}}) \lesssim \epsilon$ . Because the Gaussian is its own conjugate prior and  $w^u_{\widehat{T}}|C^\theta_t \propto \exp(-\frac{1}{2}\|w^u_{\widehat{T}}-C^\theta_t\|^2)$ , we can compute  $w^u_{\widehat{T}}|C^\theta_t \sim \mathcal{N}(C^\theta_t,\mathrm{Id})$  and  $w^u_{\widehat{T}} \sim \mathcal{N}(0,2\mathrm{Id}+\alpha e^{-2t}uu^\top)$ . Applying Lemmas 8 and 11, we can explicitly compute

$$1-\frac{1}{2}H^2(w^u_{\widehat{T}},w^v_{\widehat{T}})\lesssim \sqrt{\frac{2+\alpha e^{-2\widehat{T}}}{(2+\alpha e^{-2\widehat{T}}(1+\langle u,v\rangle)/2)(2+\alpha e^{-2\widehat{T}}(1-\langle u,v\rangle)/2)}}\lesssim \sqrt{\phi(\alpha e^{-2\widehat{T}})}\leq \epsilon.$$

To compute  $T_{\mathrm{before}}^C$ , we first use the data processing inequality to reduce the difference in the emitted tokens to the difference in the paths of the context vectors, and then apply the approximation error bounds from Theorem 10 to bound the differences in path measures. When  $\widehat{T} \geq T_{\mathrm{before}}^C$ , we can use the triangle inequality to write  $\mathrm{TV}(p^{\{u\},\widehat{T}},p) = \mathrm{TV}(p^u_{\widehat{T}},p^{\{u,v\}}_{\widehat{T}}) \leq \mathrm{TV}(p^u_{\widehat{T}},p^v_{\widehat{T}})$ . Note that  $p^\theta_{\widehat{T}}$  is the distribution of the first  $T-\widehat{T}$  tokens generated by the model under  $\theta$ . Note that  $p^u_{\widehat{T}}$  is a function of  $(C^u_t)_{t\in \mathrm{In}(\widehat{T},T)}$  and

 $p_{\widehat{T}}^{\nu}$  is a function of  $(C_t^{\nu})_{t \in I \cap [\widehat{T}, T]}$ . By the data processing inequality, we can bound the difference in terms of the distributions over the tokens in terms of the law of the process of the discourse vectors,

$$\mathrm{TV}(p_{\widehat{T}}^u, p_{\widehat{T}}^v) \leq \mathrm{TV}((C_t^u)_{t \in \mathrm{I} \cap [\widehat{T}]}, (C_t^v)_{t \in \mathrm{I} \cap [\widehat{T}, T]}) \leq \mathrm{TV}((C_t^u)_{t \in [\widehat{T}, T]}, (C_t^v)_{t \in [\widehat{T}, T]}).$$

Note that for  $\theta \in \Theta$ ,  $(C_t^{\theta})_{t \in [\widehat{T},T]}$  is generated by the following reverse time SDE,

$$dC_t^{\theta} = \{C_t^{\theta} + 2\nabla \ln q_t^{\theta}(C_t^{\theta})\} dt + \sqrt{2} dB_t, \qquad t \in [\widehat{T}, T], C_T^{\theta} \sim q_T^{\theta}.$$

Now we define  $(C_t^{u\to v})_{t\in [\widehat{T},T]}$  to be the reverse SDE defined by initializing at  $q_T^u$  but with the score of  $q_t^v$ ,

$$\mathrm{d}C_t^{u\to v} = \{C_t^{u\to v} + 2\nabla \ln q_t^v(C_t^{u\to v})\}\,\mathrm{d}t + \sqrt{2}\,\mathrm{d}B_t, \qquad t\in [\widehat{T},T], C_T^{u\to v}\sim q_T^u.$$

By the triangle inequality, we have

$$\text{TV}((C_t^u)_{t\in [\widehat{T},T]}, (C_t^v)_{t\in [\widehat{T},T]}) \leq \underbrace{\text{TV}((C_t^u)_{t\in [\widehat{T},T]}, (C_t^{u\to v})_{t\in [\widehat{T},T]})}_{\text{(I)}} + \underbrace{\text{TV}((C_t^{u\to v})_{t\in [\widehat{T},T]}, (C_t^v)_{t\in [\widehat{T},T]})}_{\text{(II)}}.$$

To bound (I), observe that the SDEs have different scores but the same initializations. We apply Theorem 10 to  $TV((C_t^u)_{t\in [\widehat{T},T]}, (C_t^{u\to v})_{t\in [\widehat{T},T]})$  and obtain

$$\begin{split} \text{TV}((C^u_t)_{t \in [\widehat{T}, T]}, (C^{u \to v}_t)_{t \in [\widehat{T}, T]}) &\leq \sqrt{\text{KL}((C^u_t)_{t \in [\widehat{T}, T]} || (C^{u \to v}_t)_{t \in [\widehat{T}, T]})} \\ &\leq \sqrt{\int_{\widehat{T}}^T \mathbb{E}_{X \sim C^u_t} ||\nabla \ln p^u_t(X) - \nabla \ln p^v_t(X)||^2 dt} \end{split}$$

We simplify the inner expectation by using the

$$\begin{split} \|\nabla \ln p_t^u(X) - \nabla \ln p_t^v(X)\| &= \left\| \left[ (\operatorname{Id} + \alpha e^{-2t} u u^\top)^{-1} - (\operatorname{Id} + \alpha e^{-2t} v v^\top)^{-1} \right] x \right\| \\ &= \left\| \left[ \left( \operatorname{Id} - \frac{\alpha e^{-2t}}{1 + \alpha e^{-2t}} u u^\top \right) - \left( \operatorname{Id} - \frac{\alpha e^{-2t}}{1 + \alpha e^{-2t}} v v^\top \right) \right] x \right\| \\ &\lesssim \alpha e^{-2t} \|u u^\top - v v^\top\|_{\operatorname{op}} \left\| \prod_{\operatorname{span}(u,v)} x \right\|, \\ &= \alpha e^{-2t} \sqrt{1 - \langle u, v \rangle^2} \left\| \prod_{\operatorname{span}(u,v)} x \right\|. \quad \text{(Lemma 11)} \end{split}$$

We can upper bound  $\mathbb{E}_{X \sim C_t^u} \| \prod_{\text{span}(u,v)} X \|^2$  by considering right-triangular L such that  $L^\top L = \text{Id} + \alpha u u^\top$ . and  $\prod_{\text{span}(u,v)} X = \prod_{\text{span}(u,v)} LY$ , where  $Y \sim \mathcal{N}(0,\text{Id})$ . The operator norm of  $\prod_{\text{span}(u,v)} L$  is

$$\left\| \prod_{\text{span}(u,v)} \circ L \right\|_{\text{op}} \le \|L\|_{\text{op}} \le \sqrt{1 + \alpha^2 e^{-2t}} \le \sqrt{1 + \alpha^2}.$$

 $\prod_{\mathrm{span}(u,v)} \circ L \text{ is also rank 2 and } \left(\prod_{\mathrm{span}(u,v)} \circ L\right) Y = \left(\prod_{\mathrm{span}(u,v)} \circ L\right) \circ \prod_{L^{-1}\mathrm{span}(u,v)} Y, \text{ where } \prod_{L^{-1}\mathrm{span}(u,v)} Y \sim \mathcal{N}(0,\mathrm{Id}_2).$  Thus we have

$$\mathbb{E}_{Y \sim \mathcal{N}(0, \mathrm{Id})} \left[ \left\| \left( \prod_{\mathrm{span}(u, v)} \circ L \right) Y \right\|^2 \right] = \mathbb{E}_{Y' \sim \mathcal{N}(0, \mathrm{Id}_2)} \left[ \left\| \left( \prod_{\mathrm{span}(u, v)} \circ L \right) Y' \right\|^2 \right] \lesssim 1 + \alpha^2$$

Combining this information together, we are able to compute,

$$\begin{aligned} \text{TV}((C^u_t)_{t \in [\widehat{T}, T]}, (C^{u \to v}_t)_{t \in [\widehat{T}, T]}) &\leq \sqrt{\int_{\widehat{T}}^T \mathbb{E}_{X \sim C^u_t} \|\nabla \ln p^u_t(X) - \nabla \ln p^v_t(X)\|^2 dt} \\ &\lesssim \alpha \sqrt{1 + \alpha^2} \sqrt{1 - \langle u, v \rangle^2} e^{-2\widehat{T}} \\ &\lesssim \epsilon. \end{aligned}$$

To bound (II), we observe that both are run with the same score so we need only bound the difference at initialization. By the data processing inequality, we again have  $\mathrm{TV}((C_t^{u \to v})_{t \in [\widehat{T}, T]}, (C_t^v)_{t \in [\widehat{T}, T]}) \leq \mathrm{TV}(p_T^u, p_T^v)$ . We can again apply the triangle inequality to get  $\mathrm{TV}(p_T^u, p_T^v) \leq \mathrm{TV}(p_T^u, \gamma^d) + \mathrm{TV}(\gamma^d, p_T^v) \leq .$  For any  $\theta \in \Theta$ , we have by the forward convergence of the OU process  $\mathrm{TV}(p_T^u, \gamma^d) \leq \exp(-T)\sqrt{\mathrm{KL}(p_u||\gamma^d)}$ . We can explicitly compute  $\mathrm{KL}(p_u||\gamma^d)$  as

$$\mathrm{KL}(p_u||\gamma^d) = \frac{1}{2} \left[ d + \alpha - d - \log(\mathrm{Id} + \alpha u u^\top) \right] = \frac{1}{2} \left[ \alpha - \log(1 + \alpha) \right].$$

Thus, we obtain the following bound on  $\mathrm{TV}((C_t^{u \to v})_{t \in [\widehat{T}, T]}, (C_t^v)_{t \in [\widehat{T}, T]})$  of

$$TV((C_t^{u\to v})_{t\in [\widehat{T},T]}, (C_t^v)_{t\in [\widehat{T},T]}) \lesssim \exp(-T)\sqrt{\alpha - \log(1+\alpha)} \lesssim \epsilon.$$

#### C.2.4 Interweaving transitions from other distributions

In this section, we extend our critical windows framework to the setting where at certain steps of sampling procedure, instead of using the reverse Markov transition kernel from the original stochastic localization sampler, we use an alternative distribution which is not necessarily related to the original sampler. This includes many important applications of generative models, in which one seeks to combine the priors learned from data with some other algorithm. For example, one may want to combine the language model with a problem generation oracle in in-context learning [DLD<sup>+</sup>24].

As [XRLM22] points out, the transition from the answer to one problem to the problem statement of another example in-context learning is determined by an alternative transition kernel (which they call  $p_{\text{prompt}}$ ). Although the probability of transition from one answer to the problem statement of another example is extremely low under the natural data distribution, one still hopes that with sufficiently many samples, the model selects the correct  $\theta^* \in \Theta$  if these lower probability transitions are overcome by the distributional difference for  $\theta \in \Theta$  with  $\theta \neq \theta^*$ . Similarly, under our critical windows framework, we can hope to capture the idea that we specialize to a particular  $\theta^*$  given a sufficiently long context. In Section C.2.5, we first present a general framework for characterizing critical windows in this setting. Then, in section C.2.6, we consider the case of in-context learning by autoregressive language models and prove convergence.

## <span id="page-34-0"></span>C.2.5 General interweaving framework

We present this framework for the case where the index set  $\mathbf{I} = \{0, 1, \dots, m\}$  is discrete. Like before, assume we have a series of reverse Markov transition kernels  $P_{k \to k-1}^{\leftarrow, p}(\cdot|\cdot)$ , for  $k \in \mathbf{I}$ , but we also assume we have an alternative distribution  $P_{k \to k-1}^{\leftarrow, \text{alt}}(\cdot|\cdot)$  that we use to sample for transitions  $k \in A \subsetneq \mathbf{I}$ . For our sampling procedure, we sample  $Y_m$ , and for  $k = m-1, m-2, \ldots, 0$ , we take  $Y_k \sim P_{k+1 \to k}^{\leftarrow, p}(\cdot|Y_{k+1})$  for  $k \in \mathbf{I} - A$  and  $Y_k \sim P_{k+1 \to k}^{\leftarrow, \text{alt}}(\cdot|Y_{k+1})$  for  $k \in A$ . We denote the final distribution  $p^{\text{alt}}$ .

Now, we also need to adjust our definitions of  $p^S$  to this particular sampling procedure. We define  $p^{S,\text{alt}}$  for  $S \subset \Theta$  to the distribution over outputs when we instead use the kernels  $P_{k+1 \to k}^{\leftarrow}(\cdot|Y_{k+1},S)$  instead of  $P_{k+1 \to k}^{\leftarrow}(\cdot|Y_{k+1})$ . To relate  $p^{\theta,\text{alt}}$  to  $p^{\theta,\text{alt}}$  for  $\theta \in \Theta$ , we need to assume transitions from alt do not affect the posterior distribution over  $p^{\Theta}(\theta|Y_t)$ .

<span id="page-35-1"></span>**Assumption 1.** For all  $y \in \mathbb{R}$  and  $x \in \text{supp}(P_{k+1 \to k}^{\leftarrow, \text{alt}}(\cdot|y))$ , we have for all  $\theta, \theta' \in \Theta$ , the equality  $P_{k+1 \to k}^{\leftarrow}(x|y,\theta) = P_{k+1 \to k}^{\leftarrow}(x|y,\theta')$ .

Adopting our definitions from Section 3, we let

$$\begin{split} & T_{\text{lower,alt}}(\epsilon) \in \{t \in \mathbf{I} : \text{TV}(p_t^{S_{\text{init}}, \text{alt}}, p_t^{S_{\text{target}}, \text{alt}}) \leq \epsilon\} \\ & T_{\text{upper,alt}}(\epsilon) \in \{t \in \mathbf{I} : \text{TV}(p_t^{S_{\text{target}}, \text{alt}}, p_t^{\Theta - S_{\text{target}}, \text{alt}}) \geq 1 - \epsilon^2\}. \end{split}$$

The main challenge of the below corollary is simply show that the final distribution  $p^{\Theta,\text{alt}}$  can be written as a mixture of  $p^{\theta,\text{alt}}$  with the same mixing weights as before.

<span id="page-35-2"></span>**Corollary 3.** Under Assumption 1, for  $\epsilon > 0$ , if  $\widehat{T} \geq T_{\text{lower,alt}}(\epsilon)$  and  $\widehat{T} \leq T_{\text{upper,alt}}(\epsilon)$ , then

$$\text{TV}(p^{S_{\text{init}},\widehat{T}}, p^{S_{\text{target}}, \text{alt}}) \leq \left(1 + \sqrt{2} \max\left(1, \frac{\sum_{\theta \in \Theta - S_{\text{target}}} w_{\theta}}{\sum_{\theta \in S_{\text{target}}} w_{\theta}}\right) / 2\right) \epsilon.$$

*Proof.* We need only show that  $p^{\Theta, \mathrm{alt}} \triangleq \sum_{\theta \in \Theta} w_{\theta} p^{\theta, \mathrm{alt}}$ . It suffices to shows that the probability of generating a path  $Y_m, Y_{m-1}, \ldots, Y_0$  are the same under both density functions. We need only consider transitions for  $k \in \mathbf{I} - A$ , because for  $k \in A$ , the transitions are both given by the alternative distribution. For the transitions not given by alt, note that we are using the original model, so

$$p^{\Theta,\text{alt}}(Y_{k-1}|Y_k) = \frac{\sum_{\theta \in \Theta} w_{\theta} p^{\theta}(Y_k) P_{k \to k-1}^{\leftarrow}(Y_{k-1}|Y_k, \theta)}{\sum_{\theta \in \Theta} w_{\theta} p^{\theta}(Y_k)}.$$

Furthermore, for the mixture model, this probability is

$$p^{\text{mix}} = \frac{\sum_{\theta \in \Theta} w_{\theta} p^{\theta, \text{alt}}(Y_m, Y_{m-1}, \dots, Y_{k+1}, Y_k) P_{k \to k-1}^{\leftarrow}(Y_{k-1} | Y_k, \theta)}{\sum_{\theta \in \Theta} w_{\theta} p^{\theta, \text{alt}}(Y_m, Y_{m-1}, \dots, Y_{k+1}, Y_k)}.$$

The distinction between Equation C.2.5 and Equation C.2.5 is that in the former we are using the likelihood of  $p^{\theta}$  instead of  $p^{\theta, alt}$ . Thus it suffices to show that  $p^{\theta} \propto p^{\theta, alt}$ . We explicitly write out the probability,

$$p^{\theta}(Y_m, Y_{m-1}, \dots, Y_{k+1}, Y_k) = \prod_{i=k+1}^m p^{\theta}(Y_{i-1}|Y_i) \propto \prod_{i=k+1, i \notin A}^m p^{\theta}(Y_{i-1}|Y_i),$$

where the proportionality follows from the fact that we can ignore the probability of the transitions produced by alt under Assumption 1. By definition, this is proportional to  $p^{\theta, \text{alt}}(Y_m, Y_{m-1}, \dots, Y_{k+1}, Y_k)$  up to a normalization constant independent of  $\theta$ .

#### <span id="page-35-0"></span>C.2.6 In-context learning

Now, we will specialize our framework to the case of in-context learning. As in [XRLM22], we assume that the language model is given inputs of the form  $[x_1, y_1, o, x_2, y_2, o, \dots, x_t, y_t, o, x_{t+1}]$ , where  $x_1$  is the input,  $y_1$  is the output, and o is a delimiter token that separate different in-context samples from each other. We assume that the transitions  $y_i \to o \to x_{i+1}$  are sampled by some alternative probability distribution  $P_{\text{other}}(\cdot|\cdot)$ . We require that  $P_{\text{other}}$  selects the  $x_i$  i.i.d.

<span id="page-36-2"></span>**Assumption 2.** The distribution of  $P_{other}(x_{t+1}|x_1, y_1, o, x_2, y_2, o, ..., x_t, y_t, o) = P_{other}(x_1)$ .

Then we assume that the transitions  $x_i \to y_i$  are generated by some  $\theta^* \in \Theta$ , which does not depend on any of the previous tokens before the delimiter.

<span id="page-36-3"></span>**Assumption 3.** (Well-specification) There exists some  $\theta^* \in \Theta$  such that  $y_i$  is generated from  $y_i \sim P^{\leftarrow}(\cdot|[x_1,y_1,o_1,\ldots,o_{i-1},x_i],\theta^*)$ .

<span id="page-36-4"></span>**Assumption 4.** For all  $\theta \in \Theta$ , we have  $P^{\leftarrow}(\cdot|[x_1,y_1,o_1,\ldots,o_{i-1},x_i],\theta) = P^{\leftarrow}(\cdot|x_i,\theta)$ .

We also assume statistical separation of  $\theta^*$  from  $\Theta - \{\theta^*\}$  in terms of Hellinger distance.

<span id="page-36-5"></span>**Assumption 5.** Let  $p_{(x,y)}^S$  for  $S \subset \Theta$  be the distribution of  $(x_1, y_1, o)$  where  $x_1 \sim P_{other}(\cdot | o)$  and  $y_1 \sim p^S(\cdot | x_1)$ . There exists  $\delta > 0$  such that  $H^2(p_{(x,y)}^{\Theta - \{\theta^*\}}, p_{(x,y)}^{\theta^*}) \geq \delta$ .

<span id="page-36-1"></span>**Example 11.** Let  $T \ge \ln\left(\frac{1-\delta/2}{\epsilon}\right)$ . Under Assumptions 1, 2, 3, 4, and 5, we have

$$TV([x_1,...,x_{T+1},y_{T+1}],[x_1,...,x_{T+1},\tilde{y}_{T+1}]) \lesssim \epsilon/w_{\theta^*}.$$

*Proof.* It suffices to upper bound  $1 - \frac{1}{2}H^2(P_{3T}^{\theta,\text{alt}},P_{3T}^{\Theta-\{\theta\},\text{alt}})$  by  $O(\epsilon)$ . First observe that the distribution  $p^{S,\text{alt}}$  for  $S \subset \Theta$  factors along the delimiters by a factor independent of S using Assumptions 2 and 4, so we have

$$P_{3t}^{S,\text{alt}}(x_1, y_1, o_1, x_2, y_2, o_2, \dots, x_t, y_t, o_t) \propto \prod_{i=1}^t P_3^{S,\text{alt}}(x_i, y_i, o_i).$$

Using the tensorization property of Hellinger distance and our definition of T,  $\delta$ , we have

$$1 - \frac{1}{2}H^2(P_{3T}^{\theta,\text{alt}}, P_{3T}^{\Theta - \{\theta\},\text{alt}}) \lesssim \left[1 - \frac{1}{2}H^2(P_3^{\theta,\text{alt}}, P_3^{\Theta - \{\theta\},\text{alt}})\right]^T \leq \epsilon. \quad \Box$$

## <span id="page-36-0"></span>**D** Experiments

## **D.1** Jailbreak Experiments

## D.1.1 Reproducing critical windows for jailbreaks from existing papers

Existing work has already identified the presence of critical windows in the domain of jailbreaks. Here we present critical windows for a simplified prefill jailbreak based on the prefill attack [Hai24b] and repeating token jailbreak [NCH+23] for LLAMA-3.1-8B-Instruct. In the first figure, we plot the probability of the model giving a harmful response, computed using the StrongReject Gemma 7b auditor from [SLB+24], as a function of the fraction of the phrase Sure, here is how to appended to the front of the model's generation. We can see that there is a large jump in the attack success rate after only including a few tokens in the prefix. The second figure is a reproduction of Figure 12 from [NCH+23]. It shows that the probability of repeating the next token increases substantially as the first few tokens are included a few times in the context.

![](_page_37_Figure_0.jpeg)

Figure 7: Examples of critical windows in jailbreaks for LLAMA-3.1-8B-Instruct. **Left:** Critical window for a prefill jailbreak [[Hai24b](#page-22-1)]. **Right:** Critical window for a repeating token jailbreak [[NCH](#page-24-14)+23].

## <span id="page-37-0"></span>**D.1.2 Experimental details from jailbreak**

Now we apply our theory to develop a new jailbreak detection method, based on a likelihood ratio between an aligned and unaligned model. Intuitively, our theory states that when the unaligned component assigns a high probability to the text compared to the entire model, the model is likely to be jailbroken. We use a LLAMA-3.1-8B-Instruct model jailbroken with LoRA to not refuse harmful prompts [[gri24](#page-22-13)] as a proxy for the unaligned model. We evaluate these different methods on a dataset of jailbreaks and benign prompts from [[BSS](#page-21-10)+24].

**Dataset.** We use the same dataset as [[BSS](#page-21-10)+24] but provide details here for completeness. The benign dataset consists of inputs from UltraChat [[DCX](#page-21-11)+23], a large dialogue dataset, and Xstest [[RKV](#page-24-15)+24], which contains benign queries that are often incorrectly refused by language models. The benign queries are filtered to ensure that LLAMA-3.1-8B-Instruct does not refuse any of them. The dataset of harmful prompts is based off of the Circuit Breakers dataset [[ZPW](#page-25-9)+24]. The datasets include the following jailbreaking methods from the extant literature: PAIR [[CRD](#page-21-12)+23], AutoDAN [[LXCX24](#page-23-10)], Many-Shot Jailbreaking (MSJ) [[ADS](#page-20-8)+24], Multi-Turn Attacks [[LHS](#page-23-11)+24, [Hai24a](#page-22-14)], Prefill, GCG [[ZWC](#page-25-10)+23], and other Misc. attakcs from [[WHS23](#page-25-11)]. For each jailbreaking method, it is applied to a prompt from the Circuit Breaker dataset and evaluated to see if the generation from LLAMA-3.1-8B-Instruct is helpful and harmful, as determined by the StrongReject jailbreaking classifier [[SLB](#page-25-8)+24]).

**Evaluation Metric.** As is standard in the jailbreak detection literature [[BSS](#page-21-10)+24], we report the recall at the false positive rate at 0.01.

Table [2](#page-38-0) displays the recall and several other baselines. Crucially, the log likelihood ratio methods does obtain recall *>* 1 for 5 different categories of jailbreaks. While our methods do perform worse than existing methods, it is important to note that they still work and that their poor performance could be explained by the fact that we have to use a proxy for the unaligned mode of the model.

<span id="page-38-0"></span>Table 2: Recall (FPR=0.01) for our likelihood ratio threshold, a perplexity threshold [AK23], and a MLP-based detector trained on activations [BSS<sup>+</sup>24] for predicting different jailbreaks. prompt/gen denote the logprobs of the prompt and generation, respectively.

|                                                                                               | AutoDAN | GCG   | Multi-Turn | Misc  | MSJ   | Pair  | Prefill |
|-----------------------------------------------------------------------------------------------|---------|-------|------------|-------|-------|-------|---------|
| $\log p_{\mathrm{prompt}}^{\mathrm{unaligned}} - \log p_{\mathrm{prompt}}^{\mathrm{aligned}}$ | 0.000   | 0.000 | 0.028      | 0.000 | 0.063 | 0.000 | 0.077   |
| $\log p_{\rm gen}^{\rm unaligned} - \log p_{\rm gen}^{\rm aligned}$                           | 0.082   | 0.030 | 0.000      | 0.100 | 0.000 | 0.061 | 0.051   |
| $\log p_{\mathrm{prompt}}^{\mathrm{aligned}}$                                                 | 0.000   | 0.576 | 0.056      | 0.063 | 0.013 | 0.000 | 0.077   |
| $\log p_{\rm gen}^{\rm aligned}$                                                              | 0.205   | 0.150 | 0.570      | 0.200 | 0.006 | 0.015 | 0.416   |
| MLP                                                                                           | 1.00    | 0.956 | 0.873      | 0.663 | 1.00  | 0.833 | 1.00    |

## <span id="page-38-1"></span>D.2 Chain of thought experiments

### **D.2.1** Experimental details

We describe the prompts we used in our evaluation of different datasets. The system prompt for all datasets is Produce a correct solution to the following /TASK/ question., where /TASK/ is the type of question of the dataset, i.e. science, math, or logic. For each question, we create a user prompt by appending Think of the /TASK/ question thoroughly step by step. Please only respond with the answer after reasoning thoroughly. in front of the question. Once the model completes its generation (max generation length set to 2048 and default sampling parameters), we append the user prompt Given all of the above, what's the single, most likely answer? Your answer should have the format "The answer is ANSWER", where ANSWER is your answer. for the multiple choice benchmarks and Given all of the above, what's the single, most likely answer? Simplify it completely. Your answer should have the format "The answer is \$ANSWER\$", where ANSWER is your answer in LaTeX. Note that when we ask the model for the final answer, we set the temperature to 0.

Across three different models (LLAMA-3.1-8B-Instruct, Phi-3-7B-Instruct, and Qwen-2.5-7B-Instruct), we find that conditioned on the occurrence of critical windows, the model generation is significantly less accurate compared to generations without critical windows.

<span id="page-39-1"></span>![](_page_39_Figure_0.jpeg)

Figure 8: Left-to-right: LLAMA-3.1-8B-Instruct, Phi-3-7B-Instruct, and Qwen-2.5-7B-Instruct barplots of original generation accuracy for generations with (≥ 0.5 jump in probability) and without critical windows (*<* 0.5).

<span id="page-39-0"></span>![](_page_39_Figure_2.jpeg)

Figure 9: Probability that the answer is the same as a function of the percentage of the generation remaining for different math and reasoning benchmarks for Phi-3-7B-Instruct. Highlighted are generations with a 50% increase and no 30% decrease in the probability over one step.

<span id="page-40-0"></span>![](_page_40_Figure_0.jpeg)

Figure 10: Probability that the answer is the same as a function of the percentage of the generation remaining for different math and reasoning benchmarks for Qwen-2.5-7B-Instruct. Highlighted are generations with a 50% increase and no 30% decrease in the probability over one step.

## **D.2.2 Comparison with directly answering for LLAMA-**3.1**-**8**B-Instruct**

<span id="page-40-1"></span>It is interesting to compare our approach in Figure [8](#page-39-1) with that from [[LCR](#page-23-4)+23], in which the model is directly asked for the answer (Figure [11\)](#page-40-1). We can see that directly answering has a lower consistency than resampling across different amounts of generation remaining. Interestingly, the discrepancy is much higher for the math-datasets (MATH, AQUA) compared to extremely small difference in consistency for the other reasoning datasets. This may be because it is difficult to *extract* the answer from the middle of a chain of thought for math datasets than for other datasets.

![](_page_40_Figure_4.jpeg)

Figure 11: Difference between resampling to obtain the answer versus directly asking the model for the answer (as in [[LCR](#page-23-4)+23]) for LLAMA-3.1-8B-Instruct.

## D.2.3 Examples of critical windows

We provide a few more examples of critical windows (bolded).

## LogiQA example (LLAMA-3.1-8B-Instruct)

Q: Recently, some countries in the world, especially the United States and Japan, have blamed their domestic economic mystery on China's fixed exchange rate system between the RMB and the US dollar. Some domestic manufacturing companies have caused pressure. The following are all questions about the above views, except? (A) Japan has been in recession for more than ten years. At that time, China's merchandise exports were very small, and the renminbi was very weak. (B) The US economic recession was mainly caused by the destruction of the myth of the Internet economy, and the "9.11" terrorist attacks exacerbated this recession. (C) China's GDP accounts for only 3.9 of the world's total, foreign trade is less than 5 of the world's, and it does not have the ability to affect the world market price and supply and demand. (D) Consumers in Western countries have benefited greatly from cheap goods in China.

A: The question is asking which of the following is NOT a view about the views that the countries have blamed their domestic economic mystery on China's fixed exchange rate system between the RMB and the US dollar. We can analyze each choice one by one. Choices A, B, and D are all views that the countries have blamed their domestic economic mystery on China's fixed exchange rate system between the RMB and the US dollar. Choice C is an argument that China's fixed exchange rate system between the RMB and the US dollar has no effect on the world market price and supply and demand. So, it is not a view about the views that the countries have blamed their domestic economic mystery on China's fixed exchange rate system between the RMB and the US dollar. Therefore, the correct answer is C.

Correct answer: D

## MATH example (LLAMA-3.1-8B-Instruct)

**Q**: Let *a* and *b* be the solutions of the equation  $2x^2 - 10x + 5 = 0$ . What is the value of (2a - 3)(4b - 6)?

A: To find the value of (2a-3)(4b-6), we need to first find the values of a and b by solving the equation  $2x^2-10x+5=0$ . We can use the quadratic formula to solve for a and b:  $x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}$ . In this case, a=2, b=-10, and c=5. Plugging these values into the quadratic formula, we get:  $x=\frac{-(-10)\pm\sqrt{(-10)^2-4(2)(5)}}{2(2)}$   $x=\frac{10\pm\sqrt{100-40}}{4}$   $x=\frac{10\pm\sqrt{60}}{4}$   $x=\frac{10\pm2\sqrt{15}}{4}$  So, the solutions are  $a=\frac{10+2\sqrt{15}}{4}$  and  $b=\frac{10-2\sqrt{15}}{4}$ . Now that we have the values of a and b, we can substitute them into the expression (2a-3)(4b-6):  $(2a-3)(4b-6)=(2(\frac{10+2\sqrt{15}}{2}-3)(\frac{10-2\sqrt{15}}{4})-3)(4(\frac{10-2\sqrt{15}}{4})-6)$  Simplifying the expression, we get:  $(2a-3)(4b-6)=(\frac{10+2\sqrt{15}}{2})(\frac{10-2\sqrt{15}-6}{1})$  Now, we can simplify the expression further by multiplying the numerators and denominators:  $(2a-3)(4b-6)=\frac{(4+2\sqrt{15})(4-2\sqrt{15})}{2}$  Using the difference of squares formula  $(a+b)(a-b)=a^2-b^2$ , we can simplify the numerator:  $(2a-3)(4b-6)=\frac{4^2-(2\sqrt{15})^2}{2}$   $(2a-3)(4b-6)=\frac{16-60}{2}$  Simplifying the expression further, we get:  $(2a-3)(4b-6)=\frac{4^2-(2\sqrt{15})^2}{2}$  (2a-3)(4b-6)=-22 Therefore, the value of (2a-3)(4b-6) is -22.

Correct answer: -22

### ARC Easy example (LLAMA-3.1-8B-Instruct)

- **Q**: Why are diseases transmitted through the bites of infected animals not considered contagious diseases?
- (A) The infected animals do not show symptoms. (B) The person in direct contact is the only one infected.
- (C) Humans do not become sick from animal diseases. (D) The diseases are treatable with antibiotics.

**A**: To answer this question, we need to understand the definitions of contagious and non-contagious diseases. Contagious diseases are those that are spread through direct or indirect contact with an infected person. Non-contagious diseases, on the other hand, are spread through other means, such as vectors like insects or animals. Step 1: Consider the definition of contagious diseases. Contagious diseases are spread through direct or indirect contact with an infected person. Step 2: Consider the definition of non-contagious diseases. Non-contagious diseases are spread through vectors like insects or animals. Step 3: Analyze the given options. Option A states that the infected animals do not show symptoms. This is not necessarily true, as some infected animals may show symptoms. Option B states that the person in direct **contact is the only one infected, which is not a characteristic of contagious diseases. Option D states that the diseases are treatable with antibiotics, which is not relevant to the mode of transmission. Step 4: Based on the analysis, the correct** answer is the one that accurately describes the mode of transmission of the diseases. The final answer is: *C* .

**Correct answer**: (B)

#### AQUA-RAT example (Phi-3-7B-Instruct)

**Q**: At a conference, one team is made up of 4 men and 4 women. Four presenters are chosen to present the team's findings in front of the entire conference. How many different groups of presenters can be chosen from the team if a team cannot be composed of men only or women only? (Two groups of presenters are considered different if at least one presenter is different.) A)120 B)19 C)180 D)420 E)460

**A**: 1. There are 8 people in the team, 4 men and 4 women. 2. We need to choose 4 presenters for the conference, without having an all-male or all-female group. 3. We can choose 4 out of 8 using combinations: C(8,4) = 8! / (4! \* (8-4)!) = 70 4. We'll subtract two groups that aren't allowed: all men and all women groups. So we'll subtract combinations of just men (4C4) and just women (4C4). So, we have: In all, we need to choose from combinations of (8C4 - 4C4 - 4C4) which equals: 70 (total combination for 4 people) - 1 (all men group) - 1 (all women group) = 68 5. However, we should also consider the mixed group of 2 men and 2 women which would be another prohibited combination. Since the two men and two women groups would be the same, we subtract: C(4,2) \* C(4,2) = 6 6. So, finally, our calculation will be: 68 - **6 = 62. The correct answer among the choices we have isn't amongst those. There seems to be a mistake in the question or the given options. But close to the choices, an answer of 420 could seem interconnected if we consider combinations when men and women are mixed in all possible ways which is not the deducted understanding in our solution. Correct answer: E)**420 should be revised or further clarified in the question context. Otherwise, based on our understanding of how combinations are applied in scenarios like this, none of these options is quite correct. The answer is E)420.

**Correct answer**: B

## LogiQA example (Phi-3-7B-Instruct)

**Q**: The person in charge of the relevant department said that pilots will be carried out in some areas this year to prepare for the system and policy preparations for the comprehensive clean-up of "small property houses", and require all localities to confirm the registration and certification of rural collective land. All small-property houses are not subject to registration and are not protected by law. Therefore, this newly built house in Hexi Village is not protected by law. Which of the following is true will most weaken the above argument. (A) This newly built house in Hexi Village has been tacitly approved by relevant departments. (B) This newly built house in Hexi Village is a small property house. (C) This newly built house in Hexi Village is built on rural collective land. (D) Some of the newly built houses in Hexi Village are not small property houses.

**A**: The statement implies that small property houses are not protected by law and can be demolished if necessary in the preparations for the system and policy preparations for the comprehensive clean-up. -We have to find the option that weakens this argument the most. Option A suggests that the department has tacitly approved the newly built house. This weakens the argument, **but it can still be argued that the house is subject to demolition if the comprehensive clean-up policy is implemented. Option B states that this newly built house in Hexi Village is a small property house. This directly contradicts the argument that all small-property houses are not protected by law** and not subject to registration and certification. Therefore, this house would not be demolished and would be protected by law. Option C says that the house is built on rural collective land. This doesn't disprove the argument – it might still be a small-property house without legal protection. Option D makes only some of the newly built houses in Hexi Village not small property houses. This doesn't address the argument that all small-property houses are not protected by law and not subject to registration, but only carries an exceptional circumstance. Thus, the option that most weakens the argument is (B) as it directly contradicts the statement by implying that the specific house is protected by law. The answer is B.

**Correct answer**: D

## MATH example (Qwen-2.5-7B-Instruct)

**Q**: Carson flips over the cards of a standard 52-card deck one at a time. What is the probability that he flips over the ace of spades before any face card (jack, queen or king)?

**A**: 1. First, we need to identify the relevant cards for our problem. The ace of spades (A♠) and the face cards (J, Q, K) are the ones we're focusing on. There are 4 face cards in a standard deck of 52 cards (12 face cards total, but we only consider 4 for this problem). 2. There are 52 cards in total, and 5 of these (A♠, J♠, Q♠, K♠, J♥, Q♥, K♥, J♦, Q♦, K♦, J♣, Q♣, K♣) are either the ace of spades or a face card. Therefore, there are 47 other cards that are neither the ace of spades nor face cards. 3. The problem can be simplified by considering the probability of the ace of spades coming before any face card in a sequence of drawing these 9 cards (the ace of spades and the **8 face cards). 4. We can think of the problem as a sequence of drawing 9 cards, where we are interested in the position of the ace of spades relative to the face cards. The ace of spades can come in any of the 9 positions, and we want to find the probability that it comes before any of the face cards. 5. For the ace of spades to come before any** face card, it must be in one of the first 5 positions (since there are 4 face cards, and the ace of spades must come before any of them). There are 9 positions in total, and 5 of these are favorable. 6. Therefore, the probability that the ace of spades comes before any face card is the ratio of the number of favorable outcomes to the total number of

outcomes, which is <sup>5</sup> 9 . So, the probability that Carson flips over the ace of spades before any face card is <sup>5</sup>

9 .

**Correct answer**: 13

#### AQUA-RAT example (Qwen-2.5-7B-Instruct)

**Q**: ABCDE is a regular pentagon with F at its center. How many different quadrilaterals can be formed by joining 4 of the points A,B,C,D,E and F? A)12 B)10 C)5 D)15 E)20

A: We need to form quadrilaterals by selecting 4 points from the 6 points (A, B, C, D, E, F) where F is the center of the regular pentagon. To form a quadrilateral, we need to select 4 points out of these 6. However, we need to ensure that the quadrilateral is valid. A key observation here is **that the center F must be included in the selection of 4 points because otherwise, we would only have points on the perimeter of the pentagon, which would not form a closed quadrilateral with a center point. So, we need to choose 3 points from the 5 points A, B, C, D, E to form the quadrilateral along with F. The number of ways to choose 3 points from 5 is given by the combination formula \binom{5}{3}:** 

$$\binom{5}{3} = \frac{5!}{3!(5-3)!} = \frac{5 \times 4}{2 \times 1} = 10$$

Thus, there are 10 different ways to choose 3 points from the 5 perimeter points, and each choice, along with the center F, forms a valid quadrilateral. Therefore, the correct answer is  $\boxed{10}$ .

Correct answer: D

## MATH example (Phi-3-7B-Instruct)

**Q**: Square ABCD has side lengths of 13 units. Point E lies in the interior of the square such that AE = 5 units and BE = 12 units. What is the distance from E to side AD?

A: 1. Since ABCD is a square, we know that all sides are equal and the diagonals bisect each other at right angles. 2. Drawing a diagram, we can see that  $\triangle ABE$  and  $\triangle BDE$  are right triangles with right angles at point B. 3. We can use Pythagoras' theorem to find DE:  $DE = \sqrt{BE^2 - AE^2} = \sqrt{12^2 - 5^2} = \sqrt{144 - 25} = \sqrt{119} \approx 10.91$  units. 4. Since ABCD is a square, AD = 13 units. So, the distance from E to AD is AD = 13 - 5 = 8 units. The distance from E to side AD is 8 units. The answer is 8.

Correct answer:  $\frac{25}{13}$ 

### MMLU example (Qwen-2.5-7B-Instruct)

**Q**: A store owner noticed that an expensive electronic device was missing from the store. He assumed that a shoplifter had taken it. Later that day, the store owner was cleaning up the small room in the back of the store that the employees of the store used to store their personal items while they were at work. The store owner was straightening up the coat rack in the back room, when he accidentally knocked one of the employee's coats off the hanger. As the coat fell onto the floor, the missing electronic device fell out of the pocket of the coat. It was still in its original packaging, and the store owner checked the serial number on the package to confirm that it was the device missing from the store's shelves. The store owner knew that the coat belonged to a woman who worked as a cashier in the store. The store owner felt bad, but decided that he had no choice but to call the police. Two police officers arrived at the store. They gathered information from the store owner, who signed a criminal complaint against the woman. The officers then discussed the matter with the woman. Although she denied stealing anything, the officers arrested her and informed her that they would need to take her to the police station. After the police took the woman away, another employee of the store found out that the woman had been arrested. That employee immediately went to the store owner and admitted that he was actually the one who had stolen the device, and that he had hidden it in the woman's coat, intending to retrieve it later before the end of the woman's work shift. The store owner promptly called the police and told them that there had been a mistake and that the theft charge against the woman was invalid. The police released the woman, who spent a total of approximately 30 minutes in police custody. If the woman asserts a claim for false imprisonment against the arresting officers, will the woman prevail? (A) Yes, because the woman had not committed a crime at the time she was arrested. (B) Yes, because the woman was arrested and held in custody. (C) No, because the woman was detained for only 30 minutes. (D) No, because the officers had probable cause to arrest the woman based on the store owner's complaint and honest belief that the woman had stolen the device.

**A**: 1) The key issue here is whether the woman's arrest was lawful. 2) For an arrest to be lawful, there must be probable cause to believe the person committed a crime. 3) The store owner believed the woman had stolen the device based on the device falling out of her coat. **4) However, it was later revealed that the device was actually planted in her coat by another employee. 5) This means the store owner's belief was mistaken, and there was no probable cause for the arrest.** 6) False imprisonment occurs when a person is unlawfully detained. 7) The arrest was unlawful because it was based on a false belief. 8) Therefore, the woman's claim for false imprisonment is valid because she was unlawfully detained. Given these points, the correct answer is: (A) Yes, because the woman had not committed a crime at the time she was arrested. This answer is correct because the arrest was based on a mistaken belief and there was no probable cause, which means the arrest was unlawful, and the woman was falsely imprisoned. The answer is (A).

**Correct answer**: D

## **D.2.4 Varied temperature sampling**

In our above experiments with critical windows, we have been generating model responses at their default sampling temperatures: LLAMA-3.1-8B-Instruct: 0.6, Phi-3-7B-Instruct: 1.0, and Qwen-2.5-7B-Instruct: 0.7. A natural question to ask is whether the sampling temperature affects the presence or strength of critical windows across these models. At one extreme, sampling at infinite temperature exhibits no critical windows since each token is generated uniformly at random. At the other extreme, zero temperature sampling is deterministic, so there is only one mode of behavior and critical windows again cannot exist. At intermediate temperatures, decreasing sampling temperature reduces token variance, so the sharpness of transition from general mixture to target submixture can be heightened. At the same time, reduced sampling variance also reduces the likelihood of transitioning between modes as well, leaving the overall effect on critical windows unclear.

First, we report the frequency of critical windows across the different temperatures for the MATH

dataset: 0.1,0.4,0.7, and 1.0. Then, we report the average size of a critical window on 400 samples from MATH, running the forward-reverse experiment on each sample 25 times and taking the max consecutive jump as the critical window size for that problem. From Figures [12](#page-46-0) and [13,](#page-46-1) each model has its own relationship between temperature and critical window frequency and size, with no clear definite relationship across all models.

<span id="page-46-0"></span>![](_page_46_Figure_1.jpeg)

<span id="page-46-1"></span>Figure 12: Temperature versus critical window frequency for LLAMA-3.1-8B-Instruct, Phi-3-7B-Instruct, and Qwen-2.5-7B-Instruct.

![](_page_46_Figure_3.jpeg)

Figure 13: Temperature versus mean critical window size for LLAMA-3.1-8B-Instruct, Phi-3-7B-Instruct, and Qwen-2.5-7B-Instruct.