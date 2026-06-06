---
type: paper-fulltext
slug: diffusion-sampling-with-momentum-for-mitigating-divergence-artifacts
arxiv_id: null
source_path: /Users/saris.kia.adm/.paper-scholar/diffusion-sampling-with-momentum-for-mitigating-divergence-artifacts/01-diffusion-sampling-momentum-iclr2024.md
paper: "[[diffusion-sampling-with-momentum-for-mitigating-divergence-artifacts]]"
---
## Diffusion Sampling with Momentum for Mitigating Divergence Artifacts

Suttisak Wizadwongsa

VISTEC, Thailand suttisak.w\_s19@vistec.ac.th

#### Worameth Chinchuthakun

Tokyo Institute of Technology, Japan chinchuthakun.w.aa@m.titech.ac.jp

Pramook Khungurn

pixiv Inc. pramook@gmail.com

Amit Raj Google amitrajs@google.com

Supasorn Suwajanakorn VISTEC, Thailand supasorn.s@vistec.ac.th

## Abstract

Despite the remarkable success of diffusion models in image generation, slow sampling remains a persistent issue. To accelerate the sampling process, prior studies have reformulated diffusion sampling as an ODE/SDE and introduced higher-order numerical methods. However, these methods often produce *divergence* artifacts, especially with a low number of sampling steps, which limits the achievable acceleration. In this paper, we investigate the potential causes of these artifacts and suggest that the small stability regions of these methods could be the principal cause. To address this issue, we propose two novel techniques. The first technique involves the incorporation of Heavy Ball (HB) momentum, a wellknown technique for improving optimization, into existing diffusion numerical methods to expand their stability regions. We also prove that the resulting methods have first-order convergence. The second technique, called Generalized Heavy Ball (GHVB), constructs a new high-order method that offers a variable trade-off between accuracy and artifact suppression. Experimental results show that our techniques are highly effective in reducing artifacts and improving image quality, surpassing state-of-the-art diffusion solvers on both pixel-based and latent-based diffusion models for low-step sampling. Our research provides novel insights into the design of numerical methods for future diffusion work.

## 1 Introduction

Diffusion models [\[1,](#page-9-0) [2\]](#page-9-1) are a type of generative models that has garnered considerable attention due to their remarkable image quality. Unlike Generative Adversarial Networks (GANs) [\[3\]](#page-9-2), which may suffer from mode collapse and instabilities during training, diffusion models offer reduced sensitivity to hyperparameters [\[1,](#page-9-0) [4\]](#page-9-3) and improve sampling quality [\[5\]](#page-9-4). Additionally, diffusion models have been successfully applied to various image-related tasks, such as text-to-image generation [\[6\]](#page-9-5), imageto-image translation [\[7\]](#page-9-6), image composition [\[8\]](#page-10-0), adversarial purification [\[9,](#page-10-1) [10\]](#page-10-2), super-resolution [\[11\]](#page-10-3), and text-to-audio conversion [\[12\]](#page-10-4).

One significant drawback of diffusion models, however, is their slow sampling speed. This is because the sampling process involves a Markov chain that requires a large number of iterations to generate high-quality results. Recent attempts to accelerate the process include improvements to the noise schedule [\[13,](#page-10-5) [14\]](#page-10-6) and network distillation [\[15–](#page-10-7)[17\]](#page-10-8). Fortunately, the sampling process can be represented by ordinary or stochastic differential equations, and numerical methods can be used to reduce the number of iterations required. While DDIM [\[2\]](#page-9-1), a 1st-order method, is the most commonly used approach, it still requires a considerable number of iterations. Higher-order numerical methods,

<span id="page-1-0"></span>![](_page_1_Figure_0.jpeg)

Figure 1: We demonstrate the occurrence of divergence artifacts in DPM-Solver++[\[19\]](#page-10-9) and PLMS4[\[20\]](#page-10-10) with 15 sampling steps, where s denotes the text-guidance scale. By integrating HB momentum into these methods, we effectively mitigate the artifacts. Additionally, we compare the results with our GHVB 1.5 method. Prompt: "photo of a girl face" (a) Realistic Vision v2.0[\[22\]](#page-10-11), (b) Anything Diffusion v4.0[\[23\]](#page-10-12), (c) Deliberate Diffusion[\[24\]](#page-10-13)

such as DEIS [\[18\]](#page-10-14), DPM-Solver [\[19\]](#page-10-9), and PLMS [\[20\]](#page-10-10), have been proposed to generate high-quality images in fewer steps. However, these methods begin to produce artifacts (see Figure [1\)](#page-1-0) when the number of steps is decreased beyond a certain value, thereby limiting how much we can reduce the sampling time.

In this study, we investigate the potential causes of these artifacts and found that the narrow stability region of high-order numerical methods can cause solutions to diverge, resulting in divergence artifacts. To address this issue and enable low-step, artifact-free sampling, we propose two techniques. The first technique involves incorporating Polyak's Heavy Ball (HB) momentum [\[21\]](#page-10-15), a well-known technique for improving optimization, into existing diffusion numerical methods. This approach effectively reduces divergence artifacts, but its accuracy only has first order of convergence. In this context, the accuracy measures how close the approximated, low-step solution is to the solution computed from a very high-step solver (e.g., 1,000-step DDIM). The second technique, called Generalized Heavy Ball (GHVB), is a new high-order numerical method that offers a variable tradeoff between accuracy and artifact suppression. Both techniques are training-free and incur negligible additional computational costs. Figure [1](#page-1-0) demonstrates the superiority of both techniques in reducing divergence artifacts compared to previous diffusion sampling methods. Furthermore, our experiments show that our techniques are effective on both pixel-based and latent-based diffusion models.

The paper is structured as follows. Section [2](#page-1-1) covers background and related work on the diffusion sampling process in differential equation forms and stability region. Section [3](#page-3-0) analyzes visual artifacts in diffusion sampling and establishes a connection to the stability region of the solver. Section [4](#page-4-0) proposes a technique to apply momentum to existing numerical methods, as well as a technique that generalizes momentum to high-order numerical methods. Section [5](#page-7-0) presents experiments and ablation studies. Finally, Section [6](#page-9-7) concludes and discusses the implications and impacts of our work.

## <span id="page-1-1"></span>2 Background

This section first presents the theoretical foundation of diffusion sampling when modeled as an ordinary differential equation (ODE) and related numerical methods. Second, we discuss ODE forms for guided diffusion sampling and prior splitting numerical methods. Third, we cover the concept of stability region, which is our primary analysis tool.

#### 2.1 Diffusion in ODE Form

Modeling diffusion sampling as an ODE is commonly based on the non-Markovian sampling of Denoising Diffusion Implicit Model (DDIM) [2]. DDIM is well-known for its simplicity, as it enables deterministic sampling after a random initialization, given by:

<span id="page-2-0"></span>
$$x_{t-1} = \sqrt{\frac{\alpha_{t-1}}{\alpha_t}} \left( x_t - \sqrt{1 - \alpha_t} \epsilon_{\theta}(x_t, t) \right) + \sqrt{1 - \alpha_{t-1}} \epsilon_{\theta}(x_t, t). \tag{1}$$

Here,  $\epsilon_{\theta}(x_t,t)$  is a neural network that predicts noise, with learnable parameters  $\theta$  that take the current state  $x_t$  and time t as input. The parameter  $\alpha_t$  is a schedule that controls the degree of diffusion at each time step. Previous research has shown that DDIM 1 can be rewritten into an ODE, making it possible to use numerical methods to accelerate the sampling process. Two ODEs have been proposed in the literature:

<span id="page-2-2"></span><span id="page-2-1"></span>
$$\frac{d\bar{x}}{d\sigma} = \bar{\epsilon}(\bar{x}, \sigma), \qquad (2) \qquad \frac{d\tilde{x}}{d\tilde{\sigma}} = s(\tilde{x}, \tilde{\sigma}), \qquad (3)$$

Equation 2 can be obtained by re-parameterizing  $\sigma = \sqrt{1-\alpha_t}/\sqrt{\alpha_t}$ ,  $\bar{x} = x_t/\sqrt{\alpha_t}$ , and  $\bar{\epsilon}(\bar{x},\sigma) = \epsilon_{\theta}(x_t,t)$ . These transformations are widely used in various diffusion solvers [18–20, 25]. If  $\epsilon_{\theta}(x_t,t)$  is a sum of multiple terms, such as in guided diffusion, we can easily split Equation 2 and solve each resulting equation separately, as demonstrated in [26]. Another ODE, given by Equation 3, can be derived by defining  $\tilde{\sigma} = \sqrt{\alpha_t}/\sqrt{1-\alpha_t}$  and  $\tilde{x} = x_t/\sqrt{1-\alpha_t}$ , where  $s(\tilde{x},\tilde{\sigma}) = (x_t - \sqrt{1-\alpha_t}\epsilon_{\theta}(x_t,t))/\sqrt{\alpha_t}$ , which is an approximation of the final result. This ODE has the advantage of keeping the differentiation bounded within the pixel value range in pixel-based diffusion. Recent research on DPM-Solver++ [27] has shown that Equation 3 outperforms Equation 2 in many cases.

#### 2.2 Guided Diffusion Sampling

Guided diffusion sampling is a widely used technique for conditional sampling, such as text-to-image and class-to-image generation. There are main two approaches for guided sampling:

Classifier guidance [5, 2] uses a pre-trained classifier model  $p_{\phi}(c \mid x_t, t)$  to define the conditional noise prediction model at inference time:

$$\hat{\epsilon}(x_t, t \mid c) = \epsilon_{\theta}(x_t, t) - s\nabla \log p_{\theta}(c \mid x_t, t), \tag{4}$$

where s > 0 is a "guidance" scale. The model can be extended to accept any guidance function, such as CLIP function [28] for text-to-image generation [29]. This approach only modifies the sampling equation at inference time and thus can be applied to a trained diffusion model without retraining.

Classifier-free guidance, proposed by Ho et al. [30], trains a conditional noise model  $\epsilon_{\theta}(x_t, t \mid c)$  to generate data samples with the label c:

$$\hat{\epsilon}(x_t, t \mid c) = \epsilon_{\theta}(x_t, t \mid \phi) + s(\epsilon_{\theta}(x_t, t \mid c) - \epsilon_{\theta}(x_t, t \mid \phi)), \tag{5}$$

where  $\phi$  is a null label to allow for unconditional sampling. The sampling equations in both approaches can be expressed as a "guided ODE" of the form

<span id="page-2-3"></span>
$$\frac{d\bar{x}}{d\sigma} = \bar{\epsilon}(\bar{x}, \sigma) + g(\bar{x}, \sigma),\tag{6}$$

where  $g(\bar{x},\sigma)$  represents a guidance function. To accelerate guided diffusion sampling, splitting numerical methods have been proposed, such as Lie-Trotter Splitting (LTSP) [26]. This method divides Equation 6 into two subproblems, i)  $\frac{dy}{d\sigma} = \bar{\epsilon}(y,\sigma)$  and ii)  $\frac{dz}{d\sigma} = g(z,\sigma)$ , but can only apply high-order numerical methods to the first equation while resorting to the Euler method for the second equation to avoid numerical instability. Higher-order splitting methods, such as Strang Splitting (STSP) [26], are also able to mitigate artifacts. However, these methods require solving the second equation twice per step, which is comparable to increasing the total sampling step to avoid artifacts. Both approaches require non-negligible computation.

#### <span id="page-2-4"></span>2.3 Stability Region

The stability region is a fundamental concept in numerical methods for solving ODEs. It determines the step sizes that enable numerical approximations to converge. To illustrate this concept, let us consider the Euler method, a simple, first-order method for solving ODEs, given by

$$x_{n+1} = x_n + \delta f(x_n), \tag{7}$$

where  $x_n$  is the approximate solution and  $\delta$  is the step size. To analyze the stability of the Euler method, we can consider a test equation of the form  $x' = \lambda x$ , where  $\lambda$  is a complex constant. The solution of this test equation can be expressed as

$$x_{n+1} = x_n + \delta \lambda x_n = (1 + \delta \lambda) x_n = (1 + \delta \lambda)^{n+1} x_0,$$
 (8)

where  $x_0$  is the initial value. For the approximate solution to converge to the true solution, it is necessary that  $|1+\delta\lambda|\leq 1$ . Hence, the stability region of the Euler method is  $S=\{z\in\mathbb{C}:|1+z|\leq 1\}$  because if  $z=\delta\lambda$  lies outside of S, the solution  $x_n$  will tend to  $\pm\infty$  as  $n\to\infty$ .

In diffusion sampling, another common numerical solver is the Adams-Bashforth (AB) methods, also referred to as Pseudo Linear Multi-Step (PLMS). AB methods encompass the Euler method as its first-order special case (AB1), and the second-order AB2 is given by:

<span id="page-3-3"></span>
$$x_{n+1} = x_n + \delta \left( \frac{3}{2} f(x_n) - \frac{1}{2} f(x_{n-1}) \right).$$
 (9)

The stability regions of AB methods of various orders are derived in Appendix A. To visualize these regions, we use the boundary locus technique [31], which determines the boundaries of the stability regions as depicted in Figure 2. As the order of the method increases, the stability region decreases in size, and its boundary becomes more restrictive.

<span id="page-3-2"></span>![](_page_3_Figure_8.jpeg)

<span id="page-3-1"></span>Figure 2: Boundaries of stability regions of the first 4 Adams-Bashforth methods

### <span id="page-3-0"></span>3 Understanding Artifacts in Diffusion Sampling

One unique issue in diffusion sampling is the occurrence of "divergence" artifacts, which are characterized by regions with unrealistic, oversaturated pixels in the output. This problem typically arises due to several factors, including the use of high-order numerical solvers, too few sampling steps, or a high guidance scale. (See discussion in Appendix K). The current solution is to simply avoid these factors, albeit at the cost of slower sampling speed or less effective guidance. This section investigates the source of these artifacts, then we propose solutions that do not sacrifice sampling speed in the next section.

#### <span id="page-3-4"></span>3.1 Analyzing Diffusion Artifacts

We analyze the areas where divergence artifacts occur during sampling by examining the magnitudes of the latent variables in those areas. Specifically, we use Stable Diffusion [4], which operates and performs diffusion sampling on a latent space of dimension  $64 \times 64 \times 4$ , to generate images with and without artifacts by varying the number of steps. Then, we visualize each latent variable  $z \in \mathbb{R}^4$  in the  $64 \times 64$  spatial grid by subtracting the channel-wise mean and dividing by the channel-wise standard deviation, computed from the COCO dataset [32]. Figure 3 shows the magnitudes of the normalized latent variables after max pooling for visualization purposes.

We found that artifacts mainly appear in areas where the latent magnitudes are higher than usual. Note that images without artifacts can also have high latent magnitudes in some regions, although this is very rare. Conversely, when artifacts appear, those regions almost always have high magnitudes. In pixel-based diffusion models, the artifacts manifest directly as pixel values near 1 or 0 due to clipping, which can be observed in Figure 16 in Appendix H.

#### <span id="page-3-5"></span>3.2 Connection Between ODE Solver and Artifacts

We hypothesize that numerical instability during sampling is the cause of these visual artifacts. To see this mathematically, we analyze the ODE for diffusion sampling in Equation 2 using the problem

<span id="page-4-1"></span>![](_page_4_Figure_0.jpeg)

Figure 3: Comparison of generated images and latent variable magnitudes with and without artifacts, obtained using low and high sampling steps. Latent magnitude maps are max-pooled to 16x16, with brighter colors indicating higher values. These results suggest a relationship between artifacts and large latent magnitudes.

reduction technique for stiffness analysis [33]. Assuming that the effect of  $\sigma$  on the function  $\bar{\epsilon}$  is negligible, we use Taylor expansion to approximate the RHS of Equation 2, which yields

$$\frac{d\bar{x}}{d\sigma} = \nabla \bar{\epsilon}(x^*)(\bar{x} - x^*) + \mathcal{O}(\|\bar{x} - x^*\|^2). \tag{10}$$

Here,  $x^*$  denotes the converged solution that should not have any noise left (i.e.  $\bar{\epsilon}(x^*) = 0$ ), and  $\nabla \bar{\epsilon}(x^*)$  denotes the Jacobian matrix at  $x^*$ . As  $\bar{x}$  converges to  $x^*$ , the term  $\mathcal{O}(\|\bar{x} - x^*\|^2)$  becomes negligibly small, so we may drop it from the equation.

Let  $\lambda$  be an eigenvalue of  $\nabla \bar{\epsilon}(x^*)^T$  and v be the corresponding normalized eigenvector such that  $\nabla \bar{\epsilon}(x^*)^T v = \lambda v$ . We define  $u = v^T(\bar{x} - x^*)$  and obtain  $u' = \lambda u$  as our test equation. According to Section2.3, if  $\delta \lambda$  falls outside the stability region of a numerical method, the numerical solution to u may diverge, resulting in diffusion sampling results with larger magnitudes that later manifest as divergence artifacts. Therefore, when the stability region is too small, divergence artifacts are more likely to occur. Although some numerical methods have infinite stability regions, those used in diffusion sampling have only finite stability regions, which implies that the solution will always diverge if the step size  $\delta$  is sufficiently high. More details about the derivation can be found in Appendix B and a 2D toy example illustrating this effect is provided in Appendix C.

One possible solution to mitigate artifacts is to reduce the step size  $\delta$ , which shifts  $\delta\lambda$  closer to the origin of the complex plane. However, this approach increases the number of steps, making the process slower. Instead, we will modify the numerical methods to enlarge their stability regions.

#### <span id="page-4-0"></span>4 Methodology

This section describes two techniques for improving stability region and reducing divergence artifacts. Specifically, we first show how to apply Polyak's Heavy Ball Momentum (HB) to diffusion sampling, and secondly, how to generalize HB to higher orders. Our techniques are designed to be simple to implement and do not require additional training.

#### 4.1 Polyak's Heavy Ball Momentum for Diffusion Sampling

Recall that Polyak's Heavy Ball Momentum [21] is an optimization algorithm that enhances gradient descent  $(x_{n+1} = x_n - \beta_n \nabla f(x_n))$ . The method takes inspiration from the physical analogy of a heavy ball moving through a field of potential with damping friction. The update rule for Polyak's HB optimization algorithm is given by:

$$x_{n+1} = x_n + \alpha_n(x_n - x_{n-1}) - \beta_n \nabla f(x_n), \tag{11}$$

where α<sup>n</sup> and β<sup>n</sup> are parameters. We can apply HB to the Euler method [7,](#page-3-2) in which case we typically set α<sup>n</sup> = (1 − βn), to obtain

<span id="page-5-3"></span><span id="page-5-0"></span>
$$x_{n+1} = x_n + (1 - \beta_n)(x_n - x_{n-1}) + \delta \beta_n f(x_n), \tag{12}$$

and we may show that the numerical method above has the same order of convergence as the original Euler method. For simplicity, we assume that β<sup>n</sup> = β ∈ (0, 1], which is a constant known as the *damping coefficient.* Then, we can reformulate Equation [12](#page-5-0) as:

$$v_{n+1} = (1 - \beta)v_n + \beta f(x_n), \qquad x_{n+1} = x_n + \delta v_{n+1}, \tag{13}$$

Here, we may interpret x<sup>n</sup> as the heavy ball's position, and vn+1—the exponential moving average of f(xn)—as its velocity. We can see that position is updated with "displacement = time × velocity," much like in physics.

Consider a high-order method of the form xn+1 = x<sup>n</sup> + δ P<sup>k</sup> <sup>i</sup>=0 bif(xn−i). We can apply HB to it as follows:

$$v_{n+1} = (1 - \beta)v_n + \beta \sum_{i=0}^k b_i f(x_{n-i}), \qquad x_{n+1} = x_n + \delta v_{n+1}.$$
 (14)

The resulting numerical method has a larger stability region, as can be seen in Figures [4b](#page-5-1) to [4d,](#page-5-1) in which we show stability boundaries of AB methods after HB is applied to them with varying βs. (We use HB 0.4 to denote β = 0.4). However, Theorem [1](#page-19-0) in Appendix [F](#page-18-0) shows that as soon as β deviates from 1, the theoretical order of convergence drops to 1, leading to a significant decrease in image quality, as illustrated in Figure [5.](#page-5-2) In the next subsection, we propose an alternative approach that increases the stability region while maintaining high order of convergence.

<span id="page-5-1"></span>![](_page_5_Figure_8.jpeg)

Figure 4: Boundaries of stability regions of 1st- to 4th-order AB methods with HB applied to them with different values of the damping coefficient β.

<span id="page-5-2"></span>![](_page_5_Figure_10.jpeg)

Figure 5: Comparison between the two techniques we propose: (a) HB and (b) GHVB, applied to PLMS4 [\[20\]](#page-10-10) with 15 sampling steps. Both are effective at reducing artifacts, but HB's accuracy drops faster than GHVB's as β moves away from 1. Positions of the lanterns in Row (a) deviate more from the ground truth (1000 steps DDIM) than those in Row (b). Moreover, the image at β = 0.2 in Row (a) becomes blurry as HB yields a numerical method with a lower order of convergence than what GHVB does. Prompt: "A beautiful illustration of people releasing lanterns near a river".

#### <span id="page-6-4"></span>4.2 Generalizing Polyak's Heavy Ball to Higher Orders

In this section, we generalize Euler method with HB momentum to achieve high-order convergence in a similar way to how the Adams–Bashforth methods generalize the Euler method. We define the backward difference operator  $\Delta$  as  $\Delta x_n = x_n - x_{n-1}$ . According to [34], we can express the AB formula as:

<span id="page-6-5"></span><span id="page-6-0"></span>
$$\Delta x_{n+1} = \delta \left( 1 + \frac{1}{2} \Delta + \frac{5}{12} \Delta^2 + \frac{3}{8} \Delta^3 + \frac{251}{720} \Delta^4 + \frac{95}{288} \Delta^5 + \dots \right) f(x_n). \tag{15}$$

The order convergence is determined by the number of terms on the RHS. For example, the  $2^{nd}$ -order AB method can be written as  $\Delta x_{n+1} = \delta \left(1 + \frac{1}{2}\Delta\right) f(x_n)$ . The update rule for  $v_n$  in Equation 13 can be rewritten as  $(\beta + (1 - \beta)\Delta)v_{n+1} = \beta f(x_n)$ . Multiplying both sides of Equation 15 by  $(\beta + (1 - \beta)\Delta)$ , we have:

$$(\beta + (1 - \beta)\Delta)\Delta x_{n+1} = \delta \left(\beta + \frac{2 - \beta}{2}\Delta + \frac{6 - \beta}{12}\Delta^2 + \frac{10 - \beta}{24}\Delta^3 + \dots\right) f(x_n).$$
 (16)

Next, we can choose the order of convergence by fixing the number of terms on the RHS. To get, say, a 2<sup>nd</sup>-order method, we may choose:

<span id="page-6-2"></span>
$$(\beta + (1 - \beta)\Delta)\Delta x_{n+1} = \delta \left(\beta + \frac{2 - \beta}{2}\Delta\right) f(x_n) = \delta \left(1 + \frac{2 - \beta}{2\beta}\Delta\right) \beta f(x_n)$$
(17)  
$$= \delta \left(1 + \frac{2 - \beta}{2\beta}\Delta\right) (\beta + (1 - \beta)\Delta)v_{n+1}.$$
(18)

Eliminating  $(\beta - (1 - \beta)\Delta)$  from both sides, we obtain the 2<sup>nd</sup>-order generalized HB method:

<span id="page-6-1"></span>
$$v_{n+1} = (1-\beta)v_n + \beta f(x_n), \qquad x_{n+1} = x_n + \delta \left(\frac{2+\beta}{2\beta}v_{n+1} + \frac{2-\beta}{2\beta}v_n\right).$$
 (19)

Algorithm 2 details a complete implementation. When  $\beta=1$ , the formulation in Equation 19 is equivalent to the AB2 formulation in Equation 9. As  $\beta$  approaches 0, Equation 17 converges to the 1<sup>st</sup>-order Euler method 7. Thus, this generalization also serves as an interpolating technique between two adjacent-order AB methods, except for the 1<sup>st</sup>-order GHVB, which is equivalent to the Euler method with HB momentum in Equation 13.

We call this new method the Generalized Heavy Ball (GHVB) and associate with it a *momentum number*, whose ceiling indicates the method's order. For example, GHVB 1.8 refers to the  $2^{\rm nd}$ -order GHVB with  $\beta=0.8$ . The main difference between HB and GHVB is that HB calculates the moving average after summing high-order coefficients, whereas GHVB calculates it before the summation.

We analyze the stability region of GHVB using the same approach as before and visualize the region's locus curve in Figure 6. The theoretical order of accuracy of this method is given by Theorem 2 in Appendix F. We discuss alternative momentum methods, such as Nesterov's momentum, which can offer comparable performance but are less simple in Appendix E.

<span id="page-6-3"></span>![](_page_6_Figure_12.jpeg)

Figure 6: Boundary of stability regions for 1st-to 4th- order Generalized Heavy Ball Methods (GHVB).

### <span id="page-7-0"></span>**Experiments**

We present a series of experiments to evaluate the effectiveness of our techniques. In Section 5.1, we assess the reduction of divergence artifacts through qualitative results and quantitative measurements of the latent magnitudes in a text-to-image diffusion model. Besides reducing artifacts, another important goal is to ensure that the overall sampling quality improves and does not degenerate (e.g., becoming color blobs). We test this with experiments on both pixel-based and latent-based diffusion models trained on ImageNet@256[35] (Section 5.2 and 5.3), which show that our techniques indeed significantly improve image quality, as measured by the standard Fréchet Inception Distance (FID) score. Lastly, in Section 5.4, we present an ablation study of GHVB methods with varying degrees of orders. A similar study on HB methods can be found in Appendix M.

#### <span id="page-7-1"></span>5.1 Artifacts Mitigation

In this experiment, we apply our HB and GHVB techniques to the most popular 2<sup>nd</sup> and 4<sup>th</sup>-order solvers, DPM-Solver++ [19] and PLMS4 [20], using 15 sampling steps and various guidance scales on three different text-to-image diffusion models. The qualitative results in Figure 1 show our techniques significantly reduce the divergence artifacts and produce realistic results (columns a, c). More qualitative results are in Figure 15 in Appendix G.

Quantitatively measuring divergence artifacts can be challenging, as metrics like MSE or LPIPS may only capture the discrepancy between the approximated and the true solutions, which does not necessarily indicate the presence of divergence artifacts. In this study, we use the magnitudes of latent variables as introduced in Section 3.1 as a proxy metric to measure artifacts. In particular, we define a magnitude score  $v = \sum_{i,j} f(z'_{ij})$  that sums over the latent variables in a maxpooled latent grid, where f(x) = x if  $x \ge \tau$  and 0 otherwise. We generate 160 samples from the same set of text prompts and seeds for each method from a fine-tuned Stable Diffusion Figure 7: Average magnitude scores model called Anything V4 [23].

The results using  $\tau=3$  (magnitude considered high when above 3 std.) are shown in Figures 7 and 8. We observe that the magnitude score increases as the number of sampling steps decreases and higher-order methods result in higher magnitude scores. Figure 7 shows that adding HB momentum to PLMS4 [20] or DPM-Solver++[27] can reduce their magnitude scores, while Figure 8 shows that GHVB can also reduce the magnitude scores by reducing the momentum number. Next, we show that our results with less artifacts also have good image quality.

![](_page_7_Figure_6.jpeg)

<span id="page-7-3"></span>

![](_page_7_Figure_8.jpeg)

<span id="page-7-4"></span>Figure 8: Average magnitude scores

#### <span id="page-7-2"></span>5.2 Experiments on Pixel-based Diffusion Models

We evaluate our techniques using classifier-guided diffusion sampling with ADM [36], an unconditioned pixel-based diffusion model, with their classifier model. Additionally, we compare our methods with two other diffusion sampling methods, namely DPM-Solver++ [27] and LTSP [26], which have demonstrated strong performance in classifier-guided sampling.

For DPM-Solver++, we use a 2<sup>nd</sup>-order multi-step method and compare the results with and without HB momentum. For LTSP, a split numerical method, we use PLMS4 [20] to solve the first subproblem (see [26]) and compare different methods for solving the second subproblem, including regular Euler method and Euler method with HB momentum (equivalent to GHVB 0.8).

Our techniques effectively improve FID scores for both DPM-Solver++ and LTSP, as shown in Figure 9. Notably, applying our HB momentum to LTSP consistently produces the lowest FID scores. This experiment highlights the benefits of using HB momentum, which provides a better choice than Euler method. Table 1 presents additional results, and Figure 16

<span id="page-8-2"></span>![](_page_8_Figure_2.jpeg)

method. Table 1 presents additional results, and Figure 16 Figure 9: FID scores on ADM. (†ours) provides examples of the generated images.

#### <span id="page-8-0"></span>5.3 Experiment on Latent-based Diffusion Models

We evaluate our techniques using classifier-free guidance diffusion sampling with DiT-XL [36], a pre-trained latent-space diffusion model. In this particular setting, 4th-order solvers, such as PLMS4 [20], demonstrate superior performance compared to other methods (refer to Appendix I), making it our selected method for comparison.

In Figure 10, a significant gap in FID scores can be observed between 4th-order PLMS4 and 1st-order DDIM, but this is mostly due to the difference in convergence order rather than divergence artifacts. Our GHVB 3.8 and 3.9 techniques successfully mitigate numerical divergence and lead to improved FID scores compared to PLMS4, particularly when the number of steps is below 10. Additionally, HB 0.9 also improves FID scores. However, using HB 0.8 with PLMS4 can worsen FID scores compared to using PLMS4 alone, since the method has 1st-order convergence, which is the same as DDIM. For high sampling steps, both HB and GHVB achieve comparable performance to PLMS4 without significant degradation of quality.

<span id="page-8-3"></span>![](_page_8_Figure_7.jpeg)

We provide additional results in Table 2, and example images Figure 10: FID score on DiT. (†ours) in Figure 17.

#### <span id="page-8-1"></span>5.4 Ablation Study of GHVB

In this section, we conduct an ablation study of the GHVB method. As explained in Section 4.2, the damping coefficient  $\beta$  of GHVB interpolates between two existing AB methods, DDIM and PLMS2. Our goal here is to analyze the convergence error of GHVB methods. The comparison is done on Stable Diffusion 1.5 with the target results obtained from a 1,000-step PLMS4 method. We measure the mean L2 distance between the sampled results and the target results in the latent space. The results in Figure 11 suggest that the convergence error of GHVB 1.1 to GHVB 1.9 interpolates between the convergence errors of DDIM and PLMS2 accordingly.

Furthermore, we empirically verify that GHVB does achieve high order of convergence as predicted by Theorem 2. We compute the numerical order of convergence using the formula  $q \approx \frac{\log(e_{\text{new}}/e_{\text{old}})}{\log(k_{\text{new}}/k_{\text{old}})}$ , where e is the error between the sampled and the target latent codes, and k is the number of sampling steps. As shown in Figure 12, the numerical orders of GHVB 0.5 and GHVB 1.5 approach 0.5 and 1.5, respectively, as the number of steps increases. However, for GHVB 2.5 and GHVB 3.5, the estimated error e may be too small when tested with large numbers of steps, and other sources of error may hinder their convergence. Nonetheless, these GHVB methods can achieve high orders of convergence. A detailed analysis of this and other methods is in Appendix L.

![](_page_9_Figure_0.jpeg)

![](_page_9_Figure_1.jpeg)

<span id="page-9-8"></span>Figure 11: L2 distance in latent space between different sampling methods and the 1,000-step PLMS4 method.

<span id="page-9-9"></span>Figure 12: the numerical order of convergence for GHVB.

#### <span id="page-9-7"></span>6 Discussion

The findings of our study highlight an issue when employing high-order methods for sampling diffusion models with a low number of steps. This can result in solution divergence and the emergence of artifacts. To tackle these challenges, we propose two techniques inspired by Polyak's HB momentum, which effectively reduce artifacts while maintaining efficient sampling.

Our work is closely related to several other approaches aimed at improving the sampling speed of diffusion models. One approach involves training separate models that can be sampled faster, which include model distillation [37, 15], Schrödinger bridge [38], consistency models [17], and GENIE [39]. Another approach focuses on creating better samplers, such as high-order numerical methods, that can be applied to existing diffusion models. While some samplers were designed for the SDE formulation of diffusion models [40–42], most of them deal with the ODE formulation. These include linear multistep methods [20, 27, 18], predictor-corrector methods [43, 44, 25], and splitting methods [26]. Our paper specifically proposes new numerical methods for the ODE formulation, but these methods can also be extended to other sampling approaches involving multiple steps, including the SDE formulation. These techniques are not mutually exclusive.

#### References

- <span id="page-9-0"></span>[1] Ho, J., A. Jain, P. Abbeel. (2020), Denoising diffusion probabilistic models. In *Proceedings of the 34th International Conference on Neural Information Processing Systems*, pages 6840–6851.
- <span id="page-9-1"></span>[2] Song, J., C. Meng, S. Ermon. (2020), Denoising diffusion implicit models. In *International Conference on Learning Representations*.
- <span id="page-9-2"></span>[3] Goodfellow, I., J. Pouget-Abadie, M. Mirza, et al. (2014), Generative adversarial nets. *Advances in neural information processing systems*, 27.
- <span id="page-9-3"></span>[4] Rombach, R., A. Blattmann, D. Lorenz, et al. (2022), High-resolution image synthesis with latent diffusion models. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 10684–10695.
- <span id="page-9-4"></span>[5] Dhariwal, P., A. Nichol. (2021), Diffusion models beat GANs on image synthesis. *Advances in Neural Information Processing Systems*, 34:8780–8794.
- <span id="page-9-5"></span>[6] Nichol, A. Q., P. Dhariwal, A. Ramesh, et al. (2022), GLIDE: Towards photorealistic image generation and editing with text-guided diffusion models. In *International Conference on Machine Learning*, pages 16784–16804. PMLR.
- <span id="page-9-6"></span>[7] Su, X., J. Song, C. Meng, et al. (2022), Dual diffusion implicit bridges for image-to-image translation. In *The Eleventh International Conference on Learning Representations*.

- <span id="page-10-0"></span>[8] Sasaki, H., C. G. Willcocks, T. P. Breckon. (2021), Unit-DDPM: Unpaired image translation with denoising diffusion probabilistic models. *arXiv preprint arXiv:2104.05358*.
- <span id="page-10-1"></span>[9] Wang, J., Z. Lyu, D. Lin, et al. (2022), Guided diffusion model for adversarial purification. *arXiv preprint arXiv:2205.14969*.
- <span id="page-10-2"></span>[10] Wu, Q., H. Ye, Y. Gu. (2022), Guided diffusion model for adversarial purification from random noise. *arXiv preprint arXiv:2206.10875*.
- <span id="page-10-3"></span>[11] Choi, J., S. Kim, Y. Jeong, et al. (2021), ILVR: Conditioning method for denoising diffusion probabilistic models. In *2021 IEEE/CVF International Conference on Computer Vision (ICCV)*, pages 14347–14356. IEEE.
- <span id="page-10-4"></span>[12] Ghosal, D., N. Majumder, A. Mehrish, et al. (2023), Text-to-audio generation using instructiontuned LLM and latent diffusion model. *arXiv preprint arXiv:2304.13731*.
- <span id="page-10-5"></span>[13] Nichol, A. Q., P. Dhariwal. (2021), Improved denoising diffusion probabilistic models. In *International Conference on Machine Learning*, pages 8162–8171. PMLR.
- <span id="page-10-6"></span>[14] Watson, D., J. Ho, M. Norouzi, et al. (2021), Learning to efficiently sample from diffusion probabilistic models. *arXiv preprint arXiv:2106.03802*.
- <span id="page-10-7"></span>[15] Salimans, T., J. Ho. (2022), Progressive distillation for fast sampling of diffusion models. In *International Conference on Learning Representations*.
- [16] Watson, D., W. Chan, J. Ho, et al. (2022), Learning fast samplers for diffusion models by differentiating through sample quality. In *International Conference on Learning Representations*.
- <span id="page-10-8"></span>[17] Song, Y., P. Dhariwal, M. Chen, et al. (2023), Consistency models. *International Conference on Learning Representations*.
- <span id="page-10-14"></span>[18] Zhang, Q., Y. Chen. (2022), Fast sampling of diffusion models with exponential integrator. In *NeurIPS 2022 Workshop on Score-Based Methods*.
- <span id="page-10-9"></span>[19] Lu, C., Y. Zhou, F. Bao, et al. (2022), DPM-Solver: A fast ODE solver for diffusion probabilistic model sampling in around 10 steps. In *Advances in Neural Information Processing Systems*.
- <span id="page-10-10"></span>[20] Liu, L., Y. Ren, Z. Lin, et al. (2022), Pseudo numerical methods for diffusion models on manifolds. In *International Conference on Learning Representations*.
- <span id="page-10-15"></span>[21] Polyak, B. T. (1987), Introduction to optimization. optimization software. *Inc., Publications Division, New York*, 1:32.
- <span id="page-10-11"></span>[22] Realistic vision v2.0. [https://huggingface.co/SG161222/Realistic\\_Vision\\_V2.0](https://huggingface.co/SG161222/Realistic_Vision_V2.0), 2023.
- <span id="page-10-12"></span>[23] Anything diffusion v4.0. <https://huggingface.co/andite/anything-v4.0>, 2023.
- <span id="page-10-13"></span>[24] Deliberate diffuson. <https://huggingface.co/XpucT/Deliberate>, 2023.
- <span id="page-10-16"></span>[25] Zhao, W., L. Bai, Y. Rao, et al. (2023), UniPC: A unified predictor-corrector framework for fast sampling of diffusion models. *arXiv preprint arXiv:2302.04867*.
- <span id="page-10-17"></span>[26] Wizadwongsa, S., S. Suwajanakorn. (2023), Accelerating guided diffusion sampling with splitting numerical methods. *International Conference on Learning Representations*.
- <span id="page-10-18"></span>[27] Lu, C., Y. Zhou, F. Bao, et al. (2022), DPM-Solver++: Fast solver for guided sampling of diffusion probabilistic models. *arXiv preprint arXiv:2211.01095*.
- <span id="page-10-19"></span>[28] Radford, A., J. W. Kim, C. Hallacy, et al. (2021), Learning transferable visual models from natural language supervision. In *International Conference on Machine Learning*, pages 8748– 8763. PMLR.
- <span id="page-10-20"></span>[29] Letts, A., C. Scalf, A. Spirin, et al. Disco diffusion. [https://github.com/alembics/](https://github.com/alembics/disco-diffusion) [disco-diffusion](https://github.com/alembics/disco-diffusion), 2021.

- <span id="page-11-0"></span>[30] Ho, J., T. Salimans. (2021), Classifier-free diffusion guidance. In *NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications*.
- <span id="page-11-1"></span>[31] Lambert, J. D., et al. (1991), *Numerical methods for ordinary differential systems*, vol. 146. Wiley New York.
- <span id="page-11-2"></span>[32] Lin, T.-Y., M. Maire, S. Belongie, et al. (2014), Microsoft COCO: Common objects in context. In *Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13*, pages 740–755. Springer.
- <span id="page-11-3"></span>[33] Higham, D. J., L. N. Trefethen. (1993), Stiffness of ODEs. *BIT Numerical Mathematics*, 33:285–303.
- <span id="page-11-4"></span>[34] Berry, M. M., L. M. Healy. (2004), Implementation of Gauss-Jackson integration for orbit propagation. *The Journal of the Astronautical Sciences*.
- <span id="page-11-5"></span>[35] Russakovsky, O., J. Deng, H. Su, et al. (2015), ImageNet large scale visual recognition challenge. *International journal of computer vision*, 115(3):211–252.
- <span id="page-11-6"></span>[36] Peebles, W., S. Xie. (2022), Scalable diffusion models with transformers. *arXiv preprint arXiv:2212.09748*.
- <span id="page-11-7"></span>[37] Luhman, E., T. Luhman. (2021), Knowledge distillation in iterative generative models for improved sampling speed. *arXiv preprint arXiv:2101.02388*.
- <span id="page-11-8"></span>[38] De Bortoli, V., J. Thornton, J. Heng, et al. (2021), Diffusion Schrödinger bridge with applications to score-based generative modeling. *Advances in Neural Information Processing Systems*, 34:17695–17709.
- <span id="page-11-9"></span>[39] Dockhorn, T., A. Vahdat, K. Kreis. (2022), GENIE: Higher-order denoising diffusion solvers. *arXiv preprint arXiv:2210.05475*.
- <span id="page-11-10"></span>[40] Tachibana, H., M. Go, M. Inahara, et al. (2021), Itô-Taylor sampling scheme for denoising diffusion probabilistic models using ideal derivatives. *arXiv e-prints*, pages arXiv–2112.
- [41] Dockhorn, T., A. Vahdat, K. Kreis. (2022), Score-based generative modeling with criticallydamped langevin diffusion. *International Conference on Learning Representations*.
- <span id="page-11-11"></span>[42] Song, Y., J. Sohl-Dickstein, D. P. Kingma, et al. (2020), Score-based generative modeling through stochastic differential equations. In *International Conference on Learning Representations*.
- <span id="page-11-12"></span>[43] Karras, T., M. Aittala, T. Aila, et al. (2022), Elucidating the design space of diffusion-based generative models. In *NeurIPS 2022 Workshop on Deep Generative Models and Downstream Applications*.
- <span id="page-11-13"></span>[44] Zhang, Q., M. Tao, Y. Chen. (2023), gDDIM: Generalized denoising diffusion implicit models. *International Conference on Learning Representations*.
- <span id="page-11-14"></span>[45] Lucas, J., S. Sun, R. Zemel, et al. (2018), Aggregated momentum: Stability through passive damping. *arXiv preprint arXiv:1804.00325*.
- <span id="page-11-15"></span>[46] Nesterov, Y. (1983), A method for unconstrained convex minimization problem with the rate of convergence o(1/kˆ2). In *Doklady an ussr*, vol. 269, pages 543–547.
- <span id="page-11-16"></span>[47] Kynkäänniemi, T., T. Karras, S. Laine, et al. (2019), Improved precision and recall metric for assessing generative models. *Advances in Neural Information Processing Systems*, 32.
- <span id="page-11-17"></span>[48] Zhang, R., P. Isola, A. A. Efros, et al. (2018), The unreasonable effectiveness of deep features as a perceptual metric. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 586–595.

## Part I

# **Appendices**

| Appendix contents | Ap | pendix | contents |
|-------------------|----|--------|----------|
|-------------------|----|--------|----------|

| A Stability Region of Adam-Bashforth Method A.1 The Boundary Locus Technique  B Derivation of Test Equation C Toy ODE Problem D Implementation Details of PLMS with HB and GHVB Methods E Variance Momentum Methods F Order of Convergence G Qualitative Comparisons H Experimental Details and Results of ADM I Experimental Details and Results of DiT J Extended Comparison on Text-to-Image Comparison K Factors Contributing to Artifact Occurrence in Fine-tuned Diffusion Models L Elaboration on the Order of Convergence Approximation M Ablation Study on HB Momentum N Ablation Study on Nesterov Momentum O Statistical Reports P Ablation on Magnitude Score P.1 Results with Alternative Parameter Settings P.2 Results on Alternative Models  Q Frequently Asked Questions |      |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------|
| B Derivation of Test Equation C Toy ODE Problem D Implementation Details of PLMS with HB and GHVB Methods E Variance Momentum Methods F Order of Convergence G Qualitative Comparisons H Experimental Details and Results of ADM I Experimental Details and Results of DiT J Extended Comparison on Text-to-Image Comparison K Factors Contributing to Artifact Occurrence in Fine-tuned Diffusion Models L Elaboration on the Order of Convergence Approximation M Ablation Study on HB Momentum N Ablation Study on Nesterov Momentum O Statistical Reports P Ablation on Magnitude Score P.1 Results with Alternative Parameter Settings P.2 Results on Alternative Models                                                                                                             | 13   |
| C Toy ODE Problem  D Implementation Details of PLMS with HB and GHVB Methods  E Variance Momentum Methods  F Order of Convergence  G Qualitative Comparisons  H Experimental Details and Results of ADM  I Experimental Details and Results of DiT  J Extended Comparison on Text-to-Image Comparison  K Factors Contributing to Artifact Occurrence in Fine-tuned Diffusion Models  L Elaboration on the Order of Convergence Approximation  M Ablation Study on HB Momentum  N Ablation Study on Nesterov Momentum  O Statistical Reports  P Ablation on Magnitude Score  P.1 Results with Alternative Parameter Settings  P.2 Results on Alternative Models                                                                                                                            | . 14 |
| D Implementation Details of PLMS with HB and GHVB Methods  E Variance Momentum Methods  F Order of Convergence  G Qualitative Comparisons  H Experimental Details and Results of ADM  I Experimental Details and Results of DiT  J Extended Comparison on Text-to-Image Comparison  K Factors Contributing to Artifact Occurrence in Fine-tuned Diffusion Models  L Elaboration on the Order of Convergence Approximation  M Ablation Study on HB Momentum  N Ablation Study on Nesterov Momentum  O Statistical Reports  P Ablation on Magnitude Score  P.1 Results with Alternative Parameter Settings  P.2 Results on Alternative Models                                                                                                                                               | 15   |
| E Variance Momentum Methods  F Order of Convergence  G Qualitative Comparisons  H Experimental Details and Results of ADM  I Experimental Details and Results of DiT  J Extended Comparison on Text-to-Image Comparison  K Factors Contributing to Artifact Occurrence in Fine-tuned Diffusion Models  L Elaboration on the Order of Convergence Approximation  M Ablation Study on HB Momentum  N Ablation Study on Nesterov Momentum  O Statistical Reports  P Ablation on Magnitude Score  P.1 Results with Alternative Parameter Settings  P.2 Results on Alternative Models                                                                                                                                                                                                          | 15   |
| F Order of Convergence G Qualitative Comparisons H Experimental Details and Results of ADM Experimental Details and Results of DiT J Extended Comparison on Text-to-Image Comparison K Factors Contributing to Artifact Occurrence in Fine-tuned Diffusion Models L Elaboration on the Order of Convergence Approximation M Ablation Study on HB Momentum N Ablation Study on Nesterov Momentum O Statistical Reports P Ablation on Magnitude Score P.1 Results with Alternative Parameter Settings P.2 Results on Alternative Models                                                                                                                                                                                                                                                     | 16   |
| G Qualitative Comparisons  H Experimental Details and Results of ADM  I Experimental Details and Results of DiT  J Extended Comparison on Text-to-Image Comparison  K Factors Contributing to Artifact Occurrence in Fine-tuned Diffusion Models  L Elaboration on the Order of Convergence Approximation  M Ablation Study on HB Momentum  N Ablation Study on Nesterov Momentum  O Statistical Reports  P Ablation on Magnitude Score  P.1 Results with Alternative Parameter Settings  P.2 Results on Alternative Models                                                                                                                                                                                                                                                               | 17   |
| H Experimental Details and Results of ADM  I Experimental Details and Results of DiT  J Extended Comparison on Text-to-Image Comparison  K Factors Contributing to Artifact Occurrence in Fine-tuned Diffusion Models  L Elaboration on the Order of Convergence Approximation  M Ablation Study on HB Momentum  N Ablation Study on Nesterov Momentum  O Statistical Reports  P Ablation on Magnitude Score  P.1 Results with Alternative Parameter Settings  P.2 Results on Alternative Models                                                                                                                                                                                                                                                                                          | 19   |
| I Experimental Details and Results of DiT  J Extended Comparison on Text-to-Image Comparison  K Factors Contributing to Artifact Occurrence in Fine-tuned Diffusion Models  L Elaboration on the Order of Convergence Approximation  M Ablation Study on HB Momentum  N Ablation Study on Nesterov Momentum  O Statistical Reports  P Ablation on Magnitude Score  P.1 Results with Alternative Parameter Settings  P.2 Results on Alternative Models                                                                                                                                                                                                                                                                                                                                     | 21   |
| J Extended Comparison on Text-to-Image Comparison  K Factors Contributing to Artifact Occurrence in Fine-tuned Diffusion Models  L Elaboration on the Order of Convergence Approximation  M Ablation Study on HB Momentum  N Ablation Study on Nesterov Momentum  O Statistical Reports  P Ablation on Magnitude Score  P.1 Results with Alternative Parameter Settings  P.2 Results on Alternative Models                                                                                                                                                                                                                                                                                                                                                                                | 21   |
| K Factors Contributing to Artifact Occurrence in Fine-tuned Diffusion Models  L Elaboration on the Order of Convergence Approximation  M Ablation Study on HB Momentum  N Ablation Study on Nesterov Momentum  O Statistical Reports  P Ablation on Magnitude Score  P.1 Results with Alternative Parameter Settings  P.2 Results on Alternative Models                                                                                                                                                                                                                                                                                                                                                                                                                                   | 24   |
| L Elaboration on the Order of Convergence Approximation  M Ablation Study on HB Momentum  N Ablation Study on Nesterov Momentum  O Statistical Reports  P Ablation on Magnitude Score P.1 Results with Alternative Parameter Settings P.2 Results on Alternative Models                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | 24   |
| M Ablation Study on HB Momentum  N Ablation Study on Nesterov Momentum  O Statistical Reports  P Ablation on Magnitude Score P.1 Results with Alternative Parameter Settings P.2 Results on Alternative Models                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | 26   |
| N Ablation Study on Nesterov Momentum  O Statistical Reports  P Ablation on Magnitude Score P.1 Results with Alternative Parameter Settings P.2 Results on Alternative Models                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | 27   |
| O Statistical Reports  P Ablation on Magnitude Score P.1 Results with Alternative Parameter Settings                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | 32   |
| P Ablation on Magnitude Score P.1 Results with Alternative Parameter Settings                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | 33   |
| P.1 Results with Alternative Parameter Settings                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | 34   |
| P.2 Results on Alternative Models                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | 34   |
| P.2 Results on Alternative Models                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | . 35 |
| Q Frequently Asked Questions                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |      |
|                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | 35   |

## <span id="page-12-0"></span>A Stability Region of Adam-Bashforth Method

To investigate the stability of the AB2 method, we apply AB2 to the test equation  $x'=\lambda x$ , which was also used with the Euler method (Section 2.3). We have  $x_{n+1}=x_n+\delta\left(\frac{3}{2}\lambda x_n-\frac{1}{2}\lambda x_{n-1}\right)$ . To

solve this linear recurrence relation, we substitute  $x_n = r^n$  into the formula, where r is a complex constant. Simplifying the resulting equation, we obtain the characteristic equation:

$$r^2 - \left(1 + \frac{3}{2}\delta\lambda\right)r + \frac{1}{2}\delta\lambda = 0,\tag{20}$$

which has the solutions

$$r_1 = \frac{1}{2} \left( 1 + \frac{3}{2} \delta \lambda + \sqrt{\left( 1 + \frac{3}{2} \delta \lambda \right)^2 - 2\delta \lambda} \right),\tag{21}$$

$$r_2 = \frac{1}{2} \left( 1 + \frac{3}{2} \delta \lambda - \sqrt{\left( 1 + \frac{3}{2} \delta \lambda \right)^2 - 2\delta \lambda} \right). \tag{22}$$

The general formulation of  $x_n$  can be expressed as

$$x_n = a_1 r_1^n + a_2 r_2^n, (23)$$

where  $a_1$  and  $a_2$  are constants. The numerical solution  $x_n$  tends to 0 as n tends to infinity when both  $|r_1| < 1$  and  $|r_2| < 1$ , which means the stability region of AB2 is determined by the complex region

$$S = \left\{ z \in \mathbb{C} : \left| \frac{1}{2} \left( 1 + \frac{3}{2} z \pm \sqrt{\left( 1 + \frac{3}{2} z \right)^2 - 2z} \right) \right| \le 1 \right\}. \tag{24}$$

Solving for the complex area from the roots of the characteristic equation can pose significant challenges in numerical analysis. One commonly employed graphical technique to visualize the stability region is the boundary locus technique [31].

#### <span id="page-13-0"></span>A.1 The Boundary Locus Technique

The boundary locus technique [31] begins by defining the shift operator E such that  $Ex_k = x_{k-1}$ . Note that  $E^2x_k = Ex_{k-1} = x_{k-2}$ . Generally, a numerical method can be represented in the following form:

<span id="page-13-1"></span>
$$A(E)x_n = \delta B(E)f(x_n), \tag{25}$$

where A and B are polynomials of E. For example, in the case of the AB2 method, we have A(E)=1-E and  $B(E)=\frac{3}{2}E-\frac{1}{2}E^2$ .

To determine the stability region of a numerical method, we apply the boundary locus technique to the general form given by Equation 25. The characteristic equation of the method can be obtained by substituting  $f(x_n) = \lambda x_n$  (i.e., the test equation) and  $x_n = r^n$ , which yields

$$A(r^{-1}) = \delta \lambda B(r^{-1}),\tag{26}$$

where r is the root of the method's characteristic equation. The stability region of the method is the area in the complex plane where the characteristic root r have modulus less than 1. The boundary of the stability region can be determined by substituting r with a modulus of 1 (which means that  $r=e^{i\theta}$  for some real value  $\theta$ ) into the characteristic equation and solving for  $z=\delta\lambda$ . This yields the locus of points in the complex plane where the characteristic roots of the method are on the boundary of the stability region. Specifically, we can obtain the curve  $z=s(\theta)=A(e^{-i\theta})/B(e^{-i\theta})$ , where  $\theta\in [-\pi,\pi]$ , that represents the boundary of the stability region in the complex plane. By comparing the stability regions of different numerical methods, we can determine which method is more stable and accurate for a given problem. The boundary locus technique provides a powerful tool for analyzing the stability of numerical methods and can help guide the selection of appropriate methods for solving ODE problems.

**Example A.1.** (Euler Method) The Euler method, a numerical technique for approximating solutions of ODE, can be expressed as:

$$(1 - E)x_n = \delta E f(x_n) \tag{27}$$

The associated polynomials for this method are:

$$A(z) = 1 - z, \quad B(z) = z$$
 (28)

The stability region of the Euler method corresponds to the locus curve in which the solution remains bounded. This region can be determined by evaluating the complex function:

$$s(\theta) = \frac{A(e^{-i\theta})}{B(e^{-i\theta})} = \frac{1 - e^{-i\theta}}{e^{-i\theta}} = e^{i\theta} - 1, \quad \theta \in [-\pi, \pi].$$
 (29)

The locus curve forms a perfect circle with a radius of 1 and a center at -1.

**Example A.2.** (AB Methods) The 2<sup>nd</sup>-order Adams-Bashforth (AB2) method is given by:

$$(1 - E)x_n = \delta\left(\frac{3}{2}E - \frac{1}{2}E^2\right)f(x_n).$$
 (30)

The locus curve representing the stability region of this method is given by:

$$s(\theta) = \frac{1 - e^{-i\theta}}{\frac{3}{2}e^{-i\theta} - \frac{1}{2}e^{-2i\theta}} = \frac{2(1 - e^{-i\theta})}{3e^{-i\theta} - e^{-2i\theta}}, \quad \theta \in [-\pi, \pi].$$
 (31)

Similarly, the stability regions for the AB3 and AB4 methods can be obtained by evaluating the complex functions:

$$s(\theta) = \frac{12(1 - e^{-i\theta})}{23e^{-i\theta} - 16e^{-2i\theta} + 5e^{-3i\theta}}, \quad \theta \in [-\pi, \pi].$$
 (32)

$$s(\theta) = \frac{24(1 - e^{-i\theta})}{55e^{-i\theta} - 59e^{-2i\theta} + 37e^{-3i\theta} - 9e^{-4i\theta}}, \quad \theta \in [-\pi, \pi].$$
 (33)

The locus curves for the boundary of stability regions of first four AB methods are visualized in Figure 2

#### <span id="page-14-0"></span>**B** Derivation of Test Equation

This section presents the derivation of the test equation  $u' = \lambda u$ , which serves as a fundamental tool for analyzing the stability of numerical methods in diffusion sampling, as discussed in the Section 3.2.

Starting from the differential equation for  $\bar{x}$ , we have

$$\frac{d\bar{x}}{d\sigma} = \nabla \bar{\epsilon}(x^*)(\bar{x} - x^*). \tag{34}$$

We then define  $u=v^T(\bar x-x^*)$ , where v is a normalized eigenvector of  $\nabla \bar \epsilon(x^*)^T$  corresponding to the eigenvalue  $\lambda$ . Taking the derivative of u with respect to  $\sigma$  and using the chain rule, we have:

$$\frac{du}{d\sigma} = v^T \frac{d}{d\sigma} (\bar{x} - x^*) = v^T [\nabla \bar{\epsilon}(x^*)] (\bar{x} - x^*)$$
(35)

$$= [\nabla \bar{\epsilon}(x^*)^T v]^T (\bar{x} - x^*) = (\lambda v)^T (\bar{x} - x^*)$$
(36)

$$= \lambda u \tag{37}$$

Thus, we obtain the test equation  $u' = \lambda u$ .

#### <span id="page-14-1"></span>C Toy ODE Problem

This section aims to demonstrate how the solutions yielded by numerical methods can diverge when the stability regions of the methods are too small. Additionally, we illustrate how our momentum-based techniques can enlarge the stability region. The demonstration is conducted on a 2D toy ODE problem given by:

$$\frac{dx}{dt} = \begin{bmatrix} 0 & 1 \\ -9 & -10 \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}, \qquad x(0) = \begin{bmatrix} -1 \\ 0 \end{bmatrix}. \tag{38}$$

The eigenvalues of the  $2 \times 2$  matrix are -9 and -1, and the exact solution of Equation 38 is given by:

<span id="page-15-1"></span>
$$x(t) = \frac{1}{8} \begin{bmatrix} 1 \\ -9 \end{bmatrix} e^{-9t} + \frac{9}{8} \begin{bmatrix} -1 \\ 1 \end{bmatrix} e^{-t}.$$
 (39)

As t increases, x(t) converges to the origin.

Let us say we want to numerically compute x(3) by integrating the ODE for 26 steps with a numerical method. We divide the time interval [0,3] into 26 equal intervals, resulting in a step size of  $\delta=3/26$ . For this particular setting, it turns out that the the  $2^{\rm nd}$ -order Adams-Bashforth (AB2) method diverges, but the Euler method converges. To see this, observe that the stability region of the AB2 method only cover the interval [-1,0] of the real line, as depicted in Figures 13b and 13d. So, for the eigenvalue  $\lambda=-9$ , the product  $\delta\lambda=-27/26$  lies just outside the region. Consequently, the numerical solution yielded by AB2 diverges, as indicated by the blue line in Figures 13a and 13c. In contrast, the Euler method's stability region contains both values of  $\delta\lambda$ , and the numerical solution, represented by the green line in Figures 13a and 13c, appears to be more accurate.

The AB2 method can be made to more accurately compute x(3) by applying to it any of our proposed techniques: Heavy Ball momentum (HB) and GHVB. The stability regions of the modified AB2 methods are given by the red ( $\beta=0.8$ ) and yellow ( $\beta=0.9$ ) lines in Figures 13b and 13d. Observe that they contains the points associated with the  $\lambda\delta$  values. As a result, the numerical solutions of the modified methods converge to the origin, as demonstrated by the red and yellow lines in Figures 13a and 13c, respectively.

<span id="page-15-2"></span>![](_page_15_Figure_6.jpeg)

Figure 13: Comparison of solution trajectories and stability regions of various numerical methods when applied to the toy ODE problem. Here, we seek to compute x(3) in 26 steps with the Euler method, the AB2 method, and methods resulting from modifying AB2 with our momentum-based techniques. Subfigure (a) presents the numerical solutions obtained using our modified AB2 method with HB momentum, while subfigure (c) showcases those obtained using our GHVB. The stability regions of the methods are depicted in subfigures (b) and (d) respectively.

#### <span id="page-15-0"></span>D Implementation Details of PLMS with HB and GHVB Methods

In this section, we present the complete algorithms for the PLMS method with the HB momentum and the GHVB method in Algorithm 1 and Algorithm 2, respectively. Additionally, we include the locus curves that represent the boundaries of the stability regions for each method.

For the PLMS method with HB momentum, the locus curves representing the stability regions with parameter  $\beta$  are given by:

#### **PLMS1** with HB $\beta$ :

$$s(\theta) = \frac{(1 - e^{-i\theta})(1 - (1 - \beta)e^{-i\theta})}{\beta e^{-i\theta}}$$
(40)

**PLMS2** with HB  $\beta$ :

$$s(\theta) = \frac{2(1 - e^{-i\theta})(1 - (1 - \beta)e^{-i\theta})}{\beta(3e^{-i\theta} - e^{-2i\theta})}$$
(41)

**PLMS3** with HB  $\beta$ :

$$s(\theta) = \frac{12(1 - e^{-i\theta})(1 - (1 - \beta)e^{-i\theta})}{\beta(23e^{-i\theta} - 16e^{-2i\theta} + 5e^{-3i\theta})}$$
(42)

**PLMS4** with HB  $\beta$ :

$$s(\theta) = \frac{24(1 - e^{-i\theta})(1 - (1 - \beta)e^{-i\theta})}{\beta(55e^{-i\theta} - 59e^{-2i\theta} + 37e^{-3i\theta} - 9e^{-4i\theta})}$$
(43)

The locus curves representing the boundaries of the stability regions for different values of  $\beta$  are illustrated in Figure 4.

For GHVB method, the locus curve representing the stability region with parameter  $\beta$  is given by:

**1st-order GHVB** (equivalent to PLMS1 with HB):

$$s(\theta) = \frac{(1 - e^{-i\theta})(1 - (1 - \beta)e^{-i\theta})}{\beta e^{-i\theta}}$$
(44)

2nd-order GHVB:

$$s(\theta) = \frac{2(1 - e^{-i\theta})(1 - (1 - \beta)e^{-i\theta})}{((2 + \beta)e^{-i\theta} - (2 - \beta)e^{-2i\theta})}$$
(45)

3rd-order GHVB:

$$s(\theta) = \frac{12(1 - e^{-i\theta})(1 - (1 - \beta)e^{-i\theta})}{(18 + 5\beta)e^{-i\theta} - (24 - 8\beta)e^{-2i\theta} + (6 - \beta)e^{-3i\theta}}$$
(46)

4th-order GHVB:

$$s(\theta) = \frac{24(1 - e^{-i\theta})(1 - (1 - \beta)e^{-i\theta})}{(46 + 9\beta)e^{-i\theta} - (78 - 19\beta)e^{-2i\theta} + (42 - 5\beta)e^{-3i\theta} - (10 - \beta)e^{-4i\theta}}$$
(47)

These locus curves describe the boundaries of the stability regions and are shown in Figure 6.

#### Algorithm 1: PLMS with HB momentum

```
input: \bar{x}_n (previous result), \delta (step size), \{e_i\}_{i < n} (evaluation buffer), r (method order), v_n (previous velocity); e_n = \bar{\epsilon}_\sigma(\bar{x}_n) \; ; c = \min(r, n) \; ; if c == 1 then \hat{e} = e_n \; ; else if c == 2 then \hat{e} = (3e_n - e_{n-1})/2 \; ; else if c == 3 then \hat{e} = (23e_n - 16e_{n-1} + 5e_{n-2})/12 \; ; else \hat{e} = (55e_n - 59e_{n-1} + 37e_{n-3} - 9e_{n-4})/24 \; ; v_{n+1} = (1-\beta)v_n + \beta \hat{e}; Result: \bar{x}_n + \delta v_{n+1}
```

#### <span id="page-16-0"></span>**E** Variance Momentum Methods

In 2019, a variant of Polyak's HB momentum called aggregated momentum [45] was proposed. Its objective is to enhance stability while also offering convergence advantages. This modification

#### **Algorithm 2:** GHVB

```
input: \bar{x}_n (previous result), \delta (step size), \beta (damping parameter)
\{v_i\}_{i \le n} (evaluation buffer), r (method order), ;
   v_{n+1} = (1-\beta)v_n + \beta\bar{\epsilon}_{\sigma}(\bar{x}_n);
   c = \min(r, n);
   if c == 1 then
       \hat{e} = v_{n+1};
   else if c == 2 then
       \hat{e} = ((2+\beta)v_{n+1} - (2-\beta)v_n)/2\beta;
   else if c == 3 then
       \hat{e} = ((18 + 5\beta)v_{n+1} - (24 - 8\beta)v_n
           +(6-\beta)v_{n+1})/12\beta;
   else if c == 4 then
       \hat{e} = ((46 + 9\beta)v_{n+1} - (78 - 19\beta)v_n \\ + (42 - 5\beta)v_{n-1} - (10 - \beta)v_{n-2})/24\beta ; 
       \hat{e} = ((1650 + 251\beta)v_{n+1} - (3420 - 646\beta)v_n
          +(2880 - 264\beta)v_{n-1} - (1380 - 106\beta)v_{n-2}
          +(270-19\beta)v_{n-3})/720\beta;
   Result: \bar{x}_n + \delta \hat{e}
```

introduces multiple velocities, denoted by  $v_n^{(i)}$ , each associated with its specific damping coefficient  $\beta^{(i)}$ .

$$v_{n+1}^{(i)} = (1 - \beta^{(i)})v_n^{(i)} + \beta^{(i)}f(x_n), \qquad x_{n+1} = x_n + \delta \sum_{i=1}^K w^{(i)}v_{n+1}^{(i)}$$
(48)

Nesterov's momentum [46] is one version of the classic momentum that can also be applied to diffusion sampling processes to improve stability. It can be written as follows:

$$y_{n+1} = x_n + \delta \beta f(x_n), \qquad x_{n+1} = y_{n+1} + (1 - \beta)(y_{n+1} - y_n)$$
 (49)

In fact, Nesterov's momentum can be obtained from aggregated momentum by considering the following:

$$v_{n+1}^{(1)} = (1 - \beta)v_n^{(1)} + \beta f(x_n), v_{n+1}^{(2)} = f(x_n),$$
  

$$x_{n+1} = x_n + \delta((1 - \beta)v_{n+1}^{(1)} + \beta v_{n+1}^{(2)}). (50)$$

The stability regions of Nesterov's momentum when applied to the Euler method and high-order Adams-Bashforth methods are illustrated in Figures 14a through 14d. Observe that the stability regions of methods with Nesterov's momentum become larger in a similar manner to those with Polyak's HB momentum. However, the enlargement due to Nesterov's momentum is more pronounced in the vertical direction, while the Polyak's HB momentum's enlargement is more horizontal in nature. (See Figures 13b and 13d) The differences in the shapes of the stability regions suggest one type of momentum is more suitable to certain ODE problems than the other.

<span id="page-17-1"></span>![](_page_17_Figure_9.jpeg)

Figure 14: Comparison of stability regions for different methods with different levels of Nesterov's momentum.

While generalizing the aggregated momentum method to higher-order methods is possible, it is no longer as straightforward as it is with the HB method. As an example, we will consider the 2<sup>nd</sup>-order generalization of Nesterov's momentum method.

We begin by noting that  $(\beta + (1 - \beta)\Delta)v_{n+1} = \beta f(x_n)$ . Our goal is to find polynomials B and C such that

$$\Delta x_{n+1} = \delta(B(\Delta)v_{n+1} + C(\Delta)f(x_n)). \tag{51}$$

Multiplying both sides of the equation by  $(\beta + (1 - \beta)\Delta)$ , we get

$$(\beta + (1 - \beta)\Delta)\Delta x_{n+1} = \delta(\beta B(\Delta)f(x_n) + (\beta + (1 - \beta)\Delta)C(\Delta)f(x_n)). \tag{52}$$

Replace the left side with the first two terms from Equation 16, we obtain

$$\delta\left(\beta + \frac{2+\beta}{2}\Delta\right)f(x_n) = \delta(\beta B(\Delta) + (\beta + (1-\beta)\Delta)C(\Delta))f(x_n). \tag{53}$$

Let  $B(\Delta) = b_0 + b_1 \Delta$  and  $C(\Delta) = c_0$ . Then, by balancing the coefficients of  $\Delta$  on both sides, we have  $1 = b_0 + c_0$  and  $\frac{2+\beta}{2} = \beta b_1 + (1-\beta)c_0$ . We can now write the final formulation as follows:

$$x_{n+1} = x_n + \delta(b_0 v_{n+1} + b_1(v_{n+1} - v_n) + c_0 f(x_n)).$$
(54)

This suggests that there are countless different ways to expand the stability region of a numerical method, which offer many new research opportunities.

### <span id="page-18-0"></span>F Order of Convergence

When solving ODEs numerically, it is important to consider the accuracy of the method used. One way to measure accuracy is by considering the order method's of convergence of the. Suppose we have a numerical method of the form

$$A(E)x_n = \delta B(E)f(x_n), \tag{55}$$

where  $A(E)=a_0+a_1E+a_2E^2+...+a_sE^s$  and  $B(E)=b_0+b_1E+...+b_sE^s$ . The method is said to be of  $p^{th}$  order if and only if, for all sufficiently smooth functions x, we have that

$$\sum_{m=0}^{s} a_m x(\sigma - m\delta) - \delta \sum_{m=0}^{s} b_m x'(\sigma - m\delta) = \mathcal{O}(\delta^{p+1}), \tag{56}$$

where x' denotes the derivative of x.

To derive the order of convergence, we use Taylor expansion for both x and x', yielding

$$\begin{aligned} \text{L.H.S.} &= \sum_{m=0}^{s} a_m \sum_{k=0}^{\infty} \frac{(-m\delta)^k}{k!} x^{(k)}(\sigma) - \delta \sum_{m=0}^{s} b_m \sum_{k=0}^{\infty} \frac{(-m\delta)^k}{k!} x^{(k+1)}(\sigma) \\ &= \sum_{k=0}^{\infty} \left( \sum_{m=0}^{s} a_m \frac{(-m\delta)^k}{k!} \right) x^{(k)}(\sigma) - \delta \sum_{k=0}^{\infty} \left( \sum_{m=0}^{s} b_m \frac{(-m\delta)^k}{k!} \right) x^{(k+1)}(\sigma) \\ &= \sum_{k=0}^{\infty} \left( \sum_{m=0}^{s} a_m \frac{(-m\delta)^k}{k!} \right) x^{(k)}(\sigma) + \sum_{k=1}^{\infty} \left( \sum_{m=0}^{s} b_m \frac{m^{k-1}(-\delta)^k}{(k-1)!} \right) x^k(\sigma) \\ &= \sum_{m=0}^{s} a_m + \sum_{k=1}^{\infty} \left( \sum_{m=0}^{s} a_m \frac{m^k}{k!} + \sum_{m=0}^{s} b_m \frac{m^{k-1}}{(k-1)!} \right) x^k(\sigma) (-\delta)^k \end{aligned}$$

where  $x^{(k)}(\sigma)$  denotes the  $k^{th}$  derivative of x evaluated at  $\sigma$ .

Therefore, the method has order p if and only if the coefficients satisfy the conditions given by

<span id="page-18-1"></span>
$$\sum_{m=0}^{s} a_m = 0,$$

$$\sum_{m=0}^{s} a_m \frac{m^k}{k!} + \sum_{m=0}^{s} b_m \frac{m^{k-1}}{(k-1)!} = 0, \quad k = 0, 1, ..., p.$$
(57)

Now, we discuss the convergence order of any numerical method after HB momentum is applied to it. An example of such an algorithm is the modified PLMS method, presented in Algorithm 1.

<span id="page-19-0"></span>**Theorem 1** (Convergence order of numerical methods with HB momentum). Suppose that a  $p^{th}$ -order numerical method has the form  $x_{n+1} = x_n + \delta \sum_{m=0}^s b_m f(x_{n-m})$ , where  $p \geq 1$ . The modified method that uses HB momentum can be expressed as follows:

$$v_{n+1} = (1 - \beta)v_n + \beta \sum_{m=0}^{s} b_m f(x_{n-m}),$$
(58)

$$x_{n+1} = x_n + \delta v_{n+1}. (59)$$

It has first-order convergence.

*Proof.* From the condition given in 57, it follows that  $\sum_{m=0}^{s} b_m = 1$ . We can rewrite these equations as:

$$x_{n+1} - x_n - (1 - \beta)(x_n - x_{n-1}) = \delta\beta \sum_{m=0}^{s} b_m f(x_{n-m}).$$
 (60)

To estimate the order of the modified method, we evaluate Equation 60 and obtain:

<span id="page-19-2"></span>
$$\sum_{m=0}^{s} a_m = 1 - 1 - (1 - \beta)(1 - 1) = 0,$$
(61)

$$\sum_{m=0}^{s} a_m \frac{m^1}{1!} + \sum_{m=0}^{s} b_m \frac{m^0}{0!} = 0 - 1 - (1 - \beta)(1 - 2) + \beta \sum_{m=0}^{s} b_m$$
$$= -\beta + \beta = 0.$$
 (62)

Therefore, we have shown that the modification to the method has first-order convergence.  $\Box$ 

Next, we turn our attention to the GHVB method.

<span id="page-19-1"></span>**Theorem 2** (Convergence order of the GHVB method). The  $r^{th}$ -order GHVB (Algorithm 2) has order of convergence of r.

*Proof.* We will use the 2<sup>nd</sup>-order method as an example. Using Equation 19, we can write an equivalent equation as:

$$x_{n+1} - x_n - (1 - \beta)(x_n - x_{n-1}) = \delta\left(\frac{2 + \beta}{2}f(x_n) - \frac{2 - \beta}{2}f(x_{n-1})\right).$$
 (63)

To estimate the order of the modified method, we evaluate Equation 60 and obtain:

$$\sum_{m=0}^{s} a_m = 1 - 1 - (1 - \beta)(1 - 1) = 0,$$

$$\sum_{m=0}^{s} a_m \frac{m^1}{1!} + \sum_{m=0}^{s} b_m \frac{m^0}{0!} = 0 - 1 - (1 - \beta)(1 - 2) + \left(\frac{2 + \beta}{2} - \frac{2 - \beta}{2}\right) = 0,$$

$$\sum_{m=0}^{s} a_m \frac{m^2}{2!} + \sum_{m=0}^{s} b_m \frac{m^1}{1!} = \frac{1}{2}(0 - 1^2 - (1 - \beta)(1^2 - 2^2)) + \left(\frac{2 + \beta}{2}1^1 - \frac{2 - \beta}{2}2^1\right)$$

$$= \frac{2 - 3\beta}{2} - \frac{2 - 3\beta}{2} = 0.$$

Thus, the method has a convergence order of two. Methods of other orders can be dealt with in a similar fashion.  $\Box$ 

#### <span id="page-20-1"></span>**G** Oualitative Comparisons

Figure 1 compares our momentum-based methods, HB and GHVB, with two different diffusion solver methods, DPM-Solver++ [27] and PLMS4 [20], without momentum. The number of sampling steps is held constant while varying the guidance scale s to intentionally induce divergence artifacts. (Note that the guidance scales that yield such artifacts are different between diffusion models.) The figure demonstrates that, under the difficult settings of low step counts and high guidance scales where the baseline methods produce artifacts, our proposed techniques can successfully eliminate them

We present additional qualitative results to show the effect of the damping parameter  $\beta$  on the quality of images generated by methods modified with HB momentum. We use methods of varying orders, including DPLM-Solver++[27], UniPC[25], and PLMS4[20]. The diffusion models utilized in our analysis are Realistic Vision v2.0<sup>1</sup>, Anything Diffusion v4.0<sup>2</sup>, Counterfeit Diffusion V2.5<sup>3</sup>, Pastel-Mix<sup>4</sup>, Deliberate Diffusion<sup>5</sup>, and Dreamlink Diffusion V1.0<sup>6</sup>. The results are shown in Figure 15. Notice that stronger momentum (lower  $\beta$ ) leads to fewer and less severe artifacts.

#### <span id="page-20-0"></span>H Experimental Details and Results of ADM

In this section, we present additional details and results for the ADM experiment in Section 5.2. The primary objective of this experiment was to provide a quantitative evaluation of class-conditional diffusion sampling in the context of pixel-based images. The experiment was conducted using the pre-trained diffusion and classifier model at the following link: <sup>7</sup>. The implementation used in our experiment was obtained directly from the official DPM-Solver GitHub repository <sup>8</sup>.

To enhance the capabilities of DPM-Solver++, we incorporated HB momentum into DPM-Solver++ (just change a few lines of code) and implemented the splitting method LTSP both with and without HB momentum into the DPM-Solver code for comparative purposes. The experiment was done on four NVIDIA RTX A4000 GPUs and a 24-core AMD Threadripper 3960x CPU.

The results of the experiment, as measured by the full FID score, are presented in Table 1 (as well as in Figrue 9). Our technique is highlighted in grey within the table, while the best FID score for each number of steps is indicated in bold. Additionally, Figure 16 showcases sample images from this experiment. As this experiment uses pixel-based diffusion models, the observed divergence artifacts differ from those in latent-based diffusion models. Specifically, the artifacts may display excessive brightness or darkness caused by pixel values nearing the maximum or minimum thresholds.

<span id="page-20-2"></span>

| Number of Steps        |       |       |       |       |       |       |       |       |  |
|------------------------|-------|-------|-------|-------|-------|-------|-------|-------|--|
| Method                 | 10    | 12    | 14    | 16    | 18    | 20    | 25    | 30    |  |
| DPM-Solver++           | 66.77 | 46.77 | 34.56 | 26.97 | 21.87 | 19.48 | 16.31 | 15.63 |  |
| DPM-Solver++ w/ HB 0.8 | 47.10 | 33.65 | 25.61 | 21.42 | 18.94 | 17.76 | 15.98 | 15.53 |  |
| LTSP [PLMS4, PLMS1]    | 45.32 | 34.08 | 26.58 | 21.54 | 18.54 | 17.15 | 15.79 | 15.51 |  |
| LTSP [PLMS4, GHVB 0.8] | 37.43 | 29.74 | 23.60 | 20.23 | 18.46 | 17.14 | 16.07 | 15.79 |  |

Table 1: FID scores on classifier-guidance ADM models

<span id="page-20-3"></span>https://huggingface.co/SG161222/Realistic\_Vision\_V2.0

<span id="page-20-4"></span><sup>&</sup>lt;sup>2</sup>https://huggingface.co/andite/anything-v4.0

<span id="page-20-5"></span><sup>3</sup>https://huggingface.co/gsdf/Counterfeit-V2.5

<span id="page-20-6"></span><sup>4</sup>https://huggingface.co/andite/pastel-mix

<span id="page-20-7"></span><sup>&</sup>lt;sup>5</sup>https://huggingface.co/XpucT/Deliberate

<span id="page-20-8"></span><sup>6</sup>https://huggingface.co/dreamlike-art/dreamlike-diffusion-1.0

<span id="page-20-9"></span><sup>&</sup>lt;sup>7</sup>https://github.com/openai/guided-diffusion

<span id="page-20-10"></span><sup>8</sup>https://github.com/LuChengTHU/dpm-solver

<span id="page-21-0"></span>![](_page_21_Figure_0.jpeg)

(c) PLMS4 [\[20\]](#page-10-10), a 4th-order method, using 15 steps, Prompt: "cute humanoid red panda"

Figure 15: The impact of different damping coefficients β on HB momentum for 2nd-order (a), 3 rd-order (b), and 4th-order (c) numerical methods. Notably, it is observed that incorporating higher momentum values (lower β) helps mitigate the occurrence of divergence artifacts.

<span id="page-22-0"></span>![](_page_22_Figure_0.jpeg)

Figure 16: Samples from various sampling methods employing classifier guidance diffusion with a guidance scale of 10 and 20 sampling steps.

## <span id="page-23-0"></span>I Experimental Details and Results of DiT

This section presents supplementary details of the DiT experiment conducted in Figure [10](#page-8-3) of Section [5.3](#page-8-0) and further investigates the performance of the DiT model [\[36\]](#page-11-6). The code implementation and pre-trained DiT model were obtained directly from the official GitHub repository[9](#page-23-3) . The experiment was done on four NVIDIA GeForce RTX 2080 Ti GPUs and a 24-core AMD Threadripper 3960x CPU.

Our baselines include DDIM [\[2\]](#page-9-1), DPM-Solver++ [\[27\]](#page-10-18), LTSP4 [\[26\]](#page-10-17), and PLMS4 [\[20\]](#page-10-10). Based on the FID score, PLMS4 emerged as the most effective sampling method within the chosen context. As a result, only PLMS4 was included in Section [5.3](#page-8-0) of our main paper. Our variations of the PLMS4 with HB 0.8 and 0.9, as well as GHVB 3.8 and 3.9. The results of the experiment are presented in Table [2.](#page-23-1) Our technique is highlighted in grey in the table, and the best FID score for each number of steps is in bold. Notably, our method outperforms the others.

Moreover, we include the improved Precision and Recall metrics [\[47\]](#page-11-16) in Tables [3](#page-23-4) and [4,](#page-23-4) respectively, where higher values indicate superior performance. Additionally, Figure [17](#page-24-0) displays sample images generated from different sampling methods.

<span id="page-23-1"></span>

|                      | Number of Steps |             |                                              |      |    |                     |    |    |
|----------------------|-----------------|-------------|----------------------------------------------|------|----|---------------------|----|----|
| Method               | 6               | 7           | 8                                            | 9    | 10 | 15                  | 20 | 25 |
| DDIM                 |                 |             | 55.35 36.97 26.06 19.47 15.02 8.04 6.52 5.94 |      |    |                     |    |    |
| DPM-Solver++         |                 | 18.60 10.80 | 7.93                                         | 6.72 |    | 6.13 5.49 5.30 5.24 |    |    |
| LTSP4 [PLMS4, PLMS1] | 13.33           | 9.01        | 7.49                                         | 6.55 |    | 6.09 5.32 5.20 5.17 |    |    |
| PLMS4                | 13.10           | 8.94        | 7.31                                         | 6.51 |    | 6.03 5.32 5.21 5.17 |    |    |
| PLMS4 w/ HB 0.8      | 14.35           | 9.25        | 7.46                                         | 6.68 |    | 6.19 5.47 5.29 5.24 |    |    |
| PLMS4 w/ HB 0.9      | 11.66           | 8.16        | 6.69                                         | 6.21 |    | 5.78 5.29 5.19 5.17 |    |    |
| GHVB 3.8             | 10.99           | 7.93        | 6.63                                         | 6.19 |    | 5.80 5.31 5.22 5.18 |    |    |
| GHVB 3.9             | 11.67           | 8.29        | 6.83                                         | 6.30 |    | 5.87 5.31 5.22 5.18 |    |    |

Table 2: FID scores on DiT-XL

<span id="page-23-4"></span>

|                 | Number of Steps |                     |    |    |  |  |
|-----------------|-----------------|---------------------|----|----|--|--|
| Method          | 6               | 8                   | 10 | 20 |  |  |
| DDIM            |                 | 0.36 0.56 0.67 0.79 |    |    |  |  |
| DPM-Solver++    |                 | 0.63 0.75 0.79 0.81 |    |    |  |  |
| LTSP4           |                 | 0.67 0.74 0.78 0.81 |    |    |  |  |
| PLMS4           |                 | 0.68 0.75 0.78 0.81 |    |    |  |  |
| PLMS4 w/ HB 0.8 |                 | 0.68 0.77 0.79 0.81 |    |    |  |  |
| PLMS4 w/ HB 0.9 |                 | 0.70 0.77 0.79 0.81 |    |    |  |  |
| GHVB3.8         |                 | 0.71 0.77 0.79 0.81 |    |    |  |  |
| GHVB3.9         |                 | 0.70 0.76 0.78 0.81 |    |    |  |  |

Table 3: Precision on DiT-XL

|                 | Number of Steps |                     |    |    |  |  |  |
|-----------------|-----------------|---------------------|----|----|--|--|--|
| Method          | 6               | 8                   | 10 | 20 |  |  |  |
| DDIM            |                 | 0.60 0.64 0.65 0.67 |    |    |  |  |  |
| DPM-Solver++    |                 | 0.67 0.68 0.68 0.68 |    |    |  |  |  |
| LTSP4           |                 | 0.70 0.70 0.69 0.68 |    |    |  |  |  |
| PLMS4           |                 | 0.70 0.70 0.69 0.68 |    |    |  |  |  |
| PLMS4 w/ HB 0.8 |                 | 0.68 0.68 0.67 0.68 |    |    |  |  |  |
| PLMS4 w/ HB 0.9 |                 | 0.69 0.69 0.67 0.68 |    |    |  |  |  |
| GHVB3.8         |                 | 0.69 0.70 0.68 0.68 |    |    |  |  |  |
| GHVB3.9         |                 | 0.70 0.70 0.69 0.68 |    |    |  |  |  |

Table 4: Recall on DiT-XL

## <span id="page-23-2"></span>J Extended Comparison on Text-to-Image Comparison

To provide a more comprehensive evaluation of the methods discussed in Section [5.1,](#page-7-1) we utilize a fine-tuned variant of Stable-Diffusion called Anything V4. We consider full-path samples generated by PLMS4 [\[20\]](#page-10-10) at 1,000 steps as reference solutions. The performance of each method is evaluated by measuring the image similarity between the generated samples produced using a reduced number of steps and the reference samples. Importantly, both sets of samples originate from identical initial noise maps. This comparison allows us to assess how well the solution from each configuration matches the full-path reference solution.

<span id="page-23-3"></span><sup>9</sup> <https://github.com/facebookresearch/DiT>

<span id="page-24-0"></span>![](_page_24_Figure_0.jpeg)

Figure 17: Comparison of samples generated from DiT-XL with a guidance scale of 3, using different sampling methods and sampling steps.

To quantify image similarity, we use the Learned Perceptual Image Patch Similarity (LPIPS) [\[48\]](#page-11-17), where lower values indicate higher similarity. Additionally, we measure similarity using the L2 norm in the latent space, as discussed in Section [4.2,](#page-6-4) again with lower values indicating higher similarity. The outcomes of these analyses are visually presented in Figure [18.](#page-25-1)

Discrepancies between the numerical solutions and the 1,000-step reference solution can arise from two primary factors: the accuracy of the employed method and the presence of divergence artifacts. Notably, in this particular context, divergence artifacts tend to outweigh errors stemming from method accuracy. Consequently, higher-order methods exhibit greater discrepancies in both LPIPS and L2 similarity measurements. It is worth highlighting that our techniques demonstrate a remarkable ability to minimize deviations from the reference solution in the majority of cases. Furthermore, Figure [18](#page-25-1) consistently demonstrates the superiority of our techniques compared to other methods, as evidenced by the lower LPIPS and L2 similarity scores. These results indicate that our techniques effectively both reduce divergence artifacts and handle errors related to method accuracy.

<span id="page-25-1"></span>![](_page_25_Figure_0.jpeg)

Figure 18: Comparison of LPIPS in the image space and L2 distance in the latent space across different sampling methods, with and without the utilization of our momentum techniques. The experimental setting is similar to that in Section 5.1.

# <span id="page-25-0"></span>K Factors Contributing to Artifact Occurrence in Fine-tuned Diffusion Models

This section investigates factors that influence the occurrence of divergence artifacts in diffusion sampling, namely the number of steps, guidance scale, and the choice of diffusion models. The analysis includes a qualitative assessment that compares the results obtained from Stable Diffusion 1.5 (original) (Figure 20) with three fine-tuned diffusion models designed for specific purposes: generating Midjourney-style images (Figure 22), Japanese animation-style images (Figure 24), and photorealistic images (Figure 26).

Our observations reveal that several factors contribute to the occurrence of artifacts in diffusion sampling, including the number of steps, guidance scale, and choice of diffusion models. Insufficient numbers of steps and high guidance scales positively correlate with the presence of divergence artifacts in the generated samples. Fine-tuned models exhibit a higher sensitivity to these factors, resulting in a greater incidence of artifacts compared to Stable Diffusion 1.5. Consistent with the findings presented in Section 3.2, reducing the number of steps increases the likelihood of artifact occurrence. Furthermore, increasing the guidance scale amplifies the magnitude of eigenvalues, contributing to the presence of artifacts.

The choice of diffusion model also has an impact on artifact occurrence. Stable Diffusion 1.5 exhibits the fewest artifacts compared to the fine-tuned models, which demonstrate a higher incidence of divergence artifacts. Among the fine-tuned models, Openjourney demonstrates the lowest occurrence of artifacts, while also producing results that look similar to those obtained using the original Stable Diffusion 1.5 model. This suggests that extensive changes to the model may alter the eigenvalues and result in an increased presence of artifacts.

Additionally, we present the results of our techniques for handling divergence artifacts in Figure 21, 23, 25, and 27. The choice of the parameter  $\beta$  plays a crucial role in achieving a balance between reducing artifacts and maintaining accuracy, with its optimal value being influenced by the chosen guidance scale and diffusion model.

<span id="page-25-2"></span><sup>&</sup>lt;sup>10</sup>https://huggingface.co/runwayml/stable-diffusion-v1-5

![](_page_26_Figure_0.jpeg)

Figure 19: Comparison of samples generated from Stable Diffusion 1.5 [10](#page-25-2)with different sampling steps and method orders. The guidance scale is fixed at 10. Prompt: "A cat sitting on a window sill"

## <span id="page-26-0"></span>L Elaboration on the Order of Convergence Approximation

In Appendix [F,](#page-18-0) we explored the theoretical aspects of the order of convergence for numerical methods. In this section, we will delve into the estimation of the order of convergence specifically for GHVB in Section [4.2.](#page-6-4)

To assess the order of convergence, we focus on the error e, referred to as the global truncation error. This error is quantified by measuring the absolute difference in the latent space between the numerical solution and an approximate exact solution obtained through 1,000-step PLMS4 sampling. The order of convergence for a numerical method is defined as q, where the error e follows the relationship e = O(δ q ), with δ representing the step size.

To estimate the order of convergence practically, we adopt a straightforward approach. It involves selecting two distinct step sizes, denoted as δnew and δold, and computing the corresponding errors enew and eold. These errors can be approximated using the following formulas:

$$e_{\text{new}} \approx C_{\text{new}}(\delta_{\text{new}})^q, \qquad e_{\text{old}} \approx C_{\text{old}}(\delta_{\text{old}})^q$$
 (64)

Here, we make the assumption that Cnew is approximately equal to Cold. By taking the ratio of enew to eold, we obtain:

<span id="page-26-1"></span><sup>11</sup><https://huggingface.co/runwayml/stable-diffusion-v1-5>

<span id="page-26-2"></span><sup>12</sup><https://huggingface.co/prompthero/openjourney>

<span id="page-26-3"></span><sup>13</sup><https://huggingface.co/hakurei/waifu-diffusion>

<span id="page-26-4"></span><sup>14</sup><https://huggingface.co/dreamlike-art/dreamlike-photoreal-2.0>

<span id="page-27-0"></span>![](_page_27_Figure_0.jpeg)

Figure 20: Comparison of samples generated from Stable Diffusion 1.5 [11](#page-26-1)using PLMS4 [\[20\]](#page-10-10) with different sampling steps and guidance scale.

<span id="page-27-1"></span>![](_page_27_Figure_2.jpeg)

Figure 21: Comparison of samples generated from Stable Diffusion 1.5 using PLMS4 with HB β under various sampling steps and guidance scale s. Specifically, we employ β = 0.9 for s = 7.5, β = 0.8 for s = 15, and β = 0.6 for s = 22.5 to account for the varying degrees of artifact manifestation associated with each guidance scale.

<span id="page-28-0"></span>![](_page_28_Figure_0.jpeg)

Figure 22: Comparison of samples generated from Openjourney [12](#page-26-2)using PLMS4 [\[20\]](#page-10-10) with different sampling steps and guidance scale.

<span id="page-28-1"></span>![](_page_28_Figure_2.jpeg)

Figure 23: Comparison of samples generated from Openjourney using PLMS4 with HB β under various sampling steps and guidance scale s. Specifically, we employ β = 0.8 for s = 7.5, β = 0.6 for s = 15, and β = 0.6 for s = 22.5 to account for the varying degrees of artifact manifestation associated with each guidance scale.

<span id="page-29-0"></span>![](_page_29_Figure_0.jpeg)

Figure 24: Comparison of samples generated from Waifu Diffusion V1.4 [13](#page-26-3)using PLMS4 [\[20\]](#page-10-10) with different sampling steps and guidance scale.

<span id="page-29-1"></span>![](_page_29_Figure_2.jpeg)

Figure 25: Comparison of samples generated from Waifu Diffusion V1.4 using PLMS4 with HB β under various sampling steps and guidance scale s. Specifically, we employ β = 0.8 for s = 7.5, β = 0.7 for s = 15, and β = 0.6 for s = 22.5 to account for the varying degrees of artifact manifestation associated with each guidance scale.

<span id="page-30-0"></span>![](_page_30_Figure_0.jpeg)

Figure 26: Comparison of samples generated from Dreamlike Photoreal 2.0 [14](#page-26-4)using PLMS4 [\[20\]](#page-10-10) with different sampling steps and guidance scale.

<span id="page-30-1"></span>![](_page_30_Figure_2.jpeg)

Figure 27: Comparison of samples generated from Dreamlike Photoreal V2.0 using PLMS4 with HB β under various sampling steps and guidance scale s. Specifically, we employ β = 0.7 for s = 7.5, β = 0.6 for s = 15, and β = 0.6 for s = 22.5 to account for the varying degrees of artifact manifestation associated with each guidance scale.

![](_page_31_Figure_0.jpeg)

Figure 28: Comparison of samples generated from Stable Diffusion 1.5 using GHVB(3.0 + β) under various sampling steps and guidance scale s. Specifically, we employ β = 0.5 for s = 7.5, β = 0.2 for s = 15, and β = 0.1 for s = 22.5 to account for the varying degrees of artifact manifestation associated with each guidance scale.

$$\frac{e_{\text{new}}}{e_{\text{old}}} \approx \left(\frac{\delta_{\text{new}}}{\delta_{\text{old}}}\right)^q$$
 (65)

Consequently, we can estimate the order of convergence, denoted as q, by evaluating the logarithmic ratio of errors and step sizes:

$$q \approx \frac{\log(e_{\text{new}}/e_{\text{old}})}{\log(\delta_{\text{new}}/\delta_{\text{old}})}$$
(66)

In our investigation of GHVB in Section [4.2,](#page-6-4) we conducted sampling experiments using 20, 40, 80, 160, 320, and 640 steps. This choice of an exponential sequence for the number of steps was intentional, as it allowed us to approximate δnew/δold ≈ 1/2. By doing so, we facilitated the estimation process. The results, representing the approximated order of convergence for GHVB, are visually depicted in Figure [12.](#page-9-9)

## <span id="page-31-0"></span>M Ablation Study on HB Momentum

Incorporating Polyak's Heavy Ball (HB) momentum directly into existing diffusion sampling methods is a more straightforward approach to mitigating divergence artifacts than GHVB. This can be achieved by modifying a few lines of code. In this section, we conduct a comprehensive analysis of the convergence speed of this approach.

To evaluate its effectiveness, we generate target results using the 1,000-step PLMS4 method. We compare the target results with those obtained from several methods with and without HB momentum, using LPIPS in the image space and L2 in the latent space. We then estimate their orders of convergence, as explained in Appendix [L.](#page-26-0) The results of this analysis are visually presented in Figure [29.](#page-32-1)

<span id="page-32-1"></span>![](_page_32_Figure_0.jpeg)

Figure 29: Comparison of LPIPS, mean L2 distance, and order of convergence of HB when using different damping coefficients. Statistical means are averaged from 160 initial latent codes.

<span id="page-32-2"></span>![](_page_32_Figure_2.jpeg)

Figure 30: Comparison of LPIPS, mean L2 distance, and order of convergence of Nesterov's momentum when using different damping coefficients. Statistical means are averaged from 160 initial latent codes.

In contrast to the interpolation-like behavior observed in Figure 12 for GHVB, we observe that the use of HB momentum leads to an increase in both the LPIPS score and the L2 distance when selecting values of  $\beta$  that are less than 1. This is even worse than the 1<sup>st</sup>-order method DDIM when  $\beta$  is below 0.7. These findings indicate a deviation from the desired convergence behavior, highlighting a potential decrease in solution accuracy, even though HB momentum has been shown to successfully mitigate divergence artifacts.

Additionally, we find that the numerical orders of convergence also tend to approach the same value. These observations align with our analysis in Theorem 1 of Appendix F, indicating that when  $\beta$  deviates from 1, the employed approach exhibits 1<sup>st</sup>-order convergence and is unable to achieve high-order convergence. These conclusions emphasize the importance of carefully considering the choice of  $\beta$  in order to strike a balance between convergence speed and solution quality. Further details and insights into the performance of the HB momentum approach can be obtained from Figure 29, enhancing our understanding of its behavior within the context of the studied problem.

#### <span id="page-32-0"></span>N Ablation Study on Nesterov Momentum

In Appendix E, we investigated the potential of incorporating different types of momentum, such as Nesterov's momentum, into existing diffusion sampling methods to mitigate divergence artifacts. Similar to the analysis conducted in Section 5.4 and Appendix M, the primary objective of this section is to explore the convergence speed of Nesterov's momentum by comparing two key metrics: LPIPS in the image space and L2 in the latent space.

Figure 30 presents the results, which reveal intriguing parallels with the behavior of HB momentum observed in Figure 29. When Nesterov's momentum is applied to the PLMS2 method, the accuracy of the model progressively diminishes as the value of  $\beta$  deviates from 1, as indicated by the corresponding increase in both LPIPS and L2 metrics. Notably, the model's accuracy drops below that of the DDIM when  $\beta$  falls below 0.5.

Furthermore, our analysis of the order of convergence demonstrates that Nesterov's momentum does not achieve a high order of convergence, similar to HB momentum. These findings emphasize the importance of carefully considering the choice of momentum method, along with the specific values assigned to  $\beta$ , in order to strike an optimal balance between convergence speed and solution quality.

<span id="page-33-2"></span>

|                                                                                               | Number of steps |    |    |                                                                               |    |    |
|-----------------------------------------------------------------------------------------------|-----------------|----|----|-------------------------------------------------------------------------------|----|----|
| Method                                                                                        | 10              | 15 | 20 | 25                                                                            | 30 | 60 |
| DPM                                                                                           |                 |    |    | 1.113 ± .090 1.369 ± .102 1.087 ± .083 0.972 ± .075 0.919 ± .068 0.869 ± .067 |    |    |
| DPM w/ HB 0.8                                                                                 |                 |    |    | 0.974 ± .078 1.057 ± .079 0.916 ± .072 0.857 ± .070 0.831 ± .067 0.834 ± .065 |    |    |
| DPM w/ HB 0.9                                                                                 |                 |    |    | 1.043 ± .082 1.186 ± .088 0.986 ± .075 0.921 ± .073 0.867 ± .068 0.844 ± .065 |    |    |
| PLMS4 w/ HB 0.8 1.958 ± .116 1.469 ± .105 1.213 ± .097 1.060 ± .087 0.963 ± .076 0.838 ± .063 |                 |    |    |                                                                               |    |    |
| PLMS4 w/ HB 0.9 2.499 ± .118 1.888 ± .112 1.534 ± .112 1.270 ± .104 1.116 ± .091 0.887 ± .066 |                 |    |    |                                                                               |    |    |
| PLMS4                                                                                         |                 |    |    | 3.149 ± .116 2.460 ± .116 1.911 ± .115 1.597 ± .116 1.372 ± .106 0.957 ± .075 |    |    |

Table 5: 95% confidence intervals for the magnitude scores of HB (Figure [7\)](#page-7-3)

<span id="page-33-3"></span>

|        | Number of steps |    |                                                                                       |    |    |    |
|--------|-----------------|----|---------------------------------------------------------------------------------------|----|----|----|
| Method | 10              | 15 | 20                                                                                    | 25 | 30 | 60 |
| DDIM   |                 |    | 0.844 ± .076 0.765 ± .064 0.728 ± .060 0.744 ± .063 0.761 ± .062 0.778 ± .062         |    |    |    |
|        |                 |    | GHVB2.1 1.238 ± .097 0.924 ± .072 0.832 ± .067 0.820 ± .066 0.825 ± .066 0.829 ± .064 |    |    |    |
|        |                 |    | GHVB2.3 1.291 ± .102 0.952 ± .072 0.872 ± .070 0.836 ± .064 0.831 ± .066 0.828 ± .062 |    |    |    |
|        |                 |    | GHVB2.5 1.392 ± .103 1.016 ± .081 0.907 ± .071 0.851 ± .069 0.842 ± .065 0.845 ± .063 |    |    |    |
|        |                 |    | GHVB2.7 1.514 ± .105 1.095 ± .085 0.968 ± .077 0.877 ± .068 0.864 ± .067 0.826 ± .063 |    |    |    |
|        |                 |    | GHVB2.9 1.673 ± .107 1.203 ± .090 1.023 ± .079 0.934 ± .075 0.901 ± .071 0.835 ± .063 |    |    |    |
| PLMS4  |                 |    | 3.149 ± .116 2.460 ± .116 1.911 ± .115 1.597 ± .116 1.372 ± .106 0.957 ± .075         |    |    |    |

Table 6: 95% confidence intervals for the magnitude scores of GHVB (Figure [8\)](#page-7-4)

## <span id="page-33-0"></span>O Statistical Reports

In this section, we present detailed statistical reports for the experiments conducted in Section [5.1](#page-7-1) and Section [4.2.](#page-6-4) These reports provide detailed information, including mean values and their corresponding 95% confidence intervals, to offer a thorough understanding of the experimental results.

Firstly, we focus on the experiment related to mitigating the magnitude score in Section [5.1.](#page-7-1) The results depicted in Figure [7](#page-7-3) are presented in Table [5.](#page-33-2) Additionally, the outcomes illustrated in Figure [8](#page-7-4) are reported in Table [6.](#page-33-3) For the ablation study of GHVB in Section [4.2,](#page-6-4) we provide the results shown in Figure [11](#page-9-8) in Table [7.](#page-33-4) Similarly, the findings presented in Figure [12](#page-9-9) are reported in Table [8.](#page-34-3)

Furthermore, we include a runtime comparison of each sampling method in Table [9,](#page-34-4) detailing the wall clock time required for each method. The results indicate that all the methods exhibit similar sampling times, ensuring a fair comparison across the different approaches.

## <span id="page-33-1"></span>P Ablation on Magnitude Score

In this section, our objective is to provide further verification and justification of the experiment conducted in Section [5.1](#page-7-1) by exploring various parameter settings for the magnitude score and assessing their effects on the selected model.

<span id="page-33-4"></span>

|        | Number of steps |    |    |                                                                                                    |     |     |     |
|--------|-----------------|----|----|----------------------------------------------------------------------------------------------------|-----|-----|-----|
| Method | 10              | 20 | 40 | 80                                                                                                 | 160 | 320 | 640 |
| DDIM   |                 |    |    | 0.584 ± .034 0.409 ± .029 0.304 ± .029 0.210 ± .026 0.139 ± .022 0.085 ± .014 0.048 ± .009         |     |     |     |
|        |                 |    |    | GHVB1.1 0.592 ± .034 0.406 ± .029 0.295 ± .030 0.189 ± .026 0.113 ± .020 0.054 ± .010 0.019 ± .005 |     |     |     |
|        |                 |    |    | GHVB1.3 0.609 ± .035 0.410 ± .029 0.276 ± .029 0.158 ± .023 0.086 ± .017 0.030 ± .007 0.009 ± .003 |     |     |     |
|        |                 |    |    | GHVB1.5 0.624 ± .036 0.409 ± .029 0.261 ± .029 0.145 ± .023 0.067 ± .014 0.021 ± .005 0.006 ± .002 |     |     |     |
|        |                 |    |    | GHVB1.7 0.645 ± .037 0.411 ± .030 0.254 ± .028 0.133 ± .023 0.053 ± .011 0.016 ± .005 0.004 ± .002 |     |     |     |
|        |                 |    |    | GHVB1.9 0.663 ± .037 0.414 ± .030 0.246 ± .028 0.123 ± .021 0.044 ± .009 0.013 ± .004 0.003 ± .001 |     |     |     |
| PLMS2  |                 |    |    | 0.676 ± .038 0.418 ± .030 0.246 ± .028 0.119 ± .021 0.041 ± .009 0.011 ± .003 0.003 ± .001         |     |     |     |

Table 7: 95% confidence intervals for L2 norm of GHVB (Figrue [11\)](#page-9-8)

<span id="page-34-3"></span>

|        | Number of steps (knew)                                                   |    |     |     |     |  |  |
|--------|--------------------------------------------------------------------------|----|-----|-----|-----|--|--|
| Method | 40                                                                       | 80 | 160 | 320 | 640 |  |  |
|        | GHVB0.5 0.247 ± .030 0.235 ± .029 0.351 ± .045 0.450 ± .057 0.474 ± .057 |    |     |     |     |  |  |
|        | GHVB1.5 0.550 ± .072 0.717 ± .086 0.922 ± .089 1.337 ± .102 1.519 ± .102 |    |     |     |     |  |  |
|        | GHVB2.5 0.624 ± .077 1.121 ± .115 1.546 ± .132 1.906 ± .153 1.846 ± .147 |    |     |     |     |  |  |
|        | GHVB3.5 0.459 ± .063 0.920 ± .107 1.877 ± .170 1.960 ± .147 1.779 ± .163 |    |     |     |     |  |  |

<span id="page-34-4"></span>Table 8: 95% confidence intervals for the numerical orders of convergence of GHVB (Figure [12\)](#page-9-9)

|                                       | Number of steps |                |    |
|---------------------------------------|-----------------|----------------|----|
| Method                                | 15              | 30             | 60 |
| DPM-Solver++                          |                 | 2.49 4.84 9.54 |    |
| DPM-Solver++ w/ HB 0.9 2.49 4.84 9.54 |                 |                |    |
| PLMS4                                 |                 | 2.49 4.84 9.54 |    |
| PLMS4 w/ HB 0.9                       |                 | 2.46 4.79 9.43 |    |
| PLMS4 w/ NT 0.9                       |                 | 2.53 4.93 9.70 |    |
| GHVB3.9                               |                 | 2.50 4.84 9.54 |    |

Table 9: Comparison of the average sampling time per image (in seconds) when using different numbers of steps in Stable Diffusion 1.5 on an NVIDIA GeForce RTX 3080. The time differences are marginal.

#### <span id="page-34-0"></span>P.1 Results with Alternative Parameter Settings

To gain deeper insights into the integration of momentum into sampling methods, we analyze the results of the magnitude scores depicted in Figure [7](#page-7-3) (Section [5.1\)](#page-7-1). This analysis involves varying the threshold τ and the kernel size k for max-pooling in the calculation of the magnitude score. By investigating different parameter settings, we aim to validate the outcomes of the experiment and uncover the scaling impact of the magnitude score. The results, shown in Figure [32,](#page-37-0) highlight the influence of threshold τ and kernel size k on the magnitude score. It is important to note that while extreme values of τ or k may introduce ambiguity in interpreting the outcomes, the overall observed trends remain consistent.

#### <span id="page-34-1"></span>P.2 Results on Alternative Models

In this section, we present the findings from our analysis conducted on alternative diffusion models, namely Stable Diffusion 1.5, Waifu Diffusion V1.4, and Dreamlike Photoreal 2.0. The primary aim of this investigation is to assess the impact of different models on the magnitude score and determine whether the trends identified in Section [5.1](#page-7-1) hold across diverse model architectures.

For this analysis, we employed the same magnitude score parameters as in Section [5.1.](#page-7-1) The results of our examination are illustrated in Figure [33,](#page-37-1) which showcases the magnitude scores for each model. One important observation is that the change in model architecture only affects the scale of the magnitude score, while the overall trend remains consistent across all models.

## <span id="page-34-2"></span>Q Frequently Asked Questions

#### Q: What does the term "divergence artifacts" refer to?

In this paper, the term "divergence artifacts" is used to describe visual anomalies that occur when the numerical solution diverges, resulting in unusually large magnitudes of the results. In the context of latent-based diffusion, we specifically define divergence artifacts as visual artifacts caused by latent codes with magnitudes that exceed the usual range. These artifacts commonly arise when the stability region of the numerical method fails to handle all eigenvalues of the system, leading to a divergent numerical solution. To visually demonstrate the presence of divergence artifacts, we have included Figure [34.](#page-38-0) This illustration showcases the process of starting with diffusion results and subsequently scaling the latent code within a 4 × 4 square located at the center of the latent image. This scaling is achieved by multiplying the latent code with a constant factor. As a result of this manipulation, divergence artifacts become distinctly visible, particularly at the center of the resulting image. This

illustration provides a clear representation of the impact that scaling the latent code can have on the occurrence of divergence artifacts.

## Q: Can we directly interpolate two existing numerical methods instead of using the GHVB method?

Indeed, this is possible. However, the order of the resulting method will be the lowest order of the two methods. To illustrate this point, let us consider a direct interpolation between the  $1^{st}$ -order Euler method (AB1) and the  $2^{nd}$ -order Adams-Bashford method (AB2), expressed as follows:

$$x_{n+1} = x_n + \delta \left( (1 - \beta)f(x_n) + \beta \frac{3}{2}f(x_n) - \beta \frac{1}{2}f(x_{n-1}) \right)$$
(67)

As outlined in Appendix F, despite the orders of the interpolated methods, the resulting method is a 1<sup>st</sup>-order method.

![](_page_36_Figure_0.jpeg)

(c) Prompt: "A painting of an old European castle in a deep forest with a Blood Moon in the background"

Figure 31: Comparison between two variations of momentum: (a) Polyak's Heavy Ball (HB) and (b) Nesterov (NT). These momentum variations are applied to PLMS4 [\[20\]](#page-10-10) on a fine-tuned Stable Diffusion model called Anything V4 [\[23\]](#page-10-12) with 15 sampling steps and a guidance scale of 15. Both variations effectively reduce artifacts. However, the choice of the effectiveness parameter β might differ due to the distinct shapes of their respective stability regions.

<span id="page-37-0"></span>![](_page_37_Figure_0.jpeg)

Figure 32: Comparison of magnitude scores on Anything V4 on different combinations of threshold  $\tau$  and kernel size k used in max-pooling. #samples = 160

<span id="page-37-1"></span>![](_page_37_Figure_2.jpeg)

Figure 33: Comparison of magnitude scores in different diffusion models. #samples = 160

<span id="page-38-0"></span>![](_page_38_Figure_0.jpeg)

Figure 34: Visualization of divergence artifacts through latent code scaling. The figure illustrates the diffusion results obtained by scaling the latent code within a 4 × 4 square positioned at the center of the latent image. This scaling process involves multiplying the latent code with a constant factor. As a consequence, the resulting image showcases the emergence of divergence artifacts, which are particularly prominent at the center. The samples presented in this figure were generated using PLMS4 [\[20\]](#page-10-10) with a guidance scale of 15 and 250 sampling steps. Prompt: "A beautiful illustration of a couple looking at fireworks in a summer festival in Japan"