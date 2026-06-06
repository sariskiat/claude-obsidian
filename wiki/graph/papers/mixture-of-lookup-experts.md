---
type: paper
slug: mixture-of-lookup-experts
title: Mixture of Lookup Experts
authors: Shibo Jie, Yehui Tang, Kai Han, Yitong Li, Duyu Tang, Zhi-Hong Deng, Yunhe Wang
source_path: /Users/saris.kia.adm/.paper-scholar/mixture-of-lookup-experts/2503.15798.md
ingested_at: '2026-06-05 05:48:44'
authors_list: []
sections:
- id: 453
  heading: Abstract
  role: section
  order_index: 0
  summary: Mixture-of-Experts (MoE) activates only a subset of experts during inference, allowing the model to maintain low inference FLOPs and latency even as the parameter count scales up. However, since MoE dynamically selects the experts, all the experts need to be loaded into VRAM.
- id: 454
  heading: 1. Introduction
  role: section
  order_index: 1
  summary: Scaling laws indicate that, with sufficient data for training, the performance of large language models (LLMs) improves as the model size increases [\(Kaplan et al.,](#page-9-0) [2020\)](#page-9-0). However, larger LLMs also result in slower inference speeds, which can degrade the user experience.
- id: 455
  heading: 2. Related Work
  role: section
  order_index: 2
  summary: ''
- id: 456
  heading: 2.1. Mixture-of-Experts
  role: section
  order_index: 3
  summary: The concept of MoE was initially introduced by [Jacobs](#page-8-3) [et al.](#page-8-3) [\(1991\)](#page-8-3); [Jordan & Jacobs](#page-8-4) [\(1994\)](#page-8-4) and has been widely explored and developed through subsequent research [\(Col](#page-8-5)[lobert et al.,](#page-8-5) [2002;](#page-8-5) [Rasmussen & Ghahramani,](#page-9-3) [2001;](#page-9-3) [Shah](#page-9-4)[baba & Neal,](#page-9-4) [2009;](#page-9-4) [Eigen et al.,](#page-8-6) [2014;](#page-8-6) [Theis & Bethge,](#page-9-5) [2015;](#page-9-5) [Deisenroth & Ng,](#page-8-7) [2015;](#page-8-7) [Aljundi et al.,](#page-8-8) [2017;](#page-8-8) [Shazeer](#page-9-6) [et al.,](#page-9-6) [2017\)](#page-9-6). MoE posits that different parts of the model, *i.e.*, the experts, focus on distinct tasks or encapsulate different kinds of knowledge.
- id: 457
  heading: 2.2. Expert Offloading
  role: section
  order_index: 4
  summary: Offloading techniques typically transfer a portion of the model parameters to CPU RAM or disk when GPU memory is insufficient. However, current mainstream offloading frameworks, such as Zero-Infinity [\(Rajbhandari et al.,](#page-9-9) [2021\)](#page-9-9), are designed for dense LLMs and load model parameters layer by layer on-demand.
- id: 458
  heading: 3. Mixture of Lookup Experts
  role: section
  order_index: 5
  summary: ''
- id: 459
  heading: 3.1. Preliminary
  role: section
  order_index: 6
  summary: First, we briefly introduce the structure of MoE and the challenges it faces during inference.
- id: 460
  heading: 3.2. Training Phase
  role: section
  order_index: 7
  summary: As illustrated in Figure 2, during training, MoLE and MoE have similar structures, including N routed experts
- id: 461
  heading: 3.3. Inference Phase
  role: section
  order_index: 8
  summary: After training, MoLE can be directly used for inference like other LLMs. However, to further reduce VRAM overhead, we can re-parameterize the experts.
- id: 462
  heading: 3.4. Complexity Analysis
  role: section
  order_index: 9
  summary: Consider an MoE layer with MLP-based FFNs as experts. Let the hidden layer dimension of the routed experts be Dr, and the hidden layer dimension of the shared experts be Ds.
- id: 463
  heading: 4. Experiments
  role: section
  order_index: 10
  summary: ''
- id: 464
  heading: 4.1. Experimental Setup
  role: section
  order_index: 11
  summary: '**Model Architectures.** As shown in Table 2, we implement models with activation parameter counts of 160M, 410M, and 1B. For the dense model, we basically follow the Pythia (Biderman et al., 2023) setup.'
- id: 465
  heading: 4.2. Main Results
  role: section
  order_index: 12
  summary: As shown in Table 3, both MoE and MoLE significantly improve performance over the dense baseline. In the com-
- id: 466
  heading: 4.3. Ablation Experiments
  role: section
  order_index: 13
  summary: '**Training loss.** Unlike MoE, MoLE is a fully differentiable model, so during training, we do not encounter issues like router collapse or instability. Therefore, we do not use any additional auxiliary losses.'
- id: 467
  heading: 4.4. Efficiency
  role: section
  order_index: 14
  summary: We measure the per-step decoding latency of models with 410M activated parameters on NVIDIA V100 GPU using Huggingface's transformers package. Since the specific speed of parameter loading is largely influenced by the underlying implementation, we estimate the latency of loading parameters based on the maximum PCIe bandwidth of the V100, which is 16GB/s.
- id: 468
  heading: 4.5. Reducing the Size of LUTs
  role: section
  order_index: 15
  summary: Although MoLE significantly reduces data transfer in offloading scenarios compared to MoE, it has a larger storage footprint. While storage space may not be as constrained as VRAM, reducing the size of the LUTs can still alleviate deployment burdens.
- id: 469
  heading: 5. Conclusion
  role: section
  order_index: 16
  summary: In this paper, we address the issues of high memory consumption and loading latency in MoE by proposing MoLE, a novel language model architecture. MoLE restricts the input to experts to a limited set (embedding tokens), allowing the experts to be re-parameterized into LUTs before inference, thus avoiding the need to load expert parameters.
- id: 470
  heading: Impact Statement
  role: section
  order_index: 17
  summary: This paper advances LLMs, which have potential societal impacts, including concerns around bias, misinformation, and accessibility. Continued ethical oversight and interdisciplinary collaboration are essential as the field evolves.
- id: 471
  heading: References
  role: section
  order_index: 18
  summary: '- <span id="page-8-8"></span>Aljundi, R., Chakravarty, P., and Tuytelaars, T. Expert gate: Lifelong learning with a network of experts.'
- id: 472
  heading: Appendix
  role: section
  order_index: 19
  summary: ''
- id: 473
  heading: A. Pseudocode
  role: section
  order_index: 20
  summary: ''
- id: 474
  heading: A.1. Training Phase
  role: section
  order_index: 21
  summary: '``` class MoleDecoderLayer(nn.Module): def __init__(self, config): super().__init__() self.self_attn = Attention(config) self.shared_expert = MLP(config) self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False) self.routed_expert = nn.ModuleList([MLP(config) for _ in config.num_experts]) self.input_layernorm = RMSNorm(config.hidden_size) self.post_attention_layernorm = RMSNorm(config.hidden_size) self.expert_layernorm = RMSNorm(config.hidden_size) def forward(self, hidden_states, embedding_states): ''''''Attention'''''' residual = hidden_states hidden_states = self.input_layernorm(hidden_states) hidden_states = self.self_attn(hidden_states) hidden_states = residual + hidden_states ''''''Shared Expert'''''' residual = hidden_states hidden_states = self.post_attention_layernorm(hidden_states) shared_output = self.shared_expert(hidden_states) ''''''Routed Expert'''''' router_value = nn.functional.softmax(self.router(hidden_states), dim=-1) embedding_states = self.expert_layernorm(embedding_states) routed_output = torch.stack([expert(embedding_states) for expert in self.routed_expert], dim=2) routed_output = (routed_output * router_value.unsqueeze(-1)).sum(dim=2) hidden_states = residual + shared_output + routed_output return hidden_states ```'
- id: 475
  heading: A.2. Inference Phase
  role: section
  order_index: 22
  summary: '``` class MoleDecoderLayer(nn.Module): def __init__(self, config): super().__init__() self.self_attn = Attention(config) self.shared_expert = MLP(config) self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False) self.lut = LookupTable(config.vocab_size, config.num_experts * config.hidden_size) self.input_layernorm = RMSNorm(config.hidden_size) self.post_attention_layernorm = RMSNorm(config.hidden_size) def forward(self, hidden_states, input_ids): ''''''Lookup'''''' lookup_results = self.lut(input_ids).to(hidden_states.device, non_blocking=True) ''''''Attention'''''' residual = hidden_states hidden_states = self.input_layernorm(hidden_states) hidden_states = self.self_attn(hidden_states) hidden_states = residual + hidden_states ''''''Shared Expert'''''' residual = hidden_states ```'
- id: 476
  heading: Mixture of Lookup Experts
  role: section
  order_index: 23
  summary: '``` hidden_states = self.post_attention_layernorm(hidden_states) shared_output = self.shared_expert(hidden_states) ''''''Routed Expert'''''' router_value = nn.functional.softmax(self.router(hidden_states), dim=-1) lookup_results = lookup_results.view(-1, config.num_experts, config.hidden_size) routed_output = (lookup_results * router_value.unsqueeze(-1)).sum(dim=2) hidden_states = residual + shared_output + routed_output return hidden_states ```'
- id: 477
  heading: B. Hyper-parameters
  role: section
  order_index: 24
  summary: ''
---

# Mixture of Lookup Experts

## [section] Abstract
Mixture-of-Experts (MoE) activates only a subset of experts during inference, allowing the model to maintain low inference FLOPs and latency even as the parameter count scales up. However, since MoE dynamically selects the experts, all the experts need to be loaded into VRAM.

## [section] 1. Introduction
Scaling laws indicate that, with sufficient data for training, the performance of large language models (LLMs) improves as the model size increases [\(Kaplan et al.,](#page-9-0) [2020\)](#page-9-0). However, larger LLMs also result in slower inference speeds, which can degrade the user experience.

## [section] 2. Related Work


## [section] 2.1. Mixture-of-Experts
The concept of MoE was initially introduced by [Jacobs](#page-8-3) [et al.](#page-8-3) [\(1991\)](#page-8-3); [Jordan & Jacobs](#page-8-4) [\(1994\)](#page-8-4) and has been widely explored and developed through subsequent research [\(Col](#page-8-5)[lobert et al.,](#page-8-5) [2002;](#page-8-5) [Rasmussen & Ghahramani,](#page-9-3) [2001;](#page-9-3) [Shah](#page-9-4)[baba & Neal,](#page-9-4) [2009;](#page-9-4) [Eigen et al.,](#page-8-6) [2014;](#page-8-6) [Theis & Bethge,](#page-9-5) [2015;](#page-9-5) [Deisenroth & Ng,](#page-8-7) [2015;](#page-8-7) [Aljundi et al.,](#page-8-8) [2017;](#page-8-8) [Shazeer](#page-9-6) [et al.,](#page-9-6) [2017\)](#page-9-6). MoE posits that different parts of the model, *i.e.*, the experts, focus on distinct tasks or encapsulate different kinds of knowledge.

## [section] 2.2. Expert Offloading
Offloading techniques typically transfer a portion of the model parameters to CPU RAM or disk when GPU memory is insufficient. However, current mainstream offloading frameworks, such as Zero-Infinity [\(Rajbhandari et al.,](#page-9-9) [2021\)](#page-9-9), are designed for dense LLMs and load model parameters layer by layer on-demand.

## [section] 3. Mixture of Lookup Experts


## [section] 3.1. Preliminary
First, we briefly introduce the structure of MoE and the challenges it faces during inference.

## [section] 3.2. Training Phase
As illustrated in Figure 2, during training, MoLE and MoE have similar structures, including N routed experts

## [section] 3.3. Inference Phase
After training, MoLE can be directly used for inference like other LLMs. However, to further reduce VRAM overhead, we can re-parameterize the experts.

## [section] 3.4. Complexity Analysis
Consider an MoE layer with MLP-based FFNs as experts. Let the hidden layer dimension of the routed experts be Dr, and the hidden layer dimension of the shared experts be Ds.

## [section] 4. Experiments


## [section] 4.1. Experimental Setup
**Model Architectures.** As shown in Table 2, we implement models with activation parameter counts of 160M, 410M, and 1B. For the dense model, we basically follow the Pythia (Biderman et al., 2023) setup.

## [section] 4.2. Main Results
As shown in Table 3, both MoE and MoLE significantly improve performance over the dense baseline. In the com-

## [section] 4.3. Ablation Experiments
**Training loss.** Unlike MoE, MoLE is a fully differentiable model, so during training, we do not encounter issues like router collapse or instability. Therefore, we do not use any additional auxiliary losses.

## [section] 4.4. Efficiency
We measure the per-step decoding latency of models with 410M activated parameters on NVIDIA V100 GPU using Huggingface's transformers package. Since the specific speed of parameter loading is largely influenced by the underlying implementation, we estimate the latency of loading parameters based on the maximum PCIe bandwidth of the V100, which is 16GB/s.

## [section] 4.5. Reducing the Size of LUTs
Although MoLE significantly reduces data transfer in offloading scenarios compared to MoE, it has a larger storage footprint. While storage space may not be as constrained as VRAM, reducing the size of the LUTs can still alleviate deployment burdens.

## [section] 5. Conclusion
In this paper, we address the issues of high memory consumption and loading latency in MoE by proposing MoLE, a novel language model architecture. MoLE restricts the input to experts to a limited set (embedding tokens), allowing the experts to be re-parameterized into LUTs before inference, thus avoiding the need to load expert parameters.

## [section] Impact Statement
This paper advances LLMs, which have potential societal impacts, including concerns around bias, misinformation, and accessibility. Continued ethical oversight and interdisciplinary collaboration are essential as the field evolves.

## [section] References
- <span id="page-8-8"></span>Aljundi, R., Chakravarty, P., and Tuytelaars, T. Expert gate: Lifelong learning with a network of experts.

## [section] Appendix


## [section] A. Pseudocode


## [section] A.1. Training Phase
``` class MoleDecoderLayer(nn.Module): def __init__(self, config): super().__init__() self.self_attn = Attention(config) self.shared_expert = MLP(config) self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False) self.routed_expert = nn.ModuleList([MLP(config) for _ in config.num_experts]) self.input_layernorm = RMSNorm(config.hidden_size) self.post_attention_layernorm = RMSNorm(config.hidden_size) self.expert_layernorm = RMSNorm(config.hidden_size) def forward(self, hidden_states, embedding_states): '''Attention''' residual = hidden_states hidden_states = self.input_layernorm(hidden_states) hidden_states = self.self_attn(hidden_states) hidden_states = residual + hidden_states '''Shared Expert''' residual = hidden_states hidden_states = self.post_attention_layernorm(hidden_states) shared_output = self.shared_expert(hidden_states) '''Routed Expert''' router_value = nn.functional.softmax(self.router(hidden_states), dim=-1) embedding_states = self.expert_layernorm(embedding_states) routed_output = torch.stack([expert(embedding_states) for expert in self.routed_expert], dim=2) routed_output = (routed_output * router_value.unsqueeze(-1)).sum(dim=2) hidden_states = residual + shared_output + routed_output return hidden_states ```

## [section] A.2. Inference Phase
``` class MoleDecoderLayer(nn.Module): def __init__(self, config): super().__init__() self.self_attn = Attention(config) self.shared_expert = MLP(config) self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False) self.lut = LookupTable(config.vocab_size, config.num_experts * config.hidden_size) self.input_layernorm = RMSNorm(config.hidden_size) self.post_attention_layernorm = RMSNorm(config.hidden_size) def forward(self, hidden_states, input_ids): '''Lookup''' lookup_results = self.lut(input_ids).to(hidden_states.device, non_blocking=True) '''Attention''' residual = hidden_states hidden_states = self.input_layernorm(hidden_states) hidden_states = self.self_attn(hidden_states) hidden_states = residual + hidden_states '''Shared Expert''' residual = hidden_states ```

## [section] Mixture of Lookup Experts
``` hidden_states = self.post_attention_layernorm(hidden_states) shared_output = self.shared_expert(hidden_states) '''Routed Expert''' router_value = nn.functional.softmax(self.router(hidden_states), dim=-1) lookup_results = lookup_results.view(-1, config.num_experts, config.hidden_size) routed_output = (lookup_results * router_value.unsqueeze(-1)).sum(dim=2) hidden_states = residual + shared_output + routed_output return hidden_states ```

## [section] B. Hyper-parameters
