---
type: paper
slug: 3-speculative-decoding-for-heterogeneous-vocabularies-with-string-level-verification
title: Accelerating LLM Inference with Lossless Speculative Decoding Algorithms for Heterogeneous Vocabularies
authors: Nadav Timor, Jonathan Mamou, Daniel Korat, Moshe Berchansky, Gaurav Jain, Oren Pereg, Moshe Wasserblat, David Harel
source_path: /Users/saris.kia.adm/.paper-scholar/3-speculative-decoding-for-heterogeneous-vocabularies-with-string-level-verification/2502.05202.md
ingested_at: '2026-06-05 05:47:57'
authors_list: []
sections:
- id: 422
  heading: Accelerating LLM Inference with Lossless Speculative Decoding Algorithms for Heterogeneous Vocabularies
  role: section
  order_index: 0
  summary: Nadav Timor <sup>1</sup> Jonathan Mamou <sup>2</sup> Daniel Korat <sup>2</sup> Moshe Berchansky <sup>2</sup> Gaurav Jain <sup>3</sup> Oren Pereg <sup>2</sup> Moshe Wasserblat <sup>2</sup> David Harel <sup>1</sup>
- id: 423
  heading: Abstract
  role: section
  order_index: 1
  summary: Accelerating the inference of large language models (LLMs) is a critical challenge in generative AI. Speculative decoding (SD) methods offer substantial efficiency gains by generating multiple tokens using a single target forward pass.
- id: 424
  heading: 1 Introduction
  role: section
  order_index: 2
  summary: Speculative decoding (SD; [Leviathan et al.,](#page-10-0) [2023;](#page-10-0) [Chen](#page-9-0) [et al.,](#page-9-0) [2023\)](#page-9-0) is an effective method for reducing the latency of LLM inference and increasing its throughput. A necessary condition for SD to be effective is that the drafter is sufficiently fast and accurate in approximating the target distribution [\(Timor et al.,](#page-12-0) [2025;](#page-12-0) [Chen et al.,](#page-9-1) [2024\)](#page-9-1).
- id: 425
  heading: 2 Motivating Examples
  role: section
  order_index: 3
  summary: Existing SD methods are designed to work with a single vocabulary, where the drafter samples from the same vocabulary as the target model. As an example, see Algorithm 5, which is the standard SD algorithm proposed by Leviathan et al.
- id: 426
  heading: 3 Speculative Decoding for Heterogeneous Vocabularies with String-Level Verification
  role: section
  order_index: 4
  summary: '**Notation.** Vocabularies are finite sets of strings, also called tokens. We say that a string a is *expressible* in a vocabulary B if there exist strings $b_1, b_2, \ldots, b_n \in B$ such that $a = b_1 \oplus b_2 \oplus \ldots \oplus b_n$ , where $\oplus$ denotes string concatenation.'
- id: 427
  heading: 3.3 Verification via Rejection Sampling
  role: section
  order_index: 7
  summary: The standard verification method of SD guarantees that the output tokens are distributed according to the target distribution, but it does not guarantee that the output tokens are exactly the target tokens, as in exact matching. For example, if the drafter is another instance of the target model p, the standard verification method of SD will accept all the draft tokens because, in general, the expected acceptance rate satisfies P <sup>t</sup>∈<sup>T</sup> min {p(t), q(t)} for any drafter q and vocabulary T, according to [Leviathan et al.](#page-10-0) [\(2023\)](#page-10-0).
- id: 428
  heading: 4 Speculative Decoding for Heterogeneous Vocabularies with Token-Level Verification
  role: section
  order_index: 9
  summary: This section introduces additional algorithms that extend the standard SD framework to operate over heterogeneous vocabularies, namely, where the drafter's vocabulary differs from the target's. Unlike Section 3, the algorithms in this section do not use strings as an intermediate, shared representation.
- id: 429
  heading: Lossless Speculative Decoding Algorithms for Heterogeneous Vocabularies
  role: section
  order_index: 11
  summary: Tables [8,](#page-19-0) [9,](#page-20-1) and [10](#page-21-0) in Appendix [E](#page-19-2) examine the vocabularies of widely used off-the-shelf target and drafter models. Table [8](#page-19-0) shows the vocabulary size of each model.
- id: 430
  heading: 6 Discussion
  role: section
  order_index: 12
  summary: To speed up the inference of a given target model, we need to select a drafter and a decoding algorithm. Table [3](#page-8-0) summarizes the expected probability of accepting the next token for all the speculation algorithms when the drafter has a different vocabulary than the target.
- id: 431
  heading: Acknowledgments
  role: section
  order_index: 13
  summary: We are grateful to Roy Schwartz from The Hebrew University of Jerusalem for his valuable feedback in improving this work. We thank Joao Gante and the Hugging Face team ˜ for reviewing the code and providing valuable feedback that contributed to its implementation in the Transformers library.
- id: 432
  heading: Impact Statement
  role: section
  order_index: 14
  summary: This work lowers the cost and latency of LLM inference making the serving of these models cheaper, faster, and more accessible to a wider range of users.
- id: 433
  heading: References
  role: section
  order_index: 15
  summary: Kauffmann, P., Lee, J. R., Lee, Y.
- id: 434
  heading: Standard Speculative Decoding
  role: section
  order_index: 17
  summary: Generating the next token via autoregressive decoding requires computing a target forward pass. Standard SD methods, like Algorithm 5, tend to utilize this target forward pass to verify multiple candidate tokens at once via a data parallelism technique known as batching, which is supported by modern hardware such as GPUs and TPUs.
- id: 435
  heading: Lossless Speculative Decoding Algorithms for Heterogeneous Vocabularies
  role: section
  order_index: 21
  summary: 'Table 7: Full benchmark for TLI (Algorithm [4\)](#page-6-0).'
- id: 436
  heading: F Injectivity of Tokenizers Under the CMM-DM Dataset
  role: section
  order_index: 23
  summary: The experiment sampled uniformly at random examples from the CNN-DM dataset (Nallapati et al., 2016b), and took the prefix of 100 characters from each example. Using a SentencePiece tokenizer (Kudo & Richardson, 2018) or various other Hugging Face Transformers tokenizers (Wolf et al., 2020), we encoded the prefix into tokens, and then decoded the tokens back into text.
---

# Accelerating LLM Inference with Lossless Speculative Decoding Algorithms for Heterogeneous Vocabularies

## [section] Accelerating LLM Inference with Lossless Speculative Decoding Algorithms for Heterogeneous Vocabularies
Nadav Timor <sup>1</sup> Jonathan Mamou <sup>2</sup> Daniel Korat <sup>2</sup> Moshe Berchansky <sup>2</sup> Gaurav Jain <sup>3</sup> Oren Pereg <sup>2</sup> Moshe Wasserblat <sup>2</sup> David Harel <sup>1</sup>

## [section] Abstract
Accelerating the inference of large language models (LLMs) is a critical challenge in generative AI. Speculative decoding (SD) methods offer substantial efficiency gains by generating multiple tokens using a single target forward pass.

## [section] 1 Introduction
Speculative decoding (SD; [Leviathan et al.,](#page-10-0) [2023;](#page-10-0) [Chen](#page-9-0) [et al.,](#page-9-0) [2023\)](#page-9-0) is an effective method for reducing the latency of LLM inference and increasing its throughput. A necessary condition for SD to be effective is that the drafter is sufficiently fast and accurate in approximating the target distribution [\(Timor et al.,](#page-12-0) [2025;](#page-12-0) [Chen et al.,](#page-9-1) [2024\)](#page-9-1).

## [section] 2 Motivating Examples
Existing SD methods are designed to work with a single vocabulary, where the drafter samples from the same vocabulary as the target model. As an example, see Algorithm 5, which is the standard SD algorithm proposed by Leviathan et al.

## [section] 3 Speculative Decoding for Heterogeneous Vocabularies with String-Level Verification
**Notation.** Vocabularies are finite sets of strings, also called tokens. We say that a string a is *expressible* in a vocabulary B if there exist strings $b_1, b_2, \ldots, b_n \in B$ such that $a = b_1 \oplus b_2 \oplus \ldots \oplus b_n$ , where $\oplus$ denotes string concatenation.

## [section] 3.3 Verification via Rejection Sampling
The standard verification method of SD guarantees that the output tokens are distributed according to the target distribution, but it does not guarantee that the output tokens are exactly the target tokens, as in exact matching. For example, if the drafter is another instance of the target model p, the standard verification method of SD will accept all the draft tokens because, in general, the expected acceptance rate satisfies P <sup>t</sup>∈<sup>T</sup> min {p(t), q(t)} for any drafter q and vocabulary T, according to [Leviathan et al.](#page-10-0) [\(2023\)](#page-10-0).

## [section] 4 Speculative Decoding for Heterogeneous Vocabularies with Token-Level Verification
This section introduces additional algorithms that extend the standard SD framework to operate over heterogeneous vocabularies, namely, where the drafter's vocabulary differs from the target's. Unlike Section 3, the algorithms in this section do not use strings as an intermediate, shared representation.

## [section] Lossless Speculative Decoding Algorithms for Heterogeneous Vocabularies
Tables [8,](#page-19-0) [9,](#page-20-1) and [10](#page-21-0) in Appendix [E](#page-19-2) examine the vocabularies of widely used off-the-shelf target and drafter models. Table [8](#page-19-0) shows the vocabulary size of each model.

## [section] 6 Discussion
To speed up the inference of a given target model, we need to select a drafter and a decoding algorithm. Table [3](#page-8-0) summarizes the expected probability of accepting the next token for all the speculation algorithms when the drafter has a different vocabulary than the target.

## [section] Acknowledgments
We are grateful to Roy Schwartz from The Hebrew University of Jerusalem for his valuable feedback in improving this work. We thank Joao Gante and the Hugging Face team ˜ for reviewing the code and providing valuable feedback that contributed to its implementation in the Transformers library.

## [section] Impact Statement
This work lowers the cost and latency of LLM inference making the serving of these models cheaper, faster, and more accessible to a wider range of users.

## [section] References
Kauffmann, P., Lee, J. R., Lee, Y.

## [section] Standard Speculative Decoding
Generating the next token via autoregressive decoding requires computing a target forward pass. Standard SD methods, like Algorithm 5, tend to utilize this target forward pass to verify multiple candidate tokens at once via a data parallelism technique known as batching, which is supported by modern hardware such as GPUs and TPUs.

## [section] Lossless Speculative Decoding Algorithms for Heterogeneous Vocabularies
Table 7: Full benchmark for TLI (Algorithm [4\)](#page-6-0).

## [section] F Injectivity of Tokenizers Under the CMM-DM Dataset
The experiment sampled uniformly at random examples from the CNN-DM dataset (Nallapati et al., 2016b), and took the prefix of 100 characters from each example. Using a SentencePiece tokenizer (Kudo & Richardson, 2018) or various other Hugging Face Transformers tokenizers (Wolf et al., 2020), we encoded the prefix into tokens, and then decoded the tokens back into text.
