---
type: paper
slug: latent-speech-text-transformer
title: Latent Speech-Text Transformer
authors: Yen-Ju Lu, Yashesh Gaur, Wei Zhou, Benjamin Muller, Jesus Villalba, Najim Dehak, Luke Zettlemoyer, Gargi Ghosh, Mike Lewis, Srinivasan Iyer, Duc Le
source_path: /Users/saris.kia.adm/.paper-scholar/latent-speech-text-transformer
ingested_at: '2026-06-04 11:33:30'
authors_list:
- Yen-Ju Lu
- Yashesh Gaur
- Wei Zhou
- Benjamin Muller
- Jesus Villalba
- Najim Dehak
- Luke Zettlemoyer
- Gargi Ghosh
- Mike Lewis
- Srinivasan Iyer
- Duc Le
sections:
- id: 304
  heading: Abstract
  role: Introduction
  order_index: 0
  summary: 'Introduces LST: aggregates speech tokens into latent patches as higher-level autoregressive units.'
- id: 305
  heading: 1 Introduction
  role: Introduction
  order_index: 1
  summary: Speech tokens 3 orders of magnitude longer than text for same content.
- id: 306
  heading: 2 Background
  role: Background
  order_index: 2
  summary: Reviews GSLM, speech tokenization, interleaved speech-text models.
- id: 307
  heading: 3 Latent Speech-Text Transformer
  role: Methods
  order_index: 3
  summary: Patch encoder compresses speech tokens. Three patching strategies.
- id: 308
  heading: 4 Experiments
  role: Evaluation
  order_index: 4
  summary: Compute-controlled and data-controlled scaling experiments.
- id: 309
  heading: 5 Results
  role: Results
  order_index: 5
  summary: LST improves performance under controlled budgets. Gains grow with scale.
- id: 310
  heading: 6 Related Work
  role: Discussion
  order_index: 6
  summary: Reviews LLMs using speech tokens, transferring textual knowledge, efficiency.
- id: 311
  heading: 7 Limitations
  role: Discussion
  order_index: 7
  summary: Half-duplex only, no instruction fine-tuning, alignment dependency.
- id: 312
  heading: 8 Conclusion
  role: Conclusion
  order_index: 8
  summary: LST aggregates speech tokens into latent units to improve efficiency.
---

# Latent Speech-Text Transformer

## [Introduction] Abstract
Introduces LST: aggregates speech tokens into latent patches as higher-level autoregressive units.

## [Introduction] 1 Introduction
Speech tokens 3 orders of magnitude longer than text for same content.

## [Background] 2 Background
Reviews GSLM, speech tokenization, interleaved speech-text models.

## [Methods] 3 Latent Speech-Text Transformer
Patch encoder compresses speech tokens. Three patching strategies.

## [Evaluation] 4 Experiments
Compute-controlled and data-controlled scaling experiments.

## [Results] 5 Results
LST improves performance under controlled budgets. Gains grow with scale.

## [Discussion] 6 Related Work
Reviews LLMs using speech tokens, transferring textual knowledge, efficiency.

## [Discussion] 7 Limitations
Half-duplex only, no instruction fine-tuning, alignment dependency.

## [Conclusion] 8 Conclusion
LST aggregates speech tokens into latent units to improve efficiency.
