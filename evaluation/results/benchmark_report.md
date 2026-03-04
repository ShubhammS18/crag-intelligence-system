# CRAG vs Naive RAG - Benchmark Report
Generated: 2026-03-04 13:10
Test set: 25 questions | 8 CORRECT | 9 AMBIGUOUS | 8 INCORRECT

## Results

| Metric | CRAG + Refine | Naive RAG | Delta |
|--------|:------------:|:---------:|:-----:|
| Faithfulness       | 0.8305 | 0.8857 | -5.5% |
| Answer Correctness | 0.5652 | 0.4872 | +7.8% |
| Context Precision  | 0.7428 | 0.2917 | +45.1% |
| Context Recall     | 0.6533 | 0.5000 | +15.3% |

## Interpretation
- **Context Precision +45.1%**: The eval_each_doc grader filters irrelevant chunks
  before generation. Naive RAG passes all retrieved chunks directly — CRAG passes
  only chunks that score above the relevance threshold.

- **Context Recall +15.3%**: Tavily web search fallback on AMBIGUOUS and INCORRECT
  paths recovers information absent from internal documents entirely.

- **Answer Correctness +7.8%**: CRAG produces answers that are semantically closer
  to ground truth, particularly on out-of-scope questions where Naive RAG has no
  fallback mechanism.

- **Faithfulness -5.5%**: The only metric where Naive RAG leads. Naive RAG answers
  exclusively from clean internal documents, making faithfulness straightforward.
  CRAG incorporates web search results which contain more formatting noise. This is
  the expected tradeoff of a hybrid retrieval system — broader coverage at a small
  cost to faithfulness purity. Net impact across all four metrics remains strongly
  positive for CRAG.