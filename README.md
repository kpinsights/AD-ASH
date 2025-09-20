# Beyond Monolithic LLMs: Modular AI for Online Harassment Detection

## Background

This repository contains supplementary materials for the paper "Beyond Monolithic LLMs: Modular AI for Online Harassment Detection" which introduces AD-ASH, a modular framework for detecting online sexual harassment.

The research addresses a critical limitation in harassment detection: no single model architecture excels at all harassment types. Through systematic evaluation of various models (CNNs, RNNs, BERT, and LLMs) on the SafeCity dataset, we found that different models perform best for different harassment categories:

- **BERT-FT**: Best for "Commenting" (83.2% accuracy)
- **CNN-RNN**: Best for "Ogling" (84.1% accuracy)  
- **DeepSeek-FT**: Best for "Groping" (88.7% accuracy)

AD-ASH leverages these complementary strengths by assigning each harassment type to its best-performing specialist, achieving 66.0% exact-match ratio and 85.3% Hamming score.

## Dataset


The experiments use the SafeCity dataset containing 9,892 anonymized narratives of sexual harassment reports. Following Karlekar and Bansal (2018), we use their established data partition:
- Training: 7,201 examples
- Validation: 990 examples  
- Testing: 1,701 examples

This split inheritance ensures direct comparability with baseline results and maintains consistency with prior work on this dataset.
  
## Implementation Details

The framework evaluates three harassment categories from the SafeCity dataset:
- **Commenting**: Verbal harassment
- **Ogling**: Visual harassment  
- **Groping**: Physical harassment

### Model Training
- Fine-tuned models: BERT, GPT-2, Llama-3.1, DeepSeek
- Classic architectures: CNN-RNN combinations
- Training setup: Batch size 8, AdamW optimizer, learning rate 2e-5

### Evaluation Approaches
- **Fine-tuning**: Standard supervised learning
- **Zero-shot prompting**: Task-specific instructions without examples
- **Few-shot prompting**: 3 positive + 3 negative examples
- **RAG-enhanced**: Dynamic retrieval of similar examples using Sentence-BERT

## Instruction Prompts Used

### Fine-tuning Instruction
```Classify if the following statement falls under Groping related to sexual harassment. The output must be a single label: 'True' or 'False'.```

### Zero-shot Prompting

```
You are an expert in identifying inappropriate physical behaviors in public or real-life scenarios.

### Task:
Determine whether the statement below involves **Groping**. Answer with "Yes" or "No".

### Statement:
{statement_text}

### Response:
Groping:
```

### One-shot Prompting

```
You are an expert in identifying inappropriate physical behaviors in public or real-life scenarios.

### Example:
**Statement**: "Was walking along a crowded street, holding mum's hand, when an elderly man groped my butt. I turned to look at him and he looked away, and did it again after a while. I was 12 years old then."  
**Groping**: Yes

### Task:
Based on the example above, determine whether the statement below involves **Groping**. Answer with "Yes" or "No".

### Statement:
{statement_text}

### Response:
Groping:
```

### Few-shot Prompting
Similar to one-shot but includes 3 positive and 3 negative examples manually selected to provide representative context across different harassment scenarios.

### RAG-enhanced Prompting
Uses the same structure as few-shot prompting but dynamically retrieves the top-6 most semantically similar examples from the training set using Sentence-BERT embeddings and FAISS indexing.

## Label Quality Validation Study

### Background
Recognizing that real-world, crowdsourced datasets can contain inconsistencies, we conducted an independent validation study to assess the quality of the original SafeCity labels.

### Methodology
- **Sample size**: 200 randomly selected reports from the 7,201 training examples
- **Expert re-labeling**: Each instance was independently re-labeled by domain experts to create a gold-standard reference
- **Statistical analysis**: Compared original vs. expert labels using one-sided proportion Z-test and McNemar's exact test

### Key Findings
Our validation revealed notable label quality issues:

| Category | Label Mismatch Rate | Statistical Significance |
|----------|-------------------|-------------------------|
| Commenting | 22.0% | Above 18% threshold |
| Ogling | 21.5% | Above 18% threshold |
| Groping | 14.0% | Above 11% threshold |

### Critical Discovery
- **Systematic bias identified**: The "Groping" category showed statistically significant directional bias (p < 0.01)
- **Under-reporting pattern**: False negatives significantly more frequent than false positives for groping incidents
- **Research implications**: These findings affect absolute performance interpretation but preserve relative model comparisons

### Impact on Results
While label quality issues affect all models uniformly (maintaining validity of relative performance comparisons), they constrain interpretation of absolute performance metrics. This discovery represents an important contribution to the harassment detection research community, highlighting the need for improved annotation protocols in sensitive AI applications.
