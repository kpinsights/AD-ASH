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

This split inheritance ensures direct comparability with baseline results and maintains consistency with prior work on this dataset. The original dataset and additional information can be found at https://github.com/swkarlekar/safecity.
## Implementation Details

The framework evaluates three harassment categories from the SafeCity dataset:
- **Commenting**: Verbal harassment
- **Ogling**: Visual harassment  
- **Groping**: Physical harassment

\section{Training of Candidate Models}
\label{sec:training}

Here are the detailed training procedures and implementation specifics for each candidate model evaluated in our study.

\noindent\textbf{CNN-RNN} was replicated based on the architecture described by Karlekar and Bansal \cite{karlekar-bansal-2018-safecity} in order to serve as a key baseline. The model initially processes input text via a 300-dimensional embedding layer. Subsequently, a Convolutional Neural Network with 100 filters for each kernel size of [3, 4, and 5] is applied to extract local features. These features are then passed to a Long Short-Term Memory (LSTM) network with 300 hidden units to capture sequential dependencies, with the final hidden state being used to produce the classification logits.

\noindent\textbf{BERT-FT} utilizes the implementation from the Hugging Face Transformers library ``BertForSequenceClassification``, which appends a classification head to BERT's final hidden layer to produce logits for each class. During fine-tuning, all model parameters, including the classification head, are optimized jointly. For binary classification, the model outputs two logits passed through a sigmoid activation. For multi-label classification (commenting, ogling, groping), three logits are generated and passed through independent sigmoid functions. Utterances are input directly into the encoder with no additional prompt or instructional text, and gold-standard labels are used for supervision. A maximum sequence length of 512 tokens is used, matching BERT's supported input size.

\noindent\textbf{GPT2-FT} leverages the implementation ``GPT2ForSequenceClassification``, which attaches a fully connected classification head to GPT-2’s final hidden representation. All parameters are fine-tuned end-to-end. For binary classification, two logits are produced and passed through a softmax function. For multi-class prediction, three logits are generated and passed through sigmoid activations for each class. Unlike BERT, GPT-2 is guided by short task-specific prompts to steer generation during both training and inference. A maximum sequence length of 512 tokens is used.


\noindent\textbf{Llama-3.1-FT}  utilizes the Llama-3.1-8B-Instruct ``LlamaForSequenceClassification`` model from the Hugging Face Transformers library to perform binary classification with the ``Llama-3.1-8B-Instruct`` model. To facilitate binary classification, the model is provided with a task-specific instructional prompt that describes the classification objective clearly (e.g., determining whether a statement reflects commenting, ogling, or groping), enabling the model to align its outputs with the expected label format (e.g., \texttt{True}/\texttt{False} combinations). 

To efficiently manage this large model, we apply 4-bit quantization using the BitsAndBytes library. Additionally, we enhance Llama-3.1 with Low-Rank Adaptation (LoRA) through the PEFT framework. A LoRA configuration is set with a rank of 8, an alpha scaling factor of 16, and a dropout rate of 0.1. To conserve memory during training, gradient checkpointing is enabled and the model is prepared for k-bit training before integrating the LoRA adapters.
The fine-tuning process is performed end-to-end on our binary classification task. Input text is tokenized to determine the maximum sequence length and then tokenized with padding accordingly. The tokenized data is converted into PyTorch tensors and structured into a dataset compatible with the Hugging Face Trainer. We optimize the model using cross-entropy loss.

\noindent\textbf{DeepSeek-7B-FT} We adopt the instruction-tuned ``deepseek-llm-7b-chat`` model for binary classification using ``AutoModelForSequenceClassification``. The model is fine-tuned with LoRA and 4-bit quantization similar to Llama-3.1. Instructional prompts are used during training and inference to specify the classification task, enabling the model to produce structured outputs aligned with the required label formats. Tokenization uses \texttt{AutoTokenizer}, and training is performed with the Hugging Face Trainer framework using AdamW optimizer and linear learning rate scheduler.


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
