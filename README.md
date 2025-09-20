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

## Training of Candidate Models

Here are the detailed training procedures and implementation specifics for each candidate model evaluated in our study.

**CNN-RNN** was replicated based on the architecture described by Karlekar and Bansal in order to serve as a key baseline. The model initially processes input text via a 300-dimensional embedding layer. Subsequently, a Convolutional Neural Network with 100 filters for each kernel size of [3, 4, and 5] is applied to extract local features. These features are then passed to a Long Short-Term Memory (LSTM) network with 300 hidden units to capture sequential dependencies, with the final hidden state being used to produce the classification logits.

**BERT-FT** utilizes the implementation from the Hugging Face Transformers library `BertForSequenceClassification`, which appends a classification head to BERT's final hidden layer to produce logits for each class. During fine-tuning, all model parameters, including the classification head, are optimized jointly. For binary classification, the model outputs two logits passed through a sigmoid activation. For multi-label classification (commenting, ogling, groping), three logits are generated and passed through independent sigmoid functions. Utterances are input directly into the encoder with no additional prompt or instructional text, and gold-standard labels are used for supervision. A maximum sequence length of 512 tokens is used, matching BERT's supported input size.

**GPT2-FT** leverages the implementation `GPT2ForSequenceClassification`, which attaches a fully connected classification head to GPT-2's final hidden representation. All parameters are fine-tuned end-to-end. For binary classification, two logits are produced and passed through a softmax function. For multi-class prediction, three logits are generated and passed through sigmoid activations for each class. Unlike BERT, GPT-2 is guided by short task-specific prompts to steer generation during both training and inference. A maximum sequence length of 512 tokens is used.

**Llama-3.1-FT** utilizes the Llama-3.1-8B-Instruct `LlamaForSequenceClassification` model from the Hugging Face Transformers library to perform binary classification with the `Llama-3.1-8B-Instruct` model. To facilitate binary classification, the model is provided with a task-specific instructional prompt that describes the classification objective clearly (e.g., determining whether a statement reflects commenting, ogling, or groping), enabling the model to align its outputs with the expected label format (e.g., `True`/`False` combinations). 

To efficiently manage this large model, we apply 4-bit quantization using the BitsAndBytes library. Additionally, we enhance Llama-3.1 with Low-Rank Adaptation (LoRA) through the PEFT framework. A LoRA configuration is set with a rank of 8, an alpha scaling factor of 16, and a dropout rate of 0.1. To conserve memory during training, gradient checkpointing is enabled and the model is prepared for k-bit training before integrating the LoRA adapters.

The fine-tuning process is performed end-to-end on our binary classification task. Input text is tokenized to determine the maximum sequence length and then tokenized with padding accordingly. The tokenized data is converted into PyTorch tensors and structured into a dataset compatible with the Hugging Face Trainer. We optimize the model using cross-entropy loss.

**DeepSeek-7B-FT** We adopt the instruction-tuned `deepseek-llm-7b-chat` model for binary classification using `AutoModelForSequenceClassification`. The model is fine-tuned with LoRA and 4-bit quantization similar to Llama-3.1. Instructional prompts are used during training and inference to specify the classification task, enabling the model to produce structured outputs aligned with the required label formats. Tokenization uses `AutoTokenizer`, and training is performed with the Hugging Face Trainer framework using AdamW optimizer and linear learning rate scheduler.

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
- **Statistical analysis**: 
To analyze the nature of the disagreements, we used two statistical tests. First, to determine if the raw label mismatch rate was statistically significant, we used a one-sided Z-test for proportions. This test evaluates the null hypothesis (H₀) that the true mismatch rate (p) is less than or equal to a predefined acceptable error threshold (p₀).

Second, to assess if there was a systematic pattern in the errors, we used McNemar's exact test for directional bias. This test compares the number of false negatives (cases where the original label was 0 but the expert label was 1) against the number of false positives (cases where the original label was 1 but the expert label was 0). It evaluates the null hypothesis that there is no directional bias, meaning false positives and false negatives are equally likely. The findings from this validation study are presented and discussed in our Results section.

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

## Interpretability Analysis

| **Model** | **Observed Word Clusters** |
|-----------|---------------------------|
| **AD-ASH** |  |
| **Commenting** | shameful, disrespectful, misbehaved, vulgar, inappropriate |
| **Groping** | groping, inappropriate touch, assault, harassment, molestation |
| **Ogling** | gestures, visually, disturbing, voyeur, leering |
| **BERT** |  |
| **Ogling** | encounter, surrounded, talk, embarrassed, leering |

In this section we provide a range of visualization techniques to analyze our best performing model. Each visualization method takes a unique approach, providing fresh insights or reinforcing existing conclusions. These visualizations enhance our understanding of the model, helping to uncover patterns, identify potential issues, and validate assumptions.

### Word Clusters

We selected seed words corresponding to class labels and identified the nearest neighbors of each seed word's vector by reducing the dimensionality of the word embeddings using t-SNE. This visualization not only confirms that our model has effectively learned meaningful word embeddings but also reveals that each type of sexual harassment is associated with a distinct context. Additionally, it demonstrates that our model, AD-ASH, captures related words and concepts specific to each harassment category. We observe that BERT underperformed for the "ogling" category, while the CNN-RNN model used in our adaptive approach achieved better results. This is reflected in the words extracted from our adaptive model, which more accurately represent this specific harassment categories compared to those from the BERT model.

### Saliency Heat Map

<img width="937" height="681" alt="saliency_map_bw_clean6" src="https://github.com/user-attachments/assets/1c7ae457-aa0a-48cd-9d7a-52c7b2e2a8db" />

*Figure 1: Saliency heat-map for a correctly classified example of "commenting".*


Saliency heatmaps highlight which words in an input have the greatest impact on the final classification.

In Fig. 1, the word "laughing" has the most significant influence on the classification, followed by "girls" and "noises". These words lead the model to predict the label "commenting", which matches the true label. This corresponds to a scenario where a group of boys makes remarks and strange noises toward girls—behavior that falls under the "commenting" category of sexual harassment.

<img width="958" height="682" alt="saliency_map_bw_clean2" src="https://github.com/user-attachments/assets/2ea4ce69-2a09-48f1-b361-320f7001ab16" />

*Figure 2: Saliency heat-map for a correctly classified example of "non-commenting".*


To understand why the model classifies certain incidents as non-commenting, consider Fig. 2. Here, the word "touched", followed by "bus", has the greatest influence, resulting in the model predicting the label "non-commenting", which again aligns with the true label. The model appears to associate "touching" with physical acts such as "groping", which are categorized under a different type of sexual harassment.

### LIME Analysis

LIME (Local Interpretable Model-Agnostic Explanations) is a technique that helps interpret a model's decision-making process by explaining predictions for specific instances. In the context of our binary classification models, LIME identifies the key features that influence the model's prediction for individual inputs. It does this by approximating the model's decision boundary with a simpler, interpretable model in the local vicinity of the instance, striking a balance between fidelity and interpretability.

This approach provides valuable insights into the features most relevant to a given classification, enhancing our understanding of how the model interprets specific examples. For instance, in the sentence "The guy at first was staring at me and later started passing cheap comments," LIME analysis identified the word "comments" as the most important feature, followed by "passing" and "cheap", indicating the model's recognition of the "commenting" category of sexual harassment. In another example, the phrase "touching/groping, commenting, ogling, and sexual invites" (labeled as "ogling") highlighted the word "ogling" as the most influential feature, demonstrating the model's ability to detect key terms associated with this harassment type. Similarly, in the sentence "A man standing too close to me in a semi-crowded metro station continued to touch me indecently till I pushed him away," LIME identified "touch", "pushed", "standing", and "close" as the most significant terms, aligning with the "groping" classification.

Overall, LIME analysis offers meaningful insights into the linguistic cues driving the model's predictions, contributing to a clearer understanding of how the classifier distinguishes between types of sexual harassment such as "commenting", "groping", and "ogling".
