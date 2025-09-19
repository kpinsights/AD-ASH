print('Loading Libraries...')
import pandas as pd
import torch
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training, PeftModel
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    AdamW,
    get_scheduler,
    BitsAndBytesConfig,
    logging
)
from datasets import Dataset
import bitsandbytes as bnb
from tqdm import tqdm  # For progress bars

# Check GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")

print('Libraries Imported ....')
# Enable logging for Transformers
logging.set_verbosity_info()

# Ensure environment is set to use GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU

# Define model ID and load the tokenizer and model
model_id = "/home/kpokhrel/projects/def-kaz/kpokhrel/DeepSeek/deepseek-llm-7b-chat"
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# Using 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    num_labels=2,
    device_map="auto",  
    quantization_config=bnb_config,
    use_cache=False
)

print(f"Model device: {next(model.parameters()).device}")  # Check if model is on GPU

model.config.pad_token_id = tokenizer.pad_token_id

# LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1
)
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

print("Model prepared with LoRA adapters.")

# Load datasets
print("Loading datasets...")
train_path = '/home/kpokhrel/projects/def-kaz/kpokhrel/DeepSeek/Datasets/Groping/train.csv'
test_path = '/home/kpokhrel/projects/def-kaz/kpokhrel/DeepSeek/Datasets/Groping/test.csv'
val_path = '/home/kpokhrel/projects/def-kaz/kpokhrel/DeepSeek/Datasets/Groping/dev.csv'
train_data = pd.read_csv(train_path)
val_data = pd.read_csv(val_path)
test_data = pd.read_csv(test_path)

train_data.head()

# For DeepSeek LLM, we need to incorporate the system message into the user prompt
system_instruction = "Classify if the following statement falls under Groping related to sexual harassment. The output must be a single label: 'True' or 'False'."

train = []
test = []
val = []

for _, row in train_data.iterrows():
    # Using the chat template for DeepSeek LLM
    user_message = {"role": "user", "content": f"{system_instruction}\n\n{row['Description']}"}
    messages = [user_message]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    train.append({"prompt": prompt, "label": row["Category"]})
formatted_train = pd.DataFrame(train)
formatted_train.to_csv("/home/kpokhrel/projects/def-kaz/kpokhrel/DeepSeek/Datasets/Groping/formatted_training_data_Groping_llm.csv", index=False)

for _, row in test_data.iterrows():
    user_message = {"role": "user", "content": f"{system_instruction}\n\n{row['Description']}"}
    messages = [user_message]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    test.append({"prompt": prompt, "label": row["Category"]})
formatted_test = pd.DataFrame(test)
formatted_test.to_csv("/home/kpokhrel/projects/def-kaz/kpokhrel/DeepSeek/Datasets/Groping/formatted_test_data_Groping_llm.csv", index=False)

for _, row in val_data.iterrows():
    user_message = {"role": "user", "content": f"{system_instruction}\n\n{row['Description']}"}
    messages = [user_message]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    val.append({"prompt": prompt, "label": row["Category"]})
formatted_val = pd.DataFrame(val)
formatted_val.to_csv("/home/kpokhrel/projects/def-kaz/kpokhrel/DeepSeek/Datasets/Groping/formatted_val_data_Groping_llm.csv", index=False)

train[:5]

# Load formatted data
train = pd.read_csv('/home/kpokhrel/projects/def-kaz/kpokhrel/DeepSeek/Datasets/Groping/formatted_training_data_Groping_llm.csv')
test = pd.read_csv('/home/kpokhrel/projects/def-kaz/kpokhrel/DeepSeek/Datasets/Groping/formatted_test_data_Groping_llm.csv')
val = pd.read_csv('/home/kpokhrel/projects/def-kaz/kpokhrel/DeepSeek/Datasets/Groping/formatted_val_data_Groping_llm.csv')

train.head()

# Tokenization without padding 
def tokenize_conversation(examples):
    return tokenizer(examples['prompt'], truncation=False)

# Find the max token length across the datasets
def find_max_token_length(dataframe):
    print("Finding max token length...")
    dataset = Dataset.from_pandas(dataframe)
    tokenized_dataset = dataset.map(
        tokenize_conversation,
        batched=True,
        desc="Tokenizing dataset"
    )
    input_ids_list = tokenized_dataset['input_ids']

    max_len = 0
    for input_ids in tqdm(input_ids_list, desc="Calculating max token length"):
        max_len = max(max_len, len(input_ids))

    print(f"Max token length found: {max_len}")
    return max_len

# Finding max token lengths for train, val, and test datasets
print("Processing Train Dataset...")
max_train_len = find_max_token_length(train)

print("Processing Validation Dataset...")
max_val_len = find_max_token_length(val)

print("Processing Test Dataset...")
max_test_len = find_max_token_length(test)

# Tokenization with padding to the max token length
def tokenize_conversation_maxlen(examples, max_len):
    return tokenizer(examples['prompt'], max_length=max_len, padding='max_length', truncation=True)

def apply_tokenization_with_padding(dataframe, max_len):
    dataset = Dataset.from_pandas(dataframe)
    tokenized_dataset = dataset.map(lambda x: tokenize_conversation_maxlen(x, max_len), batched=True)

    input_ids = torch.tensor(tokenized_dataset['input_ids'])
    attention_masks = torch.tensor(tokenized_dataset['attention_mask'])
    # Binary classification output (Category is either 0 or 1)
    labels = torch.tensor(tokenized_dataset['label'])

    return input_ids, attention_masks, labels

# Apply tokenization with padding to max length for train, test, and val datasets
print("Tokenizing Train Dataset...")
train_input_ids, train_attention_masks, train_labels = apply_tokenization_with_padding(train, max_train_len)

print("Tokenizing Test Dataset...")
test_input_ids, test_attention_masks, test_labels = apply_tokenization_with_padding(test, max_test_len)

print("Tokenizing Validation Dataset...")
val_input_ids, val_attention_masks, val_labels = apply_tokenization_with_padding(val, max_val_len)

# Create dataset dictionaries for trainer compatibility
print("Preparing dataset dictionaries for Trainer...")
train_dataset = [{'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}
                 for input_ids, attention_mask, labels in tqdm(zip(train_input_ids, train_attention_masks, train_labels),
                                                             desc="Processing Train Dataset", total=len(train_input_ids))]

val_dataset = [{'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}
               for input_ids, attention_mask, labels in tqdm(zip(val_input_ids, val_attention_masks, val_labels),
                                                             desc="Processing Validation Dataset", total=len(val_input_ids))]

test_dataset = [{'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}
                for input_ids, attention_mask, labels in tqdm(zip(test_input_ids, test_attention_masks, test_labels),
                                                             desc="Processing Test Dataset", total=len(test_input_ids))]

print("All datasets are ready for training.")

# Training arguments with GPU optimization
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    save_strategy="epoch",
    report_to="none",
    load_best_model_at_end=True,
    fp16=True, 
    dataloader_num_workers=4,  
    gradient_accumulation_steps=2,  
    remove_unused_columns=False,  
)

# Defining the optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)
# Defining the learning rate scheduler
num_training_steps = len(train_dataset) * 3 // training_args.per_device_train_batch_size
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

# Function to compute evaluation metrics (binary classification)
def compute_metrics(p):
    predictions, labels = p
    preds = torch.argmax(torch.tensor(predictions), dim=-1)  # Use argmax for classification
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    accuracy = accuracy_score(labels, preds)
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy
    }

# Initializing the trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    optimizers=(optimizer, lr_scheduler),
    compute_metrics=compute_metrics
)

# Training the model
print("Starting training...")
trainer.train()

# Evaluating the model
print("Evaluating model...")
results = trainer.evaluate(test_dataset)
print("Evaluation Results:", results)

# Generate predictions for the test set and save them to CSV
print("Generating test predictions and saving to CSV...")
predictions = trainer.predict(test_dataset)
predicted_labels = torch.argmax(torch.tensor(predictions.predictions), dim=-1).tolist()
true_labels = predictions.label_ids.tolist()

# Generate predictions for the test set and save them to CSV
print("Generating test predictions and saving to CSV...")
predictions = trainer.predict(test_dataset)
predicted_labels = torch.argmax(torch.tensor(predictions.predictions), dim=-1).tolist()
true_labels = predictions.label_ids.tolist()

# Get original statements from test data
original_statements = test["prompt"].tolist()
# Extract just the user's description (remove the chat template formatting)
# This extraction might need adjustment based on the actual chat template structure
descriptions = []
for statement in original_statements:
    # Extract just the description part from the formatted chat template
    # First, try to find the original description after the system instruction
    if system_instruction in statement:
        parts = statement.split(system_instruction)
        if len(parts) > 1:
            # Take the text after the system instruction
            descriptions.append(parts[1].strip())
        else:
            descriptions.append(statement)  # Fallback
    else:
        descriptions.append(statement)  # Fallback

# Create a DataFrame with original statements, true labels, and predicted labels
results_df = pd.DataFrame({
    "Description": descriptions,
    "True_Label": true_labels,
    "Predicted_Label": predicted_labels,
    # Add additional metrics
    "Correct_Prediction": [true == pred for true, pred in zip(true_labels, predicted_labels)]
})


# Save the results to CSV
results_csv_path = "/home/kpokhrel/projects/def-kaz/kpokhrel/DeepSeek/deepseek_llm_7b_Groping_predictions.csv"
results_df.to_csv(results_csv_path, index=False)
print(f"Predictions saved to {results_csv_path}")

# Save paths
lora_model_path = "/home/kpokhrel/projects/def-kaz/kpokhrel/DeepSeek/Finetuned_Models/Groping/fine-tuned-DeepSeek-LLM-7B_Groping_LoRA"
merged_model_path = "/home/kpokhrel/projects/def-kaz/kpokhrel/DeepSeek/Finetuned_Models/Groping/fine-tuned-DeepSeek-LLM-7B_Groping_Merged"

# Saving the LoRA model
print("Saving LoRA model...")
model.save_pretrained(lora_model_path)
tokenizer.save_pretrained(lora_model_path)

# Create merged model
print("Creating and saving merged model...")
try:
    # Load the base model
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=2,
        device_map="auto",
        torch_dtype=torch.float16  # Using float16 for the merged model to save memory
    )
    
    # Load the trained LoRA model
    lora_model = PeftModel.from_pretrained(base_model, lora_model_path)
    
    # Merge weights
    merged_model = lora_model.merge_and_unload()
    
    # Save the merged model
    merged_model.save_pretrained(merged_model_path)
    tokenizer.save_pretrained(merged_model_path)
    print(f"Merged model saved successfully to {merged_model_path}")
except Exception as e:
    print(f"Error when creating merged model: {e}")
    print("Proceeding without merged model.")

print("Training, evaluation, and model saving completed successfully.")
