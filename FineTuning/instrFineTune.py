import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, PeftModelForCausalLM
from trl import SFTTrainer
from datasets import Dataset

# Paths and dataset names
save_path = "./"
dataset_train_name = 'train'
dataset_val_name = 'eval'
file_name_train_chatml = f"{dataset_train_name}_chatml.json"
file_name_val_chatml = f"{dataset_val_name}_chatml.json"

# Load training and validation datasets from JSON files
with open(save_path + file_name_train_chatml, 'r') as f:
    data_train = json.load(f)

with open(save_path + file_name_val_chatml, 'r') as f:
    data_val = json.load(f)

# Convert dictionaries to Hugging Face Dataset format
dataset_train = Dataset.from_dict(data_train)
dataset_val = Dataset.from_dict(data_val)

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token  # Set pad_token if necessary

# Load model with mixed precision (float16)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    device_map="auto",
    torch_dtype=torch.float16  # Ensure model is loaded in float16
)

# LoRA configuration for PEFT (Parameter-Efficient Fine-Tuning)
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=16,
    task_type='CAUSAL_LM'
)

# Training arguments for Hugging Face Trainer (use eval_strategy instead of evaluation_strategy)
training_arguments = TrainingArguments(
    output_dir="./",
    eval_strategy="steps",  # Replace deprecated evaluation_strategy with eval_strategy
    logging_strategy="steps",
    lr_scheduler_type="constant",
    logging_steps=20,
    eval_steps=20,
    save_steps=20,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=8,
    eval_accumulation_steps=8,
    num_train_epochs=1,
    fp16=True,  # Enable mixed precision training with float16
    group_by_length=True,
    optim="paged_adamw_32bit",
    max_steps=100
)

# Tokenization function that returns both input_ids and attention_mask
def tokenize_function(examples):
    return tokenizer(examples['content'], padding=True, truncation=True, return_tensors="pt", return_attention_mask=True)

# Apply tokenization to datasets
tokenized_train_dataset = dataset_train.map(tokenize_function, batched=True)
tokenized_val_dataset = dataset_val.map(tokenize_function, batched=True)

# Initialize SFTTrainer with correct arguments
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    peft_config=peft_config,
    max_seq_length=500,  # Set max sequence length for tokenization
    args=training_arguments
)

# Train the model
trainer.train()

# Save the trained model
trainer.model.save_pretrained("./")