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
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token  # Set pad_token if necessary
template = """
{% for message in messages %}
<|{{ message['role'] }}|> {{ message['content'] }} </s>
{% endfor %}
"""
tokenizer.chat_template = template

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b-it",
    device_map="auto"
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
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=16,
    eval_accumulation_steps=16,
    num_train_epochs=1,
    fp16=True,
    group_by_length=True,
    optim="paged_adamw_32bit",
    max_steps=100
)

# Initialize SFTTrainer with correct arguments
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset_train,
    eval_dataset=dataset_val,
    peft_config=peft_config,
    max_seq_length=500,  # Set max sequence length for tokenization
    args=training_arguments
)

# Train the model
trainer.train()

# Save the trained model
trainer.model.save_pretrained("./")