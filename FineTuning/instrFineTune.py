import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, PeftModelForCausalLM
from trl import SFTTrainer
from datasets import Dataset

# Datasets
save_path = "./"
dataset_train_name = 'train'
dataset_val_name = 'eval'
file_name_train_chatml = f"{dataset_train_name}_chatml.json"
file_name_val_chatml = f"{dataset_val_name}_chatml.json"

with open(save_path + file_name_train_chatml, 'r') as f:
    data_train = json.load(f)

with open(save_path + file_name_val_chatml, 'r') as f:
    data_val = json.load(f)

dataset_train = Dataset.from_dict(data_train)
dataset_val = Dataset.from_dict(data_val)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    device_map="auto",
    torch_dtype=torch.float16
)

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=16,
    task_type='CAUSAL_LM'
)

training_arguments = TrainingArguments(
    output_dir="./",
    eval_strategy="steps",
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
    fp16=True,
    group_by_length=True,
    optim="paged_adamw_32bit",
    max_steps=100
)

def tokenize_function(examples):
    return tokenizer(examples['content'], padding=True, truncation=True, return_tensors="pt", return_attention_mask=True)

tokenized_train_dataset = dataset_train.map(tokenize_function, batched=True)
tokenized_val_dataset = dataset_val.map(tokenize_function, batched=True)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    peft_config=peft_config,
    max_seq_length=500,
    args=training_arguments
)

trainer.train()
trainer.model.save_pretrained("./")