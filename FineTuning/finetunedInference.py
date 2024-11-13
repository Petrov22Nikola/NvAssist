import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load the model name
model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the base model
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# Load the LoRA adapter (which you saved after training)
model = PeftModel.from_pretrained(base_model, "./")  # Path where adapter_config.json and adapter_model.safetensors are saved

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define your input prompt (for example, a question or instruction)
prompt = "{\"VUID-vkCmdDecompressMemoryNV-None-07684\", \"The memoryDecompression feature must be enabled\", \"1.3-extensions\"}"

# Prepare messages for chat template
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]

# Apply chat template and tokenize input
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# Tokenize and move inputs to device
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# Generate response from the model
with torch.no_grad():
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=300,  # Set max length for generated text
        num_return_sequences=1,
        do_sample=True,
        temperature=0.7  # Adjust temperature for diversity in responses
    )

# Remove input tokens from generated output and decode response
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("Generated response:", generated_text)