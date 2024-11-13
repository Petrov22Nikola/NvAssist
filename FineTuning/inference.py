import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")

# Ensure pad token is set (if not already)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load the base model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct", device_map="auto")

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define your input prompt (for example, a question or instruction)
input_text = "{\"VUID-VkPhysicalDeviceSurfaceInfo2KHR-sType-unique\", \"The sType value of each struct in the pNext chain must be unique\", \"1.3-extensions\"}"

# Tokenize the input text and generate attention mask
inputs = tokenizer(input_text, return_tensors="pt", padding=True, return_attention_mask=True).to(device)

# Generate response from the model
with torch.no_grad():
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,  # Pass attention mask here
        max_new_tokens=300,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.7  # Adjust temperature for diversity in responses
    )

# Decode the generated tokens back into text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Generated response:", generated_text)