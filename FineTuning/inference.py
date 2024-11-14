import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct", device_map="auto")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

input_text = "{\"VUID-VkPhysicalDeviceSurfaceInfo2KHR-sType-unique\", \"The sType value of each struct in the pNext chain must be unique\", \"1.3-extensions\"}"

inputs = tokenizer(input_text, return_tensors="pt", padding=True, return_attention_mask=True).to(device)

with torch.no_grad():
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=300,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.7
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Generated response:", generated_text)