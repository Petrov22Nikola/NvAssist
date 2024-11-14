import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, "./")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
prompt = "{\"VUID-vkCmdDecompressMemoryNV-None-07684\", \"The memoryDecompression feature must be enabled\", \"1.3-extensions\"}"

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

with torch.no_grad():
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=300,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.7
    )

generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("Generated response:", generated_text)