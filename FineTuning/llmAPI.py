from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import os
import gc

class PromptRequest(BaseModel):
    prompt: str

app = FastAPI()
# GPU preferred
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

if device == "cuda":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64,expandable_segments:True"
    torch.cuda.empty_cache() 

gc.collect()
torch.cuda.empty_cache()

# Qwen-Coder-Instruct-7B
qwenModelName = "Qwen/Qwen2.5-Coder-7B-Instruct"
qwenAdapterPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen7b-instr-finetuned")
qwenTokenizer = AutoTokenizer.from_pretrained(qwenModelName, local_files_only=False)
qwenModel = AutoModelForCausalLM.from_pretrained(
    qwenModelName,
    torch_dtype=torch.float16,
    device_map="auto",
    offload_folder="offload",
    local_files_only=False,
    low_cpu_mem_usage=True
)
qwenModel = PeftModel.from_pretrained(qwenModel, qwenAdapterPath)
# qwenModel.to(device)

@app.post("/autocomplete")
async def autocomplete(request: PromptRequest):
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Generate clean, well-structured code that follows best practices. Adhere to the whitespace and indendentation of the prompt. Complete the code segment, continue from where the code leaves off."},
            {"role": "user", "content": request.prompt}
        ]

        text = qwenTokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        modelInputs = qwenTokenizer([text], return_tensors="pt").to(qwenModel.device)

        with torch.no_grad():
            genIds = qwenModel.generate(
                **modelInputs,
                max_new_tokens=300,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.5
            )

        genIds = [
            outIds[len(inIds):] for inIds, outIds in zip(modelInputs.input_ids, genIds)
        ]
        genText = qwenTokenizer.batch_decode(genIds, skip_special_tokens=True)[0]

        return genText

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate")
async def generateText(request: PromptRequest):
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Generate clean, well-structured code that follows best practices. Adhere to the whitespace and indendentation of the prompt."},
            {"role": "user", "content": request.prompt}
        ]

        text = qwenTokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        modelInputs = qwenTokenizer([text], return_tensors="pt").to(qwenModel.device)

        with torch.no_grad():
            genIds = qwenModel.generate(
                **modelInputs,
                max_new_tokens=300,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.5
            )

        genIds = [
            outIds[len(inIds):] for inIds, outIds in zip(modelInputs.input_ids, genIds)
        ]
        genText = qwenTokenizer.batch_decode(genIds, skip_special_tokens=True)[0]

        return genText

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

import uvicorn
uvicorn.run(app, host="127.0.0.1", port=8000)