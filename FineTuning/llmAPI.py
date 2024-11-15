from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import os

class PromptRequest(BaseModel):
    prompt: str

app = FastAPI()
# GPU preferred
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

if (device == "cuda"):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
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
    local_files_only=False
)
qwenModel = PeftModel.from_pretrained(qwenModel, qwenAdapterPath)
qwenModel.to(device)

# CodeGemma-2B
gemmaModelName = "codegemma2b/base_model"
gemmaTokenizerPath = "codegemma2b/base_model_tokenizer"
gemmaTokenizer = AutoTokenizer.from_pretrained(gemmaTokenizerPath, local_files_only=True)
if gemmaTokenizer.pad_token is None:
    gemmaTokenizer.pad_token = gemmaTokenizer.eos_token
gemmaModel = AutoModelForCausalLM.from_pretrained(
    gemmaModelName,
    torch_dtype=torch.float16,
    device_map="auto",
    offload_folder="offload",
    local_files_only=True
)
gemmaModel.to(device)

from transformers import StoppingCriteria, StoppingCriteriaList

class CustomStopCriteria(StoppingCriteria):
    def __init__(self, stopIds):
        super().__init__()
        self.stopIds = stopIds

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
        for stopId in self.stopIds:
            stop_tensor = torch.tensor(stopId, device=input_ids.device)
            
            if torch.equal(input_ids[0][-len(stop_tensor):], stop_tensor):
                return True
        return False

@app.post("/autocomplete")
async def autocomplete(request: PromptRequest):
    try:
        model_inputs = gemmaTokenizer([request.prompt], return_tensors="pt").to(gemmaModel.device)
        stop_tokens = ["<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>", "<|file_separator|>"]
        stopIds = [gemmaTokenizer.encode(token, add_special_tokens=False) for token in stop_tokens]
        stopping_criteria = StoppingCriteriaList([CustomStopCriteria(stopIds)])

        with torch.no_grad():
            genIds = gemmaModel.generate(
                **model_inputs,
                max_new_tokens=100,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.5,
                stopping_criteria=stopping_criteria
            )

        # Filter out input prompt tokens
        genIds = genIds[:, len(model_inputs['input_ids'][0]):]
        genText = gemmaTokenizer.decode(genIds[0], skip_special_tokens=True)
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