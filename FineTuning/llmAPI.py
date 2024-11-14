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

# Qwen-Coder-Instruct-7B
qwenModelName = "Qwen/Qwen2.5-Coder-7B-Instruct"
qwenAdapterPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen7b-instr-finetuned")
qwenTokenizer = AutoTokenizer.from_pretrained(qwenModelName)
qwenModel = AutoModelForCausalLM.from_pretrained(
    qwenModelName,
    torch_dtype="auto",
    device_map="auto"
)
qwenModel = PeftModel.from_pretrained(qwenModel, qwenAdapterPath)
qwenModel.to(device)

# CodeGemma-2B
gemmaModelName = "google/codegemma-2b"
gemmaAdapterPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "codegemma2b")
gemmaTokenizer = AutoTokenizer.from_pretrained(gemmaModelName)
if gemmaTokenizer.pad_token is None:
    gemmaTokenizer.pad_token = gemmaTokenizer.eos_token
gemmaModel = AutoModelForCausalLM.from_pretrained(
    gemmaModelName,
    torch_dtype="auto",
    device_map="auto"
)
gemmaModel = PeftModel.from_pretrained(gemmaModel, gemmaAdapterPath)
gemmaModel.to(device)

from transformers import StoppingCriteria, StoppingCriteriaList

class CustomStopCriteria(StoppingCriteria):
    def __init__(self, stopIds):
        super().__init__()
        self.stopIds = stopIds

    def __call__(self, input_ids, scores, **kwargs):
        for stopId in self.stopIds:
            if stopId in input_ids[0][-len(stopId):]:
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
            gen_ids = gemmaModel.generate(
                **model_inputs,
                max_new_tokens=100,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.5,
                stopping_criteria=stopping_criteria
            )

        # Filter out input prompt tokens
        gen_ids = gen_ids[:, len(model_inputs['input_ids'][0]):]
        gen_text = gemmaTokenizer.decode(gen_ids[0], skip_special_tokens=True)
        return gen_text

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)