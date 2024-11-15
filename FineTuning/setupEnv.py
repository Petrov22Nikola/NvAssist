import subprocess
import sys
import os

def EnsurePipInstalled():
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "--version"])
    except subprocess.CalledProcessError:
        subprocess.check_call([sys.executable, "-m", "ensurepip", "--upgrade"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

EnsurePipInstalled()

install("torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
install("transformers")
install("fastapi")
install("uvicorn")
install("peft")

# Start FastAPI server in a separate process
curDir = os.path.dirname(os.path.abspath(__file__))
llmAPIPath = os.path.join(curDir, "llmAPI.py")
process = subprocess.Popen([sys.executable, llmAPIPath], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
stdout, stderr = process.communicate()
print(f"stdout: {stdout}")
print(f"stderr: {stderr}")