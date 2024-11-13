import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install required packages
install("transformers")
install("torch")
install("fastapi")
install("uvicorn")

# Start FastAPI server in a separate process
subprocess.Popen([sys.executable, "llmAPI.py"])