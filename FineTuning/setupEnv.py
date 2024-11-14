import subprocess
import sys
import os

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install required packages
install("transformers")
install("torch")
install("fastapi")
install("uvicorn")

# Start FastAPI server in a separate process
curDir = os.path.dirname(os.path.abspath(__file__))
llmAPIPath = os.path.join(curDir, "llmAPI.py")
process = subprocess.Popen([sys.executable, llmAPIPath], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
stdout, stderr = process.communicate()
print(f"stdout: {stdout}")
print(f"stderr: {stderr}")