import os

from dotenv import load_dotenv
from icecream import ic
from litellm import completion

""" 
Requires local Ollama installation (https://ollama.com/)
Selected local model (for example gemma2:2b) must be downloaded/ran prior to code execution using "ollama run gemma2:2b"
Default Ollama port is 11434, but if it changes, it can be adjusted via the .env file
"""

# Load environment variables from .env file
load_dotenv()


OLLAMA_API_BASE = f"http://localhost:{os.getenv('OLLAMA_PORT')}"

response = completion(
    model="ollama/gemma2:2b",
    messages=[{"content": "Hi! Tell me something about yourself in 2-3 sentences.", "role": "user"}],
    api_base=OLLAMA_API_BASE,
    stream=False,
)
ic(response['choices'][0]['message']['content'])
