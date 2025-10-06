# config.py
import os

INSTITUTION = "Purdue University"   # can be changed by user

LLM_API_URL = "https://genai.rcac.purdue.edu/api/chat/completions"
_api_key = None

def set_api_key(key: str):
    """Set the LLM API key programmatically."""
    global _api_key
    _api_key = key

def get_api_key():
    """Get the LLM API key, checking multiple sources."""
    if _api_key is not None:
        return _api_key
    env_key = os.getenv("LLM_API_KEY")
    if env_key is not None:
        return env_key
    raise ValueError("LLM API key not set. Use set_api_key() or set LLM_API_KEY environment variable.")

MODEL_NAME = "gpt-oss:latest"

# Puppeteer options
BROWSER_ARGS = [
    "--no-sandbox",
    "--disable-setuid-sandbox",
    "--disable-blink-features=AutomationControlled"
]
