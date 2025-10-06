# config.py

INSTITUTION = "Purdue University"   # can be changed by user

LLM_API_URL = "https://genai.rcac.purdue.edu/api/chat/completions"
LLM_API_KEY = "sk-3f1dbbf2450e46ab9541dffba4f18ec6"
MODEL_NAME = "gpt-oss:latest"

# Puppeteer options
BROWSER_ARGS = [
    "--no-sandbox",
    "--disable-setuid-sandbox",
    "--disable-blink-features=AutomationControlled"
]
