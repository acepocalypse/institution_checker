# config.py
import os

INSTITUTION = "Purdue University"   # can be changed by user

# Campus filtering policy
# Set to True to accept IUPUI/IUPFW as Purdue connections
# Set to False to reject them (they are joint IU-Purdue campuses)
ACCEPT_JOINT_CAMPUSES = False  # Default: reject IUPUI/IUPFW

# Joint campus identifiers to filter
JOINT_CAMPUS_PATTERNS = [
    "iupui",  # Indiana University-Purdue University Indianapolis
    "iupfw",  # Indiana University-Purdue University Fort Wayne
    "indiana university purdue university",
    "indiana university-purdue university",
    "iu-purdue",
]

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
