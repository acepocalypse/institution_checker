# config.py
import os
from typing import Iterable

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

MODEL_NAME = "gpt-oss:120b"

# Model type detection
THINKING_MODEL_KEYWORDS = [
    "gpt-oss",
    "o1",
    "o3",
    "deepseek",
    "reasoning",
    "think",
]

def is_thinking_model(model_name: str = None) -> bool:
    """Detect if the model is a reasoning/thinking model based on name."""
    if model_name is None:
        model_name = MODEL_NAME
    
    model_lower = str(model_name).lower()
    return any(keyword in model_lower for keyword in THINKING_MODEL_KEYWORDS)


def get_model_config(model_name: str = None) -> dict:
    """Get configuration for the model type (thinking vs non-thinking)."""
    if model_name is None:
        model_name = MODEL_NAME
    
    is_thinking = is_thinking_model(model_name)
    
    return {
        "is_thinking": is_thinking,
        "model_name": model_name,
        "has_reasoning_tokens": is_thinking,
        "parse_reasoning": is_thinking,
    }

# Puppeteer options
BROWSER_ARGS = [
    "--no-sandbox",
    "--disable-setuid-sandbox",
    "--disable-blink-features=AutomationControlled"
]

# Shared temporal analysis terms (used across search and llm_processor modules)
CURRENT_TERMS = [
    "currently",
    "current",
    "now",
    "presently",
    "serving as",
    "now at",
    "active",
    "as of",
    "is a",
    "is an",
    "works at",
    "working at",
    "employed",
]

PAST_TERMS = [
    "former",
    "previous",
    "previously",
    "retired",
    "emeritus",
    "alumni",
    "alumnus",
    "alumna",
    "alum",
    "alumnae",
    "graduate",
    "graduated",
    "school of",
    "college of",
    "was at",
    "was a",
    "worked at",
    "served as",
    "left",
    "departed",
    "ex-",
    "until",
    "passed away",
    "died",
    "late",
    "deceased",
]


def _contains_any(text: str, phrases: Iterable[str]) -> bool:
    """Check if any phrase appears in text.
    
    Args:
        text: Text to search in
        phrases: Phrases to look for
        
    Returns:
        True if any phrase is found in text
    """
    return any(phrase in text for phrase in phrases)

