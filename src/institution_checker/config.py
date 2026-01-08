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

# Dataset profile configuration for optimized processing
# Controls filtering aggressiveness and search strategy based on expected connection rate
DATASET_PROFILE = "high_connection"  # Options: "high_connection" or "low_connection"

# VIP configuration (for near-100% recall on high-value names)
# Add exact names (case-insensitive) you want to treat as VIPs.
VIP_NAMES = []

# Keywords that flag VIP candidates in search results.
VIP_SIGNAL_KEYWORDS = []

# When enabled, only VIPs run enhanced search; non-VIPs stay on the fast path.
VIP_ONLY_ENHANCED = False

# VIP rescue: extra targeted queries if a VIP is still not connected.
VIP_RESCUE_ENABLED = False
VIP_RESCUE_RESULTS_PER_QUERY = 8
VIP_RESCUE_QUERIES = [
    '"{name}" Purdue',
    '"{name}" "Purdue Univ"',
    'site:purdue.edu "{name}"',
    '"{name}" Purdue (alumni OR visiting OR professor OR PhD)',
]

# Page excerpt fetching limits (used before LLM analysis).
LLM_EXCERPT_LIMIT = 6
VIP_EXCERPT_LIMIT = 16

# DuckDuckGo acceptance thresholds (lower = faster, higher = safer).
DDG_MIN_RESULTS_DEFAULT = 3
DDG_MIN_RESULTS_LOW_CONNECTION = 2
DDG_MIN_RESULTS_VIP = 5

# Profile characteristics:
# - high_connection: 30%+ expected connections (e.g., alumni lists, department searches)
#   → Conservative filtering, run all search strategies, maximize recall
# - low_connection: <10% expected connections (e.g., Nobel Prize winners, random names)
#   → Aggressive filtering, early-exit searches, minimize wasted LLM calls

def get_filtering_mode(profile: str = None) -> str:
    """Get filtering mode based on dataset profile.
    
    Args:
        profile: Dataset profile ("high_connection" or "low_connection")
                If None, uses DATASET_PROFILE config
    
    Returns:
        "conservative" or "aggressive"
    """
    if profile is None:
        profile = DATASET_PROFILE
    
    profile_lower = str(profile).lower()
    if "low" in profile_lower:
        return "aggressive"
    return "conservative"

def get_early_exit_threshold(profile: str = None) -> int:
    """Get relevance score threshold for early-exit search strategies.
    
    Args:
        profile: Dataset profile
    
    Returns:
        Minimum relevance score to continue searching
    """
    if profile is None:
        profile = DATASET_PROFILE
    
    profile_lower = str(profile).lower()
    if "low" in profile_lower:
        return 12  # High threshold for low-connection datasets
    return 5  # Low threshold for high-connection datasets (always search thoroughly)

def get_skip_threshold(profile: str = None) -> int:
    """Get relevance score threshold for skipping LLM calls.
    
    Args:
        profile: Dataset profile
    
    Returns:
        Maximum relevance score to skip LLM
    """
    if profile is None:
        profile = DATASET_PROFILE
    
    profile_lower = str(profile).lower()
    if "low" in profile_lower:
        return 8  # Skip results with score < 8 in low-connection datasets
    return 0  # Only skip garbage (score 0) in high-connection datasets

# Multi-tier aggressive filtering thresholds (for low-connection datasets)
# These constants define the 3-tier skip logic in should_skip_llm()
# Designed to increase skip rate from ~80% to 95%+ while preserving accuracy
AGGRESSIVE_TIER1_THRESHOLD = 8   # Tier 1: Skip if score < 8 with no strong signals
AGGRESSIVE_TIER2_THRESHOLD = 12  # Tier 2: Skip if score 8-12 without excellent evidence
AGGRESSIVE_TIER3_THRESHOLD = 15  # Tier 3: Skip if score < 15 with no .edu domain
AGGRESSIVE_SIGNAL_THRESHOLD = 30  # High signal score indicating multiple strong keywords
AGGRESSIVE_DOMAIN_COUNT_MIN = 2  # Minimum .edu domains for "excellent evidence"

# Tier 4: Confident negative filtering thresholds (ultra-safe patterns)
CONFIDENT_NEGATIVE_SIGNAL_THRESHOLD = -30  # Require strong negative evidence (multiple negative keywords)
LOW_QUALITY_AUTHORITY_THRESHOLD = -5        # Low-quality domains (prabook, alchetron)
NO_MENTION_SCORE_MAX = 3                    # Virtually no relevance (citation lists)

# Evidence quality requirements for Tier 2 (marginal scores)
# At least one of these must be true to avoid skipping:
# - purdue_domain_hits >= AGGRESSIVE_DOMAIN_COUNT_MIN (multiple authoritative sources)
# - purdue_domain_hits >= 1 AND has_explicit_connection (single strong source)
# - total_signal_score >= AGGRESSIVE_SIGNAL_THRESHOLD (many strong keywords)

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
    "--disable-blink-features=AutomationControlled",
    "--disable-extensions",
    "--disable-gpu",
    "--disable-dev-shm-usage",
    "--blink-settings=imagesEnabled=false",
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

