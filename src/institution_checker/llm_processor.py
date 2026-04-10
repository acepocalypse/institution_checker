from __future__ import annotations  # Increased to provide more context to LLMort annotations

import asyncio
import json
import re
import ssl
import sys
import time
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

import aiohttp
import unicodedata

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from .config import (
    ALLOW_HEURISTIC_POSITIVE_RESCUE,
    CURRENT_TERMS,
    LLM_API_URL,
    MODEL_NAME,
    PAST_TERMS,
    _contains_any,
    get_api_key,
    is_thinking_model,
)
from .search import _institution_tokens

_session: Optional[aiohttp.ClientSession] = None
_session_lock = asyncio.Lock()
_last_refresh_time = 0.0
_PROMPT_LOGGING_ENABLED = False

MAX_RESULTS_FOR_PROMPT = 20
VIP_RESULTS_FOR_PROMPT = 28

# STRICT EVIDENCE-BASED PROMPT - Balanced for speed and accuracy
PROMPT_TEMPLATE = """You are an evidence-driven verifier confirming whether {name} has a past or present relationship with {institution}. Read all {max_results} search findings below and decide if there is a real connection.

CONNECTED: Return verdict="connected" ONLY when the evidence explicitly and directly shows:
1. Alumni: Earned degree (Bachelor, Master, PhD, etc.) from the institution
2. Attended: Enrolled/studied as a student (including "studied under") without earning a degree
    - This includes short-form academic enrollment/training such as summer school, summer sessions, summer programs, or summer institutes at the institution
3. Faculty: Teaching position, professor (including Distinguished/Emeritus), instructor, lecturer
4. Staff: Administrative or research staff position (including Research Associate)
5. Executive: President, dean, director, provost, chancellor role
6. Postdoc: Postdoctoral fellowship position
7. Visiting: Visiting scholar, visiting professor, or visiting fellow role
8. Nobel Laureate: Affiliated with the institution (as faculty/student) at any time, or specifically for the prize-winning work

STRICT EXCLUSIONS - These do NOT count as connections:
- Publications where a symposium/conference was held AT the institution (e.g., "Proceedings of the SIGPLAN Symposium...Purdue University" in a citation)
- Honorary degrees or awards (unless accompanied by a real role like Professor or Alumnus)
- Invited talks, seminars, or guest lectures (e.g., "Gave the XYZ Lecture at Purdue")
- External advisory boards or consulting roles
- Temporary event participation that does not involve enrollment/training status (e.g., conference attendance without student/trainee role)
- Publications citing or mentioning the institution
- Joint or affiliate campus mentions (e.g., IUPUI, IUPFW) unless the connection is explicitly to the main campus

The person must have had a direct, institutional role or status (student, employee, degree-holder) at the institution. Brief mentions or publications held at a venue are NOT connections.

Important: "Visiting Professor", "Visiting Scholar", and "Adjunct" roles ARE valid connections.
Important: "attended summer school/session/program at {institution}" is a valid Attended connection when explicitly linked to {name}.
Critical identity rule: Evidence must refer to {name}. If evidence clearly refers to a different person, return not_connected.

Prefer authoritative sources (.edu, official bios/CVs, reputable news). Quote the EXACT evidence that directly supports a connection.

Use "current" when the language is present tense or references this year/{prior_year}; otherwise choose "past". Pick "unknown" when the timeframe cannot be inferred.

CORRECTION RULES (apply if evidence matches):
- If mentions "Nobel Prize" AND the institution name (implying affiliation/alumni status) → set connected=Y, type=Faculty or Alumni (High Confidence)
- If mentions "professor", "faculty", "PhD from", "Distinguished Professor" AND explicitly mentions {institution} (or an official institution domain/source) → set connected=Y
- If mentions "honorary", "lecture", "speaker", "parents" → set connected=N
- If mentions "Processing error" → mark verification_status=needs_review
- If only LinkedIn/blog source → mark verification_status=needs_review

Return JSON only (no markdown) using this schema:
{{
    "verdict": "connected|not_connected|uncertain",
    "relationship_type": "Alumni|Attended|Faculty|Staff|Executive|Postdoc|Visiting|Other|None",
    "relationship_timeframe": "current|past|unknown",
    "verification_detail": "Quote the EXACT evidence or state why the connection fails",
    "summary": "One concise sentence describing the relationship decision",
    "primary_source": "Best supporting URL (prefer .edu)",
    "confidence": "high|medium|low",
    "verification_status": "verified|needs_review",
    "temporal_context": "Dates/years mentioned or 'unknown'"
}}

Search findings:
{search_findings}"""

DEFAULT_CLIENT_TIMEOUT = aiohttp.ClientTimeout(
    total=240,      # Increased from 180 to 240s for total request time
    connect=30,     # Increased from 15 to 30s for connection establishment  
    sock_read=180   # Increased from 150 to 180s for reading response
)
JSON_BLOCK_RE = re.compile(r"```(?:json)?(.*?)```", re.DOTALL | re.IGNORECASE)
JSON_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")

# CURRENT_TERMS and PAST_TERMS are now imported from config.py
# Add year pattern for temporal detection
YEAR_PATTERN = re.compile(r'\b(19|20)\d{2}\b')
# Pattern for date ranges like "2010-2015" or "2010 - 2015"
DATE_RANGE_PATTERN = re.compile(r'\b(\d{4})\s*[\-–—]\s*(\d{4})\b')
# Pattern for "from X to Y"
FROM_TO_PATTERN = re.compile(r'\bfrom\s+(\d{4})\s+to\s+(\d{4})\b', re.IGNORECASE)
# Pattern for "since X" or "starting X"
SINCE_PATTERN = re.compile(r'\b(?:since|starting|began\s+in|started\s+in)\s+(\d{4})\b', re.IGNORECASE)
# Pattern for "until X" or "through X"
UNTIL_PATTERN = re.compile(r'\b(?:until|through|ended\s+in|retired\s+in|left\s+in)\s+(\d{4})\b', re.IGNORECASE)

_NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v", "vi"}
_EVIDENCE_WINDOW_CHARS = 220


class RetryableLLMError(RuntimeError):
    """Raised for transient LLM API responses that should be retried."""


_DEGREE_VERBS = [
    "graduated",
    "graduate",
    "earned",
    "received",
    "obtained",
    "completed",
    "holds",
    "was awarded",
]
_DEGREE_NOUNS = [
    "degree",
    "b.s.",
    "b.s",
    "bs",
    "bsc",
    "b.a.",
    "ba",
    "mba",
    "m.s.",
    "ms",
    "msc",
    "m.a.",
    "ma",
    "ph.d",
    "phd",
    "doctorate",
    "jd",
    "law degree",
    "engineering degree",
    "bachelors",
    "bachelor",
    "masters",
    "master",
]
_DEGREE_PHRASES = ["degree from", "degree at"]
_ALUMNI_TERMS = {
    "alumnus",
    "alumna",
    "alumni",
    "alum",
    "alumnae",
    "graduate",
    "graduated",
    "school of",
    "college of",
}
_ATTENDED_TERMS = {
    "attended",
    "studied at",
    "studied under",
    "studied in",
    "student at",
    "enrolled at",
    "enrolled in",
    "matriculated at",
    "went to",
    "summer school at",
    "summer session at",
    "summer program at",
    "summer institute at",
    "summer student at",
    "attended summer school",
    "attended a summer program",
    "attended summer program",
}
_SUMMER_ATTENDED_TERMS = {
    "summer school",
    "summer session",
    "summer program",
    "summer institute",
    "summer student",
    "summer course",
    "summer courses",
}
_HONORARY_TERMS = ["honorary", "honoris causa"]
_VISITING_TERMS = [
    "visiting professor",
    "visiting scholar",
    "visiting fellow",
]
_FACULTY_TERMS = [
    "professor at",
    "faculty at",
    "faculty member",
    "served as professor",
    "served as a professor",
    "was professor",
    "was a professor",
    "held appointment",
    "appointed professor",
    "teaching position",
    "lecturer at",
    "instructor at",
]
_NON_RESCUE_CONTEXT_TERMS = [
    "honorary",
    "lecture",
    "speaker",
    "keynote",
    "seminar",
    "workshop",
    "conference",
    "symposium",
    "guest lecture",
    "advisory board",
    "consultant",
]


def _coerce_message_text(value: Any) -> str:
    """Coerce OpenAI-style message content/reasoning payloads into plain text.

    Providers may return a string, list of content blocks, dict blocks, or None.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: List[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                # Common block formats: {"text": ...}, {"content": ...}, {"output_text": ...}
                text_part = item.get("text") or item.get("content") or item.get("output_text")
                if text_part is not None:
                    parts.append(_safe_text(text_part))
        return "\n".join(part for part in parts if part).strip()
    if isinstance(value, dict):
        text_part = value.get("text") or value.get("content") or value.get("output_text")
        if text_part is not None:
            return _safe_text(text_part)
    return _safe_text(value)


def _extract_response_content(message: Dict[str, Any], debug: bool = False) -> tuple[Optional[str], Optional[str]]:
    """Extract content and reasoning from an LLM response message.
    
    Handles both thinking models (with reasoning tokens) and non-thinking models.
    
    Returns:
        Tuple of (content, reasoning)
        - content: The final JSON output or answer
        - reasoning: The thinking/reasoning text (if present), None otherwise
    """
    reasoning = _coerce_message_text(
        message.get("reasoning") or message.get("reasoning_content")
    )
    content = _coerce_message_text(message.get("content"))
    
    if debug:
        print(f"[LLM PARSE] Message keys: {list(message.keys())}")
        if reasoning:
            print(f"[LLM PARSE] Detected reasoning tokens (length: {len(reasoning)})")
    
    # If we have reasoning but content is empty or doesn't look like final output,
    # try to extract JSON from the reasoning
    content_stripped = content.strip() if content else ""
    
    # Check if content looks like it might be part of thinking tokens rather than final output
    if reasoning and (not content_stripped or not content_stripped.startswith("{")):
        if debug:
            print(f"[LLM PARSE] Content missing or doesn't start with JSON. Searching in reasoning...")
        
        # Try to find a JSON object in the reasoning text
        # Look for the final JSON block (it might contain thinking before it)
        try:
            extracted_json = _clean_json_blob(reasoning)
            if extracted_json and extracted_json.strip().startswith("{"):
                if debug:
                    print(f"[LLM PARSE] Extracted JSON from reasoning: {extracted_json[:100]}...")
                return extracted_json, reasoning
        except Exception:
            pass
        if debug:
            print(f"[LLM PARSE] No valid JSON object found in reasoning")
    
    # Normal case: return content as-is (either from non-thinking model or thinking model's final output)
    return content, reasoning if reasoning else None


def _should_extract_thinking(debug: bool = False) -> bool:
    """Check if the current model is a thinking model and we should look for reasoning tokens."""
    should = is_thinking_model(MODEL_NAME)
    if debug:
        print(f"[LLM CONFIG] Model '{MODEL_NAME}' is thinking model: {should}")
    return should


async def get_session() -> aiohttp.ClientSession:
    """Return a shared aiohttp session, creating it on demand.
    
    The session can be safely shared across concurrent requests once created.
    The lock is only needed during creation to prevent multiple sessions.
    """
    global _session
    # Fast path: if session exists, return it immediately without locking
    if _session is not None and not _session.closed:
        return _session
    
    # Slow path: need to create session, so acquire lock
    async with _session_lock:
        # Double-check after acquiring lock (another coroutine might have created it)
        if _session is None or _session.closed:
            _session = aiohttp.ClientSession(timeout=DEFAULT_CLIENT_TIMEOUT)
        return _session


def set_prompt_logging(enabled: bool) -> None:
    """Globally enable or disable printing of LLM prompts."""
    global _PROMPT_LOGGING_ENABLED
    _PROMPT_LOGGING_ENABLED = bool(enabled)


def enable_prompt_logging() -> None:
    """Convenience helper to enable prompt logging."""
    set_prompt_logging(True)


def disable_prompt_logging() -> None:
    """Convenience helper to disable prompt logging."""
    set_prompt_logging(False)


async def refresh_session() -> None:
    """Force the shared session to refresh after a cool-down."""
    global _session, _last_refresh_time
    current = time.time()
    if current - _last_refresh_time < 5.0:
        return
    async with _session_lock:
        if current - _last_refresh_time < 5.0:
            return
        if _session and not _session.closed:
            try:
                await _session.close()
            except Exception:
                pass
            finally:
                _session = None
        _last_refresh_time = current


async def close_session() -> None:
    """Close the shared aiohttp session if it exists."""
    global _session
    async with _session_lock:
        if _session and not _session.closed:
            try:
                await _session.close()
            except Exception:
                pass
            finally:
                _session = None


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _console_safe(text: str) -> str:
    """Coerce text into the current console encoding without raising encoding errors."""
    if text is None:
        return ""
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        return text.encode(encoding, errors="replace").decode(encoding, errors="replace")
    except Exception:
        return text.encode("ascii", errors="replace").decode("ascii", errors="replace")


def _strip_accents(text: str) -> str:
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _normalize_simple(text: str) -> str:
    return re.sub(r"\s+", " ", _strip_accents(_safe_text(text)).lower()).strip()


def _tokenize_name(name: str) -> List[str]:
    base = _normalize_simple(name)
    if not base:
        return []
    tokens = [token for token in re.split(r"[^a-z0-9]+", base) if token]
    # Remove trailing generational suffixes
    while tokens and tokens[-1] in _NAME_SUFFIXES:
        tokens.pop()
    return tokens


def _institution_aliases(institution: str) -> List[str]:
    trimmed = _normalize_simple(institution)
    if not trimmed:
        return []
    aliases = {trimmed}
    parts = [token for token in trimmed.split() if token]
    aliases.update(parts)
    if "university" in trimmed:
        aliases.add(trimmed.replace("university", "").strip())
    if "college" in trimmed:
        aliases.add(trimmed.replace("college", "").strip())
    aliases.discard("")
    return sorted(alias for alias in aliases if alias)


def _institution_domain_guess(institution: str) -> Optional[str]:
    trimmed = _normalize_simple(institution)
    if not trimmed:
        return None
    words = [token for token in re.split(r"\s+", trimmed) if token and token not in {"the", "of", "at"}]
    if not words:
        return None
    primary = re.sub(r"[^a-z]", "", words[0])
    if not primary:
        return None
    return f"{primary}.edu"


def _result_combined_text(result: Dict[str, Any]) -> str:
    blocks = [
        _safe_text(result.get("page_excerpt")),
        _safe_text(result.get("snippet")),
        _safe_text(result.get("title")),
    ]
    return " ".join(block for block in blocks if block)


def _result_mentions_person(result: Dict[str, Any], name_tokens: List[str]) -> bool:
    """Check if a result mentions the person, using lenient matching to avoid false negatives.
    
    This is used to determine if we should include a result for LLM analysis.
    Being too strict here causes us to skip results that might contain connections,
    leading to missed true positives.
    
    Strategy:
    1. Trust explicit signals from search (has_person_name)
    2. Use token-based matching with multiple fallbacks
    3. Accept results with strong academic context even if name matching is uncertain
    """
    if not name_tokens:
        return False
    
    signals = result.get("signals") or {}
    
    # Trust explicit person signals from search
    if signals.get("has_person_name") or signals.get("has_person"):
        return True
    
    text = _normalize_simple(_result_combined_text(result))
    if not text:
        return False
    
    url_text = _normalize_simple(_safe_text(result.get("url")))
    last_token = name_tokens[-1] if name_tokens else ""
    first_token = name_tokens[0] if name_tokens else ""
    middle_tokens = name_tokens[1:-1]
    
    # Strategy 1: Check for last name (most reliable for academic contexts)
    if len(last_token) > 2 and last_token not in text:
        # Fallback: check URL for last name (e.g., /people/smith)
        if not (url_text and last_token in url_text):
            # Additional fallback: if we have strong academic + institution signals,
            # accept it anyway (the LLM will determine if it's the right person)
            has_strong_context = (
                signals.get("has_institution") and
                (signals.get("has_academic_role") or signals.get("relevance_score", 0) >= 8)
            )
            if not has_strong_context:
                return False
    
    # Strategy 2: Check for first name or first initial
    if len(first_token) > 2 and first_token not in text:
        first_initial = first_token[0]
        initial_with_last = re.compile(
            rf"\b{re.escape(first_initial)}\.?\s+{re.escape(last_token)}\b"
        )
        has_initial_combo = bool(initial_with_last.search(text))
        has_middle_token = any(len(token) > 2 and token in text for token in middle_tokens)
        name_in_url = url_text and first_token in url_text
        
        # More lenient: if we have institution + academic role + last name match,
        # accept even without first name (could be different format, e.g., "Prof. Smith")
        has_institution_and_role = (
            signals.get("has_institution") and 
            signals.get("has_academic_role") and
            last_token in text
        )
        
        if not (has_initial_combo or has_middle_token or name_in_url or has_institution_and_role):
            return False
    
    # Strategy 3: Handle first initial cases (e.g., "J. Smith")
    if len(first_token) == 1:
        initial_pattern = re.compile(rf"\b{re.escape(first_token)}\.?\b")
        has_initial = bool(initial_pattern.search(text))
        has_middle_token = any(len(token) > 2 and token in text for token in middle_tokens)
        
        # Accept if we have initial OR middle name OR strong context
        has_strong_context = (
            signals.get("has_institution") and
            last_token in text and
            (signals.get("has_academic_role") or url_text.endswith(".edu"))
        )
        
        if not (has_initial or has_middle_token or has_strong_context):
            return False
    
    return True


def _result_mentions_person_strict(result: Dict[str, Any], name_tokens: List[str]) -> bool:
    """Strict person-match check used to validate positive LLM decisions.

    Requires a reliable last-name match plus first-name/initial support.
    """
    if not name_tokens:
        return False

    text = _normalize_simple(_result_combined_text(result))
    url_text = _normalize_simple(_safe_text(result.get("url")))
    haystack = f"{text} {url_text}".strip()
    if not haystack:
        return False

    last_token = name_tokens[-1]
    first_token = name_tokens[0]

    if len(last_token) <= 2:
        return False
    if last_token not in haystack:
        return False

    if len(first_token) > 1 and first_token in haystack:
        return True

    first_initial = first_token[:1]
    if first_initial:
        initial_last = re.compile(rf"\b{re.escape(first_initial)}\.?\s+{re.escape(last_token)}\b")
        if initial_last.search(haystack):
            return True

    for token in name_tokens[1:-1]:
        if len(token) > 2 and token in haystack:
            return True

    return False


def _decision_mentions_target_person(
    decision: Dict[str, str],
    name: str,
    results: Optional[List[Dict[str, Any]]] = None,
) -> bool:
    """Verify connected evidence appears to reference the target person."""
    name_tokens = _tokenize_name(name)
    if not name_tokens:
        return True

    detail = _safe_text(decision.get("verification_detail"))
    summary = _safe_text(decision.get("summary"))
    primary_source = _safe_text(decision.get("primary_source") or decision.get("supporting_url"))
    decision_text = _normalize_simple(f"{detail} {summary} {primary_source}")

    if decision_text:
        synthetic_result = {
            "title": summary,
            "snippet": detail,
            "page_excerpt": detail,
            "url": primary_source,
        }
        if _result_mentions_person_strict(synthetic_result, name_tokens):
            return True

    for result in (results or [])[:10]:
        if _result_mentions_person_strict(result, name_tokens):
            return True

    return False


def _decision_mentions_target_institution(
    decision: Dict[str, str],
    institution: str,
    results: Optional[List[Dict[str, Any]]] = None,
) -> bool:
    """Verify connected evidence is explicitly tied to the target institution."""
    aliases = _institution_aliases(institution)
    guess_domain = _institution_domain_guess(institution)
    if not aliases and not guess_domain:
        return True

    alias_pattern = re.compile("|".join(re.escape(alias) for alias in aliases), re.IGNORECASE) if aliases else None

    detail = _safe_text(decision.get("verification_detail"))
    summary = _safe_text(decision.get("summary"))
    primary_source = _safe_text(decision.get("primary_source") or decision.get("supporting_url"))
    decision_text = _normalize_simple(f"{detail} {summary} {primary_source}")

    if alias_pattern and alias_pattern.search(decision_text):
        return True

    if primary_source:
        domain = urlparse(primary_source).netloc.lower()
        if guess_domain and guess_domain in domain:
            return True
        if alias_pattern and alias_pattern.search(domain):
            return True

    # Fallback: require at least one supporting result that mentions both person and institution.
    for result in (results or [])[:20]:
        signals = result.get("signals") or {}
        if not signals.get("has_person_name"):
            continue

        if signals.get("has_institution"):
            return True

        url = _safe_text(result.get("url"))
        domain = urlparse(url).netloc.lower() if url else ""
        text = _normalize_simple(_result_combined_text(result))
        if alias_pattern and alias_pattern.search(text):
            return True
        if guess_domain and guess_domain in domain:
            return True
        if alias_pattern and alias_pattern.search(domain):
            return True

    return False


def _trim_evidence_text(text: str) -> str:
    clean = _safe_text(text)
    if len(clean) <= 220:
        return clean
    return clean[:217].rstrip() + "..."


def _classify_evidence_window(window_lower: str) -> Optional[str]:
    if not window_lower:
        return None
    if any(term in window_lower for term in _HONORARY_TERMS):
        return None
    if any(term in window_lower for term in _VISITING_TERMS):
        return "Visiting"
    if any(term in window_lower for term in _FACULTY_TERMS):
        return "Faculty"
    if any(term in window_lower for term in _SUMMER_ATTENDED_TERMS):
        return "Attended"
    if any(term in window_lower for term in _ATTENDED_TERMS):
        return "Attended"
    if any(term in window_lower for term in _ALUMNI_TERMS):
        return "Alumni"
    if any(verb in window_lower for verb in _DEGREE_VERBS) and any(
        phrase in window_lower for phrase in (*_DEGREE_NOUNS, *_DEGREE_PHRASES)
    ):
        return "Alumni"
    return None


def _looks_like_non_rescue_context(window_lower: str, evidence_type: Optional[str]) -> bool:
    if not window_lower:
        return False
    # Summer attendance is a valid enrolled/training signal, not just event noise.
    if evidence_type == "Attended" and any(term in window_lower for term in _SUMMER_ATTENDED_TERMS):
        return False
    if evidence_type == "Visiting":
        filtered_terms = [term for term in _NON_RESCUE_CONTEXT_TERMS if term not in {"lecture", "speaker", "seminar"}]
    else:
        filtered_terms = _NON_RESCUE_CONTEXT_TERMS
    return any(term in window_lower for term in filtered_terms)


def _find_search_evidence(
    name: str,
    institution: str,
    results: Iterable[Dict[str, Any]],
) -> Optional[Tuple[str, str, str, str]]:
    """Scan search results for strong alumni/attended evidence.

    Returns tuple (connection_type, detail, url, confidence).
    """
    name_tokens = _tokenize_name(name)
    if not name_tokens:
        return None
    aliases = _institution_aliases(institution)
    if not aliases:
        return None
    alias_pattern = re.compile("|".join(re.escape(alias) for alias in aliases), re.IGNORECASE)

    for result in results or []:
        url = _safe_text(result.get("url"))
        if not url:
            continue
        domain = urlparse(url).netloc.lower()
        text_original = _result_combined_text(result)
        if not text_original:
            continue
        text_lower = text_original.lower()
        if not _result_mentions_person(result, name_tokens):
            continue
        for match in alias_pattern.finditer(text_lower):
            start = max(0, match.start() - _EVIDENCE_WINDOW_CHARS)
            end = min(len(text_lower), match.end() + _EVIDENCE_WINDOW_CHARS)
            window_lower = text_lower[start:end]
            evidence_type = _classify_evidence_window(window_lower)
            if evidence_type:
                if _looks_like_non_rescue_context(window_lower, evidence_type):
                    continue
                window_original = text_original[start:end]
                detail = _trim_evidence_text(window_original)
                confidence = "high" if domain.endswith(".edu") else "medium"
                return (evidence_type, detail, url, confidence)
    return None


def _result_has_historical_role_evidence(result: Dict[str, Any]) -> bool:
    text = _result_combined_text(result).lower()
    if not text:
        return False
    evidence_type = _classify_evidence_window(text)
    if not evidence_type:
        return False
    if _looks_like_non_rescue_context(text, evidence_type):
        return False
    return True


def _result_has_historical_anchor_evidence(result: Dict[str, Any]) -> bool:
    signals = result.get("signals") or {}
    if any(
        signals.get(key)
        for key in (
            "has_historical_role_anchor",
            "has_historical_education_anchor",
            "has_historical_profile_anchor",
        )
    ):
        return True
    return _result_has_historical_role_evidence(result)


def _should_attempt_historical_llm_rescue(
    decision: Dict[str, str],
    institution: str,
    results: List[Dict[str, Any]],
    *,
    vip_mode: bool = False,
) -> bool:
    if decision.get("connected") == "Y":
        return False
    if not results:
        return False

    authoritative_hits = 0
    repeated_support = 0
    role_hits = 0
    for result in results:
        url = _safe_text(result.get("url"))
        domain = urlparse(url).netloc.lower() if url else ""
        signals = result.get("signals") or {}
        has_anchor = _result_has_historical_anchor_evidence(result)
        if has_anchor:
            role_hits += 1
        if domain.endswith(".edu") and has_anchor:
            authoritative_hits += 1
        if (
            signals.get("has_person_name")
            and signals.get("has_institution")
            and has_anchor
        ):
            repeated_support += 1

    if authoritative_hits >= 1 and repeated_support >= 1:
        return True
    if role_hits >= 2 and repeated_support >= 2:
        return True
    return False


def _result_prompt_priority(result: Dict[str, Any]) -> Tuple[int, int, int]:
    signals = result.get("signals", {}) or {}
    url = _safe_text(result.get("url"))
    domain = urlparse(url).netloc.lower() if url else ""
    role_priority = 1 if _result_has_historical_anchor_evidence(result) else 0
    edu_priority = 1 if domain.endswith(".edu") else 0
    explicit_priority = 1 if signals.get("has_explicit_connection") else 0
    profile_priority = 1 if signals.get("has_historical_profile_anchor") else 0
    return (role_priority + explicit_priority + profile_priority, edu_priority, signals.get("relevance_score", 0))


def _has_institution_signal(results: Iterable[Dict[str, Any]], institution: str) -> bool:
    aliases = _institution_aliases(institution)
    guess = _institution_domain_guess(institution)
    if not aliases and not guess:
        return False
    alias_pattern = re.compile("|".join(re.escape(alias) for alias in aliases), re.IGNORECASE) if aliases else None

    for result in results or []:
        signals = result.get("signals") or {}
        if signals.get("has_institution"):
            return True
        url = _safe_text(result.get("url"))
        domain = urlparse(url).netloc.lower() if url else ""
        text_lower = _result_combined_text(result).lower()
        if alias_pattern and alias_pattern.search(text_lower):
            return True
        if guess and guess in domain:
            return True
        if domain.endswith(".edu") and alias_pattern and alias_pattern.search(domain):
            return True
    return False


def _normalize_verdict(value: Any) -> str:
    text = _safe_text(value).lower()
    if text in {"connected", "yes", "y"}:
        return "connected"
    if text in {"not_connected", "no", "n"}:
        return "not_connected"
    if text in {"uncertain", "uncertain_connection", "maybe"}:
        return "uncertain"
    return "uncertain"


def _normalize_connected(verdict: str) -> str:
    if verdict == "connected":
        return "Y"
    if verdict == "not_connected":
        return "N"
    return "N"


def _normalize_connection_type(value: Any, verdict: str) -> str:
    """Normalize connection type to valid options.
    
    Note: Attended and Alumni are valid connection types and should be preserved
    even if the verdict is uncertain or not_connected in edge cases.
    """
    text = _safe_text(value).strip()
    if verdict != "connected":
        return "None"

    valid_types = {
        "alumni": "Alumni",
        "attended": "Attended",
        "executive": "Executive",
        "faculty": "Faculty",
        "postdoc": "Postdoc",
        "staff": "Staff",
        "visiting": "Visiting",
        "other": "Other",
        "others": "Other"
    }

    text_lower = text.lower()
    if text_lower in valid_types:
        return valid_types[text_lower]

    if "alum" in text_lower or "graduate" in text_lower:
        return "Alumni"
    if "student" in text_lower and "post" not in text_lower:
        return "Attended"
    if any(term in text_lower for term in ("prof", "teach", "lecturer", "instructor")):
        return "Faculty"
    if "postdoc" in text_lower or "post-doc" in text_lower:
        return "Postdoc"
    if any(term in text_lower for term in ("president", "dean", "director", "admin", "chancellor", "provost")):
        return "Executive"
    if any(term in text_lower for term in ("staff", "researcher", "scientist", "engineer")):
        return "Staff"
    if "visit" in text_lower:
        return "Visiting"

    return "Other"


def _normalize_timeframe(value: Any, verdict: str) -> str:
    text = _safe_text(value).lower()
    if text in {"current", "present"}:
        return "current"
    if text in {"past", "former", "previous"}:
        return "past"
    if text in {"na", "n/a", "unknown", "unsure"}:
        return "unknown"
    return "unknown" if verdict != "connected" else "unknown"


def _normalize_verification_status(value: Any, verdict: str) -> str:
    text = _safe_text(value).lower()
    if text in {"verified", "complete", "confirmed"}:
        return "verified"
    if text in {"needs_review", "review", "manual"}:
        return "needs_review"
    return "needs_review" if verdict != "connected" else "verified"


def _normalize_confidence(value: Any) -> str:
    text = _safe_text(value).lower()
    if text in {"high", "medium", "low"}:
        return text
    if text in {"strong"}:
        return "high"
    if text in {"uncertain", "weak"}:
        return "low"
    return "medium"


def _clean_json_blob(text: str) -> str:
    """Extract and clean JSON from LLM response, handling various formats."""
    # Remove control characters
    text = CONTROL_CHAR_RE.sub("", text)
    
    # Try to extract from markdown code block first
    match = JSON_BLOCK_RE.search(text)
    if match:
        candidate = match.group(1).strip()
        if candidate:
            return candidate
    
    # Check if already pure JSON
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    
    # Extract from code fence
    fence_match = JSON_FENCE_RE.search(text)
    if fence_match:
        inner = fence_match.group(0).strip("`").strip()
        if inner.startswith("{"):
            return inner
    
    # Find first complete JSON object
    start = text.find("{")
    if start == -1:
        return text
    
    # Find matching closing brace
    brace_count = 0
    in_string = False
    escape_next = False
    
    for i in range(start, len(text)):
        char = text[i]
        
        if escape_next:
            escape_next = False
            continue
        
        if char == '\\':
            escape_next = True
            continue
        
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        
        if in_string:
            continue
        
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0:
                return text[start:i + 1]
    
    # Fallback to rfind if balanced parsing failed
    end = text.rfind("}")
    if end != -1 and start < end:
        return text[start:end + 1]
    
    return text


def _build_error(reason: str) -> Dict[str, str]:
    message = _safe_text(reason) or "Unknown error"
    return {
        "verdict": "not_connected",
        "connected": "N",
        "relationship_type": "None",
        "relationship_timeframe": "unknown",
        "verification_detail": f"Error: {message}",
        "summary": f"Processing error: {message}",
        "primary_source": "",
        "confidence": "low",
        "verification_status": "needs_review",
        "temporal_context": f"Processing error: {message}",
        "connection_type": "Others",
        "connection_detail": f"Error: {message}",
        "current_or_past": "N/A",
        "supporting_url": "",
        "temporal_evidence": f"Processing error: {message}",
    }


def _normalize_decision(payload: Dict[str, Any]) -> Dict[str, str]:
    verdict_source = payload.get("verdict") or payload.get("connected")
    verdict = _normalize_verdict(verdict_source)
    connected = _normalize_connected(verdict)

    relationship_type_source = payload.get("relationship_type") or payload.get("connection_type")
    relationship_type = _normalize_connection_type(relationship_type_source, verdict)

    timeframe_source = payload.get("relationship_timeframe") or payload.get("current_or_past")
    relationship_timeframe = _normalize_timeframe(timeframe_source, verdict)

    if verdict == "connected" and relationship_type in {"Alumni", "Attended"}:
        if relationship_timeframe in {"current", "unknown"}:
            relationship_timeframe = "past"

    verification_detail = _safe_text(
        payload.get("verification_detail") or payload.get("connection_detail")
    )
    summary_text = _safe_text(payload.get("summary"))
    primary_source = _safe_text(
        payload.get("primary_source") or payload.get("supporting_url")
    )
    confidence = _normalize_confidence(payload.get("confidence"))
    verification_status = _normalize_verification_status(payload.get("verification_status"), verdict)
    temporal_context = _safe_text(
        payload.get("temporal_context") or payload.get("temporal_evidence")
    )

    if not summary_text:
        if verification_detail:
            summary_text = verification_detail[:220]
        elif verdict == "connected":
            summary_text = f"Confirmed {relationship_type.lower()} link to institution."
        elif verdict == "not_connected":
            summary_text = "No verified institutional relationship discovered."
        else:
            summary_text = "Insufficient evidence to confirm a relationship."

    normalized = {
        "verdict": verdict,
        "connected": connected,
        "relationship_type": relationship_type,
        "relationship_timeframe": relationship_timeframe,
        "verification_detail": verification_detail,
        "summary": summary_text,
        "primary_source": primary_source,
        "confidence": confidence,
        "verification_status": verification_status,
        "temporal_context": temporal_context or "unknown",
    }

    # Backward-compatible aliases for downstream consumers that still expect legacy keys
    legacy_type = relationship_type if verdict == "connected" else "Others"
    legacy_timeframe = (
        relationship_timeframe
        if verdict == "connected" and relationship_timeframe in {"current", "past"}
        else "N/A"
    )
    normalized.update(
        {
            "connection_type": legacy_type,
            "connection_detail": verification_detail or summary_text,
            "current_or_past": legacy_timeframe,
            "supporting_url": primary_source,
            "temporal_evidence": temporal_context or "unknown",
        }
    )

    if verdict != "connected":
        normalized["confidence"] = _normalize_confidence("low")

    return normalized


def _format_result_row(result: Dict[str, Any], index: int) -> str:
    """Format a search result for LLM consumption with clear structure."""
    title = _safe_text(result.get("title"))
    snippet = _safe_text(result.get("snippet"))
    excerpt = _safe_text(result.get("page_excerpt"))
    url = _safe_text(result.get("url"))
    rank = result.get("rank")
    
    # Format with clear labels for better LLM parsing
    rank_prefix = f"#{rank} " if isinstance(rank, int) else ""
    lines = []
    
    if title:
        lines.append(f"Title: {rank_prefix}{title}")
    if snippet:
        trimmed_snippet = snippet if len(snippet) <= 220 else f"{snippet[:217].rstrip()}..."
        lines.append(f"Snippet: {trimmed_snippet}")
    if excerpt:
        trimmed_excerpt = excerpt if len(excerpt) <= 300 else f"{excerpt[:297].rstrip()}..."
        lines.append(f"Excerpt: {trimmed_excerpt}")
    if url:
        lines.append(f"URL: {url}")
    
    # Use newlines for better readability by LLM
    payload = "\n   ".join(lines)
    return f"{index + 1}. {payload}"


def _summarise_results(
    results: Iterable[Dict[str, Any]],
    limit: int = MAX_RESULTS_FOR_PROMPT,
    min_relevance_score: int = 8,
    allow_low_quality: bool = False,
) -> str:
    """Format search results for LLM prompt with quality filtering.
    
    Filters low-quality results using a configurable minimum relevance score.
    A score around 8 indicates reasonable relevance (Name + Institution + some signal).
    """
    rows = []
    result_count = 0
    ordered_results = sorted(
        list(results),
        key=lambda item: (
            -_result_prompt_priority(item)[0],
            -_result_prompt_priority(item)[1],
            -_result_prompt_priority(item)[2],
            item.get("rank", 9999),
        ),
    )

    for item in ordered_results:
        # Skip low-quality results (ads, irrelevant content, weak mentions)
        # Threshold lowered to 8 to capture more context
        signals = item.get("signals", {}) or {}
        relevance_score = signals.get("relevance_score", 0)
        if relevance_score < min_relevance_score:
            if not allow_low_quality:
                continue
            url = _safe_text(item.get("url"))
            domain = urlparse(url).netloc.lower() if url else ""
            has_excerpt = bool(_safe_text(item.get("page_excerpt")))
            has_explicit = bool(signals.get("has_explicit_connection"))
            is_edu = domain.endswith(".edu")
            if not (has_excerpt or has_explicit or is_edu):
                continue
        
        rows.append(_format_result_row(item, result_count))
        result_count += 1
        
        if result_count >= limit:
            break
    
    return "\n".join(rows) if rows else "(no search results available)"


def _build_prompt(
    name: str,
    institution: str,
    results: List[Dict[str, Any]],
    vip_mode: bool = False,
) -> str:
    now = datetime.utcnow()
    limit = VIP_RESULTS_FOR_PROMPT if vip_mode else MAX_RESULTS_FOR_PROMPT
    min_score = 4 if vip_mode else 8
    findings = _summarise_results(
        results,
        limit=limit,
        min_relevance_score=min_score,
        allow_low_quality=vip_mode,
    )
    return PROMPT_TEMPLATE.format(
        current_date=now.strftime("%B %d, %Y"),
        current_year=now.year,
        prior_year=now.year - 1,
        name=name,
        institution=institution,
        search_findings=findings,
        max_results=min(len(results), limit),
    )


def _extract_fields_with_regex(text: str) -> Dict[str, Any]:
    """Extract fields using regex patterns as a fallback when JSON parsing fails."""
    result = {}
    
    # Pattern for each field - very flexible
    patterns = {
        "verdict": r'"verdict"\s*:\s*"(connected|not_connected|uncertain)"',
        "relationship_type": r'"relationship_type"\s*:\s*"(Alumni|Attended|Executive|Faculty|Postdoc|Staff|Visiting|Other|None)"',
        "relationship_timeframe": r'"relationship_timeframe"\s*:\s*"(current|past|unknown)"',
        "verification_detail": r'"verification_detail"\s*:\s*"([^"]*(?:\\.[^"]*)*)"',
        "summary": r'"summary"\s*:\s*"([^"]*(?:\\.[^"]*)*)"',
        "primary_source": r'"primary_source"\s*:\s*"([^"]*)"',
        "confidence": r'"confidence"\s*:\s*"(high|medium|low)"',
        "verification_status": r'"verification_status"\s*:\s*"(verified|needs_review)"',
        "temporal_context": r'"temporal_context"\s*:\s*"([^"]*(?:\\.[^"]*)*)"',
        # Legacy fields (for compatibility with older responses)
        "connected": r'"connected"\s*:\s*"([YN])"',
        "connection_type": r'"connection_type"\s*:\s*"(Alumni|Attended|Executive|Faculty|Postdoc|Staff|Other|Others)"',
        "connection_detail": r'"connection_detail"\s*:\s*"([^"]*(?:\\.[^"]*)*)"',
        "current_or_past": r'"current_or_past"\s*:\s*"(current|past|N/A)"',
        "supporting_url": r'"supporting_url"\s*:\s*"([^"]*)"',
        "temporal_evidence": r'"temporal_evidence"\s*:\s*"([^"]*(?:\\.[^"]*)*)"',
    }
    
    for field, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            value = match.group(1)
            # Unescape common patterns
            value = value.replace('\\"', '"').replace('\\n', ' ').replace('\\\\', '\\')
            result[field] = value

    if not result:
        return {}

    # Set defaults for any missing optional fields once we know at least one key matched
    defaults = {
        "verdict": "uncertain",
        "relationship_type": "None",
        "relationship_timeframe": "unknown",
        "verification_detail": "Unable to determine",
        "summary": "Insufficient information to confirm relationship",
        "primary_source": "",
        "confidence": "low",
        "verification_status": "needs_review",
        "temporal_context": "Insufficient information",
        "connected": "N",
        "connection_type": "Others",
        "connection_detail": "Unable to determine",
        "current_or_past": "N/A",
        "supporting_url": "",
        "temporal_evidence": "Insufficient information",
    }
    for field, default in defaults.items():
        result.setdefault(field, default)

    return result


def _sanitize_json_string(text: str) -> str:
    """Pre-process text to fix common JSON formatting issues."""
    # Remove any markdown code block markers
    text = re.sub(r'```(?:json)?', '', text)
    text = text.strip('`').strip()
    
    # Remove BOM and other invisible characters
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
    
    # Fix common issues with quotes in strings
    # This is a heuristic approach - look for patterns like: "text "word" more"
    # and replace the inner quotes with escaped quotes
    def fix_unescaped_quotes(match):
        key = match.group(1)
        value = match.group(2)
        # Count quotes to see if they're unbalanced
        if value.count('"') % 2 == 1:  # Odd number of quotes = problem
            # Escape all internal quotes
            value = value.replace('"', '\\"')
        return f'"{key}": "{value}"'
    
    # Try to fix field values with unescaped quotes
    text = re.sub(
        r'"(\w+)"\s*:\s*"((?:[^"\\]|\\.)*)"',
        lambda m: f'"{m.group(1)}": "{m.group(2)}"',
        text
    )
    
    # Remove trailing commas
    text = re.sub(r',\s*([}\]])', r'\1', text)
    
    # Fix newlines in string values (replace with spaces)
    # Find strings and replace newlines within them
    def remove_newlines_in_strings(match):
        return match.group(0).replace('\n', ' ').replace('\r', '')
    
    text = re.sub(r'"[^"]*"', remove_newlines_in_strings, text)
    
    return text


def _ensure_json_object(text: str) -> str:
    """Ensure the text looks like a JSON object by adding braces if missing."""
    stripped = text.strip()
    if not stripped:
        return stripped

    has_open = stripped.startswith("{")
    has_close = stripped.endswith("}")

    if not has_open:
        stripped = "{" + stripped
    if not has_close:
        stripped = stripped.rstrip(", \n\r\t")
        stripped = stripped + "}"

    return stripped


def _parse_response(raw: str) -> Dict[str, Any]:
    """Parse LLM response with multiple fallback strategies."""
    # Strategy 1: Clean and extract JSON
    candidate = _clean_json_blob(raw)
    
    # Strategy 2: Sanitize before parsing
    sanitized = _sanitize_json_string(candidate)
    sanitized_wrapped = _ensure_json_object(sanitized)
    
    # Try parsing the sanitized version
    try:
        return json.loads(sanitized_wrapped)
    except json.JSONDecodeError:
        pass
    
    # Strategy 3: Try the original candidate
    try:
        return json.loads(_ensure_json_object(candidate))
    except json.JSONDecodeError:
        pass
    
    # Strategy 4: Try common fixes
    try:
        # Fix trailing commas
        fixed = re.sub(r',\s*([}\]])', r'\1', candidate)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    try:
        # Try replacing single quotes with double quotes
        fixed = candidate.replace("'", '"')
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    try:
        # Remove newlines and try again
        fixed = candidate.replace('\n', ' ').replace('\r', '')
        fixed = re.sub(r',\s*([}\]])', r'\1', fixed)
        return json.loads(_ensure_json_object(fixed))
    except json.JSONDecodeError:
        pass
    
    # Strategy 5: Use regex extraction as last resort
    try:
        extracted = _extract_fields_with_regex(raw)
        if extracted:
            return extracted
    except Exception:
        pass
    
    # If all strategies fail, raise the original error
    raise json.JSONDecodeError(
        f"Could not parse response after trying multiple strategies. Raw: {raw[:200]}...",
        raw,
        0
    )


def _format_llm_response_diagnostics(response: aiohttp.ClientResponse, body_text: str = "") -> str:
    """Extract request/rate-limit diagnostics from provider responses."""
    try:
        headers = response.headers or {}
    except Exception:
        headers = {}

    def _first_header(*names: str) -> str:
        for name in names:
            value = headers.get(name)
            if value:
                return str(value)
        return ""

    request_id = _first_header("x-request-id", "request-id", "openai-request-id")
    retry_after = _first_header("retry-after")
    rpm_limit = _first_header("x-ratelimit-limit-requests")
    rpm_remaining = _first_header("x-ratelimit-remaining-requests")
    rpm_reset = _first_header("x-ratelimit-reset-requests")

    parts: List[str] = []
    if request_id:
        parts.append(f"request_id={request_id}")
    if retry_after:
        parts.append(f"retry_after={retry_after}")
    if rpm_limit:
        parts.append(f"rpm_limit={rpm_limit}")
    if rpm_remaining:
        parts.append(f"rpm_remaining={rpm_remaining}")
    if rpm_reset:
        parts.append(f"rpm_reset={rpm_reset}")

    stripped = (body_text or "").strip().lower()
    if stripped == "null":
        parts.append("provider_body=null")
    if "rate limit" in stripped or "too many requests" in stripped:
        parts.append("rate_limit_text_detected=true")

    return ", ".join(parts) if parts else "no_diagnostics"


async def _call_llm(prompt: str, debug: bool = False) -> Dict[str, Any]:
    """Call LLM with temperature 0 for fully deterministic outputs."""
    session = await get_session()
    headers = {
        "Authorization": f"Bearer {get_api_key()}",
        "Content-Type": "application/json",
    }
    base_payload: Dict[str, Any] = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that outputs only valid JSON objects. Never include markdown, reasoning, or explanations.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,  # Deterministic output
        "response_format": {"type": "json_object"},
        "stream": False,
    }

    # Single payload attempt - retries are handled at a higher level in analyze_connection
    payload = dict(base_payload)
    last_error_text: Optional[str] = None
    
    # Make single API call with provided temperature
    try:
        async with session.post(LLM_API_URL, json=payload, headers=headers) as response:
            text = await response.text()
            last_error_text = text
            diagnostics = _format_llm_response_diagnostics(response, text)

            if debug:
                print(f"[LLM] Response status: {response.status}")
                print(f"[LLM] Response length: {len(text)} chars")

            if response.status >= 500:
                raise RetryableLLMError(
                    f"LLM server error {response.status}: {text[:200]} (diag: {diagnostics})"
                )

            if response.status == 429:
                raise RetryableLLMError(
                    f"LLM rate limited (429): {text[:200]} (diag: {diagnostics})"
                )

            if response.status >= 400:
                raise RuntimeError(
                    f"LLM request failed ({response.status}): {text[:200]} (diag: {diagnostics})"
                )

            try:
                data = json.loads(text)
            except json.JSONDecodeError as exc:
                if debug:
                    print("[LLM] Failed to parse server response as JSON")
                raise RetryableLLMError(f"Invalid JSON from server: {exc}") from exc
        
            # Defensive check: ensure data is a dict (not null/list/other type)
            if not isinstance(data, dict):
                if debug:
                    print(f"[LLM] ERROR: Response is not a dict, it's a {type(data).__name__}")
                    print(f"[LLM] Raw response text preview: {text[:200]}")
                raise RetryableLLMError(
                    f"LLM response is not a JSON object: got {type(data).__name__}"
                    f" (raw_text={text[:80]!r}, diag: {diagnostics})"
                )
        
            # Debug: Print full API response structure
            if debug:
                print(f"[LLM DEBUG] Full API response keys: {list(data.keys())}")
                print(f"[LLM DEBUG] Full API response: {json.dumps(data, indent=2)}")

            choices = data.get("choices") or []
            if not choices:
                if debug:
                    print("[LLM] No choices in response")
                raise RetryableLLMError("LLM response contained no choices")

            first_choice = choices[0] if choices else {}
            if not isinstance(first_choice, dict):
                raise RuntimeError(f"LLM choice is not a JSON object: got {type(first_choice).__name__}")

            message = first_choice.get("message", {})
            if not isinstance(message, dict):
                raise RuntimeError(f"LLM message is not a JSON object: got {type(message).__name__}")
        
            # Debug: Print the entire message structure
            if debug:
                print(f"[LLM DEBUG] Message keys: {list(message.keys())}")
                print(f"[LLM DEBUG] Full message structure: {json.dumps(message, indent=2)}")
        
            # Extract content from both thinking and non-thinking models
            content, reasoning = _extract_response_content(message, debug=debug)
            content_stripped = content.strip() if content else ""
        
            if debug:
                if reasoning:
                    print(f"[LLM] Reasoning length: {len(reasoning)} chars")
                    print(f"[LLM] Reasoning preview: {reasoning[:300]}...")
                print(f"[LLM] Content length: {len(content_stripped)} chars")
                if len(content_stripped) < 500:
                    print(f"[LLM] Full content: {content_stripped}")
                else:
                    print(f"[LLM] Content preview: {content_stripped[:300]}...")
                if not content_stripped.endswith("}"):
                    print("[LLM] WARNING: Response doesn't end with }, likely TRUNCATED")
                    print(f"[LLM] Last 200 chars: ...{content_stripped[-200:]}")

            if debug or _PROMPT_LOGGING_ENABLED:
                separator = "=" * 48
                if reasoning:
                    print(f"{separator}\n[LLM REASONING]\n{separator}")
                    print(reasoning)
                    print(f"{separator}\n[END REASONING]\n{separator}")
                print(f"{separator}\n[LLM RESPONSE (JSON)]\n{separator}")
                print(content_stripped)
                print(f"{separator}\n[END RESPONSE]\n{separator}")

            if not content_stripped:
                if debug:
                    print(f"[LLM] Empty content received from API")
                raise RetryableLLMError("LLM returned empty content")

            return _parse_response(content_stripped)
    except (aiohttp.ClientError, ssl.SSLError, OSError) as exc:
        message = str(exc)
        if "APPLICATION_DATA_AFTER_CLOSE_NOTIFY" in message or "SSL" in message.upper():
            raise RetryableLLMError(f"Transient SSL transport error: {message}") from exc
        raise RetryableLLMError(f"Transient transport error: {message}") from exc


def _extract_domain(url: str) -> str:
    try:
        netloc = urlparse(url).netloc.lower()
        # Remove common prefixes for better matching
        for prefix in ['www.', 'web.', 'm.']:
            if netloc.startswith(prefix):
                netloc = netloc[len(prefix):]
        return netloc
    except Exception:
        return ""

def _extract_years(text: str) -> List[int]:
    """Extract all year mentions from text."""
    return [int(m.group(0)) for m in YEAR_PATTERN.finditer(text)]


def _extract_date_ranges(text: str) -> List[tuple[int, int]]:
    """Extract date ranges from text."""
    ranges = []
    
    # Pattern 1: "2010-2015"
    for match in DATE_RANGE_PATTERN.finditer(text):
        start_year = int(match.group(1))
        end_year = int(match.group(2))
        ranges.append((start_year, end_year))
    
    # Pattern 2: "from 2010 to 2015"
    for match in FROM_TO_PATTERN.finditer(text):
        start_year = int(match.group(1))
        end_year = int(match.group(2))
        ranges.append((start_year, end_year))
    
    return ranges


def _extract_temporal_bounds(text: str) -> tuple[Optional[int], Optional[int]]:
    """Extract start and end years from text.
    
    Returns:
        (start_year, end_year) where None means unbounded
    """
    start_year = None
    end_year = None
    
    # Look for "since X"
    since_matches = list(SINCE_PATTERN.finditer(text))
    if since_matches:
        start_year = int(since_matches[-1].group(1))  # Most recent mention
    
    # Look for "until X"
    until_matches = list(UNTIL_PATTERN.finditer(text))
    if until_matches:
        end_year = int(until_matches[-1].group(1))  # Most recent mention
    
    # Look for explicit ranges
    ranges = _extract_date_ranges(text)
    if ranges:
        # Use the most recent/relevant range
        most_recent_range = max(ranges, key=lambda r: r[1])  # Range with latest end year
        if start_year is None:
            start_year = most_recent_range[0]
        if end_year is None:
            end_year = most_recent_range[1]
    
    return start_year, end_year


def _has_end_date(text: str, current_year: int) -> bool:
    """Check if text contains evidence of an ended position (past tense)."""
    _, end_year = _extract_temporal_bounds(text)
    
    if end_year and end_year < current_year - 1:
        return True
    
    return False


def _infer_temporal_status(text: str, current_year: int) -> Optional[str]:
    """Infer if position is current or past based on dates in text.
    
    Returns:
        "current", "past", or None if unclear
    """
    start_year, end_year = _extract_temporal_bounds(text)
    
    # If we have an explicit end date in the past, it's past
    if end_year and end_year < current_year - 1:
        return "past"
    
    # If we have a recent end date (last year or this year), it might still be current
    # unless there's explicit "former" language
    if end_year and end_year >= current_year - 1:
        # Check for past tense indicators
        if _contains_any(text.lower(), PAST_TERMS):
            return "past"
        return "current"
    
    # If we have only a start date with no end date
    if start_year and not end_year:
        # If start is recent (within last 15 years), assume current unless past language
        if start_year >= current_year - 15:
            if _contains_any(text.lower(), PAST_TERMS):
                return "past"
            return "current"
        else:
            # Old start date with no end date - likely past
            return "past"
    
    # If we have a recent year mention
    all_years = _extract_years(text)
    if all_years:
        most_recent = max(all_years)
        if most_recent >= current_year - 1:
            # Recent year, check for temporal language
            if _contains_any(text.lower(), CURRENT_TERMS):
                return "current"
            elif _contains_any(text.lower(), PAST_TERMS):
                return "past"
    
    return None  # Unclear


def _count_temporal_signals(text: str, terms: List[str]) -> int:
    """Count how many temporal signal terms appear in text."""
    count = 0
    text_lower = text.lower()
    for term in terms:
        count += text_lower.count(term)
    return count


def _score_url_quality(url: str, domain: str) -> str:
    """Score the quality/reliability of a source URL.
    
    Returns:
        "high", "medium", or "low"
    """
    url_lower = url.lower()
    domain_lower = domain.lower()
    
    # HIGH quality: Official institutional pages
    if domain_lower.endswith('.edu'):
        # Faculty pages, department pages, official directories
        high_quality_paths = [
            '/faculty/', '/people/', '/staff/', '/directory/', '/profile/',
            '/about/people', '/about/faculty', '/dept/', '/department/',
            '/academics/faculty', '/our-faculty', '/our-people'
        ]
        if any(path in url_lower for path in high_quality_paths):
            return "high"
        
        # News pages from .edu domains are also high quality
        if '/news/' in url_lower or '/newsroom/' in url_lower:
            return "high"
        
        # Other .edu pages are still high quality by default
        return "high"
    
    # LOW quality: Known unreliable or user-generated content sites
    low_quality_domains = [
        'alchetron.com',
        'prabook.com',
        'everipedia.org',
        'medium.com',  # Generic Medium posts (not official org Medium)
        'linkedin.com',  # LinkedIn is self-reported
        'facebook.com',
        'twitter.com',
        'reddit.com',
        'quora.com',
        'answers.com',
        'fandom.com',
        'wikia.com'
    ]
    
    if any(low_domain in domain_lower for low_domain in low_quality_domains):
        return "low"
    
    # MEDIUM quality: Wikipedia and other generally reliable sources
    medium_quality_domains = [
        'wikipedia.org',
        'en.wikipedia.org',
        'britannica.com',
        'scholar.google.com',
        'researchgate.net',
        'semanticscholar.org',
        'orcid.org'
    ]
    
    if any(med_domain in domain_lower for med_domain in medium_quality_domains):
        return "medium"
    
    # News organizations (generally medium-high quality)
    news_indicators = [
        'news', 'times', 'post', 'journal', 'tribune', 'herald',
        'reuters', 'ap.org', 'bbc.', 'npr.org', 'pbs.org',
        'nytimes', 'wsj.com', 'washingtonpost'
    ]
    
    if any(indicator in domain_lower or indicator in url_lower for indicator in news_indicators):
        return "medium"
    
    # Government and official organization sites
    if domain_lower.endswith('.gov') or domain_lower.endswith('.org'):
        return "medium"
    
    # Default: medium-low
    return "medium"


def _postprocess_decision(
    name: str,
    institution: str,
    results: List[Dict[str, Any]],
    decision: Dict[str, str],
) -> Dict[str, str]:
    """Improve temporal classification and adjust confidence using targeted cues.

    The logic here intentionally trusts the LLM output unless we find explicit,
    contradictory evidence in the supporting material.
    """
    try:
        if not decision:
            return decision

        verdict = decision.get("verdict", "uncertain")

        if verdict != "connected":
            if decision.get("confidence", "").lower() != "low":
                decision = dict(decision)
                decision["confidence"] = "low"
            return decision

        current_year = datetime.utcnow().year
        classification = decision.get("relationship_timeframe", "unknown")
        relationship_type_label = decision.get("relationship_type")
        supporting_url = decision.get("primary_source", "")

        if relationship_type_label in {"Alumni", "Attended"} and classification == "current":
            decision = dict(decision)
            decision["relationship_timeframe"] = "past"
            decision["current_or_past"] = "past"
            classification = "past"

        # Locate the supporting search result (if available) for richer context
        supporting_result: Optional[Dict[str, Any]] = None
        if supporting_url:
            for item in results or []:
                if _safe_text(item.get("url")) == supporting_url:
                    supporting_result = item
                    break

        # Assemble the evidence text we want to reason over
        evidence_parts: List[str] = [
            _safe_text(decision.get("verification_detail")),
            _safe_text(decision.get("temporal_context")),
            _safe_text(decision.get("summary")),
        ]
        if supporting_result:
            evidence_parts.extend(
                [
                    _safe_text(supporting_result.get("title")),
                    _safe_text(supporting_result.get("snippet")),
                    _safe_text(supporting_result.get("page_excerpt")),
                ]
            )

        evidence_text = " ".join(part for part in evidence_parts if part)
        evidence_text_lower = evidence_text.lower()

        # Extract any years mentioned across the evidence block
        evidence_years: List[int] = []
        for part in evidence_parts:
            if part:
                evidence_years.extend(_extract_years(part))
        latest_year = max(evidence_years) if evidence_years else None

        # Quick helpers for phrase detection (kept tight to avoid accidental matches)
        def _has_phrase(text: str, phrases: Iterable[str]) -> bool:
            return any(phrase in text for phrase in phrases)

        strong_current_markers = [
            " is currently ",
            " currently serves ",
            " currently is ",
            " currently works ",
            " currently teaches ",
            " currently leads ",
            " currently directs ",
            " currently at ",
            " currently holds ",
            " serves as ",
            " serving as ",
            " now at ",
            " now serves as ",
            " now works as ",
            " appointed as ",
            " is an assistant ",
            " is an associate ",
            " is a professor ",
            " is the professor ",
            " is a dean ",
            " is the dean ",
            " is a director ",
            " is the director ",
            " is a president ",
            " is the president ",
            " is a chair ",
            " is the chair ",
        ]

        strong_past_markers = [
            " former ",
            " formerly ",
            " previous ",
            " previously ",
            " retired ",
            " emeritus ",
            " was a ",
            " was an ",
            " was the ",
            " was professor ",
            " was president ",
            " was dean ",
            " was director ",
            " was chair ",
            " was faculty ",
            " was staff ",
            " was student ",
            " was member ",
            " served as ",
            " served from ",
            " served until ",
            " until ",
            " stepped down ",
            " left ",
            " departed ",
            " passed away ",
            " died ",
            " deceased ",
            " the late ",
        ]

        has_current_language = _has_phrase(evidence_text_lower, strong_current_markers)
        has_past_language = _has_phrase(evidence_text_lower, strong_past_markers)
        mentions_present = "present" in evidence_text_lower

        # Rule 1: downgrade "current" to "past" if we clearly see past-tense cues or old end dates
        if classification == "current":
            downgrade_reasons: List[str] = []

            if has_past_language:
                downgrade_reasons.append("past-tense language detected in supporting evidence")

            if latest_year and latest_year < current_year - 1 and not mentions_present:
                downgrade_reasons.append(f"latest year mention is {latest_year}")

            if downgrade_reasons:
                updated = dict(decision)
                updated["relationship_timeframe"] = "past"
                updated["current_or_past"] = "past"
                if updated.get("confidence", "").lower() == "high":
                    updated["confidence"] = "medium"
                note = "; ".join(downgrade_reasons)
                existing = _safe_text(updated.get("temporal_context"))
                merged = f"{existing}; {note}" if existing else note
                updated["temporal_context"] = merged
                updated["temporal_evidence"] = merged
                return updated

        # Rule 2: upgrade "past" to "current" only when evidence explicitly says so
        elif classification == "past" and relationship_type_label not in {"Alumni", "Attended"}:
            upgrade_reasons: List[str] = []

            if has_current_language or mentions_present:
                upgrade_reasons.append("explicit present-tense language detected")

            if latest_year and latest_year >= current_year - 1:
                upgrade_reasons.append(f"recent year mention {latest_year}")

            # Do not override if we simultaneously see strong past cues
            if upgrade_reasons and not has_past_language:
                updated = dict(decision)
                updated["relationship_timeframe"] = "current"
                updated["current_or_past"] = "current"
                existing = _safe_text(updated.get("temporal_context"))
                note = "; ".join(upgrade_reasons)
                merged = f"{existing}; {note}" if existing else note
                updated["temporal_context"] = merged
                updated["temporal_evidence"] = merged
                return updated

        # Adjust confidence based on supporting URL quality
        if supporting_url and decision.get("connected") == "Y":
            domain = _extract_domain(supporting_url)
            url_quality = _score_url_quality(supporting_url, domain)
            current_confidence = decision.get("confidence", "medium").lower()

            if url_quality == "low" and current_confidence in ["high", "medium"]:
                updated = dict(decision)
                updated["confidence"] = "low"
                return updated

            if url_quality == "medium" and current_confidence == "high":
                high_quality_count = sum(
                    1
                    for r in results or []
                    if _score_url_quality(r.get("url", ""), _extract_domain(r.get("url", ""))) == "high"
                )
                if high_quality_count < 2:
                    updated = dict(decision)
                    updated["confidence"] = "medium"
                    return updated

        return decision
    except Exception:
        return decision


def _auto_rescue_decision(
    decision: Dict[str, str],
    name: str,
    institution: str,
    results: List[Dict[str, Any]],
    *,
    vip_mode: bool = False,
) -> Optional[Dict[str, str]]:
    """If the LLM said no connection, try to rescue using deterministic evidence."""
    if not ALLOW_HEURISTIC_POSITIVE_RESCUE:
        return None
    if decision.get("connected") == "Y":
        return None
    if not _should_attempt_historical_llm_rescue(
        decision,
        institution,
        results,
        vip_mode=vip_mode,
    ):
        return None
    evidence = _find_search_evidence(name, institution, results)
    if not evidence:
        return None
    connection_type, detail, url, confidence = evidence
    raw_payload = {
        "verdict": "connected",
        "relationship_type": connection_type,
        "relationship_timeframe": "past",
        "verification_detail": detail or f"Auto-confirmed {connection_type.lower()} evidence",
        "summary": f"Deterministically confirmed {connection_type.lower()} link via search heuristic.",
        "primary_source": url,
        "confidence": confidence,
        "verification_status": "verified" if confidence == "high" else "needs_review",
        "temporal_context": detail,
    }
    return _normalize_decision(raw_payload)


def _validate_decision(
    decision: Dict[str, str],
    name: str,
    institution: str,
    results: Optional[List[Dict[str, Any]]] = None,
) -> bool:
    """Validate that the decision makes logical sense."""
    verdict = decision.get("verdict", "")
    if verdict not in {"connected", "not_connected", "uncertain"}:
        return False
    
    connected = decision.get("connected", "")
    if connected not in {"Y", "N"}:
        return False
    if verdict == "connected" and connected != "Y":
        return False
    if verdict != "connected" and connected != "N":
        return False

    relationship_timeframe = decision.get("relationship_timeframe", "")
    if relationship_timeframe not in {"current", "past", "unknown"}:
        return False

    confidence = decision.get("confidence", "")
    if confidence not in ["high", "medium", "low"]:
        return False

    relationship_type = decision.get("relationship_type", "")
    valid_relationship_types = [
        "Alumni",
        "Attended",
        "Executive",
        "Faculty",
        "Postdoc",
        "Staff",
        "Visiting",
        "Other",
        "None",
    ]
    legacy_type = decision.get("connection_type", "")

    if relationship_type not in valid_relationship_types:
        return False
    if verdict == "connected" and relationship_type == "None":
        return False
    # Alumni and Attended are valid connection types for "connected" verdict
    # Other types (Executive, Faculty, etc.) are also valid
    if verdict != "connected" and relationship_type not in {"None", "Other"}:
        return False
    if legacy_type and legacy_type not in [
        "Alumni",
        "Attended",
        "Executive",
        "Faculty",
        "Postdoc",
        "Staff",
        "Visiting",
        "Other",
        "Others",
    ]:
        return False
    
    verification_status = decision.get("verification_status", "")
    if verification_status not in {"verified", "needs_review"}:
        return False

    if verdict == "connected":
        detail = _safe_text(decision.get("verification_detail"))
        summary = _safe_text(decision.get("summary"))
        if len(detail) < 5 and len(summary) < 5:
            return False
        if "error" in detail.lower() or "error" in summary.lower():
            return False
        if relationship_timeframe not in {"current", "past", "unknown"}:
            return False
        if not _decision_mentions_target_person(decision, name, results):
            return False
        if not _decision_mentions_target_institution(decision, institution, results):
            return False
    else:
        if relationship_timeframe not in {"unknown"}:
            return False

    if verdict != "connected":
        temporal_context = _safe_text(decision.get("temporal_context"))
        if temporal_context and "error" not in temporal_context.lower():
            # For a "no" verdict, temporal context should usually be 'unknown' or contain rationale
            pass

    # Legacy checks for backward compatibility
    current_or_past = decision.get("current_or_past", "")
    if current_or_past not in ["current", "past", "N/A"]:
        return False
    
    return True


def _detect_false_positive_patterns(decision: Dict[str, str], results: List[Dict[str, Any]]) -> Optional[str]:
    """Detect common false positive patterns and downgrade confidence if necessary.
    
    Returns:
        Rejection reason if a false positive pattern is detected, None otherwise
    """
    if decision.get("connected") != "Y":
        return None  # Only check positive connections
    
    detail = _safe_text(decision.get("verification_detail") or decision.get("summary")).lower()
    connection_type = decision.get("relationship_type") or decision.get("connection_type", "")
    
    # Collect all text for pattern matching
    all_text = detail + " "
    for result in results or []:
        title = _safe_text(result.get("title", "")).lower()
        snippet = _safe_text(result.get("snippet", "")).lower()
        all_text += f" {title} {snippet}"

    context_warnings: List[str] = []
    critical_issues: List[str] = []
    
    # Pattern 1: Event participation (HIGHEST PRIORITY - very common false positive)
    event_terms = [
        "keynote", "speaker", "gave talk", "gave a talk", "presented at", "symposium",
        "conference", "workshop", "seminar", "guest lecture", "distinguished lecture",
        "axelrod", "lecture series", "colloquium", "invited talk", "talk at"
    ]
    for term in event_terms:
        if term in detail:
            if connection_type != "Visiting" and "visiting professor" not in detail and "visiting scholar" not in detail:
                critical_issues.append(f"Evidence references event participation term '{term}'")
        elif term in all_text:
            context_warnings.append(f"Search context mentions event participation term '{term}'")
    
    # Pattern 2: Press/publication prizes
    press_prize_terms = [
        "press prize", "press award", "prize winner", "prize from", "award from"
    ]
    for term in press_prize_terms:
        if term in detail:
            employment_terms = [
                "professor", "faculty", "staff", "employee", "worked", "teaches", 
                "graduated", "degree from", "studied", "distinguished", "emeritus", 
                "alumnus", "alumna", "alum"
            ]
            if not any(emp_term in detail for emp_term in employment_terms):
                critical_issues.append(f"Evidence only references prize term '{term}' without employment")
        elif term in all_text:
            context_warnings.append(f"Search context references prize term '{term}' without appearing in evidence detail")
    
    # Pattern 3: Publishing relationships
    publishing_terms = [
        "published by", "publisher", "editor for", "guest editor", "editorial board of",
        "edited by", "book series", "journal of"
    ]
    for term in publishing_terms:
        if term in detail:
            employment_terms = [
                "professor", "faculty", "staff", "employee", "worked", "teaches", 
                "graduated", "degree from", "studied", "distinguished", "emeritus",
                "alumnus", "alumna", "alum"
            ]
            if not any(emp_term in detail for emp_term in employment_terms):
                critical_issues.append(f"Evidence references publishing term '{term}' without employment markers")
        elif term in all_text:
            context_warnings.append(f"Publishing-related phrase '{term}' seen in search context")
    
    # Pattern 4: External advisory roles / External boards
    external_board_terms = [
        "board of trustees of", "board of directors of", "board member of",
        "advisory board for", "external review", "consultant for",
        "advisory committee for", "foundation board", "institute board"
    ]
    for term in external_board_terms:
        if term in detail:
            external_org_indicators = ["foundation", "institute at", "center at", "board of trustees of"]
            if any(indicator in detail for indicator in external_org_indicators):
                critical_issues.append(f"Evidence focuses on external board term '{term}'")
        elif term in all_text:
            context_warnings.append(f"External board phrase '{term}' detected in search context")
    
    # Pattern 5: Weak inference without explicit statement
    weak_inference_terms = [
        "connected to", "associated with", "linked to", "related to",
        "research group drawn from", "research group from", "team from",
        "drawn from researchers at"
    ]
    for term in weak_inference_terms:
        if term in detail:
            explicit_terms = [
                "professor at", "faculty at", "graduated from", "degree from",
                "earned", "received", "employed at", "works at", "president of",
                "dean of", "director of"
            ]
            if not any(explicit_term in detail for explicit_term in explicit_terms):
                critical_issues.append(f"Evidence uses weak phrasing '{term}' without explicit employment/degree")
        elif term in all_text:
            context_warnings.append(f"Weak phrasing '{term}' found in search context")
    
    # Pattern 6: Joint campus mentions (IUPUI/IUPFW)
    from .config import JOINT_CAMPUS_PATTERNS, ACCEPT_JOINT_CAMPUSES
    if not ACCEPT_JOINT_CAMPUSES:
        if any(pattern in detail for pattern in JOINT_CAMPUS_PATTERNS):
            if "iupui" in detail or "iupfw" in detail:
                critical_issues.append("Evidence references IUPUI/IUPFW joint campus rather than Purdue main campus")
        elif any(pattern in all_text for pattern in JOINT_CAMPUS_PATTERNS):
            context_warnings.append("Joint campus language detected in search context")
    
    # Pattern 7: Honorary degrees (without other connections)
    honorary_terms = ["honorary degree", "honorary doctorate", "honoris causa"]
    if any(term in detail for term in honorary_terms):
        genuine_terms = ["earned degree", "phd from", "bachelor from", "master from", "graduated", "professor", "faculty", "employed"]
        if not any(term in detail for term in genuine_terms):
            critical_issues.append("Evidence refers to honorary degree without earned degree/employment")
    elif any(term in all_text for term in honorary_terms):
        context_warnings.append("Honorary degree phrasing appears in search context")

    if critical_issues or context_warnings:
        confidence = decision.get("confidence", "medium").lower()
        issues_summary = []
        if critical_issues:
            issues_summary.extend(critical_issues)
            if confidence != "low":
                decision["confidence"] = "low"
            note = " | ".join(critical_issues)
            existing = _safe_text(decision.get("temporal_context"))
            merged = f"{existing}; Flagged by heuristics: {note}" if existing else f"Flagged by heuristics: {note}"
            decision["temporal_context"] = merged
            decision["temporal_evidence"] = merged
        elif context_warnings:
            issues_summary.extend(context_warnings)
            if confidence == "high":
                decision["confidence"] = "medium"
            note = " | ".join(context_warnings)
            existing = _safe_text(decision.get("temporal_context"))
            merged = f"{existing}; Heuristic warning: {note}" if existing else f"Heuristic warning: {note}"
            decision["temporal_context"] = merged
            decision["temporal_evidence"] = merged
        return "; ".join(dict.fromkeys(issues_summary))

    return None


async def analyze_connection(
    name: str,
    institution: str,
    results: List[Dict[str, Any]],
    *,
    debug: bool = False,
    max_retries: int = 3,
    per_attempt_timeout: float = 30.0,  # Reduced to 30s for faster failure detection (allows 3 retries within 180s)
    vip_mode: bool = False,
) -> Dict[str, str]:
    """Analyze connection with retries and per-attempt timeout.
    
    Args:
        name: Person's name
        institution: Institution to check
        results: Search results
        debug: Enable debug output
        max_retries: Maximum retry attempts (default 3)
        per_attempt_timeout: Maximum seconds per LLM API call (default 30s, increases with exponential backoff)
    """
    name = _safe_text(name)
    institution = _safe_text(institution)

    prompt = _build_prompt(name, institution, results or [], vip_mode=vip_mode)
    if debug or _PROMPT_LOGGING_ENABLED:
        separator = "=" * 48
        header = _console_safe(f"{separator}\n[LLM PROMPT] {name} <-> {institution}\n{separator}")
        print(header)
        print(_console_safe(prompt))
        print(f"{separator}\n[END PROMPT]\n{separator}")

    last_error: Optional[str] = None
    
    extra_transient_retries = 2
    max_total_attempts = max_retries + extra_transient_retries

    for attempt in range(1, max_total_attempts + 1):
        try:
            if debug and attempt > 1:
                print(f"[LLM] Retry attempt {attempt}/{max_total_attempts}")
            
            # Add per-attempt timeout with exponential backoff
            timeout = per_attempt_timeout * (1.5 ** (attempt - 1))  # Increase timeout on retries
            try:
                # Use temperature 0 for fully deterministic outputs
                parsed = await asyncio.wait_for(
                    _call_llm(prompt, debug=debug),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                if debug:
                    print(f"[LLM] Attempt {attempt} timed out after {timeout}s")
                if attempt < max_retries:
                    # Exponential backoff before retry
                    delay = min(2.0 * attempt, 5.0)
                    if debug:
                        print(f"[LLM] Waiting {delay}s before retry...")
                    await asyncio.sleep(delay)
                    await refresh_session()
                    continue
                raise RuntimeError(f"LLM call timed out after {timeout}s")
            
            decision = _normalize_decision(parsed)
            
            # Validate decision
            if not _validate_decision(decision, name, institution, results):
                if debug:
                    print(f"[LLM] Decision failed validation: {decision}")
                if attempt < max_retries:
                    delay = 1.0 * attempt
                    if debug:
                        print(f"[LLM] Waiting {delay}s before retry...")
                    await asyncio.sleep(delay)
                    await refresh_session()
                    continue
                # On last attempt, try to salvage if connected=Y
                if decision.get("connected") == "Y":
                    if debug:
                        print(f"[LLM] Using potentially invalid decision (last attempt, connected=Y)")
                else:
                    raise ValueError("Decision failed validation")
            
            # Post-process with relaxed override thresholds
            decision = _postprocess_decision(name, institution, results or [], decision)
            
            # Check for false positive patterns
            warning = _detect_false_positive_patterns(decision, results or [])
            if warning:
                if debug:
                    print(f"[LLM] {warning}")
                # Add warning to temporal_evidence for user visibility
                existing_evidence = decision.get("temporal_context", "")
                merged = f"{existing_evidence}; {warning}" if existing_evidence else warning
                decision["temporal_context"] = merged
                decision["temporal_evidence"] = merged
                # Reduce confidence if high
                if decision.get("confidence") == "high":
                    decision["confidence"] = "medium"
                    if debug:
                        print(f"[LLM] Reduced confidence to medium due to false positive pattern")
            
            # NEW: Confidence calibration for weak positives
            # If LLM says connected=Y but max_relevance_score < 10, downgrade confidence
            if decision.get("connected") == "Y" and results:
                max_relevance = max((r.get("signals", {}).get("relevance_score", 0) for r in results), default=0)
                if max_relevance < 10:
                    if debug:
                        print(f"[LLM] Weak positive detected: connected=Y but max_relevance_score={max_relevance} < 10")
                    # Downgrade confidence and mark for review
                    if decision.get("confidence") != "low":
                        decision["confidence"] = "low"
                        if debug:
                            print(f"[LLM] Downgraded confidence to low")
                    decision["verification_status"] = "needs_review"
                    # Add note to temporal_context
                    existing_evidence = decision.get("temporal_context", "")
                    note = f"Weak search signals (max relevance: {max_relevance})"
                    merged = f"{existing_evidence}; {note}" if existing_evidence else note
                    decision["temporal_context"] = merged
                    decision["temporal_evidence"] = merged

            if decision.get("connected") != "Y":
                rescued = _auto_rescue_decision(
                    decision,
                    name,
                    institution,
                    results or [],
                    vip_mode=vip_mode,
                )
                if rescued:
                    if debug:
                        print(f"[LLM] Auto-rescued connection using deterministic evidence: {rescued}")
                    return rescued
            
            if debug:
                print(f"[LLM] Analysis complete: {decision}")
            
            return decision
            
        except Exception as exc:
            last_error = str(exc) or f"{exc.__class__.__name__}: {exc!r}"
            if debug:
                print(f"[LLM] Attempt {attempt} failed: {exc}")

            retryable = isinstance(exc, RetryableLLMError)
            can_standard_retry = attempt < max_retries
            can_extra_transient_retry = retryable and attempt < max_total_attempts

            if can_standard_retry or can_extra_transient_retry:
                # Longer backoff for transient provider-side issues
                if retryable:
                    delay = min(2.0 * attempt, 6.0)
                else:
                    delay = min(1.5 * attempt, 4.0)
                if debug:
                    print(f"[LLM] Waiting {delay}s before retry...")
                await asyncio.sleep(delay)
                await refresh_session()
                continue
            break

    # Failed after retries
    error_result = _build_error(last_error or "LLM analysis failed")
    rescue_on_error = _auto_rescue_decision(
        {"connected": "N"},
        name,
        institution,
        results or [],
        vip_mode=vip_mode,
    )
    if rescue_on_error:
        if debug:
            print("[LLM] Auto-rescued connection after error using deterministic evidence")
        return rescue_on_error
    if debug:
        print(f"[LLM] All attempts failed, returning error result")
    return error_result
