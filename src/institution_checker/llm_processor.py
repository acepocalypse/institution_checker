from __future__ import annotations  # Increased to provide more context to LLMort annotations

import asyncio
import json
import re
import time
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlparse

import aiohttp

from .config import get_api_key, LLM_API_URL, MODEL_NAME

_session: Optional[aiohttp.ClientSession] = None
_session_lock = asyncio.Lock()
_last_refresh_time = 0.0

MAX_RESULTS_FOR_PROMPT = 20  # Use all search results to maximize evidence coverage

# STRICT EVIDENCE-BASED PROMPT - Balanced for speed and accuracy
PROMPT_TEMPLATE = """Verify if {name} has official connection to {institution}. 
READ ALL {max_results} RESULTS BELOW - evidence may be in any result.

Search results:
{search_findings}

REQUIRED EVIDENCE (need ≥1 for "Y"):
1. Official .edu page: faculty/dept directory, alumni registry, official bio
2. Explicit degree: "earned PhD from", "graduated from", "received BS from" 
3. Explicit title: "Professor at", "President of", "Dean of", "worked at"
4. Official CV/bio clearly stating role and institution

VALID EXAMPLES → Mark "Y":
• "earned his PhD from Stanford" → Alumni, past
• "is a Professor of Biology at MIT" → Faculty, current  
• "President of Harvard 2010-2015" → Executive, past
• "graduated with BS from Purdue" → Alumni, past

AUTO-REJECT → Mark "N":
❌ Events: keynote, Axelrod Lecturer, symposium, distinguished lecture, gave talk, invited speaker
❌ Press/prizes: "press prize winner", "press award", journal editor (without employment)
❌ External boards: "Board of [X] at [institution]" where X is external partner org
❌ Weak inference: "research group from", "associated with", "connected to" (without explicit role)
❌ Joint campus: IUPUI, IUPFW (unless policy allows)
❌ Honorary degrees alone (without earned degree or employment)

CONNECTION TYPES:
Alumni (earned degree), Attended (studied, no confirmed degree), Executive (President/Provost/Dean/Director),
Faculty (Professor/Instructor), Postdoc, Staff (employee/researcher), Visiting (formal appointment),
Other (verified connection), Others (only if connected="N")

TEMPORAL: Current = "is"/"serves"/present tense/{current_year}-{prior_year} | Past = "was"/"former"/end date/Alumni
CONFIDENCE: High = .edu + explicit | Medium = reliable source + explicit | Low = weak/N

JSON only (no markdown):
{{{{
  "connected": "Y" or "N",
  "connection_type": "Alumni|Attended|Executive|Faculty|Postdoc|Staff|Visiting|Other|Others",
  "connection_detail": "Quote explicit evidence",
  "current_or_past": "current|past|N/A",
  "supporting_url": "Best URL (prefer .edu)",
  "confidence": "high|medium|low",
  "temporal_evidence": "Dates/years or 'unknown'"
}}}}"""

DEFAULT_CLIENT_TIMEOUT = aiohttp.ClientTimeout(total=180, connect=15, sock_read=150)
JSON_BLOCK_RE = re.compile(r"```(?:json)?(.*?)```", re.DOTALL | re.IGNORECASE)
JSON_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")

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

# Add year pattern for temporal detection
YEAR_PATTERN = re.compile(r'\b(19|20)\d{2}\b')
# Pattern for date ranges like "2010-2015" or "2010 - 2015"
DATE_RANGE_PATTERN = re.compile(r'\b(\d{4})\s*[-–—]\s*(\d{4})\b')
# Pattern for "from X to Y"
FROM_TO_PATTERN = re.compile(r'\bfrom\s+(\d{4})\s+to\s+(\d{4})\b', re.IGNORECASE)
# Pattern for "since X" or "starting X"
SINCE_PATTERN = re.compile(r'\b(?:since|starting|began\s+in|started\s+in)\s+(\d{4})\b', re.IGNORECASE)
# Pattern for "until X" or "through X"
UNTIL_PATTERN = re.compile(r'\b(?:until|through|ended\s+in|retired\s+in|left\s+in)\s+(\d{4})\b', re.IGNORECASE)


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


def _normalize_connected(value: Any) -> str:
    text = _safe_text(value).upper()
    if text in {"Y", "YES"}:
        return "Y"
    if text in {"N", "NO"}:
        return "N"
    return "N"


def _normalize_connection_type(value: Any, connected: str) -> str:
    """Normalize connection type to valid options."""
    text = _safe_text(value).strip()
    # If not connected, must be "Others"
    if connected != "Y":
        return "Others"
    
    # Valid connection types (case-insensitive matching)
    valid_types = {
        "alumni": "Alumni",
        "attended": "Attended", 
        "executive": "Executive",
        "faculty": "Faculty",
        "postdoc": "Postdoc",
        "staff": "Staff",
        "visiting": "Visiting",
        "other": "Other"}
    
    text_lower = text.lower()
    if text_lower in valid_types:
        return valid_types[text_lower]
    
    # Try partial matching for common variations
    if "alum" in text_lower or "graduate" in text_lower:
        return "Alumni"
    if "student" in text_lower and "post" not in text_lower:
        return "Attended"
    if "prof" in text_lower or "teach" in text_lower or "lecturer" in text_lower or "instructor" in text_lower:
        return "Faculty"
    if "postdoc" in text_lower or "post-doc" in text_lower:
        return "Postdoc"
    if "president" in text_lower or "dean" in text_lower or "director" in text_lower or "admin" in text_lower or "chancellor" in text_lower:
        return "Executive"
    if "staff" in text_lower or "researcher" in text_lower or "scientist" in text_lower:
        return "Staff"
    if "visit" in text_lower:
        return "Visiting"
    
    # Default to "Other" for connected cases with unclear type
    return "Other"


def _normalize_current_or_past(value: Any) -> str:
    text = _safe_text(value).lower()
    if text in {"current", "present"}:
        return "current"
    if text in {"past", "former", "previous"}:
        return "past"
    return "N/A"


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
        "connected": "N",
        "connection_type": "Others",
        "connection_detail": f"Error: {message}",
        "current_or_past": "N/A",
        "supporting_url": "",
        "confidence": "low",
        "temporal_evidence": f"Processing error: {message}",
    }


def _normalize_decision(payload: Dict[str, Any]) -> Dict[str, str]:
    connected = _normalize_connected(payload.get("connected"))
    normalized = {
        "connected": connected,
        "connection_type": _normalize_connection_type(payload.get("connection_type"), connected),
        "connection_detail": _safe_text(payload.get("connection_detail")),
        "current_or_past": _normalize_current_or_past(payload.get("current_or_past")),
        "supporting_url": _safe_text(payload.get("supporting_url")),
        "confidence": _normalize_confidence(payload.get("confidence")),
        "temporal_evidence": _safe_text(payload.get("temporal_evidence")),
    }
    if normalized["connected"] != "Y":
        normalized["current_or_past"] = "N/A"
        normalized["confidence"] = _normalize_confidence("low")
        normalized["connection_type"] = "Others"
    return normalized


def _format_result_row(result: Dict[str, Any], index: int) -> str:
    """Format a search result for LLM consumption with clear structure."""
    title = _safe_text(result.get("title"))
    snippet = _safe_text(result.get("snippet"))
    url = _safe_text(result.get("url"))
    rank = result.get("rank")
    signals = result.get("signals", {})
    relevance = signals.get("relevance_score", 0)
    has_current = signals.get("has_current", False)
    has_past = signals.get("has_past", False)
    
    # Format with clear labels for better LLM parsing
    rank_prefix = f"#{rank} " if isinstance(rank, int) else ""
    lines = []
    
    if title:
        lines.append(f"Title: {rank_prefix}{title}")
    if snippet:
        lines.append(f"Description: {snippet}")
    if url:
        lines.append(f"URL: {url}")
    
    # Add temporal signals to help LLM identify current vs past
    if relevance > 0 or has_current or has_past:
        signal_info = f"Relevance: {relevance}"
        if has_current:
            signal_info += " [Contains CURRENT indicators]"
        if has_past:
            signal_info += " [Contains PAST indicators]"
        lines.append(signal_info)
    
    # Use newlines for better readability by LLM
    payload = "\n   ".join(lines)
    return f"{index + 1}. {payload}"


def _summarise_results(results: Iterable[Dict[str, Any]], limit: int = MAX_RESULTS_FOR_PROMPT) -> str:
    rows = []
    for idx, item in enumerate(results):
        if idx >= limit:
            break
        rows.append(_format_result_row(item, idx))
    return "\n".join(rows) if rows else "(no search results available)"


def _build_prompt(name: str, institution: str, results: List[Dict[str, Any]]) -> str:
    now = datetime.utcnow()
    findings = _summarise_results(results)
    return PROMPT_TEMPLATE.format(
        current_date=now.strftime("%B %d, %Y"),
        current_year=now.year,
        prior_year=now.year - 1,
        name=name,
        institution=institution,
        search_findings=findings,
        max_results=min(len(results), MAX_RESULTS_FOR_PROMPT),
    )


def _extract_fields_with_regex(text: str) -> Dict[str, Any]:
    """Extract fields using regex patterns as a fallback when JSON parsing fails."""
    result = {}
    
    # Pattern for each field - very flexible
    patterns = {
        "connected": r'"connected"\s*:\s*"([YN])"',
        "connection_type": r'"connection_type"\s*:\s*"(Alumni|Attended|Executive|Faculty|Postdoc|Staff|Other|Others)"',
        "connection_detail": r'"connection_detail"\s*:\s*"([^"]*(?:\\.[^"]*)*)"',
        "current_or_past": r'"current_or_past"\s*:\s*"(current|past|N/A)"',
        "supporting_url": r'"supporting_url"\s*:\s*"([^"]*)"',
        "confidence": r'"confidence"\s*:\s*"(high|medium|low)"',
        "temporal_evidence": r'"temporal_evidence"\s*:\s*"([^"]*(?:\\.[^"]*)*)"',
    }
    
    for field, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            value = match.group(1)
            # Unescape common patterns
            value = value.replace('\\"', '"').replace('\\n', ' ').replace('\\\\', '\\')
            result[field] = value
    
    # Set defaults for missing fields
    result.setdefault("connected", "N")
    result.setdefault("connection_type", "Others")
    result.setdefault("connection_detail", "Unable to determine")
    result.setdefault("current_or_past", "N/A")
    result.setdefault("supporting_url", "")
    result.setdefault("confidence", "low")
    result.setdefault("temporal_evidence", "Insufficient information")
    
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


def _parse_response(raw: str) -> Dict[str, Any]:
    """Parse LLM response with multiple fallback strategies."""
    # Strategy 1: Clean and extract JSON
    candidate = _clean_json_blob(raw)
    
    # Strategy 2: Sanitize before parsing
    sanitized = _sanitize_json_string(candidate)
    
    # Try parsing the sanitized version
    try:
        return json.loads(sanitized)
    except json.JSONDecodeError:
        pass
    
    # Strategy 3: Try the original candidate
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as first_error:
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
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # Strategy 5: Use regex extraction as last resort
    try:
        extracted = _extract_fields_with_regex(raw)
        if extracted.get("connected") in ["Y", "N"]:
            return extracted
    except Exception:
        pass
    
    # If all strategies fail, raise the original error
    raise json.JSONDecodeError(
        f"Could not parse response after trying multiple strategies. Raw: {raw[:200]}...",
        raw,
        0
    )


async def _call_llm(prompt: str, debug: bool = False, temperature: float = 0.2) -> Dict[str, Any]:
    """Call LLM with moderate temperature for consistent but flexible reasoning."""
    session = await get_session()
    headers = {
        "Authorization": f"Bearer {get_api_key()}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a JSON generator. Output ONLY valid JSON with no markdown or extra text. You are thorough and read all provided information before making decisions."},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "reasoning_effort": "medium",  # Changed from "low" to "medium" for better reasoning
        # No max_tokens - thinking models need unlimited tokens for reasoning + output
    }
    
    async with session.post(LLM_API_URL, json=payload, headers=headers) as response:
        text = await response.text()
        
        if debug:
            print(f"[LLM] Response status: {response.status}")
            print(f"[LLM] Response length: {len(text)} chars")
        
        if response.status >= 500:
            raise RuntimeError(f"LLM server error {response.status}: {text[:200]}")
        if response.status >= 400:
            raise RuntimeError(f"LLM request failed ({response.status}): {text[:200]}")
        
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            if debug:
                print(f"[LLM] Failed to parse server response as JSON")
            raise RuntimeError(f"Invalid JSON from server: {exc}") from exc
        
        choices = data.get("choices") or []
        if not choices:
            if debug:
                print(f"[LLM] No choices in response")
            raise RuntimeError("LLM response contained no choices")
        
        content = choices[0].get("message", {}).get("content", "")
        
        if debug:
            print(f"[LLM] Content length: {len(content)} chars")
            if len(content) < 200:
                print(f"[LLM] Full content: {content}")
            else:
                print(f"[LLM] Content preview: {content[:200]}...")
            
            # Check for truncation
            if not content.strip().endswith("}"):
                print(f"[LLM] ⚠️  WARNING: Response doesn't end with }}, likely TRUNCATED")
                print(f"[LLM] Last 100 chars: ...{content[-100:]}")
        
        if not content or not content.strip():
            raise RuntimeError("LLM returned empty content")
        
        return _parse_response(content)


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


def _contains_any(text: str, terms: Iterable[str]) -> bool:
    return any(term in text for term in terms)


def _institution_tokens(institution: str) -> List[str]:
    trimmed = (institution or "").strip().lower()
    if not trimmed:
        return []
    tokens = {trimmed}
    tokens.update([part for part in re.split(r"\s+", trimmed) if part])
    short = trimmed.replace("university", "").replace("college", "").strip()
    if short:
        tokens.add(short)
    tokens.discard("university")
    tokens.discard("college")
    return [token for token in tokens if token]


def _institution_domain_guess(institution: str) -> Optional[str]:
    trimmed = (institution or "").strip().lower()
    if not trimmed:
        return None
    # Extract first meaningful word
    words = [w for w in re.split(r'\s+', trimmed) if w and w not in {'the', 'of', 'at'}]
    if not words:
        return None
    primary = re.sub(r"[^a-z]", "", words[0])
    if not primary:
        return None
    return f"{primary}.edu"


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
    """Improve current/past classification and confidence using search result analysis.
    
    Now includes source quality assessment to adjust confidence.
    Note: Only override LLM with VERY STRONG evidence to maintain consistency.
    """
    try:
        # Force LOW confidence for all N decisions
        if decision and decision.get("connected") == "N":
            if decision.get("confidence", "").lower() != "low":
                decision = dict(decision)
                decision["confidence"] = "low"
        
        if not decision or decision.get("connected") != "Y":
            return decision

        tokens = _institution_tokens(institution)
        target_domain = _institution_domain_guess(institution)
        current_year = datetime.utcnow().year

        # Collect evidence from search results with enhanced date analysis
        target_current_count = 0
        target_past_count = 0
        other_current_edu_urls: List[str] = []
        target_urls_with_end_dates: List[str] = []
        target_recent_years: List[int] = []
        inferred_temporal_statuses: List[str] = []

        for item in results or []:
            title = _safe_text(item.get("title"))
            snippet = _safe_text(item.get("snippet"))
            url = _safe_text(item.get("url"))
            domain = _extract_domain(url)
            text = f"{title} {snippet}".lower()

            # Check if this result is about the target institution
            has_target = any(tok and tok in text for tok in tokens)
            if target_domain:
                domain_base = target_domain.replace('.edu', '')
                has_target = has_target or domain_base in domain
            
            if has_target:
                # Count temporal signals
                current_signals = _count_temporal_signals(text, CURRENT_TERMS)
                past_signals = _count_temporal_signals(text, PAST_TERMS)
                
                target_current_count += current_signals
                target_past_count += past_signals
                
                # Use enhanced date extraction
                inferred_status = _infer_temporal_status(text, current_year)
                if inferred_status:
                    inferred_temporal_statuses.append(inferred_status)
                
                # Check for date ranges indicating ended position
                if _has_end_date(text, current_year):
                    target_urls_with_end_dates.append(url)
                
                # Extract recent years
                years = _extract_years(text)
                target_recent_years.extend(y for y in years if y >= current_year - 2)
            else:
                # Check if person is currently at another institution
                is_edu = domain.endswith(".edu")
                if is_edu and _count_temporal_signals(text, CURRENT_TERMS) > 0:
                    other_current_edu_urls.append(url)

        # Decision logic: Use date-based inference + temporal signals
        current_classification = decision.get("current_or_past", "current")
        
        # Count inferred statuses
        inferred_current = inferred_temporal_statuses.count("current")
        inferred_past = inferred_temporal_statuses.count("past")
        
        # Rule 1: If LLM says current, but we have VERY STRONG past evidence
        if current_classification == "current":
            # Strong past evidence: multiple date-based indicators
            strong_past_evidence = (
                len(target_urls_with_end_dates) >= 2 or  # Multiple ended positions
                inferred_past >= 3 or  # Multiple results infer past
                (inferred_past >= 2 and len(other_current_edu_urls) >= 2)  # Past + elsewhere
            )
            
            if strong_past_evidence:
                updated = dict(decision)
                updated["current_or_past"] = "past"
                conf = updated.get("confidence", "medium").lower()
                if conf == "high":
                    updated["confidence"] = "medium"
                
                evidence = updated.get("temporal_evidence", "").strip()
                if len(target_urls_with_end_dates) >= 2:
                    extra = f"Position clearly ended based on dates in {len(target_urls_with_end_dates)} sources"
                elif inferred_past >= 3:
                    extra = f"Strong date-based evidence of past affiliation ({inferred_past} sources)"
                else:
                    extra = "Evidence shows past affiliation and current position elsewhere"
                
                updated["temporal_evidence"] = f"{evidence}; {extra}" if evidence else extra
                return updated
        
        # Rule 2: If LLM says past, but we have VERY STRONG current evidence
        elif current_classification == "past":
            # Strong current evidence: multiple recent indicators
            strong_current_evidence = (
                len(target_recent_years) >= 3 or  # Multiple recent year mentions
                inferred_current >= 3 or  # Multiple results infer current
                (inferred_current >= 2 and target_current_count >= 5)  # Inference + signals
            )
            
            if strong_current_evidence:
                updated = dict(decision)
                updated["current_or_past"] = "current"
                
                evidence = updated.get("temporal_evidence", "").strip()
                if len(target_recent_years) >= 3:
                    years_str = ", ".join(str(y) for y in sorted(set(target_recent_years))[:3])
                    extra = f"Strong recent activity found ({years_str})"
                elif inferred_current >= 3:
                    extra = f"Multiple sources ({inferred_current}) indicate current affiliation"
                else:
                    extra = f"Strong current indicators with recent dates"
                
                updated["temporal_evidence"] = f"{evidence}; {extra}" if evidence else extra
                return updated

        # NEW: Source quality-based confidence adjustment
        # Score the quality of the supporting_url
        supporting_url = decision.get("supporting_url", "")
        if supporting_url and decision.get("connected") == "Y":
            domain = _extract_domain(supporting_url)
            url_quality = _score_url_quality(supporting_url, domain)
            current_confidence = decision.get("confidence", "medium").lower()
            
            # Downgrade confidence based on source quality
            if url_quality == "low":
                # LOW quality sources: max confidence is "low"
                if current_confidence in ["high", "medium"]:
                    updated = dict(decision)
                    updated["confidence"] = "low"
                    return updated
            
            elif url_quality == "medium":
                # MEDIUM quality sources: max confidence is "medium"
                if current_confidence == "high":
                    # Check if we have multiple high-quality sources
                    high_quality_count = sum(
                        1 for r in results or []
                        if _score_url_quality(r.get("url", ""), _extract_domain(r.get("url", ""))) == "high"
                    )
                    # Only keep "high" if we have 2+ high-quality sources
                    if high_quality_count < 2:
                        updated = dict(decision)
                        updated["confidence"] = "medium"
                        return updated
            
            # If url_quality == "high" and it's a .edu, we can keep high confidence
            # But ONLY if the connection_detail has explicit evidence (checked by LLM prompt)

        return decision
    except Exception:
        return decision


def _validate_decision(decision: Dict[str, str], name: str, institution: str) -> bool:
    """Validate that the decision makes logical sense."""
    # Check required fields exist and have valid values
    connected = decision.get("connected", "")
    if connected not in ["Y", "N"]:
        return False
    
    current_or_past = decision.get("current_or_past", "")
    if current_or_past not in ["current", "past", "N/A"]:
        return False
    
    confidence = decision.get("confidence", "")
    if confidence not in ["high", "medium", "low"]:
        return False
    
    # Validate connection_type
    connection_type = decision.get("connection_type", "")
    valid_connection_types = ["Alumni", "Attended", "Executive", "Faculty", "Postdoc", "Staff", "Visiting", "Other", "Others"]
    if connection_type not in valid_connection_types:
        return False
    
    # If connected, must have valid connection type (not "Others")
    if connected == "Y" and connection_type == "Others":
        return False
    
    # If not connected, must be "Others"
    if connected == "N" and connection_type != "Others":
        return False
    
    # If connected, must have details (relaxed requirement)
    if connected == "Y":
        detail = decision.get("connection_detail", "").strip()
        if not detail or len(detail) < 5:  # Relaxed from 10 to 5
            return False
        if "error" in detail.lower():  # Removed "unable to determine" check - that can be valid
            return False
        # Must specify current or past
        if current_or_past == "N/A":
            return False
    
    # Relaxed validation for "N" responses - removed suspicious checks that may reject valid responses
    if connected == "N" and current_or_past != "N/A":
        return False
    
    return True


def _detect_false_positive_patterns(decision: Dict[str, str], results: List[Dict[str, Any]]) -> Optional[str]:
    """Detect common false positive patterns and AUTO-REJECT them.
    
    This function will CHANGE connected="Y" to "N" for obvious false positives.
    
    Returns:
        Rejection reason if a false positive pattern is detected, None otherwise
    """
    if decision.get("connected") != "Y":
        return None  # Only check positive connections
    
    detail = decision.get("connection_detail", "").lower()
    connection_type = decision.get("connection_type", "")
    
    # Collect all text for pattern matching
    all_text = detail + " "
    for result in results or []:
        title = _safe_text(result.get("title", "")).lower()
        snippet = _safe_text(result.get("snippet", "")).lower()
        all_text += f" {title} {snippet}"
    
    # Pattern 1: Event participation (HIGHEST PRIORITY - very common false positive)
    event_terms = [
        "keynote", "speaker", "gave talk", "gave a talk", "presented at", "symposium",
        "conference", "workshop", "seminar", "guest lecture", "distinguished lecture",
        "axelrod", "lecture series", "colloquium", "invited talk", "talk at"
    ]
    for term in event_terms:
        if term in detail or term in all_text:
            # Unless it's explicitly a Visiting appointment with formal term
            if connection_type != "Visiting" and "visiting professor" not in detail and "visiting scholar" not in detail:
                # AUTO-REJECT
                decision["connected"] = "N"
                decision["connection_type"] = "Others"
                decision["connection_detail"] = f"REJECTED: Event participation ({term}) is not an official connection"
                decision["confidence"] = "low"
                return f"AUTO-REJECTED: Event participation detected ({term})"
    
    # Pattern 2: Press/publication prizes
    press_prize_terms = [
        "press prize", "press award", "prize winner", "prize from", "award from"
    ]
    for term in press_prize_terms:
        if term in detail:
            # Check if it's ONLY about prize, not actual employment
            employment_terms = ["professor", "faculty", "staff", "employee", "worked", "teaches", "graduated", "degree from"]
            if not any(emp_term in detail for emp_term in employment_terms):
                # AUTO-REJECT
                decision["connected"] = "N"
                decision["connection_type"] = "Others"
                decision["connection_detail"] = f"REJECTED: Press/publication prize ({term}) is not an affiliation"
                decision["confidence"] = "low"
                return f"AUTO-REJECTED: Press prize detected ({term})"
    
    # Pattern 3: Publishing relationships
    publishing_terms = [
        "published by", "publisher", "editor for", "guest editor", "editorial board of",
        "edited by", "book series", "journal of"
    ]
    for term in publishing_terms:
        if term in detail:
            # Check if it's ONLY about publishing, not actual employment
            employment_terms = ["professor", "faculty", "staff", "employee", "worked", "teaches", "graduated", "degree from"]
            if not any(emp_term in detail for emp_term in employment_terms):
                # AUTO-REJECT
                decision["connected"] = "N"
                decision["connection_type"] = "Others"
                decision["connection_detail"] = f"REJECTED: Publishing relationship ({term}) is not an affiliation"
                decision["confidence"] = "low"
                return f"AUTO-REJECTED: Publishing relationship detected ({term})"
    
    # Pattern 4: External advisory roles / External boards
    external_board_terms = [
        "board of trustees of", "board of directors of", "board member of",
        "advisory board for", "external review", "consultant for",
        "advisory committee for", "foundation board", "institute board"
    ]
    for term in external_board_terms:
        if term in detail:
            # Check if it's an external organization's board (e.g., "Board of X at Purdue" where X is external org)
            external_org_indicators = ["foundation", "institute at", "center at", "board of trustees of"]
            if any(indicator in detail for indicator in external_org_indicators):
                # AUTO-REJECT
                decision["connected"] = "N"
                decision["connection_type"] = "Others"
                decision["connection_detail"] = f"REJECTED: External organization board ({term}) is not institutional employment"
                decision["confidence"] = "low"
                return f"AUTO-REJECTED: External board detected ({term})"
    
    # Pattern 5: Weak inference without explicit statement
    weak_inference_terms = [
        "connected to", "associated with", "linked to", "related to",
        "research group drawn from", "research group from", "team from",
        "drawn from researchers at"
    ]
    for term in weak_inference_terms:
        if term in detail:
            # Check if there's an explicit statement of employment or degree
            explicit_terms = [
                "professor at", "faculty at", "graduated from", "degree from",
                "earned", "received", "employed at", "works at", "president of",
                "dean of", "director of"
            ]
            if not any(explicit_term in detail for explicit_term in explicit_terms):
                # AUTO-REJECT
                decision["connected"] = "N"
                decision["connection_type"] = "Others"
                decision["connection_detail"] = f"REJECTED: Weak inference ({term}) without explicit employment/degree statement"
                decision["confidence"] = "low"
                return f"AUTO-REJECTED: Weak inference detected ({term})"
    
    # Pattern 6: Joint campus mentions (IUPUI/IUPFW)
    from .config import JOINT_CAMPUS_PATTERNS, ACCEPT_JOINT_CAMPUSES
    if not ACCEPT_JOINT_CAMPUSES:
        if any(pattern in all_text for pattern in JOINT_CAMPUS_PATTERNS):
            # Check if it's clearly IUPUI/IUPFW vs Purdue main
            if "iupui" in all_text or "iupfw" in all_text:
                # AUTO-REJECT
                decision["connected"] = "N"
                decision["connection_type"] = "Others"
                decision["connection_detail"] = "REJECTED: IUPUI/IUPFW is a joint IU-Purdue campus, not Purdue main campus"
                decision["confidence"] = "low"
                return "AUTO-REJECTED: Joint campus (IUPUI/IUPFW) detected"
    
    # Pattern 7: Honorary degrees (without other connections)
    honorary_terms = ["honorary degree", "honorary doctorate", "honoris causa"]
    if any(term in detail for term in honorary_terms):
        # Check if there's mention of actual study or employment
        genuine_terms = ["earned degree", "phd from", "bachelor from", "master from", "graduated", "professor", "faculty", "employed"]
        if not any(term in detail for term in genuine_terms):
            # AUTO-REJECT
            decision["connected"] = "N"
            decision["connection_type"] = "Others"
            decision["connection_detail"] = "REJECTED: Honorary degree alone is not an earned degree or employment"
            decision["confidence"] = "low"
            return "AUTO-REJECTED: Honorary degree only"
    
    return None


async def analyze_connection(
    name: str,
    institution: str,
    results: List[Dict[str, Any]],
    *,
    debug: bool = False,
    max_retries: int = 2,
    per_attempt_timeout: float = 45.0,  # Increased from 30s to 45s for thinking models
) -> Dict[str, str]:
    """Analyze connection with retries and per-attempt timeout.
    
    Args:
        name: Person's name
        institution: Institution to check
        results: Search results
        debug: Enable debug output
        max_retries: Maximum retry attempts
        per_attempt_timeout: Maximum seconds per LLM API call (default 45s)
    """
    name = _safe_text(name)
    institution = _safe_text(institution)
    prompt = _build_prompt(name, institution, results or [])

    last_error: Optional[str] = None
    
    for attempt in range(1, max_retries + 1):
        try:
            if debug and attempt > 1:
                print(f"[LLM] Retry attempt {attempt}/{max_retries}")
            
            # Add per-attempt timeout
            try:
                # Use temperature 0.2 for better consistency while allowing some flexibility
                parsed = await asyncio.wait_for(
                    _call_llm(prompt, debug=debug, temperature=0.2),
                    timeout=per_attempt_timeout
                )
            except asyncio.TimeoutError:
                if debug:
                    print(f"[LLM] Attempt {attempt} timed out after {per_attempt_timeout}s")
                raise RuntimeError(f"LLM call timed out after {per_attempt_timeout}s")
            
            decision = _normalize_decision(parsed)
            
            # Validate decision
            if not _validate_decision(decision, name, institution):
                if debug:
                    print(f"[LLM] Decision failed validation: {decision}")
                if attempt < max_retries:
                    await asyncio.sleep(1.0)
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
                existing_evidence = decision.get("temporal_evidence", "")
                decision["temporal_evidence"] = f"{existing_evidence}; {warning}" if existing_evidence else warning
                # Reduce confidence if high
                if decision.get("confidence") == "high":
                    decision["confidence"] = "medium"
                    if debug:
                        print(f"[LLM] Reduced confidence to medium due to false positive pattern")
            
            if debug:
                print(f"[LLM] Analysis complete: {decision}")
            
            return decision
            
        except Exception as exc:
            last_error = str(exc)
            if debug:
                print(f"[LLM] Attempt {attempt} failed: {exc}")
            
            if attempt < max_retries:
                await asyncio.sleep(1.5)
                await refresh_session()
                continue
            break

    # Failed after retries
    error_result = _build_error(last_error or "LLM analysis failed")
    if debug:
        print(f"[LLM] All attempts failed, returning error result")
    return error_result