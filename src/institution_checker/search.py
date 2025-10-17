import asyncio
import random
import re
import shutil
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional
from urllib.parse import quote, urlparse, urlsplit, urlunsplit, parse_qsl, urlencode

import httpx
from bs4 import BeautifulSoup

try:
    import ftfy
    FTFY_AVAILABLE = True
except ImportError:
    FTFY_AVAILABLE = False

from .config import BROWSER_ARGS, ACCEPT_JOINT_CAMPUSES, JOINT_CAMPUS_PATTERNS, CURRENT_TERMS, PAST_TERMS, _contains_any

try:
    from pyppeteer import launch
    from pyppeteer.errors import TimeoutError as PyppeteerTimeoutError
except ImportError:  # pragma: no cover - optional dependency
    launch = None
    PyppeteerTimeoutError = Exception

# Custom exception handler to only suppress pyppeteer protocol errors, not all exceptions
def _custom_exception_handler(loop, context):
    exception = context.get('exception')
    message = context.get('message', '')
    
    # Only suppress specific pyppeteer protocol errors
    if exception and 'pyppeteer' in str(type(exception)).lower():
        return
    if 'protocol' in message.lower() and 'error' in message.lower():
        return
    
    # For all other exceptions, use default handler to make them visible
    loop.default_exception_handler(context)

# Set the custom handler
asyncio.get_event_loop().set_exception_handler(_custom_exception_handler)


CHROME_PATH = (
    shutil.which("chrome")
    or shutil.which("msedge")
    or r"C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"
)

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36",
]

SELECTORS = [
    "li.b_algo",
    "li.b_ans",
    ".b_algo",
    ".b_entityTP",
    "[data-idx]",
]

# CURRENT_TERMS and PAST_TERMS are now imported from config.py
ACADEMIC_TERMS = [
    "assistant professor",
    "associate professor",
    "professor",
    "lecturer",
    "instructor",
    "faculty",
    "postdoctoral",
    "postdoc",
    "researcher",
    "scientist",
    "student",
    "doctoral",
    "phd candidate",
    "phd from",
    "earned degree",
    "bachelor",
    "master",
    "doctorate",
]
TRANSITION_TERMS = [
    "joined",
    "appointed",
    "moved to",
    "hired",
    "transitioned",
    "relocated",
    "left",
    "departed",
    "recently joined",
    "new position",
    "now at",
]
CV_TERMS = ["curriculum vitae", "cv", "resume", "biography", "bio", "profile"]
DIRECTORY_TERMS = ["faculty", "staff", "directory", "people", "department", "school of", "college of"]
PROFILE_SITES = ["linkedin.com", "researchgate.net", "orcid.org", "google.com/citations"]

_CURRENT_YEAR = datetime.utcnow().year
YEAR_WINDOW = range(_CURRENT_YEAR - 12, _CURRENT_YEAR + 1)
RECENT_YEAR_STRINGS = {str(_CURRENT_YEAR - offset) for offset in range(0, 3)}

BING_URL = "https://www.bing.com/search"
EXCERPT_FETCH_LIMIT = 4
EXCERPT_HTTP_TIMEOUT = 8.0  # seconds
EXCERPT_MAX_CHARS = 600
_NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v", "vi"}
_EXCERPT_SKIP_EXTENSIONS = (
    ".pdf",
    ".doc",
    ".docx",
    ".ppt",
    ".pptx",
    ".xls",
    ".xlsx",
    ".zip",
    ".rar",
    ".7z",
    ".mp3",
    ".mp4",
    ".mov",
    ".wmv",
    ".avi",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".ico",
)
_EXCERPT_EVIDENCE_HINTS = [
    "professor",
    "faculty",
    "graduate",
    "phd",
    "degree",
    "alumni",
    "staff",
    "postdoc",
    "director",
    "dean",
    "chair",
    "works at",
    "employed at",
    "currently",
]
MAX_HTTP_ATTEMPTS = 3
BROWSER_FETCH_ATTEMPTS = 2
CONSENT_SELECTORS = [
    "#bnp_btn_accept",
    "#bnp_btn_accept_all",
    "button#bnp_btn_accept",
    "button#bnp_btn_accept_all",
    'button[aria-label*="Accept"]',
    'button[aria-label*="accept"]',
]




@dataclass
class QueryStrategy:
    name: str
    query: str
    limit: int
    boost: int = 0
    ensure_tokens: bool = True


_http_client: Optional[httpx.AsyncClient] = None
_http_lock = asyncio.Lock()
_global_browser = None
_browser_lock = asyncio.Lock()
_browser_page_pool: List = []
_browser_page_lock = asyncio.Lock()
_BROWSER_PAGE_POOL_SIZE = 2
_FILTERED_RESOURCE_TYPES = {"image", "stylesheet", "font", "media"}


def _page_is_closed(page) -> bool:
    try:
        return page.isClosed()
    except Exception:
        return True


def _compose_query(*parts: str) -> str:
    return " ".join(part for part in parts if part).strip()


def _fix_text_encoding(text: str) -> str:
    """Fix mojibake and encoding errors in text using ftfy.
    
    Examples:
        "Câˆšâ‰¥rdova" -> "Córdova"
        "FranÃ§ois" -> "François"
    
    Args:
        text: Text that may contain encoding errors
        
    Returns:
        Fixed text with proper Unicode characters
    """
    if not text:
        return text
    
    # If ftfy is available, use it to fix encoding issues
    if FTFY_AVAILABLE:
        try:
            fixed = ftfy.fix_text(text)
            # Only return the fixed version if it's actually different and looks better
            # (ftfy is conservative and won't change text unless it's clearly broken)
            return fixed
        except Exception:
            # If ftfy fails, return original text
            return text
    
    # Fallback if ftfy is not available: just return the text as-is
    return text


def _normalise_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _remove_diacritics(text: str) -> str:
    """Remove diacritical marks from text for fuzzy matching."""
    # Normalize to NFD (decomposed form) then filter out combining characters
    nfd = unicodedata.normalize('NFD', text)
    return ''.join(char for char in nfd if unicodedata.category(char) != 'Mn')


def _normalize_name_for_matching(name: str) -> str:
    """Normalize a name for flexible matching: lowercase, no diacritics, no punctuation."""
    # First fix any encoding errors
    normalized = _fix_text_encoding(name)
    normalized = _normalise_whitespace(normalized).lower()
    # Remove diacritics
    normalized = _remove_diacritics(normalized)
    # Remove punctuation except spaces
    normalized = re.sub(r'[^\w\s]', '', normalized)
    # Collapse multiple spaces
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return normalized


def _extract_name_components(person_name: str) -> Dict[str, List[str]]:
    """Extract first, middle, and last name components with variations."""
    normalized = _normalize_name_for_matching(person_name)
    tokens = [token for token in normalized.split() if token and len(token) > 0]
    
    suffixes: List[str] = []
    while tokens and tokens[-1] in _NAME_SUFFIXES:
        suffixes.insert(0, tokens.pop())
    
    if not tokens:
        return {"first": [], "middle": [], "last": [], "initials": [], "suffix": suffixes}
    
    components = {
        "first": [tokens[0]] if len(tokens) > 0 else [],
        "middle": tokens[1:-1] if len(tokens) > 2 else [],
        "last": [tokens[-1]] if len(tokens) > 1 else [],
        "initials": [],
        "suffix": suffixes,
    }
    
    # Add middle initials
    for middle in components["middle"]:
        if middle:
            components["initials"].append(middle[0])
    
    # If only one token, it might be last name only
    if len(tokens) == 1:
        components["last"] = [tokens[0]]
        components["first"] = []
    
    return components


def _quote_clause(text: str) -> str:
    cleaned = _normalise_whitespace(text)
    if not cleaned:
        return ""
    if not (cleaned.startswith('"') and cleaned.endswith('"')):
        cleaned = f'"{cleaned}"'
    return cleaned


def _institution_tokens(institution: str) -> List[str]:
    trimmed = institution.strip().lower()
    if not trimmed:
        return []
    tokens = {trimmed}
    tokens.update(re.split(r"\s+", trimmed))
    tokens.update({trimmed.replace("university", "").strip(), trimmed.replace("college", "").strip()})
    tokens.discard("university")
    tokens.discard("college")
    return [token for token in tokens if token]


def _institution_domain_guess(institution: str) -> Optional[str]:
    trimmed = institution.strip().lower()
    if not trimmed:
        return None
    primary = re.sub(r"[^a-z]", "", trimmed.split()[0])
    if not primary:
        return None
    return f"{primary}.edu"


def _is_institution_domain(url: str, institution: str, signals: Optional[Dict[str, object]] = None) -> bool:
    if not url:
        return False
    domain = urlparse(url).netloc.lower()
    if not domain:
        return False
    if domain.endswith(".edu"):
        if not signals or signals.get("has_institution"):
            return True
    guess = _institution_domain_guess(institution)
    if guess and guess in domain:
        return True
    return False


def _ensure_name_and_institution(query: str, person_name: str, institution: str) -> str:
    parts: List[str] = []
    lowered = query.lower()
    if person_name:
        name_clause = _quote_clause(person_name)
        if name_clause and name_clause.lower() not in lowered:
            parts.append(name_clause)
    if institution:
        institution_clause = _quote_clause(institution)
        if institution_clause and institution_clause.lower() not in lowered:
            parts.append(institution_clause)
    if parts:
        query = _compose_query(*parts, query)
    return query


def _name_matches(text: str, person_name: str) -> bool:
    """Enhanced name matching with support for:
    - Middle name variations (John S. Smith vs John Samuel Smith)
    - Diacritics (Cordova with or without accent marks)
    - Different name orderings
    - Initials
    """
    # Normalize both text and name for comparison
    normalized_text = _normalize_name_for_matching(text)
    normalized_name = _normalize_name_for_matching(person_name)
    
    # Quick exact match
    if normalized_name in normalized_text:
        return True
    
    # Extract name components
    components = _extract_name_components(person_name)
    first_names = components["first"]
    middle_names = components["middle"]
    last_names = components["last"]
    middle_initials = components["initials"]
    suffixes = components.get("suffix", [])
    
    if not first_names or not last_names:
        # If we only have one name part, do simple matching
        return normalized_name in normalized_text
    
    def _token_in_text(token: str) -> bool:
        if not token:
            return False
        return bool(re.search(rf"\b{re.escape(token)}\b", normalized_text))
    
    candidate_firsts: List[str] = []
    candidate_firsts.extend(first_names)
    if first_names and len(first_names[0]) == 1 and middle_names:
        candidate_firsts.append(middle_names[0])
    if not candidate_firsts and middle_names:
        candidate_firsts.append(middle_names[0])
    candidate_firsts = [token for token in dict.fromkeys(candidate_firsts) if token]
    
    candidate_lasts: List[str] = [token for token in dict.fromkeys(last_names) if token]
    if not candidate_lasts and suffixes:
        candidate_lasts = [token for token in dict.fromkeys(suffixes) if token]
    
    if not candidate_lasts:
        return normalized_name in normalized_text
    
    if not any(_token_in_text(last_token) for last_token in candidate_lasts):
        return False
    if candidate_firsts and not any(_token_in_text(first_token) for first_token in candidate_firsts):
        return False
    
    # At this point, we have first/last presence.
    # Now check for common contiguous patterns to confirm high-confidence match.
    
    for first_token in candidate_firsts or [""]:
        for last_token in candidate_lasts or [""]:
            if not first_token or not last_token:
                continue
            
            # Pattern 1: First Middle Last
            for middle in middle_names:
                if not middle:
                    continue
                pattern = re.compile(
                    rf'\b{re.escape(first_token)}\s+{re.escape(middle)}\s+{re.escape(last_token)}\b'
                )
                if pattern.search(normalized_text):
                    return True
            
            # Pattern 2: First M. Last (middle initial with period)
            for initial in middle_initials:
                if not initial:
                    continue
                pattern = re.compile(
                    rf'\b{re.escape(first_token)}\s+{re.escape(initial)}\.?\s+{re.escape(last_token)}\b'
                )
                if pattern.search(normalized_text):
                    return True
            
            # Pattern 3: First Last (no middle)
            pattern = re.compile(rf'\b{re.escape(first_token)}\s+{re.escape(last_token)}\b')
            if pattern.search(normalized_text):
                return True
            
            # Pattern 4: Last, First (reversed with optional comma)
            pattern = re.compile(rf'\b{re.escape(last_token)}\s*,?\s+{re.escape(first_token)}\b')
            if pattern.search(normalized_text):
                return True
    
    # Fallback: first/last tokens both present somewhere in the text
    return True


def _name_matches_url(url: str, person_name: str) -> bool:
    """Detect whether the person's name is encoded in the URL path/netloc.

    Helps rescue high-value pages where search snippets omit the name but the
    page slug clearly targets the person (e.g., /people/john-doe).
    """
    if not url or not person_name:
        return False

    try:
        parsed = urlsplit(url)
        candidate = f"{parsed.netloc} {parsed.path}"
    except ValueError:
        candidate = url

    normalized_candidate = _normalize_name_for_matching(candidate)
    normalized_name = _normalize_name_for_matching(person_name)
    if normalized_name and normalized_name in normalized_candidate:
        return True

    components = _extract_name_components(person_name)
    first = components["first"][0] if components["first"] else ""
    last = components["last"][0] if components["last"] else ""

    # Require both tokens when possible to avoid noise from common surnames
    if first and last and first in normalized_candidate and last in normalized_candidate:
        return True

    # Fallback: accept distinctive last-name-only matches for .edu slugs, etc.
    if last and len(last) >= 4 and last in normalized_candidate:
        return True

    return False

def _normalise_url(raw_url: str) -> str:
    if not raw_url:
        return ""
    try:
        parsed = urlsplit(raw_url)
    except ValueError:
        return raw_url
    if parsed.scheme not in {"http", "https"}:
        return raw_url
    filtered_query = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=False)
        if not str(key).lower().startswith("utm_") and str(key).lower() not in {"ref", "referrer", "source"}
    ]
    normalized = urlunsplit(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path or "/",
            urlencode(filtered_query),
            "",
        )
    )
    return normalized.rstrip("/")


def _resolve_bing_redirect(url: str) -> str:
    """Resolve Bing redirect URLs to actual target URLs.
    
    Bing often uses redirect URLs like:
    - https://www.bing.com/ck/a?!&&p=...&u=<encoded_url>
    - Extract the actual target URL from the 'u' parameter
    """
    if not url:
        return url
    
    try:
        parsed = urlparse(url)
        
        # Check if this is a Bing redirect
        if 'bing.com' in parsed.netloc.lower() and '/ck/a' in parsed.path.lower():
            # Parse query parameters
            params = dict(parse_qsl(parsed.query))
            
            # Look for the 'u' parameter which contains the actual URL
            if 'u' in params:
                actual_url = params['u']
                # The URL might be base64 encoded or URL encoded
                # Try to decode and normalize
                try:
                    # First try URL decoding
                    from urllib.parse import unquote
                    decoded = unquote(actual_url)
                    
                    # Validate it's a proper URL
                    test_parsed = urlparse(decoded)
                    if test_parsed.scheme in {'http', 'https'} and test_parsed.netloc:
                        return _normalise_url(decoded)
                except Exception:
                    pass
        
        # If not a redirect or couldn't resolve, return normalized original
        return _normalise_url(url)
    except Exception:
        return url


def _is_joint_campus_mention(text: str) -> bool:
    """Check if text mentions a joint IU-Purdue campus (IUPUI, IUPFW, etc.)."""
    text_lower = text.lower()
    return any(pattern in text_lower for pattern in JOINT_CAMPUS_PATTERNS)


def _compute_signals(
    title: str,
    snippet: str,
    url: str,
    institution: str,
    person_name: str,
) -> Dict[str, object]:
    text = f"{title} {snippet}".lower()
    domain = urlparse(url).netloc.lower()

    has_institution = any(token in text for token in _institution_tokens(institution))
    has_person = bool(person_name and _name_matches(text, person_name))
    has_current = _contains_any(text, CURRENT_TERMS)
    has_past = _contains_any(text, PAST_TERMS)
    has_academic_role = _contains_any(text, ACADEMIC_TERMS)
    has_transition = _contains_any(text, TRANSITION_TERMS)
    has_recent_year = any(year in text for year in RECENT_YEAR_STRINGS)
    has_timeline = any(str(year) in text for year in YEAR_WINDOW)
    is_joint_campus = _is_joint_campus_mention(text)

    score = 0
    if has_person:
        score += 5
    if has_institution:
        score += 4
    if has_academic_role:
        score += 3
    if has_current:
        score += 4
    if has_past:
        score += 2
    if has_transition:
        score += 3
    if has_recent_year:
        score += 2
    if has_timeline:
        score += 1
    if domain.endswith(".edu"):
        score += 2
    if _contains_any(text, CV_TERMS):
        score += 1
    if _contains_any(text, DIRECTORY_TERMS):
        score += 1
    if any(site in domain for site in PROFILE_SITES):
        score += 1
    
    # Apply penalty for joint campus mentions if not accepting them
    if is_joint_campus and not ACCEPT_JOINT_CAMPUSES:
        score = max(0, score - 8)  # Heavy penalty to deprioritize

    return {
        "has_person_name": bool(has_person),
        "has_institution": bool(has_institution),
        "has_academic_role": bool(has_academic_role),
        "career_transition": bool(has_transition),
        "has_current": bool(has_current),
        "has_past": bool(has_past),
        "has_recent_year": bool(has_recent_year),
        "has_timeline": bool(has_timeline),
        "is_joint_campus": bool(is_joint_campus),
        "domain": domain,
        "relevance_score": score,
    }


def _ensure_person_signal(
    signals: Dict[str, object],
    url: str,
    institution: str,
    person_name: str,
) -> Dict[str, object]:
    """Confirm a result is person-specific, boosting signals when URL/domain imply it.
    
    We prefer to keep potentially relevant results so the LLM can decide. Instead of dropping
    uncertain matches, we keep them with a soft penalty while annotating confidence.
    """
    if not person_name:
        return signals
    if signals.get("has_person_name"):
        return signals

    updated = dict(signals)
    base_score = updated.get("relevance_score", 0)

    if _name_matches_url(url, person_name):
        updated["has_person_name"] = True
        updated["relevance_score"] = base_score + 3
        updated["person_match_confidence"] = "url"
        return updated

    if _is_institution_domain(url, institution, signals):
        updated["has_person_name"] = True
        updated["relevance_score"] = base_score + 2
        updated["person_match_confidence"] = "institution_domain"
        return updated

    # Soft keep: retain the result but note the uncertainty so downstream consumers can adjust.
    updated["person_match_confidence"] = "soft"
    if base_score <= 0:
        updated["relevance_score"] = 1
    return updated


async def _get_http_client() -> httpx.AsyncClient:
    global _http_client
    async with _http_lock:
        if _http_client is None:
            timeout = httpx.Timeout(10.0, connect=5.0, read=10.0)
            limits = httpx.Limits(max_connections=10, max_keepalive_connections=5)
            _http_client = httpx.AsyncClient(
                timeout=timeout,
                limits=limits,
                headers={"Accept-Language": "en-US,en;q=0.9"},
            )
        return _http_client


async def _fetch_with_httpx(query: str, count: int, debug: bool = False) -> str:
    """Fetch search results via HTTP with retry logic."""
    client = await _get_http_client()
    params = {"q": query, "count": str(count), "form": "QBRE"}
    
    for attempt in range(1, MAX_HTTP_ATTEMPTS + 1):
        try:
            response = await client.get(
                BING_URL,
                params=params,
                headers={"User-Agent": random.choice(USER_AGENTS)},
                timeout=15.0,  # Explicit timeout
            )
            response.raise_for_status()
            
            html = response.text
            if html and len(html) > 100:  # Basic validation
                if debug:
                    print(f"[search] HTTP fetch successful on attempt {attempt}")
                return html
            else:
                if debug:
                    print(f"[search] HTTP attempt {attempt}: empty or too short response")
                
        except httpx.TimeoutException as exc:
            if debug:
                print(f"[search] HTTP attempt {attempt} timed out: {exc}")
        except httpx.HTTPStatusError as exc:
            if debug:
                print(f"[search] HTTP attempt {attempt} status error {exc.response.status_code}")
            # Don't retry on 4xx errors (except 429)
            if 400 <= exc.response.status_code < 500 and exc.response.status_code != 429:
                return ""
        except httpx.HTTPError as exc:
            if debug:
                print(f"[search] HTTP attempt {attempt} failed: {exc}")
        
        if attempt < MAX_HTTP_ATTEMPTS:
            delay = 0.5 * (1.5 ** (attempt - 1))  # Exponential backoff
            if debug:
                print(f"[search] Waiting {delay:.1f}s before retry...")
            await asyncio.sleep(delay)
    
        if debug:
            print(f"[search] HTTP fetch failed after {MAX_HTTP_ATTEMPTS} attempts")
    return ""


def _should_fetch_page_excerpt(result: Dict[str, object]) -> bool:
    """Determine if a result needs extra on-page evidence fetched."""
    url = _normalise_whitespace(str(result.get("url") or ""))
    if not url or not url.startswith(("http://", "https://")):
        return False
    lowered = url.lower()
    if any(lowered.endswith(ext) for ext in _EXCERPT_SKIP_EXTENSIONS):
        return False

    if result.get("page_excerpt"):
        return False

    signals = result.get("signals") or {}
    score = signals.get("relevance_score", 0)
    has_person = bool(signals.get("has_person_name"))
    has_institution = bool(signals.get("has_institution"))
    match_confidence = signals.get("person_match_confidence")

    # Bail out only for extremely weak/irrelevant results
    if score <= 0 and not has_person and not has_institution:
        return False

    # Prefer to enrich soft matches or results missing clear evidence
    if match_confidence == "soft":
        return True

    snippet = _normalise_whitespace(str(result.get("snippet") or ""))
    if not snippet or len(snippet) < 80:
        return True

    snippet_lower = snippet.lower()
    evidence_terms_present = any(term in snippet_lower for term in _EXCERPT_EVIDENCE_HINTS)
    if not evidence_terms_present:
        return True

    # If both person and institution already identified and snippet contains explicit evidence, skip fetch
    if has_person and has_institution:
        return False

    return True


async def _get_page_excerpt(url: str, person_name: str) -> Optional[str]:
    """Fetch a short excerpt from the page, favouring sentences that mention the person."""
    if not url:
        return None

    client = await _get_http_client()
    try:
        response = await client.get(
            url,
            headers={"User-Agent": random.choice(USER_AGENTS)},
            timeout=EXCERPT_HTTP_TIMEOUT,
            follow_redirects=True,
        )
        response.raise_for_status()
    except (httpx.HTTPError, httpx.TimeoutException):
        return None

    content_type = (response.headers.get("Content-Type") or "").lower()
    binary_markers = ["pdf", "image", "audio", "video", "zip", "octet", "msword", "excel", "powerpoint"]
    if any(marker in content_type for marker in binary_markers):
        return None

    html = response.text
    if not html or len(html) < 120:
        return None

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()

    text = _normalise_whitespace(soup.get_text(" ", strip=True))
    if not text or len(text) < 60:
        return None

    normalized_target = _normalize_name_for_matching(person_name) if person_name else ""
    if normalized_target:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        for sentence in sentences:
            if normalized_target in _normalize_name_for_matching(sentence):
                excerpt = sentence.strip()
                if len(excerpt) > EXCERPT_MAX_CHARS:
                    excerpt = excerpt[:EXCERPT_MAX_CHARS].rstrip() + "..."
                return excerpt

    excerpt = text[:EXCERPT_MAX_CHARS].strip()
    if len(text) > EXCERPT_MAX_CHARS:
        excerpt += "..."
    return excerpt


async def _attach_page_excerpt(result: Dict[str, object], person_name: str) -> None:
    try:
        excerpt = await _get_page_excerpt(str(result.get("url") or ""), person_name)
    except Exception:
        return
    if excerpt:
        result["page_excerpt"] = excerpt


async def _enrich_with_page_excerpts(
    results: Iterable[Dict[str, object]],
    person_name: str,
    limit: int = EXCERPT_FETCH_LIMIT,
) -> None:
    if not results or not person_name or limit <= 0:
        return

    selected: List[Dict[str, object]] = []
    seen_urls = set()
    for item in results:
        if len(selected) >= limit:
            break
        url = str(item.get("url") or "")
        if not url or url in seen_urls:
            continue
        if _should_fetch_page_excerpt(item):
            selected.append(item)
            seen_urls.add(url)

    if not selected:
        return

    tasks = [asyncio.create_task(_attach_page_excerpt(item, person_name)) for item in selected]
    await asyncio.gather(*tasks, return_exceptions=True)


async def _get_browser():
    if launch is None or CHROME_PATH is None:
        return None
    global _global_browser
    async with _browser_lock:
        if _global_browser is None or _global_browser.process.returncode is not None:
            _global_browser = await launch(
                headless=True,
                executablePath=CHROME_PATH,
                args=BROWSER_ARGS + [
                    "--disable-web-security",
                    "--disable-features=VizDisplayCompositor",
                    "--no-first-run",
                    "--disable-default-apps",
                    "--disable-sync",
                    "--memory-pressure-off",
                ],
            )
        return _global_browser


async def _dismiss_bing_consent(page, debug: bool = False) -> bool:
    selectors = CONSENT_SELECTORS + [
        "a#bnp_btn_accept",
        "a#bnp_btn_accept_all",
    ]
    frames = [page]
    try:
        frames.extend(page.frames)
    except Exception:
        pass
    for frame in frames:
        for selector in selectors:
            try:
                element = await frame.querySelector(selector)
            except Exception:
                continue
            if element:
                try:
                    await element.click()
                    if debug:
                        print(f"[search] Dismissed consent prompt via selector {selector!r}")
                    await asyncio.sleep(0.6)
                    return True
                except Exception as exc:
                    if debug:
                        print(f"[search] Consent click failed for selector {selector!r}: {exc}")
    return False


async def _configure_browser_page(page) -> None:
    if getattr(page, "_ic_configured", False):
        return
    try:
        await page.setRequestInterception(True)
    except Exception:
        setattr(page, "_ic_configured", True)
        return

    async def _handle_request(request):
        try:
            if request.resourceType in _FILTERED_RESOURCE_TYPES:
                await request.abort()
            else:
                await request.continue_()
        except Exception:
            pass

    page.on("request", lambda request: asyncio.create_task(_handle_request(request)))
    setattr(page, "_ic_configured", True)


async def _ensure_page_headers(page) -> None:
    if getattr(page, "_ic_headers_set", False):
        return
    try:
        await page.setExtraHTTPHeaders({"Accept-Language": "en-US,en;q=0.9"})
    except Exception:
        pass
    setattr(page, "_ic_headers_set", True)


async def _release_browser_page(page) -> None:
    if not page or _page_is_closed(page):
        return
    async with _browser_page_lock:
        if len(_browser_page_pool) < _BROWSER_PAGE_POOL_SIZE:
            _browser_page_pool.append(page)
        else:
            try:
                await page.close()
            except Exception:
                pass


async def _acquire_browser_page():
    browser = await _get_browser()
    if browser is None:
        return None, None
    while True:
        async with _browser_page_lock:
            page = _browser_page_pool.pop() if _browser_page_pool else None
        if page is None:
            break
        if _page_is_closed(page):
            continue
        await _configure_browser_page(page)
        return page, _release_browser_page
    page = await browser.newPage()
    await _configure_browser_page(page)
    return page, _release_browser_page


async def _fetch_with_browser(query: str, count: int, debug: bool = False) -> str:
    page = None
    release_page = None
    result_html = ""
    target_url = f"{BING_URL}?q={quote(query)}&count={count}&form=QBRE"
    try:
        for attempt in range(1, BROWSER_FETCH_ATTEMPTS + 1):
            if page is None:
                page, release_page = await _acquire_browser_page()
                if page is None:
                    if debug:
                        print("[search] Pyppeteer browser not available")
                    break
            try:
                await _ensure_page_headers(page)
                user_agent = random.choice(USER_AGENTS)
                if getattr(page, "_ic_user_agent", None) != user_agent:
                    await page.setUserAgent(user_agent)
                    page._ic_user_agent = user_agent
                await page.goto(target_url, timeout=20000, waitUntil="domcontentloaded")
                await asyncio.sleep(0.25)
                await _dismiss_bing_consent(page, debug=debug)
                result_ready = False
                for selector in SELECTORS:
                    try:
                        await page.waitForSelector(selector, {"timeout": 6000})
                        result_ready = True
                        break
                    except PyppeteerTimeoutError:
                        continue
                    except Exception:
                        continue
                if not result_ready and debug:
                    print(f"[search] Browser attempt {attempt}: no result selectors found")
                html = await page.content()
                if html:
                    result_html = html
                    break
            except Exception as exc:
                if debug:
                    print(f"[search] Browser attempt {attempt} failed: {exc}")
                try:
                    await page.close()
                except Exception:
                    pass
                page = None
                release_page = None
            if attempt < BROWSER_FETCH_ATTEMPTS:
                await asyncio.sleep(0.5 * attempt)
    finally:
        if release_page and page and not _page_is_closed(page):
            await release_page(page)
    return result_html


def _build_extended_snippet(element) -> str:
    """Extract a richer snippet from the search result element without visiting the page."""
    seen_lower = set()
    parts: List[str] = []

    def _add_text(raw: str) -> None:
        text = _normalise_whitespace(raw)
        if not text:
            return
        lowered = text.lower()
        if lowered in seen_lower:
            return
        seen_lower.add(lowered)
        parts.append(text)

    candidate_selectors = [
        ".b_caption p",
        ".b_snippet",
        ".b_secondaryText",
        ".b_factrow",
        ".b_paractl",
        ".b_lineclamp",
        ".b_text",
    ]
    for selector in candidate_selectors:
        for node in element.select(selector):
            _add_text(node.get_text(" ", strip=True))

    # Capture bullet lists or fact rows for additional context.
    for node in element.select(".b_factrow li, ul li"):
        _add_text(node.get_text(" ", strip=True))

    if not parts:
        for node in element.find_all("p", limit=2):
            _add_text(node.get_text(" ", strip=True))

    if not parts:
        fallback_text = element.get_text(" ", strip=True)
        _add_text(fallback_text)

    combined = " ".join(parts)
    if len(combined) > 800:
        combined = combined[:797].rstrip() + "..."
    return combined


def _parse_result_element(
    element,
    institution: str,
    person_name: str,
) -> Optional[Dict[str, object]]:
    anchor = element.select_one("h2 a[href]") or element.select_one("a[href]")
    if not anchor:
        return None
    url = anchor.get("href", "").strip()
    if not url.startswith(("http://", "https://")):
        return None
    
    # Resolve Bing redirects to get actual target URL
    url = _resolve_bing_redirect(url)
    url = _normalise_url(url)
    
    title = _normalise_whitespace(anchor.get_text(" ", strip=True))
    if not title:
        return None
    snippet = _build_extended_snippet(element)
    signals = _compute_signals(title, snippet, url, institution, person_name)
    signals = _ensure_person_signal(signals, url, institution, person_name)
    return {
        "title": title,
        "url": url,
        "snippet": snippet,
        "signals": signals,
    }


def _extract_results(
    html: str,
    institution: str,
    limit: int,
    person_name: str,
    debug: bool = False,
) -> List[Dict[str, object]]:
    soup = BeautifulSoup(html, "html.parser")
    seen_elements = set()
    candidates = []
    for selector in SELECTORS:
        matches = soup.select(selector)
        if debug:
            print(f"[search] Selector {selector}: {len(matches)} matches")
        for element in matches:
            if id(element) in seen_elements:
                continue
            seen_elements.add(id(element))
            candidates.append(element)
    results: List[Dict[str, object]] = []
    seen_urls = set()
    rank_counter = 0
    for element in candidates:
        parsed = _parse_result_element(element, institution, person_name)
        if not parsed:
            continue
        url = parsed["url"]
        if url in seen_urls:
            continue
        rank_counter += 1
        seen_urls.add(url)
        signals = parsed.get("signals", {})
        if "source_rank" not in signals:
            signals["source_rank"] = rank_counter
        parsed["signals"] = signals
        parsed["rank"] = rank_counter
        results.append(parsed)
        if len(results) >= limit:
            break
    if not results:  # fallback to generic anchors if structured parsing failed
        for anchor in soup.select("a[href]"):
            url = anchor.get("href", "").strip()
            if not url.startswith(("http://", "https://")):
                continue
            
            # Resolve Bing redirects
            url = _resolve_bing_redirect(url)
            url = _normalise_url(url)
            
            if url in seen_urls:
                continue
            rank_counter += 1
            title = _normalise_whitespace(anchor.get_text(" ", strip=True))
            if len(title) < 8:
                continue
            snippet = ""
            signals = _compute_signals(title, snippet, url, institution, person_name)
            signals = _ensure_person_signal(signals, url, institution, person_name)
            signals.setdefault("source_rank", rank_counter)
            results.append({
                "title": title,
                "url": url,
                "snippet": snippet,
                "signals": signals,
                "rank": rank_counter,
            })
            seen_urls.add(url)
            if len(results) >= limit:
                break
    return results


async def bing_search(
    query: str,
    num_results: int = 20,
    institution: str = "",
    person_name: str = "",
    debug: bool = False,
) -> List[Dict[str, object]]:
    query = _ensure_name_and_institution(query, person_name, institution)
    target = max(num_results, 10)
    fetch_count = min(target * 2, 50)
    html = await _fetch_with_browser(query, count=fetch_count, debug=debug)
    if not html:
        if debug:
            print("[search] Browser fetch returned no content, falling back to HTTP client")
        html = await _fetch_with_httpx(query, count=fetch_count, debug=debug)
    if not html:
        return []
    results = _extract_results(
        html,
        institution=institution,
        limit=num_results * 2,
        person_name=person_name,
        debug=debug,
    )
    results.sort(key=lambda item: item.get("rank") or item["signals"].get("source_rank", float("inf")))
    top_results = results[:num_results]
    await _enrich_with_page_excerpts(top_results, person_name, limit=min(EXCERPT_FETCH_LIMIT, len(top_results)))
    return top_results


def _build_strategies(
    name: str,
    institution: str,
    per_strategy_limit: int,
) -> List[QueryStrategy]:
    clean_name = _normalise_whitespace(name)
    clean_institution = _normalise_whitespace(institution)
    if not clean_name:
        return []

    name_clause = _quote_clause(clean_name)
    institution_clause = _quote_clause(clean_institution)
    institution_fragment = institution_clause or clean_institution

    # Increased limits to ensure we capture true positives
    base_limit = max(4, min(6, per_strategy_limit))

    strategies: List[QueryStrategy] = [
        # HIGHEST PRIORITY: Name + Institution combined (ensures both appear together)
        QueryStrategy(
            name="name_institution_combined",
            query=f"{name_clause} {institution_clause}",  # Both terms together
            limit=base_limit + 3,  # Highest limit
            boost=6,  # Highest boost - these are most relevant
        ),
        # Connection-focused search (looking for explicit relationships)
        QueryStrategy(
            name="explicit_connection",
            query=_compose_query(
                name_clause,
                institution_fragment,
                '("professor at" OR "faculty" OR "graduated from" OR "degree from" OR "alumnus" OR "worked at")',
            ),
            limit=base_limit + 2,
            boost=5,
        ),
        # Core profile search (general biographical info)
        QueryStrategy(
            name="core_profile",
            query=_compose_query(name_clause, institution_fragment),
            limit=base_limit + 1,
            boost=4,
        ),
        # Current status (high value for current vs past determination)
        QueryStrategy(
            name="current_status",
            query=_compose_query(
                name_clause,
                institution_fragment,
                '("currently" OR "now" OR "presently")',
            ),
            limit=base_limit,
            boost=3,
        ),
        # Career history (CV/bio pages are very informative)
        QueryStrategy(
            name="career_timeline",
            query=_compose_query(
                name_clause,
                institution_fragment,
                '("curriculum vitae" OR "cv" OR "biography")',
            ),
            limit=base_limit,
            boost=3,
        ),
        # Education/degree search (important for alumni connections)
        QueryStrategy(
            name="education",
            query=_compose_query(
                name_clause,
                institution_fragment,
                '("degree" OR "graduated" OR "alumni" OR "PhD" OR "bachelor")',
            ),
            limit=base_limit,
            boost=3,
        ),
    ]
    
    # Add directory search only if we have institution domain
    domain_guess = _institution_domain_guess(clean_institution)
    if domain_guess:
        site_filter = f"site:{domain_guess}"
        strategies.append(
            QueryStrategy(
                name="directory",
                query=_compose_query(
                    name_clause,
                    institution_fragment,
                    site_filter,
                    '("faculty" OR "staff" OR "directory")',
                ),
                limit=base_limit + 2,  # Higher for official pages
                boost=5,  # Very high boost for official domain
            )
        )

    for index, strategy in enumerate(strategies):
        setattr(strategy, "order", index)

    return strategies


def _annotate_signals(
    result: Dict[str, object],
    strategy: QueryStrategy,
    effective_query: str,
) -> Dict[str, object]:
    signals = dict(result["signals"])
    if "source_rank" not in signals and isinstance(result.get("rank"), int):
        signals["source_rank"] = result["rank"]
    signals["strategy"] = strategy.name
    strategy_order = getattr(strategy, "order", None)
    if strategy_order is not None:
        signals["strategy_order"] = strategy_order
    signals["strategy_boost"] = strategy.boost
    signals["queries"] = [effective_query]
    signals["strategies"] = [strategy.name]
    signals["strategy_hits"] = 1
    return {
        "title": result.get("title", ""),
        "url": result.get("url", ""),
        "snippet": result.get("snippet", ""),
        "signals": signals,
        "rank": result.get("rank"),
    }


def _merge_results(existing: Dict[str, object], candidate: Dict[str, object]) -> Dict[str, object]:
    merged = dict(existing)
    merged_signals = dict(existing.get("signals", {}))
    candidate_signals = candidate.get("signals", {})

    existing_strategies = set(merged_signals.get("strategies", []))
    if merged_signals.get("strategy"):
        existing_strategies.add(merged_signals["strategy"])
    for value in candidate_signals.get("strategies", []):
        existing_strategies.add(value)
    if candidate_signals.get("strategy"):
        existing_strategies.add(candidate_signals["strategy"])
    merged_signals["strategies"] = sorted(existing_strategies)

    existing_queries = list(merged_signals.get("queries", []))
    if candidate_signals.get("queries"):
        existing_queries.extend(candidate_signals["queries"])
    merged_signals["queries"] = sorted({query for query in existing_queries if query})

    for key in (
        "has_person_name",
        "has_institution",
        "has_academic_role",
        "career_transition",
        "has_current",
        "has_past",
        "has_recent_year",
        "has_timeline",
    ):
        merged_signals[key] = bool(merged_signals.get(key) or candidate_signals.get(key))

    existing_score = merged_signals.get("relevance_score", 0)
    candidate_score = candidate_signals.get("relevance_score", 0)
    merged_signals["relevance_score"] = max(existing_score, candidate_score)

    existing_boost = merged_signals.get("strategy_boost", 0)
    candidate_boost = candidate_signals.get("strategy_boost", 0)
    merged_signals["strategy_boost"] = max(existing_boost, candidate_boost)
    merged_signals["strategy_hits"] = len(merged_signals["strategies"])

    existing_rank_value = merged_signals.get("source_rank")
    candidate_rank_value = candidate_signals.get("source_rank")
    if candidate_rank_value is not None:
        if existing_rank_value is None or candidate_rank_value < existing_rank_value:
            merged_signals["source_rank"] = candidate_rank_value

    existing_rank_attr = merged.get("rank")
    candidate_rank_attr = candidate.get("rank")
    if isinstance(candidate_rank_attr, int):
        if not isinstance(existing_rank_attr, int) or candidate_rank_attr < existing_rank_attr:
            merged["rank"] = candidate_rank_attr

    existing_order = merged_signals.get("strategy_order")
    candidate_order = candidate_signals.get("strategy_order")
    if candidate_order is not None:
        if existing_order is None or candidate_order < existing_order:
            merged_signals["strategy_order"] = candidate_order

    # Preserve highest-confidence person match annotation
    candidate_confidence = candidate_signals.get("person_match_confidence")
    if candidate_confidence:
        existing_confidence = merged_signals.get("person_match_confidence")
        if not existing_confidence or existing_confidence == "soft":
            merged_signals["person_match_confidence"] = candidate_confidence

    if len(candidate.get("snippet", "")) > len(merged.get("snippet", "")):
        merged["snippet"] = candidate.get("snippet", "")
    if candidate_score >= existing_score:
        merged["title"] = candidate.get("title", merged.get("title", ""))
        if candidate_signals.get("strategy"):
            merged_signals["strategy"] = candidate_signals["strategy"]
        if candidate_boost > existing_boost:
            merged_signals["strategy_boost"] = candidate_boost

    merged["signals"] = merged_signals
    return merged


async def _execute_strategy(
    strategy: QueryStrategy,
    institution: str,
    person_name: str,
    debug: bool = False,
    timeout: float = 20.0,
) -> List[Dict[str, object]]:
    """Execute a single search strategy with reasonable timeout.
    
    Timeout is generous (20s) to allow legitimate slow responses while still
    preventing indefinite hangs. The real speedup comes from fewer strategies
    and early termination when we have good results.
    """
    clean_person = _normalise_whitespace(person_name)
    clean_institution = _normalise_whitespace(institution)
    raw_query = strategy.query
    format_params = {
        "name": clean_person,
        "institution": clean_institution,
        "quoted_name": _quote_clause(clean_person),
        "quoted_institution": _quote_clause(clean_institution),
    }
    try:
        raw_query = raw_query.format(**format_params)
    except Exception:
        pass
    query = _normalise_whitespace(raw_query)
    if strategy.ensure_tokens:
        query = _ensure_name_and_institution(query, clean_person, clean_institution)
    
    try:
        # Timeout is generous to avoid cutting off legitimate searches
        results = await asyncio.wait_for(
            bing_search(
                query,
                num_results=strategy.limit,
                institution=institution,
                person_name=clean_person,
                debug=debug,
            ),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        if debug:
            print(f"[search] Strategy '{strategy.name}' timed out after {timeout}s")
        return []
    except Exception as exc:
        if debug:
            print(f"[search] Strategy '{strategy.name}' failed: {exc}")
        return []
    
    annotated: List[Dict[str, object]] = []
    for result in results:
        entry = _annotate_signals(result, strategy, query)
        annotated.append(entry)
    return annotated


async def _enhanced_search_impl(
    name: str,
    institution: str = "",
    num_results: int = 30,
    debug: bool = False,
) -> List[Dict[str, object]]:
    """Internal implementation of enhanced search without timeout wrapper."""
    clean_name = _normalise_whitespace(name)
    if not clean_name:
        return []

    # OPTIMIZATION: Reduced from num_results // 3 to num_results // 4 for faster processing
    per_strategy_limit = max(3, min(8, num_results // 4 if num_results > 0 else 3))
    strategies = _build_strategies(clean_name, institution, per_strategy_limit)
    if not strategies:
        return await bing_search(
            clean_name,
            num_results=num_results,
            institution=institution,
            person_name=clean_name,
            debug=debug,
        )

    # Reduce concurrent strategies to 3 to avoid overwhelming slow connections
    sem = asyncio.Semaphore(3)

    async def run(strategy: QueryStrategy) -> List[Dict[str, object]]:
        async with sem:
            return await _execute_strategy(strategy, institution, clean_name, debug=debug, timeout=20.0)

    tasks = [asyncio.create_task(run(strategy)) for strategy in strategies]
    combined: Dict[str, Dict[str, object]] = {}

    # OPTIMIZATION: Early termination - if we get good results quickly, don't wait for slow strategies
    # This is SAFE because we only terminate when we already have sufficient high-quality results
    early_termination_threshold = max(10, num_results // 2)  # e.g., 10 results for 20 requested
    high_quality_threshold = 15  # relevance score for "high quality"
    
    completed_strategies = 0
    for coro in asyncio.as_completed(tasks):
        try:
            payload = await coro
            completed_strategies += 1
            
            if isinstance(payload, Exception):
                if debug:
                    print(f"[search] A strategy failed: {payload}")
                continue
            
            # Merge results from this strategy
            strategy_idx = None
            for idx, task in enumerate(tasks):
                if task == coro:
                    strategy_idx = idx
                    break
            
            strategy = strategies[strategy_idx] if strategy_idx is not None else None
            
            for result in payload:
                url = result.get("url")
                if not url:
                    continue
                if url in combined:
                    combined[url] = _merge_results(combined[url], result)
                else:
                    combined[url] = result
            
            if debug and strategy:
                print(f"[search] Completed '{strategy.name}' ({completed_strategies}/{len(strategies)}): {len(combined)} total results")
            
            # Check for early termination: if we have enough high-quality results, stop waiting
            high_quality_results = [
                r for r in combined.values()
                if r["signals"].get("relevance_score", 0) >= high_quality_threshold
            ]
            
            # Terminate early if:
            # 1. We have at least threshold number of total results AND
            # 2. At least half of them are high-quality AND
            # 3. We've completed at least half the strategies (to ensure diversity)
            if (len(combined) >= early_termination_threshold and 
                len(high_quality_results) >= early_termination_threshold // 2 and
                completed_strategies >= len(strategies) // 2):
                
                if debug:
                    print(f"[search] Early termination: {len(combined)} results ({len(high_quality_results)} high-quality) after {completed_strategies}/{len(strategies)} strategies")
                
                # Cancel remaining tasks
                for task in tasks:
                    if not task.done():
                        task.cancel()
                break
                
        except asyncio.CancelledError:
            # Expected when we cancel tasks during early termination
            continue
        except Exception as exc:
            if debug:
                print(f"[search] Unexpected error processing strategy result: {exc}")
            continue

    # Wait for any remaining tasks (should be quick or already done)
    await asyncio.gather(*tasks, return_exceptions=True)

    # NEW: Apply source quality boosting/penalties
    # Note: Penalties reduced to avoid filtering out good results from top positions
    for url, result in combined.items():
        domain = result.get("url", "").lower()
        
        # Boost official .edu domains significantly
        if ".edu" in domain:
            # Extra boost for faculty/people pages
            if any(path in url.lower() for path in ["/faculty/", "/people/", "/staff/", "/directory/", "/profile/"]):
                result["signals"]["relevance_score"] = result["signals"].get("relevance_score", 0) + 30
            else:
                # Regular .edu boost
                result["signals"]["relevance_score"] = result["signals"].get("relevance_score", 0) + 15
        
        # Penalize ONLY the most unreliable domains (reduced penalty to avoid losing good results)
        unreliable_domains = [
            "alchetron.com",  # Known unreliable
            "prabook.com",    # Known unreliable
        ]
        if any(unreliable in domain for unreliable in unreliable_domains):
            # Moderate penalty - don't completely eliminate these results
            result["signals"]["relevance_score"] = max(0, result["signals"].get("relevance_score", 0) - 10)
        
        # Boost reliable sources moderately
        reliable_sources = [
            "wikipedia.org", "britannica.com", "researchgate.net",
            "semanticscholar.org", "scholar.google.com"
        ]
        if any(reliable in domain for reliable in reliable_sources):
            result["signals"]["relevance_score"] = result["signals"].get("relevance_score", 0) + 8

    ordered = sorted(
        combined.values(),
        key=lambda item: (
            item["signals"].get("source_rank", float("inf")),
            -item["signals"].get("relevance_score", 0),
            -item["signals"].get("strategy_hits", 0),
            -len(item.get("snippet", "")),
        ),
    )

    # SAFETY: If we don't have enough results (e.g., all strategies timed out or failed),
    # fall back to basic search to ensure we don't return empty results
    min_acceptable_results = max(5, num_results // 4)
    if len(ordered) < min_acceptable_results:
        if debug:
            print(f"[search] Only {len(ordered)} results from enhanced search, falling back to basic search")
        return await bing_search(
            clean_name,
            num_results=num_results,
            institution=institution,
            person_name=clean_name,
            debug=debug,
        )

    top_results = ordered[:num_results]
    for idx, result in enumerate(top_results, start=1):
        result["rank"] = idx
    await _enrich_with_page_excerpts(top_results, clean_name, limit=min(EXCERPT_FETCH_LIMIT, len(top_results)))
    return top_results


ENHANCED_SEARCH_TIMEOUT = 45.0  # Maximum time for enhanced search to complete


async def enhanced_search(
    name: str,
    institution: str = "",
    num_results: int = 30,
    debug: bool = False,
) -> List[Dict[str, object]]:
    """Execute enhanced search with timeout protection.
    
    This wrapper ensures that enhanced_search never takes more than ENHANCED_SEARCH_TIMEOUT
    seconds, preventing cascading strategy timeouts from blocking the batch.
    If timeout occurs, falls back to basic search.
    """
    try:
        return await asyncio.wait_for(
            _enhanced_search_impl(name, institution, num_results, debug),
            timeout=ENHANCED_SEARCH_TIMEOUT
        )
    except asyncio.TimeoutError:
        if debug:
            print(f"[search] Enhanced search timed out after {ENHANCED_SEARCH_TIMEOUT}s, falling back to basic search")
        # Fallback to basic search on timeout
        clean_name = _normalise_whitespace(name)
        if not clean_name:
            return []
        return await bing_search(
            clean_name,
            num_results=num_results,
            institution=institution,
            person_name=clean_name,
            debug=debug,
        )


async def force_browser_recreation():
    """Force recreation of the browser instance to prevent memory accumulation."""
    global _global_browser
    async with _browser_lock:
        if _global_browser:
            try:
                await _global_browser.close()
            except Exception:
                pass
            finally:
                _global_browser = None


async def refresh_http_client():
    """Refresh the HTTP client to clear connection pool and prevent connection exhaustion."""
    global _http_client
    async with _http_lock:
        if _http_client:
            try:
                await _http_client.aclose()
            except Exception:
                pass
            finally:
                _http_client = None


async def cleanup_batch_resources():
    """Refresh lightweight HTTP resources while leaving browser sessions active."""
    # Only refresh HTTP client between batches to avoid closing browsers that
    # may still be servicing slow tasks. Full browser resets can be triggered
    # manually via force_browser_recreation() if required.
    await refresh_http_client()
    # Optional: could add browser page cleanup here if needed
    # await force_browser_recreation()  # Commented out to prevent connection errors


async def close_search_clients():
    global _global_browser, _http_client
    if _global_browser:
        try:
            await _global_browser.close()
        except Exception:
            pass
        finally:
            _global_browser = None
    if _http_client:
        try:
            await _http_client.aclose()
        except Exception:
            pass
        finally:
            _http_client = None


async def close_browser():  # backward compatibility helper
    await close_search_clients()
