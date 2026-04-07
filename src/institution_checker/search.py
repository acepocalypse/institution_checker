import asyncio
import random
import re
import shutil
import time
import unicodedata
import base64
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple
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

try:
    from ddgs import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    try:
        from duckduckgo_search import DDGS
        DDGS_AVAILABLE = True
    except ImportError:
        DDGS_AVAILABLE = False

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
    "visiting professor",  # Added to boost score for visiting roles
    "visiting scholar",    # Added to boost score for visiting roles
    "lecturer",
    "instructor",
    "faculty",
    "postdoctoral",
    "postdoc",
    "post-doc",
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
    "nobel",
    "nobel prize",
    "laureate",
]

# Explicit connection phrases (high-value signals)
EXPLICIT_CONNECTION_PHRASES = [
    "professor at",
    "professor of",
    "assistant professor at",
    "associate professor at",
    "faculty at",
    "faculty member at",
    "works at",
    "employed at",
    "graduated from",
    "degree from",
    "phd from",
    "postdoc at",
    "post-doc at",
    "postdoctoral fellow at",
    "research fellow at",
    "visiting scholar at",
    "visiting professor at",
    "bachelor from",
    "master from",
    "alumnus of",
    "alumna of",
    "alumni of",
    "studied at",
    "student at",
    "summer school at",
    "summer session at",
    "summer program at",
    "summer institute at",
    "summer student at",
    "attended summer school at",
    "attended a summer program at",
]

# Event/prize patterns (reduce false positives)
EVENT_PRIZE_PATTERNS = [
    "keynote",
    "speaker at",
    "gave talk at",
    "presented at",
    "lecture at",
    "symposium at",
    "conference at",
    # "prize from",  # Removed to prevent false negatives for award lists
    # "award from",  # Removed to prevent false negatives for award lists
    # "honorary degree", # Removed to prevent false negatives
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

DUCKDUCKGO_HTML_URL = "https://html.duckduckgo.com/html/"


@dataclass
class ValidationSearchContext:
    """Shared context for staged validation search calls.

    Keeps lightweight cache/telemetry so the staged pre-LLM pipeline can
    aggregate cache and backend usage across basic/rescue/enhanced phases.
    """

    cache_enabled: bool = True
    allow_bing_fallback: bool = False
    allow_slow_ddg_fallback: bool = False
    cache: Dict[str, List[Dict[str, object]]] = None
    cache_hits: int = 0
    cache_misses: int = 0
    backend_hits: Dict[str, int] = None

    def __post_init__(self) -> None:
        if self.cache is None:
            self.cache = {}
        if self.backend_hits is None:
            self.backend_hits = {"cache": 0, "ddg": 0, "bing": 0, "enhanced": 0}


def _validation_cache_key(
    query: str,
    institution: str,
    person_name: str,
    limit: int,
    ensure_tokens: bool,
) -> str:
    return "||".join(
        [
            _normalise_whitespace(query).lower(),
            _normalise_whitespace(institution).lower(),
            _normalise_whitespace(person_name).lower(),
            str(limit),
            "1" if ensure_tokens else "0",
        ]
    )


async def validation_search_query(
    query: str,
    *,
    institution: str,
    person_name: str,
    limit: int,
    debug: bool = False,
    context: Optional[ValidationSearchContext] = None,
    prefer_backend: str = "ddg",
    allow_bing_fallback: bool = False,
    allow_slow_ddg_fallback: bool = False,
    ensure_tokens: bool = True,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    ctx = context or ValidationSearchContext()
    cache_key = _validation_cache_key(query, institution, person_name, limit, ensure_tokens)

    if ctx.cache_enabled and cache_key in ctx.cache:
        ctx.cache_hits += 1
        ctx.backend_hits["cache"] = ctx.backend_hits.get("cache", 0) + 1
        return list(ctx.cache[cache_key]), {
            "backend_used": "cache",
            "cache_hit": True,
            "network_queries_used": 0,
            "ddg_manual_retry_used": False,
            "ddg_browser_fallback_used": False,
            "bing_fallback_used": False,
            "network_attempt_count": 0,
        }

    ctx.cache_misses += 1
    effective_allow_bing = allow_bing_fallback or ctx.allow_bing_fallback
    effective_allow_slow = allow_slow_ddg_fallback or ctx.allow_slow_ddg_fallback

    if prefer_backend == "enhanced":
        results = await enhanced_search(
            person_name or query,
            institution=institution,
            num_results=limit,
            debug=debug,
            fetch_excerpts=False,
        )
        backend = "enhanced"
        ctx.backend_hits["enhanced"] = ctx.backend_hits.get("enhanced", 0) + 1
        ddg_manual_retry_used = False
        ddg_browser_fallback_used = False
        bing_fallback_used = False
    else:
        results = await bing_search(
            query,
            num_results=limit,
            institution=institution,
            person_name=person_name,
            debug=debug,
            allow_fallback=True,
            fetch_excerpts=False,
            ensure_tokens=ensure_tokens,
        )
        backend = "ddg"
        if effective_allow_bing:
            backend = "ddg|bing"
        ctx.backend_hits["ddg"] = ctx.backend_hits.get("ddg", 0) + 1
        if effective_allow_bing:
            ctx.backend_hits["bing"] = ctx.backend_hits.get("bing", 0) + 1
        ddg_manual_retry_used = bool(effective_allow_slow)
        ddg_browser_fallback_used = False
        bing_fallback_used = bool(effective_allow_bing)

    if ctx.cache_enabled:
        ctx.cache[cache_key] = list(results)

    return results, {
        "backend_used": backend,
        "cache_hit": False,
        "network_queries_used": 1,
        "ddg_manual_retry_used": ddg_manual_retry_used,
        "ddg_browser_fallback_used": ddg_browser_fallback_used,
        "bing_fallback_used": bing_fallback_used,
        "network_attempt_count": 1,
    }


async def validation_basic_search(
    name: str,
    institution: str,
    *,
    debug: bool = False,
    context: Optional[ValidationSearchContext] = None,
    allow_bing_fallback: bool = False,
    allow_slow_ddg_fallback: bool = False,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    query = f'"{name}" "{institution}"'
    return await validation_search_query(
        query,
        institution=institution,
        person_name=name,
        limit=20,
        debug=debug,
        context=context,
        prefer_backend="ddg",
        allow_bing_fallback=allow_bing_fallback,
        allow_slow_ddg_fallback=allow_slow_ddg_fallback,
        ensure_tokens=True,
    )


async def validation_enhanced_search(
    name: str,
    institution: str,
    *,
    debug: bool = False,
    context: Optional[ValidationSearchContext] = None,
    allow_bing_fallback: bool = False,
    allow_slow_ddg_fallback: bool = False,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    ctx = context or ValidationSearchContext()
    cache_key = _validation_cache_key(name, institution, name, 30, False)

    if ctx.cache_enabled and cache_key in ctx.cache:
        ctx.cache_hits += 1
        ctx.backend_hits["cache"] = ctx.backend_hits.get("cache", 0) + 1
        return list(ctx.cache[cache_key]), {
            "backend_used": "cache",
            "cache_hit": True,
            "network_queries_used": 0,
            "ddg_manual_retry_used": False,
            "ddg_browser_fallback_used": False,
            "bing_fallback_used": False,
            "network_attempt_count": 0,
        }

    ctx.cache_misses += 1
    results = await enhanced_search(
        name,
        institution=institution,
        num_results=30,
        debug=debug,
        fetch_excerpts=False,
    )
    if ctx.cache_enabled:
        ctx.cache[cache_key] = list(results)
    ctx.backend_hits["enhanced"] = ctx.backend_hits.get("enhanced", 0) + 1

    return results, {
        "backend_used": "enhanced",
        "cache_hit": False,
        "network_queries_used": 1,
        "ddg_manual_retry_used": bool(allow_slow_ddg_fallback or ctx.allow_slow_ddg_fallback),
        "ddg_browser_fallback_used": False,
        "bing_fallback_used": bool(allow_bing_fallback or ctx.allow_bing_fallback),
        "network_attempt_count": 1,
    }
BING_URL = "https://www.bing.com/search"
EXCERPT_FETCH_LIMIT = 16  # Increased from 4 to ensure we get evidence for top results
EXCERPT_HTTP_TIMEOUT = 6.0  # seconds (reduced for speed)
EXCERPT_MAX_CHARS = 800  # Increased from 600 to capture more context
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

DOMAIN_BLACKLIST = {
    "zhihu.com",
    "facebook.com",
    "twitter.com",
    "instagram.com",
    "tiktok.com",
    "pinterest.com",
    "reddit.com",
    "quora.com",
    "youtube.com",
    "vimeo.com",
    "dailymotion.com",
}




@dataclass
class QueryStrategy:
    name: str
    query: str
    limit: int
    boost: int = 0
    ensure_tokens: bool = True


@dataclass(frozen=True)
class TargetTokens:
    person: Tuple[str, ...]
    institution: Tuple[str, ...]

    @property
    def is_empty(self) -> bool:
        return not (self.person or self.institution)

    @property
    def has_person_tokens(self) -> bool:
        return bool(self.person)

    @property
    def has_institution_tokens(self) -> bool:
        return bool(self.institution)

    @property
    def all_tokens(self) -> Tuple[str, ...]:
        return self.person + self.institution


_http_client: Optional[httpx.AsyncClient] = None
_http_lock = asyncio.Lock()
_global_browser = None
_browser_lock = asyncio.Lock()
_browser_page_pool: List = []
_browser_page_lock = asyncio.Lock()
_BROWSER_PAGE_POOL_SIZE = 30
_browser_semaphore = asyncio.Semaphore(8)  # Limit concurrent browser pages (balanced for speed/stability)

# Global rate limiter for Bing
_last_request_time = 0.0
_request_interval = 0.6  # Minimum seconds between requests (0.6s = ~6000 req/hr)
_rate_limit_lock = asyncio.Lock()

async def _wait_for_rate_limit():
    global _last_request_time
    async with _rate_limit_lock:
        now = time.time()
        elapsed = now - _last_request_time
        if elapsed < _request_interval:
            delay = _request_interval - elapsed
            await asyncio.sleep(delay)
        _last_request_time = time.time()

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
            return _strip_control_characters(fixed)
        except Exception:
            pass
    
    # Fallback if ftfy is not available: just return text stripped of control characters
    return _strip_control_characters(text)


def _strip_control_characters(text: str) -> str:
    if not text:
        return text
    return "".join(
        ch for ch in text
        if (unicodedata.category(ch)[0] != "C") or ch in {"\n", "\r", "\t"}
    )


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
    # Remove punctuation (replace with space to avoid merging words)
    normalized = re.sub(r'[^\w\s]', ' ', normalized)
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
        # Check if the unquoted name is already in the query (not just the quoted version)
        normalized_name = _normalise_whitespace(person_name).lower()
        if normalized_name and normalized_name not in lowered:
            # Name not in query, add it with quotes to keep full name together
            name_clause = _quote_clause(person_name)
            parts.append(name_clause)
    if institution:
        # Check if the unquoted institution is already in the query
        normalized_institution = _normalise_whitespace(institution).lower()
        # FIX: If site: operator is present, we don't need to force institution name
        has_site_operator = "site:" in lowered
        if normalized_institution and normalized_institution not in lowered and not has_site_operator:
            # Institution not in query, add it with quotes
            institution_clause = _quote_clause(institution)
            parts.append(institution_clause)
    if parts:
        query = _compose_query(*parts, query)
    return query


def _collect_target_tokens(person_name: str, institution: str) -> TargetTokens:
    person_tokens: set[str] = set()
    normalized_person = _normalize_name_for_matching(person_name)
    if normalized_person and " " in normalized_person:
        person_tokens.add(normalized_person)
    components = _extract_name_components(person_name)
    for value in components.get("last", []):
        normalized = _normalize_name_for_matching(value)
        if normalized and len(normalized) >= 3:
            person_tokens.add(normalized)

    institution_tokens: set[str] = set()
    normalized_institution = _normalize_name_for_matching(institution)
    if normalized_institution:
        institution_tokens.add(normalized_institution)
    for token in _institution_tokens(institution):
        normalized = _normalize_name_for_matching(token)
        if normalized and len(normalized) >= 4 and normalized not in {"university", "college"}:
            institution_tokens.add(normalized)

    def _ordered(items: Iterable[str]) -> Tuple[str, ...]:
        return tuple(sorted({item for item in items if item}, key=lambda value: (-len(value), value)))

    return TargetTokens(
        person=_ordered(person_tokens),
        institution=_ordered(institution_tokens),
    )


def _result_mentions_target(result: Dict[str, object], tokens: TargetTokens) -> bool:
    if tokens.is_empty:
        return False

    signals = result.get("signals", {}) or {}
    text_parts = [
        result.get("title", ""),
        result.get("snippet", ""),
        result.get("page_excerpt", ""),
        result.get("description", ""),
        result.get("url", ""),
    ]
    merged = _normalize_name_for_matching(
        " ".join(str(part) for part in text_parts if part)
    )

    person_hit = not tokens.has_person_tokens
    if tokens.has_person_tokens:
        if signals.get("has_person_name"):
            person_hit = True
        elif merged:
            person_hit = any(token in merged for token in tokens.person)

    institution_hit = not tokens.has_institution_tokens
    if tokens.has_institution_tokens:
        if signals.get("has_institution"):
            institution_hit = True
        elif merged:
            institution_hit = any(token in merged for token in tokens.institution)

    return person_hit and institution_hit


def _count_target_hits(results: List[Dict[str, object]], tokens: TargetTokens, limit: int = 10) -> int:
    if tokens.is_empty or not results:
        return 0
    hits = 0
    for result in results[:limit]:
        if _result_mentions_target(result, tokens):
            hits += 1
    return hits


def _needs_duckduckgo_fallback(
    results: List[Dict[str, object]],
    person_name: str,
    institution: str,
) -> bool:
    if not results:
        return True
    
    # Quality-Aware Fallback: Check max relevance score
    # If httpx results have max score < 5, they are likely garbage (shopping links, unrelated content)
    # This triggers DDG fallback to find better results
    max_score = max((r.get("signals", {}).get("relevance_score", 0) for r in results), default=0)
    if max_score < 5:
        return True
    
    # Also check if we have any high-quality results (score >= 10)
    # If we have results but none are good, we need backup
    has_high_quality = any(
        r.get("signals", {}).get("relevance_score", 0) >= 10 
        for r in results
    )
    if not has_high_quality:
        return True

    tokens = _collect_target_tokens(person_name, institution)
    if tokens.is_empty:
        return False
    subset = results[: min(len(results), 10)]
    hits = _count_target_hits(subset, tokens, limit=len(subset))
    if len(subset) <= 3:
        return hits == 0
    if len(subset) <= 5:
        return hits < 2
    return hits < 3


def _has_sufficient_target_hits(
    results: List[Dict[str, object]],
    person_name: str,
    institution: str,
) -> bool:
    if not results:
        return False
    tokens = _collect_target_tokens(person_name, institution)
    if tokens.is_empty:
        return bool(results)
    subset = results[: min(len(results), 10)]
    hits = _count_target_hits(subset, tokens, limit=len(subset))
    if len(subset) <= 3:
        return hits >= 1
    if len(subset) <= 5:
        return hits >= 2
    return hits >= 3


def _prioritize_target_hits(
    results: List[Dict[str, object]],
    person_name: str,
    institution: str,
) -> List[Dict[str, object]]:
    tokens = _collect_target_tokens(person_name, institution)
    if tokens.is_empty:
        return results
    relevant: List[Dict[str, object]] = []
    others: List[Dict[str, object]] = []
    for item in results:
        if _result_mentions_target(item, tokens):
            relevant.append(item)
        else:
            others.append(item)
    if not relevant:
        return results
    return relevant + others


_PLACE_SUFFIX_TOKENS = {
    "weg",
    "street",
    "st",
    "road",
    "rd",
    "strasse",
    "straße",
    "platz",
    "plaza",
    "hall",
    "building",
    "tower",
    "center",
    "centre",
    "wing",
    "lab",
    "laboratory",
    "auditorium",
    "library",
}

_PLACE_PREFIX_TOKENS = {
    "hall",
    "building",
    "tower",
    "center",
    "centre",
    "laboratory",
    "lab",
}


def _name_pair_place_status(text_tokens: List[str], first_token: str, last_token: str) -> Tuple[bool, bool]:
    """Return (has_pair, has_non_place_pair) for contiguous first/last tokens."""
    has_pair = False
    has_non_place_pair = False
    if not first_token or not last_token:
        return has_pair, has_non_place_pair
    upper_bound = len(text_tokens) - 1
    for idx in range(upper_bound):
        if text_tokens[idx] != first_token or text_tokens[idx + 1] != last_token:
            continue
        has_pair = True
        prev_token = text_tokens[idx - 1] if idx > 0 else ""
        next_idx = idx + 2
        next_token = text_tokens[next_idx] if next_idx < len(text_tokens) else ""
        if next_token not in _PLACE_SUFFIX_TOKENS and prev_token not in _PLACE_PREFIX_TOKENS:
            has_non_place_pair = True
    return has_pair, has_non_place_pair


def _name_matches(text: str, person_name: str) -> bool:
    """Enhanced name matching with support for:
    - Middle name variations (John S. Smith vs John Samuel Smith)
    - Diacritics (Cordova with or without accent marks)
    - Different name orderings
    - Initials
    """
    # Normalize both text and name for comparison
    normalized_text = _normalize_name_for_matching(text)
    text_tokens = normalized_text.split()
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
    place_only_pair = False
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
            pair_found, pair_clean = _name_pair_place_status(text_tokens, first_token, last_token)
            if pair_clean:
                return True
            if pair_found:
                place_only_pair = True
            
            # Pattern 4: Last, First (reversed with optional comma)
            pattern = re.compile(rf'\b{re.escape(last_token)}\s*,?\s+{re.escape(first_token)}\b')
            if pattern.search(normalized_text):
                return True
    
    # Fallback: first/last tokens both present somewhere in the text
    if place_only_pair:
        return False
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
                
                # Try Base64 decoding (Bing often uses a1 + base64)
                if actual_url.startswith('a1'):
                    try:
                        # Strip 'a1' prefix and add padding if needed
                        b64_str = actual_url[2:]
                        # Replace characters that might be different in Bing's encoding if needed
                        # But usually it's standard base64url or similar.
                        # Fix padding
                        padding = len(b64_str) % 4
                        if padding:
                            b64_str += '=' * (4 - padding)
                        
                        # Try standard base64 first, then urlsafe
                        try:
                            decoded_bytes = base64.urlsafe_b64decode(b64_str)
                        except Exception:
                            decoded_bytes = base64.b64decode(b64_str)
                            
                        decoded = decoded_bytes.decode('utf-8')
                        return _normalise_url(decoded)
                    except Exception:
                        pass

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


def _decode_duckduckgo_redirect(url: str) -> str:
    """Decode DuckDuckGo redirect links (//duckduckgo.com/l/...) to destination URLs."""
    if not url:
        return ""
    if url.startswith("//"):
        url = f"https:{url}"
    try:
        parsed = urlparse(url)
    except Exception:
        return url
    if "duckduckgo.com" in parsed.netloc.lower() and parsed.path.startswith("/l/"):
        params = dict(parse_qsl(parsed.query))
        target = params.get("uddg")
        if target:
            return _normalise_url(target)
    return _normalise_url(url)


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
    
    # NEW: Check for explicit connection phrases (high-value signals)
    has_explicit_connection = _contains_any(text, EXPLICIT_CONNECTION_PHRASES)
    
    # NEW: Check for event/prize patterns (false positive indicators)
    has_event_prize_pattern = _contains_any(text, EVENT_PRIZE_PATTERNS)
    
    # NEW: Check proximity of person and institution mentions
    has_proximity = False
    if has_person and has_institution:
        # Simple proximity check: both appear in title or within same sentence
        title_lower = title.lower()
        if person_name and institution:
            person_in_title = any(token.lower() in title_lower for token in person_name.split() if len(token) > 2)
            inst_in_title = any(token in title_lower for token in _institution_tokens(institution))
            has_proximity = person_in_title and inst_in_title

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
    
    # NEW: Bonus for explicit connection phrases
    if has_explicit_connection:
        score += 5
        if domain.endswith(".edu"):
            score += 3  # Extra bonus for .edu + explicit connection
    
    # NEW: Bonus for PhD/Degree + Institution (strong alumni signal)
    has_degree_signal = any(term in text for term in ["phd", "degree", "doctorate", "graduated"])
    if has_degree_signal and has_institution:
        score += 5
    
    # NEW: Bonus for proximity
    if has_proximity:
        score += 4
    
    # NEW: Penalty for event/prize patterns
    if has_event_prize_pattern and not has_explicit_connection:
        score = max(0, score - 6)  # Moderate penalty unless explicit connection phrase present
    
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
        "has_explicit_connection": bool(has_explicit_connection),
        "has_event_prize_pattern": bool(has_event_prize_pattern),
        "has_proximity": bool(has_proximity),
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


def _deduplicate_results(results: List[Dict[str, object]]) -> List[Dict[str, object]]:
    seen: Dict[str, Dict[str, object]] = {}
    ordered: List[str] = []
    for result in results:
        url = result.get("url")
        if not url:
            continue
        normalized = _normalise_url(url)
        if normalized in seen:
            existing = seen[normalized]
            merged = _merge_results(existing, result)
            seen[normalized] = merged
        else:
            seen[normalized] = result
            ordered.append(normalized)
    return [seen[key] for key in ordered]


def _prepare_ranked_results(results: List[Dict[str, object]]) -> List[Dict[str, object]]:
    deduped = _deduplicate_results(results)
    deduped.sort(
        key=lambda item: (
            item.get("signals", {}).get("source_rank", float("inf")),
            -item.get("signals", {}).get("relevance_score", 0),
            -len(item.get("snippet", "")),
        )
    )
    for idx, item in enumerate(deduped, start=1):
        item["rank"] = idx
        signals = item.setdefault("signals", {})
        if "source_rank" not in signals or not isinstance(signals.get("source_rank"), int):
            signals["source_rank"] = idx
    return deduped


def _split_duckduckgo_text(text: str) -> Tuple[str, str]:
    working = text.strip()
    working = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", working)
    working = working.replace("Flag AI-generated image", "")
    working = working.replace("Report inappropriate image", "")
    working = working.replace("**", "")
    if working.startswith("### "):
        working = working[4:]
    working = _normalise_whitespace(working)
    if " - " in working:
        title, snippet = working.split(" - ", 1)
        return _normalise_whitespace(title), _normalise_whitespace(snippet)
    return _normalise_whitespace(working), ""


async def _get_http_client() -> httpx.AsyncClient:
    global _http_client
    async with _http_lock:
        if _http_client is None:
            timeout = httpx.Timeout(10.0, connect=5.0, read=10.0)
            limits = httpx.Limits(max_connections=30, max_keepalive_connections=10)
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
            await _wait_for_rate_limit()
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


# Global semaphore for DuckDuckGo to prevent rate limiting
_ddg_semaphore = asyncio.Semaphore(3)
DDG_LIBRARY_TIMEOUT = 20.0
DDG_STAGE_SOFT_TIMEOUT = 12.0

_DDG_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0",
]

async def _duckduckgo_search(
    query: str,
    institution: str,
    person_name: str,
    limit: int,
    debug: bool = False,
) -> List[Dict[str, object]]:
    # 1. Try duckduckgo_search library first if available
    if DDGS_AVAILABLE:
        try:
            if debug:
                print(f"[search] Using duckduckgo_search library for: {query}")
            
            def _run_ddgs():
                res = []
                # Use context manager to ensure cleanup
                with DDGS() as ddgs:
                    # fetch a few more to filter
                    for r in ddgs.text(query, max_results=max(limit, 20)):
                        res.append(r)
                return res

            loop = asyncio.get_running_loop()
            async with _ddg_semaphore:
                # Use executor to avoid blocking the event loop
                ddg_results = await asyncio.wait_for(
                    loop.run_in_executor(None, _run_ddgs),
                    timeout=DDG_LIBRARY_TIMEOUT,
                )
            
            if debug:
                 print(f"[search] duckduckgo_search library returned {len(ddg_results)} raw results")

            results: List[Dict[str, object]] = []
            for r in ddg_results:
                if len(results) >= limit:
                    break
                
                title = r.get("title", "")
                url = r.get("href", "")
                snippet = r.get("body", "")
                
                if not url.startswith(("http://", "https://")):
                    continue
                
                domain = urlparse(url).netloc.lower()
                base_domain = domain.replace("www.", "")
                if any(blocked in base_domain for blocked in DOMAIN_BLACKLIST):
                    continue

                signals = _compute_signals(title, snippet, url, institution, person_name)
                # Ensure we have person match signals
                signals = _ensure_person_signal(signals, url, institution, person_name)

                results.append({
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                    "signals": signals,
                    "source": "duckduckgo_lib"
                })
            
            if results:
                 if debug:
                     print(f"[search] duckduckgo_search library yielded {len(results)} valid results")
                 return results
                 
        except asyncio.TimeoutError:
            if debug:
                print(f"[search] duckduckgo_search library timed out after {DDG_LIBRARY_TIMEOUT:.0f}s, falling back to manual scraping")
        except Exception as e:
            if debug:
                print(f"[search] duckduckgo_search library failed, falling back to manual scraping: {e}")
            pass

    client = await _get_http_client()
    # Base headers
    headers = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://html.duckduckgo.com/",
        "Upgrade-Insecure-Requests": "1",
        "Cache-Control": "max-age=0",
        "Sec-Ch-Ua": '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"Windows"',
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "same-origin",
        "Sec-Fetch-User": "?1"
    }
    
    # Use direct HTML endpoint
    url = "https://html.duckduckgo.com/html/"
    params = {"q": query, "kl": "us-en"}
    
    # Retry logic with backoff and semaphore
    html = ""
    async with _ddg_semaphore:
        # Add random delay to spread out requests
        await asyncio.sleep(random.uniform(0.8, 1.5))
        
        for attempt in range(1, 4):
            # Rotate UA
            headers["User-Agent"] = random.choice(_DDG_USER_AGENTS)
            
            try:
                response = await client.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=20.0,
                    follow_redirects=True
                )
                response.raise_for_status()
                html = response.text
                
                # Check for CAPTCHA
                if "bots use DuckDuckGo too" in html or "anomaly-modal" in html:
                    if debug:
                        print(f"[search] DuckDuckGo CAPTCHA detected on attempt {attempt}")
                    # Treat as error to trigger retry
                    raise httpx.HTTPStatusError("DDG CAPTCHA", request=response.request, response=response)

                if html and len(html) > 200:
                    break
            except (httpx.HTTPError, httpx.TimeoutException) as exc:
                if debug:
                    print(f"[search] DuckDuckGo attempt {attempt} failed: {exc}")
                if attempt < 3:
                    await asyncio.sleep(3.0 * attempt)
    
    if not html or len(html) < 200:
        if debug:
            print("[search] DuckDuckGo HTTP failed/blocked, trying browser fallback...")
        html = await _fetch_ddg_with_browser(query, debug=debug)
    
    if not html or len(html) < 200:
        if debug:
            print("[search] DuckDuckGo response too short/empty after retries and fallback")
        return []

    # Parse HTML directly
    soup = BeautifulSoup(html, "html.parser")
    results: List[Dict[str, object]] = []
    
    # DDG HTML structure: .result -> .result__body -> .result__title -> a.result__a
    for result_div in soup.select(".result"):
        if len(results) >= limit:
            break
            
        title_tag = result_div.select_one(".result__a")
        if not title_tag:
            continue
            
        title = title_tag.get_text(strip=True)
        url = title_tag.get("href", "")
        
        # Extract snippet
        snippet_tag = result_div.select_one(".result__snippet")
        snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""
        
        # Resolve DDG redirect in URL if needed (usually /l/?kh=-1&uddg=...)
        if "/l/?" in url:
            try:
                parsed = urlparse(url)
                qs = dict(parse_qsl(parsed.query))
                if "uddg" in qs:
                    url = qs["uddg"]
            except Exception:
                pass
                
        if not url.startswith(("http://", "https://")):
            continue
            
        # Check blacklist
        domain = urlparse(url).netloc.lower()
        base_domain = domain.replace("www.", "")
        if any(blocked in base_domain for blocked in DOMAIN_BLACKLIST):
            continue
            
        # Compute signals
        signals = _compute_signals(title, snippet, url, institution, person_name)
        signals = _ensure_person_signal(signals, url, institution, person_name)
        
        results.append({
            "title": title,
            "url": url,
            "snippet": snippet,
            "signals": signals,
        })

    if debug:
        print(f"[search] DuckDuckGo found {len(results)} results")
        if len(results) == 0:
            print(f"[search] HTML Preview: {html[:500]}...")
            with open("debug_ddg.html", "w", encoding="utf-8") as f:
                f.write(html)
        
    return results


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
        # BUT if the person match came only from URL/Domain/Recovery (not snippet), we still need to fetch text!
        # Otherwise the LLM only sees a generic snippet and might reject the connection.
        if match_confidence in ("url", "institution_domain", "directory_recovery", "soft"):
            return True
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


async def enrich_with_page_excerpts(
    results: Iterable[Dict[str, object]],
    person_name: str,
    limit: int = EXCERPT_FETCH_LIMIT,
) -> None:
    """Public helper to attach page excerpts on-demand."""
    await _enrich_with_page_excerpts(results, person_name, limit=limit)


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
                    await asyncio.sleep(0.2)  # Reduced from 0.6s for faster processing
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
    
    # Acquire semaphore to limit concurrent browser usage
    try:
        async with _browser_semaphore:
            # SAFETY: Wrap the entire browser interaction in a timeout to prevent hanging
            # This ensures the semaphore is always released even if pyppeteer hangs
            async def _do_browser_fetch():
                nonlocal page, release_page, result_html
                for attempt in range(1, BROWSER_FETCH_ATTEMPTS + 1):
                    if page is None:
                        page, release_page = await _acquire_browser_page()
                        if page is None:
                            if debug:
                                print("[search] Pyppeteer browser not available")
                            break
                    try:
                        await _wait_for_rate_limit()
                        await _ensure_page_headers(page)
                        user_agent = random.choice(USER_AGENTS)
                        if getattr(page, "_ic_user_agent", None) != user_agent:
                            await page.setUserAgent(user_agent)
                            page._ic_user_agent = user_agent
                        
                        # Page load timeout 35s (increased for stability)
                        await page.goto(target_url, timeout=35000, waitUntil="domcontentloaded")
                        await asyncio.sleep(0.25)
                        await _dismiss_bing_consent(page, debug=debug)
                        result_ready = False
                        for selector in SELECTORS:
                            try:
                                await page.waitForSelector(selector, {"timeout": 10000}) # 10s selector wait
                                result_ready = True
                                break
                            except PyppeteerTimeoutError:
                                continue
                            except Exception:
                                continue
                        if not result_ready and debug:
                            print(f"[search] Browser attempt {attempt}: no result selectors found")
                        
                        html = await page.content()
                        
                        # VALIDATION: Check for soft blocks / captchas
                        lower_html = html.lower()
                        if "captcha" in lower_html or "unusual traffic" in lower_html or "verify you are human" in lower_html:
                            print(f"[search] Browser attempt {attempt}: DETECTED CAPTCHA/BLOCK")
                            html = None # Treat as failure to trigger retry
                        
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
            
            # Hard timeout for the browser operation
            await asyncio.wait_for(_do_browser_fetch(), timeout=15.0)
            
    except asyncio.TimeoutError:
        if debug:
            print(f"[search] Browser fetch HARD TIMEOUT for query: {query[:30]}...")
    except Exception as e:
        if debug:
            print(f"[search] Browser fetch error: {e}")
    finally:
        if release_page and page and not _page_is_closed(page):
            await release_page(page)
            
    return result_html


async def _fetch_ddg_with_browser(query: str, debug: bool = False) -> str:
    """Fetch DuckDuckGo results using a browser (fallback for CAPTCHA)."""
    page = None
    release_page = None
    result_html = ""
    # Use HTML version in browser too for consistent parsing
    target_url = f"https://html.duckduckgo.com/html/?q={quote(query)}&kl=us-en"
    
    try:
        async with _browser_semaphore:
            async def _do_browser_fetch():
                nonlocal page, release_page, result_html
                for attempt in range(1, BROWSER_FETCH_ATTEMPTS + 1):
                    if page is None:
                        page, release_page = await _acquire_browser_page()
                        if page is None:
                            if debug:
                                print("[search] Pyppeteer browser not available for DDG")
                            break
                    try:
                        await _wait_for_rate_limit()
                        await _ensure_page_headers(page)
                        user_agent = random.choice(USER_AGENTS)
                        if getattr(page, "_ic_user_agent", None) != user_agent:
                            await page.setUserAgent(user_agent)
                            page._ic_user_agent = user_agent
                        
                        if debug:
                            print(f"[search] DDG Browser attempt {attempt} for '{query}'")
                        
                        await page.goto(target_url, timeout=30000, waitUntil="domcontentloaded")
                        await asyncio.sleep(1.5) # Wait for JS/Rendering
                        
                        # Check for results (HTML version uses .result)
                        try:
                            await page.waitForSelector(".result", {"timeout": 5000})
                        except Exception:
                            pass
                        
                        html = await page.content()
                        
                        # Check for CAPTCHA
                        if "bots use DuckDuckGo too" in html or "anomaly-modal" in html:
                            if debug:
                                print(f"[search] DDG Browser attempt {attempt}: CAPTCHA detected")
                            html = None # Retry
                        
                        if html:
                            result_html = html
                            break
                            
                    except Exception as exc:
                        if debug:
                            print(f"[search] DDG Browser attempt {attempt} failed: {exc}")
                        try:
                            await page.close()
                        except Exception:
                            pass
                        page = None
                        release_page = None
                    
                    if attempt < BROWSER_FETCH_ATTEMPTS:
                        await asyncio.sleep(1.0 * attempt)
            
            await asyncio.wait_for(_do_browser_fetch(), timeout=45.0)
            
    except Exception as e:
        if debug:
            print(f"[search] DDG Browser fetch error: {e}")
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
    
    # Ensure resolved URL is valid and absolute
    if not url.startswith(("http://", "https://")):
        return None
    
    # Check blacklist
    domain = urlparse(url).netloc.lower()
    # Remove www. prefix for checking
    base_domain = domain.replace("www.", "")
    if any(blocked in base_domain for blocked in DOMAIN_BLACKLIST):
        return None
    
    title = _normalise_whitespace(anchor.get_text(" ", strip=True))
    if not title:
        return None
    snippet = _build_extended_snippet(element)
    signals = _compute_signals(title, snippet, url, institution, person_name)
    signals = _ensure_person_signal(signals, url, institution, person_name)
    
    # QUALITY FILTER 1: Exclude completely irrelevant results (neither person nor institution)
    # These are pure noise (e.g., generic Reddit links, ads, unrelated content)
    has_person = signals.get("has_person_name", False)
    has_institution = signals.get("has_institution", False)
    
    # NEW: Strict Person Filter
    # If we are searching for a person, the result MUST mention them (at least last name).
    if person_name:
        # If signals found the person, we are good.
        # If not, we check for last name tokens to avoid keeping generic institution pages.
        if not has_person:
            combined_text = f"{title} {snippet} {url}".lower()
            
            # Use _collect_target_tokens to get robust last name tokens
            target_tokens = _collect_target_tokens(person_name, "")
            person_tokens = target_tokens.person
            
            found_token = False
            if person_tokens:
                for token in person_tokens:
                    # Use word boundary check for short tokens to avoid false positives
                    if len(token) < 4:
                        if re.search(rf"\b{re.escape(token)}\b", combined_text):
                            found_token = True
                            break
                    elif token in combined_text:
                        found_token = True
                        break
            else:
                # Fallback for names that didn't generate tokens
                normalized_name = _normalize_name_for_matching(person_name)
                if normalized_name in combined_text:
                    found_token = True
            
            if not found_token:
                # RECOVERY: If this looks like a relevant institutional page (directory, department, alumni list),
                # keep it to allow deep inspection (fetching page excerpt).
                # This rescues pages like "Purdue Chemistry Faculty" where the name is on the page but not the snippet.
                is_directory_page = False
                if has_institution:
                    # Check for directory indicators in URL or text
                    # Expanded list to capture lists of people/awards
                    dir_keywords = {
                        "directory", "faculty", "staff", "people", "department", "school", "college", 
                        "alumni", "laureates", "awards", "recipients", "winners", "fellows", "members",
                        "history", "archive", "news"
                    }
                    if any(kw in combined_text for kw in dir_keywords):
                        is_directory_page = True
                        # Boost relevance slightly to ensure it survives ranking if it was low
                        if signals.get("relevance_score", 0) < 5:
                            signals["relevance_score"] = 5
                            signals["person_match_confidence"] = "directory_recovery"

                if not is_directory_page:
                    # RELAXED: If we have a strong institution match, keep it but mark as low confidence
                    # This allows the LLM or page fetcher to verify.
                    if has_institution:
                        signals["person_match_confidence"] = "soft_recovery"
                    else:
                        return None  # Filter out: no person token found AND not a directory page

    if not has_person and not has_institution:
        # Double-check: Sometimes names/institutions are in URL or snippet but missed by signals
        combined_text = f"{title} {snippet} {url}".lower()
        
        # Check for person name (last name at minimum, 4+ chars to avoid false positives)
        person_in_text = False
        if person_name:
            name_parts = person_name.lower().split()
            if len(name_parts) >= 2:
                last_name = name_parts[-1]
                if len(last_name) >= 4 and last_name in combined_text:
                    person_in_text = True
        
        # Check for institution (any token)
        institution_in_text = False
        if institution:
            inst_tokens = _institution_tokens(institution)
            if any(token in combined_text for token in inst_tokens):
                institution_in_text = True
        
        # If neither appears anywhere, this is completely irrelevant
        if not person_in_text and not institution_in_text:
            # DEBUG: Print why we are dropping
            # print(f"[search] Dropping irrelevant result: {title} (No person/inst match)")
            return None  # Filter out: no connection to search query
    
    # QUALITY FILTER 2: Exclude weak pronoun-only mentions
    # Results that mention institution but only use pronouns (not actual name) are usually weak
    if has_institution and not has_person:
        combined_text = f"{title} {snippet}".lower()
        
        # Check for weak pronoun patterns that often indicate indirect/weak mentions
        weak_pronoun_patterns = [
            r'\bhe\s+was\b',
            r'\bshe\s+was\b',
            r'\bthey\s+were\b',
            r'\bhis\s+work\b',
            r'\bher\s+research\b',
            r'\btheir\s+study\b',
        ]
        
        has_weak_pronoun = any(re.search(pattern, combined_text) for pattern in weak_pronoun_patterns)
        
        if has_weak_pronoun:
            # Only keep if it has other strong signals that justify inclusion
            relevance_score = signals.get("relevance_score", 0)
            has_explicit_connection = signals.get("has_explicit_connection", False)
            
            # Keep only if score is high (15+) OR explicit connection phrase found
            if relevance_score < 15 and not has_explicit_connection:
                return None  # Filter out: weak pronoun reference without strong evidence
    
    # Passed quality filters - return the result
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
    return results


async def bing_search(
    query: str,
    num_results: int = 20,
    institution: str = "",
    person_name: str = "",
    debug: bool = False,
    allow_fallback: bool = True,
    fetch_excerpts: bool = True,
    ddg_min_results: int = 5,
    ensure_tokens: bool = True,
) -> List[Dict[str, object]]:
    if ensure_tokens:
        query = _ensure_name_and_institution(query, person_name, institution)
    target = max(num_results, 10)
    ddg_prefetched: List[Dict[str, object]] = []
    
    # STRATEGY UPDATE: Prioritize DuckDuckGo for speed and reliability (avoids Bing blocks)
    # This helps meet the <4s per name target and improves recall on blocked queries.
    if allow_fallback:
        if debug:
            print("[search] Attempting DuckDuckGo first for speed/reliability...")
        try:
            ddg_results = await asyncio.wait_for(
                _duckduckgo_search(
                    query,
                    institution=institution,
                    person_name=person_name,
                    limit=max(num_results, 20),
                    debug=debug,
                ),
                timeout=DDG_STAGE_SOFT_TIMEOUT,
            )
        except asyncio.TimeoutError:
            if debug:
                print(
                    f"[search] DDG stage soft timeout after {DDG_STAGE_SOFT_TIMEOUT:.0f}s; "
                    "continuing with Bing path"
                )
            ddg_results = []
        ddg_prefetched = ddg_results
        
        # If DDG gives good results, use them and skip slow Bing scraping
        if ddg_results and (
            len(ddg_results) >= ddg_min_results
            or _has_sufficient_target_hits(ddg_results, person_name, institution)
        ):
            if debug:
                print(f"[search] DuckDuckGo returned {len(ddg_results)} results, skipping Bing")
            
            prepared = _prepare_ranked_results(ddg_results)
            prepared = _prioritize_target_hits(prepared, person_name, institution)
            top_results = prepared[:num_results]
            if top_results and fetch_excerpts:
                await _enrich_with_page_excerpts(top_results, person_name, limit=min(EXCERPT_FETCH_LIMIT, len(top_results)))
            return top_results
            
    # Fallback to Bing if DDG failed or returned nothing
    # Fetch more results from HTML to account for filtering/deduplication
    fetch_count = min(target * 5, 150)
    if person_name:
        fetch_count = max(fetch_count, 100)
        
    # Try HTTPX first (faster than browser)
    html = await _fetch_with_httpx(query, count=fetch_count, debug=debug)
    
    # Only use browser if HTTPX fails (slow path)
    if not html:
        if debug:
            print("[search] HTTP fetch failed, trying browser (slow path)...")
        html = await _fetch_with_browser(query, count=fetch_count, debug=debug)

    if not html:
        # If both Bing methods failed and we haven't tried DDG yet (allow_fallback=False), return empty
        # If we already tried DDG (allow_fallback=True) and it failed, we also return empty
        return []

    results = _extract_results(
        html,
        institution=institution,
        limit=num_results * 3,
        person_name=person_name,
        debug=debug,
    )
    
    # If Bing returned results but they are poor, and we haven't tried DDG yet (unlikely with new logic),
    # or if we want to mix in DDG results (not implemented here to keep it simple).
    # Since we moved DDG to the front, we don't need the post-search fallback as much.
    
    prepared = _prepare_ranked_results(results)
    if ddg_prefetched:
        merged = _prepare_ranked_results(list(ddg_prefetched) + prepared)
        prepared = _prioritize_target_hits(merged, person_name, institution)
    else:
        prepared = _prioritize_target_hits(prepared, person_name, institution)

    top_results = prepared[:num_results]
    if top_results and fetch_excerpts:
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

    # Increased limits to ensure we capture true positives (fewer strategies, deeper depth)
    base_limit = max(5, min(8, per_strategy_limit + 2))

    strategies: List[QueryStrategy] = [
        # 1. Consolidated Basic (Catches general mentions)
        # Unquoted name allows for variations (e.g. "E. M. Purcell" -> "Edward Mills Purcell")
        # "Purdue University" is strong enough to anchor it.
        QueryStrategy(
            name="basic_consolidated",
            query=f'{clean_name} {clean_institution}',
            limit=base_limit + 2,
            boost=5,
        ),
    ]
    
    # NEW: Relaxed Name Variation (e.g. "Ben R. Mottelson" -> "Ben Mottelson")
    # This helps when the middle initial/name is present in input but missing in evidence
    components = _extract_name_components(clean_name)
    if components["first"] and components["last"] and (components["middle"] or components["initials"]):
        relaxed_name = f"{components['first'][0]} {components['last'][0]}"
        strategies.append(
            QueryStrategy(
                name="relaxed_name_variation",
                # Use quoted relaxed name + institution to force exact match on the simplified name
                query=f'"{relaxed_name}" {clean_institution}',
                limit=base_limit,
                boost=6,
            )
        )
    
    # 2. Site Search (Merged Unquoted/Quoted into one strong query)
    domain_guess = _institution_domain_guess(clean_institution)
    if domain_guess:
        site_filter = f"site:{domain_guess}"
        strategies.append(
            QueryStrategy(
                name="site_consolidated",
                query=f'"{clean_name}" {site_filter}',
                limit=base_limit + 4,
                boost=9, 
            )
        )
        
    # 3. Role & CV (Heavy hitting keywords for professional connection)
    strategies.append(
        QueryStrategy(
            name="role_evidence",
            query=f'"{clean_name}" {clean_institution} (professor OR faculty OR alumni OR resume OR CV OR visiting OR adjunct)',
            limit=base_limit,
            boost=8,
        )
    )
    
    # 4. Education & Early Career (Heavy hitting keywords for student/grad connection)
    strategies.append(
        QueryStrategy(
            name="education_evidence",
            query=f'"{clean_name}" {clean_institution} (degree OR B.S. OR B.A. OR PhD OR student OR graduated OR 19*)',
            limit=base_limit,
            boost=7,
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
    timeout: float = 45.0,
    fetch_excerpts: bool = True,
    ddg_min_results: int = 5,
) -> List[Dict[str, object]]:
    """Execute a single search strategy with reasonable timeout.
    
    Timeout is generous (45s) to allow legitimate slow responses while still
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
        # CRITICAL FIX: Re-enable DDG fallback for enhanced strategies
        # When Bing returns garbage results (e.g., score=1 for all results),
        # no amount of strategy variety will help - we need DDG as backup.
        results = await asyncio.wait_for(
            bing_search(
                query,
                num_results=strategy.limit,
                institution=institution,
                person_name=clean_person,
                debug=debug,
                allow_fallback=True,  # Re-enable DDG fallback to rescue garbage Bing results
                fetch_excerpts=fetch_excerpts,
                ddg_min_results=ddg_min_results,
                ensure_tokens=False,
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
    dataset_profile: str = None,
    fetch_excerpts: bool = True,
    ddg_min_results: int = 5,
) -> List[Dict[str, object]]:
    """Internal implementation of enhanced search without timeout wrapper.
    
    Args:
        name: Person's name
        institution: Institution to search for
        num_results: Number of results to return
        debug: Enable debug output
        dataset_profile: Dataset profile ("high_connection" or "low_connection")
    """
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
            fetch_excerpts=fetch_excerpts,
            ddg_min_results=ddg_min_results,
        )

    # NEW: Get early-exit threshold based on dataset profile
    from .config import get_early_exit_threshold
    early_exit_threshold = get_early_exit_threshold(dataset_profile)
    if debug:
        print(f"[search] Early exit threshold: {early_exit_threshold} (profile: {dataset_profile or 'default'})")

    # OPTIMIZATION: 4 concurrent strategies per name (matches consolidated list)
    sem = asyncio.Semaphore(4)

    async def run(strategy: QueryStrategy) -> List[Dict[str, object]]:
        async with sem:
            # 60s timeout allows thorough searching
            return await _execute_strategy(
                strategy,
                institution,
                clean_name,
                debug=debug,
                timeout=60.0,
                fetch_excerpts=fetch_excerpts,
                ddg_min_results=ddg_min_results,
            )

    tasks = [asyncio.create_task(run(strategy)) for strategy in strategies]
    combined: Dict[str, Dict[str, object]] = {}

    # OPTIMIZATION: Early termination - if we get good results quickly, don't wait for slow strategies
    # This is SAFE because we only terminate when we already have sufficient high-quality results
    early_termination_threshold = max(10, num_results // 2)  # e.g., 10 results for 20 requested
    high_quality_threshold = 15  # relevance score for "high quality"
    
    completed_strategies = 0
    successful_strategies = 0
    
    for coro in asyncio.as_completed(tasks):
        try:
            payload = await coro
            completed_strategies += 1
            
            if isinstance(payload, Exception):
                if debug:
                    print(f"[search] A strategy failed: {payload}")
                continue
            
            successful_strategies += 1
            
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
            
            # NEW: Check for early exit (low-connection datasets)
            # If first 2 SUCCESSFUL strategies return 0 results with score >= early_exit_threshold, exit
            # We use successful_strategies instead of completed_strategies to avoid exiting if strategies fail
            if successful_strategies >= 2 and dataset_profile and "low" in str(dataset_profile).lower():
                high_score_results = [
                    r for r in combined.values()
                    if r["signals"].get("relevance_score", 0) >= early_exit_threshold
                ]
                if len(high_score_results) == 0:
                    if debug:
                        print(f"[search] Early exit: No high-quality results (>={early_exit_threshold}) after {successful_strategies} successful strategies")
                    
                    # Cancel remaining tasks
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    break
            
            # Check for early termination: if we have enough high-quality results, stop waiting
            high_quality_results = [
                r for r in combined.values()
                if r["signals"].get("relevance_score", 0) >= high_quality_threshold
            ]
            
            # Terminate early if:
            # 1. We have at least 15 high-quality results OR
            # 2. We have 10+ results with at least 5 high-quality ones
            # 3. After completing at least 2-3 strategies (not 50%)
            if (completed_strategies >= 2 and 
                ((len(high_quality_results) >= 15) or 
                 (len(combined) >= 10 and len(high_quality_results) >= 5))):
                
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
            "semanticscholar.org", "scholar.google.com", "nobelprize.org",
            "biography.com"
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
            fetch_excerpts=fetch_excerpts,
            ddg_min_results=ddg_min_results,
        )

    top_results = ordered[:num_results]
    for idx, result in enumerate(top_results, start=1):
        result["rank"] = idx
    if fetch_excerpts:
        await _enrich_with_page_excerpts(top_results, clean_name, limit=min(EXCERPT_FETCH_LIMIT, len(top_results)))
    return top_results


ENHANCED_SEARCH_TIMEOUT = 90.0  # Maximum time for enhanced search to complete


async def enhanced_search(
    name: str,
    institution: str = "",
    num_results: int = 30,
    debug: bool = False,
    dataset_profile: str = None,
    fetch_excerpts: bool = True,
    ddg_min_results: int = 5,
) -> List[Dict[str, object]]:
    """Execute enhanced search with timeout protection.
    
    This wrapper ensures that enhanced_search never takes more than ENHANCED_SEARCH_TIMEOUT
    seconds, preventing cascading strategy timeouts from blocking the batch.
    If timeout occurs, falls back to basic search.
    """
    try:
        return await asyncio.wait_for(
            _enhanced_search_impl(
                name,
                institution,
                num_results,
                debug,
                dataset_profile,
                fetch_excerpts,
                ddg_min_results,
            ),
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
            fetch_excerpts=fetch_excerpts,
            ddg_min_results=ddg_min_results,
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
