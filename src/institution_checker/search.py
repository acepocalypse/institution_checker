import asyncio
import random
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional
from urllib.parse import quote, urlparse, urlsplit, urlunsplit, parse_qsl, urlencode

import httpx
from bs4 import BeautifulSoup

from .config import BROWSER_ARGS

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

CURRENT_TERMS = ["currently", "current", "now", "presently", "serving as", "now at", "active"]
PAST_TERMS = [
    "former",
    "previous",
    "previously",
    "retired",
    "emeritus",
    "alumni",
    "was at",
    "worked at",
    "served as",
    "left",
    "departed",
    "ex-",
]
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
MAX_HTTP_ATTEMPTS = 3
BROWSER_FETCH_ATTEMPTS = 2
BROWSER_RESULT_SELECTORS = [
    "li.b_algo",
    "li.b_ans",
    ".b_algo",
    ".b_entityTP",
    "[data-idx]",
]
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


def _normalise_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _quote_clause(text: str) -> str:
    cleaned = _normalise_whitespace(text)
    if not cleaned:
        return ""
    if not (cleaned.startswith('"') and cleaned.endswith('"')):
        cleaned = f'"{cleaned}"'
    return cleaned


def _contains_any(text: str, phrases: Iterable[str]) -> bool:
    return any(phrase in text for phrase in phrases)


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
    normalized = _normalise_whitespace(person_name).lower()
    if not normalized:
        return False
    tokens = [token for token in re.split(r"[^a-z]+", normalized) if token]
    if not tokens:
        return False
    if len(tokens) == 1:
        return tokens[0] in text
    first, last = tokens[0], tokens[-1]
    if first in text and last in text:
        return True
    pattern = re.compile(rf"{re.escape(first)}\s+[a-z]\.\s*{re.escape(last)}")
    if pattern.search(text):
        return True
    pattern = re.compile(rf"{re.escape(first)}\s+[a-z]+\s+{re.escape(last)}")
    if pattern.search(text):
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

    return {
        "has_person_name": bool(has_person),
        "has_institution": bool(has_institution),
        "has_academic_role": bool(has_academic_role),
        "career_transition": bool(has_transition),
        "has_current": bool(has_current),
        "has_past": bool(has_past),
        "has_recent_year": bool(has_recent_year),
        "has_timeline": bool(has_timeline),
        "domain": domain,
        "relevance_score": score,
    }


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
                for selector in BROWSER_RESULT_SELECTORS:
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
    url = _normalise_url(url)
    title = _normalise_whitespace(anchor.get_text(" ", strip=True))
    if not title:
        return None
    snippet_node = (
        element.select_one(".b_caption p")
        or element.select_one(".b_snippet")
        or element.select_one("p")
    )
    snippet = _normalise_whitespace(snippet_node.get_text(" ", strip=True) if snippet_node else "")
    if len(snippet) > 500:
        snippet = snippet[:497] + "..."
    signals = _compute_signals(title, snippet, url, institution, person_name)
    if person_name and not signals["has_person_name"]:
        return None
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
    for element in candidates:
        parsed = _parse_result_element(element, institution, person_name)
        if not parsed:
            continue
        url = parsed["url"]
        if url in seen_urls:
            continue
        seen_urls.add(url)
        results.append(parsed)
        if len(results) >= limit:
            break
    if not results:  # fallback to generic anchors if structured parsing failed
        for anchor in soup.select("a[href]"):
            url = anchor.get("href", "").strip()
            if not url.startswith(("http://", "https://")):
                continue
            url = _normalise_url(url)
            if url in seen_urls:
                continue
            title = _normalise_whitespace(anchor.get_text(" ", strip=True))
            if len(title) < 8:
                continue
            snippet = ""
            signals = _compute_signals(title, snippet, url, institution, person_name)
            if person_name and not signals["has_person_name"]:
                continue
            results.append({
                "title": title,
                "url": url,
                "snippet": snippet,
                "signals": signals,
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
    results.sort(key=lambda item: item["signals"]["relevance_score"], reverse=True)
    return results[:num_results]


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

    # OPTIMIZATION: Reduced from max(3, per_strategy_limit) to reduce redundancy
    base_limit = max(3, min(5, per_strategy_limit))

    strategies: List[QueryStrategy] = [
        QueryStrategy(
            name="core_profile",
            query=_compose_query(name_clause, institution_fragment),
            limit=base_limit,
            boost=5,
        ),
        QueryStrategy(
            name="current_status",
            query=_compose_query(
                name_clause,
                institution_fragment,
                '("currently" OR "now" OR "serving as" OR "presently" OR "active")',
            ),
            limit=base_limit,
            boost=4,
        ),
        QueryStrategy(
            name="institution_history",
            query=_compose_query(
                name_clause,
                institution_fragment,
                '("assistant professor" OR "associate professor" OR "faculty" OR "alumni" OR "student" OR "graduate")',
            ),
            limit=base_limit,
            boost=3,
        ),
        QueryStrategy(
            name="career_timeline",
            query=_compose_query(
                name_clause,
                institution_fragment,
                '("curriculum vitae" OR "cv" OR "biography" OR "resume" OR "profile")',
            ),
            limit=max(3, base_limit - 1),
            boost=2,
        ),
        QueryStrategy(
            name="transition",
            query=_compose_query(
                name_clause,
                institution_fragment,
                '("joined" OR "appointed" OR "moved to" OR "left" OR "hired" OR "transition" OR "departed")',
            ),
            limit=base_limit,
            boost=3,
        ),
    ]
    if clean_institution:
        strategies.insert(
            1,
            QueryStrategy(
                name="unquoted_institution",
                query="{quoted_name} {institution}",
                limit=base_limit,
                boost=4,
                ensure_tokens=False,
            ),
        )

    domain_guess = _institution_domain_guess(clean_institution)
    site_filter = f"site:{domain_guess}" if domain_guess else "site:.edu"
    strategies.append(
        QueryStrategy(
            name="directory",
            query=_compose_query(
                name_clause,
                institution_fragment,
                site_filter,
                '("faculty" OR "staff" OR "directory" OR "people" OR "profile")',
            ),
            limit=base_limit,
            boost=2,
        )
    )

    return strategies


def _annotate_signals(
    result: Dict[str, object],
    strategy: QueryStrategy,
    effective_query: str,
) -> Dict[str, object]:
    signals = dict(result["signals"])
    signals["strategy"] = strategy.name
    signals["strategy_boost"] = strategy.boost
    signals["queries"] = [effective_query]
    signals["strategies"] = [strategy.name]
    signals["strategy_hits"] = 1
    return {
        "title": result.get("title", ""),
        "url": result.get("url", ""),
        "snippet": result.get("snippet", ""),
        "signals": signals,
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
) -> List[Dict[str, object]]:
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
    results = await bing_search(
        query,
        num_results=strategy.limit,
        institution=institution,
        person_name=clean_person,
        debug=debug,
    )
    annotated: List[Dict[str, object]] = []
    for result in results:
        entry = _annotate_signals(result, strategy, query)
        annotated.append(entry)
    return annotated


async def enhanced_search(
    name: str,
    institution: str = "",
    num_results: int = 30,
    debug: bool = False,
) -> List[Dict[str, object]]:
    clean_name = _normalise_whitespace(name)
    if not clean_name:
        return []

    # OPTIMIZATION: Reduced from num_results // 2 to num_results // 3 to reduce redundancy
    per_strategy_limit = max(4, min(10, num_results // 3 if num_results > 0 else 4))
    strategies = _build_strategies(clean_name, institution, per_strategy_limit)
    if not strategies:
        return await bing_search(
            clean_name,
            num_results=num_results,
            institution=institution,
            person_name=clean_name,
            debug=debug,
        )

    sem = asyncio.Semaphore(4)

    async def run(strategy: QueryStrategy) -> List[Dict[str, object]]:
        async with sem:
            return await _execute_strategy(strategy, institution, clean_name, debug=debug)

    tasks = [asyncio.create_task(run(strategy)) for strategy in strategies]
    combined: Dict[str, Dict[str, object]] = {}

    results_per_strategy = await asyncio.gather(*tasks, return_exceptions=True)

    for strategy, payload in zip(strategies, results_per_strategy):
        if isinstance(payload, Exception):
            if debug:
                print(f"[search] Strategy '{strategy.name}' failed: {payload}")
            continue
        for result in payload:
            url = result.get("url")
            if not url:
                continue
            if url in combined:
                combined[url] = _merge_results(combined[url], result)
            else:
                combined[url] = result
        if debug:
            print(f"[search] Aggregated {len(combined)} results after '{strategy.name}'")

    ordered = sorted(
        combined.values(),
        key=lambda item: (
            item["signals"].get("relevance_score", 0),
            item["signals"].get("strategy_hits", 1),
            len(item.get("snippet", "")),
        ),
        reverse=True,
    )

    if not ordered:
        return await bing_search(
            clean_name,
            num_results=num_results,
            institution=institution,
            person_name=clean_name,
            debug=debug,
        )

    return ordered[:num_results]


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
