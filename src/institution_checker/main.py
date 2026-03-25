import asyncio
import os
import random
import re
import sys
import time
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import pandas as pd
from tqdm.auto import tqdm  # Auto-detects Jupyter notebooks vs terminal

from .config import (
    DATASET_PROFILE,
    DDG_MIN_RESULTS_DEFAULT,
    DDG_MIN_RESULTS_LOW_CONNECTION,
    DDG_MIN_RESULTS_VIP,
    INSTITUTION,
    JOINT_CAMPUS_PATTERNS,
    LLM_EXCERPT_LIMIT,
    VIP_EXCERPT_LIMIT,
    VIP_NAMES,
    VIP_ONLY_ENHANCED,
    VIP_RESCUE_ENABLED,
    VIP_RESCUE_QUERIES,
    VIP_RESCUE_RESULTS_PER_QUERY,
    VIP_SIGNAL_KEYWORDS,
)
from .llm_processor import analyze_connection, close_session, refresh_session, _build_error, _safe_text
from .search import (
    bing_search,
    close_search_clients,
    enhanced_search,
    cleanup_batch_resources,
    enrich_with_page_excerpts,
    force_browser_recreation,
    _fix_text_encoding,
)

DEFAULT_BATCH_SIZE = 20
DEFAULT_INPUT_PATH = "data/input_names.csv"
RESULTS_PATH = "data/results.csv"
PARTIAL_RESULTS_PATH = "data/results_partial.csv"
INTER_BATCH_DELAY = 0.5  # seconds between batches
NAME_TIMEOUT = 240.0  # maximum time allowed per name (balanced for accuracy)
SEARCH_TIMEOUT = 180.0  # maximum time for search phase (restored for better accuracy)
MAX_CONCURRENT_LLM_CALLS = int(os.getenv("INSTITUTION_CHECKER_MAX_CONCURRENT_LLM", "6"))  # tune down by default for API stability
MAX_CONCURRENT_SEARCHES = 8  # limit concurrent searches (increased to 8 for throughput)

STRONG_POSITIVE_SIGNAL_KEYWORDS: List[str] = [
    "faculty",
    "professor",
    "assistant professor",
    "associate professor",
    "professor emeritus",
    "emeritus",
    "distinguished professor",
    "instructor",
    "lecturer",
    "alumni",
    "alumnus",
    "alumna",
    "alum",
    "graduate",
    "graduated",
    "graduate studies",
    "undergraduate",
    "degree from",
    "earned degree",
    "bachelor of science",
    "master of science",
    "bachelor of arts",
    "master of arts",
    "bsee",
    "bsche",
    "bs",
    "b.s.",
    "ms",
    "m.s.",
    "msc",
    "phd",
    "ph.d",
    "ph d",
    "doctorate",
    "doctoral candidate",
    "doctoral student",
    "staff",
    "directory",
    "employee",
    "employed",
    "postdoc",
    "postdoctoral",
    "postdoctoral fellow",
    "postdoctoral fellowship",
    "researcher",
    "dean",
    "chair",
    "visiting professor",
    "visiting scholar",
    "distinguished fellow",
    "nobel",
    "nobel prize",
    "laureate",
]

KNOWN_FALSE_POSITIVE_SIGNAL_KEYWORDS: List[str] = [
    "conference",
    "speaker",
    "keynote",
    "gave a talk",
    "presented at",
    "seminar",
    "guest lecture",
    "workshop",
    # "award",  # Removed to prevent false negatives for award lists (e.g. Nobel)
    # "prize",  # Removed to prevent false negatives for award lists
    "honorary",
] + list(JOINT_CAMPUS_PATTERNS)

LOW_QUALITY_SOURCE_DOMAINS = {"prabook.com", "alchetron.com"}
LINKEDIN_DOMAIN = "linkedin.com"
HIGH_AUTHORITY_PATH_HINTS = [
    "/people/",
    "/faculty/",
    "/directory/",
    "/staff/",
    "/profiles/",
    "/profile/",
]


class NameProcessingTimeout(Exception):
    """Raised when processing a single name exceeds the allowed timeout."""

    def __init__(self, name: str, timeout: float = NAME_TIMEOUT):
        super().__init__(f"Timed out processing '{name}' after {timeout:.0f}s")
        self.name = name
        self.timeout = timeout


def load_names(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    data = pd.read_csv(path)
    if "name" not in data.columns:
        raise ValueError('CSV file must contain a "name" column')
    return [
        _fix_text_encoding(str(value).strip())
        for value in data["name"].dropna()
        if str(value).strip()
    ]


def _ensure_parent_dir(path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


async def process_name_search(name: str, use_enhanced_search: bool, debug: bool = False, dataset_profile: str = None) -> tuple[str, List[Dict[str, Any]]]:
    """Phase 1: Perform search for a name and return results.
    
    Applies SEARCH_TIMEOUT to ensure searches complete quickly even if individual
    strategies are slow. This prevents one slow search from blocking the entire batch.
    """
    search_start = time.time()
    vip_name = _is_vip_name(name)
    effective_profile = dataset_profile or DATASET_PROFILE
    if vip_name:
        ddg_min_results = DDG_MIN_RESULTS_VIP
    elif effective_profile == "low_connection":
        ddg_min_results = DDG_MIN_RESULTS_LOW_CONNECTION
    else:
        ddg_min_results = DDG_MIN_RESULTS_DEFAULT
    
    print(f"[PROGRESS] Starting search for: {name}")
    
    async def _do_search() -> tuple[str, List[Dict[str, Any]]]:
        """Inner function that performs the actual search."""
        try:
            # Cascading search strategy (OPTIMIZATION: try basic first to reduce redundancy)
            allow_enhanced = use_enhanced_search and (vip_name or not VIP_ONLY_ENHANCED)
            force_enhanced = allow_enhanced or vip_name
            if force_enhanced:
                if vip_name:
                    print(f"[PROGRESS] VIP override: forcing enhanced search for {name}")
                    results = await enhanced_search(
                        name,
                        INSTITUTION,
                        num_results=40,
                        debug=debug,
                        dataset_profile=effective_profile,
                        fetch_excerpts=False,
                        ddg_min_results=ddg_min_results,
                    )
                    search_elapsed = time.time() - search_start
                    print(f"[PROGRESS] Search completed for {name} in {search_elapsed:.1f}s, found {len(results)} results")
                    if not results:
                        print(f"[WARN] No search results found for {name}")
                    return name, results
                # Try basic search first - use flexible query without over-quoting
                # Request 20 results so after filtering/extraction we get ~15 for LLM
                # OPTIMIZATION: Unquote institution to find "Purdue" or "Purdue Univ" variants
                query = f'{name} {INSTITUTION}'
                print(f"[PROGRESS] Name: '{name}' | Query: '{query}'")
                results = await bing_search(
                    query,
                    institution=INSTITUTION,
                    person_name=name,
                    num_results=20,
                    debug=debug,
                    fetch_excerpts=False,
                    ddg_min_results=ddg_min_results,
                )
                
                # If we got results, show what they are
                if results:
                    print(f"[PROGRESS] Result sample: {results[0]['title'][:80] if results[0].get('title') else 'N/A'}")
                
                # OPTIMIZATION: Check for name variations if strict search failed
                # e.g. "Ben R. Mottelson" might fail, but "Ben Mottelson" might succeed
                if not results and len(name.split()) > 2:
                    parts = name.split()
                    # Construct relaxed name (First + Last)
                    relaxed_name = f"{parts[0]} {parts[-1]}"
                    relaxed_query = f'{relaxed_name} {INSTITUTION}'
                    print(f"[PROGRESS] Strict search failed. Trying relaxed query: '{relaxed_query}'")
                    
                    try:
                        results = await bing_search(
                            relaxed_query,
                            institution=INSTITUTION,
                            person_name=relaxed_name,
                            num_results=20,
                            debug=debug,
                            fetch_excerpts=False,
                            ddg_min_results=ddg_min_results,
                        )
                        if results:
                            print(f"[PROGRESS] Relaxed search found {len(results)} results")
                    except Exception as e:
                        print(f"[WARN] Relaxed search failed: {e}")
                
                # OPTIMIZATION: "Quick Site Check" fallback for stubborn negatives
                # If both strict and relaxed basic searches failed (0 results), try ONE targeted site search.
                # This is much faster than full enhanced_search but safer than giving up immediately.
                if not results:
                    from .search import _institution_domain_guess
                    domain = _institution_domain_guess(INSTITUTION) or "purdue.edu"
                    site_query = f'site:{domain} "{name}"'
                    print(f"[PROGRESS] Basic search failed. Last resort quick check: '{site_query}'")
                    try:
                        results = await bing_search(
                            site_query,
                            institution=INSTITUTION,
                            person_name=name,
                            num_results=10,
                            debug=debug,
                            fetch_excerpts=False,
                            ddg_min_results=ddg_min_results,
                        )
                        if results:
                            print(f"[PROGRESS] Quick site check found {len(results)} results")
                    except Exception as e:
                        print(f"[WARN] Quick site check failed: {e}")

                # If basic search returns very few results OR low quality results, escalate to enhanced
                # Enhanced search tries multiple query strategies to catch real connections
                max_score = max((r.get('signals', {}).get('relevance_score', 0) for r in results), default=0)
                
                should_escalate = False
                if len(results) < 5:
                    should_escalate = True
                if max_score < 5:
                    should_escalate = True
                
                # Check for VIP/critical keywords in basic results that should force escalation
                has_vip_signals = _has_vip_signals(results)

                # NOTE: Removed "Strong Match Skip" optimization because it caused false negatives
                # (e.g. Wolfgang Pauli found only a library book with high score, failing LLM check).
                # We need Deep Search to find the *relationship* evidence (e.g. "Visiting Professor").

                # Force enhanced search for high_connection profile to ensure max recall
                # CRITICAL OPTIMIZATION: Smart Escalation
                # Only run expensive enhanced search if there is a REASON to suspect a connection.
                # If basic search found essentially nothing (Score < 8) and no keywords, trust the negative.
                if effective_profile == "high_connection":
                     # Condition 1: Critical keywords found (ALWAYS escalate)
                     if has_vip_signals:
                         should_escalate = True
                     
                     # Condition 2: Decent relevance score (likely a name match at the institution)
                     elif max_score >= 8:
                         should_escalate = True
                         
                     # Condition 3: Basic search failure (0 results or very weak noise) -> DO NOT ESCALATE
                     # This is the massive speed win for the 95% of random names.
                     else:
                         if debug:
                             print(f"[OPTIMIZATION] Skipping enhanced search (Max Score {max_score} < 8, No Keywords). Trusting negative.")
                         should_escalate = False

                # OPTIMIZATION for low_connection:
                # If basic search found very few results, it's very likely a true negative.
                # Don't waste time on enhanced search unless we have a reason to suspect a connection.
                if effective_profile == "low_connection" and not has_vip_signals:
                    if len(results) <= 1:
                        print(f"[OPTIMIZATION] Basic search found <= 1 result. Skipping enhanced search (low_connection profile).")
                        should_escalate = False
                    elif len(results) <= 3 and max_score < 5:
                        print(f"[OPTIMIZATION] Basic search found weak results (count={len(results)}, score={max_score}). Skipping enhanced search.")
                        should_escalate = False
                
                if should_escalate:
                    reason = "few results" if len(results) < 5 else f"low quality (max_score={max_score})"
                    if allow_enhanced or has_vip_signals:
                        print(f"[PROGRESS] Basic search returned {len(results)} results, {reason}, escalating to enhanced search...")
                        results = await enhanced_search(
                            name,
                            INSTITUTION,
                            num_results=30,
                            debug=debug,
                            dataset_profile=effective_profile,
                            fetch_excerpts=False,
                            ddg_min_results=ddg_min_results,
                        )
                    else:
                        print(f"[PROGRESS] Skipping enhanced search (non-VIP fast path).")
                else:
                    print(f"[PROGRESS] Basic search returned {len(results)} results with max_score={max_score}, sufficient for analysis")
            else:
                query = f'{name} {INSTITUTION}'
                results = await bing_search(
                    query,
                    institution=INSTITUTION,
                    person_name=name,
                    num_results=25,
                    debug=debug,
                    fetch_excerpts=False,
                    ddg_min_results=ddg_min_results,
                )
            
            search_elapsed = time.time() - search_start
            print(f"[PROGRESS] Search completed for {name} in {search_elapsed:.1f}s, found {len(results)} results")
            
            if not results:
                print(f"[WARN] No search results found for {name}")
            
            return name, results
            
        except Exception as e:
            elapsed = time.time() - search_start
            print(f"[ERROR] Search failed for {name} after {elapsed:.1f}s: {e}")
            raise
    
    # Apply timeout to search to fail fast if it's taking too long
    try:
        return await asyncio.wait_for(_do_search(), timeout=SEARCH_TIMEOUT)
    except asyncio.TimeoutError:
        elapsed = time.time() - search_start
        print(f"[ERROR] Search timed out for {name} after {elapsed:.1f}s (limit: {SEARCH_TIMEOUT}s)")
        raise TimeoutError(f"Search exceeded {SEARCH_TIMEOUT}s timeout")


async def process_name_llm(name: str, results: List[Dict[str, Any]], debug: bool = False, dataset_profile: str = None) -> Dict[str, str]:
    """Phase 2: Analyze search results with LLM."""
    llm_start = time.time()
    print(f"[PROGRESS] Starting LLM analysis for: {name}")
    
    try:
        vip_mode = _is_vip_name(name) or _has_vip_signals(results)

        # QUALITY CHECK: Count high-quality results before sending to LLM
        # Use dynamic threshold based on dataset profile
        from .config import get_skip_threshold
        
        # Determine threshold: use configured skip threshold
        # FIX: For high_connection (threshold 0), use a lower floor (4) to capture weak signals like "Visiting Professor"
        # For low_connection (threshold 8), keep the floor at 8 to filter garbage
        configured_threshold = get_skip_threshold(dataset_profile)
        if configured_threshold < 5:
            threshold = 4
        else:
            threshold = max(8, configured_threshold)
        
        high_quality_count = sum(
            1 for r in results 
            if r.get('signals', {}).get('relevance_score', 0) >= threshold
        )
        
        # If we have < 1 high-quality results after filtering, skip LLM call
        # The LLM will receive "(no search results available)" which wastes API calls
        if high_quality_count < 1 and not vip_mode:
            print(f"[QUALITY-SKIP] Skipping LLM for {name}: only {high_quality_count} high-quality results (threshold: {threshold})")
            return _build_immediate_not_connected(
                name, 
                INSTITUTION, 
                f"Insufficient high-quality search results ({high_quality_count} results with score >= {threshold})"
            )

        excerpt_limit = VIP_EXCERPT_LIMIT if vip_mode else LLM_EXCERPT_LIMIT
        if results and excerpt_limit > 0:
            await enrich_with_page_excerpts(results, name, limit=min(excerpt_limit, len(results)))
        
        decision = await analyze_connection(name, INSTITUTION, results, debug=debug, vip_mode=vip_mode)

        if vip_mode and VIP_RESCUE_ENABLED and decision.get("verdict") != "connected":
            rescue_results = await _vip_rescue_search(
                name,
                INSTITUTION,
                ddg_min_results=ddg_min_results,
                debug=debug,
            )
            if rescue_results:
                merged_results = _dedupe_by_url(list(results) + rescue_results)
                await enrich_with_page_excerpts(
                    merged_results,
                    name,
                    limit=min(VIP_EXCERPT_LIMIT, len(merged_results)),
                )
                decision = await analyze_connection(
                    name,
                    INSTITUTION,
                    merged_results,
                    debug=debug,
                    vip_mode=True,
                )
        
        # Defensive check: ensure decision is not None
        if decision is None:
            raise RuntimeError("analyze_connection returned None")
        
        llm_elapsed = time.time() - llm_start
        print(f"[PROGRESS] LLM analysis completed for {name} in {llm_elapsed:.1f}s")
        
        payload = {"name": name, "institution": INSTITUTION}
        payload.update(decision)
        return payload
        
    except Exception as e:
        elapsed = time.time() - llm_start
        print(f"[ERROR] LLM analysis failed for {name} after {elapsed:.1f}s: {e}")
        raise


async def process_name(name: str, use_enhanced_search: bool, debug: bool = False, dataset_profile: str = None) -> Dict[str, str]:
    """Process a name with focused error handling - rely on component-level robustness.
    
    This is kept for backward compatibility and simple single-name processing.
    For batch processing, use the split phase approach in process_batch().
    """
    start = time.time()
    
    try:
        # Use the split-phase approach for consistency
        result_name, results = await process_name_search(name, use_enhanced_search, debug=debug, dataset_profile=dataset_profile)
        decision = await process_name_llm(result_name, results, debug=debug, dataset_profile=dataset_profile)
        
        payload = {"name": name, "institution": INSTITUTION}
        payload.update(decision)
        return payload
        
    except Exception as e:
        elapsed = time.time() - start
        print(f"[ERROR] Failed to process {name} after {elapsed:.1f}s: {e}")
        raise


async def _process_name_with_timeout(name: str, use_enhanced_search: bool, debug: bool = False) -> Dict[str, str]:
    async def _run(selected_enhanced: bool) -> Dict[str, str]:
        return await asyncio.wait_for(
            process_name(name, selected_enhanced, debug=debug),
            timeout=NAME_TIMEOUT,
        )

    try:
        return await _run(use_enhanced_search)
    except asyncio.TimeoutError as exc:  # pragma: no cover - guarded by runtime behaviour
        if use_enhanced_search:
            print(f"[WARN] {name}: enhanced search timed out after {NAME_TIMEOUT:.0f}s, retrying with basic search...")
            try:
                return await _run(False)
            except asyncio.TimeoutError as fallback_exc:  # pragma: no cover - guarded by runtime behaviour
                raise NameProcessingTimeout(name, NAME_TIMEOUT) from fallback_exc
        raise NameProcessingTimeout(name, NAME_TIMEOUT) from exc


def _build_immediate_not_connected(name: str, institution: str, reason: str) -> Dict[str, str]:
    """Build an immediate 'not_connected' result without LLM call.
    
    Used ONLY for obvious cases where we can skip LLM to save API quota.
    This should be called very rarely - only when results are completely off-topic.
    
    ACCURACY NOTE: Being too aggressive with skips causes false negatives.
    The conservative skip policy ensures we maintain 95%+ accuracy while
    still achieving scalability through batching and parallelization.
    """
    return {
        "name": name,
        "institution": institution,
        "verdict": "not_connected",
        "connected": "N",
        "relationship_type": "None",
        "relationship_timeframe": "unknown",
        "verification_detail": reason,
        "summary": reason,
        "primary_source": "",
        "confidence": "low",
        "verification_status": "needs_review",
        "temporal_context": "N/A",
        "connection_type": "None",
        "connection_detail": reason,
        "current_or_past": "N/A",
        "supporting_url": "",
        "temporal_evidence": "N/A",
    }


def _aggregate_result_text(result: Dict[str, Any]) -> str:
    parts = [
        _safe_text(result.get("title", "")),
        _safe_text(result.get("snippet", "")),
        _safe_text(result.get("page_excerpt", "")),
        _safe_text(result.get("description", "")),
        _safe_text(result.get("url", "")),
    ]
    combined = " ".join(part for part in parts if part)
    return combined.lower()


_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_MULTISPACE_RE = re.compile(r"\s+")


def _normalize_signal_text(text: str) -> Tuple[str, str, str]:
    lowered = text.lower()
    spaced = _MULTISPACE_RE.sub(
        " ",
        _NON_ALNUM_RE.sub(" ", lowered)
    ).strip()
    compact = spaced.replace(" ", "")
    return lowered, spaced, compact


@lru_cache(maxsize=512)
def _keyword_forms(keyword: str) -> Tuple[str, str, str]:
    lowered = keyword.lower().strip()
    spaced = _MULTISPACE_RE.sub(
        " ",
        _NON_ALNUM_RE.sub(" ", lowered)
    ).strip()
    compact = spaced.replace(" ", "")
    return lowered, spaced, compact


def _keyword_matches_text(keyword: str, lowered: str, spaced: str, compact: str) -> bool:
    base, spaced_kw, compact_kw = _keyword_forms(keyword)
    if base and base in lowered:
        return True
    if spaced_kw and spaced_kw in spaced:
        return True
    if compact_kw and compact_kw in compact:
        return True
    return False


def _normalize_name_key(value: str) -> str:
    return _MULTISPACE_RE.sub(" ", _safe_text(value).strip().lower())


_VIP_NAME_LOOKUP = {_normalize_name_key(name) for name in VIP_NAMES if _safe_text(name).strip()}


def _is_vip_name(name: str) -> bool:
    if not name or not _VIP_NAME_LOOKUP:
        return False
    return _normalize_name_key(name) in _VIP_NAME_LOOKUP


def _has_vip_signals(results: List[Dict[str, Any]]) -> bool:
    if not results or not VIP_SIGNAL_KEYWORDS:
        return False
    for result in results:
        text = _aggregate_result_text(result)
        lowered, spaced, compact = _normalize_signal_text(text)
        if any(_keyword_matches_text(k, lowered, spaced, compact) for k in VIP_SIGNAL_KEYWORDS):
            return True
    return False


def _dedupe_by_url(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for result in results:
        url = _safe_text(result.get("url"))
        if not url or url in seen:
            continue
        seen.add(url)
        deduped.append(result)
    return deduped


async def _vip_rescue_search(
    name: str,
    institution: str,
    *,
    ddg_min_results: int,
    debug: bool = False,
) -> List[Dict[str, Any]]:
    if not VIP_RESCUE_QUERIES:
        return []
    combined: List[Dict[str, Any]] = []
    for template in VIP_RESCUE_QUERIES:
        query = template.format(name=name)
        try:
            results = await bing_search(
                query,
                institution=institution,
                person_name=name,
                num_results=VIP_RESCUE_RESULTS_PER_QUERY,
                debug=debug,
                fetch_excerpts=False,
                ddg_min_results=ddg_min_results,
                ensure_tokens=False,
            )
        except Exception as exc:
            if debug:
                print(f"[VIP-RESCUE] Query failed: {query} ({exc})")
            continue
        combined.extend(results)
    return _dedupe_by_url(combined)


def _score_source_authority(url: str) -> Tuple[int, List[str]]:
    if not url:
        return 0, []
    
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    path = parsed.path.lower()
    if not domain:
        return 0, []
    
    score = 0
    factors: List[str] = []
    
    if domain.endswith(".edu"):
        score += 5
        factors.append(f".edu:{domain}")
    
    if "purdue.edu" in domain and any(hint in path for hint in HIGH_AUTHORITY_PATH_HINTS):
        score += 10
        factors.append(f"authority:{domain}{path}")
    
    for low_domain in LOW_QUALITY_SOURCE_DOMAINS:
        if low_domain in domain:
            score -= 5
            factors.append(f"low_quality:{low_domain}")
            break
    
    if LINKEDIN_DOMAIN in domain:
        score -= 2
        factors.append("self_report:linkedin")
    
    return score, factors


def should_skip_llm(
    results: List[Dict[str, Any]],
    dataset_profile: str = None,
    name: Optional[str] = None,
) -> tuple[bool, Optional[str]]:
    """Heuristic triage that mirrors pre-LLM false-positive detection.
    
    Args:
        results: Search results to evaluate
        dataset_profile: Dataset profile for adaptive filtering ("high_connection" or "low_connection")
        name: Optional name for VIP override
    
    Returns:
        (should_skip: bool, reason: Optional[str])
    """
    from .config import get_filtering_mode, get_skip_threshold
    
    filtering_mode = get_filtering_mode(dataset_profile)
    skip_threshold = get_skip_threshold(dataset_profile)

    vip_mode = _is_vip_name(name) or _has_vip_signals(results)
    if vip_mode:
        return False, None
    
    if not results:
        return True, "No search results found"
    
    total_signal_score = 0
    total_authority_score = 0
    positive_matches: set[str] = set()
    negative_matches: set[str] = set()
    authority_markers: List[str] = []
    purdue_mentions = 0
    purdue_domain_hits = 0
    institution_signal_hits = 0
    max_relevance_score = 0
    has_explicit_connection = False
    
    for result in results:
        text = _aggregate_result_text(result)
        lowered, spaced, compact = _normalize_signal_text(text)
        if "purdue" in lowered or "purdue" in spaced:
            purdue_mentions += 1
        
        for keyword in STRONG_POSITIVE_SIGNAL_KEYWORDS:
            if keyword and _keyword_matches_text(keyword, lowered, spaced, compact):
                total_signal_score += 10
                positive_matches.add(keyword)
        
        for keyword in KNOWN_FALSE_POSITIVE_SIGNAL_KEYWORDS:
            if keyword and _keyword_matches_text(keyword, lowered, spaced, compact):
                total_signal_score -= 10
                negative_matches.add(keyword)
        
        url = result.get("url") or ""
        url_lower = url.lower()
        if "purdue.edu" in url_lower:
            purdue_domain_hits += 1

        signals = result.get("signals") or {}
        if signals.get("has_institution"):
            institution_signal_hits += 1
        if signals.get("has_explicit_connection"):
            has_explicit_connection = True
        
        # Track max relevance score
        relevance_score = signals.get("relevance_score", 0)
        max_relevance_score = max(max_relevance_score, relevance_score)

        authority_score, factors = _score_source_authority(url)
        total_authority_score += authority_score
        if factors:
            authority_markers.extend(factors)
    
    # ADAPTIVE FILTERING based on dataset profile
    if filtering_mode == "aggressive":
        # Low-connection datasets (e.g., Nobel laureates): Skip MUCH more aggressively
        # to reduce wasted LLM API calls from ~200 to ~50 for datasets with <2% true connections
        
        # Check for critical keywords that should always trigger LLM review
        # NOTE: Removed "nobel" to allow unrelated laureates to skip
        has_critical_keywords = False 
        
        # SAFETY: Only bypass if we have overwhelming keyword matches
        if purdue_mentions >= 4 and institution_signal_hits >= 2:
             # Let LLM verify - very dense mentions of Purdue
             return False, None
        
        # Tier 1: Skip if below threshold (score < 8) with NO strong signals
        if max_relevance_score < skip_threshold:
            has_strong_signals = (
                (purdue_domain_hits > 0 and has_explicit_connection) or 
                total_signal_score > 30
            )
            if not has_strong_signals:
                return True, f"Aggressive filter (Tier 1): max_relevance_score={max_relevance_score} < {skip_threshold}, no strong rescue signals"
        
        # Tier 2: Skip if score is marginal (8-15) without excellent evidence
        # Score 8 ~ 15 = name + institution mention + weak signal, often garbage
        # Increased threshold to 15 to catch more "loose" connections
        if 8 <= max_relevance_score < 15:
            has_excellent_evidence = (
                purdue_domain_hits >= 2 or  # Multiple .edu results = authoritative
                (purdue_domain_hits >= 1 and has_explicit_connection) or  # .edu + explicit = strong
                total_signal_score >= 40 or # Very high signal score = multiple strong keywords
                (has_explicit_connection and max_relevance_score >= 12) # Explicit + decent score
            )
            if not has_excellent_evidence:
                return True, f"Aggressive filter (Tier 2): max_relevance_score={max_relevance_score} is marginal, lacks excellent evidence (low-connection dataset)"
        
        # Tier 3: Skip if only weak positive signals (< 10) without institutional backing
        # No .edu domain AND weak relevance = very likely false positive (citations, lists, etc.)
        # CRITICAL: Keep threshold at 10 (not 15) to avoid skipping genuine connections with modest initial scores
        if max_relevance_score < 10 and purdue_domain_hits == 0:
            if not has_explicit_connection and total_signal_score < 10:
                # Only skip if BOTH no explicit connection AND low signal score
                return True, f"Aggressive filter (Tier 3): max_relevance_score={max_relevance_score}, no .edu domain, no explicit connection, low signals (low-connection dataset)"
        
        # Tier 4: CONFIDENT NEGATIVE - Ultra-safe patterns to catch obvious non-connections
        # Goal: Improve skip rate from 80% to 95%+ while maintaining 100% recall
        
        # Pattern A: Dominant negative keywords with NO positive signals
        # Example: Conference speaker, prize winner with no employment keywords
        if (total_signal_score <= -30 and 
            len(positive_matches) == 0 and 
            purdue_domain_hits == 0):
            return True, f"Confident negative (Tier 4-A): dominant negative signals (score={total_signal_score}), no positive keywords, no .edu domain"
        
        # Pattern B: Only low-quality sources with weak relevance
        # Example: prabook.com, alchetron.com mentions with no authoritative sources
        if (max_relevance_score < 10 and
            total_authority_score <= -5 and
            purdue_domain_hits == 0 and
            not has_explicit_connection):
            return True, f"Confident negative (Tier 4-B): only low-quality sources (authority={total_authority_score}), score={max_relevance_score}, no .edu, no explicit connection"
        
        # Pattern C: Virtually no institution mention at all
        # Example: Person name appears but institution barely mentioned (citation list)
        # FIX: Lowered threshold from 3 to 1 to avoid skipping soft matches (score=1) that might be valid
        if (purdue_mentions == 0 and 
            institution_signal_hits == 0 and 
            max_relevance_score < 1):
            return True, f"Confident negative (Tier 4-C): no institution mentions, score={max_relevance_score} (virtually zero relevance)"
    
    # Conservative policy: only skip completely off-topic results
    # For high_connection profile, use SMART filtering - skip when NO evidence exists
    if filtering_mode == "conservative":
        # SMART SKIP FOR HIGH_CONNECTION: Skip when there's genuinely NO Purdue evidence
        # This balances accuracy (don't miss real connections) with efficiency (don't waste LLM calls)
        
        # Skip Pattern 1: Zero relevance - no institution mention at all
        if max_relevance_score == 0 and purdue_mentions == 0 and institution_signal_hits == 0:
            return True, f"Smart skip: zero relevance (no institution signals found)"
        
        # Skip Pattern 2: Very low relevance with no Purdue mentions
        # Score < 3 means barely any match, and if Purdue isn't even mentioned, it's not relevant
        if max_relevance_score < 3 and purdue_mentions == 0 and purdue_domain_hits == 0:
            return True, f"Smart skip: very low relevance (score={max_relevance_score}, no Purdue mentions)"
        
        # Skip Pattern 3: Low relevance with ONLY negative signals
        # If we have some results but they're all negative context (conference, speaker, etc.)
        if (max_relevance_score < 5 and 
            total_signal_score < 0 and 
            not positive_matches and
            purdue_domain_hits == 0):
            return True, f"Smart skip: low relevance with negative context (score={max_relevance_score}, signal={total_signal_score})"
        
        # Skip Pattern 4: No authoritative sources and weak signals
        # If there's no .edu domain and relevance is marginal, likely a false lead
        if (max_relevance_score < 6 and
            purdue_domain_hits == 0 and
            purdue_mentions <= 1 and
            total_signal_score <= 0 and
            not has_explicit_connection):
            return True, f"Smart skip: weak evidence without authority (score={max_relevance_score}, no .edu, mentions={purdue_mentions})"
        
        # Otherwise, send to LLM for analysis
    else:
        # LOW CONNECTION MODE: Original stricter filtering
        if (
            total_signal_score < 0
            and not positive_matches
            and purdue_mentions == 0
            and purdue_domain_hits == 0
            and institution_signal_hits == 0
        ):
            negatives_summary = ", ".join(sorted(negative_matches)) if negative_matches else "negative context"
            return True, (
                f"Heuristic triage: negative evidence dominates "
                f"(signal score {total_signal_score}, negatives: {negatives_summary})"
            )
        
        if (
            total_signal_score == 0
            and total_authority_score < 5
            and not positive_matches
            and purdue_mentions == 0
            and purdue_domain_hits == 0
            and institution_signal_hits == 0
        ):
            unique_markers = sorted(set(authority_markers))
            marker_summary = f"; authority markers: {', '.join(unique_markers[:3])}" if unique_markers else ""
            return True, (
                f"Heuristic triage: no meaningful evidence "
                f"(signal score {total_signal_score}, authority score {total_authority_score}){marker_summary}"
            )

    if total_signal_score > 20 and total_authority_score > 10:
        return False, None
    
    return False, None


def print_result_summary(result: Dict[str, str]) -> None:
    name = result.get("name", "")
    verdict = result.get("verdict", "uncertain")
    confidence = result.get("confidence", "medium")
    summary = result.get("summary") or result.get("verification_detail") or "No explanation provided"
    relationship_type = result.get("relationship_type") or result.get("connection_type", "Other")
    timeframe = result.get("relationship_timeframe") or result.get("current_or_past", "unknown")
    verification_status = result.get("verification_status", "needs_review")

    if verdict == "connected":
        print(f"[OK] {name}: {relationship_type} ({timeframe}, {confidence}, {verification_status}) - {summary}")
    elif verdict == "not_connected":
        print(f"[--] {name}: no verified connection ({confidence}, {verification_status}) - {summary}")
    else:
        print(f"[??] {name}: inconclusive ({confidence}, {verification_status}) - {summary}")


def has_error(result: Dict[str, str]) -> bool:
    """Check if a result contains an error in any field."""
    # Check for explicit error markers in summary-related fields
    error_fields = [
        result.get("temporal_context", ""),
        result.get("temporal_evidence", ""),
        result.get("verification_detail", ""),
        result.get("connection_detail", ""),
        result.get("summary", ""),
    ]
    for raw in error_fields:
        value = str(raw or "").strip()
        if not value:
            continue
        if value.startswith("Error:") or "Processing error:" in value:
            return True

    confidence = result.get("confidence", "").strip().lower()
    if confidence not in {"high", "medium", "low"}:
        return True

    verdict = result.get("verdict", "").strip()
    relationship_timeframe = result.get("relationship_timeframe") or result.get("current_or_past", "")

    if verdict == "connected":
        detail = str(result.get("verification_detail") or result.get("connection_detail") or "").strip()
        summary = str(result.get("summary") or "").strip()
        if len(detail) < 5 and len(summary) < 5:
            return True
        if relationship_timeframe not in {"current", "past", "unknown"}:
            return True
    elif verdict == "not_connected":
        if relationship_timeframe not in {"unknown", "N/A", ""}:
            return True

    return False


def merge_results(original: List[Dict[str, str]], retried: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Merge retried results back into original, replacing failed records."""
    retry_lookup = {r["name"]: r for r in retried}
    merged = []
    for record in original:
        name = record.get("name", "")
        if name in retry_lookup:
            merged.append(retry_lookup[name])
        else:
            merged.append(record)
    return merged


async def process_batch(names: List[str], use_enhanced_search: bool, debug: bool = False, dataset_profile: str = None) -> List[Dict[str, str]]:
    """Process a batch of names with parallel search and LLM phases for optimal performance.
    
    Args:
        names: List of names to process
        use_enhanced_search: Whether to use enhanced search
        debug: Enable debug output
        dataset_profile: Dataset profile for adaptive filtering ("high_connection" or "low_connection")
    """
    print(f"[BATCH] Processing {len(names)} names: {', '.join(names)}")
    batch_start = time.time()

    # ===== PHASE 1: Search (all in parallel) =====
    print(f"[BATCH] Phase 1: Running searches in parallel for all {len(names)} names (max {MAX_CONCURRENT_SEARCHES} concurrent)")
    search_phase_start = time.time()
    
    # Create semaphore to limit concurrent searches
    search_semaphore = asyncio.Semaphore(MAX_CONCURRENT_SEARCHES)

    async def search_with_semaphore(name: str):
        # Add random jitter to prevent thundering herd
        # Reduced jitter (0.5-2s) as DDG is prioritized
        await asyncio.sleep(random.uniform(0.5, 2.0))
        async with search_semaphore:
            # Use process_name_search which has its own internal timeout (SEARCH_TIMEOUT)
            # We do NOT wrap this in an outer timeout because queuing time might be significant
            return await process_name_search(name, use_enhanced_search, debug=debug, dataset_profile=dataset_profile)

    search_coroutines = [
        search_with_semaphore(name)
        for name in names
    ]
    search_outcomes = await asyncio.gather(*search_coroutines, return_exceptions=True)
    
    search_phase_elapsed = time.time() - search_phase_start
    print(f"[BATCH] Phase 1 completed in {search_phase_elapsed:.1f}s")
    
    # Process search results
    search_results: Dict[str, List[Dict[str, Any]]] = {}
    failed_searches: Dict[str, Exception] = {}
    
    for name, outcome in zip(names, search_outcomes):
        if isinstance(outcome, BaseException):
            error_msg = str(outcome)
            if "timeout" in error_msg.lower():
                print(f"[BATCH] Search timed out for {name}: {error_msg}")
            else:
                print(f"[BATCH] Search failed for {name}: {error_msg}")
            failed_searches[name] = outcome
            search_results[name] = []  # Empty results for failed searches
        else:
            result_name, results = outcome
            search_results[result_name] = results
            print(f"[BATCH] Search succeeded for {name}: {len(results)} results")
    
    # ===== PHASE 2: LLM Analysis (with early termination for obvious cases) =====
    print(f"[BATCH] Phase 2: Evaluating search results and running LLM analysis in parallel for all {len(names)} names (max {MAX_CONCURRENT_LLM_CALLS} concurrent)")
    llm_phase_start = time.time()
    
    # First pass: identify which names need LLM vs which can skip it
    names_needing_llm: List[str] = []
    skipped_results: Dict[str, Dict[str, str]] = {}
    skipped_count = 0
    
    for name in names:
        results = search_results.get(name, [])
        should_skip, skip_reason = should_skip_llm(results, dataset_profile=dataset_profile, name=name)
        
        if should_skip:
            skipped_result = _build_immediate_not_connected(name, INSTITUTION, skip_reason or "No results found")
            skipped_results[name] = skipped_result
            skipped_count += 1
            # Enhanced logging: Show which tier/pattern triggered the skip
            tier_marker = ""
            if "Tier 1" in skip_reason:
                tier_marker = "[T1]"
            elif "Tier 2" in skip_reason:
                tier_marker = "[T2]"
            elif "Tier 3" in skip_reason:
                tier_marker = "[T3]"
            elif "Tier 4" in skip_reason or "Confident negative" in skip_reason:
                tier_marker = "[T4]"
            elif "negative evidence dominates" in skip_reason:
                tier_marker = "[NEG]"
            else:
                tier_marker = "[SKIP]"
            print(f"[BATCH] {tier_marker} Skipping LLM for {name}: {skip_reason}")
        else:
            names_needing_llm.append(name)
    
    if skipped_count > 0:
        print(f"[BATCH] Skipped LLM for {skipped_count} name(s) with obvious non-connections, {len(names_needing_llm)} names need LLM")
    
    # Create semaphore to limit concurrent LLM calls
    llm_semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM_CALLS)
    
    async def llm_with_semaphore(name: str, results: List[Dict[str, Any]]) -> Dict[str, str]:
        async with llm_semaphore:
            return await asyncio.wait_for(
                process_name_llm(name, results, debug=debug, dataset_profile=dataset_profile),
                timeout=NAME_TIMEOUT
            )
    
    llm_coroutines = [
        llm_with_semaphore(name, search_results.get(name, []))
        for name in names_needing_llm
    ]
    llm_outcomes = await asyncio.gather(*llm_coroutines, return_exceptions=True)
    
    llm_phase_elapsed = time.time() - llm_phase_start
    print(f"[BATCH] Phase 2 completed in {llm_phase_elapsed:.1f}s ({len(names_needing_llm)} LLM calls, {skipped_count} skipped)")
    
    # ===== Combine results =====
    ordered_results: List[Dict[str, str]] = []
    completed_count = 0
    
    # First add skipped results (they're already complete)
    for name in names:
        if name in skipped_results:
            ordered_results.append(skipped_results[name])
            completed_count += 1
    
    # Then add LLM results in order
    llm_outcome_map: Dict[str, Any] = dict(zip(names_needing_llm, llm_outcomes))
    
    for name in names:
        if name in skipped_results:
            continue  # Already added above
        
        llm_outcome = llm_outcome_map.get(name)
        completed_count += 1
        elapsed = time.time() - batch_start
        
        # Check if search failed first
        if name in failed_searches:
            error = failed_searches[name]
            if "timeout" in str(error).lower():
                print(f"[BATCH] Timed out {completed_count}/{len(names)} ({name}) - search timeout after {elapsed:.1f}s")
            else:
                print(f"[BATCH] Failed {completed_count}/{len(names)} ({name}) - search error")
            error_result = _build_error(str(error))
            error_result["name"] = name
            error_result["institution"] = INSTITUTION
            result = error_result
        elif llm_outcome is None:
            # This shouldn't happen (should_skip_llm or LLM should provide result), but defensive check
            print(f"[BATCH] Failed {completed_count}/{len(names)} ({name}) - unexpected None outcome")
            error_result = _build_error("Unexpected None outcome from LLM phase")
            error_result["name"] = name
            error_result["institution"] = INSTITUTION
            result = error_result
        elif isinstance(llm_outcome, BaseException):
            if isinstance(llm_outcome, asyncio.TimeoutError):
                print(f"[BATCH] Timed out {completed_count}/{len(names)} ({name}) - LLM timeout after {elapsed:.1f}s")
                error_result = _build_error(str(NameProcessingTimeout(name, NAME_TIMEOUT)))
                error_result["name"] = name
                error_result["institution"] = INSTITUTION
                result = error_result
            else:
                print(f"[BATCH] Failed {completed_count}/{len(names)} ({name}) in {elapsed:.1f}s: {llm_outcome}")
                error_result = _build_error(str(llm_outcome))
                error_result["name"] = name
                error_result["institution"] = INSTITUTION
                result = error_result
        else:
            print(f"[BATCH] Completed {completed_count}/{len(names)} ({name}) in {elapsed:.1f}s")
            result = llm_outcome
            print_result_summary(result)
        
        ordered_results.append(result)
    
    batch_elapsed = time.time() - batch_start
    print(f"[BATCH] Batch completed in {batch_elapsed:.1f}s (search: {search_phase_elapsed:.1f}s, LLM: {llm_phase_elapsed:.1f}s)")
    return ordered_results


def write_partial_results(results: List[Dict[str, str]]) -> None:
    if not results:
        return
    _ensure_parent_dir(PARTIAL_RESULTS_PATH)
    pd.DataFrame(results).to_csv(PARTIAL_RESULTS_PATH, index=False)


def save_final_results(results: List[Dict[str, str]]) -> None:
    if not results:
        return
    _ensure_parent_dir(RESULTS_PATH)
    pd.DataFrame(results).to_csv(RESULTS_PATH, index=False)
    print(f"[INFO] Results saved to {RESULTS_PATH}")


async def _process_llm_batch(llm_batch: List[tuple], debug: bool = False, dataset_profile: str = None) -> List[Dict[str, str]]:
    """Process a batch of names through LLM with controlled parallelism.
    
    Args:
        llm_batch: List of (name, search_results) tuples
        debug: Enable debug output
        dataset_profile: Dataset profile for adaptive filtering
        
    Returns:
        List of LLM results in the same order as input
    """
    if not llm_batch:
        return []
    
    print(f"[LLM-BATCH] Processing {len(llm_batch)} names through LLM...")
    llm_batch_start = time.time()
    
    # Create semaphore to limit concurrent LLM calls
    llm_semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM_CALLS)
    
    async def llm_with_semaphore(name: str, results: List[Dict[str, Any]]) -> Dict[str, str]:
        async with llm_semaphore:
            return await asyncio.wait_for(
                process_name_llm(name, results, debug=debug, dataset_profile=dataset_profile),
                timeout=NAME_TIMEOUT
            )
    
    # Launch all LLM tasks in parallel (controlled by semaphore)
    llm_coroutines = [
        llm_with_semaphore(name, search_results)
        for name, search_results in llm_batch
    ]
    llm_outcomes = await asyncio.gather(*llm_coroutines, return_exceptions=True)
    
    # Process outcomes
    llm_results = []
    for (name, search_results), outcome in zip(llm_batch, llm_outcomes):
        if isinstance(outcome, BaseException):
            error_msg = str(outcome)
            if "timeout" in error_msg.lower():
                print(f"[LLM-BATCH] LLM timed out for {name}: {error_msg}")
                result = _build_error(str(NameProcessingTimeout(name, NAME_TIMEOUT)))
            else:
                print(f"[LLM-BATCH] LLM failed for {name}: {error_msg}")
                result = _build_error(error_msg)
            result["name"] = name
            result["institution"] = INSTITUTION
        else:
            result = outcome
            print(f"[LLM-BATCH] LLM succeeded for {name}")
            if debug:
                print_result_summary(result)
        
        llm_results.append(result)
    
    llm_batch_elapsed = time.time() - llm_batch_start
    print(f"[LLM-BATCH] Batch completed in {llm_batch_elapsed:.1f}s")
    
    return llm_results


async def _run_pipeline_dynamic_batching(
    names: List[str],
    search_batch_size: int,
    use_enhanced_search: bool,
    inter_batch_delay: float,
    debug: bool,
    dataset_profile: str
) -> List[Dict[str, str]]:
    """Run pipeline with dynamic batching - accumulate names needing LLM and batch efficiently.
    
    DYNAMIC BATCHING STRATEGY:
    1. Process names in large search batches (fast filtering).
    2. Queue probable connections for LLM.
    3. Flush LLM queue in small batches (5) to ensure steady progress.
    """
    total = len(names)
    LLM_BATCH_TRIGGER = 5  # Flush LLM queue when it hits this size
    
    search_label = "enhanced" if use_enhanced_search else "basic"
    print(f"[PIPELINE] Starting DYNAMIC BATCHING mode")
    print(f"[PIPELINE] Total: {total} names, Search batch: {search_batch_size}, LLM Flush Size: {LLM_BATCH_TRIGGER}")
    print(f"[PIPELINE] Using {search_label} search, Dataset profile: {dataset_profile or 'default'}")
    
    all_results: List[Dict[str, str]] = []
    start_time = time.time()
    
    # Split names into search batches
    search_batches = [names[i:i + search_batch_size] for i in range(0, total, search_batch_size)]
    
    # Accumulator for names needing LLM
    llm_batch_accumulator: List[tuple] = []  # List of (name, search_results) tuples
    llm_batch_results_map: Dict[str, Dict[str, str]] = {}  # name -> LLM result
    
    # Track results in order (for skipped names, we add immediately; for LLM names, we add placeholder)
    result_order: List[str] = []  # Names in original order
    result_map: Dict[str, Dict[str, str]] = {}  # name -> result (for skipped)
    
    # Create semaphore to limit concurrent searches
    search_semaphore = asyncio.Semaphore(MAX_CONCURRENT_SEARCHES)

    async def search_with_semaphore(name: str):
        # Add random jitter to prevent thundering herd
        await asyncio.sleep(random.uniform(0.5, 2.0))
        async with search_semaphore:
             return await process_name_search(name, use_enhanced_search, debug=debug, dataset_profile=dataset_profile)

    # Process search batches
    for search_batch_idx, search_batch in enumerate(tqdm(search_batches, desc="Processing search batches", unit="batch"), 1):
        print(f"\n[PIPELINE] ===== SEARCH BATCH {search_batch_idx}/{len(search_batches)} =====")
        print(f"[PIPELINE] Names: {', '.join(search_batch)}")
        
        search_batch_start = time.time()
        
        # Phase 1: Parallel searches
        print(f"[SEARCH] Running parallel searches for {len(search_batch)} names (max {MAX_CONCURRENT_SEARCHES} concurrent)...")
        search_tasks = []
        for name in search_batch:
            task = search_with_semaphore(name)
            search_tasks.append(task)
        
        search_outcomes = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        # Process search results
        search_results_map: Dict[str, List[Dict[str, Any]]] = {}
        
        for name, outcome in zip(search_batch, search_outcomes):
            result_order.append(name)  # Track order
            
            if isinstance(outcome, BaseException):
                error_msg = str(outcome)
                print(f"[SEARCH] Search failed for {name}: {error_msg}")
                
                # Create error result immediately so it can be retried
                error_result = _build_error(f"Search failed: {error_msg}")
                error_result["name"] = name
                error_result["institution"] = INSTITUTION
                result_map[name] = error_result
            else:
                result_name, results = outcome
                search_results_map[result_name] = results
                print(f"[SEARCH] Search succeeded for {name}: {len(results)} results")
        
        # Phase 2: Skip evaluation and accumulation
        print(f"[SKIP-EVAL] Evaluating skip logic for {len(search_batch)} names (dataset_profile={dataset_profile})...")
        skipped_count = 0
        accumulated_count = 0
        
        for name in search_batch:
            # Skip if already handled (e.g. search failure)
            if name in result_map:
                continue

            # Get results (may be empty if search failed)
            results = search_results_map.get(name, [])
            
            # DEBUG: Show max relevance score for diagnostics
            max_rel_score = 0
            purdue_mentions = 0
            purdue_domain_hits = 0
            if results:
                max_rel_score = max(r.get('signals', {}).get('relevance_score', 0) for r in results)
                for r in results:
                    text = (r.get('title', '') + r.get('snippet', '')).lower()
                    if 'purdue' in text:
                        purdue_mentions += 1
                    url = r.get('url', '').lower()
                    if 'purdue.edu' in url:
                        purdue_domain_hits += 1
            
            # VIP TRACING: Always log VIP names regardless of debug mode
            is_vip = _is_vip_name(name) or _has_vip_signals(results)
            if is_vip:
                print(f"[VIP-TRACE] {name}: {len(results)} results, max_score={max_rel_score:.1f}, purdue_mentions={purdue_mentions}, .edu_hits={purdue_domain_hits}")
            
            should_skip, skip_reason = should_skip_llm(results, dataset_profile=dataset_profile, name=name)
            
            # VIP TRACING: Always log skip decision for VIP names
            if is_vip:
                print(f"[VIP-TRACE] {name}: skip={should_skip}, reason={skip_reason}")
            
            if debug and not should_skip:
                print(f"[SKIP-EVAL] {name}: NOT skipped (max_score={max_rel_score:.1f})")
            
            if should_skip:
                skip_result = _build_immediate_not_connected(name, INSTITUTION, skip_reason or "No results found")
                result_map[name] = skip_result
                skipped_count += 1
                
                # Enhanced logging: Show which tier/pattern triggered the skip
                tier_marker = ""
                if "Tier 1" in skip_reason:
                    tier_marker = "[T1]"
                elif "Tier 2" in skip_reason:
                    tier_marker = "[T2]"
                elif "Tier 3" in skip_reason:
                    tier_marker = "[T3]"
                elif "Tier 4" in skip_reason or "Confident negative" in skip_reason:
                    tier_marker = "[T4]"
                elif "negative evidence dominates" in skip_reason:
                    tier_marker = "[NEG]"
                else:
                    tier_marker = "[SKIP]"
                print(f"{tier_marker} Skipped: {name} - {skip_reason}")
            else:
                # Accumulate for LLM processing
                llm_batch_accumulator.append((name, results))
                accumulated_count += 1
                print(f"[ACCUM] Added {name} to LLM batch (accumulated: {len(llm_batch_accumulator)})")
        
        search_batch_elapsed = time.time() - search_batch_start
        print(f"[SEARCH-BATCH] Completed in {search_batch_elapsed:.1f}s - Skipped: {skipped_count}, Accumulated: {accumulated_count}")
        
        # Phase 3: Check if we should trigger LLM batch processing
        # OPTIMIZED: Trigger when queue hits 5, not waiting for full search batch size
        if len(llm_batch_accumulator) >= LLM_BATCH_TRIGGER:
            print(f"\n[LLM-TRIGGER] Accumulated {len(llm_batch_accumulator)} names, triggering LLM batch...")
            
            # Process full batches of LLM_BATCH_TRIGGER
            while len(llm_batch_accumulator) >= LLM_BATCH_TRIGGER:
                current_llm_batch = llm_batch_accumulator[:LLM_BATCH_TRIGGER]
                llm_batch_accumulator = llm_batch_accumulator[LLM_BATCH_TRIGGER:]
                
                llm_results = await _process_llm_batch(current_llm_batch, debug=debug, dataset_profile=dataset_profile)
                
                # Store results in map
                for (name, _), llm_result in zip(current_llm_batch, llm_results):
                    llm_batch_results_map[name] = llm_result
                
                print(f"[LLM-TRIGGER] Processed LLM batch, {len(llm_batch_accumulator)} names still accumulated")

        
        # Save partial results after each search batch
        partial_results = []
        for name in result_order:
            if name in result_map:
                partial_results.append(result_map[name])
            elif name in llm_batch_results_map:
                partial_results.append(llm_batch_results_map[name])
        
        if partial_results:
            write_partial_results(partial_results)
        
        # RESOURCE CLEANUP: Prevent memory leaks and connection exhaustion
        # 1. Lightweight cleanup every batch (HTTP client)
        await cleanup_batch_resources()
        
        # 2. Heavy cleanup every 20 batches (Browser restart)
        # This prevents Chrome process from bloating over time
        if search_batch_idx % 20 == 0:
            print(f"[RESOURCE] Performing hard browser reset (Batch {search_batch_idx})...")
            await force_browser_recreation()
            # Add extra delay after hard reset to let system settle
            await asyncio.sleep(2.0)
        
        # Inter-batch delay
        if search_batch_idx < len(search_batches):
            print(f"[PIPELINE] Waiting {inter_batch_delay}s before next search batch...")
            await asyncio.sleep(inter_batch_delay)
    
    # Phase 4: Flush remaining LLM batch
    if llm_batch_accumulator:
        print(f"\n[LLM-FLUSH] Flushing {len(llm_batch_accumulator)} remaining names...")
        llm_results = await _process_llm_batch(llm_batch_accumulator, debug=debug, dataset_profile=dataset_profile)
        
        # Store results in map
        for (name, _), llm_result in zip(llm_batch_accumulator, llm_results):
            llm_batch_results_map[name] = llm_result
        
        llm_batch_accumulator = []
    
    # Phase 5: Assemble final results in original order
    print(f"\n[PIPELINE] Assembling final results in original order...")
    for name in result_order:
        if name in result_map:
            all_results.append(result_map[name])
        elif name in llm_batch_results_map:
            all_results.append(llm_batch_results_map[name])
        else:
            # This shouldn't happen, but add error result just in case
            error_result = _build_error("Missing result - processing error")
            error_result["name"] = name
            error_result["institution"] = INSTITUTION
            all_results.append(error_result)
            print(f"[ERROR] Missing result for {name}")
    
    total_elapsed = time.time() - start_time
    avg = total_elapsed / total if total else 0.0
    
    skipped_total = len([r for r in all_results if "SKIP" in r.get("reasoning", "") or r.get("confidence") == "N/A"])
    llm_total = len(llm_batch_results_map)
    
    print(f"\n[PIPELINE] ===== COMPLETED (DYNAMIC BATCHING) =====")
    print(f"[PIPELINE] Total time: {total_elapsed:.1f}s")
    print(f"[PIPELINE] Average per name: {avg:.1f}s")
    print(f"[PIPELINE] Total names: {total}")
    print(f"[PIPELINE] Skipped: {skipped_total} ({100*skipped_total/total:.1f}%)")
    print(f"[PIPELINE] LLM processed: {llm_total} ({100*llm_total/total:.1f}%)")
    print(f"[PIPELINE] Total results: {len(all_results)}")
    
    return all_results


async def run_pipeline(names: List[str], batch_size: int, use_enhanced_search: bool, inter_batch_delay: float = INTER_BATCH_DELAY, debug: bool = False, dataset_profile: str = None, use_dynamic_batching: bool = True) -> List[Dict[str, str]]:
    """Run the full pipeline with optimized accuracy and scalability.
    
    ACCURACY VS SCALABILITY STRATEGY:
    ===================================
    
    Goal: 95%+ accuracy on 100k+ names while minimizing API costs
    
    Key Design Decisions:
    1. **Adaptive Filtering**: Skip policy adapts to dataset_profile
       - high_connection: Conservative - only skip garbage results
       - low_connection: Aggressive - skip weak results to save time
    
    2. **Dynamic Batch Processing** (NEW):
       - Search phase: Process names in batches, run parallel searches
       - Skip evaluation: Determine which names need LLM
       - LLM accumulation: Accumulate names needing LLM across multiple search batches
       - Dynamic LLM batching: Process LLM calls when accumulated >= batch_size
       - Maximizes efficiency on datasets with high skip rate (95%+ for Nobel dataset)
       - Reduces wasted searches and optimizes LLM call parallelization
    
    3. **Smart Search Strategy**:
       - Try basic search first (faster, often sufficient)
       - Escalate to enhanced search only if <5 results
       - Enhanced search uses multiple query strategies for recall
    
    4. **Robust Error Handling**:
       - Per-name timeout (180s) prevents one slow case from blocking batch
       - Automatic retries with exponential backoff
       - Failed records retried at end with smaller batches
    
    5. **Quality Filtering AFTER LLM**:
       - Post-processing catches false positives (events, prizes, etc.)
       - But LLM makes the primary decision (it's good at nuance)
       - Heuristics supplement rather than override LLM
    
    Cost Optimization:
    - Dynamic batching accumulates names needing LLM processing
    - Parallel execution reduces wall-clock time
    - Aggressive skipping on low-connection datasets (skip 95%+)
    - Result caching via partial saves
    
    At 100k names (low_connection dataset):
    - Expected LLM calls: ~2-5k (after skipping 95%+ obvious non-matches)
    - Wall-clock time: ~20-30 minutes (with dynamic batching + aggressive skipping)
    - Expected accuracy: 95-98% (based on multi-tier skip policy + LLM quality)
    """
    total = len(names)
    
    if use_dynamic_batching:
        # OPTIMIZATION: If using enhanced search, ensure batch size is manageable.
        # Previous limit of 4 was too conservative after search optimizations.
        # Now that we disabled DDG fallback for enhanced strategies, we can handle more.
        # Increased to 24 to match browser pool size (24) + buffer
        effective_batch_size = batch_size
        if use_enhanced_search and batch_size > 24:
            print(f"[OPTIMIZATION] Cap search batch size at 24 for enhanced search (requested {batch_size})")
            effective_batch_size = 24
            
        return await _run_pipeline_dynamic_batching(
            names, effective_batch_size, use_enhanced_search, inter_batch_delay, debug, dataset_profile
        )
    
    # Fall back to original fixed batching
    batches = [names[i:i + batch_size] for i in range(0, total, batch_size)]
    search_label = "enhanced" if use_enhanced_search else "basic"
    print(f"[PIPELINE] Starting: {total} name(s) in {len(batches)} batch(es) using {search_label} search")
    print(f"[PIPELINE] Batch size: {batch_size}, Inter-batch delay: {inter_batch_delay}s")
    
    all_results: List[Dict[str, str]] = []
    start_time = time.time()

    for index, batch in enumerate(tqdm(batches, desc="Processing batches", unit="batch"), 1):
        print(f"\n[PIPELINE] ===== BATCH {index}/{len(batches)} =====")
        print(f"[PIPELINE] Names in this batch: {batch}")
        batch_start = time.time()
        
        try:
            batch_results = await process_batch(batch, use_enhanced_search, dataset_profile=dataset_profile, debug=debug)
            all_results.extend(batch_results)
            batch_elapsed = time.time() - batch_start
            pipeline_elapsed = time.time() - start_time
            
            print(f"[PIPELINE] Batch {index} completed in {batch_elapsed:.1f}s")
            print(f"[PIPELINE] Total pipeline time so far: {pipeline_elapsed:.1f}s")
            print(f"[PIPELINE] Total results collected: {len(all_results)}")
            
            write_partial_results(all_results)
            print(f"[PIPELINE] Partial results written to {PARTIAL_RESULTS_PATH}")
            
        except Exception as e:
            batch_elapsed = time.time() - batch_start
            print(f"[ERROR] Batch {index} failed after {batch_elapsed:.1f}s: {e}")
            # Add error results for all names in failed batch
            for name in batch:
                error_result = _build_error(str(e))
                error_result["name"] = name
                error_result["institution"] = INSTITUTION
                all_results.append(error_result)
        
        # Clean up resources between batches to prevent performance degradation
        if index < len(batches):  # Don't cleanup after the last batch
            print(f"[PIPELINE] Starting cleanup and {inter_batch_delay}s delay before batch {index + 1}...")
            cleanup_start = time.time()
            
            try:
                await cleanup_batch_resources()
                await refresh_session()  # Also refresh LLM session
                cleanup_elapsed = time.time() - cleanup_start
                print(f"[PIPELINE] Cleanup completed in {cleanup_elapsed:.1f}s")
                
                print(f"[PIPELINE] Waiting {inter_batch_delay}s before next batch...")
                await asyncio.sleep(inter_batch_delay)
                print(f"[PIPELINE] Ready to start batch {index + 1}")
                
            except Exception as e:
                cleanup_elapsed = time.time() - cleanup_start
                print(f"[WARN] Cleanup failed after {cleanup_elapsed:.1f}s: {e}")

    # Check for failed records and retry them
    failed_results = [r for r in all_results if has_error(r)]
    if failed_results:
        failed_names = [r["name"] for r in failed_results]
        print(f"\n[PIPELINE] ===== RETRY PHASE =====")
        print(f"[PIPELINE] Found {len(failed_names)} failed record(s), retrying...")
        print(f"[PIPELINE] Failed names: {', '.join(failed_names)}")
        
        retry_start = time.time()
        try:
            # Retry with smaller batches for better reliability
            retry_batch_size = max(1, batch_size // 2)
            retry_batches = [failed_names[i:i + retry_batch_size] for i in range(0, len(failed_names), retry_batch_size)]
            
            retry_results: List[Dict[str, str]] = []
            for retry_index, retry_batch in enumerate(tqdm(retry_batches, desc="Retrying failed batches", unit="batch"), 1):
                print(f"[RETRY] Processing batch {retry_index}/{len(retry_batches)}: {retry_batch}")
                retry_batch_results = await process_batch(retry_batch, use_enhanced_search, dataset_profile=dataset_profile, debug=debug)
                retry_results.extend(retry_batch_results)
                
                if retry_index < len(retry_batches):
                    print(f"[RETRY] Waiting {inter_batch_delay}s before next retry batch...")
                    await asyncio.sleep(inter_batch_delay)
            
            # Merge retry results back into all_results
            all_results = merge_results(all_results, retry_results)
            retry_elapsed = time.time() - retry_start
            
            # Check how many are still failing
            still_failed = [r for r in all_results if has_error(r)]
            successful_retries = len(failed_names) - len(still_failed)
            
            print(f"[RETRY] Retry phase completed in {retry_elapsed:.1f}s")
            print(f"[RETRY] Successfully recovered {successful_retries}/{len(failed_names)} record(s)")
            if still_failed:
                print(f"[RETRY] {len(still_failed)} record(s) still have errors")
            
            write_partial_results(all_results)
            
        except Exception as e:
            retry_elapsed = time.time() - retry_start
            print(f"[ERROR] Retry phase failed after {retry_elapsed:.1f}s: {e}")

    total_elapsed = time.time() - start_time
    avg = total_elapsed / total if total else 0.0
    print(f"\n[PIPELINE] ===== COMPLETED =====")
    print(f"[PIPELINE] Total time: {total_elapsed:.1f}s")
    print(f"[PIPELINE] Average per name: {avg:.1f}s")
    print(f"[PIPELINE] Total results: {len(all_results)}")
    
    # Final error summary
    final_errors = [r for r in all_results if has_error(r)]
    if final_errors:
        print(f"[PIPELINE] Warning: {len(final_errors)} record(s) completed with errors")
    else:
        print(f"[PIPELINE] All records processed successfully")
    
    return all_results


async def main_async(debug: bool, batch_size: int, use_enhanced_search: bool, input_path: str, inter_batch_delay: float) -> None:
    try:
        names = load_names(input_path)
    except Exception as error:
        print(f"[ERROR] Failed to load names: {error}")
        return

    if not names:
        fallback = input("Enter a name to evaluate: ").strip()
        if not fallback:
            print("[INFO] No names provided. Exiting.")
            return
        names = [fallback]

    print(f"[INFO] Target institution: {INSTITUTION}")
    print(f"[INFO] Batch size: {batch_size}")
    if len([names[i:i + batch_size] for i in range(0, len(names), batch_size)]) > 1:
        print(f"[INFO] Inter-batch delay: {inter_batch_delay}s")

    results = await run_pipeline(names, batch_size, use_enhanced_search, inter_batch_delay, debug=debug)
    save_final_results(results)


def parse_cli(argv: List[str]):
    debug = "--debug" in argv
    use_enhanced = "--basic" not in argv and "--basic-search" not in argv
    batch_size = DEFAULT_BATCH_SIZE
    input_path = DEFAULT_INPUT_PATH
    inter_batch_delay = INTER_BATCH_DELAY

    for arg in argv:
        if arg.startswith("--batch-size="):
            value = arg.split("=", 1)[1]
            try:
                batch_size = max(1, min(int(value), 50))
            except ValueError:
                print(f"[WARN] Invalid batch size '{value}', using default {DEFAULT_BATCH_SIZE}")
                batch_size = DEFAULT_BATCH_SIZE
        elif arg.startswith("--input="):
            value = arg.split("=", 1)[1]
            if value:
                input_path = value
        elif arg.startswith("--batch-delay="):
            value = arg.split("=", 1)[1]
            try:
                inter_batch_delay = max(0.0, float(value))
            except ValueError:
                print(f"[WARN] Invalid batch delay '{value}', using default {INTER_BATCH_DELAY}")
                inter_batch_delay = INTER_BATCH_DELAY

    return debug, batch_size, use_enhanced, input_path, inter_batch_delay


def main() -> None:
    debug, batch_size, use_enhanced_search, input_path, inter_batch_delay = parse_cli(sys.argv[1:])

    async def runner():
        try:
            await main_async(debug, batch_size, use_enhanced_search, input_path, inter_batch_delay)
        finally:
            await close_search_clients()
            await close_session()

    asyncio.run(runner())


if __name__ == "__main__":
    main()

