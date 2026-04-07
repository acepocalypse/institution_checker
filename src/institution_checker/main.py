import asyncio
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse
from urllib.parse import urlsplit

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
    PRE_LLM_SURVEY_ENABLE_BORDERLINE_RESCUE,
    PRE_LLM_SURVEY_ENABLED,
    PRE_LLM_SURVEY_HARD_NO_THRESHOLD,
    PRE_LLM_SURVEY_HARD_NO_THRESHOLD_LOW_CONNECTION,
    PRE_LLM_SURVEY_LOW_CONNECTION_MIN_RELEVANCE_FOR_ESCALATION,
    PRE_LLM_SURVEY_LOW_CONNECTION_REQUIRE_STRONG_DOMAIN,
    PRE_LLM_SURVEY_PLAUSIBLE_THRESHOLD,
    PRE_LLM_SURVEY_PLAUSIBLE_THRESHOLD_LOW_CONNECTION,
    PRE_LLM_SURVEY_RESCUE_NUM_RESULTS,
    PRE_LLM_SURVEY_RESCUE_QUERY_TEMPLATE,
    PRE_LLM_SURVEY_TRI_STATE,
    PRE_LLM_SURVEY_VIP_BYPASS,
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
    ValidationSearchContext,
    bing_search,
    close_search_clients,
    enhanced_search,
    cleanup_batch_resources,
    enrich_with_page_excerpts,
    force_browser_recreation,
    _fix_text_encoding,
    validation_basic_search,
    validation_enhanced_search,
    validation_search_query,
)

DEFAULT_BATCH_SIZE = 20
DEFAULT_INPUT_PATH = "data/input_names.csv"
RESULTS_PATH = "data/results.csv"
PARTIAL_RESULTS_PATH = "data/results_partial.csv"
INTER_BATCH_DELAY = 0.5  # seconds between batches
NAME_TIMEOUT = 240.0  # maximum time allowed per name (balanced for accuracy)
SEARCH_TIMEOUT = 180.0  # accuracy-friendly search timeout; avoid premature cutoff of difficult names
MAX_CONCURRENT_LLM_CALLS = int(os.getenv("INSTITUTION_CHECKER_MAX_CONCURRENT_LLM", "6"))  # tune down by default for API stability
MAX_CONCURRENT_SEARCHES = 16  # higher search parallelism for low-connection bulk runs
SEARCH_JITTER_MIN = 0.05
SEARCH_JITTER_MAX = 0.25

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

SURVEY_HARD_NO = "hard_no"
SURVEY_BORDERLINE = "borderline"
SURVEY_PLAUSIBLE = "plausible"

SURVEY_STAGE_REJECT_FAST = "reject_fast"
SURVEY_STAGE_REJECT_AFTER_RESCUE = "reject_after_rescue"
SURVEY_STAGE_RESCUE_CANDIDATE = "rescue_candidate"
SURVEY_STAGE_PASS_STRONG = "pass_strong"
SURVEY_STAGE_PASS_AFTER_RESCUE = "pass_after_rescue"
SURVEY_STAGE_PASS_WEAK = "pass_weak"

SURVEY_REASON_SUMMARIES = {
    "survey_disabled": "Pre-LLM survey disabled",
    "vip_bypass": "VIP bypass kept this name eligible for LLM review",
    "no_results": "No search results found",
    "explicit_connection": "Explicit connection language found in search results",
    "authoritative_institution_page": "Authoritative institution page found",
    "institution_domain_hit": "Institution domain hit found",
    "edu_signal": "Educational domain evidence found",
    "multi_result_support": "Multiple supporting search results found",
    "person_institution_match": "Person and institution co-occur in results",
    "strong_relevance": "High relevance score found",
    "joint_campus_only": "Only joint-campus evidence was found",
    "weak_institution_evidence": "Institution evidence is weak",
    "dominant_negative_signals": "Negative context dominates results",
    "non_affiliation_shape": "Results look like celebrity or media coverage rather than institutional affiliation",
    "weak_profile_anchor": "Person-specific Purdue profile evidence exists but wording is sparse",
    "historical_anchor": "Direct historical role, degree, or training evidence was found",
    "legacy_positive_shape": "Historical or indirect evidence still shows a plausible Purdue connection",
    "low_authority_sources": "Only weak sources support the match",
    "rescue_query_promoted": "Rescue query found additional institutional evidence",
    "rescue_query_failed": "Rescue query did not improve evidence quality",
    "low_connection_strict_gate": "Low-connection profile requires stronger authoritative evidence",
}


@dataclass(frozen=True)
class PreLlmSurveyDecision:
    bucket: str
    score: int
    reason_codes: Tuple[str, ...]
    summary: str
    used_rescue_query: bool = False
    metrics: Dict[str, Any] = field(default_factory=dict)


class NameProcessingTimeout(Exception):
    """Raised when processing a single name exceeds the allowed timeout."""

    def __init__(self, name: str, timeout: float = NAME_TIMEOUT):
        super().__init__(f"Timed out processing '{name}' after {timeout:.0f}s")
        self.name = name
        self.timeout = timeout


def _is_low_connection_profile(profile: Optional[str]) -> bool:
    effective = (profile or DATASET_PROFILE or "").lower()
    return "low" in effective


def _profile_survey_thresholds(profile: Optional[str]) -> tuple[int, int]:
    if _is_low_connection_profile(profile):
        return (
            PRE_LLM_SURVEY_HARD_NO_THRESHOLD_LOW_CONNECTION,
            PRE_LLM_SURVEY_PLAUSIBLE_THRESHOLD_LOW_CONNECTION,
        )
    return (PRE_LLM_SURVEY_HARD_NO_THRESHOLD, PRE_LLM_SURVEY_PLAUSIBLE_THRESHOLD)


def _classify_pre_llm_stage(
    bucket: str,
    metrics: Dict[str, Any],
    *,
    used_rescue_query: bool,
) -> str:
    if bucket == SURVEY_HARD_NO:
        return SURVEY_STAGE_REJECT_AFTER_RESCUE if used_rescue_query else SURVEY_STAGE_REJECT_FAST
    if bucket == SURVEY_BORDERLINE:
        return SURVEY_STAGE_RESCUE_CANDIDATE

    if used_rescue_query:
        return SURVEY_STAGE_PASS_AFTER_RESCUE

    strong_evidence = (
        metrics.get("explicit_connection_hits", 0) > 0
        or metrics.get("authoritative_institution_hits", 0) > 0
        or metrics.get("purdue_domain_hits", 0) > 0
    )
    return SURVEY_STAGE_PASS_STRONG if strong_evidence else SURVEY_STAGE_PASS_WEAK


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


def _select_ddg_min_results(dataset_profile: str = None, *, vip_name: bool = False) -> int:
    effective_profile = dataset_profile or DATASET_PROFILE
    if vip_name:
        return DDG_MIN_RESULTS_VIP
    if effective_profile == "low_connection":
        return DDG_MIN_RESULTS_LOW_CONNECTION
    return DDG_MIN_RESULTS_DEFAULT


async def process_name_search(name: str, use_enhanced_search: bool, debug: bool = False, dataset_profile: str = None) -> tuple[str, List[Dict[str, Any]]]:
    """Phase 1: Perform search for a name and return results.
    
    Applies SEARCH_TIMEOUT to ensure searches complete quickly even if individual
    strategies are slow. This prevents one slow search from blocking the entire batch.
    """
    search_start = time.time()
    
    print(f"[PROGRESS] Starting search for: {name}")
    
    async def _do_search() -> tuple[str, List[Dict[str, Any]]]:
        """Inner function that performs the staged production search."""
        try:
            allow_recovery_fallback = _should_allow_slow_recovery_fallback(name)
            allow_enhanced = use_enhanced_search and (_is_vip_name(name) or not VIP_ONLY_ENHANCED)
            results, decision, metadata = await _run_staged_pre_llm_search(
                name,
                dataset_profile=dataset_profile,
                debug=debug,
                allow_enhanced=allow_enhanced,
                allow_bing_recovery_fallback=allow_recovery_fallback,
                allow_slow_ddg_recovery_fallback=allow_recovery_fallback,
                cache_enabled=False,
            )
            
            search_elapsed = time.time() - search_start
            print(
                f"[PROGRESS] Search completed for {name} in {search_elapsed:.1f}s, "
                f"found {len(results)} results "
                f"(mode={metadata.get('search_mode_used')}, queries={metadata.get('network_queries_used')}, "
                f"bucket={decision.bucket})"
            )
            
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


def _validation_positive_anchor_count(
    metrics: Dict[str, Any],
    reason_codes: Tuple[str, ...] | List[str],
) -> int:
    count = 0
    if metrics.get("explicit_connection_hits", 0) > 0:
        count += 1
    if metrics.get("weak_profile_anchor_hits", 0) > 0:
        count += 1
    if metrics.get("authoritative_institution_hits", 0) > 0:
        count += 1
    if metrics.get("historical_role_anchor_hits", 0) > 0:
        count += 1
    if metrics.get("historical_education_anchor_hits", 0) > 0:
        count += 1
    if metrics.get("historical_profile_anchor_hits", 0) > 0:
        count += 1
    if (
        metrics.get("person_institution_hits", 0) >= 3
        and metrics.get("purdue_domain_hits", 0) >= 1
        and (
            metrics.get("weak_profile_anchor_hits", 0) > 0
            or metrics.get("authoritative_institution_hits", 0) > 0
            or metrics.get("explicit_connection_hits", 0) > 0
            or metrics.get("historical_role_anchor_hits", 0) > 0
            or metrics.get("historical_education_anchor_hits", 0) > 0
        )
    ):
        count += 1
    if "legacy_positive_shape" in reason_codes:
        count += 1
    return count


def _has_historical_anchor_metrics(metrics: Dict[str, Any]) -> bool:
    return any(
        metrics.get(key, 0) > 0
        for key in (
            "historical_role_anchor_hits",
            "historical_education_anchor_hits",
            "historical_profile_anchor_hits",
        )
    )


def classify_validation_borderline(
    name: str,
    decision: PreLlmSurveyDecision,
) -> tuple[str, str]:
    metrics = decision.metrics or {}
    reason_codes = tuple(decision.reason_codes)
    if decision.bucket != SURVEY_BORDERLINE:
        if decision.bucket == SURVEY_PLAUSIBLE:
            return "already_plausible", "already_plausible"
        return "hard_no", "hard_no"

    positive_anchor_count = _validation_positive_anchor_count(metrics, reason_codes)
    negative_shape = metrics.get("negative_shape_hits", 0) > 0
    historical_anchor = _has_historical_anchor_metrics(metrics)

    if "legacy_positive_shape" in reason_codes:
        return "borderline_likely_positive", "legacy_positive_shape"
    if historical_anchor:
        return "borderline_likely_positive", "historical_anchor"
    if metrics.get("explicit_connection_hits", 0) > 0:
        return "borderline_likely_positive", "explicit_connection"
    if metrics.get("weak_profile_anchor_hits", 0) > 0:
        return "borderline_likely_positive", "weak_profile_anchor"
    if metrics.get("authoritative_institution_hits", 0) > 0:
        return "borderline_likely_positive", "authoritative_profile"
    if (
        metrics.get("person_institution_hits", 0) >= 3
        and metrics.get("purdue_domain_hits", 0) >= 1
        and positive_anchor_count >= 1
    ):
        return "borderline_likely_positive", "strong_person_purdue_overlap"
    if negative_shape and positive_anchor_count == 0:
        return "borderline_low_signal", "negative_shape_without_anchor"
    return "borderline_low_signal", "low_signal_borderline"


def should_attempt_validation_rescue(
    name: str,
    decision: PreLlmSurveyDecision,
) -> tuple[bool, str, str]:
    subtype, reason = classify_validation_borderline(name, decision)
    if subtype == "borderline_likely_positive":
        return True, reason, subtype
    return False, reason, subtype


def should_attempt_validation_enhanced(
    name: str,
    decision: PreLlmSurveyDecision,
) -> tuple[bool, str, str]:
    metrics = decision.metrics or {}
    subtype, _ = classify_validation_borderline(name, decision)
    if decision.bucket == SURVEY_PLAUSIBLE:
        return False, "already_plausible", subtype
    if decision.bucket != SURVEY_BORDERLINE:
        if metrics.get("explicit_connection_hits", 0) > 0:
            return True, "explicit_connection_without_promotion", "borderline_likely_positive"
        if metrics.get("weak_profile_anchor_hits", 0) > 0:
            return True, "weak_profile_anchor", "borderline_likely_positive"
        if _has_historical_anchor_metrics(metrics):
            return True, "historical_anchor_without_promotion", "borderline_likely_positive"
        return False, "obvious_negative", subtype
    if metrics.get("negative_shape_hits", 0) > 0 and metrics.get("explicit_connection_hits", 0) == 0:
        if not _has_historical_anchor_metrics(metrics):
            return False, "negative_shape_post_rescue", subtype
    if subtype == "borderline_likely_positive":
        if metrics.get("explicit_connection_hits", 0) > 0:
            return True, "explicit_connection", subtype
        if metrics.get("weak_profile_anchor_hits", 0) > 0:
            return True, "weak_profile_anchor", subtype
        if metrics.get("authoritative_institution_hits", 0) > 0:
            return True, "authoritative_profile", subtype
        if _has_historical_anchor_metrics(metrics):
            return True, "historical_anchor", subtype
        if "legacy_positive_shape" in decision.reason_codes:
            return True, "legacy_positive_shape", subtype
    return False, "borderline_low_signal", subtype


def _should_demote_validation_borderline_to_hard_no(
    name: str,
    decision: PreLlmSurveyDecision,
    borderline_subtype: str,
    dataset_profile: str = None,
) -> bool:
    if not _is_low_connection_profile(dataset_profile):
        return False
    metrics = decision.metrics or {}
    if decision.bucket != SURVEY_BORDERLINE:
        return False
    if borderline_subtype != "borderline_low_signal":
        return False
    if metrics.get("explicit_connection_hits", 0) > 0:
        return False
    if metrics.get("weak_profile_anchor_hits", 0) > 0:
        return False
    if metrics.get("authoritative_institution_hits", 0) > 0:
        return False
    if _has_historical_anchor_metrics(metrics):
        return False
    if metrics.get("negative_shape_hits", 0) > 0:
        return True
    if metrics.get("weak_institution_evidence"):
        return True
    return False


def _should_allow_slow_recovery_fallback(name: str) -> bool:
    return False


def _merge_backend_labels(*labels: str) -> str:
    return "|".join(sorted({label for label in labels if label}))


def _finalize_search_metadata(
    metadata: Dict[str, Any],
    search_context: ValidationSearchContext,
) -> Dict[str, Any]:
    metadata["cache_hits_total"] = search_context.cache_hits
    metadata["cache_misses_total"] = search_context.cache_misses
    metadata["backend_hits_total"] = dict(search_context.backend_hits)
    return metadata


def _print_search_metadata_summary(prefix: str, metadata_items: List[Dict[str, Any]]) -> None:
    if not metadata_items:
        return
    total = len(metadata_items)
    rescue_count = sum(1 for item in metadata_items if item.get("rescue_attempted"))
    enhanced_count = sum(1 for item in metadata_items if item.get("enhanced_escalated"))
    slow_fallback_count = sum(
        1
        for item in metadata_items
        if item.get("ddg_manual_retry_used") or item.get("ddg_browser_fallback_used") or item.get("bing_fallback_used")
    )
    total_queries = sum(int(item.get("network_queries_used", 0)) for item in metadata_items)
    total_attempts = sum(int(item.get("network_attempt_count", 0)) for item in metadata_items)
    avg_queries = total_queries / total if total else 0.0
    avg_attempts = total_attempts / total if total else 0.0
    print(
        f"{prefix} Search telemetry: avg_queries={avg_queries:.2f}, avg_attempts={avg_attempts:.2f}, "
        f"rescue={rescue_count}, enhanced={enhanced_count}, slow_fallback={slow_fallback_count}"
    )


async def _run_staged_pre_llm_search(
    name: str,
    *,
    dataset_profile: str = None,
    debug: bool = False,
    allow_enhanced: bool = True,
    allow_bing_recovery_fallback: bool = False,
    allow_slow_ddg_recovery_fallback: bool = False,
    cache_enabled: bool = False,
    context: Optional[ValidationSearchContext] = None,
) -> tuple[List[Dict[str, Any]], PreLlmSurveyDecision, Dict[str, Any]]:
    search_total_start = time.time()
    basic_elapsed = 0.0
    rescue_elapsed = 0.0
    enhanced_elapsed = 0.0

    search_context = context or ValidationSearchContext(
        cache_enabled=cache_enabled,
        allow_bing_fallback=allow_bing_recovery_fallback,
        allow_slow_ddg_fallback=allow_slow_ddg_recovery_fallback,
    )

    basic_start = time.time()
    basic_results, basic_meta = await validation_basic_search(
        name,
        INSTITUTION,
        debug=debug,
        context=search_context,
        allow_bing_fallback=False,
        allow_slow_ddg_fallback=False,
    )
    empty_result_recovery_attempted = False
    empty_result_recovery_succeeded = False
    if not basic_results:
        empty_result_recovery_attempted = True
        recovery_context = ValidationSearchContext(
            cache_enabled=False,
            allow_bing_fallback=True,
            allow_slow_ddg_fallback=True,
        )
        recovered_results, recovery_meta = await validation_basic_search(
            name,
            INSTITUTION,
            debug=debug,
            context=recovery_context,
            allow_bing_fallback=True,
            allow_slow_ddg_fallback=True,
        )
        basic_meta = {
            "backend_used": _merge_backend_labels(
                str(basic_meta.get("backend_used", "")),
                str(recovery_meta.get("backend_used", "")),
            ),
            "cache_hit": bool(basic_meta.get("cache_hit", False)) and bool(recovery_meta.get("cache_hit", False)),
            "network_queries_used": int(basic_meta.get("network_queries_used", 0)) + int(recovery_meta.get("network_queries_used", 0)),
            "ddg_manual_retry_used": bool(basic_meta.get("ddg_manual_retry_used", False)) or bool(recovery_meta.get("ddg_manual_retry_used", False)),
            "ddg_browser_fallback_used": bool(basic_meta.get("ddg_browser_fallback_used", False)) or bool(recovery_meta.get("ddg_browser_fallback_used", False)),
            "bing_fallback_used": bool(basic_meta.get("bing_fallback_used", False)) or bool(recovery_meta.get("bing_fallback_used", False)),
            "network_attempt_count": int(basic_meta.get("network_attempt_count", 0)) + int(recovery_meta.get("network_attempt_count", 0)),
        }
        if recovered_results:
            basic_results = recovered_results
            empty_result_recovery_succeeded = True
    basic_elapsed = time.time() - basic_start

    def _finalize_with_timing(meta: Dict[str, Any]) -> Dict[str, Any]:
        total_elapsed = time.time() - search_total_start
        meta["timing_basic_s"] = basic_elapsed
        meta["timing_rescue_s"] = rescue_elapsed
        meta["timing_enhanced_s"] = enhanced_elapsed
        meta["timing_total_s"] = total_elapsed
        return _finalize_search_metadata(meta, search_context)

    decision = evaluate_pre_llm_survey(
        basic_results,
        dataset_profile=dataset_profile,
        name=name,
    )
    current_results: List[Dict[str, Any]] = list(basic_results)
    rescue_attempted = False
    rescue_reason = "not_needed"
    enhanced_reason = "not_evaluated"
    borderline_subtype = classify_validation_borderline(name, decision)[0]

    metadata: Dict[str, Any] = {
        "search_mode_used": "basic_only",
        "enhanced_escalated": False,
        "rescue_attempted": False,
        "basic_result_count": len(basic_results),
        "final_result_count": len(current_results),
        "escalation_reason": enhanced_reason,
        "rescue_reason": rescue_reason,
        "enhanced_reason": enhanced_reason,
        "borderline_subtype": borderline_subtype,
        "backend_used": str(basic_meta.get("backend_used", "ddg")),
        "cache_hit": bool(basic_meta.get("cache_hit", False)),
        "network_queries_used": int(basic_meta.get("network_queries_used", 0)),
        "ddg_manual_retry_used": bool(basic_meta.get("ddg_manual_retry_used", False)),
        "ddg_browser_fallback_used": bool(basic_meta.get("ddg_browser_fallback_used", False)),
        "bing_fallback_used": bool(basic_meta.get("bing_fallback_used", False)),
        "network_attempt_count": int(basic_meta.get("network_attempt_count", 0)),
        "empty_result_recovery_attempted": empty_result_recovery_attempted,
        "empty_result_recovery_succeeded": empty_result_recovery_succeeded,
    }
    if empty_result_recovery_succeeded:
        metadata["search_mode_used"] = "basic_empty_recovered"

    should_rescue = False
    if PRE_LLM_SURVEY_ENABLE_BORDERLINE_RESCUE:
        should_rescue, rescue_reason, borderline_subtype = should_attempt_validation_rescue(name, decision)
    metadata["rescue_reason"] = rescue_reason
    metadata["borderline_subtype"] = borderline_subtype

    if should_rescue:
        rescue_attempted = True
        rescue_query = PRE_LLM_SURVEY_RESCUE_QUERY_TEMPLATE.format(name=name)
        rescue_start = time.time()
        rescue_results, rescue_meta = await validation_search_query(
            rescue_query,
            institution=INSTITUTION,
            person_name=name,
            limit=PRE_LLM_SURVEY_RESCUE_NUM_RESULTS,
            debug=debug,
            context=search_context,
            prefer_backend="ddg",
            allow_bing_fallback=False,
            allow_slow_ddg_fallback=False,
            ensure_tokens=False,
        )
        rescue_elapsed = time.time() - rescue_start
        current_results = _dedupe_by_url(list(current_results) + list(rescue_results))
        decision = evaluate_pre_llm_survey(
            current_results,
            dataset_profile=dataset_profile,
            name=name,
            used_rescue_query=True,
        )
        metadata.update(
            {
                "search_mode_used": "basic_plus_rescue",
                "rescue_attempted": True,
                "final_result_count": len(current_results),
                "backend_used": _merge_backend_labels(
                    str(metadata.get("backend_used", "")),
                    str(rescue_meta.get("backend_used", "")),
                ),
                "cache_hit": bool(metadata.get("cache_hit", False)) and bool(rescue_meta.get("cache_hit", False)),
                "network_queries_used": int(metadata.get("network_queries_used", 0)) + int(rescue_meta.get("network_queries_used", 0)),
                "ddg_manual_retry_used": bool(metadata.get("ddg_manual_retry_used", False)) or bool(rescue_meta.get("ddg_manual_retry_used", False)),
                "ddg_browser_fallback_used": bool(metadata.get("ddg_browser_fallback_used", False)) or bool(rescue_meta.get("ddg_browser_fallback_used", False)),
                "bing_fallback_used": bool(metadata.get("bing_fallback_used", False)) or bool(rescue_meta.get("bing_fallback_used", False)),
                "network_attempt_count": int(metadata.get("network_attempt_count", 0)) + int(rescue_meta.get("network_attempt_count", 0)),
            }
        )
        borderline_subtype = classify_validation_borderline(name, decision)[0]
        metadata["borderline_subtype"] = borderline_subtype
    elif decision.bucket == SURVEY_BORDERLINE and PRE_LLM_SURVEY_ENABLE_BORDERLINE_RESCUE:
        metadata["rescue_reason"] = rescue_reason or "borderline_without_positive_anchor"

    if _should_demote_validation_borderline_to_hard_no(name, decision, borderline_subtype, dataset_profile):
        demoted_metrics = dict(decision.metrics)
        demoted_metrics["stage"] = SURVEY_STAGE_REJECT_AFTER_RESCUE if decision.used_rescue_query else SURVEY_STAGE_REJECT_FAST
        decision = PreLlmSurveyDecision(
            bucket=SURVEY_HARD_NO,
            score=decision.score,
            reason_codes=decision.reason_codes,
            summary=decision.summary,
            used_rescue_query=decision.used_rescue_query,
            metrics=demoted_metrics,
        )
        metadata["search_mode_used"] = f"{metadata['search_mode_used']}_demoted" if metadata["search_mode_used"] != "basic_only" else "basic_only_demoted"
        metadata["enhanced_reason"] = "demoted_low_signal_negative"
        metadata["escalation_reason"] = "demoted_low_signal_negative"
        metadata["final_result_count"] = len(current_results)
        return current_results, decision, _finalize_with_timing(metadata)

    should_escalate, enhanced_reason, borderline_subtype = should_attempt_validation_enhanced(name, decision)
    metadata["enhanced_reason"] = enhanced_reason
    metadata["escalation_reason"] = enhanced_reason
    metadata["borderline_subtype"] = borderline_subtype

    if not allow_enhanced or not should_escalate:
        metadata["final_result_count"] = len(current_results)
        return current_results, decision, _finalize_with_timing(metadata)

    enhanced_start = time.time()
    enhanced_results, enhanced_meta = await validation_enhanced_search(
        name,
        INSTITUTION,
        debug=debug,
        context=search_context,
        allow_bing_fallback=allow_bing_recovery_fallback,
        allow_slow_ddg_fallback=allow_slow_ddg_recovery_fallback,
    )
    enhanced_elapsed = time.time() - enhanced_start
    merged_results = _dedupe_by_url(list(current_results) + list(enhanced_results))
    final_decision = evaluate_pre_llm_survey(
        merged_results,
        dataset_profile=dataset_profile,
        name=name,
        used_rescue_query=decision.used_rescue_query,
    )
    metadata.update(
        {
            "search_mode_used": "basic_plus_rescue_plus_enhanced" if rescue_attempted else "basic_plus_enhanced",
            "enhanced_escalated": True,
            "final_result_count": len(merged_results),
            "enhanced_reason": enhanced_reason,
            "borderline_subtype": classify_validation_borderline(name, final_decision)[0],
            "backend_used": _merge_backend_labels(
                str(metadata.get("backend_used", "")),
                str(enhanced_meta.get("backend_used", "")),
            ),
            "cache_hit": bool(metadata.get("cache_hit", False)) and bool(enhanced_meta.get("cache_hit", False)),
            "network_queries_used": int(metadata.get("network_queries_used", 0)) + int(enhanced_meta.get("network_queries_used", 0)),
            "ddg_manual_retry_used": bool(metadata.get("ddg_manual_retry_used", False)) or bool(enhanced_meta.get("ddg_manual_retry_used", False)),
            "ddg_browser_fallback_used": bool(metadata.get("ddg_browser_fallback_used", False)) or bool(enhanced_meta.get("ddg_browser_fallback_used", False)),
            "bing_fallback_used": bool(metadata.get("bing_fallback_used", False)) or bool(enhanced_meta.get("bing_fallback_used", False)),
            "network_attempt_count": int(metadata.get("network_attempt_count", 0)) + int(enhanced_meta.get("network_attempt_count", 0)),
        }
    )
    if _should_demote_validation_borderline_to_hard_no(
        name,
        final_decision,
        metadata.get("borderline_subtype", ""),
        dataset_profile,
    ):
        demoted_metrics = dict(final_decision.metrics)
        demoted_metrics["stage"] = SURVEY_STAGE_REJECT_AFTER_RESCUE if final_decision.used_rescue_query else SURVEY_STAGE_REJECT_FAST
        final_decision = PreLlmSurveyDecision(
            bucket=SURVEY_HARD_NO,
            score=final_decision.score,
            reason_codes=final_decision.reason_codes,
            summary=final_decision.summary,
            used_rescue_query=final_decision.used_rescue_query,
            metrics=demoted_metrics,
        )
        metadata["search_mode_used"] = f"{metadata['search_mode_used']}_demoted"
        metadata["enhanced_reason"] = "demoted_low_signal_negative"
        metadata["escalation_reason"] = "demoted_low_signal_negative"
    return merged_results, final_decision, _finalize_with_timing(metadata)


async def run_pre_llm_validation_search(
    name: str,
    *,
    dataset_profile: str = None,
    debug: bool = False,
    allow_enhanced: bool = True,
    allow_bing_fallback: bool = False,
    allow_slow_ddg_fallback: bool = False,
    cache_enabled: bool = True,
    context: Optional[ValidationSearchContext] = None,
) -> tuple[List[Dict[str, Any]], PreLlmSurveyDecision, Dict[str, Any]]:
    return await _run_staged_pre_llm_search(
        name,
        dataset_profile=dataset_profile,
        debug=debug,
        allow_enhanced=allow_enhanced,
        allow_bing_recovery_fallback=allow_bing_fallback,
        allow_slow_ddg_recovery_fallback=allow_slow_ddg_fallback,
        cache_enabled=cache_enabled,
        context=context,
    )


async def process_name_llm(
    name: str,
    results: List[Dict[str, Any]],
    debug: bool = False,
    dataset_profile: str = None,
    pre_llm_decision: Optional[PreLlmSurveyDecision] = None,
) -> Dict[str, str]:
    """Phase 2: Analyze search results with LLM."""
    llm_start = time.time()
    print(f"[PROGRESS] Starting LLM analysis for: {name}")
    
    try:
        vip_mode = _is_vip_name(name) or _has_vip_signals(results)

        excerpt_limit = VIP_EXCERPT_LIMIT if vip_mode else LLM_EXCERPT_LIMIT
        if results and excerpt_limit > 0:
            await enrich_with_page_excerpts(results, name, limit=min(excerpt_limit, len(results)))
        
        decision = await analyze_connection(name, INSTITUTION, results, debug=debug, vip_mode=vip_mode)

        if vip_mode and VIP_RESCUE_ENABLED and decision.get("verdict") != "connected":
            ddg_min_results = DDG_MIN_RESULTS_VIP if vip_mode else DDG_MIN_RESULTS_DEFAULT
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
        return _apply_pre_llm_audit(payload, pre_llm_decision)
        
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
        allow_recovery_fallback = _should_allow_slow_recovery_fallback(name)
        surveyed_results, survey_decision, _ = await _run_staged_pre_llm_search(
            name,
            dataset_profile=dataset_profile,
            debug=debug,
            allow_enhanced=use_enhanced_search and (_is_vip_name(name) or not VIP_ONLY_ENHANCED),
            allow_bing_recovery_fallback=allow_recovery_fallback,
            allow_slow_ddg_recovery_fallback=allow_recovery_fallback,
            cache_enabled=False,
        )
        if survey_decision.bucket == SURVEY_HARD_NO:
            return _build_immediate_not_connected(
                name,
                INSTITUTION,
                survey_decision.summary,
                pre_llm_decision=survey_decision,
            )

        decision = await process_name_llm(
            name,
            surveyed_results,
            debug=debug,
            dataset_profile=dataset_profile,
            pre_llm_decision=survey_decision,
        )
        
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


def _build_immediate_not_connected(
    name: str,
    institution: str,
    reason: str,
    pre_llm_decision: Optional[PreLlmSurveyDecision] = None,
) -> Dict[str, str]:
    """Build an immediate 'not_connected' result without LLM call.
    
    Used ONLY for obvious cases where we can skip LLM to save API quota.
    This should be called very rarely - only when results are completely off-topic.
    
    ACCURACY NOTE: Being too aggressive with skips causes false negatives.
    The conservative skip policy ensures we maintain 95%+ accuracy while
    still achieving scalability through batching and parallelization.
    """
    payload = {
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
    return _apply_pre_llm_audit(payload, pre_llm_decision)


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


def _result_has_degree_signal(text: str) -> bool:
    degree_terms = (
        "degree",
        "graduated",
        "graduate",
        "phd",
        "ph.d",
        "doctorate",
        "bachelor",
        "master",
        "student",
        "attended",
        "studied",
        "coursework",
    )
    return any(term in text for term in degree_terms)


_AUTHORITATIVE_PROFILE_PATH_HINTS = (
    "/directory/",
    "/people/",
    "/faculty/",
    "/staff/",
    "/profiles/",
    "/profile/",
    "/alumni/",
)

_NON_PROFILE_PATH_HINTS = (
    "/news",
    "/newsroom",
    "/releases",
    "/events",
    "/event",
    "/calendar",
    "/stories",
    "/story",
    "/article",
    "/articles",
    "/speaker",
    "/speakers",
    "/lecture",
    "/lectures",
    "/forum",
    "/library",
    "/discovery",
    "/catalog",
    "/search",
    "/find",
)

_NON_AFFILIATION_URL_HINTS = (
    "/news",
    "/newsroom",
    "/releases",
    "/events",
    "/event",
    "/calendar",
    "/stories",
    "/story",
    "/article",
    "/articles",
    "/speaker",
    "/speakers",
    "/lecture",
    "/lectures",
    "/forum",
    "/sinai-forum",
    "/presidential-lecture",
    "/brand-studio",
    "/marketing",
    "/campaign",
    "/feature",
    "/features",
    "/library",
    "/discovery",
    "/catalog",
    "/search",
    "/find",
)

_NON_AFFILIATION_TEXT_HINTS = (
    "keynote",
    "speaker",
    "past speakers",
    "lecture series",
    "presidential lecture",
    "forum",
    "sinai forum",
    "concert",
    "festival",
    "watch party",
    "feature story",
    "featured in",
    "highlighted in",
    "highlighted by",
    "article about",
    "story about",
    "news story",
    "campus article",
    "campus feature",
    "opinion column",
    "pop culture column",
    "tribute to",
    "inspired by",
    "themed",
    "campaign",
    "marketing",
    "brand studio",
    "case study",
    "mentioned in",
    "referenced in",
    "book",
    "catalog",
    "discovery",
    "library",
    "biography of",
)

_PERSON_SPECIFIC_PATH_HINTS = (
    "/directory/",
    "/people/",
    "/faculty/",
    "/staff/",
    "/profiles/",
    "/profile/",
    "/alumni/",
    "/person/",
)

_DIRECT_CONNECTION_TERMS = (
    "faculty",
    "professor",
    "assistant professor",
    "associate professor",
    "lecturer",
    "instructor",
    "staff",
    "researcher",
    "research scientist",
    "research associate",
    "director",
    "dean",
    "chair",
    "president",
    "provost",
    "chancellor",
    "executive",
    "postdoc",
    "postdoctoral",
    "visiting professor",
    "visiting scholar",
    "visiting fellow",
    "alumni",
    "alumnus",
    "alumna",
    "alum",
    "student at",
    "studied at",
    "attended",
    "graduated from",
    "degree from",
    "earned degree",
    "earned a",
    "earned an",
    "phd from",
    "ph.d. from",
)

_AUTHORITATIVE_AFFILIATION_TERMS = (
    "faculty",
    "professor",
    "staff",
    "researcher",
    "directory",
    "profile",
    "alumni",
    "alumnus",
    "alumna",
    "student",
    "postdoc",
    "postdoctoral",
    "visiting professor",
    "visiting scholar",
    "emeritus",
    "dean",
    "director",
    "chair",
)


def _is_non_profile_page(url: str) -> bool:
    return any(hint in url for hint in _NON_PROFILE_PATH_HINTS)


def _looks_like_generic_profile_query_page(url: str) -> bool:
    try:
        parsed = urlsplit(url)
    except ValueError:
        return False
    path = (parsed.path or "").lower().rstrip("/")
    query = (parsed.query or "").lower()
    generic_paths = {
        "",
        "/",
        "/directory",
        "/people",
        "/faculty",
        "/staff",
        "/profiles",
        "/profile",
    }
    if path in generic_paths:
        return True
    if any(token in query for token in ("search=", "query=", "searchstring=", "q=")):
        return True
    return False


def _is_non_main_purdue_context(url: str) -> bool:
    lowered = _safe_text(url).lower()
    return any(token in lowered for token in ("pnw.edu", "purdueglobal.edu", "global.purdue.edu"))


def _looks_like_non_affiliation_result(result: Dict[str, Any]) -> bool:
    url = _safe_text(result.get("url", "")).lower()
    text = _aggregate_result_text(result).lower()

    if _is_non_main_purdue_context(url):
        return True
    if any(hint in url for hint in _NON_AFFILIATION_URL_HINTS):
        return True
    return any(term in text for term in _NON_AFFILIATION_TEXT_HINTS)


def _has_person_specific_profile_anchor(result: Dict[str, Any]) -> bool:
    url = _safe_text(result.get("url", "")).lower()
    if "purdue.edu" not in url or _is_non_main_purdue_context(url):
        return False
    if _is_non_profile_page(url) or _looks_like_non_affiliation_result(result):
        return False
    if _looks_like_generic_profile_query_page(url):
        return False

    signals = result.get("signals") or {}
    confidence = signals.get("person_match_confidence")
    if confidence == "url":
        return True

    return any(hint in url for hint in _PERSON_SPECIFIC_PATH_HINTS)


def _is_profile_like_result(result: Dict[str, Any]) -> bool:
    url = _safe_text(result.get("url", "")).lower()
    if "purdue.edu" not in url or _is_non_main_purdue_context(url):
        return False
    if _is_non_profile_page(url) or _looks_like_non_affiliation_result(result):
        return False

    title = _safe_text(result.get("title", "")).lower()
    snippet = _safe_text(result.get("snippet", "")).lower()
    text = f"{title} {snippet}"

    if _has_person_specific_profile_anchor(result):
        return True

    profile_terms = (
        "directory",
        "profile",
        "faculty",
        "staff",
        "alumni",
        "people",
        "biography",
        "curriculum vitae",
        "cv",
    )
    return any(term in text for term in profile_terms) and (
        (result.get("signals") or {}).get("person_match_confidence") == "url"
    )


def _result_has_direct_connection_family(result: Dict[str, Any]) -> bool:
    text = _aggregate_result_text(result)
    url = _safe_text(result.get("url", "")).lower()
    content_text = " ".join(
        _safe_text(result.get(field, ""))
        for field in ("title", "snippet", "page_excerpt", "description")
    ).lower()
    lowered = content_text
    signals = result.get("signals") or {}
    has_historical_role_anchor = bool(signals.get("has_historical_role_anchor"))
    has_historical_education_anchor = bool(signals.get("has_historical_education_anchor"))

    if signals.get("has_event_prize_pattern"):
        return False
    if (
        _looks_like_non_affiliation_result(result)
        and not _has_person_specific_profile_anchor(result)
        and not has_historical_role_anchor
        and not has_historical_education_anchor
    ):
        return False
    if (
        _is_non_profile_page(url)
        and not _is_profile_like_result(result)
        and not has_historical_role_anchor
        and not has_historical_education_anchor
    ):
        return False
    if not signals.get("has_person_name") or not signals.get("has_institution"):
        return False

    if has_historical_role_anchor or has_historical_education_anchor:
        return True

    has_direct_terms = any(term in lowered for term in _DIRECT_CONNECTION_TERMS)
    has_degree_signal = _result_has_degree_signal(lowered)
    strong_page_context = _has_person_specific_profile_anchor(result)

    if has_direct_terms and (
        strong_page_context
        or (signals.get("has_explicit_connection") and (signals.get("has_academic_role") or has_degree_signal))
    ):
        return True

    if signals.get("has_explicit_connection") and (
        signals.get("has_academic_role") or has_degree_signal
    ):
        return True

    return False


def _result_supports_corroboration(result: Dict[str, Any]) -> bool:
    signals = result.get("signals") or {}
    if not signals.get("has_person_name") or not signals.get("has_institution"):
        return False
    if signals.get("has_event_prize_pattern"):
        return False
    if _looks_like_non_affiliation_result(result) and not _has_person_specific_profile_anchor(result):
        if not signals.get("has_historical_role_anchor") and not signals.get("has_historical_education_anchor"):
            return False
    if _result_has_direct_connection_family(result):
        return True
    if _is_profile_like_result(result):
        return True
    if signals.get("has_historical_profile_anchor"):
        return True

    relevance = int(signals.get("relevance_score", 0) or 0)
    url = _safe_text(result.get("url", "")).lower()
    if _is_non_profile_page(url):
        return False
    if "purdue.edu" not in url or _is_non_main_purdue_context(url):
        return False
    if not _has_person_specific_profile_anchor(result):
        return False
    return relevance >= 10 and signals.get("person_match_confidence") in {"url", "institution_domain"}


def _is_authoritative_institution_result(result: Dict[str, Any]) -> bool:
    signals = result.get("signals") or {}
    if not _is_profile_like_result(result):
        if not signals.get("has_historical_profile_anchor"):
            return False
    if _looks_like_non_affiliation_result(result) and not _has_person_specific_profile_anchor(result):
        if not signals.get("has_historical_profile_anchor"):
            return False
    if not _has_person_specific_profile_anchor(result) and not signals.get("has_historical_profile_anchor"):
        return False
    text = _aggregate_result_text(result).lower()
    return (
        _has_person_specific_profile_anchor(result) or signals.get("has_historical_profile_anchor")
    ) and (
        _result_has_direct_connection_family(result) or any(
        term in text for term in _AUTHORITATIVE_AFFILIATION_TERMS
        )
    )


def _compute_pre_llm_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "result_count": len(results),
        "max_relevance_score": 0,
        "purdue_mentions": 0,
        "purdue_domain_hits": 0,
        "edu_hits": 0,
        "institution_signal_hits": 0,
        "explicit_connection_hits": 0,
        "person_institution_hits": 0,
        "academic_role_hits": 0,
        "degree_hits": 0,
        "authoritative_institution_hits": 0,
        "joint_campus_hits": 0,
        "positive_keyword_hits": 0,
        "negative_keyword_hits": 0,
        "negative_result_hits": 0,
        "total_signal_score": 0,
        "total_authority_score": 0,
        "authority_markers": [],
        "corroborating_result_hits": 0,
        "negative_shape_hits": 0,
        "weak_profile_anchor_hits": 0,
        "historical_role_anchor_hits": 0,
        "historical_education_anchor_hits": 0,
        "historical_profile_anchor_hits": 0,
        "has_direct_connection_family": False,
        "has_authoritative_profile_family": False,
        "has_corroborating_support_family": False,
    }

    authority_markers: List[str] = []

    for result in results:
        text = _aggregate_result_text(result)
        lowered, spaced, compact = _normalize_signal_text(text)
        url = _safe_text(result.get("url", "")).lower()
        signals = result.get("signals") or {}

        if "purdue" in lowered or "purdue" in spaced:
            metrics["purdue_mentions"] += 1
        if "purdue.edu" in url:
            metrics["purdue_domain_hits"] += 1
        if urlparse(url).netloc.lower().endswith(".edu"):
            metrics["edu_hits"] += 1
        if signals.get("has_institution"):
            metrics["institution_signal_hits"] += 1
        direct_connection_family = _result_has_direct_connection_family(result)
        authoritative_profile_family = _is_authoritative_institution_result(result)
        supports_corroboration = _result_supports_corroboration(result)
        historical_role_anchor = bool(signals.get("has_historical_role_anchor"))
        historical_education_anchor = bool(signals.get("has_historical_education_anchor"))
        historical_profile_anchor = bool(signals.get("has_historical_profile_anchor"))

        if direct_connection_family:
            metrics["explicit_connection_hits"] += 1
        if historical_role_anchor:
            metrics["historical_role_anchor_hits"] += 1
        if historical_education_anchor:
            metrics["historical_education_anchor_hits"] += 1
        if historical_profile_anchor:
            metrics["historical_profile_anchor_hits"] += 1
        if signals.get("has_person_name") and signals.get("has_institution"):
            metrics["person_institution_hits"] += 1
        if direct_connection_family and signals.get("has_academic_role"):
            metrics["academic_role_hits"] += 1
        if direct_connection_family and _result_has_degree_signal(lowered):
            metrics["degree_hits"] += 1
        if signals.get("has_event_prize_pattern"):
            metrics["negative_result_hits"] += 1
        if authoritative_profile_family:
            metrics["authoritative_institution_hits"] += 1
        if supports_corroboration:
            metrics["corroborating_result_hits"] += 1
        if _looks_like_non_affiliation_result(result):
            metrics["negative_shape_hits"] += 1
        if _has_person_specific_profile_anchor(result) and not direct_connection_family:
            metrics["weak_profile_anchor_hits"] += 1
        if any(pattern in lowered or pattern in url for pattern in JOINT_CAMPUS_PATTERNS):
            metrics["joint_campus_hits"] += 1

        metrics["max_relevance_score"] = max(
            metrics["max_relevance_score"],
            int(signals.get("relevance_score", 0) or 0),
        )

        for keyword in STRONG_POSITIVE_SIGNAL_KEYWORDS:
            if keyword and _keyword_matches_text(keyword, lowered, spaced, compact):
                metrics["positive_keyword_hits"] += 1
                metrics["total_signal_score"] += 10

        for keyword in KNOWN_FALSE_POSITIVE_SIGNAL_KEYWORDS:
            if keyword and _keyword_matches_text(keyword, lowered, spaced, compact):
                metrics["negative_keyword_hits"] += 1
                metrics["total_signal_score"] -= 10

        authority_score, factors = _score_source_authority(url)
        metrics["total_authority_score"] += authority_score
        if factors:
            authority_markers.extend(factors)

    metrics["authority_markers"] = sorted(set(authority_markers))
    metrics["multi_result_support"] = (
        metrics["corroborating_result_hits"] >= 2
        or (
            metrics["corroborating_result_hits"] >= 1
            and metrics["authoritative_institution_hits"] >= 1
            and metrics["explicit_connection_hits"] >= 1
        )
    )
    metrics["has_direct_connection_family"] = metrics["explicit_connection_hits"] > 0
    metrics["has_authoritative_profile_family"] = (
        metrics["authoritative_institution_hits"] > 0
        or metrics["historical_profile_anchor_hits"] > 0
    )
    metrics["has_corroborating_support_family"] = metrics["multi_result_support"]
    metrics["weak_institution_evidence"] = (
        metrics["explicit_connection_hits"] == 0
        and metrics["authoritative_institution_hits"] == 0
        and metrics["purdue_domain_hits"] == 0
        and metrics["edu_hits"] == 0
        and metrics["institution_signal_hits"] == 0
    )
    metrics["dominant_negative_signals"] = (
        metrics["negative_keyword_hits"] >= max(1, metrics["positive_keyword_hits"] + 1)
        or (
            metrics["negative_result_hits"] >= 2
            and metrics["explicit_connection_hits"] == 0
            and metrics["authoritative_institution_hits"] == 0
        )
    )
    return metrics


def _score_pre_llm_metrics(metrics: Dict[str, Any]) -> int:
    score = 0
    max_relevance = metrics["max_relevance_score"]
    score += min(max_relevance, 12)
    score += min(metrics["purdue_mentions"], 4)
    score += min(metrics["institution_signal_hits"] * 2, 6)
    score += min(metrics["person_institution_hits"] * 3, 6)
    score += min(metrics["authoritative_institution_hits"] * 7, 14)
    score += min(metrics["explicit_connection_hits"] * 8, 16)
    score += min(metrics["purdue_domain_hits"] * 4, 8)
    score += min(metrics["edu_hits"] * 2, 4)
    score += min(metrics["degree_hits"], 3)
    score += min(metrics["academic_role_hits"], 3)
    score += min(metrics["positive_keyword_hits"] // 2, 4)

    if metrics["total_authority_score"] >= 10:
        score += 3
    elif metrics["total_authority_score"] <= -5:
        score -= 4

    if metrics["total_signal_score"] >= 20:
        score += 3
    elif metrics["total_signal_score"] <= -20:
        score -= 4

    if metrics["dominant_negative_signals"]:
        score -= 8
    if metrics["joint_campus_hits"] > 0 and metrics["purdue_domain_hits"] == 0:
        score -= 8
    if metrics["weak_institution_evidence"]:
        score -= 6
    if metrics["negative_shape_hits"] > 0 and metrics["explicit_connection_hits"] == 0:
        score -= min(metrics["negative_shape_hits"] * 4, 12)
    if metrics["result_count"] <= 1 and metrics["explicit_connection_hits"] == 0 and metrics["authoritative_institution_hits"] == 0:
        score -= 3
    if metrics["result_count"] == 0:
        score -= 10

    return score


def _build_pre_llm_summary(bucket: str, reason_codes: List[str]) -> str:
    if not reason_codes:
        if bucket == SURVEY_PLAUSIBLE:
            return "Search results show at least some plausible institutional evidence."
        if bucket == SURVEY_BORDERLINE:
            return "Search results are inconclusive and need a lightweight rescue check."
        return "Search results do not show a plausible institutional connection."

    pieces = [SURVEY_REASON_SUMMARIES.get(code, code.replace("_", " ")) for code in reason_codes[:3]]
    if bucket == SURVEY_PLAUSIBLE:
        return "; ".join(pieces)
    if bucket == SURVEY_BORDERLINE:
        return f"Borderline evidence: {'; '.join(pieces)}"
    return f"No plausible connection: {'; '.join(pieces)}"


def evaluate_pre_llm_survey(
    results: List[Dict[str, Any]],
    dataset_profile: str = None,
    name: Optional[str] = None,
    *,
    used_rescue_query: bool = False,
) -> PreLlmSurveyDecision:
    hard_no_threshold, plausible_threshold = _profile_survey_thresholds(dataset_profile)
    low_connection_profile = _is_low_connection_profile(dataset_profile)

    if not PRE_LLM_SURVEY_ENABLED:
        metrics = {
            "result_count": len(results),
            "stage": SURVEY_STAGE_PASS_WEAK,
        }
        return PreLlmSurveyDecision(
            bucket=SURVEY_PLAUSIBLE,
            score=plausible_threshold,
            reason_codes=("survey_disabled",),
            summary=_build_pre_llm_summary(SURVEY_PLAUSIBLE, ["survey_disabled"]),
            used_rescue_query=used_rescue_query,
            metrics=metrics,
        )

    vip_mode = PRE_LLM_SURVEY_VIP_BYPASS and (_is_vip_name(name) or _has_vip_signals(results))
    metrics = _compute_pre_llm_metrics(results)
    score = _score_pre_llm_metrics(metrics)
    reason_codes: List[str] = []

    if vip_mode:
        reason_codes.append("vip_bypass")
        summary = _build_pre_llm_summary(SURVEY_PLAUSIBLE, reason_codes)
        metrics = dict(metrics)
        metrics["stage"] = SURVEY_STAGE_PASS_STRONG
        return PreLlmSurveyDecision(
            bucket=SURVEY_PLAUSIBLE,
            score=max(score, plausible_threshold),
            reason_codes=tuple(reason_codes),
            summary=summary,
            used_rescue_query=used_rescue_query,
            metrics=metrics,
        )

    if metrics["result_count"] == 0:
        reason_codes.append("no_results")
    if metrics["explicit_connection_hits"] > 0:
        reason_codes.append("explicit_connection")
    if metrics["authoritative_institution_hits"] > 0:
        reason_codes.append("authoritative_institution_page")
    if metrics["purdue_domain_hits"] > 0:
        reason_codes.append("institution_domain_hit")
    if metrics["edu_hits"] > 0:
        reason_codes.append("edu_signal")
    if metrics["multi_result_support"]:
        reason_codes.append("multi_result_support")
    if metrics["person_institution_hits"] > 0:
        reason_codes.append("person_institution_match")
    if metrics["max_relevance_score"] >= 12:
        reason_codes.append("strong_relevance")
    if metrics["weak_institution_evidence"]:
        reason_codes.append("weak_institution_evidence")
    if metrics["joint_campus_hits"] > 0 and metrics["purdue_domain_hits"] == 0:
        reason_codes.append("joint_campus_only")
    if metrics["dominant_negative_signals"]:
        reason_codes.append("dominant_negative_signals")
    if metrics["negative_shape_hits"] > 0:
        reason_codes.append("non_affiliation_shape")
    if metrics["weak_profile_anchor_hits"] > 0:
        reason_codes.append("weak_profile_anchor")
    if _has_historical_anchor_metrics(metrics):
        reason_codes.append("historical_anchor")
    if metrics["total_authority_score"] <= 0 and metrics["authoritative_institution_hits"] == 0:
        reason_codes.append("low_authority_sources")

    has_direct_connection_family = bool(metrics["has_direct_connection_family"])
    has_authoritative_profile_family = bool(metrics["has_authoritative_profile_family"])
    has_corroborating_support_family = bool(metrics["has_corroborating_support_family"])
    signal_family_count = sum(
        1
        for flag in (
            has_direct_connection_family,
            has_authoritative_profile_family,
            has_corroborating_support_family,
        )
        if flag
    )

    strong_domain_evidence = (
        has_authoritative_profile_family
        or has_direct_connection_family
        or metrics["purdue_domain_hits"] > 0
    )

    obvious_direct_profile_case = (
        has_direct_connection_family
        and has_authoritative_profile_family
        and metrics["purdue_domain_hits"] > 0
        and metrics["person_institution_hits"] > 0
    )

    weak_true_profile_case = (
        metrics["weak_profile_anchor_hits"] > 0
        and metrics["purdue_domain_hits"] > 0
        and metrics["person_institution_hits"] > 0
        and metrics["negative_shape_hits"] == 0
    )
    anchored_historical_case = (
        _has_historical_anchor_metrics(metrics)
        and metrics["person_institution_hits"] > 0
    )

    legacy_positive_shape = (
        used_rescue_query
        and metrics["explicit_connection_hits"] > 0
        and metrics["authoritative_institution_hits"] == 0
        and metrics["person_institution_hits"] >= 5
        and metrics["purdue_domain_hits"] >= 2
        and metrics["edu_hits"] >= 4
        and metrics["max_relevance_score"] >= 25
        and metrics["total_authority_score"] >= 20
        and metrics["negative_shape_hits"] <= max(2, metrics["explicit_connection_hits"] + 1)
    )
    if legacy_positive_shape:
        reason_codes.append("legacy_positive_shape")

    score_promotable = (
        score >= plausible_threshold
        and (
            signal_family_count >= 2
            or weak_true_profile_case
            or anchored_historical_case
            or legacy_positive_shape
        )
    )

    direct_plausible = (
        obvious_direct_profile_case
        or signal_family_count >= 2
        or weak_true_profile_case
        or anchored_historical_case
        or legacy_positive_shape
    )
    allow_score_plausible = True
    direct_hard_no = (
        metrics["result_count"] == 0
        or (
            metrics["explicit_connection_hits"] == 0
            and metrics["authoritative_institution_hits"] == 0
            and metrics["purdue_domain_hits"] == 0
            and metrics["edu_hits"] == 0
            and metrics["max_relevance_score"] < 6
            and metrics["purdue_mentions"] <= 1
            and metrics["person_institution_hits"] == 0
            and metrics["total_authority_score"] <= 0
            and not _has_historical_anchor_metrics(metrics)
        )
        or (
            metrics["dominant_negative_signals"]
            and metrics["authoritative_institution_hits"] == 0
            and metrics["explicit_connection_hits"] == 0
            and metrics["total_authority_score"] <= 0
            and not _has_historical_anchor_metrics(metrics)
        )
        or (
            metrics["joint_campus_hits"] > 0
            and metrics["purdue_domain_hits"] == 0
            and metrics["authoritative_institution_hits"] == 0
            and metrics["explicit_connection_hits"] == 0
            and not _has_historical_anchor_metrics(metrics)
        )
        or (
            used_rescue_query
            and metrics["explicit_connection_hits"] == 0
            and metrics["authoritative_institution_hits"] == 0
            and metrics["purdue_domain_hits"] > 0
            and metrics["person_institution_hits"] > 0
            and not _has_historical_anchor_metrics(metrics)
        )
        or (
            used_rescue_query
            and metrics["negative_shape_hits"] > 0
            and metrics["explicit_connection_hits"] == 0
            and metrics["authoritative_institution_hits"] == 0
            and metrics["weak_profile_anchor_hits"] == 0
            and not _has_historical_anchor_metrics(metrics)
        )
        or (
            used_rescue_query
            and metrics["negative_shape_hits"] > 0
            and metrics["weak_profile_anchor_hits"] == 0
            and metrics["purdue_domain_hits"] > 0
            and metrics["person_institution_hits"] > 0
            and signal_family_count < 2
            and not _has_historical_anchor_metrics(metrics)
        )
    )

    if low_connection_profile:
        very_weak_low_connection = (
            metrics["explicit_connection_hits"] == 0
            and metrics["authoritative_institution_hits"] == 0
            and metrics["person_institution_hits"] == 0
            and metrics["purdue_domain_hits"] == 0
            and metrics["max_relevance_score"] < PRE_LLM_SURVEY_LOW_CONNECTION_MIN_RELEVANCE_FOR_ESCALATION
        )
        if very_weak_low_connection:
            direct_hard_no = True

        if PRE_LLM_SURVEY_LOW_CONNECTION_REQUIRE_STRONG_DOMAIN and not strong_domain_evidence:
            if score >= plausible_threshold:
                reason_codes.append("low_connection_strict_gate")
            direct_plausible = False
            allow_score_plausible = False

    force_negative_hard_no = (
        direct_hard_no
        and used_rescue_query
        and metrics["negative_shape_hits"] > 0
        and metrics["explicit_connection_hits"] == 0
        and metrics["authoritative_institution_hits"] == 0
        and metrics["weak_profile_anchor_hits"] == 0
    )

    if direct_plausible or (allow_score_plausible and score_promotable):
        bucket = SURVEY_PLAUSIBLE
    elif force_negative_hard_no or (direct_hard_no and score <= hard_no_threshold):
        bucket = SURVEY_HARD_NO
    elif PRE_LLM_SURVEY_TRI_STATE:
        bucket = SURVEY_BORDERLINE
    else:
        bucket = SURVEY_PLAUSIBLE

    metrics = dict(metrics)
    metrics["stage"] = _classify_pre_llm_stage(
        bucket,
        metrics,
        used_rescue_query=used_rescue_query,
    )
    summary = _build_pre_llm_summary(bucket, reason_codes)
    return PreLlmSurveyDecision(
        bucket=bucket,
        score=score,
        reason_codes=tuple(reason_codes),
        summary=summary,
        used_rescue_query=used_rescue_query,
        metrics=metrics,
    )


def _serialize_pre_llm_reason_codes(reason_codes: Tuple[str, ...]) -> str:
    return "|".join(reason_codes)


def _apply_pre_llm_audit(
    result: Dict[str, str],
    decision: Optional[PreLlmSurveyDecision],
) -> Dict[str, str]:
    payload = dict(result)
    payload["pre_llm_bucket"] = decision.bucket if decision else ""
    payload["pre_llm_stage"] = str(decision.metrics.get("stage", "")) if decision else ""
    payload["pre_llm_score"] = str(decision.score) if decision else ""
    payload["pre_llm_summary"] = decision.summary if decision else ""
    payload["pre_llm_reason_codes"] = _serialize_pre_llm_reason_codes(decision.reason_codes) if decision else ""
    payload["pre_llm_used_rescue_query"] = "Y" if decision and decision.used_rescue_query else "N"
    return payload


async def run_pre_llm_survey(
    name: str,
    results: List[Dict[str, Any]],
    dataset_profile: str = None,
    debug: bool = False,
) -> tuple[List[Dict[str, Any]], PreLlmSurveyDecision]:
    decision = evaluate_pre_llm_survey(results, dataset_profile=dataset_profile, name=name)
    borderline_subtype = classify_validation_borderline(name, decision)[0]
    if _should_demote_validation_borderline_to_hard_no(name, decision, borderline_subtype, dataset_profile):
        demoted_metrics = dict(decision.metrics)
        demoted_metrics["stage"] = SURVEY_STAGE_REJECT_FAST
        demoted_decision = PreLlmSurveyDecision(
            bucket=SURVEY_HARD_NO,
            score=decision.score,
            reason_codes=decision.reason_codes,
            summary=decision.summary,
            used_rescue_query=decision.used_rescue_query,
            metrics=demoted_metrics,
        )
        return results, demoted_decision

    if decision.bucket != SURVEY_BORDERLINE or not PRE_LLM_SURVEY_ENABLE_BORDERLINE_RESCUE:
        return results, decision

    should_rescue, _, borderline_subtype = should_attempt_validation_rescue(name, decision)
    if not should_rescue:
        return results, decision

    rescue_query = PRE_LLM_SURVEY_RESCUE_QUERY_TEMPLATE.format(name=name)
    if debug:
        print(f"[SURVEY] Borderline result for {name}; running rescue query: {rescue_query}")

    rescue_results, _ = await validation_search_query(
        rescue_query,
        institution=INSTITUTION,
        person_name=name,
        limit=PRE_LLM_SURVEY_RESCUE_NUM_RESULTS,
        debug=debug,
        prefer_backend="ddg",
        allow_bing_fallback=False,
        allow_slow_ddg_fallback=False,
        ensure_tokens=False,
    )
    merged_results = _dedupe_by_url(list(results) + list(rescue_results))
    rescued_decision = evaluate_pre_llm_survey(
        merged_results,
        dataset_profile=dataset_profile,
        name=name,
        used_rescue_query=True,
    )

    reason_codes = list(rescued_decision.reason_codes)
    if rescued_decision.bucket == SURVEY_PLAUSIBLE:
        if "rescue_query_promoted" not in reason_codes:
            reason_codes.append("rescue_query_promoted")
    else:
        if _should_demote_validation_borderline_to_hard_no(
            name,
            rescued_decision,
            classify_validation_borderline(name, rescued_decision)[0],
            dataset_profile,
        ):
            if "rescue_query_failed" not in reason_codes:
                reason_codes.append("rescue_query_failed")
            rescued_metrics = dict(rescued_decision.metrics)
            rescued_metrics["stage"] = _classify_pre_llm_stage(
                SURVEY_HARD_NO,
                rescued_metrics,
                used_rescue_query=True,
            )
            rescued_decision = PreLlmSurveyDecision(
                bucket=SURVEY_HARD_NO,
                score=rescued_decision.score,
                reason_codes=tuple(reason_codes),
                summary=_build_pre_llm_summary(SURVEY_HARD_NO, reason_codes),
                used_rescue_query=True,
                metrics=rescued_metrics,
            )
            return merged_results, rescued_decision
        if "rescue_query_failed" not in reason_codes:
            reason_codes.append("rescue_query_failed")

    rescued_metrics = dict(rescued_decision.metrics)
    rescued_metrics["stage"] = _classify_pre_llm_stage(
        rescued_decision.bucket,
        rescued_metrics,
        used_rescue_query=True,
    )
    rescued_decision = PreLlmSurveyDecision(
        bucket=rescued_decision.bucket,
        score=rescued_decision.score,
        reason_codes=tuple(reason_codes),
        summary=_build_pre_llm_summary(rescued_decision.bucket, reason_codes),
        used_rescue_query=True,
        metrics=rescued_metrics,
    )
    return merged_results, rescued_decision


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
    decision = evaluate_pre_llm_survey(results, dataset_profile=dataset_profile, name=name)
    if decision.bucket == SURVEY_HARD_NO:
        return True, decision.summary
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
    print(f"[BATCH] Phase 2: Streaming LLM as each search completes (max {MAX_CONCURRENT_LLM_CALLS} concurrent)")
    search_phase_start = time.time()
    
    # Create semaphore to limit concurrent searches
    search_semaphore = asyncio.Semaphore(MAX_CONCURRENT_SEARCHES)

    async def search_with_semaphore(name: str):
        # Add random jitter to prevent thundering herd
        # Reduced jitter (0.5-2s) as DDG is prioritized
        await asyncio.sleep(random.uniform(SEARCH_JITTER_MIN, SEARCH_JITTER_MAX))
        async with search_semaphore:
            allow_recovery_fallback = _should_allow_slow_recovery_fallback(name)
            allow_enhanced = use_enhanced_search and (_is_vip_name(name) or not VIP_ONLY_ENHANCED)
            return await asyncio.wait_for(
                _run_staged_pre_llm_search(
                    name,
                    dataset_profile=dataset_profile,
                    debug=debug,
                    allow_enhanced=allow_enhanced,
                    allow_bing_recovery_fallback=allow_recovery_fallback,
                    allow_slow_ddg_recovery_fallback=allow_recovery_fallback,
                    cache_enabled=False,
                ),
                timeout=SEARCH_TIMEOUT,
            )

    async def search_with_name(name: str):
        try:
            outcome = await search_with_semaphore(name)
            return name, outcome, None
        except Exception as exc:
            return name, None, exc

    search_tasks = [asyncio.create_task(search_with_name(name)) for name in names]

    # Process search/LLM as results arrive to avoid one slow search blocking the whole batch.
    search_metadata_map: Dict[str, Dict[str, Any]] = {}
    failed_searches: Dict[str, Exception] = {}
    skipped_results: Dict[str, Dict[str, str]] = {}
    llm_task_map: Dict[str, asyncio.Task] = {}
    llm_decision_map: Dict[str, PreLlmSurveyDecision] = {}
    llm_outcome_map: Dict[str, Any] = {}

    survey_counts = {
        SURVEY_HARD_NO: 0,
        SURVEY_BORDERLINE: 0,
        SURVEY_PLAUSIBLE: 0,
        "rescue_used": 0,
        "rescue_promoted": 0,
    }

    # Create semaphore to limit concurrent LLM calls
    llm_semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM_CALLS)

    async def llm_with_semaphore(
        name: str,
        results: List[Dict[str, Any]],
        survey_decision: PreLlmSurveyDecision,
    ) -> Dict[str, str]:
        async with llm_semaphore:
            return await asyncio.wait_for(
                process_name_llm(
                    name,
                    results,
                    debug=debug,
                    dataset_profile=dataset_profile,
                    pre_llm_decision=survey_decision,
                ),
                timeout=NAME_TIMEOUT
            )

    for completed in asyncio.as_completed(search_tasks):
        name, outcome, error = await completed
        if error is None:
            results, survey_decision, search_metadata = outcome
            search_metadata_map[name] = search_metadata
            timing_basic = float(search_metadata.get("timing_basic_s", 0.0) or 0.0)
            timing_rescue = float(search_metadata.get("timing_rescue_s", 0.0) or 0.0)
            timing_enhanced = float(search_metadata.get("timing_enhanced_s", 0.0) or 0.0)
            timing_total = float(search_metadata.get("timing_total_s", 0.0) or 0.0)
            print(
                f"[BATCH] Search succeeded for {name}: {len(results)} results "
                f"(mode={search_metadata.get('search_mode_used')}, queries={search_metadata.get('network_queries_used')}, "
                f"bucket={survey_decision.bucket}, t_basic={timing_basic:.1f}s, "
                f"t_rescue={timing_rescue:.1f}s, t_enh={timing_enhanced:.1f}s, t_total={timing_total:.1f}s)"
            )

            survey_counts[survey_decision.bucket] += 1
            if survey_decision.used_rescue_query:
                survey_counts["rescue_used"] += 1
                if "rescue_query_promoted" in survey_decision.reason_codes:
                    survey_counts["rescue_promoted"] += 1

            if survey_decision.bucket == SURVEY_HARD_NO:
                skipped_results[name] = _build_immediate_not_connected(
                    name,
                    INSTITUTION,
                    survey_decision.summary,
                    pre_llm_decision=survey_decision,
                )
                print(f"[BATCH] [HARD-NO] Skipping LLM for {name}: {survey_decision.summary}")
            else:
                llm_decision_map[name] = survey_decision
                llm_task_map[name] = asyncio.create_task(llm_with_semaphore(name, results, survey_decision))
        else:
            error_msg = str(error)
            if "timeout" in error_msg.lower():
                print(f"[BATCH] Search timed out for {name}: {error_msg}")
            else:
                print(f"[BATCH] Search failed for {name}: {error_msg}")
            failed_searches[name] = error

    search_phase_elapsed = time.time() - search_phase_start
    print(f"[BATCH] Phase 1 completed in {search_phase_elapsed:.1f}s")

    _print_search_metadata_summary("[BATCH]", list(search_metadata_map.values()))
    
    # ===== PHASE 2: LLM Analysis (overlapped with search) =====
    llm_phase_start = time.time()
    skipped_count = len(skipped_results)
    names_needing_llm = list(llm_task_map.keys())

    print(
        f"[BATCH] Survey counts: hard_no={survey_counts[SURVEY_HARD_NO]}, "
        f"borderline={survey_counts[SURVEY_BORDERLINE]}, plausible={survey_counts[SURVEY_PLAUSIBLE]}, "
        f"rescue_used={survey_counts['rescue_used']}, rescue_promoted={survey_counts['rescue_promoted']}"
    )
    if skipped_count > 0:
        print(f"[BATCH] Skipped LLM for {skipped_count} name(s) with obvious non-connections, {len(names_needing_llm)} names need LLM")

    if llm_task_map:
        llm_names = list(llm_task_map.keys())
        llm_tasks = [llm_task_map[name] for name in llm_names]
        llm_outcomes = await asyncio.gather(*llm_tasks, return_exceptions=True)
        llm_outcome_map = {name: outcome for name, outcome in zip(llm_names, llm_outcomes)}

    llm_phase_elapsed = time.time() - llm_phase_start
    print(f"[BATCH] Phase 2 completed in {llm_phase_elapsed:.1f}s ({len(names_needing_llm)} LLM calls, {skipped_count} skipped)")
    
    # ===== Combine results =====
    ordered_results: List[Dict[str, str]] = []
    completed_count = 0

    for name in names:
        completed_count += 1
        elapsed = time.time() - batch_start

        if name in skipped_results:
            ordered_results.append(skipped_results[name])
            print(f"[BATCH] Completed {completed_count}/{len(names)} ({name}) in {elapsed:.1f}s")
            continue

        llm_outcome = llm_outcome_map.get(name)
        llm_decision = llm_decision_map.get(name)
        
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
            result = _apply_pre_llm_audit(error_result, None)
        elif llm_outcome is None:
            # This shouldn't happen (should_skip_llm or LLM should provide result), but defensive check
            print(f"[BATCH] Failed {completed_count}/{len(names)} ({name}) - unexpected None outcome")
            error_result = _build_error("Unexpected None outcome from LLM phase")
            error_result["name"] = name
            error_result["institution"] = INSTITUTION
            result = _apply_pre_llm_audit(error_result, llm_decision)
        elif isinstance(llm_outcome, BaseException):
            if isinstance(llm_outcome, asyncio.TimeoutError):
                print(f"[BATCH] Timed out {completed_count}/{len(names)} ({name}) - LLM timeout after {elapsed:.1f}s")
                error_result = _build_error(str(NameProcessingTimeout(name, NAME_TIMEOUT)))
                error_result["name"] = name
                error_result["institution"] = INSTITUTION
                result = _apply_pre_llm_audit(error_result, llm_decision)
            else:
                print(f"[BATCH] Failed {completed_count}/{len(names)} ({name}) in {elapsed:.1f}s: {llm_outcome}")
                error_result = _build_error(str(llm_outcome))
                error_result["name"] = name
                error_result["institution"] = INSTITUTION
                result = _apply_pre_llm_audit(error_result, llm_decision)
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


async def _process_llm_batch(
    llm_batch: List[tuple[str, List[Dict[str, Any]], PreLlmSurveyDecision]],
    debug: bool = False,
    dataset_profile: str = None,
) -> List[Dict[str, str]]:
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
    
    async def llm_with_semaphore(
        name: str,
        results: List[Dict[str, Any]],
        survey_decision: PreLlmSurveyDecision,
    ) -> Dict[str, str]:
        async with llm_semaphore:
            return await asyncio.wait_for(
                process_name_llm(
                    name,
                    results,
                    debug=debug,
                    dataset_profile=dataset_profile,
                    pre_llm_decision=survey_decision,
                ),
                timeout=NAME_TIMEOUT
            )
    
    # Launch all LLM tasks in parallel (controlled by semaphore)
    llm_coroutines = [
        llm_with_semaphore(name, search_results, survey_decision)
        for name, search_results, survey_decision in llm_batch
    ]
    llm_outcomes = await asyncio.gather(*llm_coroutines, return_exceptions=True)
    
    # Process outcomes
    llm_results = []
    for (name, search_results, survey_decision), outcome in zip(llm_batch, llm_outcomes):
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
            result = _apply_pre_llm_audit(result, survey_decision)
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
    llm_batch_accumulator: List[tuple[str, List[Dict[str, Any]], PreLlmSurveyDecision]] = []
    llm_batch_results_map: Dict[str, Dict[str, str]] = {}  # name -> LLM result
    
    # Track results in order (for skipped names, we add immediately; for LLM names, we add placeholder)
    result_order: List[str] = []  # Names in original order
    result_map: Dict[str, Dict[str, str]] = {}  # name -> result (for skipped)
    survey_totals = {
        SURVEY_HARD_NO: 0,
        SURVEY_BORDERLINE: 0,
        SURVEY_PLAUSIBLE: 0,
        "rescue_used": 0,
        "rescue_promoted": 0,
    }
    
    # Create semaphore to limit concurrent searches
    search_semaphore = asyncio.Semaphore(MAX_CONCURRENT_SEARCHES)

    async def search_with_semaphore(name: str):
        # Add random jitter to prevent thundering herd
        await asyncio.sleep(random.uniform(SEARCH_JITTER_MIN, SEARCH_JITTER_MAX))
        async with search_semaphore:
             allow_recovery_fallback = _should_allow_slow_recovery_fallback(name)
             allow_enhanced = use_enhanced_search and (_is_vip_name(name) or not VIP_ONLY_ENHANCED)
             return await asyncio.wait_for(
                 _run_staged_pre_llm_search(
                     name,
                     dataset_profile=dataset_profile,
                     debug=debug,
                     allow_enhanced=allow_enhanced,
                     allow_bing_recovery_fallback=allow_recovery_fallback,
                     allow_slow_ddg_recovery_fallback=allow_recovery_fallback,
                     cache_enabled=False,
                 ),
                 timeout=SEARCH_TIMEOUT,
             )

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
        search_decision_map: Dict[str, PreLlmSurveyDecision] = {}
        search_metadata_map: Dict[str, Dict[str, Any]] = {}
        
        for name, outcome in zip(search_batch, search_outcomes):
            result_order.append(name)  # Track order
            
            if isinstance(outcome, BaseException):
                error_msg = str(outcome)
                print(f"[SEARCH] Search failed for {name}: {error_msg}")
                
                # Create error result immediately so it can be retried
                error_result = _build_error(f"Search failed: {error_msg}")
                error_result["name"] = name
                error_result["institution"] = INSTITUTION
                result_map[name] = _apply_pre_llm_audit(error_result, None)
            else:
                results, survey_decision, search_metadata = outcome
                search_results_map[name] = results
                search_decision_map[name] = survey_decision
                search_metadata_map[name] = search_metadata
                timing_basic = float(search_metadata.get("timing_basic_s", 0.0) or 0.0)
                timing_rescue = float(search_metadata.get("timing_rescue_s", 0.0) or 0.0)
                timing_enhanced = float(search_metadata.get("timing_enhanced_s", 0.0) or 0.0)
                timing_total = float(search_metadata.get("timing_total_s", 0.0) or 0.0)
                print(
                    f"[SEARCH] Search succeeded for {name}: {len(results)} results "
                    f"(mode={search_metadata.get('search_mode_used')}, queries={search_metadata.get('network_queries_used')}, "
                    f"bucket={survey_decision.bucket}, t_basic={timing_basic:.1f}s, "
                    f"t_rescue={timing_rescue:.1f}s, t_enh={timing_enhanced:.1f}s, t_total={timing_total:.1f}s)"
                )

        _print_search_metadata_summary("[SEARCH]", list(search_metadata_map.values()))
        
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
            
            surveyed_results = results
            survey_decision = search_decision_map.get(name)
            if survey_decision is None:
                surveyed_results, survey_decision = await run_pre_llm_survey(
                    name,
                    results,
                    dataset_profile=dataset_profile,
                    debug=debug,
                )
            survey_totals[survey_decision.bucket] += 1
            if survey_decision.used_rescue_query:
                survey_totals["rescue_used"] += 1
                if "rescue_query_promoted" in survey_decision.reason_codes:
                    survey_totals["rescue_promoted"] += 1

            # VIP TRACING: Always log skip decision for VIP names
            if is_vip:
                print(
                    f"[VIP-TRACE] {name}: bucket={survey_decision.bucket}, "
                    f"score={survey_decision.score}, reason={survey_decision.summary}"
                )
            
            if debug and survey_decision.bucket != SURVEY_HARD_NO:
                print(f"[SKIP-EVAL] {name}: survey={survey_decision.bucket} (score={survey_decision.score})")
            
            if survey_decision.bucket == SURVEY_HARD_NO:
                skip_result = _build_immediate_not_connected(
                    name,
                    INSTITUTION,
                    survey_decision.summary,
                    pre_llm_decision=survey_decision,
                )
                result_map[name] = skip_result
                skipped_count += 1
                print(f"[HARD-NO] Skipped: {name} - {survey_decision.summary}")
            else:
                # Accumulate for LLM processing
                llm_batch_accumulator.append((name, surveyed_results, survey_decision))
                accumulated_count += 1
                print(
                    f"[ACCUM] Added {name} to LLM batch as {survey_decision.bucket} "
                    f"(accumulated: {len(llm_batch_accumulator)})"
                )
        
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
                for (name, _, _), llm_result in zip(current_llm_batch, llm_results):
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
            await asyncio.sleep(min(inter_batch_delay, 0.1))
    
    # Phase 4: Flush remaining LLM batch
    if llm_batch_accumulator:
        print(f"\n[LLM-FLUSH] Flushing {len(llm_batch_accumulator)} remaining names...")
        llm_results = await _process_llm_batch(llm_batch_accumulator, debug=debug, dataset_profile=dataset_profile)
        
        # Store results in map
        for (name, _, _), llm_result in zip(llm_batch_accumulator, llm_results):
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
            all_results.append(_apply_pre_llm_audit(error_result, None))
            print(f"[ERROR] Missing result for {name}")
    
    total_elapsed = time.time() - start_time
    avg = total_elapsed / total if total else 0.0
    
    skipped_total = len([r for r in all_results if "SKIP" in r.get("reasoning", "") or r.get("confidence") == "N/A"])
    llm_total = len(llm_batch_results_map)
    
    print(f"\n[PIPELINE] ===== COMPLETED (DYNAMIC BATCHING) =====")
    print(f"[PIPELINE] Total time: {total_elapsed:.1f}s")
    print(f"[PIPELINE] Average per name: {avg:.1f}s")
    print(f"[PIPELINE] Total names: {total}")
    skipped_total = survey_totals[SURVEY_HARD_NO]
    print(f"[PIPELINE] Skipped: {skipped_total} ({100*skipped_total/total:.1f}%)")
    print(
        f"[PIPELINE] Survey totals: hard_no={survey_totals[SURVEY_HARD_NO]}, "
        f"borderline={survey_totals[SURVEY_BORDERLINE]}, plausible={survey_totals[SURVEY_PLAUSIBLE]}"
    )
    print(
        f"[PIPELINE] Rescue usage: used={survey_totals['rescue_used']}, "
        f"promoted={survey_totals['rescue_promoted']}"
    )
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
                all_results.append(_apply_pre_llm_audit(error_result, None))
        
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

