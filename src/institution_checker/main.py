import asyncio
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .config import INSTITUTION
from .llm_processor import analyze_connection, close_session, refresh_session, _build_error
from .search import bing_search, close_search_clients, enhanced_search, cleanup_batch_resources, _fix_text_encoding

DEFAULT_BATCH_SIZE = 6
DEFAULT_INPUT_PATH = "data/input_names.csv"
RESULTS_PATH = "data/results.csv"
PARTIAL_RESULTS_PATH = "data/results_partial.csv"
INTER_BATCH_DELAY = 3  # seconds between batches to prevent rate limiting
NAME_TIMEOUT = 180.0  # maximum time allowed per name (gives buffer for LLM retries: 30s * 1.5^2 * 3 = ~150s)
SEARCH_TIMEOUT = 60.0  # maximum time for search phase per name (should complete well before NAME_TIMEOUT)
MAX_CONCURRENT_LLM_CALLS = 2  # limit concurrent LLM API calls (reduced to 2 to reduce SSL timeouts)


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


async def process_name_search(name: str, use_enhanced_search: bool, debug: bool = False) -> tuple[str, List[Dict[str, Any]]]:
    """Phase 1: Perform search for a name and return results.
    
    Applies SEARCH_TIMEOUT to ensure searches complete quickly even if individual
    strategies are slow. This prevents one slow search from blocking the entire batch.
    """
    search_start = time.time()
    
    print(f"[PROGRESS] Starting search for: {name}")
    
    async def _do_search() -> tuple[str, List[Dict[str, Any]]]:
        """Inner function that performs the actual search."""
        try:
            # Cascading search strategy (OPTIMIZATION: try basic first to reduce redundancy)
            if use_enhanced_search:
                # Try basic search first (faster, less redundant)
                query = f'"{name}" "{INSTITUTION}"'
                print(f"[PROGRESS] Trying basic search first for efficiency...")
                results = await bing_search(query, institution=INSTITUTION, person_name=name, num_results=25, debug=debug)
                
                # Evaluate if we need enhanced search
                # Check for high-quality results with strong signals
                high_quality_count = sum(
                    1 for r in results 
                    if r.get('signals', {}).get('relevance_score', 0) >= 10
                )
                
                # STRICT escalation to enhanced search only for genuinely ambiguous cases:
                # - Less than 2 total results (extremely sparse), AND
                # - Zero high-quality results (no strong evidence)
                # This prevents ~600 Purdue names from escalating while still catching edge cases.
                needs_enhanced = len(results) < 2 and high_quality_count == 0
                
                if needs_enhanced:
                    print(f"[PROGRESS] Basic search returned {len(results)} results ({high_quality_count} high-quality), escalating to enhanced search...")
                    results = await enhanced_search(name, INSTITUTION, num_results=30, debug=debug)
                else:
                    print(f"[PROGRESS] Basic search returned {len(results)} results ({high_quality_count} high-quality), sufficient for analysis")
                
                # Final fallback if still no results
                if not results:
                    print(f"[WARN] No results from either search method")
            else:
                query = f'"{name}" {INSTITUTION}'
                results = await bing_search(query, institution=INSTITUTION, person_name=name, num_results=25, debug=debug)
            
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


async def process_name_llm(name: str, results: List[Dict[str, Any]], debug: bool = False) -> Dict[str, str]:
    """Phase 2: Analyze search results with LLM."""
    llm_start = time.time()
    print(f"[PROGRESS] Starting LLM analysis for: {name}")
    
    try:
        decision = await analyze_connection(name, INSTITUTION, results, debug=debug)
        
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


async def process_name(name: str, use_enhanced_search: bool, debug: bool = False) -> Dict[str, str]:
    """Process a name with focused error handling - rely on component-level robustness.
    
    This is kept for backward compatibility and simple single-name processing.
    For batch processing, use the split phase approach in process_batch().
    """
    start = time.time()
    
    try:
        # Use the split-phase approach for consistency
        result_name, results = await process_name_search(name, use_enhanced_search, debug=debug)
        decision = await process_name_llm(result_name, results, debug=debug)
        
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
    
    Used for obvious cases where we can skip LLM to save API quota.
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


def should_skip_llm(results: List[Dict[str, Any]]) -> tuple[bool, Optional[str]]:
    """Check if we can skip LLM call for obvious cases.
    
    Returns: (should_skip: bool, reason: str)
    
    Skip LLM ONLY when:
    - Zero search results found (no evidence to analyze at all)
    
    Important: We do NOT skip based on score thresholds because:
    - Low scores might still contain valid connections (name collisions, mixed results)
    - The LLM is good at finding connections even in weak result sets
    - False negatives are worse than wasted LLM calls for 99.4% non-Purdue cases
    """
    if not results:
        return True, "No search results found"
    
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


async def process_batch(names: List[str], use_enhanced_search: bool, debug: bool = False) -> List[Dict[str, str]]:
    """Process a batch of names with parallel search and LLM phases for optimal performance."""
    print(f"[BATCH] Processing {len(names)} names: {', '.join(names)}")
    batch_start = time.time()

    # ===== PHASE 1: Search (all in parallel) =====
    print(f"[BATCH] Phase 1: Running searches in parallel for all {len(names)} names")
    search_phase_start = time.time()
    
    search_coroutines = [
        asyncio.wait_for(
            process_name_search(name, use_enhanced_search, debug=debug),
            timeout=NAME_TIMEOUT
        )
        for name in names
    ]
    search_outcomes = await asyncio.gather(*search_coroutines, return_exceptions=True)
    
    search_phase_elapsed = time.time() - search_phase_start
    print(f"[BATCH] Phase 1 completed in {search_phase_elapsed:.1f}s")
    
    # Process search results
    search_results: Dict[str, List[Dict[str, Any]]] = {}
    failed_searches: Dict[str, Exception] = {}
    
    for name, outcome in zip(names, search_outcomes):
        if isinstance(outcome, Exception):
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
        should_skip, skip_reason = should_skip_llm(results)
        
        if should_skip:
            skipped_result = _build_immediate_not_connected(name, INSTITUTION, skip_reason or "No results found")
            skipped_results[name] = skipped_result
            skipped_count += 1
            print(f"[BATCH] Skipping LLM for {name}: {skip_reason}")
        else:
            names_needing_llm.append(name)
    
    if skipped_count > 0:
        print(f"[BATCH] Skipped LLM for {skipped_count} name(s) with obvious non-connections, {len(names_needing_llm)} names need LLM")
    
    # Create semaphore to limit concurrent LLM calls
    llm_semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM_CALLS)
    
    async def llm_with_semaphore(name: str, results: List[Dict[str, Any]]) -> Dict[str, str]:
        async with llm_semaphore:
            return await asyncio.wait_for(
                process_name_llm(name, results, debug=debug),
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
        elif isinstance(llm_outcome, Exception):
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


async def run_pipeline(names: List[str], batch_size: int, use_enhanced_search: bool, inter_batch_delay: float = INTER_BATCH_DELAY, debug: bool = False) -> List[Dict[str, str]]:
    total = len(names)
    batches = [names[i:i + batch_size] for i in range(0, total, batch_size)]
    search_label = "enhanced" if use_enhanced_search else "basic"
    print(f"[PIPELINE] Starting: {total} name(s) in {len(batches)} batch(es) using {search_label} search")
    print(f"[PIPELINE] Batch size: {batch_size}, Inter-batch delay: {inter_batch_delay}s")
    
    all_results: List[Dict[str, str]] = []
    start_time = time.time()

    for index, batch in enumerate(batches, 1):
        print(f"\n[PIPELINE] ===== BATCH {index}/{len(batches)} =====")
        print(f"[PIPELINE] Names in this batch: {batch}")
        batch_start = time.time()
        
        try:
            batch_results = await process_batch(batch, use_enhanced_search, debug=debug)
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
                all_results.append(build_error_result(name, e))
        
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
            for retry_index, retry_batch in enumerate(retry_batches, 1):
                print(f"[RETRY] Processing batch {retry_index}/{len(retry_batches)}: {retry_batch}")
                retry_batch_results = await process_batch(retry_batch, use_enhanced_search, debug=debug)
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

