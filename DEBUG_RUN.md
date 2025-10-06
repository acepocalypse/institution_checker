# Debugging LLM Inconsistent Results

## Problem
The LLM is giving inconsistent results for the same names:
- A. Stephen Morse: Sometimes "connected", sometimes "no connection"
- Akasha Gloria Hull: Sometimes "connected", sometimes "no connection"  
- A. Leon Higginbotham: Getting "no connection"

## Observation
When LLM calls take exactly 121.7 seconds (suspiciously same time), they tend to return incorrect results.

## Diagnosis Steps

### 1. Run with Debug Mode
Enable debug mode to see what the LLM is actually receiving and returning:

```python
# In your notebook:
cluster_results = await run_pipeline(
    names_to_check, 
    batch_size=8, 
    use_enhanced_search=True,
    debug=True  # ← Add this
)
```

### 2. Test Individual Names
Test a single problematic name to see the full output:

```python
from src.institution_checker.main import process_name
from src.institution_checker.config import INSTITUTION

# Test one name with debug
result = await process_name("A. Stephen Morse", use_enhanced_search=True, debug=True)
print(result)
```

### 3. Check LLM API Timeout
The LLM API might have a server-side timeout of ~120 seconds. I've increased the client timeout to 180 seconds, but the server might still timeout.

### 4. Reduce Concurrency
Try reducing `MAX_CONCURRENT_LLM_CALLS` to 2 or 3 to see if the API performs better with fewer concurrent requests:

```python
# In src/institution_checker/main.py, change line 19:
MAX_CONCURRENT_LLM_CALLS = 2  # Try reducing from 4 to 2
```

## Root Causes to Investigate

1. **LLM API Server Timeout**: The API might have a 120s server-side timeout that returns incomplete responses
2. **Rate Limiting**: Multiple concurrent requests might trigger rate limiting
3. **Response Truncation**: The LLM might be returning truncated JSON when it runs out of time
4. **Search Results Quality**: The search results for these names might be poor quality

## Quick Fix
For now, the retry logic should catch these failures. The improved `has_error()` function should better detect malformed responses.

## Long-term Fix
Consider:
1. Adding response validation to detect truncated/incomplete LLM responses
2. Implementing exponential backoff for retries
3. Adding checksums or validation markers to LLM responses
4. Using a different LLM model or API provider with better reliability
