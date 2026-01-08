
import asyncio
import sys
import os
import pandas as pd
import time
from typing import List

# Add src to path to ensure we use the local version
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from institution_checker import run_pipeline, set_api_key
from institution_checker.main import MAX_CONCURRENT_SEARCHES, MAX_CONCURRENT_LLM_CALLS

# Configuration
API_KEY = "sk-3f1dbbf2450e46ab9541dffba4f18ec6"
set_api_key(API_KEY)

VIP_NAMES = [
    "Akira Suzuki", "Ei-ichi Negishi", "Herbert C. Brown", 
    "John B. Fenn", "Vernon L. Smith", "Ben R. Mottelson", 
    "E. M. Purcell", "Julian Schwinger", "Wolfgang Pauli"
]

async def run_test():
    print(f"🚀 Starting Optimization Test")
    print(f"   Concurrent Searches: {MAX_CONCURRENT_SEARCHES}")
    print(f"   Concurrent LLM: {MAX_CONCURRENT_LLM_CALLS}")
    
    # 1. Load Data
    try:
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), '../data/nobel_sample.csv'))
        filler_names = df['name'].tolist()
    except Exception:
        print("⚠️ Could not load nobel_sample.csv, using dummy fillers")
        filler_names = [f"Test Person {i}" for i in range(50)]

    # Combine VIPs and Fillers (ensure unique)
    all_names = list(set(VIP_NAMES + filler_names))
    
    # Ensure we have enough names to trigger batching logic
    while len(all_names) < 40:
        all_names.append(f"Filler Person {len(all_names)}")
        
    print(f"   Total Names: {len(all_names)}")
    print(f"   VIPs present: {all(vip in all_names for vip in VIP_NAMES)}")

    # 2. Run Pipeline
    start_time = time.time()
    results = await run_pipeline(
        all_names,
        batch_size=15,
        use_enhanced_search=True,
        dataset_profile="high_connection",
        use_dynamic_batching=True,
        debug=False # Keep false to reduce noise, VIP tracing is enabled in code
    )
    elapsed = time.time() - start_time
    
    # 3. Analyze Results
    print("\n📊 Test Results")
    print(f"   Total Time: {elapsed:.1f}s")
    print(f"   Avg per name: {elapsed/len(all_names):.2f}s")
    
    vip_results = {r['name']: r for r in results if r['name'] in VIP_NAMES}
    
    correct_count = 0
    print("\n   VIP Verification:")
    for vip in VIP_NAMES:
        if vip not in vip_results:
            print(f"   ❌ {vip}: MISSING from results")
            continue
            
        res = vip_results[vip]
        verdict = res.get('verdict')
        institution = res.get('institution', '')
        is_connected = verdict == 'connected' and 'purdue' in str(institution).lower()
        
        if is_connected:
            print(f"   ✅ {vip}: Connected ({res.get('relationship_type')})")
            correct_count += 1
        else:
            print(f"   ❌ {vip}: {verdict} - {res.get('verification_detail')}")
            
    print(f"\n   Score: {correct_count}/{len(VIP_NAMES)}")
    
    if correct_count == len(VIP_NAMES):
        print("\n✅ SUCCESS: All VIPs detected!")
    else:
        print("\n❌ FAILURE: Missed VIPs")

if __name__ == "__main__":
    asyncio.run(run_test())
