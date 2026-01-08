import asyncio
import time
import random
from institution_checker.search import bing_search

async def benchmark_search():
    print("🚀 Benchmarking Search Performance...")
    print("Target: ~3s per name (assuming 3 queries per name)")
    
    # Simulate 10 names (approx 30 queries)
    names = [f"Test Name {i}" for i in range(10)]
    
    start_time = time.time()
    
    async def mock_pipeline_step(name):
        # Simulate 3 queries per name (enhanced search)
        # We use a real search but for a dummy term to test the rate limiter
        # Using "Purdue" + random number to avoid caching if any
        q1 = f"Purdue University {name} {random.randint(1, 1000)}"
        q2 = f"Purdue University {name} faculty {random.randint(1, 1000)}"
        q3 = f"Purdue University {name} alumni {random.randint(1, 1000)}"
        
        # Run 3 searches concurrently or sequentially as the pipeline would
        # The pipeline runs strategies sequentially usually, but we want to test the global limiter
        await bing_search(q1, num_results=1)
        await bing_search(q2, num_results=1)
        await bing_search(q3, num_results=1)
        return name

    tasks = [mock_pipeline_step(name) for name in names]
    
    # Run with concurrency limit similar to main pipeline
    # We can't easily use the semaphore from main.py here without importing it, 
    # but asyncio.gather will launch them all and they will hit the search.py semaphore/rate limiter
    
    await asyncio.gather(*tasks)
    
    end_time = time.time()
    duration = end_time - start_time
    avg_per_name = duration / len(names)
    
    print(f"\n📊 Results:")
    print(f"   Total time: {duration:.2f}s")
    print(f"   Names processed: {len(names)}")
    print(f"   Average time per name: {avg_per_name:.2f}s")
    
    if 2.5 <= avg_per_name <= 3.5:
        print("✅ SUCCESS: Performance is within target range (2.5s - 3.5s)")
    elif avg_per_name < 2.5:
        print("⚠️  WARNING: Too fast! Might trigger rate limits.")
    else:
        print("⚠️  WARNING: Too slow! Need to optimize further.")

if __name__ == "__main__":
    asyncio.run(benchmark_search())
