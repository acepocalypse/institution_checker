
import asyncio
from institution_checker.main import process_name_search, should_skip_llm

# VIPs (Must NOT skip)
vips = [
    "Akira Suzuki", 
    "Ei-ichi Negishi", 
    "Herbert C. Brown", 
    "John B. Fenn", 
    "Vernon L. Smith"
]

# Random Nobels (Should SKIP mostly)
# "John Steinbeck", "John Hume", "John O'Keefe" from user report
others = [
    "John Steinbeck", 
    "John Hume", 
    "John O'Keefe",
    "John R. Hicks",
    "John Pople",
    "Albert Einstein", # Unlikely Purdue
    "Marie Curie"
]

async def test_skip_logic():
    print(f"{'Name':<20} | {'Score':<5} | {'Purdue?':<7} | {'Explicit?':<9} | {'Action':<10} | {'Reason'}")
    print("-" * 100)
    
    all_names = vips + others
    
    for name in all_names:
        # Run search
        _, results = await process_name_search(name, use_enhanced_search=True, dataset_profile="low_connection")
        
        # Check skip logic
        should_skip, reason = should_skip_llm(results, dataset_profile="low_connection")
        
        # Get metrics for display
        max_score = 0
        purdue_hits = 0
        explicit = False
        
        if results:
            for r in results:
                signals = r.get('signals', {})
                max_score = max(max_score, signals.get('relevance_score', 0))
                if signals.get('has_explicit_connection'):
                    explicit = True
                if "purdue" in r.get('title', '').lower() or "purdue" in r.get('snippet', '').lower():
                    purdue_hits += 1
        
        action = "SKIP" if should_skip else "QUEUE"
        print(f"{name:<20} | {max_score:<5} | {purdue_hits:<7} | {str(explicit):<9} | {action:<10} | {reason}")

if __name__ == "__main__":
    asyncio.run(test_skip_logic())
