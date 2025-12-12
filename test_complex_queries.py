"""Test script for complex queries with 5-node sequential GPU setup."""
import requests
import json
import time

BASE_URL = "http://localhost:8000"

# Complex test queries that benefit from 5-node consensus
TEST_QUERIES = [
    # Ambiguous queries
    {
        "query": "What is the capital of America?",
        "description": "Ambiguous (USA vs continents)"
    },
    {
        "query": "Who invented the telephone?",
        "description": "Controversial (Bell vs. Meucci)"
    },
    {
        "query": "How do you spell color?",
        "description": "Regional (color vs colour)"
    },
    
    # Multi-step reasoning
    {
        "query": "If I have 3 apples and buy 2 more, then give away half, how many do I have?",
        "description": "Multi-step math"
    },
    {
        "query": "What's 15% of 200, plus 10?",
        "description": "Percentage + addition"
    },
    
    # Factual with nuance
    {
        "query": "What is machine learning?",
        "description": "Multiple valid definitions"
    },
    {
        "query": "When did Python programming language release?",
        "description": "Different version dates"
    },
    
    # Complex explanations
    {
        "query": "Explain quantum computing in simple terms",
        "description": "Complex concept simplification"
    },
    {
        "query": "What are the pros and cons of neural networks?",
        "description": "Multi-aspect analysis"
    },
]


def test_query(query_data, show_details=False):
    """Test a single query and display results."""
    query = query_data["query"]
    description = query_data["description"]
    
    print(f"\n{'='*70}")
    print(f"Query: {query}")
    print(f"Type: {description}")
    print(f"{'='*70}")
    
    try:
        start_time = time.time()
        
        response = requests.post(
            f"{BASE_URL}/query",
            json={"query": query},
            timeout=60
        )
        
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"✅ Answer: {result.get('answer', 'N/A')}")
            print(f"⏱️  Time: {elapsed:.2f}s")
            
            # Show consensus and node info
            if show_details:
                print(f"\n📊 Details:")
                aggregated = result.get('aggregated', {})
                for chunk_id, chunk_data in aggregated.items():
                    consensus = chunk_data.get('consensus', 0.0)
                    node_count = chunk_data.get('node_count', 0)
                    print(f"  Chunk {chunk_id}:")
                    print(f"    - Nodes: {node_count}")
                    print(f"    - Consensus: {consensus:.2f}")
                    
                    # Show all node responses if available
                    all_responses = chunk_data.get('all_responses', [])
                    if all_responses:
                        print(f"    - Node fitness scores:")
                        for resp in all_responses:
                            node_id = resp.get('node_id', 'unknown')
                            fitness = resp.get('fitness', 0.0)
                            confidence = resp.get('confidence', 0.0)
                            weight = fitness * confidence
                            print(f"      • {node_id}: fitness={fitness:.3f}, conf={confidence:.3f}, weight={weight:.3f}")
            
            return True
            
        else:
            print(f"❌ Error: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"⏱️  Timeout after 60 seconds")
        return False
    except requests.exceptions.ConnectionError:
        print(f"❌ Connection error - is the server running?")
        print(f"   Start with: python minimal/main.py")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Run all test queries."""
    print("="*70)
    print("🚀 Testing 5-Node Sequential GPU Setup")
    print("="*70)
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/", timeout=2)
        print(f"✅ Server is running!")
    except:
        print(f"❌ Server is not running!")
        print(f"   Start with: python minimal/main.py")
        print(f"   Then run this test again.")
        return
    
    # Ask for detail level
    print(f"\nShow detailed node information? (y/n): ", end="")
    show_details = input().strip().lower() == 'y'
    
    # Run tests
    success_count = 0
    total_count = len(TEST_QUERIES)
    
    for i, query_data in enumerate(TEST_QUERIES, 1):
        print(f"\n[Test {i}/{total_count}]")
        if test_query(query_data, show_details):
            success_count += 1
        
        # Small delay between queries to let GPU memory settle
        if i < total_count:
            time.sleep(0.5)
    
    # Summary
    print(f"\n{'='*70}")
    print(f"📊 Test Summary")
    print(f"{'='*70}")
    print(f"Success: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
    print(f"Failed:  {total_count - success_count}/{total_count}")
    
    if success_count == total_count:
        print(f"\n🎉 All tests passed! 5-node sequential GPU is working!")
    elif success_count > 0:
        print(f"\n⚠️  Some tests passed, but check for OOM errors above.")
    else:
        print(f"\n❌ All tests failed - check server logs for errors.")


if __name__ == "__main__":
    main()

