"""Minimal client for testing."""
import requests
import json


def query_server(query: str, base_url: str = "http://localhost:8000", timeout: int = 300):
    """Query the minimal DELLM (Distributed Evolutionary LLM) server."""
    print(f"Sending query to {base_url}...")
    print(f"Query: {query}")
    print(f"Timeout: {timeout}s (this may take a while with local models)...")
    
    try:
        response = requests.post(
            f"{base_url}/query",
            json={"query": query},
            timeout=timeout
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        print(f"\nError: Request timed out after {timeout} seconds.")
        print("The server may be processing (local models can be slow) or may be stuck.")
        print("Try:")
        print("  1. Check if server is running: curl http://localhost:8000/health")
        print("  2. Check server logs for errors")
        print("  3. Increase timeout or use mock mode for testing")
        return None
    except requests.exceptions.ConnectionError:
        print(f"\nError: Could not connect to {base_url}")
        print("Make sure the server is running:")
        print("  python minimal/main.py")
        return None
    except Exception as e:
        print(f"\nError: {e}")
        return None


def display_result(result: dict):
    """Display result with trace."""
    print("="*70)
    print("QUERY RESULT")
    print("="*70)
    
    print(f"\nAnswer: {result.get('answer', 'N/A')}")
    
    print(f"\nDecomposition:")
    for chunk in result.get('chunks', []):
        print(f"  {chunk['chunk_id']}: {chunk['operation']} - {chunk['text']}")
    
    print(f"\nRouting:")
    for chunk_id, node_ids in result.get('assignments', {}).items():
        print(f"  {chunk_id} -> {node_ids}")
    
    print(f"\nAggregated Results:")
    for chunk_id, agg in result.get('aggregated', {}).items():
        print(f"  {chunk_id}: {agg.get('answer', 'N/A')[:60]}...")
        print(f"    Consensus: {agg.get('consensus', 0.0):.2%}")
    
    print(f"\nFitness Updates:")
    for node_id, fitness in result.get('fitness_updates', {}).items():
        print(f"  {node_id}: {fitness:.3f}")
    
    print("="*70)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Enter query: ")
    
    result = query_server(query)
    if result:
        display_result(result)

