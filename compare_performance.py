"""
Performance comparison script: Single LLM vs Distributed DELLM System

This script runs the same query through both systems and compares:
- Response time
- Answer quality
- Resource usage
"""

import time
import requests
import subprocess
import sys
import torch
from test_single_llm import load_model_gpu, answer_query


def test_dellm_system(query: str, host: str = "http://localhost", port: int = 5003):
    """
    Test query using the distributed DELLM system.
    
    Args:
        query: The query to test
        host: API host
        port: API port
    
    Returns:
        Tuple of (answer, response_time, error_message)
    """
    url = f"{host}:{port}/query"
    
    print(f"\n{'='*60}")
    print("TESTING: DISTRIBUTED DELLM SYSTEM")
    print(f"{'='*60}")
    print(f"Query: {query}")
    print(f"URL: {url}\n")
    
    try:
        start_time = time.time()
        response = requests.post(
            url,
            json={"query": query},
            timeout=300
        )
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            answer = data.get('answer', 'No answer returned')
            
            print(f"Success")
            print(f"Answer: {answer}")
            print(f"Response time: {response_time:.2f}s")
            
            # Show additional info if available
            if 'consensus' in data:
                print(f"Consensus: {data['consensus']:.2f}")
            if 'node_count' in data:
                print(f"Nodes used: {data['node_count']}")
            
            return answer, response_time, None
        else:
            error = f"HTTP {response.status_code}: {response.text}"
            print(f"Error: {error}")
            return None, None, error
            
    except requests.exceptions.ConnectionError:
        error = "Cannot connect to DELLM server. Is it running? (python minimal/main.py)"
        print(f"❌ Error: {error}")
        return None, None, error
    except Exception as e:
        error = str(e)
        print(f"❌ Error: {error}")
        return None, None, error


def test_single_llm(model, tokenizer, device, query: str):
    """
    Test query using single LLM.
    
    Args:
        model: The loaded model
        tokenizer: The tokenizer
        device: Device
        query: The query to test
    
    Returns:
        Tuple of (answer, response_time)
    """
    print(f"\n{'='*60}")
    print("TESTING: SINGLE LLM (Direct Inference)")
    print(f"{'='*60}")
    print(f"Query: {query}\n")
    
    answer, gen_time = answer_query(model, tokenizer, device, query)
    return answer, gen_time


def compare_results(query, dellm_result, llm_result):
    """
    Compare and display results side by side.
    
    Args:
        query: The original query
        dellm_result: Tuple of (answer, time, error) from DELLM
        llm_result: Tuple of (answer, time) from single LLM
    """
    print(f"\n{'='*70}")
    print("COMPARISON RESULTS")
    print(f"{'='*70}")
    print(f"Query: {query}\n")
    
    dellm_answer, dellm_time, dellm_error = dellm_result
    llm_answer, llm_time = llm_result
    
    # Display answers
    print("┌" + "─"*68 + "┐")
    print("│ " + "DISTRIBUTED DELLM SYSTEM".ljust(66) + " │")
    print("├" + "─"*68 + "┤")
    if dellm_error:
        print(f"│ Error: {dellm_error[:60].ljust(60)} │")
    else:
        answer_lines = [dellm_answer[i:i+62] for i in range(0, len(dellm_answer), 62)]
        for line in answer_lines[:3]:  # Show first 3 lines
            print(f"│ {line.ljust(66)} │")
        if len(answer_lines) > 3:
            print(f"│ {'...'.ljust(66)} │")
        print(f"│ Time: {dellm_time:.2f}s".ljust(67) + " │")
    print("└" + "─"*68 + "┘\n")
    
    print("┌" + "─"*68 + "┐")
    print("│ " + "SINGLE LLM (Direct)".ljust(66) + " │")
    print("├" + "─"*68 + "┤")
    answer_lines = [llm_answer[i:i+62] for i in range(0, len(llm_answer), 62)]
    for line in answer_lines[:3]:  # Show first 3 lines
        print(f"│ {line.ljust(66)} │")
    if len(answer_lines) > 3:
        print(f"│ {'...'.ljust(66)} │")
    print(f"│ Time: {llm_time:.2f}s".ljust(67) + " │")
    print("└" + "─"*68 + "┘\n")
    
    # Performance comparison
    if dellm_time:
        speedup = dellm_time / llm_time
        if speedup > 1:
            print(f">> Single LLM is {speedup:.2f}x FASTER")
        else:
            print(f">> DELLM System is {1/speedup:.2f}x FASTER")
        
        print(f"\nTime difference: {abs(dellm_time - llm_time):.2f}s")
    
    print()


def main():
    """Main entry point."""
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON: Single LLM vs Distributed DELLM")
    print("="*70)
    print("\nThis script compares the same query across both systems.\n")
    
    # Get query from command line or use default
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "what's 10% of 1000 and what's the tallest mountain?"
    
    print(f"Query to test: {query}\n")
    
    # Load single LLM
    print("Loading single LLM for comparison...")
    try:
        model, tokenizer, device = load_model_gpu()
    except Exception as e:
        print(f"\nError loading model: {e}")
        print("\nCannot proceed without loading the model.")
        return
    
    print("\nStarting comparison tests...\n")
    print("="*70)
    
    # Test single LLM first
    llm_result = test_single_llm(model, tokenizer, device, query)
    
    # Test DELLM system
    dellm_result = test_dellm_system(query)
    
    # Compare results
    compare_results(query, dellm_result, llm_result)
    
    # Cleanup
    print("Cleaning up...")
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Done\n")


if __name__ == "__main__":
    main()

