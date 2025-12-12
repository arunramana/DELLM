"""
Test script for single LLM direct inference (no nodes, no distributed processing).

This script loads a single model directly on GPU and answers queries for performance comparison.
"""

import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

def load_model_gpu(model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    """
    Load model directly on GPU.
    
    Args:
        model_name: HuggingFace model name
    
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"\n{'='*60}")
    print(f"Loading {model_name} on GPU...")
    print(f"{'='*60}\n")
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("WARNING: No GPU detected! Using CPU (will be slow)")
        device = "cpu"
    else:
        device = "cuda"
        print(f"[+] GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"[+] GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")
    
    start_time = time.time()
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print("Loading model...")
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,  # Use FP16 on GPU for speed
            device_map="auto",  # Auto GPU placement
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float32,
        )
    
    if device == "cpu":
        model = model.to(device)
    
    model.eval()
    
    load_time = time.time() - start_time
    print(f"[+] Model loaded in {load_time:.2f}s")
    print(f"[+] Device: {device}")
    print(f"[+] Dtype: {model.dtype}\n")
    
    return model, tokenizer, device


def answer_query(model, tokenizer, device: str, query: str, max_new_tokens: int = 500, temperature: float = 0.7):
    """
    Answer a query using direct LLM generation.
    
    Args:
        model: The loaded model
        tokenizer: The tokenizer
        device: Device ('cuda' or 'cpu')
        query: The question to answer
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
    
    Returns:
        Tuple of (answer, generation_time)
    """
    print(f"\n{'='*60}")
    print(f"QUERY: {query}")
    print(f"{'='*60}\n")
    
    # Format prompt (TinyLlama chat format)
    prompt = f"<|user|>\n{query}\n<|assistant|>\n"
    
    print("Tokenizing input...")
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    
    print(f"Input tokens: {input_ids.shape[1]}")
    print(f"Generating answer (max {max_new_tokens} tokens)...\n")
    
    start_time = time.time()
    
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=50,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    generation_time = time.time() - start_time
    
    # Extract only the generated part (remove prompt)
    generated_tokens = generated_ids[0][input_ids.shape[1]:]
    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    print(f"{'='*60}")
    print(f"ANSWER: {answer}")
    print(f"{'='*60}")
    print(f"Generation time: {generation_time:.2f}s")
    print(f"Tokens generated: {len(generated_tokens)}")
    print(f"Tokens/second: {len(generated_tokens) / generation_time:.2f}")
    print()
    
    return answer, generation_time


def interactive_mode(model, tokenizer, device, max_tokens=500):
    """Interactive question-answering mode."""
    print(f"\n{'='*60}")
    print("INTERACTIVE MODE - Single LLM Direct Inference")
    print(f"{'='*60}")
    print(f"Max tokens per answer: {max_tokens}")
    print("Enter your queries (or 'quit' to exit)")
    print()
    
    total_queries = 0
    total_time = 0.0
    
    while True:
        try:
            query = input("Your query: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            answer, gen_time = answer_query(model, tokenizer, device, query, max_new_tokens=max_tokens)
            total_queries += 1
            total_time += gen_time
            
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
            break
        except Exception as e:
            print(f"\nError: {e}\n")
    
    if total_queries > 0:
        print(f"\n{'='*60}")
        print("SESSION SUMMARY")
        print(f"{'='*60}")
        print(f"Total queries: {total_queries}")
        print(f"Total generation time: {total_time:.2f}s")
        print(f"Average time per query: {total_time / total_queries:.2f}s")
        print()


def benchmark_mode(model, tokenizer, device, max_tokens=500):
    """Benchmark with predefined queries."""
    print(f"\n{'='*60}")
    print("BENCHMARK MODE - Predefined Queries")
    print(f"{'='*60}")
    print(f"Max tokens per answer: {max_tokens}\n")
    
    queries = [
        "what's 10% of 1000?",
        "what's the tallest mountain?",
        "what's 10% of 1000 and what's the tallest mountain?",
        "who is the president of the United States?",
        "calculate 25 * 4",
    ]
    
    results = []
    
    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}/{len(queries)}")
        answer, gen_time = answer_query(model, tokenizer, device, query, max_new_tokens=max_tokens)
        results.append({
            'query': query,
            'answer': answer,
            'time': gen_time
        })
    
    # Summary
    print(f"\n{'='*60}")
    print("BENCHMARK RESULTS")
    print(f"{'='*60}\n")
    
    total_time = sum(r['time'] for r in results)
    
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['query']}")
        print(f"   Answer: {result['answer'][:100]}{'...' if len(result['answer']) > 100 else ''}")
        print(f"   Time: {result['time']:.2f}s\n")
    
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per query: {total_time / len(results):.2f}s")
    print()


def main():
    """Main entry point."""
    print("\n" + "="*60)
    print("SINGLE LLM DIRECT INFERENCE TEST")
    print("="*60)
    print("\nThis script loads a single model directly (no distributed nodes)")
    print("and answers queries for performance comparison.\n")
    
    # Parse arguments
    mode = "interactive"
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    max_tokens = 500  # Default: 500 tokens (~375 words)
    
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg in ['benchmark', 'bench', 'b']:
            mode = "benchmark"
        elif arg in ['interactive', 'i']:
            mode = "interactive"
        elif arg in ['--max-tokens', '-m']:
            if i + 1 < len(sys.argv):
                max_tokens = int(sys.argv[i + 1])
                i += 1
            else:
                print("Error: --max-tokens requires a value")
                return
        elif not arg.startswith('-'):
            model_name = arg
        i += 1
    
    # Load model
    try:
        model, tokenizer, device = load_model_gpu(model_name)
    except Exception as e:
        print(f"\n[!] Error loading model: {e}")
        print("\nMake sure you have:")
        print("  1. transformers installed: pip install transformers")
        print("  2. torch with CUDA support: pip install torch --index-url https://download.pytorch.org/whl/cu118")
        print("  3. Enough GPU memory (~2GB for TinyLlama)")
        return
    
    # Run selected mode
    try:
        if mode == "benchmark":
            benchmark_mode(model, tokenizer, device, max_tokens)
        else:
            interactive_mode(model, tokenizer, device, max_tokens)
    finally:
        # Cleanup
        print("\nCleaning up...")
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Done\n")


if __name__ == "__main__":
    main()

