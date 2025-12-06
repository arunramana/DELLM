"""Test decomposer with various query types."""
from minimal.decomposer import QueryDecomposer

decomposer = QueryDecomposer()

test_cases = [
    "Calculate 10% of 1000 then find best savings accounts",
    "Calculate 2+2 and calculate 3+3",
    "What is 2+2?",
    "Find hotels, compare prices, then book",
    "Calculate tax and calculate tip",
    "What is the capital of France?"
]

print("Testing Query Decomposer\n" + "="*70)

for query in test_cases:
    chunks = decomposer.decompose(query)
    print(f"\nQuery: {query}")
    print(f"Chunks: {len(chunks)}")
    for chunk in chunks:
        print(f"  {chunk['chunk_id']}: {chunk['operation']} - {chunk['text']}")

print("\n" + "="*70)
print("Decomposer test complete!")

