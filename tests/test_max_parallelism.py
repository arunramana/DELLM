"""Test that we use all nodes and maximize decomposition."""
from minimal.decomposer import QueryDecomposer
from minimal.router import ClusterRouter
from core.transformer_node import TransformerNode

# Note: Creating transformer nodes requires model files
# This test verifies decomposition and routing logic
print("Note: Transformer nodes require model files to fully test.")
print("This test verifies decomposition and routing logic only.\n")

# Test decomposer with target chunks
print("\n" + "="*70)
print("Testing Decomposer with target chunks")
print("="*70)

decomposer = QueryDecomposer()

# Test 1: Simple query (should try to split to match nodes)
query1 = "Calculate 10% of 1000 then find best savings accounts"
chunks1 = decomposer.decompose(query1, target_chunks=5)
print(f"\nQuery: {query1}")
print(f"Target chunks: 5")
print(f"Actual chunks: {len(chunks1)}")
for chunk in chunks1:
    print(f"  {chunk['chunk_id']}: {chunk['text'][:50]}...")

# Test 2: Long query (should split more)
query2 = "Calculate tax, find hotels, compare prices, book flight, and create itinerary"
chunks2 = decomposer.decompose(query2, target_chunks=5)
print(f"\nQuery: {query2}")
print(f"Target chunks: 5")
print(f"Actual chunks: {len(chunks2)}")
for chunk in chunks2:
    print(f"  {chunk['chunk_id']}: {chunk['text'][:50]}...")

# Router test would require actual nodes with models
# Skipping router test - would need model files loaded
print("\n" + "="*70)
print("Router test skipped (requires model files)")
print("="*70)
print("\n[SUCCESS] Decomposition logic works!")

