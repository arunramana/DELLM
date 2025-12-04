"""Test that we use all nodes and maximize decomposition."""
from minimal.decomposer import QueryDecomposer
from minimal.router import ClusterRouter
from core.node import Node

# Create test nodes
print("Creating 5 test nodes...")
nodes = {
    f"node-{i}": Node(f"node-{i}") for i in range(1, 6)
}

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

# Test router
print("\n" + "="*70)
print("Testing Router - All Nodes Distribution")
print("="*70)

router = ClusterRouter(nodes)
assignments = router.route(chunks1, use_all_nodes=True)

print(f"\nChunks: {len(chunks1)}")
print(f"Nodes available: {len(nodes)}")
print(f"\nAssignments:")
for chunk_id, node_list in assignments.items():
    node_ids = [n.node_id for n in node_list]
    print(f"  {chunk_id} -> {node_ids} ({len(node_ids)} node(s))")

# Check node usage
all_used_nodes = set()
for node_list in assignments.values():
    all_used_nodes.update(n.node_id for n in node_list)

print(f"\nNode usage:")
print(f"  Total nodes: {len(nodes)}")
print(f"  Nodes used: {len(all_used_nodes)}")
print(f"  Nodes: {sorted(all_used_nodes)}")

if len(all_used_nodes) == len(nodes):
    print("\n[SUCCESS] All nodes are being used!")
else:
    print(f"\n[WARNING] Only {len(all_used_nodes)}/{len(nodes)} nodes used")

