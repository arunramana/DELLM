"""Test minimal implementation."""
from minimal.decomposer import QueryDecomposer
from minimal.router import ClusterRouter
from core.node import Node

# Test decomposer
print("Testing Decomposer...")
decomposer = QueryDecomposer()
chunks = decomposer.decompose("Calculate 10% of 1000 then find best savings accounts")
print(f"  Decomposed into {len(chunks)} chunks:")
for chunk in chunks:
    print(f"    {chunk['chunk_id']}: {chunk['operation']} - {chunk['text']}")

# Test router
print("\nTesting Router...")
nodes = {
    "node-1": Node("node-1"),
    "node-2": Node("node-2"),
    "node-3": Node("node-3")
}
router = ClusterRouter(nodes)
assignments = router.route(chunks, redundancy=2)
print(f"  Routing assignments:")
for chunk_id, node_list in assignments.items():
    node_ids = [n.node_id for n in node_list]
    print(f"    {chunk_id} -> {node_ids}")

print("\n[SUCCESS] Minimal components work!")

