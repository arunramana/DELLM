"""Cluster Router: Routes chunks to nodes based on latency + fitness."""
import heapq
from typing import List, Dict, Any
from core.node import Node


class ClusterRouter:
    """Routes query chunks to optimal nodes."""
    
    def __init__(self, nodes: Dict[str, Node]):
        self.nodes = nodes
        # Track node latencies (updated from actual executions)
        self.node_latencies = {nid: 1.0 for nid in nodes.keys()}
    
    def update_latency(self, node_id: str, latency: float):
        """Update node latency from actual execution."""
        self.node_latencies[node_id] = latency
    
    def route(self, chunks: List[Dict[str, Any]], use_all_nodes: bool = True) -> Dict[str, List[Node]]:
        """
        Route chunks to nodes using all available compute power.
        
        Args:
            chunks: List of chunk dicts
            use_all_nodes: If True, distribute chunks across all nodes (round-robin)
                          If False, use redundancy (multiple nodes per chunk)
        
        Returns:
            Dict mapping chunk_id to list of nodes
        """
        assignments = {}
        all_nodes = list(self.nodes.values())
        
        if not all_nodes:
            return assignments
        
        if use_all_nodes:
            # Distribute chunks across all nodes
            # If we have more nodes than chunks, assign multiple nodes per chunk (parallel processing)
            # If we have fewer nodes than chunks, assign one node per chunk (round-robin)
            
            if len(all_nodes) >= len(chunks):
                # More nodes than chunks: assign multiple nodes per chunk for parallel processing
                nodes_per_chunk = len(all_nodes) // len(chunks)
                remainder = len(all_nodes) % len(chunks)
                
                node_idx = 0
                for i, chunk in enumerate(chunks):
                    chunk_id = chunk['chunk_id']
                    # Distribute nodes evenly, with remainder going to first chunks
                    num_nodes_for_chunk = nodes_per_chunk + (1 if i < remainder else 0)
                    assigned_nodes = all_nodes[node_idx:node_idx + num_nodes_for_chunk]
                    assignments[chunk_id] = assigned_nodes
                    node_idx += num_nodes_for_chunk
            else:
                # Fewer nodes than chunks: round-robin assignment
                for i, chunk in enumerate(chunks):
                    chunk_id = chunk['chunk_id']
                    node_index = i % len(all_nodes)
                    assigned_node = all_nodes[node_index]
                    assignments[chunk_id] = [assigned_node]
        else:
            # Original behavior: assign multiple nodes per chunk (redundancy)
            redundancy = min(2, len(all_nodes))  # At least 2 if available
            for chunk in chunks:
                chunk_id = chunk['chunk_id']
                # Use all nodes if we have fewer nodes than redundancy requested
                if len(all_nodes) <= redundancy:
                    assignments[chunk_id] = all_nodes
                else:
                    # Select nodes based on latency + fitness
                    scores = []
                    for node_id, node in self.nodes.items():
                        latency = self.node_latencies.get(node_id, 1.0)
                        fitness = node.fitness
                        score = 0.7 * latency + 0.3 * (1 - fitness)
                        scores.append((score, node_id, node))
                    
                    scores.sort(key=lambda x: x[0])
                    selected_nodes = [node for _, _, node in scores[:redundancy]]
                    assignments[chunk_id] = selected_nodes
        
        return assignments

