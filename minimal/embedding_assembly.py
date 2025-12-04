"""Embedding Assembly: Collects embeddings from transformer nodes."""
import asyncio
import torch
from typing import List, Dict, Any
from core.transformer_node import TransformerNode
from utils.embedding_assembler import EmbeddingAssembler


class EmbeddingAssembly:
    """Assembles embeddings from transformer nodes using weighted voting."""
    
    def __init__(self):
        """Initialize embedding assembly."""
        self.assembler = EmbeddingAssembler()
    
    async def collect_embeddings(self, assignments: Dict[str, List[TransformerNode]], 
                                 chunk_embeddings: Dict[str, torch.Tensor],
                                 progressive_callback=None) -> Dict[str, Dict[str, Any]]:
        """
        Collect processed embeddings from nodes in parallel.
        
        Args:
            assignments: chunk_id -> list of nodes
            chunk_embeddings: chunk_id -> input embeddings tensor
            progressive_callback: Optional callback(chunk_id, result) called when each chunk completes
        
        Returns:
            Dict mapping chunk_id to aggregated result with processed embeddings
        """
        aggregated = {}
        
        # Process all chunks in parallel
        task_to_chunk = {}
        for chunk_id, nodes in assignments.items():
            if chunk_id in chunk_embeddings and nodes:
                embeddings = chunk_embeddings[chunk_id]
                task = self._process_chunk(chunk_id, nodes, embeddings)
                task_to_chunk[task] = chunk_id
        
        # Use as_completed to get results as they finish (progressive)
        for coro in asyncio.as_completed(task_to_chunk.keys()):
            chunk_id, result = await coro
            aggregated[chunk_id] = result
            
            # Call progressive callback if provided
            if progressive_callback:
                progressive_callback(chunk_id, result)
        
        return aggregated
    
    async def _process_chunk(self, chunk_id: str, nodes: List[TransformerNode], 
                            embeddings: torch.Tensor) -> tuple:
        """Process a single chunk with multiple nodes."""
        print(f"\n  [Assembly] Processing chunk {chunk_id}: embeddings shape {embeddings.shape}")
        print(f"  [Assembly] Sending to {len(nodes)} node(s): {[n.node_id for n in nodes]}")
        
        # Execute all nodes in parallel
        tasks = [self._node_process(node, embeddings) for node in nodes]
        responses = await asyncio.gather(*tasks)
        
        # Log all responses
        print(f"\n  [Assembly] Chunk {chunk_id} - All node responses:")
        for resp in responses:
            node_id = resp.get('node_id', 'unknown')
            output_shape = resp.get('output_shape', [])
            latency = resp.get('latency', 0)
            print(f"    {node_id}: output shape {output_shape} (latency: {latency:.2f}s)")
        
        # Weighted voting: fitness × confidence
        weights = []
        for resp in responses:
            weight = resp['fitness'] * resp['confidence']
            weights.append(weight)
        
        # Select winner (highest weight)
        winner_idx = weights.index(max(weights))
        winner = responses[winner_idx]
        
        # Calculate consensus (based on embedding similarity)
        embeddings_list = [r['embeddings'] for r in responses]
        consensus = self._calculate_consensus(embeddings_list)
        
        print(f"  [Assembly] Chunk {chunk_id} - Winner: {winner.get('node_id')} (consensus: {consensus*100:.1f}%)")
        
        return (chunk_id, {
            'embeddings': winner['embeddings'],
            'confidence': winner['confidence'],
            'consensus': consensus,
            'node_count': len(nodes),
            'all_responses': responses
        })
    
    def _calculate_consensus(self, embeddings_list: List[torch.Tensor]) -> float:
        """
        Calculate consensus based on embedding similarity.
        
        Args:
            embeddings_list: List of embedding tensors from different nodes
        
        Returns:
            Consensus score (0.0 to 1.0)
        """
        if len(embeddings_list) <= 1:
            return 1.0
        
        # Normalize embeddings
        normalized = [torch.nn.functional.normalize(emb, p=2, dim=1) for emb in embeddings_list]
        
        # Calculate pairwise cosine similarities
        similarities = []
        for i in range(len(normalized)):
            for j in range(i + 1, len(normalized)):
                # Compute cosine similarity
                sim = torch.mean(torch.sum(normalized[i] * normalized[j], dim=1))
                similarities.append(sim.item())
        
        # Average similarity as consensus
        consensus = sum(similarities) / len(similarities) if similarities else 0.0
        return max(0.0, min(1.0, consensus))  # Clamp to [0, 1]
    
    async def _node_process(self, node: TransformerNode, embeddings: torch.Tensor) -> Dict[str, Any]:
        """Process embeddings through node (async wrapper)."""
        import time
        start = time.time()
        print(f"  [Assembly] Processing {node.node_id}...")
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, node.process_embeddings, embeddings),
                timeout=120.0  # 2 minute timeout per node
            )
            elapsed = time.time() - start
            print(f"  [Assembly] {node.node_id} completed in {elapsed:.1f}s")
            result['fitness'] = node.fitness  # Include current fitness
            result['node_id'] = node.node_id  # Ensure node_id is included
            return result
        except asyncio.TimeoutError:
            print(f"  [Assembly] WARNING: {node.node_id} timed out after 120s")
            # Return dummy embeddings on timeout
            dummy_embeddings = torch.zeros_like(embeddings)
            return {
                'embeddings': dummy_embeddings,
                'confidence': 0.0,
                'latency': 120.0,
                'node_id': node.node_id,
                'fitness': node.fitness,
                'error': 'timeout',
                'output_shape': list(embeddings.shape)
            }

