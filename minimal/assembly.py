"""Streaming Assembly: Collect node responses with weighted voting."""
import asyncio
from typing import List, Dict, Any
from core.node import Node


class StreamingAssembly:
    """Assembles node responses using weighted voting."""
    
    async def collect_responses(self, assignments: Dict[str, List[Node]], chunks: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Collect responses from nodes in parallel.
        
        Args:
            assignments: chunk_id -> list of nodes
            chunks: List of chunk dicts
        
        Returns:
            Dict mapping chunk_id to aggregated result
        """
        aggregated = {}
        
        # Process all chunks in parallel
        tasks = []
        for chunk in chunks:
            chunk_id = chunk['chunk_id']
            nodes = assignments.get(chunk_id, [])
            text = chunk['text']
            
            if nodes:
                task = self._process_chunk(chunk_id, nodes, text)
                tasks.append(task)
        
        # Wait for all chunks
        results = await asyncio.gather(*tasks)
        
        for chunk_id, result in results:
            aggregated[chunk_id] = result
        
        return aggregated
    
    async def _process_chunk(self, chunk_id: str, nodes: List[Node], text: str) -> tuple:
        """Process a single chunk with multiple nodes."""
        # Execute all nodes in parallel
        tasks = [self._node_execute(node, text) for node in nodes]
        responses = await asyncio.gather(*tasks)
        
        # Weighted voting: fitness × confidence
        weights = []
        for resp in responses:
            weight = resp['fitness'] * resp['confidence']
            weights.append(weight)
        
        # Select winner (highest weight)
        winner_idx = weights.index(max(weights))
        winner = responses[winner_idx]
        
        # Calculate consensus
        answers = [r['answer'] for r in responses]
        consensus = answers.count(winner['answer']) / len(answers) if answers else 0.0
        
        return (chunk_id, {
            'answer': winner['answer'],
            'confidence': winner['confidence'],
            'consensus': consensus,
            'node_count': len(nodes),
            'all_responses': responses
        })
    
    async def _node_execute(self, node: Node, text: str) -> Dict[str, Any]:
        """Execute node (async wrapper)."""
        import time
        start = time.time()
        print(f"  [Assembly] Executing {node.node_id}...")
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, node.execute, text),
                timeout=120.0  # 2 minute timeout per node
            )
            elapsed = time.time() - start
            print(f"  [Assembly] {node.node_id} completed in {elapsed:.1f}s")
            result['fitness'] = node.fitness  # Include current fitness
            result['node_id'] = node.node_id  # Ensure node_id is included
            return result
        except asyncio.TimeoutError:
            print(f"  [Assembly] WARNING: {node.node_id} timed out after 120s")
            return {
                'answer': '[Node timeout - model may be stuck]',
                'confidence': 0.0,
                'latency': 120.0,
                'node_id': node.node_id,
                'fitness': node.fitness,
                'error': 'timeout'
            }

