"""Main Orchestrator: Coordinates the entire query flow."""
import asyncio
from typing import Dict, Any
from minimal.decomposer import QueryDecomposer
from minimal.router import ClusterRouter
from minimal.assembly import StreamingAssembly
from minimal.synthesis import TransformerSynthesis
from minimal.training import RLTrainer


class Orchestrator:
    """Orchestrates query decomposition → routing → execution → synthesis → training."""
    
    def __init__(self, nodes: Dict[str, Any], synthesis_model_path: str = None):
        self.decomposer = QueryDecomposer()
        self.router = ClusterRouter(nodes)
        self.assembly = StreamingAssembly()
        self.synthesis = TransformerSynthesis(synthesis_model_path)
        self.trainer = RLTrainer()
        self.nodes = nodes
    
    async def process_query(self, query: str, correct_answer: str = None) -> Dict[str, Any]:
        """
        Process query through full pipeline.
        
        Returns:
            Dict with answer and trace information
        """
        import time
        start_time = time.time()
        
        print(f"[Orchestrator] Processing query: {query[:60]}...")
        
        # Step 1: Decompose - try to match number of available nodes
        num_nodes = len(self.nodes)
        print(f"[Orchestrator] Step 1: Decomposing query (target: {num_nodes} chunks for {num_nodes} nodes)...")
        chunks = self.decomposer.decompose(query, target_chunks=num_nodes)
        print(f"[Orchestrator] Decomposed into {len(chunks)} chunk(s) (have {num_nodes} nodes available)")
        
        # Step 2: Route - use all available nodes
        print("[Orchestrator] Step 2: Routing to nodes (using all available compute power)...")
        assignments = self.router.route(chunks, use_all_nodes=True)
        total_nodes = sum(len(nodes) for nodes in assignments.values())
        unique_nodes = set()
        for nodes_list in assignments.values():
            unique_nodes.update(n.node_id for n in nodes_list)
        print(f"[Orchestrator] Routed to {total_nodes} node assignment(s) across {len(unique_nodes)} unique node(s)")
        
        # Step 3 & 4: Process and Assemble (parallel)
        print("[Orchestrator] Step 3-4: Executing nodes (this may take a while)...")
        node_start = time.time()
        aggregated = await self.assembly.collect_responses(assignments, chunks)
        node_time = time.time() - node_start
        print(f"[Orchestrator] Node execution completed in {node_time:.1f}s")
        
        # Update router latencies from actual executions
        for chunk_id, agg in aggregated.items():
            for resp in agg.get('all_responses', []):
                node_id = resp.get('node_id')
                latency = resp.get('latency', 1.0)
                if node_id:
                    self.router.update_latency(node_id, latency)
        
        # Step 5: Synthesize
        print("[Orchestrator] Step 5: Synthesizing answer...")
        synth_start = time.time()
        final_answer = self.synthesis.synthesize(aggregated, original_query=query)
        synth_time = time.time() - synth_start
        print(f"[Orchestrator] Synthesis completed in {synth_time:.1f}s")
        
        # Step 6: Train (update fitness)
        print("[Orchestrator] Step 6: Updating fitness...")
        nodes_used = []
        for nodes_list in assignments.values():
            nodes_used.extend(nodes_list)
        # Remove duplicates
        nodes_used = list({n.node_id: n for n in nodes_used}.values())
        fitness_updates = self.trainer.update_fitness(nodes_used, aggregated, correct_answer)
        
        total_time = time.time() - start_time
        print(f"[Orchestrator] Query processing completed in {total_time:.1f}s")
        
        return {
            'answer': final_answer,
            'chunks': chunks,
            'assignments': {cid: [n.node_id for n in nodes] for cid, nodes in assignments.items()},
            'aggregated': aggregated,
            'fitness_updates': fitness_updates
        }
    
    def process_query_sync(self, query: str, correct_answer: str = None) -> Dict[str, Any]:
        """Synchronous wrapper for process_query."""
        return asyncio.run(self.process_query(query, correct_answer))
    
    def close(self):
        """Close all models in nodes."""
        for node in self.nodes.values():
            if hasattr(node, 'close'):
                node.close()
        if hasattr(self.synthesis, 'llm') and hasattr(self.synthesis.llm, 'close'):
            self.synthesis.llm.close()

