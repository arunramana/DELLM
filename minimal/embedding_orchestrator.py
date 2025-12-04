"""Embedding-based Orchestrator: Coordinates embedding flow."""
import asyncio
import torch
from typing import Dict, Any, Optional
from minimal.decomposer import QueryDecomposer
from minimal.router import ClusterRouter
from minimal.embedding_assembly import EmbeddingAssembly
from minimal.training import RLTrainer
from utils.embedding_service import EmbeddingService
from utils.decoder_service import DecoderService
from utils.embedding_assembler import EmbeddingAssembler


class EmbeddingOrchestrator:
    """Orchestrates embedding-based query processing."""
    
    def __init__(self, nodes: Dict[str, Any], embedding_model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                 decoder_model_name: Optional[str] = None, device: str = "cpu"):
        """
        Initialize embedding-based orchestrator.
        
        Args:
            nodes: Dict of node_id -> TransformerNode
            embedding_model_name: Model for generating embeddings
            decoder_model_name: Model for decoding (if None, uses embedding_model_name)
            device: Device to run on ('cpu' or 'cuda')
        """
        self.decomposer = QueryDecomposer()
        self.router = ClusterRouter(nodes)
        self.assembly = EmbeddingAssembly()
        self.trainer = RLTrainer()
        self.nodes = nodes
        
        # Initialize embedding and decoder services
        self.embedding_service = EmbeddingService(embedding_model_name, device)
        decoder_model = decoder_model_name or embedding_model_name
        self.decoder_service = DecoderService(decoder_model, device)
        
        # Embedding assembler for progressive assembly
        self.assembler = EmbeddingAssembler()
    
    async def process_query(self, query: str, correct_answer: str = None) -> Dict[str, Any]:
        """
        Process query through embedding-based pipeline.
        
        Returns:
            Dict with answer and trace information
        """
        import time
        start_time = time.time()
        
        print(f"[Orchestrator] Processing query: {query[:60]}...")
        
        # Step 1: Decompose
        num_nodes = len(self.nodes)
        print(f"[Orchestrator] Step 1: Decomposing query (target: {num_nodes} chunks)...")
        chunks = self.decomposer.decompose(query, target_chunks=num_nodes)
        print(f"[Orchestrator] Decomposed into {len(chunks)} chunk(s)")
        
        # Step 2: Generate embeddings for each chunk
        print("[Orchestrator] Step 2: Generating embeddings for chunks...")
        embed_start = time.time()
        chunk_embeddings = {}
        for chunk in chunks:
            chunk_id = chunk['chunk_id']
            chunk_text = chunk['text']
            embeddings = self.embedding_service.encode(chunk_text)
            chunk_embeddings[chunk_id] = embeddings
            print(f"  [Orchestrator] Generated embeddings for {chunk_id}: {embeddings.shape}")
        embed_time = time.time() - embed_start
        print(f"[Orchestrator] Embedding generation completed in {embed_time:.1f}s")
        
        # Step 3: Route
        print("[Orchestrator] Step 3: Routing to nodes...")
        assignments = self.router.route(chunks, use_all_nodes=True)
        total_nodes = sum(len(nodes) for nodes in assignments.values())
        unique_nodes = set()
        for nodes_list in assignments.values():
            unique_nodes.update(n.node_id for n in nodes_list)
        print(f"[Orchestrator] Routed to {total_nodes} node assignment(s) across {len(unique_nodes)} unique node(s)")
        
        # Step 4: Process embeddings through nodes (progressive)
        print("[Orchestrator] Step 4: Processing embeddings through nodes...")
        node_start = time.time()
        
        aggregated = {}
        assembler = EmbeddingAssembler()
        
        def progressive_callback(chunk_id: str, result: Dict[str, Any]):
            """Called when each chunk completes."""
            nonlocal aggregated, assembler
            
            aggregated[chunk_id] = result
            processed_embeddings = result.get('embeddings')
            
            if processed_embeddings is not None:
                # Add to assembler
                assembler.add_chunk(chunk_id, processed_embeddings)
                print(f"[Orchestrator] Added {chunk_id} to assembler ({assembler.get_chunk_count()}/{len(chunks)} chunks)")
        
        # Collect processed embeddings
        aggregated = await self.assembly.collect_embeddings(
            assignments, 
            chunk_embeddings, 
            progressive_callback
        )
        node_time = time.time() - node_start
        print(f"[Orchestrator] Node processing completed in {node_time:.1f}s")
        
        # Step 5: Assemble embeddings
        print("[Orchestrator] Step 5: Assembling embeddings...")
        assemble_start = time.time()
        final_embeddings = assembler.get_assembled()
        assemble_time = time.time() - assemble_start
        print(f"[Orchestrator] Embedding assembly completed in {assemble_time:.1f}s: {final_embeddings.shape}")
        
        # Step 6: Decode to text
        print("[Orchestrator] Step 6: Decoding embeddings to text...")
        decode_start = time.time()
        final_answer = self.decoder_service.decode(final_embeddings)
        decode_time = time.time() - decode_start
        print(f"[Orchestrator] Decoding completed in {decode_time:.1f}s")
        
        # Print summary
        print("\n" + "="*70)
        print("[Orchestrator] DATA SUMMARY:")
        print("="*70)
        print(f"\nOriginal Query: {query}")
        print(f"\nDecomposition ({len(chunks)} chunks):")
        for chunk in chunks:
            print(f"  {chunk['chunk_id']}: {chunk['operation']} - {chunk['text']}")
        print(f"\nFinal Answer:")
        print(f"  {final_answer}")
        print("\n" + "="*70)
        
        # Step 7: Training (update fitness)
        print("[Orchestrator] Step 7: Updating fitness...")
        nodes_used = []
        for nodes_list in assignments.values():
            nodes_used.extend(nodes_list)
        nodes_used = list({n.node_id: n for n in nodes_used}.values())
        
        # Convert aggregated to format expected by trainer (with 'answer' field)
        aggregated_for_training = {}
        for chunk_id, result in aggregated.items():
            # For training, we need text answers - use decoded embeddings or consensus
            aggregated_for_training[chunk_id] = {
                'answer': final_answer,  # Use final answer for now
                'confidence': result.get('confidence', 0.85),
                'consensus': result.get('consensus', 0.0),
                'all_responses': result.get('all_responses', [])
            }
        
        fitness_updates = self.trainer.update_fitness(nodes_used, aggregated_for_training, correct_answer)
        
        print(f"Fitness Updates:")
        for node_id, fitness in sorted(fitness_updates.items()):
            print(f"  {node_id}: {fitness:.3f}")
        
        total_time = time.time() - start_time
        print(f"\n[Orchestrator] Query processing completed in {total_time:.1f}s")
        
        return {
            'answer': final_answer,
            'chunks': chunks,
            'assignments': {cid: [n.node_id for n in nodes] for cid, nodes in assignments.items()},
            'aggregated': aggregated,
            'fitness_updates': fitness_updates,
            'embedding_shape': list(final_embeddings.shape)
        }
    
    def close(self):
        """Close all services and nodes."""
        if self.embedding_service:
            self.embedding_service.close()
        if self.decoder_service:
            self.decoder_service.close()
        for node in self.nodes.values():
            if hasattr(node, 'close'):
                node.close()

