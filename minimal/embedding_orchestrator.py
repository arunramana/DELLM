"""Embedding-based Orchestrator: Coordinates embedding flow."""
import asyncio
import torch
import re
from typing import Dict, Any, Optional
from minimal.decomposer import QueryDecomposer
from minimal.router import ClusterRouter
from minimal.embedding_assembly import EmbeddingAssembly
from minimal.training import RLTrainer
from utils.embedding_service import EmbeddingService
from utils.decoder_service import DecoderService
from utils.embedding_assembler import EmbeddingAssembler
from utils.web_search import WebSearchService


class EmbeddingOrchestrator:
    """Orchestrates embedding-based query processing."""
    
    def __init__(self, nodes: Dict[str, Any], embedding_model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                 decoder_model_name: Optional[str] = None, device: str = "cpu", enable_web_search: bool = True):
        """
        Initialize embedding-based orchestrator.
        
        Args:
            nodes: Dict of node_id -> TransformerNode
            embedding_model_name: Model for generating embeddings
            decoder_model_name: Model for decoding (if None, uses embedding_model_name)
            device: Device to run on ('cpu' or 'cuda')
            enable_web_search: Whether to enable web search for factual queries
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
        
        # Web search service for factual queries
        self.web_search = WebSearchService() if enable_web_search else None
    
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
        
        # Step 5: Decode each chunk separately to get answers in order
        print("[Orchestrator] Step 5: Decoding chunks to text (in order)...")
        decode_start = time.time()
        
        # Sort chunks by their position to maintain order
        sorted_chunks = sorted(chunks, key=lambda c: c['chunk_id'])
        chunk_answers = []
        
        for chunk in sorted_chunks:
            chunk_id = chunk['chunk_id']
            chunk_text = chunk['text']
            
            if chunk_id in aggregated:
                # Get processed embeddings for this chunk
                processed_embeddings = aggregated[chunk_id].get('embeddings')
                if processed_embeddings is not None:
                    print(f"[Orchestrator] Decoding chunk {chunk_id}: {chunk_text[:50]}...")
                    
                    # Check if this chunk needs web search (factual queries)
                    search_context = ""
                    if self.web_search and self._needs_web_search(chunk):
                        print(f"[Orchestrator] Performing web search for chunk {chunk_id}...")
                        search_context = self.web_search.get_search_context(chunk_text, max_results=2)
                        if search_context:
                            print(f"[Orchestrator] Found search results for {chunk_id}")
                            # Enhance chunk text with search context
                            enhanced_query = f"{chunk_text}\n\nContext from web search:\n{search_context}"
                        else:
                            enhanced_query = chunk_text
                    else:
                        enhanced_query = chunk_text
                    
                    # Decode this chunk's embeddings using the chunk text (with optional search context)
                    chunk_answer = self.decoder_service.decode(
                        processed_embeddings, 
                        query=enhanced_query,
                        max_length=128  # Shorter for individual chunks
                    )
                    chunk_answers.append(chunk_answer)
                    print(f"[Orchestrator] Chunk {chunk_id} answer: {chunk_answer[:80]}...")
                else:
                    print(f"[Orchestrator] WARNING: No embeddings for chunk {chunk_id}")
                    chunk_answers.append("")
            else:
                print(f"[Orchestrator] WARNING: Chunk {chunk_id} not in aggregated results")
                chunk_answers.append("")
        
        # Clean and format answers
        cleaned_answers = []
        for i, chunk in enumerate(sorted_chunks):
            if i < len(chunk_answers) and chunk_answers[i]:
                cleaned = self._clean_answer(chunk_answers[i], chunk)
                cleaned_answers.append(cleaned)
        
        # Combine answers in order with better formatting
        if len(cleaned_answers) == 1:
            final_answer = cleaned_answers[0]
        elif len(cleaned_answers) > 1:
            final_answer = " and ".join(cleaned_answers)
        else:
            final_answer = "Unable to generate answer."
        
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
        
        # Calculate total embedding shape from aggregated chunks
        total_embedding_shape = None
        if aggregated:
            # Sum up the sequence lengths from all chunks
            total_seq_len = sum(
                result.get('embeddings', torch.tensor([])).shape[0] 
                for result in aggregated.values() 
                if result.get('embeddings') is not None
            )
            if total_seq_len > 0:
                # Get hidden dim from first chunk
                first_emb = next((r.get('embeddings') for r in aggregated.values() if r.get('embeddings') is not None), None)
                if first_emb is not None:
                    hidden_dim = first_emb.shape[1]
                    total_embedding_shape = [total_seq_len, hidden_dim]
        
        return {
            'answer': final_answer,
            'chunks': chunks,
            'assignments': {cid: [n.node_id for n in nodes] for cid, nodes in assignments.items()},
            'aggregated': aggregated,
            'fitness_updates': fitness_updates,
            'embedding_shape': total_embedding_shape
        }
    
    def _clean_answer(self, answer: str, chunk: Dict[str, Any]) -> str:
        """
        Clean and format answer to be concise and relevant.
        
        Args:
            answer: Raw answer from decoder
            chunk: Chunk metadata with operation type
        
        Returns:
            Cleaned answer
        """
        if not answer:
            return ""
        
        # Remove leading/trailing whitespace
        answer = answer.strip()
        
        # Remove any lines with clearly wrong patterns like "100 = 1,000"
        lines = answer.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            # Skip lines that match wrong patterns like "100 = 1,000" (number equals different number)
            wrong_pattern = re.match(r'^(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)$', line)
            if wrong_pattern and wrong_pattern.group(1) != wrong_pattern.group(2):
                # Skip this line - it's a wrong pattern
                continue
            if line:
                cleaned_lines.append(line)
        answer = ' '.join(cleaned_lines) if cleaned_lines else answer
        
        # For math operations, format nicely and verify calculation
        if chunk.get('operation') == 'MATH_OP' or any(kw in chunk.get('text', '').lower() for kw in ['%', 'percent', 'calculate', 'what is', 'what\'s']):
            # Get the chunk text (e.g., "what's 10% of 1000")
            chunk_text = chunk.get('text', '').strip()
            # Extract the math expression from chunk text
            math_expr = re.sub(r'^(what\'?s?|what is|calculate)\s+', '', chunk_text, flags=re.IGNORECASE)
            math_expr = math_expr.strip()
            
            # Try to calculate the correct answer for percentage questions
            calculated_result = None
            percent_match = re.search(r'(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)', chunk_text, re.IGNORECASE)
            if percent_match:
                percent = float(percent_match.group(1))
                base = float(percent_match.group(2))
                calculated_result = (percent / 100.0) * base
                # Round to reasonable precision
                if calculated_result == int(calculated_result):
                    calculated_result = int(calculated_result)
                else:
                    calculated_result = round(calculated_result, 2)
            
            # Extract the number from the answer
            numbers = re.findall(r'\b\d+(?:\.\d+)?\b', answer)
            result_number = None
            
            if numbers:
                # Use the first number found (usually the answer)
                result_number = numbers[0]
            else:
                # If no number found, try to extract after "="
                match = re.search(r'=\s*(\d+(?:\.\d+)?)', answer)
                if match:
                    result_number = match.group(1)
            
            # If we calculated the correct answer and it differs from what the model said, use our calculation
            if calculated_result is not None:
                try:
                    model_answer = float(result_number) if result_number else None
                    # If model answer is wrong (more than 1% difference), use calculated answer
                    if model_answer is None or abs(model_answer - calculated_result) > max(1, calculated_result * 0.01):
                        result_number = str(calculated_result)
                        print(f"[Orchestrator] Corrected math answer: {model_answer} -> {calculated_result}")
                except (ValueError, TypeError):
                    # If we can't parse model answer, use calculated
                    result_number = str(calculated_result)
            
            if result_number:
                return f"{math_expr} = {result_number}"
            else:
                # Fallback: return the original answer if we can't extract/calculate
                return answer
        
        # For other answers, remove verbose prefixes
        # Remove common verbose patterns
        verbose_patterns = [
            r'^Answer:\s*',
            r'^The answer is\s*',
            r'^To answer your question[^:]*:\s*',
            r'^Based on[^:]*:\s*',
            r'^According to[^:]*:\s*',
        ]
        for pattern in verbose_patterns:
            answer = re.sub(pattern, '', answer, flags=re.IGNORECASE)
        
        # Remove redundant repetition (e.g., "10% of 1,000 is: 10% of 1,000 = 100")
        # Take the last meaningful line if multiple lines
        if '\n' in answer:
            lines = [l.strip() for l in answer.split('\n') if l.strip()]
            if lines:
                # Prefer lines that look like complete answers
                for line in reversed(lines):
                    if len(line) > 10 and not line.startswith(('Answer:', 'The answer')):
                        answer = line
                        break
                else:
                    answer = lines[-1]
        
        # Extract first sentence if answer is too long
        if len(answer) > 200:
            sentences = answer.split('.')
            if sentences:
                answer = sentences[0].strip() + '.'
        
        return answer.strip()
    
    def _needs_web_search(self, chunk: Dict[str, Any]) -> bool:
        """
        Determine if a chunk needs web search (factual queries).
        
        Args:
            chunk: Chunk metadata
        
        Returns:
            True if web search should be performed
        """
        chunk_text = chunk.get('text', '').lower()
        operation = chunk.get('operation', '').upper()
        
        # Factual queries that benefit from web search
        factual_keywords = [
            'tallest', 'highest', 'largest', 'smallest', 'longest', 'shortest',
            'first', 'last', 'oldest', 'newest', 'famous', 'known for',
            'located in', 'capital of', 'president of', 'population of',
            'when did', 'who is', 'what is', 'where is'
        ]
        
        # Don't search for math operations
        if operation == 'MATH_OP':
            return False
        
        # Check if chunk contains factual keywords
        return any(keyword in chunk_text for keyword in factual_keywords)
    
    def close(self):
        """Close all services and nodes."""
        if self.embedding_service:
            self.embedding_service.close()
        if self.decoder_service:
            self.decoder_service.close()
        for node in self.nodes.values():
            if hasattr(node, 'close'):
                node.close()

