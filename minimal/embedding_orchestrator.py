"""Embedding-based Orchestrator: Coordinates embedding flow."""
import asyncio
import torch
import re
from typing import Dict, Any, Optional
from minimal.decomposer import QueryDecomposer
from minimal.router import ClusterRouter
from minimal.embedding_assembly import EmbeddingAssembly
from minimal.training import RLTrainer
from minimal.query_classifier import OperationType
from utils.embedding_service import EmbeddingService
from utils.decoder_service import DecoderService
from utils.embedding_assembler import EmbeddingAssembler
from utils.web_search import WebSearchService
from utils.config_loader import config


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
        self.nodes = nodes
        
        # Initialize embedding and decoder services
        # Keep services on CPU to avoid GPU memory conflicts with multiple nodes
        self.embedding_service = EmbeddingService(embedding_model_name, device, force_cpu=True)
        
        # Initialize trainer with embedding service for semantic similarity
        self.trainer = RLTrainer(embedding_service=self.embedding_service)
        decoder_model = decoder_model_name or embedding_model_name
        self.decoder_service = DecoderService(decoder_model, device, force_cpu=True)
        
        # Embedding assembler for progressive assembly
        self.assembler = EmbeddingAssembler()
        
        # Web search service for factual queries (use config if not explicitly set)
        web_search_enabled = enable_web_search if enable_web_search is not None else config.get('web_search', 'enabled', default=True)
        self.web_search = WebSearchService() if web_search_enabled else None
    
    async def process_query(self, query: str, correct_answer: str = None) -> Dict[str, Any]:
        """
        Process query through embedding-based pipeline.
        
        Returns:
            Dict with answer and trace information
        """
        import time
        start_time = time.time()
        
        # Step 1: Decompose
        num_nodes = len(self.nodes)
        chunks = self.decomposer.decompose(query, target_chunks=num_nodes)
        
        # Step 2: Generate embeddings for each chunk
        chunk_embeddings = {}
        for chunk in chunks:
            chunk_id = chunk['chunk_id']
            chunk_text = chunk['text']
            embeddings = self.embedding_service.encode(chunk_text)
            chunk_embeddings[chunk_id] = embeddings
        
        # Step 3: Route
        assignments = self.router.route(chunks, use_all_nodes=True)
        
        # Step 4: Process embeddings through nodes
        aggregated = {}
        assembler = EmbeddingAssembler()
        
        def progressive_callback(chunk_id: str, result: Dict[str, Any]):
            """Called when each chunk completes."""
            nonlocal aggregated, assembler
            aggregated[chunk_id] = result
            processed_embeddings = result.get('embeddings')
            if processed_embeddings is not None:
                assembler.add_chunk(chunk_id, processed_embeddings)
        
        # Collect processed embeddings
        aggregated = await self.assembly.collect_embeddings(
            assignments, 
            chunk_embeddings, 
            progressive_callback
        )
        
        # Step 5: Decode each chunk separately to get answers in order
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
                    # Check if this chunk needs web search (factual queries)
                    search_context = ""
                    if self.web_search and self._needs_web_search(chunk):
                        max_results = config.get('web_search', 'max_results', default=2)
                        search_context = self.web_search.get_search_context(chunk_text, max_results=max_results)
                        if search_context:
                            enhanced_query = f"{chunk_text}\n\nContext from web search:\n{search_context}"
                        else:
                            enhanced_query = chunk_text
                    else:
                        enhanced_query = chunk_text
                    
                    # Decode this chunk's embeddings
                    # Use fast decode only for math/factual queries, slow generation for creative tasks
                    operation_type = chunk.get('operation', 'unknown')
                    use_fast = (operation_type in [OperationType.MATH_OP.value, OperationType.FACTUAL_OP.value])
                    
                    # Use longer max_length for generation tasks
                    max_length = 1000 if operation_type == OperationType.GENERATION_OP.value else 128
                    
                    chunk_answer = self.decoder_service.decode(
                        processed_embeddings, 
                        query=enhanced_query,
                        max_length=max_length,
                        use_fast_decode=use_fast
                    )
                    chunk_answers.append(chunk_answer)
                else:
                    chunk_answers.append("")
            else:
                chunk_answers.append("")
        
        # Clean and format answers, and store them in aggregated results
        cleaned_answers = []
        for i, chunk in enumerate(sorted_chunks):
            chunk_id = chunk['chunk_id']
            if i < len(chunk_answers) and chunk_answers[i]:
                cleaned = self._clean_answer(chunk_answers[i], chunk)
                cleaned_answers.append(cleaned)
                # Store the decoded answer in aggregated for client access
                if chunk_id in aggregated:
                    aggregated[chunk_id]['answer'] = cleaned
        
        # Combine answers in order with better formatting
        if len(cleaned_answers) == 1:
            final_answer = cleaned_answers[0]
        elif len(cleaned_answers) > 1:
            final_answer = " and ".join(cleaned_answers)
        else:
            final_answer = "Unable to generate answer."
        
        # Step 7: Training (update fitness + online training)
        nodes_used = []
        for nodes_list in assignments.values():
            nodes_used.extend(nodes_list)
        nodes_used = list({n.node_id: n for n in nodes_used}.values())
        
        # Update fitness based on final answer quality
        fitness_updates = self.trainer.update_fitness(
            nodes_used, final_answer, correct_answer, assignments, aggregated
        )
        
        # Online training (if enabled and correct_answer provided)
        quality_score = 0.0
        if correct_answer and config.get('training', 'enable_online_training', default=True):
            quality_score = self.trainer.calculate_final_answer_quality(final_answer, correct_answer)
            min_quality = config.get('training', 'min_quality_for_training', default=0.0)
            
            if quality_score >= min_quality:
                self._trigger_online_training(nodes_used, assignments, aggregated, chunk_embeddings, 
                                             quality_score, correct_answer, chunks)
        
        total_time = time.time() - start_time
        
        # Return with quality score for tracking
        return {
            'answer': final_answer,
            'chunks': chunks,
            'assignments': {cid: [n.node_id for n in nodes] for cid, nodes in assignments.items()},
            'aggregated': aggregated,
            'fitness_updates': fitness_updates,
            'quality_score': quality_score,
            'processing_time': total_time
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
        
        # For generation tasks, preserve newlines; for others join with spaces
        operation_type = chunk.get('operation', 'unknown')
        if operation_type == 'GENERATION_OP':
            answer = '\n'.join(cleaned_lines) if cleaned_lines else answer
        else:
            answer = ' '.join(cleaned_lines) if cleaned_lines else answer
        
        # Get config values
        math_keywords = config.get('math_detection', 'keywords', default=[])
        enable_calculation = config.get('math_detection', 'enable_calculation', default=True)
        calculation_tolerance = config.get('math_detection', 'calculation_tolerance', default=0.01)
        max_answer_length = config.get('answer_processing', 'max_answer_length', default=200)
        min_line_length = config.get('answer_processing', 'min_line_length', default=10)
        
        # For math operations, format nicely and verify calculation
        # Use the operation type from the chunk (already classified by hybrid classifier)
        is_math_op = chunk.get('operation') == 'MATH_OP'
        
        if is_math_op:
            # Get the chunk text (e.g., "what's 10% of 1000")
            chunk_text = chunk.get('text', '').strip()
            # Extract the math expression from chunk text
            math_expr = re.sub(r'^(what\'?s?|what is|calculate)\s+', '', chunk_text, flags=re.IGNORECASE)
            math_expr = math_expr.strip()
            
            # Try to calculate the correct answer for percentage questions (if enabled)
            calculated_result = None
            if enable_calculation:
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
            if enable_calculation and calculated_result is not None:
                try:
                    model_answer = float(result_number) if result_number else None
                    # If model answer is wrong (beyond tolerance), use calculated answer
                    tolerance = max(1, calculated_result * calculation_tolerance)
                    if model_answer is None or abs(model_answer - calculated_result) > tolerance:
                        result_number = str(calculated_result)
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
        
        # For generation tasks, keep the full multi-line answer
        # For other tasks, take the last meaningful line if multiple lines
        if operation_type != 'GENERATION_OP' and '\n' in answer:
            lines = [l.strip() for l in answer.split('\n') if l.strip()]
            if lines:
                # Prefer lines that look like complete answers
                for line in reversed(lines):
                    if len(line) > min_line_length and not line.startswith(('Answer:', 'The answer')):
                        answer = line
                        break
                else:
                    answer = lines[-1]
        
        # Extract first sentence if answer is too long
        if len(answer) > max_answer_length:
            sentences = answer.split('.')
            if sentences:
                answer = sentences[0].strip() + '.'
        
        return answer.strip()
    
    def _trigger_online_training(self, nodes_used, assignments, aggregated, chunk_embeddings,
                                quality_score, correct_answer, chunks):
        """Trigger online training for nodes."""
        learning_rate = config.get('training', 'learning_rate', default=1e-5)
        
        # Generate target embeddings from correct answer
        # For multi-chunk queries, split correct answer
        correct_chunks = correct_answer.split(' and ')
        if len(correct_chunks) < len(chunks):
            # If fewer chunks than queries, use full answer for each
            correct_chunks = [correct_answer] * len(chunks)
        
        # Sort chunks to match order
        sorted_chunks = sorted(chunks, key=lambda c: c['chunk_id'])
        
        trained_count = 0
        for i, chunk in enumerate(sorted_chunks):
            chunk_id = chunk['chunk_id']
            
            if chunk_id not in aggregated:
                continue
            
            result = aggregated[chunk_id]
            processed_emb = result.get('embeddings')
            input_emb = chunk_embeddings.get(chunk_id)
            
            if processed_emb is None or input_emb is None:
                continue
            
            # Get node that processed this chunk (winner from consensus)
            all_responses = result.get('all_responses', [])
            if not all_responses:
                continue
            
            # Find winner node
            winner_node_id = None
            max_weight = 0.0
            for resp in all_responses:
                node_id = resp.get('node_id')
                node = next((n for n in nodes_used if n.node_id == node_id), None)
                if node:
                    weight = node.fitness * resp.get('confidence', 0.85)
                    if weight > max_weight:
                        max_weight = weight
                        winner_node_id = node_id
            
            if not winner_node_id:
                continue
            
            node = next((n for n in nodes_used if n.node_id == winner_node_id), None)
            if not node:
                continue
            
            # Enable training if not already enabled
            if not hasattr(node, 'training_enabled') or not node.training_enabled:
                node.enable_training(learning_rate)
            
            # Generate target embeddings from correct answer chunk
            target_text = correct_chunks[i] if i < len(correct_chunks) else correct_answer
            target_embeddings = self.embedding_service.encode(target_text)
            
            # Process target through transformer to get processed target
            # Use the node's full model forward pass instead of layer-by-layer
            # This ensures proper handling of position embeddings
            with torch.no_grad():
                # Use model_device (where model actually is) not node.device (target device)
                compute_device = node.model_device
                target_hidden = target_embeddings.unsqueeze(0).to(compute_device)
                
                # Convert to same dtype as model (handle FP16 if mixed precision is enabled)
                if node.use_mixed_precision and compute_device == "cuda":
                    target_hidden = target_hidden.half()
                
                # Generate position_ids on the correct device
                seq_len = target_hidden.shape[1]
                position_ids = torch.arange(0, seq_len, dtype=torch.long, device=compute_device).unsqueeze(0)
                
                # Use the model's forward method properly
                if node.is_llama:
                    if hasattr(node.model, 'model'):
                        llama_model = node.model.model
                    else:
                        llama_model = node.model
                    
                    outputs = llama_model(
                        inputs_embeds=target_hidden,
                        position_ids=position_ids,
                        output_hidden_states=True,
                        return_dict=True
                    )
                    processed_target = outputs.last_hidden_state.squeeze(0)
                else:
                    # For non-Llama architectures
                    for layer in node.transformer_layers:
                        try:
                            target_hidden = layer(target_hidden, position_ids=position_ids)[0]
                        except TypeError:
                            target_hidden = layer(target_hidden)[0]
                    processed_target = target_hidden.squeeze(0)
            
            # Train node
            try:
                loss = node.train_step(input_emb, processed_target, quality_weight=quality_score)
                trained_count += 1
            except Exception as e:
                pass  # Silent training errors
    
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
        
        # Get factual keywords from config
        factual_keywords = config.get('web_search', 'factual_keywords', default=[])
        
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

