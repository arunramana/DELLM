"""Transformer Synthesis: Creates coherent answer from chunk results."""
from utils.llm_client import LLMClient
from typing import Dict, Any, Optional


class TransformerSynthesis:
    """Synthesizes chunk results into coherent final answer."""
    
    def __init__(self, model_path: Optional[str] = None):
        # Use SuperLLM's model or smaller synthesis model
        # Increase context for synthesis to handle longer aggregated results
        self.llm = LLMClient(
            model_path=model_path,
            model_name="synthesis",
            n_ctx=4096,  # Larger context for synthesis
            n_threads=4
        )
    
    def synthesize(self, aggregated: Dict[str, Dict[str, Any]], original_query: str = None) -> str:
        """
        Synthesize chunk results into coherent answer.
        
        Args:
            aggregated: Dict of chunk_id -> result dict
            original_query: Original user query for context
        
        Returns:
            Final synthesized answer
        """
        # Collect unique answers (avoid duplicates from multiple nodes processing same chunk)
        chunk_answers = {}
        for chunk_id, result in sorted(aggregated.items()):  # Sort for consistent ordering
            answer = result.get('answer', '').strip()
            # Skip timeout/error messages
            if answer and not answer.startswith('[Node timeout') and not answer.startswith('[Model'):
                # If multiple nodes processed same chunk, take the best one (longest/non-empty)
                if chunk_id not in chunk_answers or len(answer) > len(chunk_answers[chunk_id]):
                    chunk_answers[chunk_id] = answer
        
        # Build synthesis prompt with context
        if not chunk_answers:
            return "Unable to generate answer from node responses."
        
        # Format results by chunk
        results_text = []
        for chunk_id in sorted(chunk_answers.keys()):
            results_text.append(f"Result {chunk_id}: {chunk_answers[chunk_id]}")
        
        combined = "\n".join(results_text)
        
        query_context = f"\n\nOriginal question: {original_query}" if original_query else ""
        
        prompt = f"""You are synthesizing results from multiple processing steps into one coherent answer.

Results from each step:
{combined}{query_context}

Instructions:
1. Combine all results into a single, comprehensive answer
2. Address all parts of the original question
3. Remove any duplicate information
4. Present the answer clearly and logically
5. If results seem incomplete or contradictory, note this

Provide the final synthesized answer:"""
        
        answer = self.llm.complete(prompt, max_tokens=512)
        
        if not answer or answer.strip() == "":
            # Fallback: join unique answers
            unique_answers = list(chunk_answers.values())
            return ". ".join(unique_answers) + "."
        
        return answer

