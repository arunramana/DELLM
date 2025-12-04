"""Node Block: Dual-model voting execution unit."""
import time
from typing import Dict, Any
from utils.llm_client import LLMClient


class Node:
    """Node executes queries using dual lightweight LLM voting."""
    
    def __init__(self, node_id: str, model_a_path: str = None, model_b_path: str = None,
                 model_a_name: str = "phi-3-mini", model_b_name: str = "qwen2.5-1.5b"):
        self.node_id = node_id
        
        # Use model paths if provided, otherwise use names (for mock mode)
        # Note: n_threads will be reduced to 1 for thread safety in LLMClient
        self.model_a = LLMClient(
            model_path=model_a_path,
            model_name=model_a_name,
            n_ctx=1024,  # Smaller context for lightweight models
            n_threads=1   # Reduced to 1 for thread safety (LLMClient will enforce this)
        )
        self.model_b = LLMClient(
            model_path=model_b_path,
            model_name=model_b_name,
            n_ctx=1024,
            n_threads=1  # Reduced to 1 for thread safety
        )
        self.fitness = 0.7  # Initial fitness
    
    def execute(self, query: str) -> Dict[str, Any]:
        """Execute query using dual-model voting."""
        start_time = time.time()
        
        # Get answers from both models
        answer_a = self.model_a.complete(query, max_tokens=256)  # Shorter for nodes
        answer_b = self.model_b.complete(query, max_tokens=256)
        
        # Vote on best answer
        final_answer, confidence = self.vote(answer_a, answer_b, query)
        
        latency = time.time() - start_time
        
        return {
            "answer": final_answer,
            "confidence": confidence,
            "latency": latency,
            "node_id": self.node_id
        }
    
    def vote(self, answer_a: str, answer_b: str, query: str) -> tuple:
        """
        Vote on best answer from two models.
        Returns (final_answer, confidence)
        """
        # If answers identical → high confidence
        if answer_a == answer_b:
            return answer_a, 0.95
        
        # Check similarity
        similarity = self.calculate_similarity(answer_a, answer_b)
        
        if similarity > 0.8:
            # Pick more detailed answer
            final = answer_a if len(answer_a) > len(answer_b) else answer_b
            return final, 0.85
        else:
            # Different answers - for prototype, pick first one with lower confidence
            return answer_a, 0.70
    
    def calculate_similarity(self, a: str, b: str) -> float:
        """Simple word overlap similarity metric."""
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        
        if not words_a and not words_b:
            return 1.0
        
        overlap = len(words_a & words_b)
        union = len(words_a | words_b)
        
        return overlap / union if union > 0 else 0.0
    
    def update_fitness(self, fitness: float):
        """Update node fitness score."""
        self.fitness = max(0.0, min(1.0, fitness))
    
    def close(self):
        """Close all models in this node."""
        if self.model_a:
            self.model_a.close()
        if self.model_b:
            self.model_b.close()

