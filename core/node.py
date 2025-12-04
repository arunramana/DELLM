"""Node Block: Single-model execution unit."""
import time
from typing import Dict, Any
from utils.llm_client import LLMClient


class Node:
    """Node executes queries using a single lightweight LLM."""
    
    def __init__(self, node_id: str, model_path: str = None,
                 model_name: str = "tinyllama-1.1b"):
        self.node_id = node_id
        
        # Use model path if provided, otherwise use name (for mock mode)
        # Note: n_threads will be reduced to 1 for thread safety in LLMClient
        self.model = LLMClient(
            model_path=model_path,
            model_name=model_name,
            n_ctx=1024,  # Smaller context for lightweight models
            n_threads=1   # Reduced to 1 for thread safety (LLMClient will enforce this)
        )
        self.fitness = 0.7  # Initial fitness
    
    def execute(self, query: str) -> Dict[str, Any]:
        """Execute query using single model."""
        start_time = time.time()
        
        # Get answer from model
        answer = self.model.complete(query, max_tokens=256)  # Shorter for nodes
        
        latency = time.time() - start_time
        
        # Default confidence (can be improved with self-assessment later)
        confidence = 0.85
        
        return {
            "answer": answer,
            "confidence": confidence,
            "latency": latency,
            "node_id": self.node_id
        }
    
    def update_fitness(self, fitness: float):
        """Update node fitness score."""
        self.fitness = max(0.0, min(1.0, fitness))
    
    def close(self):
        """Close model in this node."""
        if self.model:
            self.model.close()

