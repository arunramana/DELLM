"""Transformer Node: Processes embeddings through transformer blocks."""
import torch
import time
from typing import Dict, Any, Optional
from transformers import AutoModel
import threading
from utils.config_loader import config


class TransformerNode:
    """Node that processes embeddings through transformer blocks only."""
    
    def __init__(self, node_id: str, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
                 device: str = "cpu", num_layers: Optional[int] = None):
        """
        Initialize transformer node.
        
        Args:
            node_id: Unique node identifier
            model_name: HuggingFace model name or path
            device: Device to run on ('cpu' or 'cuda')
            num_layers: Number of transformer layers to use (None = all layers)
        """
        self.node_id = node_id
        self.model_name = model_name
        self.device = device
        self.num_layers = num_layers
        self.model = None
        self.lock = threading.Lock()
        self.fitness = config.get('defaults', 'initial_fitness', default=0.7)
        self._load_model()
    
    def _load_model(self):
        """Load model and extract transformer blocks."""
        try:
            print(f"Loading transformer node {self.node_id} model: {self.model_name}...")
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            # Get transformer layers (blocks) - handle different model architectures
            # For Llama models (TinyLlama): model.model.layers
            # For GPT-2: model.transformer.h
            # For others: model.layers
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                # Llama architecture (TinyLlama)
                self.transformer_layers = self.model.model.layers
            elif hasattr(self.model, 'layers'):
                # Direct layers
                self.transformer_layers = self.model.layers
            elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
                # GPT-2 architecture
                self.transformer_layers = self.model.transformer.h
            else:
                raise ValueError(f"Could not find transformer layers in model {self.model_name}. "
                               f"Model attributes: {dir(self.model)}")
            
            # Limit layers if specified
            if self.num_layers:
                self.transformer_layers = self.transformer_layers[:self.num_layers]
            
            print(f"Transformer node {self.node_id} loaded: {len(self.transformer_layers)} layers")
        except Exception as e:
            print(f"Error loading transformer node {self.node_id}: {e}")
            raise
    
    def process_embeddings(self, embeddings: torch.Tensor) -> Dict[str, Any]:
        """
        Process embeddings through transformer blocks.
        
        Args:
            embeddings: Input embeddings tensor of shape [seq_len, hidden_dim]
        
        Returns:
            Dict with processed embeddings and metadata
        """
        start_time = time.time()
        
        with self.lock:
            # Add batch dimension: [seq_len, hidden] -> [1, seq_len, hidden]
            hidden_states = embeddings.unsqueeze(0).to(self.device)
            
            # Process through transformer layers
            with torch.no_grad():
                for layer in self.transformer_layers:
                    hidden_states = layer(hidden_states)[0]  # [0] gets hidden states, not tuple
            
            # Remove batch dimension: [1, seq_len, hidden] -> [seq_len, hidden]
            processed_embeddings = hidden_states.squeeze(0)
        
        latency = time.time() - start_time
        
        # Log processing
        print(f"  [Node {self.node_id}] Processed embeddings: {embeddings.shape} -> {processed_embeddings.shape}")
        print(f"  [Node {self.node_id}] Latency: {latency:.2f}s")
        
        # Default confidence from config
        confidence = config.get('defaults', 'default_confidence', default=0.85)
        
        return {
            "embeddings": processed_embeddings,
            "confidence": confidence,
            "latency": latency,
            "node_id": self.node_id,
            "input_shape": list(embeddings.shape),
            "output_shape": list(processed_embeddings.shape)
        }
    
    def update_fitness(self, fitness: float):
        """Update node fitness score."""
        self.fitness = max(0.0, min(1.0, fitness))
    
    def close(self):
        """Close model in this node."""
        if self.model:
            del self.model
            self.model = None
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

