"""Embedding Service: Converts text to embeddings for nodes."""
import torch
from typing import Optional, List
from transformers import AutoModel, AutoTokenizer
import threading


class EmbeddingService:
    """Service to generate embeddings from text using transformer models."""
    
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0", device: str = "cpu"):
        """
        Initialize embedding service.
        
        Args:
            model_name: HuggingFace model name or path
            device: Device to run on ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.device = device
        self.tokenizer = None
        self.model = None
        self.lock = threading.Lock()
        self._load_model()
    
    def _load_model(self):
        """Load model and tokenizer."""
        try:
            print(f"Loading embedding model: {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            print(f"Embedding model loaded successfully on {self.device}")
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            raise
    
    def encode(self, text: str, max_length: int = 512) -> torch.Tensor:
        """
        Convert text to embeddings.
        
        Args:
            text: Input text
            max_length: Maximum sequence length
        
        Returns:
            Embeddings tensor of shape [seq_len, hidden_dim]
        """
        with self.lock:
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get embeddings (from embedding layer, before transformer blocks)
            # Handle different model architectures:
            # - Llama: model.model.embed_tokens
            # - GPT-2/BERT: model.embeddings.word_embeddings or model.embeddings
            # - Others: model.embeddings
            with torch.no_grad():
                input_ids = inputs['input_ids']
                
                # Try Llama architecture first (model.model.embed_tokens)
                if hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
                    embeddings = self.model.model.embed_tokens(input_ids)
                # Try direct embed_tokens
                elif hasattr(self.model, 'embed_tokens'):
                    embeddings = self.model.embed_tokens(input_ids)
                # Try embeddings.word_embeddings (BERT-style)
                elif hasattr(self.model, 'embeddings') and hasattr(self.model.embeddings, 'word_embeddings'):
                    embeddings = self.model.embeddings.word_embeddings(input_ids)
                # Try direct embeddings
                elif hasattr(self.model, 'embeddings'):
                    embeddings = self.model.embeddings(input_ids)
                else:
                    # Fallback: use model's forward pass and take first layer output
                    # This is less efficient but works for any model
                    outputs = self.model(**inputs, output_hidden_states=True)
                    embeddings = outputs.hidden_states[0]  # First layer (embeddings)
                
                # Remove batch dimension: [batch, seq_len, hidden] -> [seq_len, hidden]
                embeddings = embeddings.squeeze(0)
            
            return embeddings
    
    def close(self):
        """Cleanup model."""
        if self.model:
            del self.model
            self.model = None
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

